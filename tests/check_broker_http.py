"""WP7 broker HTTP routes over the integrated app (create_app: KeylessHostGuard, Guard + Authenticator, BrokerDispatch,
the routes), against the WP4 loopback mocks with the netguard installed (zero non-loopback attempts).

- POST /v1/moderations: a cloud openai target passes through; a local model is 503 classifier_unavailable (fail
  closed), an unknown one 404, both with zero upstream hits;
- envelopes per route family (OpenAI, Anthropic, Gemini, ElevenLabs, Ollama) with X-Titanos-Error-Origin;
- provenance headers on cloud and local answers;
- image url re-hosting into an owner-bound /v1/files artifact (the owner and bootstrap read it, another key gets 404)
  and the refusal for the JPEG (xAI-like) mock before any hit;
- POST /v1/messages/count_tokens: Anthropic passthrough, and the llama-cpp local path (R22: a fake native worker answers
  /apply-template then /tokenize), other local models 400;
- /v1beta: the action allowlist (404 for anything else), the model listings, and generateContent bridged to a local
  model while the broker is inactive;
- Responses handle affinity: the owner retrieves, another key gets 404 response_not_found, previous_response_id from
  another key is 404, all with zero extra hits; the handle records the served model (R17);
- ElevenLabs text-to-speech without model_id reaches the unchanged local route, records a Decision and applies the
  key's model and class constraints (R16);
- a committed stream is ended by the request deadline with the protocol's terminal frame (R15; Anthropic's is
  timeout_error);
- the Gemini ?key= refusal keeps its guidance readable (not mangled by redaction) and carries query_credential_refused
  as a google.rpc.ErrorInfo detail;
- ElevenLabs speech-to-text bridged to a local transcription model passes the real ASRRequest validation of
  /v1/audio/transcriptions (one timestamp granularity) and answers the ElevenLabs shape.
"""
from __future__ import annotations

import sys
import time
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import AnthropicMock, OpenAIMock, netguard  # noqa: E402

GPT, CLAUDE = 'gpt-4o-mini@openai', 'claude-sonnet-4-5@anthropic'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
MOCKS: dict = {}
LOCAL = {'chat-local': 'chat', 'voice': 'speech', 'embed-local': 'embeddings', 'asr-local': 'transcription'}


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['anthropic'] = AnthropicMock()
    MOCKS['xai'] = OpenAIMock(variant='xai')
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def document() -> dict:
    m = MOCKS
    return {
        'policy': {'cloud_policy': 'explicit'},
        'upstreams': {
            'openai': W.S.upstream(m['openai'], 'openai', credential='oa',
                                   models=['gpt-4o-mini', 'dall-e-3', 'omni-moderation-latest'],
                                   data_policy={'allow_store': True}),
            'anthropic': W.S.upstream(m['anthropic'], 'anthropic', credential='an', models=['claude-sonnet-4-5']),
            'xai': W.S.upstream(m['xai'], 'xai', credential='xa', models=['grok-2-image']),
        },
    }


class FakeNativeClient:
    """The leased llama-server client: /apply-template then /tokenize (records the sequence)."""

    def __init__(self, seen: list):
        self.seen = seen

    async def post(self, route, json=None):
        import httpx
        self.seen.append((route, json))
        if route == '/apply-template':
            text = ''.join(m.get('content') or '' for m in json['messages'] if isinstance(m.get('content'), str))
            payload = {'prompt': '<|start|>' + text + '<|end|>'}
        else:
            payload = {'tokens': list(range(len(json['content'].split())))}  # one token per word
        return httpx.Response(200, json=payload)


def fake_native(full, name: str) -> list:
    """Register a llama-cpp model (registry and live NativePool) whose lease yields FakeNativeClient (no process)."""
    from titanos.config import ModelSpec
    seen: list = []
    spec = ModelSpec.model_construct(path=str(full.root), task='chat', backend='llama-cpp', defaults={}, voices={},
                                     device='auto')
    full.config.models[name] = spec
    native = full.app.state.native
    native.models[name] = spec

    @asynccontextmanager
    async def lease(model, worker=None, *, require_loaded=False):
        yield SimpleNamespace(client=FakeNativeClient(seen))
    native.lease = lease
    return seen


class Base(unittest.TestCase):
    keyed = True

    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
        self.full = W.FullApp(document(), models=LOCAL, api_key='wp7-bootstrap-key-0001' if self.keyed else None)
        self.c = self.full.http
        if self.keyed:
            self.cloud_id, self.cloud = self.full.mint('inference', 'cloud', 'cloud:store', name='cloud')
            self.other_id, self.other = self.full.mint('inference', 'cloud', name='other')

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def hits(self) -> int:
        return W.S.total_hits(*MOCKS.values())

    def post(self, path, body, token=None, headers=None):
        return self.c.post(path, json=body, headers=headers or W.bearer(token or self.cloud))


class Moderation(Base):
    def test_cloud_passthrough_and_local_fail_closed(self):
        reply = self.post('/v1/moderations', {'model': 'omni-moderation-latest@openai', 'input': 'hello'})
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        self.assertEqual(reply.headers['x-titanos-served-by'], 'cloud:openai')
        self.assertEqual(MOCKS['openai'].hits[-1].surface, 'openai.moderations')
        before = self.hits()
        local = self.post('/v1/moderations', {'model': 'chat-local', 'input': 'hello'})
        self.assertEqual((local.status_code, local.json()['error']['code']), (503, 'classifier_unavailable'))
        self.assertEqual(local.headers.get('x-should-retry'), 'false')
        unknown = self.post('/v1/moderations', {'model': 'nope', 'input': 'hello'})
        self.assertEqual((unknown.status_code, unknown.json()['error']['code']), (404, 'model_not_found'))
        self.assertEqual(self.hits(), before)


class Envelopes(Base):
    def test_error_envelope_per_route_family(self):
        openai = self.post('/v1/chat/completions', {'model': 'nope@openai', 'messages': CHAT})
        self.assertEqual(openai.status_code, 404)
        self.assertEqual(openai.json()['error']['code'], 'model_not_found')
        anthropic = self.post('/v1/messages/count_tokens', {'model': 'nope', 'messages': CHAT})
        self.assertEqual(anthropic.json(), {'type': 'error', 'error': {'type': 'not_found_error',
                                                                       'message': anthropic.json()['error']['message']}})
        gemini = self.post('/v1beta/models/nope:generateContent', {'contents': [{'parts': [{'text': 'hi'}]}]})
        self.assertEqual((gemini.status_code, gemini.json()['error']['status']), (404, 'NOT_FOUND'))
        eleven = self.c.post('/v1/speech-to-text', content=b'not multipart', headers={
            **W.bearer(self.cloud), 'content-type': 'text/plain'})
        self.assertEqual(eleven.status_code, 400)
        self.assertIn('status', eleven.json()['detail'])
        ollama = self.post('/api/chat', {'model': 'nope@openai', 'messages': CHAT, 'stream': False})
        self.assertEqual(ollama.status_code, 404)
        self.assertIsInstance(ollama.json()['error'], str)
        for reply in (openai, anthropic, gemini, eleven, ollama):
            self.assertIn(reply.headers.get('x-titanos-error-origin'), ('titanos', 'policy'))
        self.assertEqual(self.hits(), 0)


class Provenance(Base):
    def test_provenance_headers_on_cloud_and_local_answers(self):
        cloud = self.post('/v1/chat/completions', {'model': GPT, 'messages': CHAT},
                          headers={**W.bearer(self.cloud), 'X-Request-ID': 'req-wp7-0001'})
        self.assertEqual(cloud.status_code, 200, cloud.text[:300])
        h = cloud.headers
        self.assertEqual((h['x-request-id'], h['x-titanos-served-by'], h['x-titanos-requested-model'],
                          h['x-titanos-served-model'], h['x-titanos-fallback-count'], h['x-titanos-translation']),
                         ('req-wp7-0001', 'cloud:openai', GPT, GPT, '0', 'passthrough'))
        self.assertTrue(h['x-titanos-decision-id'])
        local = self.post('/v1/chat/completions', {'model': 'chat-local', 'messages': CHAT})
        self.assertEqual(local.status_code, 200, local.text[:300])
        self.assertEqual((local.headers['x-titanos-served-by'], local.headers['x-titanos-translation']),
                         ('local:engine', 'local'))
        self.assertEqual(local.json()['choices'][0]['message']['content'], 'local hello')


class ImageRehosting(Base):
    def test_url_rehosting_is_owner_bound_and_jpeg_is_refused(self):
        reply = self.post('/v1/images/generations', {'model': 'dall-e-3@openai', 'prompt': 'a cat',
                                                     'response_format': 'url'})
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        url = reply.json()['data'][0]['url']
        # absolute, from the request's base URL, like the local image route and job artifacts (SDKs fetch it as is)
        self.assertRegex(url, r'^http://127\.0\.0\.1:8000/v1/files/[0-9a-f]{32}\.png$')
        own = self.c.get(url, headers=W.bearer(self.cloud))
        self.assertEqual(own.status_code, 200)
        self.assertTrue(own.content.startswith(b'\x89PNG'))
        self.assertEqual(self.c.get(url, headers=W.bearer(self.other)).status_code, 404)
        self.assertEqual(self.c.get(url, headers=W.bearer('wp7-bootstrap-key-0001')).status_code, 200)
        # xAI answers its own image URLs (extras image_url_native): passed through, not re-hosted, not refused
        jpeg = self.post('/v1/images/generations', {'model': 'grok-2-image@xai', 'prompt': 'a cat',
                                                    'response_format': 'url'})
        self.assertEqual(jpeg.status_code, 200, jpeg.text[:300])
        self.assertTrue(jpeg.json()['data'][0]['url'].startswith('https://example.invalid/xai/'))
        self.assertEqual(len(MOCKS['xai'].hits), 1)


class CountTokens(Base):
    def test_anthropic_passthrough_and_llama_cpp_local_count(self):
        reply = self.post('/v1/messages/count_tokens', {'model': CLAUDE, 'messages': CHAT})
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        self.assertIsInstance(reply.json()['input_tokens'], int)
        self.assertEqual(MOCKS['anthropic'].hits[-1].surface, 'anthropic.count_tokens')
        seen = fake_native(self.full, 'llama-local')
        local = self.post('/v1/messages/count_tokens', {'model': 'llama-local', 'system': 'Be brief.',
                                                        'messages': CHAT})
        self.assertEqual(local.status_code, 200, local.text[:300])
        self.assertEqual([route for route, _ in seen], ['/apply-template', '/tokenize'])
        self.assertEqual(seen[0][1]['messages'][0], {'role': 'system', 'content': 'Be brief.'})
        self.assertEqual(seen[1][1]['add_special'], True)
        self.assertEqual(local.json(), {'input_tokens': 3})
        self.assertEqual(local.headers['x-titanos-served-by'], 'local:native')
        other = self.post('/v1/messages/count_tokens', {'model': 'chat-local', 'messages': CHAT})
        self.assertEqual((other.status_code, other.json()['error']['type']), (400, 'invalid_request_error'))

    def test_local_count_while_the_broker_is_inactive(self):
        self.full.close()
        self.full = W.FullApp(None, models=LOCAL)
        self.c = self.full.http
        seen = fake_native(self.full, 'llama-local')
        reply = self.c.post('/v1/messages/count_tokens', json={'model': 'llama-local', 'messages': CHAT})
        self.assertEqual((reply.status_code, reply.json()), (200, {'input_tokens': 2}), reply.text[:300])
        self.assertEqual(len(seen), 2)
        gemini = self.c.post('/v1beta/models/llama-local:countTokens', json={'contents': [{'parts': [{'text': 'hi'}]}]})
        self.assertEqual((gemini.status_code, gemini.json()), (200, {'totalTokens': 1}), gemini.text[:300])


class Gemini(Base):
    keyed = False

    def test_action_allowlist_and_listings(self):
        bad = self.c.post('/v1beta/models/chat-local:frobnicate', json={})
        self.assertEqual((bad.status_code, bad.json()['error']['status']), (404, 'NOT_FOUND'))
        listing = self.c.get('/v1beta/models').json()['models']
        names = {row['name']: row['supportedGenerationMethods'] for row in listing}
        self.assertIn('generateContent', names['models/chat-local'])
        self.assertEqual(names['models/embed-local'], ['embedContent', 'batchEmbedContents'])
        self.assertNotIn('models/voice', names)
        one = self.c.get('/v1beta/models/models/chat-local')
        self.assertEqual((one.status_code, one.json()['baseModelId']), (200, 'chat-local'))
        self.assertEqual(self.c.get('/v1beta/models/nope').status_code, 404)

    def test_generate_content_reaches_a_local_model(self):
        reply = self.c.post('/v1beta/models/chat-local:generateContent',
                            json={'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]})
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        text = reply.json()['candidates'][0]['content']['parts'][0]['text']
        self.assertEqual(text, 'local hello')
        self.assertEqual(self.full.engine.calls[-1][:2], ('chat-local', 'chat'))


class Responses(Base):
    def test_handle_affinity_and_owner_only(self):
        created = self.post('/v1/responses', {'model': GPT, 'input': 'hi', 'store': True})
        self.assertEqual(created.status_code, 200, created.text[:300])
        rid = created.json()['id']
        self.assertEqual(self.full.broker.handles.owner('responses', rid).model, GPT)  # R17
        got = self.c.get(f'/v1/responses/{rid}', headers=W.bearer(self.cloud))
        self.assertEqual((got.status_code, got.json()['id']), (200, rid))
        before = self.hits()
        other = self.c.get(f'/v1/responses/{rid}', headers=W.bearer(self.other))
        self.assertEqual((other.status_code, other.json()['error']['code']), (404, 'response_not_found'))
        chained = self.post('/v1/responses', {'model': GPT, 'input': 'again', 'previous_response_id': rid},
                            token=self.other)
        self.assertEqual((chained.status_code, chained.json()['error']['code']), (404, 'response_not_found'))
        items = self.c.get(f'/v1/responses/{rid}/input_items', headers=W.bearer(self.other))
        self.assertEqual(items.status_code, 404)
        self.assertEqual(self.hits(), before)
        unknown = self.c.get('/v1/responses/resp_unknown', headers=W.bearer(self.cloud))
        self.assertEqual(unknown.status_code, 404)

    def test_narrowed_model_globs_hide_an_owned_handle(self):
        created = self.post('/v1/responses', {'model': GPT, 'input': 'hi', 'store': True})
        rid = created.json()['id']
        record = self.full.broker.keys.get(self.cloud_id)
        from titanos.broker.keys import KeyPatch
        self.full.broker.keys.update(record.id, KeyPatch(constraints={'models': ['claude-*']}), actor='test')
        self.full.broker.keys._refresh(force=True)
        before = self.hits()
        got = self.c.get(f'/v1/responses/{rid}', headers=W.bearer(self.cloud))
        self.assertEqual((got.status_code, got.json()['error']['code']), (404, 'response_not_found'))
        self.assertEqual(self.hits(), before)


class TextToSpeech(Base):
    def test_no_model_id_reaches_the_local_route_with_a_decision(self):
        reply = self.post('/v1/text-to-speech/default?output_format=wav', {'text': 'hi'})
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        self.assertTrue(reply.content.startswith(b'RIFF'))
        self.assertEqual(self.full.engine.calls[-1][:2], ('voice', 'speech'))
        self.assertTrue(reply.headers.get('x-titanos-decision-id'))
        decisions = self.full.broker.decisions.query(limit=5, principal=W.S.BOOTSTRAP)
        self.assertIn('model_absent:local_route', decisions[0]['notes'])
        _, cloud_only = self.full.mint('inference', 'cloud', name='cloudonly',
                                       constraints={'target_classes': ['cloud']})
        denied = self.post('/v1/text-to-speech/default?output_format=wav', {'text': 'hi'}, token=cloud_only)
        self.assertEqual(denied.status_code, 404)
        _, other_model = self.full.mint('inference', name='othermodel', constraints={'models': ['chat-*']})
        denied = self.post('/v1/text-to-speech/default?output_format=wav', {'text': 'hi'}, token=other_model)
        self.assertEqual(denied.status_code, 404)
        stream = self.post('/v1/text-to-speech/default/stream?output_format=wav', {'text': 'hi', 'model_id': 'voice'})
        self.assertEqual(stream.status_code, 200, stream.text[:300])
        self.assertTrue(stream.content.startswith(b'RIFF'))


class Unconfigured(Base):
    keyed = False

    def test_disabled_lane_tts_is_unchanged(self):
        self.full.close()
        self.full = W.FullApp(None, models=LOCAL)
        self.c = self.full.http
        reply = self.c.post('/v1/text-to-speech/default?output_format=wav', json={'text': 'hi'})
        self.assertEqual(reply.status_code, 200)
        self.assertNotIn('x-titanos-decision-id', reply.headers)
        self.assertEqual(self.full.engine.calls[-1][:2], ('voice', 'speech'))
        self.assertFalse((self.full.home / 'broker').exists())  # the disabled lane writes nothing under TITANOS_HOME


class Catalogs(Base):
    def test_live_catalog_refresh_only_for_principals_that_see_the_upstream(self):
        """R21: GET /v1/models refreshes a stale live catalog only for a principal allowed to see that upstream."""
        self.full.close()
        doc = document()
        doc['upstreams']['openai']['catalog'] = 'live'
        self.full = W.FullApp(doc, models=LOCAL, api_key='wp7-bootstrap-key-0001')
        self.c = self.full.http
        _, blind = self.full.mint('inference', 'cloud', 'models:read', name='blind',
                                  constraints={'upstreams': ['anthropic']})
        _, sees = self.full.mint('inference', 'cloud', 'models:read', name='sees')
        catalog_hits = lambda: [h for h in MOCKS['openai'].hits if h.path.endswith('/models')]  # noqa: E731
        self.assertEqual(self.c.get('/v1/models', headers=W.bearer(blind)).status_code, 200)
        self.assertEqual(catalog_hits(), [])
        listed = self.c.get('/v1/models', headers=W.bearer(sees)).json()['data']
        self.assertEqual(len(catalog_hits()), 1)
        ids = {row['id'] for row in listed}
        self.assertIn(GPT, ids)
        self.assertTrue(any(i.endswith('@openai') and i != GPT for i in ids))  # live-only models appear
        self.c.get('/v1/models', headers=W.bearer(sees))
        self.assertEqual(len(catalog_hits()), 1)  # fresh within the TTL
        forced = self.c.get('/api/broker/catalog?refresh=true', headers=W.bearer('wp7-bootstrap-key-0001')).json()
        self.assertEqual(forced['upstream_status']['openai']['source'], 'live')
        self.assertEqual(len(catalog_hits()), 2)


class Reasoning(unittest.TestCase):
    """R19: the advisor's reasoning_mode reaches Anthropic thinking and Gemini thinkingBudget when the request sets none."""

    def test_passthrough_injection(self):
        from titanos.broker.dispatch import inject_reasoning
        notes: list = []
        body = {'model': 'm', 'max_tokens': 8000, 'messages': [{'role': 'user', 'content': 'hi'}]}
        out = inject_reasoning('anthropic.messages', body, 'high', frozenset({'thinking'}), notes)
        self.assertEqual(out['thinking'], {'type': 'enabled', 'budget_tokens': 7999})
        self.assertNotIn('thinking', body)
        self.assertEqual(inject_reasoning('anthropic.messages', {**body, 'temperature': 0.2}, 'low',
                                          frozenset({'thinking'}), [])['temperature'], 0.2)
        self.assertNotIn('thinking', inject_reasoning('anthropic.messages', {**body, 'temperature': 0.2}, 'low',
                                                      frozenset({'thinking'}), []))
        self.assertNotIn('thinking', inject_reasoning('anthropic.messages', {**body, 'max_tokens': 512}, 'low',
                                                      frozenset({'thinking'}), []))
        gem = inject_reasoning('gemini.generateContent', {'contents': []}, 'medium', frozenset({'thinking'}), notes)
        self.assertEqual(gem['generationConfig']['thinkingConfig'], {'thinkingBudget': 4096})
        self.assertEqual(inject_reasoning('openai.chat', {'model': 'm'}, 'low', frozenset({'reasoning'}), notes)
                         ['reasoning_effort'], 'low')
        self.assertEqual(notes, ['injected:thinking', 'injected:thinkingBudget', 'injected:reasoning_effort'])

    def test_translated_requests_carry_the_reasoning(self):
        from titanos.broker import inbound
        from titanos.broker.profiles import PROFILES
        from titanos.broker.translate import anthropic_messages, gemini
        ir = inbound.decode_chat('openai.chat', {'model': 'x', 'max_tokens': 9000,
                                                 'messages': [{'role': 'user', 'content': 'hi'}]},
                                 requested='x', stream=False, notes=[])
        anth = inbound._with_reasoning(ir, 'medium', PROFILES['anthropic'], [])
        body = anthropic_messages.encode_request(anth, PROFILES['anthropic'], default_max_tokens=4096, notes=[])
        self.assertEqual(body['thinking'], {'type': 'enabled', 'budget_tokens': 4096})
        gem = inbound._with_reasoning(ir, 'high', PROFILES['gemini'], [])
        self.assertEqual(gemini.encode_request(gem, PROFILES['gemini'], notes=[])['generationConfig']['thinkingConfig']
                         ['thinkingBudget'], 16384)
        self.assertIs(inbound._with_reasoning(ir, None, PROFILES['anthropic'], []), ir)


class HandleFormat(unittest.TestCase):
    def test_version_one_handles_are_read(self):
        """R17: handles.json gained a version and the served model; the previous flat layout is still read."""
        import json
        import shutil
        import tempfile
        from titanos.broker.handles import HandleStore
        home = Path(tempfile.mkdtemp(prefix='titanos-wp7-handles-'))
        self.addCleanup(shutil.rmtree, home, True)
        path = home / 'handles.json'
        path.write_text(json.dumps({'responses:resp_old': {'upstream_id': 'openai', 'credential_id': 'oa',
                                                            'key_id': 'k1', 'created_at': 1.0, 'expires_at': 9e12}}))
        store = HandleStore(path)
        old = store.owner('responses', 'resp_old')
        self.assertEqual((old.upstream_id, old.key_id, old.model), ('openai', 'k1', None))
        store.bind('responses', 'resp_new', 'openai', 'oa', 'k1', model=GPT)
        document = json.loads(path.read_text())
        self.assertEqual(document['version'], 2)
        self.assertEqual(document['handles']['responses:resp_new']['model'], GPT)
        self.assertIsNone(document['handles']['responses:resp_old']['model'])
        self.assertEqual(HandleStore(path).owner('responses', 'resp_new').model, GPT)
        path.write_text(json.dumps({'version': 9, 'handles': {}}))
        self.assertIsNone(HandleStore(path).owner('responses', 'resp_new'))  # an unknown version binds nothing


class StreamDeadline(Base):
    def test_committed_stream_ends_with_the_deadline_terminal_frame(self):
        """R15: a committed cloud stream outliving X-Titanos-Deadline-Ms ends with the protocol's terminal frame."""
        MOCKS['openai'].frame_delay = 0.4
        try:
            started = time.monotonic()
            reply = self.post('/v1/chat/completions', {'model': GPT, 'messages': CHAT, 'stream': True},
                              headers={**W.bearer(self.cloud), 'X-Titanos-Deadline-Ms': '700'})
        finally:
            MOCKS['openai'].frame_delay = 0
        self.assertEqual(reply.status_code, 200)
        frames = [line for line in reply.text.split('\n\n') if line.strip()]
        self.assertIn('"code": "deadline_exceeded"', frames[-2])
        self.assertEqual(frames[-1].strip(), 'data: [DONE]')
        self.assertLess(time.monotonic() - started, 5)

    def test_anthropic_deadline_frame_is_timeout_error(self):
        """R15 on the Anthropic stream: the terminal error event carries Anthropic's own 504 type, timeout_error, so a
        client can tell an expired deadline from a provider api_error."""
        import json
        MOCKS['anthropic'].frame_delay = 0.4
        try:
            reply = self.c.post('/v1/messages', json={'model': CLAUDE, 'max_tokens': 32, 'messages': CHAT,
                                                      'stream': True},
                                headers={'x-api-key': self.cloud, 'X-Titanos-Deadline-Ms': '700'})
        finally:
            MOCKS['anthropic'].frame_delay = 0
        self.assertEqual(reply.status_code, 200)
        last = [block for block in reply.text.split('\n\n') if block.strip()][-1]
        self.assertTrue(last.startswith('event: error'), last)
        error = json.loads(last.split('data: ', 1)[1])['error']
        self.assertEqual(error['type'], 'timeout_error')


class QueryCredential(Base):
    def test_gemini_query_key_refusal_is_readable_and_typed(self):
        reply = self.c.post('/v1beta/models/chat-local:generateContent?key=' + self.cloud,
                            json={'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                            headers={'x-goog-api-key': self.cloud})
        self.assertEqual(reply.status_code, 401)
        error = reply.json()['error']
        self.assertEqual((error['code'], error['status']), (401, 'UNAUTHENTICATED'))
        self.assertNotIn('<redacted>', error['message'])
        self.assertIn('x-goog-api-key', error['message'])
        self.assertEqual(error['details'][0]['reason'], 'QUERY_CREDENTIAL_REFUSED')
        self.assertEqual(error['details'][0]['metadata'], {'code': 'query_credential_refused'})
        self.assertNotIn(self.cloud, reply.text)


class SpeechToTextBridge(Base):
    def test_elevenlabs_stt_reaches_a_local_transcription_model(self):
        seen = []

        def engine(name, task, data, emit=None, cancel=None, **kw):  # a backend answering verbose_json metadata
            seen.append((name, task, data.get('timestamp_granularities')))
            return {'task': 'transcribe', 'text': 'local transcript', 'duration': 1.0, 'sample_rate': 16000,
                    'decoded_samples': 16000, 'language': 'en', 'language_source': 'backend',
                    'requested_language': None, 'segments': None, 'backend': 'fake', 'timestamps': {}, 'quality': {},
                    'words': [{'id': 0, 'start': 0.0, 'end': 0.4, 'word': 'local'},
                              {'id': 1, 'start': 0.5, 'end': 0.9, 'word': 'transcript'}]}
        self.full.app.state.engine.target.infer = engine
        wav = W.WAV + bytes(3200)
        files = {'file': ('a.wav', wav, 'audio/wav')}
        direct = self.c.post('/v1/audio/transcriptions', headers=W.bearer(self.cloud), files=files,
                             data={'model': 'asr-local', 'response_format': 'verbose_json',
                                   'timestamp_granularities[]': ['word', 'segment']})
        self.assertEqual(direct.status_code, 400, direct.text[:300])  # the local route takes one granularity
        reply = self.c.post('/v1/speech-to-text', headers={'xi-api-key': self.cloud}, files=files,
                            data={'model_id': 'asr-local'})
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        body = reply.json()
        self.assertEqual(body['text'], 'local transcript')
        self.assertEqual([w['text'] for w in body['words'] if w['type'] == 'word'], ['local', 'transcript'])
        self.assertTrue(reply.headers['x-titanos-served-by'].startswith('local'), reply.headers)
        self.assertEqual(seen, [('asr-local', 'transcription', ['word'])])  # only the direct 400 call never ran
        self.assertEqual(self.hits(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
