"""BrokerDispatch (WP6): the ASGI middleware over a fake inner app (tests/broker_wp6_support.FakeLocal) and the WP4
loopback mocks (netguard installed; zero non-loopback attempts).

- local fallback triggers (409 model_not_loaded, 429 rate_limit_error, 429 queue_full, 503 queue_full, 503
  worker_unavailable) advance only before commit; a local 4xx never falls back;
- an Anthropic 200 stream whose first event is an error falls back before commit: the client sees only the second
  target's bytes;
- no keepalive is emitted before commit (a slow first byte many keepalive intervals long, on a scaled clock), and
  keepalives do flow after commit while the upstream is idle;
- a mid-stream upstream disconnect ends with the in-protocol terminal frame and no second hit;
- no fallback for images.edits, store=true or previous_response_id; an upstream 400 is passed through, never retried;
- provenance headers on every path; the inactive fast path touches neither scope nor body, and install_broker creates
  no files and opens no sockets;
- alias/virtual body rewrite on JSON and multipart; a client disconnect closes the upstream;
- the budget is admitted before any hit; PNG url re-hosting; handle-bound routes; Gemini and ElevenLabs STT bridges to
  local models;
- live lane follow-ups (run live-20261002T100017Z-835910c9): a budgeted key's output cap reaches token-generating
  surfaces only (exact upstream bodies for images, speech, transcription, embeddings, moderations and count_tokens; the
  translated path too); a provider 404/410 for a model is 404 model_not_found in every inbound envelope with breaker and
  credential untouched; transcription settles by tokens, provider seconds or the measured upload, and keeps the
  reservation (basis 'reservation') only when nothing can be priced.
"""
from __future__ import annotations

import base64
import json
import shutil
import socket
import tempfile
import threading
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

import broker_wp6_support as S
from broker_mocks import AnthropicMock, OpenAIMock, netguard, wait_until

from titanos.broker.dispatch import FALLBACK_CLOUD, FALLBACK_LOCAL, BrokerDispatch, LocalAttempt
from titanos.broker.facade import install_broker
from titanos.broker.sse import KEEPALIVE

GPT, CLAUDE = 'gpt-4o-mini@openai', 'claude-sonnet-4-5@anthropic'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
CLOUD = S.key('inference', 'cloud', id='kdispatch01')
LOCAL = {'qwen-local': ('chat', 'llama-cpp'), 'llama-b': 'chat', 'whisper-local': 'transcription',
         'img-local': 'image'}
MOCKS: dict = {}


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
        'policy': {'cloud_policy': 'fallback'},
        'upstreams': {
            'openai': S.upstream(m['openai'], 'openai', credential='oa', models=['gpt-4o-mini', 'dall-e-2', 'dall-e-3'],
                                 data_policy={'allow_store': True}),
            'anthropic': S.upstream(m['anthropic'], 'anthropic', credential='an', models=['claude-sonnet-4-5']),
            'xai': S.upstream(m['xai'], 'xai', credential='xa', models=['grok-2-image']),
        },
        'models': {'mini': {'target': GPT}},
        'virtual_models': {
            'local-pair': {'capability': 'chat', 'targets': [{'ref': 'qwen-local'}, {'ref': 'llama-b'}]},
            'ant-then-oa': {'capability': 'chat', 'targets': [{'ref': CLAUDE}, {'ref': GPT}]},
            'oa-then-ant': {'capability': 'chat', 'targets': [{'ref': GPT}, {'ref': CLAUDE}]},
            'oa-then-local': {'capability': 'chat', 'targets': [{'ref': GPT}, {'ref': 'qwen-local'}]},
            'edits-vm': {'capability': 'images', 'max_attempts': 2,
                         'targets': [{'ref': 'dall-e-2@openai'}, {'ref': 'img-local'}]},
            'chat-vm': {'capability': 'chat', 'targets': [{'ref': 'qwen-local'}]},
            'stt-vm': {'capability': 'transcription', 'targets': [{'ref': 'whisper-local'}]},
        },
        'prices': [{'match': GPT, 'input_per_mtok': '1000.00', 'output_per_mtok': '1000.00'}],
    }


class Case(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
        self.lab = S.Lab(document(), local=LOCAL)
        self.broker = self.lab.broker
        self.local = S.FakeLocal()
        self.app = BrokerDispatch(self.local, self.broker)

    def tearDown(self):
        for item in MOCKS.values():
            item.gate.set()
        self.lab.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def hits(self) -> int:
        return S.total_hits(*MOCKS.values())

    def chat(self, model, *, principal=CLOUD, stream=False, path='/v1/chat/completions', headers=None, **extra):
        body = {'model': model, 'messages': CHAT, 'stream': stream}
        body.update(extra)
        return S.call(self.app, 'POST', path, json_body=body, principal=principal, headers=headers)

    def messages(self, model, *, stream=True, **kw):
        body = {'model': model, 'max_tokens': 64, 'messages': CHAT, 'stream': stream}
        return S.call(self.app, 'POST', '/v1/messages', json_body=body, principal=CLOUD, **kw)


class LocalFallback(Case):
    def test_local_triggers_advance_only_before_commit(self):
        for status, code in ((409, 'model_not_loaded'), (429, 'rate_limit_error'), (429, 'queue_full'),
                             (503, 'queue_full'), (503, 'worker_unavailable')):
            with self.subTest(status=status, code=code):
                self.local.calls.clear()
                self.local.errors = {'qwen-local': (status, code)}
                reply = self.chat('local-pair')
                self.assertEqual(reply.status, 200)
                self.assertEqual(reply.header('x-titanos-served-model'), 'llama-b')
                self.assertEqual(reply.header('x-titanos-fallback-count'), '1')
                self.assertEqual([c['model'] for c in self.local.calls], ['qwen-local', 'llama-b'])
        self.assertTrue({'queue_full', 'model_not_loaded', 'worker_unavailable'} <= FALLBACK_LOCAL)
        self.assertIn('upstream_rate_limited', FALLBACK_CLOUD)

    def test_local_4xx_and_success_never_fall_back(self):
        for status, code in ((400, 'invalid_request_error'), (404, 'model_not_found'), (422, 'unprocessable')):
            with self.subTest(status=status):
                self.local.calls.clear()
                self.local.errors = {'qwen-local': (status, code)}
                reply = self.chat('local-pair')
                self.assertEqual((reply.status, reply.json()['error']['code']), (status, code))
                self.assertEqual(len(self.local.calls), 1)
                self.assertEqual(reply.header('x-titanos-fallback-count'), '0')
        self.local.errors = {}
        self.local.calls.clear()
        reply = self.chat('local-pair')
        self.assertEqual((reply.status, len(self.local.calls)), (200, 1))
        self.assertEqual(self.hits(), 0)

    def test_local_attempt_flushes_large_and_unknown_errors(self):
        sent = []

        async def send(message):
            sent.append(message)
        import asyncio
        attempt = LocalAttempt(send, fallback_codes=frozenset({'queue_full'}), buffer_limit=8)

        async def go():
            await attempt({'type': 'http.response.start', 'status': 503, 'headers': []})
            await attempt({'type': 'http.response.body', 'body': b'0123456789', 'more_body': True})
            await attempt({'type': 'http.response.body', 'body': b'tail', 'more_body': False})
        asyncio.run(go())
        self.assertTrue(attempt.committed)
        self.assertEqual(b''.join(m.get('body', b'') for m in sent), b'0123456789tail')


class CloudFallback(Case):
    def test_anthropic_error_first_stream_falls_back_before_commit(self):
        MOCKS['anthropic'].faults.push('error_first_event')
        reply = self.messages('ant-then-oa')
        self.assertEqual(reply.status, 200, reply.body[:300])
        self.assertEqual(reply.header('x-titanos-served-by'), 'cloud:openai')
        self.assertEqual(reply.header('x-titanos-translation'), 'anthropic->openai')
        self.assertEqual(reply.header('x-titanos-fallback-count'), '1')
        self.assertNotIn(b'overloaded', reply.body)
        self.assertNotIn(b'"type":"error"', reply.body.replace(b' ', b''))
        self.assertTrue(reply.body.startswith(b'event: message_start'))
        self.assertIn(b'event: message_stop', reply.body)
        self.assertEqual((len(MOCKS['anthropic'].hits), len(MOCKS['openai'].hits)), (1, 1))

    def test_upstream_400_is_passed_through_and_never_retried(self):
        MOCKS['openai'].faults.push('status', status=400, leak=False)
        reply = self.chat('oa-then-ant')
        self.assertEqual(reply.status, 400)
        self.assertEqual(reply.header('x-titanos-error-origin'), 'upstream')
        self.assertEqual(len(MOCKS['anthropic'].hits), 0)
        MOCKS['openai'].faults.push('status', status=503)
        reply = self.chat('oa-then-ant')
        self.assertEqual((reply.status, reply.header('x-titanos-served-by')), (200, 'cloud:anthropic'))
        self.assertEqual(reply.header('x-titanos-translation'), 'openai->anthropic')
        self.assertEqual(reply.header('x-titanos-injected'), 'max_tokens')  # Anthropic requires max_tokens
        self.assertEqual(MOCKS['anthropic'].hits[-1].json()['model'], 'claude-sonnet-4-5')
        self.assertEqual(reply.json()['model'], 'oa-then-ant')  # translated answers carry the requested name

    def test_no_fallback_for_edits_store_or_previous_response_id(self):
        MOCKS['openai'].faults.push('status', status=503, times=3)
        body, ctype = S.multipart([('model', 'edits-vm'), ('prompt', 'a cat'),
                                   ('image', ('cat.png', b'\x89PNG\r\n\x1a\nfake', 'image/png'))])
        reply = S.call(self.app, 'POST', '/v1/images/edits', body=body, content_type=ctype, principal=CLOUD)
        self.assertEqual((reply.status, reply.json()['error']['code']), (503, 'upstream_unavailable'))
        storer = S.key('inference', 'cloud', 'cloud:store')
        reply = S.call(self.app, 'POST', '/v1/responses', principal=storer,
                       json_body={'model': 'oa-then-local', 'input': 'hi', 'store': True})
        self.assertEqual(reply.status, 503)
        self.broker.handles.bind('responses', 'resp_prev', 'openai', 'oa', storer.key_id)
        reply = S.call(self.app, 'POST', '/v1/responses', principal=storer,
                       json_body={'model': 'oa-then-local', 'input': 'hi', 'previous_response_id': 'resp_prev'})
        self.assertEqual(reply.status, 503)
        self.assertEqual(self.local.calls, [])
        self.assertEqual(len(MOCKS['openai'].hits), 3)


class Streaming(Case):
    def test_no_keepalive_before_commit_and_keepalives_after_it(self):
        self.app.keepalive_seconds = 0.05  # the 15 s keepalive interval, scaled down
        openai = MOCKS['openai']
        openai.faults.push('slow_first_byte', delay=0.8)  # more than 15 keepalive intervals before the first byte
        openai.hold(frame=3)
        timer = threading.Timer(1.3, openai.gate.set)
        timer.start()
        try:
            reply = self.messages(GPT)
        finally:
            timer.cancel()
            openai.gate.set()
        self.assertEqual(reply.status, 200)
        self.assertGreaterEqual(reply.started, 0.7)
        kinds = [m['type'] for _, m in reply.messages]
        self.assertEqual(kinds[0], 'http.response.start')  # nothing at all was sent before commit
        keepalives = [t for t, m in reply.messages if m.get('body') == KEEPALIVE]
        self.assertTrue(keepalives, 'no keepalive while the committed stream was idle')
        self.assertTrue(all(t >= reply.started for t in keepalives))
        self.assertIn(b'event: message_stop', reply.body)
        self.assertEqual(reply.header('cache-control'), 'no-cache')
        # passthrough streams are byte-faithful: no keepalive comments are inserted
        openai.hold(frame=3)
        timer = threading.Timer(0.4, openai.gate.set)
        timer.start()
        try:
            plain = self.chat(GPT, stream=True)
        finally:
            timer.cancel()
            openai.gate.set()
        self.assertNotIn(KEEPALIVE, plain.body)

    def test_mid_stream_disconnect_gives_the_terminal_frame_and_no_second_hit(self):
        MOCKS['openai'].faults.push('disconnect_after_frames', frames=2)
        reply = self.chat('oa-then-ant', stream=True)
        self.assertEqual(reply.status, 200)
        self.assertIn(b'"type": "stream_interrupted"', reply.body.replace(b'"type":"', b'"type": "'))
        self.assertTrue(reply.body.endswith(b'data: [DONE]\n\n'))
        self.assertEqual((len(MOCKS['openai'].hits), len(MOCKS['anthropic'].hits)), (1, 0))
        MOCKS['openai'].faults.push('disconnect_after_frames', frames=2)
        reply = self.messages('oa-then-ant')
        self.assertIn(b'event: error', reply.body)
        self.assertEqual(len(MOCKS['anthropic'].hits), 0)
        decision = self.broker.decisions.query(principal=S.BOOTSTRAP, limit=1)[0]
        self.assertEqual(decision['committed_at_attempt'], 0)

    def test_client_disconnect_closes_the_upstream(self):
        openai = MOCKS['openai']
        openai.hold(frame=3)
        reply = S.call(self.app, 'POST', '/v1/chat/completions', principal=CLOUD,
                       json_body={'model': GPT, 'messages': CHAT, 'stream': True}, disconnect_after=1)
        self.assertEqual(reply.status, 200)
        hit = openai.hits[-1]
        self.assertTrue(wait_until(lambda: hit.disconnected or hit.aborted, 5.0), 'the upstream stayed open')
        self.assertFalse(hit.completed)
        counts = self.broker.metrics.to_dict()['requests_by_surface_class_outcome']
        self.assertIn('cancelled', {row['outcome'] for row in counts})


class Provenance(Case):
    REQUIRED = ('x-request-id', 'x-titanos-decision-id', 'x-titanos-fallback-count')

    def check(self, reply, *, served=True):
        for name in self.REQUIRED:
            self.assertIn(name, reply.headers, (name, reply.status))
        if served:
            for name in ('x-titanos-served-by', 'x-titanos-served-model', 'x-titanos-translation',
                         'x-titanos-requested-model'):
                self.assertIn(name, reply.headers, (name, reply.status))

    def test_provenance_headers_on_every_path(self):
        rid = {'X-Request-ID': 'req-wp6-0001'}
        local = self.chat('qwen-local', headers=rid)
        self.check(local)
        self.assertEqual((local.header('x-request-id'), local.header('x-titanos-served-by')),
                         ('req-wp6-0001', 'local:native'))
        self.assertEqual(local.header('x-titanos-translation'), 'local')
        engine = self.chat('llama-b')
        self.assertEqual(engine.header('x-titanos-served-by'), 'local:engine')
        self.local.errors = {'llama-b': (400, 'invalid_request_error')}
        self.check(self.chat('llama-b'))
        cloud = self.chat(GPT, stream=True)
        self.check(cloud)
        self.assertEqual(cloud.header('x-titanos-translation'), 'passthrough')
        self.assertTrue(cloud.header('x-titanos-upstream-request-id'))
        self.check(self.messages(GPT, stream=False))
        refused = self.chat('nope')
        self.check(refused, served=False)
        self.assertEqual(refused.header('x-titanos-error-origin'), 'titanos')
        self.assertEqual(refused.header('x-titanos-requested-model'), 'nope')


class FastPath(Case):
    def test_inactive_fast_path_touches_neither_scope_nor_body(self):
        home = Path(tempfile.mkdtemp(prefix='titanos-wp6-empty-'))
        self.addCleanup(shutil.rmtree, home, True)
        broker = S.Broker(settings=S.settings(), config=S.local_config(LOCAL), home=home)
        local = S.FakeLocal()
        app = BrokerDispatch(local, broker)
        for principal in (None, S.BOOTSTRAP, S.key('inference', kind='anonymous_loopback', id='anonymous')):
            with self.subTest(principal=getattr(principal, 'kind', None)):
                reply = S.call(app, 'POST', '/v1/chat/completions', principal=principal,
                               json_body={'model': 'gpt-4o-mini@openai', 'messages': CHAT})
                self.assertEqual(reply.status, 200)
                call = local.calls[-1]
                self.assertNotIn('titanos.broker.target', call['scope'])
                self.assertNotIn('x-titanos-decision-id', reply.headers)
                self.assertEqual(call['receive'].__qualname__, 'acall.<locals>.receive')  # the caller's own receive
                self.assertEqual(json.loads(call['body'])['model'], 'gpt-4o-mini@openai')
        self.assertEqual(app.intercepted, 0)
        self.assertEqual(sorted(p.name for p in home.iterdir()), [])
        # a key principal makes the broker look at the request even without broker.json
        keyed = S.call(app, 'POST', '/v1/chat/completions', principal=S.key('inference'),
                       json_body={'model': 'qwen-local', 'messages': CHAT})
        self.assertEqual((keyed.status, app.intercepted), (200, 1))
        self.assertIn('x-titanos-decision-id', keyed.headers)

    def test_model_absent_and_unparseable_bodies_reach_the_local_route_unchanged(self):
        body = json.dumps({'text': 'hello'}).encode()
        reply = S.call(self.app, 'POST', '/v1/text-to-speech/voice-1', body=body, content_type='application/json',
                       principal=CLOUD)
        self.assertEqual(reply.status, 200)
        self.assertEqual((self.local.calls[-1]['path'], self.local.calls[-1]['body']), ('/v1/text-to-speech/voice-1',
                                                                                        body))
        broken = b'{"model": "gpt-4o-mini@openai", '
        S.call(self.app, 'POST', '/v1/chat/completions', body=broken, content_type='application/json',
               principal=S.BOOTSTRAP)
        self.assertEqual(self.local.calls[-1]['body'], broken)
        # a scoped key never reaches a local route with a body the broker could not read (fail closed)
        calls = len(self.local.calls)
        refused = S.call(self.app, 'POST', '/v1/chat/completions', body=broken, content_type='application/json',
                         principal=CLOUD)
        self.assertEqual((refused.status, refused.json()['error']['code']), (400, 'invalid_request'))
        self.assertEqual(len(self.local.calls), calls)
        self.assertEqual(self.hits(), 0)

    def test_bom_and_utf16_json_bodies_are_parsed_as_the_local_routes_parse_them(self):
        """A UTF-8 BOM or a UTF-16 body is parsed (json.loads on the raw bytes), so key constraints and limits apply:
        a key limited to one local model cannot reach another one by re-encoding its body."""
        narrow = S.key('inference', 'inference:chat', models=('qwen-local',), classes=('local',))
        plain = json.dumps({'model': 'llama-b', 'messages': CHAT})
        for label, raw in (('plain', plain.encode()), ('utf-8 bom', b'\xef\xbb\xbf' + plain.encode()),
                           ('utf-16', plain.encode('utf-16')), ('utf-32', plain.encode('utf-32'))):
            with self.subTest(label):
                calls = len(self.local.calls)
                reply = S.call(self.app, 'POST', '/v1/chat/completions', body=raw, content_type='application/json',
                               principal=narrow)
                self.assertEqual((reply.status, reply.json()['error']['code']), (404, 'model_not_found'))
                self.assertIn('x-titanos-decision-id', reply.headers)
                self.assertEqual(len(self.local.calls), calls)
        allowed = json.dumps({'model': 'qwen-local', 'messages': CHAT}).encode('utf-16')
        reply = S.call(self.app, 'POST', '/v1/chat/completions', body=allowed, content_type='application/json',
                       principal=narrow)
        self.assertEqual(reply.status, 200)
        self.assertTrue(reply.header('x-titanos-served-by').startswith('local:'))
        self.assertEqual(self.local.calls[-1]['model'], 'qwen-local')
        self.assertEqual(self.hits(), 0)

    def test_install_broker_creates_no_files_and_opens_no_sockets(self):
        home = Path(tempfile.mkdtemp(prefix='titanos-wp6-install-'))
        self.addCleanup(shutil.rmtree, home, True)
        created = []
        original = socket.socket.__init__

        def counting(self_, *args, **kw):
            created.append(args)
            return original(self_, *args, **kw)

        class App:
            state = type('State', (), {})()
        app = App()
        with mock.patch.object(socket.socket, '__init__', counting):
            broker = install_broker(app, settings=S.settings(), config=S.local_config(LOCAL), home=home,
                                    outputs=home / 'outputs')
            self.assertIs(app.state.broker, broker)
            self.assertFalse(broker.active(None))
            self.assertFalse(broker.configured)
            self.assertEqual(broker.catalog_for(None), [])
        self.assertEqual(created, [])
        self.assertEqual(list(home.iterdir()), [])


class Rewrites(Case):
    def test_alias_rewrite_on_json_and_multipart(self):
        reply = self.chat('chat-vm')
        self.assertEqual(reply.status, 200)
        call = self.local.calls[-1]
        self.assertEqual(json.loads(call['body'])['model'], 'qwen-local')
        self.assertEqual(int(call['headers']['content-length']), len(call['body']))
        audio = b'RIFF\x00\x00\x00\x00WAVEfmt ' + bytes(range(64))
        body, ctype = S.multipart([('model', 'stt-vm'), ('file', ('a.wav', audio, 'audio/wav')),
                                   ('response_format', 'json')], boundary='keep-this-boundary')
        reply = S.call(self.app, 'POST', '/v1/audio/transcriptions', body=body, content_type=ctype, principal=CLOUD)
        self.assertEqual(reply.status, 200)
        call = self.local.calls[-1]
        self.assertIn(b'--keep-this-boundary', call['body'])
        self.assertIn(audio, call['body'])
        self.assertEqual(call['model'], 'whisper-local')
        self.assertEqual(call['body'].replace(b'whisper-local', b'stt-vm'), body)
        self.assertEqual(int(call['headers']['content-length']), len(call['body']))
        self.assertEqual(call['target'].local_name, 'whisper-local')

    def test_gemini_and_elevenlabs_stt_reach_local_models_through_the_openai_route(self):
        gemini = {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]}
        reply = S.call(self.app, 'POST', '/v1beta/models/qwen-local:generateContent', json_body=gemini,
                       principal=CLOUD)
        self.assertEqual(reply.status, 200, reply.body[:300])
        self.assertEqual(reply.json()['candidates'][0]['content']['parts'][0]['text'], 'local hello')
        self.assertEqual(self.local.calls[-1]['path'], '/v1/chat/completions')
        self.assertEqual(reply.header('x-titanos-translation'), 'gemini->openai')
        stream = S.call(self.app, 'POST', '/v1beta/models/qwen-local:streamGenerateContent', query='alt=sse',
                        json_body=gemini, principal=CLOUD)
        self.assertEqual(stream.status, 200)
        self.assertIn(b'local ', stream.body)
        self.assertIn(b'finishReason', stream.body)
        body, ctype = S.multipart([('model_id', 'whisper-local'), ('file', ('a.wav', b'RIFFxxxx', 'audio/wav'))])
        stt = S.call(self.app, 'POST', '/v1/speech-to-text', body=body, content_type=ctype, principal=CLOUD)
        self.assertEqual((stt.status, stt.json()['text']), (200, 'local transcript'))
        self.assertEqual(self.local.calls[-1]['path'], '/v1/audio/transcriptions')
        self.assertEqual(self.hits(), 0)


class Budgets(Case):
    def test_budget_is_admitted_before_any_hit(self):
        tight = S.key('inference', 'cloud', budget={'usd_per_day': '0.000001'})
        reply = self.chat(GPT, principal=tight)
        self.assertEqual((reply.status, reply.json()['error']['code']), (402, 'budget_exceeded'))
        self.assertEqual(reply.header('x-should-retry'), 'false')
        unpriced = self.chat(CLAUDE, principal=tight)
        self.assertEqual((unpriced.status, unpriced.json()['error']['code']), (402, 'budget_unpriced_target'))
        self.assertEqual(self.hits(), 0)
        roomy = S.key('inference', 'cloud', budget={'usd_per_day': '1000'})
        reply = self.chat(GPT, principal=roomy)
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.header('x-titanos-injected'), 'max_tokens')
        self.assertEqual(MOCKS['openai'].hits[-1].json()['max_tokens'], self.broker.settings.broker_default_max_output_tokens)
        usage = self.broker.usage.totals(key_id=roomy.key_id)
        self.assertGreater(usage[0], 0)

    def test_budgeted_transcription_without_usage_spends_its_reservation(self):
        # response_format=text answers a bare string: no usage, no duration. The pre-dispatch reservation is the
        # upload's duration (exact for WAV, size at the 1000 B/s floor otherwise) and is kept as the spend.
        doc = document()
        doc['upstreams']['openai']['models'] = doc['upstreams']['openai']['models'] + ['whisper-1']
        doc['prices'] = doc['prices'] + [{'match': 'whisper-1@openai', 'per_audio_minute': '0.60'}]
        self.lab.close()
        self.lab = S.Lab(doc, local=LOCAL)
        self.broker = self.lab.broker
        self.app = BrokerDispatch(self.local, self.broker)
        tight = S.key('inference', 'cloud', budget={'usd_per_day': '0.10'})
        payload = b'\x00' * 30000  # not a WAV: 30 s at the floor rate -> USD 0.30
        body, ctype = S.multipart([('model', 'whisper-1@openai'), ('file', ('a.mp3', payload, 'audio/mpeg')),
                                   ('response_format', 'text')])
        reply = S.call(self.app, 'POST', '/v1/audio/transcriptions', body=body, content_type=ctype, principal=tight)
        self.assertEqual((reply.status, reply.json()['error']['code']), (402, 'budget_exceeded'))
        self.assertEqual(self.hits(), 0)
        roomy = S.key('inference', 'cloud', budget={'usd_per_day': '5'})
        reply = S.call(self.app, 'POST', '/v1/audio/transcriptions', body=body, content_type=ctype, principal=roomy)
        self.assertEqual(reply.status, 200, reply.body[:300])
        self.assertEqual(len(MOCKS['openai'].hits), 1)
        self.assertEqual(Decimal(self.broker.limiter.remaining(roomy)['usd_today']), Decimal('4.70'))
        self.assertEqual(self.broker.usage.totals(key_id=roomy.key_id)[0], Decimal('0.30'))
        import io
        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as writer:  # 2 s of 8 kHz mono PCM: an exact duration
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(8000)
            writer.writeframes(b'\x00\x00' * 16000)
        body, ctype = S.multipart([('model', 'whisper-1@openai'), ('file', ('a.wav', buffer.getvalue(), 'audio/wav')),
                                   ('response_format', 'text')])
        reply = S.call(self.app, 'POST', '/v1/audio/transcriptions', body=body, content_type=ctype, principal=roomy)
        self.assertEqual(reply.status, 200, reply.body[:300])
        self.assertEqual(Decimal(self.broker.limiter.remaining(roomy)['usd_today']), Decimal('4.68'))
        self.assertEqual(self.broker.usage.totals(key_id=roomy.key_id)[0], Decimal('0.32'))


class Images(Case):
    def test_png_url_rehosting_and_jpeg_refusal(self):
        outputs = Path(tempfile.mkdtemp(prefix='titanos-wp6-files-'))
        self.addCleanup(shutil.rmtree, outputs, True)
        self.broker.bind_artifact_writer(lambda name: outputs / name)
        reply = S.call(self.app, 'POST', '/v1/images/generations', principal=CLOUD,
                       json_body={'model': 'dall-e-3@openai', 'prompt': 'a cat', 'response_format': 'url'})
        self.assertEqual(reply.status, 200, reply.body[:300])
        url = reply.json()['data'][0]['url']
        self.assertRegex(url, r'^(https?://[^/]+)?/v1/files/[0-9a-f]{32}\.png$')  # absolute when the scope has a host
        stored = outputs / url.rsplit('/', 1)[1]
        self.assertTrue(stored.read_bytes().startswith(b'\x89PNG'))
        # every OpenAI image model refuses response_format (live 2026-10-02): the base64 default answer is re-hosted
        self.assertNotIn('response_format', MOCKS['openai'].hits[-1].json())
        self.assertEqual(self.broker.ownership.owner('artifact', stored.name), CLOUD.key_id)
        # xAI answers its own image URLs (JPEG; extras image_url_native): a url request is passed through, never
        # re-hosted and never refused (the PNG-only re-hosting refusal no longer applies to a url-native provider)
        jpeg = S.call(self.app, 'POST', '/v1/images/generations', principal=CLOUD,
                      json_body={'model': 'grok-2-image@xai', 'prompt': 'a cat', 'response_format': 'url'})
        self.assertEqual(jpeg.status, 200, jpeg.body[:300])
        self.assertTrue(jpeg.json()['data'][0]['url'].startswith('https://example.invalid/xai/'))
        self.assertEqual(MOCKS['xai'].hits[-1].json().get('response_format'), 'url')
        b64 = S.call(self.app, 'POST', '/v1/images/generations', principal=CLOUD,
                     json_body={'model': 'grok-2-image@xai', 'prompt': 'a cat', 'response_format': 'b64_json'})
        self.assertEqual(b64.status, 200)
        self.assertEqual(base64.b64decode(b64.json()['data'][0]['b64_json'])[:3], b'\xff\xd8\xff')

    def test_openai_images_never_send_response_format(self):
        """Live 2026-10-02: OpenAI's Images API answers 400 unknown_parameter for response_format on dall-e models too
        (image_b64_only_globs '*'). The upstream body or form never carries the field; a client url still gets re-hosted
        absolute URLs; b64_json or no format returns the upstream base64 unchanged (generations and edits)."""
        outputs = Path(tempfile.mkdtemp(prefix='titanos-wp6-files-'))
        self.addCleanup(shutil.rmtree, outputs, True)
        self.broker.bind_artifact_writer(lambda name: outputs / name)
        openai = MOCKS['openai']
        for model in ('dall-e-2@openai', 'dall-e-3@openai'):
            for fmt in ('url', 'b64_json', None):
                with self.subTest(model=model, response_format=fmt):
                    body = {'model': model, 'prompt': 'a cat', 'n': 1, 'size': '256x256'}
                    if fmt is not None:
                        body['response_format'] = fmt
                    before = len(openai.hits)
                    reply = S.call(self.app, 'POST', '/v1/images/generations', principal=CLOUD, json_body=body)
                    self.assertEqual(reply.status, 200, reply.body[:300])
                    self.assertEqual(len(openai.hits), before + 1, 'one upstream call, never a retry')
                    sent = openai.hits[-1].json()
                    self.assertNotIn('response_format', sent)
                    self.assertEqual(sent, {k: v for k, v in dict(body, model=model.split('@')[0]).items()
                                            if k != 'response_format'})
                    item = reply.json()['data'][0]
                    if fmt == 'url':
                        self.assertNotIn('b64_json', item)
                        self.assertRegex(item['url'], r'^(https?://[^/]+)?/v1/files/[0-9a-f]{32}\.png$')
                        self.assertTrue((outputs / item['url'].rsplit('/', 1)[1]).read_bytes()
                                        .startswith(b'\x89PNG\r\n\x1a\n'))
                    else:
                        self.assertNotIn('url', item)
                        self.assertEqual(base64.b64decode(item['b64_json'])[:8], b'\x89PNG\r\n\x1a\n')
                    decision = self.broker.decisions.query(limit=1, principal=S.BOOTSTRAP)[0]
                    self.assertEqual('dropped:response_format' in decision['notes'], fmt is not None)
        png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 16
        for fmt in ('url', 'b64_json', None):
            with self.subTest(surface='images.edits', response_format=fmt):
                fields = [('model', 'dall-e-2@openai'), ('prompt', 'make it blue'),
                          ('image', ('square.png', png, 'image/png'))]
                if fmt is not None:
                    fields.append(('response_format', fmt))
                body, ctype = S.multipart(fields)
                before = len(openai.hits)
                reply = S.call(self.app, 'POST', '/v1/images/edits', body=body, content_type=ctype, principal=CLOUD)
                self.assertEqual(reply.status, 200, reply.body[:300])
                self.assertEqual(len(openai.hits), before + 1)
                sent = openai.hits[-1].body
                self.assertNotIn(b'name="response_format"', sent)
                self.assertIn(b'name="model"\r\n\r\ndall-e-2\r\n', sent)
                item = reply.json()['data'][0]
                if fmt == 'url':
                    self.assertNotIn('b64_json', item)
                    self.assertRegex(item['url'], r'^(https?://[^/]+)?/v1/files/[0-9a-f]{32}\.png$')
                else:
                    self.assertEqual(base64.b64decode(item['b64_json'])[:8], b'\x89PNG\r\n\x1a\n')


class HandleRoutes(Case):
    def test_handle_bound_routes_serve_the_owner_only(self):
        owner = S.key('inference', 'cloud', 'cloud:store')
        created = S.call(self.app, 'POST', '/v1/responses', principal=owner,
                         json_body={'model': 'mini', 'input': 'hi', 'store': True})
        self.assertEqual(created.status, 200, created.body[:300])
        rid = created.json()['id']
        got = S.call(self.app, 'GET', f'/v1/responses/{rid}', principal=owner)
        self.assertEqual((got.status, got.json()['id']), (200, rid))
        self.assertEqual(got.header('x-titanos-served-by'), 'cloud:openai')
        before = len(MOCKS['openai'].hits)
        other = S.call(self.app, 'GET', f'/v1/responses/{rid}', principal=CLOUD)
        self.assertEqual((other.status, other.json()['error']['code']), (404, 'response_not_found'))
        self.assertEqual(len(MOCKS['openai'].hits), before)
        gone = S.call(self.app, 'DELETE', f'/v1/responses/{rid}', principal=owner)
        self.assertEqual(gone.status, 200)
        self.assertIsNone(self.broker.handles.owner('responses', rid))
        admin = S.call(self.app, 'GET', f'/v1/responses/{rid}', principal=S.BOOTSTRAP)
        self.assertEqual(admin.status, 404)

    def test_handle_bound_routes_apply_the_principal_and_policy_gates(self):
        """The owner's key id alone is not enough: cloud scope, upstream and class constraints, the key's locations,
        X-Titanos-Location and the 1-hop rule are enforced exactly as on the main path, with zero upstream hits."""
        scopes = ('inference', 'cloud', 'cloud:store')
        owner = S.key(*scopes, id='kowner00001')
        created = S.call(self.app, 'POST', '/v1/responses', principal=owner,
                         json_body={'model': GPT, 'input': 'hi', 'store': True})
        self.assertEqual(created.status, 200, created.body[:300])
        rid = created.json()['id']
        before = len(MOCKS['openai'].hits)
        cases = [
            ('no cloud scope', S.key('inference', id='kowner00001'), {}, 404, 'response_not_found'),
            ('other upstreams only', S.key(*scopes, id='kowner00001', upstreams=('other',)), {}, 404,
             'response_not_found'),
            ('local class only', S.key(*scopes, id='kowner00001', classes=('local',)), {}, 404, 'response_not_found'),
            ('local_only key', S.key(*scopes, id='kowner00001', locations=('local_only',)), {}, 400,
             'location_unsatisfiable'),
            ('hop header', owner, {'X-Titanos-Hop': 'abcdef0123456789'}, 400, 'location_unsatisfiable'),
            ('local_only header', owner, {'X-Titanos-Location': 'local_only'}, 400, 'location_unsatisfiable'),
        ]
        for label, principal, headers, status, code in cases:
            for method in ('GET', 'DELETE'):
                with self.subTest(label, method=method):
                    reply = S.call(self.app, method, f'/v1/responses/{rid}', principal=principal, headers=headers)
                    self.assertEqual((reply.status, reply.json()['error']['code']), (status, code))
        self.assertEqual(len(MOCKS['openai'].hits), before)
        self.assertIsNotNone(self.broker.handles.owner('responses', rid))
        self.lab.rewrite(dict(document(), policy={'cloud_policy': 'off'}))
        self.broker.reload()
        off = S.call(self.app, 'GET', f'/v1/responses/{rid}', principal=owner)
        self.assertEqual((off.status, off.json()['error']['code']), (403, 'cloud_disabled'))
        self.assertEqual(len(MOCKS['openai'].hits), before)
        self.lab.rewrite(document())
        self.broker.reload()
        got = S.call(self.app, 'GET', f'/v1/responses/{rid}', principal=owner)
        self.assertEqual((got.status, got.json()['id']), (200, rid))


class Rehost(Case):
    """PNG url re-hosting validates every item before writing any file, fails closed when the owner cannot be bound,
    and maps undecodable base64 to 502 upstream_bad_response."""

    def setUp(self):
        super().setUp()
        self.outputs = Path(tempfile.mkdtemp(prefix='titanos-wp6-rehost-'))
        self.addCleanup(shutil.rmtree, self.outputs, True)
        self.broker.bind_artifact_writer(lambda name: self.outputs / name)
        scope = {'type': 'http', 'scheme': 'http', 'server': ('127.0.0.1', 8000), 'root_path': '',
                 'path': '/v1/images/generations', 'headers': [(b'host', b'127.0.0.1:8000')]}
        self.run_ = type('RunStub', (), {'principal': CLOUD, 'scope': scope})()  # the url base comes from the scope

    def answer(self, *items) -> bytes:
        return json.dumps({'created': 1, 'data': [{'b64_json': item} for item in items]}).encode()

    def refused(self, data: bytes):
        from titanos.broker.errors import BrokerError
        with self.assertRaises(BrokerError) as caught:
            self.app._rehost(self.run_, data)
        return caught.exception

    def test_rehost_error_paths_leave_no_files(self):
        png = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 16).decode()
        jpeg = base64.b64encode(b'\xff\xd8\xff\xe0' + b'\x00' * 16).decode()
        error = self.refused(self.answer(png, jpeg))
        self.assertEqual((error.code, error.status), ('broker_capability_unavailable', 400))
        self.assertEqual(list(self.outputs.iterdir()), [])
        error = self.refused(self.answer(png, 'abc'))  # incorrect padding
        self.assertEqual((error.code, error.status), ('upstream_bad_response', 502))
        self.assertEqual(list(self.outputs.iterdir()), [])
        with mock.patch.object(self.broker.ownership, 'bind', side_effect=OSError('read-only')):
            error = self.refused(self.answer(png, png))
        self.assertEqual(error.status, 503)
        self.assertEqual(list(self.outputs.iterdir()), [])
        document_ = json.loads(self.app._rehost(self.run_, self.answer(png, png)))
        names = sorted(item['url'].rsplit('/', 1)[1] for item in document_['data'])
        self.assertEqual(sorted(p.name for p in self.outputs.iterdir()), names)
        self.assertTrue(all(self.broker.ownership.owner('artifact', name) == CLOUD.key_id for name in names))


def wav_bytes(seconds: float, *, rate: int = 24000, streamed: bool = False) -> bytes:
    """PCM16 mono WAV; streamed=True writes the 0xFFFFFFFF placeholder sizes a streaming TTS answer carries (OpenAI)."""
    import io
    import struct
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b'\x00\x00' * int(round(seconds * rate)))
    data = bytearray(buffer.getvalue())
    if streamed:
        data[4:8] = struct.pack('<I', 0xFFFFFFFF)
        data[40:44] = struct.pack('<I', 0xFFFFFFFF)
    return bytes(data)


class OutputCapOnlyOnTokenSurfaces(Case):
    """Live lane run live-20261002T100017Z-835910c9 follow-up: a budgeted key's default output cap is injected on the
    token-generating surfaces only. The exact upstream body of images, speech, transcription, embeddings, moderations and
    count_tokens is the client's body with the model rewritten, and no X-Titanos-Injected header is stamped."""

    PRICED = ['gpt-4o-mini', 'dall-e-2', 'dall-e-3', 'text-embedding-3-small', 'omni-moderation-latest',
              'gpt-4o-mini-tts', 'gpt-4o-mini-transcribe']

    def setUp(self):
        super().setUp()
        doc = document()
        doc['upstreams']['openai']['models'] = self.PRICED
        doc['prices'] = doc['prices'] + [
            {'match': 'dall-e-2@openai', 'per_image': '0.02'},
            {'match': 'text-embedding-3-small@openai', 'input_per_mtok': '0.02', 'output_per_mtok': '0'},
            {'match': 'omni-moderation-latest@openai', 'input_per_mtok': '0', 'output_per_mtok': '0'},
            {'match': 'gpt-4o-mini-tts@openai', 'per_1k_chars': '0.015'},
            {'match': 'gpt-4o-mini-transcribe@openai', 'per_audio_minute': '0.006'},
            {'match': CLAUDE, 'input_per_mtok': '3', 'output_per_mtok': '15'}]
        self.lab.close()
        self.lab = S.Lab(doc, local=LOCAL)
        self.broker = self.lab.broker
        self.app = BrokerDispatch(self.local, self.broker)
        self.budgeted = S.key('inference', 'cloud', budget={'usd_per_day': '50'})

    def last_decision(self) -> dict:
        return self.broker.decisions.query(limit=1, principal=S.BOOTSTRAP)[0]

    def test_no_output_cap_on_non_token_surfaces(self):
        openai = MOCKS['openai']
        cases = [
            # the live lane's image body; OpenAI refuses response_format for every image model (live 2026-10-02), so
            # the field is the one thing dropped from it
            ('/v1/images/generations', {'model': 'dall-e-2@openai', 'prompt': 'a plain blue square', 'n': 1,
                                        'size': '256x256', 'response_format': 'b64_json'}),
            ('/v1/embeddings', {'model': 'text-embedding-3-small@openai', 'input': 'hello', 'encoding_format': 'float'}),
            ('/v1/moderations', {'model': 'omni-moderation-latest@openai', 'input': 'hello'}),
            ('/v1/audio/speech', {'model': 'gpt-4o-mini-tts@openai', 'input': 'hello', 'voice': 'alloy',
                                  'response_format': 'wav'}),
        ]
        for path, body in cases:
            with self.subTest(path):
                before = len(openai.hits)
                reply = S.call(self.app, 'POST', path, json_body=body, principal=self.budgeted)
                self.assertEqual(reply.status, 200, reply.body[:300])
                self.assertEqual(len(openai.hits), before + 1)
                expected = dict(body, model=body['model'].split('@', 1)[0])
                if path == '/v1/images/generations':
                    expected.pop('response_format')
                self.assertEqual(openai.hits[-1].json(), expected)  # the exact upstream body: nothing injected
                self.assertIsNone(reply.header('x-titanos-injected'))
                self.assertNotIn('injected:max_tokens', self.last_decision()['notes'])
        audio = wav_bytes(1.0)
        body, ctype = S.multipart([('model', 'gpt-4o-mini-transcribe@openai'),
                                   ('file', ('probe.wav', audio, 'audio/wav')), ('response_format', 'json')])
        reply = S.call(self.app, 'POST', '/v1/audio/transcriptions', body=body, content_type=ctype,
                       principal=self.budgeted)
        self.assertEqual(reply.status, 200, reply.body[:300])
        sent = openai.hits[-1].body
        import re
        self.assertEqual(sorted(re.findall(rb'; name="([^"]+)"', sent)), [b'file', b'model', b'response_format'])
        self.assertIn(b'name="model"\r\n\r\ngpt-4o-mini-transcribe\r\n', sent)
        self.assertNotIn(b'max_tokens', sent)
        self.assertIn(audio, sent)
        self.assertIsNone(reply.header('x-titanos-injected'))
        count = {'model': CLAUDE, 'messages': CHAT}
        reply = S.call(self.app, 'POST', '/v1/messages/count_tokens', json_body=count, principal=self.budgeted)
        self.assertEqual(reply.status, 200, reply.body[:300])
        self.assertEqual(MOCKS['anthropic'].hits[-1].json(), dict(count, model='claude-sonnet-4-5'))
        self.assertIsNone(reply.header('x-titanos-injected'))

    def test_output_cap_still_injected_on_token_surfaces(self):
        default = self.broker.settings.broker_default_max_output_tokens
        openai = MOCKS['openai']
        reply = self.chat(GPT, principal=self.budgeted)
        self.assertEqual(reply.status, 200, reply.body[:300])
        self.assertEqual(openai.hits[-1].json(), {'model': 'gpt-4o-mini', 'messages': CHAT, 'stream': False,
                                                  'max_tokens': default})
        self.assertEqual(reply.header('x-titanos-injected'), 'max_tokens')
        self.assertIn('injected:max_tokens', self.last_decision()['notes'])
        reply = S.call(self.app, 'POST', '/v1/responses', json_body={'model': GPT, 'input': 'hi', 'store': False},
                       principal=self.budgeted)
        self.assertEqual(reply.status, 200, reply.body[:300])
        self.assertEqual(openai.hits[-1].json()['max_output_tokens'], default)
        # the translated path obeys the same rule: Ollama chat to an OpenAI target gets the cap, Ollama embed does not
        reply = S.call(self.app, 'POST', '/api/chat', json_body={'model': GPT, 'messages': CHAT, 'stream': False},
                       principal=self.budgeted)
        self.assertEqual(reply.status, 200, reply.body[:300])
        self.assertEqual(reply.header('x-titanos-translation'), 'ollama->openai')
        self.assertEqual(openai.hits[-1].json()['max_tokens'], default)
        reply = S.call(self.app, 'POST', '/api/embed', json_body={'model': 'text-embedding-3-small@openai',
                                                                  'input': 'hello'}, principal=self.budgeted)
        self.assertEqual(reply.status, 200, reply.body[:300])
        sent = openai.hits[-1].json()
        self.assertEqual(sent['model'], 'text-embedding-3-small')
        self.assertFalse({'max_tokens', 'max_completion_tokens', 'max_output_tokens'} & set(sent), sent)
        self.assertIsNone(reply.header('x-titanos-injected'))

    def test_the_limiter_and_inject_helper_never_compute_a_cap_off_token_surfaces(self):
        from titanos.broker import inbound
        from titanos.broker.router import estimate_for
        from titanos.broker.surfaces import SURFACES, TOKEN_OUTPUT_SURFACES
        from titanos.broker.types import BodySummary, Hints
        from titanos.broker.types import RouteContext, Target
        target = Target(ref='dall-e-2@openai', klass='cloud', upstream_id='openai', upstream_model='dall-e-2')
        for name, surface in SURFACES.items():
            if surface.kind != 'intercept' or surface.capability == 'catalog':
                continue
            with self.subTest(name):
                ctx = RouteContext(surface=surface, requested_model='dall-e-2@openai', principal=self.budgeted,
                                   hints=Hints(), body=BodySummary())
                estimate = estimate_for(ctx)
                self.assertEqual(estimate.token_output, name in TOKEN_OUTPUT_SURFACES)
                if name not in TOKEN_OUTPUT_SURFACES:
                    admission = self.broker.limiter.admit(self.budgeted, target, estimate, price=Decimal('0.01'))
                    self.assertIsNone(admission.injected_max_tokens)
                    admission.release()
                    self.assertEqual(inbound.inject_output_cap(name, {'model': 'm'}, 4096), {'model': 'm'})
        self.assertEqual(inbound.inject_output_cap('openai.chat', {'model': 'm'}, 64), {'model': 'm', 'max_tokens': 64})
        self.assertEqual(inbound.inject_output_cap('gemini.generateContent', {}, 64),
                         {'generationConfig': {'maxOutputTokens': 64}})
        self.assertEqual(inbound.inject_output_cap('ollama.chat', {}, 64), {'options': {'num_predict': 64}})


class ProviderModelGone(Case):
    """A provider 404 or 410 for a model (the NVIDIA API catalog answers 410 Gone for retired models) is 404
    model_not_found in every inbound envelope, origin upstream, never a breaker failure, never a credential mark and
    never retried."""

    def setUp(self):
        super().setUp()
        doc = document()
        doc['upstreams']['openai']['models'] = ['gpt-4o-mini', 'gpt-4o-mini-tts', 'text-embedding-3-small']
        doc['voice_maps'] = {'elevenlabs_to_openai': {'voice1': 'alloy'}}
        self.lab.close()
        self.lab = S.Lab(doc, local=LOCAL)
        self.broker = self.lab.broker
        self.app = BrokerDispatch(self.local, self.broker)

    def requests(self):
        chat = {'model': GPT, 'messages': CHAT}
        gemini = {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]}
        return [
            ('openai', 'POST', '/v1/chat/completions', dict(chat, stream=False), None),
            ('openai', 'POST', '/v1/chat/completions', dict(chat, stream=True), None),
            ('openai', 'POST', '/v1/embeddings', {'model': 'text-embedding-3-small@openai', 'input': 'x'}, None),
            ('responses', 'POST', '/v1/responses', {'model': GPT, 'input': 'hi', 'store': False}, None),
            ('anthropic', 'POST', '/v1/messages', dict(chat, max_tokens=16), None),
            ('gemini', 'POST', f'/v1beta/models/{GPT}:generateContent', gemini, None),
            ('ollama', 'POST', '/api/chat', dict(chat, stream=False), None),
            ('elevenlabs', 'POST', '/v1/text-to-speech/voice1', {'text': 'hello', 'model_id': 'gpt-4o-mini-tts@openai'},
             'output_format=mp3_44100_128'),
        ]

    def check_envelope(self, protocol: str, document_: dict, status: int):
        word = 'retired (HTTP 410 Gone)' if status == 410 else 'not available (HTTP 404)'
        if protocol in ('openai', 'responses'):
            error = document_['error']
            self.assertEqual((error['code'], error['type']), ('model_not_found', 'model_not_found'))
            message = error['message']
        elif protocol == 'anthropic':
            self.assertEqual((document_['type'], document_['error']['type']), ('error', 'not_found_error'))
            message = document_['error']['message']
        elif protocol == 'gemini':
            self.assertEqual((document_['error']['code'], document_['error']['status']), (404, 'NOT_FOUND'))
            message = document_['error']['message']
        elif protocol == 'elevenlabs':
            self.assertEqual(document_['detail']['status'], 'model_not_found')
            message = document_['detail']['message']
        else:
            self.assertEqual(set(document_), {'error'})
            message = document_['error']
        self.assertIn('The provider reports', message)
        self.assertIn(word, message)
        self.assertIn('Function not found for account', message)  # the provider's own message follows

    def test_404_and_410_are_model_not_found_in_every_envelope(self):
        openai = MOCKS['openai']
        validation = self.lab.vault.record('oa').validation
        self.assertFalse(validation.startswith('failed'))
        for status in (404, 410):
            for protocol, method, path, body, query in self.requests():
                with self.subTest(status=status, protocol=protocol, path=path, stream=body.get('stream')):
                    openai.faults.push('status', status=status, message='Function not found for account')
                    before = len(openai.hits)
                    reply = S.call(self.app, method, path, json_body=body, principal=CLOUD, query=query or '')
                    self.assertEqual(reply.status, 404, reply.body[:300])
                    self.assertEqual(len(openai.hits), before + 1, 'never retried')
                    self.assertEqual(reply.header('x-titanos-error-origin'), 'upstream')
                    self.check_envelope(protocol, reply.json(), status)
                    decision = self.broker.decisions.query(limit=1, principal=S.BOOTSTRAP)[0]
                    self.assertEqual(decision['attempts'][-1]['error_code'], 'model_not_found')
                    self.assertIn(f'upstream_model_unavailable:{status}', decision['notes'])
        breaker = self.broker.breakers.get('openai', 'oa').snapshot()
        self.assertEqual((breaker['breaker'], breaker['consecutive_failures']), ('closed', 0))
        self.assertNotIn('server_error', breaker['last_outcomes'])
        self.assertEqual(self.lab.vault.record('oa').validation, validation)
        # the upstream still serves: nothing was taken offline
        self.assertEqual(self.chat(GPT).status, 200)

    def test_404_about_another_resource_stays_not_found(self):
        """A provider 404 that is not about the model (a voice, a file, a response id, a bare 404) keeps 404 not_found
        with the provider's own message: never rewritten to model_not_found, never noted as a model outage."""
        openai = MOCKS['openai']
        chat = {'model': GPT, 'messages': CHAT, 'stream': False}
        for code, message in (('voice_not_found', 'A voice with the voice_id voice1 was not found.'),
                              (None, 'No such File object: file-abc123'),
                              (None, "Response with id 'resp_abc123' not found.")):
            with self.subTest(code=code, message=message):
                openai.faults.push('status', status=404, message=message, code=code)
                reply = S.call(self.app, 'POST', '/v1/chat/completions', json_body=chat, principal=CLOUD)
                self.assertEqual(reply.status, 404, reply.body[:300])
                error = reply.json()['error']
                self.assertEqual(error['type'], 'not_found')
                self.assertEqual(error['code'], code or 'invalid_request_error')  # the provider's own code or type
                self.assertEqual(error['message'], message)
                decision = self.broker.decisions.query(limit=1, principal=S.BOOTSTRAP)[0]
                self.assertEqual(decision['attempts'][-1]['error_code'], 'not_found')
                self.assertFalse([n for n in decision['notes'] if n.startswith('upstream_model_unavailable')])
        for code, message in (('model_not_found', 'The model `gpt-4o-mini` does not exist or you do not have '
                                                  'access to it.'),
                              (None, 'gpt-4o-mini is not served here')):
            with self.subTest(code=code, message=message):
                openai.faults.push('status', status=404, message=message, code=code)
                reply = S.call(self.app, 'POST', '/v1/chat/completions', json_body=chat, principal=CLOUD)
                self.assertEqual((reply.status, reply.json()['error']['type']), (404, 'model_not_found'))
                decision = self.broker.decisions.query(limit=1, principal=S.BOOTSTRAP)[0]
                self.assertIn('upstream_model_unavailable:404', decision['notes'])

    def test_handle_bound_404_stays_not_found(self):
        from titanos.broker.translate.upstream_errors import UpstreamFailure, model_failure
        from titanos.broker.types import UpstreamError
        gone = UpstreamFailure(UpstreamError(kind='client_error', status=410, provider_code=None, message='Gone'))
        self.assertEqual((gone.code, gone.status), ('not_found', 404))
        mapped = model_failure(gone, 'm@u')
        self.assertEqual((mapped.code, mapped.status), ('model_not_found', 404))
        self.assertEqual(str(mapped), "The provider reports model 'm@u' is retired (HTTP 410 Gone): Gone")
        other = UpstreamFailure(UpstreamError(kind='client_error', status=400, provider_code=None, message='bad'))
        self.assertIs(model_failure(other, 'm@u'), other)


class TranscriptionMock(OpenAIMock):
    """The WP4 OpenAI mock whose JSON transcription answer carries `usage` (None: the plain fixture answer)."""
    usage = None

    async def transcriptions(self, request):
        if self.usage is None:
            return await super().transcriptions(request)
        await self.form_fields(request)
        from broker_mocks.server import json_reply
        return json_reply({'text': 'Hello world.', 'usage': self.usage})


class TranscriptionPricingBases(Case):
    """Live lane follow-up: gpt-4o-mini-transcribe reported tokens (29 in, 9 out) and no seconds on a row priced only
    per audio minute; the ledger charged USD 0.0141644, the reservation for the streamed TTS WAV's 141644 bytes at the
    1000 B/s floor (141.644 s at 0.006/min). Settlement now prices by tokens when the row has token prices, by the
    provider's seconds or the measured upload otherwise, and keeps the reservation only when nothing can be priced,
    recording the basis every time."""

    @classmethod
    def setUpClass(cls):
        cls.stt = TranscriptionMock()
        cls.stt.start()

    @classmethod
    def tearDownClass(cls):
        cls.stt.stop()

    def setUp(self):
        super().setUp()
        self.stt.reset()
        self.stt.usage = None

    def lab_with(self, *rows, model='gpt-4o-mini-transcribe'):
        doc = {'policy': {'cloud_policy': 'fallback'},
               'upstreams': {'stt': S.upstream(self.stt, 'openai', credential='st', models=[model])},
               'prices': list(rows)}
        self.lab.close()
        self.lab = S.Lab(doc, local=LOCAL)
        self.broker = self.lab.broker
        self.app = BrokerDispatch(self.local, self.broker)
        return f'{model}@stt'

    def transcribe(self, ref, audio, principal, fmt='json', filename='probe.wav'):
        body, ctype = S.multipart([('model', ref), ('file', (filename, audio, 'audio/wav')),
                                   ('response_format', fmt)])
        reply = S.call(self.app, 'POST', '/v1/audio/transcriptions', body=body, content_type=ctype,
                       principal=principal)
        self.assertEqual(reply.status, 200, reply.body[:300])
        decision = self.broker.decisions.query(limit=1, principal=S.BOOTSTRAP)[0]
        rows = [json.loads(line) for path in sorted(self.broker.usage.directory.glob('*.jsonl'))
                for line in path.read_text('utf-8').splitlines() if line.strip()]
        return decision, rows[-1]

    def test_the_live_lane_case_prices_the_measured_upload(self):
        ref = self.lab_with({'match': 'gpt-4o-mini-transcribe@stt', 'per_audio_minute': '0.006'})
        self.stt.usage = {'type': 'tokens', 'input_tokens': 29, 'output_tokens': 9,
                          'input_token_details': {'text_tokens': 0, 'audio_tokens': 29}}
        audio = wav_bytes(2.95, streamed=True)  # a streamed WAV from the speech probe: 0xFFFFFFFF sizes
        self.assertEqual(len(audio), 141644)
        key = S.key('inference', 'cloud', budget={'usd_per_day': '0.50'})
        decision, row = self.transcribe(ref, audio, key)
        expected = Decimal('2.95') * Decimal('0.006') / 60
        self.assertEqual(Decimal(decision['cost_usd']), expected)
        self.assertNotEqual(decision['cost_usd'], '0.0141644')
        self.assertEqual(decision['price_basis'], 'audio_seconds:upload')
        self.assertNotIn('cost:reserved_estimate', decision['notes'])
        self.assertEqual((row['cost_usd'], row['price_basis']), (decision['cost_usd'], 'audio_seconds:upload'))
        self.assertEqual((row['input_tokens'], row['output_tokens']), (29, 9))
        self.assertEqual(Decimal(self.broker.limiter.remaining(key)['usd_today']), Decimal('0.50') - expected)

    def test_token_usage_on_a_token_priced_row_is_priced_by_tokens(self):
        ref = self.lab_with({'match': 'gpt-4o-mini-transcribe@stt', 'input_per_mtok': '3', 'output_per_mtok': '5',
                             'per_audio_minute': '0.003'})
        self.stt.usage = {'type': 'tokens', 'input_tokens': 29, 'output_tokens': 9}
        key = S.key('inference', 'cloud', budget={'usd_per_day': '0.50'})
        decision, row = self.transcribe(ref, wav_bytes(2.95, streamed=True), key)
        expected = (Decimal(29) * 3 + Decimal(9) * 5) / 1_000_000
        self.assertEqual((Decimal(decision['cost_usd']), decision['price_basis']), (expected, 'tokens'))
        self.assertEqual((Decimal(row['cost_usd']), row['price_basis']), (expected, 'tokens'))

    def test_provider_seconds_win_over_the_upload(self):
        ref = self.lab_with({'match': 'gpt-4o-mini-transcribe@stt', 'per_audio_minute': '0.006'})
        self.stt.usage = {'type': 'duration', 'seconds': 2}
        decision, row = self.transcribe(ref, wav_bytes(2.95), S.key('inference', 'cloud', budget={'usd_per_day': '1'}))
        self.assertEqual((Decimal(decision['cost_usd']), decision['price_basis']),
                         (Decimal('0.0002'), 'audio_seconds:provider'))
        self.assertEqual(row['audio_seconds'], 2)

    def test_nothing_priceable_keeps_the_reservation_fail_closed(self):
        # no usage and an upload whose duration cannot be measured (not a WAV, FLAC or MP3): the reservation (30 s at
        # the 1000 B/s floor) stands, says so, and is never USD 0 for a budgeted key
        ref = self.lab_with({'match': 'gpt-4o-mini-transcribe@stt', 'per_audio_minute': '0.60'})
        key = S.key('inference', 'cloud', budget={'usd_per_day': '5'})
        decision, row = self.transcribe(ref, b'\x00' * 30000, key, fmt='text', filename='probe.ogg')
        self.assertEqual((decision['cost_usd'], decision['price_basis']), ('0.3', 'reservation'))
        self.assertIn('cost:reserved_estimate', decision['notes'])
        self.assertEqual((row['cost_usd'], row['price_basis']), ('0.3', 'reservation'))
        self.assertEqual(Decimal(self.broker.limiter.remaining(key)['usd_today']), Decimal('4.70'))
        # a measurable upload and no usage: priced by the upload, not by the reservation
        decision, row = self.transcribe(ref, wav_bytes(2.0, rate=8000), key, fmt='text')
        self.assertEqual((decision['cost_usd'], decision['price_basis']), ('0.02', 'audio_seconds:upload'))
        # a token-only row and no usage: nothing priceable, the token reservation stands (2 s -> 50 in, 20 out)
        ref = self.lab_with({'match': 'gpt-4o-mini-transcribe@stt', 'input_per_mtok': '3', 'output_per_mtok': '5'})
        key = S.key('inference', 'cloud', budget={'usd_per_day': '5'})
        decision, row = self.transcribe(ref, wav_bytes(2.0, rate=8000), key, fmt='text')
        self.assertEqual((Decimal(decision['cost_usd']), decision['price_basis']),
                         ((Decimal(50) * 3 + Decimal(20) * 5) / 1_000_000, 'reservation'))
        # an unbudgeted key has no reservation: the cost is unknown and recorded as such, never a guessed estimate
        decision, row = self.transcribe(ref, wav_bytes(2.0, rate=8000), CLOUD, fmt='text')
        self.assertEqual((decision['cost_usd'], decision['price_basis']), (None, 'unmeasured'))


if __name__ == '__main__':
    unittest.main(verbosity=1)
