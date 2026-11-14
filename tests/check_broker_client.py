"""WP8 Python client: every new Client method against the integrated in-process app (BrokerDispatch, the admin routes)
with loopback provider mocks, plus the header and stream contracts on an httpx.MockTransport.

- protocol methods: Anthropic messages (non-stream and SSE through _sse_events ending on message_stop) and
  count_tokens, OpenAI moderations, the handle-bound Responses methods (404 response_not_found for an unknown id),
  Gemini generate (non-stream and alt=sse, ending at the end of the stream), embed (single and batch) and countTokens,
  ElevenLabs speech (bytes and streamed chunks), speech_to_text and voices;
- broker read, upstream, credential, configuration, key and audit methods (the minted key secret is returned once in
  CreatedKey, whose repr hides it; rotate_key returns a new secret);
- default X-Titanos-Location / X-Titanos-Target headers, per-call _location/_target overrides that never reach the body,
  Client.provenance and last_provenance;
- set_credential (and rotate) refuse an http server URL that is not loopback before anything is sent;
- OpenAI streams through the unchanged _stream end on [DONE]; an Anthropic stream cut before message_stop raises.
Every test installs the netguard and asserts zero non-loopback connection attempts. Synthetic credentials only.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import AnthropicMock, ElevenLabsMock, GeminiMock, OpenAIMock, netguard  # noqa: E402
from titanos.client import Client, CreatedKey, TitanosError  # noqa: E402

BOOT = 'wp8-client-bootstrap-0001'
MOCKS: dict = {}
WAV = W.WAV


def setUpModule():
    MOCKS.update(openai=OpenAIMock(), anthropic=AnthropicMock(), gemini=GeminiMock(), eleven=ElevenLabsMock())
    for mock in MOCKS.values():
        mock.start()


def tearDownModule():
    for mock in MOCKS.values():
        mock.stop()


def document() -> dict:
    m = MOCKS
    return {
        'policy': {'cloud_policy': 'explicit'},
        'upstreams': {
            'openai': W.S.upstream(m['openai'], 'openai', credential='oa',
                                   models=['gpt-4o-mini', 'omni-moderation-latest']),
            'anthropic': W.S.upstream(m['anthropic'], 'anthropic', credential='an', models=['claude-sonnet-4-5']),
            'gem': W.S.upstream(m['gemini'], 'gemini', credential='ge', models=['gemini-2.5-flash', 'text-embedding-004']),
            'eleven': W.S.upstream(m['eleven'], 'elevenlabs', credential='el',
                                   models=['eleven_multilingual_v2', 'scribe_v1']),
        },
        'models': {
            'mini': {'target': 'gpt-4o-mini@openai', 'capabilities': ['chat']},
            'mod': {'target': 'omni-moderation-latest@openai', 'capabilities': ['moderation']},
            'sonnet': {'target': 'claude-sonnet-4-5@anthropic', 'capabilities': ['chat']},
            'flash': {'target': 'gemini-2.5-flash@gem', 'capabilities': ['chat']},
            'gembed': {'target': 'text-embedding-004@gem', 'capabilities': ['embeddings']},
            'voice2': {'target': 'eleven_multilingual_v2@eleven', 'capabilities': ['speech']},
            'scribe': {'target': 'scribe_v1@eleven', 'capabilities': ['transcription']},
        },
    }


class AppTransport(httpx.BaseTransport):
    """Sends each Client request through the app's TestClient (loopback peer, lifespan running) and hands back the
    buffered answer; streams are therefore read whole, which still exercises their framing end to end."""

    def __init__(self, test_client):
        self.test_client = test_client

    def handle_request(self, request):
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ('host', 'content-length')}
        answer = self.test_client.request(request.method, str(request.url), headers=headers, content=request.read())
        headers = [(k, v) for k, v in answer.headers.multi_items() if k.lower() not in ('content-encoding', 'transfer-encoding')]
        return httpx.Response(answer.status_code, headers=headers, content=answer.content)


class Base(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        for mock in MOCKS.values():
            mock.reset()
        self.full = W.FullApp(document(), models={'chat-local': 'chat', 'voice': 'speech'}, api_key=BOOT,
                              creds={'gem': 'gemini', 'eleven': 'elevenlabs'})
        self.addCleanup(self.full.close)
        self.client = Client('http://127.0.0.1:8000', api_key=BOOT, timeout=30, transport=AppTransport(self.full.http))
        self.addCleanup(self.client.close)

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')


class ProtocolMethods(Base):
    def test_anthropic_messages_and_count_tokens(self):
        msg = [{'role': 'user', 'content': 'Say hello.'}]
        answer = self.client.messages('sonnet', msg, max_tokens=64)
        self.assertEqual(answer['type'], 'message')
        self.assertTrue(self.client.last_provenance['x-titanos-served-by'].startswith('cloud:anthropic'))
        events = list(self.client.messages('sonnet', msg, max_tokens=64, stream=True))
        self.assertEqual(events[0]['type'], 'message_start')
        self.assertEqual(events[-1]['type'], 'message_stop')
        counted = self.client.count_tokens('sonnet', msg)
        self.assertIsInstance(counted['input_tokens'], int)

    def test_moderations_and_response_handles(self):
        result = self.client.moderations('mod', 'hello')
        self.assertIn('results', result)
        for call in (self.client.response, self.client.delete_response, self.client.cancel_response,
                     self.client.response_input_items):
            with self.assertRaises(TitanosError) as caught:
                call('resp_unknown0001')
            self.assertEqual(caught.exception.status_code, 404)
            self.assertEqual(caught.exception.code, 'response_not_found')

    def test_gemini_methods(self):
        contents = [{'role': 'user', 'parts': [{'text': 'Say hello.'}]}]
        answer = self.client.gemini_generate('flash', contents)
        self.assertIn('candidates', answer)
        chunks = list(self.client.gemini_generate('flash', contents, stream=True))
        self.assertTrue(chunks and all('candidates' in c or 'usageMetadata' in c for c in chunks))
        self.assertIn('embedding', self.client.gemini_embed('gembed', 'hello'))
        self.assertEqual(len(self.client.gemini_embed('gembed', ['a', 'b'], batch=True)['embeddings']), 2)
        self.assertIn('totalTokens', self.client.gemini_count_tokens('flash', contents))

    def test_elevenlabs_methods(self):
        audio = self.client.elevenlabs_speech('voice-a', 'Hello.', model_id='voice2')
        self.assertIsInstance(audio, bytes)
        self.assertTrue(audio)
        streamed = b''.join(self.client.elevenlabs_speech('voice-a', 'Hello.', model_id='voice2', stream=True))
        self.assertTrue(streamed)
        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / 'clip.wav'
            clip.write_bytes(WAV)
            text = self.client.speech_to_text('scribe', clip)
        self.assertIn('text', text)
        voices = self.client.voices()
        self.assertTrue(any(v['labels']['model_id'] == 'voice' for v in voices['voices']))

    def test_openai_stream_ends_on_done(self):
        events = list(self.client.chat('mini', [{'role': 'user', 'content': 'hi'}], stream=True))
        self.assertTrue(events)
        self.assertEqual(self.client.last_provenance['x-titanos-served-model'], 'gpt-4o-mini@openai')


class AdminMethods(Base):
    def test_broker_reads(self):
        self.assertEqual(self.client.broker_status()['upstreams']['total'], 4)
        self.assertTrue(self.client.broker_profiles()['data'])
        catalog = self.client.broker_catalog()
        self.assertIn('mini', {row['id'] for row in catalog['data']})
        decision = self.client.broker_explain('openai.chat', {'model': 'mini', 'messages': [{'role': 'user', 'content': 'x'}]})
        self.assertTrue(decision['dry_run'])
        self.assertIsInstance(self.client.broker_decisions(limit=5)['data'], list)
        self.assertIsInstance(self.client.broker_usage()['rows'], list)
        self.assertIn('ttfb_ms_buckets', self.client.broker_metrics())
        self.assertEqual(self.client.broker_reload()['status'], 'reloaded')

    def test_upstreams_and_credentials(self):
        self.assertEqual({row['id'] for row in self.client.upstreams()['data']}, {'openai', 'anthropic', 'gem', 'eleven'})
        row = self.client.upstream('openai')
        self.assertEqual(row['profile'], 'openai')
        self.assertIn('breaker', self.client.upstream_health('openai'))
        self.assertTrue(self.client.test_upstream('openai')['ok'])
        self.assertEqual(self.client.reset_breaker('openai')['id'], 'openai')
        created = self.client.put_upstream('spare', profile='openai', base_url=MOCKS['openai'].api_base)
        self.assertEqual(created['id'], 'spare')
        sentinel = W.SENTINELS['openai']
        meta = self.client.set_credential('spare-key', upstream='spare', secret=sentinel)
        self.assertEqual(meta['validation'], 'ok')
        self.assertNotIn(sentinel, json.dumps(meta))
        self.assertNotIn(sentinel, json.dumps(vars(self.client), default=str))
        listed = self.client.credentials()
        self.assertNotIn(sentinel, json.dumps(listed))
        self.assertEqual(self.client.validate_credential('spare-key')['validation'], 'ok')
        self.assertEqual(self.client.rotate_credential('spare-key', secret=sentinel)['validation'], 'ok')
        self.assertTrue(self.client.delete_credential('spare-key')['deleted'])
        self.assertTrue(self.client.delete_upstream('spare')['deleted'])
        with self.assertRaises(ValueError):
            self.client.set_credential('x', upstream='spare', secret=sentinel, env='VAR')

    def test_configuration_methods(self):
        self.assertIn('mini', {row['name'] for row in self.client.aliases()['data']})
        self.assertEqual(self.client.put_alias('mini2', 'gpt-4o-mini@openai', capabilities=('chat',))['name'], 'mini2')
        self.assertTrue(self.client.delete_alias('mini2')['deleted'])
        vm = self.client.put_virtual_model('auto-chat', capability='chat', targets=[{'ref': 'chat-local'}, {'ref': 'mini'}])
        self.assertEqual(vm['name'], 'auto-chat')
        self.assertIn('auto-chat', {row['name'] for row in self.client.virtual_models()['data']})
        self.assertTrue(self.client.delete_virtual_model('auto-chat')['deleted'])
        before = self.client.policy()
        after = self.client.set_policy(default_location='prefer_local')
        self.assertEqual(after['default_location'], 'prefer_local')
        self.assertEqual(after['cloud_policy'], before['cloud_policy'])
        advisor = self.client.put_advisor('adv', profile='generic-v1', url='http://127.0.0.1:9/v1/advise')
        self.assertEqual(advisor['id'], 'adv')
        self.assertIn('adv', {row['id'] for row in self.client.advisors()['data']})
        self.assertIn('ok', self.client.test_advisor('adv'))
        rows = [{'match': 'gpt-4o-mini@openai', 'input_per_mtok': '0.15', 'output_per_mtok': '0.60'}]
        self.assertEqual(self.client.put_prices(rows)['data'][0]['match'], 'gpt-4o-mini@openai')
        self.assertEqual(len(self.client.prices()['data']), 1)

    def test_keys_and_audit(self):
        created = self.client.create_key(name='app', scopes=['inference'])
        self.assertIsInstance(created, CreatedKey)
        self.assertTrue(created.secret.startswith('tk_' + created.id))
        self.assertNotIn(created.secret, repr(created))
        self.assertNotIn('secret', created.record)
        self.assertEqual(self.client.key(created.id)['name'], 'app')
        self.assertIn(created.id, {row['id'] for row in self.client.keys()['data']})
        self.assertEqual(self.client.update_key(created.id, note='ci')['note'], 'ci')
        rotated = self.client.rotate_key(created.id)
        self.assertNotEqual(rotated.secret, created.secret)
        with Client('http://127.0.0.1:8000', api_key=rotated.secret, transport=AppTransport(self.full.http)) as scoped:
            self.assertEqual(scoped.whoami()['id'], created.id)
        self.assertEqual(self.client.revoke_key(created.id)['status'], 'revoked')
        self.assertEqual(self.client.whoami()['kind'], 'bootstrap')
        self.assertTrue(self.client.audit(kind='key.mint')['data'])
        self.assertTrue(self.client.audit_verify()['ok'])


class HeaderContract(unittest.TestCase):
    def setUp(self):
        self.seen = []

        def handler(request):
            self.seen.append(request)
            if request.url.path == '/v1/messages':
                frames = [{'type': 'message_start', 'message': {'id': 'm'}}, {'type': 'content_block_delta'}]
                body = ''.join(f'event: {f["type"]}\ndata: {json.dumps(f)}\n\n' for f in frames)
                return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=body)
            if request.url.path == '/v1/chat/completions':
                body = 'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: [DONE]\n\n'
                return httpx.Response(200, headers={'content-type': 'text/event-stream', 'X-Titanos-Served-By': 'local:engine'},
                                      text=body)
            return httpx.Response(200, json={'ok': True}, headers={'X-Titanos-Served-By': 'cloud:openai',
                                                                   'X-Titanos-Fallback-Count': '1', 'X-Request-ID': 'r1',
                                                                   'Content-Type': 'application/json'})
        self.transport = httpx.MockTransport(handler)

    def test_default_and_per_call_hints(self):
        with Client('http://127.0.0.1:1', api_key='', transport=self.transport, location='local_only', target='m@u') as client:
            client.embeddings('e', 'x')
            client.embeddings('e', 'x', _location='cloud_only', _target='other@u')
            list(client.chat('c', [], stream=True, _location='auto'))
            provenance = client.last_provenance
        first, second, third = self.seen
        self.assertEqual(first.headers['x-titanos-location'], 'local_only')
        self.assertEqual(first.headers['x-titanos-target'], 'm@u')
        self.assertEqual(second.headers['x-titanos-location'], 'cloud_only')
        self.assertEqual(second.headers['x-titanos-target'], 'other@u')
        self.assertNotIn('_location', json.loads(second.content))
        self.assertNotIn('_target', json.loads(second.content))
        self.assertEqual(third.headers['x-titanos-location'], 'auto')
        self.assertNotIn('_location', json.loads(third.content))
        self.assertEqual(provenance, {'x-titanos-served-by': 'local:engine'})
        with self.assertRaises(ValueError):
            Client('http://127.0.0.1:1', transport=self.transport, location='anywhere')

    def test_provenance_helper(self):
        response = httpx.Response(200, headers={'X-Titanos-Served-By': 'cloud:openai', 'X-Request-ID': 'r1', 'Server': 'x'})
        self.assertEqual(Client.provenance(response), {'x-titanos-served-by': 'cloud:openai', 'x-request-id': 'r1'})
        with Client('http://127.0.0.1:1', transport=self.transport) as client:
            client.broker_status()
            self.assertEqual(client.last_provenance['x-titanos-fallback-count'], '1')

    def test_credential_writes_refuse_insecure_transport(self):
        for url in ('http://192.0.2.10:11435', 'http://titanos.example:11435'):
            with Client(url, transport=self.transport) as client:
                with self.assertRaises(ValueError):
                    client.set_credential('k', upstream='u', secret='sk-test-0123456789abcdef')
                with self.assertRaises(ValueError):
                    client.rotate_credential('k', env='VAR')
        self.assertEqual(self.seen, [])
        for url in ('https://titanos.example', 'http://127.0.0.1:1', 'http://[::1]:1', 'http://localhost:1'):
            with Client(url, transport=self.transport) as client:
                client.set_credential('k', upstream='u', env='VAR')
        self.assertEqual(len(self.seen), 4)
        self.assertEqual(json.loads(self.seen[0].content), {'upstream': 'u', 'env': 'VAR', 'purpose': 'inference',
                                                            'validate': True})

    def test_anthropic_stream_without_message_stop_raises(self):
        with Client('http://127.0.0.1:1', transport=self.transport) as client:
            with self.assertRaises(TitanosError):
                list(client.messages('m', [], max_tokens=1, stream=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
