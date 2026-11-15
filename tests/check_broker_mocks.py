"""Broker loopback mocks (WP4): every mock refuses a missing or wrong credential with its provider's native envelope; the hit
recorder never keeps a credential value; every fault kind behaves as tests/broker_mocks/faults.py specifies; the gate holds
frame N; MockServer starts and stops deterministically. The netguard refuses 192.0.2.1 and 198.51.100.1 (and the metadata,
link-local, unspecified and IPv4-mapped forms) on every connect path with the guard installed: sync httpx, async httpx on the
default event loop (Proactor on Windows), raw socket.create_connection, socket.connect/connect_ex, loop.sock_connect and
loop.create_connection with IP literals (Proactor and selector loops), and getaddrinfo for names; a subprocess started with
netguard_site/sitecustomize.py and TITANOS_TEST_NETGUARD=1 is blocked too and writes its counter file.

Synthetic credentials only; every socket is loopback (the guard is installed for the whole process before any mock starts).
"""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from broker_mocks import (AdvisorMock, AnthropicMock, AzureMock, ElevenLabsMock, GeminiMock, MockServer,  # noqa: E402
                          OllamaCloudMock, OpenAIMock, RouterUpstreamMock, SENTINELS, fingerprint, netguard, parse_sse)
from broker_mocks.faults import KINDS  # noqa: E402

import httpx  # noqa: E402

netguard.install()
WIN = sys.platform == 'win32'
BLOCKED = ('192.0.2.1', '198.51.100.1')
WRONG = SENTINELS['wrong']
CHAT = {'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'Hello!'}]}
AN_BODY = {'model': 'claude-sonnet-4-5', 'max_tokens': 64, 'messages': [{'role': 'user', 'content': 'Hello!'}]}
GE_BODY = {'contents': [{'role': 'user', 'parts': [{'text': 'Hello!'}]}]}


def client():
    return httpx.Client(timeout=10.0, trust_env=False)


def bearer(name):
    return {'authorization': 'Bearer ' + SENTINELS[name]}


AN_OK = {'x-api-key': SENTINELS['anthropic'], 'anthropic-version': '2023-06-01'}


class Mocks(unittest.TestCase):
    """One instance of every mock for the class (started once, reset per test)."""

    @classmethod
    def setUpClass(cls):
        cls.openai = OpenAIMock()
        cls.azure = AzureMock()
        cls.anthropic = AnthropicMock()
        cls.gemini = GeminiMock()
        cls.eleven = ElevenLabsMock()
        cls.ollama = OllamaCloudMock()
        cls.openrouter = RouterUpstreamMock('openrouter')
        cls.hivemind = RouterUpstreamMock('hivemind')
        cls.xai = OpenAIMock(variant='xai')
        cls.mistral = OpenAIMock(variant='mistral')
        cls.generic = AdvisorMock('generic-v1')
        cls.gim = AdvisorMock('gim')
        cls.all = [cls.openai, cls.azure, cls.anthropic, cls.gemini, cls.eleven, cls.ollama, cls.openrouter, cls.hivemind,
                   cls.xai, cls.mistral, cls.generic, cls.gim]
        for mock in cls.all:
            mock.start()
        cls.http = client()

    @classmethod
    def tearDownClass(cls):
        cls.http.close()
        for mock in cls.all:
            mock.stop()

    def setUp(self):
        for mock in self.all:
            mock.reset()


class AuthStyles(Mocks):
    def assert_no_raw(self, mock, value):
        for hit in mock.hits:
            self.assertNotIn(value, json.dumps(hit.headers) + hit.query + repr(hit.credentials))

    def test_openai_bearer(self):
        url = self.openai.api_base + '/chat/completions'
        missing = self.http.post(url, json=CHAT)
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(set(missing.json()['error']), {'message', 'type', 'param', 'code'})
        self.assertIsNone(missing.json()['error']['code'])
        wrong = self.http.post(url, json=CHAT, headers={'authorization': 'Bearer ' + WRONG})
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(wrong.json()['error']['code'], 'invalid_api_key')
        self.assertNotIn(WRONG, wrong.text)  # masked echo
        self.assertEqual(self.http.post(url, json=CHAT, headers={'x-api-key': SENTINELS['openai']}).status_code, 401)
        self.assertEqual(self.http.post(url, json=CHAT, headers=bearer('openai')).status_code, 200)
        self.assert_no_raw(self.openai, WRONG)

    def test_azure_api_key_deployment_and_version(self):
        base = self.azure.api_base + '/deployments/chat-dep/chat/completions'
        q = {'api-version': '2024-10-21'}
        ok = {'api-key': SENTINELS['azure_openai']}
        missing = self.http.post(base, params=q, json=CHAT)
        self.assertEqual((missing.status_code, missing.json()['error']['code']), (401, '401'))
        self.assertIn('missing subscription key', missing.json()['error']['message'])
        wrong = self.http.post(base, params=q, json=CHAT, headers={'api-key': WRONG})
        self.assertEqual((wrong.status_code, wrong.json()['error']['code']), (401, '401'))
        bearer_only = self.http.post(base, params=q, json=CHAT, headers={'authorization': 'Bearer ' + SENTINELS['azure_openai']})
        self.assertEqual(bearer_only.status_code, 401)
        no_version = self.http.post(base, json=CHAT, headers=ok)
        self.assertEqual((no_version.status_code, no_version.json()['error']['code']), (404, '404'))
        unknown = self.http.post(self.azure.api_base + '/deployments/nope/chat/completions', params=q, json=CHAT, headers=ok)
        self.assertEqual(unknown.json()['error']['code'], 'DeploymentNotFound')
        good = self.http.post(base, params=q, json={'messages': CHAT['messages']}, headers=ok)
        self.assertEqual(good.status_code, 200)
        self.assertEqual(good.json()['model'], 'gpt-4o-mini')
        self.assertIn('prompt_filter_results', good.json())
        self.assertEqual(self.azure.hits[-1].query_params.get('api-version'), '2024-10-21')

    def test_anthropic_key_version_beta_and_max_tokens(self):
        url = self.anthropic.api_base + '/v1/messages'
        missing = self.http.post(url, json=AN_BODY, headers={'anthropic-version': '2023-06-01'})
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.json()['type'], 'error')
        self.assertEqual(missing.json()['error']['type'], 'authentication_error')
        wrong = self.http.post(url, json=AN_BODY, headers={'x-api-key': WRONG, 'anthropic-version': '2023-06-01'})
        self.assertEqual((wrong.status_code, wrong.json()['error']['message']), (401, 'invalid x-api-key'))
        no_version = self.http.post(url, json=AN_BODY, headers={'x-api-key': SENTINELS['anthropic']})
        self.assertEqual((no_version.status_code, no_version.json()['error']['type']), (400, 'invalid_request_error'))
        no_max = self.http.post(url, json={k: v for k, v in AN_BODY.items() if k != 'max_tokens'}, headers=AN_OK)
        self.assertEqual((no_max.status_code, no_max.json()['error']['message']), (400, 'max_tokens: Field required'))
        ok = self.http.post(url, json=AN_BODY, headers=dict(AN_OK, **{'anthropic-beta': 'structured-outputs-2025-11-13'}))
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.headers['x-mock-anthropic-beta'], 'structured-outputs-2025-11-13')
        self.assertEqual(self.anthropic.hits[-1].headers['anthropic-beta'], 'structured-outputs-2025-11-13')
        self.assertEqual(self.anthropic.hits[-1].headers['x-api-key'], '<sentinel:anthropic>')

    def test_gemini_goog_header_and_query_key_refused(self):
        url = self.gemini.api_base + '/models/gemini-2.5-flash:generateContent'
        missing = self.http.post(url, json=GE_BODY)
        self.assertEqual((missing.status_code, missing.json()['error']['status']), (403, 'PERMISSION_DENIED'))
        wrong = self.http.post(url, json=GE_BODY, headers={'x-goog-api-key': WRONG})
        self.assertEqual((wrong.status_code, wrong.json()['error']['status']), (400, 'INVALID_ARGUMENT'))
        self.assertEqual(wrong.json()['error']['details'][0]['reason'], 'API_KEY_INVALID')
        query = self.http.post(url, params={'key': SENTINELS['gemini']}, json=GE_BODY,
                               headers={'x-goog-api-key': SENTINELS['gemini']})
        self.assertEqual((query.status_code, query.json()['error']['status']), (401, 'UNAUTHENTICATED'))
        hit = self.gemini.hits[-1]
        self.assertIn('%3Csentinel%3Agemini%3E', hit.query)
        self.assertNotIn(SENTINELS['gemini'], hit.query)
        self.assertIn(('query:key', '<sentinel:gemini>'), hit.credentials)
        self.assertEqual(self.http.post(url, json=GE_BODY, headers={'x-goog-api-key': SENTINELS['gemini']}).status_code, 200)

    def test_elevenlabs_xi_key_and_output_format(self):
        url = self.eleven.api_base + '/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM'
        body = {'text': 'Hello world.', 'model_id': 'eleven_multilingual_v2'}
        missing = self.http.post(url, json=body)
        self.assertEqual((missing.status_code, missing.json()['detail']['status']), (401, 'needs_authorization'))
        wrong = self.http.post(url, json=body, headers={'xi-api-key': WRONG})
        self.assertEqual((wrong.status_code, wrong.json()['detail']['status']), (401, 'invalid_api_key'))
        ok = {'xi-api-key': SENTINELS['elevenlabs']}
        bad = self.http.post(url, params={'output_format': 'mp3_1_1'}, json=body, headers=ok)
        self.assertEqual((bad.status_code, bad.json()['detail']['status']), (400, 'invalid_output_format'))
        default = self.http.post(url, json=body, headers=ok)
        self.assertEqual(default.headers['content-type'], 'audio/mpeg')
        self.assertEqual(default.headers['character-cost'], str(len(body['text'])))
        pcm = self.http.post(url + '/stream', params={'output_format': 'pcm_24000'}, json=body, headers=ok)
        self.assertEqual(pcm.headers['content-type'], 'audio/pcm')
        self.assertEqual(self.eleven.hits[-1].query_params['output_format'], 'pcm_24000')
        self.assertEqual(self.eleven.hits[-1].surface, 'elevenlabs.tts_stream')

    def test_ollama_bearer(self):
        url = self.ollama.api_base + '/api/chat'
        body = {'model': 'gpt-oss:120b', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': False}
        for headers in ({}, {'authorization': 'Bearer ' + WRONG}):
            reply = self.http.post(url, json=body, headers=headers)
            self.assertEqual((reply.status_code, reply.json()), (401, {'error': 'unauthorized'}))
        streamed = self.http.post(url, json=dict(body, stream=True), headers=bearer('ollama_cloud'))
        self.assertEqual(streamed.headers['content-type'], 'application/x-ndjson')
        lines = [json.loads(line) for line in streamed.text.splitlines()]
        self.assertTrue(lines[-1]['done'])
        default = self.http.post(url, json={k: v for k, v in body.items() if k != 'stream'}, headers=bearer('ollama_cloud'))
        self.assertEqual(default.headers['content-type'], 'application/x-ndjson')  # Ollama streams by default

    def test_router_mocks(self):
        url = self.openrouter.api_base + '/chat/completions'
        self.assertEqual(self.openrouter.api_base[-7:], '/api/v1')
        missing = self.http.post(url, json=CHAT)
        self.assertEqual(missing.json(), {'error': {'message': 'No auth credentials found', 'code': 401}})
        self.assertEqual(self.http.post(url, json=CHAT, headers={'authorization': 'Bearer ' + WRONG}).status_code, 401)
        ok = self.http.post(url, json=CHAT, headers=bearer('openrouter'))
        self.assertEqual(ok.json()['provider'], 'MockProvider')
        hive = self.hivemind.api_base + '/chat/completions'
        self.assertEqual(self.http.post(hive, json=CHAT, headers={'x-titanos-hop': 'abcd1234abcd1234'}).status_code, 200)
        self.assertEqual(self.http.post(hive, json=CHAT, headers={'authorization': 'Bearer ' + WRONG}).status_code, 401)
        self.assertEqual(self.http.get(self.hivemind.base_url + '/v1/health/ready').json(), {'status': 'ready'})
        self.assertEqual(self.hivemind.hops, ['abcd1234abcd1234'])

    def test_mistral_and_xai_refusals(self):
        mistral = self.mistral.api_base + '/chat/completions'
        body = dict(CHAT, model='mistral-small-latest')
        for headers in ({}, {'authorization': 'Bearer ' + WRONG}, {'x-api-key': SENTINELS['mistral']},
                        {'authorization': 'Bearer ' + SENTINELS['openai']}):
            reply = self.http.post(mistral, json=body, headers=headers)
            self.assertEqual(reply.status_code, 401, headers.keys())
            self.assertEqual(set(reply.json()), {'message', 'request_id'})  # Mistral's own envelope, no 'error' object
            self.assertEqual(reply.json()['message'], 'Unauthorized')
            self.assertNotIn(WRONG, reply.text)
        self.assertEqual(self.http.post(mistral, json=body, headers=bearer('mistral')).status_code, 200)
        xai = self.xai.api_base + '/chat/completions'
        body = dict(CHAT, model='grok-3-mini')
        missing = self.http.post(xai, json=body)
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(set(missing.json()['error']), {'message', 'type', 'param', 'code'})
        self.assertIsNone(missing.json()['error']['code'])
        for token in (WRONG, SENTINELS['openai']):
            wrong = self.http.post(xai, json=body, headers={'authorization': 'Bearer ' + token})
            self.assertEqual((wrong.status_code, wrong.json()['error']['code']), (401, 'invalid_api_key'))
            self.assertNotIn(token, wrong.text)  # masked echo
        self.assertEqual(self.http.post(xai, json=body, headers=bearer('xai')).status_code, 200)
        for mock in (self.mistral, self.xai):
            self.assert_no_raw(mock, WRONG)

    def test_advisor_mocks(self):
        request = {'contract': 'titanos.router-advisor/v1', 'candidates': [{'ref': 'a'}, {'ref': 'b@x'}, {'ref': 'c@y'}]}
        self.assertEqual(self.http.post(self.generic.url, json=request, headers={'authorization': 'Bearer ' + WRONG})
                         .status_code, 401)
        answer = self.http.post(self.generic.url, json=request).json()
        self.assertEqual((answer['choice'], answer['fallbacks']), ('a', ['b@x', 'c@y']))
        self.generic.choice = 'c@y'
        self.assertEqual(self.http.post(self.generic.url, json=request).json()['choice'], 'c@y')
        self.assertEqual(self.http.post(self.generic.url, json=dict(request, contract='other')).status_code, 400)
        self.generic.answer = {'contract': 'titanos.router-advisor/v1', 'choice': 'not-a-candidate', 'fallbacks': []}
        self.assertEqual(self.http.post(self.generic.url, json=request).json()['choice'], 'not-a-candidate')
        gim_url = self.gim.url + '/gim/llm_router/func_1'
        result = self.http.post(gim_url, json={'messages': [], 'available_models': ['LLM_MID', 'LLM_BIG']}).json()
        self.assertTrue(result['ok'])
        self.assertEqual(set(result['result']), {'recommended_model', 'reasoning_mode', 'confidence', 'route_reason', 'scores',
                                                 'fallback_chain', 'classification', 'latency_ms'})
        self.assertEqual(result['result']['recommended_model'], 'LLM_BIG')
        self.assertEqual(result['result']['fallback_chain'], ['LLM_MID'])
        self.gim.recommend = 'LLM_MID'
        self.assertEqual(self.http.post(gim_url, json={'available_models': ['LLM_MID', 'LLM_BIG']}).json()['result']
                         ['recommended_model'], 'LLM_MID')
        bad = self.http.post(gim_url, content=b'{not json', headers={'content-type': 'application/json'})
        self.assertEqual((bad.status_code, bad.json()), (400, {'ok': False, 'error': 'Invalid JSON'}))


class Recorder(Mocks):
    def test_credentials_are_recorded_as_sentinel_names_or_fingerprints(self):
        url = self.openai.api_base + '/embeddings'
        body = {'model': 'text-embedding-3-small', 'input': 'x'}
        self.http.post(url, json=body, headers=bearer('openai'))
        foreign = 'tk_abcdefgh23_' + 'Q' * 43
        self.http.post(url, json=body, headers={'authorization': 'Bearer ' + foreign})
        first, second = self.openai.hits
        self.assertEqual(first.headers['authorization'], 'Bearer <sentinel:openai>')
        self.assertEqual(first.credential, 'Bearer <sentinel:openai>')
        self.assertEqual(second.headers['authorization'], f'Bearer <foreign:{fingerprint(foreign)}>')
        self.assertTrue(second.presented(foreign))
        self.assertFalse(first.presented(foreign))
        for hit in (first, second):
            self.assertNotIn(foreign, repr(hit))
            self.assertNotIn(SENTINELS['openai'], repr(hit.headers) + repr(hit.credentials))
        raw = json.dumps(body).encode()
        self.assertEqual(first.body_sha256, hashlib.sha256(first.body).hexdigest())
        self.assertEqual(first.json(), body)
        self.assertEqual((first.method, first.path, first.surface, first.status), ('POST', '/v1/embeddings',
                                                                                    'openai.embeddings', 200))
        self.assertTrue(raw)

    def test_body_leaks_are_visible_to_presented(self):
        self.http.post(self.openai.api_base + '/embeddings', headers=bearer('openai'),
                       json={'model': 'm', 'input': 'leak ' + SENTINELS['leak']})
        self.assertTrue(self.openai.hits[-1].presented(SENTINELS['leak']))

    def test_duplicate_credential_headers_are_kept_and_refused(self):
        client_token = 'tk_abcdefgh23_' + 'D' * 43  # the shape of an inbound Titanos key; synthetic
        url = self.openai.api_base + '/chat/completions'
        forwarded = httpx.Headers([('authorization', 'Bearer ' + client_token),
                                   ('authorization', 'Bearer ' + SENTINELS['openai'])])
        reply = self.http.post(url, json=CHAT, headers=forwarded)
        self.assertEqual(reply.status_code, 400)  # never authenticated by the appended upstream key
        self.assertEqual(set(reply.json()['error']), {'message', 'type', 'param', 'code'})
        self.assertNotIn(client_token, reply.text)
        hit = self.openai.hits[-1]
        self.assertEqual(hit.duplicate_credentials, ['authorization'])
        self.assertEqual(hit.credentials, [('header:authorization', f'Bearer <foreign:{fingerprint(client_token)}>'),
                                           ('header:authorization', 'Bearer <sentinel:openai>')])
        self.assertEqual(hit.headers['authorization'],
                         f'Bearer <foreign:{fingerprint(client_token)}>, Bearer <sentinel:openai>')
        self.assertTrue(hit.presented(client_token))
        self.assertTrue(hit.presented(SENTINELS['openai']))
        self.assertNotIn(client_token, repr(hit))
        self.assertNotIn(SENTINELS['openai'], repr(hit.headers) + repr(hit.header_list) + repr(hit.credentials))
        anthropic = self.http.post(self.anthropic.api_base + '/v1/messages', json=AN_BODY, headers=httpx.Headers(
            [('x-api-key', client_token), ('x-api-key', SENTINELS['anthropic']), ('anthropic-version', '2023-06-01')]))
        self.assertEqual((anthropic.status_code, anthropic.json()['type']), (400, 'error'))
        self.assertEqual(self.anthropic.hits[-1].duplicate_credentials, ['x-api-key'])
        self.assertTrue(self.anthropic.hits[-1].presented(client_token))
        joined = self.http.post(url, json=CHAT,
                                headers={'authorization': f'Bearer {client_token}, Bearer {SENTINELS["openai"]}'})
        self.assertEqual(joined.status_code, 401)  # one comma-joined header: a wrong key, not a duplicate
        self.assertTrue(self.openai.hits[-1].presented(client_token))
        self.assertEqual(self.openai.hits[-1].duplicate_credentials, [])
        self.http.post(url, json=CHAT, headers=bearer('openai'))
        self.assertEqual((self.openai.hits[-1].status, self.openai.hits[-1].duplicate_credentials), (200, []))

    def test_presented_searches_every_header_query_value_and_path(self):
        client_token = 'tk_abcdefgh23_' + 'F' * 43
        url = self.openai.api_base + '/embeddings'
        body = {'model': 'text-embedding-3-small', 'input': 'x'}
        self.http.post(url, json=body, headers=dict(bearer('openai'), **{'x-forwarded-client-key': client_token}))
        self.http.post(url, params={'token': client_token}, json=body, headers=bearer('openai'))
        self.http.post(url, json=body, headers=dict(bearer('openai'), cookie='session=' + client_token + '; theme=1'))
        self.http.post(self.openai.api_base + '/files/' + client_token, headers=bearer('openai'))
        self.http.post(url, json=body, headers=bearer('openai'))
        in_header, in_query, in_cookie, in_path, clean = self.openai.hits[-5:]
        for hit in (in_header, in_query, in_cookie, in_path):
            self.assertTrue(hit.presented(client_token), hit.path)
        self.assertNotIn(client_token, repr(in_cookie))  # cookie is a credential header: kept only as a fingerprint
        self.assertFalse(clean.presented(client_token))
        self.assertFalse(in_header.presented('tk_abcdefgh23_' + 'G' * 43))
        self.assertFalse(clean.presented(''))

    def test_import_needs_base_tier_only_and_changes_nothing(self):
        code = ('import logging, socket, sys\n'
                'before = (socket.socket.connect, socket.getaddrinfo, list(logging.getLogger("uvicorn.error").filters))\n'
                'import broker_mocks\n'
                'after = (socket.socket.connect, socket.getaddrinfo, list(logging.getLogger("uvicorn.error").filters))\n'
                'heavy = sorted(m for m in ("numpy", "PIL", "torch", "openai", "anthropic") if m in sys.modules)\n'
                'print(before == after, heavy, "uvicorn" in sys.modules)\n')
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'tests'), PYTHONDONTWRITEBYTECODE='1')
        env.pop('TITANOS_TEST_NETGUARD', None)
        proc = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), 'True [] False')  # uvicorn loads only when a server starts

    def test_server_lifecycle_is_deterministic(self):
        mock = OpenAIMock()
        url = mock.start()
        self.assertTrue(url.startswith('http://127.0.0.1:'))
        self.assertTrue(mock.server.running)
        self.assertEqual(mock.start(), url)  # idempotent
        mock.stop()
        self.assertFalse(mock.server.running)
        mock.stop()  # idempotent
        with self.assertRaises(ValueError):
            MockServer(object(), host='0.0.0.0')


class Faults(Mocks):
    def stream(self, url, body, headers):
        return self.http.post(url, json=dict(body, stream=True), headers=headers)

    def test_every_kind_is_specified(self):
        self.assertEqual(KINDS, {'status', 'connect_refused', 'slow_first_byte', 'error_first_event',
                                 'disconnect_after_frames', 'malformed_json', 'malformed_sse_line', 'oversized_body',
                                 'redirect_foreign', 'set_cookie', 'hold_frame'})
        with self.assertRaises(ValueError):
            self.openai.faults.push('teapot')
        with self.assertRaises(ValueError):
            self.openai.faults.push('status')

    def test_status_codes_in_native_envelopes(self):
        url = self.openai.api_base + '/chat/completions'
        for status in (429, 500, 502, 503, 529, 401, 403):
            self.openai.faults.push('status', status=status)
            reply = self.http.post(url, json=CHAT, headers=bearer('openai'))
            self.assertEqual(reply.status_code, status)
            self.assertIn('error', reply.json())
            self.assertNotIn(SENTINELS['leak'], reply.text)
        self.openai.faults.push('status', status=429, retry_after=11)
        self.assertEqual(self.http.post(url, json=CHAT, headers=bearer('openai')).headers['retry-after'], '11')
        self.openai.faults.push('status', status=429)
        reply = self.http.post(url, json=CHAT, headers=bearer('openai'))
        self.assertEqual((reply.headers['retry-after'], reply.json()['error']['type']), ('7', 'requests'))
        self.anthropic.faults.push('status', status=529)
        overloaded = self.http.post(self.anthropic.api_base + '/v1/messages', json=AN_BODY, headers=AN_OK)
        self.assertEqual((overloaded.status_code, overloaded.json()['error']['type']), (529, 'overloaded_error'))
        self.gemini.faults.push('status', status=429)
        exhausted = self.http.post(self.gemini.api_base + '/models/m:generateContent', json=GE_BODY,
                                   headers={'x-goog-api-key': SENTINELS['gemini']})
        self.assertEqual(exhausted.json()['error']['status'], 'RESOURCE_EXHAUSTED')

    def test_400_body_carries_the_secret_sentinels(self):
        self.openai.faults.push('status', status=400)
        reply = self.http.post(self.openai.api_base + '/chat/completions', json=CHAT, headers=bearer('openai'))
        self.assertEqual(reply.status_code, 400)
        self.assertIn(SENTINELS['leak'], reply.text)
        self.assertIn(SENTINELS['openai'], reply.text)  # echoed presented credential, for scrub tests
        self.assertEqual(self.openai.hits[-1].fault, 'status')

    def test_connect_refused_until_resume(self):
        url = self.ollama.api_base + '/api/tags'
        self.ollama.faults.push('connect_refused')
        with self.assertRaises(httpx.ConnectError):
            httpx.get(url, headers=bearer('ollama_cloud'), timeout=10.0, trust_env=False)
        self.ollama.faults.resume()
        self.assertEqual(self.http.get(url, headers=bearer('ollama_cloud')).status_code, 200)
        self.ollama.faults.push('connect_refused', seconds=0.3)
        self.assertTrue(_poll(lambda: _status(url, bearer('ollama_cloud')) == 200, 15.0))

    def test_slow_first_byte(self):
        self.openai.faults.push('slow_first_byte', delay=0.6)
        started = time.monotonic()
        reply = self.http.get(self.openai.api_base + '/models', headers=bearer('openai'))
        self.assertGreaterEqual(time.monotonic() - started, 0.55)
        self.assertEqual(reply.status_code, 200)

    def test_error_first_event_per_protocol(self):
        self.anthropic.faults.push('error_first_event')
        reply = self.stream(self.anthropic.api_base + '/v1/messages', AN_BODY, AN_OK)
        events = parse_sse(reply.content)
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][0], 'error')
        self.assertEqual(json.loads(events[0][1])['error']['type'], 'overloaded_error')
        self.openai.faults.push('error_first_event')
        first = parse_sse(self.stream(self.openai.api_base + '/chat/completions', CHAT, bearer('openai')).content)[0]
        self.assertIn('error', json.loads(first[1]))
        self.gemini.faults.push('error_first_event')
        reply = self.http.post(self.gemini.api_base + '/models/m:streamGenerateContent', params={'alt': 'sse'}, json=GE_BODY,
                               headers={'x-goog-api-key': SENTINELS['gemini']})
        self.assertEqual(json.loads(parse_sse(reply.content)[0][1])['error']['status'], 'UNAVAILABLE')
        self.ollama.faults.push('error_first_event')
        reply = self.http.post(self.ollama.api_base + '/api/chat', json={'model': 'm', 'messages': []},
                               headers=bearer('ollama_cloud'))
        self.assertIn('error', json.loads(reply.text.splitlines()[0]))

    def test_stream_faults_wait_for_a_streaming_answer(self):
        self.openai.faults.push('error_first_event')
        plain = self.http.post(self.openai.api_base + '/chat/completions', json=CHAT, headers=bearer('openai'))
        self.assertEqual(plain.json()['object'], 'chat.completion')
        self.assertEqual(len(self.openai.faults.pending()), 1)
        streamed = self.stream(self.openai.api_base + '/chat/completions', CHAT, bearer('openai'))
        self.assertIn('error', json.loads(parse_sse(streamed.content)[0][1]))
        self.assertEqual(self.openai.faults.pending(), [])

    def test_disconnect_after_frames(self):
        self.openai.faults.push('disconnect_after_frames', frames=2)
        got = []
        with self.assertRaises(httpx.RemoteProtocolError):
            with self.http.stream('POST', self.openai.api_base + '/chat/completions', json=dict(CHAT, stream=True),
                                  headers=bearer('openai')) as reply:
                for chunk in reply.iter_raw():
                    got.append(chunk)
        self.assertEqual(len(parse_sse(b''.join(got))), 2)
        self.assertTrue(_poll(lambda: self.openai.hits[-1].aborted, 5.0))
        self.assertEqual(self.openai.hits[-1].frames_sent, 2)

    def test_malformed_json_and_sse(self):
        self.openai.faults.push('malformed_json')
        reply = self.http.post(self.openai.api_base + '/chat/completions', json=CHAT, headers=bearer('openai'))
        self.assertEqual(reply.status_code, 200)
        with self.assertRaises(ValueError):
            reply.json()
        self.openai.faults.push('malformed_sse_line', at=2)
        frames = parse_sse(self.stream(self.openai.api_base + '/chat/completions', CHAT, bearer('openai')).content)
        json.loads(frames[0][1])
        with self.assertRaises(ValueError):
            json.loads(frames[1][1])

    def test_oversized_body(self):
        self.openai.faults.push('oversized_body', size=300_000)
        reply = self.http.get(self.openai.api_base + '/models', headers=bearer('openai'))
        self.assertGreaterEqual(len(reply.content), 300_000)
        self.assertNotIn('content-length', reply.headers)
        self.openai.faults.push('oversized_body', size=100_000, declared_length=True)
        reply = self.http.get(self.openai.api_base + '/models', headers=bearer('openai'))
        self.assertEqual(reply.headers['content-length'], str(len(reply.content)))

    def test_redirect_foreign_records_auth_at_the_target(self):
        self.openai.faults.push('redirect_foreign')
        url = self.openai.api_base + '/chat/completions'
        reply = self.http.post(url, json=CHAT, headers=bearer('openai'))
        self.assertEqual(reply.status_code, 307)
        location = reply.headers['location']
        self.assertTrue(location.startswith(self.openai.sink.base_url))
        self.assertEqual(self.openai.sink.hits, [])
        self.http.post(location, json=CHAT, headers=bearer('openai'))  # what a client that follows would send
        self.assertEqual(self.openai.sink.hits[-1].credential, 'Bearer <sentinel:openai>')
        self.openai.faults.push('redirect_foreign', target=self.ollama, status=302)
        reply = self.http.post(url, json=CHAT, headers=bearer('openai'))
        self.assertEqual(reply.status_code, 302)
        self.assertTrue(reply.headers['location'].startswith(self.ollama.base_url + '/v1/chat/completions'))

    def test_set_cookie(self):
        self.openai.faults.push('set_cookie')
        reply = self.http.get(self.openai.api_base + '/models', headers=bearer('openai'))
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(len(reply.headers.get_list('set-cookie')), 2)
        self.assertIn('location', reply.headers)

    def test_faults_match_path_and_count(self):
        self.openai.faults.push('status', status=503, path=r'/embeddings$', times=2)
        self.assertEqual(self.http.get(self.openai.api_base + '/models', headers=bearer('openai')).status_code, 200)
        body = {'model': 'm', 'input': 'x'}
        for _ in range(2):
            self.assertEqual(self.http.post(self.openai.api_base + '/embeddings', json=body,
                                            headers=bearer('openai')).status_code, 503)
        self.assertEqual(self.http.post(self.openai.api_base + '/embeddings', json=body, headers=bearer('openai')).status_code,
                         200)
        self.openai.faults.push('status', status=500, surface='openai.models')
        self.assertEqual(self.http.get(self.openai.api_base + '/models', headers=bearer('openai')).status_code, 500)

    def test_hold_frame_gate(self):
        chunks: queue.Queue = queue.Queue()
        self.anthropic.hold(frame=2)
        self.assertFalse(self.anthropic.gate.is_set())

        def read():
            with httpx.Client(timeout=30.0, trust_env=False) as c:
                with c.stream('POST', self.anthropic.api_base + '/v1/messages', json=dict(AN_BODY, stream=True),
                              headers=AN_OK) as reply:
                    for chunk in reply.iter_raw():
                        chunks.put(chunk)
            chunks.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        first = chunks.get(timeout=10)
        self.assertEqual(parse_sse(first)[0][0], 'message_start')
        with self.assertRaises(queue.Empty):
            chunks.get(timeout=0.4)
        self.assertEqual(self.anthropic.hits[-1].frames_sent, 1)
        self.anthropic.gate.set()
        rest = []
        while True:
            item = chunks.get(timeout=10)
            if item is None:
                break
            rest.append(item)
        reader.join(10)
        events = [name for name, _ in parse_sse(first + b''.join(rest))]
        self.assertEqual(events[0], 'message_start')
        self.assertEqual(events[-1], 'message_stop')
        deadline = time.monotonic() + 5  # completed is set after the mock's final send() returns
        while not self.anthropic.hits[-1].completed and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.anthropic.hits[-1].completed)

    def test_client_disconnect_is_recorded(self):
        self.openai.hold(frame=2)
        with httpx.Client(timeout=10.0, trust_env=False) as c:
            with c.stream('POST', self.openai.api_base + '/chat/completions', json=dict(CHAT, stream=True),
                          headers=bearer('openai')) as reply:
                next(reply.iter_raw())
        self.assertTrue(_poll(lambda: self.openai.hits[-1].disconnected, 10.0))
        self.assertEqual(self.openai.hits[-1].frames_sent, 1)


class Surfaces(Mocks):
    def test_responses_lifecycle_and_previous_response_id(self):
        base = self.openai.api_base + '/responses'
        h = bearer('openai')
        created = self.http.post(base, json={'model': 'gpt-4.1-mini', 'input': 'Hello!'}, headers=h).json()
        rid = created['id']
        self.assertEqual(self.http.get(f'{base}/{rid}', headers=h).json()['id'], rid)
        items = self.http.get(f'{base}/{rid}/input_items', headers=h).json()
        self.assertEqual(items['object'], 'list')
        self.assertEqual(items['data'][0]['content'][0]['text'], 'Hello!')
        follow = self.http.post(base, json={'model': 'gpt-4.1-mini', 'input': 'More', 'previous_response_id': rid},
                                headers=h).json()
        self.assertEqual(follow['previous_response_id'], rid)
        unknown = self.http.post(base, json={'model': 'gpt-4.1-mini', 'input': 'x', 'previous_response_id': 'resp_nope'},
                                 headers=h)
        self.assertEqual((unknown.status_code, unknown.json()['error']['code']), (400, 'previous_response_not_found'))
        self.assertEqual(self.http.post(f'{base}/{rid}/cancel', headers=h).status_code, 400)
        self.assertTrue(self.http.delete(f'{base}/{rid}', headers=h).json()['deleted'])
        self.assertEqual(self.http.get(f'{base}/{rid}', headers=h).status_code, 404)
        background = self.http.post(base, json={'model': 'gpt-4.1-mini', 'input': 'x', 'background': True}, headers=h).json()
        self.assertEqual(background['status'], 'in_progress')
        self.assertEqual(self.http.post(f"{base}/{background['id']}/cancel", headers=h).json()['status'], 'cancelled')
        unstored = self.http.post(base, json={'model': 'gpt-4.1-mini', 'input': 'x', 'store': False}, headers=h).json()
        self.assertFalse(unstored['store'])
        self.assertEqual(self.http.get(f"{base}/{unstored['id']}", headers=h).status_code, 404)

    def test_responses_stream_family(self):
        reply = self.http.post(self.openai.api_base + '/responses', json={'model': 'm', 'input': 'x', 'stream': True},
                               headers=bearer('openai'))
        events = parse_sse(reply.content)
        sequence = [json.loads(data)['sequence_number'] for _, data in events]
        self.assertEqual(sequence, list(range(len(events))))
        self.assertEqual(events[0][0], 'response.created')
        self.assertEqual(events[-1][0], 'response.completed')

    def test_openai_media_surfaces(self):
        h = bearer('openai')
        for fmt, ctype in (('mp3', 'audio/mpeg'), ('wav', 'audio/wav'), ('pcm', 'audio/pcm'), ('opus', 'audio/opus'),
                           ('aac', 'audio/aac'), ('flac', 'audio/flac')):
            reply = self.http.post(self.openai.api_base + '/audio/speech', headers=h,
                                   json={'model': 'tts-1', 'input': 'Hi', 'voice': 'alloy', 'response_format': fmt})
            self.assertEqual(reply.headers['content-type'], ctype)
            self.assertNotIn('content-length', reply.headers)  # chunked
        wav = self.http.post(self.openai.api_base + '/audio/speech', headers=h,
                             json={'model': 'tts-1', 'input': 'Hi', 'voice': 'alloy', 'response_format': 'wav'}).content
        self.assertEqual(wav[:4] + wav[8:12], b'RIFFWAVE')
        files = {'file': ('a.wav', wav, 'audio/wav')}
        text = self.http.post(self.openai.api_base + '/audio/transcriptions', headers=h, files=files,
                              data={'model': 'whisper-1', 'response_format': 'text'})
        self.assertEqual((text.headers['content-type'], text.text), ('text/plain; charset=utf-8', 'Hello world.\n'))
        verbose = self.http.post(self.openai.api_base + '/audio/translations', headers=h, files=files,
                                 data={'model': 'whisper-1', 'response_format': 'verbose_json'}).json()
        self.assertEqual(verbose['task'], 'translate')
        self.assertEqual(self.http.post(self.openai.api_base + '/audio/transcriptions', headers=h,
                                        data={'model': 'whisper-1'}).status_code, 400)
        refused = self.http.post(self.openai.api_base + '/images/generations', headers=h,
                                 json={'model': 'gpt-image-1', 'prompt': 'x', 'response_format': 'url'})
        self.assertEqual(refused.json()['error']['code'], 'unknown_parameter')
        png = base64.b64decode(self.http.post(self.openai.api_base + '/images/generations', headers=h,
                                              json={'model': 'gpt-image-1', 'prompt': 'x'}).json()['data'][0]['b64_json'])
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
        # the current OpenAI Images API (live 2026-10-02): dall-e refuses response_format too and answers b64_json
        for model in ('dall-e-2', 'dall-e-3', 'gpt-image-1'):
            for fmt in ('b64_json', 'url'):
                for path, kw in (('/images/generations', {'json': {'model': model, 'prompt': 'x', 'response_format': fmt}}),
                                 ('/images/edits', {'data': {'model': model, 'prompt': 'x', 'response_format': fmt},
                                                    'files': {'image': ('a.png', png, 'image/png')}})):
                    with self.subTest(model=model, response_format=fmt, path=path):
                        refused = self.http.post(self.openai.api_base + path, headers=h, **kw)
                        self.assertEqual(refused.status_code, 400)
                        self.assertEqual(refused.json()['error'], {
                            'message': "Unknown parameter: 'response_format'.", 'type': 'invalid_request_error',
                            'param': 'response_format', 'code': 'unknown_parameter'})
        b64 = self.http.post(self.openai.api_base + '/images/generations', headers=h,
                             json={'model': 'dall-e-3', 'prompt': 'x', 'n': 2}).json()
        self.assertEqual(len(b64['data']), 2)
        self.assertEqual(base64.b64decode(b64['data'][0]['b64_json'])[:8], b'\x89PNG\r\n\x1a\n')
        edit = self.http.post(self.openai.api_base + '/images/edits', headers=h, data={'model': 'dall-e-2', 'prompt': 'x'},
                              files={'image': ('a.png', png, 'image/png')}).json()
        self.assertEqual(base64.b64decode(edit['data'][0]['b64_json'])[:8], b'\x89PNG\r\n\x1a\n')
        with OpenAIMock(images_accept_response_format=True) as legacy:  # an upstream with the pre-2026-10 behaviour
            url = self.http.post(legacy.api_base + '/images/generations', headers=h,
                                 json={'model': 'dall-e-3', 'prompt': 'x', 'n': 2}).json()
            self.assertEqual(len(url['data']), 2)
            self.assertTrue(url['data'][0]['url'].startswith('https://example.invalid/'))
            legacy_b64 = self.http.post(legacy.api_base + '/images/generations', headers=h,
                                        json={'model': 'dall-e-3', 'prompt': 'x', 'response_format': 'b64_json'})
            self.assertIn('b64_json', legacy_b64.json()['data'][0])
        jpeg = base64.b64decode(self.http.post(self.xai.api_base + '/images/generations', headers=bearer('xai'),
                                               json={'model': 'grok-2-image', 'prompt': 'x'}).json()['data'][0]['b64_json'])
        self.assertEqual(jpeg[:3], b'\xff\xd8\xff')
        emb = self.http.post(self.openai.api_base + '/embeddings', headers=h,
                             json={'model': 'm', 'input': ['a', 'b', 'c'], 'encoding_format': 'base64'}).json()
        self.assertEqual([d['index'] for d in emb['data']], [0, 1, 2])
        self.assertIsInstance(emb['data'][0]['embedding'], str)

    def test_mistral_tool_ids(self):
        h = bearer('mistral')
        url = self.mistral.api_base + '/chat/completions'
        tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {'type': 'object'}}}]
        answer = self.http.post(url, json=dict(CHAT, tools=tools), headers=h).json()
        call_id = answer['choices'][0]['message']['tool_calls'][0]['id']
        self.assertRegex(call_id, r'^[A-Za-z0-9]{9}$')
        history = CHAT['messages'] + [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_mock0001abc', 'type': 'function',
                                                                 'function': {'name': 'f', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_mock0001abc', 'content': '{}'}]
        refused = self.http.post(url, json=dict(CHAT, messages=history), headers=h)
        self.assertEqual(refused.status_code, 422)
        self.assertEqual(refused.json()['object'], 'error')
        good = [dict(m) for m in history]
        good[1] = dict(good[1], tool_calls=[dict(good[1]['tool_calls'][0], id=call_id)])
        good[2] = dict(good[2], tool_call_id=call_id)
        self.assertEqual(self.http.post(url, json=dict(CHAT, messages=good), headers=h).status_code, 200)

    def test_gemini_array_stream_pagination_and_embeddings(self):
        h = {'x-goog-api-key': SENTINELS['gemini']}
        array = self.http.post(self.gemini.api_base + '/models/gemini-2.5-flash:streamGenerateContent', json=GE_BODY,
                               headers=h)
        self.assertIsInstance(array.json(), list)
        self.assertEqual(self.gemini.hits[-1].surface, 'gemini.streamGenerateContent')
        page, names = {'pageSize': '1'}, []
        while True:
            body = self.http.get(self.gemini.api_base + '/models', params=page, headers=h).json()
            names += [m['name'] for m in body['models']]
            if 'nextPageToken' not in body:
                break
            page = {'pageSize': '1', 'pageToken': body['nextPageToken']}
        self.assertEqual(len(names), 3)
        batch = self.http.post(self.gemini.api_base + '/models/e:batchEmbedContents', headers=h,
                               json={'requests': [{'content': {'parts': [{'text': str(i)}]}} for i in range(5)]}).json()
        self.assertEqual(len(batch['embeddings']), 5)
        self.assertEqual(self.http.get(self.gemini.api_base + '/models/gemini-2.5-pro', headers=h).json()['name'],
                         'models/gemini-2.5-pro')

    def test_gemini_crlf_sse_framing_option(self):
        h = {'x-goog-api-key': SENTINELS['gemini']}
        path = '/models/gemini-2.5-flash:streamGenerateContent'
        lf = self.http.post(self.gemini.api_base + path, params={'alt': 'sse'}, json=GE_BODY, headers=h).content
        self.assertNotIn(b'\r', lf)
        with GeminiMock(sse_crlf=True) as crlf_mock:
            crlf = self.http.post(crlf_mock.api_base + path, params={'alt': 'sse'}, json=GE_BODY, headers=h).content
            self.assertIn(b'\r\n\r\n', crlf)
            self.assertNotIn(b'\n', crlf.replace(b'\r\n', b''))  # every line ending is CRLF
            self.assertEqual(crlf, lf.replace(b'\n', b'\r\n'))
            self.assertEqual(parse_sse(crlf), parse_sse(lf))
            crlf_mock.faults.push('error_first_event')
            error = self.http.post(crlf_mock.api_base + path, params={'alt': 'sse'}, json=GE_BODY, headers=h).content
            self.assertTrue(error.endswith(b'\r\n\r\n'))
            self.assertEqual(json.loads(parse_sse(error)[0][1])['error']['status'], 'UNAVAILABLE')

    def test_elevenlabs_stt_and_catalogs(self):
        h = {'xi-api-key': SENTINELS['elevenlabs']}
        missing = self.http.post(self.eleven.api_base + '/v1/speech-to-text', headers=h,
                                 files={'file': ('a.wav', b'RIFF', 'audio/wav')})
        self.assertEqual((missing.status_code, missing.json()['detail'][0]['loc']), (422, ['body', 'model_id']))
        stt = self.http.post(self.eleven.api_base + '/v1/speech-to-text', headers=h, data={'model_id': 'scribe_v1'},
                             files={'file': ('a.wav', b'RIFF', 'audio/wav')}).json()
        self.assertEqual(set(stt), {'language_code', 'language_probability', 'text', 'words'})
        self.assertIsInstance(self.http.get(self.eleven.api_base + '/v1/models', headers=h).json(), list)
        self.assertEqual(self.http.get(self.eleven.api_base + '/v1/voices/nope', headers=h).json()['detail']['status'],
                         'voice_not_found')


def _status(url, headers):
    try:
        return httpx.get(url, headers=headers, timeout=10.0, trust_env=False).status_code
    except httpx.HTTPError:
        return None


def _poll(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class Netguard(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()

    def blocked(self, address, action, *, vias):
        before = netguard.blocked_count()
        with self.assertRaises((OSError, httpx.TransportError)):
            action()
        attempts = netguard.attempts()[before:]
        self.assertTrue(attempts, f'{address}: the guard recorded nothing')
        self.assertEqual(attempts[0]['host'], address)
        self.assertIn(attempts[0]['via'], vias)
        return attempts[0]['via']

    def test_installed_on_every_path(self):
        self.assertTrue(netguard.installed())
        self.assertTrue(getattr(socket.socket.connect, '__titanos_netguard__', False))
        self.assertTrue(getattr(socket.getaddrinfo, '__titanos_netguard__', False))
        for name in ('gethostbyname', 'gethostbyname_ex', 'gethostbyaddr'):
            self.assertTrue(getattr(getattr(socket, name), '__titanos_netguard__', False), name)
        self.assertTrue(getattr(socket.socket.sendto, '__titanos_netguard__', False))
        if WIN:
            from asyncio import windows_events
            self.assertTrue(getattr(windows_events.IocpProactor.connect, '__titanos_netguard__', False))
        netguard.install()  # idempotent

    def test_legacy_resolvers_and_udp_sendto(self):
        self.blocked('example.invalid', lambda: socket.gethostbyname('example.invalid'), vias={'gethostbyname'})
        self.blocked('example.invalid', lambda: socket.gethostbyname_ex('example.invalid'), vias={'gethostbyname_ex'})
        for address in BLOCKED:
            self.blocked(address, lambda: socket.gethostbyaddr(address), vias={'gethostbyaddr'})

            def udp():
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.sendto(b'x', (address, 9))
            self.assertEqual(self.blocked(address, udp, vias={'socket.sendto'}), 'socket.sendto')

            def udp_flags():
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.sendto(b'x', 0, (address, 9))
            self.assertEqual(self.blocked(address, udp_flags, vias={'socket.sendto'}), 'socket.sendto')
        self.assertEqual(socket.gethostbyname('127.0.0.1'), '127.0.0.1')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver, \
                socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            receiver.bind(('127.0.0.1', 0))
            receiver.settimeout(5.0)
            sender.sendto(b'loopback', receiver.getsockname())
            self.assertEqual(receiver.recv(64), b'loopback')

    def test_uninstall_restores_and_install_repatches(self):
        try:
            netguard.uninstall()
            self.assertFalse(netguard.installed())
            self.assertFalse(getattr(socket.socket.connect, '__titanos_netguard__', False))
            self.assertFalse(getattr(socket.getaddrinfo, '__titanos_netguard__', False))
            self.assertFalse(getattr(socket.gethostbyname, '__titanos_netguard__', False))
            self.assertFalse(getattr(socket.socket.sendto, '__titanos_netguard__', False))
        finally:
            netguard.install()  # no connect happens in between; the guard is back for the rest of the process
        self.assertTrue(getattr(socket.socket.connect_ex, '__titanos_netguard__', False))

    def test_sync_httpx(self):
        for address in BLOCKED:
            with httpx.Client(timeout=2.0, trust_env=False) as c:
                self.blocked(address, lambda: c.get(f'http://{address}:9/'), vias={'getaddrinfo', 'socket.connect'})

    def test_async_httpx_on_the_default_loop(self):
        kinds = []

        async def go(address):
            kinds.append(type(asyncio.get_running_loop()).__name__)
            async with httpx.AsyncClient(timeout=2.0, trust_env=False) as c:
                await c.get(f'http://{address}:9/')

        for address in BLOCKED:
            self.blocked(address, lambda: asyncio.run(go(address)),
                         vias={'getaddrinfo', 'socket.connect', 'IocpProactor.connect'})
        if WIN:
            self.assertEqual(set(kinds), {'ProactorEventLoop'})

    def test_raw_sockets(self):
        for address in BLOCKED:
            self.blocked(address, lambda: socket.create_connection((address, 9), timeout=1.0),
                         vias={'getaddrinfo', 'socket.connect'})

            def direct():
                with socket.socket() as s:
                    s.settimeout(1.0)
                    s.connect((address, 9))
            self.assertEqual(self.blocked(address, direct, vias={'socket.connect'}), 'socket.connect')

            def direct_ex():
                with socket.socket() as s:
                    s.settimeout(1.0)
                    s.connect_ex((address, 9))
            self.assertEqual(self.blocked(address, direct_ex, vias={'socket.connect_ex'}), 'socket.connect_ex')

    def test_loop_sock_connect_and_create_connection_with_ip_literals(self):
        async def sock_connect(address):
            loop = asyncio.get_running_loop()
            with socket.socket() as s:
                s.setblocking(False)
                await loop.sock_connect(s, (address, 9))

        async def create_connection(address):
            await asyncio.get_running_loop().create_connection(asyncio.Protocol, address, 9)

        default_via = 'IocpProactor.connect' if WIN else 'socket.connect'
        for address in BLOCKED:
            self.assertEqual(self.blocked(address, lambda: asyncio.run(sock_connect(address)), vias={default_via}),
                             default_via)
            self.assertEqual(self.blocked(address, lambda: asyncio.run(create_connection(address)), vias={default_via}),
                             default_via)
            selector = asyncio.SelectorEventLoop()
            try:
                self.assertEqual(self.blocked(address, lambda: selector.run_until_complete(sock_connect(address)),
                                              vias={'socket.connect'}), 'socket.connect')
            finally:
                selector.close()

    def test_names_are_refused_before_any_lookup(self):
        self.blocked('example.invalid', lambda: socket.getaddrinfo('example.invalid', 443), vias={'getaddrinfo'})
        self.assertTrue(socket.getaddrinfo('localhost', 80))
        self.assertTrue(socket.getaddrinfo('127.0.0.1', 80))

    def test_address_classes(self):
        for host in ('169.254.169.254', '::ffff:169.254.169.254', 'fd00:ec2::254', '0.0.0.0', '224.0.0.1', '192.0.2.1',
                     '198.51.100.1', '::', 'example.invalid', '127.0.0.1.example.invalid'):
            self.assertFalse(netguard.is_loopback_host(host), host)
        for host in ('127.0.0.1', '127.8.9.10', '::1', '[::1]', '::ffff:127.0.0.1', 'localhost', 'LOCALHOST'):
            self.assertTrue(netguard.is_loopback_host(host), host)

    def test_loopback_mocks_still_work(self):
        with OpenAIMock() as mock:
            reply = httpx.get(mock.api_base + '/models', headers=bearer('openai'), timeout=10.0, trust_env=False)
            self.assertEqual(reply.status_code, 200)

    def test_subprocess_with_sitecustomize_is_blocked(self):
        site = ROOT / 'tests' / 'broker_mocks' / 'netguard_site'
        code = ('import socket, sys\n'
                'flag = getattr(socket.socket.connect, "__titanos_netguard__", False)\n'
                'errors = []\n'
                'for target in ("192.0.2.1", "198.51.100.1"):\n'
                '    try:\n'
                '        socket.create_connection((target, 9), timeout=1)\n'
                '    except OSError as exc:\n'
                '        errors.append(type(exc).__name__)\n'
                '    s = socket.socket(); s.settimeout(1)\n'
                '    try:\n'
                '        s.connect((target, 9))\n'
                '    except OSError as exc:\n'
                '        errors.append(type(exc).__name__)\n'
                '    finally:\n'
                '        s.close()\n'
                'print(flag, ",".join(errors))\n')
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'netguard.jsonl'
            env = dict(os.environ, TITANOS_TEST_NETGUARD='1', TITANOS_TEST_NETGUARD_REPORT=str(report),
                       PYTHONPATH=os.pathsep.join([str(site)] + [p for p in [os.environ.get('PYTHONPATH')] if p]),
                       PYTHONDONTWRITEBYTECODE='1')
            proc = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            flag, errors = proc.stdout.strip().split(' ', 1)
            self.assertEqual(flag, 'True')
            self.assertEqual(errors.split(','), ['NetguardLookupBlocked', 'NetguardBlocked'] * 2)
            entries = [json.loads(line) for line in report.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([e['host'] for e in entries], ['192.0.2.1', '192.0.2.1', '198.51.100.1', '198.51.100.1'])
            self.assertEqual({e['via'] for e in entries}, {'getaddrinfo', 'socket.connect'})
            off = dict(env)
            off.pop('TITANOS_TEST_NETGUARD')
            probe = subprocess.run([sys.executable, '-B', '-c', 'import socket; print(getattr(socket.socket.connect, '
                                    '"__titanos_netguard__", False))'], env=off, capture_output=True, text=True, timeout=60)
            self.assertEqual(probe.stdout.strip(), 'False')  # the site hook is inert without the switch


if __name__ == '__main__':
    unittest.main(verbosity=1)
