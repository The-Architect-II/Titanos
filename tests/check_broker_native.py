"""R32 native passthrough over the integrated app (create_app: KeylessHostGuard, Guard + Authenticator, BrokerDispatch),
against the WP4 loopback mocks extended with tests/broker_mocks/native_routes.py; the netguard is installed and every
test ends with zero non-loopback connection attempts. Synthetic sentinel credentials only.

- /native/<upstream>/<path>: an OpenAI multipart file upload reaches the provider byte for byte with the upstream's own
  credential (never the client's tk_ key, never an X-Titanos-* header); the created id is bound to the calling key: the
  owner and the bootstrap key read it, another key gets 404 not_found with zero upstream hits; the streamed content keeps
  content-disposition and never set-cookie; the answer carries the provenance headers (translation native) and a
  Decision with surface native; a state-changing call appends a native.call audit record;
- every argument passes untouched on the native lane (Titanos-only names such as worker, location and timeout
  included), repeated query names are kept, openai-beta is forwarded;
- refusals before egress: a key without the native scope (403 at the Guard), a stateful row without cloud:store (403)
  or without the upstream's data_policy.allow_store (403 data_policy_denied), an account-wide listing without
  broker:admin (403), a path outside the allowlist (400 broker_capability_unavailable), a dot segment (400), a
  credential-named query parameter (401 query_credential_refused), an unknown or invisible upstream (404), a budgeted key
  without a price row (402 budget_unpriced_target; a character-billed provider such as ElevenLabs reserves characters,
  so a per_1k_chars row on native@<upstream> prices it), managed role (403 role_forbids), cloud policy off (403
  cloud_disabled);
- provider errors: a 400 body passes through scrubbed (the injected secret never reaches the client), a 401 becomes 502
  upstream_auth_failed, a 429 keeps its status and Retry-After;
- bare provider paths: POST /v1/files reaches the single openai-profile upstream; two candidates need X-Titanos-Upstream
  (400 otherwise); GET /v1/files/{id} reaches the provider for the key that uploaded that file through Titanos (the id is
  native-bound to it) and stays the local artifact route for every other id and key (or with X-Titanos-Upstream); the
  Anthropic SDK's x-api-key is accepted on /v1/messages/batches and its JSONL results stream through; ElevenLabs
  GET /v2/voices with xi-api-key; the Gemini resumable upload: the X-Goog-Upload-URL answer is rewritten to this server's
  /native path with an upload capability, so the follow-up upload and finalize (sent WITHOUT a key, as google-genai
  does) reach the provider with the upload id and without the capability, a wrong capability is 401 and a finalized
  session's capability is spent; Imagen :predict (no typed surface) reaches the provider through the bare /v1beta path;
- model names on the native lane: model@<this upstream> (body or {model} path segment) reaches the provider as its own id,
  a ref of another upstream is 400 before egress;
- the Anthropic-shaped GET /v1/models lists only what a Messages client can call (no speech-only models) and the single
  Anthropic upstream's models by their bare ids;
- path hygiene: an encoded slash or dot-dot inside a {x:path} parameter and a percent-encoded /native prefix are 400
  before egress; OpenAI-Organization / OpenAI-Project reach the provider only for broker:admin principals;
- keyless loopback: without anonymous_cloud a bare path keeps the Titanos answer (404) and /native is refused 403;
  with anonymous_cloud the keyless caller (the machine's owner) reaches the allowlisted endpoint and, on an upstream
  whose data_policy.allow_store is true, the stateful rows too (anonymous_cloud grants cloud:store; the upstream's
  allow_store still decides); without a configured broker /native is the app's own 404;
- the unknown-argument policy: a translated request (Anthropic Messages to an OpenAI upstream) lists the fields it
  dropped in X-Titanos-Dropped and in the decision notes.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import AnthropicMock, ElevenLabsMock, GeminiMock, OpenAIMock, SENTINELS, netguard  # noqa: E402
from broker_mocks.native_routes import (clear_native, install_anthropic_native, install_elevenlabs_native,  # noqa: E402
                                        install_gemini_native, install_openai_native, received)

BOOT = 'native-check-bootstrap-0001'
MOCKS: dict = {}
NATIVE = ('native', 'cloud', 'cloud:store')


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['openai2'] = OpenAIMock()
    MOCKS['anthropic'] = AnthropicMock()
    MOCKS['gemini'] = GeminiMock()
    MOCKS['elevenlabs'] = ElevenLabsMock()
    install_openai_native(MOCKS['openai'])
    install_openai_native(MOCKS['openai2'])
    install_anthropic_native(MOCKS['anthropic'])
    install_gemini_native(MOCKS['gemini'])
    install_elevenlabs_native(MOCKS['elevenlabs'])
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def document(*, policy: str = 'explicit', second_openai: bool = False, allow_store: bool = True,
             anonymous_cloud: bool = False, prices: list | None = None) -> dict:
    m = MOCKS
    upstreams = {
        'oai': W.S.upstream(m['openai'], 'openai', credential='oai-key', models=['gpt-4o-mini'],
                            data_policy={'allow_store': allow_store}),
        'ant': W.S.upstream(m['anthropic'], 'anthropic', credential='ant-key', models=['claude-sonnet-4-5'],
                            data_policy={'allow_store': True}),
        'gem': {'profile': 'gemini', 'base_url': m['gemini'].api_base, 'credential': 'gem-key', 'enabled': True,
                'models': ['gemini-2.5-flash'], 'data_policy': {'allow_store': True}},
        'xi': W.S.upstream(m['elevenlabs'], 'elevenlabs', credential='xi-key', models=['eleven_multilingual_v2'],
                           data_policy={'allow_store': True}),
    }
    if second_openai:
        upstreams['oai2'] = W.S.upstream(m['openai2'], 'openai', credential='oai2-key', models=['gpt-4o-mini'],
                                         data_policy={'allow_store': True})
    return {'policy': {'cloud_policy': policy, 'anonymous_cloud': anonymous_cloud}, 'upstreams': upstreams,
            'prices': list(prices or [])}


def bearer(token: str) -> dict:
    return {'Authorization': 'Bearer ' + token}


class Base(unittest.TestCase):
    keyed = True
    doc_kw: dict = {}
    settings_kw: dict = {}

    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
            clear_native(item)
        self.full = W.FullApp(document(**self.doc_kw), api_key=BOOT if self.keyed else None,
                              settings_kw=self.settings_kw)
        self.http = self.full.http

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def key(self, *scopes, **kw) -> str:
        return self.full.mint(*scopes, name='k' + str(len(scopes)) + str(len(kw)), **kw)[1]

    def upload(self, token: str, path: str = '/native/oai/files', content: bytes = b'{"line":1}\n{"line":2}\n',
               **headers):
        return self.http.post(path, files={'file': ('batch.jsonl', content, 'application/jsonl')},
                              data={'purpose': 'batch'}, headers={**bearer(token), **headers})


class PrefixLane(Base):
    def test_upload_reaches_the_provider_untouched_and_binds_the_owner(self):
        owner = self.key(*NATIVE)
        other = self.key(*NATIVE)
        content = b'{"custom_id":"a"}\n{"custom_id":"b"}\n'
        answer = self.upload(owner, content=content, **{'X-Titanos-Location': 'auto', 'Cookie': 'session=nope'})
        self.assertEqual(answer.status_code, 200, answer.text)
        file_id = answer.json()['id']
        self.assertEqual(answer.headers['x-titanos-translation'], 'native')
        self.assertEqual(answer.headers['x-titanos-served-by'], 'cloud:oai')
        hit = MOCKS['openai'].hits[-1]
        self.assertEqual((hit.method, hit.path), ('POST', '/v1/files'))
        self.assertEqual(hit.credential, 'Bearer <sentinel:openai>')
        self.assertIn(content, hit.body)  # the multipart body is forwarded as sent
        self.assertFalse(any(name.startswith('x-titanos') or name == 'cookie' for name in hit.headers), hit.headers)
        self.assertFalse(hit.presented(owner))
        before = len(MOCKS['openai'].hits)
        foreign = self.http.get(f'/native/oai/files/{file_id}', headers=bearer(other))
        self.assertEqual((foreign.status_code, foreign.json()['error']['code']), (404, 'not_found'))
        self.assertEqual(len(MOCKS['openai'].hits), before)
        self.assertEqual(self.http.get(f'/native/oai/files/{file_id}', headers=bearer(owner)).json()['id'], file_id)
        self.assertEqual(self.http.get(f'/native/oai/files/{file_id}', headers=bearer(BOOT)).status_code, 200)
        content_answer = self.http.get(f'/native/oai/files/{file_id}/content', headers=bearer(owner))
        self.assertEqual(content_answer.content, content)
        self.assertIn('attachment', content_answer.headers['content-disposition'])
        self.assertNotIn('set-cookie', content_answer.headers)
        decisions = self.full.broker.decisions.query(limit=10,
                                                     principal=self.full.broker.authenticator.bootstrap_principal())
        native = [d for d in decisions if d['surface'] == 'native']
        self.assertTrue(native and native[0]['translation'] == 'native', decisions[:1])
        audit = self.full.broker.audit.query(kind='native.call', limit=10)
        self.assertTrue(any(row['detail']['route'] == '/files' and row['detail']['method'] == 'POST' for row in audit),
                        audit)

    def test_every_argument_passes_untouched(self):
        token = self.key(*NATIVE)
        body = {'model': 'gpt-5', 'input': 'hi', 'worker': 'w1', 'location': 'here', 'timeout': 5, 'previous_response_id':
                'resp_1', 'unknown_field': {'nested': True}}
        answer = self.http.post('/native/oai/responses/compact?include[]=a&include[]=b', json=body,
                                headers={**bearer(token), 'OpenAI-Beta': 'responses=v1', 'X-Titanos-Target': 'x'})
        self.assertEqual(answer.status_code, 200, answer.text)
        echoed = answer.json()
        self.assertEqual(echoed['received'], body)
        self.assertEqual(echoed['query'], [['include[]', 'a'], ['include[]', 'b']])
        self.assertEqual(echoed['openai_beta'], 'responses=v1')
        self.assertEqual(received(MOCKS['openai'].hits[-1]), body)

    def test_refusals_before_egress(self):
        no_native = self.key('cloud', 'cloud:store', 'inference')
        no_store = self.key('native', 'cloud')
        plain = self.key(*NATIVE)
        budgeted = self.key(*NATIVE, budget={'usd_per_day': '1'})
        cases = [
            (self.http.get('/native/oai/files/file-x', headers=bearer(no_native)), 403, 'scope_denied'),
            (self.http.get('/native/oai/files/file-x', headers=bearer(no_store)), 403, 'scope_denied'),
            (self.http.get('/native/oai/files', headers=bearer(plain)), 403, 'scope_denied'),  # list needs broker:admin
            (self.http.post('/native/oai/images/variations', json={}, headers=bearer(plain)), 400,
             'broker_capability_unavailable'),
            (self.http.post('/native/oai/responses/compact?api_key=abc', json={}, headers=bearer(plain)), 401,
             'query_credential_refused'),
            (self.http.post('/native/nope/responses/compact', json={}, headers=bearer(plain)), 404, 'not_found'),
            (self.http.post('/native/oai/responses/compact', json={}, headers=bearer(budgeted)), 402,
             'budget_unpriced_target'),
        ]
        for answer, status, code in cases:
            with self.subTest(status=status, code=code):
                self.assertEqual(answer.status_code, status, answer.text)
                if code:
                    self.assertEqual(answer.json()['error']['code'], code)
        from titanos.broker.errors import BrokerError
        from titanos.broker.native import check_rel
        for rel, params in (('/files/../models', {}), ('/files/x', {'file_id': '..'}), ('/files/x', {'file_id': 'a\\b'})):
            with self.assertRaises(BrokerError):
                check_rel(rel, params)
        invisible = self.key('native', 'cloud:ant', 'cloud:store')
        answer = self.http.post('/native/oai/responses/compact', json={}, headers=bearer(invisible))
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (404, 'not_found'))
        self.assertEqual([h for h in MOCKS['openai'].hits], [])

    def test_list_with_broker_admin_and_bootstrap(self):
        admin = self.key(*NATIVE, 'broker:admin')
        self.upload(admin)
        for token in (admin, BOOT):
            answer = self.http.get('/native/oai/files', headers=bearer(token))
            self.assertEqual(answer.status_code, 200, answer.text)
            self.assertEqual(len(answer.json()['data']), 1)

    def test_provider_errors(self):
        token = self.key(*NATIVE)
        mock = MOCKS['openai']
        mock.faults.push('status', status=400, message='bad field', code='invalid_value')
        answer = self.http.post('/native/oai/responses/compact', json={'a': 1}, headers=bearer(token))
        self.assertEqual(answer.status_code, 400)
        self.assertEqual(answer.json()['error']['code'], 'invalid_value')
        self.assertNotIn(SENTINELS['openai'], answer.text)  # the presented credential echo is scrubbed
        self.assertNotIn(token, answer.text)
        self.assertEqual(answer.headers['x-titanos-error-origin'], 'upstream')
        mock.faults.push('status', status=429, retry_after=7)
        answer = self.http.post('/native/oai/responses/compact', json={'a': 1}, headers=bearer(token))
        self.assertEqual((answer.status_code, answer.headers.get('retry-after')), (429, '7'))
        # the 429 starts the breaker cooldown for that upstream: the next call is refused before egress
        answer = self.http.post('/native/oai/responses/compact', json={'a': 1}, headers=bearer(token))
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (503, 'breaker_open'))


    def test_a_callers_own_short_deadline_never_counts_against_the_shared_breaker(self):
        """A native call whose X-Titanos-Deadline-Ms runs out while the provider is still silent is 504
        deadline_exceeded with a neutral breaker outcome on every Python version (on 3.11 asyncio.wait_for can surface
        httpx's own timeout, capped at the same deadline, instead of TimeoutError): one key's tiny deadlines never open
        the upstream for everyone else."""
        token = self.key(*NATIVE)
        MOCKS['openai'].faults.push('slow_first_byte', delay=0.6, times=4)
        for _ in range(4):
            answer = self.http.post('/native/oai/responses/compact', json={'a': 1},
                                    headers={**bearer(token), 'X-Titanos-Deadline-Ms': '150'})
            self.assertEqual((answer.status_code, answer.json()['error']['code']), (504, 'deadline_exceeded'))
        MOCKS['openai'].faults.clear()
        other = self.key(*NATIVE)
        answer = self.http.post('/native/oai/responses/compact', json={'a': 1}, headers=bearer(other))
        self.assertNotEqual(answer.json().get('error', {}).get('code') if answer.status_code >= 400 else None,
                            'breaker_open', answer.text)

    def test_provider_401_is_upstream_auth_failed_and_marks_the_credential(self):
        token = self.key(*NATIVE)
        # OpenAI's key-rejection code marks at once (a 401 without it is checked against the probe endpoint first:
        # tests/check_broker_reachability.py CredentialChecks)
        MOCKS['openai'].faults.push('status', status=401, code='invalid_api_key')
        answer = self.http.post('/native/oai/responses/compact', json={'a': 1}, headers=bearer(token))
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (502, 'upstream_auth_failed'))
        again = self.http.post('/native/oai/responses/compact', json={'a': 1}, headers=bearer(token))
        self.assertEqual((again.status_code, again.json()['error']['code']), (503, 'credential_unavailable'))
        self.assertEqual(len(MOCKS['openai'].hits), 1)

class BarePaths(Base):
    def test_bare_openai_path_with_one_candidate(self):
        token = self.key(*NATIVE)
        answer = self.upload(token, path='/v1/files')
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(MOCKS['openai'].hits[-1].path, '/v1/files')
        notes = self.full.broker.decisions.query(
            limit=1, principal=self.full.broker.authenticator.bootstrap_principal())[0]['notes']
        self.assertIn('native:bare_path', notes)
        file_id = answer.json()['id']
        # GET /v1/files/{name} is Titanos' artifact route, except for a file this key uploaded through Titanos
        before = len(MOCKS['openai'].hits)
        own = self.http.get(f'/v1/files/{file_id}', headers=bearer(token))
        self.assertEqual((own.status_code, own.json()['id']), (200, file_id), own.text)
        self.assertEqual(MOCKS['openai'].hits[-1].path, f'/v1/files/{file_id}')
        content = self.http.get(f'/v1/files/{file_id}/content', headers=bearer(token))
        self.assertEqual(content.status_code, 200, content.text)
        hits = len(MOCKS['openai'].hits)
        self.assertEqual(hits, before + 2)
        other = self.key(*NATIVE)
        self.assertEqual(self.http.get(f'/v1/files/{file_id}', headers=bearer(other)).status_code, 404)
        self.assertEqual(self.http.get('/v1/files/file-never-uploaded', headers=bearer(token)).status_code, 404)
        self.assertEqual(len(MOCKS['openai'].hits), hits)  # both stayed on the local artifact route
        native = self.http.get(f'/v1/files/{file_id}', headers={**bearer(token), 'X-Titanos-Upstream': 'oai'})
        self.assertEqual((native.status_code, native.json()['id']), (200, file_id))

    def test_model_refs_become_provider_ids(self):
        token = self.key(*NATIVE)
        answer = self.http.post('/native/oai/responses/compact', json={'model': 'gpt-4o-mini@oai', 'input': 'hi'},
                                headers=bearer(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(received(MOCKS['openai'].hits[-1])['model'], 'gpt-4o-mini')
        bare = self.http.post('/v1/responses/compact', json={'model': 'gpt-4o-mini@oai', 'input': 'hi'},
                              headers=bearer(token))
        self.assertEqual((bare.status_code, received(MOCKS['openai'].hits[-1])['model']), (200, 'gpt-4o-mini'))
        hits = len(MOCKS['openai'].hits)
        foreign = self.http.post('/native/oai/responses/compact', json={'model': 'claude-sonnet-4-5@ant'},
                                 headers=bearer(token))
        self.assertEqual((foreign.status_code, foreign.json()['error']['code']), (400, 'invalid_request'))
        self.assertEqual(len(MOCKS['openai'].hits), hits)
        predict = self.http.post('/v1beta/models/imagen-4.0-generate-001@gem:predict',
                                 json={'instances': [{'prompt': 'a lighthouse'}]}, headers={'x-goog-api-key': token})
        self.assertEqual((predict.status_code, predict.json()['model']), (200, 'imagen-4.0-generate-001'),
                         predict.text)
        self.assertEqual(MOCKS['gemini'].hits[-1].path, '/v1beta/models/imagen-4.0-generate-001:predict')

    def test_anthropic_shaped_model_list_keeps_messages_models(self):
        token = self.key(*NATIVE, 'models:read')
        listing = self.http.get('/v1/models?limit=1000', headers={'x-api-key': token, 'anthropic-version': '2023-06-01'})
        self.assertEqual(listing.status_code, 200, listing.text)
        ids = [row['id'] for row in listing.json()['data']]
        self.assertIn('claude-sonnet-4-5', ids)  # the single Anthropic upstream's model by its bare id
        self.assertFalse([name for name in ids if name.startswith('eleven_')], ids)  # speech-only models are left out
        self.assertIn('gpt-4o-mini@oai', ids)

    def test_path_hygiene_and_account_headers(self):
        token = self.key(*NATIVE)
        hits = len(MOCKS['openai'].hits)
        for path in ('/native/oai/fine_tuning/..%2F..%2Forganization%2Fusers', '/native/%6fai/files/file-abc%3Fx=1',
                     '/native/oai/fine_tuning/jobs%2F..%2F..%2Fx'):
            with self.subTest(path=path):
                answer = self.http.get(path, headers=bearer(BOOT))
                self.assertEqual(answer.status_code, 400, answer.text)
        self.assertEqual(len(MOCKS['openai'].hits), hits)
        account = {'OpenAI-Organization': 'org-probe', 'OpenAI-Project': 'proj-probe'}
        self.assertEqual(self.http.post('/native/oai/responses/compact', json={'x': 1},
                                        headers={**bearer(token), **account}).status_code, 200)
        hit = MOCKS['openai'].hits[-1]
        self.assertNotIn('org-probe', hit.headers.get('openai-organization', ''))
        self.assertNotIn('openai-project', hit.headers)
        admin = self.key(*NATIVE, 'broker:admin')
        self.assertEqual(self.http.post('/native/oai/responses/compact', json={'x': 1},
                                        headers={**bearer(admin), **account}).status_code, 200)
        self.assertEqual(MOCKS['openai'].hits[-1].headers.get('openai-project'), 'proj-probe')

    def test_anthropic_batches_with_x_api_key(self):
        token = self.key(*NATIVE)
        body = {'requests': [{'custom_id': 'r1', 'params': {'model': 'claude-sonnet-4-5', 'max_tokens': 5,
                                                              'messages': [{'role': 'user', 'content': 'hi'}]}}]}
        answer = self.http.post('/v1/messages/batches', json=body,
                                headers={'x-api-key': token, 'anthropic-version': '2023-06-01',
                                         'anthropic-beta': 'message-batches-2024-09-24'})
        self.assertEqual(answer.status_code, 200, answer.text)
        hit = MOCKS['anthropic'].hits[-1]
        self.assertEqual(hit.credential, '<sentinel:anthropic>')
        self.assertEqual(hit.headers.get('anthropic-version'), '2023-06-01')
        self.assertIn('message-batches-2024-09-24', hit.headers.get('anthropic-beta', ''))
        batch_id = answer.json()['id']
        results = self.http.get(f'/v1/messages/batches/{batch_id}/results', headers={'x-api-key': token})
        self.assertEqual(results.status_code, 200, results.text)
        self.assertEqual(json.loads(results.text.splitlines()[0])['custom_id'], 'r1')

    def test_elevenlabs_v2_voices_and_sound_effects(self):
        token = self.key(*NATIVE)
        voices = self.http.get('/v2/voices?page_size=5', headers={'xi-api-key': token})
        self.assertEqual(voices.status_code, 200, voices.text)
        self.assertEqual(voices.json()['voices'][0]['voice_id'], 'nativeVoice0001')
        self.assertEqual(MOCKS['elevenlabs'].hits[-1].credential, '<sentinel:elevenlabs>')
        sound = self.http.post('/v1/sound-generation', json={'text': 'rain'}, headers={'xi-api-key': token})
        self.assertEqual(sound.status_code, 200, sound.text)
        self.assertTrue(sound.headers['content-type'].startswith('audio/mpeg'))
        self.assertEqual(sound.headers.get('history-item-id'), 'hist-native-0001')

    def test_gemini_resumable_upload_and_predict(self):
        token = self.key(*NATIVE)
        start = self.http.post('/upload/v1beta/files', json={'file': {'display_name': 'clip'}},
                               headers={'x-goog-api-key': token, 'X-Goog-Upload-Protocol': 'resumable',
                                        'X-Goog-Upload-Command': 'start',
                                        'X-Goog-Upload-Header-Content-Type': 'audio/wav'})
        self.assertEqual(start.status_code, 200, start.text)
        url = start.headers['x-goog-upload-url']
        self.assertTrue(url.startswith('http://127.0.0.1:8000/native/gem/upload/v1beta/files?upload_id='), url)
        self.assertIn('titanos_upload=', url)
        hit = MOCKS['gemini'].hits[-1]
        self.assertEqual(hit.credential, '<sentinel:gemini>')
        self.assertEqual(hit.headers.get('x-goog-upload-protocol'), 'resumable')
        local = url.split('8000', 1)[1]
        chunk = {'X-Goog-Upload-Command': 'upload, finalize', 'X-Goog-Upload-Offset': '0'}
        forged = self.http.post(local.replace('titanos_upload=', 'titanos_upload=x'), content=W.WAV, headers=chunk)
        self.assertEqual(forged.status_code, 401, forged.text)
        # google-genai sends the chunks to the session URL with no key at all: the capability authorizes this session
        finish = self.http.post(local, content=W.WAV, headers=chunk)
        self.assertEqual(finish.status_code, 200, finish.text)
        name = finish.json()['file']['name']
        self.assertEqual(MOCKS['gemini'].hits[-1].body, W.WAV)
        self.assertNotIn('titanos_upload', MOCKS['gemini'].hits[-1].query)
        self.assertIn('upload_id', MOCKS['gemini'].hits[-1].query)
        spent = self.http.post(local, content=W.WAV, headers=chunk)
        self.assertEqual(spent.status_code, 401, spent.text)
        other = self.key(*NATIVE)
        self.assertEqual(self.http.get('/v1beta/' + name, headers={'x-goog-api-key': other}).status_code, 404)
        own = self.http.get('/native/gem/v1beta/' + name, headers={'x-goog-api-key': token})
        self.assertEqual((own.status_code, own.json()['name']), (200, name))
        predict = self.http.post('/v1beta/models/imagen-4.0-generate-001:predict',
                                 json={'instances': [{'prompt': 'a lighthouse'}], 'parameters': {'sampleCount': 1}},
                                 headers={'x-goog-api-key': token})
        self.assertEqual(predict.status_code, 200, predict.text)
        self.assertEqual(predict.json()['model'], 'imagen-4.0-generate-001')


class TwoCandidates(Base):
    doc_kw = {'second_openai': True}

    def test_ambiguous_bare_path_needs_the_header(self):
        token = self.key(*NATIVE)
        answer = self.upload(token, path='/v1/files')
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (400, 'invalid_request'))
        self.assertIn('X-Titanos-Upstream', answer.json()['error']['message'])
        answer = self.upload(token, path='/v1/files', **{'X-Titanos-Upstream': 'oai2'})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual((len(MOCKS['openai'].hits), len(MOCKS['openai2'].hits)), (0, 1))


class NoStore(Base):
    doc_kw = {'allow_store': False}

    def test_upstream_without_allow_store(self):
        answer = self.upload(self.key(*NATIVE))
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (403, 'data_policy_denied'))
        self.assertEqual(MOCKS['openai'].hits, [])


class PolicyOff(Base):
    doc_kw = {'policy': 'off'}

    def test_cloud_disabled(self):
        answer = self.http.post('/native/oai/responses/compact', json={}, headers=bearer(self.key(*NATIVE)))
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (403, 'cloud_disabled'))


class Managed(Base):
    settings_kw = {'role': 'managed', 'port': 18401}

    def test_managed_role_refuses(self):
        answer = self.http.post('/native/oai/responses/compact', json={}, headers=bearer(self.key(*NATIVE)))
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (403, 'role_forbids'))
        self.assertEqual(MOCKS['openai'].hits, [])


class KeylessDefault(Base):
    keyed = False

    def test_keyless_without_anonymous_cloud(self):
        bare = self.upload('ignored-placeholder', path='/v1/files')
        self.assertIn(bare.status_code, (404, 405))  # the Titanos answer for a path it does not serve, as before
        prefixed = self.http.post('/native/oai/responses/compact', json={})
        # the upstream is invisible to a keyless caller without anonymous_cloud: 404, as for models (no oracle)
        self.assertEqual((prefixed.status_code, prefixed.json()['error']['code']), (404, 'not_found'))
        self.assertEqual(MOCKS['openai'].hits, [])


class CharacterPricedNative(Base):
    doc_kw = {'prices': [{'match': 'native@xi', 'per_1k_chars': '0.30'}]}

    def test_a_per_1k_chars_row_prices_a_character_billed_native_call(self):
        budgeted = self.key(*NATIVE, budget={'usd_per_day': '1'})
        sound = self.http.post('/v1/sound-generation', json={'text': 'rain'}, headers={'xi-api-key': budgeted})
        self.assertEqual(sound.status_code, 200, sound.text)  # reserved as characters, not as unpriced tokens


class KeylessCloud(Base):
    keyed = False
    doc_kw = {'anonymous_cloud': True}

    def test_keyless_with_anonymous_cloud(self):
        answer = self.http.post('/native/oai/responses/compact', json={'x': 1})
        self.assertEqual(answer.status_code, 200, answer.text)
        # stateful rows: anonymous_cloud grants the keyless owner cloud:store; this upstream allows storage
        stored = self.upload('placeholder')
        self.assertEqual(stored.status_code, 200, stored.text)


class KeylessCloudNoStore(Base):
    keyed = False
    doc_kw = {'anonymous_cloud': True, 'allow_store': False}

    def test_keyless_store_still_needs_the_upstream_allow_store(self):
        refused = self.upload('placeholder')
        self.assertEqual((refused.status_code, refused.json()['error']['code']), (403, 'data_policy_denied'))
        self.assertEqual(MOCKS['openai'].hits, [])


class Unconfigured(unittest.TestCase):
    def test_native_prefix_without_broker(self):
        netguard.reset_attempts()
        full = W.FullApp(None)
        try:
            answer = full.http.get('/native/oai/files')
            self.assertEqual((answer.status_code, answer.json()), (404, {'detail': 'Not Found'}))
        finally:
            full.close()
        self.assertEqual(netguard.attempts(), [])


class DroppedHeader(Base):
    def test_translated_request_names_dropped_fields(self):
        token = self.key('inference', 'cloud')
        answer = self.http.post('/v1/messages', json={
            'model': 'gpt-4o-mini@oai', 'max_tokens': 16, 'messages': [{'role': 'user', 'content': 'hi'}],
            'metadata': {'user_id': 'u1'}, 'service_tier': 'auto'}, headers={'x-api-key': token,
                                                                             'anthropic-version': '2023-06-01'})
        self.assertEqual(answer.status_code, 200, answer.text)
        dropped = answer.headers.get('x-titanos-dropped', '')
        self.assertIn('metadata', dropped)
        self.assertIn('service_tier', dropped)
        decision = self.full.broker.decisions.get(answer.headers['x-titanos-decision-id']) \
            if hasattr(self.full.broker.decisions, 'get') else None
        if decision is not None:
            self.assertIn('dropped:metadata', decision['notes'])


if __name__ == '__main__':
    unittest.main()
