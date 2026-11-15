"""Broker key-leak check over a REAL `titanos serve` subprocess (the tests/check_broker_e2e.py harness: fresh TITANOS_HOME,
loopback-only egress, the netguard sitecustomize, env: credential references holding the WP4 sentinels).

The server runs with --log-level debug, TITANOS_BROKER_DECISION_LOG_FILE=true and a chained test sitecustomize that sets
the root logger to DEBUG with a plain stderr handler, so httpx and httpcore DEBUG records, uvicorn's access lines and every
broker log record reach the captured log. All of them are emitted after uvicorn's dictConfig replaced the handlers that
existed when create_app ran (serve builds the app, then uvicorn configures logging, then the lifespan re-applies the
redaction filters).

Secrets: every upstream credential value (the sentinels, including one that only a different upstream accepts), the
bootstrap key, every minted tk_ token, provider-shaped keys generated per run (sk-proj-, AIza, hf_, nvapi-, xai-, a tk_
shape) and an unregistered value sent only as ?key= / ?api_key= / ?access_token=.
Injected into:
- upstream error bodies: 400 bodies echoing the presented key and a trace token (OpenAI, Anthropic, Gemini, ElevenLabs,
  Ollama), custom 400 and 5xx messages carrying shaped keys, the bootstrap key and a tk_ token, a 401 that echoes the key
  (502 upstream_auth_failed), errors inside an MCP tool call;
- exceptions and httpx errors: a refused connect (httpx.ConnectError), a mid-stream disconnect, a malformed JSON body, a
  foreign redirect (502 upstream_redirect_refused);
- uvicorn access lines carrying ?key=, ?api_key= and ?access_token=;
- the decision log (HTTP and file), usage (HTTP and the ledger), audit (HTTP, CLI and the chain files), diagnostics
  (/api/broker, upstreams, health, metrics, catalog, explain, /api/info), every Studio providers-panel API response, MCP
  tool results and remote CLI output.
Asserted: no secret appears in any HTTP response (except a minted token in its own mint answer), any MCP result, any CLI
output, the server log (also with whitespace removed, so a wrapped line cannot hide one) or any file under TITANOS_HOME;
the log really holds DEBUG httpx/httpcore records and the redacted access lines; no mock ever recorded an inbound tk_
token, the bootstrap key or a client Authorization value; each mock only ever saw its own upstream's credential (never
another upstream's, never after a base_url change to another origin, never at a redirect target).
"""
from __future__ import annotations

import json
import os
import re
import string
import sys
import unittest
from pathlib import Path
from random import SystemRandom

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_broker_e2e as E  # noqa: E402  (installs the netguard)
from broker_mocks import SENTINELS, OpenAIMock, netguard, wait_until  # noqa: E402

import httpx  # noqa: E402

GPT = E.GPT
CHAT = E.CHAT
_RANDOM = SystemRandom()
LOG_SITE = ('import logging, sys\n'
            'logging.basicConfig(level=logging.DEBUG, stream=sys.stderr,\n'
            "                    format='%(levelname)s %(name)s: %(message)s')\n")
STUDIO_ROUTES = ('/api/broker', '/api/broker/credentials', '/api/broker/upstreams', '/api/broker/upstreams/openai',
                 '/api/broker/upstreams/openai/health', '/api/broker/decisions?limit=1000', '/api/broker/usage',
                 '/api/broker/models', '/api/broker/virtual-models', '/api/broker/policy', '/api/keys', '/api/keys/self',
                 '/api/info', '/api/broker/catalog', '/api/broker/metrics', '/api/broker/profiles', '/api/audit',
                 '/api/audit/verify', '/healthz', '/v1/models', '/', '/ui/app.js')
OWN_SENTINEL = {'openai': 'openai', 'backup': 'openai', 'flaky': 'openai', 'anthropic': 'anthropic', 'gemini': 'gemini',
                'elevenlabs': 'elevenlabs', 'ollama': 'ollama_cloud', 'spare': 'leak', 'evil': 'openai'}


def _token(prefix: str, length: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return prefix + ''.join(_RANDOM.choice(alphabet) for _ in range(length))


SHAPED = {
    'sk-proj': _token('sk-proj-', 40),
    'aiza': _token('AIza', 35, string.ascii_letters + string.digits + '_-'),
    'hf': _token('hf_', 34),
    'nvapi': _token('nvapi-', 40),
    'xai': _token('xai-', 40),
    'tk_shape': _token('tk_', 10, 'abcdefghijklmnopqrstuvwxyz234567') + '_' + _token('', 43),
}
QUERY_SECRET = 'e2eq' + os.urandom(12).hex()  # never registered and never shaped: only the ?key= rule can scrub it
QUERY_SECRET_2 = 'e2eq' + os.urandom(12).hex()


class Recorder:
    """Every HTTP answer the test receives: (label, status, headers text, body text)."""

    def __init__(self):
        self.rows: list[tuple[str, int, str, str]] = []

    def hook(self, response: httpx.Response) -> None:
        response.read()
        request = response.request
        label = f'{request.method} {request.url.path}'
        headers = '\n'.join(f'{k}: {v}' for k, v in response.headers.items())
        self.rows.append((label, response.status_code, headers, response.content.decode('utf-8', 'replace')))

    def client(self, serve, token=None, **headers) -> httpx.Client:
        if token is not None:
            headers.setdefault('Authorization', 'Bearer ' + token)
        client = httpx.Client(base_url=serve.url, trust_env=False, timeout=60.0, headers=headers,
                              event_hooks={'response': [self.hook]})
        serve._clients.append((dict(headers, hooked=id(client)), client))  # never handed out again by serve.client()
        return client


def setUpModule():
    E.setUpModule()
    E.MOCKS['spare'] = OpenAIMock(credential=SENTINELS['leak'])  # accepts only the 'leak' sentinel
    E.MOCKS['evil'] = OpenAIMock()  # the new origin of a moved upstream: records whatever it is sent
    for name in ('spare', 'evil'):
        E.MOCKS[name].start()


def tearDownModule():
    E.tearDownModule()


class Redaction(unittest.TestCase):
    """All traffic runs in setUpClass (the server is stopped before any assertion reads its log); each test method then
    checks one output channel."""

    @classmethod
    def document(cls) -> dict:
        up = E.upstream
        return {
            'policy': {'cloud_policy': 'fallback'},
            'upstreams': {
                'openai': up('openai', 'openai', 'oa', [GPT, 'omni-moderation-latest']),
                'backup': up('backup', 'openai', 'bk', [GPT]),
                'moved': up('backup', 'openai', 'mv', [GPT]),
                'spare': up('spare', 'openai', 'sp', [GPT]),
                'anthropic': up('anthropic', 'anthropic', 'an', ['claude-sonnet-4-5']),
                'gemini': up('gemini', 'gemini', 'gm', ['gemini-2.5-flash']),
                'elevenlabs': up('elevenlabs', 'elevenlabs', 'el', ['eleven_multilingual_v2', 'scribe_v1']),
                'ollama': up('ollama', 'ollama_cloud', 'ol', ['gpt-oss:120b']),
            },
            'models': {'mini': {'target': GPT + '@openai'}},
            'virtual_models': {'vm': {'capability': 'chat', 'targets': [{'ref': GPT + '@openai'}, {'ref': GPT + '@backup'}]}},
            'prices': E.PRICES,
        }

    @classmethod
    def setUpClass(cls):
        netguard.reset_attempts()
        for mock_ in E.MOCKS.values():
            mock_.reset()
        cls.bootstrap = E.synthetic_key('redaction')
        cls.serve = E.Serve('redaction', cls.document(), bootstrap=cls.bootstrap, log_level='debug',
                            sitecustomize=LOG_SITE,
                            env={'E2E_CRED_SPARE': SENTINELS['leak'], 'TITANOS_BROKER_DECISION_LOG_FILE': 'true',
                                 # breakers are not under test here: every injected failure must reach its mock
                                 'TITANOS_BROKER_BREAKER_FAILURES': '1000'})
        cls.recorder = Recorder()
        cls.mint_replies: dict[str, str] = {}
        cls.tokens: dict[str, str] = {}
        cls.mcp_texts: list[str] = []
        cls.cli_texts: list[str] = []
        try:
            cls.serve.start()
            cls.drive()
        except BaseException:
            cls.serve.close()
            raise
        cls.stopped = cls.serve.stop()
        cls.log = cls.serve.log()
        cls.files = cls.serve.home_files()

    @classmethod
    def tearDownClass(cls):
        for mock_ in E.MOCKS.values():
            mock_.gate.set()
            mock_.faults.clear()
        cls.serve.close()

    # traffic --------------------------------------------------------------------------------------------------------
    @classmethod
    def mint(cls, name: str, *scopes: str, **fields) -> str:
        plain = cls.serve.client(cls.bootstrap)  # not recorded: the answer carries the secret once, by design
        reply = plain.post('/api/keys', json={'name': name, 'scopes': list(scopes), **fields})
        if reply.status_code != 201:
            raise RuntimeError(f'mint {name}: {reply.status_code} {reply.text[:300]}')
        token = reply.json()['secret']
        cls.tokens[name] = token
        cls.mint_replies[name] = reply.text
        return token

    @classmethod
    def drive(cls):
        rec, serve, mocks = cls.recorder, cls.serve, E.MOCKS
        admin = rec.client(serve, cls.bootstrap)
        credentials = {'oa': ('openai', 'E2E_CRED_OPENAI'), 'bk': ('backup', 'E2E_CRED_BACKUP'),
                       'mv': ('moved', 'E2E_CRED_BACKUP'), 'sp': ('spare', 'E2E_CRED_SPARE'),
                       'an': ('anthropic', 'E2E_CRED_ANTHROPIC'), 'gm': ('gemini', 'E2E_CRED_GEMINI'),
                       'el': ('elevenlabs', 'E2E_CRED_ELEVENLABS'), 'ol': ('ollama', 'E2E_CRED_OLLAMA')}
        for cid, (upstream_id, var) in credentials.items():
            reply = admin.put(f'/api/broker/credentials/{cid}', json={'upstream': upstream_id, 'env': var, 'validate': True})
            if reply.status_code not in (200, 201) or reply.json().get('validation') != 'ok':
                raise RuntimeError(f'credential {cid}: {reply.status_code} {reply.text[:300]}')
        full = cls.mint('full', 'inference', 'cloud', 'broker:read', 'models:read', 'mcp')
        reader = cls.mint('reader', 'broker:read', 'mcp')
        cls.mint('bystander', 'inference', 'cloud')
        c = rec.client(serve, full)
        injected = ' '.join(SHAPED.values()) + ' ' + cls.bootstrap + ' ' + cls.tokens['bystander']

        # 1. upstream error bodies that echo the presented key and carry a trace token (the spare upstream's credential)
        mocks['openai'].faults.push('status', status=400, surface='openai.chat')
        c.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT})
        mocks['openai'].faults.push('status', status=400, surface='openai.chat', leak=False,
                                    message='Invalid request; keys seen: ' + injected)
        c.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT})
        mocks['openai'].faults.push('status', status=500, surface='openai.chat', leak=False,
                                    message='server exploded holding ' + injected)
        c.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT, 'stream': True})
        anthropic = rec.client(serve, None, **{'x-api-key': full, 'anthropic-version': E.ANTHROPIC_VERSION})
        mocks['anthropic'].faults.push('status', status=400, surface='anthropic.messages')
        anthropic.post('/v1/messages', json={'model': 'claude-sonnet-4-5@anthropic', 'max_tokens': 16, 'messages': CHAT})
        gemini = rec.client(serve, None, **{'x-goog-api-key': full})
        mocks['gemini'].faults.push('status', status=400, surface='gemini.generateContent')
        gemini.post('/v1beta/models/gemini-2.5-flash@gemini:generateContent', json=E.GEMINI_BODY)
        eleven = rec.client(serve, None, **{'xi-api-key': full})
        mocks['elevenlabs'].faults.push('status', status=400, surface='elevenlabs.tts')
        eleven.post(f'/v1/text-to-speech/{E.VOICE}?output_format=mp3_44100_128',
                    json={'text': 'hi', 'model_id': 'eleven_multilingual_v2@elevenlabs'})
        mocks['ollama'].faults.push('status', status=400, surface='ollama.chat')
        c.post('/api/chat', json={'model': 'gpt-oss:120b@ollama', 'messages': CHAT, 'stream': False})
        # an upstream 401 that echoes the key: 502 upstream_auth_failed (the spare credential goes offline)
        mocks['spare'].faults.push('status', status=401, surface='openai.chat', leak=True)
        c.post('/v1/chat/completions', json={'model': GPT + '@spare', 'messages': CHAT})

        # 2. exceptions and httpx errors
        mocks['openai'].faults.push('connect_refused')
        c.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT})
        mocks['openai'].faults.resume()
        mocks['openai'].faults.push('disconnect_after_frames', frames=1, surface='openai.chat')
        c.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT, 'stream': True})
        mocks['openai'].faults.push('malformed_json', surface='openai.chat')
        c.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT})
        mocks['openai'].faults.push('redirect_foreign', surface='openai.chat')
        c.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT})
        cls.sink = mocks['openai'].sink
        # successful traffic on every family (usage, decisions)
        c.post('/v1/chat/completions', json={'model': 'vm', 'messages': CHAT})
        c.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT, 'stream': True})
        anthropic.post('/v1/messages', json={'model': GPT + '@openai', 'max_tokens': 16, 'messages': CHAT})
        gemini.post('/v1beta/models/gemini-2.5-flash@gemini:streamGenerateContent?alt=sse', json=E.GEMINI_BODY)
        eleven.post('/v1/speech-to-text', data={'model_id': 'scribe_v1@elevenlabs'},
                    files={'file': ('a.wav', E.WAV, 'audio/wav')})
        c.post('/api/chat', json={'model': 'gpt-oss:120b@ollama', 'messages': CHAT, 'stream': False})
        c.post('/v1/moderations', json={'model': 'omni-moderation-latest@openai', 'input': 'hello'})

        # 3. a moved upstream: the bound credential is never presented to the new origin
        moved = admin.put('/api/broker/upstreams/moved', json={
            'profile': 'openai', 'base_url': mocks['evil'].api_base, 'enabled': True, 'credential': 'mv', 'models': [GPT]})
        cls.moved_status = moved.status_code
        cls.moved_chat = c.post('/v1/chat/completions', json={'model': GPT + '@moved', 'messages': CHAT})
        admin.post('/api/broker/upstreams/moved/test')

        # 4. query credentials: uvicorn access lines and Titanos's refusal
        open_client = rec.client(serve)
        open_client.get('/v1/models', params={'key': QUERY_SECRET})
        open_client.get('/healthz', params={'access_token': QUERY_SECRET_2})
        rec.client(serve, cls.bootstrap).get('/api/info', params={'api_key': QUERY_SECRET})
        open_client.post(f'/v1beta/models/gemini-2.5-flash@gemini:generateContent?key={full}', json=E.GEMINI_BODY)
        open_client.post(f'/v1beta/models/gemini-2.5-flash@gemini:generateContent?key={QUERY_SECRET}',
                         json=E.GEMINI_BODY)
        rec.client(serve, cls.bootstrap).get('/api/broker', params={'key': cls.bootstrap})
        # wrong credentials of every spelling (auth failures are logged and audited)
        rec.client(serve, SHAPED['tk_shape']).get('/v1/models')
        rec.client(serve, None, **{'x-api-key': SHAPED['sk-proj']}).post('/v1/messages', json={
            'model': 'mini', 'max_tokens': 8, 'messages': CHAT})

        # 5. diagnostics and every Studio providers-panel route, as admin and as a broker:read key
        for token in (cls.bootstrap, reader):
            client = rec.client(serve, token)
            for route in STUDIO_ROUTES:
                client.get(route)
            client.post('/api/broker/explain', json={'surface': 'openai.chat', 'body': {'model': 'vm', 'messages': CHAT}})
        admin.get('/api/broker/upstreams/spare')
        admin.get('/api/broker/upstreams/moved/health')

        # 6. MCP tool results, including an upstream error inside a tool call
        mocks['openai'].faults.push('status', status=400, surface='openai.chat', leak=True)
        results = E.mcp_calls(serve.url, full, [
            ('chat', {'payload': {'model': 'mini', 'messages': CHAT}}),
            ('chat', {'payload': {'model': 'mini', 'messages': CHAT}}),
            ('whoami', {}), ('broker_status', {}), ('list_upstreams', {}), ('broker_usage', {}),
            ('broker_decisions', {'limit': 200}), ('upstream_health', {'id': 'spare'}),
            ('broker_explain', {'surface': 'openai.chat', 'body': {'model': 'mini', 'messages': CHAT}}),
            ('moderate', {'model': 'omni-moderation-latest@openai', 'input': 'hi'}),
        ])
        results += E.mcp_calls(serve.url, reader, [('whoami', {}), ('broker_status', {}), ('broker_catalog', {})])
        for result in results:
            cls.mcp_texts.append(json.dumps({'content': [getattr(c_, 'text', str(c_)) for c_ in result.content],
                                             'structured': result.structured_content, 'meta': result.meta,
                                             'error': result.is_error}, default=str))
        cls.mcp_results = results

        # 7. remote CLI output (the bootstrap key travels in TITANOS_API_KEY, never argv)
        for argv in (('broker', 'status'), ('broker', 'credential', 'list'), ('broker', 'upstream', 'list'),
                     ('keys', 'list'), ('audit', 'verify'), ('audit',)):
            done = serve.cli(*argv, keyed=True, url=True)
            cls.cli_texts.append(f'{argv} rc={done.returncode}\n{done.stdout}\n{done.stderr}')
        hit = mocks['openai'].hits[-1] if mocks['openai'].hits else None
        if hit is not None:
            wait_until(lambda: hit.completed or hit.disconnected or hit.aborted, 5.0)

    # helpers --------------------------------------------------------------------------------------------------------
    @classmethod
    def secrets(cls) -> dict[str, str]:
        out = {f'sentinel:{name}': SENTINELS[name] for name in ('openai', 'anthropic', 'gemini', 'elevenlabs',
                                                               'ollama_cloud', 'leak')}
        out['bootstrap'] = cls.bootstrap
        out.update({f'token:{name}': token for name, token in cls.tokens.items()})
        out.update({f'shaped:{name}': value for name, value in SHAPED.items()})
        out['query'] = QUERY_SECRET
        out['query2'] = QUERY_SECRET_2
        return out

    def assert_clean(self, where: str, text: str, *, allow: tuple = ()) -> None:
        compact = re.sub(r'\s+', '', text)
        for name, value in self.secrets().items():
            if name in allow:
                continue
            self.assertNotIn(value, text, f'{name} leaked into {where}')
            self.assertNotIn(value, compact, f'{name} leaked into {where} (across whitespace)')

    # checks ---------------------------------------------------------------------------------------------------------
    def test_server_ran_and_stopped_without_egress(self):
        self.assertTrue(self.stopped, 'serve did not shut down cleanly:\n' + self.log[-3000:])
        self.assertEqual(self.serve.netguard_attempts(), [])
        self.assertEqual(netguard.attempts(), [])

    def test_injections_really_happened(self):
        """The scrubbed outputs are not vacuous: the faults reached the server and produced the expected answers."""
        statuses = {}
        for label, status, _, body in self.recorder.rows:
            statuses.setdefault(label, []).append(status)
        chat = statuses['POST /v1/chat/completions']
        self.assertIn(400, chat)
        self.assertIn(502, chat)  # upstream 401 -> upstream_auth_failed, and the refused redirect
        self.assertIn(400, statuses['POST /v1/messages'])
        self.assertIn(401, statuses['POST /v1beta/models/gemini-2.5-flash@gemini:generateContent'])
        codes = [json.loads(body).get('error', {}).get('code') for label, status, _, body in self.recorder.rows
                 if label == 'POST /v1/chat/completions' and status == 502 and body.startswith('{')]
        self.assertIn('upstream_auth_failed', codes)
        self.assertIn('upstream_redirect_refused', codes)
        self.assertTrue(any(r.is_error for r in self.mcp_results), 'the MCP error path was not exercised')
        self.assertIn(self.moved_status, (200, 201))

    def test_http_responses_carry_no_secret(self):
        for label, status, headers, body in self.recorder.rows:
            self.assert_clean(f'{label} [{status}]', headers + '\n' + body)
        # the upstream error bodies were passed through scrubbed, not dropped
        bodies = [body for label, status, _, body in self.recorder.rows if status == 400]
        self.assertTrue(any('<redacted>' in body for body in bodies), 'no scrubbed upstream error body was observed')
        # a minted token appears only in its own mint answer, once
        for name, text in self.mint_replies.items():
            self.assertEqual(text.count(self.tokens[name]), 1)
            self.assert_clean(f'mint {name}', text, allow=(f'token:{name}',))

    def test_mcp_results_carry_no_secret(self):
        self.assertTrue(self.mcp_texts)
        for text in self.mcp_texts:
            self.assert_clean('MCP result', text)

    def test_cli_output_carries_no_secret(self):
        self.assertTrue(self.cli_texts)
        for text in self.cli_texts:
            self.assertIn('rc=0', text.splitlines()[0], text[:2000])
            self.assert_clean('CLI output', text)

    def test_server_log_is_scrubbed_after_dictconfig(self):
        log = self.log
        self.assert_clean('serve log', log)
        # the scanned channels really exist: DEBUG httpx/httpcore records and uvicorn access lines with a query key
        self.assertRegex(log, r'DEBUG httpcore[.\w]*: ')
        self.assertRegex(log, r'(INFO|DEBUG) httpx: HTTP Request: ')
        access = [line for line in log.splitlines() if 'GET /v1/models?' in line or 'GET /healthz?' in line]
        self.assertTrue(access, 'no access line with a query string was logged')
        for line in access:
            self.assertIn('<redacted>', line, line)
        self.assertIn('Application shutdown complete', log)

    def test_home_files_carry_no_secret(self):
        names = set(self.files)
        for directory in ('broker/decisions/', 'broker/usage/', 'broker/audit/'):
            self.assertTrue(any(name.startswith(directory) for name in names), (directory, sorted(names)))
        self.assertIn('broker/keys.json', names)
        self.assertIn('broker/vault/index.json', names)
        for name, data in self.files.items():
            self.assert_clean(f'TITANOS_HOME/{name}', data.decode('utf-8', 'replace'))
        index = json.loads(self.files['broker/vault/index.json'])
        records = index.get('records', index)
        self.assertTrue(records)

    def test_mocks_saw_only_their_own_credential(self):
        inbound = [self.bootstrap, *self.tokens.values()]
        for name, mock_ in E.MOCKS.items():
            own = OWN_SENTINEL[name]
            for hit in mock_.hits:
                for token in inbound:
                    self.assertFalse(hit.presented(token), f'{name} {hit.path} was sent an inbound credential')
                for where, spelled in hit.credentials:
                    self.assertIn(f'<sentinel:{own}>', spelled, f'{name} {hit.path} {where}: {spelled}')
                for other, value in SENTINELS.items():
                    if other != own:
                        self.assertFalse(hit.presented(value), f'{name} {hit.path} was sent the {other} credential')
                self.assertEqual(hit.duplicate_credentials, [], (name, hit.path))
        self.assertIsNotNone(self.sink, 'the redirect fault never started its sink')
        for hit in self.sink.hits:
            self.assertEqual(hit.credentials, [], 'a redirect target received a credential')
        for hit in E.MOCKS['evil'].hits:
            self.assertFalse(hit.presented(SENTINELS['openai']), 'a moved upstream presented its credential to the new origin')
        self.assertNotEqual(self.moved_chat.status_code, 200, self.moved_chat.text[:300])
        for name in ('openai', 'anthropic', 'gemini', 'elevenlabs', 'ollama'):
            self.assertTrue(E.MOCKS[name].hits, name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
