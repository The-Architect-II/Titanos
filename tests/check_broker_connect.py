"""R29 keyless Studio provider setup, R30 environment key discovery and R31 one-step connect, over the integrated app
and the CLI, against the WP4 OpenAI loopback mock. The netguard is installed; every test ends with zero non-loopback
connection attempts. Synthetic sentinel credentials only; no secret value appears in any answer, stdout or file.

- GET /api/broker/env-candidates: names and booleans for every profile with well-known variable names (a set synthetic
  value never appears in the answer); keyless needs the X-Titanos-Studio header (403 keyless_mode_refused otherwise);
  the Python client reaches it;
- POST /api/broker/connect with a keyed broker:admin + credentials:write caller: the upstream is created from its
  profile, the secret stored and validated with one probe, the upstream enabled, cloud_policy raised from off to
  explicit, the live catalog read, aliases published for the provider's bare model ids (aliases: true), so a chat with
  the bare id 'gpt-4o-mini' reaches the provider; broker.connect is audited; the answer and broker.json hold no secret;
  a key without credentials:write is refused (403); a refused credential is 422 credential_validation_failed and leaves
  the upstream disabled; an env reference needs the admin scope (403 for broker:admin alone); an env reference the
  server process does not hold is refused (503 credential_unavailable), and one that disappears from the server's
  environment after connect makes the upstream not ready (ready false, readiness credential_unavailable, 404 with zero
  egress) until the variable is back;
- R29 keyless Studio: with X-Titanos-Studio: 1 and a JSON body, connect stores the credential and, on this keyless
  loopback standalone server, turns anonymous_cloud on by default, so the keyless caller then reaches the cloud model by
  its qualified ref and by the provider's bare id (gpt-4o-mini); an env reference is accepted only for the profile's
  well-known variable (OPENAI_API_KEY) and only for the profile's default origin, file: references are refused, and no
  fingerprint of a host value is returned (the review's env-exfiltration finding); anonymous_cloud: false keeps keyless
  cloud use off, and the keyless caller then gets 403 cloud_disabled naming policy.anonymous_cloud (not 404); connect
  allow_store sets the upstream's data_policy.allow_store; credential writes are accepted and audited with actor
  keyless-studio; without the header (or with a non-JSON body) they stay 403 keyless_mode_refused; Studio may change
  anonymous_cloud but not allow_http_lan; managed role never elevates the header (403) and refuses keyed connect (409
  role_forbids);
- CLI: `titanos broker quickstart` lists names and the connect commands, never values; `titanos broker connect openai
  --from-env VAR --base-url URL --aliases` (offline, TITANOS_HOME files) stores an env reference, enables the upstream,
  raises the policy and publishes aliases, and its next steps say that the server reads the variable from its own
  environment and that a running server needs `titanos broker reload`; the variable's value is in no output or file;
  once keys.json holds keys, the offline notes (prices import, broker reload, connect next steps) say the reload needs a
  broker:admin key in TITANOS_API_KEY or a restart, and name the admin mint while no active key holds that scope.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import SENTINELS, OpenAIMock, netguard  # noqa: E402

BOOT = 'connect-check-bootstrap-0001'
VAR = 'NATIVE_CHECK_OPENAI_KEY'
STUDIO = {'X-Titanos-Studio': '1'}
MOCKS: dict = {}
CHAT = [{'role': 'user', 'content': 'Say hello.'}]


def setUpModule():
    MOCKS['openai'] = OpenAIMock(models=['gpt-4o-mini', 'text-embedding-3-small'])
    MOCKS['openai'].start()


def tearDownModule():
    MOCKS['openai'].stop()


def bearer(token: str) -> dict:
    return {'Authorization': 'Bearer ' + token}


def connect_body(**kw) -> dict:
    body = {'provider': 'openai', 'base_url': MOCKS['openai'].api_base,
            'credential': {'value': SENTINELS['openai']}}
    body.update(kw)
    return body


class Base(unittest.TestCase):
    keyed = True
    settings_kw: dict = {}

    def setUp(self):
        netguard.reset_attempts()
        MOCKS['openai'].reset()
        self.full = W.FullApp(None, api_key=BOOT if self.keyed else None, settings_kw=self.settings_kw)
        self.http = self.full.http

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def key(self, *scopes) -> str:
        return self.full.mint(*scopes, name='c' + str(len(scopes)))[1]

    def config_text(self) -> str:
        path = self.full.home / 'broker.json'
        return path.read_text(encoding='utf-8') if path.exists() else ''


class Keyed(Base):
    def test_env_candidates_names_only(self):
        token = self.key('broker:admin')
        value = 'sk-test-envcandidate-' + os.urandom(6).hex()
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': value}):
            answer = self.http.get('/api/broker/env-candidates', headers=bearer(token))
            from fastapi.testclient import TestClient
            from titanos.client import Client
            client = Client('http://127.0.0.1:8000', api_key=token)
            client.http.close()  # the client's own httpx.Client is replaced by an in-process one with the same key
            client.http = TestClient(self.full.app, base_url='http://127.0.0.1:8000', client=('127.0.0.1', 50001),
                                     headers=bearer(token))
            via_client = client.broker_env_candidates()
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertNotIn(value, answer.text)
        self.assertEqual(answer.headers.get('cache-control'), 'no-store')
        rows = {row['provider']: row for row in answer.json()['data']}
        self.assertEqual(rows['openai']['variables'], [{'name': 'OPENAI_API_KEY', 'set': True}])
        self.assertEqual([v['name'] for v in rows['gemini']['variables']], ['GEMINI_API_KEY', 'GOOGLE_API_KEY'])
        self.assertEqual(via_client['data'], answer.json()['data'])
        self.assertEqual(self.http.get('/api/broker/env-candidates', headers=bearer(self.key('inference'))).status_code,
                         403)

    def test_connect_end_to_end_with_aliases(self):
        token = self.key('broker:admin', 'credentials:write')
        answer = self.http.post('/api/broker/connect', json=connect_body(aliases=True), headers=bearer(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        result = answer.json()
        self.assertNotIn(SENTINELS['openai'], answer.text)
        self.assertEqual(result['upstream']['enabled'], True)
        self.assertEqual(result['validation'], 'ok')
        self.assertEqual(result['cloud_policy']['file'], 'explicit')
        self.assertEqual(result['policy_changes'], {'cloud_policy': 'explicit'})
        self.assertEqual(sorted(result['models']), ['gpt-4o-mini', 'text-embedding-3-small'])
        self.assertEqual(sorted(result['aliases']), ['gpt-4o-mini', 'text-embedding-3-small'])
        self.assertEqual(result['credential']['upstream_id'], 'openai')
        self.assertNotIn(SENTINELS['openai'], self.config_text())
        paths = [hit.path for hit in MOCKS['openai'].hits]
        self.assertEqual(paths.count('/v1/models'), 2)  # one validation probe, one catalog refresh
        kinds = [row['kind'] for row in self.full.broker.audit.query(limit=50)]
        for kind in ('upstream.write', 'credential.put', 'credential.validate', 'broker.connect', 'alias.write'):
            self.assertIn(kind, kinds)
        user = self.key('inference', 'cloud')
        chat = self.http.post('/v1/chat/completions', json={'model': 'gpt-4o-mini', 'messages': CHAT},
                              headers=bearer(user))
        self.assertEqual(chat.status_code, 200, chat.text)
        self.assertEqual(chat.headers['x-titanos-served-by'], 'cloud:openai')
        again = self.http.post('/api/broker/connect', json=connect_body(aliases=True), headers=bearer(token))
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual(again.json()['aliases'], [])  # already published to the same targets: nothing new

    def test_connect_refusals(self):
        broker_admin = self.key('broker:admin')
        answer = self.http.post('/api/broker/connect', json=connect_body(), headers=bearer(broker_admin))
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (403, 'scope_denied'))
        writer = self.key('broker:admin', 'credentials:write')
        env = self.http.post('/api/broker/connect', json=connect_body(credential={'env': VAR}), headers=bearer(writer))
        self.assertEqual((env.status_code, env.json()['error']['code']), (403, 'scope_denied'))
        wrong = self.http.post('/api/broker/connect',
                               json=connect_body(credential={'value': 'sk-test-wrong-' + os.urandom(6).hex()}),
                               headers=bearer(writer))
        self.assertEqual((wrong.status_code, wrong.json()['error']['code']), (422, 'credential_validation_failed'))
        upstream = self.http.get('/api/broker/upstreams/openai', headers=bearer(BOOT)).json()
        self.assertFalse(upstream['enabled'])
        bad = self.http.post('/api/broker/connect', json={'provider': 'pending_contract'}, headers=bearer(writer))
        self.assertEqual(bad.status_code, 400)

    def test_env_reference_the_server_lacks_is_not_ready(self):
        # `titanos broker connect --from-env` validates from the CLI's environment; a server process that does not hold
        # the variable must say so (ready false, readiness credential_unavailable), not report ready and answer 404.
        # Over HTTP the server checks from its own environment, so the same connect is refused there (docs/BROKER.md)
        self.assertNotIn(VAR, os.environ)
        refused = self.http.post('/api/broker/connect', json=connect_body(credential={'env': VAR}), headers=bearer(BOOT))
        self.assertEqual((refused.status_code, refused.json()['error']['code']), (503, 'credential_unavailable'))
        self.assertEqual(MOCKS['openai'].hits, [])
        with mock.patch.dict(os.environ, {VAR: SENTINELS['openai']}):
            answer = self.http.post('/api/broker/connect', json=connect_body(credential={'env': VAR}),
                                    headers=bearer(BOOT))
            self.assertEqual(answer.status_code, 200, answer.text)
            row = self.http.get('/api/broker/upstreams/openai', headers=bearer(BOOT)).json()
            self.assertEqual((row['ready'], row['readiness']), (True, None), row)
        self.assertNotIn(VAR, os.environ)
        row = self.http.get('/api/broker/upstreams/openai', headers=bearer(BOOT)).json()
        self.assertFalse(row['ready'], row)
        self.assertEqual(row['readiness'], {'reason': 'unvalidated', 'detail': 'credential_unavailable'})
        self.assertEqual(row['credential_validation'], 'ok')  # the stored validation stands; the server cannot use it
        user = self.key('inference', 'cloud')
        MOCKS['openai'].reset()
        chat = self.http.post('/v1/chat/completions', json={'model': 'gpt-4o-mini@openai', 'messages': CHAT},
                              headers=bearer(user))
        self.assertEqual((chat.status_code, chat.json()['error']['code']), (404, 'model_not_found'))
        native = self.http.post('/native/openai/responses/compact', json={'a': 1},
                                headers=bearer(self.key('native', 'cloud', 'cloud:store')))
        self.assertEqual((native.status_code, native.json()['error']['code']), (503, 'credential_unavailable'))
        self.assertIn('not available in the server process', native.json()['error']['message'])
        self.assertEqual(MOCKS['openai'].hits, [])  # both refused before egress
        with mock.patch.dict(os.environ, {VAR: SENTINELS['openai']}):  # e.g. serve restarted with the variable
            row = self.http.get('/api/broker/upstreams/openai', headers=bearer(BOOT)).json()
            self.assertEqual((row['ready'], row['readiness']), (True, None), row)
            chat = self.http.post('/v1/chat/completions', json={'model': 'gpt-4o-mini@openai', 'messages': CHAT},
                                  headers=bearer(user))
            self.assertEqual(chat.status_code, 200, chat.text)
            self.assertEqual(chat.headers['x-titanos-served-by'], 'cloud:openai')


class FailedReconnect(Base):
    def test_a_failed_reconnect_never_replaces_the_working_credential(self):
        """Connect probes a candidate under a scratch id first, so a re-connect whose env: variable this server lacks
        (503 credential_unavailable) or whose key the provider refuses leaves the working, validated record exactly as it
        was (no rotation, no 'unvalidated' state) and leaves no scratch record behind; the upstream keeps serving."""
        with mock.patch.dict(os.environ, {VAR: SENTINELS['openai']}):
            answer = self.http.post('/api/broker/connect', json=connect_body(credential={'env': VAR}),
                                    headers=bearer(BOOT))
            self.assertEqual(answer.status_code, 200, answer.text)
        before = self.http.get('/api/broker/credentials', headers=bearer(BOOT)).json()['data']
        self.assertEqual([row['id'] for row in before], ['openai-key'])
        other = 'CONNECT_CHECK_ABSENT_KEY'  # TITANOS_* names are refused as references by design
        self.assertNotIn(other, os.environ)
        refused = self.http.post('/api/broker/connect', json=connect_body(credential={'env': other}),
                                 headers=bearer(BOOT))
        self.assertEqual((refused.status_code, refused.json()['error']['code']), (503, 'credential_unavailable'))
        rejected = self.http.post('/api/broker/connect',
                                  json=connect_body(credential={'value': 'sk-test-' + 'x' * 40}), headers=bearer(BOOT))
        self.assertEqual(rejected.json()['error']['code'], 'credential_validation_failed', rejected.text)
        after = self.http.get('/api/broker/credentials', headers=bearer(BOOT)).json()['data']
        self.assertEqual(after, before)
        self.assertFalse(any('-chk-' in row['id'] for row in after), after)
        with mock.patch.dict(os.environ, {VAR: SENTINELS['openai']}):
            chat = self.http.post('/v1/chat/completions', json={'model': 'gpt-4o-mini@openai', 'messages': CHAT},
                                  headers=bearer(self.key('inference', 'cloud')))
            self.assertEqual(chat.status_code, 200, chat.text)


class KeylessStudio(Base):
    keyed = False

    def test_studio_connect_turns_keyless_cloud_on_and_bare_ids_work(self):
        plain = self.http.post('/api/broker/connect', json=connect_body())
        self.assertEqual((plain.status_code, plain.json()['error']['code']), (403, 'keyless_mode_refused'))
        answer = self.http.post('/api/broker/connect', json=connect_body(allow_store=True), headers=STUDIO)
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertTrue(answer.json()['anonymous_cloud'])  # the default for the keyless loopback owner
        self.assertTrue(answer.json()['allow_store'])
        self.assertNotIn(SENTINELS['openai'], answer.text + self.config_text())
        for model in ('gpt-4o-mini@openai', 'gpt-4o-mini'):
            chat = self.http.post('/v1/chat/completions', json={'model': model, 'messages': CHAT})
            self.assertEqual(chat.status_code, 200, chat.text)
        self.assertEqual(json.loads(self.config_text())['upstreams']['openai']['data_policy']['allow_store'], True)
        actors = {row['actor'] for row in self.full.broker.audit.query(limit=50) if row['kind'] == 'broker.connect'}
        self.assertEqual(actors, {'keyless-studio'})

    def test_keyless_cloud_off_is_a_typed_403(self):
        answer = self.http.post('/api/broker/connect', json=connect_body(anonymous_cloud=False), headers=STUDIO)
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertFalse(answer.json()['anonymous_cloud'])
        chat = self.http.post('/v1/chat/completions', json={'model': 'gpt-4o-mini@openai', 'messages': CHAT})
        self.assertEqual((chat.status_code, chat.json()['error']['code']), (403, 'cloud_disabled'), chat.text)
        self.assertIn('anonymous_cloud', chat.json()['error']['message'])

    def test_studio_host_sources_are_narrow(self):
        """The keyless Studio principal may bind only the profile's well-known variable, only for the profile's default
        origin, never a file: path, and never learns a fingerprint of a host value."""
        with mock.patch.dict(os.environ, {VAR: SENTINELS['openai'], 'OPENAI_API_KEY': SENTINELS['openai']}):
            cases = (({'env': VAR}, None, 'well-known'),
                     ({'env': 'OPENAI_API_KEY'}, MOCKS['openai'].api_base, 'default address'),
                     ({'file': str(Path(tempfile.gettempdir()) / 'x.key')}, None, 'file:'))
            for credential, base_url, why in cases:
                with self.subTest(credential=credential):
                    body = connect_body(credential=credential, upstream='probe', validate=False)
                    if base_url is None:
                        body.pop('base_url')
                    else:
                        body['base_url'] = base_url
                    refused = self.http.post('/api/broker/connect', json=body, headers=STUDIO)
                    self.assertEqual((refused.status_code, refused.json()['error']['code']), (403, 'scope_denied'),
                                     refused.text)
                    self.assertIn(why, refused.json()['error']['message'])
            body = connect_body(credential={'env': 'OPENAI_API_KEY'}, upstream='viaenv', validate=False, enable=False)
            body.pop('base_url')
            accepted = self.http.post('/api/broker/connect', json=body, headers=STUDIO)
            self.assertEqual(accepted.status_code, 200, accepted.text)
            self.assertEqual(accepted.json()['credential']['source'], 'env')
            self.assertIsNone(accepted.json()['credential']['fingerprint'])
            listed = self.http.get('/api/broker/credentials').json()['data']
            self.assertTrue(all(row['fingerprint'] is None for row in listed if row['source'] == 'env'), listed)
            # the same rule on the credential routes: a custom variable, or a base_url the caller picked
            upstream = {'profile': 'openai', 'base_url': MOCKS['openai'].api_base, 'enabled': False}
            self.assertIn(self.http.put('/api/broker/upstreams/moved', json=upstream).status_code, (200, 201))
            for body in ({'upstream': 'moved', 'env': 'OPENAI_API_KEY', 'validate': False},
                         {'upstream': 'viaenv', 'env': VAR, 'validate': False}):
                refused = self.http.put('/api/broker/credentials/c-' + body['upstream'], json=body, headers=STUDIO)
                self.assertEqual((refused.status_code, refused.json()['error']['code']), (403, 'scope_denied'),
                                 refused.text)
        self.assertEqual(MOCKS['openai'].hits, [])
        env = self.http.get('/api/broker/env-candidates')
        self.assertEqual((env.status_code, env.json()['error']['code']), (403, 'keyless_mode_refused'))
        self.assertEqual(self.http.get('/api/broker/env-candidates', headers=STUDIO).status_code, 200)

    def test_studio_credential_and_policy_writes(self):
        upstream = {'profile': 'openai', 'base_url': MOCKS['openai'].api_base, 'enabled': False}
        self.assertIn(self.http.put('/api/broker/upstreams/oa', json=upstream).status_code, (200, 201))
        body = {'upstream': 'oa', 'secret': SENTINELS['openai']}
        refused = self.http.put('/api/broker/credentials/oa-key', json=body)
        self.assertEqual((refused.status_code, refused.json()['error']['code']), (403, 'keyless_mode_refused'))
        text_body = self.http.put('/api/broker/credentials/oa-key', content=json.dumps(body),
                                  headers={**STUDIO, 'Content-Type': 'text/plain'})
        self.assertEqual(text_body.status_code, 403)
        stored = self.http.put('/api/broker/credentials/oa-key', json=body, headers=STUDIO)
        self.assertEqual(stored.status_code, 201, stored.text)
        self.assertNotIn(SENTINELS['openai'], stored.text)
        rows = [r for r in self.full.broker.audit.query(limit=50) if r['kind'] == 'credential.put']
        self.assertEqual({r['actor'] for r in rows}, {'keyless-studio'})
        policy = self.http.get('/api/broker/policy').json()
        allowed = {k: policy[k] for k in ('cloud_policy', 'default_location', 'anonymous_cloud', 'allow_http_lan',
                                          'max_fallbacks', 'auto_model')}
        allowed['anonymous_cloud'] = True
        self.assertEqual(self.http.put('/api/broker/policy', json=allowed, headers=STUDIO).status_code, 200)
        self.assertEqual(self.http.put('/api/broker/policy', json={**allowed, 'anonymous_cloud': False}).status_code,
                         403)
        lan = self.http.put('/api/broker/policy', json={**allowed, 'allow_http_lan': True}, headers=STUDIO)
        self.assertEqual((lan.status_code, lan.json()['error']['code']), (403, 'keyless_mode_refused'))


class ManagedKeyless(Base):
    keyed = False
    settings_kw = {'role': 'managed', 'port': 18402}

    def test_header_never_elevates_in_managed_role(self):
        answer = self.http.post('/api/broker/connect', json=connect_body(), headers=STUDIO)
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (403, 'keyless_mode_refused'))
        self.assertEqual(MOCKS['openai'].hits, [])


class ManagedKeyed(Base):
    settings_kw = {'role': 'managed', 'port': 18403}

    def test_managed_role_refuses_connect(self):
        answer = self.http.post('/api/broker/connect', json=connect_body(), headers=bearer(BOOT))
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (409, 'role_forbids'))
        self.assertEqual(MOCKS['openai'].hits, [])


def run_cli(argv, env):
    from titanos.cli import main
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


class Cli(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        MOCKS['openai'].reset()
        self.home = Path(tempfile.mkdtemp(prefix='titanos-connect-cli-'))
        self.addCleanup(shutil.rmtree, self.home, True)
        self.env = {'TITANOS_HOME': str(self.home), 'TITANOS_BROKER_EGRESS': 'loopback-only',
                    'TITANOS_BROKER_VAULT_BACKEND': 'refs-only', VAR: SENTINELS['openai']}

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def files_text(self) -> str:
        return '\n'.join(p.read_bytes().decode('utf-8', 'replace') for p in sorted(self.home.rglob('*')) if p.is_file())

    def test_quickstart_lists_names_only(self):
        value = 'sk-test-quickstart-' + os.urandom(6).hex()
        code, out, err = run_cli(['broker', 'quickstart'], {**self.env, 'OPENAI_API_KEY': value})
        self.assertEqual(code, 0, err)
        self.assertNotIn(value, out + err)
        result = json.loads(out)
        plan = {item['provider']: item for item in result['plan']}
        self.assertEqual(plan['openai']['command'], 'titanos broker connect openai --from-env OPENAI_API_KEY')

    def test_offline_connect_from_env(self):
        code, out, err = run_cli(['broker', 'connect', 'openai', '--from-env', VAR, '--base-url',
                                  MOCKS['openai'].api_base, '--aliases'], self.env)
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertTrue(result['upstream']['enabled'])
        self.assertEqual(result['credential']['source'], 'env')
        self.assertIn('gpt-4o-mini', result['aliases'])
        self.assertNotIn(SENTINELS['openai'], out + err + self.files_text())
        config = json.loads((self.home / 'broker.json').read_text(encoding='utf-8'))
        self.assertEqual(config['policy']['cloud_policy'], 'explicit')
        self.assertEqual(config['upstreams']['openai']['credential'], 'openai-key')
        self.assertEqual(config['models']['gpt-4o-mini']['target'], 'gpt-4o-mini@openai')
        self.assertEqual(MOCKS['openai'].hits[0].credential, 'Bearer <sentinel:openai>')
        # this CLI validated from its own environment and wrote the files: the next steps say what the server needs
        steps = ' '.join(result['next_steps'])
        self.assertIn(f'The server reads {VAR} from its own environment', steps)
        self.assertIn('titanos broker reload --url', steps)

    def test_offline_apply_hint_follows_key_enforcement(self):
        # a broker.json change reaches a running server by restart or reload; once keys are enforced the reload needs
        # broker:admin, which the quick start's inference key lacks (the release review's 401/403 reload finding)
        prices = self.home / 'prices.json'
        prices.write_text(json.dumps([{'match': 'gpt-4o-mini@openai', 'input_per_mtok': '0.15',
                                       'output_per_mtok': '0.60'}]), encoding='utf-8')

        def import_note() -> str:
            code, _, err = run_cli(['broker', 'prices', 'import', str(prices)], self.env)
            self.assertEqual(code, 0, err)
            return err
        note = import_note()
        self.assertIn('titanos broker reload --url URL', note)
        self.assertNotIn('enforces', note)
        code, _, err = run_cli(['keys', 'create', '--name', 'laptop', '--scope', 'inference', '--scope', 'cloud:openai',
                                '--expires', '30d'], self.env)
        self.assertEqual(code, 0, err)
        note = import_note()
        self.assertIn('none of the active ones holds broker:admin', note)
        self.assertIn('titanos keys create --name admin --scope broker:admin --expires 1d', note)
        self.assertIn('restart', note)
        code, out, err = run_cli(['broker', 'reload'], self.env)
        self.assertEqual(code, 0, err)
        self.assertIn('none of the active ones holds broker:admin', json.loads(out)['note'])
        code, out, err = run_cli(['broker', 'connect', 'openai', '--from-env', VAR, '--base-url',
                                  MOCKS['openai'].api_base], self.env)
        self.assertEqual(code, 0, err)
        self.assertIn('none of the active ones holds broker:admin', ' '.join(json.loads(out)['next_steps']))
        code, _, err = run_cli(['keys', 'create', '--name', 'admin', '--scope', 'broker:admin', '--expires', '1d'],
                               self.env)
        self.assertEqual(code, 0, err)
        note = import_note()
        self.assertIn('with a key holding broker:admin in TITANOS_API_KEY', note)
        self.assertNotIn('none of the active ones', note)


if __name__ == '__main__':
    unittest.main()
