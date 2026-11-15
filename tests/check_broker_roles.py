"""WP7 roles over the integrated app (standalone, broker preconditions, managed) with the WP4 loopback mocks.

managed:
- cloud and fallback-eligible requests give typed errors with zero upstream hits, including an auth-none custom_openai
  upstream; the vault is sealed at startup; no probe and no live catalog fetch happens, even past the catalog TTL;
  credential validation and upstream tests are refused (409 role_forbids);
- delegation works only with all five conditions (managed_delegation=parent, an authenticated key holding delegate,
  X-Titanos-Delegate: cloud, a local credential, an explicit target), never for an alias;
- the parent deadline gives 504 deadline_exceeded; the parent trace is echoed as X-Request-ID; a parent placement
  local_only is fail-closed; an inbound hop restricts the request to local targets;
- /api/info reports role managed (orchestrated true) with standalone still true; /v1/health/ready and /api/version are
  public (the latter only in managed role).
broker preconditions refuse with a message naming the switch: no bootstrap key, a LAN bind without native TLS,
behind_tls_proxy with a LAN bind, an unverifiable audit log, an egress key without a budget; the port refusal set.
standalone with no broker.json: no egress at all (no egress client is ever built).
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import OpenAIMock, netguard  # noqa: E402

from titanos.broker.role import RESERVED_PORTS, check_serve_preconditions  # noqa: E402
from titanos.broker.types import KeyStoreSummary  # noqa: E402
from titanos.engine import EngineError  # noqa: E402
from titanos.settings import Settings  # noqa: E402

GPT = 'gpt-4o-mini@openai'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
BOOT = 'wp7-roles-bootstrap-0001'
MOCKS: dict = {}


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['router'] = OpenAIMock()
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def document() -> dict:
    return {
        'policy': {'cloud_policy': 'fallback'},
        'upstreams': {
            'openai': W.S.upstream(MOCKS['openai'], 'openai', credential='oa', models=['gpt-4o-mini'],
                                   catalog='live'),
            'nocred': {'profile': 'openai', 'base_url': MOCKS['openai'].api_base, 'enabled': True,
                       'models': ['gpt-4o-mini']},
            'custom': {'profile': 'custom_openai', 'base_url': MOCKS['router'].api_base, 'enabled': True, 'auth': 'none',
                       'models': ['gpt-4o-mini']},
        },
        'models': {'mini': {'target': GPT}},
        'virtual_models': {'vm': {'capability': 'chat', 'targets': [{'ref': 'chat-local'}, {'ref': GPT}]}},
    }


class Case(unittest.TestCase):
    settings_kw: dict = {'role': 'managed'}
    api_key = BOOT

    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
        self.full = W.FullApp(document(), models={'chat-local': 'chat'}, api_key=self.api_key,
                              settings_kw=self.settings_kw)
        self.c = self.full.http

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def hits(self) -> int:
        return W.S.total_hits(*MOCKS.values())

    def chat(self, model, token=BOOT, **headers):
        return self.c.post('/v1/chat/completions', json={'model': model, 'messages': CHAT},
                           headers={**W.bearer(token), **headers})


class Managed(Case):
    def test_cloud_and_fallback_give_typed_errors_with_zero_hits(self):
        for model in (GPT, 'gpt-4o-mini@custom', 'mini'):
            reply = self.chat(model)
            self.assertEqual(reply.status_code, 403, (model, reply.text[:200]))
            self.assertEqual(reply.json()['error']['code'], 'role_forbids')
        busy = EngineError('Worker is busy', 503, 'worker_unavailable')

        def failing(*args, **kw):
            raise busy
        self.full.app.state.engine.target.infer = failing
        reply = self.chat('vm')
        self.assertEqual((reply.status_code, reply.json()['error']['code']), (503, 'worker_unavailable'))
        self.assertEqual(self.hits(), 0)

    def test_vault_sealed_no_probes_no_live_catalogs(self):
        self.assertTrue(self.full.vault.sealed)
        catalog = self.full.broker.snapshot().catalog
        catalog.ttl = 0.0  # every live catalog is stale at once
        for _ in range(2):
            self.assertEqual(self.c.get('/v1/models', headers=W.bearer(BOOT)).status_code, 200)
            refreshed = self.c.get('/api/broker/catalog?refresh=true', headers=W.bearer(BOOT))
            self.assertEqual(refreshed.status_code, 200)
        self.assertEqual(refreshed.json()['upstream_status']['openai']['error'], 'live_catalogs_off')
        self.assertIsNone(self.full.broker._components.get('monitor'))
        validate = self.c.post('/api/broker/credentials/oa/validate', headers=W.bearer(BOOT))
        self.assertEqual((validate.status_code, validate.json()['error']['code']), (409, 'role_forbids'))
        test = self.c.post('/api/broker/upstreams/openai/test', headers=W.bearer(BOOT))
        self.assertEqual((test.status_code, test.json()['error']['code']), (409, 'role_forbids'))
        cred = self.c.put('/api/broker/credentials/oa2', json={'upstream': 'openai', 'secret': 'z' * 32},
                          headers=W.bearer(BOOT))
        self.assertEqual((cred.status_code, cred.json()['error']['code']), (409, 'role_forbids'))
        self.assertEqual(self.hits(), 0)

    def test_parent_headers(self):
        trace = self.chat('chat-local', **{'x-hivemind-trace': 'trace-wp7-0001'})
        self.assertEqual((trace.status_code, trace.headers['x-request-id']), (200, 'trace-wp7-0001'))
        hop = self.chat('chat-local', **{'X-Titanos-Hop': 'another-instance'})
        self.assertEqual(hop.status_code, 200)

        def slow(*args, **kw):
            time.sleep(1.0)
            return {'text': 'late', 'finish_reason': 'stop', 'usage': {}}
        self.full.app.state.engine.target.infer = slow
        late = self.chat('chat-local', **{'X-Titanos-Deadline-Ms': '150'})
        self.assertEqual((late.status_code, late.json()['error']['code']), (504, 'deadline_exceeded'))

    def test_info_and_public_routes(self):
        info = self.c.get('/api/info', headers=W.bearer(BOOT)).json()
        self.assertEqual((info['role'], info['orchestrated'], info['standalone']), ('managed', True, True))
        self.assertEqual(self.c.get('/v1/health/ready').json()['role'], 'managed')
        version = self.c.get('/api/version')
        self.assertEqual((version.status_code, version.json()['name'], version.json()['role']),
                         (200, 'Titanos', 'managed'))
        self.assertEqual(self.c.get('/v1/models').status_code, 401)  # every other route still needs the key


class Delegation(Case):
    settings_kw = {'role': 'managed', 'managed_delegation': 'parent'}

    def test_delegation_needs_all_five_conditions(self):
        self.assertFalse(self.full.vault.sealed)
        _, delegate = self.full.mint('inference', 'cloud', 'delegate', name='parent')
        _, plain = self.full.mint('inference', 'cloud', name='plain')
        ok = self.chat(GPT, delegate, **{'X-Titanos-Delegate': 'cloud'})
        self.assertEqual(ok.status_code, 200, ok.text[:300])
        self.assertEqual(len(MOCKS['openai'].hits), 1)
        refused = [
            self.chat(GPT, delegate),  # no X-Titanos-Delegate
            self.chat(GPT, plain, **{'X-Titanos-Delegate': 'cloud'}),  # no delegate scope
            self.chat('mini', delegate, **{'X-Titanos-Delegate': 'cloud'}),  # an alias is not explicit
            self.chat('gpt-4o-mini@nocred', delegate, **{'X-Titanos-Delegate': 'cloud'}),  # no local credential
            self.chat(GPT, BOOT, **{'X-Titanos-Delegate': 'cloud'}),  # the bootstrap key is not a delegate key
            self.chat(GPT, delegate, **{'X-Titanos-Delegate': 'cloud', 'x-hivemind-placement': 'local_only'}),
            self.chat(GPT, delegate, **{'X-Titanos-Delegate': 'cloud', 'X-Titanos-Hop': 'another-instance'}),
        ]
        for reply in refused:
            self.assertIn(reply.status_code, (400, 403, 404, 503), reply.text[:200])
        self.assertEqual(refused[-2].json()['error']['code'], 'location_unsatisfiable')  # placement is fail-closed
        self.assertEqual(len(MOCKS['openai'].hits), 1)
        self.assertEqual(len(MOCKS['router'].hits), 0)


class Standalone(unittest.TestCase):
    def test_no_broker_json_means_no_egress_at_all(self):
        netguard.reset_attempts()
        full = W.FullApp(None, models={'chat-local': 'chat'})
        try:
            c = full.http
            self.assertEqual(c.post('/v1/chat/completions', json={'model': GPT, 'messages': CHAT}).status_code, 404)
            self.assertEqual(c.get('/v1/models').status_code, 200)
            self.assertEqual(c.get('/api/broker').json()['upstreams']['total'], 0)
            self.assertNotIn('egress', full.broker._components)
            self.assertEqual(c.get('/api/version').json()['role'], 'standalone')  # registered for Ollama clients; public only in managed role
        finally:
            full.close()
        self.assertEqual(netguard.attempts(), [])


class Preconditions(unittest.TestCase):
    def check(self, settings, **kw):
        values = {'bootstrap_key': BOOT, 'keystore': KeyStoreSummary(), 'explicit_port': True,
                  'vault_backend': 'dpapi', 'audit_ok': True}
        values.update(kw)
        return check_serve_preconditions(settings, **values)

    def refuses(self, needle, settings_kw=None, **kw):
        try:
            settings = Settings(**(settings_kw or {}))
            self.check(settings, **kw)
        except ValueError as exc:
            self.assertIn(needle, str(exc))
            return
        self.fail(f'expected a refusal naming {needle}')

    def test_broker_role_refusals_name_the_switch(self):
        self.refuses('TITANOS_API_KEY', {'role': 'broker'}, bootstrap_key=None)
        self.refuses('TITANOS_TLS_CERTFILE', {'role': 'broker', 'host': '0.0.0.0'})
        self.refuses('TITANOS_BEHIND_TLS_PROXY', {'role': 'broker', 'host': '0.0.0.0', 'behind_tls_proxy': True})
        self.refuses('audit', {'role': 'broker'}, audit_ok=False)
        self.refuses('budget', {'role': 'broker'},
                     keystore=KeyStoreSummary(has_records=True, active=1, active_admin=0,
                                              cloud_keys_missing_budget=('abcdefghij',)))
        self.assertEqual(self.check(Settings(role='broker', limit_concurrency=64)), [])

    def test_managed_port_refusal_set(self):
        self.assertEqual(RESERVED_PORTS, frozenset({11434, 11435, 1234, 1235, 8000, 8001}))
        for port in sorted(RESERVED_PORTS):
            self.refuses(str(port), {'role': 'managed', 'port': port})
        self.refuses('explicit', {'role': 'managed', 'port': 18080}, explicit_port=False)
        self.assertEqual(self.check(Settings(role='managed', port=18080)), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
