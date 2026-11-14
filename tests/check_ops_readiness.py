"""OPS (a): GET /v1/health/ready is truthful (titanos/readiness.py behind broker/routes.py health_ready).

- standalone: 200 {status ready, version, role, models, checks{registry loaded, engine available}}, Cache-Control no-store,
  no upstream block (standalone and managed roles never consult upstreams);
- while the registry is being replaced (gate reloading) the probe answers 503 registry_reloading with Retry-After; after a
  failed reload (gate / EngineSlot unavailable) 503 registry_unavailable; both clear when the gate clears;
- broker role: the enabled upstreams are required; with every breaker closed 200 and checks.upstreams lists them; an open
  breaker (three network failures) gives 503 upstream_unhealthy naming the upstream, a reset gives 200 again;
- broker role with a broker.json that does not parse: 503 broker_config_unavailable;
- managed role: 200 with role managed and no upstream block;
- the probe stays public in keyed mode (no bearer), for 200 and 503 alike; the 503 body names exactly one typed reason.
Loopback mocks only; the netguard asserts zero non-loopback connection attempts.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import OpenAIMock, netguard  # noqa: E402
from titanos import readiness  # noqa: E402

BOOT = 'ops-readiness-bootstrap-0001'
MOCKS: dict = {}


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['openai'].start()


def tearDownModule():
    MOCKS['openai'].stop()


def gate_of(app) -> dict:
    """The server's reload gate: the Guard middleware is constructed with it (server.create_app)."""
    for middleware in app.user_middleware:
        if 'gate' in middleware.kwargs:
            return middleware.kwargs['gate']
    raise AssertionError('no Guard middleware carries the gate')


def broker_document() -> dict:
    return {'policy': {'cloud_policy': 'fallback'},
            'upstreams': {'openai': W.S.upstream(MOCKS['openai'], 'openai', credential='oa', models=['gpt-4o-mini'])}}


class Guarded(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        MOCKS['openai'].reset()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def full(self, document=None, **kw):
        full = W.FullApp(document, **kw)
        self.addCleanup(full.close)
        return full

    def assert_not_ready(self, answer, reason: str):
        self.assertEqual(answer.status_code, 503, answer.text[:300])
        body = answer.json()
        self.assertEqual((body['status'], body['reason']), ('not_ready', reason), body)
        self.assertTrue(body['message'])
        self.assertEqual(answer.headers.get('retry-after'), '1')
        return body


class Standalone(Guarded):
    def test_ready_shape(self):
        full = self.full(models={'chat-local': 'chat', 'emb': 'embeddings'})
        answer = full.http.get('/v1/health/ready')
        self.assertEqual(answer.status_code, 200, answer.text[:300])
        body = answer.json()
        self.assertEqual((body['status'], body['role'], body['models']), ('ready', 'standalone', 2))
        self.assertEqual(body['checks'], {'registry': 'loaded', 'engine': 'available'})
        self.assertNotIn('reason', body)
        self.assertTrue(body['version'])
        self.assertEqual(answer.headers.get('cache-control'), 'no-store')
        self.assertIsNone(answer.headers.get('retry-after'))

    def test_reload_and_failed_reload_are_not_ready(self):
        full = self.full(models={'chat-local': 'chat'})
        gate = gate_of(full.app)
        self.assertEqual(full.http.get('/v1/health/ready').status_code, 200)
        gate['reloading'] = True
        try:
            body = self.assert_not_ready(full.http.get('/v1/health/ready'), 'registry_reloading')
            self.assertEqual(body['checks']['registry'], 'reloading')
        finally:
            gate['reloading'] = False
        self.assertEqual(full.http.get('/v1/health/ready').status_code, 200)
        gate['unavailable'] = True
        try:
            body = self.assert_not_ready(full.http.get('/v1/health/ready'), 'registry_unavailable')
            self.assertEqual(body['checks']['engine'], 'unavailable')
        finally:
            gate['unavailable'] = False
        # the EngineSlot flag alone (what every engine route consults) is enough
        full.app.state.engine.unavailable = True
        try:
            self.assert_not_ready(full.http.get('/v1/health/ready'), 'registry_unavailable')
        finally:
            full.app.state.engine.unavailable = False
        self.assertEqual(full.http.get('/v1/health/ready').json()['status'], 'ready')

    def test_public_in_keyed_mode_for_200_and_503(self):
        full = self.full(models={'chat-local': 'chat'}, api_key=BOOT)
        self.assertEqual(full.http.get('/v1/models').status_code, 401)  # keyed server
        self.assertEqual(full.http.get('/v1/health/ready').status_code, 200)
        gate = gate_of(full.app)
        gate['reloading'] = True
        try:
            self.assert_not_ready(full.http.get('/v1/health/ready'), 'registry_reloading')
        finally:
            gate['reloading'] = False


class BrokerRole(Guarded):
    def test_required_upstreams_decide(self):
        full = self.full(broker_document(), models={'chat-local': 'chat'}, api_key=BOOT, settings_kw={'role': 'broker'})
        answer = full.http.get('/v1/health/ready')
        self.assertEqual(answer.status_code, 200, answer.text[:300])
        body = answer.json()
        self.assertEqual(body['role'], 'broker')
        self.assertEqual(body['checks']['upstreams']['required'], ['openai'])
        self.assertEqual(body['checks']['upstreams']['unhealthy'], [])
        breaker = full.broker.monitor.board.get('openai', 'oa')
        for _ in range(3):  # broker_breaker_failures default 3
            breaker.record('network')
        self.assertEqual(breaker.state, 'open')
        body = self.assert_not_ready(full.http.get('/v1/health/ready'), 'upstream_unhealthy')
        self.assertEqual(body['checks']['upstreams']['unhealthy'], ['openai'])
        self.assertIn('openai', body['message'])
        self.assertEqual(body['checks'], {**body['checks'], 'registry': 'loaded', 'engine': 'available'})
        breaker.reset()
        self.assertEqual(full.http.get('/v1/health/ready').json()['status'], 'ready')
        self.assertEqual(MOCKS['openai'].hits, [], 'readiness never probes an upstream itself')

    def test_no_enabled_upstream_requires_none(self):
        full = self.full({'upstreams': {}}, models={'chat-local': 'chat'}, api_key=BOOT, settings_kw={'role': 'broker'})
        body = full.http.get('/v1/health/ready').json()
        self.assertEqual((body['status'], body['checks']['upstreams']['required']), ('ready', []))

    def test_unloadable_broker_config_is_not_ready(self):
        full = self.full(None, models={'chat-local': 'chat'}, api_key=BOOT, settings_kw={'role': 'broker'})
        (full.home / 'broker.json').write_text('{"upstreams": [1, 2', encoding='utf-8')
        body = self.assert_not_ready(full.http.get('/v1/health/ready'), 'broker_config_unavailable')
        self.assertTrue(body['checks']['upstreams'].get('error'))


class ManagedRole(Guarded):
    def test_managed_is_ready_without_upstream_checks(self):
        full = self.full(broker_document(), models={'chat-local': 'chat'}, api_key=BOOT, settings_kw={'role': 'managed'})
        body = full.http.get('/v1/health/ready').json()
        self.assertEqual((body['status'], body['role']), ('ready', 'managed'))
        self.assertNotIn('upstreams', body['checks'])
        self.assertEqual(MOCKS['openai'].hits, [])


class ReportFunction(unittest.TestCase):
    def test_report_without_a_gate_or_broker(self):
        config = SimpleNamespace(models={'a': 1})
        status, body = readiness.report(None, gate=None, config=config, engine=SimpleNamespace(unavailable=False),
                                        role='standalone')
        self.assertEqual((status, body['status'], body['models']), (200, 'ready', 1))
        status, body = readiness.report(None, gate={'reloading': True}, config=config,
                                        engine=SimpleNamespace(unavailable=False), role='standalone')
        self.assertEqual((status, body['reason']), (503, 'registry_reloading'))
        status, body = readiness.report(None, gate={}, config=config, engine=SimpleNamespace(unavailable=True),
                                        role='managed')
        self.assertEqual((status, body['reason'], body['role']), (503, 'registry_unavailable', 'managed'))
        self.assertNotIn('upstreams', body['checks'])

    def test_broker_role_reasons_are_ordered(self):
        class FakeBroker:
            def snapshot(self):
                return SimpleNamespace(load_error=None, config=SimpleNamespace(enabled_upstreams=lambda: {'b': 1, 'a': 2}))

            def upstream_health(self, uid):
                return {'breaker': 'open' if uid == 'a' else 'half_open'}
        status, body = readiness.report(FakeBroker(), gate={}, config=SimpleNamespace(models={}),
                                        engine=SimpleNamespace(unavailable=False), role='broker')
        self.assertEqual((status, body['reason']), (503, 'upstream_unhealthy'))
        self.assertEqual(body['checks']['upstreams'], {'required': ['a', 'b'], 'unhealthy': ['a'],
                                                       'breakers': {'a': 'open', 'b': 'half_open'}})
        # a registry problem outranks an upstream problem (nothing local would be served either)
        status, body = readiness.report(FakeBroker(), gate={'reloading': True}, config=SimpleNamespace(models={}),
                                        engine=SimpleNamespace(unavailable=False), role='broker')
        self.assertEqual(body['reason'], 'registry_reloading')


if __name__ == '__main__':
    unittest.main(verbosity=2)
