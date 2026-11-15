"""Public-route pre-read rejection and bounded body reads at the Guard front door.

Actual create_app/Guard/TrustedHost/ASGI with a recording receive callable; inert fixtures only.
No listener, no model, no device, no network, no request to any running service.
"""
from pathlib import Path
import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fastapi.testclient import TestClient  # noqa: E402
from titanos import server  # noqa: E402
from titanos.config import Config  # noqa: E402
from titanos.settings import Settings  # noqa: E402

# /v1/health/ready (the broker readiness probe) joined the public set; /api/version is public only in managed role
EXPECTED_PUBLIC = {'/healthz', '/v1/health/ready', '/', '/ui/app.js', '/ui/i18n.js', '/ui/style.css', '/ui/retrieval.js',
                   '/ui/titanos-wordmark-v2.svg', '/ui/titanos-honeycomb-t-v2.svg',
                   # API documentation views: readable without the bearer (GET only) in both key modes; see docs/CONFIGURATION.md
                   '/docs', '/openapi.json', '/docs/assets/swagger-ui.css', '/docs/assets/swagger-ui-bundle.js'}
EMPTY = {'type': 'http.request', 'body': b'', 'more_body': False}


def make_scope(method, path, headers=(), host='127.0.0.1:8000'):
    raw = [(b'host', host.encode('latin-1'))] + [(k.lower().encode('latin-1'), v.encode('latin-1')) for k, v in headers]
    return {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method, 'scheme': 'http',
            'path': path, 'raw_path': path.encode(), 'query_string': b'', 'root_path': '', 'headers': raw,
            'client': ('127.0.0.1', 50000), 'server': ('127.0.0.1', 8000)}


class Recorder:
    """Fake ASGI receive: counts invocations, optionally blocks until a gate opens."""

    def __init__(self, chunks=None, gate=None):
        self.calls = 0
        self.chunks = list(chunks if chunks is not None else [EMPTY])
        self.gate = gate

    async def __call__(self):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        return self.chunks.pop(0) if self.chunks else {'type': 'http.disconnect'}


async def call(app, scope, receive):
    sent = []

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    start = next(m for m in sent if m['type'] == 'http.response.start')
    headers = {k.decode('latin-1').lower(): v.decode('latin-1') for k, v in start['headers']}
    body = b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body')
    return start['status'], headers, body


class PublicRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='titanos-security-public-')
        self.addCleanup(self.tmp.cleanup)
        # the broker reads TITANOS_HOME lazily (keys, broker.json): keep it inside this fixture
        home = patch.dict(os.environ, {'TITANOS_HOME': str(Path(self.tmp.name) / 'home')})
        home.start()
        self.addCleanup(home.stop)

    def client(self, api_key=None, name='app', env=None, settings=None):
        """Build the app. Body-read budgets arrive the operator's way: TITANOS_BODY_READ_* environment values read by
        create_app's load_settings(), or an explicit Settings instance; there are no module constants to patch."""
        with patch.dict(os.environ, env or {}):
            app = server.create_app(Config(), api_key=api_key, output_dir=Path(self.tmp.name) / name, settings=settings)
        client = TestClient(app, base_url='http://127.0.0.1:8000')
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return app, client

    def test_public_set_is_explicit(self):
        self.assertEqual(set(server.PUBLIC_ROUTES), EXPECTED_PUBLIC)
        self.assertEqual(server.PUBLIC_METHODS, ('GET',))  # FastAPI 0.141 routes GET only; HEAD gets the router's own 405

    def test_post_healthz_with_body_is_405_before_receive(self):
        app, client = self.client()
        for path in sorted(EXPECTED_PUBLIC):
            receive = Recorder([{'type': 'http.request', 'body': b'x' * 5, 'more_body': False}])
            scope = make_scope('POST', path, [('content-type', 'application/json'), ('content-length', '5')])
            status, headers, body = client.portal.call(call, app, scope, receive)
            self.assertEqual(status, 405, (path, body))
            self.assertEqual(headers.get('allow'), 'GET')
            self.assertEqual(receive.calls, 0, path)
            self.assertLess(len(body), 512)
            self.assertEqual(json.loads(body)['error']['code'], 'method_not_allowed')
        # Through the HTTP test client as well (status + Allow header).
        reply = client.post('/healthz', json={'x': 1})
        self.assertEqual((reply.status_code, reply.headers['allow']), (405, 'GET'))
        self.assertEqual(client.put('/', content=b'abc').status_code, 405)
        self.assertEqual(client.delete('/ui/app.js').status_code, 405)

    def test_get_healthz_with_declared_body_is_rejected_before_receive(self):
        app, client = self.client()
        for path in ('/healthz', '/v1/health/ready'):
            for framing in ([('content-length', '3')], [('transfer-encoding', 'chunked')],
                            [('content-type', 'text/plain'), ('content-length', '1')]):
                receive = Recorder([{'type': 'http.request', 'body': b'abc', 'more_body': False}])
                status, headers, body = client.portal.call(call, app, make_scope('GET', path, framing), receive)
                self.assertEqual(status, 400, (path, framing, body))
                self.assertEqual(headers.get('connection'), 'close')
                self.assertEqual(receive.calls, 0, framing)
                self.assertEqual(json.loads(body)['error']['code'], 'unexpected_body')
            # Content-Length: 0 is not a body.
            receive = Recorder()
            status, _, _ = client.portal.call(call, app, make_scope('GET', path, [('content-length', '0')]), receive)
            self.assertEqual(status, 200, path)

    def test_plain_public_get_still_serves_without_receive(self):
        app, client = self.client()
        receive = Recorder()
        status, _, body = client.portal.call(call, app, make_scope('GET', '/healthz'), receive)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['status'], 'ok')
        self.assertEqual(receive.calls, 0)
        receive = Recorder()  # HEAD is not served by this FastAPI; the Guard answers 405 without receive
        status, headers, _ = client.portal.call(call, app, make_scope('HEAD', '/healthz'), receive)
        self.assertEqual((status, headers.get('allow'), receive.calls), (405, 'GET', 0))
        self.assertEqual(client.get('/healthz').json()['status'], 'ok')
        self.assertEqual(client.get('/healthz').json()['role'], 'standalone')
        receive = Recorder()
        status, _, body = client.portal.call(call, app, make_scope('GET', '/v1/health/ready'), receive)
        self.assertEqual((status, json.loads(body)['status'], json.loads(body)['role'], receive.calls),
                         (200, 'ready', 'standalone', 0))
        receive = Recorder()
        status, headers, _ = client.portal.call(call, app, make_scope('HEAD', '/v1/health/ready'), receive)
        self.assertEqual((status, headers.get('allow'), receive.calls), (405, 'GET', 0))
        self.assertEqual(client.get('/api/version').status_code, 200)  # registered in every role (Ollama clients); public only in managed role
        self.assertEqual(client.get('/').status_code, 200)
        self.assertEqual(client.get('/ui/style.css').status_code, 200)
        self.assertEqual(client.get('/v1/models').status_code, 200)  # non-public route unaffected

    def test_public_rules_are_independent_of_api_key(self):
        app, client = self.client(api_key='fixture-secret', name='keyed')
        receive = Recorder([{'type': 'http.request', 'body': b'{}', 'more_body': False}])
        status, headers, _ = client.portal.call(call, app, make_scope('POST', '/healthz', [('content-length', '2')]), receive)
        self.assertEqual((status, headers.get('allow'), receive.calls), (405, 'GET', 0))
        self.assertEqual(client.get('/healthz').status_code, 200)
        self.assertEqual(client.get('/v1/health/ready').status_code, 200)  # public in keyed mode too
        receive = Recorder([{'type': 'http.request', 'body': b'{}', 'more_body': False}])
        status, headers, _ = client.portal.call(call, app, make_scope('POST', '/v1/health/ready',
                                                                      [('content-length', '2')]), receive)
        self.assertEqual((status, headers.get('allow'), receive.calls), (405, 'GET', 0))
        self.assertEqual(client.get('/v1/models').status_code, 401)
        self.assertEqual(client.get('/v1/models', headers={'Authorization': 'Bearer fixture-secret'}).status_code, 200)

    def test_slow_body_on_authenticated_route_is_bounded(self):
        app, client = self.client(api_key='fixture-secret', name='slow', env={'TITANOS_BODY_READ_SECONDS': '0.05'})
        app.state.engine.target = MagicMock()
        gate = asyncio.Event()  # never opened: the body never arrives
        receive = Recorder([{'type': 'http.request', 'body': b'{"model":"m"}', 'more_body': False}], gate=gate)
        scope = make_scope('POST', '/api/models/load', [('authorization', 'Bearer fixture-secret'),
                                                         ('content-type', 'application/json'), ('content-length', '13')])
        started = time.monotonic()
        status, headers, body = client.portal.call(call, app, scope, receive)
        elapsed = time.monotonic() - started
        self.assertEqual(status, 408, body)
        self.assertEqual(headers.get('connection'), 'close')
        self.assertLess(elapsed, 5.0)
        self.assertEqual(receive.calls, 1)
        self.assertEqual(app.state.engine.target.mock_calls, [])
        self.assertEqual(json.loads(body)['error']['code'], 'request_timeout')

    def test_concurrent_body_reads_are_bounded(self):
        app, client = self.client(api_key='fixture-secret', name='slots',
                                  env={'TITANOS_BODY_READ_CONCURRENCY': '1', 'TITANOS_BODY_READ_WAIT_SECONDS': '0.05',
                                       'TITANOS_BODY_READ_SECONDS': '5'})
        app.state.engine.target = MagicMock()
        app.state.engine.target.load.return_value = {'model': 'm', 'status': 'loaded'}
        chunk = {'type': 'http.request', 'body': b'{"model":"m"}', 'more_body': False}
        headers = [('authorization', 'Bearer fixture-secret'), ('content-type', 'application/json'), ('content-length', '13')]

        async def scenario():
            gate = asyncio.Event()
            holder = Recorder([chunk], gate=gate)
            first = asyncio.ensure_future(call(app, make_scope('POST', '/api/models/load', headers), holder))
            for _ in range(50):
                await asyncio.sleep(0.005)
                if holder.calls:
                    break
            self.assertEqual(holder.calls, 1, 'first request must be holding the single read slot')
            waiting = Recorder([chunk])
            second = await call(app, make_scope('POST', '/api/models/load', headers), waiting)
            # A body-less request does not need a read slot while the slot is held.
            free = await call(app, make_scope('GET', '/v1/models', [('authorization', 'Bearer fixture-secret')]), Recorder())
            gate.set()
            released = await first
            # Slot returned: a later body-bearing request proceeds.
            third = await call(app, make_scope('POST', '/api/models/load', headers), Recorder([chunk]))
            return second, waiting.calls, free, released, third

        second, waiting_calls, free, released, third = client.portal.call(scenario)
        self.assertEqual(second[0], 503, second[2])
        self.assertEqual(second[1].get('retry-after'), '1')
        self.assertEqual(second[1].get('connection'), 'close')
        self.assertEqual(waiting_calls, 0, 'refused request never had its body read')
        self.assertEqual(json.loads(second[2])['error']['code'], 'server_busy')
        self.assertEqual(free[0], 200)
        self.assertEqual(released[0], 200, released[2])
        self.assertEqual(third[0], 200, third[2])
        self.assertEqual(app.state.engine.target.load.call_count, 2)

    def test_declares_body_framing(self):
        self.assertFalse(server.declares_body({}))
        self.assertFalse(server.declares_body({b'content-length': b'0'}))
        self.assertFalse(server.declares_body({b'content-type': b'application/json'}))
        self.assertTrue(server.declares_body({b'content-length': b'1'}))
        self.assertTrue(server.declares_body({b'transfer-encoding': b'Chunked'}))
        self.assertEqual(server.declared_length({b'content-length': b' 13 '}), 13)
        self.assertIsNone(server.declared_length({b'content-length': b'abc'}))
        self.assertIsNone(server.declared_length({}))

    def test_trailing_slash_variants_of_public_routes_get_the_same_pre_read_answer(self):
        app, client = self.client()
        for path in ('/healthz/', '/ui/app.js/', '/healthz//'):
            with self.subTest(path=path):
                receive = Recorder([{'type': 'http.request', 'body': b'x' * 13, 'more_body': False}])
                status, headers, body = client.portal.call(call, app, make_scope('POST', path, [('content-length', '13')]), receive)
                self.assertEqual((status, headers.get('allow'), receive.calls), (405, 'GET', 0), (path, body))
                receive = Recorder([{'type': 'http.request', 'body': b'abc', 'more_body': False}])
                status, _, _ = client.portal.call(call, app, make_scope('GET', path, [('content-length', '3')]), receive)
                self.assertEqual((status, receive.calls), (400, 0), path)
        # A plain GET on the variant still reaches the router, which answers with its own redirect; no receive either.
        receive = Recorder()
        status, headers, _ = client.portal.call(call, app, make_scope('GET', '/healthz/'), receive)
        self.assertEqual((status, receive.calls), (307, 0))
        self.assertTrue(headers.get('location', '').endswith('/healthz'))
        self.assertEqual(server.public_path('/healthz/'), '/healthz')
        self.assertEqual(server.public_path('/'), '/')
        self.assertEqual(server.public_path('/v1/models/'), '/v1/models')
        # Non-public paths are untouched by the normalisation: still keyed/bounded as before.
        appk, clientk = self.client(api_key='fixture-secret', name='keyed-slash')
        receive = Recorder([{'type': 'http.request', 'body': b'{}', 'more_body': False}])
        status, _, _ = clientk.portal.call(call, appk, make_scope('POST', '/v1/models/', [('content-length', '2')]), receive)
        self.assertEqual((status, receive.calls), (401, 0))

    def test_declared_length_over_the_limit_is_413_before_receive(self):
        app, client = self.client(api_key='fixture-secret', name='too-large')
        app.state.engine.target = MagicMock()
        too_big = str(server.load_settings().max_body_bytes + 1)
        receive = Recorder([{'type': 'http.request', 'body': b'x', 'more_body': True}])
        scope = make_scope('POST', '/api/models/load', [('authorization', 'Bearer fixture-secret'),
                                                         ('content-type', 'application/json'), ('content-length', too_big)])
        status, headers, body = client.portal.call(call, app, scope, receive)
        self.assertEqual(status, 413, body)
        self.assertEqual(headers.get('connection'), 'close')
        self.assertEqual(receive.calls, 0, 'the framing alone proves the overflow')
        self.assertEqual(json.loads(body)['error']['code'], 'request_too_large')
        self.assertEqual(app.state.engine.target.mock_calls, [])
        # Unauthenticated callers in keyed mode still get 401 first (no size oracle without the bearer).
        receive = Recorder()
        status, _, _ = client.portal.call(call, app, make_scope('POST', '/api/models/load', [('content-length', too_big)]), receive)
        self.assertEqual((status, receive.calls), (401, 0))

    def test_body_read_budget_scales_with_declared_length(self):
        defaults = Settings.from_env({})
        base, rate = defaults.body_read_seconds, defaults.body_read_min_rate
        self.assertEqual((base, rate, defaults.body_read_concurrency, defaults.body_read_wait_seconds), (30.0, 256 * 1024, 8, 2.0))
        self.assertEqual(server.body_read_budget({}, defaults), base)
        self.assertEqual(server.body_read_budget({b'content-length': b'13'}, defaults), base)
        self.assertEqual(server.body_read_budget({b'transfer-encoding': b'chunked'}, defaults), base)
        large = 64 * 1024 * 1024
        self.assertEqual(server.body_read_budget({b'content-length': str(large).encode()}, defaults), max(base, large / rate))
        self.assertGreater(server.body_read_budget({b'content-length': str(large).encode()}, defaults), base)
        # An explicit Settings changes the arithmetic; without one the current process settings (the environment) apply.
        tuned = Settings.from_env({'TITANOS_BODY_READ_SECONDS': '1.5', 'TITANOS_BODY_READ_MIN_RATE': '1024'})
        self.assertEqual(server.body_read_budget({b'content-length': b'4096'}, tuned), 4.0)
        self.assertEqual(server.body_read_budget({}, tuned), 1.5)
        with patch.dict(os.environ, {'TITANOS_BODY_READ_SECONDS': '7'}):
            self.assertEqual(server.body_read_budget({}), 7.0)
        # No module constants remain for a test or an operator to patch by mistake.
        for name in ('BODY_READ_SECONDS', 'BODY_READ_MIN_RATE', 'BODY_READ_CONCURRENCY', 'BODY_READ_WAIT_SECONDS'):
            self.assertFalse(hasattr(server, name), name)
        # The 408 message names the budget that applied; here the budget arrives as an explicit Settings instance.
        app, client = self.client(api_key='fixture-secret', name='budget',
                                  settings=Settings.from_env({'TITANOS_BODY_READ_SECONDS': '0.05'}))
        app.state.engine.target = MagicMock()
        receive = Recorder([{'type': 'http.request', 'body': b'{}', 'more_body': False}], gate=asyncio.Event())
        scope = make_scope('POST', '/api/models/load', [('authorization', 'Bearer fixture-secret'),
                                                         ('content-type', 'application/json'), ('content-length', '2')])
        status, _, body = client.portal.call(call, app, scope, receive)
        self.assertEqual(status, 408, body)
        self.assertIn('0.05 s', json.loads(body)['error']['message'])

    def test_guard_semaphore_is_sized_from_settings_per_loop(self):
        guard = server.Guard(app=None, settings=Settings.from_env({'TITANOS_BODY_READ_CONCURRENCY': '3'}))
        self.assertEqual(guard.settings.body_read_concurrency, 3)

        async def sizes():
            first = guard.body_slots()
            return first._value, guard.body_slots() is first

        value, same = asyncio.run(sizes())
        self.assertEqual((value, same), (3, True))
        # A new event loop sizes a fresh semaphore from the same Settings instance (restart-to-change semantics).
        value2, _ = asyncio.run(sizes())
        self.assertEqual(value2, 3)
        # Environment values reach the Guard through create_app -> load_settings(): the app built under
        # TITANOS_BODY_READ_CONCURRENCY=1 above proved the slot count; here the defaults are confirmed.
        app, _ = self.client(name='defaults')
        middleware = app.middleware_stack
        while middleware is not None and not isinstance(middleware, server.Guard):
            middleware = getattr(middleware, 'app', None)
        self.assertIsInstance(middleware, server.Guard)
        self.assertEqual((middleware.settings.body_read_seconds, middleware.settings.body_read_min_rate,
                          middleware.settings.body_read_concurrency, middleware.settings.body_read_wait_seconds), (30.0, 262144, 8, 2.0))


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PublicRoutes))
    print(json.dumps({'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
                      'scope': 'Guard public-route pre-read rejection and bounded body reads; in-process ASGI fixtures, no listener/model/network'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
