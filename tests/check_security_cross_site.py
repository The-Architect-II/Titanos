"""Keyless loopback mode refuses cross-site browser actions before any inventory or engine call.

Actual create_app/Guard/TrustedHost with a MagicMock engine and a patched device inventory;
attack-shaped headers are inert fixtures. No listener, model, device, network, or running service.
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fastapi.testclient import TestClient  # noqa: E402
from titanos import devices, server  # noqa: E402
from titanos.config import Config, ModelSpec  # noqa: E402

CROSS = {'Origin': 'http://evil.example', 'Sec-Fetch-Site': 'cross-site'}
SAME = {'Origin': 'http://127.0.0.1:8000', 'Sec-Fetch-Site': 'same-origin'}
PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 16  # inert bytes; the request must be refused before any decoding
WAV = b'RIFF' + b'\x00' * 40  # inert bytes for a multipart audio upload that must be refused before any decoding
PROOFS = []  # navigation and cross-site outcomes recorded by the navigation test and printed with the summary


def make_scope(method, path, headers=(), host='127.0.0.1:8000', extra=None):
    raw = [(b'host', host.encode('latin-1'))] + [(k.lower().encode('latin-1'), v.encode('latin-1')) for k, v in headers]
    return {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method, 'scheme': 'http',
            'path': path, 'raw_path': path.encode(), 'query_string': b'', 'root_path': '', 'headers': raw,
            'client': ('127.0.0.1', 50000), 'server': ('127.0.0.1', 8000), **(extra or {})}


async def call(app, scope, body=b''):
    sent = []
    chunks = [{'type': 'http.request', 'body': body, 'more_body': False}]

    async def receive():
        return chunks.pop(0) if chunks else {'type': 'http.disconnect'}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    start = next(m for m in sent if m['type'] == 'http.response.start')
    return start['status'], b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body')


class CrossSite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='titanos-security-cross-site-')
        self.addCleanup(self.tmp.cleanup)
        self.inventory_calls = []

        def fake_inventory(refresh=False, backend='all'):
            self.inventory_calls.append((refresh, backend))
            return {'object': 'list', 'data': [], 'fixture': True}
        patcher = patch.object(devices, 'inventory', fake_inventory)
        patcher.start()
        self.addCleanup(patcher.stop)

    def client(self, api_key=None, base_url='http://127.0.0.1:8000', name='app'):
        config = Config(models={'m': ModelSpec(path='./absent', task='chat')})  # never loaded; the engine is a MagicMock
        app = server.create_app(config, api_key=api_key, output_dir=Path(self.tmp.name) / name)
        engine = MagicMock(name='engine')
        engine.load.return_value = {'model': 'm', 'status': 'loaded'}
        app.state.engine.target = engine
        client = TestClient(app, base_url=base_url)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return app, client, engine

    def assert_refused(self, reply, engine):
        self.assertEqual(reply.status_code, 403, reply.text)
        self.assertEqual(reply.json()['error']['code'], 'cross_site_refused')
        self.assertEqual(engine.mock_calls, [])
        self.assertEqual(self.inventory_calls, [])

    def test_cross_site_refresh_and_multipart_inference_refused_before_work(self):
        app, client, engine = self.client()
        self.assert_refused(client.get('/api/devices', params={'refresh': 'true'}, headers=CROSS), engine)
        self.assert_refused(client.get('/api/assignments/devices', params={'refresh': 'true'}, headers=CROSS), engine)
        self.assert_refused(client.post('/v1/images/edits', data={'model': 'm', 'prompt': 'x'},
                                        files={'image': ('a.png', PNG, 'image/png')}, headers=CROSS), engine)
        self.assert_refused(client.post('/v1/videos', data={'model': 'm', 'prompt': 'x'},
                                        files={'input_reference': ('a.png', PNG, 'image/png')}, headers=CROSS), engine)
        self.assert_refused(client.post('/api/models/load', json={'model': 'm'}, headers=CROSS), engine)
        self.assert_refused(client.post('/api/jobs', json={'model': 'm', 'task': 'image', 'prompt': 'x'}, headers=CROSS), engine)
        self.assert_refused(client.post('/api/models/register', json={}, headers=CROSS), engine)
        self.assert_refused(client.post('/mcp', json={}, headers=CROSS), engine)
        # Origin alone (no Sec-Fetch-Site) and Sec-Fetch-Site alone are each sufficient.
        self.assert_refused(client.post('/api/models/load', json={'model': 'm'}, headers={'Origin': 'http://evil.example'}), engine)
        self.assert_refused(client.get('/api/devices', params={'refresh': 'true'}, headers={'Sec-Fetch-Site': 'cross-site'}), engine)
        self.assert_refused(client.get('/api/devices', params={'refresh': 'true'}, headers={'Sec-Fetch-Site': 'same-site'}), engine)

    def test_null_https_and_port_mismatched_origins_refused(self):
        app, client, engine = self.client()
        for origin in ('null', 'http://127.0.0.1:8001', 'http://127.0.0.1', 'http://localhost:9000', 'https://127.0.0.1:8000',
                       'http://127.0.0.1.evil.example:8000', 'http://127.0.0.1:8000.evil.example', 'http://[::1]:8001', ''):
            with self.subTest(origin=origin):
                self.assert_refused(client.post('/api/models/load', json={'model': 'm'}, headers={'Origin': origin}), engine)
        # Same-origin Origin with a non-same-origin Sec-Fetch-Site is still refused.
        self.assert_refused(client.post('/api/models/load', json={'model': 'm'},
                                        headers={'Origin': 'http://127.0.0.1:8000', 'Sec-Fetch-Site': 'same-site'}), engine)

    def test_same_origin_studio_passes_to_the_mock(self):
        app, client, engine = self.client()
        reply = client.get('/api/devices', params={'refresh': 'true'}, headers=SAME)
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(self.inventory_calls, [(True, 'all')])
        reply = client.post('/api/models/load', json={'model': 'm'}, headers=SAME)
        self.assertEqual(reply.status_code, 200, reply.text)
        engine.load.assert_called_once_with('m', None)
        # Browser GET fetches carry Sec-Fetch-Site without Origin.
        reply = client.get('/api/devices', params={'refresh': 'true'}, headers={'Sec-Fetch-Site': 'same-origin'})
        self.assertEqual(reply.status_code, 200, reply.text)
        # Origin comparison is case-insensitive.
        reply = client.post('/api/models/load', json={'model': 'm'}, headers={'Origin': 'HTTP://127.0.0.1:8000'})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(engine.load.call_count, 2)

    def test_localhost_origin_matches_localhost_host(self):
        app, client, engine = self.client(base_url='http://localhost:8000', name='localhost')
        reply = client.post('/api/models/load', json={'model': 'm'}, headers={'Origin': 'http://LocalHost:8000'})
        self.assertEqual(reply.status_code, 200, reply.text)
        engine.load.assert_called_once_with('m', None)
        engine.reset_mock()  # the refused request below must add nothing
        self.assert_refused(client.post('/api/models/load', json={'model': 'm'}, headers={'Origin': 'http://localhost:8001'}), engine)
        # Host without an explicit port means port 80: only a port-less loopback Origin matches.
        app80, client80, engine80 = self.client(base_url='http://127.0.0.1', name='port80')
        self.assertEqual(client80.post('/api/models/load', json={'model': 'm'}, headers={'Origin': 'http://127.0.0.1'}).status_code, 200)
        self.assertEqual(client80.post('/api/models/load', json={'model': 'm'}, headers={'Origin': 'http://127.0.0.1:8000'}).status_code, 403)

    def test_header_less_local_clients_keep_working(self):
        app, client, engine = self.client()
        self.assertEqual(client.get('/api/devices', params={'refresh': 'true'}).status_code, 200)
        self.assertEqual(self.inventory_calls, [(True, 'all')])
        self.assertEqual(client.post('/api/models/load', json={'model': 'm'}).status_code, 200)
        engine.load.assert_called_once_with('m', None)
        # In-process MCP dispatch marks its scope; a browser cannot set an ASGI scope field.
        scope = make_scope('POST', '/api/models/load', [('content-type', 'application/json'), ('content-length', '13'),
                                                         ('origin', 'http://evil.example')], extra={'titanos.internal': True})
        status, body = client.portal.call(call, app, scope, b'{"model":"m"}')
        self.assertEqual(status, 200, body)
        self.assertEqual(engine.load.call_count, 2)

    def test_sec_fetch_site_none_is_user_navigation_for_safe_methods_only(self):
        app, client, engine = self.client()
        navigation = {'Sec-Fetch-Site': 'none', 'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Dest': 'document'}
        self.assertEqual(client.get('/healthz', headers=navigation).status_code, 200)
        self.assertEqual(client.get('/', headers=navigation).status_code, 200)
        self.assertEqual(client.get('/ui/app.js', headers=navigation).status_code, 200)
        # Public GETs are readable from any site; a cross-site POST to them is a 405 from the public-route rule.
        self.assertEqual(client.get('/healthz', headers=CROSS).status_code, 200)
        self.assertEqual(client.post('/healthz', headers=CROSS).status_code, 405)
        # 'none' is only set for user-initiated navigation (address bar, bookmark, pasted link), which no page can
        # produce, so safe methods on non-public routes pass: docs, OpenAPI, readiness, API reads, artifact links.
        for path in ('/docs', '/openapi.json', '/readyz', '/api/info', '/v1/models'):
            with self.subTest(path=path):
                self.assertEqual(client.get(path, headers=navigation).status_code, 200, path)
        self.assertEqual(client.get('/v1/files/absent', headers=navigation).status_code, 404)  # route reached, not 403
        self.assertEqual(client.get('/api/devices', params={'refresh': 'true'}, headers={'Sec-Fetch-Site': 'none'}).status_code, 200)
        self.assertEqual(self.inventory_calls, [(True, 'all')])
        self.inventory_calls.clear()
        engine.reset_mock()  # the navigations above legitimately read the catalog; refusals below must add nothing
        # State-changing methods with 'none' are refused, as is 'none' paired with a foreign Origin on any method.
        self.assert_refused(client.post('/api/models/load', json={'model': 'm'}, headers={'Sec-Fetch-Site': 'none'}), engine)
        self.assert_refused(client.post('/api/models/load', json={'model': 'm'},
                                        headers={'Sec-Fetch-Site': 'none', 'Origin': 'http://evil.example'}), engine)
        self.assert_refused(client.get('/api/devices', params={'refresh': 'true'},
                                       headers={'Sec-Fetch-Site': 'none', 'Origin': 'http://evil.example'}), engine)
        self.assert_refused(client.delete('/api/models/registrations/m', headers={'Sec-Fetch-Site': 'none'}), engine)
        # A same-origin Origin with 'none' (a POST from our own page can never carry 'none', but it must not weaken anything).
        self.assertEqual(client.post('/api/models/load', json={'model': 'm'},
                                     headers={'Sec-Fetch-Site': 'none', 'Origin': 'http://127.0.0.1:8000'}).status_code, 403)

    def prove(self, form, reply, expected, method, path):
        """Record one navigation/cross-site outcome (printed with the summary) and assert the expected status."""
        PROOFS.append({'form': form, 'request': f'{method} {path}', 'status': reply.status_code, 'expected': expected,
                       'outcome': 'passes' if reply.status_code < 400 else 'refused'})
        self.assertEqual(reply.status_code, expected, (form, method, path, reply.text[:200]))
        return reply

    def test_direct_navigation_and_pasted_links_pass_while_cross_site_actions_fail(self):
        app, client, engine = self.client(name='navigation')
        outputs = Path(self.tmp.name) / 'navigation'  # create_app(output_dir=...) is where /v1/files serves from
        outputs.mkdir(parents=True, exist_ok=True)
        asset = 'a1' * 16 + '.png'  # the name shape /v1/images/generations mints (32 hex + png/mp4/wav)
        (outputs / asset).write_bytes(PNG)
        navigation = {'Sec-Fetch-Site': 'none', 'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Dest': 'document'}  # no Origin
        # Direct browser navigation (address bar, bookmark): Sec-Fetch-Site none and no Origin.
        self.prove('direct navigation (Sec-Fetch-Site: none, no Origin)', client.get('/docs', headers=navigation), 200, 'GET', '/docs')
        reply = self.prove('direct navigation (Sec-Fetch-Site: none, no Origin)', client.get('/openapi.json', headers=navigation), 200, 'GET', '/openapi.json')
        self.assertIn('/v1/files/{name}', reply.json()['paths'])
        # A pasted link to a published asset, and the same asset embedded by the same-origin Studio page.
        reply = self.prove('pasted link (Sec-Fetch-Site: none, no Origin)', client.get('/v1/files/' + asset, headers=navigation), 200, 'GET', '/v1/files/<id>')
        self.assertEqual(reply.content, PNG)
        reply = self.prove('same-origin page (Sec-Fetch-Site: same-origin)', client.get('/v1/files/' + asset,
                           headers={'Sec-Fetch-Site': 'same-origin', 'Sec-Fetch-Mode': 'no-cors', 'Sec-Fetch-Dest': 'image'}), 200, 'GET', '/v1/files/<id>')
        self.assertEqual(reply.content, PNG)
        # Header-less local clients (CLI, SDK, MCP client) reach the same routes.
        for path in ('/docs', '/openapi.json', '/v1/files/' + asset):
            self.prove('header-less local client', client.get(path), 200, 'GET', path if not path.startswith('/v1/files/') else '/v1/files/<id>')
        self.assertEqual(engine.mock_calls, [])
        self.assertEqual(self.inventory_calls, [])
        # Cross-site action attempts are refused with 403 before any inventory or engine call.
        reply = self.prove('cross-site fetch (Sec-Fetch-Site: cross-site)', client.get('/api/devices', params={'refresh': 'true'},
                           headers={'Sec-Fetch-Site': 'cross-site'}), 403, 'GET', '/api/devices?refresh=true')
        self.assertEqual(reply.json()['error']['code'], 'cross_site_refused')
        self.assertEqual(self.inventory_calls, [])
        reply = self.prove('cross-site multipart POST (Origin: http://attacker.example)', client.post('/v1/images/edits', data={'model': 'm', 'prompt': 'x'},
                           files={'image': ('a.png', PNG, 'image/png')}, headers={'Origin': 'http://attacker.example'}), 403, 'POST', '/v1/images/edits')
        self.assertEqual(reply.json()['error']['code'], 'cross_site_refused')
        reply = self.prove('cross-site multipart POST (Origin: http://attacker.example, Sec-Fetch-Site: cross-site)',
                           client.post('/v1/audio/transcriptions', data={'model': 'm'}, files={'file': ('a.wav', WAV, 'audio/wav')},
                                       headers={'Origin': 'http://attacker.example', 'Sec-Fetch-Site': 'cross-site'}), 403, 'POST', '/v1/audio/transcriptions')
        self.assertEqual(reply.json()['error']['code'], 'cross_site_refused')
        # A published asset hot-linked from another site is refused too (the artifact route is not public).
        self.prove('cross-site hotlink (Sec-Fetch-Site: cross-site)', client.get('/v1/files/' + asset,
                   headers={'Sec-Fetch-Site': 'cross-site', 'Sec-Fetch-Mode': 'no-cors', 'Sec-Fetch-Dest': 'image'}), 403, 'GET', '/v1/files/<id>')
        self.assertEqual(engine.mock_calls, [], 'the mocked engine was never called by a cross-site request')
        self.assertEqual(self.inventory_calls, [], 'the patched inventory was never called by a cross-site request')

    def test_duplicate_origin_headers_are_refused(self):
        app, client, engine = self.client()
        for order in (('http://evil.example', 'http://127.0.0.1:8000'), ('http://127.0.0.1:8000', 'http://evil.example'),
                      ('http://127.0.0.1:8000', 'http://127.0.0.1:8000')):
            with self.subTest(order=order):
                scope = make_scope('POST', '/api/models/load', [('content-type', 'application/json'), ('content-length', '13')]
                                   + [('origin', value) for value in order])
                status, body = client.portal.call(call, app, scope, b'{"model":"m"}')
                self.assertEqual(status, 403, body)
                self.assertEqual(json.loads(body)['error']['code'], 'cross_site_refused')
        self.assertEqual(engine.mock_calls, [])

    def test_referer_is_a_fallback_signal_for_pre_fetch_metadata_browsers(self):
        app, client, engine = self.client()
        # Legacy browser <img>/simple GET: no Origin, no Sec-Fetch-Site, but a Referer naming the hostile page.
        self.assert_refused(client.get('/api/devices', params={'refresh': 'true'}, headers={'Referer': 'http://evil.example/'}), engine)
        self.assert_refused(client.get('/api/devices', params={'refresh': 'true'},
                                       headers={'Referer': 'http://127.0.0.1:8001/studio'}), engine)
        self.assert_refused(client.post('/api/models/load', json={'model': 'm'}, headers={'Referer': 'https://127.0.0.1:8000/'}), engine)
        self.assert_refused(client.get('/api/devices', params={'refresh': 'true'}, headers={'Referer': 'not a url'}), engine)
        self.assert_refused(client.get('/api/devices', params={'refresh': 'true'},
                                       headers={'Referer': 'http://127.0.0.1:8000@evil.example/'}), engine)
        # A blank Referer carries no signal and is treated as absent (header-less local client).
        self.assertEqual(client.get('/api/devices', params={'refresh': 'true'}, headers={'Referer': ''}).status_code, 200)
        self.inventory_calls.clear()
        # Our own page as Referer passes, with or without Fetch Metadata; public routes stay readable.
        self.assertEqual(client.get('/api/devices', params={'refresh': 'true'},
                                    headers={'Referer': 'http://127.0.0.1:8000/'}).status_code, 200)
        self.assertEqual(client.get('/api/devices', params={'refresh': 'true'},
                                    headers={'Referer': 'http://LOCALHOST:8000/?x=1', 'Sec-Fetch-Site': 'same-origin'}).status_code, 200)
        self.assertEqual(self.inventory_calls, [(True, 'all'), (True, 'all')])
        self.assertEqual(client.get('/healthz', headers={'Referer': 'http://evil.example/'}).status_code, 200)
        # Sec-Fetch-Site: same-origin is authoritative when present, so an odd Referer does not override it.
        self.assertEqual(client.post('/api/models/load', json={'model': 'm'},
                                     headers={'Sec-Fetch-Site': 'same-origin', 'Referer': 'http://evil.example/'}).status_code, 200)
        engine.load.assert_called_once_with('m', None)
        self.assertEqual(server.referer_origin('http://127.0.0.1:8000/a/b?c'), 'http://127.0.0.1:8000')
        self.assertIsNone(server.referer_origin('about:blank'))
        self.assertIsNone(server.referer_origin(''))

    def test_keyed_mode_keeps_bearer_as_the_gate(self):
        app, client, engine = self.client(api_key='fixture-secret', name='keyed')
        # Existing behaviour is unchanged: the bearer decides; the keyless cross-site check is not applied.
        self.assertEqual(client.post('/api/models/load', json={'model': 'm'}, headers=CROSS).status_code, 401)
        reply = client.post('/api/models/load', json={'model': 'm'}, headers={**CROSS, 'Authorization': 'Bearer fixture-secret'})
        self.assertEqual(reply.status_code, 200, reply.text)
        engine.load.assert_called_once_with('m', None)

    def test_origin_helpers(self):
        self.assertTrue(server.origin_is_own('http://127.0.0.1:8000', 8000))
        self.assertTrue(server.origin_is_own('http://LOCALHOST:8000', 8000))
        self.assertTrue(server.origin_is_own('http://[::1]:8000', 8000))
        self.assertTrue(server.origin_is_own('http://localhost', 80))
        for bad in ('null', 'http://[::1]:8001', 'http://localhost:80', 'https://localhost:8000', 'http://127.0.0.1:8000/', 'http://', 'file://'):
            self.assertFalse(server.origin_is_own(bad, 8000), bad)
        self.assertEqual(server.own_port({}, {b'host': b'localhost'}), 80)
        self.assertEqual(server.own_port({}, {b'host': b'127.0.0.1:8000'}), 8000)
        self.assertEqual(server.own_port({}, {b'host': b'[::1]:9'}), 9)
        self.assertEqual(server.own_port({'server': ('127.0.0.1', 7000)}, {}), 7000)
        self.assertEqual(server.own_port({}, {}), 80)
        self.assertEqual(set(server.LOOPBACK_ORIGIN_HOSTS), {'127.0.0.1', 'localhost', '[::1]'})


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CrossSite))
    print('Navigation forms that pass and cross-site forms that fail (keyless loopback mode, in-process ASGI):')
    for proof in PROOFS:
        print(f"  {proof['outcome']:7} {proof['status']}  {proof['form']}: {proof['request']}")
    print(json.dumps({'passed': result.wasSuccessful(), 'tests_run': result.testsRun, 'navigation_proofs': PROOFS,
                      'scope': 'Keyless loopback cross-site refusal at the Guard; MagicMock engine and patched inventory; no listener/model/network. '
                               'Residual not covered by design: a pre-Fetch-Metadata browser sending neither Origin nor Referer is indistinguishable '
                               'from a local non-browser client and passes.'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
