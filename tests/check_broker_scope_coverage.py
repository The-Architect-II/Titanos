"""WP7 scope coverage of the integrated app: every OpenAPI operation of create_app (standalone and managed role, with
the runtime-setup routes registered) and every mount or non-schema route in app.routes (/mcp, the documentation views,
the Studio assets) has a broker.scopes.ROUTE_SCOPES entry or is public; a route outside the table needs admin (403
scope_denied for a key without it, served for bootstrap; keyless loopback keeps today's open semantics).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import netguard  # noqa: E402

from titanos.broker.role import public_routes  # noqa: E402
from titanos.broker.scopes import ROUTE_SCOPES, required_scope, scope_for_template  # noqa: E402

BOOT = 'wp7-scope-bootstrap-0001'
NON_ADMIN = sorted({'inference', 'jobs', 'mcp', 'models:read', 'models:write', 'settings:read', 'cloud', 'cloud:store',
                    'route:explicit', 'delegate', 'broker:read', 'broker:admin', 'credentials:write', 'keys:admin',
                    'audit:read'})


def uncovered(app, settings) -> list[str]:
    public = public_routes(settings)
    missing = []
    for path, operations in app.openapi()['paths'].items():
        for method in operations:
            if path in public:
                continue
            if scope_for_template(method.upper(), path) is None:
                missing.append(f'{method.upper()} {path}')
    for route in app.routes:
        path = getattr(route, 'path', '')
        methods = getattr(route, 'methods', None)
        if path.rstrip('/') == '/mcp' or path.startswith('/mcp'):
            if required_scope('POST', '/mcp') != 'mcp' or ('*', '/mcp') not in ROUTE_SCOPES:
                missing.append('* /mcp')
            continue
        for method in methods or ():
            if method == 'HEAD' or path in public:
                continue
            if scope_for_template(method, path) is None:
                missing.append(f'{method} {path}')
    return sorted(set(missing))


class Coverage(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    def test_every_operation_and_mount_is_listed_or_public(self):
        for kw in ({}, {'role': 'managed'}):
            with self.subTest(role=kw.get('role', 'standalone')):
                full = W.FullApp(None, models={'chat-local': 'chat'}, settings_kw=kw)
                try:
                    self.assertEqual(uncovered(full.app, full.settings), [])
                    paths = {getattr(r, 'path', '') for r in full.app.routes}
                    self.assertIn('/mcp', paths)
                    self.assertIn('/docs/assets/{name}', paths)
                    self.assertIn('/ui/{name}', paths)
                    self.assertIn('/api/version', paths)  # every role (protocols.py; broker/routes.py in managed, where it is public)
                finally:
                    full.close()

    def test_runtime_setup_routes_are_listed(self):
        from titanos.runtime_api import register_http
        full = W.FullApp(None, models={'chat-local': 'chat'})
        try:
            class Setup:
                pass
            register_http(full.app, Setup())
            full.app.openapi_schema = None
            self.assertEqual(uncovered(full.app, full.settings), [])
        finally:
            full.close()

    def test_an_unlisted_route_needs_admin(self):
        full = W.FullApp(None, models={'chat-local': 'chat'}, api_key=BOOT)
        try:
            @full.app.get('/wp7/unlisted')
            def unlisted():
                return {'ok': True}
            self.assertEqual(required_scope('GET', '/wp7/unlisted'), 'admin')
            _, token = full.mint(*NON_ADMIN, name='wide', budget={'usd_per_day': '1'},
                                 expires_at='2030-01-01T00:00:00Z')
            c = full.http
            denied = c.get('/wp7/unlisted', headers=W.bearer(token))
            self.assertEqual((denied.status_code, denied.json()['error']['code']), (403, 'scope_denied'))
            self.assertEqual(c.get('/wp7/unlisted', headers=W.bearer(BOOT)).status_code, 200)
            self.assertEqual(c.get('/v1/models', headers=W.bearer(token)).status_code, 200)
        finally:
            full.close()
        keyless = W.FullApp(None, models={'chat-local': 'chat'})
        try:
            @keyless.app.get('/wp7/unlisted')
            def open_route():
                return {'ok': True}
            self.assertEqual(keyless.http.get('/wp7/unlisted').status_code, 200)  # keyless loopback: unchanged
        finally:
            keyless.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
