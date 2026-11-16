"""WP7 inbound authentication over the integrated app (KeylessHostGuard, Guard + Authenticator, BrokerDispatch).

- keyless mode keeps today's semantics: an OpenAI-SDK-style `Bearer local` is ignored and accepted;
- keyed mode: a wrong key from loopback is 401 (the throttle never needs a credential-less request);
- scope matrix samples: a key answers only the routes its scopes cover (403 scope_denied before the body is read);
- the /api/info settings block needs settings:read;
- credential PUT and key POST are refused in keyless mode (403 keyless_mode_refused) and from a non-loopback peer
  without TLS (403 transport_insecure); TLS (an https scope) from the same peer is accepted;
- a forged X-Forwarded-Proto: https from a non-loopback peer does not satisfy transport_secure: the app runs under
  uvicorn in a subprocess with the pinned proxy settings (proxy_headers off, and on with forwarded_allow_ips 127.0.0.1);
- KeylessHostGuard: the invalid-Host refusal applies only while nothing is enforced; after the last key is revoked at
  runtime enforcement stays on (sticky), so every Host reaches the credential check;
- a keyless PUT that changes policy.anonymous_cloud is 403 keyless_mode_refused;
- an audit failure (AuditChainError) during a key write answers a 5xx envelope and persists nothing; the same holds for
  broker.json writes (the file and the live snapshot stay unchanged) and vault writes (put, rotate, delete), because
  every admin write appends its audit record before it changes anything;
- a file: credential reference that the vault refuses is named by its file name only, never its path;
- a filelock timeout on the audit lock is 503 audit_unavailable; on any other state lock it is 503 server_busy;
- mounted MCP (threat T7): a tool call over /mcp runs as the outer request's principal, never the bootstrap key: a key
  holding only 'mcp' cannot chat, list jobs or read the settings block through the tools; a key holding the inner
  route's scope can, also in keys-only mode (no bootstrap key) where every tool used to fail with 401.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import netguard  # noqa: E402
from titanos.broker.audit import AuditChainError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOOT = 'wp7-auth-bootstrap-0001'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
LAN = '192.0.2.10'  # TEST-NET-1: never routed; only used as a scope peer address


class Case(unittest.TestCase):
    api_key = BOOT
    document = None

    def setUp(self):
        netguard.reset_attempts()
        self.full = W.FullApp(self.document, models={'chat-local': 'chat', 'embed-local': 'embeddings'},
                              api_key=self.api_key)
        self.c = self.full.http

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')


class Keyless(Case):
    api_key = None

    def test_placeholder_bearer_is_ignored_in_keyless_mode(self):
        for headers in ({}, {'Authorization': 'Bearer local'}, {'Authorization': 'Basic Zm9vOmJhcg=='}):
            reply = self.c.get('/v1/models', headers=headers)
            self.assertEqual(reply.status_code, 200, (headers, reply.text[:200]))
        chat = self.c.post('/v1/chat/completions', json={'model': 'chat-local', 'messages': CHAT},
                           headers={'Authorization': 'Bearer local'})
        self.assertEqual(chat.status_code, 200, chat.text[:200])
        self.assertIn('settings', self.c.get('/api/info').json())  # keyless loopback holds settings:read

    def test_keyless_key_and_credential_writes_and_guarded_policy_fields(self):
        mint = self.c.post('/api/keys', json={'name': 'k', 'scopes': ['inference']})
        self.assertEqual((mint.status_code, mint.json()['error']['code']), (403, 'keyless_mode_refused'))
        cred = self.c.put('/api/broker/credentials/oa', json={'upstream': 'openai', 'secret': 'x' * 32})
        self.assertEqual((cred.status_code, cred.json()['error']['code']), (403, 'keyless_mode_refused'))
        self.assertNotIn('x' * 32, cred.text)
        lan = self.c.put('/api/broker/policy', json={'cloud_policy': 'off', 'allow_http_lan': True})
        self.assertEqual((lan.status_code, lan.json()['error']['code']), (403, 'keyless_mode_refused'))
        anon = self.c.put('/api/broker/policy', json={'cloud_policy': 'off', 'anonymous_cloud': True})
        self.assertEqual((anon.status_code, anon.json()['error']['code']), (403, 'keyless_mode_refused'))
        self.assertFalse((self.full.home / 'broker.json').exists())
        ok = self.c.put('/api/broker/policy', json={'cloud_policy': 'off', 'default_location': 'local_only'})
        self.assertEqual(ok.status_code, 200, ok.text[:200])
        self.assertEqual(ok.json()['default_location'], 'local_only')

    def test_host_guard_until_enforcement_and_sticky_after_revocation(self):
        bad = self.c.get('/v1/models', headers={'Host': 'evil.example'})
        self.assertEqual((bad.status_code, bad.text), (400, 'Invalid host header'))
        key_id, token = self.full.mint('inference', 'models:read', name='first')
        self.full.broker.keys._refresh(force=True)
        self.assertTrue(self.full.broker.authenticator.enforced)
        with_key = self.c.get('/v1/models', headers={'Host': 'broker.example', **W.bearer(token)})
        self.assertEqual(with_key.status_code, 200)
        self.full.broker.keys.revoke(key_id, actor='test')
        self.full.broker.keys._refresh(force=True)
        self.assertTrue(self.full.broker.authenticator.enforced)  # any record keeps enforcement on
        after = self.c.get('/v1/models', headers={'Host': 'evil.example'})
        self.assertEqual(after.status_code, 401)  # the Host passes; the credential check answers
        revoked = self.c.get('/v1/models', headers=W.bearer(token))
        self.assertEqual((revoked.status_code, revoked.json()['error']['code']), (401, 'key_revoked'))


class Keyed(Case):
    def test_wrong_key_from_loopback_is_401(self):
        reply = self.c.get('/v1/models', headers=W.bearer('wrong-key-value-123'))
        self.assertEqual((reply.status_code, reply.json()['error']['code']), (401, 'authentication_error'))
        self.assertEqual(self.c.get('/v1/models').status_code, 401)
        self.assertEqual(self.c.get('/v1/models', headers=W.bearer(BOOT)).status_code, 200)
        anthropic = self.c.post('/v1/messages', json={'model': 'chat-local', 'max_tokens': 8, 'messages': CHAT},
                                headers={'x-api-key': 'wrong-key-value-123'})
        self.assertEqual((anthropic.status_code, anthropic.json()['type']), (401, 'error'))

    def test_scope_matrix_samples(self):
        _, chat = self.full.mint('inference:chat', name='chat')
        _, reader = self.full.mint('models:read', name='reader')
        cases = [
            (chat, 'POST', '/v1/chat/completions', {'model': 'chat-local', 'messages': CHAT}, 200),
            (chat, 'POST', '/v1/embeddings', {'model': 'embed-local', 'input': 'x'}, 403),
            (chat, 'GET', '/v1/models', None, 403),
            (chat, 'POST', '/api/models/load', {'model': 'chat-local'}, 403),
            (chat, 'GET', '/api/keys', None, 403),
            (chat, 'GET', '/api/broker', None, 403),
            (chat, 'GET', '/api/info', None, 200),
            (chat, 'GET', '/api/keys/self', None, 200),
            (reader, 'GET', '/v1/models', None, 200),
            (reader, 'POST', '/v1/chat/completions', {'model': 'chat-local', 'messages': CHAT}, 403),
            (reader, 'GET', '/api/audit', None, 403),
            (BOOT, 'GET', '/api/audit', None, 200),
            (BOOT, 'GET', '/api/keys', None, 200),
        ]
        for token, method, path, body, status in cases:
            with self.subTest(path=path, method=method, status=status):
                reply = self.c.request(method, path, json=body, headers=W.bearer(token))
                self.assertEqual(reply.status_code, status, reply.text[:200])
                if status == 403:
                    self.assertEqual(reply.json()['error']['code'], 'scope_denied')
        me = self.c.get('/api/keys/self', headers=W.bearer(chat)).json()
        self.assertEqual((me['kind'], me['scopes']), ('key', ['inference:chat']))

    def test_settings_block_needs_settings_read(self):
        _, plain = self.full.mint('inference', name='plain')
        _, settings = self.full.mint('inference', 'settings:read', name='settings')
        hidden = self.c.get('/api/info', headers=W.bearer(plain)).json()
        self.assertNotIn('settings', hidden)
        self.assertEqual((hidden['role'], hidden['orchestrated'], hidden['standalone']), ('standalone', False, True))
        self.assertIn('broker', hidden)
        self.assertIn('settings', self.c.get('/api/info', headers=W.bearer(settings)).json())
        self.assertIn('settings', self.c.get('/api/info', headers=W.bearer(BOOT)).json())

    def test_transport_secure_for_key_and_credential_writes(self):
        lan = self.full.client(peer=LAN)
        mint = lan.post('/api/keys', json={'name': 'k', 'scopes': ['inference']}, headers=W.bearer(BOOT))
        self.assertEqual((mint.status_code, mint.json()['error']['code']), (403, 'transport_insecure'))
        cred = lan.put('/api/broker/credentials/oa', json={'upstream': 'openai', 'secret': 'y' * 32},
                       headers=W.bearer(BOOT))
        self.assertEqual((cred.status_code, cred.json()['error']['code']), (403, 'transport_insecure'))
        self.assertNotIn('y' * 32, cred.text)
        forged = lan.post('/api/keys', json={'name': 'k', 'scopes': ['inference']},
                          headers={**W.bearer(BOOT), 'X-Forwarded-Proto': 'https', 'X-Forwarded-For': '127.0.0.1'})
        self.assertEqual(forged.status_code, 403)
        tls = self.full.client(peer=LAN, scheme='https')
        ok = tls.post('/api/keys', json={'name': 'k', 'scopes': ['inference']}, headers=W.bearer(BOOT))
        self.assertEqual(ok.status_code, 201, ok.text[:200])
        self.assertEqual(ok.headers['cache-control'], 'no-store')
        self.assertTrue(ok.json()['secret'].startswith('tk_'))
        local = self.c.post('/api/keys', json={'name': 'k2', 'scopes': ['inference']}, headers=W.bearer(BOOT))
        self.assertEqual(local.status_code, 201)

    def test_audit_failure_is_a_5xx_envelope_and_persists_nothing(self):
        from titanos.broker.audit import AuditChainError
        before = len(self.full.broker.keys.list())
        with mock.patch.object(type(self.full.broker.audit), 'event', side_effect=AuditChainError('tail unreadable')):
            reply = self.c.post('/api/keys', json={'name': 'k', 'scopes': ['inference']}, headers=W.bearer(BOOT))
        self.assertEqual((reply.status_code, reply.json()['error']['code']), (503, 'audit_unavailable'))
        self.assertEqual(len(self.full.broker.keys.list()), before)


class AuditFirst(Case):
    """Every admin write appends its audit record before it changes anything (finding: broker.json was replaced first)."""
    document = {'policy': {'cloud_policy': 'off'},
                'upstreams': {'openai': {'profile': 'openai', 'base_url': 'http://127.0.0.1:9/v1', 'enabled': False}}}

    def broken_audit(self):
        return mock.patch.object(type(self.full.broker.audit), 'event', side_effect=AuditChainError('tail unreadable'))

    def test_config_writes_with_an_unavailable_audit_change_nothing(self):
        path = self.full.home / 'broker.json'
        before = path.read_bytes()
        version = self.full.broker.snapshot().version
        with self.broken_audit():
            policy = self.c.put('/api/broker/policy', json={'cloud_policy': 'off', 'default_location': 'prefer_local'},
                                headers=W.bearer(BOOT))
            upstream = self.c.put('/api/broker/upstreams/extra', headers=W.bearer(BOOT),
                                  json={'profile': 'openai', 'base_url': 'http://127.0.0.1:9/v1', 'enabled': False})
        for reply in (policy, upstream):
            self.assertEqual((reply.status_code, reply.json()['error']['code']), (503, 'audit_unavailable'))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.full.broker.snapshot().version, version)
        self.assertEqual(self.c.get('/api/broker/policy', headers=W.bearer(BOOT)).json()['default_location'],
                         self.full.broker.snapshot().config.policy.default_location)
        ok = self.c.put('/api/broker/policy', json={'cloud_policy': 'off', 'default_location': 'prefer_local'},
                        headers=W.bearer(BOOT))
        self.assertEqual((ok.status_code, ok.json()['default_location']), (200, 'prefer_local'))
        records = self.full.broker.audit.query(kind='policy.write')
        self.assertEqual(len(records), 1)  # only the write that happened is audited

    def test_vault_writes_with_an_unavailable_audit_change_nothing(self):
        vault = self.full.broker.vault
        body = {'upstream': 'openai', 'secret': 'sk-test-' + 'a' * 32, 'validate': False}
        with self.broken_audit():
            refused = self.c.put('/api/broker/credentials/oa', json=body, headers=W.bearer(BOOT))
        self.assertEqual((refused.status_code, refused.json()['error']['code']), (503, 'audit_unavailable'))
        self.assertEqual(vault.records(), [])
        created = self.c.put('/api/broker/credentials/oa', json=body, headers=W.bearer(BOOT))
        self.assertEqual(created.status_code, 201, created.text[:300])
        fingerprint = vault.record('oa').fingerprint
        with self.broken_audit():
            rotate = self.c.post('/api/broker/credentials/oa/rotate', json={'secret': 'sk-test-' + 'b' * 32},
                                 headers=W.bearer(BOOT))
            delete = self.c.delete('/api/broker/credentials/oa', headers=W.bearer(BOOT))
        for reply in (rotate, delete):
            self.assertEqual((reply.status_code, reply.json()['error']['code']), (503, 'audit_unavailable'))
        self.assertEqual(vault.record('oa').fingerprint, fingerprint)
        self.assertNotIn('a' * 32, created.text + rotate.text + delete.text)

    def test_file_reference_refusals_name_the_file_only(self):
        missing = self.full.root / 'private-dir' / 'provider.cred'
        reply = self.c.put('/api/broker/credentials/oa', headers=W.bearer(BOOT),
                           json={'upstream': 'openai', 'file': str(missing), 'validate': False})
        self.assertEqual((reply.status_code, reply.json()['error']['code']), (503, 'credential_unavailable'))
        message = reply.json()['error']['message']
        self.assertIn('provider.cred', message)
        self.assertNotIn('private-dir', message)
        self.assertNotIn(str(self.full.root), reply.text)
        self.assertEqual(self.full.broker.vault.records(), [])
        last = self.full.broker.audit.query(kind='credential.put')[-1]
        self.assertEqual(last['detail'], {'ok': False, 'code': 'credential_unavailable'})  # the follow-up record

    def test_lock_timeouts_are_typed_by_lock(self):
        from filelock import Timeout
        from titanos.broker import admin
        cases = ((str(self.full.home / 'broker' / 'audit' / '.audit.lock'), 'audit_unavailable'),
                 (str(self.full.home / 'broker.json.lock'), 'server_busy'))
        for lock_file, code in cases:
            with self.subTest(code=code), mock.patch.object(admin, 'update_broker_config', side_effect=Timeout(lock_file)):
                reply = self.c.put('/api/broker/policy', json={'cloud_policy': 'off'}, headers=W.bearer(BOOT))
                self.assertEqual((reply.status_code, reply.json()['error']['code']), (503, code))
                self.assertNotIn(str(self.full.home), reply.text)


class HostCredentialSources(Case):
    """env: and file: credential sources name values on the server host: TITANOS_* names (the bootstrap key above all)
    are never an upstream credential, and naming any host source needs the admin scope, so a credentials:write plus
    broker:admin key (which also controls base_url) cannot send a host value to an origin of its choosing."""
    document = {'policy': {'cloud_policy': 'explicit'},
                'upstreams': {'openai': {'profile': 'openai', 'base_url': 'http://127.0.0.1:9/v1', 'enabled': False}}}

    def put(self, token, body, cred='oa'):
        return self.c.put(f'/api/broker/credentials/{cred}', json={'upstream': 'openai', 'validate': False, **body},
                          headers=W.bearer(token))

    def test_reserved_names_and_host_sources_need_admin(self):
        _, operator = self.full.mint('credentials:write', 'broker:admin', name='operator')
        with mock.patch.dict(os.environ, {'WP9_HOST_SECRET_VAR': 'test-host-secret-not-a-key-0001'}):
            for body in ({'env': 'TITANOS_API_KEY'}, {'env': 'WP9_HOST_SECRET_VAR'},
                         {'file': str(Path(self.full.root) / 'secret.txt')}):
                with self.subTest(body=body):
                    reply = self.put(operator, body)
                    self.assertEqual((reply.status_code, reply.json()['error']['code']), (403, 'scope_denied'))
            self.assertEqual(self.full.broker.vault.records(), [])
            supplied = self.put(operator, {'secret': 'test-openai-not-a-key-' + 'ab' * 8})
            self.assertEqual(supplied.status_code, 201, supplied.text[:300])
            reserved = self.put(BOOT, {'env': 'TITANOS_API_KEY'}, cred='boot')
            self.assertEqual((reserved.status_code, reserved.json()['error']['code']), (400, 'invalid_request'))
            self.assertIn('TITANOS_', reserved.json()['error']['message'])
            ok = self.put(BOOT, {'env': 'WP9_HOST_SECRET_VAR'}, cred='host')
            self.assertEqual(ok.status_code, 201, ok.text[:300])
            self.assertEqual(ok.json()['env_var'], 'WP9_HOST_SECRET_VAR')
        rotate = self.c.post('/api/broker/credentials/oa/rotate', json={'env': 'WP9_HOST_SECRET_VAR'},
                             headers=W.bearer(operator))
        self.assertEqual((rotate.status_code, rotate.json()['error']['code']), (403, 'scope_denied'))
        self.assertNotIn(BOOT, ''.join(r.text for r in (reserved, ok, rotate)))

    def test_moving_a_stored_credential_to_a_new_origin_needs_admin(self):
        """credentials:write plus broker:admin may change base_url, but re-binding the stored (write-only) value to the
        new origin would present it there: that re-validation needs admin and is refused before any connection."""
        _, operator = self.full.mint('credentials:write', 'broker:admin', name='operator')
        stored = self.put(BOOT, {'secret': 'test-openai-not-a-key-' + 'cd' * 8})
        self.assertEqual(stored.status_code, 201, stored.text[:300])
        moved = self.c.put('/api/broker/upstreams/openai', headers=W.bearer(operator),
                           json={'profile': 'openai', 'base_url': 'http://127.0.0.1:10/v1', 'enabled': False})
        self.assertIn(moved.status_code, (200, 201), moved.text[:300])
        refused = self.c.post('/api/broker/credentials/oa/validate', headers=W.bearer(operator))
        self.assertEqual((refused.status_code, refused.json()['error']['code']), (403, 'scope_denied'))
        self.assertEqual(self.full.broker.vault.record('oa').origin, 'http://127.0.0.1:9')  # still bound to the old one
        admin = self.c.post('/api/broker/credentials/oa/validate', headers=W.bearer(BOOT))
        self.assertNotEqual(admin.status_code, 403, admin.text[:300])  # admin may re-bind (the probe then fails here)


class MountedMcp(unittest.TestCase):
    """T7: tool calls over the mounted /mcp run as the outer principal (uvicorn in a thread, the MCP 2.2 client)."""

    def serve(self, api_key):
        import threading
        import uvicorn
        full = W.FullApp(None, models={'chat-local': 'chat'}, api_key=api_key)
        self.addCleanup(full.close)
        sock = bound_socket()
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(full.app, host='127.0.0.1', port=port, log_level='warning',
                                               lifespan='on'))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
        thread.start()

        def stop():
            server.should_exit = True
            thread.join(15)
            sock.close()
        self.addCleanup(stop)
        deadline = time.monotonic() + 30
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                self.fail('uvicorn did not start')
            time.sleep(0.05)
        return full, f'http://127.0.0.1:{port}'

    @staticmethod
    def tools(url, token, calls):
        import asyncio
        import httpx2
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client

        async def run():
            out = {}
            async with httpx2.AsyncClient(headers={'Authorization': 'Bearer ' + token}, trust_env=False) as http:
                async with Client(streamable_http_client(url + '/mcp/', http_client=http),
                                  read_timeout_seconds=30) as client:
                    for name, args in calls:
                        out[name] = await client.call_tool(name, args)
            return out
        return asyncio.run(run())

    CALLS = [('chat', {'payload': {'model': 'chat-local', 'messages': CHAT}}), ('server_info', {}), ('list_jobs', {})]

    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def test_tools_run_as_the_outer_principal_never_the_bootstrap_key(self):
        full, url = self.serve(BOOT)
        _, mcp_only = full.mint('mcp', name='mcp-only')
        _, chatter = full.mint('mcp', 'inference', name='chatter')
        narrow = self.tools(url, mcp_only, self.CALLS)
        self.assertTrue(narrow['chat'].is_error)
        self.assertIn('scope', str(narrow['chat'].content))
        self.assertTrue(narrow['list_jobs'].is_error)
        self.assertFalse(narrow['server_info'].is_error)
        self.assertNotIn('settings', narrow['server_info'].structured_content or {})
        self.assertEqual(full.engine.calls, [])
        allowed = self.tools(url, chatter, self.CALLS[:1])
        self.assertFalse(allowed['chat'].is_error, str(allowed['chat'].content)[:300])
        self.assertEqual(len(full.engine.calls), 1)
        boot = self.tools(url, BOOT, self.CALLS[1:])
        self.assertFalse(boot['list_jobs'].is_error)
        self.assertIn('settings', boot['server_info'].structured_content or {})

    def test_keys_only_mode_tools_work_for_a_scoped_key(self):
        full, url = self.serve(None)
        _, chatter = full.mint('mcp', 'inference', name='chatter')
        self.assertTrue(full.broker.authenticator.enforced)
        result = self.tools(url, chatter, self.CALLS[:1])
        self.assertFalse(result['chat'].is_error, str(result['chat'].content)[:300])
        self.assertEqual(len(full.engine.calls), 1)


SERVER = textwrap.dedent('''
    import json, os, sys, tempfile
    from pathlib import Path
    sys.path.insert(0, os.environ['WP7_SRC'])
    import uvicorn
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    from titanos.config import Config
    from titanos.server import create_app
    from titanos.settings import Settings
    out = Path(os.environ['TITANOS_HOME']).parent  # inside the test's temporary directory
    app = create_app(Config(), api_key=os.environ['WP7_BOOT'], output_dir=out / 'outputs',
                     settings=Settings(broker_egress='loopback-only'))
    inner = ProxyHeadersMiddleware(app, trusted_hosts='127.0.0.1') if os.environ['WP7_PROXY'] == '1' else app

    async def lan_peer(scope, receive, send):
        # the request arrives from a non-loopback peer (TEST-NET-1), outside the pinned proxy handling
        if scope['type'] in ('http', 'websocket'):
            scope = dict(scope, client=('192.0.2.10', 40000))
        await inner(scope, receive, send)
    import socket
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port_file = Path(os.environ['WP7_PORT_FILE'])
    port_file.with_suffix('.tmp').write_text(str(sock.getsockname()[1]))
    port_file.with_suffix('.tmp').replace(port_file)
    uvicorn.Server(uvicorn.Config(lan_peer, host='127.0.0.1', port=sock.getsockname()[1], proxy_headers=False,
                                  forwarded_allow_ips='127.0.0.1', log_level='warning')).run(sockets=[sock])
''')


def bound_socket() -> socket.socket:
    """A loopback socket bound to a free port and handed to uvicorn as it is: closing it and binding the port again
    lets another socket take the port in between (Windows then refuses uvicorn's bind with WinError 10013)."""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    return sock


class UvicornPinnedProxy(unittest.TestCase):
    def run_server(self, proxy: str):
        home = tempfile.mkdtemp(prefix='titanos-wp7-uvicorn-')
        self.addCleanup(shutil.rmtree, home, True)  # runs last: after the process cleanup registered below
        script = Path(home) / 'serve.py'
        script.write_text(SERVER, encoding='utf-8')
        report = Path(home) / 'netguard.jsonl'
        port_file = Path(home) / 'port'
        env = {**os.environ, 'WP7_SRC': str(ROOT / 'src'), 'WP7_BOOT': BOOT, 'WP7_PROXY': proxy,
               'WP7_PORT_FILE': str(port_file), 'TITANOS_HOME': str(Path(home) / 'home'), 'TITANOS_TEST_NETGUARD': '1',
               'TITANOS_TEST_NETGUARD_REPORT': str(report),
               'PYTHONPATH': os.pathsep.join([str(ROOT / 'tests' / 'broker_mocks' / 'netguard_site'),
                                              str(ROOT / 'src')])}
        process = subprocess.Popen([sys.executable, '-B', str(script)], env=env, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
        self.addCleanup(lambda: (process.kill(), process.wait(10)))
        deadline = time.monotonic() + 60
        while not port_file.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                self.fail('the uvicorn subprocess did not bind: ' + process.stderr.read().decode()[-500:])
            time.sleep(0.05)
        import httpx
        client = httpx.Client(base_url=f'http://127.0.0.1:{port_file.read_text()}', trust_env=False, timeout=10)
        self.addCleanup(client.close)
        while True:
            try:
                if client.get('/healthz').status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if process.poll() is not None or time.monotonic() > deadline:
                self.fail('the uvicorn subprocess did not start: ' + process.stderr.read().decode()[-500:])
            time.sleep(0.1)
        return client, report

    def test_forged_forwarded_proto_never_satisfies_transport_secure(self):
        for proxy in ('0', '1'):
            with self.subTest(proxy_headers=proxy):
                client, report = self.run_server(proxy)
                reply = client.post('/api/keys', json={'name': 'k', 'scopes': ['inference']},
                                    headers={'Authorization': 'Bearer ' + BOOT, 'X-Forwarded-Proto': 'https',
                                             'X-Forwarded-For': '127.0.0.1'})
                self.assertEqual((reply.status_code, reply.json()['error']['code']), (403, 'transport_insecure'))
                self.assertFalse(report.exists() and report.read_text().strip(), 'non-loopback connect attempted')


if __name__ == '__main__':
    unittest.main(verbosity=2)
