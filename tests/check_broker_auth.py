"""Broker inbound authentication (WP3), ASGI-level over a stub app: Bearer scheme case-insensitivity; vendor headers
accepted only in their route family (x-api-key on /v1/messages* and /v1/models*, xi-api-key on the ElevenLabs paths, x-goog-api-key on
/v1beta); ?key= refused (401 query_credential_refused); ambiguous credentials (400); keyless mode ignores placeholder
keys and grants anonymous_loopback; keyed mode answers a wrong key with 401 even from loopback; key records mean
enforcement with no bootstrap key, and expired-only records still do; a non-loopback bind with nothing enforced fails
closed; throttling (429 auth_throttled) and auth.failure audit records without tokens; route scopes (403 scope_denied);
KeylessHostGuard toggles with enforcement and its refusal bytes equal TrustedHostMiddleware's; transport_secure;
the internal-scope principal rule; and the source assertions (hmac.compare_digest on every token comparison, no
network code in WP3 modules). Synthetic keys only; no server, model or network."""
import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone
import importlib.util
import ipaddress
import json
from pathlib import Path
import re
import socket
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402
from titanos.broker.audit import AuditLog  # noqa: E402
from titanos.broker.auth import AuthDenial, Authenticator, KeylessHostGuard  # noqa: E402
from titanos.broker.keys import KeySpec, KeyStore  # noqa: E402
from titanos.broker.types import AuthFailure, Principal  # noqa: E402
from titanos.settings import Settings  # noqa: E402

PACKAGE = ROOT / 'src' / 'titanos' / 'broker'
WP3_MODULES = ('auth.py', 'keys.py', 'limits.py', 'pricing.py', 'usage.py', 'decisions.py', 'audit.py', 'ownership.py')


def install_netguard():
    """tests/broker_mocks/netguard.py (WP4), loaded by file path so the mock package itself is not imported; without that
    file, a minimal guard that refuses every non-loopback socket connect."""
    path = ROOT / 'tests' / 'broker_mocks' / 'netguard.py'
    if path.is_file():
        spec = importlib.util.spec_from_file_location('titanos_test_netguard', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.install()
        return

    def loopback(address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            return ipaddress.ip_address(str(host).split('%', 1)[0]).is_loopback
        except ValueError:
            return False
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded(original):
        def call(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6) and not loopback(address):
                raise ConnectionRefusedError('netguard: non-loopback connect refused in a broker check')
            return original(self, address)
        return call
    socket.socket.connect = guarded(connect)
    socket.socket.connect_ex = guarded(connect_ex)


install_netguard()
BOOT = 'boot-' + 'b' * 24
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


async def stub_app(scope, receive, send):
    principal = scope.get('titanos.principal')
    body = json.dumps({'id': principal.id if principal else None, 'kind': principal.kind if principal else None,
                       'scopes': sorted(principal.scopes) if principal else []}).encode()
    await send({'type': 'http.response.start', 'status': 200, 'headers': [(b'content-type', b'application/json')]})
    await send({'type': 'http.response.body', 'body': body})


class Gate:
    """What server.Guard does with the Authenticator: authenticate, then answer the failure or call the app."""

    def __init__(self, app, auth):
        self.app, self.auth = app, auth

    async def __call__(self, scope, receive, send):
        outcome = self.auth.authenticate(scope, dict(scope.get('headers', [])))
        if isinstance(outcome, AuthFailure):
            return await outcome(scope, receive, send)
        return await self.app(scope, receive, send)


def call(app, method='GET', path='/v1/models', headers=(), *, client=('127.0.0.1', 50000), scheme='http', query=b'',
         host='127.0.0.1:11435', extra_scope=None):
    raw = [(b'host', host.encode())] + [(k.encode().lower(), v.encode()) for k, v in headers]
    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method, 'scheme': scheme,
             'path': path, 'raw_path': path.encode(), 'query_string': query, 'root_path': '', 'headers': raw,
             'client': client, 'server': ('127.0.0.1', 11435)}
    scope.update(extra_scope or {})
    messages = []

    async def receive():
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    async def send(message):
        messages.append(message)
    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m['type'] == 'http.response.start')
    body = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
    return start['status'], [(k.decode(), v.decode()) for k, v in start['headers']], body, scope


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.mono = [1000.0]
        self.now = [NOW]
        self.audit = AuditLog(self.home / 'broker' / 'audit', clock=lambda: self.now[0])
        self.settings = Settings()
        self.keystore = KeyStore(self.home / 'broker' / 'keys.json', self.settings, self.audit,
                                 clock=lambda: self.now[0], monotonic=lambda: self.mono[0])

    def tearDown(self):
        self.tmp.cleanup()

    def auth(self, bootstrap=None, settings=None, **kw):
        kw.setdefault('clock', lambda: self.mono[0])
        return Authenticator(bootstrap_key=bootstrap, keystore=self.keystore, settings=settings or self.settings,
                             audit=self.audit, **kw)

    def gate(self, auth):
        return Gate(stub_app, auth)

    def mint(self, scopes=('inference', 'models:read'), **kw):
        record, token = self.keystore.mint(KeySpec(name='k', scopes=list(scopes), **kw), created_by='bootstrap')
        self.mono[0] += 2  # past the one-second mtime pacing
        return record, token

    def doc(self, body):
        return json.loads(body)


class Keyless(Fixture):
    def test_placeholder_credentials_are_ignored(self):
        auth = self.auth()
        self.assertFalse(auth.enforced)
        for headers in ([], [('authorization', 'Bearer local')], [('authorization', 'Basic xyz')],
                        [('authorization', 'Bearer a'), ('authorization', 'Bearer b')]):
            with self.subTest(headers=headers):
                status, _, body, scope = call(self.gate(auth), headers=headers)
                self.assertEqual(status, 200)
                self.assertEqual(self.doc(body)['kind'], 'anonymous_loopback')
                self.assertIs(scope['titanos.principal'].kind, 'anonymous_loopback')
        scopes = set(self.doc(call(self.gate(auth))[2])['scopes'])
        self.assertNotIn('cloud', scopes)
        self.assertTrue({'credentials:write', 'keys:admin', 'audit:read', 'route:explicit', 'delegate'}.isdisjoint(scopes))
        self.assertEqual(call(self.gate(auth), 'GET', '/api/keys')[0], 403)  # keys:admin is never keyless

    def test_anonymous_cloud_only_in_standalone(self):
        auth = self.auth(anonymous_cloud=lambda: True)
        self.assertIn('cloud', self.doc(call(self.gate(auth))[2])['scopes'])
        managed = self.auth(anonymous_cloud=lambda: True, role='managed')
        self.assertNotIn('cloud', self.doc(call(self.gate(managed))[2])['scopes'])

    def test_non_loopback_bind_with_nothing_enforced_fails_closed(self):
        auth = self.auth(bind_loopback=False)
        status, headers, body, _ = call(self.gate(auth), headers=[('authorization', 'Bearer local')])
        self.assertEqual(status, 401)
        self.assertEqual(self.doc(body)['error']['code'], 'authentication_error')

    def test_bind_loopback_defaults_to_the_settings_host(self):
        # omitting bind_loopback on a 0.0.0.0 bind must never hand LAN peers the anonymous principal
        lan = dataclasses.replace(self.settings, host='0.0.0.0')
        keystore = KeyStore(self.home / 'broker' / 'keys.json', lan, self.audit, clock=lambda: self.now[0],
                            monotonic=lambda: self.mono[0])
        auth = Authenticator(bootstrap_key=None, keystore=keystore, settings=lan, audit=self.audit)
        self.assertFalse(auth.bind_loopback)
        for client in (('192.0.2.50', 5555), ('127.0.0.1', 5555)):
            with self.subTest(client=client):
                self.assertEqual(call(self.gate(auth), 'POST', '/v1/chat/completions', client=client)[0], 401)
        for host in ('127.0.0.1', 'localhost', '::1'):
            with self.subTest(host=host):
                loop = dataclasses.replace(self.settings, host=host)
                self.assertTrue(Authenticator(bootstrap_key=None, keystore=self.keystore, settings=loop).bind_loopback)

    def test_key_and_credential_writes_are_keyless_mode_refused(self):
        auth = self.auth()
        for method, path in (('POST', '/api/keys'), ('PATCH', '/api/keys/abcdefgh23'),
                             ('POST', '/api/keys/abcdefgh23/revoke'), ('PUT', '/api/broker/credentials/openai'),
                             ('DELETE', '/api/broker/credentials/openai')):
            with self.subTest(method=method, path=path):
                status, headers, body, _ = call(self.gate(auth), method, path)
                self.assertEqual((status, self.doc(body)['error']['code']), (403, 'keyless_mode_refused'))
                self.assertIn(('x-should-retry', 'false'), headers)
                self.assertIn('titanos', self.doc(body)['error']['message'])  # points at the CLI
        # reads stay scope_denied: they are not writes
        for method, path in (('GET', '/api/keys'), ('GET', '/api/audit')):
            status, _, body, _ = call(self.gate(auth), method, path)
            self.assertEqual((status, self.doc(body)['error']['code']), (403, 'scope_denied'))
        # a keyed principal lacking the scope gets scope_denied, not keyless_mode_refused
        _, token = self.mint(scopes=('inference',))
        status, _, body, _ = call(self.gate(auth), 'POST', '/api/keys', [('authorization', 'Bearer ' + token)])
        self.assertEqual((status, self.doc(body)['error']['code']), (403, 'scope_denied'))

    def test_query_key_refused_even_keyless(self):
        auth = self.auth()
        status, _, body, _ = call(self.gate(auth), 'POST', '/v1beta/models/m:generateContent', query=b'key=abc12345')
        self.assertEqual(status, 401)
        self.assertEqual(self.doc(body)['error']['status'], 'UNAUTHENTICATED')
        self.assertIn('query', self.doc(body)['error']['message'])


class Keyed(Fixture):
    def test_bearer_scheme_is_case_insensitive_and_trimmed(self):
        auth = self.auth(BOOT)
        for value in (f'Bearer {BOOT}', f'bearer {BOOT}', f'BEARER   {BOOT}  ', f'  BeArEr\t{BOOT}'):
            with self.subTest(value=value):
                status, _, body, _ = call(self.gate(auth), headers=[('authorization', value)])
                self.assertEqual(status, 200)
                self.assertEqual(self.doc(body)['kind'], 'bootstrap')

    def test_wrong_or_missing_key_from_loopback_is_401(self):
        auth = self.auth(BOOT)
        for headers in ([('authorization', 'Bearer wrong-key-value')], [], [('authorization', 'Basic ' + BOOT)],
                        [('authorization', 'Bearer')], [('x-api-key', BOOT)]):
            with self.subTest(headers=headers):
                # x-api-key is a credential on /v1/models (the Anthropic SDK's catalog), not on chat completions
                path = '/v1/chat/completions' if headers and headers[0][0] == 'x-api-key' else '/v1/models'
                status, response_headers, body, _ = call(self.gate(auth), 'POST' if path != '/v1/models' else 'GET',
                                                         path, headers=headers)
                self.assertEqual(status, 401)
                error = self.doc(body)['error']
                self.assertEqual((error['message'], error['code']), ('Invalid API key', 'authentication_error'))
                self.assertIn(('x-titanos-error-origin', 'titanos'), response_headers)

    def test_vendor_headers_only_in_their_family(self):
        record, token = self.mint(scopes=('inference', 'models:read'))
        auth = self.auth()
        self.assertTrue(auth.enforced, 'a key record enforces auth without a bootstrap key')
        accepted = [('x-api-key', 'POST', '/v1/messages'), ('x-api-key', 'POST', '/v1/messages/count_tokens'),
                    ('x-api-key', 'GET', '/v1/models'), ('x-api-key', 'GET', '/v1/models/m'),
                    ('xi-api-key', 'POST', '/v1/text-to-speech/voice1'),
                    ('xi-api-key', 'POST', '/v1/text-to-speech/voice1/stream'),
                    ('xi-api-key', 'POST', '/v1/speech-to-text'), ('xi-api-key', 'GET', '/v1/voices'),
                    ('xi-api-key', 'GET', '/v1/voices/v1'), ('x-goog-api-key', 'GET', '/v1beta/models'),
                    ('x-goog-api-key', 'POST', '/v1beta/models/m:generateContent'),
                    ('authorization', 'POST', '/v1/chat/completions')]
        for name, method, path in accepted:
            with self.subTest(accepted=(name, path)):
                value = ('Bearer ' + token) if name == 'authorization' else token
                status, _, body, _ = call(self.gate(auth), method, path, [(name, value)])
                self.assertEqual(status, 200, body)
                self.assertEqual(self.doc(body)['id'], record.id)
        refused = [('x-api-key', 'POST', '/v1/chat/completions'), ('x-api-key', 'POST', '/v1/messagesx'),
                   ('xi-api-key', 'POST', '/v1/messages'), ('xi-api-key', 'POST', '/v1/audio/speech'),
                   ('x-goog-api-key', 'POST', '/v1/messages'), ('x-goog-api-key', 'GET', '/v1/models')]
        for name, method, path in refused:
            with self.subTest(refused=(name, path)):
                self.assertEqual(call(self.gate(auth), method, path, [(name, token)])[0], 401)

    def test_query_credential_refused_on_v1beta(self):
        _, token = self.mint()
        auth = self.auth()
        status, _, body, _ = call(self.gate(auth), 'POST', '/v1beta/models/m:generateContent',
                                  [('x-goog-api-key', token)], query=b'alt=sse&key=' + token.encode())
        self.assertEqual(status, 401)
        error = self.doc(body)['error']
        self.assertEqual((error['code'], error['status']), (401, 'UNAUTHENTICATED'))
        self.assertNotIn(token, body.decode())
        # ?key= is only a credential on the Gemini family
        self.assertEqual(call(self.gate(auth), 'GET', '/v1/models', [('authorization', 'Bearer ' + token)],
                              query=b'key=x')[0], 200)

    def test_ambiguous_credentials(self):
        _, token = self.mint()
        _, other = self.mint()
        auth = self.auth()
        status, _, body, _ = call(self.gate(auth), 'POST', '/v1/messages',
                                  [('authorization', 'Bearer ' + token), ('x-api-key', other)])
        self.assertEqual(status, 400)
        self.assertEqual(self.doc(body)['error']['type'], 'invalid_request_error')  # Anthropic envelope
        status, _, _, _ = call(self.gate(auth), 'POST', '/v1/messages',
                               [('authorization', 'Bearer ' + token), ('x-api-key', token)])
        self.assertEqual(status, 200)
        status, _, body, _ = call(self.gate(auth), 'GET', '/v1/models',
                                  [('authorization', 'Bearer ' + token), ('authorization', 'Bearer ' + other)])
        self.assertEqual((status, self.doc(body)['error']['code']), (400, 'ambiguous_credentials'))

    def test_expired_and_revoked_keys_keep_enforcement(self):
        record, token = self.mint(expires_at=NOW + timedelta(minutes=1))
        auth = self.auth()
        self.assertEqual(call(self.gate(auth), headers=[('authorization', 'Bearer ' + token)])[0], 200)
        self.now[0] += timedelta(minutes=2)
        self.mono[0] += 2
        self.assertTrue(auth.enforced, 'expired-only records still enforce auth (no keyless reopening)')
        status, _, body, _ = call(self.gate(auth), headers=[('authorization', 'Bearer ' + token)])
        self.assertEqual((status, self.doc(body)['error']['code']), (401, 'key_expired'))
        status, _, body, _ = call(self.gate(auth), headers=[('authorization', 'Bearer local')])
        self.assertEqual((status, self.doc(body)['error']['code']), (401, 'authentication_error'))
        self.assertEqual(call(self.gate(auth), headers=[])[0], 401)
        live, token2 = self.mint()
        self.keystore.revoke(live.id)
        self.mono[0] += 2
        status, _, body, _ = call(self.gate(auth), headers=[('authorization', 'Bearer ' + token2)])
        self.assertEqual((status, self.doc(body)['error']['code']), (401, 'key_revoked'))
        self.assertTrue(auth.enforced)

    def test_bootstrap_and_key_principals(self):
        record, token = self.mint(scopes=('inference', 'models:read', 'delegate'))
        auth = self.auth(BOOT)
        self.assertEqual(self.doc(call(self.gate(auth), headers=[('authorization', 'Bearer ' + BOOT)])[2])['scopes'],
                         ['admin'])
        self.assertEqual(self.doc(call(self.gate(auth), headers=[('authorization', 'Bearer ' + token)])[2])['kind'],
                         'parent')

    def test_scope_denied_before_the_route(self):
        _, token = self.mint(scopes=('inference:chat',))
        auth = self.auth()
        status, headers, body, scope = call(self.gate(auth), 'GET', '/api/keys', [('authorization', 'Bearer ' + token)])
        self.assertEqual((status, self.doc(body)['error']['code']), (403, 'scope_denied'))
        self.assertIn(('x-should-retry', 'false'), headers)
        self.assertNotIn('titanos.principal', scope)
        self.assertEqual(call(self.gate(auth), 'POST', '/v1/chat/completions',
                              [('authorization', 'Bearer ' + token)])[0], 200)
        self.assertEqual(call(self.gate(auth), 'POST', '/v1/embeddings', [('authorization', 'Bearer ' + token)])[0], 403)

    def test_internal_principal_rule(self):
        _, token = self.mint(scopes=('mcp', 'inference'))
        auth = self.auth(BOOT)
        preset = Principal(id='abcdefgh23', kind='key', scopes=frozenset({'inference'}))
        status, _, body, _ = call(self.gate(auth), 'POST', '/v1/chat/completions',
                                  extra_scope={'titanos.internal': True, 'titanos.principal': preset})
        self.assertEqual((status, self.doc(body)['id']), (200, 'abcdefgh23'))
        # an outer principal never widens: the inner route scope is still checked
        self.assertEqual(call(self.gate(auth), 'GET', '/api/keys',
                              extra_scope={'titanos.internal': True, 'titanos.principal': preset})[0], 403)
        # on a non-internal scope a pre-set principal is discarded and the credential decides
        status, _, _, scope = call(self.gate(auth), 'POST', '/v1/chat/completions',
                                   extra_scope={'titanos.principal': preset})
        self.assertEqual(status, 401)
        self.assertNotIn('titanos.principal', scope)

    def test_throttling_and_audit(self):
        record, token = self.mint()
        settings = dataclasses.replace(self.settings, broker_auth_failures_per_minute=3)
        auth = self.auth(settings=settings)
        wrong = 'tk_' + record.id + '_' + 'Z' * 43
        for _ in range(3):
            self.assertEqual(call(self.gate(auth), headers=[('authorization', 'Bearer ' + wrong)])[0], 401)
        status, headers, body, _ = call(self.gate(auth), headers=[('authorization', 'Bearer ' + wrong)])
        self.assertEqual((status, self.doc(body)['error']['code']), (429, 'auth_throttled'))
        retry = dict(headers)['retry-after']
        self.assertTrue(1 <= int(retry) <= 60)
        # the address is throttled even for a valid key; another address is not
        self.assertEqual(call(self.gate(auth), headers=[('authorization', 'Bearer ' + token)])[0], 429)
        self.assertEqual(call(self.gate(auth), headers=[('authorization', 'Bearer ' + token)],
                              client=('127.0.0.2', 1))[0], 200)
        self.mono[0] += 61
        self.assertEqual(call(self.gate(auth), headers=[('authorization', 'Bearer ' + token)])[0], 200)
        failures = self.audit.query(kind='auth.failure')
        self.assertGreaterEqual(len(failures), 4)
        self.assertEqual(failures[0]['subject'], record.id)  # the id prefix, never the token
        self.assertEqual(failures[0]['actor'], '127.0.0.1')
        self.assertIn('auth_throttled', [f['detail']['code'] for f in failures])
        text = '\n'.join(p.read_text(encoding='utf-8') for p in (self.home / 'broker' / 'audit').glob('*.jsonl'))
        self.assertNotIn('Z' * 43, text)
        self.assertNotIn(token.split('_', 2)[2], text)
        self.assertTrue(self.audit.verify().ok)

    def test_absent_credentials_do_not_feed_the_throttle(self):
        # credential-less local probes are answered 401 but cannot lock out the holder of a valid key
        settings = dataclasses.replace(self.settings, broker_auth_failures_per_minute=2)
        auth = self.auth(BOOT, settings=settings)
        for _ in range(10):
            self.assertEqual(call(self.gate(auth))[0], 401)
        self.assertEqual(call(self.gate(auth), headers=[('authorization', 'Bearer ' + BOOT)])[0], 200)
        # presented-but-invalid credentials still count
        for _ in range(2):
            self.assertEqual(call(self.gate(auth), headers=[('authorization', 'Bearer wrong-value-1')])[0], 401)
        self.assertEqual(call(self.gate(auth), headers=[('authorization', 'Bearer ' + BOOT)])[0], 429)

    def test_bootstrap_key_is_registered_for_scrubbing(self):
        from titanos.broker.redact import scrub
        secret = 'boot-' + 'q' * 27
        self.assertIn(secret, scrub('value ' + secret))
        self.auth(secret)
        self.assertNotIn(secret, scrub('value ' + secret))

    def test_auth_denial_is_an_auth_failure_with_headers(self):
        denial = AuthDenial(429, 'auth_throttled', 'slow down', 'anthropic', retry_after=2.2)
        self.assertIsInstance(denial, AuthFailure)
        self.assertEqual(denial.headers()['Retry-After'], '3')
        status, headers, body, _ = call(denial)
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(body)['error']['type'], 'rate_limit_error')


class HostGuardAndTransport(Fixture):
    def test_keyless_host_guard_matches_trusted_host_bytes_and_toggles(self):
        auth = self.auth()
        guard = KeylessHostGuard(stub_app, auth)
        trusted = TrustedHostMiddleware(stub_app, allowed_hosts=['localhost', '127.0.0.1', '[::1]'])
        for host in ('evil.example.invalid', '192.0.2.5:11435', 'localhost.evil.invalid'):
            with self.subTest(host=host):
                ours = call(guard, host=host)[:3]
                theirs = call(trusted, host=host)[:3]
                self.assertEqual(ours, theirs)
                self.assertEqual(ours[0], 400)
                self.assertEqual(ours[2], b'Invalid host header')
        for host in ('127.0.0.1:11435', '127.0.0.1', 'localhost', 'localhost:11435'):
            self.assertEqual(call(guard, host=host)[0], 200)
        # TrustedHostMiddleware splits the port on the first ':', so a bracketed IPv6 Host is refused today; keyless
        # keeps exactly that answer
        for host in ('[::1]', '[::1]:11435'):
            self.assertEqual(call(guard, host=host)[:3], call(trusted, host=host)[:3])
        self.mint()
        self.assertTrue(auth.enforced)
        self.assertEqual(call(guard, host='evil.example.invalid')[0], 200, 'enforced mode leaves Host to the bearer check')
        bootstrap_guard = KeylessHostGuard(stub_app, self.auth(BOOT))
        self.assertEqual(call(bootstrap_guard, host='evil.example.invalid')[0], 200)

    def test_transport_secure(self):
        auth = self.auth(BOOT)
        cases = [({'scheme': 'http', 'client': ('192.0.2.10', 4000)}, False),
                 ({'scheme': 'https', 'client': ('192.0.2.10', 4000)}, True),
                 ({'scheme': 'http', 'client': ('127.0.0.1', 4000)}, True),
                 ({'scheme': 'http', 'client': ('::1', 4000)}, True),
                 ({'scheme': 'http', 'client': ('testclient', 50000)}, False),
                 ({'scheme': 'http', 'client': None}, False),
                 ({'scheme': 'wss', 'client': ('192.0.2.10', 1)}, True)]
        for scope, expected in cases:
            with self.subTest(scope=scope):
                self.assertIs(auth.transport_secure(scope), expected)


class SourceAssertions(unittest.TestCase):
    def test_token_comparisons_are_constant_time(self):
        for name in ('auth.py', 'keys.py'):
            text = (PACKAGE / name).read_text(encoding='utf-8')
            self.assertIn('hmac.compare_digest', text, name)
            for line in text.splitlines():
                code = line.split('#', 1)[0]
                self.assertIsNone(re.search(r'(sha256|digest|token|secret|_bootstrap)\w*\s*[!=]=|[!=]=\s*\w*(token|digest|'
                                            r'secret|sha256)\b', code), f'{name}: non-constant-time compare: {line.strip()}')

    def test_the_network_guard_is_active(self):
        # the guard's own refusal (WP4 netguard or the fallback), not merely any OSError a real attempt could raise
        with self.assertRaises(OSError) as caught:
            socket.create_connection(('192.0.2.1', 9), timeout=0.5)
        self.assertIn('netguard', str(caught.exception))
        module = sys.modules.get('titanos_test_netguard')
        if module is not None:
            self.assertTrue(module.installed())

    def test_wp3_modules_do_no_network_io(self):
        forbidden = re.compile(r'^\s*(import|from)\s+(socket|ssl|httpx|httpcore|requests|urllib\.request|http\.client|'
                               r'aiohttp|asyncio\.streams)\b|open_connection|create_connection|urlopen', re.M)
        for name in WP3_MODULES:
            with self.subTest(module=name):
                self.assertIsNone(forbidden.search((PACKAGE / name).read_text(encoding='utf-8')))


if __name__ == '__main__':
    unittest.main()
