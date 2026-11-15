"""Broker egress (WP2): the single outbound client and its guarded resolver, against loopback MockServer providers (WP4)
and httpx.MockTransport (the tests/check_remote_pool.py pattern), with an injected resolver and an instrumented network
backend for name cases.

Refused: userinfo URLs, a different port, http to a non-loopback host, suffix and IDN look-alikes, path tricks, a 307 to a
foreign host (nothing followed, no credential on any second hop), HTTPS_PROXY/HTTP_PROXY (unused), names resolving to
169.254.169.254, ::ffff:169.254.169.254, fd00:ec2::254, 0.0.0.0, 224.0.0.1 and other blocked forms (refused before any
connect), RFC1918 without allow_http_lan, public addresses in loopback-only mode, credential query parameters, an expired
deadline. Allowed: the exact Azure resource host. A credential is never attached when its record origin differs from the
upstream origin (base_url changed). Managed role refuses every purpose with role_forbids before any socket opens, including
auth-none upstreams. Also: response size caps, set-cookie and location dropped, no cookie replay, inbound authorization /
x-api-key / cookie never forwarded, worker / location / timeout stripped from bodies, the hop header on hop profiles, the
per-upstream concurrency slot, the first-byte timeout and the upstream-401 credential marking (an inference 401 marks at
once only with the provider's key-rejection code, otherwise it calls on_auth_suspect; a restricted key's 'Missing
scopes' 401 never marks, on any purpose). Review fixes: compressed
bodies are decoded inside the cap (gzip and deflate bombs refused while decoding, other codings refused), the media / SSE
cap, translated IPv6 forms (6to4, NAT64, IPv4-compatible) never pass as loopback, str and whitespace-led raw JSON is
stripped, the returned response's request carries no auth header, the deadline clamps the timeouts, a failure in the
after-response bookkeeping releases the slot, and (when the cryptography package is importable) a loopback TLS server
proves SNI and the certificate hostname check use the origin host while the connection goes to the checked IP.

The netguard (tests/broker_mocks) is installed for the whole run; every connect in this file is loopback or instrumented.
Synthetic sentinels only.
"""
import asyncio
import gzip
import http.server
import ipaddress
import json
import os
import ssl
import threading
import zlib
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from broker_mocks import OpenAIMock, RouterUpstreamMock, SENTINELS, SinkMock, netguard, wait_until  # noqa: E402

netguard.install()

import httpcore  # noqa: E402
import httpx  # noqa: E402

from titanos.broker import egress as egress_module  # noqa: E402
from titanos.broker.config import Upstream  # noqa: E402
from titanos.broker.egress import EgressClient, GuardedBackend, check_address  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.profiles import PROFILES  # noqa: E402
from titanos.broker.role import switches_for  # noqa: E402
from titanos.broker.types import GRANT_PURPOSES, EgressGrant  # noqa: E402
from titanos.broker.vault import CredentialHandle, Secret  # noqa: E402
from titanos.broker.vault.store import Vault, platform_default  # noqa: E402
from titanos.settings import Settings  # noqa: E402

LOOPBACK = Settings(broker_egress='loopback-only')
INTERNET = Settings(broker_egress='internet')
INBOUND_TOKEN = 'tk_' + 'a' * 10 + '_' + 'B' * 43
PRIVATE_10 = str(ipaddress.IPv4Address(0x0A000005))       # an RFC1918 address, computed
PUBLIC = str(ipaddress.IPv4Address(0x5DB8D822))           # a public address, computed
BLOCKED = ('169.254.169.254', '::ffff:169.254.169.254', 'fd00:ec2::254', '0.0.0.0', '224.0.0.1', '255.255.255.255',
           'fe80::1', '64:ff9b::a9fe:a9fe', '2002:a9fe:a9fe::1', '::a9fe:a9fe', '0.1.2.3', 'ff02::1')


def grant(origin, credential=None, *, purpose='inference', upstream_id='up', delegated=False, hop=None, deadline=30.0,
          request_id='req-1'):
    return EgressGrant(decision_id='dec-1', request_id=request_id, purpose=purpose, delegated=delegated, target=None,
                       upstream_id=upstream_id, origin=origin, credential=credential,
                       deadline=time.monotonic() + deadline, hop_header=hop)


def bearer(origin, name='openai'):
    return CredentialHandle(origin, 'bearer', Secret(SENTINELS[name]), credential_id=name)


def run(coro):
    return asyncio.run(coro)


class Capture:
    """httpx.MockTransport that records every request it would have sent."""

    def __init__(self, status=200, headers=None, body=b'{"ok": true}'):
        self.requests = []
        self.status, self.headers, self.body = status, headers or {'content-type': 'application/json'}, body
        self.sync = httpx.MockTransport(self.handle)
        self.async_ = httpx.MockTransport(self.handle)

    def handle(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, headers=self.headers, content=self.body)


class CountingBackend(httpcore.AsyncNetworkBackend):
    """Instrumented inner backend: records connect attempts and never opens a socket."""

    def __init__(self):
        self.connects = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connects.append((host, port))
        raise httpcore.ConnectError('instrumented backend: no sockets')

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise httpcore.ConnectError('instrumented backend: no sockets')

    async def sleep(self, seconds):
        await asyncio.sleep(seconds)


class CountingSyncBackend(httpcore.NetworkBackend):
    def __init__(self):
        self.connects = []

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connects.append((host, port))
        raise httpcore.ConnectError('instrumented backend: no sockets')

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise httpcore.ConnectError('instrumented backend: no sockets')

    def sleep(self, seconds):
        time.sleep(seconds)


class MemoryBackend:
    def __init__(self, name):
        self.name, self.values = name, {}

    def available(self):
        return True

    def store(self, id, value):
        self.values[id] = bytes(value)

    def load(self, id):
        if id not in self.values:
            raise BrokerError('missing', 'credential_unavailable')
        return self.values[id]

    def delete(self, id):
        self.values.pop(id, None)


class EgressCase(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def client(self, settings=LOOPBACK, **kw):
        client = EgressClient(settings, kw.pop('vault', None), switches_for(settings), **kw)
        self.addCleanup(client.close)
        return client

    def assertCode(self, code, call, *args, **kw):
        with self.assertRaises(BrokerError) as ctx:
            call(*args, **kw)
        self.assertEqual(ctx.exception.code, code, ctx.exception)
        return ctx.exception

    def assertCodeAsync(self, code, coro):
        with self.assertRaises(BrokerError) as ctx:
            run(coro)
        self.assertEqual(ctx.exception.code, code, ctx.exception)
        return ctx.exception


class MockRoundTrip(EgressCase):
    def test_bound_credential_only_and_inbound_secrets_never_forwarded(self):
        with OpenAIMock() as provider:
            upstream = Upstream(profile='openai', base_url=provider.api_base, enabled=True)
            origin = provider.origin
            inbound = {'Authorization': 'Bearer ' + INBOUND_TOKEN, 'x-api-key': INBOUND_TOKEN, 'cookie': 's=' + INBOUND_TOKEN,
                       'x-goog-api-key': INBOUND_TOKEN, 'X-Titanos-Target': 'x@y', 'anthropic-beta': 'beta-1',
                       'proxy-authorization': 'Basic ' + INBOUND_TOKEN}
            body = {'model': 'gpt-4.1-mini', 'messages': [{'role': 'user', 'content': 'hi'}], 'worker': 'gpu0',
                    'location': 'cloud_only', 'timeout': 9}
            egress = self.client()

            async def call():
                try:
                    return await egress.send(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'POST',
                                             '/chat/completions', json=body, headers=inbound)
                finally:
                    await egress.aclose()
            response = run(call())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['object'], 'chat.completion')
            sync = egress.send_sync(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'POST',
                                    '/chat/completions', json=body, headers=inbound)
            self.assertEqual(sync.status_code, 200)
            self.assertEqual(len(provider.hits), 2)
            for hit in provider.hits:
                self.assertEqual(hit.credentials, [('header:authorization', 'Bearer <sentinel:openai>')])
                self.assertFalse(hit.presented(INBOUND_TOKEN))
                for name in ('x-api-key', 'cookie', 'x-goog-api-key', 'x-titanos-target', 'proxy-authorization'):
                    self.assertNotIn(name, hit.headers)
                self.assertEqual(hit.headers.get('anthropic-beta'), 'beta-1')
                self.assertEqual(hit.headers.get('accept-encoding'), 'identity')
                self.assertEqual(set(hit.json()), {'model', 'messages'})

    def test_raw_json_content_and_form_fields_are_stripped_too(self):
        capture = Capture()
        upstream = Upstream(profile='custom_openai', base_url='http://127.0.0.1:9/v1')
        origin = 'http://127.0.0.1:9'
        egress = self.client(sync_transport=capture.sync)
        egress.send_sync(grant(origin), upstream, PROFILES['custom_openai'], 'POST', '/chat/completions',
                         content=b'{"model":"m","worker":"w","location":"auto","timeout":1}')
        egress.send_sync(grant(origin), upstream, PROFILES['custom_openai'], 'POST', '/audio/transcriptions',
                         data={'model': 'm', 'worker': 'w'}, files={'file': ('a.wav', b'RIFF', 'audio/wav')})
        egress.send_sync(grant(origin), upstream, PROFILES['custom_openai'], 'POST', '/chat/completions',
                         content=b' \n\t{"model":"m","worker":"w","location":"cloud_only"}')
        egress.send_sync(grant(origin), upstream, PROFILES['custom_openai'], 'POST', '/chat/completions',
                         content='{"model":"m","worker":"w","timeout":3}')
        self.assertEqual(capture.requests[0].content, b'{"model":"m"}')
        multipart = capture.requests[1].read()
        self.assertIn(b'name="model"', multipart)
        self.assertNotIn(b'name="worker"', multipart)
        self.assertNotIn('authorization', capture.requests[0].headers)
        self.assertEqual(json.loads(capture.requests[2].content), {'model': 'm'})
        self.assertEqual(json.loads(capture.requests[3].content), {'model': 'm'})


class Refusals(EgressCase):
    def test_userinfo_port_scheme_and_look_alike_urls(self):
        capture = Capture()
        egress = self.client(INTERNET, transport=capture.async_, sync_transport=capture.sync)
        openai_origin = 'https://api.openai.com:443'
        handle = bearer(openai_origin)
        cases = [
            # (upstream base_url bypassing config validation, grant origin)
            ('https://user:pw@api.openai.com/v1', openai_origin),
            ('https://api.openai.com:8443/v1', openai_origin),
            ('http://api.openai.com/v1', openai_origin),
            ('https://api.openai.com.example.invalid/v1', openai_origin),
            ('https://api.openai.com.example.invalid/v1', 'https://api.openai.com.example.invalid:443'),
            ('https://xn--pi-6kc.openai.com/v1', 'https://xn--pi-6kc.openai.com:443'),
            ('https://аpi.openai.com/v1', 'https://xn--pi-6kc.openai.com:443'),
            ('http://' + PRIVATE_10 + '/v1', 'http://' + PRIVATE_10 + ':80'),
        ]
        for base_url, origin in cases:
            with self.subTest(base_url=base_url):
                upstream = Upstream.model_construct(profile='openai', base_url=base_url, azure=None, headers={},
                                                    timeouts=None, max_concurrency=4, anthropic_version=None,
                                                    anthropic_betas=[])
                self.assertCode('egress_refused', egress.send_sync, grant(origin, handle), upstream,
                                PROFILES['openai'], 'GET', '/models')
                self.assertCodeAsync('egress_refused', egress.send(grant(origin, handle), upstream, PROFILES['openai'],
                                                                   'GET', '/models'))
        self.assertEqual(capture.requests, [])

    def test_path_tricks_and_query_credentials(self):
        capture = Capture()
        egress = self.client(sync_transport=capture.sync)
        upstream = Upstream(profile='custom_openai', base_url='http://127.0.0.1:9/v1')
        origin = 'http://127.0.0.1:9'
        for path, params in [('//evil.example.invalid/x', None), ('/../models', None), ('/a/%2e%2e/b', None),
                             ('models', None), ('/models#frag', None), ('/models\\x', None),
                             ('/models', {'key': 'x'}), ('/models?api_key=x', None), ('/{model}', None)]:
            with self.subTest(path=path, params=params):
                self.assertCode('egress_refused', egress.send_sync, grant(origin), upstream,
                                PROFILES['custom_openai'], 'GET', path, params=params)
        self.assertCode('egress_refused', egress.send_sync, grant(origin), upstream, PROFILES['custom_openai'], 'GET',
                        '/models/{model}', path_params={'model': '..'})
        egress.send_sync(grant(origin), upstream, PROFILES['custom_openai'], 'GET', '/models/{model}',
                         path_params={'model': 'org/../name@x'})
        self.assertEqual(len(capture.requests), 1)
        self.assertEqual(capture.requests[0].url.raw_path, b'/v1/models/org%2F..%2Fname@x')
        self.assertEqual(capture.requests[0].url.host, '127.0.0.1')

    def test_grant_origin_and_different_port(self):
        with OpenAIMock() as first, OpenAIMock() as second:
            upstream = Upstream(profile='openai', base_url=second.api_base)
            egress = self.client()
            # the grant and the credential name the first mock; the upstream now points at the second
            self.assertCode('egress_refused', egress.send_sync, grant(first.origin, bearer(first.origin)), upstream,
                            PROFILES['openai'], 'GET', '/models')
            self.assertCode('egress_refused', egress.send_sync, grant(second.origin, bearer(first.origin)), upstream,
                            PROFILES['openai'], 'GET', '/models')
            self.assertEqual((first.hits, second.hits), ([], []))

    def test_redirect_to_a_foreign_host_is_refused_and_never_followed(self):
        with OpenAIMock() as provider:
            upstream = Upstream(profile='openai', base_url=provider.api_base)
            origin = provider.origin
            egress = self.client()
            provider.faults.push('redirect_foreign', times=3)
            self.assertCode('upstream_redirect_refused', egress.send_sync, grant(origin, bearer(origin)), upstream,
                            PROFILES['openai'], 'GET', '/models')

            async def calls():
                for stream in (False, True):
                    with self.assertRaises(BrokerError) as ctx:
                        await egress.send(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET', '/models',
                                          stream=stream)
                    self.assertEqual(ctx.exception.code, 'upstream_redirect_refused')
                    self.assertEqual(ctx.exception.response_headers().get('x-should-retry'), 'false')
                await egress.aclose()
            run(calls())
            self.assertEqual(len(provider.hits), 3)
            self.assertIsNotNone(provider.sink)
            self.assertEqual(provider.sink.hits, [], 'the redirect target was contacted')

    def test_proxy_environment_is_never_used(self):
        with OpenAIMock() as provider, SinkMock() as proxy:
            env = {'HTTPS_PROXY': proxy.base_url, 'HTTP_PROXY': proxy.base_url, 'ALL_PROXY': proxy.base_url,
                   'https_proxy': proxy.base_url, 'http_proxy': proxy.base_url, 'NO_PROXY': '', 'no_proxy': ''}
            with mock.patch.dict(os.environ, env):
                egress = self.client()
                upstream = Upstream(profile='openai', base_url=provider.api_base)
                origin = provider.origin
                self.assertEqual(egress.send_sync(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET',
                                                  '/models').status_code, 200)

                async def call():
                    try:
                        return await egress.send(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET',
                                                 '/models')
                    finally:
                        await egress.aclose()
                self.assertEqual(run(call()).status_code, 200)
            self.assertEqual(proxy.hits, [])
            self.assertEqual(len(provider.hits), 2)

    def test_auth_required_without_credential_and_slot_profiles(self):
        capture = Capture()
        egress = self.client(INTERNET, sync_transport=capture.sync)
        self.assertCode('upstream_unconfigured', egress.send_sync, grant('https://api.openai.com:443'),
                        Upstream(profile='openai'), PROFILES['openai'], 'GET', '/models')
        self.assertCode('upstream_unconfigured', egress.send_sync, grant('https://x.example.invalid:443'),
                        Upstream(profile='pending_contract'), PROFILES['pending_contract'], 'GET', '/models')
        self.assertEqual(capture.requests, [])

    def test_template_must_match_the_profile(self):
        capture = Capture()
        egress = self.client(INTERNET, sync_transport=capture.sync)
        origin = 'https://api.openai.com:443'
        wrong = CredentialHandle(origin, 'anthropic', Secret(SENTINELS['openai']), credential_id='openai')
        self.assertCode('egress_refused', egress.send_sync, grant(origin, wrong), Upstream(profile='openai'),
                        PROFILES['openai'], 'GET', '/models')
        self.assertEqual(capture.requests, [])

    def test_expired_deadline(self):
        capture = Capture()
        egress = self.client(sync_transport=capture.sync)
        self.assertCode('deadline_exceeded', egress.send_sync, grant('http://127.0.0.1:9', deadline=-1),
                        Upstream(profile='custom_openai', base_url='http://127.0.0.1:9/v1'), PROFILES['custom_openai'],
                        'GET', '/models')
        self.assertEqual(capture.requests, [])

    def test_ambient_base_urls_and_keys_are_ignored(self):
        capture = Capture()
        egress = self.client(INTERNET, sync_transport=capture.sync)
        with mock.patch.dict(os.environ, {'ANTHROPIC_BASE_URL': 'http://127.0.0.1:1/x', 'OPENAI_BASE_URL':
                                          'http://127.0.0.1:1/y', 'OPENAI_API_KEY': SENTINELS['leak']}):
            origin = 'https://api.anthropic.com:443'
            handle = CredentialHandle(origin, 'anthropic', Secret(SENTINELS['anthropic']), credential_id='anthropic')
            egress.send_sync(grant(origin, handle), Upstream(profile='anthropic'), PROFILES['anthropic'], 'POST',
                             '/v1/messages', json={'model': 'm', 'max_tokens': 1, 'messages': []})
            egress.send_sync(grant('https://contoso-llm.example.invalid:443'),
                             Upstream(profile='custom_openai', base_url='https://contoso-llm.example.invalid/v1'),
                             PROFILES['custom_openai'], 'GET', '/models')
        sent = capture.requests[0]
        self.assertEqual(str(sent.url), PROFILES['anthropic'].default_base_url + '/v1/messages')
        self.assertEqual(sent.headers['x-api-key'], SENTINELS['anthropic'])
        self.assertEqual(sent.headers['anthropic-version'], '2023-06-01')
        self.assertNotIn('authorization', capture.requests[1].headers)
        for request in capture.requests:
            self.assertNotIn(SENTINELS['leak'], str(request.headers))


class Resolver(EgressCase):
    UPSTREAM = Upstream(profile='custom_openai', base_url='https://upstream.example.invalid/v1')
    ORIGIN = 'https://upstream.example.invalid:443'

    def clients(self, settings, answers, allow_lan=False):
        inner, sync_inner = CountingBackend(), CountingSyncBackend()
        egress = self.client(settings, network_backend=inner, sync_network_backend=sync_inner,
                             resolver=lambda host, port: list(answers), allow_http_lan=allow_lan)
        return egress, inner, sync_inner

    def outcome(self, egress, inner):
        async def call():
            try:
                await egress.send(grant(self.ORIGIN), self.UPSTREAM, PROFILES['custom_openai'], 'GET', '/models')
            finally:
                await egress.aclose()
        try:
            run(call())
        except BrokerError as exc:
            return exc.code
        except httpx.ConnectError:
            return 'connect'
        return 'sent'

    def sync_outcome(self, egress):
        try:
            egress.send_sync(grant(self.ORIGIN), self.UPSTREAM, PROFILES['custom_openai'], 'GET', '/models')
        except BrokerError as exc:
            return exc.code
        except httpx.ConnectError:
            return 'connect'
        return 'sent'

    def test_blocked_resolved_addresses_are_refused_before_connect(self):
        for address in BLOCKED:
            for settings in (INTERNET, Settings(broker_egress='lan')):
                with self.subTest(address=address, mode=settings.broker_egress):
                    egress, inner, sync_inner = self.clients(settings, [address], allow_lan=True)
                    self.assertEqual(self.outcome(egress, inner), 'egress_refused')
                    self.assertEqual(self.sync_outcome(egress), 'egress_refused')
                    self.assertEqual((inner.connects, sync_inner.connects), ([], []))

    def test_every_resolved_address_is_checked(self):
        egress, inner, sync_inner = self.clients(INTERNET, [PUBLIC, '169.254.169.254'])
        self.assertEqual(self.outcome(egress, inner), 'egress_refused')
        self.assertEqual(self.sync_outcome(egress), 'egress_refused')
        self.assertEqual((inner.connects, sync_inner.connects), ([], []))

    def test_rfc1918_needs_allow_http_lan(self):
        egress, inner, sync_inner = self.clients(INTERNET, [PRIVATE_10])
        self.assertEqual(self.outcome(egress, inner), 'egress_refused')
        self.assertEqual(self.sync_outcome(egress), 'egress_refused')
        self.assertEqual(inner.connects, [])
        egress, inner, sync_inner = self.clients(INTERNET, [PRIVATE_10], allow_lan=True)
        self.assertEqual(self.outcome(egress, inner), 'connect')
        self.assertEqual(self.sync_outcome(egress), 'connect')
        self.assertEqual(inner.connects, [(PRIVATE_10, 443)])  # the checked address, never the name
        self.assertEqual(sync_inner.connects, [(PRIVATE_10, 443)])

    def test_egress_modes(self):
        table = [
            (Settings(broker_egress='loopback-only'), PUBLIC, True, 'egress_refused'),
            (Settings(broker_egress='loopback-only'), PRIVATE_10, True, 'egress_refused'),
            (Settings(broker_egress='loopback-only'), '127.0.0.1', False, 'connect'),
            (Settings(broker_egress='loopback-only'), '::1', False, 'connect'),
            (Settings(broker_egress='lan'), PUBLIC, True, 'egress_refused'),
            (Settings(broker_egress='lan'), PRIVATE_10, True, 'connect'),
            (INTERNET, PUBLIC, False, 'connect'),
        ]
        for settings, address, allow_lan, expected in table:
            with self.subTest(mode=settings.broker_egress, address=address):
                egress, inner, sync_inner = self.clients(settings, [address], allow_lan=allow_lan)
                self.assertEqual(self.outcome(egress, inner), expected)
                self.assertEqual(self.sync_outcome(egress), expected)
                self.assertEqual(len(inner.connects), 1 if expected == 'connect' else 0)

    def test_check_address_table(self):
        for address in BLOCKED:
            with self.assertRaises(BrokerError):
                check_address(address, mode='internet', allow_lan=True)
        check_address('127.0.0.1', mode='loopback-only', allow_lan=False)
        check_address('::ffff:127.0.0.1', mode='loopback-only', allow_lan=False)
        check_address('fd12::1', mode='internet', allow_lan=True)
        with self.assertRaises(BrokerError):
            check_address('fd12::1', mode='internet', allow_lan=False)
        with self.assertRaises(BrokerError):
            check_address(str(ipaddress.IPv4Address(0x64400001)), mode='internet', allow_lan=False)  # 100.64/10
        check_address(PUBLIC, mode='internet', allow_lan=False)

    def test_translated_ipv6_forms_never_pass_as_loopback(self):
        loopback_forms = ('2002:7f00:1::1', '64:ff9b::7f00:1', '::7f00:1')
        for address in loopback_forms:
            for mode in ('loopback-only', 'lan', 'internet'):
                with self.subTest(address=address, mode=mode):
                    with self.assertRaises(BrokerError):
                        check_address(address, mode=mode, allow_lan=True)
        public = ipaddress.IPv4Address(PUBLIC)
        sixtofour = str(ipaddress.IPv6Address((0x2002 << 112) | (int(public) << 80) | 1))
        nat64 = str(ipaddress.IPv6Address(int(ipaddress.IPv6Address('64:ff9b::')) | int(public)))
        for address in (sixtofour, nat64):
            with self.subTest(address=address):
                check_address(address, mode='internet', allow_lan=False)  # a public translated address
                for mode in ('loopback-only', 'lan'):
                    with self.assertRaises(BrokerError):
                        check_address(address, mode=mode, allow_lan=True)
        private = ipaddress.IPv4Address(PRIVATE_10)
        nat64_private = str(ipaddress.IPv6Address(int(ipaddress.IPv6Address('64:ff9b::')) | int(private)))
        with self.assertRaises(BrokerError):
            check_address(nat64_private, mode='internet', allow_lan=False)
        check_address('::ffff:127.0.0.1', mode='loopback-only', allow_lan=False)  # IPv4-mapped IS the IPv4 address
        egress, inner, sync_inner = self.clients(Settings(broker_egress='loopback-only'), ['64:ff9b::7f00:1'])
        self.assertEqual(self.outcome(egress, inner), 'egress_refused')
        self.assertEqual(self.sync_outcome(egress), 'egress_refused')
        self.assertEqual((inner.connects, sync_inner.connects), ([], []))

    def test_unresolvable_name_is_a_connect_error(self):
        backend = GuardedBackend(mode='internet', resolver=lambda host, port: [], inner=CountingBackend())
        with self.assertRaises(httpcore.ConnectError):
            run(backend.connect_tcp('nothing.example.invalid', 443))


class AzureAndOrigins(EgressCase):
    def azure(self):
        return Upstream(profile='azure_openai', azure={'resource': 'contoso', 'api_version': '2024-10-21',
                                                        'deployments': {'gpt-4o': 'chat-dep'}})

    def test_exact_azure_resource_host_is_allowed(self):
        capture = Capture()
        egress = self.client(INTERNET, sync_transport=capture.sync)
        upstream = self.azure()
        origin = upstream.origin(allow_http_lan=False)
        self.assertEqual(origin, 'https://contoso.openai.azure.com:443')
        handle = CredentialHandle(origin, 'azure_api_key', Secret(SENTINELS['azure_openai']), credential_id='azure')
        egress.send_sync(grant(origin, handle), upstream, PROFILES['azure_openai'], 'POST',
                         PROFILES['azure_openai'].surfaces['openai.chat'], path_params={'deployment': 'chat-dep'},
                         json={'messages': []})
        sent = capture.requests[0]
        self.assertEqual(sent.url.host, 'contoso.openai.azure.com')
        self.assertEqual(sent.url.path, '/openai/deployments/chat-dep/chat/completions')
        self.assertEqual(sent.url.params['api-version'], '2024-10-21')
        self.assertEqual(sent.headers['api-key'], SENTINELS['azure_openai'])
        self.assertNotIn('authorization', sent.headers)

    def test_azure_look_alikes_and_other_resources_are_refused(self):
        capture = Capture()
        egress = self.client(INTERNET, sync_transport=capture.sync)
        upstream = self.azure()
        origin = upstream.origin(allow_http_lan=False)
        for bound in ('https://other.openai.azure.com:443', 'https://contoso.openai.azure.com.example.invalid:443',
                      'https://contoso.cognitiveservices.azure.com:443', 'http://contoso.openai.azure.com:443'):
            with self.subTest(bound=bound):
                handle = CredentialHandle(bound, 'azure_api_key', Secret(SENTINELS['azure_openai']))
                self.assertCode('egress_refused', egress.send_sync, grant(origin, handle), upstream,
                                PROFILES['azure_openai'], 'POST', '/deployments/{deployment}/chat/completions',
                                path_params={'deployment': 'chat-dep'}, json={})
        self.assertEqual(capture.requests, [])

    def test_credential_never_attached_after_a_base_url_change(self):
        home = Path(tempfile.mkdtemp(prefix='titanos-egress-'))
        self.addCleanup(lambda: __import__('shutil').rmtree(home, ignore_errors=True))
        name = platform_default()
        vault = Vault(Settings(), home, backends={name: MemoryBackend(name)})
        with OpenAIMock() as first, OpenAIMock() as second:
            vault.put_secret('openai', SENTINELS['openai'], upstream_id='openai', origin=first.origin)
            vault.mark_validated('openai', True, None, None)
            old = vault.handle('openai', origin=first.origin, template='bearer')
            moved = Upstream(profile='openai', base_url=second.api_base)
            # the issuer asks the vault for the upstream's current origin: refused
            self.assertCode('credential_unavailable', vault.handle, 'openai', origin=second.origin, template='bearer')
            egress = self.client(vault=vault)
            # a stale handle in a grant for the new origin is refused before anything is sent
            self.assertCode('egress_refused', egress.send_sync, grant(second.origin, old, upstream_id='openai'), moved,
                            PROFILES['openai'], 'GET', '/models')
            # the base_url change resets validation: even the old origin is unusable until re-validated
            vault.invalidate_for_upstream('openai', origin=second.origin)
            self.assertCode('credential_unavailable', vault.handle, 'openai', origin=first.origin, template='bearer')
            self.assertEqual((first.hits, second.hits), ([], []))

    def test_upstream_401_marks_the_credential_failed(self):
        home = Path(tempfile.mkdtemp(prefix='titanos-egress-'))
        self.addCleanup(lambda: __import__('shutil').rmtree(home, ignore_errors=True))
        name = platform_default()
        vault = Vault(Settings(), home, backends={name: MemoryBackend(name)})
        with OpenAIMock() as provider:
            vault.put_secret('openai', SENTINELS['openai'], upstream_id='openai', origin=provider.origin)
            vault.mark_validated('openai', True, None, None)
            egress = self.client(vault=vault)
            upstream = Upstream(profile='openai', base_url=provider.api_base)
            handle = vault.handle('openai', origin=provider.origin, template='bearer')
            ok = egress.send_sync(grant(provider.origin, handle), upstream, PROFILES['openai'], 'GET', '/models')
            self.assertEqual(ok.status_code, 200)
            self.assertIsNotNone(vault.record('openai').last_used_at)
            provider.faults.push('status', status=401, code='invalid_api_key')
            denied = egress.send_sync(grant(provider.origin, handle), upstream, PROFILES['openai'], 'GET', '/models')
            self.assertEqual(denied.status_code, 401)
            self.assertEqual(vault.record('openai').validation, 'failed:upstream_auth')
            self.assertCode('credential_unavailable', vault.handle, 'openai', origin=provider.origin, template='bearer')

    def test_inference_401_without_a_key_rejection_code_asks_for_a_check(self):
        """An inference call's endpoint and body are the caller's choice: a 401 that does not carry the provider's
        key-rejection code (OpenAI's restricted-key 'Missing scopes', a bare 401) never marks the credential itself; a
        bare 401 calls on_auth_suspect (the Broker then asks the probe endpoint), a scope refusal not even that. The same
        bare 401 on the operator-chosen probe or catalog endpoint marks at once, a scope refusal never."""
        home = Path(tempfile.mkdtemp(prefix='titanos-egress-'))
        self.addCleanup(lambda: __import__('shutil').rmtree(home, ignore_errors=True))
        name = platform_default()
        scopes = ('You have insufficient permissions for this operation. Missing scopes: api.videos.write. Check that '
                  'you have the correct role in your organization (Reader, Writer, Owner) and project (Member, Owner), '
                  'and if you are using a restricted key, that it has the necessary scopes.')
        for purpose in ('inference', 'catalog', 'probe'):
            for message in (scopes, None):
                with self.subTest(purpose=purpose, scopes=message is not None), OpenAIMock() as provider:
                    vault = Vault(Settings(), home / f'{purpose}-{message is None}', backends={name: MemoryBackend(name)})
                    vault.put_secret('openai', SENTINELS['openai'], upstream_id='openai', origin=provider.origin)
                    vault.mark_validated('openai', True, None, None)
                    egress = self.client(vault=vault)
                    suspects = []
                    egress.on_auth_suspect = lambda *args: suspects.append(args)
                    upstream = Upstream(profile='openai', base_url=provider.api_base)
                    handle = vault.handle('openai', origin=provider.origin, template='bearer')
                    provider.faults.push('status', status=401, **({'message': message} if message else {}))
                    denied = egress.send_sync(grant(provider.origin, handle, purpose=purpose, upstream_id='oai'),
                                              upstream, PROFILES['openai'], 'GET', '/models')
                    self.assertEqual(denied.status_code, 401)
                    marked = purpose != 'inference' and message is None
                    self.assertEqual(vault.record('openai').validation, 'failed:upstream_auth' if marked else 'ok')
                    self.assertEqual(suspects, [('oai', 'openai', None)] if purpose == 'inference' and message is None
                                     else [])
                    if purpose == 'inference':  # async send: the same rule, the hook gets the running loop
                        provider.faults.push('status', status=401, **({'message': message} if message else {}))

                        async def call():
                            response = await egress.send(grant(provider.origin, handle, upstream_id='oai'), upstream,
                                                         PROFILES['openai'], 'GET', '/models', stream=True)
                            await response.aclose()
                            await egress.aclose()
                            return asyncio.get_running_loop()
                        loop = run(call())
                        self.assertEqual(vault.record('openai').validation, 'ok')
                        self.assertEqual(suspects[1:], [('oai', 'openai', loop)] if message is None else [])


class ManagedRole(EgressCase):
    def test_every_purpose_is_refused_before_any_socket(self):
        upstream = Upstream(profile='custom_openai', base_url='https://upstream.example.invalid/v1')
        origin = 'https://upstream.example.invalid:443'
        for settings in (Settings(role='managed'), Settings(role='managed', managed_delegation='parent')):
            inner, sync_inner = CountingBackend(), CountingSyncBackend()
            resolved = []
            egress = self.client(settings, network_backend=inner, sync_network_backend=sync_inner,
                                 resolver=lambda host, port: resolved.append(host) or ['127.0.0.1'])
            for purpose in GRANT_PURPOSES:
                for delegated in (False, True):
                    allowed = (settings.managed_delegation == 'parent' and delegated and purpose == 'inference')
                    if allowed:
                        continue
                    with self.subTest(delegation=settings.managed_delegation, purpose=purpose, delegated=delegated):
                        g = grant(origin, purpose=purpose, delegated=delegated)
                        error = self.assertCode('role_forbids', egress.send_sync, g, upstream,
                                                PROFILES['custom_openai'], 'GET', '/models')
                        self.assertEqual(error.origin, 'policy')
                        self.assertEqual(error.response_headers()['x-should-retry'], 'false')
                        self.assertCodeAsync('role_forbids', egress.send(g, upstream, PROFILES['custom_openai'],
                                                                         'GET', '/models'))
            self.assertEqual((inner.connects, sync_inner.connects, resolved), ([], [], []))

    def test_delegated_inference_reaches_the_network_layer_only_with_parent_delegation(self):
        upstream = Upstream(profile='custom_openai', base_url='https://upstream.example.invalid/v1')
        sync_inner = CountingSyncBackend()
        egress = self.client(Settings(role='managed', managed_delegation='parent', broker_egress='loopback-only'),
                             sync_network_backend=sync_inner, resolver=lambda host, port: ['127.0.0.1'])
        with self.assertRaises(httpx.ConnectError):
            egress.send_sync(grant('https://upstream.example.invalid:443', delegated=True), upstream,
                             PROFILES['custom_openai'], 'GET', '/models')
        self.assertEqual(sync_inner.connects, [('127.0.0.1', 443)])


class Responses(EgressCase):
    def test_response_size_caps(self):
        settings = Settings(broker_egress='loopback-only', broker_max_upstream_json_bytes=256 * 1024)
        with OpenAIMock() as provider:
            upstream = Upstream(profile='openai', base_url=provider.api_base)
            origin = provider.origin
            egress = self.client(settings)
            provider.faults.push('oversized_body', size=1024 * 1024)
            self.assertCode('upstream_response_too_large', egress.send_sync, grant(origin, bearer(origin)), upstream,
                            PROFILES['openai'], 'GET', '/models')
            provider.faults.push('oversized_body', size=1024 * 1024, declared_length=True)
            self.assertCode('upstream_response_too_large', egress.send_sync, grant(origin, bearer(origin)), upstream,
                            PROFILES['openai'], 'GET', '/models')
            provider.faults.push('oversized_body', size=1024 * 1024)

            async def streamed():
                response = await egress.send(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET',
                                             '/models', stream=True)
                try:
                    with self.assertRaises(BrokerError) as ctx:
                        async for _ in response.aiter_raw():
                            pass
                    self.assertEqual(ctx.exception.code, 'upstream_response_too_large')
                finally:
                    await response.aclose()
                    await egress.aclose()
            run(streamed())
            ok = egress.send_sync(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET', '/models')
            self.assertEqual(ok.status_code, 200)

    def test_set_cookie_and_location_are_dropped_and_cookies_never_replayed(self):
        with OpenAIMock() as provider:
            upstream = Upstream(profile='openai', base_url=provider.api_base)
            origin = provider.origin
            egress = self.client()
            provider.faults.push('set_cookie', cookie='session=' + SENTINELS['leak'])
            first = egress.send_sync(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET', '/models')
            for name in ('set-cookie', 'location', 'server', 'date', 'content-length'):
                self.assertNotIn(name, first.headers)
            self.assertEqual(first.headers['content-type'].split(';')[0], 'application/json')
            egress.send_sync(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET', '/models')
            self.assertNotIn('cookie', provider.hits[1].headers)
            self.assertFalse(provider.hits[1].presented(SENTINELS['leak']))

    def test_hop_header_and_request_id_on_hop_profiles_only(self):
        with RouterUpstreamMock('hivemind') as router, OpenAIMock() as provider:
            egress = self.client()
            hop = 'abcdef0123456789'
            egress.send_sync(grant(router.origin, hop=hop, request_id='req-77'),
                             Upstream(profile='hivemind_gateway', base_url=router.api_base), PROFILES['hivemind_gateway'],
                             'POST', '/chat/completions', json={'model': 'm', 'messages': [{'role': 'user', 'content': 'x'}]},
                             headers={'X-Titanos-Hop': 'forged'})
            self.assertEqual(router.hops, [hop])
            self.assertEqual(router.hits[0].headers.get('x-request-id'), 'req-77')
            egress.send_sync(grant(provider.origin, bearer(provider.origin), hop=hop),
                             Upstream(profile='openai', base_url=provider.api_base), PROFILES['openai'], 'GET', '/models')
            self.assertNotIn('x-titanos-hop', provider.hits[0].headers)

    def test_first_byte_timeout(self):
        with OpenAIMock() as provider:
            upstream = Upstream(profile='openai', base_url=provider.api_base,
                                timeouts={'connect': 2, 'first_byte': 0.3, 'idle': 5})
            egress = self.client()
            provider.faults.push('slow_first_byte', delay=1.5)
            started = time.monotonic()
            with self.assertRaises(httpx.TimeoutException):
                egress.send_sync(grant(provider.origin, bearer(provider.origin)), upstream, PROFILES['openai'], 'GET',
                                 '/models')
            self.assertLess(time.monotonic() - started, 1.4)
            wait_until(lambda: provider.hits and provider.hits[0].completed, 5)  # let the mock finish before it stops

    def test_idle_timeout_between_stream_chunks(self):
        with OpenAIMock() as provider:
            upstream = Upstream(profile='openai', base_url=provider.api_base,
                                timeouts={'connect': 2, 'first_byte': 5, 'idle': 0.3})
            egress = self.client()
            provider.hold(frame=2, timeout=3)  # frame 1 arrives, then the stream stalls
            body = {'model': 'gpt-4.1-mini', 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}
            response = egress.send_sync(grant(provider.origin, bearer(provider.origin)), upstream, PROFILES['openai'],
                                        'POST', '/chat/completions', json=body, stream=True)
            started = time.monotonic()
            try:
                with self.assertRaises(httpx.ReadTimeout):
                    for _ in response.iter_raw():
                        pass
            finally:
                response.close()
                provider.gate.set()
            self.assertLess(time.monotonic() - started, 2.5)
            wait_until(lambda: provider.hits and (provider.hits[0].completed or provider.hits[0].disconnected), 5)

    def test_per_upstream_concurrency_slot(self):
        with OpenAIMock() as provider:
            upstream = Upstream(profile='openai', base_url=provider.api_base, max_concurrency=1,
                                timeouts={'connect': 2, 'first_byte': 0.4, 'idle': 5})
            egress = self.client()
            origin = provider.origin

            async def calls():
                held = await egress.send(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET', '/models',
                                         stream=True)
                with self.assertRaises(BrokerError) as ctx:
                    await egress.send(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET', '/models')
                self.assertEqual(ctx.exception.code, 'concurrency_exceeded')
                await held.aclose()
                again = await egress.send(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET', '/models')
                self.assertEqual(again.status_code, 200)
                await egress.aclose()
            run(calls())
            held = egress.send_sync(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET', '/models',
                                    stream=True)
            self.assertCode('concurrency_exceeded', egress.send_sync, grant(origin, bearer(origin)), upstream,
                            PROFILES['openai'], 'GET', '/models')
            held.close()
            self.assertEqual(egress.send_sync(grant(origin, bearer(origin)), upstream, PROFILES['openai'], 'GET',
                                              '/models').status_code, 200)


def body_stream(data: bytes) -> httpx.ByteStream:
    """A response body that httpx does not pre-read (as a real network stream), so the guarded stream sees every byte."""
    return httpx.ByteStream(data)


class Streamed:
    """MockTransport answering with a fixed status, headers and an unread body."""

    def __init__(self, body: bytes, headers: dict, status=200):
        self.body, self.headers, self.status, self.requests = body, headers, status, []
        self.sync = httpx.MockTransport(self.handle)
        self.async_ = httpx.MockTransport(self.handle)

    def handle(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, headers=self.headers, stream=body_stream(self.body))


class CompressedAndCapped(EgressCase):
    CAP = 256 * 1024
    UPSTREAM = Upstream(profile='custom_openai', base_url='http://127.0.0.1:9/v1')
    ORIGIN = 'http://127.0.0.1:9'

    def egress_for(self, transport: Streamed, **settings):
        values = dict(broker_egress='loopback-only', broker_max_upstream_json_bytes=self.CAP,
                      broker_max_upstream_media_bytes=self.CAP)
        values.update(settings)
        return self.client(Settings(**values), transport=transport.async_, sync_transport=transport.sync)

    def outcomes(self, egress):
        """(sync, async, async stream=True) outcome: the decoded byte count or the BrokerError code."""
        results = []
        try:
            results.append(len(egress.send_sync(grant(self.ORIGIN), self.UPSTREAM, PROFILES['custom_openai'], 'GET',
                                                '/models').content))
        except BrokerError as exc:
            results.append(exc.code)

        async def calls():
            try:
                response = await egress.send(grant(self.ORIGIN), self.UPSTREAM, PROFILES['custom_openai'], 'GET',
                                             '/models')
                results.append(len(response.content))
            except BrokerError as exc:
                results.append(exc.code)
            try:
                response = await egress.send(grant(self.ORIGIN), self.UPSTREAM, PROFILES['custom_openai'], 'GET',
                                             '/models', stream=True)
            except BrokerError as exc:
                results.append(exc.code)
                return
            total = 0
            try:
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                results.append(total)
            except BrokerError as exc:
                results.append(exc.code)
            finally:
                await response.aclose()
        run(calls())
        return results

    def test_gzip_and_deflate_bombs_are_refused_while_decoding(self):
        plain = b'0' * (self.CAP * 8)
        bodies = {'gzip': gzip.compress(plain, 9), 'x-gzip': gzip.compress(plain, 9), 'deflate': zlib.compress(plain, 9)}
        raw = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        bodies['deflate (raw)'] = raw.compress(plain) + raw.flush()
        for coding, body in bodies.items():
            with self.subTest(coding=coding):
                self.assertLess(len(body), self.CAP)  # small on the wire
                transport = Streamed(body, {'content-type': 'application/json',
                                            'content-encoding': coding.split(' ')[0]})
                self.assertEqual(self.outcomes(self.egress_for(transport)), ['upstream_response_too_large'] * 3)

    def test_a_transport_that_pre_reads_the_body_still_meets_the_cap(self):
        bomb = gzip.compress(b'0' * (self.CAP * 8), 9)

        def handle(request):  # httpx.Response(content=...) reads and decodes the body inside the transport itself
            return httpx.Response(200, headers={'content-type': 'application/json', 'content-encoding': 'gzip'},
                                  content=bomb)
        transport = Streamed(b'', {})
        transport.sync, transport.async_ = httpx.MockTransport(handle), httpx.MockTransport(handle)
        self.assertEqual(self.outcomes(self.egress_for(transport)), ['upstream_response_too_large'] * 3)

    def test_small_compressed_bodies_decode_once_and_lose_content_encoding(self):
        payload = json.dumps({'object': 'list', 'data': [{'id': 'm' * 40}] * 50}).encode()
        multi = gzip.compress(payload[:100]) + gzip.compress(payload[100:])  # two gzip members
        for coding, body in (('gzip', gzip.compress(payload)), ('deflate', zlib.compress(payload)), ('gzip', multi),
                             ('identity', payload)):
            with self.subTest(coding=coding, size=len(body)):
                transport = Streamed(body, {'content-type': 'application/json', 'content-encoding': coding})
                egress = self.egress_for(transport)
                response = egress.send_sync(grant(self.ORIGIN), self.UPSTREAM, PROFILES['custom_openai'], 'GET',
                                            '/models')
                self.assertEqual(response.content, payload)
                self.assertNotIn('content-encoding', response.headers)
                self.assertEqual(self.outcomes(egress), [len(payload)] * 3)

    def test_other_codings_and_corrupt_bodies_are_bad_responses(self):
        for coding, body in (('br', b'\x1b\x00'), ('zstd', b'\x28\xb5\x2f\xfd'), ('compress', b'\x1f\x9d'),
                             ('gzip, gzip', gzip.compress(gzip.compress(b'{}'))), ('gzip', b'not gzip at all')):
            with self.subTest(coding=coding):
                transport = Streamed(body, {'content-type': 'application/json', 'content-encoding': coding})
                self.assertEqual(self.outcomes(self.egress_for(transport)), ['upstream_bad_response'] * 3)

    def test_media_sse_and_ndjson_use_the_media_cap(self):
        big = b'x' * (96 * 1024)
        for content_type in ('text/event-stream', 'audio/mpeg', 'application/x-ndjson', 'image/png'):
            with self.subTest(content_type=content_type):
                transport = Streamed(big, {'content-type': content_type})
                capped = self.egress_for(transport, broker_max_upstream_json_bytes=1024 * 1024,
                                         broker_max_upstream_media_bytes=64 * 1024)
                self.assertEqual(self.outcomes(capped), ['upstream_response_too_large'] * 3)
                roomy = self.egress_for(transport, broker_max_upstream_json_bytes=1024,
                                        broker_max_upstream_media_bytes=1024 * 1024)
                self.assertEqual(self.outcomes(roomy), [len(big)] * 3)
        transport = Streamed(big, {'content-type': 'application/json'})
        self.assertEqual(self.outcomes(self.egress_for(transport, broker_max_upstream_json_bytes=64 * 1024,
                                                       broker_max_upstream_media_bytes=1024 * 1024)),
                         ['upstream_response_too_large'] * 3)


class RequestHygiene(EgressCase):
    def test_returned_request_carries_no_auth_header(self):
        cases = [('openai', 'bearer', 'authorization', 'https://api.openai.com:443', '/models'),
                 ('anthropic', 'anthropic', 'x-api-key', 'https://api.anthropic.com:443', '/v1/models'),
                 ('elevenlabs', 'xi', 'xi-api-key', 'https://api.elevenlabs.io:443', '/v1/models'),
                 ('gemini', 'goog', 'x-goog-api-key', 'https://generativelanguage.googleapis.com:443', '/models')]
        for profile_id, template, header, origin, path in cases:
            profile = PROFILES[profile_id]
            self.assertEqual(profile.auth, template)
            with self.subTest(profile=profile_id):
                capture = Capture()
                egress = self.client(INTERNET, transport=capture.async_, sync_transport=capture.sync)
                secret = SENTINELS.get(profile_id) or SENTINELS['openai']
                handle = CredentialHandle(origin, template, Secret(secret), credential_id=profile_id)
                upstream = Upstream(profile=profile_id)
                response = egress.send_sync(grant(origin, handle), upstream, profile, 'GET', path)
                self.assertIn(header, capture.requests[-1].headers)  # the transport saw the credential
                self.assertNotIn(header, response.request.headers)
                self.assertNotIn(secret, repr(response.request.headers) + str(dict(response.request.headers)))

                async def call():
                    try:
                        return await egress.send(grant(origin, handle), upstream, profile, 'GET', path, stream=True)
                    finally:
                        await egress.aclose()
                streamed = run(call())
                self.assertNotIn(header, streamed.request.headers)
                self.assertEqual(streamed.request.url, capture.requests[-1].url)

    def test_exception_requests_are_scrubbed(self):
        def fail(request):
            raise httpx.ConnectError('instrumented failure', request=request)
        egress = self.client(INTERNET, sync_transport=httpx.MockTransport(fail))
        origin = 'https://api.anthropic.com:443'
        handle = CredentialHandle(origin, 'anthropic', Secret(SENTINELS['anthropic']), credential_id='anthropic')
        with self.assertRaises(httpx.ConnectError) as ctx:
            egress.send_sync(grant(origin, handle), Upstream(profile='anthropic'), PROFILES['anthropic'], 'GET',
                             '/v1/models')
        self.assertNotIn('x-api-key', ctx.exception.request.headers)

    def test_deadline_clamps_connect_and_first_byte(self):
        capture = Capture()
        egress = self.client(sync_transport=capture.sync)
        upstream = Upstream(profile='custom_openai', base_url='http://127.0.0.1:9/v1',
                            timeouts={'connect': 20, 'first_byte': 60, 'idle': 30})
        egress.send_sync(grant('http://127.0.0.1:9', deadline=0.75), upstream, PROFILES['custom_openai'], 'GET',
                         '/models')
        timeout = capture.requests[0].extensions['timeout']
        self.assertLessEqual(timeout['connect'], 0.75)
        self.assertLessEqual(timeout['pool'], 0.75)
        self.assertLessEqual(timeout['write'], 0.75)
        self.assertEqual(timeout['read'], 30)  # the idle timeout takes over once the headers arrived

    def test_bookkeeping_failure_releases_the_slot(self):
        class Interrupted(BaseException):
            pass

        class ExplodingVault:
            def mark_validated(self, *args, **kwargs):
                raise Interrupted()  # stands in for a cancellation while the bookkeeping runs

            def touch_due(self, id):
                raise RuntimeError('broken vault')

            def touch(self, id):
                raise AssertionError('not reached')

        origin = 'http://127.0.0.1:9'
        upstream = Upstream(profile='custom_openai', base_url='http://127.0.0.1:9/v1', max_concurrency=1,
                            timeouts={'connect': 1, 'first_byte': 0.3, 'idle': 1})
        handle = CredentialHandle(origin, 'bearer', Secret(SENTINELS['openai']), credential_id='custom')
        for status in (401, 200):
            with self.subTest(status=status):
                # the 401 carries OpenAI's key-rejection code, so an inference call marks at once (R10)
                body = b'{"error": {"code": "invalid_api_key"}}' if status == 401 else b'{}'
                transport = Streamed(body, {'content-type': 'application/json'}, status=status)
                egress = self.client(vault=ExplodingVault(), transport=transport.async_, sync_transport=transport.sync)

                def sync_call():
                    return egress.send_sync(grant(origin, handle), upstream, PROFILES['custom_openai'], 'GET', '/models',
                                            stream=True)

                async def async_call():
                    return await egress.send(grant(origin, handle), upstream, PROFILES['custom_openai'], 'GET',
                                             '/models', stream=True)

                async def twice():
                    for _ in range(2):  # the second call would get concurrency_exceeded if the slot leaked
                        if status == 401:
                            with self.assertRaises(Interrupted):
                                await async_call()
                        else:
                            response = await async_call()
                            await response.aclose()
                    await egress.aclose()

                def all_calls():
                    run(twice())
                    for _ in range(2):
                        if status == 401:
                            with self.assertRaises(Interrupted):
                                sync_call()
                        else:
                            sync_call().close()
                if status == 401:
                    all_calls()
                else:
                    with self.assertLogs('titanos.broker', 'WARNING') as logs:  # logged, never raised
                        all_calls()
                    self.assertEqual(len(logs.records), 4)
                    self.assertNotIn(SENTINELS['openai'], '\n'.join(logs.output))


def _tls_material(directory: Path, hostname: str):
    """A throwaway CA and a leaf certificate for hostname (cryptography; nothing leaves the temp directory)."""
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key, key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'titanos test CA')])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
          .not_valid_after(now + datetime.timedelta(hours=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                       key_encipherment=False, data_encipherment=False, key_agreement=False,
                                       encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    leaf = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
            .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(hours=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=False, crl_sign=False,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    cert_path, key_path = directory / f'{hostname}.pem', directory / f'{hostname}.key'
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return ca.public_bytes(serialization.Encoding.PEM).decode('ascii'), cert_path, key_path


class _TlsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b'{"object":"list","data":[]}'
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(body)))
        self.send_header('connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class LoopbackTlsServer:
    """HTTPS on 127.0.0.1 with a certificate for a fixed name; records the SNI each handshake presented."""

    def __init__(self, cert_path: Path, key_path: Path):
        self.server_names = []
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert_path), str(key_path))
        context.sni_callback = lambda sock, name, ctx: self.server_names.append(name)
        self.httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _TlsHandler)
        self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)


def _have_cryptography() -> bool:
    try:
        import cryptography.x509  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_have_cryptography(), 'untried: the cryptography package is not installed (no throwaway CA)')
class TlsNameBinding(EgressCase):
    HOST = 'upstream.example.invalid'

    def setUp(self):
        super().setUp()
        self.dir = Path(tempfile.mkdtemp(prefix='titanos-egress-tls-'))
        self.addCleanup(lambda: __import__('shutil').rmtree(self.dir, ignore_errors=True))

    def call(self, ca_pem: str, port: int):
        context = ssl.create_default_context(cadata=ca_pem)
        resolved = []

        def resolver(host, port_):
            resolved.append(host)
            return ['127.0.0.1']
        with mock.patch.object(egress_module, '_ssl_context', lambda: context):
            egress = self.client(resolver=resolver)
            upstream = Upstream(profile='custom_openai', base_url=f'https://{self.HOST}:{port}/v1')
            origin = f'https://{self.HOST}:{port}'
            try:
                return egress.send_sync(grant(origin), upstream, PROFILES['custom_openai'], 'GET', '/models'), \
                    resolved, egress
            finally:
                egress.close()

    def test_sni_and_hostname_check_use_the_origin_host_over_the_checked_ip(self):
        ca_pem, cert, key = _tls_material(self.dir, self.HOST)
        with LoopbackTlsServer(cert, key) as server:
            response, resolved, egress = self.call(ca_pem, server.port)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(server.server_names, [self.HOST])  # SNI carried the origin host, not 127.0.0.1
            self.assertEqual(resolved, [self.HOST])
            self.assertEqual(egress.backends[0].policy.connects, 1)  # dialled the checked address

    def test_a_certificate_for_another_name_is_refused(self):
        ca_pem, cert, key = _tls_material(self.dir, 'other.example.invalid')
        with LoopbackTlsServer(cert, key) as server:
            with self.assertRaises(httpx.ConnectError) as ctx:
                self.call(ca_pem, server.port)
            self.assertIn('certificate', str(ctx.exception).lower())


if __name__ == '__main__':
    unittest.main()
