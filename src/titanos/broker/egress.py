"""EgressClient: the single outbound HTTP client of the broker, and the only module that attaches upstream credentials.

Every outbound call needs an EgressGrant (issued by router.GrantIssuer). send()/send_sync() then, in order and before
any socket opens:
1. purpose: role.purpose_allowed(switches, grant.purpose, delegated=grant.delegated), else 403 role_forbids (managed
   role refuses every non-delegated purpose, auth-none upstreams included);
2. deadline: an expired grant gives 504 deadline_exceeded;
3. URL: upstream base_url (broker.json, the profile default or the Azure binding; never ambient *_BASE_URL variables) +
   the path template filled with percent-encoded path_params; the Azure api-version query is added; credential-named
   query parameters (key, api_key, access_token) are refused;
4. origin: the request URL's origin (scheme, punycode lower-case host, explicit port; no userinfo) must equal the
   upstream's origin and grant.origin, else 502 egress_refused;
5. credential: grant.credential (a vault.CredentialHandle) must be bound to exactly that origin and use the profile's
   auth template; CredentialHandle.apply() then sets the AUTH_TEMPLATES header (it re-checks the request origin). An
   upstream whose profile needs auth and has no credential gives 503 upstream_unconfigured;
6. headers: inbound headers reduced to profiles.INBOUND_HEADER_ALLOW (never authorization, x-api-key, xi-api-key,
   x-goog-api-key, api-key, cookie or x-titanos-*); the upstream's static headers; anthropic-version and the operator's
   anthropic betas; X-Titanos-Hop for profiles with hop_header; Accept-Encoding: identity;
7. body: TITANOS_ONLY_FIELDS (worker, location, timeout) stripped from JSON bodies, raw JSON content (bytes or str,
   leading whitespace allowed) and form fields.

Transport: one httpx.AsyncClient and one httpx.Client per upstream, trust_env=False (no proxy or SSL environment),
follow_redirects=False, a cookie jar that stores nothing, and a transport over httpcore.(Async)ConnectionPool whose
network backend is GuardedBackend: it resolves the host itself and checks EVERY resolved address (check_address) before
connecting to the checked address; TLS SNI and certificate checks still use the origin host, which httpcore passes to
start_tls separately. Address policy:
- always refused: 0.0.0.0/8, ::, link-local 169.254.0.0/16 and fe80::/10, fd00:ec2::254, multicast, 240.0.0.0/4
  (broadcast included), fec0::/10, and the IPv4-mapped forms of these;
- an IPv4-mapped address (::ffff:a.b.c.d) is judged as its IPv4 address. A translated form (6to4 2002::/16, NAT64
  64:ff9b::/96, IPv4-compatible ::a.b.c.d) is refused when its embedded IPv4 address is refused or is loopback (a
  translator's loopback is not ours); otherwise the IPv6 address itself is judged as a public address;
- loopback is always allowed; RFC1918, 100.64.0.0/10 and fc00::/7 need policy.allow_http_lan;
- TITANOS_BROKER_EGRESS clamps everything: internet (default), lan (no public addresses) or loopback-only.

Responses: a 3xx gives 502 upstream_redirect_refused (nothing is followed); only allowlisted response headers are
returned (adapters.base.RESPONSE_HEADER_ALLOW plus the rate-limit families); bodies are capped by content type (JSON and
text: broker_max_upstream_json_bytes; media, SSE and NDJSON: broker_max_upstream_media_bytes; 502
upstream_response_too_large). Accept-Encoding: identity is requested; an upstream that still compresses with gzip or
deflate is decoded inside the capped stream by a bounded zlib decompressor, so the cap counts DECODED bytes (and the raw
bytes too); content-encoding is then dropped from the returned headers. Any other content-encoding (br, zstd, compress,
stacked codings) gives 502 upstream_bad_response. Timeouts: connect, first byte (headers) and idle (between body
chunks). A per-upstream
semaphore of upstream.max_concurrency bounds concurrent calls (429 concurrency_exceeded when no slot frees in time); the
slot is held until the response is closed. Credential marking (ratified R10, rules in
translate.upstream_errors.credential_failure): an upstream 401 marks the credential failed:upstream_auth (ElevenLabs:
only for its key-rejection codes, since it also answers 401 for an exhausted quota or a missing key permission; never
for a restricted key's per-endpoint refusal such as OpenAI's 401 'insufficient permissions ... Missing scopes'); a 403
(and a Gemini 400) marks it only when the body carries the provider's auth error code, so a plain 403 (moderation,
region, permission) never takes a shared credential offline. On an inference call (CONFIRM_PURPOSES: the caller chose
the endpoint, a native row or a typed surface, and the body) only an answer that carries the provider's key-rejection
code (upstream_errors.key_rejected) marks at once; any other 401 calls on_auth_suspect instead, and the Broker has the
upstream's own probe endpoint (purpose probe, GET models) answer for the credential: that answer marks it (or not) by
the same rules. So no caller can take a shared credential offline by calling an endpoint the operator's key may not
use. To decide, the body of a 401 or 403 (and of a Gemini 400) is read within
the cap, also for stream=True (callers then use response.content / aread()). Successful calls refresh last_used_at (at
most once a minute). The returned response's .request is a copy without the auth header
(and an httpx exception's .request has it removed), so a later repr or log of either never shows a provider key.

transport / sync_transport replace the whole transport (httpx.MockTransport in tests); network_backend /
sync_network_backend replace the inner network backend under GuardedBackend; resolver replaces getaddrinfo.

Native passthrough (R32, broker/native.py) calls send() with native=NativeOptions: the path may be relative to the
upstream ORIGIN instead of its base_url (relative_to='origin'), the query keeps repeated names, inbound headers from the
row's allowlist (never credentials, cookies, framing or X-Titanos-*) join INBOUND_HEADER_ALLOW, the body is sent exactly
as the client sent it (no Titanos-field stripping: on the native lane every argument passes untouched), the row's extra
response headers are kept, and every answer is capped at broker_max_upstream_media_bytes. Everything else above applies
unchanged (purpose, deadline, origin and credential binding, resolved-address policy, no redirects, concurrency slots).

open_websocket() is the WebSocket form of the same choke point (OpenAI Realtime, Gemini Live, ElevenLabs stream-input):
_prepare() builds and checks the GET request exactly as send() does (credential applied by CredentialHandle.apply for that
origin only), then a minimal RFC 6455 client runs over a GuardedBackend connection (every resolved address checked;
TLS with the origin host name for https origins): the opening handshake (Sec-WebSocket-Key / Accept verified, the
upstream's 101 required; any other status is read within 64 KiB and classified like an HTTP error, with R10 credential
marking), then masked client frames, ping/pong, fragmentation and the close handshake, every message capped at
broker_max_upstream_media_bytes. No WebSocket library is needed on the client side.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import os
import inspect
import ipaddress
import json as jsonlib
import logging
import re
import socket
import threading
import time
import zlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from http.cookiejar import CookieJar, DefaultCookiePolicy
from urllib.parse import quote, urlsplit

import httpcore
import httpx

from .adapters.base import (filter_inbound_headers, filter_response_headers, profile_inbound_headers,
                            profile_response_headers)
from .config import METADATA_ADDRESSES, RFC1918, parse_origin
from .errors import BrokerError, EgressRefused
from .profiles import AUTH_TEMPLATES as _AUTH_HEADERS
from .profiles import TITANOS_ONLY_FIELDS
from .role import RoleSwitches, purpose_allowed
from .translate.upstream_errors import credential_failure, key_rejected
from .types import EgressGrant
from .vault import CredentialHandle, request_origin

log = logging.getLogger('titanos.broker')

AUTH_TEMPLATES = frozenset(_AUTH_HEADERS)  # bearer, anthropic, xi, azure_api_key, goog, none
_AUTH_HEADER_NAMES = frozenset(spec[0].lower() for spec in _AUTH_HEADERS.values() if spec)
EGRESS_MODES = ('internet', 'lan', 'loopback-only')
HOP_HEADER = 'X-Titanos-Hop'
QUERY_CREDENTIALS = frozenset({'key', 'api_key', 'apikey', 'access_token'})
AUTH_FAILURE_PURPOSES = frozenset({'inference', 'probe', 'catalog', 'advisor'})
# purposes whose endpoint and body a caller chooses: a 401 without the provider's key-rejection code marks the credential
# only after the upstream's probe endpoint rejects it too (on_auth_suspect)
CONFIRM_PURPOSES = frozenset({'inference'})
_V4 = ipaddress.IPv4Network
_V6 = ipaddress.IPv6Network
BLOCKED_NETWORKS = (
    _V4('0.0.0.0/8'), _V4('169.254.0.0/16'), _V4('224.0.0.0/4'), _V4('240.0.0.0/4'),
    _V6('::/128'), _V6('fe80::/10'), _V6('fec0::/10'), _V6('ff00::/8'),
)
LAN_NETWORKS = (*RFC1918, _V4('100.64.0.0/10'), _V6('fc00::/7'))
_NAT64 = _V6('64:ff9b::/96')
_COMPAT = _V6('::/96')
_SEGMENT_SAFE = "-._~:@!$&'()*+,;="
_PATH_BAD = re.compile(r'[\x00-\x20\x7f\\#]')
_DOT_SEGMENT = re.compile(r'(?:^|/)(?:\.|%2e){1,2}(?:/|$)', re.IGNORECASE)
_JSON_OBJECT_START = re.compile(rb'(?:\xef\xbb\xbf)?[ \t\r\n]*\{')
_JSON_TYPES = ('application/json', 'text/plain', 'text/html', 'application/problem+json')


# ---------------------------------------------------------------------------------------------------------------------
# address policy

def _translated_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address embedded in a 6to4, NAT64 or IPv4-compatible address (not IPv4-mapped: that one IS the IPv4
    address and is judged as such)."""
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip in _NAT64 or (ip in _COMPAT and int(ip) > 1):
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def check_address(ip, *, mode: str, allow_lan: bool) -> None:
    """Raise EgressRefused unless a resolved address may be connected to under mode (internet | lan | loopback-only)
    and allow_lan (policy.allow_http_lan)."""
    if mode not in EGRESS_MODES:
        raise EgressRefused(f'Unknown egress mode {mode!r}')
    try:
        address = ip if isinstance(ip, (ipaddress.IPv4Address, ipaddress.IPv6Address)) else \
            ipaddress.ip_address(str(ip).strip('[]').split('%', 1)[0])
    except ValueError:
        raise EgressRefused('The upstream resolved to something that is not an IP address') from None
    if address.version == 6:
        if address.ipv4_mapped is not None:
            return check_address(address.ipv4_mapped, mode=mode, allow_lan=allow_lan)
        embedded = _translated_ipv4(address)
        if embedded is not None:
            # the embedded address must pass on its own (a translator would reach it) and may not be loopback (that is
            # the translator's loopback, not ours); the IPv6 address itself is then judged below as a public address
            if embedded.is_loopback:
                raise EgressRefused(f'Outbound connection to {address} (embedded loopback) refused')
            check_address(embedded, mode=mode, allow_lan=allow_lan)
    if address in METADATA_ADDRESSES or any(address in net for net in BLOCKED_NETWORKS):
        raise EgressRefused(f'Outbound connection to blocked address {address} refused')
    if address.is_loopback:
        return
    if mode == 'loopback-only':
        raise EgressRefused(f'TITANOS_BROKER_EGRESS=loopback-only refuses {address}')
    if any(address in net for net in LAN_NETWORKS):
        if not allow_lan:
            raise EgressRefused(f'Private address {address} needs policy.allow_http_lan')
        return
    if mode == 'lan':
        raise EgressRefused(f'TITANOS_BROKER_EGRESS=lan refuses public address {address}')


def _literal(host: str):
    try:
        return ipaddress.ip_address(host.strip('[]'))
    except ValueError:
        return None


def _dedupe(addresses: Iterable) -> list[str]:
    seen, out = set(), []
    for value in addresses:
        text = str(value[4][0] if isinstance(value, tuple) and len(value) == 5 else value).split('%', 1)[0]
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


class _Policy:
    def __init__(self, mode: str, allow_lan: Callable[[], bool]):
        if mode not in EGRESS_MODES:
            raise ValueError(f'Unknown egress mode {mode!r}')
        self.mode = mode
        self.allow_lan = allow_lan
        self.refused = 0
        self.connects = 0

    def check(self, addresses: list[str], host: str) -> None:
        if not addresses:
            raise httpcore.ConnectError(f'{host} did not resolve to any address')
        allow = bool(self.allow_lan())
        try:
            for address in addresses:
                check_address(address, mode=self.mode, allow_lan=allow)
        except EgressRefused:
            self.refused += 1
            raise


class GuardedBackend(httpcore.AsyncNetworkBackend):
    """httpcore network backend that resolves the host and checks every resolved address before connecting."""

    def __init__(self, *, mode: str = 'internet', allow_lan: Callable[[], bool] | bool = False, resolver=None,
                 inner: httpcore.AsyncNetworkBackend | None = None):
        self.policy = _Policy(mode, allow_lan if callable(allow_lan) else (lambda: bool(allow_lan)))
        self.resolver = resolver
        self.inner = inner or httpcore.AnyIOBackend()

    async def _resolve(self, host: str, port: int) -> list[str]:
        literal = _literal(host)
        if literal is not None:
            return [str(literal)]
        try:
            if self.resolver is not None:
                result = self.resolver(host, port)
                if inspect.isawaitable(result):
                    result = await result
            else:
                import anyio
                result = await anyio.to_thread.run_sync(lambda: socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError as exc:
            raise httpcore.ConnectError(f'Could not resolve {host} ({type(exc).__name__})') from None
        return _dedupe(result)

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        addresses = await self._resolve(host, port)
        self.policy.check(addresses, host)
        last: Exception | None = None
        for address in addresses:
            try:
                self.policy.connects += 1
                return await self.inner.connect_tcp(address, port, timeout=timeout, local_address=local_address,
                                                    socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last = exc
        raise last if last is not None else httpcore.ConnectError(f'Could not connect to {host}')

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise EgressRefused('Unix-socket egress is not used by the broker')

    async def sleep(self, seconds: float) -> None:
        await self.inner.sleep(seconds)


class GuardedSyncBackend(httpcore.NetworkBackend):
    """The synchronous GuardedBackend (worker-thread callers: jobs and API Plus stages)."""

    def __init__(self, *, mode: str = 'internet', allow_lan: Callable[[], bool] | bool = False, resolver=None,
                 inner: httpcore.NetworkBackend | None = None):
        self.policy = _Policy(mode, allow_lan if callable(allow_lan) else (lambda: bool(allow_lan)))
        self.resolver = resolver
        self.inner = inner or httpcore.SyncBackend()

    def _resolve(self, host: str, port: int) -> list[str]:
        literal = _literal(host)
        if literal is not None:
            return [str(literal)]
        try:
            result = self.resolver(host, port) if self.resolver is not None else \
                socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise httpcore.ConnectError(f'Could not resolve {host} ({type(exc).__name__})') from None
        if inspect.isawaitable(result):
            raise TypeError('The synchronous backend needs a synchronous resolver')
        return _dedupe(result)

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        addresses = self._resolve(host, port)
        self.policy.check(addresses, host)
        last: Exception | None = None
        for address in addresses:
            try:
                self.policy.connects += 1
                return self.inner.connect_tcp(address, port, timeout=timeout, local_address=local_address,
                                              socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last = exc
        raise last if last is not None else httpcore.ConnectError(f'Could not connect to {host}')

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise EgressRefused('Unix-socket egress is not used by the broker')

    def sleep(self, seconds: float) -> None:
        self.inner.sleep(seconds)


@functools.lru_cache(maxsize=1)
def _ssl_context():
    """One verifying TLS context (certifi roots, no SSL_CERT_* environment) shared by every guarded pool."""
    return httpx.create_ssl_context(trust_env=False)


def _limits(max_connections: int) -> httpx.Limits:
    return httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections, keepalive_expiry=5.0)


class _GuardedAsyncTransport(httpx.AsyncHTTPTransport):
    """httpx's transport with its connection pool rebuilt over a guarded network backend."""

    def __init__(self, backend: httpcore.AsyncNetworkBackend, max_connections: int):
        # httpx's own __init__ would build (and load CA roots for) a pool that is replaced here, so it is not called;
        # handle_request uses only self._pool
        limits = _limits(max_connections)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=_ssl_context(), max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections, keepalive_expiry=limits.keepalive_expiry,
            http1=True, http2=False, retries=0, network_backend=backend)


class _GuardedSyncTransport(httpx.HTTPTransport):
    def __init__(self, backend: httpcore.NetworkBackend, max_connections: int):
        # httpx's own __init__ would build (and load CA roots for) a pool that is replaced here, so it is not called;
        # handle_request uses only self._pool
        limits = _limits(max_connections)
        self._pool = httpcore.ConnectionPool(
            ssl_context=_ssl_context(), max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections, keepalive_expiry=limits.keepalive_expiry,
            http1=True, http2=False, retries=0, network_backend=backend)


def _no_cookies() -> CookieJar:
    """A jar that stores nothing: upstream Set-Cookie never comes back on a later request."""
    return CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))


# ---------------------------------------------------------------------------------------------------------------------
# response wrappers: size cap, slot release

class _Release:
    def __init__(self, release: Callable[[], None]):
        self._release = release
        self._done = False
        self._lock = threading.Lock()

    def __call__(self) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        self._release()


def _too_large(cap: int) -> BrokerError:
    return BrokerError(f'Upstream response exceeds the {cap}-byte limit', 'upstream_response_too_large',
                       origin='upstream')


def _bad_response(message: str) -> BrokerError:
    return BrokerError(message, 'upstream_bad_response', origin='upstream')


def _running_loop() -> asyncio.AbstractEventLoop | None:
    """The event loop of the calling thread, or None (send_sync runs in worker threads)."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


_DECODABLE = {'gzip': 'gzip', 'x-gzip': 'gzip', 'deflate': 'deflate'}


def _decoder_for(values: list[str]) -> '_BoundedDecoder | None':
    """The decoder for a response's content-encoding values (None for identity); anything that cannot be decoded with a
    bounded output is refused."""
    codings = [v.strip().lower() for v in values if v.strip() and v.strip().lower() != 'identity']
    if not codings:
        return None
    if len(codings) > 1 or codings[0] not in _DECODABLE:
        raise _bad_response('The upstream answered with a content-encoding the broker does not accept (identity was '
                            'requested)')
    return _BoundedDecoder(_DECODABLE[codings[0]])


class _BoundedDecoder:
    """gzip / deflate decoding that never produces more than the remaining budget plus one byte per step, so a
    compression bomb is refused as upstream_response_too_large without being inflated."""

    def __init__(self, coding: str):
        self.coding = coding
        self._started = False
        self._obj = self._new()

    def _new(self, raw: bool = False):
        if self.coding == 'gzip':
            return zlib.decompressobj(16 + zlib.MAX_WBITS)
        return zlib.decompressobj(-zlib.MAX_WBITS if raw else zlib.MAX_WBITS)

    def decode(self, data: bytes, budget: int, cap: int) -> bytes:
        out: list[bytes] = []
        produced = 0
        while data:
            if self._obj.eof:
                if self.coding != 'gzip':
                    break  # trailing bytes after a deflate stream are ignored (as httpx does)
                self._obj = self._new()  # a further gzip member
            try:
                chunk = self._obj.decompress(data, max(budget - produced, 0) + 1)
            except zlib.error:
                if self.coding == 'deflate' and not self._started:
                    self._started = True
                    self._obj = self._new(raw=True)  # some servers send raw deflate without the zlib header
                    continue
                raise _bad_response('The upstream sent a body that does not decode') from None
            self._started = True
            produced += len(chunk)
            if produced > budget:
                raise _too_large(cap)
            out.append(chunk)
            data = self._obj.unconsumed_tail or (self._obj.unused_data if self._obj.eof else b'')
        return b''.join(out)

    def flush(self, budget: int, cap: int) -> bytes:
        try:
            tail = self._obj.flush()
        except zlib.error:
            raise _bad_response('The upstream sent a body that does not decode') from None
        if len(tail) > budget:
            raise _too_large(cap)
        return tail


class _Counter:
    """Raw and decoded byte counting against one cap."""

    def __init__(self, cap: int, decoder: _BoundedDecoder | None):
        self.cap, self.decoder, self.raw, self.decoded = cap, decoder, 0, 0

    def feed(self, chunk: bytes) -> bytes:
        self.raw += len(chunk)
        if self.raw > self.cap:
            raise _too_large(self.cap)
        if self.decoder is not None:
            chunk = self.decoder.decode(chunk, self.cap - self.decoded, self.cap)
        self.decoded += len(chunk)
        if self.decoded > self.cap:
            raise _too_large(self.cap)
        return chunk

    def finish(self) -> bytes:
        if self.decoder is None:
            return b''
        tail = self.decoder.flush(self.cap - self.decoded, self.cap)
        self.decoded += len(tail)
        return tail


class _CappedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, inner, cap: int, release: _Release, decoder: _BoundedDecoder | None = None):
        self._inner, self._cap, self._release, self._decoder = inner, cap, release, decoder

    async def __aiter__(self):
        counter = _Counter(self._cap, self._decoder)
        async for chunk in self._inner:
            chunk = counter.feed(chunk)
            if chunk:
                yield chunk
        tail = counter.finish()
        if tail:
            yield tail

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            self._release()


class _CappedSyncStream(httpx.SyncByteStream):
    def __init__(self, inner, cap: int, release: _Release, decoder: _BoundedDecoder | None = None):
        self._inner, self._cap, self._release, self._decoder = inner, cap, release, decoder

    def __iter__(self):
        counter = _Counter(self._cap, self._decoder)
        for chunk in self._inner:
            chunk = counter.feed(chunk)
            if chunk:
                yield chunk
        tail = counter.finish()
        if tail:
            yield tail

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            self._release()


# ---------------------------------------------------------------------------------------------------------------------
# the client

@dataclass(frozen=True)
class NativeOptions:
    """send()/open_websocket() options of the R32 native passthrough (see the module docstring)."""
    relative_to: str = 'base'  # 'base' (the upstream base_url) or 'origin' (its scheme://host:port)
    request_headers: frozenset = frozenset()  # extra inbound header names forwarded (lower-case)
    response_headers: frozenset = frozenset()  # extra upstream response header names returned (lower-case)


_NATIVE_NEVER = frozenset({'authorization', 'x-api-key', 'xi-api-key', 'x-goog-api-key', 'api-key', 'cookie',
                           'proxy-authorization', 'host', 'content-length', 'transfer-encoding', 'connection', 'upgrade',
                           'te', 'trailer', 'keep-alive', 'accept-encoding', 'origin'})


@dataclass(frozen=True)
class _Plan:
    key: tuple
    origin: str
    max_concurrency: int
    connect: float
    first_byte: float
    idle: float


class EgressClient:
    """See the module docstring. One instance per Broker; aclose()/close() release every pooled connection."""

    def __init__(self, settings, vault, switches: RoleSwitches, *, transport: httpx.AsyncBaseTransport | None = None,
                 sync_transport: httpx.BaseTransport | None = None,
                 network_backend: httpcore.AsyncNetworkBackend | None = None,
                 sync_network_backend: httpcore.NetworkBackend | None = None, resolver=None,
                 allow_http_lan: Callable[[], bool] | bool = False, clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.vault = vault
        self.switches = switches
        self.mode = getattr(settings, 'broker_egress', 'internet')
        if self.mode not in EGRESS_MODES:
            raise ValueError(f'Unknown broker_egress mode {self.mode!r}')
        self._allow_lan = allow_http_lan if callable(allow_http_lan) else (lambda value=bool(allow_http_lan): value)
        self._transport = transport
        self._sync_transport = sync_transport
        self._network_backend = network_backend
        self._sync_network_backend = sync_network_backend
        self._resolver = resolver
        self._clock = clock
        self._lock = threading.Lock()
        self._async_clients: dict[tuple, httpx.AsyncClient] = {}
        self._sync_clients: dict[tuple, httpx.Client] = {}
        self._async_slots: dict[tuple, tuple[asyncio.AbstractEventLoop, asyncio.Semaphore]] = {}
        self._sync_slots: dict[tuple, threading.BoundedSemaphore] = {}
        self.backends: list = []  # GuardedBackend / GuardedSyncBackend instances (diagnostics and tests)
        # on_auth_suspect(upstream_id, credential_id, loop or None): called (from any thread) for an inference 401 that
        # does not carry the provider's key-rejection code; the Broker installs Broker._auth_suspect. None: never marked
        self.on_auth_suspect: Callable[[str | None, str, asyncio.AbstractEventLoop | None], None] | None = None

    # request preparation ------------------------------------------------------------------------------------------
    def _check_purpose(self, grant: EgressGrant) -> None:
        if not isinstance(grant, EgressGrant):
            raise TypeError('EgressClient needs an EgressGrant issued by router.GrantIssuer')
        try:
            allowed = purpose_allowed(self.switches, grant.purpose, delegated=grant.delegated)
        except ValueError:
            allowed = False
        if not allowed:
            raise BrokerError(f'Egress purpose {grant.purpose!r} is not allowed in this role', 'role_forbids',
                              origin='policy')

    @staticmethod
    def _format_path(template: str, path_params: Mapping[str, object] | None) -> str:
        if not isinstance(template, str) or not template.startswith('/'):
            raise EgressRefused('Upstream paths are relative to the base_url and start with /')
        path, _, query = template.partition('?')
        if path_params:
            values = {}
            for name, value in path_params.items():
                text = str(value)
                if text in ('', '.', '..'):
                    raise EgressRefused(f'Path parameter {name} is empty or a dot segment')
                values[name] = quote(text, safe=_SEGMENT_SAFE)
            try:
                path = path.format_map(values)
            except (KeyError, IndexError, ValueError):
                raise EgressRefused('Path template parameters are missing') from None
        if '{' in path or '}' in path or _PATH_BAD.search(path) or '//' in path or _DOT_SEGMENT.search(path):
            raise EgressRefused('Upstream path is not a plain path')
        return path + ('?' + query if query else '')

    def _plan(self, key: tuple, origin: str, upstream, remaining: float | None) -> _Plan:
        timeouts = getattr(upstream, 'timeouts', None) if upstream is not None else None
        settings = self.settings
        connect = timeouts.connect if timeouts else settings.broker_connect_timeout
        first_byte = timeouts.first_byte if timeouts else settings.broker_first_byte_timeout
        idle = timeouts.idle if timeouts else settings.broker_stream_idle_timeout
        if remaining is not None:
            connect, first_byte = min(connect, remaining), min(first_byte, remaining)
        concurrency = getattr(upstream, 'max_concurrency', None) or 16
        return _Plan(key, origin, int(concurrency), float(connect), float(first_byte), float(idle))

    def _prepare(self, grant: EgressGrant, upstream, profile, method: str, path: str, *, json, content, data, files,
                 params, headers, path_params, native: NativeOptions | None = None) -> tuple[httpx.Request, _Plan]:
        self._check_purpose(grant)
        remaining = None
        if grant.deadline is not None:
            remaining = grant.deadline - self._clock()
            if remaining <= 0:
                raise BrokerError('The request deadline passed before the upstream call', 'deadline_exceeded')
        allow_lan = bool(self._allow_lan())

        if upstream is None:  # advisors and tests: an absolute URL whose origin is the grant's
            if profile is not None:
                raise EgressRefused('A profile needs its upstream')
            parts = urlsplit(path)
            if not parts.scheme or not parts.netloc or parts.fragment:
                raise EgressRefused('Without an upstream the path must be an absolute URL')
            try:
                upstream_origin = parse_origin(f'{parts.scheme}://{parts.netloc}', allow_http_lan=allow_lan)
            except ValueError as exc:
                raise EgressRefused(f'Refused URL: {exc}') from None
            url = f'{parts.scheme}://{parts.netloc}' + self._format_path(
                (parts.path or '/') + ('?' + parts.query if parts.query else ''), path_params)
            key = ('url', upstream_origin)
        else:
            if profile is None:
                raise EgressRefused('An upstream call needs its profile')
            if profile.kind == 'slot' or profile.adapter == 'none':
                raise BrokerError(f'Profile {profile.id} has no wire contract', 'upstream_unconfigured')
            base_url = upstream.effective_base_url()
            if not base_url:
                raise BrokerError('The upstream has no base_url', 'upstream_unconfigured')
            try:
                upstream_origin = parse_origin(base_url, allow_http_lan=allow_lan)
            except ValueError as exc:
                raise EgressRefused(f'Refused upstream base_url: {exc}') from None
            root = base_url.rstrip('/')
            if native is not None and native.relative_to == 'origin':
                parts = urlsplit(base_url)
                root = f'{parts.scheme}://{parts.netloc}'
            url = root + self._format_path(path, path_params)
            key = ('upstream', grant.upstream_id, upstream_origin)
        if grant.origin != upstream_origin:
            raise EgressRefused('The grant was issued for another origin')

        if native is not None and isinstance(params, (list, tuple)):
            query = [(str(k), str(v)) for k, v in params]  # repeated names kept (include[]=a&include[]=b)
            if profile is not None and profile.id == 'azure_openai' and getattr(upstream, 'azure', None) is not None:
                name = profile.extras.get('api_version_query', 'api-version')
                if not any(k == name for k, _ in query):
                    query.append((name, upstream.azure.api_version))
        else:
            query = dict(params or {})
            if profile is not None and profile.id == 'azure_openai' and getattr(upstream, 'azure', None) is not None:
                query.setdefault(profile.extras.get('api_version_query', 'api-version'), upstream.azure.api_version)

        out_headers = self._headers(grant, upstream, profile, headers, native=native)
        if native is not None:
            body_json, body_content, body_data = json, content, data  # native lane: every argument passes untouched
        else:
            body_json, body_content, body_data = self._strip_body(json, content, data)
        plan = self._plan(key, upstream_origin, upstream, remaining)
        timeout = httpx.Timeout(connect=plan.connect, read=plan.first_byte, write=plan.first_byte, pool=plan.first_byte)
        try:
            request = httpx.Request(method.upper(), url, params=query or None, headers=out_headers, json=body_json,
                                    content=body_content, data=body_data, files=files,
                                    extensions={'timeout': timeout.as_dict()})
            same_origin = not request.url.userinfo and request_origin(request.url) == upstream_origin
        except (httpx.InvalidURL, UnicodeError, ValueError):
            raise EgressRefused('The request URL is not valid') from None
        if not same_origin:
            raise EgressRefused('The request URL left the upstream origin')
        if any(name.lower() in QUERY_CREDENTIALS for name in request.url.params.keys()):
            raise EgressRefused('Credentials are never sent in URL query parameters')

        handle = grant.credential
        if handle is not None:
            if not isinstance(handle, CredentialHandle):
                raise TypeError('grant.credential must be a vault.CredentialHandle')
            if profile is not None:
                chosen = getattr(upstream, 'auth', None)  # broker.json Upstream.auth (R11); None = the profile default
                allowed = {chosen} if chosen else {profile.auth} | set(profile.extras.get('auth_options', ()))
                if handle.template not in allowed:
                    raise EgressRefused('The credential uses another auth scheme than the upstream profile')
            if handle.origin != upstream_origin:
                raise EgressRefused('The credential is bound to another origin; re-validate it for this upstream')
            handle.apply(request)  # sets the AUTH_TEMPLATES header after re-checking the request origin
        elif profile is not None and profile.auth != 'none' and not profile.auth_optional:
            raise BrokerError('The upstream needs a credential and none is configured', 'upstream_unconfigured')
        return request, plan

    @staticmethod
    def _headers(grant: EgressGrant, upstream, profile, headers, *, native: NativeOptions | None = None) -> dict[str, str]:
        out = filter_inbound_headers(headers or {}, extra=profile_inbound_headers(profile))
        if native is not None:
            allowed = {'content-type', 'accept'} | set(native.request_headers)
            items = headers.items() if hasattr(headers, 'items') else (headers or ())
            for name, value in items:
                name = name.decode('latin-1') if isinstance(name, bytes) else str(name)
                value = value.decode('latin-1') if isinstance(value, bytes) else str(value)
                lowered = name.lower()
                if lowered in allowed and lowered not in _NATIVE_NEVER and lowered not in _AUTH_HEADER_NAMES \
                        and not lowered.startswith('x-titanos-'):
                    out[lowered] = value
        if upstream is not None:
            for name, value in (getattr(upstream, 'headers', None) or {}).items():
                if profile is None or name in profile.static_headers:
                    out[name] = value
        if profile is not None and profile.adapter == 'anthropic':
            out['anthropic-version'] = (getattr(upstream, 'anthropic_version', None)
                                        or profile.extras.get('default_anthropic_version', '2023-06-01'))
            betas = [b.strip() for b in out.get('anthropic-beta', '').split(',') if b.strip()]
            for beta in getattr(upstream, 'anthropic_betas', None) or ():
                if beta not in betas:
                    betas.append(beta)
            if betas:
                out['anthropic-beta'] = ','.join(betas)
        if profile is not None and profile.hop_header and grant.hop_header:
            out[HOP_HEADER] = grant.hop_header
        if profile is not None and profile.extras.get('forward_request_id') and grant.request_id:
            out['X-Request-ID'] = grant.request_id
        out['Accept-Encoding'] = 'identity'
        out.setdefault('accept', '*/*')
        return out

    @staticmethod
    def _strip_body(json, content, data):
        if json is not None and content is not None:
            raise ValueError('Pass json or content, not both')
        if isinstance(json, Mapping):
            json = {k: v for k, v in json.items() if k not in TITANOS_ONLY_FIELDS}
        if isinstance(data, Mapping):
            data = {k: v for k, v in data.items() if k not in TITANOS_ONLY_FIELDS}
        raw = content.encode('utf-8') if isinstance(content, str) else content
        if isinstance(raw, (bytes, bytearray)) and _JSON_OBJECT_START.match(raw):
            try:
                parsed = jsonlib.loads(bytes(raw))
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and TITANOS_ONLY_FIELDS & parsed.keys():
                content = jsonlib.dumps({k: v for k, v in parsed.items() if k not in TITANOS_ONLY_FIELDS},
                                        separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        return json, content, data

    # responses ------------------------------------------------------------------------------------------------
    def _cap(self, content_type: str) -> int:
        kind = content_type.split(';', 1)[0].strip().lower()
        if kind.startswith(_JSON_TYPES) or (kind.startswith('text/') and kind != 'text/event-stream'):
            return int(self.settings.broker_max_upstream_json_bytes)
        return int(self.settings.broker_max_upstream_media_bytes)

    def _guard_response(self, response: httpx.Response, plan: _Plan, release: _Release, *, sync: bool,
                        native: NativeOptions | None = None, profile=None, media_cap: bool = False) -> None:
        cap = self._cap(response.headers.get('content-type', ''))
        if native is not None or media_cap:
            cap = int(self.settings.broker_max_upstream_media_bytes)
        refused: BrokerError | None = None
        try:
            decoder = _decoder_for(response.headers.get_list('content-encoding', split_commas=True))
        except BrokerError as exc:
            decoder, refused = None, exc
        response.stream = (_CappedSyncStream if sync else _CappedAsyncStream)(response.stream, cap, release, decoder)
        if 300 <= response.status_code < 400:
            raise BrokerError(f'The upstream answered a redirect ({response.status_code}); redirects are never followed',
                              'upstream_redirect_refused', origin='upstream')
        if refused is not None:
            raise refused
        declared = response.headers.get('content-length')
        if declared and declared.isdigit() and int(declared) > cap:
            raise _too_large(cap)
        if response.is_stream_consumed and len(response.content) > cap:
            raise _too_large(cap)  # a transport that hands back an already-read (and decoded) body: the cap still holds
        extra = (native.response_headers if native is not None else frozenset()) | profile_response_headers(profile)
        kept = filter_response_headers(response.headers.multi_items(), extra=extra)
        kept.pop('content-encoding', None)  # decoded (bounded) inside the capped stream, so httpx must not decode again
        response.headers = httpx.Headers(kept)
        timeouts = response.request.extensions.get('timeout')
        if isinstance(timeouts, dict):
            timeouts['read'] = plan.idle  # body chunks: the idle timeout from here on

    def _needs_body(self, grant: EgressGrant, response: httpx.Response, profile) -> bool:
        """Whether the credential decision needs the error body (a 401 or 403, or a Gemini 400; R10)."""
        handle = grant.credential
        if self.vault is None or handle is None or not handle.credential_id or grant.purpose not in AUTH_FAILURE_PURPOSES:
            return False
        status = response.status_code
        adapter = getattr(profile, 'adapter', None)
        return status in (401, 403) or (status == 400 and adapter == 'gemini')

    def _after(self, grant: EgressGrant, response: httpx.Response, profile=None) -> tuple[Callable, tuple] | None:
        """The credential bookkeeping call for this response, or None. Never raises."""
        handle = grant.credential
        if self.vault is None or handle is None or not handle.credential_id:
            return None
        try:
            status = response.status_code
            if status in (400, 401, 403) and grant.purpose in AUTH_FAILURE_PURPOSES:
                adapter = getattr(profile, 'adapter', None)
                try:
                    body = response.content
                except (httpx.ResponseNotRead, httpx.StreamError):
                    body = b''
                if not credential_failure(status, body, adapter=adapter):
                    return None
                if grant.purpose in CONFIRM_PURPOSES and not key_rejected(status, body, adapter=adapter):
                    # the caller chose this endpoint and body: a 401 without the provider's key-rejection code (a
                    # restricted key refused there, say) is checked against the upstream's own probe endpoint first
                    hook = self.on_auth_suspect
                    if hook is None:
                        return None
                    return hook, (grant.upstream_id, handle.credential_id, _running_loop())
                return self.vault.mark_validated, (handle.credential_id, False, 'upstream_auth', None)
            touch, due = getattr(self.vault, 'touch', None), getattr(self.vault, 'touch_due', None)
            if 200 <= response.status_code < 300 and callable(touch) and callable(due) and due(handle.credential_id):
                return touch, (handle.credential_id,)
        except Exception as exc:  # bookkeeping never fails a request
            log.warning('egress: credential bookkeeping failed (%s)', type(exc).__name__)
        return None

    @staticmethod
    def _scrub_request(request: httpx.Request) -> None:
        """Remove every auth-template header from a request that has been sent (exception paths)."""
        for name in _AUTH_HEADER_NAMES:
            request.headers.pop(name, None)

    @staticmethod
    def _sanitize(response: httpx.Response) -> None:
        """Give the returned response a copy of its request without the auth header (the transport saw the original)."""
        request = response.request
        if not any(name in request.headers for name in _AUTH_HEADER_NAMES):
            return
        headers = request.headers.copy()
        for name in _AUTH_HEADER_NAMES:
            headers.pop(name, None)
        try:
            body = {'content': request.content}
        except httpx.RequestNotRead:
            body = {'stream': request.stream}
        response.request = httpx.Request(request.method, request.url, headers=headers, extensions=request.extensions,
                                         **body)

    @staticmethod
    def _run_after(call) -> None:
        if call is None:
            return
        try:
            call[0](*call[1])
        except Exception as exc:  # bookkeeping never fails a request
            log.warning('egress: credential bookkeeping failed (%s)', type(exc).__name__)

    # clients --------------------------------------------------------------------------------------------------
    def _async_client(self, plan: _Plan) -> httpx.AsyncClient:
        with self._lock:
            client = self._async_clients.get(plan.key)
            if client is None:
                transport = self._transport
                if transport is None:
                    backend = GuardedBackend(mode=self.mode, allow_lan=self._allow_lan, resolver=self._resolver,
                                             inner=self._network_backend)
                    self.backends.append(backend)
                    transport = _GuardedAsyncTransport(backend, plan.max_concurrency)
                client = httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                                           cookies=_no_cookies(), timeout=None)
                self._async_clients[plan.key] = client
            return client

    def _sync_client(self, plan: _Plan) -> httpx.Client:
        with self._lock:
            client = self._sync_clients.get(plan.key)
            if client is None:
                transport = self._sync_transport
                if transport is None:
                    backend = GuardedSyncBackend(mode=self.mode, allow_lan=self._allow_lan, resolver=self._resolver,
                                                 inner=self._sync_network_backend)
                    self.backends.append(backend)
                    transport = _GuardedSyncTransport(backend, plan.max_concurrency)
                client = httpx.Client(transport=transport, trust_env=False, follow_redirects=False,
                                      cookies=_no_cookies(), timeout=None)
                self._sync_clients[plan.key] = client
            return client

    @staticmethod
    def _busy() -> BrokerError:
        return BrokerError('Too many concurrent calls to this upstream', 'concurrency_exceeded', retry_after=1)

    async def _acquire(self, plan: _Plan) -> _Release:
        loop = asyncio.get_running_loop()
        with self._lock:
            entry = self._async_slots.get(plan.key)
            if entry is None or entry[0] is not loop:
                entry = (loop, asyncio.Semaphore(plan.max_concurrency))
                self._async_slots[plan.key] = entry
        semaphore = entry[1]
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=plan.first_byte)
        except asyncio.TimeoutError:
            raise self._busy() from None
        return _Release(semaphore.release)

    def _acquire_sync(self, plan: _Plan) -> _Release:
        with self._lock:
            semaphore = self._sync_slots.get(plan.key)
            if semaphore is None:
                semaphore = self._sync_slots[plan.key] = threading.BoundedSemaphore(plan.max_concurrency)
        if not semaphore.acquire(timeout=plan.first_byte):
            raise self._busy()
        return _Release(semaphore.release)

    # public API -----------------------------------------------------------------------------------------------
    async def send(self, grant: EgressGrant, upstream, profile, method: str, path: str, *, json=None, content=None,
                   data=None, files=None, params=None, headers=None, path_params=None,
                   stream: bool = False, native: NativeOptions | None = None,
                   media_cap: bool = False) -> httpx.Response:
        """One outbound call. stream=True returns an open response (the caller must aclose() it; the concurrency slot
        is held until then); otherwise the body is read (within the size cap) and the response is closed. native: the
        R32 native passthrough options (module docstring). media_cap: a JSON answer that carries media (OpenAI images
        with n base64 images) is capped by broker_max_upstream_media_bytes instead of the JSON cap."""
        request, plan = self._prepare(grant, upstream, profile, method, path, json=json, content=content, data=data,
                                      files=files, params=params, headers=headers, path_params=path_params,
                                      native=native)
        client = self._async_client(plan)
        release = await self._acquire(plan)
        try:
            response = await client.send(request, stream=True)
        except BaseException:
            release()
            self._scrub_request(request)
            raise
        try:
            self._guard_response(response, plan, release, sync=False, native=native, profile=profile,
                                 media_cap=media_cap)
            self._sanitize(response)
            if not stream:
                await response.aread()
                await response.aclose()
            elif self._needs_body(grant, response, profile):
                await response.aread()  # an error body (within the cap); closes the stream and releases the slot
            after = self._after(grant, response, profile)
            if after is not None:
                import anyio
                await anyio.to_thread.run_sync(self._run_after, after)
        except BaseException:  # cancellation included: the slot and the connection are released before re-raising
            await response.aclose()
            self._scrub_request(request)
            raise
        return response

    def send_sync(self, grant: EgressGrant, upstream, profile, method: str, path: str, *, json=None, content=None,
                  data=None, files=None, params=None, headers=None, path_params=None,
                  stream: bool = False) -> httpx.Response:
        """send() for worker threads (CloudInvoker): the same checks over the synchronous client."""
        request, plan = self._prepare(grant, upstream, profile, method, path, json=json, content=content, data=data,
                                      files=files, params=params, headers=headers, path_params=path_params)
        client = self._sync_client(plan)
        release = self._acquire_sync(plan)
        try:
            response = client.send(request, stream=True)
        except BaseException:
            release()
            self._scrub_request(request)
            raise
        try:
            self._guard_response(response, plan, release, sync=True, profile=profile)
            self._sanitize(response)
            if not stream:
                response.read()
                response.close()
            elif self._needs_body(grant, response, profile):
                response.read()
            self._run_after(self._after(grant, response, profile))
        except BaseException:
            response.close()
            self._scrub_request(request)
            raise
        return response

    # WebSocket (R32 native passthrough) -----------------------------------------------------------------------
    async def open_websocket(self, grant: EgressGrant, upstream, profile, path: str, *, params=None, headers=None,
                             subprotocols=(), native: NativeOptions | None = None) -> 'UpstreamWebSocket':
        """Open a WebSocket to the upstream through the same checks as send() (module docstring). Raises
        UpstreamHandshakeRefused for a non-101 answer (status, allowlisted headers and at most 64 KiB of body), an
        httpcore/OSError for transport trouble, BrokerError for refusals before connect."""
        request, plan = self._prepare(grant, upstream, profile, 'GET', path, json=None, content=None, data=None,
                                      files=None, params=params, headers=headers, path_params=None, native=native)
        url = request.url
        if url.scheme not in ('http', 'https'):
            raise EgressRefused('WebSocket upstreams use http or https origins')
        release = await self._acquire(plan)
        backend = GuardedBackend(mode=self.mode, allow_lan=self._allow_lan, resolver=self._resolver,
                                 inner=self._network_backend)
        self.backends.append(backend)
        stream = None
        try:
            port = url.port or (443 if url.scheme == 'https' else 80)
            stream = await backend.connect_tcp(url.host, port, timeout=plan.connect)
            if url.scheme == 'https':
                stream = await stream.start_tls(_ws_ssl_context(), server_hostname=url.host, timeout=plan.connect)
            key = base64.b64encode(os.urandom(16)).decode('ascii')
            host = url.host if url.port is None else f'{url.host}:{url.port}'
            if ':' in url.host and not url.host.startswith('['):
                host = f'[{url.host}]' + ('' if url.port is None else f':{url.port}')
            lines = [f'GET {url.raw_path.decode("ascii")} HTTP/1.1', f'Host: {host}', 'Upgrade: websocket',
                     'Connection: Upgrade', f'Sec-WebSocket-Key: {key}', 'Sec-WebSocket-Version: 13']
            offered = [p for p in subprotocols if _TOKEN.fullmatch(p)]
            if offered:
                lines.append('Sec-WebSocket-Protocol: ' + ', '.join(offered))
            for name, value in request.headers.multi_items():
                if name.lower() in _WS_REQUEST_SKIP:
                    continue
                lines.append(f'{name}: {value}')
            payload = ('\r\n'.join(lines) + '\r\n\r\n').encode('latin-1')
            del lines
            await stream.write(payload, timeout=plan.first_byte)
            del payload
            status, raw_headers, rest = await _read_head(stream, plan.first_byte)
            if status != 101:
                body = rest[:_HANDSHAKE_BODY_CAP]
                length = raw_headers.get('content-length')
                want = min(int(length), _HANDSHAKE_BODY_CAP) if length and length.isdigit() else _HANDSHAKE_BODY_CAP
                while len(body) < want:
                    try:
                        chunk = await stream.read(65536, timeout=min(plan.first_byte, 5.0))
                    except (httpcore.ReadTimeout, httpcore.ReadError, OSError):
                        break
                    if not chunk:
                        break
                    body += chunk
                body = body[:want]
                fake = httpx.Response(status, headers=list(raw_headers.items()), content=body,
                                      request=httpx.Request('GET', str(url)))
                self._run_after(self._after(grant, fake, profile))
                raise UpstreamHandshakeRefused(status, filter_response_headers(raw_headers.items()), body)
            accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode('ascii')).digest()).decode('ascii')
            if raw_headers.get('upgrade', '').lower() != 'websocket' or raw_headers.get('sec-websocket-accept') != accept:
                raise _bad_response('The upstream answered the WebSocket handshake without a valid upgrade')
            chosen = raw_headers.get('sec-websocket-protocol')
            if chosen is not None and chosen not in offered:
                raise _bad_response('The upstream chose a WebSocket subprotocol that was not offered')
            touch, due = getattr(self.vault, 'touch', None), getattr(self.vault, 'touch_due', None)
            handle = grant.credential
            if handle is not None and handle.credential_id and callable(touch) and callable(due) \
                    and due(handle.credential_id):
                self._run_after((touch, (handle.credential_id,)))
            kept = filter_response_headers(raw_headers.items(),
                                           extra=native.response_headers if native is not None else frozenset())
            return UpstreamWebSocket(stream, _Release(release), rest, chosen, kept,
                                     int(self.settings.broker_max_upstream_media_bytes))
        except BaseException:
            release()
            self._scrub_request(request)
            if stream is not None:
                try:
                    await stream.aclose()
                except BaseException:
                    pass
            raise

    async def aclose(self) -> None:
        """Close every pooled connection (lifespan shutdown). Later calls build fresh clients."""
        with self._lock:
            clients, self._async_clients = list(self._async_clients.values()), {}
        for client in clients:
            await client.aclose()
        self.close()

    def close(self) -> None:
        with self._lock:
            clients, self._sync_clients = list(self._sync_clients.values()), {}
        for client in clients:
            client.close()


# ---------------------------------------------------------------------------------------------------------------------
# WebSocket client (RFC 6455): just enough for the native relay

_WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}")
_WS_REQUEST_SKIP = frozenset({'host', 'connection', 'upgrade', 'content-length', 'transfer-encoding',
                              'sec-websocket-key', 'sec-websocket-version', 'sec-websocket-protocol',
                              'sec-websocket-extensions'})
_HEAD_CAP = 65536
_HANDSHAKE_BODY_CAP = 65536
OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class UpstreamHandshakeRefused(Exception):
    """The upstream answered the WebSocket opening handshake with something other than 101."""

    def __init__(self, status: int, headers: dict, body: bytes):
        super().__init__(f'upstream answered {status}')
        self.status = status
        self.headers = headers
        self.body = body


class WebSocketProtocolError(Exception):
    """The upstream broke RFC 6455 (masked server frame, bad opcode, oversized message, invalid UTF-8)."""


@functools.lru_cache(maxsize=1)
def _ws_ssl_context():
    """A verifying TLS context (certifi roots, no SSL_CERT_* environment) that offers only http/1.1 via ALPN."""
    context = httpx.create_ssl_context(trust_env=False)
    try:
        context.set_alpn_protocols(['http/1.1'])
    except (NotImplementedError, AttributeError):
        pass
    return context


async def _read_head(stream, timeout: float) -> tuple[int, dict, bytes]:
    """(status, lower-case header map, bytes after the head) of an HTTP/1.1 response head (at most 64 KiB)."""
    buffer = b''
    while b'\r\n\r\n' not in buffer:
        if len(buffer) > _HEAD_CAP:
            raise _bad_response('The upstream WebSocket handshake answer is too large')
        chunk = await stream.read(65536, timeout=timeout)
        if not chunk:
            raise _bad_response('The upstream closed the connection during the WebSocket handshake')
        buffer += chunk
    head, _, rest = buffer.partition(b'\r\n\r\n')
    lines = head.decode('latin-1').split('\r\n')
    parts = lines[0].split(' ', 2)
    if len(parts) < 2 or not parts[0].startswith('HTTP/1.') or not parts[1].isdigit():
        raise _bad_response('The upstream WebSocket handshake answer is not HTTP/1.1')
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, sep, value = line.partition(':')
        if not sep:
            continue
        name = name.strip().lower()
        value = value.strip()
        headers[name] = headers[name] + ', ' + value if name in headers else value
    return int(parts[1]), headers, rest


class UpstreamWebSocket:
    """One upstream WebSocket connection (client side: frames masked). recv() returns ('text', str), ('bytes', bytes)
    or None once the upstream closed (close_code / close_reason then hold its close frame); pings are answered."""

    def __init__(self, stream, release: '_Release', buffered: bytes, subprotocol: str | None, headers: dict,
                 max_message: int):
        self._stream = stream
        self._release = release
        self._buffer = bytearray(buffered)
        self.subprotocol = subprotocol
        self.headers = headers
        self.max_message = int(max_message)
        self.close_code: int | None = None
        self.close_reason: str = ''
        self._closed = False
        self._close_sent = False
        self._write_lock = asyncio.Lock()
        self.bytes_in = 0
        self.bytes_out = 0

    async def _read_exact(self, count: int) -> bytes:
        while len(self._buffer) < count:
            chunk = await self._stream.read(65536, timeout=None)
            if not chunk:
                raise ConnectionError('upstream WebSocket connection closed')
            self._buffer.extend(chunk)
        data = bytes(self._buffer[:count])
        del self._buffer[:count]
        return data

    async def _frame(self) -> tuple[bool, int, bytes]:
        first, second = await self._read_exact(2)
        fin, opcode = bool(first & 0x80), first & 0x0F
        if first & 0x70:
            raise WebSocketProtocolError('reserved bits set (no extension was negotiated)')
        if second & 0x80:
            raise WebSocketProtocolError('a server frame was masked')
        length = second & 0x7F
        if length == 126:
            length = int.from_bytes(await self._read_exact(2), 'big')
        elif length == 127:
            length = int.from_bytes(await self._read_exact(8), 'big')
        if opcode >= 0x8 and (length > 125 or not fin):
            raise WebSocketProtocolError('an invalid control frame')
        if length > self.max_message:
            raise WebSocketProtocolError('a frame exceeds the message limit')
        payload = await self._read_exact(length) if length else b''
        self.bytes_in += length
        return fin, opcode, payload

    async def _send(self, opcode: int, payload: bytes) -> None:
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += length.to_bytes(2, 'big')
        else:
            header.append(0x80 | 127)
            header += length.to_bytes(8, 'big')
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload)) if length < 4096 else \
            (int.from_bytes(payload, 'big') ^ int.from_bytes((mask * (length // 4 + 1))[:length], 'big')).to_bytes(
                length, 'big')
        async with self._write_lock:
            await self._stream.write(bytes(header) + masked, timeout=None)
        self.bytes_out += length

    async def recv(self):
        parts: list[bytes] = []
        kind = None
        size = 0
        while True:
            if self._closed:
                return None
            try:
                fin, opcode, payload = await self._frame()
            except (ConnectionError, httpcore.ReadError, OSError):
                self._closed = True
                self.close_code = self.close_code or 1006
                return None
            if opcode == OP_PING:
                try:
                    await self._send(OP_PONG, payload)
                except (httpcore.WriteError, OSError):
                    pass
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                code = int.from_bytes(payload[:2], 'big') if len(payload) >= 2 else 1005
                self.close_code = code
                self.close_reason = payload[2:].decode('utf-8', 'replace')[:123]
                if not self._close_sent:
                    try:
                        self._close_sent = True
                        await self._send(OP_CLOSE, payload[:2])
                    except (httpcore.WriteError, OSError):
                        pass
                self._closed = True
                return None
            if opcode in (OP_TEXT, OP_BINARY):
                if kind is not None:
                    raise WebSocketProtocolError('a new message started inside a fragmented one')
                kind = 'text' if opcode == OP_TEXT else 'bytes'
            elif opcode == OP_CONT:
                if kind is None:
                    raise WebSocketProtocolError('a continuation frame without a message')
            else:
                raise WebSocketProtocolError(f'unknown opcode {opcode}')
            size += len(payload)
            if size > self.max_message:
                raise WebSocketProtocolError('a message exceeds the message limit')
            parts.append(payload)
            if fin:
                data = b''.join(parts)
                if kind == 'text':
                    try:
                        return 'text', data.decode('utf-8')
                    except UnicodeDecodeError:
                        raise WebSocketProtocolError('a text message is not valid UTF-8') from None
                return 'bytes', data

    async def send_text(self, text: str) -> None:
        await self._send(OP_TEXT, text.encode('utf-8'))

    async def send_bytes(self, data: bytes) -> None:
        await self._send(OP_BINARY, bytes(data))

    async def close(self, code: int = 1000, reason: str = '') -> None:
        """Send a close frame once (when the connection still works), then release the socket and the slot."""
        try:
            if not self._close_sent and not self._closed:
                self._close_sent = True
                if code in (1005, 1006, 1015) or not (1000 <= code < 5000):
                    code = 1000
                payload = code.to_bytes(2, 'big') + reason.encode('utf-8')[:123]
                try:
                    await asyncio.wait_for(self._send(OP_CLOSE, payload), timeout=5.0)
                except (asyncio.TimeoutError, httpcore.WriteError, OSError):
                    pass
        finally:
            self._closed = True
            try:
                await self._stream.aclose()
            except BaseException:
                pass
            self._release()


__all__ = ['AUTH_TEMPLATES', 'BLOCKED_NETWORKS', 'EGRESS_MODES', 'EgressClient', 'EgressRefused', 'GuardedBackend',
           'GuardedSyncBackend', 'HOP_HEADER', 'LAN_NETWORKS', 'NativeOptions', 'UpstreamHandshakeRefused',
           'UpstreamWebSocket', 'WebSocketProtocolError', 'check_address']
