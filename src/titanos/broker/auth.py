"""Inbound authentication (used by server.Guard) and the KeylessHostGuard middleware.

Principals: bootstrap (TITANOS_API_KEY, scope admin), key and parent (issued tk_ keys; parent holds delegate) and
anonymous_loopback (only while nothing is enforced).

- enforced = bool(bootstrap) or keystore.has_records(). Any record counts, expired and revoked included, so expiry or
  revocation never reopens keyless loopback admin. There is no override; only the offline `titanos keys reset --keyless`
  returns to keyless mode.
- Keyless mode ignores presented credentials (stock SDKs send placeholder keys), exactly as before the broker. A
  non-loopback bind with nothing enforced fails closed: 401 on every gated route.
- Keyed mode: Authorization: Bearer (scheme matched case-insensitively, whitespace trimmed, RFC 7235); x-api-key only on
  /v1/messages* and /v1/models* (the Anthropic SDK's model list); xi-api-key only on /v1/text-to-speech*, /v1/speech-to-text and /v1/voices*; x-goog-api-key only on
  /v1beta/*. Vendor headers outside their family are not credentials. Two different presented values give 400
  ambiguous_credentials. A failed claim gives 401 even from loopback.
- A ?key= query credential on /v1beta gives 401 query_credential_refused in every mode (credentials never travel in URLs).
- Failures are throttled per remote address (broker_auth_failures_per_minute, then 429 auth_throttled with Retry-After)
  and audited as auth.failure with the key id prefix when parseable, never the token. Only a presented credential that
  fails (wrong, malformed, ambiguous, expired, revoked or in the URL query) counts toward the throttle; a request that
  presents no credential at all is answered 401 without counting, so credential-less local probes cannot lock out the
  holder of a valid key.
- Keyless mode refuses key and credential writes (non-GET routes needing keys:admin or credentials:write) with 403
  keyless_mode_refused rather than scope_denied, so the route answers with the errors-table code (decision 7).
- bind_loopback defaults to role.loopback_host(settings.host), so omitting it can never open keyless access on a
  network bind.
- R29 keyless Studio provider setup (standalone role, loopback bind, nothing enforced): a request carrying
  `X-Titanos-Studio: 1` (a custom header forces a CORS preflight, which Titanos never approves, so a cross-origin page
  cannot send it) and, for a request with a body, Content-Type application/json, is the principal 'keyless-studio': the
  keyless loopback scopes plus credentials:write, so the credential, connect, upstream, policy and alias writes of
  Studio's providers panel are accepted and audited as keyless-studio (KeylessHostGuard and the keyless cross-site
  refusal still apply first). Minting keys stays CLI or keyed only; broker and managed roles never elevate.
- Native passthrough (R32): a keyless loopback principal without the native scope (anonymous_cloud off) is not refused
  here; BrokerDispatch then serves the Titanos route of that path as before, or refuses the native call itself. Under
  /native every vendor key header is a credential too, and on the bare native paths the profile's own SDK header
  (native.vendor_families). A request that presents no credential but carries a resumable-upload capability
  (?titanos_upload=..., minted when Titanos rewrote the provider's upload URL: native.UploadSessions) is the principal
  that started that upload, for that one session's path and upload_id only (POST or PUT under /native/); a key
  revoked or expired since then is refused as usual.
- tk_ tokens verify against the KeyStore; any other token is compared with the bootstrap key as SHA-256 digests through
  hmac.compare_digest, so the comparison is constant-length.
- The principal is written into the same scope dict (scope['titanos.principal']), so the mounted /mcp sees it. A
  pre-set principal is trusted only on scopes marked titanos.internal (in-process MCP dispatch, which clients cannot
  forge); on any other scope it is discarded.
- The route scope (scopes.required_scope) is enforced here, before the body is read: 403 scope_denied.
- Peer and scheme come from the raw ASGI scope. serve pins uvicorn proxy_headers to behind_tls_proxy and
  forwarded_allow_ips to 127.0.0.1, so no X-Forwarded-* header can change them without the operator's attest.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import parse_qsl

from starlette.middleware.trustedhost import TrustedHostMiddleware

from .envelopes import protocol_for, render
from .errors import BrokerError, error_headers
from .keys import KeyStore, parse_token
from .redact import MIN_SECRET_LENGTH, REGISTRY
from .role import loopback_host, public_routes
from .scopes import BOOTSTRAP_SCOPES, allows, anonymous_scopes, required_scope
from .types import AuthFailure, Principal

LOOPBACK_HOSTS = ('localhost', '127.0.0.1', '[::1]')
# Route scopes whose non-GET routes are key or credential writes: refused in keyless mode with keyless_mode_refused.
KEYLESS_WRITE_SCOPES = frozenset({'keys:admin', 'credentials:write'})
_SAFE_METHODS = frozenset({'GET', 'HEAD', 'OPTIONS'})
INVALID_KEY_MESSAGE = 'Invalid API key'  # the existing Guard message (server.py)
THROTTLE_WINDOW = 60.0
THROTTLE_MAX_ADDRESSES = 4096

_MALFORMED = object()  # a presented credential that cannot be a token (non-Bearer scheme, empty value)


def _family(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == p or path.startswith(p + '/') for p in prefixes)


# Vendor header -> the route families that accept it (the typed families, plus the native passthrough families: /native
# for every vendor header and the bare native paths of each profile's allowlist).
_TYPED_VENDOR_HEADERS: dict[bytes, tuple[str, ...]] = {
    b'x-api-key': ('/v1/messages', '/v1/models'),  # /v1/models: the Anthropic SDK's models.list / retrieve
    b'xi-api-key': ('/v1/text-to-speech', '/v1/speech-to-text', '/v1/voices'),
    b'x-goog-api-key': ('/v1beta',),
}


def _vendor_headers() -> dict[bytes, tuple[str, ...]]:
    from .native import vendor_families
    out = {name: list(prefixes) for name, prefixes in _TYPED_VENDOR_HEADERS.items()}
    for name, prefixes in vendor_families().items():
        bucket = out.setdefault(name, [])
        for prefix in prefixes:
            if not any(prefix == p or prefix.startswith(p + '/') for p in bucket):
                bucket.append(prefix)
    return {name: tuple(prefixes) for name, prefixes in out.items()}


VENDOR_HEADERS: dict[bytes, tuple[str, ...]] = _vendor_headers()
STUDIO_HEADER = b'x-titanos-studio'
STUDIO_PRINCIPAL_ID = 'keyless-studio'
_BODY_METHODS = frozenset({'POST', 'PUT', 'PATCH'})


@dataclass(frozen=True)
class AuthDenial(AuthFailure):
    """An AuthFailure that also carries Retry-After and the error origin, and can render itself."""
    retry_after: float | None = None
    origin: str = 'titanos'

    def headers(self) -> dict[str, str]:
        return error_headers(self.code, origin=self.origin, retry_after=self.retry_after)

    def response(self):
        """A JSONResponse in the inbound protocol's envelope with the broker error headers. The Gemini envelope has no
        code field of its own, so it carries the Titanos code as a google.rpc.ErrorInfo detail (reason in Google's
        UPPER_SNAKE style, metadata.code the Titanos code), as Google's own key errors carry API_KEY_INVALID."""
        response = render(self.protocol, self.status, self.code, self.message, retry_after=self.retry_after,
                          origin=self.origin)
        if self.protocol != 'gemini':
            return response
        import json
        from starlette.responses import JSONResponse
        document = json.loads(response.body)
        document['error']['details'] = [{'@type': 'type.googleapis.com/google.rpc.ErrorInfo',
                                         'reason': self.code.upper(), 'domain': 'titanos',
                                         'metadata': {'code': self.code}}]
        return JSONResponse(document, status_code=self.status, headers=self.headers())

    async def __call__(self, scope, receive, send):
        await self.response()(scope, receive, send)


class FailureThrottle:
    """Sliding one-minute failure window per remote address. limit <= 0 disables throttling. Bounded memory: the least
    recently failing addresses are forgotten beyond max_addresses."""

    def __init__(self, limit: int, *, window: float = THROTTLE_WINDOW, clock: Callable[[], float] = time.monotonic,
                 max_addresses: int = THROTTLE_MAX_ADDRESSES):
        self.limit, self.window, self.clock, self.max_addresses = int(limit), float(window), clock, max_addresses
        self._failures: OrderedDict[str, deque] = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self, address: str, now: float) -> deque | None:
        times = self._failures.get(address)
        if times is None:
            return None
        while times and now - times[0] >= self.window:
            times.popleft()
        if not times:
            del self._failures[address]
            return None
        return times

    def blocked(self, address: str) -> float | None:
        """Seconds until the address may try again, or None when it is not throttled."""
        if self.limit <= 0:
            return None
        with self._lock:
            now = self.clock()
            times = self._prune(address, now)
            if times is None or len(times) < self.limit:
                return None
            return max(0.0, self.window - (now - times[-self.limit]))

    def failure(self, address: str) -> None:
        if self.limit <= 0:
            return
        with self._lock:
            now = self.clock()
            times = self._prune(address, now) or deque()
            times.append(now)
            while len(times) > self.limit:
                times.popleft()
            self._failures[address] = times
            self._failures.move_to_end(address)
            while len(self._failures) > self.max_addresses:
                self._failures.popitem(last=False)


def _peer(scope) -> str:
    client = scope.get('client')
    if client and client[0]:
        return str(client[0])
    return 'unknown'


def _is_loopback_address(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.strip('[]').split('%', 1)[0]).is_loopback
    except ValueError:
        return False


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode('utf-8')).digest()


class Authenticator:
    """Implements types.AuthenticatorProtocol. role is the instance role (standalone|broker|managed); anonymous_cloud
    reads broker.json policy.anonymous_cloud on demand; bind_loopback is whether serve bound a loopback host (None, the
    default, derives it from settings.host, so a caller that omits it on a 0.0.0.0 bind still fails closed). The
    bootstrap key is registered with redact.REGISTRY for exact-value scrubbing."""

    def __init__(self, *, bootstrap_key: str | None, keystore: KeyStore, settings, audit=None, role: str = 'standalone',
                 bind_loopback: bool | None = None, anonymous_cloud: Callable[[], bool] = lambda: False,
                 clock: Callable[[], float] = time.monotonic):
        self._bootstrap = bootstrap_key or None
        if self._bootstrap and len(self._bootstrap) >= MIN_SECRET_LENGTH:
            REGISTRY.add(self._bootstrap)
        # The digest of a random value when no bootstrap key exists, so the compare still runs (and never matches).
        self._bootstrap_digest = _digest(self._bootstrap if self._bootstrap else os.urandom(32).hex())
        self.keystore = keystore
        self.settings = settings
        self.audit = audit
        self.role = role
        if bind_loopback is None:
            bind_loopback = loopback_host(str(getattr(settings, 'host', '') or ''))
        self.bind_loopback = bool(bind_loopback)
        self.anonymous_cloud = anonymous_cloud
        self.throttle = FailureThrottle(getattr(settings, 'broker_auth_failures_per_minute', 20), clock=clock)
        self._public = public_routes(settings)
        self._throttle_audited: dict[str, float] = {}
        self.failures = 0  # metrics counter

    # -- state ---------------------------------------------------------------------------------------------------------
    @property
    def enforced(self) -> bool:
        return bool(self._bootstrap) or self.keystore.has_records()

    def transport_secure(self, scope) -> bool:
        """TLS (scope scheme https/wss) or a loopback peer, as uvicorn reports them with proxy headers pinned."""
        if scope.get('scheme') in ('https', 'wss'):
            return True
        client = scope.get('client')
        return bool(client) and _is_loopback_address(str(client[0]))

    def bootstrap_principal(self) -> Principal:
        return Principal(id='bootstrap', kind='bootstrap', scopes=BOOTSTRAP_SCOPES, name='bootstrap')

    def anonymous_principal(self) -> Principal:
        try:
            cloud = bool(self.anonymous_cloud())
        except Exception:  # an unreadable policy grants nothing extra
            cloud = False
        return Principal(id='anonymous', kind='anonymous_loopback',
                         scopes=anonymous_scopes(anonymous_cloud=cloud, role=self.role), name='anonymous')

    def studio_request(self, scope) -> bool:
        """R29: whether a keyless request is Studio's provider-setup write (see the module docstring). Only meaningful
        while nothing is enforced on a loopback bind in standalone role."""
        if self.role != 'standalone' or self.enforced or not self.bind_loopback:
            return False
        studio = content_type = None
        count = 0
        for name, value in scope.get('headers') or ():
            name = name.lower() if isinstance(name, bytes) else str(name).lower().encode('latin-1')
            if name == STUDIO_HEADER:
                count += 1
                studio = value.decode('latin-1') if isinstance(value, bytes) else str(value)
            elif name == b'content-type':
                content_type = value.decode('latin-1') if isinstance(value, bytes) else str(value)
        if count != 1 or (studio or '').strip() != '1':
            return False
        if str(scope.get('method', 'GET')).upper() in _BODY_METHODS:
            kind = (content_type or '').split(';', 1)[0].strip().lower()
            if kind != 'application/json':
                return False
        return True

    def studio_principal(self) -> Principal:
        base = self.anonymous_principal()
        return Principal(id=STUDIO_PRINCIPAL_ID, kind='anonymous_loopback',
                         scopes=base.scopes | {'credentials:write'}, name=STUDIO_PRINCIPAL_ID)

    # -- parsing -------------------------------------------------------------------------------------------------------
    def _presented(self, scope, headers, path: str) -> list:
        raw = scope.get('headers')
        if raw is None:
            raw = list((headers or {}).items())
        values = []
        for name, value in raw:
            if isinstance(name, str):
                name = name.encode('latin-1')
            if isinstance(value, bytes):
                value = value.decode('latin-1')
            name = name.lower()
            if name == b'authorization':
                parts = value.strip().split(None, 1)
                if len(parts) == 2 and parts[0].lower() == 'bearer' and parts[1].strip():
                    values.append(parts[1].strip())
                else:
                    values.append(_MALFORMED)
            elif name in VENDOR_HEADERS and _family(path, VENDOR_HEADERS[name]):
                values.append(value.strip() or _MALFORMED)
        return values

    @staticmethod
    def _query_key(scope, path: str) -> bool:
        if not _family(path, ('/v1beta',)):
            return False
        query = scope.get('query_string') or b''
        if isinstance(query, bytes):
            query = query.decode('latin-1')
        return any(name == 'key' for name, _ in parse_qsl(query, keep_blank_values=True))

    # -- decisions -----------------------------------------------------------------------------------------------------
    def _fail(self, scope, status: int, code: str, message: str, protocol: str, *, key_id: str | None = None,
              count: bool = True) -> AuthDenial:
        address = _peer(scope)
        self.failures += 1
        if count:
            self.throttle.failure(address)
        if self.audit is not None:
            detail = {'code': code, 'method': scope.get('method', ''), 'path': scope.get('path', '')}
            if key_id:
                detail['key_id'] = key_id
            try:
                self.audit.event('auth.failure', address, key_id or '-', detail)
            except Exception:  # an audit failure must not turn a 401 into a 500
                pass
        return AuthDenial(status, code, message, protocol)

    def _throttled(self, scope, retry_after: float, protocol: str) -> AuthDenial:
        address = _peer(scope)
        self.failures += 1
        now = time.monotonic()
        last = self._throttle_audited.get(address)
        if self.audit is not None and (last is None or now - last >= THROTTLE_WINDOW):
            self._throttle_audited[address] = now
            if len(self._throttle_audited) > THROTTLE_MAX_ADDRESSES:
                self._throttle_audited.pop(next(iter(self._throttle_audited)))
            try:
                self.audit.event('auth.failure', address, '-', {'code': 'auth_throttled', 'path': scope.get('path', '')})
            except Exception:
                pass
        return AuthDenial(429, 'auth_throttled', 'Too many failed authentication attempts; retry later', protocol,
                          retry_after=max(1.0, retry_after))

    def _authorize(self, scope, principal: Principal, protocol: str) -> Principal | AuthDenial:
        method = 'WS' if scope.get('type') == 'websocket' else str(scope.get('method', 'GET')).upper()
        needed = required_scope(method, scope.get('path', '/'), public_routes=self._public)
        if needed == 'admin' and principal.kind == 'anonymous_loopback':
            # keyless loopback mode keeps today's semantics for routes outside ROUTE_SCOPES (embedder-added routes):
            # the fail-closed admin default binds keys; key and credential writes are listed and still refused
            needed = None
        if needed == 'native' and principal.kind == 'anonymous_loopback' and not allows(principal, 'native'):
            # BrokerDispatch decides: the Titanos route of that shape as before, or a typed native refusal
            needed = None
        if not allows(principal, needed):
            if (principal.kind == 'anonymous_loopback' and method not in _SAFE_METHODS
                    and any(part in KEYLESS_WRITE_SCOPES for part in str(needed or '').split('|'))):
                return AuthDenial(403, 'keyless_mode_refused',
                                  'Keys and credentials cannot be written in keyless mode; use the titanos CLI '
                                  '(titanos keys create, titanos broker credential set)', protocol, origin='policy')
            return AuthDenial(403, 'scope_denied', f'This credential lacks the scope this route requires ({needed})',
                              protocol, origin='policy')
        scope['titanos.principal'] = principal
        return principal

    def _upload_principal(self, scope, path: str) -> Principal | None:
        """The principal bound to a resumable-upload capability on this exact session (see the module docstring)."""
        from .native import NATIVE_PREFIX, UPLOADS, split_upload_token
        if not path.startswith(NATIVE_PREFIX) or scope.get('type') != 'http':
            return None
        token, upload_id, _ = split_upload_token(scope.get('query_string') or b'')
        if not token:
            return None
        principal = UPLOADS.claim(token, str(scope.get('method', '')), path, upload_id)
        if principal is None:
            return None
        if principal.kind in ('key', 'parent') and principal.key_id:
            try:
                record = self.keystore.get(principal.key_id)
            except Exception:
                record = None
            if record is None or not record.active():
                return None
            return self.keystore.to_principal(record)
        return principal

    def _verify(self, scope, token, protocol: str) -> Principal | AuthDenial:
        if token is _MALFORMED:
            hmac.compare_digest(self._bootstrap_digest, _digest(''))
            return self._fail(scope, 401, 'authentication_error', INVALID_KEY_MESSAGE, protocol)
        if hmac.compare_digest(self._bootstrap_digest, _digest(token)) and self._bootstrap:
            return self.bootstrap_principal()
        key_id = parse_token(token)
        try:
            record = self.keystore.verify(token)
        except BrokerError as exc:  # key_expired / key_revoked for a matching token
            return self._fail(scope, exc.status, exc.code, str(exc), protocol, key_id=key_id)
        if record is None:
            return self._fail(scope, 401, 'authentication_error', INVALID_KEY_MESSAGE, protocol, key_id=key_id)
        return self.keystore.to_principal(record)

    def authenticate(self, scope, headers=None) -> Principal | AuthFailure:
        """The principal for an http/websocket scope (also written to scope['titanos.principal']), or an AuthDenial
        (an AuthFailure subclass with retry_after and response()). headers is the Guard's dict of the raw headers; the
        raw scope list is preferred so repeated headers are seen."""
        path = scope.get('path', '/') or '/'
        protocol = protocol_for(path)
        if scope.get('titanos.internal'):
            preset = scope.get('titanos.principal')
            if isinstance(preset, Principal):
                return self._authorize(scope, preset, protocol)
        else:
            scope.pop('titanos.principal', None)  # only in-process dispatch may carry a principal in
        if self._query_key(scope, path):
            # worded so redact.scrub (which redacts whatever follows a credential header name) leaves it intact
            return self._fail(scope, 401, 'query_credential_refused',
                              'Credentials are not accepted in the URL query; send the key in a request header '
                              '(x-goog-api-key, or Authorization with the Bearer scheme)', protocol)
        if not self.enforced:
            if not self.bind_loopback:
                return self._fail(scope, 401, 'authentication_error',
                                  'Authentication is required on a network interface; no key is configured', protocol,
                                  count=False)
            if self.studio_request(scope):
                return self._authorize(scope, self.studio_principal(), protocol)
            return self._authorize(scope, self.anonymous_principal(), protocol)
        retry_after = self.throttle.blocked(_peer(scope))
        if retry_after is not None:
            return self._throttled(scope, retry_after, protocol)
        presented = self._presented(scope, headers, path)
        distinct = []
        for value in presented:
            if not any(value is seen or (isinstance(value, str) and isinstance(seen, str)
                                         and hmac.compare_digest(_digest(value), _digest(seen))) for seen in distinct):
                distinct.append(value)
        if len(distinct) > 1:
            return self._fail(scope, 400, 'ambiguous_credentials',
                              'More than one different credential was presented; send exactly one', protocol)
        if not distinct:  # nothing presented: 401, but not a failed claim, so it does not feed the throttle
            uploader = self._upload_principal(scope, path)
            if uploader is not None:
                return self._authorize(scope, uploader, protocol)
            return self._fail(scope, 401, 'authentication_error', INVALID_KEY_MESSAGE, protocol, count=False)
        outcome = self._verify(scope, distinct[0], protocol)
        if isinstance(outcome, AuthFailure):
            return outcome
        return self._authorize(scope, outcome, protocol)


class KeylessHostGuard:
    """Replaces the conditional TrustedHostMiddleware: while auth is not enforced, the loopback Host allowlist applies
    with TrustedHostMiddleware's own response (so keyless behaviour is byte-identical); once enforced, every Host passes
    to the bearer check. Re-evaluated per request, so minting the first key (or the offline keyless reset) takes effect
    without a restart."""

    def __init__(self, app, auth):
        self.app = app
        self.auth = auth
        self._trusted = TrustedHostMiddleware(app, allowed_hosts=list(LOOPBACK_HOSTS))

    async def __call__(self, scope, receive, send):
        if scope.get('type') in ('http', 'websocket') and not self.auth.enforced:
            return await self._trusted(scope, receive, send)
        return await self.app(scope, receive, send)


__all__ = ['AuthDenial', 'Authenticator', 'FailureThrottle', 'INVALID_KEY_MESSAGE', 'KeylessHostGuard', 'LOOPBACK_HOSTS',
           'STUDIO_HEADER', 'STUDIO_PRINCIPAL_ID', 'VENDOR_HEADERS']
