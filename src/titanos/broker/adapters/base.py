"""Provider adapter contract.

Adapters never read broker.json, environment variables or the vault. Each call receives a sealed EgressGrant (issued by
router.GrantIssuer) and the EgressClient, the only code that attaches outbound credentials. Inbound requests reach an
adapter as a RawRequest whose headers were filtered to profiles.INBOUND_HEADER_ALLOW; upstream responses leave it as an
UpstreamResult whose headers were filtered to RESPONSE_HEADER_ALLOW (plus the rate-limit families).
"""
from __future__ import annotations

import abc
import asyncio
import fnmatch
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..profiles import INBOUND_HEADER_ALLOW, Profile
from ..surfaces import Surface
from ..translate.ir import IREvent, IRRequest, IRResponse
from ..types import EgressGrant, UpstreamError, Usage

RESPONSE_HEADER_ALLOW = frozenset({'content-type', 'retry-after', 'request-id', 'x-request-id', 'openai-processing-ms',
                                   'character-cost'})
RESPONSE_HEADER_ALLOW_PATTERNS = ('x-ratelimit-*', 'anthropic-ratelimit-*')
# Never forwarded upstream, whatever the allowlist says.
INBOUND_HEADER_NEVER = frozenset({'authorization', 'x-api-key', 'xi-api-key', 'x-goog-api-key', 'api-key', 'cookie',
                                  'proxy-authorization'})


def response_header_allowed(name: str) -> bool:
    name = name.lower()
    return name in RESPONSE_HEADER_ALLOW or any(fnmatch.fnmatchcase(name, p) for p in RESPONSE_HEADER_ALLOW_PATTERNS)


# Never returned to a client, whatever a native row's response_headers say.
RESPONSE_HEADER_NEVER = frozenset({'set-cookie', 'set-cookie2', 'location', 'connection', 'keep-alive', 'transfer-encoding',
                                   'te', 'trailer', 'upgrade', 'proxy-authenticate', 'www-authenticate',
                                   'content-length', 'content-encoding'})


def filter_response_headers(headers, extra: frozenset = frozenset()) -> dict[str, str]:
    """Upstream response headers reduced to the allowlist (lower-case names) plus extra (a native row's
    response_headers); set-cookie, location and hop-by-hop headers never pass. A kept value that reflects a registered
    secret (a credential an upstream echoes in a request-id or rate-limit header) has that value replaced by the
    redaction marker (redact.REGISTRY, exact values only)."""
    from ..redact import REGISTRY
    items = headers.items() if isinstance(headers, Mapping) or hasattr(headers, 'items') else headers
    return {str(k).lower(): (REGISTRY.scrub(v) if isinstance(v, str) else v) for k, v in items
            if response_header_allowed(str(k)) or (str(k).lower() in extra
                                                   and str(k).lower() not in RESPONSE_HEADER_NEVER)}


def profile_inbound_headers(profile) -> frozenset[str]:
    """A profile row's extra forwarded inbound header names (extras inbound_headers: a router's session headers)."""
    extras = getattr(profile, 'extras', None) or {}
    return frozenset(name.lower() for name in extras.get('inbound_headers', ()))


def profile_response_headers(profile) -> frozenset[str]:
    """A profile row's extra returned upstream response header names (extras response_headers)."""
    extras = getattr(profile, 'extras', None) or {}
    return frozenset(name.lower() for name in extras.get('response_headers', ()))


def filter_inbound_headers(headers, extra: frozenset = frozenset()) -> dict[str, str]:
    """Inbound request headers forwarded on passthrough: profiles.INBOUND_HEADER_ALLOW plus extra (a profile row's
    inbound_headers), never credentials, cookies or x-titanos-* hints."""
    items = headers.items() if isinstance(headers, Mapping) or hasattr(headers, 'items') else headers
    out = {}
    for name, value in items:
        if isinstance(name, bytes):
            name = name.decode('latin-1')
        if isinstance(value, bytes):
            value = value.decode('latin-1')
        lowered = name.lower()
        if (lowered in INBOUND_HEADER_ALLOW or lowered in extra) and lowered not in INBOUND_HEADER_NEVER \
                and not lowered.startswith('x-titanos-'):
            out[lowered] = value
    return out


@dataclass
class RawRequest:
    method: str
    path_params: dict = field(default_factory=dict)
    query: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)  # filter_inbound_headers() output only
    json: dict | None = None
    raw: bytes | None = None
    form: list | None = None
    header_allow: frozenset = frozenset()  # the profile's extra inbound header names (profile_inbound_headers)

    def __post_init__(self):
        allowed = INBOUND_HEADER_ALLOW | self.header_allow
        bad = [k for k in self.headers if k.lower() not in allowed or k.lower() in INBOUND_HEADER_NEVER
               or k.lower().startswith('x-titanos-')]
        if bad:
            raise ValueError('RawRequest headers must pass filter_inbound_headers: ' + ', '.join(sorted(bad)))


@dataclass
class UpstreamResult:
    status: int
    headers: dict  # filter_response_headers() output only
    content_type: str
    body: AsyncIterator[bytes]
    upstream_request_id: str | None
    usage: 'asyncio.Future[Usage | None]'
    aclose: Callable[[], Any]


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    status: int | None
    latency_ms: float | None
    code: str | None = None


class ProviderAdapter(abc.ABC):
    """One wire family. Every method gets its authority from the grant; none may read configuration or secrets."""
    native_protocol: str = ''

    @abc.abstractmethod
    def supports(self, profile: Profile, surface: Surface) -> bool:
        """Whether this adapter can serve surface (natively or translated) for profile."""

    @abc.abstractmethod
    async def passthrough(self, grant: EgressGrant, egress, surface: Surface, raw: RawRequest) -> UpstreamResult:
        """Same-protocol call: model rewrite, credential swap, bytes through."""

    @abc.abstractmethod
    async def invoke(self, grant: EgressGrant, egress, ir: IRRequest) -> IRResponse | AsyncIterator[IREvent]:
        """Translated call: an IRResponse, or an IREvent iterator when ir.stream."""

    @abc.abstractmethod
    def invoke_sync(self, grant: EgressGrant, egress, ir: IRRequest) -> IRResponse:
        """Non-streaming translated call on a worker thread (jobs, API Plus stages)."""

    @abc.abstractmethod
    async def list_models(self, grant: EgressGrant, egress) -> list[str]:
        """Live catalog (purpose catalog)."""

    @abc.abstractmethod
    async def probe(self, grant: EgressGrant, egress) -> ProbeResult:
        """Health probe (purpose probe)."""

    @abc.abstractmethod
    def map_error(self, status: int, headers, body: bytes) -> UpstreamError:
        """Classify an upstream error response; the message must already be scrubbed."""

    @abc.abstractmethod
    def usage_from_body(self, surface: Surface, body: dict, headers) -> Usage | None:
        """Usage from a complete non-streaming response body (and usage headers)."""


__all__ = ['INBOUND_HEADER_NEVER', 'ProbeResult', 'ProviderAdapter', 'RESPONSE_HEADER_ALLOW',
           'RESPONSE_HEADER_ALLOW_PATTERNS', 'RESPONSE_HEADER_NEVER', 'RawRequest', 'UpstreamResult', 'filter_inbound_headers',
           'filter_response_headers', 'profile_inbound_headers', 'profile_response_headers', 'response_header_allowed']
