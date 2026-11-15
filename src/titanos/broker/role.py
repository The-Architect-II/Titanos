"""Role switch table, parent-orchestrator header parsing, serve preconditions and port policy.

The role (standalone | broker | managed) is fixed at serve start. In every role Titanos never discovers peers, fans out,
aggregates other hosts or load-balances other servers; the switches below only turn this instance's own egress,
fallback, delegation, advisors and probes on or off.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import time
import uuid
from dataclasses import dataclass

from .errors import BrokerError
from .naming import REF_MAX
from .scopes import BASE_PUBLIC_ROUTES, MANAGED_PUBLIC_ROUTES, allows
from .types import LOCATIONS, GrantPurpose, Hints, KeyStoreSummary

log = logging.getLogger('titanos.broker')

# Reserved ports: Ollama, LM Studio and vLLM defaults plus 11435.
RESERVED_PORTS = frozenset({11434, 11435, 1234, 1235, 8000, 8001})
LAN_BIND_MESSAGE = 'Set TITANOS_API_KEY before binding to a network interface'
_SAFE_ID = re.compile(r'[A-Za-z0-9._:/=+-]{1,128}')
_SAFE_SHORT = re.compile(r'[A-Za-z0-9._:/=+-]{1,64}')
_WARNED: set[str] = set()


@dataclass(frozen=True)
class RoleSwitches:
    cloud_egress: bool
    cloud_fallback: bool
    router_upstreams: bool
    advisors: bool
    health_probes: bool
    live_catalogs: bool
    anonymous_cloud_allowed: bool
    requires_keys: bool
    requires_tls_off_loopback: bool
    vault_sealed: bool
    delegation: bool
    parent_headers_authoritative: bool


def switches_for(settings) -> RoleSwitches:
    """The switch table of the roles spec. cloud_egress and cloud_fallback are still subject to the effective cloud policy
    (and fallback needs cloud_policy=fallback); managed role turns every non-delegated egress purpose off."""
    role = settings.role
    if role == 'managed':
        delegation = settings.managed_delegation == 'parent'
        return RoleSwitches(cloud_egress=False, cloud_fallback=False, router_upstreams=False, advisors=False,
                            health_probes=False, live_catalogs=False, anonymous_cloud_allowed=False, requires_keys=False,
                            requires_tls_off_loopback=False, vault_sealed=not delegation, delegation=delegation,
                            parent_headers_authoritative=True)
    if role not in ('standalone', 'broker'):
        raise ValueError(f'Unknown role: {role!r}')
    broker = role == 'broker'
    return RoleSwitches(cloud_egress=True, cloud_fallback=True, router_upstreams=True, advisors=True, health_probes=True,
                        live_catalogs=True, anonymous_cloud_allowed=not broker, requires_keys=broker,
                        requires_tls_off_loopback=broker, vault_sealed=False, delegation=False,
                        parent_headers_authoritative=False)


def purpose_allowed(switches: RoleSwitches, purpose: GrantPurpose, *, delegated: bool) -> bool:
    """Whether an egress purpose may run at all. inference needs cloud egress, or (managed) delegation plus a delegated
    request; probe, validate and test follow health_probes; catalog live_catalogs; advisor advisors. Delegation never
    enables a non-inference purpose."""
    if purpose == 'inference':
        return switches.cloud_egress or (switches.delegation and delegated)
    if purpose in ('probe', 'validate', 'test'):
        return switches.health_probes
    if purpose == 'catalog':
        return switches.live_catalogs
    if purpose == 'advisor':
        return switches.advisors
    raise ValueError(f'Unknown egress purpose: {purpose!r}')


def public_routes(settings) -> frozenset[str]:
    """The per-app public route set: server.PUBLIC_ROUTES plus /v1/health/ready, plus /api/version in managed role."""
    return BASE_PUBLIC_ROUTES | (MANAGED_PUBLIC_ROUTES if settings.role == 'managed' else frozenset())


def _header(headers: dict, name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        value = headers.get(name.encode('latin-1'))
    if isinstance(value, bytes):
        value = value.decode('latin-1')
    return value.strip() if isinstance(value, str) else None


def _lower(headers) -> dict:
    items = headers.items() if hasattr(headers, 'items') else headers
    out = {}
    for name, value in items:
        if isinstance(name, bytes):
            name = name.decode('latin-1')
        out.setdefault(name.lower(), value)
    return out


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(message)


def parse_parent_headers(headers, settings, principal) -> Hints:
    """Request hints from headers (case-insensitive; str or bytes names). Malformed Titanos hints give 400
    invalid_request; unknown parent formats are logged once and ignored.

    - request_id: X-Request-ID when it is at most 128 safe characters, else a uuid4 hex.
    - location: managed/parent placement local_only (x-titanos-location or x-hivemind-placement) is authoritative;
      otherwise X-Titanos-Location; otherwise a valid parent placement value.
    - target_header: X-Titanos-Target (its use needs route:explicit, checked by the router).
    - delegated: X-Titanos-Delegate: cloud, honoured only in managed role with managed_delegation=parent for an
      authenticated key holding the delegate scope.
    - hop / hop_value: x-titanos-hop or x-hivemind-peer-hop (never forward; 508 when equal to our own hop value).
    - deadline: the first present header of managed_deadline_headers (x-titanos-deadline-ms is relative milliseconds),
      capped by client_timeout, as a time.monotonic() value.
    - attempt: x-titanos-attempt or x-hivemind-attempt (recorded only). trace_id: first of managed_trace_headers."""
    h = _lower(headers)
    request_id = _header(h, 'x-request-id')
    if not request_id or not _SAFE_ID.fullmatch(request_id):
        request_id = uuid.uuid4().hex

    titanos_location = _header(h, 'x-titanos-location')
    if titanos_location is not None and titanos_location not in LOCATIONS:
        raise BrokerError('X-Titanos-Location must be local_only, prefer_local, auto or cloud_only', 'invalid_request')
    placement = _header(h, 'x-hivemind-placement')
    if placement is not None and placement not in LOCATIONS:
        _warn_once('placement', 'x-hivemind-placement value format is not pinned by a fixture; ignored')
        placement = None
    if placement == 'local_only' or titanos_location == 'local_only':
        location = 'local_only'
    else:
        location = titanos_location or placement

    target = _header(h, 'x-titanos-target')
    if target is not None and (not target or len(target) > REF_MAX):
        raise BrokerError('X-Titanos-Target must be a model reference of at most 256 characters', 'invalid_request')

    delegate = _header(h, 'x-titanos-delegate')
    if delegate is not None and delegate != 'cloud':
        raise BrokerError('X-Titanos-Delegate accepts only the value cloud', 'invalid_request')
    delegated = bool(delegate == 'cloud' and settings.role == 'managed' and settings.managed_delegation == 'parent'
                     and principal is not None and principal.kind in ('key', 'parent') and allows(principal, 'delegate'))

    hop_value = _header(h, 'x-titanos-hop') or _header(h, 'x-hivemind-peer-hop')
    hop = hop_value is not None
    if hop_value is not None and not _SAFE_SHORT.fullmatch(hop_value):
        hop_value = None  # still a hop (local-only), but never equal to our own hex hop value

    now = time.monotonic()
    budget = float(settings.client_timeout)
    for name in (part.strip().lower() for part in settings.managed_deadline_headers.split(',')):
        value = _header(h, name)
        if value is None:
            continue
        if name != 'x-titanos-deadline-ms':
            _warn_once('deadline:' + name, f'deadline header {name} has no pinned format; ignored')
            continue
        if not value.isdigit() or not 0 < int(value) <= 86_400_000:
            raise BrokerError('X-Titanos-Deadline-Ms must be a positive integer of milliseconds', 'invalid_request')
        budget = min(budget, int(value) / 1000)
        break
    if _header(h, 'x-hivemind-deadline') is not None:
        _warn_once('x-hivemind-deadline', 'X-HiveMind-Deadline format is not pinned by a fixture; ignored')

    attempt = _header(h, 'x-titanos-attempt') or _header(h, 'x-hivemind-attempt')
    if attempt is not None and not _SAFE_SHORT.fullmatch(attempt):
        attempt = None
    trace_id = None
    for name in (part.strip().lower() for part in settings.managed_trace_headers.split(',')):
        value = _header(h, name)
        if value and _SAFE_ID.fullmatch(value):
            trace_id = value
            break
    return Hints(location=location, target_header=target, delegated=delegated, hop=hop, deadline=now + budget,
                 attempt=attempt, trace_id=trace_id, request_id=request_id, hop_value=hop_value)


def loopback_host(host: str) -> bool:
    try:
        return ipaddress.ip_address(host.strip('[]')).is_loopback
    except ValueError:
        return host == 'localhost'


def check_serve_preconditions(settings, *, bootstrap_key: str | None, keystore: KeyStoreSummary, explicit_port: bool,
                              vault_backend: str, audit_ok: bool) -> list[str]:
    """Replaces cli.py's single LAN-bind check. Returns warnings; raises ValueError naming the switch that refuses.

    - Any role: a non-loopback bind requires enforcement (bootstrap key or an active admin-capable key) and keeps the
      existing message; behind_tls_proxy requires a loopback bind.
    - broker: a bootstrap key; native TLS (tls_certfile+tls_keyfile) off loopback; a writable audit log whose chain
      verifies; a vault posture (an available backend, or refs-only); no active cloud key without budget and expiry;
      a warning when limit_concurrency is below 64.
    - managed: an explicit --port or TITANOS_PORT outside RESERVED_PORTS."""
    warnings: list[str] = []
    loopback = loopback_host(settings.host)
    enforced = bool(bootstrap_key) or keystore.active_admin > 0
    if not loopback and not enforced:
        raise ValueError(LAN_BIND_MESSAGE)
    if settings.behind_tls_proxy and not loopback:
        raise ValueError('TITANOS_BEHIND_TLS_PROXY is valid only with a loopback bind; use TITANOS_TLS_CERTFILE and '
                         'TITANOS_TLS_KEYFILE for a network bind')
    if settings.role == 'broker':
        if not bootstrap_key:
            raise ValueError('TITANOS_ROLE=broker requires TITANOS_API_KEY (a bootstrap key); serve refuses keyless')
        if not loopback and not (settings.tls_certfile and settings.tls_keyfile):
            raise ValueError('TITANOS_ROLE=broker refuses a network bind without native TLS: set TITANOS_TLS_CERTFILE '
                             'and TITANOS_TLS_KEYFILE')
        if not audit_ok:
            raise ValueError('TITANOS_ROLE=broker requires a writable audit log whose chain verifies')
        if vault_backend in ('', 'none', 'unavailable') and settings.broker_vault_backend != 'refs-only':
            raise ValueError('TITANOS_ROLE=broker requires a vault posture: an available OS store, '
                             'TITANOS_BROKER_VAULT_BACKEND=sealed-file with a passphrase file, or refs-only')
        if keystore.cloud_keys_missing_budget:
            raise ValueError('TITANOS_ROLE=broker refuses active keys with a cloud scope but no budget and expiry: '
                             + ', '.join(sorted(keystore.cloud_keys_missing_budget)))
        if settings.limit_concurrency < 64:
            warnings.append('TITANOS_LIMIT_CONCURRENCY below 64 limits concurrent broker streams')
    elif settings.role == 'managed':
        if not explicit_port:
            raise ValueError('TITANOS_ROLE=managed requires an explicit --port or TITANOS_PORT')
        if settings.port in RESERVED_PORTS:
            raise ValueError(f'TITANOS_ROLE=managed refuses reserved port {settings.port} '
                             f'({", ".join(map(str, sorted(RESERVED_PORTS)))})')
    return warnings


__all__ = ['LAN_BIND_MESSAGE', 'RESERVED_PORTS', 'RoleSwitches', 'check_serve_preconditions', 'loopback_host',
           'parse_parent_headers', 'public_routes', 'purpose_allowed', 'switches_for']
