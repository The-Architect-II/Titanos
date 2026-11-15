"""Shared broker types: the frozen records and Protocols every broker package builds against.

Data only; importing this module does no I/O. Decision is the one mutable record: dispatch assembles it across a request
(candidates, attempts, usage) and sinks persist its to_dict() form, which never carries prompts, outputs or secrets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .surfaces import Surface
    from .vault import CredentialHandle

Role = Literal['standalone', 'broker', 'managed']
TargetClass = Literal['local', 'cloud', 'router']
Location = Literal['local_only', 'prefer_local', 'auto', 'cloud_only']
CloudPolicy = Literal['off', 'explicit', 'fallback']
OutcomeKind = Literal['success', 'client_error', 'rate_limited', 'server_error', 'network', 'timeout', 'auth', 'cancelled',
                      'refused']
GrantPurpose = Literal['inference', 'probe', 'catalog', 'advisor', 'validate', 'test']
PrincipalKind = Literal['bootstrap', 'key', 'anonymous_loopback', 'parent']
ExclusionReason = Literal['scope', 'policy', 'role', 'breaker_open', 'capability', 'disabled', 'data_policy', 'location',
                          'unvalidated']

ROLES: tuple[str, ...] = ('standalone', 'broker', 'managed')
TARGET_CLASSES: frozenset[str] = frozenset({'local', 'cloud', 'router'})
LOCATIONS: tuple[str, ...] = ('local_only', 'prefer_local', 'auto', 'cloud_only')
CLOUD_POLICIES: tuple[str, ...] = ('off', 'explicit', 'fallback')  # ordered: off < explicit < fallback
OUTCOME_KINDS: frozenset[str] = frozenset({'success', 'client_error', 'rate_limited', 'server_error', 'network', 'timeout',
                                           'auth', 'cancelled', 'refused'})
GRANT_PURPOSES: tuple[str, ...] = ('inference', 'probe', 'catalog', 'advisor', 'validate', 'test')
PRINCIPAL_KINDS: frozenset[str] = frozenset({'bootstrap', 'key', 'anonymous_loopback', 'parent'})
EXCLUSION_REASONS: frozenset[str] = frozenset({'scope', 'policy', 'role', 'breaker_open', 'capability', 'disabled',
                                               'data_policy', 'location', 'unvalidated'})


def min_cloud_policy(*policies: str) -> str:
    """The most restrictive policy (off < explicit < fallback), e.g. min(ceiling, broker.json cloud_policy)."""
    return min(policies, key=CLOUD_POLICIES.index)


@dataclass(frozen=True)
class KeyConstraints:
    models: tuple[str, ...] = ('*',)
    upstreams: tuple[str, ...] = ('*',)
    target_classes: frozenset[str] = TARGET_CLASSES
    locations: frozenset[str] = frozenset(LOCATIONS)


@dataclass(frozen=True)
class KeyLimits:
    rpm: int | None = None
    tpm: int | None = None
    concurrent: int | None = None


@dataclass(frozen=True)
class KeyBudget:
    usd_per_day: Decimal | None = None
    usd_per_month: Decimal | None = None
    tokens_per_day: int | None = None
    max_request_usd: Decimal | None = None
    hard: bool = True


@dataclass(frozen=True)
class Principal:
    id: str
    kind: PrincipalKind
    scopes: frozenset[str]
    constraints: KeyConstraints = KeyConstraints()
    limits: KeyLimits | None = None
    budget: KeyBudget | None = None
    expires_at: datetime | None = None
    name: str | None = None
    default_location: Location | None = None

    @property
    def key_id(self) -> str | None:
        """The issued key id for key and parent principals; None for bootstrap and anonymous loopback."""
        return self.id if self.kind in ('key', 'parent') else None


@dataclass(frozen=True)
class Target:
    ref: str
    klass: TargetClass
    upstream_id: str | None = None
    upstream_model: str | None = None
    local_name: str | None = None
    capabilities: frozenset[str] = frozenset()
    adapter: str = 'local'
    profile_id: str | None = None


@dataclass(frozen=True)
class Hints:
    location: Location | None = None
    target_header: str | None = None
    delegated: bool = False
    hop: bool = False
    deadline: float | None = None  # time.monotonic() value
    attempt: str | None = None
    trace_id: str | None = None
    request_id: str = ''
    hop_value: str | None = None  # the inbound X-Titanos-Hop value, compared with our own for 508 loop_detected


@dataclass(frozen=True)
class BodySummary:
    stream: bool = False
    approx_input_tokens: int = 0
    max_output_tokens: int | None = None
    has_images: bool = False
    has_tools: bool = False
    has_audio: bool = False
    store: bool | None = None
    previous_response_id: str | None = None
    bytes: int = 0
    audio_seconds: float | None = None  # uploaded audio duration estimate (limits.estimate_audio_seconds)
    # the uploaded audio's measured duration (limits.measure_audio_seconds) when every file states it, else None
    audio_measured: float | None = None


@dataclass(frozen=True)
class RouteContext:
    surface: 'Surface'
    requested_model: str
    principal: Principal
    hints: Hints
    body: BodySummary
    dry_run: bool = False


@dataclass(frozen=True)
class EffectivePolicy:
    cloud_policy: CloudPolicy
    location: Location
    role: Role
    delegated: bool = False


@dataclass(frozen=True)
class Exclusion:
    ref: str
    reason: ExclusionReason
    detail: str | None = None


@dataclass(frozen=True)
class AdvisorOutcome:
    profile: str
    ms: float
    choice: str | None
    fallbacks: tuple[str, ...] = ()
    valid: bool = False
    reason: str | None = None  # advisor_invalid | advisor_timeout | advisor_failed when not valid


@dataclass(frozen=True)
class RoutePlan:
    chain: tuple[Target, ...]
    effective: EffectivePolicy
    exclusions: tuple[Exclusion, ...] = ()
    advisor: AdvisorOutcome | None = None
    max_attempts: int = 1


@dataclass(frozen=True)
class EgressGrant:
    """Authority for exactly one outbound call. Constructed only by router.GrantIssuer (enforced by a source scan)."""
    decision_id: str
    request_id: str
    purpose: GrantPurpose
    delegated: bool
    target: Target | None
    upstream_id: str | None
    origin: str
    credential: 'CredentialHandle | None'
    deadline: float
    reservation: Any = None
    hop_header: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    characters: int | None = None
    audio_seconds: float | None = None
    images: int | None = None
    estimated: bool = False
    # Anthropic cache writes (cache_creation_input_tokens): part of input_tokens, billed at the cache-write price (R18)
    cache_write_tokens: int | None = None
    # Modality token details (OpenAI audio and image token billing): parts of input_tokens / output_tokens that the
    # provider bills at their own rates (PriceRow audio_input_per_mtok, audio_output_per_mtok, image_input_per_mtok;
    # the text price applies when the row has none)
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None
    image_input_tokens: int | None = None
    # Anthropic billing details: the part of cache_write_tokens written with the 1-hour TTL (PriceRow
    # cache_write_1h_per_mtok), server-side web search requests (usage.server_tool_use.web_search_requests, PriceRow
    # per_web_search_request) and fast mode (usage.speed 'fast': token prices times PriceRow fast_multiplier)
    cache_write_1h_tokens: int | None = None
    web_search_requests: int | None = None
    fast: bool | None = None


@dataclass(frozen=True)
class UpstreamError:
    kind: OutcomeKind
    status: int
    provider_code: str | None
    message: str  # already scrubbed
    retry_after: float | None = None
    # False only for httpx.ConnectError, ConnectTimeout, PoolTimeout and pre-connect refusals
    request_bytes_sent: bool = True
    # OpenAI-compatible envelopes: error.param and error.type when they are plain identifiers (kept on passthrough 4xx)
    provider_param: str | None = None
    provider_type: str | None = None


@dataclass(frozen=True)
class Candidate:
    ref: str
    klass: TargetClass
    eligible: bool
    excluded: ExclusionReason | None = None


@dataclass(frozen=True)
class Attempt:
    ref: str
    outcome: str
    status: int | None = None
    error_code: str | None = None
    ms_to_headers: float | None = None
    ms_to_first_byte: float | None = None
    request_bytes_sent: bool = False


@dataclass(frozen=True)
class Served:
    ref: str
    klass: TargetClass
    upstream: str | None = None
    upstream_model: str | None = None


@dataclass(kw_only=True)
class Decision:
    """One routing decision per request, refusals included (data_model (5)). Metadata only. Keyword-only, in the
    data-model field order."""
    ts: str
    request_id: str
    decision_id: str
    trace_id: str | None = None
    key_id: str | None = None
    role: Role
    surface: str
    inbound_protocol: str
    requested_model: str
    dry_run: bool = False
    hints: dict[str, Any] = field(default_factory=dict)  # {location, target_header, delegated, hop}
    effective: dict[str, Any] = field(default_factory=dict)  # {cloud_policy, location}
    candidates: list[Candidate] = field(default_factory=list)
    advisor: AdvisorOutcome | None = None
    chain: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    served: Served | None = None
    translation: str | None = None  # passthrough | local | <in>-><out>
    notes: list[str] = field(default_factory=list)
    fallback_count: int = 0
    committed_at_attempt: int | None = None
    usage: Usage | None = None
    cost_usd: str | None = None
    latency_ms: float | None = None
    # what cost_usd was computed from (pricing.Priced.basis): 'tokens', 'characters', 'images',
    # 'audio_seconds:provider', 'audio_seconds:upload' (joined with '+'), 'reservation' (nothing could be priced: the
    # reserved estimate stands, fail closed), 'unpriced' (no price row) or None (local, refused, not served)
    price_basis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KeyStoreSummary:
    """What serve preconditions need from the key store, without secrets."""
    has_records: bool = False
    active: int = 0
    active_admin: int = 0
    # ids of active keys holding a cloud scope without the broker-role budget and expiry
    cloud_keys_missing_budget: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthFailure:
    status: int
    code: str
    message: str
    protocol: str = 'openai'


# Protocols (implemented by later packages; checked structurally).

@runtime_checkable
class RouterProtocol(Protocol):
    def resolve(self, ctx: RouteContext) -> RoutePlan: ...

    def explain(self, ctx: RouteContext) -> Decision: ...


@runtime_checkable
class AdvisorProtocol(Protocol):
    async def advise(self, request: Any) -> Any | None: ...  # AdvisorRequest -> AdvisorResponse | None (broker.advisor)


@runtime_checkable
class AuthenticatorProtocol(Protocol):
    @property
    def enforced(self) -> bool: ...

    def authenticate(self, scope: dict, headers: dict) -> Principal | AuthFailure: ...


@runtime_checkable
class LimiterProtocol(Protocol):
    def admit(self, principal: Principal, target: Target, estimate: Any) -> Any: ...  # -> Admission (broker.limits)


@runtime_checkable
class UsageSink(Protocol):
    def record(self, rec: Any) -> None: ...  # UsageRecord (broker.usage)


@runtime_checkable
class DecisionSink(Protocol):
    def append(self, decision: Decision) -> None: ...


@runtime_checkable
class AuditSink(Protocol):
    def event(self, kind: str, actor: str, subject: str, detail: dict) -> str: ...


__all__ = ['AdvisorOutcome', 'AdvisorProtocol', 'Attempt', 'AuditSink', 'AuthFailure', 'AuthenticatorProtocol',
           'BodySummary', 'CLOUD_POLICIES', 'Candidate', 'CloudPolicy', 'Decision', 'DecisionSink', 'EXCLUSION_REASONS',
           'EffectivePolicy', 'EgressGrant', 'Exclusion', 'ExclusionReason', 'GRANT_PURPOSES', 'GrantPurpose', 'Hints',
           'KeyBudget', 'KeyConstraints', 'KeyLimits', 'KeyStoreSummary', 'LOCATIONS', 'LimiterProtocol', 'Location',
           'OUTCOME_KINDS', 'OutcomeKind', 'PRINCIPAL_KINDS', 'Principal', 'PrincipalKind', 'ROLES', 'Role', 'RouteContext',
           'RoutePlan', 'RouterProtocol', 'Served', 'TARGET_CLASSES', 'Target', 'TargetClass', 'UpstreamError', 'Usage',
           'UsageSink', 'min_cloud_policy']
