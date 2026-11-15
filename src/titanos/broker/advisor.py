"""Router-advisor contract titanos.router-advisor/v1 (routers R4) and its three profiles.

- generic-v1: Titanos POSTs an AdvisorRequest to the advisor's url and reads an AdvisorResponse.
- hivemind-gim-router: POST {url}/gim/llm_router/func_1 with {messages, routing_preference, system_context}; the answer
  {"ok": true, "result": {recommended_model, reasoning_mode, confidence, route_reason, scores, fallback_chain,
  classification, latency_ms}} is mapped onto the candidates through label_map (simple | complex | reasoning). The GIM's
  fallback_chain IS used (each entry is mapped to a label, then to a candidate ref). The GIM needs the prompt, so this
  profile is consulted only with share_excerpt=true; otherwise it is never called (reason advisor_unconfigured).
- pending-contract: an inert, named slot (status contract_unknown). It is never called; a url is refused with 409
  advisor_profile_unconfigured by the broker.json schema.

Every call goes through EgressClient with a GrantIssuer purpose 'advisor' grant (managed role refuses it). The advisor
never receives credentials, upstream URLs or prompt text unless share_excerpt is set (then at most 512 scrubbed characters).
validate_advice keeps an answer only when the contract string matches and the choice and every fallback are candidates;
the router may then only reorder or narrow its deterministic chain. An invalid, slow or failed answer leaves the
deterministic order (reasons advisor_invalid, advisor_timeout, advisor_failed).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import parse_origin
from .errors import BrokerError
from .redact import scrub
from .types import AdvisorOutcome

CONTRACT = 'titanos.router-advisor/v1'
EXCERPT_MAX = 512
EXPLANATION_MAX = 256
GIM_PATH = '/gim/llm_router/func_1'
LABELS = ('simple', 'complex', 'reasoning')
PROFILES = ('generic-v1', 'hivemind-gim-router', 'pending-contract')
REASONING_MODES = (None, 'none', 'low', 'medium', 'high')


class AdvisorHints(BaseModel):
    model_config = ConfigDict(extra='forbid')
    location: str | None = None
    latency_class: str | None = None
    max_cost_usd: str | None = None


class AdvisorSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message_count: int = 0
    approx_input_tokens: int = 0
    has_images: bool = False
    has_tools: bool = False
    has_audio: bool = False
    excerpt: str | None = Field(None, max_length=EXCERPT_MAX)


class AdvisorPrice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    input_per_mtok: str | None = None
    output_per_mtok: str | None = None


class AdvisorCandidate(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    ref: str
    klass: Literal['local', 'cloud', 'router'] = Field(alias='class')
    upstream: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    breaker: Literal['closed', 'half_open'] = 'closed'
    price: AdvisorPrice | None = None
    p50_ttfb_ms: int | None = None


class AdvisorRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    contract: Literal['titanos.router-advisor/v1'] = CONTRACT
    request_id: str
    surface: str
    capability: str
    requested_model: str
    hints: AdvisorHints = Field(default_factory=AdvisorHints)
    summary: AdvisorSummary = Field(default_factory=AdvisorSummary)
    candidates: list[AdvisorCandidate]

    def wire(self) -> dict:
        return self.model_dump(mode='json', by_alias=True)


class AdvisorResponse(BaseModel):
    model_config = ConfigDict(extra='ignore')
    contract: str
    choice: str
    fallbacks: list[str] = Field(default_factory=list, max_length=16)
    reasoning_mode: Literal['none', 'low', 'medium', 'high'] | None = None
    confidence: float | None = Field(None, ge=0, le=1)
    explanation: str | None = Field(None, max_length=EXPLANATION_MAX)


def validate_advice(resp, candidates) -> tuple[str, list[str]] | None:
    """(choice, fallbacks) when resp is a valid AdvisorResponse (or dict) over these candidates (refs, or objects with
    .ref, or dicts with 'ref'); None otherwise (wrong contract, a foreign choice or fallback, a malformed answer)."""
    if resp is None:
        return None
    if not isinstance(resp, AdvisorResponse):
        try:
            resp = AdvisorResponse.model_validate(resp)
        except (ValidationError, TypeError, ValueError):
            return None
    if resp.contract != CONTRACT:
        return None
    refs = []
    for item in candidates or ():
        ref = item if isinstance(item, str) else (item.get('ref') if isinstance(item, dict) else getattr(item, 'ref', None))
        if isinstance(ref, str):
            refs.append(ref)
    allowed = set(refs)
    if resp.choice not in allowed or any(ref not in allowed for ref in resp.fallbacks):
        return None
    fallbacks = []
    for ref in resp.fallbacks:
        if ref != resp.choice and ref not in fallbacks:
            fallbacks.append(ref)
    return resp.choice, fallbacks


def excerpt_text(text: str | None) -> str | None:
    """At most EXCERPT_MAX characters, scrubbed of secret shapes."""
    if not text:
        return None
    return scrub(str(text))[:EXCERPT_MAX]


class _Advisor:
    profile = ''

    def __init__(self, advisor_id: str, spec, egress=None, issuer=None, *, settings=None):
        self.id = advisor_id
        self.spec = spec
        self.egress = egress
        self.issuer = issuer
        self.settings = settings
        self.calls = 0

    @property
    def share_excerpt(self) -> bool:
        return bool(getattr(self.spec, 'share_excerpt', False))

    def timeout_seconds(self) -> float:
        """The advisor's timeout_ms when the operator set it, else settings.broker_advisor_timeout_ms (default 300)."""
        spec_ms = getattr(self.spec, 'timeout_ms', None)
        explicit = 'timeout_ms' in getattr(self.spec, 'model_fields_set', set())
        ms = spec_ms if explicit and spec_ms else getattr(self.settings, 'broker_advisor_timeout_ms', spec_ms or 300)
        return max(1, min(int(ms), 5000)) / 1000.0

    def callable(self) -> str | None:
        """None when the advisor may be called; otherwise the reason it is not."""
        return None

    async def _post(self, url: str, payload: dict, *, decision_id: str, request_id: str, deadline: float) -> Any:
        if self.egress is None or self.issuer is None:
            raise BrokerError('Advisor egress is not configured', 'upstream_unconfigured')
        origin = parse_origin(url, allow_http_lan=self.issuer.allow_http_lan())
        grant = self.issuer.issue('advisor', upstream_id=None, decision_id=decision_id, request_id=request_id,
                                  deadline=deadline, origin=origin, advisor_id=self.id,
                                  credential_id=getattr(self.spec, 'credential', None))
        self.calls += 1
        response = await self.egress.send(grant, None, None, 'POST', url, json=payload)
        if not 200 <= response.status_code < 300:
            raise BrokerError(f'Advisor answered {response.status_code}', 'upstream_bad_response', origin='upstream')
        try:
            return json.loads(response.content.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise BrokerError('Advisor answered with a body that is not JSON', 'upstream_bad_response',
                              origin='upstream') from None

    async def advise(self, request: AdvisorRequest, *, decision_id: str = '', deadline: float | None = None):
        raise NotImplementedError


class GenericV1Advisor(_Advisor):
    profile = 'generic-v1'

    def callable(self) -> str | None:
        return None if getattr(self.spec, 'url', None) else 'advisor_unconfigured'

    async def advise(self, request: AdvisorRequest, *, decision_id: str = '', deadline: float | None = None):
        payload = request.wire()
        if not self.share_excerpt:
            payload['summary']['excerpt'] = None
        answer = await self._post(self.spec.url, payload, decision_id=decision_id, request_id=request.request_id,
                                  deadline=deadline if deadline is not None else time.monotonic() + 5)
        try:
            return AdvisorResponse.model_validate(answer)
        except (ValidationError, TypeError, ValueError):
            return None


def gim_label(model_name: str | None, result: dict | None = None) -> str | None:
    """The Titanos label (simple | complex | reasoning) of a GIM model name or recommendation: reasoning when the GIM
    turns reasoning on for its recommendation, complex for its big tiers, simple for the mid and small tiers."""
    if result is not None and result.get('reasoning_mode') == 'on':
        return 'reasoning'
    name = (model_name or '').upper()
    if not name:
        return None
    if 'BIG' in name or 'NIM' in name or name == 'LLM_UNIVERSAL':
        return 'complex'
    if 'MID' in name or 'SMALL' in name:
        return 'simple'
    return None


class GimRouterAdvisor(_Advisor):
    profile = 'hivemind-gim-router'

    def callable(self) -> str | None:
        if not getattr(self.spec, 'url', None):
            return 'advisor_unconfigured'
        if not self.share_excerpt:
            return 'advisor_unconfigured'  # the GIM classifies the prompt itself; without share_excerpt it is not called
        return None

    def map_result(self, result: dict, candidate_refs) -> AdvisorResponse | None:
        refs = list(candidate_refs)
        label_map = dict(getattr(self.spec, 'label_map', None) or {})

        def to_ref(name, *, with_result=None):
            if name in refs:
                return name
            label = gim_label(name, with_result)
            return label_map.get(label) if label else None

        choice = to_ref(result.get('recommended_model'), with_result=result)
        if choice is None:
            return None
        fallbacks = []
        for name in result.get('fallback_chain') or []:
            ref = to_ref(name) if isinstance(name, str) else None
            if ref and ref != choice and ref not in fallbacks:
                fallbacks.append(ref)
        mode = {'on': 'medium', 'off': None}.get(result.get('reasoning_mode'))
        confidence = result.get('confidence')
        return AdvisorResponse(contract=CONTRACT, choice=choice, fallbacks=fallbacks, reasoning_mode=mode,
                               confidence=confidence if isinstance(confidence, (int, float)) and 0 <= confidence <= 1
                               else None,
                               explanation=str(result.get('route_reason') or '')[:EXPLANATION_MAX] or None)

    async def advise(self, request: AdvisorRequest, *, decision_id: str = '', deadline: float | None = None):
        excerpt = request.summary.excerpt if self.share_excerpt else None
        payload = {'messages': [{'role': 'user', 'content': excerpt}] if excerpt else [],
                   'routing_preference': 'balanced', 'system_context': {}}
        url = self.spec.url.rstrip('/') + GIM_PATH
        answer = await self._post(url, payload, decision_id=decision_id, request_id=request.request_id,
                                  deadline=deadline if deadline is not None else time.monotonic() + 5)
        if not isinstance(answer, dict) or answer.get('ok') is not True or not isinstance(answer.get('result'), dict):
            return None
        return self.map_result(answer['result'], [c.ref for c in request.candidates])


class PendingContractAdvisor(_Advisor):
    """The inert slot: never called."""
    profile = 'pending-contract'
    status = 'contract_unknown'

    def callable(self) -> str | None:
        return 'contract_unknown'

    async def advise(self, request: AdvisorRequest, *, decision_id: str = '', deadline: float | None = None):
        return None


_CLASSES = {'generic-v1': GenericV1Advisor, 'hivemind-gim-router': GimRouterAdvisor,
            'pending-contract': PendingContractAdvisor}


def build_advisor(spec, egress, vault, issuer, *, advisor_id: str = 'advisor', settings=None):
    """The advisor for a broker.json AdvisorSpec. vault is not used directly: the advisor credential (purpose advisor) is
    resolved by the GrantIssuer at call time."""
    cls = _CLASSES.get(getattr(spec, 'profile', None))
    if cls is None:
        raise ValueError(f'Unknown advisor profile: {getattr(spec, "profile", None)!r}')
    return cls(advisor_id, spec, egress, issuer, settings=settings)


async def consult(advisor, request: AdvisorRequest, *, decision_id: str = '', deadline: float | None = None,
                  clock=time.perf_counter):
    """(AdvisorOutcome, (choice, fallbacks) | None, reasoning_mode). Never raises for advisor trouble."""
    started = clock()
    why = advisor.callable()
    if why is not None:
        return AdvisorOutcome(profile=advisor.profile, ms=0.0, choice=None, valid=False, reason=why), None, None
    timeout = advisor.timeout_seconds()
    reason = None
    resp = None
    try:
        resp = await asyncio.wait_for(advisor.advise(request, decision_id=decision_id, deadline=deadline),
                                      timeout=timeout)
    except asyncio.TimeoutError:
        reason = 'advisor_timeout'
    except BrokerError as exc:
        reason = 'advisor_timeout' if exc.code in ('upstream_timeout', 'deadline_exceeded') else 'advisor_failed'
    except Exception:  # network trouble: the deterministic order stands
        reason = 'advisor_failed'
    ms = round((clock() - started) * 1000.0, 3)
    if reason is not None:
        return AdvisorOutcome(profile=advisor.profile, ms=ms, choice=None, valid=False, reason=reason), None, None
    picked = validate_advice(resp, request.candidates)
    if picked is None:
        choice = getattr(resp, 'choice', None) if isinstance(resp, AdvisorResponse) else None
        return (AdvisorOutcome(profile=advisor.profile, ms=ms, choice=scrub(choice)[:256] if choice else None,
                               valid=False, reason='advisor_invalid'), None, None)
    choice, fallbacks = picked
    return (AdvisorOutcome(profile=advisor.profile, ms=ms, choice=choice, fallbacks=tuple(fallbacks), valid=True),
            picked, resp.reasoning_mode)


__all__ = ['AdvisorCandidate', 'AdvisorHints', 'AdvisorRequest', 'AdvisorResponse', 'AdvisorSummary', 'CONTRACT',
           'EXCERPT_MAX', 'GenericV1Advisor', 'GimRouterAdvisor', 'LABELS', 'PROFILES', 'PendingContractAdvisor',
           'build_advisor', 'consult', 'excerpt_text', 'gim_label', 'validate_advice']
