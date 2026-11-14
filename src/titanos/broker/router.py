"""BrokerRouter: the single cloud-authorization choke point, and GrantIssuer, the only constructor of EgressGrant.

resolve(ctx) runs routing steps 1-8 in order; every refusal happens before any EgressGrant exists, and each step records
candidates and exclusions in the request's Decision:
1. loop and hop: an inbound hop equal to ours -> 508 loop_detected; any inbound hop keeps the request local (1-hop rule);
   a request that re-enters with an X-Request-ID this instance is currently sending to a router upstream
   (ROUTER_INFLIGHT) and would be sent to a router upstream AGAIN -> 508 loop_detected at that attempt (BrokerDispatch):
   a router such as Switchyard drops X-Titanos-Hop but forwards X-Request-ID, so a route that leads back into the
   router is refused on its first re-entry instead of recursing until the upstream's slots are exhausted, while a
   router route that legitimately targets this instance (a local model, or a cloud ref on another upstream) still
   works;
2. role gate: role.switches_for decides which target classes may exist at all (applied in the policy filter);
3. name resolution (catalog.resolve_name): local name, virtual model, published alias, qualified ref, compat suffix, a
   vendor SDK's bare model id on its own protocol's surface (catalog.resolve_vendor, opt-in per profile), else
   404 model_not_found; X-Titanos-Target (needs route:explicit, else 403 scope_denied) replaces it with one target
   (a virtual model named there is 400 invalid_request: the header never expands into a chain with fallback);
   model 'auto' uses policy.auto_model;
4. expansion: one target, or a virtual model's targets ordered by strategy (ordered; weighted: smooth weighted
   round-robin over targets whose breakers admit, per snapshot; cheapest: PriceTable.estimate ascending, local 0, unpriced
   last; lowest_latency: EWMA time-to-first-byte, unknown last; advisor: as listed, then step 7);
5. policy filter (policy.filter_chain), principal visibility first;
6. explicit override: X-Titanos-Target replaced steps 3-4 and is still filtered by step 5;
7. advisor (route() only; strategy advisor, or model 'auto' whose virtual model names an advisor; switches.advisors):
   the advisor may only reorder or narrow the survivors; an invalid, slow or failed answer keeps the deterministic order;
8. truncate to min(1 + broker_max_fallbacks, policy.max_fallbacks + 1, the virtual model's max_attempts); a single target
   gets one attempt.
Step 9 (admission) is admit(); dispatch calls it before the first egress and Admission.retarget per fallback attempt.

GrantIssuer.issue(purpose, ...) checks role.purpose_allowed for every purpose (managed role without delegation refuses all
of them) and resolves the credential handle bound to the upstream's exact origin; EgressClient re-checks the purpose.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from .advisor import AdvisorCandidate, AdvisorHints, AdvisorRequest, AdvisorSummary, consult, excerpt_text
from .errors import BrokerError
from .limits import Estimate
from .naming import check_model_text
from .policy import FilterContext, effective_policy, filter_chain, refusal
from .pricing import money
from .profiles import PROFILES
from .role import purpose_allowed
from .scopes import allows
from .surfaces import TOKEN_OUTPUT_SURFACES
from .types import (GRANT_PURPOSES, AdvisorOutcome, Candidate, Decision, EgressGrant, Exclusion, RouteContext, RoutePlan,
                    Target)
from .usage import utc_ts


class InflightRequestIds:
    """X-Request-IDs this process is currently sending to router upstreams (a multiset: retries and fallbacks of one
    request may hold the same id more than once)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def add(self, request_id: str) -> None:
        with self._lock:
            self._counts[request_id] = self._counts.get(request_id, 0) + 1

    def remove(self, request_id: str) -> None:
        with self._lock:
            left = self._counts.get(request_id, 0) - 1
            if left > 0:
                self._counts[request_id] = left
            else:
                self._counts.pop(request_id, None)

    def __contains__(self, request_id) -> bool:
        with self._lock:
            return request_id in self._counts


ROUTER_INFLIGHT = InflightRequestIds()

EWMA_ALPHA = 0.3


@dataclass(frozen=True)
class BrokerPlan(RoutePlan):
    """types.RoutePlan plus what dispatch needs: the virtual model's fallback triggers, the resolution kind, the alias
    inject defaults per target ref, store permission per target ref, the advisor's reasoning mode and decision notes."""
    fallback_on: frozenset = frozenset()
    resolution: str = 'local'
    named: bool = False
    virtual: str | None = None
    alias: str | None = None
    inject: dict = field(default_factory=dict)
    allow_store: dict = field(default_factory=dict)
    reasoning_mode: str | None = None
    notes: tuple = ()


class Latency:
    """EWMA of time-to-first-byte per target ref (shared across snapshots)."""

    def __init__(self, alpha: float = EWMA_ALPHA):
        self.alpha = alpha
        self._lock = threading.Lock()
        self._values: dict[str, float] = {}

    def observe(self, ref: str, ms: float | None) -> None:
        if ms is None or ms < 0:
            return
        with self._lock:
            old = self._values.get(ref)
            self._values[ref] = ms if old is None else (1 - self.alpha) * old + self.alpha * ms

    def get(self, ref: str) -> float | None:
        with self._lock:
            return self._values.get(ref)


def hop_value(instance_id: str) -> str:
    """The hop header value: sha256(instance id)[:16] (hex)."""
    return hashlib.sha256(instance_id.encode('utf-8')).hexdigest()[:16]


class GrantIssuer:
    """The only constructor of EgressGrant (a source scan asserts it). config() returns the current BrokerConfig; hop()
    the instance hop value (computed lazily on first egress)."""

    def __init__(self, settings, switches, vault, *, config: Callable[[], Any], hop: Callable[[], str] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.switches = switches
        self.vault = vault
        self._config = config
        self._hop = hop
        self._clock = clock
        self.issued = 0

    def allow_http_lan(self) -> bool:
        cfg = self._config()
        return bool(cfg.policy.allow_http_lan) if cfg is not None else False

    def issue(self, purpose: str, *, plan_target: Target | None = None, upstream_id: str | None = None,
              decision_id: str = '', request_id: str = '', delegated: bool = False, reservation=None,
              deadline: float | None = None, origin: str | None = None, advisor_id: str | None = None,
              credential_id: str | None = None, credential=None) -> EgressGrant:
        """An EgressGrant for one outbound call. Raises 403 role_forbids when the role refuses the purpose, and the
        vault's credential_unavailable / vault_sealed when the credential cannot be used. upstream purposes resolve the
        upstream's credential for its exact origin (template Upstream.auth or the profile's); advisor grants take the
        advisor url's origin and an optional purpose-advisor credential. credential= passes an already-built handle
        (the credentials:write validate and rotate flows)."""
        if purpose not in GRANT_PURPOSES:
            raise ValueError(f'Unknown egress purpose: {purpose!r}')
        if not purpose_allowed(self.switches, purpose, delegated=bool(delegated)):
            raise BrokerError(f'Egress purpose {purpose} is not allowed in role {getattr(self.settings, "role", "?")}',
                              'role_forbids', origin='policy')
        if deadline is None:
            deadline = self._clock() + float(getattr(self.settings, 'client_timeout', 600.0))
        if upstream_id is not None:
            cfg = self._config()
            upstream = cfg.upstreams.get(upstream_id) if cfg is not None else None
            if upstream is None:
                raise BrokerError(f'Upstream {upstream_id} is not configured', 'upstream_unconfigured')
            profile = PROFILES[upstream.profile]
            if profile.kind == 'slot' or profile.adapter == 'none':
                raise BrokerError(f'Upstream {upstream_id} has no wire contract', 'upstream_unconfigured')
            try:
                origin = upstream.origin(allow_http_lan=bool(cfg.policy.allow_http_lan))
            except ValueError as exc:
                raise BrokerError(f'Upstream {upstream_id} has an unusable base_url: {exc}', 'upstream_unconfigured') \
                    from None
            if origin is None:
                raise BrokerError(f'Upstream {upstream_id} has no base_url', 'upstream_unconfigured')
            if credential is None and upstream.credential:
                credential = self.vault.handle(upstream.credential, origin=origin, template=upstream.effective_auth(),
                                               purpose='inference', upstream_id=upstream_id)
        else:
            if origin is None:
                raise ValueError('A grant without an upstream needs the origin of its URL')
            if credential is None and credential_id:
                credential = self.vault.handle(credential_id, origin=origin, template='bearer', purpose='advisor')
        hop = None
        if self._hop is not None:
            try:
                hop = self._hop()
            except Exception:  # the hop value is best effort (a read-only home); the cap still bounds loops
                hop = None
        self.issued += 1
        return EgressGrant(decision_id=decision_id, request_id=request_id, purpose=purpose, delegated=bool(delegated),
                           target=plan_target, upstream_id=upstream_id, origin=origin, credential=credential,
                           deadline=deadline, reservation=reservation, hop_header=hop)


@dataclass
class _Draft:
    ctx: RouteContext
    eff: Any
    resolution: str
    named: bool
    targets: list
    survivors: list
    exclusions: list
    delegable: bool = False
    virtual: Any = None
    virtual_name: str | None = None
    alias: str | None = None
    inject: dict = field(default_factory=dict)
    advisor_id: str | None = None
    advisor: AdvisorOutcome | None = None
    reasoning_mode: str | None = None
    notes: list = field(default_factory=list)


class BrokerRouter:
    """Routing for one BrokerSnapshot. config: the snapshot's BrokerConfig (or an object with .config)."""

    def __init__(self, config, settings, switches, catalog, breakers=None, advisors=None, limiter=None, prices=None,
                 handles=None, *, latency: Latency | None = None, hop: Callable[[], str] | None = None):
        self.config = getattr(config, 'config', config)
        self.settings = settings
        self.switches = switches
        self.catalog = catalog
        self.breakers = breakers
        self.advisors = dict(advisors or {})
        self.limiter = limiter
        self.prices = prices
        self.handles = handles
        self.latency = latency or Latency()
        self._hop = hop
        self._swrr: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    # -- decisions -------------------------------------------------------------------------------------------------------
    def new_decision(self, ctx: RouteContext, *, inbound_protocol: str | None = None) -> Decision:
        hints = ctx.hints
        return Decision(ts=utc_ts(), request_id=hints.request_id or uuid.uuid4().hex, decision_id=uuid.uuid4().hex,
                        trace_id=hints.trace_id, key_id=ctx.principal.key_id if ctx.principal is not None else None,
                        role=getattr(self.settings, 'role', 'standalone'), surface=ctx.surface.name,
                        inbound_protocol=inbound_protocol or ctx.surface.protocol, requested_model=ctx.requested_model,
                        dry_run=ctx.dry_run,
                        hints={'location': hints.location, 'target_header': hints.target_header,
                               'delegated': hints.delegated, 'hop': hints.hop})

    # -- steps -----------------------------------------------------------------------------------------------------------
    def _loop(self, ctx: RouteContext) -> None:
        inbound = ctx.hints.hop_value
        if inbound and self._hop is not None:
            try:
                ours = self._hop()
            except Exception:
                ours = None
            if ours and inbound == ours:
                raise BrokerError('This request already passed through this Titanos instance', 'loop_detected',
                                  origin='policy')

    def _resolve_name(self, ctx: RouteContext) -> _Draft:
        principal = ctx.principal
        name = ctx.requested_model
        explicit = ctx.hints.target_header
        if explicit is not None:
            if not allows(principal, 'route:explicit'):
                raise BrokerError('X-Titanos-Target needs the route:explicit scope', 'scope_denied', origin='policy')
            name = explicit
        check_model_text(name)
        cfg = self.config
        if explicit is not None:
            # X-Titanos-Target replaces name resolution and expansion with ONE target: a virtual model never expands
            # here (no chain, no fallback), so a delegated or explicit-policy request cannot widen through the header.
            found = self.catalog.resolve_name(name, allow_virtual=False)
            if found is None and name in cfg.virtual_models:
                raise BrokerError('X-Titanos-Target names one target; a virtual model cannot be used there',
                                  'invalid_request', origin='policy')
        else:
            found = self.catalog.resolve_name(name)
            if found is None:
                # a vendor SDK's bare model id (eleven_multilingual_v2, gpt-4o-mini, gemini-2.5-flash) on its own
                # protocol's surface; local models, aliases and virtual models were tried first
                found = self.catalog.resolve_vendor(name, ctx.surface.protocol, principal)
        advisor_id = None
        if found is None and name == 'auto' and cfg.policy.auto_model and explicit is None:
            found = self.catalog.resolve_name(cfg.policy.auto_model)
            if found is not None and found.virtual is not None:
                advisor_id = found.virtual.advisor
        if found is None:
            raise BrokerError(f'Model {name!r} not found', 'model_not_found')
        draft = _Draft(ctx=ctx, eff=None, resolution=found.kind, named=explicit is not None or found.kind in
                       ('alias', 'qualified', 'compat', 'vendor'), targets=[], survivors=[], exclusions=[],
                       delegable=found.kind != 'virtual' and (explicit is not None or found.kind == 'qualified'))
        if found.kind == 'virtual':
            draft.virtual, draft.virtual_name = found.virtual, found.virtual_name or name
            if found.virtual.strategy == 'advisor':
                advisor_id = found.virtual.advisor
            for item in found.virtual.targets:
                try:
                    resolved = self.catalog.resolve_name(item.ref, allow_virtual=False)
                except BrokerError:
                    resolved = None
                if resolved is None or resolved.target is None:
                    draft.exclusions.append(Exclusion(item.ref, 'disabled', 'unresolved'))
                    continue
                draft.targets.append(resolved.target)
                if resolved.inject:
                    draft.inject[resolved.target.ref] = dict(resolved.inject)
            if not draft.targets:
                raise BrokerError(f'Model {name!r} not found', 'model_not_found')
            draft.targets = self._order(found.virtual, draft.virtual_name, draft.targets, ctx)
        else:
            draft.targets = [found.target]
            draft.alias = found.alias
            if found.inject:
                draft.inject[found.target.ref] = dict(found.inject)
        draft.advisor_id = advisor_id
        return draft

    def _order(self, virtual, name: str, targets: list, ctx: RouteContext) -> list:
        strategy = virtual.strategy
        if strategy == 'weighted':
            weights = {item.ref: item.weight or 1 for item in virtual.targets}
            return self._weighted(name, targets, weights)
        if strategy == 'cheapest':
            estimate = estimate_for(ctx)

            def cost(target):
                if target.klass == 'local':
                    return (0, Decimal(0))
                price = self.prices.estimate(target.ref, estimate) if self.prices is not None else None
                return (1, Decimal(0)) if price is None else (0, price)
            indexed = list(enumerate(targets))
            indexed.sort(key=lambda pair: (cost(pair[1]), pair[0]))
            return [t for _, t in indexed]
        if strategy == 'lowest_latency':
            indexed = list(enumerate(targets))

            def latency(pair):
                value = self.latency.get(pair[1].ref)
                return (1, 0.0, pair[0]) if value is None else (0, value, pair[0])
            indexed.sort(key=latency)
            return [t for _, t in indexed]
        return list(targets)

    def _admits(self, target: Target) -> bool:
        if target.klass == 'local' or self.breakers is None:
            return True
        upstream = self.config.upstreams.get(target.upstream_id)
        if upstream is None:
            return False
        from .health import breaker_for
        return bool(breaker_for(self.breakers, target, upstream).allow(consume=False))

    def _weighted(self, name: str, targets: list, weights: dict) -> list:
        """Smooth weighted round-robin (nginx): the pick goes first, the rest follow by weight, then listing order."""
        eligible = [t for t in targets if self._admits(t)]
        if not eligible:
            return list(targets)
        with self._lock:
            state = self._swrr.setdefault(name, {})
            total = 0
            best = None
            for target in eligible:
                weight = weights.get(target.ref, 1)
                state[target.ref] = state.get(target.ref, 0) + weight
                total += weight
                if best is None or state[target.ref] > state[best.ref]:
                    best = target
            state[best.ref] -= total
        rest = [t for t in targets if t is not best]
        rest.sort(key=lambda t: -weights.get(t.ref, 1))
        return [best] + rest

    def _filter(self, draft: _Draft, *, body_location: str | None = None) -> None:
        ctx = draft.ctx
        if draft.eff is None:
            draft.eff = effective_policy(self.settings, self.config, self.switches, ctx.principal, ctx.hints,
                                         ctx.surface, body_location=body_location)
        context = FilterContext(cfg=self.config, switches=self.switches, catalog=self.catalog, named=draft.named,
                                hints=ctx.hints, delegable=draft.delegable)
        survivors, exclusions = filter_chain(draft.targets, draft.eff, ctx.principal, ctx.surface, ctx.body,
                                             breakers=self.breakers, handles=self.handles, context=context)
        draft.survivors = survivors
        draft.exclusions.extend(exclusions)

    def _record(self, draft: _Draft, decision: Decision | None) -> None:
        if decision is None:
            return
        if draft.eff is not None:
            decision.effective = {'cloud_policy': draft.eff.cloud_policy, 'location': draft.eff.location}
        excluded = {}
        for item in draft.exclusions:
            excluded.setdefault(item.ref, item.reason)
        survivors = {t.ref for t in draft.survivors}
        decision.candidates = [Candidate(ref=t.ref, klass=t.klass, eligible=t.ref in survivors,
                                         excluded=None if t.ref in survivors else excluded.get(t.ref))
                               for t in draft.targets]
        decision.advisor = draft.advisor
        for note in draft.notes:
            if note not in decision.notes:
                decision.notes.append(note)

    def _prepare(self, ctx: RouteContext, decision: Decision | None, *, body_location: str | None = None) -> _Draft:
        self._loop(ctx)
        draft = self._resolve_name(ctx)
        try:
            self._filter(draft, body_location=body_location)
        finally:
            self._record(draft, decision)
        if not draft.survivors:
            raise refusal(draft.exclusions, named=draft.named)
        return draft

    def _allow_store(self, target: Target, principal) -> bool:
        if target.klass == 'local':
            return False
        upstream = self.config.upstreams.get(target.upstream_id)
        return bool(upstream is not None and upstream.data_policy.allow_store and allows(principal, 'cloud:store'))

    def _finish(self, draft: _Draft, decision: Decision | None) -> BrokerPlan:
        cfg = self.config
        ctx = draft.ctx
        limit = min(1 + int(getattr(self.settings, 'broker_max_fallbacks', 2)), cfg.policy.max_fallbacks + 1)
        fallback_on = frozenset()
        if draft.virtual is not None:
            fallbacks = min(cfg.policy.max_fallbacks, int(getattr(self.settings, 'broker_max_fallbacks', 2)))
            limit = min(limit, draft.virtual.effective_max_attempts(fallbacks))
            fallback_on = frozenset(draft.virtual.fallback_on)
        else:
            limit = 1
        if ctx.surface.capability not in ('chat', 'responses', 'embeddings') and draft.virtual is None:
            limit = 1
        if getattr(draft.eff, 'role', None) == 'managed' and any(t.klass != 'local' for t in draft.survivors):
            # managed role: an explicit target only, never a fallback (roles table; the parent retries)
            limit, fallback_on = 1, frozenset()
        chain = tuple(draft.survivors[:max(1, limit)])
        notes = list(draft.notes)
        allow_store = {t.ref: self._allow_store(t, ctx.principal) for t in chain if t.klass != 'local'}
        if ctx.body.store and not all(allow_store.get(t.ref, False) for t in chain if t.klass != 'local'):
            notes.append('forced:store=false')
        plan = BrokerPlan(chain=chain, effective=draft.eff, exclusions=tuple(draft.exclusions), advisor=draft.advisor,
                          max_attempts=len(chain), fallback_on=fallback_on, resolution=draft.resolution,
                          named=draft.named, virtual=draft.virtual_name, alias=draft.alias,
                          inject={k: v for k, v in draft.inject.items()}, allow_store=allow_store,
                          reasoning_mode=draft.reasoning_mode, notes=tuple(notes))
        if decision is not None:
            decision.chain = [t.ref for t in chain]
            for note in notes:
                if note not in decision.notes:
                    decision.notes.append(note)
        return plan

    async def _advise(self, draft: _Draft, decision: Decision | None, advisor_input: Callable[[], dict] | None) -> None:
        advisor_id = draft.advisor_id
        if advisor_id is None or not getattr(self.switches, 'advisors', False) or len(draft.survivors) < 1:
            return
        advisor = self.advisors.get(advisor_id)
        if advisor is None:
            return
        ctx = draft.ctx
        extra = {}
        if advisor_input is not None:
            try:
                extra = advisor_input() or {}
            except Exception:
                extra = {}
        candidates = []
        for target in draft.survivors:
            price = None
            row = self.prices.match(target.ref) if (self.prices is not None and target.klass != 'local') else None
            if row is not None:
                price = {'input_per_mtok': money(row.input_per_mtok), 'output_per_mtok': money(row.output_per_mtok)}
            breaker = 'closed'
            if target.klass != 'local' and self.breakers is not None:
                upstream = self.config.upstreams[target.upstream_id]
                from .health import breaker_for
                breaker = breaker_for(self.breakers, target, upstream).state
                breaker = 'half_open' if breaker == 'half_open' else 'closed'
            latency = self.latency.get(target.ref)
            candidates.append(AdvisorCandidate(ref=target.ref, klass=target.klass, upstream=target.upstream_id,
                                               capabilities=sorted(target.capabilities), breaker=breaker, price=price,
                                               p50_ttfb_ms=int(latency) if latency is not None else None))
        body = ctx.body
        request = AdvisorRequest(
            request_id=ctx.hints.request_id or uuid.uuid4().hex, surface=ctx.surface.name,
            capability=ctx.surface.capability, requested_model=ctx.requested_model,
            hints=AdvisorHints(location=draft.eff.location if draft.eff else None),
            summary=AdvisorSummary(message_count=int(extra.get('message_count') or 0),
                                   approx_input_tokens=body.approx_input_tokens, has_images=body.has_images,
                                   has_tools=body.has_tools, has_audio=body.has_audio,
                                   excerpt=excerpt_text(extra.get('excerpt')) if advisor.share_excerpt else None),
            candidates=candidates)
        outcome, picked, mode = await consult(advisor, request, decision_id=decision.decision_id if decision else '',
                                              deadline=ctx.hints.deadline)
        draft.advisor = outcome
        if picked is None:
            if outcome.reason:
                draft.notes.append(f'advisor:{outcome.reason}')
            return
        choice, fallbacks = picked
        by_ref = {t.ref: t for t in draft.survivors}
        ordered = [by_ref[choice]] + [by_ref[ref] for ref in fallbacks if ref in by_ref]
        draft.survivors = ordered
        draft.reasoning_mode = mode
        if mode:
            draft.notes.append(f'advisor:reasoning_mode={mode}')

    # -- public API ------------------------------------------------------------------------------------------------------
    def resolve(self, ctx: RouteContext, decision: Decision | None = None, *, body_location: str | None = None
                ) -> BrokerPlan:
        """Steps 1-6 and 8 (no advisor). Raises BrokerError before any egress."""
        draft = self._prepare(ctx, decision, body_location=body_location)
        return self._finish(draft, decision)

    def recheck(self, ctx: RouteContext, target: Target, *, body_location: str | None = None) -> None:
        """Re-check one target that was resolved earlier (a queued job, an API Plus stage) against THIS snapshot and the
        principal as it is now: the effective policy and the filter stages (visibility, role and hop, location, cloud
        policy, capability, data policy, readiness). The target counts as named and delegable (it was resolved under
        those rules); breakers are left to the caller. Raises the router's refusal; no egress."""
        eff = effective_policy(self.settings, self.config, self.switches, ctx.principal, ctx.hints, ctx.surface,
                               body_location=body_location)
        context = FilterContext(cfg=self.config, switches=self.switches, catalog=self.catalog, named=True,
                                hints=ctx.hints, delegable=True)
        survivors, exclusions = filter_chain([target], eff, ctx.principal, ctx.surface, ctx.body, breakers=None,
                                             handles=self.handles, context=context)
        if not survivors:
            raise refusal(exclusions, named=True)

    async def route(self, ctx: RouteContext, decision: Decision | None = None, *, advisor: bool = True,
                    advisor_input: Callable[[], dict] | None = None, body_location: str | None = None) -> BrokerPlan:
        """Steps 1-8, the advisor included when advisor=True."""
        draft = self._prepare(ctx, decision, body_location=body_location)
        if advisor:
            await self._advise(draft, decision, advisor_input)
        plan = self._finish(draft, decision)
        if decision is not None:
            decision.advisor = draft.advisor
        return plan

    async def explain(self, ctx: RouteContext, *, advisor: bool = False,
                      advisor_input: Callable[[], dict] | None = None) -> Decision:
        """Steps 1-8 with no egress, no reservation and no advisor call unless advisor=True: the Decision with
        dry_run=True and served=None (a refusal is recorded as the note refused:<code>)."""
        ctx = replace(ctx, dry_run=True)
        decision = self.new_decision(ctx)
        try:
            await self.route(ctx, decision, advisor=advisor, advisor_input=advisor_input)
        except BrokerError as exc:
            decision.notes.append(f'refused:{exc.code}')
        decision.served = None
        return decision

    def admit(self, plan: BrokerPlan, ctx: RouteContext, estimate: Estimate | None = None):
        """Step 9: Limiter.admit against the first non-local target (budget reservation and injected max_tokens), else
        the first (local) target for RPM, TPM and concurrency only. None without a limiter."""
        if self.limiter is None or not plan.chain:
            return None
        first = next((t for t in plan.chain if t.klass != 'local'), plan.chain[0])
        return self.limiter.admit(ctx.principal, first, estimate if estimate is not None else estimate_for(ctx))


def estimate_for(ctx: RouteContext) -> Estimate:
    """The pre-dispatch Estimate of one request. Only a token-generating surface (surfaces.TOKEN_OUTPUT_SURFACES) keeps
    the body's output cap, None when it sets none (a budgeted key then gets the default cap injected); every other
    surface is token_output=False, so the limiter never computes or injects a cap for it."""
    body = ctx.body
    capability = ctx.surface.capability
    if ctx.surface.name in TOKEN_OUTPUT_SURFACES:
        return Estimate(input_tokens=body.approx_input_tokens, max_output_tokens=body.max_output_tokens)
    if capability in ('chat', 'responses', 'embeddings', 'moderation', 'rerank'):
        # embeddings, moderations, rerank, the token counters and the handle-bound Responses routes generate no tokens
        return Estimate(input_tokens=body.approx_input_tokens, max_output_tokens=0, token_output=False)
    if capability == 'speech':
        return Estimate(max_output_tokens=0, characters=max(0, body.approx_input_tokens * 3), token_output=False)
    if capability == 'images':
        return Estimate(max_output_tokens=0, images=1, token_output=False)
    if capability == 'transcription':
        # the upload's duration (measured when the container states it, an over-estimate otherwise); a body without a
        # file (a URL reference) reserves its byte size at the same floor rate, at least one second
        seconds = body.audio_seconds
        if seconds is None:
            from .limits import AUDIO_FLOOR_BYTES_PER_SECOND
            seconds = max(1.0, body.bytes / AUDIO_FLOOR_BYTES_PER_SECOND)
        return Estimate(input_tokens=0, max_output_tokens=0, audio_seconds=float(seconds), token_output=False)
    return Estimate(input_tokens=0, max_output_tokens=0, token_output=False)


__all__ = ['BrokerPlan', 'BrokerRouter', 'GrantIssuer', 'Latency', 'estimate_for', 'hop_value']
