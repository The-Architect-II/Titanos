"""Effective policy and the candidate filter (routing step 5).

effective_policy: the cloud policy is min(TITANOS_BROKER_CLOUD_CEILING, broker.json cloud_policy) ordered
off < explicit < fallback; managed role forces off unless the request is parent-delegated (then the minimum of that and
explicit, so a broker.json off stays off). The location follows the precedence: managed parent placement local_only
(authoritative, fail-closed), X-Titanos-Location (or a valid parent placement), body.location on API Plus routes, the
key's default_location, broker.json policy.default_location; it must be one of the key's constraints.locations
(400 location_unsatisfiable otherwise).

filter_chain runs the stages below in order. Each removes candidates and records an Exclusion; when a stage removes the
last candidate the filter stops, and refusal() turns that stage's reason into the error, so principal visibility always
answers before any policy refusal (no 403 oracle for invisible targets):
  1. visibility (reason scope): inference scope for the surface; cloud or cloud:<upstream> and the key's model, upstream
     and target-class constraints (scopes.can_use_target) -> 404 model_not_found. The one exception is the keyless
     loopback principal (the machine's owner, who may already list every upstream): a cloud or router target it misses
     only because policy.anonymous_cloud is off answers 403 cloud_disabled naming that switch (detail keyless_cloud_off);
  2. handle ownership (reason policy, detail handle): a previous_response_id the caller does not own -> 404
     response_not_found (raised at once); an owned one keeps only targets on the owning upstream;
  3. role and hop (reason role / location): managed role without delegation, router upstreams switched off, delegated
     requests whose target is not a qualified ref or X-Titanos-Target (aliases and compat suffixes do not count), and
     the 1-hop rule (an inbound hop header keeps local targets);
  4. location: local_only keeps local targets, cloud_only keeps cloud and router targets, prefer_local sorts local first
     and keeps non-local targets only under cloud_policy fallback (a chain without any local target is left as it is),
     auto keeps the configured order -> 400 location_unsatisfiable;
  5. cloud policy (reason policy): off drops cloud and router targets (403 cloud_disabled); explicit keeps them only when
     the caller named them (a qualified ref, a published alias or X-Titanos-Target), and a virtual model keeps only targets
     of its first target's class; fallback keeps everything;
  6. capability (cloud and router targets; local targets keep their own strict validation): the surface must be served
     natively or translated by the upstream's adapter and its capability must be one the target offers -> 400
     broker_capability_unavailable;
  7. data_policy: the modalities in the body (text, image, audio) must be allowed by the upstream -> 403
     data_policy_denied;
  8. readiness: enabled, credential validated for the current origin (or auth none), not the pending_contract slot ->
     404 model_not_found (the slot and a credential-less upstream that needs one: 503 upstream_unconfigured);
  9. breakers: an open breaker or a 429 cooldown -> 503 breaker_open with Retry-After = the minimum remaining time; when
     every remaining candidate is only waiting out a provider 429 cooldown -> 429 upstream_rate_limited (Retry-After).
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import BrokerError
from .profiles import PROFILES
from .scopes import allows, can_use_target
from .types import EffectivePolicy, Exclusion, LOCATIONS, min_cloud_policy

LOCATION_ORDER = LOCATIONS  # local_only < prefer_local < auto < cloud_only (how far a request may leave the host)
REMOTE = frozenset({'cloud', 'router'})


def cloud_policy_for(settings, cfg, switches, *, delegated: bool) -> str:
    """min(TITANOS_BROKER_CLOUD_CEILING, broker.json cloud_policy). Managed role only lowers it further: off unless the
    request is parent-delegated, and then at most explicit (broker.json off stays off)."""
    ceiling = getattr(settings, 'broker_cloud_ceiling', 'fallback')
    configured = cfg.policy.cloud_policy if cfg is not None else 'off'
    cloud = min_cloud_policy(ceiling, configured)
    if getattr(settings, 'role', 'standalone') == 'managed':
        return min_cloud_policy(cloud, 'explicit') if delegated else 'off'
    if not getattr(switches, 'cloud_egress', True):
        return 'off'
    return cloud


def effective_policy(settings, cfg, switches, principal, hints, surface, *, body_location: str | None = None
                     ) -> EffectivePolicy:
    role = getattr(settings, 'role', 'standalone')
    delegated = bool(getattr(hints, 'delegated', False)) and bool(getattr(switches, 'delegation', False))
    cloud = cloud_policy_for(settings, cfg, switches, delegated=delegated)
    hinted = getattr(hints, 'location', None)
    if getattr(switches, 'parent_headers_authoritative', False) and hinted == 'local_only':
        location = 'local_only'
    elif hinted:
        location = hinted
    elif body_location:
        location = body_location
    elif principal is not None and principal.default_location:
        location = principal.default_location
    else:
        location = cfg.policy.default_location if cfg is not None else 'auto'
    if location not in LOCATIONS:
        raise BrokerError(f'Unknown location {location!r}', 'invalid_request')
    if principal is not None and location not in principal.constraints.locations:
        raise BrokerError(f'Location {location} is not allowed for this key', 'location_unsatisfiable', origin='policy')
    return EffectivePolicy(cloud_policy=cloud, location=location, role=role, delegated=delegated)


@dataclass(frozen=True)
class FilterContext:
    """What filter_chain needs beyond the candidates. named: the caller named the target (qualified ref, published alias,
    compat suffix or X-Titanos-Target) rather than a virtual model. delegable: the narrower explicit target a delegated
    managed request needs (roles condition 5: a qualified ref or X-Titanos-Target naming one target)."""
    cfg: object
    switches: object
    catalog: object
    named: bool = False
    hints: object = None
    delegable: bool = False


def _modalities(body) -> set[str]:
    needed = {'text'}
    if getattr(body, 'has_images', False):
        needed.add('image')
    if getattr(body, 'has_audio', False):
        needed.add('audio')
    return needed


def filter_chain(candidates, eff: EffectivePolicy, principal, surface, body, *, breakers, handles, context: FilterContext
                 ) -> tuple[list, list[Exclusion]]:
    """(survivors, exclusions). Survivors keep the candidate order except for prefer_local's stable sort. Stops at the
    stage that removes the last candidate (that stage's exclusions are last in the list)."""
    cfg, switches, catalog = context.cfg, context.switches, context.catalog
    hints = context.hints
    survivors = list(candidates)
    exclusions: list[Exclusion] = []

    def stage(keep, reason, detail=None):
        nonlocal survivors
        kept = []
        for target in survivors:
            verdict = keep(target)
            if verdict is True:
                kept.append(target)
            else:
                exclusions.append(Exclusion(target.ref, reason, verdict if isinstance(verdict, str) else detail))
        survivors = kept
        return bool(survivors)

    # 1. principal visibility
    scope_ok = allows(principal, surface.scope)

    def visible(target):
        if scope_ok and can_use_target(principal, target):
            return True
        if (scope_ok and getattr(principal, 'kind', None) == 'anonymous_loopback' and target.klass in REMOTE
                and eff.role == 'standalone' and not allows(principal, 'cloud')):
            return 'keyless_cloud_off'
        return 'invisible'
    if not stage(visible, 'scope'):
        return survivors, exclusions
    # 2. handle ownership (previous_response_id)
    previous = getattr(body, 'previous_response_id', None)
    if previous and any(t.klass != 'local' for t in survivors):
        binding = None
        if handles is not None:
            try:
                binding = handles.owner('responses', previous)
            except ValueError:
                binding = None
            if binding is not None:
                binding = handles.owned(principal, 'responses', previous)  # a foreign handle: 404 at once
        if not stage(lambda t: t.klass == 'local' or (binding is not None and t.upstream_id == binding.upstream_id),
                     'policy', 'handle'):
            return survivors, exclusions
    # 3. role switches and the 1-hop rule
    delegated = eff.delegated

    def role_ok(target):
        if target.klass == 'local':
            return True
        if getattr(hints, 'hop', False):
            return 'hop'
        if eff.role == 'managed':
            if not delegated:
                return 'delegation_required' if getattr(switches, 'delegation', False) else 'role_forbids'
            return True if context.delegable else 'delegation_explicit_only'
        if target.klass == 'router' and not getattr(switches, 'router_upstreams', True):
            return 'role_forbids'
        if not getattr(switches, 'cloud_egress', True):
            return 'role_forbids'
        return True
    if not stage(role_ok, 'role'):
        if exclusions[-1].detail == 'hop':
            exclusions[-1] = Exclusion(exclusions[-1].ref, 'location', 'hop')
        return survivors, exclusions
    # 4. location
    location = eff.location
    if location == 'local_only':
        if not stage(lambda t: t.klass == 'local', 'location', location):
            return survivors, exclusions
    elif location == 'cloud_only':
        if not stage(lambda t: t.klass in REMOTE, 'location', location):
            return survivors, exclusions
    elif location == 'prefer_local' and any(t.klass == 'local' for t in survivors):
        survivors.sort(key=lambda t: 0 if t.klass == 'local' else 1)
        if eff.cloud_policy != 'fallback':
            stage(lambda t: t.klass == 'local', 'location', location)
    # 5. cloud policy
    if eff.cloud_policy == 'off':
        if not stage(lambda t: t.klass == 'local', 'policy', 'cloud_disabled'):
            return survivors, exclusions
    elif eff.cloud_policy == 'explicit' and not context.named and survivors:
        first = survivors[0].klass
        if not stage(lambda t: t.klass == first, 'policy', 'explicit'):
            return survivors, exclusions
    # 6. capability
    capability = surface.capability

    def capable(target):
        if target.klass == 'local':
            return True
        upstream = cfg.upstreams.get(target.upstream_id)
        adapter = catalog.adapters.get(target.upstream_id) if catalog is not None else None
        if upstream is None or adapter is None:
            return 'no adapter'
        profile = PROFILES[upstream.profile]
        try:
            supported = adapter.supports(profile, surface)
        except Exception:
            supported = False
        if not supported:
            return f'{surface.name} unsupported'
        offered = set(target.capabilities)
        if 'chat' in offered:
            offered.add('responses')
        if capability not in offered and capability not in ('asset', 'catalog'):
            return f'{capability} not offered'
        # a router route that declares tool_calling = false or vision = false on its live catalog (switchyard): refused
        # here, with zero egress, instead of by the router
        meta = catalog.model_meta(target.upstream_id, target.upstream_model) \
            if catalog is not None and hasattr(catalog, 'model_meta') else {}
        if meta.get('tool_calling') is False and getattr(body, 'has_tools', False):
            return 'tools not offered by this route'
        if meta.get('vision') is False and getattr(body, 'has_images', False):
            return 'vision not offered by this route'
        return True
    if not stage(capable, 'capability'):
        return survivors, exclusions
    # 7. data policy
    needed = _modalities(body)

    def data_ok(target):
        if target.klass == 'local':
            return True
        upstream = cfg.upstreams[target.upstream_id]
        missing = needed - set(upstream.effective_modalities())
        return True if not missing else 'modalities ' + ','.join(sorted(missing))
    if not stage(data_ok, 'data_policy'):
        return survivors, exclusions
    # 8. readiness

    def ready(target):
        if target.klass == 'local':
            return True
        problem = catalog.readiness(target.upstream_id)
        return True if problem is None else problem

    kept = []
    for target in survivors:
        verdict = ready(target)
        if verdict is True:
            kept.append(target)
        else:
            exclusions.append(Exclusion(target.ref, verdict[0], verdict[1]))
    survivors = kept
    if not survivors:
        return survivors, exclusions
    # 9. breakers and 429 cooldowns

    def closed(target):
        if target.klass == 'local' or breakers is None:
            return True
        upstream = cfg.upstreams[target.upstream_id]
        from .health import breaker_for
        verdict = breaker_for(breakers, target, upstream).allow(consume=False)
        if verdict:
            return True
        cooling = 'cooldown:' if getattr(verdict, 'reason', '') == 'cooldown' else ''
        return f'{cooling}retry_after={verdict.retry_after:.3f}'
    stage(closed, 'breaker_open')
    return survivors, exclusions


def refusal(exclusions: list[Exclusion], *, named: bool) -> BrokerError:
    """The error for a chain the filter emptied: the reason of the stage that removed the last candidate."""
    if not exclusions:
        return BrokerError('Model not found', 'model_not_found')
    last = exclusions[-1]
    reason, detail = last.reason, last.detail or ''
    if reason == 'scope':
        if detail == 'keyless_cloud_off':
            return BrokerError('Keyless cloud use is off: set policy.anonymous_cloud to true (Studio providers panel, '
                               'titanos broker connect, or PUT /api/broker/policy) or send a key holding cloud',
                               'cloud_disabled', origin='policy')
        return BrokerError('Model not found', 'model_not_found')
    if reason == 'policy' and detail == 'handle':
        return BrokerError('Response not found (a cloud response is stored and can be continued only when it was created '
                           'by a key holding cloud:store on an upstream with data_policy.allow_store)',
                           'response_not_found', origin='policy')
    if reason == 'location':
        return BrokerError('No target satisfies the requested location'
                           + (' (requests from another broker hop stay local)' if detail == 'hop' else ''),
                           'location_unsatisfiable', origin='policy')
    if reason == 'role':
        if detail == 'delegation_required':
            return BrokerError('Cloud use in managed role needs parent delegation', 'delegation_required',
                               origin='policy')
        if detail == 'delegation_explicit_only':
            return BrokerError('Delegated cloud use needs an explicit target', 'delegation_required', origin='policy')
        return BrokerError('This role does not allow cloud targets', 'role_forbids', origin='policy')
    if reason == 'policy':
        return BrokerError('Cloud targets are disabled by policy', 'cloud_disabled', origin='policy')
    if reason == 'capability':
        return BrokerError(f'The target cannot serve this request ({detail})', 'broker_capability_unavailable',
                           origin='policy')
    if reason == 'data_policy':
        return BrokerError(f'The upstream data policy does not allow this request ({detail})', 'data_policy_denied',
                           origin='policy')
    if reason == 'disabled' and detail == 'slot':
        return BrokerError('The upstream slot has no wire contract', 'upstream_unconfigured')
    if reason == 'unvalidated' and detail == 'unconfigured':
        return BrokerError('The upstream needs a credential and none is configured', 'upstream_unconfigured')
    if reason in ('disabled', 'unvalidated'):
        return BrokerError('Model not found', 'model_not_found')
    if reason == 'breaker_open':
        waits, cooling = [], []
        for item in exclusions:
            detail = item.detail or ''
            if item.reason != 'breaker_open':
                continue
            cooling.append(detail.startswith('cooldown:'))
            detail = detail.removeprefix('cooldown:')
            if detail.startswith('retry_after='):
                try:
                    waits.append(float(detail.split('=', 1)[1]))
                except ValueError:
                    pass
        if cooling and all(cooling):
            # every candidate is waiting out the provider's own 429: answer as the provider did (429 + Retry-After)
            return BrokerError('The upstream rate limit for this model is in effect; retry after the indicated time',
                               'upstream_rate_limited', retry_after=min(waits) if waits else 1.0, origin='upstream')
        return BrokerError('Every upstream for this request is temporarily unavailable', 'breaker_open',
                           retry_after=min(waits) if waits else 1.0)
    return BrokerError('Model not found', 'model_not_found')


__all__ = ['FilterContext', 'LOCATION_ORDER', 'cloud_policy_for', 'effective_policy', 'filter_chain', 'refusal']
