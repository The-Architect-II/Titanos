"""Unified target catalog: local registry names, published aliases, virtual models and upstream catalogs.

Name resolution (model_naming; the first match wins, no vendor-prefix guessing):
  local registry name -> virtual model -> published alias -> qualified ref '<model>@<upstream>' -> optional HiveMind
  compat suffix (longest first, expanded only into a qualified ref) -> on a vendor-protocol surface only, a bare provider
  model id on the single matching upstream that opts in (resolve_vendor; ElevenLabs) -> nothing (the router answers 404
  model_not_found).
A qualified ref names a target only when the upstream exists and lists the model (upstream.models, an Azure deployment),
allows any model, or (catalog=live) has it in its last successful live catalog. Whether the upstream is enabled and its
credential validated is decided later by the policy filter (the router still answers 404 with zero egress).

Live catalogs are fetched through GrantIssuer purpose 'catalog' grants, only for enabled upstreams with a usable credential
and only when role switches.live_catalogs is on, with a TTL of broker_catalog_ttl; a failed fetch keeps the static list
(and the previous live list) and records the error. Resolution never fetches: it reads the cache. The live cache is in
memory only, so a fresh process knows no live model until a fetch: live_pending() and default_pending() name the
live-catalog upstreams whose stale (never fetched, or past the TTL) catalog could make a name resolve for a principal, and
the async callers (BrokerDispatch, the catalog and stage routes) fetch exactly those on demand through
Broker.ensure_catalogs (single-flight; a wait bounded by the request deadline and facade.CATALOG_WAIT_SECONDS; a failing
upstream retried at most once a minute after its last attempt ended) before resolving, so a well-formed qualified ref or
bare provider id never answers 404 only because no client listed models first. A request that cannot be placed in the
cloud (a local model, local_only, another broker hop) names nothing and never waits (BrokerDispatch._stage).

entries() lists the broker's own entries (cloud and virtual) for a principal; the catalog handlers merge them after the
unchanged local entries (engine.catalog() and native.catalog()), so /v1/models is byte-identical when nothing is published.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import BrokerConfig, VirtualModel
from .errors import BrokerError
from .naming import expand_compat_suffix, format_ref, parse_ref
from .profiles import PROFILES, model_capabilities
from .scopes import allows, can_use_target
from .surfaces import SURFACES
from .types import Target

# Local registry task -> capabilities a local target offers (local strict validation still has the last word).
LOCAL_CAPABILITIES: Mapping[str, frozenset[str]] = {
    'chat': frozenset({'chat', 'tools', 'responses'}),
    'vision': frozenset({'chat', 'vision', 'responses'}),
    'embeddings': frozenset({'embeddings'}),
    'speech': frozenset({'speech'}),
    'transcription': frozenset({'transcription'}),
    'image': frozenset({'images'}),
    'translation': frozenset({'translation'}),
}
INTERCEPT_SURFACES = tuple(s for s in SURFACES.values() if s.kind == 'intercept')
# the inbound protocol each adapter speaks natively (resolve_vendor and vendor_default)
_PROTOCOL_BY_ADAPTER = {'openai_compat': 'openai', 'anthropic': 'anthropic', 'gemini': 'gemini',
                        'elevenlabs': 'elevenlabs', 'ollama_native': 'ollama'}


@dataclass(frozen=True)
class LocalModel:
    task: str
    kind: str = 'engine'  # engine | native


@dataclass(frozen=True)
class Resolution:
    """How a name resolved. kind: local | virtual | alias | qualified | compat | vendor (resolve_vendor)."""
    kind: str
    target: Target | None = None
    virtual: VirtualModel | None = None
    virtual_name: str | None = None
    alias: str | None = None
    inject: Mapping = field(default_factory=dict)


@dataclass
class LiveEntry:
    models: tuple[str, ...] = ()
    fetched_at: str | None = None
    fetched_mono: float | None = None
    error: str | None = None
    source: str = 'static'
    # per-model capability metadata a router publishes on its catalog ({model: {tool_calling, vision, context_length}})
    meta: dict = field(default_factory=dict)


def local_models_from(config) -> dict[str, LocalModel]:
    """LocalModel per registry name of a titanos Config (llama-cpp backends run in the NativePool)."""
    out = {}
    for name, spec in (getattr(config, 'models', None) or {}).items():
        out[name] = LocalModel(task=getattr(spec, 'task', 'chat'),
                               kind='native' if getattr(spec, 'backend', '') == 'llama-cpp' else 'engine')
    return out


def profile_capabilities(adapter, profile, upstream=None) -> frozenset[str]:
    """Capabilities an upstream offers through its adapter (native or translated surfaces) plus declared features."""
    caps = set()
    if adapter is not None:
        for surface in INTERCEPT_SURFACES:
            try:
                supported = adapter.supports(profile, surface)
            except Exception:
                supported = False
            if supported:
                caps.add(surface.capability)
    if 'chat' in caps:
        caps.update(f for f in ('tools', 'vision') if f in profile.features)
    return frozenset(caps)


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class TargetCatalog:
    """The catalog of one BrokerSnapshot. live_cache is shared across snapshots (the Broker owns it)."""

    def __init__(self, config: BrokerConfig, *, settings=None, switches=None, local_models: Callable[[], Mapping] = dict,
                 adapters: Mapping | None = None, vault=None, breakers=None, issuer=None, egress=None,
                 live_cache: dict | None = None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time):
        self.config = config
        self.settings = settings
        self.switches = switches
        self._local_models = local_models
        self.adapters = dict(adapters or {})
        self.vault = vault
        self.breakers = breakers
        self.issuer = issuer
        self.egress = egress
        self.live = live_cache if live_cache is not None else {}
        self._clock = clock
        self._wall = wall
        self.ttl = float(getattr(settings, 'broker_catalog_ttl', 3600.0))
        self._caps: dict[str, frozenset[str]] = {}

    # -- local -----------------------------------------------------------------------------------------------------------
    def local_models(self) -> Mapping[str, LocalModel]:
        found = self._local_models() or {}
        return {name: (value if isinstance(value, LocalModel) else LocalModel(task=str(value)))
                for name, value in found.items()}

    def local_target(self, name: str) -> Target | None:
        model = self.local_models().get(name)
        if model is None:
            return None
        return Target(ref=name, klass='local', local_name=name, capabilities=LOCAL_CAPABILITIES.get(model.task,
                      frozenset({model.task})), adapter='local')

    def local_kind(self, name: str) -> str:
        model = self.local_models().get(name)
        return model.kind if model is not None else 'engine'

    # -- upstreams -------------------------------------------------------------------------------------------------------
    def capabilities(self, upstream_id: str) -> frozenset[str]:
        if upstream_id not in self._caps:
            upstream = self.config.upstreams[upstream_id]
            self._caps[upstream_id] = profile_capabilities(self.adapters.get(upstream_id), PROFILES[upstream.profile],
                                                           upstream)
        return self._caps[upstream_id]

    def model_listed(self, upstream_id: str, model: str) -> bool:
        upstream = self.config.upstreams.get(upstream_id)
        if upstream is None or not model:
            return False
        if model in upstream.models or upstream.effective_allow_any_model():
            return True
        if upstream.azure is not None and model in upstream.azure.deployments:
            return True
        if upstream.catalog == 'live':
            entry = self.live.get(upstream_id)
            return entry is not None and model in entry.models
        return False

    def qualified_target(self, model: str, upstream_id: str, *, capabilities=None) -> Target | None:
        upstream = self.config.upstreams.get(upstream_id)
        if upstream is None:
            return None
        profile = PROFILES[upstream.profile]
        try:
            ref = format_ref(model, upstream_id)
        except ValueError:
            return None
        caps = self.capabilities(upstream_id) if capabilities is None else frozenset(capabilities)
        meta = self.model_meta(upstream_id, model) if capabilities is None else {}
        narrow = getattr(self.adapters.get(upstream_id), 'model_capabilities', None)
        if capabilities is None and callable(narrow):  # per-model families (Gemini: TTS, image, embedding models)
            caps = frozenset(narrow(model, caps, meta))
        if meta and 'chat' in caps:  # a router route that declares tool_calling / vision (switchyard /v1/models)
            narrowed = set(caps)
            for cap, key in (('tools', 'tool_calling'), ('vision', 'vision')):
                if meta.get(key) is False:
                    narrowed.discard(cap)
                elif meta.get(key) is True:
                    narrowed.add(cap)
            caps = frozenset(narrowed)
        return Target(ref=ref, klass='router' if profile.kind == 'router' else 'cloud', upstream_id=upstream_id,
                      upstream_model=model, capabilities=caps, adapter=profile.adapter, profile_id=profile.id)

    def credential_usable(self, upstream) -> bool:
        return self._credential_problem(upstream) is None

    def _credential_problem(self, upstream) -> str | None:
        """None when the upstream's credential can serve from THIS process; 'unvalidated' when it is missing, not
        validated or bound to another origin; 'credential_unavailable' when it is an env: reference whose variable this
        process does not hold (the CLI validates from its own environment, so a `titanos serve` started without the
        variable would otherwise report ready and answer every call 404)."""
        if upstream.credential is None or self.vault is None:
            return 'unvalidated'
        try:
            record = self.vault.record(upstream.credential)
        except BrokerError:
            return 'unvalidated'
        if not record.usable:
            return 'unvalidated'
        try:
            origin = upstream.origin(allow_http_lan=self.config.policy.allow_http_lan)
        except ValueError:
            return 'unvalidated'
        if record.origin != origin:
            return 'unvalidated'
        available = getattr(self.vault, 'reference_available', None)
        if callable(available) and not available(record):
            return 'credential_unavailable'
        return None

    def readiness(self, upstream_id: str) -> tuple[str, str] | None:
        """None when the upstream may serve; otherwise (exclusion reason, detail): ('disabled', 'disabled' | 'slot'),
        ('unvalidated', 'unvalidated' | 'unconfigured' | 'credential_unavailable'). credential_unavailable: an env:
        credential whose variable the server process does not hold."""
        upstream = self.config.upstreams.get(upstream_id)
        if upstream is None:
            return 'disabled', 'unknown'
        profile = PROFILES[upstream.profile]
        if profile.kind == 'slot' or profile.adapter == 'none':
            return 'disabled', 'slot'
        if not upstream.enabled:
            return 'disabled', 'disabled'
        if upstream.credential is None:
            if upstream.effective_auth() == 'none' or profile.auth_optional:
                return None
            return 'unvalidated', 'unconfigured'
        problem = self._credential_problem(upstream)
        if problem is not None:
            return 'unvalidated', problem
        return None

    # -- names -----------------------------------------------------------------------------------------------------------
    def resolve_name(self, name: str, *, allow_virtual: bool = True) -> Resolution | None:
        """The Resolution of a name, or None (404). parse_ref raises model_required / invalid_request."""
        ref = parse_ref(name)
        cfg = self.config
        if ref.kind == 'plain':
            local = self.local_target(name)
            if local is not None:
                return Resolution('local', target=local)
            if allow_virtual and name in cfg.virtual_models:
                return Resolution('virtual', virtual=cfg.virtual_models[name], virtual_name=name)
            alias = cfg.models.get(name)
            if alias is not None:
                target_ref = parse_ref(alias.target)
                caps = set(alias.capabilities)
                if 'chat' in caps:
                    caps.add('responses')
                target = self.qualified_target(target_ref.model, target_ref.upstream, capabilities=caps)
                if target is None:
                    return None
                return Resolution('alias', target=target, alias=name, inject=dict(alias.inject))
            expanded = expand_compat_suffix(name, cfg.compat)
            if expanded is not None:
                found = self.resolve_name(expanded, allow_virtual=False)
                if found is not None and found.kind == 'qualified':
                    return Resolution('compat', target=found.target)
            return None
        if not self.model_listed(ref.upstream, ref.model):
            return None
        target = self.qualified_target(ref.model, ref.upstream)
        return Resolution('qualified', target=target) if target is not None else None

    def resolve_vendor(self, name: str, protocol: str, principal) -> Resolution | None:
        """A bare provider model id on a vendor-protocol surface (ElevenLabs SDKs send eleven_multilingual_v2, never a
        qualified ref): the qualified ref on the ONE enabled, ready upstream whose profile speaks that protocol, opts in
        (extras bare_model_ids), lists the model and is visible to the principal. None for zero or several candidates
        (the router then answers 404 model_not_found as before). Resolution kind 'vendor'."""
        from .naming import is_local_name
        if not isinstance(name, str) or '@' in name or not is_local_name(name):
            return None
        found = []
        for upstream_id, upstream in sorted(self.config.upstreams.items()):
            profile = PROFILES[upstream.profile]
            if not profile.extras.get('bare_model_ids') or _PROTOCOL_BY_ADAPTER.get(profile.adapter) != protocol:
                continue
            if not upstream.enabled or self.readiness(upstream_id) is not None or not self.model_listed(upstream_id, name):
                continue
            target = self.qualified_target(name, upstream_id)
            if target is not None and can_use_target(principal, target):
                found.append(target)
        return Resolution('vendor', target=found[0]) if len(found) == 1 else None

    def vendor_default(self, protocol: str, capability: str, principal) -> Target | None:
        """The provider's default model (extras default_models[capability]) on the ONE enabled, ready upstream of that
        protocol the principal may use (for a vendor request that names no model), or None."""
        found = []
        for upstream_id, upstream in sorted(self.config.upstreams.items()):
            profile = PROFILES[upstream.profile]
            model = (profile.extras.get('default_models') or {}).get(capability)
            if not model or _PROTOCOL_BY_ADAPTER.get(profile.adapter) != protocol:
                continue
            if not upstream.enabled or self.readiness(upstream_id) is not None or not self.model_listed(upstream_id,
                                                                                                         model):
                continue
            target = self.qualified_target(model, upstream_id)
            if target is not None and can_use_target(principal, target):
                found.append(target)
        return found[0] if len(found) == 1 else None

    def lookup(self, name: str) -> Target | VirtualModel | None:
        try:
            found = self.resolve_name(name)
        except BrokerError:
            return None
        if found is None:
            return None
        return found.virtual if found.kind == 'virtual' else found.target

    # -- on-demand live catalogs -----------------------------------------------------------------------------------------
    def _pending(self, upstream_id: str, model: str, principal) -> bool:
        """Whether fetching upstream_id's live catalog could make model@upstream_id resolve for principal: an enabled,
        ready live-catalog upstream (live catalogs on) whose cache is stale and does not list the model yet, and a target
        the principal may see (R21: an upstream the principal cannot see is never contacted on its behalf)."""
        upstream = self.config.upstreams.get(upstream_id)
        if upstream is None or upstream.catalog != 'live' or not upstream.enabled or not model or principal is None:
            return False
        if self.switches is not None and not self.switches.live_catalogs:
            return False
        if not self.stale(upstream_id) or self.model_listed(upstream_id, model) or self.readiness(upstream_id) is not None:
            return False
        target = self.qualified_target(model, upstream_id)
        return target is not None and can_use_target(principal, target)

    def live_pending(self, name, protocol: str | None, principal) -> list[str]:
        """The upstreams whose stale live catalog could make `name` resolve (see the module docstring): the upstream of a
        qualified ref or compat suffix, the qualified targets of a virtual model, and for a bare provider id the upstreams
        of `protocol` whose profile opts in to bare_model_ids (resolve_vendor). Local names and published aliases resolve
        without any catalog, so they name nothing. Never raises for a malformed name (the router answers it)."""
        from .naming import is_local_name
        out: list[str] = []

        def add(upstream_id: str, model: str) -> None:
            if upstream_id not in out and self._pending(upstream_id, model, principal):
                out.append(upstream_id)

        def qualified(text) -> None:
            try:
                ref = parse_ref(text)
            except (BrokerError, ValueError):
                return
            if ref.kind != 'plain':
                add(ref.upstream, ref.model)

        if not isinstance(name, str) or not name:
            return out
        try:
            ref = parse_ref(name)
        except (BrokerError, ValueError):
            return out
        cfg = self.config
        if ref.kind != 'plain':
            add(ref.upstream, ref.model)
            return out
        if self.local_target(name) is not None or name in cfg.models:
            return out
        virtual = cfg.virtual_models.get(name)
        if virtual is not None:
            for item in virtual.targets:
                qualified(item.ref)
            return out
        try:
            expanded = expand_compat_suffix(name, cfg.compat)
        except (TypeError, ValueError):
            expanded = None
        if expanded is not None:
            qualified(expanded)
        if protocol and is_local_name(name):
            for upstream_id, upstream in sorted(cfg.upstreams.items()):
                profile = PROFILES[upstream.profile]
                if profile.extras.get('bare_model_ids') and _PROTOCOL_BY_ADAPTER.get(profile.adapter) == protocol:
                    add(upstream_id, name)
        return out

    def default_pending(self, protocol: str, capability: str, principal) -> list[str]:
        """The upstreams of `protocol` whose stale live catalog could list their profile's default model for capability
        (vendor_default: a request that names no model)."""
        out = []
        for upstream_id, upstream in sorted(self.config.upstreams.items()):
            profile = PROFILES[upstream.profile]
            model = (profile.extras.get('default_models') or {}).get(capability)
            if model and _PROTOCOL_BY_ADAPTER.get(profile.adapter) == protocol and self._pending(upstream_id, model,
                                                                                                principal):
                out.append(upstream_id)
        return out

    # -- listings --------------------------------------------------------------------------------------------------------
    def _status(self, upstream_id: str, target: Target | None = None) -> str:
        upstream = self.config.upstreams[upstream_id]
        if self.breakers is not None:
            from .health import breaker_for
            breaker = breaker_for(self.breakers, target, upstream) if target is not None else \
                self.breakers.get(upstream_id, upstream.credential)
            if not breaker.allow(consume=False):
                return 'breaker_open'
        return 'available'

    def model_meta(self, upstream_id: str, model: str | None) -> dict:
        """The live catalog's capability metadata for one upstream model ({} when the upstream publishes none)."""
        entry = self.live.get(upstream_id)
        if entry is None or not model or (self.switches is not None and not self.switches.live_catalogs):
            return {}
        return dict(entry.meta.get(model) or {})

    def _entry(self, name: str, target: Target, family: str) -> dict:
        info = {'class': target.klass, 'upstream': target.upstream_id, 'capabilities': sorted(target.capabilities),
                'status': self._status(target.upstream_id, target)}
        context = self.model_meta(target.upstream_id, target.upstream_model).get('context_length')
        if context:
            info['context_length'] = context
        row = PROFILES.get(target.profile_id) if target.profile_id else None
        hint = model_capabilities(row, target.upstream_model) if row is not None else None
        if hint is None and callable(getattr(self.adapters.get(target.upstream_id), 'model_capabilities', None)):
            hint = tuple(sorted(target.capabilities))  # adapters that narrow per model (Gemini, ElevenLabs)
        if hint is not None:  # what this model family serves, beside the upstream's capabilities
            info['model_capabilities'] = list(hint)
        if family == 'ollama':
            return {'name': name, 'model': name, 'modified_at': '1970-01-01T00:00:00Z', 'size': 0, 'digest': '',
                    'details': {'format': 'remote', 'family': target.upstream_id}, 'remote': True, 'titanos': info}
        return {'id': name, 'object': 'model', 'created': 0, 'owned_by': target.upstream_id, 'remote': True,
                'titanos': info}

    def upstream_models(self, upstream_id: str) -> list[str]:
        upstream = self.config.upstreams[upstream_id]
        models = list(upstream.models)
        if upstream.azure is not None:
            models.extend(m for m in upstream.azure.deployments if m not in models)
        if upstream.catalog == 'live' and (self.switches is None or self.switches.live_catalogs):
            entry = self.live.get(upstream_id)
            if entry is not None:
                models.extend(m for m in entry.models if m not in models)
        return models

    def entries(self, principal, *, family: str = 'openai', loaded_only: bool = False) -> list[dict]:
        """The broker entries a principal may use: published aliases and qualified refs of enabled, validated upstreams
        (owned_by the upstream id, remote: true) and exposed virtual models (owned_by titanos-virtual; their targets
        only for broker:read). Nothing for loaded_only (cloud targets are never 'loaded')."""
        if loaded_only or principal is None:
            return []
        out, seen = [], set()
        cfg = self.config
        for name, alias in sorted(cfg.models.items()):
            found = self.resolve_name(name)
            if found is None or found.target is None:
                continue
            target = found.target
            if self.readiness(target.upstream_id) is not None or not can_use_target(principal, target):
                continue
            out.append(self._entry(name, target, family))
            seen.add(name)
        for upstream_id in sorted(cfg.upstreams):
            if self.readiness(upstream_id) is not None:
                continue
            for model in self.upstream_models(upstream_id):
                target = self.qualified_target(model, upstream_id)
                if target is None or target.ref in seen or not can_use_target(principal, target):
                    continue
                out.append(self._entry(target.ref, target, family))
                seen.add(target.ref)
        if family == 'openai':
            for name, virtual in sorted(cfg.virtual_models.items()):
                if not virtual.expose:
                    continue
                visible = [t for t in (self._virtual_target(v.ref) for v in virtual.targets) if t is not None
                           and can_use_target(principal, t)
                           and (t.klass == 'local' or self.readiness(t.upstream_id) is None)]
                if not visible:
                    continue
                info = {'class': 'virtual', 'capability': virtual.capability, 'strategy': virtual.strategy}
                if allows(principal, 'broker:read'):
                    info['targets'] = [t.ref for t in visible]
                out.append({'id': name, 'object': 'model', 'created': 0, 'owned_by': 'titanos-virtual',
                            'titanos': info})
        return out

    def _virtual_target(self, ref: str) -> Target | None:
        try:
            found = self.resolve_name(ref, allow_virtual=False)
        except BrokerError:
            return None
        return found.target if found is not None else None

    def entry(self, name: str, principal) -> dict | None:
        """One entry for GET /v1/models/{name} (a published alias, qualified ref or exposed virtual model)."""
        for item in self.entries(principal):
            if item.get('id') == name:
                return item
        return None

    # -- live catalogs ---------------------------------------------------------------------------------------------------
    def stale(self, upstream_id: str) -> bool:
        entry = self.live.get(upstream_id)
        return entry is None or entry.fetched_mono is None or self._clock() - entry.fetched_mono >= self.ttl

    async def refresh_upstream(self, upstream_id: str) -> dict:
        """Fetch one upstream's live catalog (purpose catalog). Returns its status entry; never raises for upstream
        trouble (the static list stays in force and the error is recorded)."""
        upstream = self.config.upstreams.get(upstream_id)
        if upstream is None:
            raise BrokerError(f'Upstream {upstream_id} does not exist', 'not_found')
        entry = self.live.setdefault(upstream_id, LiveEntry())
        if self.switches is not None and not self.switches.live_catalogs:
            entry.error = 'live_catalogs_off'
            return self.status_of(upstream_id)
        not_ready = self.readiness(upstream_id)
        if not_ready is not None:
            entry.error = not_ready[1]
            return self.status_of(upstream_id)
        adapter = self.adapters.get(upstream_id)
        if adapter is None or self.issuer is None or self.egress is None:
            entry.error = 'upstream_unconfigured'
            return self.status_of(upstream_id)
        try:
            deadline = self._clock() + float(getattr(self.settings, 'broker_connect_timeout', 10.0)) + float(
                getattr(self.settings, 'broker_first_byte_timeout', 120.0))
            grant = self.issuer.issue('catalog', upstream_id=upstream_id, decision_id=uuid.uuid4().hex,
                                      request_id=uuid.uuid4().hex, deadline=deadline)
            with_meta = getattr(adapter, 'list_models_meta', None)
            if callable(with_meta):  # a router that publishes per-model capabilities (switchyard)
                models, meta = await with_meta(grant, self.egress)
            else:
                models, meta = await adapter.list_models(grant, self.egress), {}
        except BrokerError as exc:
            entry.error = exc.code
            return self.status_of(upstream_id)
        except Exception as exc:  # transport trouble: keep the static list
            entry.error = type(exc).__name__
            return self.status_of(upstream_id)
        entry.models = tuple(dict.fromkeys(m for m in models if isinstance(m, str) and m))
        entry.meta = {m: dict(v) for m, v in (meta or {}).items() if m in entry.models and isinstance(v, dict)}
        entry.fetched_mono = self._clock()
        entry.fetched_at = _iso(self._wall())
        entry.error = None
        entry.source = 'live'
        return self.status_of(upstream_id)

    async def refresh_stale(self) -> list[str]:
        """Refresh every live-catalog upstream whose cache is older than the TTL; returns the refreshed ids."""
        done = []
        for upstream_id, upstream in sorted(self.config.upstreams.items()):
            if upstream.catalog == 'live' and upstream.enabled and self.stale(upstream_id):
                await self.refresh_upstream(upstream_id)
                done.append(upstream_id)
        return done

    def status_of(self, upstream_id: str) -> dict:
        entry = self.live.get(upstream_id)
        if entry is None or entry.source != 'live':
            return {'source': 'static', 'fetched_at': None, 'error': entry.error if entry else None}
        return {'source': 'live', 'fetched_at': entry.fetched_at, 'error': entry.error}

    def status(self) -> dict:
        return {upstream_id: self.status_of(upstream_id) for upstream_id in sorted(self.config.upstreams)}


__all__ = ['LOCAL_CAPABILITIES', 'LiveEntry', 'LocalModel', 'Resolution', 'TargetCatalog', 'local_models_from',
           'profile_capabilities']
