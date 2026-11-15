"""The Broker facade: everything server.py, api_plus.py, the CLI and MCP need from the broker, built lazily.

install_broker(app, ...) only records the Broker on app.state: no vault resolution, no network, no file or directory
creation. Every component (key store, vault, audit log, decision log, usage ledger, ownership and handle stores,
authenticator, egress client, limiter, breakers) is constructed on first use, and broker.json is read on first use of the
snapshot. server.py adds BrokerDispatch (app.add_middleware(BrokerDispatch, broker=broker)) on the line before Guard.

BrokerSnapshot(config, catalog, router, prices, switches, adapters, issuer, advisors) is immutable and swapped RCU-style
by reload(); in-flight requests keep the snapshot they started with. reload() keeps the role (it is fixed at serve start)
and refuses with 409 role_forbids any change that would enable egress in managed role without delegation.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import BrokerConfig, broker_dir, broker_path, check_name_conflicts, read_broker_config
from .errors import BrokerError
from .profiles import PROFILES
from .role import switches_for
from .scopes import allows, anonymous_scopes
from .policy import cloud_policy_for
from .types import Principal

log = logging.getLogger('titanos.broker')
TTFB_BUCKETS = (50, 100, 250, 500, 1000, 2500, 5000, 10000)
# the longest a request waits for an on-demand live catalog fetch (Broker.ensure_catalogs), whatever its own deadline: a
# provider that accepts the connection and never answers holds a caller this long at most (the fetch itself keeps
# running, bounded by the connect and first-byte timeouts, and its answer is cached for the next request)
CATALOG_WAIT_SECONDS = 10.0
# an inference 401 without the provider's key-rejection code asks the upstream's probe endpoint about the credential
# (Broker._auth_suspect): at most one such check per (upstream, credential) in flight, and one per this many seconds
CREDENTIAL_CHECK_SECONDS = 60.0


class Metrics:
    """JSON counters for /api/broker/metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests: Counter = Counter()
        self.fallbacks: Counter = Counter()
        self.egress: Counter = Counter()
        self.ttfb: Counter = Counter()

    def request(self, surface: str, klass: str, outcome: str, ttfb_ms: float | None) -> None:
        with self._lock:
            self.requests[(surface, klass, outcome)] += 1
            if ttfb_ms is not None:
                bucket = next((b for b in TTFB_BUCKETS if ttfb_ms <= b), 'inf')
                self.ttfb[str(bucket)] += 1

    def fallback(self, trigger: str | None) -> None:
        with self._lock:
            self.fallbacks[trigger or 'unknown'] += 1

    def egress_bytes(self, upstream_id: str, count: int) -> None:
        with self._lock:
            self.egress[upstream_id] += int(count)

    def to_dict(self, *, breakers=None, limiter=None, authenticator=None) -> dict:
        with self._lock:
            return {'requests_by_surface_class_outcome': [{'surface': s, 'class': c, 'outcome': o, 'count': n}
                                                           for (s, c, o), n in sorted(self.requests.items())],
                    'fallbacks_by_trigger': dict(self.fallbacks),
                    'breaker_transitions': dict(getattr(breakers, 'transitions', {}) or {}),
                    'ttfb_ms_buckets': {'bounds': list(TTFB_BUCKETS), 'counts': dict(self.ttfb)},
                    'egress_bytes_by_upstream': dict(self.egress),
                    'auth_failures': int(getattr(authenticator, 'failures', 0) or 0) if authenticator else 0,
                    'budget_stops': int(getattr(limiter, 'budget_stops', 0) or 0) if limiter else 0}


@dataclass(frozen=True)
class BrokerSnapshot:
    config: BrokerConfig
    catalog: Any
    router: Any
    prices: Any
    switches: Any
    adapters: dict = field(default_factory=dict)
    issuer: Any = None
    advisors: dict = field(default_factory=dict)
    version: int = 0
    load_error: str | None = None


def _call(value):
    return value() if callable(value) else value


class Broker:
    """Lazy broker facade (see the module docstring). Injection points for tests: home, vault, egress, clock."""

    def __init__(self, *, settings, config=None, engine_slot=None, native_getter=None, bootstrap_key: str | None = None,
                 registry_path=None, outputs=None, home=None, vault=None, egress=None,
                 clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self._local_config = config
        self.engine_slot = engine_slot
        self.native_getter = native_getter
        self.bootstrap_key = bootstrap_key or None
        self.registry_path = registry_path
        self.outputs = Path(outputs) if outputs is not None else None
        self._home = Path(home) if home is not None else None
        self._clock = clock
        self._lock = threading.RLock()
        self._snapshot: BrokerSnapshot | None = None
        self._version = 0
        self._components: dict[str, Any] = {}
        if vault is not None:
            self._components['vault'] = vault
        if egress is not None:
            self._components['egress'] = egress
        self.switches = switches_for(settings)
        self.metrics = Metrics()
        self.artifact_writer: Callable[[str], Path] | None = None
        self._live_cache: dict = {}
        self._refresh_tried: dict[str, float] = {}
        self._catalog_tasks: dict[str, asyncio.Task] = {}  # the in-flight live catalog fetch per upstream
        self._voice_cache: dict[tuple, tuple[float, tuple[dict, ...]]] = {}
        self._credential_tasks: dict[tuple, asyncio.Task] = {}  # the in-flight credential check per (upstream, cred)
        self._credential_tried: dict[tuple, float] = {}
        self.credential_checks = 0  # finished credential checks (diagnostics and tests)
        self._loop: asyncio.AbstractEventLoop | None = None  # the serving loop (startup), for worker-thread callers
        self._hop: str | None = None
        self._warned_managed = False
        from .router import Latency
        self.latency = Latency()

    # -- paths -----------------------------------------------------------------------------------------------------------
    @property
    def home(self) -> Path:
        if self._home is not None:
            return self._home
        from ..config import home
        return home()

    @property
    def directory(self) -> Path:
        return self._home / 'broker' if self._home is not None else broker_dir()

    @property
    def config_path(self) -> Path:
        if self._home is not None and not self.settings.broker_config:
            return self._home / 'broker.json'
        return broker_path(self.settings)

    # -- lazy components -------------------------------------------------------------------------------------------------
    def _component(self, name: str, build: Callable[[], Any]):
        with self._lock:
            value = self._components.get(name)
            if value is None:
                value = self._components[name] = build()
            return value

    @property
    def audit(self):
        from .audit import AuditLog
        return self._component('audit', lambda: AuditLog(self.directory / 'audit',
                                                         self.settings.broker_audit_retention_days))

    @property
    def keys(self):
        from .keys import KeyStore
        return self._component('keys', lambda: KeyStore(self.directory / 'keys.json', self.settings, self.audit,
                                                        bootstrap_key_present=bool(self.bootstrap_key)))

    @property
    def vault(self):
        from .vault.store import open_vault
        return self._component('vault', lambda: open_vault(self.settings, self.home))

    @property
    def decisions(self):
        from .decisions import DecisionLog
        return self._component('decisions', lambda: DecisionLog(
            self.settings.broker_decision_log_size,
            self.directory / 'decisions' if self.settings.broker_decision_log_file else None))

    @property
    def usage(self):
        from .usage import UsageLedger
        return self._component('usage', lambda: UsageLedger(self.directory / 'usage',
                                                            retention_days=self.settings.broker_usage_retention_days))

    @property
    def ownership(self):
        from .ownership import OwnershipStore
        return self._component('ownership', lambda: OwnershipStore(self.directory / 'ownership.json',
                                                                   ttl_seconds=self.settings.artifact_ttl_seconds))

    @property
    def handles(self):
        from .handles import HandleStore
        return self._component('handles', lambda: HandleStore(self.directory / 'handles.json'))

    @property
    def authenticator(self):
        from .auth import Authenticator
        return self._component('authenticator', lambda: Authenticator(
            bootstrap_key=self.bootstrap_key, keystore=self.keys, settings=self.settings, audit=_AuthAudit(self),
            role=self.settings.role, anonymous_cloud=lambda: bool(self.snapshot().config.policy.anonymous_cloud)))

    @property
    def egress(self):
        from .egress import EgressClient
        client = self._component('egress', lambda: EgressClient(
            self.settings, self.vault, self.switches,
            allow_http_lan=lambda: bool(self.snapshot().config.policy.allow_http_lan)))
        if getattr(client, 'on_auth_suspect', False) is None:  # an injected EgressClient gets the hook too
            client.on_auth_suspect = self._auth_suspect
        return client

    @property
    def breakers(self):
        from .health import BreakerBoard
        return self._component('breakers', lambda: BreakerBoard(self.settings, clock=self._clock, audit=self.audit))

    @property
    def limiter(self):
        from .limits import Limiter
        return self._component('limiter', lambda: Limiter(self.settings, usage=self.usage,
                                                          prices=self.snapshot().prices, audit=self.audit))

    @property
    def monitor(self):
        from .health import HealthMonitor
        return self._component('monitor', lambda: HealthMonitor(
            self.breakers, switches=self.switches, settings=self.settings, upstreams=self._probe_targets,
            probe=self._probe, clock=self._clock))

    # -- local registry --------------------------------------------------------------------------------------------------
    def local_config(self):
        return _call(self._local_config)

    def set_local_config(self, config) -> None:
        """The registry after a successful reload (server.py)."""
        self._local_config = config

    def local_models(self) -> dict:
        from .catalog import local_models_from
        return local_models_from(self.local_config())

    def local_kind(self, name: str) -> str:
        found = self.local_models().get(name)
        return found.kind if found is not None else 'engine'

    # -- snapshot --------------------------------------------------------------------------------------------------------
    def instance_hop(self) -> str:
        """sha256(instance id)[:16]; the instance id file is created on first need (never at install time)."""
        if self._hop is None:
            from .router import hop_value
            self._hop = hop_value(self.vault.instance_id())
        return self._hop

    def _build(self, cfg: BrokerConfig, *, load_error: str | None = None) -> BrokerSnapshot:
        from .adapters.anthropic import AnthropicAdapter
        from .adapters.elevenlabs import ElevenLabsAdapter
        from .adapters.gemini import GeminiAdapter
        from .adapters.ollama_native import OllamaNativeAdapter
        from .adapters.openai_compat import OpenAICompatAdapter
        from .adapters.split import SplitProtocolAdapter
        from .advisor import build_advisor
        from .catalog import TargetCatalog
        from .pricing import PriceTable
        from .router import BrokerRouter, GrantIssuer
        classes = {'openai_compat': OpenAICompatAdapter, 'anthropic': AnthropicAdapter, 'gemini': GeminiAdapter,
                   'elevenlabs': ElevenLabsAdapter, 'ollama_native': OllamaNativeAdapter}
        adapters = {}
        for upstream_id, upstream in cfg.upstreams.items():
            profile = PROFILES[upstream.profile]
            cls = classes.get(profile.adapter)
            if cls is OpenAICompatAdapter and profile.extras.get('surface_adapters'):
                cls = SplitProtocolAdapter  # a router that also speaks another wire family natively
            if cls is not None:
                adapters[upstream_id] = cls(profile, upstream,
                                            default_max_tokens=self.settings.broker_default_max_output_tokens,
                                            voice_maps=cfg.voice_maps)
        holder: dict = {}
        issuer = GrantIssuer(self.settings, self.switches, _LazyVault(self), config=lambda: holder['config'],
                             hop=self.instance_hop, clock=self._clock)
        holder['config'] = cfg
        prices = PriceTable(cfg.prices)
        egress = _LazyEgress(self)
        catalog = TargetCatalog(cfg, settings=self.settings, switches=self.switches, local_models=self.local_models,
                                adapters=adapters, vault=_LazyVault(self), breakers=_LazyBreakers(self), issuer=issuer,
                                egress=egress, live_cache=self._live_cache, clock=self._clock)
        advisors = {aid: build_advisor(spec, egress, None, issuer, advisor_id=aid, settings=self.settings)
                    for aid, spec in cfg.advisors.items()}
        router = BrokerRouter(cfg, self.settings, self.switches, catalog, _LazyBreakers(self), advisors,
                              _LazyLimiter(self), prices, _LazyHandles(self), latency=self.latency, hop=self.instance_hop)
        self._version += 1
        return BrokerSnapshot(config=cfg, catalog=catalog, router=router, prices=prices, switches=self.switches,
                              adapters=adapters, issuer=issuer, advisors=advisors, version=self._version,
                              load_error=load_error)

    def _load(self) -> tuple[BrokerConfig, str | None]:
        try:
            cfg = read_broker_config(self.config_path)
        except (ValueError, BrokerError) as exc:
            log.error('broker.json could not be loaded; the broker stays disabled: %s', type(exc).__name__)
            return BrokerConfig(), getattr(exc, 'code', None) or 'invalid_broker_config'
        try:
            check_name_conflicts(cfg, self.local_models().keys())
        except BrokerError as exc:
            log.error('broker.json names collide with local models; the broker stays disabled')
            return BrokerConfig(), exc.code
        return cfg, None

    def snapshot(self) -> BrokerSnapshot:
        snap = self._snapshot
        if snap is not None:
            return snap
        with self._lock:
            if self._snapshot is None:
                cfg, error = self._load()
                self._snapshot = self._build(cfg, load_error=error)
                limiter = self._components.get('limiter')
                if limiter is not None:
                    limiter.prices = self._snapshot.prices
            return self._snapshot

    @property
    def configured(self) -> bool:
        return not self.snapshot().config.is_disabled()

    def active(self, principal) -> bool:
        """Whether BrokerDispatch must look at the request: the broker is configured, or the principal is a key."""
        if principal is not None and getattr(principal, 'kind', None) in ('key', 'parent'):
            return True
        return self.configured

    def default_principal(self) -> Principal:
        """The keyless loopback principal (used when no authenticated principal is on the scope)."""
        cfg = self.snapshot().config
        return Principal(id='anonymous', kind='anonymous_loopback',
                         scopes=anonymous_scopes(anonymous_cloud=bool(cfg.policy.anonymous_cloud),
                                                 role=self.settings.role))

    def effective_cloud_policy(self) -> str:
        """The deployment's cloud policy (for a delegated request in managed role; per-request hints only lower it)."""
        return cloud_policy_for(self.settings, self.snapshot().config, self.switches,
                                delegated=bool(getattr(self.switches, 'delegation', False)))

    @staticmethod
    def _enables_egress(old: BrokerConfig, new: BrokerConfig) -> bool:
        old_on = set(old.enabled_upstreams())
        new_on = set(new.enabled_upstreams())
        order = ('off', 'explicit', 'fallback')
        return bool(new_on - old_on) or order.index(new.policy.cloud_policy) > order.index(old.policy.cloud_policy) \
            or bool(set(new.advisors) - set(old.advisors))

    def reload(self, *, actor: str = 'admin') -> dict:
        """Re-read broker.json and swap the snapshot. 400 for an invalid document, 409 broker_name_conflict for a name
        that collides with a local model, 409 role_forbids for a change that would enable egress in managed role without
        delegation. The role itself never changes here."""
        try:
            cfg = read_broker_config(self.config_path)
        except BrokerError:
            raise
        except ValueError as exc:
            raise BrokerError(f'broker.json is invalid: {str(exc)[:300]}', 'invalid_request') from None
        check_name_conflicts(cfg, self.local_models().keys())
        old = self.snapshot().config
        if self.settings.role == 'managed' and not self.switches.delegation and self._enables_egress(old, cfg):
            raise BrokerError('This change would enable egress in managed role without delegation', 'role_forbids',
                              status=409, origin='policy')
        with self._lock:
            self._snapshot = self._build(cfg)
            limiter = self._components.get('limiter')
            if limiter is not None:
                limiter.prices = self._snapshot.prices
        try:
            self.audit.event('config.reload', actor, 'broker.json', {'version': self._snapshot.version,
                                                                      'upstreams': len(cfg.upstreams)})
        except Exception:
            log.warning('broker: audit of the reload failed', exc_info=True)
        return {'status': 'reloaded', 'snapshot': self.snapshot_summary()}

    def snapshot_summary(self) -> dict:
        snap = self.snapshot()
        cfg = snap.config
        return {'version': snap.version, 'upstreams': len(cfg.upstreams), 'enabled': len(cfg.enabled_upstreams()),
                'aliases': len(cfg.models), 'virtual_models': len(cfg.virtual_models), 'advisors': len(cfg.advisors),
                'load_error': snap.load_error}

    def on_registry_candidate(self, candidate) -> None:
        """Called by reload_registry right after candidate validation: a local name that would shadow a published alias
        or virtual model refuses the candidate (409 broker_name_conflict) before anything is retired."""
        from .catalog import local_models_from
        check_name_conflicts(self.snapshot().config, local_models_from(candidate).keys())

    # -- catalogs and info -----------------------------------------------------------------------------------------------
    def catalog_for(self, principal, *, loaded_only: bool = False, family: str = 'openai') -> list[dict]:
        """The broker entries (cloud and virtual) to append after the local entries of a catalog route."""
        if not self.active(principal):
            return []
        return self.snapshot().catalog.entries(principal or self.default_principal(), family=family,
                                               loaded_only=loaded_only)

    def visible_upstream(self, principal, upstream_id: str) -> bool:
        """Whether principal may see upstream_id at all: cloud or cloud:<id>, the key's upstream globs and target
        classes (the same rule as catalog visibility; admin and bootstrap see every upstream)."""
        import fnmatch
        cfg = self.snapshot().config
        upstream = cfg.upstreams.get(upstream_id)
        if principal is None or upstream is None:
            return False
        if allows(principal, 'admin'):
            return True
        klass = 'router' if PROFILES[upstream.profile].kind == 'router' else 'cloud'
        constraints = principal.constraints
        return (klass in constraints.target_classes
                and any(fnmatch.fnmatchcase(upstream_id, p) for p in constraints.upstreams)
                and allows(principal, 'cloud:' + upstream_id))

    @staticmethod
    def visible_local(principal, name) -> bool:
        """Whether a local catalog row (a registered model name) is listed for principal: key and parent principals see
        only the local models their target classes and model globs allow (the rule inference applies, where an invisible
        target is 404 model_not_found); every other principal (keyless, bootstrap, none) sees every local model, so
        their catalogs stay byte-identical."""
        import fnmatch
        if principal is None or getattr(principal, 'kind', None) not in ('key', 'parent'):
            return True
        constraints = principal.constraints
        return ('local' in constraints.target_classes and isinstance(name, str)
                and any(fnmatch.fnmatchcase(name, p) for p in constraints.models))

    async def refresh_catalogs(self, principal, *, force: bool = False) -> list[str]:
        """R21: refresh the live catalogs (purpose catalog) of enabled live-catalog upstreams the principal may see: the
        stale ones, or every visible one with force (GET /api/broker/catalog?refresh=true). Upstreams the principal
        cannot see are never contacted on its behalf. Returns the refreshed upstream ids."""
        if not self.active(principal):
            return []
        principal = principal or self.default_principal()
        snap = self.snapshot()
        tasks = {}
        for upstream_id, upstream in sorted(snap.config.upstreams.items()):
            if upstream.catalog != 'live' or not upstream.enabled or not self.visible_upstream(principal, upstream_id):
                continue
            task = self._catalog_fetch(snap.catalog, upstream_id, force=force)
            if task is not None:
                tasks[upstream_id] = task
        if tasks:
            await asyncio.wait(set(tasks.values()))
        return sorted(tasks)

    def _catalog_fetch(self, catalog, upstream_id: str, *, force: bool = False):
        """The live catalog fetch of upstream_id as a task: the one already in flight (single-flight: concurrent callers
        join it), a new one when forced or when the cache is stale and the last attempt is older than the retry window (a
        failing upstream is retried at most once a minute, the minute counted from the END of the last attempt, so a
        provider whose fetch hangs for its whole first-byte budget is not fetched back to back), else None. The task
        outlives a caller that stops waiting (deadline or disconnect), so its answer is still cached for the next
        request."""
        loop = asyncio.get_running_loop()
        task = self._catalog_tasks.get(upstream_id)
        if task is not None and not task.done() and task.get_loop() is loop:
            return task
        now = self._clock()
        last = self._refresh_tried.get(upstream_id)
        retry_window = min(float(getattr(self.settings, 'broker_catalog_ttl', 3600.0)), 60.0)
        if not force and not (catalog.stale(upstream_id) and (last is None or now - last >= retry_window)):
            return None
        self._refresh_tried[upstream_id] = now
        task = loop.create_task(catalog.refresh_upstream(upstream_id))
        self._catalog_tasks[upstream_id] = task

        def finished(done, uid=upstream_id):
            self._refresh_tried[uid] = self._clock()  # the retry window runs from the end of the attempt
            if self._catalog_tasks.get(uid) is done:
                self._catalog_tasks.pop(uid, None)
            if not done.cancelled() and done.exception() is not None:
                log.warning('broker: live catalog of %s failed (%s)', uid, type(done.exception()).__name__)

        task.add_done_callback(finished)
        return task

    async def ensure_catalogs(self, principal, upstream_ids=None, *, deadline: float | None = None) -> list[str]:
        """On-demand live catalogs (catalog.live_pending names which): fetch the stale live catalogs of upstream_ids (every
        live-catalog upstream when None) that principal may see, single-flight per upstream, and wait for them until
        deadline (time.monotonic(); a deadline at or before now starts the fetches without waiting) and never longer
        than CATALOG_WAIT_SECONDS (a fetch is itself bounded by the connect and first-byte timeouts and keeps running
        after the caller stops waiting). Upstreams the principal cannot see are never contacted (R21). Returns the ids
        whose fetch finished in time; never raises for upstream trouble (the catalog records it)."""
        if not self.switches.live_catalogs or not self.active(principal):
            return []
        principal = principal or self.default_principal()
        snap = self.snapshot()
        cfg = snap.config
        wanted = sorted(cfg.upstreams) if upstream_ids is None else sorted(set(upstream_ids))
        tasks = {}
        for upstream_id in wanted:
            upstream = cfg.upstreams.get(upstream_id)
            if upstream is None or upstream.catalog != 'live' or not upstream.enabled \
                    or not self.visible_upstream(principal, upstream_id):
                continue
            task = self._catalog_fetch(snap.catalog, upstream_id)
            if task is not None:
                tasks[upstream_id] = task
        if not tasks:
            return []
        timeout = CATALOG_WAIT_SECONDS if deadline is None else min(CATALOG_WAIT_SECONDS,
                                                                    max(0.0, deadline - time.monotonic()))
        if timeout <= 0.0:
            done = {task for task in tasks.values() if task.done()}
        else:
            done, _ = await asyncio.wait(set(tasks.values()), timeout=timeout)
        return sorted(uid for uid, task in tasks.items() if task in done)

    async def cloud_voices(self, principal) -> list[dict]:
        """The upstream ElevenLabs voices GET /v1/voices merges after the local voices: from every enabled, ready
        upstream on the elevenlabs adapter that principal may see (cloud:<id>, the key's upstream globs and target
        classes; the catalog visibility rule). Fetched with a purpose-catalog grant (so never in managed role, where
        live catalogs are off) and cached per upstream, origin and credential for broker_catalog_ttl; a failing upstream
        lists nothing and is retried at most once a minute. Each voice keeps the provider's fields and gains
        titanos: {remote: true, upstream: <id>}."""
        if not self.active(principal) or not self.switches.live_catalogs:
            return []
        principal = principal or self.default_principal()
        snap = self.snapshot()
        ttl = float(getattr(self.settings, 'broker_catalog_ttl', 3600.0))
        out: list[dict] = []
        for upstream_id, upstream in sorted(snap.config.upstreams.items()):
            adapter = snap.adapters.get(upstream_id)
            if (not upstream.enabled or adapter is None or getattr(adapter, 'adapter_name', '') != 'elevenlabs'
                    or 'elevenlabs.voices' not in adapter.profile.surfaces
                    or snap.catalog.readiness(upstream_id) is not None
                    or not self.visible_upstream(principal, upstream_id)):
                continue
            key = (upstream_id, upstream.base_url, upstream.credential)
            now = self._clock()
            cached = self._voice_cache.get(key)
            max_age = ttl if cached is not None and cached[1] else min(ttl, 60.0)  # an empty answer: retry in a minute
            if cached is None or now - cached[0] >= max_age:
                import uuid
                voices: tuple[dict, ...] = ()
                try:
                    deadline = now + float(self.settings.broker_connect_timeout) + float(
                        self.settings.broker_first_byte_timeout)
                    grant = snap.issuer.issue('catalog', upstream_id=upstream_id, decision_id=uuid.uuid4().hex,
                                              request_id=uuid.uuid4().hex, deadline=deadline)
                    voices = tuple(await adapter.list_voices(grant, self.egress))
                except BrokerError as exc:
                    log.info('broker: voices of %s unavailable (%s)', upstream_id, exc.code)
                except Exception as exc:  # transport trouble: list nothing, try again later
                    log.info('broker: voices of %s unavailable (%s)', upstream_id, type(exc).__name__)
                self._voice_cache[key] = cached = (now, voices)
            for voice in cached[1]:
                out.append({**voice, 'titanos': {'remote': True, 'upstream': upstream_id}})
        return out

    def model_entry(self, name: str, principal, *, protocol: str | None = 'openai') -> dict | None:
        """One broker entry for a vendor-shaped model lookup (OpenAI GET /v1/models/{id}): a published alias, qualified
        ref or exposed virtual model, else, exactly as inference resolves it, a bare provider id of `protocol` on the
        single visible upstream that lists it (catalog.resolve_vendor; the caller tried local models first): that
        target's entry under the id the client asked for, with titanos.ref naming the qualified ref. None otherwise."""
        if not self.active(principal):
            return None
        principal = principal or self.default_principal()
        catalog = self.snapshot().catalog
        found = catalog.entry(name, principal)
        if found is not None or not protocol:
            return found
        try:
            vendor = catalog.resolve_vendor(name, protocol, principal)
        except BrokerError:
            vendor = None
        target = getattr(vendor, 'target', None)
        entry = catalog.entry(target.ref, principal) if target is not None else None
        if entry is None:
            return None
        return {**entry, 'id': name, 'titanos': {**(entry.get('titanos') or {}), 'ref': target.ref}}

    def info(self, principal=None) -> dict:
        """The /api/info broker block: no URLs, no secrets."""
        snap = self.snapshot()
        cfg = snap.config
        try:
            backend = self.vault.backend_name
            sealed = bool(self.vault.sealed)
        except Exception:
            backend, sealed = 'unavailable', False
        return {'configured': not cfg.is_disabled(), 'cloud_policy_effective': self.effective_cloud_policy(),
                'upstreams_enabled': len(cfg.enabled_upstreams()),
                'advisors': sorted(cfg.advisors), 'vault_backend': backend, 'sealed': sealed}

    def cloud_reachable(self) -> bool:
        """Whether any cloud target could be served right now (policy not off, an upstream ready)."""
        if self.effective_cloud_policy() == 'off':
            return False
        snap = self.snapshot()
        return any(snap.catalog.readiness(uid) is None for uid in snap.config.enabled_upstreams())

    # -- thread-side gateways --------------------------------------------------------------------------------------------
    def gateway(self, native, principal, location):
        from .stages import GatewayRouter
        return GatewayRouter(native, self, principal, location)

    def stage_router(self):
        from .stages import StageRouter
        return StageRouter(self)

    def invoker(self):
        from .stages import CloudInvoker
        return self._component('invoker', lambda: CloudInvoker(self))

    def bind_artifact_writer(self, artifact: Callable[[str], Path]) -> None:
        """artifact(name) -> the Path to write a /v1/files artifact (PNG url re-hosting, cloud job outputs)."""
        self.artifact_writer = artifact

    # -- usage -----------------------------------------------------------------------------------------------------------
    def record_usage(self, run, decision, target, usage, cost, total_ms: float) -> None:
        from .pricing import money
        from .usage import UsageRecord
        prices = run.snapshot.prices
        try:
            record = UsageRecord(
                ts=decision.ts, request_id=decision.request_id, decision_id=decision.decision_id, key_id=decision.key_id,
                role=self.settings.role, surface=run.surface.name,
                target_class=target.klass if target is not None else 'none',
                upstream=target.upstream_id if target is not None else None,
                served_ref=target.ref if target is not None else None,
                upstream_model=target.upstream_model if target is not None else None,
                upstream_request_id=run.upstream_request_id,
                input_tokens=getattr(usage, 'input_tokens', None), output_tokens=getattr(usage, 'output_tokens', None),
                cached_input_tokens=getattr(usage, 'cached_input_tokens', None),
                reasoning_tokens=getattr(usage, 'reasoning_tokens', None), characters=getattr(usage, 'characters', None),
                audio_seconds=getattr(usage, 'audio_seconds', None), images=getattr(usage, 'images', None),
                cost_usd=money(cost) if cost is not None else None,
                cost_basis=(prices.cost_basis(target.ref) if (target is not None and target.klass != 'local'
                                                              and prices is not None) else 'unpriced'),
                price_basis=getattr(decision, 'price_basis', None),
                estimated=bool(getattr(usage, 'estimated', False)), outcome=run.outcome, status=run.status,
                ttfb_ms=round(run.ttfb_ms, 3) if run.ttfb_ms is not None else None, total_ms=round(total_ms, 3))
            self.usage.record(record)
        except Exception:
            log.warning('broker: recording usage failed', exc_info=True)

    # -- health ----------------------------------------------------------------------------------------------------------
    def _probe_targets(self) -> list:
        cfg = self.snapshot().config
        return [(uid, up.credential) for uid, up in cfg.enabled_upstreams().items()]

    async def _probe(self, upstream_id: str):
        import uuid
        snap = self.snapshot()
        adapter = snap.adapters.get(upstream_id)
        if adapter is None:
            from .adapters.base import ProbeResult
            return ProbeResult(ok=False, status=None, latency_ms=None, code='upstream_unconfigured')
        deadline = self._clock() + float(self.settings.broker_connect_timeout) + float(
            self.settings.broker_first_byte_timeout)
        grant = snap.issuer.issue('probe', upstream_id=upstream_id, decision_id=uuid.uuid4().hex,
                                  request_id=uuid.uuid4().hex, deadline=deadline)
        return await adapter.probe(grant, self.egress)

    # -- credential checks (R10) -----------------------------------------------------------------------------------------
    def _auth_suspect(self, upstream_id: str | None, credential_id: str, loop=None) -> None:
        """EgressClient.on_auth_suspect: an inference call got a 401 without the provider's key-rejection code. Any key
        with the scopes chooses the endpoint and body of an inference call (a native row, a typed surface such as
        moderations or images), and OpenAI answers a restricted or project-limited key with 401 on every endpoint
        outside its permissions, so that answer alone never takes the shared credential offline for every other caller:
        the upstream's own probe endpoint (GET models, purpose probe) is asked instead, and EgressClient marks the
        credential failed:upstream_auth only if that answer rejects it too. Thread-safe: send() runs it in a worker
        thread and send_sync() from CloudInvoker threads (loop None: the serving loop recorded at startup)."""
        loop = loop or self._loop
        if not upstream_id or loop is None or loop.is_closed():
            log.info('broker: an upstream 401 on %s was not checked against its probe endpoint (no event loop)',
                     upstream_id)
            return
        try:
            loop.call_soon_threadsafe(self._start_credential_check, upstream_id, credential_id)
        except RuntimeError:  # the loop closed meanwhile
            pass

    def _start_credential_check(self, upstream_id: str, credential_id: str) -> None:
        """Start the probe of upstream_id (on the loop): single-flight per (upstream, credential), and at most one per
        CREDENTIAL_CHECK_SECONDS counted from the end of the last check, so 401s cost the provider one GET a minute."""
        key = (upstream_id, credential_id)
        task = self._credential_tasks.get(key)
        if task is not None and not task.done():
            return
        last = self._credential_tried.get(key)
        if last is not None and self._clock() - last < CREDENTIAL_CHECK_SECONDS:
            return
        self._credential_tried[key] = self._clock()
        task = asyncio.get_running_loop().create_task(self._check_credential(upstream_id))
        self._credential_tasks[key] = task

        def finished(done, key=key):
            self._credential_tried[key] = self._clock()
            if self._credential_tasks.get(key) is done:
                self._credential_tasks.pop(key, None)
        task.add_done_callback(finished)

    async def _check_credential(self, upstream_id: str) -> None:
        try:
            result = await self._probe(upstream_id)
        except Exception as exc:  # a check that cannot run leaves the credential as it was
            log.info('broker: credential check of %s after an upstream 401 did not run (%s)', upstream_id,
                     type(exc).__name__)
        else:
            log.info('broker: credential check of %s after an upstream 401: %s', upstream_id,
                     'accepted' if getattr(result, 'ok', False) else (getattr(result, 'code', None) or 'failed'))
        finally:
            self.credential_checks += 1

    def upstream_health(self, upstream_id: str) -> dict:
        return self.monitor.status(upstream_id)

    def metrics_dict(self) -> dict:
        return self.metrics.to_dict(breakers=self._components.get('breakers'), limiter=self._components.get('limiter'),
                                    authenticator=self._components.get('authenticator'))

    # -- explain ---------------------------------------------------------------------------------------------------------
    async def explain(self, surface_name: str, body: dict, headers: dict | None, principal, *,
                      advisor: bool = False) -> dict:
        """POST /api/broker/explain: steps 1-8 with no egress and no reservation; the Decision (dry_run, served null)."""
        from . import inbound
        from .role import parse_parent_headers
        from .surfaces import SURFACES
        from .types import RouteContext
        surface = SURFACES.get(surface_name)
        if surface is None or surface.kind == 'catalog':
            raise BrokerError(f'Unknown surface {surface_name!r}', 'invalid_request')
        principal = principal or self.default_principal()
        hints = parse_parent_headers(headers or {}, self.settings, principal)
        document = body if isinstance(body, dict) else {}
        model = document.get(surface.model_from.split(':', 1)[1]) if surface.model_from.startswith('json:') else \
            document.get('model')
        summary = inbound.summarize(surface, document, None, len(str(document)))
        ctx = RouteContext(surface=surface, requested_model=model if isinstance(model, str) else '',
                           principal=principal, hints=hints, body=summary, dry_run=True)
        decision = await self.snapshot().router.explain(ctx, advisor=advisor,
                                                        advisor_input=lambda: inbound.advisor_input(surface, document))
        stored = self.decisions.append(decision)
        return stored

    # -- lifecycle -------------------------------------------------------------------------------------------------------
    async def startup(self) -> None:
        """Lifespan start: logging handler filters re-applied (after uvicorn's dictConfig), managed-role vault seal, and
        the audit role.start record (when the broker is configured or the role is broker); health probes start lazily."""
        from .redact import install_logging_redaction
        install_logging_redaction()
        self._loop = asyncio.get_running_loop()
        cfg = self.snapshot().config
        if self.settings.role == 'managed':
            if not self.switches.delegation:
                self.vault.seal('managed role')
            enabled = len(cfg.enabled_upstreams())
            if enabled and not self.switches.delegation and not self._warned_managed:
                self._warned_managed = True
                log.warning('managed role: %d configured upstreams ignored; set TITANOS_MANAGED_DELEGATION=parent to '
                            'allow parent-delegated use', enabled)
        if not cfg.is_disabled() or self.settings.role == 'broker':
            try:
                self.audit.event('role.start', 'system', self.settings.role,
                                 {'configured': not cfg.is_disabled(), 'upstreams': len(cfg.enabled_upstreams())})
            except Exception:
                log.warning('broker: audit role.start failed', exc_info=True)
        if not cfg.is_disabled() and self.switches.health_probes:
            self.monitor.start()

    async def close(self) -> None:
        for task in list(self._credential_tasks.values()):
            task.cancel()
        monitor = self._components.get('monitor')
        if monitor is not None:
            await monitor.stop()
        egress = self._components.get('egress')
        if egress is not None:
            await egress.aclose()


class _AuthAudit:
    """The audit sink of the Authenticator: auth.failure records are written when the audit log is on (roles table:
    'ON if the broker is configured', required in broker role, on in managed role, and on whenever a key record
    exists). A standalone server with only TITANOS_API_KEY and no broker.json writes nothing, exactly as before."""

    def __init__(self, broker: 'Broker'):
        self._broker = broker

    def enabled(self) -> bool:
        broker = self._broker
        if broker.settings.role != 'standalone':
            return True
        try:
            return broker.configured or broker.keys.has_records()
        except Exception:
            return True

    def event(self, kind, actor, subject, detail=None):
        if not self.enabled():
            return None
        return self._broker.audit.event(kind, actor, subject, detail)


class _Lazy:
    """Defers a component until first attribute use (snapshots are built before the component may be needed)."""
    _name = ''

    def __init__(self, broker: Broker):
        object.__setattr__(self, '_broker', broker)

    def __getattr__(self, name):
        return getattr(getattr(self._broker, self._name), name)


class _LazyVault(_Lazy):
    _name = 'vault'


class _LazyEgress(_Lazy):
    _name = 'egress'


class _LazyBreakers(_Lazy):
    _name = 'breakers'


class _LazyLimiter(_Lazy):
    _name = 'limiter'


class _LazyHandles(_Lazy):
    _name = 'handles'


def install_broker(app, *, settings, config, engine_slot=None, native_getter=None, bootstrap_key: str | None = None,
                   registry_path=None, outputs=None, home=None) -> Broker:
    """Create the lazy Broker and record it on app.state.broker. No files, no sockets, no vault resolution."""
    broker = Broker(settings=settings, config=config, engine_slot=engine_slot, native_getter=native_getter,
                    bootstrap_key=bootstrap_key, registry_path=registry_path, outputs=outputs, home=home)
    state = getattr(app, 'state', None)
    if state is not None:
        state.broker = broker
    return broker


__all__ = ['Broker', 'BrokerSnapshot', 'Metrics', 'TTFB_BUCKETS', 'install_broker']
