"""Circuit breakers per (upstream_id, credential_id) and the lazy HealthMonitor.

Outcome classes (types.OutcomeKind):
- success: closes a half-open breaker and resets the consecutive-failure count;
- client_error, auth, cancelled, refused: never counted (an upstream that answered is healthy; auth marks the credential
  in egress instead, by upstream_errors.credential_failure, an inference 401 without a key-rejection code only after the
  probe endpoint rejects the credential too);
- rate_limited: a cooldown until Retry-After (the upstream's value, else 1 s); the breaker does NOT open, but the upstream
  is skipped by the router until the cooldown ends. Profiles whose provider limits per model (extras ratelimit_per_model:
  OpenAI, Anthropic, Gemini) keep that cooldown per (upstream, credential, model) instead (breaker_for returns a
  ScopedBreaker), so one chat model at its limit never takes embeddings, speech or other models offline, and the router
  answers a request for the cooling model itself with 429 upstream_rate_limited (policy.refusal), not 503;
- server_error, network, timeout: counted. broker_breaker_failures consecutive counted failures open the breaker.

Open duration = min(broker_breaker_base_seconds x 2^(opens-1), broker_breaker_max_seconds), then +-20% jitter. When it
elapses the breaker is half-open and admits exactly ONE probe (the next real request, or a HealthMonitor GET models probe
through a GrantIssuer purpose 'probe' grant): success closes it, a counted failure re-opens it with the duration doubled.

Rate-limit headroom is read from x-ratelimit-remaining-* and anthropic-ratelimit-*-remaining response headers.

HealthMonitor probes only upstreams that are half-open, or that served traffic within broker_probe_idle_cutoff, and only
when switches.health_probes is on, so an idle Titanos makes no outbound connections. Every clock is injectable.
"""
from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger('titanos.broker')

COUNTED = frozenset({'server_error', 'network', 'timeout'})
NEUTRAL = frozenset({'client_error', 'auth', 'cancelled', 'refused'})
JITTER = 0.2
DEFAULT_COOLDOWN = 1.0
HALF_OPEN_BUSY_RETRY = 1.0
LAST_OUTCOMES = 20


@dataclass(frozen=True)
class Permit:
    """Admission through a breaker. probe=True: this call is the single half-open probe."""
    probe: bool = False

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class BreakerOpen:
    """The breaker (or a 429 cooldown) refuses; retry_after seconds until it may admit again."""
    retry_after: float
    reason: str = 'breaker_open'  # breaker_open | cooldown | probe_in_flight

    def __bool__(self) -> bool:
        return False


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def ratelimit_headroom(headers) -> dict:
    """{requests_remaining, tokens_remaining, reset} from OpenAI-style or Anthropic rate-limit headers (None if absent)."""
    if not headers:
        return {}
    get = headers.get if hasattr(headers, 'get') else dict(headers).get
    out = {}
    for name in ('x-ratelimit-remaining-requests', 'anthropic-ratelimit-requests-remaining'):
        if get(name) is not None:
            out['requests_remaining'] = _number(get(name))
            break
    for name in ('x-ratelimit-remaining-tokens', 'anthropic-ratelimit-tokens-remaining',
                 'anthropic-ratelimit-input-tokens-remaining'):
        if get(name) is not None:
            out['tokens_remaining'] = _number(get(name))
            break
    for name in ('x-ratelimit-reset-requests', 'anthropic-ratelimit-requests-reset', 'x-ratelimit-reset-tokens'):
        if get(name) is not None:
            out['reset'] = str(get(name))[:64]
            break
    return out


def _retry_after(headers) -> float | None:
    if not headers:
        return None
    get = headers.get if hasattr(headers, 'get') else dict(headers).get
    value = _number(get('retry-after') or get('Retry-After'))
    return value


class CircuitBreaker:
    """One breaker. state is computed: an open breaker whose time has passed reports half_open."""

    def __init__(self, upstream_id: str, credential_id: str | None, *, failures: int = 3, base: float = 2.0,
                 maximum: float = 300.0, clock: Callable[[], float] = time.monotonic, rng: random.Random | None = None,
                 on_transition: Callable[['CircuitBreaker', str, str], None] | None = None):
        self.upstream_id = upstream_id
        self.credential_id = credential_id
        self.failures = max(1, int(failures))
        self.base = float(base)
        self.maximum = float(maximum)
        self._clock = clock
        self._rng = rng or random.Random()
        self._on_transition = on_transition
        self._lock = threading.Lock()
        self._state = 'closed'
        self.consecutive = 0
        self.opens = 0
        self.open_until = 0.0
        self.last_duration: float | None = None
        self.cooldown_until = 0.0
        self.probe_in_flight = False
        self.last_used: float | None = None
        self.last_outcomes: deque = deque(maxlen=LAST_OUTCOMES)
        self.ratelimit: dict = {}

    # -- state -----------------------------------------------------------------------------------------------------------
    def _set(self, state: str) -> None:
        old, self._state = self._state, state
        if old != state and self._on_transition is not None:
            try:
                self._on_transition(self, old, state)
            except Exception:  # bookkeeping never fails a request
                log.debug('breaker transition hook failed', exc_info=True)

    def _tick(self, now: float) -> None:
        if self._state == 'open' and now >= self.open_until:
            self.probe_in_flight = False
            self._set('half_open')

    @property
    def state(self) -> str:
        with self._lock:
            self._tick(self._clock())
            return self._state

    def duration(self, opens: int) -> float:
        """The un-jittered open duration for the opens-th opening."""
        return min(self.base * (2 ** max(0, opens - 1)), self.maximum)

    def _open(self, now: float) -> None:
        self.opens += 1
        duration = self.duration(self.opens) * self._rng.uniform(1 - JITTER, 1 + JITTER)
        self.last_duration = duration
        self.open_until = now + duration
        self.probe_in_flight = False
        self._set('open')

    def retry_after(self) -> float:
        with self._lock:
            now = self._clock()
            self._tick(now)
            waits = [self.cooldown_until - now]
            if self._state == 'open':
                waits.append(self.open_until - now)
            elif self._state == 'half_open' and self.probe_in_flight:
                waits.append(HALF_OPEN_BUSY_RETRY)
            return max(0.0, max(waits))

    # -- API -------------------------------------------------------------------------------------------------------------
    def allow(self, *, consume: bool = True) -> Permit | BreakerOpen:
        """Permit, or BreakerOpen(retry_after). consume=False only inspects (the router's policy filter); dispatch and the
        monitor consume the half-open probe slot."""
        with self._lock:
            now = self._clock()
            self._tick(now)
            if now < self.cooldown_until:
                return BreakerOpen(self.cooldown_until - now, 'cooldown')
            if self._state == 'open':
                return BreakerOpen(max(0.0, self.open_until - now))
            if self._state == 'half_open':
                if self.probe_in_flight:
                    return BreakerOpen(HALF_OPEN_BUSY_RETRY, 'probe_in_flight')
                if consume:
                    self.probe_in_flight = True
                return Permit(probe=True)
            return Permit()

    def record(self, outcome: str, headers=None, *, retry_after: float | None = None) -> None:
        """Record one outcome (types.OutcomeKind) and the response headers when there were any."""
        with self._lock:
            now = self._clock()
            self._tick(now)
            self.last_used = now
            self.last_outcomes.append(outcome)
            if headers:
                headroom = ratelimit_headroom(headers)
                if headroom:
                    self.ratelimit = headroom
            if outcome == 'success':
                self.consecutive = 0
                self.probe_in_flight = False
                if self._state != 'closed':
                    self.opens = 0
                    self._set('closed')
                return
            if outcome == 'rate_limited':
                wait = retry_after if retry_after is not None else _retry_after(headers)
                self.cooldown_until = max(self.cooldown_until, now + (wait if wait is not None else DEFAULT_COOLDOWN))
                self.probe_in_flight = False
                return
            if outcome in COUNTED:
                if self._state == 'half_open':
                    self._open(now)
                    return
                self.consecutive += 1
                if self._state == 'closed' and self.consecutive >= self.failures:
                    self.consecutive = 0
                    self._open(now)
                return
            # neutral outcomes: the probe slot is released, nothing is counted
            self.probe_in_flight = False

    def touch(self) -> None:
        with self._lock:
            self.last_used = self._clock()

    def reset(self) -> None:
        with self._lock:
            self.consecutive = 0
            self.opens = 0
            self.open_until = 0.0
            self.cooldown_until = 0.0
            self.probe_in_flight = False
            self._set('closed')

    def snapshot(self) -> dict:
        with self._lock:
            now = self._clock()
            self._tick(now)
            return {'upstream': self.upstream_id, 'credential': self.credential_id, 'breaker': self._state,
                    'opens': self.opens, 'consecutive_failures': self.consecutive,
                    'retry_after': round(max(0.0, (self.open_until - now) if self._state == 'open' else 0.0,
                                             self.cooldown_until - now), 3),
                    'last_outcomes': list(self.last_outcomes), 'ratelimit': dict(self.ratelimit),
                    **({'model': self.model} if getattr(self, 'model', None) else {})}


class BreakerBoard:
    """Every breaker, keyed (upstream_id, credential_id). transitions counts breaker state changes for metrics; an
    opening is written to the audit log (breaker.open)."""

    def __init__(self, settings=None, *, clock: Callable[[], float] = time.monotonic, rng: random.Random | None = None,
                 audit=None):
        self.failures = int(getattr(settings, 'broker_breaker_failures', 3))
        self.base = float(getattr(settings, 'broker_breaker_base_seconds', 2.0))
        self.maximum = float(getattr(settings, 'broker_breaker_max_seconds', 300.0))
        self._clock = clock
        self._rng = rng or random.Random()
        self.audit = audit
        self._lock = threading.Lock()
        self._breakers: dict[tuple[str, str | None], CircuitBreaker] = {}
        self.transitions: dict[str, int] = {}

    def _transition(self, breaker: CircuitBreaker, old: str, new: str) -> None:
        key = f'{old}->{new}'
        self.transitions[key] = self.transitions.get(key, 0) + 1
        if new == 'open' and self.audit is not None:
            try:
                self.audit.event('breaker.open', 'system', breaker.upstream_id,
                                 {'opens': breaker.opens, 'seconds': round(breaker.last_duration or 0, 3)})
            except Exception:
                log.debug('breaker audit failed', exc_info=True)

    def get(self, upstream_id: str, credential_id: str | None, model: str | None = None) -> CircuitBreaker:
        """model: a router row that picks a backend per model (switchyard) keeps one breaker per routed model, so one
        failing route never takes the others offline (breaker_for)."""
        key = (upstream_id, credential_id) if model is None else (upstream_id, credential_id, model)
        with self._lock:
            breaker = self._breakers.get(key)
            if breaker is None:
                breaker = self._breakers[key] = CircuitBreaker(
                    upstream_id, credential_id, failures=self.failures, base=self.base, maximum=self.maximum,
                    clock=self._clock, rng=self._rng, on_transition=self._transition)
                if model is not None:
                    breaker.model = model  # a per-route breaker (snapshot names the route)
            return breaker

    def find(self, upstream_id: str) -> list[CircuitBreaker]:
        with self._lock:
            return [b for key, b in self._breakers.items() if key[0] == upstream_id]

    def snapshot(self) -> list[dict]:
        with self._lock:
            breakers = list(self._breakers.values())
        return [b.snapshot() for b in breakers]

    def reset(self, upstream_id: str) -> int:
        found = self.find(upstream_id)
        for breaker in found:
            breaker.reset()
        return len(found)


class ScopedBreaker:
    """One upstream's shared failure breaker plus a per-model rate-limit cooldown (extras ratelimit_per_model): a 429
    is recorded on the model's own breaker (which only ever cools down), every other outcome on the shared one;
    admission needs both. The interface is CircuitBreaker's (allow, record, touch, retry_after, state, snapshot)."""

    def __init__(self, shared: CircuitBreaker, model: CircuitBreaker):
        self.shared = shared
        self.model_breaker = model
        self.upstream_id = shared.upstream_id
        self.credential_id = shared.credential_id

    def allow(self, *, consume: bool = True):
        cooling = self.model_breaker.allow(consume=False)
        if not cooling:
            return cooling
        return self.shared.allow(consume=consume)

    def record(self, outcome: str, headers=None, *, retry_after: float | None = None) -> None:
        if outcome == 'rate_limited':
            self.model_breaker.record(outcome, headers, retry_after=retry_after)
            self.shared.record('refused', headers)  # releases a half-open probe slot; counts nothing
            return
        self.shared.record(outcome, headers, retry_after=retry_after)

    def touch(self) -> None:
        self.shared.touch()

    def retry_after(self) -> float:
        return max(self.shared.retry_after(), self.model_breaker.retry_after())

    @property
    def state(self) -> str:
        return self.shared.state

    def snapshot(self) -> dict:
        return self.shared.snapshot()


def breaker_for(breakers, target, upstream, *, model: str | None = None):
    """The breaker of one cloud or router target: per (upstream, credential), or per routed model as well when the
    upstream's profile sets extras breaker_per_model (a router such as Switchyard answers each route from other
    backends, and passes their failures through), or a ScopedBreaker (shared failures, per-model 429 cooldowns) when it
    sets extras ratelimit_per_model. model overrides the target's upstream model (the native lane passes 'native')."""
    from .profiles import PROFILES
    row = PROFILES.get(getattr(upstream, 'profile', None))
    name = model or getattr(target, 'upstream_model', None)
    upstream_id = getattr(target, 'upstream_id', None) or getattr(target, 'id', None)
    if row is not None and row.extras.get('breaker_per_model') and name:
        return breakers.get(upstream_id, upstream.credential, model=name)
    shared = breakers.get(upstream_id, upstream.credential)
    if row is not None and row.extras.get('ratelimit_per_model') and name:
        return ScopedBreaker(shared, breakers.get(upstream_id, upstream.credential, model='ratelimit:' + name))
    return shared


def native_breaker(breakers, upstream_id: str, upstream):
    """The breaker of the native lane of one upstream: the shared (upstream, credential) breaker, with a 429 cooldown of
    its own ('ratelimit:native') on ratelimit_per_model profiles, so a provider 429 on a files or batch call never
    cools down the typed surfaces' models."""
    from .profiles import PROFILES
    shared = breakers.get(upstream_id, upstream.credential)
    row = PROFILES.get(getattr(upstream, 'profile', None))
    if row is not None and row.extras.get('ratelimit_per_model'):
        return ScopedBreaker(shared, breakers.get(upstream_id, upstream.credential, model='ratelimit:native'))
    return shared


ProbeFn = Callable[[str], Awaitable[object]]  # upstream_id -> adapters.base.ProbeResult


class HealthMonitor:
    """Lazy prober. upstreams() yields (upstream_id, credential_id) of the enabled upstreams of the current snapshot;
    probe(upstream_id) performs the GET models probe through a purpose 'probe' grant and returns a ProbeResult."""

    def __init__(self, board: BreakerBoard, *, switches, settings=None, upstreams: Callable[[], list], probe: ProbeFn,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Awaitable] = asyncio.sleep):
        self.board = board
        self.switches = switches
        self.ttl = float(getattr(settings, 'broker_health_ttl', 300.0))
        self.idle_cutoff = float(getattr(settings, 'broker_probe_idle_cutoff', 900.0))
        self._upstreams = upstreams
        self._probe = probe
        self._clock = clock
        self._sleep = sleep
        self._task: asyncio.Task | None = None
        self.probes = 0
        self.last_probe: dict[str, float] = {}

    def due(self, upstream_id: str, credential_id: str | None) -> bool:
        """Whether this upstream may be probed now: half-open (probe slot free), or used within the idle cutoff and not
        probed within the TTL."""
        if not self.switches.health_probes:
            return False
        breaker = self.board.get(upstream_id, credential_id)
        state = breaker.state
        if state == 'half_open':
            return not breaker.probe_in_flight
        if state == 'open':
            return False
        now = self._clock()
        if breaker.last_used is None or now - breaker.last_used > self.idle_cutoff:
            return False
        last = self.last_probe.get(upstream_id)
        return last is None or now - last >= self.ttl

    async def tick(self) -> list[str]:
        """One probing round; returns the probed upstream ids."""
        if not self.switches.health_probes:
            return []
        probed = []
        for upstream_id, credential_id in list(self._upstreams() or ()):
            if not self.due(upstream_id, credential_id):
                continue
            breaker = self.board.get(upstream_id, credential_id)
            permit = breaker.allow()
            if not permit:
                continue
            self.last_probe[upstream_id] = self._clock()
            self.probes += 1
            probed.append(upstream_id)
            try:
                result = await self._probe(upstream_id)
            except Exception as exc:  # a failing probe is a network outcome, never a crash of the monitor
                log.debug('probe of %s failed: %s', upstream_id, type(exc).__name__)
                breaker.record('network')
                continue
            ok = bool(getattr(result, 'ok', False))
            code = getattr(result, 'code', None)
            if ok:
                breaker.record('success')
            elif code in ('upstream_rate_limited',):
                breaker.record('rate_limited')
            elif code in ('upstream_auth_failed', 'credential_unavailable', 'role_forbids', 'egress_refused'):
                breaker.record('auth' if code == 'upstream_auth_failed' else 'refused')
            elif code in ('timeout', 'upstream_timeout'):
                breaker.record('timeout')
            else:
                breaker.record('server_error' if getattr(result, 'status', None) else 'network')
        return probed

    def start(self) -> None:
        """Start the background loop (no-op when probes are off or it already runs). Needs a running event loop."""
        if not self.switches.health_probes or self._task is not None:
            return
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while True:
            await self._sleep(min(self.ttl, 60.0))
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug('health monitor round failed', exc_info=True)

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    def status(self, upstream_id: str) -> dict:
        """{breaker, opens, retry_after, last_outcomes, ratelimit:{requests_remaining, tokens_remaining, reset}}."""
        found = self.board.find(upstream_id)
        if not found:
            return {'breaker': 'closed', 'opens': 0, 'retry_after': 0.0, 'last_outcomes': [],
                    'ratelimit': {'requests_remaining': None, 'tokens_remaining': None, 'reset': None}}
        snap = found[0].snapshot()
        rl = snap['ratelimit']
        return {'breaker': snap['breaker'], 'opens': snap['opens'], 'retry_after': snap['retry_after'],
                'last_outcomes': snap['last_outcomes'],
                'ratelimit': {'requests_remaining': rl.get('requests_remaining'),
                              'tokens_remaining': rl.get('tokens_remaining'), 'reset': rl.get('reset')}}


__all__ = ['BreakerBoard', 'BreakerOpen', 'COUNTED', 'CircuitBreaker', 'HealthMonitor', 'JITTER', 'NEUTRAL', 'Permit',
           'ratelimit_headroom']
