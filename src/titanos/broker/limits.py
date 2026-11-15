"""Pre-dispatch admission: per-key RPM and TPM token buckets, a concurrency cap and the budget reservation; settlement
after the response.

- rpm: a bucket of rpm tokens refilled at rpm/60 per second; empty -> 429 rate_limit_exceeded with Retry-After.
- tpm: the estimate is ceil(utf-8 body bytes / 3) plus the output cap (estimate_input_tokens); a request larger than the
  whole bucket needs a full bucket. Settlement reconciles the bucket with the actual tokens.
- concurrent: in-flight admissions per key; at the cap -> 429 concurrency_exceeded (Retry-After 1).
- budget (cloud and router targets only; local inference costs 0 and is only rate-limited): usd_per_day and
  usd_per_month over UTC windows, tokens_per_day, max_request_usd. The estimated cost is reserved before dispatch;
  exceeding the remainder -> 402 budget_exceeded; a budgeted key calling an unpriced target -> 402
  budget_unpriced_target (fail closed). Both carry x-should-retry: false (errors table). A soft budget (hard=false)
  admits and flags the admission instead.
- A budgeted request without an output cap (Estimate.max_output_tokens is None) gets broker_default_max_output_tokens
  injected; Admission.injected_max_tokens tells dispatch to rewrite the body and stamp X-Titanos-Injected: max_tokens.
  Only token-generating surfaces (surfaces.TOKEN_OUTPUT_SURFACES) can get one: an Estimate with token_output=False
  (images, audio, embeddings, moderations, token counters) never has a cap computed or injected.
- Spend survives restarts: the first touch of a key's day or month window is seeded from the usage ledger.
- A budget counts only when it has a bound (usd_per_day, usd_per_month, tokens_per_day or max_request_usd), the same
  predicate KeyStore uses for the broker-role rule; a bound-less budget ({hard: true}) is treated as no budget.
- Fallback: Admission.retarget(target, price=...) re-checks the budget for the next attempt's target and swaps the
  reservation and the injected max_tokens decision under the lock, without taking another RPM token or concurrency slot
  ("budgets are re-checked per attempt"). A refused retarget leaves the admission as it was.
- Audio uploads (transcription, translation, speech-to-text) reserve their duration: measured when the container states
  it (WAV, also streamed WAV with placeholder sizes; FLAC; MP3: measure_audio_seconds), else the upload size at
  AUDIO_FLOOR_BYTES_PER_SECOND (an over-estimate for compressed speech; estimate_audio_seconds). Settlement prices the
  answer (pricing.PriceTable.price: tokens, the provider's seconds or the measured upload duration); only when nothing
  can be priced does the reservation stand as the spend (Admission.charged, basis 'reservation'), so a budgeted key
  never transcribes for free.
Residual overshoot is bounded by concurrent x output cap x price (documented). Only key and parent principals carry
limits or budgets; bootstrap and anonymous loopback are never limited here.
"""
from __future__ import annotations

import math
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal

from .errors import BrokerError
from .types import Principal, Target, Usage

_AUTO = object()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Estimate:
    """Pre-dispatch size of a request. max_output_tokens None means the body sets no output cap (budgeted keys get the
    default injected); 0 means the surface produces no output tokens (embeddings, speech, transcription, images).
    token_output False marks a surface that generates no tokens (surfaces.TOKEN_OUTPUT_SURFACES lists the ones that do):
    the limiter never injects an output cap for it and counts no output tokens, whatever max_output_tokens says."""
    input_tokens: int = 0
    max_output_tokens: int | None = None
    characters: int | None = None
    audio_seconds: float | None = None
    images: int | None = None
    token_output: bool = True

    def tokens(self, default_output: int = 0) -> int:
        if not self.token_output:
            out = 0
        else:
            out = self.max_output_tokens if self.max_output_tokens is not None else default_output
        return max(0, int(self.input_tokens or 0)) + max(0, int(out or 0))


def estimate_input_tokens(body_bytes: int) -> int:
    """ceil(utf-8 body bytes / 3): a deliberately high token estimate for TPM and budget reservations."""
    return math.ceil(max(0, int(body_bytes)) / 3)


# 8 kbit/s, below the bitrate of the common speech codecs (Opus voice, MP3, AAC, FLAC, PCM), so a non-WAV upload's
# duration is over-estimated rather than under-estimated (budgets fail closed).
AUDIO_FLOOR_BYTES_PER_SECOND = 1000
# Token-billed transcription rows (no per_audio_minute price): audio input and transcript output tokens per second of
# audio, both above what the token-billed speech models report, used only for the pre-dispatch reservation.
AUDIO_INPUT_TOKENS_PER_SECOND = 25
TRANSCRIPT_TOKENS_PER_SECOND = 10


def _wav_seconds(raw: bytes) -> float | None:
    """The duration of a RIFF/RF64 WAVE file from its fmt and data chunks. A streamed WAV (OpenAI and others write the
    placeholder sizes 0 or 0xFFFFFFFF before the length is known) is measured from the data bytes actually present."""
    if len(raw) < 12 or raw[:4] not in (b'RIFF', b'RF64') or raw[8:12] != b'WAVE':
        return None
    position, bytes_per_second = 12, None
    while position + 8 <= len(raw):
        chunk, size = raw[position:position + 4], struct.unpack('<I', raw[position + 4:position + 8])[0]
        start = position + 8
        if chunk == b'fmt ' and size >= 16 and start + 16 <= len(raw):
            tag, channels, rate, byte_rate, align, _bits = struct.unpack('<HHIIHH', raw[start:start + 16])
            if tag in (1, 3, 0xFFFE) and rate > 0 and align > 0:  # PCM, IEEE float, extensible: exact frame size
                bytes_per_second = rate * align
            elif byte_rate > 0:
                bytes_per_second = byte_rate
        elif chunk == b'data':
            if not bytes_per_second:
                return None
            present = len(raw) - start
            length = present if size in (0, 0xFFFFFFFF) or size > present else size
            return length / bytes_per_second
        if size == 0xFFFFFFFF:  # an unsized chunk before the data chunk: the layout cannot be followed
            return None
        position = start + size + (size & 1)
    return None


def _flac_seconds(raw: bytes) -> float | None:
    """The duration from a FLAC STREAMINFO block (total samples / sample rate); None when the total is unknown (0)."""
    if len(raw) < 42 or raw[:4] != b'fLaC' or raw[4] & 0x7F != 0:
        return None
    info = raw[8:42]
    packed = int.from_bytes(info[10:18], 'big')
    rate = packed >> 44
    total = packed & ((1 << 36) - 1)
    return total / rate if rate > 0 and total > 0 else None


_MP3_BITRATES = {  # kbit/s by (MPEG-1?, layer) and bitrate index 1..14
    (True, 1): (32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    (True, 2): (32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    (True, 3): (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    (False, 1): (32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    (False, 2): (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    (False, 3): (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
_MP3_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def _mp3_seconds(raw: bytes) -> float | None:
    """The duration of an MPEG audio (MP3) stream by walking every frame header after an optional ID3v2 tag. Any break
    in the frame chain (a format this walk does not know) gives None, never a guess."""
    position = 0
    if raw[:3] == b'ID3' and len(raw) >= 10:
        size = (raw[6] & 0x7F) << 21 | (raw[7] & 0x7F) << 14 | (raw[8] & 0x7F) << 7 | (raw[9] & 0x7F)
        position = 10 + size + (10 if raw[5] & 0x10 else 0)
    end = len(raw) - (128 if raw[-128:-125] == b'TAG' else 0)
    seconds, frames = 0.0, 0
    while position + 4 <= end:
        b1, b2 = raw[position + 1], raw[position + 2]
        if raw[position] != 0xFF or b1 & 0xE0 != 0xE0:
            break
        version, layer_bits = (b1 >> 3) & 3, (b1 >> 1) & 3
        bitrate_index, rate_index, padding = b2 >> 4, (b2 >> 2) & 3, (b2 >> 1) & 1
        if version == 1 or layer_bits == 0 or bitrate_index in (0, 15) or rate_index == 3:
            break
        mpeg1, layer = version == 3, 4 - layer_bits
        bitrate = _MP3_BITRATES[(mpeg1, layer)][bitrate_index - 1] * 1000
        rate = _MP3_RATES[version][rate_index]
        if layer == 1:
            length, samples = (12 * bitrate // rate + padding) * 4, 384
        elif layer == 2 or mpeg1:
            length, samples = 144 * bitrate // rate + padding, 1152
        else:
            length, samples = 72 * bitrate // rate + padding, 576
        if length < 4:
            break
        seconds += samples / rate
        frames += 1
        position += length
    # the walk must reach the end of the stream (a truncated last frame is allowed)
    if frames < 2 or end - position >= 4:
        return None
    return seconds


def measure_audio_seconds(data) -> float | None:
    """The exact duration of one uploaded audio file when its container states it: WAV (also a streamed WAV whose
    header carries placeholder sizes), FLAC (STREAMINFO) and MP3 (every frame header). None for any other format (Ogg,
    WebM, M4A ...) and for a file this reader cannot follow: callers then fall back to estimate_audio_seconds."""
    raw = bytes(data or b'')
    for reader in (_wav_seconds, _flac_seconds, _mp3_seconds):
        try:
            seconds = reader(raw)
        except (IndexError, KeyError, struct.error, ValueError):
            seconds = None
        if seconds is not None:
            return float(seconds)
    return None


def estimate_audio_seconds(data) -> float:
    """Pre-dispatch duration of one uploaded audio file: the measured duration when the container states it
    (measure_audio_seconds), otherwise the size at AUDIO_FLOOR_BYTES_PER_SECOND; at least one second."""
    raw = bytes(data or b'')
    seconds = measure_audio_seconds(raw)
    if seconds is None:
        seconds = len(raw) / AUDIO_FLOOR_BYTES_PER_SECOND
    return max(1.0, float(seconds))


class _Bucket:
    __slots__ = ('capacity', 'rate', 'tokens', 'at')

    def __init__(self, capacity: int, now: float):
        self.capacity = float(capacity)
        self.rate = float(capacity) / 60.0
        self.tokens = float(capacity)
        self.at = now

    def refill(self, now: float) -> None:
        if now > self.at:
            self.tokens = min(self.capacity, self.tokens + (now - self.at) * self.rate)
        self.at = now

    def wait_for(self, need: float) -> float:
        return 0.0 if self.tokens >= need else (need - self.tokens) / self.rate


class _KeyState:
    def __init__(self):
        self.rpm: _Bucket | None = None
        self.tpm: _Bucket | None = None
        self.active = 0
        self.usd_spent: dict[str, Decimal] = {}
        self.usd_reserved: dict[str, Decimal] = {}
        self.tokens_spent: dict[str, int] = {}
        self.tokens_reserved: dict[str, int] = {}
        self.seeded: set[str] = set()


def _windows(day: date) -> tuple[str, str]:
    return 'd:' + day.isoformat(), 'm:' + day.strftime('%Y-%m')


def bounded_budget(budget) -> bool:
    """Whether a KeyBudget sets at least one bound (keys.KeyRecord.has_bounded_budget uses the same rule)."""
    return budget is not None and any(getattr(budget, name, None) is not None for name in
                                      ('usd_per_day', 'usd_per_month', 'tokens_per_day', 'max_request_usd'))


class Admission:
    """One admitted request. Call settle() after the response (or release() when nothing was sent); both are idempotent,
    and using the admission as a context manager releases it if neither ran."""

    def __init__(self, limiter: 'Limiter', principal: Principal, target: Target, estimate: Estimate, *,
                 price: Decimal | None, injected_max_tokens: int | None, tpm_taken: float, usd: Decimal, tokens: int,
                 windows: tuple[str, str], over_budget: bool, limited: bool, requested: Estimate | None = None):
        self.limiter = limiter
        self.principal = principal
        self.target = target
        self.estimate = estimate
        self.price = price
        self.injected_max_tokens = injected_max_tokens
        self.over_budget = over_budget
        self.reserved_usd = usd
        self.reserved_tokens = tokens
        self._tpm_taken = tpm_taken
        self._windows = windows
        self._limited = limited
        self._requested = requested if requested is not None else estimate  # the caller's estimate, before injection
        self.settled = False
        self.released = False
        self.charged: Decimal | None = None  # the USD spend settle() recorded (the reservation when the cost was unknown)

    @property
    def done(self) -> bool:
        return self.settled or self.released

    def settle(self, usage: Usage | None, cost: Decimal | None) -> None:
        """Replace the reservation with the actual spend: cost when known, else the reserved estimate (conservative);
        actual tokens when reported, else the estimate."""
        if self.done:
            return
        self.settled = True
        self.limiter._settle(self, usage, cost)

    def retarget(self, target: Target, *, price=_AUTO) -> 'Admission':
        """Move this held admission to a fallback target: re-check the budget at the new target's price and swap the
        reservation and the injected max_tokens decision, without another RPM token or concurrency slot. Raises
        BrokerError budget_exceeded / budget_unpriced_target (402) and leaves the admission unchanged when the new target
        is refused. Returns self."""
        if self.done:
            raise RuntimeError('this admission was already settled or released')
        self.limiter._retarget(self, target, price)
        return self

    def release(self) -> None:
        """Cancel the reservation without spending (nothing reached the upstream); the RPM token is not refunded."""
        if self.done:
            return
        self.released = True
        self.limiter._release(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class Limiter:
    """Implements types.LimiterProtocol. usage (a UsageLedger) seeds spend per window; prices (a PriceTable) prices an
    admission when the caller does not pass price; audit records budget stops once per key, code and day."""

    def __init__(self, settings, *, usage=None, prices=None, audit=None, clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = _utcnow):
        self.settings = settings
        self.usage = usage
        self.prices = prices
        self.audit = audit
        self._clock = clock
        self._now = now
        self._lock = threading.Lock()
        self._state: dict[str, _KeyState] = {}
        self._audited: set[tuple[str, str, str]] = set()
        self.budget_stops = 0
        self.rate_stops = 0

    @property
    def default_output(self) -> int:
        return int(getattr(self.settings, 'broker_default_max_output_tokens', 4096))

    def _state_for(self, principal: Principal) -> _KeyState:
        state = self._state.get(principal.id)
        if state is None:
            state = self._state[principal.id] = _KeyState()
        return state

    def _seed(self, state: _KeyState, key_id: str, day: date, windows: tuple[str, str]) -> None:
        for window in windows:
            if window in state.seeded:
                continue
            state.seeded.add(window)
            for old in [w for w in state.seeded if w[0] == window[0] and w != window]:
                state.seeded.discard(old)
                for table in (state.usd_spent, state.usd_reserved, state.tokens_spent, state.tokens_reserved):
                    table.pop(old, None)
            if self.usage is None:
                continue
            since = day if window.startswith('d:') else day.replace(day=1)
            usd, tokens = self.usage.totals(key_id=key_id, since=since, until=day)
            state.usd_spent[window] = state.usd_spent.get(window, Decimal(0)) + usd
            state.tokens_spent[window] = state.tokens_spent.get(window, 0) + tokens

    def _stop(self, principal: Principal, target: Target, code: str, message: str, day: date) -> BrokerError:
        self.budget_stops += 1
        mark = (principal.id, code, day.isoformat())
        if self.audit is not None and mark not in self._audited:
            self._audited.add(mark)
            try:
                self.audit.event('budget.stop', principal.id, target.ref, {'code': code, 'day': day.isoformat()})
            except Exception:
                pass
        return BrokerError(message, code, origin='policy')

    def admit(self, principal: Principal, target: Target, estimate: Estimate, *, price=_AUTO) -> Admission:
        """Admit or raise BrokerError: rate_limit_exceeded / concurrency_exceeded (429, Retry-After), budget_exceeded /
        budget_unpriced_target (402). Nothing is consumed when the request is refused."""
        limits = principal.limits
        budget = principal.budget
        requested = estimate
        budgeted, estimate, injected, price = self._price(principal, target, estimate, price)
        est_tokens = estimate.tokens(self.default_output)
        now = self._clock()
        day = self._now().astimezone(timezone.utc).date()
        windows = _windows(day)
        with self._lock:
            state = self._state_for(principal)
            over_budget = False
            usd = Decimal(0)
            tokens = 0
            if budgeted:
                self._seed(state, principal.id, day, windows)
                problem = self._budget_problem(state, budget, price, est_tokens, windows)
                if problem is not None:
                    if budget.hard:
                        raise self._stop(principal, target, problem[0], problem[1], day)
                    over_budget = True
                usd = price if price is not None else Decimal(0)
                tokens = est_tokens
            tpm_need = 0.0
            if limits is not None:
                if limits.concurrent is not None and state.active >= limits.concurrent:
                    self.rate_stops += 1
                    raise BrokerError(f'This key allows {limits.concurrent} concurrent requests', 'concurrency_exceeded',
                                      retry_after=1, origin='policy')
                if limits.rpm is not None:
                    if state.rpm is None or state.rpm.capacity != limits.rpm:
                        state.rpm = _Bucket(limits.rpm, now)
                    state.rpm.refill(now)
                    if state.rpm.tokens < 1:
                        self.rate_stops += 1
                        raise BrokerError(f'This key allows {limits.rpm} requests per minute', 'rate_limit_exceeded',
                                          retry_after=state.rpm.wait_for(1), origin='policy')
                if limits.tpm is not None:
                    if state.tpm is None or state.tpm.capacity != limits.tpm:
                        state.tpm = _Bucket(limits.tpm, now)
                    state.tpm.refill(now)
                    tpm_need = float(min(est_tokens, limits.tpm))
                    if state.tpm.tokens < tpm_need:
                        self.rate_stops += 1
                        raise BrokerError(f'This key allows {limits.tpm} tokens per minute', 'rate_limit_exceeded',
                                          retry_after=state.tpm.wait_for(tpm_need), origin='policy')
            # commit
            if limits is not None:
                if limits.rpm is not None:
                    state.rpm.tokens -= 1
                if limits.tpm is not None:
                    state.tpm.tokens -= tpm_need
            state.active += 1
            if budgeted:
                self._reserve(state, windows, usd, tokens)
        return Admission(self, principal, target, estimate, price=price, injected_max_tokens=injected, tpm_taken=tpm_need,
                         usd=usd, tokens=tokens, windows=windows, over_budget=over_budget, limited=budgeted,
                         requested=requested)

    def _price(self, principal: Principal, target: Target, estimate: Estimate, price):
        """(budgeted, the estimate after any max_tokens injection, the injected value or None, price) for one target."""
        cloud = target.klass != 'local'
        budgeted = cloud and bounded_budget(principal.budget)
        injected = None
        if budgeted and estimate.token_output and estimate.max_output_tokens is None:
            injected = self.default_output
            estimate = replace(estimate, max_output_tokens=injected)
        if price is _AUTO:
            price = self.prices.estimate(target.ref, estimate) if (cloud and self.prices is not None) else None
        if not cloud:
            price = Decimal(0)
        return budgeted, estimate, injected, price

    @staticmethod
    def _reserve(state: _KeyState, windows: tuple[str, str], usd: Decimal, tokens: int) -> None:
        for window in windows:
            state.usd_reserved[window] = state.usd_reserved.get(window, Decimal(0)) + usd
        state.tokens_reserved[windows[0]] = state.tokens_reserved.get(windows[0], 0) + tokens

    def _retarget(self, admission: Admission, target: Target, price) -> None:
        principal = admission.principal
        budget = principal.budget
        budgeted, estimate, injected, price = self._price(principal, target, admission._requested, price)
        est_tokens = estimate.tokens(self.default_output)
        day = self._now().astimezone(timezone.utc).date()
        windows = _windows(day)
        with self._lock:
            if admission.done:
                raise RuntimeError('this admission was already settled or released')
            state = self._state_for(principal)
            self._unreserve(state, admission)
            over_budget = False
            usd = Decimal(0)
            tokens = 0
            if budgeted:
                self._seed(state, principal.id, day, windows)
                problem = self._budget_problem(state, budget, price, est_tokens, windows)
                if problem is not None:
                    if budget.hard:
                        if admission._limited:  # put the previous reservation back: the admission is unchanged
                            self._reserve(state, admission._windows, admission.reserved_usd, admission.reserved_tokens)
                        raise self._stop(principal, target, problem[0], problem[1], day)
                    over_budget = True
                usd = price if price is not None else Decimal(0)
                tokens = est_tokens
                self._reserve(state, windows, usd, tokens)
            admission.target, admission.estimate, admission.price = target, estimate, price
            admission.injected_max_tokens, admission.over_budget = injected, over_budget
            admission.reserved_usd, admission.reserved_tokens = usd, tokens
            admission._windows, admission._limited = windows, budgeted

    @staticmethod
    def _budget_problem(state: _KeyState, budget, price, est_tokens: int, windows) -> tuple[str, str] | None:
        if price is None:
            return 'budget_unpriced_target', 'This key has a budget and the target has no operator price row'
        if budget.max_request_usd is not None and price > budget.max_request_usd:
            return 'budget_exceeded', 'The estimated request cost exceeds this key\'s max_request_usd'
        day, month = windows
        for window, limit, label in ((day, budget.usd_per_day, 'daily'), (month, budget.usd_per_month, 'monthly')):
            if limit is None:
                continue
            used = state.usd_spent.get(window, Decimal(0)) + state.usd_reserved.get(window, Decimal(0))
            if used + price > limit:
                return 'budget_exceeded', f'This key\'s {label} USD budget is exhausted'
        if budget.tokens_per_day is not None:
            used = state.tokens_spent.get(day, 0) + state.tokens_reserved.get(day, 0)
            if used + est_tokens > budget.tokens_per_day:
                return 'budget_exceeded', 'This key\'s daily token budget is exhausted'
        return None

    def _unreserve(self, state: _KeyState, admission: Admission) -> None:
        if not admission._limited:
            return
        for window in admission._windows:
            if window in state.usd_reserved:
                state.usd_reserved[window] -= admission.reserved_usd
        day = admission._windows[0]
        if day in state.tokens_reserved:
            state.tokens_reserved[day] -= admission.reserved_tokens

    def _settle(self, admission: Admission, usage: Usage | None, cost: Decimal | None) -> None:
        with self._lock:
            state = self._state_for(admission.principal)
            state.active = max(0, state.active - 1)
            actual_tokens = None
            if usage is not None and (usage.input_tokens is not None or usage.output_tokens is not None):
                actual_tokens = (usage.input_tokens or 0) + (usage.output_tokens or 0)
            if state.tpm is not None and admission._tpm_taken and actual_tokens is not None:
                state.tpm.refill(self._clock())
                state.tpm.tokens = min(state.tpm.capacity, state.tpm.tokens + admission._tpm_taken - actual_tokens)
            if admission._limited:
                self._unreserve(state, admission)
                spent = cost if cost is not None else admission.reserved_usd
                admission.charged = spent
                for window in admission._windows:
                    if window in state.seeded:  # a window that rolled over meanwhile is no longer tracked
                        state.usd_spent[window] = state.usd_spent.get(window, Decimal(0)) + spent
                day = admission._windows[0]
                used = actual_tokens if actual_tokens is not None else admission.reserved_tokens
                if day in state.seeded:
                    state.tokens_spent[day] = state.tokens_spent.get(day, 0) + used

    def _release(self, admission: Admission) -> None:
        with self._lock:
            state = self._state_for(admission.principal)
            state.active = max(0, state.active - 1)
            if state.tpm is not None and admission._tpm_taken:
                state.tpm.refill(self._clock())
                state.tpm.tokens = min(state.tpm.capacity, state.tpm.tokens + admission._tpm_taken)
            self._unreserve(state, admission)

    def remaining(self, principal: Principal) -> dict[str, str | int | None]:
        """Budget remaining today and this month (GET /api/keys/self budget_remaining); None for unbounded windows."""
        budget = principal.budget
        if budget is None:
            return {}
        day = self._now().astimezone(timezone.utc).date()
        windows = _windows(day)
        with self._lock:
            state = self._state_for(principal)
            self._seed(state, principal.id, day, windows)

            def left(window, limit):
                if limit is None:
                    return None
                used = state.usd_spent.get(window, Decimal(0)) + state.usd_reserved.get(window, Decimal(0))
                return format(max(Decimal(0), limit - used), 'f')
            tokens_left = None
            if budget.tokens_per_day is not None:
                used = state.tokens_spent.get(windows[0], 0) + state.tokens_reserved.get(windows[0], 0)
                tokens_left = max(0, budget.tokens_per_day - used)
            return {'usd_today': left(windows[0], budget.usd_per_day), 'usd_month': left(windows[1], budget.usd_per_month),
                    'tokens_today': tokens_left}


__all__ = ['AUDIO_FLOOR_BYTES_PER_SECOND', 'AUDIO_INPUT_TOKENS_PER_SECOND', 'Admission', 'Estimate', 'Limiter',
           'TRANSCRIPT_TOKENS_PER_SECOND', 'bounded_budget', 'estimate_audio_seconds', 'estimate_input_tokens',
           'measure_audio_seconds']
