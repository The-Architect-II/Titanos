"""Operator-supplied price table. No prices ship in code (prices drift); money is Decimal throughout.

A row matches a model reference by fnmatch (case-sensitive). The most specific matching row wins: the most literal
characters, then the fewest '*' wildcards, then the later effective_from, then the later row. Rows whose
effective_from lies in the future are ignored.

Estimates (pre-dispatch reservations): every quantity of the Estimate is priced; a quantity the matched row has no price
for (for example images on a token-priced row) makes the estimate unpriced (None): budgets fail closed on unpriced
targets. An audio upload on a row with token prices is reserved as audio input and transcript output tokens; on a row
with both a per-minute and token prices the larger of the two reservations is taken.

Settlement (PriceTable.price) picks ONE basis per answer and records it (Priced.basis, Decision.price_basis,
UsageRecord.price_basis):
- tokens: the provider reported token usage and the row has token prices. Every reported token quantity must be priced
  (else unpriced); the row's unit prices (per image, per minute, per 1k characters) are not added on top, because token
  billing already covers what they would measure.
- units: otherwise the reported characters, images and audio seconds at the row's unit prices. Audio seconds are the
  provider's ('audio_seconds:provider') or, when the provider reports none and the row has a per-minute price, the
  uploaded audio's measured duration ('audio_seconds:upload', limits.measure_audio_seconds). Every present unit
  quantity must be priced; token counts are then informational.
- nothing priceable: cost None (basis 'unmeasured'); dispatch keeps the admission's reservation as the spend and records
  basis 'reservation' (fail closed), so no estimate is ever charged without saying so.
"""
from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

from .config import PriceRow

MILLION = Decimal(1_000_000)
THOUSAND = Decimal(1000)
SIXTY = Decimal(60)
UNPRICED = 'unpriced'
# the Usage fields a price applies to; a Usage with none of them set has an unknown cost (PriceTable.cost -> None)
MEASURED = ('input_tokens', 'output_tokens', 'characters', 'audio_seconds', 'images')
UNMEASURED = 'unmeasured'
RESERVATION = 'reservation'


@dataclass(frozen=True)
class Priced:
    """A settled cost and its basis: 'tokens', 'characters', 'images', 'audio_seconds:provider',
    'audio_seconds:upload' (unit bases joined with '+'), 'unmeasured' (cost None: nothing priceable) or 'unpriced'
    (cost None: no row)."""
    cost: Decimal | None
    basis: str


def _today() -> date:
    return datetime.now(timezone.utc).date()


def specificity(pattern: str) -> tuple[int, int]:
    """(literal characters, -number of '*'): larger is more specific. A [...] class counts as one non-literal."""
    collapsed = re.sub(r'\[[^\]]*\]', '?', pattern)
    literal = len(collapsed.replace('*', '').replace('?', ''))
    return literal, -collapsed.count('*')


def _ref(ref) -> str:
    return ref if isinstance(ref, str) else getattr(ref, 'ref')


def money(value: Decimal | None) -> str | None:
    """A Decimal as a plain (non-exponent) string, normalised; None stays None."""
    if value is None:
        return None
    text = format(value, 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.') or '0'
    return text


class PriceTable:
    def __init__(self, rows: Iterable[PriceRow | dict] = (), *, today: Callable[[], date] = _today):
        self.rows: tuple[PriceRow, ...] = tuple(r if isinstance(r, PriceRow) else PriceRow.model_validate(r) for r in rows)
        self._today = today

    def __len__(self) -> int:
        return len(self.rows)

    def match(self, ref) -> PriceRow | None:
        """The most specific row in effect today whose pattern matches ref, or None."""
        text = _ref(ref)
        today = self._today()
        best, best_key = None, None
        for index, row in enumerate(self.rows):
            if row.effective_from is not None and row.effective_from > today:
                continue
            if not fnmatch.fnmatchcase(text, row.match):
                continue
            key = (*specificity(row.match), row.effective_from or date.min, index)
            if best_key is None or key > best_key:
                best, best_key = row, key
        return best

    def cost_basis(self, ref) -> str:
        """The matched row's pattern, or 'unpriced' (UsageRecord.cost_basis)."""
        row = self.match(ref)
        return row.match if row is not None else UNPRICED

    @staticmethod
    def _sum(parts: list[tuple[Decimal | int | float | None, Decimal | None, Decimal]]) -> Decimal | None:
        total = Decimal(0)
        for quantity, price, unit in parts:
            if not quantity:
                continue
            if price is None:
                return None
            total += Decimal(str(quantity)) * price / unit
        return total

    def estimate(self, ref, est) -> Decimal | None:
        """Pre-dispatch cost of an Estimate: input tokens, the output cap, characters, audio seconds and images, each at
        the matched row's price; None when unpriced. A token_output=False estimate counts no output tokens."""
        row = self.match(ref)
        if row is None:
            return None
        output = est.max_output_tokens if getattr(est, 'token_output', True) else 0
        audio_seconds = est.audio_seconds
        token_audio = None
        if audio_seconds and row.input_per_mtok is not None:
            # a token-billed audio row: the audio is reserved as input tokens and the transcript as output tokens
            from .limits import AUDIO_INPUT_TOKENS_PER_SECOND, TRANSCRIPT_TOKENS_PER_SECOND
            seconds = Decimal(str(audio_seconds))
            audio_in = int((seconds * AUDIO_INPUT_TOKENS_PER_SECOND).to_integral_value('ROUND_CEILING'))
            audio_out = int((seconds * TRANSCRIPT_TOKENS_PER_SECOND).to_integral_value('ROUND_CEILING')) \
                if row.output_per_mtok is not None else 0
            token_audio = self._sum([((est.input_tokens or 0) + audio_in, row.input_per_mtok, MILLION),
                                     ((output or 0) + audio_out, row.output_per_mtok, MILLION),
                                     (est.characters, row.per_1k_chars, THOUSAND),
                                     (est.images, row.per_image, Decimal(1))])
            if row.per_audio_minute is None:
                return token_audio
        by_unit = self._sum([(est.input_tokens, row.input_per_mtok, MILLION),
                             (output, row.output_per_mtok, MILLION),
                             (est.characters, row.per_1k_chars, THOUSAND),
                             (audio_seconds, row.per_audio_minute, SIXTY),
                             (est.images, row.per_image, Decimal(1))])
        if token_audio is not None and by_unit is not None:
            return max(token_audio, by_unit)  # a row with both prices: reserve the larger
        return by_unit

    def price(self, ref, usage, *, upload_audio_seconds: float | None = None) -> Priced:
        """Settled cost of a Usage (None counts as nothing reported) and its basis (module docstring). Cached input
        tokens are billed at cached_input_per_mtok (input_per_mtok when the row has none), cache writes
        (Usage.cache_write_tokens, Anthropic cache_creation_input_tokens) at cache_write_per_mtok (input_per_mtok when
        the row has none) and the rest at input_per_mtok; reasoning tokens are part of output_tokens and are not added
        again. Anthropic details: 1-hour-TTL cache writes (Usage.cache_write_1h_tokens) at cache_write_1h_per_mtok (the
        cache-write price when absent); a fast-mode answer (Usage.fast) multiplies the token cost by fast_multiplier and
        web search requests add per_web_search_request each; either without its price leaves the answer unpriced (the
        reservation stands). OpenAI modality tokens: audio and image input tokens (parts of the uncached input) and audio
        output tokens (part of output_tokens) at audio_input_per_mtok, image_input_per_mtok and audio_output_per_mtok
        (the text token price when absent). upload_audio_seconds: the request's measured audio duration (transcription surfaces),
        used only when the provider reports no seconds and the row has a per-minute price."""
        row = self.match(ref)
        if row is None:
            return Priced(None, UNPRICED)
        get = (lambda name: getattr(usage, name, None)) if usage is not None else (lambda name: None)
        if get('input_tokens') is not None or get('output_tokens') is not None:
            if row.input_per_mtok is not None or row.output_per_mtok is not None:
                cached = get('cached_input_tokens') or 0
                written = get('cache_write_tokens') or 0
                fresh = max(0, (get('input_tokens') or 0) - cached - written)
                cached_price = row.cached_input_per_mtok if row.cached_input_per_mtok is not None else row.input_per_mtok
                write_price = row.cache_write_per_mtok if row.cache_write_per_mtok is not None else row.input_per_mtok
                # Anthropic 1-hour-TTL cache writes (a part of the writes) at their own price
                written_1h = min(written, get('cache_write_1h_tokens') or 0)
                write_1h_price = row.cache_write_1h_per_mtok if row.cache_write_1h_per_mtok is not None else write_price
                # OpenAI modality tokens (parts of the uncached input and of the output) at their own prices
                audio_in = min(fresh, get('audio_input_tokens') or 0)
                image_in = min(fresh - audio_in, get('image_input_tokens') or 0)
                output = get('output_tokens')
                audio_out = min(output or 0, get('audio_output_tokens') or 0)
                audio_in_price = row.audio_input_per_mtok if row.audio_input_per_mtok is not None else row.input_per_mtok
                image_in_price = row.image_input_per_mtok if row.image_input_per_mtok is not None else row.input_per_mtok
                audio_out_price = (row.audio_output_per_mtok if row.audio_output_per_mtok is not None
                                   else row.output_per_mtok)
                cost = self._sum([(fresh - audio_in - image_in, row.input_per_mtok, MILLION),
                                  (audio_in, audio_in_price, MILLION),
                                  (image_in, image_in_price, MILLION),
                                  (cached, cached_price, MILLION),
                                  (written - written_1h, write_price, MILLION),
                                  (written_1h, write_1h_price, MILLION),
                                  (output - audio_out if output is not None else None, row.output_per_mtok, MILLION),
                                  (audio_out, audio_out_price, MILLION)])
                if cost is not None and get('fast'):
                    # Anthropic fast mode: token prices times the row's multiplier (unpriced without one)
                    cost = cost * row.fast_multiplier if row.fast_multiplier is not None else None
                if cost is not None and get('web_search_requests'):
                    searches = self._sum([(get('web_search_requests'), row.per_web_search_request, Decimal(1))])
                    cost = cost + searches if searches is not None else None
                if cost is not None:
                    return Priced(cost, 'tokens')
        units: list[tuple[str, object, Decimal | None, Decimal]] = []
        if get('characters') is not None:
            units.append(('characters', get('characters'), row.per_1k_chars, THOUSAND))
        if get('images') is not None:
            units.append(('images', get('images'), row.per_image, Decimal(1)))
        if get('audio_seconds') is not None:
            units.append(('audio_seconds:provider', get('audio_seconds'), row.per_audio_minute, SIXTY))
        elif upload_audio_seconds is not None and row.per_audio_minute is not None:
            units.append(('audio_seconds:upload', upload_audio_seconds, row.per_audio_minute, SIXTY))
        if not units or any(price is None for _, _, price, _ in units):
            return Priced(None, UNMEASURED)
        cost = self._sum([(quantity, price, unit) for _, quantity, price, unit in units])
        return Priced(cost, '+'.join(name for name, _, _, _ in units))

    def cost(self, ref, usage, *, upload_audio_seconds: float | None = None) -> Decimal | None:
        """The settled cost alone (price(...).cost); None when unpriced or nothing priceable was reported."""
        return self.price(ref, usage, upload_audio_seconds=upload_audio_seconds).cost


__all__ = ['PriceTable', 'Priced', 'RESERVATION', 'UNMEASURED', 'UNPRICED', 'money', 'specificity']
