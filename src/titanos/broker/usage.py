"""Usage ledger: one JSONL file per UTC day at TITANOS_HOME/broker/usage/YYYY-MM-DD.jsonl, in-memory rollups by
(key, upstream, served model, surface, day), and retention (broker_usage_retention_days).

Records are metadata only (never prompts or outputs) and every string is scrubbed before it is written. The directory
is created lazily on the first record, owner-only. Local targets record their usage with cost null.
"""
from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .keys import ensure_private_dir
from .redact import scrub_obj

_DAY_FILE = re.compile(r'(\d{4}-\d{2}-\d{2})\.jsonl')
GROUPS = {'key': 'key_id', 'upstream': 'upstream', 'model': 'served_ref', 'surface': 'surface', 'day': 'day'}
_CELL = ('key_id', 'upstream', 'served_ref', 'surface', 'day')
_COUNTERS = ('requests', 'input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_tokens', 'characters',
             'images', 'unpriced_requests', 'errors')


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utc_ts(moment: datetime | None = None) -> str:
    """ISO 8601 UTC with millisecond precision and a Z suffix (the ts form of usage, decisions and audit)."""
    moment = (moment or _utcnow()).astimezone(timezone.utc)
    return moment.strftime('%Y-%m-%dT%H:%M:%S.') + f'{moment.microsecond // 1000:03d}Z'


def parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    text = str(value).strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def as_date(value) -> date | None:
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip()[:10])


@dataclass(kw_only=True)
class UsageRecord:
    """data_model (4). cost_usd is a Decimal string or None; cost_basis is the matched price pattern or 'unpriced';
    price_basis says which quantities were priced (pricing.Priced.basis) or 'reservation' (the reserved estimate stood)."""
    ts: str
    request_id: str
    decision_id: str | None = None
    key_id: str | None = None
    role: str = 'standalone'
    surface: str = ''
    target_class: str = 'local'
    upstream: str | None = None
    served_ref: str | None = None
    upstream_model: str | None = None
    upstream_request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    characters: int | None = None
    audio_seconds: float | None = None
    images: int | None = None
    cost_usd: str | None = None
    cost_basis: str = 'unpriced'
    # what cost_usd was computed from (Decision.price_basis): tokens | characters | images | audio_seconds:provider |
    # audio_seconds:upload | reservation | unmeasured | unpriced; None in records written before this field existed
    price_basis: str | None = None
    estimated: bool = False
    outcome: str = 'success'
    status: int | None = None
    ttfb_ms: float | None = None
    total_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'UsageRecord':
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    @property
    def day(self) -> str:
        return parse_ts(self.ts).astimezone(timezone.utc).date().isoformat()


def _decimal(value) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


class UsageLedger:
    """Implements types.UsageSink. retention_days bounds both the files and the in-memory cells."""

    def __init__(self, directory, *, retention_days: int = 90, clock: Callable[[], datetime] = _utcnow):
        self.directory = Path(directory)
        self.retention_days = int(retention_days)
        self._clock = clock
        self._lock = threading.Lock()
        self._cells: dict[tuple, dict[str, Any]] | None = None

    # -- cells ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _new_cell() -> dict[str, Any]:
        cell = {name: 0 for name in _COUNTERS}
        cell['audio_seconds'] = 0.0
        cell['cost_usd'] = Decimal(0)
        cell['tokens'] = 0
        return cell

    def _add(self, data: dict[str, Any]) -> None:
        try:
            day = parse_ts(data['ts']).astimezone(timezone.utc).date().isoformat()
        except (KeyError, TypeError, ValueError):
            return
        key = (data.get('key_id'), data.get('upstream'), data.get('served_ref'), data.get('surface'), day)
        cell = self._cells.get(key)
        if cell is None:
            cell = self._cells[key] = self._new_cell()
        cell['requests'] += 1
        for name in ('input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_tokens', 'characters', 'images'):
            value = data.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cell[name] += int(value)
        cell['tokens'] += int(data.get('input_tokens') or 0) + int(data.get('output_tokens') or 0)
        seconds = data.get('audio_seconds')
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            cell['audio_seconds'] += float(seconds)
        cost = _decimal(data.get('cost_usd'))
        if cost is not None:
            cell['cost_usd'] += cost
        if data.get('cost_basis', 'unpriced') == 'unpriced' and data.get('target_class') in ('cloud', 'router'):
            cell['unpriced_requests'] += 1
        if data.get('outcome', 'success') != 'success':
            cell['errors'] += 1

    def _day_files(self) -> list[tuple[date, Path]]:
        if not self.directory.is_dir():
            return []
        out = []
        for path in self.directory.iterdir():
            found = _DAY_FILE.fullmatch(path.name)
            if found:
                try:
                    out.append((date.fromisoformat(found.group(1)), path))
                except ValueError:
                    continue
        return sorted(out)

    def _ensure_loaded(self) -> None:
        if self._cells is not None:
            return
        self._cells = {}
        cutoff = self._cutoff()
        for day, path in self._day_files():
            if day < cutoff:
                continue
            with path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(data, dict):
                        self._add(data)

    def _cutoff(self) -> date:
        return self._clock().astimezone(timezone.utc).date() - timedelta(days=max(0, self.retention_days))

    # -- API -----------------------------------------------------------------------------------------------------------
    def record(self, rec: UsageRecord | dict) -> None:
        data = rec.to_dict() if isinstance(rec, UsageRecord) else dict(rec)
        data = scrub_obj(data)
        day = parse_ts(data['ts']).astimezone(timezone.utc).date().isoformat()
        line = json.dumps(data, ensure_ascii=False, separators=(',', ':'), default=str) + '\n'
        with self._lock:
            self._ensure_loaded()
            ensure_private_dir(self.directory)
            with (self.directory / f'{day}.jsonl').open('a', encoding='utf-8', newline='\n') as handle:
                handle.write(line)
            self._add(data)

    def rollup(self, group_by: list[str] | tuple[str, ...] = ('key', 'upstream', 'model', 'day'), since=None, until=None,
               key_id: str | None = None) -> list[dict[str, Any]]:
        """Rows grouped by any of key, upstream, model, surface and day (inclusive UTC date bounds; key_id filters).
        Each row carries the group fields under their group_by names plus requests, token, character, audio and image
        sums, cost_usd (Decimal string), unpriced_requests and errors."""
        unknown = [g for g in group_by if g not in GROUPS]
        if unknown:
            raise ValueError('group_by accepts key, upstream, model, surface and day; not ' + ', '.join(unknown))
        since, until = as_date(since), as_date(until)
        rows: dict[tuple, dict[str, Any]] = {}
        with self._lock:
            self._ensure_loaded()
            for cell_key, cell in self._cells.items():
                values = dict(zip(_CELL, cell_key))
                day = date.fromisoformat(values['day'])
                if (since and day < since) or (until and day > until):
                    continue
                if key_id is not None and values['key_id'] != key_id:
                    continue
                group = tuple(values[GROUPS[g]] for g in group_by)
                row = rows.get(group)
                if row is None:
                    row = rows[group] = self._new_cell()
                for name in (*_COUNTERS, 'audio_seconds', 'cost_usd', 'tokens'):
                    row[name] += cell[name]
        out = []
        for group, row in sorted(rows.items(), key=lambda item: tuple('' if v is None else str(v) for v in item[0])):
            entry = dict(zip(group_by, group))
            entry.update({name: row[name] for name in _COUNTERS})
            entry['audio_seconds'] = round(row['audio_seconds'], 3)
            entry['cost_usd'] = format(row['cost_usd'], 'f')
            out.append(entry)
        return out

    def totals(self, *, key_id: str, since=None, until=None) -> tuple[Decimal, int]:
        """(USD spent, input+output tokens) for one key over an inclusive UTC date range (Limiter seeding)."""
        since, until = as_date(since), as_date(until)
        usd, tokens = Decimal(0), 0
        with self._lock:
            self._ensure_loaded()
            for cell_key, cell in self._cells.items():
                if cell_key[0] != key_id:
                    continue
                day = date.fromisoformat(cell_key[4])
                if (since and day < since) or (until and day > until):
                    continue
                usd += cell['cost_usd']
                tokens += cell['tokens']
        return usd, tokens

    def prune(self, retention_days: int | None = None) -> int:
        """Delete day files (and drop cells) older than retention_days before today (UTC). Returns files removed."""
        days = self.retention_days if retention_days is None else int(retention_days)
        cutoff = self._clock().astimezone(timezone.utc).date() - timedelta(days=max(0, days))
        removed = 0
        with self._lock:
            for day, path in self._day_files():
                if day < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            if self._cells is not None:
                for cell_key in [k for k in self._cells if date.fromisoformat(k[4]) < cutoff]:
                    del self._cells[cell_key]
        return removed


__all__ = ['GROUPS', 'UsageLedger', 'UsageRecord', 'as_date', 'parse_ts', 'utc_ts']
