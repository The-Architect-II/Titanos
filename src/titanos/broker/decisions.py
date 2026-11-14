"""Routing decision log: a bounded ring of broker_decision_log_size entries, plus optional JSONL at
TITANOS_HOME/broker/decisions/YYYY-MM-DD.jsonl when broker_decision_log_file is set.

One Decision per request, refusals and explain dry-runs included. Each is scrubbed (redact.scrub_obj) before it is
stored; Decision carries metadata only, never prompts or outputs.

Visibility (query): broker:admin sees every decision; broker:read sees only its own key's decisions; any other
principal is refused with 403 scope_denied.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from datetime import timezone
from pathlib import Path
from typing import Any

from .errors import BrokerError
from .keys import ensure_private_dir
from .redact import scrub_obj
from .scopes import allows
from .types import Decision
from .usage import parse_ts, utc_ts


def _decision_dict(decision) -> dict[str, Any]:
    if isinstance(decision, Decision):
        return decision.to_dict()
    if isinstance(decision, dict):
        return dict(decision)
    raise TypeError('DecisionLog.append takes a Decision')


class DecisionLog:
    """Implements types.DecisionSink. path is the JSONL directory, or None for the ring only."""

    def __init__(self, size: int = 1000, path=None):
        self.size = max(1, int(size))
        self.path = Path(path) if path is not None else None
        self._ring: deque[dict[str, Any]] = deque(maxlen=self.size)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._ring)

    def append(self, decision: Decision) -> dict[str, Any]:
        """Scrub and store one decision; returns the stored form."""
        stored = scrub_obj(json.loads(json.dumps(_decision_dict(decision), default=str)))
        if not stored.get('ts'):
            stored['ts'] = utc_ts()
        with self._lock:
            self._ring.append(stored)
            if self.path is not None:
                day = parse_ts(stored['ts']).astimezone(timezone.utc).date().isoformat()
                ensure_private_dir(self.path)
                with (self.path / f'{day}.jsonl').open('a', encoding='utf-8', newline='\n') as handle:
                    handle.write(json.dumps(stored, ensure_ascii=False, separators=(',', ':')) + '\n')
        return stored

    def query(self, *, limit: int = 100, key_id: str | None = None, model: str | None = None, since=None,
              principal) -> list[dict[str, Any]]:
        """Newest first. model matches the requested model or the served ref; since is a datetime, date or ISO text."""
        if principal is None or not allows(principal, 'broker:read'):
            raise BrokerError('Reading decisions needs broker:read', 'scope_denied', origin='policy')
        if not allows(principal, 'broker:admin'):
            own = principal.key_id
            if own is None or (key_id is not None and key_id != own):
                return []
            key_id = own
        limit = max(0, min(int(limit), self.size))
        threshold = None
        if since is not None and since != '':
            threshold = parse_ts(since)
        with self._lock:
            items = list(self._ring)
        out = []
        for item in reversed(items):
            if len(out) >= limit:
                break
            if key_id is not None and item.get('key_id') != key_id:
                continue
            if model is not None:
                served = (item.get('served') or {}).get('ref') if isinstance(item.get('served'), dict) else None
                if item.get('requested_model') != model and served != model:
                    continue
            if threshold is not None:
                try:
                    if parse_ts(item.get('ts')) < threshold:
                        continue
                except (TypeError, ValueError):
                    continue
            out.append(item)
        return out


__all__ = ['DecisionLog']
