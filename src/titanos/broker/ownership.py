"""Owner binding for things that outlive a request: /v1/files artifacts, jobs and Responses ids.

- Bound: artifacts created on the request path while a key principal is active (context.current_principal), job-produced
  artifacts (bound at job completion from the job record's owner_key_id, because job threads do not inherit the
  request context), and job ids.
- Persisted at TITANOS_HOME/broker/ownership.json as {"<kind>:<name>": {key_id, created_at}}, written atomically and
  owner-only; loaded lazily on first use. TTL defaults to artifact_ttl_seconds; at most 10,000 entries per kind (the
  oldest are evicted).
- allows(): admin, bootstrap and anonymous loopback see everything; a key sees what it owns; an unbound name is visible
  only to admin, bootstrap and anonymous loopback. Callers answer a refusal with 404, never 403.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from .keys import write_private
from .scopes import allows as scope_allows
from .types import Principal

Kind = Literal['artifact', 'job', 'response']
KINDS = ('artifact', 'job', 'response')
MAX_PER_KIND = 10_000
MAX_NAME = 256


def _key_id(principal_or_key_id) -> str | None:
    if principal_or_key_id is None:
        return None
    if isinstance(principal_or_key_id, Principal):
        return principal_or_key_id.key_id
    value = str(principal_or_key_id)
    return value or None


class OwnershipStore:
    def __init__(self, path, *, ttl_seconds: float = 86400.0, ttl_by_kind: dict[str, float] | None = None,
                 max_per_kind: int = MAX_PER_KIND, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.ttl_seconds = float(ttl_seconds)
        self.ttl_by_kind = dict(ttl_by_kind or {})
        self.max_per_kind = int(max_per_kind)
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, dict] | None = None

    @staticmethod
    def _check(kind: str, name: str) -> str:
        if kind not in KINDS:
            raise ValueError(f'Unknown ownership kind: {kind!r}')
        if not isinstance(name, str) or not name or len(name) > MAX_NAME:
            raise ValueError('ownership names are 1-256 characters')
        return f'{kind}:{name}'

    def _ttl(self, kind: str) -> float:
        return float(self.ttl_by_kind.get(kind, self.ttl_seconds))

    def _load(self) -> dict[str, dict]:
        if self._entries is None:
            entries: dict[str, dict] = {}
            try:
                document = json.loads(self.path.read_text(encoding='utf-8'))
            except FileNotFoundError:
                document = {}
            except (OSError, ValueError):
                document = {}  # unreadable: every name is unbound (visible to admin principals only)
            if isinstance(document, dict):
                for key, value in document.items():
                    if (isinstance(key, str) and ':' in key and isinstance(value, dict)
                            and isinstance(value.get('key_id'), str) and isinstance(value.get('created_at'), (int, float))):
                        entries[key] = {'key_id': value['key_id'], 'created_at': float(value['created_at'])}
            self._entries = entries
        return self._entries

    def _expired(self, key: str, entry: dict, now: float) -> bool:
        kind = key.split(':', 1)[0]
        return now - entry['created_at'] >= self._ttl(kind)

    def _sweep(self, now: float) -> None:
        entries = self._entries
        for key in [k for k, v in entries.items() if self._expired(k, v, now)]:
            del entries[key]
        for kind in KINDS:
            keys = [k for k in entries if k.startswith(kind + ':')]
            excess = len(keys) - self.max_per_kind
            if excess > 0:
                for key in sorted(keys, key=lambda k: entries[k]['created_at'])[:excess]:
                    del entries[key]

    def _save(self) -> None:
        data = json.dumps(self._entries, sort_keys=True, separators=(',', ':')).encode('utf-8')
        write_private(self.path, data)

    def bind(self, kind: Kind, name: str, principal_or_key_id) -> bool:
        """Bind name to the key behind principal_or_key_id (a Principal or a key id). Non-key principals (bootstrap,
        anonymous loopback) bind nothing. A live binding to a different key is never moved (returns False): a second
        binder cannot take a name from its owner; rebinding by the same key refreshes it, and an expired or unbound name
        may be bound again. Returns whether a binding was written."""
        key = self._check(kind, name)
        key_id = _key_id(principal_or_key_id)
        if key_id is None:
            return False
        with self._lock:
            entries = self._load()
            now = self._clock()
            existing = entries.get(key)
            if existing is not None and not self._expired(key, existing, now) and existing['key_id'] != key_id:
                return False
            entries.pop(key, None)
            entries[key] = {'key_id': key_id, 'created_at': now}
            self._sweep(now)
            self._save()
        return True

    def owner(self, kind: Kind, name: str) -> str | None:
        key = self._check(kind, name)
        with self._lock:
            entry = self._load().get(key)
            if entry is None or self._expired(key, entry, self._clock()):
                return None
            return entry['key_id']

    def unbind(self, kind: Kind, name: str) -> None:
        key = self._check(kind, name)
        with self._lock:
            if self._load().pop(key, None) is not None:
                self._save()

    def allows(self, principal: Principal | None, kind: Kind, name: str) -> bool:
        if principal is None:
            return False
        if principal.kind in ('bootstrap', 'anonymous_loopback') or scope_allows(principal, 'admin'):
            return True
        owner = self.owner(kind, name)
        return owner is not None and owner == principal.key_id


__all__ = ['KINDS', 'MAX_PER_KIND', 'OwnershipStore']
