"""Handle affinity for provider-side objects (OpenAI Responses created with store=true).

A binding maps a provider object id to the upstream and credential that created it, to the key that asked for it and to
the served model reference (R17): TITANOS_HOME/broker/handles.json
{"version": 2, "handles": {"responses:<id>": {upstream_id, credential_id, key_id, model, created_at, expires_at}}}.
The version-1 layout (the flat {"responses:<id>": {...}} mapping without a model) is still read; its bindings carry
model None and are rewritten in the version-2 layout on the next write. TTL 30 days, at most 10,000 entries (the oldest
go first). The table is read lazily on first use and written owner-only with an atomic replace; constructing a
HandleStore touches no file.

Checked for the handle-bound routes (retrieve, delete, cancel, input_items) AND for previous_response_id in new Responses
requests: a caller that does not own the handle gets 404 response_not_found (never 403, so ids are not an oracle), before
any egress. Admin principals (bootstrap, or a key holding admin) may use any handle; anonymous loopback principals own the
handles they created without a key (key_id None).
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import BrokerError
from .keys import write_private
from .scopes import allows

TTL_SECONDS = 30 * 86400.0
VERSION = 2
MAX_ENTRIES = 10_000
KINDS = ('responses', 'native')  # native: provider objects created through the R32 native passthrough (native.py)
MAX_ID = 256


@dataclass(frozen=True)
class Binding:
    kind: str
    provider_id: str
    upstream_id: str
    credential_id: str | None
    key_id: str | None
    created_at: float
    expires_at: float
    model: str | None = None  # the served model reference (None for bindings read from the version-1 layout)

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop('kind')
        data.pop('provider_id')
        return data


def _check(kind: str, provider_id: str) -> str:
    if kind not in KINDS:
        raise ValueError(f'Unknown handle kind: {kind!r}')
    if not isinstance(provider_id, str) or not provider_id or len(provider_id) > MAX_ID:
        raise ValueError('handle ids are 1-256 characters')
    return f'{kind}:{provider_id}'


def is_admin(principal) -> bool:
    return principal is not None and (principal.kind == 'bootstrap' or allows(principal, 'admin'))


class HandleStore:
    """Bounded, persisted handle table. Thread-safe."""

    def __init__(self, path, *, ttl_seconds: float = TTL_SECONDS, max_entries: int = MAX_ENTRIES,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, dict] | None = None

    # -- storage ---------------------------------------------------------------------------------------------------------
    def _load(self) -> dict[str, dict]:
        if self._entries is None:
            entries: dict[str, dict] = {}
            try:
                document = json.loads(self.path.read_text(encoding='utf-8'))
            except FileNotFoundError:
                document = {}
            except (OSError, ValueError):
                document = {}  # unreadable: nothing is bound, so every handle answers 404 to non-admin callers
            if isinstance(document, dict) and 'version' in document:
                # version 2: {"version": 2, "handles": {...}}; an unknown version binds nothing (fail closed)
                document = document.get('handles') if document.get('version') == VERSION else {}
            if isinstance(document, dict):
                for key, value in document.items():
                    kind, _, provider_id = key.partition(':') if isinstance(key, str) else ('', '', '')
                    if kind not in KINDS or not provider_id or not isinstance(value, dict):
                        continue
                    try:
                        model = value.get('model')
                        entries[key] = {'upstream_id': str(value['upstream_id']),
                                        'credential_id': value.get('credential_id'),
                                        'key_id': value.get('key_id'),
                                        'created_at': float(value['created_at']),
                                        'expires_at': float(value['expires_at']),
                                        'model': model if isinstance(model, str) and model else None}
                    except (KeyError, TypeError, ValueError):
                        continue
            self._entries = entries
        return self._entries

    def _sweep(self, now: float) -> None:
        entries = self._entries
        for key in [k for k, v in entries.items() if v['expires_at'] <= now]:
            del entries[key]
        excess = len(entries) - self.max_entries
        if excess > 0:
            for key in sorted(entries, key=lambda k: entries[k]['created_at'])[:excess]:
                del entries[key]

    def _save(self) -> None:
        data = json.dumps({'version': VERSION, 'handles': self._entries}, sort_keys=True,
                          separators=(',', ':')).encode('utf-8')
        write_private(self.path, data)

    # -- API -------------------------------------------------------------------------------------------------------------
    def bind(self, kind: str, provider_id: str, upstream_id: str, credential_id: str | None, key_id: str | None,
             model: str | None = None) -> Binding:
        """Record (or refresh) the owner of a provider object and the model reference that served it."""
        key = _check(kind, provider_id)
        with self._lock:
            entries = self._load()
            now = self._clock()
            entries.pop(key, None)
            entries[key] = {'upstream_id': upstream_id, 'credential_id': credential_id, 'key_id': key_id,
                            'created_at': now, 'expires_at': now + self.ttl_seconds, 'model': model}
            self._sweep(now)
            self._save()
            return Binding(kind, provider_id, **entries[key])

    def owner(self, kind: str, provider_id: str) -> Binding | None:
        key = _check(kind, provider_id)
        with self._lock:
            entries = self._load()
            entry = entries.get(key)
            if entry is None:
                return None
            if entry['expires_at'] <= self._clock():
                entries.pop(key, None)
                return None
            return Binding(kind, provider_id, **entry)

    def forget(self, kind: str, provider_id: str) -> None:
        key = _check(kind, provider_id)
        with self._lock:
            entries = self._load()
            if entries.pop(key, None) is not None:
                self._save()

    def __len__(self) -> int:
        with self._lock:
            return len(self._load())

    def owned(self, principal, kind: str, provider_id: str) -> Binding:
        """The binding when principal may use it; 404 response_not_found otherwise (unknown, expired or foreign)."""
        try:
            binding = self.owner(kind, provider_id)
        except ValueError:
            binding = None
        if binding is None:
            raise BrokerError('Response not found', 'response_not_found', origin='policy')
        if is_admin(principal):
            return binding
        key_id = getattr(principal, 'key_id', None)
        if binding.key_id != key_id:
            raise BrokerError('Response not found', 'response_not_found', origin='policy')
        return binding


__all__ = ['Binding', 'HandleStore', 'KINDS', 'MAX_ENTRIES', 'TTL_SECONDS', 'VERSION', 'is_admin']
