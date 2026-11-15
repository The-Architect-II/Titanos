"""Inbound client keys: the table at TITANOS_HOME/broker/keys.json.

- Token: tk_<id>_<secret>. id is 10 characters of lowercase base32 ([a-z2-7]); secret is 43 characters of base64url from
  32 os-random bytes. keys.json stores only the hex SHA-256 of the whole token. A slow KDF is unnecessary for 256-bit
  random secrets: no dictionary exists to search.
- Verification looks the id up, then compares digests with hmac.compare_digest; an unknown id runs a dummy compare so the
  work done does not depend on whether the id exists.
- The secret is returned exactly once, by mint() and rotate(); it is registered with redact.REGISTRY at that moment.
- The table is re-read when keys.json's mtime, size or file id changes (checked at most once per second), so offline CLI
  mints, revocations and rotations take effect without a restart. Writes are atomic replaces under the FileLock shared
  with the CLI (config.broker_lock), with an owner-only ACL.
- Any record, including expired and revoked ones, keeps enforcement on (has_records). Only the offline
  `titanos keys reset --keyless` (reset_keyless) returns the instance to keyless mode, and it is audited. Enforcement is
  also persisted in a marker file beside the table (keys.enforced, written with the first record, removed only by
  reset_keyless), so a keys.json that disappears any other way (a manual delete, sync or backup tooling) keeps
  enforcement on instead of silently reopening keyless loopback admin.
- Every key change is audited inside the table lock BEFORE keys.json is replaced: when the audit cannot be appended
  (an unreadable chain tail, a lock timeout) the change is refused and nothing is persisted, so there is never an
  unaudited change or an orphan minted key whose token was not returned.
- A transient read error (OSError, e.g. a Windows sharing violation) fails closed only until the next one-second
  recheck; malformed content stays refused until the file changes.
- Broker role: a key holding an egress scope (admin, cloud, cloud:<upstream>, cloud:store) needs a budget and an
  expires_at at most 365 days ahead (400 budget_required).
- Revoking (or narrowing, or expiring) the last active admin-capable key on a non-loopback bind without a bootstrap key
  answers 409 last_key_required.
- last_used_at is written at most once per minute per key, and only when both the in-process and the file lock are free
  at that moment (never waits), so authentication is not stalled by a CLI holding the lock.
Nothing here performs network I/O.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import broker_lock, replace_with_retry
from .errors import BrokerError
from .redact import REGISTRY, looks_secret
from .role import loopback_host
from .scopes import valid_scope
from .types import (LOCATIONS, TARGET_CLASSES, KeyBudget, KeyConstraints, KeyLimits, KeyStoreSummary, Principal)

log = logging.getLogger('titanos.broker')

TOKEN_RE = '^tk_([a-z2-7]{10})_([A-Za-z0-9_-]{43})$'
_TOKEN = re.compile(TOKEN_RE)
_ID = re.compile(r'[a-z2-7]{10}')
_ID_ALPHABET = 'abcdefghijklmnopqrstuvwxyz234567'
_DUMMY_DIGEST = hashlib.sha256(b'titanos-keys-dummy-compare').hexdigest()
ADMIN_CAPABLE_SCOPES = frozenset({'admin', 'keys:admin'})
MAX_EXPIRY_DAYS_BROKER = 365
LAST_USED_INTERVAL = 60.0
RECHECK_SECONDS = 1.0
_NAME = re.compile(r'[^\x00-\x1f\x7f]{1,64}')
_Strict = ConfigDict(extra='forbid', hide_input_in_errors=True, allow_inf_nan=False)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip()
        moment = datetime.fromisoformat(text[:-1] + '+00:00' if text.endswith('Z') else text)
    if moment.tzinfo is None:
        raise ValueError('timestamps must carry a timezone (use UTC, e.g. 2026-12-31T00:00:00Z)')
    return moment.astimezone(timezone.utc).replace(microsecond=0)


def token_digest(token: str) -> str:
    """Hex SHA-256 of a whole token (what keys.json stores)."""
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def parse_token(token) -> str | None:
    """The key id of a tk_ token, or None when the text is not tk_-shaped."""
    if not isinstance(token, str):
        return None
    found = _TOKEN.fullmatch(token)
    return found.group(1) if found else None


def new_token(key_id: str | None = None) -> tuple[str, str]:
    """(id, token) with a fresh 256-bit secret; a new id unless key_id is given (rotation keeps the id)."""
    if key_id is None:
        key_id = ''.join(secrets.choice(_ID_ALPHABET) for _ in range(10))
    secret = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b'=').decode('ascii')
    token = f'tk_{key_id}_{secret}'
    if not _TOKEN.fullmatch(token):  # pragma: no cover - the construction always matches
        raise RuntimeError('token construction failed')
    return key_id, token


def egress_scope(scope: str) -> bool:
    """Scopes that can reach a cloud provider: admin (covers everything), cloud, cloud:<upstream> and cloud:store."""
    return scope == 'admin' or scope == 'cloud' or scope.startswith('cloud:')


# ---------------------------------------------------------------------------------------------------------------------
# Owner-only file helpers (shared by the broker's other state writers: audit, usage, decisions, ownership).
# They delegate to vault.acl (protected owner DACL on Windows, 0600/0700 on POSIX) when that module is present; without
# it they fall back to POSIX modes, and on Windows to the inherited ACL of TITANOS_HOME.

def _acl():
    try:
        from .vault import acl  # type: ignore[attr-defined]
    except ImportError:
        return None
    return acl


_PREPARED: set[str] = set()


def ensure_private_dir(path) -> Path:
    """Create path (and parents) and make the directory owner-only. Idempotent; the ACL is applied once per directory
    per process (a directory removed meanwhile is prepared again). On POSIX an existing broker directory that is group-
    or world-accessible (for example one created by config.broker_lock's plain mkdir) is tightened to 0700."""
    path = Path(path)
    key = str(path)
    if key in _PREPARED and path.is_dir():
        return path
    created = not path.exists()
    path.mkdir(parents=True, exist_ok=True)
    acl = _acl()
    if acl is not None and hasattr(acl, 'make_owner_only'):
        try:
            acl.make_owner_only(path, directory=True)
        except OSError as exc:  # pragma: no cover - platform specific
            log.warning('could not restrict broker directory permissions: %s', type(exc).__name__)
    elif os.name == 'posix' and (created or path.stat().st_mode & 0o077):
        os.chmod(path, 0o700)
    _PREPARED.add(key)
    return path


def write_private(path, data: bytes) -> None:
    """Atomically replace path with data, owner-only, with the bounded Windows replace retry."""
    path = Path(path)
    ensure_private_dir(path.parent)
    acl = _acl()
    if acl is not None and hasattr(acl, 'write_owner_only'):
        acl.write_owner_only(path, data)
        return
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        if os.name == 'posix':
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def is_private(path) -> bool | None:
    """True/False when the platform can tell (vault.acl, or POSIX modes); None when it cannot."""
    acl = _acl()
    if acl is not None and hasattr(acl, 'is_owner_only'):
        return bool(acl.is_owner_only(Path(path)))
    if os.name == 'posix':
        mode = Path(path).stat().st_mode
        return mode & 0o077 == 0
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Request models (HTTP bodies and CLI flags validate through these).

Location = Literal['local_only', 'prefer_local', 'auto', 'cloud_only']
TargetClass = Literal['local', 'cloud', 'router']


def _no_secret(value: str | None, what: str) -> str | None:
    if value is not None and looks_secret(value):
        raise ValueError(f'{what} must not contain a secret-shaped value')
    return value


class KeyConstraintsSpec(BaseModel):
    model_config = _Strict
    models: list[str] = Field(default_factory=lambda: ['*'], min_length=1, max_length=64)
    upstreams: list[str] = Field(default_factory=lambda: ['*'], min_length=1, max_length=64)
    target_classes: list[TargetClass] = Field(default_factory=lambda: sorted(TARGET_CLASSES), min_length=1)
    locations: list[Location] = Field(default_factory=lambda: list(LOCATIONS), min_length=1)

    @field_validator('models', 'upstreams')
    @classmethod
    def patterns(cls, value):
        for item in value:
            if not isinstance(item, str) or not item or len(item) > 256 or item != item.strip():
                raise ValueError('patterns are non-empty fnmatch strings of at most 256 characters')
            _no_secret(item, 'a constraint pattern')
        return list(dict.fromkeys(value))


class KeyLimitsSpec(BaseModel):
    model_config = _Strict
    rpm: int | None = Field(None, ge=1, le=1_000_000)
    tpm: int | None = Field(None, ge=1, le=10_000_000_000)
    concurrent: int | None = Field(None, ge=1, le=100_000)


class KeyBudgetSpec(BaseModel):
    model_config = _Strict
    usd_per_day: Decimal | None = None
    usd_per_month: Decimal | None = None
    tokens_per_day: int | None = Field(None, ge=1)
    max_request_usd: Decimal | None = None
    hard: bool = True

    @field_validator('usd_per_day', 'usd_per_month', 'max_request_usd')
    @classmethod
    def money(cls, value):
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError('budget amounts are finite nonnegative decimals')
        return value

    def bounded(self) -> bool:
        return any(v is not None for v in (self.usd_per_day, self.usd_per_month, self.tokens_per_day,
                                           self.max_request_usd))


class _KeyFields(BaseModel):
    model_config = _Strict

    @field_validator('name', check_fields=False)
    @classmethod
    def valid_name(cls, value):
        if value is None:
            return value
        if not _NAME.fullmatch(value) or value != value.strip():
            raise ValueError('name is 1-64 printable characters without surrounding whitespace')
        return _no_secret(value, 'name')

    @field_validator('note', check_fields=False)
    @classmethod
    def valid_note(cls, value):
        if value is not None and (len(value) > 512 or any(ord(c) < 32 and c not in '\t' for c in value)):
            raise ValueError('note is at most 512 printable characters')
        return _no_secret(value, 'note')

    @field_validator('scopes', check_fields=False)
    @classmethod
    def valid_scopes(cls, value):
        if value is None:
            return value
        bad = [s for s in value if not valid_scope(s)]
        if bad:
            raise ValueError('unknown scopes: ' + ', '.join(sorted(map(str, bad)))[:200])
        return sorted(set(value))

    @field_validator('expires_at', check_fields=False)
    @classmethod
    def aware(cls, value):
        return _parse_time(value) if value is not None else None


class KeySpec(_KeyFields):
    """POST /api/keys and `titanos keys create`."""
    name: str
    scopes: list[str] = Field(min_length=1, max_length=64)
    constraints: KeyConstraintsSpec = Field(default_factory=KeyConstraintsSpec)
    limits: KeyLimitsSpec | None = None
    budget: KeyBudgetSpec | None = None
    default_location: Location | None = None
    expires_at: datetime | None = None
    note: str | None = None


class KeyPatch(_KeyFields):
    """PATCH /api/keys/{id} and `titanos keys update`: an omitted field is unchanged; an explicit null clears limits,
    budget, default_location, expires_at or note (name, scopes and constraints cannot be null)."""
    name: str | None = None
    scopes: list[str] | None = Field(None, min_length=1, max_length=64)
    constraints: KeyConstraintsSpec | None = None
    limits: KeyLimitsSpec | None = None
    budget: KeyBudgetSpec | None = None
    default_location: Location | None = None
    expires_at: datetime | None = None
    note: str | None = None

    @model_validator(mode='after')
    def required_not_null(self):
        for name in ('name', 'scopes', 'constraints'):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f'{name} cannot be null')
        return self


# ---------------------------------------------------------------------------------------------------------------------

def _budget_json(budget: KeyBudgetSpec | None) -> dict | None:
    if budget is None:
        return None
    out = budget.model_dump(mode='python')
    for name in ('usd_per_day', 'usd_per_month', 'max_request_usd'):
        if out[name] is not None:
            out[name] = format(out[name], 'f')
    return out


@dataclass(frozen=True)
class KeyRecord:
    """One keys.json record. sha256 is the digest of the whole token; the token itself is never stored."""
    id: str
    name: str
    sha256: str
    scopes: tuple[str, ...]
    constraints: dict[str, list[str]]
    limits: dict[str, int | None] | None = None
    budget: dict[str, Any] | None = None
    default_location: str | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None
    created_by: str = ''
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None

    def expired(self, now: datetime | None = None) -> bool:
        return self.expires_at is not None and (now or _utcnow()) >= self.expires_at

    def active(self, now: datetime | None = None) -> bool:
        return not self.revoked and not self.expired(now)

    @property
    def admin_capable(self) -> bool:
        return bool(ADMIN_CAPABLE_SCOPES & set(self.scopes))

    @property
    def egress_capable(self) -> bool:
        return any(egress_scope(s) for s in self.scopes)

    def has_bounded_budget(self) -> bool:
        b = self.budget or {}
        return any(b.get(k) is not None for k in ('usd_per_day', 'usd_per_month', 'tokens_per_day', 'max_request_usd'))

    def to_json(self) -> dict[str, Any]:
        """The keys.json form (includes the digest)."""
        out = {'name': self.name, 'sha256': self.sha256, 'scopes': list(self.scopes), 'constraints': self.constraints,
               'limits': self.limits, 'budget': self.budget, 'default_location': self.default_location,
               'expires_at': _iso(self.expires_at), 'created_at': _iso(self.created_at), 'created_by': self.created_by,
               'revoked_at': _iso(self.revoked_at), 'last_used_at': _iso(self.last_used_at), 'note': self.note}
        out.update(self.extra)
        return out

    def public(self, now: datetime | None = None) -> dict[str, Any]:
        """Metadata only (no digest), for listings, the CLI, Studio and audit."""
        out = self.to_json()
        out.pop('sha256')
        for name in self.extra:
            out.pop(name, None)
        out['id'] = self.id
        out['status'] = 'revoked' if self.revoked else 'expired' if self.expired(now) else 'active'
        return out

    @classmethod
    def from_json(cls, key_id: str, data: dict[str, Any]) -> 'KeyRecord':
        if not _ID.fullmatch(key_id):
            raise ValueError('invalid key id in keys.json')
        digest = data.get('sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError(f'key {key_id}: sha256 must be 64 lowercase hex characters')
        known = {'name', 'sha256', 'scopes', 'constraints', 'limits', 'budget', 'default_location', 'expires_at',
                 'created_at', 'created_by', 'revoked_at', 'last_used_at', 'note'}
        constraints = KeyConstraintsSpec.model_validate(data.get('constraints') or {}).model_dump(mode='json')
        limits = data.get('limits')
        limits = KeyLimitsSpec.model_validate(limits).model_dump() if limits is not None else None
        budget = data.get('budget')
        budget = _budget_json(KeyBudgetSpec.model_validate(budget)) if budget is not None else None
        scopes = data.get('scopes') or []
        if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
            raise ValueError(f'key {key_id}: scopes must be a list of strings')
        location = data.get('default_location')
        if location is not None and location not in LOCATIONS:
            raise ValueError(f'key {key_id}: invalid default_location')
        return cls(id=key_id, name=str(data.get('name') or key_id), sha256=digest, scopes=tuple(sorted(set(scopes))),
                   constraints=constraints, limits=limits, budget=budget, default_location=location,
                   expires_at=_parse_time(data.get('expires_at')), created_at=_parse_time(data.get('created_at')),
                   created_by=str(data.get('created_by') or ''), revoked_at=_parse_time(data.get('revoked_at')),
                   last_used_at=_parse_time(data.get('last_used_at')), note=data.get('note'),
                   extra={k: v for k, v in data.items() if k not in known})


class KeyStore:
    """The inbound key table. Thread-safe; cross-process writes go through the shared FileLock.

    bootstrap_key_present and bind_loopback feed the last-key guard (bind_loopback defaults to settings.host). offline
    is set only by the CLI: reset_keyless() refuses otherwise. clock returns an aware UTC datetime; monotonic paces the
    once-per-second mtime check (both injectable for tests)."""

    def __init__(self, path, settings, audit=None, *, bootstrap_key_present: bool = False,
                 bind_loopback: bool | None = None, offline: bool = False,
                 clock: Callable[[], datetime] | None = None, monotonic: Callable[[], float] | None = None,
                 recheck_seconds: float = RECHECK_SECONDS):
        self.path = Path(path)
        self.settings = settings
        self.audit = audit
        self.bootstrap_key_present = bool(bootstrap_key_present)
        self.bind_loopback = loopback_host(settings.host) if bind_loopback is None else bool(bind_loopback)
        self.offline = offline
        self._clock = clock or _utcnow
        self._monotonic = monotonic or time.monotonic
        self._recheck = recheck_seconds
        self._lock = threading.RLock()
        self._keys: dict[str, KeyRecord] = {}
        self._stamp: tuple | None = None
        self._checked: float | None = None
        self._unreadable = False
        self._marker = False
        self._marker_warned = False
        self._touched: dict[str, float] = {}

    @property
    def marker_path(self) -> Path:
        """The sticky-enforcement marker beside keys.json (keys.enforced)."""
        return self.path.with_name(self.path.stem + '.enforced')

    # -- loading -------------------------------------------------------------------------------------------------------
    def _stat(self):
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size, getattr(st, 'st_ino', 0))

    def _load(self) -> None:
        stamp = self._stat()
        if stamp is None:
            self._keys, self._stamp, self._unreadable = {}, None, False
            return
        try:
            document = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(document, dict) or document.get('version') != 1 or not isinstance(document.get('keys'), dict):
                raise ValueError('keys.json must be {"version": 1, "keys": {...}}')
            keys = {kid: KeyRecord.from_json(kid, data) for kid, data in document['keys'].items()}
        except (OSError, ValueError) as exc:
            # Fail closed: an unreadable table keeps enforcement on and verifies nothing. Bad content stays pinned to
            # its stamp (reloaded when the file changes); a transient read error (OSError) leaves the stamp unset so the
            # next recheck reads the file again.
            log.error('keys.json is unreadable; every key is refused until it is repaired: %s', type(exc).__name__)
            self._keys, self._unreadable = {}, True
            self._stamp = None if isinstance(exc, OSError) else stamp
            return
        self._keys, self._stamp, self._unreadable = keys, stamp, False
        if keys and not self._marker:
            self._mark()  # a table written before the marker existed (or by hand) still makes enforcement sticky

    def _mark(self) -> None:
        """Write the sticky-enforcement marker (best effort: has_records also holds while the table has records)."""
        try:
            if not self.marker_path.exists():
                write_private(self.marker_path, b'Enforcement marker: only `titanos keys reset --keyless` removes it.\n')
            self._marker = True
        except OSError as exc:
            log.warning('could not write the key enforcement marker: %s', type(exc).__name__)

    def _refresh(self, force: bool = False) -> None:
        with self._lock:
            now = self._monotonic()
            if not force and self._checked is not None and now - self._checked < self._recheck:
                return
            self._checked = now
            if force or self._stat() != self._stamp or (self._stamp is None and (self._keys or self._unreadable)):
                self._load()
            self._marker = self.marker_path.exists()
            if self._marker and not self._keys and not self._unreadable and not self._marker_warned and not self.offline:
                self._marker_warned = True
                log.error('keys.json is missing but key enforcement was on; authentication stays enforced until '
                          '`titanos keys reset --keyless` is run offline')

    def _write(self, keys: dict[str, KeyRecord]) -> None:
        document = {'version': 1, 'keys': {kid: rec.to_json() for kid, rec in sorted(keys.items())}}
        data = (json.dumps(document, indent=2, sort_keys=False) + '\n').encode('utf-8')
        if keys and not self._marker:
            self._mark()
        write_private(self.path, data)
        self._keys = dict(keys)
        self._stamp = self._stat()
        self._checked = self._monotonic()
        self._unreadable = False

    def _locked(self, timeout: float = 30):
        return broker_lock(self.path, timeout=timeout)

    def _event(self, kind: str, actor: str, subject: str, detail: dict) -> None:
        """Append the audit record for a change; called under the table lock BEFORE the change is written, so an audit
        failure (AuditChainError, a lock timeout) propagates and nothing is persisted."""
        if self.audit is not None:
            self.audit.event(kind, actor, subject, detail)

    # -- queries -------------------------------------------------------------------------------------------------------
    def has_records(self) -> bool:
        """True when keys.json holds any record (expired and revoked included) or cannot be read: enforcement is sticky."""
        self._refresh()
        return bool(self._keys) or self._unreadable or self._marker

    def has_active(self) -> bool:
        self._refresh()
        now = self._clock()
        return any(rec.active(now) for rec in self._keys.values())

    def get(self, key_id: str) -> KeyRecord:
        self._refresh()
        record = self._keys.get(key_id) if isinstance(key_id, str) else None
        if record is None:
            raise BrokerError('Key not found', 'not_found')
        return record

    def list(self) -> list[KeyRecord]:
        self._refresh()
        return sorted(self._keys.values(), key=lambda r: (r.created_at or datetime.min.replace(tzinfo=timezone.utc), r.id))

    def summary(self) -> KeyStoreSummary:
        self._refresh()
        now = self._clock()
        active = [r for r in self._keys.values() if r.active(now)]
        missing = tuple(sorted(r.id for r in active if r.egress_capable and self._budget_problem(r, now) is not None))
        return KeyStoreSummary(has_records=bool(self._keys) or self._unreadable or self._marker, active=len(active),
                               active_admin=sum(1 for r in active if r.admin_capable),
                               cloud_keys_missing_budget=missing)

    # -- verification --------------------------------------------------------------------------------------------------
    def verify(self, token: str) -> KeyRecord | None:
        """The record for a valid token, None for an unknown id or wrong secret (constant-time compare, dummy compare on
        an unknown id), BrokerError key_revoked / key_expired for a matching token of a revoked or expired key."""
        key_id = parse_token(token)
        digest = token_digest(token) if isinstance(token, str) else _DUMMY_DIGEST
        if key_id is None:
            hmac.compare_digest(_DUMMY_DIGEST, digest)
            return None
        self._refresh()
        record = self._keys.get(key_id)
        expected = record.sha256 if record is not None else _DUMMY_DIGEST
        matched = hmac.compare_digest(expected, digest)
        if record is None or not matched:
            return None
        if record.revoked:
            raise BrokerError('API key has been revoked', 'key_revoked')
        now = self._clock()
        if record.expired(now):
            raise BrokerError('API key has expired', 'key_expired')
        self._touch(record, now)
        return record

    def _touch(self, record: KeyRecord, now: datetime) -> None:
        """Record last_used_at at most once per LAST_USED_INTERVAL per key; never fails the request."""
        mono = self._monotonic()
        last = self._touched.get(record.id)
        if last is not None and mono - last < LAST_USED_INTERVAL:
            return
        if record.last_used_at is not None and (now - record.last_used_at).total_seconds() < LAST_USED_INTERVAL:
            self._touched[record.id] = mono
            return
        self._touched[record.id] = mono
        # Never wait on the authentication path: a busy in-process or file lock skips this minute's write.
        if not self._lock.acquire(blocking=False):
            return
        try:
            with self._locked(timeout=0):
                self._refresh(force=True)
                current = self._keys.get(record.id)
                if current is None or not hmac.compare_digest(current.sha256, record.sha256):
                    return
                keys = dict(self._keys)
                keys[record.id] = replace(current, last_used_at=now.replace(microsecond=0))
                self._write(keys)
        except Exception as exc:  # a busy lock or a read-only home must not fail authentication
            if type(exc).__name__ == 'Timeout':
                log.debug('key table busy; last_used_at skipped this minute')
            else:
                log.warning('could not record key last_used_at: %s', type(exc).__name__)
        finally:
            self._lock.release()

    def to_principal(self, record: KeyRecord) -> Principal:
        c = record.constraints
        constraints = KeyConstraints(models=tuple(c.get('models') or ('*',)), upstreams=tuple(c.get('upstreams') or ('*',)),
                                     target_classes=frozenset(c.get('target_classes') or TARGET_CLASSES),
                                     locations=frozenset(c.get('locations') or LOCATIONS))
        limits = KeyLimits(**record.limits) if record.limits else None
        budget = None
        if record.budget:
            b = record.budget
            money = {k: (Decimal(b[k]) if b.get(k) is not None else None)
                     for k in ('usd_per_day', 'usd_per_month', 'max_request_usd')}
            budget = KeyBudget(tokens_per_day=b.get('tokens_per_day'), hard=bool(b.get('hard', True)), **money)
        kind = 'parent' if 'delegate' in record.scopes else 'key'
        return Principal(id=record.id, kind=kind, scopes=frozenset(record.scopes), constraints=constraints,
                         limits=limits, budget=budget, expires_at=record.expires_at, name=record.name,
                         default_location=record.default_location)

    # -- rules ---------------------------------------------------------------------------------------------------------
    def _budget_problem(self, record: KeyRecord, now: datetime) -> str | None:
        if not record.egress_capable:
            return None
        if not record.has_bounded_budget():
            return 'a key with a cloud scope needs a budget (usd_per_day, usd_per_month, tokens_per_day or max_request_usd)'
        if record.expires_at is None:
            return 'a key with a cloud scope needs expires_at'
        if record.expires_at > now + timedelta(days=MAX_EXPIRY_DAYS_BROKER):
            return f'a key with a cloud scope must expire within {MAX_EXPIRY_DAYS_BROKER} days'
        return None

    def _check_broker_rule(self, record: KeyRecord, now: datetime) -> None:
        if getattr(self.settings, 'role', 'standalone') != 'broker' or not record.active(now):
            return
        problem = self._budget_problem(record, now)
        if problem is not None:
            raise BrokerError(f'Broker role: {problem}', 'budget_required')

    def _check_last_key(self, before: dict[str, KeyRecord], after: dict[str, KeyRecord], now: datetime) -> None:
        if self.bind_loopback or self.bootstrap_key_present:
            return
        had = any(r.active(now) and r.admin_capable for r in before.values())
        has = any(r.active(now) and r.admin_capable for r in after.values())
        if had and not has:
            raise BrokerError('This is the last active admin-capable key on a network bind without TITANOS_API_KEY; '
                              'mint another admin key first', 'last_key_required')

    # -- writes --------------------------------------------------------------------------------------------------------
    def mint(self, spec: KeySpec, *, created_by: str) -> tuple[KeyRecord, str]:
        """Create a key. Returns (record, token); the token is never retrievable again."""
        if not isinstance(spec, KeySpec):
            spec = KeySpec.model_validate(spec)
        now = self._clock().replace(microsecond=0)
        if spec.expires_at is not None and spec.expires_at <= now:
            raise BrokerError('expires_at must be in the future', 'invalid_request')
        with self._lock, self._locked():
            self._refresh(force=True)
            if self._unreadable:
                raise BrokerError('keys.json is unreadable; repair it before minting', 'invalid_request')
            key_id, token = new_token()
            while key_id in self._keys:  # pragma: no cover - 50-bit ids
                key_id, token = new_token()
            record = KeyRecord(id=key_id, name=spec.name, sha256=token_digest(token), scopes=tuple(spec.scopes),
                               constraints=spec.constraints.model_dump(mode='json'),
                               limits=spec.limits.model_dump() if spec.limits is not None else None,
                               budget=_budget_json(spec.budget), default_location=spec.default_location,
                               expires_at=spec.expires_at, created_at=now, created_by=str(created_by), note=spec.note)
            self._check_broker_rule(record, now)
            keys = dict(self._keys)
            keys[key_id] = record
            # audit first: a failed append refuses the mint before anything is persisted
            self._event('key.mint', str(created_by), key_id, {'name': record.name, 'scopes': list(record.scopes),
                                                               'expires_at': _iso(record.expires_at),
                                                               'budget': record.budget is not None})
            self._write(keys)
            REGISTRY.add(token)
        return record, token

    def update(self, key_id: str, patch: KeyPatch, *, actor: str = 'admin') -> KeyRecord:
        if not isinstance(patch, KeyPatch):
            patch = KeyPatch.model_validate(patch)
        now = self._clock().replace(microsecond=0)
        changed = sorted(patch.model_fields_set)
        with self._lock, self._locked():
            self._refresh(force=True)
            current = self._keys.get(key_id)
            if current is None:
                raise BrokerError('Key not found', 'not_found')
            values: dict[str, Any] = {}
            for name in changed:
                value = getattr(patch, name)
                if name == 'scopes':
                    value = tuple(value)
                elif name == 'constraints':
                    value = value.model_dump(mode='json')
                elif name == 'limits':
                    value = value.model_dump() if value is not None else None
                elif name == 'budget':
                    value = _budget_json(value)
                values[name] = value
            updated = replace(current, **values)
            self._check_broker_rule(updated, now)
            keys = dict(self._keys)
            keys[key_id] = updated
            self._check_last_key(self._keys, keys, now)
            self._event('key.update', str(actor), key_id, {'fields': changed})
            self._write(keys)
        return updated

    def revoke(self, key_id: str, *, actor: str = 'admin') -> KeyRecord:
        now = self._clock().replace(microsecond=0)
        with self._lock, self._locked():
            self._refresh(force=True)
            current = self._keys.get(key_id)
            if current is None:
                raise BrokerError('Key not found', 'not_found')
            if current.revoked:
                return current
            updated = replace(current, revoked_at=now)
            keys = dict(self._keys)
            keys[key_id] = updated
            self._check_last_key(self._keys, keys, now)
            self._event('key.revoke', str(actor), key_id, {'name': current.name})
            self._write(keys)
        return updated

    def rotate(self, key_id: str, *, actor: str = 'admin') -> tuple[KeyRecord, str]:
        """A new secret for the same id; the old token stops verifying at once. Returns (record, token) once."""
        with self._lock, self._locked():
            self._refresh(force=True)
            current = self._keys.get(key_id)
            if current is None:
                raise BrokerError('Key not found', 'not_found')
            if current.revoked:
                raise BrokerError('A revoked key cannot be rotated; mint a new key', 'invalid_request')
            _, token = new_token(key_id)
            updated = replace(current, sha256=token_digest(token), last_used_at=None)
            keys = dict(self._keys)
            keys[key_id] = updated
            self._event('key.rotate', str(actor), key_id, {'name': current.name})
            self._write(keys)
            self._touched.pop(key_id, None)
            REGISTRY.add(token)
        return updated, token

    def reset_keyless(self, *, actor: str = 'cli') -> int:
        """Offline only (`titanos keys reset --keyless`): delete keys.json, returning the number of records removed. The
        next server start (or the running server's next mtime check) is keyless again. Audited."""
        if not self.offline:
            raise BrokerError('keys reset --keyless runs only offline from the CLI', 'role_forbids')
        with self._lock, self._locked():
            self._refresh(force=True)
            count = len(self._keys)
            self._event('key.reset_keyless', str(actor), 'keys.json', {'removed': count})
            self.path.unlink(missing_ok=True)
            self.marker_path.unlink(missing_ok=True)
            self._keys, self._stamp, self._unreadable, self._marker = {}, None, False, False
        return count


__all__ = ['ADMIN_CAPABLE_SCOPES', 'KeyBudgetSpec', 'KeyConstraintsSpec', 'KeyLimitsSpec', 'KeyPatch', 'KeyRecord',
           'KeySpec', 'KeyStore', 'MAX_EXPIRY_DAYS_BROKER', 'TOKEN_RE', 'egress_scope', 'ensure_private_dir', 'is_private',
           'new_token', 'parse_token', 'token_digest', 'write_private']
