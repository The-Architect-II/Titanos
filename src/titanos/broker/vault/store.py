"""The Vault: credential records (TITANOS_HOME/broker/vault/index.json), their values in an OS store, a sealed file or an
env:/file: reference, and the only way to turn a record into an outbound credential (handle()).

- index.json holds metadata only (CredentialRecord; never a value). It is written atomically (acl.write_owner_only) under
  a FileLock shared with the CLI (index.json.lock) and re-read whenever its mtime or size changes, so an offline
  `titanos broker credential set` takes effect at the server's next use.
- Backend: TITANOS_BROKER_VAULT_BACKEND. auto picks dpapi on Windows, keychain on macOS and secret-service on Linux when
  available; otherwise auto becomes refs-only (posture 'refs-only': env:/file: references still work and a PUT of a
  secret answers 503 vault_unavailable naming both options). sealed-file is never auto-selected. An explicitly
  configured store that cannot be used (dpapi off Windows, sealed-file without a usable passphrase file, ...) is
  'unavailable', which TITANOS_ROLE=broker refuses. refs-only accepts references only.
- Each record keeps the backend that stored it, so changing the setting never strands existing values.
- Resolved values are wrapped in Secret (registered with redact.REGISTRY on construction) and cached for at most
  broker_secret_cache_seconds. file: references re-check their permissions on every handle(), cached or not.
- env: values are never written anywhere.
- handle() refuses unless the record is validated (ok or skipped), bound to exactly the requested origin, has the
  requested purpose (inference by default) and, when the caller names one, is bound to the requested upstream id;
  validation_handle() serves the credentials:write validate and rotate flows (purpose validate), which re-bind the origin
  through mark_validated(..., origin=).
- seal() makes every handle() and validation_handle() raise vault_sealed (managed role).
- Lazy: constructing a Vault (open_vault) reads and creates nothing; reads never create files; the vault directory, the
  index, the instance id and DPAPI's entropy file are created on the first write.
- All backends are user-scoped: the server and the CLI must run as the same OS user.
- Helper processes (security, secret-tool) get a scrubbed environment that also drops every variable named by an env:
  record (referenced_env_vars()).
- A PUT (put_secret / put_env_ref / put_file_ref) over an existing record replaces it at once with validation
  'unvalidated': the upstream stops serving until the new value validates, and a failed validation does not restore the
  previous source. Only a same-source vault value can be replaced without an outage, through stage_rotation() ->
  validation_handle(staged=True) -> commit_rotation() / discard_rotation(). Callers (the credentials API and the CLI)
  must use rotation for vault values and state the outage for source changes.

There is deliberately no method that returns a credential value: values leave the vault only inside a CredentialHandle,
whose apply() is called by broker/egress.py.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import parse_origin
from ..errors import BrokerError
from ..naming import valid_upstream_id
from ..profiles import AUTH_TEMPLATES
from ..redact import MIN_SECRET_LENGTH, REGISTRY
from . import CredentialHandle, CredentialRecord, Secret, acl, fingerprint, refs, valid_credential_id

log = logging.getLogger('titanos.broker')

INDEX_VERSION = 1
INDEX_FILE = 'index.json'
INSTANCE_ID_FILE = 'instance-id'
STAGED = '.next'
BACKEND_CHOICES = ('auto', 'dpapi', 'keychain', 'secret-service', 'sealed-file', 'refs-only')
STORE_BACKENDS = ('dpapi', 'keychain', 'secret-service', 'sealed-file')
TOUCH_INTERVAL = 60.0
_CONTROL = re.compile(r'[\x00-\x1f\x7f]')
_CODE = re.compile(r'[a-z0-9_]{1,64}')
VAULT_UNAVAILABLE_MESSAGE = ('No credential store is available on this host. Use an env: or file: reference '
                             '(`titanos broker credential set <id> --env VAR` or `--file /abs/path`), or set '
                             'TITANOS_BROKER_VAULT_BACKEND=sealed-file with an owner-only '
                             'TITANOS_BROKER_VAULT_PASSPHRASE_FILE')


def _now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime('%Y-%m-%dT%H:%M:%S.') + f'{now.microsecond // 1000:03d}Z'


def platform_default() -> str:
    return {'win32': 'dpapi', 'darwin': 'keychain'}.get(sys.platform, 'secret-service')


def _invalid(message: str) -> BrokerError:
    return BrokerError(message, 'invalid_request')


class Vault:
    """VaultProtocol over index.json and the configured backend. Thread-safe; safe to share between the server and the
    CLI (separate processes coordinate through the index FileLock)."""

    def __init__(self, settings, home, *, backends: dict[str, Any] | None = None, environ=None, clock=time.monotonic,
                 lock_timeout: float = 30.0):
        self.settings = settings
        self.home = Path(home)
        self.broker_dir = self.home / 'broker'
        self.directory = self.broker_dir / 'vault'
        self.index_path = self.directory / INDEX_FILE
        self.lock_timeout = lock_timeout
        self._environ = os.environ if environ is None else environ
        self._clock = clock
        self._injected = dict(backends or {})
        self._backends: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._index: dict | None = None
        self._stamp = None
        self._cache: dict[tuple[str, bool], tuple[Secret, float, str | None]] = {}
        self._touched: dict[str, float] = {}
        self._instance_id: str | None = None
        self.sealed = False
        self.seal_reason: str | None = None

    # backend selection ------------------------------------------------------------------------------------------
    @property
    def configured_backend(self) -> str:
        return getattr(self.settings, 'broker_vault_backend', 'auto') or 'auto'

    @property
    def backend_name(self) -> str:
        """dpapi | keychain | secret-service | sealed-file | refs-only (configured, or auto with no OS store on this
        host), or 'unavailable' when the explicitly configured store cannot be used (references still work). No I/O
        beyond availability probes."""
        choice = self.configured_backend
        if choice == 'refs-only':
            return 'refs-only'
        name = platform_default() if choice == 'auto' else choice
        if self._backend(name).available():
            return name
        return 'refs-only' if choice == 'auto' else 'unavailable'

    def posture(self) -> str:
        """The value role.check_serve_preconditions(vault_backend=) expects."""
        return self.backend_name

    def _backend(self, name: str):
        if name in self._injected:
            return self._injected[name]
        if name not in self._backends:
            if name == 'dpapi':
                from .dpapi import DpapiBackend
                backend = DpapiBackend(self.directory)
            elif name == 'keychain':
                from .keychain import KeychainBackend
                backend = KeychainBackend(instance_id=self.instance_id, environ=self._environ,
                                          deny=self.referenced_env_vars)
            elif name == 'secret-service':
                from .secret_service import SecretServiceBackend
                backend = SecretServiceBackend(instance_id=self.instance_id, environ=self._environ,
                                               deny=self.referenced_env_vars)
            elif name == 'sealed-file':
                from .sealed_file import SealedFileBackend
                backend = SealedFileBackend(self.directory,
                                            passphrase_file=getattr(self.settings, 'broker_vault_passphrase_file', ''))
            else:
                raise ValueError(f'Unknown vault backend: {name!r}')
            self._backends[name] = backend
        return self._backends[name]

    def _writable_backend(self):
        name = self.backend_name
        if name == 'refs-only' and self.configured_backend == 'refs-only':
            raise BrokerError('This vault accepts env: and file: references only (TITANOS_BROKER_VAULT_BACKEND='
                              'refs-only)', 'vault_unavailable')
        if name == 'refs-only':  # auto found no OS store on this host
            raise BrokerError(VAULT_UNAVAILABLE_MESSAGE, 'vault_unavailable')
        if name == 'unavailable':
            reason = ''
            if self.configured_backend == 'sealed-file':
                why = self._backend('sealed-file').unavailable_reason()
                reason = f' (sealed-file: {why})' if why else ''
            raise BrokerError(VAULT_UNAVAILABLE_MESSAGE + reason, 'vault_unavailable')
        return self._backend(name)

    # instance id ------------------------------------------------------------------------------------------------
    def instance_id(self) -> str:
        """TITANOS_HOME/broker/instance-id: 128 random bits as 32 hex characters, created on first need."""
        if self._instance_id is not None:
            return self._instance_id
        path = self.broker_dir / INSTANCE_ID_FILE
        try:
            value = path.read_text(encoding='ascii').strip()
        except FileNotFoundError:
            value = ''
        if not re.fullmatch(r'[0-9a-f]{32}', value):
            value = os.urandom(16).hex()
            self._ensure_dirs()
            acl.write_owner_only(path, value.encode('ascii') + b'\n')
        self._instance_id = value
        return value

    # index ------------------------------------------------------------------------------------------------------
    def _ensure_dirs(self) -> None:
        for directory in (self.broker_dir, self.directory):
            if not directory.is_dir():
                directory.mkdir(parents=True, exist_ok=True)
                acl.make_owner_only(directory, directory=True)
            elif not acl.is_owner_only(directory):
                acl.make_owner_only(directory, directory=True)

    @staticmethod
    def _empty() -> dict:
        return {'version': INDEX_VERSION, 'install_entropy_ref': 'entropy.bin', 'credentials': {}, 'staging': {}}

    def _read(self) -> dict:
        """The current index (re-read when the file's mtime or size changed). Never creates anything."""
        with self._lock:
            try:
                st = self.index_path.stat()
            except FileNotFoundError:
                if self._stamp is not None or self._index is None:
                    self._cache.clear()
                self._index, self._stamp = self._empty(), None
                return self._index
            stamp = (st.st_mtime_ns, st.st_size)
            if self._index is not None and stamp == self._stamp:
                return self._index
            try:
                document = json.loads(self.index_path.read_text(encoding='utf-8'))
                if document.get('version') != INDEX_VERSION:
                    raise ValueError('unsupported index version')
                credentials = {cid: CredentialRecord.from_dict(raw).to_dict()
                               for cid, raw in document.get('credentials', {}).items()}
                staging = dict(document.get('staging', {}))
            except (OSError, ValueError, TypeError) as exc:
                raise BrokerError(f'The vault index cannot be read ({type(exc).__name__}); restore or remove '
                                  f'{INDEX_FILE}', 'vault_unavailable') from None
            index = self._empty()
            index['credentials'], index['staging'] = credentials, staging
            self._index, self._stamp = index, stamp
            self._cache.clear()
            return index

    def _write(self, index: dict) -> None:
        data = json.dumps(index, indent=2, sort_keys=True).encode('utf-8') + b'\n'
        acl.write_owner_only(self.index_path, data)
        st = self.index_path.stat()
        self._index, self._stamp = index, (st.st_mtime_ns, st.st_size)

    def _file_lock(self, timeout: float | None = None):
        from filelock import FileLock
        self._ensure_dirs()
        return FileLock(str(self.index_path) + '.lock', timeout=self.lock_timeout if timeout is None else timeout)

    def _mutate(self, change, *, timeout: float | None = None):
        """Run change(index) on a fresh copy of the index under both locks, then write it atomically."""
        with self._lock, self._file_lock(timeout):
            self._stamp = None  # force a re-read under the lock
            current = self._read()
            working = json.loads(json.dumps(current))
            result = change(working)
            self._write(working)
            return result

    # validation helpers -----------------------------------------------------------------------------------------
    @staticmethod
    def _check_id(id: str) -> None:
        if not valid_credential_id(id):
            raise _invalid('Credential ids match ^[a-z][a-z0-9_-]{0,63}$')

    @staticmethod
    def _check_origin(origin: str) -> None:
        try:
            canonical = parse_origin(origin, allow_http_lan=True) if isinstance(origin, str) else None
        except ValueError:
            canonical = None
        if canonical != origin:
            raise _invalid('A credential is bound to an exact origin scheme://host:port')

    @classmethod
    def _check_binding(cls, upstream_id: str, origin: str, purpose: str) -> None:
        if not valid_upstream_id(upstream_id):
            raise _invalid('A credential is bound to one upstream id')
        if purpose not in ('inference', 'advisor'):
            raise _invalid('Credential purpose is inference or advisor')
        cls._check_origin(origin)

    @staticmethod
    def _check_secret(secret: str) -> None:
        if not isinstance(secret, str) or len(secret) < MIN_SECRET_LENGTH:
            raise _invalid(f'Secrets shorter than {MIN_SECRET_LENGTH} characters are refused')
        if secret != secret.strip() or _CONTROL.search(secret):
            raise _invalid('Secrets may not carry surrounding whitespace or control characters')

    def _drop_value(self, record: dict | None) -> None:
        """Delete the stored value of a vault-sourced record (best effort; the index no longer points at it)."""
        if not record or record.get('source') != 'vault' or not record.get('backend'):
            return
        try:
            backend = self._backend(record['backend'])
            backend.delete(record['id'])
            backend.delete(record['id'] + STAGED)
        except (BrokerError, OSError) as exc:
            log.warning('vault: could not delete the old value of a credential (%s)', type(exc).__name__)

    def _forget(self, id: str) -> None:
        self._cache.pop((id, False), None)
        self._cache.pop((id, True), None)

    def _refuse_sealed(self) -> None:
        if self.sealed:
            raise BrokerError('The vault is sealed in managed role; credentials are ignored', 'vault_sealed')

    # writes -----------------------------------------------------------------------------------------------------
    def _put(self, id: str, fields: dict, *, upstream_id: str, origin: str, purpose: str,
             before=None) -> CredentialRecord:
        now = _now()
        holder: dict = {}

        def change(index):
            if before is not None:
                before()  # the value write, under the same locks as the index update
            old = index['credentials'].get(id)
            record = CredentialRecord(id=id, upstream_id=upstream_id, origin=origin, purpose=purpose,
                                      created_at=old['created_at'] if old else now,
                                      rotated_at=now if old else None, validation='unvalidated', **fields)
            index['credentials'][id] = record.to_dict()
            holder['staged'] = index['staging'].pop(id, None)
            holder['old'], holder['record'] = old, record
        self._mutate(change)
        old = holder['old']
        if holder['staged'] and old and old.get('backend'):
            try:  # a pending rotation of the replaced value is void
                self._backend(old['backend']).delete(id + STAGED)
            except (BrokerError, OSError) as exc:
                log.warning('vault: could not delete a staged value (%s)', type(exc).__name__)
        if old and old.get('source') == 'vault' and (fields.get('source') != 'vault'
                                                     or old.get('backend') != fields.get('backend')):
            self._drop_value(old)
        self._forget(id)
        return holder['record']

    def put_secret(self, id: str, secret: str, *, upstream_id: str, origin: str,
                   purpose: str = 'inference') -> CredentialRecord:
        """Store a value in the configured store and bind it to upstream_id and origin (validation unvalidated)."""
        self._refuse_sealed()
        self._check_id(id)
        self._check_binding(upstream_id, origin, purpose)
        self._check_secret(secret)
        backend = self._writable_backend()
        REGISTRY.add(secret)
        return self._put(id, {'source': 'vault', 'backend': backend.name, 'fingerprint': fingerprint(secret)},
                         upstream_id=upstream_id, origin=origin, purpose=purpose,
                         before=lambda: backend.store(id, secret.encode('utf-8')))

    def put_env_ref(self, id: str, var: str, *, upstream_id: str, origin: str,
                    purpose: str = 'inference') -> CredentialRecord:
        """Bind an env: reference. Only the variable name is stored; the fingerprint is computed when it is set."""
        self._refuse_sealed()
        self._check_id(id)
        self._check_binding(upstream_id, origin, purpose)
        try:
            refs.check_env_var(var)
        except ValueError as exc:
            raise _invalid(str(exc)) from None
        try:
            fp = fingerprint(refs.resolve_env(var, self._environ))
        except BrokerError:
            fp = None
        return self._put(id, {'source': 'env', 'env_var': var, 'fingerprint': fp}, upstream_id=upstream_id,
                         origin=origin, purpose=purpose)

    def put_file_ref(self, id: str, path: str, *, upstream_id: str, origin: str,
                     purpose: str = 'inference') -> CredentialRecord:
        """Bind a file: reference; the file must already be owner-only (credential_unavailable otherwise)."""
        self._refuse_sealed()
        self._check_id(id)
        self._check_binding(upstream_id, origin, purpose)
        try:
            refs.check_file_path(path)
        except ValueError as exc:
            raise _invalid(str(exc)) from None
        fp = fingerprint(refs.resolve_file(path))
        return self._put(id, {'source': 'file', 'path': str(path), 'fingerprint': fp}, upstream_id=upstream_id,
                         origin=origin, purpose=purpose)

    def mark_validated(self, id: str, ok: bool, code: str | None, fingerprint: str | None, *,
                       origin: str | None = None) -> CredentialRecord:
        """Record a validation outcome: ok (code 'skipped' records 'skipped'), or failed:<code>. origin re-binds the
        record after a successful validation against a changed upstream base_url."""
        self._check_id(id)
        if code is not None and not _CODE.fullmatch(code):
            raise _invalid('Validation codes are short snake_case error codes')
        if origin is not None:
            self._check_origin(origin)
        if fingerprint is not None and not re.fullmatch(r'[0-9a-f]{8}', fingerprint):
            raise _invalid('A fingerprint is 8 hex characters')
        now = _now()

        def change(index):
            raw = index['credentials'].get(id)
            if raw is None:
                raise BrokerError(f'Credential {id} does not exist', 'not_found')
            record = CredentialRecord.from_dict(raw)
            if ok:
                validation = 'skipped' if code == 'skipped' else 'ok'
            else:
                validation = f'failed:{code or "unknown"}'
            record = replace(record, validation=validation, validated_at=now,
                             fingerprint=fingerprint or record.fingerprint,
                             origin=origin if (ok and origin) else record.origin)
            index['credentials'][id] = record.to_dict()
            return record
        record = self._mutate(change)
        if not ok:
            self._forget(id)
        return record

    def invalidate_for_upstream(self, upstream_id: str, *, origin: str | None = None) -> list[str]:
        """Reset validation to 'unvalidated' for every record bound to upstream_id (when origin is given: only those not
        bound to it). Called when an upstream's base_url changes; the ids changed are returned."""
        def change(index):
            changed = []
            for cid, raw in index['credentials'].items():
                if raw['upstream_id'] == upstream_id and (origin is None or raw['origin'] != origin) \
                        and raw['validation'] != 'unvalidated':
                    raw['validation'] = 'unvalidated'
                    changed.append(cid)
            return changed
        if not any(r.upstream_id == upstream_id for r in self.records()):
            return []
        changed = self._mutate(change)
        for cid in changed:
            self._forget(cid)
        return changed

    def delete(self, id: str) -> None:
        self._check_id(id)
        holder = {}

        def change(index):
            raw = index['credentials'].pop(id, None)
            if raw is None:
                raise BrokerError(f'Credential {id} does not exist', 'not_found')
            index['staging'].pop(id, None)
            holder['old'] = raw
        self._mutate(change)
        self._drop_value(holder['old'])
        self._forget(id)

    def stage_rotation(self, id: str, secret: str) -> str:
        """Write a new value to the staging slot (<id>.next) of a vault-sourced record; returns its fingerprint. The
        live value keeps serving until commit_rotation()."""
        self._refuse_sealed()
        self._check_id(id)
        self._check_secret(secret)
        record = self.record(id)
        if record.source != 'vault':
            raise _invalid(f'Credential {id} is an {record.source}: reference; change the referenced value, then '
                           'validate')
        backend = self._backend(record.backend)
        if not backend.available():
            raise BrokerError(VAULT_UNAVAILABLE_MESSAGE, 'vault_unavailable')
        REGISTRY.add(secret)
        fp = fingerprint(secret)

        def change(index):
            if id not in index['credentials']:
                raise BrokerError(f'Credential {id} does not exist', 'not_found')
            backend.store(id + STAGED, secret.encode('utf-8'))
            index['staging'][id] = {'fingerprint': fp, 'staged_at': _now(), 'backend': backend.name}
        self._mutate(change)
        self._cache.pop((id, True), None)
        return fp

    def commit_rotation(self, id: str, *, validation: str = 'ok') -> CredentialRecord:
        """Swap the staged value in atomically (one replace of the stored value), delete the staging slot (the old value
        is overwritten, so it no longer exists) and record the rotation. validation is the staged value's result."""
        self._refuse_sealed()
        self._check_id(id)
        if validation not in ('ok', 'skipped', 'unvalidated'):
            raise _invalid('commit_rotation records ok, skipped or unvalidated')
        now = _now()

        def change(index):
            raw = index['credentials'].get(id)
            staged = index['staging'].get(id)
            if raw is None or staged is None:
                raise BrokerError(f'Credential {id} has no staged rotation', 'not_found')
            backend = self._backend(raw['backend'])
            value = backend.load(id + STAGED)
            backend.store(id, value)
            backend.delete(id + STAGED)
            record = replace(CredentialRecord.from_dict(raw), fingerprint=staged['fingerprint'], rotated_at=now,
                             validation=validation, validated_at=now if validation != 'unvalidated' else None)
            index['credentials'][id] = record.to_dict()
            del index['staging'][id]
            return record
        record = self._mutate(change)
        self._forget(id)
        return record

    def discard_rotation(self, id: str) -> None:
        """Drop a staged value (its validation failed)."""
        self._check_id(id)

        def change(index):
            staged = index['staging'].pop(id, None)
            raw = index['credentials'].get(id)
            if staged is not None and raw is not None and raw.get('backend'):
                self._backend(raw['backend']).delete(id + STAGED)
        self._mutate(change)
        self._cache.pop((id, True), None)

    def touch_due(self, id: str) -> bool:
        last = self._touched.get(id)
        return not self.sealed and (last is None or self._clock() - last >= TOUCH_INTERVAL)

    def touch(self, id: str) -> None:
        """last_used_at, at most once per minute per credential; skipped (never waits) when the index lock is busy."""
        if not self.touch_due(id):
            return
        self._touched[id] = self._clock()
        if not self._lock.acquire(blocking=False):
            return
        try:
            from filelock import Timeout
            try:
                def change(index):
                    if id in index['credentials']:
                        index['credentials'][id]['last_used_at'] = _now()
                self._mutate(change, timeout=0)
            except (Timeout, OSError, BrokerError):
                pass
        finally:
            self._lock.release()

    def seal(self, reason: str) -> None:
        """Refuse every resolution from now on (managed role). Logs one line, without credential names."""
        with self._lock:
            if self.sealed:
                return
            self.sealed = True
            self.seal_reason = reason
            self._cache.clear()
        try:
            count = len(self._read()['credentials'])
        except BrokerError:
            count = 0
        log.warning('%s: %d credentials configured and ignored', reason, count)

    # reads ------------------------------------------------------------------------------------------------------
    def record(self, id: str) -> CredentialRecord:
        raw = self._read()['credentials'].get(id)
        if raw is None:
            raise BrokerError(f'Credential {id} does not exist', 'not_found')
        return CredentialRecord.from_dict(raw)

    def records(self) -> list[CredentialRecord]:
        return [CredentialRecord.from_dict(raw) for _, raw in sorted(self._read()['credentials'].items())]

    def staged(self, id: str) -> dict | None:
        """Staging metadata of a pending rotation ({fingerprint, staged_at, backend}) or None."""
        value = self._read()['staging'].get(id)
        return dict(value) if value else None

    def referenced_env_vars(self) -> frozenset[str]:
        """Every variable named by an env: record (child processes must not inherit them)."""
        try:
            return frozenset(r.env_var for r in self.records() if r.source == 'env' and r.env_var)
        except BrokerError:
            return frozenset()

    def summary(self) -> dict:
        """Metadata for status views: no values, no paths."""
        records = self.records()
        return {'backend': self.backend_name, 'sealed': self.sealed, 'credentials': len(records),
                'usable': sum(1 for r in records if r.usable)}

    def reference_available(self, record: CredentialRecord) -> bool:
        """Whether an env: reference resolves in THIS process's environment now (the readiness check: the CLI that
        stored and validated it may hold a variable the server lacks). Other sources answer True; their use-time
        checks stay in _resolve. The value is neither returned nor cached."""
        if record.source != 'env':
            return True
        try:
            refs.resolve_env(record.env_var, self._environ)
        except BrokerError:
            return False
        return True

    # resolution -------------------------------------------------------------------------------------------------
    def _resolve(self, record: CredentialRecord, *, staged: bool = False) -> Secret:
        key = (record.id, staged)
        now = self._clock()
        if record.source == 'file' and not acl.is_owner_only(record.path):
            self._cache.pop(key, None)
            refs.resolve_file(record.path)  # raises credential_unavailable with the reason
        cached = self._cache.get(key)
        signature = (self.staged(record.id) or {}).get('fingerprint') if staged else record.fingerprint
        if cached is not None and cached[1] > now and cached[2] == signature:
            return cached[0]
        if record.source == 'vault':
            if staged and self.staged(record.id) is None:
                raise BrokerError(f'Credential {record.id} has no staged rotation', 'not_found')
            value = self._backend(record.backend).load(record.id + (STAGED if staged else ''))
            try:
                text = value.decode('utf-8')
            except UnicodeDecodeError:
                raise BrokerError(f'Credential {record.id} holds an unreadable value', 'credential_unavailable') from None
        elif record.source == 'env':
            text = refs.resolve_env(record.env_var, self._environ)
        else:
            text = refs.resolve_file(record.path)
        if len(text) < MIN_SECRET_LENGTH:
            raise BrokerError(f'Credential {record.id} holds an unusable value', 'credential_unavailable')
        secret = Secret(text)  # registers with redact.REGISTRY
        ttl = float(getattr(self.settings, 'broker_secret_cache_seconds', 0) or 0)
        if ttl > 0:
            self._cache[key] = (secret, now + ttl, signature)
        return secret

    def _configured(self, id: str) -> CredentialRecord:
        try:
            return self.record(id)
        except BrokerError as exc:
            if exc.code != 'not_found':
                raise
            raise BrokerError(f'Credential {id} is not configured', 'credential_unavailable') from None

    def handle(self, id: str, *, origin: str, template: str, purpose: str = 'inference',
               upstream_id: str | None = None) -> CredentialHandle:
        """A CredentialHandle for an outbound request to exactly origin. credential_unavailable unless the record exists,
        is validated (ok or skipped), is bound to origin, was written for purpose (inference | advisor: an advisor
        credential never serves inference, nor the reverse) and, when upstream_id is given, is bound to that upstream
        (the GrantIssuer passes upstream.id); vault_sealed in managed role."""
        self._refuse_sealed()
        record = self._configured(id)
        if not record.usable:
            raise BrokerError(f'Credential {id} is {record.validation}; validate it (POST '
                              f'/api/broker/credentials/{id}/validate) before use', 'credential_unavailable')
        if record.origin != origin:
            raise BrokerError(f'Credential {id} is bound to another origin; the upstream base_url changed, so it must be '
                              're-validated by a credentials:write principal', 'credential_unavailable')
        if record.purpose != purpose:
            raise BrokerError(f'Credential {id} was written for {record.purpose} use, not {purpose}',
                              'credential_unavailable')
        if upstream_id is not None and record.upstream_id != upstream_id:
            raise BrokerError(f'Credential {id} is bound to another upstream', 'credential_unavailable')
        return self._make_handle(record, origin, template, staged=False)

    def validation_handle(self, id: str, *, origin: str, template: str, staged: bool = False) -> CredentialHandle:
        """For the credentials:write validate and rotate flows only (purpose validate): a handle for origin whatever the
        record's validation state or bound origin, optionally over the staged rotation value. A successful probe then
        records the binding with mark_validated(id, True, None, fp, origin=origin)."""
        self._refuse_sealed()
        record = self._configured(id)
        return self._make_handle(record, origin, template, staged=staged)

    def _make_handle(self, record: CredentialRecord, origin: str, template: str, *, staged: bool) -> CredentialHandle:
        if template not in AUTH_TEMPLATES:
            raise ValueError(f'Unknown auth template: {template!r}')
        secret = None if AUTH_TEMPLATES[template] is None else self._resolve(record, staged=staged)
        return CredentialHandle(origin, template, secret, credential_id=record.id)


def open_vault(settings, home) -> Vault:
    """The Vault for TITANOS_HOME `home` (files under home/broker/vault). Lazy: nothing is read or created here."""
    return Vault(settings, Path(home))


__all__ = ['BACKEND_CHOICES', 'INDEX_FILE', 'STORE_BACKENDS', 'VAULT_UNAVAILABLE_MESSAGE', 'Vault', 'open_vault',
           'platform_default']
