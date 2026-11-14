"""Append-only, SHA-256 hash-chained audit at TITANOS_HOME/broker/audit/YYYY-MM-DD.jsonl.

- Each record is {seq, ts, kind, actor, subject, detail, prev_sha256, sha256}. sha256 is the SHA-256 of the record's
  canonical JSON (sorted keys, compact separators, UTF-8) without the sha256 field; prev_sha256 is the previous record's
  sha256 (64 zeros for the first record ever). seq increases by one. The chain continues across day files.
- Writers append under a FileLock shared with the offline CLI (audit/.audit.lock), so the chain stays linear across
  processes; the tail is re-read under the lock whenever the latest file changed.
- detail, actor and subject are scrubbed; the audit never records secrets (token ids and fingerprints only).
- Retention (broker_audit_retention_days): day files older than the window are deleted, never the file holding the
  tail, and an audit.prune record carrying the last pruned seq and sha256 anchors the retained chain, so verify()
  still proves it.
- verify() reports the first bad record as '<file>:<line>'.
Nothing here performs network I/O; the directory is created owner-only on the first write.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .keys import ensure_private_dir
from .redact import scrub, scrub_obj
from .usage import as_date, utc_ts

GENESIS = '0' * 64
AUDIT_KINDS = frozenset({
    'key.mint', 'key.update', 'key.revoke', 'key.rotate', 'key.reset_keyless',
    'credential.put', 'credential.validate', 'credential.rotate', 'credential.delete',
    'upstream.write', 'alias.write', 'virtual.write', 'policy.write', 'advisor.write', 'prices.write', 'config.reload',
    'auth.failure', 'budget.stop', 'breaker.open', 'role.start', 'audit.prune',
    'native.call', 'broker.connect',
})
FIELDS = ('seq', 'ts', 'kind', 'actor', 'subject', 'detail', 'prev_sha256', 'sha256')
LOCK_NAME = '.audit.lock'
_DAY_FILE = re.compile(r'(\d{4}-\d{2}-\d{2})\.jsonl')
_HEX64 = re.compile(r'[0-9a-f]{64}')


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical(record: dict[str, Any]) -> bytes:
    body = {k: v for k, v in record.items() if k != 'sha256'}
    return json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def record_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(record)).hexdigest()


@dataclass(frozen=True)
class ChainReport:
    ok: bool
    records: int
    first_bad: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {'ok': self.ok, 'records': self.records, 'first_bad': self.first_bad, 'reason': self.reason}


class AuditChainError(RuntimeError):
    """The chain tail cannot be read, so nothing may be appended (fail closed)."""


class AuditLog:
    """Implements types.AuditSink."""

    def __init__(self, directory, retention_days: int = 365, *, clock: Callable[[], datetime] = _utcnow,
                 lock_timeout: float = 30.0):
        self.dir = Path(directory)
        self.retention_days = int(retention_days)
        self._clock = clock
        self._lock_timeout = lock_timeout
        self._thread_lock = threading.RLock()
        self._tail_cache: tuple[tuple, int, str] | None = None
        self._pruned_on: date | None = None

    # -- files ---------------------------------------------------------------------------------------------------------
    def _lock(self):
        from filelock import FileLock
        ensure_private_dir(self.dir)
        return FileLock(str(self.dir / LOCK_NAME), timeout=self._lock_timeout)

    def _day_files(self) -> list[tuple[date, Path]]:
        if not self.dir.is_dir():
            return []
        out = []
        for path in self.dir.iterdir():
            found = _DAY_FILE.fullmatch(path.name)
            if found:
                try:
                    out.append((date.fromisoformat(found.group(1)), path))
                except ValueError:
                    continue
        return sorted(out)

    @staticmethod
    def _last_line(path: Path) -> str | None:
        with path.open('rb') as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if end == 0:
                return None
            block = 4096
            data = b''
            pos = end
            while pos > 0:
                step = min(block, pos)
                pos -= step
                handle.seek(pos)
                data = handle.read(step) + data
                stripped = data.rstrip(b'\r\n')
                if b'\n' in stripped or pos == 0:
                    break
            lines = [line for line in data.splitlines() if line.strip()]
            return lines[-1].decode('utf-8') if lines else None

    def _tail(self) -> tuple[int, str, date | None]:
        """(seq, sha256, day) of the last record; (0, GENESIS, None) for an empty log."""
        for day, path in reversed(self._day_files()):
            st = path.stat()
            stamp = (str(path), st.st_mtime_ns, st.st_size)
            if self._tail_cache is not None and self._tail_cache[0] == stamp:
                return self._tail_cache[1], self._tail_cache[2], day
            line = self._last_line(path)
            if line is None:
                continue
            try:
                record = json.loads(line)
                seq, sha = int(record['seq']), record['sha256']
                if not isinstance(sha, str) or not _HEX64.fullmatch(sha):
                    raise ValueError('bad sha256')
            except (ValueError, KeyError, TypeError) as exc:
                raise AuditChainError(f'audit tail in {path.name} is unreadable; run titanos audit verify') from exc
            self._tail_cache = (stamp, seq, sha)
            return seq, sha, day
        return 0, GENESIS, None

    def _append_locked(self, kind: str, actor: str, subject: str, detail: dict) -> str:
        seq, prev, tail_day = self._tail()
        now = self._clock().astimezone(timezone.utc)
        day = now.date() if tail_day is None or now.date() >= tail_day else tail_day  # never write behind the tail
        record = {'seq': seq + 1, 'ts': utc_ts(now), 'kind': kind, 'actor': scrub(str(actor)),
                  'subject': scrub(str(subject)),
                  'detail': scrub_obj(json.loads(json.dumps(detail or {}, default=str))), 'prev_sha256': prev}
        record['sha256'] = record_digest(record)
        line = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n'
        path = self.dir / f'{day.isoformat()}.jsonl'
        with path.open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        st = path.stat()
        self._tail_cache = ((str(path), st.st_mtime_ns, st.st_size), record['seq'], record['sha256'])
        return record['sha256']

    # -- API -----------------------------------------------------------------------------------------------------------
    def event(self, kind: str, actor: str, subject: str, detail: dict | None = None) -> str:
        """Append one record; returns its sha256. Runs retention at most once per UTC day."""
        if kind not in AUDIT_KINDS:
            raise ValueError(f'Unknown audit kind: {kind!r}')
        with self._thread_lock, self._lock():
            digest = self._append_locked(kind, actor, subject, detail or {})
            today = self._clock().astimezone(timezone.utc).date()
            if self.retention_days > 0 and self._pruned_on != today:
                self._pruned_on = today
                self._prune_locked(self.retention_days)
        return digest

    def prune(self, retention_days: int | None = None) -> str | None:
        """Delete day files older than the retention window and append the audit.prune anchor. Returns the anchor's
        sha256, or None when nothing was pruned."""
        days = self.retention_days if retention_days is None else int(retention_days)
        with self._thread_lock, self._lock():
            return self._prune_locked(days)

    def _prune_locked(self, days: int) -> str | None:
        today = self._clock().astimezone(timezone.utc).date()
        cutoff = today - timedelta(days=max(0, days))
        files = self._day_files()
        old = [(day, path) for day, path in files[:-1] if day < cutoff]  # never the file holding the tail
        if not old:
            return None
        last_seq, last_sha = None, None
        for _day, path in reversed(old):
            line = self._last_line(path)
            if line is not None:
                record = json.loads(line)
                last_seq, last_sha = int(record['seq']), record['sha256']
                break
        for _day, path in old:
            path.unlink(missing_ok=True)
        if last_sha is None:
            return None
        detail = {'last_pruned_seq': last_seq, 'last_pruned_sha256': last_sha, 'files': len(old),
                  'before': cutoff.isoformat()}
        return self._append_locked('audit.prune', 'titanos', 'audit', detail)

    def _records(self, since: date | None = None):
        for day, path in self._day_files():
            if since is not None and day < since:
                continue
            with path.open('r', encoding='utf-8', newline='') as handle:
                for number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    yield f'{path.name}:{number}', line

    def query(self, since=None, kind: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """The most recent `limit` records (chronological order), optionally from a UTC date and of one kind."""
        since = as_date(since)
        out: list[dict[str, Any]] = []
        for _where, line in self._records(since):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if kind is not None and record.get('kind') != kind:
                continue
            out.append(record)
        limit = max(0, int(limit))
        return out[-limit:] if limit else []

    def verify(self, since=None) -> ChainReport:
        """Check every record's digest, sequence and link. From the start of the log the first record must link to the
        genesis value with seq 1, or to an audit.prune anchor; with since, the first record in range is taken as given."""
        since = as_date(since)
        parsed: list[tuple[str, dict | None]] = []
        anchors: set[tuple[int, str]] = set()
        for where, line in self._records(since):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError
            except ValueError:
                parsed.append((where, None))
                continue
            parsed.append((where, record))
            if record.get('kind') == 'audit.prune' and isinstance(record.get('detail'), dict):
                d = record['detail']
                if isinstance(d.get('last_pruned_seq'), int) and isinstance(d.get('last_pruned_sha256'), str):
                    anchors.add((d['last_pruned_seq'], d['last_pruned_sha256']))
        prev_sha, prev_seq = None, None
        count = 0
        for where, record in parsed:
            if record is None:
                return ChainReport(False, count, where, 'unparseable line')
            if tuple(sorted(record)) != tuple(sorted(FIELDS)):
                return ChainReport(False, count, where, 'unexpected fields')
            seq, sha, prev = record.get('seq'), record.get('sha256'), record.get('prev_sha256')
            if not isinstance(seq, int) or not isinstance(sha, str) or not isinstance(prev, str):
                return ChainReport(False, count, where, 'malformed record')
            if record_digest(record) != sha:
                return ChainReport(False, count, where, 'digest mismatch')
            if prev_sha is None:
                if since is None and not ((seq == 1 and prev == GENESIS) or (seq - 1, prev) in anchors):
                    return ChainReport(False, count, where, 'chain head has no genesis or prune anchor')
            elif prev != prev_sha or seq != prev_seq + 1:
                return ChainReport(False, count, where, 'broken link')
            prev_sha, prev_seq = sha, seq
            count += 1
        return ChainReport(True, count)

    def writable(self) -> bool:
        """Whether this process can take the lock and create files in the audit directory (no record is written)."""
        try:
            with self._thread_lock, self._lock():
                fd, name = tempfile.mkstemp(prefix='.writable-', dir=self.dir)
                os.close(fd)
                os.unlink(name)
                self._tail()
            return True
        except Exception:
            return False


__all__ = ['AUDIT_KINDS', 'AuditChainError', 'AuditLog', 'ChainReport', 'GENESIS', 'canonical', 'record_digest']
