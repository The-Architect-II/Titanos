"""env: and file: credential sources, resolved at use time and never persisted.

- env:<VAR> reads the process environment when the credential is used. The value is never written anywhere (index.json
  stores only the variable name) and never echoed. Names ending in _BASE_URL are refused: base URLs come only from
  broker.json, never from ambient variables. Titanos' own variables (every TITANOS_* name, the bootstrap key
  TITANOS_API_KEY included) are refused too: they are never presented to an upstream as a provider credential.
- file:<absolute path> re-checks the file's permissions on every read. POSIX requires a regular file owned by the
  effective user with mode & 0o077 == 0, opened with O_NOFOLLOW and checked on the open descriptor. Windows requires an
  owner-only DACL (acl.is_owner_only). Default ACLs on non-system drives usually grant Authenticated Users, so such files
  are refused until the operator tightens them (icacls <file> /inheritance:r /grant:r <user>:F).

A missing value, a short value or a refused permission raises BrokerError credential_unavailable (503, no-retry). Error
messages name the variable or path, never the value; an env: reference gets one message whether the variable is unset,
empty or too short (no presence or length oracle).
"""
from __future__ import annotations

import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

from ..errors import BrokerError
from ..redact import MIN_SECRET_LENGTH
from . import acl

ENV_VAR_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,127}$')
MAX_FILE_BYTES = 64 * 1024
_CONTROL = re.compile(r'[\x00-\x1f\x7f]')


RESERVED_ENV_PREFIX = 'TITANOS_'


def check_env_var(var: str) -> str:
    """The variable name, or ValueError: a valid environment name that does not end in _BASE_URL and is not one of
    Titanos' own TITANOS_* variables (the bootstrap key and the server settings)."""
    if not isinstance(var, str) or not ENV_VAR_RE.fullmatch(var):
        raise ValueError('An env: reference names one environment variable ([A-Za-z_][A-Za-z0-9_]*)')
    if var.upper().endswith('_BASE_URL'):
        raise ValueError('Base-URL variables are never read; set the upstream base_url in broker.json')
    if var.upper().startswith(RESERVED_ENV_PREFIX):
        raise ValueError('TITANOS_* variables are Titanos settings and keys; they are never an upstream credential')
    return var


def check_file_path(path) -> Path:
    """The path, or ValueError: file: references are absolute paths."""
    if not isinstance(path, (str, os.PathLike)) or not str(path) or str(path) != str(path).strip():
        raise ValueError('A file: reference is an absolute path')
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError('A file: reference must be an absolute path')
    return candidate


def _usable(value: str, where: str) -> str:
    value = value.strip()
    if len(value) < MIN_SECRET_LENGTH or _CONTROL.search(value):
        raise BrokerError(f'Credential source {where} does not hold a usable value (at least {MIN_SECRET_LENGTH} '
                          'printable characters)', 'credential_unavailable')
    return value


def resolve_env(var: str, environ: Mapping[str, str] | None = None) -> str:
    """The value of an env: reference at use time; credential_unavailable when it is not set."""
    try:
        check_env_var(var)
    except ValueError as exc:
        raise BrokerError(str(exc), 'credential_unavailable') from None
    env = os.environ if environ is None else environ
    value = (env.get(var) or '').strip()
    if len(value) < MIN_SECRET_LENGTH or _CONTROL.search(value):
        # one message for unset, empty, short and malformed values: the answer never tells a caller anything about a
        # variable's presence or length (R30: names and booleans only, and only for the well-known provider names)
        raise BrokerError(f'Environment variable {var} does not hold a usable credential in the server process',
                          'credential_unavailable')
    return value


def _refuse_file(path: Path, why: str) -> BrokerError:
    return BrokerError(f'Credential file {path} is refused: {why}', 'credential_unavailable')


def resolve_file(path) -> str:
    """The value of a file: reference; the permissions are checked on every call."""
    try:
        candidate = check_file_path(path)
    except ValueError as exc:
        raise BrokerError(str(exc), 'credential_unavailable') from None
    if sys.platform == 'win32':
        try:
            if candidate.is_symlink() or not candidate.is_file():
                raise _refuse_file(candidate, 'it is missing, not a regular file, or a link')
        except OSError:
            raise _refuse_file(candidate, 'it cannot be inspected') from None
        if not acl.is_owner_only(candidate):
            raise _refuse_file(candidate, 'other accounts can read it; restrict it to the current user '
                                          '(icacls <file> /inheritance:r /grant:r <user>:F)')
        try:
            with open(candidate, 'rb') as handle:
                data = handle.read(MAX_FILE_BYTES + 1)
        except OSError:
            raise _refuse_file(candidate, 'it cannot be read') from None
    else:
        flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
        try:
            fd = os.open(candidate, flags)
        except OSError:
            raise _refuse_file(candidate, 'it is missing, unreadable or a symlink') from None
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise _refuse_file(candidate, 'it is not a regular file')
            if st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode) & 0o077:
                raise _refuse_file(candidate, 'it must be owned by the server user with mode 0600 (chmod 600)')
            chunks, size = [], 0
            while size <= MAX_FILE_BYTES:
                chunk = os.read(fd, MAX_FILE_BYTES + 1 - size)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            data = b''.join(chunks)
        finally:
            os.close(fd)
    if len(data) > MAX_FILE_BYTES:
        raise _refuse_file(candidate, f'it is larger than {MAX_FILE_BYTES} bytes')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        raise _refuse_file(candidate, 'it is not UTF-8 text') from None
    return _usable(text, f'file:{candidate}')


__all__ = ['ENV_VAR_RE', 'MAX_FILE_BYTES', 'RESERVED_ENV_PREFIX', 'check_env_var', 'check_file_path', 'resolve_env', 'resolve_file']
