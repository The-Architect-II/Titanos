"""macOS vault backend: the login (or a given) Keychain through /usr/bin/security.

- Writes run `security -i` and send the command on stdin:
      add-generic-password -U -s titanos.broker -a <instance_id>/<id> -w "<secret>" ["<keychain>"]
  with every argument double-quoted and escaped for security's command parser, so the secret never appears in argv.
- Reads run `security find-generic-password -s titanos.broker -a <instance_id>/<id> -w ["<keychain>"]` (no secret in
  argv) and read the value from the piped stdout. Deletes run delete-generic-password the same way.
- Every child gets a scrubbed environment (redact.child_env, minus the vault's env: credential variables). Output is
  never logged.
- An optional keychain path lets tests use a TEMPORARY keychain (temporary_keychain(): create-keychain and
  unlock-keychain over stdin with a password from os.urandom), never the user's login keychain.
- `executable` is the argv prefix of the tool (default /usr/bin/security); tests inject a recording fake.
"""
from __future__ import annotations

import hmac
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ..errors import BrokerError
from ..redact import child_env

SERVICE = 'titanos.broker'
DEFAULT_EXECUTABLE = ('/usr/bin/security',)
TIMEOUT = 30.0
NOT_FOUND = 44  # errSecItemNotFound exit status of security(1)
_UNSAFE = re.compile(r'[\x00-\x1f\x7f]')


def quote(value: str) -> str:
    """A double-quoted argument for security -i (backslash and double quote escaped). Control characters are refused,
    so a value can never end the command line."""
    if _UNSAFE.search(value):
        raise ValueError('Keychain arguments may not contain control characters')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _argv(executable) -> list[str]:
    return [executable] if isinstance(executable, (str, os.PathLike)) else [str(part) for part in executable]


def deny_names(deny) -> frozenset[str]:
    """Extra variable names to drop from a helper's environment: an iterable or a callable returning one (the vault
    passes Vault.referenced_env_vars, so every env: credential variable is withheld)."""
    if deny is None:
        return frozenset()
    names = deny() if callable(deny) else deny
    return frozenset(str(name) for name in names)


def run_tool(argv: Sequence[str], *, stdin: str | None = None, timeout: float = TIMEOUT,
             environ: Mapping[str, str] | None = None, deny=()) -> subprocess.CompletedProcess:
    """Run a vault helper tool: no shell, scrubbed env (redact.child_env plus deny), bounded time, captured output that
    is never logged."""
    env = child_env(os.environ if environ is None else environ, extra_deny=deny_names(deny))
    try:
        return subprocess.run(list(argv), input=stdin, capture_output=True, text=True, encoding='utf-8',
                              timeout=timeout, env=env, check=False, shell=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BrokerError(f'The vault helper {Path(str(argv[0])).name} could not run ({type(exc).__name__})',
                          'vault_unavailable') from None


class KeychainBackend:
    """VaultBackend over generic-password items (service titanos.broker, account <instance_id>/<id>)."""
    name = 'keychain'

    def __init__(self, *, keychain: Path | None = None, instance_id: str | Callable[[], str] = '',
                 executable: str | Sequence[str] = DEFAULT_EXECUTABLE, environ: Mapping[str, str] | None = None,
                 timeout: float = TIMEOUT, deny=()):
        self.keychain = Path(keychain) if keychain is not None else None
        self._instance_id = instance_id
        self.executable = _argv(executable)
        self.environ = environ
        self.timeout = timeout
        self.deny = deny  # extra variable names (or a callable) withheld from the helper's environment

    def available(self) -> bool:
        if self.executable == list(DEFAULT_EXECUTABLE) and sys.platform != 'darwin':
            return False
        return Path(self.executable[-1] if len(self.executable) > 1 else self.executable[0]).exists()

    def account(self, id: str) -> str:
        instance = self._instance_id() if callable(self._instance_id) else self._instance_id
        return f'{instance}/{id}' if instance else id

    def _keychain_args(self) -> list[str]:
        return [str(self.keychain)] if self.keychain is not None else []

    def store(self, id: str, value: bytes) -> None:
        text = bytes(value).decode('utf-8')
        command = ' '.join(['add-generic-password', '-U', '-s', quote(SERVICE), '-a', quote(self.account(id)), '-w',
                            quote(text), *(quote(k) for k in self._keychain_args())]) + '\n'
        result = run_tool([*self.executable, '-i'], stdin=command, timeout=self.timeout, environ=self.environ,
                          deny=self.deny)
        if result.returncode != 0 or 'error' in (result.stderr or '').lower():
            raise BrokerError(f'The keychain refused to store credential {id} (exit status {result.returncode})',
                              'vault_unavailable')
        # security -i may report command failures only on stderr: read the item back and compare
        if not hmac.compare_digest(self.load(id), bytes(value)):
            raise BrokerError(f'The keychain did not store credential {id}', 'vault_unavailable')

    def load(self, id: str) -> bytes:
        result = run_tool([*self.executable, 'find-generic-password', '-s', SERVICE, '-a', self.account(id), '-w',
                           *self._keychain_args()], timeout=self.timeout, environ=self.environ, deny=self.deny)
        if result.returncode != 0:
            raise BrokerError(f'Credential {id} is not readable in the keychain (exit status {result.returncode}); the '
                              'CLI and the server must run as the same macOS user', 'credential_unavailable')
        value = result.stdout[:-1] if result.stdout.endswith('\n') else result.stdout
        return value.encode('utf-8')

    def delete(self, id: str) -> None:
        result = run_tool([*self.executable, 'delete-generic-password', '-s', SERVICE, '-a', self.account(id),
                           *self._keychain_args()], timeout=self.timeout, environ=self.environ, deny=self.deny)
        if result.returncode not in (0, NOT_FOUND):
            raise BrokerError(f'The keychain refused to delete credential {id} (exit status {result.returncode})',
                              'vault_unavailable')


def temporary_keychain(path: Path, *, executable: str | Sequence[str] = DEFAULT_EXECUTABLE) -> None:
    """Create and unlock a keychain at path for tests. The password comes from os.urandom and travels over stdin only;
    it is not kept. Remove it with delete_keychain()."""
    password = os.urandom(24).hex()
    command = (f'create-keychain -p {quote(password)} {quote(str(path))}\n'
               f'unlock-keychain -p {quote(password)} {quote(str(path))}\n')
    result = run_tool([*_argv(executable), '-i'], stdin=command)
    if result.returncode != 0 or not Path(path).exists():
        raise BrokerError('Could not create the temporary keychain', 'vault_unavailable')


def delete_keychain(path: Path, *, executable: str | Sequence[str] = DEFAULT_EXECUTABLE) -> None:
    run_tool([*_argv(executable), 'delete-keychain', str(path)])


__all__ = ['DEFAULT_EXECUTABLE', 'KeychainBackend', 'SERVICE', 'delete_keychain', 'deny_names', 'quote', 'run_tool',
           'temporary_keychain']
