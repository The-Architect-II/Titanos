"""Linux vault backend: the freedesktop Secret Service through the secret-tool command.

- store: `secret-tool store --label=Titanos service titanos.broker account <instance_id>/<id>` with the secret on stdin
  (never in argv).
- lookup: `secret-tool lookup service titanos.broker account <instance_id>/<id>`, value read from the piped stdout.
- clear: `secret-tool clear service titanos.broker account <instance_id>/<id>`.
Available only when secret-tool is on PATH and DBUS_SESSION_BUS_ADDRESS is set. Every child gets a scrubbed environment
(redact.child_env keeps DBUS_SESSION_BUS_ADDRESS); output is never logged. `executable` is the argv prefix of the tool
(default: secret-tool from PATH); tests inject a recording fake.
"""
from __future__ import annotations

import hmac
import os
import shutil
from collections.abc import Callable, Mapping, Sequence

from ..errors import BrokerError
from .keychain import TIMEOUT, run_tool

SERVICE = 'titanos.broker'
LABEL = 'Titanos'


class SecretServiceBackend:
    """VaultBackend over Secret Service items (attributes service=titanos.broker, account=<instance_id>/<id>)."""
    name = 'secret-service'

    def __init__(self, *, instance_id: str | Callable[[], str] = '', executable: str | Sequence[str] | None = None,
                 environ: Mapping[str, str] | None = None, timeout: float = TIMEOUT, deny=()):
        self._instance_id = instance_id
        self._executable = executable
        self.environ = environ
        self.timeout = timeout
        self.deny = deny  # extra variable names (or a callable) withheld from the helper's environment

    @property
    def executable(self) -> list[str] | None:
        if self._executable is None:
            found = shutil.which('secret-tool')
            return [found] if found else None
        if isinstance(self._executable, (str, os.PathLike)):
            return [str(self._executable)]
        return [str(part) for part in self._executable]

    def available(self) -> bool:
        env = os.environ if self.environ is None else self.environ
        return self.executable is not None and bool(env.get('DBUS_SESSION_BUS_ADDRESS'))

    def account(self, id: str) -> str:
        instance = self._instance_id() if callable(self._instance_id) else self._instance_id
        return f'{instance}/{id}' if instance else id

    def _exe(self) -> list[str]:
        exe = self.executable
        if exe is None:
            raise BrokerError('secret-tool is not installed', 'vault_unavailable')
        return exe

    def store(self, id: str, value: bytes) -> None:
        text = bytes(value).decode('utf-8')
        result = run_tool([*self._exe(), 'store', f'--label={LABEL}', 'service', SERVICE, 'account', self.account(id)],
                          stdin=text, timeout=self.timeout, environ=self.environ, deny=self.deny)
        if result.returncode != 0:
            raise BrokerError(f'The Secret Service refused to store credential {id} (exit status {result.returncode})',
                              'vault_unavailable')
        if not hmac.compare_digest(self.load(id), bytes(value)):
            raise BrokerError(f'The Secret Service did not store credential {id}', 'vault_unavailable')

    def load(self, id: str) -> bytes:
        result = run_tool([*self._exe(), 'lookup', 'service', SERVICE, 'account', self.account(id)],
                          timeout=self.timeout, environ=self.environ, deny=self.deny)
        if result.returncode != 0 or not result.stdout:
            raise BrokerError(f'Credential {id} is not readable in the Secret Service (exit status '
                              f'{result.returncode}); the CLI and the server must run as the same user',
                              'credential_unavailable')
        value = result.stdout[:-1] if result.stdout.endswith('\n') else result.stdout
        return value.encode('utf-8')

    def delete(self, id: str) -> None:
        run_tool([*self._exe(), 'clear', 'service', SERVICE, 'account', self.account(id)], timeout=self.timeout,
                 environ=self.environ, deny=self.deny)


__all__ = ['LABEL', 'SERVICE', 'SecretServiceBackend']
