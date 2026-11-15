"""The vault interface: the Secret wrapper, credential handles and records, and the backend/store Protocols.

Backends (DPAPI, keychain, secret-service, sealed-file) and the store are implemented by the vault package's own modules.
Secret.reveal() may be called only from broker/egress.py and broker/vault/* (a source scan enforces this); a
CredentialHandle is the only way an outbound request receives a credential, and it refuses any origin other than the one
the credential record was bound to at write time.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from ..errors import EgressRefused
from ..profiles import AUTH_TEMPLATES
from ..redact import MIN_SECRET_LENGTH, REGISTRY

if TYPE_CHECKING:  # pragma: no cover - annotations only
    import httpx

CREDENTIAL_ID_RE = '^[a-z][a-z0-9_-]{0,63}$'
_CREDENTIAL_ID = re.compile(CREDENTIAL_ID_RE)
FINGERPRINT_PREFIX = 'titanos-fp-v1:'
CredentialSource = Literal['vault', 'env', 'file']
CredentialPurpose = Literal['inference', 'advisor']
VALIDATION_STATES = frozenset({'ok', 'skipped', 'unvalidated'})  # plus 'failed:<code>'


def fingerprint(secret: str) -> str:
    """First 8 hex characters of SHA-256('titanos-fp-v1:' + secret): safe to show, log and store."""
    return hashlib.sha256((FINGERPRINT_PREFIX + secret).encode('utf-8')).hexdigest()[:8]


def valid_credential_id(value: str) -> bool:
    return isinstance(value, str) and _CREDENTIAL_ID.fullmatch(value) is not None


class Secret:
    """An opaque secret value. Its text forms show only the fingerprint, it cannot be pickled or copied by pickling, and
    its value is registered with redact.REGISTRY so it is scrubbed wherever it might appear."""
    __slots__ = ('_value', 'fingerprint')

    def __init__(self, value: str):
        if not isinstance(value, str) or len(value) < MIN_SECRET_LENGTH or value != value.strip():
            raise ValueError(f'A secret must be a string of at least {MIN_SECRET_LENGTH} characters without surrounding '
                             'whitespace')
        object.__setattr__(self, '_value', value)
        object.__setattr__(self, 'fingerprint', fingerprint(value))
        REGISTRY.add(value)

    def __setattr__(self, name, value):
        raise AttributeError('Secret is immutable')

    def reveal(self) -> str:
        """The plaintext. Call sites are restricted to broker/egress.py and broker/vault/* by source scan."""
        return self._value

    def __repr__(self) -> str:
        return f'<secret fp={self.fingerprint}>'

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __reduce__(self):
        raise TypeError('Secret values cannot be pickled')

    def __reduce_ex__(self, protocol):
        raise TypeError('Secret values cannot be pickled')

    def __eq__(self, other) -> bool:
        if not isinstance(other, Secret):
            return NotImplemented
        return hmac.compare_digest(self._value.encode('utf-8'), other._value.encode('utf-8'))

    __hash__ = None  # never a dict key or set member


def request_origin(url) -> str:
    """'scheme://host:port' of an httpx.URL (or URL text), lower-case host, default ports filled in."""
    import httpx
    url = httpx.URL(str(url)) if not isinstance(url, httpx.URL) else url
    scheme = url.scheme.lower()
    host = url.raw_host.decode('ascii').lower().rstrip('.')  # IDNA (punycode) form, as parse_origin stores it
    port = url.port or {'https': 443, 'http': 80}.get(scheme)
    if ':' in host:
        host = '[' + host + ']'
    return f'{scheme}://{host}:{port}'


class CredentialHandle:
    """A credential bound to one exact origin. apply() sets the auth header named by template (a profiles.AUTH_TEMPLATES
    key) and raises EgressRefused unless the request's origin equals the bound origin. Immutable: the origin binding
    cannot be re-pointed after construction.

    apply() checks the origin once. httpx strips only Authorization on a cross-origin redirect, so every client that
    carries a handle must keep follow_redirects=False (egress refuses any 3xx as upstream_redirect_refused)."""
    __slots__ = ('origin', 'template', 'credential_id', '_secret')

    def __init__(self, origin: str, template: str, secret: Secret | None, *, credential_id: str | None = None):
        if template not in AUTH_TEMPLATES:
            raise ValueError(f'Unknown auth template: {template!r}')
        if (secret is None) != (AUTH_TEMPLATES[template] is None):
            raise ValueError('An auth template needs a secret exactly when it names a header')
        for name, value in (('origin', origin), ('template', template), ('credential_id', credential_id),
                            ('_secret', secret)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError('Credential handles are immutable')

    def __delattr__(self, name):
        raise AttributeError('Credential handles are immutable')

    @property
    def fingerprint(self) -> str | None:
        return self._secret.fingerprint if self._secret is not None else None

    def apply(self, request: 'httpx.Request') -> None:
        if request_origin(request.url) != self.origin:
            raise EgressRefused('Credential origin does not match the request origin')
        if self._secret is None:
            return
        header, value = AUTH_TEMPLATES[self.template]
        request.headers[header] = value.format(secret=self._secret.reveal())

    def __repr__(self) -> str:
        return f'<credential-handle {self.credential_id or "-"} {self.template} {self.origin} fp={self.fingerprint}>'

    def __reduce__(self):
        raise TypeError('Credential handles cannot be pickled')


@dataclass(frozen=True)
class CredentialRecord:
    """Credential metadata (index.json). Never a secret value: env and file sources store only the reference."""
    id: str
    source: CredentialSource
    upstream_id: str
    origin: str
    purpose: CredentialPurpose = 'inference'
    env_var: str | None = None
    path: str | None = None
    backend: str | None = None  # dpapi | keychain | secret-service | sealed-file | None
    fingerprint: str | None = None
    created_at: str | None = None
    rotated_at: str | None = None
    validated_at: str | None = None
    validation: str = 'unvalidated'  # ok | skipped | unvalidated | failed:<code>
    last_used_at: str | None = None

    def __post_init__(self):
        if not valid_credential_id(self.id):
            raise ValueError(f'Invalid credential id: {self.id!r}')
        if self.source not in ('vault', 'env', 'file') or self.purpose not in ('inference', 'advisor'):
            raise ValueError('Invalid credential source or purpose')
        if (self.source == 'env') != (self.env_var is not None) or (self.source == 'file') != (self.path is not None):
            raise ValueError('env_var is set exactly for env sources and path exactly for file sources')
        if self.validation not in VALIDATION_STATES and not re.fullmatch(r'failed:[a-z0-9_]{1,64}', self.validation):
            raise ValueError(f'Invalid credential validation state: {self.validation!r}')

    @property
    def usable(self) -> bool:
        return self.validation in ('ok', 'skipped')

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'CredentialRecord':
        return cls(**data)


@runtime_checkable
class VaultBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def store(self, id: str, value: bytes) -> None: ...

    def load(self, id: str) -> bytes: ...

    def delete(self, id: str) -> None: ...


@runtime_checkable
class VaultProtocol(Protocol):
    backend_name: str
    sealed: bool

    def put_secret(self, id: str, secret: str, *, upstream_id: str, origin: str, purpose: str) -> CredentialRecord: ...

    def put_env_ref(self, id: str, var: str, *, upstream_id: str, origin: str, purpose: str) -> CredentialRecord: ...

    def put_file_ref(self, id: str, path: str, *, upstream_id: str, origin: str, purpose: str) -> CredentialRecord: ...

    def handle(self, id: str, *, origin: str, template: str) -> CredentialHandle: ...

    def record(self, id: str) -> CredentialRecord: ...

    def records(self) -> list[CredentialRecord]: ...

    def mark_validated(self, id: str, ok: bool, code: str | None, fingerprint: str | None) -> None: ...

    def invalidate_for_upstream(self, upstream_id: str) -> None: ...

    def delete(self, id: str) -> None: ...

    def stage_rotation(self, id: str, secret: str) -> str: ...

    def commit_rotation(self, id: str) -> None: ...

    def seal(self, reason: str) -> None: ...


__all__ = ['CREDENTIAL_ID_RE', 'CredentialHandle', 'CredentialRecord', 'FINGERPRINT_PREFIX', 'Secret', 'VALIDATION_STATES',
           'VaultBackend', 'VaultProtocol', 'fingerprint', 'request_origin', 'valid_credential_id']
