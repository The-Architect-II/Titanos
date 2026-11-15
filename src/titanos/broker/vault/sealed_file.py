"""sealed-file: an explicit, opt-in vault backend using only the standard library. Weaker than the OS stores.

It is never auto-selected: it needs TITANOS_BROKER_VAULT_BACKEND=sealed-file plus TITANOS_BROKER_VAULT_PASSPHRASE_FILE
naming an owner-only file (checked on every use; the backend is unavailable otherwise).

Construction (encrypt-then-MAC, versioned):
- key derivation: hashlib.scrypt(passphrase, salt=32 random bytes, n=2**15, r=8, p=1, maxmem=64 MiB, dklen=64). maxmem
  is required: the derivation needs about 32 MiB (128*r*n), which exceeds hashlib's default limit and would raise. The
  64-byte result splits into enc_key (first 32) and mac_key (last 32).
- keystream: block i = HMAC-SHA256(enc_key, nonce16 || i as 8 bytes big-endian); ciphertext = plaintext XOR keystream.
- tag: HMAC-SHA256(mac_key, b'TSF1' || version || len(id) as 2 bytes || id || nonce || ciphertext), compared with
  hmac.compare_digest before anything is decrypted.

File <id>.sealed (written with acl.write_owner_only): b'TSF1' | version (1 byte) | salt (32) | nonce (16) | ciphertext |
tag (32). A changed salt derives other keys and fails the tag like any other changed byte.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from ..errors import BrokerError
from ..redact import MIN_SECRET_LENGTH
from . import acl

FORMAT_VERSION = 1
MAGIC = b'TSF1'
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024
SALT_BYTES = 32
NONCE_BYTES = 16
TAG_BYTES = 32
KEY_BYTES = 64
_HEADER = len(MAGIC) + 1 + SALT_BYTES + NONCE_BYTES
SUFFIX = '.sealed'


class SealedFileError(ValueError):
    """A sealed blob is malformed, of an unknown version, or fails authentication (wrong passphrase or tampering)."""


def derive_keys(passphrase: bytes, salt: bytes) -> tuple[bytes, bytes]:
    """(enc_key, mac_key) from scrypt with the fixed parameters."""
    if len(salt) != SALT_BYTES:
        raise ValueError(f'salt must be {SALT_BYTES} bytes')
    key = hashlib.scrypt(passphrase, salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=SCRYPT_MAXMEM, dklen=KEY_BYTES)
    return key[:32], key[32:]


def keystream(enc_key: bytes, nonce: bytes, length: int) -> bytes:
    """HMAC-SHA256 in counter mode: block i = HMAC(enc_key, nonce || i.to_bytes(8, 'big'))."""
    blocks = []
    for counter in range((length + 31) // 32):
        blocks.append(hmac.new(enc_key, nonce + counter.to_bytes(8, 'big'), hashlib.sha256).digest())
    return b''.join(blocks)[:length]


def _tag(mac_key: bytes, version: int, cred_id: str, nonce: bytes, ciphertext: bytes) -> bytes:
    ident = cred_id.encode('utf-8')
    if len(ident) > 0xFFFF:
        raise ValueError('credential id too long')
    message = MAGIC + bytes([version]) + len(ident).to_bytes(2, 'big') + ident + nonce + ciphertext
    return hmac.new(mac_key, message, hashlib.sha256).digest()


def seal_bytes(passphrase: bytes, cred_id: str, plaintext: bytes, *, salt: bytes | None = None,
               nonce: bytes | None = None) -> bytes:
    """A sealed blob. salt and nonce are random unless given (known-answer tests pass fixed values)."""
    salt = os.urandom(SALT_BYTES) if salt is None else salt
    nonce = os.urandom(NONCE_BYTES) if nonce is None else nonce
    if len(nonce) != NONCE_BYTES:
        raise ValueError(f'nonce must be {NONCE_BYTES} bytes')
    enc_key, mac_key = derive_keys(passphrase, salt)
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, keystream(enc_key, nonce, len(plaintext))))
    return MAGIC + bytes([FORMAT_VERSION]) + salt + nonce + ciphertext + _tag(mac_key, FORMAT_VERSION, cred_id, nonce,
                                                                             ciphertext)


def open_bytes(passphrase: bytes, cred_id: str, blob: bytes) -> bytes:
    """The plaintext of a sealed blob; SealedFileError unless the format, version and tag all check out."""
    if len(blob) < _HEADER + TAG_BYTES or blob[:4] != MAGIC:
        raise SealedFileError('not a sealed credential file')
    version = blob[4]
    if version != FORMAT_VERSION:
        raise SealedFileError(f'unsupported sealed-file version {version}')
    salt = blob[5:5 + SALT_BYTES]
    nonce = blob[5 + SALT_BYTES:_HEADER]
    ciphertext, tag = blob[_HEADER:-TAG_BYTES], blob[-TAG_BYTES:]
    enc_key, mac_key = derive_keys(passphrase, salt)
    if not hmac.compare_digest(tag, _tag(mac_key, version, cred_id, nonce, ciphertext)):
        raise SealedFileError('sealed credential failed authentication (wrong passphrase or modified file)')
    return bytes(a ^ b for a, b in zip(ciphertext, keystream(enc_key, nonce, len(ciphertext))))


class SealedFileBackend:
    """VaultBackend over <directory>/<id>.sealed files; the passphrase file is re-checked owner-only on every use."""
    name = 'sealed-file'

    def __init__(self, directory, *, passphrase_file: str | os.PathLike | None):
        self.directory = Path(directory)
        self.passphrase_file = Path(passphrase_file) if passphrase_file else None

    def unavailable_reason(self) -> str | None:
        if self.passphrase_file is None:
            return 'TITANOS_BROKER_VAULT_PASSPHRASE_FILE is not set'
        if not self.passphrase_file.is_absolute():
            return 'the passphrase file path is not absolute'
        if not self.passphrase_file.is_file():
            return 'the passphrase file does not exist'
        if not acl.is_owner_only(self.passphrase_file):
            return 'the passphrase file is readable by other accounts (it must be owner-only)'
        return None

    def available(self) -> bool:
        return self.unavailable_reason() is None

    def _passphrase(self) -> bytes:
        reason = self.unavailable_reason()
        if reason is not None:
            raise BrokerError(f'sealed-file vault is unavailable: {reason}', 'vault_unavailable')
        value = self.passphrase_file.read_bytes().rstrip(b'\r\n')
        if len(value) < MIN_SECRET_LENGTH:
            raise BrokerError(f'sealed-file vault is unavailable: the passphrase is shorter than {MIN_SECRET_LENGTH} '
                              'bytes', 'vault_unavailable')
        return value

    def path(self, key: str) -> Path:
        return self.directory / f'{key}{SUFFIX}'

    def store(self, id: str, value: bytes) -> None:
        acl.write_owner_only(self.path(id), seal_bytes(self._passphrase(), id, bytes(value)))

    def load(self, id: str) -> bytes:
        passphrase = self._passphrase()
        try:
            blob = self.path(id).read_bytes()
        except FileNotFoundError:
            raise BrokerError(f'Credential {id} has no sealed value', 'credential_unavailable') from None
        try:
            return open_bytes(passphrase, id, blob)
        except SealedFileError as exc:
            raise BrokerError(f'Credential {id} cannot be opened: {exc}', 'credential_unavailable') from None

    def delete(self, id: str) -> None:
        self.path(id).unlink(missing_ok=True)


__all__ = ['FORMAT_VERSION', 'MAGIC', 'SCRYPT_MAXMEM', 'SCRYPT_N', 'SCRYPT_P', 'SCRYPT_R', 'SealedFileBackend',
           'SealedFileError', 'derive_keys', 'keystream', 'open_bytes', 'seal_bytes']
