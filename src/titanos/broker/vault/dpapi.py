"""Windows vault backend: DPAPI (CryptProtectData / CryptUnprotectData) through ctypes crypt32.

- Current-user scope (no CRYPTPROTECT_LOCAL_MACHINE) with CRYPTPROTECT_UI_FORBIDDEN, so a blob opens only for the same
  Windows user; the CLI and the server must therefore run as the same OS user.
- Per-install 32-byte entropy in <vault>/entropy.bin (owner-only, created on the first write) is passed as the optional
  entropy, so a blob copied to another install of the same user does not open.
- Blobs are <vault>/<id>.dpapi (rotation staging <id>.next.dpapi), written through acl.write_owner_only into the
  owner-only vault directory. Output buffers are released with LocalFree; input copies are zeroed after use.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ..errors import BrokerError
from . import acl

ENTROPY_FILE = 'entropy.bin'
ENTROPY_BYTES = 32
SUFFIX = '.dpapi'
CRYPTPROTECT_UI_FORBIDDEN = 0x1


class _Crypt:
    _instance = None

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_char))]

        self.ctypes = ctypes
        self.DATA_BLOB = DATA_BLOB
        crypt32 = ctypes.WinDLL('crypt32', use_last_error=True)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        PBLOB = ctypes.POINTER(DATA_BLOB)
        crypt32.CryptProtectData.argtypes = [PBLOB, wintypes.LPCWSTR, PBLOB, ctypes.c_void_p, ctypes.c_void_p,
                                             wintypes.DWORD, PBLOB]
        crypt32.CryptProtectData.restype = wintypes.BOOL
        crypt32.CryptUnprotectData.argtypes = [PBLOB, ctypes.POINTER(wintypes.LPWSTR), PBLOB, ctypes.c_void_p,
                                               ctypes.c_void_p, wintypes.DWORD, PBLOB]
        crypt32.CryptUnprotectData.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        self.crypt32 = crypt32
        self.kernel32 = kernel32

    @classmethod
    def get(cls) -> '_Crypt':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _blob(self, data: bytes):
        buffer = self.ctypes.create_string_buffer(data, len(data))
        blob = self.DATA_BLOB(len(data), self.ctypes.cast(buffer, self.ctypes.POINTER(self.ctypes.c_char)))
        return blob, buffer

    def _call(self, protect: bool, data: bytes, entropy: bytes) -> bytes:
        ctypes = self.ctypes
        data_in, data_buffer = self._blob(data)
        entropy_in, entropy_buffer = self._blob(entropy)
        out = self.DATA_BLOB()
        try:
            if protect:
                ok = self.crypt32.CryptProtectData(ctypes.byref(data_in), None, ctypes.byref(entropy_in), None, None,
                                                   CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out))
            else:
                ok = self.crypt32.CryptUnprotectData(ctypes.byref(data_in), None, ctypes.byref(entropy_in), None, None,
                                                     CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out))
            if not ok:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                return ctypes.string_at(out.pbData, out.cbData)
            finally:
                if not protect:
                    ctypes.memset(out.pbData, 0, out.cbData)
                self.kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))
        finally:
            ctypes.memset(data_buffer, 0, len(data))
            ctypes.memset(entropy_buffer, 0, len(entropy))

    def protect(self, data: bytes, entropy: bytes) -> bytes:
        return self._call(True, data, entropy)

    def unprotect(self, blob: bytes, entropy: bytes) -> bytes:
        return self._call(False, blob, entropy)


def dpapi_available() -> bool:
    if sys.platform != 'win32':
        return False
    try:
        _Crypt.get()
    except OSError:
        return False
    return True


class DpapiBackend:
    """VaultBackend over DPAPI blobs in the vault directory."""
    name = 'dpapi'

    def __init__(self, directory):
        self.directory = Path(directory)

    def available(self) -> bool:
        return dpapi_available()

    @property
    def entropy_path(self) -> Path:
        return self.directory / ENTROPY_FILE

    def path(self, key: str) -> Path:
        return self.directory / f'{key}{SUFFIX}'

    def _entropy(self, *, create: bool) -> bytes:
        try:
            value = self.entropy_path.read_bytes()
        except FileNotFoundError:
            if not create:
                raise BrokerError('The vault entropy file is missing; credentials stored before cannot be opened',
                                  'credential_unavailable') from None
            value = os.urandom(ENTROPY_BYTES)
            acl.write_owner_only(self.entropy_path, value)  # callers hold the vault write lock
        if len(value) != ENTROPY_BYTES:
            raise BrokerError('The vault entropy file is damaged', 'credential_unavailable')
        return value

    def store(self, id: str, value: bytes) -> None:
        if not self.available():
            raise BrokerError('DPAPI is not available on this system', 'vault_unavailable')
        blob = _Crypt.get().protect(bytes(value), self._entropy(create=True))
        acl.write_owner_only(self.path(id), blob)

    def load(self, id: str) -> bytes:
        if not self.available():
            raise BrokerError('DPAPI is not available on this system', 'vault_unavailable')
        try:
            blob = self.path(id).read_bytes()
        except FileNotFoundError:
            raise BrokerError(f'Credential {id} has no stored value', 'credential_unavailable') from None
        try:
            return _Crypt.get().unprotect(blob, self._entropy(create=False))
        except OSError:
            raise BrokerError(f'Credential {id} cannot be decrypted: the CLI and the server must run as the same '
                              'Windows user that stored it', 'credential_unavailable') from None

    def delete(self, id: str) -> None:
        self.path(id).unlink(missing_ok=True)


__all__ = ['CRYPTPROTECT_UI_FORBIDDEN', 'DpapiBackend', 'ENTROPY_BYTES', 'ENTROPY_FILE', 'dpapi_available']
