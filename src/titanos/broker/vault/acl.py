"""Owner-only permission helpers for broker state (vault blobs, entropy, index, keys and the other broker files).

Windows (ctypes, advapi32): the current user's SID comes from GetTokenInformation(TokenUser) on the process token. A
protected DACL is built from SDDL with ConvertStringSecurityDescriptorToSecurityDescriptorW and applied with
SetNamedSecurityInfoW(PROTECTED_DACL_SECURITY_INFORMATION):

    directories  D:P(A;OICI;FA;;;<user SID>)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)   (children inherit it)
    files        D:P(A;;FA;;;<user SID>)(A;;FA;;;SY)(A;;FA;;;BA)

is_owner_only() walks the DACL: every access-allowed ACE must name the user, SYSTEM, Administrators or OWNER RIGHTS, and
the owner must be the user, SYSTEM or Administrators (so OWNER RIGHTS names one of them too). Any other allowed-ACE type
(object or callback ACEs) or a NULL DACL fails. It does not require a protected DACL: a file that inherits only from an
owner-only directory is owner-only too (acl_summary() reports `protected` separately). Note that the default ACL of a
non-system drive usually grants Authenticated Users, so files there are refused until the operator tightens them.

POSIX: files are created with os.open(O_CREAT|O_EXCL|O_NOFOLLOW, 0o600), directories are 0o700, and is_owner_only()
requires st_uid == geteuid() and mode & 0o077 == 0 (never a symlink).

write_owner_only() writes a temporary file beside the target that is owner-only from creation (Windows: CreateFileW with
the protected file DACL as its security descriptor; POSIX: mode 0o600 at O_CREAT|O_EXCL), then atomically replaces the
target with the bounded Windows PermissionError retry (config.replace_with_retry). Importing this
module does no I/O and loads no library.
"""
from __future__ import annotations

import os
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import replace_with_retry

SYSTEM_SID = 'S-1-5-18'
ADMINISTRATORS_SID = 'S-1-5-32-544'
OWNER_RIGHTS_SID = 'S-1-3-4'  # the object's owner, which must itself be the user, SYSTEM or Administrators
_WINDOWS = sys.platform == 'win32'


@dataclass(frozen=True)
class AclSummary:
    """What is_owner_only() saw (tests and diagnostics; never a secret)."""
    owner_only: bool
    protected: bool | None  # Windows: SE_DACL_PROTECTED; None on POSIX
    owner: str | None
    allowed: tuple[str, ...]  # SIDs (Windows) or ('mode:0o600',) (POSIX)
    reason: str | None = None


# ---------------------------------------------------------------------------------------------------------------------
# Windows

class _Win:
    """Lazily bound advapi32/kernel32 entry points."""
    _instance = None

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.wintypes = wintypes
        advapi32 = ctypes.WinDLL('advapi32', use_last_error=True)
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        HANDLE, DWORD, BOOL, LPVOID = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, ctypes.c_void_p
        PVOID_P = ctypes.POINTER(ctypes.c_void_p)

        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = HANDLE
        kernel32.CloseHandle.argtypes = [HANDLE]
        kernel32.CloseHandle.restype = BOOL
        kernel32.LocalFree.argtypes = [LPVOID]
        kernel32.LocalFree.restype = LPVOID

        advapi32.OpenProcessToken.argtypes = [HANDLE, DWORD, ctypes.POINTER(HANDLE)]
        advapi32.OpenProcessToken.restype = BOOL
        advapi32.GetTokenInformation.argtypes = [HANDLE, ctypes.c_int, LPVOID, DWORD, ctypes.POINTER(DWORD)]
        advapi32.GetTokenInformation.restype = BOOL
        advapi32.ConvertSidToStringSidW.argtypes = [LPVOID, ctypes.POINTER(wintypes.LPWSTR)]
        advapi32.ConvertSidToStringSidW.restype = BOOL
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            wintypes.LPCWSTR, DWORD, PVOID_P, ctypes.POINTER(wintypes.ULONG)]
        advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = BOOL
        advapi32.GetSecurityDescriptorDacl.argtypes = [LPVOID, ctypes.POINTER(BOOL), PVOID_P, ctypes.POINTER(BOOL)]
        advapi32.GetSecurityDescriptorDacl.restype = BOOL
        advapi32.GetSecurityDescriptorControl.argtypes = [LPVOID, ctypes.POINTER(wintypes.WORD), ctypes.POINTER(DWORD)]
        advapi32.GetSecurityDescriptorControl.restype = BOOL
        advapi32.SetNamedSecurityInfoW.argtypes = [wintypes.LPWSTR, ctypes.c_int, DWORD, LPVOID, LPVOID, LPVOID, LPVOID]
        advapi32.SetNamedSecurityInfoW.restype = DWORD
        advapi32.GetNamedSecurityInfoW.argtypes = [wintypes.LPCWSTR, ctypes.c_int, DWORD, PVOID_P, PVOID_P, PVOID_P,
                                                   PVOID_P, PVOID_P]
        advapi32.GetNamedSecurityInfoW.restype = DWORD
        advapi32.GetAclInformation.argtypes = [LPVOID, LPVOID, DWORD, ctypes.c_int]
        advapi32.GetAclInformation.restype = BOOL
        advapi32.GetAce.argtypes = [LPVOID, DWORD, PVOID_P]
        advapi32.GetAce.restype = BOOL

        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = [('nLength', DWORD), ('lpSecurityDescriptor', LPVOID), ('bInheritHandle', BOOL)]
        self.SECURITY_ATTRIBUTES = SECURITY_ATTRIBUTES
        kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, DWORD, DWORD, ctypes.POINTER(SECURITY_ATTRIBUTES), DWORD,
                                         DWORD, HANDLE]
        kernel32.CreateFileW.restype = HANDLE
        self.advapi32 = advapi32
        self.kernel32 = kernel32
        self._user_sid: str | None = None

    @classmethod
    def get(cls) -> '_Win':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def error(self, code: int | None = None) -> OSError:
        return self.ctypes.WinError(self.ctypes.get_last_error() if code is None else code)

    def sid_string(self, psid) -> str:
        ctypes, wintypes = self.ctypes, self.wintypes
        text = wintypes.LPWSTR()
        if not self.advapi32.ConvertSidToStringSidW(psid, ctypes.byref(text)):
            raise self.error()
        try:
            return text.value
        finally:
            self.kernel32.LocalFree(text)

    def user_sid(self) -> str:
        """The current process user's SID string, from GetTokenInformation(TokenUser)."""
        if self._user_sid is not None:
            return self._user_sid
        ctypes, wintypes = self.ctypes, self.wintypes
        token = wintypes.HANDLE()
        TOKEN_QUERY, TOKEN_USER_CLASS = 0x0008, 1
        if not self.advapi32.OpenProcessToken(self.kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)):
            raise self.error()
        try:
            size = wintypes.DWORD(0)
            self.advapi32.GetTokenInformation(token, TOKEN_USER_CLASS, None, 0, ctypes.byref(size))
            if not size.value:
                raise self.error()
            buffer = ctypes.create_string_buffer(size.value)
            if not self.advapi32.GetTokenInformation(token, TOKEN_USER_CLASS, buffer, size, ctypes.byref(size)):
                raise self.error()
            # TOKEN_USER { SID_AND_ATTRIBUTES User { PSID Sid; DWORD Attributes } }: the first field is the SID pointer
            psid = ctypes.c_void_p.from_buffer(buffer).value
            self._user_sid = self.sid_string(psid)
            return self._user_sid
        finally:
            self.kernel32.CloseHandle(token)

    def apply_sddl(self, path: str, sddl: str) -> None:
        ctypes, wintypes = self.ctypes, self.wintypes
        psd = ctypes.c_void_p()
        if not self.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(psd), None):
            raise self.error()
        try:
            present, defaulted = wintypes.BOOL(), wintypes.BOOL()
            pdacl = ctypes.c_void_p()
            if not self.advapi32.GetSecurityDescriptorDacl(psd, ctypes.byref(present), ctypes.byref(pdacl),
                                                           ctypes.byref(defaulted)):
                raise self.error()
            SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION = 1, 0x4, 0x80000000
            flags = DACL_SECURITY_INFORMATION | (PROTECTED_DACL_SECURITY_INFORMATION if sddl.startswith('D:P') else 0)
            code = self.advapi32.SetNamedSecurityInfoW(path, SE_FILE_OBJECT, flags, None, None, pdacl, None)
            if code:
                raise self.error(code)
        finally:
            self.kernel32.LocalFree(psd)

    def create_new(self, path: str, sddl: str) -> int:
        """CreateFileW(CREATE_NEW) with the SDDL as the file's security descriptor, so the file is owner-only from the
        moment it exists (no window with the parent's inherited DACL). Returns a C runtime fd opened for writing."""
        import msvcrt
        ctypes = self.ctypes
        psd = ctypes.c_void_p()
        if not self.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(psd), None):
            raise self.error()
        try:
            attributes = self.SECURITY_ATTRIBUTES(ctypes.sizeof(self.SECURITY_ATTRIBUTES), psd, False)
            GENERIC_WRITE, CREATE_NEW, FILE_ATTRIBUTE_NORMAL = 0x40000000, 1, 0x80
            handle = self.kernel32.CreateFileW(path, GENERIC_WRITE, 0, ctypes.byref(attributes), CREATE_NEW,
                                               FILE_ATTRIBUTE_NORMAL, None)
            if handle is None or handle == ctypes.c_void_p(-1).value:
                raise self.error()
        finally:
            self.kernel32.LocalFree(psd)
        try:
            return msvcrt.open_osfhandle(handle, os.O_WRONLY | getattr(os, 'O_BINARY', 0))
        except OSError:
            self.kernel32.CloseHandle(handle)
            raise

    def summary(self, path: str) -> AclSummary:
        ctypes, wintypes = self.ctypes, self.wintypes
        SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION, DACL_SECURITY_INFORMATION = 1, 0x1, 0x4
        owner, dacl, psd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        code = self.advapi32.GetNamedSecurityInfoW(path, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                                                   ctypes.byref(owner), None, ctypes.byref(dacl), None, ctypes.byref(psd))
        if code:
            raise self.error(code)
        try:
            control, revision = wintypes.WORD(), wintypes.DWORD()
            if not self.advapi32.GetSecurityDescriptorControl(psd, ctypes.byref(control), ctypes.byref(revision)):
                raise self.error()
            protected = bool(control.value & 0x1000)  # SE_DACL_PROTECTED
            owner_sid = self.sid_string(owner) if owner.value else None
            owners = {self.user_sid(), SYSTEM_SID, ADMINISTRATORS_SID}
            allowed_set = owners | {OWNER_RIGHTS_SID}
            if not dacl.value:
                return AclSummary(False, protected, owner_sid, (), 'NULL DACL grants everyone')

            class ACL_SIZE_INFORMATION(ctypes.Structure):
                _fields_ = [('AceCount', wintypes.DWORD), ('AclBytesInUse', wintypes.DWORD),
                            ('AclBytesFree', wintypes.DWORD)]
            info = ACL_SIZE_INFORMATION()
            if not self.advapi32.GetAclInformation(dacl, ctypes.byref(info), ctypes.sizeof(info), 2):
                raise self.error()
            allowed: list[str] = []
            reason = None
            for index in range(info.AceCount):
                pace = ctypes.c_void_p()
                if not self.advapi32.GetAce(dacl, index, ctypes.byref(pace)):
                    raise self.error()
                ace_type = ctypes.c_ubyte.from_address(pace.value).value
                if ace_type == 1:  # ACCESS_DENIED_ACE: never widens access
                    continue
                if ace_type != 0:  # object, callback or unknown allowed-ACE types: not understood, so not owner-only
                    reason = reason or f'unsupported ACE type {ace_type}'
                    allowed.append(f'<ace type {ace_type}>')
                    continue
                mask = ctypes.c_uint32.from_address(pace.value + 4).value
                sid = self.sid_string(pace.value + 8)  # ACCESS_ALLOWED_ACE: header(4) + mask(4) + SidStart
                allowed.append(sid)
                if mask and sid not in allowed_set:
                    reason = reason or 'an access-allowed ACE names another account'
            if owner_sid not in owners:
                reason = reason or 'the owner is another account'
            return AclSummary(reason is None, protected, owner_sid, tuple(allowed), reason)
        finally:
            self.kernel32.LocalFree(psd)


def _user_sddl(directory: bool, sid: str) -> str:
    inherit = 'OICI' if directory else ''
    return f'D:P(A;{inherit};FA;;;{sid})(A;{inherit};FA;;;SY)(A;{inherit};FA;;;BA)'


def current_user_sid() -> str:
    """Windows only: the current process user's SID string."""
    return _Win.get().user_sid()


def _apply_sddl(path, sddl: str) -> None:
    """Windows only: apply an SDDL DACL to path (tests use it to build a deliberately broad ACL)."""
    _Win.get().apply_sddl(str(path), sddl)


# ---------------------------------------------------------------------------------------------------------------------
# Public helpers

def make_owner_only(path, directory: bool = False) -> None:
    """Restrict path to the current user: a protected owner DACL (plus SYSTEM and Administrators) on Windows, 0o700 for
    directories and 0o600 for files on POSIX. Raises OSError on failure."""
    path = Path(path)
    if _WINDOWS:
        win = _Win.get()
        win.apply_sddl(str(path), _user_sddl(directory, win.user_sid()))
        return
    if path.is_symlink():
        raise OSError(f'Refusing to change permissions through a symlink: {path}')
    os.chmod(path, 0o700 if directory else 0o600)


def acl_summary(path) -> AclSummary:
    """The owner-only verdict with the details that produced it."""
    path = Path(path)
    if _WINDOWS:
        return _Win.get().summary(str(path))
    if path.is_symlink():
        return AclSummary(False, None, None, (), 'symlink')
    st = path.stat()
    mode = stat.S_IMODE(st.st_mode)
    reason = None
    if st.st_uid != os.geteuid():
        reason = 'the owner is another account'
    elif mode & 0o077:
        reason = 'group or other permission bits are set'
    return AclSummary(reason is None, None, str(st.st_uid), (f'mode:{oct(mode)}',), reason)


def is_owner_only(path) -> bool:
    """True when only the current user (and on Windows SYSTEM and Administrators) can access path. False when the path
    is missing or cannot be inspected."""
    try:
        return acl_summary(path).owner_only
    except OSError:
        return False


def write_owner_only(path, data: bytes) -> None:
    """Atomically replace path with data, owner-only from the first byte written. The parent directory is created
    owner-only when it does not exist. Windows PermissionError on replace is retried a bounded number of times."""
    path = Path(path)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError('write_owner_only takes bytes')
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        make_owner_only(path.parent, directory=True)
    tmp = path.with_name(f'.{path.name}.{secrets.token_hex(6)}.tmp')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    try:
        if _WINDOWS:
            win = _Win.get()
            # created WITH the protected owner-only DACL: no other account can open it at any moment
            fd = win.create_new(str(tmp), _user_sddl(False, win.user_sid()))
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            fd = os.open(tmp, flags, 0o600)
            with os.fdopen(fd, 'wb') as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        replace_with_retry(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


__all__ = ['ADMINISTRATORS_SID', 'AclSummary', 'OWNER_RIGHTS_SID', 'SYSTEM_SID', 'acl_summary', 'current_user_sid',
           'is_owner_only', 'make_owner_only', 'write_owner_only']
