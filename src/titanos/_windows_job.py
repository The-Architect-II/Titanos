"""Bound a newly launched Windows child and all descendants to one private job."""
import ctypes
from ctypes import wintypes as w
import subprocess
import time

class BasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', w.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', ctypes.c_uint64 * 6),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class Accounting(ctypes.Structure):
    _fields_ = [('Times', ctypes.c_int64 * 4), ('TotalPageFaultCount', w.DWORD),
                ('TotalProcesses', w.DWORD), ('ActiveProcesses', w.DWORD), ('TotalTerminatedProcesses', w.DWORD)]


kernel = ctypes.WinDLL('kernel32', use_last_error=True)
native = ctypes.WinDLL('ntdll')
native.NtResumeProcess.argtypes = [w.HANDLE]
native.NtResumeProcess.restype = w.LONG
for name, args, result in [
    ('CreateJobObjectW', [ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
    ('SetInformationJobObject', [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
    ('AssignProcessToJobObject', [w.HANDLE, w.HANDLE], w.BOOL),
    ('QueryInformationJobObject', [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
    ('TerminateJobObject', [w.HANDLE, w.UINT], w.BOOL),
    ('CloseHandle', [w.HANDLE], w.BOOL),
]:
    function = getattr(kernel, name)
    function.argtypes, function.restype = args, result


def require(ok):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


class OwnedJob:
    def __init__(self, argv, **kwargs):
        flags = kwargs.pop('creationflags', 0)
        if flags & 0x01000000 or kwargs.get('close_fds') is False:
            raise ValueError('No breakaway or unrelated handle inheritance is permitted')
        kwargs['close_fds'] = True
        self.handle = kernel.CreateJobObjectW(None, None)
        require(self.handle)
        self.process = None
        self.cleanup_errors = []
        try:
            limits = ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE, no breakaway.
            require(kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
            self.process = subprocess.Popen(argv, creationflags=flags | 4 | subprocess.CREATE_NO_WINDOW, **kwargs)
            # CPython's retained process handle is the exact new suspended process.
            self._assign()
            status = native.NtResumeProcess(int(self.process._handle))
            if status < 0:
                raise OSError(f'Owned process resume failed (NTSTATUS 0x{status & 0xffffffff:08x})')
        except BaseException as primary:
            failures = []
            if self.process is not None:
                # The retained process handle targets only this newly created child.
                self._cleanup_step('kill_child', self.process.kill, failures)
                self._cleanup_step('wait_child', lambda: self.process.wait(timeout=10), failures)
            self._cleanup_step('close_job', self._close_handle, failures)
            self._retain_cleanup(primary, failures)
            raise

    def _cleanup_step(self, step, action, failures):
        try:
            action()
        except BaseException as exc:
            if not hasattr(self, 'cleanup_errors'):
                self.cleanup_errors = []
            self.cleanup_errors.append({'step': step, 'type': type(exc).__name__, 'message': str(exc)})
            failures.append(exc)

    def _close_handle(self):
        if self.handle is not None:
            require(kernel.CloseHandle(self.handle))
            self.handle = None  # Only a confirmed CloseHandle clears ownership.

    def _retain_cleanup(self, primary, failures):
        # Constructor failure otherwise loses the only reference to a suspended
        # child/uncertain Job. Expose the owned object for explicit recovery.
        primary.owned_job = self
        for exc in failures:
            primary.add_note('Additional owned-job cleanup failure: ' + repr(exc))

    def _assign(self):
        require(kernel.AssignProcessToJobObject(self.handle, int(self.process._handle)))

    def active_count(self):
        info = Accounting()
        require(kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return info.ActiveProcesses

    def terminate(self, timeout=10):
        require(kernel.TerminateJobObject(self.handle, 1))
        deadline = time.monotonic() + timeout
        while self.active_count():
            if time.monotonic() >= deadline:
                raise TimeoutError('Owned job did not empty within cleanup deadline')
            time.sleep(.02)
        return self.process.wait(timeout=max(.01, deadline-time.monotonic()))

    def __enter__(self):
        return self

    def __exit__(self, kind, primary, tb):
        failures = []
        def finish():
            if self.active_count():
                self.terminate()
            else:
                self.process.wait(timeout=10)
        self._cleanup_step('finish_child', finish, failures)
        self._cleanup_step('close_job', self._close_handle, failures)
        if failures:
            if primary is not None:
                self._retain_cleanup(primary, failures)
            else:
                failure = failures[0]
                self._retain_cleanup(failure, failures[1:])
                raise failure
        return False
