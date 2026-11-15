"""Contained admission for the matrix kit: an enforceable owned process-tree commit cap, active resource monitors, and the pure
qualification rules that keep resource evidence separate from output correctness.

A lowered host-memory admission is honest only when it is enforced, not declared. The kit therefore never takes a caller-supplied
byte estimate as an admission input; the only lower floor it knows is a Windows Job Object commit cap set on the Titanos server
while that server is still suspended (before any weight allocation) and read back before it resumes. Every descendant the server
spawns (the isolated music adapter is a grandchild) joins the same job, so the cap bounds the whole owned tree and nothing else.

The bound is commit-only (receipt field resource_qualification.bound = 'job_commit'): JOB_OBJECT_LIMIT_JOB_MEMORY caps, and
PeakJobMemoryUsed measures, the commit charge of the tree. What that cap does with each allocation class was measured on this host
on 2026-09-25 with synthetic children (CAP_SEMANTICS below; control records control/music-gate-20260925/JOB_CAP_SEMANTICS_2159Z.json
and NESTED_JOB_GPU_COMMIT_TEST_*.json in the lab records, kept internal; tests/check_matrix_kit_containment.py reproduces the gaps):
  enforced fail-closed at allocation time: heap/VirtualAlloc commit, CUDA device allocations (WDDM charges them 1:1 to the process
    commit, so torch raises OutOfMemoryError with the GPU nearly empty), pinned host memory, safetensors safe_open (OSError 1455 at
    open, the whole shard file charged) and copy-on-write file-mapping views made through MapViewOfFile FILE_MAP_COPY / Python mmap
    ACCESS_COPY (OSError 1455 at map time);
  accounted but NOT refused: copy-on-write file mappings created by torch.UntypedStorage.from_file(shared=False): the view is created
    although its charge exceeds the cap, PeakJobMemoryUsed records the charge, the process commit reading does not move, and the first
    write to a mapped page kills the process natively with 0xC0000005 (the test reproduces the exit code run r3 showed; r3's DiT load
    went through from_pretrained/safe_open, it is not shown which class r3 hit, and its cause stays unproven:
    ROOT_R3_PEAK_AND_RELIEF_SCOPE_CORRECTION.json, ROOT_R3_NATIVE_CRASH_TIMELINE_CORRECTION.json);
  neither accounted nor refused: anonymous pagefile-backed sections (mmap(-1, n)) and shared writable file-mapping views.
PeakJobMemoryUsed also includes charges that were attempted and refused, so a peak over the cap is evidence of an attempted charge, not
proof that committed memory exceeded the cap. The cap is therefore a fail-closed bound only for the enforced classes; the CellGuard stays
the active monitor (STOP, cap hit, host commit and physical starvation, GPU starvation), the admission and the guard sample the host's
available physical memory (GlobalMemoryStatusEx ullAvailPhys) beside available commit, the tree's peak working set (psutil, best
effort) is recorded uncapped, and the owned child must avoid the two gap classes (the adapter's streamed loader reads tensors by
explicit file reads instead of mapping whole shards). Every contained receipt carries CAP_SEMANTICS verbatim as
resource_qualification.cap_semantics so it states what its bound enforced and what it could not.

    ContainedJob(argv, limit_bytes, **popen_kwargs)   the server tree under a verified cap; query() peaks, close() cleanup evidence
                                                       (any OSError the OS raises while the child is suspended becomes a ContainmentError)
    CellGuard(...).check(phase)                        STOP, missing readings, cap hit, host (commit or physical) or GPU starvation -> GuardTrip
    derive_job_limit / qualification_verdict / profile_key   pure rules, tested without a job

OwnedJob (src/titanos/_windows_job.py) is loaded by file path so the kit does not import the titanos package here.
"""
import ctypes
import hashlib
import importlib.util
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB_SOURCE = ROOT / 'src' / 'titanos' / '_windows_job.py'
JOB_MEMORY, KILL_ON_CLOSE, BREAKAWAY = 0x200, 0x2000, 0x1800  # JOB_OBJECT_LIMIT_JOB_MEMORY, _KILL_ON_JOB_CLOSE, _BREAKAWAY_OK | _SILENT_BREAKAWAY_OK
REQUIRED_FLAGS = JOB_MEMORY | KILL_ON_CLOSE
EXTENDED_LIMITS, ACCOUNTING = 9, 1  # JobObjectExtendedLimitInformation, JobObjectBasicAccountingInformation
MIB = 1 << 20

# What JOB_OBJECT_LIMIT_JOB_MEMORY does with each allocation class, measured on this Windows host on 2026-09-25 with synthetic children
# in owned jobs (control records control/music-gate-20260925/JOB_CAP_SEMANTICS_2159Z.json and the NESTED_JOB_GPU_COMMIT_TEST_*.json
# beside it in the lab records, kept internal; tests/check_matrix_kit_containment.py cases 10a-10c reproduce the gaps and the refusal
# accounting). Written verbatim into every contained receipt's resource_qualification.cap_semantics: a contained cell claims exactly this.
CAP_SEMANTICS = {
    'bound': 'job_commit',
    'enforced': [  # refused fail-closed when the charge would pass the cap; the attempted charge still appears in PeakJobMemoryUsed
        'heap and VirtualAlloc commit: MemoryError / DefaultCPUAllocator not enough memory at the cap',
        'CUDA device allocations under WDDM: charged 1:1 to the process commit, torch OutOfMemoryError although the GPU has memory free',
        'pinned host memory: cudaHostAlloc / pin_memory fails at the cap',
        'safetensors safe_open on cpu or cuda: OSError 1455 at open, the whole shard file is charged',
        'copy-on-write file-mapping views through MapViewOfFile FILE_MAP_COPY, including Python mmap ACCESS_COPY: OSError 1455 at map time, the view charged',
    ],
    'accounted_not_refused': [
        'copy-on-write file mappings created by torch.UntypedStorage.from_file(shared=False): the view is created although its charge exceeds the cap, PeakJobMemoryUsed records the charge, the process commit reading does not move, and the first write to a mapped page kills the process natively with 0xC0000005 (the same exit code run r3 showed; r3 loaded its DiT through from_pretrained/safe_open, it is not shown which class r3 hit, and its cause stays unproven: ROOT_R3_PEAK_AND_RELIEF_SCOPE_CORRECTION.json)',
    ],
    'unaccounted': [
        'anonymous pagefile-backed sections (mmap(-1, n), CreateFileMapping over INVALID_HANDLE_VALUE): neither counted in PeakJobMemoryUsed nor refused',
        'shared writable file-mapping views (FILE_MAP_ALL_ACCESS; pages backed by the file itself): neither counted nor refused',
    ],
    'peak_semantics': 'PeakJobMemoryUsed includes charges that were attempted and refused; a peak over the cap is evidence of an attempted charge, not proof that committed memory exceeded the cap',
    'monitor': 'CellGuard remains the active monitor (STOP, cap hit, host commit and physical starvation, GPU starvation); the owned child must avoid the accounted_not_refused and unaccounted classes (the adapter streams tensors by explicit file reads instead of mapping whole shards)',
    'measured': '2026-09-25, Windows 11 WDDM host, synthetic children in owned Job Objects (JOB_CAP_SEMANTICS_2159Z.json, NESTED_JOB_GPU_COMMIT_TEST_*.json, tests/check_matrix_kit_containment.py)',
}


class ContainmentError(RuntimeError):
    """The owned tree could not be capped as required; the caller must not run the cell uncontained under a lowered floor."""


class GuardTrip(RuntimeError):
    """An active monitor tripped. kind is 'stopped' (the STOP file) or 'budget_violated' (a missing reading, a cap hit, host or
    GPU starvation); phase names what the cell was doing. str() is the receipt error text '<kind>_during_<phase>: <reason>'."""

    def __init__(self, kind, phase, reason):
        self.kind, self.phase, self.reason = kind, phase, reason
        super().__init__(f'{kind}_during_{phase}: {reason}')


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_vendor():
    """src/titanos/_windows_job.py by file path (its module body binds kernel32/ntdll, so Windows only)."""
    if os.name != 'nt':
        raise ContainmentError('Windows Job Objects are unavailable on this platform')
    spec = importlib.util.spec_from_file_location('matrix_kit_windows_job', JOB_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    module.kernel.IsProcessInJob.restype = ctypes.c_int
    module.kernel.GetCurrentProcess.restype = ctypes.c_void_p
    return module


try:
    _V = load_vendor() if os.name == 'nt' else None
except Exception:  # noqa: BLE001  (a host without the vendor module or kernel bindings is simply not contained)
    _V = None


def available():
    """True when this host can create a Job Object (Windows and CreateJobObjectW works); the caller falls back to the full artifact
    floor otherwise and records the reason."""
    if _V is None:
        return False
    try:
        handle = _V.kernel.CreateJobObjectW(None, None)
        if not handle:
            return False
        _V.kernel.CloseHandle(handle)
        return True
    except Exception:  # noqa: BLE001
        return False


def current_process_in_job():
    """Whether the kit itself already runs inside a job (a nested assignment then; Windows 8+ allows it). None off Windows."""
    if _V is None:
        return None
    flag = ctypes.c_int(0)
    if not _V.kernel.IsProcessInJob(_V.kernel.GetCurrentProcess(), None, ctypes.byref(flag)):
        return None
    return bool(flag.value)


class ContainedJob(_V.OwnedJob if _V else object):
    """OwnedJob whose job carries JOB_OBJECT_LIMIT_JOB_MEMORY | KILL_ON_JOB_CLOSE and a commit cap, set and read back inside
    _assign() while the child is still suspended (the vendor resumes it only after _assign returns). Any readback mismatch raises
    ContainmentError, on which the vendor kills the suspended child and closes the job (nothing ever ran). An OSError the OS raises
    once the child exists (SetInformationJobObject, AssignProcessToJobObject, QueryInformationJobObject or NtResumeProcess refused,
    for example a nested-job assignment denied) is converted into a ContainmentError with the WinError chained and this object as
    exc.owned_job, after the vendor's own kill-and-close cleanup; a failure to create the child at all (Popen) stays an OSError."""

    def __init__(self, argv, limit_bytes, **kwargs):
        if _V is None:
            raise ContainmentError('Windows Job Objects are unavailable on this platform')
        if not isinstance(limit_bytes, int) or limit_bytes <= 0:
            raise ContainmentError(f'job commit limit must be a positive integer: {limit_bytes!r}')
        self.unaligned_limit_bytes = limit_bytes
        self.limit_bytes = page_aligned(limit_bytes)  # the kernel keeps the limit in pages; request what it will read back
        if self.limit_bytes <= 0:
            raise ContainmentError(f'job commit limit below one page: {limit_bytes!r}')
        self.readback = None
        self.handle, self.process, self.cleanup_errors = None, None, []
        self.cleanup = {'active_processes_after': None, 'job_handle_closed': False, 'forced_cleanup': False}
        try:
            super().__init__(argv, **kwargs)
        except ContainmentError:
            self.cleanup['job_handle_closed'] = self.handle is None
            self.cleanup['vendor_cleanup_errors'] = list(self.cleanup_errors)
            raise
        except OSError as exc:
            self.cleanup['job_handle_closed'] = self.handle is None
            self.cleanup['vendor_cleanup_errors'] = list(self.cleanup_errors)
            if self.process is None:
                raise  # the child was never created: not a containment failure
            error = ContainmentError(f'the host refused the owned job while the child was suspended: {type(exc).__name__}: {exc}')
            error.owned_job = self
            raise error from exc
        except BaseException:
            self.cleanup['job_handle_closed'] = self.handle is None
            raise

    def _extended(self):
        limits = _V.ExtendedLimits()
        _V.require(_V.kernel.QueryInformationJobObject(self.handle, EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits), None))
        return limits

    def _assign(self):
        limits = _V.ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = REQUIRED_FLAGS
        limits.JobMemoryLimit = self.limit_bytes
        _V.require(_V.kernel.SetInformationJobObject(self.handle, EXTENDED_LIMITS, ctypes.byref(limits), ctypes.sizeof(limits)))
        super()._assign()  # AssignProcessToJobObject on the exact suspended child
        actual = self._extended()
        flags = int(actual.BasicLimitInformation.LimitFlags)
        in_job = ctypes.c_int(0)
        member = bool(_V.kernel.IsProcessInJob(ctypes.c_void_p(int(self.process._handle)), self.handle, ctypes.byref(in_job))) and bool(in_job.value)
        self.readback = {'requested_limit_bytes': self.limit_bytes, 'unaligned_request_bytes': self.unaligned_limit_bytes, 'job_memory_limit_bytes': int(actual.JobMemoryLimit), 'limit_flags': flags,
                         'child_in_job': member, 'parent_in_job': current_process_in_job(), 'verified_before_resume': False, 'utc': utc()}
        if actual.JobMemoryLimit != self.limit_bytes or flags & REQUIRED_FLAGS != REQUIRED_FLAGS or flags & BREAKAWAY or not member:
            raise ContainmentError(f'job cap readback mismatch before resume: limit {int(actual.JobMemoryLimit)} (requested {self.limit_bytes}), '
                                   f'flags 0x{flags:x} (need 0x{REQUIRED_FLAGS:x} set, 0x{BREAKAWAY:x} clear), child in job {member}')
        self.readback['verified_before_resume'] = True  # the vendor resumes the child only after this returns

    def query(self):
        """Peaks and population of the owned tree (valid while the handle is open, after the processes exit too)."""
        ext = self._extended()
        acc = _V.Accounting()
        _V.require(_V.kernel.QueryInformationJobObject(self.handle, ACCOUNTING, ctypes.byref(acc), ctypes.sizeof(acc), None))
        return {'peak_job_commit_bytes': int(ext.PeakJobMemoryUsed), 'peak_process_commit_bytes': int(ext.PeakProcessMemoryUsed),
                'active_processes': int(acc.ActiveProcesses), 'total_processes': int(acc.TotalProcesses), 'job_memory_limit_bytes': int(ext.JobMemoryLimit)}

    def terminate(self, timeout=10):
        """End the owned tree (TerminateJobObject reaches only this job's processes) and wait until the job is empty."""
        self.cleanup['forced_cleanup'] = True
        return super().terminate(timeout)

    def close(self, timeout=30, grace_s=2.0):
        """Reap what remains of the owned tree, take the final peaks, close the handle. The tree gets grace_s to empty on its own
        after the launched process exits (a console host or a venv launcher's interpreter child tears down a moment later); only
        what is still alive after that is terminated (forced_cleanup). Returns the cleanup evidence; never raises."""
        evidence = self.cleanup
        try:
            if self.handle is None:
                evidence['job_handle_closed'] = True
                return evidence
            try:
                if self.process is not None and self.process.poll() is None:
                    try:
                        self.process.wait(timeout)
                    except Exception as exc:  # noqa: BLE001  (still running after the timeout: the tree below is terminated)
                        evidence['wait_error'] = f'{type(exc).__name__}: {exc}'
                deadline = time.monotonic() + grace_s
                while self.active_count() and time.monotonic() < deadline:
                    time.sleep(0.02)
                evidence['active_before_close'] = self.active_count()
                if evidence['active_before_close']:
                    self.terminate(timeout)
            except Exception as exc:  # noqa: BLE001
                evidence['cleanup_error'] = f'{type(exc).__name__}: {exc}'
            try:
                final = self.query()
                evidence.update(final_query=final, active_processes_after=final['active_processes'])
            except Exception as exc:  # noqa: BLE001
                evidence['final_query_error'] = f'{type(exc).__name__}: {exc}'
            try:
                self._close_handle()
            except Exception as exc:  # noqa: BLE001
                evidence['close_error'] = f'{type(exc).__name__}: {exc}'
            evidence['job_handle_closed'] = self.handle is None
            if self.process is not None:
                evidence['server_returncode'] = self.process.poll()
            return evidence
        finally:
            evidence['utc'] = utc()

    def __exit__(self, kind, primary, tb):
        self.close()
        return False


def derive_job_limit(first_available_bytes, reserve_bytes, profile_peak_bytes, margin_bytes):
    """The owned-tree commit cap and its source. A measured profile (a within-bound pass of the same key) gives 1.5 x peak + margin;
    without one the first run is capped at what the host can spare now minus the shared-host reserve. Never below zero. The reserve
    and the margin must be positive: a non-positive reserve would let the first-run cap reach or exceed what the host has and disable
    the starvation trip, so it is refused here as well as at the CLI."""
    if int(reserve_bytes) <= 0 or int(margin_bytes) <= 0:
        raise ValueError(f'the shared-host reserve and the margin must be positive: reserve {reserve_bytes}, margin {margin_bytes}')
    if profile_peak_bytes is not None and int(profile_peak_bytes) > 0:
        return page_aligned(int(int(profile_peak_bytes) * 1.5) + int(margin_bytes)), 'measured_profile'
    return page_aligned(max(0, int(first_available_bytes) - int(reserve_bytes))), 'first_run_containment_cap'


PAGE_BYTES = 4096


def page_aligned(limit_bytes):
    """JobMemoryLimit is kept by the kernel as a multiple of the page size: SetInformationJobObject rounds a value down and the
    readback then differs from the request (observed: 4,509,715,660 requested -> 4,509,712,384 read back), which the strict
    readback check treats as a mismatch. Caps are therefore aligned down to 4096 bytes before they are requested."""
    return (int(limit_bytes) // PAGE_BYTES) * PAGE_BYTES


def qualification_verdict(peak_bytes, limit_bytes, limit_hit, readings_complete):
    """'within' | 'exceeded' | 'unknown'. Unknown whenever the readings are incomplete or no peak was measured: an unknown is never
    a pass. A cap hit (or a peak at or over the cap) is exceeded whatever the exercise produced."""
    if not readings_complete or peak_bytes is None or limit_bytes is None:
        return 'unknown'
    if limit_hit or int(peak_bytes) >= int(limit_bytes):
        return 'exceeded'
    return 'within'


def profile_key(matrix_identity, receipt_sha256, adapter_sha256, placement, dtype, copies):
    """sha256 of the canonical JSON of the inputs that fix a measured peak's applicability: the row identity, the exact music-runtime
    receipt and adapter, the placement, the requested dtype and the number of weight copies."""
    canonical = json.dumps({'matrix_identity': matrix_identity, 'receipt_sha256': receipt_sha256, 'adapter_sha256': adapter_sha256,
                            'placement': placement, 'dtype': dtype, 'copies': int(copies)}, sort_keys=True, separators=(',', ':'), ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class CellGuard:
    """Active monitor of one admitted cell. check(phase) is called at least every 2 s during server start, model load and every
    active request (requests_catalog calls the guard callable in its wait loops). Every call checks STOP; readings are refreshed
    at most every interval_s (a 5 Hz healthz poll must not fork nvidia-smi five times a second) unless force is set.
    Trips: STOP present; a required reading missing (fail closed: available commit always, GPU free memory on a GPU slot, available
    physical memory whenever a reserve applies); job peak at or over the cap; available commit or available physical memory below
    the shared-host reserve (contained cells: reserve_bytes is set only for them); GPU free below half the margin (margin_bytes is
    passed only for contained cells; an artifact-floor cell records a low GPU reading without tripping, since its GPU floor is
    artifact bytes + margin and request-time activations legitimately use part of that margin). Samples are retained whatever
    happens. The commit figures are the job's commit charge; the tree's peak working set is recorded beside them, uncapped."""

    def __init__(self, stop_file, job, reserve_bytes, job_limit, gpu_uuid, gpu_floor_mib, margin_bytes, *, commit_reader, gpu_free_reader,
                 gpu_used_reader=None, phys_reader=None, clock=time.monotonic, interval_s=2.0, stop_exists=None):
        self.stop_file, self.job, self.reserve_bytes, self.job_limit = stop_file, job, reserve_bytes, job_limit
        self.gpu_uuid, self.gpu_floor_mib, self.margin_bytes = gpu_uuid, gpu_floor_mib, margin_bytes
        self.commit_reader, self.gpu_free_reader, self.gpu_used_reader, self.phys_reader = commit_reader, gpu_free_reader, gpu_used_reader, phys_reader
        self.clock, self.interval_s = clock, interval_s
        self.stop_exists = stop_exists or (lambda: bool(stop_file) and Path(stop_file).exists())
        self.samples, self.owned_work, self.trip = [], {}, None
        self.limit_hit, self.readings_complete = False, True
        self.peak_job, self.peak_process = None, None
        self._last, self._lock = None, threading.Lock()

    def note(self, **owned_work):
        """Record owned work the cell may have to cancel (the Titanos job id of a music request)."""
        self.owned_work.update({k: v for k, v in owned_work.items() if v is not None})

    def sample(self, phase):
        s = {'utc': utc(), 'phase': phase, 'available_commit_bytes': None, 'available_physical_bytes': None, 'gpu_free_mib': None, 'gpu_used_mib': None}
        try:
            s['available_commit_bytes'] = self.commit_reader()
        except Exception as exc:  # noqa: BLE001
            s['commit_error'] = f'{type(exc).__name__}: {exc}'
        if self.phys_reader is not None:
            try:
                s['available_physical_bytes'] = self.phys_reader()
            except Exception as exc:  # noqa: BLE001
                s['phys_error'] = f'{type(exc).__name__}: {exc}'
        if self.gpu_uuid:
            try:
                s['gpu_free_mib'] = self.gpu_free_reader(self.gpu_uuid)
            except Exception as exc:  # noqa: BLE001
                s['gpu_error'] = f'{type(exc).__name__}: {exc}'
            if self.gpu_used_reader is not None:
                try:
                    s['gpu_used_mib'] = self.gpu_used_reader(self.gpu_uuid)
                except Exception:  # noqa: BLE001
                    pass
        if self.job is not None:
            try:
                q = self.job.query()
                s.update(peak_job_commit_bytes=q['peak_job_commit_bytes'], peak_process_commit_bytes=q['peak_process_commit_bytes'], active_processes=q['active_processes'])
                self.peak_job = max(self.peak_job or 0, q['peak_job_commit_bytes'])
                self.peak_process = max(self.peak_process or 0, q['peak_process_commit_bytes'])
            except Exception as exc:  # noqa: BLE001
                s['job_error'] = f'{type(exc).__name__}: {exc}'
            s['tree_peak_working_set_bytes'] = tree_peak_working_set(getattr(self.job, 'process', None))
        return s

    def _tripped(self, kind, phase, reason):
        self.trip = {'kind': kind, 'phase': phase, 'reason': reason, 'utc': utc()}
        raise GuardTrip(kind, phase, reason)

    def check(self, phase, force=False):
        if self.trip is not None:
            raise GuardTrip(self.trip['kind'], self.trip['phase'], self.trip['reason'])
        if self.stop_exists():
            self._tripped('stopped', phase, f'STOP file present: {self.stop_file}')
        with self._lock:
            now = self.clock()
            if not force and self._last is not None and now - self._last < self.interval_s and self.samples:
                return self.samples[-1]
            self._last = now
            s = self.sample(phase)
            self.samples.append(s)
        missing = []
        if s['available_commit_bytes'] is None:
            missing.append('available commit')
        if self.gpu_uuid and s['gpu_free_mib'] is None:
            missing.append(f'GPU free memory ({self.gpu_uuid})')
        if self.job is not None and 'job_error' in s:
            missing.append('owned job peak (' + s['job_error'] + ')')
        if self.reserve_bytes is not None and s['available_physical_bytes'] is None:
            missing.append('available physical memory')
        if missing:
            self.readings_complete = False
            self._tripped('budget_violated', phase, 'resource_reading_unavailable: ' + ', '.join(missing))
        if self.job is not None and self.job_limit is not None and s.get('peak_job_commit_bytes', 0) >= self.job_limit:
            self.limit_hit = True
            self._tripped('budget_violated', phase, f"contained job commit peak {s['peak_job_commit_bytes']} reached the limit {self.job_limit}")
        if self.reserve_bytes is not None and s['available_commit_bytes'] < self.reserve_bytes:
            self._tripped('budget_violated', phase, f"host starvation: available commit {s['available_commit_bytes']} below the shared-host reserve {self.reserve_bytes}")
        if self.reserve_bytes is not None and s['available_physical_bytes'] < self.reserve_bytes:
            self._tripped('budget_violated', phase, f"host physical starvation: available physical memory {s['available_physical_bytes']} below the shared-host reserve {self.reserve_bytes} (copy-on-write file mappings and pagefile-backed sections are outside fail-closed enforcement)")
        if self.gpu_uuid and self.margin_bytes and s['gpu_free_mib'] < (self.margin_bytes // MIB) / 2:
            self._tripped('budget_violated', phase, f"gpu starvation: {s['gpu_free_mib']} MiB free on {self.gpu_uuid} below half the margin {self.margin_bytes // MIB} MiB")
        return s

    def summary(self):
        return {'samples': list(self.samples), 'measured_peak_job_commit_bytes': self.peak_job, 'measured_peak_process_commit_bytes': self.peak_process,
                'limit_hit': self.limit_hit, 'readings_complete': self.readings_complete, 'trip': self.trip, 'owned_work': dict(self.owned_work)}


def tree_peak_working_set(process):
    """Sum of the peak working sets of a launched process and its live descendants (psutil, Windows peak_wset; rss elsewhere), or None.
    An uncapped observation of physical memory the tree touched, including file-backed pages the commit cap does not see."""
    if process is None or getattr(process, 'pid', None) is None:
        return None
    try:
        import psutil
        root = psutil.Process(process.pid)
        total = 0
        for p in [root] + root.children(recursive=True):
            try:
                info = p.memory_info()
                total += int(getattr(info, 'peak_wset', None) or getattr(info, 'rss', 0) or 0)
            except Exception:  # noqa: BLE001  (a process that ended between the listing and the read)
                continue
        return total
    except Exception:  # noqa: BLE001
        return None


def guarded_call(guard, phase, fn, *args, poll_s=2.0, **kwargs):
    """Run one blocking call (a model load, a long request) in a worker thread while guard.check(phase) runs every poll_s. A trip
    propagates at once; the caller then cancels owned work, which ends the blocked call. Without a guard the call runs inline."""
    if guard is None:
        return fn(*args, **kwargs)
    box = {}

    def target():
        try:
            box['result'] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            box['error'] = exc
    worker = threading.Thread(target=target, name=f'matrix-kit-{phase}', daemon=True)
    guard.check(phase, force=True)
    worker.start()
    while worker.is_alive():
        worker.join(timeout=poll_s)
        if worker.is_alive():
            guard.check(phase)
    if 'error' in box:
        raise box['error']
    return box['result']
