"""Independent verifier proof for blocker 1 (containment before heavy work). Windows only (SKIP, exit 0 elsewhere).
1. A grandchild spawned by the contained child with plain subprocess.Popen cannot escape the cap: its allocation past a 256 MiB cap
   fails with MemoryError, the job's PeakJobMemoryUsed stays bounded and TotalProcesses counts the grandchild.
2. A grandchild spawned through a nested OwnedJob (the route the Titanos server uses for the isolated music adapter) is still under
   the outer cap.
3. A readback mismatch (QueryInformationJobObject reporting a different JobMemoryLimit) raises ContainmentError while the child is
   suspended: the child never runs (its marker file is never written) and the job handle is closed.
4. A missing JOB_OBJECT_LIMIT_JOB_MEMORY flag on readback is refused the same way.
Small synthetic children only; nothing loads a model or touches a GPU."""
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'matrix_kit'))
import containment  # noqa: E402

if os.name != 'nt' or not containment.available():
    print(json.dumps({'skipped': True, 'reason': 'Windows Job Object containment is not available on this host', 'os': os.name}))
    sys.exit(0)

MIB = 1 << 20
failures = []


def expect(cond, msg):
    if not cond:
        failures.append(msg)


GRANDCHILD_ALLOC = ("import sys\nblocks = []\ntry:\n    for i in range(16):\n        blocks.append(bytearray(64 << 20))\nexcept MemoryError:\n"
                    "    print('grandchild MemoryError after', len(blocks), flush=True)\n    sys.exit(3)\nprint('grandchild allocated', len(blocks), flush=True)\n")
CHILD_PLAIN = ("import subprocess, sys\nr = subprocess.run([sys.executable, '-c', %r])\nprint('child saw grandchild rc', r.returncode, flush=True)\nsys.exit(r.returncode)\n" % GRANDCHILD_ALLOC)
CHILD_NESTED = ("import importlib.util, sys\nspec = importlib.util.spec_from_file_location('wj', %r)\nm = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(m)\n"
                "with m.OwnedJob([sys.executable, '-c', %r]) as job:\n    rc = job.process.wait(120)\nprint('child saw nested grandchild rc', rc, flush=True)\nsys.exit(rc)\n"
                % (str(ROOT / 'src' / 'titanos' / '_windows_job.py'), GRANDCHILD_ALLOC))

observations = {'parent_in_job': containment.current_process_in_job()}
for label, body in (('plain_popen_grandchild', CHILD_PLAIN), ('nested_ownedjob_grandchild', CHILD_NESTED)):
    job = containment.ContainedJob([sys.executable, '-c', body], 256 * MIB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = job.process.communicate(timeout=180)
    live = job.query()
    evidence = job.close()
    peak = max(live['peak_job_commit_bytes'], (evidence.get('final_query') or {}).get('peak_job_commit_bytes', 0))
    total = (evidence.get('final_query') or {}).get('total_processes')
    expect(job.readback and job.readback['verified_before_resume'] and job.readback['job_memory_limit_bytes'] == 256 * MIB, f'{label}: cap verified before resume: {job.readback}')
    # the nested OwnedJob does not inherit our stdout pipe (close_fds), so its grandchild's text is absent; the child relays the rc 3
    expect(job.process.returncode == 3 and ('grandchild MemoryError' in out or 'grandchild rc 3' in out), f'{label}: the grandchild must hit the cap: rc {job.process.returncode} out {out.strip()[:300]}')
    expect(128 * MIB <= peak <= 256 * MIB + 64 * MIB, f'{label}: job peak {peak >> 20} MiB must lie between 128 MiB and cap + 64 MiB')
    expect(total is not None and total >= 2, f'{label}: the grandchild must be counted in the job: total_processes {total}')
    expect(evidence['active_processes_after'] == 0 and evidence['job_handle_closed'], f'{label}: clean close: {evidence}')
    observations[label] = {'peak_mib': peak >> 20, 'total_processes': total, 'rc': job.process.returncode, 'out': out.strip()[:200]}

# 3/4. readback mismatch while suspended: nothing runs, the handle is closed
kernel = containment._V.kernel
real_query = kernel.QueryInformationJobObject


def make_lying_query(mutate):
    def lying(handle, cls, buf, size, ret):
        ok = real_query(handle, cls, buf, size, ret)
        if ok and cls == containment.EXTENDED_LIMITS:
            limits = ctypes.cast(buf, ctypes.POINTER(containment._V.ExtendedLimits)).contents
            mutate(limits)
        return ok
    return lying


def lower_limit(limits):
    limits.JobMemoryLimit = limits.JobMemoryLimit + MIB


def drop_flag(limits):
    limits.BasicLimitInformation.LimitFlags = limits.BasicLimitInformation.LimitFlags & ~containment.JOB_MEMORY


for label, mutate, needle in (('limit_mismatch', lower_limit, 'limit'), ('flag_missing', drop_flag, 'flags')):
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / 'ran.txt'
        kernel.QueryInformationJobObject = make_lying_query(mutate)
        try:
            try:
                containment.ContainedJob([sys.executable, '-c', f"open({str(marker)!r}, 'w').write('ran')"], 256 * MIB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                expect(False, f'{label}: a readback mismatch must raise ContainmentError')
                owned = None
            except containment.ContainmentError as exc:
                owned = getattr(exc, 'owned_job', None)
                expect('readback mismatch before resume' in str(exc) and needle in str(exc), f'{label}: mismatch text: {exc}')
                expect(owned is not None and owned.handle is None and owned.cleanup['job_handle_closed'] and owned.process is not None and owned.process.poll() is not None,
                       f'{label}: the suspended child must be killed and the job closed: {getattr(owned, "cleanup", None)}')
                expect(owned is not None and owned.readback is not None and owned.readback['verified_before_resume'] is False, f'{label}: readback recorded as unverified: {getattr(owned, "readback", None)}')
                observations[label] = {'error': str(exc)[:200], 'cleanup': owned.cleanup if owned else None}
        finally:
            kernel.QueryInformationJobObject = real_query
        time.sleep(0.3)
        expect(not marker.exists(), f'{label}: the child must never have run (marker {marker} exists)')

assert not failures, '\n'.join(failures)
print(json.dumps({'passed': True, 'observations': observations}, default=str))
