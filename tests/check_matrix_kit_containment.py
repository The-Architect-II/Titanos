"""Matrix-kit containment contracts on Windows (prints SKIP and exits 0 elsewhere): a ContainedJob sets the owned-tree commit cap
and reads it back before the suspended child resumes; a synthetic child allocating past a 256 MiB cap fails with MemoryError while
the parent survives and the measured job peak lands between 128 MiB and cap + 64 MiB with verdict 'exceeded'; a child under a
512 MiB cap is 'within'; a CellGuard over a sleeping child trips on a STOP file, the owned job alone is terminated (active
processes after 0, handle closed) and the receipt shape stays untried with its samples; a guard sees a cap hit as a budget trip; a
missing reading fails closed; guarded_call surfaces a trip while a blocking call is in flight; a bad cap never launches anything.
Cap semantics (case 10, synthetic files and children under small caps, observations recorded): a copy-on-write FILE_MAP_COPY view
over the cap is charged to PeakJobMemoryUsed and refused or failed at touch, never fully touched; torch.UntypedStorage.from_file(shared=False)
over the cap (when torch is importable) is created without refusal, charged to the job peak, and dies natively at the first page write
(the same exit code r3 showed, 0xC0000005; r3's cause is unproven); an anonymous pagefile-backed section is written in full with the job peak far under the cap (unaccounted). Every
contained receipt carries containment.CAP_SEMANTICS. Nested-job behaviour is recorded (this test may itself run inside a job)."""
import hashlib
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

failures = []


def expect(cond, msg):
    if not cond:
        failures.append(msg)


MIB = 1 << 20
observations = {'parent_in_job': containment.current_process_in_job()}
ALLOC = ("import sys\nblocks = []\ntry:\n    for i in range({n}):\n        blocks.append(bytearray(64 << 20))\nexcept MemoryError:\n"
         "    print('MemoryError after', len(blocks), flush=True)\n    sys.exit(3)\nprint('allocated', len(blocks), flush=True)\n")


def readback_ok(rb, limit):
    return bool(rb and rb['verified_before_resume'] and rb['job_memory_limit_bytes'] == limit and rb['requested_limit_bytes'] == limit
                and rb['limit_flags'] & 0x2200 == 0x2200 and not rb['limit_flags'] & 0x1800 and rb['child_in_job'])


# 1. exceeded: 64 MiB steps up to 1 GiB under a 256 MiB cap -> the child fails, the parent survives, the peak is bounded
job = containment.ContainedJob([sys.executable, '-c', ALLOC.format(n=16)], 256 * MIB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
out, _ = job.process.communicate(timeout=120)
expect(readback_ok(job.readback, 256 * MIB), f'cap readback before resume: {job.readback}')
live = job.query()
evidence = job.close()
peak = max(live['peak_job_commit_bytes'], (evidence.get('final_query') or {}).get('peak_job_commit_bytes', 0))
expect(job.process.returncode == 3 and 'MemoryError' in out, f'the capped child must fail with MemoryError: rc {job.process.returncode} out {out.strip()[:200]}')
expect(128 * MIB <= peak <= 256 * MIB + 64 * MIB, f'job peak {peak >> 20} MiB must lie between 128 MiB and cap + 64 MiB')
verdict = containment.qualification_verdict(peak, 256 * MIB, peak >= 256 * MIB, True)
expect(verdict == 'exceeded', f'a capped-out child is exceeded, never a pass plus a note: peak {peak >> 20} MiB vs cap 256 MiB -> {verdict}')
expect(evidence['active_processes_after'] == 0 and evidence['job_handle_closed'] and not evidence['forced_cleanup'], f'clean close after the child exited (the tree gets a grace period to empty): {evidence}')
survivor = bytearray(64 * MIB)  # the parent is not bound by the child's job
del survivor
# the allocation happened in whatever the launched command spawned (a venv python.exe launcher runs the real interpreter as a child,
# and a hidden console host joins too): total_processes counts the whole tree the cap bounded, not one process
observations['exceeded'] = {'peak_mib': peak >> 20, 'returncode': job.process.returncode, 'readback': job.readback, 'processes_in_job': (evidence.get('final_query') or {}).get('total_processes')}

# 2. within: one 64 MiB block under a 512 MiB cap
job2 = containment.ContainedJob([sys.executable, '-c', ALLOC.format(n=1)], 512 * MIB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
out2, _ = job2.process.communicate(timeout=120)
evidence2 = job2.close()
peak2 = (evidence2.get('final_query') or {}).get('peak_job_commit_bytes', 0)
expect(job2.process.returncode == 0 and 'allocated 1' in out2, f'the child under a roomy cap completes: rc {job2.process.returncode} {out2.strip()[:100]}')
expect(64 * MIB <= peak2 < 512 * MIB and containment.qualification_verdict(peak2, 512 * MIB, False, True) == 'within', f'within verdict: peak {peak2 >> 20} MiB under 512 MiB')
expect(readback_ok(job2.readback, 512 * MIB), f'cap readback for the within case: {job2.readback}')
observations['within'] = {'peak_mib': peak2 >> 20}

# 3. a sleeping child whose guard sees a STOP file: GuardTrip, only the owned job terminated, receipt untried with samples retained
with tempfile.TemporaryDirectory() as tmp:
    stop = Path(tmp) / 'STOP'
    job3 = containment.ContainedJob([sys.executable, '-c', 'import time\ntime.sleep(600)\n'], 512 * MIB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    guard = containment.CellGuard(str(stop), job3, 1, 512 * MIB, None, None, 2 << 30, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: None, phys_reader=lambda: 100 << 30, interval_s=0)
    sample = guard.check('request', force=True)
    expect(sample['active_processes'] == 1 and sample['peak_job_commit_bytes'] > 0 and sample['phase'] == 'request', f'a live sample of the owned tree: {sample}')
    expect(sample['available_physical_bytes'] == 100 << 30 and isinstance(sample.get('tree_peak_working_set_bytes'), int) and sample['tree_peak_working_set_bytes'] > 0,
           f'the sample carries the physical-memory reading and the uncapped tree working set: {sample}')
    stop.write_text('stop', encoding='utf-8')
    trip = None
    try:
        guard.check('request', force=True)
    except containment.GuardTrip as exc:
        trip = exc
    expect(trip is not None and trip.kind == 'stopped' and trip.phase == 'request' and str(trip).startswith('stopped_during_request: STOP file present'), f'STOP trips the guard: {trip}')
    expect(job3.process.poll() is None, 'the trip itself does not kill anything; the cell cancels its owned work explicitly')
    job3.terminate(timeout=30)
    evidence3 = job3.close()
    expect(job3.process.poll() is not None and evidence3['forced_cleanup'] and evidence3['active_processes_after'] == 0 and evidence3['job_handle_closed'], f'owned job terminated and empty: {evidence3}')
    summary = guard.summary()
    receipt = {'state': 'untried', 'exercise': {'state': 'untried', 'error': str(trip)},
               'resource_qualification': {'kind': 'contained_resident', 'verdict': containment.qualification_verdict(summary['measured_peak_job_commit_bytes'], 512 * MIB, summary['limit_hit'], summary['readings_complete']),
                                          'samples': summary['samples'], 'cleanup': evidence3, 'trip': summary['trip']}}
    expect(receipt['resource_qualification']['verdict'] == 'within' and len(receipt['resource_qualification']['samples']) >= 1 and receipt['resource_qualification']['trip']['kind'] == 'stopped',
           f'the partial receipt keeps its samples and trip: {receipt["resource_qualification"]["trip"]} samples {len(receipt["resource_qualification"]["samples"])}')
    json.dumps(receipt, default=str)

# 4. a guard over a child that fills its cap: the cap hit is a budget trip, then the owned tree is reaped
job4 = containment.ContainedJob([sys.executable, '-c', 'import time\nblocks = []\ntry:\n    for i in range(16):\n        blocks.append(bytearray(64 << 20))\nexcept MemoryError:\n    pass\ntime.sleep(600)\n'],
                                256 * MIB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
guard4 = containment.CellGuard(None, job4, None, 256 * MIB, None, None, 2 << 30, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: None, interval_s=0)
trip4 = None
for _ in range(300):
    try:
        guard4.check('model_load', force=True)
    except containment.GuardTrip as exc:
        trip4 = exc
        break
    time.sleep(0.05)
expect(trip4 is not None and trip4.kind == 'budget_violated' and 'reached the limit 268435456' in str(trip4) and guard4.limit_hit, f'a cap hit trips the guard during load: {trip4}')
job4.terminate(timeout=30)
evidence4 = job4.close()
expect(evidence4['active_processes_after'] == 0 and evidence4['job_handle_closed'], f'cap-hit tree reaped: {evidence4}')
expect(containment.qualification_verdict(guard4.peak_job, 256 * MIB, guard4.limit_hit, guard4.readings_complete) == 'exceeded', 'a cap hit is exceeded')
observations['cap_hit'] = {'peak_mib': (guard4.peak_job or 0) >> 20, 'samples': len(guard4.samples)}

# 5. a missing reading fails closed (readings_complete false -> verdict unknown), a GPU reading is required only on a GPU slot
guard5 = containment.CellGuard(None, None, None, None, 'GPU-test', 4096, 2 << 30, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: None, interval_s=0)
try:
    guard5.check('request', force=True)
    expect(False, 'a missing GPU reading must trip')
except containment.GuardTrip as exc:
    expect(exc.kind == 'budget_violated' and 'resource_reading_unavailable' in str(exc) and not guard5.readings_complete, f'missing reading trips fail-closed: {exc}')
expect(containment.qualification_verdict(1, 2, False, guard5.readings_complete) == 'unknown', 'incomplete readings are unknown, never within')
guard5b = containment.CellGuard(None, None, None, None, None, None, 2 << 30, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: None, interval_s=0)
expect(guard5b.check('request', force=True)['available_commit_bytes'] == 100 << 30, 'a CPU cell needs no GPU reading')
# host starvation and GPU starvation trips
guard5c = containment.CellGuard(None, None, 10 << 30, None, None, None, 2 << 30, commit_reader=lambda: 5 << 30, gpu_free_reader=lambda u: None, phys_reader=lambda: 100 << 30, interval_s=0)
try:
    guard5c.check('request', force=True)
    expect(False, 'commit below the reserve must trip')
except containment.GuardTrip as exc:
    expect('host starvation' in str(exc), f'reserve trip: {exc}')
# physical memory below the reserve trips too (file-backed mappings are outside the commit cap); a reserve without a physical reading fails closed
guard5f = containment.CellGuard(None, None, 10 << 30, None, None, None, 2 << 30, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: None, phys_reader=lambda: 5 << 30, interval_s=0)
try:
    guard5f.check('request', force=True)
    expect(False, 'physical memory below the reserve must trip')
except containment.GuardTrip as exc:
    expect(exc.kind == 'budget_violated' and 'host physical starvation' in str(exc), f'physical reserve trip: {exc}')
guard5g = containment.CellGuard(None, None, 10 << 30, None, None, None, 2 << 30, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: None, interval_s=0)
try:
    guard5g.check('request', force=True)
    expect(False, 'a reserve without a physical-memory reading must fail closed')
except containment.GuardTrip as exc:
    expect('resource_reading_unavailable' in str(exc) and 'available physical memory' in str(exc) and not guard5g.readings_complete, f'missing physical reading: {exc}')
# no GPU-starvation trip without a margin (an artifact-floor cell): the low reading is recorded
guard5h = containment.CellGuard(None, None, None, None, 'GPU-test', 4096, None, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: 500, interval_s=0)
expect(guard5h.check('request', force=True)['gpu_free_mib'] == 500, 'without a margin a low GPU reading is a sample, not a trip')
guard5d = containment.CellGuard(None, None, None, None, 'GPU-test', 4096, 2 << 30, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: 500, interval_s=0)
try:
    guard5d.check('request', force=True)
    expect(False, 'GPU free below half the margin must trip')
except containment.GuardTrip as exc:
    expect('gpu starvation' in str(exc), f'gpu trip: {exc}')
# throttling: repeated checks within the interval reuse the last sample; STOP is still checked every call
calls = []
guard5e = containment.CellGuard(None, None, None, None, None, None, 2 << 30, commit_reader=lambda: calls.append(1) or (100 << 30), gpu_free_reader=lambda u: None, interval_s=60)
for _ in range(5):
    guard5e.check('server_start')
expect(len(calls) == 1 and len(guard5e.samples) == 1, f'readings are refreshed at most once per interval: {len(calls)} reads')

# 6. guarded_call: a trip surfaces while a blocking call is in flight
with tempfile.TemporaryDirectory() as tmp:
    stop = Path(tmp) / 'STOP'
    guard6 = containment.CellGuard(str(stop), None, None, None, None, None, 2 << 30, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: None, interval_s=0)

    def blocking():
        time.sleep(0.6)
        stop.write_text('x', encoding='utf-8')
        time.sleep(30)
        return 'never'
    t0 = time.monotonic()
    try:
        containment.guarded_call(guard6, 'model_load', blocking, poll_s=0.2)
        expect(False, 'guarded_call must raise the trip')
    except containment.GuardTrip as exc:
        expect(exc.kind == 'stopped' and exc.phase == 'model_load' and time.monotonic() - t0 < 10, f'trip surfaced while the call was blocked: {exc} after {time.monotonic() - t0:.1f}s')
    expect(containment.guarded_call(None, 'x', lambda: 7) == 7, 'no guard: the call runs inline')

# 7. a bad cap never launches anything; a zero/negative limit is refused before Popen
for bad in (0, -1, 2.5, None):
    try:
        containment.ContainedJob([sys.executable, '-c', 'print(1)'], bad)
        expect(False, f'limit {bad!r} must be refused')
    except containment.ContainmentError:
        pass

# 8b. the OS refusing the job while the child is suspended: the vendor's WinError becomes one ContainmentError carrying the owned job,
# the suspended child is killed and the handle closed (nothing ran); a child that cannot be created at all stays an OSError
import ctypes  # noqa: E402
kernel = containment._V.kernel
real_assign = kernel.AssignProcessToJobObject


def refuse_assign(handle, process):
    ctypes.set_last_error(5)  # ERROR_ACCESS_DENIED, as a refused nested-job assignment reports it
    return 0


kernel.AssignProcessToJobObject = refuse_assign
try:
    try:
        containment.ContainedJob([sys.executable, '-c', 'print(1)'], 256 * MIB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        expect(False, 'a refused assignment must raise')
    except containment.ContainmentError as exc:
        owned = getattr(exc, 'owned_job', None)
        expect(isinstance(exc.__cause__, OSError) and 'refused the owned job' in str(exc) and 'PermissionError' in str(exc), f'the WinError is chained into a ContainmentError: {exc!r} cause {exc.__cause__!r}')
        expect(owned is not None and owned.process is not None and owned.process.poll() is not None and owned.cleanup['job_handle_closed'] and owned.handle is None,
               f'the suspended child was killed and the job closed before the error surfaced: {getattr(owned, "cleanup", None)} rc {getattr(getattr(owned, "process", None), "returncode", None)}')
        observations['refused_assignment'] = {'error': str(exc)[:160], 'cleanup': owned.cleanup if owned else None}
    except OSError as exc:
        expect(False, f'an OSError must not escape ContainedJob once the child exists: {exc!r}')
finally:
    kernel.AssignProcessToJobObject = real_assign
try:
    containment.ContainedJob([str(Path(tempfile.gettempdir()) / 'no-such-interpreter-matrix-kit.exe'), '-c', 'print(1)'], 256 * MIB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    expect(False, 'a missing interpreter must raise')
except containment.ContainmentError as exc:
    expect(False, f'a child that was never created is not a containment failure: {exc!r}')
except OSError:
    pass

# 8. pure rules stay pure
expect(containment.derive_job_limit(40 << 30, 6 << 30, None, 2 << 30) == (34 << 30, 'first_run_containment_cap') and containment.derive_job_limit(40 << 30, 6 << 30, 4 << 30, 2 << 30) == (8 << 30, 'measured_profile'), 'derive_job_limit')
expect(len(containment.profile_key({'a': 1}, 'r', 'ad', 'resident', 'bfloat16', 1)) == 64, 'profile_key is a sha256 hex')

observations['nested_job'] = ('this test process already ran inside a job; the owned job was created as a nested job and the child joined it (child_in_job readback true)'
                              if observations['parent_in_job'] else 'this test process ran outside any job; the owned job was the child\'s only job')

# 9. run_cells.serve_and_exercise end to end with a sleeping fake 'server' (never Titanos: a `titanos` package placed in the cell work
# dir shadows the real one under `python -m titanos`, writes STOP when asked and sleeps): contained launch -> guard trip -> cancel ->
# job.close -> receipt; a tiny cap -> exceeded; a missing commit reading -> nothing launched, unknown; STOP before the launch -> nothing
# launched; the artifact floor + STOP -> orphans reaped through psutil; a refused job assignment -> a record, not an escaped error.
sys.path.insert(0, str(ROOT / 'tools' / 'matrix_kit'))
import run_cells  # noqa: E402
FAKE_SERVER = ("import os, sys, time\nfrom pathlib import Path\nstop = os.environ.get('KIT_TEST_STOP')\nif stop:\n    Path(stop).write_text('stop', encoding='utf-8')\n"
               "sys.stdout.write('fake server sleeping\\n')\nsys.stdout.flush()\ntime.sleep(600)\n")
cpu_slot = {'slot': 'win-x64-cpu', 'device': 'cpu', 'device_named': 'cpu', 'runtime': 'cpu'}


def fake_cell(tmp, name, write_stop):
    root = Path(tmp) / name
    (root / 'titanos').mkdir(parents=True)
    (root / 'titanos' / '__init__.py').write_text('', encoding='utf-8')
    (root / 'titanos' / '__main__.py').write_text(FAKE_SERVER, encoding='utf-8')
    stop = Path(tmp) / f'{name}.STOP'
    entry = {'model_id': 'fake', 'task': 'chat', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {}, 'env': {'KIT_TEST_STOP': str(stop)} if write_stop else {}}
    cfg, _ = run_cells.build_config(entry, cpu_slot, 'single', None, 'auto')
    return entry, cfg, root, stop


def contained_resource(job_limit):
    return {'kind': 'contained_resident', 'reason': 'first run', 'floors': {'commit': job_limit + 1, 'gpu': None}, 'margin_bytes': 2 << 30, 'admission_samples': [],
            'job_limit': job_limit, 'job_limit_source': 'first_run_containment_cap', 'reserve_bytes': 1, 'profile_key': hashlib.sha256(b'containment-profile').hexdigest()}


with tempfile.TemporaryDirectory() as tmp:
    # a. contained: the server tree writes STOP after it started; the guard trips during server start, the owned tree alone is reaped
    entry, cfg, root, stop = fake_cell(tmp, 'a', write_stop=True)
    rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', None, 30, stop_file=str(stop), resource=contained_resource(512 * MIB))
    rq = rec['resource_qualification']
    expect(rec['state'] == 'untried' and rec['error'].startswith('stopped_during_server_start: STOP file present') and rec['exercise']['state'] == 'untried',
           f"a STOP while the contained tree starts leaves the attempt untried with the trip: {rec['state']} {rec['error']}")
    expect(rq['kind'] == 'contained_resident' and rq['bound'] == 'job_commit' and rq['containment']['verified_before_resume'] and rq['job_commit_limit_bytes'] == 512 * MIB,
           f'the cap was verified before resume and the bound is recorded as commit-only: {rq.get("containment")}')
    expect(rq['cap_semantics'] == containment.CAP_SEMANTICS and 'driver_model' in rq and 'device_resident_estimate' in rq,
           f'every contained receipt states the cap semantics (enforced classes and the two gaps) and carries the driver-model / device-resident fields: {sorted(rq)}')
    expect(rq['cleanup']['active_processes_after'] == 0 and rq['cleanup']['job_handle_closed'] and rq['cleanup']['final_query']['total_processes'] >= 1 and rq['trip']['kind'] == 'stopped',
           f'the owned tree ended (on its own within the grace period after terminate, or forced) and the handle closed: {rq["cleanup"]}')
    expect(rq['verdict'] == 'within' and rq['measured_peak_job_commit_bytes'] and rq['measured_peak_job_commit_bytes'] < 512 * MIB and len(rq['samples']) >= 1 and rq['samples'][0]['phase'] == 'launch',
           f'samples retained from the launch check on; the sleeping tree stayed within the cap: {rq["verdict"]} peak {rq["measured_peak_job_commit_bytes"]} samples {len(rq["samples"])}')
    expect(rec['after_exit']['launched'] and isinstance(rec['after_exit']['available_commit_bytes'], int) and isinstance(rec['after_exit']['available_physical_bytes'], int),
           f'cleanup evidence carries host commit and physical readings after the tree ended: {rec["after_exit"]}')
    observations['serve_stop_contained'] = {'error': rec['error'][:120], 'cleanup': rq['cleanup'], 'samples': len(rq['samples'])}
    # b. a 1 MiB cap: the interpreter cannot start; the cap hit is exceeded resource evidence, never a fail or a pass
    entry, cfg, root, stop = fake_cell(tmp, 'b', write_stop=False)
    rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', None, 30, stop_file=str(stop), resource=contained_resource(1 * MIB))
    rq = rec['resource_qualification']
    expect(rec['state'] == 'untried' and rq['verdict'] == 'exceeded' and rq['limit_hit'] and (rec['error'].startswith('budget_violated_during_server_start') or rec['error'].startswith('host_resources_bound_exceeded')),
           f"a cap hit is 'exceeded' and untried: {rec['state']} {rec['error'][:160]} verdict {rq['verdict']} limit_hit {rq['limit_hit']}")
    expect(rq['cleanup']['active_processes_after'] == 0 and rq['cleanup']['job_handle_closed'], f'cap-hit tree reaped: {rq["cleanup"]}')
    # c. a missing commit reading: fail closed at the launch check, nothing is created, the qualification is unknown
    entry, cfg, root, stop = fake_cell(tmp, 'c', write_stop=False)
    real_commit = run_cells.available_commit_bytes
    run_cells.available_commit_bytes = lambda: None
    try:
        rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', None, 30, stop_file=str(stop), resource=contained_resource(512 * MIB))
    finally:
        run_cells.available_commit_bytes = real_commit
    rq = rec['resource_qualification']
    expect(rec['state'] == 'untried' and rec['error'].startswith('budget_violated_during_launch: resource_reading_unavailable') and rec['after_exit']['launched'] is False
           and rq['verdict'] == 'unknown' and rq['cleanup'].get('launched') is False and 'containment' not in rq,
           f"a missing reading before the launch creates nothing and stays unknown: {rec['state']} {rec['error'][:120]} launched {rec['after_exit']['launched']} verdict {rq['verdict']}")
    # d. STOP already present: the launch check trips, nothing is created
    entry, cfg, root, stop = fake_cell(tmp, 'd', write_stop=False)
    stop.write_text('stop', encoding='utf-8')
    rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', None, 30, stop_file=str(stop), resource=contained_resource(512 * MIB))
    expect(rec['state'] == 'untried' and rec['error'].startswith('stopped_during_launch: STOP file present') and rec['after_exit']['launched'] is False
           and rec['resource_qualification']['cleanup']['active_processes_after'] == 0 and rec['resource_qualification']['samples'] == [],
           f"a STOP before the launch creates no process: {rec['state']} {rec['error'][:120]} {rec['after_exit']}")
    # e. artifact floor + STOP written by the tree: untried with the trip, verdict not_contained, the psutil children snapshot reaps the orphans
    entry, cfg, root, stop = fake_cell(tmp, 'e', write_stop=True)
    rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', None, 30, stop_file=str(stop), resource={'kind': 'artifact_floor', 'floors': {'commit': 1, 'gpu': None}, 'margin_bytes': 2 << 30, 'admission_samples': []})
    rq = rec['resource_qualification']
    expect(rec['state'] == 'untried' and rec['error'].startswith('stopped_during_server_start') and rq['verdict'] == 'not_contained' and rq['bound'] is None and rec['server_returncode'] is not None,
           f"an artifact-floor cell honours STOP during server start: {rec['state']} {rec['error'][:120]} {rq['verdict']}")
    import psutil  # noqa: E402
    def _gone(pid):
        try:
            return not psutil.Process(pid).is_running()
        except psutil.NoSuchProcess:  # ended between the listing and this check
            return True
    expect(all(_gone(pid) for pid in rq['cleanup'].get('orphan_workers_killed') or []) and rq['cleanup']['job_handle_closed'] is None,
           f'orphans of an uncontained cell are reaped through the children snapshot: {rq["cleanup"]}')
    # f. the OS refuses the job assignment: a record is returned (untried containment_unavailable with the vendor cleanup), the run goes on
    entry, cfg, root, stop = fake_cell(tmp, 'f', write_stop=False)
    kernel.AssignProcessToJobObject = refuse_assign
    try:
        rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', None, 30, stop_file=str(stop), resource=contained_resource(512 * MIB))
    finally:
        kernel.AssignProcessToJobObject = real_assign
    rq = rec['resource_qualification']
    expect(rec['state'] == 'untried' and rec['error'].startswith('containment_unavailable: the host refused the owned job') and 'PermissionError' in rec['error']
           and rq['cleanup']['job_handle_closed'] and rq['cleanup'].get('launched') is False and rec['after_exit']['launched'] is False and rq['verdict'] == 'unknown',
           f"a refused job assignment is recorded, never run uncontained and never escapes: {rec['state']} {rec['error'][:160]} {rq['cleanup']}")
    observations['serve_refused_assignment'] = rec['error'][:160]

# 9g. a complete contained PASS on a fake chat HTTP server (never Titanos, no model, no GPU: the GPU readers are stubbed with constants
# for a fake GPU-... device string) -> the receipt the kit writes through cell_receipt is receipt_schema 2 with the complete measured
# block (cap readback verified before resume, peaks, samples across model_load and request, identity, cleanup), it passes
# ingest.validate_contained_block with no problems, and ingest.py binds it as an upheld pass in a kit_schema 2 run.
import ingest  # noqa: E402
FAKE_CHAT_SERVER = '''
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port = int(sys.argv[sys.argv.index('--port') + 1])
LOAD_STATUS = 200


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/healthz':
            return self.reply(200, {'ok': True})
        if self.path == '/api/models/loaded':
            return self.reply(200, {'data': [{'id': 'chat'}]})
        self.reply(404, {})

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n:
            self.rfile.read(n)
        if self.path == '/api/models/load' and LOAD_STATUS != 200:
            return self.reply(LOAD_STATUS, {'error': 'load refused by the fake server'})
        if self.path == '/v1/chat/completions':
            return self.reply(200, {'choices': [{'message': {'content': 'Hello.'}}]})
        self.reply(200, {'ok': True})


ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
'''
GATE_IDENTITY = {'receipt_sha256': hashlib.sha256(b'music-runtime.json').hexdigest(), 'adapter_sha256': hashlib.sha256(b'adapter').hexdigest(), 'placement': 'resident', 'dtype': 'auto', 'copies': 1}
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'g'
    (root / 'titanos').mkdir(parents=True)
    (root / 'titanos' / '__init__.py').write_text('', encoding='utf-8')
    (root / 'titanos' / '__main__.py').write_text(FAKE_CHAT_SERVER, encoding='utf-8')
    entry = {'model_id': 'fake', 'display_name': 'Fake', 'task': 'chat', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {}, 'matrix_identity': {'repo': 'fake'}}
    cfg, _ = run_cells.build_config(entry, cpu_slot, 'single', None, 'auto')
    resource = dict(contained_resource(512 * MIB), floors={'commit': 512 * MIB + 1, 'gpu': 4096}, profile_inputs=GATE_IDENTITY)
    real_free, real_used = run_cells.gpu_free_mib, run_cells.gpu_used_mib
    run_cells.gpu_free_mib, run_cells.gpu_used_mib = (lambda u: 20000), (lambda u: 12000)  # stubbed readings for the fake device string; nvidia-smi is never run
    try:
        rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', 'GPU-fake', 30, stop_file=str(Path(tmp) / 'g.STOP'), resource=resource)
    finally:
        run_cells.gpu_free_mib, run_cells.gpu_used_mib = real_free, real_used
    rq = rec['resource_qualification']
    expect(rec['state'] == 'pass' and rec['exercise']['state'] == 'pass' and rq['verdict'] == 'within' and rq['cleanup'].get('launched') is True,
           f"the fake chat server passes the exercise under the cap: {rec['state']} {rec.get('error')} verdict {rq['verdict']} {rq.get('reason')} cleanup {rq['cleanup']}")
    fake_gpu_slot = dict(cpu_slot, slot='win-x64-nvidia-cuda', device='GPU-fake', device_named='fake')
    receipt = run_cells.cell_receipt('kit-e2e', entry, fake_gpu_slot, 'single', None, {'hostname': 'test'}, [rec], 'pass', 'chat pass on the fake device', rec)
    problems = ingest.validate_contained_block(receipt['resource_qualification'], receipt)
    expect(receipt['receipt_schema'] == run_cells.RECEIPT_SCHEMA == 2 and problems == [], f'the receipt the kit writes for a contained pass satisfies the typed ingest gate: {problems}')
    expect(receipt['resource_qualification']['cap_semantics'] == containment.CAP_SEMANTICS, 'the kit-written pass receipt carries the cap semantics table verbatim')
    kit = Path(tmp) / 'kit'
    (kit / 'receipts').mkdir(parents=True)
    (kit / 'receipts' / 'fake.json').write_text(json.dumps(receipt, default=str), encoding='utf-8')
    # the RUN_RESULT row and run-level fields the kit writes (ingest binds the receipt to the row and the reserve to the run)
    row = {'model_id': 'fake', 'slot': 'win-x64-nvidia-cuda', 'mode': 'single', 'state': 'pass', 'device': 'GPU-fake', 'device_named': 'fake', 'receipt': 'receipts/fake.json', 'note': 'chat pass', 'dtype': 'auto'}
    run_result = {'run_id': 'kit-e2e', 'kit_schema': run_cells.RECEIPT_SCHEMA, 'utc': run_cells.utc(), 'cells': [row], 'host_reserve_bytes': resource['reserve_bytes'],
                  'floors': {'commit': 4 << 30, 'gpu': 4096}, 'containment_available': True}
    (kit / 'RUN_RESULT.json').write_text(json.dumps(run_result), encoding='utf-8')
    raw = Path(tmp) / 'raw.json'
    raw.write_text(json.dumps({'catalog': {}, 'families': [], 'mode_cells': []}), encoding='utf-8')
    vp = Path(tmp) / 'verdicts.json'
    vp.write_text(json.dumps({'fake|win-x64-nvidia-cuda|single': {'upheld': True, 'corrected_state': 'pass', 'reasoning': 'ok'}}), encoding='utf-8')
    out = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'ingest.py'), '--raw', str(raw), '--evidence', 'E-000', '--result', str(kit / 'RUN_RESULT.json'), '--verdicts', str(vp)],
                         capture_output=True, text=True, cwd=str(ROOT))
    data = json.loads(raw.read_text(encoding='utf-8'))
    fam = next((f for f in data['families'] if f['family'] == 'matrix-kit:kit-e2e'), {'cells': []})
    cell = next((c for c in fam['cells'] if c['model_id'] == 'fake'), {})
    expect(out.returncode == 0 and cell.get('state') == 'pass' and cell.get('verified') is True and fam.get('kit_schema') == 2,
           f'ingest binds the kit-written contained receipt as an upheld pass in a current run: rc {out.returncode} {cell} {out.stderr[-300:]}')
    # the same receipt with its measurement stripped back to identity (the owner's negative case) is refused by the same ingest
    thin = {k: receipt['resource_qualification'][k] for k in ('kind', 'bound', 'verdict', 'profile_key', *run_cells.IDENTITY_FIELDS)}
    (kit / 'receipts' / 'fake.json').write_text(json.dumps(dict(receipt, resource_qualification=thin), default=str), encoding='utf-8')
    out = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'ingest.py'), '--raw', str(raw), '--evidence', 'E-000', '--result', str(kit / 'RUN_RESULT.json'), '--verdicts', str(vp)],
                         capture_output=True, text=True, cwd=str(ROOT))
    data = json.loads(raw.read_text(encoding='utf-8'))
    cell = next((c for f in data['families'] for c in f['cells'] if c['model_id'] == 'fake'), {})
    expect(out.returncode == 0 and cell.get('state') == 'untried' and 'resource_qualification_evidence_incomplete' in (cell.get('note') or ''), f'identity without measurement is refused by the same gate: {cell}')
    observations['serve_contained_pass'] = {'peak_mib': (rq['measured_peak_job_commit_bytes'] or 0) >> 20, 'samples': len(rq['samples']), 'phases': sorted({s['phase'] for s in rq['samples']}),
                                            'cleanup': rq['cleanup'], 'readback': rq['containment']}

# 9h. a contained cell that FAILS at model load on the same fake server (HTTP 500 from /api/models/load; still no model, GPU or Titanos):
# the kit path records exercise fail with verdict within (peak under the cap, launch/server_start/model_load phases, cleanup), and the
# kit-written receipt enters the matrix as FAIL through ingest, never as untried: the fail signal of a measured within-bound runtime is kept.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'h'
    (root / 'titanos').mkdir(parents=True)
    (root / 'titanos' / '__init__.py').write_text('', encoding='utf-8')
    (root / 'titanos' / '__main__.py').write_text(FAKE_CHAT_SERVER.replace('LOAD_STATUS = 200', 'LOAD_STATUS = 500'), encoding='utf-8')
    entry = {'model_id': 'fake', 'display_name': 'Fake', 'task': 'chat', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {}, 'matrix_identity': {'repo': 'fake'}}
    cfg, _ = run_cells.build_config(entry, cpu_slot, 'single', None, 'auto')
    resource = dict(contained_resource(512 * MIB), floors={'commit': 512 * MIB + 1, 'gpu': 4096}, profile_inputs=GATE_IDENTITY)
    real_free, real_used = run_cells.gpu_free_mib, run_cells.gpu_used_mib
    run_cells.gpu_free_mib, run_cells.gpu_used_mib = (lambda u: 20000), (lambda u: 12000)
    try:
        rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', 'GPU-fake', 30, stop_file=str(Path(tmp) / 'h.STOP'), resource=resource)
    finally:
        run_cells.gpu_free_mib, run_cells.gpu_used_mib = real_free, real_used
    rq = rec['resource_qualification']
    phases = [s['phase'] for s in rq['samples']]
    expect(rec['state'] == 'fail' and rec['exercise']['state'] == 'fail' and 'load refused HTTP 500' in (rec['exercise'].get('error') or '') and rq['verdict'] == 'within'
           and 'request' not in phases and 'model_load' in phases and phases[0] == 'launch' and rq['cleanup'].get('launched') is True,
           f"the fake server's load refusal is a measured within-bound fail: {rec['state']} {rec.get('error')} verdict {rq['verdict']} phases {phases} cleanup {rq['cleanup']}")
    fake_gpu_slot = dict(cpu_slot, slot='win-x64-nvidia-cuda', device='GPU-fake', device_named='fake')
    receipt = run_cells.cell_receipt('kit-fail', entry, fake_gpu_slot, 'single', None, {'hostname': 'test'}, [dict(rec, dtype='auto')], 'fail', 'chat fail on the fake device: load refused', None)
    expect(ingest.validate_contained_block(receipt['resource_qualification'], receipt) == [] and ingest.validate_contained_block(receipt['resource_qualification'], receipt, 'pass'),
           f"the kit-written fail receipt satisfies the gate for a fail row and not for a pass row: {ingest.validate_contained_block(receipt['resource_qualification'], receipt)}")
    kit = Path(tmp) / 'kit'
    (kit / 'receipts').mkdir(parents=True)
    (kit / 'receipts' / 'fake.json').write_text(json.dumps(receipt, default=str), encoding='utf-8')
    row = {'model_id': 'fake', 'slot': 'win-x64-nvidia-cuda', 'mode': 'single', 'state': 'fail', 'device': 'GPU-fake', 'device_named': 'fake', 'receipt': 'receipts/fake.json', 'note': 'chat fail', 'dtype': None}
    (kit / 'RUN_RESULT.json').write_text(json.dumps({'run_id': 'kit-fail', 'kit_schema': run_cells.RECEIPT_SCHEMA, 'utc': run_cells.utc(), 'cells': [row], 'host_reserve_bytes': resource['reserve_bytes'],
                                                     'floors': {'commit': 4 << 30, 'gpu': 4096}, 'containment_available': True}), encoding='utf-8')
    raw = Path(tmp) / 'raw.json'
    raw.write_text(json.dumps({'catalog': {}, 'families': [], 'mode_cells': []}), encoding='utf-8')
    out = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'ingest.py'), '--raw', str(raw), '--evidence', 'E-000', '--result', str(kit / 'RUN_RESULT.json')],
                         capture_output=True, text=True, cwd=str(ROOT))
    data = json.loads(raw.read_text(encoding='utf-8'))
    cell = next((c for f in data['families'] for c in f['cells'] if c['model_id'] == 'fake'), {})
    expect(out.returncode == 0 and cell.get('state') == 'fail' and cell.get('verified') is None and cell.get('note') == '[single] chat fail',
           f'ingest binds the kit-written fail-at-load receipt as fail, not untried: rc {out.returncode} {cell} {out.stderr[-300:]}')
    observations['serve_contained_fail_at_load'] = {'phases': phases, 'peak_mib': (rq['measured_peak_job_commit_bytes'] or 0) >> 20, 'state': cell.get('state')}

# 10. cap semantics under a small cap: what JOB_OBJECT_LIMIT_JOB_MEMORY refuses, what it only accounts, and what it cannot see (synthetic files,
# synthetic children, observations recorded beside the assertions). The receipts above carry containment.CAP_SEMANTICS verbatim.
import importlib.util  # noqa: E402
expect(all(isinstance(containment.CAP_SEMANTICS.get(k), list) and containment.CAP_SEMANTICS[k] for k in ('enforced', 'accounted_not_refused', 'unaccounted')),
       'CAP_SEMANTICS lists the enforced classes and the two gaps')
NO_DIALOG = 'import ctypes\nctypes.windll.kernel32.SetErrorMode(0x8003)\n'  # a native fault ends the child quietly instead of raising a WER dialog


def run_capped(code, args, cap, timeout=600):
    """One synthetic child under `cap`; returns (stdout, returncode, peak job commit bytes, cleanup evidence)."""
    job = containment.ContainedJob([sys.executable, '-c', code] + list(args), cap, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = job.process.communicate(timeout=timeout)
    live = job.query()
    evidence = job.close()
    peak = max(live['peak_job_commit_bytes'], (evidence.get('final_query') or {}).get('peak_job_commit_bytes', 0))
    return out or '', job.process.returncode, peak, evidence


def clean(evidence):
    return evidence.get('active_processes_after') == 0 and evidence.get('job_handle_closed') is True


with tempfile.TemporaryDirectory() as tmp:
    big = Path(tmp) / 'shard.bin'
    with big.open('wb') as f:
        f.seek(512 * MIB - 1)
        f.write(b'\0')
    # 10a. a copy-on-write view (Python mmap ACCESS_COPY = CreateFileMapping PAGE_WRITECOPY + MapViewOfFile FILE_MAP_COPY) of the 512 MiB
    # file under a 256 MiB cap, then a write to every page. The view's charge lands in PeakJobMemoryUsed (> cap) whether the view is refused
    # at map time (OSError 1455: this host, 2026-09-25, also the owner's 5 GB shard record) or created and failed at touch; the child never
    # touches every page and exits nonzero; the reap is clean. Which of the two happened is recorded, not assumed.
    COW = NO_DIALOG + ("import mmap, sys\npath, n = sys.argv[1], 512 << 20\nwith open(path, 'rb') as f:\n    try:\n        m = mmap.mmap(f.fileno(), n, access=mmap.ACCESS_COPY)\n"
                       "    except OSError as exc:\n        print('map_error', exc, flush=True)\n        sys.exit(4)\nprint('mapped', n, flush=True)\ntouched = 0\ntry:\n"
                       "    for off in range(0, n, 4096):\n        m[off] = 1\n        touched += 1\nexcept Exception as exc:\n    print('touch_error', type(exc).__name__, str(exc)[:120], 'after', touched, flush=True)\n"
                       "    sys.exit(5)\nprint('touched_all', touched, flush=True)\n")
    out10a, rc10a, peak10a, ev10a = run_capped(COW, [str(big)], 256 * MIB)
    mapped10a = 'mapped ' in out10a
    expect(peak10a > 256 * MIB, f'a copy-on-write view over the cap is charged to PeakJobMemoryUsed whether or not it is refused: peak {peak10a >> 20} MiB vs cap 256 MiB')
    expect(rc10a != 0 and 'touched_all' not in out10a, f'the child never touches every page of a view over the cap: rc {rc10a} {out10a.strip()[-160:]}')
    expect(clean(ev10a) and not ev10a.get('forced_cleanup'), f'clean reap after the copy-on-write child ended: {ev10a}')
    observations['cow_file_map_copy_view'] = {'outcome': 'mapped_then_failed_at_touch' if mapped10a else 'refused_at_map_time', 'returncode': rc10a, 'returncode_hex': hex(rc10a & 0xFFFFFFFF),
                                              'peak_mib': peak10a >> 20, 'cap_mib': 256, 'tail': out10a.strip()[-160:]}
    # 10c. an anonymous pagefile-backed section (mmap(-1, 512 MiB)) written page by page under the same 256 MiB cap: neither counted nor refused
    ANON = NO_DIALOG + "import mmap\nn = 512 << 20\nm = mmap.mmap(-1, n)\nprint('mapped', n, flush=True)\nfor off in range(0, n, 4096):\n    m[off] = 1\nprint('written_all', flush=True)\n"
    out10c, rc10c, peak10c, ev10c = run_capped(ANON, [], 256 * MIB)
    expect(rc10c == 0 and 'written_all' in out10c, f'an anonymous section twice the cap is written in full (unaccounted, unenforced): rc {rc10c} {out10c.strip()[-100:]}')
    expect(peak10c < 128 * MIB, f'the job peak stays well under the cap while 512 MiB of section pages are written: {peak10c >> 20} MiB')
    expect(clean(ev10c), f'clean reap after the anonymous-section child: {ev10c}')
    observations['anonymous_section'] = {'returncode': rc10c, 'peak_mib': peak10c >> 20, 'cap_mib': 256, 'written_mib': 512}
# 10b. torch.UntypedStorage.from_file(shared=False): the accounted-but-not-refused class (not shown to be r3's path), run only when torch is
# importable by this interpreter (no GPU, no weights: a synthetic 2 GiB file). Under a 2560 MiB cap with torch imported (~1.6 GB of commit)
# the 2 GiB view is created without refusal, PeakJobMemoryUsed passes the cap, the process commit reading does not move, and the first write
# to a mapped page ends the process natively (0xC0000005 on this host, the exit code r3 showed) - in any case the child never touches every page.
if importlib.util.find_spec('torch') is None:
    observations['torch_from_file_cow'] = 'skipped: torch is not importable by this interpreter'
else:
    with tempfile.TemporaryDirectory() as tmp:
        big = Path(tmp) / 'shard.bin'
        with big.open('wb') as f:
            f.seek(2 * (1 << 30) - 1)
            f.write(b'\0')
        FROM_FILE = NO_DIALOG + ("import sys, psutil\npath, n = sys.argv[1], 2 << 30\nme = psutil.Process()\nprint('start', me.memory_info().pagefile, flush=True)\nimport torch\n"
                                 "print('torch_imported', me.memory_info().pagefile, flush=True)\nst = torch.UntypedStorage.from_file(path, shared=False, nbytes=n)\n"
                                 "print('from_file_mapped', me.memory_info().pagefile, flush=True)\nt = torch.empty(0, dtype=torch.uint8).set_(st)\nfor off in range(0, n, 4096):\n    t[off] = 1\n"
                                 "print('touched_all', me.memory_info().pagefile, flush=True)\n")
        out10b, rc10b, peak10b, ev10b = run_capped(FROM_FILE, [str(big)], 2560 * MIB, timeout=900)
        lines = {ln.split()[0]: ln for ln in out10b.strip().splitlines() if ln.strip()}
        if 'torch_imported' not in lines:
            observations['torch_from_file_cow'] = {'outcome': 'torch_import_failed_under_cap', 'returncode': rc10b, 'peak_mib': peak10b >> 20, 'tail': out10b.strip()[-300:]}
        else:
            commit_after_import = int(lines['torch_imported'].split()[1])
            commit_after_map = int(lines['from_file_mapped'].split()[1]) if 'from_file_mapped' in lines else None
            expect('from_file_mapped' in lines, f'torch from_file(shared=False) over the cap is created without refusal (accounted, not refused): {out10b.strip()[-300:]}')
            expect(peak10b > 2560 * MIB, f'the job peak records the mapping charge beyond the cap: {peak10b >> 20} MiB vs cap 2560 MiB')
            expect(rc10b != 0 and 'touched_all' not in lines, f'the pages are not all touchable: the child ends nonzero (native fault expected): rc {rc10b} 0x{rc10b & 0xFFFFFFFF:x}')
            expect(clean(ev10b), f'clean reap after the from_file child: {ev10b}')
            observations['torch_from_file_cow'] = {'outcome': 'mapped_then_native_fault_at_touch' if (rc10b & 0xFFFFFFFF) == 0xC0000005 else 'mapped_then_nonzero_exit',
                                                   'returncode': rc10b, 'returncode_hex': hex(rc10b & 0xFFFFFFFF), 'peak_mib': peak10b >> 20, 'cap_mib': 2560, 'map_mib': 2048,
                                                   'process_commit_after_import': commit_after_import, 'process_commit_after_map': commit_after_map,
                                                   'process_commit_moved_at_map': None if commit_after_map is None else (commit_after_map - commit_after_import) > (64 * MIB)}
assert not failures, '\n'.join(failures)
print(json.dumps({'passed': True, 'observations': observations}, default=str))
