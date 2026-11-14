"""Blocker 3 (fail closed) independent check over fake readers, no host reading, no process, no model.

wait_admission: a None required commit/GPU/physical reading is never good; three consecutive misses end the wait untried
'resource_reading_unavailable'; misses interleaved with good samples never admit (the good streak restarts after every miss);
STOP arriving during a miss streak wins at the top of the next iteration; every sample is taken synchronously (one reader call
per iteration, its own utc stamp, the sample carries exactly that iteration's value: nothing is cached across iterations).
CellGuard: a None commit reading at the forced launch check trips before anything is created.
"""
import sys
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools' / 'matrix_kit'))
import containment  # noqa: E402
import run_cells  # noqa: E402

FAILED = []


def expect(cond, msg):
    if not cond:
        FAILED.append(msg)
        print('FAIL', msg)


GIB = 1 << 30
A = 'GPU-aaaaaaaa-0000-0000-0000-000000000001'
quiet = lambda s: None  # noqa: E731
logs = []


def seq(values):
    it = iter(values)
    calls = []

    def reader(*_):
        calls.append(1)
        return next(it)
    reader.calls = calls
    return reader


# 1. freshness: one commit read per iteration, sample value == that iteration's read, distinct utc stamps, no reuse
values = [10 * GIB, 11 * GIB, 12 * GIB]
rd = seq(values)
r = run_cells.wait_admission({'commit': GIB, 'gpu': None}, None, None, logs.append, commit_reader=rd, gpu_reader=lambda u: None, sleep=quiet)
expect(r['admitted'] is True and len(rd.calls) == 3, f'three good samples = three synchronous reads: {r}')
expect([s['available_commit_bytes'] for s in r['samples']] == values, f'each sample carries its own iteration read (no cached value): {r["samples"]}')
expect(all(isinstance(s['utc'], str) and s['utc'] for s in r['samples']), 'every sample carries a utc stamp')
expect(r['samples'][0]['utc'] <= r['samples'][1]['utc'] <= r['samples'][2]['utc'], 'utc stamps are monotone per iteration')

# 2. interleaved None / good never admits: the good streak restarts after every miss; the miss streak restarts after every good reading
pattern = [50 * GIB, None, 50 * GIB, 50 * GIB, None, 50 * GIB, 50 * GIB, None, 50 * GIB, 50 * GIB, None, None, None]
rd = seq(pattern)
r = run_cells.wait_admission({'commit': GIB, 'gpu': None}, None, None, logs.append, commit_reader=rd, gpu_reader=lambda u: None, sleep=quiet)
expect(r['admitted'] is False and r['reason'] == 'resource_reading_unavailable' and len(rd.calls) == len(pattern),
       f'good samples split by misses never reach three consecutive; three trailing misses end the wait untried: {r["reason"]} after {len(rd.calls)} reads')
expect(all(s['available_commit_bytes'] is None for s in r['samples']), f'the recorded last three samples are the three misses: {r["samples"]}')

# 3. STOP during a miss streak wins at the top of the next iteration (before a third miss is counted)
with tempfile.TemporaryDirectory() as tmp:
    stop = Path(tmp) / 'STOP'
    reads = []

    def missing_then_stop():
        reads.append(1)
        if len(reads) == 2:
            stop.write_text('x', encoding='utf-8')
        return None
    r = run_cells.wait_admission({'commit': GIB, 'gpu': None}, None, str(stop), logs.append, commit_reader=missing_then_stop, gpu_reader=lambda u: None, sleep=quiet)
    expect(r['admitted'] is False and r['reason'] == 'stop' and len(reads) == 2, f'STOP checked at the top beats a pending miss streak: {r["reason"]} after {len(reads)} reads')

# 4. GPU reading required on a GPU slot with a GPU floor even when commit is fine and intermittently present
gpu = seq([20000, None, None, None])
r = run_cells.wait_admission({'commit': GIB, 'gpu': 4096}, A, None, logs.append, commit_reader=lambda: 50 * GIB, gpu_reader=gpu, sleep=quiet)
expect(r['admitted'] is False and r['reason'] == 'resource_reading_unavailable' and len(gpu.calls) == 4, f'one good GPU read then three misses: untried, never admitted: {r["reason"]}')

# 5. contained: physical reading missing on the third sample only -> the two good samples do not carry; the wait continues, never admits on stale data
contained = {'reserve_bytes': 6 * GIB, 'profile_peak_bytes': None, 'margin_bytes': 2 * GIB}
phys = seq([100 * GIB, 100 * GIB, None, None, None])
r = run_cells.wait_admission({'commit': None, 'gpu': None}, A, None, logs.append, contained=contained, commit_reader=lambda: 40 * GIB, gpu_reader=lambda u: 20000, sleep=quiet, phys_reader=phys)
expect(r['admitted'] is False and r['reason'] == 'resource_reading_unavailable' and len(phys.calls) == 5, f'contained: a missing physical reading after two good samples is not admitted: {r["reason"]}')

# 6. a reader raising is never admitted (propagates; nothing admitted, nothing launched)
def boom():
    raise OSError('reader broke')
try:
    r = run_cells.wait_admission({'commit': GIB, 'gpu': None}, None, None, logs.append, commit_reader=boom, gpu_reader=lambda u: None, sleep=quiet)
    expect(r['admitted'] is not True, f'a raising commit reader must never admit: {r}')
except OSError:
    pass

# 7. the guard: None commit at the forced launch check trips 'resource_reading_unavailable' before any launch
g = containment.CellGuard(None, None, None, None, None, None, None, commit_reader=lambda: None, gpu_free_reader=lambda u: None, interval_s=0)
try:
    g.check('launch', force=True)
    expect(False, 'a None commit reading at launch must trip')
except containment.GuardTrip as exc:
    expect(exc.kind == 'budget_violated' and 'resource_reading_unavailable' in str(exc) and g.readings_complete is False, f'launch guard trips on None commit: {exc}')
g2 = containment.CellGuard(None, None, None, None, A, 4096, None, commit_reader=lambda: 100 * GIB, gpu_free_reader=lambda u: None, interval_s=0)
try:
    g2.check('launch', force=True)
    expect(False, 'a None GPU reading on a GPU slot at launch must trip')
except containment.GuardTrip as exc:
    expect('GPU free memory' in str(exc), f'launch guard trips on None GPU reading: {exc}')
# a forced check always takes a fresh sample (no reuse of the throttled last sample)
calls = []
g3 = containment.CellGuard(None, None, None, None, None, None, None, commit_reader=lambda: calls.append(1) or (100 * GIB), gpu_free_reader=lambda u: None, interval_s=60)
g3.check('launch', force=True)
g3.check('model_load', force=True)
expect(len(calls) == 2 and len(g3.samples) == 2 and g3.samples[0] is not g3.samples[1], f'forced checks each read afresh: {len(calls)} reads')

# 8. the error text the caller records for the untried cell names the fail-closed reason
expect(run_cells.ADMISSION_ERRORS['resource_reading_unavailable'].startswith('resource_reading_unavailable:'), 'untried error text')

print('FAILED' if FAILED else 'OK', len(FAILED))
sys.exit(1 if FAILED else 0)
