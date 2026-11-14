"""Bind matrix-kit run results into the execution-matrix raw input consumed by tools/build_execution_matrix.py.

    python tools/matrix_kit/ingest.py --raw <execution_matrix_raw.json> --result <RUN_RESULT.json> [--result ...] --evidence E-0NN [--verdicts verdicts.json]
                                      [--legacy-run <run_id>] [--legacy-manifest legacy_results.json]

For every kit cell with a state of pass, fail or na:
  - the `single` mode result becomes the (model, slot) cell in a reader family named `matrix-kit:<run_id>` (when several result
    folders share one run id, every folder after the first is qualified as `matrix-kit:<run_id>#<folder>` so none replaces another);
  - every mode result (single, replicas, model_sharding, tensor_parallel) becomes a `mode_cells` entry.
Kit cells left untried (artifact absent, no execution path, blocked before admission) are not written: the matrix keeps its own
untried cell and the reason stays in the receipt. The `receipt` field written for readers is a lab-neutral display reference,
`<lab-records>/<parent folder>/<run folder>/<receipt>` (tools/lab_paths.records_reference); the receipt that decides eligibility is
resolved against the run folder itself (read_receipt below) and no host location enters the RAW input or the published matrix.

A pass enters as `verified: true` only when the verdicts file upholds it: {"<model_id>|<slot>|<mode>": {"upheld": bool,
"corrected_state": "...", "reasoning": "..."}}. Without an upholding verdict the builder downgrades the pass to partial, which
is the intended fail-safe. The evidence id must already be cited in docs/CAPABILITY_MATRIX.md (the builder drops cells whose
evidence the matrix does not cite).

When one run executed a cell on two adapters that share a slot (two AMD dGPUs, two Intel Arcs; the kit tags such receipts
`<slot>@<device>`), the cells are merged per (model, slot, mode): the best outcome (an upheld pass over a refuted pass over a
fail over a refusal) becomes the cell, the other adapter's outcome is kept in the note, and a verdict may be keyed per adapter
as "<model_id>|<slot>@<device tag>|<mode>" (the plain key applies to every adapter otherwise).

Schema generations and the legacy boundary. A RUN_RESULT is CURRENT when it carries `kit_schema` >= 2 (run_cells.RECEIPT_SCHEMA;
every cell receipt of such a run carries `receipt_schema` 2 and its top-level `resource_qualification` is the complete measured
block: cap, readback, peaks, samples, identity, cleanup). A RUN_RESULT without the marker is LEGACY only when ALL of: its `utc`
parses and lies before LEGACY_RESULT_BOUNDARY_UTC, AND the sha256 of the RUN_RESULT.json file is listed in the pinned legacy
population (legacy_results.json beside this file: the 78 results bound before the boundary, hashed read-only) - or --legacy-run
names its run id explicitly (a human decision, recorded in the family). Its cells then enter as the pre-boundary ingest bound them
for old full-artifact receipts (nothing relabelled; the family records kit_schema 1), except that a contained_resident block is never
legacy evidence (see resource_gate). A RUN_RESULT without the marker whose utc is missing or at/after the boundary, or whose hash is not
pinned, is refused with an error before anything is written (fail closed): a self-reported earlier utc alone never opens the legacy path.

Resource qualification gate, current runs. For every pass or fail cell ingest reads the detailed receipt the row names (folder /
cell['receipt'], which must resolve inside the run folder); a receipt that is missing, outside the folder, unparseable, without
`receipt_schema` >= 2 or without a `resource_qualification` dict makes the cell untried (`receipt_missing_or_unparseable: ...` /
`receipt_schema_missing: ...`); the RUN_RESULT row's resource_kind/resource_verdict/resource_identity are display fields and never
decide eligibility. The receipt must be BOUND to the row (binding_problems): its run_id equals the RUN_RESULT run_id and its model_id,
slot (slot['slot']), mode and device equal the row's, so no cell can enter on another cell's receipt. A `contained_resident` block must
pass validate_contained_block: typed cap and readback verified before resume; the reserve equal to the RUN_RESULT host_reserve_bytes and
floors.commit == cap + reserve (the kit's contained admission rule); a measured peak under the cap that is at least every sample peak and
the final owned-job query peak (itself under the cap); a sensor sequence of typed samples in the kit's phase order with host commit,
physical and GPU readings that never dipped under the reserve and job peaks that never reached the cap - for a pass covering the
model_load and request phases, for a fail the recorded prefix up to the failing step with the exercise error; the placement identity
with 64-hex sha256 keys whose dtype equals the row dtype and whose copies match the mode; cleanup evidence with the job handle closed
and no live process; exercise state matching the row. An `artifact_floor` block must carry verdict `not_contained` and the kit-written
admission plan reason consistent with the receipt's `spec_placement` and device (a resident spec on a GPU slot may run under the
artifact floor only when the host could not contain it: reason containment_unavailable and RUN_RESULT containment_available false).
Any problem makes the cell untried `resource_qualification_evidence_incomplete: <problems>; functional output retained separately
(exercise <state>)`; verdict `exceeded` or `unknown` is untried as before with the exercise state in the note. A valid WAV under an
unmeasured, exceeded or undocumented bound is output evidence, never resource qualification. A verifier's `upheld` flag counts only
when it is exactly boolean true; any other value leaves the pass unverified.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/: lab_paths (lab-neutral display references)
from lab_paths import records_reference  # noqa: E402

RANK = {'pass': 3, 'fail': 2, 'na': 1, 'untried': 0}
ACCEPTED_VERDICTS = {'within', 'not_contained'}
IDENTITY_FIELDS = ('receipt_sha256', 'adapter_sha256', 'placement', 'dtype', 'copies')  # the profile-key inputs run_cells writes into the block
HEX64 = re.compile(r'[0-9a-f]{64}')  # a real sha256 hex digest: lowercase, exactly 64 characters
CURRENT_KIT_SCHEMA = 2  # run_cells.RECEIPT_SCHEMA: RUN_RESULT 'kit_schema' and receipt 'receipt_schema' at or above this are current
LEGACY_RESULT_BOUNDARY_UTC = '2026-09-25T19:00:00+00:00'  # results without a kit_schema marker are legacy only when their utc is earlier AND their hash is pinned
LEGACY_RESULTS_MANIFEST = Path(__file__).with_name('legacy_results.json')  # the pinned pre-boundary population (sha256 of each RUN_RESULT.json)
REQUIRED_FLAGS, BREAKAWAY_FLAGS = 0x2200, 0x1800  # JOB_OBJECT_LIMIT_JOB_MEMORY | KILL_ON_JOB_CLOSE must be set; the breakaway flags must be clear
PHASE_ORDER = ('launch', 'server_start', 'model_load', 'request', 'unload')  # the guard phases in the order run_cells.serve_and_exercise reaches them
REQUIRED_PHASES = ('model_load', 'request')  # a pass must have been measured through both; a fail carries the prefix up to its failing step
# the artifact_floor admission-plan reasons run_cells.contained_admission_plan writes; an artifact_floor block may carry no other reason
ARTIFACT_REASON_NO_GPU = 'resident placement on a slot without a GPU: no contained admission, full artifact floor'
ARTIFACT_REASON_UNAVAILABLE = 'containment_unavailable: this host cannot cap the owned process tree (no Windows Job Object); the full artifact floor applies unchanged'


def device_tag(cell):
    return re.sub(r'[^A-Za-z0-9.-]+', '_', str(cell.get('device') or ''))


def normalize_key(key):
    """'<model>|<slot>[@<device label>]|<mode>' -> the device label reduced to the receipt's tag: the text before any ' (' label
    (verifiers write 'GPU.0 (Intel Arc B580, OpenVINO GPU)' for the receipt device 'GPU.0'), sanitised like device_tag."""
    parts = key.split('|')
    if len(parts) != 3 or '@' not in parts[1]:
        return key
    slot, _, device = parts[1].partition('@')
    device = device.split(' (')[0].strip()
    return f"{parts[0]}|{slot}@{re.sub(r'[^A-Za-z0-9.-]+', '_', device)}|{parts[2]}"


def parse_utc(text):
    """An aware datetime for an ISO-8601 timestamp (a trailing 'Z' accepted; a naive value is taken as UTC), or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        value = datetime.fromisoformat(text.strip().replace('Z', '+00:00'))
    except ValueError:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def load_legacy_manifest(path=LEGACY_RESULTS_MANIFEST):
    """The pinned sha256 digests of the pre-boundary RUN_RESULT.json files, as a frozenset. A missing or unreadable manifest pins
    nothing (fail closed: every unmarked result then needs --legacy-run)."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return frozenset()
    results = data.get('results') if isinstance(data, dict) else None
    return frozenset(r['sha256'] for r in (results or []) if isinstance(r, dict) and isinstance(r.get('sha256'), str) and HEX64.fullmatch(r['sha256']))


def result_generation(result, legacy_runs=(), digest=None, pinned=frozenset()):
    """('current', None) for a RUN_RESULT with kit_schema >= CURRENT_KIT_SCHEMA; ('legacy', basis) for one without the marker whose run
    id --legacy-run names (basis 'legacy_run_flag') or whose utc is before the boundary AND whose file sha256 `digest` is in the pinned
    legacy population (basis 'pinned_sha256'); ('refused', reason) otherwise (fail closed: a post-boundary result without the marker,
    or a back-dated one whose hash nobody pinned, is neither legacy nor current until a human names it)."""
    schema = result.get('kit_schema')
    if type(schema) is int and schema >= CURRENT_KIT_SCHEMA:
        return 'current', None
    run_id = result.get('run_id')
    if run_id in set(legacy_runs or ()):
        return 'legacy', 'legacy_run_flag'
    utc = parse_utc(result.get('utc'))
    boundary = parse_utc(LEGACY_RESULT_BOUNDARY_UTC)
    if utc is None:
        why = f"its utc {result.get('utc')!r} does not parse"
    elif utc >= boundary:
        why = f"its utc {result.get('utc')} is not before the legacy boundary {LEGACY_RESULT_BOUNDARY_UTC}"
    elif isinstance(digest, str) and digest in pinned:
        return 'legacy', 'pinned_sha256'
    else:
        why = (f"its utc {result.get('utc')} is before the legacy boundary but its file sha256 {digest} is not in the pinned legacy population "
               f"({len(pinned)} results in {LEGACY_RESULTS_MANIFEST.name}; a self-reported utc alone never opens the legacy path)")
    return 'refused', (f"RUN_RESULT {run_id!r} carries no kit_schema >= {CURRENT_KIT_SCHEMA} and {why}: not a legacy run, not a current run; "
                       f"refused (fail closed). Re-run it with the current kit, or name it explicitly with --legacy-run {run_id}")


# --- legacy generation (pinned results before the boundary): the pre-boundary rules for full-artifact receipts -----------------------

def resource_qualification(folder, cell):
    """LEGACY runs only. The receipt's resource_qualification block for a kit cell, or None when the receipt (older kit) carries none;
    the RUN_RESULT row's resource_verdict/resource_kind stand in when the receipt cannot be read. Current runs never use this."""
    receipt = cell.get('receipt')
    if receipt:
        path = Path(folder) / receipt
        if path.is_file():
            try:
                block = json.loads(path.read_text(encoding='utf-8')).get('resource_qualification')
            except (OSError, ValueError):
                block = None
            if isinstance(block, dict):
                return block
    if cell.get('resource_verdict') is not None:
        return {'verdict': cell.get('resource_verdict'), 'kind': cell.get('resource_kind'), 'profile_key': cell.get('resource_profile_key'),
                **(cell.get('resource_identity') or {})}
    return None


def missing_identity(block):
    """The placement-identity fields a contained_resident block lacks: a profile_key that is a real sha256 hex digest (64 lowercase hex
    characters; run_cells writes hashlib.sha256().hexdigest()), the music-runtime receipt and adapter shas, placement 'resident', a dtype
    and an integer copies >= 1 (booleans are not integers here). [] when the identity is complete."""
    missing = []
    key = block.get('profile_key')
    if not (isinstance(key, str) and HEX64.fullmatch(key)):
        missing.append('profile_key')
    for field in IDENTITY_FIELDS:
        value = block.get(field)
        if field == 'placement':
            ok = value == 'resident'
        elif field == 'copies':
            ok = type(value) is int and value >= 1
        else:
            ok = isinstance(value, str) and bool(value.strip())
        if not ok:
            missing.append(field)
    return missing


def resource_gate(block):
    """LEGACY runs only. None when the block accepts the cell's own state (no block, or verdict not_contained on an artifact_floor
    block); otherwise the untried note. The legacy boundary covers old full-artifact receipts only: a contained_resident block is never
    legacy evidence (the pre-boundary ingest accepted verdict within on the placement identity alone, without cap, readback, peak,
    samples or cleanup), so such a cell enters untried `resource_qualification_legacy_contained` whatever its verdict says; a contained
    bound enters the matrix only through a current-kit detailed receipt. exceeded/unknown keep their verdict notes."""
    if not block:
        return None
    verdict = block.get('verdict')
    if verdict not in ACCEPTED_VERDICTS:
        return verdict_note(block)
    if block.get('kind') == 'contained_resident':
        missing = missing_identity(block)
        return ('resource_qualification_legacy_contained: not accepted into the matrix (contained_resident; the legacy boundary covers artifact_floor '
                'receipts only: a contained bound enters through a current-kit detailed receipt with the measured cap, readback, peak, samples and cleanup'
                + (f"; identity missing {', '.join(missing)}" if missing else '') + ')')
    return None


def verdict_note(block):
    detail = block.get('reason') or ''
    verdict = block.get('verdict')
    if verdict == 'exceeded':
        detail = f"contained job commit peak {block.get('measured_peak_job_commit_bytes')} vs limit {block.get('job_commit_limit_bytes')}" + (f'; {detail}' if detail else '')
    return f"resource_qualification_{verdict}: not accepted into the matrix ({block.get('kind') or 'kind unknown'}; {detail or 'no measured bound'})"


# --- current generation (kit_schema >= 2): the detailed receipt is the only eligibility evidence ------------------------------------

def _int(value, minimum=None):
    return type(value) is int and (minimum is None or value >= minimum)


def read_receipt(folder, cell):
    """(receipt dict, None) for the detailed receipt a current row names, or (None, problem). The named path must resolve inside the
    run folder: a row cannot point at another run's file or traverse out of its folder."""
    name = cell.get('receipt')
    if not isinstance(name, str) or not name:
        return None, 'the RUN_RESULT row names no receipt file'
    base = Path(folder).resolve()
    try:
        path = (base / name).resolve()
    except (OSError, RuntimeError) as exc:
        return None, f'receipt path {name!r} does not resolve: {type(exc).__name__}: {exc}'
    if path == base or not path.is_relative_to(base):
        return None, f'receipt path {name!r} resolves outside the run folder {base.as_posix()}'
    if not path.is_file():
        return None, f'receipt file not found: {path.as_posix()}'
    try:
        body = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return None, f'receipt {name} unreadable or not JSON: {type(exc).__name__}: {exc}'
    if not isinstance(body, dict):
        return None, f'receipt {name} is not a JSON object'
    return body, None


def binding_problems(receipt, cell, run_id):
    """Every way the receipt fails to be the row's own: run_cells.cell_receipt writes run_id, model_id, slot (the slot dict), mode into
    each receipt; all must equal the RUN_RESULT run id and the row, and the slot's device must equal the row's device when the row
    names one. A complete receipt of another cell (or another run) qualifies nothing."""
    problems = []
    if receipt.get('run_id') != run_id:
        problems.append(f"receipt run_id {receipt.get('run_id')!r} differs from the RUN_RESULT run_id {run_id!r}")
    if receipt.get('model_id') != cell.get('model_id'):
        problems.append(f"receipt model_id {receipt.get('model_id')!r} differs from the row model_id {cell.get('model_id')!r}")
    slot = receipt.get('slot')
    slot_id = slot.get('slot') if isinstance(slot, dict) else None
    if slot_id != cell.get('slot'):
        problems.append(f"receipt slot {slot_id!r} differs from the row slot {cell.get('slot')!r}")
    if receipt.get('mode') != cell.get('mode'):
        problems.append(f"receipt mode {receipt.get('mode')!r} differs from the row mode {cell.get('mode')!r}")
    if cell.get('device') is not None:
        device = slot.get('device') if isinstance(slot, dict) else None
        if device != cell.get('device'):
            problems.append(f"receipt device {device!r} differs from the row device {cell.get('device')!r}")
    return problems


def receipt_device(receipt, cell=None):
    slot = receipt.get('slot')
    if isinstance(slot, dict) and slot.get('device') is not None:
        return slot.get('device')
    return (cell or {}).get('device')


def exercise_problems(receipt, row_state):
    """The receipt's exercise block must exist, be a pass or fail, agree with the row (and the receipt's own) state, and carry the
    error text of a fail (the exercise, not the resource block, says what failed)."""
    problems = []
    exercise = receipt.get('exercise')
    if not isinstance(exercise, dict):
        problems.append('exercise block missing')
    else:
        state = exercise.get('state')
        if state not in ('pass', 'fail'):
            problems.append(f'exercise state {state!r} is not pass or fail')
        elif state != row_state:
            problems.append(f'exercise state {state!r} differs from the row state {row_state!r}')
        if state == 'fail' and row_state == 'fail' and not (isinstance(exercise.get('error'), str) and exercise['error'].strip()):
            problems.append('exercise fail carries no error text')
    if receipt.get('state') != row_state:
        problems.append(f"receipt state {receipt.get('state')!r} differs from the row state {row_state!r}")
    return problems


def validate_contained_block(block, receipt, row_state=None, cell=None, result=None):
    """Every problem with a contained_resident resource_qualification block (the complete measured block run_cells writes into a
    receipt_schema 2 receipt), as strings; [] when the block qualifies a within-bound cell. Pure: reads only the dicts given.
    row_state defaults to the receipt's own state. With `cell` (the RUN_RESULT row) the identity is compared with the row (dtype, copies
    for the mode); with `result` (the RUN_RESULT) the reserve is compared with the run's host_reserve_bytes."""
    problems = []
    if not isinstance(block, dict):
        return ['resource_qualification is not a dict']
    row_state = receipt.get('state') if row_state is None else row_state
    if block.get('kind') != 'contained_resident':
        problems.append(f"kind {block.get('kind')!r} is not contained_resident")
    if block.get('bound') != 'job_commit':
        problems.append(f"bound {block.get('bound')!r} is not job_commit")
    if block.get('verdict') != 'within':
        problems.append(f"verdict {block.get('verdict')!r} is not within")
    limit, reserve = block.get('job_commit_limit_bytes'), block.get('host_reserve_bytes')
    if not _int(limit, 1):
        problems.append(f'job_commit_limit_bytes {limit!r} is not a positive int')
    if not _int(reserve, 1):
        problems.append(f'host_reserve_bytes {reserve!r} is not a positive int')
    if result is not None:
        run_reserve = result.get('host_reserve_bytes')
        if not _int(run_reserve, 1):
            problems.append(f'RUN_RESULT host_reserve_bytes {run_reserve!r} is not a positive int')
        elif _int(reserve, 1) and reserve != run_reserve:
            problems.append(f'host_reserve_bytes {reserve} differs from the RUN_RESULT host_reserve_bytes {run_reserve}')
    floors = block.get('floors')
    if not (isinstance(floors, dict) and _int(floors.get('commit')) and _int(floors.get('gpu'))):
        problems.append(f'floors {floors!r} lack int commit and gpu')
    elif _int(limit, 1) and _int(reserve, 1) and floors['commit'] != limit + reserve:
        problems.append(f"floors commit {floors['commit']} is not the cap {limit} + reserve {reserve} (the kit's contained admission floor)")
    # the cap readback taken while the child was still suspended
    rb = block.get('containment')
    if not isinstance(rb, dict):
        problems.append('containment readback missing')
    else:
        if not (_int(rb.get('requested_limit_bytes')) and rb.get('requested_limit_bytes') == rb.get('job_memory_limit_bytes') == limit):
            problems.append(f"readback limits {rb.get('requested_limit_bytes')!r}/{rb.get('job_memory_limit_bytes')!r} do not equal the cap {limit!r}")
        flags = rb.get('limit_flags')
        if not _int(flags) or flags & REQUIRED_FLAGS != REQUIRED_FLAGS or flags & BREAKAWAY_FLAGS:
            problems.append(f'readback limit_flags {flags!r} do not have 0x{REQUIRED_FLAGS:x} set with 0x{BREAKAWAY_FLAGS:x} clear')
        if rb.get('child_in_job') is not True:
            problems.append('readback child_in_job is not true')
        if rb.get('verified_before_resume') is not True:
            problems.append('readback verified_before_resume is not true')
    peak = block.get('measured_peak_job_commit_bytes')
    if not _int(peak, 1):
        problems.append(f'measured_peak_job_commit_bytes {peak!r} is not a positive int')
    elif _int(limit, 1) and peak >= limit:
        problems.append(f'measured peak {peak} is not under the cap {limit}')
    if not _int(block.get('measured_peak_process_commit_bytes'), 0):
        problems.append(f"measured_peak_process_commit_bytes {block.get('measured_peak_process_commit_bytes')!r} is not an int >= 0")
    if block.get('limit_hit') is not False:
        problems.append(f"limit_hit {block.get('limit_hit')!r} is not false")
    if block.get('readings_complete') is not True:
        problems.append(f"readings_complete {block.get('readings_complete')!r} is not true")
    if block.get('trip') is not None:
        problems.append(f"trip {block.get('trip')!r} is not null")
    samples = block.get('samples')
    problems += sample_problems(samples, limit, reserve, row_state)
    # the declared measured peak is the maximum of what the guard and the final owned-job query saw (run_cells), never below either
    sample_peaks = [s['peak_job_commit_bytes'] for s in (samples if isinstance(samples, list) else []) if isinstance(s, dict) and _int(s.get('peak_job_commit_bytes'), 0)]
    cleanup = block.get('cleanup')
    final = cleanup.get('final_query') if isinstance(cleanup, dict) else None
    final_peak = final.get('peak_job_commit_bytes') if isinstance(final, dict) else None
    if _int(peak, 1) and sample_peaks and peak < max(sample_peaks):
        problems.append(f'measured peak {peak} is below a sample peak {max(sample_peaks)}')
    if _int(peak, 1) and _int(final_peak, 0) and peak < final_peak:
        problems.append(f'measured peak {peak} is below the final owned-job query peak {final_peak}')
    if _int(final_peak, 0) and _int(limit, 1) and final_peak >= limit:
        problems.append(f'final owned-job query peak {final_peak} is not under the cap {limit}')
    missing = missing_identity(block)
    if missing:
        problems.append('placement identity incomplete: missing ' + ', '.join(missing))
    for field in ('receipt_sha256', 'adapter_sha256'):
        value = block.get(field)
        if isinstance(value, str) and value.strip() and not HEX64.fullmatch(value):
            problems.append(f'{field} is not a 64-character lowercase sha256 hex')
    if cell is not None:
        # the bound was measured for exactly this placement: the row's dtype (a pass row always names the dtype that passed) and the
        # number of weight copies the mode holds (run_cells: 2 for replicas, else 1)
        expected_copies = 2 if cell.get('mode') == 'replicas' else 1
        if block.get('copies') != expected_copies:
            problems.append(f"copies {block.get('copies')!r} is not {expected_copies} for mode {cell.get('mode')!r}")
        row_dtype = cell.get('dtype')
        if (row_dtype is not None or row_state == 'pass') and block.get('dtype') != row_dtype:
            problems.append(f"dtype {block.get('dtype')!r} differs from the row dtype {row_dtype!r}")
    problems += cleanup_problems(cleanup)
    problems += exercise_problems(receipt, row_state)
    return problems


def sample_problems(samples, limit, reserve, row_state='pass'):
    """The guard's sensor sequence: a non-empty list of typed samples (utc, phase, host commit, physical, GPU free/used, job peak), utc
    non-decreasing, phases from PHASE_ORDER in that order, never under the reserve and never at or over the cap (the guard trips at the
    cap). The 'launch' sample is the kit's check immediately before the tree is created (STOP or a missing host/GPU reading there creates
    no process), so it carries the host and GPU readings but no job peak; every sample of a later phase is taken while the owned tree
    exists and must carry its peak. A pass must cover model_load and request; a fail carries the prefix up to its failing step."""
    if not isinstance(samples, list) or not samples:
        return ['samples missing or empty']
    problems, last, phases, last_phase = [], None, set(), -1
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            problems.append(f'sample {i} is not a dict')
            continue
        when = parse_utc(s.get('utc'))
        if when is None:
            problems.append(f"sample {i} utc {s.get('utc')!r} does not parse")
        elif last is not None and when < last:
            problems.append(f"sample {i} utc {s.get('utc')} is earlier than the previous sample")
        last = when if when is not None else last
        phase = s.get('phase')
        if not isinstance(phase, str) or not phase:
            problems.append(f'sample {i} has no phase')
        elif phase not in PHASE_ORDER:
            problems.append(f"sample {i} phase {phase!r} is not one of {', '.join(PHASE_ORDER)}")
        else:
            phases.add(phase)
            index = PHASE_ORDER.index(phase)
            if index < last_phase:
                problems.append(f'sample {i} phase {phase} comes after {PHASE_ORDER[last_phase]} (out of the kit order)')
            last_phase = max(last_phase, index)
        fields = ('available_commit_bytes', 'available_physical_bytes', 'gpu_free_mib', 'gpu_used_mib') + (() if phase == 'launch' else ('peak_job_commit_bytes',))
        for field in fields:
            if not _int(s.get(field), 0):
                problems.append(f'sample {i} {field} {s.get(field)!r} is not an int >= 0')
        if _int(s.get('available_commit_bytes')) and _int(reserve) and s['available_commit_bytes'] < reserve:
            problems.append(f"sample {i} available_commit_bytes {s['available_commit_bytes']} is under the reserve {reserve}")
        if _int(s.get('available_physical_bytes')) and _int(reserve) and s['available_physical_bytes'] < reserve:
            problems.append(f"sample {i} available_physical_bytes {s['available_physical_bytes']} is under the reserve {reserve}")
        if _int(s.get('peak_job_commit_bytes')) and _int(limit) and s['peak_job_commit_bytes'] >= limit:
            problems.append(f"sample {i} peak_job_commit_bytes {s['peak_job_commit_bytes']} reached the cap {limit}")
    if 'launch' not in phases:
        problems.append('samples lack the launch phase')
    if row_state != 'fail':
        absent = [p for p in REQUIRED_PHASES if p not in phases]
        if absent:
            problems.append('samples lack the phases ' + ', '.join(absent))
    return problems


def cleanup_problems(cleanup):
    if not isinstance(cleanup, dict):
        return ['cleanup evidence missing']
    problems = []
    if cleanup.get('launched') is not True:
        problems.append('cleanup launched is not true')
    if cleanup.get('active_processes_after') != 0 or type(cleanup.get('active_processes_after')) is not int:
        problems.append(f"cleanup active_processes_after {cleanup.get('active_processes_after')!r} is not 0")
    if cleanup.get('job_handle_closed') is not True:
        problems.append('cleanup job_handle_closed is not true')
    if type(cleanup.get('forced_cleanup')) is not bool:
        problems.append(f"cleanup forced_cleanup {cleanup.get('forced_cleanup')!r} is not a bool")
    final = cleanup.get('final_query')
    if not (isinstance(final, dict) and _int(final.get('peak_job_commit_bytes'), 0)):
        problems.append('cleanup final_query lacks an int peak_job_commit_bytes')
    if 'final_query_error' in cleanup:
        problems.append(f"cleanup final_query_error present: {cleanup.get('final_query_error')!r}")
    return problems


def artifact_floor_reason(placement, gpu):
    """The admission-plan reason run_cells.contained_admission_plan writes when a cell keeps the full artifact floor: the spec placement
    is not 'resident' (auto/None never lowers the floor), a resident spec sits on a slot without a GPU, or a resident spec on a GPU slot
    found a host that cannot contain the owned tree. Mirrored by run_cells (tests hold the two equal)."""
    if placement != 'resident':
        return f'placement {placement!r} is not resident; auto/None never lowers the floor (auto may fall back to CPU offload)'
    if not gpu:
        return ARTIFACT_REASON_NO_GPU
    return ARTIFACT_REASON_UNAVAILABLE


def validate_artifact_block(block, receipt, row_state=None, cell=None, result=None):
    """The typed check of an artifact_floor block: kind artifact_floor, verdict not_contained, the kit-written admission plan reason that
    matches the receipt's spec_placement and device (so a receipt cannot relabel a contained cell as artifact_floor: a resident spec on
    a GPU slot keeps the artifact floor only when the host could not contain it, RUN_RESULT containment_available false), floors and
    cleanup present with the tree launched, exercise agreeing with the row."""
    if not isinstance(block, dict):
        return ['resource_qualification is not a dict']
    row_state = receipt.get('state') if row_state is None else row_state
    problems = []
    if block.get('kind') != 'artifact_floor':
        problems.append(f"kind {block.get('kind')!r} is not artifact_floor")
    if block.get('verdict') != 'not_contained':
        problems.append(f"verdict {block.get('verdict')!r} is not not_contained")
    if 'spec_placement' not in receipt:
        problems.append('receipt carries no spec_placement (the manifest placement the admission plan read)')
    else:
        placement = receipt.get('spec_placement')
        device = receipt_device(receipt, cell)
        gpu = isinstance(device, str) and device.startswith('GPU-')
        expected = artifact_floor_reason(placement, gpu)
        if block.get('reason') != expected:
            problems.append(f"reason {str(block.get('reason'))[:160]!r} is not the kit's artifact_floor plan reason for placement {placement!r} on device {device!r} ({expected[:60]}...)")
        if placement == 'resident' and gpu and result is not None and result.get('containment_available') is not False:
            problems.append(f"a resident spec on a GPU slot ran under the artifact floor although RUN_RESULT containment_available is {result.get('containment_available')!r}, not false")
    if not isinstance(block.get('floors'), dict):
        problems.append(f"floors {block.get('floors')!r} missing")
    cleanup = block.get('cleanup')
    if not (isinstance(cleanup, dict) and cleanup.get('launched') is True):
        problems.append(f'cleanup {cleanup!r} lacks launched true')
    return problems + exercise_problems(receipt, row_state)


def current_gate(folder, cell, result):
    """CURRENT runs: None when the detailed receipt, bound to this row of this RUN_RESULT, qualifies the cell's own pass/fail state;
    otherwise the untried note. There is no fallback to RUN_RESULT row fields."""
    receipt, problem = read_receipt(folder, cell)
    if receipt is None:
        return f'receipt_missing_or_unparseable: not accepted into the matrix ({problem})'
    schema = receipt.get('receipt_schema')
    if not (type(schema) is int and schema >= CURRENT_KIT_SCHEMA):
        return f"receipt_schema_missing: not accepted into the matrix (receipt {cell.get('receipt')} carries receipt_schema {schema!r}, need >= {CURRENT_KIT_SCHEMA})"
    block = receipt.get('resource_qualification')
    if not isinstance(block, dict):
        return f"receipt_schema_missing: not accepted into the matrix (receipt {cell.get('receipt')} has no resource_qualification block)"
    verdict, kind = block.get('verdict'), block.get('kind')
    if verdict in ('exceeded', 'unknown'):
        return verdict_note(block)
    problems = binding_problems(receipt, cell, result.get('run_id'))
    # the kind the kit declared decides which typed check applies; the verdict is then checked against that kind
    if kind == 'contained_resident':
        problems += validate_contained_block(block, receipt, cell['state'], cell, result)
    elif kind == 'artifact_floor':
        problems += validate_artifact_block(block, receipt, cell['state'], cell, result)
    else:
        problems.append(f'kind {kind!r} is not contained_resident or artifact_floor (verdict {verdict!r})')
    if problems:
        return ('resource_qualification_evidence_incomplete: not accepted into the matrix (' + '; '.join(problems)[:1200]
                + f"); functional output retained separately (exercise {cell['state']})")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, type=Path)
    ap.add_argument('--result', required=True, action='append', type=Path)
    ap.add_argument('--evidence', required=True)
    ap.add_argument('--verdicts', type=Path)
    ap.add_argument('--legacy-run', action='append', default=[], help=f'run id of a RUN_RESULT without a kit_schema marker whose utc is missing or at/after {LEGACY_RESULT_BOUNDARY_UTC}, or whose file hash the legacy manifest does not pin, to be bound under the legacy rules anyway (an explicit human decision, recorded in the family; otherwise such a result is refused)')
    ap.add_argument('--legacy-manifest', type=Path, default=LEGACY_RESULTS_MANIFEST, help='the pinned legacy population: sha256 of every pre-boundary RUN_RESULT.json (default legacy_results.json beside ingest.py)')
    ap.add_argument('--skip', action='append', default=[], help="'<run_id>:<model_id>[@<slot>][#<mode>],...' cells (or one mode of a cell) of that run to leave out; when several result folders share one run id, folders after the first are addressed as '<run_id>#<folder>' (superseded by a later rerun, or filed under a slot the kit later corrected; the receipts stay on disk)")
    a = ap.parse_args()
    skip = {}  # run_id -> set of model_id or 'model_id@slot'
    for item in a.skip:
        run, models = item.split(':', 1)
        skip.setdefault(run, set()).update(m for m in models.split(',') if m)

    used_skips = set()

    def skipped(run_id, c):
        s = skip.get(run_id, set())
        for key in (c['model_id'], f"{c['model_id']}@{c['slot']}", f"{c['model_id']}#{c['mode']}", f"{c['model_id']}@{c['slot']}#{c['mode']}"):
            if key in s:
                used_skips.add((run_id, key))
        m, sl, mode = c['model_id'], c['slot'], c['mode']
        # '<model>', '<model>@<slot>', or a single mode: '<model>#<mode>', '<model>@<slot>#<mode>' (an environment fail of one mode
        # that a later rerun or the verifier's classification supersedes, while the other modes of the same cell stay bound)
        return m in s or f"{m}@{sl}" in s or f"{m}#{mode}" in s or f"{m}@{sl}#{mode}" in s
    # every result is classified before anything is read further or written: one refused result aborts the whole ingest unchanged
    pinned = load_legacy_manifest(a.legacy_manifest)
    results = []
    for path in a.result:
        data = path.read_bytes()
        result = json.loads(data.decode('utf-8'))
        generation, basis = result_generation(result, a.legacy_run, hashlib.sha256(data).hexdigest(), pinned)
        if generation == 'refused':
            raise SystemExit(f'ingest refused ({path}): {basis}')
        results.append((path, result, generation, basis))
    raw = json.loads(a.raw.read_text(encoding='utf-8'))
    verdicts = json.loads(a.verdicts.read_text(encoding='utf-8')) if a.verdicts else {}
    verdicts = {normalize_key(k): v for k, v in verdicts.items()} | {k: v for k, v in verdicts.items() if normalize_key(k) != k}
    raw.setdefault('mode_cells', [])
    existing_mode = {(c['model_id'], c['slot'], c['mode'], c.get('source')) for c in raw['mode_cells']}
    added = {'cells': 0, 'mode_cells': 0, 'verdicts': 0, 'skipped_untried': 0, 'merged_shared_slot': 0, 'legacy_runs': [], 'current_runs': []}

    def verdict_for(c):
        tag = device_tag(c)
        if tag:
            v = verdicts.get(f"{c['model_id']}|{c['slot']}@{tag}|{c['mode']}")
            if v is not None:
                return v
        return verdicts.get(f"{c['model_id']}|{c['slot']}|{c['mode']}")

    seen_run_ids = set()
    for path, result, generation, basis in results:
        run_id = result['run_id']
        # Two result files with one run id (a host script that launched several runs under a single --run-id) would replace each
        # other's family; qualify the later ones by their folder so every folder stays a distinct, receipt-backed family.
        same_file = lambda src: Path(src).resolve() == path.resolve() if src else False  # noqa: E731
        if run_id in seen_run_ids or any(f['family'] == f'matrix-kit:{run_id}' and not same_file(f.get('source')) for f in raw['families']):
            run_id = f"{run_id}#{path.parent.name}"
        seen_run_ids.add(run_id)
        legacy = generation == 'legacy'
        added['legacy_runs' if legacy else 'current_runs'].append(run_id)
        family = {'family': f'matrix-kit:{run_id}', 'source': str(path), 'cells': [], 'verdicts': [],
                  'kit_schema': result.get('kit_schema') if not legacy else 1,
                  **({'legacy_boundary_utc': LEGACY_RESULT_BOUNDARY_UTC, 'legacy_basis': basis,
                      'legacy_note': 'legacy receipt (kit schema 1): bound under the pre-boundary rules for full-artifact receipts, nothing relabelled; contained_resident blocks are not legacy evidence'} if legacy else {})}
        # re-ingesting a run replaces its own mode cells too (a recovered verdict or a corrected note must reach the mode table)
        raw['mode_cells'] = [c for c in raw['mode_cells'] if c.get('source') != run_id]
        existing_mode = {(c['model_id'], c['slot'], c['mode'], c.get('source')) for c in raw['mode_cells']}
        groups = {}
        for c in result['cells']:
            if c['state'] in ('pass', 'fail'):
                gate = resource_gate(resource_qualification(path.parent, c)) if legacy else current_gate(path.parent, c, result)
            else:
                gate = None
            if gate is not None:
                # resource qualification not established (exceeded, unknown, or in a current run: receipt or evidence missing):
                # the cell enters as untried with the reason, never as pass or fail; the exercise state stays in the note
                c = dict(c, state='untried', note=gate + ' | exercise ' + c['state'] + ': ' + (c.get('note') or ''))
                added['gated_resource_qualification'] = added.get('gated_resource_qualification', 0) + 1
            elif c['state'] not in ('pass', 'fail', 'na'):
                added['skipped_untried'] += 1
                continue
            if skipped(run_id, c):
                added['skipped_superseded'] = added.get('skipped_superseded', 0) + 1
                continue
            groups.setdefault((c['model_id'], c['slot'], c['mode']), []).append(c)
        for (model_id, slot, mode), cells in groups.items():
            scored = []
            for c in cells:
                v = verdict_for(c)
                upheld = v is not None and v.get('upheld') is True  # exactly boolean True; 'true', 'false', 1 or [True] never uphold
                scored.append((RANK[c['state']], 1 if (c['state'] == 'pass' and upheld) else 0, c, v, upheld))
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            _, _, c, v, upheld = scored[0]
            note = f"[{mode}] " + (c.get('note') or '')
            if v is not None and type(v.get('upheld')) is not bool:
                note += f" | verifier flag not boolean: {v.get('upheld')!r}"[:80]
            if len(scored) > 1:
                added['merged_shared_slot'] += len(scored) - 1
                note += ' | also on ' + '; '.join(
                    f"{o.get('device_named') or o.get('device')}: {o['state']}" + (' (verifier refuted)' if (o['state'] == 'pass' and ov is not None and not ou) else '')
                    for _, _, o, ov, ou in scored[1:])
            base = {'model_id': model_id, 'slot': slot, 'state': c['state'], 'evidence_ids': [a.evidence], 'device_named': c.get('device_named', ''),
                    # display reference only (<lab-records>/<runs>/<run folder>/<receipt>, tools/lab_paths.py): the receipt that decided
                    # eligibility was resolved against the run folder above (read_receipt), never from this string, and no host path is written
                    'package_or_runtime': c.get('package_or_runtime', ''), 'receipt': records_reference(path.parent, c.get('receipt', '')), 'note': note,
                    'verified': upheld if c['state'] == 'pass' else None, 'dtype': c.get('dtype')}
            if c.get('device'):
                base['device'] = c['device']
            if mode == 'single':
                family['cells'].append(dict(base))
                if c['state'] == 'pass' and v is not None:
                    family['verdicts'].append({'model_id': model_id, 'slot': slot, 'upheld': upheld, 'corrected_state': v.get('corrected_state', 'partial'), 'reasoning': v.get('reasoning', '')})
                    added['verdicts'] += 1
                added['cells'] += 1
            mkey = (model_id, slot, mode, run_id)
            if mkey not in existing_mode:
                raw['mode_cells'].append(dict(base, mode=mode, source=run_id))
                existing_mode.add(mkey)
                added['mode_cells'] += 1
        raw['families'] = [f for f in raw['families'] if f['family'] != family['family']] + [family]
    unmatched = sorted(f'{run}:{key}' for run, keys in skip.items() for key in keys if (run, key) not in used_skips)
    if unmatched:
        added['unmatched_skips'] = unmatched  # a skip that matched no cell: a typo, or a run id that ingest qualified as '<run_id>#<folder>'
    a.raw.write_text(json.dumps(raw, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(added))


if __name__ == '__main__':
    main()
