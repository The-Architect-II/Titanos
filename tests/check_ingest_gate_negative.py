"""Negative proof of the matrix-kit ingest gate (tools/matrix_kit/ingest.py) for CURRENT runs (RUN_RESULT kit_schema >= 2): a contained
cell enters the matrix as pass or fail only through an existing, parseable, receipt_schema 2 detailed receipt whose resource_qualification
block carries the typed cap and readback, a measured peak under the cap, a valid sensor sequence, the complete placement identity and
cleanup evidence. Anything less is untried with the functional output retained separately in the note. Old full-artifact receipts are
bound only through the explicit legacy boundary (no kit_schema marker AND utc before ingest.LEGACY_RESULT_BOUNDARY_UTC, or --legacy-run).

Red proof first: the two owner-reported cases are driven through the PRE-CHANGE ingest (commit 467f764, read with `git show`, never
checked out) and must be accepted there (pass, verified) - the defect - and refused by the working-tree ingest.

  N1  contained_resident within + valid identity, no cap/readback/peak/samples/cleanup  -> untried resource_qualification_evidence_incomplete
  N2  RUN_RESULT row within + identity, detailed receipt file missing (or not JSON)      -> untried receipt_missing_or_unparseable
  N3  receipt present, receipt_schema missing (or < 2, or no block) in a kit_schema 2 run -> untried receipt_schema_missing
  N4  a sample under the reserve, a non-monotonic utc, a missing phase, a None GPU reading -> untried evidence_incomplete
  N5  legacy run (no kit_schema, utc before the boundary, file hash pinned in the legacy manifest or --legacy-run) with old
      full-artifact receipts -> enters as before; a contained_resident block is never legacy evidence (untried legacy_contained);
      an unmarked back-dated result whose hash is not pinned -> refused
  N6  a run without kit_schema whose utc is after the boundary (or missing)                 -> refused with an error unless --legacy-run
  N7  the receipt is not the row's own: another cell's complete receipt, a path outside the run folder, a receipt whose run_id /
      model_id / slot / mode / device differ from the row, a dtype or copies that differ from the row -> untried, never pass
  N8  the receipt's own sensors contradict the verdict: a sample peak at the cap, a final owned-job query peak at the cap, a declared
      measured peak below a sample or final peak; reserve differing from RUN_RESULT host_reserve_bytes; floors.commit != cap + reserve
  N9  a music/resident receipt relabelled artifact_floor (no kit plan reason, or containment_unavailable on a host whose RUN_RESULT
      says containment was available) -> untried; the kit's real artifact_floor reasons for non-resident specs -> pass
  N10 a contained cell that FAILED at model load (phases launch, server_start, model_load; exercise error) -> enters as fail, never
      untried; the same phases on a pass row -> untried; phases out of the kit order -> untried
  P1  a complete synthetic contained receipt                                                -> pass, verified true under a boolean upheld verdict
  plus: exceeded/unknown stay untried in current runs; artifact_floor needs not_contained; profile_key is a real 64-hex sha256 in both
  ingest.missing_identity and run_cells.missing_identity ('k' * 64 and uppercase hex are rejected); run_cells.artifact_floor_reason and
  ingest.artifact_floor_reason are the same three texts.

Run from the repo root: PYTHONPATH=src python tests/check_ingest_gate_negative.py (exit 0 = pass).
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'tools' / 'matrix_kit'
sys.path.insert(0, str(KIT))
import ingest  # noqa: E402
import run_cells  # noqa: E402

INGEST = KIT / 'ingest.py'
PRE_CHANGE_COMMIT = '467f764'  # the resident-placement / contained-admission repair whose ingest accepted N1 and N2
SLOT = 'win-x64-nvidia-cuda'
BEFORE_BOUNDARY, AFTER_BOUNDARY = '2026-09-25T05:00:00+00:00', '2026-09-25T19:30:00+00:00'
CAP, RESERVE, GIB = 20 << 30, 6 << 30, 1 << 30
failures, observations = [], {}


def expect(cond, msg):
    if not cond:
        failures.append(msg)


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


KEY, RECEIPT_SHA, ADAPTER_SHA = sha('profile-key'), sha('music-runtime.json'), sha('adapter.safetensors')
IDENTITY = {'profile_key': KEY, 'receipt_sha256': RECEIPT_SHA, 'adapter_sha256': ADAPTER_SHA, 'placement': 'resident', 'dtype': 'bfloat16', 'copies': 1}
THIN_WITHIN = {'kind': 'contained_resident', 'bound': 'job_commit', 'verdict': 'within', **IDENTITY}  # the owner's N1 block: identity, no measurement


def sample(second, phase, **over):
    """A guard sample in the shape containment.CellGuard.sample writes. The 'launch' sample precedes the owned tree (the kit's check
    immediately before creating any process) and so carries host and GPU readings only; later phases carry the job peak."""
    s = {'utc': f'2026-09-25T20:00:{second:02d}+00:00', 'phase': phase, 'available_commit_bytes': 40 * GIB, 'available_physical_bytes': 30 * GIB,
         'gpu_free_mib': 20000, 'gpu_used_mib': 12000}
    if phase != 'launch':
        s.update(peak_job_commit_bytes=(8 + second) * GIB // 4, peak_process_commit_bytes=2 * GIB, active_processes=2, tree_peak_working_set_bytes=9 * GIB)
    s.update(over)
    return s


def complete_block(**over):
    """A gate-valid contained_resident block in the shape run_cells writes (cap, readback, peaks, samples, identity, cleanup)."""
    block = {'kind': 'contained_resident', 'bound': 'job_commit', 'verdict': 'within', 'floors': {'commit': CAP + RESERVE, 'gpu': 8192},
             'host_reserve_bytes': RESERVE, 'job_commit_limit_bytes': CAP, 'job_limit_source': 'first_run_containment_cap', **IDENTITY,
             'containment': {'requested_limit_bytes': CAP, 'job_memory_limit_bytes': CAP, 'limit_flags': 0x2200, 'child_in_job': True, 'parent_in_job': False,
                             'verified_before_resume': True, 'utc': '2026-09-25T19:59:59+00:00'},
             'measured_peak_job_commit_bytes': 11 * GIB, 'measured_peak_process_commit_bytes': 9 * GIB, 'measured_peak_tree_working_set_bytes': 12 * GIB,
             'limit_hit': False, 'readings_complete': True, 'trip': None, 'reason': 'first run',
             'samples': [sample(0, 'launch'), sample(2, 'server_start'), sample(4, 'model_load'), sample(6, 'model_load'), sample(8, 'request'), sample(10, 'request'), sample(12, 'unload')],
             'cleanup': {'launched': True, 'active_processes_after': 0, 'job_handle_closed': True, 'forced_cleanup': False, 'active_before_close': 0,
                         'final_query': {'peak_job_commit_bytes': 11 * GIB, 'peak_process_commit_bytes': 9 * GIB, 'active_processes': 0, 'total_processes': 3, 'job_memory_limit_bytes': CAP},
                         'server_returncode': 1, 'utc': '2026-09-25T20:00:20+00:00'}}
    block.update(over)
    return block


SLOT_DICT = {'slot': SLOT, 'device': 'GPU-0', 'device_named': 'x', 'runtime': 'torch-cuda'}
ARTIFACT_REASON = ingest.artifact_floor_reason(None, True)  # a non-resident spec on a GPU slot: the kit's artifact_floor plan reason


def receipt(state, rq, schema=2, exercise='same', **binding):
    """A detailed receipt in the shape run_cells.cell_receipt writes: the binding fields (run_id, slot dict, mode; model_id is filled from
    the receipts key by run_ingest), spec_placement (resident for a contained block, None otherwise) and the exercise beside the block."""
    if exercise == 'same':
        exercise = {'state': state, 'output': {'wav_bytes': 241732}} if state == 'pass' else {'state': state, 'error': 'RuntimeError: load refused HTTP 500'}
    body = {'state': state, 'note': f'{state} note', 'exercise': exercise, 'run_id': 'gate-run', 'slot': dict(SLOT_DICT), 'mode': 'single',
            'spec_placement': 'resident' if isinstance(rq, dict) and rq.get('kind') == 'contained_resident' else None}
    if schema is not None:
        body['receipt_schema'] = schema
    if rq is not None:
        body['resource_qualification'] = rq
    body.update(binding)
    return body


def row(mid, state, rq=None, receipt_name='default'):
    r = {'model_id': mid, 'slot': SLOT, 'mode': 'single', 'state': state, 'device': 'GPU-0', 'device_named': 'x', 'note': f'{state} note', 'dtype': 'bfloat16'}
    if receipt_name is not None:
        r['receipt'] = f'receipts/{mid}.json' if receipt_name == 'default' else receipt_name
    if rq:  # the display fields run_cells copies into RUN_RESULT rows; they must never decide eligibility
        r.update(resource_kind=rq.get('kind'), resource_verdict=rq.get('verdict'), resource_profile_key=rq.get('profile_key'),
                 resource_identity={k: rq.get(k) for k in ingest.IDENTITY_FIELDS})
    return r


def run_ingest(rows, receipts, *, kit_schema=2, utc=AFTER_BOUNDARY, run_id='gate-run', verdicts='uphold', extra=(), ingest_path=INGEST, raw_receipts=None,
               result_extra=None, outside=None, pin=False):
    """Drive ingest.py as a subprocess on a synthetic run folder. Returns (proc, family cells by model, family verdicts by model, mode cells
    by model, family dict, raw after). The RUN_RESULT carries the kit's run-level fields (host_reserve_bytes, containment_available,
    floors) unless result_extra overrides them; `outside` writes files beside (not inside) the run folder; pin=True writes a legacy
    manifest pinning the RUN_RESULT's sha256 and passes it as --legacy-manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / 'raw.json'
        raw.write_text(json.dumps({'catalog': {}, 'families': [], 'mode_cells': []}), encoding='utf-8')
        d = Path(tmp) / 'kit'
        (d / 'receipts').mkdir(parents=True)
        for mid, body in receipts.items():
            body = dict(body)
            body.setdefault('model_id', mid)
            (d / 'receipts' / f'{mid}.json').write_text(json.dumps(body), encoding='utf-8')
        for name, text in (raw_receipts or {}).items():
            (d / 'receipts' / name).write_text(text, encoding='utf-8')
        for name, body in (outside or {}).items():
            (Path(tmp) / 'other').mkdir(exist_ok=True)
            (Path(tmp) / 'other' / name).write_text(json.dumps(body), encoding='utf-8')
        result = {'run_id': run_id, 'cells': rows, 'host_reserve_bytes': RESERVE, 'containment_available': True, 'floors': {'commit': 4 * GIB, 'gpu': 8192}}
        result.update(result_extra or {})
        result = {k: v for k, v in result.items() if v is not ...}
        if kit_schema is not None:
            result['kit_schema'] = kit_schema
        if utc is not None:
            result['utc'] = utc
        (d / 'RUN_RESULT.json').write_text(json.dumps(result), encoding='utf-8')
        cmd = [sys.executable, str(ingest_path), '--raw', str(raw), '--evidence', 'E-000', '--result', str(d / 'RUN_RESULT.json'), *extra]
        if pin:
            mp = Path(tmp) / 'legacy_results.json'
            mp.write_text(json.dumps({'results': [{'sha256': hashlib.sha256((d / 'RUN_RESULT.json').read_bytes()).hexdigest(), 'run_id': run_id}]}), encoding='utf-8')
            cmd += ['--legacy-manifest', str(mp)]
        if verdicts is not None:
            vp = Path(tmp) / 'verdicts.json'
            if verdicts == 'uphold':
                verdicts = {f"{r['model_id']}|{SLOT}|single": {'upheld': True, 'corrected_state': 'pass', 'reasoning': 'valid WAV'} for r in rows}
            vp.write_text(json.dumps(verdicts), encoding='utf-8')
            cmd += ['--verdicts', str(vp)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        data = json.loads(raw.read_text(encoding='utf-8'))
        fam = next((f for f in data['families'] if f['family'] == f'matrix-kit:{run_id}'), {'cells': [], 'verdicts': []})
        return proc, {c['model_id']: c for c in fam['cells']}, {v['model_id']: v for v in fam.get('verdicts', [])}, {c['model_id']: c for c in data['mode_cells']}, fam, data


# --- RED: the owner's two cases against the pre-change ingest (commit 467f764), read-only through git show --------------------------
with tempfile.TemporaryDirectory() as tmp:
    shown = subprocess.run(['git', 'show', f'{PRE_CHANGE_COMMIT}:tools/matrix_kit/ingest.py'], capture_output=True, text=True, cwd=str(ROOT))
    if shown.returncode != 0 or 'def resource_gate' not in shown.stdout:
        observations['red_proof'] = {'available': False, 'reason': f'git show {PRE_CHANGE_COMMIT}:tools/matrix_kit/ingest.py failed: {shown.stderr[-200:]}'}
    else:
        old = Path(tmp) / 'ingest_pre_change.py'
        old.write_text(shown.stdout, encoding='utf-8')
        red = {'commit': PRE_CHANGE_COMMIT, 'ingest_sha256': hashlib.sha256(shown.stdout.encode('utf-8')).hexdigest(), 'available': True}
        proc, cells, fv, modes, _, _ = run_ingest([row('n1', 'pass', THIN_WITHIN)], {'n1': receipt('pass', THIN_WITHIN)}, ingest_path=old)
        red['N1'] = {'rc': proc.returncode, 'state': cells.get('n1', {}).get('state'), 'verified': cells.get('n1', {}).get('verified'), 'note': (cells.get('n1', {}).get('note') or '')[:100]}
        expect(proc.returncode == 0 and red['N1']['state'] == 'pass' and red['N1']['verified'] is True,
               f'RED N1: the pre-change ingest accepted a contained within block without any measurement as an upheld pass (that is the defect this test pins): {red["N1"]} {proc.stderr[-200:]}')
        proc, cells, fv, modes, _, _ = run_ingest([row('n2', 'pass', THIN_WITHIN)], {}, ingest_path=old)
        red['N2'] = {'rc': proc.returncode, 'state': cells.get('n2', {}).get('state'), 'verified': cells.get('n2', {}).get('verified'), 'note': (cells.get('n2', {}).get('note') or '')[:100]}
        expect(proc.returncode == 0 and red['N2']['state'] == 'pass' and red['N2']['verified'] is True,
               f'RED N2: the pre-change ingest fell back to the RUN_RESULT row when the receipt was missing and accepted an upheld pass: {red["N2"]} {proc.stderr[-200:]}')
        # N7 red: a row pointing at another cell's receipt (a complete measured block of cell 'a') entered as an upheld pass for 'b'
        proc, cells, fv, modes, _, _ = run_ingest([row('b', 'pass', THIN_WITHIN, receipt_name='receipts/a.json')], {'a': receipt('pass', complete_block())}, ingest_path=old)
        red['N7'] = {'rc': proc.returncode, 'state': cells.get('b', {}).get('state'), 'verified': cells.get('b', {}).get('verified')}
        expect(proc.returncode == 0 and red['N7']['state'] == 'pass' and red['N7']['verified'] is True, f"RED N7: the pre-change ingest never bound a row to its own receipt: {red['N7']}")
        # N5 red: an unmarked result with a back-dated utc took the legacy path on its own say-so (no pinned population)
        proc, cells, fv, modes, _, _ = run_ingest([row('n5', 'pass', THIN_WITHIN)], {}, kit_schema=None, utc=BEFORE_BOUNDARY, run_id='backdated', ingest_path=old)
        red['N5'] = {'rc': proc.returncode, 'state': cells.get('n5', {}).get('state')}
        expect(proc.returncode == 0 and red['N5']['state'] == 'pass', f"RED N5: the pre-change ingest trusted the self-reported utc for the legacy path: {red['N5']}")
        observations['red_proof'] = red

# --- N1: identity without measurement, receipt present, current run ------------------------------------------------------------------
proc, cells, fv, modes, fam, _ = run_ingest([row('n1', 'pass', THIN_WITHIN)], {'n1': receipt('pass', THIN_WITHIN)})
c = cells.get('n1', {})
expect(proc.returncode == 0, f'N1 ingest runs: rc {proc.returncode} {proc.stderr[-400:]}')
expect(c.get('state') == 'untried' and c.get('verified') is None and (c.get('note') or '').startswith('[single] resource_qualification_evidence_incomplete: '),
       f"N1: a contained within block with identity but no cap/readback/peak/samples/cleanup enters untried: {c}")
expect('functional output retained separately (exercise pass)' in (c.get('note') or '') and '| exercise pass:' in (c.get('note') or ''), f"N1: the exercise pass is retained in the note: {c.get('note')}")
for needle in ('job_commit_limit_bytes', 'containment readback missing', 'measured_peak_job_commit_bytes', 'samples missing or empty', 'cleanup evidence missing'):
    expect(needle in (c.get('note') or ''), f'N1: the note names the missing evidence {needle!r}: {c.get("note")}')
expect(modes.get('n1', {}).get('state') == 'untried' and 'n1' not in fv, f"N1: the mode cell is untried and no verdict is bound: {modes.get('n1')} {fv}")
observations['N1'] = {'state': c.get('state'), 'note': (c.get('note') or '')[:160]}
direct = ingest.validate_contained_block(THIN_WITHIN, receipt('pass', THIN_WITHIN))
expect(len(direct) >= 8 and not [p for p in direct if 'identity' in p], f'N1 pure validator: measurement problems, none about the (complete) identity: {direct}')

# --- N2: the row says within + identity, the detailed receipt is missing or unparseable -----------------------------------------------
proc, cells, fv, modes, _, _ = run_ingest([row('n2', 'pass', THIN_WITHIN), row('n2b', 'pass', complete_block(), receipt_name='receipts/n2b.json')], {},
                                          raw_receipts={'n2b.json': '{"state": "pass", "receipt_schema": 2, '})
for mid in ('n2', 'n2b'):
    c = cells.get(mid, {})
    expect(proc.returncode == 0 and c.get('state') == 'untried' and c.get('verified') is None and (c.get('note') or '').startswith('[single] receipt_missing_or_unparseable: ')
           and '| exercise pass:' in (c.get('note') or ''), f"N2 {mid}: a missing or unparseable detailed receipt never falls back to the row fields: rc {proc.returncode} {c} {proc.stderr[-200:]}")
    expect(modes.get(mid, {}).get('state') == 'untried', f'N2 {mid}: mode cell untried too: {modes.get(mid)}')
observations['N2'] = {m: (cells.get(m, {}).get('note') or '')[:120] for m in ('n2', 'n2b')}
proc, cells, _, _, _, _ = run_ingest([row('n2c', 'pass', complete_block(), receipt_name=None)], {'n2c': receipt('pass', complete_block())})
expect(cells.get('n2c', {}).get('state') == 'untried' and 'names no receipt file' in (cells.get('n2c', {}).get('note') or ''), f'N2c: a row without a receipt name is untried: {cells.get("n2c")}')

# --- N3: receipt present but not schema 2 (or without the block) in a current run -----------------------------------------------------
proc, cells, fv, modes, _, _ = run_ingest([row('n3', 'pass', complete_block()), row('n3-one', 'pass', complete_block()), row('n3-noblock', 'pass', complete_block()), row('n3-str', 'pass', complete_block())],
                                          {'n3': receipt('pass', complete_block(), schema=None), 'n3-one': receipt('pass', complete_block(), schema=1),
                                           'n3-noblock': receipt('pass', None), 'n3-str': receipt('pass', complete_block(), schema='2')})
for mid in ('n3', 'n3-one', 'n3-noblock', 'n3-str'):
    c = cells.get(mid, {})
    expect(proc.returncode == 0 and c.get('state') == 'untried' and (c.get('note') or '').startswith('[single] receipt_schema_missing: ') and '| exercise pass:' in (c.get('note') or ''),
           f"N3 {mid}: a receipt without receipt_schema >= 2 (int) or without the block is untried, even with a complete block: {c} {proc.stderr[-200:]}")
observations['N3'] = {m: (cells.get(m, {}).get('note') or '')[:120] for m in ('n3', 'n3-noblock')}

# --- N4: sensor sequence defects -------------------------------------------------------------------------------------------------------
under_reserve = complete_block(samples=[sample(0, 'launch'), sample(2, 'model_load'), sample(4, 'request', available_commit_bytes=RESERVE - 1)])
backwards = complete_block(samples=[sample(0, 'launch'), sample(9, 'model_load'), sample(4, 'request')])
proc, cells, fv, modes, _, _ = run_ingest([row('n4-reserve', 'pass', under_reserve), row('n4-utc', 'pass', backwards)], {'n4-reserve': receipt('pass', under_reserve), 'n4-utc': receipt('pass', backwards)})
c1, c2 = cells.get('n4-reserve', {}), cells.get('n4-utc', {})
expect(proc.returncode == 0 and c1.get('state') == 'untried' and 'under the reserve' in (c1.get('note') or '') and c1.get('note', '').startswith('[single] resource_qualification_evidence_incomplete'),
       f'N4: a sample under the shared-host reserve makes the cell untried: {c1} {proc.stderr[-200:]}')
expect(c2.get('state') == 'untried' and 'earlier than the previous sample' in (c2.get('note') or ''), f'N4: a non-monotonic sample utc makes the cell untried: {c2}')
observations['N4'] = {'reserve': (c1.get('note') or '')[:140], 'utc': (c2.get('note') or '')[:140]}
# the pure validator, one defect at a time: every mutation of a complete block must be named (and the complete block must be clean)
expect(ingest.validate_contained_block(complete_block(), receipt('pass', complete_block())) == [], f'a complete block has no problems: {ingest.validate_contained_block(complete_block(), receipt("pass", complete_block()))}')
mutations = {
    'kind': lambda b: b.update(kind='artifact_floor'), 'bound': lambda b: b.update(bound=None), 'verdict': lambda b: b.update(verdict='not_contained'),
    'cap zero': lambda b: b.update(job_commit_limit_bytes=0), 'cap bool': lambda b: b.update(job_commit_limit_bytes=True), 'cap str': lambda b: b.update(job_commit_limit_bytes=str(CAP)),
    'reserve missing': lambda b: b.pop('host_reserve_bytes'), 'floors gpu None': lambda b: b['floors'].update(gpu=None), 'floors missing': lambda b: b.pop('floors'),
    'readback missing': lambda b: b.pop('containment'), 'readback limit differs': lambda b: b['containment'].update(job_memory_limit_bytes=CAP - 1),
    'readback requested differs': lambda b: b['containment'].update(requested_limit_bytes=CAP + 1), 'readback flags no kill-on-close': lambda b: b['containment'].update(limit_flags=0x200),
    'readback breakaway': lambda b: b['containment'].update(limit_flags=0x2200 | 0x800), 'readback child not in job': lambda b: b['containment'].update(child_in_job=False),
    'readback unverified': lambda b: b['containment'].update(verified_before_resume=False), 'readback flags str': lambda b: b['containment'].update(limit_flags='0x2200'),
    'peak missing': lambda b: b.update(measured_peak_job_commit_bytes=None), 'peak at cap': lambda b: b.update(measured_peak_job_commit_bytes=CAP), 'peak zero': lambda b: b.update(measured_peak_job_commit_bytes=0),
    'process peak missing': lambda b: b.pop('measured_peak_process_commit_bytes'), 'limit_hit': lambda b: b.update(limit_hit=True), 'limit_hit None': lambda b: b.update(limit_hit=None),
    'readings incomplete': lambda b: b.update(readings_complete=False), 'readings missing': lambda b: b.pop('readings_complete'), 'trip': lambda b: b.update(trip={'kind': 'stopped'}),
    'samples empty': lambda b: b.update(samples=[]), 'samples missing': lambda b: b.pop('samples'), 'sample not dict': lambda b: b['samples'].append('x'),
    'sample no phase': lambda b: b['samples'][2].pop('phase'), 'sample bad utc': lambda b: b['samples'][1].update(utc='yesterday'),
    'sample gpu None': lambda b: b['samples'][3].update(gpu_free_mib=None), 'sample gpu used None': lambda b: b['samples'][3].update(gpu_used_mib=None),
    'sample physical None': lambda b: b['samples'][4].update(available_physical_bytes=None), 'sample commit missing': lambda b: b['samples'][4].pop('available_commit_bytes'),
    'sample peak missing': lambda b: b['samples'][4].pop('peak_job_commit_bytes'), 'sample peak over cap': lambda b: b['samples'][4].update(peak_job_commit_bytes=CAP + 1),
    'launch sample no host reading': lambda b: b['samples'][0].update(available_commit_bytes=None), 'launch sample no gpu reading': lambda b: b['samples'][0].pop('gpu_free_mib'),
    'server_start sample no peak (tree exists)': lambda b: b['samples'][1].pop('peak_job_commit_bytes'),
    'sample physical under reserve': lambda b: b['samples'][5].update(available_physical_bytes=RESERVE - 1),
    'no request phase': lambda b: b.update(samples=[s for s in b['samples'] if s['phase'] != 'request']), 'no model_load phase': lambda b: b.update(samples=[s for s in b['samples'] if s['phase'] != 'model_load']),
    'profile_key k*64': lambda b: b.update(profile_key='k' * 64), 'profile_key upper': lambda b: b.update(profile_key=KEY.upper()), 'profile_key 63': lambda b: b.update(profile_key=KEY[:63]),
    'receipt sha short': lambda b: b.update(receipt_sha256='abc'), 'adapter sha upper': lambda b: b.update(adapter_sha256=ADAPTER_SHA.upper()), 'placement auto': lambda b: b.update(placement='auto'),
    'dtype empty': lambda b: b.update(dtype=''), 'copies bool': lambda b: b.update(copies=True), 'copies zero': lambda b: b.update(copies=0), 'identity missing': lambda b: [b.pop(k) for k in ingest.IDENTITY_FIELDS],
    'cleanup missing': lambda b: b.pop('cleanup'), 'cleanup not launched': lambda b: b['cleanup'].update(launched=False), 'cleanup launched missing': lambda b: b['cleanup'].pop('launched'),
    'cleanup live process': lambda b: b['cleanup'].update(active_processes_after=1), 'cleanup active None': lambda b: b['cleanup'].update(active_processes_after=None),
    'cleanup handle open': lambda b: b['cleanup'].update(job_handle_closed=False), 'cleanup forced str': lambda b: b['cleanup'].update(forced_cleanup='no'),
    'cleanup final_query missing': lambda b: b['cleanup'].pop('final_query'), 'cleanup final_query_error': lambda b: b['cleanup'].update(final_query_error='OSError: handle'),
    # N8: the receipt's own sensors must agree with the verdict and with the kit's admission rule
    'sample peak at cap': lambda b: b['samples'][4].update(peak_job_commit_bytes=CAP), 'final_query peak at cap': lambda b: b['cleanup']['final_query'].update(peak_job_commit_bytes=CAP),
    'final_query peak over cap': lambda b: b['cleanup']['final_query'].update(peak_job_commit_bytes=CAP + GIB),
    'measured peak below a sample peak': lambda b: b['samples'][4].update(peak_job_commit_bytes=19 * GIB), 'measured peak below final peak': lambda b: b['cleanup']['final_query'].update(peak_job_commit_bytes=12 * GIB),
    'floors commit not cap+reserve': lambda b: b['floors'].update(commit=CAP + RESERVE + 1), 'floors commit below cap': lambda b: b['floors'].update(commit=CAP - 1),
    # N10: the phase sequence is the kit's
    'phase unknown': lambda b: b['samples'][2].update(phase='warmup'), 'phase out of order': lambda b: b['samples'][5].update(phase='server_start'),
    'no launch phase': lambda b: b.update(samples=b['samples'][1:]),
}
for name, mutate in mutations.items():
    block = complete_block()
    mutate(block)
    problems = ingest.validate_contained_block(block, receipt('pass', block))
    expect(problems, f'validator mutation {name!r} must be reported (got no problems)')
observations['validator_mutations'] = len(mutations)
# the launch sample legitimately has no job peak (no tree yet); a launch sample that happens to carry one is fine too
expect(ingest.validate_contained_block(complete_block(), receipt('pass', complete_block())) == [] and 'peak_job_commit_bytes' not in complete_block()['samples'][0]
       and ingest.validate_contained_block(complete_block(samples=[sample(0, 'launch', peak_job_commit_bytes=0)] + complete_block()['samples'][1:]), receipt('pass', complete_block())) == [],
       'the pre-launch sample carries host and GPU readings without a job peak; later phases must carry the peak')
# exercise block: missing, untried, or disagreeing with the row
expect(ingest.validate_contained_block(complete_block(), receipt('pass', complete_block(), exercise=None)) and ingest.validate_contained_block(complete_block(), receipt('pass', complete_block(), exercise={'state': 'untried'}))
       and ingest.validate_contained_block(complete_block(), receipt('pass', complete_block()), row_state='fail'), 'the exercise block must exist, be pass/fail and agree with the row')

# --- N5: legacy run (no kit_schema, utc before the boundary): exactly as before -------------------------------------------------------
legacy_thin_noid = {'kind': 'contained_resident', 'verdict': 'within', 'profile_key': KEY}
legacy_rows = [row('l-old', 'pass'), row('l-fail', 'fail'), row('l-thin', 'pass', THIN_WITHIN), row('l-noid', 'pass', legacy_thin_noid), row('l-missing-receipt', 'pass', receipt_name='receipts/none.json'),
               row('l-exceeded', 'pass', {'kind': 'contained_resident', 'verdict': 'exceeded', 'measured_peak_job_commit_bytes': 30, 'job_commit_limit_bytes': 20}), row('l-na', 'na')]
legacy_receipts = {'l-old': receipt('pass', None, schema=None), 'l-fail': receipt('fail', None, schema=None), 'l-thin': receipt('pass', THIN_WITHIN, schema=None), 'l-noid': receipt('pass', legacy_thin_noid, schema=None),
                   'l-exceeded': receipt('pass', {'kind': 'contained_resident', 'verdict': 'exceeded', 'measured_peak_job_commit_bytes': 30, 'job_commit_limit_bytes': 20}, schema=None)}
proc, cells, fv, modes, fam, _ = run_ingest(legacy_rows, legacy_receipts, kit_schema=None, utc=BEFORE_BOUNDARY, run_id='legacy-run', pin=True)
expect(proc.returncode == 0, f'N5: a pinned legacy run ingests: rc {proc.returncode} {proc.stderr[-400:]}')
expect(cells.get('l-old', {}).get('state') == 'pass' and cells['l-old'].get('verified') is True and fv.get('l-old', {}).get('upheld') is True and cells['l-old']['note'] == '[single] pass note',
       f"N5: an old full-artifact receipt without the block enters as before (pass, verified per upheld, note untouched): {cells.get('l-old')} {fv.get('l-old')}")
expect(cells.get('l-fail', {}).get('state') == 'fail' and cells.get('l-na', {}).get('state') == 'na', f"N5: legacy fail and na unchanged: {cells.get('l-fail')} {cells.get('l-na')}")
for mid in ('l-thin', 'l-noid'):
    c = cells.get(mid, {})
    expect(c.get('state') == 'untried' and c.get('verified') is None and (c.get('note') or '').startswith('[single] resource_qualification_legacy_contained: ') and 'artifact_floor receipts only' in c['note']
           and '| exercise pass:' in c['note'], f"N5 {mid}: a contained_resident block is never legacy evidence (the boundary covers full-artifact receipts only): {c}")
expect('identity missing profile_key' not in (cells.get('l-thin', {}).get('note') or '') and 'identity missing receipt_sha256' in (cells.get('l-noid', {}).get('note') or ''),
       f"N5: the legacy note still names the missing identity fields: {cells.get('l-noid', {}).get('note')}")
expect(cells.get('l-missing-receipt', {}).get('state') == 'pass', f"N5: a legacy row whose receipt file is absent enters as before (no block, no gate): {cells.get('l-missing-receipt')}")
expect(cells.get('l-exceeded', {}).get('state') == 'untried' and 'resource_qualification_exceeded' in (cells['l-exceeded'].get('note') or ''), f"N5: legacy exceeded stays untried: {cells.get('l-exceeded')}")
expect(fam.get('kit_schema') == 1 and fam.get('legacy_boundary_utc') == ingest.LEGACY_RESULT_BOUNDARY_UTC and 'legacy receipt (kit schema 1)' in (fam.get('legacy_note') or '') and fam.get('legacy_basis') == 'pinned_sha256',
       f'N5: the family records the legacy generation, the boundary and the pin explicitly: {{k: v for k, v in fam.items() if k != "cells"}}')
counts = json.loads(proc.stdout) if proc.returncode == 0 else {}
expect(counts.get('legacy_runs') == ['legacy-run'] and counts.get('current_runs') == [] and counts.get('gated_resource_qualification') == 3, f'N5: ingest reports the legacy run and the gated contained cells: {counts}')
observations['N5'] = {m: cells.get(m, {}).get('state') for m in ('l-old', 'l-thin', 'l-noid', 'l-missing-receipt')}
# the same unmarked, back-dated result without its hash pinned (the repo manifest pins only the real pre-boundary population) is refused
proc, cells, fv, modes, fam, data = run_ingest(legacy_rows, legacy_receipts, kit_schema=None, utc=BEFORE_BOUNDARY, run_id='legacy-run')
expect(proc.returncode != 0 and 'ingest refused' in proc.stderr and 'not in the pinned legacy population' in proc.stderr and '--legacy-run legacy-run' in proc.stderr and data['families'] == [],
       f'N5: an unmarked back-dated result whose sha256 is not pinned is refused before anything is written: rc {proc.returncode} {proc.stderr[-300:]}')
proc, cells, fv, modes, fam, _ = run_ingest(legacy_rows, legacy_receipts, kit_schema=None, utc=BEFORE_BOUNDARY, run_id='legacy-run', extra=('--legacy-run', 'legacy-run'))
expect(proc.returncode == 0 and cells.get('l-old', {}).get('state') == 'pass' and fam.get('legacy_basis') == 'legacy_run_flag' and cells.get('l-thin', {}).get('state') == 'untried',
       f'N5: --legacy-run binds it explicitly with the basis recorded, contained blocks still untried: rc {proc.returncode} {cells.get("l-old")} {fam.get("legacy_basis")}')
observations['N5_unpinned'] = proc.stderr.strip()[-160:]
# the pinned population: the repo manifest lists the 78 real pre-boundary results (sha256 of each RUN_RESULT.json), nothing else
pinned = ingest.load_legacy_manifest()
manifest = json.loads(ingest.LEGACY_RESULTS_MANIFEST.read_text(encoding='utf-8'))
expect(ingest.LEGACY_RESULTS_MANIFEST.is_file() and len(pinned) == len(manifest['results']) == 78 and all(ingest.HEX64.fullmatch(s) for s in pinned)
       and manifest.get('boundary_utc') == ingest.LEGACY_RESULT_BOUNDARY_UTC and all(ingest.parse_utc(r.get('utc')) is not None and ingest.parse_utc(r['utc']) < ingest.parse_utc(ingest.LEGACY_RESULT_BOUNDARY_UTC) for r in manifest['results']),
       f'the legacy manifest pins 78 hex digests of results dated before the boundary: {len(pinned)} {manifest.get("boundary_utc")}')
expect(ingest.load_legacy_manifest(Path('does/not/exist.json')) == frozenset(), 'a missing manifest pins nothing (fail closed)')
D = 'a' * 64
expect(ingest.result_generation({'run_id': 'x', 'utc': BEFORE_BOUNDARY}, digest=D, pinned={D}) == ('legacy', 'pinned_sha256')
       and ingest.result_generation({'run_id': 'x', 'utc': '2026-09-25T18:59:59Z'}, digest=D, pinned={D}) == ('legacy', 'pinned_sha256')
       and ingest.result_generation({'run_id': 'x', 'kit_schema': 2}) == ('current', None) and ingest.result_generation({'run_id': 'x', 'kit_schema': 3, 'utc': BEFORE_BOUNDARY}) == ('current', None)
       and ingest.result_generation({'run_id': 'x', 'kit_schema': ingest.CURRENT_KIT_SCHEMA - 1, 'utc': BEFORE_BOUNDARY}, digest=D, pinned={D}) == ('legacy', 'pinned_sha256'), 'result_generation: marker, or boundary + pinned hash')
for bad in ({'run_id': 'x', 'utc': AFTER_BOUNDARY}, {'run_id': 'x', 'utc': ingest.LEGACY_RESULT_BOUNDARY_UTC}, {'run_id': 'x'}, {'run_id': 'x', 'utc': 'not a date'}, {'run_id': 'x', 'kit_schema': '2', 'utc': AFTER_BOUNDARY},
            {'run_id': 'x', 'kit_schema': True, 'utc': AFTER_BOUNDARY}):
    gen, why = ingest.result_generation(bad, digest=D, pinned={D})
    expect(gen == 'refused' and 'refused' in (why or '') and '--legacy-run x' in (why or ''), f'result_generation refuses {bad}: {gen} {why}')
for unpinned in ({'run_id': 'x', 'utc': BEFORE_BOUNDARY},):
    expect(ingest.result_generation(unpinned, digest=D, pinned=frozenset())[0] == 'refused' and ingest.result_generation(unpinned, digest='b' * 64, pinned={D})[0] == 'refused'
           and ingest.result_generation(unpinned)[0] == 'refused' and 'not in the pinned legacy population' in ingest.result_generation(unpinned)[1], 'a back-dated unmarked result without a pinned hash is refused')
expect(ingest.result_generation({'run_id': 'x', 'utc': AFTER_BOUNDARY}, legacy_runs=['x']) == ('legacy', 'legacy_run_flag'), '--legacy-run names a post-boundary result explicitly')

# --- N6: no kit_schema and utc after the boundary (or missing): refused unless --legacy-run ------------------------------------------
for utc in (AFTER_BOUNDARY, None, ingest.LEGACY_RESULT_BOUNDARY_UTC):
    proc, cells, fv, modes, fam, data = run_ingest([row('n6', 'pass')], {'n6': receipt('pass', None, schema=None)}, kit_schema=None, utc=utc, run_id='post-boundary')
    expect(proc.returncode != 0 and 'ingest refused' in proc.stderr and 'kit_schema' in proc.stderr and '--legacy-run post-boundary' in proc.stderr,
           f'N6 utc={utc!r}: a result without kit_schema at/after the boundary is refused with a clear error: rc {proc.returncode} {proc.stderr[-300:]}')
    expect(data['families'] == [] and data['mode_cells'] == [] and not cells, f'N6 utc={utc!r}: nothing is written on refusal: {data}')
observations['N6'] = proc.stderr.strip()[-200:]
proc, cells, fv, modes, fam, _ = run_ingest([row('n6', 'pass')], {'n6': receipt('pass', None, schema=None)}, kit_schema=None, utc=AFTER_BOUNDARY, run_id='post-boundary', extra=('--legacy-run', 'post-boundary'))
expect(proc.returncode == 0 and cells.get('n6', {}).get('state') == 'pass' and fam.get('kit_schema') == 1, f'N6: --legacy-run binds the named result under the legacy rules: rc {proc.returncode} {cells.get("n6")} {proc.stderr[-200:]}')
# a refused result among several aborts the whole ingest before anything is written
proc, cells, fv, modes, fam, data = run_ingest([row('n6b', 'pass', complete_block())], {'n6b': receipt('pass', complete_block())}, kit_schema=None, utc=None, run_id='mixed')
expect(proc.returncode != 0 and data['families'] == [], f'N6: a current-looking folder without the marker is refused too (the marker, not the receipt detail, decides the generation): {proc.stderr[-200:]}')

# --- P1: a complete synthetic contained receipt, plus the other current-run outcomes --------------------------------------------------
exceeded = complete_block(verdict='exceeded', measured_peak_job_commit_bytes=CAP, limit_hit=True)
unknown = complete_block(verdict='unknown', readings_complete=False, measured_peak_job_commit_bytes=None, reason='resource readings incomplete: nvidia-smi missing')
artifact = {'kind': 'artifact_floor', 'bound': None, 'verdict': 'not_contained', 'floors': {'commit': 10 * GIB, 'gpu': 4096}, 'samples': [], 'reason': ARTIFACT_REASON,
            'cleanup': {'active_processes_after': None, 'job_handle_closed': None, 'forced_cleanup': False, 'launched': True}}
artifact_within = dict(artifact, verdict='within')
rows = [row('p1', 'pass', complete_block()), row('p1-fail', 'fail', complete_block()), row('p1-exceeded', 'pass', exceeded), row('p1-unknown', 'pass', unknown),
        row('p1-artifact', 'pass', artifact), row('p1-artifact-fail', 'fail', artifact), row('p1-artifact-within', 'pass', artifact_within), row('p1-na', 'na'),
        row('p1-refuted', 'pass', complete_block()), row('p1-strflag', 'pass', complete_block()), row('p1-untried', 'untried')]
receipts = {'p1': receipt('pass', complete_block()), 'p1-fail': receipt('fail', complete_block()), 'p1-exceeded': receipt('pass', exceeded), 'p1-unknown': receipt('pass', unknown),
            'p1-artifact': receipt('pass', artifact), 'p1-artifact-fail': receipt('fail', artifact), 'p1-artifact-within': receipt('pass', artifact_within),
            'p1-refuted': receipt('pass', complete_block()), 'p1-strflag': receipt('pass', complete_block())}
verdicts = {f'{m}|{SLOT}|single': {'upheld': True, 'corrected_state': 'pass', 'reasoning': 'valid output'} for m in receipts}
verdicts[f'p1-refuted|{SLOT}|single'] = {'upheld': False, 'corrected_state': 'partial', 'reasoning': 'refuted'}
verdicts[f'p1-strflag|{SLOT}|single'] = {'upheld': 'true', 'corrected_state': 'pass', 'reasoning': 'string flag'}
proc, cells, fv, modes, fam, _ = run_ingest(rows, receipts, verdicts=verdicts)
expect(proc.returncode == 0, f'P1 ingest runs: rc {proc.returncode} {proc.stderr[-400:]}')
c = cells.get('p1', {})
expect(c.get('state') == 'pass' and c.get('verified') is True and fv.get('p1', {}).get('upheld') is True and type(fv.get('p1', {}).get('upheld')) is bool and modes.get('p1', {}).get('verified') is True,
       f'P1: a complete contained receipt enters as an upheld pass (verified true, boolean verdict): {c} {fv.get("p1")}')
expect(cells.get('p1-fail', {}).get('state') == 'fail' and cells['p1-fail'].get('verified') is None, f"P1: a within-bound fail enters as fail: {cells.get('p1-fail')}")
expect(cells.get('p1-exceeded', {}).get('state') == 'untried' and (cells['p1-exceeded'].get('note') or '').startswith('[single] resource_qualification_exceeded: not accepted into the matrix (contained_resident; contained job commit peak')
       and '| exercise pass:' in cells['p1-exceeded']['note'], f"P1: exceeded stays untried with the exercise retained: {cells.get('p1-exceeded')}")
expect(cells.get('p1-unknown', {}).get('state') == 'untried' and 'resource_qualification_unknown' in (cells['p1-unknown'].get('note') or ''), f"P1: unknown stays untried: {cells.get('p1-unknown')}")
expect(cells.get('p1-artifact', {}).get('state') == 'pass' and cells['p1-artifact'].get('verified') is True and cells.get('p1-artifact-fail', {}).get('state') == 'fail',
       f"P1: artifact_floor not_contained cells keep their state: {cells.get('p1-artifact')} {cells.get('p1-artifact-fail')}")
expect(cells.get('p1-artifact-within', {}).get('state') == 'untried' and 'resource_qualification_evidence_incomplete' in (cells['p1-artifact-within'].get('note') or '') and "verdict 'within' is not not_contained" in cells['p1-artifact-within']['note'],
       f"P1: an artifact_floor block must say not_contained: {cells.get('p1-artifact-within')}")
expect(cells.get('p1-na', {}).get('state') == 'na' and 'p1-untried' not in cells, f"P1: na enters, untried is skipped as before: {cells.get('p1-na')} {'p1-untried' in cells}")
expect(cells.get('p1-refuted', {}).get('state') == 'pass' and cells['p1-refuted'].get('verified') is False and fv.get('p1-refuted', {}).get('upheld') is False, f"P1: a refuted pass stays unverified: {cells.get('p1-refuted')}")
expect(cells.get('p1-strflag', {}).get('state') == 'pass' and cells['p1-strflag'].get('verified') is False and 'verifier flag not boolean' in (cells['p1-strflag'].get('note') or ''), f"P1: a string upheld flag never verifies: {cells.get('p1-strflag')}")
expect(fam.get('kit_schema') == 2 and 'legacy_boundary_utc' not in fam, f'P1: the family records the current generation: {{k: v for k, v in fam.items() if k != "cells"}}')
counts = json.loads(proc.stdout) if proc.returncode == 0 else {}
expect(counts.get('gated_resource_qualification') == 3 and counts.get('current_runs') == ['gate-run'] and counts.get('cells') == 10, f'P1: ingest reports the gated cells and the current run: {counts}')
observations['P1'] = {'p1': (c.get('state'), c.get('verified')), 'gated': counts.get('gated_resource_qualification')}

# --- N7: the receipt must be the row's own (binding) and the identity must be the row's placement -------------------------------------
n7_rows = [row('b', 'pass', THIN_WITHIN, receipt_name='receipts/a.json'), row('n7-out', 'pass', complete_block(), receipt_name='../other/x.json'),
           row('n7-run', 'pass', complete_block()), row('n7-model', 'pass', complete_block()), row('n7-slot', 'pass', complete_block()), row('n7-mode', 'pass', complete_block()),
           row('n7-device', 'pass', complete_block()), row('n7-dtype', 'pass', complete_block()), row('n7-copies', 'pass', complete_block()), row('n7-rep-ok', 'pass', complete_block(copies=2)),
           row('n7-fail-nodtype', 'fail', complete_block()), row('a', 'pass', complete_block())]
n7_rows[7]['dtype'] = 'float16'
n7_rows[8]['mode'] = n7_rows[9]['mode'] = 'replicas'
n7_rows[10]['dtype'] = None  # the kit writes no row dtype for a fail (no dtype passed)
n7_receipts = {'a': receipt('pass', complete_block()), 'n7-run': receipt('pass', complete_block(), run_id='other-run'), 'n7-model': receipt('pass', complete_block(), model_id='zzz'),
               'n7-slot': receipt('pass', complete_block(), slot=dict(SLOT_DICT, slot='other-slot')), 'n7-mode': receipt('pass', complete_block(), mode='replicas'),
               'n7-device': receipt('pass', complete_block(), slot=dict(SLOT_DICT, device='GPU-1')), 'n7-dtype': receipt('pass', complete_block()),
               'n7-copies': receipt('pass', complete_block(), mode='replicas'), 'n7-rep-ok': receipt('pass', complete_block(copies=2), mode='replicas'), 'n7-fail-nodtype': receipt('fail', complete_block())}
proc, cells, fv, modes, _, _ = run_ingest(n7_rows, n7_receipts, outside={'x.json': receipt('pass', complete_block(), model_id='n7-out')})
expect(proc.returncode == 0, f'N7 ingest runs: rc {proc.returncode} {proc.stderr[-400:]}')
expect(cells.get('a', {}).get('state') == 'pass' and cells['a'].get('verified') is True, f"N7: cell a's own complete receipt is a pass: {cells.get('a')}")
c = cells.get('b', {})
expect(c.get('state') == 'untried' and c.get('verified') is None and "receipt model_id 'a' differs from the row model_id 'b'" in (c.get('note') or '') and 'b' not in fv,
       f"N7: a row pointing at another cell's complete receipt never enters on it: {c}")
c = cells.get('n7-out', {})
expect(c.get('state') == 'untried' and (c.get('note') or '').startswith('[single] receipt_missing_or_unparseable: ') and 'outside the run folder' in c['note'], f'N7: a receipt path outside the run folder is refused: {c}')
for mid, needle in (('n7-run', "receipt run_id 'other-run' differs from the RUN_RESULT run_id 'gate-run'"), ('n7-model', "receipt model_id 'zzz' differs"), ('n7-slot', "receipt slot 'other-slot' differs"),
                    ('n7-mode', "receipt mode 'replicas' differs from the row mode 'single'"), ('n7-device', "receipt device 'GPU-1' differs from the row device 'GPU-0'"),
                    ('n7-dtype', "dtype 'bfloat16' differs from the row dtype 'float16'")):
    c = cells.get(mid, {})
    expect(c.get('state') == 'untried' and c.get('verified') is None and needle in (c.get('note') or '') and mid not in fv, f'N7 {mid}: {needle!r} makes the cell untried: {c}')
c = modes.get('n7-copies', {})
expect(c.get('state') == 'untried' and "copies 1 is not 2 for mode 'replicas'" in (c.get('note') or ''), f'N7: a bound measured for one copy never qualifies a replicas row: {c}')
expect(modes.get('n7-rep-ok', {}).get('state') == 'pass', f"N7: two copies qualify the replicas row: {modes.get('n7-rep-ok')}")
expect(cells.get('n7-fail-nodtype', {}).get('state') == 'fail', f"N7: a fail row without a row dtype (the kit writes none) enters as fail: {cells.get('n7-fail-nodtype')}")
observations['N7'] = {m: cells.get(m, modes.get(m, {})).get('state') for m in ('a', 'b', 'n7-out', 'n7-run', 'n7-dtype', 'n7-copies', 'n7-rep-ok')}
expect(ingest.binding_problems(receipt('pass', complete_block(), model_id='m'), row('m', 'pass'), 'gate-run') == [] and len(ingest.binding_problems({}, row('m', 'pass'), 'gate-run')) == 5,
       'binding_problems: run_id, model_id, slot, mode and device')

# --- N8: the receipt's own sensors and the RUN_RESULT must agree with the verdict ------------------------------------------------------
big = complete_block(host_reserve_bytes=1, job_commit_limit_bytes=1 << 60, floors={'commit': (1 << 60) + 1, 'gpu': 8192},
                     containment={'requested_limit_bytes': 1 << 60, 'job_memory_limit_bytes': 1 << 60, 'limit_flags': 0x2200, 'child_in_job': True, 'verified_before_resume': True})
proc, cells, fv, modes, _, _ = run_ingest([row('n8-self', 'pass', big), row('n8-ok', 'pass', complete_block())], {'n8-self': receipt('pass', big), 'n8-ok': receipt('pass', complete_block())})
c = cells.get('n8-self', {})
expect(proc.returncode == 0 and c.get('state') == 'untried' and f'host_reserve_bytes 1 differs from the RUN_RESULT host_reserve_bytes {RESERVE}' in (c.get('note') or ''),
       f'N8: a self-declared reserve and cap never satisfy the gate; the RUN_RESULT reserve is the rule: {c} {proc.stderr[-200:]}')
expect(cells.get('n8-ok', {}).get('state') == 'pass', f"N8: the kit's own reserve and cap + reserve floor pass: {cells.get('n8-ok')}")
proc, cells, fv, modes, _, _ = run_ingest([row('n8-noreserve', 'pass', complete_block())], {'n8-noreserve': receipt('pass', complete_block())}, result_extra={'host_reserve_bytes': ...})
c = cells.get('n8-noreserve', {})
expect(c.get('state') == 'untried' and 'RUN_RESULT host_reserve_bytes None is not a positive int' in (c.get('note') or ''), f'N8: a RUN_RESULT without host_reserve_bytes qualifies no contained cell: {c}')
proc, cells, fv, modes, _, _ = run_ingest([row('n8-other', 'pass', complete_block())], {'n8-other': receipt('pass', complete_block())}, result_extra={'host_reserve_bytes': RESERVE + 1})
expect(cells.get('n8-other', {}).get('state') == 'untried' and 'differs from the RUN_RESULT host_reserve_bytes' in (cells.get('n8-other', {}).get('note') or ''), f"N8: reserve mismatch: {cells.get('n8-other')}")
at_cap = complete_block(samples=[sample(0, 'launch'), sample(2, 'model_load'), sample(4, 'request', peak_job_commit_bytes=CAP)])
final_over = complete_block(cleanup=dict(complete_block()['cleanup'], final_query=dict(complete_block()['cleanup']['final_query'], peak_job_commit_bytes=CAP + GIB)))
proc, cells, fv, modes, _, _ = run_ingest([row('n8-cap', 'pass', at_cap), row('n8-final', 'pass', final_over)], {'n8-cap': receipt('pass', at_cap), 'n8-final': receipt('pass', final_over)})
expect(cells.get('n8-cap', {}).get('state') == 'untried' and f'peak_job_commit_bytes {CAP} reached the cap {CAP}' in (cells.get('n8-cap', {}).get('note') or '') and 'measured peak' in cells['n8-cap']['note'],
       f"N8: a sample peak at the cap (the guard's trip point) is never within: {cells.get('n8-cap')}")
expect(cells.get('n8-final', {}).get('state') == 'untried' and 'final owned-job query peak' in (cells.get('n8-final', {}).get('note') or ''), f"N8: a final query peak at or over the cap is never within: {cells.get('n8-final')}")
observations['N8'] = {m: (cells.get(m, {}).get('note') or '')[:120] for m in ('n8-cap', 'n8-final')}

# --- N9: an artifact_floor block must carry the kit's plan reason for the receipt's spec placement and device ---------------------------
relabelled = {'kind': 'artifact_floor', 'verdict': 'not_contained', 'floors': {'commit': 1, 'gpu': 1}, 'samples': [], 'cleanup': {}}  # a music receipt with its contained block replaced
unavailable = dict(artifact, reason=ingest.ARTIFACT_REASON_UNAVAILABLE)
n9_receipts = {'n9-music': receipt('pass', relabelled, task='music', backend='ace-step', spec_placement='resident'),
               'n9-unavail': receipt('pass', unavailable, spec_placement='resident'), 'n9-wrongtext': receipt('pass', artifact, spec_placement='resident'),
               'n9-nogpu-text': receipt('pass', dict(artifact, reason=ingest.ARTIFACT_REASON_NO_GPU)), 'n9-noreason': receipt('pass', {k: v for k, v in artifact.items() if k != 'reason'}),
               'n9-ok': receipt('pass', artifact), 'n9-nokey': receipt('pass', artifact)}
del n9_receipts['n9-nokey']['spec_placement']
n9_rows = [row(m, 'pass', n9_receipts[m]['resource_qualification']) for m in n9_receipts]
proc, cells, fv, modes, _, _ = run_ingest(n9_rows, n9_receipts)
expect(proc.returncode == 0, f'N9 ingest runs: rc {proc.returncode} {proc.stderr[-400:]}')
c = cells.get('n9-music', {})
expect(c.get('state') == 'untried' and c.get('verified') is None and "is not the kit's artifact_floor plan reason for placement 'resident'" in (c.get('note') or '') and 'lacks launched true' in c['note'],
       f'N9: a resident music receipt relabelled artifact_floor with no measurement is untried: {c}')
c = cells.get('n9-unavail', {})
expect(c.get('state') == 'untried' and 'RUN_RESULT containment_available is True, not false' in (c.get('note') or ''),
       f'N9: containment_unavailable on a host whose RUN_RESULT says containment was available is untried: {c}')
for mid in ('n9-wrongtext', 'n9-nogpu-text', 'n9-noreason'):
    expect(cells.get(mid, {}).get('state') == 'untried' and "artifact_floor plan reason" in (cells.get(mid, {}).get('note') or ''), f'N9 {mid}: a reason that is not the plan reason for this placement/device is untried: {cells.get(mid)}')
expect(cells.get('n9-nokey', {}).get('state') == 'untried' and 'no spec_placement' in (cells.get('n9-nokey', {}).get('note') or ''), f"N9: a receipt without spec_placement is untried: {cells.get('n9-nokey')}")
expect(cells.get('n9-ok', {}).get('state') == 'pass' and cells['n9-ok'].get('verified') is True, f"N9: the kit's reason for a non-resident spec on a GPU slot passes: {cells.get('n9-ok')}")
proc, cells, fv, modes, _, _ = run_ingest([row('n9-unavail', 'pass', unavailable)], {'n9-unavail': receipt('pass', unavailable, spec_placement='resident')}, result_extra={'containment_available': False})
expect(cells.get('n9-unavail', {}).get('state') == 'pass', f"N9: a resident spec under the full artifact floor on a host that truly cannot contain (RUN_RESULT containment_available false) keeps its state: {cells.get('n9-unavail')}")
observations['N9'] = {m: (cells.get(m, {}).get('state')) for m in ('n9-unavail',)}
expect(ingest.validate_artifact_block(artifact, receipt('pass', artifact)) == [] and ingest.validate_artifact_block(relabelled, receipt('pass', relabelled, spec_placement='resident')), 'pure artifact validator')
for placement, gpu in ((None, True), ('auto', True), ('auto', False), ('resident', False), ('resident', True)):
    expect(run_cells.artifact_floor_reason(placement, gpu) == ingest.artifact_floor_reason(placement, gpu), f'run_cells and ingest agree on the artifact_floor reason for {placement!r} gpu={gpu}')
music_entry = {'model_id': 'ace', 'task': 'music', 'backend': 'ace-step', 'path': 'p', 'spec': {'placement': 'resident'}}
plan = run_cells.contained_admission_plan(music_entry, 'GPU-x', 'single', 'bfloat16', 1, 1, {}, available=lambda: False)
expect(plan == {'kind': 'artifact_floor', 'reason': ingest.ARTIFACT_REASON_UNAVAILABLE}, f'the kit writes the containment_unavailable reason only when the host cannot contain: {plan}')
expect(run_cells.contained_admission_plan(dict(music_entry, spec={'placement': 'auto'}), 'GPU-x', 'single', 'auto', 1, 1, {})['reason'] == ingest.artifact_floor_reason('auto', True)
       and run_cells.contained_admission_plan(music_entry, None, 'single', 'auto', 1, 1, {})['reason'] == ingest.ARTIFACT_REASON_NO_GPU, 'the kit writes the not-resident and no-GPU reasons')
built_art = run_cells.cell_receipt('r', music_entry, dict(SLOT_DICT), 'single', None, {}, [{'state': 'pass', 'exercise': {'state': 'pass'}, 'resource_qualification': artifact}], 'pass', 'n')
expect(built_art['spec_placement'] == 'resident' and built_art['run_id'] == 'r' and built_art['model_id'] == 'ace' and built_art['slot'] == SLOT_DICT and built_art['mode'] == 'single',
       f'cell_receipt writes spec_placement and the binding fields: {({k: built_art.get(k) for k in ("spec_placement", "run_id", "model_id", "mode")})}')
src_rc_row = (KIT / 'run_cells.py').read_text(encoding='utf-8')
expect("'spec_placement': receipt['spec_placement']" in src_rc_row, 'the RUN_RESULT row carries spec_placement too (display)')

# --- N10: a contained cell that failed before the request phase enters as fail with the measured prefix; a pass needs the whole run ----
fail_block = complete_block(samples=[sample(0, 'launch'), sample(2, 'server_start'), sample(4, 'model_load')])
fail_receipt = receipt('fail', fail_block)
n10_rows = [row('n10-fail', 'fail', fail_block), row('n10-pass', 'pass', fail_block), row('n10-noerror', 'fail', fail_block), row('n10-order', 'fail', fail_block)]
n10_rows[0]['dtype'] = n10_rows[2]['dtype'] = n10_rows[3]['dtype'] = None
out_of_order = complete_block(samples=[sample(0, 'launch'), sample(2, 'model_load'), sample(4, 'server_start')])
proc, cells, fv, modes, _, _ = run_ingest(n10_rows, {'n10-fail': fail_receipt, 'n10-pass': receipt('pass', fail_block), 'n10-noerror': receipt('fail', fail_block, exercise={'state': 'fail'}),
                                                     'n10-order': receipt('fail', out_of_order)})
expect(proc.returncode == 0, f'N10 ingest runs: rc {proc.returncode} {proc.stderr[-400:]}')
c = cells.get('n10-fail', {})
expect(c.get('state') == 'fail' and c.get('verified') is None and (c.get('note') or '') == '[single] fail note', f'N10: a measured within-bound fail at model load enters as fail, not untried: {c}')
c = cells.get('n10-pass', {})
expect(c.get('state') == 'untried' and 'samples lack the phases request' in (c.get('note') or ''), f'N10: the same prefix on a pass row is not a measured pass: {c}')
expect(cells.get('n10-noerror', {}).get('state') == 'untried' and 'exercise fail carries no error text' in (cells.get('n10-noerror', {}).get('note') or ''), f"N10: a fail without its exercise error is untried: {cells.get('n10-noerror')}")
expect(cells.get('n10-order', {}).get('state') == 'untried' and 'out of the kit order' in (cells.get('n10-order', {}).get('note') or ''), f"N10: phases out of the kit order are untried: {cells.get('n10-order')}")
observations['N10'] = {m: cells.get(m, {}).get('state') for m in ('n10-fail', 'n10-pass', 'n10-noerror', 'n10-order')}
expect(ingest.validate_contained_block(fail_block, fail_receipt) == [] and ingest.validate_contained_block(fail_block, receipt('pass', fail_block)) == ['samples lack the phases request'],
       f'pure validator: the fail prefix is complete for a fail and incomplete for a pass: {ingest.validate_contained_block(fail_block, fail_receipt)}')

# --- profile_key shape: production regex is the real sha256 hex digest, in ingest and run_cells alike ---------------------------------
for checker, where in ((ingest.missing_identity, 'ingest'), (run_cells.missing_identity, 'run_cells')):
    expect(checker(dict(IDENTITY)) == [], f'{where}.missing_identity accepts a real sha256 profile_key: {checker(dict(IDENTITY))}')
    for bad in ('k' * 64, KEY.upper(), KEY[:63], KEY + 'a', 'g' * 64, None, 64):
        expect(checker(dict(IDENTITY, profile_key=bad)) == ['profile_key'], f'{where}.missing_identity rejects profile_key {str(bad)[:20]!r}...')
expect(ingest.HEX64.pattern == '[0-9a-f]{64}', f'ingest HEX64 is exactly the lowercase 64-hex sha256 shape: {ingest.HEX64.pattern}')
src_rc = (KIT / 'run_cells.py').read_text(encoding='utf-8')
expect("re.fullmatch(r'[0-9a-f]{64}', key)" in src_rc and '[0-9A-Za-z]{64}' not in src_rc and '[0-9A-Za-z]{64}' not in (KIT / 'ingest.py').read_text(encoding='utf-8'),
       'the alphanumeric profile_key shape is gone from production')
# the kit stamps its schema: receipts and RUN_RESULT
expect(run_cells.RECEIPT_SCHEMA == ingest.CURRENT_KIT_SCHEMA == 2 and "'kit_schema': RECEIPT_SCHEMA" in src_rc and "'receipt_schema': RECEIPT_SCHEMA" in src_rc, 'run_cells stamps kit_schema and receipt_schema 2')
built = run_cells.cell_receipt('r', {'model_id': 'm', 'task': 'music', 'backend': 'ace-step', 'path': 'p'}, {'slot': SLOT}, 'single', None, {'hostname': 'h'},
                               [{'state': 'pass', 'exercise': {'state': 'pass', 'steps': [1], 'output': {'wav_bytes': 1}}, 'resource_qualification': complete_block()}], 'pass', 'note')
expect(built['receipt_schema'] == 2 and built['resource_qualification'] == complete_block() and built['exercise'] == {'state': 'pass', 'output': {'wav_bytes': 1}} and built['state'] == 'pass',
       'cell_receipt writes the complete block (not a summary) beside the exercise, with the schema marker')
expect(ingest.validate_contained_block(built['resource_qualification'], built) == [], 'a receipt built by the kit from a complete block passes the validator')
expect(run_cells.cell_receipt('r', {'model_id': 'm', 'task': 't', 'backend': 'b', 'path': 'p'}, {}, 'single', None, {}, [{'dtype': 'auto', 'state': 'fail', 'error': 'config_error: x'}], 'fail', 'n')['resource_qualification'] is None,
       'a config refusal before any launch has no block (ingest then keeps such a current cell untried receipt_schema_missing)')

if failures:
    print(json.dumps({'passed': False, 'failures': failures, 'observations': observations}, indent=1, default=str))
    sys.exit(1)
print(json.dumps({'passed': True, 'observations': observations}, default=str))
