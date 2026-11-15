"""Blocker 6 (ingest gate and verdict typing) proof for tools/matrix_kit/ingest.py.

1. The verdict 'upheld' flag counts only when it is exactly boolean True: a string 'false', 'true', 1 or any non-bool must never
   make a kit pass `verified: true`.
2. A contained cell (resource_qualification.kind == 'contained_resident') whose verdict is 'within' but whose placement identity
   (profile key fields: receipt sha, adapter sha, placement, dtype, copies) is missing enters as untried with the reason, the exercise
   block staying on the receipt as evidence; a complete identity keeps the pass.
3. Receipts without the block (older kits) enter exactly as before - for LEGACY runs (no kit_schema, utc before
   ingest.LEGACY_RESULT_BOUNDARY_UTC, and the file hash pinned in tools/matrix_kit/legacy_results.json or --legacy-run naming the run).
   A contained_resident block is never legacy evidence (untried legacy_contained even with a complete identity). In a CURRENT run
   (kit_schema 2) the identity alone is no evidence: the same within block without cap/readback/peak/samples/cleanup, a receipt without
   the block, and a missing receipt all enter untried (the full typed gate is tests/check_ingest_gate_negative.py).

Run from the repo root: PYTHONPATH=src python tests/check_daybreak_blocker6.py (exit 0 = pass).
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INGEST = ROOT / 'tools' / 'matrix_kit' / 'ingest.py'
SLOT = 'win-x64-nvidia-cuda'
failures = []


def expect(cond, msg):
    if not cond:
        failures.append(msg)


def ingest(rows, receipts, verdicts=None, kit_schema=None, utc='2026-09-25T05:00:00+00:00', legacy_flag=True):
    """Drive ingest.py on a synthetic run. The default is a LEGACY result (no kit_schema, utc before the boundary, named with --legacy-run
    because a synthetic file is not in the pinned legacy population); kit_schema=2 makes it current; legacy_flag=False drops the flag."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / 'raw.json'
        raw.write_text(json.dumps({'catalog': {}, 'families': [], 'mode_cells': []}), encoding='utf-8')
        d = Path(tmp) / 'kit'
        (d / 'receipts').mkdir(parents=True)
        for mid, body in receipts.items():
            body = dict(body)
            body.setdefault('model_id', mid)  # the binding fields run_cells.cell_receipt writes; a current run requires them to equal the row
            (d / 'receipts' / f'{mid}.json').write_text(json.dumps(body), encoding='utf-8')
        result = {'run_id': 'b6-run', 'utc': utc, 'cells': rows, 'host_reserve_bytes': 6 << 30, 'containment_available': True}
        if kit_schema is not None:
            result['kit_schema'] = kit_schema
        (d / 'RUN_RESULT.json').write_text(json.dumps(result), encoding='utf-8')
        cmd = [sys.executable, str(INGEST), '--raw', str(raw), '--evidence', 'E-000', '--result', str(d / 'RUN_RESULT.json')]
        if kit_schema is None and legacy_flag:
            cmd += ['--legacy-run', 'b6-run']
        if verdicts is not None:
            vp = Path(tmp) / 'verdicts.json'
            vp.write_text(json.dumps(verdicts), encoding='utf-8')
            cmd += ['--verdicts', str(vp)]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        data = json.loads(raw.read_text(encoding='utf-8')) if out.returncode == 0 else {'mode_cells': [], 'families': []}
        fam = next((f for f in data['families'] if f['family'] == 'matrix-kit:b6-run'), {'cells': [], 'verdicts': []})
        return out, {c['model_id']: c for c in fam['cells']}, fam.get('verdicts', []), {c['model_id']: c for c in data['mode_cells']}


def row(mid, state, rq=None):
    r = {'model_id': mid, 'slot': SLOT, 'mode': 'single', 'state': state, 'device': 'GPU-0', 'device_named': 'x', 'receipt': f'receipts/{mid}.json', 'note': f'{state} note'}
    if rq:
        r.update(resource_kind=rq.get('kind'), resource_verdict=rq.get('verdict'))
    return r


def receipt(state, rq=None, schema=None):
    body = {'state': state, 'exercise': {'state': state, 'output': {'wav_bytes': 241732}}, 'run_id': 'b6-run', 'slot': {'slot': SLOT, 'device': 'GPU-0'}, 'mode': 'single',
            'spec_placement': 'resident' if (rq or {}).get('kind') == 'contained_resident' else None}
    if schema is not None:
        body['receipt_schema'] = schema
    if rq is not None:
        body['resource_qualification'] = rq
    return body


# the artifact_floor block the kit writes for a non-resident spec on a GPU slot (ingest requires the kit's plan reason, floors and the launched tree)
ARTIFACT = {'kind': 'artifact_floor', 'verdict': 'not_contained', 'reason': "placement None is not resident; auto/None never lowers the floor (auto may fall back to CPU offload)",
            'floors': {'commit': 4 << 30, 'gpu': 4096}, 'samples': [], 'cleanup': {'launched': True, 'active_processes_after': None, 'job_handle_closed': None, 'forced_cleanup': False}}


# 1. upheld typing: only boolean True upholds
bad_flags = {'m-str-false': 'false', 'm-str-true': 'true', 'm-int-1': 1, 'm-str-yes': 'yes', 'm-list': [True]}
rows = [row(m, 'pass') for m in list(bad_flags) + ['m-bool-true', 'm-bool-false']]
receipts = {r['model_id']: receipt('pass') for r in rows}
verdicts = {f'{m}|{SLOT}|single': {'upheld': flag, 'corrected_state': 'partial', 'reasoning': 'typed flag'} for m, flag in bad_flags.items()}
verdicts[f'm-bool-true|{SLOT}|single'] = {'upheld': True, 'corrected_state': 'pass', 'reasoning': 'ok'}
verdicts[f'm-bool-false|{SLOT}|single'] = {'upheld': False, 'corrected_state': 'partial', 'reasoning': 'refuted'}
out, cells, fam_verdicts, modes = ingest(rows, receipts, verdicts)
expect(out.returncode == 0, f'ingest with typed flags must succeed: rc {out.returncode} {out.stderr[-400:]}')
for m in bad_flags:
    expect(cells.get(m, {}).get('verified') is False, f"{m}: upheld={bad_flags[m]!r} must not verify the pass (family cell verified={cells.get(m, {}).get('verified')!r})")
    expect(modes.get(m, {}).get('verified') is False, f"{m}: upheld={bad_flags[m]!r} must not verify the mode cell (verified={modes.get(m, {}).get('verified')!r})")
fv = {v['model_id']: v for v in fam_verdicts}
for m in bad_flags:
    expect(fv.get(m, {}).get('upheld') is False, f"{m}: the family verdict upheld flag must be boolean False for {bad_flags[m]!r} (got {fv.get(m, {}).get('upheld')!r})")
expect(cells.get('m-bool-true', {}).get('verified') is True and fv.get('m-bool-true', {}).get('upheld') is True, 'a boolean True upholds')
expect(cells.get('m-bool-false', {}).get('verified') is False and fv.get('m-bool-false', {}).get('upheld') is False, 'a boolean False refutes')

# 2. contained cell within-bound but missing its placement identity: untried with the reason, exercise kept as evidence
KEY = hashlib.sha256(b'blocker6-profile').hexdigest()  # a real sha256 hex digest: production accepts exactly this shape ([0-9a-f]{64})
identity = {'receipt_sha256': hashlib.sha256(b'music-runtime.json').hexdigest(), 'adapter_sha256': hashlib.sha256(b'adapter').hexdigest(), 'placement': 'resident', 'dtype': 'bfloat16', 'copies': 1}
full = {'kind': 'contained_resident', 'bound': 'job_commit', 'verdict': 'within', 'profile_key': KEY, 'measured_peak_job_commit_bytes': 10, 'job_commit_limit_bytes': 20, 'limit_hit': False, **identity}
missing_cases = {'m-no-identity': {k: v for k, v in full.items() if k not in identity},
                 'm-no-key': {k: v for k, v in full.items() if k != 'profile_key'}}
for field in identity:
    missing_cases[f'm-no-{field}'] = {k: v for k, v in full.items() if k != field}
rows = [row(m, 'pass', rq) for m, rq in missing_cases.items()] + [row('m-full', 'pass', full), row('m-old', 'pass'), row('m-artifact', 'pass', ARTIFACT)]
receipts = {m: receipt('pass', rq) for m, rq in missing_cases.items()}
receipts.update({'m-full': receipt('pass', full), 'm-old': receipt('pass'), 'm-artifact': receipt('pass', ARTIFACT)})
verdicts = {f'{m}|{SLOT}|single': {'upheld': True, 'corrected_state': 'pass', 'reasoning': 'valid WAV'} for m in receipts}
out, cells, fam_verdicts, modes = ingest(rows, receipts, verdicts)
expect(out.returncode == 0, f'ingest with identity cases must succeed: rc {out.returncode} {out.stderr[-400:]}')
for m in missing_cases:
    c = cells.get(m, {})
    expect(c.get('state') == 'untried' and c.get('verified') is None,
           f"{m}: a contained within-bound cell missing its placement identity must enter untried (got state={c.get('state')!r} verified={c.get('verified')!r})")
    expect('exercise pass' in (c.get('note') or '') and 'identity missing' in (c.get('note') or ''), f"{m}: the exercise stays as evidence and the missing identity is named: {c.get('note')!r}")
    expect(modes.get(m, {}).get('state') == 'untried', f"{m}: the mode cell is untried too (got {modes.get(m, {}).get('state')!r})")
# a contained_resident block is never legacy evidence: the legacy boundary covers old full-artifact receipts only, so even the complete
# identity (which the pre-boundary ingest accepted alone, without cap/readback/peak/samples/cleanup) enters untried in a legacy run
expect(cells.get('m-full', {}).get('state') == 'untried' and cells['m-full'].get('verified') is None and (cells['m-full'].get('note') or '').startswith('[single] resource_qualification_legacy_contained: ')
       and 'identity missing' not in cells['m-full']['note'], f"a legacy contained within block with its complete identity is untried, not pass: {cells.get('m-full')}")
expect(cells.get('m-old', {}).get('state') == 'pass' and cells['m-old'].get('verified') is True, f"an older receipt without the block is untouched: {cells.get('m-old')}")
expect(cells.get('m-artifact', {}).get('state') == 'pass' and cells['m-artifact'].get('verified') is True, f"an artifact-floor cell needs no placement identity: {cells.get('m-artifact')}")
for bad_key in ('k' * 64, KEY.upper()):
    out, cells, fam_verdicts, modes = ingest([row('m-badkey', 'pass', dict(full, profile_key=bad_key))], {'m-badkey': receipt('pass', dict(full, profile_key=bad_key))}, verdicts)
    expect(cells.get('m-badkey', {}).get('state') == 'untried' and 'missing profile_key' in (cells.get('m-badkey', {}).get('note') or ''),
           f"a profile_key that is not a lowercase 64-hex sha256 ({bad_key[:8]}...) is missing identity: {cells.get('m-badkey')}")
# the same legacy-shaped result without --legacy-run: its sha256 is not in the pinned legacy population, so it is refused (a self-reported
# earlier utc alone never opens the legacy path)
out, cells, fam_verdicts, modes = ingest([row('m-old', 'pass')], {'m-old': receipt('pass')}, verdicts, legacy_flag=False)
expect(out.returncode != 0 and 'not in the pinned legacy population' in out.stderr and not cells, f'an unmarked back-dated result whose hash is not pinned is refused: rc {out.returncode} {out.stderr[-200:]}')

# 3. the same cells in a CURRENT run (kit_schema 2): identity alone, a block-less receipt or a missing receipt is never pass evidence
rows = [row('c-full', 'pass', full), row('c-old', 'pass'), row('c-missing', 'pass', full), row('c-artifact', 'pass', ARTIFACT)]
receipts = {'c-full': receipt('pass', full, schema=2), 'c-old': receipt('pass', schema=2), 'c-artifact': receipt('pass', ARTIFACT, schema=2)}
verdicts = {f'{m}|{SLOT}|single': {'upheld': True, 'corrected_state': 'pass', 'reasoning': 'valid WAV'} for m in ('c-full', 'c-old', 'c-missing', 'c-artifact')}
out, cells, fam_verdicts, modes = ingest(rows, receipts, verdicts, kit_schema=2, utc='2026-09-25T19:30:00+00:00')
expect(out.returncode == 0, f'current-run ingest must succeed: rc {out.returncode} {out.stderr[-400:]}')
c = cells.get('c-full', {})
expect(c.get('state') == 'untried' and c.get('verified') is None and (c.get('note') or '').startswith('[single] resource_qualification_evidence_incomplete: ') and 'exercise pass' in (c.get('note') or ''),
       f"current run: a within block with a complete identity but no cap/readback/peak/samples/cleanup is untried, exercise retained: {c}")
expect(cells.get('c-old', {}).get('state') == 'untried' and (cells.get('c-old', {}).get('note') or '').startswith('[single] receipt_schema_missing: '), f"current run: a receipt without the block is untried: {cells.get('c-old')}")
expect(cells.get('c-missing', {}).get('state') == 'untried' and (cells.get('c-missing', {}).get('note') or '').startswith('[single] receipt_missing_or_unparseable: '), f"current run: a missing receipt never falls back to the row: {cells.get('c-missing')}")
expect(cells.get('c-artifact', {}).get('state') == 'pass' and cells['c-artifact'].get('verified') is True, f"current run: an artifact_floor not_contained receipt keeps the upheld pass: {cells.get('c-artifact')}")
expect('c-full' not in fam_verdicts and 'c-old' not in fam_verdicts and 'c-missing' not in fam_verdicts, f'no verdict is bound to a gated cell: {sorted(fam_verdicts)}')
# and without the marker a post-boundary result is refused outright
out, cells, fam_verdicts, modes = ingest(rows, receipts, verdicts, kit_schema=None, utc='2026-09-25T19:30:00+00:00', legacy_flag=False)
expect(out.returncode != 0 and 'ingest refused' in out.stderr and not cells, f'a post-boundary result without kit_schema is refused: rc {out.returncode} {out.stderr[-200:]}')
out, cells, fam_verdicts, modes = ingest(rows, receipts, verdicts, kit_schema=None, utc='2026-09-25T19:30:00+00:00', legacy_flag=True)
expect(out.returncode == 0 and cells.get('c-old', {}).get('state') == 'pass' and cells.get('c-full', {}).get('state') == 'untried', f'--legacy-run binds it explicitly; the contained block stays untried: rc {out.returncode} {cells.get("c-full")}')

if failures:
    print('\n'.join(failures))
    sys.exit(1)
print(json.dumps({'passed': True, 'checks': 'upheld flag exactly boolean True; contained within-bound cell without placement identity is untried; legacy receipts unchanged; current runs need the detailed receipt'}))
