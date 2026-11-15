"""Render docs/execution_matrix.json into EXECUTION_MATRIX.md (one table per modality family, symbols per slot, evidence footnotes),
execution_matrix.html (colour grid with per-cell titles) and execution_matrix.csv (long format). Run with --check to fail on drift."""
import argparse
import csv
import html
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'docs' / 'execution_matrix.json'
MD = ROOT / 'docs' / 'EXECUTION_MATRIX.md'
KIT_ADMISSION = ("Matrix-kit admission and resource qualification (tools/matrix_kit/run_cells.py, containment.py, ingest.py). The kit admits a cell only when three consecutive samples clear its floors. The default floor is the artifact floor (max(global floor, 1.5 x artifact bytes x copies + margin) of host commit; artifact bytes x copies + margin of GPU memory) and no manifest field lowers it. The only lower host floor is *contained admission*: the row's spec names placement `resident` explicitly (`auto` may fall back to CPU offload and is never admitted under a lowered floor), the slot has a GPU, and the host can cap the owned process tree with a Windows Job Object; the Titanos server is then created suspended inside a kill-on-close job, the job commit cap is set and read back before the child resumes (a readback mismatch, or the OS refusing the job while the child is suspended, leaves the cell untried `containment_unavailable`, never an uncontained run), a shared-host reserve (--host-reserve-gib, positive, validated at the CLI with --floor-margin-gib) stays free, and the cap is 1.5 x a measured within-bound peak + margin (a profile entry is used only when it is bound to an existing pass receipt with verdict within and the same key, peak and cap; the receipt records the provenance) or, on a first run, what the host can spare minus the reserve. The bound is commit-only (`resource_qualification.bound` = `job_commit`): JOB_OBJECT_LIMIT_JOB_MEMORY caps and measures the tree's commit charge, and it is a fail-closed bound only for the allocation classes containment.CAP_SEMANTICS lists as enforced (measured 2026-09-25 with synthetic children: heap/VirtualAlloc commit, CUDA device allocations under WDDM, pinned host memory, safetensors safe_open, and FILE_MAP_COPY / mmap ACCESS_COPY views, each refused at the cap); two gaps remain: copy-on-write mappings made by torch.UntypedStorage.from_file(shared=False) are accounted in PeakJobMemoryUsed but not refused at map time and fail at page touch (the same exit code run r3 showed while a DiT shard loaded, 0xC0000005; which class r3 hit is not proven), and anonymous pagefile-backed sections are neither accounted nor refused. PeakJobMemoryUsed also includes attempted charges that were refused, so a peak over the cap is evidence of an attempted charge, not proof of committed memory over the cap. Every contained receipt carries that table as `resource_qualification.cap_semantics`; the guard remains the active monitor, available physical memory is sampled beside commit at admission and by the guard (a contained cell trips when it drops under the reserve), the tree's peak working set is recorded uncapped, and the owned child must avoid the gap classes (the adapter's streamed loader). Before a contained cell launches, a WDDM-aware pre-launch check reads the GPU driver model (nvidia-smi driver_model.current, recorded as `driver_model`) and sizes the entry's safetensors headers at the resident dtype (numel x itemsize; peak set = max(planner, dit) + text encoder + vae for the music bundle): on WDDM, where device allocations are charged to the process commit, the cap must be at least 3 GiB base (torch import + CUDA context) + peak set + 4 GiB activation headroom, else the cell is untried `blocked_by_host_resources: contained budget below the device-resident need on WDDM (cap X < base + peak set + headroom Y)` with the numbers in `device_resident_estimate` and nothing launched; on TCC the estimate is recorded only; an unreadable driver model or unreadable headers fail closed. Admission fails closed: a missing required commit, physical-memory (contained) or GPU reading is never treated as clear, three missing readings leave the cell untried (resource_reading_unavailable), STOP is checked once more immediately before success and once more immediately before the launch (a STOP there creates no process). While the cell runs, a guard samples host commit, physical memory, GPU memory and the job peak at least every 2 s during server start, model load, every active request and the unload (the request catalogue polls it in its wait loops); STOP, a missing reading, a cap hit, host commit or physical memory below the reserve, or (contained cells only) GPU memory below half the margin trips it, only owned work is cancelled (the Titanos job through DELETE /api/jobs, then the owned tree), and the partial receipt is kept as untried with its samples and cleanup evidence, including host commit, physical and GPU readings after the tree ended. Every receipt separates `exercise` (output correctness) from `resource_qualification` (kind, bound, verdict within/exceeded/unknown/not_contained, cap, measured peaks, samples, cleanup): a cell is a pass only when the exercise passed and the verdict is within or not_contained; exceeded or unknown makes it untried even when a valid output (a WAV) was written, and a cap hit is never a pass plus a note. Every receipt carries `receipt_schema` 2 with that complete block (not a summary) and RUN_RESULT carries `kit_schema` 2. ingest.py reads a current run (kit_schema >= 2) through its detailed receipts alone: a pass/fail cell whose receipt is missing, unparseable, without receipt_schema >= 2 or without the block is untried (receipt_missing_or_unparseable / receipt_schema_missing; no fallback to the RUN_RESULT row, whose resource_* fields are display only), and a contained_resident block is accepted only when its typed validation finds nothing wrong: verdict within under bound job_commit, positive integer cap and reserve, integer floors, the cap readback equal to the cap with JOB_MEMORY | KILL_ON_JOB_CLOSE set and breakaway clear, child in job, verified before resume, a measured job peak strictly under the cap, limit_hit false, readings_complete true, no trip, a non-empty sample sequence with non-decreasing timestamps covering model_load and request in which host commit and physical memory never dipped under the reserve and the job peak never exceeded the cap, a complete placement identity (profile_key, receipt_sha256 and adapter_sha256 as 64-character lowercase sha256 hex, placement resident, dtype, integer copies >= 1), cleanup evidence (launched, active_processes_after 0, job handle closed, a final query without error) and an exercise block equal to the row state; an artifact_floor block must say not_contained. Any problem makes the cell untried resource_qualification_evidence_incomplete with the functional output retained separately in the note; exceeded and unknown stay untried as before. Old full-artifact receipts are preserved only through the explicit legacy boundary: a RUN_RESULT without kit_schema is legacy when its utc is before 2026-09-25T19:00:00+00:00 (ingest.LEGACY_RESULT_BOUNDARY_UTC) and is then bound exactly as before (the family records kit schema 1); one without the marker at/after the boundary is refused unless --legacy-run names it. A verifier's upheld flag counts only when it is exactly boolean true (ingest and the builder alike).")
HTML = ROOT / 'docs' / 'execution_matrix.html'
CSV = ROOT / 'docs' / 'execution_matrix.csv'
COLORS = {'pass': '#1f8a4c', 'fail': '#b3261e', 'partial': '#d9a300', 'untried': '#e6e6e6', 'na': '#9aa0a6'}


def cell_map(data):
    return {(c['model_id'], c['slot']): c for c in data['cells']}


def render_md(data):
    sym = {k: v['symbol'] for k, v in data['cell_states'].items()}
    slots = data['slots']
    L = ['# Execution matrix: every model x every platform slot', '',
         f"Generated from `docs/execution_matrix.json` (built {data['built_utc']}) by `tools/render_execution_matrix.py`; do not edit by hand. Spec: `{data['spec']}`.", '',
         'One row per individual model (identity, task and backend are separate columns). One column per platform slot (OS, architecture, accelerator). One state per cell:', '']
    L += [f"- {v['symbol']} `{k}`: {v['meaning']}" for k, v in data['cell_states'].items()]
    c = data['counts']
    L += ['', f"Counts: {c['rows']} models x {c['slots']} slots = {c['cells']} cells: {c['pass']} pass ({c['verified_pass']} verifier-upheld), {c['fail']} fail, {c['partial']} partial, {c['untried']} untried, {c['na']} not applicable.", '',
          '## Platform slots (columns)', '', '| # | Slot id | OS | Architecture | Accelerator |', '|---|---|---|---|---|']
    for i, s in enumerate(slots, 1):
        L.append(f"| {i} | `{s['id']}` | {s['os']} | {s['arch']} | {s['accelerator']} |")
    notes = [(s['id'], n) for s in slots for n in (s.get('notes') or [])]
    if notes:
        L += ['', 'Slot notes (dated changes to a slot definition):', ''] + [f"- `{sid}`: {n}" for sid, n in notes]
    cm = cell_map(data)
    families = sorted({r['modality_family'] for r in data['rows']})
    head = '| Model | Identity | Task | Backend | ' + ' | '.join(str(i) for i in range(1, len(slots) + 1)) + ' |'
    sep = '|---|---|---|---|' + '|'.join(['---'] * len(slots)) + '|'
    for fam in families:
        L += ['', f'## {fam}', '', head, sep]
        for r in [r for r in data['rows'] if r['modality_family'] == fam]:
            marks = []
            for s in slots:
                cell = cm[(r['model_id'], s['id'])]
                marks.append(sym[cell['state']])
            L.append(f"| {r['display_name']} | `{r['identity']}` | {r['task']} | {r['backend']} | " + ' | '.join(marks) + ' |')
        L += ['', 'Evidence per executed cell (slot number: state, E-ids, device, note):', '']
        for r in [r for r in data['rows'] if r['modality_family'] == fam]:
            ex = [(i, cm[(r['model_id'], s['id'])]) for i, s in enumerate(slots, 1) if cm[(r['model_id'], s['id'])]['state'] in ('pass', 'fail', 'partial')]
            if ex:
                L.append(f"- **{r['display_name']}**: " + '; '.join(f"{i}: {cell['state']} [{', '.join(cell['evidence_ids'])}] {cell.get('device_named') or ''} {('- ' + cell['note'][:160]) if cell.get('note') else ''}".strip() for i, cell in ex))
    L += ['', '## Named devices per slot (proven, present but untried, missing)', '', '| Slot | Device | Generation | Status | Evidence | Note |', '|---|---|---|---|---|---|']
    for dv in data.get('named_devices', []):
        L.append(f"| `{dv['slot']}` | {dv['device']} | {dv['generation']} | {dv['status']} | {', '.join(dv['evidence_ids']) or '-'} | {dv['note']} |")
    L += ['', '## Execution modes', '', '| Mode | State | Slots with evidence | Plan items | Note |', '|---|---|---|---|---|']
    for m in data['execution_modes']:
        L.append(f"| {m['mode']} | {sym[m['state']]} {m['state']} | {', '.join(m['slots'])} | {', '.join(m['plan_items'])} | {m['note']} |")
    mc = data.get('mode_cells') or []
    if mc:
        L += ['', '## Execution modes per model (model x slot pairs with mode receipts)', '', 'Each column is one mode run by the matrix kit on that slot: single, two replicas, model sharding over a homogeneous GPU group, tensor parallel over that group. `na` means Titanos refused the mode for that model by design (recorded from its refusal text, or by construction from its own admission rules in src/titanos/config.py, or shared from the single/replicas refusal of the same pair), or the slot can neither form the GPU group the mode needs nor declare the explicit CPU/XPU group through a runtime available for the row backend, by construction; `untried` means no receipt for that mode on a slot where Titanos admits it and the slot can form or declare the group (a GPU slot without a group in the run, or the explicit CPU/XPU tensor-parallel group the kit does not run).', '',
              KIT_ADMISSION, '',
              '| Model | Slot | single | replicas | model_sharding | tensor_parallel | Evidence |', '|---|---|---|---|---|---|---|']
        by = {}
        for c in mc:
            by.setdefault((c['model_id'], c['slot']), {})[c['mode']] = c
        names = {r['model_id']: r['display_name'] for r in data['rows']}
        for (mid, slot), modes in sorted(by.items()):
            ev = sorted({e for c in modes.values() for e in c['evidence_ids']})
            marks = ' | '.join(f"{sym[modes[m]['state']]} {modes[m]['state']}" if m in modes else f"{sym['untried']} untried" for m in ('single', 'replicas', 'model_sharding', 'tensor_parallel'))
            L.append(f"| {names.get(mid, mid)} | `{slot}` | {marks} | {', '.join(ev) or '-'} |")
    L += ['', '## Reading this matrix', '', 'A pass is one named model on one named slot with a receipt; it is not a claim about other models, other devices of the same family or other workloads. Untried cells are the build-toward list: runnable by construction and not yet executed (the artifact present or still to be acquired; the note names the runtime tokens that grant the path), runnable by construction on a slot no fleet host covers yet (a host circumstance, never a construction fact), or executed by a reader without matrix-cited evidence. Not-applicable cells are refusals Titanos recorded by design or impossibilities by construction, not product decisions: the row backend has no execution path on the slot runtimes (a converted artifact is its own row), a runtime it needs publishes no package for the slot platform under the pyproject pins, or the row identity names no acquirable artifact (recorded executions stand); each carries its reason in the note.', '']
    return '\n'.join(L)


def render_html(data):
    slots = data['slots']
    cm = cell_map(data)
    sym = {k: v['symbol'] for k, v in data['cell_states'].items()}
    parts = ['<!doctype html><meta charset="utf-8"><title>Titanos execution matrix</title>',
             '<style>body{font:13px system-ui,Segoe UI,Arial;margin:16px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:3px 5px;white-space:nowrap}th.rot{height:150px;vertical-align:bottom}th.rot div{writing-mode:vertical-rl;transform:rotate(180deg);font-weight:500}td.c{text-align:center;width:26px}.legend span{display:inline-block;padding:2px 8px;margin-right:8px;border:1px solid #999}h2{margin-top:28px}small{color:#555}</style>',
             f"<h1>Titanos execution matrix</h1><p>Built {html.escape(data['built_utc'])}. {html.escape(data['counts']['rows'].__str__())} models x {len(slots)} slots. Hover a cell for evidence and notes.</p><p class=legend>"]
    for k, v in data['cell_states'].items():
        parts.append(f"<span style='background:{COLORS[k]}'>{v['symbol']} {k}</span>")
    parts.append('</p>')
    families = sorted({r['modality_family'] for r in data['rows']})
    for fam in families:
        parts.append(f'<h2>{html.escape(fam)}</h2><table><tr><th>Model</th><th>Identity</th><th>Task</th><th>Backend</th>')
        for s in slots:
            parts.append(f"<th class=rot title='{html.escape(s['id'])}'><div>{html.escape(s['os'])} · {html.escape(s['arch'])} · {html.escape(s['accelerator'])}</div></th>")
        parts.append('</tr>')
        for r in [r for r in data['rows'] if r['modality_family'] == fam]:
            parts.append(f"<tr><td><b>{html.escape(r['display_name'])}</b></td><td><small>{html.escape(r['identity'])}</small></td><td>{html.escape(r['task'])}</td><td>{html.escape(r['backend'])}</td>")
            for s in slots:
                cell = cm[(r['model_id'], s['id'])]
                title = f"{cell['state']} | {', '.join(cell['evidence_ids'])} | {cell.get('device_named') or ''} | {cell.get('package_or_runtime') or ''} | {cell.get('note') or ''}"
                parts.append(f"<td class=c style='background:{COLORS[cell['state']]}' title='{html.escape(title)}'>{sym[cell['state']]}</td>")
            parts.append('</tr>')
        parts.append('</table>')
    parts.append('<h2>Named devices per slot</h2><table><tr><th>Slot</th><th>Device</th><th>Generation</th><th>Status</th><th>Evidence</th><th>Note</th></tr>')
    for dv in data.get('named_devices', []):
        parts.append(f"<tr><td>{html.escape(dv['slot'])}</td><td>{html.escape(dv['device'])}</td><td>{html.escape(dv['generation'])}</td><td>{html.escape(dv['status'])}</td><td>{html.escape(', '.join(dv['evidence_ids']))}</td><td>{html.escape(dv['note'])}</td></tr>")
    parts.append('</table>')
    parts.append('<h2>Execution modes</h2><table><tr><th>Mode</th><th>State</th><th>Slots with evidence</th><th>Plan items</th><th>Note</th></tr>')
    for m in data['execution_modes']:
        parts.append(f"<tr><td>{m['mode']}</td><td style='background:{COLORS[m['state']]}'>{sym[m['state']]} {m['state']}</td><td>{html.escape(', '.join(m['slots']))}</td><td>{', '.join(m['plan_items'])}</td><td>{html.escape(m['note'])}</td></tr>")
    mc = data.get('mode_cells') or []
    if mc:
        parts.append('</table><h2>Execution modes per model</h2><table><tr><th>Model</th><th>Slot</th><th>single</th><th>replicas</th><th>model_sharding</th><th>tensor_parallel</th><th>Evidence</th></tr>')
        by = {}
        for c in mc:
            by.setdefault((c['model_id'], c['slot']), {})[c['mode']] = c
        names = {r['model_id']: r['display_name'] for r in data['rows']}
        for (mid, slot), modes in sorted(by.items()):
            ev = sorted({e for c in modes.values() for e in c['evidence_ids']})
            parts.append(f"<tr><td><b>{html.escape(names.get(mid, mid))}</b></td><td>{html.escape(slot)}</td>")
            for m in ('single', 'replicas', 'model_sharding', 'tensor_parallel'):
                cell = modes.get(m)
                state = cell['state'] if cell else 'untried'
                title = html.escape((cell.get('note') or '') if cell else 'no receipt for this mode on this slot')
                parts.append(f"<td class=c style='background:{COLORS[state]}' title='{title}'>{sym[state]}</td>")
            parts.append(f"<td>{html.escape(', '.join(ev))}</td></tr>")
    parts.append('</table><p><small>A pass is one named model on one named slot with a receipt. Untried cells are the build-toward list (runnable by construction and not yet executed, runnable but on a slot no fleet host covers yet, or executed without matrix-cited evidence). Not-applicable cells are Titanos refusals by design or cannot exist by construction (no execution path for the backend on the slot runtimes, no published runtime package for the slot platform, or an identity naming no acquirable artifact); hover a cell for the reason.</small></p>')
    return '\n'.join(parts)


def render_csv(data):
    slot_by_id = {s['id']: s for s in data['slots']}
    rows_by_id = {r['model_id']: r for r in data['rows']}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(['model_id', 'display_name', 'identity', 'task', 'backend', 'modality_family', 'slot', 'os', 'arch', 'accelerator', 'state', 'evidence_ids', 'device_named', 'package_or_runtime', 'receipt', 'note'])
    for c in data['cells']:
        r, s = rows_by_id[c['model_id']], slot_by_id[c['slot']]
        w.writerow([c['model_id'], r['display_name'], r['identity'], r['task'], r['backend'], r['modality_family'], c['slot'], s['os'], s['arch'], s['accelerator'], c['state'], ' '.join(c['evidence_ids']), c.get('device_named', ''), c.get('package_or_runtime', ''), c.get('receipt', ''), c.get('note', '')])
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    data = json.loads(SRC.read_text(encoding='utf-8'))
    outputs = {MD: render_md(data), HTML: render_html(data), CSV: render_csv(data)}
    if a.check:
        for p, text in outputs.items():
            cur = p.read_text(encoding='utf-8').replace('\r\n', '\n') if p.exists() else ''
            if cur != text:
                raise SystemExit(f'{p.name} is out of date; run python tools/render_execution_matrix.py')
        print('execution matrix renderings match execution_matrix.json')
        return
    for p, text in outputs.items():
        p.write_text(text, encoding='utf-8')
    print(json.dumps({'written': [p.name for p in outputs], 'md_lines': len(outputs[MD].splitlines())}))


if __name__ == '__main__':
    main()
