"""Assemble docs/execution_matrix.json from reviewed reader output (a JSON file with {catalog, families[{cells, verdicts}]}).

Rules
- one row per catalog model; one cell per (model, platform slot) from docs/execution_matrix_spec.json; missing cells are `untried`.
- a refuted pass verdict replaces the cell state with the verifier's corrected state and keeps the reasoning.
- when readers disagree on the same cell: a verified pass wins (earlier failures are kept in the note), then fail, then partial,
  then na, then untried. Evidence ids are kept only when docs/CAPABILITY_MATRIX.md cites them; a pass/fail/partial cell with no
  surviving evidence is downgraded to untried with a note.
- a cell with no reader state, or whose readers only say `untried` (the absence of a claim), is decided by the spec's construction
  rules: `na` when the row identity names no acquirable artifact (unrecorded_identity_rule) or the row backend has no execution
  path through an available runtime on the slot (backend_runtime_map with runtime_availability; a backend's cpu_path_tokens are
  all-of, the runtimes it imports); otherwise `untried` with the builder's note naming the runtime tokens that grant the path
  (whether a fleet host covers the slot is a host circumstance, not part of the claim), or, on a slot no fleet host covers (no_fleet_host, a host
  circumstance and never a construction na), `untried: no fleet host for slot ...`. A reader-recorded execution (pass, fail,
  partial) or reader `na` always wins over the construction rules and keeps its evidence ids; only untried and
  builder-construction na carry none.
- published references are lab-neutral (tools/lab_paths.py, tools/public_export.json): drive-rooted lab records roots and agent
  work trees become `<lab-records>/...`, user-profile roots `<user-profile>/...`, Linux homes `<linux-home>/...` and hosts their
  stable role labels (lab_paths.HOST_ROLE_LABELS through the private host map) in every string written to docs/execution_matrix.json;
  the named-devices table names hardware and role labels only.
- mode cells: a (model, slot) pair with receipts but none for a multi-GPU mode gets `na` when the pair's receipts are all refusals,
  when Titanos refuses the mode for the row's backend/task by design (multi_gpu_modes_rule.admission, src/titanos/config.py) or when
  the slot can neither form the GPU group the mode needs nor declare the explicit CPU/XPU group through a runtime available for
  the row's backend; else an explicit `untried`.
- execution modes are a separate small table bound to plan items.
Usage: python tools/build_execution_matrix.py RAW.json"""
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lab_paths import neutralize_records  # tools/lab_paths.py: this script runs from tools/
from rollup_execution_matrix import derive, dumps

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / 'docs' / 'execution_matrix_spec.json').read_text(encoding='utf-8'))
SLOTS = [s['id'] for s in SPEC['platform_slots']]
STATES = list(SPEC['cell_states'])
RANK = {'pass': 4, 'fail': 3, 'partial': 2, 'na': 1, 'untried': 0}
EID = re.compile(r'\bE-\d{3}\b')
SLOT_RUNTIMES = {s['id']: list(s['runtimes']) for s in SPEC['platform_slots']}
BACKEND_MAP = SPEC['backend_runtime_map']
CPU_TOKENS = set(BACKEND_MAP['cpu_tokens'])
AVAILABILITY = SPEC['runtime_availability']
IDENTITY_RULE = SPEC['unrecorded_identity_rule']
IDENTITY_TRIGGER = re.compile(IDENTITY_RULE['trigger_regex'], re.I)
ACQUIRABLE = {k: re.compile(v) for k, v in IDENTITY_RULE['acquirable_regexes'].items()}
NO_FLEET_HOST = {h['slot']: h for h in SPEC['no_fleet_host']['slots']}
VARIANT_RULE = SPEC['unsupported_variant_rule']
VARIANTS = [(re.compile(v['row_id_regex']), re.compile(v['identity_regex']), v) for v in VARIANT_RULE['variants']]
MODES_RULE = SPEC['multi_gpu_modes_rule']
ADMISSION = MODES_RULE['admission']
NA_PREFIX = 'not applicable by construction: '
UNTRIED_PREFIX = 'untried: runnable by construction'
NO_HOST_PREFIX = 'untried: no fleet host for slot'
# A reader saw the cell executed (its own downgrade, or a receipt it could not bind to a cited E-id): such a cell is never na.
# (2026-09-25: the word boundary before `(a|the)` had been written as a literal backspace byte, so that alternative never matched.)
EXECUTED_UNBOUND = re.compile(r'downgraded from|by the evidence rule|\b(a|the) (private )?receipt records', re.I)
# The matrix kit's own construction na (tools/matrix_kit/run_cells.py no_path_cell) is computed from the runtime the host presented
# for the slot (slots.py: an Intel iGPU is `openvino-gpu` on a host without torch XPU, `torch-xpu` on one with it), so on a slot whose
# spec runtimes include a path that host lacked it records a host circumstance, not a construction fact. The spec decides: such a
# reader na is set aside (kept in the note) and the cell stays untried, runnable by construction, until a host with the path runs it.
KIT_RUNTIME_NA = re.compile(r'not applicable by construction: backend \S+ has no execution path on slot runtime \S+')


def unavailable_tokens(slot):
    """Runtime tokens that publish no package for the slot platform (spec runtime_availability): token -> fact."""
    return {tok: fact for tok, fact in (AVAILABILITY['slots'].get(slot) or {}).items() if fact.get('available') is False}


def cpu_path_tokens(backend, slot):
    """The backend's own CPU runtime tokens when the cpu rule applies (the backend lists `cpu` and the slot lists any cpu token);
    they are the runtimes the backend imports, so the CPU path needs every one of them (all-of: onnx-diffusers needs onnxruntime
    and torch, src/titanos/onnx_images.py:123, 193)."""
    if 'cpu' in BACKEND_MAP['backends'][backend] and set(SLOT_RUNTIMES[slot]) & CPU_TOKENS:
        return set(BACKEND_MAP['cpu_path_tokens'][backend])
    return set()


def candidate_tokens(backend, slot):
    """Slot runtime tokens in the backend's list plus its cpu_path_tokens when the cpu rule applies; before runtime_availability."""
    return (set(SLOT_RUNTIMES[slot]) & set(BACKEND_MAP['backends'][backend])) | cpu_path_tokens(backend, slot)


def granting_tokens(backend, slot):
    """The runtime tokens through which the backend's execution path on the slot runs: the slot tokens in its list that
    runtime_availability leaves available, plus all of its cpu_path_tokens when every one is available; empty when it has no
    path. The untried note names these, never the slot's raw runtime list."""
    if backend not in BACKEND_MAP['backends']:
        return []
    gone = set(unavailable_tokens(slot))
    direct = (set(SLOT_RUNTIMES[slot]) & set(BACKEND_MAP['backends'][backend])) - gone
    cpu_path = cpu_path_tokens(backend, slot)
    if cpu_path & gone:
        cpu_path = set()
    return sorted(direct | cpu_path)


def path_reason(backend, slot):
    """None when the catalog backend has an execution path through an available runtime on the slot; else why it has none.
    `cpu` in the backend's list means the kit's cpu runtime: on a slot listing any cpu token the backend's own cpu_path_tokens
    carry the path, unless runtime_availability removes any one of them for that slot (the path needs all of them)."""
    if backend not in BACKEND_MAP['backends']:
        return None  # a backend the map does not know is never declared impossible
    if granting_tokens(backend, slot):
        return None
    unavailable = unavailable_tokens(slot)
    blocked = sorted(candidate_tokens(backend, slot) & set(unavailable))
    if blocked:
        facts = '; '.join(f"{t}: {unavailable[t]['summary']}" for t in blocked)
        cpu_path = cpu_path_tokens(backend, slot)
        needs = f"; the CPU path of {backend} needs every one of [{', '.join(sorted(cpu_path))}]" if cpu_path - set(blocked) else ''
        return f"backend {backend} has no available runtime on slot {slot} (runtime_availability, PyPI checked {AVAILABILITY['checked']}): {facts}{needs}"
    return f"backend {backend} has no execution path on slot runtimes [{', '.join(SLOT_RUNTIMES[slot])}]; a converted artifact would be its own catalog row"


def acquirable_artifacts(identity):
    """What the identity names that could be acquired: repository slugs (owner/name, excluding Titanos tree paths, document
    references and version-like owners), URLs, local filesystem paths and content digests."""
    found = [('repository', m.group(0)) for m in ACQUIRABLE['repository_slug'].finditer(identity) if not ACQUIRABLE['slug_exclude'].search(m.group(0))]
    for key in ('url', 'local_path', 'digest'):
        m = ACQUIRABLE[key].search(identity)
        if m:
            found.append((key, m.group(0)))
    return found


def identity_unrecorded(identity):
    """The identity says the model is unrecorded AND names nothing acquirable (no repository slug, URL, local path or digest)."""
    return bool(IDENTITY_TRIGGER.search(identity)) and not acquirable_artifacts(identity)


def unsupported_variant(row):
    """The spec variant entry whose row id and identity regexes both match this row (unsupported_variant_rule), or None."""
    for row_re, identity_re, v in VARIANTS:
        if row_re.search(row['model_id']) and identity_re.search(row['identity']):
            return v
    return None


def construction_reason(row, slot, executed=()):
    """Why a (row, slot) cell cannot exist by construction, or None when it is runnable and merely unexecuted.
    `executed` lists (slot, evidence_ids) of the row's reader-recorded executions, which always stand and are named in the reason."""
    variant = unsupported_variant(row)
    if variant:
        return f"Titanos refuses this model variant before load ({VARIANT_RULE['since']}): {variant['reason']}"
    if identity_unrecorded(row['identity']):
        reason = f"{IDENTITY_RULE['reason']} ({IDENTITY_RULE['since']}): {row['identity'][:160]}"
        if executed:
            reason += '; recorded execution: ' + ', '.join(f"{s} [{', '.join(ids) or 'no cited E-id'}]" for s, ids in executed)
        return reason
    return path_reason(row['backend'], slot)


def runnable_note(row, slot):
    """The untried kind of a cell the construction rules leave runnable, naming the granting runtime tokens; on a slot no fleet
    host covers (spec no_fleet_host, a host circumstance and never a construction na) the kind is `untried: no fleet host`."""
    tokens = ', '.join(granting_tokens(row['backend'], slot))
    host = NO_FLEET_HOST.get(slot)
    if host:
        return f"{NO_HOST_PREFIX} {slot} (since {host['since']}): {host['reason']}; runnable by construction (backend {row['backend']} has an execution path through runtime [{tokens}], identity names an acquirable artifact) and waits for a host"
    return f"{UNTRIED_PREFIX} (backend {row['backend']} has an execution path through available runtime [{tokens}] on slot {slot}, identity names an acquirable artifact); not yet executed on this slot"


def mode_refusal(row, mode):
    """Titanos's own refusal of a multi-GPU mode for the row's backend and task (src/titanos/config.py, lines cited in the spec's
    multi_gpu_modes_rule.admission), or None when Titanos admits the mode and only the slot's group decides."""
    b, t = row['backend'], row['task']
    if b in ADMISSION['one_device_backends']:
        return f"Titanos refuses {mode} for backend {b} by design: one device per worker (src/titanos/config.py:237-238)"
    if t in ADMISSION['single_device_tasks']:
        return f"Titanos refuses {mode} for task {t} by design: single-device workers only ({ADMISSION['single_device_task_lines'][t]})"
    if mode == 'tensor_parallel' and not (b in ADMISSION['tensor_parallel']['backends'] and t in ADMISSION['tensor_parallel']['tasks']):
        return f"Titanos refuses tensor_parallel for backend {b} task {t} by design: native tensor parallelism is implemented only for Transformers chat, vision and embeddings (src/titanos/config.py:233-234, 345-346)"
    if mode == 'model_sharding' and b not in ADMISSION['model_sharding']['backends']:
        return f"Titanos refuses model_sharding for backend {b} by design: model sharding requires Transformers or Diffusers and this adapter refuses parallelism other than single (src/titanos/config.py:347-348; {ADMISSION['refusal_lines'].get(b, 'ModelSpec validator')})"
    return None


def mode_gap_reason(row, slot, mode):
    """Why a multi-GPU mode cannot exist for this row on this slot, or None when the slot can form the GPU group or explicitly
    declare the CPU/XPU group through a runtime that is available there for the row's backend (runtime_availability)."""
    rule = MODES_RULE['modes'][mode]
    unavailable = set(unavailable_tokens(slot))
    tokens = set(SLOT_RUNTIMES[slot]) - unavailable
    if tokens & set(MODES_RULE['group_runtimes']):
        return None
    explicit = set(rule['explicit_group_runtimes'])
    cpu_path = cpu_path_tokens(row['backend'], slot) if row['backend'] in BACKEND_MAP['backends'] else set()
    cpu_group = 'cpu' in explicit and bool(cpu_path) and not (cpu_path & unavailable)
    if cpu_group or tokens & explicit:
        return None
    return f"{mode} needs a homogeneous multi-GPU group and slot {slot} runtimes [{', '.join(SLOT_RUNTIMES[slot])}] cannot form one or declare an explicit group through a runtime available for backend {row['backend']}: {rule['reason']}"


def source_commit():
    """The full HEAD commit of this checkout, `-dirty` when tracked files differ from it; None outside a git checkout."""
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ('-dirty' if dirty else '')


def main(raw_path):
    raw = json.loads(Path(raw_path).read_text(encoding='utf-8'))
    matrix_ids = set(EID.findall((ROOT / 'docs' / 'CAPABILITY_MATRIX.md').read_text(encoding='utf-8')))
    catalog = raw['catalog']
    seen_ids = set()
    rows = []
    for m in catalog:
        mid = re.sub(r'[^a-z0-9.+-]+', '-', m['model_id'].lower()).strip('-')
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        rows.append({'model_id': mid, 'display_name': m['display_name'], 'identity': m['identity'], 'task': m['task'], 'backend': m['backend'], 'modality_family': m.get('modality_family') or m['task'], 'source': m.get('source', ''), 'notes': m.get('notes', '')})
    row_ids = {r['model_id'] for r in rows}
    alias = {}
    for r in rows:  # tolerate reader slugs that differ only by punctuation
        alias[re.sub(r'[^a-z0-9]', '', r['model_id'])] = r['model_id']

    def resolve(mid):
        k = re.sub(r'[^a-z0-9]', '', (mid or '').lower())
        return alias.get(k)

    verdicts = {}
    for fam in raw['families']:
        for v in fam.get('verdicts') or []:
            key = (resolve(v['model_id']), v['slot'])
            if key[0]:
                verdicts[key] = v
    candidates = defaultdict(list)
    unresolved = []
    for fam in raw['families']:
        for c in fam.get('cells') or []:
            mid = resolve(c['model_id'])
            if not mid or c['slot'] not in SLOTS:
                unresolved.append({'family': fam['family'], 'model_id': c['model_id'], 'slot': c['slot']})
                continue
            cell = dict(c)
            cell['model_id'] = mid
            cell['reader'] = fam['family']
            cell['evidence_ids'] = sorted({e for e in EID.findall(' '.join(str(x) for x in (c.get('evidence_ids') or []))) if e in matrix_ids})
            v = verdicts.get((mid, c['slot']))
            if cell['state'] == 'pass' and v is not None and v.get('upheld') is not True:  # only a boolean True upholds; 'true', 'false' or 1 never do
                cell['state'] = v['corrected_state']
                cell['note'] = (cell.get('note') or '') + ' | verifier: ' + v['reasoning']
                cell['verified'] = False
            elif cell['state'] == 'pass' and v is not None:
                cell['verified'] = True
            elif cell['state'] == 'pass':
                cell['state'] = 'partial'
                cell['note'] = (cell.get('note') or '') + ' | downgraded: pass cell was not covered by a verifier'
                cell['verified'] = False
            if cell['state'] in ('pass', 'fail', 'partial') and not cell['evidence_ids']:
                cell['note'] = (cell.get('note') or '') + f" | downgraded from {cell['state']}: no evidence id that the capability matrix cites"
                cell['state'] = 'untried'
            if cell['state'] == 'untried':
                cell['evidence_ids'] = []
            candidates[(mid, c['slot'])].append(cell)
    # Reader-recorded executions per row (slot, cited evidence): they always stand, and an identity-unrecorded row's na reason names them.
    executed_by_row = defaultdict(list)
    for (mid, slot), group in candidates.items():
        if any(c['state'] in ('pass', 'fail', 'partial') for c in group):
            executed_by_row[mid].append((slot, sorted({e for c in group if c['state'] in ('pass', 'fail', 'partial') for e in c['evidence_ids']})))
    for mid in executed_by_row:
        executed_by_row[mid].sort(key=lambda x: SLOTS.index(x[0]))
    cells = []
    for r in rows:
        for slot in SLOTS:
            group = candidates.get((r['model_id'], slot)) or []
            reason = construction_reason(r, slot, executed_by_row.get(r['model_id'], []))
            if not group:
                if reason:
                    cells.append({'model_id': r['model_id'], 'slot': slot, 'state': 'na', 'evidence_ids': [], 'note': NA_PREFIX + reason, 'readers': []})
                else:
                    cells.append({'model_id': r['model_id'], 'slot': slot, 'state': 'untried', 'evidence_ids': [], 'note': runnable_note(r, slot) + '; no record shows this model executing on this slot', 'readers': []})
                continue
            set_aside, set_aside_readers = '', []
            if path_reason(r['backend'], slot) is None:
                # the set-aside concerns the backend's path on the slot only: a row na by identity or variant keeps that construction
                # reason below even when a kit recorded a host-runtime na for it
                host_na = [c for c in group if c['state'] == 'na' and KIT_RUNTIME_NA.search(c.get('note') or '')]
                if host_na:
                    group = [c for c in group if c not in host_na]
                    set_aside_readers = [c['reader'] for c in host_na]
                    set_aside = ' | host-runtime na set aside (the spec grants the pair a path that host lacked): ' + '; '.join(f"{c['reader']}: {(c.get('note') or '')[:140]}" for c in host_na)
                    if not group:
                        if reason:  # na by identity or variant: the construction reason stands, the host-runtime na is only recorded
                            cells.append({'model_id': r['model_id'], 'slot': slot, 'state': 'na', 'evidence_ids': [], 'note': NA_PREFIX + reason + set_aside, 'readers': sorted(set(set_aside_readers))})
                        else:
                            cells.append({'model_id': r['model_id'], 'slot': slot, 'state': 'untried', 'evidence_ids': [], 'note': runnable_note(r, slot) + '; no record shows this model executing on this slot' + set_aside, 'readers': sorted(set(set_aside_readers))})
                        continue
            group.sort(key=lambda c: RANK[c['state']], reverse=True)
            best = group[0]
            others = [c for c in group[1:] if c['state'] != best['state']]
            note = (best.get('note') or '') + set_aside
            if others:
                note += ' | other readers: ' + '; '.join(f"{c['reader']}: {c['state']} ({(c.get('note') or '')[:120]})" for c in others)
            state = best['state']
            if state == 'untried':
                # A reader's untried is the absence of a claim, so the construction rules decide the cell; the reader's reason is kept.
                # A cell some reader saw executed but could not bind to cited evidence stays untried (evidence binding is pending work).
                executed_unbound = any(EXECUTED_UNBOUND.search(c.get('note') or '') for c in group)
                if executed_unbound:
                    note = 'untried: execution recorded by a reader but no evidence id the capability matrix cites; evidence binding pending | reader: ' + note
                elif reason:
                    state, note = 'na', NA_PREFIX + reason + ' | reader: ' + note
                else:
                    note = runnable_note(r, slot) + ' | reader: ' + note
            # Evidence ids stay on every reader-provided state including a reader's na (a receipt-backed refusal); only untried and a
            # builder-construction na (the readers said untried) carry none.
            builder_na = state == 'na' and best['state'] != 'na'
            evidence = [] if state == 'untried' or builder_na else sorted({e for c in group for e in c['evidence_ids']})
            cells.append({'model_id': r['model_id'], 'slot': slot, 'state': state, 'evidence_ids': evidence,
                          'device_named': best.get('device_named', ''), 'package_or_runtime': best.get('package_or_runtime', ''), 'receipt': best.get('receipt', ''), 'note': note.strip(' |'), 'verified': best.get('verified'), 'readers': sorted({c['reader'] for c in group} | set(set_aside_readers))})
    modes = [
        {'mode': 'single', 'state': 'pass', 'slots': ['win-x64-cpu', 'win-x64-nvidia-cuda', 'linux-x64-cpu', 'linux-x64-nvidia-cuda', 'macos-arm64-mps', 'win-arm64-cpu'], 'plan_items': ['LIFE-01'], 'note': 'the default mode of every executed cell above'},
        {'mode': 'replicas', 'state': 'pass', 'slots': ['win-x64-cpu', 'win-x64-nvidia-cuda'], 'plan_items': ['LIFE-04', 'HW-16'], 'note': 'independent replica workers with routing, failure isolation and cleanup proven (E-032 pool receipts, E-047/E-064 to E-067 device receipts)'},
        {'mode': 'model_sharding', 'state': 'partial', 'slots': ['win-x64-nvidia-cuda'], 'plan_items': ['LIFE-05'], 'note': 'implemented (Transformers device_map balanced); a plan-recorded frozen cohort result exists but no bound per-rank evidence; not qualified'},
        {'mode': 'tensor_parallel', 'state': 'partial', 'slots': ['win-x64-nvidia-cuda'], 'plan_items': ['LIFE-05'], 'note': 'implemented for transformers chat/vision/embeddings; no bound device/rank/operator evidence; not qualified'},
        {'mode': 'multi_host_pool', 'state': 'pass', 'slots': ['win-x64-cpu', 'win-arm64-adreno-directml', 'linux-x64-cpu'], 'plan_items': ['HW-16', 'LIFE-04'], 'note': 'matching Windows x64 CPU + x64 CPU and mixed x64 OpenVINO CPU + ARM64 DirectML topologies (E-032); live cross-machine footage not filmed'},
    ]
    # Named devices per slot: hardware and role labels only (tools/lab_paths.HOST_ROLE_LABELS; docs/MATRIX_KIT.md, Host labels), never
    # a computer name, an inventory name, an address or a lab plan path. Evidence ids are the capability rows that bind the device (its
    # cells on the slot, or the row that qualified it; the filter below keeps only ids docs/CAPABILITY_MATRIX.md cites). Status: proven (a verifier-upheld pass cell on
    # this slot names the device), partial (executed, qualification open), present_untried (in the fleet, no Titanos execution on this
    # slot), source_only (build flags only), deferred_untried (a host class outside the current scope), missing (no host).
    devices = [
        {'slot': 'win-x64-nvidia-cuda', 'device': 'NVIDIA RTX PRO 6000 Blackwell (GPU0, this workstation)', 'generation': 'Blackwell', 'status': 'proven', 'evidence_ids': ['E-064', 'E-066', 'E-069', 'E-077', 'E-088', 'E-090', 'E-092', 'E-104'], 'note': 'current native and frozen candidates; speech, chat, vision, embeddings, llama.cpp CUDA; the local matrix-kit campaigns and the music gate'},
        {'slot': 'win-x64-nvidia-cuda', 'device': 'NVIDIA RTX 5070 (rtx5070-desktop-win-x64)', 'generation': 'Blackwell', 'status': 'proven', 'evidence_ids': ['E-008', 'E-009', 'E-010', 'E-011', 'E-097'], 'note': 'speech and image/video cohorts 0.5.17 to 0.5.27; matrix-kit CUDA receipts (E-097)'},
        {'slot': 'win-x64-nvidia-cuda', 'device': 'NVIDIA RTX 3500 Ada (precision-5680-laptop-win-x64)', 'generation': 'Ada', 'status': 'proven', 'evidence_ids': ['E-002'], 'note': 'released 0.5.1 CUDA bundle chat with UUID verification'},
        {'slot': 'win-x64-nvidia-cuda', 'device': 'NVIDIA RTX 2070 Max-Q (turing-laptop-win-x64)', 'generation': 'Turing (compute 7.5, no bfloat16)', 'status': 'proven', 'evidence_ids': ['E-096'], 'note': 'matrix-kit CUDA receipts (fleet-turing-laptop-win-x64-20260924: seven fixture models, single and replicas, float16 policy); the same GPU passed under DirectML (E-098)'},
        {'slot': 'win-x64-nvidia-cuda', 'device': 'NVIDIA GTX 1080 Ti (rtx5090-desktop-win-x64)', 'generation': 'Pascal (compute 6.1, no bfloat16)', 'status': 'present_untried', 'evidence_ids': [], 'note': 'hidden from the host by the current driver; reachable only through a Hyper-V DDA guest with a Pascal-supported driver (planned, not executed); no Titanos execution yet'},
        {'slot': 'win-x64-nvidia-cuda', 'device': 'NVIDIA RTX 5090 (rtx5090-desktop-win-x64)', 'generation': 'Blackwell', 'status': 'present_untried', 'evidence_ids': [], 'note': 'no Titanos receipt on the CUDA slot (the device was held by a foreign workload during the fleet campaign); the host ran the CPU and DirectML campaigns (E-095)'},
        {'slot': 'win-x64-nvidia-cuda', 'device': 'NVIDIA Maxwell, Volta, Ampere, Hopper', 'generation': 'older/other', 'status': 'source_only', 'evidence_ids': [], 'note': 'CUDA build flags cover them; no such device in the fleet'},
        {'slot': 'linux-x64-nvidia-cuda', 'device': 'NVIDIA RTX 4090 x2 (linux-cuda-server-x64)', 'generation': 'Ada', 'status': 'proven', 'evidence_ids': ['E-022', 'E-023', 'E-024', 'E-025', 'E-040', 'E-065', 'E-093'], 'note': 'source 9b98 and native Linux CUDA packages; matrix-kit fleet receipts on one RTX 4090 (E-093)'},
        {'slot': 'linux-x64-nvidia-cuda', 'device': 'NVIDIA GTX 1080 (linux-cuda-server-x64)', 'generation': 'Pascal (compute 6.1, no bfloat16)', 'status': 'present_untried', 'evidence_ids': [], 'note': 'unbound on the host; a VFIO guest with driver 535 is the planned route (not executed); no Titanos execution yet'},
        {'slot': 'win-x64-nvidia-directml', 'device': 'NVIDIA RTX PRO 6000 Blackwell (DirectML, this workstation)', 'generation': 'Blackwell', 'status': 'proven', 'evidence_ids': ['E-091', 'E-092'], 'note': 'the local DirectML campaign; one fail retained (E-092)'},
        {'slot': 'win-x64-nvidia-directml', 'device': 'NVIDIA RTX 3500 Ada (DirectML, precision-5680-laptop-win-x64)', 'generation': 'Ada', 'status': 'proven', 'evidence_ids': ['E-002'], 'note': 'MiniLM DirectML graph'},
        {'slot': 'win-x64-nvidia-directml', 'device': 'NVIDIA RTX 2070 Max-Q (DirectML, turing-laptop-win-x64)', 'generation': 'Turing', 'status': 'proven', 'evidence_ids': ['E-098'], 'note': 'matrix-kit DirectML receipt'},
        {'slot': 'win-x64-amd-igpu-directml', 'device': 'AMD Radeon 840M (ryzen-ai-laptop-win-x64); Ryzen 7 7800X3D Raphael iGPU (rtx5090-desktop-win-x64)', 'generation': 'RDNA', 'status': 'proven', 'evidence_ids': ['E-002', 'E-039', 'E-095', 'E-096', 'E-097', 'E-098'], 'note': 'prepared graphs (E-002, E-039); matrix-kit DirectML receipts; one sd14 fail retained (E-099)'},
        {'slot': 'win-x64-amd-dgpu-directml', 'device': 'AMD RX 7900 XTX (rtx5070-desktop-win-x64)', 'generation': 'RDNA3', 'status': 'proven', 'evidence_ids': ['E-095', 'E-100'], 'note': 'matrix-kit DirectML receipts; two sd14 fails retained (E-099)'},
        {'slot': 'win-x64-amd-dgpu-directml', 'device': 'AMD RX 9070 XT (rtx5070-desktop-win-x64)', 'generation': 'RDNA4', 'status': 'present_untried', 'evidence_ids': [], 'note': 'present on the same host; no Titanos receipt names it'},
        {'slot': 'win-x64-amd-ryzen-ai-npu', 'device': 'AMD Ryzen AI 5 340 NPU (ryzen-ai-laptop-win-x64)', 'generation': 'XDNA2', 'status': 'partial', 'evidence_ids': ['E-002', 'E-097', 'E-098'], 'note': 'VitisAI EP passes and fails retained (E-097, E-098); VitisAI/VAIML packaging and provenance unresolved (HW-12)'},
        {'slot': 'win-x64-intel-dgpu-openvino', 'device': 'Intel Arc B580; Intel Arc A770 (intel-arc-win-x64-openvino)', 'generation': 'Battlemage / Alchemist', 'status': 'proven', 'evidence_ids': ['E-002', 'E-007', 'E-021', 'E-037', 'E-096', 'E-099', 'E-101'], 'note': 'OpenVINO GPU and OpenVINO GenAI SD1.5; PyTorch XPU passes (E-099); llama.cpp SYCL fails on both Arcs retained (E-101)'},
        {'slot': 'win-x64-intel-igpu', 'device': 'Intel Arc 130V (lunar-lake-laptop-win-x64)', 'generation': 'Xe2 (Lunar Lake)', 'status': 'proven', 'evidence_ids': ['E-026', 'E-028', 'E-095', 'E-096', 'E-100', 'E-101'], 'note': 'OpenVINO GPU, DirectML and PyTorch XPU receipts'},
        {'slot': 'win-x64-intel-igpu', 'device': 'Intel Iris Xe (precision-5680-laptop-win-x64)', 'generation': 'Xe-LP', 'status': 'proven', 'evidence_ids': ['E-002', 'E-100', 'E-101'], 'note': 'MiniLM embeddings; OpenVINO GPU and llama.cpp SYCL receipts'},
        {'slot': 'win-x64-intel-igpu', 'device': 'Intel UHD Graphics 630 (turing-laptop-win-x64)', 'generation': 'Gen9.5', 'status': 'proven', 'evidence_ids': ['E-098', 'E-100'], 'note': 'DirectML receipts'},
        {'slot': 'win-x64-intel-igpu', 'device': 'Intel integrated graphics (this workstation)', 'generation': 'Core Ultra iGPU', 'status': 'proven', 'evidence_ids': ['E-091', 'E-092'], 'note': 'the local DirectML and OpenVINO GPU campaigns; one DirectML fail retained (E-092)'},
        {'slot': 'win-x64-intel-npu-openvino', 'device': 'Intel AI Boost NPU (lunar-lake-laptop-win-x64)', 'generation': 'Lunar Lake NPU', 'status': 'proven', 'evidence_ids': ['E-002', 'E-026', 'E-028', 'E-095', 'E-098', 'E-100'], 'note': 'prepared MiniLM and digit graphs, CPU fallback disabled; matrix-kit OpenVINO NPU receipts'},
        {'slot': 'win-x64-intel-npu-openvino', 'device': 'Intel AI Boost NPU (this workstation)', 'generation': 'Core Ultra NPU', 'status': 'proven', 'evidence_ids': ['E-105'], 'note': 'source-runtime kit run through the typed ingest gate'},
        {'slot': 'win-arm64-adreno-directml', 'device': 'Qualcomm Adreno X1-45 (snapdragon-x-laptop-win-arm64)', 'generation': 'Snapdragon X', 'status': 'proven', 'evidence_ids': ['E-002', 'E-026', 'E-027', 'E-028', 'E-102', 'E-103'], 'note': 'prepared graphs; matrix-kit DirectML receipts'},
        {'slot': 'win-arm64-qnn-npu', 'device': 'Qualcomm Hexagon HTP (snapdragon-x-laptop-win-arm64)', 'generation': 'Snapdragon X', 'status': 'proven', 'evidence_ids': ['E-002', 'E-026', 'E-027', 'E-028', 'E-102', 'E-103'], 'note': 'prepared graphs; matrix-kit QNN receipts'},
        {'slot': 'linux-arm64-nvidia', 'device': 'NVIDIA GB10; Jetson AGX Thor', 'generation': 'Blackwell', 'status': 'deferred_untried', 'evidence_ids': [], 'note': 'a 0.5.0 TinyLlama HTTP chat on GB10 is recorded without evidence id or revision; GB10 and Thor hosts are out of the current scope'},
        {'slot': 'macos-arm64-mps', 'device': 'Apple M4 Max; Apple M1', 'generation': 'Apple Silicon', 'status': 'proven', 'evidence_ids': ['E-002', 'E-038'], 'note': 'native macOS ARM64 package 5270d444'},
        {'slot': 'linux-x64-amd-rocm', 'device': 'none in the fleet', 'generation': '-', 'status': 'missing', 'evidence_ids': [], 'note': 'HW-08 requested; no AMD Linux host'},
        {'slot': 'linux-arm64-cpu', 'device': 'none bound (the GB10 and Thor hosts are out of the current scope)', 'generation': '-', 'status': 'missing', 'evidence_ids': [], 'note': 'HW-04 requested'},
        {'slot': 'macos-x64', 'device': 'Intel MacBook Pro i9 (intel-macbook-pro-macos-x64)', 'generation': 'Intel', 'status': 'proven', 'evidence_ids': ['E-094', 'E-097', 'E-098', 'E-101'], 'note': 'matrix-kit receipts on the host CPU runtime (HW-15); the onnxruntime 1.23.2 deviation is recorded in the spec'},
    ]
    for dv in devices:
        dv['evidence_ids'] = [e for e in dv['evidence_ids'] if e in matrix_ids]
    # Per-model execution-mode results (single, replicas, model_sharding, tensor_parallel) from matrix-kit runs: same evidence
    # and verification rules as the main cells; an unverified pass is downgraded to partial, an unbound state to untried.
    mode_cells = []
    row_by_id = {r['model_id']: r for r in rows}
    for c in raw.get('mode_cells') or []:
        mid = resolve(c['model_id'])
        if not mid or c['slot'] not in SLOTS or c.get('mode') not in SPEC['execution_modes']:
            unresolved.append({'family': 'mode_cells', 'model_id': c.get('model_id'), 'slot': c.get('slot'), 'mode': c.get('mode')})
            continue
        cell = dict(c)
        cell['model_id'] = mid
        cell['evidence_ids'] = sorted({e for e in EID.findall(' '.join(str(x) for x in (c.get('evidence_ids') or []))) if e in matrix_ids})
        if cell['state'] == 'pass' and cell.get('verified') is not True:
            cell['state'] = 'partial'
            cell['note'] = (cell.get('note') or '') + ' | downgraded: pass not upheld by a verifier'
        if cell['state'] in ('pass', 'fail', 'partial') and not cell['evidence_ids']:
            cell['note'] = (cell.get('note') or '') + f" | downgraded from {cell['state']}: no evidence id that the capability matrix cites"
            cell['state'] = 'untried'
            cell['evidence_ids'] = []
        if cell['state'] == 'na' and KIT_RUNTIME_NA.search(cell.get('note') or '') and not construction_reason(row_by_id[mid], c['slot']):
            # the same host-runtime na as above, per mode: the spec grants the pair a path, so the mode stays untried unless the
            # builder's own multi-GPU rule (Titanos's refusal of the mode, or a slot that cannot form or declare the group) makes it na
            row = row_by_id[mid]
            refusal = mode_refusal(row, c['mode']) if c['mode'] in MODES_RULE['modes'] else None
            gap = mode_gap_reason(row, c['slot'], c['mode']) if c['mode'] in MODES_RULE['modes'] and not refusal else None
            if refusal or gap:
                cell['note'] = NA_PREFIX + (refusal or gap) + ' | host-runtime na set aside: ' + (cell.get('note') or '')
            else:
                cell['note'] = f"untried: runnable by construction (the spec grants backend {row['backend']} a path on slot {c['slot']} that the recording host lacked) | host-runtime na set aside: " + (cell.get('note') or '')
                cell['state'] = 'untried'
            cell['evidence_ids'] = []
        mode_cells.append(cell)
    # One published entry per (model, slot, mode): several runs record the same mode (a rerun, a second host on the slot, a host whose
    # kit runtime lacked the path); the best-ranked state stands (pass > fail > partial > na > untried, a verified pass first) and the
    # other runs' states stay in its note, so a later construction-derived untried can never displace a receipt-backed execution.
    grouped = defaultdict(list)
    for c in mode_cells:
        grouped[(c['model_id'], c['slot'], c['mode'])].append(c)
    mode_cells = []
    for key, group in grouped.items():
        group.sort(key=lambda c: (RANK[c['state']], 1 if c.get('verified') is True else 0), reverse=True)
        best = dict(group[0])
        others = group[1:]
        if others:
            best['note'] = (best.get('note') or '') + ' | also recorded: ' + '; '.join(f"{o.get('source', '?')}: {o['state']}" for o in others)
            best['other_sources'] = [{'source': o.get('source'), 'state': o['state'], 'evidence_ids': o.get('evidence_ids') or []} for o in others]
        mode_cells.append(best)
    # A pair with receipts for some mode but none for a multi-GPU mode (a receipt-backed state above always wins): na with the
    # pair's own refusal when every receipt-backed cell of the pair is na; na when Titanos refuses the mode for the row's
    # backend/task by design (spec admission, src/titanos/config.py); na when the slot can neither form the group nor declare the
    # explicit CPU/XPU group through a runtime available for the row's backend; otherwise an explicit untried.
    have = {(c['model_id'], c['slot'], c['mode']) for c in mode_cells}
    pairs = defaultdict(list)
    for c in mode_cells:
        pairs[(c['model_id'], c['slot'])].append(c)
    for (mid, slot), pair in sorted(pairs.items()):
        refused_pair = all(c['state'] == 'na' for c in pair)
        for mode in MODES_RULE['modes']:
            if (mid, slot, mode) in have:
                continue
            if refused_pair:
                lead = next((c for c in pair if c['mode'] == 'single'), pair[0])
                state, note = 'na', f"not applicable: {mode} shares the pair's refusal ({lead['mode']} is na): {re.sub(r'^\[\w+\] ', '', lead.get('note') or '')[:300]}"
            else:
                refusal = mode_refusal(row_by_id[mid], mode)
                gap = None if refusal else mode_gap_reason(row_by_id[mid], slot, mode)
                if refusal or gap:
                    state, note = 'na', NA_PREFIX + (refusal or gap)
                else:
                    state, note = 'untried', f"{UNTRIED_PREFIX} (Titanos admits {mode} for backend {row_by_id[mid]['backend']} task {row_by_id[mid]['task']}; slot {slot} can form or explicitly declare the group); no receipt for this mode on this slot"
            mode_cells.append({'model_id': mid, 'slot': slot, 'mode': mode, 'state': state, 'verified': False, 'evidence_ids': [], 'note': note, 'source': 'builder'})
    out = {'schema_version': 1, 'named_devices': devices, 'mode_cells': mode_cells, 'built_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'spec': 'docs/execution_matrix_spec.json',
           'basis': (raw.get('catalog_summary', '') + ' References are written as lab-neutral references: drive-rooted lab records roots, agent work trees and Windows user-profile roots are reduced to <lab-records>/... and <user-profile>/..., Linux home directories to <linux-home>/..., and hosts to their stable role labels (tools/lab_paths.HOST_ROLE_LABELS) by tools/lab_paths at generation (policy: tools/public_export.json).').strip(), 'slots': SPEC['platform_slots'], 'cell_states': SPEC['cell_states'], 'rows': rows, 'cells': cells, 'execution_modes': modes,
           'unresolved_reader_cells': unresolved[:50], 'counts': {'rows': len(rows), 'slots': len(SLOTS), 'cells': len(cells), **{s: sum(c['state'] == s for c in cells) for s in STATES}, 'verified_pass': sum(1 for c in cells if c['state'] == 'pass' and c.get('verified'))}}
    # Lab-neutral references (tools/public_export.json): every string the matrix publishes (rows' source/identity/notes, cell and mode-cell
    # receipt/note/device_named/source/readers, named_devices, the basis) has lab records roots, agent work trees, user-profile roots and
    # Linux homes reduced to their labels and hosts reduced to their role labels before writing; where a record was kept and what a
    # machine was called are operator circumstances, not part of the claim.
    out = neutralize_records(out)
    # derived fields (tools/rollup_execution_matrix.py): per-slot rollup, per-cell evidence_build, and the commit this build ran from
    out = derive(out, built_from_commit=source_commit())
    (ROOT / 'docs' / 'execution_matrix.json').write_text(dumps(out), encoding='utf-8')
    print(json.dumps(out['counts'] | {'unresolved': len(unresolved)}))


if __name__ == '__main__':
    main(sys.argv[1])
