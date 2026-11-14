"""The execution matrix is complete, consistent and evidence-bound: every model x every platform slot has exactly one cell; every
state is in the spec vocabulary; pass/fail/partial cells cite evidence ids the capability matrix cites; untried cells cite none;
not-applicable cells carry a reason; pass cells were upheld by a verifier; the rendered Markdown/HTML/CSV match the JSON.
Every untried cell is one of the honest kinds (runnable by construction and not yet executed, whether the artifact is present or
still to be acquired, its note naming the runtime tokens that grant the path; runnable by construction on a slot no fleet host
covers, no_fleet_host, a host circumstance that is never na; or executed by a reader without matrix-cited evidence): no untried
cell is not-applicable by the spec's construction rules (backend_runtime_map with runtime_availability, unrecorded_identity_rule),
re-derived here independently of the builder, and no na cell cites a missing fleet host. Every executed cell (pass, fail, partial) has an execution path under those same rules, so no
construction rule or availability fact contradicts a recorded execution; the identity rule's row list in the spec equals the
derived set; builder-filled multi-GPU mode cells obey Titanos's admission (multi_gpu_modes_rule.admission) and the group rule, and
a pair whose receipts are all refusals has no builder-filled mode that is not na. The runtime_availability package/platform tables
are complete and every slot entry agrees with them (a token whose package publishes no wheel for the slot platform is declared
unavailable or carries an executed-deviation exception), and every catalogued native llama.cpp profile is mapped to slots of its
platform or listed as unmapped with a reason (one rule over the whole catalog)."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import render_execution_matrix as renderer  # noqa: E402

data = json.loads((ROOT / 'docs' / 'execution_matrix.json').read_text(encoding='utf-8'))
spec = json.loads((ROOT / 'docs' / 'execution_matrix_spec.json').read_text(encoding='utf-8'))
matrix_ids = set(re.findall(r'\bE-\d{3}\b', (ROOT / 'docs' / 'CAPABILITY_MATRIX.md').read_text(encoding='utf-8')))
slots = [s['id'] for s in spec['platform_slots']]
states = set(spec['cell_states'])
rows = data['rows']
slot_runtimes = {s['id']: set(s['runtimes']) for s in spec['platform_slots']}
backend_map = spec['backend_runtime_map']
cpu_tokens = set(backend_map['cpu_tokens'])
availability = spec['runtime_availability']
identity_rule = spec['unrecorded_identity_rule']
trigger_re = re.compile(identity_rule['trigger_regex'], re.I)
acquirable_re = {k: re.compile(v) for k, v in identity_rule['acquirable_regexes'].items()}
no_fleet_host = {h['slot'] for h in spec['no_fleet_host']['slots']}
modes_rule = spec['multi_gpu_modes_rule']
admission = modes_rule['admission']
row_by_id = {r['model_id']: r for r in rows}
assert spec['no_fleet_host'].get('kind') == 'untried' and all(h['since'] and h['reason'] for h in spec['no_fleet_host']['slots']), 'no_fleet_host is an untried kind whose slots need a dated reason'
assert no_fleet_host <= set(slot_runtimes), f'no_fleet_host names unknown slots: {sorted(no_fleet_host - set(slot_runtimes))}'
backend_tokens = {t for toks in backend_map['backends'].values() for t in toks if t != 'cpu'}
all_slot_tokens = {t for toks in slot_runtimes.values() for t in toks}
assert backend_tokens <= all_slot_tokens, f'backend_runtime_map names tokens no slot lists: {sorted(backend_tokens - all_slot_tokens)}'
assert all_slot_tokens <= backend_tokens | cpu_tokens, f'slot runtime tokens no backend can use: {sorted(all_slot_tokens - backend_tokens - cpu_tokens)}'
kit_tokens = {t for toks in backend_map['kit_runtimes'].values() for t in toks}
assert all_slot_tokens <= kit_tokens | cpu_tokens | set(backend_map['tokens_without_kit_runtime']), f'slot tokens in no kit runtime and not declared as such: {sorted(all_slot_tokens - kit_tokens - cpu_tokens - set(backend_map["tokens_without_kit_runtime"]))}'
assert all(t in all_slot_tokens and t not in kit_tokens and reason for t, reason in backend_map['tokens_without_kit_runtime'].items()), 'tokens_without_kit_runtime must be slot tokens no kit runtime maps, each with a reason'
assert set(r['backend'] for r in rows) <= set(backend_map['backends']), f"backend_runtime_map lacks backends: {set(r['backend'] for r in rows) - set(backend_map['backends'])}"
assert set(backend_map['cpu_path_tokens']) >= {b for b, allowed in backend_map['backends'].items() if 'cpu' in allowed}, 'every cpu-capable backend needs cpu_path_tokens'
assert all(set(toks) <= cpu_tokens for toks in backend_map['cpu_path_tokens'].values()), 'cpu_path_tokens must be cpu_tokens'
assert not any('onnx CUDA' in toks for toks in list(slot_runtimes.values()) + list(backend_map['backends'].values())), 'onnx CUDA is not a runtime token: src/titanos/compiled.py PROVIDERS has no CUDA execution provider'
for slot_id, facts in availability['slots'].items():
    assert slot_id in slot_runtimes, f'runtime_availability names unknown slot {slot_id}'
    for tok, f in facts.items():
        assert f.get('available') is False and f.get('package') and f.get('versions_checked') and f.get('checked') and f.get('reason') and f.get('summary'), f'runtime_availability {slot_id}/{tok} needs package, versions_checked, checked, reason and summary'
assert all(x.get('reason') for x in identity_rule['rows']), 'every unrecorded_identity_rule row needs a reason'
assert set(admission['single_device_task_lines']) == set(admission['single_device_tasks']), 'every single-device task needs its refusing config.py lines (and no stray ones)'
NA_REASON = re.compile(r'not applicable|refused by design|^na\b|na per', re.I)
UNTRIED_KINDS = re.compile(r'^untried: (runnable by construction|no fleet host for slot|execution recorded by a reader but no evidence id)')
NOTE_TOKENS = re.compile(r'\[([^\]]*)\]')
# a matrix-kit construction na computed from the runtime one host presented for the slot (tools/matrix_kit/run_cells.py no_path_cell)
KIT_RUNTIME_NA = re.compile(r'(\[\w+\] )?not applicable by construction: backend \S+ has no execution path on slot runtime \S+')


def unavailable(slot):
    return {tok for tok, f in (availability['slots'].get(slot) or {}).items() if f.get('available') is False}


def path_tokens(backend, slot, with_availability=True):
    """The runtime tokens that grant the backend a path on the slot: the available slot tokens in its list plus all of its
    cpu_path_tokens when the cpu rule applies and every one is available (all-of: they are the runtimes the backend imports)."""
    allowed = set(backend_map['backends'][backend])
    tokens = slot_runtimes[slot]
    gone = unavailable(slot) if with_availability else set()
    cpu_path = set(backend_map['cpu_path_tokens'][backend]) if 'cpu' in allowed and tokens & cpu_tokens else set()
    if cpu_path & gone:
        cpu_path = set()
    return ((tokens & allowed) - gone) | cpu_path


def has_path(backend, slot, with_availability=True):
    return bool(path_tokens(backend, slot, with_availability))


def acquirable(identity):
    slugs = [m.group(0) for m in acquirable_re['repository_slug'].finditer(identity) if not acquirable_re['slug_exclude'].search(m.group(0))]
    return bool(slugs) or any(acquirable_re[k].search(identity) for k in ('url', 'local_path', 'digest'))


def identity_unrecorded(identity):
    return bool(trigger_re.search(identity)) and not acquirable(identity)


variant_rule = spec['unsupported_variant_rule']


def unsupported_variant(row):
    return any(re.search(v['row_id_regex'], row['model_id']) and re.search(v['identity_regex'], row['identity']) for v in variant_rule['variants'])


def na_by_construction(row, slot):
    return unsupported_variant(row) or identity_unrecorded(row['identity']) or not has_path(row['backend'], slot)



def mode_admitted(row, mode):
    b, t = row['backend'], row['task']
    if b in admission['one_device_backends'] or t in admission['single_device_tasks']:
        return False
    if mode == 'tensor_parallel':
        return b in admission['tensor_parallel']['backends'] and t in admission['tensor_parallel']['tasks']
    return b in admission['model_sharding']['backends']


def slot_forms_group(row, slot, mode):
    """The slot can form the GPU group, or declare the explicit CPU/XPU group through a runtime available there for the row's backend."""
    gone = unavailable(slot)
    tokens = slot_runtimes[slot] - gone
    explicit = set(modes_rule['modes'][mode]['explicit_group_runtimes'])
    cpu_path = set(backend_map['cpu_path_tokens'].get(row['backend'], ())) if slot_runtimes[slot] & cpu_tokens else set()
    cpu_group = 'cpu' in explicit and bool(cpu_path) and not (cpu_path & gone)
    return bool(tokens & set(modes_rule['group_runtimes'])) or bool(tokens & explicit) or cpu_group


derived_unrecorded = {r['model_id'] for r in rows if identity_unrecorded(r['identity'])}
listed_unrecorded = {x['model_id'] for x in identity_rule['rows']}
assert derived_unrecorded == listed_unrecorded, f'unrecorded_identity_rule.rows drifted from the derived set: +{sorted(derived_unrecorded - listed_unrecorded)} -{sorted(listed_unrecorded - derived_unrecorded)}'
# runtime_availability: the package/platform tables decide, and every slot entry must agree with them, so one standard applies to
# every slot platform the pins do not cover (a torch-backed token on a platform torch publishes no wheel for is declared
# unavailable or carries an executed-deviation exception; never an unexplained 'available runtime' claim)
slot_platform = {s['id']: f"{s['os']}/{s['arch']}" for s in spec['platform_slots']}
platform_tags = {k: v for k, v in availability['platform_tags'].items() if k != 'rule'}
token_packages = {k: v for k, v in availability['token_packages'].items() if k != 'rule'}
packages = availability['packages']
assert set(slot_platform.values()) <= set(platform_tags), f'runtime_availability.platform_tags lacks slot platforms: {sorted(set(slot_platform.values()) - set(platform_tags))}'
assert set(token_packages) <= all_slot_tokens | cpu_tokens, f'runtime_availability.token_packages names unknown tokens: {sorted(set(token_packages) - all_slot_tokens - cpu_tokens)}'
assert set(token_packages.values()) <= set(packages), f'runtime_availability.token_packages names unmeasured packages: {sorted(set(token_packages.values()) - set(packages))}'
for name, pkg in packages.items():
    assert pkg.get('pin') and pkg.get('checked') and pkg.get('versions_checked') and pkg.get('published_platforms'), f'runtime_availability.packages.{name} needs pin, checked, versions_checked and published_platforms'
deviation_keys = {(d['slot'], d['token']) for d in availability.get('deviations', [])}
table_failures = []
for slot_id in slot_runtimes:
    family = platform_tags[slot_platform[slot_id]]
    relevant = set(slot_runtimes[slot_id]) | (cpu_tokens if slot_runtimes[slot_id] & cpu_tokens else set())
    for tok in sorted(relevant & set(token_packages)):
        published = family in packages[token_packages[tok]]['published_platforms']
        declared = tok in unavailable(slot_id)
        if not published and not declared and (slot_id, tok) not in deviation_keys:
            table_failures.append(f"{slot_id}/{tok}: {token_packages[tok]} publishes no {family} wheel under its pin, but the token is neither declared unavailable (runtime_availability.slots) nor an executed deviation (runtime_availability.deviations)")
        if declared and published:
            table_failures.append(f"{slot_id}/{tok}: declared unavailable, but runtime_availability.packages.{token_packages[tok]} lists {family} as published")
        if declared and availability['slots'][slot_id][tok].get('package') != token_packages[tok]:
            table_failures.append(f"{slot_id}/{tok}: slot entry package {availability['slots'][slot_id][tok].get('package')!r} differs from token_packages {token_packages[tok]!r}")
    for tok in set(availability['slots'].get(slot_id) or {}):
        if tok not in relevant:
            table_failures.append(f"{slot_id}/{tok}: runtime_availability entry for a token the slot cannot use")
assert not table_failures, '\n'.join(table_failures)
# native llama.cpp profiles: one rule over Titanos's whole catalog (src/titanos/data/native-profiles.json)
native_rule = backend_map['native_llama_cpp_profiles']
native_catalog = json.loads((ROOT / 'src' / 'titanos' / 'data' / 'native-profiles.json').read_text(encoding='utf-8'))['profiles']
mapped, unmapped = native_rule['mapped'], native_rule['unmapped']
assert not set(mapped) & set(unmapped), f'native profiles both mapped and unmapped: {sorted(set(mapped) & set(unmapped))}'
assert set(mapped) | set(unmapped) == set(native_catalog), f'native_llama_cpp_profiles drifted from native-profiles.json: +{sorted((set(mapped) | set(unmapped)) - set(native_catalog))} -{sorted(set(native_catalog) - set(mapped) - set(unmapped))}'
assert all(isinstance(reason, str) and reason for reason in unmapped.values()), 'every unmapped native profile needs a reason'
slot_by_id = {s['id']: s for s in spec['platform_slots']}
native_platforms = native_rule['platforms']
for pid, m in mapped.items():
    prof = native_catalog[pid]
    assert m['tokens'] and set(m['tokens']) <= set(backend_map['backends']['llama-cpp']) | {'llama.cpp CPU'}, f'native profile {pid} maps to tokens backend_runtime_map.backends.llama-cpp does not list: {m["tokens"]}'
    assert m['slots'], f'native profile {pid} is mapped to no slot'
    for sid in m['slots']:
        s_ = slot_by_id[sid]
        assert s_['os'] == native_platforms[prof['os']] and s_['arch'] in native_platforms[prof['arch']], f'native profile {pid} ({prof["os"]}/{prof["arch"]}) mapped to slot {sid} ({s_["os"]}/{s_["arch"]})'
        lists = bool(set(s_['runtimes']) & set(m['tokens'])) or ('llama.cpp CPU' in m['tokens'] and bool(set(s_['runtimes']) & cpu_tokens))
        assert lists, f'native profile {pid} mapped to slot {sid}, which lists none of its tokens {m["tokens"]}'


failures = []
ids = [r['model_id'] for r in rows]
# unsupported_variant_rule sanity: every entry matches at least one row by both regexes, never a row whose identity regex misses,
# names a `reason` and cites checked-in lines that still carry the refusal
for v in variant_rule['variants']:
    hit = [r['model_id'] for r in rows if re.search(v['row_id_regex'], r['model_id'])]
    both = [r['model_id'] for r in rows if re.search(v['row_id_regex'], r['model_id']) and re.search(v['identity_regex'], r['identity'])]
    if not hit or hit != both:
        failures.append(f"unsupported_variant_rule {v['row_id_regex']}: row ids matched {hit} but identities matched {both}")
    if not v.get('reason') or not v.get('citations'):
        failures.append(f"unsupported_variant_rule {v['row_id_regex']}: reason and citations are required")
    # the regexes must leave the family's supported variant untouched (ACE_SUPPORTED_VARIANTS in src/titanos/music_backend.py)
    supported = [r['model_id'] for r in rows if r['backend'] == 'ace-step' and 'acestep-v15-xl-sft@' in r['identity']]
    if any(re.search(v['row_id_regex'], mid) for mid in supported):
        failures.append(f"unsupported_variant_rule {v['row_id_regex']}: matches the supported variant row(s) {supported}")
# every runtime_availability deviation is backed by an executed cell (pass/fail/partial) on that slot for a backend that uses the token:
# a deviation is a fact about receipts, never a way to relabel an unavailable token
cells_by_slot = {}
for c in data['cells']:
    cells_by_slot.setdefault(c['slot'], []).append(c)
for dev in spec['runtime_availability'].get('deviations', []):
    tok = dev['token']
    users = {b for b, toks in backend_map['backends'].items() if tok in toks or (tok in backend_map['cpu_path_tokens'].get(b, ()))}
    executed = [c for c in cells_by_slot.get(dev['slot'], []) if c['state'] in ('pass', 'fail', 'partial') and row_by_id[c['model_id']]['backend'] in users]
    if not executed:
        failures.append(f"runtime_availability deviation {dev['slot']} / {tok}: no executed cell of a backend using that token on the slot backs it")
    for cite in v.get('citations', []):
        path, _, lines = cite.partition(':')
        text = (ROOT / path).read_text(encoding='utf-8').splitlines() if (ROOT / path).exists() else []
        lo, _, hi = lines.partition('-')
        window = '\n'.join(text[int(lo) - 1:int(hi or lo)])
        if not text or not ('ACE_SUPPORTED_VARIANTS' in window or 'unsupported_adapter' in window or 'this variant remains a discovery candidate' in window):
            failures.append(f"unsupported_variant_rule citation {cite} does not show the refusal")
assert len(ids) == len(set(ids)), 'duplicate model ids'
for r in rows:
    for k in ('display_name', 'identity', 'task', 'backend', 'modality_family'):
        if not r.get(k):
            failures.append(f"{r['model_id']}: missing {k}")
    exact = bool(re.search(r'[0-9a-f]{7,64}|@|\d+\.\d+|release|asset|sha', r['identity'], re.I))
    unrecorded = bool(trigger_re.search(r['identity']))
    if not exact and not unrecorded:
        failures.append(f"{r['model_id']}: identity not exact ({r['identity']!r})")
    if not exact and any(c['state'] == 'pass' for c in data['cells'] if c['model_id'] == r['model_id']):
        failures.append(f"{r['model_id']}: pass cell on a model whose identity is not recorded")
cells = {(c['model_id'], c['slot']): c for c in data['cells']}
assert len(cells) == len(data['cells']), 'duplicate cells'
for r in rows:
    for s in slots:
        c = cells.get((r['model_id'], s))
        if c is None:
            failures.append(f"{r['model_id']} x {s}: missing cell")
            continue
        if c['state'] not in states:
            failures.append(f"{r['model_id']} x {s}: bad state {c['state']}")
        if c['state'] in ('pass', 'fail', 'partial'):
            if not c['evidence_ids']:
                failures.append(f"{r['model_id']} x {s}: {c['state']} without evidence")
            for e in c['evidence_ids']:
                if e not in matrix_ids:
                    failures.append(f"{r['model_id']} x {s}: evidence {e} not in the capability matrix")
            # a recorded execution is the ground truth: the construction rules must grant it a path, or the rule is wrong
            if not has_path(r['backend'], s, with_availability=False):
                failures.append(f"{r['model_id']} x {s}: {c['state']} recorded but backend_runtime_map grants backend {r['backend']} no path on the slot; fix the map, not the cell")
            elif not has_path(r['backend'], s):
                failures.append(f"{r['model_id']} x {s}: {c['state']} recorded but a runtime_availability fact removes every path for backend {r['backend']} on the slot; the receipt contradicts the fact, which must not be encoded (unless the cell is a fail whose note is only an import or install error, which is not an execution: refile it as untried)")
            if s in no_fleet_host:
                failures.append(f"{r['model_id']} x {s}: {c['state']} recorded on a slot no_fleet_host declares hostless")
        if c['state'] == 'untried' and c['evidence_ids']:
            failures.append(f"{r['model_id']} x {s}: untried with evidence")
        if c['state'] == 'na' and not (c.get('note') and NA_REASON.search(c['note'])):
            failures.append(f"{r['model_id']} x {s}: na without reason ({(c.get('note') or '')[:80]!r})")
        if c['state'] == 'untried':
            note = c.get('note') or ''
            if not UNTRIED_KINDS.search(note):
                failures.append(f"{r['model_id']} x {s}: untried without an honest kind in the note ({note[:80]!r})")
            if note.startswith(('untried: runnable by construction', 'untried: no fleet host for slot')):
                if na_by_construction(r, s):
                    failures.append(f"{r['model_id']} x {s}: untried but not applicable by construction (backend {r['backend']}, identity {r['identity'][:40]!r})")
                # the note names the tokens that grant the path (never the slot's raw runtime list): each must grant it
                m = NOTE_TOKENS.search(note)
                named = {t for t in (m.group(1).split(', ') if m else []) if t}
                if not named or not named <= path_tokens(r['backend'], s):
                    failures.append(f"{r['model_id']} x {s}: untried note names runtime tokens {sorted(named)}, not the granting set {sorted(path_tokens(r['backend'], s))} for backend {r['backend']}")
            if note.startswith('untried: runnable by construction') and s in no_fleet_host:
                failures.append(f"{r['model_id']} x {s}: 'runnable by construction' kind on a slot no_fleet_host lists (must be the no-fleet-host kind)")
            if note.startswith('untried: no fleet host for slot') and s not in no_fleet_host:
                failures.append(f"{r['model_id']} x {s}: untried for a missing fleet host on a slot no_fleet_host does not list")
        if c['state'] == 'na' and re.match(r'not applicable( by construction)?: no fleet host', c.get('note') or ''):
            failures.append(f"{r['model_id']} x {s}: na for a missing fleet host, which is a host circumstance and never a construction fact")
        if c['state'] == 'na' and KIT_RUNTIME_NA.match(c.get('note') or '') and not na_by_construction(r, s):
            failures.append(f"{r['model_id']} x {s}: na from a kit's host-presented slot runtime although the spec grants backend {r['backend']} a path on the slot (a host circumstance, never a construction fact)")
        if c['state'] == 'pass' and c.get('verified') is not True:
            failures.append(f"{r['model_id']} x {s}: pass not upheld by a verifier")
for c in data['cells']:
    if c['model_id'] not in set(ids) or c['slot'] not in slots:
        failures.append(f"stray cell {c['model_id']} x {c['slot']}")
for c in data.get('mode_cells', []):
    if c['model_id'] not in set(ids) or c['slot'] not in slots or c.get('mode') not in spec['execution_modes'] or c['state'] not in states:
        failures.append(f"mode cell invalid: {c.get('model_id')} x {c.get('slot')} [{c.get('mode')}] {c.get('state')}")
        continue
    if c['state'] in ('pass', 'fail', 'partial') and (not c['evidence_ids'] or any(e not in matrix_ids for e in c['evidence_ids'])):
        failures.append(f"mode cell {c['model_id']} x {c['slot']} [{c['mode']}]: {c['state']} without matrix-cited evidence")
    if c['state'] == 'pass' and c.get('verified') is not True:
        failures.append(f"mode cell {c['model_id']} x {c['slot']} [{c['mode']}]: pass not upheld by a verifier")
    if c['state'] == 'na' and not (c.get('note') and NA_REASON.search(c['note'])):
        failures.append(f"mode cell {c['model_id']} x {c['slot']} [{c['mode']}]: na without reason")
    if c['state'] == 'na' and KIT_RUNTIME_NA.match(c.get('note') or '') and not na_by_construction(row_by_id[c['model_id']], c['slot']):
        failures.append(f"mode cell {c['model_id']} x {c['slot']} [{c['mode']}]: na from a kit's host-presented slot runtime although the spec grants a path on the slot")
    if c['state'] == 'untried' and not (c.get('note') or '').startswith(('untried: ', 'not run: ')):
        failures.append(f"mode cell {c['model_id']} x {c['slot']} [{c['mode']}]: untried without reason")
    if c.get('source') == 'builder':
        row = row_by_id[c['model_id']]
        if c['evidence_ids']:
            failures.append(f"mode cell {c['model_id']} x {c['slot']} [{c['mode']}]: builder-filled cell with evidence")
        if c['state'] == 'untried' and not mode_admitted(row, c['mode']):
            failures.append(f"mode cell {c['model_id']} x {c['slot']} [{c['mode']}]: untried but Titanos refuses {c['mode']} for backend {row['backend']} task {row['task']} by design")
        if c['state'] == 'untried' and not slot_forms_group(row, c['slot'], c['mode']):
            failures.append(f"mode cell {c['model_id']} x {c['slot']} [{c['mode']}]: untried but the slot cannot form or declare the group")
# one entry per (model, slot, mode); a receipt-backed state is never displaced by a lower-ranked run's state (review finding #11)
seen_keys = {}
rank = {'pass': 4, 'fail': 3, 'partial': 2, 'na': 1, 'untried': 0}
for c in data.get('mode_cells', []):
    k = (c['model_id'], c['slot'], c['mode'])
    if k in seen_keys:
        failures.append(f"mode cell {k}: duplicated entry (the renderer would publish whichever comes last)")
    seen_keys[k] = c
    for o in c.get('other_sources') or []:
        if rank.get(o['state'], -1) > rank[c['state']]:
            failures.append(f"mode cell {k}: published {c['state']} although run {o['source']} recorded {o['state']}")
pairs = {}
for c in data.get('mode_cells', []):
    pairs.setdefault((c['model_id'], c['slot']), []).append(c)
for (mid, slot), pair in pairs.items():
    receipt_backed = [c for c in pair if c.get('source') != 'builder']
    if receipt_backed and all(c['state'] == 'na' for c in receipt_backed):
        for c in pair:
            if c.get('source') == 'builder' and c['state'] != 'na':
                failures.append(f"mode cell {mid} x {slot} [{c['mode']}]: {c['state']} although every receipt-backed mode of the pair is a refusal")
for m in data['execution_modes']:
    if m['mode'] not in spec['execution_modes'] or m['state'] not in states or not m['plan_items']:
        failures.append(f"execution mode row invalid: {m}")
for name, path in (('MD', renderer.MD), ('HTML', renderer.HTML), ('CSV', renderer.CSV)):
    rendered = {'MD': renderer.render_md, 'HTML': renderer.render_html, 'CSV': renderer.render_csv}[name](data)
    current = path.read_text(encoding='utf-8').replace('\r\n', '\n') if path.exists() else ''
    if current != rendered:
        failures.append(f'{path.name} is out of date; run python tools/render_execution_matrix.py')
# derived fields (tools/rollup_execution_matrix.py): current, recomputed here independently, and described by the JSON Schema
import rollup_execution_matrix as rollup  # noqa: E402
probe = 'source HEAD 9b98d0a4; titanos-e93f066 source; source 9b98d0a49d49...; E-007: 0.5.25.dev0 bundle; torch 2.10.0.dev20250101+cu130; Titanos 0.5.2 source; llama.cpp commit 5266f24da'
if rollup.evidence_build({'state': 'pass', 'package_or_runtime': probe}) != {'commits': ['9b98d0a49d49', 'e93f066'], 'versions': ['0.5.2', '0.5.25.dev0']} or rollup.evidence_build({'state': 'untried', 'package_or_runtime': probe}) is not None or rollup.evidence_build({'state': 'fail', 'package_or_runtime': 'titanos repo source'}) is not None:
    failures.append('rollup_execution_matrix.evidence_build extraction rules changed (commit/version spellings, nightly exclusion, prefix merge, executed-only)')
if rollup.derive(data) != data:
    failures.append('docs/execution_matrix.json derived fields are out of date; run python tools/rollup_execution_matrix.py')
for slot, r in zip(slots, data.get('slot_rollup') or []):
    mine = [c for c in data['cells'] if c['slot'] == slot]
    if r['slot'] != slot or r['counts'] != {s: sum(c['state'] == s for c in mine) for s in data['cell_states']} or r['verified_pass_models'] != sorted(c['model_id'] for c in mine if c['state'] == 'pass' and c['verified'] is True):
        failures.append(f'slot_rollup for {slot} disagrees with its cells')
if [r['slot'] for r in data.get('slot_rollup') or []] != slots or sum(r['counts']['pass'] for r in data['slot_rollup']) != data['counts']['pass']:
    failures.append('slot_rollup does not cover the spec slots in order, or its pass total differs from counts.pass')
for c in data['cells']:
    b = c.get('evidence_build')
    if b is not None and (c['state'] not in ('pass', 'fail', 'partial') or not all(x in c['package_or_runtime'] for x in b['commits'] + b['versions'])):
        failures.append(f"cell {c['model_id']} x {c['slot']}: evidence_build names a build its package_or_runtime record does not")
built = data.get('built_from_commit')
if built is not None and not re.fullmatch(r'[0-9a-f]{40}(-dirty)?', built):
    failures.append(f'built_from_commit is not a full commit id: {built!r}')
schema = json.loads((ROOT / 'docs' / 'execution_matrix.schema.json').read_text(encoding='utf-8'))
try:
    import jsonschema
except ImportError:
    jsonschema = None
schema_validated = jsonschema is not None
if schema_validated:
    failures += [f'schema: {"/".join(map(str, e.absolute_path))}: {e.message[:200]}' for e in jsonschema.Draft202012Validator(schema).iter_errors(data)][:20]
elif not set(schema['required']) <= set(data):
    failures.append(f"schema: missing top-level keys {sorted(set(schema['required']) - set(data))}")
# both files ship in every source copy: the sdist (MANIFEST.in recursive-include docs *.json) and the native SOURCE/ docs tree
# (tools/build_binary.py copies docs/ minus tools/public_export.json excluded_paths)
excluded = [str(p).strip('/').casefold() for p in json.loads((ROOT / 'tools' / 'public_export.json').read_text(encoding='utf-8'))['excluded_paths']]
manifest = (ROOT / 'MANIFEST.in').read_text(encoding='utf-8')
for name in ('docs/execution_matrix.json', 'docs/execution_matrix.schema.json'):
    if any(name.casefold() == p or name.casefold().startswith(p + '/') for p in excluded) or not re.search(r'^recursive-include docs .*\*\.json', manifest, re.M):
        failures.append(f'{name} would be left out of the source bundles')
assert not failures, '\n'.join(failures[:40])
counts = Counter(c['state'] for c in data['cells'])
print(json.dumps({'passed': True, 'rows': len(rows), 'slots': len(slots), 'cells': len(data['cells']), 'states': dict(counts), 'verified_pass': sum(1 for c in data['cells'] if c['state'] == 'pass' and c.get('verified')), 'schema_validated': schema_validated, 'scope': 'coverage, vocabulary and evidence binding; no execution qualification'}))
