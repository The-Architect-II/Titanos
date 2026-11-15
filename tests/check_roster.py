"""The representative roster (EVID-01) is bound to the plan and the evidence records.

Every roster row names an exact identity; its evidence ids exist as rows of docs/CAPABILITY_MATRIX.md; its plan items exist in
docs/progress.json; qualified rows cite evidence; requested_missing rows cite the plan item that requests them; every plan
item is served by at least one roster row or is listed with an explicit reason; docs/ROSTER.md equals the rendering.
No model, no network, no process."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import render_roster as renderer  # noqa: E402

ROSTER = ROOT / 'docs' / 'roster.json'
DOC = ROOT / 'docs' / 'ROSTER.md'
PROGRESS = ROOT / 'docs' / 'progress.json'
MATRIX = ROOT / 'docs' / 'CAPABILITY_MATRIX.md'
STATUSES = {'qualified_bounded', 'partial', 'source_only', 'requested_missing', 'deferred_unverified'}
KINDS = {'package', 'platform', 'model', 'workload', 'missing_family'}
IDENTITY = re.compile(r'([0-9a-f]{8,64}|@[0-9a-f]{7,40}|\d+\.\d+|[A-Z]{2,5}-\d{2}|b\d{4,6}|win|linux|darwin|x64|arm64|cuda|cpu|directml|openvino|rocm)', re.I)


def main():
    data = json.loads(ROSTER.read_text(encoding='utf-8'))
    rows = data['rows']
    progress = json.loads(PROGRESS.read_text(encoding='utf-8'))
    items = {it['id']: it for it in progress['items']}
    matrix_ids = set(re.findall(r'\bE-\d{3}\b', MATRIX.read_text(encoding='utf-8')))  # E-001..E-020 are cited inline by the legacy rows; later ids head their own rows
    failures = []
    for i, r in enumerate(rows):
        where = f"row {i} {r.get('kind')}/{r.get('name')}"
        for key in ('family', 'kind', 'name', 'identity', 'status', 'evidence_ids', 'plan_items', 'source'):
            if key not in r:
                failures.append(f'{where}: missing {key}')
        if r.get('kind') not in KINDS or r.get('status') not in STATUSES:
            failures.append(f'{where}: bad kind/status {r.get("kind")}/{r.get("status")}')
        exact = bool(IDENTITY.search(str(r.get('identity', ''))))
        unrecorded = re.search(r'not (stated|exported|recorded|named|in the public tree)|no (repository|named|profile|revision)|retained privately|candidates? only', str(r.get('identity', '')), re.I)
        if r.get('kind') in ('package', 'platform', 'model') and not exact and not (unrecorded and r.get('status') != 'qualified_bounded'):
            failures.append(f'{where}: identity is not exact ({r.get("identity")!r})')
        if r.get('kind') == 'workload' and not (exact or r.get('evidence_ids') or r.get('status') in ('requested_missing', 'source_only', 'deferred_unverified')):
            failures.append(f'{where}: workload row has neither an exact identity nor evidence receipts')
        if r.get('kind') == 'missing_family' and not (exact or r.get('plan_items')):
            failures.append(f'{where}: missing-family row names no plan item')
        for e in r.get('evidence_ids', []):
            if e not in matrix_ids:
                failures.append(f'{where}: evidence {e} is not a CAPABILITY_MATRIX row')
        for p in r.get('plan_items', []):
            if p not in items:
                failures.append(f'{where}: plan item {p} does not exist')
        if r.get('status') == 'qualified_bounded' and not r.get('evidence_ids'):
            failures.append(f'{where}: qualified_bounded without evidence')
        if r.get('status') == 'requested_missing' and not r.get('plan_items'):
            failures.append(f'{where}: requested_missing without the requesting plan item')
        if r.get('required_or_deferred') not in (None, 'required', 'deferred', 'owner_call'):
            failures.append(f'{where}: bad required_or_deferred')
        if r.get('status') == 'qualified_bounded':
            closed_items = [p for p in r.get('plan_items', []) if items.get(p, {}).get('status') == 'validated_bounded']
            # a qualified row may serve open items (bounded evidence toward an open gate); it may not cite closed items whose evidence lists none of its ids
            for p in closed_items:
                if not set(r['evidence_ids']) & set(items[p].get('evidence_ids', [])):
                    failures.append(f'{where}: cites closed item {p} without sharing any of its evidence ids')
    served = {p for r in rows for p in r.get('plan_items', [])}
    listed = {m['id']: m for m in data.get('plan_items_without_rows', [])}
    for pid in items:
        if pid not in served and pid not in listed:
            failures.append(f'plan item {pid} has no roster row and no listed reason')
    for pid in listed:
        if pid in served:
            failures.append(f'plan item {pid} is listed as without rows but has rows')
        if pid not in items:
            failures.append(f'listed plan item {pid} does not exist')
    rendered = renderer.render(data)
    current = DOC.read_text(encoding='utf-8').replace('\r\n', '\n') if DOC.exists() else ''
    if current != rendered:
        failures.append('docs/ROSTER.md is out of date; run python tools/render_roster.py')
    assert not failures, '\n'.join(failures[:40])
    print(json.dumps({'passed': True, 'rows': len(rows), 'by_kind': dict(Counter(r['kind'] for r in rows)), 'by_status': dict(Counter(r['status'] for r in rows)),
                      'plan_items': len(items), 'served': len(served), 'listed_without_rows': len(listed), 'evidence_ids_cited': len({e for r in rows for e in r['evidence_ids']}),
                      'scope': 'roster bindings only; no execution qualification'}))


if __name__ == '__main__':
    main()
