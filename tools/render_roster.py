"""Render docs/ROSTER.md from docs/roster.json (EVID-01: the frozen representative roster and per-family gates).

Each row names one package, model, platform, workload or requested-but-missing family with an exact identity, its status,
the evidence ids that bind it and the plan items whose gates it serves. tests/check_roster.py asserts the bindings against
docs/progress.json and docs/CAPABILITY_MATRIX.md and fails when this Markdown drifts. Run: python tools/render_roster.py [--check]"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs' / 'roster.json'
TARGET = ROOT / 'docs' / 'ROSTER.md'
KINDS = ['package', 'platform', 'model', 'workload', 'missing_family']


def cell(text):
    return str(text or '').replace('|', '\\|').replace('\n', ' ').strip()


def render(data):
    rows = data['rows']
    lines = ['# Representative roster and per-family gates', '',
             f"Generated from `docs/roster.json` (schema {data['schema_version']}, frozen {data['frozen']}) by `tools/render_roster.py`; do not edit by hand. {data['scope']}", '',
             '## Statuses', ''] + [f"- `{k}`: {v}" for k, v in data['statuses'].items()] + ['', '## Required versus deferred', '', data['required_deferred_rule'], '']
    counts = Counter((r['kind'], r['status']) for r in rows)
    lines += ['## Summary', '', '| Kind | Rows | qualified_bounded | partial | source_only | requested_missing | deferred_unverified |', '|---|---|---|---|---|---|---|']
    for kind in KINDS:
        sub = [r for r in rows if r['kind'] == kind]
        lines.append(f"| {kind} | {len(sub)} | " + ' | '.join(str(sum(r['status'] == s for r in sub)) for s in ('qualified_bounded', 'partial', 'source_only', 'requested_missing', 'deferred_unverified')) + ' |')
    lines.append('')
    for kind in KINDS:
        sub = [r for r in rows if r['kind'] == kind]
        if not sub:
            continue
        lines += [f"## {kind.replace('_', ' ').title()} rows", '', '| Family | Name | Identity | Status | Required/deferred | Evidence | Plan items | Gate | Source |', '|---|---|---|---|---|---|---|---|---|']
        for r in sub:
            lines.append(f"| {cell(r['family'])} | {cell(r['name'])} | `{cell(r['identity'])}` | {r['status']} | {r.get('required_or_deferred', 'owner_call')} | {', '.join(r.get('evidence_ids', [])) or '-'} | {', '.join(r.get('plan_items', [])) or '-'} | {cell(r.get('gate'))} | {cell(r.get('source'))} |")
        lines.append('')
    lines += ['## Families without a roster row', '']
    missing = data.get('plan_items_without_rows', [])
    lines += ([f"- {m['id']}: {m['reason']}" for m in missing] if missing else ['- none: every plan item is served by at least one roster row.'])
    lines += ['', '## Reading this roster', '', 'A row is not a claim of universal coverage. `qualified_bounded` means the named identity executed with recorded evidence for the linked plan items only; new source, a new build or a different device never inherits it. '
              'Closing a family requires every `required` row of that family to be `qualified_bounded` at the family\'s acceptance layer and the remaining condition in `docs/progress.json` to be met; `deferred` rows are retained as unverified and are excluded from closure, not counted as passing.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = render(json.loads(SOURCE.read_text(encoding='utf-8')))
    if args.check:
        current = TARGET.read_text(encoding='utf-8').replace('\r\n', '\n') if TARGET.exists() else ''
        if current != text:
            raise SystemExit('docs/ROSTER.md is out of date; run python tools/render_roster.py')
        print('ROSTER.md matches roster.json')
        return
    TARGET.write_text(text, encoding='utf-8')
    print(f'wrote {TARGET} ({len(text.splitlines())} lines)')


if __name__ == '__main__':
    main()
