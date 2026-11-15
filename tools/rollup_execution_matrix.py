"""Derived fields of docs/execution_matrix.json, computed from the matrix itself so they need no rebuild from the private reader input.

- `slot_rollup`: per platform slot (spec order), the cell count of every state and the sorted model ids whose cell is a pass upheld
  by a verifier (`verified` exactly true): the list an external "Tested" mark may cite.
- `evidence_build` on every cell: the Titanos source commits and versions the cell's own `package_or_runtime` record names for an
  executed cell (pass, fail, partial), as {"commits": [...], "versions": [...]}; null when the record names neither, and always null
  for untried and na cells. Nothing is guessed: a commit is a hex id written after `source`, `HEAD` or in a `titanos-<hex>` tree
  name; abbreviations of one id merge to the longest spelling; several distinct ids are all listed. A version is a Titanos
  development version (`0.5.27.dev0+assetjobs1`, `0.5.25.dev0`; a dated nightly such as `2.10.0.dev20250101` is not one) or a release written after `Titanos`/`titanos-engine`.
- `built_from_commit`: the git commit tools/build_execution_matrix.py ran from (with `-dirty` when the tree had changes). This pass
  keeps the value it finds and writes null when there is none (the committed matrix was built on 2026-09-28 without recording it).

Usage: python tools/rollup_execution_matrix.py [--check]   (the builder applies derive() before it writes)"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'docs' / 'execution_matrix.json'
EXECUTED = ('pass', 'fail', 'partial')
COMMIT = re.compile(r'\btitanos-([0-9a-f]{7,40}) source\b|\b(?:source|HEAD) (?:HEAD )?([0-9a-f]{7,40})\b')
VERSION = re.compile(r'\b(\d+\.\d+\.\d+\.dev\d{1,4}(?:\+[A-Za-z0-9]+)?)(?![\w.+])|\b(?:Titanos|titanos-engine)[ -](\d+\.\d+\.\d+)(?![\w.+])')
KEEP = object()


def merge_prefixes(ids):
    """Distinct ids with every abbreviation folded into the longest spelling that extends it."""
    out = []
    for i in sorted(set(ids), key=len, reverse=True):
        if not any(o.startswith(i) for o in out):
            out.append(i)
    return sorted(out)


def evidence_build(cell):
    if cell['state'] not in EXECUTED:
        return None
    text = cell.get('package_or_runtime') or ''
    commits = merge_prefixes(a or b for a, b in COMMIT.findall(text))
    versions = sorted({a or b for a, b in VERSION.findall(text)})
    return {'commits': commits, 'versions': versions} if commits or versions else None


def slot_rollup(data):
    states = list(data['cell_states'])
    out = []
    for slot in data['slots']:
        cells = [c for c in data['cells'] if c['slot'] == slot['id']]
        out.append({'slot': slot['id'], 'counts': {s: sum(c['state'] == s for c in cells) for s in states},
                    'verified_pass_models': sorted(c['model_id'] for c in cells if c['state'] == 'pass' and c.get('verified') is True)})
    return out


def derive(data, built_from_commit=KEEP):
    """A copy of the matrix with the derived fields in place; idempotent, key order stable (built_from_commit after built_utc,
    slot_rollup after counts, evidence_build last in each cell)."""
    commit = data.get('built_from_commit') if built_from_commit is KEEP else built_from_commit
    out = {}
    for key, value in data.items():
        if key in ('built_from_commit', 'slot_rollup'):
            continue
        if key == 'cells':
            value = [{**{k: v for k, v in c.items() if k != 'evidence_build'}, 'evidence_build': evidence_build(c)} for c in value]
        out[key] = value
        if key == 'built_utc':
            out['built_from_commit'] = commit
    out['slot_rollup'] = slot_rollup(out)
    return out


def dumps(data):
    return json.dumps(data, indent=1, ensure_ascii=False) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='fail when docs/execution_matrix.json lacks the current derived fields')
    a = ap.parse_args()
    data = json.loads(SRC.read_text(encoding='utf-8'))
    derived = derive(data)
    if a.check:
        if derived != data:
            raise SystemExit('docs/execution_matrix.json derived fields are out of date; run python tools/rollup_execution_matrix.py')
        print('execution matrix derived fields match')
        return
    SRC.write_text(dumps(derived), encoding='utf-8')
    print(json.dumps({'slots': len(derived['slot_rollup']), 'cells_with_build': sum(1 for c in derived['cells'] if c['evidence_build']),
                      'built_from_commit': derived['built_from_commit']}))


if __name__ == '__main__':
    main()
