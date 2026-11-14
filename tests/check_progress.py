"""docs/PROGRESS.md is a rendered view of docs/progress.json (the acceptance checklist) and is bound to the evidence records.

Every checklist item renders exactly one row and no row lacks an item; each row's box, status and layer equal the item's; the
header counts equal the items recounted by status and by acceptance layer, and progress.json's own totals and layer_totals agree
with that recount; statuses are in the vocabulary; every cited evidence id is a row of docs/CAPABILITY_MATRIX.md; every matrix id
no item cites lies in the header's named range and is cited by docs/execution_matrix.json, so the header's description of those
rows stays true; source_version equals the pyproject.toml version; the committed bytes equal the renderer's output in the file's
own newline style (tools/render_progress.py --check); and no lab-identifier detection class of tools/public_export.json, no
narrative class and no private-map name or word hits the page (file, line and class are reported, never the matched text).
No model, no network, no process."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import lab_paths  # noqa: E402
import render_progress as renderer  # noqa: E402

PROGRESS = ROOT / 'docs' / 'progress.json'
DOC = ROOT / 'docs' / 'PROGRESS.md'
MATRIX = ROOT / 'docs' / 'CAPABILITY_MATRIX.md'
EXECUTION_MATRIX = ROOT / 'docs' / 'execution_matrix.json'
PYPROJECT = ROOT / 'pyproject.toml'
POLICY = json.loads((ROOT / 'tools' / 'public_export.json').read_text(encoding='utf-8'))
STATUSES = {'validated_bounded', 'partial', 'active', 'planned', 'missing'}
LAYERS = {'ENGINE_CONTRACT', 'MODEL_QUALITY', 'HUMAN_PRESENTATION'}
ROW = re.compile(r'^- \[(?P<box>[ x])\] \*\*(?P<id>[A-Z]+-\d+) — (?P<status>[a-z_]+) — (?P<layer>[A-Z_]+)\*\*: ')
HEADER = re.compile(r'\*\*(?P<items>\d+) items: (?P<closed>\d+) closed within recorded scope, (?P<open>\d+) open\*\* '
                    r'\((?P<partial>\d+) partial, (?P<missing>\d+) missing, (?P<active>\d+) active, (?P<planned>\d+) planned\)\. '
                    r'By primary acceptance layer: (?P<layers>.*?)\. Counts combine')
LAYER = re.compile(r'(?P<layer>[A-Z_]+) (?P<items>\d+)(?: rows)? \((?P<closed>\d+) closed\)')


def whole_word(names):
    """A whole-word, ASCII-case-insensitive pattern over `names`, or None when there are none (a public tree has no private map)."""
    names = sorted(set(names), key=len, reverse=True)
    return re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in names) + r')(?![A-Za-z0-9])', re.IGNORECASE | re.ASCII) if names else None


def identifier_hits(page):
    """[(line, class)] for every detection_patterns and narrative_patterns class of the policy and for the private map's names and
    words; the matched text is never returned."""
    classes = {name: re.compile(pattern) for name, pattern in POLICY['detection_patterns'].items()}
    classes.update({name: re.compile(pattern) for name, pattern in POLICY.get('narrative_patterns', {}).items()})
    for label, pattern in (('private_denylist', whole_word(lab_paths.private_names())), ('coordination_narrative', whole_word(lab_paths.narrative_words()))):
        if pattern is not None:
            classes[label] = pattern
    hits = []
    for label, pattern in classes.items():
        for match in pattern.finditer(page):
            hits.append((page.count('\n', 0, match.start()) + 1, label))
    return sorted(hits), len(classes)


def main():
    data = json.loads(PROGRESS.read_text(encoding='utf-8'))
    items = data['items']
    ids = [i['id'] for i in items]
    existing = DOC.read_bytes() if DOC.exists() else b''
    page = existing.decode('utf-8')
    lines = page.replace('\r\n', '\n').split('\n')
    matrix_ids = set(re.findall(r'\bE-\d{3}\b', MATRIX.read_text(encoding='utf-8')))
    executed_ids = set(re.findall(r'\bE-\d{3}\b', EXECUTION_MATRIX.read_text(encoding='utf-8')))
    failures = []

    # the ledger itself: unique ids, vocabulary, layers, closed flag consistent with status, evidence ids exist
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        failures.append(f'progress.json repeats ids {duplicates}')
    for i in items:
        if i['status'] not in STATUSES:
            failures.append(f"{i['id']}: status {i['status']!r} is not in the vocabulary")
        if i['acceptance_layer'] not in LAYERS:
            failures.append(f"{i['id']}: acceptance_layer {i['acceptance_layer']!r} is not in the vocabulary")
        if bool(i['closed']) != (i['status'] == 'validated_bounded'):
            failures.append(f"{i['id']}: closed={i['closed']} disagrees with status {i['status']}")
        for e in i.get('evidence_ids', []):
            if e not in matrix_ids:
                failures.append(f"{i['id']}: evidence {e} is not a CAPABILITY_MATRIX row")
    closed = sum(1 for i in items if i['closed'])
    by_status = Counter(i['status'] for i in items)
    by_layer = {}
    for i in items:
        n, c = by_layer.get(i['acceptance_layer'], (0, 0))
        by_layer[i['acceptance_layer']] = (n + 1, c + (1 if i['closed'] else 0))
    if data['totals'] != {'items': len(items), 'closed': closed, 'open': len(items) - closed}:
        failures.append(f"progress.json totals {data['totals']} differ from the recount {len(items)}/{closed}/{len(items) - closed}")
    for layer, (n, c) in by_layer.items():
        if data['layer_totals'].get(layer) != {'items': n, 'closed': c, 'open': n - c}:
            failures.append(f"progress.json layer_totals[{layer}] differs from the recount {n} items, {c} closed")
    version = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding='utf-8'), re.M)
    if not version or version.group(1) != data['source_version']:
        failures.append(f"progress.json source_version {data['source_version']!r} is not the pyproject.toml version {version and version.group(1)!r}")

    # the uncited matrix ids the header describes: inside the renderer's named range and execution-matrix evidence
    uncited = renderer.uncited_ids(data)
    first, last, _ = renderer.RECENT_EVIDENCE
    for e in uncited:
        if not first <= e <= last:
            failures.append(f'{e} is cited by no checklist item and lies outside the header range {first}..{last}; cite it or update tools/render_progress.py RECENT_EVIDENCE')
        if e not in executed_ids:
            failures.append(f'{e} is cited by no checklist item and not by docs/execution_matrix.json, so the header cannot call it {renderer.UNCITED_KIND}')

    # the page: one row per item, in the JSON's words; header counts equal the recount
    rows = [ROW.match(line) for line in lines]
    rows = [m for m in rows if m]
    row_ids = [m['id'] for m in rows]
    row_counts = Counter(row_ids)
    for i in items:
        if row_counts.get(i['id'], 0) != 1:
            failures.append(f"{i['id']}: {row_counts.get(i['id'], 0)} rows in PROGRESS.md, expected exactly one")
    for extra in sorted(set(row_ids) - set(ids)):
        failures.append(f'PROGRESS.md row {extra} has no progress.json item')
    by_id = {i['id']: i for i in items}
    for m in rows:
        item = by_id.get(m['id'])
        if item and ((m['box'] == 'x') != bool(item['closed']) or m['status'] != item['status'] or m['layer'] != item['acceptance_layer']):
            failures.append(f"{m['id']}: row box/status/layer differ from progress.json")
    header = HEADER.search(page)
    if not header:
        failures.append('PROGRESS.md header does not carry the item counts in the expected form')
    else:
        want = {'items': len(items), 'closed': closed, 'open': len(items) - closed,
                'partial': by_status['partial'], 'missing': by_status['missing'], 'active': by_status['active'], 'planned': by_status['planned']}
        got = {k: int(header[k]) for k in want}
        if got != want:
            failures.append(f'PROGRESS.md header counts {got} differ from progress.json {want}')
        header_layers = {m['layer']: (int(m['items']), int(m['closed'])) for m in LAYER.finditer(header['layers'])}
        if header_layers != by_layer:
            failures.append(f'PROGRESS.md header layer counts {header_layers} differ from progress.json {by_layer}')
    if f"Updated **{data['updated']}** for source `{data['source_version']}`" not in page:
        failures.append('PROGRESS.md header does not name the progress.json updated date and source_version')

    # the fixed point: exact bytes in the file's own newline style
    style = renderer.newline_style(existing)
    if renderer.rendered_bytes(renderer.render(data), style) != existing:
        failures.append('docs/PROGRESS.md is out of date; run python tools/render_progress.py')

    # no lab identifier class hits the page
    hits, classes = identifier_hits(page)
    for line, label in hits[:40]:
        failures.append(f'docs/PROGRESS.md:{line} {label}')

    assert not failures, '\n'.join(failures[:40])
    print(json.dumps({'passed': True, 'items': len(items), 'closed': closed, 'open': len(items) - closed, 'by_status': dict(by_status),
                      'by_layer': {k: {'items': v[0], 'closed': v[1]} for k, v in by_layer.items()}, 'rows': len(rows),
                      'categories': len(dict.fromkeys(i['category'] for i in items)),
                      'evidence_ids_cited': len({e for i in items for e in i.get('evidence_ids', [])}), 'matrix_ids_uncited': len(uncited),
                      'identifier_classes_scanned': classes, 'newline': 'CRLF' if style == '\r\n' else 'LF',
                      'scope': 'ledger rendering and bindings only; no execution qualification'}))


if __name__ == '__main__':
    main()
