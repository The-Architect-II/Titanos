"""Render docs/PROGRESS.md from docs/progress.json (the development acceptance checklist).

Every checklist row is a faithful view of one JSON item and the header counts are recounted from the items, so the page cannot
say something the ledger does not. tests/check_progress.py asserts the bindings (one row per item, header counts, evidence ids
present in docs/CAPABILITY_MATRIX.md, no lab-identifier class) and fails when this Markdown drifts.
Run: python tools/render_progress.py [--check]

The file keeps the newline style it already has on disk (CRLF on a Windows checkout, LF elsewhere; a missing file is written
LF). --check compares the exact bytes in that style and exits 1 on drift, so a hand edit, a stale render or a mixed newline
file all fail."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs' / 'progress.json'
MATRIX = ROOT / 'docs' / 'CAPABILITY_MATRIX.md'
TARGET = ROOT / 'docs' / 'PROGRESS.md'
EVIDENCE_ID = re.compile(r'\bE-\d{3}\b')
OPEN_STATUSES = ('partial', 'missing', 'active', 'planned')
LAYERS = ('ENGINE_CONTRACT', 'MODEL_QUALITY', 'HUMAN_PRESENTATION')
# Header facts the JSON does not carry yet. The source commit the ledger was reconciled against is read from
# data['source_commit'] when progress.json gains that field; until then this constant is the value. RECENT_EVIDENCE names the
# evidence range the dated paragraph explains (first id, last id, what those rows are); UNCITED_KIND describes the rows of that
# range no checklist item cites. tests/check_progress.py asserts every uncited id lies inside the range so the wording stays true.
SOURCE_COMMIT = '0e8ee71'
RECENT_EVIDENCE = ('E-090', 'E-106', 'the matrix-kit campaigns, the music gate and the music-inclusive candidates')
UNCITED_KIND = 'execution-matrix evidence'


def text(value):
    return str(value or '').replace('\r\n', ' ').replace('\n', ' ').strip()


def id_ranges(ids):
    """'E-093 to E-105, E-107' for a set of evidence ids: consecutive numbers collapse into a range."""
    numbers = sorted(int(i[2:]) for i in ids)
    out, k = [], 0
    while k < len(numbers):
        j = k
        while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:
            j += 1
        out.append(f'E-{numbers[k]:03d}' if k == j else f'E-{numbers[k]:03d} to E-{numbers[j]:03d}')
        k = j + 1
    return ', '.join(out)


def matrix_ids(matrix_text=None):
    """Every evidence id that appears in docs/CAPABILITY_MATRIX.md."""
    return set(EVIDENCE_ID.findall(MATRIX.read_text(encoding='utf-8') if matrix_text is None else matrix_text))


def uncited_ids(data, matrix_text=None):
    """Evidence ids present in the capability matrix that no checklist item cites."""
    cited = {e for item in data['items'] for e in item.get('evidence_ids', [])}
    return sorted(matrix_ids(matrix_text) - cited)


def render(data, matrix_text=None):
    """The Markdown page (LF newlines, trailing newline) for a progress.json document."""
    items = data['items']
    closed = sum(1 for i in items if i['closed'])
    by_status = Counter(i['status'] for i in items)
    layers = {}
    for i in items:
        n, c = layers.get(i['acceptance_layer'], (0, 0))
        layers[i['acceptance_layer']] = (n + 1, c + (1 if i['closed'] else 0))
    layer_order = list(LAYERS) + [layer for layer in layers if layer not in LAYERS]
    layer_parts = [f"{layer} {layers.get(layer, (0, 0))[0]}{' rows' if k == 0 else ''} ({layers.get(layer, (0, 0))[1]} closed)"
                   for k, layer in enumerate(layer_order)]
    first, last, about = RECENT_EVIDENCE
    recent_citing = [i['id'] for i in items if any(first <= e <= last for e in i.get('evidence_ids', []))]
    uncited = uncited_ids(data, matrix_text)
    commit = data.get('source_commit') or SOURCE_COMMIT

    lines = ['# Progress checklist', '',
             f"Updated **{data['updated']}** for source `{data['source_version']}` (commit `{commit}`). The rows below are a rendered view of "
             f"[progress.json](progress.json); edit the JSON, not this page. **{len(items)} items: {closed} closed within recorded scope, "
             f"{len(items) - closed} open** ({', '.join(f'{by_status[s]} {s}' for s in OPEN_STATUSES)}). By primary acceptance layer: "
             f"{', '.join(layer_parts)}. Counts combine separate layers and are not a percentage-complete figure.",
             '', '## How a row reads', '',
             '`- [x]` is closed within the linked evidence only; `- [ ]` is open. **ID — status — layer**: the condition. *Evidence*: the E-ids in '
             '[CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md) that bear on the row, followed by the row\'s qualification note. **Remaining:** what is still unproven. '
             '**Next:** the concrete next action. Statuses: `validated_bounded` (closed for the named model/package/platform/workload only), `partial` (some bounded '
             'evidence, condition not met), `active` (being worked), `planned` (not started), `missing` (no evidence yet).',
             '', '## Rules', '',
             'Stable ids are preserved. A row closes only with named model/runtime/package/device evidence, request and output identity, failure and cleanup results, '
             'and its remaining exclusions stated. Failed results are retained; a later pass never rewrites an earlier fail. A new binary, a new source commit or a '
             'different device inherits nothing. ENGINE_CONTRACT covers coherent intended-modality output, actual encoding/storage, controls, protocol, routing and '
             'lifecycle. MODEL_QUALITY retains factual, semantic and cosmetic outcomes. HUMAN_PRESENTATION covers media and human review. A coherent wrong answer or '
             'cosmetic image miss alone does not fail the engine; broken requested schema enforcement, controls or lifecycle still does.',
             '']
    dated = (f"Dated ledger changes (the 2026-09-22 reconciliation and every later continuation) are indexed in [history/README.md](history/README.md). "
             f"Evidence rows {first} to {last} ({about}) are cited by {', '.join(recent_citing) or 'no checklist row yet'}; ")
    if uncited:
        dated += f"rows {id_ranges(uncited)} are {UNCITED_KIND} not yet cited by any checklist row, and they change no closure. "
    else:
        dated += 'every evidence row of the capability matrix is cited by at least one checklist row. '
    dated += 'Every evidence id cited below exists in [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md); see [STATUS.md](../STATUS.md) for the summary.'
    lines.append(dated)

    for category in dict.fromkeys(i['category'] for i in items):
        lines += ['', f'## {category}', '']
        for i in items:
            if i['category'] != category:
                continue
            box = '[x]' if i['closed'] else '[ ]'
            parts = [f"- {box} **{i['id']} — {i['status']} — {i['acceptance_layer']}**: {text(i['condition'])}"]
            parts.append('Evidence: ' + (', '.join(i['evidence_ids']) + '.' if i.get('evidence_ids') else 'none yet.'))
            if i.get('qualification_note'):
                parts.append(text(i['qualification_note']))
            if i.get('remaining_condition'):
                parts.append('**Remaining:** ' + text(i['remaining_condition']))
            if i.get('next_action'):
                parts.append('**Next:** ' + text(i['next_action']))
            lines.append(' '.join(parts))
    return '\n'.join(lines) + '\n'


def newline_style(existing):
    """'\\r\\n' when the first line ending of the existing bytes is CRLF, else '\\n' (also for an empty or missing file)."""
    index = existing.find(b'\n')
    return '\r\n' if index > 0 and existing[index - 1:index] == b'\r' else '\n'


def rendered_bytes(page, style):
    return page.replace('\n', style).encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='compare docs/PROGRESS.md with a fresh render; exit 1 on drift, write nothing')
    args = parser.parse_args()
    page = render(json.loads(SOURCE.read_text(encoding='utf-8')))
    existing = TARGET.read_bytes() if TARGET.exists() else b''
    style = newline_style(existing)
    expected = rendered_bytes(page, style)
    label = 'CRLF' if style == '\r\n' else 'LF'
    if args.check:
        if existing != expected:
            raise SystemExit('docs/PROGRESS.md is out of date; run python tools/render_progress.py')
        print(f'PROGRESS.md matches progress.json ({len(page.splitlines())} lines, {label})')
        return
    TARGET.write_bytes(expected)
    print(f'wrote {TARGET} ({len(page.splitlines())} lines, {label})')


if __name__ == '__main__':
    main()
