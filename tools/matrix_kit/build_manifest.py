"""Build the per-host run manifest for the matrix kit.

Inputs:
  docs/execution_matrix.json            the rows (model_id, task, backend, identity) that define the matrix
  <model_sources.json>                  this host's local artifact locations: {model_id: {path, present, identity_check, spec: {...}}}
Output:
  <manifest.json>                       one entry per matrix row: runnable (path present) or unavailable (with the reason)

`spec` in model_sources may carry ModelSpec fields the row needs beyond path/task/backend (allow_pickle, trust_remote_code,
graph, native, voices, load_options, defaults, dtype, and for music the runtime weight placement auto/offload/resident). The spec
passes through unchanged into the generated ModelSpec; it carries no admission input (no byte estimate or basis lowers a floor:
run_cells derives contained admission from an explicit placement 'resident' and an enforced owned-tree cap alone). Nothing here is
machine-independent: model_sources.json is written per host and stays out of the repository.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', required=True, type=Path, help='model_sources.json for this host')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--matrix', type=Path, default=ROOT / 'docs' / 'execution_matrix.json')
    a = ap.parse_args()
    matrix = json.loads(a.matrix.read_text(encoding='utf-8'))
    sources = json.loads(a.sources.read_text(encoding='utf-8'))
    entries, unavailable = [], []
    for r in matrix['rows']:
        src = sources.get(r['model_id']) or {}
        entry = {'model_id': r['model_id'], 'display_name': r['display_name'], 'task': r['task'], 'backend': r['backend'],
                 'modality_family': r['modality_family'], 'matrix_identity': r['identity']}
        if src.get('present') and src.get('path') and Path(src['path']).exists():
            entry.update(path=src['path'], spec=src.get('spec') or {}, identity_check=src.get('identity_check') or {'method': 'none', 'matches': 'unknown', 'detail': 'no check recorded'},
                         bytes=src.get('bytes'), request=src.get('request') or {}, env=src.get('env') or {}, home_files=src.get('home_files') or {})
            entries.append(entry)
        else:
            entry['reason'] = src.get('acquisition_if_absent') or ('path missing on this host: ' + str(src.get('path'))) if src else 'no source record for this model on this host'
            unavailable.append(entry)
    out = {'schema_version': 1, 'matrix_built_utc': matrix.get('built_utc'), 'models': entries, 'unavailable': unavailable,
           'counts': {'runnable': len(entries), 'unavailable': len(unavailable)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out['counts']))


if __name__ == '__main__':
    main()
