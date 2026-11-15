"""Run in an ONNX-only Titanos environment; prove serving does not depend on PyTorch."""
import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
import titanos
from titanos.config import Config, ModelSpec, WorkerSpec
from titanos.server import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True, help='Prepared embedding model directory')
    parser.add_argument('--graph', default='model.onnx')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    assert importlib.util.find_spec('torch') is None, 'Use the standalone ONNX runtime for this check'
    cfg = Config(models={'embed': ModelSpec(path=str(args.model.resolve()), graph=args.graph,
        task='embeddings', backend='onnx')}, workers={name: WorkerSpec(models=['embed'], threads=2) for name in ['a', 'b']})
    with tempfile.TemporaryDirectory() as tmp:
        with TestClient(create_app(cfg, output_dir=Path(tmp)), base_url='http://127.0.0.1') as client:
            assert client.post('/api/models/load', json={'model': 'embed'}).status_code == 200
            for worker in ['a', 'b']:
                result = client.post('/v1/embeddings', json={'model': 'embed', 'worker': worker, 'input': ['Hello', 'Different sentence']})
                assert result.status_code == 200, result.text
                assert len(result.json()['data']) == 2
                for row in result.json()['data']:
                    assert abs(sum(x*x for x in row['embedding'])-1) < 1e-5
            workers = client.get('/api/workers').json()['data']
            assert len({w['pid'] for w in workers}) == 2
    report = {'success': True, 'torch_installed': False, 'module': titanos.__file__, 'version': titanos.__version__,
        'checks': ['real ONNX embeddings from installed wheel', 'two independently targeted processes', 'normalized batches', 'shutdown'],
        'workers': workers}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: real ONNX embeddings, two workers, no PyTorch')


if __name__ == '__main__':
    main()
