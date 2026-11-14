"""Real compiled-model HTTP checks. No accelerator availability is mocked as inference.

python tests/check_compiled.py --registry /path/models.json --onnx NAME --openvino NAME [--genai NAME]
"""
import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from titanos.compiled import onnx_session, tensor_payload, runtime_check
from titanos.config import Config, ModelSpec, WorkerSpec, read_config
from titanos.server import create_app


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--onnx', required=True)
    p.add_argument('--openvino', required=True)
    p.add_argument('--genai')
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    report = {'passed': [], 'hardware': {}}
    for backend in ['onnx', 'openvino']:
        result = runtime_check(backend, check=True)
        assert result['check']['passed'], result
        report['hardware'][backend] = result
    # Provider rejection contract; this is not a native Core ML hardware test.
    with patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider']):
        try:
            onnx_session('unused.onnx', 'coreml')
            raise AssertionError('provider absence was silently accepted')
        except ValueError as exc:
            assert 'not installed' in str(exc)
    for item in [{'dtype': 'int8', 'shape': [1], 'data': [256]},
                 {'dtype': 'int64', 'shape': [1], 'data': [1.5]},
                 {'dtype': 'float32', 'shape': [2], 'data': [1]},
                 {'dtype': 'float32', 'shape': [1], 'data': [float('nan')]}]:
        try:
            tensor_payload(item)
            raise AssertionError('invalid tensor accepted')
        except ValueError:
            pass
    cfg = read_config(args.registry)
    graph = cfg.models[args.onnx]
    cfg.models['graph'] = ModelSpec(path=graph.path, task='tensor', backend='onnx', graph=graph.graph, device='cpu')
    with tempfile.TemporaryDirectory() as tmp:
        with TestClient(create_app(cfg, api_key='test-key', output_dir=Path(tmp)), base_url='http://127.0.0.1') as c:
            assert c.post('/api/models/details', json={'model': args.onnx}).status_code == 401
            c.headers['Authorization'] = 'Bearer test-key'
            vectors = []
            for name in [args.onnx, args.openvino]:
                result = c.post('/v1/embeddings', json={'model': name, 'input': ['Hello world', 'A separate sentence']})
                assert result.status_code == 200, result.text
                rows = [d['embedding'] for d in result.json()['data']]
                assert len(rows) == 2
                np.testing.assert_allclose(np.linalg.norm(rows, axis=1), [1, 1], atol=1e-5)
                vectors.append(rows)
                detail = c.post('/api/models/details', json={'model': name})
                assert detail.status_code == 200 and detail.json()['backend'] == cfg.models[name].backend
                assert c.post('/v1/chat/completions', json={'model': name, 'messages': [{'role':'user','content':'hello'}]}).status_code == 400
            np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-4, rtol=1e-4)
            report['passed'].append('ONNX and OpenVINO HTTP embeddings agree, are normalized, swap correctly, and reject wrong tasks')
            details = c.post('/api/models/details', json={'model': 'graph'}).json()
            inputs = {name: {'dtype': item['dtype'], 'shape': item['shape'],
                             'data': [1] * int(np.prod(item['shape']))} for name, item in details['inputs'].items()}
            result = c.post('/api/infer', json={'model': 'graph', 'inputs': inputs})
            assert result.status_code == 200, result.text
            assert 'last_hidden_state' in result.json()['outputs']
            first = next(iter(inputs))
            inputs[first]['data'][0] = 1.5
            assert c.post('/api/infer', json={'model': 'graph', 'inputs': inputs}).status_code == 400
            report['passed'].append('typed graph API executes real model and rejects fractional integer input')
            if args.genai:
                body = {'model': args.genai, 'messages': [{'role': 'user', 'content': 'Say hello.'}], 'max_tokens': 16, 'temperature': 0}
                r = c.post('/v1/chat/completions', json=body)
                assert r.status_code == 200, r.text
                result = r.json()
                report['chat_text'] = result['choices'][0]['message']['content']
                assert result['usage']['completion_tokens'] > 0 and report['chat_text']
                streamed = c.post('/v1/chat/completions', json={**body, 'stream': True})
                assert streamed.status_code == 200, streamed.text
                chunks = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
                text = ''.join(x['choices'][0]['delta'].get('content', '') for x in chunks if x.get('choices'))
                assert text == report['chat_text'], (text, report['chat_text'])
                # A stop spanning chunks must not leak into streamed output.
                stop = report['chat_text'][:3]
                r = c.post('/v1/chat/completions', json={**body, 'stop': stop})
                assert r.status_code == 200 and r.json()['choices'][0]['message']['content'] == '', r.text
                again = c.post('/v1/chat/completions', json=body).json()
                assert again['choices'][0]['message']['content'] == report['chat_text']
                report['passed'].append('OpenVINO GenAI real chat, matching SSE, stop handling and independent requests')
            for name in [args.onnx, args.openvino] + ([args.genai] if args.genai else []):
                assert c.post('/api/models/unload', json={'model': name}).status_code == 200
    report['passed'].append('missing provider and invalid tensor contracts passed; no NPU hardware result inferred')
    from check_parallel import serving, stream_and_disconnect
    if args.genai:
        with serving(Config(models={'chat': cfg.models[args.genai]})) as (client, engine):
            stream_and_disconnect(client, engine, {'model': 'chat', 'messages': [{'role': 'user', 'content': 'Say hello.'}],
                'max_tokens': 16, 'temperature': 0})
        report['passed'].append('OpenVINO GenAI real socket disconnect releases model and next request succeeds')
    from concurrent.futures import ThreadPoolExecutor
    workers = {name: WorkerSpec(models=[name], threads=2) for name in [args.onnx, args.openvino]}
    with serving(Config(models={name: cfg.models[name] for name in workers}, workers=workers)) as (client, engine):
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda name: client.post('/v1/embeddings', json={'model': name, 'input': 'Hello world'}), workers))
        assert all(r.status_code == 200 for r in results), [r.text for r in results]
        status = client.get('/api/workers').json()['data']
        assert len({w['pid'] for w in status}) == 2
        assert {w['current_model'] for w in status} == set(workers)
    report['passed'].append('different ONNX and OpenVINO models remain loaded in independent workers')
    report['success'] = True
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
