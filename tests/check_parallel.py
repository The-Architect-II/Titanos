"""Replica lifecycle and real parallel-mode qualification using a local chat model.

Run on two GPUs: python tests/check_parallel.py --model /checkpoint --gpus 0 1 --report gpu.json
No GPU or inaccessible distributed backend is recorded as untested/blocked, never passed.
Use --require-all on a release qualification host to fail for any missing mode.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
import tempfile
import threading
import time

import httpx
import uvicorn
from pydantic import ValidationError

from titanos.config import Config, ModelSpec, WorkerSpec, write_config
from titanos.server import create_app


@contextmanager
def serving(config):
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(config, output_dir=Path(tmp))
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
        thread.start()
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=600, trust_env=False) as client:
            for _ in range(300):
                if server.started:
                    break
                time.sleep(.02)
            assert server.started, 'server failed to start'
            try:
                yield client, app.state.engine
            finally:
                server.should_exit = True
                thread.join(timeout=30)
                assert not thread.is_alive(), 'server did not shut down'
                sock.close()


def post(client, route, body):
    result = client.post(route, json=body)
    assert result.status_code == 200, (result.status_code, result.text)
    return result.json()


def stream_and_disconnect(client, engine, body, worker=None):
    chunks = []
    streamed = client.post('/v1/chat/completions', json={**body, 'stream': True})
    assert streamed.status_code == 200, streamed.text
    for line in streamed.text.splitlines():
        if line.startswith('data: ') and line != 'data: [DONE]':
            item = json.loads(line[6:])
            assert 'error' not in item, item
            if item.get('choices'):
                chunks.append(item['choices'][0]['delta'].get('content', ''))
    expected = post(client, '/v1/chat/completions', body)['choices'][0]['message']['content']
    assert ''.join(chunks) == expected
    with client.stream('POST', '/v1/chat/completions', json={**body,
            'messages': [{'role': 'user', 'content': 'Write an extremely long essay about computing.'}],
            'stream': True, 'max_tokens': 1000, 'worker': worker}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith('data: ') and 'content' in line:
                break
    started = time.monotonic()
    post(client, '/api/models/unload', {'model': 'chat', 'worker': worker})
    assert time.monotonic() - started < 20, 'disconnect did not release the model promptly'
    assert post(client, '/v1/chat/completions', body)['choices'][0]['message']['content'] == expected
    return expected


def boundary_checks(model):
    from types import SimpleNamespace
    from titanos.parallel import validate_plan
    plan = {'layers.*.self_attn.q_proj': 'colwise', 'layers.*.self_attn.k_proj': 'colwise'}
    assert validate_plan(SimpleNamespace(base_model_tp_plan=plan, num_attention_heads=4, num_key_value_heads=2), 2)
    try:
        validate_plan(SimpleNamespace(base_model_tp_plan=plan, num_attention_heads=9, num_key_value_heads=3), 2)
        raise AssertionError('incompatible attention heads accepted')
    except ValueError:
        pass
    for values in [dict(models=['chat'], gpus=['0', '1']),
                   dict(models=['chat'], gpus=['0'], mode='tensor_parallel'),
                   dict(models=['chat'], mode='model_sharding')]:
        try:
            WorkerSpec(**values)
            raise AssertionError('invalid GPU mode accepted')
        except ValidationError:
            pass
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'models.json'
        cfg = Config(models={'chat': model})
        write_config(path, cfg)
        original = path.read_bytes()
        cfg.workers['bad'] = WorkerSpec(models=['absent'])
        try:
            write_config(path, cfg)
            raise AssertionError('dangling worker saved')
        except ValidationError:
            assert path.read_bytes() == original


def replica_check(model, gpus):
    cfg = Config(models={'chat': model}, workers={name: WorkerSpec(models=['chat'],
        device='cpu', gpus=[gpus[i]] if gpus else [], profile=bool(gpus), threads=2)
        for i, name in enumerate(['a', 'b'])})
    body = {'model': 'chat', 'messages': [{'role': 'user', 'content': 'Say hello.'}], 'temperature': 0, 'max_tokens': 16}
    with serving(cfg) as (client, engine):
        post(client, '/api/models/load', {'model': 'chat'})
        before = client.get('/api/workers').json()['data']
        assert len({w['pid'] for w in before}) == 2
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(post, client, '/v1/chat/completions', {**body, 'max_tokens': 128,
                'messages': [{'role': 'user', 'content': 'Write a long essay about computing.'}]}) for _ in range(2)]
            overlap = False
            while not all(f.done() for f in futures):
                overlap |= sum(w['busy'] for w in engine.worker_status()) == 2
                time.sleep(.01)
            results = [f.result() for f in futures]
            assert overlap and all(r['usage']['completion_tokens'] > 0 for r in results)
        state = client.get('/api/workers').json()['data']
        assert all(w['completed_requests'] >= 1 for w in state)
        if gpus:
            assert all(w['observed_gpu_indices'] == [0] for w in state), state
        text = stream_and_disconnect(client, engine, body, 'a')
        assert client.post('/v1/chat/completions', json={**body, 'worker': 'absent'}).status_code == 404
        # Kill a process during a real request. No automatic retry can hide failure.
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(client.post, '/v1/chat/completions', json={**body, 'worker': 'a', 'max_tokens': 1000,
                'messages': [{'role': 'user', 'content': 'Write an extremely long essay about computing.'}]})
            for _ in range(500):
                if engine.workers[0].busy:
                    break
                time.sleep(.01)
            assert engine.workers[0].busy
            engine.workers[0].process.kill()
            failed = future.result()
            assert failed.status_code == 503, failed.text
        post(client, '/v1/chat/completions', {**body, 'worker': 'a'})
        after = client.get('/api/workers').json()['data']
        assert after[0]['pid'] != before[0]['pid'] and after[1]['pid'] == before[1]['pid']
        assert client.get('/healthz').status_code == 200
    assert not mp.active_children(), mp.active_children()
    return {'support': True, 'passed': True, 'checks': ['two live PIDs', 'overlapping requests',
        'both replicas used', 'SSE matches response', 'socket disconnect cancellation',
        'worker targeting', 'crash fails without replay', 'next request restarts worker', 'clean shutdown'],
        'text': text, 'workers': state}


def parallel_check(model, mode, gpus):
    cfg = Config(models={'chat': model}, workers={'group': WorkerSpec(models=['chat'],
        device='cpu', gpus=gpus, mode=mode, threads=2, profile=True)})
    body = {'model': 'chat', 'messages': [{'role': 'user', 'content': 'Say hello.'}], 'temperature': 0, 'max_tokens': 16}
    with serving(cfg) as (client, engine):
        loaded = client.post('/api/models/load', json={'model': 'chat'})
        if loaded.status_code != 200:
            assert not engine.catalog(True)['data']
            message = loaded.text
            blocked = 'Operation not permitted' in message
            return {'support': 'candidate', 'passed': None if blocked else False,
                    'status': 'environment_blocked' if blocked else 'failed', 'error': message}
        result = post(client, '/v1/chat/completions', body)
        text = result['choices'][0]['message']['content']
        assert text
        details = post(client, '/api/models/details', {'model': 'chat'})
        runtime = details['process']['runtime']
        if mode == 'tensor_parallel':
            assert runtime['world_size'] == (len(gpus) or 2)
            assert all(r['tp_plan'] and r['completed_requests'] for r in runtime['ranks'])
            if gpus:
                assert {i for r in runtime['ranks'] for i in r['observed_gpu_indices']} == set(range(len(gpus)))
        elif gpus:
            assert details['process']['observed_gpu_indices'] == list(range(len(gpus)))
        stream_and_disconnect(client, engine, body, 'group')
        if mode == 'tensor_parallel':
            current = post(client, '/api/models/details', {'model': 'chat'})
            pid = current['process']['runtime']['ranks'][-1]['pid']
            os.kill(pid, signal.SIGTERM)
            failed = client.post('/v1/chat/completions', json=body)
            assert failed.status_code in {500, 503}, failed.text
            post(client, '/api/models/unload', {'model': 'chat'})
            post(client, '/v1/chat/completions', body)
    assert not mp.active_children(), mp.active_children()
    return {'support': True, 'passed': True, 'text': text, 'details': details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tp-model', type=Path, help='Optional distinct TP-compatible chat checkpoint; e.g. attention head counts divisible by two')
    parser.add_argument('--gpus', nargs='+', default=[])
    parser.add_argument('--mode', choices=['all', 'replicas', 'model_sharding', 'tensor_parallel'], default='all')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    assert not args.gpus or len(args.gpus) == 2, 'this qualification check uses exactly two GPUs'
    if args.gpus:
        import torch
        if not torch.version.hip:
            from titanos.devices import pin
            args.gpus = pin(args.gpus)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    model = ModelSpec(path=str(args.model.resolve()), task='chat', device='cpu')
    from titanos.hardware import discover
    import importlib.metadata
    from titanos import __version__
    report = {'code_version': __version__, 'hardware': discover(), 'model': model.model_dump(),
        'config_sha256': hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest(),
        'versions': {n: importlib.metadata.version(n) for n in ['titanos-engine', 'torch', 'transformers', 'accelerate']}, 'modes': {}}
    try:
        boundary_checks(model)
        for mode in (['replicas', 'model_sharding', 'tensor_parallel'] if args.mode == 'all' else [args.mode]):
            if mode == 'model_sharding' and not args.gpus:
                report['modes'][mode] = {'support': 'candidate', 'passed': None, 'status': 'requires_two_GPUs'}
                continue
            try:
                selected = model
                if mode == 'tensor_parallel' and args.tp_model:
                    selected = ModelSpec(path=str(args.tp_model.resolve()), task='chat', device='cpu')
                report['modes'][mode] = replica_check(selected, args.gpus) if mode == 'replicas' else parallel_check(selected, mode, args.gpus)
                report['modes'][mode]['model_path'] = selected.path
                report['modes'][mode]['model_config_sha256'] = hashlib.sha256((Path(selected.path) / 'config.json').read_bytes()).hexdigest()
            except Exception as exc:
                report['modes'][mode] = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
    finally:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    if any(v.get('passed') is False or (args.require_all and v.get('passed') is not True) for v in report['modes'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
