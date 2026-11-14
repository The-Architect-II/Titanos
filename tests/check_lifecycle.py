"""Control-path regressions with fake inference; no model or accelerator claims."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from titanos.config import Config, ModelSpec, WorkerSpec, read_config, update_config, write_config
from titanos.engine import Engine
from titanos.server import create_app


class FakeModel:
    closed = []

    def __init__(self, spec):
        self.spec = spec

    def infer(self, task, data, emit=None, cancel=None):
        if task == 'speech':
            return b'fake-wave'
        text = 'reply:' + self.spec.task
        if emit:
            emit(text)
        return {'text': text, 'finish_reason': 'stop',
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}

    def details(self):
        return {'backend': 'test-double', 'task': self.spec.task}

    def close(self):
        self.closed.append(self.spec.task)


class FakeProcess:
    def is_alive(self):
        return True


class FakeWorker:
    """Keep scheduling and model lifecycle real; replace child-process transport."""

    def __init__(self, name, assignment, config, settings=None):
        self.name, self.assignment = name, assignment
        self.engine = Engine(Config(models={n: config.models[n] for n in assignment.models}))
        self.process = self.cancel = None
        self.busy, self.last_used = False, 0
        self.meta = {'current_model': None}
        self.error = self.failed_model = None

    def call(self, command, name, task=None, data=None, emit=None, cancel=None):
        self.process = FakeProcess()
        if command == 'infer':
            result = self.engine.infer(name, task, {k: v for k, v in data.items() if k != 'worker'}, emit, cancel)
        else:
            result = getattr(self.engine, command)(name)
        self.meta['current_model'] = self.engine.current
        return result

    def stop(self):
        self.engine.close()
        self.meta['current_model'] = None
        self.process = None


def mixed_lifecycle(root):
    cfg = Config(models={
        'chat': ModelSpec(path=str(root), task='chat'),
        'voice': ModelSpec(path=str(root), task='speech'),
        'other': ModelSpec(path=str(root), task='chat'),
    }, workers={'chat-cpu': WorkerSpec(models=['chat'])})
    FakeModel.closed = []
    with patch('titanos.backends.LoadedModel', FakeModel), patch('titanos.replicas.Worker', FakeWorker):
        app = create_app(cfg, output_dir=root / 'outputs')
        engine = app.state.engine
        with TestClient(app, base_url='http://127.0.0.1') as client:
            request = {'model': 'chat', 'worker': 'chat-cpu', 'messages': [{'role': 'user', 'content': 'hello'}]}
            assert client.post('/v1/chat/completions', json=request).json()['choices'][0]['message']['content'] == 'reply:chat'
            assert client.post('/v1/audio/speech', json={'model': 'voice', 'input': 'hello'}).content == b'fake-wave'
            assert {m['id'] for m in client.get('/api/models/loaded').json()['data']} == {'chat', 'voice'}
            assert client.post('/api/models/details', json={'model': 'voice'}).json()['task'] == 'speech'
            # An explicit mismatching worker must never silently run locally.
            wrong = client.post('/v1/audio/speech', json={'model': 'voice', 'input': 'hello', 'worker': 'chat-cpu'})
            assert wrong.status_code == 400 and 'worker' in wrong.json()['error']['message']
            request['worker'] = 'missing'
            assert client.post('/v1/chat/completions', json=request).status_code == 404
            assert client.post('/v1/audio/speech', json={'model': 'missing', 'input': 'hello'}).status_code == 404
            assert client.post('/v1/audio/speech', json={'model': 'chat', 'input': 'hello'}).status_code == 400
            assert client.post('/api/models/load', json={'model': 'other'}).status_code == 200
            assert {m['id'] for m in client.get('/api/models/loaded').json()['data']} == {'chat', 'other'}
            assert client.post('/api/models/unload', json={'model': 'other'}).status_code == 200
            assert {m['id'] for m in client.get('/api/models/loaded').json()['data']} == {'chat'}
            cancelled = threading.Event()
            cancelled.set()
            try:
                engine.infer('voice', 'speech', {'input': 'hello'}, cancel=cancelled)
                raise AssertionError('Cancelled local request ran')
            except Exception as exc:
                assert getattr(exc, 'code', None) == 'cancelled', exc
            assert engine.local.current is None
            assert client.post('/api/models/load', json={'model': 'voice'}).status_code == 200
        assert not engine.catalog(loaded_only=True)['data']
        assert 'speech' in FakeModel.closed and 'chat' in FakeModel.closed
        engine.close()


def registry_writer(root, index):
    root = Path(root)
    (root / ('ready-' + index)).touch()
    deadline = time.monotonic() + 15
    while not (root / 'start').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Writer start barrier timed out')
        time.sleep(0.01)
    def change(cfg):
        # Force overlap between independent writers; locking must preserve both.
        time.sleep(0.05)
        cfg.models['model-' + index] = ModelSpec(path=str(root), task='chat')
    update_config(root / 'models.json', change)


def registry_transactions(root):
    path = root / 'models.json'
    write_config(path, Config(models={'existing': ModelSpec(path=str(root), task='chat')}))
    processes = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--writer', str(root), str(i)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(4)]
    try:
        deadline = time.monotonic() + 15
        while len(list(root.glob('ready-*'))) != len(processes):
            assert time.monotonic() < deadline, 'Independent registry writers did not start'
            assert all(p.poll() is None for p in processes), 'A registry writer exited before the barrier'
            time.sleep(0.01)
        (root / 'start').touch()
        for process in processes:
            out, err = process.communicate(timeout=20)
            assert process.returncode == 0, (out, err)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    assert set(read_config(path).models) == {'existing', *(f'model-{i}' for i in range(4))}
    original = path.read_bytes()
    def rejected(cfg):
        cfg.models.clear()
        raise ValueError('Rejected change')
    try:
        update_config(path, rejected)
        raise AssertionError('Rejected transaction was accepted')
    except ValueError:
        pass
    assert path.read_bytes() == original
    def invalid(cfg):
        cfg.models['?'] = ModelSpec(path=str(root), task='chat')
    try:
        update_config(path, invalid)
        raise AssertionError('Invalid registry was accepted')
    except ValueError:
        pass
    assert path.read_bytes() == original
    update_config(path, lambda cfg: cfg.models.pop('model-0'))
    assert 'model-0' not in read_config(path).models


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--writer':
        registry_writer(sys.argv[2], sys.argv[3])
        return
    with tempfile.TemporaryDirectory(prefix='titanos-lifecycle-') as temporary:
        root = Path(temporary)
        mixed_lifecycle(root)
        registry_transactions(root)
    print(json.dumps({'passed': True, 'checks': ['mixed worker/local routing and lifecycle',
          'explicit worker rejection and local cancellation', 'four independent registry writers',
          'rejected and invalid transaction preservation'], 'scope': 'Fake inference control paths; no hardware or model execution qualification'}))


if __name__ == '__main__':
    main()
