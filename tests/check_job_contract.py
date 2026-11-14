"""Job lifecycle contract for asynchronous clients (Cauldron and others): idempotent replay, lookup by the client's
Idempotency-Key, status filters, an explicit cancel acknowledgement, and restart interruption marked retryable."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from titanos.config import Config, ModelSpec, write_config
from titanos.engine import EngineError
from titanos.jobs import VideoJobs
from titanos.server import create_app


def until(call, terminal, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = call()
        if value['status'] in terminal:
            return value
        time.sleep(.02)
    raise AssertionError('Job did not reach a terminal state')


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def store_contract(root, checks):
    store = VideoJobs(root / 'queue')
    # idempotent replay and conflict
    first = store.submit('m', lambda cancel: {'artifact': 'a'}, 'image', key('k1'), 'fp1')
    assert store.submit('m', lambda cancel: {'artifact': 'b'}, 'image', key('k1'), 'fp1')['id'] == first['id']
    try:
        store.submit('m', lambda cancel: {}, 'image', key('k1'), 'fp2')
        raise AssertionError('a different body under the same key must conflict')
    except EngineError as exc:
        assert exc.status == 409 and exc.code == 'idempotency_conflict', exc
    until(lambda: store.get(first['id']), {'completed'})
    # a deleted record no longer claims its key: the same key and body start a new job
    store.delete(first['id'])
    again = store.submit('m', lambda cancel: {'artifact': 'c'}, 'image', key('k1'), 'fp1')
    assert again['id'] != first['id'], 'a deleted job must not be replayed'
    until(lambda: store.get(again['id']), {'completed'})
    checks.append('idempotent replay, 409 on a changed body, deleted records release their key')

    # cancel acknowledgement on a running job, then on the terminal job
    started = threading.Event()

    def blocking(cancel):
        started.set()
        cancel.wait(10)
        return {'ignored': True}

    running = store.submit('m', blocking, 'video')
    assert started.wait(5)
    ack = store.cancel(running['id'])
    assert ack['cancel_accepted'] is True and ack['status'] == 'cancelling', ack
    assert isinstance(ack['cancel_requested_at'], float), ack
    done = until(lambda: store.get(running['id']), {'cancelled'})
    assert done['cancel_requested_at'] == ack['cancel_requested_at'], done
    late = store.cancel(running['id'])
    assert late['cancel_accepted'] is False and late['status'] == 'cancelled', late
    assert 'cancel_accepted' not in store.get(running['id']), 'the acknowledgement is not part of the stored record'
    finished = store.submit('m', lambda cancel: {'artifact': 'd'}, 'image')
    until(lambda: store.get(finished['id']), {'completed'})
    already = store.cancel(finished['id'])
    assert already['cancel_accepted'] is False and already['status'] == 'completed', already
    assert 'cancel_requested_at' not in already, already
    checks.append('cancel answers cancel_accepted (true while active, false once terminal) and keeps cancel_requested_at')

    # a queued job behind a running one is cancelled before it runs
    gate = threading.Event()
    blocker = store.submit('m', lambda cancel: gate.wait(10) and {'x': 1}, 'video')
    queued = store.submit('m', lambda cancel: {'never': True}, 'video')
    queued_ack = store.cancel(queued['id'])
    assert queued_ack['cancel_accepted'] is True, queued_ack
    gate.set()
    until(lambda: store.get(blocker['id']), {'completed'})
    assert until(lambda: store.get(queued['id']), {'cancelled'})['execution_started'] is False
    checks.append('a queued job cancelled before it runs ends cancelled with execution_started false')

    # filters and lookup by key
    keyed = store.submit('m', lambda cancel: {'artifact': 'e'}, 'music', key('cauldron:j9:s1:0'), 'fp9')
    until(lambda: store.get(keyed['id']), {'completed'})
    found = store.list(request_key=key('cauldron:j9:s1:0'))['data']
    assert [r['id'] for r in found] == [keyed['id']], found
    assert store.list(request_key=key('nope'))['data'] == []
    assert {r['status'] for r in store.list(status='cancelled')['data']} == {'cancelled'}
    assert all(r['task'] == 'music' for r in store.list(task='music')['data'])
    checks.append('list filters by status, task and Idempotency-Key hash')

    # restart: unfinished records become interrupted, time-stamped and retryable
    store.write({'id': 'a' * 32, 'status': 'running', 'task': 'video', 'model': 'm', 'created_at': time.time()})
    store.write({'id': 'b' * 32, 'status': 'cancelling', 'task': 'video', 'model': 'm', 'created_at': time.time()})
    store.close()
    reopened = VideoJobs(root / 'queue')
    for identifier in ('a' * 32, 'b' * 32):
        record = reopened.get(identifier)
        assert record['status'] == 'interrupted' and record['retryable'] is True, record
        assert isinstance(record['interrupted_at'], float), record
    assert reopened.get(again['id'])['status'] == 'completed'
    assert 'retryable' not in reopened.get(again['id'])
    reopened.close()
    checks.append('restart marks unfinished jobs interrupted with interrupted_at and retryable')


def server_contract(root, checks):
    class Model:
        def __init__(self, spec):
            self.spec = spec
            self.device = 'cpu'

        def infer(self, task, data, emit=None, cancel=None):
            return [Image.new('RGB', (data['width'], data['height']), 'green')]

    with patch('titanos.backends.LoadedModel', Model):
        weights = root / 'w'
        weights.mkdir()
        registry = root / 'models.json'
        cfg = Config(models={'image': ModelSpec(path=str(weights), task='image', backend='diffusers')})
        write_config(registry, cfg)
        app = create_app(cfg, output_dir=root / 'output', registry_path=registry)
        with TestClient(app, base_url='http://localhost') as c:
            body = {'task': 'image', 'model': 'image', 'prompt': 'green', 'size': '32x32', 'steps': 1}
            headers = {'Idempotency-Key': 'cauldron:j1:s1:0'}
            job = c.post('/api/jobs', json=body, headers=headers)
            assert job.status_code == 202, job.text
            identifier = job.json()['id']
            until(lambda: c.get('/api/jobs/' + identifier).json(), {'completed'})
            assert c.post('/api/jobs', json=body, headers=headers).json()['id'] == identifier
            found = c.get('/api/jobs', params={'idempotency_key': 'cauldron:j1:s1:0'})
            assert found.status_code == 200 and [r['id'] for r in found.json()['data']] == [identifier], found.text
            assert 'request_key' not in found.json()['data'][0]
            assert c.get('/api/jobs', params={'idempotency_key': 'cauldron:other'}).json()['data'] == []
            assert c.get('/api/jobs', params={'status': 'completed'}).json()['data'][0]['id'] == identifier
            assert c.get('/api/jobs', params={'status': 'bogus'}).status_code == 422
            assert c.get('/api/jobs', params={'idempotency_key': 'x' * 129}).status_code == 422
            ack = c.delete('/api/jobs/' + identifier)
            assert ack.status_code == 200 and ack.json()['cancel_accepted'] is False, ack.text
            info = c.get('/api/info').json()['jobs']
            assert info['lookup'] == {'idempotency_key': 'GET /api/jobs?idempotency_key=', 'status': 'GET /api/jobs?status='}, info
            assert info['cancel'] == {'route': 'DELETE /api/jobs/{id}', 'acknowledgement': 'cancel_accepted',
                                      'terminal': ['cancelled', 'completed', 'failed', 'interrupted']}, info
    checks.append('HTTP: GET /api/jobs?idempotency_key=&status= (validated), DELETE answers cancel_accepted, /api/info documents both')


def main():
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store_contract(root / 'store', checks)
        (root / 'server').mkdir()
        server_contract(root / 'server', checks)
    print(json.dumps({'success': True, 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
