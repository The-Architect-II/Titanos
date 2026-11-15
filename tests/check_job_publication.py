"""Model-free cancellation/publication regressions; HTTP uses in-process ASGI."""
import argparse
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from titanos.config import Config, ModelSpec
from titanos.jobs import VideoJobs
from titanos.server import create_app
from titanos.settings import Settings


TIMEOUT = 5
TERMINAL = {'cancelled', 'failed', 'completed'}


def finished(read):
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        record = read()
        if record['status'] in TERMINAL:
            return record
        time.sleep(.01)
    raise AssertionError('Job did not publish a terminal record')


def check_http_image(root, cancelled):
    closing, release = threading.Event(), threading.Event()
    if not cancelled:
        release.set()

    class Frame:
        closed = False

        def save(self, path):
            Path(path).write_bytes(b'controlled image output')

        def close(self):
            closing.set()
            assert release.wait(TIMEOUT), 'Frame.close was not released'
            self.closed = True

    frame = Frame()
    app = create_app(Config(models={'asset': ModelSpec(
        path=str(root), backend='openvino-genai', task='image', device='CPU')}),
        output_dir=root/'out', settings=Settings(client_timeout=TIMEOUT * 2))
    app.state.engine.target.infer = lambda *args, **kwargs: [frame]
    foreign = root/'out'/'foreign.png'
    foreign.write_bytes(b'unrelated output')
    with TestClient(app, base_url='http://127.0.0.1') as client:
        try:
            response = client.post('/api/jobs', json={
                'task': 'image', 'model': 'asset', 'prompt': 'x', 'size': '64x64', 'steps': 1})
            assert response.status_code == 202, response.text
            identifier = response.json()['id']
            assert closing.wait(TIMEOUT), 'Image did not reach final close'
            owned = set((root/'out').glob('*.png')) - {foreign}
            assert len(owned) == 1, owned
            if cancelled:
                response = client.delete('/api/jobs/' + identifier)
                assert response.status_code == 200, response.text
                assert response.json()['status'] == 'cancelling', response.json()
        finally:
            release.set()
        done = finished(lambda: client.get('/api/jobs/' + identifier).json())
        assert frame.closed
        assert foreign.read_bytes() == b'unrelated output'
        if cancelled:
            assert done['status'] == 'cancelled' and 'result' not in done, done
            assert not any(path.exists() for path in owned), 'Cancelled image artifact remains'
        else:
            assert done['status'] == 'completed', done
            assert all(path.exists() for path in owned)
            artifact = client.get(done['result']['data'][0]['url'])
            assert artifact.status_code == 200 and artifact.content == b'controlled image output'


def check_cleanup(root, cancelled, cleanup_fails=False):
    store = VideoJobs(root/'jobs', settings=Settings())
    ready, release = threading.Event(), threading.Event()
    owned = root/'owned.png'
    cleanups = []

    def generate(cancel):
        owned.write_bytes(b'owned output')
        ready.set()
        assert release.wait(TIMEOUT), 'Generator was not released'
        if not cancelled:
            raise ValueError('generation failed')
        return {'url': 'http://127.0.0.1/v1/files/owned.png'}

    def cleanup():
        cleanups.append(True)
        if cleanup_fails:
            raise OSError('cleanup failed')
        owned.unlink(missing_ok=True)

    try:
        job = store.submit('asset', generate, task='image', cleanup=cleanup)
        assert ready.wait(TIMEOUT)
        if cancelled:
            assert store.cancel(job['id'])['status'] == 'cancelling'
        release.set()
        done = finished(lambda: store.get(job['id']))
        assert done['status'] == ('cancelled' if cancelled else 'failed'), done
        assert 'result' not in done and job['id'] not in store.events, done
        assert len(cleanups) == 1, cleanups
        assert owned.exists() == cleanup_fails
        if cleanup_fails:
            assert 'cleanup failed' in done['cleanup_error'], done
            if cancelled:
                assert 'cleanup failed' in done['error'], done
            else:
                assert done['error'] == 'generation failed', done
        else:
            assert 'cleanup_error' not in done, done
        assert store.cancel(job['id'])['status'] == done['status']
        assert len(cleanups) == 1
    finally:
        release.set()
        store.close()


class PublicationBoundary:
    """Expose a split decision/write gap, or hold the atomic terminal write."""
    def __init__(self, lock, active):
        self.lock = lock
        self.active = active
        self.depth = threading.local()
        self.worker = None
        self.contender = None
        self.written = False
        self.split_gap = False
        self.blocked = None
        self.ready = threading.Event()
        self.release = threading.Event()
        self.attempted = threading.Event()

    def __enter__(self):
        depth = getattr(self.depth, 'value', 0)
        if threading.current_thread() is self.contender and depth == 0:
            acquired = self.lock.acquire(blocking=False)
            self.blocked = not acquired
            self.attempted.set()
            if not acquired:
                self.lock.acquire()
        else:
            self.lock.acquire()
        self.depth.value = depth + 1
        return self

    def __exit__(self, *args):
        self.depth.value -= 1
        worker_exit = threading.get_ident() == self.worker and self.depth.value == 0
        if worker_exit and (not self.written or self.active()):
            self.split_gap = True
        self.lock.release()
        if worker_exit and not self.written and not self.ready.is_set():
            # Old code releases the decision lock before its finally write.
            self.ready.set()
            assert self.release.wait(TIMEOUT), 'Publication gap was not released'


def check_atomic_publication(root):
    store = VideoJobs(root/'jobs', settings=Settings())
    boundary = PublicationBoundary(store.lock, lambda: bool(store.events))
    store.lock = boundary
    write = store.write
    outcome = []
    artifact = root/'completed.png'

    def publication_write(record):
        if record['status'] == 'completed':
            boundary.ready.set()
            assert boundary.release.wait(TIMEOUT), 'Terminal write was not released'
        write(record)
        if record['status'] in TERMINAL:
            boundary.written = True

    def generate(cancel):
        artifact.write_bytes(b'completed output')
        boundary.worker = threading.get_ident()
        return {'artifact': artifact.name}

    store.write = publication_write
    try:
        job = store.submit('asset', generate)
        assert boundary.ready.wait(TIMEOUT), 'No publication boundary reached'
        boundary.contender = threading.Thread(
            target=lambda: outcome.append(store.cancel(job['id'])), name='publication-cancel')
        boundary.contender.start()
        assert boundary.attempted.wait(TIMEOUT), 'Cancel did not contend for the publication lock'
        boundary.release.set()
        boundary.contender.join(TIMEOUT)
        assert not boundary.contender.is_alive()
        done = finished(lambda: store.get(job['id']))
        assert not boundary.split_gap, 'Completion decision, write, and event removal released their lock'
        assert boundary.blocked, 'Cancellation entered while terminal publication was pending'
        assert outcome and outcome[0]['status'] == 'completed', outcome
        assert done['status'] == 'completed' and done['result'] == {'artifact': artifact.name}, done
        assert job['id'] not in store.events and artifact.read_bytes() == b'completed output'
    finally:
        boundary.release.set()
        if boundary.contender is not None:
            boundary.contender.join(TIMEOUT)
        store.close()


def check_success_cleanup(root):
    store = VideoJobs(root/'jobs', settings=Settings())
    artifact = root/'kept.png'
    artifact.write_bytes(b'completed output')
    cleanups = []
    try:
        job = store.submit('asset', lambda cancel: {'artifact': artifact.name},
                           cleanup=lambda: cleanups.append(True))
        done = finished(lambda: store.get(job['id']))
        assert done['status'] == 'completed' and done['result']['artifact'] == artifact.name, done
        assert not cleanups and artifact.exists() and job['id'] not in store.events
    finally:
        store.close()


def check_publication_failure(root):
    store = VideoJobs(root/'jobs', settings=Settings())
    artifact = root/'unpublished.png'; artifact.write_bytes(b'owned output')
    original = store.write
    failed = []
    cleanups = []
    def write(record):
        if record['status'] == 'completed' and not failed:
            failed.append(True)
            raise OSError('injected terminal write failure')
        original(record)
    def cleanup():
        cleanups.append(True); artifact.unlink(missing_ok=True)
    store.write = write
    try:
        job = store.submit('asset', lambda cancel: {'artifact': artifact.name}, cleanup=cleanup)
        done = finished(lambda: store.get(job['id']))
        assert done['status'] == 'failed' and 'result' not in done, done
        assert 'publication failed' in done['error'].lower()
        assert cleanups == [True] and not artifact.exists() and job['id'] not in store.events
    finally:
        store.close()


def check_dispatch_failure(root, retained=False, publication_failures=0, cleanup_fails=False):
    store = VideoJobs(root/'jobs', settings=Settings(queue_capacity=1, job_workers=1))
    store.start()
    dispatch_error = RuntimeError('injected executor submission failure')
    generated, cleanups, callbacks, failed_writes = [], [], [], []
    owned = root/'owned.png'
    owned.write_bytes(b'owned output')
    write = store.write

    def reject(callback, *args):
        if retained:
            callbacks.append((callback, args))
        raise dispatch_error

    def generate(cancel):
        generated.append(True)
        owned.write_bytes(b'late output')
        return {'artifact': owned.name}

    def cleanup():
        cleanups.append(True)
        if cleanup_fails:
            raise OSError('injected dispatch cleanup failure')
        owned.unlink(missing_ok=True)

    def publication(record):
        if record['status'] == 'failed' and len(failed_writes) < publication_failures:
            failed_writes.append(True)
            raise OSError('injected dispatch publication failure')
        write(record)

    try:
        with patch.object(store, 'write', publication), patch.object(store.executor, 'submit', reject):
            try:
                store.submit('asset', generate, task='image', request_key='first',
                             fingerprint='same', cleanup=cleanup)
            except RuntimeError as exc:
                assert exc is dispatch_error, 'Submission error was replaced'
            else:
                raise AssertionError('Rejected submission did not propagate its error')
            assert not generated and cleanups == [True] and not store.events
            assert owned.exists() == cleanup_fails
            records = store.list()['data']
            assert len(records) == 1, records
            done = records[0]
            assert 'result' not in done, done
            if publication_failures < 2:
                assert done['status'] == 'failed' and done['error'] == str(dispatch_error), done
                if publication_failures:
                    assert done['publication_error'] == 'OSError: injected dispatch publication failure', done
                if cleanup_fails:
                    assert done['cleanup_error'] == 'OSError: injected dispatch cleanup failure', done
            else:
                # Both terminal writes failed: only the initial queued record is
                # durable. The exception must still expose every secondary fault.
                assert done['status'] == 'queued', done
                assert any('Could not persist failed publication' in note for note in dispatch_error.__notes__)
            if publication_failures:
                assert any('Job publication failed' in note for note in dispatch_error.__notes__)
            if cleanup_fails:
                assert any('Artifact cleanup failed' in note for note in dispatch_error.__notes__)
            duplicate = store.submit('asset', generate, task='image', request_key='first',
                                     fingerprint='same', cleanup=cleanup)
            assert duplicate == done, (duplicate, done)
            assert len(callbacks) == int(retained) and cleanups == [True]
        before = (store.root/(done['id']+'.json')).read_bytes()
        for callback, args in callbacks:
            callback(*args)
        assert not generated and cleanups == [True] and not store.events
        assert (store.root/(done['id']+'.json')).read_bytes() == before
        assert owned.exists() == cleanup_fails
        good = store.submit('asset', lambda cancel: {'artifact': 'next'},
                            request_key='next', fingerprint='next', cleanup=cleanup)
        completed = finished(lambda: store.get(good['id']))
        assert completed['status'] == 'completed' and completed['result'] == {'artifact': 'next'}, completed
        assert not store.events and cleanups == [True]
        assert store.get(done['id']) == done
    finally:
        store.close()


def check_thread_start_failure(root):
    store = VideoJobs(root/'jobs', settings=Settings(queue_capacity=1, job_workers=1))
    dispatch_error = RuntimeError('injected worker thread startup failure')
    generated, cleanups = [], []
    try:
        # ThreadPoolExecutor queues its work item before starting the worker.
        # A later successful submission will run that retained callback first.
        with patch.object(threading.Thread, 'start', side_effect=dispatch_error):
            try:
                store.submit('asset', lambda cancel: generated.append(True),
                             request_key='first', fingerprint='same', cleanup=lambda: cleanups.append(True))
            except RuntimeError as exc:
                assert exc is dispatch_error
            else:
                raise AssertionError('Thread startup failure was not propagated')
        failed = store.list()['data'][0]
        assert failed['status'] == 'failed' and failed['error'] == str(dispatch_error), failed
        assert not store.events and not generated and cleanups == [True]
        good = store.submit('asset', lambda cancel: {'artifact': 'next'})
        completed = finished(lambda: store.get(good['id']))
        assert completed['status'] == 'completed' and completed['result'] == {'artifact': 'next'}, completed
        assert not generated and cleanups == [True] and not store.events
        assert store.get(failed['id']) == failed
        assert store.submit('asset', lambda cancel: generated.append(True),
                            request_key='first', fingerprint='same') == failed
        assert not generated
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    cases = [
        ('http_image_cancel_during_close', lambda root: check_http_image(root, True)),
        ('http_image_success_keeps_artifact', lambda root: check_http_image(root, False)),
        ('cancel_before_publication_cleans_once', lambda root: check_cleanup(root, True)),
        ('failed_generation_cleans_once', lambda root: check_cleanup(root, False)),
        ('cancelled_cleanup_error_is_visible', lambda root: check_cleanup(root, True, True)),
        ('failed_cleanup_error_preserves_original_error', lambda root: check_cleanup(root, False, True)),
        ('atomic_terminal_write_and_event_removal', check_atomic_publication),
        ('successful_publication_skips_cleanup', check_success_cleanup),
        ('failed_publication_discards_owned_artifact', check_publication_failure),
        ('dispatch_rejection_releases_capacity', check_dispatch_failure),
        ('retained_callback_cannot_revive_failed_dispatch', lambda root: check_dispatch_failure(root, retained=True)),
        ('dispatch_error_survives_publication_and_cleanup_errors',
         lambda root: check_dispatch_failure(root, retained=True, publication_failures=1, cleanup_fails=True)),
        ('dispatch_error_survives_persistent_publication_failure',
         lambda root: check_dispatch_failure(root, retained=True, publication_failures=2, cleanup_fails=True)),
        ('thread_start_failure_retained_work_is_inert', check_thread_start_failure),
    ]
    with tempfile.TemporaryDirectory() as temp:
        for name, check in cases:
            root = Path(temp)/name
            root.mkdir()
            check(root)
    result = {'success': True, 'checks': [name for name, _ in cases],
              'scope': 'In-process HTTP and deterministic thread barriers; no models or listeners'}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
