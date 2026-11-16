"""/api/infer takes the request's budget (X-Request-Budget-Ms) as a deadline and refuses stale work before running it.

The cluster facade forwards a client's X-Request-Budget-Ms unchanged. TitanOS counts it from the request's arrival:
a request still waiting for the model when its budget passes is answered 504 budget_passed and never reaches the
runtime; one answered within it runs as before; without the header nothing changes. The value is checked as the
facade checks it (ASCII digits, 1 to 600000 ms, named once), anything else 400 budget_invalid before any work.
A stand-in runtime records each call; no model, GPU or network.
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
HOME = tempfile.TemporaryDirectory(prefix='titanos-infer-deadline-home-', ignore_cleanup_errors=True)
os.environ['TITANOS_HOME'] = HOME.name  # nothing from this check reaches ~/.titanos

from fastapi.testclient import TestClient  # noqa: E402
from titanos import server  # noqa: E402
from titanos.config import Config, ModelSpec  # noqa: E402
from titanos.settings import Settings  # noqa: E402

TENSOR = {'model': 'tens', 'inputs': {'x': {'dtype': 'float32', 'shape': [1], 'data': [1.0]}}}


class Runtime:
    device = 'cpu'

    def __init__(self):
        self.calls = []

    def infer(self, task, data, emit=None, cancel=None):
        self.calls.append(task)
        return {'outputs': {'y': {'dtype': 'float32', 'shape': [1], 'data': [2.0]}}}


def code(reply):
    return reply.json()['error']['code']


def main():
    with tempfile.TemporaryDirectory(prefix='titanos-infer-deadline-') as scratch:
        config = Config(models={'tens': ModelSpec(path=scratch, task='tensor', backend='onnx', device='cpu')})
        app = server.create_app(config, output_dir=Path(scratch) / 'out', settings=Settings(), runtime_catalog=False)
        engine = app.state.engine.target
        runtime = Runtime()
        engine.runtime, engine.current, engine.states['tens'] = runtime, 'tens', 'loaded'
        with TestClient(app, base_url='http://127.0.0.1:8000') as client:
            def infer(budget=None, headers=None):
                return client.post('/api/infer', json=TENSOR,
                                   headers=headers or ({} if budget is None else {'X-Request-Budget-Ms': budget}))

            reply = infer()
            assert reply.status_code == 200 and runtime.calls == ['tensor'], reply.text  # no header: as before
            reply = infer('5000')
            assert reply.status_code == 200 and runtime.calls == ['tensor'] * 2, reply.text

            for bad in ('0', '600001', '1000000', 'abc', '+5', '-5', '1.5', '1e3', '', '1 2'):
                reply = infer(bad)
                assert reply.status_code == 400 and code(reply) == 'budget_invalid', (bad, reply.text)
            reply = client.post('/api/infer', json=TENSOR, headers=[('X-Request-Budget-Ms', '100'),
                                                                    ('X-Request-Budget-Ms', '100')])
            assert reply.status_code == 400 and code(reply) == 'budget_invalid', reply.text  # named twice
            assert runtime.calls == ['tensor'] * 2, runtime.calls  # no refused request reached the runtime
            assert infer('600000').status_code == 200 and len(runtime.calls) == 3

            # another request holds the model past this one's budget: the waiting request is refused, unrun
            release = threading.Timer(0.6, engine.lock.release)
            engine.lock.acquire()
            release.start()
            started = time.monotonic()
            reply = infer('150')
            release.join()
            assert reply.status_code == 504 and code(reply) == 'budget_passed', reply.text
            assert time.monotonic() - started >= 0.5 and len(runtime.calls) == 3, runtime.calls

            # held for less than the budget: it waits and runs
            release = threading.Timer(0.1, engine.lock.release)
            engine.lock.acquire()
            release.start()
            reply = infer('5000')
            release.join()
            assert reply.status_code == 200 and len(runtime.calls) == 4, reply.text

            # an engine error inside the budget keeps its own answer
            assert client.post('/api/infer', json={**TENSOR, 'model': 'nope'},
                               headers={'X-Request-Budget-Ms': '5000'}).status_code == 404
    print('check_infer_deadline: ok')


if __name__ == '__main__':
    main()
