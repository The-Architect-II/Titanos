"""Actual spawn-argument serialization and child settings with inert runtimes.

No child process, listener, model, Torch, ONNX or OpenVINO runtime is launched.
The actual Worker.start boundary is observed with a process double; its payload
uses multiprocessing's real ForkingPickler before the actual child entry runs
in a fresh Context. Compiled loader calls use provider doubles.
"""
from contextvars import Context
from dataclasses import FrozenInstanceError, asdict, fields
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import copy
import os
import pickle
import sys
import tempfile
import threading
import unittest

from titanos import compiled, replicas
from titanos.config import Config, ModelSpec, WorkerSpec
from titanos.server import create_app
from titanos.settings import Settings, _active_settings, current_settings


def configuration(threads=6):
    return Config(models={'m': ModelSpec(path='.', task='tensor', backend='onnx', device='cpu')},
                  workers={'cpu': WorkerSpec(models=['m'], threads=threads, device='cpu')})


class SpawnCapture:
    def __init__(self):
        self.payloads = []
        self.child_closed = False

    def Pipe(self):
        return NS(close=lambda: None), NS(close=lambda: setattr(self, 'child_closed', True))

    def Event(self):
        return threading.Event()

    def Process(self, *, target, args, **kwargs):
        assert target is replicas.worker_main and len(args) == 5
        assert kwargs['daemon'] is False
        # Real spawn serialization of precisely the configuration/assignment/
        # settings passed by Worker.start, without starting a native process.
        encoded = bytes(ForkingPickler.dumps(args[2:]))
        decoded = pickle.loads(encoded)
        self.payloads.append(decoded)
        return NS(start=lambda: None, is_alive=lambda: False, pid=None)


def capture(worker):
    context = SpawnCapture()
    before = dict(os.environ)
    with patch.object(replicas.mp, 'get_context', return_value=context) as get_context:
        worker.start()
    get_context.assert_called_once_with('spawn')
    assert context.child_closed and os.environ == before
    assert len(context.payloads) == 1
    return context.payloads[0]


def consume(payload, *, failure=None, existing=None):
    """Real worker entry/config/tensor parser; fake engine and pipe only."""
    cfg, assignment, settings = copy.deepcopy(payload)
    observations, sent = [], []
    child = Context()
    if existing is not None:
        child.run(_active_settings.set, existing)
    parse = Config.model_validate

    def validate(value):
        observations.append(('config', asdict(current_settings())))
        if failure == 'config':
            raise ValueError('inert config failure')
        return parse(value)

    class Engine:
        def __init__(self, cfg):
            observations.append(('engine', asdict(current_settings())))
            if failure == 'engine':
                raise RuntimeError('inert engine failure')
            self.current, self.runtime = 'm', None

        def infer(self, name, task, data, emit, cancel):
            observations.append(('infer', asdict(current_settings())))
            value = compiled.tensor_payload(data['inputs']['x'])
            return {'elements': value.size, 'limit': current_settings().max_tensor_elements}

        def details(self, name):
            return {}

        def _runtime_fault(self):
            return None

        def unload(self):
            observations.append(('unload', asdict(current_settings())))
            if failure == 'unload':
                raise RuntimeError('inert unload failure')

    requests = iter([('infer', 'm', 'tensor', {'inputs': {'x': {
        'dtype': 'int32', 'shape': [4], 'data': [1, 2, 3, 4]}}}), ('close', None, None, None)])
    connection = NS(send=sent.append, recv=lambda: next(requests), close=lambda: None)
    child_os = NS(name='nt', environ={'TITANOS_MAX_TENSOR_ELEMENTS': '999'}, getpid=lambda: 123)
    error = None
    before = dict(os.environ)
    with patch.object(replicas, 'Engine', Engine), patch.object(replicas, 'os', child_os), \
            patch.object(Config, 'model_validate', side_effect=validate):
        try:
            child.run(replicas.worker_main, connection, threading.Event(), cfg, assignment, settings)
        except (ValueError, TypeError, RuntimeError) as exc:
            error = exc
    assert os.environ == before
    assert child.run(_active_settings.get) is existing
    return observations, sent, error, child_os.environ


class SettingsPropagation(unittest.TestCase):
    def test_two_apps_keep_their_settings_across_spawn(self):
        narrow = Settings(threads=12, max_tensor_elements=3, max_text_chars=17,
                          detection_threshold=.2, replica_queue_capacity=2)
        broad = Settings(threads=18, max_tensor_elements=6, max_text_chars=29,
                         detection_threshold=.8, replica_queue_capacity=7)
        outer = Settings(threads=23, max_tensor_elements=77)
        token = _active_settings.set(outer)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
                'TITANOS_THREADS': '31', 'TITANOS_MAX_TENSOR_ELEMENTS': '999',
                'TITANOS_API_KEY': 'inert-parent-key-not-a-setting'}):
            apps = [create_app(configuration(), settings=value, api_key='inert-app-key',
                               output_dir=Path(temporary)/str(i)) for i, value in enumerate((narrow, broad))]
            try:
                for app, selected in zip(apps, (narrow, broad)):
                    engine = app.state.engine.target
                    payload = capture(engine.workers[0])
                    self.assertEqual(payload[2], asdict(selected))
                    self.assertEqual(set(payload[2]), {f.name for f in fields(Settings)})
                    self.assertNotIn('inert-parent-key', repr(payload))
                    self.assertNotIn('inert-app-key', repr(payload))
                    observed, sent, error, environment = consume(payload)
                    self.assertIsNone(error)
                    expected = {**asdict(selected), 'threads': 6}
                    self.assertTrue(observed)
                    self.assertTrue(all(value == expected for _, value in observed))
                    self.assertEqual(environment['TITANOS_THREADS'], '6')
                    if selected is narrow:
                        self.assertTrue(any(kind == 'error' and 'at most 3 elements' in value[0]
                                            for kind, value in sent))
                        self.assertFalse(any(kind == 'done' for kind, _ in sent))
                    else:
                        self.assertTrue(any(kind == 'done' and value[0] == {'elements': 4, 'limit': 6}
                                            for kind, value in sent))
                    self.assertIs(current_settings(), outer)
            finally:
                for app in apps:
                    app.state.engine.target.close()
                _active_settings.reset(token)

    def test_snapshot_default_environment_and_assignment_precedence(self):
        with patch.dict(os.environ, {'TITANOS_WORKER_THREADS': '5', 'TITANOS_THREADS': '11',
                                     'TITANOS_MAX_TENSOR_ELEMENTS': '9'}):
            cfg = Config(models=configuration().models, workers={'cpu': WorkerSpec(models=['m'])})
            engine = replicas.ReplicaEngine(cfg)
            os.environ['TITANOS_MAX_TENSOR_ELEMENTS'] = '99'
            try:
                payload = capture(engine.workers[0])
                self.assertEqual(payload[2]['max_tensor_elements'], 9)
                self.assertEqual(payload[1]['threads'], 5)
                observed, _, error, _ = consume(payload)
                self.assertIsNone(error)
                self.assertTrue(all(value['threads'] == 5 and value['max_tensor_elements'] == 9
                                    for _, value in observed))
                with self.assertRaises(FrozenInstanceError):
                    engine.workers[0].settings.max_tensor_elements = 1
            finally:
                engine.close()

    def test_invalid_child_settings_stop_before_config_or_engine(self):
        cfg = configuration()
        for change in ({'max_tensor_elements': 0}, {'max_tensor_elements': True},
                       {'threads': float('nan')}, {'api_key': 'inert-unsupported-field'}):
            payload = (cfg.model_dump(), {**cfg.workers['cpu'].model_dump(), 'resolved_gpus': []},
                       {**asdict(Settings()), **change})
            observed, sent, error, environment = consume(payload, existing=Settings(threads=22))
            self.assertIsNotNone(error)
            self.assertEqual(observed, [])
            self.assertEqual(sent, [])
            self.assertNotIn('TITANOS_THREADS', environment)

    def test_invalid_assignment_thread_does_not_override_validation(self):
        cfg = configuration()
        for threads in (0, True, 1.5):
            payload = (cfg.model_dump(), {**cfg.workers['cpu'].model_dump(), 'resolved_gpus': [], 'threads': threads},
                       asdict(Settings()))
            observed, _, error, _ = consume(payload)
            self.assertIsInstance(error, ValueError)
            self.assertEqual(observed, [])

    def test_child_context_restores_on_success_and_failures(self):
        cfg = configuration()
        payload = (cfg.model_dump(), {**cfg.workers['cpu'].model_dump(), 'resolved_gpus': []},
                   asdict(Settings(max_tensor_elements=6)))
        for failure in (None, 'config', 'engine', 'unload'):
            with self.subTest(failure=failure):
                observed, _, error, _ = consume(payload, failure=failure, existing=Settings(threads=21))
                self.assertTrue(observed)
                self.assertEqual(error is None, failure is None)

    def test_compiled_loaders_use_active_settings_and_explicit_model_override(self):
        onnx_threads, ov_threads = [], []

        def session(path, sess_options, **kwargs):
            onnx_threads.append(sess_options.intra_op_num_threads)
            return NS(disable_fallback=lambda: None, get_providers=lambda: ['CPUExecutionProvider'])

        def compile_model(model, device, options):
            ov_threads.append(options['INFERENCE_NUM_THREADS'])
            return NS(inputs=[], outputs=[], get_property=lambda key: ['CPU'])

        ort = NS(SessionOptions=lambda: NS(), get_available_providers=lambda: ['CPUExecutionProvider'],
                 InferenceSession=session)
        ov = NS(Core=lambda: NS(available_devices=['CPU'], read_model=lambda path: object(), compile_model=compile_model))
        with patch.dict(sys.modules, {'onnxruntime': ort, 'openvino': ov}), \
                patch.object(compiled, 'graph_path', return_value=Path('inert-no-model-file')), \
                patch.dict(os.environ, {'TITANOS_THREADS': 'invalid-ambient-value'}):
            for threads in (12, 19):
                token = _active_settings.set(Settings(threads=threads))
                try:
                    compiled.onnx_session('inert', 'cpu')
                    compiled.CompiledModel(ModelSpec(path='.', task='tensor', backend='openvino', device='CPU'))
                    compiled.CompiledModel(ModelSpec(path='.', task='tensor', backend='openvino', device='CPU',
                                                      load_options={'INFERENCE_NUM_THREADS': 7}))
                finally:
                    _active_settings.reset(token)
            self.assertEqual(onnx_threads, [12, 19])
            self.assertEqual(ov_threads, [12, 7, 19, 7])
            with self.assertRaises(ValueError):
                compiled.onnx_session('inert', 'cpu')
            self.assertEqual(onnx_threads, [12, 19])
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(sys.modules, {'onnxruntime': ort}):
                compiled.onnx_session('inert', 'cpu')
            self.assertEqual(onnx_threads[-1], Settings().threads)


if __name__ == '__main__':
    unittest.main()
