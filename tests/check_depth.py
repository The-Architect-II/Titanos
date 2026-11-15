"""Depth semantics, exact numeric encoding, limits and public interfaces; no models."""
import asyncio
import base64
from contextlib import nullcontext, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from titanos.api_plus import DepthRequest
from titanos.client import Client
from titanos.cli import main as cli
from titanos.config import Config, ModelSpec
from titanos.depth import DepthModel, DepthResult, check_dimensions
from titanos.engine import EngineError
from titanos.server import create_app
from titanos.settings import Settings


class Tensor:
    def __init__(self, value): self.value = np.asarray(value); self.shape = self.value.shape; self.ndim = self.value.ndim
    def is_floating_point(self): return self.value.dtype.kind == 'f'
    def to(self, **kwargs): return self
    def detach(self): return self
    def float(self): return Tensor(self.value.astype('float32'))
    def cpu(self): return self
    def numpy(self): return self.value


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = Image.new('RGB', (4, 2), (19, 71, 143)); self.addCleanup(self.image.close)
        buffer = io.BytesIO(); self.image.save(buffer, format='PNG'); self.png = buffer.getvalue()
        self.values = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype='<f4')
        self.spec = ModelSpec(path=str(self.root), task='depth-estimation', device='cpu')
        self.backend = DepthModel.__new__(DepthModel); self.backend.spec = self.spec; self.backend.device = 'cpu'
        self.calls = []
        outer = self
        class Processor:
            def __call__(self, **kwargs): return {'pixel_values': Tensor(np.zeros((1, 3, 2, 4), dtype='float32'))}
            def post_process_depth_estimation(self, output, target_sizes):
                outer.assertEqual(target_sizes, [(2, 4)])
                return [{'predicted_depth': Tensor(outer.values)}]
        class Model:
            dtype = 'float32'
            def __call__(self, **kwargs):
                outer.calls.append(kwargs); return NS(predicted_depth=Tensor(np.ones((1, 2, 2), dtype='float32')))
        self.backend.processor = Processor(); self.backend.model = Model()
        # Restore only this double, not unrelated lazy imports such as crypto.
        previous_torch = sys.modules.get('torch')
        sys.modules['torch'] = NS(inference_mode=nullcontext,
            cuda=NS(is_available=lambda: False), backends=NS(mps=NS(is_available=lambda: False)))
        def restore_torch():
            if previous_torch is None: sys.modules.pop('torch', None)
            else: sys.modules['torch'] = previous_torch
        self.addCleanup(restore_torch)

    def result(self, **kwargs):
        return self.backend.infer('depth-estimation', {'model': 'depth', 'image': self.image}, **kwargs)

    def test_numeric_geometry_semantics_and_limits(self):
        result = DepthResult.model_validate(self.result())
        self.assertEqual(base64.b64decode(result.data), self.values.tobytes())
        self.assertEqual(result.model_output_shape, [2, 2])
        self.assertEqual(result.input_rgb_sha256, hashlib.sha256(self.image.tobytes()).hexdigest())
        self.assertEqual(result.semantics, 'relative_inverse_depth'); self.assertIsNone(result.units)
        self.assertEqual(result.normalization, 'none'); self.assertFalse(result.calibrated)
        settings = Settings(max_tensor_elements=7)
        with patch('titanos.depth.current_settings', return_value=settings):
            before = len(self.calls)
            with self.assertRaises(ValueError): self.result()
            self.assertEqual(len(self.calls), before)
        for alteration in [{'units': 'metres'}, {'width': 3}, {'data': result.data[:-1]},
                           {'data': '!' * len(result.data)}, {'calibrated': True}, {'model_output_shape': [True, 2]}]:
            with self.assertRaises(ValueError): DepthResult.model_validate({**result.model_dump(), **alteration})

    def test_invalid_output_and_preprocessing_cancellation(self):
        for values in [np.zeros((2, 3)), np.full((2, 4), np.nan), np.full((2, 4), np.inf)]:
            self.values = values
            with self.assertRaises(EngineError) as error: self.result()
            self.assertEqual(error.exception.code, 'invalid_backend_response')
        cancelled = threading.Event()
        original = self.backend.processor
        self.backend.processor = lambda **kwargs: cancelled.set() or original(**kwargs)
        before = len(self.calls)
        with self.assertRaises(EngineError) as error: self.result(cancel=cancelled)
        self.assertEqual(error.exception.code, 'cancelled'); self.assertEqual(len(self.calls), before)

    def test_registration_is_safe_and_single_device(self):
        for changes in [{'allow_pickle': True}, {'trust_remote_code': True}, {'defaults': {'max_tokens': 5}},
                        {'load_options': {'config': {}}}, {'load_options': {'ignore_mismatched_sizes': True}},
                        {'parallelism': 'model_sharding'}, {'backend': 'onnx'}]:
            with self.assertRaises(ValueError): ModelSpec.model_validate({**self.spec.model_dump(), **changes})

    def test_cancellation_during_transfer_prevents_forward(self):
        cancelled = threading.Event()
        class TransferTensor(Tensor):
            def to(self, **kwargs): cancelled.set(); return self
        self.backend.processor = lambda **kwargs: {'pixel_values': TransferTensor(np.zeros((1, 3, 2, 4), dtype='float32'))}
        with self.assertRaises(EngineError) as error: self.result(cancel=cancelled)
        self.assertEqual(error.exception.code, 'cancelled'); self.assertFalse(self.calls)

    def test_loader_rejects_metric_missing_weights_and_changed_config(self):
        config_file = self.root / 'config.json'; config_file.write_text('{}')
        config = NS(model_type='depth_anything', depth_estimation_type='relative', backbone=None,
                    use_pretrained_backbone=False, use_timm_backbone=False, backbone_config=NS(model_type='dinov2'))
        info = {}; loaded = []
        model = NS(config=config, parameters=lambda: [NS(device='cpu')], buffers=lambda: [])
        model.to = lambda device: model; model.eval = lambda: model
        def load(*args, **kwargs): loaded.append(kwargs); return model, info
        fake = NS(AutoConfig=NS(from_pretrained=lambda *a, **k: config),
                  AutoModelForDepthEstimation=NS(from_pretrained=load),
                  AutoImageProcessor=NS(from_pretrained=lambda *a, **k: self.backend.processor))
        with patch.dict(sys.modules, {'transformers': fake}):
            backend = DepthModel(self.spec, 'cpu', {'local_files_only': True, 'use_safetensors': True})
            self.assertEqual(backend.device, 'cpu'); self.assertFalse(loaded[-1]['trust_remote_code'])
            config.depth_estimation_type = 'metric'; before = len(loaded)
            with self.assertRaises(ValueError): DepthModel(self.spec, 'cpu', {})
            self.assertEqual(len(loaded), before)
            config.depth_estimation_type = 'relative'
            for flag, value, original in [('backbone', 'outside/checkpoint', None), ('use_pretrained_backbone', True, False),
                    ('use_timm_backbone', True, False), ('backbone_kwargs', {'trust_remote_code': True}, None)]:
                setattr(config, flag, value); before = len(loaded)
                with self.assertRaises(ValueError): DepthModel(self.spec, 'cpu', {})
                self.assertEqual(len(loaded), before); setattr(config, flag, original)
            info['missing_keys'] = ['depth_head.weight']
            with self.assertRaises(ValueError): DepthModel(self.spec, 'cpu', {})
            info.clear()
            def changed(*args, **kwargs): config_file.write_text('{"changed":true}'); return model, info
            fake.AutoModelForDepthEstimation.from_pretrained = changed
            with self.assertRaises(ValueError): DepthModel(self.spec, 'cpu', {})

    def app(self):
        app = create_app(Config(models={'depth': self.spec}), output_dir=self.root / 'assets', settings=Settings())
        app.state.engine.target.infer = lambda model, task, data, emit=None, cancel=None: self.backend.infer(task, {k:v for k,v in data.items() if k != 'worker'}, cancel)
        return app

    def test_http_extension_mcp_and_deadline(self):
        app = self.app(); body = {'model': 'depth', 'image_base64': base64.b64encode(self.png).decode()}
        with TestClient(app, base_url='http://127.0.0.1') as client:
            for route in ['/v1/images/depth', '/v1/extensions/infer']:
                request = {**body, **({'operation': 'depth'} if route.endswith('infer') else {})}
                response = client.post(route, json=request)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(base64.b64decode(response.json()['data']), self.values.tobytes())
                self.assertEqual(client.post(route, json={**request, 'max_tokens': 7}).status_code, 422)
            for tool, request in [('estimate_depth', body), ('extension_infer', {**body, 'operation': 'depth'})]:
                reply = asyncio.run(app.state.mcp.call_tool(tool, {'payload': request}))
                self.assertFalse(reply.is_error, reply)
                data = reply.structured_content; data = data.get('result', data)
                self.assertEqual(data['semantics'], 'relative_inverse_depth')
            before = len(self.calls)
            cancel = threading.Event(); cancel.set()
            for event, deadline in [(cancel, None), (None, time.monotonic()-1)]:
                with self.assertRaises(EngineError): app.state.extension_handlers['depth'](DepthRequest(**body), event, deadline)
            self.assertEqual(len(self.calls), before)
            with patch('titanos.api_plus._image', side_effect=AssertionError('Must not decode')):
                with self.assertRaises(EngineError): app.state.extension_handlers['depth'](DepthRequest(**body), cancel, None)

    def test_python_and_cli(self):
        result = self.result(); requests = []
        filename = self.root / 'image.png'; filename.write_bytes(self.png)
        def handler(request): requests.append(json.loads(request.content)); return httpx.Response(200, json=result)
        with Client(transport=httpx.MockTransport(handler)) as client:
            output = client.depth('depth', filename, timeout=500)
            self.assertIsInstance(output, DepthResult); self.assertEqual(requests[-1]['timeout'], 500)
        with patch.object(Client, 'depth', return_value=DepthResult.model_validate(result)) as call, redirect_stdout(io.StringIO()):
            cli(['depth', 'depth', str(filename), '--deadline-ms', '500'])
            self.assertEqual(call.call_args.kwargs, {'timeout': 500})

    def test_output_budgets_reject_before_rgb_and_model(self):
        for dimensions, limits in [((4, 2), {'max_tensor_elements': 7}), ((20, 20), {'max_body_bytes': 800})]:
            with Image.new('RGB', dimensions) as image:
                stream = io.BytesIO(); image.save(stream, format='PNG'); raw = stream.getvalue()
            app = create_app(Config(models={'depth': self.spec}), output_dir=self.root/'limits', settings=Settings(**limits))
            app.state.engine.target.infer = lambda *a, **k: self.fail('Budget rejection must precede inference')
            with TestClient(app, base_url='http://127.0.0.1') as client, patch.object(Image.Image, 'convert', side_effect=AssertionError('Must reject before pixel conversion')):
                for route in ['/v1/images/depth', '/v1/extensions/infer']:
                    body = {'model': 'depth', 'image_base64': base64.b64encode(raw).decode()}
                    if route.endswith('infer'): body['operation'] = 'depth'
                    response = client.post(route, json=body)
                    self.assertEqual(response.status_code, 413, response.text)
                    self.assertEqual(response.json()['error']['code'], 'request_too_large')


if __name__ == '__main__': unittest.main(verbosity=2)
