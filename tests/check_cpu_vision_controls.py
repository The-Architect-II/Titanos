"""CPU vision lifecycle and decoder budgets with doubles; no Torch/models loaded."""
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from titanos.api_plus import OCRRequest, DetectionRequest
from titanos.config import Config, ModelSpec
from titanos.engine import EngineError
from titanos.extra_backends import ExtraBackend
from titanos.server import create_app
from titanos.settings import Settings


class Image:
    width = 640
    height = 96
    def __init__(self): self.closed = False
    def close(self): self.closed = True


class Prefix:
    def __init__(self, values): self.values = values; self.shape = (1, len(values))
    def __getitem__(self, index): return NS(item=lambda: self.values[index[1]])


def adapter_budgets():
    backend = ExtraBackend.__new__(ExtraBackend)
    backend.spec = NS(defaults={})
    backend.device = 'cpu'
    positions = NS(weight=NS(shape=(514, 8)), offset=2)
    decoder = NS(config=NS(model_type='trocr', max_position_embeddings=512,
                          use_learned_position_embeddings=True),
                 model=NS(decoder=NS(embed_positions=positions)))
    generated = []
    backend.model = NS(config=NS(model_type='vision-encoder-decoder'),
                       generation_config=NS(decoder_start_token_id=2, bos_token_id=0),
                       get_decoder=lambda: decoder,
                       generate=lambda **kw: generated.append(kw) or [[2, 7]])
    backend.processor = lambda **kw: {'pixel_values': object()}
    backend.decoder = NS(batch_decode=lambda *a, **k: ['controlled output'])
    backend._inputs = lambda inputs: inputs
    fake_transformers = NS(StoppingCriteria=object, StoppingCriteriaList=list)
    with patch.dict(sys.modules, {'transformers': fake_transformers}), \
            patch('titanos.extra_backends.load_settings', return_value=Settings(default_max_tokens=16)):
        def run(limit=None):
            data = {'image': Image()}
            if limit is not None: data['max_tokens'] = limit
            return backend.image_to_text(data, None)
        # Omitted request uses model aliases and then process default, without clamping.
        for defaults, explicit, expected in [({}, None, 16), ({'max_new_tokens': 24}, None, 24),
                ({'max_tokens': 32, 'max_new_tokens': 24}, None, 32), ({'max_tokens': 32}, 7, 7)]:
            backend.spec.defaults = defaults
            run(explicit)
            assert generated[-1]['max_new_tokens'] == expected
        backend.spec.defaults = {}
        run(511)  # One decoder start token plus 511 new tokens fits 512 positions.
        before = len(generated)
        for limit in [512, 8192, 8193, 10**1000, True]:
            try: run(limit)
            except ValueError: pass
            else: raise AssertionError('Invalid decoder budget reached generation')
        assert len(generated) == before
        # This TrOCR-specific bound must not impose its capacity on other families.
        backend.model.config.model_type = 'other-image-to-text'
        run(8192)
        backend.model.config.model_type = 'vision-encoder-decoder'
        decoder.config.use_learned_position_embeddings = False
        run(8192)  # Sinusoidal positions expand; the initial512 configuration is not a hard cap.
        decoder.config.use_learned_position_embeddings = True
        # The loaded position table is authoritative when smaller than configuration.
        positions.weight.shape = (10, 8)
        run(7)
        try: run(8)
        except ValueError: pass
        else: raise AssertionError('Actual position table was ignored')
        positions.weight.shape = (514, 8)
        for values, fits in [([2, 9, 8], 509), ([9, 8], 509)]:
            backend.processor = lambda **kw: {'pixel_values': object(), 'decoder_input_ids': Prefix(values)}
            run(fits)
            before = len(generated)
            try: run(fits + 1)
            except ValueError: pass
            else: raise AssertionError('Prepared decoder prefix was not counted')
            assert len(generated) == before
        # Cancellation during preprocessing must not start a generation/forward.
        cancel = threading.Event()
        backend.processor = lambda **kw: cancel.set() or {'pixel_values': object()}
        for task in ['image-to-text', 'object-detection']:
            cancel.clear(); backend.spec.task = task
            before = len(generated)
            with patch.dict(sys.modules, {'torch': NS()}):
                try: backend.infer(task, {'image': Image()}, cancel)
                except EngineError as exc: assert exc.code == 'cancelled'
                else: raise AssertionError('Cancelled preprocessing ran the model')
            assert len(generated) == before


def app_for(folder):
    return create_app(Config(models={'ocr': ModelSpec(path=folder, task='image-to-text', device='cpu'),
        'detect': ModelSpec(path=folder, task='object-detection', device='cpu')}), output_dir=Path(folder))


def result(task):
    return {'text': 'line one\nline two'} if task == 'image-to-text' else {
        'objects': [], 'width': 640, 'height': 96, 'threshold': 0.0}


def routes():
    for op, model, direct in [('ocr', 'ocr', '/v1/images/ocr'),
                              ('detection', 'detect', '/v1/images/detect')]:
        for route in [direct, '/v1/extensions/infer']:
            yield op, model, route


def payload_for(op, model, route, **values):
    payload = {'model': model, 'image_base64': 'AA==', **values}
    if route.endswith('/infer'): payload['operation'] = op
    return payload


def decode_deadline_and_cleanup():
    with tempfile.TemporaryDirectory() as folder:
        app = app_for(folder); calls = []; images = []; slow = False
        def infer(name, task, data, emit=None, cancel=None):
            calls.append(data)
            return result(task)
        app.state.engine.target.infer = infer
        def decode(payload):
            image = Image(); images.append(image)
            if slow: time.sleep(.03)
            return image
        with patch('titanos.api_plus._image', side_effect=decode), TestClient(app, base_url='http://127.0.0.1') as client:
            for op, model, route in routes():
                extra = {'max_tokens': 7, 'temperature': 0} if op == 'ocr' else {'threshold': 0}
                response = client.post(route, json=payload_for(op, model, route, **extra))
                assert response.status_code == 200, response.text
                assert images[-1].closed
                if op == 'ocr':
                    assert response.json()['lines'] == ['line one', 'line two']
                    assert calls[-1]['max_tokens'] == 7 and calls[-1]['temperature'] == 0
                else: assert calls[-1]['threshold'] == 0
                before = len(calls)
                slow = True
                response = client.post(route, json=payload_for(op, model, route, timeout=5))
                assert response.status_code == 504 and response.json()['error']['code'] == 'deadline_exceeded'
                assert len(calls) == before and images[-1].closed
                slow = False
            # An already cancelled/expired operation cannot even decode an image.
            cancel = threading.Event(); cancel.set()
            for operation, cls, model in [('ocr', OCRRequest, 'ocr'), ('detection', DetectionRequest, 'detect')]:
                before = len(images)
                for event, deadline, code in [(cancel, time.monotonic()+5, 'cancelled'),
                                               (None, time.monotonic()-1, 'deadline_exceeded')]:
                    try: app.state.extension_handlers[operation](cls(model=model, image_base64='AA=='), event, deadline)
                    except EngineError as exc: assert exc.code == code
                    else: raise AssertionError('Late image request decoded input')
                assert len(images) == before


async def queue_deadline():
    with tempfile.TemporaryDirectory() as folder:
        app = app_for(folder); actual_to_thread = asyncio.to_thread
        async def delayed(function, *args, **kwargs):
            await asyncio.sleep(.03)
            return await actual_to_thread(function, *args, **kwargs)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
                with patch('titanos.api_plus._image', side_effect=AssertionError('Expired queue decoded an image')), \
                        patch('titanos.api_plus.asyncio.to_thread', side_effect=delayed):
                    for op, model, route in routes():
                        response = await client.post(route, json=payload_for(op, model, route, timeout=5))
                        assert response.status_code == 504, response.text
        finally:
            await app.state.native.close()
            await asyncio.to_thread(app.state.engine.close)


async def cancellation():
    # Direct/extension ASGI disconnect and actual MCP task cancellation, both adapters.
    for op, model, route in routes():
        modes = ['http_disconnect', 'mcp_task_cancel'] if route.endswith('/infer') else ['http_disconnect']
        for mode in modes:
            started, stopped, observed = threading.Event(), threading.Event(), threading.Event()
            images = []
            def infer(name, task, data, emit=None, cancel=None):
                started.set()
                try:
                    assert cancel.wait(2), 'Caller cancellation did not reach model worker'
                    observed.set()
                    return result(task)  # A late normal result must never become a successful response.
                finally: stopped.set()
            def decode(payload):
                image = Image(); images.append(image); return image
            with tempfile.TemporaryDirectory() as folder:
                app = app_for(folder); app.state.engine.target.infer = infer
                payload = payload_for(op, model, route, timeout=5000)
                try:
                    with patch('titanos.api_plus._image', side_effect=decode):
                        if mode == 'http_disconnect':
                            content = json.dumps(payload).encode(); first = True; sent = []
                            async def receive():
                                nonlocal first
                                if first:
                                    first = False
                                    return {'type': 'http.request', 'body': content, 'more_body': False}
                                while not started.is_set(): await asyncio.sleep(.005)
                                return {'type': 'http.disconnect'}
                            async def send(message): sent.append(message)
                            scope = {'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
                                'scheme':'http','path':route,'raw_path':route.encode(),'root_path':'','query_string':b'',
                                'client':('127.0.0.1',50000),'server':('127.0.0.1',80),
                                'headers':[(b'host',b'127.0.0.1'),(b'content-type',b'application/json')]}
                            await asyncio.wait_for(app(scope, receive, send), 3)
                            assert next(m['status'] for m in sent if m['type'] == 'http.response.start') == 499
                            body = json.loads(b''.join(m.get('body', b'') for m in sent))
                            assert body['error']['code'] == 'cancelled'
                        else:
                            pending = asyncio.create_task(app.state.mcp.call_tool('extension_infer', {'payload':payload}))
                            assert await asyncio.to_thread(started.wait, 1)
                            pending.cancel()
                            try: await asyncio.wait_for(pending, 3)
                            except asyncio.CancelledError: pass
                            else: raise AssertionError('MCP cancellation was swallowed')
                        assert stopped.is_set() and observed.is_set() and images[-1].closed
                finally:
                    await app.state.native.close()
                    await asyncio.to_thread(app.state.engine.close)


if __name__ == '__main__':
    adapter_budgets(); decode_deadline_and_cleanup()
    asyncio.run(queue_deadline()); asyncio.run(cancellation())
    assert 'torch' not in sys.modules, 'This suite must not import the real model runtime'
    print('PASS: request/model/process OCR budgets, actual TrOCR table/prefix capacity, pre-forward cancellation, '
          'decode/queue deadlines and cleanup, direct/extension ASGI disconnect and MCP task cancellation; no Torch/models')
