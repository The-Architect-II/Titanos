"""Deterministic filter semantics and cross-interface controls; no model runtime."""
from contextlib import redirect_stdout
import asyncio
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

from fastapi.testclient import TestClient
from titanos.config import Config, ModelSpec
from titanos.detection import options, suppress
from titanos.engine import EngineError
from titanos.server import create_app
from titanos.settings import Settings


def item(label, score, x=0):
    return {'label': str(label), 'label_id': label, 'score': score,
            'box': dict(zip(('xmin', 'ymin', 'xmax', 'ymax'), (x, 0, x + 10, 10)))}


def main():
    objects = [item(1, .6), item(1, .9), item(2, .8), item(1, .7, 20), item(1, .9)]
    original = deepcopy(objects)
    kept, record = suppress(objects, .7)
    assert kept == [objects[1], objects[2], objects[3]]
    assert record['retained_indices'] == [1, 2, 3] and record['removed_indices'] == [0, 4]
    assert objects == original and not record['scores_modified'] and not record['boxes_clipped']
    assert suppress(objects, 1)[0] == objects
    assert suppress([item(1, .9), item(1, .8, 10)], 0)[0] == [item(1, .9), item(1, .8, 10)]
    assert suppress([], .7)[0] == []
    cancelled = threading.Event(); cancelled.set()
    for values in [objects, []]:
        try: suppress(values, .7, cancelled)
        except EngineError as exc: assert exc.code == 'cancelled'
        else: raise AssertionError('Cancelled postprocess succeeded')
    class DuringPass:
        calls = 0
        def is_set(self): self.calls += 1; return self.calls > 8
    try: suppress(objects, .7, DuringPass())
    except EngineError as exc: assert exc.code == 'cancelled' and objects == original
    else: raise AssertionError('Cancellation inside the filter was ignored')
    for bad in [dict(item(1, .9), score=float('nan')), dict(item(1, .9), score=1.1), item(-1, .9),
                dict(item(1, .9), box={'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 1})]:
        try: suppress([bad], .7)
        except EngineError as exc: assert exc.status == 502 and exc.code == 'invalid_backend_response'
        else: raise AssertionError('Invalid prediction accepted')
    settings = Settings.from_env({'TITANOS_DETECTION_NMS': 'true', 'TITANOS_DETECTION_NMS_IOU': '0', 'TITANOS_DETECTION_THRESHOLD': '0'})
    with patch('titanos.detection.current_settings', return_value=settings):
        assert options({}, {}) == {'nms': True, 'nms_iou': 0, 'threshold': 0}
        assert options({'nms': False, 'nms_iou': 0}, {'nms': True, 'nms_iou': .8}) == {'nms': False, 'nms_iou': 0, 'threshold': 0}
        assert options({}, {'nms': None, 'nms_iou': .2})['nms_iou'] == .2
        for invalid in [{'nms': 'false'}, {'nms_iou': float('nan')}, {'threshold': True}]:
            try: options({}, invalid)
            except ValueError: pass
            else: raise AssertionError('Invalid model policy accepted')
    for values in [{'detection_nms_iou': 1.1}, {'detection_threshold': -1}, {'detection_nms': 'false'}]:
        try: Settings(**values)
        except ValueError: pass
        else: raise AssertionError('Invalid process policy accepted')

    class Image:
        def close(self): pass
    with tempfile.TemporaryDirectory() as directory:
        app = create_app(Config(models={'detector': ModelSpec(path=directory, task='object-detection', device='cpu')}),
                         output_dir=Path(directory), settings=Settings())
        received = []
        def infer(name, task, data, emit=None, cancel=None):
            received.append(data)
            return {'objects': [], 'width': 10, 'height': 10, 'threshold': data['threshold']}
        app.state.engine.target.infer = infer
        with patch('titanos.api_plus._image', return_value=Image()), TestClient(app, base_url='http://127.0.0.1') as client:
            for route in ['/v1/images/detect', '/v1/extensions/infer']:
                body = {'model': 'detector', 'image_base64': 'AA==', 'threshold': 0, 'nms': False, 'nms_iou': 0}
                if route.endswith('/infer'): body['operation'] = 'detection'
                response = client.post(route, json=body)
                assert response.status_code == 200 and received[-1]['nms'] is False and received[-1]['nms_iou'] == 0
                before = len(received)
                for change in [{'nms': 'false'}, {'nms_iou': True}, {'nms_iou': 1.1}]:
                    assert client.post(route, json={**body, **change}).status_code == 422
                assert len(received) == before
            before = len(received)
            asyncio.run(app.state.mcp.call_tool('extension_infer', {'payload': {
                'operation': 'detection', 'model': 'detector', 'image_base64': 'AA==',
                'threshold': 0, 'nms': False, 'nms_iou': 0}}))
            assert len(received) == before + 1 and received[-1]['nms'] is False and received[-1]['nms_iou'] == 0

        from titanos.cli import main as cli
        class Client:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def detect(self, model, image, **kwargs): received.append(kwargs); return {}
        with patch('titanos.client.Client', Client), redirect_stdout(io.StringIO()):
            cli(['detect', 'detector', str(Path(directory) / 'image.png'), '--no-nms', '--nms-iou', '0', '--threshold', '0'])
        assert received[-1]['nms'] is False and received[-1]['nms_iou'] == 0 and received[-1]['threshold'] == 0
    assert 'torch' not in sys.modules
    print(json.dumps({'passed': True, 'scope': 'Filter selection/order/ties/classes, explicit policy inheritance and invalid/cancelled input, HTTP/extension controls; no model inference'}))


if __name__ == '__main__':
    main()
