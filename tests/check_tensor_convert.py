"""Tensor input conversion and output views on /api/infer: a uint8 input with "cast": "float32" is converted to float32
(a plain cast, or with "normalize" {mean, std, axis} (u8 - mean) / std in float64 along the axis with one final cast),
and "output_views" returns a slice of an output before encoding; a real ONNX Runtime CPU graph built here, in process and
through the replica worker entry.

The conversion must be bit-identical to the reference client formula (its expand() and rebuild(), copied below
verbatim), so a client may send a quarter of the bytes without changing a single float the graph sees.
"""
import base64
import json
from multiprocessing import Pipe
import os
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch
import warnings

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(f'check_tensor_convert needs {exc.name}') from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
HOME = tempfile.TemporaryDirectory(prefix='titanos-tensor-convert-home-', ignore_cleanup_errors=True)
os.environ['TITANOS_HOME'] = HOME.name  # nothing from this check reaches ~/.titanos

from fastapi.testclient import TestClient  # noqa: E402
from titanos import compiled, replicas, server  # noqa: E402
from titanos.compiled import CompiledModel, tensor_bytes, tensor_input, tensor_payload, tensor_result  # noqa: E402
from titanos.config import Config, ModelSpec, WorkerSpec  # noqa: E402
from titanos.settings import Settings, current_settings  # noqa: E402

# rtmpose's normalisation as the reference client holds it (float64 numpy arrays, RGB order applied to BGR pixels)
MEAN, STD = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])
NORM = {'mean': MEAN.tolist(), 'std': STD.tolist(), 'axis': 1}


# --- the reference client formula, verbatim from the client ---
def _channel(norm, ndim):
    shape = [1] * ndim
    shape[norm["axis"]] = -1
    return (np.asarray(norm["mean"], np.float64).reshape(shape), np.asarray(norm["std"], np.float64).reshape(shape))


def rebuild(pixels, norm):
    if norm is None:
        return pixels.astype(np.float32)
    mean, std = _channel(norm, pixels.ndim)
    return ((pixels.astype(np.float64) - mean) / std).astype(np.float32)


def expand(entry):
    raw = np.frombuffer(base64.b64decode(entry["b64"], validate=True), dtype=np.dtype(entry["dtype"]).newbyteorder("<"))
    raw = raw.astype(np.dtype(entry["dtype"])).reshape(entry["shape"])
    if "cast" not in entry:
        return raw
    if entry["dtype"] != "uint8" or entry["cast"] != "float32":
        raise ValueError("cast: uint8 to float32 only")
    return rebuild(raw, entry.get("normalize"))
# --- end of the client copy ---


def refused(call, text):
    try:
        call()
    except ValueError as exc:
        assert text in str(exc), f'{text!r} not in {exc}'
        return exc
    raise AssertionError(f'accepted; expected ValueError {text!r}')


def b64(array, **extra):
    array = np.asarray(array)
    raw = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder('<')).tobytes()
    return {'dtype': array.dtype.name, 'shape': list(array.shape), 'b64': base64.b64encode(raw).decode('ascii'), **extra}


def listed(array, **extra):
    array = np.asarray(array)
    return {'dtype': array.dtype.name, 'shape': list(array.shape), 'data': array.reshape(-1).tolist(), **extra}


def u8(form, array, norm=None):
    """A uint8 entry with cast float32 and, given norm, normalize."""
    return form(array, cast='float32', **({'normalize': norm} if norm is not None else {}))


def decoded(value):
    raw = base64.b64decode(value['b64'], validate=True)
    return np.frombuffer(raw, dtype=np.dtype(value['dtype']).newbyteorder('<')).reshape(value['shape'])


def client_crops(hwc_crops):
    """The reference client's pose arithmetic (its crop and call): float64 maths, one float32 cast."""
    return np.ascontiguousarray(np.stack([((img - MEAN) / STD).transpose(2, 0, 1) for img in hwc_crops]), dtype=np.float32)


def bits(array):
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def legacy_input(value):
    """tensor_input before cast/normalize existed, verbatim: the reference for byte-equal unconverted requests."""
    if isinstance(value, dict) and set(value) == {'dtype', 'shape', 'b64'}:
        return tensor_bytes(value)
    return tensor_payload(value)


def legacy_infer(self, task, data, emit=None, cancel=None):
    """CompiledModel.infer's tensor branch before cast/normalize and output_views existed, verbatim."""
    values = self.run({k: legacy_input(v) for k, v in data['inputs'].items()})
    limit = current_settings().max_tensor_elements
    if sum(v.size for v in values.values()) > limit:
        raise ValueError(f"Combined tensor output exceeds {limit} elements")
    return {'outputs': {k: tensor_result(v, data.get('output_encoding') or 'list') for k, v in values.items()},
            'execution': self.execution}


def write_graph(path):
    """pix (float32 [n, 3, h, w]) passed through, img (uint8 [2, 2]) passed through."""
    nodes = [helper.make_node('Identity', ['pix'], ['pix_out']), helper.make_node('Identity', ['img'], ['img_out'])]
    value = helper.make_tensor_value_info
    inputs = [value('pix', TensorProto.FLOAT, ['n', 3, 'h', 'w']), value('img', TensorProto.UINT8, [2, 2])]
    outputs = [value('pix_out', TensorProto.FLOAT, ['n', 3, 'h', 'w']), value('img_out', TensorProto.UINT8, [2, 2])]
    model = helper.make_model(helper.make_graph(nodes, 'titanos-tensor-convert-check', inputs, outputs),
                              opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def every_pixel():
    """One crop (h=1, w=256) holding every uint8 value in each of the three channels: all 768 (channel, value) pairs."""
    return np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1)[None]  # hwc [1, 256, 3]


def check_codec():
    """(a) every (channel, value) pair and every plain cast bit-equal to the client's expand()/rebuild(); (e) at the codec."""
    hwc = every_pixel()
    chw = hwc.transpose(2, 0, 1)[None]  # [1, 3, 1, 256] uint8
    expected = client_crops([hwc])
    assert np.array_equal(bits(rebuild(chw, NORM)), bits(expected))  # the client's two formulas agree with each other
    for form in (b64, listed):
        converted = tensor_input(u8(form, chw, NORM))
        assert converted.dtype == np.float32 and converted.shape == (1, 3, 1, 256)
        assert np.array_equal(bits(converted), bits(rebuild(chw, NORM))), 'normalisation differs from rebuild() in some bit'
        assert np.array_equal(bits(converted), bits(expand(u8(b64, chw, NORM))))
        assert np.array_equal(bits(tensor_input(u8(form, chw, {**NORM, 'axis': -3}))), bits(expected))
        values = np.arange(256, dtype=np.uint8)
        plain = tensor_input(u8(form, values))
        assert plain.dtype == np.float32 and np.array_equal(bits(plain), bits(rebuild(values, None)))
        assert np.array_equal(bits(plain), bits(expand(u8(b64, values)))) and np.array_equal(plain, np.arange(256))
    lut = ((np.arange(256, dtype=np.float64)[None] - MEAN[:, None]) / STD[:, None]).astype(np.float32)  # float32[3, 256]
    assert np.array_equal(bits(lut[:, None, None, :].transpose(1, 0, 2, 3)), bits(expected))
    # negative control: the same formula in float32 arithmetic differs in some bit, so the float64 path is what matches
    float32_maths = (chw.astype(np.float32) - MEAN.astype(np.float32)[:, None, None]) / STD.astype(np.float32)[:, None, None]
    assert not np.array_equal(bits(float32_maths), bits(expected))
    batch = np.random.default_rng(7).integers(0, 256, (3, 5, 4, 3), dtype=np.uint8)  # three hwc crops
    sent = np.stack([img.transpose(2, 0, 1) for img in batch])
    assert np.array_equal(bits(tensor_input(u8(b64, sent, NORM))), bits(client_crops(batch)))
    last = tensor_input(u8(b64, batch, {**NORM, 'axis': -1}))  # channels last
    assert np.array_equal(bits(last), bits(((batch - MEAN) / STD).astype(np.float32)))
    assert np.array_equal(bits(last), bits(expand(u8(b64, batch, {**NORM, 'axis': -1}))))
    ints = tensor_input(u8(b64, chw, {'mean': [0, 1, 2], 'std': [1, 2, 3], 'axis': 1}))
    assert np.array_equal(ints, ((chw - np.array([0., 1, 2])[:, None, None]) / np.array([1., 2, 3])[:, None, None]).astype(np.float32))
    empty = tensor_input(u8(b64, np.zeros((0, 3), np.uint8), {'mean': [0, 0, 0], 'std': [1, 1, 1], 'axis': 1}))
    assert empty.shape == (0, 3) and empty.dtype == np.float32
    # finite stats whose result overflows float32 are refused, without a numpy overflow warning
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for norm in ({**NORM, 'std': [1e-300, 1, 1]}, {**NORM, 'mean': [1e41, 0, 0]}, {**NORM, 'std': [-1e-300, 1, 1]}):
            try:
                tensor_input(u8(b64, chw, norm))
            except ValueError as exc:
                assert str(exc) == 'Tensor normalize result must be finite in float32', exc
            else:
                raise AssertionError(f'overflowing normalize accepted: {norm}')
    # without cast the entry is decoded as before, whatever its dtype
    assert tensor_input(b64(chw)).dtype == np.uint8 and tensor_input(listed(chw)).dtype == np.uint8

    def bad(entry_extra, array=chw, form=b64):
        return lambda: tensor_input(form(array, **entry_extra))
    for form in (b64, listed):
        refused(bad({'cast': 'float32', 'normalize': {**NORM, 'scale': 2}}, form=form), 'Tensor normalize takes exactly mean, std and axis')
        refused(bad({'normalize': NORM}, form=form), 'Tensor normalize requires cast "float32"')
        refused(bad({'cast': 'float16'}, form=form), 'Tensor cast converts a uint8 input to "float32" only')
        refused(bad({'cast': 'float32'}, chw.astype(np.float32), form), 'Tensor cast converts a uint8 input')
        refused(bad({'cast': 'float32', 'extra': 1}, form=form), 'Each tensor requires exactly dtype, shape')
    refused(bad({'cast': None}), 'Tensor cast converts a uint8 input')
    refused(bad({'cast': ['float32']}), 'Tensor cast converts a uint8 input')
    refused(bad({'cast': 'float32'}, chw.astype(np.int8)), 'Tensor cast converts a uint8 input')
    refused(bad({'cast': 'float32', 'normalize': NORM}, chw.astype(np.float32)), 'Tensor cast converts a uint8 input')
    refused(bad({'cast': 'float32', 'normalize': None}), 'Tensor normalize takes exactly mean, std and axis')
    refused(bad({'cast': 'float32', 'normalize': 'imagenet'}), 'Tensor normalize takes exactly mean, std and axis')
    refused(bad({'cast': 'float32', 'normalize': {'mean': NORM['mean'], 'std': NORM['std']}}), 'Tensor normalize takes exactly')
    refused(bad({'cast': 'float32', 'normalize': {'mean': NORM['mean'], 'axis': 1}}), 'Tensor normalize takes exactly')
    refused(bad({'cast': 'float32', 'normalize': {'std': NORM['std'], 'axis': 1}}), 'Tensor normalize takes exactly')
    for axis in (4, -5, None, 1.0, True, '1'):
        refused(bad({'cast': 'float32', 'normalize': {**NORM, 'axis': axis}}), 'Tensor normalize axis must be an integer within the 4 input axes')
    refused(bad({'cast': 'float32', 'normalize': {**NORM, 'axis': 3}}), 'Tensor normalize mean must list 256 numbers, one per index of axis 3')
    refused(bad({'cast': 'float32', 'normalize': {**NORM, 'mean': MEAN.tolist()[:2]}}), 'Tensor normalize mean must list 3 numbers')
    refused(bad({'cast': 'float32', 'normalize': {**NORM, 'std': STD.tolist() + [1.0]}}), 'Tensor normalize std must list 3 numbers')
    refused(bad({'cast': 'float32', 'normalize': {**NORM, 'mean': 'abc'}}), 'Tensor normalize mean must list 3 numbers')
    refused(bad({'cast': 'float32', 'normalize': {**NORM, 'mean': [1, True, 2]}}), 'Tensor normalize mean must list 3 numbers')
    refused(bad({'cast': 'float32', 'normalize': {**NORM, 'std': [1, '2', 3]}}), 'Tensor normalize std must list 3 numbers')
    refused(bad({'cast': 'float32', 'normalize': {**NORM, 'std': [58.395, 0, 57.375]}}), 'Tensor normalize std must not contain 0')
    refused(bad({'cast': 'float32', 'normalize': {**NORM, 'std': [58.395, -0.0, 57.375]}}), 'Tensor normalize std must not contain 0')
    for value in (float('nan'), float('inf'), -float('inf'), 10 ** 400):
        refused(bad({'cast': 'float32', 'normalize': {**NORM, 'mean': [value, 1, 2]}}), 'Tensor normalize mean values must be finite')
        refused(bad({'cast': 'float32', 'normalize': {**NORM, 'std': [1, value, 2]}}), 'Tensor normalize std values must be finite')
    limit = current_settings().max_tensor_elements  # the source's element limit still applies
    refused(lambda: tensor_input({'dtype': 'uint8', 'shape': [limit + 1], 'b64': '', 'cast': 'float32'}),
            f'contain at most {limit} elements')
    refused(lambda: tensor_input({'dtype': 'uint8', 'shape': [limit + 1], 'data': [], 'cast': 'float32'}),
            f'contain at most {limit} elements')
    refused(lambda: tensor_input({'dtype': 'uint8', 'shape': [3], 'b64': 'AAAAAA==', 'cast': 'float32'}),
            'must encode exactly 3 bytes')
    refused(lambda: tensor_input({'dtype': 'uint8', 'shape': [2], 'data': [0, 256], 'cast': 'float32'}),
            'outside its dtype range')

    views = compiled.output_views({'y': {'slice': [[0, 1], [None, None], [-2, None]]}}, ['y'])
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    viewed = compiled.output_view(value, 'y', views['y'])
    assert viewed.flags.c_contiguous and np.array_equal(viewed, value[0:1, :, -2:])
    assert compiled.output_views(None, ['y']) == {} and compiled.output_views({}, ['y']) == {}
    refused(lambda: compiled.output_view(value, 'y', views['y'][:2]), 'output_views y needs one [start, stop] pair per axis')


class Fixture:
    def __init__(self, scratch, workers=None):
        write_graph(scratch / 'model.onnx')
        self.config = Config(models={'tens': ModelSpec(path=str(scratch), task='tensor', backend='onnx', device='cpu')},
                             workers=workers or {})
        self.app = server.create_app(self.config, output_dir=scratch / 'out', settings=Settings(), runtime_catalog=False)
        if not workers:
            engine = self.app.state.engine.target
            engine.runtime = CompiledModel(self.config.models['tens'])
            engine.current, engine.states['tens'] = 'tens', 'loaded'
        self.client = TestClient(self.app, base_url='http://127.0.0.1:8000')
        self.client.__enter__()

    def infer(self, inputs, **fields):
        return self.client.post('/api/infer', json={'model': 'tens', 'inputs': inputs, **fields})

    def raw(self, text):
        return self.client.post('/api/infer', content=text.encode(), headers={'content-type': 'application/json'})

    def close(self):
        self.client.__exit__(None, None, None)
        self.app.state.engine.target.close()


IMG = np.array([[0, 255], [17, 128]], dtype=np.uint8)


def check_route(fixture):
    rng = np.random.default_rng(11)
    crops = rng.integers(0, 256, (2, 6, 5, 3), dtype=np.uint8)  # two hwc crops
    chw = np.stack([img.transpose(2, 0, 1) for img in crops])
    expected = client_crops(crops)

    def ok(response):
        assert response.status_code == 200, response.text
        return response

    def failed(response, status, text):
        assert response.status_code == status, (response.status_code, response.text)
        assert text in response.json()['error']['message'], (text, response.json())

    # (b) without cast/normalize or output_views: the same bytes as the code before they existed
    plain = {'pix': b64(expected), 'img': b64(IMG)}
    for encoding in ('list', 'base64'):
        current = ok(fixture.infer(plain, output_encoding=encoding)).content
        with patch.object(CompiledModel, 'infer', legacy_infer):
            legacy = ok(fixture.infer(plain, output_encoding=encoding)).content
            legacy_listed = ok(fixture.infer({'pix': listed(expected), 'img': listed(IMG)}, output_encoding=encoding)).content
        assert current == legacy, encoding
        assert ok(fixture.infer({'pix': listed(expected), 'img': listed(IMG)}, output_encoding=encoding)).content == legacy_listed
        assert legacy_listed == current
        assert ok(fixture.infer(plain, output_encoding=encoding, output_views={})).content == current
        assert ok(fixture.infer(plain, output_encoding=encoding, output_views=None)).content == current

    # (c) uint8 + cast (+ normalize) answers the bytes of the client's own float32 request
    for encoding in ('list', 'base64'):
        old_style = ok(fixture.infer(plain, output_encoding=encoding)).content
        for form in (b64, listed):
            converted = ok(fixture.infer({'pix': u8(form, chw, NORM), 'img': b64(IMG)}, output_encoding=encoding))
            assert converted.content == old_style, (encoding, form.__name__)
        cast = ok(fixture.infer({'pix': u8(b64, chw), 'img': b64(IMG)}, output_encoding=encoding)).content
        assert cast == ok(fixture.infer({'pix': b64(chw.astype(np.float32)), 'img': b64(IMG)}, output_encoding=encoding)).content
    sent = ok(fixture.infer({'pix': u8(b64, chw, NORM), 'img': b64(IMG)}, output_encoding='base64'))
    assert np.array_equal(bits(decoded(sent.json()['outputs']['pix_out'])), bits(expand(u8(b64, chw, NORM))))
    every = every_pixel()
    sent = ok(fixture.infer({'pix': u8(b64, every.transpose(2, 0, 1)[None], NORM), 'img': b64(IMG)}, output_encoding='base64'))
    assert np.array_equal(bits(decoded(sent.json()['outputs']['pix_out'])), bits(client_crops([every])))  # all 768

    # (d) the graph's dtype check still runs, after conversion
    failed(fixture.infer({'pix': b64(chw), 'img': b64(IMG)}), 400, 'Input pix has incorrect dtype or rank')
    failed(fixture.infer({'pix': listed(chw), 'img': b64(IMG)}), 400, 'Input pix has incorrect dtype or rank')
    failed(fixture.infer({'pix': u8(b64, chw, NORM), 'img': u8(b64, IMG)}), 400, 'Input img has incorrect dtype or rank')

    # (e) every malformed cast/normalize is a 400 naming it
    for extra, text in (({'cast': 'float32', 'normalize': {**NORM, 'scale': 1}}, 'Tensor normalize takes exactly mean, std and axis'),
                        ({'cast': 'float32', 'normalize': {'mean': NORM['mean'], 'axis': 1}}, 'Tensor normalize takes exactly'),
                        ({'normalize': NORM}, 'Tensor normalize requires cast "float32"'),
                        ({'cast': 'float64'}, 'Tensor cast converts a uint8 input to "float32" only'),
                        ({'cast': 'float32', 'normalize': {**NORM, 'mean': [1, 2]}}, 'Tensor normalize mean must list 3 numbers'),
                        ({'cast': 'float32', 'normalize': {**NORM, 'std': [1, 0, 2]}}, 'Tensor normalize std must not contain 0'),
                        ({'cast': 'float32', 'normalize': {**NORM, 'axis': 4}}, 'Tensor normalize axis must be an integer within the 4'),
                        ({'cast': 'float32', 'normalize': {**NORM, 'axis': -5}}, 'Tensor normalize axis must be an integer within the 4'),
                        ({'cast': 'float32', 'normalize': {**NORM, 'axis': '1'}}, 'Tensor normalize axis must be an integer'),
                        ({'cast': 'float32', 'normalize': {**NORM, 'mean': [10 ** 400, 1, 2]}}, 'Tensor normalize mean values must be finite'),
                        ({'cast': 'float32', 'normalize': {**NORM, 'std': [1e-300, 1, 1]}}, 'Tensor normalize result must be finite in float32'),
                        ({'cast': 'float32', 'normalize': {**NORM, 'mean': [1e41, 0, 0]}}, 'Tensor normalize result must be finite in float32'),
                        ({'cast': 'float32', 'scale': 2}, 'Each tensor requires exactly dtype, shape')):
        failed(fixture.infer({'pix': b64(chw, **extra), 'img': b64(IMG)}), 400, text)
    failed(fixture.infer({'pix': b64(chw.astype(np.float32), cast='float32'), 'img': b64(IMG)}), 400,
           'Tensor cast converts a uint8 input')
    for token in ('NaN', 'Infinity', '-Infinity'):  # non-finite JSON numbers reach the check as floats
        body = {'model': 'tens', 'inputs': {'pix': u8(b64, chw, {**NORM, 'std': [1, 2, '@']}), 'img': b64(IMG)}}
        failed(fixture.raw(json.dumps(body).replace('"@"', token)), 400, 'Tensor normalize std values must be finite')

    # (f) output_views: the numpy slice before encoding; malformed views refused by name
    values = {'pix_out': expected, 'img_out': IMG}
    for spec in ({'pix_out': {'slice': [[0, 1], [None, None], [1, 4], [None, 3]]}},
                 {'pix_out': {'slice': [[-1, None], [2, 3], [None, -2], [0, 5]]}, 'img_out': {'slice': [[1, 2], [None, None]]}},
                 {'img_out': {'slice': [[0, 0], [None, None]]}}):
        for encoding in ('list', 'base64'):
            response = ok(fixture.infer({'pix': u8(b64, chw, NORM), 'img': b64(IMG)}, output_encoding=encoding,
                                        output_views=spec)).json()
            for name, value in values.items():
                want = value[tuple(slice(a, b) for a, b in spec[name]['slice'])] if name in spec else value
                got = response['outputs'][name]
                assert got['shape'] == list(want.shape) and got['dtype'] == want.dtype.name, (name, got['shape'])
                assert (got == tensor_result(want, 'list')) if encoding == 'list' else (
                    np.array_equal(decoded(got), want) and got == tensor_result(np.ascontiguousarray(want), 'base64'))
    good = {'pix': u8(b64, chw, NORM), 'img': b64(IMG)}
    for views, text in (({'nope': {'slice': [[0, 1]]}}, "output_views names ['nope']; the graph outputs are"),
                        ({'pix_out': {'slice': [[0, 1]]}}, 'output_views pix_out needs one [start, stop] pair per axis'),
                        ({'img_out': {'slice': [[0, 1], [0, 1], [0, 1]]}}, 'output_views img_out needs one [start, stop]'),
                        ({'img_out': {'slice': [[0, 1.5], [0, 1]]}}, 'output_views img_out takes exactly {"slice"'),
                        ({'img_out': {'slice': [[0, True], [0, 1]]}}, 'output_views img_out takes exactly'),
                        ({'img_out': {'slice': [['0', 1], [0, 1]]}}, 'output_views img_out takes exactly'),
                        ({'img_out': {'slice': [[0, 1, 1], [0, 1]]}}, 'output_views img_out takes exactly'),
                        ({'img_out': {'slice': [[0], [0, 1]]}}, 'output_views img_out takes exactly'),
                        ({'img_out': {'slice': [[0, 1], [0, 1]], 'reduce': 'argmax'}}, 'output_views img_out takes exactly'),
                        ({'img_out': {}}, 'output_views img_out takes exactly'),
                        ({'img_out': {'slice': '0:1'}}, 'output_views img_out takes exactly')):
        failed(fixture.infer(good, output_views=views), 400, text)
    failed(fixture.infer(good, output_views={'img_out': [[0, 1], [0, 1]]}), 422, 'output_views')  # schema: a dict per output
    failed(fixture.infer(good, output_views=[1]), 422, 'output_views')


class ThreadProcess:
    """A worker "process" running the real worker entry on a thread over a real multiprocessing pipe (pickled both ways);
    no pid, so no PID-based stop can ever target this check's own process."""
    def __init__(self, target, args):
        self.thread = threading.Thread(target=target, args=args, daemon=True)
        self.pid = None
        self.payloads = args[2:]

    def start(self):
        self.thread.start()

    def is_alive(self):
        return self.thread.is_alive()

    def join(self, timeout=None):
        self.thread.join(timeout)

    def terminate(self):
        pass

    kill = terminate


class ThreadContext:
    def __init__(self):
        self.processes, self.requests = [], []

    def Pipe(self):
        parent, child = Pipe()
        requests, self.children = self.requests, [child]

        class Recorded:  # the worker side records every request envelope it receives
            def recv(self):
                message = child.recv()
                requests.append(message)
                return message

            def close(self):  # the parent closes its copy of a spawned child's end; here both share one
                pass

            def __getattr__(self, name):
                return getattr(child, name)
        return parent, Recorded()

    def Event(self):
        return threading.Event()

    def Process(self, *, target, args, **kwargs):
        assert target is replicas.worker_main
        self.processes.append(ThreadProcess(target, args))
        return self.processes[-1]


def check_replica(scratch):
    """(g) cast/normalize and output_views cross the replica worker pipe and run in the worker's own Engine and CompiledModel."""
    rng = np.random.default_rng(13)
    crops = rng.integers(0, 256, (1, 4, 3, 3), dtype=np.uint8)
    chw = np.stack([img.transpose(2, 0, 1) for img in crops])
    context = ThreadContext()
    seen = []
    real_infer = CompiledModel.infer

    def observed(self, task, data, emit=None, cancel=None):
        seen.append((threading.current_thread().name, data))
        return real_infer(self, task, data, emit, cancel)
    with patch.object(replicas.mp, 'get_context', return_value=context), \
            patch.object(CompiledModel, 'infer', observed), patch.dict(os.environ):
        fixture = Fixture(scratch, workers={'cpu': WorkerSpec(models=['tens'], threads=1, device='cpu')})
        try:
            views = {'pix_out': {'slice': [[None, None], [1, 3], [0, 2], [-1, None]]}}
            body = {'pix': u8(b64, chw, NORM), 'img': u8(listed, IMG)}
            response = fixture.infer({**body, 'img': b64(IMG)}, output_encoding='base64', output_views=views)
            assert response.status_code == 200, response.text
            want = client_crops(crops)[:, 1:3, 0:2, -1:]
            got = decoded(response.json()['outputs']['pix_out'])
            assert np.array_equal(bits(got), bits(want)) and got.shape == (1, 2, 2, 1)
            assert np.array_equal(bits(got), bits(expand(u8(b64, chw, NORM))[:, 1:3, 0:2, -1:]))
            assert np.array_equal(decoded(response.json()['outputs']['img_out']), IMG)
            refusal = fixture.infer(body)  # the worker's conversion feeds the graph check: img must stay uint8
            assert refusal.status_code == 400 and 'Input img has incorrect dtype or rank' in refusal.text, refusal.text
            bad = fixture.infer({**body, 'img': b64(IMG)}, output_views={'pix_out': {'slice': [[0, 1]]}})
            assert bad.status_code == 400 and 'output_views pix_out needs one' in bad.text, bad.text
            bad = fixture.infer({**body, 'pix': u8(b64, chw, {**NORM, 'std': [1, 0, 1]})})
            assert bad.status_code == 400 and 'Tensor normalize std must not contain 0' in bad.json()['error']['message'], bad.text
        finally:
            fixture.close()
            for child in context.children:
                child.close()
    assert len(context.processes) == 1 and not context.processes[0].is_alive()
    infers = [message for message in context.requests if message[0] == 'infer']
    assert len(infers) == 4 and infers[0][3]['output_views'] == views, infers[0][3]
    assert infers[0][3]['inputs']['pix']['cast'] == 'float32' and infers[0][3]['inputs']['pix']['normalize'] == NORM
    assert infers[1][3]['inputs']['img']['cast'] == 'float32' and 'normalize' not in infers[1][3]['inputs']['img']
    assert len(seen) == 4 and all(name != threading.main_thread().name for name, _ in seen)
    assert seen[0][1]['output_views'] == views and seen[0][1]['inputs']['pix']['normalize'] == NORM


def check_mcp_forwarding():
    """MCP infer_tensor takes a TensorRequest and posts payload.model_dump(mode='json', exclude_unset=True) to /api/infer
    (mcp_server.py infer): cast, normalize and output_views survive that dump, and an absent output_views stays absent."""
    entry = u8(b64, every_pixel(), {**NORM, 'axis': -1})
    views = {'pix_out': {'slice': [[0, 1], [None, None], [None, None], [None, None]]}}
    dumped = server.TensorRequest(model='tens', inputs={'pix': entry}, output_views=views).model_dump(mode='json', exclude_unset=True)
    assert dumped == {'model': 'tens', 'inputs': {'pix': entry}, 'output_views': views}, dumped
    assert 'output_views' not in server.TensorRequest(model='tens', inputs={'pix': entry}).model_dump(mode='json', exclude_unset=True)


def main():
    with tempfile.TemporaryDirectory(prefix='titanos-tensor-convert-', ignore_cleanup_errors=True) as temp:
        check_codec()
        check_mcp_forwarding()
        (Path(temp) / 'local').mkdir()
        fixture = Fixture(Path(temp) / 'local')
        try:
            check_route(fixture)
        finally:
            fixture.close()
        (Path(temp) / 'replica').mkdir()
        check_replica(Path(temp) / 'replica')
    print('PASS: uint8 cast/normalize bit-identical to the client expand()/rebuild() on all 768 (channel, value) pairs and '
          'all 256 casts (float32 maths differs); unconverted requests byte-equal to the code before it; converted requests '
          'byte-equal to float32 ones; output_views equal numpy slices; malformed cast, normalize and views refused by name; '
          'both reach the replica worker')


if __name__ == '__main__':
    main()
