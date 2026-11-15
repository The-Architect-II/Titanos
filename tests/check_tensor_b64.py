"""Tensor base64 transport: /api/infer takes {dtype, shape, b64} inputs beside flat lists and answers output_encoding
base64; a real ONNX Runtime CPU graph built here, no registry and no hardware claim.

Base64 carries the C-order little-endian bytes, so a float32 detector frame costs 4/3 of its raw size on the wire
instead of a JSON number per element; the dtype, shape, element-limit and finite checks are the list form's.
"""
import base64
import math
import os
from pathlib import Path
import sys
import tempfile

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
    import onnxruntime  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(f'check_tensor_b64 needs {exc.name}') from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
HOME = tempfile.TemporaryDirectory(prefix='titanos-tensor-b64-home-', ignore_cleanup_errors=True)
os.environ['TITANOS_HOME'] = HOME.name  # nothing from this check reaches ~/.titanos

from fastapi.testclient import TestClient  # noqa: E402
from titanos import server  # noqa: E402
from titanos.compiled import TYPES, CompiledModel, tensor_bytes, tensor_input, tensor_result  # noqa: E402
from titanos.config import Config, ModelSpec  # noqa: E402
from titanos.settings import Settings, current_settings  # noqa: E402

EXTRA_KEYS = 'Each tensor requires exactly dtype, shape and flat data, or dtype, shape and b64'


def refused(call, text):
    try:
        call()
    except ValueError as exc:
        assert text in str(exc), f'{text!r} not in {exc}'
        return exc
    raise AssertionError(f'accepted; expected ValueError {text!r}')


def b64(array):
    array = np.asarray(array)
    raw = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder('<')).tobytes()
    return {'dtype': array.dtype.name, 'shape': list(array.shape), 'b64': base64.b64encode(raw).decode('ascii')}


def listed(array):
    array = np.asarray(array)
    return {'dtype': array.dtype.name, 'shape': list(array.shape), 'data': array.reshape(-1).tolist()}


def decoded(value):
    assert set(value) == {'dtype', 'shape', 'b64'}, value
    raw = base64.b64decode(value['b64'], validate=True)
    return np.frombuffer(raw, dtype=np.dtype(value['dtype']).newbyteorder('<')).reshape(value['shape'])


def write_graph(path):
    """x + 1 (float32), ids passed through (int64), not flag (bool), img passed through (uint8)."""
    one = helper.make_tensor('one', TensorProto.FLOAT, [], [1.0])
    nodes = [helper.make_node('Add', ['x', 'one'], ['x_plus_one']), helper.make_node('Identity', ['ids'], ['ids_out']),
             helper.make_node('Not', ['flag'], ['flag_not']), helper.make_node('Identity', ['img'], ['img_out'])]
    value = helper.make_tensor_value_info
    inputs = [value('x', TensorProto.FLOAT, [2, 3]), value('ids', TensorProto.INT64, [4]),
              value('flag', TensorProto.BOOL, [3]), value('img', TensorProto.UINT8, [2, 2])]
    outputs = [value('x_plus_one', TensorProto.FLOAT, [2, 3]), value('ids_out', TensorProto.INT64, [4]),
               value('flag_not', TensorProto.BOOL, [3]), value('img_out', TensorProto.UINT8, [2, 2])]
    model = helper.make_model(helper.make_graph(nodes, 'titanos-tensor-b64-check', inputs, outputs, [one]),
                              opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def check_codec():
    samples = {'float32': [-1.5, 0.0, 3.25, 1e-30], 'float64': [-2.5, 0.0, 1e300, -1e-300],
               'float16': [-0.5, 0.0, 65504.0, 0.0009765625]}
    for name in sorted(set(TYPES.values())):
        dtype = np.dtype(name)
        if dtype.kind == 'b':
            array = np.array([[True, False], [False, True]])
        elif dtype.kind in 'iu':
            bounds = np.iinfo(dtype)
            array = np.array([[bounds.min, 0], [1, bounds.max]], dtype=dtype)
        else:
            array = np.array(samples[name], dtype=dtype).reshape(2, 2)
        encoded = tensor_result(array, 'base64')
        assert encoded['dtype'] == name and encoded['shape'] == [2, 2], encoded
        assert base64.b64decode(encoded['b64']) == array.astype(dtype.newbyteorder('<')).tobytes(), name
        back = tensor_input(encoded)
        assert back.dtype == dtype and back.shape == (2, 2) and np.array_equal(back, array), name
        assert tensor_result(array, 'list') == listed(array), name

    big = np.array([1, 258], dtype='>i4')  # big-endian in memory, little-endian on the wire
    assert base64.b64decode(tensor_result(big, 'base64')['b64']) == b'\x01\x00\x00\x00\x02\x01\x00\x00'
    turned = np.arange(6, dtype=np.int16).reshape(2, 3).T  # not C-contiguous: C-order bytes of the view
    encoded = tensor_result(turned, 'base64')
    assert encoded['shape'] == [3, 2] and np.array_equal(tensor_bytes(encoded), turned)
    scalar = tensor_result(np.array(3.5, dtype=np.float32), 'base64')
    assert scalar['shape'] == [] and tensor_bytes(scalar).shape == () and float(tensor_bytes(scalar)) == 3.5
    empty = tensor_bytes({'dtype': 'float32', 'shape': [0, 3], 'b64': ''})
    assert empty.shape == (0, 3) and empty.dtype == np.float32
    refused(lambda: tensor_result(np.zeros(2, np.complex64), 'base64'), 'Output dtype complex64 has no base64 encoding')
    refused(lambda: tensor_result(np.zeros(2, np.float32), 'hex'), 'output_encoding must be list or base64')

    def bad(dtype, shape, text):
        return lambda: tensor_input({'dtype': dtype, 'shape': shape, 'b64': text})
    refused(bad('uint8', [3], 'AAAAAA=='), 'must encode exactly 3 bytes for its dtype and shape')
    refused(bad('uint8', [2], 'AA=='), 'must encode exactly 2 bytes')  # four characters, one byte
    refused(bad('uint8', [3], 123), 'must encode exactly 3 bytes')
    refused(bad('uint8', [3], 'AAA*'), 'Tensor b64 is not valid base64')
    refused(bad('uint8', [3], '-___'), 'Tensor b64 is not valid base64')  # the URL-safe alphabet is refused
    refused(bad('bool', [1], base64.b64encode(b'\x02').decode()), 'Boolean tensors accept only true/false')
    for value in (math.nan, math.inf):
        text = base64.b64encode(np.array([value], '<f4').tobytes()).decode()
        refused(bad('float32', [1], text), 'Tensor values must be finite and representable')
    refused(bad('complex64', [1], 'AAAAAAAAAAA='), 'Unsupported tensor dtype')
    for shape in ([-1], [True], [1] * 9, 'x'):
        refused(bad('uint8', shape, ''), 'Tensor shape requires at most eight nonnegative integer dimensions')
    limit = current_settings().max_tensor_elements
    refused(bad('uint8', [limit + 1], ''), f'contain at most {limit} elements')
    refused(lambda: tensor_input({'dtype': 'uint8', 'shape': [1], 'b64': 'AA==', 'data': [0]}), EXTRA_KEYS)
    refused(lambda: tensor_input({'dtype': 'uint8', 'b64': 'AA=='}), EXTRA_KEYS)


class Fixture:
    def __init__(self, scratch):
        write_graph(scratch / 'model.onnx')
        self.config = Config(models={'tens': ModelSpec(path=str(scratch), task='tensor', backend='onnx', device='cpu')})
        self.app = server.create_app(self.config, output_dir=scratch / 'out', settings=Settings(),
                                     runtime_catalog=False)
        engine = self.app.state.engine.target
        engine.runtime = CompiledModel(self.config.models['tens'])
        engine.current, engine.states['tens'] = 'tens', 'loaded'
        self.client = TestClient(self.app, base_url='http://127.0.0.1:8000')
        self.client.__enter__()

    def infer(self, inputs, **fields):
        return self.client.post('/api/infer', json={'model': 'tens', 'inputs': inputs, **fields})

    def close(self):
        self.client.__exit__(None, None, None)


def check_route(fixture):
    x = (np.arange(6, dtype=np.float32).reshape(2, 3) * 0.5 - 1).astype(np.float32)
    ids = np.array([-2, 0, 2 ** 40, 7], dtype=np.int64)
    flag = np.array([True, False, True])
    img = np.array([[0, 255], [17, 128]], dtype=np.uint8)
    arrays = {'x': x, 'ids': ids, 'flag': flag, 'img': img}
    expected = {'x_plus_one': x + 1, 'ids_out': ids, 'flag_not': ~flag, 'img_out': img}

    response = fixture.infer({name: b64(value) for name, value in arrays.items()}, output_encoding='base64')
    assert response.status_code == 200, response.text
    outputs = response.json()['outputs']
    assert set(outputs) == set(expected)
    for name, value in expected.items():
        assert outputs[name]['dtype'] == value.dtype.name and outputs[name]['shape'] == list(value.shape)
        np.testing.assert_array_equal(decoded(outputs[name]), value)
    assert 'CPUExecutionProvider' in response.json()['execution']['providers']

    listed_response = fixture.infer({name: listed(value) for name, value in arrays.items()})  # the default stays list
    assert listed_response.status_code == 200, listed_response.text
    assert listed_response.json()['outputs'] == {name: listed(value) for name, value in expected.items()}
    mixed = fixture.infer({'x': listed(x), 'ids': b64(ids), 'flag': listed(flag), 'img': b64(img)},
                          output_encoding='list')
    assert mixed.status_code == 200 and mixed.json()['outputs'] == listed_response.json()['outputs'], mixed.text

    def failed(inputs, status, text, **fields):
        response = fixture.infer({**{name: b64(value) for name, value in arrays.items()}, **inputs}, **fields)
        assert response.status_code == status, (response.status_code, response.text)
        error = response.json()['error']
        assert text in error['message'], (text, error)
        return error
    failed({}, 422, 'output_encoding', output_encoding='hex')
    failed({'x': {**b64(x), 'b64': '*' * 32}}, 400, 'Tensor b64 is not valid base64')
    failed({'x': {**b64(x), 'b64': 'AAAA'}}, 400, 'Tensor b64 must encode exactly 24 bytes for its dtype and shape')
    failed({'x': {**b64(x), 'data': x.reshape(-1).tolist()}}, 400, EXTRA_KEYS)
    failed({'x': b64(x.astype(np.float64))}, 400, 'Input x has incorrect dtype or rank')
    failed({'flag': {**b64(flag), 'b64': base64.b64encode(b'\x01\x00\x02').decode()}}, 400,
           'Boolean tensors accept only true/false')


def main():
    with tempfile.TemporaryDirectory(prefix='titanos-tensor-b64-', ignore_cleanup_errors=True) as temp:
        check_codec()
        fixture = Fixture(Path(temp))
        try:
            check_route(fixture)
        finally:
            fixture.close()
    print('PASS: tensor b64 inputs and base64 outputs round-trip every supported dtype little-endian through a real '
          'ONNX Runtime CPU graph; list form unchanged; malformed b64 refused by name')


if __name__ == '__main__':
    main()
