"""Prepared graph execution through vendor runtimes; no implicit CPU substitution."""

import math
import json
import platform
import re
import threading
from pathlib import Path
from .settings import current_settings

import numpy as np

PROVIDERS = {"cpu": "CPUExecutionProvider", "qnn": "QNNExecutionProvider", "dml": "DmlExecutionProvider",
             "coreml": "CoreMLExecutionProvider", "vitisai": "VitisAIExecutionProvider"}
TYPES = {"float": "float32", "double": "float64", "float16": "float16",
         "int64": "int64", "int32": "int32", "int16": "int16", "int8": "int8",
         "uint64": "uint64", "uint32": "uint32", "uint16": "uint16", "uint8": "uint8", "bool": "bool"}
_qnn_lock = threading.Lock()
_qnn_registered = False


def qnn_devices(ort):
    global _qnn_registered
    import onnxruntime_qnn as qnn
    with _qnn_lock:
        if not _qnn_registered:
            ort.register_execution_provider_library(PROVIDERS['qnn'], qnn.get_library_path())
            _qnn_registered = True
    devices = [d for d in ort.get_ep_devices() if d.ep_name == PROVIDERS['qnn']]
    if not devices:
        raise ValueError("QNN plugin found no usable devices; check Qualcomm hardware and drivers")
    return devices, qnn.get_qnn_htp_path()


def onnx_session(path, device, allow_cpu_fallback=False, options=None, profile_prefix=None, expected_adapter=None):
    import onnxruntime as ort
    if device not in PROVIDERS:
        raise ValueError(f"ONNX device must be one of {list(PROVIDERS)}")
    provider = PROVIDERS[device]
    opts = dict(options or {})
    so = ort.SessionOptions()
    so.intra_op_num_threads = current_settings().threads
    if device == 'dml':
        if platform.system() != 'Windows':
            raise ValueError('DirectML requires Windows')
        # DirectML requires sequential execution and disables ORT memory patterns.
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        index = opts.get('device_id', '0')
        if isinstance(index, bool) or not str(index).isdigit():
            raise ValueError('DirectML device_id must be a nonnegative adapter index')
        opts['device_id'] = str(int(index))
    if profile_prefix:
        so.enable_profiling = True
        so.profile_file_prefix = str(profile_prefix)
    if not allow_cpu_fallback and device != "cpu":
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if device == "qnn":
        devices, htp_path = qnn_devices(ort)
        if set(opts) & {"backend_path", "backend_type"}:
            raise ValueError("Titanos qnn selects the HTP NPU backend; do not override backend_path/type")
        opts['backend_path'] = htp_path
        so.add_provider_for_devices(devices, opts)
        session = ort.InferenceSession(str(path), sess_options=so)
    else:
        if provider not in ort.get_available_providers():
            raise ValueError(f"{provider} is not installed; no CPU fallback was substituted")
        if device == 'dml':
            from .dxgi import adapters
            selected = next((d for d in adapters() if d['device_id'] == int(opts['device_id'])), None)
            if selected is None or selected['software']:
                raise ValueError('DirectML requires a present hardware adapter; use device=cpu for software inference')
            if expected_adapter is not None and selected.get('luid') != expected_adapter.get('luid'):
                raise ValueError('DirectML adapter identity changed after selection; no index substitution was used')
        if device == "coreml":
            if platform.system() != "Darwin":
                raise ValueError("Core ML requires an Apple operating system")
            opts = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndNeuralEngine",
                    "RequireStaticInputShapes": "1", **opts}
        providers = [(provider, opts)]
        if allow_cpu_fallback and device != 'cpu':
            providers.append('CPUExecutionProvider')
        session = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    session.disable_fallback()  # Disable Python's retry on another EP after a runtime failure.
    if provider not in session.get_providers():
        raise ValueError(f"Requested provider {provider} was not activated")
    return session


def openvino_address(requested, available):
    """Only exact devices and their documented singleton .0 aliases are valid."""
    requested = requested.upper()
    if requested in available and re.fullmatch(r'CPU|(?:GPU|NPU)(?:\.\d+){0,2}', requested):
        return requested
    for base in ('GPU', 'NPU'):
        if requested == base + '.0' and base in available:
            return base
        if requested == base and base + '.0' in available:
            return base + '.0'
    raise ValueError(f'OpenVINO device {requested!r} is unavailable; no CPU fallback was substituted')


def compiled_placement(backend, requested, *, task=None, options=None, core=None):
    """Use the same auto policy as API advice, with fresh runtime observations."""
    from .devices import installed_runtime_preference, preferred_runtime_device, runtime_records, visibility_constraints
    family = 'openvino' if backend.startswith('openvino') else 'onnx'
    opts = dict(options or {})
    automatic = requested.lower() == 'auto'
    cpu = 'CPU' if family == 'openvino' else 'cpu'
    selected = None
    if requested.lower() == 'cpu':
        device = cpu
        reason = 'explicit_cpu'
    else:
        if family == 'openvino' and core is None:
            import openvino as ov
            core = ov.Core()
        if automatic or requested.startswith(('OV-', 'DML-')) or requested.startswith('dml:'):
            records = runtime_records(family, core=core) if family == 'openvino' else runtime_records(family)
            if automatic:
                selected = preferred_runtime_device(records, backend=backend, task=task,
                    installed=installed_runtime_preference(), options=opts)
                if selected is None:
                    raise ValueError('No supported visible device/provider is available for this compiled task')
                device = selected['device']
                reason = selected['selection_reason']
            else:
                matches = [r for r in records if requested.lower() in {
                    str(r.get('id', '')).lower(), str(r.get('uuid', '')).lower(), str(r.get('runtime_address', '')).lower()}
                    and r.get('available', True) and not r.get('software')]
                if len(matches) != 1:
                    raise ValueError('Pinned compiled device is unavailable or ambiguous; no identity substitution was used')
                selected = matches[0]
                device = 'dml' if family == 'onnx' else selected['runtime_address']
                reason = 'explicit_runtime_identity'
        else:
            device = requested.upper() if family == 'openvino' else requested
            reason = 'explicit_runtime_address'
        if family == 'openvino':
            device = openvino_address(device, core.available_devices)
            if backend == 'openvino-genai' and task == 'image' and device.startswith('NPU'):
                raise ValueError('NPU image execution requires a static reshape/compile path that this SD/SDXL adapter does not implement; select an available GPU or explicit CPU')
        elif device not in PROVIDERS:
            raise ValueError(f'ONNX device must be one of {list(PROVIDERS)} or an observed DirectML identity')
    if device == 'dml' and selected is not None:
        selected_options = selected.get('provider_options', {})
        if 'device_id' in opts and str(opts['device_id']) != str(selected_options.get('device_id')):
            raise ValueError('DirectML provider device_id conflicts with the selected adapter identity')
        opts = {**selected_options, **opts}
    metadata = {'configured_device': requested, 'device': device, 'selection_reason': reason,
                'visibility': visibility_constraints(), 'kernel_execution_verified': False}
    if selected is not None:
        metadata.update({k: selected[k] for k in ('uuid', 'id', 'runtime_address', 'identity_kind',
            'identity_strength', 'provider', 'provider_options') if k in selected})
    return {'device': device, 'options': opts, 'selection': metadata,
            'expected_adapter': selected if device == 'dml' and selected and selected.get('luid') else None}


def compiled_device(backend, requested, *, task=None, core=None, selection=None):
    value = compiled_placement(backend, requested, task=task, core=core)
    if selection is not None:
        selection.update(value['selection'])
    return value['device']


def graph_path(spec):
    root = Path(spec.path).resolve()
    filename = spec.graph or ('openvino_model.xml' if spec.backend == 'openvino' else 'model.onnx')
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Prepared graph must exist inside the model directory: {filename}")
    return path


def tensor_payload(value):
    """Flat typed data keeps dimensions, integer precision and validation explicit."""
    if set(value) != {'dtype', 'shape', 'data'}:
        raise ValueError("Each tensor requires exactly dtype, shape and flat data, or dtype, shape and b64")
    if value['dtype'] not in set(TYPES.values()):
        raise ValueError("Unsupported tensor dtype")
    shape, data = value['shape'], value['data']
    if not isinstance(shape, list) or len(shape) > 8 or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError("Tensor shape requires at most eight nonnegative integer dimensions")
    size = math.prod(shape)
    limit = current_settings().max_tensor_elements
    if size > limit or not isinstance(data, list) or len(data) != size:
        raise ValueError(f"Tensor data must be flat, match shape and contain at most {limit} elements")
    dtype = np.dtype(value['dtype'])
    if dtype.kind in 'iu':
        bounds = np.iinfo(dtype)
        if any(type(v) is not int or not bounds.min <= v <= bounds.max for v in data):
            raise ValueError("Integer tensor data is fractional, boolean or outside its dtype range")
    elif dtype.kind == 'b':
        if any(type(v) is not bool for v in data):
            raise ValueError("Boolean tensors accept only true/false")
    elif any(type(v) not in (int, float) for v in data):
        raise ValueError("Floating tensors accept only numbers")
    with np.errstate(over='ignore', invalid='ignore'):
        try:
            result = np.asarray(data, dtype=dtype).reshape(shape)
        except OverflowError as exc:
            raise ValueError('Tensor values exceed the requested dtype') from exc
    if not np.isfinite(result).all():
        raise ValueError("Tensor values must be finite and representable")
    return result


class CompiledModel:
    def __init__(self, spec):
        self.spec = spec
        path = graph_path(spec)
        core = None
        if spec.backend == 'openvino':
            import openvino as ov
            core = ov.Core()
        placement = compiled_placement(spec.backend, spec.device, task=spec.task,
                                       options=spec.load_options, core=core)
        self.device, self.selection = placement['device'], placement['selection']
        if spec.backend == 'onnx':
            self.model = onnx_session(path, self.device, spec.allow_cpu_fallback, placement['options'],
                                      expected_adapter=placement['expected_adapter'])
            self.inputs = {p.name: {'shape': p.shape, 'dtype': TYPES.get(p.type.removeprefix('tensor(').removesuffix(')'))} for p in self.model.get_inputs()}
            self.output_names = [p.name for p in self.model.get_outputs()]
            self.execution = {'providers': self.model.get_providers(),
                              'cpu_ep_fallback_allowed': spec.allow_cpu_fallback,
                              'placement': 'provider selection; internal hardware placement is not proven'}
            if self.device == 'coreml':
                self.execution['placement'] = 'Core ML may use CPU internally; Neural Engine-only execution is not guaranteed'
        else:
            # Explicit devices avoid AUTO/HETERO silently relaxing the hardware request.
            self.device = openvino_address(self.device, core.available_devices)
            model = core.read_model(path)
            options = dict(spec.load_options)
            if self.device == 'CPU':
                options.setdefault('INFERENCE_PRECISION_HINT', 'f32')
                options.setdefault('INFERENCE_NUM_THREADS', current_settings().threads)
            self.model = core.compile_model(model, self.device, options)
            self.inputs = {p.any_name: {'shape': [d.get_length() if d.is_static else None for d in p.partial_shape],
                                      'dtype': np.dtype(p.element_type.to_dtype()).name} for p in self.model.inputs}
            self.output_names = [p.any_name for p in self.model.outputs]
            executed = self.model.get_property('EXECUTION_DEVICES')
            self.execution = {'execution_devices': [executed] if isinstance(executed,str) else list(executed),
                              'cpu_ep_fallback_allowed': False}
            canonical = lambda d: d + '.0' if d in {'GPU', 'NPU'} else d
            if {canonical(str(d).upper()) for d in self.execution['execution_devices']} != {canonical(self.device)}:
                raise ValueError('OpenVINO execution devices differ from the selected device; CPU or other-device substitution was not accepted')
        self.execution['selection'] = self.selection
        if any(v['dtype'] is None for v in self.inputs.values()):
            raise ValueError("Graph uses an unsupported input type")
        if spec.task == 'embeddings':
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(spec.path, local_files_only=True, trust_remote_code=False)
            if set(self.inputs) - {'input_ids', 'attention_mask', 'token_type_ids'}:
                raise ValueError('Embedding graph has inputs unsupported by this tokenizer adapter')
            if 'input_ids' not in self.inputs or 'attention_mask' not in self.inputs:
                raise ValueError("Embedding graphs require input_ids and attention_mask")
            if any(len(p['shape']) != 2 for p in self.inputs.values()):
                raise ValueError("Embedding graph inputs must have batch and sequence dimensions")
            if any(isinstance(p['shape'][0], int) and p['shape'][0] != 1 for p in self.inputs.values()):
                raise ValueError("Embedding graphs need dynamic batch or fixed batch=1")
            if 'last_hidden_state' not in self.output_names and 'sentence_embedding' not in self.output_names:
                raise ValueError("Embedding graph must output last_hidden_state or sentence_embedding")

    def details(self):
        return {'backend': self.spec.backend, 'device': self.device, 'inputs': self.inputs,
                'configured_device': self.spec.device, 'selection': self.selection,
                'outputs': self.output_names, **self.execution}

    def run(self, inputs):
        if set(inputs) != set(self.inputs):
            raise ValueError(f"Graph requires inputs {list(self.inputs)}")
        for name, value in inputs.items():
            expected = self.inputs[name]
            if value.dtype.name != expected['dtype'] or len(value.shape) != len(expected['shape']):
                raise ValueError(f"Input {name} has incorrect dtype or rank")
            if any(isinstance(d, int) and d >= 0 and d != n for d, n in zip(expected['shape'], value.shape)):
                raise ValueError(f"Input {name} has incorrect shape; expected {expected['shape']}")
        if self.spec.backend == 'onnx':
            values = self.model.run(None, inputs)
        else:
            results = self.model(inputs)
            values = [results[p] for p in self.model.outputs]
        if any(not np.isfinite(v).all() for v in values):
            raise ValueError("Model produced non-finite outputs")
        return dict(zip(self.output_names, values))

    def infer(self, task, data, emit=None, cancel=None):
        if task == 'embeddings':
            from .text_inputs import input_limit, truncation_policy, token_budget, prepare_inputs, check_cancel
            texts = data['input'] if isinstance(data['input'], list) else [data['input']]
            vectors, tokens = [], 0
            length = self.inputs['input_ids']['shape'][1]
            fixed = length if isinstance(length, int) and length > 0 else None
            maximum = token_budget(data.get('max_tokens'), self.spec.defaults, 'max_tokens', current_settings().embedding_max_tokens)
            limit = input_limit(self.processor, maximum, fixed_length=fixed)
            truncate = truncation_policy(data.get('truncate'), self.spec.defaults, current_settings().embedding_truncate)
            batch, details, policy = prepare_inputs(self.processor, texts, limit, truncate, tensor_type='np', fixed_length=fixed, cancel=cancel)
            for index in range(len(texts)):
                check_cancel(cancel)
                # ponytail: batch=1 also serves static NPU graphs; batch only after benchmarking.
                inputs = {name: value[index:index + 1] for name, value in batch.items()}
                feed = {name: np.asarray(inputs[name], dtype=p['dtype']) for name, p in self.inputs.items()}
                out = self.run(feed)
                check_cancel(cancel)
                mask = np.asarray(inputs['attention_mask'])
                if 'sentence_embedding' in out:
                    vector = out['sentence_embedding'].astype(np.float32)
                else:
                    hidden = out['last_hidden_state'].astype(np.float32)
                    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape:
                        raise ValueError("Invalid last_hidden_state dimensions")
                    vector = (hidden * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
                if vector.ndim != 2 or vector.shape[0] != 1:
                    raise ValueError("Invalid sentence_embedding dimensions")
                vector = vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
                vectors.extend(vector.tolist())
                tokens += int(mask.sum())
            check_cancel(cancel)
            return {'vectors': vectors, 'usage': {'prompt_tokens': tokens, 'total_tokens': tokens},
                    'input_details': details, 'truncation': policy}
        views = output_views(data['output_views'], self.output_names) if data.get('output_views') else {}
        values = self.run({k: tensor_input(v) for k, v in data['inputs'].items()})
        limit = current_settings().max_tensor_elements
        if sum(v.size for v in values.values()) > limit:
            raise ValueError(f"Combined tensor output exceeds {limit} elements")
        return {'outputs': {k: tensor_result(output_view(v, k, views[k]) if k in views else v, data.get('output_encoding') or 'list')
                            for k, v in values.items()},
                'execution': self.execution}


def runtime_check(backend, device='auto', check=False, options=None):
    import base64
    import json
    import tempfile
    from importlib.metadata import version
    from .config import ModelSpec
    # ONNX opset 17: y = ReLU(x @ identity(4)) @ ones(4,1), fixed input [1,4].
    graph = base64.b64decode('CAg63wEKEQoBeAoBdxIBYSIGTWF0TXVsCgwKAWESAWIiBFJlbHUKEQoBYgoBdhIBeSIGTWF0TXVsEhV0aXRhbm9zLXJ1bnRpbWUtY2hlY2sqSwgECAQQASJAAACAPwAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAACAP0IBdyobCAQIARABIhAAAIA/AACAPwAAgD8AAIA/QgF2WhMKAXgSDgoMCAESCAoCCAEKAggEYhMKAXkSDgoMCAESCAoCCAEKAggBQgQKABAR')
    info = {'backend': backend, 'scope': 'two-layer graph only; full model and modality qualification is separate'}
    try:
        placement = compiled_placement(backend, device, options=options)
        device, options = placement['device'], placement['options']
        info['selection'] = placement['selection']
        if backend.startswith('openvino'):
            import openvino as ov
            info.update(version=version('openvino'), devices=ov.Core().available_devices)
            backend = 'openvino'
        else:
            import onnxruntime as ort
            info.update(version=ort.__version__, providers=ort.get_available_providers())
            if device == 'dml':
                from .dxgi import adapters
                info['adapters'] = adapters()
                info['provider_options'] = dict(options or {})
            if device == 'qnn':
                devices, _ = qnn_devices(ort)
                info['qnn_devices'] = len(devices)
        info['device'] = device
        if check:
            with tempfile.TemporaryDirectory(prefix='titanos-check-') as tmp:
                path = Path(tmp)
                (path / 'model.onnx').write_bytes(graph)
                model_device = placement['expected_adapter']['id'] if placement['expected_adapter'] else device
                spec = ModelSpec(path=tmp, task='tensor', backend=backend, graph='model.onnx', device=model_device, load_options=options or {})
                model = CompiledModel(spec)
                expected = np.array([[6]], dtype=np.float32)
                actual = model.run({'x': np.array([[-1, 1, 2, 3]], dtype=np.float32)})['y']
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                info['check'] = {'passed': True, 'execution': model.details(), 'expected': expected.tolist(), 'actual': actual.tolist()}
                if backend == 'onnx':
                    # A short, separate profiled execution records ORT's observed node providers.
                    session = onnx_session(path / 'model.onnx', device, options=options, profile_prefix=path / 'profile',
                                           expected_adapter=placement['expected_adapter'])
                    session.run(None, {'x': np.array([[-1, 1, 2, 3]], dtype=np.float32)})
                    events = json.loads(Path(session.end_profiling()).read_text())
                    info['check']['observed_node_providers'] = sorted({e['args']['provider'] for e in events if e.get('args', {}).get('provider')})
    except Exception as exc:
        info['check'] = {'passed': False, 'error': str(exc)}
    return info


def tensor_input(value):
    """An input as flat list data (tensor_payload) or as base64 of its little-endian bytes (tensor_bytes); an optional
    "cast" (with an optional "normalize") converts the decoded uint8 source to float32 (tensor_cast)."""
    if isinstance(value, dict) and ('cast' in value or 'normalize' in value):
        if 'cast' not in value:
            raise ValueError('Tensor normalize requires cast "float32"')
        if value['cast'] != 'float32' or value.get('dtype') != 'uint8':
            raise ValueError('Tensor cast converts a uint8 input to "float32" only')
        source = tensor_input({k: v for k, v in value.items() if k not in ('cast', 'normalize')})
        return tensor_cast(source, value['normalize']) if 'normalize' in value else source.astype(np.float32)
    if isinstance(value, dict) and set(value) == {'dtype', 'shape', 'b64'}:
        return tensor_bytes(value)
    return tensor_payload(value)


def tensor_bytes(value):
    """Base64 of the C-order little-endian bytes: exact length, the same dtype, shape, element and finite checks as lists."""
    import base64
    if value['dtype'] not in set(TYPES.values()):
        raise ValueError("Unsupported tensor dtype")
    shape, text = value['shape'], value['b64']
    if not isinstance(shape, list) or len(shape) > 8 or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError("Tensor shape requires at most eight nonnegative integer dimensions")
    size = math.prod(shape)
    limit = current_settings().max_tensor_elements
    if size > limit:
        raise ValueError(f"Tensor data must be flat, match shape and contain at most {limit} elements")
    dtype = np.dtype(value['dtype'])
    expected = size * dtype.itemsize
    if not isinstance(text, str) or len(text) != 4 * -(-expected // 3):
        raise ValueError(f"Tensor b64 must encode exactly {expected} bytes for its dtype and shape")
    try:
        raw = base64.b64decode(text, validate=True)
    except ValueError as exc:
        raise ValueError("Tensor b64 is not valid base64") from exc
    if len(raw) != expected:
        raise ValueError(f"Tensor b64 must encode exactly {expected} bytes for its dtype and shape")
    if dtype.kind == 'b' and raw and np.frombuffer(raw, dtype=np.uint8).max() > 1:
        raise ValueError("Boolean tensors accept only true/false")
    result = np.frombuffer(raw, dtype=dtype.newbyteorder('<')).astype(dtype).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError("Tensor values must be finite and representable")
    return result


def tensor_result(value, encoding):
    """An output as flat list data, or for output_encoding "base64" as base64 of its C-order little-endian bytes."""
    if encoding == 'list':
        return {'dtype': value.dtype.name, 'shape': list(value.shape), 'data': value.reshape(-1).tolist()}
    if encoding != 'base64':
        raise ValueError("output_encoding must be list or base64")
    import base64
    if value.dtype.name not in set(TYPES.values()):
        raise ValueError(f"Output dtype {value.dtype.name} has no base64 encoding")
    raw = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder('<')).tobytes()
    return {'dtype': value.dtype.name, 'shape': list(value.shape), 'b64': base64.b64encode(raw).decode('ascii')}


def tensor_cast(value, normalize):
    """A uint8 input's "normalize" {mean, std, axis}: ((u8 - mean) / std) in float64 along axis with one final float32 cast,
    as a client computing the same in numpy (float64 mean/std arrays, one float32 cast), so the floats are bit-identical
    to its own; a result outside float32's finite range is refused like any other non-finite input."""
    if not isinstance(normalize, dict) or set(normalize) != {'mean', 'std', 'axis'}:
        raise ValueError('Tensor normalize takes exactly mean, std and axis')
    axis = normalize['axis']
    if type(axis) is not int or not -value.ndim <= axis < value.ndim:
        raise ValueError(f'Tensor normalize axis must be an integer within the {value.ndim} input axes')
    axis %= value.ndim
    count, shape, stats = value.shape[axis], [1] * value.ndim, []
    shape[axis] = -1
    for key in ('mean', 'std'):
        values = normalize[key]
        if not isinstance(values, list) or len(values) != count or any(type(v) not in (int, float) for v in values):
            raise ValueError(f'Tensor normalize {key} must list {count} numbers, one per index of axis {axis}')
        try:
            stats.append(np.asarray(values, dtype=np.float64).reshape(shape))
        except OverflowError as exc:
            raise ValueError(f'Tensor normalize {key} values must be finite') from exc
        if not np.isfinite(stats[-1]).all():
            raise ValueError(f'Tensor normalize {key} values must be finite')
    if not stats[1].all():
        raise ValueError('Tensor normalize std must not contain 0')
    with np.errstate(over='ignore', invalid='ignore'):
        result = ((value.astype(np.float64) - stats[0]) / stats[1]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Tensor normalize result must be finite in float32')
    return result


def output_views(views, names):
    """output_views: {output name: {"slice": [[start, stop], ...]}}, one integer-or-null pair per axis and no step; as slices."""
    if not views:
        return {}
    if set(views) - set(names):
        raise ValueError(f'output_views names {sorted(set(views) - set(names))}; the graph outputs are {names}')
    for name, view in views.items():
        if not isinstance(view, dict) or set(view) != {'slice'} or not isinstance(view['slice'], list) or any(
                not isinstance(pair, list) or len(pair) != 2 or any(b is not None and type(b) is not int for b in pair)
                for pair in view['slice']):
            raise ValueError(f'output_views {name} takes exactly {{"slice": [[start, stop], ...]}} with integer or null bounds')
    return {name: tuple(slice(*pair) for pair in view['slice']) for name, view in views.items()}


def output_view(value, name, view):
    """One output sliced by its view (output_views), C-contiguous for encoding."""
    if len(view) != value.ndim:
        raise ValueError(f'output_views {name} needs one [start, stop] pair per axis; the output has {value.ndim} axes')
    return np.array(value[view], order='C')
