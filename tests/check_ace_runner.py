"""Portable ACE runner checks; fake framework observations and synthetic PCM only.

Run with Python 3.11+ and NumPy. No ACE/Torch installation, model files, devices,
network access or private source checkout are needed.
"""
import ast
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/music_runtime/ace_adapter.py'
spec = importlib.util.spec_from_file_location('ace_runner_check_subject', RUNNER)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
UUID = 'GPU-12345678-1234-1234-1234-123456789abc'


GIB = 1024 ** 3
# Synthetic per-component parameter counts used by placement checks; no weight file of this size exists.
COUNTS = {'planner': 4_000_000_000, 'dit': 5_000_000_000, 'text_encoder': 600_000_000, 'vae': 200_000_000}
TOTAL = sum(COUNTS.values())  # 9.8e9 parameters: 19.6 GB at bfloat16, 39.2 GB at float32.
PEAK = max(COUNTS['planner'], COUNTS['dit']) + COUNTS['text_encoder'] + COUNTS['vae']
PLACEMENT_KEYS = {'requested', 'resolved', 'device_free_bytes', 'required_bytes', 'total_resident_bytes',
                  'peak_set_bytes', 'headroom_bytes', 'accelerate_version', 'basis'}
SAFETENSORS_ITEMSIZE = {'F32': 4, 'BF16': 2}


def safetensors_fixture(path, tensors):
    """Write a structurally valid safetensors file whose data region is zeros: header length, JSON table, data."""
    table, offset = {'__metadata__': {'format': 'pt', 'note': 'inert fixture'}}, 0
    for name, (dtype, shape) in tensors.items():
        size = math.prod(shape) * SAFETENSORS_ITEMSIZE[dtype]
        table[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [offset, offset + size]}
        offset += size
    header = json.dumps(table).encode('utf-8')
    path.write_bytes(len(header).to_bytes(8, 'little') + header + bytes(offset))
    return offset


def raw_safetensors(path, header, data):
    """Write header length + header bytes (any JSON text, valid or not) + data bytes."""
    header = header if isinstance(header, bytes) else json.dumps(header).encode('utf-8')
    path.write_bytes(len(header).to_bytes(8, 'little') + header + data)
    return path


def safetensors_payload(path, tensors, order=None):
    """Write a safetensors file with real tensor bytes: name -> (dtype, shape, bytes). `order` chooses the
    order of the JSON entries (the data layout follows insertion order regardless)."""
    table, data, offset = {}, b'', 0
    for name, (dtype, shape, raw) in tensors.items():
        assert len(raw) == math.prod(shape) * adapter.SAFETENSORS_DTYPES[dtype][1]
        table[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [offset, offset + len(raw)]}
        data += raw
        offset += len(raw)
    ordered = {'__metadata__': {'format': 'pt'}}
    for name in (order or list(table)):
        ordered[name] = table[name]
    return raw_safetensors(path, ordered, data)


class FakeTorch:
    """The runner's observation interface; this object performs no device work."""
    __version__ = 'synthetic-framework-fixture'
    version = SimpleNamespace(hip=None)

    def __init__(self, available=True, uuids=(UUID,), bf16=True, free=80 * GIB, total=96 * GIB):
        self.calls = []
        self.threads = self.interop = 0

        def called(name, value):
            self.calls.append(name)
            return value

        self.cuda = SimpleNamespace(
            is_available=lambda: called('available', available),
            device_count=lambda: called('count', len(uuids)),
            get_device_properties=lambda i: SimpleNamespace(uuid=uuids[i]),
            set_device=lambda i: called('set_device', i),
            is_bf16_supported=lambda: called('bf16', bf16),
            mem_get_info=lambda i: called('mem_get_info', (free, total)),
        )

    def set_num_threads(self, value): self.threads = value
    def get_num_threads(self): return self.threads
    def set_num_interop_threads(self, value): self.interop = value
    def get_num_interop_threads(self): return self.interop


class FakeModule:
    def __init__(self, device='cpu', dtype='float32'):
        self.parameter = SimpleNamespace(device=device, dtype='torch.' + dtype)
        self.hook = None
        self.moves = []

    def to(self, *args, **kwargs):
        # Record the requested move and reflect it on the fixture parameter, as a real module would.
        self.moves.append((args, kwargs))
        device = kwargs.get('device', args[0] if args else None)
        if device is not None:
            self.parameter.device = device
        if kwargs.get('dtype') is not None:
            self.parameter.dtype = 'torch.' + str(kwargs['dtype'])
        return self

    def parameters(self): return iter([self.parameter])

    def register_forward_pre_hook(self, hook):
        self.hook = hook
        return SimpleNamespace(remove=lambda: setattr(self, 'hook', None))

    def execute_fixture(self): self.hook(self, ())


class FakeDType:
    def __init__(self, name, itemsize, floating):
        self.name, self.itemsize, self.floating = name, itemsize, floating

    def __repr__(self): return 'torch.' + self.name


class FakeTensor:
    """Bytes + dtype + shape + device; every .to() returns a new object and records what it did."""
    def __init__(self, data, dtype, shape, device='cpu', casts=(), moves=()):
        self.data, self.dtype, self.shape, self.device = data, dtype, tuple(shape), device
        self.casts, self.moves = tuple(casts), tuple(moves)

    def reshape(self, shape):
        assert math.prod(shape) == math.prod(self.shape)
        return FakeTensor(self.data, self.dtype, shape, self.device, self.casts, self.moves)

    def to(self, device=None, dtype=None):
        casts = self.casts + (((self.device, dtype.name),) if dtype is not None else ())
        moves = self.moves + (((self.device, device),) if device is not None else ())
        return FakeTensor(self.data, dtype or self.dtype, self.shape, device or self.device, casts, moves)

    def is_floating_point(self): return self.dtype.floating

    def numel(self): return math.prod(self.shape)


class FakeTensorTorch(FakeTorch):
    """Adds the tensor-construction surface stream_safetensors/build_resident use; no device work happens."""
    float64, float32, float16, bfloat16 = FakeDType('float64', 8, True), FakeDType('float32', 4, True), FakeDType('float16', 2, True), FakeDType('bfloat16', 2, True)
    int64, int32, int16, int8, uint8, bool = (FakeDType('int64', 8, False), FakeDType('int32', 4, False), FakeDType('int16', 2, False),
                                              FakeDType('int8', 1, False), FakeDType('uint8', 1, False), FakeDType('bool', 1, False))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffers, self.default_dtype = [], self.float32

    def frombuffer(self, buffer, dtype):
        self.buffers.append(type(buffer).__name__)
        assert len(buffer) and len(buffer) % dtype.itemsize == 0
        return FakeTensor(bytes(buffer), dtype, (len(buffer) // dtype.itemsize,))

    def empty(self, shape, dtype):
        assert math.prod(shape) == 0
        return FakeTensor(b'', dtype, shape)

    def set_default_dtype(self, dtype): self.default_dtype = dtype
    def get_default_dtype(self): return self.default_dtype


class FakeResidentModel:
    """Module stand-in: meta parameters until assigned, host buffers until moved, optional tie binding."""
    def __init__(self, keys, tied=(), tie=True, buffers=(), bind=True, move_buffers=True, stubborn=(), rename=None):
        self.config = SimpleNamespace(tie_word_embeddings=tie)
        self._tied_weights_keys = list(tied)
        meta = FakeTensor(b'', FakeTensorTorch.bfloat16, (1,), 'meta')
        self.params = {key: meta for key in keys}
        self.stubborn = {key: FakeTensor(b'\0\0\0\0', FakeTensorTorch.float32, (1,), 'cpu') for key in stubborn}
        self.params.update(self.stubborn)
        self.buffers = {name: FakeTensor(b'\0\0\0\0', FakeTensorTorch.float32, (1,)) for name in buffers}
        self.bind, self.move_buffers, self.evaluated, self.moved_to = bind, move_buffers, False, []
        if rename is not None:
            self._fix_state_dict_key_on_load = rename

    def load_state_dict(self, state, strict, assign):
        assert strict is False and assign is True
        for key, value in state.items():
            if key in self.params and key not in self.stubborn:
                self.params[key] = value
        return [key for key in self.params if key not in state], [key for key in state if key not in self.params]

    def tie_weights(self):
        if self.bind:
            source = next(key for key in self.params if key not in self._tied_weights_keys)
            for key in self._tied_weights_keys:
                self.params[key] = self.params[source]

    def named_parameters(self, remove_duplicate=True): return list(self.params.items())
    def named_buffers(self, remove_duplicate=True): return list(self.buffers.items())

    def to(self, device):
        self.moved_to.append(device)
        if self.move_buffers:
            self.buffers = {name: value.to(device=device) for name, value in self.buffers.items()}
        return self

    def eval(self):
        self.evaluated = True
        return self


class RejectingResidentModel(FakeResidentModel):
    """What torch does for a checkpoint tensor whose shape differs from the parameter, even with strict=False."""
    def load_state_dict(self, state, strict, assign):
        raise RuntimeError('Error(s) in loading state_dict for Fake:\n\tsize mismatch for model.norm.weight: copying a param with shape '
                           'torch.Size([17]) from checkpoint, the shape in current model is torch.Size([1]).')


def fake_accelerate(seen):
    module = ModuleType('accelerate')
    module.__version__ = 'fixture'

    @contextmanager
    def init_empty_weights(include_buffers=None):
        seen.append(include_buffers)
        yield
    module.init_empty_weights = init_empty_weights
    return module


def request(device='cpu', dtype='auto', placement='auto'):
    return {
        'bundle': str(ROOT), 'variant': 'xl-sft', 'code_revision': adapter.CODE_REVISION,
        'model_revision': adapter.MODEL_REVISION, 'offline': True,
        'prompt': 'An original instrumental arrangement', 'mode': 'instrumental', 'duration': 10,
        'runtime_contract': {
            'version': 1, 'requested_device': device, 'device': device,
            'expected_uuid': None if device in {'cpu', 'auto'} else UUID,
            'visibility': {'CUDA_VISIBLE_DEVICES': '' if device == 'cpu' else UUID},
            'dtype': dtype, 'threads': 2, 'interop_threads': 1, 'max_audio_seconds': 600,
            'max_output_bytes': 134217728, 'max_samples': 60000000,
            'max_steps': 100, 'max_prompt_chars': 8000, 'max_text_chars': 100000,
            'placement': placement,
        },
    }


class AdapterChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='titanos-ace-check-')
        self.root = Path(self.temp.name).resolve()

    def tearDown(self): self.temp.cleanup()

    def test_import_does_not_load_model_frameworks(self):
        for name in ('torch', 'diffusers', 'transformers', 'acestep'):
            self.assertNotIn(name, sys.modules)

    def test_request_limits_and_unsupported_features(self):
        adapter.validate_request(request())
        cases = [
            ('offline', False), ('variant', 'xl-turbo'), ('reference_audio', 'reference.wav'),
            ('mode', 'unsupported'), ('lyrics', 'Cannot discard supplied lyrics'),
            ('duration', float('nan')), ('seed', True), ('steps', 0),
        ]
        for key, value in cases:
            candidate = request()
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.validate_request(candidate)
        for key, value in [('device', 'xpu'), ('threads', True), ('interop_threads', True),
                           ('max_samples', 10), ('max_output_bytes', 44)]:
            candidate = request()
            candidate['runtime_contract'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.validate_request(candidate)
        # Planner route flags: booleans only; caption and language need the metadata pass, and so does thinking off.
        for flags in ({'thinking': 1}, {'use_cot_metas': 'yes'}, {'use_cot_caption': True}, {'use_cot_language': True},
                      {'thinking': False}, {'thinking': False, 'use_cot_caption': True}):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                adapter.validate_request({**request(), **flags})
        for flags in ({'thinking': True}, {'thinking': True, 'use_cot_metas': True, 'use_cot_caption': True, 'use_cot_language': True},
                      {'thinking': False, 'use_cot_metas': True}):
            with self.subTest(flags=flags):
                self.assertEqual(adapter.planner_flags(adapter.validate_request({**request(), **flags})),
                                 {**adapter.PLANNER_DEFAULTS, **flags})
        self.assertEqual(adapter.planner_flags(request()),
                         {'thinking': True, 'use_cot_metas': False, 'use_cot_caption': False, 'use_cot_language': False})

    def test_visibility_binding_preserves_driver_context_and_removes_token(self):
        contract = request('cuda:0')['runtime_contract']
        env = {**contract['visibility'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'HF_TOKEN': 'synthetic-secret'}
        adapter.bind_environment(contract, self.root, env)
        self.assertEqual(env['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
        self.assertNotIn('HF_TOKEN', env)
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['OMP_NUM_THREADS'], '2')
        with self.assertRaises(ValueError):
            adapter.bind_environment(contract, self.root, {'CUDA_VISIBLE_DEVICES': '0'})

    def test_configured_steps_and_independent_text_limits(self):
        value = request()
        value['steps'] = 1001
        with self.assertRaises(ValueError): adapter.validate_request(value)
        value['runtime_contract']['max_steps'] = 1200
        self.assertIs(adapter.validate_request(value), value)
        value['runtime_contract']['max_steps'] = 1000
        with self.assertRaises(ValueError): adapter.validate_request(value)
        for field, maximum, size in (('prompt', 'max_prompt_chars', 32769), ('lyrics', 'max_text_chars', 100001)):
            candidate = request()
            candidate.update(mode='song', **{field: 'x' * size})
            with self.assertRaises(ValueError): adapter.validate_request(candidate)
            candidate['runtime_contract'][maximum] = size
            adapter.validate_request(candidate)
            candidate['runtime_contract'][maximum] = size - 1
            with self.assertRaises(ValueError): adapter.validate_request(candidate)
        for raised, lowered, field in (('max_prompt_chars', 'max_text_chars', 'lyrics'),
                                       ('max_text_chars', 'max_prompt_chars', 'prompt')):
            candidate = request()
            candidate.update(mode='song', prompt='short', lyrics='short')
            candidate['runtime_contract'].update({raised: 200000, lowered: 4})
            with self.assertRaises(ValueError): adapter.validate_request(candidate)
            candidate[field] = 'four'
            adapter.validate_request(candidate)
        candidate = request()
        candidate.update(mode='song', lyrics='x' * 65537)
        adapter.validate_request(candidate)  # Already within the default configured text bound.
        for field in ('max_steps', 'max_prompt_chars', 'max_text_chars'):
            for invalid in (True, 0, -1, 2.5):
                candidate = request()
                candidate['runtime_contract'][field] = invalid
                with self.assertRaises(ValueError): adapter.validate_request(candidate)

    def test_configured_unicode_text_is_not_capped_by_request_json_bytes(self):
        value = request()
        value.update(mode='song', lyrics='\u65e5' * 30000)
        path = self.root / 'large-request.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        self.assertGreater(path.stat().st_size, 128 * 1024)
        accepted = adapter.validate_request(adapter.read_json(path, limit=None))
        self.assertEqual(accepted['lyrics'], value['lyrics'])

    def test_trained_token_window_still_rejects_without_truncation(self):
        class Tokenizer:
            def __init__(self): self.calls = []
            def __call__(self, text, **options):
                self.calls.append(options)
                return {'input_ids': list(range(len(text)))}
        raw = Tokenizer()
        tokenizer = adapter.NoTruncationTokenizer(raw)
        with self.assertRaisesRegex(ValueError, 'conditioning window'):
            tokenizer('long', truncation=True, max_length=3)
        self.assertEqual(len(raw.calls), 1)
        self.assertFalse(raw.calls[0]['truncation'])

    def test_cpu_resolution_does_not_consult_cuda(self):
        fake = FakeTorch()
        contract = request()['runtime_contract']
        with patch.dict(os.environ, contract['visibility'], clear=True):
            runtime, dtype = adapter.resolve_runtime(fake, contract)
        self.assertEqual(fake.calls, [])
        self.assertEqual((runtime['device_uuid'], runtime['visible_devices'], dtype), (None, [], 'float32'))

    def test_explicit_cuda_identity_dtype_and_no_fallback(self):
        contract = request('cuda:0', 'bfloat16')['runtime_contract']
        with patch.dict(os.environ, contract['visibility'], clear=True):
            runtime, dtype = adapter.resolve_runtime(FakeTorch(), contract)
            self.assertEqual((runtime['device_uuid'], dtype), (UUID, 'bfloat16'))
            failures = (
                FakeTorch(available=False), FakeTorch(uuids=(UUID, UUID)), FakeTorch(bf16=False),
                FakeTorch(uuids=('GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',)),
            )
            for fake in failures:
                with self.assertRaises(ValueError):
                    adapter.resolve_runtime(fake, contract)
            contract['dtype'] = 'auto'
            self.assertEqual(adapter.resolve_runtime(FakeTorch(bf16=False), contract)[1], 'float32')

    def resolved(self, device, placement='auto', dtype='auto', **fake):
        contract = request(device, dtype, placement)['runtime_contract']
        with patch.dict(os.environ, contract['visibility'], clear=True):
            runtime, dtype = adapter.resolve_runtime(FakeTorch(**fake), contract)
        return contract, runtime, dtype

    def test_placement_resolution_records_measured_basis(self):
        accelerate = ModuleType('accelerate')
        accelerate.__version__ = 'fixture'
        refusal = re.escape(adapter.PLACEMENT_REFUSAL)
        bf16_bill, f32_bill = TOTAL * 2 + 4 * GIB, TOTAL * 4 + 4 * GIB
        with patch.object(adapter, 'bundle_parameter_counts', return_value=COUNTS):
            contract, runtime, dtype = self.resolved('cpu')
            fake = FakeTorch()
            value = adapter.resolve_placement(fake, runtime, contract, ROOT, dtype)
            self.assertEqual((value['requested'], value['resolved'], value['device_free_bytes'], value['accelerate_version']),
                             ('auto', 'host', None, None))
            # CPU resolves float32: itemsize 4 for every component, the whole bundle plus 4 GiB headroom.
            self.assertEqual((value['total_resident_bytes'], value['peak_set_bytes'], value['headroom_bytes'], value['required_bytes']),
                             (TOTAL * 4, PEAK * 4, 4 * GIB, f32_bill))
            self.assertNotIn('mem_get_info', fake.calls)
            self.assertIs(runtime['placement'], value)
            contract, runtime, dtype = self.resolved('cpu', 'resident')
            with self.assertRaisesRegex(ValueError, '^' + refusal + 'resident placement requires a CUDA device$'):
                adapter.resolve_placement(FakeTorch(), runtime, contract, ROOT, dtype)
            self.assertNotIn('placement', runtime)
            self.assertEqual((runtime['placement_refused']['resolved'], runtime['placement_refused']['required_bytes']), (None, f32_bill))
            with patch.dict(sys.modules, {'accelerate': accelerate}):
                contract, runtime, dtype = self.resolved('cuda:0', 'auto', 'bfloat16')
                fits = adapter.resolve_placement(FakeTorch(free=80 * GIB), runtime, contract, ROOT, dtype)
                self.assertEqual((fits['resolved'], fits['device_free_bytes'], fits['required_bytes'], fits['total_resident_bytes'],
                                  fits['peak_set_bytes'], fits['accelerate_version']),
                                 ('resident', 80 * GIB, bf16_bill, TOTAL * 2, PEAK * 2, 'fixture'))
                self.assertIn('accelerate fixture', fits['basis'])
                # Informational only, and only where WDDM applies: device allocations also charge process commit.
                self.assertEqual(adapter.COMMIT_NOTE in fits['basis'], sys.platform == 'win32')
                tight = adapter.resolve_placement(FakeTorch(free=bf16_bill - 1), runtime, contract, ROOT, dtype)
                self.assertNotIn(adapter.COMMIT_NOTE, tight['basis'])
                self.assertEqual(tight['resolved'], 'offload')
                self.assertIn('device free {} < required {}'.format(bf16_bill - 1, bf16_bill), tight['basis'])
                self.assertEqual(adapter.resolve_placement(FakeTorch(free=bf16_bill), runtime, contract, ROOT, dtype)['resolved'], 'resident')
                contract, runtime, dtype = self.resolved('cuda:0', 'resident', 'bfloat16')
                with self.assertRaisesRegex(ValueError, '^' + refusal + 'resident placement: device free 1024 < required {} = {} resident bytes'.format(bf16_bill, TOTAL * 2)):
                    adapter.resolve_placement(FakeTorch(free=1024), runtime, contract, ROOT, dtype)
                self.assertNotIn('placement', runtime)
                self.assertEqual(runtime['placement_refused']['device_free_bytes'], 1024)
                self.assertEqual(adapter.resolve_placement(FakeTorch(free=80 * GIB), runtime, contract, ROOT, dtype)['resolved'], 'resident')
                unsupported = dict(runtime, bfloat16_supported=False)
                with self.assertRaisesRegex(ValueError, '^' + refusal + 'resident placement: resolved dtype bfloat16 is not supported'):
                    adapter.resolve_placement(FakeTorch(free=80 * GIB), unsupported, contract, ROOT, 'bfloat16')
                contract, runtime, dtype = self.resolved('cuda:0', 'offload', 'bfloat16')
                offload = adapter.resolve_placement(FakeTorch(free=80 * GIB), runtime, contract, ROOT, dtype)
                self.assertEqual((offload['requested'], offload['resolved'], offload['accelerate_version']), ('offload', 'offload', None))
                contract, runtime, dtype = self.resolved('cuda:0', 'auto', 'float32')
                # 30 GiB holds the bfloat16 bill (about 22.3 GiB) but not the float32 bill (about 40.5 GiB).
                doubled = adapter.resolve_placement(FakeTorch(free=30 * GIB), runtime, contract, ROOT, dtype)
                self.assertEqual((doubled['required_bytes'], doubled['total_resident_bytes'], doubled['resolved']), (f32_bill, TOTAL * 4, 'offload'))
                self.assertEqual(adapter.resolve_placement(FakeTorch(free=48 * GIB), runtime, contract, ROOT, dtype)['resolved'], 'resident')
            with patch.dict(sys.modules, {'accelerate': None}):
                contract, runtime, dtype = self.resolved('cuda:0', 'auto', 'bfloat16')
                absent = adapter.resolve_placement(FakeTorch(free=80 * GIB), runtime, contract, ROOT, dtype)
                self.assertEqual((absent['resolved'], absent['accelerate_version']), ('offload', None))
                self.assertIn('accelerate is not importable', absent['basis'])
                contract, runtime, dtype = self.resolved('cuda:0', 'resident', 'bfloat16')
                with self.assertRaisesRegex(ValueError, '^' + refusal + 'resident placement: accelerate is not importable'):
                    adapter.resolve_placement(FakeTorch(free=80 * GIB), runtime, contract, ROOT, dtype)
            for value in (fits, tight, offload, doubled, absent):
                self.assertTrue(0 < len(value['basis']) <= 400)
                self.assertEqual(set(value), PLACEMENT_KEYS)
        contract['placement'] = 'pinned'
        with self.assertRaises(ValueError):
            adapter.validate_request({**request('cuda:0'), 'runtime_contract': contract})

    def test_parameter_counts_read_safetensors_headers_not_file_bytes(self):
        bundle, payloads, shapes = self.build_bundle()
        counts = adapter.bundle_parameter_counts(bundle)
        self.assertEqual(counts, {name: sum(math.prod(shape) for shape in rows) for name, rows in shapes.items()})
        # The fixture DiT is stored F32 like the real one: its file bytes are its numel x 4, while the bf16 bill
        # counts numel x 2; the two shards are summed.
        dit_bytes = sum(len(payloads[name]) - 8 - int.from_bytes(payloads[name][:8], 'little')
                        for name in payloads if name.startswith('dit/') and name.endswith('.safetensors'))
        self.assertEqual(dit_bytes, counts['dit'] * 4)
        with patch.object(adapter, 'bundle_parameter_counts', wraps=adapter.bundle_parameter_counts):
            contract, runtime, dtype = self.resolved('cuda:0', 'offload', 'bfloat16')
            value = adapter.resolve_placement(FakeTorch(), runtime, contract, bundle, dtype)
        self.assertEqual(value['total_resident_bytes'], sum(counts.values()) * 2)
        self.assertLess(value['total_resident_bytes'], sum(len(payloads[n]) for n in payloads if n.endswith('.safetensors')))
        stray = bundle / 'core/extra/model.safetensors'
        stray.parent.mkdir()
        safetensors_fixture(stray, {'w': ('BF16', (2, 2))})
        for entry in (bundle / 'core-inventory.json',):
            inventory = json.loads(entry.read_text(encoding='utf-8'))
            inventory['files'].append({'path': 'core/extra/model.safetensors', 'bytes': stray.stat().st_size, 'sha256': adapter.digest(stray)})
            entry.write_text(json.dumps(inventory), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'outside the loaded components'):
            adapter.bundle_parameter_counts(bundle)
        broken = self.root / 'broken.safetensors'
        for raw in (b'', (5000).to_bytes(8, 'little') + b'{}', (2).to_bytes(8, 'little') + b'{}',
                    (len(b'{"__metadata__": {}}')).to_bytes(8, 'little') + b'{"__metadata__": {}}'):
            broken.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                adapter.safetensors_numel(broken)
        duplicate = b'{"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}'
        broken.write_bytes(len(duplicate).to_bytes(8, 'little') + duplicate + bytes(4))
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            adapter.safetensors_numel(broken)

    def test_main_refuses_before_any_loader_and_retains_the_measured_receipt(self):
        import numpy as np
        bundle, payloads, shapes = self.build_bundle()
        accelerate = ModuleType('accelerate')
        accelerate.__version__ = 'fixture'
        calls = []

        def fake_generate(request_value, root, runtime, dtype_name, progress):
            calls.append(dict(runtime['placement']))
            progress('sampling')
            return np.tile(np.array([[0.5], [-0.25]], dtype=np.float32), (1, 480000)), 48000, {'fixture': True}

        def run(placement, free):
            value = request('cuda:0', 'bfloat16', placement)
            value['bundle'] = str(bundle)
            folder = self.root / 'main-{}-{}'.format(placement, free)
            folder.mkdir()
            (folder / 'request.json').write_text(json.dumps(value), encoding='utf-8')
            argv = ['ace_adapter', '--request', str(folder / 'request.json'), '--output', str(folder / 'out.wav'),
                    '--result', str(folder / 'result.json'), '--progress', str(folder / 'progress.json')]
            with patch.dict(os.environ, value['runtime_contract']['visibility'], clear=True), \
                    patch.dict(sys.modules, {'torch': FakeTorch(free=free), 'accelerate': accelerate}), \
                    patch.object(adapter, 'verify_source', return_value={'revision': 'fixture', 'files_verified': 0}), \
                    patch.object(adapter, 'generate', side_effect=fake_generate), \
                    patch.object(sys, 'argv', argv):
                try:
                    adapter.main()
                except ValueError as error:
                    outcome = str(error)
                else:
                    outcome = None
            result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            progress = json.loads((folder / 'progress.json').read_text(encoding='utf-8'))
            return outcome, result, progress, folder

        outcome, result, progress, folder = run('resident', 1024)
        self.assertTrue(outcome.startswith(adapter.PLACEMENT_REFUSAL + 'resident placement: device free 1024 < required '), outcome)
        self.assertEqual(calls, [])  # Refused before generate(): no from_pretrained, no torch.load, no weights.
        self.assertFalse((folder / 'out.wav').exists())
        self.assertEqual((result['evidence']['status'], result['evidence']['error']['message'], progress['stage']),
                         ('failed', outcome, 'failed'))
        refused = result['evidence']['runtime']['placement_refused']
        self.assertEqual((refused['requested'], refused['resolved'], refused['device_free_bytes']), ('resident', None, 1024))
        self.assertNotIn('placement', result['evidence']['runtime'])
        outcome, result, progress, folder = run('auto', 1024)
        self.assertIsNone(outcome)
        self.assertEqual([row['resolved'] for row in calls], ['offload'])
        self.assertEqual((result['evidence']['status'], result['evidence']['runtime']['placement']['resolved'], progress['stage']),
                         ('completed', 'offload', 'completed'))
        self.assertEqual(result['evidence']['audio']['frames'], 480000)

    def test_component_context_retains_when_resident_and_offloads_otherwise(self):
        stages = []
        for resident, device in ((True, 'cuda:0'), (False, 'cuda:0'), (False, 'cpu')):
            contract, runtime, dtype = self.resolved(device)
            observer = adapter.PlacementObserver(runtime, dtype)
            reclaimed = []
            context = adapter.component_manager(device, dtype, resident, observer, stages.append, {}, lambda: reclaimed.append(1))
            model = FakeModule(device, dtype)
            with context('dit', model):
                with context('dit', model):
                    pass  # Re-entry performs no second move and no second record.
                self.assertEqual(model.moves, [((), {'device': device, 'dtype': dtype})])
            rows = runtime['parameter_placements']
            if resident:
                self.assertEqual(len(model.moves), 1)
                self.assertEqual([row['stage'] for row in rows], ['retain'])
            else:
                self.assertEqual(model.moves[1], (('cpu',), {}))
                self.assertEqual([row['stage'] for row in rows], ['offload'])
            self.assertEqual(reclaimed, [1])
            with self.assertRaises(ValueError):
                with context('vae', None):
                    pass
        self.assertEqual(stages, ['sampling'] * 3)

    def test_receipt_carries_resolved_placement(self):
        for placement, offload in (('resident', False), ('offload', True), ('host', False)):
            check = {'requested': 'auto', 'resolved': placement, 'device_free_bytes': 1, 'required_bytes': 2,
                     'total_resident_bytes': 3, 'peak_set_bytes': 4, 'headroom_bytes': 5, 'accelerate_version': None, 'basis': 'fixture'}
            runtime = {'placement': check, 'torch_num_threads': 2, 'torch_num_interop_threads': 1}
            params = SimpleNamespace(inference_steps=8, guidance_scale=7.0, dcw_enabled=False, thinking=True, use_cot_metas=False,
                                     use_cot_caption=False, use_cot_language=False)
            receipt = adapter.generation_receipt(request(), runtime, 5, params, 'bfloat16', 'sdpa', 'unknown', 10.0,
                                                 {'planner_released': True, 'precomputed_hint_calls': 1})
            self.assertEqual((receipt['cpu_offload'], receipt['placement'], receipt['steps'], receipt['seed'], receipt['dcw_enabled']),
                             (offload, placement, 8, 5, False))
            self.assertEqual((receipt['thinking'], receipt['use_cot_metas'], receipt['use_cot_caption'], receipt['use_cot_language'],
                              receipt['planner_plan'], receipt['dit_conditioning']), (True, False, False, False, None, None))
            self.assertEqual(receipt['placement_check'], check)
            self.assertIsNot(receipt['placement_check'], check)
            self.assertFalse(receipt['decode']['device_fallback'])
        # The streaming loader's measurements ride in the receipt copy only; the parent's placement block keeps its key set.
        loader = {'mode': 'streamed', 'host_mapping': 'none', 'files': 8, 'tensors': 2, 'bytes_read': 3, 'largest_tensor_bytes': 1, 'components': {}}
        runtime = {'placement': check, 'resident_loader': loader, 'torch_num_threads': 2, 'torch_num_interop_threads': 1}
        receipt = adapter.generation_receipt(request(), runtime, 5, params, 'bfloat16', 'sdpa', 'unknown', 10.0,
                                             {'planner_released': True, 'precomputed_hint_calls': 1})
        self.assertEqual(receipt['placement_check'], {**check, 'loader': loader})
        self.assertIsNot(receipt['placement_check']['loader'], loader)
        self.assertEqual(set(runtime['placement']), PLACEMENT_KEYS)

    def test_observed_placement_satisfies_actual_parent_validator(self):
        # Execute the real validator body without importing the serving stack.
        source = ROOT / 'src/titanos/music_backend.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_verify_runtime')
        devices = ModuleType('titanos.devices')
        devices.normalized_uuid = lambda value: adapter.normalized_uuid(value)[4:]
        namespace = {'__package__': 'titanos', 're': re, 'EngineError': RuntimeError,
                     '_DTYPES': {'float32', 'float16', 'bfloat16'}, '_PLACEMENT_KEYS': PLACEMENT_KEYS}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        accelerate = ModuleType('accelerate')
        accelerate.__version__ = 'fixture'
        for device, placement in (('cpu', 'auto'), ('cuda:0', 'offload'), ('cuda:0', 'resident')):
            contract, runtime, dtype = self.resolved(device, placement)
            with patch.object(adapter, 'bundle_parameter_counts', return_value=COUNTS), \
                    patch.dict(sys.modules, {'accelerate': accelerate}):
                adapter.resolve_placement(FakeTorch(), runtime, contract, ROOT, dtype)
            resident = runtime['placement']['resolved'] == 'resident'
            observer = adapter.PlacementObserver(runtime, dtype)
            context = adapter.component_manager(device, dtype, resident, observer, lambda stage: None, {}, lambda: None)
            for component in ('dit', 'text_encoder', 'vae', 'planner'):
                model = FakeModule(device, dtype)
                observer.attach(component, model)
                self.assertNotIn(component, observer.executed)
                with context(component, model):
                    model.execute_fixture()
                self.assertIn(component, observer.executed)
                observer.detach(component)
                self.assertIsNone(model.hook)
            stages = {row['stage'] for row in runtime['parameter_placements']}
            self.assertEqual(stages, {'execute', 'retain' if resident else 'offload'})
            metadata = {'effective': {'dtype': dtype, 'threads': 2, 'interop_threads': 1,
                                      'cpu_offload': runtime['placement']['resolved'] == 'offload'},
                        'evidence': {'runtime': runtime}}
            with patch.dict(sys.modules, {'titanos.devices': devices}):
                self.assertIs(namespace['_verify_runtime'](metadata, contract), runtime)
                changes = [lambda m: m['evidence']['runtime']['parameter_placements'][0].update(device='cuda:1'),
                           lambda m: m['evidence']['runtime'].pop('placement'),
                           lambda m: m['evidence']['runtime']['placement'].update(requested='resident' if placement != 'resident' else 'auto'),
                           lambda m: m['evidence']['runtime']['placement'].update(resolved='offload' if resident else 'retain'),
                           lambda m: m['evidence']['runtime']['placement'].pop('peak_set_bytes'),
                           lambda m: m['evidence']['runtime']['placement'].update(total_resident_bytes=-1),
                           lambda m: m['effective'].update(cpu_offload=not m['effective']['cpu_offload']),
                           lambda m: m['effective'].pop('cpu_offload'),
                           lambda m: m['evidence']['runtime']['parameter_placements'].append(
                               {**m['evidence']['runtime']['parameter_placements'][0], 'stage': 'offload' if resident else 'retain'})]
                if resident:
                    changes.append(lambda m: m['evidence']['runtime']['placement'].update(accelerate_version=None))
                for change in changes:
                    bad = deepcopy(metadata)
                    change(bad)
                    with self.assertRaises(RuntimeError):
                        namespace['_verify_runtime'](bad, contract)
            with self.assertRaises(ValueError):
                observer.record('dit', 'execute', FakeModule('cuda:1'), execution=True)

    def test_source_hash_check_reads_inert_fixture_without_import(self):
        package = self.root / 'acestep'
        package.mkdir()
        entry = package / '__init__.py'
        entry.write_text('raise AssertionError("fixture must never execute")', encoding='utf-8')
        pins = {'__init__.py': adapter.digest(entry)}
        with patch.object(adapter, 'SOURCE_PINS', pins), patch.object(
            adapter.importlib.util, 'find_spec', return_value=SimpleNamespace(origin=str(entry))
        ):
            self.assertEqual(adapter.verify_source()['files_verified'], 1)
            entry.write_text('changed fixture', encoding='utf-8')
            with self.assertRaises(ValueError):
                adapter.verify_source()

    def build_bundle(self):
        bundle = self.root / 'bundle'
        bundle.mkdir()
        payloads = {
            'dit/config.json': {'model_type': 'acestep'},
            'dit/silence_latent.pt': b'INERT FIXTURE; NEVER DESERIALIZED',
            'core/vae/config.json': {'_class_name': 'AutoencoderOobleck', 'sampling_rate': 48000,
                                     'audio_channels': 2, 'downsampling_ratios': [2, 4, 4, 6, 10]},
            'core/Qwen3-Embedding-0.6B/config.json': {'model_type': 'qwen3', 'architectures': ['Qwen3Model']},
            'core/Qwen3-Embedding-0.6B/tokenizer_config.json': {'tokenizer_class': 'Qwen2Tokenizer'},
            'planner/config.json': {'model_type': 'qwen3', 'architectures': ['Qwen3ForCausalLM']},
            'planner/tokenizer_config.json': {'tokenizer_class': 'Qwen2Tokenizer'},
        }
        # Inert safetensors weights: the DiT is stored F32 across two shards like the real bundle, the rest BF16.
        weights = {
            'planner/model-00001-of-00002.safetensors': ('planner', {'a': ('BF16', (64, 32))}),
            'planner/model-00002-of-00002.safetensors': ('planner', {'b': ('BF16', (16,))}),
            'dit/model-00001-of-00002.safetensors': ('dit', {'c': ('F32', (48, 24))}),
            'dit/model-00002-of-00002.safetensors': ('dit', {'d': ('F32', (8, 8, 2)), 'e': ('F32', ())}),
            'core/Qwen3-Embedding-0.6B/model.safetensors': ('text_encoder', {'f': ('BF16', (32, 8))}),
            'core/vae/diffusion_pytorch_model.safetensors': ('vae', {'g': ('BF16', (4, 4, 4))}),
        }
        shapes = {name: [] for name in adapter.PLACEMENT_COMPONENTS}
        entries = {}
        for name, value in payloads.items():
            path = bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
            entries[name] = {'path': name, 'bytes': path.stat().st_size, 'sha256': adapter.digest(path)}
        for name, (component, tensors) in weights.items():
            path = bundle / name
            safetensors_fixture(path, tensors)
            shapes[component].extend(shape for _dtype, shape in tensors.values())
            payloads[name] = path.read_bytes()
            entries[name] = {'path': name, 'bytes': path.stat().st_size, 'sha256': adapter.digest(path)}
        manifest = {'schema_version': 1, 'family': 'ace-step-1.5', 'variant': 'xl-sft',
                    'code_revision': adapter.CODE_REVISION, 'model_revision': adapter.MODEL_REVISION, 'components': {}}
        for component, revision in adapter.COMPONENT_REVISIONS.items():
            path = bundle / (component + '-inventory.json')
            path.write_text(json.dumps({'revision': revision, 'files': [v for k, v in entries.items()
                                                                        if k.startswith(component + '/')]}), encoding='utf-8')
            manifest['components'][component] = {'path': path.name, 'revision': revision, 'sha256': adapter.digest(path)}
        (bundle / 'titanos-music.json').write_text(json.dumps(manifest), encoding='utf-8')
        return bundle, payloads, shapes

    def test_bundle_inventory_rejects_tamper_unlisted_payload_and_escape(self):
        bundle, payloads, shapes = self.build_bundle()
        value = request()
        value['bundle'] = str(bundle)
        root, proof = adapter.verify_bundle(value, lambda stage: None)
        self.assertEqual(proof['payload_files_verified'], len(payloads))
        extra = bundle / 'dit/model.safetensors'
        extra.write_bytes(b'UNLISTED INERT FIXTURE')
        with self.assertRaises(ValueError): adapter.verify_bundle(value, lambda stage: None)
        extra.unlink()
        for name in ('../outside.json', '/absolute.json', 'C:/foreign.json', 'x\\y'):
            with self.assertRaises(ValueError): adapter.member(root, name)
        (bundle / 'dit/config.json').write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError): adapter.verify_bundle(value, lambda stage: None)

    def test_synthetic_stereo_pcm_readback_and_resource_bounds(self):
        import numpy as np
        value = request()
        samples = np.tile(np.array([[1.0], [-2.0]], dtype=np.float32), (1, 480000))
        path = self.root / 'synthetic-fixture.wav'
        proof = adapter.write_audio(samples, 48000, path, value)
        self.assertEqual((proof['channels'], proof['frames'], proof['peak_scale']), (2, 480000, 2.0))
        with wave.open(str(path), 'rb') as stream:
            self.assertEqual((stream.getframerate(), stream.getsampwidth()), (48000, 2))
            first = np.frombuffer(stream.readframes(1), dtype='<i2')
        self.assertGreater(first[0], 0)
        self.assertLess(first[1], 0)
        self.assertGreater(proof['rms_pcm16'], 0)
        invalid = (np.zeros_like(samples), np.full_like(samples, 1e-20), samples[:, :1],
                   np.full_like(samples, float('nan')), samples[:1])
        for bad in invalid:
            with self.assertRaises(ValueError):
                adapter.write_audio(bad, 48000, self.root / 'rejected.wav', value)
        for key, maximum in [('max_samples', 10), ('max_output_bytes', 44), ('max_audio_seconds', 1)]:
            limited = deepcopy(value)
            limited['runtime_contract'][key] = maximum
            with self.assertRaises(ValueError):
                adapter.write_audio(samples, 48000, self.root / 'too-large.wav', limited)

    def test_instrumental_marker_and_supplied_song_lyrics(self):
        self.assertEqual(adapter.generation_parameters(request(), 123, 'unknown')['lyrics'], '[Instrumental]')
        value = request()
        value.update(mode='song', lyrics='Exact user lyrics\nSecond line', bpm=120, key_scale='C minor')
        song = adapter.generation_parameters(value, 456, 'en')
        self.assertEqual((song['lyrics'], song['seed'], song['bpm'], song['keyscale']),
                         (value['lyrics'], 456, 120, 'C minor'))
        self.assertFalse(song['instrumental'])
        self.assertFalse(song['use_cot_lyrics'])

    def test_prompt_is_the_caption_and_dcw_stays_off(self):
        # Upstream enables DCW only for Turbo (_resolve_dcw_enabled -> is_turbo_model()). Forced on over 50 XL-SFT steps
        # it compounded until the seed decided the track and different prompts gave the same piece (2026-10-22).
        value = request()
        value.update(prompt='solo concert harp lullaby, slow, gentle, no drums', steps=50)
        params = adapter.generation_parameters(value, 314159, 'unknown')
        self.assertEqual((params['caption'], params['inference_steps'], params['seed']), (value['prompt'], 50, 314159))
        self.assertIs(params['dcw_enabled'], False)

    def test_planner_route_flags_default_to_the_codes_route_and_pass_through(self):
        # Absent flags keep the route every earlier receipt recorded: the planner writes audio codes from the user metadata.
        params = adapter.generation_parameters(request(), 7, 'unknown')
        self.assertEqual({key: params[key] for key in adapter.PLANNER_DEFAULTS},
                         {'thinking': True, 'use_cot_metas': False, 'use_cot_caption': False, 'use_cot_language': False})
        for flags in ({'thinking': True, 'use_cot_metas': True, 'use_cot_caption': True, 'use_cot_language': True},
                      {'thinking': False, 'use_cot_metas': True}):
            with self.subTest(flags=flags):
                params = adapter.generation_parameters({**request(), **flags}, 7, 'unknown')
                self.assertEqual({key: params[key] for key in adapter.PLANNER_DEFAULTS}, {**adapter.PLANNER_DEFAULTS, **flags})
                self.assertIs(params['use_cot_lyrics'], False)
                self.assertIs(params['dcw_enabled'], False)

    def test_bounded_json_rejects_duplicate_keys(self):
        path = self.root / 'input.json'
        path.write_text('{"seed": 1, "seed": 2}', encoding='utf-8')
        with self.assertRaises(ValueError): adapter.read_json(path)
        path.write_text('{"seed": 1}', encoding='utf-8')
        with self.assertRaises(ValueError): adapter.read_json(path, limit=1)
        self.assertEqual(adapter.read_json(path), {'seed': 1})

    def test_safetensors_table_is_strict_about_layout_dtype_and_shape(self):
        path = self.root / 'table.safetensors'
        valid = {'__metadata__': {'format': 'pt'},
                 'scalar': {'dtype': 'F32', 'shape': [], 'data_offsets': [8, 12]},
                 'empty': {'dtype': 'BF16', 'shape': [0, 3], 'data_offsets': [12, 12]},
                 'ints': {'dtype': 'I32', 'shape': [2], 'data_offsets': [0, 8]}}
        entries, length, data_bytes = adapter.safetensors_table(raw_safetensors(path, valid, bytes(12)))
        self.assertEqual((set(entries), data_bytes), ({'scalar', 'empty', 'ints'}, 12))
        self.assertEqual(entries['scalar'], {'dtype': 'F32', 'shape': [], 'data_offsets': [8, 12]})
        self.assertEqual(adapter.safetensors_numel(path), 3)  # 0-dim counts one element, the empty tensor none.
        rejected = {
            'gap': ({'a': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}, 'b': {'dtype': 'F32', 'shape': [1], 'data_offsets': [8, 12]}}, bytes(12), 'overlap or leave a gap'),
            'overlap': ({'a': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}, 'b': {'dtype': 'F32', 'shape': [2], 'data_offsets': [4, 12]}}, bytes(12), 'overlap or leave a gap'),
            'beyond file': ({'a': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}}, bytes(8), 'outside the file'),
            'negative begin': ({'a': {'dtype': 'F32', 'shape': [1], 'data_offsets': [-4, 0]}}, bytes(4), 'outside the file'),
            'missing offsets': ({'a': {'dtype': 'F32', 'shape': [1]}}, bytes(4), 'outside the file'),
            'float offsets': ({'a': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0.0, 4.0]}}, bytes(4), 'outside the file'),
            'undescribed tail': ({'a': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}, bytes(8), 'not fully described'),
            'unknown dtype': ({'a': {'dtype': 'F8_E4M3', 'shape': [4], 'data_offsets': [0, 4]}}, bytes(4), 'Unsupported safetensors dtype F8_E4M3'),
            'span differs': ({'a': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 4]}}, bytes(4), 'byte span differs'),
            'negative dim': ({'a': {'dtype': 'F32', 'shape': [-1], 'data_offsets': [0, 4]}}, bytes(4), 'Invalid safetensors tensor entry'),
            'entry not a table': ({'a': 'F32'}, bytes(4), 'Invalid safetensors tensor entry'),
            'metadata only': ({'__metadata__': {}}, b'', 'Empty safetensors table'),
            'duplicate key': (b'{"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}', bytes(4), 'Duplicate JSON key'),
        }
        for label, (header, data, message) in rejected.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                adapter.safetensors_table(raw_safetensors(path, header, data))
        for raw in (b'', (5000).to_bytes(8, 'little') + b'{}', (2).to_bytes(8, 'little') + b'{}'):
            path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                adapter.safetensors_table(path)

    def test_stream_safetensors_reads_one_tensor_at_a_time_onto_the_device(self):
        path = self.root / 'stream.safetensors'
        payload = {'a': ('F32', (2, 3), struct.pack('<6f', 0, 1, 2, 3, 4, 5)), 'b': ('I32', (3,), struct.pack('<3i', 1, -2, 3)),
                   'c': ('F32', (), struct.pack('<f', 2.5)), 'd': ('BF16', (0, 3), b''), 'e': ('BOOL', (2,), b'\x01\x00')}
        safetensors_payload(path, payload, order=['e', 'c', 'a', 'd', 'b'])  # JSON order differs from data order.
        fake = FakeTensorTorch()
        loader = adapter.new_loader_evidence()
        got = adapter.stream_safetensors(fake, path, 'cuda:0', fake.bfloat16, loader)
        self.assertEqual(list(got), ['a', 'b', 'c', 'd', 'e'])  # Read in data-offset order, not JSON order.
        for name, (dtype, shape, raw) in payload.items():
            with self.subTest(name=name):
                tensor = got[name]
                self.assertEqual((tensor.data, tensor.shape, tensor.device), (raw, shape, 'cuda:0'))
                floating = adapter.SAFETENSORS_DTYPES[dtype][0] in {'float32', 'bfloat16'}
                self.assertEqual(tensor.dtype.name, 'bfloat16' if floating else adapter.SAFETENSORS_DTYPES[dtype][0])
                # Exactly one host->device move per tensor at the stored dtype; a cast, when needed, happens on the device.
                self.assertEqual(tensor.moves, (('cpu', 'cuda:0'),))
                self.assertEqual(tensor.casts, (('cuda:0', 'bfloat16'),) if dtype == 'F32' else ())
        self.assertEqual(fake.buffers, ['bytearray'] * 4)  # Writable buffers; the empty tensor never touches frombuffer.
        self.assertEqual(loader, {'mode': 'streamed', 'host_mapping': 'none', 'files': 1, 'tensors': 5, 'bytes_read': 42,
                                  'largest_tensor_bytes': 24, 'components': {}})
        untouched = adapter.stream_safetensors(fake, path, 'cpu', None)
        self.assertEqual((untouched['a'].dtype.name, untouched['a'].casts, untouched['b'].dtype.name), ('float32', (), 'int32'))
        # A file whose data region is shorter than its header claims is refused before any tensor is read.
        raw_safetensors(path, {'a': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}, bytes(6))
        with self.assertRaisesRegex(ValueError, 'outside the file'):
            adapter.stream_safetensors(fake, path, 'cpu', None)
        self.assertEqual(len(fake.buffers), 8)  # Two successful streams so far; the refused file added no read.

    def test_resident_path_uses_no_host_mapping_primitives(self):
        tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.update(node.name.split('.'))
                if node.asname:
                    names.add(node.asname)
        self.assertFalse(names & {'safe_open', 'load_file', 'mmap', 'memmap', 'from_file', 'UntypedStorage', 'load_state_dict_from_file'})
        # Keyword forms map files too (torch.load(..., mmap=True)); numpy.memmap is an attribute the walk above catches.
        self.assertFalse([node.arg for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg in {'mmap', 'memmap'}])
        generate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'generate')
        offload_only = set()
        for node in ast.walk(generate):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'resident':
                for branch in node.orelse:
                    offload_only.update(id(child) for child in ast.walk(branch))
        loads = [node for node in ast.walk(generate) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'from_pretrained' and isinstance(node.func.value, ast.Name)
                 and node.func.value.id in {'Qwen3ForCausalLM', 'Qwen3Model', 'AceStepConditionGenerationModel', 'AutoencoderOobleck'}]
        self.assertEqual(len(loads), 4)
        self.assertTrue(all(id(node) in offload_only for node in loads))  # Weight-loading from_pretrained only when not resident.

    def test_build_resident_accounts_for_every_key_and_refuses_otherwise(self):
        fake = FakeTensorTorch()
        seen = []
        shard_a = safetensors_payload(self.root / 'a.safetensors', {'model.embed.weight': ('BF16', (2,), b'\x80\x3f\x80\x3f'),
                                                                      'model.norm.weight': ('BF16', (1,), b'\x80\x3f')})
        shard_b = safetensors_payload(self.root / 'b.safetensors', {'model.head.bias': ('BF16', (1,), b'\x00\x00'), 'model.ids': ('I32', (1,), struct.pack('<i', 7))})
        keys = ['model.embed.weight', 'model.norm.weight', 'model.head.bias', 'model.ids', 'lm_head.weight']
        refusal = re.escape(adapter.RESIDENT_REFUSAL)

        def build(model, files=(shard_a, shard_b), tied=('lm_head.weight',), loader=None):
            loader = adapter.new_loader_evidence() if loader is None else loader
            constructed = []

            def construct():
                self.assertEqual(seen[-1], False)  # Constructed under init_empty_weights(include_buffers=False).
                constructed.append(model)
                return model
            with patch.dict(sys.modules, {'accelerate': fake_accelerate(seen)}):
                built = adapter.build_resident(fake, 'planner', construct, list(files), 'cuda:0', fake.bfloat16, set(tied), loader)
            self.assertEqual(constructed, [model])
            return built, loader

        model = FakeResidentModel(keys, tied=['lm_head.weight'], buffers=['rotary.inv_freq'])
        built, loader = build(model)
        self.assertIs(built, model)
        self.assertTrue(model.evaluated)
        self.assertIs(model.params['lm_head.weight'], model.params['model.embed.weight'])
        self.assertEqual({str(value.device) for value in model.params.values()}, {'cuda:0'})
        self.assertEqual((model.params['model.ids'].dtype.name, model.params['model.embed.weight'].dtype.name), ('int32', 'bfloat16'))
        self.assertEqual((model.buffers['rotary.inv_freq'].device, model.moved_to), ('cuda:0', ['cuda:0']))
        self.assertEqual(loader['components'], {'planner': {'files': 2, 'parameters': 5, 'tied_keys': ['lm_head.weight'],
                                                            'renamed_keys': 0, 'buffers': {'float32': 1}}})
        self.assertEqual((loader['files'], loader['tensors'], loader['bytes_read'], loader['largest_tensor_bytes']), (2, 4, 12, 4))
        cases = {
            'tied key without tie_word_embeddings': (FakeResidentModel(keys, tied=['lm_head.weight'], tie=False), (shard_a, shard_b), (), 'missing checkpoint keys'),
            'recipe expects no tied key': (FakeResidentModel(keys, tied=['lm_head.weight']), (shard_a, shard_b), (), 'tied-weight keys'),
            'missing non-tied key': (FakeResidentModel(keys, tied=['lm_head.weight']), (shard_a,), ('lm_head.weight',), 'missing checkpoint keys'),
            'unexpected key': (FakeResidentModel(keys[:2] + keys[4:], tied=['lm_head.weight']), (shard_a, shard_b), ('lm_head.weight',), 'unexpected checkpoint keys'),
            'tie does not bind': (FakeResidentModel(keys, tied=['lm_head.weight'], bind=False), (shard_a, shard_b), ('lm_head.weight',), 'tied key lm_head.weight is not bound'),
            'buffer left on host': (FakeResidentModel(keys, tied=['lm_head.weight'], buffers=['b'], move_buffers=False), (shard_a, shard_b), ('lm_head.weight',), 'buffer b is on cpu'),
            'parameter kept off device/dtype': (FakeResidentModel(keys, tied=['lm_head.weight'], stubborn=['model.norm.weight']), (shard_a, shard_b), ('lm_head.weight',), 'parameter model.norm.weight is on cpu as torch.float32'),
            'duplicate key across shards': (FakeResidentModel(keys, tied=['lm_head.weight']), (shard_a, shard_a), ('lm_head.weight',), 'duplicate checkpoint key model.embed.weight across shards'),
            'shape mismatch rejected by the module': (RejectingResidentModel(keys, tied=['lm_head.weight']), (shard_a, shard_b), ('lm_head.weight',),
                                                       re.escape('state rejected by the module: Error(s) in loading state_dict for Fake: size mismatch for model.norm.weight: copying a param')),
        }
        for label, (candidate, files, tied, message) in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, '^' + refusal + 'planner: ' + message):
                build(candidate, files, tied)
            self.assertFalse(candidate.evaluated)
        # A checkpoint that also ships the configured tied key is still tied (from_pretrained ties unconditionally): one device tensor, not two.
        shipped = safetensors_payload(self.root / 'h.safetensors', {'lm_head.weight': ('BF16', (2,), b'\x00\x00\x00\x00')})
        model = FakeResidentModel(keys, tied=['lm_head.weight'])
        _built, loader = build(model, (shard_a, shard_b, shipped), ('lm_head.weight',))
        self.assertIs(model.params['lm_head.weight'], model.params['model.embed.weight'])
        self.assertEqual((loader['components']['planner']['tied_keys'], loader['tensors']), (['lm_head.weight'], 5))
        with self.assertRaisesRegex(ValueError, '^' + refusal + 'planner: tied key lm_head.weight is not bound'):
            build(FakeResidentModel(keys, tied=['lm_head.weight'], bind=False), (shard_a, shard_b, shipped), ('lm_head.weight',))
        # transformers' per-key rename hook is honoured and counted when the class has one.
        renamed = safetensors_payload(self.root / 'r.safetensors', {'conv.weight_g': ('BF16', (1,), b'\x80\x3f'), 'conv.weight_v': ('BF16', (1,), b'\x80\x3f')})

        def rename(key):
            for old, new in (('weight_g', 'parametrizations.weight.original0'), ('weight_v', 'parametrizations.weight.original1')):
                if key.endswith(old):
                    return key[:-len(old)] + new, True
            return key, False
        model = FakeResidentModel(['conv.parametrizations.weight.original0', 'conv.parametrizations.weight.original1'], rename=rename)
        _built, loader = build(model, (renamed,), ())
        self.assertEqual(loader['components']['planner']['renamed_keys'], 2)
        self.assertEqual({str(value.device) for value in model.params.values()}, {'cuda:0'})

    def test_generate_resident_branch_streams_every_component_through_build_resident(self):
        """generate() in resident placement, driven end to end with fake torch/transformers/diffusers/acestep modules: the
        four components are built through resident_constructor/build_resident with the inventoried files, the one shared
        loader evidence and the recipe's tied keys, in the pinned order (planner; then, after the planner is released,
        DiT, text encoder, VAE through the construct_vae closure); no model class's from_pretrained runs; the silence
        tensor is deserialised onto the device; every component executes on the device at the resolved dtype and the
        receipt carries the loader evidence. Nothing real is imported and no device work happens."""
        receipt, _events = self.drive_resident_generate()
        self.assertEqual(receipt['precomputed_lm_hints_calls'], 1)
        self.assertEqual({key: receipt[key] for key in adapter.PLANNER_DEFAULTS}, adapter.PLANNER_DEFAULTS)
        self.assertEqual(receipt['planner_plan'], {'duration': 10})
        self.assertEqual(receipt['dit_conditioning'], {'caption': request()['prompt'], 'bpm': None, 'key_scale': '', 'time_signature': '',
                                                       'vocal_language': 'unknown', 'audio_codes': True})

    def test_generate_with_the_planner_metadata_caption_and_language(self):
        """Option 1 of the 2026-10-22 same-track trace: the metadata pass runs, its caption replaces the prompt at the DiT,
        planner codes still condition the DiT, and the receipt records the bounded plan and what the DiT received."""
        flags = {'thinking': True, 'use_cot_metas': True, 'use_cot_caption': True, 'use_cot_language': True}
        receipt, events = self.drive_resident_generate(**flags)
        self.assertEqual(receipt['precomputed_lm_hints_calls'], 1)
        self.assertEqual({key: receipt[key] for key in adapter.PLANNER_DEFAULTS}, flags)
        caption = ('planner caption ' * 300)[:adapter.PLAN_TEXT_LIMIT]
        self.assertEqual(len(caption), 4000)
        self.assertEqual(receipt['planner_plan'], {'bpm': 96, 'caption': caption, 'duration': 10, 'genres': 'folk', 'keyscale': 'D minor',
                                                   'language': 'en', 'timesignature': '4'})
        self.assertEqual(receipt['dit_conditioning'], {'caption': caption, 'bpm': 96, 'key_scale': 'D minor', 'time_signature': '4',
                                                       'vocal_language': 'unknown', 'audio_codes': True})
        self.assertIn(('hints', '<|audio_code_1|>'), events)

    def test_generate_without_planner_codes_keeps_the_metadata_pass(self):
        """Option 2 of the trace: thinking off with the metadata pass, so the DiT samples from the prompt and the planner's
        metadata with no planner audio codes, and no conditioning hints are decoded."""
        receipt, events = self.drive_resident_generate(thinking=False, use_cot_metas=True)
        self.assertEqual(receipt['precomputed_lm_hints_calls'], 0)
        self.assertEqual({key: receipt[key] for key in adapter.PLANNER_DEFAULTS},
                         {'thinking': False, 'use_cot_metas': True, 'use_cot_caption': False, 'use_cot_language': False})
        self.assertEqual(receipt['dit_conditioning'], {'caption': request()['prompt'], 'bpm': 96, 'key_scale': 'D minor',
                                                       'time_signature': '4', 'vocal_language': 'unknown', 'audio_codes': False})
        self.assertEqual(receipt['planner_plan'], {'bpm': 96, 'duration': 10, 'genres': 'folk', 'keyscale': 'D minor', 'timesignature': '4'})
        self.assertFalse(any(isinstance(event, tuple) and event[0] == 'hints' for event in events))

    def drive_resident_generate(self, **flags):
        """Run generate() in resident placement with the fake modules described on the test above and the given planner
        route flags in the request; check the placement evidence every route shares and return the receipt and events."""
        bundle, _payloads, _shapes = self.build_bundle()
        fake, device, events, stages, seen = FakeTensorTorch(), 'cuda:0', [], [], []

        class Module(FakeResidentModel):
            """FakeResidentModel plus the module surface generate() touches: construction from a config, transformers'
            default-dtype hook, forward pre-hooks (fired by the fake pipeline), a decoder submodule, a device+dtype .to()."""
            keys, tied = (), ()

            def __init__(self, config=None):
                super().__init__(list(self.keys), tied=list(self.tied), tie=getattr(config, 'tie_word_embeddings', False))
                if config is not None:
                    self.config = config
                self.hooks = []
                self.decoder = SimpleNamespace(register_forward_pre_hook=self.register_forward_pre_hook)
                events.append(('constructed', type(self).__name__, getattr(config, '_attn_implementation', None), fake.default_dtype.name))

            @classmethod
            def _set_default_dtype(cls, dtype):
                previous = fake.default_dtype
                fake.set_default_dtype(dtype)
                return previous

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                raise AssertionError('from_pretrained must not run in resident placement: ' + cls.__name__)

            def parameters(self): return list(self.params.values())

            def register_forward_pre_hook(self, hook):
                self.hooks.append(hook)
                return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

            def fire(self):
                for hook in list(self.hooks):
                    hook(self, ())

            def to(self, device=None, dtype=None):
                return super().to(device)

        class Qwen3ForCausalLM(Module): keys, tied = ('a', 'b', 'lm_head.weight'), ('lm_head.weight',)
        class AceStepConditionGenerationModel(Module): keys = ('c', 'd', 'e')
        class Qwen3Model(Module): keys = ('f',)

        class AutoencoderOobleck(Module):
            keys = ('g',)

            @classmethod
            def load_config(cls, directory, local_files_only=False):
                events.append(('vae_config', Path(directory), local_files_only))
                return SimpleNamespace(_attn_implementation=None)

            @classmethod
            def from_config(cls, config): return cls(config)

        class Tensor:
            """The silence tensor and the hint latents: shape, device, transpose and a device/dtype .to()."""
            def __init__(self, shape, device): self.shape, self.device, self.ndim = tuple(shape), device, len(shape)

            def transpose(self, a, b):
                shape = list(self.shape)
                shape[a], shape[b] = shape[b], shape[a]
                return Tensor(shape, self.device)

            def to(self, device=None, dtype=None):
                moved = Tensor(self.shape, device or self.device)
                moved.dtype = dtype
                return moved

            def numel(self): return math.prod(self.shape)

        def torch_load(path, map_location, weights_only):
            events.append(('torch.load', Path(path), map_location, weights_only))
            return Tensor((1, 64, 1000), map_location)

        @contextmanager
        def inference_mode():
            events.append('inference_mode')
            yield

        fake.Tensor, fake.load, fake.inference_mode = Tensor, torch_load, inference_mode
        fake.isfinite = lambda _value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True))
        fake.cuda.empty_cache = lambda: events.append('empty_cache')

        class LLMHandler: pass

        class AceStepHandler:
            def _sync_alignment_config(self): events.append('alignment_synced')

            def _decode_audio_codes_to_latents(self, text):
                events.append(('hints', text))
                return Tensor((1, 8, 64), device)

            def generate_music(self, captions, lyrics='', bpm=None, key_scale='', time_signature='', vocal_language='en',
                               audio_code_string='', **_kwargs):
                # The fake pipeline body: every component executes once inside the strict per-component context; planner
                # audio codes, when the orchestrator passes any, are decoded into conditioning hints first.
                events.append(('pipeline', captions, audio_code_string))
                with self._load_model_context('model'):
                    if audio_code_string:
                        self._decode_audio_codes_to_latents(audio_code_string)
                    self.model.fire()
                with self._load_model_context('text_encoder'):
                    self.text_encoder.fire()
                with self._load_model_context('vae'):
                    self.vae.fire()
                return 'pipeline-done'

        def generate_music(handler, planner, params, config, save_dir=None, progress=None):
            # The pinned orchestrator's shape: the planner runs (its metadata pass with use_cot_metas), then the DiT handler
            # gets the conditioning by keyword, with planner audio codes only under thinking.
            events.append(('generate_music', save_dir, progress))
            with planner._load_model_context():
                planner.llm.fire()
            plan = {'duration': int(params.duration)}
            if params.use_cot_metas:
                plan.update(bpm=96, keyscale='D minor', timesignature='4', genres='folk', unexpected='dropped')
                if params.use_cot_caption:
                    plan['caption'] = 'planner caption ' * 300
                if params.use_cot_language:
                    plan['language'] = 'en'
            self.assertEqual(handler.generate_music(
                captions=plan['caption'] if params.use_cot_caption else params.caption, lyrics=params.lyrics,
                bpm=params.bpm or plan.get('bpm'), key_scale=params.keyscale or plan.get('keyscale', ''),
                time_signature=plan.get('timesignature', ''), vocal_language=params.vocal_language,
                inference_steps=params.inference_steps, seed=config.seeds[0],
                audio_code_string='<|audio_code_1|>' if params.thinking else ''), 'pipeline-done')
            pcm = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(float=lambda: 'pcm-fixture')))
            return SimpleNamespace(success=True, audios=[{'tensor': pcm, 'sample_rate': 48000}], error=None, status_message='',
                                   extra_outputs={'lm_metadata': plan})

        class MetadataConstrainedLogitsProcessor:
            def __init__(self, **kwargs): events.append(('logits_processor', sorted(kwargs)))

        def module(name, **attrs):
            value = ModuleType(name)
            value.__dict__.update(attrs)
            return value

        @contextmanager
        def no_init_weights():
            events.append('no_init_weights')
            yield

        def config_from_pretrained(directory, local_files_only=False):
            events.append(('config', Path(directory).name, local_files_only))
            return SimpleNamespace(tie_word_embeddings=True, _attn_implementation=None, dtype=None, name_or_path='', sub_configs={})

        modeling_utils = module('transformers.modeling_utils', no_init_weights=no_init_weights)
        modules = {
            'torch': fake, 'accelerate': fake_accelerate(seen),
            'transformers': module('transformers', modeling_utils=modeling_utils, Qwen3Model=Qwen3Model, Qwen3ForCausalLM=Qwen3ForCausalLM,
                                   Qwen3Config=SimpleNamespace(from_pretrained=config_from_pretrained),
                                   Qwen2TokenizerFast=SimpleNamespace(from_pretrained=lambda directory, local_files_only=False: SimpleNamespace(directory=Path(directory)))),
            'transformers.modeling_utils': modeling_utils,
            'diffusers': module('diffusers', AutoencoderOobleck=AutoencoderOobleck),
            'acestep': module('acestep'), 'acestep.handler': module('acestep.handler', AceStepHandler=AceStepHandler),
            'acestep.llm_inference': module('acestep.llm_inference', LLMHandler=LLMHandler),
            'acestep.constrained_logits_processor': module('acestep.constrained_logits_processor', MetadataConstrainedLogitsProcessor=MetadataConstrainedLogitsProcessor),
            'acestep.inference': module('acestep.inference', GenerationParams=lambda **kw: SimpleNamespace(**kw), GenerationConfig=lambda **kw: SimpleNamespace(**kw),
                                        generate_music=generate_music),
            'acestep.constants': module('acestep.constants', VALID_LANGUAGES={'unknown', 'en'}),
            'acestep.models': module('acestep.models'), 'acestep.models.common': module('acestep.models.common'),
            'acestep.models.common.configuration_acestep_v15': module('acestep.models.common.configuration_acestep_v15',
                                                                      AceStepConfig=SimpleNamespace(from_pretrained=config_from_pretrained)),
            'acestep.models.xl_sft': module('acestep.models.xl_sft'),
            'acestep.models.xl_sft.modeling_acestep_v15_xl_base': module('acestep.models.xl_sft.modeling_acestep_v15_xl_base',
                                                                         AceStepConditionGenerationModel=AceStepConditionGenerationModel),
            'pytorch_wavelets': module('pytorch_wavelets', DWT1DForward=object, DWT1DInverse=object),
        }
        original, calls = adapter.build_resident, []

        def recording_build_resident(torch, component, construct, files, device, dtype, tied_keys, loader):
            calls.append((component, list(files), device, dtype, set(tied_keys), loader, construct))
            return original(torch, component, construct, files, device, dtype, tied_keys, loader)

        value = request(device, placement='resident')
        value['bundle'], value['seed'] = str(bundle), 11
        value.update(flags)
        _contract, runtime, dtype_name = self.resolved(device, placement='resident')
        runtime['placement'] = {'requested': 'resident', 'resolved': 'resident', 'device_free_bytes': 1, 'required_bytes': 2,
                                'total_resident_bytes': 3, 'peak_set_bytes': 4, 'headroom_bytes': 5, 'accelerate_version': 'fixture', 'basis': 'fixture'}
        self.assertEqual(dtype_name, 'bfloat16')
        with patch.dict(sys.modules, modules), patch.object(adapter, 'build_resident', recording_build_resident):
            audio, rate, receipt = adapter.generate(value, bundle, runtime, dtype_name, stages.append)
        self.assertEqual((audio, rate), ('pcm-fixture', 48000))
        # The three explicit build_resident calls and the construct_vae closure: argument order, inventoried files, shared loader, tied keys.
        files, loader = adapter.component_safetensors(bundle), runtime['resident_loader']
        self.assertEqual([call[:5] for call in calls], [('planner', files['planner'], device, fake.bfloat16, {'lm_head.weight'}),
                                                        ('dit', files['dit'], device, fake.bfloat16, set()),
                                                        ('text_encoder', files['text_encoder'], device, fake.bfloat16, set()),
                                                        ('vae', files['vae'], device, fake.bfloat16, set())])
        self.assertTrue(all(call[5] is loader for call in calls))
        self.assertEqual(seen, [False] * 4)  # Every component constructed under init_empty_weights(include_buffers=False).
        constructed = [event for event in events if isinstance(event, tuple) and event[0] == 'constructed']
        self.assertEqual(constructed, [('constructed', 'Qwen3ForCausalLM', 'sdpa', 'bfloat16'), ('constructed', 'AceStepConditionGenerationModel', 'sdpa', 'bfloat16'),
                                       ('constructed', 'Qwen3Model', 'sdpa', 'bfloat16'), ('constructed', 'AutoencoderOobleck', None, 'float32')])
        self.assertIs(fake.default_dtype, fake.float32)  # The transformers default-dtype hook was restored after each construction.
        self.assertIn(('vae_config', bundle / 'core/vae', True), events)
        self.assertIn(('torch.load', bundle / 'dit/silence_latent.pt', device, True), events)  # The silence tensor lands on the device.
        self.assertIn(('config', 'planner', True), events)
        self.assertEqual(stages, ['loading_planner', 'planning', 'loading_dit', 'loading_text_encoder', 'loading_vae', 'sampling', 'conditioning', 'decoding'])
        # The planner was released (and the device cache reclaimed) before the DiT constructed.
        self.assertLess(events.index('empty_cache'), events.index(constructed[1]))
        self.assertTrue(receipt['planner_released_before_dit_load'])
        # Loader evidence: every inventoried shard streamed, the planner's tied key recorded, the DiT's largest F32 tensor measured.
        self.assertEqual({name: (part['files'], part['parameters'], part['tied_keys']) for name, part in loader['components'].items()},
                         {'planner': (2, 3, ['lm_head.weight']), 'dit': (2, 3, []), 'text_encoder': (1, 1, []), 'vae': (1, 1, [])})
        self.assertEqual((loader['mode'], loader['host_mapping'], loader['files'], loader['tensors'], loader['bytes_read'], loader['largest_tensor_bytes']),
                         ('streamed', 'none', 6, 7, 9892, 48 * 24 * 4))
        self.assertEqual(receipt['placement_check']['loader'], loader)
        self.assertEqual((receipt['placement'], receipt['cpu_offload'], receipt['seed'], receipt['dtype'], receipt['attention']), ('resident', False, 11, 'bfloat16', 'sdpa'))
        # Every component executed on the device at the resolved dtype and was retained there (no offload row anywhere).
        rows = {(row['component'], row['stage'], row['device'], row['dtype']) for row in runtime['parameter_placements']}
        self.assertEqual(rows, {(component, stage, device, 'bfloat16') for component in ('planner', 'dit', 'text_encoder', 'vae') for stage in ('execute', 'retain')})
        return receipt, events

    def test_resident_constructor_mirrors_from_pretrained_config_mutations(self):
        fake = FakeTensorTorch()
        events = []
        modeling_utils = ModuleType('transformers.modeling_utils')

        @contextmanager
        def no_init_weights():
            events.append('no_init_weights')
            yield
            events.append('init_restored')
        modeling_utils.no_init_weights = no_init_weights
        transformers = ModuleType('transformers')
        transformers.modeling_utils = modeling_utils

        class Model:
            def __init__(self, config):
                events.append(('constructed', config._attn_implementation, fake.default_dtype.name, 'no_init_weights' in events))
                self.config = config

            @classmethod
            def _set_default_dtype(cls, dtype):
                previous = fake.default_dtype
                fake.set_default_dtype(dtype)
                events.append(('default_dtype', dtype.name))
                return previous
        original = SimpleNamespace(_attn_implementation=None, dtype='bfloat16', name_or_path='', sub_configs={'text_config'},
                                   text_config=SimpleNamespace(dtype=None))
        construct = adapter.resident_constructor(fake, Model, original, Path('C:/fixture/planner'), 'sdpa', fake.float16)
        self.assertEqual((original._attn_implementation, original.dtype, original.name_or_path, original.text_config.dtype),
                         (None, 'bfloat16', '', None))  # The caller's config is untouched (from_pretrained deep-copies it).
        self.assertEqual(events, [])  # Nothing runs until construction is requested (inside init_empty_weights).
        with patch.dict(sys.modules, {'transformers': transformers, 'transformers.modeling_utils': modeling_utils}):
            model = construct()
        self.assertEqual((model.config._attn_implementation, model.config.dtype, model.config.name_or_path, model.config.text_config.dtype),
                         ('sdpa', fake.float16, str(Path('C:/fixture/planner')), fake.float16))
        self.assertEqual(events, [('default_dtype', 'float16'), 'no_init_weights', ('constructed', 'sdpa', 'float16', True), 'init_restored'])
        self.assertIs(fake.default_dtype, fake.float32)  # Restored after construction.


if __name__ == '__main__':
    unittest.main(verbosity=2)
