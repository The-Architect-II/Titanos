"""Node-local memory admission for llama-cpp runtimes: refuse a start that would not fit, before any process exists.

A GPU start needs the offloaded share of the weights (every GGUF shard and the projector), the KV cache for
native.context_size positions (titanos.gguf_header.kv_cache_bytes, with the registration's cache types) and a compute
margin of COMPUTE_FLOOR_MIB or COMPUTE_SHARE of the rest, whichever is larger. These are Titanos's own estimates, not a
measurement: a start is admitted only when the estimate fits in each selected GPU's free memory (nvidia-smi, per UUID, at
start time) minus TITANOS_ADMISSION_RESERVE_MIB, the memory other owners of the GPU keep. TITANOS_ADMISSION_FREE_MIB,
when nonzero, caps the measured free memory (a test or an operator can lower it, never raise it). A refusal is
AdmissionRefused: 503 insufficient_device_memory with Retry-After, raised before start_process creates a directory, a key
or the runtime process. TITANOS_NATIVE_ADMISSION picks the mode: auto (default) checks when the GGUF header and
nvidia-smi can be read and otherwise admits, recording why; on fails closed when either cannot be read; off never checks.
CPU placements and non-CUDA native devices are admitted without a check (host memory is not measured here).
"""
import csv
import io
import math
import shutil
import subprocess
from pathlib import Path

from .engine import EngineError
from .gguf_header import cached_model_facts, kv_cache_bytes

MIB = 1024 * 1024
COMPUTE_FLOOR_MIB = 512
COMPUTE_SHARE = 0.10
RETRY_AFTER_SECONDS = 30


class AdmissionRefused(EngineError):
    """503 insufficient_device_memory; headers carries Retry-After for the HTTP answer."""

    def __init__(self, message, estimate=None):
        super().__init__(message, 503, 'insufficient_device_memory')
        self.headers = {'Retry-After': str(RETRY_AFTER_SECONDS)}
        self.estimate = estimate


def free_memory_mib():
    """{GPU UUID: free MiB} from nvidia-smi now. Raises ValueError when nvidia-smi is missing or fails."""
    command = shutil.which('nvidia-smi')
    if not command:
        raise ValueError('Memory admission requires nvidia-smi for a CUDA placement')
    try:
        result = subprocess.run([command, '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], check=True,
                                capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('nvidia-smi memory query failed; the start was not admitted') from exc
    rows = {}
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) == 2 and row[0].strip().startswith('GPU-') and row[1].strip().isdigit():
            rows[row[0].strip().lower()] = int(row[1].strip())
    return rows


def estimate(spec, *, files, slot_context=None):
    """The bytes a GPU start of spec needs: {'weights', 'kv_cache', 'compute', 'total', 'offload_fraction', 'context_size',
    'basis'}. files are the GGUF paths the runtime loads (model shards and projector); slot_context overrides the
    registered per-slot context. A header without KV counts adds no KV estimate and says so in basis."""
    native = spec.native
    files = [Path(item) for item in files]
    facts = cached_model_facts(files[0], files=files[1:])
    context = (slot_context or native.context_size // native.slots) * native.slots
    layers = facts['block_count']
    if native.gpu_layers in ('auto', 'all') or not layers:
        fraction = 1.0 if native.gpu_layers != 0 else 0.0
    else:
        fraction = min(1.0, native.gpu_layers / layers)
    kv = kv_cache_bytes(facts, context, native.cache_type_k, native.cache_type_v)
    weights = math.ceil(facts['weight_bytes'] * fraction)
    kv = math.ceil((kv or 0) * fraction)
    compute = max(COMPUTE_FLOOR_MIB * MIB, math.ceil((weights + kv) * COMPUTE_SHARE)) if fraction else 0
    basis = 'gguf weights + kv cache + compute margin' if facts['head_count'] else 'gguf weights + compute margin (no kv counts in header)'
    return {'weights': weights, 'kv_cache': kv, 'compute': compute, 'total': weights + kv + compute,
            'offload_fraction': fraction, 'context_size': context, 'basis': basis}


def shares(devices, split):
    """The fraction of a start each device carries: native.tensor_split proportions, else an even split."""
    if split and len(split) == len(devices):
        total = sum(split)
        return [value / total for value in split]
    return [1 / len(devices)] * len(devices)


def admit(spec, placement, *, files, settings, slot_context=None, released_bytes=0, free=None):
    """Raise AdmissionRefused unless the start fits on every GPU placement resolved; return the decision record otherwise.
    released_bytes is memory the caller frees first on the same devices (the runtime a resize stops). free is the
    {uuid: MiB} reading to use (default: free_memory_mib() now)."""
    mode = settings.native_admission
    resolved = (placement or {}).get('resolved_device') or 'cpu'
    devices = [item for item in resolved.split(',') if item.startswith('GPU-')]
    if mode == 'off' or not devices:
        return {'admitted': True, 'checked': False, 'mode': mode, 'device': resolved}
    try:
        need = estimate(spec, files=files, slot_context=slot_context)
        reading = free_memory_mib() if free is None else {key.lower(): value for key, value in free.items()}
        missing = [device for device in devices if device.lower() not in reading]
        if missing:
            raise ValueError('nvidia-smi does not list ' + ', '.join(missing))
    except (OSError, ValueError) as exc:
        if mode == 'on':
            raise AdmissionRefused(f'Memory admission could not measure this start ({exc}); nothing was started') from exc
        return {'admitted': True, 'checked': False, 'mode': mode, 'device': resolved, 'reason': str(exc)[:300]}
    reserve = settings.admission_reserve_mib * MIB
    decision = {'admitted': True, 'checked': True, 'mode': mode, 'estimate': need,
                'reserve_mib': settings.admission_reserve_mib, 'devices': []}
    for device, share in zip(devices, shares(devices, spec.native.tensor_split)):
        free_bytes = reading[device.lower()] * MIB
        if settings.admission_free_mib:
            free_bytes = min(free_bytes, settings.admission_free_mib * MIB)
        required = math.ceil(need['total'] * share)
        available = free_bytes + math.ceil(released_bytes * share) - reserve
        decision['devices'].append({'uuid': device, 'free_mib': free_bytes // MIB, 'required_mib': math.ceil(required / MIB),
                                    'available_mib': available // MIB})
        if required > available:
            raise AdmissionRefused(
                f'Loading needs about {math.ceil(required / MIB)} MiB on {device} ({need["basis"]}, context '
                f'{need["context_size"]}), but {max(0, available) // MIB} MiB is available after the '
                f'{settings.admission_reserve_mib} MiB reserve for other owners; nothing was started. Retry later, lower '
                'options.num_ctx or the registration\'s context, or unload another model', need)
    return decision


WEIGHT_SUFFIXES = ('.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.onnx', '.onnx_data', '.gguf')
WEIGHTS_FACTOR = 1.2


def weights_on_disk(path):
    """Bytes of the weight files at a registered path: the file itself, else under the folder its .safetensors files when
    it has any (a repository shipping .bin and .safetensors copies is counted once), else its other WEIGHT_SUFFIXES files."""
    path = Path(path).expanduser()
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    sizes = {True: 0, False: 0}
    for item in path.rglob('*'):
        if item.is_file() and item.suffix.lower() in WEIGHT_SUFFIXES:
            sizes[item.suffix.lower() == '.safetensors'] += item.stat().st_size
    return sizes[True] or sizes[False]


def model_need(spec):
    """(method, bytes or None, detail) for one GPU holding spec: the admission estimate for llama-cpp, else the weights on
    disk x WEIGHTS_FACTOR (a heuristic: no activations, KV cache or runtime overhead are measured)."""
    if spec.backend == 'llama-cpp' and spec.native is not None:
        from types import SimpleNamespace
        from .native import runtime_files
        try:
            need = estimate(spec, files=runtime_files(SimpleNamespace(spec=spec)))
        except (OSError, ValueError) as exc:
            return 'gguf_estimate', None, {'reason': str(exc)[:300]}
        return 'gguf_estimate', need['total'], need
    try:
        size = weights_on_disk(spec.path)
    except OSError as exc:
        return 'weights_size_heuristic', None, {'reason': str(exc)[:300]}
    if not size:
        return 'weights_size_heuristic', None, {'reason': 'no weight files found at the registered path'}
    return 'weights_size_heuristic', math.ceil(size * WEIGHTS_FACTOR), {'weights_on_disk': size, 'factor': WEIGHTS_FACTOR}


def candidate_devices(spec, reading, settings):
    """(uuid, share of the estimate) rows memory_view reports: none for cpu or native:DEVICE placements (nvidia-smi does
    not measure them); the registered GPU UUIDs with their native.tensor_split share, as admit() splits; else every listed
    GPU with the whole estimate, without TITANOS_EXCLUDE_GPU_UUIDS for llama-cpp (its automatic placement skips them)."""
    if spec.device == 'cpu' or spec.device.startswith('native:'):
        return []
    pinned = [item.strip() for item in spec.device.split(',') if item.strip().startswith('GPU-')]
    if pinned:
        return list(zip(pinned, shares(pinned, spec.native.tensor_split if spec.native else None)))
    excluded = {item.strip().lower() for item in settings.exclude_gpu_uuids.split(',')} if spec.backend == 'llama-cpp' else set()
    return [(uuid, 1.0) for uuid in reading if uuid not in excluded]


def memory_view(models, settings, free=None):
    """GET /api/models/memory: per registered model the estimate, and per visible NVIDIA GPU its free memory now and
    whether the estimate fits (true, false or 'unknown'), with admit()'s reserve and cap. Never loads a model. free is the
    {uuid: MiB} reading to use (default: free_memory_mib() now; its absence leaves the device list empty)."""
    reading = {}
    source = {'source': 'nvidia-smi', 'available': True}
    try:
        reading = {key.lower(): value for key, value in (free_memory_mib() if free is None else free).items()}
    except ValueError as exc:
        source = {'source': 'nvidia-smi', 'available': False, 'reason': str(exc)[:300]}
    reserve = settings.admission_reserve_mib * MIB
    data = []
    for name, spec in models.items():
        method, need, detail = model_need(spec)
        devices = []
        for uuid, share in (candidate_devices(spec, reading, settings) if source['available'] else []):
            required = None if need is None else math.ceil(need * share)
            if uuid.lower() not in reading:
                devices.append({'uuid': uuid, 'estimate_bytes': required, 'free_bytes': None, 'available_bytes': None,
                                'fits': 'unknown'})
                continue
            free_bytes = reading[uuid.lower()] * MIB
            if settings.admission_free_mib:
                free_bytes = min(free_bytes, settings.admission_free_mib * MIB)
            available = free_bytes - reserve
            devices.append({'uuid': uuid, 'estimate_bytes': required, 'free_bytes': free_bytes, 'available_bytes': available,
                            'fits': 'unknown' if required is None else required <= available})
        data.append({'model': name, 'backend': spec.backend, 'device': spec.device, 'method': method,
                     'estimate_bytes': need, 'detail': detail, 'devices': devices})
    return {'object': 'list', 'data': data, 'free_memory': source, 'reserve_mib': settings.admission_reserve_mib}


__all__ = ['AdmissionRefused', 'admit', 'estimate', 'free_memory_mib', 'memory_view', 'model_need', 'shares']
