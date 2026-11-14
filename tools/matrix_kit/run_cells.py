"""Matrix kit runner: execute every runnable model on every platform slot this host covers, in every execution mode the host
can offer, one server process per cell, and write receipts in execution-matrix cell shape.

    python tools/matrix_kit/run_cells.py --manifest <manifest.json> --out <receipts dir> [--slots auto|id,id] [--modes single,replicas,model_sharding,tensor_parallel]
        [--allowed-gpus GPU-...,GPU-...] [--min-available-commit-gib 4] [--min-gpu-free-mib 4096] [--stop-file PATH] [--only model_id,...]
        [--host-reserve-gib G] [--admission-profiles PATH]

Cell states written here:
  pass   server started, model loaded on the named device, one real request returned a verified output, the model was listed
         as loaded, it unloaded, and the server exited cleanly
  fail   any step above failed at runtime (error text kept, server log tail kept)
  na     not applicable by construction: the backend has no execution path on the slot runtime (a converted artifact would be its
         own catalog row), Titanos refused the configuration by design (for example tensor parallel for a task that uses one device
         per worker), or a multi-GPU mode on a slot whose runtime cannot form a GPU group; the reason is the note
  untried (with reason) the model artifact is absent on this host, the runtime package the backend imports (torch, onnxruntime,
         openvino) is not installed on this host (the slots.py probe; an import error is not an execution), the host-resource
         admission floor never cleared before STOP, a required commit/GPU reading stayed missing (admission fails closed), a multi-GPU
         mode needs a group this GPU slot (or an explicit CPU/XPU tensor-parallel group) does not offer in this run, a guard tripped
         while the cell was active (STOP or a budget: '<stopped|budget_violated>_during_<phase>'), or the cell's resource
         qualification is exceeded or unknown (never a pass plus a note, even when the exercise wrote a valid output)

Admission and containment (containment.py). The default floor is the artifact floor: max(global floor, 1.5 x artifact bytes x copies
+ margin) of host commit and, on a GPU, artifact bytes x copies + margin of free device memory; no manifest field lowers it. The only
lower host floor is contained admission ('contained_resident'): the entry's spec names placement 'resident' explicitly ('auto' may
fall back to CPU offload and is never admitted under a lowered floor), the slot has a GPU, and the host can cap the owned process
tree (a Windows Job Object). The server is then created suspended inside a kill-on-close job whose commit cap is set and read back
before the child resumes, a shared-host reserve (--host-reserve-gib) stays free, and the cap is 1.5 x a measured within-bound peak
+ margin (--admission-profiles, used only when the entry is bound to an existing within-bound pass receipt) or, on a first run, what
the host can spare minus the reserve. The cap is commit-only (JOB_OBJECT_LIMIT_JOB_MEMORY; receipt bound 'job_commit') and is a
fail-closed bound only for the allocation classes containment.CAP_SEMANTICS lists as enforced (heap/VirtualAlloc, CUDA allocations
under WDDM, pinned memory, safetensors safe_open, FILE_MAP_COPY views); copy-on-write mappings made by torch's from_file are
accounted in the job peak but not refused at map time (pages fail at touch with the same exit code r3 showed, 0xC0000005; r3's own
cause is unproven), and anonymous pagefile-backed
sections are neither accounted nor refused. Every contained receipt records that table as resource_qualification.cap_semantics,
available physical memory is sampled beside commit and a contained cell trips when it drops under the reserve. Before a contained
cell launches, the WDDM-aware pre-launch check (device_resident_check) reads the GPU driver model (nvidia-smi driver_model.current)
and sizes the entry's safetensors headers at the requested dtype (numel x itemsize per component; never tensor data): on WDDM,
where every device allocation is charged to the process commit, the cap must cover BASE_RESIDENT_BYTES (torch import + a CUDA
context, measured 2.7 GB) + the peak resident set (max(planner, dit) + text encoder + vae) + ACTIVATION_HEADROOM_BYTES, else the
cell is untried 'blocked_by_host_resources: contained budget below the device-resident need on WDDM (cap X < base + peak set +
headroom Y)' with the numbers in resource_qualification.device_resident_estimate; on TCC the estimate is recorded only; an
unreadable driver model or unreadable headers fail closed. A CellGuard samples host commit, physical memory, GPU memory and
the job peak at least every 2 s during server start, model load, every active request and the unload; a trip cancels only owned
work and keeps the partial receipt. The GPU-starvation trip (free below half the margin) applies to contained cells only. Each
attempt separates 'exercise' (output correctness) from 'resource_qualification' (the enforced bound and what was measured); the
cell is a pass only when both hold. Every receipt carries 'receipt_schema' = RECEIPT_SCHEMA (2) with the complete measured block
(cap, readback, peaks, samples, identity, cleanup) as its top-level 'resource_qualification', and RUN_RESULT carries 'kit_schema' =
RECEIPT_SCHEMA; ingest.py requires exactly that detail before a current contained cell can enter as pass or fail, and binds receipts
of older kits (no marker, before its legacy boundary) exactly as before. Admission and
the guard fail closed when the commit reading is None: on macOS/Linux hosts without /proc/meminfo that reading needs psutil in the
kit environment, otherwise every cell is recorded untried 'resource_reading_unavailable'.

The runner never writes 'partial' and never marks a cell verified: a verifier upholds a pass against the receipt, and the
maintainer binds the evidence id. Everything is bounded by resource floors and the STOP file, not by wall-clock windows.
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))  # lab_paths: the generated machine-name class a run id or host label must never carry
import lab_paths  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

import containment  # noqa: E402
import requests_catalog  # noqa: E402
import slots as slotmod  # noqa: E402
from titanos.config import Config, ModelSpec, WorkerSpec, write_config  # noqa: E402

# Which model backends have an execution path on which slot runtime. A pair outside this table cannot exist by construction (a
# converted artifact for that runtime is its own catalog row) and is recorded as not-applicable with the reason, never as a failure.
# Mirrored by docs/execution_matrix_spec.json backend_runtime_map, which the builder applies to cells no reader executed.
# OpenVINO reads model.onnx directly, so the onnx backend has a path on the OpenVINO GPU and NPU runtimes (uci-digits-8x8-onnx
# executed exactly that: E-002/E-026/E-028 on the Intel NPU, E-002/E-037 on Arc GPU.0); Titanos's own refusal of a particular
# model there (a CPU-only adapter such as VAD) is recorded from its refusal text at run time.
COMPAT = {
    'cpu': {'transformers', 'diffusers', 'xtts', 'qwen3-tts', 'ace-step', 'llama-cpp', 'onnx', 'sherpa-onnx', 'tesseract', 'openvino', 'openvino-genai', 'onnx-diffusers'},
    'torch-cuda': {'transformers', 'diffusers', 'xtts', 'qwen3-tts', 'ace-step', 'llama-cpp'},
    'torch-rocm': {'transformers', 'diffusers', 'llama-cpp'},
    'torch-mps': {'transformers', 'diffusers', 'llama-cpp'},
    'torch-xpu': {'transformers', 'diffusers'},
    'openvino-gpu': {'openvino', 'openvino-genai', 'onnx'},
    'openvino-npu': {'openvino', 'openvino-genai', 'onnx'},
    'onnx-dml': {'onnx', 'onnx-diffusers'},
    'onnx-qnn': {'onnx'},
    'onnx-coreml': {'onnx'},
    'onnx-vitisai': {'onnx'},
    # a SYCL llama.cpp build on an Intel GPU (slots.py probe_native_sycl): Titanos's native runner takes the device as native:SYCLn
    'native-sycl': {'llama-cpp'},
}
OOM = re.compile(r'not enough memory|out of memory|DefaultCPUAllocator|CUDA error: out of memory|cudaErrorMemoryAllocation|OutOfMemoryError|paging file is too small|MemoryError', re.I)
UNSUPPORTED = re.compile(r'requires|one device per worker|not implemented|does not support|unsupported|needs two|homogeneous|identical|must be|only supports|cannot|Tensor parallelism|Model sharding|use replicas|CPU-only|cpu only', re.I)
MULTI_GPU_MODES = ('model_sharding', 'tensor_parallel')
# Schema generation of everything this kit writes: every cell receipt gets 'receipt_schema' and RUN_RESULT 'kit_schema' = RECEIPT_SCHEMA.
# ingest.py treats a RUN_RESULT with kit_schema >= 2 as current (the detailed receipt is the only pass/fail eligibility evidence) and
# results without the marker as legacy only before its explicit boundary timestamp.
RECEIPT_SCHEMA = 2
# Slot runtimes whose devices can form a homogeneous GPU group (slots.py builds groups from torch CUDA/ROCm devices only), and the
# runtimes on which Titanos accepts an explicit rank group without GPUs (a CPU Gloo or XPU XCCL tensor-parallel group, docs/MULTIGPU.md).
GROUP_RUNTIMES = {'torch-cuda', 'torch-rocm'}
EXPLICIT_GROUP_RUNTIMES = {'model_sharding': set(), 'tensor_parallel': {'cpu', 'torch-xpu'}}
NA_PREFIX = 'not applicable by construction: '
# Titanos's own admission of these multi-GPU modes (this kit sets ModelSpec.parallelism = mode and WorkerSpec.mode = mode), from
# src/titanos/config.py: 237-238 xtts/qwen3-tts use one device per worker; 211-212 and 336-337 the API Plus tasks use single-device
# workers (so do diarization, 74-77 through diarization.py:33-38 and 334-335, and vad, 170-172; SINGLE_DEVICE_TASK_LINES cites each);
# 233-234 and 345-346 tensor parallelism exists only for Transformers chat/vision/embeddings; 347-348 model sharding requires
# Transformers or Diffusers (the compiled, native, sherpa and tesseract adapters refuse parallelism other than single: 80-82, 97-100,
# 150-151, 171-172, 190-191, onnx_images.py:19, tesseract.py:57). Mirrored by the spec's multi_gpu_modes_rule.admission.
ONE_DEVICE_BACKENDS = {'xtts', 'qwen3-tts'}
SINGLE_DEVICE_TASKS = {'translation', 'image-to-text', 'object-detection', 'sentiment', 'rerank', 'pair-classification', 'depth-estimation', 'image-segmentation', 'music', 'diarization', 'vad'}
SINGLE_DEVICE_TASK_LINES = {'translation': 'src/titanos/config.py:211-212, 336-337', 'image-to-text': 'src/titanos/config.py:211-212, 336-337',
                            'object-detection': 'src/titanos/config.py:211-212, 336-337', 'sentiment': 'src/titanos/config.py:211-212, 336-337',
                            'rerank': 'src/titanos/config.py:211-212, 336-337', 'pair-classification': 'src/titanos/config.py:211-212',
                            'depth-estimation': 'src/titanos/config.py:211-212, 336-337', 'image-segmentation': 'src/titanos/config.py:117-118, 336-337',
                            'music': 'src/titanos/config.py:80-82, 336-337', 'diarization': 'src/titanos/config.py:74-77 (diarization.py:33-38), 334-335',
                            'vad': 'src/titanos/config.py:170-172'}
TENSOR_PARALLEL_BACKENDS, TENSOR_PARALLEL_TASKS = {'transformers'}, {'chat', 'vision', 'embeddings'}
MODEL_SHARDING_BACKENDS = {'transformers', 'diffusers'}


def mode_refusal(backend, task, mode):
    """Titanos's own refusal of a multi-GPU mode for this backend and task (config.py lines above), or None when it admits the mode."""
    if backend in ONE_DEVICE_BACKENDS:
        return f'Titanos refuses {mode} for backend {backend} by design: one device per worker (src/titanos/config.py:237-238)'
    if task in SINGLE_DEVICE_TASKS:
        return f'Titanos refuses {mode} for task {task} by design: single-device workers only ({SINGLE_DEVICE_TASK_LINES[task]})'
    if mode == 'tensor_parallel' and not (backend in TENSOR_PARALLEL_BACKENDS and task in TENSOR_PARALLEL_TASKS):
        return f'Titanos refuses tensor_parallel for backend {backend} task {task} by design: native tensor parallelism is implemented only for Transformers chat, vision and embeddings (src/titanos/config.py:233-234, 345-346)'
    if mode == 'model_sharding' and backend not in MODEL_SHARDING_BACKENDS:
        return f'Titanos refuses model_sharding for backend {backend} by design: model sharding requires Transformers or Diffusers (src/titanos/config.py:347-348)'
    return None


def no_path_cell(entry, s, mode, refusal=None):
    """Cell for a backend outside COMPAT on this slot runtime: not applicable by construction (Titanos's own refusal text, when the
    config probe produced one, is kept as the more specific reason)."""
    if refusal and UNSUPPORTED.search(refusal):
        note = f"not applicable: Titanos refuses backend {entry['backend']} on this device by design: {refusal[:300]}"
    else:
        note = f"{NA_PREFIX}backend {entry['backend']} has no execution path on slot runtime {s['runtime']}; a converted artifact for that runtime would be its own catalog row"
    return {'model_id': entry['model_id'], 'slot': s['slot'], 'mode': mode, 'state': 'na', 'verified': False, 'evidence_ids': [], 'device_named': s['device_named'], 'note': note}


# The Python runtimes each backend imports on the CPU slot (all of them must be present), keyed by the slots.py probe that reports
# whether each is installed (GPU and accelerator slots exist only when their probe succeeded). onnx-diffusers needs onnxruntime and
# torch (src/titanos/onnx_images.py:123 imports the diffusers ONNX pipeline classes, :193 Torch CPU PNDM scheduling), mirroring the
# spec's all-of cpu_path_tokens. A backend any of whose runtimes the probe found absent is recorded as untried with that reason:
# an ImportError is not an execution, so such a cell is never a fail, and it never contradicts the spec's runtime_availability
# facts (which describe published wheels, not this host). sherpa-onnx, tesseract and llama-cpp bring their own runtimes, which the
# kit does not probe.
RUNTIME_PROBE = {'transformers': ('torch',), 'diffusers': ('torch',), 'xtts': ('torch',), 'qwen3-tts': ('torch',), 'ace-step': ('torch',),
                 'onnx': ('onnx',), 'onnx-diffusers': ('onnx', 'torch'), 'openvino': ('openvino',), 'openvino-genai': ('openvino',)}
PROBE_PACKAGE = {'torch': 'torch', 'onnx': 'onnxruntime', 'openvino': 'openvino'}


def missing_runtime_cell(entry, s, mode, probe_name, probe):
    """Cell for a backend whose runtime package the host probe found absent: untried with the reason, never fail."""
    note = (f"not run: runtime {PROBE_PACKAGE[probe_name]} for backend {entry['backend']} is not installed on this host "
            f"({probe_name} probe: {str(probe.get('error') or 'not available')[:200]}); an import error is not an execution")
    return {'model_id': entry['model_id'], 'slot': s['slot'], 'mode': mode, 'state': 'untried', 'verified': False, 'evidence_ids': [], 'device_named': s['device_named'], 'note': note}


def missing_group_cell(entry, s, mode):
    """Cell for a multi-GPU mode when this host offers no homogeneous group on the slot: na when Titanos refuses the mode for this
    backend/task by design (mode_refusal), na when the slot runtime cannot form a group by construction (a CPU, DirectML, OpenVINO,
    QNN, Core ML, VitisAI or MPS slot), untried on a GPU slot whose group is simply absent from this run, and untried on a CPU/XPU
    slot for tensor parallel because Titanos accepts an explicit rank group there."""
    refusal = mode_refusal(entry['backend'], entry.get('task'), mode)
    if refusal:
        note, state = NA_PREFIX + refusal, 'na'
    elif s['runtime'] in GROUP_RUNTIMES:
        note = f'not run: {mode} needs a homogeneous multi-GPU group on this slot; none in the allowed set'
        state = 'untried'
    elif s['runtime'] in EXPLICIT_GROUP_RUNTIMES[mode]:
        note = f"not run: {mode} over an explicit {s['runtime']} rank group (docs/MULTIGPU.md) is accepted by Titanos but not a mode this kit runs; no GPU group on this slot"
        state = 'untried'
    else:
        note = f"{NA_PREFIX}{mode} needs a homogeneous multi-GPU group and slot runtime {s['runtime']} cannot form one"
        state = 'na'
    return {'model_id': entry['model_id'], 'slot': s['slot'], 'mode': mode, 'state': state, 'verified': False, 'evidence_ids': [], 'device_named': s['device_named'], 'note': note}


def utc():
    return datetime.now(timezone.utc).isoformat()


def host_memory():
    """{'available_commit_bytes', 'available_physical_bytes'}, either None when unreadable. Windows: GlobalMemoryStatusEx ullAvailPageFile
    (what a new process may still commit) and ullAvailPhys (RAM not in use; file-backed mappings the commit cap does not see live
    here). Elsewhere: psutil's available figure for both (MemAvailable from /proc/meminfo without psutil), so a macOS host without
    psutil reads None and the fail-closed admission records its cells untried 'resource_reading_unavailable'."""
    if os.name == 'nt':
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        s = MEMORYSTATUSEX()
        s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return {'available_commit_bytes': None, 'available_physical_bytes': None}
        return {'available_commit_bytes': int(s.ullAvailPageFile), 'available_physical_bytes': int(s.ullAvailPhys)}
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
        return {'available_commit_bytes': available, 'available_physical_bytes': available}
    except Exception:  # noqa: BLE001
        try:
            for line in open('/proc/meminfo', encoding='utf-8'):
                if line.startswith('MemAvailable:'):
                    available = int(line.split()[1]) * 1024
                    return {'available_commit_bytes': available, 'available_physical_bytes': available}
        except OSError:
            pass
    return {'available_commit_bytes': None, 'available_physical_bytes': None}


def available_commit_bytes():
    return host_memory()['available_commit_bytes']


def available_physical_bytes():
    return host_memory()['available_physical_bytes']


def gpu_free_mib(uuid):
    if not uuid or not uuid.startswith('GPU-'):
        return None
    try:
        r = subprocess.run(['nvidia-smi', f'--id={uuid}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def gpu_used_mib(uuid):
    if not uuid or not uuid.startswith('GPU-'):
        return None
    try:
        r = subprocess.run(['nvidia-smi', f'--id={uuid}', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def gpu_driver_model(uuid, runner=subprocess.run):
    """'WDDM' | 'TCC' (nvidia-smi --query-gpu=driver_model.current for the exact GPU) or None when unreadable (no nvidia-smi, a
    non-NVIDIA device string, 'N/A' on Linux, a timeout). None is never treated as TCC: the pre-launch check fails closed on it."""
    if not uuid or not uuid.startswith('GPU-'):
        return None
    try:
        r = runner(['nvidia-smi', f'--id={uuid}', '--query-gpu=driver_model.current', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        value = (r.stdout or '').strip().splitlines()[0].strip().upper()
        return value if value in ('WDDM', 'TCC') else None
    except Exception:  # noqa: BLE001
        return None


# WDDM-aware device-resident sizing for contained cells. On a WDDM host every CUDA allocation is charged 1:1 to the owning process's
# commit (RUN_R3_RESULT_AND_WDDM_2149Z.json wddm_commit_probe: torch import 1.70 GB, CUDA context +0.98 GB, a 4 GiB device tensor +4.3 GB
# of commit, returned on free), so the job commit cap must cover the device-resident weights themselves besides the host footprint.
BASE_RESIDENT_BYTES = 3 << 30  # torch import + a CUDA context: 2.7 GB of commit measured on 2026-09-25, rounded up to 3 GiB
# Activations, KV cache and allocator slack beside the resident weights (the adapter's PLACEMENT_HEADROOM_BYTES). The same 4 GiB also
# absorbs the streamed loader's one-tensor transient: one host buffer plus the device copy at the stored dtype and, when the dtype
# differs, the device cast copy (on WDDM both device copies are commit) - for the pinned bundle at bf16 at most about 2.2 GB (the planner
# embedding, 1,112,084,480 B BF16, host + device; the DiT's F32 tensors are far smaller, 157,286,400 B stored, plus their bf16 cast).
# That transient peaks while a component loads, before any activation or KV cache
# exists, so it fits inside the headroom the activations use later and is not a separate term of the estimate.
ACTIVATION_HEADROOM_BYTES = 4 << 30
# The four components the resident music adapter loads (tools/music_runtime/ace_adapter.py PLACEMENT_COMPONENTS), relative to the bundle
# root; the planner is released before the DiT loads, so the peak resident set is max(planner, dit) + text encoder + vae.
RESIDENT_COMPONENTS = {'planner': 'planner', 'dit': 'dit', 'text_encoder': 'core/Qwen3-Embedding-0.6B', 'vae': 'core/vae'}
RESIDENT_ITEMSIZE = {'bfloat16': 2, 'float16': 2, 'float32': 4}  # bytes per parameter at the requested dtype (the adapter's PLACEMENT_ITEMSIZE)
SAFETENSORS_HEADER_LIMIT = 64 << 20


def safetensors_header_numel(path):
    """Parameter count from a safetensors header (an 8-byte little-endian header length, then the JSON tensor table); only the header is
    read, tensor data is never touched. Ported from the adapter's safetensors_numel: numel is counted, not file bytes, because the DiT
    is stored F32 and loads at the requested dtype. Raises ValueError for a malformed header."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f'truncated safetensors header: {path.name}')
        length = int.from_bytes(prefix, 'little')
        if not 0 < length <= SAFETENSORS_HEADER_LIMIT or 8 + length > size:
            raise ValueError(f'invalid safetensors header length {length}: {path.name}')
        header = stream.read(length)
    if len(header) != length:
        raise ValueError(f'truncated safetensors header: {path.name}')
    try:
        table = json.loads(header.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f'unreadable safetensors header ({exc}): {path.name}') from exc
    if not isinstance(table, dict) or not any(name != '__metadata__' for name in table):
        raise ValueError(f'empty safetensors table: {path.name}')
    total = 0
    for name, entry in table.items():
        if name == '__metadata__':
            continue
        shape = entry.get('shape') if isinstance(entry, dict) else None
        if not (isinstance(shape, list) and all(type(value) is int and value >= 0 for value in shape) and isinstance(entry.get('dtype'), str)):
            raise ValueError(f'invalid safetensors tensor entry {name!r}: {path.name}')
        numel = 1
        for value in shape:
            numel *= value
        total += numel
    return total


def resident_itemsize(dtype, slot):
    """Bytes per parameter the resident load will use: the requested dtype's, or for 'auto' what the adapter resolves on this device
    (bfloat16 when the CUDA device supports it, compute capability >= 8.0, else float32). An unknown capability sizes at float32
    (the larger, conservative estimate). Returns (itemsize, resolved dtype name)."""
    if dtype in RESIDENT_ITEMSIZE:
        return RESIDENT_ITEMSIZE[dtype], dtype
    try:
        cap = float(((slot or {}).get('gpu') or {}).get('capability') or 0)
    except (TypeError, ValueError):
        cap = 0
    resolved = 'bfloat16' if cap >= 8.0 else 'float32'
    return RESIDENT_ITEMSIZE[resolved], resolved


def bundle_component_bytes(root, itemsize):
    """Per-component sizing of the safetensors under an artifact root at `itemsize` bytes per parameter, from headers alone. With the
    resident music bundle layout (RESIDENT_COMPONENTS present) each component is sized separately; otherwise every safetensors under
    the root is one component 'weights' (peak set = total). Returns {'layout', 'components': {name: {'files', 'numel', 'bytes'}},
    'problems': [...]}; a component without a safetensors file or with a malformed header is a problem (the estimate is unusable)."""
    root = Path(root)
    components, problems = {}, []
    if all((root / rel).is_dir() for rel in RESIDENT_COMPONENTS.values()):
        layout, groups = 'resident-music-bundle', {name: root / rel for name, rel in RESIDENT_COMPONENTS.items()}
    else:
        layout, groups = 'flat', {'weights': root}
    for name, folder in groups.items():
        files = sorted(p for p in folder.rglob('*.safetensors') if p.is_file()) if folder.is_dir() else []
        numel = 0
        for p in files:
            try:
                numel += safetensors_header_numel(p)
            except (OSError, ValueError) as exc:
                problems.append(f'{name}: {exc}')
        if not files:
            problems.append(f'{name}: no safetensors under {folder}')
        components[name] = {'files': len(files), 'numel': numel, 'bytes': numel * int(itemsize)}
    return {'layout': layout, 'components': components, 'problems': problems}


def device_resident_estimate(components, job_limit, driver_model, base_bytes=BASE_RESIDENT_BYTES, headroom_bytes=ACTIVATION_HEADROOM_BYTES):
    """Pure rule of the WDDM-aware pre-launch check. `components` maps component name to bytes at the requested dtype (from
    bundle_component_bytes). peak_set_bytes = max(planner, dit) + text_encoder + vae for the resident music layout, else the total;
    required_bytes = base + peak set + headroom. On 'WDDM' (device allocations are commit) the cap must be at least required_bytes or
    the cell is blocked with the reason; on 'TCC' the estimate is recorded and nothing is blocked; any other driver model fails closed
    (blocked, resource_reading_unavailable). Never reads the host."""
    sizes = {name: int(value) for name, value in (components or {}).items()}
    total = sum(sizes.values())
    if set(RESIDENT_COMPONENTS) <= set(sizes):
        peak = max(sizes['planner'], sizes['dit']) + sizes['text_encoder'] + sizes['vae']
        formula = 'max(planner, dit) + text_encoder + vae'
    else:
        peak, formula = total, 'total of every component (no released-before-next component known)'
    required = int(base_bytes) + peak + int(headroom_bytes)
    est = {'driver_model': driver_model, 'components_bytes': sizes, 'total_resident_bytes': total, 'peak_set_bytes': peak, 'peak_set_formula': formula,
           'base_bytes': int(base_bytes), 'activation_headroom_bytes': int(headroom_bytes), 'required_bytes': required, 'job_commit_limit_bytes': job_limit,
           'fits': None, 'enforcement': None, 'blocked': False, 'reason': None}
    if driver_model == 'WDDM':
        fits = job_limit is not None and int(job_limit) >= required
        est.update(fits=fits, enforcement='wddm_commit_covers_device_resident_bytes', blocked=not fits)
        if not fits:
            est['reason'] = (f'blocked_by_host_resources: contained budget below the device-resident need on WDDM (cap {job_limit} < base + peak set + headroom '
                             f'{required}: base {int(base_bytes)} + peak set {peak} [{formula}] + headroom {int(headroom_bytes)}; device allocations are charged to the process commit on WDDM)')
    elif driver_model == 'TCC':
        est.update(fits=(job_limit is not None and int(job_limit) >= required), enforcement='estimate_only_tcc', blocked=False)
    else:
        est.update(blocked=True, enforcement='unknown_driver_model',
                   reason=f'resource_reading_unavailable: GPU driver model {driver_model!r} is neither WDDM nor TCC (nvidia-smi driver_model.current); the WDDM device-resident check fails closed')
    return est


def device_resident_check(entry, slot, dtype, job_limit, gpu_uuid, driver_model_reader=gpu_driver_model):
    """The pre-launch check of one contained_resident attempt: driver model of the exact GPU, the entry's safetensors headers sized at the
    dtype the resident load will use, then device_resident_estimate. Unreadable headers fail closed (blocked, resource_reading_unavailable)
    whatever the driver model; the record is written into resource_qualification.device_resident_estimate either way."""
    driver_model = driver_model_reader(gpu_uuid)
    itemsize, resolved = resident_itemsize(dtype, slot)
    sizing = bundle_component_bytes(entry.get('path') or '', itemsize)
    est = device_resident_estimate({name: c['bytes'] for name, c in sizing['components'].items()}, job_limit, driver_model)
    est.update(requested_dtype=dtype, sized_dtype=resolved, itemsize=itemsize, layout=sizing['layout'], components=sizing['components'], header_problems=sizing['problems'], path=entry.get('path'))
    if sizing['problems']:
        est.update(blocked=True, fits=None, enforcement='headers_unreadable',
                   reason='resource_reading_unavailable: device-resident estimate unavailable (' + '; '.join(sizing['problems'])[:600] + '); the WDDM device-resident check fails closed')
    return est


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def build_config(entry, slot, mode, gpus, dtype):
    """Return (Config, description) or raise ValueError when Titanos refuses the configuration by design."""
    name = entry['task']
    spec_fields = dict(entry.get('spec') or {})
    spec_fields.pop('path', None)
    spec_fields.pop('task', None)
    spec_fields.pop('backend', None)
    spec_fields.setdefault('dtype', dtype)
    if slot['runtime'] == 'cpu' and spec_fields.get('dtype') in ('float16', 'bfloat16') and entry['backend'] in ('transformers', 'diffusers'):
        spec_fields['dtype'] = 'auto'  # a GPU-registered half-precision spec placed on the CPU slot loads in the CPU-native precision
    if entry['task'] in ('image', 'video') and entry['backend'] == 'diffusers':
        # Pickle weights are allowed only when the artifact has no safetensors at all (the tiny test pipelines); a safetensors
        # checkpoint must load with allow_pickle false, which some families (instruction editing) also require by design.
        has_safetensors = any(Path(entry['path']).rglob('*.safetensors')) if Path(entry['path']).is_dir() else False
        spec_fields.setdefault('allow_pickle', entry['task'] == 'image' and not has_safetensors)
        if entry['task'] == 'image':
            spec_fields.setdefault('load_options', {'safety_checker': None})
        if entry['task'] == 'video':
            spec_fields.setdefault('defaults', {'max_sequence_length': 16})
    device = slot['device']
    if entry['backend'] == 'llama-cpp' and device == 'cpu' and isinstance(spec_fields.get('native'), dict):
        spec_fields['native'] = dict(spec_fields['native'], gpu_layers=0)  # a GPU-registered GGUF placed on the CPU slot must offload nothing
    if entry['backend'] == 'llama-cpp' and slot['runtime'] == 'native-sycl' and isinstance(spec_fields.get('native'), dict):
        # the slot's SYCL build serves the row on the named native device with full offload; Titanos verifies the address against the
        # binary's own --list-devices and refuses when it is absent (src/titanos/native.py:101-103)
        native = dict(spec_fields['native'], executable=slot['executable'])
        if native.get('gpu_layers') in (0, '0', None):
            native['gpu_layers'] = 'all'
        spec_fields['native'] = native
    if entry['backend'] == 'onnx-diffusers' and str(device).startswith('dml:'):
        # ONNX image pipelines take the plain 'dml' device and pin the adapter through load_options (device_id, adapter_luid).
        options = dict(spec_fields.get('load_options') or {}, device_id=int(str(device).split(':', 1)[1]))
        luid = (slot.get('adapter') or {}).get('luid')
        if luid:
            options['adapter_luid'] = str(luid)
        spec_fields['load_options'] = options
        device = 'dml'
    backend = entry['backend']
    served = {}
    if backend == 'llama-cpp' and slot['runtime'] == 'torch-mps' and isinstance(spec_fields.get('native'), dict):
        # Titanos's native runner resolves non-CUDA devices only from the binary's own --list-devices table (src/titanos/native.py
        # native_devices / resolve_native_placement), which on Apple silicon names MTL0, never `mps`; `auto` is Titanos's declared
        # placement (no CUDA mask -> the first listed native device) and the receipt records resolved_device native:MTL0.
        device = 'auto'
        served = {'served_by': "llama.cpp Metal build placed with device 'auto' (Titanos resolves the first --list-devices entry, MTL0)"}
    if backend == 'onnx' and slot['runtime'] in ('openvino-gpu', 'openvino-npu'):
        # An ONNX graph on an OpenVINO slot is served by Titanos's OpenVINO backend, which reads .onnx graphs directly
        # (src/titanos/compiled.py graph_path + core.read_model); the onnxruntime backend has no OpenVINO provider. Same
        # artifact, no conversion: the row stays the ONNX row (E-026/E-037 executed exactly this on Intel NPU/GPU).
        backend = 'openvino'
        spec_fields['graph'] = spec_fields.get('graph') or 'model.onnx'
        served = {'served_by': f"openvino backend reading the ONNX graph {spec_fields['graph']}"}
    if mode == 'single':
        spec = ModelSpec(path=entry['path'], task=entry['task'], backend=backend, device=device, **spec_fields)
        return Config(models={name: spec}), {'device': device, **served}
    if mode == 'replicas':
        if device.startswith('GPU-') or device.startswith('HIP-'):
            workers = {'r1': WorkerSpec(models=[name], gpus=[device]), 'r2': WorkerSpec(models=[name], gpus=[device])}
        else:
            workers = {'r1': WorkerSpec(models=[name], device=device), 'r2': WorkerSpec(models=[name], device=device)}
        spec = ModelSpec(path=entry['path'], task=entry['task'], backend=backend, device='auto' if device.startswith(('GPU-', 'HIP-')) else device, **spec_fields)
        return Config(models={name: spec}, workers=workers), {'device': device, 'replicas': 2, **served}
    if mode in MULTI_GPU_MODES:
        if not gpus or len(gpus) < 2:
            raise ValueError(f'{mode} requires a homogeneous GPU group of at least two; this host offers none in the allowed set')
        kwargs = {'mode': mode, 'gpus': list(gpus)}
        if mode == 'tensor_parallel':
            kwargs['tensor_parallel_size'] = len(gpus)
            spec_fields['parallelism'] = 'tensor_parallel'
            spec_fields['tensor_parallel_size'] = len(gpus)
        else:
            spec_fields['parallelism'] = 'model_sharding'
        spec = ModelSpec(path=entry['path'], task=entry['task'], backend=backend, device='auto', **spec_fields)
        return Config(models={name: spec}, workers={'group': WorkerSpec(models=[name], **kwargs)}), {'gpus': list(gpus), 'mode': mode, **served}
    raise ValueError(f'unknown mode {mode}')


def place_home_files(entry, root):
    """Copy the source entry's `home_files` ({name: absolute source path}) into the cell's TITANOS_HOME (the work root) before the
    server starts: Titanos reads installed receipts such as music-runtime.json from home() (src/titanos/config.py:353,
    src/titanos/music_backend.py runtime_receipt). Plain copies with their sha256 recorded; a missing source is a harness error."""
    placed = {}
    for name, source in (entry.get('home_files') or {}).items():
        if '/' in name or '\\' in name or ':' in name or name in ('', '.', '..') or Path(name).name != name or Path(name).is_absolute():
            raise RuntimeError(f'home_files name must be a bare file name: {name!r}')
        src = Path(source)
        if not src.is_file():
            raise RuntimeError(f'home_files source missing for {name}: {source}')
        data = src.read_bytes()
        (root / name).write_bytes(data)
        placed[name] = {'source': str(src), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return placed


def cancel_owned_work(port, owned_work, request_timeout=15):
    """Cancel only the owned Titanos work a tripped guard left behind: the music job the request submitted (DELETE /api/jobs/<id>,
    src/titanos/server.py video_job_cancel) on our own server. Best effort; the evidence is returned, never raised."""
    evidence = {}
    job_id = (owned_work or {}).get('music_job')
    if job_id:
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=request_timeout, trust_env=False) as c:
                r = c.delete(f'/api/jobs/{job_id}')
                evidence['music_job_cancel'] = {'job': job_id, 'http': r.status_code, 'body': r.text[:300]}
        except Exception as exc:  # noqa: BLE001
            evidence['music_job_cancel'] = {'job': job_id, 'error': f'{type(exc).__name__}: {exc}'}
    return evidence


IDENTITY_FIELDS = ('receipt_sha256', 'adapter_sha256', 'placement', 'dtype', 'copies')


def plan_identity(plan):
    """The placement-identity fields of a contained admission plan (its profile-key inputs); None each for another kind."""
    inputs = plan.get('inputs') or {}
    return {field: inputs.get(field) for field in IDENTITY_FIELDS}


def missing_identity(qual):
    """The placement-identity fields a contained_resident qualification lacks: a profile_key that is a real sha256 hex digest (64 lowercase
    hex characters, containment.profile_key), the music-runtime receipt and adapter shas, placement 'resident', a dtype and an integer
    copies >= 1. [] when the measured bound is bound to a complete identity."""
    missing = []
    key = qual.get('profile_key')
    if not (isinstance(key, str) and re.fullmatch(r'[0-9a-f]{64}', key)):
        missing.append('profile_key')
    for field in IDENTITY_FIELDS:
        value = qual.get(field)
        if field == 'placement':
            ok = value == 'resident'
        elif field == 'copies':
            ok = type(value) is int and value >= 1
        else:
            ok = isinstance(value, str) and bool(value.strip())
        if not ok:
            missing.append(field)
    return missing


def final_attempt_state(exercise, qual):
    """Output correctness and resource qualification are separate records; the attempt is a pass only when the exercise passed
    AND the verdict is 'within' (contained) or 'not_contained' (artifact floor). 'exceeded' or 'unknown' leaves the attempt untried
    with the qualification reason even when the exercise passed and wrote a valid output (the exercise record is kept beside it).
    A contained 'within' verdict counts only when the attempt carries the placement identity its bound is measured for (profile_key
    and the profile-key inputs); without it the attempt is untried 'resource_qualification_identity_missing' with the exercise beside
    it. A cap hit with a failed exercise is 'host_resources_bound_exceeded'. An exercise already untried (a guard trip, a refusal
    before launch) keeps its own reason."""
    verdict = qual.get('verdict')
    ex_state, ex_error = exercise.get('state'), exercise.get('error')
    if ex_state == 'untried':
        return 'untried', ex_error
    tail = f'; exercise {ex_state}' + (f': {str(ex_error)[:300]}' if ex_error else '')
    if qual.get('kind') == 'contained_resident' and verdict == 'within':
        missing = missing_identity(qual)
        if missing:
            return 'untried', f"resource_qualification_identity_missing: contained within-bound attempt without its placement identity (missing {', '.join(missing)}){tail}"
    if verdict in ('within', 'not_contained'):
        return ex_state, ex_error
    peak, limit = qual.get('measured_peak_job_commit_bytes'), qual.get('job_commit_limit_bytes')
    if verdict == 'exceeded' and qual.get('limit_hit') and ex_state != 'pass':
        return 'untried', f'host_resources_bound_exceeded: contained job commit peak {peak} reached the limit {limit}{tail}'
    if verdict == 'exceeded':
        return 'untried', f'resource_qualification_exceeded: contained job commit peak {peak} reached the limit {limit}{tail}'
    return 'untried', f"resource_qualification_{verdict}: {qual.get('reason') or 'resource readings incomplete'}{tail}"


FAILURE_DIAGNOSTICS_DIR = Path('outputs') / 'music-failures'  # written by titanos.music_backend under the cell's TITANOS_HOME
FAILURE_DIAGNOSTIC_LIMIT = 4
FAILURE_DIAGNOSTIC_BYTES = 16384


_DIAGNOSTIC_RESERVED_KEYS = ('file', 'problem', 'attributed')  # the reader's own row markers; a record may not overwrite them


def _parse_utc(value):
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)


def failure_diagnostics(root, limit=FAILURE_DIAGNOSTIC_LIMIT, max_bytes=FAILURE_DIAGNOSTIC_BYTES, since=None):
    """The bounded adapter failure diagnostics Titanos retained under the cell's TITANOS_HOME (outputs/music-failures/*.json:
    exit code, redacted stderr tail, result/progress evidence), newest first, at most `limit` files of at most `max_bytes` each.
    Each row carries the file name; an unreadable or oversized file is listed with its problem instead of its content. The cell
    root and --out are reused across attempts and runs, so with `since` (the attempt's utc_start, ISO 8601) every row is tagged
    'attributed': True when its record 'utc' (or, without one, the file's mtime) is at or after that start, False when it predates
    it, i.e. belongs to an earlier attempt or run and is listed only as prior evidence."""
    directory = Path(root) / FAILURE_DIAGNOSTICS_DIR
    if not directory.is_dir():
        return []
    start = _parse_utc(since) if since is not None else None
    rows = []
    for path in sorted(directory.glob('*.json'), key=lambda p: p.name, reverse=True)[:limit]:
        row = {'file': path.name}
        stamp = None
        try:
            stat = path.stat()
            stamp = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            if stat.st_size > max_bytes:
                row['problem'] = f'diagnostic file is {stat.st_size} bytes, over the {max_bytes} byte bound'
            else:
                body = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(body, dict):
                    reserved = {key: body.pop(key) for key in _DIAGNOSTIC_RESERVED_KEYS if key in body}
                    row.update(body)
                    if reserved:
                        row['record_reserved_keys'] = reserved  # kept, but never as the row's own markers
                    stamp = _parse_utc(body.get('utc')) or stamp
                else:
                    row['problem'] = 'diagnostic file is not a JSON object'
        except (OSError, ValueError) as exc:
            row['problem'] = f'{type(exc).__name__}: {str(exc)[:200]}'
        if since is not None:  # an unparsable start attributes nothing rather than everything
            row['attributed'] = bool(start is not None and stamp is not None and stamp >= start)
        rows.append(row)
    return rows


def attach_failure_diagnostics(exercise, root, since=None):
    """A failed exercise carries the retained adapter diagnostics (exercise['diagnostics'], a list, possibly empty) so the persistent
    receipt keeps the child's exit code and stderr tail; with `since` (the attempt's utc_start) each row says whether it is
    attributed to this attempt or is prior evidence from an earlier attempt/run in the same cell root. A pass is unchanged."""
    if exercise.get('state') == 'fail':
        exercise['diagnostics'] = failure_diagnostics(root, since=since)
    return exercise


def cell_guard(stop_file, gpu_uuid, resource, qual):
    """The CellGuard of one attempt over the module readers (host commit and physical memory, GPU memory, and the owned-job peaks once
    the tree exists). The shared-host reserve and the cap apply to contained cells only, and so does the GPU-starvation trip:
    margin_bytes is passed only for a contained cell, because an artifact-floor cell's GPU floor is artifact bytes + margin and
    request-time activations of a large model legitimately use part of that margin (a low reading is recorded, never a trip there).
    STOP and missing required readings trip every cell."""
    contained = qual['kind'] == 'contained_resident'
    return containment.CellGuard(stop_file, None, qual['host_reserve_bytes'], qual['job_commit_limit_bytes'], gpu_uuid, (resource.get('floors') or {}).get('gpu'),
                                 resource.get('margin_bytes') if contained else None, commit_reader=available_commit_bytes, gpu_free_reader=gpu_free_mib,
                                 gpu_used_reader=gpu_used_mib, phys_reader=available_physical_bytes)


def serve_and_exercise(entry, cfg, root, python, threads, gpu_uuid, request_timeout, stop_file=None, resource=None):
    """Start one Titanos server for the cell, load, request, unload, exit; return the attempt record.

    `resource` is the admission decision for this attempt. None or kind 'artifact_floor' launches the server as a plain child under
    the full artifact floor (resource verdict 'not_contained'). Kind 'contained_resident' launches it through containment.ContainedJob:
    the job commit cap (resource['job_limit']) is set and read back while the child is still suspended, so no weight is allocated
    outside the cap; a ContainmentError (a readback mismatch, or the OS refusing the job while the child was suspended, which the
    vendor surfaces as a WinError) leaves the attempt untried 'containment_unavailable' with the vendor's cleanup evidence (the cell
    is never run uncontained under a lowered floor), and a failure to create the child at all is a 'launch_error' fail; neither ends
    the run. The guard is checked once more immediately before the launch: a STOP or a missing reading there leaves the attempt
    untried '<stopped|budget_violated>_during_launch' without creating any process. Either way a CellGuard checks STOP and the
    budgets at least every 2 s during server start, model load, the active request and the listed/details/unload calls
    (requests_catalog polls it in its wait loops; blocking calls run under guarded_call); a trip cancels only owned work (the music
    job on our server, then the owned tree) and the partial record is kept as untried with every sample. After the tree has ended,
    'after_exit' records host commit, physical memory and GPU memory on every path (pass, fail, trip). The record separates
    'exercise' (output correctness) from 'resource_qualification' (the bound, commit-only: 'job_commit', and what was measured)."""
    name = entry['task']
    path = root / 'models.json'
    write_config(path, cfg)
    port = free_port()
    env = dict(os.environ, TITANOS_HOME=str(root), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
    env.pop('TITANOS_API_KEY', None)
    env.update({str(k): str(v) for k, v in (entry.get('env') or {}).items()})  # per-model settings such as TITANOS_TESSERACT_EXECUTABLE
    home_files = place_home_files(entry, root)
    rec = {'utc_start': utc(), 'port': port, 'steps': [], 'gpu_used_mib': {'before': gpu_used_mib(gpu_uuid)}, **({'home_files': home_files} if home_files else {})}
    resource = resource or {}
    kind = resource.get('kind') or 'artifact_floor'
    contained = kind == 'contained_resident'
    qual = {'kind': kind, 'bound': 'job_commit' if contained else None, 'verdict': 'unknown' if contained else 'not_contained', 'floors': resource.get('floors'),
            'host_reserve_bytes': resource.get('reserve_bytes') if contained else None, 'job_commit_limit_bytes': int(resource['job_limit']) if contained else None,
            'job_limit_source': resource.get('job_limit_source') if contained else None, 'profile_key': resource.get('profile_key') if contained else None,
            'profile': resource.get('profile') if contained else None, 'inputs_detail': resource.get('inputs_detail') if contained else None,
            # the placement identity the measured bound is bound to (the profile-key inputs): required by final_attempt_state and ingest for 'within'
            **{field: ((resource.get('profile_inputs') or {}).get(field) if contained else None) for field in IDENTITY_FIELDS},
            'admission_samples': resource.get('admission_samples'), 'measured_peak_job_commit_bytes': None, 'measured_peak_process_commit_bytes': None,
            'limit_hit': False, 'samples': [], 'cleanup': {}, 'reason': resource.get('reason'),
            # what the commit cap enforces and what it cannot (constant, containment.CAP_SEMANTICS), the GPU driver model and the WDDM-aware
            # device-resident estimate of the pre-launch check: a contained receipt states exactly what its bound means
            'cap_semantics': containment.CAP_SEMANTICS if contained else None, 'driver_model': resource.get('driver_model') if contained else None,
            'device_resident_estimate': resource.get('device_resident_estimate') if contained else None}
    exercise = {'state': 'untried', 'steps': rec['steps']}
    guard = cell_guard(stop_file, gpu_uuid, resource, qual)
    log_path = root / 'server.log'
    command = [python, '-m', 'titanos', '--config', str(path), 'serve', '--port', str(port), '--threads', str(threads)]
    job, process, trip = None, None, None
    with log_path.open('w+') as log:
        try:
            guard.check('launch', force=True)  # STOP or a missing reading immediately before the launch: nothing is created
            if contained:
                # suspended child -> cap set -> readback verified -> resume, all inside ContainedJob; the whole owned tree shares the cap
                job = containment.ContainedJob(command, int(resource['job_limit']), cwd=root, env=env, stdout=log, stderr=log)
                process, guard.job = job.process, job
                qual['containment'] = job.readback
            else:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
        except containment.GuardTrip as exc:
            trip = guard.trip or {'kind': exc.kind, 'phase': exc.phase, 'reason': exc.reason}
            exercise.update(state='untried', error=str(exc))
            qual['cleanup'].update(launched=False, active_processes_after=0, forced_cleanup=False)
        except containment.ContainmentError as exc:
            owned = getattr(exc, 'owned_job', None)  # the vendor already killed the suspended child and closed the job
            exercise.update(state='untried', error=f'containment_unavailable: {exc}')
            qual.update(containment=getattr(owned, 'readback', None), cleanup=dict(getattr(owned, 'cleanup', None) or {}, launched=False), reason=str(exc))
        except OSError as exc:
            owned = getattr(exc, 'owned_job', None)  # the child could not be created (a missing interpreter): a harness failure, not resource evidence
            exercise.update(state='fail', error=f'launch_error: {type(exc).__name__}: {exc}')
            qual.update(cleanup=dict(getattr(owned, 'cleanup', None) or {}, launched=False))
        if process is not None:
            qual['cleanup']['launched'] = True  # a process tree was created: the cleanup evidence below is about it
            try:
                with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=request_timeout, trust_env=False) as c:
                    deadline = time.monotonic() + 120
                    while True:
                        guard.check('server_start')
                        try:
                            # A per-call timeout matching the guard cadence: a socket that listens but never answers cannot hold the kit
                            # unguarded for the client's request timeout; a slow start keeps looping under the deadline and the guard.
                            if c.get('/healthz', timeout=2.0).status_code == 200:
                                break
                        except (httpx.ConnectError, httpx.TimeoutException):
                            pass
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('server failed to start')
                        time.sleep(0.2)
                    rec['steps'].append({'started': True, 'utc': utc()})
                    t0 = time.monotonic()
                    r = containment.guarded_call(guard, 'model_load', c.post, '/api/models/load', json={'model': name})
                    if r.status_code != 200:
                        raise RuntimeError(f'load refused HTTP {r.status_code}: {r.text[:1500]}')
                    rec['steps'].append({'loaded': True, 'seconds': round(time.monotonic() - t0, 2), 'response': r.json() if 'json' in r.headers.get('content-type', '') else r.text[:400]})
                    rec['gpu_used_mib']['after_load'] = gpu_used_mib(gpu_uuid)
                    rec['load_sample'] = guard.check('model_load', force=True)  # load-time host commit, physical memory, GPU memory and owned-tree peak
                    t1 = time.monotonic()

                    def request_guard(**owned_work):
                        guard.note(**owned_work)  # the music job id, so a trip can cancel exactly that job
                        guard.check('request')
                    # The request's own kwargs are bound into the callable: nothing named 'guard' reaches guarded_call's signature.
                    output = containment.guarded_call(guard, 'request', lambda: requests_catalog.run(entry['task'], c, name, entry, guard=request_guard))
                    rec['steps'].append({'request_verified': True, 'seconds': round(time.monotonic() - t1, 2), 'output': output})
                    rec['request_sample'] = guard.check('request', force=True)  # transient peak after the request

                    def list_details_unload():
                        loaded = c.get('/api/models/loaded').json()
                        ids = [m.get('id') for m in loaded.get('data', [])]
                        if name not in ids:
                            raise RuntimeError(f'model not listed as loaded after the request: {ids}')
                        rec['steps'].append({'listed_loaded': True})
                        details = c.post('/api/models/details', json={'model': name})
                        if details.status_code == 200:
                            rec['details'] = details.json()
                        u = c.post('/api/models/unload', json={'model': name})
                        if u.status_code != 200:
                            raise RuntimeError(f'unload HTTP {u.status_code}: {u.text[:500]}')
                        rec['steps'].append({'unloaded': True})
                    containment.guarded_call(guard, 'unload', list_details_unload)  # a hung unload is still under STOP and the budgets
                    rec['gpu_used_mib']['after_unload'] = gpu_used_mib(gpu_uuid)
                    rec['unload_sample'] = guard.check('unload', force=True)
                exercise.update(state='pass', output=output)
            except containment.GuardTrip as exc:
                trip = guard.trip or {'kind': exc.kind, 'phase': exc.phase, 'reason': exc.reason}
                exercise.update(state='untried', error=str(exc))
                qual['cleanup'].update(cancel_owned_work(port, guard.owned_work))
            except Exception as exc:  # noqa: BLE001
                exercise.update(state='fail', error=f'{type(exc).__name__}: {exc}')
            finally:
                # Replica workers and the music adapter are descendants; on Windows terminating the server does not end them. A contained
                # cell reaps them through its job (owned tree only); an uncontained cell through the psutil children snapshot.
                children = []
                if not contained:
                    try:
                        import psutil
                        children = psutil.Process(process.pid).children(recursive=True)
                    except Exception:  # noqa: BLE001
                        children = []
                process.terminate()
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=60)
                    rec['server_killed'] = True
                if contained:
                    qual['cleanup'].update(job.close(timeout=60))
                else:
                    survivors = []
                    for child in children:
                        try:
                            if child.is_running():
                                child.kill()
                                survivors.append(child.pid)
                        except Exception:  # noqa: BLE001
                            pass
                    if survivors:
                        rec['orphan_workers_killed'] = survivors
                    qual['cleanup'].update({'active_processes_after': None, 'job_handle_closed': None, 'forced_cleanup': bool(survivors), 'orphan_workers_killed': survivors})
                rec['server_returncode'] = process.returncode
                rec['gpu_used_mib']['after_exit'] = gpu_used_mib(gpu_uuid)
        # cleanup evidence on every path (pass, fail, trip, never launched): host commit and physical memory beside the GPU reading
        rec['after_exit'] = {'utc': utc(), 'launched': process is not None, 'available_commit_bytes': available_commit_bytes(), 'available_physical_bytes': available_physical_bytes(),
                             'gpu_free_mib': gpu_free_mib(gpu_uuid), 'gpu_used_mib': gpu_used_mib(gpu_uuid)}
    if exercise['state'] == 'pass' and rec.get('server_killed'):
        exercise.update(state='fail', error='server did not exit cleanly on terminate')
    attach_failure_diagnostics(exercise, root, since=rec['utc_start'])  # the adapter's exit code and redacted stderr tail survive the cell's temp directories; older rows are prior attempts
    summary = guard.summary()
    final = (qual['cleanup'].get('final_query') or {}) if contained else {}
    job_peaks = [p for p in (summary['measured_peak_job_commit_bytes'], final.get('peak_job_commit_bytes')) if p is not None]
    process_peaks = [p for p in (summary['measured_peak_process_commit_bytes'], final.get('peak_process_commit_bytes')) if p is not None]
    qual.update(samples=summary['samples'], trip=trip, measured_peak_job_commit_bytes=max(job_peaks) if job_peaks else None,
                measured_peak_process_commit_bytes=max(process_peaks) if process_peaks else None, limit_hit=summary['limit_hit'])
    working_sets = [s.get('tree_peak_working_set_bytes') for s in summary['samples'] if s.get('tree_peak_working_set_bytes') is not None]
    qual['measured_peak_tree_working_set_bytes'] = max(working_sets) if working_sets else None  # uncapped: includes file-backed pages
    if contained:
        limit = qual['job_commit_limit_bytes']
        peak = qual['measured_peak_job_commit_bytes']
        qual['limit_hit'] = bool(summary['limit_hit'] or (peak is not None and peak >= limit))
        readings_complete = bool(summary['readings_complete'] and final and 'final_query_error' not in qual['cleanup'])
        qual['readings_complete'] = readings_complete
        qual['verdict'] = containment.qualification_verdict(peak, limit, qual['limit_hit'], readings_complete)
        if qual['verdict'] == 'unknown':
            qual['reason'] = 'resource readings incomplete: ' + (trip['reason'] if trip and not summary['readings_complete'] else ('no final owned-job query: ' + str(qual['cleanup'].get('final_query_error') or 'job handle unavailable')))
    rec.update(exercise=exercise, resource_qualification=qual, utc_end=utc(), server_log_tail=log_path.read_text(encoding='utf-8', errors='replace')[-6000:])
    rec['state'], rec['error'] = final_attempt_state(exercise, qual)
    return rec


def dtype_plan(slot, entry, requested):
    if requested != 'auto-policy':
        return [d for d in requested.split(',') if d]
    plan = ['auto']
    gpu = slot.get('gpu') or {}
    try:
        cap = float(gpu.get('capability', 0) or 0)
    except ValueError:
        cap = 0
    if slot['runtime'] in ('torch-cuda', 'torch-rocm') and cap and cap < 8.0:
        plan = ['float16', 'float32']
    elif slot['runtime'] in ('torch-mps', 'torch-xpu'):
        plan = ['auto', 'float32']
    if entry['backend'] in ('onnx', 'openvino', 'openvino-genai', 'sherpa-onnx', 'tesseract', 'llama-cpp', 'onnx-diffusers'):
        plan = ['auto']
    return plan


ADMISSION_ERRORS = {
    'stop': 'blocked_by_host_resources: STOP before admission',
    'resource_reading_unavailable': 'resource_reading_unavailable: a required commit/GPU reading was missing for three consecutive admission samples (admission fails closed)',
}


def wait_admission(floors, gpu_uuid, stop_file, log, probe_only=False, contained=None, commit_reader=available_commit_bytes, gpu_reader=gpu_free_mib, sleep=time.sleep,
                   phys_reader=available_physical_bytes):
    """Block until the resource floors hold for three consecutive samples, or STOP appears, or a required reading stays missing.

    Returns {'admitted': True | False | None, 'reason': None | 'stop' | 'deferred' | 'resource_reading_unavailable' |
    'contained_budget_below_minimum', 'samples': [last three], 'floors': the floors applied, 'job_limit', 'job_limit_source'}.
    Fail closed: a required reading (commit when a commit floor or contained admission applies, GPU free memory when a GPU floor
    applies on a GPU slot) that is None is never good, and three consecutive missing readings end the wait with
    'resource_reading_unavailable' so the caller records the cell untried. STOP is checked at the top of every iteration and once
    more immediately before success. With probe_only, a sample below the floor returns admitted None ('deferred') so the caller can
    defer the cell instead of blocking.
    `contained` = {'reserve_bytes', 'profile_peak_bytes', 'margin_bytes'} selects contained admission: the owned-tree commit cap is
    fixed from the first sample of the admitting run (containment.derive_job_limit), must be at least 2 x margin, and all three
    samples need available commit >= cap + reserve (the reserve stays free for the shared host) besides the unchanged GPU floor.
    Every sample also records available physical memory (phys_reader); contained admission requires that reading (the commit cap does
    not see file-backed mappings) and needs it at or above the reserve in all three samples."""
    floors = dict(floors)
    samples, consecutive, missing_streak = [], 0, 0
    job_limit, job_source = None, None

    def result(admitted, reason):
        return {'admitted': admitted, 'reason': reason, 'samples': samples[-3:], 'floors': floors, 'job_limit': job_limit, 'job_limit_source': job_source}

    def stopped():
        return bool(stop_file) and Path(stop_file).exists()
    while True:
        if stopped():
            return result(False, 'stop')
        commit = commit_reader()
        free = gpu_reader(gpu_uuid) if gpu_uuid else None
        try:
            phys = phys_reader() if phys_reader is not None else None
        except Exception:  # noqa: BLE001
            phys = None
        sample = {'utc': utc(), 'available_commit_bytes': commit, 'available_physical_bytes': phys, 'gpu_free_mib': free}
        samples.append(sample)
        commit_required = floors['commit'] is not None or contained is not None
        gpu_required = floors['gpu'] is not None and bool(gpu_uuid)
        if (commit_required and commit is None) or (gpu_required and free is None) or (contained is not None and phys is None):
            consecutive, missing_streak = 0, missing_streak + 1
            if missing_streak >= 3:
                return result(False, 'resource_reading_unavailable')
            log(f"admission reading missing (fail closed): {sample}")
            sleep(2)
            continue
        missing_streak = 0
        if contained is not None and consecutive == 0:
            job_limit, job_source = containment.derive_job_limit(commit, contained['reserve_bytes'], contained.get('profile_peak_bytes'), contained['margin_bytes'])
            if job_limit < 2 * int(contained['margin_bytes']):
                return result(False, 'contained_budget_below_minimum')
            floors['commit'] = job_limit + int(contained['reserve_bytes'])
        good = (floors['commit'] is None or commit >= floors['commit']) and (floors['gpu'] is None or not gpu_uuid or free >= floors['gpu'])
        if contained is not None and phys < int(contained['reserve_bytes']):
            good = False  # the shared-host reserve must be free in physical memory too, not only in commit
        consecutive = consecutive + 1 if good else 0
        if consecutive >= 3:
            if stopped():  # once more, immediately before success
                return result(False, 'stop')
            return result(True, None)
        if probe_only and not good:
            return result(None, 'deferred')
        if good:
            sleep(2)  # confirm a clear floor with two more close samples
            continue
        log(f"waiting for admission: {sample} floors {floors}")
        sleep(30)


def classify_refusal(error):
    """An adapter refusal before any weight loaded ('placement_refused_before_load', the music adapter's residency check) is never a
    fail: 'na' when it says resident placement requires a CUDA device (the slot cannot host residency by construction), otherwise
    untried 'blocked_by_device_resources' (capacity, accelerate or dtype cannot support residency here). None for other errors."""
    if not error or 'placement_refused_before_load' not in str(error):
        return None
    text = ' '.join(str(error).split())
    if re.search(r'resident[^;]{0,160}?(require|need)[^;]{0,80}?CUDA|CUDA device[^;]{0,60}?(require|need)', text, re.I):
        return 'na', f'not applicable: the adapter refused resident placement before loading any weights because it requires a CUDA device: {text[:400]}'
    return 'untried', f'blocked_by_device_resources: {text[:600]}'


def oom_disposition(error, contained, already_boosted):
    """What a host-memory exhaustion during load means for the cell: 'requeue' (artifact floor, first time: doubled floor),
    'untried' (the cell stays untried with the reason), None when the error is not an OOM. A contained cell is never re-queued
    with a doubled floor: its bound was enforced and measured, so the exceedance is resource evidence, not a floor to raise."""
    if not error or not OOM.search(str(error)):
        return None
    if contained or already_boosted:
        return 'untried'
    return 'requeue'


def resident_profile_inputs(entry):
    """The profile-key inputs read from the entry's home_files music-runtime.json: (receipt_sha256, adapter_sha256, detail). The
    adapter's pin is the receipt's 'files' entry for the adapter path relative to 'destination'. Missing pieces are None (the key
    stays deterministic) with the reason in detail."""
    source = (entry.get('home_files') or {}).get('music-runtime.json')
    if not source:
        return None, None, 'no music-runtime.json among home_files'
    try:
        data = Path(source).read_bytes()
    except OSError as exc:
        return None, None, f'music-runtime.json unreadable: {exc}'
    receipt_sha = hashlib.sha256(data).hexdigest()
    try:
        receipt = json.loads(data)
    except ValueError as exc:
        return receipt_sha, None, f'music-runtime.json is not JSON: {exc}'
    adapter, destination, files = receipt.get('adapter'), receipt.get('destination'), receipt.get('files') or {}
    if not adapter:
        return receipt_sha, None, 'receipt names no adapter'
    rel = None
    try:
        rel = Path(adapter).resolve().relative_to(Path(destination).resolve()).as_posix() if destination and Path(adapter).is_absolute() else Path(adapter).as_posix()
    except ValueError:
        rel = None
    pin = files.get(rel) if rel else None
    digest = pin.get('sha256') if isinstance(pin, dict) else pin
    if not isinstance(digest, str):
        return receipt_sha, None, f'adapter {rel or adapter} has no sha256 pin in the receipt files'
    return receipt_sha, digest, 'ok'


def verify_profile(key, profile, profiles_path, out=None):
    """A measured profile lowers the contained cap only when it is bound to the receipt that measured it: the entry names a receipt that
    exists (absolute, or relative to the profiles file's folder or the run's --out), whose cell state is pass, whose
    resource_qualification verdict is 'within', and whose profile_key, measured job peak and cap equal the entry's. Anything else (a
    hand-edited, stale or unbound entry, a missing receipt) is a first run with the reason recorded; the enforced cap keeps a wrong
    profile honest either way, but no unverified input lowers a floor. Returns (peak_bytes or None, reason, provenance or None)."""
    if not isinstance(profile, dict):
        return None, 'first run: no measured profile for this key', None
    peak, receipt = profile.get('measured_peak_commit_bytes'), profile.get('receipt')
    if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0 or not isinstance(receipt, str) or not receipt:
        return None, 'first run: the profile for this key is rejected (no positive integer measured peak or no receipt reference)', None
    candidates = [Path(receipt)] if Path(receipt).is_absolute() else [Path(profiles_path).parent / receipt] + ([Path(out) / receipt] if out else [])
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None, f'first run: the profile for this key is rejected (receipt {receipt} not found)', None
    try:
        data = path.read_bytes()
        body = json.loads(data.decode('utf-8'))
    except (OSError, ValueError) as exc:
        return None, f'first run: the profile for this key is rejected (receipt {receipt} unreadable: {exc})', None
    rq = (body.get('resource_qualification') if isinstance(body, dict) else None) or {}
    problems = []
    if body.get('state') != 'pass':
        problems.append(f"receipt state {body.get('state')!r} is not pass")
    if rq.get('verdict') != 'within':
        problems.append(f"receipt verdict {rq.get('verdict')!r} is not within")
    if rq.get('profile_key') != key:
        problems.append('receipt profile_key differs from this key')
    if rq.get('measured_peak_job_commit_bytes') != peak:
        problems.append(f"receipt measured peak {rq.get('measured_peak_job_commit_bytes')} differs from the profile peak {peak}")
    if rq.get('job_commit_limit_bytes') != profile.get('job_commit_limit_bytes'):
        problems.append(f"receipt cap {rq.get('job_commit_limit_bytes')} differs from the profile cap {profile.get('job_commit_limit_bytes')}")
    if problems:
        return None, 'first run: the profile for this key is rejected (' + '; '.join(problems) + ')', None
    provenance = {'key': key, 'receipt': str(path), 'receipt_sha256': hashlib.sha256(data).hexdigest(), 'run_id': body.get('run_id') or profile.get('run_id'),
                  'utc': profile.get('utc'), 'measured_peak_commit_bytes': peak, 'job_commit_limit_bytes': profile.get('job_commit_limit_bytes')}
    return peak, 'measured profile', provenance


def artifact_floor_reason(placement, gpu):
    """The admission-plan reason recorded when a cell keeps the full artifact floor. The reason is written into the receipt's
    resource_qualification and, together with the receipt's spec_placement and device, is what ingest.py accepts for an artifact_floor
    block (ingest.artifact_floor_reason mirrors these three texts exactly), so a contained cell cannot be relabelled artifact_floor."""
    if placement != 'resident':
        return f'placement {placement!r} is not resident; auto/None never lowers the floor (auto may fall back to CPU offload)'
    if not gpu:
        return 'resident placement on a slot without a GPU: no contained admission, full artifact floor'
    return 'containment_unavailable: this host cannot cap the owned process tree (no Windows Job Object); the full artifact floor applies unchanged'


def contained_admission_plan(entry, gpu_uuid, mode, dtype, reserve_bytes, margin_bytes, profiles, available=None, profiles_path=None, out=None):
    """Which admission this cell gets. 'contained_resident' ONLY when the entry's spec says placement 'resident' explicitly AND the slot
    has a GPU AND the host can cap the owned tree (containment.available()); otherwise 'artifact_floor' with the reason recorded
    ('auto' or no placement never lowers the floor because auto may fall back to CPU offload; a CPU slot has no residency; a host
    without Job Object caps keeps the full floor even for a resident spec). No caller-supplied byte estimate is read here. A measured
    profile is used only when verify_profile binds it to its within-bound pass receipt, and only when the profile-key inputs are
    complete (inputs_detail 'ok': the music-runtime receipt and the adapter pin were read); with incomplete inputs the cell runs
    contained under the first-run cap alone and the caller records no profile for the key."""
    available = containment.available if available is None else available
    placement = (entry.get('spec') or {}).get('placement')
    if placement != 'resident' or not gpu_uuid or not available():
        return {'kind': 'artifact_floor', 'reason': artifact_floor_reason(placement, bool(gpu_uuid))}
    receipt_sha, adapter_sha, detail = resident_profile_inputs(entry)
    copies = 2 if mode == 'replicas' else 1
    inputs = {'matrix_identity': entry.get('matrix_identity'), 'receipt_sha256': receipt_sha, 'adapter_sha256': adapter_sha, 'placement': placement, 'dtype': dtype, 'copies': copies}
    key = containment.profile_key(**inputs)
    profile = (profiles or {}).get(key)
    if detail != 'ok':
        peak, reason, provenance = None, f'first run: profile inputs incomplete ({detail}); profiles are ignored for this key and none is recorded', None
    else:
        peak, reason, provenance = verify_profile(key, profile, profiles_path, out) if profiles_path is not None else (None, 'first run: no profiles file for this run', None)
    return {'kind': 'contained_resident', 'profile_key': key, 'inputs': inputs, 'inputs_detail': detail, 'profile': profile, 'profile_peak_bytes': peak,
            'profile_provenance': provenance, 'profiles_usable': detail == 'ok', 'reason': reason}


def cell_receipt(run_id, entry, s, mode, group, host, attempts, state, note, final=None):
    """The detailed cell receipt (receipt_schema RECEIPT_SCHEMA). Its top-level 'resource_qualification' is the COMPLETE measured block
    of the attempt that decided the cell (the pass, else the last attempt): kind, bound, verdict, cap and its source, reserve, floors,
    the cap readback ('containment'), measured peaks, every guard sample, the placement identity, cleanup evidence, trip and reason.
    ingest.py validates exactly this block before a current contained cell can enter the matrix; 'exercise' (output correctness) is
    kept beside it, never mixed into it. None when the deciding attempt recorded no block (a config refusal before any launch).
    run_id, model_id, slot (the slot dict) and mode bind the receipt to its RUN_RESULT row (ingest.binding_problems); spec_placement
    is the manifest placement the admission plan read (ingest checks an artifact_floor block's reason against it and the device)."""
    last = final or (attempts[-1] if attempts else {})
    rq = last.get('resource_qualification') or {}
    exercise = {k: v for k, v in (last.get('exercise') or {}).items() if k != 'steps'} or None
    return {'receipt_schema': RECEIPT_SCHEMA, 'run_id': run_id, 'model_id': entry['model_id'], 'display_name': entry.get('display_name'), 'task': entry['task'],
            'backend': entry['backend'], 'matrix_identity': entry.get('matrix_identity'), 'identity_check': entry.get('identity_check'), 'path': entry['path'], 'slot': s,
            'spec_placement': (entry.get('spec') or {}).get('placement'),
            'mode': mode, 'group': group, 'host': host, 'attempts': attempts, 'state': state, 'note': note, 'utc': utc(),
            'exercise': exercise, 'resource_qualification': dict(rq) if rq else None}


def machine_name_in(text, local_hostname=None):
    """Why `text` (a run id or host label) names a machine instead of a role: 'a generated Windows machine name' when it carries the
    policy's generated-name class (tools/lab_paths.WINDOWS_GENERATED_HOST), 'the machine hostname' when it carries this machine's own
    hostname whole-word (any ASCII case); None otherwise. Role labels (tools/lab_paths.HOST_ROLE_LABELS) pass; a computer name never does."""
    if lab_paths.WINDOWS_GENERATED_HOST.search(text or ''):
        return 'a generated Windows machine name'
    name = (local_hostname or '').strip()
    if len(name) >= 3 and re.search(r'(?<![A-Za-z0-9])' + re.escape(name) + r'(?![A-Za-z0-9])', text or '', re.IGNORECASE | re.ASCII):
        return 'the machine hostname'
    return None


def load_profiles(path):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).is_file() else {}
    except (OSError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def record_profile(path, profiles, plan, rec, run_id, receipt):
    """After a within-bound pass: the measured peak for this profile key (the next contained cap is 1.5 x peak + margin)."""
    rq = rec['resource_qualification']
    profiles[plan['profile_key']] = {**plan['inputs'], 'measured_peak_commit_bytes': rq['measured_peak_job_commit_bytes'], 'measured_peak_process_commit_bytes': rq.get('measured_peak_process_commit_bytes'),
                                     'job_commit_limit_bytes': rq['job_commit_limit_bytes'], 'job_limit_source': rq.get('job_limit_source'), 'run_id': run_id, 'receipt': receipt, 'utc': utc()}
    Path(path).write_text(json.dumps(profiles, indent=2, sort_keys=True, default=str), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--slots', default='auto')
    ap.add_argument('--modes', default='single,replicas,model_sharding,tensor_parallel')
    ap.add_argument('--allowed-gpus', default=None, help='comma-separated GPU UUIDs this run may use; default all')
    ap.add_argument('--allowed-devices', default=None, help="comma-separated Titanos device strings to keep (for example 'cpu,dml:1,GPU.0'); other detected devices are reported but never assigned")
    ap.add_argument('--native-sycl', default=None, help='path of a SYCL llama.cpp llama-server; its Intel GPUs become native-sycl entries of the Intel GPU slots for llama-cpp rows')
    ap.add_argument('--dtypes', default='auto-policy', help="'auto-policy' or an explicit comma list tried in order")
    ap.add_argument('--min-available-commit-gib', type=float, default=4.0)
    ap.add_argument('--min-gpu-free-mib', type=int, default=4096)
    ap.add_argument('--floor-margin-gib', type=float, default=2.0, help='fixed margin added to every size-based admission floor (commit and GPU); a kit guard for the serving process footprint, recorded in RUN_RESULT')
    ap.add_argument('--host-reserve-gib', type=float, default=None, help='shared-host commit reserve kept free of every contained cell (default: --min-available-commit-gib + --floor-margin-gib); recorded in RUN_RESULT')
    ap.add_argument('--admission-profiles', type=Path, default=None, help='measured within-bound peaks per profile key (default <out>/ADMISSION_PROFILES.json); a measured profile sets the next contained cap to 1.5 x peak + margin; no other input ever lowers a floor')
    ap.add_argument('--stop-file', default=None)
    ap.add_argument('--only', default='', help='comma-separated model ids')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--threads', default='4')
    ap.add_argument('--request-timeout', type=float, default=3600)
    ap.add_argument('--run-id', default=None, help='run label (default matrix-kit-<host label or os-arch>-<UTC stamp>); a fleet campaign passes fleet-<host-label>-<yyyymmdd>; it becomes the reader family in the matrix and never carries a computer name')
    ap.add_argument('--host-label', default=None, help='stable role label of this host (tools/lab_paths.HOST_ROLE_LABELS, for example turing-laptop-win-x64; docs/MATRIX_KIT.md, Host labels): written into every receipt as host.label and used in the default run id; never a computer name')
    a = ap.parse_args()
    # The shared-host reserve and the margin are enforced at the boundary: a non-positive margin admits a cap of one byte (the cap
    # minimum is 2 x margin), a non-positive reserve inflates the first-run cap past what the host has and disables the starvation trip.
    if a.floor_margin_gib is None or a.floor_margin_gib <= 0:
        raise SystemExit(f'--floor-margin-gib must be positive (got {a.floor_margin_gib}): the margin is part of every size-based floor and of the contained cap minimum')
    if a.host_reserve_gib is not None and a.host_reserve_gib <= 0:
        raise SystemExit(f'--host-reserve-gib must be positive (got {a.host_reserve_gib}): the shared-host reserve is kept free of every contained cell and bounds the first-run cap')
    if a.min_available_commit_gib < 0 or a.min_gpu_free_mib < 0:
        raise SystemExit('--min-available-commit-gib and --min-gpu-free-mib must not be negative')
    # Servers are launched with cwd = the cell work dir and given the registry path, so every path must be absolute here;
    # a relative --out made each server exit with 'No registry at <work dir>/run/receipts/...' (fleet run, 2026-09-24).
    a.out = a.out.resolve()
    a.manifest = a.manifest.resolve()
    if a.stop_file:
        a.stop_file = str(Path(a.stop_file).resolve())
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    allowed = a.allowed_gpus.split(',') if a.allowed_gpus else None
    det = slotmod.detect(allowed_gpus=allowed, native_sycl=a.native_sycl)
    det['host']['label'] = a.host_label  # the role label is the only host name a receipt carries; the OS name stays in HOST_DETECTION.json
    (a.out / 'HOST_DETECTION.json').write_text(json.dumps(det, indent=2, default=str), encoding='utf-8')
    log_file = (a.out / 'run.log').open('a', encoding='utf-8')

    def log(msg):
        line = f'{utc()} {msg}'
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    slots = det['slots'] if a.slots == 'auto' else [s for s in det['slots'] if s['slot'] in a.slots.split(',')]
    shared = {k for k in {s['slot'] for s in slots} if sum(1 for s in slots if s['slot'] == k) > 1}

    def slot_tag(s):
        # two adapters under one slot id (two AMD dGPUs, two Intel Arcs) get distinct receipts and work dirs
        return s['slot'] + ('@' + re.sub(r'[^A-Za-z0-9.-]+', '_', str(s['device'])) if s['slot'] in shared else '')
    if a.allowed_devices:
        keep = set(a.allowed_devices.split(','))
        det['excluded_devices'] = [s['device'] for s in slots if s['device'] not in keep]
        slots = [s for s in slots if s['device'] in keep]
        (a.out / 'HOST_DETECTION.json').write_text(json.dumps(det, indent=2, default=str), encoding='utf-8')
    modes = [m for m in a.modes.split(',') if m]
    only = {m for m in a.only.split(',') if m}
    floors = {'commit': int(a.min_available_commit_gib * (1 << 30)) if a.min_available_commit_gib else None, 'gpu': a.min_gpu_free_mib or None}
    margin_bytes = int(a.floor_margin_gib * (1 << 30))
    host_reserve_bytes = int((a.host_reserve_gib if a.host_reserve_gib is not None else (a.min_available_commit_gib or 0) + a.floor_margin_gib) * (1 << 30))
    profiles_path = (a.admission_profiles or (a.out / 'ADMISSION_PROFILES.json')).resolve()
    profiles = load_profiles(profiles_path)
    containment_available = containment.available()
    cells, receipts_dir = [], a.out / 'receipts'
    work, deferred = [], []
    oom_boost, oom_receipts = {}, {}
    receipts_dir.mkdir(exist_ok=True)
    host_label = a.host_label or f"{det['host']['os']}-{det['host']['arch']}"
    run_id = a.run_id or f"matrix-kit-{host_label}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    named = machine_name_in(run_id, det.get('local_hostname')) or machine_name_in(a.host_label, det.get('local_hostname'))
    if named:
        raise SystemExit(f'run ids and host labels name a host by its role label, never by {named} (docs/MATRIX_KIT.md, Host labels; tools/public_export.json machine_name_policy)')
    log(f"run {run_id}: {len(manifest['models'])} runnable models, {len(manifest.get('unavailable', []))} unavailable, slots {[s['slot'] for s in slots]}, groups {det['gpu_groups']}")
    for entry in manifest.get('unavailable', []):
        for s in slots:
            cells.append({'model_id': entry['model_id'], 'slot': s['slot'], 'mode': 'single', 'state': 'untried', 'verified': False, 'evidence_ids': [],
                          'note': 'not run: model artifact absent on this host: ' + str(entry.get('reason'))[:300]})
    for entry in manifest['models']:
        if only and entry['model_id'] not in only:
            continue
        for s in slots:
            gpu_uuid = s['device'] if str(s['device']).startswith('GPU-') else None
            if entry['backend'] not in COMPAT.get(s['runtime'], set()):
                # No execution path for this backend on this runtime: not applicable by construction. When Titanos itself refuses the
                # device for this backend/task (a CPU-only backend on a GPU slot) its refusal text is the more specific reason.
                refusal = None
                try:
                    build_config(entry, s, 'single', None, 'auto')
                except ValueError as exc:
                    refusal = str(exc).replace('\n', ' ')
                except Exception as exc:  # noqa: BLE001
                    refusal = None
                    log(f"config probe error for {entry['model_id']} on {s['slot']}: {exc!r}")
                for mode in modes:
                    cells.append(no_path_cell(entry, s, mode, refusal))
                continue
            host_probes = det['host'].get('probes') or {}
            absent = next(((name, host_probes[name]) for name in RUNTIME_PROBE.get(entry['backend'], ()) if (host_probes.get(name) or {}).get('available') is False), None)
            if s['runtime'] == 'cpu' and absent is not None:
                # A runtime package the backend imports is not installed here: untried with the reason (an import error is not an execution).
                for mode in modes:
                    cells.append(missing_runtime_cell(entry, s, mode, *absent))
                continue
            for mode in modes:
                group = None
                if mode in MULTI_GPU_MODES:
                    group = next((g for g in det['gpu_groups'] if g.get('slot') == s['slot']), None)
                    if group is None:
                        cells.append(missing_group_cell(entry, s, mode))
                        continue
                work.append((entry, s, mode, group))

    def write_summary():
        summary = {'run_id': run_id, 'kit_schema': RECEIPT_SCHEMA, 'utc': utc(), 'host': det['host'], 'slots': slots, 'gpu_groups': det['gpu_groups'], 'unprobed': det['unprobed'], 'cells': cells, 'floors': floors, 'floor_margin_gib': a.floor_margin_gib,
                   'host_reserve_bytes': host_reserve_bytes, 'admission_profiles': str(profiles_path), 'containment_available': containment_available,
                   'deferred_pending': [f"{e['model_id']} x {s['slot']} [{m}]" for e, s, m, _ in deferred],
                   'counts': {st: sum(1 for c in cells if c['state'] == st) for st in ('pass', 'fail', 'na', 'untried')},
                   'scope': 'engine contract per model x slot x mode on the named device; verifier must uphold before any pass enters the matrix'}
        (a.out / 'RUN_RESULT.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')

    def execute(entry, s, mode, group, probe_only):
        """Run one cell. Returns 'deferred' when probe_only and the first admission sample is below the floor."""
        gpu_uuid = s['device'] if str(s['device']).startswith('GPU-') else None
        attempts, final, na_reason = [], None, None
        receipt_name = f"{slot_tag(s)}.{entry['model_id']}.{mode}.json"
        for dtype in dtype_plan(s, entry, a.dtypes):
            try:
                cfg, desc = build_config(entry, s, mode, group['gpus'] if group else None, dtype)
            except ValueError as exc:
                msg = str(exc)
                # A by-design refusal is na with Titanos's own text: the refusal vocabulary, or a multi-GPU mode that the config.py
                # admission (mode_refusal) says Titanos refuses for this backend/task, whatever wording its validator used.
                if UNSUPPORTED.search(msg) or (mode in MULTI_GPU_MODES and mode_refusal(entry['backend'], entry['task'], mode)):
                    na_reason = msg
                else:
                    attempts.append({'dtype': dtype, 'state': 'fail', 'error': 'config_error: ' + msg[:800]})
                break
            # Size-aware floor: a model needs room to load; require max(global floor, 1.5 x artifact bytes + margin) of commit and,
            # on a GPU, max(global floor, artifact bytes + margin) of free device memory. The margin (--floor-margin-gib, default 2)
            # is a kit guard for the serving process's own footprint, recorded in RUN_RESULT; an 8 GB host needs a smaller one.
            size = int(entry.get('bytes') or 0)
            copies = 2 if mode == 'replicas' else 1  # two replica workers hold two copies of the weights
            key = (entry['model_id'], s['slot'], mode)
            boost = oom_boost.get(key, 1)
            cell_floors = {'commit': max(floors['commit'] or 0, int(size * 1.5 * copies * boost) + margin_bytes) if (floors['commit'] or size) else None,
                           # a GPU slot always records an int GPU floor (0 when --min-gpu-free-mib 0 and the artifact size is unknown: the reading is
                           # still required, the floor is trivially met); ingest requires int floors in a contained block
                           'gpu': max(floors['gpu'] or 0, (size * copies * boost) // (1 << 20) + margin_bytes // (1 << 20)) if gpu_uuid and (floors['gpu'] or size) else (0 if gpu_uuid else floors['gpu'])}
            # Contained admission only for an explicit resident spec on a GPU slot of a host that can cap the owned tree; the GPU floor
            # is unchanged and the commit floor becomes cap + reserve, fixed at the first admitting sample. Everything else keeps the
            # artifact floor (resource kind 'artifact_floor', verdict 'not_contained').
            plan = contained_admission_plan(entry, gpu_uuid, mode, dtype, host_reserve_bytes, margin_bytes, profiles, available=lambda: containment_available,
                                            profiles_path=profiles_path, out=a.out)
            contained = plan['kind'] == 'contained_resident'
            if contained:
                admission = wait_admission({'commit': None, 'gpu': cell_floors['gpu']}, gpu_uuid, a.stop_file, log, probe_only=probe_only,
                                           contained={'reserve_bytes': host_reserve_bytes, 'profile_peak_bytes': plan.get('profile_peak_bytes'), 'margin_bytes': margin_bytes})
            else:
                admission = wait_admission(cell_floors, gpu_uuid, a.stop_file, log, probe_only=probe_only)
            samples = admission['samples']
            if admission['admitted'] is None:
                log(f"deferring {entry['model_id']} x {s['slot']} [{mode}]: floor {admission['floors']} not met at first sample {samples[-1] if samples else None}")
                return 'deferred'
            def refused_qualification(reason, **extra):
                # the resource block of an attempt refused before any launch (admission, or the WDDM device-resident check): the plan identity,
                # the admission samples and, for a contained cell, the cap semantics the launch would have claimed
                return {'kind': plan['kind'], 'verdict': 'unknown' if contained else 'not_contained', 'reason': reason, 'floors': admission['floors'],
                        'host_reserve_bytes': host_reserve_bytes if contained else None, 'job_commit_limit_bytes': admission['job_limit'],
                        'job_limit_source': admission['job_limit_source'], 'profile_key': plan.get('profile_key'), **plan_identity(plan),
                        'admission_samples': samples, 'cap_semantics': containment.CAP_SEMANTICS if contained else None, **extra}
            if not admission['admitted']:
                reason = admission['reason']
                if reason == 'contained_budget_below_minimum':
                    error = f"blocked_by_host_resources: contained budget below the minimum (owned-tree cap {admission['job_limit']} < 2 x margin {2 * margin_bytes} with reserve {host_reserve_bytes})"
                else:
                    error = ADMISSION_ERRORS.get(reason, f'blocked_by_host_resources: {reason}')
                attempts.append({'dtype': dtype, 'state': 'untried', 'error': error, 'floors': admission['floors'], 'samples': samples, 'resource_qualification': refused_qualification(reason)})
                break
            resource = {'kind': plan['kind'], 'reason': plan.get('reason'), 'floors': admission['floors'], 'margin_bytes': margin_bytes, 'admission_samples': samples}
            if contained:
                resource.update(job_limit=admission['job_limit'], job_limit_source=admission['job_limit_source'], reserve_bytes=host_reserve_bytes, profile_key=plan['profile_key'],
                                profile=plan.get('profile_provenance'), profile_inputs=plan.get('inputs'), inputs_detail=plan.get('inputs_detail'))
                # WDDM-aware pre-launch check: on a WDDM host the cap must cover the device-resident weights (charged to the process commit)
                # besides torch/CUDA and activations; a blocked or fail-closed estimate leaves the cell untried with the numbers, nothing launched
                estimate = device_resident_check(entry, s, dtype, admission['job_limit'], gpu_uuid)
                resource.update(driver_model=estimate['driver_model'], device_resident_estimate=estimate)
                if estimate['blocked']:
                    log(f"cell {entry['model_id']} x {s['slot']} [{mode}, {dtype}] not launched: {estimate['reason'][:200]}")
                    attempts.append({'dtype': dtype, 'state': 'untried', 'error': estimate['reason'], 'floors': admission['floors'], 'samples': samples,
                                     'resource_qualification': refused_qualification('device_resident_check', driver_model=estimate['driver_model'], device_resident_estimate=estimate)})
                    break
            root = a.out / 'work' / slot_tag(s) / f"{entry['model_id']}.{mode}.{dtype}"
            root.mkdir(parents=True, exist_ok=True)
            log(f"cell {entry['model_id']} x {s['slot']} [{mode}, {dtype}] on {desc} ({plan['kind']}" + (f", owned-tree cap {admission['job_limit']} bytes from {admission['job_limit_source']}, reserve {host_reserve_bytes}" if contained else '') + ')')
            rec = serve_and_exercise(entry, cfg, root, a.python, a.threads, gpu_uuid, a.request_timeout, stop_file=a.stop_file, resource=resource)
            rec.update(dtype=dtype, admission_samples=samples, config=desc)
            refusal = classify_refusal((rec.get('exercise') or {}).get('error')) if rec['state'] == 'fail' else None
            if refusal:
                rec['state'], rec['error'] = refusal  # a refusal before any weight loaded is na or untried, never fail
            attempts.append(rec)
            if rec['state'] == 'pass':
                final = rec
                if contained and plan.get('profiles_usable') and (rec.get('resource_qualification') or {}).get('verdict') == 'within':
                    record_profile(profiles_path, profiles, plan, rec, run_id, f'receipts/{receipt_name}')
                break
            if rec['state'] in ('untried', 'na'):
                break  # resource evidence or a refusal: another dtype changes nothing about it
            disposition = oom_disposition(rec.get('error'), contained, oom_boost.get(key, 1) >= 2)
            if disposition == 'requeue':
                # The host ran out of memory while loading under the artifact floor: not a model verdict. Re-queue once with a doubled
                # floor; if it happens again the cell stays untried with the reason, never fail. Never for a contained cell.
                oom_boost[key] = 2
                oom_receipts.setdefault(key, []).append(rec)
                log(f"host memory exhausted loading {entry['model_id']} x {s['slot']} [{mode}]; re-queued with a doubled floor")
                return 'deferred'
            if disposition == 'untried':
                rec['state'] = 'untried'
                rec['error'] = 'host_resources_exhausted_during_load: ' + rec['error'][:400]
                attempts = oom_receipts.get(key, []) + attempts
                break
        if na_reason is not None:
            state, note = 'na', f'{mode} refused by design: {na_reason[:400]}'
        elif final is not None:
            state, note = 'pass', f"{entry['task']} pass on {s['device_named']} [{mode}, dtype {final['dtype']}]"
        elif attempts and attempts[-1]['state'] == 'untried':
            state, note = 'untried', attempts[-1]['error']
        elif attempts and attempts[-1]['state'] == 'na':
            state, note = 'na', attempts[-1]['error']
        else:
            state = 'fail'
            note = f"{entry['task']} fail on {s['device_named']} [{mode}] after {len(attempts)} attempt(s): {(attempts[-1].get('error') if attempts else 'no attempt')}"[:600]
        name = receipt_name
        receipt = cell_receipt(run_id, entry, s, mode, group, det['host'], attempts, state, note, final)
        rq = receipt['resource_qualification'] or {}
        (receipts_dir / name).write_text(json.dumps(receipt, indent=2, default=str), encoding='utf-8')
        cells.append({'model_id': entry['model_id'], 'slot': s['slot'], 'mode': mode, 'state': state, 'verified': False, 'evidence_ids': [],
                      'device': s['device'], 'device_named': s['device_named'], 'package_or_runtime': f"titanos {ROOT.name} source + {s['runtime']}",
                      'receipt': f'receipts/{name}', 'dtype': final['dtype'] if final else None, 'note': note, 'spec_placement': receipt['spec_placement'],
                      'resource_kind': rq.get('kind'), 'resource_verdict': rq.get('verdict'), 'resource_profile_key': rq.get('profile_key'),
                      'resource_identity': {field: rq.get(field) for field in IDENTITY_FIELDS} if rq.get('kind') == 'contained_resident' else None})
        log(f"  -> {state}: {note[:160]}")
        write_summary()
        return state

    # Pass 1: run every cell whose floor is clear now; defer the rest so one starved cell never stalls the campaign.
    for item in work:
        if execute(*item, probe_only=True) == 'deferred':
            deferred.append(item)
    write_summary()
    # Pass 2: the deferred cells wait for their floors (or STOP), largest last so smaller cells clear first.
    deferred.sort(key=lambda it: int(it[0].get('bytes') or 0))
    if deferred:
        log(f"pass 2: {len(deferred)} deferred cell(s) waiting for host resources")
    while deferred:
        item = deferred.pop(0)
        execute(*item, probe_only=False)
    summary = json.loads((a.out / 'RUN_RESULT.json').read_text(encoding='utf-8')) if (a.out / 'RUN_RESULT.json').exists() else {'run_id': run_id, 'kit_schema': RECEIPT_SCHEMA, 'utc': utc(), 'cells': cells, 'counts': {}}
    summary['finished_utc'] = utc()
    (a.out / 'RUN_RESULT.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    log(f"finished: {summary.get('counts')}")


if __name__ == '__main__':
    main()
