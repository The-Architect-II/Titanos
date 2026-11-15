"""Matrix kit contracts: host-to-slot mapping is deterministic and never invents a slot; multi-GPU groups need homogeneity and
the allowed set; every matrix task has a request in the catalogue; configuration refusals classify as not-applicable only when
Titanos refuses by design; a backend outside COMPAT for the slot runtime is not applicable by construction; a multi-GPU mode Titanos
refuses for the backend/task by design (mode_refusal, mirrored by the spec's multi_gpu_modes_rule.admission and cross-checked here
against Titanos's own config validation) is na, and an admitted mode without a group is na only on a slot runtime that cannot form
one (untried on GPU slots and for the explicit CPU/XPU tensor-parallel group); the full COMPAT <-> spec backend_runtime_map mapping
agrees for every kit runtime and backend; PyTorch XPU devices map to the Intel GPU slots on Linux and Windows x64; a backend any of
whose runtime packages the host probe found absent is untried with the reason, never fail (onnx-diffusers needs onnxruntime and
torch); diarization and vad are single-device tasks, cross-checked against Titanos's own validators; contained admission
(containment.py): the pure cap/verdict/profile-key rules, floor selection (auto/None/offload never lower a floor; resident needs a GPU
slot and a capping host, else the artifact floor with the reason), fail-closed wait_admission (missing required readings, STOP re-checked
before success, the cap fixed at the first sample), the exercise/resource split in the final state (a pass under an exceeded or unknown
bound is untried), no doubled-floor re-queue for contained cells, refusal-before-load classification, the guard contract of
requests_catalog.run and the ingest gate (older receipts unchanged)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools' / 'matrix_kit'))
import requests_catalog  # noqa: E402
import run_cells  # noqa: E402
import slots  # noqa: E402

failures = []


def expect(cond, msg):
    if not cond:
        failures.append(msg)


def probes(torch=None, onnx=None, openvino=None):
    return {'torch': torch or {'available': True, 'gpus': [], 'mps': False, 'xpu': []}, 'onnx': onnx or {'available': True, 'providers': ['CPUExecutionProvider'], 'dml_adapters': []},
            'openvino': openvino or {'available': False, 'error': 'not installed'}, 'nvidia_smi': {}}


A = 'GPU-aaaaaaaa-0000-0000-0000-000000000001'
B = 'GPU-bbbbbbbb-0000-0000-0000-000000000002'
C = 'GPU-cccccccc-0000-0000-0000-000000000003'
two_4090 = [{'index': 0, 'name': 'NVIDIA GeForce RTX 4090', 'capability': '8.9', 'uuid': A, 'family': 'cuda', 'total_memory': 1},
            {'index': 1, 'name': 'NVIDIA GeForce RTX 4090', 'capability': '8.9', 'uuid': B, 'family': 'cuda', 'total_memory': 1}]
pascal = {'index': 2, 'name': 'NVIDIA GeForce GTX 1080', 'capability': '6.1', 'uuid': C, 'family': 'cuda', 'total_memory': 1}

# Windows x64 + NVIDIA + DirectML adapters (NVIDIA and AMD iGPU)
orig = slots.host_platform
slots.host_platform = lambda: ('windows', 'x64')
try:
    det = slots.detect(probes=probes(torch={'available': True, 'gpus': two_4090 + [pascal], 'mps': False, 'xpu': []},
                                     onnx={'available': True, 'providers': ['DmlExecutionProvider', 'CPUExecutionProvider'],
                                           'dml_adapters': [{'name': 'NVIDIA GeForce RTX 4090', 'device_id': 0, 'luid': 'x'}, {'name': 'AMD Radeon(TM) Graphics', 'device_id': 1, 'luid': 'y'}]}))
    ids = [s['slot'] for s in det['slots']]
    expect(ids.count('win-x64-cpu') == 1, 'windows x64 must expose exactly one CPU slot')
    expect(ids.count('win-x64-nvidia-cuda') == 3, f'three NVIDIA GPUs must yield three CUDA slot entries: {ids}')
    expect('win-x64-nvidia-directml' in ids and 'win-x64-amd-igpu-directml' in ids, f'DirectML adapters must map by vendor and integrated/discrete: {ids}')
    expect(not any(s in ids for s in ('linux-x64-cpu', 'macos-arm64-mps', 'win-arm64-qnn-npu')), 'no foreign-OS slots')
    groups = det['gpu_groups']
    expect(len(groups) == 1 and sorted(groups[0]['gpus']) == sorted([A, B]) and groups[0]['slot'] == 'win-x64-nvidia-cuda', f'only the homogeneous 4090 pair forms a group: {groups}')
    det2 = slots.detect(allowed_gpus=[A], probes=probes(torch={'available': True, 'gpus': two_4090, 'mps': False, 'xpu': []}))
    expect([s['device'] for s in det2['slots'] if s['slot'] == 'win-x64-nvidia-cuda'] == [A], 'allowed set filters GPUs')
    expect(det2['gpu_groups'] == [] and det2['host']['excluded_gpus'] == [B], 'a single allowed GPU forms no group and the excluded GPU is recorded')
    # Windows ARM64 with QNN
    slots.host_platform = lambda: ('windows', 'arm64')
    det3 = slots.detect(probes=probes(onnx={'available': True, 'providers': ['QNNExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'], 'dml_adapters': [{'name': 'Qualcomm(R) Adreno(TM) X1-45 GPU', 'device_id': 0, 'luid': 'z'}]}))
    ids3 = [s['slot'] for s in det3['slots']]
    expect(sorted(ids3) == ['win-arm64-adreno-directml', 'win-arm64-cpu', 'win-arm64-qnn-npu'], f'Snapdragon host slots: {ids3}')
    # macOS arm64 with MPS and Core ML
    slots.host_platform = lambda: ('macos', 'arm64')
    det4 = slots.detect(probes=probes(torch={'available': True, 'gpus': [], 'mps': True, 'xpu': []}, onnx={'available': True, 'providers': ['CoreMLExecutionProvider', 'CPUExecutionProvider'], 'dml_adapters': []}))
    expect(sorted(s['slot'] for s in det4['slots']) == ['macos-arm64-ane', 'macos-arm64-cpu', 'macos-arm64-mps'], 'Apple Silicon slots')
    # Linux aarch64 with a GB10
    slots.host_platform = lambda: ('linux', 'arm64')
    det5 = slots.detect(probes=probes(torch={'available': True, 'gpus': [{'index': 0, 'name': 'NVIDIA GB10', 'capability': '12.1', 'uuid': A, 'family': 'cuda', 'total_memory': 1}], 'mps': False, 'xpu': []}))
    expect(sorted(s['slot'] for s in det5['slots']) == ['linux-arm64-cpu', 'linux-arm64-nvidia'], 'Spark slots')
    # Linux x64 ROCm and Intel (OpenVINO GPU + NPU is Windows-only slot; on Linux both go to linux-x64-intel-gpu)
    slots.host_platform = lambda: ('linux', 'x64')
    det6 = slots.detect(probes=probes(torch={'available': True, 'hip': '6.4', 'gpus': [{'index': 0, 'name': 'AMD Radeon RX 7900 XTX', 'capability': 'gfx1100', 'uuid': 'HIP-1', 'family': 'rocm', 'total_memory': 1}], 'mps': False, 'xpu': []},
                                      openvino={'available': True, 'devices': [{'address': 'GPU', 'name': 'Intel(R) Arc(TM) A770 Graphics (dGPU)', 'type': 'discrete'}]}))
    expect(sorted(s['slot'] for s in det6['slots']) == ['linux-x64-amd-rocm', 'linux-x64-cpu', 'linux-x64-intel-gpu'], f'Linux ROCm + Intel slots: {[s["slot"] for s in det6["slots"]]}')
    # torch missing is reported, not guessed
    slots.host_platform = lambda: ('windows', 'x64')
    det7 = slots.detect(probes=probes(torch={'available': False, 'error': 'ModuleNotFoundError'}))
    expect([s['slot'] for s in det7['slots']] == ['win-x64-cpu'] and any(u['runtime'] == 'torch' for u in det7['unprobed']), 'missing torch leaves GPU slots unclaimed and is listed as unprobed')
    # PyTorch XPU on Windows x64 maps to the Intel GPU slots (discrete Arc -> the dGPU slot; Iris/UHD/Lunar Lake Arc 1x0V -> the iGPU
    # slot), by the OpenVINO probe's device type when it saw the same GPU, else by name (tools/profiles.json xpu covers windows-x86_64).
    xpu3 = [{'index': 0, 'name': 'Intel(R) Arc(TM) B580 Graphics'}, {'index': 1, 'name': 'Intel(R) Arc(TM) 130V GPU'}, {'index': 2, 'name': 'Intel(R) Iris(R) Xe Graphics'}]
    det8 = slots.detect(probes=probes(torch={'available': True, 'gpus': [], 'mps': False, 'xpu': xpu3}))
    got8 = [(s['slot'], s['device'], s['runtime']) for s in det8['slots'] if s['runtime'] == 'torch-xpu']
    expect(got8 == [('win-x64-intel-dgpu-openvino', 'xpu:0', 'torch-xpu'), ('win-x64-intel-igpu', 'xpu:1', 'torch-xpu'), ('win-x64-intel-igpu', 'xpu:2', 'torch-xpu')], f'Windows XPU slot mapping by name: {got8}')
    det9 = slots.detect(probes=probes(torch={'available': True, 'gpus': [], 'mps': False, 'xpu': [{'index': 0, 'name': 'Intel(R) Arc(TM) A770 Graphics'}]},
                                      openvino={'available': True, 'devices': [{'address': 'GPU', 'name': 'Intel(R) Arc(TM) A770 Graphics (dGPU)', 'type': 'discrete'}]}))
    expect(sorted((s['slot'], s['runtime']) for s in det9['slots']) == [('win-x64-cpu', 'cpu'), ('win-x64-intel-dgpu-openvino', 'openvino-gpu'), ('win-x64-intel-dgpu-openvino', 'torch-xpu')], f'the OpenVINO device type and the XPU mapping agree on the dGPU slot: {[(s["slot"], s["runtime"]) for s in det9["slots"]]}')
    expect(slots.intel_gpu_discrete('Intel(R) Arc(TM) 130V GPU', [{'address': 'GPU', 'name': 'Intel(R) Arc(TM) 130V GPU (iGPU)', 'type': 'integrated'}]) is False
           and slots.intel_gpu_discrete('Intel(R) Arc(TM) Graphics') is False and slots.intel_gpu_discrete('Intel(R) UHD Graphics 630') is False
           and slots.intel_gpu_discrete('Intel(R) Arc(TM) B580 Graphics') is True, 'integrated Arc (Lunar Lake 130V, Meteor Lake Arc Graphics), UHD and Iris are the iGPU slot; Arc A/B-series are discrete')
    slots.host_platform = lambda: ('linux', 'x64')
    det10 = slots.detect(probes=probes(torch={'available': True, 'gpus': [], 'mps': False, 'xpu': xpu3[:1]}))
    expect([(s['slot'], s['device']) for s in det10['slots'] if s['runtime'] == 'torch-xpu'] == [('linux-x64-intel-gpu', 'xpu:0')], 'Linux keeps its single Intel GPU slot for XPU')
finally:
    slots.host_platform = orig

# home_files: a source entry's files are copied into the cell's TITANOS_HOME (bare names only, missing sources are harness errors)
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / 'music-runtime.json'
    src.write_text('{"schema_version": 1}', encoding='utf-8')
    home = Path(tmp) / 'cell'
    home.mkdir()
    placed = run_cells.place_home_files({'home_files': {'music-runtime.json': str(src)}}, home)
    expect((home / 'music-runtime.json').read_text(encoding='utf-8') == '{"schema_version": 1}' and placed['music-runtime.json']['bytes'] == 21
           and len(placed['music-runtime.json']['sha256']) == 64, f'home_files copies the receipt into TITANOS_HOME and records it: {placed}')
    expect(run_cells.place_home_files({}, home) == {}, 'no home_files -> nothing placed')
    for bad in ({'home_files': {'sub/x.json': str(src)}}, {'home_files': {'gone.json': str(Path(tmp) / 'missing.json')}}):
        try:
            run_cells.place_home_files(bad, home)
            expect(False, f'home_files must refuse {bad}')
        except RuntimeError:
            pass

# ingest: result folders sharing one run id stay distinct families (folder-qualified); the same folder re-ingested replaces itself
import subprocess
with tempfile.TemporaryDirectory() as tmp:
    raw = Path(tmp) / 'raw.json'
    raw.write_text(json.dumps({'catalog': {}, 'families': [], 'mode_cells': []}), encoding='utf-8')
    cell = {'model_id': 'uci-digits-8x8-onnx', 'mode': 'single', 'state': 'pass', 'device': 'cpu', 'device_named': 'cpu', 'receipt': 'receipts/x.json', 'note': 'pass'}
    results = []
    # legacy-generation results (no kit_schema, utc before ingest.LEGACY_RESULT_BOUNDARY_UTC): the family and verdict mechanics under test
    # predate the detailed-receipt gate, which tests/check_ingest_gate_negative.py covers for current runs
    for folder, slot in (('receipts', 'win-arm64-cpu'), ('receipts-dml', 'win-arm64-adreno-directml')):
        d = Path(tmp) / folder
        d.mkdir()
        (d / 'RUN_RESULT.json').write_text(json.dumps({'run_id': 'fleet-SAME-20260925', 'utc': '2026-09-24T18:00:00+00:00', 'cells': [dict(cell, slot=slot)]}), encoding='utf-8')
        results += ['--result', str(d / 'RUN_RESULT.json')]
    # synthetic legacy results are not in the pinned legacy population (tools/matrix_kit/legacy_results.json): --legacy-run names them explicitly
    cmd = [sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'ingest.py'), '--raw', str(raw), '--evidence', 'E-000', '--legacy-run', 'fleet-SAME-20260925'] + results
    out = subprocess.run(cmd, capture_output=True, text=True)
    fams = [f['family'] for f in json.loads(raw.read_text(encoding='utf-8'))['families']]
    expect(out.returncode == 0 and fams == ['matrix-kit:fleet-SAME-20260925', 'matrix-kit:fleet-SAME-20260925#receipts-dml'], f'colliding run ids become folder-qualified families: rc {out.returncode} {fams} {out.stderr[-300:]}')
    out2 = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(raw.read_text(encoding='utf-8'))
    expect(out2.returncode == 0 and [f['family'] for f in data['families']] == fams and len(data['mode_cells']) == 2, f'the same folders re-ingested replace their own families only: {[f["family"] for f in data["families"]]} mode_cells {len(data["mode_cells"])}')
    # a re-ingest with a verdict file upgrades the run's own mode cells in place (verified true) instead of keeping the stale copies
    vp = Path(tmp) / 'verdicts.json'
    vp.write_text(json.dumps({'uci-digits-8x8-onnx|win-arm64-cpu|single': {'upheld': True, 'corrected_state': 'pass', 'reasoning': 'ok'}}), encoding='utf-8')
    out3 = subprocess.run(cmd + ['--verdicts', str(vp)], capture_output=True, text=True)
    data = json.loads(raw.read_text(encoding='utf-8'))
    ver = [c.get('verified') for c in data['mode_cells'] if c['slot'] == 'win-arm64-cpu']
    expect(out3.returncode == 0 and len(data['mode_cells']) == 2 and ver == [True], f"re-ingest replaces the run's mode cells with the verified copy: {ver} total {len(data['mode_cells'])}")

# build_manifest carries the per-model kit fields (request, env, home_files) from the sources file into the manifest entry
import build_manifest  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    srcp = Path(tmp) / 'sources.json'
    srcp.write_text(json.dumps({'ace-step-1.5-xl-sft': {'present': True, 'path': tmp, 'bytes': 1, 'spec': {}, 'request': {'job_seconds': 10}, 'env': {'X': '1'}, 'home_files': {'music-runtime.json': str(srcp)}}}), encoding='utf-8')
    outp = Path(tmp) / 'manifest.json'
    r = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'build_manifest.py'), '--sources', str(srcp), '--out', str(outp)], capture_output=True, text=True, cwd=str(ROOT))
    man = json.loads(outp.read_text(encoding='utf-8')) if outp.exists() else {}
    ent = next((m for m in man.get('models', []) if m['model_id'] == 'ace-step-1.5-xl-sft'), None)
    expect(r.returncode == 0 and ent is not None and ent['request'] == {'job_seconds': 10} and ent['env'] == {'X': '1'} and ent['home_files'] == {'music-runtime.json': str(srcp)},
           f'build_manifest keeps request, env and home_files: rc {r.returncode} {ent} {r.stderr[-200:]}')

# a llama-cpp row on the macOS MPS slot is placed with device 'auto' (Titanos resolves MTL0 from --list-devices; 'mps' is not a native address)
mps_slot = {'slot': 'macos-arm64-mps', 'device': 'mps', 'device_named': 'Apple M1 (MPS)', 'runtime': 'torch-mps'}
gguf_mps = {'model_id': 'gguf', 'task': 'chat', 'backend': 'llama-cpp', 'path': str(ROOT / 'tests'), 'spec': {'native': {'executable': '/opt/llama/llama-server', 'model_file': 'model.gguf', 'gpu_layers': 'all'}}}
cfg_m, desc_m = run_cells.build_config(gguf_mps, mps_slot, 'single', None, 'auto')
expect(cfg_m.models['chat'].device == 'auto' and cfg_m.models['chat'].native.gpu_layers == 'all' and desc_m['device'] == 'auto' and 'MTL0' in desc_m.get('served_by', ''),
       f"llama-cpp on macos-arm64-mps uses device auto with full offload: {cfg_m.models['chat'].device} {desc_m}")
cfg_t, _ = run_cells.build_config({'model_id': 'x', 'task': 'chat', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {}}, mps_slot, 'single', None, 'auto')
expect(cfg_t.models['chat'].device == 'mps', 'transformers rows keep the mps device on the MPS slot')

# ingest verdict keys: a verifier's device label reduces to the receipt's sanitised device tag (review finding #22)
import ingest  # noqa: E402
expect(ingest.normalize_key('m|win-x64-intel-dgpu-openvino@GPU.0 (Intel Arc B580, OpenVINO GPU)|single') == 'm|win-x64-intel-dgpu-openvino@GPU.0|single'
       and ingest.normalize_key('m|s@native:SYCL0|replicas') == 'm|s@native_SYCL0|replicas' and ingest.normalize_key('m|s|single') == 'm|s|single'
       and ingest.device_tag({'device': 'native:SYCL0'}) == 'native_SYCL0' and ingest.device_tag({'device': 'GPU.0'}) == 'GPU.0',
       'normalize_key strips the verifier label and sanitises like device_tag')

# the admission margin is a documented, recorded flag (default 2 GiB); the summary carries floors and floor_margin_gib
h = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'run_cells.py'), '--help'], capture_output=True, text=True, cwd=str(ROOT))
expect('--floor-margin-gib' in h.stdout and "add_argument('--floor-margin-gib', type=float, default=2.0" in (ROOT / 'tools' / 'matrix_kit' / 'run_cells.py').read_text(encoding='utf-8'), f'run_cells documents --floor-margin-gib with default 2.0: {h.stdout[-200:]}')
src_rc = (ROOT / 'tools' / 'matrix_kit' / 'run_cells.py').read_text(encoding='utf-8')
expect("'floor_margin_gib': a.floor_margin_gib" in src_rc and '(2 << 30)' not in src_rc.split('def main')[1], 'the fixed 2 GiB margin is gone from admission and the summary records the margin')

# every matrix task has a request
matrix = json.loads((ROOT / 'docs' / 'execution_matrix.json').read_text(encoding='utf-8'))
tasks = {r['task'] for r in matrix['rows']}
missing = sorted(t for t in tasks if t not in requests_catalog.CATALOG)
expect(not missing, f'tasks without a catalogue request: {missing}')

# configuration refusals: by-design refusals classify as na; multi-GPU modes without a group are refused with an unsupported message
entry = {'task': 'pair-classification', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {}}
slot = {'slot': 'linux-x64-nvidia-cuda', 'device': A, 'device_named': 'x', 'runtime': 'torch-cuda', 'gpu': {'capability': '8.9'}}
try:
    run_cells.build_config(entry, slot, 'model_sharding', [A, B], 'auto')
    failures.append('pair-classification sharding must be refused by Titanos')
except ValueError as exc:
    expect(bool(run_cells.UNSUPPORTED.search(str(exc))), f'refusal text must classify as unsupported: {exc}')
try:
    run_cells.build_config({'task': 'chat', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {}}, slot, 'tensor_parallel', None, 'auto')
    failures.append('tensor parallel without a GPU group must be refused')
except ValueError as exc:
    expect(bool(run_cells.UNSUPPORTED.search(str(exc))), 'no-group refusal classifies as unsupported')
cfg, desc = run_cells.build_config({'task': 'chat', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {}}, slot, 'tensor_parallel', [A, B], 'auto')
expect(cfg.workers['group'].mode == 'tensor_parallel' and cfg.models['chat'].parallelism == 'tensor_parallel', 'chat tensor parallel config builds with a two-GPU group')
cfg2, _ = run_cells.build_config({'task': 'embeddings', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {}}, {'slot': 'win-x64-cpu', 'device': 'cpu', 'device_named': 'cpu', 'runtime': 'cpu'}, 'replicas', None, 'auto')
expect(len(cfg2.workers) == 2 and all(w.models == ['embeddings'] for w in cfg2.workers.values()), 'replicas mode uses two workers for the same model')
# construction classification: no execution path -> na with the construction reason; Titanos's own refusal text is kept when present
tf = {'model_id': 'm', 'task': 'chat', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {}}
dml = {'slot': 'win-x64-amd-igpu-directml', 'device': 'dml:1', 'device_named': 'AMD Radeon(TM) Graphics (DirectML)', 'runtime': 'onnx-dml'}
c = run_cells.no_path_cell(tf, dml, 'single')
expect(c['state'] == 'na' and c['note'].startswith('not applicable by construction: backend transformers has no execution path on slot runtime onnx-dml'), f'no-path cell must be na by construction: {c}')
c = run_cells.no_path_cell(tf, dml, 'replicas', 'backend transformers does not support device dml:1')
expect(c['state'] == 'na' and 'refuses backend transformers on this device by design' in c['note'], f'a Titanos refusal keeps its own text: {c}')
expect('transformers' not in run_cells.COMPAT['onnx-dml'] and 'onnx' in run_cells.COMPAT['cpu'], 'COMPAT table shape')
expect('onnx' in run_cells.COMPAT['openvino-gpu'] and 'onnx' in run_cells.COMPAT['openvino-npu'], 'OpenVINO runtimes read model.onnx: the onnx backend has a path there (E-002/E-026/E-028/E-037)')
spec = json.loads((ROOT / 'docs' / 'execution_matrix_spec.json').read_text(encoding='utf-8'))
expect({'openvino GPU', 'openvino NPU'} <= set(spec['backend_runtime_map']['backends']['onnx']), 'spec backend_runtime_map grants onnx the OpenVINO GPU/NPU tokens like COMPAT')
adm = spec['multi_gpu_modes_rule']['admission']
expect(set(adm['one_device_backends']) == run_cells.ONE_DEVICE_BACKENDS and set(adm['single_device_tasks']) == run_cells.SINGLE_DEVICE_TASKS
       and set(adm['tensor_parallel']['backends']) == run_cells.TENSOR_PARALLEL_BACKENDS and set(adm['tensor_parallel']['tasks']) == run_cells.TENSOR_PARALLEL_TASKS
       and set(adm['model_sharding']['backends']) == run_cells.MODEL_SHARDING_BACKENDS and adm['single_device_task_lines'] == run_cells.SINGLE_DEVICE_TASK_LINES,
       'spec multi_gpu_modes_rule.admission and run_cells admission constants (including the per-task refusing lines) must agree')
expect({'diarization', 'vad'} <= run_cells.SINGLE_DEVICE_TASKS, 'diarization (config.py:74-77, 334-335) and vad (170-172) are single-device tasks')
expect(spec['backend_runtime_map']['cpu_path_tokens']['onnx-diffusers'] == ['onnx CPU', 'diffusers CPU'], 'the spec models onnx-diffusers as needing onnxruntime and torch on the CPU slot, like RUNTIME_PROBE')
expect({'llama.cpp SYCL', 'llama.cpp OpenVINO'} <= set(spec['backend_runtime_map']['backends']['llama-cpp']) and {'llama.cpp OpenVINO', 'llama.cpp OpenCL'} <= set(spec['backend_runtime_map']['tokens_without_kit_runtime'])
       and spec['backend_runtime_map']['kit_runtimes'].get('native-sycl') == ['llama.cpp SYCL'] and run_cells.COMPAT['native-sycl'] == {'llama-cpp'},
       'the catalogued Intel OpenVINO and Adreno OpenCL llama.cpp profiles are llama-cpp tokens the kit does not probe; the SYCL profile is the kit runtime native-sycl (llama-cpp only)')
# native-sycl: the --list-devices parser, the Intel GPU slot mapping (discrete vs integrated by name / OpenVINO type), the non-Intel
# refusal, and the config the kit builds (native:<address> device, the SYCL executable, full offload; multi-GPU modes refused)
listing = 'Available devices:\n  SYCL0: Intel(R) Arc(TM) B580 Graphics (12116 MiB, 12073 MiB free)\n  SYCL1: Intel(R) Arc(TM) A770 Graphics (16256 MiB, 16236 MiB free)\n  SYCL2: Intel(R) Arc(TM) 130V GPU (7914 MiB, 7000 MiB free)\n  SYCL3: NVIDIA GeForce RTX 4090 (24000 MiB, 23000 MiB free)\n'
devs = slots.parse_native_devices(listing)
missing = slots.probe_native_sycl(str(ROOT / 'tests' / 'no-such-llama-server.exe'))
expect(missing.get('available') is False and 'not found' in missing.get('error', ''), f'a missing SYCL executable is reported, not raised: {missing}')


class _Proc:
    def __init__(self, out, rc=0):
        self.stdout, self.stderr, self.returncode = out, '', rc


probed = slots.probe_native_sycl(__file__, runner=lambda argv, **kw: _Proc(listing) if argv[1:] == ['--list-devices'] else _Proc('', 2))
expect(probed.get('available') is True and [d['address'] for d in probed['devices']] == ['SYCL0', 'SYCL1', 'SYCL2', 'SYCL3'] and probed['executable'] == os.path.abspath(__file__), f'the probe runs --list-devices and keeps the parsed devices: {probed}')
empty = slots.probe_native_sycl(__file__, runner=lambda argv, **kw: _Proc('Available devices:\n', 0))
expect(empty.get('available') is False and 'named no SYCL device' in empty.get('error', ''), f'a listing without SYCL devices is unavailable with the reason: {empty}')
expect([d['address'] for d in devs] == ['SYCL0', 'SYCL1', 'SYCL2', 'SYCL3'] and devs[0]['name'] == 'Intel(R) Arc(TM) B580 Graphics' and devs[0]['total_mib'] == 12116 and devs[0]['free_mib'] == 12073, f'--list-devices parse: {devs}')
slots.host_platform = lambda: ('windows', 'x64')
try:
    pr = probes()
    pr['native_sycl'] = {'available': True, 'executable': 'C:/kit/native/sycl/llama-server.exe', 'devices': devs}
    det_s = slots.detect(probes=pr, native_sycl='ignored-when-probed')
    got_s = [(s['slot'], s['device'], s['runtime'], s['executable']) for s in det_s['slots'] if s['runtime'] == 'native-sycl']
    expect(got_s == [('win-x64-intel-dgpu-openvino', 'native:SYCL0', 'native-sycl', 'C:/kit/native/sycl/llama-server.exe'), ('win-x64-intel-dgpu-openvino', 'native:SYCL1', 'native-sycl', 'C:/kit/native/sycl/llama-server.exe'),
                     ('win-x64-intel-igpu', 'native:SYCL2', 'native-sycl', 'C:/kit/native/sycl/llama-server.exe')], f'SYCL devices map to the Intel GPU slots by name: {got_s}')
    expect(any(u['runtime'] == 'native-sycl' and 'SYCL3' in u['reason'] for u in det_s['unprobed']), 'a non-Intel SYCL device is reported, never assigned')
    sycl_slot = next(s for s in det_s['slots'] if s['device'] == 'native:SYCL0')
    gguf = {'model_id': 'gguf', 'task': 'chat', 'backend': 'llama-cpp', 'path': str(ROOT / 'tests'), 'spec': {'native': {'executable': 'C:/kit/native/cpu/llama-server.exe', 'model_file': 'model.gguf', 'gpu_layers': 'all'}}}
    cfg_s, desc_s = run_cells.build_config(gguf, sycl_slot, 'single', None, 'auto')
    ms = cfg_s.models['chat']
    expect(ms.device == 'native:SYCL0' and ms.native.executable == 'C:/kit/native/cpu/llama-server.exe'.replace('cpu', 'sycl') and ms.native.gpu_layers == 'all' and desc_s['device'] == 'native:SYCL0',
           f'a llama-cpp row on a native-sycl slot runs the SYCL executable on the named native device with full offload: {ms.device} {ms.native.executable} {ms.native.gpu_layers}')
    cfg_r, _ = run_cells.build_config(gguf, sycl_slot, 'replicas', None, 'auto')
    expect(set(cfg_r.workers) == {'r1', 'r2'} and all(w.device == 'native:SYCL0' for w in cfg_r.workers.values()), 'replicas on a native-sycl slot are two workers on the same named native device')
    expect('transformers' not in run_cells.COMPAT['native-sycl'] and 'onnx' not in run_cells.COMPAT['native-sycl'], 'native-sycl carries llama-cpp only')
    expect('native-sycl' not in run_cells.GROUP_RUNTIMES and not any('native-sycl' in v for v in run_cells.EXPLICIT_GROUP_RUNTIMES.values()), 'native-sycl forms no multi-GPU group')
finally:
    slots.host_platform = orig
# the full kit runtime <-> spec token mapping: for every backend and every kit runtime, COMPAT membership equals holding one of the
# runtime's tokens in backend_runtime_map (spec kit_runtimes); every slot token belongs to a kit runtime, is a cpu token, or is
# declared to have no kit runtime (tokens_without_kit_runtime, filled by the builder alone)
bm = spec['backend_runtime_map']
kit_rt = bm['kit_runtimes']
expect(set(kit_rt) == set(run_cells.COMPAT), f'spec kit_runtimes and COMPAT must name the same kit runtimes: {sorted(set(kit_rt) ^ set(run_cells.COMPAT))}')
for rt, toks in kit_rt.items():
    for backend, allowed in bm['backends'].items():
        in_compat = backend in run_cells.COMPAT.get(rt, set())
        in_spec = bool(set(toks) & set(allowed))
        expect(in_compat == in_spec, f'COMPAT[{rt}] and backend_runtime_map disagree on {backend}: kit {in_compat}, spec {in_spec} (tokens {toks})')
expect(set(bm['backends']) == set(run_cells.COMPAT['cpu']), f'every backend the kit knows is in the spec map and vice versa: {sorted(set(bm["backends"]) ^ set(run_cells.COMPAT["cpu"]))}')
kit_tokens = {t for toks in kit_rt.values() for t in toks}
slot_tokens = {t for s_ in spec['platform_slots'] for t in s_['runtimes']}
expect(slot_tokens <= kit_tokens | set(bm['cpu_tokens']) | set(bm['tokens_without_kit_runtime']), f'slot runtime tokens outside every kit runtime and not declared as such: {sorted(slot_tokens - kit_tokens - set(bm["cpu_tokens"]) - set(bm["tokens_without_kit_runtime"]))}')
expect(all(t in slot_tokens and t not in kit_tokens for t in bm['tokens_without_kit_runtime']), 'tokens_without_kit_runtime must be slot tokens that no kit runtime maps')
# Titanos's admission of the kit's multi-GPU modes, cross-checked against its own config validation with a two-GPU group: every
# combination mode_refusal calls refused must be refused by Titanos (with the cited message), every admitted one must build.
native = {'native': {'model_file': 'model.gguf'}}
probes = [  # (backend, task, spec fields, mode, expected phrase in Titanos's refusal or None when admitted)
    ('transformers', 'chat', {}, 'tensor_parallel', None), ('transformers', 'chat', {}, 'model_sharding', None),
    ('transformers', 'embeddings', {}, 'tensor_parallel', None), ('transformers', 'vision', {}, 'model_sharding', None),
    ('transformers', 'transcription', {}, 'tensor_parallel', 'tensor parallelism'), ('transformers', 'transcription', {}, 'model_sharding', None),
    ('transformers', 'speech', {}, 'tensor_parallel', 'tensor parallelism'), ('transformers', 'speech', {}, 'model_sharding', None),
    ('transformers', 'translation', {}, 'model_sharding', 'one device per worker'), ('transformers', 'sentiment', {}, 'tensor_parallel', 'one device per worker'),
    ('transformers', 'rerank', {}, 'model_sharding', 'one device per worker'), ('transformers', 'pair-classification', {}, 'model_sharding', 'one device per worker'),
    ('transformers', 'object-detection', {}, 'model_sharding', 'one device per worker'), ('transformers', 'depth-estimation', {}, 'model_sharding', 'one device per worker'),
    ('transformers', 'image-to-text', {}, 'model_sharding', 'one device per worker'), ('transformers', 'image-segmentation', {}, 'model_sharding', 'one Transformers device'),
    ('diffusers', 'image', {}, 'tensor_parallel', 'tensor parallelism'), ('diffusers', 'image', {}, 'model_sharding', None),
    ('diffusers', 'video', {}, 'tensor_parallel', 'tensor parallelism'), ('diffusers', 'video', {}, 'model_sharding', None),
    ('qwen3-tts', 'speech', {}, 'model_sharding', 'one device per worker'), ('qwen3-tts', 'speech', {}, 'tensor_parallel', 'tensor parallelism'),
    ('xtts', 'speech', {'allow_pickle': True}, 'model_sharding', 'one device per worker'),
    ('onnx', 'tensor', {}, 'model_sharding', 'not implemented'), ('onnx', 'embeddings', {}, 'tensor_parallel', 'not implemented'),
    ('onnx', 'vad', {}, 'tensor_parallel', 'single-device CPU ONNX adapter'), ('onnx', 'vad', {}, 'model_sharding', 'single-device CPU ONNX adapter'),
    ('transformers', 'diarization', {}, 'model_sharding', 'on one CPU'), ('sherpa-onnx', 'diarization', {}, 'tensor_parallel', 'on one CPU'),
    ('openvino', 'embeddings', {}, 'model_sharding', 'not implemented'), ('openvino-genai', 'chat', {}, 'model_sharding', 'not implemented'),
    ('sherpa-onnx', 'speech', {}, 'model_sharding', 'one CPU speech archive'), ('ace-step', 'music', {}, 'model_sharding', 'one isolated trusted runtime'),
    ('llama-cpp', 'chat', native, 'model_sharding', 'llama-cpp placement'), ('llama-cpp', 'chat', native, 'tensor_parallel', 'llama-cpp placement'),
    ('onnx-diffusers', 'image', {}, 'model_sharding', 'one explicit cpu/dml device'), ('tesseract', 'image-to-text', {}, 'model_sharding', 'one CPU image-to-text worker'),
]
for backend, task, fields, mode, phrase in probes:
    entry = {'model_id': 'p', 'task': task, 'backend': backend, 'path': str(ROOT / 'tests'), 'spec': dict(fields)}
    refusal = run_cells.mode_refusal(backend, task, mode)
    expect((refusal is None) == (phrase is None), f'mode_refusal({backend}, {task}, {mode}) = {refusal!r} disagrees with the expected Titanos verdict')
    try:
        run_cells.build_config(entry, slot, mode, [A, B], 'auto')
        outcome = None
    except ValueError as exc:
        outcome = str(exc)
    if phrase is None:
        expect(outcome is None, f'Titanos must admit {mode} for {backend} {task} with a two-GPU group: {outcome}')
    else:
        expect(outcome is not None and phrase in outcome, f'Titanos must refuse {mode} for {backend} {task} with {phrase!r}: {outcome}')
# multi-GPU modes without a group: na where the slot runtime cannot form one, untried on a GPU slot and for the explicit CPU/XPU TP group
cpu_slot = {'slot': 'win-x64-cpu', 'device': 'cpu', 'device_named': 'cpu', 'runtime': 'cpu'}
c = run_cells.missing_group_cell(tf, cpu_slot, 'model_sharding')
expect(c['state'] == 'na' and c['note'].startswith('not applicable by construction: model_sharding needs a homogeneous multi-GPU group'), f'CPU model sharding is na: {c}')
c = run_cells.missing_group_cell(tf, cpu_slot, 'tensor_parallel')
expect(c['state'] == 'untried' and 'explicit cpu rank group' in c['note'], f'CPU tensor parallel stays untried (explicit Gloo group is accepted by Titanos): {c}')
c = run_cells.missing_group_cell(tf, dml, 'tensor_parallel')
expect(c['state'] == 'na', f'DirectML tensor parallel without a group is na: {c}')
c = run_cells.missing_group_cell(tf, slot, 'model_sharding')
expect(c['state'] == 'untried' and c['note'].startswith('not run: model_sharding needs a homogeneous multi-GPU group'), f'GPU slot without a group stays untried: {c}')
# modes Titanos refuses by design are na before the group question, on every slot runtime
for entry, mode, phrase in ((dict(tf, task='image', backend='diffusers'), 'tensor_parallel', 'only for Transformers chat, vision and embeddings'),
                            (dict(tf, task='translation'), 'model_sharding', 'single-device workers only'),
                            (dict(tf, task='speech', backend='qwen3-tts'), 'model_sharding', 'one device per worker'),
                            (dict(tf, task='tensor', backend='onnx'), 'tensor_parallel', 'only for Transformers chat, vision and embeddings'),
                            (dict(tf, task='chat', backend='llama-cpp'), 'model_sharding', 'requires Transformers or Diffusers'),
                            (dict(tf, task='depth-estimation'), 'tensor_parallel', 'single-device workers only (src/titanos/config.py:211-212, 336-337)'),
                            (dict(tf, task='diarization'), 'model_sharding', 'single-device workers only (src/titanos/config.py:74-77 (diarization.py:33-38), 334-335)'),
                            (dict(tf, task='vad', backend='onnx'), 'tensor_parallel', 'single-device workers only (src/titanos/config.py:170-172)')):
    for s_ in (cpu_slot, slot, dml):
        c = run_cells.missing_group_cell(entry, s_, mode)
        expect(c['state'] == 'na' and c['note'].startswith('not applicable by construction: Titanos refuses') and phrase in c['note'], f"refused-by-design {mode} for {entry['backend']} {entry['task']} on {s_['runtime']} must be na with the citation: {c}")
c = run_cells.missing_group_cell(dict(tf, task='transcription'), slot, 'model_sharding')
expect(c['state'] == 'untried', f'an admitted mode (transformers transcription sharding) on a GPU slot without a group stays untried: {c}')
# a backend whose runtime package the host probe found absent is untried with the reason (an import error is not an execution), never fail
c = run_cells.missing_runtime_cell(tf, cpu_slot, 'single', 'torch', {'available': False, 'error': "ModuleNotFoundError(\"No module named 'torch'\")"})
expect(c['state'] == 'untried' and c['note'].startswith('not run: runtime torch for backend transformers is not installed on this host (torch probe: ModuleNotFoundError') and 'not an execution' in c['note'], f'missing runtime cell: {c}')
expect({run_cells.RUNTIME_PROBE[b] for b in ('transformers', 'diffusers', 'xtts', 'qwen3-tts', 'ace-step')} == {('torch',)} and run_cells.RUNTIME_PROBE['onnx-diffusers'] == ('onnx', 'torch')
       and run_cells.RUNTIME_PROBE['openvino-genai'] == ('openvino',) and not {'sherpa-onnx', 'tesseract', 'llama-cpp'} & set(run_cells.RUNTIME_PROBE),
       'RUNTIME_PROBE covers the torch, onnxruntime and openvino backends (onnx-diffusers needs both onnxruntime and torch, onnx_images.py:123) and leaves the self-contained runtimes alone')
# dtype policy: Pascal gets float16 then float32; ONNX backends stay on auto
expect(run_cells.dtype_plan({'runtime': 'torch-cuda', 'gpu': {'capability': '6.1'}}, {'backend': 'transformers'}, 'auto-policy') == ['float16', 'float32'], 'Pascal dtype plan')
expect(run_cells.dtype_plan({'runtime': 'torch-cuda', 'gpu': {'capability': '6.1'}}, {'backend': 'onnx'}, 'auto-policy') == ['auto'], 'ONNX dtype plan')
# --- contained admission, fail-closed admission, resource qualification vs exercise, ingest gate (containment.py + run_cells + ingest) ---
import containment  # noqa: E402
import hashlib  # noqa: E402
GIB = 1 << 30
# pure rules: the cap (measured profile x 1.5 + margin, else first-run cap = first available - reserve, never negative), the verdict
# (unknown whenever readings are incomplete or no peak exists; a cap hit or a peak at the cap is exceeded) and the canonical profile key
expect(containment.derive_job_limit(40 * GIB, 6 * GIB, None, 2 * GIB) == (34 * GIB, 'first_run_containment_cap'), 'first-run cap = first available - reserve')
expect(containment.derive_job_limit(40 * GIB, 6 * GIB, 4 * GIB, 2 * GIB) == (8 * GIB, 'measured_profile'), 'measured profile cap = 1.5 x peak + margin')
expect(containment.derive_job_limit(5 * GIB, 6 * GIB, None, 2 * GIB) == (0, 'first_run_containment_cap'), 'the cap never goes negative')
for reserve, margin in ((0, 2 * GIB), (-5 * GIB, 2 * GIB), (6 * GIB, 0), (6 * GIB, -1)):
    try:
        containment.derive_job_limit(40 * GIB, reserve, None, margin)
        expect(False, f'a non-positive reserve or margin must be refused by the pure rule: reserve {reserve} margin {margin}')
    except ValueError:
        pass
expect(containment.qualification_verdict(10, 20, False, True) == 'within' and containment.qualification_verdict(20, 20, False, True) == 'exceeded'
       and containment.qualification_verdict(5, 20, True, True) == 'exceeded' and containment.qualification_verdict(5, 20, False, False) == 'unknown'
       and containment.qualification_verdict(None, 20, False, True) == 'unknown' and containment.qualification_verdict(5, None, False, True) == 'unknown', 'qualification_verdict rules')
k1 = containment.profile_key({'repo': 'x'}, 'a' * 64, 'b' * 64, 'resident', 'bfloat16', 1)
expect(len(k1) == 64 and k1 == containment.profile_key({'repo': 'x'}, 'a' * 64, 'b' * 64, 'resident', 'bfloat16', 1)
       and k1 != containment.profile_key({'repo': 'x'}, 'a' * 64, 'b' * 64, 'resident', 'bfloat16', 2) and k1 != containment.profile_key({'repo': 'x'}, 'a' * 64, 'b' * 64, 'resident', 'float16', 1)
       and k1 == hashlib.sha256(json.dumps({'matrix_identity': {'repo': 'x'}, 'receipt_sha256': 'a' * 64, 'adapter_sha256': 'b' * 64, 'placement': 'resident', 'dtype': 'bfloat16', 'copies': 1},
                                           sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'profile_key is the sha256 of canonical JSON and changes with copies and dtype')
# floor selection: contained only for an explicit resident spec on a GPU slot of a capping host; auto/None/offload never lower; a CPU
# slot and a host without containment keep the artifact floor (reason recorded); the profile key reads the music-runtime receipt
with tempfile.TemporaryDirectory() as tmp:
    dest = Path(tmp) / 'runtime'
    dest.mkdir()
    adapter = dest / 'ace_adapter.py'
    adapter.write_text('x', encoding='utf-8')
    receipt = Path(tmp) / 'music-runtime.json'
    receipt.write_text(json.dumps({'adapter': str(adapter), 'destination': str(dest), 'files': {'ace_adapter.py': {'sha256': 'c' * 64, 'bytes': 1}}}), encoding='utf-8')
    ace = {'model_id': 'ace', 'task': 'music', 'backend': 'ace-step', 'matrix_identity': {'repo': 'ACE-Step/ACE-Step-v1.5'}, 'path': tmp, 'spec': {'placement': 'resident'}, 'home_files': {'music-runtime.json': str(receipt)}}
    plan = run_cells.contained_admission_plan(ace, A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {}, available=lambda: True)
    expect(plan['kind'] == 'contained_resident' and plan['inputs']['adapter_sha256'] == 'c' * 64 and plan['inputs']['receipt_sha256'] == hashlib.sha256(receipt.read_bytes()).hexdigest()
           and plan['profile_peak_bytes'] is None and plan['inputs']['copies'] == 1 and plan['inputs_detail'] == 'ok', f'resident + GPU + containment -> contained_resident keyed by receipt and adapter: {plan}')
    plan_r = run_cells.contained_admission_plan(ace, A, 'replicas', 'bfloat16', 6 * GIB, 2 * GIB, {plan['profile_key']: {'measured_peak_commit_bytes': 5 * GIB}}, available=lambda: True)
    expect(plan_r['inputs']['copies'] == 2 and plan_r['profile_key'] != plan['profile_key'] and plan_r['profile_peak_bytes'] is None, 'replicas is another profile key (two copies) and does not inherit the single profile')
    # a profile lowers the cap only when bound to its within-bound pass receipt (same key, peak and cap); anything else is a first run
    kit = Path(tmp) / 'kit'
    (kit / 'receipts').mkdir(parents=True)
    profiles_path = kit / 'ADMISSION_PROFILES.json'
    bound_receipt = {'run_id': 'r1', 'state': 'pass', 'resource_qualification': {'kind': 'contained_resident', 'bound': 'job_commit', 'verdict': 'within', 'profile_key': plan['profile_key'],
                                                                                 'measured_peak_job_commit_bytes': 5 * GIB, 'job_commit_limit_bytes': 34 * GIB}}
    (kit / 'receipts' / 'ace.json').write_text(json.dumps(bound_receipt), encoding='utf-8')
    entry_ok = {'measured_peak_commit_bytes': 5 * GIB, 'job_commit_limit_bytes': 34 * GIB, 'receipt': 'receipts/ace.json', 'run_id': 'r1'}
    plan_p = run_cells.contained_admission_plan(ace, A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {plan['profile_key']: entry_ok}, available=lambda: True, profiles_path=profiles_path)
    expect(plan_p['profile_peak_bytes'] == 5 * GIB and plan_p['reason'] == 'measured profile' and plan_p['profiles_usable'] is True
           and plan_p['profile_provenance']['receipt_sha256'] == hashlib.sha256((kit / 'receipts' / 'ace.json').read_bytes()).hexdigest() and plan_p['profile_provenance']['run_id'] == 'r1',
           f'a measured profile bound to its within-bound pass receipt is used and its provenance recorded: {plan_p}')
    plan_bare = run_cells.contained_admission_plan(ace, A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {plan['profile_key']: {'measured_peak_commit_bytes': 5 * GIB}}, available=lambda: True, profiles_path=profiles_path)
    expect(plan_bare['profile_peak_bytes'] is None and plan_bare['reason'].startswith('first run: the profile for this key is rejected') and plan_bare['kind'] == 'contained_resident',
           f'a profile without a receipt reference never lowers the cap: {plan_bare}')
    plan_nopath = run_cells.contained_admission_plan(ace, A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {plan['profile_key']: entry_ok}, available=lambda: True)
    expect(plan_nopath['profile_peak_bytes'] is None and plan_nopath['reason'].startswith('first run'), f'without a profiles file no profile is used: {plan_nopath}')
    for mutate, why in ((lambda r: r.update(state='untried'), 'receipt not a pass'), (lambda r: r['resource_qualification'].update(verdict='exceeded'), 'receipt verdict exceeded'),
                        (lambda r: r['resource_qualification'].update(measured_peak_job_commit_bytes=4 * GIB), 'hand-edited profile peak differs from the receipt'),
                        (lambda r: r['resource_qualification'].update(profile_key='x' * 64), 'receipt for another key'),
                        (lambda r: r['resource_qualification'].update(job_commit_limit_bytes=30 * GIB), 'receipt cap differs')):
        bad = json.loads(json.dumps(bound_receipt))
        mutate(bad)
        (kit / 'receipts' / 'ace.json').write_text(json.dumps(bad), encoding='utf-8')
        p = run_cells.contained_admission_plan(ace, A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {plan['profile_key']: entry_ok}, available=lambda: True, profiles_path=profiles_path)
        expect(p['profile_peak_bytes'] is None and 'rejected' in p['reason'], f'{why}: the profile must be rejected: {p["reason"]}')
    (kit / 'receipts' / 'ace.json').unlink()
    p = run_cells.contained_admission_plan(ace, A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {plan['profile_key']: entry_ok}, available=lambda: True, profiles_path=profiles_path)
    expect(p['profile_peak_bytes'] is None and 'not found' in p['reason'], f'a profile whose receipt is gone is a first run: {p["reason"]}')
    for spec in ({'placement': 'auto'}, {}, {'placement': 'offload'}, {'placement': None}):
        p = run_cells.contained_admission_plan(dict(ace, spec=spec), A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {}, available=lambda: True)
        expect(p['kind'] == 'artifact_floor' and 'never lowers the floor' in p['reason'], f'placement {spec} must keep the artifact floor: {p}')
    p = run_cells.contained_admission_plan(ace, None, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {}, available=lambda: True)
    expect(p['kind'] == 'artifact_floor' and 'without a GPU' in p['reason'], f'a CPU slot keeps the artifact floor: {p}')
    p = run_cells.contained_admission_plan(ace, A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {}, available=lambda: False)
    expect(p['kind'] == 'artifact_floor' and p['reason'].startswith('containment_unavailable'), f'resident without containment keeps the full floor with the reason: {p}')
    p = run_cells.contained_admission_plan(dict(ace, home_files={}), A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {}, available=lambda: True)
    expect(p['kind'] == 'contained_resident' and p['inputs']['receipt_sha256'] is None and 'no music-runtime.json' in p['inputs_detail'], 'a missing receipt keeps the key deterministic with the detail recorded')
    expect(p['profiles_usable'] is False and p['profile_peak_bytes'] is None and p['reason'].startswith('first run: profile inputs incomplete'), f'incomplete profile inputs: first-run cap only, no profile recorded: {p}')
    (kit / 'receipts' / 'ace.json').write_text(json.dumps(dict(bound_receipt, resource_qualification=dict(bound_receipt['resource_qualification'], profile_key=p['profile_key']))), encoding='utf-8')
    p2 = run_cells.contained_admission_plan(dict(ace, home_files={}), A, 'single', 'bfloat16', 6 * GIB, 2 * GIB, {p['profile_key']: entry_ok}, available=lambda: True, profiles_path=profiles_path)
    expect(p2['profile_peak_bytes'] is None and p2['profiles_usable'] is False, f'even a bound profile is ignored when the key inputs are incomplete: {p2}')
    # the guard of an attempt: the reserve, the cap and the GPU-starvation margin apply to contained cells only; every cell reads STOP and the readings
    g_art = run_cells.cell_guard(None, A, {'kind': 'artifact_floor', 'floors': {'commit': GIB, 'gpu': 4096}, 'margin_bytes': 2 * GIB}, {'kind': 'artifact_floor', 'host_reserve_bytes': None, 'job_commit_limit_bytes': None})
    g_con = run_cells.cell_guard(None, A, {'kind': 'contained_resident', 'floors': {'commit': GIB, 'gpu': 4096}, 'margin_bytes': 2 * GIB}, {'kind': 'contained_resident', 'host_reserve_bytes': 6 * GIB, 'job_commit_limit_bytes': 34 * GIB})
    expect(g_art.margin_bytes is None and g_art.reserve_bytes is None and g_art.job_limit is None and g_art.gpu_uuid == A and g_art.phys_reader is run_cells.available_physical_bytes,
           'an artifact-floor cell has no GPU-starvation margin, no reserve and no cap (a low GPU reading is recorded, not tripped)')
    expect(g_con.margin_bytes == 2 * GIB and g_con.reserve_bytes == 6 * GIB and g_con.job_limit == 34 * GIB, 'a contained cell carries the margin, the reserve and the cap')
    # the spec's placement reaches the generated ModelSpec unchanged (build_manifest passes spec through)
    cfg_a, _ = run_cells.build_config(ace, slot, 'single', None, 'auto')
    expect(getattr(cfg_a.models['music'], 'placement', None) == 'resident', f"placement passes through to the ModelSpec: {getattr(cfg_a.models['music'], 'placement', None)!r}")
# wait_admission: fail closed on missing required readings, STOP at the top and immediately before success, deferral unchanged,
# contained admission fixes the cap at the first sample and requires cap + reserve; a stale tuple caller fails loudly
logs = []
quiet = lambda s: None  # noqa: E731
r = run_cells.wait_admission({'commit': GIB, 'gpu': None}, None, None, logs.append, commit_reader=lambda: None, gpu_reader=lambda u: None, sleep=quiet)
expect(r['admitted'] is False and r['reason'] == 'resource_reading_unavailable' and len(r['samples']) == 3, f'a missing commit reading is never good; three in a row end the wait: {r}')
r = run_cells.wait_admission({'commit': None, 'gpu': 4096}, A, None, logs.append, commit_reader=lambda: 50 * GIB, gpu_reader=lambda u: None, sleep=quiet)
expect(r['admitted'] is False and r['reason'] == 'resource_reading_unavailable', f'a missing GPU reading on a GPU slot with a GPU floor is never good: {r}')
r = run_cells.wait_admission({'commit': GIB, 'gpu': 4096}, None, None, logs.append, commit_reader=lambda: 50 * GIB, gpu_reader=lambda u: None, sleep=quiet)
expect(r['admitted'] is True and r['reason'] is None and len(r['samples']) == 3, f'a CPU slot needs no GPU reading: {r}')
with tempfile.TemporaryDirectory() as tmp:
    stop = Path(tmp) / 'STOP'
    reads = []

    def late_stop():
        reads.append(1)
        if len(reads) == 3:
            stop.write_text('x', encoding='utf-8')  # appears after the third iteration's top-of-loop check
        return 50 * GIB
    r = run_cells.wait_admission({'commit': GIB, 'gpu': None}, None, str(stop), logs.append, commit_reader=late_stop, gpu_reader=lambda u: None, sleep=quiet)
    expect(r['admitted'] is False and r['reason'] == 'stop' and len(reads) == 3, f'STOP is re-checked immediately before success: {r}')
    r = run_cells.wait_admission({'commit': GIB, 'gpu': None}, None, str(stop), logs.append, commit_reader=lambda: 50 * GIB, gpu_reader=lambda u: None, sleep=quiet)
    expect(r['admitted'] is False and r['reason'] == 'stop' and r['samples'] == [], f'STOP at the top of the loop: {r}')
r = run_cells.wait_admission({'commit': 100 * GIB, 'gpu': None}, None, None, logs.append, probe_only=True, commit_reader=lambda: 50 * GIB, gpu_reader=lambda u: None, sleep=quiet)
expect(r['admitted'] is None and r['reason'] == 'deferred', f'probe-only deferral unchanged: {r}')
contained = {'reserve_bytes': 6 * GIB, 'profile_peak_bytes': None, 'margin_bytes': 2 * GIB}
phys_ok = lambda: 100 * GIB  # noqa: E731
r = run_cells.wait_admission({'commit': None, 'gpu': 8192}, A, None, logs.append, contained=contained, commit_reader=lambda: 40 * GIB, gpu_reader=lambda u: 20000, sleep=quiet, phys_reader=phys_ok)
expect(r['admitted'] is True and r['job_limit'] == 34 * GIB and r['job_limit_source'] == 'first_run_containment_cap' and r['floors'] == {'commit': 40 * GIB, 'gpu': 8192}
       and all(s['available_physical_bytes'] == 100 * GIB for s in r['samples']),
       f'contained first run: cap = first available - reserve, commit floor = cap + reserve, GPU floor unchanged, physical memory sampled: {r}')
r = run_cells.wait_admission({'commit': None, 'gpu': 8192}, A, None, logs.append, contained=dict(contained, profile_peak_bytes=4 * GIB), commit_reader=lambda: 40 * GIB, gpu_reader=lambda u: 20000, sleep=quiet, phys_reader=phys_ok)
expect(r['admitted'] is True and r['job_limit'] == 8 * GIB and r['job_limit_source'] == 'measured_profile' and r['floors']['commit'] == 14 * GIB, f'contained with a measured profile: {r}')
r = run_cells.wait_admission({'commit': None, 'gpu': 8192}, A, None, logs.append, contained=contained, commit_reader=lambda: 7 * GIB, gpu_reader=lambda u: 20000, sleep=quiet, phys_reader=phys_ok)
expect(r['admitted'] is False and r['reason'] == 'contained_budget_below_minimum' and r['job_limit'] == GIB, f'a cap below 2 x margin is refused: {r}')
r = run_cells.wait_admission({'commit': None, 'gpu': 8192}, A, None, logs.append, probe_only=True, contained=contained, commit_reader=lambda: 40 * GIB, gpu_reader=lambda u: 100, sleep=quiet, phys_reader=phys_ok)
expect(r['admitted'] is None and r['reason'] == 'deferred', f'the GPU floor still defers a contained cell: {r}')
r = run_cells.wait_admission({'commit': None, 'gpu': None}, A, None, logs.append, contained=contained, commit_reader=lambda: None, gpu_reader=lambda u: 20000, sleep=quiet, phys_reader=phys_ok)
expect(r['admitted'] is False and r['reason'] == 'resource_reading_unavailable', 'contained admission requires the commit reading')
r = run_cells.wait_admission({'commit': None, 'gpu': None}, A, None, logs.append, contained=contained, commit_reader=lambda: 40 * GIB, gpu_reader=lambda u: 20000, sleep=quiet, phys_reader=lambda: None)
expect(r['admitted'] is False and r['reason'] == 'resource_reading_unavailable', f'contained admission requires the physical-memory reading (the commit cap does not see file-backed mappings): {r}')
r = run_cells.wait_admission({'commit': None, 'gpu': None}, A, None, logs.append, probe_only=True, contained=contained, commit_reader=lambda: 40 * GIB, gpu_reader=lambda u: 20000, sleep=quiet, phys_reader=lambda: 5 * GIB)
expect(r['admitted'] is None and r['reason'] == 'deferred', f'physical memory below the reserve defers a contained cell even with commit to spare: {r}')
r = run_cells.wait_admission({'commit': GIB, 'gpu': None}, None, None, logs.append, commit_reader=lambda: 50 * GIB, gpu_reader=lambda u: None, sleep=quiet, phys_reader=lambda: None)
expect(r['admitted'] is True and r['samples'][-1]['available_physical_bytes'] is None, 'an artifact-floor cell records the physical reading but does not require it')
try:
    run_cells.wait_admission({'commit': None, 'gpu': None}, A, None, logs.append, contained=dict(contained, reserve_bytes=-5 * GIB), commit_reader=lambda: 40 * GIB, gpu_reader=lambda u: 20000, sleep=quiet, phys_reader=phys_ok)
    expect(False, 'a negative reserve must never inflate the cap')
except ValueError:
    pass
try:
    admitted, samples = run_cells.wait_admission({'commit': GIB, 'gpu': None}, None, None, logs.append, commit_reader=lambda: 50 * GIB, gpu_reader=lambda u: None, sleep=quiet)
    expect(False, 'the old (admitted, samples) tuple contract is gone; a stale caller must fail loudly')
except ValueError:
    pass
expect(run_cells.ADMISSION_ERRORS['resource_reading_unavailable'].startswith('resource_reading_unavailable:') and run_cells.ADMISSION_ERRORS['stop'] == 'blocked_by_host_resources: STOP before admission', 'admission error texts')
# the final state keeps the exercise (output correctness) apart from the resource qualification: pass only when both hold
fs = run_cells.final_attempt_state
expect(fs({'state': 'pass', 'output': {'job': 'j'}}, {'verdict': 'within'}) == ('pass', None) and fs({'state': 'pass'}, {'verdict': 'not_contained'}) == ('pass', None), 'exercise pass + within/not_contained -> pass')
expect(fs({'state': 'fail', 'error': 'RuntimeError: x'}, {'verdict': 'not_contained'}) == ('fail', 'RuntimeError: x'), 'exercise fail under the artifact floor stays fail')
st, er = fs({'state': 'pass', 'output': {'wav_bytes': 241732}}, {'verdict': 'exceeded', 'measured_peak_job_commit_bytes': 30, 'job_commit_limit_bytes': 20, 'limit_hit': True})
expect(st == 'untried' and er.startswith('resource_qualification_exceeded: contained job commit peak 30 reached the limit 20') and 'exercise pass' in er, f'a valid WAV under an exceeded bound is untried, never pass: {st} {er}')
st, er = fs({'state': 'pass'}, {'verdict': 'unknown', 'reason': 'resource readings incomplete: nvidia-smi missing'})
expect(st == 'untried' and er.startswith('resource_qualification_unknown: resource readings incomplete'), f'unknown qualification is untried even after a pass: {er}')
st, er = fs({'state': 'fail', 'error': 'load refused HTTP 503: MemoryError'}, {'verdict': 'exceeded', 'measured_peak_job_commit_bytes': 30, 'job_commit_limit_bytes': 20, 'limit_hit': True})
expect(st == 'untried' and er.startswith('host_resources_bound_exceeded: contained job commit peak 30 reached the limit 20'), f'cap hit with a failed exercise: {er}')
trip_error = 'budget_violated_during_request: contained job commit peak 30 reached the limit 20'
expect(fs({'state': 'untried', 'error': trip_error}, {'verdict': 'exceeded'}) == ('untried', trip_error), 'a guard trip keeps its own reason')
expect(fs({'state': 'untried', 'error': 'containment_unavailable: job cap readback mismatch'}, {'verdict': 'unknown'}) == ('untried', 'containment_unavailable: job cap readback mismatch'), 'a containment failure never runs and stays untried')
# the doubled-floor OOM re-queue never applies to a contained cell; an artifact-floor cell is re-queued once
expect(run_cells.oom_disposition('CUDA error: out of memory', False, False) == 'requeue' and run_cells.oom_disposition('CUDA error: out of memory', False, True) == 'untried'
       and run_cells.oom_disposition('RuntimeError: paging file is too small', True, False) == 'untried' and run_cells.oom_disposition('RuntimeError: shape mismatch', False, False) is None
       and run_cells.oom_disposition(None, True, False) is None, 'oom_disposition: requeue once under the artifact floor, never for contained cells')
# an adapter refusal before any weight loaded is na (resident requires a CUDA device) or untried (blocked_by_device_resources), never fail
c = run_cells.classify_refusal('RuntimeError: load refused HTTP 503: {"error": "placement_refused_before_load: resident placement requires a CUDA device; this worker was given cpu"}')
expect(c is not None and c[0] == 'na' and c[1].startswith('not applicable: the adapter refused resident placement before loading any weights because it requires a CUDA device'), f'CUDA-required refusal is na: {c}')
c = run_cells.classify_refusal('RuntimeError: load refused HTTP 503: placement_refused_before_load: GPU-x has 8 GiB free; resident bf16 residency needs 21 GiB')
expect(c is not None and c[0] == 'untried' and c[1].startswith('blocked_by_device_resources: '), f'capacity refusal is untried: {c}')
expect(run_cells.classify_refusal('CUDA error: out of memory') is None and run_cells.classify_refusal(None) is None, 'other errors are not refusals')
# requests_catalog.run honours the guard in the music wait loop, reports the owned job id, and is unchanged without a guard


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._body, self.text = code, body, json.dumps(body)

    def json(self):
        return self._body


class _MusicClient:
    def __init__(self, statuses):
        self.statuses = list(statuses)

    def post(self, path, **kw):
        return _Resp(202, {'id': 'job-1'})

    def get(self, path):
        return _Resp(200, {'status': self.statuses.pop(0)})


import time as _time  # noqa: E402
_sleep = _time.sleep
_time.sleep = lambda s: None
try:
    seen = []
    out = requests_catalog.run('music', _MusicClient(['running', 'completed']), 'music', {'request': {'steps': 1}}, guard=lambda **info: seen.append(info))
    expect(out['job'] == 'job-1' and out['status'] == 'completed' and seen == [{'music_job': 'job-1'}, {'music_job': 'job-1'}], f'the guard runs on every poll with the owned job id: {seen} {out}')

    def tripping(**info):
        raise containment.GuardTrip('stopped', 'request', 'STOP file present')
    try:
        requests_catalog.run('music', _MusicClient(['running', 'running']), 'music', {}, guard=tripping)
        expect(False, 'a raising guard aborts the wait')
    except containment.GuardTrip as exc:
        expect(getattr(exc, 'music_job', None) == 'job-1', 'the raised trip names the owned job so the cell can cancel exactly it')
    try:
        requests_catalog.run('music', _MusicClient(['failed']), 'music', {})
        expect(False, 'a failed job raises')
    except RuntimeError as exc:
        expect(getattr(exc, 'music_job', None) == 'job-1' and 'job-1' in str(exc), f'the failure names the job: {exc}')

    class _RefusingClient(_MusicClient):
        def get(self, path):
            return _Resp(200, {'id': 'job-1', 'object': 'job', 'task': 'music', 'model': 'music', 'status': 'failed', 'created_at': 1, 'execution_started': 1, 'progress': {'stage': 'x' * 900},
                               'error': 'placement_refused_before_load: resident placement: device free 8 GiB < required 21 GiB'})
    try:
        requests_catalog.run('music', _RefusingClient([]), 'music', {})
        expect(False, 'a failed job raises')
    except RuntimeError as exc:
        expect('placement_refused_before_load' in str(exc) and run_cells.classify_refusal(str(exc)) is not None and run_cells.classify_refusal(str(exc))[0] == 'untried',
               f"the job's error text survives whatever precedes it in the record, so the refusal classifies: {str(exc)[:200]}")
    expect(requests_catalog.run('music', _MusicClient(['done']), 'music', {})['status'] == 'done', 'without a guard the music request behaves as before')

    class _ChatClient:
        def post(self, path, **kw):
            return _Resp(200, {'choices': [{'message': {'content': 'Hello.'}}]})
    expect(requests_catalog.run('chat', _ChatClient(), 'chat', {}, guard=lambda **info: None) == {'text': 'Hello.'}, 'a request without a wait loop ignores the guard and runs exactly as before')
finally:
    _time.sleep = _sleep
# ingest gate, LEGACY generation (no kit_schema, utc before ingest.LEGACY_RESULT_BOUNDARY_UTC): a receipt with resource_qualification
# exceeded/unknown enters as untried with the reason (never pass/fail) whatever the exercise; within (bound to a placement identity) and
# not_contained keep their state; receipts without the block (older kits) are unchanged. Current runs (kit_schema 2) need the detailed
# receipt: tests/check_ingest_gate_negative.py, and the tail of this block.
GATE_KEY = hashlib.sha256(b'gate-profile').hexdigest()  # a real sha256 hex digest, the only profile_key shape production accepts
with tempfile.TemporaryDirectory() as tmp:
    raw = Path(tmp) / 'raw.json'
    raw.write_text(json.dumps({'catalog': {}, 'families': [], 'mode_cells': []}), encoding='utf-8')
    d = Path(tmp) / 'kit'
    (d / 'receipts').mkdir(parents=True)

    def kit_cell(mid, state, rq, schema=None, run_id='gate-run'):
        # the binding fields run_cells.cell_receipt writes (run_id, model_id, slot dict, mode, spec_placement); a current run requires them
        body = {'state': state, 'exercise': {'state': state, **({'error': 'RuntimeError: x'} if state == 'fail' else {})}, 'run_id': run_id, 'model_id': mid,
                'slot': {'slot': 'win-x64-nvidia-cuda', 'device': A}, 'mode': 'single', 'spec_placement': 'resident' if (rq or {}).get('kind') == 'contained_resident' else None}
        if schema is not None:
            body['receipt_schema'] = schema
        if rq is not None:
            body['resource_qualification'] = rq
        (d / 'receipts' / f'{mid}.json').write_text(json.dumps(body), encoding='utf-8')
        row = {'model_id': mid, 'slot': 'win-x64-nvidia-cuda', 'mode': 'single', 'state': state, 'device': A, 'device_named': 'x', 'receipt': f'receipts/{mid}.json', 'note': f'{state} note'}
        if rq:
            row.update(resource_kind=rq.get('kind'), resource_verdict=rq.get('verdict'))
        return row
    thin_within = {'kind': 'contained_resident', 'verdict': 'within', 'profile_key': GATE_KEY, 'receipt_sha256': hashlib.sha256(b'r').hexdigest(), 'adapter_sha256': hashlib.sha256(b'a').hexdigest(),
                   'placement': 'resident', 'dtype': 'bfloat16', 'copies': 1}
    rows = [kit_cell('m-exceeded', 'pass', {'kind': 'contained_resident', 'verdict': 'exceeded', 'measured_peak_job_commit_bytes': 30, 'job_commit_limit_bytes': 20, 'limit_hit': True}),
            kit_cell('m-unknown', 'pass', {'kind': 'contained_resident', 'verdict': 'unknown', 'reason': 'resource readings incomplete'}),
            kit_cell('m-within', 'pass', thin_within),
            kit_cell('m-within-noid', 'pass', {'kind': 'contained_resident', 'verdict': 'within', 'profile_key': GATE_KEY}), kit_cell('m-old', 'pass', None),
            kit_cell('m-artifact', 'fail', {'kind': 'artifact_floor', 'verdict': 'not_contained'}), kit_cell('m-fail-exceeded', 'fail', {'kind': 'contained_resident', 'verdict': 'exceeded'}),
            kit_cell('m-na', 'na', None)]
    (d / 'RUN_RESULT.json').write_text(json.dumps({'run_id': 'gate-run', 'utc': '2026-09-25T05:00:00+00:00', 'cells': rows}), encoding='utf-8')
    # verifier flags: only a boolean True upholds a pass; a string 'true' (or any non-bool) never verifies it
    gate_verdicts = d / 'verdicts.json'
    gate_verdicts.write_text(json.dumps({'m-within|win-x64-nvidia-cuda|single': {'upheld': True, 'corrected_state': 'pass', 'reasoning': 'ok'},
                                         'm-within-noid|win-x64-nvidia-cuda|single': {'upheld': True, 'corrected_state': 'pass', 'reasoning': 'ok'},
                                         'm-old|win-x64-nvidia-cuda|single': {'upheld': 'true', 'corrected_state': 'partial', 'reasoning': 'string flag'}}), encoding='utf-8')
    # a synthetic legacy result is not in the pinned population (tools/matrix_kit/legacy_results.json), so --legacy-run names it explicitly
    out = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'ingest.py'), '--raw', str(raw), '--evidence', 'E-000', '--result', str(d / 'RUN_RESULT.json'),
                          '--verdicts', str(gate_verdicts), '--legacy-run', 'gate-run'], capture_output=True, text=True)
    data = json.loads(raw.read_text(encoding='utf-8')) if out.returncode == 0 else {'mode_cells': [], 'families': []}
    states = {c['model_id']: (c['state'], c['note']) for c in data['mode_cells']}
    expect(out.returncode == 0 and states.get('m-exceeded', ('', ''))[0] == 'untried' and states['m-exceeded'][1].startswith('[single] resource_qualification_exceeded: not accepted into the matrix (contained_resident; contained job commit peak 30 vs limit 20')
           and 'exercise pass' in states['m-exceeded'][1], f"exceeded qualification enters as untried with the reason: rc {out.returncode} {states.get('m-exceeded')} {out.stderr[-300:]}")
    expect(states.get('m-unknown', ('',))[0] == 'untried' and 'resource_qualification_unknown' in states['m-unknown'][1], f'unknown qualification enters as untried: {states.get("m-unknown")}')
    expect(states.get('m-fail-exceeded', ('',))[0] == 'untried', 'an exceeded fail is untried too (resource evidence, not a model verdict)')
    expect(states.get('m-old', ('',))[0] == 'pass' and states.get('m-artifact', ('',))[0] == 'fail' and states.get('m-na', ('',))[0] == 'na',
           f'not_contained/no-block/na cells keep their state: {states}')
    # a contained_resident block is never legacy evidence: the legacy boundary covers old full-artifact receipts only, so even a within
    # block with its complete placement identity enters untried (the pre-boundary ingest accepted it on the identity alone)
    for mid in ('m-within', 'm-within-noid'):
        expect(states.get(mid, ('', ''))[0] == 'untried' and states[mid][1].startswith('[single] resource_qualification_legacy_contained: not accepted into the matrix (contained_resident;')
               and 'exercise pass' in states[mid][1], f'a legacy contained within-bound cell enters as untried with the exercise kept as evidence: {states.get(mid)}')
    expect('identity missing receipt_sha256' in states.get('m-within-noid', ('', ''))[1] and 'identity missing' not in states.get('m-within', ('', ''))[1], f'the legacy note names the missing identity: {states.get("m-within-noid")}')
    fam = next((f for f in data['families'] if f['family'] == 'matrix-kit:gate-run'), {'cells': []})
    fstates = {c['model_id']: c for c in fam['cells']}
    expect(fstates.get('m-exceeded', {}).get('state') == 'untried' and fstates['m-exceeded'].get('verified') is None and fstates.get('m-old', {}).get('state') == 'pass' and fstates.get('m-old', {}).get('verified') is False,
           f'the family cells carry the gate too; an old receipt is untouched: {fstates.get("m-exceeded")} {fstates.get("m-old")}')
    fverdicts = {v['model_id']: v for v in fam.get('verdicts', [])}
    expect(fstates.get('m-within', {}).get('verified') is None and 'm-within' not in fverdicts and fstates.get('m-old', {}).get('verified') is False
           and fverdicts.get('m-old', {}).get('upheld') is False and 'verifier flag not boolean' in (fstates.get('m-old', {}).get('note') or ''),
           f"no verdict binds to a gated cell; a string 'true' leaves a pass unverified: {fstates.get('m-within')} {fstates.get('m-old')} {fverdicts}")
    expect(fam.get('legacy_basis') == 'legacy_run_flag' and fam.get('kit_schema') == 1, f'the family records how the legacy generation was decided: {{k: v for k, v in fam.items() if k != "cells"}}')
    counts = json.loads(out.stdout) if out.returncode == 0 else {}
    expect(counts.get('gated_resource_qualification') == 5 and counts.get('cells') == 8 and counts.get('legacy_runs') == ['gate-run'], f'ingest reports the gated cells and the legacy generation: {counts}')
    # without --legacy-run the same unmarked, back-dated result is refused: its hash is not in the pinned legacy population
    refused = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'ingest.py'), '--raw', str(raw), '--evidence', 'E-000', '--result', str(d / 'RUN_RESULT.json')], capture_output=True, text=True)
    expect(refused.returncode != 0 and 'not in the pinned legacy population' in refused.stderr and '--legacy-run gate-run' in refused.stderr,
           f'an unmarked back-dated result whose sha256 is not pinned is refused: rc {refused.returncode} {refused.stderr[-300:]}')
    # the same rows as a CURRENT run (kit_schema 2): the identity-only within block and the block-less receipt are no longer evidence;
    # only a receipt_schema 2 receipt with the complete measured block (or an artifact_floor not_contained block) enters as pass/fail
    raw.write_text(json.dumps({'catalog': {}, 'families': [], 'mode_cells': []}), encoding='utf-8')
    # the artifact_floor block the kit writes for a non-resident spec on a GPU slot: the plan reason, floors and the launched tree
    artifact_block = {'kind': 'artifact_floor', 'verdict': 'not_contained', 'reason': ingest.artifact_floor_reason(None, True), 'floors': {'commit': 4 << 30, 'gpu': 4096}, 'samples': [],
                      'cleanup': {'launched': True, 'active_processes_after': None, 'job_handle_closed': None, 'forced_cleanup': False}}
    rows2 = [kit_cell('c-within', 'pass', thin_within, schema=2, run_id='gate-run-2'), kit_cell('c-old', 'pass', None, schema=2, run_id='gate-run-2'),
             kit_cell('c-artifact', 'fail', artifact_block, schema=2, run_id='gate-run-2'), kit_cell('c-noschema', 'pass', thin_within, run_id='gate-run-2')]
    (d / 'RUN_RESULT.json').write_text(json.dumps({'run_id': 'gate-run-2', 'kit_schema': 2, 'utc': '2026-09-25T19:30:00+00:00', 'cells': rows2, 'host_reserve_bytes': 6 << 30, 'containment_available': True}), encoding='utf-8')
    out = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'ingest.py'), '--raw', str(raw), '--evidence', 'E-000', '--result', str(d / 'RUN_RESULT.json')], capture_output=True, text=True)
    data = json.loads(raw.read_text(encoding='utf-8')) if out.returncode == 0 else {'mode_cells': [], 'families': []}
    states = {c['model_id']: (c['state'], c['note']) for c in data['mode_cells']}
    expect(out.returncode == 0 and states.get('c-within', ('', ''))[0] == 'untried' and states['c-within'][1].startswith('[single] resource_qualification_evidence_incomplete: ') and 'exercise pass' in states['c-within'][1],
           f'current run: a within block with identity but no measurement is untried with the exercise retained: rc {out.returncode} {states.get("c-within")} {out.stderr[-300:]}')
    expect(states.get('c-old', ('', ''))[0] == 'untried' and states['c-old'][1].startswith('[single] receipt_schema_missing: '), f'current run: a receipt without the block is untried, never pass: {states.get("c-old")}')
    expect(states.get('c-noschema', ('', ''))[0] == 'untried' and states['c-noschema'][1].startswith('[single] receipt_schema_missing: '), f'current run: a receipt without receipt_schema is untried: {states.get("c-noschema")}')
    expect(states.get('c-artifact', ('',))[0] == 'fail', f'current run: an artifact_floor not_contained fail keeps its state: {states.get("c-artifact")}')
    # a result without kit_schema after the boundary is refused (fail closed) and writes nothing
    (d / 'RUN_RESULT.json').write_text(json.dumps({'run_id': 'gate-run-3', 'utc': '2026-09-25T19:30:00+00:00', 'cells': rows2}), encoding='utf-8')
    out = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'ingest.py'), '--raw', str(raw), '--evidence', 'E-000', '--result', str(d / 'RUN_RESULT.json')], capture_output=True, text=True)
    expect(out.returncode != 0 and 'ingest refused' in out.stderr and '--legacy-run gate-run-3' in out.stderr and 'matrix-kit:gate-run-3' not in raw.read_text(encoding='utf-8'),
           f'a post-boundary result without kit_schema is refused before anything is written: rc {out.returncode} {out.stderr[-300:]}')
# the new flags are documented; the summary records the reserve and the profiles path
expect('--host-reserve-gib' in h.stdout and '--admission-profiles' in h.stdout, f'run_cells documents --host-reserve-gib and --admission-profiles: {h.stdout[-300:]}')
# the reserve and the margin are enforced at the CLI boundary before any admission (a non-positive value never reaches derive_job_limit)
with tempfile.TemporaryDirectory() as tmp:
    for flags, phrase in ((['--host-reserve-gib', '-1'], '--host-reserve-gib must be positive'), (['--host-reserve-gib', '0'], '--host-reserve-gib must be positive'),
                          (['--floor-margin-gib', '0'], '--floor-margin-gib must be positive'), (['--floor-margin-gib', '-2'], '--floor-margin-gib must be positive')):
        bad = subprocess.run([sys.executable, str(ROOT / 'tools' / 'matrix_kit' / 'run_cells.py'), '--manifest', str(Path(tmp) / 'missing.json'), '--out', str(Path(tmp) / 'out')] + flags,
                             capture_output=True, text=True, cwd=str(ROOT))
        expect(bad.returncode != 0 and phrase in bad.stderr and not (Path(tmp) / 'out').exists(), f'{flags} must be refused before anything runs: rc {bad.returncode} {bad.stderr[-300:]}')
expect("'host_reserve_bytes': host_reserve_bytes" in src_rc and "'containment_available': containment_available" in src_rc, 'RUN_RESULT records the reserve and whether containment was available')
# WDDM-aware pre-launch check (run_cells.device_resident_estimate and its helpers): pure sizing from synthetic safetensors HEADERS (no tensor
# data is ever written or read), the nvidia-smi driver-model parse, dtype -> itemsize, fits vs does not fit on WDDM, estimate-only on TCC,
# fail closed on an unknown driver model or unreadable headers; and the pinned bundle's numbers when its headers are present on this host.
import struct  # noqa: E402
import containment  # noqa: E402
expect(all(isinstance(containment.CAP_SEMANTICS.get(k), list) and containment.CAP_SEMANTICS[k] and all(isinstance(x, str) for x in containment.CAP_SEMANTICS[k])
           for k in ('enforced', 'accounted_not_refused', 'unaccounted')) and containment.CAP_SEMANTICS['bound'] == 'job_commit',
       'containment.CAP_SEMANTICS names the enforced classes and the two gaps as non-empty string lists')
expect(any('safe_open' in x for x in containment.CAP_SEMANTICS['enforced']) and any('from_file' in x for x in containment.CAP_SEMANTICS['accounted_not_refused'])
       and any('anonymous' in x for x in containment.CAP_SEMANTICS['unaccounted']), 'the cap semantics name safe_open (enforced), torch from_file (accounted, not refused) and anonymous sections (unaccounted)')


def synthetic_safetensors(path, shapes, dtype='BF16'):
    """A header-only safetensors file: the 8-byte length and a table describing tensors of the given shapes; no tensor data follows."""
    table = {f't{i}': {'dtype': dtype, 'shape': list(shape), 'data_offsets': [0, 0]} for i, shape in enumerate(shapes)}
    header = json.dumps(table).encode('utf-8')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(struct.pack('<Q', len(header)) + header)


GIB = 1 << 30
PINNED_NUMEL = {'planner': 4189554176, 'dit': 4987310726, 'text_encoder': 595776512, 'vae': 168695426}  # parameters of the pinned ACE bundle (headers, 2026-09-25)
PINNED_BF16 = {'planner': 8379108352, 'dit': 9974621452, 'text_encoder': 1191553024, 'vae': 337390852}
PEAK_BF16 = PINNED_BF16['dit'] + PINNED_BF16['text_encoder'] + PINNED_BF16['vae']  # 11,503,565,328: max(planner, dit) + text encoder + vae
REQUIRED_BF16 = run_cells.BASE_RESIDENT_BYTES + PEAK_BF16 + run_cells.ACTIVATION_HEADROOM_BYTES
expect(run_cells.BASE_RESIDENT_BYTES == 3 * GIB and run_cells.ACTIVATION_HEADROOM_BYTES == 4 * GIB and PEAK_BF16 == 11503565328 and REQUIRED_BF16 == 19019758096,
       f'the pre-launch constants and the pinned bf16 peak set: base {run_cells.BASE_RESIDENT_BYTES} headroom {run_cells.ACTIVATION_HEADROOM_BYTES} peak {PEAK_BF16} required {REQUIRED_BF16}')
with tempfile.TemporaryDirectory() as tmp:
    bundle = Path(tmp) / 'bundle'
    synthetic_safetensors(bundle / 'planner' / 'model-00001-of-00002.safetensors', [[PINNED_NUMEL['planner'] - 1000]])
    synthetic_safetensors(bundle / 'planner' / 'model-00002-of-00002.safetensors', [[10, 100]])
    synthetic_safetensors(bundle / 'dit' / 'model-00001-of-00004.safetensors', [[2, 2493655363]], 'F32')  # stored F32, sized at the requested dtype
    synthetic_safetensors(bundle / 'dit' / 'model-00002-of-00004.safetensors', [[0]], 'F32')  # a zero-element tensor is legal
    synthetic_safetensors(bundle / 'core' / 'Qwen3-Embedding-0.6B' / 'model.safetensors', [[PINNED_NUMEL['text_encoder']]])
    synthetic_safetensors(bundle / 'core' / 'vae' / 'diffusion_pytorch_model.safetensors', [[PINNED_NUMEL['vae']]])
    sizing = run_cells.bundle_component_bytes(bundle, 2)
    expect(sizing['layout'] == 'resident-music-bundle' and sizing['problems'] == [] and {k: v['bytes'] for k, v in sizing['components'].items()} == PINNED_BF16
           and sizing['components']['planner']['files'] == 2 and sizing['components']['dit']['files'] == 2,
           f'synthetic headers size every component at 2 bytes per parameter: {sizing}')
    expect({k: v['bytes'] for k, v in run_cells.bundle_component_bytes(bundle, 4)['components'].items()} == {k: 2 * v for k, v in PINNED_BF16.items()}, 'float32 sizing doubles every component')
    comps = {k: v['bytes'] for k, v in sizing['components'].items()}
    fits = run_cells.device_resident_estimate(comps, REQUIRED_BF16, 'WDDM')
    expect(fits['fits'] is True and fits['blocked'] is False and fits['reason'] is None and fits['peak_set_bytes'] == PEAK_BF16 and fits['required_bytes'] == REQUIRED_BF16
           and fits['total_resident_bytes'] == sum(PINNED_BF16.values()) and fits['enforcement'] == 'wddm_commit_covers_device_resident_bytes',
           f'WDDM: a cap equal to base + peak set + headroom fits: {fits}')
    short = run_cells.device_resident_estimate(comps, REQUIRED_BF16 - 1, 'WDDM')
    expect(short['fits'] is False and short['blocked'] is True and short['reason'].startswith(f'blocked_by_host_resources: contained budget below the device-resident need on WDDM (cap {REQUIRED_BF16 - 1} < base + peak set + headroom {REQUIRED_BF16}')
           and str(PEAK_BF16) in short['reason'] and short['job_commit_limit_bytes'] == REQUIRED_BF16 - 1, f'WDDM: one byte short blocks the launch with the numbers in the reason: {short}')
    tcc = run_cells.device_resident_estimate(comps, REQUIRED_BF16 - 1, 'TCC')
    expect(tcc['blocked'] is False and tcc['reason'] is None and tcc['fits'] is False and tcc['enforcement'] == 'estimate_only_tcc' and tcc['required_bytes'] == REQUIRED_BF16,
           f'TCC: the estimate is recorded (device allocations are not commit there) and nothing is blocked: {tcc}')
    for bad in (None, 'N/A', 'wddm '):
        unknown = run_cells.device_resident_estimate(comps, 100 * GIB, bad)
        expect(unknown['blocked'] is True and unknown['reason'].startswith('resource_reading_unavailable: GPU driver model') and unknown['enforcement'] == 'unknown_driver_model',
               f'an unreadable driver model fails closed even with a roomy cap: {bad!r} -> {unknown}')
    r3_cap = 21664305152  # run r3's first-run cap: the peak-set rule alone would have admitted r3 (the resident set did not exhaust its cap; r3's crash cause is not proven)
    expect(run_cells.device_resident_estimate(comps, r3_cap, 'WDDM')['fits'] is True, 'the r3 cap covers base + bf16 peak set + headroom: the check is necessary, not sufficient')
    # the check itself: driver model read for the exact GPU, dtype resolution ('auto' -> bfloat16 on capability >= 8.0, float32 below or unknown)
    cuda_slot = {'slot': 'win-x64-nvidia-cuda', 'device': A, 'runtime': 'torch-cuda', 'gpu': {'capability': '12.0'}}
    expect(run_cells.resident_itemsize('auto', cuda_slot) == (2, 'bfloat16') and run_cells.resident_itemsize('auto', {'gpu': {'capability': '6.1'}}) == (4, 'float32')
           and run_cells.resident_itemsize('auto', {}) == (4, 'float32') and run_cells.resident_itemsize('float16', {}) == (2, 'float16') and run_cells.resident_itemsize('float32', cuda_slot) == (4, 'float32'),
           'resident_itemsize resolves auto like the adapter and sizes unknown capabilities at float32')
    asked = []

    def wddm_reader(uuid):
        asked.append(uuid)
        return 'WDDM'
    chk = run_cells.device_resident_check({'path': str(bundle)}, cuda_slot, 'auto', REQUIRED_BF16, A, driver_model_reader=wddm_reader)
    expect(asked == [A] and chk['driver_model'] == 'WDDM' and chk['sized_dtype'] == 'bfloat16' and chk['itemsize'] == 2 and chk['required_bytes'] == REQUIRED_BF16 and chk['fits'] is True
           and chk['blocked'] is False and chk['layout'] == 'resident-music-bundle' and chk['header_problems'] == [] and chk['components']['dit']['numel'] == PINNED_NUMEL['dit'],
           f'device_resident_check reads the exact GPU and sizes auto as bfloat16 on a capability 12.0 device: {chk}')
    chk32 = run_cells.device_resident_check({'path': str(bundle)}, cuda_slot, 'float32', REQUIRED_BF16, A, driver_model_reader=wddm_reader)
    expect(chk32['sized_dtype'] == 'float32' and chk32['peak_set_bytes'] == 2 * PEAK_BF16 and chk32['blocked'] is True, f'float32 doubles the peak set and the same cap no longer fits: {chk32["peak_set_bytes"]} {chk32["blocked"]}')
    # a flat artifact (one component): peak set = total; an artifact without safetensors or with a malformed header fails closed
    flat = Path(tmp) / 'flat'
    synthetic_safetensors(flat / 'model.safetensors', [[1000, 1000]])
    f_est = run_cells.device_resident_check({'path': str(flat)}, cuda_slot, 'float16', 100 * GIB, A, driver_model_reader=wddm_reader)
    expect(f_est['layout'] == 'flat' and f_est['peak_set_bytes'] == f_est['total_resident_bytes'] == 2_000_000 and f_est['blocked'] is False, f'a flat layout is one component sized at the dtype: {f_est}')
    empty = Path(tmp) / 'empty'
    empty.mkdir()
    e_est = run_cells.device_resident_check({'path': str(empty)}, cuda_slot, 'auto', 100 * GIB, A, driver_model_reader=wddm_reader)
    expect(e_est['blocked'] is True and e_est['reason'].startswith('resource_reading_unavailable: device-resident estimate unavailable') and e_est['enforcement'] == 'headers_unreadable' and e_est['header_problems'],
           f'no safetensors headers: the check fails closed: {e_est}')
    broken = Path(tmp) / 'broken'
    broken.mkdir()
    (broken / 'model.safetensors').write_bytes(struct.pack('<Q', 5) + b'{"a":')
    b_est = run_cells.device_resident_check({'path': str(broken)}, cuda_slot, 'auto', 100 * GIB, A, driver_model_reader=lambda u: 'TCC')
    expect(b_est['blocked'] is True and 'resource_reading_unavailable' in b_est['reason'] and b_est['driver_model'] == 'TCC', f'a malformed header fails closed even on TCC: {b_est}')
    for bad_bytes in (b'\x01\x02', struct.pack('<Q', 0) + b'{}', struct.pack('<Q', 2) + b'{}', struct.pack('<Q', 30) + b'{"t": {"dtype": "F32", "shape": [-1]}}'):
        (broken / 'model.safetensors').write_bytes(bad_bytes)
        try:
            run_cells.safetensors_header_numel(broken / 'model.safetensors')
            expect(False, f'a malformed header must raise: {bad_bytes[:12]!r}')
        except ValueError:
            pass
# the nvidia-smi driver-model parse: WDDM/TCC for the exact GPU, None for anything else (never guessed), no call for a non-NVIDIA device string


class _Smi:
    def __init__(self, out):
        self.stdout, self.stderr, self.returncode = out, '', 0


calls = []


def smi(out):
    def runner(argv, **kw):
        calls.append(argv)
        if isinstance(out, Exception):
            raise out
        return _Smi(out)
    return runner


expect(run_cells.gpu_driver_model(A, runner=smi('WDDM\n')) == 'WDDM' and run_cells.gpu_driver_model(A, runner=smi('tcc\n')) == 'TCC'
       and run_cells.gpu_driver_model(A, runner=smi('N/A\n')) is None and run_cells.gpu_driver_model(A, runner=smi('')) is None
       and run_cells.gpu_driver_model(A, runner=smi(OSError('no nvidia-smi'))) is None, 'gpu_driver_model parses WDDM/TCC and returns None otherwise')
expect(calls and calls[0][:2] == ['nvidia-smi', f'--id={A}'] and '--query-gpu=driver_model.current' in calls[0], f'the driver model is asked for the exact GPU: {calls[0]}')
n_calls = len(calls)
expect(run_cells.gpu_driver_model('dml:1', runner=smi('WDDM\n')) is None and run_cells.gpu_driver_model(None, runner=smi('WDDM\n')) is None and len(calls) == n_calls, 'no nvidia-smi call for a non-NVIDIA device string')
# the pinned bundle's headers, when the operator names its folder in TITANOS_KIT_PINNED_BUNDLE and it is present on this host (headers
# only; tensor data is never read): the numbers the pre-launch check works with. No lab path is written here (tools/public_export.json).
PINNED_BUNDLE = Path(os.environ['TITANOS_KIT_PINNED_BUNDLE']) if os.environ.get('TITANOS_KIT_PINNED_BUNDLE') else None
pinned_observation = 'pinned bundle not configured (TITANOS_KIT_PINNED_BUNDLE unset)' if PINNED_BUNDLE is None else 'pinned bundle absent on this host'
if PINNED_BUNDLE is not None and all((PINNED_BUNDLE / rel).is_dir() for rel in run_cells.RESIDENT_COMPONENTS.values()):
    real = run_cells.bundle_component_bytes(PINNED_BUNDLE, 2)
    real_bytes = {k: v['bytes'] for k, v in real['components'].items()}
    expect(real['problems'] == [] and real_bytes == PINNED_BF16 and real['components']['planner']['files'] == 2 and real['components']['dit']['files'] == 4,
           f'the pinned bundle headers give the pinned bf16 bytes: {real}')
    real_est = run_cells.device_resident_estimate(real_bytes, 21664305152, 'WDDM')
    pinned_observation = {'bf16_bytes': real_bytes, 'peak_set_bytes': real_est['peak_set_bytes'], 'required_bytes': real_est['required_bytes'], 'r3_cap_fits': real_est['fits']}
assert not failures, '\n'.join(failures)
print(json.dumps({'passed': True, 'checks': 'slot mapping (incl. Windows/Linux XPU), allowed-set filtering, homogeneous groups, catalogue coverage, refusal classification, construction na classification, missing-runtime untried, multi-GPU admission cross-checked against Titanos config validation, full COMPAT/spec runtime mapping, dtype policy, WDDM-aware device-resident pre-launch check (synthetic headers)', 'tasks_covered': len(tasks),
                  'pinned_bundle_headers': pinned_observation}))

# --- page alignment: the kernel keeps JobMemoryLimit in 4096-byte pages (an unaligned request read back lower and failed the strict check)
from containment import page_aligned, derive_job_limit as _djl, PAGE_BYTES as _PB
assert page_aligned(4509715660) == 4509712384 and page_aligned(4509712384) == 4509712384 and page_aligned(4095) == 0
for args in ((28106756096, 6442450944, None, 2 << 30), (1000, 1, 3000000001, 4097), (123456789, 4096, None, 4096)):
    cap, source = _djl(*args)
    assert cap % _PB == 0, (args, cap)
print('page alignment: derive_job_limit caps are 4096-byte multiples; page_aligned rounds down')
