"""Map the host this kit runs on to execution-matrix platform slots and concrete Titanos device strings.

A slot is (OS, architecture, accelerator) as defined in docs/execution_matrix_spec.json. One host usually covers several slots
(a Windows x64 box with an NVIDIA card covers win-x64-cpu, win-x64-nvidia-cuda and win-x64-nvidia-directml). Detection is
read-only and never claims a slot whose runtime probe did not succeed; what could not be probed is reported under
`unprobed` so an untried column stays untried for the right reason.
"""
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / 'docs' / 'execution_matrix_spec.json').read_text(encoding='utf-8'))
SLOT_IDS = [s['id'] for s in SPEC['platform_slots']]

AMD_IGPU = re.compile(r'(radeon(\(tm\))?\s*(\d{3,4}[ms]\b|graphics)|raphael|phoenix|strix|rembrandt|hawk point|granite ridge|vega \d\b)', re.I)
INTEL_DGPU = re.compile(r'\barc\b|battlemage|alchemist|b\d{3}\b|a\d{3}\b', re.I)
# integrated Intel graphics by name: Iris, UHD, Meteor Lake 'Arc(TM) Graphics', Lunar Lake 'Arc 130V/140V'
INTEL_IGPU = re.compile(r'iris|uhd|integrated|\(igpu\)|\b1[0-9]0v\b|arc(?:\(tm\)|\(r\))?\s+graphics\b', re.I)


def _gpu_key(name):
    return re.sub(r'[^a-z0-9]+', ' ', re.sub(r'\((?:d|i)gpu\)', '', str(name).lower())).strip()


def intel_gpu_discrete(name, ov_devices=()):
    """Whether an Intel GPU named by torch.xpu is discrete. The OpenVINO probe's device type decides when it saw a GPU of the same
    name (the rule the OpenVINO mapping in detect() uses); otherwise the name decides (Iris/UHD/Lunar Lake Arc 1x0V integrated,
    Arc A- and B-series discrete)."""
    key = _gpu_key(name)
    for d in ov_devices or []:
        if str(d.get('address', '')).startswith('GPU') and 'intel' in str(d.get('name', '')).lower():
            other, kind = _gpu_key(d.get('name', '')), str(d.get('type', ''))
            if kind and key and (key == other or key in other or other in key):
                return 'discrete' in kind
    if INTEL_IGPU.search(str(name)):
        return False
    return bool(INTEL_DGPU.search(str(name)))


def host_platform():
    system = platform.system().lower()
    machine = platform.machine().lower()
    os_name = {'windows': 'windows', 'linux': 'linux', 'darwin': 'macos'}.get(system, system)
    arch = 'arm64' if machine in {'arm64', 'aarch64'} else 'x64' if machine in {'amd64', 'x86_64', 'x64'} else machine
    return os_name, arch


def probe_torch():
    out = {'available': False}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        out['error'] = repr(exc)
        return out
    out.update(available=True, version=torch.__version__, cuda=getattr(torch.version, 'cuda', None), hip=getattr(torch.version, 'hip', None))
    gpus = []
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                uuid = str(getattr(p, 'uuid', '') or '')
                gpus.append({'index': i, 'name': p.name, 'capability': f'{p.major}.{p.minor}', 'total_memory': p.total_memory,
                             'uuid': (uuid if uuid.startswith(('GPU-', 'HIP-')) else ('HIP-' if torch.version.hip else 'GPU-') + uuid) if uuid else None,
                             'family': 'rocm' if torch.version.hip else 'cuda'})
    except Exception as exc:  # noqa: BLE001
        out['cuda_error'] = repr(exc)
    out['gpus'] = gpus
    try:
        out['mps'] = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        out['mps'] = False
    try:
        xpu = getattr(torch, 'xpu', None)
        out['xpu'] = [{'index': i, 'name': xpu.get_device_properties(i).name} for i in range(xpu.device_count())] if xpu and xpu.is_available() else []
    except Exception as exc:  # noqa: BLE001
        out['xpu'] = []
        out['xpu_error'] = repr(exc)
    return out


def probe_onnx():
    out = {'available': False}
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        out['error'] = repr(exc)
        return out
    out.update(available=True, version=ort.__version__, providers=list(ort.get_available_providers()), dml_adapters=[])
    if 'QNNExecutionProvider' not in out['providers']:
        # onnxruntime-qnn 2.x is a plugin EP for the core runtime: it is absent from get_available_providers() until registered
        # (Titanos registers it itself, src/titanos/compiled.py qnn_devices); probe the same way so the QNN slot is claimed only
        # when the plugin actually enumerates an NPU device.
        try:
            import onnxruntime_qnn as qnn
            ort.register_execution_provider_library('QNNExecutionProvider', qnn.get_library_path())
            devs = [str(getattr(getattr(d, 'device', None), 'type', '')) for d in ort.get_ep_devices() if d.ep_name == 'QNNExecutionProvider']
            if devs:
                out['providers'].append('QNNExecutionProvider')
                out['qnn_plugin'] = {'version': getattr(qnn, '__version__', None), 'library': qnn.get_library_path(), 'devices': devs}
        except Exception as exc:  # noqa: BLE001
            out['qnn_plugin_error'] = repr(exc)
    if 'DmlExecutionProvider' in out['providers']:
        try:
            from titanos.dxgi import adapters
            out['dml_adapters'] = [dict(a) for a in adapters()]
        except Exception as exc:  # noqa: BLE001
            out['dml_error'] = repr(exc)
    return out


def probe_openvino():
    out = {'available': False}
    try:
        import openvino as ov
        core = ov.Core()
    except Exception as exc:  # noqa: BLE001
        out['error'] = repr(exc)
        return out
    devices = []
    for address in core.available_devices:
        row = {'address': address}
        try:
            row['name'] = str(core.get_property(address, 'FULL_DEVICE_NAME'))
            props = {str(p) for p in core.get_property(address, 'SUPPORTED_PROPERTIES')}
            if 'DEVICE_TYPE' in props:
                row['type'] = str(core.get_property(address, 'DEVICE_TYPE')).lower()
        except Exception as exc:  # noqa: BLE001
            row['error'] = repr(exc)
        devices.append(row)
    out.update(available=True, version=str(ov.__version__), devices=devices)
    return out


SYCL_DEVICE = re.compile(r'^\s*(SYCL\d+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)\s*$')


def parse_native_devices(text):
    """Devices a llama.cpp `--list-devices` printout names: [{address, name, total_mib, free_mib}] for SYCL entries."""
    out = []
    for line in str(text or '').splitlines():
        m = SYCL_DEVICE.match(line)
        if m:
            out.append({'address': m.group(1), 'name': m.group(2), 'total_mib': int(m.group(3)), 'free_mib': int(m.group(4))})
    return out


def probe_native_sycl(executable, runner=None):
    """Run a SYCL llama.cpp build's `--list-devices` (src/titanos/native.py native_devices reads the same inventory) and return
    {'available', 'executable', 'devices'} or {'available': False, 'error'}. The executable is the catalogued win-sycl-x64 /
    ubuntu-sycl-*-x64 release build (src/titanos/data/native-profiles.json), supplied by the operator with --native-sycl."""
    if not executable:
        return {'available': False, 'error': 'no SYCL llama-server given (--native-sycl)'}
    exe = os.path.abspath(executable)
    if not os.path.isfile(exe):
        return {'available': False, 'error': f'SYCL llama-server not found: {exe}'}
    try:
        proc = (runner or subprocess.run)([exe, '--list-devices'], capture_output=True, text=True, timeout=120, cwd=os.path.dirname(exe))
    except Exception as exc:  # noqa: BLE001
        return {'available': False, 'error': f'{type(exc).__name__}: {exc}', 'executable': exe}
    devices = parse_native_devices(proc.stdout + '\n' + proc.stderr)
    if not devices:
        return {'available': False, 'error': f'--list-devices named no SYCL device (rc {proc.returncode}): {(proc.stdout + proc.stderr)[-300:]}', 'executable': exe}
    return {'available': True, 'executable': exe, 'devices': devices}


def nvidia_smi():
    import subprocess
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,compute_cap,memory.total', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return {'error': repr(exc)}
    rows = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 6:
            rows.append({'index': int(parts[0]), 'name': parts[1], 'uuid': parts[2], 'driver': parts[3], 'capability': parts[4], 'memory': parts[5]})
    return {'returncode': r.returncode, 'gpus': rows}


def detect(allowed_gpus=None, probes=None, native_sycl=None):
    """Return {'host': {...}, 'local_hostname': str, 'slots': [{slot, device, device_named, group, runtime}], 'unprobed': [...], 'gpu_groups': [...]}.

    host carries no machine name: `label` is the role label run_cells fills from --host-label (tools/lab_paths.HOST_ROLE_LABELS) and is
    what every receipt and RUN_RESULT carry; what the operating system calls the machine is returned beside it as `local_hostname`,
    written to HOST_DETECTION.json only and never copied into a receipt (tools/public_export.json machine_name_policy).

    allowed_gpus: iterable of GPU UUIDs this run may use (None = all). GPUs outside the list are reported but never assigned.
    probes: optional dict of pre-computed probe results (for tests).
    native_sycl: path of a SYCL llama.cpp llama-server; its `--list-devices` Intel GPUs become native-sycl entries of the Intel GPU
    slots (device native:SYCLn, the address Titanos's native runner takes: src/titanos/native.py:96-111).
    """
    os_name, arch = host_platform()
    probes = probes or {'torch': probe_torch(), 'onnx': probe_onnx(), 'openvino': probe_openvino(), 'nvidia_smi': nvidia_smi() if os_name != 'macos' else {}}
    if native_sycl and 'native_sycl' not in probes:
        probes['native_sycl'] = probe_native_sycl(native_sycl)
    host = {'label': None, 'os': os_name, 'arch': arch, 'platform': platform.platform(), 'python': sys.version.split()[0], 'probes': probes}
    slots, unprobed = [], []
    allowed = set(allowed_gpus) if allowed_gpus is not None else None

    def add(slot, device, named, runtime, **extra):
        if slot in SLOT_IDS:
            slots.append({'slot': slot, 'device': device, 'device_named': named, 'runtime': runtime, **extra})

    cpu_slot = {('windows', 'x64'): 'win-x64-cpu', ('windows', 'arm64'): 'win-arm64-cpu', ('linux', 'x64'): 'linux-x64-cpu',
                ('linux', 'arm64'): 'linux-arm64-cpu', ('macos', 'arm64'): 'macos-arm64-cpu', ('macos', 'x64'): 'macos-x64'}.get((os_name, arch))
    if cpu_slot:
        add(cpu_slot, 'cpu', platform.processor() or platform.machine() or 'host CPU', 'cpu')
    torch_p = probes.get('torch') or {}
    gpus = [g for g in torch_p.get('gpus', []) if g.get('uuid')]
    usable = [g for g in gpus if allowed is None or g['uuid'] in allowed]
    excluded = [g['uuid'] for g in gpus if allowed is not None and g['uuid'] not in allowed]
    if gpus and torch_p.get('hip'):
        if (os_name, arch) == ('linux', 'x64'):
            for g in usable:
                add('linux-x64-amd-rocm', g['uuid'], f"{g['name']} (ROCm, {g['capability']})", 'torch-rocm', gpu=g)
    elif gpus:
        slot = {('windows', 'x64'): 'win-x64-nvidia-cuda', ('linux', 'x64'): 'linux-x64-nvidia-cuda', ('linux', 'arm64'): 'linux-arm64-nvidia'}.get((os_name, arch))
        if slot:
            for g in usable:
                add(slot, g['uuid'], f"{g['name']} (CUDA, compute {g['capability']})", 'torch-cuda', gpu=g)
    elif torch_p.get('available') is False:
        unprobed.append({'runtime': 'torch', 'reason': torch_p.get('error', 'not installed')})
    if torch_p.get('mps') and (os_name, arch) == ('macos', 'arm64'):
        add('macos-arm64-mps', 'mps', 'Apple GPU (Metal)', 'torch-mps')
    if torch_p.get('xpu'):
        # PyTorch XPU is a Titanos-declared path on Linux and Windows x86_64 (tools/profiles.json xpu; docs/HARDWARE.md:46). Linux has
        # one Intel GPU slot; Windows splits discrete Arc from integrated graphics, decided like the OpenVINO devices below.
        ov_devices = (probes.get('openvino') or {}).get('devices', [])
        for x in torch_p['xpu']:
            device, named = f"xpu:{x.get('index', 0)}", str(x.get('name') or 'Intel GPU') + ' (PyTorch XPU)'
            if (os_name, arch) == ('linux', 'x64'):
                add('linux-x64-intel-gpu', device, named, 'torch-xpu')
            elif (os_name, arch) == ('windows', 'x64'):
                add('win-x64-intel-dgpu-openvino' if intel_gpu_discrete(x.get('name', ''), ov_devices) else 'win-x64-intel-igpu', device, named, 'torch-xpu')
            else:
                unprobed.append({'runtime': 'torch-xpu', 'reason': f"XPU device {device} ({x.get('name')}) has no matrix slot on {os_name}-{arch}; not assigned"})
    onnx_p = probes.get('onnx') or {}
    providers = set(onnx_p.get('providers', []))
    if onnx_p.get('available') is False:
        unprobed.append({'runtime': 'onnxruntime', 'reason': onnx_p.get('error', 'not installed')})
    if onnx_p.get('available') and os_name == 'windows' and 'DmlExecutionProvider' not in providers:
        unprobed.append({'runtime': 'onnxruntime-directml', 'reason': 'onnxruntime present without the DirectML provider; DirectML slots stay unclaimed'})
    if 'DmlExecutionProvider' in providers and os_name == 'windows':
        pci_uuid = None
        for a in onnx_p.get('dml_adapters', []):
            name = str(a.get('name', ''))
            device = 'dml:' + str(a.get('device_id', 0))
            low = name.lower()
            if a.get('software') in (True, 'True'):
                continue
            if 'nvidia' in low and arch == 'x64':
                # Tie the adapter to the physical GPU nvidia-smi names (LUID -> PCI -> UUID) so the allowed set applies to DirectML too.
                gpu_uuid = a.get('gpu_uuid')
                if gpu_uuid is None:
                    try:
                        import dxgi_pci
                        pci_uuid = dxgi_pci.nvidia_pci_to_uuid() if pci_uuid is None else pci_uuid
                        gpu_uuid = pci_uuid.get(dxgi_pci.pci_address_for_luid(str(a.get('luid', ''))))
                    except Exception:  # noqa: BLE001
                        gpu_uuid = None
                if allowed is not None and gpu_uuid not in allowed:
                    unprobed.append({'runtime': 'onnx-dml', 'reason': f'adapter {device} ({name}) maps to GPU {gpu_uuid or "unknown"} outside the allowed set; not assigned'})
                    continue
                add('win-x64-nvidia-directml', device, name + f' (DirectML{", " + gpu_uuid if gpu_uuid else ""})', 'onnx-dml', adapter=a, gpu={'uuid': gpu_uuid, 'name': name} if gpu_uuid else None)
            elif ('amd' in low or 'radeon' in low) and arch == 'x64':
                add('win-x64-amd-igpu-directml' if AMD_IGPU.search(name) else 'win-x64-amd-dgpu-directml', device, name + ' (DirectML)', 'onnx-dml', adapter=a)
            elif 'intel' in low and arch == 'x64':
                if INTEL_DGPU.search(name):
                    # an Arc A- or B-series discrete adapter under DirectML has no matrix slot (the dGPU slot is OpenVINO only)
                    unprobed.append({'runtime': 'onnx-dml', 'reason': f'adapter {device} ({name}) is an Intel discrete GPU under DirectML; the matrix has no such slot, so it is not assigned'})
                else:
                    add('win-x64-intel-igpu', device, name + ' (DirectML)', 'onnx-dml', adapter=a)
            elif ('adreno' in low or 'qualcomm' in low) and arch == 'arm64':
                add('win-arm64-adreno-directml', device, name + ' (DirectML)', 'onnx-dml', adapter=a)
            else:
                unprobed.append({'runtime': 'onnx-dml', 'reason': f'adapter {device} ({name}) has no matrix slot on {os_name}-{arch}; not assigned'})
    if 'QNNExecutionProvider' in providers and (os_name, arch) == ('windows', 'arm64'):
        add('win-arm64-qnn-npu', 'qnn', 'Qualcomm Hexagon NPU (QNN HTP)', 'onnx-qnn')
    if 'CoreMLExecutionProvider' in providers and (os_name, arch) == ('macos', 'arm64'):
        add('macos-arm64-ane', 'coreml', 'Apple Neural Engine (Core ML EP)', 'onnx-coreml')
    if 'VitisAIExecutionProvider' in providers and (os_name, arch) == ('windows', 'x64'):
        add('win-x64-amd-ryzen-ai-npu', 'vitisai', 'AMD Ryzen AI NPU (VitisAI EP)', 'onnx-vitisai')
    ov_p = probes.get('openvino') or {}
    if ov_p.get('available') is False:
        unprobed.append({'runtime': 'openvino', 'reason': ov_p.get('error', 'not installed')})
    for d in ov_p.get('devices', []):
        addr, name, kind = d.get('address', ''), str(d.get('name', '')), str(d.get('type', ''))
        if addr.startswith('GPU') and 'intel' in name.lower():
            # the OpenVINO probe's device type decides: a Lunar Lake 'Arc 130V (iGPU)' is type.integrated and must not become a dGPU cell
            discrete = 'discrete' in kind or ('integrated' not in kind and bool(INTEL_DGPU.search(name)))
            if os_name == 'windows' and arch == 'x64':
                add('win-x64-intel-dgpu-openvino' if discrete else 'win-x64-intel-igpu', addr, name + ' (OpenVINO GPU)', 'openvino-gpu')
            elif os_name == 'linux' and arch == 'x64':
                add('linux-x64-intel-gpu', addr, name + ' (OpenVINO GPU)', 'openvino-gpu')
        elif addr.startswith('NPU') and (os_name, arch) == ('windows', 'x64'):
            add('win-x64-intel-npu-openvino', addr, name + ' (OpenVINO NPU)', 'openvino-npu')
    sycl_p = probes.get('native_sycl') or {}
    if sycl_p.get('available'):
        ov_devices = (probes.get('openvino') or {}).get('devices') or []
        for d in sycl_p['devices']:
            name = d['name']
            if 'intel' not in name.lower():
                unprobed.append({'runtime': 'native-sycl', 'reason': f"{d['address']} ({name}) is not an Intel GPU; the SYCL path is mapped on the Intel GPU slots only"})
                continue
            if (os_name, arch) == ('windows', 'x64'):
                slot = 'win-x64-intel-dgpu-openvino' if intel_gpu_discrete(name, ov_devices) else 'win-x64-intel-igpu'
            elif (os_name, arch) == ('linux', 'x64'):
                slot = 'linux-x64-intel-gpu'
            else:
                unprobed.append({'runtime': 'native-sycl', 'reason': f"{d['address']} ({name}) has no Intel GPU slot on {os_name}-{arch}; not assigned"})
                continue
            add(slot, f"native:{d['address']}", f"{name} (llama.cpp SYCL)", 'native-sycl', executable=sycl_p['executable'], native_device=d)
    elif sycl_p:
        unprobed.append({'runtime': 'native-sycl', 'reason': sycl_p.get('error', 'no SYCL device')})
    groups = []
    by_sig = {}
    for g in usable:
        by_sig.setdefault((g['name'], g['capability'], g['family']), []).append(g['uuid'])
    for (name, cap, fam), uuids in by_sig.items():
        if len(uuids) >= 2:
            groups.append({'name': name, 'capability': cap, 'family': fam, 'gpus': uuids, 'slot': next((s['slot'] for s in slots if s.get('gpu', {}).get('uuid') == uuids[0]), None)})
    host['excluded_gpus'] = excluded
    return {'host': host, 'local_hostname': socket.gethostname(), 'slots': slots, 'unprobed': unprobed, 'gpu_groups': groups}


if __name__ == '__main__':
    allowed = os.environ.get('MATRIX_KIT_ALLOWED_GPUS')
    print(json.dumps(detect(allowed.split(',') if allowed else None), indent=2, default=str))
