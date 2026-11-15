"""Installed hardware, runtime visibility and stable assignment admission."""
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

VISIBILITY_KEYS = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
                   'GPU_DEVICE_ORDINAL', 'NVIDIA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
                   'ZE_AFFINITY_MASK', 'ONEAPI_DEVICE_SELECTOR', 'SYCL_DEVICE_FILTER')


def visibility_constraints(environ=None):
    env = os.environ if environ is None else environ
    return {key: env[key] for key in VISIBILITY_KEYS if key in env}


def _probe(arguments, environ=None):
    """Enumerate in a fresh owned child; never remove inherited visibility."""
    from .owned_command import run_owned
    from .process_command import module_command
    from .settings import current_settings
    settings = current_settings()
    from .native_process import child_environment
    env = child_environment(dict(os.environ if environ is None else environ))
    # Prevent inherited interpreter injection without changing driver/visibility.
    for key in tuple(env):
        if (key in {'PYTHONSTARTUP', 'PYTHONHOME', 'PYTHONINSPECT', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'}
                or key.startswith('_PYI') or key.endswith(('_API_KEY', '_TOKEN', '_PASSWORD', '_SECRET'))):
            env.pop(key, None)
    if getattr(sys, 'frozen', False):
        env.pop('PYTHONPATH', None)
        if 'LD_LIBRARY_PATH_ORIG' in env:env['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH_ORIG']
        else:env.pop('LD_LIBRARY_PATH', None)
    deadline = time.monotonic() + settings.device_probe_timeout
    def check():
        if time.monotonic() > deadline:
            raise ValueError('Device discovery timed out; runtime availability remains unverified')
    with tempfile.TemporaryDirectory(prefix='titanos-device-probe-') as folder:
        directory = Path(folder)
        try:
            run_owned(module_command('titanos.device_probe', *arguments), directory, env, check,
                      2 * 1024 * 1024, label='Device discovery')
        except Exception as exc:
            from .engine import EngineError
            if not isinstance(exc,EngineError):raise
            detail=(directory/'stderr.log').read_text(encoding='utf-8',errors='replace').strip()
            # Retain the actual exception, not only PyInstaller's final wrapper
            # line. The owned command already bounds total log bytes.
            raise ValueError('Device discovery failed: '+(detail[-8000:] or str(exc))) from exc
        value = json.loads((directory / 'stdout.log').read_text(encoding='utf-8'))
    if not isinstance(value, list) or len(value) > 256 or not all(isinstance(row, dict) for row in value):
        raise ValueError('Device discovery returned an invalid bounded inventory')
    return value


def probe_visible_cuda(library_dir=None, environ=None):
    env = dict(os.environ if environ is None else environ)
    if library_dir is not None:
        env['TITANOS_CUDA_LIBRARY_DIR'] = str(library_dir)
    return _probe(['--cuda-devices'], env)


def _uuid_subset(value, physical):
    """Resolve only UUID masks here. Numeric masks need runtime observation."""
    tokens = value.split(',')
    result = []
    for token in tokens:
        token = token.strip()
        if not token or not token.startswith('GPU-') or not normalized_uuid(token):
            raise ValueError('Visibility requires complete UUIDs or current runtime UUID observation')
        matches = [row for row in physical if str(row.get('uuid', '')).lower().startswith(token.lower())]
        if len(matches) != 1:
            raise ValueError('CUDA UUID visibility is absent or ambiguous; it was not expanded')
        if matches[0]['uuid'] in {row['uuid'] for row in result}:
            raise ValueError('CUDA visibility repeats a physical GPU')
        result.append(matches[0])
    return result


def visible_nvidia_records(physical=None, *, environ=None, visible=None, library_dir=None):
    """Return allowed physical rows; nvidia-smi ordinals are never CUDA ordinals.

    A numeric/compound mask is resolved from UUIDs observed under the unchanged
    mask, using a bounded CUDA-runtime child if the caller has not supplied them.
    """
    physical = scan() if physical is None else list(physical)
    env = os.environ if environ is None else environ
    by_uuid = {}
    for row in physical:
        value = row.get('uuid')
        if not isinstance(value, str) or not value.startswith('GPU-') or not normalized_uuid(value):
            raise ValueError('Physical NVIDIA discovery requires complete nonzero GPU UUIDs')
        key = value.lower()
        if key in by_uuid:
            raise ValueError('Duplicate physical NVIDIA UUID')
        by_uuid[key] = row
    # A container visibility restriction is an outer scope, not a CUDA ordinal map.
    outer = env.get('NVIDIA_VISIBLE_DEVICES')
    if outer is not None and outer.strip().lower() in {'', 'none', 'void', '-1'}:
        return []
    mask = env.get('CUDA_VISIBLE_DEVICES')
    if mask is not None and mask.strip() in {'', '-1'}:
        return []
    allowed = physical
    if mask is not None:
        tokens = [token.strip() for token in mask.split(',')]
        if all(token.startswith('GPU-') and normalized_uuid(token) for token in tokens):
            allowed = _uuid_subset(mask, physical)
        else:
            if not tokens or any(not re.fullmatch(r'[0-9]+|GPU-[0-9a-fA-F-]+',token) for token in tokens) or len(set(tokens)) != len(tokens):
                raise ValueError('CUDA visibility is malformed or ambiguous; no ordinal fallback was used')
            observed = probe_visible_cuda(library_dir, env) if visible is None else visible
            identities = [str(row.get('uuid', '')).lower() for row in observed]
            if len(identities) != len(tokens) or len(set(identities)) != len(identities) or any(identity not in by_uuid for identity in identities):
                raise ValueError('Current-mask CUDA UUID observation is incomplete or outside installed physical identities')
            if any(token.startswith('GPU-') and not identity.startswith(token.lower()) for token,identity in zip(tokens,identities)):
                raise ValueError('Observed CUDA UUID order differs from inherited UUID visibility')
            allowed = [by_uuid[identity] for identity in identities]
    elif visible is not None:
        identities = [str(row.get('uuid', '')).lower() for row in visible]
        if len(set(identities)) != len(identities) or any(identity not in by_uuid for identity in identities):
            raise ValueError('Current runtime GPU identities are missing or ambiguous')
        allowed = [by_uuid[identity] for identity in identities]
    if outer is not None and outer.strip().lower() != 'all':
        if all(token.strip().startswith('GPU-') for token in outer.split(',')):
            outer_ids = {row['uuid'].lower() for row in _uuid_subset(outer, physical)}
            allowed = [row for row in allowed if row['uuid'].lower() in outer_ids]
        else:
            # NVIDIA container indices are driver/NVML indices, unlike CUDA
            # ordinals. If that outer namespace is no longer observable (for
            # example after container remapping), require an explicit UUID mask.
            tokens = [token.strip() for token in outer.split(',')]
            if any(not token.isdecimal() for token in tokens) or len(set(tokens)) != len(tokens):
                raise ValueError('NVIDIA container visibility requires unambiguous UUIDs or driver indices')
            selected = [row for token in tokens for row in physical if str(row.get('index')) == token]
            if len(selected) != len(tokens):
                raise ValueError('Container driver indices cannot be resolved; use complete NVIDIA UUIDs')
            outer_ids = {row['uuid'].lower() for row in selected}
            allowed = [row for row in allowed if row['uuid'].lower() in outer_ids]
    return allowed


def admit_gpu_uuids(selectors, *, family='cuda', physical=None, visible=None, environ=None):
    """Check explicit physical selection before a child narrows its visibility."""
    env = os.environ if environ is None else environ
    if family == 'cuda':
        installed = scan() if physical is None else physical
        allowed = visible_nvidia_records(installed, environ=env, visible=visible)
        prefix = 'GPU-'
    elif family == 'hip':
        prefix = 'HIP-'
        keys = ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
        if any(key in env and env[key].strip() in {'', '-1'} for key in keys):
            allowed = []
        else:
            # HIP/ROCR masks compose in different namespaces. Observe the final
            # inherited runtime view instead of assigning meaning to ordinals.
            allowed = hip_records(environ=env) if visible is None else visible
    else:
        raise ValueError('GPU assignment family must be cuda or hip')
    available = {str(row.get('uuid', '')).lower(): row.get('uuid') for row in allowed}
    result = []
    for selector in selectors:
        raw = selector.removeprefix(prefix) if isinstance(selector, str) and selector.startswith(prefix) else None
        identity = normalized_uuid(raw)
        key = (prefix + identity).lower() if identity else None
        if key not in available:
            raise ValueError('Selected physical GPU is missing, masked or unverified; inherited visibility was not expanded')
        result.append(available[key])
    if len(set(result)) != len(result):
        raise ValueError('Duplicate physical GPU assignment')
    return result


def device_signature(record):
    identity = str(record.get('uuid') or record.get('id') or '')
    vendor = next((family for prefix, family in
        (('GPU-', 'cuda'), ('HIP-', 'rocm'), ('XPU-', 'xpu'), ('OV-', 'openvino')) if identity.startswith(prefix)), None) or record.get('runtime_vendor')
    architecture = ((record.get('compute_capability') or record.get('capability') or record.get('architecture'))
                    if vendor == 'cuda' else record.get('architecture'))
    return {'runtime_vendor': vendor, 'architecture': str(architecture) if architecture not in (None, '') else None}


def require_homogeneous(records):
    signatures = [device_signature(record) for record in records]
    if (not signatures or any(not row['runtime_vendor'] or not row['architecture'] for row in signatures)
            or any(row != signatures[0] for row in signatures[1:])):
        raise ValueError('A multi-device group requires a known identical runtime vendor and architecture; unknown devices are not assumed homogeneous')
    return signatures[0]


def preferred_runtime_device(records, *, backend, task=None, installed=None, options=None):
    """Pure common auto policy; returned enumeration is not execution proof.

    Explicit matching installation pins win. Otherwise prefer an observed GPU,
    then a supported NPU/provider, then CPU. No fallback follows a missing pin.
    """
    family='openvino' if backend.startswith('openvino') else backend
    choices=[]
    providers={'QNNExecutionProvider':'qnn','CoreMLExecutionProvider':'coreml','VitisAIExecutionProvider':'vitisai','CPUExecutionProvider':'cpu'}
    for source in records:
        row=dict(source)
        if row.get('available',True) is False or row.get('present',True) is False or row.get('software'):
            continue
        address=str(row.get('runtime_address') or '')
        if family=='pytorch':
            target=address
            priority=0 if address.startswith('cuda') else 10 if address.startswith('xpu') else 20 if address=='mps' else 100 if address=='cpu' else None
        elif family=='openvino':
            target=address
            priority=0 if address.startswith('GPU') else 20 if address.startswith('NPU') and not (backend=='openvino-genai' and task=='image') else 100 if address=='CPU' else None
        elif family=='onnx':
            provider=row.get('provider') or address
            target='dml' if address.startswith('dml:') else providers.get(provider)
            priority=0 if target=='dml' else 20 if target in {'qnn','coreml','vitisai'} else 100 if target=='cpu' else None
        else:
            raise ValueError('Unknown automatic runtime family')
        if priority is None:continue
        match=re.search(r'(?:[:.])(\d+)$',address)
        row.update(device=target,auto_rank=(priority,int(match[1]) if match else 0))
        choices.append(row)
    def dml_choice(candidates):
        requested=(options or {}).get('device_id')
        if requested is None:requested=((installed or {}).get('provider_options') or {}).get('device_id')
        if requested is not None:
            if isinstance(requested,bool) or not str(requested).isdecimal():raise ValueError('DirectML device_id must be a nonnegative integer')
            matches=[row for row in candidates if row['device']=='dml' and str((row.get('provider_options') or {}).get('device_id',row['runtime_address'].split(':')[-1])).isdecimal()
                     and int((row.get('provider_options') or {}).get('device_id',row['runtime_address'].split(':')[-1]))==int(requested)]
            if len(matches)!=1:raise ValueError('Requested DirectML adapter is unavailable or ambiguous; no adapter-zero substitution was used')
            return matches[0]
        return min(candidates,key=lambda row:row['auto_rank']) if candidates else None
    if installed and (installed.get('backend','pytorch')==backend or str(installed.get('backend','')).startswith('openvino') and backend.startswith('openvino')):
        pin=installed.get('gpu_uuid') or installed.get('device')
        if pin not in (None,'auto'):
            matches=[row for row in choices if pin in {row.get('uuid'),row.get('id'),row.get('runtime_address'),row.get('device')}
                     or str(pin).lower()=='cpu' and str(row.get('device')).lower()=='cpu'
                     or row.get('uuid') and str(pin).lower()==row['uuid'].lower()
                     or family=='openvino' and {str(pin).upper(),str(row.get('runtime_address')).upper()} in ({'GPU','GPU.0'},{'NPU','NPU.0'})]
            if family=='onnx' and str(pin)=='dml':
                chosen=dml_choice(matches)
                if chosen is None:raise ValueError('Installed DirectML provider pin has no observed hardware adapter')
                return {**chosen,'selection_reason':'installed_runtime_provider_pin'}
            if len(matches)!=1:
                raise ValueError('Installed runtime device pin is unavailable or unsupported for this task; no automatic substitution was used')
            return {**matches[0],'selection_reason':'installed_runtime_pin'}
    if not choices:return None
    selected=dml_choice(choices) if family=='onnx' else min(choices,key=lambda row:row['auto_rank'])
    reason=('provider_requires_device_validation' if selected.get('device_type')=='provider' and selected['device']!='cpu' else
            'compatible_visible_accelerator' if selected['auto_rank'][0]<100 else 'cpu_no_compatible_visible_accelerator')
    return {**selected,'selection_reason':reason}


def installed_runtime_preference():
    path=Path(sys.prefix)/'titanos-runtime.json'
    if not path.is_file():return None
    if path.stat().st_size>64*1024:raise ValueError('Installed runtime preference exceeds its metadata limit')
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise ValueError('Installed runtime preference must be a JSON object')
    return value


def scan():
    command = shutil.which('nvidia-smi')
    if not command:
        raise ValueError('NVIDIA identity discovery requires nvidia-smi; no index fallback is used')
    try:
        result = subprocess.run([command, '--query-gpu=index,uuid,name,pci.bus_id,memory.total,compute_cap,driver_version',
            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        detail=(getattr(exc,'stderr',None) or getattr(exc,'stdout',None) or str(exc)).strip()
        raise ValueError('NVIDIA identity discovery failed; check the driver and nvidia-smi: '+detail[-500:]) from exc
    devices = []
    for row in csv.reader(io.StringIO(result.stdout)):
        index, identity, name, pci, memory, capability, driver = [v.strip() for v in row]
        if not identity.startswith('GPU-'):
            raise ValueError('Expected a physical NVIDIA GPU UUID; MIG identities require a separate adapter')
        devices.append(dict(id=identity, uuid=identity, index=int(index), name=name, pci_bus_id=pci,
                            memory_mib=int(memory), compute_capability=capability, driver=driver,
                            identity_kind='vendor_uuid', identity_strength='physical_uuid', stable=True,
                            device_type='gpu', device_subtype=None, runtime_vendor='cuda'))
    if len({d['uuid'] for d in devices}) != len(devices):
        raise ValueError('Duplicate GPU identities returned by discovery')
    return devices


def pin(selectors, devices=None):
    devices = scan() if devices is None else devices
    result = []
    for selector in selectors:
        matches = [d for d in devices if selector == d['uuid'] or selector == str(d['index'])]
        if len(matches) != 1:
            raise ValueError(f'GPU {selector!r} is missing or ambiguous; assignment was not substituted')
        result.append(matches[0]['uuid'])
    if len(set(result)) != len(result):
        raise ValueError('Assignment selects the same physical GPU more than once')
    return result


def inventory(refresh=False, backend='nvidia'):
    if backend == 'all':
        return discovery(refresh)
    from .config import home
    root = home()
    root.mkdir(parents=True, exist_ok=True)
    # Separate immutable identity avoids concurrent refreshes minting new host IDs.
    identity = root / 'host-id'
    try:
        with identity.open('x', encoding='utf-8') as stream:
            stream.write(str(uuid.uuid4()))
    except FileExistsError:
        pass
    host = identity.read_text().strip()
    uuid.UUID(host)
    if backend not in {'nvidia','pytorch','openvino','onnx','system'}:
        raise ValueError('Unknown inventory backend')
    path = root / ('hardware.json' if backend=='nvidia' else 'hardware-'+backend+'.json')
    previous = json.loads(path.read_text()) if path.exists() else {'version': 1, 'host_id': host, 'devices': []}
    if previous.get('version') != 1 or previous.get('host_id') != host:
        raise ValueError('Hardware inventory identity/version mismatch; existing file was preserved')
    if not refresh:
        return previous
    # Failed scans never mark cards absent or destroy the last good snapshot.
    found = scan() if backend=='nvidia' else _probe(['--inventory', backend])
    now = time.time()
    rows = {d.get('id',d['uuid']): {**d, 'present': False} for d in previous['devices']}
    for device in found:
        key=device.get('id',device['uuid'])
        old = rows.get(key, {})
        rows[key] = {**device, 'first_seen': old.get('first_seen', now), 'last_seen': now, 'present': True}
    result = {'version': 1, 'host_id': host, 'scanned_at': now, 'devices': list(rows.values()),
              'backend':backend,'visibility':visibility_constraints(),
              'scope': 'Installed hardware or current runtime enumeration; neither is inference qualification'}
    fd, temporary = tempfile.mkstemp(dir=root, prefix='hardware-', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result


def cuda_uuid(torch, index):
    value = getattr(torch.cuda.get_device_properties(index), 'uuid', None)
    if value is None:
        raise ValueError('This PyTorch build cannot report GPU UUIDs; identity verification is unavailable')
    value = normalized_uuid(value)
    if value is None:raise ValueError('Runtime returned an invalid physical GPU UUID')
    return 'GPU-' + value


def verify_visible(expected, torch):
    if torch.cuda.device_count() != len(expected):
        raise ValueError('Visible GPU count does not match the pinned assignment')
    actual = [cuda_uuid(torch, index) for index in range(len(expected))]
    if [v.lower() for v in actual] != [v.lower() for v in expected]:
        raise ValueError(f'Runtime GPU identity/order mismatch: expected {expected}, observed {actual}')
    return [{'local_device': f'cuda:{i}', 'uuid': value} for i, value in enumerate(actual)]


def runtime_device(identity, torch):
    # Resolve against CUDA's own enumeration, never nvidia-smi's numeric index.
    for index in range(torch.cuda.device_count()):
        if cuda_uuid(torch, index).lower() == identity.lower():
            return f'cuda:{index}'
    raise ValueError(f'Pinned GPU {identity} is not visible to this runtime')


def normalized_uuid(value):
    if value is None:
        return None
    try:
        # OpenVINO may expose sixteen bytes; Torch uses a _CUuuid object whose
        # canonical string is its UUID and which has no len/bytes protocol.
        raw=None
        if not isinstance(value,str):
            try:
                if len(value)==16:raw=bytes(value)
            except (TypeError,ValueError):pass
        parsed=uuid.UUID(bytes=raw) if raw is not None else uuid.UUID(str(value).removeprefix('GPU-'))
        return str(parsed) if parsed.int else None
    except (ValueError,TypeError):
        return None


def runtime_records(backend, *, core=None):
    rows=[]
    def add(address,name,identity=None,kind='runtime_address',**extra):
        rows.append({'id':identity or backend+':'+address,'uuid':identity,'name':name,
                     'runtime_address':address,'identity_kind':kind if not identity else 'vendor_uuid',
                     'identity_strength':'vendor_runtime_uuid' if identity else 'runtime_address',
                     'backend':backend,'device_type':'gpu','available':True,**extra})
    if backend=='pytorch':
        import torch
        add('cpu','CPU',id='cpu',kind='host_cpu',identity_strength='host',device_type='cpu',stable=True)
        for family in ('cuda','xpu'):
            api=getattr(torch,family,None)
            if api and api.is_available():
                for i in range(api.device_count()):
                    props=api.get_device_properties(i)
                    value=normalized_uuid(getattr(props,'uuid',None))
                    prefix='HIP-' if family=='cuda' and torch.version.hip else 'GPU-' if family=='cuda' else 'XPU-'
                    architecture = (str(props.major)+'.'+str(props.minor) if hasattr(props,'major') and hasattr(props,'minor') else
                                    getattr(props,'gcnArchName',None) or getattr(props,'architecture',None))
                    add(f'{family}:{i}',props.name,prefix+value if value else None,memory_bytes=getattr(props,'total_memory',None),
                        architecture=architecture,runtime_vendor='rocm' if prefix=='HIP-' else family,
                        stable=bool(value),dtype_support={'float32':True,'float16':True,'bfloat16':None})
        if torch.backends.mps.is_available():
            add('mps','Apple Metal GPU',device_subtype='integrated',identity_strength='runtime_address')
    elif backend=='openvino':
        if core is None:
            import openvino as ov
            core=ov.Core()
        for address in core.available_devices:
            properties={str(p) for p in core.get_property(address,'SUPPORTED_PROPERTIES')}
            value=None
            if 'DEVICE_UUID' in properties:
                value=normalized_uuid(core.get_property(address,'DEVICE_UUID'))
            extra={}
            if 'DEVICE_ARCHITECTURE' in properties:extra['architecture']=str(core.get_property(address,'DEVICE_ARCHITECTURE'))
            if 'DEVICE_TYPE' in properties:
                kind=str(core.get_property(address,'DEVICE_TYPE')).lower()
                extra['device_subtype']='integrated' if 'integrated' in kind else 'discrete' if 'discrete' in kind else None
            if 'DEVICE_LUID' in properties:
                luid=core.get_property(address,'DEVICE_LUID')
                try:
                    raw=bytes(luid)
                    if len(raw)==8:extra['luid']=f'{int.from_bytes(raw[4:],"little"):08x}:{int.from_bytes(raw[:4],"little"):08x}'
                except (TypeError,ValueError):pass
            add(address,core.get_property(address,'FULL_DEVICE_NAME'),'OV-'+value if value else None,
                id='cpu' if address=='CPU' else 'OV-'+value if value else backend+':'+address,
                device_type='cpu' if address=='CPU' else 'npu' if address.startswith('NPU') else 'gpu',
                stable=bool(value) or address=='CPU',runtime_vendor='openvino',supported_properties=sorted(properties),**extra)
    elif backend=='onnx':
        import onnxruntime as ort
        for provider in ort.get_available_providers():
            add('cpu' if provider=='CPUExecutionProvider' else provider,provider,
                id='cpu' if provider=='CPUExecutionProvider' else backend+':'+provider,
                device_type='cpu' if provider=='CPUExecutionProvider' else 'provider',provider=provider)
            if provider == 'DmlExecutionProvider':
                from .dxgi import adapters
                for adapter in adapters():
                    add('dml:'+str(adapter['device_id']),adapter['name'],kind='dxgi_luid',
                        id='DML-'+adapter['luid'],provider=provider,device='dml',identity_strength='current_boot_luid',stable=False,
                        provider_options={'device_id':str(adapter['device_id'])},
                        **{k:v for k,v in adapter.items() if k!='name'})
    elif backend=='system':
        import platform
        add('cpu',platform.processor() or platform.machine(),id='cpu',kind='host_cpu',identity_strength='host',
            device_type='cpu',stable=True,logical_cores=os.cpu_count())
        if platform.system()=='Windows':
            from .device_probe import windows_records
            rows.extend(windows_records())
        for path in Path('/sys/bus/pci/devices').glob('*'):
            try:
                device_class=(path/'class').read_text().strip()
                if device_class.startswith(('0x03','0x12')):
                    add(path.name,(path/'vendor').read_text().strip()+':'+(path/'device').read_text().strip(),kind='pci_location',
                        identity_strength='host_pci_location',device_type='gpu' if device_class.startswith('0x03') else 'accelerator',
                        pci_bus_id=path.name,device_class=device_class,stable=True)
            except OSError:
                continue
    return rows


def _pci_location(row):
    value=row.get('pci_location') or row.get('pci_bus_id')
    if not isinstance(value,str):return None
    match=re.fullmatch(r'(?:[0-9a-fA-F]{4,8}:)?([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',value)
    return tuple(int(part,16) for part in match.groups()) if match else None


def _reported_uuid(row):
    value=row.get('uuid')
    if not isinstance(value,str):return None
    return normalized_uuid(value.split('-',1)[1]) if value.startswith(('GPU-','HIP-','XPU-','OV-')) else normalized_uuid(value)


def discovery(refresh=False):
    """One device view with explicit physical identity and runtime observations.

    Never join by a display name. Exact UUID, LUID or a unique current PCI
    location can link observations. Unmatched views stay explicitly separate.
    """
    from importlib.util import find_spec
    from .settings import current_settings
    settings=current_settings();snapshots={};errors=[];now=time.time()
    constraints=visibility_constraints()
    modules={'pytorch':'torch','openvino':'openvino','onnx':'onnxruntime'}
    for backend in ('system','nvidia','pytorch','openvino','onnx'):
        snapshot=inventory(False,backend)
        try:
            module=modules.get(backend)
            if module and find_spec(module) is None:
                raise ImportError('The '+module+' runtime is not included in this installation')
            if (refresh or not snapshot.get('scanned_at') or now-snapshot['scanned_at']>settings.device_inventory_ttl
                    or snapshot.get('visibility')!=constraints):
                snapshot=inventory(True,backend)
            snapshots[backend]={**snapshot,'scan_status':'current'}
        except Exception as exc:
            # Preserve the last successful hardware snapshot; failure is not
            # absence and its runtime observations cannot admit a new assignment.
            snapshots[backend]={**snapshot,'scan_status':'unverified'}
            errors.append({'backend':backend,'code':'runtime_missing' if isinstance(exc,ImportError) else 'probe_failed',
                           'message':str(exc)[:500]})
    rows={};providers=[]
    def physical(row,backend):
        identity=row.get('id') or row.get('uuid') or backend+':'+str(row.get('runtime_address'))
        value={**row,'id':identity,'label':row.get('name') or identity,'installed':True,
            'present':row.get('present',True),'discovery_status':snapshots[backend]['scan_status'],
            'identity_kind':row.get('identity_kind') or ('vendor_uuid' if row.get('uuid') else 'runtime_address'),
            'identity_strength':row.get('identity_strength') or ('physical_uuid' if row.get('uuid') else 'runtime_address'),
            'stable':bool(row.get('stable',row.get('uuid') is not None)),'runtimes':[],'aliases':[]}
        rows[identity]=value;return value
    for row in snapshots['system'].get('devices',[]):physical(row,'system')
    if 'cpu' not in rows:
        rows['cpu']={'id':'cpu','uuid':None,'label':'CPU','name':'CPU','device_type':'cpu','installed':True,
            'present':True,'identity_kind':'host_cpu','identity_strength':'host','stable':True,
            'runtimes':[],'aliases':[],'discovery_status':'current'}
    for row in snapshots['nvidia'].get('devices',[]):
        location=_pci_location(row)
        matches=[value for value in rows.values() if location is not None and _pci_location(value)==location and value.get('vendor_id') in (None,0x10de)]
        value=physical(row,'nvidia')
        if len(matches)==1:
            old=matches[0];rows.pop(old['id'],None)
            value.update({key:item for key,item in old.items() if key in {'pnp_instance','driver_status','problem_code','luid','device_subtype'}})
            value['aliases'].append(old['id']);value['identity_join']='unique_current_pci_location'
        value['device_type']='gpu';value['runtime_vendor']='cuda'
        value['architecture']=device_signature(value)['architecture']
    for backend in ('pytorch','openvino','onnx'):
        current=snapshots[backend]['scan_status']=='current'
        for row in snapshots[backend].get('devices',[]):
            if row.get('device_type')=='provider':
                providers.append({**row,'backend':backend,'available':current and row.get('present',True)});continue
            identity=row.get('id') or row.get('uuid')
            matches=[value for value in rows.values() if identity==value['id'] or identity in value['aliases']
                     or row.get('uuid') and row.get('uuid')==value.get('uuid')
                     or _reported_uuid(row) and _reported_uuid(row)==_reported_uuid(value)
                     or row.get('luid') and row.get('luid')==value.get('luid')]
            if len(matches)==1:
                value=matches[0]
            else:
                value=physical(row,backend)
                value['physical_link']='runtime_identity_only; no name-based physical merge'
            if identity!=value['id'] and identity not in value['aliases']:value['aliases'].append(identity)
            for field in ('memory_bytes','architecture','compute_capability','device_subtype','runtime_vendor','device_type'):
                if value.get(field) is None and row.get(field) is not None:value[field]=row[field]
            value['runtimes'].append({'backend':backend,'id':identity,'runtime_address':row.get('runtime_address'),
                'available':current and row.get('present',True),'identity_kind':row.get('identity_kind'),
                'identity_strength':row.get('identity_strength'),'uuid':row.get('uuid'),
                'provider':row.get('provider'),'provider_options':row.get('provider_options'),
                'architecture':row.get('architecture'),'compute_capability':row.get('compute_capability'),
                'runtime_vendor':row.get('runtime_vendor'),'device_type':row.get('device_type')})
    nvidia=[row for row in rows.values() if str(row.get('uuid','')).startswith('GPU-') and row.get('present')]
    allowed=None;visibility_error=None
    try:
        observed=[row for row in snapshots['pytorch'].get('devices',[]) if str(row.get('uuid','')).startswith('GPU-') and row.get('present',True)]
        supplied=(observed if snapshots['pytorch']['scan_status']=='current' and 'CUDA_VISIBLE_DEVICES' in constraints else None)
        allowed={row['uuid'].lower() for row in visible_nvidia_records(nvidia,visible=supplied)}
    except (OSError,ValueError) as exc:visibility_error=str(exc)
    for row in rows.values():
        identity=str(row.get('uuid') or row['id']);status='unknown'
        if row['id']=='cpu':status='visible'
        elif identity.startswith('GPU-') and allowed is not None:status='visible' if identity.lower() in allowed else 'masked'
        elif any(view['available'] for view in row['runtimes']):status='visible'
        row['visibility']={'status':status,'constraints':constraints}
        if identity.startswith('GPU-') and visibility_error:row['visibility']['reason']=visibility_error
        row['signature']=device_signature(row)
        row.setdefault('device_type','accelerator');row['qualified']=None
    return {'version':2,'schema_version':2,'backend':'all','host_id':snapshots['system'].get('host_id'),
        'scanned_at':now,'devices':list(rows.values()),'providers':providers,'probe_errors':errors,
        'visibility':{'constraints':constraints},'qualification':None,
        'scope':'Installed hardware and separately observed runtime visibility. Enumeration does not prove model/device execution.'}


def assignment_catalog(spec=None, refresh=False):
    snapshot=inventory(refresh,'all');output=[]
    for original in snapshot.get('devices',[]):
        row=dict(original);identity=row.get('id') or row.get('uuid')
        row.setdefault('runtimes',[]);row.setdefault('present',True);row.setdefault('installed',True)
        row.setdefault('visibility',{'status':'unknown','constraints':visibility_constraints()})
        row.setdefault('identity_strength','physical_uuid' if row.get('uuid') else 'runtime_address')
        row.setdefault('stable',row.get('uuid') is not None);row.setdefault('label',row.get('name') or identity)
        row.setdefault('signature',device_signature(row));row.setdefault('device_type','gpu')
        reasons=[];compatible=None;assignable=False
        def reason(code,message):reasons.append({'code':code,'message':message})
        backend=spec.backend if spec is not None else None
        family=('pytorch' if backend in {'transformers','diffusers','xtts','qwen3-tts'} else
                'openvino' if backend in {'openvino','openvino-genai'} else 'onnx' if backend in {'onnx','onnx-diffusers'} else None)
        fixed_cpu=spec is not None and (backend in {'sherpa-onnx','tesseract'} or spec.task in {'vad','diarization'})
        views=[view for view in row['runtimes'] if family is None or view['backend']==family]
        available=[view for view in views if view.get('available')]
        if family in {'openvino','onnx'} and identity!='cpu':
            # CUDA visibility does not describe another runtime's enumeration.
            # Report the selected runtime's own observed scope without changing
            # any environment mask or inventing a cross-runtime device lease.
            row['visibility']={'status':'visible' if available else 'unknown',
                'constraints':row['visibility'].get('constraints',{}),'backend':family}
        if identity=='cpu':
            compatible=True;assignable=True
        elif not row['present']:
            reason('unavailable','This device was absent at the last successful scan.')
        elif row['visibility']['status']=='masked':
            reason('masked','The device is installed but outside inherited runtime visibility; no mask was expanded.')
        elif fixed_cpu:
            compatible=False;reason('cpu_only','This model adapter explicitly supports CPU execution only.')
        elif backend=='openvino-genai' and spec.task=='image' and row.get('device_type')=='npu':
            compatible=False;reason('unsupported_backend','The current OpenVINO GenAI image adapter supports CPU/GPU; NPU images require a separate static preparation path.')
        elif backend in {'llama-cpp','ace-step'} and str(identity).startswith('GPU-'):
            compatible=True;assignable=row['visibility']['status']=='visible' and row.get('discovery_status')=='current'
            if assignable:reason('runtime_unverified','A compatible native runtime and actual GPU offload are checked on load.')
            else:reason('visibility_unverified','Current native GPU visibility could not be established.')
        elif family is not None and available:
            selected=next((view for view in available if view.get('uuid')),available[0])
            if selected.get('uuid'):
                identity=selected['uuid'];row['runtime_address']=selected['runtime_address']
                row['identity_kind']=selected['identity_kind'];row['identity_strength']=selected['identity_strength'];row['stable']=True
                physical_signature=device_signature(original)
                row['physical_id']=original.get('id') or original.get('uuid');row['physical_signature']=physical_signature
                chosen=device_signature(selected)
                if chosen['runtime_vendor']==physical_signature['runtime_vendor'] and physical_signature['architecture']:
                    chosen['architecture']=physical_signature['architecture']
                row.update(uuid=identity,runtime_vendor=chosen['runtime_vendor'],architecture=chosen['architecture'])
                if chosen['runtime_vendor']!='cuda':row.pop('compute_capability',None);row.pop('capability',None)
                row['signature']=device_signature(row)
                compatible=True;assignable=True
            else:
                reason('weak_identity','This runtime exposes an address without a persistent vendor UUID; use its explicit expert configuration and runtime identity check.')
        elif spec is None and row['stable'] and row['visibility']['status']=='visible':
            assignable=isinstance(identity,str) and identity.startswith(('GPU-','HIP-','XPU-','OV-'))
            if not assignable:reason('weak_identity','Choose a model to resolve the matching runtime UUID; an OS instance identifier is not a runtime selector.')
        else:
            reason('runtime_missing' if not views else 'unavailable','No compatible visible runtime device was observed for this installed device.')
        if row.get('software'):
            compatible=False;assignable=False;reason('unsupported_backend','Software display adapters are not physical accelerator assignments; choose CPU explicitly.')
        row.update(id=identity,compatible=compatible,assignable=assignable,reasons=reasons)
        if row.get('memory_bytes') is not None and row.get('memory_mib') is None:row['memory_mib']=row['memory_bytes']//(1024*1024)
        output.append(row)
    output.sort(key=lambda row:(row['id']=='cpu',str(row['label']),str(row['id'])))
    return {'object':'list','schema_version':2,'data':output,'probe_errors':snapshot.get('probe_errors',[]),
            'visibility':snapshot.get('visibility',{'constraints':visibility_constraints()}),'providers':snapshot.get('providers',[]),'qualification':None}


def resolve_identity(identity,backend):
    matches=[r for r in runtime_records(backend) if r.get('uuid') and r['uuid'].lower()==identity.lower()]
    if len(matches)!=1:
        raise ValueError(f'Pinned device {identity!r} is absent or ambiguous in {backend}; no name/index substitution was used')
    return matches[0]['runtime_address']


def pin_runtime(address,backend):
    records=runtime_records(backend)
    matches=[r for r in records if r['runtime_address']==address]
    if not matches and backend=='openvino' and address in {'GPU.0','NPU.0'}:
        matches=[r for r in records if r['runtime_address']==address.split('.')[0]]
    if len(matches)!=1:
        raise ValueError(f'Device {address!r} is absent or ambiguous in {backend}')
    return matches[0]['uuid'] or address


def hip_records(environ=None):
    """Observe the inherited HIP/ROCR scope; discovery cannot grant more devices."""
    return [r for r in _probe(['--runtime-devices'], environ) if r.get('uuid','') and r['uuid'].startswith('HIP-')]


def pin_hip(selectors, records=None, indices=False):
    rows=hip_records() if records is None else records
    found=[]
    for selector in selectors:
        matches=[r for r in rows if selector in {r['uuid'],r['runtime_address'].split(':')[-1]}]
        if len(matches)!=1:
            raise ValueError('ROCm identity missing, ambiguous, or unavailable; no name fallback is used')
        found.append(matches[0]['runtime_address'].split(':')[-1] if indices else matches[0]['uuid'])
    if len(set(found))!=len(found):
        raise ValueError('Duplicate physical ROCm assignment')
    return found
