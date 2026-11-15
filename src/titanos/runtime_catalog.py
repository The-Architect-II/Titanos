"""Resolve the trusted native catalog and describe host-compatible profiles."""
import json
import math
import os
import re
from pathlib import Path

from .hardware import discover,read_location


def packaged_catalog():
    path=Path(__file__).with_name("data")/"native-profiles.json"
    if not path.is_file():raise ValueError("Packaged native runtime catalog is missing")
    return path


def resolve_catalog(explicit=None):
    """Explicit app option, then environment override, then shipped package data."""
    return str(explicit or os.environ.get("TITANOS_NATIVE_CATALOG") or packaged_catalog())


def cuda_driver_version():
    """Read the installed driver's API ceiling without creating a CUDA context."""
    import ctypes
    try:
        driver=ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')
        value=ctypes.c_int()
        if driver.cuDriverGetVersion(ctypes.byref(value))!=0:return None
        return value.value
    except (OSError,AttributeError):
        return None


def compatible_profiles(catalog):
    raw=read_location(str(catalog))
    if len(raw)>1024*1024:raise ValueError("Native runtime catalog exceeds 1 MiB")
    value=json.loads(raw)
    if value.get("version")!=1 or value.get("runtime")!="llama.cpp":
        raise ValueError("Unsupported native runtime catalog")
    hardware=discover();rows=[]
    visible=[];visibility_error=None
    if hardware.get('nvidia'):
        try:
            from .devices import visible_nvidia_records
            visible=visible_nvidia_records(hardware['nvidia'])
        except (OSError,ValueError) as exc:
            visibility_error=str(exc)
    driver_version=cuda_driver_version() if visible else None
    try:
        capabilities=[float(device.get('capability',device.get('compute_capability'))) for device in visible]
        if any(not math.isfinite(value) or value<=0 for value in capabilities):capabilities=[]
    except (TypeError,ValueError):
        capabilities=[]
    for identifier,item in value.get("profiles",{}).items():
        if (item.get("os"),item.get("arch"))!=(hardware.get("os"),hardware.get("arch")):continue
        lower=identifier.lower()
        cpu=("cpu" in lower or identifier in {"ubuntu-x64","ubuntu-arm64","macos-x64","macos-arm64"})
        flavor=("CPU" if cpu else "CUDA" if "cuda" in lower else "Vulkan" if "vulkan" in lower
                else "OpenVINO" if "openvino" in lower else "ROCm" if "rocm" in lower
                else "SYCL" if "sycl" in lower else "OpenCL" if "opencl" in lower else "Native")
        # Catalog presence offers a managed installation, not device qualification.
        candidate=False;reason='Explicit runtime choice; device support is verified when loading.'
        version=re.search(r'cuda-(\d+)\.(\d+)',lower)
        if flavor=='CUDA' and version:
            required=int(version[1])*1000+int(version[2])*10
            # CUDA13 dropped pre-Turing libraries; Blackwell first gained native toolkit support in12.8.
            architecture_ok=bool(capabilities and min(capabilities)>=(7.5 if required>=13000 else 5.0)
                                 and (max(capabilities)<10 or required>=12080))
            candidate=bool(visible and driver_version and required<=driver_version and architecture_ok)
            reason=('Visible NVIDIA architecture and driver API match this CUDA runtime candidate; binary kernels and model execution remain unverified.' if candidate else
                    visibility_error or 'No compatible visible NVIDIA device and driver API were established for this runtime.')
        rows.append({"id":identifier,"title":f"{flavor} · {item['os']} {item['arch']}",
                     "runtime":value["runtime"],"release":value["release"],
                     "license":item.get("license",value.get("license","unknown")),
                     "validation":item.get("validation"),"one_stop_supported":True,
                     "runtime_kind":flavor.lower(),"recommended":False,"selection_reason":reason,
                     "cuda_candidate":candidate,"cuda_version":required if flavor=='CUDA' and version else None,
                     "cpu_default":cpu})
    candidates=[row for row in rows if row['cuda_candidate']]
    preferred=max(candidates,key=lambda row:row['cuda_version']) if candidates else next((row for row in rows if row['cpu_default']),None)
    if preferred is not None:
        preferred['recommended']=True
        if not candidates:
            preferred['selection_reason']=visibility_error or 'Portable default; choose another managed runtime for hardware it supports.'
    rows.sort(key=lambda row:(not row['recommended'],not row['cpu_default'],row['title'],row['id']))
    for row in rows:
        for name in ('cuda_candidate','cuda_version','cpu_default'):row.pop(name)
    return rows
