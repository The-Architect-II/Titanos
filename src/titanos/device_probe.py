"""Verify CUDA runtime UUID order in a fresh subprocess without Torch."""
import ctypes
import ctypes.util
import json
import os
import sys
import uuid
import hashlib
import re
import subprocess
from contextlib import ExitStack
from pathlib import Path


def dxcore_records():
    """OS-reported GPU/NPU class and integration; no device/context creation.

    ABI: Microsoft DirectX-Headers dxcore_interface.h (MIT). No memory budget,
    reservation, Direct3D device or inference operation is requested.
    """
    c=ctypes
    from ctypes import wintypes as w
    def guid(value):return (c.c_ubyte*16).from_buffer_copy(uuid.UUID(value).bytes_le)
    def call(ptr,index,result,args,*values):
        table=c.cast(ptr,c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(result,c.c_void_p,*args)(table[index])(ptr,*values)
    def require(code):
        if code:raise OSError('DXCore enumeration failed: '+hex(code&0xffffffff))
    factory=c.c_void_p();interface=guid('78ee5945-c36e-4b13-a669-005dd11c0f06')
    library=c.WinDLL(str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/dxcore.dll'))
    create=library.DXCoreCreateAdapterFactory
    create.argtypes=[c.c_void_p,c.POINTER(c.c_void_p)];create.restype=w.LONG
    require(create(c.byref(interface),c.byref(factory)))
    rows={};list_iid=guid('526c7776-40e9-459b-b711-f32ad76dfc28');adapter_iid=guid('f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e')
    attributes=(('gpu','b69eb219-3ded-4464-979f-a00bd4687006'),('accelerator','e0b195da-58ef-4a22-90f1-1f28169cab8d'),
                ('npu','d46140c4-add7-451b-9e56-06fe8c3b58ed'),('gpu','0c9ece4d-2f6e-4f01-8c96-e89e331b47b1'))
    try:
        for category,value in attributes:
            listing=c.c_void_p();attribute=guid(value)
            code=call(factory,3,w.LONG,[w.UINT,c.c_void_p,c.c_void_p,c.POINTER(c.c_void_p)],1,c.byref(attribute),c.byref(list_iid),c.byref(listing))
            if code:continue  # Older DXCore versions lack newer hardware attributes.
            try:
                count=call(listing,4,w.UINT,[])
                if count>256:raise ValueError('DXCore returned too many adapters')
                for index in range(count):
                    adapter=c.c_void_p()
                    require(call(listing,3,w.LONG,[w.UINT,c.c_void_p,c.POINTER(c.c_void_p)],index,c.byref(adapter_iid),c.byref(adapter)))
                    try:
                        def prop(key,kind):
                            if not call(adapter,5,c.c_bool,[w.UINT],key):return None
                            buffer=kind();require(call(adapter,6,w.LONG,[w.UINT,c.c_size_t,c.c_void_p],key,c.sizeof(buffer),c.byref(buffer)))
                            return buffer
                        luid=prop(0,c.c_uint64);hardware=prop(3,c.c_uint32*4)
                        if luid is None or hardware is None:continue
                        description_size=c.c_size_t()
                        require(call(adapter,7,w.LONG,[w.UINT,c.POINTER(c.c_size_t)],2,c.byref(description_size)))
                        if not 0<description_size.value<=4096:raise ValueError('Invalid DXCore description length')
                        text=c.create_string_buffer(description_size.value)
                        require(call(adapter,6,w.LONG,[w.UINT,c.c_size_t,c.c_void_p],2,description_size.value,text))
                        integrated=prop(12,c.c_bool);is_hardware=prop(11,c.c_bool)
                        key=f'{luid.value>>32:08x}:{luid.value&0xffffffff:08x}'
                        prior=rows.get(key,{})
                        rows[key]={'luid':key,'name':text.value.decode('utf-8',errors='replace'),
                            'vendor_id':int(hardware[0]),'device_id':int(hardware[1]),'subsystem_id':int(hardware[2]),'revision':int(hardware[3]),
                            'device_type':'npu' if category=='npu' or prior.get('device_type')=='npu' else category,
                            'integrated':integrated.value if integrated is not None else None,
                            'software':not is_hardware.value if is_hardware is not None else None}
                    finally:call(adapter,2,w.ULONG,[])
            finally:call(listing,2,w.ULONG,[])
    finally:call(factory,2,w.ULONG,[])
    return list(rows.values())


def windows_records():
    """Present display/compute PnP instances, independent of installed runtimes."""
    command=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
    script=r'''$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
@(Get-PnpDevice -PresentOnly | Where-Object {$_.Class -in @('Display','ComputeAccelerator','NeuralProcessor')} | ForEach-Object {
  $device=$_
  $props=@(Get-PnpDeviceProperty -InstanceId $device.InstanceId -KeyName 'DEVPKEY_Device_BusNumber','DEVPKEY_Device_Address','DEVPKEY_Device_ProblemCode' -ErrorAction SilentlyContinue)
  [pscustomobject]@{instance_id=$device.InstanceId;name=$device.FriendlyName;class=$device.Class;status=[string]$device.Status
    bus_number=($props|Where-Object KeyName -eq 'DEVPKEY_Device_BusNumber').Data
    address=($props|Where-Object KeyName -eq 'DEVPKEY_Device_Address').Data
    problem_code=($props|Where-Object KeyName -eq 'DEVPKEY_Device_ProblemCode').Data}
}) | ConvertTo-Json -Depth 4 -Compress'''
    completed=subprocess.run([str(command),'-NoProfile','-NonInteractive','-Command',script],
        check=True,capture_output=True,text=True,encoding='utf-8',timeout=25,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    values=json.loads(completed.stdout or '[]')
    if isinstance(values,dict):values=[values]
    if not isinstance(values,list) or len(values)>256:raise ValueError('Invalid PnP device inventory')
    try:dx=dxcore_records()
    except (OSError,ValueError):dx=[]
    rows=[]
    for value in values:
        instance=value.get('instance_id')
        if not isinstance(instance,str) or not instance:continue
        ids=re.search(r'VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})(?:&SUBSYS_([0-9A-F]{8}))?(?:&REV_([0-9A-F]{2}))?',instance.upper())
        hardware=tuple(int(item,16) if item else None for item in ids.groups()) if ids else None
        matches=[row for row in dx if hardware and row['vendor_id']==hardware[0] and row['device_id']==hardware[1]
                 and (hardware[2] is None or row['subsystem_id']==hardware[2]) and (hardware[3] is None or row['revision']==hardware[3])]
        # A hardware-ID tuple is not a unique identity. Enrich only if both
        # views have exactly one matching tuple in this complete snapshot.
        count=sum(1 for row in values if hardware and re.search(r'VEN_'+format(hardware[0],'04X')+'&DEV_'+format(hardware[1],'04X'),str(row.get('instance_id','')).upper()))
        details=matches[0] if len(matches)==1 and count==1 else {}
        category=details.get('device_type') or ('gpu' if value.get('class')=='Display' else 'accelerator')
        bus,address=value.get('bus_number'),value.get('address')
        pci=(f'{bus:02x}:{address>>16:02x}.{address&0xffff:x}' if instance.upper().startswith('PCI\\') and type(bus) is int and type(address) is int else None)
        rows.append({'id':'PNP-'+hashlib.sha256(instance.upper().encode()).hexdigest()[:32], 'uuid':None,
            'name':value.get('name') or instance,'pnp_instance':instance,'identity_kind':'pnp_instance',
            'identity_strength':'host_device_instance','stable':True,'device_type':category,
            'npu_candidate':value.get('class') in {'ComputeAccelerator','NeuralProcessor'},
            'device_subtype':('integrated' if details.get('integrated') is True else 'discrete' if details.get('integrated') is False and category=='gpu' else None),
            'pnp_class':value.get('class'),'driver_status':value.get('status'),'problem_code':value.get('problem_code'),
            'pci_location':pci,'vendor_id':hardware[0] if hardware else None,'device_id':hardware[1] if hardware else None,
            'luid':details.get('luid'),'category_source':'dxcore_attribute' if details else 'pnp_class',
            'identity_join':'unique_hardware_tuple_this_scan' if details else None,'runtime_address':None})
    return rows


def cuda_records():
    """Read runtime UUID order under current masks, without allocating tensors."""
    explicit=os.environ.get('TITANOS_CUDA_LIBRARY_DIR')
    directories=([Path(explicit).absolute()] if explicit else
                 [Path(sys.executable).parent/'_internal/torch/lib',Path(sys.prefix)/'Lib/site-packages/torch/lib',
                  Path(sys.prefix)/'lib',Path('/usr/local/cuda/lib64')])
    names=[]
    for directory in directories:
        names.extend(sorted(str(p) for p in directory.glob('cudart64*.dll')))
        names.extend(sorted(str(p) for p in directory.glob('libcudart.so*')))
    if not explicit:
        names.extend([ctypes.util.find_library('cudart'),'libcudart.so','libcudart.so.12','libcudart.so.13'])
    # The selected native runtime owns its DLL cohort, not the frozen parent's
    # default Torch directory. Visibility variables are deliberately untouched.
    with ExitStack() as stack:
        if os.name=='nt':
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.SetDllDirectoryW.argtypes=[ctypes.c_wchar_p]
            kernel.SetDllDirectoryW.restype=ctypes.c_int
            if not kernel.SetDllDirectoryW(None):raise ctypes.WinError(ctypes.get_last_error())
            if explicit:
                stack.enter_context(os.add_dll_directory(str(directories[0])))
        lib=None
        for name in names:
            if name:
                try:
                    lib=ctypes.CDLL(name)
                    break
                except OSError:
                    pass
        if lib is None:
            raise ValueError('A compatible CUDA runtime library is required to map current visibility to physical UUIDs')
        return _cuda_rows(lib)


def _cuda_rows(lib, driver=None):
    """Join this CUDArt's visible PCI order to driver UUIDs, without contexts.

    CUDArt exports no cudaDeviceGetUuid. Avoid a version-dependent device-property
    structure: use its documented PCI query and the driver's fixed UUID ABI.
    """
    def check(result):
        if result:
            raise ValueError(f'CUDA runtime probe returned error {result}')
    def function(library, name, arguments):
        try:value=getattr(library,name)
        except AttributeError as exc:raise ValueError('CUDA identity API unavailable: '+name) from exc
        value.argtypes=arguments;value.restype=ctypes.c_int
        return value
    def pci_key(value):
        text=value.decode('ascii')
        if not re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',text):
            raise ValueError('CUDA returned an invalid PCI identity')
        return tuple(int(part,16) for part in re.split('[:.]',text))
    count=ctypes.c_int()
    check(function(lib,'cudaGetDeviceCount',[ctypes.POINTER(ctypes.c_int)])(ctypes.byref(count)))
    if not 0 <= count.value <= 256:raise ValueError('CUDA returned an invalid device count')
    if not count.value:return []
    if driver is None:
        driver=(ctypes.WinDLL(str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/nvcuda.dll'))
                if os.name=='nt' else ctypes.CDLL('libcuda.so.1'))
    check(function(driver,'cuInit',[ctypes.c_uint])(0))
    runtime_pci=function(lib,'cudaDeviceGetPCIBusId',[ctypes.c_void_p,ctypes.c_int,ctypes.c_int])
    driver_device=function(driver,'cuDeviceGetByPCIBusId',[ctypes.POINTER(ctypes.c_int),ctypes.c_char_p])
    driver_pci=function(driver,'cuDeviceGetPCIBusId',[ctypes.c_void_p,ctypes.c_int,ctypes.c_int])
    legacy_uuid=function(driver,'cuDeviceGetUuid',[ctypes.c_void_p,ctypes.c_int])
    instance_uuid=function(driver,'cuDeviceGetUuid_v2',[ctypes.c_void_p,ctypes.c_int])
    actual=[];locations=set();identities=set()
    for i in range(count.value):
        pci=ctypes.create_string_buffer(32);check(runtime_pci(pci,len(pci),i));location=pci_key(pci.value)
        if location in locations:raise ValueError('Ambiguous CUDA PCI mapping; MIG or duplicate devices require a separate identity path')
        locations.add(location)
        device=ctypes.c_int();check(driver_device(ctypes.byref(device),pci.value))
        observed_pci=ctypes.create_string_buffer(32);check(driver_pci(observed_pci,len(observed_pci),device.value))
        if pci_key(observed_pci.value)!=location:raise ValueError('CUDA runtime and driver PCI identities differ')
        value=(ctypes.c_ubyte*16)();physical=(ctypes.c_ubyte*16)()
        check(instance_uuid(ctypes.byref(value),device.value));check(legacy_uuid(ctypes.byref(physical),device.value))
        if bytes(value)!=bytes(physical):raise ValueError('MIG instance identity is unsupported by physical-GPU assignments')
        identity='GPU-'+str(uuid.UUID(bytes=bytes(value)))
        if not any(value) or identity in identities:raise ValueError('CUDA returned an empty or duplicate UUID')
        identities.add(identity)
        actual.append({'local_device':f'CUDA{i}','runtime_address':f'cuda:{i}',
                       'uuid':identity,'pci_bus_id':pci.value.decode('ascii'),
                       'identity_binding':'cudart_visible_pci_to_driver_uuid_v2'})
    return actual


def verify(expected):
    rows=cuda_records()
    if [row['uuid'].lower() for row in rows] != [value.lower() for value in expected]:
        raise ValueError('CUDA runtime UUID order mismatch: '+str([row['uuid'] for row in rows]))
    return rows


def main():
    if sys.argv[1:] == ['--runtime-devices']:
        from .devices import runtime_records
        result = runtime_records('pytorch')
    elif sys.argv[1:] == ['--cuda-devices']:
        result = cuda_records()
    elif len(sys.argv)==3 and sys.argv[1]=='--inventory':
        from .devices import runtime_records
        result = runtime_records(sys.argv[2])
    else:
        result = verify(sys.argv[1:])
    print(json.dumps(result))


if __name__=='__main__':
    main()
