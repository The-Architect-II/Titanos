"""Join a DXGI adapter LUID to its PCI address on Windows (D3DKMT), so a DirectML adapter can be tied to the physical GPU that
nvidia-smi names. Without this join an allowed-GPU list cannot be applied to DirectML slots; when the join fails the caller
must exclude the adapter rather than guess."""
import ctypes
import sys

KMTQAITYPE_ADAPTERADDRESS = 6


class LUID(ctypes.Structure):
    _fields_ = [('LowPart', ctypes.c_ulong), ('HighPart', ctypes.c_long)]


class D3DKMT_OPENADAPTERFROMLUID(ctypes.Structure):
    _fields_ = [('AdapterLuid', LUID), ('hAdapter', ctypes.c_uint32)]


class D3DKMT_ADAPTERADDRESS(ctypes.Structure):
    _fields_ = [('BusNumber', ctypes.c_uint32), ('DeviceNumber', ctypes.c_uint32), ('FunctionNumber', ctypes.c_uint32)]


class D3DKMT_QUERYADAPTERINFO(ctypes.Structure):
    _fields_ = [('hAdapter', ctypes.c_uint32), ('Type', ctypes.c_uint32), ('pPrivateDriverData', ctypes.c_void_p), ('PrivateDriverDataSize', ctypes.c_uint32)]


class D3DKMT_CLOSEADAPTER(ctypes.Structure):
    _fields_ = [('hAdapter', ctypes.c_uint32)]


def pci_address_for_luid(luid_text):
    """luid_text as Titanos' dxgi module prints it ('HHHHHHHH:LLLLLLLL'). Returns 'bus:device.function' like '02:00.0' or None."""
    if sys.platform != 'win32':
        return None
    try:
        high_s, low_s = luid_text.split(':')
        high, low = int(high_s, 16), int(low_s, 16)
    except (ValueError, AttributeError):
        return None
    gdi = ctypes.WinDLL('gdi32')
    for hp, lp in ((high, low), (low, high)):
        o = D3DKMT_OPENADAPTERFROMLUID()
        o.AdapterLuid.LowPart = lp & 0xFFFFFFFF
        o.AdapterLuid.HighPart = ctypes.c_long(hp & 0xFFFFFFFF).value
        if gdi.D3DKMTOpenAdapterFromLuid(ctypes.byref(o)) != 0:
            continue
        try:
            addr = D3DKMT_ADAPTERADDRESS()
            q = D3DKMT_QUERYADAPTERINFO(o.hAdapter, KMTQAITYPE_ADAPTERADDRESS, ctypes.cast(ctypes.pointer(addr), ctypes.c_void_p), ctypes.sizeof(addr))
            if gdi.D3DKMTQueryAdapterInfo(ctypes.byref(q)) == 0:
                return f'{addr.BusNumber:02x}:{addr.DeviceNumber:02x}.{addr.FunctionNumber:x}'
        finally:
            gdi.D3DKMTCloseAdapter(ctypes.byref(D3DKMT_CLOSEADAPTER(o.hAdapter)))
    return None


def nvidia_pci_to_uuid():
    """{'02:00.0': 'GPU-...'} from nvidia-smi; empty when nvidia-smi is unavailable."""
    import subprocess
    out = {}
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader'], capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return out
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 2:
            bus = parts[1].split(':', 1)[1].lower() if ':' in parts[1] else parts[1].lower()  # '00000000:02:00.0' -> '02:00.0'
            out[bus] = parts[0]
    return out


if __name__ == '__main__':
    import json
    from titanos.dxgi import adapters
    pci = nvidia_pci_to_uuid()
    for a in adapters():
        addr = pci_address_for_luid(a['luid'])
        print(json.dumps({'device_id': a['device_id'], 'name': a['name'], 'luid': a['luid'], 'pci': addr, 'gpu_uuid': pci.get(addr)}))
