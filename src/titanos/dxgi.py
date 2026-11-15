"""Read-only DXGI adapter order used by ORT's DirectML device_id."""
import ctypes as c,uuid
from ctypes import wintypes as w
def adapters():
 class LUID(c.Structure):_fields_=[('LowPart',w.DWORD),('HighPart',w.LONG)]
 class DESC(c.Structure):_fields_=[('Description',w.WCHAR*128),('VendorId',w.UINT),('DeviceId',w.UINT),('SubSysId',w.UINT),('Revision',w.UINT),('DedicatedVideoMemory',c.c_size_t),('DedicatedSystemMemory',c.c_size_t),('SharedSystemMemory',c.c_size_t),('AdapterLuid',LUID),('Flags',w.UINT)]
 def call(ptr,index,result,args,*values):
  table=c.cast(ptr,c.POINTER(c.POINTER(c.c_void_p))).contents
  return c.WINFUNCTYPE(result,c.c_void_p,*args)(table[index])(ptr,*values)
 factory=c.c_void_p();iid=(c.c_ubyte*16).from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
 hr=c.WinDLL('dxgi').CreateDXGIFactory1(c.byref(iid),c.byref(factory))
 if hr:raise OSError('CreateDXGIFactory1 failed: '+hex(hr&0xffffffff))
 result=[]
 try:
  for i in range(32):
   adapter=c.c_void_p();hr=call(factory,12,w.LONG,[w.UINT,c.POINTER(c.c_void_p)],i,c.byref(adapter))
   if (hr&0xffffffff)==0x887a0002:break
   if hr:raise OSError('EnumAdapters1 failed')
   try:
    desc=DESC();hr=call(adapter,10,w.LONG,[c.POINTER(DESC)],c.byref(desc))
    if hr:raise OSError('GetDesc1 failed')
    result.append({'device_id':i,'name':desc.Description,'vendor_id':hex(desc.VendorId),'device_id_hex':hex(desc.DeviceId),'luid':f'{desc.AdapterLuid.HighPart&0xffffffff:08x}:{desc.AdapterLuid.LowPart:08x}','dedicated_video_bytes':desc.DedicatedVideoMemory,'shared_system_bytes':desc.SharedSystemMemory,'software':bool(desc.Flags&2)})
   finally:call(adapter,2,w.ULONG,[])
 finally:call(factory,2,w.ULONG,[])
 return result
if __name__=='__main__':
 import json
 print(json.dumps(adapters(),indent=2))
