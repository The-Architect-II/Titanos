"""Identity boundary checks with deliberately reordered synthetic GPUs; not GPU inference evidence."""
import json
import os
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from titanos.devices import inventory, pin, runtime_device, verify_visible, pin_hip, normalized_uuid, resolve_identity
from titanos.config import Config, ModelSpec, WorkerSpec, write_config, read_config
from titanos.cli import main

A='GPU-11111111-1111-1111-1111-111111111111'
B='GPU-22222222-2222-2222-2222-222222222222'
C='GPU-33333333-3333-3333-3333-333333333333'
rows=[dict(uuid=u,index=i,name=n,pci_bus_id=f'0000:0{i+1}:00.0',memory_mib=m,compute_capability='12.0',driver='test')
 for i,(u,n,m) in enumerate([(A,'5090',32768),(B,'5090',32768),(C,'5060 Ti',16384)])]

def rejected(call):
    try:
        call()
        raise AssertionError('Expected refusal')
    except ValueError:
        pass


def main_check():
    assert normalized_uuid(bytes.fromhex('11'*16))=='11111111-1111-1111-1111-111111111111'
    assert normalized_uuid(bytes(16)) is None
    hip=[{'uuid':'HIP-'+A[4:],'runtime_address':'cuda:1'},{'uuid':'HIP-'+B[4:],'runtime_address':'cuda:0'}]
    assert pin_hip(['0'],hip)==['HIP-'+B[4:]]
    assert pin_hip(['HIP-'+A[4:]],hip,indices=True)==['1']
    rejected(lambda:pin_hip(['0','HIP-'+B[4:]],hip))
    with patch('titanos.devices.runtime_records',return_value=[{'uuid':'XPU-'+A[4:],'runtime_address':'xpu:2'}]):
        assert resolve_identity('XPU-'+A[4:],'pytorch')=='xpu:2'
        rejected(lambda:resolve_identity('XPU-'+B[4:],'pytorch'))
    from titanos.config import NativeOptions, native_assignment
    automatic=ModelSpec(path='.',task='chat',backend='llama-cpp',device='auto',native=NativeOptions(model_file='model.gguf'))
    assert automatic.device=='auto' and automatic.native.gpu_layers=='auto'
    model=ModelSpec(path='.',task='chat',backend='llama-cpp',device=A,native=NativeOptions(model_file='model.gguf',gpu_layers=99))
    assert native_assignment(model,WorkerSpec(models=['m'],device=B)).device==B
    assert native_assignment(model,WorkerSpec(models=['m'],device='auto')).device==A
    cpu=native_assignment(model,WorkerSpec(models=['m']))
    assert cpu.device=='cpu' and cpu.native.gpu_layers==0
    assert pin(['2','0'],rows)==[C,A]
    reordered=[{**d,'index':2-d['index']} for d in rows]
    assert pin([C,A],reordered)==[C,A]
    rejected(lambda:pin(['0',A],rows))
    rejected(lambda:pin([C],rows[:2]))
    torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2,
        get_device_properties=lambda i:SimpleNamespace(uuid=[C,A][int(str(i).split(':')[-1])][4:])))
    assert runtime_device(C,torch)=='cuda:0'
    assert verify_visible([C,A],torch)[1]['uuid']==A
    rejected(lambda:verify_visible([A,C],torch))
    rejected(lambda:runtime_device(B,torch))
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'TITANOS_HOME':tmp,'CUDA_VISIBLE_DEVICES':','.join([A,B,C])}):
        with patch('titanos.devices.scan',return_value=rows):
            first=inventory(True)
        with patch('titanos.devices.scan',return_value=reordered[:2]):
            second=inventory(True)
        assert first['host_id']==second['host_id']
        assert next(d for d in second['devices'] if d['uuid']==C)['present'] is False
        before=Path(tmp,'hardware.json').read_bytes()
        with patch('titanos.devices.scan',side_effect=ValueError('driver failure')):
            rejected(lambda:inventory(True))
        assert Path(tmp,'hardware.json').read_bytes()==before
        registry=Path(tmp,'models.json')
        write_config(registry,Config(models={'chat':ModelSpec(path=tmp,task='chat')}))
        real_torch=SimpleNamespace(version=SimpleNamespace(hip=None),cuda=SimpleNamespace(
            is_available=lambda:True,device_count=lambda:2,get_device_properties=torch.cuda.get_device_properties))
        visible=[{'uuid':identity,'runtime_address':'cuda:'+str(index)} for index,identity in enumerate([A,B,C])]
        with patch('titanos.devices.scan',return_value=rows), patch.dict(sys.modules,{'torch':real_torch}),patch('titanos.devices.probe_visible_cuda',return_value=visible):
            main(['--config',str(registry),'worker','voice','--models','chat','--gpus','2'])
            assert read_config(registry).workers['voice'].gpus==[C]
            cfg=read_config(registry)
            cfg.workers['legacy']=WorkerSpec(models=['chat'],gpus=['0'])
            write_config(registry,cfg)
            cfg.models['chat'].device='cuda:0'
            write_config(registry,cfg)
            with patch.object(real_torch.cuda,'get_device_properties',torch.cuda.get_device_properties):
                main(['--config',str(registry),'pin-devices'])
            assert read_config(registry).models['chat'].device==C
            assert read_config(registry).workers['legacy'].gpus==[A]
        from titanos.replicas import resolve_gpus
        with patch.dict(sys.modules,{'torch':real_torch}):
            rejected(lambda:resolve_gpus(['0']))
        from titanos.hardware import select_device
        Path(tmp,'titanos-runtime.json').write_text(json.dumps({'backend':'pytorch','device':'cuda:0'}))
        with patch('sys.prefix',tmp), patch.dict(sys.modules,{'torch':real_torch}):
            rejected(lambda:select_device('auto'))
    print('PASS: mixed GPUs, reordered indices, duplicate/missing cards, runtime UUID verification, persistent inventory, scan failure preservation, CLI pinning and migration; ROCm UUID reorder and XPU exact-identity resolution')

if __name__=='__main__':main_check()
