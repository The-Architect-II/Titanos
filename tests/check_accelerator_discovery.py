"""Controlled hardware, visibility and selection contracts; no driver/model calls."""
import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import uuid

from titanos import devices
from titanos.device_probe import _cuda_rows, windows_records
from titanos.hardware import select_device

A='GPU-11111111-1111-1111-1111-111111111111'
B='GPU-22222222-2222-2222-2222-222222222222'
H='HIP-33333333-3333-3333-3333-333333333333'
PHYSICAL=[{'uuid':A,'index':0,'name':'GPU A','capability':12.0,'driver':'controlled'},
          {'uuid':B,'index':1,'name':'GPU B','capability':12.0,'driver':'controlled'}]


def cuda_fixture(*, duplicate_pci=False, mismatched_pci=False, mig=False):
    def count(pointer):ctypes.cast(pointer,ctypes.POINTER(ctypes.c_int))[0]=2;return 0
    def pci(pointer,size,index):
        value=[b'0000:02:00.0',b'0000:01:00.0'][0 if duplicate_pci else index]
        ctypes.memmove(pointer,value+b'\0',len(value)+1);return 0
    def device(pointer,address):ctypes.cast(pointer,ctypes.POINTER(ctypes.c_int))[0]={b'0000:02:00.0':17,b'0000:01:00.0':29}[address];return 0
    def driver_pci(pointer,size,handle):return pci(pointer,size,1 if mismatched_pci else {17:0,29:1}[handle])
    def identity(pointer,handle):ctypes.memmove(pointer,uuid.UUID({17:B,29:A}[handle][4:]).bytes,16);return 0
    def legacy(pointer,handle):return identity(pointer,29 if mig else handle)
    return NS(cudaGetDeviceCount=count,cudaDeviceGetPCIBusId=pci),NS(cuInit=lambda flags:0,cuDeviceGetByPCIBusId=device,
        cuDeviceGetPCIBusId=driver_pci,cuDeviceGetUuid=legacy,cuDeviceGetUuid_v2=identity)


class Checks(unittest.TestCase):
    def test_runtime_join_cannot_replace_cuda_vendor_or_flatten_architectures(self):
        physical=[dict(PHYSICAL[0]),{**PHYSICAL[1],'capability':8.9}]
        snapshots={'system':[], 'nvidia':physical,'onnx':[],
            'pytorch':[{'id':p['uuid'],'uuid':p['uuid'],'architecture':str(p['capability']),'runtime_vendor':'cuda','runtime_address':'cuda:'+str(i)} for i,p in enumerate(physical)],
            'openvino':[{'id':'OV-'+p['uuid'][4:],'uuid':'OV-'+p['uuid'][4:],'architecture':'same-plugin-architecture','runtime_vendor':'openvino','runtime_address':'GPU.'+str(i)} for i,p in enumerate(physical)]}
        def inventory(refresh,backend):return {'devices':snapshots[backend],'scanned_at':1,'visibility':{}}
        with patch('titanos.devices.inventory',side_effect=inventory),patch('importlib.util.find_spec',return_value=object()),patch.dict(os.environ,{},clear=True):
            joined=devices.discovery(refresh=True)
        cards=[r for r in joined['devices'] if r['id'] in [A,B]]
        self.assertEqual([r['signature'] for r in cards],[{'runtime_vendor':'cuda','architecture':'12.0'},{'runtime_vendor':'cuda','architecture':'8.9'}])
        self.assertTrue(all(r['runtimes'][-1]['architecture']=='same-plugin-architecture' for r in cards))
        with self.assertRaises(ValueError):devices.require_homogeneous(cards)
        with patch('titanos.devices.inventory',return_value=joined):
            torch=devices.assignment_catalog(NS(backend='transformers',task='chat'))
            ov=devices.assignment_catalog(NS(backend='openvino',task='tensor'))
        torch_cards=[r for r in torch['data'] if r['id'] in [A,B]]
        self.assertTrue(all(r['runtime_vendor']=='cuda' for r in torch_cards))
        with self.assertRaises(ValueError):devices.require_homogeneous(torch_cards)
        ov_cards=[r for r in ov['data'] if r['id'].startswith('OV-')]
        self.assertTrue(all(r['signature']=={'runtime_vendor':'openvino','architecture':'same-plugin-architecture'} for r in ov_cards))
        self.assertTrue(all(r['physical_signature']['runtime_vendor']=='cuda' for r in ov_cards))

    def test_installer_discovery_uses_builtin_powershell_without_user_path(self):
        from titanos import hardware
        with patch.object(hardware.platform,'system',return_value='Windows'),patch.object(hardware.shutil,'which',return_value=None),patch.object(hardware.Path,'glob',return_value=[]),patch.object(hardware.subprocess,'run',return_value=NS(stdout='[{"PNPDeviceID":"PCI\\\\VEN_8086"}]')) as command:
            found=hardware.discover()
        self.assertEqual(found['display_vendors'],['intel'])
        self.assertTrue(str(command.call_args.args[0][0]).replace('\\','/').endswith('/System32/WindowsPowerShell/v1.0/powershell.exe'))

    def test_aggregate_links_physical_identity_and_retains_unknown_runtime(self):
        snapshots = {
            'system': [{'id':'cpu','device_type':'cpu'},
                       {'id':'PNP-card','name':'Same display name','vendor_id':0x10de,'pci_location':'01:00.0','stable':True}],
            'nvidia': [{**PHYSICAL[0],'pci_bus_id':'00000000:01:00.0'}],
            'pytorch': [{'id':A,'uuid':A,'name':'GPU A','runtime_address':'cuda:0','device_type':'gpu'}],
            'openvino': [{'id':'GPU.0','name':'Same display name','runtime_address':'GPU.0','device_type':'gpu'}],
            'onnx': [],
        }
        def inventory(refresh, backend):
            if refresh and backend == 'pytorch':raise ValueError('controlled runtime failure')
            return {'devices':snapshots[backend],'scanned_at':1,'visibility':{}}
        with patch('titanos.devices.inventory',side_effect=inventory),patch('importlib.util.find_spec',return_value=object()),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''},clear=True):
            found=devices.discovery(refresh=True)
        rows={row['id']:row for row in found['devices']}
        self.assertNotIn('PNP-card',rows)
        self.assertEqual(rows[A]['aliases'],['PNP-card'])
        self.assertEqual(rows[A]['visibility']['status'],'masked')
        self.assertFalse(rows[A]['runtimes'][0]['available'])
        self.assertIn('runtime_identity_only',rows['GPU.0']['physical_link'])
        self.assertEqual(found['probe_errors'][0]['backend'],'pytorch')
        self.assertIsNone(found['qualification'])
        weak={'id':'PNP-weak','name':'Installed adapter','stable':True,'visibility':{'status':'visible','constraints':{}},'runtimes':[]}
        with patch('titanos.devices.inventory',return_value={'devices':[weak]}):
            choice=devices.assignment_catalog()['data'][0]
        self.assertFalse(choice['assignable'])
        self.assertEqual(choice['reasons'][0]['code'],'weak_identity')

    def test_installed_hardware_survives_empty_runtime_visibility(self):
        self.assertEqual(devices.visible_nvidia_records(PHYSICAL,environ={'CUDA_VISIBLE_DEVICES':''}),[])
        self.assertEqual(len(PHYSICAL),2)
        with self.assertRaisesRegex(ValueError,'masked'):
            devices.admit_gpu_uuids([A],physical=PHYSICAL,environ={'CUDA_VISIBLE_DEVICES':''})

    def test_uuid_mask_order_and_outer_scope(self):
        value=devices.visible_nvidia_records(PHYSICAL,environ={'CUDA_VISIBLE_DEVICES':B+','+A,'NVIDIA_VISIBLE_DEVICES':A})
        self.assertEqual([row['uuid'] for row in value],[A])
        self.assertEqual(devices.admit_gpu_uuids([B],physical=PHYSICAL,environ={'CUDA_VISIBLE_DEVICES':B}),[B])
        with self.assertRaises(ValueError):devices.admit_gpu_uuids([A],physical=PHYSICAL,environ={'CUDA_VISIBLE_DEVICES':B})
        with self.assertRaises(ValueError):devices.visible_nvidia_records(PHYSICAL,environ={'CUDA_VISIBLE_DEVICES':A+','+A})

    def test_numeric_mask_uses_observed_cuda_uuid_not_driver_index(self):
        visible=[{'uuid':B,'runtime_address':'cuda:0'}]
        result=devices.visible_nvidia_records(PHYSICAL,environ={'CUDA_VISIBLE_DEVICES':'0','CUDA_DEVICE_ORDER':'FASTEST_FIRST'},visible=visible)
        self.assertEqual(result[0]['uuid'],B)
        with patch('titanos.devices.probe_visible_cuda',side_effect=ValueError('mapping unavailable')):
            with self.assertRaisesRegex(ValueError,'mapping unavailable'):
                devices.visible_nvidia_records(PHYSICAL,environ={'CUDA_VISIBLE_DEVICES':'0'})
        with self.assertRaises(ValueError):devices.visible_nvidia_records(PHYSICAL,environ={'CUDA_VISIBLE_DEVICES':'1,0'},visible=visible)

    def test_no_mask_and_container_driver_namespace(self):
        self.assertEqual(devices.visible_nvidia_records(PHYSICAL,environ={}),PHYSICAL)
        result=devices.visible_nvidia_records(PHYSICAL,environ={'NVIDIA_VISIBLE_DEVICES':'1'})
        self.assertEqual([row['uuid'] for row in result],[B])
        with self.assertRaises(ValueError):devices.visible_nvidia_records(PHYSICAL,environ={'NVIDIA_VISIBLE_DEVICES':'9'})

    def test_hip_keeps_compound_scope(self):
        env={'HIP_VISIBLE_DEVICES':'0','ROCR_VISIBLE_DEVICES':'1'}
        rows=[{'uuid':H,'runtime_address':'cuda:0'}]
        with patch('titanos.devices._probe',return_value=rows) as probe:
            self.assertEqual(devices.admit_gpu_uuids([H],family='hip',environ=env),[H])
        self.assertEqual(probe.call_args.args,(['--runtime-devices'],env))
        with self.assertRaises(ValueError):devices.admit_gpu_uuids([H],family='hip',visible=rows,environ={'HIP_VISIBLE_DEVICES':''})

    def test_homogeneity_requires_known_actual_architecture(self):
        self.assertEqual(devices.require_homogeneous(PHYSICAL),{'runtime_vendor':'cuda','architecture':'12.0'})
        with self.assertRaises(ValueError):devices.require_homogeneous([{'uuid':A},{'uuid':B}])
        with self.assertRaises(ValueError):devices.require_homogeneous([PHYSICAL[0],{**PHYSICAL[1],'capability':8.9}])

    def test_shared_auto_policy_receipt_aliases_and_task_limits(self):
        rows=[{'id':'cpu','runtime_address':'CPU','available':True},{'id':'ov-gpu','runtime_address':'GPU','available':True},
              {'id':'ov-npu','runtime_address':'NPU','available':True}]
        self.assertEqual(devices.preferred_runtime_device(rows,backend='openvino',task='tensor')['device'],'GPU')
        pin={'backend':'openvino','device':'NPU.0'}
        self.assertEqual(devices.preferred_runtime_device(rows,backend='openvino',task='tensor',installed=pin)['device'],'NPU')
        with self.assertRaises(ValueError):devices.preferred_runtime_device(rows,backend='openvino-genai',task='image',installed=pin)
        self.assertEqual(devices.preferred_runtime_device(rows[:1]+rows[2:],backend='openvino-genai',task='image')['device'],'CPU')

    def test_dml_auto_preserves_explicit_adapter_and_cpu_pin(self):
        rows=[{'id':'dml0','runtime_address':'dml:0','provider':'DmlExecutionProvider','provider_options':{'device_id':'0'},'luid':'zero','available':True},
              {'id':'dml1','runtime_address':'dml:1','provider':'DmlExecutionProvider','provider_options':{'device_id':'1'},'luid':'one','available':True},
              {'id':'cpu','runtime_address':'cpu','provider':'CPUExecutionProvider','available':True}]
        selected=devices.preferred_runtime_device(rows,backend='onnx',options={'device_id':1})
        self.assertEqual((selected['id'],selected['luid']),('dml1','one'))
        cpu=devices.preferred_runtime_device(rows,backend='onnx',installed={'backend':'onnx','device':'cpu'},options={'device_id':1})
        self.assertEqual(cpu['device'],'cpu')
        with self.assertRaises(ValueError):devices.preferred_runtime_device(rows,backend='onnx',options={'device_id':2})
        with self.assertRaises(ValueError):devices.preferred_runtime_device(rows,backend='onnx',options={'device_id':True})

    def test_cuda_abi_reader_reports_uuid_order_without_tensors(self):
        runtime,driver=cuda_fixture()
        self.assertFalse(hasattr(runtime,'cudaDeviceGetUuid'))  # The actual CUDArt ABI has no such export.
        rows=_cuda_rows(runtime,driver)
        self.assertEqual([row['uuid'] for row in rows],[B,A])
        self.assertEqual(rows[0]['runtime_address'],'cuda:0')
        self.assertEqual(rows[0]['pci_bus_id'],'0000:02:00.0')

    def test_cuda_identity_join_rejects_ambiguous_mig_and_missing_abi(self):
        for options,reason in [({'duplicate_pci':True},'Ambiguous'),({'mismatched_pci':True},'identities differ'),({'mig':True},'MIG')]:
            with self.subTest(options=options),self.assertRaisesRegex(ValueError,reason):_cuda_rows(*cuda_fixture(**options))
        runtime,driver=cuda_fixture();del driver.cuDeviceGetUuid_v2
        with self.assertRaisesRegex(ValueError,'cuDeviceGetUuid_v2'):_cuda_rows(runtime,driver)

    def test_failed_probe_retains_exception_before_frozen_wrapper_line(self):
        from titanos.engine import EngineError
        def failed(argv,directory,*args,**kwargs):
            (directory/'stderr.log').write_text("Traceback (most recent call last):\nAttributeError: CUDA symbol missing\n[PYI:ERROR] Failed to execute frozen_entry\n")
            raise EngineError('Device discovery process exited unsuccessfully',502,'native_process_failed')
        with patch('titanos.owned_command.run_owned',side_effect=failed),self.assertRaisesRegex(ValueError,'CUDA symbol missing'):
            devices._probe(['--cuda-devices'],{})

    def test_windows_physical_gpu_igpu_npu_does_not_need_numerical_runtime(self):
        raw=[{'instance_id':r'PCI\VEN_8086&DEV_1234&SUBSYS_00000001&REV_02\integrated','name':'Integrated graphics','class':'Display','status':'OK','problem_code':0,'bus_number':0,'address':2<<16},
             {'instance_id':r'PCI\VEN_8086&DEV_AD1D&SUBSYS_00000001&REV_02\neural','name':'AI accelerator','class':'ComputeAccelerator','status':'OK','problem_code':0,'bus_number':0,'address':11<<16}]
        dx=[{'vendor_id':0x8086,'device_id':0x1234,'subsystem_id':1,'revision':2,'integrated':True,'device_type':'gpu','luid':'00000000:00000001'},
            {'vendor_id':0x8086,'device_id':0xad1d,'subsystem_id':1,'revision':2,'integrated':None,'device_type':'npu','luid':'00000000:00000002'}]
        with patch('titanos.device_probe.subprocess.run',return_value=NS(stdout=json.dumps(raw))) as command,patch('titanos.device_probe.dxcore_records',return_value=dx):
            rows=windows_records()
        self.assertTrue(str(command.call_args.args[0][0]).replace('\\','/').endswith('/System32/WindowsPowerShell/v1.0/powershell.exe'))
        self.assertEqual([row['device_type'] for row in rows],['gpu','npu'])
        self.assertEqual(rows[0]['device_subtype'],'integrated')
        self.assertEqual(rows[1]['identity_strength'],'host_device_instance')
        self.assertEqual(rows[1]['pci_location'],'00:0b.0')

    def test_probe_preserves_masks_and_removes_credentials(self):
        env={'CUDA_VISIBLE_DEVICES':B,'HIP_VISIBLE_DEVICES':'','CUDA_DEVICE_ORDER':'PCI_BUS_ID','HF_TOKEN':'never-forward','OPENAI_API_KEY':'never-forward'}
        def owned(argv,directory,child_env,*args,**kwargs):
            self.assertEqual(child_env['CUDA_VISIBLE_DEVICES'],B)
            self.assertEqual(child_env['HIP_VISIBLE_DEVICES'],'')
            self.assertEqual(child_env['CUDA_DEVICE_ORDER'],'PCI_BUS_ID')
            self.assertNotIn('HF_TOKEN',child_env);self.assertNotIn('OPENAI_API_KEY',child_env)
            (directory/'stdout.log').write_text('[]')
        with patch('titanos.owned_command.run_owned',side_effect=owned):self.assertEqual(devices._probe(['--cuda-devices'],env),[])

    def test_explicit_unavailable_never_falls_back_and_auto_prefers_visible_gpu(self):
        class RuntimeUUID:
            def __str__(self):return B[4:]
        torch=NS(version=NS(hip=None),cuda=NS(is_available=lambda:True,device_count=lambda:1,
            get_device_properties=lambda i:NS(uuid=RuntimeUUID(),name='Card B',major=12,minor=0)),xpu=NS(is_available=lambda:False),backends=NS(mps=NS(is_available=lambda:False)))
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,{'torch':torch}),patch('sys.prefix',directory),patch.dict(os.environ,{},clear=True):
            rows=devices.runtime_records('pytorch')
            self.assertEqual(rows[1]['id'],B)
            self.assertEqual(devices.cuda_uuid(torch,0),B)
            snapshot={'devices':[{**PHYSICAL[1],'id':B,'present':True,'discovery_status':'current','visibility':{'status':'visible','constraints':{}},
                'runtimes':[{**rows[1],'available':True}]}]}
            with patch('titanos.devices.inventory',return_value=snapshot):
                catalog=devices.assignment_catalog(NS(backend='transformers',task='chat'))
                self.assertEqual(catalog['data'][0]['id'],B);self.assertTrue(catalog['data'][0]['assignable'])
            self.assertEqual(select_device('auto'),'cuda:0')
            Path(directory,'titanos-runtime.json').write_text(json.dumps({'backend':'pytorch','device':'cpu'}))
            self.assertEqual(select_device('auto'),'cpu')
            self.assertEqual(select_device('cpu'),'cpu')
            self.assertEqual(select_device(B),'cuda:0')
            with self.assertRaises(ValueError):select_device('cuda:1')
            with self.assertRaises(ValueError):select_device('xpu:0')
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
                with self.assertRaisesRegex(ValueError,'masked'):select_device(B)

    def test_uuid_normalization_preserves_bytes_and_rejects_weak_representations(self):
        self.assertEqual(devices.normalized_uuid(uuid.UUID(A[4:])),A[4:])
        self.assertEqual(devices.normalized_uuid(list(uuid.UUID(A[4:]).bytes)),A[4:])
        for value in [None,0,True,object(),bytes(16),'GPU-'+str(uuid.UUID(int=0)),'cuda:0','Card name',[1]*15]:
            with self.subTest(value=repr(value)):self.assertIsNone(devices.normalized_uuid(value))


if __name__=='__main__':unittest.main(verbosity=2)
