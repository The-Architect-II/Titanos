"""Controlled music process contracts. No accelerator discovery or model execution."""
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
import wave

from titanos.config import Config,ModelSpec
from titanos.engine import Engine,EngineError
from titanos import music_backend as music

A='GPU-11111111-1111-1111-1111-111111111111'
B='GPU-22222222-2222-2222-2222-222222222222'
ROWS=[{'uuid':A,'index':1},{'uuid':B,'index':0}]
SETTINGS=NS(threads=3,max_audio_seconds=600,max_body_bytes=1024,
            max_music_output_bytes=128*1024**2,max_music_samples=60_000_000,
            max_steps=100,default_steps=25,max_prompt_chars=8000,max_text_chars=100000)


def contract(device='cpu',dtype='auto',env=None,placement=None):
    env={} if env is None else env
    return music._device_contract(NS(device=device,dtype=dtype,placement=placement),env,SETTINGS)


GIB=1024**3
REFUSAL='placement_refused_before_load: resident placement: device free 1024 < required 5 = 1 resident bytes'


def placement(request,device,resolved=None):
    if resolved is None:resolved='host' if device=='cpu' else ('resident' if request['placement']=='resident' else 'offload')
    return {'requested':request['placement'],'resolved':resolved,'device_free_bytes':None if device=='cpu' else 80*GIB,
            'required_bytes':34*GIB,'total_resident_bytes':30*GIB,'peak_set_bytes':20*GIB,'headroom_bytes':4*GIB,
            'accelerate_version':'fixture' if resolved=='resident' else None,'basis':'controlled fixture; not a measured device'}


def response(request,device='cpu',uuid=None,dtype='float32',resolved=None,cpu_offload=None):
    block=placement(request,device,resolved)
    if cpu_offload is None:cpu_offload=block['resolved']=='offload'
    return {'effective':{'dtype':dtype,'threads':request['threads'],'interop_threads':1,'cpu_offload':cpu_offload},
            'evidence':{'runtime':{'contract_version':1,'backend':'cuda' if device.startswith('cuda') else 'cpu',
              'resolved_device':device,'device_uuid':uuid,
              'visible_devices':[{'local_device':'cuda:0','uuid':uuid}] if uuid else [],
              'identity_source':'torch.cuda.get_device_properties' if uuid else 'parameter.device',
              'torch_version':'controlled-fixture','visibility':request['visibility'],
              'torch_num_threads':request['threads'],'torch_num_interop_threads':1,'bfloat16_supported':True,
              'placement':block,
              'parameter_placements':[{'component':name,'stage':'execute','device':device,'dtype':dtype}
                                       for name in ('dit','text_encoder','vae')]}}}


class Checks(unittest.TestCase):
    def setUp(self):
        # Any accidentally introduced real query fails every check in this file.
        self.scan=patch('titanos.devices.scan',side_effect=AssertionError('Real GPU discovery is forbidden'))
        self.scan.start();self.addCleanup(self.scan.stop)

    def test_cpu_masks_and_dtype(self):
        env={'CUDA_VISIBLE_DEVICES':B,'HIP_VISIBLE_DEVICES':'2','ROCR_VISIBLE_DEVICES':'1'}
        value=contract('cpu',env=env)
        self.assertEqual(value['device'],'cpu')
        self.assertIsNone(value['expected_uuid'])
        self.assertTrue(all(env[name]=='' for name in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')))
        with self.assertRaisesRegex(ValueError,'float32'):contract('cpu','bfloat16')
        for device in ('HIP-'+A[4:],'xpu:0','mps','GPU-not-a-uuid'):
            with self.subTest(device=device),self.assertRaises(ValueError):contract(device)

    def test_uuid_and_inherited_ordinal_remapping(self):
        with patch('titanos.devices.scan',return_value=ROWS):
            env={'CUDA_VISIBLE_DEVICES':B+','+A}
            value=contract('cuda:0',env=env)
            self.assertEqual((value['device'],value['expected_uuid'],env['CUDA_VISIBLE_DEVICES']),('cuda:0',B,B))
            value=contract('cuda:1',env={'CUDA_VISIBLE_DEVICES':B+','+A})
            self.assertEqual(value['expected_uuid'],A)
            self.assertEqual(contract(B,env={'CUDA_VISIBLE_DEVICES':B})['expected_uuid'],B)
            with self.assertRaises(ValueError):contract(A,env={'CUDA_VISIBLE_DEVICES':B})
            with self.assertRaises(ValueError):contract('cuda:2',env={'CUDA_VISIBLE_DEVICES':B+','+A})

    def test_numeric_masks_are_cuda_ordinals_not_nvidia_smi_indices(self):
        env={'CUDA_VISIBLE_DEVICES':'1,0','CUDA_DEVICE_ORDER':'FASTEST_FIRST'}
        value=contract('cuda:0',env=env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
        self.assertEqual(env['CUDA_DEVICE_ORDER'],'FASTEST_FIRST')
        self.assertIsNone(value['expected_uuid'])
        self.assertEqual(music._verify_runtime(response(value,'cuda:0',B),value)['device_uuid'],B)
        self.assertEqual(contract('cuda:2')['visibility']['CUDA_VISIBLE_DEVICES'],'2')
        with self.assertRaises(ValueError):contract(B,env={'CUDA_VISIBLE_DEVICES':'1'})
        for mask in ('1,1',A+',0',A+','+A,'0,','MIG-unsupported'):
            with self.subTest(mask=mask),self.assertRaises(ValueError):contract('cuda:0',env={'CUDA_VISIBLE_DEVICES':mask})

    def test_auto_scope_and_missing_identity_failures(self):
        with patch('titanos.devices.scan',return_value=ROWS):
            self.assertEqual(contract('auto',env={'CUDA_VISIBLE_DEVICES':B})['expected_uuid'],B)
        self.assertEqual(contract('auto',env={'CUDA_VISIBLE_DEVICES':''})['device'],'cpu')
        self.assertEqual(contract('auto')['device'],'auto')
        env={'HIP_VISIBLE_DEVICES':'1','ROCR_VISIBLE_DEVICES':'1'}
        value=contract('auto',env=env)
        self.assertEqual(value['visibility'],env)  # No widening by dropping inherited masks.
        with patch('titanos.devices.scan',return_value=[ROWS[0]]):
            with self.assertRaisesRegex(ValueError,'missing or ambiguous'):contract(B)
        with patch('titanos.devices.scan',return_value=[ROWS[1],ROWS[1]]):
            with self.assertRaisesRegex(ValueError,'missing or ambiguous'):contract(B)

    def test_external_environment_keeps_driver_paths_not_credentials_or_frozen_python(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);frozen=root/'frozen';parent=root/'parent-python';base=root/'base-python'
            driver=root/'driver';runtime=root/'runtime';request=root/'request'
            values={'PATH':os.pathsep.join(map(str,(frozen,parent,driver))),
                    'LD_LIBRARY_PATH':str(frozen),'LD_LIBRARY_PATH_ORIG':str(driver),
                    'CUDA_VISIBLE_DEVICES':B,'HIP_VISIBLE_DEVICES':'2','CUDA_DEVICE_ORDER':'PCI_BUS_ID',
                    'CUDA_PATH':str(driver),'PYTHONPATH':str(frozen),'PYTHONHOME':str(parent),
                    'VIRTUAL_ENV':str(parent),'LD_PRELOAD':'foreign.so','HF_TOKEN':'secret',
                    'TITANOS_API_KEY':'secret','OPENAI_API_KEY':'secret','AWS_SECRET_ACCESS_KEY':'secret',
                    '_PYI_APPLICATION_HOME_DIR':str(frozen),'_MEIPASS2':str(frozen)}
            with patch.dict(os.environ,values,clear=True),patch.object(music.sys,'prefix',str(parent)),patch.object(music.sys,'base_prefix',str(base)),patch.object(music.sys,'_MEIPASS',str(frozen),create=True):
                before=dict(os.environ);env=music._child_environment(runtime/'python',request)
                self.assertEqual(dict(os.environ),before)
            self.assertEqual(env['PATH'],os.pathsep.join((str(runtime),str(driver))))
            self.assertEqual(env['LD_LIBRARY_PATH'],str(driver))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],B)
            self.assertEqual(env['HIP_VISIBLE_DEVICES'],'2')
            self.assertEqual(env['HOME'],str(request))
            self.assertEqual(env['TORCHINDUCTOR_CACHE_DIR'],str(request/'inductor-cache'))
            self.assertEqual(env['TRITON_CACHE_DIR'],str(request/'triton-cache'))
            self.assertNotIn('USERNAME',env)  # absent in the parent -> absent in the child (the pinned cache directories make it unnecessary)
            with patch.dict(os.environ,{**values,'USERNAME':'operator'},clear=True),patch.object(music.sys,'prefix',str(parent)),patch.object(music.sys,'base_prefix',str(base)):
                self.assertEqual(music._child_environment(runtime/'python',request)['USERNAME'],'operator')  # identity string passes through, credentials do not
            for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','LD_PRELOAD','HF_TOKEN','TITANOS_API_KEY','OPENAI_API_KEY','AWS_SECRET_ACCESS_KEY','_PYI_APPLICATION_HOME_DIR','_MEIPASS2','LD_LIBRARY_PATH_ORIG'):
                self.assertNotIn(key,env)
            command=music._python_command(runtime/'python',runtime/'adapter.py')
            self.assertEqual(command[1:6],['-I','-B','-X','utf8','-c'])
            self.assertIn('SetDllDirectoryW(None)',command[6])
            self.assertLess(command[6].index('SetDllDirectoryW(None)'),command[6].index('runpy.run_path'))

    def test_missing_or_mismatched_observed_runtime_is_not_placement_proof(self):
        with patch('titanos.devices.scan',return_value=ROWS):request=contract(B,'bfloat16')
        good=response(request,'cuda:0',B,'bfloat16')
        self.assertEqual(music._verify_runtime(good,request)['resolved_device'],'cuda:0')
        changes=[lambda r:r.pop('evidence'),
                 lambda r:r['evidence'].update(runtime={'requested_device':B}),
                 lambda r:r['evidence']['runtime'].update(device_uuid=A),
                 lambda r:r['evidence']['runtime'].update(visible_devices=[]),
                 lambda r:r['evidence']['runtime']['visible_devices'].append({'local_device':'cuda:1','uuid':A}),
                 lambda r:r['evidence']['runtime'].update(resolved_device='cpu'),
                 lambda r:r['evidence']['runtime'].update(visibility={}),
                 lambda r:r['evidence']['runtime'].update(torch_num_threads=99),
                 lambda r:r['evidence']['runtime'].update(bfloat16_supported=False),
                 lambda r:r['effective'].update(dtype='float32'),
                 lambda r:r['effective'].update(dtype=[]),
                 lambda r:r['evidence']['runtime']['parameter_placements'].pop(),
                 lambda r:r['evidence']['runtime']['parameter_placements'][0].update(device='cuda:1')]
        for change in changes:
            value=copy.deepcopy(good);change(value)
            with self.subTest(change=change),self.assertRaises(EngineError):music._verify_runtime(value,request)
        cpu_request=contract('cpu');missing=response(cpu_request)
        missing['evidence']['runtime'].pop('device_uuid')
        with self.assertRaises(EngineError):music._verify_runtime(missing,cpu_request)

    def test_placement_contract_and_evidence(self):
        self.assertEqual(contract('cpu')['placement'],'auto')
        self.assertEqual(contract('cuda:0',placement='resident')['placement'],'resident')
        with self.assertRaisesRegex(ValueError,'placement'):ModelSpec(path='.',task='chat',backend='llama-cpp',placement='resident',native={'model_file':'m.gguf'})
        with self.assertRaises(ValueError):ModelSpec(path='.',task='music',backend='ace-step',placement='pinned')
        spec=ModelSpec(path='.',task='music',backend='ace-step',device='cpu',placement='resident')
        self.assertEqual(music._device_contract(spec,{},SETTINGS)['placement'],'resident')
        # CPU: host is the only truthful resolution; the requested value is still echoed and checked.
        cpu=contract('cpu');self.assertEqual(music._verify_runtime(response(cpu),cpu)['placement']['resolved'],'host')
        for resolved in ('offload','resident'):
            with self.assertRaises(EngineError):music._verify_runtime(response(cpu,resolved=resolved),cpu)
        # An explicit resident contract on cpu: the child refuses before loading, so a 'host' receipt is a downgrade the parent rejects too.
        cpu_resident=contract('cpu',placement='resident')
        with self.assertRaises(EngineError):music._verify_runtime(response(cpu_resident,resolved='host'),cpu_resident)
        cuda=contract('cuda:0');env={}
        good=response(cuda,'cuda:0',B)
        self.assertEqual(music._verify_runtime(good,cuda)['placement']['resolved'],'offload')
        self.assertEqual(music._verify_runtime(response(cuda,'cuda:0',B,resolved='resident'),cuda)['placement']['resolved'],'resident')
        changes=[lambda r:r['evidence']['runtime'].pop('placement'),
                 lambda r:r['evidence']['runtime']['placement'].update(requested='resident'),
                 lambda r:r['evidence']['runtime']['placement'].update(resolved='host'),
                 lambda r:r['evidence']['runtime']['placement'].update(resolved='pinned'),
                 lambda r:r['evidence']['runtime']['placement'].update(device_free_bytes=None),
                 lambda r:r['evidence']['runtime']['placement'].update(device_free_bytes=-1),
                 lambda r:r['evidence']['runtime']['placement'].update(required_bytes='34'),
                 lambda r:r['evidence']['runtime']['placement'].update(required_bytes=29*GIB),  # below total_resident_bytes
                 lambda r:r['evidence']['runtime']['placement'].update(peak_set_bytes=31*GIB),  # above total_resident_bytes
                 lambda r:r['evidence']['runtime']['placement'].update(total_resident_bytes=-1),
                 lambda r:r['evidence']['runtime']['placement'].update(headroom_bytes=4.0),
                 lambda r:r['evidence']['runtime']['placement'].pop('peak_set_bytes'),
                 lambda r:r['evidence']['runtime']['placement'].update(accelerate_version=''),
                 lambda r:r['evidence']['runtime']['placement'].update(accelerate_version=1),
                 lambda r:r['evidence']['runtime']['placement'].update(basis=' '),
                 lambda r:r['evidence']['runtime']['placement'].update(basis='b'*401),
                 lambda r:r['evidence']['runtime']['placement'].update(extra=True),
                 lambda r:r['effective'].pop('cpu_offload'),
                 lambda r:r['effective'].update(cpu_offload=False),  # resolved offload must say so
                 lambda r:r['effective'].update(cpu_offload=1),
                 lambda r:r['evidence']['runtime']['placement'].update(accelerate_version='1.15.0'),  # auto -> offload with accelerate present and free >= required: no evidenced obstacle
                 lambda r:r['evidence']['runtime']['parameter_placements'][0].update(stage='execution-fixture'),
                 lambda r:r['evidence']['runtime']['parameter_placements'].append({'component':'dit','stage':'retain','device':'cuda:0','dtype':'float32'}),
                 lambda r:r['evidence']['runtime']['parameter_placements'].append({'component':'dit','stage':'offload','device':'cuda:0','dtype':'float32'})]
        for change in changes:
            value=copy.deepcopy(good);change(value)
            with self.subTest(change=change),self.assertRaises(EngineError):music._verify_runtime(value,cuda)
        # The block's arithmetic must agree with its resolution: resident with less free memory than the bill is rejected;
        # auto -> offload is accepted only with an evidenced obstacle (accelerate absent, or free below the bill).
        lying=response(cuda,'cuda:0',B,resolved='resident');lying['evidence']['runtime']['placement'].update(device_free_bytes=1)
        with self.assertRaises(EngineError):music._verify_runtime(lying,cuda)
        starved=response(cuda,'cuda:0',B);starved['evidence']['runtime']['placement'].update(device_free_bytes=1,accelerate_version='1.15.0')
        self.assertEqual(music._verify_runtime(starved,cuda)['placement']['resolved'],'offload')
        offload=contract('cuda:0',placement='offload');explicit=response(offload,'cuda:0',B);explicit['evidence']['runtime']['placement'].update(accelerate_version='1.15.0')
        self.assertEqual(music._verify_runtime(explicit,offload)['placement']['resolved'],'offload')  # an explicit offload needs no obstacle
        resident=contract('cuda:0',placement='resident');lying2=response(resident,'cuda:0',B);lying2['evidence']['runtime']['placement'].update(device_free_bytes=34*GIB-1)
        with self.assertRaises(EngineError):music._verify_runtime(lying2,resident)
        offloaded=copy.deepcopy(good);offloaded['evidence']['runtime']['parameter_placements'].append({'component':'dit','stage':'offload','device':'cpu','dtype':'float32'})
        music._verify_runtime(offloaded,cuda)
        # Explicit requests are not satisfied by the other resolution; execute rows alone prove execution.
        resident=contract('cuda:0',placement='resident')
        self.assertEqual(music._verify_runtime(response(resident,'cuda:0',B),resident)['placement']['resolved'],'resident')
        with self.assertRaises(EngineError):music._verify_runtime(response(resident,'cuda:0',B,resolved='offload'),resident)
        # Resident with a CPU-offload receipt, or without the accelerate that device_map loading needs, is not resident.
        with self.assertRaises(EngineError):music._verify_runtime(response(resident,'cuda:0',B,cpu_offload=True),resident)
        unplaced=response(resident,'cuda:0',B);unplaced['evidence']['runtime']['placement']['accelerate_version']=None
        with self.assertRaises(EngineError):music._verify_runtime(unplaced,resident)
        offload=contract('cuda:0',placement='offload')
        with self.assertRaises(EngineError):music._verify_runtime(response(offload,'cuda:0',B,resolved='resident'),offload)
        retained=response(resident,'cuda:0',B)
        for row in retained['evidence']['runtime']['parameter_placements']:row['stage']='retain'
        with self.assertRaises(EngineError):music._verify_runtime(retained,resident)
        # CPU evidence must report no device reading at all, not a number.
        measured=response(cpu);measured['evidence']['runtime']['placement']['device_free_bytes']=8*GIB
        with self.assertRaises(EngineError):music._verify_runtime(measured,cpu)

    def test_child_placement_refusal_reaches_the_caller_verbatim_and_other_failures_stay_generic(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);adapter=root/'adapter.py';executable=root/'python'
            bundle={'root':root,'files':{},'manifest':{'variant':'xl-sft','model_revision':music.ACE_MODELS['xl-sft'],
                    'capabilities':{'instrumental':True,'duration_min':10,'duration_max':600,'reference_audio':False}}}
            receipt={'code_revision':music.ACE_CODE_REVISION,'destination':str(root),'files':{}}
            spec=ModelSpec(path=str(root),task='music',backend='ace-step',device='cuda:0',placement='resident')
            outcomes=[]
            def failing(message,written=True):
                def runner(argv,directory,env,check,max_output_bytes,**unused):
                    if written:
                        Path(argv[argv.index('--result')+1]).write_text(json.dumps({'effective':{},'native_format':'pcm16-wav',
                            'evidence':{'status':'failed','runtime':{'placement_refused':{'device_free_bytes':1024}},
                                        'error':{'type':'ValueError','message':message}}}),encoding='utf-8')
                    raise EngineError('Music adapter process exited unsuccessfully',502,'native_process_failed')
                return runner
            with patch.object(music,'inspect_bundle',return_value=bundle),patch.object(music,'runtime_receipt',return_value=(receipt,executable,adapter)),patch.object(music,'current_settings',return_value=SETTINGS):
                model=music.AceStepModel(spec)
                self.assertEqual((model.details()['placement'],model.details()['requested_placement']),(None,'resident'))
                cases=[(REFUSAL,True,REFUSAL,'placement_refused',503),
                       ('arbitrary child text that must not reach the caller',True,'Music adapter process exited unsuccessfully','native_process_failed',502),
                       ('placement_refused_before_load: line one\nline two',True,'Music adapter process exited unsuccessfully','native_process_failed',502),
                       ('placement_refused_before_load: '+'x'*400,True,'Music adapter process exited unsuccessfully','native_process_failed',502),
                       (REFUSAL,False,'Music adapter process exited unsuccessfully','native_process_failed',502)]
                for message,written,expected,code,status in cases:
                    with self.subTest(message=message[:40],written=written),patch.object(music,'run_owned',side_effect=failing(message,written)):
                        with self.assertRaises(EngineError) as raised:model.infer('music',{'prompt':'controlled fixture'})
                        self.assertEqual((str(raised.exception),raised.exception.code,raised.exception.status),(expected,code,status))
                        outcomes.append(raised.exception)
                self.assertIsNone(model.details()['placement'])  # A refused run is never placement evidence.
                self.assertEqual(outcomes[0].__cause__.code,'native_process_failed')

    def test_actual_parent_handoff_and_details_require_child_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);adapter=root/'adapter.py';executable=root/'python'
            bundle={'root':root,'files':{},'manifest':{'variant':'xl-sft','model_revision':music.ACE_MODELS['xl-sft'],
                    'capabilities':{'instrumental':True,'duration_min':10,'duration_max':600,'reference_audio':False}}}
            receipt={'code_revision':music.ACE_CODE_REVISION,'destination':str(root),'files':{}}
            spec=ModelSpec(path=str(root),task='music',backend='ace-step',device='cpu',defaults={'duration':20})
            captured={}
            def runner(argv,directory,env,check,max_output_bytes,**unused):
                check();request=json.loads(Path(argv[argv.index('--request')+1]).read_text())
                captured.update(request=request,env=env,argv=argv,budget=max_output_bytes)
                with wave.open(argv[argv.index('--output')+1],'wb') as output:
                    output.setnchannels(2);output.setsampwidth(2);output.setframerate(24000);output.writeframes(b'\x01\x00'*4800)
                Path(argv[argv.index('--result')+1]).write_text(json.dumps(response(request['runtime_contract'])))
            with patch.object(music,'inspect_bundle',return_value=bundle),patch.object(music,'runtime_receipt',return_value=(receipt,executable,adapter)),patch.object(music,'current_settings',return_value=SETTINGS),patch.object(music,'run_owned',side_effect=runner) as child:
                model=music.AceStepModel(spec)
                self.assertIsNone(model.details()['device'])
                self.assertEqual(model.details()['requested_device'],'cpu')
                result=model.infer('music',{'prompt':'controlled fixture'})
                self.assertEqual(model.details()['device'],'cpu')
                self.assertEqual(captured['request']['duration'],20)
                self.assertEqual(captured['request']['runtime_contract']['max_samples'],SETTINGS.max_music_samples)
                for key in ('max_steps','max_prompt_chars','max_text_chars'):
                    self.assertEqual(captured['request']['runtime_contract'][key],getattr(SETTINGS,key))
                self.assertEqual(captured['budget'],SETTINGS.max_music_output_bytes)
                self.assertGreater(len(result['wav']),SETTINGS.max_body_bytes)  # HTTP input cap is not music output cap.
                calls=child.call_count
                with self.assertRaisesRegex(ValueError,'reference audio'):
                    model.infer('music',{'prompt':'fixture','reference_audio_base64':'AA=='})
                self.assertEqual(child.call_count,calls)
                bounded=NS(**{**vars(SETTINGS),'max_music_samples':100})
                with patch.object(music,'current_settings',return_value=bounded),self.assertRaises(EngineError):
                    model.infer('music',{'prompt':'sample budget fixture'})

    def test_direct_engine_checks_configured_limits_before_child_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            spec=ModelSpec(path=str(root),task='music',backend='ace-step',device='cpu')
            # Inject an inert resident adapter so this exercises the Engine entry
            # point without constructing models, an interpreter, or a runtime.
            model=music.AceStepModel.__new__(music.AceStepModel)
            model.spec=spec;model.executable=root/'python';model.adapter=root/'adapter.py'
            model.bundle={'root':root,'manifest':{'variant':'xl-sft','model_revision':music.ACE_MODELS['xl-sft'],
                'capabilities':{'instrumental':True,'song':True,'lyrics':True,'duration_min':10,'duration_max':600}}}
            engine=Engine(Config(models={'music':spec}))
            engine.current='music';engine.runtime=model;engine.states['music']='loaded'
            cases=[
                ({},{'steps':1001},False),
                ({'max_steps':1200},{'steps':1001},True),
                ({'max_steps':1000},{'steps':1001},False),
                ({},{'steps':True},False),
                ({'max_prompt_chars':40000},{'prompt':'p'*32769},True),
                ({},{'prompt':'p'*32769},False),
                ({},{'lyrics':'l'*65537},True),
                ({},{'lyrics':'l'*100001},False),
                ({'max_text_chars':110000},{'lyrics':'l'*100001},True),
                ({'max_prompt_chars':40000,'max_text_chars':4},{'lyrics':'words'},False),
                ({'max_text_chars':110000,'max_prompt_chars':4},{'prompt':'words'},False),
                ({'max_prompt_chars':4,'max_text_chars':4},{'prompt':'four','lyrics':'four'},True),
            ]
            for changes,payload,accepted in cases:
                settings=NS(**{**vars(SETTINGS),**changes})
                data={'prompt':'short','mode':'song',**payload}
                def reached(argv,*unused_args,**unused):
                    sent=json.loads(Path(argv[argv.index('--request')+1]).read_text())
                    for key in ('max_steps','max_prompt_chars','max_text_chars'):
                        self.assertEqual(sent['runtime_contract'][key],getattr(settings,key))
                    raise EngineError('controlled child boundary reached')
                with self.subTest(changes=changes,keys=list(payload)),patch.object(music,'current_settings',return_value=settings),patch.object(music,'run_owned',side_effect=reached) as child:
                    with self.assertRaises(EngineError) as raised:
                        engine.infer('music','music',data,require_loaded=True)
                    self.assertEqual(child.call_count,int(accepted))
                    self.assertEqual('controlled child boundary reached' in str(raised.exception),accepted,str(raised.exception))

    def test_portable_receipt_discovery_hash_membership_and_legacy_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'home';state.mkdir();bundle=root/'runtimes/ace-step-1.5';bundle.mkdir(parents=True)
            files={}
            for name,raw in (('python/python.exe',b'interpreter-fixture'),('ace_adapter.py',b'# adapter fixture'),('LICENSE',b'license-fixture')):
                path=bundle/name;path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
                files[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            manifest={'schema_version':1,'family':'ace-step-1.5','code_revision':music.ACE_CODE_REVISION,
                      'executable':'python/python.exe','adapter':'ace_adapter.py','files':files}
            location=bundle/'runtime.json';location.write_text(json.dumps(manifest))
            with patch.object(music,'home',return_value=state),patch.object(music.sys,'executable',str(root/'titanos.exe')):
                receipt,python,adapter=music.runtime_receipt()
                self.assertTrue(receipt['portable']);self.assertEqual(python,(bundle/'python/python.exe').resolve())
                self.assertEqual(receipt['files']['ace_adapter.py'],files['ace_adapter.py']['sha256'])
                extra=bundle/'unexpected.py';extra.write_text('unexpected')
                with self.assertRaisesRegex(ValueError,'membership'):music.runtime_receipt()
                extra.unlink()
                location.write_text(json.dumps({**manifest,'executable':'../foreign.exe'}))
                with self.assertRaises(ValueError):music.runtime_receipt()
                location.write_text(json.dumps(manifest))
                legacy={**manifest,'destination':str(bundle),'executable':str(python),'adapter':str(adapter),
                        'files':{name:pin['sha256'] for name,pin in files.items()}}
                explicit=state/'music-runtime.json';explicit.write_text(json.dumps(legacy))
                self.assertFalse(music.runtime_receipt()[0]['portable'])
                explicit.write_text(json.dumps({**legacy,'code_revision':'bad'}))
                with self.assertRaisesRegex(ValueError,'incompatible'):music.runtime_receipt()
                explicit.unlink()
                # a portable tree whose deepest loader-relevant file (.exe/.dll/.pyd) would exceed the Windows path bound is refused with the
                # directory budget before any child launch; non-loader files (sources, headers) are not gated
                deepest=len(str((bundle/'python/python.exe').resolve()))   # runtime_receipt resolves the executable's directory (8.3 names expand)
                with patch.object(music,'MAX_RUNTIME_PATH',deepest-1):
                    with self.assertRaisesRegex(ValueError,'too long .* at most'):music.runtime_receipt()
                with patch.object(music,'MAX_RUNTIME_PATH',deepest):
                    self.assertTrue(music.runtime_receipt()[0]['portable'])

    @unittest.skipUnless(os.name=='nt' or sys.platform.startswith('linux'), 'Music runtime publication currently supports Windows/Linux')
    def test_actual_packager_output_is_consumed_without_execution(self):
        fixture_path=Path(__file__).with_name('check_music_package.py')
        spec=importlib.util.spec_from_file_location('music_package_contract_fixture',fixture_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        fixture=module.MusicPackageTests('test_complete_relative_bundle_moves_without_old_prefix')
        fixture.setUp()
        try:
            manifest=fixture.build()
            packaged=module.package.verify_runtime(fixture.output)
            with patch.object(music,'home',return_value=fixture.root/'empty-state'),patch.object(music.sys,'executable',str(fixture.output.parents[1]/'titanos.exe')):
                receipt,executable,adapter=music.runtime_receipt()
            self.assertTrue(receipt['portable'])
            self.assertEqual(receipt['receipt_sha256'],hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(receipt['files'],{name:row['sha256'] for name,row in packaged['files'].items()})
            self.assertIn(b'never an executed interpreter',executable.read_bytes())
            self.assertIn(b'never execute adapter',adapter.read_bytes())
        finally:fixture.tearDown()


if __name__=='__main__':unittest.main()
