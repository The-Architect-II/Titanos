"""Real execution seams with inert loaders/probes; no GPU, model or child process."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from titanos import native, replicas, native_process
from titanos.backends import LoadedModel, require_cuda_group, verify_parameter_placement
from titanos.capabilities import execution_modes
from titanos.config import Config, ModelSpec, NativeOptions, WorkerSpec, native_assignment
from titanos.parallel import distributed_backend, validate_plan
from titanos.settings import Settings, _active_settings


A = 'GPU-11111111-1111-1111-1111-111111111111'
B = 'GPU-22222222-2222-2222-2222-222222222222'


def native_spec(device='auto', **options):
    return ModelSpec(path='inert', task='chat', backend='llama-cpp', device=device,
                     native={'model_file': 'model.gguf', **options})


def torch_double(capabilities=((12, 0), (12, 0)), *, nccl=True):
    return NS(version=NS(hip=None), float16='float16', float32='float32', bfloat16='bfloat16',
        cuda=NS(is_available=lambda: True, device_count=lambda: 2,
            get_device_capability=lambda i: capabilities[int(str(i).removeprefix('cuda:'))],
            get_device_properties=lambda i: NS(uuid=[A, B][i], name='Synthetic GPU'),
            mem_get_info=lambda i: (1000000, 2000000)),
        distributed=NS(is_available=lambda: True, is_backend_available=lambda name: name == 'gloo' or nccl,
                       is_initialized=lambda: False))


class Checks(unittest.TestCase):
    def setUp(self):
        token = _active_settings.set(Settings(threads=2))
        self.addCleanup(_active_settings.reset, token)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.physical = [{'uuid': A, 'index': 8, 'compute_capability': '12.0'},
                         {'uuid': B, 'index': 9, 'compute_capability': '12.0'}]

    def placement(self, spec, env):
        def observed(library_dir=None, environ=None):
            mask = environ.get('CUDA_VISIBLE_DEVICES')
            identities = [B, A] if mask == '1,0' else mask.split(',') if mask else [A, B] if mask is None else []
            return [{'uuid': identity, 'local_device': 'CUDA'+str(i), 'runtime_address': 'cuda:'+str(i)}
                    for i, identity in enumerate(identities)]
        with patch('titanos.devices.scan', return_value=self.physical), \
             patch('titanos.devices.probe_visible_cuda', side_effect=observed), \
             patch.object(native, 'native_devices', return_value={'CUDA0': 'Synthetic GPU', 'CUDA1': 'Synthetic GPU'}):
            return native.resolve_native_placement(spec, self.root/'llama-server.exe', env)

    def test_native_auto_and_explicit_cpu_zero_precedence(self):
        self.assertEqual(native_spec().device, 'auto')
        self.assertEqual(native_spec().native.gpu_layers, 'auto')
        self.assertEqual(native_spec('cpu').native.gpu_layers, 0)
        for value in [0, 3, 'auto', 'all']:
            self.assertEqual(NativeOptions(model_file='model.gguf',gpu_layers=value).gpu_layers, value)
        for value in [-1, True, 1.5, '3']:
            with self.assertRaises(ValueError): NativeOptions(model_file='model.gguf',gpu_layers=value)
        for raw, expected in [('0',0), ('4',4), ('auto','auto'), ('all','all')]:
            settings=Settings.from_env({'TITANOS_NATIVE_GPU_LAYERS':raw})
            self.assertEqual(settings.native_gpu_layers,expected)
        with patch.object(native,'native_devices', side_effect=AssertionError('CPU must not probe an accelerator')):
            for spec in [native_spec('cpu'),native_spec(gpu_layers=0)]:
                env={'CUDA_VISIBLE_DEVICES':B}
                result=native.resolve_native_placement(spec,self.root/'llama-server.exe',env)
                self.assertEqual((result['resolved_device'],result['resolved_gpu_layers']),('cpu',0))
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')

    def test_numeric_visibility_uses_runtime_uuid_order_then_narrows(self):
        env={'CUDA_VISIBLE_DEVICES':'1,0'}
        result=self.placement(native_spec(B),env)
        self.assertEqual(result['resolved_device'],B)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],B)
        self.assertEqual(result['native_devices'],['CUDA0'])
        self.assertEqual(result['inherited_visibility'],{'CUDA_VISIBLE_DEVICES':'1,0'})
        self.assertFalse(result['kernel_execution_verified'])

    def test_auto_stays_inside_masks_and_explicit_denial_has_no_cpu_fallback(self):
        env={'CUDA_VISIBLE_DEVICES':B}
        self.assertEqual(self.placement(native_spec(),env)['resolved_device'],B)
        with self.assertRaises(ValueError):self.placement(native_spec(A),{'CUDA_VISIBLE_DEVICES':B})
        self.assertEqual(self.placement(native_spec(),{'CUDA_VISIBLE_DEVICES':''})['resolved_device'],'cpu')
        with self.assertRaises(ValueError):self.placement(native_spec(gpu_layers='all'),{'CUDA_VISIBLE_DEVICES':''})

    def test_cpu_only_native_binary_cannot_hide_an_allowed_nvidia_gpu(self):
        with patch.object(native,'native_devices',return_value={}), \
             patch.object(native.shutil,'which',return_value='inert-nvidia-smi'), \
             patch('titanos.devices.scan',return_value=self.physical):
            for env in [{}, {'CUDA_VISIBLE_DEVICES':B}]:
                before=dict(env)
                with self.assertRaisesRegex(ValueError,'compatible CUDA runtime'):
                    native.resolve_native_placement(native_spec(),self.root/'cpu-only.exe',env)
                self.assertEqual(env,before)
            with patch('titanos.devices.scan',return_value=[]):
                result=native.resolve_native_placement(native_spec(),self.root/'cpu-only.exe',{})
                self.assertEqual(result['resolved_device'],'cpu')
        with patch.object(native,'native_devices',side_effect=AssertionError('Explicit CPU scope must not query')):
            for spec,env in [(native_spec('cpu'),{}),(native_spec(gpu_layers=0),{}),
                             (native_spec(),{'CUDA_VISIBLE_DEVICES':''}),
                             (native_spec(),{'NVIDIA_VISIBLE_DEVICES':'none'})]:
                result=native.resolve_native_placement(spec,self.root/'cpu-only.exe',env)
                self.assertEqual((result['resolved_device'],result['resolved_gpu_layers']),('cpu',0))
        with patch.object(native,'native_devices',return_value={}),patch.object(native.shutil,'which',return_value=None):
            self.assertEqual(native.resolve_native_placement(native_spec(),self.root/'cpu-only.exe',{})['resolved_device'],'cpu')

    def test_native_group_requires_known_homogeneous_architecture(self):
        spec=native_spec(A+','+B,split_mode='layer')
        self.assertEqual(self.placement(spec,{})['native_devices'],['CUDA0','CUDA1'])
        self.physical[1]['compute_capability']='8.0'
        with self.assertRaisesRegex(ValueError,'homogeneous|identical'):self.placement(spec,{})
        self.physical[1]['compute_capability']=None
        with self.assertRaises(ValueError):self.placement(spec,{})

    def test_native_worker_group_mapping_does_not_reinterpret_explicit_zero(self):
        model=native_spec()
        for mode, expected in [('model_sharding','layer'),('tensor_parallel','tensor')]:
            worker=WorkerSpec(models=['m'],gpus=[A,B],mode=mode)
            cfg=Config(models={'m':model},workers={'group':worker})
            effective=native_assignment(cfg.models['m'],worker)
            self.assertEqual((effective.parallelism,effective.native.split_mode),('single',expected))
            self.assertEqual(effective.device,A+','+B)
        with self.assertRaises(ValueError):
            Config(models={'m':native_spec('cpu')},workers={'gpu':WorkerSpec(models=['m'],gpus=[A])})
        with self.assertRaises(ValueError):
            Config(models={'m':model},workers={'cpu-tp':WorkerSpec(models=['m'],mode='tensor_parallel',device='cpu')})

    def test_effective_worker_cannot_bypass_cpu_only_or_task_mode_gates(self):
        for model, worker in [
            (ModelSpec(path='inert',backend='sherpa-onnx',task='speech'), WorkerSpec(models=['m'],device='cuda:0')),
            (ModelSpec(path='inert',task='pair-classification'),WorkerSpec(models=['m'],gpus=[A,B],mode='model_sharding'))]:
            with self.assertRaises(ValueError):Config(models={'m':model},workers={'w':worker})

    def test_replica_resolver_is_mask_aware_without_parent_torch_import(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':B},clear=True), \
             patch('titanos.devices.scan',return_value=self.physical), \
             patch.dict(sys.modules,{'torch':None}):
            self.assertEqual(replicas.resolve_gpus([B]),[B])
            with self.assertRaises(ValueError):replicas.resolve_gpus([A])
            with self.assertRaises(ValueError):replicas.resolve_gpus(['0'])

    def test_framework_collectives_and_checkpoint_plan_are_not_native_tensor_mode(self):
        torch=torch_double(nccl=False)
        with self.assertRaisesRegex(ValueError,'nccl_unavailable'):distributed_backend(torch,'cuda:0',2)
        self.assertEqual(distributed_backend(torch,'cpu',2),'gloo')
        self.assertEqual(distributed_backend(torch_double(),'cuda:0',2),'nccl')
        with self.assertRaises(ValueError):require_cuda_group(torch_double(((12,0),(8,0))))
        plan={'layers.*.q_proj':'colwise','layers.*.k_proj':'colwise'}
        with self.assertRaises(ValueError):validate_plan(NS(base_model_tp_plan=plan,num_attention_heads=9,num_key_value_heads=3),2)
        native_modes=execution_modes(native_spec())
        self.assertEqual(native_modes['tensor_parallel']['execution_backend'],'llama.cpp')
        self.assertTrue(native_modes['tensor_parallel']['experimental'])
        with patch.object(sys,'platform','win32'):
            modes=execution_modes(ModelSpec(path='inert',task='chat'))
        self.assertEqual(modes['tensor_parallel']['collectives']['cuda']['reason'],'nccl_unavailable')
        self.assertEqual(modes['single']['execution_backend'],'pytorch')

    def test_actual_loader_map_and_parameters_must_span_the_requested_group(self):
        class Model:
            hf_device_map={'a':0,'b':1}
            def __init__(self,devices):self.p=[NS(device=x) for x in devices]
            def eval(self):return self
            def to(self,*args):return self
            def parameters(self):return iter(self.p)
            def get_input_embeddings(self):return NS(weight=NS(device='cuda:0'))
        def load(devices):
            model=Model(devices)
            loader=NS(from_pretrained=Mock(return_value=model))
            fake=NS(**{name:loader for name in ['AutoTokenizer','AutoProcessor','AutoModel','AutoModelForCausalLM','AutoModelForImageTextToText']},pipeline=Mock())
            with patch.dict(sys.modules,{'torch':torch_double(),'transformers':fake}), \
                 patch('titanos.hardware.select_device',return_value='cuda:0'):
                loaded=LoadedModel(ModelSpec(path=str(self.root),task='chat',parallelism='model_sharding'))
            load_call = loader.from_pretrained.call_args_list[0].kwargs
            self.assertEqual(load_call['device_map'], 'balanced')
            self.assertEqual(set(load_call['max_memory']), {0,1})
            return loaded
        self.assertEqual(load(['cuda:0','cuda:1']).placement_devices,['cuda:0','cuda:1'])
        with self.assertRaisesRegex(ValueError,'parameter placement'):load(['cuda:0'])
        with self.assertRaisesRegex(ValueError,'parameter placement'):load(['cuda:0','cpu'])

    def test_single_gpu_loader_cannot_report_cpu_weights_as_gpu(self):
        model=NS(parameters=lambda:iter([NS(device='cpu')]))
        with self.assertRaisesRegex(ValueError,'no fallback'):verify_parameter_placement(model,{'cuda:0'})
        self.assertEqual(verify_parameter_placement(model,{'cpu'}),['cpu'])

    def test_native_query_and_offload_log_are_bounded_and_distinct_from_execution(self):
        with patch.object(native.subprocess,'check_output',return_value='load_backend: loaded a backend\nAvailable devices:\n  CUDA0: Synthetic device (1024 MiB, 512 MiB free)\n') as call:
            self.assertEqual(native.native_devices(self.root/'llama-server.exe',{}),{'CUDA0':'Synthetic device'})
        self.assertIn('--query-devices',call.call_args.args[0])
        self.assertEqual(call.call_args.kwargs['timeout'],35)
        with patch.object(native.subprocess,'check_output',return_value='load_backend: CPU library\nAvailable devices:\n  (none)\n'):
            self.assertEqual(native.native_devices(self.root/'llama-server.exe',{}),{})
        record=native.native_offload('load_tensors: offloaded 31/31 layers to GPU\nload_tensors: CUDA0 model buffer size = 67.5 MiB\n')
        self.assertEqual(record['gpu_layers'],31)
        self.assertEqual(record['model_buffer_mib'],{'CUDA0':67.5})
        self.assertFalse(record['kernel_execution_verified'])

    def test_native_startup_forwards_resolved_gpu_options_and_refuses_zero_offload(self):
        (self.root/'model.gguf').write_bytes(b'GGUF synthetic header only')
        executable=self.root/'llama-server.exe';executable.write_bytes(b'inert executable bytes')
        spec=native_spec(B,gpu_layers='all');spec.path=str(self.root)
        placement={'resolved_device':B,'resolved_gpu_layers':'all','native_devices':['CUDA0'],
                   'verified_devices':[{'uuid':B,'local_device':'CUDA0'}]}
        for offloaded in [31,0,None]:
            worker=native.NativeWorker('inert','m',spec);captured=[]
            class Process:
                def __init__(self):self.code=None;self.stdin=NS(close=lambda:setattr(self,'code',0))
                def poll(self):return self.code
                def wait(self,*args,**kwargs):return 0
            def start(argv,**kwargs):
                captured.append((argv,kwargs))
                pid=Path(argv[argv.index('titanos.native_process')+1]);pid.write_text('4242')
                # b10809 common_log_default_callback maps library INFO to TRACE
                # (4), while the server defaults to INFO (3). Health alone is
                # ready even when the library's placement records are suppressed.
                verbosity=int(argv[argv.index('--log-verbosity')+1]) if '--log-verbosity' in argv else 3
                kwargs['stdout'].write(b'I srv llama_server: model loaded\n')
                if verbosity>=4 and offloaded is not None:
                    kwargs['stdout'].write(f'I load_tensors: offloaded {offloaded}/31 layers to GPU\n'.encode())
                kwargs['stdout'].flush()
                return Process()
            with patch.object(native,'executable_path',return_value=executable), \
                 patch.object(native,'resolve_native_placement',return_value=placement), \
                 patch.object(native.subprocess,'Popen',side_effect=start), \
                 patch.object(native.httpx,'Client',return_value=contextlib.nullcontext(NS(get=lambda url:NS(status_code=200)))):
                try:
                    if offloaded:worker.start_process()
                    else:
                        with self.assertRaisesRegex(ValueError,'did not report GPU'):worker.start_process()
                    argv=captured[0][0]
                    self.assertEqual(argv[argv.index('--n-gpu-layers')+1],'all')
                    self.assertEqual(argv[argv.index('--device')+1],'CUDA0')
                    self.assertEqual(argv[argv.index('--log-verbosity')+1],'4')
                finally:asyncio.run(worker.stop())

    def test_cli_add_pull_keep_auto_all_and_explicit_zero_without_download(self):
        from titanos.cli import main
        registry=self.root/'models.json';native_options=self.root/'native.json'
        for command, layers, device in [('add',None,'auto'),('add','all',B),('add',0,'auto'),('pull','auto','auto')]:
            name='fixture-'+str(len(list(self.root.glob('case-*.json'))))
            native_options.write_text(json.dumps({} if layers is None else {'gpu_layers':layers}))
            args=['--config',str(registry),command,name,str(self.root) if command=='add' else 'fixture/repo',
                  '--task','chat','--backend','llama-cpp','--device',device,'--model-file','model.gguf',
                  '--native-options',str(native_options)]
            with contextlib.redirect_stdout(io.StringIO()), \
                 patch('titanos.hub.pull',return_value=(str(self.root),'synthetic-revision')) as download, \
                 patch.dict(sys.modules,{'torch':None}):
                main(args)
            saved=json.loads(registry.read_text())['models'][name]
            self.assertEqual(saved['device'],device)
            self.assertEqual(saved['native']['gpu_layers'],'auto' if layers is None else layers)
            self.assertEqual(download.call_count,1 if command=='pull' else 0)
            (self.root/('case-'+name+'.json')).write_text('{}')
        token=_active_settings.set(None)
        try:
            with patch.dict(os.environ,{'TITANOS_NATIVE_GPU_LAYERS':'0'}),contextlib.redirect_stdout(io.StringIO()):
                main(['--config',str(registry),'add','env-zero',str(self.root),'--task','chat',
                      '--backend','llama-cpp','--model-file','model.gguf'])
        finally:_active_settings.reset(token)
        self.assertEqual(json.loads(registry.read_text())['models']['env-zero']['native']['gpu_layers'],0)

    def test_frozen_external_environment_clears_injection_not_visibility(self):
        reset=Mock(return_value=1)
        with patch.object(sys,'frozen',True,create=True), \
             patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':B,'PYTHONPATH':'inert injected path','_PYI_ARCHIVE_FILE':'inert'},clear=True), \
             patch('ctypes.WinDLL',return_value=NS(SetDllDirectoryW=reset),create=True):  # ctypes has no WinDLL off Windows
            env=native_process.native_environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],B)
        self.assertNotIn('PYTHONPATH',env);self.assertNotIn('_PYI_ARCHIVE_FILE',env)
        if os.name=='nt':reset.assert_called_once_with(None)


if __name__=='__main__':unittest.main(verbosity=2)
