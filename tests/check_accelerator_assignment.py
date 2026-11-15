"""Actual source HTTP/registry admission with controlled hardware and no models."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from titanos.config import Config,ModelSpec,read_config,write_config
from titanos.engine import Engine
from titanos.server import create_app
from titanos.providers import AcquisitionPlan

A='GPU-11111111-1111-1111-1111-111111111111'
B='GPU-22222222-2222-2222-2222-222222222222'
C='GPU-33333333-3333-3333-3333-333333333333'
O='OV-44444444-4444-4444-4444-444444444444'

def gpu(identity,name,architecture='12.0',masked=False):
    return {'id':identity,'uuid':identity,'name':name,'label':name,'present':True,'installed':True,
        'device_type':'gpu','identity_kind':'vendor_uuid','identity_strength':'physical_uuid','stable':True,
        'discovery_status':'current','architecture':architecture,'visibility':{'status':'masked' if masked else 'visible','constraints':{}},
        'runtimes':[{'backend':'pytorch','id':identity,'uuid':identity,'runtime_address':'cuda:0','available':not masked,
                    'identity_kind':'vendor_uuid','identity_strength':'vendor_runtime_uuid'}]}

class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.registry=self.root/'models.json'
        self.environment=patch.dict(os.environ,{'TITANOS_HOME':str(self.root/'home')});self.environment.start();self.addCleanup(self.environment.stop)
        self.rows=[gpu(A,'Card A'),gpu(B,'Card B'),gpu(C,'Card C','8.9'),
            {'id':O,'uuid':O,'name':'NPU','label':'NPU','present':True,'installed':True,'device_type':'npu',
             'identity_kind':'vendor_uuid','identity_strength':'vendor_runtime_uuid','stable':True,'discovery_status':'current',
             'visibility':{'status':'visible','constraints':{}},'runtimes':[{'backend':'openvino','id':O,'uuid':O,'runtime_address':'NPU',
                'available':True,'identity_kind':'vendor_uuid','identity_strength':'vendor_runtime_uuid'}]},
            {'id':'cpu','name':'CPU','label':'CPU','present':True,'installed':True,'device_type':'cpu','stable':True,
             'visibility':{'status':'visible','constraints':{}},'runtimes':[]}]
        self.inventory=patch('titanos.devices.inventory',side_effect=lambda *a,**k:{'devices':self.rows,'visibility':{'constraints':{}},'probe_errors':[]})
        self.inventory.start();self.addCleanup(self.inventory.stop)
        self.preference=patch('titanos.devices.installed_runtime_preference',return_value=None)
        self.preference.start();self.addCleanup(self.preference.stop)
        self.no_load=patch.object(Engine,'_load',side_effect=AssertionError('A discovery/assignment call loaded a model'))
        self.no_load.start();self.addCleanup(self.no_load.stop)

    def app(self,models):
        cfg=Config(models=models);write_config(self.registry,cfg)
        return create_app(cfg,registry_path=self.registry,output_dir=self.root/'out',runtime_catalog=False)

    def text(self,device='auto'):
        return ModelSpec(path=str(self.root/'model'),task='chat',device=device)

    def native(self,device='cpu'):
        return ModelSpec(path=str(self.root/'gguf'),task='chat',backend='llama-cpp',device=device,native={'model_file':'model.gguf'})

    def test_installed_masked_rows_stay_visible_and_reject_assignment(self):
        self.rows[0]=gpu(A,'Card A',masked=True)
        with TestClient(self.app({'m':self.text()}),base_url='http://127.0.0.1') as client:
            body=client.get('/api/assignments/devices?model=m&refresh=true').json()
            row=next(row for row in body['data'] if row['id']==A)
            self.assertTrue(row['installed']);self.assertFalse(row['assignable'])
            self.assertEqual(row['reasons'][0]['code'],'masked')
            rejected=client.post('/api/assignments',json={'model':'m','groups':[[A]]})
            self.assertEqual(rejected.status_code,409)

    def test_auto_recommendation_does_not_mutate_explicit_cpu(self):
        with TestClient(self.app({'cpu':self.text('cpu'),'auto':self.text()}),base_url='http://127.0.0.1') as client:
            before=self.registry.read_bytes()
            cpu=client.get('/api/assignments/cpu').json();automatic=client.get('/api/assignments/auto').json()
            self.assertEqual(cpu['policy'],'explicit_cpu');self.assertEqual(cpu['selection']['device'],'cpu')
            self.assertEqual(automatic['policy'],'automatic');self.assertEqual(automatic['selection']['device'],A)
            self.assertFalse(automatic['selection']['observed']);self.assertIsNone(automatic['observed'])
            self.assertEqual(before,self.registry.read_bytes())
            same=client.post('/api/assignments',json={'model':'cpu','groups':[]});self.assertEqual(same.status_code,200)
            self.assertEqual(read_config(self.registry).models['cpu'].device,'cpu')
            changed=client.post('/api/assignments',json={'model':'cpu','groups':[],'policy':'auto'})
            self.assertEqual(changed.status_code,200);self.assertEqual(read_config(self.registry).models['cpu'].device,'auto')

    def test_homogeneous_shards_and_heterogeneous_replicas(self):
        with TestClient(self.app({'m':self.text()}),base_url='http://127.0.0.1') as client:
            value=client.post('/api/assignments',json={'model':'m','mode':'model_sharding','groups':[[A,B]]})
            self.assertEqual(value.status_code,200,value.text)
            mixed=client.post('/api/assignments',json={'model':'m','mode':'model_sharding','groups':[[A,C]]})
            self.assertEqual(mixed.status_code,422)
            replicas=client.post('/api/assignments',json={'model':'m','groups':[[A],[C]]})
            self.assertEqual(replicas.status_code,200,replicas.text);self.assertEqual(replicas.json()['replicas'],2)

    def test_native_gpu_intent_maps_to_native_worker_and_cpu_is_deliberate(self):
        with TestClient(self.app({'native':self.native()}),base_url='http://127.0.0.1') as client:
            value=client.post('/api/assignments',json={'model':'native','mode':'model_sharding','groups':[[A,B]]})
            self.assertEqual(value.status_code,200,value.text)
            cfg=read_config(self.registry)
            self.assertEqual(cfg.models['native'].device,'auto');self.assertEqual(cfg.models['native'].native.gpu_layers,'auto')
            self.assertEqual(next(iter(cfg.workers.values())).gpus,[A,B])
            cpu=client.post('/api/assignments',json={'model':'native','groups':[],'policy':'cpu'})
            self.assertEqual(cpu.status_code,200,cpu.text)
            cfg=read_config(self.registry);self.assertEqual(cfg.models['native'].native.gpu_layers,0)
            self.assertEqual(cfg.models['native'].device,'cpu');self.assertFalse(cfg.workers)

    def test_npu_assignment_only_for_compatible_backend(self):
        npu=ModelSpec(path=str(self.root/'graph'),task='tensor',backend='openvino',device='cpu')
        with TestClient(self.app({'graph':npu,'chat':self.text()}),base_url='http://127.0.0.1') as client:
            accepted=client.post('/api/assignments',json={'model':'graph','groups':[[O]]})
            self.assertEqual(accepted.status_code,200,accepted.text)
            worker=next(iter(read_config(self.registry).workers.values()))
            self.assertEqual(worker.device,O);self.assertEqual(worker.gpus,[])
            bad=client.post('/api/assignments',json={'model':'chat','groups':[[O]]});self.assertEqual(bad.status_code,409)

    def test_windows_torch_tp_is_not_native_tensor_mode(self):
        with patch('sys.platform','win32'),TestClient(self.app({'m':self.text(),'native':self.native()}),base_url='http://127.0.0.1') as client:
            before=self.registry.read_bytes()
            with patch('titanos.devices.assignment_catalog',side_effect=AssertionError('Known unavailable collective must refuse before slow discovery')):
                framework=client.post('/api/assignments',json={'model':'m','mode':'tensor_parallel','groups':[[A,B]],'tensor_parallel_size':2})
            self.assertEqual(framework.status_code,422);self.assertIn('nccl_unavailable',framework.text)
            self.assertEqual(before,self.registry.read_bytes())
            native=client.post('/api/assignments',json={'model':'native','mode':'tensor_parallel','groups':[[A,B]],'tensor_parallel_size':2})
            self.assertEqual(native.status_code,200,native.text)
            self.assertEqual(native.json()['modes']['tensor_parallel']['execution_backend'],'llama.cpp')

    def test_direct_pin_reports_missing_or_masked_without_worker(self):
        self.rows[0]=gpu(A,'Card A',masked=True)
        with TestClient(self.app({'m':self.text(A)}),base_url='http://127.0.0.1') as client:
            view=client.get('/api/assignments/m').json()
            self.assertFalse(view['ready']);self.assertEqual(view['unavailable_devices'],[A])
            self.rows[:]=[row for row in self.rows if row['id']!=A]
            view=client.get('/api/assignments/m').json()
            self.assertFalse(view['ready']);self.assertEqual(view['missing_devices'],[A])

    def test_cpu_replicas_and_cpu_gloo_rank_group(self):
        with TestClient(self.app({'m':self.text()}),base_url='http://127.0.0.1') as client:
            replicas=client.post('/api/assignments',json={'model':'m','groups':[['cpu'],['cpu']]})
            self.assertEqual(replicas.status_code,200,replicas.text);self.assertEqual(replicas.json()['replicas'],2)
            tensor=client.post('/api/assignments',json={'model':'m','mode':'tensor_parallel','groups':[['cpu']],'tensor_parallel_size':3})
            self.assertEqual(tensor.status_code,200,tensor.text)
            worker=next(iter(read_config(self.registry).workers.values()))
            self.assertEqual((worker.device,worker.gpus,worker.tensor_parallel_size),('cpu',[],3))
            mixed=client.post('/api/assignments',json={'model':'m','mode':'model_sharding','groups':[['cpu',A]]})
            self.assertEqual(mixed.status_code,422)

    def test_genai_images_reject_npu_but_tensor_can_select_it(self):
        model=ModelSpec(path=str(self.root/'image'),task='image',backend='openvino-genai',device='auto')
        with TestClient(self.app({'image':model}),base_url='http://127.0.0.1') as client:
            rows=client.get('/api/assignments/devices?model=image').json()['data']
            npu=next(row for row in rows if row['id']==O)
            self.assertFalse(npu['assignable']);self.assertEqual(npu['reasons'][0]['code'],'unsupported_backend')
            self.assertEqual(client.post('/api/assignments',json={'model':'image','groups':[[O]]}).status_code,409)

    def test_observed_cached_device_never_calls_details_or_load(self):
        app=self.app({'m':self.text()});owner=app.state.engine.target
        owner.current='m';owner.states['m']='loaded';owner.runtime=NS(device='cuda:0')
        with patch.object(owner,'details',side_effect=AssertionError('No details call')),TestClient(app,base_url='http://127.0.0.1') as client:
            observed=client.get('/api/assignments/m').json()['observed']
            self.assertEqual(observed['device'],'cuda:0');self.assertTrue(observed['observed'])
            self.assertFalse(observed['observations'][0]['kernel_execution_verified'])

    def test_runtime_profile_refresh_and_failed_refresh_never_report_stale_recommendation(self):
        cpu=[{'id':'controlled-cpu','one_stop_supported':True,'recommended':True}]
        cuda=[{'id':'controlled-cuda','one_stop_supported':True,'recommended':True}]
        cfg=Config();write_config(self.registry,cfg)
        with patch('titanos.runtime_catalog.resolve_catalog',return_value='controlled-catalog'),\
             patch('titanos.runtime_catalog.compatible_profiles',side_effect=[cpu,cuda,ValueError('controlled probe unavailable')]) as profiles:
            app=create_app(cfg,registry_path=self.registry,output_dir=self.root/'out')
            with TestClient(app,base_url='http://127.0.0.1') as client:
                self.assertEqual(client.get('/api/info').json()['model_setup']['runtime']['recommended_profile'],'controlled-cpu')
                self.assertEqual(client.get('/api/runtime-profiles?refresh=true').json()['recommended'],'controlled-cuda')
                self.assertEqual(client.get('/api/info').json()['model_setup']['runtime']['recommended_profile'],'controlled-cuda')
                failed=client.get('/api/runtime-profiles?refresh=true').json()
                self.assertIsNone(failed['recommended']);self.assertEqual(failed['metadata_status'],'unverified')
                self.assertFalse(any(row['recommended'] for row in failed['data']))
                info=client.get('/api/info').json()['model_setup']['runtime']
                self.assertIsNone(info['recommended_profile']);self.assertEqual(info['metadata_status'],'unverified')
        self.assertEqual(profiles.call_count,3)

    def test_activation_pins_verified_runtime_and_rejects_profile_mismatch(self):
        home=self.root/'home';home.mkdir(exist_ok=True)
        executable=self.root/'runtime'/'llama-server.exe';executable.parent.mkdir();executable.write_bytes(b'INERT NOT AN EXECUTABLE')
        sha=hashlib.sha256(executable.read_bytes()).hexdigest()
        receipt={'executable':str(executable),'binary_sha256':sha,'profile':'controlled-cuda','fingerprint':'a'*64}
        (home/'native-runtime.json').write_text(json.dumps(receipt))
        plan=AcquisitionPlan(provider='controlled',provider_title='Controlled',candidate_id='model',registry_name='new',task='chat',backend='llama-cpp',device='auto',
            source_url='https://example.org/model',requested_revision='fixed',resolved_revision='fixed',files=[{'path':'model.gguf','size':1}],
            compatibility={},runtime={},model_spec={'task':'chat','backend':'llama-cpp','device':'auto','native':{'model_file':'model.gguf'}})
        app=self.app({});record={'plan':plan.model_dump(mode='json'),'prepared_path':str(self.root/'weights'),'status':'prepared'}
        with patch.object(app.state.acquisitions,'activation_ready',return_value=record),patch.object(app.state.acquisitions,'mark_active',side_effect=lambda identifier,activation,*a:{'id':identifier,'status':'active','activation':activation}),TestClient(app,base_url='http://127.0.0.1') as client:
            route='/api/acquisitions/'+('a'*32)+'/activate'
            bad=client.post(route,json={'load':False,'runtime_profile':'controlled-cpu'})
            self.assertEqual(bad.status_code,409);self.assertEqual(bad.json()['error']['code'],'runtime_profile_mismatch')
            good=client.post(route,json={'load':False,'runtime_profile':'controlled-cuda'})
            self.assertEqual(good.status_code,200,good.text)
            self.assertEqual(good.json()['activation']['runtime_profile'],'controlled-cuda')
            self.assertEqual(read_config(self.registry).models['new'].native.executable,str(executable))
            (home/'native-runtime.json').write_text(json.dumps({**receipt,'profile':'different','executable':str(self.root/'different.exe')}))
            self.assertEqual(read_config(self.registry).models['new'].native.executable,str(executable))


if __name__=='__main__':unittest.main(verbosity=2)
