"""No models: validate routing, adapter binding, controls, cancellation and profiles."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from PIL import Image
from fastapi.testclient import TestClient
from titanos.config import Config, ModelSpec
from titanos.engine import Engine, EngineError
from titanos.compiled import onnx_session as native_session_factory
from titanos.onnx_images import ONNXImageModel, COMPONENTS, inspect_export
from titanos.server import create_app
from titanos.settings import Settings, _active_settings


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='titanos-onnx-images-')
        self.root = Path(self.temp.name); self.model = self.root/'model'; self.model.mkdir()
        self.context = _active_settings.set(Settings(default_image_size='64x64', default_steps=3))
        index = {name: ['diffusers', 'OnnxRuntimeModel'] for name in COMPONENTS}
        index.update(_class_name='StableDiffusionOnnxPipeline', scheduler=['diffusers','PNDMScheduler'],
                     tokenizer=['transformers','CLIPTokenizer'], feature_extractor=['transformers','CLIPFeatureExtractor'])
        (self.model/'model_index.json').write_text(json.dumps(index))
        files = {name+'/model.onnx': b'controlled graph, not a model' for name in COMPONENTS}
        files.update({'scheduler/scheduler_config.json': b'{"_class_name":"PNDMScheduler"}',
                      'tokenizer/merges.txt': b'merges', 'tokenizer/vocab.json': b'{}',
                      'tokenizer/tokenizer_config.json': b'{}', 'feature_extractor/preprocessor_config.json': b'{}'})
        for name, data in files.items():
            path=self.model/name; path.parent.mkdir(exist_ok=True, parents=True);path.write_bytes(data)
        self.calls=[]; self.native_calls=[]; self.returned=[]; self.hook=None; self.checker_failure=False
        owner=self
        class Session:
            def __init__(self, component, prefix): self.component,self.prefix=component,prefix
            def run(self,*a,**k):
                owner.native_calls.append(self.component)
                if self.component=='safety_checker' and owner.checker_failure:raise RuntimeError('controlled checker failure')
                return []
            def get_providers(self): return ['DmlExecutionProvider', 'CPUExecutionProvider']
            def end_profiling(self):
                path=Path(str(self.prefix)+'.json')
                path.write_text(json.dumps([{'cat':'Node','args':{'provider':'DmlExecutionProvider','op_name':'DmlFusedNode'}}]))
                return str(path)
        def session(path,device,fallback,options,prefix):
            owner.calls.append((Path(path).parent.name,device,fallback,options,prefix))
            return Session(Path(path).parent.name,prefix)
        class Wrapper:
            def __init__(self, model):self.model=model
        class Local:
            model_max_length=77
            def __call__(self,text,**kw):return {'input_ids':[0]+list(range(len(text.split())))+[1]}
            @classmethod
            def from_pretrained(cls,path,**kw):
                assert kw=={'local_files_only':True};return cls()
        class Pipeline:
            def __init__(self,**parts):
                self.parts=parts;assert parts['requires_safety_checker'] is True and parts['vae_encoder'] is None
                self.tokenizer=parts['tokenizer']
            def set_progress_bar_config(self,**kw): pass
            def __call__(self,**kw):
                owner.request=kw
                self.parts['unet'].model.run(None,{})
                if owner.hook:owner.hook(kw)
                self.parts['vae_decoder'].model.run(None,{})
                self.parts['safety_checker'].model.run(None,{})
                owner.returned=[Image.new('RGB',(kw['width'],kw['height']), 'red') for _ in range(kw['num_images_per_prompt'])]
                return NS(images=owner.returned,nsfw_content_detected=[False]*len(owner.returned))
        self.patchers=[patch('titanos.compiled.onnx_session',side_effect=session),
            patch('titanos.dxgi.adapters',return_value=[{'device_id':2,'luid':'00000000:00000012','software':False,'name':'controlled AMD','vendor_id':'0x1002'}]),
            patch.dict(sys.modules,{'diffusers':NS(OnnxStableDiffusionPipeline=Pipeline,OnnxRuntimeModel=Wrapper,PNDMScheduler=Local),
                                    'transformers':NS(CLIPTokenizer=Local,CLIPImageProcessor=Local)})]
        for p in self.patchers:p.start()
    def tearDown(self):
        for image in self.returned:image.close()
        for p in reversed(self.patchers):p.stop()
        _active_settings.reset(self.context);self.temp.cleanup()
    def spec(self, **kw):
        data=dict(path=str(self.model),backend='onnx-diffusers',task='image',device='dml',allow_cpu_fallback=True,
                  load_options={'device_id':2,'adapter_luid':'00000000:00000012','profile_dir':str(self.root/'profiles')})
        data.update(kw);return ModelSpec(**data)
    def request_data(self,**kw):return {'prompt':'one red mug','size':'64x64','n':1,'steps':3,'seed':123,**kw}
    def test_binding_and_policies(self):
        runtime=ONNXImageModel(self.spec())
        self.assertEqual(len(self.calls),4)
        self.assertTrue(all(c[1:4]==('dml',True,{'device_id':2}) for c in self.calls))
        self.assertEqual(runtime.details()['adapter']['vendor_id'],'0x1002')
        self.assertTrue(runtime.details()['safety_checker']['configured'])
        self.assertEqual(runtime.details()['safety_checker']['completed_native_calls'],0)
        runtime.close();self.assertFalse(runtime.sessions)
        for opts in ({'device_id':0,'adapter_luid':'00000000:00000012'}, {'device_id':2,'adapter_luid':'00000000:00000013'}):
            before=len(self.calls)
            with self.assertRaises(ValueError):ONNXImageModel(self.spec(load_options=opts))
            self.assertEqual(len(self.calls),before)
    def test_registration_rejects_unimplemented_controls(self):
        for kw in ({'device':'auto'}, {'device':'cuda:0'}, {'task':'video'}, {'allow_pickle':True},
                   {'defaults':{'strength':.5}}, {'load_options':{'device_id':True,'adapter_luid':'00000000:00000012'}},
                   {'load_options':{'device_id':2,'adapter_luid':'bad'}}, {'parallelism':'model_sharding'}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):self.spec(**kw)
        self.assertFalse(self.calls)
    def test_metacommands_strict_boolean_and_exact_provider_mapping(self):
        for value in ('omitted',False,True):
            options=dict(self.spec().load_options)
            expected={'device_id':2}
            if value!='omitted':
                options['disable_metacommands']=value
                expected['disable_metacommands']='true' if value else 'false'
            self.calls.clear()
            runtime=ONNXImageModel(self.spec(load_options=options))
            self.assertEqual(len(self.calls),4)
            self.assertTrue(all(call[3]==expected for call in self.calls))
            self.assertEqual(runtime.details()['requested_provider_options'],expected)
            self.assertEqual(runtime.spec.load_options,options)
            runtime.close()
        for value in (0,1,'true','false',None,{},[]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.spec(load_options={**self.spec().load_options,'disable_metacommands':value})
        for value in (False,True):
            with self.assertRaises(ValueError):
                self.spec(device='cpu',allow_cpu_fallback=False,load_options={'disable_metacommands':value})
    def test_literal_option_reaches_ort_without_boolean_coercion(self):
        vendor=MagicMock();vendor.get_available_providers.return_value=['DmlExecutionProvider','CPUExecutionProvider']
        vendor.InferenceSession.return_value.get_providers.return_value=['DmlExecutionProvider','CPUExecutionProvider']
        with patch.dict(sys.modules,{'onnxruntime':vendor}),patch('titanos.compiled.platform.system',return_value='Windows'):
            for value in (None,'false','true'):
                options={'device_id':2}
                if value is not None:options['disable_metacommands']=value
                native_session_factory('controlled-unused.onnx','dml',True,options)
                actual=vendor.InferenceSession.call_args.kwargs['providers']
                self.assertEqual(actual,[('DmlExecutionProvider',{**options,'device_id':'2'}),'CPUExecutionProvider'])
    def test_frozen_builtin_components(self):
        original=(self.model/'model_index.json').read_text(); index=json.loads(original)
        for key,value in [('safety_checker',[None,None]),('scheduler',['custom','BadScheduler']),('_class_name','StableDiffusionXLPipeline')]:
            changed={**index,key:value};(self.model/'model_index.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):inspect_export(self.model)
        (self.model/'model_index.json').write_text(original)
        self.assertTrue(inspect_export(self.model)['export_digest'])
    def test_infer_count_controls_and_profiles(self):
        runtime=ONNXImageModel(self.spec(defaults={'n':2,'guidance_scale':8.2}))
        data=self.request_data(negative_prompt='text');del data['n']
        output=runtime.infer('image',data)
        self.assertEqual(len(output),2);self.assertEqual(self.request['guidance_scale'],8.2)
        self.assertEqual(self.request['num_inference_steps'],3)
        self.assertEqual(runtime.details()['last_generation']['seed'],123)
        self.assertTrue(runtime.details()['last_generation']['completed'])
        self.assertEqual(runtime.details()['safety_checker']['completed_native_calls'],1)
        self.assertEqual(set(runtime.details()['profiles']['components']),set(COMPONENTS))
        self.assertEqual(self.native_calls,['unet','vae_decoder','safety_checker'])
        for row in runtime.details()['profiles']['components'].values():self.assertEqual(row['operators'][0]['provider'],'DmlExecutionProvider')
        runtime.close()
    def test_rejection_precedes_inference(self):
        runtime=ONNXImageModel(self.spec())
        for change in ({'n':True},{'n':0},{'steps':0},{'seed':-1},{'size':'63x64'}, {'guidance_scale':float('nan')},
                       {'negative_prompt':'ignore','guidance_scale':1}, {'strength':.5},{'image':object()}, {'prompt':'word '*80}):
            with self.subTest(change=change),self.assertRaises(EngineError):runtime.infer('image',self.request_data(**change))
        with self.assertRaises(EngineError):runtime.infer('image-edit',self.request_data())
        self.assertFalse(self.native_calls)
        runtime.close()
    def test_cancel_stops_next_component(self):
        runtime=ONNXImageModel(self.spec());cancel=threading.Event()
        self.hook=lambda _:cancel.set()
        with self.assertRaises(EngineError) as result:runtime.infer('image',self.request_data(),cancel=cancel)
        self.assertEqual(result.exception.code,'cancelled');self.assertEqual(self.native_calls,['unet'])
        self.assertFalse(runtime.details()['last_generation']['completed'])
        self.assertEqual(runtime.details()['safety_checker']['completed_native_calls'],0)
        runtime.close()
    def test_checker_failure_is_not_reported_as_completed(self):
        runtime=ONNXImageModel(self.spec());self.checker_failure=True
        with self.assertRaisesRegex(RuntimeError,'controlled checker failure'):runtime.infer('image',self.request_data())
        self.assertFalse(runtime.details()['last_generation']['completed'])
        self.assertEqual(runtime.details()['safety_checker']['completed_native_calls'],0)
        self.assertIn('safety_checker',self.native_calls)
        runtime.close()
    def test_fatal_device_error_retains_local_context_and_rejects_reuse(self):
        engine=Engine(Config(models={'image':self.spec(),'other':self.spec()}))
        engine.load('image');runtime=engine.runtime
        native_error=RuntimeError('[ONNXRuntimeError] : 1 : FAIL : DmlExecutionProvider DmlCommandRecorder.cpp 887A0006 The GPU will not respond')
        with patch.object(runtime.sessions['unet'],'run',side_effect=native_error) as run:
            with self.assertRaises(EngineError) as failure:engine.infer('image','image',self.request_data())
            self.assertEqual((failure.exception.status,failure.exception.code),(503,'device_unavailable'))
            self.assertEqual(runtime.runtime_fault['hresult'],'0x887A0006')
            self.assertEqual(runtime.runtime_fault['component'],'unet')
            self.assertEqual(run.call_count,1)
            with patch.object(runtime,'close',side_effect=AssertionError('Native destructor must not run')) as close:
                for operation in (lambda:engine.infer('image','image',self.request_data()),lambda:engine.load('image'),lambda:engine.load('other'),lambda:engine.unload('image'),engine.close):
                    with self.assertRaises(EngineError) as blocked:operation()
                    self.assertEqual(blocked.exception.code,'device_unavailable')
                self.assertEqual(close.call_count,0)
            self.assertEqual(run.call_count,1)
        self.assertIs(engine.runtime,runtime);self.assertEqual(engine.current,'image')
        info=engine.details('image');self.assertEqual(info['status'],'unavailable');self.assertTrue(info['resident'])
        rows=engine.catalog(loaded_only=True)['data'];self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['status'],'unavailable');self.assertTrue(rows[0]['resident'])
        held_model=runtime.model;held_sessions=dict(runtime.sessions)
        with self.assertRaises(EngineError):runtime.close()
        self.assertIs(runtime.model,held_model);self.assertEqual(runtime.sessions,held_sessions)
    def test_fatal_constructor_preserves_failed_runtime_for_engine(self):
        engine=Engine(Config(models={'image':self.spec()}))
        error=RuntimeError('[ONNXRuntimeError] : FAIL : DmlExecutionProvider 0x887A0005 device removed')
        with patch('titanos.compiled.onnx_session',side_effect=error),patch.object(ONNXImageModel,'close',side_effect=AssertionError('Unsafe destructor')) as close:
            with self.assertRaises(EngineError) as failure:engine.load('image')
            self.assertEqual(failure.exception.code,'device_unavailable');self.assertEqual(close.call_count,0)
        self.assertIsNotNone(engine.runtime);self.assertEqual(engine.current,'image')
        self.assertEqual(engine.runtime.runtime_fault['component'],'load')
        self.assertEqual(engine.catalog()['data'][0]['status'],'unavailable')
    def test_ordinary_validation_error_does_not_poison_context(self):
        runtime=ONNXImageModel(self.spec())
        for text in ('bad shape','out of memory','user path 887A0006','[ONNXRuntimeError] : DmlExecutionProvider 8007000E out of memory'):
            self.assertIsNone(runtime.device_failure(ValueError(text),'unet'))
        self.assertIsNone(runtime.runtime_fault)
        with patch.object(runtime.sessions['unet'],'run',side_effect=ValueError('invalid tensor shape')):
            with self.assertRaises(ValueError):runtime.infer('image',self.request_data())
        self.assertIsNone(runtime.runtime_fault)
        self.assertEqual(len(runtime.infer('image',self.request_data())),1)
        runtime.close()
    def test_http_uses_existing_contract(self):
        config=Config(models={'amd-mug':self.spec()})
        with TestClient(create_app(config, output_dir=self.root/'outputs'), base_url='http://127.0.0.1') as client:
            result=client.post('/v1/images/generations',json={'model':'amd-mug',**self.request_data(n=2)})
            self.assertEqual(result.status_code,200,result.text)
            self.assertEqual(len(result.json()['data']),2)
            with Image.open(io.BytesIO(base64.b64decode(result.json()['data'][0]['b64_json']))) as image:self.assertEqual(image.size,(64,64))
        self.assertEqual(self.request['num_images_per_prompt'],2)
    def test_hub_keeps_external_graph_data(self):
        from titanos import hub
        files=[NS(rfilename=p.relative_to(self.model).as_posix()) for p in self.model.rglob('*') if p.is_file()]
        files.append(NS(rfilename='unet/weights.pb'))
        info=NS(sha='f'*40,siblings=files)
        args=NS(anonymous=True,token_env=None,source='controlled/publisher',revision='onnx',backend='onnx-diffusers',task='image',graph=None,cache_dir=str(self.root/'pull-cache'))
        with patch.object(hub,'HfApi',return_value=NS(model_info=lambda *a,**k:info)),patch.object(hub,'snapshot_download',return_value='snapshot') as download:
            # The pull returns the plain per-revision directory it planned, never the downloader's return value, and names
            # every sibling explicitly (literal allow_patterns: the graphs' external tensor data travels with the graphs).
            self.assertEqual(hub.pull(args),(str(hub.destination(args,info)),'f'*40))
            self.assertNotIn('ignore_patterns',download.call_args.kwargs)
            self.assertEqual(download.call_args.kwargs['allow_patterns'],sorted(f.rfilename for f in files))
            self.assertIn('unet/weights.pb',download.call_args.kwargs['allow_patterns'])
            info.siblings=[f for f in files if f.rfilename!='safety_checker/model.onnx']
            with self.assertRaises(ValueError):hub.pull(args)
            self.assertEqual(download.call_count,1)

if __name__=='__main__':unittest.main()
