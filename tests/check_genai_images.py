"""SD/SDXL adapter and public lifecycle contracts using vendor/image doubles only."""
import asyncio
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import redirect_stdout

import numpy as np
from fastapi.testclient import TestClient
from titanos.config import Config, ModelSpec
from titanos.engine import Engine, EngineError
from titanos.genai_images import GenAIImageModel, inspect_export
from titanos.server import create_app
from titanos.settings import Settings, _active_settings
from titanos.client import Client
from titanos import cli


class Image:
    def __init__(self, array):
        self.array=array; self.height,self.width=array.shape[:2]; self.closed=False
    def copy(self):return Image(self.array.copy())
    def __array__(self, dtype=None, copy=None):return self.array.astype(dtype or self.array.dtype, copy=copy if copy is not None else False)
    def convert(self, mode):
        assert mode=='RGB';return self.copy()
    def close(self):self.closed=True
    def save(self, target, format=None):
        data=b'controlled-image-bytes'
        if hasattr(target,'write'):target.write(data)
        else:Path(target).write_bytes(data)
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='titanos-genai-images-');self.root=Path(self.temp.name)
        self.calls=[];self.created=[];self.returned=None;self.hook=None;self.bad=None
        owner=self
        class Pipe:
            def __init__(self,*args,**props):
                owner.created.append((type(self).__name__,args,props))
                # Model the upstream copy-constructor hazard, not just its API:
                # conversion copies helpers and does not initialize missing ones.
                source=args[0]
                self.components=source.components if isinstance(source,Pipe) else object()
                self.mask_helpers=source.mask_helpers if isinstance(source,Pipe) else object() if type(self).__name__=='Inpaint' else None
            def generate(self,prompt,*inputs,**kwargs):
                if type(self).__name__=='Inpaint' and self.mask_helpers is None:
                    raise RuntimeError('controlled null mask helpers after pipeline copy')
                owner.calls.append((prompt,inputs,kwargs))
                if owner.hook:owner.hook(kwargs['callback'])
                owner.returned=np.full((kwargs['num_images_per_prompt'],kwargs['height'],kwargs['width'],3),17,dtype=np.uint8)
                return NS(data=owner.bad if owner.bad is not None else owner.returned)
        class Text(Pipe):pass
        class Edit(Pipe):pass
        class Inpaint(Pipe):pass
        replacements={'openvino':NS(Core=lambda:NS(available_devices=['CPU','GPU.0']),Tensor=lambda a:NS(data=a)),
            'openvino_genai':NS(Text2ImagePipeline=Text,Image2ImagePipeline=Edit,InpaintingPipeline=Inpaint),
            'PIL':NS(Image=NS(Image=Image,fromarray=lambda array:Image(array)))}
        self.originals={name:sys.modules.get(name) for name in replacements}
        sys.modules.update(replacements);self.context=_active_settings.set(Settings(default_image_size='64x64',default_steps=3,threads=2))
    def tearDown(self):
        _active_settings.reset(self.context)
        for name,original in self.originals.items():
            if original is None:sys.modules.pop(name,None)
            else:sys.modules[name]=original
        self.temp.cleanup()
    def export(self, *, xl=False, edit=False):
        root=self.root/'export';root.mkdir(exist_ok=True)
        (root/'model_index.json').write_text(json.dumps({'_class_name':'StableDiffusionXLPipeline' if xl else 'StableDiffusionPipeline'}),encoding='utf-8')
        (root/'scheduler').mkdir(exist_ok=True);(root/'scheduler/scheduler_config.json').write_text('{}',encoding='utf-8')
        for component in ['text_encoder','unet','vae_decoder']+(['text_encoder_2'] if xl else [])+(['vae_encoder'] if edit else []):
            (root/component).mkdir(exist_ok=True)
            (root/component/'config.json').write_text(json.dumps({'in_channels':4} if component=='unet' else {}),encoding='utf-8')
            for ext in ['xml','bin']:(root/component/('openvino_model.'+ext)).write_bytes(b'controlled export bytes')
        for component in ['tokenizer']+(['tokenizer_2'] if xl else []):
            (root/component).mkdir(exist_ok=True)
            for ext in ['xml','bin']:(root/component/('openvino_tokenizer.'+ext)).write_bytes(b'controlled tokenizer bytes')
        return root
    def spec(self, **changes):
        return ModelSpec(path=str(self.root/'export'),backend='openvino-genai',task='image',device='CPU',**changes)
    def test_admission_layout_and_chat_dispatch_preserved(self):
        self.export()
        for values in [{'device':'GPU'},{'device':'MULTI:GPU.0,CPU'},{'dtype':'float32'},
                       {'graph':'openvino_model.xml'},{'defaults':{'mask':None}},{'load_options':{'DEVICE_ID':'1'}}]:
            data={'path':str(self.root/'export'),'backend':'openvino-genai','task':'image','device':'CPU',**values}
            with self.assertRaises(ValueError):ModelSpec(**data)
        original=self.root/'export/model_index.json';original.write_text('{"_class_name":"StableDiffusionInpaintPipeline"}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Only exported'):inspect_export(self.root/'export')
        self.export();(self.root/'export/unet/config.json').write_text('{"in_channels":9}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'inpainting'):inspect_export(self.root/'export')
        chat=ModelSpec(path='local',backend='openvino-genai',task='chat')
        with patch('titanos.genai.GenAIModel',return_value=NS()) as constructor:
            engine=Engine(Config(models={'chat':chat}));engine.load('chat');constructor.assert_called_once_with(chat)
    def test_default_registration_uses_shared_runtime_gpu_selection(self):
        root=self.export();spec=ModelSpec(path=str(root),backend='openvino-genai',task='image')
        self.assertEqual(spec.device,'auto')
        with patch('titanos.compiled.compiled_device',return_value='GPU.0') as resolver:
            model=GenAIImageModel(spec);self.assertEqual(model.device,'GPU.0')
            self.assertEqual(resolver.call_args.args,('openvino-genai','auto'))
            self.assertEqual(resolver.call_args.kwargs['task'],'image')
        self.assertEqual(self.created[-1][1][1],'GPU.0')
        for device in ['GPU.0','OV-test-identity']:
            explicit=ModelSpec(path=str(root),backend='openvino-genai',task='image',device=device)
            with patch('titanos.compiled.compiled_device',return_value='GPU.0') as resolver:
                self.assertEqual(GenAIImageModel(explicit).device,'GPU.0')
                self.assertEqual(resolver.call_args.args,('openvino-genai',device))
                self.assertEqual(resolver.call_args.kwargs['task'],'image')
    def test_single_gpu_alias_preserves_explicit_selection(self):
        root=self.export()
        with patch.object(sys.modules['openvino'],'Core',return_value=NS(available_devices=['CPU','GPU'])):
            for requested,resolved in [('GPU.0','GPU.0'),('OV-example','GPU')]:
                spec=ModelSpec(path=str(root),backend='openvino-genai',task='image',device=requested)
                with patch('titanos.compiled.compiled_device',return_value=resolved):
                    self.assertEqual(GenAIImageModel(spec).device,'GPU.0')
            missing=ModelSpec(path=str(root),backend='openvino-genai',task='image',device='GPU.1')
            with patch('titanos.compiled.compiled_device',return_value='GPU.1'):
                with self.assertRaisesRegex(ValueError,'unavailable'):GenAIImageModel(missing)
    def test_sd_controls_edit_shared_components_and_output_copy(self):
        self.export(edit=True);spec=self.spec(defaults={'n':2,'num_inference_steps':5,'seed':7,'strength':.4,'size':'64x64'})
        model=GenAIImageModel(spec)
        self.assertEqual([row[0] for row in self.created],['Inpaint','Edit','Text'])
        self.assertEqual(self.created[0][1],(spec.path,'CPU'))
        self.assertIs(self.created[1][1][0],model.inpaint);self.assertIs(self.created[2][1][0],model.inpaint)
        self.assertIs(model.edit.components,model.inpaint.components);self.assertIs(model.text.components,model.inpaint.components)
        self.assertEqual(self.created[0][2]['INFERENCE_NUM_THREADS'],2)
        images=model.infer('image',{'prompt':'blue mug','steps':2})
        kwargs=self.calls[-1][2];self.assertEqual((kwargs['num_inference_steps'],kwargs['num_images_per_prompt'],kwargs['rng_seed'],kwargs['strength']),(2,2,7,1.0))
        self.returned.fill(99);self.assertTrue(np.all(images[0].array==17))
        source=Image(np.ones((32,48,3),dtype=np.uint8))
        model.infer('image-edit',{'prompt':'red mug','image':source,'strength':.25,'n':1})
        self.assertEqual(self.calls[-1][1][0].data.shape,(1,32,48,3));self.assertEqual(self.calls[-1][2]['strength'],.25)
        self.assertFalse(source.closed);self.assertTrue(model.details()['image_edit']);self.assertTrue(model.details()['inpainting']);self.assertIsNone(model.details()['qualification'])
        model.close();self.assertIsNone(model.text);self.assertIsNone(model.edit);self.assertIsNone(model.inpaint)
    def test_owner_initializes_helpers_and_unsafe_copy_trap_is_exercised(self):
        vendor=sys.modules['openvino_genai']
        unsafe=vendor.InpaintingPipeline(vendor.Image2ImagePipeline('controlled','CPU'))
        with self.assertRaisesRegex(RuntimeError,'null mask helpers'):
            unsafe.generate('x')
        for xl in (False,True):
            self.created.clear();self.export(xl=xl,edit=True);spec=self.spec();model=GenAIImageModel(spec)
            self.assertEqual([row[0] for row in self.created],['Inpaint','Edit','Text'])
            self.assertIsNotNone(model.inpaint.mask_helpers)
            self.assertIs(model.edit.mask_helpers,model.inpaint.mask_helpers)
            self.assertIs(model.text.mask_helpers,model.inpaint.mask_helpers)
            result=model.inpaint.generate('x',num_images_per_prompt=1,height=64,width=64)
            self.assertEqual(result.data.shape,(1,64,64,3))
            self.assertEqual(model.details()['component_owner'],'InpaintingPipeline');model.close()
    def test_owner_constructor_failure_releases_partial_wrappers_without_fallback(self):
        self.export(edit=True);vendor=sys.modules['openvino_genai'];closed=[]
        original_close=GenAIImageModel.close
        def close(model):
            original_close(model);closed.append((model.text,model.edit,model.inpaint))
        with patch.object(vendor,'Image2ImagePipeline',side_effect=ValueError('controlled wrapper failure')),patch.object(GenAIImageModel,'close',close):
            with self.assertRaisesRegex(ValueError,'wrapper failure'):GenAIImageModel(self.spec())
        self.assertEqual([row[0] for row in self.created],['Inpaint'])
        self.assertEqual(closed,[(None,None,None)])
    def test_sdxl_optional_edit_and_incomplete_export(self):
        self.export(xl=True);model=GenAIImageModel(self.spec());self.assertEqual(model.details()['family'],'sdxl')
        with self.assertRaisesRegex(EngineError,'no vae_encoder'):model.infer('image-edit',{'prompt':'x','image':Image(np.zeros((32,32,3),dtype=np.uint8))})
        (self.root/'export/text_encoder_2/openvino_model.bin').unlink()
        with self.assertRaisesRegex(ValueError,'text_encoder_2'):inspect_export(self.root/'export')
    def test_retained_safety_checker_files_are_not_claimed_executed(self):
        root=self.export();path=root/'model_index.json';index=json.loads(path.read_text(encoding='utf-8'))
        index.update(requires_safety_checker=True,safety_checker=['stable_diffusion','StableDiffusionSafetyChecker'],feature_extractor=['transformers','CLIPImageProcessor'])
        path.write_text(json.dumps(index),encoding='utf-8');before=path.read_bytes()
        model=GenAIImageModel(self.spec());detail=model.details()
        self.assertTrue(detail['safety_checker']['required_by_export']);self.assertFalse(detail['safety_checker']['invoked_by_adapter'])
        self.assertNotIn('safety_checker',detail['native_components']);self.assertEqual(path.read_bytes(),before)
    def test_cache_file_links_bind_content_but_directory_links_reject(self):
        root=self.export();weight=root/'unet/openvino_model.bin';blob=self.root/'blob';blob.write_bytes(weight.read_bytes());weight.unlink()
        try:weight.symlink_to(blob)
        except OSError as exc:self.skipTest(f'File symlink creation unavailable: {exc}')
        first=inspect_export(root);self.assertTrue(weight.is_symlink())
        model=GenAIImageModel(self.spec());self.assertEqual(model.export,first)
        blob.write_bytes(b'changed target bytes');self.assertNotEqual(inspect_export(root)['export_digest'],first['export_digest'])
        link=root/'linked-folder';link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'directories cannot be links'):inspect_export(root)
    def test_omitted_seed_is_fresh_and_explicit_zero_wins(self):
        self.export();model=GenAIImageModel(self.spec())
        with patch('titanos.genai_images.secrets.randbits',side_effect=[12,13]) as random_seed:
            model.infer('image',{'prompt':'x'});self.assertEqual(self.calls[-1][2]['rng_seed'],12)
            model.infer('image',{'prompt':'x'});self.assertEqual(self.calls[-1][2]['rng_seed'],13)
            model.infer('image',{'prompt':'x','seed':0});self.assertEqual(self.calls[-1][2]['rng_seed'],0)
            self.assertEqual(random_seed.call_count,2)
        self.assertEqual(model.details()['last_generation']['seed_source'],'request')
    def test_shared_blob_content_and_change_during_load(self):
        root=self.export();weight=root/'unet/openvino_model.bin';blob=self.root/'blob';blob.write_bytes(weight.read_bytes());weight.unlink();weight.hardlink_to(blob)
        # Real shared file backing plus a link-kind double works on Windows
        # without changing symlink privileges; the real symlink case is separate.
        with patch.object(Path,'is_symlink',lambda path:path==weight):
            initial=inspect_export(root)
            self.assertIn('unet/openvino_model.bin',[row['path'] for row in initial['files']])
            def mutate(*args,**kwargs):blob.write_bytes(b'changed during native load');return NS()
            with patch.object(sys.modules['openvino_genai'],'Text2ImagePipeline',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'changed while loading'):GenAIImageModel(self.spec())
    def test_invalid_controls_and_tensor_do_not_publish(self):
        self.export();model=GenAIImageModel(self.spec())
        for data in [{'n':True},{'n':5},{'steps':0},{'size':'33x64'},{'mask_image':None},{'seed':-1},{'guidance_scale':float('nan')},{'guidance_scale':10**1000}]:
            with self.assertRaises(EngineError):model.infer('image',{'prompt':'x',**data})
        for array in [np.zeros((1,64,64,3),dtype=np.float32),np.zeros((1,3,64,64),dtype=np.uint8),np.zeros((2,64,64,3),dtype=np.uint8)]:
            self.bad=array
            with self.assertRaises(EngineError) as error:model.infer('image',{'prompt':'x'})
            self.assertEqual(error.exception.status,502)
    def test_cancel_before_step_and_native_partial_or_exception(self):
        self.export();model=GenAIImageModel(self.spec());cancel=threading.Event();cancel.set()
        with self.assertRaises(EngineError):model.infer('image',{'prompt':'x'},cancel=cancel)
        self.assertFalse(self.calls);cancel.clear()
        def stop(callback):cancel.set();self.assertTrue(callback(0,3,None))
        self.hook=stop
        with self.assertRaises(EngineError) as error:model.infer('image',{'prompt':'x'},cancel=cancel)
        self.assertEqual(error.exception.code,'cancelled')
        cancel.clear()
        def fail(callback):stop(callback);raise RuntimeError('native interrupted')
        self.hook=fail
        with self.assertRaises(EngineError) as error:model.infer('image',{'prompt':'x'},cancel=cancel)
        self.assertEqual(error.exception.code,'cancelled')
    def test_http_defaults_mask_rejection_and_shared_deadline(self):
        self.export(edit=True);app=create_app(Config(models={'asset':self.spec(defaults={'n':2,'steps':4,'size':'64x64'})}),output_dir=self.root/'outputs',settings=Settings(client_timeout=.03))
        captured=[]
        def infer(name,task,data,emit=None,cancel=None):
            captured.append((task,data,cancel))
            return [Image(np.zeros((64,64,3),dtype=np.uint8)) for _ in range(data['n'])]
        app.state.engine.target.infer=infer
        with TestClient(app,base_url='http://127.0.0.1') as client:
            response=client.post('/v1/images/generations',json={'model':'asset','prompt':'x','steps':2})
            self.assertEqual(response.status_code,200,response.text);self.assertEqual(len(response.json()['data']),2)
            self.assertEqual(captured[-1][1]['steps'],2);self.assertIsNotNone(captured[-1][2])
            before=len(captured)
            response=client.post('/v1/images/edits',data={'model':'asset','prompt':'x'},files={'image':('a.png',b'x'),'mask_image':('m.png',b'x')})
            self.assertEqual(response.status_code,400);self.assertEqual(response.json()['error']['code'],'unsupported_capability');self.assertEqual(len(captured),before)
            frame=Image(np.zeros((32,32,3),dtype=np.uint8))
            with patch('titanos.backends.image_input',return_value=frame):
                response=client.post('/v1/images/edits',data={'model':'asset','prompt':'x','strength':'0.2'},files={'image':('a.png',b'x')})
            self.assertEqual(response.status_code,200,response.text);self.assertTrue(frame.closed);self.assertEqual(captured[-1][1]['strength'],.2)
            cancelled=[]
            def timed(*args,**kwargs):
                kwargs['cancel'].wait(.2);image=Image(np.zeros((64,64,3),dtype=np.uint8));cancelled.append(image);return [image]
            app.state.engine.target.infer=timed
            response=client.post('/v1/images/generations',json={'model':'asset','prompt':'x','response_format':'url'})
            self.assertEqual(response.status_code,504,response.text);self.assertTrue(cancelled[0].closed);self.assertFalse(list((self.root/'outputs').glob('*.png')))
    def test_decode_deadline_and_mcp_caller_cancellation(self):
        import time
        self.export(edit=True);app=create_app(Config(models={'asset':self.spec()}),output_dir=self.root/'out',settings=Settings(client_timeout=.01))
        calls=[];app.state.engine.target.infer=lambda *args,**kwargs:calls.append(args)
        frame=Image(np.zeros((32,32,3),dtype=np.uint8))
        def decode(*args):time.sleep(.03);return frame
        with TestClient(app,base_url='http://127.0.0.1') as client,patch('titanos.backends.image_input',side_effect=decode):
            response=client.post('/v1/images/edits',data={'model':'asset','prompt':'x'},files={'image':('a.png',b'x')})
            self.assertEqual(response.status_code,504);self.assertFalse(calls);self.assertTrue(frame.closed)
        app=create_app(Config(models={'asset':self.spec()}),output_dir=self.root/'mcp-out',settings=Settings(client_timeout=2))
        entered=threading.Event();stopped=threading.Event()
        def infer(*args,**kwargs):
            entered.set();kwargs['cancel'].wait(1);stopped.set();raise EngineError('cancelled',499,'cancelled')
        app.state.engine.target.infer=infer
        async def cancel_tool():
            task=asyncio.create_task(app.state.mcp.call_tool('generate_image',{'payload':{'model':'asset','prompt':'x'}}))
            self.assertTrue(await asyncio.to_thread(entered.wait,.5));task.cancel()
            with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,2)
            self.assertTrue(stopped.is_set())
        from mcp.server.mcpserver.context import Context
        from unittest.mock import PropertyMock
        try:
            with patch.object(Context,'request_context',new_callable=PropertyMock,return_value=NS(request=None)):
                asyncio.run(cancel_tool())
        finally:app.state.engine.close();asyncio.run(app.state.native.close())
    def test_mcp_rejects_unknown_mask_and_client_cli_keep_interfaces(self):
        import httpx
        from mcp.server.mcpserver.context import Context
        from unittest.mock import PropertyMock
        self.export(edit=True);app=create_app(Config(models={'asset':self.spec()}),output_dir=self.root/'mcp-mask')
        calls=[]
        app.state.engine.target.infer=lambda *args,**kwargs:calls.append(args) or [Image(np.zeros((64,64,3),dtype=np.uint8))]
        async def masked():
            tools=await app.state.mcp.list_tools()
            schema=next(tool.input_schema for tool in tools if tool.name=='edit_image')
            self.assertNotIn('mask',schema['properties'])
            try:
                result=await app.state.mcp.call_tool('edit_image',{'model':'asset','prompt':'x','image_base64':base64.b64encode(b'input').decode(),'mask':base64.b64encode(b'mask').decode()})
                self.assertTrue(result.is_error,'Unknown mask must not turn into an unmasked edit')
            except Exception as exc:
                self.assertIn('mask',str(exc).lower())
        try:
            with patch.object(Context,'request_context',new_callable=PropertyMock,return_value=NS(request=None)),patch('titanos.backends.image_input',return_value=Image(np.zeros((32,32,3),dtype=np.uint8))):
                asyncio.run(masked())
            self.assertFalse(calls)
        finally:app.state.engine.close();asyncio.run(app.state.native.close())
        seen=[];response={'created':1,'data':[{'b64_json':'YQ=='}]}
        def handle(request):seen.append(request);return httpx.Response(200,json=response)
        with Client('http://127.0.0.1',transport=httpx.MockTransport(handle)) as client:
            self.assertEqual(client.image('asset','blue mug',size='64x64',steps=2,seed=0),response)
        self.assertEqual(json.loads(seen[0].content),{'model':'asset','prompt':'blue mug','size':'64x64','steps':2,'seed':0})
        with patch.object(Client,'image',return_value=response) as image,redirect_stdout(io.StringIO()):
            cli.main(['image','asset','blue mug','--size','64x64','--steps','2','--seed','0'])
        self.assertEqual(image.call_args.args,('asset','blue mug'));self.assertEqual(image.call_args.kwargs['seed'],0)


if __name__=='__main__':unittest.main(verbosity=2)
