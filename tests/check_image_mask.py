"""Real PNG/HTTP boundaries plus native/Diffusers doubles; no model imports."""
import base64
import io
import sys
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from titanos import image_mask
from titanos.backends import LoadedModel
from titanos.config import Config, ModelSpec
from titanos.engine import EngineError
from titanos.genai_images import GenAIImageModel
from titanos.server import create_app
from titanos.settings import Settings, _active_settings


def png(mode='RGB', size=(32,32), color=(5,7,11), *, mask=False):
    with Image.new(mode,size,color) as im:
        if mask:im.paste((241,13,67,0),(0,0,size[0]//2,size[1]))
        stream=io.BytesIO();im.save(stream,format='PNG');return stream.getvalue()


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.settings=Settings(default_image_size='64x64',client_timeout=2)
        self.token=_active_settings.set(self.settings)
        self.raw=png();self.mask=png('RGBA',color=(1,2,3,255),mask=True)
    def tearDown(self):
        _active_settings.reset(self.token);self.temp.cleanup()
    def test_alpha_polarity_and_copy_scope(self):
        source,mask,meta=image_mask.decode(self.raw,self.mask,self.settings)
        try:
            self.assertEqual(mask.getpixel((0,0)),255);self.assertEqual(mask.getpixel((31,0)),0)
            self.assertEqual(meta['editable_pixels'],512);self.assertEqual(meta['kept_pixels'],512)
            for preserve in (False,True):
                with Image.new('RGB',source.size,(19,23,29)) as generated:
                    image_mask.finish(generated,source,mask,preserve)
                    self.assertEqual(generated.getpixel((0,0)),(19,23,29))
                    self.assertEqual(generated.getpixel((31,0)),(5,7,11) if preserve else (19,23,29))
            with Image.new('RGB',(64,64)) as wrong:
                with self.assertRaises(EngineError):image_mask.finish(wrong,source,mask,True)
        finally:source.close();mask.close()
    def test_reject_ambiguous_geometry_channels_alpha_and_empty_region(self):
        cases=[b'bad',png('RGB'),png('L',color=0),png('RGBA',(64,32),(0,0,0,0)),
               png('RGBA',color=(0,0,0,128)),png('RGBA',color=(0,0,0,255)),png('P',color=0)]
        for value in cases:
            with self.subTest(length=len(value)),self.assertRaises(EngineError):image_mask.decode(self.raw,value,self.settings)
        with self.assertRaises(EngineError):image_mask.decode(png('RGBA',color=(1,2,3,0)),self.mask,self.settings)
        with self.assertRaises(EngineError):image_mask.decode(self.raw,self.mask,Settings(max_input_image_bytes=32))
        with self.assertRaises(EngineError):image_mask.decode(self.raw,self.mask,Settings(max_input_image_pixels=512))
        with Image.new('RGBA',(32,32),(0,0,0,0)) as first,Image.new('RGBA',(32,32),(0,0,0,255)) as second:
            stream=io.BytesIO();first.save(stream,format='PNG',save_all=True,append_images=[second])
            with self.assertRaises(EngineError):image_mask.decode(self.raw,stream.getvalue(),self.settings)
        source,mask,meta=image_mask.decode(self.raw,png('RGBA',color=(0,0,0,0)),self.settings)
        self.assertEqual(meta['editable_pixels'],1024);self.assertEqual(meta['kept_pixels'],0);source.close();mask.close()
        self.assertEqual(image_mask.inline_mask('data:image/png;base64,'+base64.b64encode(self.mask).decode(),10000),self.mask)
        for value in ['%%%','data:image/jpeg;base64,YQ==','x'*100]:
            with self.assertRaises(EngineError):image_mask.inline_mask(value,16)
    def app(self,defaults=None,settings=None):
        self.calls=[];self.output_hook=None
        app=create_app(Config(models={'asset':ModelSpec(path='unused',backend='diffusers',task='image',defaults=defaults or {})}),output_dir=self.root/'out',settings=settings or self.settings)
        def infer(name,task,data,emit=None,cancel=None):
            self.calls.append({'task':task,'data':dict(data),'mask':data.get('mask_image').tobytes() if 'mask_image' in data else None})
            images=[Image.new('RGB',(data['width'],data['height']),(19,23,29)) for _ in range(data['n'])]
            if self.output_hook:self.output_hook(images,cancel)
            return images
        app.state.engine.target.infer=infer
        return app
    def post(self,client,extra=None,mask=True,files=None):
        data={'model':'asset','prompt':'change marked area',**(extra or {})}
        files=files or {'image':('source.png',self.raw),**({'mask':('mask.png',self.mask)} if mask else {})}
        return client.post('/v1/images/edits',data=data,files=files)
    def test_http_mask_geometry_default_precedence_and_returned_pixels(self):
        for defaults,setting,explicit,expected,origin in [({},True,None,True,'process'),({},False,None,False,'process'),
                ({'preserve_unmasked':False,'size':'64x64'},True,None,False,'model'),
                ({'preserve_unmasked':False},False,'true',True,'request'),({'preserve_unmasked':True},True,'false',False,'request')]:
            with self.subTest(defaults=defaults,explicit=explicit),TestClient(self.app(defaults,Settings(preserve_unmasked=setting)),base_url='http://127.0.0.1') as client:
                response=self.post(client,{} if explicit is None else {'preserve_unmasked':explicit})
                self.assertEqual(response.status_code,200,response.text);result=response.json()
                self.assertEqual(result['masking']['preserve_unmasked'],expected)
                self.assertEqual(result['masking']['preservation_source'],origin)
                self.assertEqual(result['masking']['postprocessing'],'copy_unmasked_rgb' if expected else 'none')
                self.assertEqual(self.calls[0]['data']['size'],'32x32');self.assertEqual(self.calls[0]['task'],'image-edit')
                with Image.open(io.BytesIO(base64.b64decode(result['data'][0]['b64_json']))) as im:
                    self.assertEqual(im.size,(32,32));self.assertEqual(im.getpixel((31,0)),(5,7,11) if expected else (19,23,29))
                self.assertEqual(self.calls[0]['mask'][0],255);self.assertEqual(self.calls[0]['mask'][31],0)
    def test_http_bad_inputs_zero_forward_and_unmasked_regression(self):
        with TestClient(self.app(),base_url='http://127.0.0.1') as client:
            for extra,mask,files in [({'size':'64x64'},True,None),({'preserve_unmasked':'yes'},True,None),
                    ({'preserve_unmasked':'false'},False,None),({'mask_base64':base64.b64encode(self.mask).decode()},True,None),
                    ({'mask_image':'anything'},False,None),({'mask':''},False,None),({'mask_base64':''},False,None),
                    ({'preserve_unmasked':''},True,None),({},True,{'image':('a',self.raw),'mask':('m',b'x')}),
                    ({},True,[('image',('a',self.raw)),('mask',('m',self.mask)),('mask',('m2',self.mask))])]:
                response=self.post(client,extra,mask,files);self.assertEqual(response.status_code,400,response.text)
            self.assertEqual(self.calls,[])
            response=self.post(client,{'mask_base64':base64.b64encode(self.mask).decode()},False)
            self.assertEqual(response.status_code,200,response.text)
            response=self.post(client,mask=False);self.assertEqual(response.status_code,200,response.text)
            self.assertNotIn('masking',response.json());self.assertIsNone(self.calls[-1]['mask'])
    def test_cancel_during_generated_or_source_close_leaves_no_png(self):
        for close_source in (False,True):
            app=self.app();cancel_holder=[]
            run=app.state.inference_run
            def observe(payload,task,data,cancel,deadline):
                cancel_holder.append(cancel)
                return run(payload,task,data,cancel,deadline)
            app.state.inference_run=observe
            original_decode=image_mask.decode
            def decode(*args):
                source,mask,meta=original_decode(*args)
                if close_source:
                    original=source.close
                    def close():cancel_holder[0].set();original()
                    source.close=close
                return source,mask,meta
            def hook(images,cancel):
                if not close_source:
                    original=images[0].close
                    def close():cancel_holder[0].set();original()
                    images[0].close=close
            self.output_hook=hook
            with TestClient(app,base_url='http://127.0.0.1') as client,patch('titanos.image_mask.decode',side_effect=decode):
                response=self.post(client,{'response_format':'url'})
                self.assertEqual(response.status_code,499,response.text)
                self.assertFalse(list((self.root/'out').glob('*.png')))
    def test_invalid_backend_results_are_typed_and_leave_no_assets(self):
        for result in (None,NS(),[],[Image.new('RGB',(64,64))],[Image.new('L',(32,32))],
                       [Image.new('RGB',(32,32)),Image.new('RGB',(32,32))]):
            app=self.app();app.state.engine.target.infer=lambda *args,**kwargs:result
            with TestClient(app,base_url='http://127.0.0.1') as client:
                response=self.post(client,{'response_format':'url'})
                self.assertEqual(response.status_code,502,response.text)
                self.assertEqual(response.json()['error']['code'],'invalid_backend_response')
                self.assertFalse(list((self.root/'out').glob('*.png')))
    def test_genai_actual_mask_dispatch_copies_and_cancellation(self):
        calls=[];returned=[];stop=threading.Event()
        class Inpaint:
            def generate(self,prompt,image,mask,**kwargs):
                assert kwargs['num_images_per_prompt']==1,'Native four-channel inpainting requires batch one'
                calls.append((image.data.copy(),mask.data.copy(),kwargs))
                array=np.full((1,32,32,3),19,dtype=np.uint8);returned.append(array)
                if stop.is_set():kwargs['callback'](0,1,None)
                return NS(data=array)
        model=GenAIImageModel.__new__(GenAIImageModel);model.spec=ModelSpec(path='unused',task='image',backend='openvino-genai',device='CPU')
        model.edit=NS();model.inpaint=Inpaint();model.device='CPU'
        source,mask,_=image_mask.decode(self.raw,self.mask,self.settings)
        data={'image':source,'mask_image':mask,'prompt':'x','size':'32x32','steps':1}
        try:
            with patch.dict('sys.modules',{'openvino':NS(Tensor=lambda x:NS(data=x)),'openvino_genai':NS(InpaintingPipeline=Inpaint)}):
                result=model.infer('image-edit',data)
                self.assertEqual(calls[0][1].shape,(1,32,32,3))
                self.assertTrue(np.all(calls[0][1][0,0,0]==255));self.assertTrue(np.all(calls[0][1][0,0,31]==0))
                returned[0][:]=0;self.assertEqual(result[0].getpixel((0,0)),(19,19,19));result[0].close()
                with self.assertRaises(EngineError):model.infer('image-edit',{**data,'size':'64x64'})
                self.assertEqual(len(calls),1)
                original=Inpaint.generate
                def cancelled(pipe,*args,**kwargs):stop.set();return original(pipe,*args,**kwargs)
                with patch.object(Inpaint,'generate',cancelled),self.assertRaises(EngineError) as error:
                    model.infer('image-edit',data,cancel=stop)
                self.assertEqual(error.exception.code,'cancelled')
                model.inpaint=None;stop.clear()
                with patch.object(Inpaint,'__init__',side_effect=ValueError('unsafe conversion')) as constructor:
                    with self.assertRaises(EngineError) as error:model.infer('image-edit',data)
                    constructor.assert_not_called()
                self.assertEqual(error.exception.code,'unsupported_capability');self.assertEqual(len(calls),2)
        finally:source.close();mask.close()
    def test_genai_serial_mask_count_seed_wrap_and_partial_failure_cleanup(self):
        calls=[];copies=[];failure=[None];cancel=threading.Event()
        class Inpaint:
            def generate(self,prompt,image,mask,**kwargs):
                assert kwargs['num_images_per_prompt']==1,'Controlled native batch-one constraint'
                calls.append(kwargs['rng_seed'])
                if len(calls)==2:
                    if failure[0]=='raise':raise RuntimeError('controlled second image failure')
                    if failure[0]=='cancel':cancel.set()
                return NS(data=np.full((1,32,32,3),len(calls),dtype=np.float32 if failure[0]=='bad' and len(calls)==2 else np.uint8))
        model=GenAIImageModel.__new__(GenAIImageModel);model.spec=ModelSpec(path='unused',task='image',backend='openvino-genai',device='CPU')
        model.edit=NS();model.inpaint=Inpaint();model.device='CPU'
        source,mask,_=image_mask.decode(self.raw,self.mask,self.settings)
        data={'image':source,'mask_image':mask,'prompt':'x','size':'32x32','steps':1,'n':2,'seed':2**32-1}
        original_fromarray=Image.fromarray
        def observed_output(array):
            image=original_fromarray(array);original_copy=image.copy
            def copied():
                result=original_copy();copies.append(result);return result
            image.copy=copied
            return image
        try:
            with patch.dict('sys.modules',{'openvino':NS(Tensor=lambda x:NS(data=x))}):
                result=model.infer('image-edit',data)
                self.assertEqual(calls,[2**32-1,0]);self.assertEqual(len(result),2)
                self.assertEqual(model.last_generation['native_batch_size'],1)
                self.assertEqual(model.last_generation['rng_seeds'],calls)
                self.assertTrue(model.last_generation['completed'])
                self.assertEqual(result[0].getpixel((0,0)),(1,1,1));self.assertEqual(result[1].getpixel((0,0)),(2,2,2))
                for image in result:image.close()
                for mode in ('raise','cancel','bad'):
                    calls.clear();copies.clear();cancel.clear();failure[0]=mode
                    with patch.object(Image,'fromarray',observed_output),self.assertRaises((RuntimeError,EngineError)):
                        model.infer('image-edit',data,cancel=cancel)
                    self.assertEqual(calls,[2**32-1,0]);self.assertFalse(model.last_generation['completed'])
                    self.assertTrue(copies)
                    for image in copies:
                        with self.assertRaises(ValueError):image.getpixel((0,0))
        finally:source.close();mask.close()
    def test_cleanup_failure_preserves_primary_and_attempts_all_owned_images(self):
        for mode in ('third_native_failure','third_copy_failure','ordinary_final_cancel'):
            outputs=[];closed=[];native_calls=[];copy_calls=[];cancel=threading.Event()
            primary=ValueError('original ' + mode)
            class Owned:
                def __init__(self):self.index=len(outputs);self.closed=False;outputs.append(self)
                def close(self):
                    closed.append(self.index);self.closed=True
                    if self.index==0:raise OSError('injected first output close failure')
            class Frame:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def copy(self):return Owned()
            def fromarray(array):
                copy_calls.append(1)
                if mode=='third_copy_failure' and len(copy_calls)==3:raise primary
                return Frame()
            class Pipe:
                def generate(self,*args,**kwargs):
                    native_calls.append(1)
                    if mode=='third_native_failure' and len(native_calls)==3:raise primary
                    return NS(data=np.zeros((kwargs['num_images_per_prompt'],32,32,3),dtype=np.uint8))
            model=GenAIImageModel.__new__(GenAIImageModel);model.spec=ModelSpec(path='unused',task='image',backend='openvino-genai',device='CPU')
            model.edit=NS();model.inpaint=Pipe();model.text=Pipe();model.device='CPU'
            source,mask,_=image_mask.decode(self.raw,self.mask,self.settings)
            data={'prompt':'x','size':'32x32','steps':1,'n':2 if mode=='ordinary_final_cancel' else 3}
            if mode=='third_native_failure':data.update(image=source,mask_image=mask)
            previous=sys.getprofile()
            # Deterministically cancel after the entire ordinary copy phase has
            # returned, exercising its final error cleanup instead of copy cleanup.
            def after_copy(frame,event,arg):
                if (event=='return' and frame.f_globals.get('__name__')=='titanos.genai_images'
                        and frame.f_code.co_name=='copy_output' and isinstance(arg,list)):
                    cancel.set()
            try:
                with self.subTest(mode=mode),patch.dict('sys.modules',{'openvino':NS(Tensor=lambda x:NS(data=x))}),patch.object(Image,'fromarray',fromarray):
                    with self.assertLogs('titanos.genai_images',level='ERROR') as log,self.assertRaises((ValueError,EngineError)) as error:
                        if mode=='ordinary_final_cancel':sys.setprofile(after_copy)
                        model.infer('image-edit' if mode=='third_native_failure' else 'image',data,cancel=cancel)
                    if mode=='ordinary_final_cancel':self.assertEqual(error.exception.code,'cancelled')
                    else:self.assertIs(error.exception,primary)
                    self.assertEqual(closed,[0,1]);self.assertTrue(all(output.closed for output in outputs))
                    self.assertIn('injected first output close failure',' '.join(error.exception.__notes__))
                    self.assertIn('injected first output close failure',' '.join(log.output))
                    self.assertFalse(model.last_generation['completed'])
            finally:
                sys.setprofile(previous);source.close();mask.close()

    def test_diffusers_mask_dispatch_no_fallback_or_control_leak(self):
        calls=[];converted=[]
        class Inpaint:
            def set_progress_bar_config(self,**kwargs):pass
            def __call__(self,callback_on_step_end=None,**kwargs):
                calls.append(kwargs);return NS(images=[Image.new('RGB',(32,32))])
        def convert(pipe):converted.append(pipe);return Inpaint()
        source,mask,_=image_mask.decode(self.raw,self.mask,self.settings)
        model=LoadedModel.__new__(LoadedModel);model.device='cpu';model.spec=NS(defaults={'preserve_unmasked':True})
        data={'image':source,'mask_image':mask,'prompt':'x','width':32,'height':32,'steps':1,'n':1,'strength':.75}
        try:
            with patch.dict('sys.modules',{'torch':NS(),'diffusers':NS(AutoPipelineForInpainting=NS(from_pipe=convert))}):
                for family in ('StableDiffusionPipeline','StableDiffusionXLPipeline'):
                    model.model=type(family,(),{})();result=model.diffuse('image-edit',data);result[0].close()
                    self.assertIs(calls[-1]['mask_image'],mask);self.assertNotIn('preserve_unmasked',calls[-1])
                model.model=NS()
                with self.assertRaises(EngineError) as error:model.diffuse('image-edit',data)
                self.assertEqual(error.exception.code,'unsupported_capability');self.assertEqual(len(calls),2)
        finally:source.close();mask.close()
    def test_registration_and_process_controls(self):
        self.assertFalse(Settings.from_env({'TITANOS_PRESERVE_UNMASKED':'false'}).preserve_unmasked)
        self.assertTrue(Settings().describe()['settings']['preserve_unmasked']['default'])
        for value in ('false',None,1,{},[]):
            with self.assertRaises(ValueError):ModelSpec(path='unused',task='image',backend='diffusers',defaults={'preserve_unmasked':value})
        with self.assertRaises(ValueError):ModelSpec(path='unused',task='image',backend='diffusers',defaults={'mask_image':'bytes'})


if __name__=='__main__':unittest.main(verbosity=2)
