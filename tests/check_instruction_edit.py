"""Instruction editing: real image/API plumbing, strict pipeline doubles, no weights."""
import asyncio
from contextlib import redirect_stdout
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, PropertyMock

import httpx
from PIL import Image
from fastapi.testclient import TestClient
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from titanos import cli
from titanos.backends import LoadedModel
from titanos.capabilities import describe
from titanos.client import Client
from titanos.config import Config,ModelSpec
from titanos.engine import EngineError
from titanos.instruction_edit import PIPELINE
from titanos.server import create_app
from titanos.settings import Settings,_active_settings


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.index={'_class_name':PIPELINE,'requires_safety_checker':False,'scheduler':['diffusers','EulerAncestralDiscreteScheduler'],
            'unet':['diffusers','UNet2DConditionModel'],'vae':['diffusers','AutoencoderKL'],
            'text_encoder':['transformers','CLIPTextModel'],'tokenizer':['transformers','CLIPTokenizer'],
            'safety_checker':['stable_diffusion','StableDiffusionSafetyChecker'],'feature_extractor':['transformers','CLIPImageProcessor']}
        (self.root/'model_index.json').write_text(json.dumps(self.index),encoding='utf-8')
        (self.root/'unet').mkdir();(self.root/'unet/config.json').write_text('{"in_channels":8}',encoding='utf-8')
        self.spec=ModelSpec(path=str(self.root),task='image',backend='diffusers',device='cpu',load_options={'variant':'fp16'})
        self.image=Image.new('RGB',(32,64),(19,71,143));self.addCleanup(self.image.close)
        stream=io.BytesIO();self.image.save(stream,format='PNG');self.raw=stream.getvalue()
        self.calls=[];self.loaded=[];self.after=None;self.bad=None;outer=self
        class Pipeline:
            vae_scale_factor=8;placed='meta';components=property(lambda self:{'unet':NS(parameters=lambda:iter([NS(device=self.placed)])),'tokenizer':object()})
            @classmethod
            def from_pretrained(cls,path,**kwargs):outer.loaded.append((path,kwargs));return cls()
            def to(self,device):self.placed=device;return self  # weights stay on meta until the loader moves them, so placement verification proves the move
            def set_progress_bar_config(self,**kwargs):pass
            def __call__(self,prompt,image,num_inference_steps,guidance_scale=7.5,image_guidance_scale=1.5,
                    negative_prompt=None,num_images_per_prompt=1,generator=None,output_type='pil',return_dict=True,callback_on_step_end=None):
                # No **kwargs: passing strength/height/width must fail this test.
                outer.calls.append({'prompt':prompt,'image':image.copy(),'steps':num_inference_steps,'guidance_scale':guidance_scale,
                    'image_guidance_scale':image_guidance_scale,'negative_prompt':negative_prompt,'n':num_images_per_prompt,'generator':generator})
                callback_on_step_end(self,0,1,{'latents':None})
                images=[Image.new('RGB',image.size,(29,31,37)) for _ in range(num_images_per_prompt)]
                if outer.bad=='geometry':images[0].close();images[0]=Image.new('RGB',(64,64))
                if outer.after:outer.after(images)
                return NS(images=images)
        self.pipeline_type=type(PIPELINE,(Pipeline,),{})
        self.pipeline=self.pipeline_type();self.runtime=LoadedModel.__new__(LoadedModel)
        self.runtime.spec=self.spec;self.runtime.model=self.pipeline;self.runtime.device='cpu'
        class Generator:
            def __init__(self,device):self.device=device
            def manual_seed(self,seed):self.seed=seed;return self
        self.modules=patch.dict('sys.modules',{'torch':NS(Generator=Generator,float32='float32',version=NS(hip=None),
            cuda=NS(is_available=lambda:False),backends=NS(mps=NS(is_available=lambda:False))),
            'diffusers':NS(DiffusionPipeline=NS(from_pretrained=lambda *a,**k:(_ for _ in ()).throw(AssertionError('generic loader used'))),
                StableDiffusionInstructPix2PixPipeline=self.pipeline_type)})
        self.modules.start();self.addCleanup(self.modules.stop)
        self.addCleanup(lambda:[row['image'].close() for row in self.calls])
    def data(self,**values):
        return {'model':'edit','prompt':'make the object blue','image':self.image,'size':'32x64','width':32,'height':64,'steps':2,'n':1,**values}
    def test_explicit_builtin_loader_variant_and_capabilities(self):
        before=(self.root/'model_index.json').read_bytes()
        with patch('titanos.hardware.select_device',return_value='cpu'):
            loaded=LoadedModel(self.spec)
        self.assertEqual(type(loaded.model).__name__,PIPELINE);self.assertEqual((loaded.device,loaded.placement_devices),('cpu',['cpu']))
        kwargs=self.loaded[0][1];self.assertEqual(kwargs['variant'],'fp16');self.assertTrue(kwargs['local_files_only']);self.assertTrue(kwargs['use_safetensors'])
        self.assertNotIn('safety_checker',kwargs);self.assertEqual((self.root/'model_index.json').read_bytes(),before)
        for variant in ('non_ema','fp16-test'):
            spec=ModelSpec(path=str(self.root),task='image',backend='diffusers',device='cpu',load_options={'variant':variant})
            with patch('titanos.hardware.select_device',return_value='cpu'):LoadedModel(spec)
            self.assertEqual(self.loaded[-1][1]['variant'],variant)
        for variant in ('',' ','fp16\n','fp16\r','../fp16','fp16/test','fp16\\test','fp16.bin','é',True,16,None,[]):
            with self.assertRaises(ValueError):ModelSpec(path=str(self.root),task='image',backend='diffusers',load_options={'variant':variant})
        info=describe(self.spec);self.assertTrue(info['instruction_edit']['requires_input_image'])
        self.assertFalse(info['instruction_edit']['inpainting']);self.assertFalse(info['image_mask']['implemented']);self.assertFalse(info['modes']['model_sharding']['implemented'])
        for fields in ({'defaults':{'strength':.5}},{'defaults':{'size':'32x64'}},{'allow_pickle':True},{'load_options':{'safety_checker':None}},{'load_options':{'custom_pipeline':'external'}}):
            with self.assertRaises(ValueError):ModelSpec(path=str(self.root),task='image',backend='diffusers',**fields)
        (self.root/'unet/config.json').write_text('{"in_channels":8,"_attn_implementation_internal":"publisher/custom"}',encoding='utf-8')
        with self.assertRaises(ValueError):ModelSpec(path=str(self.root),task='image',backend='diffusers')
    def test_native_instruction_call_defaults_no_silent_controls(self):
        token=_active_settings.set(Settings(default_image_guidance_scale=1.75))
        try:
            images=self.runtime.diffuse('image-edit',self.data());images[0].close()
            self.assertEqual(self.calls[-1]['image_guidance_scale'],1.75)
            self.runtime.spec=self.spec.model_copy(update={'defaults':{'image_guidance_scale':2.0,'guidance_scale':6.0}})
            images=self.runtime.diffuse('image-edit',self.data(n=2,seed=0,image_guidance_scale=2.5));self.assertEqual(len(images),2)
            for image in images:image.close()
            self.assertEqual(self.calls[-1]['image_guidance_scale'],2.5);self.assertEqual(self.calls[-1]['guidance_scale'],6)
            self.assertEqual(self.calls[-1]['generator'].seed,0)
            before=len(self.calls)
            for extra in ({'strength':.5},{'mask_image':self.image},{'width':64},{'image_guidance_scale':0},{'preserve_unmasked':True}):
                with self.assertRaises(EngineError):self.runtime.diffuse('image-edit',self.data(**extra))
            with self.assertRaises(EngineError):self.runtime.diffuse('image',self.data())
            self.assertEqual(len(self.calls),before)
        finally:_active_settings.reset(token)
    def test_http_and_actual_mcp_preserve_guidance_and_reject_unsupported(self):
        app=create_app(Config(models={'edit':self.spec}),output_dir=self.root/'out')
        app.state.engine.target.infer=lambda name,task,data,**kwargs:self.runtime.diffuse(task,data,kwargs.get('cancel'))
        with TestClient(app,base_url='http://127.0.0.1') as client:
            response=client.post('/v1/images/edits',data={'model':'edit','prompt':'make it blue','image_guidance_scale':'2'},files={'image':('in.png',self.raw)})
            self.assertEqual(response.status_code,200,response.text);self.assertEqual(self.calls[-1]['image_guidance_scale'],2)
            self.assertEqual(self.calls[-1]['image'].size,(32,64))
            before=len(self.calls)
            for extra in ({'strength':'.5'},{'mask_base64':base64.b64encode(self.raw).decode()},{'size':'64x64'}):
                response=client.post('/v1/images/edits',data={'model':'edit','prompt':'x',**extra},files={'image':('in.png',self.raw)})
                self.assertEqual(response.status_code,400,response.text)
            response=client.post('/v1/images/generations',json={'model':'edit','prompt':'x'})
            self.assertEqual(response.status_code,400);self.assertEqual(len(self.calls),before)
            async def tool():
                schema=next(t.input_schema for t in await app.state.mcp.list_tools() if t.name=='edit_image')
                self.assertIn('image_guidance_scale',schema['properties'])
                base={'model':'edit','prompt':'make it blue','image_base64':base64.b64encode(self.raw).decode()}
                result=await app.state.mcp.call_tool('edit_image',{**base,'image_guidance_scale':2.25})
                self.assertFalse(result.is_error);self.assertEqual(self.calls[-1]['image_guidance_scale'],2.25)
                for invalid in (0,True,float('inf')):
                    with self.assertRaises(ToolError):await app.state.mcp.call_tool('edit_image',{**base,'image_guidance_scale':invalid})
            with patch.object(Context,'request_context',new_callable=PropertyMock,return_value=NS(request=None)):asyncio.run(tool())
    def test_cancel_and_invalid_output_close_owned_images(self):
        cancel=threading.Event();closed=[]
        def stop(images):
            for index,image in enumerate(images):
                original=image.close
                def close(index=index,original=original):
                    closed.append(index);original()
                    if index==0:raise OSError('controlled close failure')
                image.close=close
            cancel.set()
        self.after=stop
        with self.assertLogs('titanos.instruction_edit',level='ERROR'),self.assertRaises(EngineError) as error:
            self.runtime.diffuse('image-edit',self.data(n=2),cancel)
        self.assertEqual(error.exception.code,'cancelled');self.assertEqual(closed,[0,1]);self.assertTrue(error.exception.__notes__)
        self.after=None;self.bad='geometry'
        with self.assertRaises(EngineError) as error:self.runtime.diffuse('image-edit',self.data())
        self.assertEqual(error.exception.code,'invalid_backend_response')
    def test_unsupported_form_presence_and_hidden_mcp_dimensions_never_run(self):
        app=create_app(Config(models={'edit':self.spec}),output_dir=self.root/'edges')
        app.state.engine.target.infer=lambda name,task,data,**kwargs:self.runtime.diffuse(task,data,kwargs.get('cancel'))
        with TestClient(app,base_url='http://127.0.0.1') as client:
            for fields in ({'strength':''},{'width':'64'},{'height':'32'},{'width':''},{'height':''}):
                response=client.post('/v1/images/edits',data={'model':'edit','prompt':'x',**fields},files={'image':('in.png',self.raw)})
                self.assertEqual(response.status_code,400,response.text)
                self.assertEqual(response.json()['error']['code'],'unsupported_capability')
                self.assertIn(next(iter(fields)),response.text)  # the refusal names the rejected field; width/height are not multipart fields of this route
                self.assertFalse(self.calls)
            async def invoke():
                schema=next(t.input_schema for t in await app.state.mcp.list_tools() if t.name=='edit_image')
                self.assertNotIn('width',schema['properties']);self.assertNotIn('height',schema['properties'])
                base={'model':'edit','prompt':'x','image_base64':base64.b64encode(self.raw).decode()}
                for fields in ({'width':64},{'height':32},{'width':''},{'height':0},{'strength':''},{'strength':.5}):
                    with self.assertRaises(ToolError):await app.state.mcp.call_tool('edit_image',{**base,**fields})
                    self.assertFalse(self.calls)
            with patch.object(Context,'request_context',new_callable=PropertyMock,return_value=NS(request=None)):asyncio.run(invoke())
    def test_client_and_cli_guidance_survives(self):
        path=self.root/'source.png';path.write_bytes(self.raw)
        with Client() as client,patch.object(client,'_json',return_value={'data':[]}) as send:
            client.edit('edit',path,'change',image_guidance_scale=2)
            self.assertEqual(send.call_args.kwargs['data']['image_guidance_scale'],2)
            client.edit('edit',path,'change');self.assertNotIn('image_guidance_scale',send.call_args.kwargs['data'])
        with patch.object(Client,'edit',return_value={'data':[]}) as edit,redirect_stdout(io.StringIO()):
            cli.main(['edit','edit','change','--image',str(path),'--image-guidance-scale','2.5'])
        self.assertEqual(edit.call_args.kwargs['image_guidance_scale'],2.5)


if __name__=='__main__':unittest.main(verbosity=2)
