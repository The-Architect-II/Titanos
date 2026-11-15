"""Real PNG and API contracts with builtin-SAM-shaped doubles; no weights."""
import asyncio
import base64
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, PropertyMock

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from mcp.server.mcpserver.context import Context
from titanos.api_plus import SegmentationRequest
from titanos.config import Config, ModelSpec
from titanos.engine import EngineError
from titanos.loader_policy import validate_attention_selectors
from titanos.segmentation import SegmentationModel, SegmentationResult, SegmentationPrompt
from titanos.server import create_app
from titanos.settings import Settings, _active_settings


class Tensor:
    def __init__(self,value):self.value=np.asarray(value);self.shape=self.value.shape;self.ndim=self.value.ndim
    def numel(self):return self.value.size
    def is_floating_point(self):return self.value.dtype.kind=='f'
    def to(self,*args,**kwargs):return self
    def detach(self):return self
    def float(self):return Tensor(self.value.astype('float32'))
    def cpu(self):return self
    def numpy(self):return self.value


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.image=Image.new('RGB',(16,8),(19,71,143));self.addCleanup(self.image.close)
        stream=io.BytesIO();self.image.save(stream,format='PNG');self.png=stream.getvalue()
        self.spec=ModelSpec(path=str(self.root),task='image-segmentation',device='cpu')
        self.calls=[];self.preprocess=[];self.hook=None;self.bad=None;outer=self
        class Processor:
            def __init__(self):
                self.image_processor=NS(pad_size={'height':32,'width':32},size={'longest_edge':32},do_resize=True,do_pad=True,
                    post_process_masks=self.post)
            def __call__(self,images,return_tensors,**hints):
                outer.preprocess.append(hints)
                result={'pixel_values':Tensor(np.zeros((1,3,32,32),dtype='float32')),
                    'original_sizes':Tensor([[8,16]]),'reshaped_input_sizes':Tensor([[16,32]])}
                result.update({key:Tensor(np.asarray(value)*(1 if key=='input_labels' else 2)) for key,value in hints.items()})
                if outer.hook:outer.hook(result)
                return result
            def post(self,raw,original,reshaped,mask_threshold,binarize):
                outer.assertEqual(original.numpy().tolist(),[[8,16]]);outer.assertEqual(reshaped.numpy().tolist(),[[16,32]])
                outer.assertTrue(binarize)
                if outer.bad=='aligned':return [Tensor(np.zeros((1,1,8,15),dtype=bool))]
                count=raw.shape[2];masks=np.zeros((1,count,8,16),dtype=bool)
                for index in range(count):masks[0,index,:,index:index+4]=True
                return [Tensor(masks)]
        class Model:
            dtype='float32'
            def __call__(self,**kwargs):
                outer.calls.append(kwargs);count=3 if kwargs['multimask_output'] else 1
                raw=np.zeros((1,1,count,4,4),dtype='float32');scores=np.array([[[-.2,1.3,.4][:count]]],dtype='float32')
                if outer.bad=='nan':raw[:]=np.nan
                if outer.bad=='shape':raw=np.zeros((1,count,4,4),dtype='float32')
                if outer.bad=='score':scores[:]=np.inf
                return NS(pred_masks=Tensor(raw),iou_scores=Tensor(scores))
            def to(self,device):return self
            def eval(self):return self
            def parameters(self):return [NS(device='cpu')]
            def buffers(self):return []
        self.backend=SegmentationModel.__new__(SegmentationModel);self.backend.spec=self.spec;self.backend.device='cpu'
        self.backend.multimask_count=3;self.backend.processor=Processor();self.backend.model=Model()
        self.torch_patch=patch.dict('sys.modules',{'torch':NS(inference_mode=nullcontext,
            cuda=NS(is_available=lambda:False),backends=NS(mps=NS(is_available=lambda:False)))})
        self.torch_patch.start();self.addCleanup(self.torch_patch.stop)
    def result(self,**options):
        return self.backend.infer('image-segmentation',{'model':'sam','image':self.image,'box':[1,1,12,7],**options})
    def test_mask_polarity_scores_native_order_and_prompt_frame(self):
        for multimask in (False,True):
            result=SegmentationResult.model_validate(self.result(multimask=multimask));result.verify_pixels()
            self.assertEqual(len(result.masks),3 if multimask else 1)
            self.assertEqual(result.input_rgb_sha256,hashlib.sha256(self.image.tobytes()).hexdigest())
            self.assertEqual(result.prompt.box,[1,1,12,7]);self.assertEqual(self.calls[-1]['input_boxes'].numpy().tolist(),[[[2,2,24,14]]])
            self.assertAlmostEqual(result.masks[0].predicted_iou,-.2)
            if multimask:self.assertGreater(result.masks[1].predicted_iou,1)
            for index,row in enumerate(result.masks):
                self.assertEqual(row.native_mask_token_index,index+int(multimask))
                with Image.open(io.BytesIO(base64.b64decode(row.png_base64))) as image:
                    self.assertEqual(image.size,(16,8));self.assertEqual(image.mode,'RGBA')
                    self.assertEqual(image.getpixel((index,0))[3],0);self.assertEqual(image.getpixel((15,0))[3],255)
                self.assertEqual(row.selected_pixels,32)
        values={'model':'sam','image':self.image,'points':[[2.5,1.25],[12,5]],'point_labels':[1,0]}
        result=self.backend.infer('image-segmentation',values)
        self.assertEqual(result['prompt']['points'],values['points'])
        self.assertEqual(self.calls[-1]['input_points'].numpy().tolist(),[[[[5,2.5],[24,10]]]])
        self.assertEqual(self.calls[-1]['input_labels'].numpy().tolist(),[[[1,0]]])
    def test_prompt_bounds_and_model_process_defaults(self):
        for prompt in ({},{'box':[0,0,2,2],'points':[[1,1]],'point_labels':[1]},
                {'box':[0,0,0,2]},{'points':[[1,1]]},{'points':[[1,1]],'point_labels':[True]},
                {'points':[[1,1]],'point_labels':[2]},{'box':[0,0,float('nan'),2]},{'box':[False,0,2,2]}):
            with self.subTest(prompt=prompt),self.assertRaises(ValueError):SegmentationPrompt.model_validate(prompt)
        for box in ([0,0,17,8],[0,0,16,9]):
            with self.assertRaises(ValueError):self.result(box=box)
        self.assertFalse(self.calls)
        token=_active_settings.set(Settings(segmentation_multimask=True,segmentation_mask_threshold=-.5))
        try:
            self.assertTrue(self.result()['multimask'])
            self.backend.spec=self.spec.model_copy(update={'defaults':{'multimask':False,'mask_threshold':.25}})
            self.assertFalse(self.result()['multimask']);self.assertEqual(self.result()['mask_threshold'],.25)
            self.assertTrue(self.result(multimask=True,mask_threshold=0)['multimask'])
        finally:_active_settings.reset(token)
    def test_bad_tensors_alignment_and_cancellation(self):
        for bad in ('nan','shape','score','aligned'):
            self.bad=bad
            with self.assertRaises(EngineError) as error:self.result()
            self.assertEqual(error.exception.code,'invalid_backend_response')
        self.bad=None;cancel=threading.Event();self.hook=lambda inputs:cancel.set();before=len(self.calls)
        with self.assertRaises(EngineError):self.backend.infer('image-segmentation',{'model':'sam','image':self.image,'box':[0,0,4,4]},cancel=cancel)
        self.assertEqual(len(self.calls),before)
        self.hook=lambda values:values.update(input_boxes=Tensor([[[1,1,2,2]]]))
        with self.assertRaises(EngineError):self.result()
        self.assertEqual(len(self.calls),before)
        self.hook=None
        token=_active_settings.set(Settings(max_tensor_elements=2000))
        try:
            before=len(self.preprocess)
            with self.assertRaises(EngineError):self.result()
            self.assertEqual(len(self.preprocess),before)
        finally:_active_settings.reset(token)
    def test_http_extension_mcp_and_header_rejections(self):
        app=create_app(Config(models={'sam':self.spec}),output_dir=self.root/'out')
        app.state.engine.target.infer=lambda name,task,data,**kwargs:self.backend.infer(task,{k:v for k,v in data.items() if k!='worker'},kwargs.get('cancel'))
        body={'model':'sam','image_base64':base64.b64encode(self.png).decode(),'box':[1,1,12,7]}
        with TestClient(app,base_url='http://127.0.0.1') as client:
            response=client.post('/v1/images/segment',json=body);self.assertEqual(response.status_code,200,response.text)
            result=response.json();self.assertEqual(result['task'],'image-segmentation')
            response=client.post('/v1/extensions/infer',json={**body,'operation':'segmentation','multimask':True})
            self.assertEqual(response.status_code,200,response.text);self.assertEqual(len(response.json()['masks']),3)
            self.assertIn('segmentation',{x['operation'] for x in client.get('/v1/extensions').json()['data']})
            before=len(self.calls)
            for change in ({'box':[0,0,17,8]},{'image_base64':'bad!'},{'points':[[1,1]],'point_labels':[1]}):
                rejected=client.post('/v1/images/segment',json={**body,**change})
                self.assertIn(rejected.status_code,(400,422));self.assertEqual(len(self.calls),before)
            async def invoke():
                reply=await app.state.mcp.call_tool('segment_image',{'payload':{**body,'multimask':False}})
                self.assertFalse(reply.is_error);self.assertEqual(len(reply.structured_content['masks']),1)
            with patch.object(Context,'request_context',new_callable=PropertyMock,return_value=NS(request=None)):asyncio.run(invoke())
            # Typed header/hash validation alone cannot certify pixels; the public
            # serving boundary rejects a processor/backend's nonbinary alpha.
            bad=json.loads(json.dumps(result));image=Image.new('RGBA',(16,8),(0,0,0,128))
            stream=io.BytesIO();image.save(stream,format='PNG');image.close();raw=stream.getvalue()
            bad['masks'][0].update(png_base64=base64.b64encode(raw).decode(),png_sha256=hashlib.sha256(raw).hexdigest())
            app.state.engine.target.infer=lambda *args,**kwargs:bad
            response=client.post('/v1/images/segment',json=body)
            self.assertEqual(response.status_code,502,response.text)
    def test_cancel_during_output_host_transfer_skips_postprocessing(self):
        cancel=threading.Event();original=self.backend.model
        class CancelOnTransfer(Tensor):
            def float(self):return self
            def cpu(self):cancel.set();return self
        def model(**kwargs):
            output=original(**kwargs)
            output.pred_masks=CancelOnTransfer(output.pred_masks.numpy())
            return output
        class CallableModel:
            dtype='float32'
            def __call__(self,**kwargs):return model(**kwargs)
        self.backend.model=CallableModel()
        with patch.object(self.backend.processor.image_processor,'post_process_masks') as post:
            with self.assertRaises(EngineError) as error:
                self.backend.infer('image-segmentation',{'model':'sam','image':self.image,'box':[0,0,4,4]},cancel=cancel)
        self.assertEqual(error.exception.code,'cancelled');self.assertEqual(len(self.calls),1);post.assert_not_called()
    def test_attention_policy_raw_and_private_parsed_aliases(self):
        for selector in (None,'eager','sdpa'):
            validate_attention_selectors({'attn_implementation':selector,'nested':NS(_attn_implementation_internal=selector)})
        validate_attention_selectors(NS(_attn_implementation_autoset=True))
        for selector in ('publisher/custom-attention','flash_attention_2','',False,{'vision':'eager'},['sdpa']):
            for value in ({'attn_implementation':selector},{'nested':[{'__attn_implementation_internal':selector}]},NS(vision_config=NS(_attn_implementation_internal=selector))):
                with self.subTest(selector=selector),self.assertRaises(ValueError):validate_attention_selectors(value)
    def test_decode_uses_shared_deadline_without_model_work(self):
        app=create_app(Config(models={'sam':self.spec}),output_dir=self.root/'timeout',settings=Settings(client_timeout=.01))
        calls=[];app.state.engine.target.infer=lambda *args,**kwargs:calls.append(args)
        from titanos.api_plus import _image
        def delayed(*args,**kwargs):time.sleep(.04);return _image(*args,**kwargs)
        with TestClient(app,base_url='http://127.0.0.1') as client,patch('titanos.api_plus._image',side_effect=delayed):
            response=client.post('/v1/images/segment',json={'model':'sam','image_base64':base64.b64encode(self.png).decode(),'box':[0,0,4,4]})
            self.assertEqual(response.status_code,504,response.text);self.assertFalse(calls)
    def test_safe_builtin_loader_and_nested_override_rejection(self):
        raw={'model_type':'sam','vision_config':{'image_size':32},'mask_decoder_config':{'num_multimask_outputs':3}}
        config_path=self.root/'config.json';config_path.write_text(json.dumps(raw),encoding='utf-8')
        (self.root/'preprocessor_config.json').write_text('{}',encoding='utf-8')
        config=NS(model_type='sam',vision_config=NS(image_size=32,patch_size=16),prompt_encoder_config=NS(image_size=32,patch_size=16),mask_decoder_config=NS(num_multimask_outputs=3))
        self.backend.model.config=config;loads=[]
        def load(path,**kwargs):loads.append(kwargs);return self.backend.model,{}
        vendor=NS(SamConfig=NS(from_pretrained=lambda *args,**kwargs:config),SamModel=NS(from_pretrained=load),
            SamImageProcessor=NS(from_pretrained=lambda *args,**kwargs:self.backend.processor.image_processor),
            SamProcessor=lambda **kwargs:self.backend.processor)
        with patch.dict('sys.modules',{'transformers':vendor}):
            actual=SegmentationModel(self.spec,'cpu',{'torch_dtype':'float32'})
            self.assertEqual(actual.device,'cpu');self.assertTrue(loads[0]['use_safetensors']);self.assertFalse(loads[0]['trust_remote_code'])
            self.assertTrue(loads[0]['local_files_only']);self.assertIs(loads[0]['config'],config)
            token=_active_settings.set(Settings(segmentation_max_masks=1))
            try:
                single=SegmentationModel(self.spec,'cpu',{'torch_dtype':'float32'})
                self.assertEqual(single.multimask_count,3)
                result=single.infer('image-segmentation',{'model':'sam','image':self.image,'box':[0,0,4,4]})
                self.assertEqual(len(result['masks']),1)
                before=(len(self.preprocess),len(self.calls))
                with self.assertRaises(ValueError):single.infer('image-segmentation',{'model':'sam','image':self.image,'box':[0,0,4,4],'multimask':True})
                self.assertEqual((len(self.preprocess),len(self.calls)),before)
            finally:_active_settings.reset(token)
            for override in ({'attn_implementation':'publisher/custom-attention'},
                             {'vision_config':{'_attn_implementation_internal':'publisher/custom-attention'}},
                             {'mask_decoder_config':{'nested':[{'_attn_implementation':'publisher/custom-attention'}]}}):
                loads.clear();config_path.write_text(json.dumps({**raw,**override}),encoding='utf-8')
                with self.assertRaises(ValueError):SegmentationModel(self.spec,'cpu',{})
                self.assertFalse(loads)
            config_path.write_text(json.dumps(raw),encoding='utf-8');loads.clear()
            config.vision_config._attn_implementation_internal='publisher/custom-attention'
            with self.assertRaises(ValueError):SegmentationModel(self.spec,'cpu',{})
            self.assertFalse(loads);del config.vision_config._attn_implementation_internal
            for override in ({'auto_map':{'AutoModel':'external.Code'}},{'quantization_config':{'quant_method':'external'}}):
                loads.clear();config_path.write_text(json.dumps({**raw,'vision_config':override}),encoding='utf-8')
                with self.assertRaises(ValueError):SegmentationModel(self.spec,'cpu',{})
                self.assertFalse(loads)
        for options in ({'config':{}},{'state_dict':{}},{'backbone_kwargs':{}},{'attn_implementation':{'vision':'eager'}},{'low_cpu_mem_usage':{}}):
            with self.assertRaises(ValueError):ModelSpec(path='unused',task='image-segmentation',load_options=options)
        for changes in ({'backend':'diffusers'},{'allow_pickle':True},{'trust_remote_code':True},{'defaults':{'multimask':None}},{'defaults':{'mask_threshold':float('inf')}}):
            with self.assertRaises(ValueError):ModelSpec(path='unused',task='image-segmentation',**changes)


if __name__=='__main__':unittest.main(verbosity=2)
