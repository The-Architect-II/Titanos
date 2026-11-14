"""One-image prompted SAM segmentation, with image-aligned edit masks."""
import base64
import hashlib
import io
import json
from pathlib import Path
import struct
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .engine import EngineError
from .settings import current_settings

Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Coordinate = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Point = Annotated[list[Coordinate], Field(min_length=2, max_length=2)]
Label = Annotated[int, Field(strict=True, ge=0, le=1)]
Digest = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


class SegmentationPrompt(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    box: Annotated[list[Coordinate], Field(min_length=4, max_length=4)] | None = None
    points: list[Point] | None = Field(default=None, min_length=1)
    point_labels: list[Label] | None = Field(default=None, min_length=1)
    multimask: bool | None = Field(default=None, strict=True)
    mask_threshold: Number | None = None

    @model_validator(mode='after')
    def prompts(self):
        if (self.box is None) == (self.points is None):
            raise ValueError('Provide one box OR one point set')
        if self.box is not None:
            if self.point_labels is not None or self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
                raise ValueError('Box requires x0 < x1, y0 < y1 and no point labels')
        elif (self.point_labels is None or len(self.point_labels) != len(self.points)
              or len(self.points) > current_settings().segmentation_max_points):
            raise ValueError('Each point requires a 0/1 label within the configured point limit')
        return self

    def check_geometry(self, width, height):
        if self.box is not None:
            if self.box[2] > width or self.box[3] > height:
                raise ValueError('Box coordinates must lie within the original image boundaries')
        elif any(x >= width or y >= height for x,y in self.points):
            raise ValueError('Points must lie within the original image pixels')


PROMPT_FIELDS = set(SegmentationPrompt.model_fields)


def resolve_prompt(values, defaults):
    settings=current_settings()
    data={key:values[key] for key in PROMPT_FIELDS if values.get(key) is not None}
    for key,fallback in (('multimask',settings.segmentation_multimask),('mask_threshold',settings.segmentation_mask_threshold)):
        data.setdefault(key,defaults.get(key,fallback))
    if data['multimask'] is None or data['mask_threshold'] is None:
        raise ValueError('SAM defaults must resolve to a boolean multimask flag and finite logit threshold')
    return SegmentationPrompt.model_validate(data)


def check_dimensions(width,height,count):
    settings=current_settings()
    if any(type(x) is not int or x < 1 for x in (width,height,count)):
        raise ValueError('Segmentation dimensions/count must be positive integers')
    if count > settings.segmentation_max_masks or width*height > settings.max_input_image_pixels or width*height*count > settings.max_tensor_elements:
        raise ValueError('Segmentation mask count/pixels exceed configured limits')
    # Conservative uncompressed RGBA PNG/base64 budget, checked before model work.
    raw=height*(4*width+1)
    if count*(4*((raw+raw//100+65536+2)//3)+2048) > settings.max_body_bytes:
        raise ValueError('Segmentation response budget exceeds TITANOS_MAX_BODY_BYTES')


class SegmentationMask(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    index: int = Field(strict=True,ge=0)
    native_mask_token_index: int = Field(strict=True,ge=0)
    predicted_iou: Number
    png_base64: str
    png_sha256: Digest
    selected_pixels: int = Field(strict=True,ge=0)


class SegmentationResult(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    model: str = Field(min_length=1,max_length=128)
    task: Literal['image-segmentation']='image-segmentation'
    mask_contract: Literal['alpha-binary-v1']='alpha-binary-v1'
    score_kind: Literal['predicted_iou_uncalibrated']='predicted_iou_uncalibrated'
    calibrated: Literal[False]=False
    width: int = Field(strict=True,gt=0)
    height: int = Field(strict=True,gt=0)
    input_rgb_sha256: Digest
    multimask: bool = Field(strict=True)
    mask_threshold: Number
    prompt: SegmentationPrompt
    masks: list[SegmentationMask] = Field(min_length=1)
    model_output_shape: list[Annotated[int,Field(strict=True,gt=0)]] = Field(min_length=5,max_length=5)
    alignment: Literal['processor_resize_pad_then_post_process_to_original_pixels']='processor_resize_pad_then_post_process_to_original_pixels'
    model_class: str
    processor_class: str
    device: str
    repository: str | None = None
    revision: str | None = None

    @model_validator(mode='after')
    def validate_masks(self):
        check_dimensions(self.width,self.height,len(self.masks))
        self.prompt.check_geometry(self.width,self.height)
        if self.prompt.multimask != self.multimask or self.prompt.mask_threshold != self.mask_threshold:
            raise ValueError('Segmentation result controls differ from the declared prompt')
        if not self.multimask and len(self.masks)!=1:
            raise ValueError('Single-mask mode must return one native mask')
        if self.model_output_shape[:3] != [1,1,len(self.masks)]:
            raise ValueError('Native segmentation output shape/count mismatch')
        if len(self.model_dump_json().encode('utf-8')) > current_settings().max_body_bytes:
            raise ValueError('Segmentation response exceeds TITANOS_MAX_BODY_BYTES')
        for index,mask in enumerate(self.masks):
            if mask.index != index or mask.native_mask_token_index != index+int(self.multimask) or mask.selected_pixels > self.width*self.height:
                raise ValueError('Invalid native mask identity or selected pixel count')
            raw=base64.b64decode(mask.png_base64,validate=True)
            if (hashlib.sha256(raw).hexdigest()!=mask.png_sha256 or raw[:8]!=b'\x89PNG\r\n\x1a\n'
                    or raw[12:16]!=b'IHDR' or len(raw)<33 or raw[24:26]!=bytes((8,6))
                    or struct.unpack('>II',raw[16:24])!=(self.width,self.height)):
                raise ValueError('Mask PNG header, geometry or identity is invalid')
        return self

    def verify_pixels(self):
        # Serving already requires Pillow; the lightweight typed client needs only
        # stdlib header/hash validation, not a new image dependency.
        from PIL import Image
        for mask in self.masks:
            with Image.open(io.BytesIO(base64.b64decode(mask.png_base64))) as image:
                if image.mode!='RGBA' or image.size!=(self.width,self.height) or getattr(image,'n_frames',1)!=1:
                    raise ValueError('Mask must be a single image-aligned RGBA PNG')
                with image.getchannel('A') as alpha:
                    counts=alpha.histogram()
                if sum(counts[1:255]) or counts[0]!=mask.selected_pixels:
                    raise ValueError('Mask alpha/pixel count violates the binary edit-mask contract')


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise EngineError('Segmentation cancelled',499,'cancelled')


class SegmentationModel:
    def __init__(self,spec,device,load_kwargs):
        from transformers import SamConfig, SamModel, SamImageProcessor, SamProcessor
        from .loader_policy import validate_attention_selectors
        self.spec,self.device=spec,device
        files={name:(Path(spec.path)/name).read_bytes() for name in ('config.json','preprocessor_config.json')}
        raw=json.loads(files['config.json'])
        if raw.get('model_type')!='sam':
            raise ValueError('Prompted segmentation requires a builtin SAM checkpoint (model_type sam)')
        def no_external(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key in {'auto_map','custom_pipelines','backbone','backbone_kwargs','use_pretrained_backbone','use_timm_backbone','quantization_config','compression_config'} and item not in (None,False,{}):
                        raise ValueError('SAM configuration cannot resolve external code or backbones')
                    no_external(item)
            elif isinstance(value,list):
                for item in value:no_external(item)
        processor_config=json.loads(files['preprocessor_config.json'])
        no_external(raw);no_external(processor_config)
        validate_attention_selectors(raw);validate_attention_selectors(processor_config)
        config=SamConfig.from_pretrained(spec.path,local_files_only=True,trust_remote_code=False)
        validate_attention_selectors(config)
        self.multimask_count=config.mask_decoder_config.num_multimask_outputs
        if type(self.multimask_count) is not int or self.multimask_count < 1:
            raise ValueError('SAM native alternative count must be a positive integer')
        processor=SamImageProcessor.from_pretrained(spec.path,local_files_only=True,trust_remote_code=False)
        image_size=config.vision_config.image_size
        if (config.prompt_encoder_config.image_size!=image_size
                or config.prompt_encoder_config.patch_size!=config.vision_config.patch_size):
            raise ValueError('SAM vision and prompt encoders must share the declared image/patch coordinate scale')
        if (not processor.do_resize or not processor.do_pad or processor.size!={'longest_edge':image_size}
                or processor.pad_size!={'height':image_size,'width':image_size}):
            raise ValueError('SAM requires its declared resize/pad geometry to match the builtin vision encoder')
        self.processor=SamProcessor(image_processor=processor)
        self.model,info=SamModel.from_pretrained(spec.path,**dict(load_kwargs,config=config,local_files_only=True,
            use_safetensors=True,trust_remote_code=False,output_loading_info=True))
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError('SAM checkpoint has missing or incompatible weights')
        if (any((Path(spec.path)/name).read_bytes()!=value for name,value in files.items())
                or self.model.config.model_type!='sam' or self.model.config.mask_decoder_config.num_multimask_outputs!=self.multimask_count
                or self.model.config.vision_config.image_size!=image_size):
            raise ValueError('SAM checkpoint configuration changed during loading')
        self.model.to(device).eval()
        placement={str(value.device) for value in self.model.parameters()} | {str(value.device) for value in self.model.buffers()}
        if len(placement)!=1:
            raise ValueError('SAM requires a single loaded device')
        self.device=placement.pop()
        if self.device.split(':')[0]!=device.split(':')[0] or (':' in device and self.device!=device):
            raise ValueError('SAM did not load on the requested device')

    def details(self):
        return {'family':'sam','prompts':'one box or one labeled point set','native_multimask_count':self.multimask_count,
            'score_kind':'predicted_iou_uncalibrated','calibrated':False,'automatic_mask_generator':False,
            'model_class':type(self.model).__name__,'processor_class':type(self.processor).__name__,
            'device':self.device,'repository':self.spec.source_repo,'revision':self.spec.source_revision,'qualification':None}

    def infer(self,task,data,cancel=None):
        import numpy as np
        import torch
        from PIL import Image
        if task!='image-segmentation' or set(data)-({'model','image','timeout','location'} | PROMPT_FIELDS):
            raise EngineError('Unsupported prompted segmentation task or controls')
        check_cancel(cancel)
        prompt=resolve_prompt(data,self.spec.defaults);image=data['image']
        if not isinstance(image,Image.Image):
            raise EngineError('Segmentation requires a decoded RGB image')
        count=self.multimask_count if prompt.multimask else 1
        check_dimensions(image.width,image.height,count);prompt.check_geometry(image.width,image.height)
        pad=self.processor.image_processor.pad_size
        if 3*pad['height']*pad['width'] > current_settings().max_tensor_elements:
            raise EngineError('SAM processor tensor exceeds TITANOS_MAX_TENSOR_ELEMENTS',413,'request_too_large')
        rgb=image.convert('RGB')
        try:
            input_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
            hints={'input_boxes':[[prompt.box]]} if prompt.box is not None else {'input_points':[[prompt.points]],'input_labels':[[prompt.point_labels]]}
            inputs=self.processor(images=rgb,return_tensors='pt',**hints)
        finally:rgb.close()
        check_cancel(cancel)
        if not all(key in inputs for key in ('original_sizes','reshaped_input_sizes','pixel_values')):
            raise EngineError('SAM processor omitted required image tensors',502,'invalid_backend_response')
        original=inputs.pop('original_sizes');reshaped=inputs.pop('reshaped_input_sizes')
        def array(value):
            value=value.detach()
            if value.is_floating_point():value=value.float()
            return value.cpu().numpy()
        if array(original).tolist()!=[[image.height,image.width]]:
            raise EngineError('SAM processor lost the original image coordinate frame',502,'invalid_backend_response')
        for value in inputs.values():
            if value.numel()>current_settings().max_tensor_elements or not np.isfinite(array(value)).all():
                raise EngineError('Invalid or oversized SAM processor tensor',502,'invalid_backend_response')
        if tuple(inputs['pixel_values'].shape)!=(1,3,pad['height'],pad['width']):
            raise EngineError('SAM processor returned an invalid image batch',502,'invalid_backend_response')
        resized=array(reshaped)
        if resized.shape!=(1,2) or np.any(resized<=0) or np.any(resized>np.array([[pad['height'],pad['width']]])):
            raise EngineError('SAM processor returned invalid resized dimensions',502,'invalid_backend_response')
        sy,sx=resized[0]/np.array([image.height,image.width])
        expected={'input_boxes':np.array([[prompt.box]])*np.array([sx,sy,sx,sy])} if prompt.box is not None else {
            'input_points':np.array([[prompt.points]])*np.array([sx,sy]),'input_labels':np.array([[prompt.point_labels]])}
        if set(inputs)!={'pixel_values',*expected}:
            raise EngineError('SAM processor changed the prompt field set',502,'invalid_backend_response')
        for key,wanted in expected.items():
            actual=array(inputs[key])
            if actual.shape!=wanted.shape or not np.allclose(actual,wanted,rtol=1e-5,atol=1e-4):
                raise EngineError('SAM processor changed prompt coordinates/labels beyond its declared scale',502,'invalid_backend_response')
        inputs={name:value.to(device=self.device,dtype=self.model.dtype) if value.is_floating_point() else value.to(self.device) for name,value in inputs.items()}
        check_cancel(cancel)
        with torch.inference_mode():
            output=self.model(**inputs,multimask_output=prompt.multimask)
            check_cancel(cancel)
            raw=getattr(output,'pred_masks',None);scores=getattr(output,'iou_scores',None)
            if (raw is None or scores is None or raw.ndim!=5 or tuple(raw.shape[:3])!=(1,1,count) or min(raw.shape[-2:])<1 or not raw.is_floating_point()
                    or tuple(scores.shape)!=(1,1,count) or not scores.is_floating_point()
                    or raw.numel()>current_settings().max_tensor_elements):
                raise EngineError('SAM returned invalid mask or predicted-IoU shapes',502,'invalid_backend_response')
            raw_cpu=raw.detach().float().cpu();score_values=array(scores)
            if not np.isfinite(raw_cpu.numpy()).all() or not np.isfinite(score_values).all():
                raise EngineError('SAM returned nonfinite logits or predicted IoU',502,'invalid_backend_response')
            original_cpu=original.cpu();reshaped_cpu=reshaped.cpu()
            check_cancel(cancel)
            processed=self.processor.image_processor.post_process_masks(raw_cpu,original_cpu,reshaped_cpu,
                mask_threshold=prompt.mask_threshold,binarize=True)
        check_cancel(cancel)
        if not isinstance(processed,list) or len(processed)!=1:
            raise EngineError('SAM postprocessor returned an invalid image batch',502,'invalid_backend_response')
        masks=array(processed[0]);native_scores=score_values[0,0]
        if masks.shape!=(1,count,image.height,image.width) or masks.dtype!=np.bool_:
            raise EngineError('SAM postprocessor returned invalid aligned binary masks',502,'invalid_backend_response')
        encoded=[]
        for index,selected in enumerate(masks[0]):
            check_cancel(cancel)
            rgba=np.zeros((image.height,image.width,4),dtype=np.uint8);rgba[:,:,3]=np.where(selected,0,255)
            with Image.fromarray(rgba) as mask:
                stream=io.BytesIO();mask.save(stream,format='PNG');value=stream.getvalue()
            encoded.append(SegmentationMask(index=index,native_mask_token_index=index+int(prompt.multimask),
                predicted_iou=float(native_scores[index]),png_base64=base64.b64encode(value).decode('ascii'),
                png_sha256=hashlib.sha256(value).hexdigest(),selected_pixels=int(selected.sum())))
        result=SegmentationResult(model=data['model'],width=image.width,height=image.height,input_rgb_sha256=input_hash,
            multimask=prompt.multimask,mask_threshold=prompt.mask_threshold,prompt=prompt,masks=encoded,
            model_output_shape=list(raw.shape),model_class=type(self.model).__name__,processor_class=type(self.processor).__name__,
            device=self.device,repository=self.spec.source_repo,revision=self.spec.source_revision)
        check_cancel(cancel)
        return result.model_dump()
