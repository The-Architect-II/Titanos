"""Builtin InstructPix2Pix editing, preserving the source coordinate frame."""
import json
import logging
import math
from pathlib import Path
import sys

from .engine import EngineError
from .settings import current_settings

PIPELINE = 'StableDiffusionInstructPix2PixPipeline'
DEFAULTS = {'n','steps','num_inference_steps','seed','negative_prompt','guidance_scale','image_guidance_scale','response_format'}


def declared(spec):
    if spec is None or spec.backend!='diffusers' or spec.task!='image':return False
    path=Path(spec.path)/'model_index.json'
    if not path.is_file():return False
    value=json.loads(path.read_text(encoding='utf-8'))
    return isinstance(value,dict) and value.get('_class_name')==PIPELINE


def validate_spec(spec):
    from .loader_policy import validate_attention_selectors, validate_weight_variant
    if spec.trust_remote_code or spec.allow_pickle or spec.parallelism!='single':
        raise ValueError('Instruction editing requires builtin code, safetensors and one device')
    if set(spec.defaults)-DEFAULTS:
        raise ValueError('Unsupported InstructPix2Pix defaults: masks, strength, preservation and resize controls do not apply')
    if set(spec.load_options)-{'variant','low_cpu_mem_usage'}:
        raise ValueError('Instruction-edit load options cannot replace pipeline code, components or safety-checker settings')
    if 'low_cpu_mem_usage' in spec.load_options and type(spec.load_options['low_cpu_mem_usage']) is not bool:
        raise ValueError('low_cpu_mem_usage must be a boolean')
    if 'variant' in spec.load_options:validate_weight_variant(spec.load_options['variant'])
    if 'image_guidance_scale' in spec.defaults:
        try:guidance(spec.defaults['image_guidance_scale'])
        except EngineError as exc:raise ValueError(str(exc)) from exc
    index=json.loads((Path(spec.path)/'model_index.json').read_text(encoding='utf-8'))
    components={'unet':('diffusers',{'UNet2DConditionModel'}),'vae':('diffusers',{'AutoencoderKL'}),
        'text_encoder':('transformers',{'CLIPTextModel'}),'tokenizer':('transformers',{'CLIPTokenizer','CLIPTokenizerFast'}),
        'feature_extractor':('transformers',{'CLIPImageProcessor','CLIPFeatureExtractor'}),
        'safety_checker':('stable_diffusion',{'StableDiffusionSafetyChecker'}),
        'image_encoder':('transformers',{'CLIPVisionModelWithProjection'})}
    for name,(library,classes) in components.items():
        item=index.get(name)
        if name in {'feature_extractor','safety_checker','image_encoder'} and item in (None,[None,None]):continue
        if not isinstance(item,list) or len(item)!=2 or item[0]!=library or not isinstance(item[1],str) or item[1] not in classes:
            raise ValueError('Instruction editing requires builtin component identities: '+name)
        config_path=Path(spec.path)/name/'config.json'
        if config_path.is_file():validate_attention_selectors(json.loads(config_path.read_text(encoding='utf-8')))
    scheduler=index.get('scheduler')
    if (not isinstance(scheduler,list) or len(scheduler)!=2 or scheduler[0]!='diffusers'
            or not isinstance(scheduler[1],str) or not scheduler[1].isidentifier() or not scheduler[1].endswith('Scheduler')):
        raise ValueError('Instruction editing requires a builtin Diffusers scheduler')
    config=json.loads((Path(spec.path)/'unet/config.json').read_text(encoding='utf-8'))
    if config.get('in_channels')!=8:
        raise ValueError('InstructPix2Pix requires its eight-channel instruction-trained UNet')


def guidance(value):
    if type(value) not in (int,float) or not 1 <= value <= sys.float_info.max or not math.isfinite(value):
        raise EngineError('image_guidance_scale must be finite and at least one')
    return value


def capabilities():
    return {'family':'stable-diffusion-instruct-pix2pix','pipeline_class':PIPELINE,
            'instruction_edit':True,'requires_input_image':True,'text_to_image':False,'inpainting':False,
            'strength':False,'resizing':False,'image_guidance_active_when':'guidance_scale > 1',
            'image_guidance_scale_default':current_settings().default_image_guidance_scale,'qualification':None}


def generate(pipeline,spec,task,data,device,cancel=None):
    from PIL import Image
    import torch
    def check():
        if cancel is not None and cancel.is_set():raise EngineError('Instruction edit cancelled',499,'cancelled')
    check()
    if type(pipeline).__name__!=PIPELINE or task!='image-edit':
        raise EngineError('InstructPix2Pix requires an image-edit request with a source image',400,'unsupported_capability')
    if (set(spec.defaults)-DEFAULTS or any(key in data for key in ('mask','mask_image','mask_base64','strength','preserve_unmasked'))):
        raise EngineError('InstructPix2Pix does not support masks, strength, or unmasked-pixel compositing',400,'unsupported_capability')
    allowed={'model','worker','task','prompt','image','size','width','height','n','steps','seed','negative_prompt','guidance_scale','image_guidance_scale','response_format','timeout'}
    if set(data)-allowed:raise EngineError('Unsupported instruction-edit controls')
    image=data.get('image');settings=current_settings()
    if not isinstance(image,Image.Image) or image.mode!='RGB':
        raise EngineError('Instruction editing requires a decoded RGB source image')
    factor=getattr(pipeline,'vae_scale_factor',None)
    if type(factor) is not int or factor<1:
        raise EngineError('Instruction-edit VAE geometry is unavailable',502,'invalid_backend_response')
    if (image.size!=(data['width'],data['height']) or any(size%factor for size in image.size)
            or any(size<32 or size>settings.max_image_dimension for size in image.size)
            or image.width*image.height>min(settings.max_image_pixels,settings.max_input_image_pixels)):
        raise EngineError('Instruction edit preserves source dimensions; prepare an image within policy and divisible by the VAE scale')
    count=data['n'];steps=data['steps']
    if not isinstance(data.get('prompt'),str) or not data['prompt'].strip() or len(data['prompt'])>settings.max_prompt_chars:
        raise EngineError('Instruction editing requires a nonblank prompt within the configured limit')
    if type(count) is not int or not 1<=count<=settings.max_generated_images or type(steps) is not int or not 1<=steps<=settings.max_steps:
        raise EngineError('Instruction-edit count or steps exceed configured limits')
    def value(key,fallback=None):return data[key] if data.get(key) is not None else spec.defaults.get(key,fallback)
    scale=guidance(value('image_guidance_scale',settings.default_image_guidance_scale))
    kwargs={'prompt':data['prompt'],'image':image,'num_inference_steps':steps,'num_images_per_prompt':count,
            'image_guidance_scale':scale,'output_type':'pil','return_dict':True}
    for key in ('negative_prompt','guidance_scale'):
        item=value(key)
        if item is not None:kwargs[key]=item
    seed=value('seed')
    if seed is not None:
        if type(seed) is not int or not 0<=seed<2**32:raise EngineError('Instruction-edit seed must be an unsigned32 integer')
        kwargs['generator']=torch.Generator(device=device if device.startswith('cuda') else 'cpu').manual_seed(seed)
    def step_end(pipe,step,timestep,values):check();return values
    kwargs['callback_on_step_end']=step_end
    outputs=None
    try:
        check()
        try:response=pipeline(**kwargs)
        except Exception:
            check()
            raise
        outputs=getattr(response,'images',None);check()
        if not isinstance(outputs,list) or len(outputs)!=count or any(not isinstance(result,Image.Image) or result.mode!='RGB' or result.size!=image.size for result in outputs):
            raise EngineError('Instruction pipeline returned invalid image count, RGB mode or dimensions',502,'invalid_backend_response')
        return outputs
    except BaseException as primary:
        errors=[]
        for result in outputs if isinstance(outputs,(list,tuple)) else ():
            try:getattr(result,'close',lambda:None)()
            except BaseException as exc:errors.append(str(exc))
        if errors:
            note='Instruction-edit output cleanup failed: '+'; '.join(errors)
            primary.add_note(note);logging.getLogger(__name__).error(note)
        raise
