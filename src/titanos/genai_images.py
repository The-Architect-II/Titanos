"""Native OpenVINO SD/SDXL assets from an explicitly prepared local export."""
import hashlib
import copy
import json
import logging
import math
from pathlib import Path
import re
import sys
import secrets
import threading
import time
import uuid

from .engine import EngineError
from .link_policy import is_link_path
from .settings import current_settings

FAMILIES = {'StableDiffusionPipeline': 'sd', 'StableDiffusionXLPipeline': 'sdxl'}
DEFAULTS = {'size', 'n', 'steps', 'num_inference_steps', 'seed', 'negative_prompt', 'guidance_scale', 'strength', 'response_format', 'preserve_unmasked'}
PROPERTIES = {'INFERENCE_PRECISION_HINT', 'INFERENCE_NUM_THREADS', 'PERFORMANCE_HINT', 'NUM_STREAMS', 'CACHE_DIR'}


def _close_images(images, primary):
    """Attempt every owned close without replacing the error that led here."""
    failures=[]
    for image in images:
        try:
            image.close()
        except BaseException as exc:
            failures.append(f'{type(exc).__name__}: {exc}')
    if failures:
        note='GenAI image cleanup failed: ' + '; '.join(failures)
        primary.add_note(note)
        logging.getLogger(__name__).error(note)


def validate_registration(spec):
    if spec.graph is not None:
        raise ValueError('GenAI images require an exported component directory, not a single graph')
    if not re.fullmatch(r'(?i:auto|CPU|GPU\.\d+)|OV-[A-Za-z0-9_-]+', spec.device):
        raise ValueError('GenAI images require auto, CPU, an explicit GPU.N, or an OV- device identity; NPU static-shape compilation and multiple-device routing are not implemented')
    if set(spec.defaults) - DEFAULTS or set(spec.load_options) - PROPERTIES:
        raise ValueError('Unsupported GenAI image defaults or compile properties; mask bytes and model-family overrides are not model defaults')


def inspect_export(directory):
    root = Path(directory).absolute()
    if '..' in root.parts:
        raise ValueError('Export paths must not contain directory traversal')
    def directory_link(path):
        return is_link_path(path)  # symlink or junction, on Python 3.11 too
    if directory_link(root):
        raise ValueError('Register the export directory itself, not a directory link')
    def file(name):
        value = root / name
        for parent in value.parents:
            if parent == root:
                break
            if directory_link(parent):
                raise ValueError('Export component directories cannot be links')
        if not value.is_file():
            raise ValueError(f'GenAI image export requires a local file: {name}')
        return value
    index = json.loads(file('model_index.json').read_text(encoding='utf-8'))
    if not isinstance(index, dict):
        raise ValueError('model_index.json must contain an exported pipeline object')
    kind = index.get('_class_name')
    if kind not in FAMILIES:
        raise ValueError('Only exported StableDiffusionPipeline and StableDiffusionXLPipeline layouts are supported')
    components = ['text_encoder', 'unet', 'vae_decoder']
    tokenizers = ['tokenizer']
    if FAMILIES[kind] == 'sdxl':
        components.append('text_encoder_2'); tokenizers.append('tokenizer_2')
    edit = (root / 'vae_encoder').exists()
    if edit:
        components.append('vae_encoder')
    required = ['model_index.json', 'scheduler/scheduler_config.json']
    for component in components:
        required += [f'{component}/{name}' for name in ['openvino_model.xml', 'openvino_model.bin', 'config.json']]
    for tokenizer in tokenizers:
        required += [f'{tokenizer}/openvino_tokenizer.{suffix}' for suffix in ['xml', 'bin']]
    for name in required:
        file(name)
    unet = json.loads(file('unet/config.json').read_text(encoding='utf-8'))
    if not isinstance(unet, dict) or unet.get('in_channels') != 4:
        raise ValueError('Only four-channel SD/SDXL denoisers are supported; specialized nine-channel inpainting exports are not admitted')
    # Bind every export file: optional tokenizer/component settings can affect output.
    records = []
    for path in sorted(root.rglob('*')):
        if path.is_dir() and directory_link(path):
            raise ValueError('Export component directories cannot be links')
        if path.is_file():
            # HF snapshot files normally link into sibling blobs. Bind logical
            # names and target bytes, while disallowing linked directories.
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            records.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest})
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'family': FAMILIES[kind], 'export_class': kind, 'image_edit': edit,
            'export_digest': hashlib.sha256(encoded).hexdigest(), 'files': records,
            'native_components': components + tokenizers + ['scheduler'],
            'safety_checker': {'required_by_export': index.get('requires_safety_checker'),
                               'declared_checker': index.get('safety_checker'),
                               'declared_feature_extractor': index.get('feature_extractor'),
                               'invoked_by_adapter': False}}


class GenAIImageModel:
    def __init__(self, spec):
        validate_registration(spec)
        self.spec = spec
        self.export = inspect_export(spec.path)
        import openvino as ov
        import openvino_genai as genai
        from .compiled import compiled_device
        core = ov.Core()
        self.selection = {}
        self.device = compiled_device(spec.backend, spec.device, task='image', core=core, selection=self.selection)
        available = core.available_devices
        # OpenVINO lists its only GPU as GPU, which is the GPU.0 alias.
        if self.device == 'GPU' and 'GPU' in available:
            self.device = 'GPU.0'
        available = {'GPU.0' if value == 'GPU' else value for value in available}
        if not re.fullmatch(r'CPU|GPU\.\d+', self.device) or self.device not in available:
            raise ValueError('The explicitly selected single OpenVINO image device is unavailable')
        options = dict(spec.load_options)
        cache = options.get('CACHE_DIR')
        if cache and Path(cache).resolve().is_relative_to(Path(spec.path).resolve()):
            raise ValueError('Compile caches must be outside the immutable model export')
        if self.device == 'CPU':
            options.setdefault('INFERENCE_NUM_THREADS', current_settings().threads)
            options.setdefault('INFERENCE_PRECISION_HINT', 'f32')
        self.text = self.edit = self.inpaint = None
        try:
            if self.export['image_edit']:
                # Initialize mask helpers in the owner. The inverse conversion
                # can copy null helpers from an image-to-image pipeline.
                self.inpaint = genai.InpaintingPipeline(spec.path,self.device,**options)
                self.edit = genai.Image2ImagePipeline(self.inpaint)
                self.text = genai.Text2ImagePipeline(self.inpaint)
            else:
                self.text = genai.Text2ImagePipeline(spec.path,self.device,**options)
            if inspect_export(spec.path) != self.export:
                raise ValueError('GenAI image export changed while loading')
        except BaseException:
            self.close()
            raise

    def details(self):
        lock=getattr(self,'_generation_lock',None)
        if lock is None:last_generation=None
        else:
            with lock:last_generation=copy.deepcopy(getattr(self,'last_generation',None))
        return {'backend': 'openvino-genai', 'device': self.device, 'family': self.export['family'],
                'configured_device': self.spec.device, 'selection': self.selection,
                'export_class': self.export['export_class'], 'export_digest': self.export['export_digest'],
                'export_files': self.export['files'], 'repository': self.spec.source_repo, 'revision': self.spec.source_revision,
                'native_components': self.export['native_components'], 'safety_checker': self.export['safety_checker'],
                'image_generation': True, 'image_edit': self.edit is not None, 'inpainting': self.inpaint is not None,
                'component_owner': 'InpaintingPipeline' if self.inpaint is not None else 'Text2ImagePipeline',
                'mask_contract': 'alpha-binary-v1', 'pixel_preservation': 'optional server RGB copying, not latent-model fidelity',
                'rng': 'CppStdGenerator', 'qualification': None,
                'last_generation': last_generation,
                'placement': 'One explicitly compiled OpenVINO device; native runtime/model/callback behavior requires qualification'}

    def infer(self, task, data, emit=None, cancel=None):
        import numpy as np
        import openvino as ov
        from PIL import Image
        from .jobs import _native_generation_observer
        # Capture once: native callbacks may execute on another Python thread.
        observer=_native_generation_observer.get()
        lock=self._generation_lock=getattr(self,'_generation_lock',None) or threading.RLock()
        evidence={'schema_version':1,'request_id':uuid.uuid4().hex,'job_id':observer[0] if observer else None,
                  'native_calls':[],'output_disposition':'not_produced','cancellation_observed_at':None,
                  'observability_complete':True,'observer_errors':[]}
        generation=self.last_generation={'completed':False,'native_generation':evidence}
        def publish():
            if observer is None:return
            try:
                with lock:
                    snapshot=copy.deepcopy(evidence)
                    observer[1](snapshot)
            except Exception as exc:
                with lock:
                    evidence['observability_complete']=False
                    if len(evidence['observer_errors'])<8:evidence['observer_errors'].append(type(exc).__name__)
                try:logging.getLogger(__name__).warning('Native generation evidence publication failed: %s',type(exc).__name__)
                except Exception:pass
        def complete():
            with lock:
                generation['completed']=True;evidence['output_disposition']='returned_to_caller'
            publish()
        def check():
            if cancel is not None and cancel.is_set():
                with lock:
                    state=evidence['native_calls'][-1]['state'] if evidence['native_calls'] else None
                    evidence['cancellation_observed_at']='before_native_entry' if state is None else 'after_native_'+state
                    evidence['output_disposition']='discarded_after_native_return' if state=='returned' else 'not_produced'
                publish()
                raise EngineError('GenAI image generation cancelled', 499, 'cancelled')
        publish()
        check()
        if task not in {'image', 'image-edit'} or self.spec.task != 'image':
            raise EngineError('GenAI image adapter supports image and image-edit only')
        allowed = {'model','worker','task','prompt','size','width','height','n','steps','seed','negative_prompt','guidance_scale','strength','image','response_format','timeout','mask_image'}
        if set(data) - allowed:
            raise EngineError('Unsupported GenAI image controls')
        if 'mask_image' in data and task != 'image-edit':
            raise EngineError('A mask requires image-edit')
        if task == 'image-edit' and self.edit is None:
            raise EngineError('This export has no vae_encoder; image editing is unavailable', 400, 'unsupported_capability')
        settings = current_settings()
        defaults = self.spec.defaults
        def value(key, fallback=None):
            return data[key] if data.get(key) is not None else defaults.get(key, fallback)
        size = value('size', settings.default_image_size)
        if not isinstance(size, str) or not re.fullmatch(r'\d+x\d+', size):
            raise EngineError('Image size must be WIDTHxHEIGHT')
        width, height = map(int, size.split('x'))
        if any(d < 32 or d % 8 or d > settings.max_image_dimension for d in (width, height)) or width * height > settings.max_image_pixels:
            raise EngineError('Image dimensions exceed the configured policy or are not multiples of eight')
        if ('width' in data and data['width'] != width) or ('height' in data and data['height'] != height):
            raise EngineError('Conflicting image dimensions')
        count = value('n', 1); steps = value('steps', defaults.get('num_inference_steps', settings.default_steps))
        if type(count) is not int or not 1 <= count <= settings.max_generated_images or type(steps) is not int or not 1 <= steps <= settings.max_steps:
            raise EngineError('Image count or inference steps exceed the configured policy')
        prompt = data.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > settings.max_prompt_chars:
            raise EngineError('A nonblank prompt within the configured limit is required')
        kwargs = {'width': width, 'height': height, 'num_images_per_prompt': count, 'num_inference_steps': steps}
        for name in ['negative_prompt', 'guidance_scale']:
            item = value(name)
            if item is not None:
                if name == 'negative_prompt' and (not isinstance(item, str) or len(item) > settings.max_prompt_chars):
                    raise EngineError('Invalid negative prompt')
                if name == 'guidance_scale' and (type(item) not in (int, float) or item < 0 or item > sys.float_info.max or not math.isfinite(item)):
                    raise EngineError('Guidance scale must be finite and nonnegative')
                kwargs[name] = item
        seed = value('seed')
        seed_source = 'request' if data.get('seed') is not None else 'model' if defaults.get('seed') is not None else 'random'
        if seed is None:
            seed = secrets.randbits(32)
        if type(seed) is not int or not 0 <= seed <= 2**32-1:
            raise EngineError('Seed must be an unsigned 32-bit integer')
        kwargs['rng_seed'] = seed  # Native C++ RNG; no TorchGenerator dependency.
        with lock:generation.update(seed=seed,seed_source=seed_source,width=width,height=height,n=count,steps=steps)
        def generate(pipeline,*args,**options):
            with lock:
                call={'index':len(evidence['native_calls']),'state':'entered','entered_monotonic':time.monotonic(),
                      'callback_count':0,'first_callback':None,'last_callback':None,'first_cancel_callback':None,
                      'stop_requested':False,'callbacks_after_stop':0,'native_error_type':None}
                evidence['native_calls'].append(call)
            publish()
            def callback(step,total,latent):
                stop=bool(cancel is not None and cancel.is_set())
                try:step_value,total_value=int(step),int(total)
                except (ValueError,TypeError,OverflowError):step_value=total_value=None
                with lock:
                    call['callback_count']+=1
                    sample={'step':step_value,'total':total_value,'callback_index':call['callback_count']}
                    if call['stop_requested']:call['callbacks_after_stop']+=1
                    first=call['first_callback'] is None
                    if first:call['first_callback']=sample
                    call['last_callback']=sample
                    first_stop=stop and not call['stop_requested']
                    if first_stop:call['first_cancel_callback']=sample
                    call['stop_requested']=call['stop_requested'] or stop
                if first or first_stop:publish()
                return stop
            try:result=pipeline.generate(*args,**options,callback=callback)
            except BaseException as exc:
                with lock:call.update(state='raised',finished_monotonic=time.monotonic(),native_error_type=type(exc).__name__)
                publish();raise
            with lock:call.update(state='returned',finished_monotonic=time.monotonic())
            publish();return result
        def copy_output(tensor, expected_count):
            check()  # Never accept a stopped/partial native tensor.
            pixels = tensor.data
            if not isinstance(pixels,np.ndarray) or pixels.dtype != np.uint8 or pixels.shape != (expected_count,height,width,3):
                raise EngineError('GenAI returned an invalid image tensor; expected uint8 NHWC with the requested batch and dimensions',502,'invalid_backend_response')
            copied=[]
            try:
                for row in pixels:
                    check()
                    with Image.fromarray(row) as image:
                        copied.append(image.copy())
                check()
                return copied
            except BaseException as primary:
                _close_images(copied,primary)
                raise
        if task == 'image-edit':
            strength = value('strength', settings.default_edit_strength)
            if type(strength) not in (int, float) or not 0 < strength <= 1 or not math.isfinite(strength):
                raise EngineError('Edit strength must be finite, greater than zero and at most one')
            source = data.get('image')
            if not isinstance(source, Image.Image) or source.width * source.height > settings.max_input_image_pixels:
                raise EngineError('Image editing requires a decoded image within the input pixel limit')
            mask_pixels = None
            if 'mask_image' in data:
                from .image_mask import validate
                validate(data['mask_image'],source,(width,height))
                # Explicit RGB representation, with equal channels: white edit / black keep.
                with data['mask_image'].convert('RGB') as rgb_mask:
                    mask_pixels = np.array(rgb_mask,dtype=np.uint8,copy=True)[None]
                if self.inpaint is None:
                    raise EngineError('This GenAI model has no initialized inpainting owner',400,'unsupported_capability')
            with source.convert('RGB') as rgb:
                pixels = np.array(rgb, dtype=np.uint8, copy=True)[None]
            check()
            if mask_pixels is not None:
                # The non-specialized four-channel latent blender supports one
                # image per native call. Keep the requested count via serial calls.
                seeds=[(seed+index) % 2**32 for index in range(count)]
                with lock:generation.update(native_batch_size=1,rng_seeds=seeds)
                images=[]
                try:
                    for image_seed in seeds:
                        check()
                        try:
                            tensor=generate(self.inpaint,prompt,ov.Tensor(pixels),ov.Tensor(mask_pixels),strength=strength,
                                **{**kwargs,'num_images_per_prompt':1,'rng_seed':image_seed})
                        except Exception:
                            check()
                            raise
                        images.extend(copy_output(tensor,1))
                    check()
                    complete()
                    return images
                except BaseException as primary:
                    _close_images(images,primary)
                    raise
            try:
                tensor = generate(self.edit,prompt,ov.Tensor(pixels),strength=strength,**kwargs)
            except Exception:
                check()
                raise
        else:
            try:
                tensor = generate(self.text,prompt, strength=1.0, **kwargs)
            except Exception:
                check()
                raise
        images = copy_output(tensor,count)
        try:
            check()
            complete()
            return images
        except BaseException as primary:
            _close_images(images,primary)
            raise

    def close(self):
        self.text = self.edit = self.inpaint = None
