"""Image-conditioned Diffusers inference; no synthetic frame interpolation.

API images are inline data URLs. Pipeline signatures decide supported controls;
an image-only model never silently pretends to consume a text prompt.
"""
import inspect
import json
import math
import re
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path

from .engine import EngineError


def _wan_shell(path, component_class, dtype):
    import torch
    from accelerate import init_empty_weights
    diffusion = hasattr(component_class, 'load_config')
    config = component_class.load_config(path, local_files_only=True) if diffusion else component_class.config_class.from_pretrained(path, local_files_only=True)
    # UMT5 has no buffers; creating its embedding directly on meta avoids a
    # temporary multi-gigabyte CPU allocation. Other components keep buffers real.
    with torch.device('meta') if component_class.__name__ == 'UMT5EncoderModel' else init_empty_weights():
        component = component_class.from_config(config) if diffusion else component_class(config)
    for parameter in component.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype=dtype)
    return component


def _load_wan_component(path, component_class, dtype, device, storage_dtype=None, component=None):
    """Use official per-tensor reads and strict assignment without shard mmap.

    This opt-in path avoids Windows copy-on-write commitment for an entire
    shard and Transformers' duplicate CUDA allocator warmup. It does not reduce
    the memory required for resident model parameters or change the checkpoint.
    """
    import torch
    from safetensors import safe_open
    if 'backend' not in inspect.signature(safe_open).parameters:
        raise ValueError('safetensors_backend=pread requires safetensors with public pread support')
    diffusion = hasattr(component_class, 'load_config')
    component = component if component is not None else _wan_shell(path, component_class, dtype)
    keep_fp32 = list(getattr(component, '_keep_in_fp32_modules', None) or []) if diffusion or dtype == torch.float16 else []
    if not diffusion and dtype in (torch.float16, torch.bfloat16):
        keep_fp32 += list(getattr(component, '_keep_in_fp32_modules_strict', None) or [])
    keep_pattern = re.compile('|'.join(r'(^|\.)' + re.escape(n) + r'(\.|$)' for n in keep_fp32)) if keep_fp32 else None
    cast_names = set()
    if storage_dtype is not None:
        from diffusers.hooks.layerwise_casting import DEFAULT_SKIP_MODULES_PATTERN
        from diffusers.hooks._common import _GO_LC_SUPPORTED_PYTORCH_LAYERS
        skip = tuple(DEFAULT_SKIP_MODULES_PATTERN) + tuple(getattr(component, '_keep_in_fp32_modules', None) or []) + tuple(getattr(component, '_skip_layerwise_casting_patterns', None) or [])
        for name, module in component.named_modules():
            if isinstance(module, _GO_LC_SUPPORTED_PYTORCH_LAYERS) and not any(re.search(pattern, name) for pattern in skip):
                cast_names.update(name + '.' + key if name else key for key, value in module.named_parameters(recurse=False))
    expected = component.state_dict()
    weights = {}
    ignored = []
    ignore_rules = list(getattr(component, '_keys_to_ignore_on_load_unexpected', None) or [])
    files = sorted(Path(path).glob('*.safetensors'))
    if not files:
        raise ValueError(f'No safetensors weights found in {Path(path).name}')
    for file in files:
        # Safetensors0.8 CUDA get_tensor always allocates pinned host buffers.
        # Reading into ordinary CPU memory and copying the converted tensor
        # synchronously avoids that extra Windows commit/locked-memory cache.
        with safe_open(file, framework='pt', device='cpu', backend='pread') as tensors:
            for name in tensors.keys():
                if name not in expected and any(re.search(rule, name) for rule in ignore_rules):
                    ignored.append(name)
                    continue
                if name in weights or name not in expected:
                    raise ValueError(f'Duplicate or unexpected checkpoint tensor: {name}')
                tensor = tensors.get_tensor(name)
                if tuple(tensor.shape) != tuple(expected[name].shape):
                    raise ValueError(f'Checkpoint tensor shape mismatch: {name}')
                target_dtype = torch.float32 if keep_pattern and keep_pattern.search(name) else dtype
                if name in cast_names:
                    # Match the public Diffusers sequence: checkpoint -> compute
                    # precision -> storage precision, including BF16 rounding.
                    tensor = tensor.to(dtype=dtype)
                    target_dtype = storage_dtype
                tensor = tensor.to(dtype=target_dtype if tensor.is_floating_point() else tensor.dtype)
                weights[name] = tensor.to(device=device, non_blocking=False)
                del tensor
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    # UMT5 declares its shared embedding twice in state_dict but stores it once.
    aliases = {}
    for name, parameter in component.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        present = [name for name in names if name in weights]
        if present:
            for name in names:
                if name not in weights:
                    weights[name] = weights[present[0]]
    component.load_state_dict(weights, strict=True, assign=True)
    if hasattr(component, 'tie_weights'):
        component.tie_weights()
    component.to(device=device).eval()
    if storage_dtype is not None and not getattr(component, '_titanos_layerwise_casting', False):
        component.enable_layerwise_casting(storage_dtype=storage_dtype, compute_dtype=dtype)
        component._titanos_layerwise_casting = True
    if any(parameter.is_meta for parameter in component.parameters()) or any(buffer.is_meta for buffer in component.buffers()):
        raise ValueError('A checkpoint component retained an unmaterialized meta tensor')
    component._titanos_checkpoint_compatibility = {'class': component_class.__name__,
                                                  'official_ignore_rules': ignore_rules,
                                                  'ignored_unexpected_keys': ignored}
    return component


class _StagedWan:
    """One resident component at a time; public Accelerate instance hooks."""
    def __init__(self, path, device, dtypes, classes, storage_dtype=None):
        import torch
        from accelerate.hooks import ModelHook, add_hook_to_module
        self.path, self.device, self.storage_dtype = Path(path), device, storage_dtype
        self.components, self.descriptors, self.active = {}, {}, None
        self.pipeline = None
        self.events = []
        owner = self
        class OnDemand(ModelHook):
            execution_device = torch.device(device)
            def __init__(self, name):
                self.name = name
            def pre_forward(self, module, *args, **kwargs):
                owner.activate(self.name)
                return args, kwargs
        for name, cls in classes.items():
            dtype = dtypes.get(name, dtypes.get('default', torch.float32))
            module = _wan_shell(self.path / name, cls, dtype)
            self.components[name] = module
            self.descriptors[name] = (cls, dtype)
            add_hook_to_module(module, OnDemand(name))

    def release(self):
        import torch
        if self.active is None:
            return
        module = self.components[self.active]
        # Rotary tables and similar small nonpersistent buffers must retain their
        # values; model parameters are reread from the immutable local checkpoint.
        buffers = {name: buffer.to('cpu') for name, buffer in module.named_buffers()}
        module.to('meta')
        for name, buffer in buffers.items():
            parent, _, key = name.rpartition('.')
            module.get_submodule(parent)._buffers[key] = buffer
        if self.pipeline is not None:
            self.pipeline.hf_device_map[self.active] = 'disk'
        self.events.append({'component': self.active, 'state': 'released'})
        self.events[:] = self.events[-64:]
        self.active = None
        if str(self.device).startswith('cuda'):
            torch.cuda.empty_cache()

    def activate(self, name):
        if name == self.active:
            return
        self.release()
        cls, dtype = self.descriptors[name]
        self.active = name
        try:
            _load_wan_component(self.path / name, cls, dtype, self.device,
                                self.storage_dtype if name.startswith('transformer') else None,
                                component=self.components[name])
        except Exception:
            self.release()
            raise
        if self.pipeline is not None:
            self.pipeline.hf_device_map[name] = str(self.device)
        self.events.append({'component': name, 'state': 'loaded'})
        self.events[:] = self.events[-64:]


def load_video_pipeline(path, device, load_kwargs):
    """Return (pipeline, already_dispatched), using bounded-memory CUDA loading.

    Call only for a single-device video model. Set CUDA_VISIBLE_DEVICES before
    process startup when the worker owns a pinned GPU. The caller retains its
    normal CPU/MPS placement/compatibility flow when dispatched is false.
    """
    import torch
    from diffusers import DiffusionPipeline
    pipeline_class = DiffusionPipeline
    options = dict(load_kwargs)
    read_backend = options.pop('safetensors_backend', 'mmap')
    staged = options.pop('stage_components', False)
    storage_name = options.pop('transformer_storage_dtype', None)
    if not isinstance(staged, bool) or storage_name not in {None, 'float8_e4m3fn'}:
        raise ValueError('stage_components must be boolean and transformer_storage_dtype may be float8_e4m3fn')
    if storage_name is not None and not staged:
        raise ValueError('transformer_storage_dtype requires stage_components=true')
    if staged:
        read_backend = 'pread'
    if read_backend not in {'mmap', 'pread'}:
        raise ValueError('safetensors_backend must be mmap or pread')
    if str(device).startswith('cuda'):
        if options.get('device_map') is not None and options['device_map'] != 'cuda':
            raise ValueError('Single-device video loading supports device_map=cuda only')
        options.update(device_map='cuda', low_cpu_mem_usage=True)
        index = json.loads((Path(path) / 'model_index.json').read_text(encoding='utf-8'))
        if index.get('_class_name', '').startswith('Wan') and not isinstance(options.get('torch_dtype'), dict):
            # Wan's own pipeline reference keeps the VAE/image encoder float32.
            options['torch_dtype'] = {'default': options.get('torch_dtype', torch.float32),
                                      'vae': torch.float32, 'image_encoder': torch.float32}
        with torch.cuda.device(device):
            if read_backend == 'pread':
                from diffusers import WanTransformer3DModel, AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
                from transformers import UMT5EncoderModel, CLIPVisionModelWithProjection
                if index.get('_class_name') not in {'WanPipeline', 'WanImageToVideoPipeline'}:
                    raise ValueError('pread video loading currently supports first-party Wan pipelines only')
                # In Diffusers0.35, generic factory signature parsing uses the
                # base class's optional-component list. Use the concrete class
                # so a supplied optional transformer is never silently reloaded.
                pipeline_class = WanPipeline if index['_class_name'] == 'WanPipeline' else WanImageToVideoPipeline
                classes = {'text_encoder': UMT5EncoderModel, 'transformer': WanTransformer3DModel,
                           'transformer_2': WanTransformer3DModel, 'vae': AutoencoderKLWan,
                           'image_encoder': CLIPVisionModelWithProjection}
                included = {}
                for name, cls in classes.items():
                    entry = index.get(name)
                    if entry is None or entry == [None, None]:
                        continue
                    if entry != ['transformers' if name in {'text_encoder', 'image_encoder'} else 'diffusers', cls.__name__]:
                        raise ValueError(f'Unsupported Wan component class for pread: {name}')
                    included[name] = cls
                    dtypes = options.get('torch_dtype', {})
                    dtype = dtypes.get(name, dtypes.get('default', torch.float32)) if isinstance(dtypes, dict) else dtypes
                    if not staged:
                        options[name] = _load_wan_component(Path(path) / name, cls, dtype, device)
                if staged:
                    manager = _StagedWan(path, device, options['torch_dtype'], included,
                                         getattr(torch, storage_name) if storage_name else None)
                    options.update(manager.components)
            pipeline = pipeline_class.from_pretrained(path, **options)
            if staged:
                if any(getattr(pipeline, name) is not module for name, module in manager.components.items()):
                    raise ValueError('Pipeline replaced a staged component; refusing unmanaged model residency')
                manager.pipeline = pipeline
                pipeline.hf_device_map = {name: 'disk' for name in manager.components}
                pipeline._titanos_staging = manager
                pipeline._titanos_loading = {'mode': 'one component resident at a time', 'weights': 'local safetensors pread',
                                              'transfer': 'pageable CPU staging, synchronous device copy',
                                              'transformer_storage_dtype': storage_name or str(options['torch_dtype']['default']),
                                              'transformer_compute_dtype': str(options['torch_dtype']['default']),
                                              'sensitive_layers': 'original Diffusers precision policy'}
                if hasattr(pipeline.vae, 'enable_tiling'):
                    pipeline.vae.enable_tiling()
        return pipeline, True
    if read_backend != 'mmap':
        raise ValueError('pread video loading currently requires a pinned CUDA device')
    return DiffusionPipeline.from_pretrained(path, **options), False


def _scheduler_cancel_supported(pipeline):
    return type(pipeline).__name__ == 'I2VGenXLPipeline' and callable(getattr(getattr(pipeline, 'scheduler', None), 'step', None))


@contextmanager
def _scheduler_cancellation(pipeline, check_cancel):
    """I2VGen-XL exposes its scheduler, but no native step callback argument."""
    scheduler = pipeline.scheduler
    original = scheduler.step
    had_instance_step = 'step' in vars(scheduler)
    original_instance_step = vars(scheduler).get('step')
    @wraps(original)
    def step(*args, **kwargs):
        check_cancel()
        result = original(*args, **kwargs)
        check_cancel()
        return result
    scheduler.step = step
    try:
        yield
    finally:
        if had_instance_step:
            scheduler.step = original_instance_step
        else:
            del scheduler.step


def capabilities(pipeline):
    parameters = inspect.signature(pipeline.__call__).parameters
    result = {'image_conditioning': 'image' in parameters, 'text_conditioning': 'prompt' in parameters,
            'pipeline': type(pipeline).__name__,
            'cancellation': 'callback_on_step_end' in parameters or 'callback' in parameters or _scheduler_cancel_supported(pipeline)}
    if _scheduler_cancel_supported(pipeline):
        result['cancellation_method'] = 'instance-local scheduler step, restored after each request'
    loading = getattr(pipeline, '_titanos_loading', None)
    if loading is not None:
        result['loading'] = dict(loading)
        stage = getattr(pipeline, '_titanos_staging', None)
        if stage is not None:
            result['loading']['checkpoint_compatibility'] = {
                name: {'official_ignore_rules': info['official_ignore_rules'], 'ignored_keys': len(info['ignored_unexpected_keys'])}
                for name, module in stage.components.items()
                if (info := getattr(module, '_titanos_checkpoint_compatibility', None)) and info['ignored_unexpected_keys']}
    return result


def generate(pipeline, data, defaults, device, cancel=None):
    """Generate one video from an inline reference through the loaded pipeline."""
    import torch
    from .backends import image_input
    parameters = inspect.signature(pipeline.__call__).parameters
    if 'image' not in parameters:
        raise EngineError('This video checkpoint does not accept an image; select an image-to-video model',
                          400, 'unsupported_capability')
    reference = data.get('input_reference')
    if not isinstance(reference, str) or not reference.startswith('data:image/'):
        raise EngineError('input_reference must be an inline image data URL')

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise EngineError('Image-to-video cancelled', 499, 'cancelled')
    check_cancel()
    aliases = {'steps': 'num_inference_steps', 'frames': 'num_frames'}
    envelope = {'model', 'worker', 'task', 'size', 'n', 'response_format', 'input_reference', 'seed', 'fps', 'pipeline_options'}
    protected = {'image', 'latents', 'prompt_embeds', 'negative_prompt_embeds', 'image_embeds',
                 'last_image', 'generator', 'callback', 'callback_steps', 'callback_on_step_end',
                 'callback_on_step_end_tensor_inputs', 'output_type', 'return_dict', 'num_videos_per_prompt'}
    kwargs = {}
    for name, value in defaults.items():
        if name in envelope or value is None:
            continue
        target = aliases.get(name, name)
        if target in protected or target not in parameters:
            raise EngineError(f'Unsupported image-to-video model default: {name}', 400, 'unsupported_parameter')
        kwargs[target] = value
    model_tuning = defaults.get('pipeline_options') or {}
    request_tuning = data.get('pipeline_options') or {}
    if not isinstance(model_tuning, dict) or not isinstance(request_tuning, dict):
        raise EngineError('pipeline_options must be an object')
    tuning = {**model_tuning, **request_tuning}
    numeric = {'guidance_scale', 'guidance_scale_2', 'noise_aug_strength', 'min_guidance_scale', 'max_guidance_scale', 'eta'}
    integer = {'motion_bucket_id', 'decode_chunk_size', 'max_sequence_length', 'target_fps', 'clip_skip'}
    for name, value in tuning.items():
        if name not in numeric | integer | {'use_dynamic_cfg'} or name not in parameters:
            raise EngineError(f'Unsupported or service-owned image-to-video option: {name}', 400, 'unsupported_parameter')
        if name == 'use_dynamic_cfg':
            valid = isinstance(value, bool)
        elif name in integer:
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= (0 if name in {'motion_bucket_id', 'clip_skip'} else 1)
        else:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        if not valid:
            raise EngineError(f'Invalid numeric image-to-video option: {name}')
        kwargs[name] = value
    for name in ('width', 'height', 'steps', 'frames', 'prompt', 'negative_prompt', 'guidance_scale',
                 'guidance_scale_2', 'motion_bucket_id', 'noise_aug_strength', 'decode_chunk_size',
                 'min_guidance_scale', 'max_guidance_scale', 'max_sequence_length', 'use_dynamic_cfg'):
        if data.get(name) is None:
            continue
        target = aliases.get(name, name)
        if target not in parameters:
            if name == 'prompt' and not str(data[name]).strip():
                continue
            raise EngineError(f'{type(pipeline).__name__} does not consume {name}', 400, 'unsupported_parameter')
        kwargs[target] = data[name]
    if 'fps' in parameters:
        kwargs['fps'] = data.get('fps', defaults.get('fps', parameters['fps'].default))
    if 'num_videos_per_prompt' in parameters:
        kwargs['num_videos_per_prompt'] = 1
    if 'output_type' in parameters:
        kwargs['output_type'] = 'np'
    if 'return_dict' in parameters:
        kwargs['return_dict'] = True
    seed = data.get('seed') if data.get('seed') is not None else defaults.get('seed')
    if seed is not None:
        kwargs['generator'] = torch.Generator(device=device if str(device).startswith('cuda') else 'cpu').manual_seed(seed)
    cancellation_scope = nullcontext()
    if 'callback_on_step_end' in parameters:
        def on_step(pipe, step, timestep, values):
            check_cancel()
            return values
        kwargs['callback_on_step_end'] = on_step
    elif 'callback' in parameters:
        kwargs.update(callback=lambda *unused: check_cancel(), callback_steps=1)
    elif cancel is not None and _scheduler_cancel_supported(pipeline):
        cancellation_scope = _scheduler_cancellation(pipeline, check_cancel)
    elif cancel is not None:
        raise EngineError('This pipeline cannot honour job cancellation', 400, 'unsupported_capability')
    try:
        image = image_input(reference)
    except ValueError as exc:
        raise EngineError(str(exc)) from exc
    try:
        check_cancel()
        with cancellation_scope:
            frames = pipeline(image=image, **kwargs).frames[0]
        check_cancel()
    finally:
        image.close()
    requested = data.get('frames', kwargs.get('num_frames'))
    if requested is not None:
        if len(frames) < requested:
            raise EngineError('Image-to-video pipeline produced fewer frames than requested')
        frames = frames[:requested]
    if not len(frames):
        raise EngineError('Image-to-video pipeline produced no frames')
    return frames
