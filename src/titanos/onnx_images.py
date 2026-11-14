"""Local SD1 ONNX components, Diffusers scheduling, explicit DirectML placement."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
from collections import Counter

from .engine import EngineError
from .link_policy import is_link_path
from .settings import current_settings

DEFAULTS = {'size', 'n', 'steps', 'seed', 'negative_prompt', 'guidance_scale', 'response_format'}
COMPONENTS = ('text_encoder', 'unet', 'vae_decoder', 'safety_checker')


def validate_registration(spec):
    if (spec.task != 'image' or spec.device not in {'cpu', 'dml'} or spec.parallelism != 'single'
            or spec.dtype != 'auto' or spec.graph or spec.trust_remote_code or spec.allow_pickle or spec.native):
        raise ValueError('ONNX images require one explicit cpu/dml device and a prepared SD1 component directory')
    if set(spec.defaults) - DEFAULTS or set(spec.load_options) - {'device_id', 'adapter_luid', 'profile_dir', 'disable_metacommands'}:
        raise ValueError('Unsupported ONNX image defaults or runtime options')
    options = spec.load_options
    if 'disable_metacommands' in options and type(options['disable_metacommands']) is not bool:
        raise ValueError('disable_metacommands must be a boolean DirectML diagnostic option')
    if spec.device == 'dml':
        if type(options.get('device_id')) is not int or options['device_id'] < 0:
            raise ValueError('DirectML images require an explicit nonnegative integer device_id')
        if not isinstance(options.get('adapter_luid'), str) or not re.fullmatch(r'[0-9a-f]{8}:[0-9a-f]{8}', options['adapter_luid']):
            raise ValueError('DirectML images require the matching lowercase DXGI adapter_luid')
    elif {'device_id', 'adapter_luid', 'disable_metacommands'} & options.keys() or spec.allow_cpu_fallback:
        raise ValueError('CPU images do not accept DirectML selectors/options or CPU fallback')
    if 'profile_dir' in options and (not isinstance(options['profile_dir'], str) or not Path(options['profile_dir']).is_absolute()):
        raise ValueError('profile_dir must be an absolute operator-owned directory outside the model')


def inspect_export(directory):
    root = Path(directory).resolve()
    def file(name):
        path = root / name
        if not path.is_file():
            raise ValueError('Missing or escaping local ONNX export file: ' + name)
        return path
    index = json.loads(file('model_index.json').read_text(encoding='utf-8'))
    if index.get('_class_name') not in {'StableDiffusionOnnxPipeline', 'OnnxStableDiffusionPipeline'}:
        raise ValueError('Only the builtin Diffusers SD1 ONNX export family is supported')
    # Construct only these library classes. Checkpoint strings never select Python code.
    required = {name: ['diffusers', 'OnnxRuntimeModel'] for name in COMPONENTS}
    required.update(tokenizer=['transformers', 'CLIPTokenizer'], scheduler=['diffusers', 'PNDMScheduler'])
    if any(index.get(key) != value for key, value in required.items()):
        raise ValueError('SD1 ONNX export requires local CLIP, PNDM and all declared ONNX components including its checker')
    if index.get('feature_extractor') not in (['transformers', 'CLIPFeatureExtractor'], ['transformers', 'CLIPImageProcessor']):
        raise ValueError('SD1 ONNX checker requires the builtin CLIP image processor')
    scheduler = json.loads(file('scheduler/scheduler_config.json').read_text(encoding='utf-8'))
    if scheduler.get('_class_name') != 'PNDMScheduler':
        raise ValueError('This image family requires its PNDM scheduler configuration')
    for name in COMPONENTS: file(name + '/model.onnx')
    for name in ('tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json',
                 'feature_extractor/preprocessor_config.json'): file(name)
    rows = []
    for path in sorted(root.rglob('*')):
        if path.is_dir() and is_link_path(path):  # symlink or junction, on Python 3.11 too
            raise ValueError('ONNX export component directories cannot be links')
        if path.is_file():
            # Hub snapshots use file symlinks into a sibling blob store. Bind
            # their target bytes without permitting linked component directories.
            with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            rows.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest})
    return {'family': 'sd1-onnx-pndm', 'files': rows,
            'export_digest': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


class _CheckedSession:
    """Cancellation boundaries surround every component's synchronous native call."""
    def __init__(self, session, owner, component): self.session, self.owner, self.component = session, owner, component
    def __getattr__(self, name): return getattr(self.session, name)
    def run(self, *args, **kwargs):
        self.owner.check()
        try:
            result = self.session.run(*args, **kwargs)
        except Exception as exc:
            failure = self.owner.device_failure(exc, self.component)
            if failure is not None:raise failure from exc
            raise
        self.owner.completed_component_calls[self.component] += 1
        self.owner.check()
        return result


class ONNXImageModel:
    def current_adapter(self):
        from .dxgi import adapters
        selected = next((a for a in adapters() if a['device_id'] == self.spec.load_options['device_id']), None)
        if not selected or selected['software'] or selected['luid'] != self.spec.load_options['adapter_luid']:
            raise ValueError('DirectML adapter index/LUID binding differs from current hardware')
        return selected

    def __init__(self, spec):
        validate_registration(spec)
        self.spec, self.device, self.cancel = spec, spec.device, None
        self.sessions, self.model, self.profiles = {}, None, None
        self.runtime_fault = None
        self.component_providers = {}
        self.completed_component_calls = {name: 0 for name in COMPONENTS}
        self.export = inspect_export(spec.path)
        self.adapter = None
        if self.device == 'dml':
            self.adapter = self.current_adapter()
        profile = spec.load_options.get('profile_dir')
        if profile:
            directory = Path(profile).resolve()
            if directory.is_relative_to(Path(spec.path).resolve()):
                raise ValueError('ORT profiles must be outside the immutable model directory')
            directory.mkdir(parents=True, exist_ok=True)
            # Separate every load and component; never overwrite earlier profiles.
            profile = directory / secrets.token_hex(8)
        self.provider_options = {'device_id': spec.load_options['device_id']} if self.device == 'dml' else {}
        if 'disable_metacommands' in spec.load_options:
            # ORT 1.24.4 accepts exactly true/false strings, not boolean integers.
            self.provider_options['disable_metacommands'] = str(spec.load_options['disable_metacommands']).lower()
        from .compiled import onnx_session
        from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel, PNDMScheduler
        from transformers import CLIPTokenizer, CLIPImageProcessor
        try:
            for name in COMPONENTS:
                options = dict(self.provider_options)
                self.sessions[name] = onnx_session(Path(spec.path) / name / 'model.onnx', self.device,
                    spec.allow_cpu_fallback, options, str(profile) + '-' + name if profile else None)
                self.component_providers[name] = self.sessions[name].get_providers()
            parts = {name: OnnxRuntimeModel(_CheckedSession(session, self, name)) for name, session in self.sessions.items()}
            self.model = OnnxStableDiffusionPipeline(**parts, vae_encoder=None,
                tokenizer=CLIPTokenizer.from_pretrained(Path(spec.path) / 'tokenizer', local_files_only=True),
                scheduler=PNDMScheduler.from_pretrained(Path(spec.path) / 'scheduler', local_files_only=True),
                feature_extractor=CLIPImageProcessor.from_pretrained(Path(spec.path) / 'feature_extractor', local_files_only=True),
                requires_safety_checker=True)
            self.model.set_progress_bar_config(disable=True)
            if inspect_export(spec.path) != self.export: raise ValueError('ONNX export changed while loading')
            if self.device == 'dml': self.current_adapter()
        except BaseException as exc:
            failure = self.device_failure(exc, 'load')
            if failure is not None:
                # Engine retains this instance, avoiding native destruction of a
                # lost device while unwinding the constructor.
                failure.failed_runtime = self
                raise failure from exc
            self.close()
            raise
        self.profiling = bool(profile)

    def device_failure(self, exc, component):
        message = str(exc)
        # Narrow to native DML lost-device errors, not validation/OOM or user text.
        code = re.search(r'(?i)\b(?:0x)?(887a000[567])\b', message)
        native = '[ONNXRuntimeError]' in message and ('DmlExecutionProvider' in message or 'DmlCommandRecorder' in message)
        if self.device != 'dml' or not native or code is None:return None
        if self.runtime_fault is None:
            self.runtime_fault = {'code':'device_unavailable', 'hresult':'0x'+code[1].upper(),
                'message':message, 'component':component, 'pid':os.getpid(),
                'device':self.device, 'adapter':self.adapter,
                'scope':'This native runtime context is unusable; global hardware health is not inferred'}
        return EngineError('DirectML device context is unavailable: '+message, 503, 'device_unavailable')

    def check(self):
        if self.runtime_fault is not None:
            raise EngineError('DirectML context remains unavailable; restart its owning process after device re-admission', 503, 'device_unavailable')
        if self.cancel is not None and self.cancel.is_set():
            raise EngineError('ONNX image generation cancelled', 499, 'cancelled')

    def finish_profiles(self):
        if not getattr(self, 'profiling', False) or self.profiles is not None: return
        rows = {}
        for name, session in self.sessions.items():
            path = Path(session.end_profiling())
            raw = path.read_bytes()
            events = json.loads(raw)
            counts = Counter((e.get('args', {}).get('provider'), e.get('args', {}).get('op_name'))
                             for e in events if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
            rows[name] = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                          'operators': [{'provider': p, 'op_name': op, 'events': n} for (p, op), n in sorted(counts.items())]}
        self.profiles = {'scope': 'first attempted inference after this model load; includes session startup', 'components': rows}

    def details(self):
        return {'backend': 'onnx-diffusers', 'device': self.device, 'adapter': self.adapter,
                'family': self.export['family'], 'export_digest': self.export['export_digest'],
                'repository': self.spec.source_repo, 'revision': self.spec.source_revision,
                'component_providers': dict(self.component_providers),
                'runtime_fault': self.runtime_fault,
                'available': self.runtime_fault is None,
                'resident': self.model is not None or bool(self.sessions),
                'requested_provider_options': dict(self.provider_options),
                'cpu_ep_fallback_allowed': self.spec.allow_cpu_fallback,
                'cpu_support': 'NumPy noise/latents, CLIP tokenization/image preprocessing and Torch CPU PNDM scheduling',
                'safety_checker': {'required': True, 'configured': 'safety_checker' in self.sessions,
                                   'completed_native_calls': self.completed_component_calls['safety_checker'],
                                   'scope': 'successfully returned native checker calls since this model load'},
                'image_generation': True, 'image_edit': False,
                'profiles': self.profiles, 'last_generation': getattr(self, 'last_generation', None),
                'qualification': None,
                'placement': 'Explicit selected provider; component operator profiles and owned process GPU activity must establish hardware execution'}

    def infer(self, task, data, emit=None, cancel=None):
        import numpy as np
        from PIL import Image
        self.cancel = cancel
        self.check()
        if task != 'image' or set(data) - (DEFAULTS | {'model', 'worker', 'task', 'prompt', 'width', 'height', 'timeout'}):
            raise EngineError('ONNX SD1 supports image generation only; unsupported controls were not ignored', 400, 'unsupported_capability')
        settings = current_settings()
        def value(key, fallback=None): return data[key] if data.get(key) is not None else self.spec.defaults.get(key, fallback)
        size = value('size', settings.default_image_size)
        if not isinstance(size, str) or not re.fullmatch(r'\d+x\d+', size): raise EngineError('Image size must be WIDTHxHEIGHT')
        width, height = map(int, size.split('x'))
        if any(d < 32 or d % 8 or d > settings.max_image_dimension for d in (width, height)) or width * height > settings.max_image_pixels:
            raise EngineError('Image dimensions exceed configured limits or are not multiples of eight')
        if any(data.get(key, val) != val for key, val in [('width', width), ('height', height)]): raise EngineError('Conflicting image dimensions')
        count, steps = value('n', 1), value('steps', settings.default_steps)
        if type(count) is not int or not 1 <= count <= settings.max_generated_images or type(steps) is not int or not 1 <= steps <= settings.max_steps:
            raise EngineError('Image count or inference steps exceed configured limits')
        prompt, negative, guidance = data.get('prompt'), value('negative_prompt'), value('guidance_scale', 7.5)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > settings.max_prompt_chars: raise EngineError('Invalid image prompt')
        if negative is not None and (not isinstance(negative, str) or len(negative) > settings.max_prompt_chars): raise EngineError('Invalid negative prompt')
        if type(guidance) not in (int, float) or not math.isfinite(guidance) or guidance < 0: raise EngineError('Guidance must be finite and nonnegative')
        if negative and guidance <= 1: raise EngineError('Negative prompts require guidance_scale > 1 for this pipeline')
        prompt_limit = self.model.tokenizer.model_max_length
        if type(prompt_limit) is not int or not 1 <= prompt_limit <= 77:
            raise EngineError('This SD1 pipeline requires a finite CLIP tokenizer context of at most 77 tokens')
        for text in (prompt, negative):
            if text is not None and len(self.model.tokenizer(text, truncation=False)['input_ids']) > prompt_limit:
                raise EngineError('Prompt exceeds this checkpoint tokenizer context; silent truncation is disabled')
        if self.device == 'dml': self.current_adapter()
        seed = value('seed')
        source = 'request' if data.get('seed') is not None else 'model' if seed is not None else 'random'
        if seed is None: seed = secrets.randbits(32)
        if type(seed) is not int or not 0 <= seed < 2**32: raise EngineError('Seed must be an unsigned 32-bit integer')
        self.last_generation = {'seed': seed, 'seed_source': source, 'n': count, 'steps': steps,
                                'width': width, 'height': height, 'completed': False, 'rng': 'numpy.RandomState'}
        images = []
        try:
            result = self.model(prompt=prompt, negative_prompt=negative, width=width, height=height,
                num_images_per_prompt=count, num_inference_steps=steps, guidance_scale=guidance,
                generator=np.random.RandomState(seed), callback=lambda *_: self.check(), callback_steps=1)
            images = result.images
            self.check()
            if not isinstance(images, list) or len(images) != count or any(not isinstance(im, Image.Image) or im.size != (width, height) for im in images):
                raise EngineError('ONNX pipeline returned invalid image count or geometry', 502, 'invalid_backend_response')
            flags = result.nsfw_content_detected
            if not isinstance(flags, (list, tuple)) or len(flags) != count or any(not isinstance(flag, (bool, np.bool_)) for flag in flags):
                raise EngineError('ONNX checker returned invalid per-image flags', 502, 'invalid_backend_response')
            self.last_generation['nsfw_content_detected'] = [bool(flag) for flag in flags]
            self.finish_profiles()
            self.check()
            self.last_generation['completed'] = True
            return images
        except BaseException as primary:
            from .genai_images import _close_images
            if isinstance(images, (list, tuple)): _close_images(images, primary)
            try: self.finish_profiles()
            except Exception as exc: primary.add_note('ORT profile collection failed: ' + str(exc))
            raise
        finally:
            self.cancel = None

    def close(self):
        # Lost-device session destruction can block while holding the GIL. Keep
        # references and ownership truthful; an explicit worker can be stopped
        # by its parent without destroying this context in the frontend.
        if self.runtime_fault is not None:
            raise EngineError('Poisoned DirectML resources remain resident; restart the owning process', 503, 'device_unavailable')
        self.model = None
        self.sessions.clear()
