"""Direct, lazy loaders for established PyTorch inference libraries."""

import base64
import io
import threading
from pathlib import Path

from .config import ModelSpec
from .settings import current_settings as load_settings


def require_cuda_group(torch):
    """Observe the current child runtime, never infer identity from host indices."""
    from .devices import normalized_uuid, require_homogeneous
    hip = bool(torch.version.hip)
    rows = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        identity = normalized_uuid(getattr(props, 'uuid', None))
        if not identity:
            raise ValueError('Multi-GPU execution requires an observed physical UUID for every visible device')
        architecture = getattr(props, 'gcnArchName', None) if hip else '.'.join(map(str, torch.cuda.get_device_capability(i)))
        rows.append({'uuid': ('HIP-' if hip else 'GPU-') + identity, 'local_device': 'cuda:' + str(i),
                     'runtime_vendor': 'rocm' if hip else 'cuda', 'architecture': architecture})
    if len(rows) < 2:
        raise ValueError('Multi-GPU execution requires at least two visible assigned GPUs')
    require_homogeneous(rows)
    return rows


def verify_parameter_placement(owner, expected):
    """Reject a loader that silently leaves requested accelerator weights on CPU."""
    components = getattr(owner, 'components', None)
    components = components if isinstance(components, dict) else {'model': owner}
    actual = set()
    for component in components.values():
        parameters = getattr(component, 'parameters', None)
        if callable(parameters):
            actual.update(str(p.device) for p in parameters())
    # MPS reports its one physical device as either mps or mps:0.
    canonical = lambda name: 'mps' if name == 'mps:0' else name
    actual = {canonical(name) for name in actual}
    expected = {canonical(name) for name in expected}
    if not actual or actual != expected:
        raise ValueError(f'Loaded parameter placement {sorted(actual)} differs from requested devices {sorted(expected)}; no fallback was accepted')
    return sorted(actual)


def image_input(url: str, size_check=None):
    from PIL import Image
    from .settings import current_settings as load_settings
    settings = load_settings()
    if not url.startswith("data:image/") or ";base64," not in url:
        raise ValueError("Images must be inline base64 data URLs; remote URLs are not fetched")
    raw = base64.b64decode(url.split(";base64,", 1)[1], validate=True)
    if len(raw) > settings.max_input_image_bytes:
        raise ValueError("Image exceeds TITANOS_MAX_INPUT_IMAGE_BYTES")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.width * im.height > settings.max_input_image_pixels:
                raise ValueError("Image exceeds TITANOS_MAX_INPUT_IMAGE_PIXELS")
            if size_check is not None:
                size_check(im.width, im.height)
            return im.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or oversized image") from exc


def _mps_cogvideo_positions(module, height, width, frames, device=None):
    """Diffusers' sinusoidal helper needs float64; MPS model tensors do not."""
    import torch
    from diffusers.models.embeddings import CogVideoXPatchEmbed
    positions = CogVideoXPatchEmbed._get_positional_embeddings(module, height, width, frames, device="cpu")
    module._titanos_position_calls += 1
    return positions.to(dtype=torch.float32).to(device=device or "cpu")


def _prepare_mps_cogvideo(pipeline):
    from types import MethodType
    import torch
    from diffusers.models.embeddings import CogVideoXPatchEmbed
    patched = []
    for component in pipeline.components.values():
        if not isinstance(component, torch.nn.Module):
            continue
        for module in component.modules():
            if isinstance(module, CogVideoXPatchEmbed) and hasattr(module, "pos_embedding"):
                # Both the cached buffer and dynamic size/frame path need handling.
                # Only this loaded instance changes; Diffusers' class stays intact.
                module.pos_embedding = module.pos_embedding.to(dtype=torch.float32)
                module._titanos_position_calls = 0
                module._get_positional_embeddings = MethodType(_mps_cogvideo_positions, module)
                patched.append(module)
    return patched


class LoadedModel:
    def __init__(self, spec: ModelSpec):
        import torch
        self.spec = spec
        self.embedding_provenance = None
        self._mps_cogvideo_modules = []
        from .hardware import select_device
        self.device = select_device(spec.device)
        fast_half = self.device.startswith("cuda") and (torch.version.hip or torch.cuda.get_device_capability(self.device)[0] >= 7)
        if spec.parallelism == 'model_sharding' and self.device.startswith('cuda') and not torch.version.hip:
            fast_half = all(torch.cuda.get_device_capability(i)[0] >= 7 for i in range(torch.cuda.device_count()))
        dtype = spec.dtype if spec.dtype != "auto" else ("float16" if fast_half else "float32")
        kw = dict(spec.load_options, local_files_only=True, torch_dtype=getattr(torch, dtype),
                  use_safetensors=None if spec.allow_pickle else True)
        self.sharded = spec.parallelism == "model_sharding"
        self.tensor_parallel = spec.parallelism == "tensor_parallel"
        self.placement_devices = None
        if self.tensor_parallel:
            if not torch.distributed.is_initialized():
                raise ValueError("Tensor parallel model must run inside its distributed rank group")
            mesh = torch.distributed.init_device_mesh(self.device.split(':')[0], (torch.distributed.get_world_size(),))
            kw.update(tp_plan="auto", device_mesh=mesh)
        if self.sharded:
            if not self.device.startswith("cuda") or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise ValueError("Model sharding requires at least two visible CUDA/ROCm GPUs")
            self.group_devices = require_cuda_group(torch)
            kw.update(device_map="balanced", max_memory={i: int(torch.cuda.mem_get_info(i)[0] * 0.85) for i in range(torch.cuda.device_count())})
        path = spec.path
        if not Path(path).is_dir():
            raise ValueError(f"Model directory does not exist: {path}. Download or add a local checkpoint first.")
        if spec.task == 'image-segmentation':
            from .segmentation import SegmentationModel
            self.segmentation = SegmentationModel(spec,self.device,kw)
            self.model,self.processor,self.device = self.segmentation.model,self.segmentation.processor,self.segmentation.device
        elif spec.task == 'depth-estimation':
            from .depth import DepthModel
            self.depth = DepthModel(spec, self.device, kw)
            self.model, self.processor, self.device = self.depth.model, self.depth.processor, self.depth.device
        elif spec.task == 'pair-classification':
            from .pair_classifier import PairClassifier
            self.pair_classifier = PairClassifier(spec, self.device, kw)
            self.model, self.processor, self.device = self.pair_classifier.model, self.pair_classifier.processor, self.pair_classifier.device
        elif spec.task == 'rerank':
            from .rerank import Reranker
            self.reranker = Reranker(spec, self.device, kw)
            self.model, self.processor, self.device = self.reranker.model, self.reranker.processor, self.reranker.device
        elif spec.task == 'sentiment':
            from .sentiment import SentimentClassifier
            self.sentiment = SentimentClassifier(spec, self.device, kw)
            self.model, self.processor = self.sentiment.model, self.sentiment.processor
            self.device = self.sentiment.device
        elif spec.task in {"translation", "image-to-text", "object-detection"}:
            from .extra_backends import ExtraBackend
            self.extra = ExtraBackend(spec, self.device, kw)
            self.model, self.processor = self.extra.model, self.extra.processor
        elif spec.backend == "diffusers":
            from diffusers import DiffusionPipeline
            from .instruction_edit import declared,validate_spec,PIPELINE
            dispatched=False
            if declared(spec):
                validate_spec(spec)
                from diffusers import StableDiffusionInstructPix2PixPipeline
                self.model=StableDiffusionInstructPix2PixPipeline.from_pretrained(path,**kw)
                if type(self.model).__name__!=PIPELINE:
                    raise ValueError('Loaded pipeline does not match the declared builtin instruction-edit family')
            elif spec.task=="video" and not self.sharded:
                from .i2v import load_video_pipeline
                self.model,dispatched=load_video_pipeline(path,self.device,kw)
            else:
                self.model=DiffusionPipeline.from_pretrained(path,**kw)
            if self.device == "mps":
                self._mps_cogvideo_modules = _prepare_mps_cogvideo(self.model)
            if not dispatched and not self.sharded and not self.tensor_parallel:
                self.model.to(self.device)
            self.model.set_progress_bar_config(disable=True)
        elif spec.backend == "xtts":
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            cfg = XttsConfig()
            cfg.load_json(str(Path(path) / "config.json"))
            self.model = Xtts.init_from_config(cfg)
            self.model.load_checkpoint(cfg, checkpoint_dir=path, eval=True)
            self.model.to(self.device)
        else:
            from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, pipeline
            kw["trust_remote_code"] = spec.trust_remote_code
            pkw = {"local_files_only": True, "trust_remote_code": spec.trust_remote_code}
            if spec.task in {"chat", "embeddings"}:
                cls = AutoModelForCausalLM if spec.task == "chat" else AutoModel
                before = self.embedding_fingerprint(path) if spec.task == 'embeddings' else None
                self.model = cls.from_pretrained(path, **kw).eval()
                if not self.sharded and not self.tensor_parallel:
                    self.model.to(self.device)
                self.processor = AutoTokenizer.from_pretrained(path, **pkw)
                if spec.task == 'embeddings':
                    after = self.embedding_fingerprint(path)
                    if before is not None and before != after:
                        raise ValueError('Embedding checkpoint changed while loading')
                    self.embedding_provenance = {'repository':spec.source_repo,'revision':spec.source_revision,
                        'checkpoint_digest':after,'backend':'transformers','device':self.device,
                        'pooling':'attention_mask_mean','normalization':'l2'}
            elif spec.task == "vision":
                self.model = AutoModelForImageTextToText.from_pretrained(path, **kw).eval()
                if not self.sharded and not self.tensor_parallel:
                    self.model.to(self.device)
                self.processor = AutoProcessor.from_pretrained(path, **pkw)
            else:
                task = "automatic-speech-recognition" if spec.task == "transcription" else "text-to-speech"
                kw.pop("trust_remote_code")
                self.model = pipeline(task, model=path, device=None if self.sharded else self.device, trust_remote_code=spec.trust_remote_code, model_kwargs=kw)

        if self.sharded:
            owner = self.model.model if hasattr(self.model, "task") else self.model
            device_map = getattr(owner, "hf_device_map", None)
            used = {str(v).removeprefix("cuda:") for v in (device_map or {}).values()}
            expected = {str(i) for i in range(torch.cuda.device_count())}
            if used != expected:
                raise ValueError(f"Sharding did not place the model on exactly the assigned GPUs: {device_map}")
            if spec.task in {"chat", "vision", "embeddings"}:
                self.device = str(owner.get_input_embeddings().weight.device)
            elif spec.backend == 'diffusers':
                self.device = str(self.model._execution_device)
            self.placement_devices = verify_parameter_placement(owner, {'cuda:' + str(i) for i in range(torch.cuda.device_count())})
        elif not (spec.backend == 'diffusers' and dispatched):
            owner = self.model.model if hasattr(self.model, 'task') else self.model
            self.placement_devices = verify_parameter_placement(owner, {self.device})
        if self.embedding_provenance is not None:
            devices = {str(parameter.device) for parameter in self.model.parameters()}
            self.embedding_provenance['device'] = next(iter(devices)) if len(devices)==1 else self.device

    def details(self):
        import torch
        from .instruction_edit import PIPELINE,capabilities as instruction_capabilities
        from .tensor_storage import describe as tensor_storage
        owner = self.model.model if hasattr(self.model, "task") else self.model
        video = {}
        if self.spec.task == "video" and self.spec.backend == "diffusers":
            from .i2v import capabilities
            video = capabilities(self.model)
        return {"backend": self.spec.backend, "device": self.device, "parallelism": self.spec.parallelism,
                "tensor_storage": tensor_storage(owner),
                "video_capabilities": video,
                "instruction_edit": instruction_capabilities() if type(self.model).__name__==PIPELINE else None,
                "depth": self.depth.details() if hasattr(self, "depth") else None,
                "segmentation": self.segmentation.details() if hasattr(self,"segmentation") else None,
                "sentiment": self.sentiment.details() if hasattr(self, 'sentiment') else None,
                "rerank": self.reranker.details() if hasattr(self, 'reranker') else None,
                "pair_classification": self.pair_classifier.details() if hasattr(self, 'pair_classifier') else None,
                "embedding": getattr(self, 'embedding_provenance', None),
                "device_map": getattr(owner, "hf_device_map", None),
                "tp_plan": getattr(owner, "_tp_plan", None) if self.tensor_parallel else None,
                "configured_device": self.spec.device,
                "parameter_devices": getattr(self, 'placement_devices', None),
                "execution_backend": 'torch.distributed' if self.tensor_parallel else 'pytorch',
                "group_devices": getattr(self, 'group_devices', None),
                "kernel_execution_verified": False,
                "visible_gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                                  "capability": list(torch.cuda.get_device_capability(i)),
                                  "uuid": str(getattr(torch.cuda.get_device_properties(i), "uuid", "unavailable"))}
                                 for i in range(torch.cuda.device_count())] if self.device.startswith("cuda") else [],
                "partition": "pipeline components" if self.sharded and self.spec.backend == 'diffusers' else "model modules" if self.sharded else None,
                "compatibility": [{"component": "CogVideoXPatchEmbed", "operation": "sinusoidal positional embeddings",
                                   "calculation_device": "cpu", "output_device": "mps", "output_dtype": "float32",
                                   "model_device": self.device, "dynamic_calls": module._titanos_position_calls}
                                  for module in self._mps_cogvideo_modules],
                "placement": "loaded device map; use profiling to establish execution"}

    @staticmethod
    def embedding_fingerprint(path):
        # Preserve existing legacy/symlink layouts: provenance may be unavailable,
        # but those public embedding modes are not newly forbidden by this field.
        from .sentiment import checkpoint_fingerprint
        try:
            return checkpoint_fingerprint(path)
        except (ValueError, OSError):
            return None

    def infer(self, task: str, data: dict, emit=None, cancel=None):
        import torch
        with torch.inference_mode():
            if hasattr(self,'segmentation'):
                return self.segmentation.infer(task,{key:value for key,value in data.items() if key!='worker'},cancel)
            if hasattr(self, 'depth'):
                return self.depth.infer(task, {key: value for key, value in data.items() if key != 'worker'}, cancel)
            if hasattr(self, 'pair_classifier'):
                return self.pair_classifier.infer(task, {key:value for key,value in data.items() if key != 'worker'}, cancel)
            if hasattr(self, 'reranker'):
                return self.reranker.infer(task, {key: value for key, value in data.items() if key != 'worker'}, cancel)
            if hasattr(self, 'sentiment'):
                return self.sentiment.infer(task, {key: value for key, value in data.items() if key != 'worker'}, cancel)
            if hasattr(self, "extra"):
                return self.extra.infer(task, data, cancel)
            if task in {"chat", "completion"}:
                return self.chat(data, emit, cancel or threading.Event())
            if task == "embeddings":
                return self.embeddings(data, cancel)
            if task == "transcription":
                return self.transcribe(data, cancel)
            if task == "speech":
                return self.speak(data)
            return self.diffuse(task, data, cancel)

    def chat(self, data, emit, cancel):
        import torch
        from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
        messages = data.get("messages")
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if "prompt" in data:
            inputs = tokenizer(data["prompt"], return_tensors="pt")
        elif self.spec.task == "vision":
            images, normalized = [], []
            for msg in messages:
                parts = []
                content = msg["content"]
                for part in ([{"type": "text", "text": content}] if isinstance(content, str) else content):
                    if part["type"] == "image_url":
                        images.append(image_input(part["image_url"]["url"]))
                        parts.append({"type": "image"})
                    else:
                        parts.append(part)
                normalized.append({"role": msg["role"], "content": parts})
            prompt = self.processor.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=images or None, return_tensors="pt")
        else:
            for msg in messages:
                if not isinstance(msg["content"], str):
                    if any(p["type"] != "text" for p in msg["content"]):
                        raise ValueError("This model is text-only; select a vision model for images")
                    msg["content"] = "\n".join(p["text"] for p in msg["content"])
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.device)
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                inputs[key] = value.to(dtype=self.model.dtype)
        input_count = int(inputs["input_ids"].shape[-1])
        limit = data.get("max_tokens", 256)
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context and input_count + limit > context:
            raise ValueError(f"Prompt plus max_tokens exceeds this model's {context}-token context")
        stop = data.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        pending = ""
        stopped = False

        def push(text, final=False):
            nonlocal pending, stopped
            if stopped:
                return
            pending += text
            cuts = [pending.find(s) for s in stop if s in pending]
            if cuts:
                pending = pending[:min(cuts)]
                stopped = True
                final = True
            # Hold a possible prefix of a stop string across streamer chunks.
            hold = max((i for s in stop for i in range(1, min(len(s), len(pending)) + 1)
                        if pending.endswith(s[:i])), default=0) if not final else 0
            ready = pending[:-hold] if hold else pending
            pending = pending[-hold:] if hold else ""
            if emit and ready:
                emit(ready)

        class Streamer(TextStreamer):
            def on_finalized_text(self, text, stream_end=False):
                push(text, stream_end)

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                stop_now = cancel.is_set() or stopped
                if self_outer.tensor_parallel:
                    flag = torch.tensor(int(stop_now), device=self_outer.device)
                    torch.distributed.broadcast(flag, src=0)
                    stop_now = bool(flag.item())
                return stop_now

        self_outer = self
        temperature = data.get("temperature", 0.7);from .openai_params import transformers_bias
        defaults={k:v for k,v in self.spec.defaults.items() if k not in {"max_tokens","max_completion_tokens","max_output_tokens","temperature","top_p","repeat_penalty","stop","seed","n","stream","stream_options","min_score","min_margin","sentiment_prompt_version","sentiment_decoding"}}
        kw = dict(defaults, max_new_tokens=limit, do_sample=temperature > 0, **transformers_bias(data),
                  pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                  stopping_criteria=StoppingCriteriaList([Stop()]))
        for key in ("top_k","min_p","repeat_penalty"):
            if data.get(key) is not None:kw["repetition_penalty" if key=="repeat_penalty" else key]=data[key]
        if temperature > 0:
            kw.update(temperature=temperature, top_p=data.get("top_p", 1.0))
        # Streamer also detects stop strings for non-streaming requests.
        if emit or stop:
            kw["streamer"] = Streamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        if data.get("seed") is not None:
            torch.manual_seed(data["seed"])
        if data.get('_sentiment_label_choices'):
            from .sentiment_contract import label_prefix_allowed_tokens
            eos = getattr(self.model.generation_config, 'eos_token_id', None)
            kw['prefix_allowed_tokens_fn'] = label_prefix_allowed_tokens(tokenizer, input_count, eos if eos is not None else tokenizer.eos_token_id)
        elif data.get('_sentiment_facet_choices'):
            from .sentiment_contract import facet_prefix_allowed_tokens
            eos = getattr(self.model.generation_config, 'eos_token_id', None)
            kw['prefix_allowed_tokens_fn'] = facet_prefix_allowed_tokens(tokenizer, input_count, eos if eos is not None else tokenizer.eos_token_id)
        output = self.model.generate(**inputs, **kw)[0][input_count:]
        text = tokenizer.decode(output, skip_special_tokens=True)
        for s in stop:
            text = text.split(s, 1)[0]
        return {"text": text, "finish_reason": "length" if len(output) >= limit and not stopped else "stop",
                "usage": {"prompt_tokens": input_count, "completion_tokens": len(output), "total_tokens": input_count + len(output)},
                'provenance': {'repository': getattr(self.spec, 'source_repo', None), 'revision': getattr(self.spec, 'source_revision', None),
                               'backend': getattr(self.spec, 'backend', 'transformers'), 'device': self.device}}

    def embeddings(self, data, cancel=None):
        import torch
        from .settings import current_settings as load_settings
        from .text_inputs import input_limit, truncation_policy, token_budget, prepare_inputs, check_cancel
        maximum = token_budget(data.get('max_tokens'), self.spec.defaults, 'max_tokens', load_settings().embedding_max_tokens)
        limit = input_limit(self.processor, maximum, self.model)
        truncate = truncation_policy(data.get('truncate'), self.spec.defaults, load_settings().embedding_truncate)
        texts = data['input'] if isinstance(data['input'], list) else [data['input']]
        inputs, details, policy = prepare_inputs(self.processor, texts, limit, truncate, cancel=cancel)
        inputs = inputs.to(self.device)
        check_cancel(cancel)
        hidden = self.model(**inputs).last_hidden_state.float()
        check_cancel(cancel)
        mask = inputs["attention_mask"].to(hidden.device).unsqueeze(-1)
        vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1)
        count = int(inputs["attention_mask"].sum())
        vectors = vectors.cpu().tolist()
        check_cancel(cancel)
        return {"vectors": vectors, "usage": {"prompt_tokens": count, "total_tokens": count},
                'input_details': details, 'truncation': policy,
                'provenance':getattr(self,'embedding_provenance',None)}

    def transcribe(self, data, cancel=None):
        from .asr import transcribe
        return transcribe(self.model, self.spec, data, cancel)

    def speak(self, data):
        import numpy as np;from .openai_params import speech_speed
        voice = data.get("voice", "default");speech_speed(data, self.spec.backend)
        if self.spec.backend == "xtts":
            if voice not in self.spec.voices:
                raise ValueError(f"Unknown voice {voice!r}; configure a local reference WAV in models.json voices")
            result = self.model.synthesize(data["input"], self.model.config,
                                           speaker_wav=self.spec.voices[voice], language=data.get("language") or "en",
                                           **self.spec.defaults)
            wave, rate = result["wav"], self.model.config.audio.output_sample_rate
        else:
            if voice != "default":
                raise ValueError("This speech model supports voice='default' only")
            if data.get("language"):
                raise ValueError("Language is fixed by this Transformers checkpoint; select a language-specific model")
            defaults={k:v for k,v in self.spec.defaults.items() if k not in {"voice","language","response_format","speed"}}
            result = self.model(data["input"], **defaults)
            wave, rate = result["audio"], result["sampling_rate"]
        from .media import write_wav
        return write_wav(wave, rate)

    def diffuse(self, task, data, cancel=None):
        import torch
        pipeline = self.model
        from .instruction_edit import PIPELINE,generate
        if type(pipeline).__name__==PIPELINE:return generate(pipeline,self.spec,task,data,self.device,cancel)
        if data.get('image_guidance_scale') is not None or 'image_guidance_scale' in self.spec.defaults:
            from .engine import EngineError
            raise EngineError('image_guidance_scale requires InstructPix2Pix',400,'unsupported_capability')
        if task=="video" and data.get("input_reference"):
            from .i2v import generate
            return generate(pipeline,data,self.spec.defaults,self.device,cancel)
        if task == "video" and data.get("pipeline_options"):
            raise ValueError("pipeline_options currently requires an image-conditioned video request")
        if task == "image-edit":
            if 'mask_image' in data:
                from .image_mask import validate
                from .engine import EngineError
                validate(data['mask_image'],data['image'],(data['width'],data['height']))
                supported = {'StableDiffusionPipeline','StableDiffusionImg2ImgPipeline','StableDiffusionInpaintPipeline',
                             'StableDiffusionXLPipeline','StableDiffusionXLImg2ImgPipeline','StableDiffusionXLInpaintPipeline'}
                if type(pipeline).__name__ not in supported:
                    raise EngineError('Masked Diffusers edits currently require SD or SDXL',400,'unsupported_capability')
                try:
                    from diffusers import AutoPipelineForInpainting
                    pipeline = AutoPipelineForInpainting.from_pipe(pipeline)
                except (ImportError,ValueError,TypeError,NotImplementedError) as exc:
                    raise EngineError('This Diffusers pipeline does not support inpainting: ' + str(exc),400,'unsupported_capability') from exc
            else:
                from diffusers import AutoPipelineForImage2Image
                pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
            pipeline.set_progress_bar_config(disable=True)
        defaults={k:v for k,v in self.spec.defaults.items() if k not in {"size","steps","frames","fps","seed","n","response_format","task","strength","preserve_unmasked","mask","mask_base64","mask_image"}}
        kwargs = dict(defaults, prompt=data["prompt"], num_inference_steps=data["steps"],
                      height=data["height"], width=data["width"])
        if cancel is not None:
            import inspect
            def check_cancel():
                if cancel.is_set():
                    from .engine import EngineError
                    raise EngineError('Diffusion cancelled',499,'cancelled')
            check_cancel()
            parameters=inspect.signature(pipeline.__call__).parameters
            if 'callback_on_step_end' in parameters:
                def step_end(pipe, step, timestep, values):
                    check_cancel()
                    return values
                kwargs['callback_on_step_end']=step_end
            elif 'callback' in parameters:
                kwargs.update(callback=lambda *args:check_cancel(),callback_steps=1)
        for key in ("negative_prompt", "guidance_scale"):
            if data.get(key) is not None:
                kwargs[key] = data[key]
        if data.get("seed") is not None:
            kwargs["generator"] = torch.Generator(device=self.device if self.device.startswith("cuda") else "cpu").manual_seed(data["seed"])
        if task == "video":
            kwargs["num_frames"] = data["frames"]
            frames = pipeline(**kwargs).frames[0]
            if len(frames) < data["frames"]:
                raise ValueError("This pipeline produced fewer frames than requested; choose a supported frame count")
            return frames[:data["frames"]]
        kwargs["num_images_per_prompt"] = data["n"]
        if task == "image-edit":
            kwargs.update(image=data["image"], strength=data["strength"])
            if 'mask_image' in data:kwargs['mask_image']=data['mask_image']
        return pipeline(**kwargs).images
