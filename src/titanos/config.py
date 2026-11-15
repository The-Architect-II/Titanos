"""Versioned local model registry. Paths are relative to this file, never HM2."""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .settings import current_settings as load_settings; from .native_context import SlotContextOptions

Task = Literal["chat", "vision", "embeddings", "transcription", "speech", "image", "video", "music", "tensor", "translation", "image-to-text", "object-detection", "sentiment", "rerank", "vad", "pair-classification", "depth-estimation", "image-segmentation", "diarization"]


KVCacheType = Literal['f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1']


class NativeOptions(SlotContextOptions):  # slot_context and max_slot_context: titanos.native_context
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    executable: str | None = None
    model_file: str
    model_files: list[str] = Field(default_factory=list, description='Complete GGUF shard set, including model_file; empty means a single file')
    mmproj: str | None = None
    context_size: int = Field(default_factory=lambda: load_settings().native_context_size, ge=512)
    slots: int = Field(default_factory=lambda: load_settings().native_slots, ge=1)
    threads: int = Field(default_factory=lambda: load_settings().native_threads, ge=1)
    batch_size: int | None = Field(default_factory=lambda: load_settings().native_batch_size or None, strict=True, ge=1)
    ubatch_size: int | None = Field(default_factory=lambda: load_settings().native_ubatch_size or None, strict=True, ge=1)
    gpu_layers: int | Literal['auto', 'all'] = Field(default_factory=lambda: load_settings().native_gpu_layers)
    split_mode: Literal["none", "layer", "row", "tensor"] = "none"
    tensor_split: list[float] = Field(default_factory=list, max_length=16)
    startup_timeout: int = Field(default_factory=lambda: load_settings().native_startup_timeout, ge=1)
    pooling: Literal['none', 'mean', 'cls', 'last', 'rank'] | None = Field(default=None, description='Embedding models only: llama-server --pooling; omitted keeps the GGUF default')
    flash_attn: Literal['auto', 'on', 'off'] = Field(default='auto', description='llama-server --flash-attn; auto is the runtime default and adds no argument')
    cache_type_k: KVCacheType | None = Field(default=None, description='llama-server --cache-type-k; omitted keeps the runtime default (f16)')
    cache_type_v: KVCacheType | None = Field(default=None, description='llama-server --cache-type-v; a quantized value requires flash_attn on')
    reasoning: Literal['on', 'off', 'auto'] | None = Field(default=None, description='llama-server --reasoning; omitted adds no argument')
    reasoning_budget: int | None = Field(default=None, strict=True, ge=-1, description='llama-server --reasoning-budget: 0 ends thinking at once, -1 leaves it unrestricted; omitted adds no argument')

    @field_validator('gpu_layers', mode='before')
    @classmethod
    def valid_gpu_layers(cls, value):
        if not (type(value) is int and value >= 0 or type(value) is str and value in {'auto', 'all'}):
            raise ValueError('gpu_layers must be auto, all or a nonnegative integer; zero explicitly disables offload')
        return value

    @model_validator(mode="after")
    def local_files(self):
        for value in (self.model_file, self.mmproj, *self.model_files):
            if value and (Path(value).is_absolute() or ".." in Path(value).parts or not value.endswith('.gguf')):
                raise ValueError('Native model/projector files must be relative GGUF paths within the model directory')
        if self.model_files and (self.model_file not in self.model_files or len(set(self.model_files)) != len(self.model_files)):
            raise ValueError('model_files must name each shard once and include model_file')
        if self.batch_size is not None and self.ubatch_size is not None and self.ubatch_size > self.batch_size:
            raise ValueError('native ubatch_size cannot exceed batch_size')
        if any(v <= 0 for v in self.tensor_split):
            raise ValueError('tensor_split proportions must be positive')
        if self.cache_type_v not in {None, 'f32', 'f16', 'bf16'} and self.flash_attn != 'on':
            raise ValueError('A quantized V cache (cache_type_v) requires flash_attn on; llama.cpp refuses it otherwise')
        return self


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    source_repo: str | None = None
    source_revision: str | None = None
    task: Task
    backend: Literal["transformers", "diffusers", "xtts", "onnx", "openvino", "openvino-genai", "llama-cpp", "qwen3-tts", "sherpa-onnx", "ace-step", "tesseract", "onnx-diffusers"] = "transformers"
    native: NativeOptions | None = None
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    placement: Literal["auto", "offload", "resident"] | None = Field(default=None, description='Music only: where the isolated runtime keeps weights between stages')
    trust_remote_code: bool = False
    allow_pickle: bool = False
    graph: str | None = None
    allow_cpu_fallback: bool = False
    parallelism: Literal["single", "model_sharding", "tensor_parallel"] = "single"
    tensor_parallel_size: int = Field(default=2, ge=2, le=16)
    load_options: dict = Field(default_factory=dict)
    defaults: dict = Field(default_factory=dict)
    voices: dict[str, str] = Field(default_factory=dict)
    voice_map: dict[str, str] = Field(default_factory=dict, description="Speech only: client voice names (alloy, nova, ...) mapped onto this model's own voice ids (openai_params.speech_voice)")
    voice_texts: dict[str, str] = Field(default_factory=dict, description='qwen3-tts only: the transcript of each voice reference; a voice with one is cloned in context (ICL), others by x-vector')

    @model_validator(mode="after")
    def coherent_backend(self):
        if self.placement is not None and self.task!='music':
            raise ValueError('placement selects the isolated music runtime weight placement (auto/offload/resident); other tasks choose device and dtype only')
        if self.voice_map and self.task != 'speech':
            raise ValueError('voice_map applies to speech models only: it maps client voice names onto this model\'s voice ids')
        if any(not isinstance(v, str) or not v or not isinstance(k, str) or not k for k, v in self.voice_map.items()):
            raise ValueError('voice_map entries are nonblank voice-name strings mapped onto nonblank backend voice ids')
        if self.voice_texts and (self.task != 'speech' or self.backend != 'qwen3-tts'):
            raise ValueError('voice_texts applies to qwen3-tts speech models only: the transcript of each voice reference')
        if any(k not in self.voices or not isinstance(v, str) or not v.strip() for k, v in self.voice_texts.items()):
            raise ValueError('voice_texts maps registered voice ids (voices) onto their nonblank reference transcripts')
        if self.task == 'diarization':
            from .diarization import prepare
            prepare(self, {})
            self.device = 'cpu'
            return self
        if self.backend=='ace-step':
            if (self.task!='music' or self.parallelism!='single' or self.trust_remote_code or self.allow_pickle
                    or self.graph or self.native or self.allow_cpu_fallback or self.load_options):
                raise ValueError('ACE-Step music uses one isolated trusted runtime per worker and a complete local bundle')
            if set(self.defaults)-{'mode','duration','vocal_language','bpm','key_scale','seed','guidance_scale','steps','thinking','use_cot_metas','use_cot_caption','use_cot_language'} or any(type(self.defaults[key]) is not bool for key in ('thinking','use_cot_metas','use_cot_caption','use_cot_language') if key in self.defaults):
                raise ValueError('Unsupported ACE-Step model default')
            return self
        if self.task=='music':
            raise ValueError('Music requires the isolated ace-step backend')
        if self.backend == "onnx-diffusers":
            from .onnx_images import validate_registration
            validate_registration(self)
            return self
        if self.backend == 'tesseract':
            from .tesseract import validate_registration
            validate_registration(self)
            return self
        if self.backend == 'sherpa-onnx':
            if (self.task != 'speech' or self.device not in {'auto','cpu'} or self.parallelism != 'single'
                    or self.dtype != 'auto' or self.graph or self.native or self.trust_remote_code
                    or self.allow_pickle or self.allow_cpu_fallback or self.load_options or self.voices):
                raise ValueError('Sherpa ONNX TTS requires one CPU speech archive with builtin model metadata and no executable loader overrides')
            if set(self.defaults)-{'voice','speaker_id'}:
                raise ValueError('Sherpa ONNX TTS defaults accept only voice or speaker_id')
            speaker=self.defaults.get('speaker_id')
            if speaker is not None and (isinstance(speaker,bool) or not isinstance(speaker,int) or speaker<0):
                raise ValueError('Sherpa ONNX speaker_id must be a nonnegative integer')
            self.device='cpu'
            return self
        if self.backend=='diffusers' and self.task=='image':
            from .instruction_edit import declared,validate_spec
            if declared(self):validate_spec(self)
        if self.task=='transcription' and 'prompt' in self.defaults:
            from .settings import current_settings
            value=self.defaults['prompt']
            if not isinstance(value,str) or not value.strip() or len(value)>current_settings().max_prompt_chars:
                raise ValueError('Prompt defaults are nonblank transcription context within TITANOS_MAX_PROMPT_CHARS')
        if self.task == 'image-segmentation':
            if self.backend != 'transformers' or self.trust_remote_code or self.allow_pickle or self.parallelism != 'single':
                raise ValueError('Prompted SAM requires one Transformers device, safetensors and builtin local code')
            if set(self.load_options)-{'attn_implementation','low_cpu_mem_usage'}:
                raise ValueError('SAM load options cannot override checkpoint configuration, nested loaders or weights')
            if ('low_cpu_mem_usage' in self.load_options and type(self.load_options['low_cpu_mem_usage']) is not bool
                    or 'attn_implementation' in self.load_options and self.load_options['attn_implementation'] not in ('eager','sdpa')):
                raise ValueError('SAM load options require a boolean memory flag and builtin eager/sdpa attention')
            if set(self.defaults)-{'multimask','mask_threshold'}:
                raise ValueError('SAM defaults accept only multimask and mask_threshold')
            from .segmentation import resolve_prompt
            resolve_prompt({'box':[0,0,1,1]},self.defaults)
        if self.task == 'image' and {'mask','mask_base64','mask_image'} & self.defaults.keys():
            raise ValueError('Mask bytes are request inputs, not model defaults')
        if 'preserve_unmasked' in self.defaults:
            if self.task != 'image' or type(self.defaults['preserve_unmasked']) is not bool:
                raise ValueError('preserve_unmasked is a boolean default for image models only')
        if 'retrieval_prompt_version' in self.defaults:
            if self.task != 'chat' or self.backend != 'llama-cpp':
                raise ValueError('retrieval_prompt_version defaults apply to owned native chat models only')
            if self.defaults['retrieval_prompt_version'] is not None:
                from .retrieval_prompts import validate_version
                validate_version(self.defaults['retrieval_prompt_version'])
        from .audio_gating import ASRGateFields, GATE_FIELDS
        gate_defaults = {k:v for k,v in self.defaults.items() if k in GATE_FIELDS}
        if gate_defaults:
            if self.task != 'transcription':
                raise ValueError('ASR VAD gate defaults apply to transcription models only')
            ASRGateFields.model_validate(gate_defaults)
        if self.backend == 'llama-cpp':
            if not self.native or self.task not in {'chat', 'vision', 'embeddings'}:
                raise ValueError('llama-cpp requires native options and a chat, vision or embeddings model')
            if self.task == 'vision' and not self.native.mmproj:
                raise ValueError('Native vision requires its matching GGUF projector')
            if self.native.pooling is not None and self.task != 'embeddings':
                raise ValueError('native.pooling applies to llama-cpp embeddings models only (llama-server --pooling)')
            if self.task == 'embeddings' and self.native.mmproj:
                raise ValueError('Native embeddings take no GGUF projector')
            if self.parallelism != 'single' or self.dtype != 'auto' or self.load_options or self.graph or self.allow_pickle or self.trust_remote_code or self.allow_cpu_fallback:
                raise ValueError('Use native options for llama-cpp placement; PyTorch/compiled options do not apply')
            if self.device not in {'auto', 'cpu'} and not self.device.startswith(('GPU-', 'native:')):
                raise ValueError('Native devices use auto, cpu, full NVIDIA GPU UUIDs, or native:DEVICE names from llama-server --list-devices')
            if self.device == 'cpu' and self.native.gpu_layers == 'auto':
                self.native.gpu_layers = 0
            if self.device == 'cpu' and self.native.gpu_layers != 0:
                raise ValueError('CPU placement requires gpu_layers=0; auto resolves to zero on explicit CPU')
            if self.device not in {'auto', 'cpu'} and self.native.gpu_layers == 0:
                raise ValueError('An explicit accelerator requires gpu_layers=auto, all or a positive integer')
            count=len(self.device.removeprefix('native:').split(','))
            if len(set(self.device.removeprefix('native:').split(','))) != count:
                raise ValueError('Native splitting requires distinct devices')
            if (count>1)!=(self.native.split_mode!='none'):
                raise ValueError('Native splitting requires multiple explicitly selected devices; single-device mode uses split_mode=none')
            if self.native.tensor_split and len(self.native.tensor_split)!=count:
                raise ValueError('Native tensor_split must match the explicit device count')
            return self
        if self.native is not None:
            raise ValueError('native options require the llama-cpp backend')
        if self.task == 'vad':
            if self.backend != 'onnx' or self.device not in {'auto', 'cpu'} or self.parallelism != 'single' or self.dtype != 'auto' or self.load_options or self.allow_pickle or self.trust_remote_code or self.allow_cpu_fallback:
                raise ValueError('VAD requires the single-device CPU ONNX adapter without loader overrides')
            if self.graph and (Path(self.graph).is_absolute() or '..' in Path(self.graph).parts or not self.graph.endswith('.onnx')):
                raise ValueError('VAD graph must be a relative ONNX path')
            if set(self.defaults) - {'sample_rate', 'threshold', 'min_speech_ms', 'min_silence_ms', 'speech_pad_ms'}:
                raise ValueError('Unsupported VAD model defaults')
            import math
            for key, value in self.defaults.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)) or value < 0:
                    raise ValueError('VAD model defaults must be finite nonnegative numbers')
                if key == 'sample_rate' and (not isinstance(value, int) or value not in {8000, 16000}):
                    raise ValueError('VAD sample rate must be 8000 or 16000')
                if key == 'threshold' and not 0 < value < 1 or key == 'min_silence_ms' and value <= 0:
                    raise ValueError('Invalid VAD threshold or silence default')
                if key in {'min_speech_ms', 'min_silence_ms', 'speech_pad_ms'} and value > load_settings().max_audio_seconds * 1000:
                    raise ValueError('VAD timing defaults exceed the configured audio duration limit')
            self.device = 'cpu'
            return self
        if self.backend in {"onnx", "openvino", "openvino-genai"}:
            if self.parallelism != "single":
                raise ValueError("Compiled model sharding is not implemented")
            if self.trust_remote_code or self.allow_pickle:
                raise ValueError("Compiled adapters load prepared graphs and local tokenizers; trust flags are not applicable")
            if self.allow_cpu_fallback and self.backend != "onnx":
                raise ValueError("allow_cpu_fallback applies to ONNX Runtime only")
            if self.backend == "openvino-genai" and self.graph not in (None, "openvino_model.xml"):
                raise ValueError("OpenVINO GenAI uses the complete standard model directory")
            tasks = {"chat", "image"} if self.backend == "openvino-genai" else {"embeddings", "tensor"}
            if self.task not in tasks:
                raise ValueError(f"{self.backend} supports {sorted(tasks)}; use a task-specific model adapter for other tasks")
            if self.dtype != "auto":
                raise ValueError("Compiled model precision is set during preparation; use dtype=auto")
            if self.backend == 'openvino-genai' and self.task == 'image':
                from .genai_images import validate_registration
                validate_registration(self)
            if self.graph and (Path(self.graph).is_absolute() or ".." in Path(self.graph).parts):
                raise ValueError("graph must be a relative path inside the model directory")
            return self
        if self.graph is not None or self.allow_cpu_fallback:
            raise ValueError('graph and allow_cpu_fallback apply to compiled adapters only')
        if self.task in {"translation", "image-to-text", "object-detection", "sentiment", "rerank", "pair-classification", "depth-estimation"} and self.parallelism != "single":
            raise ValueError("API Plus adapters currently use one device per worker; use replicas for parallel requests")
        if self.task == 'depth-estimation':
            if self.trust_remote_code or self.allow_pickle or self.defaults:
                raise ValueError('Relative depth requires safetensors, built-in code and no generation defaults')
            if set(self.load_options) - {'attn_implementation', 'low_cpu_mem_usage'}:
                raise ValueError('Depth load options cannot replace checkpoint configuration or weights')
        if self.task in {'sentiment', 'rerank', 'pair-classification'}:
            if self.trust_remote_code or self.allow_pickle:
                raise ValueError('Local classification requires safetensors and built-in Transformers code')
            protected_classifier = {'config', 'state_dict', 'num_labels', 'id2label', 'label2id', 'problem_type',
                                    'ignore_mismatched_sizes', 'output_loading_info', 'from_tf', 'from_flax',
                                    'variant', 'subfolder', 'key_mapping'}
            if protected_classifier.intersection(self.load_options):
                raise ValueError('Classifier load_options cannot replace checkpoint configuration, labels or weights')
            limit = self.defaults.get('max_input_tokens')
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
                raise ValueError('Classifier max_input_tokens default must be a positive integer')
        if self.task == 'pair-classification' and set(self.defaults) - {'max_input_tokens', 'truncate'}:
            raise ValueError('Pair-classifier defaults accept only max_input_tokens and truncate; native labels are not remapped')
        if self.task in {'embeddings', 'rerank', 'pair-classification'} and self.defaults.get('truncate') is not None and not isinstance(self.defaults['truncate'], bool):
            raise ValueError('truncate default must be a boolean')
        if self.parallelism == "tensor_parallel" and (self.backend != "transformers" or self.task not in {"chat", "vision", "embeddings"}):
            raise ValueError("Native tensor parallelism requires a supported Transformers chat, vision or embedding architecture")
        if self.parallelism != 'single' and self.device.startswith(('GPU-', 'HIP-', 'XPU-')):
            raise ValueError('A single stable GPU identity cannot imply a multi-device group; use pinned worker gpus or an explicit visible group with a runtime device address')
        if self.parallelism != "single" and self.backend in {"xtts", "qwen3-tts"}:
            raise ValueError("This speech backend uses one device per worker; use replicas for parallel requests")
        if self.task == "tensor":
            raise ValueError("tensor tasks require the onnx or openvino backend")
        expected = "diffusers" if self.task in {"image", "video"} else "transformers"
        if self.backend != expected and not (self.backend in {"xtts", "qwen3-tts"} and self.task == "speech"):
            raise ValueError(f"{self.task} requires {expected} (speech also supports xtts and qwen3-tts)")
        protected = {"local_files_only", "trust_remote_code", "use_safetensors", "device_map", "torch_dtype", "dtype", "custom_pipeline", "tp_plan", "tp_size", "device_mesh", "max_memory"}
        if protected.intersection(self.load_options):
            raise ValueError("use the dedicated safety/device/dtype fields instead of load_options")
        if self.backend == "xtts" and not self.allow_pickle:
            raise ValueError("XTTS checkpoints require explicit allow_pickle=true")
        if self.backend == 'qwen3-tts':
            from .qwen_tts_backend import check_defaults
            check_defaults(self.defaults)
        return self


class WorkerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    models: list[str] = Field(min_length=1, max_length=128)
    device: str = "cpu"
    gpus: list[str] = Field(default_factory=list, max_length=16)
    mode: Literal["single", "model_sharding", "tensor_parallel"] = "single"
    tensor_parallel_size: int = Field(default=2, ge=2, le=16)
    threads: int = Field(default_factory=lambda: load_settings().worker_threads, strict=True, ge=1)
    timeout: float = Field(default_factory=lambda: load_settings().worker_timeout, strict=True, gt=0)
    profile: bool = False

    @model_validator(mode="after")
    def assignment(self):
        import re
        if len(set(self.models)) != len(self.models):
            raise ValueError('A worker model list must contain unique IDs; use separate workers for replicas')
        if len(set(self.gpus)) != len(self.gpus) or any(not re.fullmatch(r"[0-9]+|(?:GPU|HIP)-[A-Za-z0-9-]+", g) for g in self.gpus):
            raise ValueError("GPUs must be distinct indices or NVIDIA GPU UUIDs")
        if self.mode == "single" and len(self.gpus) > 1:
            raise ValueError("Use separate workers for replicas or mode=model_sharding for a GPU group")
        if self.mode == "tensor_parallel" and ((self.gpus and len(self.gpus) < 2) or (not self.gpus and self.device.split(":")[0] not in {"cpu", "xpu"})):
            raise ValueError("Tensor parallelism needs two assigned GPUs or an explicit CPU/XPU group")
        if self.mode == "model_sharding" and len(self.gpus) < 2:
            raise ValueError("Model sharding requires at least two assigned GPUs")
        return self


def native_assignment(spec,worker):
    effective=spec.model_dump()
    if worker.gpus:
        if any(not value.startswith('GPU-') for value in worker.gpus):
            raise ValueError('Native CUDA worker groups require stable NVIDIA GPU UUIDs')
        effective['device'] = ','.join(worker.gpus)
        split = {'single': 'none', 'model_sharding': 'layer', 'tensor_parallel': 'tensor'}[worker.mode]
        if worker.mode == 'model_sharding' and spec.native.split_mode == 'row':
            split = 'row'
        effective['native']['split_mode'] = split
        effective['parallelism'] = 'single'  # One llama.cpp process performs native splitting.
    else:
        if worker.mode != 'single':
            raise ValueError('Native grouped modes require an explicit stable GPU group')
        effective['device']=spec.device if worker.device=='auto' else worker.device
    effective['native']['threads']=worker.threads
    if effective['device']=='cpu':
        effective['native']['gpu_layers']=0
        effective['native']['split_mode']='none'
        effective['native']['tensor_split']=[]
    return ModelSpec.model_validate(effective)


def worker_assignment(spec, worker):
    """Revalidate effective worker placement; assignments cannot bypass model gates."""
    if spec.backend == 'llama-cpp':
        return native_assignment(spec, worker)
    values = spec.model_dump()
    values.update(device='cuda:0' if worker.gpus else worker.device, parallelism=worker.mode)
    if worker.mode == 'tensor_parallel':
        values['tensor_parallel_size'] = len(worker.gpus) or worker.tensor_parallel_size
    return ModelSpec.model_validate(values)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    workers: dict[str, WorkerSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_names(self):
        import re
        for name in list(self.models) + list(self.workers):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", name):
                raise ValueError(f"invalid model/worker id: {name!r}")
        if len(self.workers) > 64:
            raise ValueError("At most 64 workers are allowed")
        for worker in self.workers.values():
            for name in worker.models:
                if name not in self.models:
                    raise ValueError(f"Worker references unknown model {name!r}")
                spec = self.models[name]
                if spec.backend == 'tesseract' and (worker.device != 'cpu' or worker.gpus or worker.profile):
                    raise ValueError('Tesseract replicas require CPU placement without GPU or Torch profiling options')
                if spec.task == 'diarization' and (worker.device != 'cpu' or worker.gpus or worker.profile or worker.mode != 'single'):
                    raise ValueError('Diarization replicas require one CPU worker without GPU or Torch profiling options')
                if spec.task in {"translation", "image-to-text", "object-detection", "sentiment", "rerank", "depth-estimation", "image-segmentation", "music"} and worker.mode != "single":
                    raise ValueError("API Plus adapters require single-device workers; use separate replicas for parallel requests")
                if spec.backend == 'llama-cpp':
                    if len(worker.models) != 1 or worker.profile:
                        raise ValueError('A native replica names one model; Torch profiling does not apply to llama.cpp')
                    native_assignment(spec,worker)
                    continue
                if worker.gpus and spec.backend not in {"transformers", "diffusers", "xtts", "qwen3-tts", "ace-step"}:
                    raise ValueError("gpus selects CUDA/ROCm devices; use device for compiled runtimes")
                if worker.mode == "tensor_parallel" and (spec.backend != "transformers" or spec.task not in {"chat", "vision", "embeddings"}):
                    raise ValueError("This model adapter does not implement tensor parallelism")
                if worker.mode == "model_sharding" and spec.backend not in {"transformers", "diffusers"}:
                    raise ValueError("Model sharding requires Transformers or Diffusers")
                worker_assignment(spec, worker)
        return self


def home() -> Path:
    return Path(os.environ.get("TITANOS_HOME", "~/.titanos")).expanduser().resolve()


def config_path(value=None) -> Path:
    return Path(value).expanduser().resolve() if value else home() / "models.json"


def read_config(path: Path) -> Config:
    if not path.exists():
        raise ValueError(f"No registry at {path}. Run 'titanos init' first.")
    cfg = Config.model_validate_json(path.read_text(encoding="utf-8"))
    for spec in cfg.models.values():
        spec.path = str((path.parent / Path(spec.path).expanduser()).resolve())
        spec.voices = {k: str((path.parent / Path(v).expanduser()).resolve()) for k, v in spec.voices.items()}
    return cfg


def write_config(path: Path, cfg: Config):
    # Atomic replacement keeps a stopped CLI from leaving a truncated registry.
    cfg = Config.model_validate(cfg.model_dump())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="models-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg.model_dump(mode="json"), f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def registry_lock(path: Path, timeout=30):
    """The shared CLI/server registry lock; acquire/release on one thread."""
    from filelock import FileLock
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path) + ".lock", timeout=timeout)


def update_config(path: Path, change) -> Config:
    """Apply a short local mutation to the latest registry under a process lock.

    Perform downloads and other slow preparation before calling this function.
    The callback must recheck its replacement/removal conditions against `cfg`;
    an exception leaves the previous registry unchanged.
    """
    path = Path(path).expanduser().resolve()
    with registry_lock(path):
        cfg = read_config(path) if path.exists() else Config()
        change(cfg)
        write_config(path, cfg)
        return cfg
