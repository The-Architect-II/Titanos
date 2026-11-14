# Model adapters: choose the right files and runtime

Use this reference before registering a model or preparing an acquisition. A task tag, repository name, successful registration or installed library is **not proof that a checkpoint can load or run on the chosen device**. Weights remain external and replaceable; Titanos does not need PAIR or HiveMind.

For a local registry file, `titanos --config models.json capabilities name` reads that file and describes the model **without a server**. HTTP `POST /api/models/capabilities` with `{"model":"name"}`, `Client.capabilities("name")` and MCP `model_capabilities` inspect the running server's registered model. These are passive capability surfaces. Model **details/load** operations may load a model and are not substitutes for passive inspection. Consult [named qualification evidence](CAPABILITY_MATRIX.md), not a catalog's generic support label, for actual model/package/device results.

The additive **current-source** `setup` object reports `schema_version=1`, `advisory=true`, `model_loaded=false` and `qualified=null`. Its status is `blocked` for identified setup problems, `needs_review` for uncertain/incompletely inspected layouts, or `ready_for_load_check` when no such issues were found. While the same model is loaded, `/api/models/capabilities` reconciles the Qwen3-TTS advisory (`voice_wrapper_unverified`; the generic runtime details of other speech backends such as XTTS report no voice bindings, so their advisories stay passive) with the runtime details the loaded model reports: the row becomes an `info` `*_observed` row quoting the bound wrapper revision and reference voices, `model_loaded` is `true`, `observed_at_load` names the scope, and the status becomes `verified_at_load` when no error or warning remains. Nothing is loaded to answer the request and no error or unrelated warning is suppressed; a busy engine (inference in progress) and worker-assigned models (resident in other processes) keep the passive view. Read each issue's `code`, `severity`, `message`, `files` and `next_action`, plus inventory completeness and metadata byte/hash pins. Inspection limits are controlled by `TITANOS_MODEL_INSPECTION_MAX_FILES`, `TITANOS_MODEL_INSPECTION_MAX_METADATA_BYTES` and `TITANOS_MODEL_INSPECTION_MAX_DEPTH`. Runtime reporting is module discoverability only, and `architecture.compatibility` remains `unverified` even in the ready state. This is no new admission gate or native qualification; older packages may lack this source feature.

## Three independent choices

| Choice | What it means |
|---|---|
| Binary profile: `engine` or `full` | `engine` contains the application, Studio and interfaces. `full` adds the numerical model stack and optional runtimes actually available to that build. It does **not** promise every extra/backend. Check that artifact's `BUILD.json`, `DEPENDENCIES.json` and `NATIVE_COMPONENTS.json`. |
| [Python/device installation recipe](../tools/profiles.json) | Selects a compatible Python runtime/dependency set, such as CUDA, DirectML, OpenVINO or MPS. It is not itself a model or target-native qualification. |
| [Managed native catalog](../src/titanos/data/native-profiles.json) | Checksum-pinned external **llama.cpp** executables for specific OS/architecture/runtime targets. A native profile is separate from the Python numerical stack and the model's GGUF files. |

Vendor-specific builders also produce narrower packages, such as GenAI or DirectML. Their actual artifact inventories govern inclusion; a vendor package name is not an alias for the normal FULL profile.

The named Windows 130e/acd122 runtime cohort includes Torch 2.8 CUDA-capable libraries, Transformers, Diffusers, base OpenVINO, ORT 1.27, Sherpa/core 1.13.8 and FFmpeg. It does **not** include OpenVINO GenAI/tokenizers, DirectML/WindowsML/QNN, XTTS, Qwen3-TTS or ACE music runtime. Its CPU audio results do not qualify all bundled libraries on GPUs. Other packages can have different contents.

## Adapter and model-family map

Paths below are **inside a complete external model directory**. A listed filename is a necessary input, not a guarantee of compatible weights or successful execution. Preserve model-index/config/tokenizer/processor files from the same pinned export; do not repair compatibility by guessing labels or swapping components.

| Task / backend | Required family and preparation | Important boundary |
|---|---|---|
| `chat`, `vision` / `llama-cpp` | Explicit `native.model_file` GGUF; complete shard set when split; a matching `native.mmproj` GGUF for vision; a trusted installed llama-server runtime. | Managed acquisition preserves auto, explicit CPU, a full NVIDIA GPU UUID or one `native:DEVICE` selector, with dtype auto and explicit GGUF/projector selection. Configure multiple-device groups after registration. The selected runtime must actually support the device; ordinary `cuda:0` is not a native selector. Not arbitrary non-GGUF weights. |
| `chat`, `vision`, `embeddings` / `transformers` | Complete local config, safetensors or shard index/files, tokenizer; vision also needs its image processor/chat template. Installed Transformers must implement the declared architecture. | Chat uses a causal-LM loader; vision uses image-text-to-text. Embeddings use attention-mask mean pooling and L2 normalization, not every sentence-transformer pipeline or task-specific pooling rule. |
| `transcription`, `speech` / `transformers` | Complete supported ASR/TTS pipeline, weights, feature extractor/tokenizer and any required processor/vocoder assets. Audio input needs the selected FFmpeg decoder. | Whisper transcription/translation and timestamp controls depend on the actual architecture/tokenizer. A generic ASR tag does not grant all languages, timestamps or translation. See [audio contracts](AUDIO_ANALYSIS.md). |
| `translation`, `image-to-text`, `object-detection` / `transformers` | Built-in seq2seq, vision-to-sequence or object-detection checkpoint with matching tokenizer/image processor and weights. | NMT needs actual tokenizer language IDs or a declared fixed pair/prefix. TrOCR's cropped-line path is not full-page OCR. Detection requires image-coordinate postprocessing. |
| `sentiment`, `rerank`, `pair-classification` / `transformers` | Trained safetensors classifier head, config and tokenizer. Sentiment: two/three classes and complete polarity mapping. Rerank: one scalar output. Pairs: two or more named categorical outputs. | Regression/multilabel heads and missing or mismatched weights are rejected. Pair labels retain the trained config's meaning; they are not remapped to sentiment or generic entailment. Trust/pickle/head-replacement shortcuts are forbidden. |
| `image-segmentation`, `depth-estimation` / `transformers` | Original builtin SAM (`model_type="sam"`) plus its resize/pad processor and safetensors; or **relative** Depth Anything with an embedded DINOv2 backbone and aligned postprocessing. | Not SAM2, metric depth or an external/pretrained-backbone lookup. Prompts/results preserve the input pixel coordinate system; loader/weight overrides are restricted. |
| `image`, `video` / `diffusers` | Complete local `model_index.json` pipeline, configs, weights, scheduler, tokenizer/text encoder, VAE and all declared components. Video/image-conditioned families may need an image encoder and reference frame. | Exact pipeline family decides which generation/edit/mask/video controls work. Builtin InstructPix2Pix has a dedicated eight-channel contract; generic image-to-image is not instruction editing. Image geometry/conditioning rules remain family-specific. |
| `embeddings`, `tensor` / `onnx` or `openvino` | Prepared `model.onnx` or `openvino_model.xml`, with any referenced external data/BIN. `graph` can name another relative file; OpenVINO can also read a supported explicit ONNX graph. Embeddings additionally need a local tokenizer and rank-two `input_ids`/`attention_mask` graph inputs. | Tensor callers supply exact named dtype/shape/data. An exported graph is not a complete arbitrary modality adapter. Base OpenVINO is not GenAI. Compiled sharding is unsupported; CPU fallback is off unless the ORT-specific opt-in is accepted. |
| `vad` / `onnx` | CPU Silero graph with the supported input/state/sample-rate ABI; default `silero_vad.onnx`, or explicit relative `graph`. | Request-local state, 8/16 kHz and configured timing limits; not any file called VAD. Probabilities are uncalibrated, not speaker identity or ASR confidence. |
| `chat`, `image` / `openvino-genai` | Complete GenAI export and compatible GenAI/tokenizer runtime. Images support SD/SDXL: model index, scheduler, text encoder(s), UNet, VAE decoder, tokenizer XML/BIN pairs; VAE encoder is needed for edits. | Four-channel SD/SDXL exports only for this image adapter; not specialized nine-channel inpainting. A base OpenVINO XML or library alone is insufficient. |
| `image` / `onnx-diffusers` | Prepared SD1 ONNX pipeline with CLIP tokenizer/processor, PNDM scheduler, text encoder, UNet, VAE decoder and safety checker. | Explicit CPU or DirectML only; DML needs matching device ID/LUID. Not SDXL or an arbitrary ONNX diffusion export. Safety checker and prompt-budget checks are not silently removed. |
| `image-to-text` / `tesseract` | External language `.traineddata` files, plus a trusted Tesseract executable outside that model directory or included by the package. | CPU OCR controls apply; token-generation options do not. Executable location is operator configuration, not checkpoint-supplied code. |
| `speech` / `sherpa-onnx` | Recognized VITS/Piper-style archive: one ONNX model, `tokens.txt`, and its needed `espeak-ng-data` or lexicon. Pinned Sherpa Python/core 1.13.8, CPU. | The adapter constructs VITS, not every Sherpa TTS architecture. No arbitrary loader overrides, custom voices directory or accelerator fallback. Model terms differ from runtime terms. |
| `diarization` / `sherpa-onnx` | Trusted `diarization.json` profile with exact `segmentation.onnx` and `embedding.onnx` pins: reviewed pyannote 3.0/ERes2Net pair. Pinned Sherpa/core 1.13.8 and audio decoding. | Anonymous per-request CPU labels only. At most two simultaneous speakers per model frame; short clips bypass clustering. See [controls and overlap limits](AUDIO_ANALYSIS.md) and [named native result](evidence/2026-09-22-audio-analysis.md). |
| `speech` / `xtts` or `qwen3-tts` | XTTS config/checkpoint and required reference audio; Qwen3-TTS Base checkpoint, complete tokenizer components and the pinned streaming wrapper with PCM16 reference voices. | Separate optional dependencies/native audio closure are required. XTTS requires explicit trusted-pickle opt-in. The Qwen adapter is the Base voice-cloning path, not a generic promise for all Qwen TTS variants. |
| `music` / `ace-step` | Trusted `titanos-music.json`, complete pinned core/DiT components and an independently packaged ACE interpreter/adapter/runtime. Weights are excluded from the runtime bundle. | **XL-SFT is the implemented runner variant.** XL-Turbo/XL-Base are discovery/acquisition candidates, not supported inference variants. A prepared model folder or source venv alone is not standalone music runtime acceptance. See [music packaging](MUSIC_PACKAGING.md). |

## Minimal local registrations

These examples are valid registry shapes for **already prepared** folders. They do not download files or certify compatibility. Paths resolve relative to this registry file. The GGUF entry assumes a compatible managed runtime has been prepared; otherwise provide the trusted `native.executable` path explicitly.

```json
{
  "version": 1,
  "models": {
    "embed": {
      "path": "models/minilm", "task": "embeddings",
      "backend": "transformers", "device": "cpu", "dtype": "float32"
    },
    "chat": {
      "path": "models/qwen-gguf", "task": "chat",
      "backend": "llama-cpp", "device": "cpu",
      "native": {"model_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf", "gpu_layers": 0}
    },
    "vad": {
      "path": "models/silero", "task": "vad",
      "backend": "onnx", "device": "cpu", "graph": "silero_vad.onnx"
    },
    "speakers": {
      "path": "models/pyannote-eres2net", "task": "diarization",
      "backend": "sherpa-onnx", "device": "cpu"
    }
  },
  "workers": {}
}
```

Equivalent local registration and passive inspection use existing commands:

```text
titanos --config models.json add embed ./models/minilm --task embeddings --backend transformers --device cpu --dtype float32
titanos --config models.json capabilities embed
```

The CLI capabilities command uses the local registry selected by `--config`; no `serve` command is needed for it. For model acquisition, use [Find → plan → prepare → activate](MODEL_PROVIDERS.md), preserving the resolved revision, selected file manifest and license evidence. For a compatible Windows x64 native CPU runtime, `titanos runtime-prepare win-cpu-x64 --plan` inspects the configured plan; removing `--plan` starts the actual setup. Managed setup accepts configured profile names, not arbitrary executable URLs from callers.

## Safe defaults and actionable failures

- Keep `trust_remote_code=false`, `allow_pickle=false`, `allow_cpu_fallback=false` and `parallelism="single"` unless a specific supported contract and trusted input justify the opt-in. Trust flags can permit checkpoint code/deserialization; they are not compatibility fixes. The strict classifier/SAM/depth/compiled/Sherpa adapters forbid them where documented.
- Use an explicit device when placement matters. Generic framework `auto` may choose an accelerator; native GGUF and the fixed CPU adapters have different rules. Replicas, model sharding and tensor parallelism are separate mechanisms with separate family/device constraints. See [devices](DEVICES.md).
- Request controls override supported model defaults, then process settings where defined. Do not pass arbitrary backend flags through `defaults`; unsupported native defaults fail actionably. See [option parity](OPTION_PARITY.md).
- Missing files/shards: obtain the matching complete export, not an empty placeholder. Unknown architecture: use a supported built-in implementation or a justified explicit code policy; do not relabel it as a known model. Missing tokenizer/processor: recover matching preprocessing from the same revision. Missing runtime: choose a compatible installed package/profile; changing `backend` does not install it.
- Installation and capability metadata are preparatory evidence. A successful load, intended output, observed device, limits and cleanup still require a named qualification. This guide changes no acceptance closure and does not authorize held operations.

Source truth lives in [registry rules](../src/titanos/config.py), [framework loading](../src/titanos/backends.py), [prepared graphs](../src/titanos/compiled.py), [provider plans](../src/titanos/providers.py), [capabilities](../src/titanos/capabilities.py) and [build inclusion](../tools/build_binary.py). Specialized contracts are in their linked API pages. Keep Titanos's Apache-2.0 source, third-party runtime notices/source obligations and each model/data license separate; a FULL or Sherpa-containing bundle is not Apache-only.
