# Extended inference API

Titanos implements these typed local operations over registered checkpoints: text translation, text sentiment and classifier-versus-LLM sentiment composition, image-to-text/OCR, object detection, relative depth, prompted segmentation, reranking, retrieval answering, premise/hypothesis pair classification, English audio translation, voice activity detection, anonymous speaker diarization and overlap-preserving audio analysis. The media route names follow the reusable extensions in HiveMind's HLI API. Titanos loads the models itself: HiveMind, external gateways and cloud credentials are not required. When the operator has configured the optional [cloud broker](BROKER.md), a few stages can run on a cloud model instead; see [Location and cloud stages](#location-and-cloud-stages).

Use `GET /v1/extensions` to discover operations, matching registered models and their JSON Schema. A registered model still needs a compatible checkpoint and a successful load; the discovery response says this explicitly. The same typed operations are available at `POST /v1/extensions/infer` and through the MCP `extension_infer` tool. For required files, preprocessing, runtime/profile distinctions and minimal valid registrations, use the [model-adapter reference](MODEL_ADAPTERS.md); it separates configured compatibility from named model/device qualification.

## Endpoints

| Endpoint | Content type | Required input | Registered model task |
|---|---|---|---|
| `POST /v1/translations` | JSON | model, text, target_language | translation |
| `POST /v1/sentiment`; `POST /v1/sentiment/compose` | JSON | model, text; compose takes classifier_model, llm_model, text | sentiment classifier or supported local chat model; compose adds a chat model |
| `POST /v1/rerank`; `POST /v1/retrieval/answer` | JSON | model, query, documents; answer takes llm_model, embedding_model, rerank_model, query, documents | rerank; chat (llama-cpp) with embeddings and rerank |
| `POST /v1/classifications/text-pairs` | JSON | model, pairs with premise/hypothesis | pair-classification |
| `POST /v1/images/ocr`; `/v1/images/detect`; `/v1/images/depth`; `/v1/images/segment` | JSON | model, one inline image; segment adds a box or points | image-to-text; object-detection; depth-estimation; image-segmentation |
| `POST /v1/audio/vad`; `/v1/audio/diarizations`; `/v1/audio/analysis` | JSON | model, inline audio; analysis adds diarization_model | vad; diarization; transcription with diarization |
| `POST /v1/audio/transcriptions` | multipart | model, file | transcription |
| `POST /v1/audio/translations` | multipart | model, file; optional translation_model | transcription |
| `POST /v1/extensions/infer` | JSON | operation plus the corresponding input | as above |

The operation names accepted by `POST /v1/extensions/infer` and listed by `GET /v1/extensions` are `translation`, `ocr`, `detection`, `depth`, `segmentation`, `audio_translation`, `audio_transcription`, `vad`, `diarization`, `audio_analysis`, `sentiment`, `sentiment_compose`, `rerank`, `retrieval_answer` and `pair_classification`. Unknown operations fail validation. A generic transport does not create a backend for an unsupported model architecture. [Text sentiment](SENTIMENT.md) documents its dedicated classifier and supported LLM routes, uncalibrated scores, provenance and abstention policy. [Text-pair classification](PAIR_CLASSIFICATION.md) preserves the classifier's native labels and logits without sentiment/rerank remapping.

All operations accept a `worker` assignment and optional `timeout` in milliseconds. The timeout includes waiting for the model; generation cancellation is cooperative, so a running native/model operation may finish before the timeout error is returned. A composed ASR+NMT request shares its timeout budget across both stages. Use separate worker processes when an enforceable process deadline is required. JSON `location` accepts `local_only` (the default), `prefer_local`, `auto` or `cloud_only`; see below.

## Location and cloud stages

The `location` body field (or the `X-Titanos-Location` header, which takes precedence) decides where a stage may run. Without a configured broker every stage runs locally and `location` changes nothing. With the [cloud broker](BROKER.md) configured:

| `location` | Effect |
|---|---|
| `local_only` (default) | Local models only; a cloud model name answers 400 `location_unsatisfiable` |
| `prefer_local` | Local first; a cloud target is kept only when the cloud policy is `fallback` |
| `auto` | The configured order of the named model or virtual model |
| `cloud_only` | Cloud and router targets only |

Cloud-eligible stages, reached by naming a cloud model (a published alias or a qualified ref of at most 128 characters) with a location that allows it:

- the ASR stage of `/v1/audio/transcriptions`, `/v1/audio/translations` and `/v1/audio/analysis` (the local VAD gate still runs first, and a clip without speech sends nothing);
- the LLM stage of `/v1/sentiment` and `/v1/sentiment/compose`;
- `/v1/translations` through a cloud chat model with the versioned prompt `broker-translation-v1`. `include_provenance: true` with a cloud target answers 400 `provenance_unavailable_for_cloud`, because the provider's internals cannot be attested.

Local-only stages (OCR, detection, depth, segmentation, VAD, diarization, pair classification and rerank) answer `cloud_only` with 400 `broker_capability_unavailable`; `auto` and `prefer_local` keep them local. `/v1/retrieval/answer` refuses a cloud LLM with a typed 400 and no upstream call, because exact context accounting and checkpoint provenance need a local native model. `GET /v1/extensions` reports `runtime` (`standalone-local`, or `local+broker` with the list of `cloud_stages`). The key's scope for these routes is `inference:plus`, plus `cloud` or `cloud:<upstream>` for a cloud target. Evidence per stage is in [BROKER_MATRIX.md](BROKER_MATRIX.md).

## Translation

```json
{
  "model": "translator",
  "text": "The house is wonderful.",
  "source_language": "eng_Latn",
  "target_language": "fra_Latn",
  "max_tokens": 128
}
```

The result includes `text`, `translations:[{text:...}]`, source/target language IDs, model and `translated`. Equal source and target languages return the unchanged text with `translated:false`.

Optional `include_provenance:true` returns [versioned translation stages](TRANSLATION_PROVENANCE.md) with actual loaded NMT checkpoint/device and tokenizer choices. Ordinary outputs remain unchanged when it is omitted. Audio stages distinguish the observed ASR language from caller NMT IDs; fixed Marian fr-en callers must explicitly select `fr` and `en`, rather than assuming the audio target default `eng_Latn` is a tokenizer alias.

Multilingual tokenizers such as NLLB use their own language IDs. Fixed-direction models need matching `source_language` and `target_language` in registry `defaults`; a different requested direction is rejected. T5-style models can also specify a `translation_prefix`. Example registry entry for an explicitly downloaded local T5 checkpoint:

```json
{
  "path": "./models/t5-small",
  "task": "translation",
  "backend": "transformers",
  "device": "auto",
  "defaults": {
    "source_language": "eng_Latn",
    "target_language": "fra_Latn",
    "translation_prefix": "translate English to French: ",
    "max_tokens": 128,
    "max_input_tokens": 512,
    "generation_kwargs": {"num_beams": 4}
  }
}
```

`max_tokens`, `temperature`, and `top_p` may be supplied per request. Each resolves in the order request, then model `defaults`, then the process settings `TITANOS_DEFAULT_MAX_TOKENS`, `TITANOS_DEFAULT_TEMPERATURE` and `TITANOS_DEFAULT_TOP_P`. A `temperature` of zero at any layer selects greedy decoding and `top_p` is then not applied, matching the chat routes. Long translation input is rejected when it exceeds `defaults.max_input_tokens`, so the engine does not silently translate only a truncated document. If this model setting is absent, `TITANOS_EMBEDDING_MAX_TOKENS` is used as the conservative input-token limit.

## Images

Provide exactly one of `image_base64`, `image_data_url`, `image`, or `image_buffer`. The first three accept inline base64; data URLs must have `data:image/...;base64,` form. The HLI-compatible `image_buffer` accepts bare base64 or one encoded-file byte array nested in a list. Remote image URLs and arbitrary server file paths are not fetched.

```json
{
  "operation": "ocr",
  "model": "printed-text",
  "image_data_url": "data:image/png;base64,...",
  "max_tokens": 128
}
```

OCR returns `{model,text,lines,task}`. The `image-to-text` task can also run a captioning checkpoint, so choose an OCR checkpoint when text extraction is the actual requirement. TrOCR is intended for cropped single text lines; it is not a complete multipage document layout system. Prompt-capable image-to-text checkpoints may accept `prompt`; TrOCR's vision-encoder-decoder adapter rejects prompts it cannot honor. [TrOCR model card](https://huggingface.co/microsoft/trocr-small-printed)

Object detection accepts optional `threshold` between 0 and 1, overriding `defaults.threshold` and then `TITANOS_DETECTION_THRESHOLD` (0.5). It returns `objects` with model label, label_id, score and pixel-coordinate `{xmin,ymin,xmax,ymax}` boxes, plus image dimensions and the applied threshold. Detection labels come from the selected checkpoint; they are not an unrestricted vocabulary.

Optional `nms:true` filters overlapping boxes within each native class. `nms_iou` selects the overlap threshold (0–1, default0.7); it takes effect when NMS is enabled. Each value follows request → model defaults → `TITANOS_DETECTION_NMS` / `TITANOS_DETECTION_NMS_IOU`. Explicit `nms:false` overrides an enabled default. NMS is off by default, preserving the original response shape and model outputs. It chooses higher scores first, breaks ties by original index, and returns retained boxes in their original order. It adds `postprocessing` with original/retained indices, applied threshold and method. It does not change scores, clip geometry, or determine that overlapping detections represent the same real object. Different classes never suppress each other.

Use `titanos detect detector image.png --nms --nms-iou 0.7`, `Client.detect(..., nms=True, nms_iou=0.7)`, or the same typed fields in HTTP/MCP. The Studio detection settings expose inheritance, explicit off/on and the threshold. A cancelled or expired postprocessing pass returns an error rather than partial success.

OCR and detection share the request deadline across executor waiting, image decoding and inference. Caller disconnects and MCP task cancellation signal the active backend, and a result completed after cancellation is rejected. Cancellation is cooperative: image decoding and one detection forward may finish before the worker can stop; no kernel preemption is implied. Image handles are closed after worker completion or failure. TrOCR models with learned position embeddings also check the effective new-token budget plus decoder prefix against the loaded decoder's fixed position capacity and actual table size. Sinusoidal TrOCR tables can grow and retain their existing generation path. An oversized budget is rejected before generation; it is never silently reduced. Omitted token limits still use model defaults and then the process default.

## Audio translation

`POST /v1/audio/translations` takes a `file` upload, explicit transcription `model`, optional input `language`, and `response_format=json|text|verbose_json`; undeclared multipart fields (for example `prompt`, `temperature`) are refused with 400 `unsupported_capability`. A multilingual Whisper checkpoint executes its native English translation task. English-only Whisper checkpoints and non-Whisper ASR models cannot perform this direct operation.

The transcription `language` hint follows the checkpoint: multilingual Whisper checkpoints receive it unchanged; English-only checkpoints (for example `whisper-tiny.en`) have no language tokens, so an English hint (`en`, `english`, regional `en-*`) is accepted and satisfied without being forwarded, and any other language is refused with a clear error instead of the library's generic rejection. Verbose responses still report the requested hint separately from the language the backend observed.

For a composed local pipeline, supply `translation_model` and its explicit `source_language` token ID. `language` is the ASR hint (for example `fr`); `source_language` is the NMT tokenizer code (for example `fra_Latn`). They are deliberately separate. `target_language` must name English; `translation_worker` optionally selects the NMT worker. Both models are checked before inference begins.

For composed ASR + NMT, an omitted `target_language` first inherits the
translation model's registered `defaults.target_language`, then falls back to
`eng_Latn`. An explicit request value wins. This permits fixed-direction
tokenizers such as Marian fr-en to use `en` without hardcoding that ID for
multilingual NLLB-style checkpoints. Direct Whisper translation remains
English and does not require an NMT target ID.

The JSON equivalent uses the generic endpoint:

```json
{
  "operation": "audio_translation",
  "model": "multilingual-whisper",
  "audio_base64": "...",
  "language": "fr"
}
```

This returns English `text`, the ASR model, optional translation model, `language:"en"` and task name. `max_tokens` controls generation where supported. Audio uploads are decoded locally using the same audio backend as transcription.

## Configuration and limits

Image limits use `TITANOS_MAX_INPUT_IMAGE_BYTES` and `TITANOS_MAX_INPUT_IMAGE_PIXELS`. Audio limits use `TITANOS_MAX_BODY_BYTES`, `TITANOS_MAX_AUDIO_SECONDS` and `TITANOS_MAX_AUDIO_SAMPLES`. Text/token limits use `TITANOS_MAX_TEXT_CHARS`, `TITANOS_MAX_PROMPT_CHARS`, `TITANOS_MAX_TOKENS` and `TITANOS_DEFAULT_MAX_TOKENS`. The default request deadline comes from `TITANOS_CLIENT_TIMEOUT` in seconds. Use `titanos settings` for the active settings and environment names.

The translation, OCR, detection and dedicated sentiment adapters currently use one device per worker. Independent replicas are supported. Model sharding and tensor parallelism are not implemented for them and are rejected rather than silently downgraded. Model directories must be local. Dedicated sentiment requires safe local weights and rejects remote checkpoint code and pickle; other adapters preserve the registry's `trust_remote_code` and `allow_pickle` decisions.

## Scope and evidence

These routes are an implemented local subset of HiveMind HLI's extended inference surface, kept name-compatible where a route exists in both projects. They are not a certification against any published standard. The per-field behaviour is generated into [PROTOCOL_DEVIATIONS.md](PROTOCOL_DEVIATIONS.md) and the per-route contract into [ROUTE_CONTRACT_MATRIX.md](ROUTE_CONTRACT_MATRIX.md).

VAD (`/v1/audio/vad`), speaker diarization (`/v1/audio/diarizations`) and overlap-preserving audio analysis (`/v1/audio/analysis`) are implemented typed operations; their model/hardware qualification is recorded in the capability matrix. Compound speaker identity workflows and ElevenLabs voice management (`/v1/voices*`) require additional adapters/workflows and are not advertised as working Titanos capabilities. Music generation is a separate job-only route, `POST /v1/music/generations` (equivalently `POST /api/jobs` with `task: "music"`), backed by the isolated ACE-Step runtime described in [Music](MUSIC.md); it is not an extension operation, and its model/hardware qualification is recorded separately. 3D generation has no adapter and no route.

Run `python tests/check_api_plus.py` for protocol/error checks. With local `t5-small`, `trocr-small-printed`, `yolos-tiny` checkpoints and a COCO test image, add `--models DIRECTORY --detection-image FILE --device cpu|cuda:0|mps --report FILE` for real model checks. If `whisper-base` and `mms-tts-fra` are also present, the optional French audio translation test runs. Model weights are not included in the engine package.

Native Windows CPU and CUDA tests demonstrated French text translation and typed detection of two cats. TrOCR recognized the words `HELLO WORLD` but removed their space; that is preserved as an observed quality limitation. Small Whisper translation quality also varies: retain the actual translated text and quality verdict, rather than treating HTTP 200 as semantic correctness. The French MMS speech fixture is validation-only under its model's CC-BY-NC-4.0 terms. These sample results do not establish model quality across arbitrary inputs or languages. [MMS model card](https://huggingface.co/facebook/mms-tts-fra)
