# Interfaces

Titanos exposes one local inference implementation through five interfaces. All of them share the same model registry, model lifecycle, limits, jobs and artifacts.

| Interface | Where | On this page |
|---|---|---|
| HTTP API | `http://127.0.0.1:11435` by default | OpenAI-compatible `/v1` routes, Titanos `/api` control routes and other wire shapes: [HTTP routes](#http-routes) |
| MCP | Streamable HTTP at `/mcp/`; stdio through `titanos mcp` | Official Python SDK, 72 typed tools: [MCP](#mcp) |
| CLI | `titanos <subcommand>` | 67 subcommands: [CLI](#cli) |
| Python client | `from titanos.client import Client` | Synchronous httpx wrapper: [Python client](#python-client) |
| Studio | `http://127.0.0.1:11435/` | Browser interface served by the same process: [Titanos Studio](#titanos-studio) |

This page is the map. The optional cloud broker, which can serve the same routes from cloud providers the operator configures, is described in [BROKER.md](BROKER.md). The per-operation binding of every route to its request and response models, MCP tools, CLI subcommands and client methods is generated into [ROUTE_CONTRACT_MATRIX.md](ROUTE_CONTRACT_MATRIX.md). Field-by-field parity with upstream protocols is generated into [PROTOCOL_DEVIATIONS.md](PROTOCOL_DEVIATIONS.md). Nothing here is model or hardware qualification; the [capability matrix](CAPABILITY_MATRIX.md) records what has actually run.

## Server, address and authentication

- `titanos serve` runs one server on `TITANOS_HOST:TITANOS_PORT`, default `127.0.0.1:11435`. `--host`, `--port`, `--threads`, `--limit-concurrency`, `--keep-alive-timeout`, `--log-level`, `--native-catalog`, `--role standalone|broker|managed` ([TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md)) and `--open` override or extend the environment. Running `titanos` with no arguments is the same as `titanos serve --open`.
- **Keyless mode is loopback-only.** Without `TITANOS_API_KEY` and without any minted key the server accepts only `localhost` and `127.0.0.1` as the Host header (anything else, including the bracketed IPv6 form `[::1]`, is answered 400 `Invalid host header`, so serving on `::1` needs the key) and refuses cross-site browser requests (403 `cross_site_refused`). `serve` refuses to bind a non-loopback address without an enforced key. Presented keys are ignored in keyless mode, so SDKs that send a placeholder keep working.
- **Keys.** `TITANOS_API_KEY` is the bootstrap key (every scope). `titanos keys create`, `POST /api/keys` and Studio mint scoped keys (`tk_<id>_<secret>`); once any key exists every route that is not public requires one (401 `authentication_error`, also from loopback), and a route the key's scopes do not cover answers 403 `scope_denied` before the body is read. Credentials: `Authorization: Bearer <key>` everywhere; `x-api-key` on `/v1/messages*`, `xi-api-key` on the ElevenLabs routes and `x-goog-api-key` on `/v1beta/*` (and each on `/native/*` and its vendor's bare native paths, [BROKER.md](BROKER.md#native-passthrough-and-websockets)); `?key=` is refused (401 `query_credential_refused`). Clients read the key from `TITANOS_API_KEY`. Scopes, limits and budgets: [BROKER_SECURITY.md](BROKER_SECURITY.md#inbound-keys).
- **Public routes** (GET only, no bearer, no request body): `/healthz`, `/v1/health/ready`, `/`, the Studio assets under `/ui/`, `/docs`, `/docs/assets/swagger-ui.css`, `/docs/assets/swagger-ui-bundle.js` and `/openapi.json`; in managed role also `/api/version`. Another method answers 405; a request body answers 400.
- **Errors** use one envelope everywhere: `{"error": {"message": ..., "type": ..., "param": null, "code": ...}}`. Unknown JSON fields are refused with 422 (a managed instance with `TITANOS_UNKNOWN_FIELDS=ignore` strips unknown top-level fields on the chat, completions, responses and embeddings routes and names them in `X-Titanos-Ignored`; [TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md)). Unknown multipart fields are refused with 400 `unsupported_capability`, naming the fields. Bodies above `TITANOS_MAX_BODY_BYTES` (64 MiB by default) answer 413. While the registry is being replaced, non-public routes answer 503 `registry_reloading` or `registry_unavailable`. Errors that the broker raises (routing, keys, budgets, upstream failures) use the native envelope of the route's own protocol (Anthropic, Gemini, ElevenLabs, Ollama) and the codes in [BROKER.md](BROKER.md#error-codes).
- **Schema and reference:** `/openapi.json` is generated in-process. `/docs` is Swagger UI served from packaged assets and requests nothing outside the server's own origin. ReDoc is not served.

## HTTP routes

### Health, discovery and documentation

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness: `{"status": "ok", "version": ..., "role": ...}` (public) |
| `GET /v1/health/ready` | Readiness (public): 200 `{"status": "ready", "version", "role", "models", "checks"}` only while the registry is loaded and not being replaced and, in broker role, no required (enabled) upstream has an open circuit breaker; otherwise 503 `{"status": "not_ready", "reason", "message", ...}` with `Retry-After: 1` (`registry_reloading`, `registry_unavailable`, `upstream_unhealthy`, `broker_config_unavailable`); [OPERATIONS.md](OPERATIONS.md#readiness-and-liveness) |
| `GET /api/version` | Managed role only: `{"name": "Titanos", "version": ..., "role": "managed"}` (public there, absent otherwise) |
| `GET /readyz` | 200 with the loaded model IDs; 503 while nothing is resident, which is the normal idle state |
| `GET /api/info` | `role`, `orchestrated` and a `broker` block (no URLs or secrets); the `settings` block needs the `settings:read` scope. Server identity, interface paths, `inference_contracts.require_loaded`, model-setup routes, registered tasks and models, job limits and the effective settings (never keys) |
| `GET /v1/models`, `GET /v1/models/{name}` | OpenAI-shaped catalog rows with Titanos `task` and `status` fields |
| `GET /api/models/loaded` | Resident models only |
| `POST /api/models/capabilities` | Static capability record of a registered model, without loading it |
| `GET /api/models/memory` | Dry-run memory estimate of every registered model the caller can see, never loading one (`models:read`). Each row has `backend`, `method`, `estimate_bytes` and `detail`, plus one `devices` row per GPU the model could use: its registered GPU UUIDs (with their `native.tensor_split` share of the estimate; one `nvidia-smi` does not list is `"unknown"`), else every GPU `nvidia-smi` lists (llama-cpp skips `TITANOS_EXCLUDE_GPU_UUIDS`). Each has `estimate_bytes`, `free_bytes` now (capped by `TITANOS_ADMISSION_FREE_MIB`), `available_bytes` (free minus `TITANOS_ADMISSION_RESERVE_MIB`) and `fits` (`true`, `false` or `"unknown"` without an estimate). `method: "gguf_estimate"` is the llama-cpp admission estimate for one GPU holding the whole start (weights, KV cache, compute margin). Other backends use `method: "weights_size_heuristic"`: the weight files on disk x 1.2 (only the `.safetensors` files when a folder has any), with no activations or runtime overhead counted. Without `nvidia-smi`, `free_memory.available` is false and `devices` is empty; `device: cpu` and `native:DEVICE` models have no device rows |
| `POST /api/models/details` | Runtime details of a model; may load its weights |
| `GET /api/workers` | Configured worker observations (replicas and native processes) |
| `GET /api/devices?backend=all\|nvidia\|pytorch\|openvino\|onnx\|system&refresh=false` | Hardware inventory; cached unless `refresh=true` |
| `GET /v1/extensions` | Extended operations, the registered models that match each one, and their JSON Schema |
| `GET /docs`, `GET /openapi.json` | API reference and schema (public) |

### Model registry, lifecycle and device assignment

| Route | Purpose |
|---|---|
| `POST /api/models/load`, `POST /api/models/unload` | Explicit lifecycle for `{"model": ..., "worker": ...}`; `worker` is optional |
| `POST /api/models/register` | Register a local checkpoint directory. Loader trust flags (`trust_remote_code`, `allow_pickle`) and native GGUF options are refused here and need `titanos add` |
| `POST /api/models/reload` | Re-read the registry file after an external edit |
| `DELETE /api/models/registrations/{name}` | Remove a registration; weights stay on disk. Refused (409) while a worker assignment names the model |
| `GET /api/assignments/devices?model=&refresh=` | Stable device identities a model may be assigned to (`GPU-`, `HIP-`, `XPU-`, `OV-` identifiers and `cpu`) |
| `GET /api/assignments/{model}`, `POST /api/assignments` | Read or save one model's device policy (`auto` or `cpu`) or its ordered device groups in `single`, `model_sharding` or `tensor_parallel` mode |

### Model providers, acquisitions and native runtime setup

| Route | Purpose |
|---|---|
| `GET /api/model-providers` | Provider descriptors: `huggingface`, `github-sherpa`, `github-sherpa-diarization`, `ace-step` |
| `GET /api/model-providers/{provider}/search?task=&query=&limit=` | Task-first model discovery |
| `POST /api/model-providers/inspect` | Exact source and revision metadata without downloading |
| `POST /api/model-providers/import` | Normalize a permitted model-page link and inspect a recognized provider |
| `POST /api/acquisitions/plan` | Acquisition plan pinned to an exact source revision |
| `POST /api/acquisitions` (202, optional `Idempotency-Key`) | Start a durable download; the model is registered only by activation |
| `GET /api/acquisitions`, `GET /api/acquisitions/{id}` | List and inspect acquisitions |
| `DELETE /api/acquisitions/{id}`, `POST /api/acquisitions/{id}/retry` (202) | Cancel or retry |
| `POST /api/acquisitions/{id}/activate` | Verify integrity, register and optionally load (`accepted_license_sha256`, `load`, `runtime_profile`) |
| `DELETE /api/acquisitions/{id}/weights` | Delete an acquisition's downloaded weights |
| `GET /api/runtime-profiles?refresh=` | Packaged llama.cpp runtime profiles compatible with this host |
| `POST /api/runtime-setups/plan`, `POST /api/runtime-setups` (202) | Plan or start a checksum-pinned, server-owned runtime install |
| `GET /api/runtime-setups`, `GET /api/runtime-setups/{id}`, `DELETE /api/runtime-setups/{id}`, `POST /api/runtime-setups/{id}/retry` (202) | Runtime setup records: list, inspect, cancel, retry |

The runtime routes exist whenever a runtime catalog resolves (`serve --native-catalog`, then `TITANOS_NATIVE_CATALOG`, then the catalog packaged with Titanos), which is the default. Details: [MODEL_PROVIDERS.md](MODEL_PROVIDERS.md) and [NATIVE.md](NATIVE.md).

### OpenAI-compatible inference (`/v1`)

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat and vision. SSE with `"stream": true`. `"require_loaded": true` refuses a cold model with 409 `model_not_loaded` instead of loading it. `tools`, `tool_choice` and `response_format` are accepted only for llama.cpp models |
| `POST /v1/completions` | Text completion over the same engine |
| `POST /v1/responses` | Stateless Responses: text or inline-image input, text output, SSE streaming; llama.cpp models pass through with their native tool support ([RESPONSES.md](RESPONSES.md)) |
| `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel`, `GET /v1/responses/{id}/input_items` | Stored responses at a cloud upstream, for the key that created them only; there is no local response store (404 `response_not_found`) |
| `POST /v1/embeddings` | Embeddings with an explicit `truncate` policy |
| `POST /v1/audio/speech` | Text to speech; `response_format` is `wav` or `pcm`; `stream_format` `audio` streams the audio as it is made on qwen3-tts ([QWEN_TTS.md](QWEN_TTS.md)) |
| `POST /v1/audio/transcriptions` (multipart) | Speech to text; `response_format` is `json`, `text` or `verbose_json`; optional VAD gate ([AUDIO.md](AUDIO.md)) |
| `POST /v1/audio/translations` (multipart) | Speech to English with a multilingual Whisper checkpoint, or composed ASR plus NMT ([API_PLUS.md](API_PLUS.md)) |
| `POST /v1/images/generations` | Synchronous image generation (`b64_json` or `url`) |
| `POST /v1/moderations` | Moderation through a cloud OpenAI upstream; there is no local classifier (503 `classifier_unavailable`) |
| `POST /v1/images/edits` (multipart) | Image edit with an optional binary-alpha mask, or an instruction edit ([image-masks.md](image-masks.md), [INSTRUCTION_EDIT.md](INSTRUCTION_EDIT.md)) |
| `POST /v1/videos` (202), `GET /v1/videos`, `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content`, `DELETE /v1/videos/{id}` | Standard video-job shape over the Titanos job store ([VIDEO_JOBS.md](VIDEO_JOBS.md)) |

### Titanos jobs, artifacts and typed inference

| Route | Purpose |
|---|---|
| `POST /api/jobs` (202, optional `Idempotency-Key` of 1 to 128 characters) | Durable asset job; `task` is `image`, `speech`, `video` or `music` |
| `GET /api/jobs?limit=&status=&task=&idempotency_key=`, `GET /api/jobs/{id}`, `DELETE /api/jobs/{id}` | List (filter by status, task, or the Idempotency-Key a job was submitted with), inspect, cancel; the cancel answer carries `cancel_accepted`, the record is kept after cancellation (contract: VIDEO_JOBS.md) |
| `POST /v1/music/generations` (202) | Music job over an ACE-Step bundle; job-only, no synchronous route. Real music generation is not yet qualified ([MUSIC.md](MUSIC.md)) |
| `POST /api/video/jobs` (202), `GET /api/video/jobs`, `GET` and `DELETE /api/video/jobs/{id}` | Video-only view of the same job store |
| `POST /v1/videos/generations` | Synchronous video generation; the response waits for the MP4 |
| `GET /v1/files/{name}` | Generated `png`, `mp4` and `wav` artifacts by their Titanos identifier; they expire after `TITANOS_ARTIFACT_TTL_SECONDS` (24 hours by default) |
| `POST /api/infer` | Arbitrary tensor inference on prepared ONNX or OpenVINO graphs; an input is `{dtype, shape, data}` with flat list data or `{dtype, shape, b64}` with its little-endian bytes in base64, and `output_encoding` `base64` answers outputs the second way. A `uint8` input in either form (`b64`, or `data` for symmetry) may add `"cast": "float32"`, which converts it to float32 before the graph's dtype check, and optionally `"normalize": {"mean": [...], "std": [...], "axis": n}`, which computes `(u8 - mean) / std` in float64 along that axis (negative counts from the end; one finite number per index, no zero std) with one final cast to float32, bit-identical to a client doing the same in numpy; `normalize` without `cast`, a `cast` other than `float32` or on another dtype, a `normalize` without exactly those three keys, or one whose result leaves float32's finite range, are a 400 naming `cast` or `normalize`, any other key is refused as before, and the element limit applies to the uint8 source. `output_views` `{output: {"slice": [[start, stop], ...]}}` returns that slice of an output, one integer-or-null pair per axis and no step, before encoding; an unknown output, a wrong axis count or a malformed view is a 400 (an `output_views` or a view that is not an object fails request validation, 422), and the element limit applies to the full outputs |
| `POST /v1/translations` | Text translation |
| `POST /v1/images/ocr`, `/v1/images/detect`, `/v1/images/depth`, `/v1/images/segment` | OCR or captioning, object detection, relative depth and prompted segmentation on one inline image ([TESSERACT.md](TESSERACT.md), [DEPTH.md](DEPTH.md), [SEGMENTATION.md](SEGMENTATION.md)) |
| `POST /v1/sentiment`, `POST /v1/sentiment/compose` | Text sentiment; classifier-versus-LLM composition ([SENTIMENT.md](SENTIMENT.md)) |
| `POST /v1/rerank`, `POST /v1/retrieval/answer` | Cross-encoder rerank; embedding, rerank and llama.cpp answer over supplied passages ([RETRIEVAL.md](RETRIEVAL.md), [RETRIEVAL_ANSWER.md](RETRIEVAL_ANSWER.md)) |
| `POST /v1/classifications/text-pairs` | Premise and hypothesis classification with the model's native labels ([PAIR_CLASSIFICATION.md](PAIR_CLASSIFICATION.md)) |
| `POST /v1/audio/vad`, `/v1/audio/diarizations`, `/v1/audio/analysis` | Speech activity, anonymous speaker intervals, overlap-preserving audio analysis ([AUDIO_ANALYSIS.md](AUDIO_ANALYSIS.md), [ASR_VAD_GATING.md](ASR_VAD_GATING.md)) |
| `POST /v1/extensions/infer` | One typed envelope for the operations above. `operation` is one of `translation`, `ocr`, `detection`, `depth`, `segmentation`, `audio_translation`, `audio_transcription`, `vad`, `diarization`, `audio_analysis`, `sentiment`, `sentiment_compose`, `rerank`, `retrieval_answer`, `pair_classification` |

The extended routes are described in [API_PLUS.md](API_PLUS.md).

### Cloud broker, keys and audit

| Route | Purpose |
|---|---|
| `GET /api/broker`, `POST /api/broker/reload`, `GET /api/broker/profiles` | Broker summary, reload of `broker.json`, shipped upstream profiles |
| `GET`, `PUT`, `DELETE /api/broker/upstreams/{id}`; `POST .../test`, `GET .../health`, `POST .../breaker/reset`; `GET /api/broker/upstreams` | Upstreams, probes and breakers |
| `GET /api/broker/credentials`; `PUT`, `DELETE /api/broker/credentials/{id}`; `POST .../rotate`, `POST .../validate` | Write-only provider credentials (metadata and fingerprints only) |
| `/api/broker/models[/{name}]`, `/api/broker/virtual-models[/{name}]`, `/api/broker/policy`, `/api/broker/advisors[/{id}]`, `/api/broker/prices` | Aliases, virtual models, cloud policy, advisors, operator prices |
| `GET /api/broker/catalog`, `POST /api/broker/explain`, `GET /api/broker/decisions`, `GET /api/broker/usage`, `GET /api/broker/metrics` | Catalog, dry-run routing (zero egress), decisions, usage and counters |
| `GET`, `POST /api/keys`; `GET`, `PATCH /api/keys/{id}`; `POST /api/keys/{id}/revoke`, `POST /api/keys/{id}/rotate`; `GET /api/keys/self` | Key management; `self` shows the caller's own scopes and remaining budget |
| `GET /api/audit`, `GET /api/audit/verify` | Hash-chained audit log and its verification |

Each route needs its scope (`broker:read`, `broker:admin`, `credentials:write`, `keys:admin` or `audit:read`); see [BROKER.md](BROKER.md) and [BROKER_SECURITY.md](BROKER_SECURITY.md).

### Other wire shapes

| Shape | Routes | Notes |
|---|---|---|
| Ollama | `GET /api/version`, `GET /api/tags`, `GET /api/ps`, `POST /api/show`, `POST /api/chat`, `POST /api/generate`, `POST /api/embed`, `POST /api/embeddings`; `POST /api/pull`, `POST /api/create`, `POST /api/copy`, `DELETE /api/delete` (typed 400 refusals) | JSON and NDJSON streaming. `/api/tags` lists registered models with `details.format` (backend) and `details.family` (task); `/api/show` adds the registry entry and the passive inspection without loading; `/api/ps` returns Titanos catalog rows for loaded Python-engine and native models; `keep_alive` and `options.num_ctx` are accepted (`keep_alive: 0` unloads after the answer); the library routes refuse with the Titanos acquisition path |
| Anthropic Messages | `POST /v1/messages`, `POST /v1/messages/count_tokens` | Text and inline base64 images; tool calls on llama.cpp models, nonstreaming and streaming (`tool_use` blocks with `input_json_delta`); token counts for llama.cpp models (and Anthropic upstreams) |
| ElevenLabs | `POST /v1/text-to-speech/{voice_id}`, `POST /v1/text-to-speech/{voice_id}/stream`, `POST /v1/speech-to-text`, `GET /v1/voices`, `GET /v1/voices/{voice_id}` | Local speech is WAV only; `model_id` is a Titanos model name and `voice_id` a voice alias; `/v1/voices` lists local voices |
| Gemini | `GET /v1beta/models`, `GET /v1beta/models/{model}`, `POST /v1beta/models/{model}:{generateContent,streamGenerateContent,embedContent,batchEmbedContents,countTokens}` | Local chat and embedding models are answered through the OpenAI routes; streaming needs `alt=sse` |
| MCP | `/mcp/` | Streamable HTTP, see [MCP](#mcp) |

With a configured key the Anthropic, ElevenLabs and Gemini shapes accept their native key headers (`x-api-key`, `xi-api-key`, `x-goog-api-key`) as well as `Authorization: Bearer`; stock Ollama clients send no key. When the broker is configured these routes can also be served by cloud upstreams ([BROKER.md](BROKER.md#surfaces-and-translation)). Field-level behaviour: [PROTOCOL_COMPATIBILITY.md](PROTOCOL_COMPATIBILITY.md) and [PROTOCOL_DEVIATIONS.md](PROTOCOL_DEVIATIONS.md).

## MCP

Titanos uses the official Model Context Protocol Python SDK, pinned `mcp==2.2.0` and installed with Titanos. The server is named `Titanos` and uses the same registry, limits, jobs and artifacts as HTTP.

| Transport | How |
|---|---|
| Streamable HTTP | `http://127.0.0.1:11435/mcp/` inside `titanos serve`, behind the same bearer |
| stdio | `titanos mcp --url http://127.0.0.1:11435` bridges a host application to an already-running server; it reads `TITANOS_URL` and `TITANOS_API_KEY` from its environment |

Neither transport starts a second inference runtime. Host configuration for stdio launchers:

```json
{"command": "titanos", "args": ["mcp", "--transport", "stdio", "--url", "http://127.0.0.1:11435"]}
```

The 72 tools, grouped:

| Group | Tools |
|---|---|
| Discovery | `server_info`, `list_models`, `model_capabilities`, `list_devices`, `list_workers`, `list_runtime_profiles`, `list_runtime_setups` |
| Model sources | `list_model_providers`, `discover_models`, `inspect_model_source`, `import_model_page`, `plan_model_acquisition` |
| Acquisition | `submit_model_acquisition`, `list_model_acquisitions`, `get_model_acquisition`, `cancel_model_acquisition`, `retry_model_acquisition`, `activate_model_acquisition` |
| Runtime setup | `plan_runtime_setup`, `prepare_runtime`, `get_runtime_setup`, `cancel_runtime_setup`, `retry_runtime_setup` |
| Device assignment | `list_assignment_devices`, `get_model_assignment`, `save_model_assignment` |
| Lifecycle | `load_model`, `unload_model`, `model_details`, `register_model`, `reload_registry`, `unregister_model` |
| Text and vision | `chat`, `complete`, `responses`, `messages`, `count_tokens`, `moderate` |
| Embeddings, retrieval and prepared graphs | `embed`, `rerank`, `retrieval_answer`, `classify_pairs`, `infer_tensor` |
| Audio | `speak`, `transcribe`, `translate_audio`, `vad`, `diarize`, `analyze_audio` |
| Extended text | `translate`, `sentiment`, `compose_sentiment` |
| Images and video | `generate_image`, `edit_image`, `estimate_depth`, `segment_image`, `generate_video` |
| Typed envelope | `extension_infer` over the 15 extension operations; OCR and detection are reached this way |
| Jobs and artifacts | `submit_job` (image, speech, video, music), `list_jobs`, `get_job`, `cancel_job`, `read_artifact` |
| Cloud broker | `broker_status`, `broker_catalog`, `broker_explain`, `broker_decisions`, `broker_usage`, `list_upstreams`, `upstream_health`, `test_upstream`, `whoami`; no tool writes credentials, keys or policy |

Resources: `titanos://info`, `titanos://models`, `titanos://jobs`, `titanos://acquisitions`, `titanos://extensions`, `titanos://schema`. Templates: `titanos://models/{model}/capabilities`, `titanos://jobs/{id}`, `titanos://acquisitions/{id}` and `titanos://artifacts/{id}.png`, `.wav`, `.mp4`. Prompts: `create_asset`, `diagnose_model`.

Inference tools take the HTTP request model as `payload`; `transcribe` and `edit_image` take flat arguments. Media go inline as base64. Every call returns one complete result (no token streaming); `speak` returns an audio content block and `read_artifact` an embedded binary resource. Host details, limits and embedding in another ASGI application: [MCP.md](MCP.md).

## CLI

`titanos [--config PATH] [--version] <subcommand> ...`; `titanos <subcommand> --help` lists the options. Subcommands that talk to a server accept `--url` (default `TITANOS_URL`, then `http://127.0.0.1:11435`) and `--timeout`.

| Group | Subcommands |
|---|---|
| Local setup, no server | `init`, `settings [--json]`, `add NAME DIR --task T [--backend B]`, `pull NAME REPO --task T`, `search [QUERY]`, `inspect REPO`, `login`, `logout`, `list`, `remove NAME`, `capabilities NAME`, `worker NAME --models ...`, `remove-worker NAME`, `devices [--backend B] [--refresh]`, `pin-devices`, `doctor [--check]`, `prepare SOURCE NAME --output DIR --backend onnx\|openvino`, `native-install --profile P`, `upgrade --catalog C`, `pair-manifest`, `serve` |
| Server inspection and lifecycle | `info`, `models`, `ps`, `workers`, `load NAME`, `unload NAME`, `details NAME`, `mcp [--url U] [--transport stdio]` |
| Inference | `run NAME [PROMPT] [--stream] [--json] [--system S] [--history FILE] [--messages FILE] [--image FILE]`, `embed NAME TEXT...`, `speak NAME TEXT --output FILE`, `transcribe NAME FILE`, `translate-audio NAME FILE`, `translate NAME TEXT --target-language L`, `image NAME PROMPT [--output FILE]`, `edit NAME PROMPT --image FILE`, `video NAME PROMPT [--image FILE] [--async]`, `music NAME PROMPT [--mode instrumental\|song] [--async]`, `ocr NAME IMAGE`, `detect NAME IMAGE`, `depth NAME IMAGE`, `segment NAME IMAGE --box X0 Y0 X1 Y1` or `--points FILE`, `sentiment NAME TEXT`, `sentiment-compose CLASSIFIER LLM TEXT`, `rerank NAME QUERY [DOCUMENT...]`, `retrieval-answer LLM QUERY --documents-file F --embedding-model E --rerank-model R`, `classify-pairs NAME [--pairs-file F]`, `vad NAME FILE`, `diarize NAME FILE`, `analyze-audio NAME FILE --diarization-model D` |
| Jobs | `jobs list`, `jobs submit REQUEST.json`, `jobs get ID`, `jobs wait ID [--output FILE]`, `jobs cancel ID` |
| Operations | `service install\|start\|stop\|uninstall\|status [--name N] [--system] [--dry-run] [--json] [--wait SECONDS]` (a Scheduled Task, systemd unit or launchd plist that runs `serve` with `TITANOS_HOME`; `install` checks the registry and the executable first and, like `start`, waits for `GET /healthz`), `hivemind register-service GATEWAY --job-type JT [--port P] [--url U] [--key-env VAR]` (one `POST /register-service` with a bearer from `TITANOS_API_KEY` or the named variable); [OPERATIONS.md](OPERATIONS.md) |
| Cloud broker, keys and audit | `broker status\|profiles\|reload\|catalog\|explain\|decisions\|usage`, `broker upstream add\|list\|show\|enable\|disable\|remove\|test\|health\|reset-breaker`, `broker credential set\|list\|show\|validate\|rotate\|remove`, `broker alias`, `broker virtual`, `broker policy`, `broker advisor`, `broker prices`, `broker live-smoke`, `keys create\|list\|show\|update\|revoke\|rotate\|whoami\|reset`, `audit [verify]`, `hivemind-manifest`. They edit `TITANOS_HOME` files offline or call a server with `--url`; no option takes a secret value |
| Model acquisition and runtime | `providers`, `discover PROVIDER --task T`, `source-inspect PROVIDER ID`, `import-link URL`, `acquire NAME PROVIDER ID --task T --backend B [--wait] [--activate]`, `acquisitions list\|get\|cancel\|retry\|wait\|activate\|delete-weights [ID]`, `runtime-profiles`, `runtime-prepare PROFILE [--plan] [--wait]`, `runtime-setups list\|get\|cancel\|retry\|wait [ID]`, `assignment-devices [--model M]`, `assignment get\|save MODEL [--mode M] [--group ...] [--policy auto\|cpu]` |

Inference subcommands and `run` accept `--location` and `--target` (sent as `X-Titanos-Location` and `X-Titanos-Target`) and print the `X-Titanos-*` provenance to standard error with `--verbose`. `run` without a prompt reads standard input, or opens an interactive loop on a terminal (`/exit` leaves it); `run --stream --json` prints JSON Lines for agents. `serve` options are listed under [Server, address and authentication](#server-address-and-authentication).

## Python client

`titanos.client.Client` is a small synchronous httpx wrapper; importing it loads no model framework. `base_url` defaults to `TITANOS_URL`, then `http://127.0.0.1:11435`. `api_key=None` reads `TITANOS_API_KEY`; `api_key=""` means a keyless loopback server.

```python
from titanos.client import Client, image_part

with Client("http://127.0.0.1:11435", api_key="") as client:
    print(client.info()["version"])
    print([m["id"] for m in client.models()["data"]])

    reply = client.chat("chat", [{"role": "user", "content": "Explain local inference in two sentences."}], max_tokens=96)
    print(reply["choices"][0]["message"]["content"])

    for event in client.chat("chat", [{"role": "user", "content": "Hello"}], stream=True):
        delta = event["choices"][0]["delta"].get("content")
        if delta:
            print(delta, end="", flush=True)

    # Vision: attach a local image as an inline part
    parts = [{"type": "text", "text": "Describe this."}, image_part("photo.png")]
    reply = client.chat("vision", [{"role": "user", "content": parts}])

    # Durable asset job: submit, wait, save
    job = client.submit_job("image", "painter", prompt="a lighthouse at dusk", size="512x512")
    record = client.wait_job(job["id"], timeout=600)
    if record["status"] == "completed":
        client.save_asset(record["result"]["data"][0], "lighthouse.png")
```

Method families: lifecycle `load`, `unload`, `details`, `capabilities`; text `chat`, `embeddings`, `rerank`, `retrieval_answer`, `classify_pairs`, `translate`, `sentiment`, `compose_sentiment`; audio `speech`, `transcribe`, `translate_audio`, `vad`, `diarize`, `analyze_audio`; images `image`, `edit`, `ocr`, `detect`, `depth`, `segment`, `extension`; jobs `video`, `submit_job`, `submit_video`, `submit_music`, `jobs`, `job`, `cancel_job`, `wait_job`, `save_asset`; acquisition `providers`, `discover_models`, `inspect_model_source`, `import_model_link`, `plan_acquisition`, `submit_acquisition`, `acquisitions`, `acquisition`, `cancel_acquisition`, `retry_acquisition`, `activate_acquisition`, `delete_acquired_weights`, `wait_acquisition`; runtime `runtime_profiles`, `plan_runtime_setup`, `submit_runtime_setup`, `runtime_setups`, `runtime_setup`, `cancel_runtime_setup`, `retry_runtime_setup`, `wait_runtime_setup`; assignment `assignment_devices`, `assignment`, `save_assignment`; discovery `info`, `models`; other wire shapes `messages`, `count_tokens`, `moderations`, `response`, `delete_response`, `cancel_response`, `response_input_items`, `gemini_generate`, `gemini_embed`, `gemini_count_tokens`, `elevenlabs_speech`, `speech_to_text`, `voices`; broker `broker_status`, `broker_profiles`, `broker_catalog`, `broker_explain`, `broker_decisions`, `broker_usage`, `broker_metrics`, `broker_reload`, `upstreams`, `upstream`, `put_upstream`, `delete_upstream`, `test_upstream`, `upstream_health`, `reset_breaker`, `credentials`, `set_credential`, `rotate_credential`, `validate_credential`, `delete_credential`, `aliases`, `put_alias`, `delete_alias`, `virtual_models`, `put_virtual_model`, `delete_virtual_model`, `policy`, `set_policy`, `advisors`, `put_advisor`, `test_advisor`, `prices`, `put_prices`; keys and audit `keys`, `create_key`, `key`, `update_key`, `revoke_key`, `rotate_key`, `whoami`, `audit`, `audit_verify`; `provenance(response)` and `last_provenance` for the `X-Titanos-*` headers; `request(method, route, **kwargs)` for anything else; and `close`, or use the client as a context manager.

`Client(..., location=..., target=...)` sets default hint headers, and `_location` and `_target` override them per call. The 118 public methods are listed in [ROUTE_CONTRACT_MATRIX.md](ROUTE_CONTRACT_MATRIX.md). Failures raise `TitanosError` with `status_code`, `code` and the server's `error` object. Redirects are never followed and artifact downloads must come from the configured origin. For several servers, `titanos.pool.Pool` hands out explicit leases ([MULTIHOST_CLIENT.md](MULTIHOST_CLIENT.md)).

## Titanos Studio

Studio is the browser interface served at `/` by `titanos serve`; `--open` launches the default browser once `/healthz` answers. It is plain HTML, CSS and JavaScript from `src/titanos/web` with no build step and no external requests, and it uses only the HTTP routes above.

| Aspect | Studio |
|---|---|
| Modes | **Simple** (default): task-first model setup, prompt, result and one obvious action. **Advanced**: local-folder registration, generation controls, runtime details, device assignments and links to `/docs` and `/openapi.json` |
| Languages | English, Español, 日本語, 简体中文. Mode and language persist in browser storage; the API key stays in the tab's memory |
| Tasks | Chat and vision (streamed), embeddings, image generation and editing, speech, video, music, transcription with speech translation and speaker analysis, OCR, detection, depth, segmentation, text translation, sentiment, pair classification, rerank, VAD, diarization, and a document-answer panel |
| API-only | Arbitrary tensor models (`/api/infer`) |

Details: [STUDIO_USAGE.md](STUDIO_USAGE.md), [STUDIO_BACKENDS.md](STUDIO_BACKENDS.md), [STUDIO_LOCALIZATION.md](STUDIO_LOCALIZATION.md), [STUDIO_DESIGN.md](STUDIO_DESIGN.md).

## Not implemented

- Inference never downloads a model; acquisition is a separate, explicit workflow.
- No `/redoc`, no transcription streaming, no synchronous music route.
- Ollama `/api/push` and `/api/blobs/*` are absent; `/api/pull`, `/api/create`, `/api/copy` and `/api/delete` are typed 400 refusals naming Titanos's acquisition path; `/api/version` (`{name, version, role}`, public only in managed role) and `/api/show` are implemented ([PROTOCOL_COMPATIBILITY.md](PROTOCOL_COMPATIBILITY.md)).
- ElevenLabs `/with-timestamps`, websocket, `/v1/user*` and `/v1/history` are absent.
- On local models, Anthropic `thinking`, `metadata`, `service_tier`, `betas`, `container`, `mcp_servers` and `output_format` are refused (422); an Anthropic cloud upstream receives the provider's own fields on passthrough.
- Unknown request fields are refused on local models, never approximated. Training, hardened multi-tenant isolation, automatic failover across machines and cross-machine sharding are outside every interface described here. Within one server, scoped keys and the broker's virtual-model fallback between configured targets exist ([BROKER.md](BROKER.md)).
