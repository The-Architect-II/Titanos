# Option parity and effective configuration

Titanos exposes a portable set of controls over several backends. It is not an
unrestricted llama.cpp command-line proxy or a complete Ollama replacement.
Unknown request fields are rejected (422 by schema on JSON routes, except the identity, cache, storage and reasoning hints that the chat, Responses and Messages routes accept, ignore and name in `X-Titanos-Ignored`, listed in [PROTOCOL_COMPATIBILITY.md](PROTOCOL_COMPATIBILITY.md#agent-harness-fields-on-the-local-routes); 400 `unsupported_capability` on the multipart transcription, audio-translation and image-edit routes; 400 on the manually parsed `/api/embed`/`/api/embeddings`). Managed llama.cpp model defaults that
cannot reach a supported request now fail explicitly instead of disappearing.

This map was checked against Titanos source and official documentation on
2026-09-22 and against the b10809 CPU `llama-server` binary on 2026-10-05 (the
OpenAI sampling, embeddings and speech fields below). llama.cpp comparisons use
**b10809**, matching the retained CPU runtime evidence; Ollama comparisons
describe the current official API and Modelfile documentation, not an executed
Ollama installation.

## Which value wins

For supported generation fields, precedence is **explicit request → registered
model default → process setting/default**. `TITANOS_*` settings are read at
process/configuration initialization; changing a shell variable does not edit a
running server. Persisted model/native values remain explicit values when a
configuration is reloaded. `titanos settings --json` describes the invoked
process; `/api/info` describes the serving process.

The CLI merges `--options request.json` first, then explicit flags such as
`--max-tokens`, `--temperature`, `--seed` and `--worker`. Those become request
values. A valid zero or false stays explicit: temperature/top-k/min-p/seed zero
and `parallel_tool_calls:false` do not fall through to model defaults. Explicit
null on an optional field also suppresses inheritance; null on a required
numeric field such as `max_tokens` is invalid. The optional
`max_completion_tokens:null` retains the existing unset-alias behavior, so it
does not inherit the model's output-token default.

[`request_defaults.py`](../src/titanos/request_defaults.py) is shared by public
generation and internal native completion. [`cli.py`](../src/titanos/cli.py)
contains `request_options`/`run_chat`; [`settings.py`](../src/titanos/settings.py)
contains startup parsing and validation. `model`, `worker`, `task` and
`require_loaded` are deliberately not inherited from model generation defaults.
Sentiment/retrieval policy defaults remain in Titanos's composition layer.

Native `/v1/responses` applies that same merge to its typed Responses payload
before forwarding. Its output field is `max_output_tokens`; registered model
defaults named `max_tokens`, `max_new_tokens`, `max_output_tokens` or
`max_completion_tokens` map to that field. An explicit request budget wins, and
a null optional `max_completion_tokens` model alias remains unspecified.
Required numeric request fields still reject null. Existing tool/text/history,
`store:false` and streaming fields stay in the native Responses envelope.
Defaults outside that protocol's supported fields are rejected before proxying;
Chat-only controls do not become Responses features merely by being model
defaults. The non-native Responses-to-Chat path retains its existing behavior.

## Generation controls

Ollama documents model parameters and per-request `options`; Titanos translates
only the supported subset below. Unsupported Ollama fields are errors, not
silently accepted compatibility promises. See the official
[Modelfile parameter reference](https://docs.ollama.com/modelfile#valid-parameters-and-values)
and [generate request](https://docs.ollama.com/api/generate).

| Intent | Ollama name | Titanos request/default | Effective behavior and boundary |
|---|---|---|---|
| Output budget | `num_predict` | `max_tokens`; chat also accepts `max_completion_tokens` | Default 256, operator ceiling `TITANOS_MAX_TOKENS=8192`. Raise the ceiling and select a positive budget. Model defaults also recognize `max_new_tokens`/`max_output_tokens`; `max_completion_tokens` now works for completion/Ollama-generate too. This is separate from input/context capacity. |
| Temperature | `temperature` | `temperature` | Request/model override `TITANOS_DEFAULT_TEMPERATURE` (0.7). Portable schema range 0–2 is a Titanos policy, not a physical llama.cpp limit. Zero selects greedy behavior in Transformers/GenAI. |
| Probability mass | `top_p` | `top_p` | Request/model override `TITANOS_DEFAULT_TOP_P` (1.0); range `(0,1]`. It is not applied to deterministic Transformers/GenAI sampling. |
| Top-k / min-p | `top_k`, `min_p` | Same names | Request or model default; otherwise backend default. Zero is preserved. No dedicated environment default exists for these fields. |
| Repetition | `repeat_penalty` | `repeat_penalty` | Positive value; maps to Transformers/GenAI `repetition_penalty`. `repeat_last_n` is not exposed by Titanos and now produces an actionable error if placed in managed-native defaults. |
| Presence / frequency penalties | `presence_penalty`, `frequency_penalty` | Same names (OpenAI range -2..2) on `/v1/chat/completions`, `/v1/completions` and Ollama `options` | Forwarded to a managed llama.cpp model (b10809 honours both; verified against the CPU binary). Transformers and OpenVINO GenAI have no generation-config equivalent: a nonzero value answers 400 `unsupported_capability` naming the field and the backend; `0` is the OpenAI default and is served. |
| Token log probabilities | — | Chat `logprobs` (bool) + `top_logprobs` (0..20); completions `logprobs` (0..5) | Forwarded to llama.cpp, which returns `choices[].logprobs` (also in SSE chunks); on `/v1/completions` llama.cpp's chat-shaped `logprobs.content[]` is rewritten to the legacy completions shape the stock SDK types (`tokens`, `token_logprobs`, `top_logprobs` maps, `text_offset`), in the body and in every chunk. In-process backends: 400 naming the field; `false`/`0` are served. |
| Logit bias | — | `logit_bias` (token id -> bias, at most 300 entries) | Forwarded to llama.cpp (token ids or token text keys). Transformers maps integer-id keys to `sequence_bias`; a non-integer key is a 400 naming `logit_bias`. OpenVINO GenAI refuses it by its own field check. |
| Seed | `seed` | `seed` | Optional unsigned 32-bit portable seed (0..2^32-1; out of range is 400 on the Ollama routes because `Options.seed` is re-validated by the chat schema, 422 on `/v1/*`); `0` is a fixed seed, not Ollama's unset meaning. It is not a cross-runtime reproducibility guarantee. Top-level `images` is refused (400) on `/api/generate` and `/api/chat`; only per-message images are mapped. |
| Stop strings | `stop` | `stop` | String/list; operator bounds `TITANOS_MAX_STOP_SEQUENCES` (4) and `TITANOS_MAX_STOP_CHARS` (128). These bounds can be changed. |
| Streaming / result count | `stream` | `stream`, `stream_options.include_usage`, `n` | Ollama compatibility defaults streaming on; OpenAI-style Titanos requests default off. `n` (1..`TITANOS_MAX_CHOICES`, default 8) is forwarded to llama.cpp, which returns `n` choices; in-process backends return one choice and answer 400 naming `n` for `n >= 2`. |
| System/history/template | `system`, `TEMPLATE`, `MESSAGE` | Message roles, CLI `--system`, `--messages`/`--history` | CLI system text is added only when starting an empty history. No arbitrary native template or template-kwargs override is exposed. |
| Tools / JSON output | `tools`, `format` | Chat `tools`, `tool_choice`, `parallel_tool_calls`, `response_format` | Managed llama.cpp supports these interfaces on `/v1/chat/completions` and on `/api/chat`, where the model emits id-less Ollama `tool_calls` and Ollama-format tool-result turns (`role: tool` with `tool_name`) are bound to the pending call of that name (400 when there is none to answer), so a stock Ollama client completes the tool loop, nonstream and with `stream: true` (the argument fragments llama.cpp streams are gathered into whole `tool_calls` with object arguments, emitted in one frame ahead of the final frame). Other adapters reject unsupported agent/structured requests. Use `/api/chat`, not Titanos's Ollama `/api/generate`, for this subset. Model ability is separate. |

Titanos does **not** expose unrestricted native sampling fields such as
`repeat_last_n`, typical sampling, Mirostat, dynamic temperature, sampler order
or speculative-generation options through its public schema. A field existing
in llama.cpp does not make it a supported Titanos option. Native default-key
validation reports these deviations before worker selection/load. Other
adapters may retain their documented backend-specific model defaults; do not
transfer those dictionaries blindly between backends. See
[`protocols.py`](../src/titanos/protocols.py),
[`server.py`](../src/titanos/server.py) (`Generation`/`Chat`/`Completion`),
[`openai_params.py`](../src/titanos/openai_params.py) (per-backend forwarding
and typed refusals), [`backends.py`](../src/titanos/backends.py) and
[`genai.py`](../src/titanos/genai.py).

## Standard OpenAI fields per backend

The stock OpenAI SDK's standard fields are no longer schema refusals. Each is
forwarded where the backend implements it and answered with a typed 400 naming
the field where it does not; inert values (`false`, `0`, `{}`, `n=1`) are the
OpenAI defaults and are always served. Checked against the b10809 CPU
`llama-server` binary with a Qwen2.5-0.5B GGUF on 2026-10-05.

| Route | Field | Managed llama.cpp (b10809) | Transformers | OpenVINO GenAI |
|---|---|---|---|---|
| chat, completions | `logprobs`, `top_logprobs` | forwarded; `choices[].logprobs` returned, also in SSE chunks (completions: rewritten to the legacy `tokens` / `token_logprobs` / `top_logprobs` / `text_offset` shape) | 400 naming the field | 400 naming the field |
| chat, completions | `logit_bias` | forwarded (token id or token text keys) | `sequence_bias` over integer token ids | 400 (GenAI field check) |
| chat, completions | `presence_penalty`, `frequency_penalty` | forwarded | 400 naming the field | 400 naming the field |
| chat, completions | `n` (bounded by `TITANOS_MAX_CHOICES`) | forwarded; `n` choices returned | 400 for `n >= 2` | 400 for `n >= 2` |
| chat, completions | `seed` | forwarded; repeated seeded runs matched on the CPU binary | `torch.manual_seed` | `rng_seed` |
| chat, completions | `stop` (string or array) | forwarded (`TITANOS_MAX_STOP_SEQUENCES` / `TITANOS_MAX_STOP_CHARS` still apply) | stop strings honoured | stop strings honoured |
| chat | `response_format` `json_object` / `json_schema` | forwarded (grammar-constrained) | 400 (agent capability refusal) | 400 |
| embeddings | `encoding_format` `float` / `base64` | forwarded (base64 answered by llama-server) | base64 packed locally | — |
| embeddings | `user` | accepted, stripped, named in `X-Titanos-Ignored` | same | — |
| embeddings | `dimensions` | 400 naming `dimensions` (b10809 ignores the field, so forwarding it would drop it silently) | 400 naming `dimensions` | — |
| embeddings | `max_tokens`, `truncate` (Titanos-only) | 400 naming the field (the registered `native.context_size` bounds input) | honoured | — |
| speech | `voice` | — | per-model `voice_map` (below), then the backend's own voice check | — |
| speech | `speed` (0.25..4.0) | — | honoured by sherpa-onnx; transformers, xtts and qwen3-tts answer 400 naming `speed` for a value other than 1.0 | — |
| speech | `response_format` `wav` / `pcm` | — | both on every local speech backend; `pcm` is bare PCM16 named in `X-Sample-Rate`, `X-Bits-Per-Sample`, `X-Channels` | — |
| speech | `stream_format` `audio` / `sse` | — | `audio` streams as it is made on qwen3-tts; other backends answer the complete body; `sse` 400 for local models | — |
| speech | `temperature`, `top_k`, `top_p`, `subtalker_*`, `frame_budget` | — | qwen3-tts only, in gim_api TTS Super's ranges ([QWEN_TTS.md](QWEN_TTS.md)); other backends 400 naming the fields | — |

### GGUF embedding models

Register a GGUF embedding checkpoint with `task: embeddings`, `backend:
llama-cpp` and native options as for chat; `native.pooling` (`none`, `mean`,
`cls`, `last`, `rank`) selects `llama-server --pooling` when the GGUF carries no
pooling type. The owned process starts with `--embeddings`; `POST
/v1/embeddings`, Ollama `/api/embed` and `/api/embeddings` are proxied to it
(`native.proxy`), and a generation route on that model answers 400 naming the
task. Verified with the b10809 CPU binary and the Qwen2.5-0.5B GGUF under
`--pooling mean` (an instruct model used only to prove the proxy path; it is
not an embedding-quality claim).

### Speech voice map

`ModelSpec.voice_map` (speech models only) maps client voice names onto the
model's own voice ids, so the OpenAI SDK's mandatory `voice="alloy"` reaches a
transformers model as `default`, a sherpa-onnx speaker id or a registered
xtts/qwen3-tts reference alias:

```json
{"voice_map": {"alloy": "default", "nova": "default", "echo": "0"}}
```

The map is applied inside the engine for `POST /v1/audio/speech`, speech asset
jobs, the ElevenLabs shape and MCP; a name outside the map passes through to
the backend's own check (unknown voices still fail rather than substitute).

Unlimited/prefill sentinel budgets are not mapped: Ollama/llama.cpp modes such
as `num_predict=-1`/`n_predict=-1` are distinct from Titanos's positive bounded
token contract. Sending them is an explicit error; increasing
`TITANOS_MAX_TOKENS` does not expand a checkpoint's trained context or available
memory. The pinned native API has additional samplers and output semantics;
see [b10809 server requests](https://github.com/ggml-org/llama.cpp/blob/b10809/tools/server/README.md).

## Load-time and native controls

These are **registration/startup settings**, not per-generation overrides.
Use local registry JSON or `titanos add --native-options native.json`.
Authenticated public registration deliberately does not accept executable
native-loader settings. Explicit worker assignments override the model's native
device/threads; generation requests can select a configured worker, not invent
new placement. See [`NativeOptions` and `native_assignment`](../src/titanos/config.py)
and [`NativeWorker.start_process`](../src/titanos/native.py).

| Intent | Titanos configuration | Environment default / native argument |
|---|---|---|
| Context allocation (`num_ctx` in Ollama) | `native.context_size`, integer ≥512 | `TITANOS_NATIVE_CONTEXT_SIZE=8192` → `--ctx-size`; no model-auto zero sentinel. Context/parallel-slot allocation follows the loaded native runtime; do not equate it to output tokens. Ollama's per-request `options.num_ctx` sizes the per-slot context (`context_size` over `slots` at a cold start) up to the largest per-slot context (`native.max_slot_context`, `TITANOS_NATIVE_MAX_SLOT_CONTEXT` or the GGUF trained context) and is refused with 400 `context_limit` above it ([NATIVE.md](NATIVE.md)); on backends without a context option it is named in `X-Titanos-Dropped`. |
| Parallel slots | `native.slots`, integer ≥1 | `TITANOS_NATIVE_SLOTS=4` → `--parallel` (2,048 tokens per slot with the default context; see [PERFORMANCE.md](PERFORMANCE.md)); independent of Titanos's waiting-queue capacity. |
| CPU generation threads | `native.threads` | `TITANOS_NATIVE_THREADS=4` → `--threads`. General `TITANOS_THREADS`/`serve --threads` is a different direct-backend setting. A worker assignment always replaces `native.threads` with `WorkerSpec.threads` for that worker, and `WorkerSpec.threads` defaults to `TITANOS_WORKER_THREADS` (4), not `TITANOS_NATIVE_THREADS`; an assignment created without `--threads` therefore runs the native process with 4 threads. |
| Logical / physical batch | `native.batch_size`, `native.ubatch_size` | `TITANOS_NATIVE_BATCH_SIZE`/`TITANOS_NATIVE_UBATCH_SIZE` default 0 means omit the flag; explicit model values must be positive, with microbatch ≤ batch. |
| Attention kernel / KV cache types | `native.flash_attn`, `native.cache_type_k`, `native.cache_type_v` | `flash_attn` `auto` (default) adds no argument; `on`/`off` pass to `--flash-attn`. Cache types are omitted unless named (runtime default f16); allowed values follow the runtime (`f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`), and a quantized V cache requires `flash_attn: on`. |
| GPU offload | `native.gpu_layers` plus model `device` | `TITANOS_NATIVE_GPU_LAYERS=auto` → `--n-gpu-layers`; `auto`, `all`, and nonnegative integers are accepted. Explicit CPU or zero selects `--device none` with zero layers. Automatic device choice stays inside inherited visibility; an explicit GPU cannot silently fall back to CPU. See [accelerator execution](ACCELERATOR_EXECUTION.md). |
| Multi-device layout | `native.split_mode`, `native.tensor_split` | `none`, `layer`, `row`, `tensor`; positive proportions must match the explicit device count. Native support is not hardware/model qualification. |
| Startup readiness | `native.startup_timeout` | `TITANOS_NATIVE_STARTUP_TIMEOUT=300`; bounds the readiness loop after hashing/probing/spawn, not the entire preparation path. |
| Native HTTP I/O | `WorkerSpec.timeout`, otherwise settings fallback | `TITANOS_WORKER_TIMEOUT=600`; the implicit native worker now uses this setting too. Connect is fixed at 5s. This is an I/O timeout, not a complete stream wall-clock deadline. |

The owner always binds the internal native server to loopback and supplies its
port, alias and private key. It enables Jinja, continuous batching and metrics,
disables context shifting, and sets `--fit off`. Ambient `LLAMA_*` and `MTMD_*`
variables are stripped. Separate batch-thread/affinity controls, flash attention,
KV-cache precision/offload, load mode/NUMA, main-GPU selection and speculative
draft configuration are not currently exposed. These are explicit product
surface differences, not declarations that the underlying runtime lacks them.
The official [b10809 launch reference](https://raw.githubusercontent.com/ggml-org/llama.cpp/b10809/tools/server/README.md)
and [argument implementation](https://github.com/ggml-org/llama.cpp/blob/b10809/common/arg.cpp)
are the comparison authority; backend environment tricks are not a supported
escape hatch.

## Queues, lifetime, media and resource boundaries

| Area | Operator setting / behavior |
|---|---|
| Native waiting queue | `TITANOS_NATIVE_QUEUE_CAPACITY=32`, `TITANOS_NATIVE_QUEUE_TIMEOUT=30`; full/expired admission is rejected, distinct from the native slot count. Replica queues have separate `TITANOS_REPLICA_QUEUE_*` settings. |
| Asset jobs | `TITANOS_QUEUE_CAPACITY=8`, `TITANOS_JOB_WORKERS=1`; capacity includes queued/running jobs. Artifact/job retention defaults to 86400s. |
| Model lifetime | Explicit load/unload and owner cleanup; no model-idle eviction. Ollama's per-request `keep_alive` is accepted: `0` unloads after the answer (the unload operation), a negative value is the resident default, a positive duration has no timer and is named in `X-Titanos-Dropped`. `TITANOS_KEEP_ALIVE_TIMEOUT` controls HTTP connections, not model residency. Ollama's distinct field is documented [here](https://docs.ollama.com/api/generate#body-keep-alive). |
| Deadlines | Client HTTP timeout, client `wait_job` elapsed deadline, native queue/startup/I/O timeouts and internal completion's shared deadline are different scopes. Public streaming proxy does not gain a total-stream deadline merely from `TITANOS_CLIENT_TIMEOUT`. |
| Text/media ceilings | `TITANOS_MAX_BODY_BYTES`, `MAX_TEXT_CHARS`, `MAX_PROMPT_CHARS`, `MAX_SPEECH_CHARS`, image dimensions/pixels/counts, `MAX_STEPS`, video frames/fps, audio duration/samples and music output/sample ceilings are configurable settings. Model/context/decoder constraints still apply; unsupported controls fail rather than promising arbitrary media. |
| Threads vs hard quotas | Thread settings are not RAM/VRAM quotas. Settings/NativeOptions do not provide a first-class aggregate native memory or OS CPU quota. The separately recorded Job/cgroup validation limits are controller controls, not application features. |

Settings accept positive capacities/durations rather than arbitrary upstream
zero/negative disable sentinels. Some enumerations and protocol ranges are fixed
policies: e.g. one output choice, temperature 0–2 and minimum native context512.
Do not describe these policies as physical model limits. Backend/model limits
include context/tokenizer capacity, supported tensor operators, available
precision/devices and memory. Network page-fetch safety ceilings are separately
bounded; they are not generation-quality knobs.

## Practical setup

For a compatible local GGUF and a bundled or already prepared managed native
runtime, keep startup and generation defaults separate. The packaged application
does not require the user to install Python; a core profile and a full numerical
profile carry different dependencies. Model weights remain external.

`native.json`:

```json
{"model_file":"model.gguf","context_size":8192,"slots":1,"threads":2,
 "batch_size":128,"ubatch_size":128,"gpu_layers":0}
```

`defaults.json`:

```json
{"max_tokens":512,"temperature":0.3,"top_p":0.9,"top_k":40}
```

```text
titanos add local-chat ./models/qwen --task chat --backend llama-cpp --device cpu --native-options native.json --defaults defaults.json
titanos run local-chat "Explain local inference briefly." --max-tokens 64 --temperature 0 --seed 7
```

Set environment defaults/ceilings **before** starting the relevant process or
creating omitted native fields. For example, increase `TITANOS_MAX_TOKENS` to
32768 before requesting more than 8192 tokens; a model default cannot bypass the
server's configured ceiling. Keep sufficient native context and physical memory.
Use `--options` for other supported request fields rather than expecting every
field to have a dedicated CLI switch. Use `/api/models/capabilities` and
`/api/info` to inspect the configured backend/settings; capability discovery is
not successful execution of all controls.

## Evidence and remaining qualification

The focused `tests/check_option_parity.py` checks actual request/model/settings
precedence, optional-null/zero/false behavior, token aliases including the real
Ollama-generate bridge, pre-selection refusal of unsupported native defaults,
and implicit/explicit worker timeout selection. Its 17 checks also exercise
native Responses through actual ASGI requests and a native proxy spy, preserving
the tool/text/stream envelope while applying defaults before serialization.
The existing synthetic Responses suite passed separately for non-native and
native transport regressions. No model/process execution was involved; original
failing checks are retained privately. These source corrections are not
retroactively part of previously sealed binaries.

Retained b10809 CPU evidence binds native executable
`cb29f66008d4d73cce17cab2c2569ab318eb244b0b2ca2a15024b44d90dfcd3f`
and observed CPU device selection, zero GPU layers, two threads, context 1024 and
one slot. Batch/microbatch 128 were configured and source-mapped, not independently
captured in the full executed argv. Its enclosing 2519 controller outcome remains
failed at descendant-cleanup reporting; successful inference observations are a
subset. It is not all-options or accelerator qualification.

This guide supplies the maintained parity/deviation map for CFG-03. CFG-02 also
requires meaningful target-native qualification; source checks and this table
do not close every generation/load/queue/media/resource combination. No Ollama
installation, model call, native helper invocation or broad test replay was
performed for this update.
