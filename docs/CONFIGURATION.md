# Configure Titanos

Tune Titanos through environment variables, model registrations and individual
requests. No HiveMind process is involved. Start with:

```text
titanos settings --json
titanos serve --help
titanos run --help
```

`settings --json` reports this process's effective values, built-in defaults,
types and exact environment names. It never includes API keys or Hub tokens.
Environment values are read when the process starts; restart after changing
them. A CLI flag overrides its matching environment setting.

Generation options resolve in this order: an explicitly supplied request field,
the registered model's `defaults`, the server's environment defaults, then the
adapter's own defaults for adapter-specific options. A request is still checked
against configured limits and the model's physical context. Raising a ceiling
does not increase model context, available RAM or VRAM.

## Common changes

PowerShell:

```powershell
$env:TITANOS_PORT = '11435'
$env:TITANOS_THREADS = '12'
$env:TITANOS_DEFAULT_MAX_TOKENS = '1024'
$env:TITANOS_MAX_TOKENS = '32768'
$env:TITANOS_CLIENT_TIMEOUT = '1800'
titanos serve
```

Linux/macOS shell:

```sh
TITANOS_THREADS=12 TITANOS_DEFAULT_MAX_TOKENS=1024 TITANOS_MAX_TOKENS=32768 titanos serve
```

For larger images, raise both the dimension and pixel ceilings. For example,
`TITANOS_MAX_IMAGE_DIMENSION=4096` and
`TITANOS_MAX_IMAGE_PIXELS=16777216` permit a 4096 by 4096 request if the selected
model and device can produce it. Image dimensions remain multiples of eight.

`TITANOS_MAX_TENSOR_ELEMENTS` defaults to `4000000`. It bounds each incoming typed
tensor and the combined element count of a tensor response. Raise it for larger
prepared graphs, for example `TITANOS_MAX_TENSOR_ELEMENTS=8000000`. The independent
`TITANOS_MAX_BODY_BYTES` request limit still applies. Values must remain finite,
match the declared dtype and shape, and use the tensor API's rank limit of eight.
The rank limit is a structural API constraint, not a memory-capacity setting.

Replica processes receive the application's validated settings snapshot, including
raised or lowered limits; they do not reconstruct it from ambient environment
variables. The saved worker's `threads` value overrides the general `threads`
setting in that child. ONNX and OpenVINO CPU defaults use the active settings;
an explicit OpenVINO `INFERENCE_NUM_THREADS` model load option remains more
specific. Worker timeout and queue settings keep their existing precedence.

`--host`, `--port`, `--threads`, `--limit-concurrency`,
`--keep-alive-timeout`, `--log-level` and `--role` on `serve` override their environment
values. Network binds require `TITANOS_API_KEY` (or an active admin key minted with
`titanos keys create`); clients read their key from their own environment. `TITANOS_URL` selects a client endpoint. `TITANOS_HOME` selects
the model registry and local artifact directory. `--config` selects another
registry explicitly. These existing path and credential settings are kept out
of the numeric settings catalog.

Front-door rules (`Guard` in `src/titanos/server.py`): the public routes (`/healthz`, `/`, the Studio assets under `/ui/`, and the API documentation views `/docs`, the two packaged Swagger UI assets under `/docs/assets/` and `/openapi.json`) accept `GET` only and refuse any other method with 405 and any body-bearing request with 400 before the body is read; every other route's body read is bounded (a time budget scaled to the declared length, a concurrency slot with a short bounded wait, 413 for a declared length over the limit). In keyless loopback mode (no `TITANOS_API_KEY`, loopback bind) a browser request that carries a foreign `Origin`, a `Sec-Fetch-Site` of `cross-site` or `same-site`, or more than one `Origin` header is refused with 403 on non-public routes before any device or engine work; same-origin Studio requests, user navigations (`Sec-Fetch-Site: none` on safe methods) and header-less local clients such as the CLI and the MCP client keep working. With an API key configured the bearer check applies as before and these rules are additive. The body-read bounds are ordinary settings with the usual `TITANOS_` environment names: `body_read_seconds` (`TITANOS_BODY_READ_SECONDS`, default `30`) is the minimum whole-body read budget per request and answers 408 `request_timeout` when it expires; `body_read_min_rate` (`TITANOS_BODY_READ_MIN_RATE`, default `262144` bytes/s) widens that budget to `Content-Length / rate` for larger declared bodies; `body_read_concurrency` (`TITANOS_BODY_READ_CONCURRENCY`, default `8`) is the number of request bodies read at once; `body_read_wait_seconds` (`TITANOS_BODY_READ_WAIT_SECONDS`, default `2`) bounds the wait for a free read slot before 503 `server_busy` with `Retry-After: 1`. All four must be positive and finite. Like every other setting they are read when `serve` starts (the read-slot semaphore is sized on first use), so a change takes effect on the next start; `settings --json` shows the effective values.

## API documentation views

`/docs` is Swagger UI rendered from assets packaged inside Titanos (`swagger-ui-dist` 5.33.0, Apache-2.0, unmodified,
served by the application under `/docs/assets/swagger-ui.css` and `/docs/assets/swagger-ui-bundle.js`; the favicon is the
packaged Studio mark). The page requests nothing outside the server's own loopback origin: no CDN script or style, no remote
favicon, and the Swagger validator badge is disabled (observed in real headless Chrome with every non-loopback host
unresolvable, keyless and keyed, `tests/check_docs_offline.py` plus the retained browser receipts). `/openapi.json` is
generated locally by the application. The documentation views are answered even while the model registry is reloading (they
are outside the registry gate like `/healthz` and the Studio shell), and both `/docs` and `/` carry `X-Frame-Options: DENY`
and `Content-Security-Policy: frame-ancestors 'none'` so another origin cannot frame them. ReDoc is deliberately not served: `/redoc` is 404 in keyless mode and, like every
other non-public path in keyed mode, meets the bearer check first (401). The vendored assets are hash-bound to
`tools/licenses/vendored_assets.json`, their LICENSE and NOTICE are retained under `THIRD_PARTY_LICENSES/swagger-ui-dist/` in a
build, and the SBOM carries a `pkg:npm/swagger-ui-dist@5.33.0` component (`tests/check_docs_offline.py`).

Key policy: the documentation shell, its two assets and the OpenAPI schema are readable without the bearer in both key modes
(`GET` only; any other method is 405 and a body is 400, answered before the body is read, exactly like the other public
routes). They describe the API: the schema discloses the application version and the full route surface to an
unauthenticated loopback reader, and nothing from the registry, jobs, assets or devices. Every model and control route,
including the ones Swagger UI's "Try it out" button calls, stays behind the bearer in keyed mode, so an unauthenticated browser
can read the documentation but cannot act; the schema declares no security scheme, so Swagger UI shows no Authorize control and
`/docs` is read-only reference in keyed mode (every "Try it out" answers 401). Browser subresource requests never carry the Studio bearer, which is why the assets are
public rather than keyed.

## Model and request defaults

Save a JSON file such as `chat-defaults.json`:

```json
{"max_tokens": 512, "temperature": 0.2, "top_p": 0.95}
```

Register it with local weights, or supply the same `--defaults` option to `pull`:

```text
titanos add assistant /path/to/checkpoint --task chat --defaults chat-defaults.json
titanos run assistant "Explain this briefly" --max-tokens 64 --temperature 0 --stream
```

`run` sends a generation field only when supplied, so its parser does not replace
registered model defaults. `--options options.json` accepts additional supported
API parameters such as native tool definitions and structured response formats.
It cannot replace the model, prompt or streaming controls supplied separately.
`--messages messages.json` accepts an entire conversation with tool results.
`--history conversation.json` saves successful turns atomically and includes
earlier turns in later requests. Interrupted responses are not saved to history.
`--json --stream` emits one original server event per stdout line.

Use the API's canonical names for common defaults: `max_tokens`, `temperature`,
`top_p`, `size`, `steps`, `frames`, `fps`. Adapter-specific model defaults remain
specific to that adapter; a Transformers or Diffusers option is not a portable
llama.cpp option. Unsupported request fields return an explicit error.

## Music weight placement

Registered `music` models (backend `ace-step`) accept an optional `placement`
field: `auto` (default when omitted), `offload` or `resident`. It is passed to
the isolated music runtime as the contract's `placement`; any other task with
`placement` set is rejected at registration.

- `offload`: every component is loaded on the host and moved to the device only
  for its stage, then back. The host must hold the bundle's weights.
- `resident`: the planner, text encoder and DiT load straight onto the device
  (`device_map`) and stay there; those three never touch host memory. The VAE
  (about 337 MB) is the exception: it materialises on the host once and moves
  once after loading, then stays. Nothing returns to the host and there is no
  CPU-offload fallback. The runtime refuses **before
  loading any weights** when the device cannot hold every parameter of the
  bundle at the resolved dtype (element count times 2 bytes for bfloat16 or
  float16, 4 for float32, read from the safetensors headers) plus 4 GiB of
  headroom, when `accelerate` is not importable, or when the dtype is not
  supported; the job error text starts with `placement_refused_before_load:`
  and names the measured numbers. `resident` on a CPU device is refused.
- `auto`: `resident` when those same checks pass, otherwise `offload`. On CPU
  every value runs on the host.

`placement` only chooses where weights live; it never lowers a resource
admission floor, and no request field can supply its own byte estimate. The
child records the requested and resolved placement, the measured free device
bytes, the resident/peak/headroom/required byte counts, the accelerate version
and the deciding basis as evidence; the parent rejects results whose placement
evidence is missing or inconsistent: a `resident` result whose receipt reports
CPU offload, a `resident` result with less free device memory than the
recorded bill, or an `auto` request resolved to `offload` without an evidenced
obstacle (accelerate absent or free below the bill). See
`MUSIC_RUNTIME_CONTRACT.md`.

## OpenVINO GenAI text generation

The GenAI adapter resolves explicitly supplied request values before canonical
model defaults. Zero is retained for controls that permit it: for example,
`temperature: 0`, `top_k: 0`, `min_p: 0` and `seed: 0`. `None` means omitted.

| Portable control | GenAI handling |
| --- | --- |
| `max_tokens` | `max_new_tokens`; the resolved limit still participates in the physical-context check |
| `top_k`, `min_p` | Forwarded under the same native names |
| `repeat_penalty` | Translated to `repetition_penalty` |
| `seed` | Translated to `rng_seed` |
| `stop` | Applied by Titanos's existing streaming/text-trimming wrapper; an explicit empty list clears model stop defaults |
| `temperature`, `top_p` | Sampling controls; zero temperature selects greedy decoding |

Canonical aliases such as `seed`, `repeat_penalty`, `stop` and token-budget API
names are never passed as unknown native keyword arguments. A portable request
also overrides the corresponding native model-default alias. Sampling filters
can be supplied with greedy decoding, but GenAI applies them only when sampling.

Trusted local model registrations may use other native generation options. The
adapter checks their names against the installed GenAI `GenerationConfig`
instance, so supported vendor options do not require a fixed Titanos inventory.
Absent, private or method names fail before generation. Portable request fields
outside the text adapter's contract also fail explicitly; native options in a
local model registration do not extend the public request schema. The adapter
still returns one result and retains its existing chat-template, cancellation,
stop handling and context boundaries.

On CPU, `INFERENCE_NUM_THREADS` inherits the active Titanos `threads` setting,
including an explicit server/worker Settings override. An explicitly configured
`load_options.INFERENCE_NUM_THREADS` takes precedence. This affects CPU loading
only; it does not move the selected device.

## Settings catalog

Every row below has an environment variable named `TITANOS_` plus the uppercase
field name. For example, `artifact_ttl_seconds` is
`TITANOS_ARTIFACT_TTL_SECONDS`. Run `settings --json` for exact effective values.

| Field | Built-in default | Purpose |
| --- | ---: | --- |
| `host` | `127.0.0.1` | HTTP bind host |
| `port` | `11435` | HTTP bind port. Titanos binds exactly the requested address and never substitutes another port. If `serve` exits with a bind error such as Windows `[winerror 10013]`, the operating system refused that address/port; see "Bind failures" below |
| `threads` | up to 8 available CPUs | PyTorch/CPU worker thread default |
| `client_timeout` | 600 seconds | CLI/SDK HTTP timeout and default job wait deadline |
| `max_body_bytes` | 67108864 | Encoded HTTP request body ceiling |
| `body_read_seconds` | 30 seconds | Minimum whole-body read budget per request; 408 `request_timeout` on expiry |
| `body_read_min_rate` | 262144 bytes/s | Widens the body budget to `Content-Length / rate` for larger declared bodies |
| `body_read_concurrency` | 8 | Request bodies read at once (read-slot semaphore, sized on first use) |
| `body_read_wait_seconds` | 2 seconds | Bounded wait for a read slot before 503 `server_busy` |
| `max_tokens` | 8192 | Requested generated-token ceiling |
| `default_max_tokens` | 256 | Default generated-token budget |
| `default_temperature` | 0.7 | Default sampling temperature |
| `default_top_p` | 1.0 | Default nucleus sampling probability |
| `max_images` | 8 | Images per vision-chat request |
| `max_image_dimension` | 2048 | Maximum generated image width or height |
| `max_image_pixels` | 4194304 | Maximum generated image area |
| `max_input_image_bytes` | 20971520 | Decoded upload image byte ceiling |
| `max_input_image_pixels` | 16000000 | Decoded upload image pixel ceiling |
| `max_generated_images` | 4 | Images produced per image request |
| `max_steps` | 100 | Diffusion inference step ceiling |
| `default_image_size` | `512x512` | Default generated image dimensions |
| `default_steps` | 25 | Default diffusion inference steps |
| `max_video_frames` | 129 | Video frame ceiling |
| `default_video_frames` | 17 | Default requested video frames |
| `max_video_fps` | 60 | Video frame-rate ceiling |
| `default_video_fps` | 8 | Default output video frame rate |
| `max_audio_seconds` | 600 | Input audio duration ceiling |
| `max_audio_samples` | 20000000 | Total decoded input audio sample ceiling |
| `max_music_samples` | 60000000 | Generated music samples across all channels; permits 600 seconds at 48 kHz stereo |
| `max_music_output_bytes` | 134217728 | Music WAV/owned-process output budget, independent of HTTP input size |
| `embedding_max_tokens` | 512 | Default embedding tokenizer sequence bound |
| `embedding_max_batch` | 64 | Texts per embedding request |
| `detection_threshold` | 0.5 | Detection score filter; overridden by model defaults and request threshold |
| `detection_nms` | false | Enable class-aware filtering of overlapping detection boxes |
| `detection_nms_iou` | 0.7 | IoU threshold when NMS is enabled; explicit zero is valid |
| `max_messages` | 128 | Messages per chat request |
| `max_tools` | 32 | Tools/tool calls per supported agent request |
| `max_text_chars` | 100000 | General text input ceiling |
| `max_prompt_chars` | 8000 | Image/video prompt character ceiling |
| `max_speech_chars` | 5000 | Speech generation input character ceiling |
| `queue_capacity` | 8 | Outstanding asset-job admission ceiling |
| `job_workers` | 1 | Asset-job executor workers |
| `artifact_ttl_seconds` | 86400 | Generated artifact and job retention |
| `acquisition_max_bytes` | 8589934592 | Maximum planned remote model download bytes; archives also require expansion headroom |
| `acquisition_workers` | 1 | Concurrent model acquisition/setup workers, separate from inference jobs |
| `acquisition_retention_seconds` | 604800 | Acquisition record/partial-state retention for explicit retry/resume |
| `model_page_timeout` | 15 seconds | Whole DNS/TLS/redirect/body budget for public model-page metadata |
| `model_page_max_bytes` | 2000000 | Maximum uncompressed public model-page HTML bytes |
| `model_page_max_redirects` | 3 | Public HTTPS redirects revalidated with fresh DNS pinning |
| `model_page_max_links` | 50 | Maximum page-declared model/license links retained |
| `model_page_max_text_chars` | 2000 | Title/description/link-label character bound |
| `limit_concurrency` | 64 | Uvicorn HTTP concurrency ceiling over every open connection, idle keep-alive connections and streams included; above it uvicorn answers a plain 503 before Titanos sees the request, on `/healthz` too ([PERFORMANCE.md](PERFORMANCE.md)) |
| `health_port` | 0 | When nonzero, `serve` also listens on this port (same host) for `/healthz`, `/readyz` and `/v1/health/ready` only, outside `limit_concurrency`, so health probes keep answering under load ([OPERATIONS.md](OPERATIONS.md#readiness-and-liveness)) |
| `keep_alive_timeout` | 5 seconds | Idle HTTP connection timeout |
| `log_level` | `info` | Uvicorn log level |
| `log_file` | empty | Absolute path of a server log file, written in addition to standard error and rotated by size ([OPERATIONS.md](OPERATIONS.md#logs-and-rotation)) |
| `log_max_bytes` | 67108864 | Size at which `log_file` is rotated to `log_file.1` (at least 4096) |
| `log_backups` | 4 | Rotated files kept (`.1` newest); the directory holds at most (backups + 1) x `log_max_bytes` plus one line |
| `log_max_age_days` | 0 | When nonzero, rotated files older than this are deleted at startup and at each rotation |
| `native_context_size` | 8192 | New native registration's total context, one allocation shared by its slots (2,048 tokens per slot at the default 4 slots); saved registrations keep their values |
| `native_slots` | 4 | New native registration's parallel slots (`--parallel`); raise `native_context_size` with it to keep the per-slot context ([PERFORMANCE.md](PERFORMANCE.md)) |
| `native_max_slot_context` | 0 | Largest per-slot context Ollama's `options.num_ctx` may size a llama-cpp runtime to; 0 uses each GGUF's trained `context_length`. A registration's `native.max_slot_context` takes precedence ([NATIVE.md](NATIVE.md)) |
| `native_threads` | 4 | New native registration's CPU threads |
| `native_gpu_layers` | `auto` | New native registration's GPU offload layers: `auto`, `all`, or a nonnegative integer; explicit `0` disables offload |
| `exclude_gpu_uuids` | empty | Comma list of NVIDIA GPU UUIDs native (llama-cpp) placement never uses: automatic placement takes the first visible GPU not listed, and an explicit selection of a listed GPU fails ([OPERATIONS.md](OPERATIONS.md#gpu-memory-admission)) |
| `device_probe_timeout` | 30 seconds | Bounded device/runtime discovery; inherited visibility is preserved |
| `device_inventory_ttl` | 30 seconds | Device/runtime inventory refresh cache lifetime |
| `native_startup_timeout` | 300 seconds | New native registration's startup deadline |
| `native_admission` | `auto` | GPU memory admission before a llama-cpp runtime starts: `auto` checks when the GGUF header and `nvidia-smi` can be read, `on` refuses when they cannot, `off` never checks ([OPERATIONS.md](OPERATIONS.md#gpu-memory-admission)) |
| `admission_reserve_mib` | 0 | MiB of each GPU's free memory that admission leaves to other owners |
| `admission_free_mib` | 0 | When nonzero, caps the free memory admission reads (lowers it, never raises it); for tests and for an operator budget |
| `native_queue_capacity` | 32 | Maximum requests waiting for a native worker slot, separate from active slots and asset jobs |
| `native_queue_timeout` | 30 seconds | Maximum native slot wait; positive fractional seconds are supported |
| `replica_queue_capacity` | 32 | Requests waiting for busy local replica workers; active workers do not consume this capacity |
| `replica_queue_timeout` | 30 seconds | Maximum local replica wait, including positive fractional seconds |
| `worker_threads` | 4 | CPU threads for a newly configured explicit worker |
| `worker_timeout` | 600 seconds | Execution timeout for a newly configured explicit worker |

`--native-catalog` / `TITANOS_NATIVE_CATALOG` is startup policy rather than a
request setting. Precedence is explicit argument, environment override, then
the trusted catalog shipped at `titanos/data/native-profiles.json`. Clients may
select only compatible profile names from that catalog and can never supply
executable URLs, target roots or runtime plans.

`TITANOS_WORKER_THREADS` and `TITANOS_WORKER_TIMEOUT` provide defaults for
explicit worker registrations. Worker JSON values and `titanos worker --threads`
or `--timeout` override those defaults. Effective values are saved with the
worker, so changing an environment variable does not rewrite an existing
assignment. Application-supplied settings also apply while parsing a registry
inside that application's settings context.

Threads must be positive integers. Worker timeouts must be finite positive
numbers and accept fractional seconds, such as `--timeout 0.5`. Titanos no longer
imposes the former 256-thread or 86,400-second schema ceilings. Actual backend,
operating-system and hardware limits still apply. Very short timeouts may stop
a worker before its first model load completes. This execution timeout starts
after the parent has started the worker; it is separate from replica queue wait,
HTTP client timeout and any explicitly configured native-runtime startup limit.
An execution failure is not automatically replayed. GPU/rank/model-count limits
and their compatibility checks are unchanged by these two controls.
The parent waits only for the remaining deadline and rejects responses observed
after it. Stopping the process and completing cleanup can take additional time;
the setting is not an end-to-end HTTP wall-clock guarantee.

Set `TITANOS_REPLICA_QUEUE_CAPACITY` and `TITANOS_REPLICA_QUEUE_TIMEOUT`
before starting the server. These settings apply to local process replicas,
including compiled model adapters. An idle matching worker can accept work
even when requests for other busy workers fill the waiting queue. A full queue
returns HTTP 429; an expired replica wait returns HTTP 503. The worker's
separate configured execution timeout still bounds model execution. Application
settings remain fixed across registry reloads and appear in `/api/info`.

Positive limits must be finite. Defaults cannot exceed their ceilings. Zero is
permitted for temperature, native GPU layers and HTTP keep-alive timeout.
HTTP keep-alive controls idle connections; it does not unload model weights.
Invalid values fail with the setting's name; they are not silently ignored.
Additional job executor threads do not make a single model adapter concurrently
reentrant: its inference lock still serializes work. Independent workers/replicas
are the explicit mechanism for separate model residency.

## Cloud broker, roles and TLS

These settings belong to the optional [cloud broker](BROKER.md) and the deployment
[roles](TOPOLOGY_ROLES.md). None of them holds a secret: provider credentials live
in the broker vault ([BROKER_SECURITY.md](BROKER_SECURITY.md)), never in settings.
With no `broker.json` the broker is inactive and these values change nothing except
the role, TLS and proxy-header rules.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `TITANOS_ROLE` | `standalone` | `standalone`, `broker` or `managed`; also `serve --role`; fixed until restart |
| `TITANOS_BROKER_CONFIG` | empty | Absolute path of `broker.json`; empty means `TITANOS_HOME/broker.json` |
| `TITANOS_BROKER_CLOUD_CEILING` | `fallback` | Upper bound of the cloud policy (`off`, `explicit`, `fallback`); the effective policy is the lower of this and `broker.json` |
| `TITANOS_BROKER_EGRESS` | `internet` | `internet`, `lan` (no public addresses) or `loopback-only`; checked on resolved addresses |
| `TITANOS_BROKER_VAULT_BACKEND` | `auto` | `auto`, `dpapi`, `keychain`, `secret-service`, `sealed-file` or `refs-only`; `auto` never picks `sealed-file` and becomes `refs-only` when no OS store is available |
| `TITANOS_BROKER_VAULT_PASSPHRASE_FILE` | empty | Absolute path of the owner-only passphrase file; required for `sealed-file` |
| `TITANOS_BROKER_CONNECT_TIMEOUT` | 10 seconds | Upstream connect timeout |
| `TITANOS_BROKER_FIRST_BYTE_TIMEOUT` | 120 seconds | Upstream headers plus the first complete frame |
| `TITANOS_BROKER_STREAM_IDLE_TIMEOUT` | 120 seconds | Longest pause between upstream stream chunks |
| `TITANOS_BROKER_MAX_FALLBACKS` | 2 | Fallbacks per request (at most 5); `broker.json` `policy.max_fallbacks` can only lower it |
| `TITANOS_BROKER_BREAKER_FAILURES` | 3 | Consecutive counted failures that open an upstream's breaker |
| `TITANOS_BROKER_BREAKER_BASE_SECONDS` | 2 | First open duration; doubles per reopening, with jitter |
| `TITANOS_BROKER_BREAKER_MAX_SECONDS` | 300 | Longest open duration |
| `TITANOS_BROKER_HEALTH_TTL` | 300 seconds | Probe interval for half-open or recently used upstreams |
| `TITANOS_BROKER_PROBE_IDLE_CUTOFF` | 900 seconds | An upstream idle this long is never probed, so an idle Titanos makes no outbound connections |
| `TITANOS_BROKER_CATALOG_TTL` | 3600 seconds | Lifetime of a fetched live catalog |
| `TITANOS_BROKER_ADVISOR_TIMEOUT_MS` | 300 | Default advisor timeout (at most 5000) |
| `TITANOS_BROKER_DECISION_LOG_SIZE` | 1000 | Decisions kept in memory |
| `TITANOS_BROKER_DECISION_LOG_FILE` | false | Also write decisions to `TITANOS_HOME/broker/decisions/` |
| `TITANOS_BROKER_USAGE_RETENTION_DAYS` | 90 | Usage ledger retention |
| `TITANOS_BROKER_AUDIT_RETENTION_DAYS` | 365 | Audit log retention |
| `TITANOS_BROKER_SECRET_CACHE_SECONDS` | 300 | How long a resolved credential stays cached in memory |
| `TITANOS_BROKER_DEBUG_UPSTREAM_URLS` | false | Add `X-Titanos-Upstream-URL` to answers for admin callers |
| `TITANOS_BROKER_DEFAULT_MAX_OUTPUT_TOKENS` | 4096 | Output cap injected for budgeted keys without one, and on every translation into Anthropic |
| `TITANOS_BROKER_MAX_UPSTREAM_JSON_BYTES` | 16777216 | Cap on an upstream JSON or text answer |
| `TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES` | 67108864 | Cap on an upstream media answer or stream |
| `TITANOS_BROKER_AUTH_FAILURES_PER_MINUTE` | 20 | Failed credential claims per address before 429 `auth_throttled` (0 disables) |
| `TITANOS_TLS_CERTFILE`, `TITANOS_TLS_KEYFILE` | empty | Absolute paths; set both to serve HTTPS natively (any role); broker role needs them for a network bind |
| `TITANOS_BEHIND_TLS_PROXY` | false | A TLS terminator on this host fronts Titanos; valid only with a loopback bind |
| `TITANOS_MANAGED_DELEGATION` | `off` | `parent` lets a managed instance serve parent-delegated cloud requests |
| `TITANOS_MANAGED_DEADLINE_HEADERS` | `x-titanos-deadline-ms` | Comma list of parent deadline headers (relative milliseconds) |
| `TITANOS_MANAGED_TRACE_HEADERS` | `x-request-id,traceparent,x-hivemind-trace` | Comma list of parent trace headers echoed as `X-Request-ID` |
| `TITANOS_UNKNOWN_FIELDS` | `strict` | `ignore` (managed role only; other roles stay strict) strips unknown top-level fields on the OpenAI and Ollama chat, completions, responses and embeddings routes ([TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md)) and names them in `X-Titanos-Ignored` instead of answering 422 |

**Proxy headers (behaviour change).** `serve` passes `proxy_headers` equal to
`TITANOS_BEHIND_TLS_PROXY` and `forwarded_allow_ips='127.0.0.1'` to uvicorn. Without
`TITANOS_BEHIND_TLS_PROXY=true`, `X-Forwarded-For` and `X-Forwarded-Proto` are no
longer trusted, even from 127.0.0.1 (uvicorn's own default trusted them), and a
`FORWARDED_ALLOW_IPS` environment value cannot widen this. A deployment whose local
reverse proxy relied on rewriting the client address or scheme must set
`TITANOS_BEHIND_TLS_PROXY=true` and bind to loopback.

Provider base URLs, keys and proxies are never read from ambient variables such as
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL` or `HTTPS_PROXY`; a provider
key in the environment is used only when a credential record references it by name
(`titanos broker credential set ID --upstream U --from-env OPENAI_API_KEY`). Native
workers, replica processes and device probes do not inherit provider variables or
variables named by such records.

The broker's own files live under `TITANOS_HOME/broker/` (keys, vault, audit, usage,
decisions, handles, ownership, instance id); the keys file and the vault are
owner-only.

## Native runtimes and practical parity

`--native-options native.json` provides the native model's context, slots,
threads, GPU layers, splitting and startup settings. Explicit native JSON values
take precedence over environment defaults. They are saved in the registry, so
changing an environment default does not rewrite existing registrations. Inspect
`titanos details MODEL` to see actual runtime and slot information. Context
allocation across slots depends on the installed native runtime.

[Ollama's configuration documentation](https://docs.ollama.com/faq) describes
environment settings for host, context, queueing, parallelism and model retention.
[llama.cpp's server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
describes native context, CPU threads, slots, GPU offload and generation controls.
Titanos exposes its implemented controls through the same familiar environment,
model and request layers. This is not full Ollama/llama.cpp flag or wire-protocol
parity: unsupported native flags and automatic model idle eviction are not
implied by similarly named settings. HTTP keep-alive controls connections, not
how long a model remains resident.

## Python client and durable assets

```python
from titanos.client import Client

with Client() as engine:
    reply = engine.chat("assistant", [{"role": "user", "content": "Hello"}], max_tokens=64)
    print(reply["choices"][0]["message"]["content"])
    image = engine.image("artist", "A small wooden cabin", steps=20)
    engine.save_asset(image["data"][0], "cabin.png")
    job = engine.submit_job("video", "animator", prompt="A slow sunrise", idempotency_key="sunrise-v1")
    result = engine.wait_job(job["id"], timeout=1800)
    if result["status"] == "completed":
        engine.save_asset(result["result"], "sunrise.mp4")
```

The client imports no model framework. `chat(..., stream=True)` yields original
JSON stream events, preserving tool calls and usage. A premature stream end is
an error. `wait_job` returns completed, failed, cancelled or interrupted records;
a local wait timeout leaves the server job running. Use `cancel_job` explicitly.
`save_asset` writes atomically and downloads only from the configured server's
origin with redirects disabled, so an asset response cannot redirect its API key
to another server. Save useful outputs before their retention window expires.

`titanos jobs submit request.json`, `jobs list`, `jobs get ID`, `jobs wait ID
--output result.mp4` and `jobs cancel ID` expose the same lifecycle for agents.
Failed/cancelled/interrupted terminal jobs print their JSON record and exit
nonzero. Use `video --async` to submit without waiting.


### Additional sampling and media controls

Chat/completion requests also accept `top_k`, `min_p`, and `repeat_penalty`; the PyTorch adapter maps the latter to Transformers `repetition_penalty`. These can be put in a model's `defaults` JSON. Unsupported architecture-specific generation parameters belong in a reviewed model configuration and must be tested with that checkpoint.

`TITANOS_MAX_STOP_SEQUENCES` and `TITANOS_MAX_STOP_CHARS` control stop-string limits. `TITANOS_MAX_CHOICES` (default 8) bounds the OpenAI `n` field on chat and completions requests (honoured by llama-cpp models; an in-process backend answers 400 for `n` above 1). `TITANOS_DEFAULT_EDIT_STRENGTH` sets the image edit default, overridden by model defaults and then a request strength. `TITANOS_VIDEO_CODEC` and `TITANOS_VIDEO_BITRATE` select an encoder provided by the installed media runtime. Source environments default to libx264; source-traced native application bundles set their platform encoder (Windows h264_mf, Mac h264_videotoolbox). Windows Media Foundation requires video dimensions of at least64px. This is an encoder constraint, not a hidden token/model limit.

For an explicit video diagnostic run, set `TITANOS_VIDEO_DIAGNOSTICS_DIR` to an absolute, private directory before starting the server. The default is empty: there is no diagnostic array conversion, hashing, filesystem work or extra response field when disabled. This operator setting applies to direct video requests and background video jobs; it cannot be supplied through an inference request.

Each enabled run creates a new directory named after the generated video's ID. Before opening the encoder, Titanos stores every model-returned frame as an individual lossless `.npy` array, with a `manifest.json` containing frame order, dtype, shape, file hashes, model ID, FPS and encoder settings. These are the frames returned after the pipeline's own VAE/postprocessing, before Titanos's floating-point pixel normalization and ImageIO/FFmpeg encoding. They are not intermediate latents or a kernel trace. Floating-point values, including NaN/Inf, are preserved for diagnosis; ordinary encoder validation still rejects invalid output.

`TITANOS_VIDEO_DIAGNOSTICS_MAX_BYTES` defaults to 268435456 (256 MiB) and limits the total NPY frame-file bytes **per video**, including NPY headers. The small JSON manifest is additional. The limit is not a total-directory quota. A capture that exceeds the cap or cannot be written fails the request before encoding; it never truncates a frame to fit. Cancellation is checked between captures and before encoding; existing encoding cancellation remains checked on return from the encoder.

Diagnostic files are intentionally retained on success, failure and cancellation until the operator removes them. They are outside ordinary artifact TTL/deletion and have no public download URL. Partial captures have `capture_complete: false`; a completed raw-frame set has `capture_complete: true`. The manifest's `status` describes the capture/encoder phase (`capturing`, `frames_captured`, `encoded`, `failed`, or observed `cancelled`), not job publication or user acceptance. A later job cancellation can remove the MP4 after encoding has finished. Consult the job record separately. Use a dedicated private folder for a bounded run, harvest it, then disable diagnostics and remove it when no longer needed.

Load each frame with `numpy.load(path, allow_pickle=False)`. To isolate encoding, replay the retained arrays through the same `titanos.media.write_video` implementation with the recorded FPS/codec/bitrate; this requires no model execution. If a raw frame is already gray, the defect preceded the encoder. If the raw frame is sound and the decoded video is not, investigate normalization, ImageIO, the encoder and decoder separately. A different host's passing codec fixture does not clear a failed native take.

The server uses the same effective settings through HTTP, MCP, extension handlers and background asset jobs. `GET /api/info` and `titanos settings` expose values and environment names, never API keys. Application code can pass an explicit `Settings` object to `create_app`; request processing propagates it to adapters and job threads.

For [masked image edits](image-masks.md), `TITANOS_PRESERVE_UNMASKED` defaults to true. A model's boolean `preserve_unmasked` default overrides it, and an explicit request overrides the model. This performs disclosed RGB pixel copying after generation; false returns raw model output without that guarantee. Masked edit dimensions always follow the source; a conflicting request size is rejected rather than resizing the mask.

## Process environment

`titanos serve` sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from the active `threads` setting when they are unset and exports `TITANOS_THREADS` for its workers. It also provides a `USERNAME` value when none of `LOGNAME`, `USER`, `LNAME` or `USERNAME` is present in the process environment: PyTorch's compiled-kernel cache asks the operating system for a user name while `torch._inductor` is imported, and without one of those variables that lookup falls back to the Unix-only `pwd` module. On Windows the first request that reaches the import (the Qwen3-TTS speech path does so through `torch.compiler.cudagraph_mark_step_begin`) then fails with `No module named 'pwd'`. The value is the logged-in account name, or `titanos` when the operating system cannot supply one; a variable that is already set is never changed. Service environments that strip user variables can also set `USERNAME` themselves.

Children of the engine never receive provider credentials. A llama-server runtime, its supervisor and the device probe
start from an allowlist (`titanos.child_env.allowlisted_env`): operating-system and locale basics, temporary and home
directories, accelerator and visibility controls (`CUDA_*`, `GGML_*`, `HIP_*` and the like), thread controls, Python and
PyInstaller plumbing, offline cache locations and `TITANOS_*` settings. `TITANOS_CHILD_ENV_ALLOW` adds names or `NAME*`
prefixes (comma separated) for a driver or runtime variable the list lacks. Replica workers and FFmpeg keep the rest of
the environment. Every child loses the known provider variables, names ending in `_API_KEY`, `_BASE_URL`, `_TOKEN`,
`_SECRET`, `_PASSWORD`, `_SECRET_KEY`, `_SECRET_ACCESS_KEY` or `_ACCESS_KEY_ID`, and the variables the broker vault's
`env:` credentials name, even when `TITANOS_CHILD_ENV_ALLOW` lists them.

## Bind failures

`titanos serve` requests exactly the configured host and port and exits if the operating system refuses the bind; it does not retry on another port or change security settings. A refusal has several possible causes and the error code alone does not identify which applies: the port is held by another process or service, it lies inside an operating-system port exclusion or reservation (on Windows, `netsh interface ipv4 show excludedportrange protocol=tcp` lists current ranges, and the dynamic range reported by `netsh interface ipv4 show dynamicport tcp` is also used for outbound connections), or a filtering policy applies to the address, port or executable. Inspect the current TCP table and reservations at the time of failure, keep the requested address, and resolve the actual condition rather than guessing. Windows `[winerror 10013]` (WSAEACCES) is reported for several of these conditions.
