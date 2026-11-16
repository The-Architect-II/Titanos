# Titanos release notes

Newest first. Each entry names the source version it describes. A version number is not a qualification claim: delivered packages keep their own manifests and validation receipts, and the [capability matrix](CAPABILITY_MATRIX.md) records what has actually run. `.github/workflows/release.yml` attaches this file to every draft release it creates.

## 0.5.31 (15 Nov 2026)

A release can now be installed from a pinned interpreter and a hash lock alone, and `/api/infer` takes a deadline.

- **Pinned-interpreter install.** Each release carries a hash lock per platform (`python-lock-win_amd64.txt`, `python-lock-linux_x86_64.txt`): one `name==version` line per package with its `--hash=sha256:` values, every package a wheel on PyPI or the PyTorch CUDA 12.8 index (torch 2.8.0, torchvision 0.23.0, torchaudio 2.8.0). Both locks carry the CUDA build of PyTorch, so for now a machine with an AMD or Intel GPU installed this way runs the engine's PyTorch backends on the processor; the native runtimes keep their ROCm, SYCL, OpenVINO and Vulkan builds. The speech wrapper qwen-tts, a git dependency until now, ships as the release asset `qwen_tts-0.0.4-py3-none-any.whl`, built from its pinned commit; with the engine's own wheel it installs offline. `release.json` gains `python_lock`: the CPython 3.12 standalone builds by URL, sha256 and size, the indexes, the wheels and the locks. [RELEASING.md](RELEASING.md#pinned-interpreter-install) lists the steps; `tools/build_python_lock.py` rebuilds the locks and `tests/check_python_lock.py` checks them. The install catalog and the native profiles are unchanged, and `install.py` still installs the same way.
- The locks cover the base dependencies and the `models` and `qwen3-tts` extras. gradio is not in them: only qwen-tts's demo program imports it ([QWEN_TTS.md](QWEN_TTS.md)).
- **Request budget.** `POST /api/infer` reads `X-Request-Budget-Ms` (1 to 600000, counted from the request's arrival). Work whose budget has passed before it runs, or while it waits for the model, is refused with 504 `budget_passed` and never started; a malformed or repeated header is 400 `budget_invalid`. Without the header nothing changes.
- **File links behind a proxy.** The image, video and audio links in `/api/jobs` results follow `X-Forwarded-Host` and `X-Forwarded-Proto` when a loopback proxy sends them, and are relative paths when the forwarded host is malformed; the engine's own loopback address no longer reaches a client behind a proxy.
- Tests: `tests/check_sentiment_language.py` allows the language guess 2 s on a loaded host, and the checks that start a server hand it the socket they bound (or, for a child process, retry when another process took the port), so a port race no longer fails them.

## 0.5.30 (15 Nov 2026)

A maintenance release: the engine, its routes, the install catalog and the native profiles are unchanged from 0.5.29. Three tests are fixed.

- `tests/check_child_env.py` sets `TITANOS_HOME` for the captured native launch it checks; it passed before only on a host that already had `TITANOS_HOME` set.
- `tests/check_broker_mcp.py` hands uvicorn the socket it bound instead of closing it and binding its port again, so another socket can no longer take the port in between (on Windows that failed the server's start with WinError 10013).
- `tests/check_tesseract.py`'s Windows cancellation case has its child write the descendant's process id under a temporary name and rename it, so the check never reads a file that exists but is still empty.

## 0.5.29 (15 Nov 2026)

A maintenance release: the engine, its routes, the install catalog and the native profiles are unchanged from 0.5.28.

- `tools/build_route_contract_matrix.py` names a discriminated-union request body (`server.AssetJob`, `api_plus.ExtensionRequest`) the same way under FastAPI 0.163 and later, which hand the route's body parameter the union without its `Annotated` wrapper. The committed route matrix is unchanged; `tests/check_route_contract_matrix.py` now passes with the newest FastAPI the package allows, as it already did with older releases.

## 0.5.28 (15 Nov 2026)

No binary release carries this line yet. What changed since 0.4.0, in user terms:

### Cloud broker (optional, off by default)

- A cloud broker inside `titanos serve` ([BROKER.md](BROKER.md)) can answer the OpenAI, Anthropic, Gemini, ElevenLabs and Ollama routes from configured cloud providers, routers (OpenRouter, the Hugging Face router, LiteLLM, a HiveMind gateway) and self-hosted OpenAI-compatible servers, or from local models, through 22 upstream profiles (23 with the neutral slot). Same-protocol targets are passed through; other pairs are translated, incrementally for streams. Published aliases, virtual models with typed fallback before the first byte, per-upstream breakers, a router-advisor contract and a neutral `pending-contract` slot for an external router whose contract is not yet published.
- Off by default: without `TITANOS_HOME/broker.json` nothing leaves the host and local answers are byte-identical to a server without the broker. Cloud use needs an upstream with a validated credential, a cloud policy (`explicit` or `fallback`) and a caller allowed to use it. Every answer then says where it ran (`X-Titanos-Served-By` and the other provenance headers), and `titanos broker explain` shows where a request would go without sending it.
- Evidence: every provider path is proven against protocol-faithful loopback mocks under a network guard; the owner-run live smoke lane covers OpenAI and the Hugging Face router and is never run in CI. The generated [broker matrix](BROKER_MATRIX.md) states per cell whether it is pass (mock), live-pass, not applicable or untried.
- Proven live on 2026-10-02 (three owner-approved runs, about USD 0.0174 of a USD 0.50 cap, no secret found in any run file, audit chains verified; details in [VALIDATION.md](VALIDATION.md#live-results-of-2026-10-02)): OpenAI chat streamed and not streamed, Responses, embeddings, moderations, speech, transcription and image generation with gpt-image-1-mini (eight matrix cells), and the Hugging Face router's chat. Anthropic, Gemini, ElevenLabs, xAI, Groq, Mistral, Together, DeepSeek, OpenRouter, Fireworks, Perplexity, Azure OpenAI, Ollama Cloud, Cerebras, SambaNova and the router and self-hosted upstreams are mock-validated only.
- Credentials are write-only and bound to one upstream's exact origin; values live in DPAPI, the macOS Keychain or Secret Service, or stay in an environment variable or owner-only file that is referenced by name; an opt-in sealed file exists for hosts without an OS store ([BROKER_SECURITY.md](BROKER_SECURITY.md)).
- Scoped client keys (`titanos keys`, `/api/keys`, Studio) with per-key scopes, model constraints, rate limits, USD and token budgets checked before any upstream call, expiry and rotation; a hash-chained audit log (`titanos audit verify`); decisions, usage, cost (operator prices only) and metrics.
- Roles ([TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md)): `standalone` (default), `broker` (bootstrap key, native TLS off loopback, budgeted and expiring egress keys) and `managed` for an orchestrator such as HiveMind (sealed vault, no egress unless the parent delegates a request). New `titanos hivemind-manifest`; optional managed-role gateway registration.
- New routes: `/v1/moderations`, `/v1/responses/{id}` (retrieve, delete, cancel, input items), `/v1/messages/count_tokens`, ElevenLabs `/stream`, `/v1/speech-to-text` and `/v1/voices*`, Gemini `/v1beta/models*`, `/api/broker/*`, `/api/keys*`, `/api/audit*`, the public `GET /v1/health/ready`, and in managed role the public `GET /api/version`. MCP gains 12 tools (broker status, catalog, explain, decisions, usage, upstreams, health, test, `messages`, `count_tokens`, `moderate`, `whoami`) and no tool that writes credentials, keys or policy. Studio gains the Advanced **Cloud and routing** panel, a "Where to run" selector, External and Virtual badges and a provenance line under results.

### Provider completeness (2 Oct 2026)

"If somebody puts in a key it should just work": every endpoint listed in [BROKER.md](BROKER.md#what-works-with-a-key) is served through a typed surface or the guarded native passthrough. That page also lists what is not served (retired endpoints, admin APIs, endpoints that answer with a redirect to storage, the ElevenLabs agents platform) and the limits that still apply (request bodies are capped by `TITANOS_MAX_BODY_BYTES`, 64 MiB, below the providers' largest upload sizes; redirects are refused; an endpoint on a second provider host needs its own upstream); several allowlisted rows were seeded from provider documentation and await a check. [BROKER.md](BROKER.md#what-works-with-a-key) lists, provider by provider, each endpoint, how it is served, whether it streams and whether the proof is live or mock. Only the OpenAI and Hugging Face router cells listed above are live-proven. The typed surfaces below and the native rows the new checks exercise are proven against loopback mocks; the other allowlisted native rows are guarded like the rest but not exercised by a check yet, and BROKER.md marks them.

- **One-step connect.** `titanos broker connect <provider> --from-env NAME | --from-stdin | --from-file PATH` and `titanos broker quickstart [--yes]` create the upstream from its profile, store and validate the credential, enable it, raise `cloud_policy` from `off` to `explicit` (never above the ceiling) and list the provider's models. Over HTTP: `POST /api/broker/connect` and `GET /api/broker/env-candidates` (variable names and booleans only, never values). Python client: `broker_connect()` and `broker_env_candidates()`. Bare provider model ids (`gpt-4o-mini`, `claude-opus-5-5`, `gemini-2.5-flash`, `eleven_multilingual_v2`) resolve on the single visible upstream of that provider without aliases (`--aliases` still publishes them). On a keyless standalone server bound to loopback, connect turns `anonymous_cloud` on unless told otherwise, and `allow_store` sets the upstream's `data_policy.allow_store`. With `--from-env` the `titanos serve` process must hold the same variable: an upstream whose `env:` variable the server lacks is now reported `"ready": false` with `readiness.detail` `credential_unavailable` (Studio no longer marks it connected, and the offline CLI connect says so in its next steps) instead of ready while every call answered 404. Offline CLI changes to `broker.json` (connect, `prices import` and the rest) end with a note on how a running server applies them: a restart, or `titanos broker reload --url URL`, which needs a `broker:admin` key in `TITANOS_API_KEY` once the server enforces keys; while no active key holds that scope the note names the mint.
- **Keyless Studio provider setup.** On a standalone server bound to loopback with no key enforced, Studio may now connect providers, write credentials (a value, or the provider's well-known environment variable for the provider's default address; never another variable, a `file:` path or a host value sent to a base URL Studio picked) and set `anonymous_cloud`, with the `X-Titanos-Studio: 1` header and a JSON body; each write is audited as `keyless-studio`, and no fingerprint of a host value is returned to a keyless caller. Minting keys and `allow_http_lan` stay CLI or keyed only; broker and managed roles still refuse keyless writes.
- **Native passthrough.** `/native/<upstream>/<path>`, and the provider's own public path when exactly one visible upstream offers it, reach every allowlisted provider endpoint without translation: OpenAI files, uploads, batches, vector stores, conversations, containers, videos, voices, fine-tuning and evals; Anthropic Message Batches (the `results_url` is rewritten to this server), Files, Skills, Managed Agents and the Admin API; Gemini Files, cached contents, batches, Interactions, File Search, Imagen and Veo; ElevenLabs timestamps, dialogue, sound effects, voice changer, isolation, forced alignment, music, voice management, dubbing, pronunciation dictionaries and history; and the long-tail endpoints of Mistral, Together, Groq, xAI, Perplexity, DeepSeek, Fireworks, OpenRouter and Azure OpenAI. It needs the new `native` scope (plus `cloud:store` for provider-side state and `broker:admin` for account-wide rows), binds created ids to the calling key, audits state changes before sending, and never forwards the client's credentials.
- **Connected providers answer after a restart.** Live model catalogs are kept in memory, so before this fix a fresh `titanos serve` (after `titanos broker connect`, or any restart of a server set up over HTTP) answered every model call 404 `model_not_found` until some client listed models. The server now reads a stale live catalog itself on the first request that needs it (only upstreams the caller may see that the request's model could name, one fetch per upstream however many requests wait, a wait of at most 10 seconds within the request deadline, a failing catalog at most once a minute after its last attempt ended), for model calls, model lookups, and API Plus and job requests that name a cloud model and may be placed in the cloud. Local work never waits for a provider: a local model, a `local_only` request or a request from another broker hop fetches nothing, and Ollama's `/api/tags` answers from the cache while the fetch runs. `models.retrieve('gpt-4o-mini')` (OpenAI SDK) and `models.get('gemini-2.5-flash')` (google-genai) now resolve bare ids exactly as inference does.
- **OpenAI videos through `/v1`.** `POST /v1/videos` shares its path with Titanos' local video jobs; it now reaches OpenAI when the body's `model` names a cloud model (`sora-2`, `sora-2@<upstream>` or an alias; JSON or the SDK's multipart form), and retrieve, content and delete follow that video id, while local models and other ids keep the local job routes. Before, only `X-Titanos-Upstream` or `/native/<upstream>` reached OpenAI. The cloud call needs `native`, `cloud` and `cloud:store`; the local route still needs `jobs`.
- **Native lane cannot trip a shared breaker.** Requests refused locally on the native lane (empty or dot path segments, also after a `{model}` rewrite, egress refusals, deadline, concurrency) are answered before egress and no longer count as upstream failures. Neither do answers refused because of the object a key asked for (a redirect, or a declared size over `TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES` such as its own large file or video): as on the typed surfaces, only a malformed provider answer, a provider 5xx, a network error or a timeout counts, so one key can no longer open an upstream's breaker for every other caller.
- **No key can take a shared credential offline.** A provider 401 marks the upstream's credential failed at once only when it carries the provider's key-rejection code (OpenAI-compatible `invalid_api_key`, Anthropic `authentication_error`, Gemini `API_KEY_INVALID`, the ElevenLabs key-rejection codes, Ollama `unauthorized`). OpenAI's 401 for a restricted or project-limited key on an endpoint outside its permissions ("insufficient permissions ... Missing scopes") never marks it, so a key that calls such an endpoint (a native row such as videos, files or batches, or a typed surface such as moderations) no longer takes the upstream offline for every other key until an operator validates the credential again. Any other 401 on a client's call is checked against the upstream's own models endpoint (at most once a minute per credential), and that answer decides.
- **WebSockets.** OpenAI Realtime (also on the OpenAI SDK's own `/v1/realtime` URL), Gemini Live, Lyria RealTime and the ElevenLabs stream-input and realtime speech-to-text sockets are relayed frame by frame through a built-in RFC 6455 client. Accepting the inbound socket needs uvicorn's WebSocket support, so `websockets` is now a runtime dependency (and part of the frozen build).
- **OpenAI.** Streamed image generation and edits, transcription events and `stream_format=sse` speech are relayed with usage; stored background responses can be resumed with `?stream=true`; audio, image and web-search usage is metered at its own price-row rates; catalog entries carry `titanos.model_capabilities`; moderations without `model` use the provider default.
- **Anthropic.** Translations into Anthropic follow the target Claude model (thinking or adaptive thinking with `output_config.effort`, sampling fields the model refuses dropped, forced tool choice relaxed where refused, `json_schema` as `output_config.format`). Plain model ids from the Anthropic SDK resolve on a single visible Anthropic upstream, `GET /v1/models` answers in Anthropic's shape for the Anthropic SDK, and 1-hour cache writes, web search requests and fast mode are priced.
- **Gemini.** OpenAI speech, image generation, image edits, transcription and translation work with Gemini TTS, image and audio models through `generateContent`; both field spellings, per-model thinking settings, lossless JSON schemas, inline audio and generated images are kept; capabilities are per model.
- **ElevenLabs.** Plain model ids (`eleven_multilingual_v2`, `scribe_v1`) resolve on a single visible upstream, text-to-speech without `model_id` uses the provider default when no single local speech model would answer, every text-to-speech and speech-to-text argument is forwarded, and the SDK's own `GET /v1/models` reaches the provider.
- **OpenAI-compatible family.** New typed surface `POST /v1/rerank` for cloud rerankers (Together, Fireworks, OpenRouter, LiteLLM, custom endpoints); new profiles `azure_openai_v1` and `fireworks_audio`; Groq speech and Responses, Together and Mistral (Voxtral) audio, and more Responses rows; xAI and Together image URLs are passed through instead of being re-hosted; DeepSeek thinking is sent back in translated tool loops.
- **Dropped fields are named.** Translated requests list every field they dropped in the `X-Titanos-Dropped` response header (fields dropped after the first byte appear in the decision log only).

### Behaviour changes from the broker work

- **Local routes accept agent-harness fields.** The local chat, Responses and Messages routes accept, ignore and name in `X-Titanos-Ignored` the identity, cache, storage and reasoning fields Claude Code and Codex send (`user`, `metadata`, `store: false`, `prompt_cache_key`, `reasoning_effort`, `include`, `truncation`, `text.verbosity`, Anthropic `thinking` ...), which used to be 422; `store: true`, `previous_response_id`, `background: true`, `container` and `output_config.format` now answer a typed 400 instead of 422.
- **Vendor auth headers on more paths.** `x-api-key` is also a Titanos key on `/v1/models*`, and all three vendor headers are accepted under `/native` and on the bare native paths of their provider's allowlist.
- **Provider errors keep their meaning.** A plain 403 in the provider's own envelope (OpenAI-compatible organization verification, region or model access; Anthropic `permission_error`; Gemini `PERMISSION_DENIED`; ElevenLabs permission details) is passed through as 403 instead of 502 and never takes the shared credential offline; a provider 429 on OpenAI, Anthropic or Gemini cools down that model only (other models on the key keep serving) and further requests for it answer 429 `upstream_rate_limited` instead of 503 `breaker_open`; `insufficient_quota` and the billing codes answer 503 `upstream_unavailable` without `Retry-After`; ElevenLabs 401 `quota_exceeded` (503) and `missing_permissions` (502) no longer mark the credential failed; Groq 498 is 429 `upstream_rate_limited`.
- **Cloud arguments dropped instead of refused.** `top_k`, `min_p`, `repeat_penalty` and the VAD fields sent to providers that would refuse them are dropped and named in `X-Titanos-Dropped` (`repeat_penalty` becomes `repetition_penalty` where the provider takes it); `temperature` and `top_p` are dropped for models that refuse them (OpenAI reasoning models, recent Claude models).
- **xAI and Together images.** `response_format=url` (or none) returns the provider's own image URLs instead of re-hosting; JPEG output is no longer refused for them.
- **ElevenLabs streaming text-to-speech without `model_id`** goes to the single visible ElevenLabs upstream's default model when no single local speech model would answer (it was 400 `model_required`).
- **Keyless loopback cloud use.** With `anonymous_cloud` on, the keyless loopback principal also holds `cloud:store` (each upstream's `data_policy.allow_store` still decides); with it off, a keyless request for a cloud model answers 403 `cloud_disabled` naming the switch instead of 404 `model_not_found`.
- **Stored state stays explicit.** A client's `store: true` that Titanos forces off is named in `X-Titanos-Dropped`; Responses `background: true` and `conversation` without storage are refused 403 `scope_denied` before egress instead of being billed and lost.
- **Native lane hardening.** Model refs (`model@<upstream>`, aliases) become the provider's own ids before egress and refs of another upstream are refused; encoded path separators and percent-encoded prefixes are refused; `OpenAI-Organization` and `OpenAI-Project` are forwarded only for `broker:admin` principals; a file uploaded through Titanos is reachable on `GET /v1/files/{id}` by its owner; Gemini resumable uploads work with the stock SDK's key-less chunks through a per-session upload capability; ElevenLabs native calls are reserved in characters, so a `per_1k_chars` price row prices them.
- **Router loops.** A request that a router upstream sends back with the same `X-Request-ID` (when the router drops `X-Titanos-Hop`) is refused 508 `loop_detected` on its first re-entry.
- **MCP elevation removed.** Mounted MCP tools run as the key of the outer `/mcp` request. A scoped key that used to reach admin tools through the server's own key now gets 403 `scope_denied` there. Direct in-process calls without an HTTP request keep the key given to `create_mcp`.
- **Sticky key enforcement.** Once any key record exists (expired and revoked ones included), the server stays keyed; only the offline, audited `titanos keys reset --keyless` returns it to keyless mode.
- **Vendor auth headers accepted.** `x-api-key` on `/v1/messages*`, `xi-api-key` on the ElevenLabs routes and `x-goog-api-key` on `/v1beta/*` are accepted as Titanos keys; `?key=` on `/v1beta` is refused with 401 even in keyless mode.
- **uvicorn proxy headers are no longer trusted** unless `TITANOS_BEHIND_TLS_PROXY=true` (valid only on a loopback bind); `serve` pins `forwarded_allow_ips` to `127.0.0.1`. A local reverse proxy that relied on rewriting the client address or scheme needs the new setting. Native TLS: `TITANOS_TLS_CERTFILE` and `TITANOS_TLS_KEYFILE`.
- **`/api/info`** adds `role`, `orchestrated` and a `broker` block; its `settings` block is shown only to callers with `settings:read` (the bootstrap key, admin keys, keyless loopback, or keys granted it). `standalone: true` keeps its meaning. `/healthz` adds `role`.
- **New public route** `GET /v1/health/ready` (13 public routes instead of 12).
- **`/api/ps`** also lists loaded native llama.cpp models.
- **Keys see only their own work.** For minted keys, jobs, video jobs, `/v1/files` artifacts and stored responses are visible only to the key that created them, and local catalogs and jobs follow the key's model constraints.
- **Credential sources.** `env:` references refuse every `TITANOS_*` name (the bootstrap key is never an upstream credential). Over HTTP, `env:` and `file:` sources and re-binding a stored credential to a changed upstream origin need the `admin` scope; `credentials:write` alone stores values it sends itself.
- **Audio budgets fail closed.** Transcription and translation reserve the uploaded audio's duration against a key's budget (measured from WAV, including streamed WAV, FLAC and MP3 headers). Settlement prices token usage when the price row has token prices, otherwise the provider's seconds or the measured upload at the per-minute price; only an answer with nothing priceable is charged the reservation, never USD 0. Decisions and usage records name the basis (`price_basis`).
- **Output caps only where tokens are generated.** A budgeted key's default output cap is injected only on chat, completions, Responses, Anthropic Messages, Gemini generate and Ollama chat/generate requests; image, audio, embedding, moderation and token-counting requests reach the provider unchanged (Anthropic `count_tokens` and Gemini `countTokens` requests from a budgeted key had been sent an output cap).
- **Retired provider models.** A provider 410 Gone for a model, or a 404 whose provider code or message is about the model, answers 404 `model_not_found` in the caller's protocol, saying the provider reports the model unavailable or retired, instead of 502 `upstream_bad_response` (a provider answered 410 Gone live for retired models on 2026-10-02); a 404 about anything else (an ElevenLabs voice, a file, a response id) stays 404 `not_found`, and so does a 404 or 410 whose provider code names another resource (it contains voice, file, response, thread, assistant, batch, vector_store or upload, and not model), whatever its message says. Neither counts against the upstream's breaker or marks the credential.
- **OpenAI images without `response_format`.** OpenAI's Images API now refuses `response_format` for dall-e models as it already did for gpt-image models (observed live on 2026-10-02: 400 `unknown_parameter`; later that day dall-e-2 answered 400 "The model 'dall-e-2' does not exist."), so Titanos never sends the field to OpenAI. A client asking for `url` still gets re-hosted `/v1/files` URLs; `b64_json` or no format returns the provider's base64. Image generation passed live with gpt-image-1-mini. Azure OpenAI dall-e deployments still receive the field; Azure gpt-image deployments, which refuse it too, no longer do.
- **Live smoke receipts.** A failed probe's receipt entry keeps the provider's error message, scrubbed and cut to 400 characters, so a live failure is diagnosable without another call; a 200 answer that fails its checks keeps its in-body or in-stream error event the same way. The approval template now proposes gpt-image-1-mini (1024x1024, quality low, a `per_image` price to fill in) for the image probe. `tools/build_broker_matrix.py --live` accepts several evidence files and records failed live calls as observations that never change a cell's state.
- **`GET /v1/voices`** merges the upstream ElevenLabs voices after the local ones for keys that may see the upstream.
- **Check runner.** `tools/run_checks.py` shows an SDK lane that could not try anything (no SDK or fixture here) as `UNTRIED`, never as `PASS`.
- **CLI.** Argument-parser errors no longer echo values that look like tokens; the `broker`, `keys`, `audit` and `hivemind-manifest` parsers refuse abbreviated options; inference subcommands gain `--location`, `--target` and `--verbose`.

### Fixes after the 26 Sep 2026 refresh

- ACE music on Windows: the adapter's progress and result publication (`progress.json`, `result.json`) retries a rename that a concurrent reader blocks (WinError 5 or 32) within a one-second budget instead of failing the generation; failure reports are published best effort and never mask the primary error. Found on a packaged candidate at stage `loading_dit`; regression `tests/check_music_progress_publication.py` holds a reader on the destination.
- First real music result from a packaged Windows x64 candidate (E-107 in the capability matrix): one 10-second ACE-Step XL-SFT instrumental generated on one NVIDIA GPU with the repaired adapter, read back over HTTP and MCP, then unloaded and stopped cleanly. One host, one GPU, one prompt; not a quality claim.

### Engine and modalities

- 19 declared task types and 12 backends (`src/titanos/config.py`): chat, vision, embeddings, transcription, speech, image, video, music, tensor, translation, image-to-text, object detection, sentiment, rerank, VAD, pair classification, depth estimation, image segmentation and anonymous diarization. A declared adapter is not proof for every checkpoint or device; see [STATUS.md](../STATUS.md) for what has run.
- Typed extended operations: embeddings, cross-encoder rerank, retrieval answer composition with citations, text-pair classification, sentiment (native labels and a classifier/LLM composition), OCR (Tesseract and TrOCR), detection with optional NMS, prompted SAM segmentation, relative depth, Silero VAD, audio translation, and anonymous two-speaker diarization with transcript analysis under one deadline.
- Speech through Qwen3-TTS, XTTS and Sherpa-ONNX adapters with declared voice controls, explicit refusals and asynchronous speech jobs; ASR language hints follow the checkpoint.
- Image generation and editing through Diffusers, OpenVINO GenAI and ONNX pipelines, including masked and instruction edits; image-to-video jobs with persistent records; a typed ACE-Step music path that runs in an isolated runtime with an enforced memory cap (one bounded pass recorded, E-104).
- Managed native llama.cpp runtime: 20 pinned b10809 profiles across Windows, Linux and macOS, explicit device selection, GGUF acquisition with validated file selection.

### Interfaces

- HTTP with OpenAPI: 144 operations over 122 paths, including OpenAI-style chat, completions and stateless Responses, Ollama, Anthropic Messages, ElevenLabs and Gemini shapes, and the broker, key and audit routes. A generated [provider deviation map](PROTOCOL_DEVIATIONS.md) classifies every upstream field and route; explicit 400/422 refusals replace silent drops.
- MCP over stdio and Streamable HTTP with 72 typed tools, 6 resources, 6 templates and 2 prompts; a CLI (67 subcommands) and a Python client (118 methods). The [route contract matrix](ROUTE_CONTRACT_MATRIX.md) binds all of them to their typed contracts and marks declared-only and rejected-in-job semantics instead of hiding them.
- Studio, the browser interface: English, Spanish, Japanese and Chinese; Simple and Advanced modes; task-first model discovery and acquisition with license acceptance and progress units; completed creations persist across language changes; stable device assignments.
- Offline API documentation: `/docs` renders from packaged Swagger UI assets, ReDoc is not served, and the documentation views are public GET-only in both key modes.
- Front-door hardening: public routes accept GET only; body reads are bounded by time, rate and size (`TITANOS_BODY_READ_*` settings); keyless loopback mode refuses cross-site requests; Hugging Face acquisition validates every selected file name before download and walks the result afterwards.

### Configuration and hardware

- Controls resolve request over model defaults over process defaults for every implemented control. The [option parity map](OPTION_PARITY.md) (120 rows, 17 source checks) names fixed policies and unsupported options.
- Behaviour change in this line: the extended-inference backends (translation, image-to-text) now honour `TITANOS_DEFAULT_TEMPERATURE` and `TITANOS_DEFAULT_TOP_P` when neither the request nor the model default names them. With the shipped default of 0.7 these requests now sample; set a model default or `TITANOS_DEFAULT_TEMPERATURE=0` for deterministic output (`tests/check_extra_backend_generation.py`).
- Accelerator profiles for CUDA (cu126, cu128), ROCm, PyTorch XPU, MPS, DirectML (x64 and ARM64), QNN, Core ML and OpenVINO CPU/GPU/NPU, with GenAI and ONNX variants; stable device assignments by GPU, HIP, XPU or OpenVINO identity; a Python client pool that distributes independent requests across servers (request distribution, not sharding or failover).
- The [execution matrix](EXECUTION_MATRIX.md) records every model against every platform slot (56 x 22) with explicit untried and not-applicable cells, and the [matrix kit](MATRIX_KIT.md) fills it from any host with receipts that a verifier must uphold before a cell turns pass.

### Packaging and release

- Portable application builder with dependency and native inventories, notices, an SBOM, one scoped CUDA vendor row per packaged torch payload in `NATIVE_COMPONENTS.json`, hash-bound license supplements for isolated-runtime distributions, and a runtime path-length refusal with a directory budget. Music-inclusive Windows x64 candidates package the isolated ACE-Step runtime with every weight external and were relocated without ambient Python (E-106); the redistribution review is still open (PKG-03).
- Public export hygiene: `tools/public_export.json` states what the exported source copies leave out and how documents are kept free of lab locators; machines are named by stable role labels; `tests/check_public_export_hygiene.py` guards both copies in CI and before every release build.
- Release workflow: `tests/check_source_package.py` and `tests/check_public_export_hygiene.py` run before `tools/build_release.py`, the sdist build and the draft-release attachment; `tests/check_release_workflow_gates.py` pins that order.

Intermediate development notes: [DEVELOPMENT_0.5.5.md](DEVELOPMENT_0.5.5.md), [DEVELOPMENT_0.5.14.md](DEVELOPMENT_0.5.14.md), [DEVELOPMENT_0.5.15.md](DEVELOPMENT_0.5.15.md), [DEVELOPMENT_0.5.16.md](DEVELOPMENT_0.5.16.md). Validation reports: [VALIDATION_0.5.0.md](VALIDATION_0.5.0.md), [VALIDATION_0.5.2.md](VALIDATION_0.5.2.md). Evidence rows E-001 to E-106 in the [capability matrix](CAPABILITY_MATRIX.md).

## 0.4.0

Historical note, kept as written for the 0.4.0 release; later results are recorded in the capability matrix.

Standalone multimodal engine, prepared for native cluster qualification.

## Gap decisions

| Earlier gap | Resolution |
|---|---|
| GGUF, quantized LLM serving | Managed, pinned llama.cpp backend; models remain external; explicit file/quantization selection and split-GGUF downloads |
| Tools and structured output | Native function-call request/result/SSE round trip and grammar-constrained JSON schema; agent executes tools outside Titanos |
| Responses API | Native stateless create and SSE; client carries conversation state, tool results and context |
| Continuous batching | Native slots handle simultaneous requests in one model process; configurable slots/context; ordinary PyTorch workers remain serial replicas |
| Asynchronous video | Bounded local queue, persistent atomic records, polling, cancellation, restart interruption and MP4 artifacts |
| Cross-vendor identity | NVIDIA UUIDs; ROCm, XPU and OpenVINO UUIDs where exposed; explicit runtime-address identity otherwise. No name-based substitution |
| Platform-native builds | Checksum-pinned upstream native catalog adds Windows/Linux/macOS x64/ARM64 options, including Windows ARM CUDA/Adreno and Windows ROCm candidates; native qualification remains required |
| Distribution/upgrades | Lightweight engine-only install, native installer, existing numerical profiles, explicit versioned upgrade, static feeds and draft-release workflow |
| Hardware validation | Local qualification aggregator, SSH cluster runner, per-mode smoke tests and HTTP load tool; actual target runs wait for cluster access |

A native binary is a model-serving subprocess owned by Titanos. Its authenticated
loopback port is private. Only selected inference endpoints are forwarded. Native
MCP, shell/file tools, model downloads, router administration and media URL fetching
are not exposed. Models may request function calls; Titanos does not execute them.
A watchdog shuts down the native child even if Titanos dies abruptly.

## Deliberate boundaries

- Universal checkpoint conversion/quantization is not a truthful capability: each
  architecture and accelerator needs a supported exporter, operators and precision.
  The existing validated encoder exporter, prepared-model pipeline and GGUF loader
  remain the explicit paths. Full-size/model-quality qualification is per checkpoint.
- Cross-machine placement, autoscaling, clustering and resource virtualization
  remain responsibilities of an orchestrator such as HiveMind, or of the caller. Titanos owns local inference.
- A duplex realtime voice session needs a voice application or a dedicated realtime
  model adapter. Existing speech/transcription endpoints remain separate operations.
- A desktop GUI is outside this engine build. CLI and interactive HTTP documentation
  expose its operations. No additional frontend framework was added.
- Updates are explicit and versioned. No background self-updater replaces a working
  inference runtime. `upgrade` preserves the old environment and prints the new executable.
- MIG partition lifecycle and universal NPU device identity are not implemented.
  Providers without a stable UUID are labelled accordingly and require local qualification.
- The GitHub draft-release workflow is supplied but has not been run.

This release does not claim every API on every backend. Agent/structured APIs use
llama.cpp; Transformers/Diffusers retain the seven-modality path; ONNX/OpenVINO
retain prepared graphs/embeddings and OpenVINO GenAI text. Unsupported combinations
fail explicitly. Native model splitting and native tensor mode are backend options,
separate from Transformers module sharding and native Transformers TP.

## Evidence and source review

HiveMind source informed the design, especially the tool translation regression
and the distinction between inventory identity and a runtime's device numbering:

- [Tool-calling guide](https://github.com/pgoode41/HM2/blob/eaad2b5ffd00eff225e7f9088b8a90e51939909a/docs/development/TOOL_CALLING_GUIDE_2026-05-14.md)
- [Actual translator](https://github.com/pgoode41/HM2/blob/main/menta_hli/gateway/api/src/routing/translate.rs)
- [GPU runtime resolver](https://github.com/pgoode41/HM2/blob/main/menta_hli/Microservices/resolve_hivemind_uuid_to_runtime_target.py)

The native runtime is llama.cpp **b10809**, commit **5266f24da**, the nightly selected
by upstream's v0.4.0 release. Its [server](https://github.com/ggml-org/llama.cpp/blob/b10809/tools/server/README.md)
and [function-calling documentation](https://github.com/ggml-org/llama.cpp/blob/b10809/docs/function-calling.md)
were inspected. Titanos reuses its inference/protocol implementation.
[HIP visibility](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html)
and [XPU device properties](https://docs.pytorch.org/docs/stable/generated/torch.xpu.get_device_properties.html)
were checked; the implementation refuses invented identities when a pinned runtime
cannot expose one. Detailed test evidence is in VALIDATION.md.
