# Titanos Studio

Titanos Studio is the standalone browser interface served by `titanos serve`.
It uses the same local registry, lifecycle, inference, extension, and job
contracts as the CLI, Python client, MCP, and HTTP API. HiveMind is
not required.

The approved v2 honeycomb-T wordmark and the numbered-bay "Cabinet" theme are
described in [Studio design](STUDIO_DESIGN.md). “Born of HiveMind” identifies design
lineage; Studio and the Titanos engine continue to operate independently. The four UI languages and the Simple/Advanced contract are specified in [Studio language and workflow contract](STUDIO_LOCALIZATION.md); backend choices in the registration form are in [Studio backend controls](STUDIO_BACKENDS.md).

## Simple, Advanced and language

Studio starts in **Simple** mode. It keeps task-first model setup, meaningful
size/license/progress, the prompt, result and one obvious action visible while
hiding backend plumbing and tuning. **Advanced** reveals local registration,
generation controls, diagnostics and stable local device assignments. Both
modes use the same registry and running work; switching does not rewrite an
expert assignment.

The explicit UI languages are English (`en`), Español (`es`), 日本語 (`ja`) and
简体中文 (`zh-CN`). A safe Titanos preference stores mode and locale; an explicit
choice wins over browser-family matching, with English fallback. API keys remain
tab-memory only. UI language does not alter prompts, model output, translation
controls, lyrics or vocal language.

The small catalog/fallback behavior is adapted from the MIT-licensed
HM2-UI / HiveMind UI locale source (HiveMind contributors, 2026). Titanos keeps its
own keys and vanilla implementation; it does not ship HiveMind, React, i18next
or Electron services.

Simple displays the server's automatic recommendation for a compatible visible
accelerator, separately from an observed loaded device. A recommendation is not
proof of execution. **Automatic** allows the backend's normal selection;
**CPU** is an explicit choice that refreshes and language changes preserve.
Existing pinned assignments stay pinned until deliberately changed. Device
metadata can be refreshed without loading a model or changing its assignment.

Advanced offers friendly names backed by stable runtime identities (`GPU-...`,
`HIP-...`, or supported `XPU-...` / `OV-...` mappings). Installed devices that
are masked, lack a runtime, or cannot be mapped safely remain visible with a
reason and cannot be selected. A restricted process can see only CPU while
other hardware is installed; Studio reports that visibility limit. A runtime
identity without a physical join is not an additional proven physical card.

Assignments save one ordered sharding/tensor group or one stable device per
independent replica. Multi-device groups require a known matching runtime
vendor and architecture. Mode availability is backend-specific: llama.cpp's
experimental native tensor split is distinct from PyTorch distributed tensor
parallelism. A missing CUDA collective can block PyTorch GPU TP while CPU Gloo
remains a separate candidate. These capability checks do not qualify a model or
device. Missing/unavailable saved devices refuse instead of silently falling
back to CPU. Unsaved per-model assignments survive language and model changes;
save or discard them before changing the Simple device policy.

## First result

1. Install Titanos from a source checkout (`python -m pip install -e .`; the README lists the PyTorch
   line that Transformers and Diffusers models need) or extract a portable bundle, which includes Python.
2. Run `titanos init`, then `titanos serve --open` (plain `titanos` does the same). Studio opens at `http://127.0.0.1:11435/`.
3. Open **Add a model**, select the task and
   source, and search—or inspect a public model page. Review the exact revision, selected files, model license,
   storage estimate, backend/runtime/device status and unmeasured limitations.
4. Choose **Download and run**. Titanos keeps acquisition progress separate
   from inference jobs, supports cancellation/retry, verifies integrity, then
   registers and loads only the complete model.
   For a supported GGUF chat model, this same action first prepares the
   checksum-pinned server-owned runtime when needed. The server recommends a
   compatible profile from visible hardware and runtime metadata; Advanced
   retains an explicit profile/device choice. An existing CPU runtime does not
   satisfy a different selected GPU profile. Selecting the CPU profile with an
   automatic device explicitly chooses CPU; a CPU profile plus an explicit GPU
   is rejected before acquisition. Changing these choices after reviewing a
   plan requires reviewing a new plan. Activation sends the selected profile
   again, so a concurrent managed-runtime change fails instead of binding the
   wrong executable; each newly activated model pins its verified runtime.
   An already-active acquisition with a different or unrecorded profile shows
   a review message; Studio does not automatically reload or migrate it.
   This workflow never installs Python or
   accepts executable URLs from the browser. Explicit server catalog overrides
   remain operator policy.
5. Enter the task input and generate the result. Restarting Titanos reuses the
   prepared model and packaged runtime.
6. When an API key is configured, enter it in
   **Connection**; Studio keeps it only in the current tab's memory.

Inference itself never downloads a model. The explicit provider/acquisition
workflow is documented in [Model providers](MODEL_PROVIDERS.md). Replacing or
removing a registration changes the registry only; it does not delete model or
voice files. Expert local-folder registration remains below the model library.

## Registration

Studio can safely register these combinations through the existing HTTP
registry:

- Transformers: chat, vision, embeddings, transcription, speech, text
  translation, learned image-to-text/OCR, object detection, sentiment,
  reranking, text-pair classification, relative depth, and prompted SAM
  segmentation.
- Diffusers: image and video, including an optional validated weight suffix.
- Qwen3-TTS: speech, with an optional JSON object mapping voice names to
  absolute local WAV paths.
- Sherpa ONNX: pinned GitHub release speech archives and the pinned speaker-diarization
  bundle, on CPU, using the separately packaged 1.13.8 Python/native runtime contract.
- ACE-Step 1.5: complete pinned external music bundles using an owned isolated
  runtime. Music and full-song jobs preserve channels and publish WAV artifacts;
  see [Music](MUSIC.md) for source/runtime and native-quality boundaries.
- ONNX: embeddings, arbitrary tensors, and CPU-only VAD. Prepared graph names
  are relative to the checkpoint directory.
- OpenVINO: embeddings and arbitrary tensors from prepared graphs.
- OpenVINO GenAI: chat and image from complete supported exports.
- Tesseract: CPU image-to-text using a folder of LSTM `.traineddata` files.
- ONNX-Diffusers: prepared SD1 image generation on CPU or an explicitly
  selected DirectML adapter. DML registration binds both the current adapter
  index and lowercase DXGI LUID.

`llama-cpp` chat/vision remains CLI-only because it requires native GGUF and
runtime options. XTTS speech remains CLI-only because it requires an explicit
trusted-pickle decision. Studio shows these choices and their `titanos add`
route but does not weaken the registration policy or send an incomplete HTTP
request.

**Load known devices** reads the cached Titanos inventory. Refresh that
inventory first with `titanos devices --backend all --refresh` (or one backend;
see [device identity](DEVICES.md)) when current hardware is needed.
Runtime addresses, vendor identities, and DXGI LUIDs have different
persistence; inventory presence is not inference qualification.

## Model lifecycle and identity

The model list distinguishes the configured backend/device, static
capabilities, source repository/revision when declared, and current catalog
state. **Load** and **Unload** call the explicit lifecycle API. **Runtime**
is available only after the catalog reports the model loaded, because
the details operation can otherwise load weights.

For chat, **Require this chat model to be loaded** appears only when the server
advertises revision 1 of that contract. It sends `require_loaded:true`; an
unloaded target fails instead of starting. Leaving it unchecked preserves
ordinary lazy loading.

**Remove** asks for confirmation and can fail while a worker,
request, or job owns the model. It never deletes the external checkpoint.

## Tasks

- Chat and vision stream text and retain conversation history until **New
  conversation**.
- Embeddings show vector count and dimensions with the complete result in
  details.
- Image generation, speech, and video use durable asset jobs. Image-reference
  video remains available for compatible pipelines.
- Image editing preserves the existing source, binary-alpha mask, and optional
  outside-pixel copying contract.
- Transcription supports an optional registered VAD gate. The audio action can
  instead translate speech to English directly with multilingual Whisper or
  through a selected ASR plus NMT pair. Composed translation accepts explicit
  NMT source and target tokenizer IDs; a blank target inherits the translation
  model's `target_language` default and then the server's `eng_Latn` fallback.
- The audio action selector switches one audio upload between **Transcribe**,
  **Translate speech to English** and **Transcribe with speakers and overlap**.
  The last runs transcription plus anonymous speaker diarization, with optional
  transcript analysis by a registered sentiment or chat model, through
  `/v1/audio/analysis`; a diarization model on its own posts to
  `/v1/audio/diarizations`.
- Music is a durable asset job over an ACE-Step bundle: instrumental or song mode,
  optional lyrics, and the duration, vocal-language, tempo, key/scale and
  reference-audio controls that the loaded bundle declares. Results are WAV
  files listed under Jobs. Real music generation is not yet qualified; see [Music](MUSIC.md).
- Learned OCR/captioning, native Tesseract OCR, detection, text translation,
  sentiment, text-pair classification, rerank, and standalone VAD use their
  typed routes and task-specific controls.
- Relative depth displays a normalized preview and offers the original
  little-endian float32 values. The values are relative inverse depth, not
  metres, and are not physically comparable across scenes.
- Prompted segmentation accepts one box or one positive/negative point set.
  Masks remain in native order with uncalibrated predicted-IoU values; Studio
  does not invent labels or select a supposedly best mask.
- The document-answer panel runs the existing embedding → rerank → owned native
  LLM workflow over an uploaded JSON passage array.
- Arbitrary tensor models can be registered, loaded, and inspected, but Studio
  does not guess tensor shapes or binary payloads. Use `/api/infer`, the CLI,
  or an agent client.

No 3D, training, hosted-provider state or absent adapter is presented as a
Studio task; arbitrary tensor models are API-only.

## Defaults, limits, errors, and cancellation

Blank controls are omitted from requests. The server therefore resolves an
implemented control in this order: explicit request, model default, then
process default. Explicit zero and false values are preserved. Studio reads
the current server limits from `/api/info`; schema bounds and checkpoint
physical limits can be lower and remain authoritative.

Errors retain the server message/code and add the next useful action for common
load, registry, capability, validation, and size failures. **Stop** aborts the
request that owns the form. Once an asset-job ID exists, Stop also asks the
server to cancel that exact job. Job records can be refreshed, opened, saved,
or cancelled after leaving and returning to the page.

All server, model, source, and error text is rendered as text rather than HTML.
New controls are keyboard reachable, status/errors use live regions, result
headings receive focus after completion, and generated media has accessible
labels.

## Cloud providers and routing

When the server reports a broker block in `/api/info` and the caller's key (or
keyless loopback) holds `broker:read` (`GET /api/keys/self`), Studio shows bay
**07 Cloud and routing** after Jobs, in Simple and Advanced. Every request goes
to this Titanos server's `/api/broker/*` and `/api/keys/*` routes through the
same `api()` helper and tab-memory key as the rest of Studio; Studio never
contacts a provider host. Each of those requests carries the header
`X-Titanos-Studio: 1`. A custom header forces a CORS preflight that Titanos
never approves, so a page on another origin cannot send it; with a bearer key
the key decides as before.

### Connect a provider

1. Open **07 Cloud and routing**. Under **07.1 Connect a provider**, pick the
   provider. The list comes from the server's provider profiles; a provider is
   marked "connected" only when an upstream of that profile is enabled, has
   a credential and is ready (the server can read that credential: a key
   referenced by an environment variable counts only when the variable is set
   in the server's own environment), "NAME found" when one of its well-known
   credential variables (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
   `ELEVENLABS_API_KEY`) is set where Titanos runs, "configured, not ready"
   when its upstream is enabled but cannot serve (for example after
   `titanos broker connect --from-env NAME` from a shell whose variable the
   server does not hold), and "configured, not enabled" when its upstream
   exists but is disabled. Connecting either one checks the key from the
   server, binds it and enables the upstream.
2. Choose the key:
   - **Use a key from this server's environment**: pick the variable by name.
     From the keyless Studio this works only for the provider's well-known
     variables and its default address; for an upstream with another base URL,
     paste the key or use the CLI on the host.
     Studio only ever sees the names and whether they are set
     (`GET /api/broker/env-candidates`); Titanos stores a reference to the
     variable, never its value.
   - **Paste a key**: a password field that is never prefilled. The value is
     sent once in the connect request, then the field is emptied and replaced;
     it is not kept in the page, browser storage or Studio's state, and the
     server never sends it back.
   - **No key** appears for profiles that run without a credential.
3. Optional: a base URL (Advanced; required for profiles without a default,
   such as an OpenAI-compatible server), and publishing the provider's model
   ids as plain aliases (Advanced). On a keyless server, a switch (on by
   default) lets Studio and other keyless programs on this computer use the new
   provider (`policy.anonymous_cloud`); otherwise only a key holding cloud
   access can call it.
4. Press **Connect**. `POST /api/broker/connect` creates or updates the upstream
   from the profile, stores the credential, checks it against the provider's
   model list, enables the upstream, raises the cloud policy from off to
   explicit (never above `TITANOS_BROKER_CLOUD_CEILING`) and lists the models
   it found. Studio shows the result (upstream, validation, credential
   fingerprint, cloud policy, keyless use, the discovered models and its own
   next steps: whether the cloud policy still refuses cloud calls, whether
   keyless programs can reach the provider, and an example model id) and
   refreshes the model list, where the new models appear under **Cloud
   routes** with the orbit glyph and the provider's name.

Who may connect: a key with `broker:admin` and `credentials:write` (over https
or a loopback address), or, on a keyless standalone server bound to loopback,
Studio itself (the server's `keyless-studio` principal). A keyless server
without that support, or a key without `credentials:write`, gets the command to
run instead: `titanos broker connect <provider> --from-env NAME` (or
`--from-stdin` to paste a key). Nothing is shown as connected unless the server
answered with a connect result: a refused key (422
`credential_validation_failed`), a vault that stores references only (503) or a
server without the route (404) is reported as a message with the server's own
reason (no hint meant for local models is appended), and the upstream stays as
it was.

### The rest of the bay

- A readout: role, cloud policy (effective, file and ceiling), vault, upstream
  counts, egress and whether keyless callers may use cloud routes.
- **07.2 Cloud routes**: one tile per upstream with its credential reference,
  fingerprint, validation, breaker state and catalog. `broker:admin` adds the
  enable switch (labelled with its current state) and a Test action.
- Advanced only, folded: **Add or replace credential** (`credentials:write`; a
  secret value, or with the `admin` scope or the keyless Studio an environment
  variable name or an owner-only file path on the server host). After Save the
  secret field is emptied and replaced; a saved credential is bound to its
  upstream when the principal may edit upstreams. The form is disabled with a
  warning when the page is neither https nor loopback.
- Advanced only: aliases and virtual models, policy (the keyless Studio may
  change keyless cloud use; plain http to LAN upstreams stays a CLI or keyed
  setting), usage and recent decisions (metadata only) with an Explain routing
  dry run, and keys (`keys:admin`; a new key's secret is shown once, held only
  in page memory, and gone when dismissed or when the key in use changes).
  Keys are never minted keylessly. Empty usage and decision tables say so
  ("No cloud requests recorded yet.").

The panel refreshes on its own, independent of the jobs list, so a key without
the `jobs` scope still sees it; Connect with another key clears the previous
rendering first.

### Cloud models in the composer

Cloud, router and virtual models appear in the model rail under **Cloud routes**
(a cyan orbit glyph, the id with its `@upstream` suffix dimmed, then the task
and the provider on the second line) and in the model picker as
`model · task · provider`. A cloud model whose id
names one modality (for example `gpt-image-1`, `tts-1`, `whisper-1` or
`text-embedding-3-small`) is listed under that task; moderation models are not
listed. Selecting one turns the select's border cyan and shows the note "This
route leaves your machine. Titanos sends only what you run here." with the
provider, and the readout below reads `Route · Cloud · provider`. The same forms
run local and cloud targets: chat and vision stream, embeddings and
transcription answer directly, and speech and images run as jobs. **Where to
run** sends `X-Titanos-Location` (this computer only, prefer this computer,
automatic, cloud only) with inference requests.

Every result ends with a receipt: "Recorded result" (or "Failed"), "External"
for a cloud route, then the provenance from the response headers (`Served by
cloud:openai · gpt-4o-mini@openai · fallbacks 0`) or the model and device, the
elapsed time, tokens (chat asks the stream for its usage chunk) or the size of
the text or asset, the cost when the call was priced (read from the decision
log by a principal with `broker:read`), and the time the result was recorded. A
job reopened from **Jobs** keeps the job's own finish time, not the time it was
opened. Images show their pixel size under them, so a very small image still
reads as a result. Device assignment and the require-loaded switch do not
appear for a cloud model, since they do not apply to it.

If Titanos stops answering, Studio says "Titanos is not reachable at
<host:port>. Start it or press Refresh.", disables Run and dims the model rail
until a refresh succeeds.

## Verification scope

The dependency-free Node checks execute the real Studio JavaScript with
DOM/fetch doubles and prove request serialization, state ownership, safe
rendering, and default omission. Python `TestClient` checks prove HTTP registry,
lifecycle, default, and job control paths with fake engines. Neither class
proves a real browser, model quality, native binary, or hardware placement.
`node tests/check_broker_ui.js` covers the Cloud and routing panel: the
write-only credential and connect fields, the `X-Titanos-Studio` header, the
connect flow by variable name and by pasted key, its typed messages for a
missing route, a refused key or an unexpected answer, the once-only key secret,
scope-gated sections, the keyless note, the insecure-origin refusal, badges and
the provenance line, the provider labels ("connected" only for a usable
upstream), the accessible names of the key fields, Studio's own next steps, the
empty-table sentences, and broker refusals without local-model hints. `node tests/check_studio_broker_browser.js` (browser tier,
Playwright) drives Studio in a real browser against a real `titanos serve` and
a loopback OpenAI mock: an environment-sourced credential saved by variable
name, validated and bound, the upstream enabled from its switch, a chat through
a cloud alias with its External badge and provenance line, a listed decision,
and on a keyless server the one-step connect: by variable name to an upstream
whose base URL is the mock (refused, since the keyless Studio sends an
environment credential only to a provider's default address, and reported with
the server's reason), then by a pasted key (validated against the mock,
enabled, keyless use allowed, models discovered), followed by a keyless cloud
chat and header-less keyless writes refused. A typed secret refused by a
references-only vault is checked on the keyed server. The typed and pasted secrets never appear
in the page, browser storage or the mock's requests. It needs Python with the
project installed (`TITANOS_PYTHON`) besides Node and Playwright, and it proves
the panel against a mock, not against a real provider.

A release acceptance run must separately record:

- a visible Edge/Chrome flow for layout, keyboard focus, progress, cancellation,
  and downloads;
- an actual compatible model/server result when one is available; and
- the exact model, backend, runtime, device, inputs, output, and cleanup.

See [Studio backend controls](STUDIO_BACKENDS.md), [configuration](CONFIGURATION.md),
[device identity](DEVICES.md), [relative depth](DEPTH.md), [segmentation](SEGMENTATION.md),
[Qwen3-TTS](QWEN_TTS.md), [ONNX images](ONNX_IMAGES.md), [OpenVINO GenAI images](GENAI_IMAGES.md),
[image masks](image-masks.md), [ASR/VAD gating](ASR_VAD_GATING.md), and
[retrieval answer](RETRIEVAL_ANSWER.md) for contract-specific requirements.
