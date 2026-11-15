# Titanos documentation

Titanos is a standalone local inference engine. It serves model weights that stay outside the application through HTTP, MCP, a CLI, a Python client and a browser interface called Studio. This page is the map of `docs/`. Each entry says what the page gives you.

Two markers appear below. A *generated* page is rebuilt by a tool under `tools/` from a JSON file beside it and is held to that output by a test; do not edit it by hand. A *historical* page describes one past version and is kept for provenance; it opens with a banner naming that version and is not updated.

Every hardware or model claim in these pages is bounded to the model, package and platform it was recorded on. The vocabulary is fixed in the [capability matrix](CAPABILITY_MATRIX.md): **Implemented** (present in source), **Profile available** (an install profile exists), **Bounded native pass** (a recorded run on a named package and platform), **Source-tested**, **Partial/failed** and **Unverified/planned**. A feature in source is not hardware proof, and a pass on one model, package and platform does not qualify another.

## Use it

| Page | What you get |
|---|---|
| [Titanos](../README.md) | What Titanos is and the shortest path from a source checkout to a running server. |
| [Getting started](GETTING_STARTED.md) | Per-platform install: which runtime profile to pick, what `titanos doctor` and `titanos devices` should show, and what is qualified versus a candidate. |
| [Interfaces](INTERFACES.md) | The five ways to talk to Titanos (HTTP, MCP, CLI, Python client, Studio), what they share, and which page covers each. |
| [Titanos Studio](STUDIO_USAGE.md) | The browser interface served by `titanos serve`: registering, loading and running models without the CLI. |
| [Studio backend controls](STUDIO_BACKENDS.md) | Which backend and task combinations the registration form offers, and how the choice narrows the task list. |
| [Studio language and workflow contract](STUDIO_LOCALIZATION.md) | The four UI languages, the Simple and Advanced modes, and what each mode shows. |
| [Titanos Studio design](STUDIO_DESIGN.md) | The identity, theme and design lineage of Studio. |
| [Configure Titanos](CONFIGURATION.md) | Environment variables, per-model defaults and request values, their precedence, the limits and settings catalog, and what happens when a port is taken. |
| [Model providers and one-stop setup](MODEL_PROVIDERS.md) | Find, inspect, pull and register a model: the task-first flow, and local-folder registration for experts. |
| [Model adapters: choose the right files and runtime](MODEL_ADAPTERS.md) | Which files and runtime each backend needs before a checkpoint can load; `titanos capabilities` inspects a registry entry without a server. |
| [Current model tensor storage](MODEL_STORAGE.md) | The `tensor_storage` report in model details: where each component's tensors sit and in which dtype. |

## Integrate it

The generated route and protocol references (every HTTP operation, MCP tool, CLI subcommand and client method; every upstream field and what Titanos does with it) are listed under [How it is qualified](#how-it-is-qualified).

| Page | What you get |
|---|---|
| [MCP interface](MCP.md) | Connect an MCP client over stdio or Streamable HTTP (`/mcp/`), the tools, resources and prompts Titanos exposes, and how to embed the server in your own ASGI application. |
| [HTTP protocol compatibility](PROTOCOL_COMPATIBILITY.md) | Which wire shapes the HTTP API speaks (OpenAI, Responses, Ollama, the HLI media routes) and where each differs from upstream. |
| [Extended inference API](API_PLUS.md) | `GET /v1/extensions` and the typed operations behind it: sentiment, translation, OCR, detection, audio translation and the other local media routes. |
| [Stateless Responses](RESPONSES.md) | `POST /v1/responses` over the local chat and vision backends, and what still passes through to native llama.cpp. |
| [Cloud broker](BROKER.md) | The optional cloud broker inside `titanos serve` (off by default): upstreams, aliases and virtual models, routing and fallback, keys and budgets, provenance headers, error codes and what leaves the host. |
| [Cloud broker security](BROKER_SECURITY.md) | How the broker stores credentials, binds them to one origin, authenticates and limits callers and keeps secrets out of logs; residual risks and the threat model. |
| [Topology and roles](TOPOLOGY_ROLES.md) | The `standalone`, `broker` and `managed` roles, managed headers and delegation, HiveMind integration and port guidance. |
| [Using several standalone servers](MULTIHOST_CLIENT.md) | The optional Python `Pool` for spreading requests across independent Titanos servers. |

## Run it on your hardware

| Page | What you get |
|---|---|
| [Hardware, runtime builds, and distribution](HARDWARE.md) | Runtime profiles per platform, the PyTorch and compiled-runtime matrix, what the installer does, and local release hosting. |
| [Device discovery and accelerator assignments](DEVICES.md) | `titanos devices`: installed hardware versus runtime visibility, explicit placement and observed execution. |
| [Accelerator execution and worker groups](ACCELERATOR_EXECUTION.md) | Placement rules inside the process's device visibility, worker groups, and why a requested GPU is never replaced by CPU. |
| [Replicas, model sharding, and tensor parallelism](MULTIGPU.md) | The three explicit multi-GPU choices and how to configure each; Titanos never turns one into another when a load fails. |
| [Managed GGUF runtime](NATIVE.md) | The pinned llama.cpp runtime: install per platform, context and slots, tool calls, and the native controls. |
| [DirectML on Windows](DIRECTML.md) | Prepared ONNX graphs on Windows GPUs: install, `provider_options`, `titanos doctor`, and the adapters that were tested. |
| [Windows native command cleanup](WINDOWS_COMMAND_CLEANUP.md) | How native child processes are owned in a private Windows Job and what a failed cleanup reports. |
| [Operations](OPERATIONS.md) | Run `titanos serve` as a service (`titanos service` on Windows, Linux and macOS), upgrade, logs and rotation, the readiness and metrics routes an orchestrator can poll, HiveMind registration, and backups of `TITANOS_HOME` with the vault caveat. |
| [Performance and tuning](PERFORMANCE.md) | What the 2026-10-06 pass observed per path (llama.cpp slots and context, the Transformers chat path, speech, embeddings, HTTP and startup), the two defaults that changed because of it, the defaults measured and kept, and the knobs to turn for a server that is utilized; every figure labelled observed on one shared host. |

## Modalities

Text

| Page | What you get |
|---|---|
| [Text sentiment](SENTIMENT.md) | `POST /v1/sentiment`, the CLI, client and MCP forms, and what a result does and does not claim. |
| [Sentiment language and composition](SENTIMENT_LANGUAGE.md) | Categorical LLM output and the two-stage classifier-plus-LLM composition behind sentiment. |
| [Native text-pair classification](PAIR_CLASSIFICATION.md) | Premise/hypothesis (NLI) classification with raw labels and scores. |
| [Text retrieval](RETRIEVAL.md) | Normalized embeddings and the cross-encoder reranker, with the rules for overlength input. |
| [Answer from supplied passages](RETRIEVAL_ANSWER.md) | `POST /v1/retrieval/answer`: embed, rerank and answer over caller-supplied passages, with no fetching and no vector store. |
| [Attributed text and audio translation](TRANSLATION_PROVENANCE.md) | `include_provenance`: stage text, hashes and the provenance the backend actually supplies. |

Audio

| Page | What you get |
|---|---|
| [Audio transcription and speech activity](AUDIO.md) | Transcription formats and metadata, timestamps, VAD, cancellation and the request deadline. |
| [Whisper transcription context](WHISPER_CONTEXT.md) | The optional `prompt` that conditions Whisper on vocabulary without editing the transcript. |
| [Configured VAD before transcription](ASR_VAD_GATING.md) | Run a registered VAD model before ASR so non-speech input can return empty text instead of an invented transcript. |
| [Anonymous diarization and composed audio analysis](AUDIO_ANALYSIS.md) | Anonymous speaker intervals alone, or combined with transcription and transcript sentiment. |
| [Qwen3-TTS speech](QWEN_TTS.md) | Qwen3-TTS Base through `POST /v1/audio/speech`: install, voice mapping, request shape and the recorded scope. |
| [Local music and complete songs](MUSIC.md) | Music as an asynchronous asset task: the request, memory admission and caps, and what is and is not qualified. |
| [Isolated music runtime contract](MUSIC_RUNTIME_CONTRACT.md) | The child-interpreter contract for music: device selection, requested versus observed device, and the metadata that proves execution. |

Images and video

| Page | What you get |
|---|---|
| [Masked image editing](image-masks.md) | The `alpha-binary-v1` PNG mask contract for `/v1/images/edits`, `preserve_unmasked`, and the `masking` response object. |
| [InstructPix2Pix image editing](INSTRUCTION_EDIT.md) | Instruction-following edits with an eight-channel InstructPix2Pix checkpoint, kept separate from SD/SDXL editing. |
| [Prompted SAM masks](SEGMENTATION.md) | One box or one point set in, native mask alternatives out as binary RGBA edit masks. |
| [Relative depth](DEPTH.md) | Relative Depth Anything inference: accepted checkpoints, rejected variants, and the HTTP, CLI and client calls. |
| [Native printed OCR](TESSERACT.md) | The Tesseract backend: register `.traineddata`, `POST /v1/images/ocr`, controls, process ownership and limits. |
| [OpenVINO GenAI image adapter](GENAI_IMAGES.md) | SD/SDXL through OpenVINO GenAI: pulling a prepared export, what the loader checks, and the recorded scope. |
| [Standalone ONNX image generation](ONNX_IMAGES.md) | `onnx-diffusers`: the SD1 ONNX pipeline, its required files, DirectML provider setup and its limits. |
| [Local asynchronous video jobs](VIDEO_JOBS.md) | `POST /api/video/jobs`: submit, poll, cancel and fetch the MP4 result. |

## Build and distribute it

| Page | What you get |
|---|---|
| [Standalone executable distribution](DISTRIBUTION.md) | `tools/build_binary.py`: what a portable application directory contains, acceptance, public-export hygiene and install path length. |
| [Releasing Titanos](RELEASING.md) | What a GitHub release carries (wheel, sdist, catalog, checksums), how the workflow builds it from a tag, and how to reproduce it locally. |
| [Titanos release notes](RELEASE_NOTES.md) | Changelog, newest first; the release workflow attaches it to the draft GitHub release. |
| [Standalone CPU text bundle](CPU_TEXT_BUNDLE.md) | The `text` build profile: Titanos, PyTorch, Transformers and MCP frozen into one folder. |
| [Standalone CPU text and audio profile](BUILD_TEXT_AUDIO.md) | The `text-audio` profile: the CPU text runtime plus ONNX Runtime VAD and the source-bound FFmpeg decoder. |
| [CPU text and vision bundle](BUILD_TEXT_VISION.md) | The `text-vision` profile: TrOCR, YOLOS and the CPU vision adapters. |
| [OpenVINO GenAI standalone profile](GENAI_BUNDLE.md) | The `genai` profile: the OpenVINO GenAI text-generation adapter with OpenVINO, its tokenizers, HTTP and MCP. |
| [Portable music runtime construction](MUSIC_PACKAGING.md) | `tools/music_package.py`: assemble the optional external ACE runtime from owned inputs without running pip or a model. |
| [ACE music runtime](../tools/music_runtime/README.md) | The adapter behind the music task and the notes that travel with the runtime. |
| [Dependency and distribution boundaries](DEPENDENCY_PROVENANCE.md) | What the Apache-2.0 source license does not cover: dependencies, native runtimes, checkpoints and voice references. |
| [Publication scope](PUBLICATION_NOTES.md) | What this repository contains, how exported copies are checked, and how older version strings relate to this tree. |

## How it is qualified

Evidence is recorded in three layers. **ENGINE_CONTRACT** means the engine executed the intended modality with coherent output, valid encoding and storage, declared controls, protocol, routing and lifecycle. **MODEL_QUALITY** records how good the output was. **HUMAN_PRESENTATION** covers demonstration media and human review. A coherent wrong answer is a model-quality observation, not an engine failure; a malformed output or a broken control is. "Vega" in these pages is the name of the project's reference narration voice, used as a speech fixture; it is not a model or a product.

| Page | What you get |
|---|---|
| [Titanos status](../STATUS.md) | What is verified, which model-quality observations are retained, and what engineering and delivery work remains. |
| [Capability and qualification matrix](CAPABILITY_MATRIX.md) | Source capabilities, hardware scope and the evidence summaries: one row per evidence id (E-001 onward), each bound to the package and platform it was recorded on. |
| [Progress checklist](PROGRESS.md) *(generated)* | The 95 stable checklist items with their acceptance layer, closure state and remaining condition, rendered by `tools/render_progress.py` from [progress.json](progress.json). |
| [Execution matrix: every model x every platform slot](EXECUTION_MATRIX.md) *(generated)* | One state per model-by-slot cell (pass, fail, partial, untried, not applicable) with the evidence behind it. Also as [CSV](execution_matrix.csv) for spreadsheets and a self-contained [HTML](execution_matrix.html) view to download and open locally; source [execution_matrix.json](execution_matrix.json). |
| [Representative roster and per-family gates](ROSTER.md) *(generated)* | The named packages, platforms, models and workloads the records bind, the requested-but-missing families, and the gate text each family must meet; source [roster.json](roster.json). |
| [Matrix kit: run the execution matrix on any host](MATRIX_KIT.md) | Turn one host into execution-matrix receipts: detect its slots, run every runnable model, write one receipt per cell in the matrix's own shape. |
| [Historical validation summaries and source checks](VALIDATION.md) | Summaries of the 0.3.0 to 0.4.0 validation rounds; they qualify nothing in the current source. |
| [Option parity and effective configuration](OPTION_PARITY.md) | Which Ollama and llama.cpp controls map to Titanos controls, which are fixed policy and which are refused, checked against the source on the date the page names. |
| [Protocol parity and deviation map](PROTOCOL_DEVIATIONS.md) *(generated)* | Field by field: what Titanos does with each upstream request field, response field and route, with the code reference that binds it; source [protocol_deviations.json](protocol_deviations.json). |
| [Broker support matrix](BROKER_MATRIX.md) *(generated)* | Every broker surface against every provider profile, stream mode, error class, role and feature, each cell pass (mock), live-pass, not applicable or untried with the evidence or reason; source [broker_matrix.json](broker_matrix.json). |
| [Route-to-typed-contract matrix](ROUTE_CONTRACT_MATRIX.md) *(generated)* | Every HTTP operation, MCP tool, CLI subcommand and Python client method bound to its request and response models; the basis line records the source commit it was last built from. Source [route_contract_matrix.json](route_contract_matrix.json). |
| [Fatal DirectML context containment](FAULT_CONTAINMENT.md) | What happens after a fatal DirectML device-removed error: the failed context is retained and never reused, and the owning process must be restarted. |
| [Profiling failure and completed results](profiling-failure-policy.md) | When profiler collection raises `MemoryError`, the completed result is still returned and the failure is reported. |

## History

The first row is the index of the historical material; each page after it is *historical*: it describes one past version, opens with a banner naming that version, and is not updated.

| Page | What you get |
|---|---|
| [History](history/README.md) | Dated records of how the current ledgers came to be; nothing there changes a current claim. |
| [Titanos 0.5.0 validation](VALIDATION_0.5.0.md) | The 0.5.0 report: native inference observed per platform, protocol checks, and the standalone executable qualification of that version. |
| [0.5.2 hardware validation](VALIDATION_0.5.2.md) | The 0.5.2 hardware window: DirectML, OpenVINO, VitisAI and QNN results on the named devices. |
| [0.5.5 development changes](DEVELOPMENT_0.5.5.md) | Sentiment composition, the reranker, transcription metadata and VAD as introduced in 0.5.5. |
| [0.5.14 development changes](DEVELOPMENT_0.5.14.md) | Masked SD/SDXL inpainting and relative depth as introduced in 0.5.14. |
| [0.5.15 development changes](DEVELOPMENT_0.5.15.md) | `n` on image edits and the OpenVINO GenAI inpainting pipeline repair in 0.5.15. |
| [0.5.16 development changes](DEVELOPMENT_0.5.16.md) | Prompted SAM segmentation as introduced in 0.5.16. |
| [GEN-01 control source validation](GEN01_SOURCE_VALIDATION.md) | The September 2026 source review of an inactive validation controller; the reviewed tree is not in the published source copies. |

## Contribute

| Page | What you get |
|---|---|
| [Contributing to Titanos](../CONTRIBUTING.md) | Development setup, the checks to run before a pull request, the coding and documentation conventions, the sign-off rule and how reviews work. |
| [Security policy](../SECURITY.md) | The threat model in plain words, how to report a vulnerability privately, and which versions receive fixes. |
| [Community conduct](../CODE_OF_CONDUCT.md) | Expectations for discussion and review, and where to raise a conduct concern. |
| [Running the Titanos checks](../tests/README.md) | The check tiers, the runner, what each tier needs installed, per-platform notes, and the conventions for adding a check. |
| [Assets](assets/README.md) | The hand-authored architecture and runtime-selection diagrams and the Studio screenshot used by the README, with their provenance, palette source and retake procedure. |

## Machine-readable files

`progress.json`, `roster.json`, `execution_matrix.json`, `execution_matrix_spec.json`, `protocol_deviations.json`, `route_contract_matrix.json` and `broker_matrix.json` are the sources behind the generated pages and the ledgers. The tests `tests/check_progress.py` (with `tools/render_progress.py --check`), `tests/check_roster.py`, `tests/check_execution_matrix.py`, `tests/check_protocol_deviations.py` and `tests/check_route_contract_matrix.py` bind the JSON to the Markdown and to the source.

## Not published

The maintainers' dated lab records (`docs/evidence/`), the 0.4.0 cluster handoff runbook and the GEN-01 control tree are excluded from every exported copy by `tools/public_export.json`. The capability matrix cites evidence by id; where a page links into `docs/evidence/`, the link resolves only in the maintainers' repository. `IMPLEMENTATION_PLAN.md`, which `roster.json`, `execution_matrix.json` and the pages rendered from them cite by line number, is the maintainers' private working plan: the citations record provenance, and the file is not in the repository.

## Conventions

- One H1 per page, sentence case, no date in the title. The H1 is the link text this index uses.
- Write a date only where it is a fact (a recorded run, a release, a decision). In prose use the `26 Sep 2026` form. Hand-maintained ledgers (`STATUS.md`, `CAPABILITY_MATRIX.md`, `HARDWARE.md`) and the generated `PROGRESS.md` open with `Updated **YYYY-MM-DD**.` on line 3, matching the ISO dates in the JSON they summarize; that date must not lag the JSON. Generated pages keep their generator's first line. Feature and reference pages carry no date.
- Bold is for a defined term on first use and for the status vocabulary (**Implemented**, **Profile available**, **Bounded native pass**, **Source-tested**, **Partial/failed**, **Unverified/planned**; **ENGINE_CONTRACT**, **MODEL_QUALITY**, **HUMAN_PRESENTATION**). Never for emphasis.
- Claims name their scope: implemented (in source), source-tested, a bounded native pass on a named model, package and platform, or unverified. Never "supports X" without that scope. A coherent wrong answer is a MODEL_QUALITY observation, not an engine failure.
- Backticks for commands, paths, routes, environment variables, JSON keys, version pins and hashes; not for product names (Titanos, Studio, HiveMind, OpenVINO, DirectML, llama.cpp).
- No lab locators and no narrative actors. Paths in examples are placeholders such as `/path/to/models`; hosts are role labels; records are cited as `<lab-records>/...` or by evidence id. Write "Titanos", "the operator", "the maintainers", or a dated fact.
- A space between a number and its unit, version or code: `PyTorch 2.8`, `schema 3`, `503`.
- Generated pages change only through their JSON and the tool named in their first line, followed by the matching `tests/check_*.py`.
- The JSON ledgers cite some Markdown lines by number. Never insert lines above a cited line; edit in place, or update the citing JSON and re-render. Cross-links are appended to existing paragraphs for the same reason.
- Historical pages carry the banner and are otherwise frozen except for defect fixes (encoding, spacing, dead links).
- Newlines and encoding: Git stores LF for every Markdown file (`docs/VALIDATION_0.5.0.md` is the one CRLF exception), and `.gitattributes` pins the two route-contract files, `*.svg` and itself to LF on every checkout. Keep a file's existing working-tree newline style when editing it; new files use LF. UTF-8 without a byte-order mark.
- New pages are `docs/UPPER_SNAKE_CASE.md` and are added to this index in the same commit. Existing paths are never renamed or moved, because the JSON ledgers cite them.
