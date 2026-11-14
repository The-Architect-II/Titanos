# Titanos status

Updated **2026-10-05** for source `0.5.28.dev0+accelerators1` (commit `fd5638a`). This page says what Titanos does today, where that has been proven, and what remains open. It describes a development source snapshot: not a binary release, and not a claim that every model runs on every device.

## How to read a claim here

Every result on this page and in the linked ledgers is bounded to the named model, package, platform and workload that produced it. Three acceptance layers are kept apart and never traded against each other:

- **ENGINE_CONTRACT**: a trained model actually executed and returned coherent output of the intended modality, with valid encoding and storage, the declared controls, protocol, placement and routing, and a complete lifecycle (load, request, unload, cleanup). A factually wrong answer does not fail this layer; a broken schema, an ignored control or a leaked process does.
- **MODEL_QUALITY**: factual, semantic, perceptual and cosmetic results, recorded beside the engine result and never used to hide it.
- **HUMAN_PRESENTATION**: demonstrations, narration and human watch/listen acceptance.

Five terms carry fixed meanings. **Implemented**: an adapter, schema or path exists in source. **Profile available**: a compatible runtime catalog entry exists; nothing was installed or executed. **Source-tested**: deterministic tests, not hardware proof. **Bounded pass**: the named model/package/platform/workload passed the recorded gate. **Untried**: no execution with matrix-cited evidence. A passed unit test, a device name, a valid file or an HTTP 200 alone qualifies nothing, and a later pass never rewrites an earlier failure.

## Numbers at a glance

| Ledger | Current state | Source of truth |
|---|---|---|
| Plan checklist | 95 stable rows: 32 closed within recorded scope, 63 open (59 partial, 2 missing, 1 active, 1 planned). By primary layer: ENGINE_CONTRACT 88 rows, 29 closed; MODEL_QUALITY 3, 3 closed; HUMAN_PRESENTATION 4, 0 closed | [progress.json](docs/progress.json), rendered as [PROGRESS.md](docs/PROGRESS.md) |
| Execution matrix | 56 individual models x 22 platform slots = 1,232 cells (built 2026-09-28, after E-108): 242 pass (every pass upheld by an independent verifier), 9 fail, 9 partial, 274 untried (205 runnable by construction, 67 on the two Linux slots no host has run, 2 awaiting evidence binding), 698 not applicable by construction | [EXECUTION_MATRIX.md](docs/EXECUTION_MATRIX.md), generated from [execution_matrix.json](docs/execution_matrix.json) |
| Evidence rows | E-001 to E-108, one bounded summary each with its limits | [CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) |
| Representative roster | 284 rows: 97 qualified_bounded, 98 partial, 34 source_only, 48 requested_missing, 7 deferred_unverified. The required-versus-deferred assignment has not been made, so no family closes on the roster alone | [ROSTER.md](docs/ROSTER.md) |
| Provider compatibility | 555 upstream field/route rows classified with a code reference; explicit 400/422 refusals replace silent drops | [PROTOCOL_DEVIATIONS.md](docs/PROTOCOL_DEVIATIONS.md) |
| Route contract | 152 HTTP operations over 130 paths (46 of them broker routes), 72 MCP tools, 69 CLI subcommands and 120 Python client methods bound to their typed contracts at commit `fd5638a`; declared-only and rejected-in-job semantics are marked, not hidden | [ROUTE_CONTRACT_MATRIX.md](docs/ROUTE_CONTRACT_MATRIX.md) |

These counts combine separate layers; none is a percentage-complete figure.

## What runs today

The source declares 19 task types and 12 backend names (`src/titanos/config.py`). A declared adapter is not proof for every checkpoint or device; the table says what was actually executed, with the evidence rows that record it.

| Family | Bounded passes (evidence) | Still open |
|---|---|---|
| Chat, completions, Responses | Transformers chat with SSE and usage from source and from frozen Windows x64 CUDA packages (E-064, E-066); native llama.cpp GGUF chat, tool round trip, JSON schema, Responses, two slots, cancellation and crash recovery on Windows CUDA (E-069); packaged GGUF chat on Linux CPU (E-036); Linux CPU and CUDA (E-022, E-093); macOS x86_64 GGUF (E-097); single and two-replica kit receipts on Windows, Linux and Snapdragon slots (E-090 to E-103) | stop/role/control breadth (TEXT-01); stateful Responses (TEXT-03); factual quality: a 135M model gave a coherent wrong answer, retained |
| Vision (image input) | SmolVLM single-question passes on CUDA (E-023) and vision fixtures in the kit campaigns (E-064, E-090, E-093) | broader input and control coverage (VIS-01) |
| Embeddings, rerank, pair classification, sentiment | MiniLM and ONNX graphs on CPU, CUDA, DirectML, OpenVINO GPU and NPU, QNN, Adreno, MPS and macOS x86_64 (E-002, E-022, E-026 to E-032, E-090 to E-105); typed rerank, pair and sentiment routes (E-067) | scores uncalibrated; wider semantic gates |
| Transcription, VAD, translation | Whisper English ASR on CUDA (E-024) and native Windows CPU (E-047); Silero VAD (E-067); T5 translation (E-065); composed audio translation (E-005); fixture ASR in the campaigns (E-090, E-093, E-103) | languages, streaming, caller cancellation (AUD-01, AUD-05) |
| Anonymous diarization and audio analysis | one known-two-speaker fixture through the frozen Windows package: VAD, ASR, Sherpa diarization and transcript analysis under one deadline, overlap F1 .8405 (E-052 to E-054, E-071) | automatic speaker count, accuracy, hard preemption, visible UI |
| Speech | Qwen3-TTS reference-voice takes byte-identical across runs (E-060, E-080); Sherpa acquisition, cancel, resume and offline reuse (E-034); declared controls and refusals (E-062); MCP speech job workflow with idempotency (E-072) | streaming, formats, human listening acceptance (AUD-04) |
| Image and image edit | SD1.5 concurrently on Arc B580 and A770 (E-021); Windows CUDA image and edit (E-008); instruct-pix2pix on Arc B580 through PyTorch XPU held at partial (E-099); the white-background prompt miss is retained as MODEL_QUALITY | per-backend controls; queued and active cancellation (GEN-01, GEN-02) |
| Video | Wan2.2 image-to-video on Windows CUDA (E-042); Wan2.1 T2V and CogVideoX-2b in kit runs (E-101, E-099); tiny fixtures on CPU and CUDA (E-064) | text-only conditioning breadth; cancellation (GEN-03) |
| OCR, detection, segmentation, depth | Tesseract and TrOCR lines, DETR and YOLOS detection, SAM masks, relative depth (E-006, E-040, E-065, E-067, E-101) | page geometry, varied scenes, metric depth (VIS-02 to VIS-05) |
| Music (ACE-Step) | one 10 s stereo WAV from the isolated runtime under an enforced memory cap (E-104); two music-inclusive Windows x64 portable candidates built, hash-verified and relocated, which refused music admission before any load (E-106); candidate H (public source 2cb870d, repaired adapter) completed one 10 s instrumental through its packaged binary on one NVIDIA GPU with HTTP and MCP readback and owned cleanup (E-107) | song quality; other durations, controls, devices and hosts; cancellation during generation |
| Raw tensor graphs | MobileNetV2 and UCI digit graphs on DirectML, OpenVINO, VitisAI and QNN (E-002, E-091, E-097, E-102) | the Ryzen AI NPU load refusals (E-097, E-098) stay recorded as fails |

## Hardware coverage

One row per platform slot of the execution matrix. "Pass" counts verifier-upheld model-by-slot cells against every executed cell, not devices; a slot with zero executed cells is open, not failed. Hosts are named by role.

| Slot | Pass / executed cells | Devices that ran |
|---|---|---|
| Windows x64 CPU | 42 / 46 | this workstation, fleet desktops and laptops, native CPU packages |
| Windows x64 NVIDIA CUDA | 32 / 32 | RTX PRO 6000 Blackwell, RTX 5070, RTX 2070 Max-Q (Turing, float16 policy) |
| Windows x64 NVIDIA DirectML | 4 / 5 | ONNX graphs; the SD1.4 ONNX pipeline refused to load |
| Windows x64 AMD iGPU and dGPU DirectML | 4 / 5 each | Radeon 840M and a Ryzen desktop iGPU; RX 7900 XTX; SD1.4 ONNX refused |
| Windows x64 AMD Ryzen AI NPU (VitisAI) | 2 / 4 | MobileNetV2 and UCI digit graphs pass; MiniLM and tiny-BERT loads refused |
| Windows x64 Intel Arc dGPU (OpenVINO, PyTorch XPU, llama.cpp SYCL) | 29 / 33 | Arc B580 and A770; llama.cpp SYCL chat fails at its first matmul |
| Windows x64 Intel iGPU | 29 / 30 | Iris Xe, Arc 130V, UHD 630 |
| Windows x64 Intel NPU (OpenVINO) | 5 / 5 | AI Boost NPU, CPU fallback disabled |
| Windows ARM64 CPU (Snapdragon X) | 31 / 31 | extended set, including audio and video with an x64 ffmpeg under emulation |
| Windows ARM64 Adreno DirectML and Hexagon QNN | 4 / 4 each | prepared ONNX graphs |
| Linux x64 CPU and NVIDIA CUDA | 8 / 9 and 12 / 12 | one Linux server with an RTX 4090; native Linux CUDA packages (E-040) |
| Linux x64 AMD ROCm; Linux x64 Intel GPU | 0 / 0 | runtime candidates exist; no host has run the kit |
| Linux ARM64 CPU; Linux ARM64 NVIDIA (Jetson, GB10) | 0 / 0 | a 0.5.0 GB10 chat is recorded without an evidence id; out of the current scope |
| macOS ARM64 CPU, MPS, Neural Engine | 12 / 13, 8 / 8, 2 / 4 | M4 Max from source and the native ARM64 package on CPU and MPS (E-038); the M1 MacBook Air kit run (E-108): MPS cells are storage-level device evidence, the Neural Engine cells prove Core ML provider activation with CPU fallback refused, not Neural Engine kernels, and two refuse by design |
| macOS x86_64 | 10 / 10 | ONNX, sherpa, tesseract and GGUF on an Intel MacBook Pro |

Older CUDA generations (Maxwell to Ampere, Hopper) are covered by build flags only; two Pascal cards are present but have not executed. Drivers, precision, operator coverage, memory and model architecture decide support: see [HARDWARE.md](docs/HARDWARE.md) and [ACCELERATOR_EXECUTION.md](docs/ACCELERATOR_EXECUTION.md). Anyone can extend the matrix on their own machine with the [matrix kit](docs/MATRIX_KIT.md); a cell becomes pass only after a verifier upholds its receipt.

## Interfaces and controls

HTTP with OpenAPI (152 operations over 130 paths at commit `fd5638a`), MCP over stdio and Streamable HTTP with 72 typed tools, 6 resources, 6 templates and 2 prompts (E-078 captured the 60 tools of the frozen ENGINE package of its day; the 12 broker tools came later and are source-only), a CLI, a Python client and the four-language Studio browser interface. The optional cloud broker added on 2026-10-01 to 2026-10-03 ([BROKER.md](docs/BROKER.md)) can answer the OpenAI, Anthropic, Gemini, ElevenLabs and Ollama routes from operator-configured providers; it is off without `broker.json`, its cells are proven against loopback mocks ([BROKER_MATRIX.md](docs/BROKER_MATRIX.md)) plus the owner-run live smoke of 2026-10-02, and none of that is model-on-hardware evidence in the capability matrix. Controls resolve request over model defaults over process defaults (E-001, E-002, E-029, E-044, E-056). Since commit `0e8ee71` the extended-inference backends apply the process sampling defaults as well: translation and image-to-text requests that name no temperature sample at the shipped default of 0.7 unless a model default or `TITANOS_DEFAULT_TEMPERATURE=0` pins them. The option-parity map (120 rows, 17 checks; E-085) and the provider deviation map (E-086) are test-bound. Cross-interface parity for acquisition, runtime, music and job-cancellation semantics remains open (INT-04, INT-07), as does complete browser action coverage (INT-05).

## Packaging and portability

Frozen Windows x64 packages ran with ambient Python removed (E-046, E-047, E-066, E-067, E-088); a relocated macOS ARM64 package ran on CPU and MPS (E-038); a corrected Linux x64 package completed the one-stop CPU journey (E-036). The current speech successor is executable `b9a0a63b5b59` (E-088), carrying the Studio, acquisition-unit and protocol-refusal fixes, with native Studio proofs for load, generate, unload and eight-view persistence (E-083, E-084). Two music-inclusive Windows x64 candidates were built from public source with the isolated ACE-Step runtime packaged and every weight external, fully re-hashed and relocated without Python on the path (E-106); candidate H then completed the first real music request from packaged bytes after the adapter's Windows progress-publication defect was repaired (E-107); no music request from their bytes has been admitted yet. Every build lists its components and notices; the redistribution review is not complete (PKG-03). Public source copies exclude the maintainers' operational records and are tested for lab locators ([public export hygiene](docs/DISTRIBUTION.md#public-export-hygiene)).

## Failures we keep

Nine matrix cells fail and stay red: llama.cpp SYCL chat on Arc B580 (E-101), MiniLM and tiny-BERT loads on the Ryzen AI NPU (E-097, E-098), SD1.4 ONNX loads on four DirectML adapters (E-092, E-099), and MiniLM and tiny-BERT on the M1 Neural Engine slot, where the Core ML provider partitions only part of the graph and CPU fallback is refused by design (E-108). Earlier reporting, startup and cancellation-cleanup failures (E-009 to E-012, E-016 to E-018) remain recorded. Forced cleanup is recorded as forced; explicit process termination is never called graceful draining.

## Retained model-quality observations

SmolLM2-135M-Instruct produced coherent but factually wrong Moon answers; its `finish_reason=length` at the requested 32-token cap is correct engine control (E-022). SD1.5 produced coherent red-mug images with gray-blue backgrounds, missing the requested white background (E-021). The digit classifier achieved 28/30 label accuracy while matching 30/30 retained reference predictions (E-026, E-027). The Wan image-to-video result is playable and coherent, with its recorded visual and motion misses retained (E-042). The reference-voice takes passed automated transcript and audio checks; human listening is unclaimed (E-041, E-060, E-080). The diarization fixture used a supplied speaker count of two; its overlap result is not benchmark DER or automatic-count accuracy (E-052).

## Next work

- Human acceptance of the engine demonstration and of the reference voice (MEDIA-03, DEL-02); an interim 25-scene overview exists (E-082), not an accepted film.
- Clean-user first use on every intended platform (APP-02).
- A real music request from a portable package (PKG-01); model sharding and tensor parallel as qualified modes (LIFE-05).
- Linux ROCm, Linux Intel GPU, Linux ARM64 and Apple Neural Engine execution (HW-04, HW-06, HW-08, HW-14 and the matrix's empty slots).
- The required-versus-deferred assignment of the roster (EVID-01); vendor-SDK compatibility beyond the deviation map (INT-06).
- Not implemented at all: training, universal checkpoint conversion, speaker identity recognition, duplex realtime voice, automatic failover across machines, MIG partition lifecycle, video understanding and tracking (VIS-06), 3D/mesh/texture asset families (GEN-04).

## How this page stays honest

`docs/progress.json`, `docs/execution_matrix.json`, `docs/roster.json` and `docs/protocol_deviations.json` are the sources of truth. Their Markdown views are generated and compared byte for byte in CI (`tests/check_progress.py` with `tools/render_progress.py --check`, `tests/check_execution_matrix.py`, `tests/check_roster.py`, `tests/check_protocol_deviations.py`, `tests/check_route_contract_matrix.py`). A cell may not be pass unless a cited receipt records that model on that slot. Stable ids never change meaning. Raw receipts stay outside the public tree, and machines are named by role, never by machine name or address. This page is refreshed in the same commit as any ledger change.

Next: [README](README.md) to run it, [CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) for each evidence row, [PROGRESS.md](docs/PROGRESS.md) for the open work, [RELEASE_NOTES.md](docs/RELEASE_NOTES.md) for what changed, and [docs/history](docs/history/README.md) for dated ledger changes and older validation reports.
