# Titanos 0.5.0 validation

> **Historical record (Titanos 0.5.0).** This page is kept for provenance and is not updated; current results are in the [capability matrix](CAPABILITY_MATRIX.md) and the [execution matrix](EXECUTION_MATRIX.md), and results recorded here do not qualify the current source or any later package. This report separates application contracts, real model execution, native hardware and standalone binary evidence. Archived 0.4 results (raw receipts not published) are historical input, not proof for this build.

## Native inference observed during development

| Platform | Runtime/device | Real work observed | Boundary |
|---|---|---|---|
| Native Windows x64 | PyTorch 2.8/CUDA 12.8, RTX PRO 6000 Blackwell | All 7 original modalities, image edit, real streaming/disconnect, lifecycle; added translation/OCR/detection/audio translation on CPU and CUDA | Small pinned checkpoints; tests do not prove every model |
| Linux x64 | PyTorch 2.8/CUDA 12.8, 2×RTX 4090 | All 7 original modalities; concurrent replicas with GPU kernel evidence; useful 512 px SDXL robot image | No ROCm/NPU claim |
| macOS arm64 | PyTorch 2.8/MPS, Apple M4 Max | All 7 original modalities, streaming, lifecycle; CogVideoX dynamic position regression | Positional metadata uses CPU float64→MPS float32; model parameters/inference stay on MPS |
| Linux aarch64 | PyTorch 2.12/CUDA 13.0, NVIDIA GB10 (Grace Blackwell) | Cached TinyLlama chat, decoded PNG and 5-frame MP4 through actual Titanos HTTP | Task environment reused numerical packages read-only; not a fresh standalone installation |
| Native Windows x64 | ONNX Runtime 1.27 CPU; OpenVINO 2026.3.1 CPU/Intel iGPU/Intel AI Boost NPU | Export parity, typed graph operation, independent ONNX/OpenVINO workers; actual 384-dimensional embeddings on Intel iGPU and NPU with CPU fallback disabled | iGPU/NPU cosine similarity above 0.999997 against CPU for two inputs; not all NPU architectures/models |

## Application and protocol checks

The base, CLI/client, settings, lifecycle, asset queue, registry, protocol and MCP checks exercise normal and failed behavior. MCP initializes over modern and legacy HTTP plus stdio, lists typed tools/resources/prompts, calls inference and lifecycle tools, handles authentication failures and returns generated artifacts. Synthetic model checks prove contracts only.

Real Chrome on native Windows exercised the bundled browser app's GPU chat and speech-job generation, audio preview and persisted job list. It does not prove a signed desktop installer or every accessibility scenario.

Settings checks establish explicit request > per-model defaults > environment defaults, raised limits, validation, and propagation through HTTP/MCP/extension/job paths. Model replacement is quiescent: active work returns 409 instead of changing weights beneath an in-flight request.

## Standalone executable qualification

The Windows CUDA candidate started with Python-related environment variables removed and PATH restricted to Windows system directories, then performed all 7 modalities, MCP HTTP/stdio and real 2-GPU worker inference. This establishes that the executable used its included interpreter/runtime rather than an installed Python entry point. Candidate source changed during development, so it is not the final release identity.

Final bundle BUILD.json, MANIFEST.json and executable/media qualification receipts identify the actual delivered package. The delivery report that names the qualified bundles is not part of this repository. Native application builds are specific to OS, architecture and numerical runtime. GPU drivers and normal OS libraries remain host prerequisites.

## Model quality and limitations

The original image/video fixtures are deliberately tiny and prove execution/serialization, not useful visual quality. A separate real SDXL checkpoint generated the visible robot example. TrOCR recovered HELLOWORLD from HELLO WORLD, losing the space. T5 translated a simple English sentence to French; Whisper base translated local French speech to English. Earlier weak Whisper tiny outputs were retained as failed/limited quality evidence rather than relabeled successful.

No tests establish universal checkpoint compatibility, training, VAD/diarization, music/3D generation, realtime speech sessions, every provider feature, large-scale performance, hardware not listed above, organizational acceptance, code signing, public publication or a comprehensive security/legal audit.

## Running the checks

Run `python tests/check.py` for source contracts and add `--models /path/to/seven-checkpoints --device cpu|cuda:0|mps` for real modalities. Add `--executable /path/to/titanos` to direct that same real HTTP suite at a compiled application. Use tests/check_api_plus.py for the additional pinned checkpoint checks, check_mcp.py for protocol qualification, check_compiled.py for prepared runtime models, and check_parallel.py for worker/partition modes. `tools/fetch_checkpoints.py` explicitly downloads the original public fixture revisions; weights are not part of any application bundle.
