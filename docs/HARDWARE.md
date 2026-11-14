# Hardware, runtime builds, and distribution

Updated **2026-09-26**. For the install steps on your platform start with [GETTING_STARTED.md](GETTING_STARTED.md); this page is the reference behind it. The tables distinguish **implemented adapters**, **available catalog profiles**, and **version-bound native observations**. The full hardware intent includes older NVIDIA generations, AMD, Intel, Qualcomm, Apple and CPU-only systems across the listed OS/architectures. No CPU or source result closes an accelerator-family gate. See [current evidence](CAPABILITY_MATRIX.md) and all 18 hardware checklist items in [PROGRESS.md](PROGRESS.md#hardware).

Titanos provides native portable application bundles and a Python wheel. Bundles
include their selected runtime dependencies; model weights stay external.
PyTorch, OpenVINO, ONNX Runtime, drivers and model architecture determine which
OS/CPU/GPU/NPU combinations can execute a model. Use the current package-specific
qualification reports alongside this runtime catalog.

**The differentiator is a consistent multimodal contract with explicit, tested
runtime selection.** It is not a promise that every checkpoint works on every
accelerator. Modality, model architecture, precision, operators, memory, driver,
OS, Python ABI, and processor architecture all affect support.

## Runtime profiles

The `engine` profile installs Titanos without PyTorch or a compiled model runtime.
Pair it with `native-install` for GGUF serving. The pinned native catalog adds
Windows/Linux/macOS x64 and ARM64 builds where published, including Windows ARM
CUDA/Adreno and Windows ROCm candidates. These are a separate backend and do not
change the PyTorch matrix below. Their device execution remains unqualified until
run on the target. See [NATIVE.md](NATIVE.md) for the managed llama.cpp catalog and [GETTING_STARTED.md](GETTING_STARTED.md) for the per-platform steps.

## PyTorch/compiled profile matrix

[0.5.2 validation](VALIDATION_0.5.2.md) separates trained native inference from
synthetic runtime checks. These representative tests do not qualify every model
or every device supported by an upstream wheel.

The last column names the [PROGRESS.md](PROGRESS.md#hardware) checklist row and
its recorded state (`validated_bounded` or `partial`) so both documents use the
same words; the evidence IDs point into [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md).

| Target | Profile / device | Status | Checklist row |
|---|---|---|---|
| Linux x86_64, Intel or AMD CPU | `cpu` / `cpu` | Bounded native passes for the named chat and MiniLM models from current source (E-022, E-036, E-040); other packages and models need their own acceptance | HW-02 validated_bounded |
| Linux ARM64 CPU | `cpu` / `cpu` | Upstream wheel available; native acceptance pending | HW-04 partial |
| Windows x86_64 CPU | `cpu` / `cpu` | Native trained-model and compiled application checks retained | HW-01 validated_bounded |
| Windows ARM64 CPU | `onnx-cpu` / `cpu`; PyTorch from source outside the catalog | Recorded from source on the Snapdragon X laptop: 31 of 31 executed rows pass, Transformers and Diffusers on torch 2.9.1+cpu win_arm64 wheels, prepared ONNX, sherpa, tesseract and GGUF (E-102, E-103); the `cpu` profile lists no windows-aarch64 build | HW-03 partial |
| NVIDIA Maxwell/Pascal, Linux/Windows x86_64 | Pinned `cu126` / selected physical GPU | Catalog architecture fit only; target model/operator/driver and native package proof remain open | HW-05 partial |
| NVIDIA Volta, Linux/Windows x86_64 | Pinned `cu126`; `cu128` also declares compatibility in this 2.8 catalog | Source selection is not V100/Volta native qualification; do not apply a different PyTorch release's architecture cutoffs | HW-05 partial |
| NVIDIA Turing, Linux/Windows x86_64 | Compatible pinned `cu126` or `cu128` | Bounded Windows x64 matrix-kit passes on one RTX 2070 Max-Q (compute 7.5, float16 policy; E-096); not broad Turing qualification | HW-05 partial |
| NVIDIA Ampere/Ada/Hopper, Linux/Windows x86_64 | Compatible pinned `cu126` or `cu128` | Recorded Ada cases (RTX 4090 on Linux, E-022 to E-024; a released 0.5.1 CUDA bundle chat on an RTX 3500 Ada laptop, [VALIDATION_0.5.2.md](VALIDATION_0.5.2.md)) retain their model/package scope; they do not qualify every Ampere/Hopper device | HW-05 partial |
| NVIDIA Blackwell, Linux/Windows x86_64 | `cu128` / `cuda:N` | Native RTX PRO 6000 CUDA source and application-bundle inference checked (E-030, E-064, E-066, E-067, E-069); RTX 5070 matrix-kit receipts (E-095, E-097) | HW-05 partial |
| NVIDIA ARM64 / Jetson | `cu130` / `cuda:0` on Linux SBSA (cc 9.0 and above, e.g. DGX Spark GB10); Jetson Orin: vendor runtime | Catalog candidate, no hardware run recorded; do not substitute an x86 wheel | HW-06 partial |
| Supported AMD GPUs, Linux x86_64 | `rocm64` / `cuda:N` | Candidate; ROCm GPU/OS matrix and native acceptance required | HW-08 partial |
| AMD/Intel/NVIDIA GPU, Windows x64 | `dml` / `dml` | Trained DirectML checks on Radeon 840M/Raphael, Iris Xe and RTX 3500 Ada; prepared ONNX only | HW-07 partial |
| Adreno GPU, Windows ARM64 | `dml-arm64` / `dml` | Native ARM64 standalone classifier/embeddings and HTTP/MCP checked with bundled runtime libraries | HW-11 validated_bounded |
| Supported Intel GPUs, Linux/Windows x86_64 | `xpu` / `xpu:N` | Windows x64 source execution recorded on Arc B580 and Arc 130V with torch 2.8.0+xpu (E-099, E-101); Linux XPU candidate; native package acceptance required | no dedicated row; Windows recorded, Linux candidate |
| Apple silicon macOS | `mps` / `mps` | Native MPS modality checks retained; not Apple Neural Engine qualification | HW-13 validated_bounded |
| Intel macOS 13+ | `onnx-intel-mac` / `cpu` | Recorded from source on an Intel MacBook Pro with onnxruntime 1.23.2: prepared ONNX embeddings, VAD and tensor graphs, sherpa, tesseract and GGUF rows (E-094, E-097, E-098, E-101); no frozen Intel Mac package | HW-15 partial |
| Linux/Windows x86_64 Intel CPU | `openvino-cpu`, `genai-cpu` / `CPU` | Graph, embedding and native GenAI chat exercised on Linux | no dedicated row |
| Supported Intel GPU / NPU | `openvino-gpu/npu` / explicit `GPU[.N]` or `NPU[.N]` | Native trained graph/embedding checks on Iris Xe, Arc B580/A770 and Intel AI Boost; GenAI LLM qualification remains separate | HW-09, HW-10 validated_bounded |
| Snapdragon Windows ARM64 NPU | `qnn` / `qnn` | Native ARM64 QNN HTP standalone classifier/embeddings and HTTP/MCP checked; native crypto/runtime bundled | HW-11 validated_bounded |
| Apple silicon Neural Engine | `coreml` / `coreml` | M4 standalone FP16 MobileNet has 102 ANE compute-plan assignments; no universal ANE-only guarantee | HW-14 partial |
| AMD Ryzen AI 5 340 NPU | Vendor runtime / `vitisai` | Trained MobileNet passed in a private standalone candidate with an external precompiled BF16 cache; model compilation is separate and redistribution remains held for required vendor notices | HW-12 partial |
| 32-bit x86 or ARM | None | No profile in the present 64-bit catalog; any future legacy target needs its own runtime/build qualification | none |

PyTorch profiles pin PyTorch 2.8.0 and torchvision 0.23.0 (`cu130` on linux-aarch64: 2.10.0 and 0.25.0). This is a deliberate release
baseline, not a claim that 2.8 is the newest PyTorch. All profiles still require
compatible system drivers and runtime prerequisites; Linux wheels also impose
their own glibc minimum. Device execution is checked after install.

### Older NVIDIA cards, Volta and mixed CUDA machines

For the **pinned PyTorch 2.8.0 catalog**, `cu126` declares compute capabilities
`5.0 <= cc < 10.0`; `cu128` declares `7.0 <= cc < 13.0`. Thus Maxwell/Pascal use
`cu126`; Volta/Turing/Ampere/Ada/Hopper fit the older profile and can also fit
`cu128` within that declared range. Blackwell 10.x/12.x requires `cu128` here; 12.1 is covered because sm_120 binaries run on sm_121 (CUDA binary compatibility within a major version), not because a 12.1 card ran. Linux ARM64 (SBSA) uses `cu130` (torch 2.10.0, `9.0 <= cc < 13.0`: Grace Hopper, Thor, DGX Spark GB10), a catalog candidate with no hardware run recorded.
These are catalog compatibility bounds, not execution proof or a rule for all
future PyTorch releases. The 2.8 notice includes restored Volta support in cu128.
Automatic selection uses the chosen catalog's available platform profiles and
finite capability bounds covering **every selected card**, preferring the lowest
compatible numeric CUDA profile. If the catalog/platform lacks cu126, a compatible
cu128 entry can serve Volta or newer supported cards. Unknown/non-finite device
capabilities and invalid catalog bounds fail with a diagnostic. Explicit
`--profile` remains exact and never silently falls back to a different runtime.
Nine deterministic selector tests cover these decisions (E-020); they are not
installation, driver or native GPU evidence.
[PyTorch's 2.8 architecture notice](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128),
[versioned installation commands](https://pytorch.org/get-started/previous-versions/#v280),
[NVIDIA Blackwell CUDA support](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/).

On a machine containing both Pascal and Blackwell, no single profile in this
catalog covers both; select a physical card with `--gpu INDEX` and use separate
environments as needed. Volta plus Blackwell can share the catalog's `cu128`
range, subject to actual driver/model validation. New CUDA receipts capture the selected physical GPU UUID
before installation and resolve that UUID against PyTorch at runtime. A card
that is no longer visible causes an error instead of substitution. Legacy CUDA
receipts without a UUID require an explicit UUID selection or runtime reinstallation;
`auto` refuses their old index. See [device identity and migration](DEVICES.md).
Windows x64 AMD/Intel display hardware without NVIDIA defaults to the DirectML
profile; an explicit profile remains available for other runtimes. ARM64 DirectML
uses its separate published wheel profile. See [DirectML setup](DIRECTML.md).
Titanos defaults Pascal to float32; newer CUDA GPUs use float16. Precision can
be explicitly configured. No mandatory FlashAttention, quantization kernel,
Triton compilation, or CUDA extension excludes old cards from the basic loader.

### Automatic profile selection

`--profile auto` (the installer default) decides from `discover()` facts only:
OS, architecture, `nvidia-smi` rows, and display vendors read from
`/sys/class/drm` on Linux or `Win32_VideoController` on Windows (AMD, Intel and
NVIDIA vendor IDs; a Qualcomm adapter is not recorded). When it cannot decide it
stops with one of the messages below. Nothing falls back silently.

| Detected | Result of `--profile auto` |
|---|---|
| NVIDIA card(s) reported by `nvidia-smi` | The lowest-numbered `cuNNN` profile published for the platform whose bounds cover every selected card (`cu126`: 5.0 <= cc < 10.0; `cu128`: 7.0 <= cc < 13.0; `cu130`, linux-aarch64 only: 9.0 <= cc < 13.0). No covering profile (Kepler 3.x, a Maxwell and Blackwell pair, Jetson Orin 8.7 on `linux-aarch64`): `No CUDA profile in this catalog covers GPU 0 CC 3.7 on windows-x86_64; select --gpu INDEX for separate environments or provide a compatible versioned catalog. CPU requires explicit --profile cpu; no fallback was selected` |
| `nvidia-smi` present but failing | `NVIDIA detection failed; inspect drivers or explicitly choose --profile cpu` |
| macOS on Apple silicon | `mps` |
| An NVIDIA display adapter but no `nvidia-smi` row (any OS) | `An NVIDIA display adapter is present but nvidia-smi reported no GPU; put the driver's nvidia-smi on PATH and rerun, or select --profile explicitly; see docs/GETTING_STARTED.md` |
| Windows x64 with an AMD or Intel display adapter | `dml` (DirectML; prepared ONNX graphs only) |
| Linux, or Windows ARM64, with an AMD or Intel display device | `Select an explicit runtime profile for linux-x86_64: cpu, engine, genai-cpu, genai-gpu, genai-npu, onnx-cpu, openvino-cpu, openvino-gpu, openvino-npu, rocm64, xpu; see docs/GETTING_STARTED.md` (the list is every non-CUDA profile the catalog publishes for that platform) |
| No display vendor recorded | `cpu` |
| Windows ARM64 or Intel macOS without an NVIDIA row or a recorded display vendor | `cpu` is chosen and then refused: `Profile cpu has no published build for windows-aarch64; see docs/HARDWARE.md` (or `darwin-x86_64`). Choose `onnx-cpu`, `dml-arm64`, `qnn`, `onnx-intel-mac` or `engine` |

Explicit `--profile` is exact. A profile not published for the platform is
refused (`Profile rocm64 has no published build for windows-x86_64; see docs/HARDWARE.md`),
a CUDA profile whose bounds do not cover the selected cards is refused
(`Selected CUDA profile does not cover the selected GPUs; choose --gpu INDEX or a different profile`),
and a CUDA profile without any NVIDIA row is refused
(`No NVIDIA GPU detected; repair the driver or select --profile cpu`). `--gpu INDEX`
is an `nvidia-smi` index, applies to CUDA profiles only, and is refused with
DirectML (`--gpu selects NVIDIA CUDA indices; use DirectML device_id in model provider options`).

### Modality coverage across runtime families

This is a routing guide, not a Cartesian-product support claim. A catalog profile
does not supply an arbitrary model's tokenizer, processor, operators or export.
Each native pass remains bound to its evidence ID, model and package.

| Modality group | CPU x64 / native LLM | NVIDIA x64 CUDA | AMD/Intel x64 GPU | ARM64, Apple and NPU paths |
|---|---|---|---|---|
| Chat/vision language models, completions, Responses | Implemented; selected CPU language cases and local router canary E-001/E-003/E-018 | Implemented compatible adapters; historical selected GPU cases only | Intel GenAI/native runtime options; AMD ROCm/DML depend on supported model/export | MPS/native/GenAI candidates are model-specific; no universal NPU LLM route |
| Embeddings, sentiment, pair classification, reranking and retrieval composition | Implemented; bounded cases E-002–E-004 with semantic misses retained | Compatible adapter paths; qualify each model and generation | Selected prepared encoder/classifier DML/OpenVINO proof E-002; not the whole family | QNN/CoreML/vendor prepared encoder/classifier/graph subsets only; composition still needs its own proof |
| Speech, ASR, VAD and text/audio translation | Implemented; historical CPU ASR/VAD/translation E-005; current serving availability separately held E-019 | Recorded speech/job and ASR context cases E-005/E-009; wider quality/lifecycle open | Backend/model-specific candidates; a display GPU or runtime install proves nothing about audio | Prepared VAD/encoder or compatible runtime only; no all-audio NPU/ARM64 claim |
| OCR, detection, segmentation and depth | Implemented; bounded CPU cases E-006, relative depth not metric calibration | Compatible model operators must be qualified on the selected GPU | Prepared or supported framework models; no full-family native pass | Prepared model/operator subsets; full-page OCR, fresh-scene masks and accuracy remain open |
| Image generation/editing | Implemented; CPU/fixture execution is not broad visual quality | Selected trained image/edit cases E-008, failures retained | Intel B580 GenAI batch E-007; prior AMD image failure retained | MPS or compatible prepared GenAI/ONNX export; target-specific native image proof required |
| Video generation | Implemented compatible Diffusers/jobs path; tiny historical fixtures E-001 | Real model execution E-008 failed its all-frame quality gate | Profile availability is not useful video execution/quality proof | No general native video qualification; this is not video understanding/tracking |
| Typed tensors/prepared graphs | Explicit graph/operator path E-002 | Selected runtime/provider must support graph/operators | DML/OpenVINO prepared-model evidence is narrow | QNN, CoreML/ANE and vendor NPU evidence is model-specific; raw tensors do not create a finished modality |
| Diarization/overlap, video understanding, music/3D and training | Remaining or tentative work | Remaining or tentative work | Remaining or tentative work | No capability inferred from another row |

Windows ARM64 CPU, Linux ARM64 CPU, NVIDIA ARM64/Jetson, Apple Silicon and Intel
macOS remain separate OS/ABI/device targets. An x86_64 CUDA wheel is never an
ARM64 runtime. MPS execution is not ANE execution, and CPU fallback never counts
as GPU/NPU proof. Matching/mixed-machine routing and tensor-parallel ranks are
separate acceptance gates, not a consequence of installing the same profile.

### Other accelerators and NPUs

AMD ROCm uses PyTorch's `cuda` device namespace. Its GPU and OS support must be
checked against AMD's matrix; a Linux wheel is not a Windows build.
[AMD installation guidance](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html),
[AMD Windows matrices](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html).

Intel XPU targets supported Intel GPUs. Apple's MPS backend targets Metal GPUs.
Neither profile implies access to the vendor's NPU.
[PyTorch Intel GPU guidance](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html),
[Apple MPS guidance](https://developer.apple.com/metal/pytorch/).

OpenVINO and ONNX Runtime are working prepared-model execution paths in this
release. They expose an explicit device/provider, input shapes/dtypes, and runtime
placement. No adapter claims to execute arbitrary PyTorch code on an NPU.
PyTorch XPU is not Intel NPU, MPS is not Apple Neural Engine, and a Snapdragon ARM
CPU runtime does not imply QNN HTP execution.

| Runtime | Release baseline | Pipeline and constraints |
|---|---|---|
| ONNX Runtime | 1.27.0; 1.23.2 for Intel Mac | Prepared `.onnx` graph and any external-data files; local tokenizer for embeddings |
| Qualcomm plugin | `onnxruntime-qnn==2.6.0` + ORT 1.27.0 | Registers the native EP library, selects QNN devices and the HTP backend; ARM64 Windows drivers required |
| OpenVINO | 2026.3.1 | Prepared XML/BIN IR or compatible ONNX; explicit CPU/GPU/NPU |
| OpenVINO GenAI | 2026.3.1.0 | Complete prepared LLM directory, including compiled tokenizer/detokenizer |
| AMD Vitis AI | Vendor Ryzen AI environment | Install Titanos `.[compiled]` into the vendor environment; its ORT must provide `VitisAIExecutionProvider` |
| Core ML | ORT 1.27.0 on Apple silicon | MLProgram with CPU+NeuralEngine compute units, static inputs by default; macOS 14+ wheel |

NPU models often need fixed shapes, compatible operators, calibration data, and
quantization. Conversion does not guarantee that a model meets those requirements.
QNN HTP does not simply accept every floating-point model. Use vendor preparation
and inspect native results; Titanos will report compile/provider failures.
The AMD adapter accepts a JSON `--provider-options` file for the vendor's required
configuration. It does not download or silently substitute a vendor SDK.

ONNX Runtime CPU EP fallback is disabled for an explicitly selected accelerator by default,
including Python's runtime retry fallback. `--allow-cpu-fallback` is an explicit
operator opt-in and appears in details. Core ML can still perform CPU work *inside*
its provider; disabling ORT's CPU EP does not prove ANE-only execution. OpenVINO
uses an explicit device, not AUTO/HETERO. Tokenization and orchestration run on CPU.
Use device profiling for actual execution evidence.

Primary references used for these decisions:
[QNN provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html),
[current QNN plugin API](https://github.com/onnxruntime/onnxruntime-qnn),
[Core ML provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html),
[Vitis AI provider](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html),
[OpenVINO NPU GenAI](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html),
[OpenVINO export tools](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-optimum-intel.html).
The QNN wheel was inspected for `get_library_path`, `get_qnn_htp_path`, and
`get_ep_name`; the implementation uses plugin registration, not the obsolete
assumption that installing an arbitrary ORT wheel adds QNN.

Intel Macs use an explicit legacy **ONNX** profile, not a fabricated current
PyTorch build. Microsoft publishes macOS 13+ x86_64 wheels for Python 3.11–3.13
in [ORT 1.23.2](https://pypi.org/project/onnxruntime/1.23.2/#files). The matrix kit
executed prepared ONNX and sherpa-onnx models on that runtime from source on an
Intel MacBook Pro (macOS 15.7, Python 3.12; E-094, E-098).
Its adapter supports typed graphs and embeddings, not the full PyTorch modality
stack. Graph IR/opset compatibility follows that runtime's version. The newer
[ORT 1.27 files](https://pypi.org/project/onnxruntime/1.27.0/#files) supply the other
listed CPU platforms. Published wheels are candidate-platform evidence, not
successful installation or model-execution evidence.

### Preparing and pulling models

Use the normal Hugging Face pipeline for prepared models:

```sh
titanos search --library openvino --limit 20
titanos inspect OWNER/PREPARED-MODEL
titanos pull chat OWNER/PREPARED-MODEL --task chat --backend openvino-genai --device NPU
titanos pull embed OWNER/ONNX-MODEL --task embeddings --backend onnx --graph onnx/model.onnx --device qnn
```

`OWNER/...` is a placeholder for a model whose vendor/runtime compatibility you
have checked. Authentication is optional and uses the same saved login, `HF_TOKEN`,
`--token-env`, or explicit `--anonymous` policy as other pulls. A prepared pull
keeps the complete pinned snapshot so external weights, tokenizer graphs, and
vendor sidecars remain available. Serving never fetches a missing file silently.
`--graph` must point inside the registered model directory.

For ordinary local encoder models, Titanos includes a narrow, validated export:

```sh
# In an export environment with the PyTorch and target runtime extras:
python -m pip install '.[models,onnx,openvino,export]'
titanos prepare embed embed-onnx --backend onnx --output /new/onnx-directory
titanos prepare embed embed-ov --backend openvino --output /new/openvino-directory
```

It exports a Transformers embedding encoder at fixed batch 1 and configurable
sequence length (`--sequence-length 128`), preserves its tokenizer, then compares
three real inputs against the original model. Registration happens only after
numerical validation on `--device` (default CPU). Output includes source provenance,
file hashes and comparison results. Existing output directories are never replaced.
The embedding API handles larger request batches as sequential graph calls.
This is not a universal LLM, diffusion, or speech exporter.

For OpenVINO LLMs, pull a prepared directory or use Intel's Optimum exporter in a
separate preparation environment. The serving adapter uses native OpenVINO GenAI;
it does not mix a new exporter dependency stack into the pinned Transformers
serving environment. CPU GenAI defaults to f32 compute for repeatable greedy
results; model weights may still be FP16 or quantized as prepared. GPU/NPU options
must match the target's supported precision and model limits.

```sh
# Release installer (needs a built or published release catalog; see "Installer behavior"):
python install.py --catalog release/release.json --profile openvino-npu
python install.py --catalog release/release.json --profile genai-npu
python install.py --catalog release/release.json --profile qnn
python install.py --catalog release/release.json --profile coreml
python install.py --catalog release/release.json --profile onnx-intel-mac
# pip equivalents: .[openvino], .[genai], onnxruntime==1.27.0 onnxruntime-qnn==2.6.0 + .[compiled],
# .[onnx] on Apple silicon, onnxruntime==1.23.2 + .[compiled] on Intel macOS (GETTING_STARTED.md)

titanos doctor --backend openvino --device NPU --check
titanos doctor --backend onnx --device qnn --check
```

Each profile is offered only for its listed platform. Drivers are installed using
the vendor's supported process. OpenVINO GenAI's installer check exercises a small
graph on the selected device; actual LLM compilation/generation remains a separate
qualification step. This source-only reconciliation exercises no NPU. Historical
qualified models above retain their separate package/device evidence.

## Installer behavior

The installer is `src/titanos/hardware.py`. `python install.py` at the repository
root runs that file, and `tools/build_release.py` copies it into `release/` as a
standalone `install.py` next to the catalog. It needs a release catalog
(`release.json`), which this tree does not contain; build one from a checkout:

```sh
python -m pip install 'setuptools>=78' wheel     # build_release.py builds the wheel without build isolation
python tools/build_release.py                      # writes release/release.json, install.py, native-profiles.json, SHA256SUMS
python install.py --catalog release/release.json --profile auto --plan
```

The catalog version must be a plain `X.Y.Z` release. The current package version
`0.5.28.dev0+accelerators1` is refused with `Titanos install: Invalid release version`,
so until a tagged release is cut the pip route in [GETTING_STARTED.md](GETTING_STARTED.md)
is the working path; the installer examples on this page apply to a published
release. The installer then:

1. Read a versioned catalog from disk or HTTPS; loopback HTTP is allowed for local development.
2. Inspect platform/architecture and NVIDIA compute capabilities, then choose a profile.
3. Print the selection. `--plan` stops here without changing anything.
4. Download and verify the Titanos wheel's SHA-256.
5. Create a new environment; install the profile's upstream binary wheels.
6. Constrain those runtime versions while installing the matching Titanos extra.
7. Execute a small neural network on the selected device, failing if it cannot run.
8. Record the selected device, hardware check, package versions, and release checksums.

The installer does not install drivers, compile PyTorch automatically, fall back
from a requested GPU to CPU, or replace an existing environment. Vendor graph/NPU
and non-default accelerator profiles require explicit compatible selection;
automatic Windows DirectML selection still does not prove that a particular
model or operator is supported. A neural-network check validates the
runtime; it does not certify every model or operator.

```sh
python install.py --catalog release/release.json --profile cu126 --gpu 0
python install.py --catalog release/release.json --profile cu128 --gpu 1
python install.py --catalog release/release.json --profile xpu
python install.py --catalog release/release.json --profile rocm64
python install.py --catalog release/release.json --profile mps
```

Each run creates `~/.titanos/runtimes/<version>-<profile>` (or `--target`) and
writes `titanos-runtime.json`, `hardware-check.json` and `requirements.lock.txt`
there. A CUDA receipt records the selected card's `GPU-...` UUID, and `--device auto`
in that environment resolves to it.

## Local release hosting

```sh
python tools/build_release.py
python -m http.server 8765 --bind 127.0.0.1 --directory release
# In another terminal:
python release/install.py --catalog http://127.0.0.1:8765/release.json --profile cpu
```

Move the same files to an HTTPS static host later. Relative wheel URLs continue
to work. SHA-256 detects corruption; public distribution also depends on trusting
the host or verifying the catalog hash through a separate trusted channel.
`--catalog-sha256` supports that check. No automatic updater runs inside inference.

For offline distribution, prepare a complete wheelhouse on the matching target
platform with pip's native download tools. The current bootstrap downloads
dependencies from upstream indexes; locally hosting Titanos alone is not a fully
air-gapped installer.

## Adding machines without adding a build farm framework

Run `titanos doctor --backend BACKEND --device DEVICE --check --output hardware.json` on each
machine, then run model acceptance on that same device and record:

- OS, CPU architecture, device name and compute capability, driver, runtime and dependency versions.
- Model repository revision, weight format, precision, input sizes and output validity.
- First-load time, time to first token/output, steady-state throughput, and peak memory.
- Streaming cancellation, model swapping, memory recovery, and meaningful quality checks.

The included model acceptance script defaults to CPU. Pass `--device cuda:0`,
`xpu:0`, or `mps` for native device qualification. It is not a performance benchmark.
Promote a particular model/device combination only after those checks pass.
Reuse upstream wheels wherever they exist. Build native packages on matching OS
and architecture only when a needed upstream artifact is absent. Such packages
require their own provenance, hashes, and model acceptance; compiling successfully
is not sufficient.

For independent replicas, module sharding, tensor parallelism, and per-rank GPU
execution evidence, use the separate [multi-GPU qualification guide](MULTIGPU.md).
