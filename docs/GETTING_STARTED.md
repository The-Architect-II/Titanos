# Getting started

Titanos serves local models over HTTP, MCP, a CLI and a browser workspace (Titanos Studio). This page gets one model answering on your machine. For each platform it gives the install commands, what `titanos doctor` and `titanos devices` should report, the qualification status in one line, and what to do when automatic selection refuses. [HARDWARE.md](HARDWARE.md) is the reference behind it.

Two facts first:

- Python 3.11, 3.12 or 3.13 is required (`pyproject.toml`; the release installer refuses other versions with `This release requires Python 3.11–3.13`).
- Weights are never bundled. `titanos pull` downloads a pinned repository once; inference never downloads anything. Everything lives under `TITANOS_HOME` (default `~/.titanos`): the registry `models.json`, pulled models and installed runtimes.

## Two install paths

| Path | What it does | Works today? |
|---|---|---|
| pip into a virtual environment | You install the PyTorch, ONNX Runtime or OpenVINO wheel for your hardware, then `python -m pip install -e ".[EXTRA]"` from a checkout of this repository | Yes. Every section below uses it. |
| Release installer: `python install.py --catalog release.json --profile NAME` | Reads a versioned catalog, detects OS, architecture and NVIDIA compute capability, prints the plan (`--plan` stops there), creates `~/.titanos/runtimes/<version>-<profile>`, installs the profile's pinned wheels and the Titanos wheel, runs `titanos doctor --check` on the selected device and writes `titanos-runtime.json`, `hardware-check.json` and `requirements.lock.txt` | Not from this tree. The catalog must carry a plain `X.Y.Z` version; the current package version `0.5.28.dev0+accelerators1` is refused with `Titanos install: Invalid release version`. No release catalog is published yet. |

The installer never installs drivers, never overwrites an existing target (`Target already exists. Choose a new --target; existing environments are never overwritten`) and never falls back from a requested GPU to CPU (`Runtime inference check failed; ... No CPU fallback was substituted`). Once a tagged release exists, [HARDWARE.md](HARDWARE.md#installer-behavior) describes building and hosting the catalog. The profile names below (`cpu`, `cu126`, `cu128`, `dml`, ...) are the installer's `--profile` values from `tools/profiles.json`; the pip commands install the same pinned wheels.

Install the runtime wheel for your hardware before the Titanos extra. The `[models]` extra accepts any `torch>=2.8,<2.13`; if torch is not already present, pip pulls the newest PyPI build instead of the pinned 2.8.0 wheel for your accelerator.

## The same five steps on every platform

```sh
titanos init                                    # empty registry at TITANOS_HOME/models.json
titanos doctor --check                          # PyTorch profiles; other backends in the table below
titanos devices --backend all --refresh         # installed hardware and what each installed runtime can see
titanos pull chat HuggingFaceTB/SmolLM2-135M-Instruct --task chat   # setup check, not a quality baseline
titanos serve --open                            # HTTP on http://127.0.0.1:11435, Studio in the browser, MCP at /mcp
```

Then, in a second terminal with the same environment: `titanos run chat "Explain local inference in two sentences." --stream`. Running `titanos` with no arguments is the same as `titanos serve --open`. `titanos <command> --help` lists the options.

Step 4 pulls a PyTorch checkpoint. On an install without PyTorch (Windows ARM64 `.[onnx]`, `.[dml-arm64]` or the QNN pair; Intel macOS; Core ML; `engine`) the pull still downloads and registers, but the model fails at load with `No module named 'torch'`. Register a model those runtimes can run instead: GGUF chat through `native-install` and the pinned pull in [NATIVE.md](NATIVE.md) (`titanos pull agent Qwen/Qwen2.5-0.5B-Instruct-GGUF --revision 9217f5db79a29953eb74d5343926648285ec7e67 --task chat --backend llama-cpp --model-file qwen2.5-0.5b-instruct-q4_k_m.gguf --device cpu --anonymous`, recorded on Windows ARM64 and Intel macOS), or a prepared ONNX graph: pull a repository that ships one (`titanos pull embed OWNER/ONNX-MODEL --task embeddings --backend onnx --graph onnx/model.onnx --device cpu`; `OWNER/...` is a placeholder for a repository whose graph you have checked), or export one on a PyTorch machine from a registered Transformers embedding model (`titanos prepare my-embedder my-embedder-onnx --backend onnx --output DIR`, [HARDWARE.md](HARDWARE.md#preparing-and-pulling-models)), copy the directory over and register it with `titanos add embed DIR --task embeddings --backend onnx --device cpu` (or `--device dml`, `qnn`, `coreml`). The recorded Snapdragon and Intel Mac ONNX cells ran such an export.

### Which doctor command

`titanos doctor --check` uses PyTorch. On an engine-only, ONNX or OpenVINO install it reports `"check": {"passed": false, "error": "No module named 'torch'"}` and exits 1. Use the backend you installed:

| Install | Command | Expect in the JSON |
|---|---|---|
| PyTorch (`cpu`, `cu126`, `cu128`, `cu130`, `rocm64`, `xpu`, `mps`) | `titanos doctor --check` | `"device"`, `"cuda_runtime"`, `"rocm_runtime"`, `"mps_available"`, `"xpu_available"`, `"check": {"passed": true, "kind": "two-layer neural network"}`; on CUDA also `"selected_gpu_uuid": "GPU-..."` |
| Engine only (`engine`; GGUF through llama.cpp) | `titanos doctor --backend engine` | `"engine_version"`, `"hardware"`, `"check": {"passed": true, "kind": "engine import"}` |
| ONNX Runtime (`onnx-cpu`, `dml`, `dml-arm64`, `qnn`, `coreml`, `onnx-intel-mac`) | `titanos doctor --backend onnx --device cpu --check` (or `dml`, `qnn`, `coreml`) | `"providers"`, `"device"`, `"check": {"passed": true, ..., "observed_node_providers": [...]}`; `dml` adds `"adapters"`, `qnn` adds `"qnn_devices"` |
| OpenVINO (`openvino-cpu`, `openvino-gpu`, `openvino-npu`, `genai-*`) | `titanos doctor --backend openvino --device CPU --check` (or `GPU`, `NPU`) | `"devices": ["CPU", ...]`, `"device"`, `"check": {"passed": true}` |

`doctor --check` proves that a PyTorch, ONNX Runtime or OpenVINO runtime executes a two-layer network or graph on the selected device. `doctor --backend engine` only confirms that the Titanos package imports and reports the hardware it sees; it does not start the installed `llama-server` (the `native-install` output's `version_output` is that check, and the first GGUF request is the first execution). Neither certifies any model.

### What `--device auto` means

Registrations default to `--device auto`. At load time PyTorch takes the first visible CUDA device (ROCm uses the same `cuda:` namespace), then Intel XPU, then Apple MPS, then CPU. OpenVINO prefers GPU, then NPU, then CPU. ONNX prefers a hardware DirectML adapter, then an installed QNN, Core ML or Vitis AI provider, then CPU. A runtime installed by the release installer pins `auto` to the profile's device. An explicit device is exact: `--device cuda:0` is stored as that card's `GPU-...` UUID, and a missing, masked or unavailable device fails (`Requested cuda:0 is unavailable in this runtime visibility; CPU was not substituted`) instead of moving to CPU. Details: [DEVICES.md](DEVICES.md).

## Windows x64: CPU

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[models]"
.venv\Scripts\titanos.exe init
.venv\Scripts\titanos.exe doctor --check
```

`doctor` reports `"device": "cpu"`, `"cuda_runtime": null` and a passed check. `titanos devices --backend all --refresh` lists the display adapters Windows knows about even though the CPU runtime cannot use them. Installer profile: `cpu` (what `auto` picks when no display adapter is detected). Status: bounded passes for named models and packages on native Windows x64 CPU (HW-01 validated_bounded).

## Windows x64: NVIDIA (CUDA)

Find the compute capability first: `nvidia-smi --query-gpu=name,compute_cap --format=csv`. `nvidia-smi` must be on `PATH`; the installer, `titanos devices --backend nvidia` and worker UUID pinning all use it (`NVIDIA identity discovery requires nvidia-smi; no index fallback is used`).

| Compute capability | Cards | Profile | pip index |
|---|---|---|---|
| 5.0 to 6.x (Maxwell, Pascal) | GTX 900 and 1000 series, P100 | `cu126` only | `https://download.pytorch.org/whl/cu126` |
| 7.0 to 9.0 (Volta, Turing, Ampere, Ada, Hopper) | V100, RTX 20/30/40 series, A100, H100 | `cu126` (what `auto` picks) or `cu128` | either |
| 10.0 and 12.x (Blackwell) | B200, RTX PRO 6000, RTX 50 series | `cu128` only | `https://download.pytorch.org/whl/cu128` |
| below 5.0 (Kepler and older) | most GTX 600 and 700 cards | none in this catalog | use `cpu` |

`auto` takes the lowest-numbered CUDA profile whose bounds cover every card (`cu126`: 5.0 <= cc < 10.0; `cu128`: 7.0 <= cc < 13.0; on Linux ARM64 `cu130`: 9.0 <= cc < 13.0). A Pascal card next to a Blackwell card has no common profile: pick one card with `--gpu INDEX` and use separate environments.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -e ".[models]"
.venv\Scripts\titanos.exe doctor --check
.venv\Scripts\titanos.exe devices --backend nvidia --refresh
```

`doctor` reports `"device": "cuda:0"`, `"cuda_runtime": "12.8"` (or `"12.6"`), `"cuda_architectures"` and `"selected_gpu_uuid": "GPU-..."`. `devices --backend nvidia` lists every card with its UUID, PCI bus id, memory, compute capability and driver. Maxwell and Pascal load in float32 by default; compute capability 7.0 and above uses float16 (`--dtype` on `add` and `pull` overrides). Multi-GPU: `titanos worker large --models chat --gpus GPU-<uuid-a> GPU-<uuid-b> --mode model_sharding`; tensor parallelism needs NCCL, which the Windows CUDA build of PyTorch lacks, so use Linux for that ([MULTIGPU.md](MULTIGPU.md)). GGUF models on the GPU: `titanos native-install --profile win-cuda-12.4-x64` (or `win-cuda-13.3-x64` when the driver supports CUDA 13), then register with `--backend llama-cpp --device GPU-<uuid>` and `gpu_layers` above 0 ([NATIVE.md](NATIVE.md)). Status: bounded passes on Blackwell (RTX PRO 6000 and RTX 5070: E-008, E-009, E-030, E-064, E-066, E-067, E-069, E-095, E-097 in [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md)) and on Turing (RTX 2070 Max-Q, compute 7.5, float16 policy: the seven kit fixture models as single and two-replica workers, E-096); an earlier released 0.5.1 CUDA bundle chat on an RTX 3500 Ada laptop is in [VALIDATION_0.5.2.md](VALIDATION_0.5.2.md); Maxwell, Pascal, Volta, Ampere and Hopper are catalog fit only (HW-05 partial).

## Windows x64: Intel CPU, GPU or NPU (OpenVINO)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[openvino]"
.venv\Scripts\titanos.exe doctor --backend openvino --device GPU --check
.venv\Scripts\titanos.exe devices --backend openvino --refresh
```

`.[openvino]` installs `openvino==2026.3.1` with no PyTorch. `doctor` lists `"devices"` as OpenVINO names them (`CPU`, `GPU`, `NPU`); pass the one you want with `--device`. `devices --backend openvino` shows each device with its architecture and an integrated or discrete subtype. Installer profiles: `openvino-cpu`, `openvino-gpu`, `openvino-npu` for prepared IR or ONNX graphs (embeddings, classifiers, typed tensors), and `genai-cpu`, `genai-gpu`, `genai-npu` (`.[genai]`, `openvino-genai==2026.3.1.0`) for prepared OpenVINO GenAI LLM and image directories. Name the profile: on an Intel laptop `auto` picks `dml`. Status: bounded passes on Intel Iris Xe, Arc B580 and A770 GPUs (HW-09 validated_bounded, including the named SD1.5 GenAI image case E-007) and on the Intel AI Boost NPU with CPU fallback disabled (HW-10 validated_bounded); GenAI LLMs remain candidates.

## Windows x64: Intel GPU through PyTorch XPU

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/xpu
.venv\Scripts\python.exe -m pip install -e ".[models]"
.venv\Scripts\titanos.exe doctor --check --device xpu:0
```

`doctor` reports `"xpu_available": true` and `"device": "xpu:0"`. This is the PyTorch route for Transformers and Diffusers models on an Intel Arc GPU, discrete or integrated; OpenVINO above runs prepared graphs and the NPU, and PyTorch XPU is not the Intel NPU. Installer profile: `xpu`, named explicitly (`auto` picks `dml` on a Windows x64 machine with an Intel adapter). Status: bounded passes from source on the Arc B580 (torch 2.8.0+xpu, E-099; one instruct-pix2pix cell held at partial) and the Arc 130V (E-101); Linux XPU has no recorded pass; no dedicated checklist row.

## Windows x64: AMD, Intel or NVIDIA GPU through DirectML

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dml]"
.venv\Scripts\titanos.exe devices --backend onnx --refresh
.venv\Scripts\titanos.exe doctor --backend onnx --device dml --check
```

`.[dml]` installs `onnxruntime-directml==1.24.4`. `devices --backend onnx` lists each adapter as `dml:N` with its LUID, memory and a software-renderer flag; `doctor` shows `"providers": ["DmlExecutionProvider", ...]`, `"adapters"` and, after the check, `"observed_node_providers"`. Choose an adapter with a JSON file `{"device_id":"1"}` passed as `--provider-options` ([DIRECTML.md](DIRECTML.md)). Installer profile: `dml`, the Windows x64 default when an AMD or Intel adapter is present. DirectML runs prepared ONNX graphs only (typed tensors, embeddings, [ONNX image pipelines](ONNX_IMAGES.md)); it does not run PyTorch checkpoints. Status: trained prepared-graph checks on Radeon 840M and Raphael, Iris Xe and RTX 3500 Ada (HW-07 partial). AMD Ryzen AI NPU: install `.[compiled]` into the vendor's Ryzen AI environment and use device `vitisai`; redistribution of that runtime is held (HW-12 partial).

## Windows ARM64 (Snapdragon)

`auto` stops here: `Profile cpu has no published build for windows-aarch64; see docs/HARDWARE.md`. Choose one route:

```powershell
python -m venv .venv
# CPU, PyTorch models from source (no catalogued installer profile; the win_arm64 pair the recorded runs used on Python 3.12)
.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[models]"
.venv\Scripts\titanos.exe doctor --check
# CPU, prepared ONNX graphs (installer profile onnx-cpu)
.venv\Scripts\python.exe -m pip install -e ".[onnx]"
.venv\Scripts\titanos.exe doctor --backend onnx --device cpu --check
# Adreno GPU through DirectML (installer profile dml-arm64)
.venv\Scripts\python.exe -m pip install -e ".[dml-arm64]"
.venv\Scripts\titanos.exe doctor --backend onnx --device dml --check
# Hexagon NPU through QNN HTP (installer profile qnn; no pip extra, so two steps)
.venv\Scripts\python.exe -m pip install onnxruntime==1.27.0 onnxruntime-qnn==2.6.0
.venv\Scripts\python.exe -m pip install -e ".[compiled]"
.venv\Scripts\titanos.exe doctor --backend onnx --device qnn --check
# GGUF chat with no numerical Python runtime (installer profile engine)
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\titanos.exe native-install --profile win-cpu-arm64
```

Use one ONNX Runtime distribution per environment (`.[onnx]`, `.[dml-arm64]` or the QNN pair); they own the same Python module. `.[dml-arm64]` installs `onnxruntime-windowsml==1.25.2.202605110140`. The QNN check reports `"qnn_devices": 1` when the HTP driver is present. Other native profiles for this platform: `win-opencl-adreno-arm64`, `win-cuda-13.4-arm64`. The PyTorch route is the only one here that loads the quick-start SmolLM2 pull; the ONNX routes need a prepared graph (see the note under the five steps) or GGUF through `native-install`. Status: 31 of 31 executed rows pass from source on the Snapdragon X laptop's CPU (E-102, E-103): Transformers and Diffusers models on torch 2.9.1+cpu (chat, vision, embeddings, rerank, sentiment, ASR, translation, speech, tiny image and video fixtures, OCR, detection, segmentation and depth), prepared ONNX graphs, sherpa, tesseract and GGUF through `win-cpu-arm64`; the audio and video rows needed an ffmpeg binary that the win_arm64 `imageio-ffmpeg` wheel does not ship (the recorded rerun used the x64 build under emulation). The catalog `cpu` profile lists no windows-aarch64 build, so the release installer cannot install this PyTorch route (HW-03 partial). Prepared graphs pass on Adreno DirectML and QNN HTP (HW-11 validated_bounded).

## Linux x64: CPU

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[models]"
titanos init
titanos doctor --check
```

`doctor` reports `"device": "cpu"` and a passed check. Installer profile: `cpu`; `auto` picks it only when `/sys/class/drm` shows no AMD, Intel or NVIDIA device, so on a desktop pass `--profile cpu`. Status: bounded passes for the named chat and MiniLM models from current source (HW-02 validated_bounded; E-022, E-036, E-040).

## Linux x64: NVIDIA (CUDA)

Same profile table as Windows; `nvidia-smi` must be on `PATH`.

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128   # or whl/cu126
python -m pip install -e ".[models]"
titanos doctor --check
titanos devices --backend nvidia --refresh
```

`doctor` reports `"device": "cuda:0"`, `"cuda_runtime"` and `"selected_gpu_uuid"`. Tensor parallelism with NCCL is available here: `titanos worker large --models chat --gpus GPU-<uuid-a> GPU-<uuid-b> --mode tensor_parallel`, and `python tests/check_parallel.py --model /path/to/chat --gpus 0 1 --report gpu-modes.json` records replicas, sharding and TP on your cards ([MULTIGPU.md](MULTIGPU.md)). GGUF: llama.cpp publishes no Linux CUDA archive, so the managed catalog offers `ubuntu-vulkan-x64` (select the card with `--device native:Vulkan0`, taken from the binary's `--list-devices` output) or you pass your own CUDA build with `--runtime /path/to/llama-server`, which still takes full `GPU-...` UUIDs ([NATIVE.md](NATIVE.md)). Status: bounded passes on Ada (RTX 4090: chat, MiniLM embeddings and single-case vision and ASR; E-022 to E-024); other architectures open (HW-05 partial).

## Linux x64: AMD GPU (ROCm 6.4), candidate

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/rocm6.4
python -m pip install -e ".[models]"
titanos doctor --check
```

`doctor` reports `"rocm_runtime": "6.4..."` and `"device": "cuda:0"`; ROCm uses PyTorch's `cuda` namespace. Devices carry `HIP-...` identities; `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` masks are honoured and never widened. Installer profile: `rocm64`, explicitly (`auto` refuses on any Linux box with a display GPU, and the profile exists for `linux-x86_64` only). GGUF: `titanos native-install --profile ubuntu-rocm-10.0-x64`. Check your GPU and distribution against AMD's ROCm matrix first. Status: candidate, no recorded native pass (HW-08 partial).

## Linux x64: Intel GPU (PyTorch XPU) or OpenVINO

```sh
python3 -m venv .venv && . .venv/bin/activate
# PyTorch XPU, candidate on Linux (recorded on Windows x64, see above)
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install -e ".[models]"
titanos doctor --check --device xpu:0
# OpenVINO, no PyTorch
python -m pip install -e ".[openvino]"
titanos doctor --backend openvino --device CPU --check
```

XPU `doctor` reports `"xpu_available": true` and `"device": "xpu:0"`; OpenVINO `doctor` lists `"devices"`. Installer profiles: `xpu`; `openvino-cpu`, `openvino-gpu`, `openvino-npu`; `genai-cpu`, `genai-gpu`, `genai-npu`. PyTorch XPU is not the Intel NPU; use OpenVINO for the NPU. GGUF: `ubuntu-sycl-fp16-x64`, `ubuntu-sycl-fp32-x64`, `ubuntu-openvino-2026.3.1-x64`. Status: Linux XPU is a candidate with no recorded pass (the recorded PyTorch XPU passes are Windows x64 records on the Arc B580 and Arc 130V, E-099 and E-101); OpenVINO CPU has been exercised on Linux x86_64; the recorded Intel GPU and NPU passes are Windows records (HW-09, HW-10).

## Linux ARM64 and NVIDIA Jetson

`--profile cpu` installs the aarch64 CPU wheel (`auto` picks it on a headless box). `onnx-cpu` (`.[onnx]`) and `engine` with `ubuntu-arm64` or `ubuntu-vulkan-arm64` also apply. NVIDIA SBSA hosts with compute capability 9.0 or above (Grace Hopper, Thor, DGX Spark GB10 at 12.1) get `cu130` (torch 2.10.0+cu130), a catalog candidate with no hardware run recorded. Jetson Orin has no bundled CUDA profile: `auto` stops with `No CUDA profile in this catalog covers GPU 0 CC 8.7 on linux-aarch64; ...`. Use the vendor's PyTorch build with the pip route and never an x86_64 wheel. Status: open (HW-04 and HW-06 partial).

## macOS on Apple silicon

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0      # PyPI wheels include Metal (MPS)
python -m pip install -e ".[models]"
titanos doctor --check
```

`doctor` reports `"mps_available": true` and `"device": "mps"`. Installer profile: `mps` (what `auto` picks). `--profile cpu` is not published for macOS (`Profile cpu has no published build for darwin-aarch64`); register CPU-only models with `--device cpu` on the `mps` install. Core ML for prepared ONNX graphs: `.[onnx]` and `titanos doctor --backend onnx --device coreml --check` (installer profile `coreml`, macOS 14 or newer); Core ML may run parts of a graph on the CPU, so this is not Neural Engine proof; it loads prepared ONNX graphs only, so register an exported or pulled graph as described under the five steps rather than the SmolLM2 pull. GGUF: `titanos native-install --profile macos-arm64`. Status: bounded MPS passes for MiniLM and SmolLM2 chat in a relocated native package (HW-13 validated_bounded); Neural Engine open (HW-14 partial).

## Intel macOS

`auto` stops: `Profile cpu has no published build for darwin-x86_64; see docs/HARDWARE.md`. No PyTorch wheel exists for this platform, so Transformers and Diffusers models do not apply and the quick-start SmolLM2 pull cannot load here; register a prepared ONNX graph (see the note under the five steps) or the GGUF pull. Two routes:

```sh
python3 -m venv .venv && . .venv/bin/activate
# Prepared ONNX graphs: embeddings, VAD, tensors (installer profile onnx-intel-mac; macOS 13 or newer)
python -m pip install onnxruntime==1.23.2
python -m pip install -e ".[compiled]"
titanos doctor --backend onnx --device cpu --check
# GGUF chat (installer profile engine)
python -m pip install -e .
titanos native-install --profile macos-x64
```

Status: recorded from source on the Intel Core i9 MacBook Pro (macOS 15.7, Python 3.12): 10 of 10 executed cells pass in the execution matrix (E-094, E-097, E-098, E-101): prepared ONNX embeddings, VAD and tensor graphs on onnxruntime 1.23.2, sherpa speech and diarization, tesseract OCR and GGUF chat through their own runtimes (`macos-x64` for GGUF; its two-replica cell is held at partial); no frozen Intel macOS package exists yet (HW-15 partial).

## Docker (Linux CPU image)

The `Dockerfile` builds a CPU-only image: `python:3.12-slim`, `torch==2.8.0` CPU and `.[models]`, a non-root user, `TITANOS_HOME=/var/lib/titanos` declared as a volume, port 11435 and a `/healthz` health check. Its default command is `serve --host 0.0.0.0`, and Titanos refuses to bind a non-loopback interface without an API key: without `-e TITANOS_API_KEY` the container exits with `titanos: Set TITANOS_API_KEY before binding to a network interface`.

```sh
docker build -t titanos .
docker volume create titanos-home
docker run --rm -v titanos-home:/var/lib/titanos titanos init
docker run --rm -v titanos-home:/var/lib/titanos titanos pull chat HuggingFaceTB/SmolLM2-135M-Instruct --task chat --anonymous
docker run -d --name titanos -e TITANOS_API_KEY=change-me -p 127.0.0.1:11435:11435 -v titanos-home:/var/lib/titanos titanos
curl -H "Authorization: Bearer change-me" http://127.0.0.1:11435/v1/models
```

`serve` reads the registry before it binds, so the `init` run is required once on a fresh volume; without it the container exits with `titanos: No registry at /var/lib/titanos/models.json. Run 'titanos init' first.`. The registry and weights live in the volume; the container itself is disposable. A bind mount must be writable by uid 10001. `-p 127.0.0.1:11435:11435` publishes only on the host's loopback; drop the address prefix to expose it on the network, and keep the key. GPU passthrough images are not provided. Status: not qualified; no recorded run used the image.

## When automatic selection refuses

Every refusal names the fix; none substitutes CPU.

| Message | What to do |
|---|---|
| `NVIDIA detection failed; inspect drivers or explicitly choose --profile cpu` | `nvidia-smi` is present but failed. Repair the driver, or pass `--profile cpu`. |
| `An NVIDIA display adapter is present but nvidia-smi reported no GPU; put the driver's nvidia-smi on PATH and rerun, or select --profile explicitly; see docs/GETTING_STARTED.md` | The adapter is visible to the OS but the driver utility is not on `PATH`. Put the driver's `nvidia-smi` on `PATH` and rerun: the CUDA profiles need its rows even when named explicitly, otherwise they stop with `No NVIDIA GPU detected; repair the driver or select --profile cpu`. Or pass `--profile dml` (Windows x64, prepared ONNX graphs only) or `--profile cpu`. |
| `No CUDA profile in this catalog covers GPU 0 CC 6.1, GPU 1 CC 12 on windows-x86_64; select --gpu INDEX for separate environments or provide a compatible versioned catalog. CPU requires explicit --profile cpu; no fallback was selected` | Mixed generations or an uncovered card. Pick one card with `--gpu 1`, or pass `--profile cpu`. |
| `Select an explicit runtime profile for linux-x86_64: cpu, engine, genai-cpu, ...; see docs/GETTING_STARTED.md` | A Linux box with an AMD or Intel display device. Pick the section above that matches your hardware and pass its profile. |
| `Profile cpu has no published build for windows-aarch64; see docs/HARDWARE.md` (or `darwin-x86_64`) | The catalog `cpu` profile lists no build for those platforms. Windows ARM64: `onnx-cpu`, `dml-arm64`, `qnn` or `engine` from the installer, or pip torch from `download.pytorch.org/whl/cpu` (win_arm64 wheels exist there; see the Windows ARM64 section). Intel macOS: `onnx-intel-mac` or `engine`; no PyTorch wheel exists. |
| `Profile rocm64 has no published build for windows-x86_64; see docs/HARDWARE.md` | The profile exists for another platform. See the profile matrix in [HARDWARE.md](HARDWARE.md). |
| `Selected CUDA profile does not cover the selected GPUs; choose --gpu INDEX or a different profile` | You asked for `cu128` on Pascal or `cu126` on Blackwell. Use the profile table above. |
| `No NVIDIA GPU detected; repair the driver or select --profile cpu` | A `cuNNN` profile was requested but `nvidia-smi` lists no card. |
| `Requested cuda:0 is unavailable in this runtime visibility; CPU was not substituted` (at load time) | The registered device is masked (`CUDA_VISIBLE_DEVICES`) or gone. Fix the mask or re-register. |
| `Legacy CUDA installation has no pinned UUID; select an explicit GPU UUID or reinstall the runtime` | An older installer receipt without a GPU UUID. Run `titanos pin-devices` or reinstall the runtime. |

Next: [HARDWARE.md](HARDWARE.md) for the full profile matrix and evidence boundaries, [DEVICES.md](DEVICES.md) for identities and visibility masks, [NATIVE.md](NATIVE.md) for GGUF runtimes, [MULTIGPU.md](MULTIGPU.md) for replicas, sharding and tensor parallelism, and [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md) for what has actually been run.
