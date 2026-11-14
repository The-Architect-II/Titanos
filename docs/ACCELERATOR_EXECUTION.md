# Accelerator execution and worker groups

Titanos resolves placement within the process's inherited device visibility.
Selecting a GPU never grants access to devices excluded by a parent, container
or operator mask. A missing device, unavailable backend or incompatible group
fails explicitly; Titanos does not retry an explicit GPU request on CPU.

| Execution path | Single device | Homogeneous group | Independent replicas |
| --- | --- | --- | --- |
| llama.cpp | CPU or a device exposed by the selected native runtime | NVIDIA UUID group: `layer`, `row`, or experimental `tensor` splitting in one native process | Separate native workers, each with its own model and slots |
| Transformers | A supported PyTorch CPU/GPU device | `model_sharding` uses a balanced device map; `tensor_parallel` requires a supported architecture and installed distributed collective | Separate process workers; no automatic replay after a failure |
| Diffusers | A supported PyTorch device | Component sharding where the pipeline supports a device map; no distributed tensor parallel adapter | Separate process workers |
| ONNX/OpenVINO and CPU-only adapters | Their declared runtime device/provider | No model-sharding or tensor-parallel adapter | Independent supported single-device workers |

Task-specific restrictions still apply. Translation, OCR, detection,
classification, depth and segmentation adapters currently require single-device
workers; use independent replicas. ACE music remains a separate isolated
runtime. Sherpa speech/diarization and Silero VAD remain CPU adapters.

## Native defaults and explicit placement

New native registrations retain `device: "auto"` until load. The selected native
runtime is queried first; an admitted visible accelerator is preferred. If visible
NVIDIA hardware exists but the selected native cohort exposes no usable CUDA
device, load requires a compatible GPU runtime or an explicit CPU choice. A
CPU-only executable does not establish that the host is CPU-only. Automatic
placement can select CPU on a CPU-only host or within an explicitly empty GPU
scope. No process-wide visibility mask is removed.
Explicit CPU placement and `gpu_layers: 0` disable GPU offload without a GPU probe.

`native.gpu_layers` and `TITANOS_NATIVE_GPU_LAYERS` accept:

- `"auto"` (default): pass the pinned native runtime's automatic layer setting.
- `"all"`: request all layers on the selected accelerator(s).
- A nonnegative integer: request that layer limit; zero explicitly disables offload.

The adapter keeps `--fit off`, preserving the requested context and layer policy.
With llama.cpp b10809, an unset/automatic layer count without fitting attempts
the model's full offloadable layer set. Insufficient capacity fails; it does not
authorize an unreported CPU fallback. `--native-options` JSON supports these
values through both `add` and `pull`. Existing registrations with an explicit
zero retain that choice. A deliberate Studio/API GPU assignment may change a
previous CPU registration to automatic offload; a refresh or ordinary load does
not make that change.

An explicit NVIDIA device uses its full physical `GPU-…` UUID. CUDA ordinals are
local addresses: the worker verifies the selected UUID order after narrowing its
mask. Expert `native:DEVICE` names refer to the chosen executable's own
`--list-devices` output. Non-CUDA native groups are refused until their stable
identity and homogeneity can be established; a native address is not a physical
UUID. Non-CUDA address selection cannot override inherited accelerator masks.

## Grouped workers

A native worker with stable NVIDIA `gpus` uses these mappings:

| Worker mode | Native behavior |
| --- | --- |
| `single` | One selected GPU, `split_mode: "none"` |
| `model_sharding` | `split_mode: "layer"`; an explicitly configured `row` split is retained |
| `tensor_parallel` | Experimental llama.cpp `split_mode: "tensor"` |

These native modes run one llama.cpp process. The model's generic `parallelism`
remains `single`. Native tensor splitting is distinct from Transformers/PyTorch
distributed tensor parallelism.

For PyTorch workers, `model_sharding` and GPU tensor parallelism require known,
matching runtime vendor and architecture. Unknown architecture does not establish
homogeneity. Sharding uses the library's balanced model-size budget, not just the
capacity of GPU 0, and verifies that the device map and actual parameters span
every selected GPU. CPU/disk placement is rejected for this sharding path.
Explicit bounded-memory video component dispatch retains its separate policy.

Transformers tensor parallelism additionally requires a native model TP plan and
compatible tensor dimensions. CUDA/ROCm use NCCL/RCCL, XPU uses XCCL, and CPU
ranks use Gloo. Standard Windows PyTorch does not provide NCCL, so CUDA distributed
TP is unavailable there even when ordinary CUDA inference and sharding work.
Native llama.cpp tensor splitting has a separate implementation. A small model
with nine attention heads and three KV heads, for example, cannot use two
column-sharded attention ranks; it can still use layer sharding or replicas.

## Compiled runtime selection

API recommendations and the compiled loaders use one selection policy. An
explicit matching installed-runtime device pin is preserved; an unavailable pin
fails rather than being replaced. Otherwise OpenVINO prefers an observed GPU,
then NPU, then CPU. Tensor and embedding graphs compile on that concrete device;
unsupported graph shapes or operations fail without a CPU retry. GenAI chat uses
the same preference, subject to its model/runtime requirements.

GenAI SD/SDXL images currently support GPU or CPU only. Automatic selection
excludes NPU, while an explicit NPU request fails before pipeline construction.
The separate static NPU image preparation path remains unimplemented and
unqualified; see [the image adapter contract](GENAI_IMAGES.md). A visible device
is a candidate, not proof that every model can execute on it.

ONNX automatic selection prefers an observed nonsoftware DirectML adapter, then
an installed supported QNN/Core ML/Vitis AI provider, then CPU. It carries the
actual DirectML adapter index and LUID into session admission, including nonzero
indices; an identity change fails. Provider enumeration does not establish a
physical device or kernel placement. Existing explicit CPU-EP fallback controls
and Core ML's internal-placement limitation remain visible in model details.

All selection uses the runtime's current enumeration without rewriting inherited
visibility. A CUDA mask is not reinterpreted as an Intel OpenVINO mask. Concrete
runtime addresses, stable identities where exposed, and selection reasons are
recorded separately from execution observations. OpenVINO graph details check
`EXECUTION_DEVICES` against the selected address. The current automatic-selection
regressions use vendor doubles and synthetic tensors; real model/hardware
qualification remains specific to a delivered package and execution receipt.

## Observation and limits

Worker details distinguish configured device, resolved device, observed UUID
mapping and parameter placement. Native details retain the native load-log
offload count and model-buffer sizes. GPU launches explicitly use native log
verbosity 4: llama.cpp b10809 maps library INFO records to that TRACE threshold,
so the server's default verbosity 3 hides the required offload record. A healthy
server with a missing or zero GPU-layer count is still rejected. CPU launches
retain the native logging default. These are placement evidence, not a trace
of kernels executed by an inference request. Per-request profiling, when
supported, is reported separately. Capability availability may remain unknown
until explicit runtime/device discovery; `qualified: null` is not a pass. If profiler exit or event collection raises `MemoryError`, the completed result is still returned and the failure is reported, as described in [profiling failure and completed results](profiling-failure-policy.md).

`TITANOS_DEVICE_PROBE_TIMEOUT` and `TITANOS_DEVICE_INVENTORY_TTL` control bounded
device discovery and refresh caching (both default to 30 seconds). Native child
processes use the selected executable's library cohort; the frozen parent's DLL
search directory is reset inside the isolated supervisor, without changing the
parent or its visibility masks. Native ownership and stdin-EOF cleanup remain
in effect.

Model weights remain external. A CUDA-capable application bundle does not by
itself include a compatible llama.cpp CUDA runtime, a model, sufficient free
VRAM or target-hardware qualification. Check the selected runtime's compiler/
driver requirements and the exact model/device execution receipts.

Pinned native option semantics: [llama.cpp b10809 server options](https://github.com/ggml-org/llama.cpp/blob/b10809/tools/server/README.md).
The required log threshold follows the [b10809 library logging callback](https://github.com/ggml-org/llama.cpp/blob/b10809/common/log.cpp#L413-L429).
