# Replicas, model sharding, and tensor parallelism

These are three explicit choices. Titanos never changes one into another because
loading failed. Multiple independently assigned workers may each contain a
sharded model or a tensor-parallel rank group.

| Choice | What is divided | Main purpose | Titanos execution |
|---|---|---|---|
| Replicas | Independent requests | Concurrent requests; independent model residency | Separate spawned worker processes, each with its own weights and assignment |
| Model sharding | Whole modules/layers, or Diffusers pipeline components | Fit one model across device memories | Accelerate/Transformers or Diffusers `device_map="balanced"` in one worker |
| Tensor parallelism | Supported weight tensors and operations | Divide computation and memory across cooperating devices | Native Transformers `tp_plan="auto"`, one distributed process per assigned GPU |

“Sharding” can generically include tensor partitioning. In Titanos's configuration,
`model_sharding` specifically means module/component placement. It is not pipeline
parallel scheduling and does not imply every GPU works simultaneously on a token.
Tensor parallelism performs communication during inference and needs a compatible
model architecture, tensor dimensions, collective backend, and interconnect.
[Transformers 4.57 native TP](https://huggingface.co/docs/transformers/v4.57.1/perf_infer_gpu_multi),
[Accelerate big-model execution](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference),
[Diffusers device placement](https://huggingface.co/docs/diffusers/v0.35.1/en/using-diffusers/loading#device-placement).

## Configure a standalone server

Register model directories first. Assign each model used by a worker-enabled
server to at least one worker. Each worker holds one resident model, so put models
that must stay resident simultaneously on different workers.

```sh
# Two copies of chat, one per GPU. 0 and 1 are current CUDA-visible ordinals, not
# nvidia-smi indices; full GPU-... UUIDs are preferred and are what gets saved:
titanos worker a --models chat --gpus 0
titanos worker b --models chat --gpus 1

# A different model can have its own worker:
titanos worker ears --models transcription --device cpu --threads 4

# Apply the saved configuration:
titanos serve
```

To use GPUs 0 and 1 together for one model, first remove conflicting assignments
or use a separate registry. Choose exactly one of these group configurations:

```sh
titanos worker large --models chat --gpus 0 1 --mode model_sharding
# Alternative configuration:
titanos worker large --models chat --gpus 0 1 --mode tensor_parallel --replace
```

NVIDIA assignments accept full `GPU-…` UUIDs (preferred) or current CUDA-visible
ordinals. The CLI observes the ordinal-to-UUID order in a fresh CUDA-runtime child
and saves UUIDs, so a later reordering does not retarget the worker. An `nvidia-smi`
index is a driver index and is never accepted as a CUDA ordinal (`CUDA-visible index
is unavailable; no driver-index fallback was used`). Legacy numeric NVIDIA
assignments require `titanos pin-devices`. Workers verify
the ordered UUIDs against PyTorch before becoming ready. Visibility is set before
inference imports. See [device identity](DEVICES.md). Inside a
worker, assigned GPUs have local indices 0, 1, etc. ROCm persists runtime UUIDs when exposed, resolves current indices in a fresh
probe, and verifies them inside the worker. It uses PyTorch's `cuda` namespace. Shared use of a GPU by multiple workers is allowed when
explicitly configured; the operator must budget VRAM. Titanos does not silently
migrate a model when memory is insufficient.

A sharding worker requires at least two visible assigned CUDA/ROCm GPUs. Its loaded
map must use every assigned GPU and may not spill to CPU/disk. The loader gives
placement up to 85% of currently free memory per GPU; activations still need headroom.
Small models or unsupported pipelines may refuse or collapse a balanced map;
Titanos reports a failed load when that does not meet the requested assignment.
Diffusers placement divides pipeline components; it is not a promise of layer-level
partitioning for every diffusion model. XTTS and compiled adapters have no sharding
implementation. They can still run replicas.

Native tensor parallelism is implemented for Transformers chat, vision, and
embedding loaders **when the architecture provides a native TP plan**. A catalog
tag alone is insufficient. The preflight also checks whole attention-head
partitioning for native column-sharded attention. For example, the 135M SmolLM2
fixture has 9 query heads and 3 KV heads, so its native plan cannot use two TP
ranks; model sharding and replicas are separate viable choices. CUDA/ROCm uses the NCCL/RCCL backend; Intel XPU groups
use XCCL when installed. CUDA groups require their assigned count to match the
requested rank count. For direct single-process configuration, set
`CUDA_VISIBLE_DEVICES` and `tensor_parallel_size` consistently. Native Windows CUDA
without NCCL is rejected; use a qualified Linux/WSL environment where applicable.
An XPU group uses `--device xpu:0 --mode tensor_parallel --tensor-parallel-size 2`
and the runtime's visible XPU devices. It does not use the CUDA `--gpus` option.

Two CPU ranks using Gloo can check native TP logic:

```sh
titanos worker test --models chat --device cpu --mode tensor_parallel --tensor-parallel-size 2
```

This is a functional qualification option, not a performance recommendation.
Collectives need local networking even when rendezvous uses a file. Ranks share
sampling seeds and broadcast stopping decisions; only rank zero emits output.
A failed rank shuts down its group and fails the request. No partial result is
represented as success. Load/inference failures and explicit unload clean up ranks.

## Routing and limits

`load NAME` warms all matching replicas; `--worker ID` targets one. CLI `run`,
`load`, `unload`, and `details` support the same selector. JSON API requests accept
`"worker":"a"`; multipart transcription and image edits accept a `worker` field.
Existing OpenAI clients can use their `extra_body` option. PAIR may continue to
send only a model name and let Titanos choose an idle replica.

Each worker runs one request at a time. The server prefers a resident model,
then the least recently used available replica. It admits at most 32 queued worker
requests with a 30-second wait; a worker operation defaults to a 600-second timeout.
Long video models can use the worker's `--timeout`. Streaming disconnects signal
cancellation and drain internal replies before reusing the worker. Streaming
failure is an SSE error event, not a completed assistant response. Crashed workers
restart on the next explicit request; the interrupted request is never replayed.
There is no continuous batching, speculative decoding, or cross-host collective
training framework in this engine.

All workers under one server share its Python/runtime installation. A Pascal
and Blackwell combination needing incompatible PyTorch builds uses separate
Titanos environments/instances. A replica on each can still serve requests through
a client or PAIR. This release does not claim tensor parallelism across incompatible
builds or arbitrary heterogeneous GPUs.

## Capabilities and qualification

```sh
titanos capabilities chat
titanos details chat --worker large
titanos workers
```

Capabilities inspect the model registration defaults: an implemented adapter,
an available native model plan, its configured candidate rank count, and a
qualification result. Worker assignments override a model's execution mode,
device, and rank count; inspect `workers` and `details --worker ID` for that
assignment's actual configuration and loaded runtime. `qualified: null` means this exact combination has not
been qualified by that inspection. It does not mean failure or success. Inspection
never executes checkpoint-supplied Python; custom architectures can report unknown.
A runtime device map or allocated VRAM proves placement, not that inference used
all assigned devices.

Add `--profile` when creating a worker to record actual PyTorch GPU kernel indices.
For TP, evidence is reported per rank. Profiling adds overhead and is normally off.
Load the model before a qualification request so weight-loading kernels cannot be
mistaken for inference. `tests/check_parallel.py` does this and writes model config
hash, package versions, hardware, assignments, PIDs, mode, and observed execution.

```sh
# CPU replica lifecycle; GPU modes are recorded as untested/blocked if unavailable:
python tests/check_parallel.py --model /local/chat --mode replicas --report cpu-workers.json

# On a machine with a qualified runtime and two GPUs:
python tests/check_parallel.py --model /local/chat --tp-model /local/tp-compatible-chat --gpus 0 1 --report gpu-modes.json --require-all
```

The optional `--tp-model` selects a different TP-compatible checkpoint; otherwise
the same checkpoint is used for every mode. The second command separately checks
two replicas, model sharding, and TP, including
streaming, cancellation, failures, and cleanup. `passed` is true, false, or null;
`--require-all` fails when any requested mode is untested. A chat smoke report does
not certify other models, modalities, precisions, sizes, or production throughput.
Run `tests/check.py --device DEVICE` for the seven-modality single-device suite.

## Evidence and HiveMind relationship

CPU replicas have passed real concurrent requests, independent residency, stream
cancellation, process failure, subsequent restart, and shutdown checks. Transformers
tensor parallelism (CPU Gloo or GPU NCCL/RCCL/XCCL), GPU model sharding and native
multi-GPU llama.cpp splitting have runnable checks (`tests/check_parallel.py`) but no
pass is recorded in [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md#hardware-scope);
**treat them as implemented and unqualified.** The recorded GPU evidence there is
single-device. The implementations and runnable qualification checks are included
without claiming those hardware results.

The placement/replica distinction and support-vs-test-result approach were grounded
in the actual HiveMind source, including
[main-fresh's GIM guide](https://github.com/pgoode41/HM2/blob/main-fresh/menta_hli/docs/GIM_MULTIGPU_GUIDE.md)
and the newer
[multi-GPU strategy at 5e45612](https://github.com/pgoode41/HM2/blob/5e45612d42bd8e676734db1c447314fa824897c3/docs/GPU_VIRTUALIZATION/MULTI_GPU_STRATEGY.md).
Titanos uses its own process lifecycle and native Transformers TP; HiveWarden and
the older third-party `tensor_parallel` package are not runtime dependencies.

A manually triggered `native-smoke.yml` workflow runs the same checks and preserves
reports. GPU runners must already have a matching, qualified PyTorch installation.
The tiny random Llama TP fixture checks execution, not language quality. The workflow
has not been run on GitHub-hosted runners.

Managed llama.cpp has its own concurrent slots, replicas and native split modes;
see [NATIVE.md](NATIVE.md). Its layer/row/tensor options are not evidence that a
Transformers TP group passed, or vice versa.
