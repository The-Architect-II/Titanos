# Managed GGUF runtime

Install the light engine (no PyTorch) and a platform-specific native runtime. From
a source checkout:

```sh
python -m pip install -e .
titanos native-install --profile ubuntu-x64 --plan    # packaged catalog; --catalog FILE overrides it
titanos native-install --profile ubuntu-x64
titanos init
titanos pull agent Qwen/Qwen2.5-0.5B-Instruct-GGUF --revision 9217f5db79a29953eb74d5343926648285ec7e67 --task chat --backend llama-cpp --model-file qwen2.5-0.5b-instruct-q4_k_m.gguf --device cpu --anonymous
titanos serve
```

`--plan` prints the asset URL, its SHA-256 and the target directory without
downloading anything. Pick the profile for your OS and architecture from the
catalog table below; `ubuntu-x64`, `ubuntu-arm64`, `win-cpu-x64`, `win-cpu-arm64`,
`macos-x64` and `macos-arm64` are the CPU builds. The release installer's `engine`
profile (`python install.py --catalog release.json --profile engine`) gives the same
engine-only environment once a release catalog with a plain `X.Y.Z` version exists;
none is published yet, see [GETTING_STARTED.md](GETTING_STARTED.md).

Public/gated downloads use the existing optional-token pipeline. Pull selects the
chosen GGUF quantization, all its shards if split, and the optional `--mmproj`.
Only complete local checkpoints are used during inference. No native runtime or
model download happens when a remote client sends an inference request.

A GGUF embedding checkpoint is registered the same way with `--task embeddings`
(registry `task: "embeddings"`, `backend: "llama-cpp"`); the owned `llama-server`
then starts with `--embeddings`, plus `--pooling` when the registration's
`native.pooling` (`none`, `mean`, `cls`, `last`, `rank`) names one, and `POST
/v1/embeddings`, Ollama `/api/embed` and `/api/embeddings` are proxied to it.
`encoding_format` reaches llama-server (base64 is answered natively); the
Titanos-only `max_tokens`/`truncate` controls and `dimensions` answer a typed 400
naming the field. See [OPTION_PARITY.md](OPTION_PARITY.md#gguf-embedding-models).

A local directory uses `add` with the same arguments. `--runtime /path/llama-server`
uses an explicit existing binary. Otherwise the most recently installed native
receipt supplies the executable; its recorded SHA-256 is checked before launch.
Pin an executable explicitly for a registry that must retain an older runtime.
All relative model/projector files live under the registered model directory.
On Windows the native child runs inside a private Job held by its supervisor, so
it cannot outlive Titanos; [Windows native command cleanup](WINDOWS_COMMAND_CLEANUP.md)
describes the same Job mechanism for OCR commands.

## Native options

Use `--native-options options.json` when registering a model. CLI `--model-file`,
`--mmproj` and `--runtime` override the matching fields in that file.

```json
{
  "model_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
  "context_size": 8192,
  "slots": 4,
  "threads": 4,
  "gpu_layers": 0,
  "split_mode": "none"
}
```

`flash_attn` (`auto`, `on`, `off`) and `cache_type_k` / `cache_type_v` (runtime default
`f16`; `q8_0` and the other runtime cache types) pass to llama-server's `--flash-attn`
and `--cache-type-k/-v`; unset values add no argument. A quantized V cache requires
`flash_attn: on`. `reasoning` (`on`, `off`, `auto`) passes to llama-server's
`--reasoning`, for example `off` for a model whose chat template thinks by
default; unset, it adds no argument. `reasoning_budget` (an integer, -1 or more)
passes to `--reasoning-budget`: `0` ends the thinking at once, for a model that
still thinks under `reasoning: off` (Nemotron 3.5 Bolt E4B under b11402); unset,
it adds no argument.

Context capacity and slots follow the pinned native runtime's semantics; inspect
`details` for actual per-slot capacity. CPU is the default. Accelerator use requires
`gpu_layers > 0` and an explicit device. For NVIDIA, use full `GPU-...` UUIDs,
comma-separated for multiple GPUs. Titanos checks CUDA runtime visibility in a
fresh process and requires a discoverable CUDA runtime library. Other native
backends use `--device native:DEVICE`, with DEVICE taken from that binary's
`--list-devices` output, for example `native:Vulkan0`. These addresses are explicitly
not guaranteed stable physical identities. Confirm them after hardware changes.

Multiple selected devices require `split_mode` of `layer`, `row` or `tensor`.
Optional `tensor_split` supplies one positive proportion per selected device.
`layer` is model/layer placement; `row` and experimental `tensor` split computations
as supported by llama.cpp. These do not rename or replace Transformers' distinct
model-sharding and tensor-parallel paths. Native accelerator placement and its
kernel execution remain pending qualification on real hardware.

Each native model is lazy-loaded into its own managed process and stays resident
until unloaded. Set up multiple copies using ordinary workers:

```sh
titanos worker agent-a --models agent
titanos worker agent-b --models agent
```

A native worker names exactly one model. `--device cpu` (the default) uses CPU;
`--device auto` inherits model placement. `worker --device GPU-...` or
`--device native:DEVICE` selects its accelerator,
so replicas of the same model can occupy different cards. The registered model
must already enable GPU layers for accelerator replicas. Worker threads and request
timeout apply; native split options remain on the model. Requests route to
an available native slot, with optional `worker` targeting. The pending queue is
bounded and times out. `TITANOS_NATIVE_QUEUE_CAPACITY` sets its positive integer
waiting-admission ceiling (default 32), and `TITANOS_NATIVE_QUEUE_TIMEOUT` sets the
positive finite wait in seconds (default 30; fractional seconds are supported).
These process settings are captured when the pool is created and retained from
the application's settings snapshot on registry reload. An embedded app's
explicit `Settings` takes precedence over ambient environment values. Restart
the application with new settings to change that snapshot. Discover the values
in `/api/info`'s settings data or `titanos settings --json`.

Capacity applies to requests waiting for an available native slot, across this
pool. A request stops counting as waiting once it reserves a slot, including
while that selected worker starts; active/pending worker slots keep their
existing separate limit. The setting is not background asset-job capacity.
Queue timeout is not model startup timeout, whole-request deadline, HTTP
keep-alive or model idle retention. A shorter request deadline/cancellation can
still end the wait earlier. Full and timed-out queues retain the existing 503
`queue_full` and `queue_timeout` errors. Inspect `/api/workers` for native PIDs, active counts and
actual native-slot processing observations. Unload refuses a busy native worker.
No failed request is replayed; a subsequent request can restart a failed process.

## API contract

`/v1/chat/completions` preserves function tools, tool choice, assistant tool calls,
tool result IDs, structured JSON formats and native SSE deltas. `/v1/responses`
supports stateless input/instructions, function tools, text format and streaming (the server's SSE is completed with sequence numbers, output/content indexes and a full `response.created` object, which stock SDK stream helpers require).
`store` must be false; stored conversation IDs, hosted tools and background
Responses are not implemented. Use an agent/client to execute tools and carry
conversation state. Model training determines the usefulness of tool selection.

The Transformers and compiled text adapters reject unsupported tool/structured
requests instead of dropping fields. Native media input accepts inline data URLs;
no remote URL or local-file fetch is exposed. Vision requires a matching GGUF
projector and a model the selected native binary supports.

## Platform catalog and upgrades

The catalog is `src/titanos/data/native-profiles.json`, shipped inside the package
and byte-identical to `tools/native-profiles.json`; `tools/build_release.py` copies it
to `release/native-profiles.json`. It pins llama.cpp release `b10809` with immutable
asset URLs and SHA-256 values. Its twenty profiles:

| Platform | Profiles |
|---|---|
| Linux x64 | `ubuntu-x64` (CPU), `ubuntu-vulkan-x64`, `ubuntu-rocm-10.0-x64`, `ubuntu-sycl-fp16-x64`, `ubuntu-sycl-fp32-x64`, `ubuntu-openvino-2026.3.1-x64` |
| Linux ARM64 | `ubuntu-arm64` (CPU), `ubuntu-vulkan-arm64` |
| Windows x64 | `win-cpu-x64`, `win-cuda-12.4-x64`, `win-cuda-13.3-x64`, `win-vulkan-x64`, `win-rocm-10.0-x64`, `win-sycl-x64`, `win-openvino-2026.3.1-x64` |
| Windows ARM64 | `win-cpu-arm64`, `win-opencl-adreno-arm64`, `win-cuda-13.4-arm64` |
| macOS | `macos-arm64`, `macos-x64` |

A binary must match OS and architecture; vendor drivers and runtime prerequisites
remain external. Every entry carries the catalog note `Native execution pending on
this platform`: a listed profile is an installable runtime, not a device
qualification. The recorded accelerator GGUF pass is E-069 in
[CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md) (Windows x64, `win-cuda-12.4-x64`,
Blackwell). CUDA package names are not a promise of support for every GPU
architecture; use the PyTorch `cu126` profile for the Pascal path until native
Pascal execution is qualified.

`src/titanos/data/native-profiles-b12284.json` pins llama.cpp `b12284` (commit
`9e9619d03`) as an additional release, not the default: the loader reads one release
per catalog, so it is chosen by path (`--catalog`, the app's runtime catalog option or
`TITANOS_NATIVE_CATALOG`). It mirrors the sixteen b10809 profiles whose asset names
b12284 still publishes, with SHA-256 and size from the release; the OpenVINO 2026.3.1
and CUDA 13.3/13.4 profiles are absent because b12284 ships 2026.4.1 and 13.5 builds
instead. It adds `win-cuda-13.5-x64` (llama zip plus cudart zip, like 12.4), which
needs a driver that supports CUDA 13.5. Three profiles have run a model through
Titanos (2026-11-13): `win-cpu-x64` on CPU, and `win-cuda-12.4-x64` and
`win-cuda-13.5-x64` on one RTX PRO 6000 Blackwell (Windows, driver 610.88) with full
offload; of these, only the 13.5 build ships native Blackwell (sm_120) kernels. The
other profiles are untried. b12284 also publishes Ubuntu CUDA archives (12.8 x64,
13.5 x64 and arm64); the catalog does not carry them yet.

Linux NVIDIA: the default b10809 catalog has no Linux CUDA archive, so it offers
`ubuntu-vulkan-x64` (select the card with `--device native:Vulkan0`, taken from the
binary's `--list-devices` output) or a self-built `llama-server` passed with
`--runtime /path/to/llama-server`; a CUDA-built local binary still takes full
`GPU-...` UUIDs. The PyTorch `cu126`/`cu128` profiles are the Linux CUDA path with
recorded evidence (Ada, E-022 to E-024) for Transformers and Diffusers models.

The server (`titanos runtime-profiles`, Studio setup) recommends one profile: the
newest CUDA build whose toolkit version the installed driver supports and whose
architecture floor covers every visible card (CUDA 13 builds need compute
capability 7.5 or higher; a Blackwell card is a candidate only for a 12.8 or newer
build), otherwise the platform's CPU build. Any listed profile can still be chosen
explicitly; device support is verified when the model loads. The recorded
Blackwell pass E-069 used the 12.4 build, which that rule never recommends for a
Blackwell card, so it was an explicit choice.

The installer verifies archives, refuses archive traversal, tests binary loading,
and only then switches its receipt. It preserves existing runtime directories.
To mirror native assets, copy the selected catalog assets into a static directory
and remove their `url` fields; the installer resolves `file` relative to that
catalog, including loopback HTTP. HTTPS is required for remote feeds.

For an engine update:

```sh
titanos upgrade --catalog /new/release.json --profile engine --plan
titanos upgrade --catalog /new/release.json --profile engine
```

Use a numerical profile instead of engine when upgrading a PyTorch/compiled runtime.
Registrations and weights remain in TITANOS_HOME; restart using the new executable.
Switching back to the previous executable is rollback. Native-install likewise
activates a separately installed native runtime. No in-place background update runs. On Windows the supervisor owns the native executable and its descendants in a private Job; see [Windows native command cleanup](WINDOWS_COMMAND_CLEANUP.md).

## Serving defaults and what they cost

A new registration without explicit `native` values takes the process defaults
`TITANOS_NATIVE_CONTEXT_SIZE` (8192) and `TITANOS_NATIVE_SLOTS` (4): four parallel
slots of 2,048 tokens each. The context is one allocation shared by the slots, so
`context_size / slots` is the per-request context a runtime starts with (a larger Ollama
`options.num_ctx` resizes it; see below); raise both together to keep it. Saved registrations keep the
values they were written with. The 2026-10-06 pass that set these defaults, observed on
one shared GPU with Qwen2.5-0.5B Q4_K_M, read a first token of 81 ms for 4 concurrent
streams on 4 slots against 346 ms on 2 slots at the same total context, for 48 MiB more
GPU memory; the tables, the batch and ubatch, flash-attention and KV-cache observations
and the memory rule to apply before a load are in [PERFORMANCE.md](PERFORMANCE.md).

## Context from num_ctx

Ollama clients, and HiveMind's gateway on every OpenAI chat request, state the context they want as `options.num_ctx`.
On a llama-cpp model it is honoured on `/api/chat`, `/api/generate`, `/v1/chat/completions` and `/v1/completions`:

| `num_ctx` | What happens |
|---|---|
| At most `context_size / slots` | Nothing; the running per-slot context covers it |
| Above that, up to the largest per-slot context | The leased runtime starts, or restarts once its in-flight requests have finished, with a per-slot context of the next power of two at or above `num_ctx` (capped at the largest), so `--ctx-size` becomes that times `slots`. It keeps that size until it is unloaded |
| Above the largest per-slot context | 400 `context_limit` on `/api/chat` and `/api/generate`, before any runtime is asked; on `/v1/chat/completions` and `/v1/completions`, where `options` is foreign, not honoured and named in `X-Titanos-Ignored` |

The largest per-slot context is the registration's `native.max_slot_context`, else `TITANOS_NATIVE_MAX_SLOT_CONTEXT`, else
the `context_length` in the GGUF header (the trained context), and never less than `context_size / slots`. A
registration can also start larger with `native.slot_context` (then `context_size` defaults to `slot_context x slots`).
A restart waits at most `TITANOS_NATIVE_QUEUE_TIMEOUT` for in-flight requests (503 `queue_timeout` after that), and a
`require_loaded` request never restarts a runtime (400 `context_limit` when the loaded one is too small). A larger
context needs a proportionally larger KV cache on the device; memory admission checks it before anything is stopped
([OPERATIONS.md](OPERATIONS.md#gpu-memory-admission)).
