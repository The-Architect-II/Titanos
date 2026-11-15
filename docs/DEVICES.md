# Device discovery and accelerator assignments

Titanos separates installed hardware, runtime visibility, configured placement and observed execution. Finding a GPU or NPU does not prove that a particular model can run on it. Model weights remain external; HiveMind is an optional client.

## Discover devices

```text
titanos devices --backend all --refresh
titanos assignment-devices --model chat --refresh
```

`all` combines installed OS/driver records with fresh, isolated runtime enumeration. Windows uses the built-in PowerShell path and DXCore metadata to discover display and compute devices, including integrated graphics and NPUs, without requiring a numerical Python package for physical discovery. Linux includes PCI display/processing devices. NVIDIA discovery reports complete physical UUIDs through the installed driver utility.

The existing interfaces expose the same information:

- `GET /api/devices?backend=all&refresh=true`
- `GET /api/assignments/devices?model=chat&refresh=true`
- `Client.assignment_devices(model="chat", refresh=True)`
- MCP `list_devices` and `list_assignment_devices`

Device records distinguish `installed`, `present`, `identity_kind`, `identity_strength`, `runtimes`, and `visibility`. Model-specific assignment choices add `compatible`, `assignable`, and actionable `reasons` with stable codes. A masked device remains listed. A failed probe preserves the last successful snapshot and reports its observations as unverified; it does not prove that hardware disappeared.

Vendor UUIDs, current-boot LUIDs, host PnP instances, PCI locations and runtime ordinals have different strengths. Exact UUID/LUID or unique current PCI associations can link views. Names alone never merge devices. Unmatched runtime views carry `physical_link=runtime_identity_only...`; they are not proof of additional physical cards. An OS instance ID is not automatically an executable runtime selector. Integration/discrete classification is reported only when the discovery source supplies it.

Runtime observations keep their own vendor and architecture fields. An OpenVINO view cannot replace a physical CUDA card's vendor or compute capability. Model-specific choices carry the selected runtime signature and preserve `physical_id`/`physical_signature` separately, so heterogeneous cards cannot become a homogeneous group merely because a plugin reports the same architecture label.

## Configure intent, then observe execution

`GET /api/assignments/{model}` reports `policy` (`automatic`, `explicit_cpu`, or `pinned`), `configured_device`, advisory `selection`, and separately cached `observed` placement when a model is already loaded. This read never calls model details or loads weights. An advisory recommendation has `observed=false`; loaded placement is not itself a kernel trace or model-quality result. Missing or masked direct model pins are reported even when there is no worker group.

```text
titanos assignment save chat --policy auto
titanos assignment save chat --policy cpu
titanos assignment save chat --group GPU-11111111-1111-1111-1111-111111111111
titanos assignment save chat --mode model_sharding --group GPU-11111111-1111-1111-1111-111111111111,GPU-22222222-2222-2222-2222-222222222222
titanos assignment save chat --group GPU-11111111-1111-1111-1111-111111111111 --group GPU-33333333-3333-3333-3333-333333333333
titanos assignment save chat --mode tensor_parallel --group cpu --tensor-parallel-size 2
```

The example UUIDs are placeholders; use IDs from this installation. A single group selects one device or one homogeneous parallel group. Repeated single-device groups are independent replicas and may use different compatible architectures. Single XPU/OpenVINO vendor UUIDs are supported where the model backend accepts them. CPU replicas and a CPU Gloo rank group are explicit choices. CPU cannot be mixed into an accelerator shard group.

A deliberate `policy=auto` or `policy=cpu` changes placement intent. Refresh, locale changes and ordinary reads do not rewrite explicit CPU or pinned choices. On the native backend, a deliberate GPU assignment transitions a legacy CPU/zero-layer model to automatic offload; an unrelated load does not reinterpret an explicit zero.

Automatic policy is shared by advice and runtime selection: prefer an observed compatible GPU, then a supported NPU/provider, then CPU. Matching installed-runtime pins are preserved; unavailable pins fail instead of choosing another device. DirectML auto selection preserves an explicit `device_id` and verifies the current LUID before a session uses it. A provider being installed is weaker evidence than a physical device being observed. OpenVINO GenAI images currently support CPU/GPU, not NPU images; raw prepared tensor/embedding models and GenAI chat have different device contracts.

Native llama.cpp layer/row splitting and experimental tensor splitting run in one owned native process. PyTorch tensor parallelism uses distributed collectives. They are not the same implementation: standard Windows CUDA Torch lacks NCCL, while CPU Gloo is a separate possibility. Group admission requires a known matching runtime vendor and architecture; two unknown architectures are not assumed homogeneous. Exact model plans, collective availability, memory, graph operators and output quality still require runtime validation.

A collective already known to be unavailable is rejected before hardware refresh or registry mutation. In particular, Windows CUDA PyTorch tensor-parallel assignments return an actionable NCCL error; this does not disable native llama.cpp tensor splitting.

## Honor runtime visibility

Inherited CUDA, HIP/ROCR, XPU/oneAPI and relevant driver visibility are preserved before child imports. A child may narrow an admitted scope; it cannot clear a mask to obtain another GPU. These are backend-specific visibility controls, not a cross-runtime workload ownership lease. Operators must coordinate actual device use separately.

A current CUDA ordinal is not an `nvidia-smi` index. New `worker --gpus` numeric input means the current CUDA-visible ordinal and is bound to an observed UUID. Full GPU/HIP UUIDs are preferred. Numeric or abbreviated CUDA masks require actual runtime UUID observation; they are never interpreted through NVML indices. Container NVIDIA index selectors use their distinct driver namespace. Missing, malformed or ambiguous scopes fail without an ordinal-zero substitute. HIP compound masks are observed in a fresh child and are not removed for discovery.

Native CUDA UUID checks use the selected runtime's own CUDA library directory, isolated from frozen-parent DLL settings. Native worker configuration with complete NVIDIA UUIDs does not require Torch. Explicit unavailable CPU/GPU/runtime choices remain errors; no silent CPU substitution follows an explicit accelerator request.

The CUDA probe maps each visible runtime PCI address to a driver UUID using documented APIs; it does not read a version-dependent device-properties structure. It requires the driver's distinct UUID-v2 API and rejects mismatched PCI identities, duplicate addresses/UUIDs, and MIG instance identities. MIG needs a separate supported assignment path. Probe failures retain bounded traceback context so a missing ABI symbol is not hidden behind a frozen-process wrapper error.

Normal Windows/driver environment context should be preserved when supervising the application. A System32-only PATH removes external Python/SDK dependence, but arbitrarily removing standard Windows OS/profile variables can prevent driver utilities from initializing. Credentials and interpreter injection variables are not required for discovery.

## Cache, refresh and managed runtimes

Inventories live under `TITANOS_HOME` with a persistent installation `host-id`. A copied home copies that installation identity; it is not a universal machine fingerprint. Refresh preserves absent historical records and writes each successful backend snapshot atomically.

`TITANOS_DEVICE_INVENTORY_TTL` controls cached observation age (default 30 seconds). `TITANOS_DEVICE_PROBE_TIMEOUT` bounds an isolated probe (default 30 seconds). Runtime profile recommendations are refreshed through the same explicit/age/scope lifecycle; a failed refresh cannot continue advertising the old recommendation as current. There is no new background poller.

Managed GGUF activation can take `runtime_profile` (CLI `acquisitions activate ... --runtime-profile PROFILE`). The server verifies that selected managed receipt and pins its executable into the model registration. Later activation of another global runtime does not silently switch that model. The acquisition's activation record includes the pinned profile/fingerprint/binary hash. An older record without this information is unknown, not presumed CPU or CUDA. Collecting weights with `load=false`, no selected profile and no installed runtime still allows a deferred, unqualified registration. Client-supplied executable paths are not accepted through acquisition activation.

`pin-devices` retains its explicit migration purpose for older NVIDIA numeric records. Historical stored worker indices came from the driver enumeration; stored `cuda:N` model settings came from the runtime enumeration. Migration cannot recover original intent after hardware has been reordered. Review the result and prefer full UUIDs for new assignments.

## Validation boundaries

The controlled discovery, assignment and interface checks exercise masks, reordered indices, missing identities, heterogeneous replicas, strict parallel groups, CPU intent, runtime-profile pinning and metadata refresh without loading models. Installed-device metadata readback is separate from frozen application execution and model/GPU qualification. See [accelerator execution](ACCELERATOR_EXECUTION.md) and the named capability evidence for actual validated combinations. A missing runtime, unsupported architecture, weak identity or unavailable collective remains visible and actionable.

Primary API references: [NVIDIA device identification](https://docs.nvidia.com/deploy/nvidia-smi/index.html), [CUDA visibility order](https://docs.nvidia.com/deploy/topics/topic_5_2_1.html), [DXCore enumeration and NPU attributes](https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore-enum-adapters), [DXCore integration properties](https://learn.microsoft.com/en-us/windows/win32/api/dxcore_interface/ne-dxcore_interface-dxcoreadapterproperty), and [Windows PCI address properties](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/nf-wdm-iogetdeviceproperty).
