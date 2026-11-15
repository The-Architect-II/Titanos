# Matrix kit: run the execution matrix on any host

The execution matrix (`docs/EXECUTION_MATRIX.md`) is filled by execution, not by narrative. The matrix kit in `tools/matrix_kit/`
turns one host into receipts: it detects which platform slots the host covers, runs every runnable model on each slot in every
execution mode the host can offer, and writes one receipt per cell in the matrix's own cell shape. Hosts run in parallel and
independently; receipts come back to the maintainer, a verifier upholds each pass against its receipt, and only then does a
cell turn green.

## One host, one command

```
git clone <repo> titanos && cd titanos            # or git pull to the commit named in the run request
python -m venv .venv && <activate>
pip install -e . && python install.py --catalog <release>/release.json --profile auto    # runtime profile for this host from a release catalog (python tools/build_release.py; docs/HARDWARE.md#installer-behavior); without a catalog, install the hardware's PyTorch wheel and then pip install -e ".[models]"
python tools/matrix_kit/build_manifest.py --sources <model_sources.json> --out <run>/manifest.json
python tools/matrix_kit/run_cells.py --manifest <run>/manifest.json --out <run>/receipts --host-label <role-label> [--run-id fleet-<role-label>-<yyyymmdd>] [--allowed-gpus GPU-...] [--stop-file <run>/STOP]
```

`model_sources.json` is written per host (it is a map of matrix `model_id` to the local artifact path and an identity check)
and stays out of the repository. `tools/matrix_kit/slots.py` prints the host detection alone. PyTorch XPU devices map to
`linux-x64-intel-gpu` on Linux x64 and, on Windows x64, to `win-x64-intel-dgpu-openvino` (discrete Arc) or `win-x64-intel-igpu`
(Iris, UHD, Meteor Lake Arc Graphics, Lunar Lake Arc 1x0V), decided by the OpenVINO probe's device type when it saw the same GPU
and by name otherwise; Titanos declares that path for both OSes (`tools/profiles.json` `xpu`, `docs/HARDWARE.md:46`). The kit
detects no llama.cpp native device, so the `llama.cpp OpenCL` token of `win-arm64-adreno-directml` (Titanos's catalogued
  2026-09-25 (wave C): the kit now probes a SYCL llama.cpp build when run_cells.py is given `--native-sycl <llama-server>` (slots.py probe_native_sycl, runtime `native-sycl`, device native:SYCLn), so the `llama.cpp SYCL` token is claimed by runs on the Intel GPU slots (E-101: pass on the Iris Xe, fails on both Arcs); only `llama.cpp OpenVINO` and `llama.cpp OpenCL` remain builder-filled.
`win-opencl-adreno-arm64` profile) and the `llama.cpp SYCL` / `llama.cpp OpenVINO` tokens of the three Intel GPU slots (the
catalogued `win-sycl-x64`, `win-openvino-2026.3.1-x64`, `ubuntu-sycl-fp16/fp32-x64` and `ubuntu-openvino-2026.3.1-x64` profiles)
are never claimed by a run; the builder fills those cells as untried (`backend_runtime_map.tokens_without_kit_runtime`).
`backend_runtime_map.native_llama_cpp_profiles` states one rule over every profile in `src/titanos/data/native-profiles.json`:
each is mapped to the slots of its platform and accelerator, or listed as unmapped with the reason (Vulkan and Windows ROCm
builds target GPUs the matrix files under slots defined by another API; a slot-axis decision for the owner).

```json
{
  "all-minilm-l6-v2": {
    "present": true,
    "path": "/path/to/models/all-MiniLM-L6-v2",
    "bytes": 91632065,
    "identity_check": {"method": "per-file sha256 vs matrix identity", "matches": "true", "detail": "sentence-transformers/all-MiniLM-L6-v2@1110a243..."},
    "spec": {"dtype": "float16", "load_options": {"variant": "fp16"}, "defaults": {}, "voices": {}},
    "request": {"voice": "vega", "size": "128x128", "steps": 2, "edit_only": false, "image_to_video": false},
    "env": {"TITANOS_TESSERACT_EXECUTABLE": "C:/tools/tesseract.exe"}
  }
}
```

`spec` carries any ModelSpec field beyond path/task/backend/device (the kit sets those per slot). `request` tunes the probe for
that model (edit-only image families, image-to-video reference frame, generation size and steps, TTS voice). `env` is added to
the server process for that model only. A model that is absent on the host is listed with `present: false` and a reason; the
kit records its cells as untried with that reason.

## Host labels

Every host in the execution matrix is named by a stable role label, never by a computer name, a generated Windows name, an
inventory name or an address. The form is `<hardware>-<form>-<os>-<arch>`, with a runtime suffix when it disambiguates. The
single mapping is `tools/lab_paths.HOST_ROLE_LABELS`; `tests/check_public_export_hygiene.py` fails when this table and that
mapping disagree.

| Label | Host |
|---|---|
| `blackwell-workstation-win-x64` | the RTX PRO 6000 Blackwell workstation (Windows x64; Core Ultra CPU with integrated graphics and AI Boost NPU); its local kit runs are labelled `matrix-kit-local-<...>` and prose says "this workstation" |
| `turing-laptop-win-x64` | the RTX 2070 Max-Q laptop with an Intel UHD Graphics 630 iGPU (Windows x64) |
| `rtx5070-desktop-win-x64` | the RTX 5070 desktop that also carries the AMD RX 7900 XTX and RX 9070 XT (Windows x64) |
| `rtx5090-desktop-win-x64` | the RTX 5090 desktop that also carries a GTX 1080 Ti (Windows x64) |
| `lunar-lake-laptop-win-x64` | the Core Ultra 5 226V laptop: Arc 130V iGPU and AI Boost NPU (Windows x64) |
| `precision-5680-laptop-win-x64` | the Dell Precision 5680 laptop: Iris Xe iGPU and RTX 3500 Ada (Windows x64) |
| `ryzen-ai-laptop-win-x64` | the Ryzen AI 5 340 laptop: Radeon 840M iGPU and XDNA2 NPU (Windows x64) |
| `intel-arc-win-x64-openvino` | the Intel Arc B580 + A770 desktop (Windows x64; OpenVINO, PyTorch XPU and llama.cpp SYCL runtimes) |
| `snapdragon-x-laptop-win-arm64` | the Snapdragon X laptop: Adreno X1-45 GPU and Hexagon NPU (Windows ARM64) |
| `intel-macbook-pro-macos-x64` | the Intel Core i9 MacBook Pro (macOS x86_64) |
| `apple-m1-macbook-air-macos-arm64` | the Apple M1 MacBook Air (macOS 15 ARM64, 8 GB unified memory; CPU, MPS and CoreML/ANE slots) |
| `linux-cuda-server-x64` | the Linux x64 server with two RTX 4090 and an unbound GTX 1080 |
| `gb10-desktop-linux-arm64` | a GB10 Linux ARM64 desktop (out of the current scope) |

Choose the label at launch. `--host-label <label>` writes it into every receipt and into `RUN_RESULT.json` as `host.label` and
into the default run id `matrix-kit-<label>-<UTC stamp>`; a fleet campaign passes `--run-id fleet-<label>-<yyyymmdd>`. The run id
becomes the reader family `matrix-kit:<run id>` and the mode-cell source in the matrix, so it carries the label and nothing else
about the machine. `run_cells.py` refuses a run id or label that contains a generated Windows machine name or the machine's own
hostname. What the operating system calls the machine is written to `HOST_DETECTION.json` only (`local_hostname`) and stays in
the run folder. A new host gets its label added to `HOST_ROLE_LABELS` and to this table first. The map from the lab's inventory
names to these labels is private and never enters the repository (`tools/public_export.json`, `private_files`); `tools/lab_paths.py`
applies it at the generators, so a rebuild of the matrix from the raw receipts reproduces the neutral output.

## What a cell state means here

| State | Written when |
|---|---|
| pass | the server started, the model loaded on the named device, one real request for the model's task returned a verified output, the model was listed as loaded, it unloaded, and the server exited cleanly |
| fail | any of those steps failed at runtime; the error text and the server log tail are in the receipt |
| na | not applicable by construction: the backend has no execution path on the slot runtime (`COMPAT`; a converted artifact for that runtime is its own catalog row; the OpenVINO runtimes read `model.onnx`, so `onnx` rows have a path there), Titanos refused the configuration by design (its refusal text is the reason when the probe produced one; for the multi-GPU modes `mode_refusal` applies the admission rules of `src/titanos/config.py` up front, for example tensor parallel outside Transformers chat/vision/embeddings, or any grouped mode for a single-device task or a one-device speech backend), or an admitted multi-GPU mode on a slot whose runtime cannot form a GPU group (CPU, DirectML, OpenVINO, QNN, Core ML, VitisAI, MPS) |
| untried (with reason) | the model artifact is absent on the host, a runtime package the backend imports (torch, onnxruntime, openvino; `onnx-diffusers` needs onnxruntime and torch) is not installed on the host (`RUNTIME_PROBE` against the `HOST_DETECTION.json` probes; an import error is not an execution, so such a cell is never a fail), the resource admission floor never cleared before STOP, a multi-GPU mode needs a homogeneous group this GPU slot does not offer in the run, or tensor parallel on a CPU/XPU slot (Titanos accepts an explicit Gloo/XCCL rank group there, which the kit does not run) |

The runner never writes `partial` and never marks a cell `verified`. A verifier reads the receipt and upholds or refutes the
pass; the maintainer binds the evidence id in `docs/CAPABILITY_MATRIX.md` and rebuilds the matrix.

## Execution modes

- `single`: one model on one device.
- `replicas`: two workers serving the same model on the same device (CPU or one GPU).
- `model_sharding`: one worker with a homogeneous GPU group (same name and compute capability) in `mode=model_sharding`.
- `tensor_parallel`: one worker with that group in `mode=tensor_parallel`, `tensor_parallel_size` = group size.

Multi-GPU modes run only when the host has at least two identical GPUs inside `--allowed-gpus`. A mode Titanos refuses for the
model by design is recorded as `na` with the reason, never left untried: with a group present Titanos's own refusal text is kept,
and without one `mode_refusal` applies the admission from `src/titanos/config.py` (tensor parallel only for Transformers chat,
vision and embeddings, lines 233-234 and 345-346; model sharding only for Transformers or Diffusers, 347-348; single-device tasks
such as translation, sentiment, rerank, pair classification, object detection, depth, segmentation and music, 211-212 and 336-337,
diarization, 74-77 through `diarization.py:33-38` and 334-335, and vad, 170-172, each cited by `SINGLE_DEVICE_TASK_LINES`;
one device per worker for xtts and qwen3-tts, 237-238; the compiled, native, sherpa and tesseract adapters refuse any parallelism
other than single). On a slot whose runtime cannot form a GPU group at all (a CPU, DirectML, OpenVINO, QNN, Core ML, VitisAI or
MPS slot) an admitted `model_sharding` is `na` by construction, and so is `tensor_parallel` except on CPU and XPU slots, where
Titanos accepts an explicit rank group (docs/MULTIGPU.md) that the kit does not run: those cells stay `untried` with that reason.
The builder (`tools/build_execution_matrix.py`) applies the same rules from `docs/execution_matrix_spec.json`
(`backend_runtime_map` with `runtime_availability`, `unrecorded_identity_rule`, `multi_gpu_modes_rule.admission`) to cells no run
has covered, and fills a pair's multi-GPU modes with the pair's own refusal when its single and replicas receipts are both
refusals; the explicit CPU tensor-parallel group counts only where every torch-backed CPU runtime the row's backend imports is
available. A backend's `cpu_path_tokens` are all-of (the runtimes it imports: `onnx-diffusers` needs `onnx CPU` and
`diffusers CPU`), so one unavailable token removes the CPU path. A cell the rules leave runnable is `untried: runnable by
construction` with the runtime tokens that grant the path named in the note (whether a fleet host covers the slot is a host
circumstance recorded in `named_devices` and `no_fleet_host`, not part of the claim), or `untried: no fleet host for slot ...`
on `linux-x64-amd-rocm` and `linux-x64-intel-gpu` (`no_fleet_host`): a missing host is a fleet circumstance, never a construction
na. `runtime_availability` records published-wheel facts per slot platform from a package/platform table (`packages`,
`platform_tags`, `token_packages`: no torch, openvino or openvino-genai wheel for macOS x86_64 or Windows ARM64 under the
pyproject pins, PyPI checked 2026-09-25; `tests/check_execution_matrix.py` derives from it that every token on every slot is
either published, declared unavailable or an executed deviation, so one standard applies to every platform); the kit needs
nothing from it because it records what is installed on the host in `HOST_DETECTION.json` and files a backend whose runtime is
absent as untried. `tests/check_matrix_kit.py` asserts the whole `COMPAT` table against the spec's `backend_runtime_map` through
`kit_runtimes` (kit runtime name to slot runtime tokens).

## Resource boundaries

Admission waits for three consecutive samples with available commit at or above `--min-available-commit-gib` and, for GPU
cells, free GPU memory at or above `--min-gpu-free-mib`; it never uses a time window. A `--stop-file` ends the wait and the
run. GPUs outside `--allowed-gpus` are reported in the receipt and never assigned; DirectML adapters are tied to the physical
GPU through the DXGI LUID and its PCI address (`tools/matrix_kit/dxgi_pci.py`), because DXGI adapter order is not CUDA order.
`--allowed-devices` keeps only the listed Titanos device strings (for example `cpu,dml:2,GPU.0`). Foreign workloads are never
touched.

Admission runs in two passes: cells whose floors are clear run first; a cell whose first sample is below its floor is deferred,
and the deferred cells wait afterwards, smallest artifact first, so one starved cell never stalls the campaign. Floors are
size-aware: at least 1.5 x the artifact bytes + 2 GiB of commit, and artifact bytes + 2 GiB of free GPU memory.

Admission and the in-cell guard fail closed on a missing commit reading: on Windows the reading comes from GlobalMemoryStatusEx;
on macOS and Linux it needs `psutil` in the kit environment (Linux falls back to `/proc/meminfo` MemAvailable). A host without
either records every cell untried `resource_reading_unavailable` instead of running it, so install `psutil` (`pip install psutil`)
on macOS hosts before a run. Contained admission (`--host-reserve-gib`, `--admission-profiles`; Windows only) caps the owned
process tree's *commit charge* (Job Object `JOB_OBJECT_LIMIT_JOB_MEMORY`; receipts record `resource_qualification.bound` =
`job_commit`). That cap is a fail-closed bound only for the allocation classes listed under *What the commit cap enforces* below;
the kit samples available physical memory beside commit, trips a contained cell when it drops under the reserve, records the tree's
peak working set uncapped, and writes the cap-semantics table into every contained receipt. The reserve and `--floor-margin-gib`
must be positive; the CLI refuses anything else before admission.

### What the commit cap enforces, and what it cannot (cap semantics)

Measured on the Windows 11 WDDM host on 2026-09-25 with synthetic children in owned Job Objects (records under
`control/music-gate-20260925/JOB_CAP_SEMANTICS_2159Z.json` and `NESTED_JOB_GPU_COMMIT_TEST_*.json` in the lab records, which stay internal;
`tests/check_matrix_kit_containment.py` case 10 reproduces the gaps under small caps). `containment.CAP_SEMANTICS` states the same
table and every contained receipt carries it verbatim as `resource_qualification.cap_semantics`
(`{'enforced': [...], 'accounted_not_refused': [...], 'unaccounted': [...]}` plus `peak_semantics`, `monitor`, `measured`):

| Class | What the Job cap does |
|---|---|
| heap / `VirtualAlloc` commit | **enforced**: refused at the cap (`MemoryError`, `DefaultCPUAllocator: not enough memory`) |
| CUDA device allocations under WDDM | **enforced**: WDDM charges every device allocation 1:1 to the process commit, so torch raises `OutOfMemoryError` at the cap although the GPU has memory free |
| pinned host memory | **enforced**: `cudaHostAlloc` / `pin_memory` fails at the cap |
| `safetensors.safe_open` on cpu or cuda | **enforced**: `OSError 1455` at open; the whole shard file is charged |
| copy-on-write file-mapping views (`MapViewOfFile FILE_MAP_COPY`, Python `mmap ACCESS_COPY`) | **enforced**: `OSError 1455` at map time; the view's charge is recorded in the peak |
| copy-on-write mappings made by `torch.UntypedStorage.from_file(shared=False)` | **accounted, not refused**: the view is created although its charge exceeds the cap, `PeakJobMemoryUsed` records the charge, the process commit reading does not move, and the first write to a mapped page kills the process natively with `0xC0000005` (the same exit code r3 showed; r3 loaded its DiT through `from_pretrained`/`safe_open`, it is not shown which class r3 hit, and the cause stays unproven; two independent correction records in the lab records say the same) |
| anonymous pagefile-backed sections (`mmap(-1, n)`) and shared writable file-mapping views | **neither accounted nor refused** |

`PeakJobMemoryUsed` also includes charges that were attempted and refused, so a job peak over the cap is evidence of an attempted
charge, not proof that committed memory exceeded the cap (r3: peak 24.16 GB against a 21.66 GB cap is an accounting/enforcement
discrepancy, not a proven committed-memory overshoot). The `CellGuard` remains the active monitor (STOP, cap hit, host commit and
physical starvation, GPU starvation); the owned child must avoid the two gap classes, which is why the adapter's resident loader
streams tensors by explicit file reads instead of mapping whole shards.

### WDDM-aware pre-launch check for contained cells

Before a `contained_resident` cell launches, `run_cells.device_resident_check` reads the GPU driver model for the exact device
(`nvidia-smi --id=<uuid> --query-gpu=driver_model.current`, `WDDM` or `TCC`; recorded as `resource_qualification.driver_model`) and
sizes the entry's safetensors **headers** at the dtype the resident load will use (numel x itemsize per component; `auto` resolves like
the adapter: bfloat16 on compute capability >= 8.0, else float32; tensor data is never read). With the resident music bundle layout
(`planner/`, `dit/`, `core/Qwen3-Embedding-0.6B/`, `core/vae/`) the peak resident set is `max(planner, dit) + text encoder + vae`
(the planner is released before the DiT loads); any other artifact is one component whose peak set is its total. On **WDDM**, where
device allocations are commit, the job cap must be at least `base 3 GiB (torch import + CUDA context, measured 2.7 GB) + peak set +
activation headroom 4 GiB`, else the cell is untried `blocked_by_host_resources: contained budget below the device-resident need on
WDDM (cap X < base + peak set + headroom Y)` with every number in `resource_qualification.device_resident_estimate` and nothing
launched. On **TCC** the estimate is recorded only. An unreadable driver model or unreadable headers fail closed
(`resource_reading_unavailable`). For the pinned bundle at bf16: planner 8,379,108,352 B, DiT 9,974,621,452 B, text encoder
1,191,553,024 B, VAE 337,390,852 B, peak set 11,503,565,328 B, required 19,019,758,096 B; r3's cap (21,664,305,152 B) covers it, so
the check is necessary, not sufficient: r3's cap satisfied the rule, so the resident set did not exhaust it; r3's crash cause is not
proven (its DiT load went through the earlier `from_pretrained`/`safe_open` loader; an independent crash-timeline correction record in the lab records agrees).
The 4 GiB headroom also has to absorb the streamed loader's one-tensor transient (one host buffer plus the device copy at the stored
dtype and, when the dtype differs, the device cast copy; on WDDM both device copies are commit): for the pinned bundle at bf16 at most
about 2.2 GB (the planner embedding, 1,112,084,480 B BF16, host + device; the DiT's F32 tensors are far smaller, 157,286,400 B stored,
plus their bf16 cast), which peaks while a component loads, before any activation or
KV cache exists, so it fits inside the headroom the activations use later and is not a separate term of the estimate.

ONNX image pipelines (`onnx-diffusers`) on a DirectML slot are registered with device `dml` and the adapter pinned by
`load_options.device_id` and the lowercase DXGI `adapter_luid`, which is what the backend validates.

## Dtype policy

`auto` first. On CUDA devices below compute capability 8.0 (Pascal, Volta, Turing) the plan is float16 then float32, because
those cards have no bfloat16 tensor cores. ONNX, OpenVINO, sherpa, tesseract and llama.cpp backends stay on `auto`. Every
attempt is kept in the receipt.

## Receipts

`<out>/HOST_DETECTION.json` (slots, GPU groups, unprobed runtimes), `<out>/receipts/<slot>.<model_id>.<mode>.json` (one per
cell with every attempt, admission samples, server log tail, `/api/models/details`), `<out>/RUN_RESULT.json` (cells in
matrix shape with counts). `tests/check_matrix_kit.py` guards the slot mapping, group rules, catalogue coverage and refusal
classification.

### Schema markers, the typed ingest gate and the legacy boundary

Every receipt the kit writes carries `receipt_schema` = `run_cells.RECEIPT_SCHEMA` (2) and `RUN_RESULT.json` carries `kit_schema` = 2.
A receipt's top-level `resource_qualification` is the complete measured block of the attempt that decided the cell (never a summary):
`kind`, `bound`, `verdict`, `job_commit_limit_bytes`, `host_reserve_bytes`, `floors`, the cap readback `containment`
(`requested_limit_bytes`, `job_memory_limit_bytes`, `limit_flags`, `child_in_job`, `verified_before_resume`),
`measured_peak_job_commit_bytes`, `measured_peak_process_commit_bytes`, `limit_hit`, `readings_complete`, `trip`, every guard
`sample` (`utc`, `phase`, `available_commit_bytes`, `available_physical_bytes`, `gpu_free_mib`, `gpu_used_mib`, `peak_job_commit_bytes`
for every phase after `launch`), the placement identity (`profile_key`, `receipt_sha256`, `adapter_sha256` as 64-character lowercase
sha256 hex, `placement`, `dtype`, `copies`), the `cleanup` evidence (`launched`, `active_processes_after`, `job_handle_closed`,
`forced_cleanup`, `final_query`), the constant `cap_semantics` table (`containment.CAP_SEMANTICS`: enforced classes, the two gaps,
peak semantics), the GPU `driver_model` and the `device_resident_estimate` of the WDDM-aware pre-launch check (also written into the
block of an attempt that check refused before any launch). `exercise` (output correctness) is written beside it and never mixed into
it. The RUN_RESULT row's `resource_kind`, `resource_verdict` and `resource_identity` are display fields only.

`tools/matrix_kit/ingest.py` reads a **current** run (`kit_schema` >= 2) through its detailed receipts alone: a pass or fail cell whose
receipt is missing, resolves outside the run folder, is unparseable, without `receipt_schema` >= 2 or without the block enters the
matrix as `untried` (`receipt_missing_or_unparseable` / `receipt_schema_missing`), with no fallback to the row. The receipt must be the
row's own (`ingest.binding_problems`): its `run_id` equals the RUN_RESULT run id and its `model_id`, `slot.slot`, `mode` and
`slot.device` equal the row's, so a cell never enters on another cell's complete receipt. A `contained_resident` block is accepted only
when `ingest.validate_contained_block` finds nothing wrong: verdict `within` under bound `job_commit`; positive integer cap and reserve,
the reserve equal to the RUN_RESULT `host_reserve_bytes` and `floors.commit` equal to cap + reserve (the kit's contained admission
floor); integer commit and GPU floors; readback equal to the cap with `JOB_OBJECT_LIMIT_JOB_MEMORY | KILL_ON_JOB_CLOSE` set, the
breakaway flags clear, the child in the job and verified before resume; a measured job peak strictly under the cap that is at least
every sample peak and the final owned-job query peak (itself under the cap), `limit_hit` false, `readings_complete` true, no trip; a
non-empty sample sequence with parseable, non-decreasing timestamps whose phases follow the kit's order (`launch`, `server_start`,
`model_load`, `request`, `unload`), in which every host commit and physical reading stays at or above the reserve and no job peak
reaches the cap (the guard's trip point) - a pass must cover `model_load` and `request`, a fail carries the recorded prefix up to its
failing step together with the exercise error, so a measured within-bound load failure enters as `fail`; a complete placement identity
whose `dtype` equals the row dtype and whose `copies` match the mode (2 for replicas, else 1); cleanup with `launched` true,
`active_processes_after` 0, the job handle closed and a final query without error; and an `exercise` block whose state is pass or fail
and equals the row state. An `artifact_floor` block must say `not_contained` and carry the kit's admission-plan `reason`
(`run_cells.artifact_floor_reason`, mirrored by `ingest.artifact_floor_reason`) for the receipt's `spec_placement` and device: a
resident spec on a GPU slot keeps the artifact floor only when the host could not contain it (`containment_unavailable`, RUN_RESULT
`containment_available` false), so a contained cell cannot be relabelled artifact_floor. Any problem makes the cell
`untried resource_qualification_evidence_incomplete: <problems>; functional output retained separately (exercise <state>)`;
`exceeded` and `unknown` stay untried as before, with the exercise state in the note. A valid WAV under an unmeasured bound is output
evidence, not qualification.

A RUN_RESULT **without** `kit_schema` is a **legacy** run only when its `utc` parses and lies before
`ingest.LEGACY_RESULT_BOUNDARY_UTC` (`2026-09-25T19:00:00+00:00`) **and** the sha256 of the RUN_RESULT.json file is listed in the pinned
legacy population `tools/matrix_kit/legacy_results.json` (the 78 results under `<lab-records>/runs` bound before the
boundary, hashed read-only on 2026-09-25; none carries a contained_resident row), or `--legacy-run <run_id>` names it explicitly (the
family records `legacy_basis` `pinned_sha256` or `legacy_run_flag`). Its cells then enter as the pre-boundary ingest bound old
full-artifact receipts (nothing relabelled) and the family records `kit_schema` 1 with the boundary; a `contained_resident` block is
never legacy evidence (`untried resource_qualification_legacy_contained`, even with a complete identity), because the pre-boundary
ingest accepted `within` on the identity alone. A result without the marker whose `utc` is missing or at/after the boundary, or whose
hash is not pinned, is refused with an error before anything is written: a self-reported earlier `utc` never opens the legacy path.
`tests/check_ingest_gate_negative.py` proves the gate red against the pre-change ingest (commit 467f764) and green now (receipt binding,
sensor/verdict consistency, artifact relabelling, fail-at-load, the pinned legacy population); `tests/check_matrix_kit_containment.py`
runs a contained pass and a contained fail at model load on a fake server and binds both kit-written receipts through ingest.

When two adapters share one slot id on a host (two AMD dGPUs under `win-x64-amd-dgpu-directml`, two Intel Arcs under `win-x64-intel-dgpu-openvino`), receipt names and work dirs carry the device: `<slot>@<device>.<model_id>.<mode>.json`, and every executed cell records `device`. `tools/matrix_kit/ingest.py` merges such cells per (model, slot, mode): the best outcome becomes the cell (an upheld pass over a refuted pass over a fail over a refusal), the other adapter stays in the note, and a verdict can be keyed per adapter as `<model_id>|<slot>@<device tag>|<mode>`.
