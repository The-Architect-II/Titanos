# Isolated music runtime contract

Titanos selects the child device before starting the isolated music interpreter.
The serving interpreter does not import the music Torch stack. Configured device
names are requests; they are not evidence that a model executed on that device.
`details()` reports `requested_device`/`requested_dtype` separately and leaves
observed `device`/`device_uuid` null until verified child execution metadata arrives.

## Device and environment boundary

The initial trusted runner contract supports CPU and NVIDIA CUDA. An explicit
HIP, XPU or MPS selection is rejected rather than substituted. Actual model and
hardware qualification remains separate from implementing this contract.

- `cpu` hides CUDA/HIP devices and requires auto/float32. Other explicit CPU
  dtypes are rejected.
- A full `GPU-...` UUID is checked with Titanos's strict `devices.pin` helper.
  It must also lie within an inherited UUID visibility mask. An explicit UUID
  combined with a numeric-only mask is ambiguous and is rejected; use a matching
  UUID mask.
- `cuda:N` indexes the inherited `CUDA_VISIBLE_DEVICES` list when present.
  The selected token becomes the child's entire mask and its runtime address is
  `cuda:0`. Without an inherited mask, N remains an ordinal in CUDA's original
  namespace. Numeric CUDA tokens are **not** reinterpreted as nvidia-smi indices.
  The child must report their observed physical UUID and exactly one visible GPU.
- `auto` retains the inherited accelerator scope. A nonempty CUDA mask selects
  its first token; an empty CUDA mask selects CPU. With no CUDA mask, the runner
  may choose a supported device within the unchanged inherited scope. Explicit
  requests never fall back to another device.
- Duplicate or malformed CUDA masks, out-of-range ordinals and missing/ambiguous
  UUIDs fail. Preserve `CUDA_DEVICE_ORDER` and other driver/visibility context.

The external child receives an allowlisted environment, its own temporary home
and cache locations, offline model flags and explicit thread counts. Parent
credentials, Python environment variables, preload hooks and PyInstaller state
are omitted. Restored original library paths retain driver directories while
excluding the parent's private Python/frozen directories. Shared system prefixes
such as `/usr` are not blanket-removed.

The interpreter runs with `-I -B`. A stdlib bootstrap resets inherited Windows
`SetDllDirectory` state in the child before importing the runner. The serving
process's environment and DLL search state are never modified by this launch.

## Request and result

The existing CLI remains:

```text
<python> <isolated bootstrap> <adapter> --request <json> --output <wav> --result <json> --progress <json>
```

The request includes the existing bundle, pinned code/model revisions, variant,
offline flag and supported generation controls, plus:

```json
{
  "runtime_contract": {
    "version": 1,
    "requested_device": "GPU-<full UUID>",
    "device": "cuda:0",
    "expected_uuid": "GPU-<full UUID>",
    "visibility": {"CUDA_VISIBLE_DEVICES": "GPU-<full UUID>"},
    "dtype": "bfloat16",
    "threads": 4,
    "interop_threads": 1,
    "max_audio_seconds": 600,
    "max_output_bytes": 134217728,
    "max_samples": 60000000,
    "max_steps": 100,
    "max_prompt_chars": 8000,
    "max_text_chars": 100000,
    "placement": "auto"
  }
}
```

Values come from the model/worker/settings, not those example constants.
`placement` carries the model's `placement` option (`auto` when unset) and
selects where weights live between stages. `offload` is the original behaviour:
every component loads on the host and moves to the device only for its stage.
`resident` builds every component (planner, text encoder, DiT and VAE) empty
on the meta device (`accelerate.init_empty_weights`, buffers created normally)
and fills it with the child's own streaming loader instead of any library
`from_pretrained`: each safetensors header is parsed strictly (known dtypes,
non-negative shapes, byte spans equal to numel x itemsize, offsets that tile
the data region exactly once), then each tensor is read into one writable
host buffer, copied to the device at its stored dtype, cast to the resolved
dtype on the device, and the host buffer is dropped before the next tensor.
No shard is mapped or opened whole: `safe_open`, `load_file`, `mmap` and
`UntypedStorage.from_file` are absent from the adapter (an AST check enforces
it). On Windows a whole-file open charges the shard to process commit at open
and a copy-on-write mapping is counted against a Job memory cap but only fails
later at page touch (both measured with small owned probes,
`JOB_CAP_SEMANTICS_2159Z.json`; run r3 exhibited the failure class this avoids,
0xC0000005 during the first DiT shard load through the earlier
`from_pretrained`/`safe_open` loader, but which class it hit is not proven:
`ROOT_R3_NATIVE_CRASH_TIMELINE_CORRECTION.json`), so
the host transient of the resident loader is exactly one tensor (the largest
pinned tensor is the planner embedding, 1,112,084,480 bytes BF16; the largest
DiT tensor is 157,286,400 bytes stored F32). On WDDM every device allocation
also charges process commit 1:1; the resident `basis` carries that as an
informational note on Windows and the launching kit does the host arithmetic.
Loading accounts for every key: a key seen twice across shards, a checkpoint
tensor whose shape the module rejects, an unexpected checkpoint key, a missing key
that is not one of the module's configured tied-weight keys (the planner's
`lm_head.weight`, which must then bind to the same tensor object as the input
embedding), a parameter that does not end on the device at the resolved dtype
or a buffer left on the host refuses the run with a message starting
`resident_loader_refused: <component>: ...`. The pinned silence tensor is
deserialised with the device as `map_location`. No component weight touches
host memory beyond that single-tensor transient; nothing returns to the host
and there is no CPU-offload fallback before or during generation (a device
out-of-memory fails the run instead of moving a component to the host). The
effective receipt's `placement_check.loader` records what the loader measured
(`{mode: "streamed", host_mapping: "none", files, tensors, bytes_read,
largest_tensor_bytes, components: {<component>: {files, parameters,
tied_keys, renamed_keys, buffers}}}`); `evidence.runtime.resident_loader`
carries the same block and is retained even when the run fails after loading,
while `evidence.runtime.placement` keeps its fixed key set. `offload` and
`host` still load through the library loaders unchanged. `auto` picks
`resident` when the same checks pass, otherwise `offload`. On CPU every request resolves to `host`; an explicit
`resident` on CPU is refused. Two receipt-level changes from the earlier
adapter apply to every placement: a CPU run now reports
`effective.cpu_offload: false` (the resolution is `host`; the earlier adapter
wrote `true` unconditionally on CPU), and the header-sizing refusals (a
safetensors file outside the four loaded components, a component without
safetensors) run before any weight loads for `host` and `offload` runs as well
as `resident`; the pinned bundle passes both.

The child sizes the run itself, before any `from_pretrained` or `torch.load`,
from the safetensors headers of every weight file the pinned recipe loads
(planner, DiT, text encoder, VAE): `total_resident_bytes` is the parameter
count of all four components times the item size of the resolved dtype (2 for
bfloat16/float16, 4 for float32; the DiT is stored F32 but counted by element
count, not file bytes), `peak_set_bytes` is the largest simultaneously held set
(max(planner, DiT) + text encoder + VAE, because the planner is released before
the DiT loads), `headroom_bytes` is a fixed 4 GiB, and `required_bytes` is
`total_resident_bytes + headroom_bytes`. Free device bytes come from
`torch.cuda.mem_get_info(0)`. No caller-supplied byte estimate or basis string
is accepted. An explicit `resident` request is refused **before loading any
weights** when `accelerate` (needed for meta-device construction) is not
importable, when the resolved dtype is not
supported by the device, or when free bytes are below `required_bytes`; the
child's error message starts with `placement_refused_before_load:` and carries
the measured numbers, the retained result records them under
`evidence.runtime.placement_refused`, and the parent surfaces that message
verbatim (`placement_refused`, 503) so the job error text names the reason. A
refused run is never placement evidence.
Every other nonzero adapter exit stays generic, but its diagnostic is no longer
lost with the request's temporary directory: `run_owned` reads the last 4096
bytes of the child's `stderr.log` before that directory is deleted, reduces it
to printable text, redacts credential values in their labelled (`api_key=`,
`token:`, `secret`, `password`, `authorization`, quoted or unquoted, with or
without a `Bearer`/`Basic` scheme word), CLI (`--api-key VALUE`) and bare
`Bearer VALUE` forms, JWT-shaped tokens and any 32+ character base64/hex-like
run that is neither a 64-hex sha256 nor identifier-shaped (letters and
underscores only), and reduces absolute Windows and `/home`, `/Users`, `/tmp`,
`/var`, `/mnt`, `/opt`, `/root` paths to `<path>/<final component>`. The parent
writes a bounded (at most 16 KiB) JSON record to
`TITANOS_HOME/outputs/music-failures/<utc>Z-<pid>-<8 hex>.json` carrying the
exit code, that redacted stderr tail, the child's `evidence.error` from
`result.json` and its last `progress.json` (each read only when at most 64
KiB), the executable/adapter basenames, the contract's device, requested
device, dtype, placement and threads, the request's duration, steps, seed and
mode, and the owning job id. The structured fields never carry the prompt or
lyrics text or a path; the `stderr_tail` field is the credential-redacted,
path-reduced tail of whatever the child printed to stderr (normally a Python
traceback naming adapter/library file basenames), so it is not a raw log copy
but it does keep any text the child itself chose to print. Retention is best
effort: a record that cannot be written never replaces the child's failure or
a refusal with another error. Refusals are retained the same way. The
API-visible job error carries only the exit code and that file's relative
name (`Music adapter process exited unsuccessfully (exit code N; diagnostic
outputs/music-failures/<file>)`, `native_process_failed`, 502), or
`diagnostic_unavailable` when the record could not be written; the matrix kit
copies up to four such records into a failed attempt's `exercise.diagnostics`
so the persistent receipt keeps them, each tagged `attributed: true` when its
record time is at or after the attempt's `utc_start` and `false` when it is
prior evidence from an earlier attempt or run in the same reused cell root.
Generation controls resolve as explicit non-null request values, then model
defaults, then process defaults. The parent explicitly supplies omitted sampling
steps from `TITANOS_DEFAULT_STEPS`; the child does not impose its own hidden
default on a normal Titanos request. Direct Engine and HTTP paths use this same
parent resolution before validating configured limits.
`max_output_bytes` is `max_music_output_bytes`; `max_samples` is
`max_music_samples` and counts frames times channels. The HTTP input body limit
still bounds uploaded reference audio. Unsupported reference audio is rejected
before launch. Owned process output, the returned WAV byte count, duration and
scalar sample count are bounded by the music settings.

`max_steps`, `max_prompt_chars` and `max_text_chars` carry the operator's
`TITANOS_MAX_STEPS`, `TITANOS_MAX_PROMPT_CHARS` and `TITANOS_MAX_TEXT_CHARS`
settings. The parent enforces them before child launch, including direct Engine
calls, and the child checks them again. Prompt and lyric bounds are independent;
raising one never raises the other. There is no additional 1,000-step ceiling
or fixed prompt/lyric character ceiling in the runner. The selected recipe's
default is 50 steps when none is supplied; a lower configured maximum requires
an explicit step count within that maximum. Requests are never truncated to fit.

The private request JSON is created by the trusted parent and has no second
fixed byte ceiling that would override these settings or penalize escaped
Unicode. Inventory/result readers retain their own byte bounds. Model-declared
duration limits, numeric types/ranges and the trained conditioning token window
still apply; raising a text character limit does not enlarge that model window.

The result outer shape remains `{effective, evidence, native_format}`. Require
`effective.dtype`, `effective.threads`, and `effective.interop_threads`, plus
`evidence.runtime` containing:

- `contract_version: 1`, observed `backend` (`cpu`/`cuda`), `resolved_device`,
  `device_uuid` (full NVIDIA UUID, or null for CPU), and `torch_version`.
- Actual `visibility`, `torch_num_threads` and `torch_num_interop_threads`.
- `visible_devices`, an ordered list of `{local_device, uuid}` from runtime
  enumeration. Explicit CUDA selection requires exactly one device. UUIDs must
  be unique and agree with the selected device and any parent-pinned identity.
- `identity_source: "torch.cuda.get_device_properties"` for CUDA or
  `"parameter.device"` for CPU. CUDA BF16 additionally requires observed
  `bfloat16_supported: true`.
- `placement`: `{requested, resolved, device_free_bytes, required_bytes,
  total_resident_bytes, peak_set_bytes, headroom_bytes, accelerate_version,
  basis}`. `requested` must equal the contract's `placement`; `resolved` is
  `host` exactly when the resolved device is CPU, otherwise `offload` or
  `resident`, and an explicit CUDA `resident`/`offload` request must resolve to
  itself. `device_free_bytes` is the measured `torch.cuda.mem_get_info` free
  value (null exactly on CPU); the four byte counts are non-negative integers
  with `peak_set_bytes <= total_resident_bytes <= required_bytes`;
  `accelerate_version` is the imported version string (required when
  `resident`, null when accelerate was not consulted or not importable); and
  `basis` is a short (at most 400 characters) statement of which condition
  decided. `effective.cpu_offload` must be a boolean equal to
  `resolved == "offload"`: a `resident` result with a CPU-offload receipt is
  rejected. The block's own arithmetic must agree with its resolution: a
  `resident` result with `device_free_bytes < required_bytes` is rejected, and
  an `auto` request resolved to `offload` must evidence the obstacle
  (`accelerate_version` null or `device_free_bytes < required_bytes`; an
  unsupported bfloat16 already fails the dtype check). The effective receipt
  repeats the block as `placement_check`.
- `parameter_placements`: actual parameter observations, each
  `{component, stage, device, dtype}` with `stage` one of `execute`, `offload`
  or `retain`. `dit`, `text_encoder`, and `vae` must each have an `execute` row
  on the resolved device; only `execute` rows count as execution evidence.
  `offload` rows are host-side observations after a stage and only appear when
  not resident; `retain` rows are on-device observations after a stage and only
  appear when resident. Placement on another GPU is rejected. Use canonical
  dtype names without `torch.`. The reported effective dtype must agree with the
  DiT; an explicit requested dtype must agree across all required components.

These observations must come from the actual framework and parameters, not a
copy of the requested configuration. Missing or inconsistent evidence prevents
publication of the returned audio. The hash-bound trusted runner is responsible
for validating its visible identity before loading models, applying dtype at
initial load, and recording truthful stage observations. Progress remains real
stage-level progress where the model has no per-step callback; process
termination must not be described as cooperative cancellation.

Progress and result publication is atomic: the adapter writes `progress.json.tmp`
or `result.json.tmp` completely, then renames it over the destination. On
Windows the parent's poll holds `progress.json` open for the duration of one
read, and a rename onto an open destination fails with `ERROR_ACCESS_DENIED`
or `ERROR_SHARING_VIOLATION`; the adapter retries that rename within a bounded
budget (`REPLACE_RETRY_ATTEMPTS` x `REPLACE_RETRY_SLEEP_SECONDS`, one second
in total) with the fully written temporary file kept in place, and surfaces
an exhausted budget or any other error unchanged. A failure report (the
`failed` result and progress) is published best effort and never replaces the
primary generation error; `tests/check_music_progress_publication.py` holds a
reader on the destination to prove both properties.

## Portable runtime discovery

With no explicit `<TITANOS_HOME>/music-runtime.json`, Titanos looks beside its
executable for `runtimes/ace-step-1.5/runtime.json`. The portable manifest has
`schema_version: 1`, `family: "ace-step-1.5"`, the approved `code_revision`,
relative POSIX `executable` and `adapter` paths, and
`files: {relative_path: {bytes, sha256}}`. Additional hash-bound source,
dependency and license inventories are permitted.

All file sizes/hashes and exact payload membership are verified, excluding only
`runtime.json` itself from its own file inventory. Escaping paths, symlinks,
Windows reparse points and unlisted payload files are refused. Model weights
remain in the separate selected model bundle. No target interpreter is executed
by receipt inspection.

Existing explicit home receipts retain priority and support their absolute
destination/executable/adapter and hash-string file map. An invalid explicit
receipt fails; it never silently falls back to another runtime. Portable file
qualification, native execution and music/listening quality are separate gates.

## Child interpreter mode

The parent launches the trusted child with `python -I -B -X utf8 -c <bootstrap> <adapter>`: isolated mode ignores `PYTHON*` variables, and UTF-8 mode keeps the child's stdout/stderr log files encodable for any character the pinned source or its libraries print (transformers prints a U+1F6A8 warning when the ACE model class is imported; under the locale codec that print raised UnicodeEncodeError before any weights loaded).

The child environment carries the user identity variables (`USERNAME`, `USER`, `LOGNAME`) when the parent has them and pins `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` inside the request directory: torch resolves its inductor cache directory with `getpass.getuser()` during import, and a child with neither a user name nor a pinned cache directory fails every diffusers import with "Artifact of type=precompile already registered in mega-cache artifact factory" (a partially imported `torch._dynamo` is re-imported by the lazy loader).

