# Model providers and one-stop setup

Titanos keeps model discovery/acquisition separate from inference asset jobs.
The ordinary flow is **Add a model → pick a task and source → Search → Download and
run**, then Run. Local-folder registration remains available for experts.

For required files, preprocessing, runtime/profile distinctions and minimal valid registrations, use the [model-adapter reference](MODEL_ADAPTERS.md). It separates configured compatibility from named model/device qualification.

## Provider contract

Every provider implements:

- task-aware `search`;
- exact source/revision `inspect`;
- a secret-free immutable acquisition plan;
- acquisition into inactive staging with progress/cancellation;
- integrity verification before a prepared path can be activated.

Plans retain provider, canonical source, requested and resolved revision,
selected files, expected bytes/digests, task/backend/device, runtime status,
license/access state and compatibility uncertainty. Search tags and model pages
are candidates, not runtime proof.

Implemented providers:

- **Hugging Face** — uses the installed supported `huggingface_hub` SDK,
  resolves the caller's revision to an exact SHA, selects one compatible format,
  reuses the Hub cache and verifies every selected local file before activation.
- **GitHub · Sherpa ONNX TTS** — uses the official
  `k2-fsa/sherpa-onnx` `tts-models` release API, requires the release asset's
  SHA-256 digest, supports resumable HTTP range downloads, safely extracts one
  recognized TTS archive, and records its model-card/license evidence.
- **Sherpa ONNX · anonymous diarization** (`github-sherpa-diarization`) —
  offers the reviewed Pyannote 3.0 segmentation and ERes2Net Base embedding pair.
  Its product catalog pins each component's bytes, ABI, license and source;
  acquisition uses the existing durable prepare/activate lifecycle. The roughly
  46 MB external bundle supports CPU anonymous speaker intervals and simultaneous
  speech. It is a reviewed model recipe, not a promise that arbitrary diarization
  checkpoints share this interface. See [audio analysis](AUDIO_ANALYSIS.md).
- **ACE-Step 1.5** — offers only the reviewed `xl-sft`, `xl-turbo`, and
  `xl-base` bundle recipes. Each plan binds the official core and selected XL
  DiT revisions, excludes checkpoint Python, and acquires a complete
  component-hash inventory for the isolated `ace-step` music adapter.

Piper and Whisper catalog data hosted by Hugging Face remain Hugging Face
sources; they are not counted as independent providers.

## Durable acquisition state

Acquisitions live under `TITANOS_HOME/acquisitions` (or the output owner's
sibling directory in embedded tests), separate from `/api/jobs`.

States are `queued`, `running`, `cancelling`, `cancelled`, `failed`,
`interrupted`, `prepared`, `active`, and `deleted`. Records include stage and
byte/file progress. An interrupted server marks active records interrupted and
does not replay them. Retry uses the same immutable plan; providers may resume
verified partial data.
Each progress record names its `unit` (`bytes` for byte stages, `files` for the
per-file download and archive-extraction stages, `steps` for runtime-setup checks);
the Studio labels totals with that unit and infers it from the stage for older records.

Only `prepared` records with their owned manifest/path can activate. Unknown
license evidence requires reviewing every retained document and acknowledging
their combined SHA-256. Failed/cancelled work never enters
the model registry. Activation atomically updates the existing safe registry,
optionally loads the model, and leaves external weights distinct from unload or
registration deletion.

### Download layout

`titanos pull` writes each pinned snapshot into a plain per-revision directory, `<TITANOS_HOME>/models/<owner--repo>@<revision12>` by default or under `--cache-dir`, and registers that directory. There are no Hugging Face cache symlinks: the cache layout links snapshots to blobs, which a stock Windows account cannot create without Developer Mode or elevation (`WinError 1314`), so downloads would complete and then fail to register. The directory is independently replaceable and can be copied between machines together with its registry entry.

## HTTP and agent interfaces

- `GET /api/model-providers`
- `GET /api/model-providers/{provider}/search`
- `POST /api/model-providers/inspect`
- `POST /api/model-providers/import`
- `POST /api/acquisitions/plan`
- `POST`, `GET /api/acquisitions`
- `GET`, `DELETE /api/acquisitions/{id}`
- `POST /api/acquisitions/{id}/retry`
- `POST /api/acquisitions/{id}/activate`

`Client`, MCP tools/resources, and the `providers`, `discover`,
`source-inspect`, `import-link`, `acquire`, and `acquisitions` CLI commands use
the same records. Idempotency keys prevent accidental duplicate submissions.

## Runtime and hardware meaning

Plans report required runtime modules, missing modules, requested device, known
cached device identities, and explicit unmeasured qualification. Runtime/package
availability and provider enumeration are not model inference proof. Titanos
does not silently substitute an unavailable accelerator.

For managed native chat, the same flow selects an explicit GGUF (preferring the
balanced CPU `Q4_K_M` variant), prepares a server-configured checksum-pinned
llama.cpp profile, acquires the exact model revision, and registers the relative
GGUF path through the existing `NativeWorker`. `RuntimeSetup` state is separate
from model acquisitions and inference jobs:

- `POST /api/runtime-setups/plan`
- `POST`, `GET /api/runtime-setups`
- `GET`, `DELETE /api/runtime-setups/{id}`
- `POST /api/runtime-setups/{id}/retry`

The Client, MCP and `runtime-prepare` / `runtime-setups` CLI surfaces use those
same records. Catalogs, asset URLs, executable choices and target roots are
server/operator policy; network callers supply only a configured profile name.
The provenance-augmented b10809 catalog ships as package data at
`src/titanos/data/native-profiles.json`. `--native-catalog` has highest
precedence, then `TITANOS_NATIVE_CATALOG`, then that trusted packaged default.
`/api/runtime-profiles`, Client, MCP, CLI and Studio expose only profiles
compatible with the server host; Studio labels the currently supported managed
CPU choices with human-readable names.
The active receipt binds every native file and is reverified before reuse.

Sherpa uses the packaging-owner contract:

- `sherpa-onnx==1.13.8`
- `sherpa-onnx-core==1.13.8`
- CPU only in this adapter

The 1.x runtime includes mixed native licensing, including GPL espeak-ng /
piper-phonemize components documented by upstream issue 3731. Do not describe a
Sherpa-containing binary as Apache-only. Model archives retain their separate
source, digest and model license.

Portable builds that collect this runtime require the packaging owner's
hash-bound `sherpa/source-bundle` (`--sherpa-source-dir`). It retains Sherpa
commit `11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf`, Piper MIT, eSpeak/ucd
GPL-3.0-or-later, Linux ALSA LGPL-2.1-or-later, embedded ONNX Runtime 1.28.2
sources/notices, native bytes and separate SBOM entries. This is provenance and
redistribution evidence, not a bit-identical rebuild or model-quality claim.

## Model-page links

Hugging Face model/tree/blob/resolve links and official Sherpa release asset
links normalize into their structured provider. Other public HTTPS pages are
fetched only for bounded metadata: DNS is restricted to public addresses, the
connection pins the validated address while TLS authenticates the hostname,
every redirect is revalidated, compressed/non-HTML/oversized responses are
rejected, and a whole-operation deadline applies. Ordinary query parameters are
retained; credential-like query names are refused.

Studio renders the page's title, description, declared license and useful links
as unverified claims. Source URL, redirects, bytes and content SHA-256 remain in
progressive technical details. Following a recognized provider link starts a
fresh provider inspect/plan step; page prose never qualifies, acquires or runs a
model. Scripts, linked assets and provider commands are never executed.

## Security

- No revision fallback.
- No remote checkpoint code or network-enabled trusted pickle.
- No automatic gated access or unknown-license acceptance.
- No embedded provider tokens in plans/records/logs.
- Archive paths, links, file counts, expanded bytes, download bytes,
  redirects and SHA-256 are bounded and checked.
- Runtime setup installs an official checksum-pinned executable into owned,
  versioned storage and atomically switches a receipt; it never mutates the
  serving Python environment or adopts a foreign process.

## Managed accelerator runtimes

Managed native setup offers the packaged profiles for the current operating
system and architecture, including accelerator builds. It recommends a CUDA
candidate when visible NVIDIA identities, the installed driver API and toolkit
architecture support agree. Explicit CPU selection is retained. A recommendation
is not an inference qualification: the selected binary must still load the model
on the requested devices.

CUDA candidates account for the [CUDA 13 removal of pre-Turing library support](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html)
and [Blackwell support beginning with CUDA 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/).
Exact packaged kernel coverage remains a runtime check. GGUF acquisition retains
`auto`, an explicit CPU, a full NVIDIA UUID, or one `native:DEVICE` selector;
multiple-device assignments are configured after registration.

## Reuse provenance

Behavioral reference was read from:

- HM2-UI / HiveMind UI model-hub mappers, MIT, copyright HiveMind contributors;
- HiveMind model-pin/scanner/runtime lifecycle sources, Apache-2.0 with
  repository NOTICE/third-party terms.

Titanos adapts the provider record, Sherpa release mapping, progress, ownership
and provenance ideas into its existing standalone engine. It does not import
HiveMind services, credentials, Warden/fleet state, or unsafe fallback behavior.
The referenced UI test-only HF resolver implementation is absent from that
checkout and is not claimed or copied. `llama_cpp`/`llamacpp` source IDs are
normalized to Titanos's `llama-cpp` only after task/backend validation.
