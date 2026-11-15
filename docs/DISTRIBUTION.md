# Standalone executable distribution

No compiled bundle is published from this repository. The GitHub release workflow attaches a wheel and a source
distribution, not a compiled application (see [Releasing](RELEASING.md)). This page describes how to build and
validate a portable application directory yourself with `tools/build_binary.py`.

**Where distribution stands.** `0.5.28` is the first tagged release. A release carries the wheel, the source
distribution, the install catalog (`release.json`, `install.py`, `native-profiles.json`) and `SHA256SUMS`
([RELEASING.md](RELEASING.md)). Compiled application directories built by `tools/build_binary.py` are recorded with
their evidence in the [capability matrix](CAPABILITY_MATRIX.md); their notices and the FFmpeg inventory are described
in [FFmpeg notices](#ffmpeg-notices-and-inventory) below.

The binary distribution is a portable application directory containing the
Titanos executable, embedded CPython, the selected inference frameworks, their
native libraries and the MCP SDK. Users extract the directory and run Titanos;
they do not install Python, pip or a numerical environment. Keep the executable
and its `_internal` directory together.

Model weights remain separate. Register an existing checkpoint directory or use
an explicit model download. A model is not embedded into the application and
changing its registration does not require recompiling Titanos.

## Build on the target platform

Create a clean development environment with the numerical runtime suitable for
the target hardware, then install this source and its build tools. For example,
after installing the intended PyTorch distribution:

```sh
python -m pip install -e ".[models,build]"
python tools/build_binary.py --profile full --ffmpeg-dir /path/to/native-ffmpeg --output /path/to/Titanos --work-dir /path/to/build-work --plan
python tools/build_binary.py --profile full --ffmpeg-dir /path/to/native-ffmpeg --output /path/to/Titanos --work-dir /path/to/build-work --zip
```

Use native paths on Windows. The builder refuses to overwrite an existing
application directory or ZIP. PyInstaller is pinned to 6.22.2. The `full` profile
requires the model-framework dependencies and collects additional supported
ONNX/OpenVINO/XTTS runtimes only when installed in the build environment.

The [CPU text/vision profile](BUILD_TEXT_VISION.md) provides a separate lean
PyTorch/Pillow bundle for slow-processor TrOCR and YOLOS checkpoints, with full
MCP and external models. Its isolated environment leaves text/audio unchanged.

The `engine` profile packages HTTP, CLI and MCP without PyTorch/Diffusers. It is
useful for a separate native-runtime distribution; it is not the full multimodal
PyTorch build. The application reports its included packages in `BUILD.json`.

Build Windows binaries on Windows, Linux binaries on Linux and macOS binaries
on macOS, with a native interpreter matching the target architecture. The script
is shared; the resulting executable and libraries are platform-specific.
PyInstaller does not turn a Windows build into a qualified Linux/Apple build.

`--ffmpeg-dir` selects a source-bound native FFmpeg build, including its executable,
required libraries, `BUILD.json` and matching source archive. The binary builder
checks their hashes, includes the source/notices, and omits the unrelated FFmpeg
binary supplied by the imageio-ffmpeg wheel. `tools/build_ffmpeg.py` builds the
minimal Windows LGPL configuration using existing MSYS2/MinGW tools;
`tools/build_ffmpeg_macos.sh` supplies the native Apple recipe. The bundle's
`BUILD.json` binds its binaries either with a `binaries` mapping (name to sha256,
the Windows and macOS bundles) or with a `members` inventory of every file (bytes
and sha256, the neutral Linux bundle of 2026-10-05); with only `members`, the
builder derives the binaries from the ELF, PE or Mach-O members by their magic
bytes and records `binaries_derived_from_members` in `NATIVE_COMPONENTS.json`,
a `members` inventory must agree with the files and list every one of them, and
a bundle with neither is refused with a typed message. See
[FFmpeg notices and inventory](#ffmpeg-notices-and-inventory) for what the build
then writes about the bundle.

The Windows configuration uses the operating system's Media Foundation H264
encoder; the Apple configuration uses VideoToolbox. No GPL codec library or
external media library is enabled in these FFmpeg recipes. A full ZIP requires
the matching FFmpeg source bundle. The original wheel binary remains useful for
local development candidates, with its separate license/source obligations.

## What the package contains

| Artifact | Purpose |
|---|---|
| `titanos.exe` or `titanos` and `_internal/` | Executable, embedded interpreter, application and numerical libraries |
| `_internal/titanos/web/swagger-ui.css`, `_internal/titanos/web/swagger-ui-bundle.js` | Packaged Swagger UI (`swagger-ui-dist` 5.33.0, Apache-2.0) behind `/docs`, so the documentation page requests nothing outside the server's own loopback origin (shown in real Chrome with every non-loopback host unresolvable); `THIRD_PARTY_LICENSES/swagger-ui-dist/` retains its LICENSE and NOTICE, `tools/licenses/vendored_assets.json` pins the upstream tarball and hashes, and `DEPENDENCIES.json` lists it under `vendored_web_assets`. ReDoc is not shipped |
| `_internal/titanos/web/fonts/` | Studio's display face, Chakra Petch 1.000 (SemiBold and Bold Italic Latin subsets as WOFF2, SIL OFL 1.1) with its `OFL.txt`; `style.css` inlines both faces and carries the licence text, so nothing is served from this directory. `tools/licenses/vendored_assets.json` binds the files by hash, `THIRD_PARTY_LICENSES/chakra-petch/` retains the OFL and `DEPENDENCIES.json` lists it under `vendored_web_assets` |
| `SOURCE/` | Titanos source, build tooling and runnable checks (the paths `tools/public_export.json` excludes are pruned from it and from the sdist; see [Public export hygiene](#public-export-hygiene)) |
| `LICENSE`, `NOTICE` | Inherited Titanos license and attribution |
| `THIRD_PARTY_LICENSES/`, `THIRD_PARTY_NOTICES.md` | Available dependency notices and redistribution review boundaries; with `--ffmpeg-dir`, `THIRD_PARTY_LICENSES/ffmpeg/` retains the bundle's `COPYING.LGPLv2.1` and `LICENSE.md` and the notices carry an FFmpeg section (see [FFmpeg notices and inventory](#ffmpeg-notices-and-inventory)) |
| `DEPENDENCIES.json`, `sbom.cdx.json` | Selected runtime dependency versions, license metadata and CycloneDX inventory |
| `NATIVE_COMPONENTS.json` | Native media/source provenance and separate GPU-runtime licensing boundaries; the FFmpeg row records the binaries and every member of `_internal/native/ffmpeg/` with the hash it ships with (the bundle's hash under `normalized_from` for the prefix-normalized text members), the retained licence copies and the corresponding source |
| `BUILD.json` | Runtime profile, target platform and source-file hashes |
| `BUILD_SMOKE.json` | Path-free summary of the explicitly selected validation mode; raw operational receipts stay outside the package |
| `BUILD_HYGIENE.json` | Path-free summary of the whole-distribution hygiene scan that ran on the unsealed directory (files and archive members scanned, classes applied, zero refused hits, counts of reported third-party residue by class, and the UTF-16LE share of that residue under `utf16le`); the full report with file names and offsets stays outside the package (see [Distribution hygiene scan](#distribution-hygiene-scan)) |
| `MANIFEST.json` | Delivered file sizes and SHA-256 hashes |

The build collects original Python source and distribution metadata because
Transformers/Diffusers use lazy imports and PyTorch can inspect source for
TorchScript. The entry point runs multiprocessing freeze support before CLI
parsing. Native-runtime supervision and device probes re-enter only Titanos's
allowlisted helper modules through the bundled executable.

Models, user registries, model caches and credentials are excluded from the
build inputs. GPU drivers remain operating-system prerequisites. A CUDA build's
bundled numerical libraries and compatible driver requirements are distinct
from a CPU build or an Apple/ROCm build.

## Acceptance

By default (`--validation full`), the builder starts the executable from a fresh temporary data directory with
ambient `PYTHONPATH` and `PYTHONHOME` removed. It checks `--version`, the
engine doctor, registry creation, a live server and MCP discovery/calls over HTTP
and stdio. Readiness is ownership-first: the launched child must be alive, must
still be the launched process, and must be the exact owner of the `127.0.0.1`
listener on the selected port before any HTTP request is sent. The same check is
repeated immediately after the `/healthz` response and again before registry
controls, so an exited, replaced or foreign listener never satisfies readiness by
answering 200. The process/listener table comes from `psutil`, declared in the
`build` extra; when the system table is unreadable (non-root macOS) the child's
own table is used and that narrower scope is recorded in the receipts. A successful build checks application startup and protocol integration;
real model inference, generated asset quality and accelerator execution require the separate runtime
qualification checks.

MCP discovery is compared against the source, not a count: `tools/build_binary.MCP_TOOL_SOURCES` names every module whose
`@mcp.tool` declarations `mcp_server.create_mcp` registers (`mcp_server.py`, `runtime_api.py`, `assignment_api.py` and
`broker/mcp_tools.py`), both validation modes expect exactly the union of their declarations from the frozen server's
`list_tools`, and `tests/check_offline_binary.py` asserts that union against the in-process `create_mcp` registration, so a
module added to `create_mcp` without a line in the tuple fails the check before it fails a build. The 2026-10-05 Windows and
Linux candidates of commit `fd5638a` were refused at this gate ("Offline stdio MCP declarations differ from source") because
the lists then named three modules, 60 of the 72 registered tools.

After collection, the builder reconciles every shipped `dist-info` METADATA with
`DEPENDENCIES.json` and `sbom.cdx.json` and re-validates the sealed directory
(`validate_packaged_distributions`). Optional imports that PyInstaller collected
outside the declared requirement closure are added to the inventory rather than
omitted. A native or embedded component may share a distribution's name (the
optional Sherpa runtime ships a PyPI wheel and a separately attributed native
library); only components with a `pkg:pypi` purl or a bare name/version establish
the Python distribution identity, separately attributed components are preserved
and reported, and duplicate or conflicting Python identities are refused. The same
validation re-checks every FFmpeg row of `NATIVE_COMPONENTS.json` against the
shipped bytes (below) and refuses a packaged `_internal/native/ffmpeg/` that no row
binds, so a package whose notices or inventory fall short of what the builder now
writes is refused at the end of `write_notices`, after the smoke and by
`tools/compose_binary.py` alike.

### FFmpeg notices and inventory

When a build includes `--ffmpeg-dir`, `write_notices` (`include_ffmpeg_component`) writes what the LGPL asks a distributor
to say about the shipped FFmpeg, and `validate_packaged_distributions` re-checks it:

- `THIRD_PARTY_NOTICES.md` gains a `## FFmpeg` section: the FFmpeg version (the bundle's `ffmpeg_version` and the
  corresponding-source archive name must agree, and the packaged `version.txt` must announce the same version), the
  licence expression (`LGPL-2.1-or-later`), that the configuration excludes GPL and nonfree components (`--disable-gpl` and
  `--disable-nonfree` must be present, no `--enable-gpl`/`--enable-nonfree`, no `gpl`/`nonfree` truth and no external codec
  library, else the build is refused), where the licence texts live, the complete corresponding source shipped in the
  distribution (`_internal/native/ffmpeg/source/ffmpeg-<version>.tar.*` with its sha256 and byte count, every build script
  beside it with its sha256; a bundle without a build script is refused) and the written offer of LGPL-2.1 section 6. The
  section describes shipped bytes and the recorded configuration; it is not a legal review.
- `THIRD_PARTY_LICENSES/ffmpeg/` receives the bundle's `COPYING.LGPLv2.1` (checked against the archive's copy) and
  `LICENSE.md`, listed with their hashes under the row's `license_files`.
- The row and the packaged `_internal/native/ffmpeg/BUILD.json` carry the same record: `binaries` as packaged (identical
  bytes, or the verified Mach-O postprocessing `tools/native_inventory.py` describes), `members` with the size and hash of
  every bundle file **as shipped**, taken after the configure `--prefix` was rewritten to `<install-prefix>` in `version.txt`,
  `license-output.txt` and `buildconf.txt`, each rewritten member keeping the bundle's hash under `normalized_from` (a
  postprocessed binary keeps it under `source_build_sha256`); any other member that changed between the bundle and the
  packaged copy, a missing member or an extra file refuses the build. `binaries_derived_from_members` says whether the
  binaries came from the bundle's `binaries` mapping or were derived from its `members`.
- The verifier refuses a row without `binaries`, `members` or `license_files`, a member or binary whose shipped bytes differ
  from the row, a packaged directory that holds files the members do not list, a packaged `BUILD.json` that disagrees with the
  row, a missing or changed retained licence text, and notices without exactly one `## FFmpeg` heading naming the licence and
  the shipped archive. That is the shape the 2026-10-06 builds of `5aa3311` shipped (both: no section and no retained texts;
  Linux: `version.txt` and `license-output.txt` recorded with their pre-normalization hashes), so such a package can no longer
  pass. `tests/check_ffmpeg_bundle.py` drives the real functions over inert bundles of all three `BUILD.json` shapes.

`--validation offline` completes compilation, source/dependency/license packaging,
local version/settings/empty-registry CLI checks, and stdio MCP initialization and
tool declarations. It never starts a Titanos HTTP listener or invokes MCP tools or
resources. Normal stdio event-loop IPC can include an internal OS socketpair; this
is not a claim that the process creates no sockets of any kind.

Offline artifacts are useful when network-serving validation is unavailable.
Their `BUILD.json` records `validation_mode=offline`; `BUILD_SMOKE.json` records
`passed=false`, `offline_passed=true`, and HTTP/registry/model/UI validation as
`not_performed`. Construction succeeds only after the selected offline checks
pass. This is not a successful full smoke or model/UI qualification. The full
mode remains the default, and verifiers must explicitly request offline scope
to accept its narrower evidence. `tools/compose_binary.py --validation offline`
uses the same checks and evidence contract when assembling verified inputs.

The offline option does not grant authority to retry a refused listener, alter
security settings, or move a service to another port. Serving and real inference
remain separate follow-on validation under the operator's current authority.

Build provenance is bound to the source hashes captured before compilation.
If application source changes while compiling, the candidate is retained in
the build work directory and must be rebuilt before publication.

Dependency metadata and copied license files are an inventory, not a blanket
legal approval. Titanos's Apache license does not replace dependency, GPU-runtime
or model licenses. Review missing notices and vendor redistribution terms before
publishing a binary. No publication, signing or installer registration occurs
as part of this local build script.

## Public export hygiene

`tools/public_export.json` is the one statement of what the exported source copies (the native `SOURCE/` directory written by `tools/build_binary.py source_package`, and the sdist selected by `MANIFEST.in`) leave out and how the rest is kept free of lab locators. `source_package` reads its `excluded_paths`; `MANIFEST.in` carries the same prune/exclude lines; `tests/check_public_export_hygiene.py` computes both sets with the real code, asserts that the build manifest covers exactly the exported set, that the two copies agree on the exclusions, and that no file in either set matches the policy's generic detection patterns (Windows user-profile paths and their MSYS, Cygwin and WSL spellings, agent work trees, RFC1918 IPv4 literals, drive-rooted lab records roots, Linux home directories, the lab's Linux hostname shape, drive-letter pairs, SSH host-key fingerprints, Windows generated machine names) outside the fixture allowlist, and that neither set names the exporting machine (its name is read from the environment at scan time, never from the repository).

- **Excluded paths:** `docs/evidence/**`, `docs/CLUSTER_HANDOFF.md`, `tools/matrix_kit/legacy_results.json`, `tools/validation/gen01` and `docs/validation` are pruned from every exported copy.
- **Parameterized (current, regenerable or product documents):** `docs/roster.json` names records as `<lab-records>/...` and hosts by stable labels, and `docs/ROSTER.md` is rendered from it; `tools/build_execution_matrix.py` writes every string of `docs/execution_matrix.json` through `tools/lab_paths.py` (`<lab-records>/...`, `<user-profile>/...`) and `tools/matrix_kit/ingest.py` emits display receipts as `<lab-records>/<runs>/<run>/<receipt>`, so the rendered `EXECUTION_MATRIX.md`, `.csv` and `.html` inherit it. Because the reviewed reader input of the execution matrix lives only in the lab run folders, the committed JSON was normalized once on 2026-09-26 with the same rule (a text normalization, not a rebuild; `built_utc` unchanged, recorded in its `basis`) and the renderings were regenerated. Prose in `docs/CAPABILITY_MATRIX.md`, `docs/MATRIX_KIT.md`, `docs/ONNX_IMAGES.md` and the `tools/matrix_kit/containment.py` comments cites record names without drive prefixes; `tests/check_matrix_kit.py` reads its pinned bundle folder from `TITANOS_KIT_PINNED_BUNDLE`.
- **Host labels:** a machine is named by a stable role label chosen at the source (in the kit run label, or in `docs/roster.json`), for example `turing-laptop-win-x64` or `intel-arc-win-x64-openvino`, never by its computer name or an address. `tools/lab_paths.py` reduces a generated Windows machine name that still reaches a generator string to `<windows-host>` as a last defence, and no name of that class may be allowlisted. What a host's operating system calls itself is a lab circumstance, not part of any claim. The labels are the keys of `tools/lab_paths.HOST_ROLE_LABELS`, documented under Host labels in [MATRIX_KIT.md](MATRIX_KIT.md#host-labels); the hygiene test fails when the two disagree. The map from the lab's own machine names to those labels is private (`tools/lab_hosts.private.json`, which the policy lists as never tracked and never exported, or the file named by `TITANOS_LAB_HOSTS_FILE`); `tools/lab_paths.py` applies it when the matrix is rebuilt, so the committed matrix carries labels, not machines.
- **Fixture allowlist:** `tests/check_model_page.py` keeps two synthetic RFC1918 literals (the URL policy under test refuses private addresses); `tests/check_music_failure_diagnostics.py` keeps a synthetic user-profile path (account `example`) for its redaction fixture. Each entry lists the exact literals and its reason in the policy file, and a hit is tolerated only when the matched text equals a listed literal in the listed file.
- **Tracked tree and tree kinds:** `tests/check_public_export_hygiene.py` scans both export copies and, outside the excluded records, every tracked file with the detection classes above and with the exporting machine's own name (read from the environment at scan time, never from the repository). The default tree kind holds the excluded paths; a public tree (`TITANOS_TREE_KIND=public`, or the marker file `.titanos-public-tree` at the repository root) must not hold them and is scanned entirely. Both export copies carry the marker when the tree they are made from has it (`MANIFEST.in` includes it; `source_package` copies it when present), so a `SOURCE/` copy or an sdist made from the public tree selects the public kind by itself.

A reader of the parameterized documents sees labels, not locations. External build or verification drivers that pin `SOURCE/` contents from a repository snapshot must apply the same `excluded_paths`, or they will report the excluded files as missing. This policy is a hygiene guard for lab locators only: it is not a secret scanner, and the legal and vendor review of the distribution (dependency notices, GPU-runtime redistributables, model licenses) remains unperformed.
The Docker image build context follows the same rule: `.dockerignore` excludes the four excluded records (`docs/evidence`,
`docs/CLUSTER_HANDOFF.md`, `docs/validation`, `tools/validation/gen01`), `tools/matrix_kit/legacy_results.json` and the private
host map `tools/lab_hosts.private.json`, so `COPY . /app` cannot pull the excluded paths into an image.

### Distribution hygiene scan

The source copy is not the only place a lab locator can reach a build. The 2026-10-05 candidates carried the lab's paths in
binaries and metadata that no `SOURCE/`-only scan looks at: `file:///<drive>:/<lab records>/...` URLs in four
`*.dist-info/direct_url.json` files of the bundled music runtime r4, the configure prefix `/home/<account>/...` inside the
Linux FFmpeg bundle (every file under `_internal/native/ffmpeg/` and the seven `libav*`/`libsw*` copies PyInstaller collected
beside it) and a `<drive>:/Users/<account>/Documents/<agent tree>/...` build path inside the Windows FFmpeg bundle's DLLs and
executables. `tools/build_binary.verify_distribution_hygiene` therefore scans **every file** of the unsealed directory after the
smoke receipts have left it and before `MANIFEST.json` seals it:

- every file is read as bytes, so UTF-8 text and the byte strings of binaries are searched alike (large files in overlapping
  chunks), in their 8-bit spelling and in UTF-16LE (the wide strings a Windows DLL carries, decoded around every wide anchor at
  its own alignment; such a hit carries `encoding: utf-16le` and no line, and `BUILD_HYGIENE.json` counts the UTF-16LE share of
  the reported residue under `utf16le`), and the members of stored or deflated zip archives (`.zip`, `.whl`, `.egg`, `.jar`;
  PyInstaller's `base_library.zip` included) are searched as well; UTF-16BE and UTF-32 strings, tarballs and the compressed
  PYZ inside the executable are outside its reach, and the text-only numeric and fingerprint classes run over decodable UTF-8
  text only. The 2026-10-06 verifier of the Windows candidate counted 1,647 UTF-16LE copies of an upstream conda builder's
  user profile in one collected DLL that the 8-bit scan of that day could not see; the scan now reports that kind of residue
  (a foreign account in third-party payload) and would refuse the same spelling of a lab locator or the building account;
- the classes are the policy's (`tools/public_export.json` `detection_patterns`, the fixture allowlist for `SOURCE/` files),
  including `posix_user_profile`, the MSYS, Cygwin and WSL spellings of a Windows user profile (a drive letter as a directory
  under `/`, `/cygdrive/` or `/mnt/`, then `Users/<account>`; the configure prefix the retained lab Windows FFmpeg bundle records
  in its `version.txt` and `buildconf.txt`, which the owned zone refuses by this class and by `lab_codex_worktree`, extended to the
  same spellings), plus a local `file:///` URL class, the exporting machine's names and the private host map's names (whole
  word), and the building account and `TITANOS_HYGIENE_DENY` entries as the account segment of a user-profile or home path; all
  names are read from the environment at scan time and never written anywhere;
- two zones: in the **owned** zone (the builder's records at the top level, `SOURCE/`, `_internal/titanos/`, the lab-built
  native bundles under `_internal/native/`, runtime inventories and every `direct_url.json`) every class refuses; in the
  **third-party** zone (collected dependency payload, retained upstream sources and notices, a relocated runtime interpreter)
  the lab shapes (drive-rooted Titanos or Codex trees, agent work trees, the lab's Linux hostname form) and the denied names
  refuse, while the generic shapes that upstream wheels legitimately carry (another project's `/home/<account>` example, a
  CI runner's `<drive>:/Users/<runner account>` build path, a contributor's generated machine name in an AUTHORS file, a
  cargo SBOM's local `file:` URL of a drive root) are reported with counts, not refused; the numeric shapes (RFC1918 addresses,
  drive-letter pairs) and the fingerprint class run over decodable text only because they match random bytes. This page and the
  tests describe the classes in words or build their planted examples from parts, never as a contiguous literal the classes
  would match, because `SOURCE/` carries the docs and tests and a self-matching spelling would refuse every build at this gate;
- a refused hit fails the build with a per-file report (file, zone, class, count and the first line or byte offset; never the
  matched text); the full report is retained beside the private smoke receipts and `BUILD_HYGIENE.json` carries the path-free
  summary. `tests/check_public_export_hygiene.py` (`DistributionHygiene`) exercises the scan on a fake distribution with
  planted hits in text, binaries and a zip member, and asserts that `main()` runs it between the smoke publication and the
  manifest.

What this means for the retained inputs: **music runtime r4 predates the exclusion of pip provenance** (`tools/music_package.py`
now leaves `*.dist-info/direct_url.json` out of a packaged runtime, records the excluded names under `exclusions.pip_provenance`
in `runtime.json` and `DEPENDENCIES.json`, and `verify_runtime` refuses a runtime that still carries one), so r4 is refused at
planning and **must be rebuilt as r5** before a sealed candidate; and the **retained FFmpeg bundles (the 0.5.27 Linux build and
the Windows build) embed their build paths**, so both must be rebuilt with a neutral `--prefix` (and a neutral build directory,
or `-ffile-prefix-map`) before they can pass. The scan over the retained 2026-10-05 directories refused exactly those files
(Linux: 30 files, all FFmpeg; Windows: the 16 FFmpeg DLLs and executables, the four runtime `direct_url.json` files and the raw
`binary-smoke.json` that a completed build moves out) and reported, without refusing, the third-party residue of both. Both
were then done: runtime r5 and neutral FFmpeg bundles (`--prefix=/opt/titanos/ffmpeg`; Linux on 2026-10-05, Windows on
2026-10-06) went into the 2026-10-06 candidates of `5aa3311`, which passed the gate with zero refused hits. The retained lab
Windows bundle's `version.txt` and `buildconf.txt` still spell their prefix as an MSYS path under a user profile, which the
`posix_user_profile` class now refuses in the owned zone, so that bundle cannot pass the gate either.

## Install path length

Windows loads executables and DLLs through paths bounded by MAX_PATH (259 characters), even where the file system and a
long-path-aware interpreter can address longer names. The application directory name is about 64 characters and the deepest
packaged file lies about 166 characters below it (PyTorch headers), so tools bound by MAX_PATH (Windows Explorer extraction,
some installers) need a parent directory of at most about 28 characters to place every file; read `MANIFEST.json` for the exact
deepest path of a given build. What matters at run time is the loader: the optional music runtime keeps its torch DLLs about 95
characters below the application directory. Observed on one workstation: the runtime imported PyTorch with its torch/lib directory
path at 165 characters and failed with WinError 206 at 218 characters (the named DLL path was 252 characters, still under 259, so
the bound is inferred to come from the loader resolving dependency paths, not from the named file alone); treat a parent directory
under roughly 99 characters as the inferred safe budget, not a measured limit. The server checks the portable music runtime's deepest executable, DLL
and extension paths before launching any child and refuses with the exact directory budget when the bound would be exceeded;
the main application's own numerical libraries sit shallower (about 58 characters below the directory). Install under a short
path such as a drive-root folder when in doubt.

## Attribution supplements

`tools/licenses/supplements.json` lists retained attribution files for packaged distributions whose upstream notices are known to be incomplete or inconsistent (currently ONNX 1.22.0, whose NOTICE labels the vendored protobuf 25.1 component Apache-2.0 while its own SBOM says BSD-3-Clause). During `write_notices` the build hash-checks each retained file, copies it under `THIRD_PARTY_LICENSES/<name>/` or `THIRD_PARTY_SOURCES/<name>/`, records it in `DEPENDENCIES.json` (`license_files`, `source_attribution`, `upstream_sbom`, `attribution_note`), nests the packaged upstream SBOM components under the shipped component in `sbom.cdx.json`, and appends a section to `THIRD_PARTY_NOTICES.md`. Upstream files are never rewritten; this is attribution, not legal approval. Sealed older packages keep their separate metadata amendments.
