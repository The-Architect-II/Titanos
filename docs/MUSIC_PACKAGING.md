# Portable music runtime construction

`tools/music_package.py` builds the optional external ACE runtime from already
owned, trusted inputs. It copies files and writes inventories; it never invokes
pip, a shell, the supplied interpreter, the adapter or a model. Weights remain in
the separately acquired music bundle. The adapter's own notes are in [tools/music_runtime/README.md](../tools/music_runtime/README.md).

The output layout is:

```text
Titanos/
  titanos[.exe]
  runtimes/ace-step-1.5/
    runtime.json
    python/                 # self-contained interpreter prefix
    adapter/ace_adapter.py  # trusted one-shot entry, not checkpoint Python
    SOURCE/ace-step-1.5/
    licenses/
    DEPENDENCIES.json
    SOURCES.json
    LICENSES.json
```

The backend discovers the sibling `runtime.json` when no explicit installed
receipt overrides it. The tool generates that receipt; users do not write it or
install Python. It contains relative `executable` and `adapter` paths, the exact
ACE code revision, and every payload file's size/SHA256. The manifest excludes
itself to avoid a hash cycle; the outer Titanos manifest must bind it when the
complete application is sealed. Inventory files are themselves hash-bound.

Example **construction command**, not a completed native artifact:

```sh
python tools/music_package.py \
  --interpreter /owned/music/python \
  --site-packages /owned/music/env/lib/python3.11/site-packages \
  --executable bin/python3.11 \
  --site-relative lib/python3.11/site-packages \
  --adapter /owned/adapter-staged/ace_adapter.py \
  --source /owned/music/official-source \
  --source-pin /owned/music/SOURCE_PIN.json \
  --notices /owned/music/licenses \
  --portable-root /owned/new-Titanos-staging
```

`--source-pin` uses the retained official-source inventory: `repository`,
`commit` and `files` mapping relative source paths to `bytes` and `sha256`.
The fixed source revision is `ca1e85fe9430179831e6bc6be790c332190a3866`.
The complete source roster and bytes must match. The actual runner and device
contract are separate implementation inputs; providing a file to this tool does
not qualify its behavior.

Use `--output` for an independent runtime directory. `--portable-root` writes
`runtimes/ace-step-1.5` under an **unsealed** application staging directory and
refuses an existing outer `MANIFEST.json`. For the normal complete application
build, construct a separate runtime with `--output`, then pass that directory
to `tools/build_binary.py --music-runtime-dir <runtime-directory>` together with
the usual platform/profile/output arguments. The builder verifies and copies it
before generating the outer file inventory and ZIP, and includes its dependency
and source/license inventories in the package metadata. It does not modify an
existing working package or silently reseal one.

## Relocation and input boundaries

The prepared environment has absolute `pyvenv.cfg`/launcher paths. Copying that
venv unchanged would not make it portable. This tool instead copies the owned
self-contained Python prefix, omits its old site-packages, and fills that relative
site directory from the selected environment. It does not copy venv activation
scripts, venv `bin` wrappers or `pyvenv.cfg`. For embedded Windows Python, a relative
site-packages entry is added to `._pth`. No setup command is executed.

Escaping/cyclic links, unsupported reparse points, missing referenced files,
editable installs and absolute `.pth` dependency paths fail construction.
Internal input links are materialized; output verification permits no links.
Derived bytecode caches are omitted. Conventional model formats (`.safetensors`,
`.gguf`, `.onnx`, `.ckpt`, `.pt`) are refused, and no checkpoint-root input exists.
Curated runtime/source directories are still required: filename filtering is not
a classifier for arbitrary disguised data.

Dependency startup imports in trusted `.pth` files remain hash-bound and are
listed in the layout inventory; the packager does not execute them. Native
libraries can also contain platform-specific loader paths. Content hashes and
relative layout do not establish native loadability or relocatability of every
third-party binary.

Construction uses a private adjacent staging directory, validates the complete
payload, then publishes with an atomic no-replace directory rename. Windows and
Linux publishers are supported; unsupported atomic publication fails closed.
An existing target is preserved, including one appearing during publication.

Verify a constructed or moved tree without executing it:

```sh
python tools/music_package.py --verify /owned/new-Titanos-staging/runtimes/ace-step-1.5
```

## Qualification still required

Tiny local fixture checks cover moved-directory receipts, complete hashes,
missing/tampered/extra payloads, pinned source changes, path controls, omitted
weights/venv configuration and atomic target preservation. They use fake
interpreter/adapter bytes and do not execute a model stack. They are construction
proof only.

The real target bundle still needs an authorized artifact build from its actual
interpreter and trusted runner, moved-prefix imports/native dependency checks,
strict observed device assignment, offline loading of the complete external
checkpoints, generation and channel/duration checks, cancellation/unload/owned
closure, and listening review. The source tool and receipt do not authorize or
claim those actions. Keep component licenses/source notices distinct from
Titanos's license and model-card declarations.
