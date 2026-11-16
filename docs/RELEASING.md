# Releasing Titanos

Titanos is not published on PyPI. A release is a GitHub Release built by `.github/workflows/release.yml` from an existing tag. It carries a wheel, a source distribution, a release catalog and checksums. The same files can be built locally with the commands below.

## Version and tag rule

| Rule | Where it is enforced |
|---|---|
| The release version is plain `X.Y.Z` (for example `0.5.28`), with no `.dev`, local `+label` or `v` inside it. | `install.py` (a copy of `src/titanos/hardware.py`) refuses a release catalog whose `version` does not match `\d+\.\d+\.\d+` with `Invalid release version`. `tools/build_release.py` takes that version from the wheel file name, so a `.dev0+label` version builds a catalog that `install.py` and `titanos upgrade` reject. |
| `pyproject.toml` `version` and `src/titanos/__init__.py` `__version__` carry the same string. | The wheel and sdist take the first; `titanos --version` and the engine doctor's `engine_version` take the second. |
| The tag is `vX.Y.Z` for version `X.Y.Z`. | Convention only. `release.yml` checks out the tag you name and titles the release `Titanos <tag>`; it does not compare the tag with the package version. |
| The tag exists on GitHub before the workflow runs. | `gh release create --verify-tag` fails otherwise. |

The first release is `0.5.28`, tagged `v0.5.28`; `docs/RELEASE_NOTES.md` lists each release, newest first, above the retained `0.4.0` note.

## Steps

1. Set the version in `pyproject.toml` and `src/titanos/__init__.py`. Update `docs/RELEASE_NOTES.md` for this version: the workflow attaches that file as the release notes, as committed at the tag.
2. Run the checks locally (see [Local reproduction](#local-reproduction)) and the generators' `--check` commands (`python tools/render_roster.py --check`, `python tools/render_progress.py --check`, `python tools/render_execution_matrix.py --check`, `python tools/render_protocol_deviations.py --check`, `python -B tests/check_route_contract_matrix.py`). Commit.
3. Tag and push the tag:

   ```sh
   git tag -a vX.Y.Z -m "Titanos X.Y.Z"
   git push origin vX.Y.Z
   ```

4. In GitHub Actions open **Build draft release** (`release.yml`), choose **Run workflow** and enter the tag. `release.yml` runs only when dispatched; pushing a tag does not build a release by itself.
5. Download `SHA256SUMS` and the wheel from the draft, verify the checksums, install the wheel into a fresh virtual environment and run `titanos --version` and `titanos doctor --check`.
6. Publish the draft when the checks and the notes are right. Until then the draft is not visible to the public.

## What release.yml does

One job, `package`, on `ubuntu-latest` with Python 3.12:

| Step | Command |
|---|---|
| check out the tag | `actions/checkout@v4` with `ref: ${{ inputs.tag }}` |
| install build tools and the package | `python -m pip install 'setuptools>=78' wheel build`, then `python -m pip install .` |
| contract checks | `python tests/check.py`, `python tests/check_hub.py`, `python tests/check_jobs.py` |
| public-export gates | `python -B tests/check_source_package.py`, `python -B tests/check_public_export_hygiene.py` |
| build | `python tools/build_release.py`, then `python -m build --sdist` |
| attach | `gh release create "$RELEASE_TAG" --verify-tag --draft --title "Titanos $RELEASE_TAG" --notes-file docs/RELEASE_NOTES.md release/* dist/*` |

`tests/check_release_workflow_gates.py` pins this shape: exactly one job named `package`; each gate exactly once, spelled with `-B`; `setuptools` installed before the gates; gates before `build_release.py`, `build_release.py` before the sdist, the sdist before `gh release create`; the attach step last, still naming `release/*` and `dist/*`. The checks workflow keeps running both gates as well.

## What a release contains

| File | Produced by | Purpose |
|---|---|---|
| `titanos_engine-<version>-py3-none-any.whl` | `tools/build_release.py` (`pip wheel --no-deps`) | the installable engine |
| `release.json` | `tools/build_release.py` | catalog: `schema_version` 1, the version, the wheel name and sha256, the numerical profiles from `tools/profiles.json`; `python_lock`: the CPython build per platform (URL, sha256, size, `python_bin`), each lock and wheel (file, sha256, size) and the indexes |
| `install.py` | copy of `src/titanos/hardware.py` | installs an isolated runtime from the catalog; `--plan` prints the chosen profile without installing |
| `native-profiles.json` | copy of `tools/native-profiles.json` | the pinned native (llama.cpp) runtime catalog |
| `qwen_tts-<version>-py3-none-any.whl` | `tools/build_release.py` (`build_python_lock.build_source_wheels`) | the speech wrapper `tools/python-lock.json` names, built from a git archive of its pinned commit (LF endings, `SOURCE_DATE_EPOCH` at the commit time); no index carries it |
| `python-lock-win_amd64.txt`, `python-lock-linux_x86_64.txt` | copies of `tools/locks/` (`tools/build_python_lock.py`) | one hash lock per platform for a pinned-interpreter install (below) |
| `SHA256SUMS` | `tools/build_release.py` | checksums of the files above, `sha256sum -c` format |
| `titanos_engine-<version>.tar.gz` | `python -m build --sdist` through `MANIFEST.in` | the source distribution; lab records are excluded by the same rules as the public tree |

Not attached: compiled application bundles (`tools/build_binary.py`, see [Standalone executable distribution](DISTRIBUTION.md)), model weights, native runtime binaries.

## Installing from a release

Install the numerical runtime for your hardware first (see [Hardware](HARDWARE.md)), then the wheel, with the extras you need:

```sh
sha256sum -c SHA256SUMS
python3 -m venv .venv && . .venv/bin/activate
python -m pip install "titanos_engine-<version>-py3-none-any.whl[models]"
titanos --version
```

On Windows PowerShell, `Get-FileHash titanos_engine-<version>-py3-none-any.whl` prints the hash to compare by eye, and the environment is activated with `.venv\Scripts\Activate.ps1`.

The catalog path installs an isolated runtime and chooses a profile from the detected hardware:

```sh
python install.py --catalog release.json --plan
titanos upgrade --catalog release.json --plan
```

Drop `--plan` to install; `--profile NAME`, `--gpu INDEX` and `--target DIR` override the selection. `install.py` also accepts `--catalog-sha256` for an out-of-band catalog checksum.

## Pinned-interpreter install

An installer that runs no script of ours can install a release from pinned files alone, per platform (Windows x64 and Linux x64):

1. Unpack the CPython build `release.json` names in `python_lock.platforms.<platform>.interpreter` (an official standalone build, checked by sha256 and size) into `<install>/python`; the interpreter is `<install>/python/<python_bin>`, the file itself (`python/python.exe`, or `python/bin/python3.12` on Linux, where `python/bin/python3` is a link to it).
2. `<interpreter> -I -m venv <install>/env`.
3. Install the release's wheels offline: `<env python> -I -m pip install --isolated --no-input --disable-pip-version-check --require-hashes --no-deps --only-binary=:all: --no-index -r wheels.txt`, where each line of `wheels.txt` is a wheel's path and `--hash=sha256:<its sha256>`.
4. Install the platform's lock with the same options from the indexes (`--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 -r python-lock-<platform>.txt` in place of `--no-index`).
5. Run `<install>/env/Scripts/titanos.exe` (Windows) or `<install>/env/bin/titanos`. The native runtime catalog ships inside the wheel (`titanos/data/native-profiles.json`), so nothing else is placed.

Each lock line is `name==version` with the sha256 of every wheel of that version for the platform on those indexes: no environment markers, URLs, paths, options or repeated names (`python -B tests/check_python_lock.py`). A lock holds `pyproject.toml`'s dependencies with the `models` and `qwen3-tts` extras, PyTorch 2.8.0 for CUDA 12.8 and what the speech wrapper needs, except `gradio`, which only its demo imports. With the CUDA build of PyTorch in both locks, a machine with an AMD or Intel GPU runs the PyTorch backends on the processor for now; the native runtimes keep their GPU builds. `python tools/build_python_lock.py --uv <uv>` rebuilds the locks. Model weights stay separate downloads.

## Local reproduction

```sh
python -m pip install "setuptools>=78" wheel build .
python -B tests/check_source_package.py
python -B tests/check_public_export_hygiene.py
python tools/build_release.py
python -m build --sdist
```

`tools/build_release.py` writes into `release/` at the repository root and catalogs the newest `titanos_engine-*.whl` it finds there, so remove an old `release/` directory first. `release/` and `dist/` are ignored by git.

## What a release does not claim

- A green `release.yml` or checks run is a contract and packaging result over small or mocked fixtures. It is not model, accelerator or modality qualification; those are recorded per row in [the capability matrix](CAPABILITY_MATRIX.md).
- Nothing is published to PyPI. Install from the release files or from source.
- The release carries no compiled application bundle and no model weights.
- Dependency notices are an inventory, not a legal review: the builders record `license_review: not_performed` ([dependency boundaries](DEPENDENCY_PROVENANCE.md)).
- Creating a draft is not publishing it. Publishing the draft, and changing the repository's visibility, are manual decisions.
- No GitHub run of `release.yml` is recorded in this repository as of 26 Sep 2026 (`docs/RELEASE_NOTES.md`). The steps above describe what the workflow file does, not a run that happened.
