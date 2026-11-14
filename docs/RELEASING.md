# Releasing Titanos

Titanos is not published on PyPI. A release is a GitHub Release built by `.github/workflows/release.yml` from an existing tag. It carries a wheel, a source distribution, a release catalog and checksums. The same files can be built locally with the commands below.

Publishing the repository itself is a separate step with its own procedure: see [Public branch](#public-branch).

## Version and tag rule

| Rule | Where it is enforced |
|---|---|
| The release version is plain `X.Y.Z` (for example `0.5.28`), with no `.dev`, local `+label` or `v` inside it. | `install.py` (a copy of `src/titanos/hardware.py`) refuses a release catalog whose `version` does not match `\d+\.\d+\.\d+` with `Invalid release version`. `tools/build_release.py` takes that version from the wheel file name, so a `.dev0+label` version builds a catalog that `install.py` and `titanos upgrade` reject. |
| `pyproject.toml` `version` and `src/titanos/__init__.py` `__version__` carry the same string. | The wheel and sdist take the first; `titanos --version` and the engine doctor's `engine_version` take the second. |
| The tag is `vX.Y.Z` for version `X.Y.Z`. | Convention only. `release.yml` checks out the tag you name and titles the release `Titanos <tag>`; it does not compare the tag with the package version. |
| The tag exists on GitHub before the workflow runs. | `gh release create --verify-tag` fails otherwise. |

As of 26 Sep 2026 the repository has no tag and no release, the version is `0.5.28.dev0+accelerators1`, and `docs/RELEASE_NOTES.md` opens with an Unreleased entry for that line above the retained `0.4.0` note, its newest released entry; the first release renames that entry to its `X.Y.Z`.

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
| `release.json` | `tools/build_release.py` | catalog: `schema_version` 1, the version, the wheel name and sha256, the numerical profiles from `tools/profiles.json` |
| `install.py` | copy of `src/titanos/hardware.py` | installs an isolated runtime from the catalog; `--plan` prints the chosen profile without installing |
| `native-profiles.json` | copy of `tools/native-profiles.json` | the pinned native (llama.cpp) runtime catalog |
| `SHA256SUMS` | `tools/build_release.py` | checksums of the four files above, `sha256sum -c` format |
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

## Local reproduction

```sh
python -m pip install "setuptools>=78" wheel build .
python -B tests/check_source_package.py
python -B tests/check_public_export_hygiene.py
python tools/build_release.py
python -m build --sdist
```

`tools/build_release.py` writes into `release/` at the repository root and catalogs the newest `titanos_engine-*.whl` it finds there, so remove an old `release/` directory first. `release/` and `dist/` are ignored by git.

## Public branch

The private repository's history carries the lab records that `tools/public_export.json` excludes, so the public repository is a fresh-history branch, not this repository made visible. `tools/build_public_branch.py` builds it; [Public branch](DISTRIBUTION.md#public-branch) in the distribution notes describes what it drops, changes and checks.

The builder needs git 2.42 or newer for `git worktree add --orphan`; with an older git (2.23 or newer) it falls back to `git worktree add --detach` followed by `git switch --orphan`, which leaves the same empty worktree, and records which form it used in its report. Its scan applies the policy's detection classes, the exporting machine's own name and, when the private host map `tools/lab_hosts.private.json` (or the file `TITANOS_LAB_HOSTS_FILE` names) is present, the names that map denies, through `tools/lab_paths.private_names()`: the same class `private_denylist` that `tests/check_public_export_hygiene.py` builds. The map is never tracked and there is no other name list to keep.

Before the flip:

1. On `main`, with a clean working tree and the checks green, preview the build. The scan must report zero hits; read the dropped list and the per-file count of references into dropped records (they dangle by design).

   ```sh
   python tools/build_public_branch.py --dry-run --report ../public-branch-report.json
   ```

2. Build the branch into a new directory outside the repository. The builder scans every file, runs `tests/check_source_package.py`, `tests/check_public_export_hygiene.py` and `tests/check_release_workflow_gates.py` inside the new worktree and commits only when all of them pass.

   ```sh
   python tools/build_public_branch.py --worktree ../titanos-public --branch public
   ```

3. In the worktree, confirm the shape and run the remaining checks you rely on (`python -B tools/run_checks.py --sync-check`, the first step of the contract job, then `python tests/check.py`, the generators' `--check` commands, `python -B tests/check_route_contract_matrix.py`):

   ```sh
   git -C ../titanos-public rev-list --count HEAD
   git -C ../titanos-public ls-files docs/evidence docs/CLUSTER_HANDOFF.md tools/validation/gen01 tools/matrix_kit/legacy_results.json
   ```

   The first prints `1`; the second prints nothing.

4. Read `README.md`, `docs/PUBLICATION_NOTES.md`, `SECURITY.md` and `CONTRIBUTING.md` in the worktree as a stranger would: they must not describe a private repository, and every command must work without the dropped files.
5. Create a new, empty GitHub repository for the public source. Push the branch there as `main`. Never push the private `main`, and never force-push over the private repository: forks, pull-request refs and caches keep old commits reachable, so an in-place rewrite is not a removal.

   ```sh
   git -C ../titanos-public push <empty public repository url> public:main
   ```

6. In that repository's settings, enable private vulnerability reporting if `SECURITY.md` points to it, then change the visibility. The private repository stays private and unchanged.

To discard a built branch: `git worktree remove --force ../titanos-public` and `git branch -D public` in the private repository.

The builder creates a fresh root commit every time. Publishing a later update without rewriting the public history is a manual step: build the new tree, copy it over a checkout of the public repository, review the diff, commit there.

## What a release does not claim

- A green `release.yml` or checks run is a contract and packaging result over small or mocked fixtures. It is not model, accelerator or modality qualification; those are recorded per row in [the capability matrix](CAPABILITY_MATRIX.md).
- Nothing is published to PyPI. Install from the release files or from source.
- The release carries no compiled application bundle and no model weights.
- Dependency notices are an inventory, not a legal review: the builders record `license_review: not_performed` ([dependency boundaries](DEPENDENCY_PROVENANCE.md)).
- Creating a draft is not publishing it. Publishing the draft, and changing the repository's visibility, are manual decisions.
- No GitHub run of `release.yml` is recorded in this repository as of 26 Sep 2026 (`docs/RELEASE_NOTES.md`). The steps above describe what the workflow file does, not a run that happened.
