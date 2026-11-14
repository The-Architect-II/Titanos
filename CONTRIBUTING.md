# Contributing to Titanos

Titanos is a standalone, local-first multimodal inference engine. This guide is for anyone who wants to report a problem, fix one, add a runtime adapter or improve the documentation. Bugs and feature requests go through [GitHub Issues](https://github.com/pgoode41/Titanos/issues). Vulnerabilities follow [SECURITY.md](SECURITY.md). Conduct expectations are in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Scope and principles

- Titanos runs on its own. PAIR and HiveMind are the maintainer's optional external client projects and design context, never required services.
- Weights stay outside the repository, the packages and the container image. A checkpoint reaches a machine only through an explicit `titanos pull`, `titanos add` or an acquisition the operator starts. Inference never downloads.
- Claims are bounded. The documentation distinguishes **implemented** (in source), **source-tested** (a check passes without a model), **profile available** (an install recipe exists) and **bounded native pass** (a named model ran on a named package and platform). A mocked model is not inference evidence. A hardware claim needs a run on that hardware.
- Prefer an existing runtime adapter over a new process or dependency.
- Nothing private goes into the tree: no credentials, host inventories, personal logs, checkpoints, caches, runtime state, build environments or operational records. Examples use neutral placeholders such as `/path/to/models`, `~/.titanos`, `127.0.0.1` and `<your-host>`, never a drive-rooted path, a user-profile path, a private address or a machine name.

## Development setup

Titanos needs Python 3.11, 3.12 or 3.13 (`requires-python` in `pyproject.toml`). Work in a virtual environment.

```sh
git clone https://github.com/pgoode41/Titanos.git
cd Titanos
python3 -m venv .venv                 # Windows: python -m venv .venv
. .venv/bin/activate                  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install "setuptools>=78"
titanos --version
```

`pip install -e .` gives you the engine, the CLI, the HTTP and MCP servers and the Python client without a numerical runtime. That is enough for protocol, contract and documentation work and for most checks. `setuptools` is imported by the two packaging gates (`tests/check_source_package.py` and `tests/check_public_export_hygiene.py`) and is not present in a fresh Python 3.12 or 3.13 virtual environment.

For inference work add a runtime. The recorded CPU recipe is:

```sh
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[models]"
```

Other accelerators have their own recipes in `tools/profiles.json` and [docs/HARDWARE.md](docs/HARDWARE.md). Each recipe records how far it has been exercised; installing one does not make a platform supported. The Studio DOM tests need Node 22.

## Running the checks

Every check is a standalone script. Run it from the repository root with `-B` so no `__pycache__` lands in the checkout; a script exits 0 on success and prints one summary line. [tests/README.md](tests/README.md) lists the tiers, what each tier needs installed, per-platform notes and the conventions for new checks. `tools/run_checks.py` runs a whole tier with one command; its options are documented there.

```sh
python -B tests/check.py                          # engine contracts, no model
python -B tests/check_source_package.py           # public source package gate
python -B tests/check_public_export_hygiene.py    # no private reference in the export copies
node tests/check_studio_modes_i18n.js             # a Studio DOM test (Node 22, no browser)
```

Before opening a pull request, run the tests next to the code you touched (`tests/check_<area>*.py`) and the repository-wide gates:

```sh
python -B tests/check_source_package.py
python -B tests/check_public_export_hygiene.py
python -B tests/check_route_contract_matrix.py
python -B tests/check_release_workflow_gates.py
python tools/render_roster.py --check
python tools/render_progress.py --check
python -B tests/check_progress.py
python tools/render_execution_matrix.py --check
python tools/render_protocol_deviations.py --check
```

Real-checkpoint checks need the `[models]` runtime and the pinned tiny fixtures: `python tools/fetch_checkpoints.py /path/to/models` once (the only step that uses the network), then `python -B tests/check.py --models /path/to/models`.

On Windows, point `TMP` and `TEMP` at a long-form path on a drive with free space before running the suite. Two checks compare resolved temporary paths or copy the whole export into the temporary directory, and an 8.3 short path or a full drive shows up as an unrelated failure.

## Coding and documentation conventions

Line endings. Git stores most text files in this repository as LF; a few, such as `.github/workflows/check.yml`, are CRLF in the index. `core.autocrlf` may show CRLF in a Windows checkout. Write a file with the style git already stores for it (`git ls-files --eol <path>` prints `i/lf` or `i/crlf`) and never mix styles in one file. `.gitattributes` pins some files byte for byte: the route-contract matrix as LF; vendored licenses, Swagger UI assets and the GEN-01 tree untouched. Do not renormalize them and do not add `* text=auto`.

Private paths. `tools/public_export.json` defines the reference classes that must not appear in the public copies: Windows user profiles, agent work trees, private IPv4 ranges, drive-rooted lab trees, Linux home directories, the lab's Linux hostname shape, drive-letter pairs, SSH host-key fingerprints and generated Windows machine names. `tests/check_public_export_hygiene.py` scans every text file in both export copies and every tracked file outside the excluded records, and it fails on the exporting machine's own name. Write examples with neutral placeholders from the start.

Generated documents. These files are rebuilt by a tool and compared by a check. Never edit them by hand.

| Generated file | Regenerate with | Drift check |
|---|---|---|
| `docs/ROSTER.md` | `python tools/render_roster.py` | `python tools/render_roster.py --check` |
| `docs/PROGRESS.md` | `python tools/render_progress.py` | `python tools/render_progress.py --check` and `python -B tests/check_progress.py` |
| `docs/EXECUTION_MATRIX.md`, `docs/execution_matrix.csv`, `docs/execution_matrix.html` | `python tools/render_execution_matrix.py` | `python tools/render_execution_matrix.py --check` |
| `docs/PROTOCOL_DEVIATIONS.md` | `python tools/render_protocol_deviations.py` | `python tools/render_protocol_deviations.py --check` |
| `docs/route_contract_matrix.json`, `docs/ROUTE_CONTRACT_MATRIX.md` | `python tools/build_route_contract_matrix.py` | `python -B tests/check_route_contract_matrix.py` |

Documentation style. One H1 per file. Sentence-case headings, no deeper than `###`. Short paragraphs, one idea per sentence, tables for parallel facts. Backticks for commands, paths, routes, environment variables and versions, not for product names. Use the bounded-claims vocabulary above and say what was run, on what, with what result. Some ledgers cite other files by line number (`docs/route_contract_matrix.json` cites `docs/RESPONSES.md`, for example), so edit in place rather than inserting lines above existing content. New pages are `docs/UPPER_SNAKE_CASE.md` and are added to [docs/README.md](docs/README.md) in the same commit. `STATUS.md`, `docs/PROGRESS.md` and the header counts in `docs/CAPABILITY_MATRIX.md` are refreshed in the same commit as any change to `docs/progress.json` or `docs/execution_matrix.json`; hosts are named by role label (`tools/lab_paths.HOST_ROLE_LABELS`), never by machine name.

Tests. A new check is a self-running script named `tests/check_<area>_<topic>.py`; Studio DOM tests are `tests/check_<topic>_ui.js`. Its docstring says what it exercises and what it does not (`no model, no network, no process`). It writes only under a temporary directory, never reads real hardware, exits 0 on success and prints one JSON summary line or `PASS`. List it in the matching tier of `tests/checks.json` (the one manifest `tools/run_checks.py` reads) and add the same command as a step of the matching `.github/workflows/check.yml` job in the same commit; `python -B tools/run_checks.py --sync-check` fails when the two differ, and a check that is not ready for CI goes into the `unscheduled` tier.

Changes. Every change explains the user-visible behaviour, the existing contract affected, the trade-off and the tests. New adapters declare their tasks, validate unsupported options, load only explicit local weights during inference, respect configured devices and cancellation, and ship a real-checkpoint smoke test. Additive API changes update the OpenAPI schema, MCP tool and resource coverage, the compatibility documentation and the route-contract matrix.

## License, attribution and sign-off

Contributions are licensed under Apache-2.0, the repository's [LICENSE](LICENSE). Keep [NOTICE](NOTICE) and existing attribution intact. Every commit carries a `Signed-off-by` line under the [Developer Certificate of Origin 1.1](https://developercertificate.org/); `git commit -s` adds it. When you add a dependency or adapt code, record its upstream source and license (see `tools/licenses/` and [docs/DEPENDENCY_PROVENANCE.md](docs/DEPENDENCY_PROVENANCE.md)). A source dependency and its binary can carry different notices, so inspect embedded native executables too. Do not claim authorship, review, testing or release approval that did not happen.

## How pull requests are reviewed

1. Open the pull request from a branch with one topic. The template asks what changes for a user, which contract or document is affected, which checks you ran, whether generated documents were regenerated, and what hardware any claim was run on.
2. CI must pass. A failing gate is fixed in the change, never by weakening the gate.
3. A maintainer listed in `.github/CODEOWNERS` reviews behaviour, contract, evidence and hygiene. Expect questions about the scope of a claim: "runs on X" needs a run on X, and a coherent but wrong model answer is a model-quality observation, not an engine defect.
4. Review happens in the pull request. There is no other channel.
5. Merging does not publish anything. Releases are built as drafts by `.github/workflows/release.yml` from an existing version tag. There is no tagged release yet. The procedure, the version rule and the public-branch steps are in [docs/RELEASING.md](docs/RELEASING.md).

## What CI runs

The workflow files under `.github/workflows/` are the authoritative list. Read each file's `on:` block for when it runs.

| Workflow | Job | What it does |
|---|---|---|
| `check.yml` | `contract` | Installs the base package (plus `setuptools` and PyYAML) on Linux with Python 3.11 to 3.13 and on Windows and macOS with 3.12, then runs the contract and regression scripts in order, including the repository-wide gates above. No numpy, pillow or Node. |
| `check.yml` | `extras` | The same five legs with `.[dev]` (numpy and pillow) and Node 22: the media scripts, the `node:vm` Studio DOM tests, the broker checks that need them, and from `tests/requirements-sdk.txt` the OpenAI SDK client tests and the broker SDK lanes (OpenAI, Anthropic, Google GenAI and ElevenLabs SDKs and the official MCP client against a real server and loopback mocks). |
| `check.yml` | `windows` | Windows only: the Job-object cleanup and native-supervisor checks. The archived GEN-01 controller suite runs only where `tools/validation/gen01` is present; the public copies omit it. |
| `check.yml` | `studio-browser` | Python 3.12 with the project installed (`pip install .`), then Node 22, Playwright 1.62.1 and Chromium: the headless Studio tests and the Studio broker lane against a real server and a loopback mock, with their evidence uploaded as an artifact. |
| `check.yml` | `package` | Installs `.[build]`, runs the five packaging checks (including the broker's inclusion in the wheel and the frozen build), then `tools/build_release.py`; uploads `release/` as an artifact. |
| `standalone.yml` | `contracts` | A short contract set on the three platforms with `pip install -e . pillow`. |
| `native-smoke.yml` | `smoke` | Manual: pinned real checkpoints on a CPU runner or a configured self-hosted GPU runner. |
| `release.yml` | `package` | Manual, from an existing tag: three contract checks and the two public-export gates, then the wheel, catalog, installer and sdist, then a draft GitHub release with `docs/RELEASE_NOTES.md` as its notes. |

## Where to talk

| Topic | Where |
|---|---|
| Bugs and feature requests | [GitHub Issues](https://github.com/pgoode41/Titanos/issues), using the templates |
| Questions and ideas | GitHub Discussions on the repository, if enabled; otherwise an issue |
| Vulnerabilities | [SECURITY.md](SECURITY.md), never a public issue |
| Conduct | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
