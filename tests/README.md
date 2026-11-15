# Running the Titanos checks

Every check is a standalone script run from the repository root: `python -B tests/check_<topic>.py` or `node tests/check_<topic>.js`. A script exits 0 on success and prints one JSON summary line or `PASS ...`. There is no pytest layer. A passing check proves the contract named in its docstring and nothing more: a mocked model is not inference evidence, and hardware claims require a run on that hardware.

`tests/checks.json` lists every check once per tier, in the order `.github/workflows/check.yml` runs it. `tools/run_checks.py` runs tiers from that file, and `--sync-check` fails when the file and the workflow drift.

## Quick start

Linux and macOS:

```sh
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -e '.[dev]'
python -B tools/run_checks.py --list
python -B tools/run_checks.py --tier base --fast
```

Windows (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
$env:TMP = "$HOME\titanos-tmp"; $env:TEMP = $env:TMP; New-Item -ItemType Directory -Force $env:TMP | Out-Null
python -m pip install -e '.[dev]'
python -B tools\run_checks.py --list
python -B tools\run_checks.py --tier base --fast
```

On Windows keep `TMP`/`TEMP` at a long-form path (no `~1` aliases) on a drive with free space. `check_public_export_hygiene.py` copies the source tree into TEMP twice, and several checks compare resolved paths.

## Tiers

| Tier | Install | Needs | Where CI runs it |
|---|---|---|---|
| `base` | `pip install .` plus `setuptools>=78` (`PyYAML` optional) | nothing else; no model, GPU or network | `check.yml` job `contract`: Python 3.11, 3.12, 3.13 on Linux; 3.12 on Windows and macOS |
| `extras` | `pip install '.[dev]'` | numpy, pillow; Node 22 for the `node tests/*.js` DOM checks | `check.yml` job `extras`, same five legs |
| `sdk` | `pip install -r tests/requirements-sdk.txt` | the `openai` client library; the broker SDK lanes also use the `anthropic`, `google-genai` and `elevenlabs` SDKs and the official MCP client, in this interpreter or one named by `TITANOS_SDK_PYTHON` / `TITANOS_SDK_PYTHON_EXT` (a missing SDK reports untried, shown as `UNTRIED` by the runner) | `check.yml` job `extras`, last steps |
| `windows` | `pip install .` | a Windows host (Job objects, native supervisor) | `check.yml` job `windows` |
| `browser` | `pip install .`, then `npm install --no-save --ignore-scripts playwright@1.62.1` then `npx playwright install --with-deps chromium` | Node 22, Chromium; Python 3.12 with the project installed for the broker lane (`TITANOS_PYTHON`, default `python`) | `check.yml` job `studio-browser` |
| `package` | `pip install 'setuptools>=78' wheel '.[build]'` | PyInstaller and friends (`check_broker_build_inclusion.py` builds a wheel offline and runs PyInstaller's module collection, no frozen build); `tools/build_release.py` runs only in CI | `check.yml` job `package` |
| `standalone` | `pip install -e . pillow` | client, settings, lifecycle, Studio and MCP contracts | `standalone.yml` (push to main, pull requests, manual dispatch); repeats four `base` checks (`check.py`, `check_jobs.py`, `check_protocols.py`, `check_hub.py`) and `check_tesseract.py` from `extras` |
| `unscheduled` | `pip install '.[dev]'` | no model; not yet in any workflow | not in CI |
| `manual` | varies | local checkpoints, a torch/ONNX/OpenVINO runtime, real hardware or built artifacts | never in CI; `--list --tier manual` prints each command with its inputs |

The `dev` extra in `pyproject.toml` is `setuptools`, `numpy`, `pillow` and `PyYAML`: what the `base` and `extras` tiers import beyond the engine's own dependencies. It is not a runtime dependency.

## The runner

```sh
python -B tools/run_checks.py --list                     # every tier; [run] or [skip: reason] per check on this host
python -B tools/run_checks.py --tier base --fast         # skip checks marked slow in tests/checks.json
python -B tools/run_checks.py --tier base --tier extras --keep-going --report checks-report.json
python -B tools/run_checks.py --tier all                 # every tier except manual; unmet needs are skipped, not failed
python -B tools/run_checks.py --sync-check               # tests/checks.json mirrors the workflow jobs; runs nothing
```

The runner executes each command with the interpreter that runs it, from the repository root, with `PYTHONDONTWRITEBYTECODE=1`. `python` in a manifest entry resolves to that interpreter and `node` to the `node` on `PATH`. A check is skipped, with the reason printed, when its tier or entry needs an OS, a Python module, `node`, a `node_modules/` package or a path this tree lacks. Skips never fail a run; the exit code is 1 only when a check fails. A lane whose JSON report says `"result": "untried"` (an SDK lane with no SDK or fixture here) exits 0 and is shown as `UNTRIED`, not `PASS`. Without `--keep-going` the first failure stops the tier and the last 40 lines of its output are printed. The runner writes nothing under the repository except the optional `--report` file.

`slow` marks in `tests/checks.json` are checks that took 8 s or more in one run on 26 Sep 2026 (the broker checks on 2 Oct 2026) on a busy Windows workstation (for example `check_registry_transaction.py` 20 s, `check_public_export_hygiene.py` 14 s, `check_security_cross_site.py` 13 s, `check_option_parity.py` 9 s); times vary with the host, so treat them as a rough guide.

## What the tiers need

- Node: CI pins Node 22 (`actions/setup-node@v4`). The `*_ui.js` and `*_i18n.js` checks in `extras` use `node:vm`, `node:assert/strict` and `node:fs` with DOM doubles; no browser. The `browser` tier drives Chromium through Playwright 1.62.1 and honours `TITANOS_PLAYWRIGHT_MODULE`, `TITANOS_CHROME_PATH`, `TITANOS_I18N_EVIDENCE_DIR` and `TITANOS_ACCELERATOR_UI_EVIDENCE`; without an evidence directory the scripts write into a temporary folder.
- numpy and pillow: the `extras` tier builds tiny arrays and PNGs; nothing loads a model.
- torch, transformers, diffusers, onnxruntime, openvino, FFmpeg: only the `manual` tier. The automated tiers patch or double these; many checks assert that they are never imported.
- Models: only the `manual` tier. `python tools/fetch_checkpoints.py /path/to/models` downloads the pinned tiny checkpoints for `tests/check.py --models`; that is the one step that uses the network.
- Windows: `check_windows_job_cleanup.py`, `check_windows_job_caller.py` and `check_native_tree_cleanup.py` exercise Job objects and the native supervisor; they print a `skipped` line and exit 0 elsewhere. The archived GEN-01 controller suite in the `windows` tier runs only where `tools/validation/gen01/tests` exists; the public source copies omit that tree, and `check.yml` guards the step with `hashFiles`. `check_daybreak_blocker1.py`, `check_daybreak_blocker2.py` and `check_matrix_kit_containment.py` (unscheduled) are Windows Job containment proofs. `daybreak` in those file names is the label of the September 2026 review of the matrix kit's containment and admission code; the number after `blocker` is the item in that review's list of blockers (1 containment before heavy work, 2 active monitoring, 3 fail-closed admission waits, 4 explicit resident placement, 6 the ingest gate and verdict typing).
- macOS: `check_music_build_inclusion.py`, `check_music_device.py` and `check_music_package.py` skip their music-runtime publication cases there (`unittest.skipUnless` Windows or Linux); the rest of each script still runs.

## Fixed-point generators

Some documents are rendered from a JSON source and committed. Their checks compare the committed bytes with a fresh render (CRLF normalised) and fail on any drift, so these files are never edited by hand:

| Rendered | Source | Generator |
|---|---|---|
| `docs/ROSTER.md` | `docs/roster.json` | `python tools/render_roster.py` (`--check` compares only) |
| `docs/PROGRESS.md` | `docs/progress.json` (the header also reads `docs/CAPABILITY_MATRIX.md` for the evidence ids no item cites) | `python tools/render_progress.py` (`--check` compares the exact bytes in the file's own newline style); `tests/check_progress.py` asserts one row per item, the header counts and no lab-identifier class |
| `docs/EXECUTION_MATRIX.md`, `docs/execution_matrix.csv`, `docs/execution_matrix.html` | `docs/execution_matrix.json` | `python tools/render_execution_matrix.py` |
| `docs/PROTOCOL_DEVIATIONS.md` | `docs/protocol_deviations.json` | `python tools/render_protocol_deviations.py` |
| `docs/route_contract_matrix.json`, `docs/ROUTE_CONTRACT_MATRIX.md` | the live app, MCP registry and source | `python tools/build_route_contract_matrix.py`; `tests/check_route_contract_matrix.py` rebuilds and compares |

Edit the JSON source, run the generator without `--check`, then run the matching check. The route matrix files are `text eol=lf` in `.gitattributes`; `docs/PROGRESS.md` keeps whatever newline style the checkout already has; the others are written with the platform newline and normalised by git.

## Checks that pin documentation text

| Check | File | Rule |
|---|---|---|
| `check_serve_bind_neutrality.py` | `docs/CONFIGURATION.md` | must contain `## Bind failures`, `never substitutes another port`, `show excludedportrange protocol=tcp`; must not contain `nothing blocks Titanos` or `` `serve` warns `` |
| `check_roster.py`, `check_execution_matrix.py`, `check_progress.py` | `docs/CAPABILITY_MATRIX.md` | every `E-NNN` id cited by `docs/roster.json`, `docs/execution_matrix.json` and `docs/progress.json` must appear |
| `check_public_export_hygiene.py` | `docs/CAPABILITY_MATRIX.md` | rows `E-090`, `E-091`, `E-092` contain `cells on this workstation;`; `E-096` and `E-098` name `the Turing laptop` |
| `check_route_contract_matrix.py` | `docs/progress.json`, `docs/RESPONSES.md`, `tests/check_responses.py` | item `INT-01` names `music` and `diarization`; the cited line ranges must exist |
| `check_public_export_hygiene.py` | `tests/check_model_page.py`, `tests/check_music_failure_diagnostics.py` | the allowlisted fixture literals in `tools/public_export.json` stay verbatim |
| `check_release_workflow_gates.py` | `.github/workflows/release.yml`, `check.yml` | gate step commands, their order, and one-line `run:` values |

## Adding a check

1. Name it `tests/check_<area>_<topic>.py` or, for a Studio DOM check, `tests/check_<topic>_ui.js`.
2. State the boundary in the first docstring lines: what is exercised and what is not (`no model, no network, no process`).
3. Write only under `tempfile`; never read real hardware; patch discovery and downloads.
4. Exit 0 on success and print one JSON line with `passed` and `scope`, or use `unittest.main(verbosity=2)`. Platform-specific scripts print `{"skipped": "... requires Windows"}` and exit 0 elsewhere.
5. Add the workflow step to the matching job in `.github/workflows/check.yml` and the same command to that tier in `tests/checks.json` in the same commit, then run `python -B tools/run_checks.py --sync-check`. A check that is not ready for CI goes into the `unscheduled` tier with a note.
6. If the check asserts documentation text, add the exact strings to the table above.

## Notes

- Pass `-B` (or set `PYTHONDONTWRITEBYTECODE=1`) so no `__pycache__` lands in the checkout; the export checks copy the tree.
- No automated check needs the network: hub and acquisition checks use controlled providers and mocked downloads; only `tools/fetch_checkpoints.py` downloads.
- `tests/studio_browser_fixture.py` is a fixture server for manual browser work, not a check.
- `check_mcp.py` (standalone) and `tests/check.py` spawn a server on a random loopback port; an interrupted run can leave a child process behind.
- `check_ingest_gate_negative.py` (unscheduled) reads a pre-change file from git history; on a checkout without that commit it reports the red proof unavailable and runs only the green proof.
- `python -B tools/run_checks.py --list` shows, per entry, notes about checks that currently fail against the source; fix the pair before moving such a check into a CI tier.
