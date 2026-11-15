# Historical validation summaries and source checks

This file summarizes earlier releases. It does not qualify every feature or platform in the current source. The dated [capability matrix](CAPABILITY_MATRIX.md) and [current status](../STATUS.md) distinguish later results, retained failures and unverified work.

## 0.4.0: Linux x86-64 CPU

The historical environment used Python 3.12.13, glibc 2.39 and CPU numerical runtimes. An engine-only environment had no PyTorch. Recorded checks covered fresh wheel installation and checksum refusal, installed source identity, local HTTP/SSE, native GGUF chat/tool/schema transport, replicas, explicit placement, cancellation, crash/restart and shutdown, persistent video jobs, prepared ONNX/OpenVINO CPU models, and a bounded 20-request/concurrency-two HTTP check. An unmodified upstream engine-manager/manual-node proxy bridge also forwarded real text requests; this was not first-class desktop or general multimodal integration.

Seven modality paths ran, but some image/video checkpoints were tiny or random fixtures. A five-frame decoded video establishes wiring and file format, not coherent production video. Forced tool-call tests establish protocol/argument handling, not autonomous tool-selection quality. Synthetic device identity/reorder tests do not prove physical GPU execution. Qualification used a candidate wheel; a final documentation-only repack was separately installed and its Python modules matched the qualified source.

The native llama.cpp baseline was b10809 / commit `5266f24da`, with a pinned Qwen2.5-0.5B Q4_K_M checkpoint. Native executables and model weights were external, explicitly selected inputs. This historical run did not qualify physical GPUs/NPUs, Windows/macOS/ARM, distributed tensor parallelism, full-sized media quality, native vision projectors, XTTS cloning, Docker, cluster SSH transport, long production soak or real authenticated gated-model downloads. E-001 summarizes this scope.

## 0.3.1: CPU and device-identity controls

Earlier installed-wheel checks covered fresh CPU installation, the seven modality fixture paths, separate live CPU replica processes, targeting/streaming, cancellation/crash/restart/stop, and registry/authentication/error contracts. CUDA/ROCm/XPU identity resolution, reorder, migration and refusal paths used simulated devices. Those controls were not physical GPU or multi-GPU acceptance.

## 0.3.0: prepared runtime and parallelism boundaries

CPU model paths and prepared ONNX/OpenVINO execution, graph input/output contracts, explicit registration/preparation, replica processes and Hub download boundaries were exercised. A compatible tiny Llama reached distributed startup, but the environment denied a Gloo network-interface operation. Successful tensor-parallel inference remained **unqualified**, not passed. A plan or device map is not execution proof.

Published runtime wheels and configuration choices did not qualify NVIDIA/AMD/Intel/Apple accelerators, Windows/ARM/Intel macOS, full-sized media, Docker or a real gated-model download in that release. Do not carry these old exclusions forward as claims that later model-specific passes never happened: consult E-002 onward in the current matrix. Conversely, later passes do not rewrite the old failed or unqualified runs.

## Source checks

The repository carries its checks as standalone scripts under `tests/` (`python -B tests/check_<topic>.py`, `node tests/check_<topic>.js`) and the fixed-point generators under `tools/` (`--check`). [tests/README.md](../tests/README.md) explains how to run them on Windows, Linux and macOS, what each tier needs and how to add a check. A test script's presence is not a passing result, and a passing contract check is not model, accelerator, package or presentation evidence.

| Tier | Needs | Covers | Where CI runs it |
|---|---|---|---|
| base | `pip install .` plus `setuptools` (`PyYAML` optional) | engine, registry, protocol, export-hygiene, security and fixed-point checks | `check.yml` job `contract`: Python 3.11-3.13 on Linux, 3.12 on Windows and macOS |
| extras | the `dev` extra (numpy, pillow) and Node 22 | image, audio, retrieval and Studio DOM checks over tiny arrays and doubles | `check.yml` job `extras` |
| sdk | `tests/requirements-sdk.txt` | the OpenAI client library against the local routes; the broker SDK lanes (OpenAI, Anthropic, Google GenAI and ElevenLabs SDKs and the official MCP client against a real server and loopback mocks) | `check.yml` job `extras`, last steps |
| windows | a Windows host | Windows Job objects and the native supervisor | `check.yml` job `windows` |
| browser | Node 22, Playwright 1.62.1 with Chromium; Python with the project for the broker lane | headless Studio pages; the Studio Cloud and routing panel against a real server and a loopback mock | `check.yml` job `studio-browser` |
| package | the `build` extra | packaging contracts (the broker in the wheel and the frozen build included) and the release directory | `check.yml` job `package` |
| standalone | `pip install -e . pillow` | client, settings, lifecycle, Studio and MCP contracts | `standalone.yml` (push to main, pull requests, manual dispatch) |
| unscheduled, manual | see tests/README.md | checks not yet in a workflow; checks that need models, hardware or built artifacts; the broker overhead measurement and the live smoke lane | not in CI |

```sh
python -m pip install '.[dev]'
python -B tools/run_checks.py --list
python -B tools/run_checks.py --tier base --fast
```

`tests/checks.json` lists every check once per tier and mirrors the workflow jobs step for step; `python -B tools/run_checks.py --sync-check` fails when the two drift. Inspect each test's options and inputs before selecting native or model tests. `tools/fetch_checkpoints.py` explicitly downloads test checkpoints; do not confuse that preparation step with inference. The [hardware guide](HARDWARE.md), [native runtime guide](NATIVE.md) and [multi-GPU guide](MULTIGPU.md) describe model/runtime requirements. A successful source mock or numerical fixture is not a replacement for native package, accelerator, semantic-quality or cleanup proof.

`check.yml` runs on every push to `main`, on `v*` tags, on every pull request and on manual dispatch; `standalone.yml` runs on push to `main`, pull requests and manual dispatch; `release.yml` and `native-smoke.yml` are dispatched by a maintainer. A green run is a contract result over mocked or tiny fixtures and implies no model, accelerator, package or release qualification. Stable evidence IDs in [CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md#evidence-summaries) record the scope and outcome of each run.

## Cloud broker lanes

The [cloud broker](BROKER.md) has its own validation lanes. Every broker check sets `TITANOS_BROKER_EGRESS=loopback-only` and installs `tests/broker_mocks/netguard.py`, which refuses every non-loopback connect and name lookup (server subprocesses load it through `tests/broker_mocks/netguard_site`), so CI and default runs make **zero** provider calls. Upstreams are protocol-faithful loopback mocks with fault injection, and credentials are synthetic sentinels. A pass therefore proves Titanos's behaviour against mocks, not a provider's.

| Lane | What it covers | Checks |
|---|---|---|
| V1 mocks | One loopback mock per wire protocol with native auth and faults; fixtures transcribed from public references and linted | `check_broker_mocks.py`, `check_broker_fixtures.py`, `check_broker_fixture_lint.py` |
| V2 adapters | Per profile row, every row through its own adapter code: URL, auth header, static and hop headers, parameter map, usage, catalog, probe, error classes | `check_broker_adapters.py`, `check_broker_contracts.py` |
| V3 translation | Every translator pair both ways, streaming and not | `check_broker_translation_golden.py` |
| V4 streaming | Incrementality, `[DONE]`, the Responses event family, keepalives after commit, disconnects, terminal frames | `check_broker_streaming.py`, `check_broker_dispatch.py` |
| V5 errors | Every upstream status class in every inbound envelope; `x-should-retry` | `check_broker_upstream_errors.py`, `check_broker_http.py` |
| V6 key leaks | Secrets scrubbed from every output, never presented to another host, never in argv | `check_broker_redaction.py`, `check_broker_redact.py`, `check_broker_source_scan.py`, `check_broker_egress.py`, `check_broker_cli.py`, `check_broker_ui.js` |
| V7 vault | DPAPI, Keychain, Secret Service (each where available, otherwise untried), sealed file, references | `check_broker_vault.py` |
| V8 inbound keys | Keys, scopes over every operation, limits and budgets before any hit, transport rules | `check_broker_keys.py`, `check_broker_auth.py`, `check_broker_limits.py`, `check_broker_auth_http.py`, `check_broker_scope_coverage.py` |
| V9 routing | Zero hits for unknown names, policy and location, virtual models, fallback, breakers, advisors, explain | `check_broker_routing.py`, `check_broker_dispatch.py`, `check_broker_breakers.py`, `check_broker_advisor.py` |
| V10 roles | Switch table, managed refusals and delegation, parent headers, ports, manifests | `check_broker_roles.py`, `check_broker_manifest.py`, `check_one_stop_interfaces.py` |
| V11 end to end | A real `titanos serve` subprocess driven by raw httpx, the official MCP client, the OpenAI, Anthropic, Google GenAI and ElevenLabs SDKs, and Playwright | `check_broker_e2e.py`, `check_broker_sdk.py`, `check_broker_sdk_vendors.py`, `check_studio_broker_browser.js` |
| V12 disabled lane | Byte-identical local answers with the broker installed but unused; overhead measurement | `check_broker_disabled_regression.py` and the existing suites; `tools/broker_overhead.py` (manual, observations only) |
| V13 matrices | Route matrix, deviation map and broker matrix are fixed points; documentation claims map to matrix cells | `check_route_contract_matrix.py`, `check_protocol_deviations.py`, `check_broker_matrix.py`, `check_broker_docs_claims.py` |
| V14 live smoke | One small request per approved modality against real providers, owner-run | `tools/broker_live_smoke.py` (manual, never in CI) |
| V15 hygiene | No private identifiers; every check registered and mirrored in the workflow | `check_public_export_hygiene.py`, `tools/run_checks.py --sync-check` |

Configuration, CLI, client, MCP, stage and job behaviour have their own checks (`check_broker_settings.py`, `check_broker_config.py`, `check_broker_naming.py`, `check_broker_accounting.py`, `check_broker_audit.py`, `check_broker_stages.py`, `check_broker_plus.py`, `check_broker_jobs.py`, `check_broker_mcp.py`, `check_broker_client.py`). `check_broker_coverage.py` drives the integrated app with the broker active for the remaining surfaces: local targets through an active dispatcher, merged catalogs and voices, deadline and authentication envelopes per protocol and upstream rate limits. `check_broker_build_inclusion.py` (package tier) proves every broker module is in the wheel and in the frozen build's collection. `tests/checks.json` names the tier of each.

Running the heavier lanes locally:

- The SDK lanes run the client in a separate interpreter: `TITANOS_SDK_PYTHON` names a Python with `tests/requirements-sdk.txt` installed (the OpenAI lane), `TITANOS_SDK_PYTHON_EXT` (or `TITANOS_SDK_PYTHON_EXTRA`) one with the Anthropic, Google GenAI and ElevenLabs SDKs. A missing SDK makes the lane report `untried` (exit 0; `--strict` turns that into a failure). `TITANOS_SDK_FIXTURE_MODELS` may name a directory of small local models for the local cases; without it those cases are untried.
- The browser lane needs Node with Playwright and Python with the project installed (`TITANOS_PYTHON`). `TITANOS_CHROME_PATH` or `TITANOS_CHROME_CHANNEL` selects an installed Chrome; otherwise Playwright's Chromium is used. `TITANOS_BROKER_BROWSER_EVIDENCE` names a directory for screenshots.
- The broker matrix is regenerated from a fresh run of every broker check with `python -B tools/build_broker_matrix.py --run` (a few minutes) and checked with `--check`; the base-tier `check_broker_matrix.py` verifies the committed matrix without re-running every check. The run includes the SDK lanes and the two JavaScript Studio checks (run with `node`), which append their own case outcomes; set the SDK and browser variables above first, or their cells record untried.

### Live smoke lane (V14)

The live lane spends real money and is run only by the owner, under a written approval, never in CI. It covers two providers by environment reference: OpenAI (`OPENAI_API_KEY`) and the Hugging Face router (`HF_TOKEN`), with a total spend cap of at most USD 0.50 per approval.

1. Write an approval file **outside the repository**: `python -B tools/broker_live_smoke.py --approval-template > approval.json`, then fill in `approved_by`, `expires_at`, `output_dir` (an absolute directory outside the repository), the models per modality and a price row for every probed model (moderation needs a zero row, because budgets refuse unpriced targets). Extra requests (a Responses call, a non-streamed chat, one small image) run only when listed under `optional`.
2. Check it without starting anything: `python -B tools/broker_live_smoke.py --plan --approval approval.json` prints each probe with its cost estimate. `python -B tools/broker_live_smoke.py --dry-run` runs the whole tool offline against the loopback mocks.
3. Put the two variables in the environment of the shell that runs the tool (never on a command line), then run `python -B tools/broker_live_smoke.py --live --i-understand-this-spends-money --approval approval.json --home <directory outside the repository>` (or `titanos broker live-smoke` with the same arguments). Without both flags and an unexpired approval it refuses.
4. The tool creates a fresh `TITANOS_HOME`, starts a real `titanos serve --role broker` on loopback with a generated bootstrap key and a refs-only vault, configures one upstream per provider with an `env:` credential (validated with the provider's free models list), mints a two-hour key whose budget is the remaining cap, and sends one request per approved modality. It checks status, shape, usage, provenance headers and the decision record, stops at the cap (the server's own budget check refuses with 402 before any call that would exceed it), revokes the key, verifies the audit chain, stops the server and scans every file it wrote for the credential values, which it never prints or stores.
5. Results go to `<output_dir>/<run>/`: `receipt.json` (fingerprints, statuses, headers, usage and decision metadata; no prompts beyond the fixed probe strings, no outputs, no secrets) and `broker_live_evidence.jsonl`, one line per probe that passed every check. Exit codes: 0 all passed, 1 a probe, check or scan failed, 2 refused. A spend ledger in `output_dir` makes the cap a total across reruns.
6. Record the evidence with `python -B tools/build_broker_matrix.py --live <output_dir>/<run>/broker_live_evidence.jsonl` (repeat `--live` for several runs: the files are merged, an id may appear once, and together they replace the live records embedded in the matrix); those cells become live-pass and keep their mock evidence. A failed live call is never recorded as evidence, and a secret found by the scan voids every record of the run. A failed call may be recorded as an observation (a line with `outcome: observation` and a note), which is listed on its cell and never changes the cell's state. Then update the live statements of [BROKER.md](BROKER.md) and run `python -B tests/check_broker_docs_claims.py`.

#### Live results of 2026-10-02

Three owner-approved runs under one approval cap of USD 0.50 spent about USD 0.0174 in total; every run's secret scan found no credential, bootstrap key or minted key in any file it wrote, and every run's audit chain verified. <!-- matrix-count: live-pass=9 -->

- **Proven live** (live-pass in the [broker matrix](BROKER_MATRIX.md), each cell citing its run's evidence id beside its mock evidence): OpenAI moderations, embeddings, chat streamed and not streamed, Responses, speech, transcription and image generation (eight cells), and the Hugging Face router's chat (not streamed). The image call used gpt-image-1-mini at 1024x1024, quality low, and returned one PNG as base64 for USD 0.002201. The second run priced transcription from the tokens OpenAI reported (USD 0.00008875); the first run's transcription cost was the reservation estimate, a defect fixed before the second run.
- **OpenAI Images API change observed live.** dall-e-2 first answered 400 `unknown_parameter` for `response_format`, and later the same day 400 "The model 'dall-e-2' does not exist." Titanos no longer sends `response_format` to OpenAI image models, and the approval template now proposes gpt-image-1-mini (size 1024x1024, quality low, a `per_image` price to fill in). OpenAI's models list also answered 500 intermittently in direct calls that day.
- **Mock-only.** Anthropic, Gemini, ElevenLabs, xAI, Groq, Mistral, Together, DeepSeek, OpenRouter, Fireworks, Perplexity, Azure OpenAI, Ollama Cloud, Cerebras, SambaNova and the router and self-hosted upstreams (LiteLLM, a HiveMind gateway and custom OpenAI-compatible servers) have no live evidence, and neither do the OpenAI and Hugging Face router cells not listed above.
