# History

Dated records that explain how the current ledgers came to be. Nothing here changes a claim in [STATUS.md](../../STATUS.md), [CAPABILITY_MATRIX.md](../CAPABILITY_MATRIX.md) or [PROGRESS.md](../PROGRESS.md); those pages and the JSON ledgers beside them are the current state. This page is an index: every record stays at the path given.

## Where the historical material lives

| Record | Path | What it is |
|---|---|---|
| Release notes | [RELEASE_NOTES.md](../RELEASE_NOTES.md) | Newest first; the 0.4.0 note is kept verbatim as history. |
| 0.4.0 and 0.3.x validation | [VALIDATION.md](../VALIDATION.md) | Historical validation summaries and source checks. Cited by line from the roster, so it does not move. |
| 0.5.0 validation | [VALIDATION_0.5.0.md](../VALIDATION_0.5.0.md) | Native inference observed during 0.5.0 development. Cited by line from the execution matrix, so it does not move. |
| 0.5.2 hardware validation | [VALIDATION_0.5.2.md](../VALIDATION_0.5.2.md) | Runtime selection changes exercised in the 0.5.2 hardware window. |
| 0.5.x development notes | [DEVELOPMENT_0.5.5.md](../DEVELOPMENT_0.5.5.md), [DEVELOPMENT_0.5.14.md](../DEVELOPMENT_0.5.14.md), [DEVELOPMENT_0.5.15.md](../DEVELOPMENT_0.5.15.md), [DEVELOPMENT_0.5.16.md](../DEVELOPMENT_0.5.16.md) | What each increment added and which limits it kept. |
| GEN-01 control source review | [GEN-01 control source validation](../GEN01_SOURCE_VALIDATION.md) | The 2026-09 source review of an inactive control candidate. The reviewed tree (`tools/validation/gen01`) is excluded from the public source copies. |
| Manifests of earlier versions | `docs/provenance/` | File inventories for the versions they name; not an inventory of this tree. |
| Dated evidence pages | `docs/evidence/` | The pages each capability-matrix row links to. They stay in the private repository and are excluded from the public source copies, so those links do not resolve there by design. |
| Cluster handoff runbook (0.4.0) | `docs/CLUSTER_HANDOFF.md` | Historical lab record; excluded from the public source copies. |
| Ledger reconciliations and continuations | [progress.json](../progress.json) (`reconciliation`, `continuations`) | Machine-readable dated changes to the checklist, with method and rationale. |
| Evidence rows | [CAPABILITY_MATRIX.md](../CAPABILITY_MATRIX.md) | One bounded summary per evidence id E-001 to E-106; the "Version" or "Source / platform" cell of each row is its historical binding. |

## Dated ledger changes

Each row summarizes one reconciliation or continuation of the checklist. Closure counts never moved except where a closure change is named; the totals have been 32 closed and 63 open since 2026-09-22.

| Date | Evidence ids | Closure changes | What happened |
|---|---|---|---|
| 2026-09-22 | E-033 to E-051 | MOD-03, PKG-04, HW-13, TEXT-02 closed | Reconciliation from retained records and source inspection, with no new inference. Four stated conditions were met by retained named evidence; the 26 earlier closures were preserved. |
| 2026-09-22 | E-052 to E-054 | AUD-03, AUD-06 closed at the named ENGINE_CONTRACT scope | One known-two-speaker Windows CPU audio composition (VAD, ASR, diarization, transcript analysis) through the corrected native package. Model quality, automatic speaker count and human presentation stay open. |
| 2026-09-22 | E-055, E-056 | none | Source corrections, a sealed unapplied metadata amendment and a frozen CLI registry workflow. |
| 2026-09-22 | E-057, E-058 | none | An owned same-binary bind diagnostic and a fresh offline ENGINE successor. |
| 2026-09-22 | E-059 to E-063 | none | A corrected ENGINE successor, the first frozen reference-voice speech take, the declared speech controls, and the username-environment fix with its regression test. |
| 2026-09-23 | E-064 to E-068 | none | Live acceptance of every core modality from source and from the frozen successor, real translation/OCR/detection models, six extension modalities on the frozen package, and a Windows acquisition fix. |
| 2026-09-23 | E-069 to E-074 | none | Native llama.cpp serving on Windows CUDA, Studio UI checks, frozen-package diarization and audio analysis, the public MCP job workflow, the ONNX attribution amendment and a 25-scene edit map. |
| 2026-09-23 | E-075 to E-082 | none | The frozen standalone candidate built under a measured 3 GiB profile and validated by reuse; native MCP discovery and the Studio acquisition/ASR journey captured; candidate speech proven by API; an interim 25-scene overview. Human acceptance stays open. |
| 2026-09-23 to 2026-09-24 | E-083 to E-089 | none | Two Studio defects fixed in source with regression tests; the option-parity map, the provider deviation map and the representative roster recorded and test-bound; the validated successor candidate `b9a0a63b5b59`; the execution matrix. |
| 2026-09-24 | E-090 to E-096 | none | First matrix-kit campaigns: this workstation (fixtures, DirectML, extended set), the Linux x64 CUDA server, the Intel MacBook Pro, and five Windows x64 fleet hosts including the first Turing and Intel Arc receipts. |
| 2026-09-25 | E-097 to E-104 | none | Second-pass campaigns (macOS GGUF, Ryzen AI NPU, PyTorch XPU, Intel NPU, Snapdragon X); native Studio proof on `b9a0a63b5b59` bound to E-083 and E-084; the music gate: one 10 s stereo WAV under an enforced memory cap. |
| 2026-09-26 | E-105, E-106 | none | A source-runtime kit run on this workstation's Intel NPU; two music-inclusive Windows x64 portable candidates built, hash-verified and relocated, with music admission refused before load. |
| 2026-09-27 | E-107 | none | The first real ACE-Step music result through a packaged Windows x64 binary (candidate H, executable `d8d71003`) after the adapter's Windows progress-publication defect was repaired; one 10-second instrumental, no quality qualification. |
| 2026-09-28 | E-108 | none | The Apple M1 MacBook Air matrix-kit run bound as a legacy run: 46 passes written, 44 upheld, 4 by-design fails over the macOS ARM64 CPU, MPS and Core ML slots; the execution matrix rebuilt (242 verifier-upheld pass cells). |
| 2026-10-01 to 2026-10-03 | none | none | The cloud broker (settings, vault, inbound keys, mocks, adapters, router, app wiring, operator surfaces, matrices and docs), its provider-completeness round and the owner-run live smoke of 2026-10-02; the Studio redesign and its vendored fonts. Proven against loopback mocks and Node UI checks, recorded in `docs/broker_matrix.json`, not in the capability matrix; no checklist row closes, and the remaining conditions of INT-06, TEXT-03 and FAIL-06 were restated to match the source. |

The 2026-09-21 separation of acceptance layers (ENGINE_CONTRACT, MODEL_QUALITY, HUMAN_PRESENTATION) is recorded in `progress.json` under `taxonomy_correction`; it preserved all 95 stable ids and moved no closure.
