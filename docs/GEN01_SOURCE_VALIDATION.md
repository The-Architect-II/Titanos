# GEN-01 control source validation

> **Historical record (GEN-01 control successor `192f`, September 2026).** This page is kept for provenance and is not updated. It records the source review of an inactive validation controller; the reviewed tree (`tools/validation/gen01`) is not part of the published source copies. Current work status is in the [progress checklist](PROGRESS.md) and the [capability matrix](CAPABILITY_MATRIX.md).

Updated **2026-09-21**. This note describes the inactive GEN-01 **control successor** whose source was independently reviewed in September 2026. It is not native package qualification, not model qualification and not a binary release. The reviewed controller source is retained in the maintainers' repository and is not part of the published source copies; the archived `plan.json` carries environment-specific target and host-key metadata and is not published.

When this source-only controller was recorded, the ledger had 13 bounded closures and 82 open items. The [current checklist](PROGRESS.md) tracks later results. GEN-01 remains open for remaining engine controls/cancellation coverage: later concurrent Arc B580/A770 execution passes its recorded engine path while preserving cosmetic image misses as separate MODEL_QUALITY (row E-021 of the [capability matrix](CAPABILITY_MATRIX.md)). This source review does not inherit those later runtime results.

## What was copied

The exact control candidate is the `192f` lineage:

- Candidate SHA-256: `192f97eb46df02b7c0a7d3c03bdf04f9ce7d8afd7dc4d8b4ac54b888abe76a3a`
- Location: `tools/validation/gen01` in the maintainers' repository; the tree is excluded from the published source copies by `tools/public_export.json`
- Sealed identity: `candidate.json` (`execution_authorized` is false)
- Recorded review: `SOURCE_REVIEW_SUMMARY.json`

Those frozen candidate bytes were not edited for this documentation pass. The copy includes the controller, finite-run contract, fake-backed tests, Apache-2.0 license, the recovered Windows Job ownership primitive, the accepted Build2 package **hash** manifest, and an inert lab plan. It does not include the native application, model weights, or raw private operational receipts.

The earlier control bundle `8475df093996a9e70d9f5d6eaadb888029cebce63e24a9ef312b095ae04bbbb4` was not reconstructed.

## Recovered accepted source versus this publication commit

Accepted Build2 **source content** from the `6d6ad826` lineage was recovered and reconciled into this tree. The original `6d` Git objects were not restored. This repository's git history is a fresh import (`7bd0f79d2c9671430bcb1ef7f5f86a68d0f78ba7`) plus later reviewed successor source. At the time of this note the engine lineage was `0.5.27.dev0+assetjobs1` / `7cea731742b3b3caab7b15d6ce06ac7da3f4c0ab`; the current source version is the one in `pyproject.toml`. Matching recovered bytes is not the same as restoring that old Git history, and it does not repeat historical runtime checks against a later commit.

See [publication scope](PUBLICATION_NOTES.md) for what this repository contains.

## What the recorded source review covers

The retained review reports **45** injected fake-input and failure-lifetime checks, zero failures, and `PASSED_NEW_CONTROL_SOURCE_ONLY_REVIEW`. The tests use fakes: they do not launch native executables or models, open a listener, load vendor Windows APIs, or connect to a machine.

That is independent **source** review of the new inactive control successor. It is not:

- runtime qualification
- model qualification
- Windows Job qualification on a live host
- SSH qualification

Default controller entry only verifies the inactive candidate. `release.template.json` is deliberately invalid and authorizes nothing. Publication verification also passed six targeted source-regression scripts. Relocation preserved 35 passing control cases; ten fixture cases initially lacked their empty evidence directory and passed after that directory was included through an ignore file. No native model, server or lab connection was started by these checks.

## Inert lab plan (metadata, not secrets)

The archived `plan.json` (not published) contains environment-specific target and public host-key metadata. It is not a portable default configuration. No passwords, tokens, private keys or credential files are included. The evidence summaries in the [capability matrix](CAPABILITY_MATRIX.md) omit machine identifiers, addresses and local paths; target-specific control data must be sanitized or replaced before public redistribution.

Remote `D:` fields describe the target machine's disk, not a drive of the reviewing workstation. The plan states that no native effects or authority come from the file itself. The expected external model-manifest hash is an admission binding; those bytes were not recovered or reread during this source cycle.

## Original finite authorization

Any later live run would need a **new** operator release that binds this exact candidate hash, a fresh operation identity, an owner reference and one original finite window (at most 600 seconds, no retries). Staging, runtime, model and listener effects must be released explicitly; flags cannot be inherited from an old grant. Credentials stay in an approved inventory outside this tree. Missing or expired authorization means the controller remains inactive.

## Historical controller gates and current acceptance layers

The original controller listed the following gates. Since the acceptance layers were separated on 21 Sep 2026, engine proof and model/presentation quality are recorded separately; this historical controller's inactive grant mechanics do not define the current work authorization:

1. Native Windows Job and host-commit limits on the live machine, under real drivers.
2. Actual Arc B580 inference, cancellation, invalid-input handling and owned closure—not fake HTTP.
3. MODEL_QUALITY/HUMAN_PRESENTATION: visual fidelity, cosmetic prompt adherence and human review are reported separately. Coherent intended-modality images can pass ENGINE_CONTRACT while missing the white-background preference. Decoding alone still does not excuse degenerate or wrong-modality output.

The remaining real control/cancellation coverage keeps [GEN-01](PROGRESS.md) partial. Cosmetic model quality or pending human film review does not independently block the engine condition. Source tests and candidate sealing alone do not close a native runtime gap.

Related status: [STATUS.md](../STATUS.md), [capability matrix](CAPABILITY_MATRIX.md), [progress checklist](PROGRESS.md).
