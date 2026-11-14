# Publication scope

This repository is the Titanos source, its tests, its build tooling and its documentation, under Apache-2.0. It is not a compiled application release. Binaries, model weights, voice references, generated media, caches, credentials and the maintainers' operational records are outside the tree, and no page here authorizes a runtime claim that the [capability matrix](CAPABILITY_MATRIX.md) and the [progress checklist](PROGRESS.md) do not record.

## What the tree excludes and why

| Excluded | Why |
|---|---|
| Model weights, prepared graphs, native runtimes and application binaries | External inputs with their own licenses. Titanos registers and runs them; it does not redistribute them. Build recipes are in [DISTRIBUTION.md](DISTRIBUTION.md). |
| Raw operational receipts, run folders, screen captures and machine state | They identify the maintainers' machines and record layout. The evidence rows in the capability matrix are their bounded summaries; the receipts stay in the private repository. |
| Personal voice references and generated media | Voice fixtures are personal data; generated media are large and not human-reviewed. |
| Credentials, `.env` files and caches | Standard exclusions. The deliberate `.env.example` and `.env.*.example` text examples stay. |
| Historical lab records: `docs/evidence/**`, `docs/CLUSTER_HANDOFF.md`, `tools/validation/gen01`, `docs/validation`, `tools/matrix_kit/legacy_results.json` | Dated lab records, a pinned population of private run hashes, and an archived control candidate whose plan carries environment-specific targeting data. `tools/public_export.json` excludes them from every exported source copy. Links into them from the capability matrix, the roster and the checklist do not resolve in those copies by design. |

## How the public copies are checked

`tools/public_export.json` is the single statement of what the exported source copies leave out and how the remaining documents are kept free of lab locators. The two copies are the sdist selected by `MANIFEST.in` and the native `SOURCE/` directory written by `tools/build_binary.py source_package`. Record drives become `<lab-records>/...`, user profiles become `<user-profile>/...`, and machines are named by stable role labels such as `turing-laptop-win-x64`, never by machine name or address.

`tests/check_public_export_hygiene.py` computes both copies with the real code and fails on any lab reference outside its fixture allowlist. It runs in CI and before every release build. It is a guard for lab-private references only: not a secret scanner, not a legal or vendor review, and not a release-readiness claim. See [Public export hygiene](DISTRIBUTION.md#public-export-hygiene).

## Source lineage

Version strings inside older documents and evidence rows describe the packages they name, not this tree; `pyproject.toml` carries the current source version. This repository's history is its own; earlier private development history is not replayed here, and an older commit id or manifest in a historical document is not a claim that this tree is byte-identical to that source.

The GEN-01 control tree is an archived, inactive candidate. Its independent source review (45 fake-input checks, no runtime, model, Windows Job or SSH qualification) is summarized in [GEN01_SOURCE_VALIDATION.md](GEN01_SOURCE_VALIDATION.md); the reviewed tree is not part of the public copies. Manifests of earlier named versions are kept under `docs/provenance/` as historical metadata, not as an inventory of this tree.

The public repository is built from one commit of the private `main` by `tools/build_public_branch.py` without the private history ([Public branch](DISTRIBUTION.md#public-branch), [Releasing](RELEASING.md#public-branch)); a `.titanos-public-tree` file at its root records the source commit and the absent records.

## Licensing and readiness

`LICENSE`, `NOTICE` and the retained third-party notice and provenance files are preserved. [DEPENDENCY_PROVENANCE.md](DEPENDENCY_PROVENANCE.md) separates source licensing from compiled dependencies, native vendor components, models and voices. No blanket binary redistribution approval is claimed: `tools/build_binary.py` records the dependency and native inventories with `license_review: not_performed` until that review happens.

Source publication, public-source readiness and binary-release readiness are separate decisions. `check.yml` runs on every push to `main`, on `v*` tags, on every pull request and on manual dispatch; `standalone.yml` runs on push to `main`, pull requests and manual dispatch; `native-smoke.yml` and `release.yml` run on manual dispatch only. A green workflow run proves the contracts its scripts name over mocked or tiny fixtures; it is not authorization to publish a binary and not model or hardware qualification. Before a compiled release, review the exact bundled dependency and native inventories and their notices in addition to runtime qualification.
