# Publication scope

This repository is the Titanos source, its tests, its build tooling and its documentation, under Apache-2.0. Tagged
releases carry the wheel, the source distribution and the install catalog ([RELEASING.md](RELEASING.md)). Binaries,
model weights, voice references, generated media, caches and credentials are not part of the tree, and no page here
authorizes a runtime claim that the [capability matrix](CAPABILITY_MATRIX.md) and the
[progress checklist](PROGRESS.md) do not record.

## What the tree does not carry and why

| Not carried | Why |
|---|---|
| Model weights, prepared graphs, native runtimes and application binaries | External inputs with their own licenses. Titanos registers and runs them; it does not redistribute them. Build recipes are in [DISTRIBUTION.md](DISTRIBUTION.md). |
| Raw run receipts, run folders, screen captures and machine state | They identify machines and record layouts. The evidence rows in the capability matrix are their bounded summaries. |
| Voice references and generated media | Voice fixtures are personal data; generated media are large and not human-reviewed. |
| Credentials, `.env` files and caches | Standard exclusions. The deliberate `.env.example` and `.env.*.example` text examples stay. |

## How exported copies are checked

`tools/public_export.json` states which paths the exported source copies prune and how the documents are kept free of
lab locators. The two copies are the sdist selected by `MANIFEST.in` and the native `SOURCE/` directory written by
`tools/build_binary.py source_package`. Record drives become `<lab-records>/...`, user profiles become
`<user-profile>/...`, and machines are named by stable role labels such as `turing-laptop-win-x64`, never by machine
name or address.

`tests/check_public_export_hygiene.py` computes both copies with the real code and fails on any lab locator outside its
fixture allowlist. It runs in CI and before every release build. It is a guard for lab locators only: not a secret
scanner, not a legal or vendor review, and not a release-readiness claim. See
[Public export hygiene](DISTRIBUTION.md#public-export-hygiene).

## Source lineage

Version strings inside older documents and evidence rows describe the packages they name, not this tree;
`pyproject.toml` carries the current source version. An older commit id or manifest in a historical document is not a
claim that this tree is byte-identical to that source. The GEN-01 control candidate's independent source review is
summarized in [GEN01_SOURCE_VALIDATION.md](GEN01_SOURCE_VALIDATION.md).

## Licensing and readiness

`LICENSE`, `NOTICE` and the retained third-party notice and provenance files are preserved. [DEPENDENCY_PROVENANCE.md](DEPENDENCY_PROVENANCE.md) separates source licensing from compiled dependencies, native vendor components, models and voices. No blanket binary redistribution approval is claimed: `tools/build_binary.py` records the dependency and native inventories with `license_review: not_performed` until that review happens.

Source publication, public-source readiness and binary-release readiness are separate decisions. `check.yml` runs on every push to `main`, on `v*` tags, on every pull request and on manual dispatch; `standalone.yml` runs on push to `main`, pull requests and manual dispatch; `native-smoke.yml` and `release.yml` run on manual dispatch only. A green workflow run proves the contracts its scripts name over mocked or tiny fixtures; it is not authorization to publish a binary and not model or hardware qualification. Before a compiled release, review the exact bundled dependency and native inventories and their notices in addition to runtime qualification.
