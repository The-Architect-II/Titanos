<!-- One topic per pull request. Keep every section; write "none" where nothing applies. -->

## What changes for a user

## Contract or document affected

<!-- OpenAPI or MCP surface, CLI options, settings, documentation pages. If a generated document changed, name the generator you ran. -->

## Checks run

<!-- Paste the commands and their results. Run from the repository root; see "Running the checks" in CONTRIBUTING.md. -->

```sh
python -B tests/check.py
```

## Generated documents

- [ ] Not affected
- [ ] Regenerated with their tools; the `--check` runs and `tests/check_route_contract_matrix.py` pass

## Hardware or model claims

<!-- Name the machine class, runtime profile and checkpoint each claim was run on, or write "none". A mocked model is not inference evidence. -->

## Hygiene

- [ ] No credentials, private hostnames or addresses, lab paths, machine names or downloaded weights
- [ ] New dependencies or adapted code carry upstream license and provenance (`tools/licenses/`, `docs/DEPENDENCY_PROVENANCE.md`)
- [ ] New checks are listed in a tier of `tests/checks.json` and, for CI tiers, as a step of the matching `.github/workflows/check.yml` job; `python -B tools/run_checks.py --sync-check` passes
- [ ] Line endings match what git stores for each file (`git ls-files --eol <path>`)
- [ ] Every commit is signed off (`git commit -s`, Developer Certificate of Origin 1.1)
