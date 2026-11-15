# Security policy

## Threat model in plain words

Titanos is a single-operator local inference application. One person installs it on their own machine, registers model weights that already sit on local disk, and talks to it from that machine. The defaults follow from that.

| Default | What it means |
|---|---|
| Binds to `127.0.0.1:11435` | Nothing outside the machine can reach the server unless you change `TITANOS_HOST`. |
| A network bind needs `TITANOS_API_KEY` | `titanos serve --host 0.0.0.0` refuses to start without a key; the Docker image, whose default command binds that way, refuses too. `TITANOS_API_KEY` is the bootstrap key with scope `admin`: it grants inference, model lifecycle, the registry operations allowed over HTTP and key management. Scoped client keys (`tk_...`, minted with `titanos keys create`, `POST /api/keys` or Studio) narrow that to named scopes, model constraints, rate limits, budgets and an expiry; only the SHA-256 of a minted key is stored. There is still no per-user data isolation: every key shares one registry, one artifact store and one `TITANOS_HOME`. Put a TLS reverse proxy in front of anything that leaves a trusted local network, or set `TITANOS_TLS_CERTFILE`/`TITANOS_TLS_KEYFILE` for native TLS; the `broker` role refuses a non-loopback bind without it. |
| Cloud egress is off until you configure it | Without `TITANOS_HOME/broker.json` nothing leaves the host and every answer comes from local models. Once an upstream is configured ([docs/BROKER.md](docs/BROKER.md)), provider credentials are write-only (an OS secret store, an environment variable or an owner-only file referenced by name; the value is never printed or served), bound to one upstream's exact origin, and used only for callers a policy allows; every cloud answer is labelled with where it ran, and a hash-chained audit log records decisions (`titanos audit verify`). The `managed` role (for an orchestrator such as HiveMind) holds no usable provider key and makes no cloud call unless the parent delegates one. How the broker stores, binds and redacts secrets, and which risks remain, is in [docs/BROKER_SECURITY.md](docs/BROKER_SECURITY.md); its "pass" cells are proven against loopback mocks, with one owner-run live smoke on 2026-10-02, not a security audit. |
| Studio on a keyless loopback server can connect a provider | Studio is served from the same loopback origin and, when no key is enforced, may connect a cloud provider and store a credential (a value, or the provider's well-known environment-variable name for the provider's default address; never another variable, a `file:` path or a host value sent to a base URL Studio picked), each write carrying the `X-Titanos-Studio: 1` header, audited, and subject to the browser-origin checks every non-public route applies in keyless mode (a foreign `Origin`, cross-site `Sec-Fetch-Site` or outside `Referer` is refused 403 before any handler). A keyless caller can never mint keys, read the audit log or hold `credentials:write` outside that Studio path. This is the single-operator trust decision stated above, made explicit: anyone who can drive a browser on your machine can do what you can. Enforce a key (`TITANOS_API_KEY`, or mint one) and those writes need `broker:admin` plus `credentials:write`, or the bootstrap key. |
| No downloads at inference | Loading reads registered local checkpoints only. Weights reach the machine through an explicit `titanos pull`, `titanos add` or an acquisition you start. |
| Pickle weights and repository code are off | Legacy pickle checkpoints and model-repository Python run only when the local operator passes `--allow-pickle` or `--trust-remote-code` on the CLI. HTTP registration refuses those options and arbitrary native executables. |
| Local configuration is trusted | The registry, per-model defaults, bundled extensions and the native runtime catalog on your disk are executable configuration. Protect `TITANOS_HOME` (default `~/.titanos`) as you protect your account. |
| Limits are limits, not a sandbox | Request size, queue capacity, artifact retention and concurrency are configurable. Use separate data directories for independent instances. Host isolation is your environment's job. |
| Model output is untrusted data | Generated text, images and downloaded model metadata are inputs to your application, not instructions to it. |

Generated file access is limited to the instance's own artifact names. Uploaded media is inline, and inference does not fetch arbitrary input URLs. API keys and Hugging Face tokens are separate values; `titanos settings` prints neither. Never commit them or paste them into a report.

## Reporting a vulnerability

Report privately through GitHub's private vulnerability reporting for this repository: open the **Security** tab and choose **Report a vulnerability**, or go to <https://github.com/pgoode41/Titanos/security/advisories/new>. Do not open a public issue or pull request for a security problem. Do not attach credentials, model files or a working exploit against a third party.

If that form is not available, private reporting has not been switched on yet. Open a public issue that says only "security report: please enable private vulnerability reporting" and nothing else. A maintainer will enable it and reply there.

Include:

- `titanos --version` and how you installed Titanos (source checkout, wheel, compiled bundle, Docker).
- Operating system, hardware and the runtime profile or extras installed.
- The configuration class: loopback without a key, keyed, or network-bound behind a proxy.
- A concise reproduction and the impact you observed.

What to expect: an acknowledgement as soon as a maintainer is available; there is no formal response-time commitment. Confirmed problems are fixed on `main` first and described in [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md). You are credited unless you ask not to be. There is no bug bounty.

## Supported versions

| Version | Receives security fixes |
|---|---|
| `main` | Yes |
| Tagged releases | There is no tagged release yet. Once there is, the latest tag and `main`. |
| Development snapshots and compiled bundles built from older commits | No. Rebuild from `main`. |

## Out of scope

- Consequences of enabling `--allow-pickle` or `--trust-remote-code` for a checkpoint you do not trust. Passing the flag is the trust decision.
- Harmful, wrong or manipulated model output. Output is untrusted data by design; a coherent but wrong answer is a model-quality observation, not an engine vulnerability.
- Exposure caused by binding to a network without TLS, or by sharing `TITANOS_API_KEY`.
- Resource exhaustion by requests within the configured limits on a host without isolation. The limits are practical defaults, not a sandbox.
- Vulnerabilities in pinned third-party runtimes such as PyTorch, ONNX Runtime, OpenVINO and llama.cpp. Report them upstream; an issue asking Titanos to move a pin is welcome.
- Anything that requires write access to the operator's account or `TITANOS_HOME`. Local configuration is trusted by design.

This policy does not claim a completed security audit or a maintenance service level.
