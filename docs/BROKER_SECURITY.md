# Cloud broker security

<!-- claims-legend -->
This page describes how the [cloud broker](BROKER.md) stores provider credentials, authenticates callers, limits
egress and keeps secrets out of logs, and which risks remain. Statements that something is proven carry the cell ids
of [BROKER_MATRIX.md](BROKER_MATRIX.md) in HTML comments in this file's source; `tests/check_broker_docs_claims.py`
checks them. Every broker check runs against loopback mocks with a network guard, so "pass" here means proven against
mocks and fakes on the host that generated the matrix, not against a real provider.

- [Credential storage](#credential-storage)
- [Credential binding and egress](#credential-binding-and-egress)
- [Write-only lifecycle](#write-only-lifecycle)
- [Inbound keys](#inbound-keys)
- [Transport: TLS and proxy headers](#transport-tls-and-proxy-headers)
- [Redaction and logging](#redaction-and-logging)
- [MCP](#mcp)
- [Residual risks](#residual-risks)
- [Threat model](#threat-model)

## Credential storage

`broker.json` never holds a secret: an upstream names a **credential id**, and the record in
`TITANOS_HOME/broker/vault/index.json` holds metadata only (source, bound upstream and origin, purpose, an 8-hex
fingerprint, validation state and times). A document that contains a secret-shaped string anywhere is refused (400
`broker_config_contains_secret`). <!-- matrix: feature:config.schema, feature:vault.write_only -->

A credential's value comes from one of three sources:

| Source | Where the value lives | Notes |
|---|---|---|
| `vault` | The OS store chosen by `TITANOS_BROKER_VAULT_BACKEND`, or a sealed file | Written with a hidden prompt, one line of stdin, or the HTTP body of a credential write |
| `env:VAR` | The server process environment, read at use time | Only the variable name is recorded; the value is never written anywhere. Names ending in `_BASE_URL` are refused, and so is every `TITANOS_*` name: the bootstrap key `TITANOS_API_KEY` and the server settings are never an upstream credential. |
| `file:/abs/path` | An owner-only file, re-checked on every read | POSIX: a regular file owned by the server's user with no group or other bits, opened without following links. Windows: an owner-only DACL. |

<!-- matrix: feature:vault.references -->

Vault backends (`TITANOS_BROKER_VAULT_BACKEND`, default `auto`):

| Backend | Platform | How | Evidence |
|---|---|---|---|
| `dpapi` <!-- matrix: feature:vault.dpapi --> | Windows | `CryptProtectData` in current-user scope with a per-install 32-byte entropy file; blobs and the vault directory get a protected owner-only DACL; atomic replace with a bounded retry | pass: a real DPAPI round trip on the Windows host that generated the matrix |
| `keychain` <!-- matrix-untried: feature:vault.keychain --> | macOS | `/usr/bin/security -i` with the command on stdin (never in argv), service `titanos.broker` | untried on the host that generated the matrix (the real round trip runs only on macOS, in a temporary keychain) |
| `secret-service` <!-- matrix-untried: feature:vault.secret_service --> | Linux desktop sessions | `secret-tool store` with the value on stdin; needs `secret-tool` on PATH and `DBUS_SESSION_BUS_ADDRESS` | untried on the host that generated the matrix |
| `sealed-file` <!-- matrix: feature:vault.sealed_file --> | any | Standard-library encryption under an owner-only passphrase file; opt-in only | pass: parameters, known-answer vectors, tamper and wrong-passphrase refusal |
| `refs-only` | any | No stored values: only `env:` and `file:` references | |

<!-- matrix: feature:vault.helper_argv -->

Rules that apply to every backend:

- **`auto` never selects `sealed-file`.** It picks DPAPI on Windows, Keychain on macOS and Secret Service on Linux when
  available. When none is available (for example a headless Linux server without a session bus), `auto` becomes
  **`refs-only`**: `env:` and `file:` references still work, and storing a secret value answers 503
  `vault_unavailable`, naming the two options (a reference, or `sealed-file` with a passphrase file). An explicitly
  configured backend that cannot be used is "unavailable", which broker role refuses at start.
- **Same OS user.** Every store is user-scoped: DPAPI blobs open only for the Windows user that wrote them, Keychain and
  Secret Service items belong to the login session. The CLI and the server must run as the same OS user, and the CLI
  refuses with a clear message when it cannot decrypt.
- **File references on Windows.** A file on a non-system drive (for example a secondary data drive) usually inherits an
  ACL that grants Authenticated Users read access, so a `file:` reference to it is refused (503
  `credential_unavailable`) until the operator tightens it, for example
  `icacls <file> /inheritance:r /grant:r <user>:F`. The error names the file, never its contents.
- **Lazy.** Opening the vault creates nothing; the directory, index, instance id and DPAPI entropy file appear on the
  first write. Managed role seals the vault (every use answers 503 `vault_sealed`) unless the parent delegates.
  <!-- matrix: feature:vault.lazy -->
- Each record remembers the backend that stored it, so changing the setting never strands existing values.

### Why sealed-file is weaker

`sealed-file` exists for hosts without an OS store that still want stored values. Construction (versioned,
encrypt-then-MAC, standard library only):

- key derivation: `hashlib.scrypt(passphrase, salt=32 random bytes, n=2**15, r=8, p=1, maxmem=64 MiB, dklen=64)`; the
  first 32 bytes encrypt and the last 32 authenticate (the explicit `maxmem` is needed because the derivation uses about
  32 MiB, above hashlib's default limit);
- keystream: block *i* = HMAC-SHA256(enc_key, nonce16 || *i* as 8 big-endian bytes), XORed with the plaintext;
- tag: HMAC-SHA256(mac_key, `TSF1` || version || id length || id || nonce || ciphertext), checked in constant time
  before anything is decrypted.

It is weaker than the OS stores because the passphrase file sits on the same disk under the same user: anyone who can
read that file and the vault directory can decrypt every value, and there is no hardware-backed or session-bound key
as with DPAPI, Keychain or Secret Service. The construction is a custom composition of standard primitives with
known-answer and tamper tests, not an audited library. The backend is unavailable unless the passphrase file is
owner-only (checked on every use), and broker role refuses to start on an unavailable backend.

## Credential binding and egress

- **Origin binding.** A credential record is bound at write time to one upstream and its exact origin
  (`scheme://host:port`, punycode host). The single outbound client attaches a credential only when the request URL's
  origin equals the record's origin and the upstream's current origin: no wildcard, no suffix match, no userinfo.
  <!-- matrix: feature:vault.origin_binding, feature:egress.url_refusals -->
- **Re-validation after a base URL change.** Changing an upstream's `base_url` resets its credential to `unvalidated`;
  the upstream then answers 503 `credential_unavailable` until it is re-validated, which re-binds the origin and is
  audited. Re-binding a credential to a changed origin presents the stored value there,
  so over HTTP it needs the `admin` scope: a principal holding `credentials:write` and `broker:admin` (who can edit
  `base_url`) gets 403 `scope_denied` before any connection. A `broker:admin` principal who edits upstreams can
  therefore never redirect a stored key, an `env:OPENAI_API_KEY` reference included.
  <!-- matrix: feature:vault.host_sources -->
- **Host sources need admin.** An `env:` or `file:` source names a value that lives on the server host. Over HTTP,
  naming one (`PUT /api/broker/credentials/{id}` or `rotate` with `env` or `file`) needs the `admin` scope;
  `credentials:write` alone may store a value it sends itself. The offline CLI (`titanos broker credential set
  --from-env`) runs as the operator on the host. <!-- matrix: feature:vault.host_sources -->
- **Resolved addresses.** The client resolves host names itself and checks every resolved address before connecting:
  `0.0.0.0/8`, `::`, link-local (`169.254.0.0/16`, `fe80::/10`), cloud metadata addresses, multicast, `240.0.0.0/4` and
  their IPv4-mapped and translated forms are always refused; RFC 1918, `100.64.0.0/10` and `fc00::/7` need
  `policy.allow_http_lan`; loopback is allowed. `TITANOS_BROKER_EGRESS` clamps everything to `internet` (default),
  `lan` (no public addresses) or `loopback-only`. TLS still verifies the origin host name.
  <!-- matrix: feature:egress.resolved_address, feature:egress.tls_name_binding -->
- **No ambient configuration.** `trust_env=False` and `follow_redirects=False`: proxy variables, `SSL_CERT_*`,
  `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL` and similar are never read, and a provider key in the environment is used only
  when an explicit `env:` record names it. A 3xx answer is 502 `upstream_redirect_refused`.
  <!-- matrix: error:redirect_refused\|openai_compat -->
- **Azure** credentials are bound to the exact resource host; look-alike hosts and other resources are refused.
  <!-- matrix: feature:egress.azure_host -->
- **Only the bound credential is presented.** The client's `Authorization`, vendor key headers, Titanos keys and other
  upstreams' credentials are never forwarded; a credential is never shown to a different upstream.
  <!-- matrix: feature:egress.credential_isolation -->
- **Response limits.** JSON and text answers are capped by `TITANOS_BROKER_MAX_UPSTREAM_JSON_BYTES` (16 MiB), media and
  streams by `TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES` (64 MiB), counted after bounded decompression; `set-cookie`,
  `location` and hop-by-hop headers are dropped. <!-- matrix: feature:egress.response_caps -->
- **Single choke point.** Credential headers are set only in the egress module, `Secret.reveal()` is called only by the
  egress module and the vault, egress grants are issued only by the router, and the broker package imports no
  subprocess or raw network module; source scans enforce all of it. <!-- matrix: feature:egress.source_scan -->

## Write-only lifecycle

- `PUT /api/broker/credentials/{id}` takes `{upstream, secret}`, `{upstream, env}` or `{upstream, file}` and by default
  validates with the profile's free probe (usually the models list) through the same egress client before committing.
  The answer is metadata only (`id, upstream, source, origin, fingerprint, validated_at, validation`) with
  `Cache-Control: no-store`. A failed probe answers 422 `credential_validation_failed` with the status class only.
- `POST .../rotate` stages the new value, validates it, swaps atomically and deletes the old value; `.../validate`
  re-validates; `DELETE` removes. Replacing a record's source (for example vault to `env:`) takes the upstream out of
  service until the new value validates.
- Every credential and key write needs TLS or a loopback peer (403 `transport_insecure`), is refused in keyless mode
  (403 `keyless_mode_refused`) and, for credentials, in managed role. One exception (R29): in standalone role on a
  loopback bind with nothing enforced, a request carrying `X-Titanos-Studio: 1` and a JSON body is the principal
  `keyless-studio`, which may write credentials, run `POST /api/broker/connect`, read `GET /api/broker/env-candidates`
  and change `policy.anonymous_cloud`; `allow_http_lan` and key minting stay CLI or keyed. Its host sources are narrow,
  because it does not hold `admin` and any loopback caller can send the header: an `env:` reference may name only one of
  the upstream profile's well-known credential variables (the R30 list, for example `OPENAI_API_KEY`), and only for an
  upstream that keeps the profile's default address (a `base_url` the caller picked would receive the host value);
  `file:` references are refused; a keyless caller never sees the fingerprint of an `env:` or `file:` value, and an
  `env:` reference that is unset, empty or too short gets one message (no presence or length oracle). A custom header forces a CORS preflight, which Titanos never approves, so a cross-origin page cannot send
  it; KeylessHostGuard and the keyless cross-site refusal apply first, and every such write is audited as
  `keyless-studio`. Every change is audited, with the fingerprint
  only, before it is applied; a change whose audit record cannot be written is refused.
  <!-- matrix: feature:inbound_auth.transport_secure, feature:inbound_auth.keyless_writes_refused, feature:audit.chain -->
- No HTTP route, CLI command, MCP tool, log line, error, decision, diagnostic or Studio view shows a value; there is
  deliberately no code path that returns one. The fingerprint is 8 hex characters of SHA-256 over a domain-separated
  prefix and the secret; it reveals no character of the secret. Secrets shorter than 8 characters are refused.
- The CLI never takes a secret in argv: values come from a hidden prompt, one line of stdin or an environment
  reference, and argument-parser errors never echo a typed value. <!-- matrix: feature:cli.no_secret_argv -->
- Studio's credential form is a password field that is never prefilled and is emptied after Save; in keyless mode it
  shows the CLI command instead. The form is disabled when the page is neither https nor loopback.

## Inbound keys

**Principals.** The bootstrap key (`TITANOS_API_KEY`, scope `admin`) keeps working exactly as before. Minted keys
(`titanos keys create`, `POST /api/keys`, Studio) have scopes, constraints, limits, a budget and an expiry. A key with
the `delegate` scope acts for an orchestrator in managed role. Without any key, the keyless loopback principal exists.

**Format and storage.** `tk_<id>_<secret>`: a 10-character base32 id and 43 characters of base64url from 32 random
bytes. `TITANOS_HOME/broker/keys.json` stores only the SHA-256 of the whole token (a slow KDF adds nothing for 256-bit
random secrets); verification looks the id up and compares digests in constant time, running a dummy compare for an
unknown id. The secret is shown once (HTTP 201 with `Cache-Control: no-store`, printed once by the CLI or written to an
owner-only `--output-file`, shown once by Studio and never stored in browser storage). Expiry, revocation and rotation
apply on the next request without a restart; offline CLI changes are picked up through the file's mtime.
<!-- matrix: feature:keys.lifecycle, feature:keys.offline_mint_seen -->

**Sticky enforcement.** Enforcement is on whenever a bootstrap key or **any** key record exists, expired and revoked
records included, and a marker file keeps it on even if `keys.json` disappears. There is no "keys present but auth off"
state. Returning to keyless mode needs the offline, audited `titanos keys reset --keyless` with a typed confirmation.
A network bind with nothing enforced fails closed, and revoking the last admin-capable key on a network bind without a
bootstrap key answers 409 `last_key_required`.
<!-- matrix: feature:inbound_auth.sticky_enforcement, feature:keys.last_admin_guard, feature:inbound_auth.keyless -->

**Headers.** `Authorization: Bearer` (scheme case-insensitive) on every route; under `/native/` every vendor key header,
and on the bare native paths of a vendor's allowlist that vendor's header; otherwise `x-api-key` only on
`/v1/messages`, `/v1/messages/count_tokens` and `/v1/models*` (the Anthropic SDK's model list, answered in Anthropic's
shape when `anthropic-version` is sent); `xi-api-key` only on `/v1/text-to-speech/*`, `/v1/speech-to-text` and
`/v1/voices*` (plus the ElevenLabs bare native paths, `/v1/models` among them: there a request with `xi-api-key` and no
`Authorization` gets the provider's model list through the native lane, so it also needs the `native` scope);
`x-goog-api-key` only on `/v1beta/*`. Two different presented credentials are 400
`ambiguous_credentials`. `?key=` on `/v1beta` is 401 `query_credential_refused` in every mode and is scrubbed from
access logs. In keyed mode a wrong key is 401 even from loopback. Failed claims are throttled per address
(`TITANOS_BROKER_AUTH_FAILURES_PER_MINUTE`, default 20, then 429 `auth_throttled`) and audited without the token;
requests without credentials are not counted, and an address already throttled stays refused even with a valid key.
<!-- matrix: feature:inbound_auth.bearer, feature:inbound_auth.vendor_headers, feature:inbound_auth.ambiguous, feature:inbound_auth.query_key_refused, feature:inbound_auth.throttle -->

**Scopes.** Checked before the body is read (403 `scope_denied`); any route not listed in
`broker.scopes.ROUTE_SCOPES` needs `admin`. <!-- matrix: feature:inbound_auth.scopes -->

| Scope | Grants |
|---|---|
| `admin` | Everything |
| `inference` | Every `inference:*` scope: `chat`, `embeddings`, `speech`, `transcription`, `images`, `moderation`, `plus` (API Plus routes), `tensor` |
| `mcp` | The `/mcp` transport; each tool still checks its own route scope |
| `jobs` | Asset jobs, video and music routes (owner-scoped for keys). `/v1/videos` also serves OpenAI videos through the native passthrough: there the Guard accepts `jobs` or `native`, and a request that stays on the local video route still needs `jobs` |
| `models:read`, `models:write` | Catalogs and model lifecycle; `models:write` includes `models:read` |
| `settings:read` | The `settings` block of `GET /api/info` |
| `cloud`, `cloud:<upstream>` | Cloud and router targets (all upstreams, or one) |
| `cloud:store` | Provider-side state (Responses or chat `store=true`, native files, batches, caches); not included in `cloud` |
| `native` | The native passthrough (`/native/<upstream>/...` and bare provider paths); not included in `cloud` |
| `route:explicit` | The `X-Titanos-Target` header |
| `delegate` | Managed role: `X-Titanos-Delegate: cloud` |
| `broker:read`, `broker:admin` | Broker status, catalog, own decisions and usage; `broker:admin` adds configuration, all decisions and usage, and includes `broker:read` |
| `credentials:write` | Credential writes (separate from `broker:admin`) |
| `keys:admin` | Key management |
| `audit:read` | The audit log |

The keyless loopback principal holds `inference`, `jobs`, `mcp`, `models:read`, `models:write`, `settings:read`,
`broker:read` and `broker:admin`, plus `cloud`, `native` and `cloud:store` only when `policy.anonymous_cloud` is true in
standalone role (the keyless loopback caller is the machine's owner; provider-side state still needs each upstream's
`data_policy.allow_store`). With `anonymous_cloud` off, its request for a cloud model answers 403 `cloud_disabled` naming
the switch (it may list every upstream anyway, so this is no oracle). It never
holds `credentials:write`, `keys:admin`, `audit:read`, `route:explicit` or `delegate`, and it cannot change
`anonymous_cloud` or `allow_http_lan` (only the offline CLI or a keyed `broker:admin` caller can).
<!-- matrix: role:standalone\|anonymous_cloud -->

**Constraints, limits and budgets.** Constraints narrow a key to model globs, upstreams, target classes and locations;
a target outside them answers 404 as if it did not exist. Limits: requests per minute, tokens per minute (estimated as
UTF-8 body bytes / 3 plus the output cap, settled against actual usage) and concurrent requests, each answering 429.
Budgets: USD per UTC day and month, tokens per day and a per-request maximum, answering 402 before any upstream request;
an unpriced target with a budget fails closed; spend survives a restart. In broker role every key with an egress scope
(`admin`, `cloud`, `cloud:<upstream>`, `cloud:store`) needs a budget and an expiry at most 365 days away, and serve
refuses to start while an active key breaks that rule.
<!-- matrix: feature:routing.visibility, feature:limits.rate, feature:limits.budgets, role:broker\|egress_key_rules -->

**Owner scoping.** For keys, jobs (get, list, cancel, idempotency matching), video routes, `/v1/files` artifacts and
Responses handles (including `previous_response_id`) belong to the key that created them; anyone else gets 404.
<!-- matrix: jobs.owner_scope\|local, feature:routing.handles -->

## Transport: TLS and proxy headers

- `serve` passes `TITANOS_TLS_CERTFILE` and `TITANOS_TLS_KEYFILE` (set both or neither, absolute paths) to uvicorn for
  native TLS in any role.
- Broker role refuses a non-loopback bind without native TLS, refuses to start without a bootstrap key, a writable
  audit log whose chain verifies and a usable vault posture.
  <!-- matrix: role:broker\|serve_preconditions -->
- `TITANOS_BEHIND_TLS_PROXY=true` says a TLS terminator on this host fronts Titanos; it is valid only with a loopback
  bind. `serve` always pins uvicorn `proxy_headers` to that setting and `forwarded_allow_ips` to `127.0.0.1`, so without
  the setting no `X-Forwarded-*` header can change the scheme or client address, and a `FORWARDED_ALLOW_IPS` environment
  value cannot widen it. This changes the previous default, under which uvicorn trusted forwarded headers from
  `127.0.0.1`. <!-- matrix: feature:cli.serve_pinning -->
- "Loopback peer" and "TLS" for the write checks are read from the connection uvicorn reports, so a forged
  `X-Forwarded-Proto` never satisfies them.

## Redaction and logging

- Resolved values are wrapped in a `Secret` whose text form is `<secret fp=........>`, cannot be pickled, and is
  registered in one exact-value registry the moment it exists. Minted key secrets and the bootstrap key are registered
  too.
- `scrub()` removes registered values, provider key shapes (`sk-`, `sk-ant-`, `sk-proj-`, `xai-`, `gsk_`, `nvapi-`,
  `hf_`, `AIza`, `tk_`), credential header values, `Bearer` and `Basic` tokens of at least 8 characters, JWTs and query
  credentials (`key=`, `api_key=`, `access_token=`).
- A global LogRecord factory scrubs every record when it is created, which covers child loggers such as httpx and
  httpcore whose records bypass logger-level filters; a handler filter on the root and uvicorn loggers is re-applied at
  startup after uvicorn's logging setup replaced the handlers, and scrubs exception text at emit. Access lines with
  `?key=` are scrubbed. <!-- matrix: feature:redaction.logging -->
- Decisions, usage records and audit entries are scrubbed before storage and hold metadata only, never prompts or
  outputs; upstream error bodies are scrubbed and truncated before logging or passthrough.
  <!-- matrix: feature:redaction.records, error:secret_bearing_codes\|openai_compat, error:secret_bearing_codes\|anthropic, error:secret_bearing_codes\|gemini, error:secret_bearing_codes\|elevenlabs, error:secret_bearing_codes\|ollama_native -->
- Child processes (native llama.cpp workers, replica processes, device probes) get an environment without provider
  variables, `*_API_KEY`, `*_BASE_URL` and every variable an `env:` record names.
  <!-- matrix: feature:redaction.child_env -->
- The end-to-end redaction lane runs a real server with debug logging and injects known secrets into upstream error
  bodies, failures and access lines; none appears in any response, MCP result, CLI output, log line or file under
  `TITANOS_HOME`. <!-- matrix: feature:e2e.netguard, feature:e2e.error_envelopes -->

## MCP

Mounted MCP tools run as the principal of each outer `/mcp` request, never as the bootstrap key; two keys on one
session keep their own principals, and a message whose outer request has no authenticated principal is refused. A key
with only the `mcp` scope can therefore no longer reach admin tools through the server's own key (it used to be
elevated). Direct in-process calls with no HTTP request (an embedding application calling the MCP server object) keep
the key the embedder passed to `create_mcp`. No MCP tool sets, rotates or deletes credentials, mints or revokes keys, or
changes policy, upstreams, aliases, prices or advisors. <!-- matrix: feature:mcp.principal_propagation, feature:e2e.mcp_scoped_key -->

## Residual risks

- **Python cannot zero strings.** A resolved secret stays in process memory as an immutable string for at least the
  secret cache lifetime (`TITANOS_BROKER_SECRET_CACHE_SECONDS`, 300 s) and until the garbage collector reuses the
  memory. A memory dump of the server process can contain it.
- **Same-user access.** Any process running as the server's OS user can read its environment (`env:` references),
  decrypt DPAPI blobs, query the Keychain or Secret Service items and read the sealed-file passphrase. Run Titanos as a
  dedicated user when that matters.
- **Budget overshoot.** Reservation happens before dispatch and settlement after the answer, so concurrent requests can
  overshoot a budget by at most the key's concurrent requests times the cost of one capped request. Transcription and
  translation reserve the uploaded audio's duration (exact for WAV, the file size at 8 kbit/s otherwise, which
  over-estimates compressed speech); an answer without usage or duration (`text`, `srt`, `vtt`) is charged that
  reservation in the limiter, the decision and the usage ledger, never USD 0. <!-- matrix: feature:limits.audio_budgets -->
  A transcription by URL reference uploads no audio and reserves only its body size. Use `--concurrent` and
  `--max-request-usd` to bound exposure.
- **Unregistered strings in provider errors.** A deterministic 4xx body from a provider is passed through after
  scrubbing; only registered secrets and recognisable key shapes are removed, so other provider-supplied text reaches
  the caller.
- **Successful answers are not scrubbed.** Allowlisted upstream response headers (request ids, rate-limit families,
  `character-cost`) have any registered secret value replaced by the redaction marker
  <!-- matrix: feature:redaction.response_headers -->, but 2xx bodies and stream frames, including an error event inside
  a committed stream, pass through unscrubbed: scrubbing them would rewrite model output. A provider that echoed the
  presented credential into a successful answer would hand it to the caller; no provider is known to do so.
- **Other runtimes inheriting the environment.** The child-environment scrub covers the native workers, replica
  processes and device probes, and the music runtime uses an allowlist. Other optional helper runtimes started with the
  full environment would inherit provider variables; keep provider keys out of the server's environment unless an
  `env:` record needs them, or use a vault backend.
- **Third-party routers** may drop `X-Titanos-Hop`, so loop detection through them is best effort; the fallback cap
  (at most 5) bounds amplification.
- **Keyless loopback.** As before, any local process on the host can call a keyless server on loopback; create a key
  when other users share the machine.
- **Plain HTTP on a keyed network bind** remains possible in standalone role; key and credential writes are refused
  there (403 `transport_insecure`), but inference keys travel in clear text. Use native TLS or a local terminator.

## Threat model

| ID | Threat | Mitigation | Evidence |
|---|---|---|---|
| T1 | Cost leak: an unknown, typo'd or blank model reaches a paid cloud | No heuristic classification; 404 or 400 before any egress grant exists; cloud policy off by default; ceiling; per-key cloud scope | <!-- matrix: feature:routing.leak_guard, role:standalone\|cloud_policy --> zero upstream hits with every upstream configured |
| T2 | Unauthorized cloud use through a caller-settable header | Cloud use needs a key scope plus deployment policy, evaluated once by the router; `X-Titanos-Target` needs `route:explicit`; keyless callers cannot enable anonymous cloud | <!-- matrix: feature:routing.visibility, feature:inbound_auth.keyless_writes_refused --> refusals before egress |
| T3 | Credential exfiltration through the API, Studio, logs, errors, records, fixtures or argv | Write-only API with fingerprints; `Secret` with restricted `reveal()`; central redaction with a global record factory and re-applied handler filters; scrubbed upstream errors; fixture linter; no argv secrets; source scans | <!-- matrix: feature:vault.write_only, feature:redaction.logging, feature:cli.no_secret_argv, feature:egress.source_scan --> end-to-end redaction lane |
| T4 | Credential sent to the wrong host (redirect, look-alike, userinfo, proxy, Azure wildcard, ambient base URL, repointed base URL, DNS rebinding) | Exact origin binding checked at send time; base URL change resets validation; resolved-address checks; no redirects, no ambient environment | <!-- matrix: feature:vault.origin_binding, feature:egress.url_refusals, feature:egress.resolved_address, feature:egress.azure_host --> |
| T5 | Plaintext at rest or spread to peers | OS store or references; explicit sealed-file; no values in `broker.json` or the index; nothing sent to peers; managed role seals the vault | <!-- matrix: feature:vault.dpapi, feature:config.schema, role:managed\|sealed_no_probes --> |
| T6 | Inbound key theft or brute force | 256-bit secrets stored as SHA-256; constant-time compare; per-address throttle; expiry and revocation without restart; shown once; writes only over TLS or loopback with pinned proxy headers | <!-- matrix: feature:keys.lifecycle, feature:inbound_auth.throttle, feature:inbound_auth.transport_secure --> |
| T7 | Privilege escalation through mounted MCP using the server key | Per-message outer principal, never the bootstrap key; principal-less mounted calls refused | <!-- matrix: feature:mcp.principal_propagation --> |
| T8 | Keys present but auth off, or expiry reopening keyless admin | Sticky enforcement; offline audited reset only; network bind fails closed; last-key guard | <!-- matrix: feature:inbound_auth.sticky_enforcement, feature:keys.last_admin_guard --> |
| T9 | Protocol-header confusion and URL credentials | Family-restricted vendor headers; 400 for two credentials; 401 for `?key=`; access-log scrub | <!-- matrix: feature:inbound_auth.vendor_headers, feature:inbound_auth.ambiguous, feature:inbound_auth.query_key_refused --> |
| T10 | Routing loops through router upstreams | Hashed hop header; 508 `loop_detected`; hop-marked requests stay local; fallback cap | <!-- matrix: role:any\|hops --> best effort through routers that drop the header |
| T11 | Fallback double spend or corrupted streams | Commit after the first non-error frame; keepalives only after commit; no fallback after upload, `store=true`, `previous_response_id`, handles or jobs; `Idempotency-Key` required for cloud jobs | <!-- matrix: feature:fallback.before_commit, feature:streaming.keepalive_and_disconnect, jobs.create\|cloud:speech --> |
| T12 | Budget bypass through concurrency or unbounded output | Reservation before dispatch; injected output cap for budgeted keys; concurrency cap; unpriced targets refused | <!-- matrix: feature:limits.budgets, feature:limits.rate --> overshoot bounded, see residual risks |
| T13 | Data leaving the host against local-first expectations | Off by default; per-key cloud scope; per-upstream data policy; `store` forced false; fail-closed `local_only`; provenance on every answer; explain before sending; no prompt text to advisors by default | <!-- matrix: feature:routing.data_policy, feature:routing.store_forced_false, feature:observability.provenance, feature:observability.explain, feature:advisor --> |
| T14 | SSRF through an operator-configured base URL or advisor URL | Writes need `broker:admin`; LAN addresses need `allow_http_lan` (not settable keyless); metadata, link-local and similar ranges always refused on resolved addresses; egress clamp | <!-- matrix: feature:egress.resolved_address --> |
| T15 | Abusive upstream answers (oversized, malformed, slow, cookies, redirects, bad tool JSON) | Response caps; parser limits; first-byte and idle timeouts; bounded queue; header allowlist; tool-argument JSON validation | <!-- matrix: feature:egress.response_caps, error:network_and_timeout\|openai_compat --> |
| T16 | Prompt-injection-driven tool misuse through MCP | Conservative static `open_world_hint`; no secret, key or policy tools; scopes enforced per principal | <!-- matrix: feature:mcp.tools --> |
| T17 | Logs becoming a privacy store | Metadata only; retention settings; owner-only `TITANOS_HOME/broker` | <!-- matrix: feature:redaction.records --> |
| T18 | Managed node bypassing the orchestrator (own keys, cloud fallback, auth-none upstreams) | Sealed vault and per-purpose egress refusal; delegation needs an operator setting, a delegate key, a per-request header and an explicit target, never fallback | <!-- matrix: role:managed\|egress_refused, role:managed\|delegation --> |
| T19 | Cross-tenant access to jobs, artifacts or Responses ids | Owner binding and 404 for non-owners | <!-- matrix: jobs.owner_scope\|local, feature:routing.handles --> |
| T20 | Public-repository leak of keys, lab names, addresses or private component names | Fixture linter; export hygiene test; fixtures use loopback, documentation address ranges and `example.invalid`; the slot for an external router without a published contract is neutrally named `pending-contract` (public components such as NVIDIA NeMo Switchyard are named) | covered by `tests/check_broker_fixture_lint.py` and `tests/check_public_export_hygiene.py` |
| T21 | Weakness of the sealed-file construction | Standard primitives, versioned format, known-answer and tamper tests; opt-in and documented as weaker | <!-- matrix: feature:vault.sealed_file --> |
| T22 | Plaintext transport of keys and credentials on a LAN broker | Broker role needs native TLS off loopback; proxy attestation only on loopback; writes need TLS or a loopback peer | <!-- matrix: role:broker\|serve_preconditions, feature:inbound_auth.transport_secure --> |
| T23 | Child processes inheriting provider keys | Child environment scrub; residual for other runtimes documented | <!-- matrix: feature:redaction.child_env --> |
| T24 | Native passthrough used to reach arbitrary provider endpoints or another tenant's objects | Per-profile method and path allowlist; `native` scope separate from `cloud`; `cloud:store` and `allow_store` for stateful rows; `broker:admin` for account-wide rows; created ids owner-bound (404 for other keys); credential attached only for the bound origin; inbound credentials and `X-Titanos-*` never forwarded; audited before egress | `tests/check_broker_native.py` |
| T25 | WebSocket relay abuse (cross-site upgrades, keys in frames or subprotocols, unbounded messages) | Authenticated in BrokerDispatch; keyless cross-site origins refused; credential frame fields removed; only allowlisted subprotocols forwarded; RFC 6455 client in egress with resolved-address checks and a per-message cap; session bounded by the request deadline | `tests/check_broker_native_ws.py` |
| T26 | A local router without inbound auth (NVIDIA NeMo Switchyard) forwarding what Titanos sends it, being reachable from the LAN, or looping requests back | The `switchyard` row sends no credential unless `auth: bearer` is set explicitly (a credential with the default auth is refused at load); only the listed session headers, never `Authorization`, vendor keys, cookies or `X-Titanos-*` hints, reach it; the documented deployment binds `127.0.0.1`; clients pointing back at Titanos carry a hop header, so a loop is refused after one pass; a request that comes back
without it but with the X-Request-ID this instance is still waiting on is refused 508 `loop_detected` at re-entry | <!-- matrix: feature:router.switchyard --> `tests/check_broker_switchyard.py`; real processes: `tests/check_switchyard_interop.py` (manual) |
| T27 | Anthropic requests reaching another tenant's provider-side state through the shared credential (workspace-wide file ids, reused containers, custom skills) or a provider URL leading the client's key off-host (a batch `results_url`) | A Messages, token-count or batch body naming a container, custom skills or a `file_id` needs `cloud:store` and `allow_store`; a file uploaded through Titanos by another key answers 404 before egress; `results_url` rewritten to this server's `/native/<upstream>/` path; Managed Agents and the Admin API `broker:admin` only | `tests/check_broker_anthropic_platform.py` |
| T28 | OpenAI endpoints that hand a client a direct line to the provider past Titanos metering and budgets (Realtime client secrets, WebRTC/SIP calls, legacy session mints, ChatKit sessions), or an organization-wide object store behind the shared key (evals, ChatKit threads, stored chat completions) | These native rows need `broker:admin`; custom voices, consents and other stateful rows need `cloud:store` and `allow_store` with owner-bound ids; retired families and the organization admin API are not allowlisted (400 before egress); a plain provider 403 never takes the shared credential offline, and neither does a 401 a key can cause by its choice of endpoint: OpenAI's restricted-key 401 ("insufficient permissions ... Missing scopes") never marks the credential, a 401 marks it at once only with the provider's key-rejection code (`invalid_api_key` and the like), and any other 401 on a client's call is checked against the upstream's own probe endpoint (at most once a minute per credential) before it can mark the credential | `tests/check_broker_openai_completeness.py`, `tests/check_broker_reachability.py` |
| T29 | Native-lane side channels: client-chosen account headers, encoded path tricks, model refs naming another upstream, and key-less resumable-upload chunks | `OpenAI-Organization`/`OpenAI-Project` forwarded only for `broker:admin` principals; `%2F`, `%5C`, `%3F`, `%23`, `%2E%2E`, dot and empty segments in path parameters (checked again after a `{model}` rewrite) and percent-encoded prefixes refused 400 before egress, and a request refused locally (path checks, `egress_refused`, deadline, concurrency) or refused because of the object the key asked for (a redirect, a declared size over `TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES`) is a neutral breaker outcome, as on the typed lane, so a key can never open the upstream's shared breaker by its own choice of request; `model@<other upstream>` and, on bare paths, local model names refused; a Gemini upload session URL carries a 256-bit capability bound to the caller, the upstream, the session path and upload id, spent at finalize and expiring after 24 hours (a revoked key's capability is refused); like the provider's own upload_id it is part of the session URL, so uvicorn's access log at level info shows it to whoever reads the server console | `tests/check_broker_native.py`, `tests/check_broker_reachability.py` |
