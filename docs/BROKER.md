# Cloud broker

Titanos can optionally route a request to a cloud provider or a router service instead of a local model. The part of
Titanos that decides this is the **broker**. It lives in the same `titanos serve` process, on the same port, behind
the same front door, and it is **off by default**: with no `TITANOS_HOME/broker.json` (or one with no enabled
upstream, alias or virtual model) nothing leaves the host and local answers are byte-identical to a Titanos without
the broker.

This page is for operators and integrators. Security details are in [BROKER_SECURITY.md](BROKER_SECURITY.md); roles
and orchestrators (PAIR, HiveMind) are in [TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md); the generated support matrix is
[BROKER_MATRIX.md](BROKER_MATRIX.md).

- [How to read the evidence](#how-to-read-the-evidence)
- [Concepts](#concepts)
- [Quick start with the CLI](#quick-start-with-the-cli)
- [Providers](#providers)
- [What works with a key](#what-works-with-a-key)
- [Surfaces and translation](#surfaces-and-translation)
- [Native passthrough and WebSockets](#native-passthrough-and-websockets)
- [What leaves the host](#what-leaves-the-host)
- [Routing, fallback and virtual models](#routing-fallback-and-virtual-models)
- [Router upstreams and advisors](#router-upstreams-and-advisors)
- [Keys, scopes, budgets and audit](#keys-scopes-budgets-and-audit)
- [Cost and prices](#cost-and-prices)
- [Provenance headers](#provenance-headers)
- [Error codes](#error-codes)
- [Observability](#observability)
- [Limits](#limits)

## How to read the evidence

Every statement on this page that something works (or does not) rests on a cell of
[BROKER_MATRIX.md](BROKER_MATRIX.md). The cell ids are written in HTML comments next to the statement in this file's
source, and `tests/check_broker_docs_claims.py` fails when a cited cell is missing or its state no longer matches.

<!-- claims-legend -->
| Matrix state | Meaning |
|---|---|
| pass (mock) | A named test ran and passed against a protocol-faithful loopback mock of the provider. No real provider was contacted: every broker check installs a network guard that refuses non-loopback sockets. |
| pass (mock, adapter column) | Proven through the reference profile of the same adapter (for example `openai` for the OpenAI-compatible family); the profile row differs only in data such as base URL, auth template and parameter map. |
| live-pass | An owner-run live smoke request reached the real provider and passed every check; the mock evidence stays cited beside it. |
| na | Not served on purpose, with the reason (usually 400 `broker_capability_unavailable` before any egress). |
| untried | No check exercised the cell yet, or its test was skipped on the host that generated the matrix. |

The live lane is run by the owner only, never in CI, and covers three providers (OpenAI, the NVIDIA API catalog and
the Hugging Face router; see [VALIDATION.md](VALIDATION.md#live-smoke-lane-v14)). Owner-approved runs on 2026-10-02
made nine cells live-pass: <!-- matrix-count: live-pass=9 -->

- OpenAI, eight cells: chat streamed and not streamed, Responses, embeddings, moderations, speech, transcription and
  image generation (with gpt-image-1-mini, 1024x1024, quality low). <!-- matrix-count: live-pass@openai=8 -->
  <!-- matrix-live: openai.chat\|openai\|off, openai.chat\|openai\|on, openai.responses\|openai\|off, openai.embeddings\|openai\|off, openai.moderations\|openai\|off, openai.audio.speech\|openai\|off, openai.audio.transcriptions\|openai\|off, openai.images.generations\|openai\|off -->
- Hugging Face router: chat, not streamed. <!-- matrix-live: openai.chat\|hf_router\|off --> <!-- matrix-count: live-pass@hf_router=1 -->
- NVIDIA API catalog: only the credential check was proven live (the key lists the catalog, `GET /v1/models` answers
  200; credential validation has no matrix cell). No chat or embedding call succeeded: the first run's models were
  retired (410 Gone), and the current models were refused by the provider for this account with 403 `Authorization
  failed` (Titanos answered 502 `upstream_auth_failed`), an account limitation rather than a Titanos defect. The NVIDIA
  cells therefore stay pass (mock) and carry those refusals as live observations, which are not evidence. <!-- matrix-count: live-pass@nvidia_api_catalog=0 --> <!-- matrix: openai.chat\|nvidia_api_catalog\|on, openai.embeddings\|nvidia_api_catalog\|off -->

Every other provider (Anthropic, Gemini, ElevenLabs, xAI, Groq, Mistral, Together, DeepSeek, OpenRouter, Fireworks,
Perplexity, Azure OpenAI, Ollama Cloud, Cerebras, SambaNova and the router and self-hosted upstreams: LiteLLM, a
HiveMind gateway, a PAIR facade, NIM and custom OpenAI-compatible servers) is mock-validated only, and so are the OpenAI
and Hugging Face router cells not listed above. <!-- matrix-count: live-pass@anthropic=0 --> <!-- matrix-count: live-pass@gemini=0 --> <!-- matrix-count: live-pass@elevenlabs=0 --> <!-- matrix-count: live-pass@azure_openai=0 --> <!-- matrix-count: live-pass@ollama_cloud=0 -->

## Concepts

| Term | Meaning |
|---|---|
| Upstream | One configured provider endpoint in `broker.json` (`upstreams.<id>`): a profile, a base URL (the profile default unless set), a credential id, `enabled`, a model allowlist or `allow_any_model`, a static or live catalog, a data policy and limits. The CLI and HTTP API call these "broker upstreams"; "provider" stays reserved for model acquisition. |
| Profile | A data row shipped in `src/titanos/broker/profiles.py`: adapter, default base URL, auth header template, native surfaces, parameter map. `titanos broker profiles` lists them. |
| Credential | A write-only record in `TITANOS_HOME/broker/vault/index.json` bound at write time to one upstream and its exact origin (`scheme://host:port`). `broker.json` names only credential ids, never values. |
| Qualified ref | `<upstream model>@<upstream id>`, split on the last `@`, for example `gpt-4o-mini@openai` or `meta-llama/Llama-3.1-8B-Instruct:fastest@hf_router`. |
| Published alias | A short public name for one target (`models.<alias>.target`), for example `gpt-4o-mini` -> `gpt-4o-mini@openai`. Aliases may not collide with local model names, virtual models or other aliases (409 `broker_name_conflict`). |
| Virtual model | A name that expands to an ordered list of targets (local and cloud) with a strategy and typed fallback triggers. |
| Cloud policy | `broker.json` `policy.cloud_policy`: `off` (default), `explicit` (cloud only when the caller names a cloud ref, alias or `X-Titanos-Target`) or `fallback` (chains may move from local to cloud). |
| Ceiling | `TITANOS_BROKER_CLOUD_CEILING` (default `fallback`). The effective policy is the lower of ceiling and file, ordered `off` < `explicit` < `fallback`, so a host can be pinned to `off` regardless of the file. <!-- matrix: role:standalone\|cloud_policy --> |
| Location | Where a request may run: `local_only`, `prefer_local`, `auto` (keep the configured order) or `cloud_only`. Taken from (in order) a managed parent's placement, the `X-Titanos-Location` header, the `location` body field on API Plus routes, the key's default and `policy.default_location`. |
| Role | `standalone` (default), `broker` or `managed`, fixed at serve start. See [TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md). |

Name resolution, first match wins and nothing is guessed from vendor prefixes:

1. A blank model is 400 `model_required`; over 256 characters is 400 `invalid_request`.
2. `X-Titanos-Target: <ref>` (needs the `route:explicit` scope) replaces steps 3 to 6 with one target.
3. An exact local registry name. Local names cannot contain `@` or `:`, so they never collide with qualified refs.
4. An exact virtual model name.
5. An exact published alias.
6. A qualified ref, when its upstream exists, is enabled, has a validated credential (or needs none) and lists the
   model, allows any model, or has it in a fetched live catalog. Live catalogs are kept in memory, so after every
   restart (and after `titanos broker connect`, which fetched in its own process) the first request that could
   resolve through a stale live catalog fetches it first: only the catalogs of upstreams the caller may see that its
   model could name, once per upstream however many requests wait (single-flight), waiting at most 10 seconds and
   never past the request deadline (the fetch keeps running and its answer serves the next request), and a failing
   catalog at most once a minute after its last attempt ended. A model call, a model lookup (`GET /v1/models/{id}`,
   `GET /v1beta/models/{id}`) or an API Plus or job request therefore never answers 404 only because no client listed
   models first. Local work never waits for a catalog: an API Plus or job request fetches only for the cloud models
   its body names (`model` and every `*_model` field), and only when it may be placed in the cloud (the
   `X-Titanos-Location` header, else the body's `location`, which defaults to `local_only` on API Plus routes; speech
   and image jobs), so a local model, a `local_only` request or a request from another broker hop (`X-Titanos-Hop`)
   contacts no provider. Ollama's `GET /api/tags` never waits: it starts the fetch and lists what is cached.
7. Optional HiveMind compatibility suffixes (`compat.hivemind_suffix_aliases`), longest suffix first, which only expand
   into a qualified ref.
8. Anything else is 404 `model_not_found` with zero egress. A model the caller may not use answers exactly like an
   unknown one, so there is no 403 oracle. <!-- matrix: feature:routing.leak_guard, feature:routing.visibility, feature:routing.aliases -->

## Quick start with the CLI

The shortest path is one command per provider. `titanos broker quickstart` lists which well-known key variables
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, ...) are set, by name only, and prints the
matching connect commands; `--yes` runs them. `titanos broker connect <provider>` creates the upstream from its
profile, stores the credential (`--from-env NAME`, `--from-stdin`, `--from-file PATH` or a hidden prompt), validates it
with one free probe, enables the upstream, raises `cloud_policy` from `off` to `explicit` (never above the ceiling) and
prints the models the provider lists. The stock SDKs' bare model ids (`gpt-4o-mini`, `claude-opus-5-5`,
`gemini-2.5-flash`, `eleven_multilingual_v2`) resolve on the single visible upstream of that provider without any alias,
also on a `titanos serve` started afterwards (the server reads the live catalog itself on the first request that needs
it); `--aliases` still publishes them as aliases when they collide with nothing. On a keyless standalone server bound to
loopback (no key enforced: the caller is the machine's owner) connect also sets `policy.anonymous_cloud` to true unless
you pass `--anonymous-cloud false`, so keyless SDKs and Studio can use the provider at once; with it off, a keyless
cloud request answers 403 `cloud_disabled` naming that switch. `--allow-store` sets the upstream's
`data_policy.allow_store` (provider-side files, batches, stored Responses and conversations).

The server, not the CLI, calls the provider. `--from-env NAME` records only the variable's name, so the `titanos serve`
process must hold the same variable in its own environment: export the key in the shell that starts the server (or in
its service definition). The CLI checks the key from its own environment, so a server started without the variable
cannot use the upstream although the connect succeeded; `/api/broker/upstreams` then shows it `"ready": false` with
`readiness.detail` `credential_unavailable`, every typed call to its models answers 404 `model_not_found` (the native
passthrough 503 `credential_unavailable`), and a restart with the variable set fixes it. Without `--url`, connect
writes the `TITANOS_HOME` files, so a server that is already running applies the change after a restart or after
`titanos broker reload --url URL`. On a keyless loopback server that reload needs no key; once the server enforces keys
(it was started with `TITANOS_API_KEY`, or any key was minted) the reload needs a key with `broker:admin` in
`TITANOS_API_KEY` (see [Applying changes to a running server](#applying-changes-to-a-running-server)). With `--url`
and an admin key in `TITANOS_API_KEY` the running server performs the connect itself and applies it at once, or refuses
it with 503 `credential_unavailable` when its own environment does not hold the variable.

```sh
# OPENAI_API_KEY set in this shell, and in the one that runs titanos serve
titanos broker quickstart
titanos broker connect openai --from-env OPENAI_API_KEY --aliases
titanos serve            # or, for a keyless server that is already running with the variable:
                         # titanos broker reload --url http://127.0.0.1:11435
```

The same steps over HTTP are `GET /api/broker/env-candidates` (names and booleans, never values) and
`POST /api/broker/connect` `{provider, credential: {value | env | file}, models?, base_url?, enable, validate, aliases,
anonymous_cloud?, allow_store?}` (broker:admin plus credentials:write; the answer never carries a secret). In keyless
standalone mode on loopback, Studio's providers panel performs these writes with the `X-Titanos-Studio: 1` header; that
keyless Studio principal may reference only the provider's well-known variable (`OPENAI_API_KEY` ...), only for the
provider's default address, never a `file:` path, and never sees a fingerprint of a host value (see
[BROKER_SECURITY.md](BROKER_SECURITY.md#write-only-lifecycle)).

The step-by-step equivalent follows. The CLI edits `TITANOS_HOME` files directly when you do not pass `--url` (it must run as the same OS user as the
server), or talks to a running server with `--url` and a key from `TITANOS_API_KEY` (or `--api-key-env VAR`). No
command accepts a secret as an argument. <!-- matrix: feature:cli.offline_and_url, feature:cli.no_secret_argv -->

```sh
# 1. Describe the upstream (disabled until it has a validated credential).
titanos broker upstream add openai --profile openai --models gpt-4o-mini

# 2. Reference the provider key by environment variable name. The value stays in the environment of the server
#    (and of this CLI for the one validation probe); only the variable name is recorded.
titanos broker credential set openai-key --upstream openai --from-env OPENAI_API_KEY
#    or a hidden prompt (the default), one line of stdin (--from-stdin) or an owner-only file (--from-file /abs/path)

# 3. Bind the credential and enable the upstream ("add" creates or replaces the upstream).
titanos broker upstream add openai --profile openai --models gpt-4o-mini --credential openai-key --enable

# 4. Publish a short name and allow explicit cloud use.
titanos broker alias set gpt-4o-mini gpt-4o-mini@openai
titanos broker policy set --cloud-policy explicit

# 5. Check where a request would go (zero egress), then serve.
titanos broker explain --model gpt-4o-mini
titanos serve
```

A keyless loopback caller is not allowed to reach the cloud unless `policy.anonymous_cloud` is true: `titanos broker
policy set --anonymous-cloud true` (standalone role only, and only from the CLI, a keyed `broker:admin` caller or Studio
on a keyless loopback server, which sends `X-Titanos-Studio: 1`); `titanos broker connect` and Studio's connect turn it
on by default on such a server. With it off, a keyless cloud request answers 403 `cloud_disabled` naming the switch;
with it on, the keyless caller also holds `cloud:store`, which each upstream's `data_policy.allow_store` still gates. <!-- matrix: role:standalone\|anonymous_cloud, feature:e2e.keyless, feature:connect.keyless_studio --> The alternative is
a scoped key:

```sh
titanos keys create --name laptop --scope inference --scope models:read --scope cloud:openai --expires 30d
```

`titanos keys create --name laptop` without `--scope` mints the default SDK key, `inference` plus `models:read`: the
scope `GET /v1/models` needs, so a stock OpenAI, Anthropic or Gemini SDK can call `models.list()` before its first
request (an `inference`-only key answers that call 403 `scope_denied`); add `cloud:<upstream>` scopes for cloud targets.
Creating any key turns on key enforcement for the whole server; see [Keys, scopes, budgets and
audit](#keys-scopes-budgets-and-audit). A key may also carry a budget (`--usd-per-day`, `--usd-per-month`,
`--tokens-per-day`, `--max-request-usd`; the broker role requires one on every key with a cloud scope). Budgets are
enforced against operator price rows and Titanos ships none, so a budgeted key reaches only targets that have a price
row; any other target answers 402 `budget_unpriced_target`. Add the rows first (your provider's current prices; see
[Cost and prices](#cost-and-prices); `prices import` replaces the whole price table). Price rows live in `broker.json`,
so a running server applies them only after a restart or a reload, and once the key above exists the server enforces
keys: `titanos broker reload --url URL` then needs a key with `broker:admin` in `TITANOS_API_KEY`, which the inference
key lacks. Restart `titanos serve`, or mint a short-lived admin key for the reload:

```sh
printf '%s\n' '[{"match": "gpt-4o-mini@openai", "input_per_mtok": "0.15", "output_per_mtok": "0.60"}]' > prices.json
titanos broker prices import prices.json
titanos keys create --name laptop --scope inference --scope models:read --scope cloud:openai --usd-per-day 2 --expires 30d
# apply the prices: restart titanos serve, or reload with an admin key (its secret is printed once)
titanos keys create --name admin --scope broker:admin --expires 1d
TITANOS_API_KEY=tk_... titanos broker reload --url http://127.0.0.1:11435
```

Then call the alias like a local model:

```sh
curl -s http://127.0.0.1:11435/v1/chat/completions -H "Authorization: Bearer $TITANOS_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}]}' -D - | grep -i x-titanos
```

Studio offers the same steps on its providers panel
([STUDIO_USAGE.md](STUDIO_USAGE.md#cloud-providers-and-routing)): with a key it uses that key's scopes, and on a keyless
standalone server bound to loopback it connects providers, stores credentials (a value, or the provider's well-known
environment variable for the provider's default address; never an arbitrary variable, a `file:` path or a base URL of
its choosing with a host value) and sets `anonymous_cloud` itself, each write sent with `X-Titanos-Studio: 1` and audited
as `keyless-studio`; minting keys and `allow_http_lan` stay CLI or keyed only. <!-- matrix: feature:connect.keyless_studio -->

### Applying changes to a running server

`titanos broker reload --url http://127.0.0.1:11435` applies `broker.json` changes (upstreams, aliases, policy, prices,
advisors) to a running server, and so does a restart; key and vault changes are picked up automatically. The reload is
an admin route: a keyless loopback server accepts it without a key, but once the server enforces keys (it was started
with `TITANOS_API_KEY`, or any key was minted) it needs a key with `broker:admin` (or the bootstrap key) in
`TITANOS_API_KEY`; without one it answers 401 `authentication_error`, and a key without the scope 403
`scope_denied`. The offline CLI commands that change `broker.json` end with a note saying which case applies, and name
the mint when no active key holds `broker:admin`.

## Providers

Each row is one profile. "Evidence" names the matrix proof level; the full per-surface grid is in
[BROKER_MATRIX.md](BROKER_MATRIX.md). "Live lane" says whether the owner-run live smoke covers the profile at all, and what it proved on 2026-10-02.

| Profile | Kind | Wire, auth header | Native surfaces | Evidence | Live lane |
|---|---|---|---|---|---|
| `openai` <!-- matrix: openai.audio.speech\|openai\|off, openai.audio.transcriptions\|openai\|off, openai.audio.translations\|openai\|off, openai.chat\|openai\|off, openai.chat\|openai\|on, openai.completions\|openai\|off, openai.completions\|openai\|on, openai.embeddings\|openai\|off, openai.images.edits\|openai\|off, openai.images.generations\|openai\|off, openai.moderations\|openai\|off, openai.responses\|openai\|off, openai.responses\|openai\|on, openai.responses.cancel\|openai\|off, openai.responses.delete\|openai\|off, openai.responses.input_items\|openai\|off, openai.responses.retrieve\|openai\|off --> | cloud | OpenAI, `Authorization: Bearer` | chat, completions, Responses (plus retrieve, delete, cancel, input items), embeddings, speech, transcription, translation, images, image edits, moderations; streamed images, image edits and transcription, speech events; native rows (see OpenAI and Azure OpenAI with one key) | pass (mock), profile row; translated surfaces pass (mock), profile row | in scope; live-pass on 2026-10-02 for chat (streamed and not), Responses, embeddings, moderations, speech, transcription and image generation |
| `azure_openai` <!-- matrix: openai.audio.speech\|azure_openai\|off, openai.audio.transcriptions\|azure_openai\|off, openai.audio.translations\|azure_openai\|off, openai.chat\|azure_openai\|off, openai.chat\|azure_openai\|on, openai.completions\|azure_openai\|off, openai.completions\|azure_openai\|on, openai.embeddings\|azure_openai\|off, openai.images.edits\|azure_openai\|off, openai.images.generations\|azure_openai\|off, feature:egress.azure_host --> | cloud | OpenAI deployments URL, `api-key` | chat, completions, embeddings, speech, transcription, translation, image generation, image edits; native files, batches, vector stores, fine-tuning, Responses and the Realtime WebSocket under `/native/<upstream>/` | pass (mock), profile row; translated surfaces pass (mock), profile row | not in scope |
| `azure_openai_v1` <!-- matrix: openai.audio.speech\|azure_openai_v1\|off, openai.audio.transcriptions\|azure_openai_v1\|off, openai.audio.translations\|azure_openai_v1\|off, openai.chat\|azure_openai_v1\|off, openai.chat\|azure_openai_v1\|on, openai.completions\|azure_openai_v1\|off, openai.completions\|azure_openai_v1\|on, openai.embeddings\|azure_openai_v1\|off, openai.images.edits\|azure_openai_v1\|off, openai.images.generations\|azure_openai_v1\|off, openai.responses\|azure_openai_v1\|off, openai.responses\|azure_openai_v1\|on, openai.responses.cancel\|azure_openai_v1\|off, openai.responses.delete\|azure_openai_v1\|off, openai.responses.input_items\|azure_openai_v1\|off, openai.responses.retrieve\|azure_openai_v1\|off --> | cloud | OpenAI v1 (`https://<resource>.openai.azure.com/openai/v1`, model = deployment name, no `api-version`), `api-key` | operator-declared subset of chat, completions, Responses (plus retrieve, delete, cancel, input items; store forced off unless the caller may store), embeddings, speech, transcription, translation, images, image edits | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `xai` <!-- matrix: openai.chat\|xai\|off, openai.chat\|xai\|on, openai.images.generations\|xai\|off, openai.responses\|xai\|off, openai.responses\|xai\|on, openai.responses.delete\|xai\|off, openai.responses.retrieve\|xai\|off --> | cloud | OpenAI, Bearer | chat (reasoning models never get `presence_penalty`, `frequency_penalty` or `stop`), Responses (plus retrieve, delete; store forced off unless the caller may store), image generation (xAI's own URLs or base64, never re-hosted) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `groq` <!-- matrix: openai.audio.speech\|groq\|off, openai.audio.transcriptions\|groq\|off, openai.audio.translations\|groq\|off, openai.chat\|groq\|off, openai.chat\|groq\|on, openai.responses\|groq\|off, openai.responses\|groq\|on --> | cloud | OpenAI, Bearer | chat, Responses (stateless), speech, transcription, translation | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `together` <!-- matrix: openai.audio.speech\|together\|off, openai.audio.transcriptions\|together\|off, openai.audio.translations\|together\|off, openai.chat\|together\|off, openai.chat\|together\|on, openai.completions\|together\|off, openai.completions\|together\|on, openai.embeddings\|together\|off, openai.images.generations\|together\|off, openai.rerank\|together\|off --> | cloud | OpenAI, Bearer | chat, completions, embeddings, image generation (own URLs; `b64_json` sent as `base64`), rerank, speech, transcription, translation | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `mistral` <!-- matrix: openai.audio.transcriptions\|mistral\|off, openai.chat\|mistral\|off, openai.chat\|mistral\|on, openai.embeddings\|mistral\|off --> | cloud | OpenAI, Bearer | chat, embeddings (tool ids remapped to 9 alphanumerics), Voxtral transcription | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `deepseek` <!-- matrix: openai.chat\|deepseek\|off, openai.chat\|deepseek\|on --> | cloud | OpenAI, Bearer | chat (translated tool loops send `reasoning_content` back) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `fireworks` <!-- matrix: openai.chat\|fireworks\|off, openai.chat\|fireworks\|on, openai.completions\|fireworks\|off, openai.completions\|fireworks\|on, openai.embeddings\|fireworks\|off, openai.rerank\|fireworks\|off, openai.responses\|fireworks\|off, openai.responses\|fireworks\|on, openai.responses.delete\|fireworks\|off, openai.responses.retrieve\|fireworks\|off --> | cloud | OpenAI, Bearer | chat, completions, embeddings, Responses (plus retrieve, delete), rerank | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `fireworks_audio` <!-- matrix: openai.audio.transcriptions\|fireworks_audio\|off, openai.audio.translations\|fireworks_audio\|off --> | cloud | OpenAI, Bearer (the Fireworks audio origin) | transcription, translation (operator-listed models) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `perplexity` <!-- matrix: openai.chat\|perplexity\|off, openai.chat\|perplexity\|on --> | cloud | OpenAI, Bearer | chat (citations, search results and cost pass through) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `nvidia_api_catalog` <!-- matrix: openai.chat\|nvidia_api_catalog\|off, openai.chat\|nvidia_api_catalog\|on, openai.embeddings\|nvidia_api_catalog\|off --> | cloud | OpenAI, Bearer | chat, embeddings (asymmetric models need `input_type`; none is injected silently); a 202 NVCF answer is 504 `upstream_timeout` | pass (mock), profile row; translated surfaces pass (mock), adapter column | in scope; on 2026-10-02 the key listed the catalog but no chat or embedding call succeeded (retired models 410; current models refused for the account with 403), so no cell is live-pass |
| `nvidia_ai_endpoints` <!-- matrix: openai.rerank\|nvidia_ai_endpoints\|off --> | cloud | NIM ranking shape, Bearer (the NVIDIA API catalog's second origin) | rerank (NIM shape translated; genai and visual NIMs through the native passthrough) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `nim_custom` <!-- matrix: openai.chat\|nim_custom\|off, openai.chat\|nim_custom\|on, openai.completions\|nim_custom\|off, openai.completions\|nim_custom\|on, openai.embeddings\|nim_custom\|off, openai.rerank\|nim_custom\|off --> | cloud | OpenAI, Bearer or none | chat, completions, embeddings, rerank (NIM `/ranking` shape) (operator-declared subset) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `cerebras` <!-- matrix: openai.chat\|cerebras\|off, openai.chat\|cerebras\|on, openai.completions\|cerebras\|off, openai.completions\|cerebras\|on --> | cloud | OpenAI, Bearer | chat, completions (text only by default) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `sambanova` <!-- matrix: openai.chat\|sambanova\|off, openai.chat\|sambanova\|on --> | cloud | OpenAI, Bearer | chat (text and images) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `openrouter` <!-- matrix: openai.chat\|openrouter\|off, openai.chat\|openrouter\|on, openai.completions\|openrouter\|off, openai.completions\|openrouter\|on, openai.embeddings\|openrouter\|off, openai.rerank\|openrouter\|off, openai.responses\|openrouter\|off, openai.responses\|openrouter\|on --> | router | OpenAI, Bearer | chat, completions, embeddings, Responses, rerank | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `hf_router` <!-- matrix: openai.chat\|hf_router\|off, openai.chat\|hf_router\|on, openai.responses\|hf_router\|off, openai.responses\|hf_router\|on --> | router | OpenAI, Bearer | chat, Responses | pass (mock), profile row; translated surfaces pass (mock), adapter column | in scope; live-pass on 2026-10-02 for chat, not streamed |
| `litellm` <!-- matrix: openai.audio.speech\|litellm\|off, openai.audio.transcriptions\|litellm\|off, openai.audio.translations\|litellm\|off, openai.chat\|litellm\|off, openai.chat\|litellm\|on, openai.completions\|litellm\|off, openai.completions\|litellm\|on, openai.embeddings\|litellm\|off, openai.images.edits\|litellm\|off, openai.images.generations\|litellm\|off, openai.moderations\|litellm\|off, openai.rerank\|litellm\|off, openai.responses\|litellm\|off, openai.responses\|litellm\|on, openai.responses.delete\|litellm\|off, openai.responses.retrieve\|litellm\|off --> | router | OpenAI, Bearer or none | operator-declared subset of chat, completions, Responses, embeddings, speech, transcription, translation, images, image edits, moderations, rerank | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `hivemind_gateway` <!-- matrix: openai.audio.speech\|hivemind_gateway\|off, openai.audio.transcriptions\|hivemind_gateway\|off, openai.chat\|hivemind_gateway\|off, openai.chat\|hivemind_gateway\|on, openai.completions\|hivemind_gateway\|off, openai.completions\|hivemind_gateway\|on, openai.embeddings\|hivemind_gateway\|off, openai.images.generations\|hivemind_gateway\|off --> | router | OpenAI, Bearer or none | operator-declared subset of chat, completions, embeddings, speech, transcription, images | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `pair_facade` <!-- matrix: openai.chat\|pair_facade\|off, openai.chat\|pair_facade\|on, openai.completions\|pair_facade\|off, openai.completions\|pair_facade\|on, openai.embeddings\|pair_facade\|off --> | router | OpenAI, Bearer or none | chat, completions, embeddings (operator-declared subset) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `custom_openai` <!-- matrix: openai.audio.speech\|custom_openai\|off, openai.audio.transcriptions\|custom_openai\|off, openai.audio.translations\|custom_openai\|off, openai.chat\|custom_openai\|off, openai.chat\|custom_openai\|on, openai.completions\|custom_openai\|off, openai.completions\|custom_openai\|on, openai.embeddings\|custom_openai\|off, openai.images.edits\|custom_openai\|off, openai.images.generations\|custom_openai\|off, openai.moderations\|custom_openai\|off, openai.rerank\|custom_openai\|off, openai.responses\|custom_openai\|off, openai.responses\|custom_openai\|on, openai.responses.cancel\|custom_openai\|off, openai.responses.delete\|custom_openai\|off, openai.responses.input_items\|custom_openai\|off, openai.responses.retrieve\|custom_openai\|off --> | cloud | OpenAI, Bearer, `api-key` or none | operator-declared (vLLM, LM Studio, llama.cpp server, private gateways) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `anthropic` <!-- matrix: anthropic.count_tokens\|anthropic\|off, anthropic.messages\|anthropic\|off, anthropic.messages\|anthropic\|on --> | cloud | Anthropic Messages, `x-api-key` plus `anthropic-version` | messages, count_tokens | pass (mock), profile row; translated surfaces pass (mock), profile row | not in scope |
| `gemini` <!-- matrix: gemini.batchEmbedContents\|gemini\|off, gemini.countTokens\|gemini\|off, gemini.embedContent\|gemini\|off, gemini.generateContent\|gemini\|off, gemini.streamGenerateContent\|gemini\|on --> | cloud | Gemini v1beta, `x-goog-api-key` (never `?key=`) | generateContent, streamGenerateContent (`alt=sse`), embedContent, batchEmbedContents, countTokens; speech, image and transcription bridges; native rows (see Gemini with one key) | pass (mock), profile row; translated surfaces pass (mock), profile row | not in scope |
| `elevenlabs` <!-- matrix: elevenlabs.stt\|elevenlabs\|off, elevenlabs.tts\|elevenlabs\|off, elevenlabs.tts_stream\|elevenlabs\|on --> | cloud | ElevenLabs, `xi-api-key` | text-to-speech, streaming text-to-speech, speech-to-text | pass (mock), profile row; translated surfaces pass (mock), profile row | not in scope |
| `ollama_cloud` <!-- matrix: ollama.chat\|ollama_cloud\|off, ollama.chat\|ollama_cloud\|on, ollama.embed\|ollama_cloud\|off, ollama.generate\|ollama_cloud\|off, ollama.generate\|ollama_cloud\|on --> | cloud | Ollama native, Bearer | `/api/chat`, `/api/generate`, `/api/embed` | pass (mock), profile row; translated surfaces pass (mock), profile row | not in scope |
| `switchyard` <!-- matrix: anthropic.count_tokens\|switchyard\|off, anthropic.messages\|switchyard\|off, anthropic.messages\|switchyard\|on, openai.chat\|switchyard\|off, openai.chat\|switchyard\|on, openai.responses\|switchyard\|off, openai.responses\|switchyard\|on, ollama.chat\|switchyard\|off, gemini.generateContent\|switchyard\|off, feature:router.switchyard --> | router | NVIDIA NeMo Switchyard: OpenAI and Anthropic Messages on one origin, no auth (Bearer only behind an authenticating proxy) | chat, Responses, messages, count_tokens (route ids as `<route>@<upstream>`; [SWITCHYARD.md](SWITCHYARD.md)) | pass (mock), profile row; translated surfaces pass (mock), adapter column | not in scope |
| `pending_contract` <!-- matrix-na: openai.chat\|pending_contract\|off --> | slot | none | none: an inert named slot, never called | na | not in scope |

Deferred and not registered as typed surfaces: AWS Bedrock, Vertex AI, Azure Entra ID, Cohere v2, Deepgram, Cartesia, Stability, OpenAI Realtime, OpenAI files and batches
and the ElevenLabs stream-input WebSocket (OpenAI streaming transcription events are served by the transcription
surface since 2026-10-02). OpenAI Realtime, OpenAI files and batches and the
ElevenLabs stream-input WebSocket are reachable as provider-native calls through the
[native passthrough](#native-passthrough-and-websockets) (no translation, no typed surface).
<!-- matrix-deferred: aws_bedrock, vertex_ai, azure_entra, cohere_v2, deepgram, cartesia, stability, openai_realtime, openai_files_batches, elevenlabs_stream_input -->

Provider terms, data handling and the right to re-serve one organisation credential to several downstream keys
(broker role) are the operator's responsibility. Enabling a provider allows every modality its profile carries unless
the upstream's `data_policy.modalities` narrows it. <!-- matrix: feature:routing.data_policy -->

## What works with a key

Put the provider key in its usual environment variable where `titanos serve` runs (the server reads it from its own
environment; see [Quick start with the CLI](#quick-start-with-the-cli)), run `titanos broker connect <provider> --from-env
NAME` (or use Studio's providers panel), and point the provider's own SDK at Titanos with a `tk_` key as its API key. The tables below
list, provider by provider, what that SDK (or any OpenAI client) can call. "Typed" means a Titanos surface with routing,
budgets, provenance headers and usage accounting (passthrough when the provider speaks the client's protocol,
translated otherwise); "native" means the guarded [native passthrough](#native-passthrough-and-websockets): the
provider's own endpoint, allowlisted per profile, with the `native` scope (plus `cloud:store` for provider-side state and
`broker:admin` for account-wide rows), the body forwarded untouched and the answer streamed back as the provider sends
it. Evidence: "live-pass" is an owner-run call to the real provider on 2026-10-02 (the live lane covers only OpenAI,
the NVIDIA API catalog and the Hugging Face router); "pass (mock)" is a test against a protocol-faithful loopback mock;
"allowlisted, not exercised by a check" means the row exists and is guarded like the others, but no test calls it yet.
Every row cites its matrix cells, so `tests/check_broker_docs_claims.py` fails when the evidence changes.
<!-- matrix-count: live-pass=9 -->

### OpenAI

Key `OPENAI_API_KEY`; the stock OpenAI SDK with `base_url=http://127.0.0.1:11435/v1` reaches the typed surfaces and the
native rows on their public paths (`.../native/<upstream>` reaches the native rows only). A bare model id such as
`gpt-4o-mini` resolves on the single visible OpenAI upstream that lists it when no local model, alias or virtual model
has that name, for inference and for the SDK's `models.retrieve('gpt-4o-mini')` alike (answered under that id, with
`titanos.ref` naming the qualified ref); `gpt-4o-mini@<upstream>` names one upstream. Videos share their paths with
Titanos' local video jobs: `POST /v1/videos` goes to OpenAI when its `model` names a cloud model of a visible OpenAI
upstream (`sora-2` and the other `sora-*` ids on the single such upstream, `sora-2@<upstream>`, or an alias of one;
JSON or the SDK's multipart form), and to the local video job otherwise (a local model name always wins); retrieve,
content and delete follow a video id created through Titanos by the same caller, every other id stays local, and
`GET /v1/videos` lists local jobs (`X-Titanos-Upstream: <id>` or `.../native/<upstream>` reaches the provider's list).
The cloud call needs `native`, `cloud` and `cloud:store` (no `jobs`); a request that stays local still needs `jobs`. A 429 cools down that model only: other models and
modalities on the same key keep serving, and the cooling model answers 429 `upstream_rate_limited` with `Retry-After`.
Storage: without `cloud:store` (or on an upstream without `data_policy.allow_store`) a client's `store: true` is forced
off and named in `X-Titanos-Dropped`, and `background: true` and `conversation` are refused (403 `scope_denied`) before
anything is sent, because such a response could never be polled, cancelled or continued through Titanos. On the native
lane a model given as `gpt-4o-mini@<upstream>` (or an alias of it) reaches OpenAI as `gpt-4o-mini`; a ref of another
upstream is refused (400).

| Endpoint | How | Streaming | Evidence |
|---|---|---|---|
| `POST /v1/chat/completions` | typed, passthrough (every field; Titanos-only fields named in `X-Titanos-Dropped`) | SSE | live-pass streamed and not streamed <!-- matrix-live: openai.chat\|openai\|off, openai.chat\|openai\|on --> |
| `POST /v1/completions` | typed, passthrough | SSE | pass (mock) <!-- matrix: openai.completions\|openai\|off, openai.completions\|openai\|on --> |
| `POST /v1/responses` | typed, passthrough (`store` forced false and named in `X-Titanos-Dropped` unless the key may store; `background` and `conversation` then refused) | SSE | live-pass not streamed, streamed pass (mock) <!-- matrix-live: openai.responses\|openai\|off --> <!-- matrix: openai.responses\|openai\|on --> |
| `GET`/`DELETE /v1/responses/{id}`, `POST .../cancel`, `GET .../input_items` | typed, owner-bound handle | `GET ...?stream=true` resumes a stored stream | pass (mock) <!-- matrix: openai.responses.retrieve\|openai\|off, openai.responses.delete\|openai\|off, openai.responses.cancel\|openai\|off, openai.responses.input_items\|openai\|off, feature:openai.media_streams --> |
| `POST /v1/embeddings` | typed, passthrough | no | live-pass <!-- matrix-live: openai.embeddings\|openai\|off --> |
| `POST /v1/moderations` (no `model`: the provider default) | typed, passthrough | no | live-pass <!-- matrix-live: openai.moderations\|openai\|off --> <!-- matrix: feature:openai.arguments --> |
| `POST /v1/audio/speech` | typed, passthrough | audio bytes; `stream_format=sse` events | live-pass (audio bytes), events pass (mock) <!-- matrix-live: openai.audio.speech\|openai\|off --> <!-- matrix: feature:openai.media_streams --> |
| `POST /v1/audio/transcriptions` | typed, passthrough (VAD fields dropped) | `stream=true` events | live-pass not streamed, events pass (mock) <!-- matrix-live: openai.audio.transcriptions\|openai\|off --> <!-- matrix: feature:openai.media_streams --> |
| `POST /v1/audio/translations` | typed, passthrough | no | pass (mock) <!-- matrix: openai.audio.translations\|openai\|off --> |
| `POST /v1/images/generations` | typed, passthrough (`url` re-hosted as PNG) | `stream=true` partial images | live-pass (gpt-image-1-mini, not streamed), streams pass (mock) <!-- matrix-live: openai.images.generations\|openai\|off --> <!-- matrix: feature:openai.media_streams, feature:images.rehosting --> |
| `POST /v1/images/edits` | typed, passthrough | `stream=true` partial images | pass (mock) <!-- matrix: openai.images.edits\|openai\|off, feature:openai.media_streams --> |
| `GET /v1/models` | Titanos catalog with `titanos.model_capabilities` | no | pass (mock) <!-- matrix: feature:catalog.model_capabilities --> |
| files (`GET /v1/files/{id}` and `.../content` reach OpenAI for the key that uploaded that file through Titanos; any other id stays Titanos' artifact route), voice consents, Responses `compact` | native (stored objects: `cloud:store`, ids bound to the creating key) | as the provider sends | pass (mock) <!-- matrix: feature:native.lane, feature:native.bare_paths, feature:native.openai --> |
| videos (`POST /v1/videos` with a cloud model, then retrieve, content and delete of that id; see above) | native (`cloud:store`, ids bound to the creating key) | content as the provider sends | exercised against the mock by `tests/check_broker_reachability.py` (no matrix cell yet) |
| uploads, batches, vector stores, conversations, containers, custom voices, stored chat completions, Responses `input_tokens` | native (stored objects: `cloud:store`, owner-bound) | as the provider sends | allowlisted, not exercised by a check |
| `POST /realtime/calls`, `POST /realtime/sessions`, `POST /chatkit/sessions` | native, `broker:admin` | as the provider sends | pass (mock) <!-- matrix: feature:native.openai --> |
| evals, ChatKit threads and session cancel, Realtime call actions (accept, hangup, refer, reject), transcription sessions and client secrets | native, `broker:admin` | as the provider sends | allowlisted, not exercised by a check |
| fine-tuning (also checkpoint permissions), model deletion | native, `broker:admin` | no | allowlisted, not exercised by a check |
| Realtime (`/v1/realtime`, the OpenAI SDK's URL for `base_url .../v1`, and `/native/<upstream>/realtime`) | native WebSocket (inbound upgrades need the `websockets` package, a runtime dependency since this round) | WebSocket | pass (mock; also through a real uvicorn server) <!-- matrix: feature:native.websocket --> |

Not served: `images/variations`, Assistants and Threads (retired) and the organization admin API (400 before egress).

### Azure OpenAI

Key `AZURE_OPENAI_API_KEY`, sent as `api-key`. `azure_openai` uses deployment URLs; `azure_openai_v1` uses the v1 API
with the deployment name as the model.

| Endpoint | How | Streaming | Evidence |
|---|---|---|---|
| chat, completions | typed, passthrough | SSE | pass (mock) <!-- matrix: openai.chat\|azure_openai\|off, openai.chat\|azure_openai\|on, openai.completions\|azure_openai\|off, openai.completions\|azure_openai\|on --> |
| embeddings, speech, transcription, translation, image generation, image edits | typed, passthrough | speech audio bytes | pass (mock) <!-- matrix: openai.embeddings\|azure_openai\|off, openai.audio.speech\|azure_openai\|off, openai.audio.transcriptions\|azure_openai\|off, openai.audio.translations\|azure_openai\|off, openai.images.generations\|azure_openai\|off, openai.images.edits\|azure_openai\|off --> |
| Responses on `azure_openai` | typed, translated onto chat | SSE | pass (mock) <!-- matrix: openai.responses\|azure_openai\|off, openai.responses\|azure_openai\|on --> |
| Responses (plus retrieve, delete, cancel, input items) on `azure_openai_v1` | typed, passthrough | SSE | pass (mock, adapter column) <!-- matrix: openai.responses\|azure_openai_v1\|off, openai.responses\|azure_openai_v1\|on, openai.responses.retrieve\|azure_openai_v1\|off --> |
| file upload (`/native/<upstream>/files`; `api-version` added unless the client sends one) | native | as the provider sends | pass (mock, egress-level check) <!-- matrix: feature:native.openai --> |
| batches, vector stores, fine-tuning, Responses, Realtime (`/native/<upstream>/...` only) | native, native WebSocket | as the provider sends | allowlisted, not exercised by a check |

### Anthropic

Key `ANTHROPIC_API_KEY`; the stock Anthropic SDK with `base_url=http://127.0.0.1:11435` sends the `tk_` key as
`x-api-key`. A bare model id such as `claude-opus-5-5` resolves on the single Anthropic upstream the caller may see, on
the Messages surfaces and in `models.retrieve()`; `models.list()` lists only models a Messages client can call (no
speech, transcription, embedding or image-only models), the single Anthropic upstream's by their bare ids.

| Endpoint | How | Streaming | Evidence |
|---|---|---|---|
| `POST /v1/messages` | typed, passthrough (every field and `anthropic-beta`; provider-side state needs `cloud:store`) | SSE | pass (mock) <!-- matrix: anthropic.messages\|anthropic\|off, anthropic.messages\|anthropic\|on, feature:e2e.anthropic_x_api_key, feature:anthropic.platform --> |
| `POST /v1/messages/count_tokens` | typed, passthrough | no | pass (mock) <!-- matrix: anthropic.count_tokens\|anthropic\|off --> |
| `GET /v1/models`, `GET /v1/models/{id}` (with `anthropic-version`) | Titanos catalog in Anthropic's shape | no | pass (mock) <!-- matrix: feature:anthropic.platform --> |
| OpenAI chat, completions and Responses, Gemini and Ollama clients to a Claude model | typed, translated per target model (thinking, effort, sampling, tool choice) | SSE | pass (mock) <!-- matrix: openai.chat\|anthropic\|off, openai.chat\|anthropic\|on, openai.responses\|anthropic\|on, gemini.streamGenerateContent\|anthropic\|on, ollama.chat\|anthropic\|on, feature:anthropic.model_rules --> |
| Message Batches (create, retrieve, list with `broker:admin`, delete, results with `results_url` rewritten to this server) | native | results as JSONL | pass (mock) <!-- matrix: feature:anthropic.platform, feature:native.bare_paths --> |
| Message Batch cancel | native | no | allowlisted, not exercised by a check |
| Files (upload, metadata, content, delete; stored, owner-bound) | native | no | pass (mock) <!-- matrix: feature:anthropic.platform --> |
| Managed Agents and the Admin API (`broker:admin`; Admin under `/native/<upstream>/` only) | native | session events as the provider sends | pass (mock) <!-- matrix: feature:anthropic.platform --> |
| Skills and their versions | native (stored) | no | allowlisted, not exercised by a check |

### Gemini

Key `GEMINI_API_KEY` or `GOOGLE_API_KEY`; the stock google-genai SDK with `base_url=http://127.0.0.1:11435` sends the
`tk_` key as `x-goog-api-key` (a `?key=` query credential is refused). google-genai puts the provider's own model id in
the path (`models/gemini-2.5-flash:generateContent`) and cannot send anything else, so a bare id resolves on the single
Gemini upstream the caller may see that lists it (with two Gemini upstreams, name one: `gemini-2.5-flash@<upstream>`).
The same rule answers `models.get('gemini-2.5-flash')` (`GET /v1beta/models/gemini-2.5-flash`, also spelled
`models/...`): that model's entry under the requested name, after local models and aliases. `files.upload` works unmodified in keyed mode: the resumable session URL Titanos hands back carries an upload capability
for that one session, because the SDK sends its upload chunks without a key.

| Endpoint | How | Streaming | Evidence |
|---|---|---|---|
| `:generateContent` | typed, passthrough (every field, both spellings) | no | pass (mock) <!-- matrix: gemini.generateContent\|gemini\|off, feature:e2e.gemini_goog_header, feature:gemini.codec --> |
| `:streamGenerateContent?alt=sse` | typed, passthrough | SSE (multi-MiB image frames included) | pass (mock) <!-- matrix: gemini.streamGenerateContent\|gemini\|on --> |
| `:embedContent`, `:batchEmbedContents`, `:countTokens` | typed, passthrough | no | pass (mock) <!-- matrix: gemini.embedContent\|gemini\|off, gemini.batchEmbedContents\|gemini\|off, gemini.countTokens\|gemini\|off --> |
| `GET /v1beta/models` | Titanos catalog with each model's published methods and limits | no | pass (mock) <!-- matrix: feature:catalog.model_capabilities --> |
| OpenAI chat, completions, Responses and embeddings to a Gemini model | typed, translated | SSE | pass (mock) <!-- matrix: openai.chat\|gemini\|off, openai.chat\|gemini\|on, openai.responses\|gemini\|on, openai.embeddings\|gemini\|off --> |
| OpenAI speech, image generation, image edits, transcription and translation to a Gemini TTS, image or audio-capable model | typed, translated through `generateContent` | no | pass (mock) <!-- matrix: openai.audio.speech\|gemini\|off, openai.images.generations\|gemini\|off, openai.images.edits\|gemini\|off, openai.audio.transcriptions\|gemini\|off, openai.audio.translations\|gemini\|off --> |
| Files (resumable `/upload/v1beta/files` with the stock SDK's key-less chunks, metadata, `/download/v1beta/...` media), Imagen `:predict` | native | no | pass (mock) <!-- matrix: feature:native.bare_paths, feature:native.gemini --> |
| Interactions (create, also streamed; get), File Search store create and `:importFile` | native (stored, owner-bound) | as the provider sends | pass (mock) <!-- matrix: feature:native.gemini --> |
| Interactions delete and cancel, File Search documents and operations | native (stored, owner-bound) | as the provider sends | allowlisted, not exercised by a check |
| webhooks, ephemeral `auth_tokens` | native, `broker:admin` | no | pass (mock) <!-- matrix: feature:native.gemini --> |
| triggers, server-managed credentials (`broker:admin`); cached contents, batches, Veo `:predictLongRunning` and operations, agents, environments, custom voices | native | as the provider sends | allowlisted, not exercised by a check |
| Live API (`v1beta`, `v1alpha`) and Lyria RealTime | native WebSocket (inbound upgrades need the `websockets` package) | WebSocket | allowlisted, not exercised by a check |

Not served: the `/v1beta/files/<id>:download` form and Veo download URIs, which answer with a redirect to storage (502
`upstream_redirect_refused`); `tunedModels`, `corpora`, `generateAnswer` and the `/v1beta/openai/` endpoint.

### ElevenLabs

Key `ELEVENLABS_API_KEY`; the stock ElevenLabs SDK with `base_url=http://127.0.0.1:11435` sends the `tk_` key as
`xi-api-key`. Plain model ids (`eleven_multilingual_v2`, `scribe_v1`) resolve on the single ElevenLabs upstream the
caller may see.

| Endpoint | How | Streaming | Evidence |
|---|---|---|---|
| `POST /v1/text-to-speech/{voice_id}` (no `model_id`: the provider default) | typed, passthrough (every field and query) | no | pass (mock) <!-- matrix: elevenlabs.tts\|elevenlabs\|off, feature:elevenlabs.typed, feature:e2e.elevenlabs_xi_api_key --> |
| `POST /v1/text-to-speech/{voice_id}/stream` | typed, passthrough | audio bytes | pass (mock) <!-- matrix: elevenlabs.tts_stream\|elevenlabs\|on --> |
| `POST /v1/speech-to-text` (`webhook=true` answers 202) | typed, passthrough (every form field) | no | pass (mock) <!-- matrix: elevenlabs.stt\|elevenlabs\|off, feature:elevenlabs.typed --> |
| `GET /v1/voices`, `GET /v1/voices/{voice_id}` | local voices merged with the upstream's | no | pass (mock) <!-- matrix: elevenlabs.voices\|catalog:cloud, elevenlabs.voice\|catalog --> |
| OpenAI `/v1/audio/speech` and `/v1/audio/transcriptions` to an ElevenLabs model | typed, translated (voice map or a raw voice id) | speech audio bytes | pass (mock) <!-- matrix: openai.audio.speech\|elevenlabs\|off, openai.audio.transcriptions\|elevenlabs\|off --> |
| timestamps, dialogue, sound effects, voice changer, audio isolation, forced alignment, music and its plan | native (usage from `character-cost`) | as the provider sends | pass (mock) <!-- matrix: feature:native.elevenlabs, feature:native.bare_paths --> |
| `/v2/voices` search, voice settings, voice cloning (`/v1/voices/add`), dubbing (stored objects owner-bound) | native | no | pass (mock) <!-- matrix: feature:native.elevenlabs --> |
| voice design, shared and similar voices, professional cloning, pronunciation dictionaries | native | no | allowlisted, not exercised by a check |
| `GET /v1/models` with `xi-api-key` and no `Authorization` | native: the provider's model list | no | pass (mock) <!-- matrix: feature:native.elevenlabs --> |
| history, subscription, single-use tokens (`broker:admin`) | native | no | pass (mock) <!-- matrix: feature:native.elevenlabs --> |
| `/v1/text-to-speech/{voice_id}/stream-input` | native WebSocket (a key in the first frame is removed) | WebSocket | pass (mock) <!-- matrix: feature:native.websocket --> |
| multi-stream-input and realtime speech-to-text sockets | native WebSocket | WebSocket | allowlisted, not exercised by a check |

Not served: the agents platform (conversational AI, workspace, studio projects) and `/v1/dubbing/project/*` (400 before
egress).

### OpenAI-compatible providers, routers and Ollama Cloud

Each of these speaks the OpenAI wire (Ollama Cloud its own), so every typed surface listed is a passthrough with SSE
streaming wherever the provider streams, and other protocols' clients are translated onto it. Key variables come from
each profile's `env_names` (`titanos broker quickstart` lists the ones that are set). Only the Hugging Face router has a
live-pass cell; the NVIDIA API catalog key listed its catalog live but no inference call succeeded for the account
(see [How to read the evidence](#how-to-read-the-evidence)). The evidence column covers the typed endpoints; of the
native rows only Groq files and Perplexity Search are exercised by a check, the others are allowlisted only.
<!-- matrix-live: openai.chat\|hf_router\|off --> <!-- matrix: feature:native.compat -->

| Profile | Typed endpoints | Native rows | Evidence |
|---|---|---|---|
| `xai` | chat, Responses (plus retrieve, delete), image generation (own URLs) | image edits (JSON), videos, `/responses/compact`, tokenizer, model listings | pass (mock) <!-- matrix: openai.chat\|xai\|off, openai.chat\|xai\|on, openai.responses\|xai\|off, openai.images.generations\|xai\|off, feature:compat.rules --> |
| `groq` | chat, Responses, speech, transcription, translation | files, batches | pass (mock) <!-- matrix: openai.chat\|groq\|off, openai.chat\|groq\|on, openai.audio.speech\|groq\|off, openai.audio.transcriptions\|groq\|off, feature:native.compat --> |
| `together` | chat, completions, embeddings, images, rerank, speech, transcription, translation | files, batches, fine-tunes, dedicated endpoints, videos | pass (mock) <!-- matrix: openai.chat\|together\|off, openai.chat\|together\|on, openai.images.generations\|together\|off, openai.rerank\|together\|off, openai.audio.speech\|together\|off --> |
| `mistral` | chat, embeddings, Voxtral transcription | FIM, OCR, classifiers, agents, conversations, batch jobs, files, speech, voices, libraries, fine-tuning | pass (mock) <!-- matrix: openai.chat\|mistral\|off, openai.chat\|mistral\|on, openai.embeddings\|mistral\|off, openai.audio.transcriptions\|mistral\|off --> |
| `deepseek` | chat (thinking round trip) | FIM, prefix completion, balance | pass (mock) <!-- matrix: openai.chat\|deepseek\|off, openai.chat\|deepseek\|on, feature:compat.rules --> |
| `fireworks`, `fireworks_audio` | chat, completions, embeddings, Responses, rerank; transcription and translation on the audio origin | image workflows, account API | pass (mock) <!-- matrix: openai.chat\|fireworks\|off, openai.chat\|fireworks\|on, openai.rerank\|fireworks\|off, openai.audio.transcriptions\|fireworks_audio\|off --> |
| `perplexity` | chat (citations and search results pass through) | Search, async chat | pass (mock) <!-- matrix: openai.chat\|perplexity\|off, openai.chat\|perplexity\|on, feature:native.compat --> |
| `nvidia_api_catalog`, `nvidia_ai_endpoints` | chat, embeddings; rerank (NIM ranking shape translated) | genai, visual and retrieval NIMs | pass (mock) <!-- matrix: openai.chat\|nvidia_api_catalog\|off, openai.chat\|nvidia_api_catalog\|on, openai.embeddings\|nvidia_api_catalog\|off, openai.rerank\|nvidia_ai_endpoints\|off --> |
| `nim_custom`, `cerebras`, `sambanova` | chat (`cerebras` and `nim_custom`: completions; `nim_custom`: embeddings, rerank) | none | pass (mock) <!-- matrix: openai.chat\|nim_custom\|off, openai.chat\|cerebras\|off, openai.chat\|cerebras\|on, openai.chat\|sambanova\|off, openai.chat\|sambanova\|on --> |
| `openrouter` | chat, completions, embeddings, Responses, rerank | generation stats, credits, providers | pass (mock) <!-- matrix: openai.chat\|openrouter\|off, openai.chat\|openrouter\|on, openai.embeddings\|openrouter\|off, openai.rerank\|openrouter\|off --> |
| `hf_router` | chat, Responses | none | chat not streamed live-pass, the rest pass (mock) <!-- matrix-live: openai.chat\|hf_router\|off --> <!-- matrix: openai.chat\|hf_router\|on, openai.responses\|hf_router\|off --> |
| `litellm`, `hivemind_gateway`, `pair_facade`, `custom_openai` | the operator-declared subset of the OpenAI surfaces | none | pass (mock) <!-- matrix: openai.chat\|litellm\|off, openai.chat\|hivemind_gateway\|on, openai.chat\|pair_facade\|off, openai.chat\|custom_openai\|on --> |
| `switchyard` | chat, Responses, Anthropic messages and count_tokens (all passthrough) | compact, input_tokens, decision, health; stats and metrics with `broker:admin` | pass (mock) <!-- matrix: openai.chat\|switchyard\|on, openai.responses\|switchyard\|on, anthropic.messages\|switchyard\|on, feature:native.switchyard --> |
| `ollama_cloud` | `/api/chat`, `/api/generate`, `/api/embed` (NDJSON streams) | `/api/show`, `/api/web_search`, `/api/web_fetch` | pass (mock) <!-- matrix: ollama.chat\|ollama_cloud\|off, ollama.chat\|ollama_cloud\|on, ollama.generate\|ollama_cloud\|on, ollama.embed\|ollama_cloud\|off --> |

## Surfaces and translation

<!-- claims-legend -->
Modes: **L** the unchanged local route answers; **L\*** the request is re-encoded as an OpenAI body and answered by
the local OpenAI route; **P** passthrough (the upstream speaks the inbound protocol natively: model rewritten,
credential swapped, bytes streamed through with a usage tee); **T** translated through one canonical intermediate
form, incrementally for streams; **na** refused with 400 `broker_capability_unavailable` before any egress (or the
refusal the matrix names). Columns use each adapter's reference profile: OAI `openai`, AZ `azure_openai`, ANT
`anthropic`, GEM `gemini`, XI `elevenlabs`, OLC `ollama_cloud`. "untried" in the Local column means no broker check
drove that local path; the existing route suites still cover the local handler itself.

| Inbound surface | Local | OAI | AZ | ANT | GEM | XI | OLC |
|---|---|---|---|---|---|---|---|
| `openai.chat` <!-- matrix: openai.chat\|local\|off=L, openai.chat\|openai\|off=P, openai.chat\|azure_openai\|off=P, openai.chat\|anthropic\|off=T, openai.chat\|gemini\|off=T, openai.chat\|ollama_cloud\|off=T --> <!-- matrix-na: openai.chat\|elevenlabs\|off=na --> | L | P | P | T | T | na | T |
| `openai.completions` <!-- matrix: openai.completions\|local\|off=L, openai.completions\|openai\|off=P, openai.completions\|azure_openai\|off=P, openai.completions\|anthropic\|off=T, openai.completions\|gemini\|off=T, openai.completions\|ollama_cloud\|off=T --> <!-- matrix-na: openai.completions\|elevenlabs\|off=na --> | L | P | P | T | T | na | T |
| `openai.responses` <!-- matrix: openai.responses\|local\|off=L, openai.responses\|openai\|off=P, openai.responses\|azure_openai\|off=T, openai.responses\|anthropic\|off=T, openai.responses\|gemini\|off=T, openai.responses\|ollama_cloud\|off=T --> <!-- matrix-na: openai.responses\|elevenlabs\|off=na --> | L | P | T | T | T | na | T |
| `openai.responses.retrieve` (also delete, cancel, input items) <!-- matrix: openai.responses.retrieve\|openai\|off=P --> <!-- matrix-na: openai.responses.retrieve\|local\|off=na, openai.responses.retrieve\|azure_openai\|off=na, openai.responses.retrieve\|anthropic\|off=na, openai.responses.retrieve\|gemini\|off=na, openai.responses.retrieve\|elevenlabs\|off=na, openai.responses.retrieve\|ollama_cloud\|off=na --> | na | P | na | na | na | na | na |
| `openai.embeddings` <!-- matrix: openai.embeddings\|local\|off=L, openai.embeddings\|openai\|off=P, openai.embeddings\|azure_openai\|off=P, openai.embeddings\|gemini\|off=T, openai.embeddings\|ollama_cloud\|off=T --> <!-- matrix-na: openai.embeddings\|anthropic\|off=na, openai.embeddings\|elevenlabs\|off=na --> | L | P | P | na | T | na | T |
| `openai.audio.speech` <!-- matrix: openai.audio.speech\|local\|off=L, openai.audio.speech\|openai\|off=P, openai.audio.speech\|azure_openai\|off=P, openai.audio.speech\|elevenlabs\|off=T, openai.audio.speech\|gemini\|off=T --> <!-- matrix-na: openai.audio.speech\|anthropic\|off=na, openai.audio.speech\|ollama_cloud\|off=na --> | L | P | P | na | T | T | na |
| `openai.audio.transcriptions` <!-- matrix: openai.audio.transcriptions\|local\|off=L, openai.audio.transcriptions\|openai\|off=P, openai.audio.transcriptions\|azure_openai\|off=P, openai.audio.transcriptions\|elevenlabs\|off=T, openai.audio.transcriptions\|gemini\|off=T --> <!-- matrix-na: openai.audio.transcriptions\|anthropic\|off=na, openai.audio.transcriptions\|ollama_cloud\|off=na --> | L | P | P | na | T | T | na |
| `openai.audio.translations` <!-- matrix: openai.audio.translations\|local\|off=L, openai.audio.translations\|openai\|off=P, openai.audio.translations\|azure_openai\|off=P, openai.audio.translations\|gemini\|off=T --> <!-- matrix-na: openai.audio.translations\|anthropic\|off=na, openai.audio.translations\|elevenlabs\|off=na, openai.audio.translations\|ollama_cloud\|off=na --> | L | P | P | na | T | na | na |
| `openai.images.generations` <!-- matrix: openai.images.generations\|local\|off=L, openai.images.generations\|openai\|off=P, openai.images.generations\|azure_openai\|off=P, openai.images.generations\|gemini\|off=T --> <!-- matrix-na: openai.images.generations\|anthropic\|off=na, openai.images.generations\|elevenlabs\|off=na, openai.images.generations\|ollama_cloud\|off=na --> | L | P | P | na | T | na | na |
| `openai.images.edits` <!-- matrix: openai.images.edits\|local\|off=L, openai.images.edits\|openai\|off=P, openai.images.edits\|azure_openai\|off=P, openai.images.edits\|gemini\|off=T --> <!-- matrix-na: openai.images.edits\|anthropic\|off=na, openai.images.edits\|elevenlabs\|off=na, openai.images.edits\|ollama_cloud\|off=na --> | L | P | P | na | T | na | na |
| `openai.moderations` <!-- matrix: openai.moderations\|openai\|off=P --> <!-- matrix-na: openai.moderations\|local\|off=na, openai.moderations\|azure_openai\|off=na, openai.moderations\|anthropic\|off=na, openai.moderations\|gemini\|off=na, openai.moderations\|elevenlabs\|off=na, openai.moderations\|ollama_cloud\|off=na --> | na | P | na | na | na | na | na |
| `openai.rerank` (query, documents, top_n, return_documents; Together, Fireworks, OpenRouter, LiteLLM, custom and the NIM ranking shape) <!-- matrix: openai.rerank\|local\|off=L, openai.rerank\|together\|off=P, openai.rerank\|nvidia_ai_endpoints\|off=P --> <!-- matrix-na: openai.rerank\|openai\|off=na, openai.rerank\|azure_openai\|off=na, openai.rerank\|anthropic\|off=na, openai.rerank\|gemini\|off=na, openai.rerank\|elevenlabs\|off=na, openai.rerank\|ollama_cloud\|off=na --> | L | na | na | na | na | na | na |
| `anthropic.messages` <!-- matrix: anthropic.messages\|local\|off=L, anthropic.messages\|openai\|off=T, anthropic.messages\|azure_openai\|off=T, anthropic.messages\|anthropic\|off=P, anthropic.messages\|gemini\|off=T, anthropic.messages\|ollama_cloud\|off=T --> <!-- matrix-na: anthropic.messages\|elevenlabs\|off=na --> | L | T | T | P | T | na | T |
| `anthropic.count_tokens` <!-- matrix: anthropic.count_tokens\|local\|off=L, anthropic.count_tokens\|anthropic\|off=P --> <!-- matrix-na: anthropic.count_tokens\|openai\|off=na, anthropic.count_tokens\|azure_openai\|off=na, anthropic.count_tokens\|gemini\|off=na, anthropic.count_tokens\|elevenlabs\|off=na, anthropic.count_tokens\|ollama_cloud\|off=na --> | L | na | na | P | na | na | na |
| `gemini.generateContent` <!-- matrix: gemini.generateContent\|local\|off=L*, gemini.generateContent\|openai\|off=T, gemini.generateContent\|azure_openai\|off=T, gemini.generateContent\|anthropic\|off=T, gemini.generateContent\|gemini\|off=P, gemini.generateContent\|ollama_cloud\|off=T --> <!-- matrix-na: gemini.generateContent\|elevenlabs\|off=na --> | L* | T | T | T | P | na | T |
| `gemini.streamGenerateContent` <!-- matrix: gemini.streamGenerateContent\|local\|on=L*, gemini.streamGenerateContent\|openai\|on=T, gemini.streamGenerateContent\|azure_openai\|on=T, gemini.streamGenerateContent\|anthropic\|on=T, gemini.streamGenerateContent\|gemini\|on=P, gemini.streamGenerateContent\|ollama_cloud\|on=T --> <!-- matrix-na: gemini.streamGenerateContent\|elevenlabs\|on=na --> | L* | T | T | T | P | na | T |
| `gemini.embedContent` <!-- matrix: gemini.embedContent\|local\|off=L*, gemini.embedContent\|openai\|off=T, gemini.embedContent\|azure_openai\|off=T, gemini.embedContent\|gemini\|off=P, gemini.embedContent\|ollama_cloud\|off=T --> <!-- matrix-na: gemini.embedContent\|anthropic\|off=na, gemini.embedContent\|elevenlabs\|off=na --> | L* | T | T | na | P | na | T |
| `gemini.batchEmbedContents` <!-- matrix: gemini.batchEmbedContents\|local\|off=L*, gemini.batchEmbedContents\|openai\|off=T, gemini.batchEmbedContents\|azure_openai\|off=T, gemini.batchEmbedContents\|gemini\|off=P, gemini.batchEmbedContents\|ollama_cloud\|off=T --> <!-- matrix-na: gemini.batchEmbedContents\|anthropic\|off=na, gemini.batchEmbedContents\|elevenlabs\|off=na --> | L* | T | T | na | P | na | T |
| `gemini.countTokens` <!-- matrix: gemini.countTokens\|local\|off=L, gemini.countTokens\|gemini\|off=P --> <!-- matrix-na: gemini.countTokens\|openai\|off=na, gemini.countTokens\|azure_openai\|off=na, gemini.countTokens\|anthropic\|off=na, gemini.countTokens\|elevenlabs\|off=na, gemini.countTokens\|ollama_cloud\|off=na --> | L | na | na | na | P | na | na |
| `ollama.chat` <!-- matrix: ollama.chat\|local\|off=L, ollama.chat\|openai\|off=T, ollama.chat\|azure_openai\|off=T, ollama.chat\|anthropic\|off=T, ollama.chat\|gemini\|off=T, ollama.chat\|ollama_cloud\|off=P --> <!-- matrix-na: ollama.chat\|elevenlabs\|off=na --> | L | T | T | T | T | na | P |
| `ollama.generate` <!-- matrix: ollama.generate\|local\|off=L, ollama.generate\|openai\|off=T, ollama.generate\|azure_openai\|off=T, ollama.generate\|anthropic\|off=T, ollama.generate\|gemini\|off=T, ollama.generate\|ollama_cloud\|off=P --> <!-- matrix-na: ollama.generate\|elevenlabs\|off=na --> | L | T | T | T | T | na | P |
| `ollama.embed` <!-- matrix: ollama.embed\|local\|off=L, ollama.embed\|openai\|off=T, ollama.embed\|azure_openai\|off=T, ollama.embed\|gemini\|off=T, ollama.embed\|ollama_cloud\|off=P --> <!-- matrix-na: ollama.embed\|anthropic\|off=na, ollama.embed\|elevenlabs\|off=na --> | L | T | T | na | T | na | P |
| `ollama.embeddings` (legacy) <!-- matrix: ollama.embeddings\|local\|off=L, ollama.embeddings\|openai\|off=T, ollama.embeddings\|azure_openai\|off=T, ollama.embeddings\|gemini\|off=T, ollama.embeddings\|ollama_cloud\|off=P --> <!-- matrix-na: ollama.embeddings\|anthropic\|off=na, ollama.embeddings\|elevenlabs\|off=na --> | L | T | T | na | T | na | P |
| `elevenlabs.tts` <!-- matrix: elevenlabs.tts\|local\|off=L, elevenlabs.tts\|openai\|off=T, elevenlabs.tts\|azure_openai\|off=T, elevenlabs.tts\|elevenlabs\|off=P --> <!-- matrix-na: elevenlabs.tts\|anthropic\|off=na, elevenlabs.tts\|gemini\|off=na, elevenlabs.tts\|ollama_cloud\|off=na --> | L | T | T | na | na | P | na |
| `elevenlabs.tts_stream` <!-- matrix: elevenlabs.tts_stream\|local\|on=L, elevenlabs.tts_stream\|openai\|on=T, elevenlabs.tts_stream\|azure_openai\|on=T, elevenlabs.tts_stream\|elevenlabs\|on=P --> <!-- matrix-na: elevenlabs.tts_stream\|anthropic\|on=na, elevenlabs.tts_stream\|gemini\|on=na, elevenlabs.tts_stream\|ollama_cloud\|on=na --> | L | T | T | na | na | P | na |
| `elevenlabs.stt` <!-- matrix: elevenlabs.stt\|local\|off=L*, elevenlabs.stt\|openai\|off=T, elevenlabs.stt\|azure_openai\|off=T, elevenlabs.stt\|elevenlabs\|off=P --> <!-- matrix-na: elevenlabs.stt\|anthropic\|off=na, elevenlabs.stt\|gemini\|off=na, elevenlabs.stt\|ollama_cloud\|off=na --> | L* | T | T | na | na | P | na |

Translation rules worth knowing:

- Errors are rendered in the **inbound** protocol's native envelope: OpenAI `{error:{message,type,param,code}}`,
  Anthropic `{type:"error",error:{type,message}}`, Gemini `{error:{code,message,status}}`, ElevenLabs
  `{detail:{status,message}}`, Ollama `{error:"..."}`. Errors of local targets stay exactly as before (OpenAI-shaped).
  <!-- matrix: envelope:broker_error\|openai, envelope:broker_error\|anthropic, envelope:broker_error\|gemini, envelope:broker_error\|elevenlabs, envelope:broker_error\|ollama -->
- On passthrough the provider's own native request fields pass through (Titanos-only fields `worker`, `location`,
  `timeout` and the local-engine knobs `require_loaded`, `include_provenance`, `vad_enabled`, `vad_model` and
  `vad_options` are always stripped). On translated paths every field goes through the profile's parameter map: mapped,
  dropped with a note in the decision, or refused with 400. `n>1`, `logprobs` and `logit_bias` are refused on every
  translated path. Local models keep their strict schemas, so the same field may be accepted for a cloud target and
  refused (422) for a local one.
- Anthropic requires `max_tokens`, so every translation into Anthropic sets it (the client's cap, else
  `TITANOS_BROKER_DEFAULT_MAX_OUTPUT_TOKENS`, at least 16000 for models whose thinking is always on) and stamps
  `X-Titanos-Injected: max_tokens`.
- A committed stream that fails mid-way ends with the inbound protocol's terminal frame (OpenAI `data:{error}` then
  `[DONE]`, Responses `response.failed`, Anthropic `event: error`, a Gemini `{error}` chunk, an Ollama `{"error"}`
  line); a binary audio stream is aborted so the client sees a truncated transfer. There is never a second upstream
  request after the first byte. <!-- matrix: envelope:mid_stream_terminal\|openai, envelope:mid_stream_terminal\|responses, envelope:mid_stream_terminal\|anthropic, envelope:mid_stream_terminal\|gemini, envelope:mid_stream_terminal\|ollama, envelope:mid_stream_terminal\|elevenlabs, feature:streaming.keepalive_and_disconnect -->
- Gemini and Ollama carry whole function calls, so their encoders release a tool call once its arguments are complete
  rather than per fragment. <!-- matrix: feature:streaming.whole_tool_calls --> Translated Gemini streams need
  `alt=sse`; the JSON-array stream form is refused with a typed error.
- Speech translation between OpenAI and ElevenLabs needs `voice_maps` in `broker.json`, or (OpenAI to ElevenLabs) a raw
  ElevenLabs voice id as the `voice` (8-64 letters and digits that are not an OpenAI voice name); otherwise the answer
  is 400 `broker_capability_unavailable`. `GET /v1/voices` lists the local voices first, then the upstream ElevenLabs
  voices for a key that may see the upstream (`cloud:<id>`), fetched with a catalog grant and cached for the catalog
  TTL; each upstream voice carries `titanos: {remote: true, upstream}`, and `GET /v1/voices/{voice_id}` finds either kind.
  <!-- matrix: elevenlabs.voices\|catalog, elevenlabs.voices\|catalog:cloud, elevenlabs.voice\|catalog -->
- A cloud image answered as base64 where the client asked for a URL is re-hosted as an absolute, owner-bound
  `/v1/files` URL; only PNG output can be re-hosted. Providers that answer their own image URLs (xAI, Together:
  profile extras `image_url_native`) serve a `url` or default request themselves, so nothing is re-hosted and their
  JPEG output is never refused. <!-- matrix: feature:images.rehosting, openai.images.generations\|xai\|off, openai.images.generations\|together\|off -->
- OpenAI image models never receive `response_format`: OpenAI's Images API refuses the field for every model (gpt-image
  models always did; a live call on 2026-10-02 showed dall-e-2 answering 400 `unknown_parameter` as well) and answers
  base64. A client `response_format=url` still gets re-hosted URLs; `b64_json` or no format gets the base64 unchanged
  (images/generations and images/edits). A second live call the same day answered 400 "The model 'dall-e-2' does not
  exist.", so the live image proof uses gpt-image-1-mini. Azure OpenAI deployments of dall-e models keep receiving the
  field; gpt-image deployments, which refuse it on Azure as well, never do.
- Field-by-field differences per adapter are in the broker sections of [PROTOCOL_DEVIATIONS.md](PROTOCOL_DEVIATIONS.md).

### OpenAI-compatible family

Every row of the OpenAI-compatible family is data in `broker/profiles.py`; the rules below come from each provider's
public reference and are exercised against a mock per provider (`tests/check_broker_compat_family.py`).

- Rerank: `POST /v1/rerank` is a typed surface. A local model name keeps Titanos' own API Plus rerank route; a cloud
  ref passes the body through to Together, Fireworks, OpenRouter, LiteLLM or a custom endpoint (each answers its own
  shape); the NIM ranking shape (the NVIDIA API catalog's second origin `nvidia_ai_endpoints`, and `nim_custom`'s
  `/ranking`) is translated: `query` and `documents` become `query.text` and `passages`, the rankings come back as
  `results` ordered by score with `top_n`, `return_documents` and `score_kind: cross_encoder_logit` applied.
  <!-- matrix: openai.rerank\|local\|off, openai.rerank\|together\|off, openai.rerank\|fireworks\|off, openai.rerank\|nvidia_ai_endpoints\|off -->
- Responses: xAI, Fireworks, LiteLLM and Azure OpenAI v1 serve `/v1/responses` natively and store responses, so `store`
  is forced false unless the caller holds `cloud:store` and the upstream allows storing; a stored response is then
  retrieved through its handle. Groq, OpenRouter and the Hugging Face router serve Responses statelessly.
  <!-- matrix: openai.responses\|xai\|off, openai.responses.retrieve\|xai\|off, openai.responses\|fireworks\|off, openai.responses\|groq\|off -->
- Reasoning: xAI reasoning models (`grok-4*`, `grok-3-mini*`, `grok-code*`, not `*non-reasoning*`) never receive
  `presence_penalty`, `frequency_penalty` or `stop` (dropped and named in `X-Titanos-Dropped`). A reasoning effort that
  Titanos adds (advisor reasoning mode) or translates (Anthropic thinking, Responses `reasoning.effort`, Gemini thinking)
  reaches only the models of the row's `reasoning_effort_globs` (for example Groq and Cerebras gpt-oss); a client's own
  `reasoning_effort` on a passthrough call is always forwarded. DeepSeek thinking mode needs every assistant turn's
  reasoning back in a tool loop, so translated calls to `deepseek` send the thinking as `reasoning_content`.
  <!-- matrix: openai.chat\|xai\|off, openai.responses\|cerebras\|off, anthropic.messages\|deepseek\|off -->
- Local-engine knobs: `repeat_penalty` becomes `repetition_penalty` on Together, Fireworks and OpenRouter (which also
  take `top_k` and `min_p`); strict rows (xAI, Groq, Mistral, DeepSeek, Cerebras; Perplexity for `repeat_penalty` and
  `min_p`) drop them with a note instead of letting the provider answer 400.
  <!-- matrix: openai.chat\|groq\|off, openai.chat\|together\|off -->
- Provider extensions pass through untouched on passthrough: Perplexity `citations`, `search_results` and the usage
  cost, OpenRouter `provider`, `models`, `transforms`, `plugins` and `usage.cost`, Groq `x_groq`, NVIDIA `nvext`,
  xAI `search_parameters`. <!-- matrix: openai.chat\|perplexity\|off, openai.embeddings\|openrouter\|off -->
- Audio: Groq and Together text-to-speech, Together and Mistral (Voxtral) transcription are typed surfaces; Voxtral's
  `usage.prompt_audio_seconds` is metered as audio seconds. Mistral speech answers JSON base64 and names the voice
  `voice_id`, so it is reached through the native passthrough. Fireworks speech-to-text lives on its own origin and is
  the `fireworks_audio` row. <!-- matrix: openai.audio.speech\|groq\|off, openai.audio.speech\|together\|off, openai.audio.transcriptions\|together\|off, openai.audio.transcriptions\|mistral\|off -->
- Errors: Groq's 498 (flex capacity exceeded) is 429 `upstream_rate_limited`; an NVIDIA cloud function's 202 Accepted
  (`NVCF-REQID`, no body) is 504 `upstream_timeout`, never an empty success. <!-- matrix: openai.chat\|groq\|off, openai.chat\|nvidia_api_catalog\|off -->
- Everything else each provider documents (Mistral FIM, OCR, classifiers, agents, conversations, batch jobs, files and
  speech; Together files, batches, fine-tunes, dedicated endpoints and videos; Groq files and batches; xAI image edits
  (JSON), videos, tokenizer and model listings; Perplexity Search and async chat; DeepSeek FIM and prefix completion;
  Fireworks image workflows and the account API; OpenRouter generation stats, credits and providers; the NVIDIA genai,
  visual and retrieval NIMs; Azure OpenAI v1 files, batches, vector stores and videos) is allowlisted for the native
  passthrough below.

## Native passthrough and WebSockets

Every provider endpoint that no typed surface covers is reachable through the guarded native passthrough, when the
profile's allowlist (`src/titanos/broker/profiles.py`, `NativeSpec`) names its method and path:

- `/native/<upstream>/<path>`: `<path>` is relative to the profile's native root, the upstream `base_url` for the
  OpenAI family (an OpenAI SDK uses `base_url=http://127.0.0.1:11435/native/openai`) and the provider origin for
  Anthropic, Gemini and ElevenLabs (their SDKs keep `/v1/...`, `/v1beta/...`, `/upload/v1beta/...` and `/v2/...`);
- the provider's own public path (for example `POST /v1/files`, `POST /v1/messages/batches`, `GET /v2/voices`,
  `POST /upload/v1beta/files`, `POST /v1beta/models/<model>:predict`), when it is not a Titanos route and exactly one
  enabled upstream the caller may see offers it. `X-Titanos-Upstream: <id>` names the upstream when several do (two
  OpenAI-family upstreams, for example) and makes a native path win over a Titanos route of the same shape such as
  `GET /v1/files/{id}`; the Anthropic SDK's `anthropic-version` and the vendor key headers pick their own family; and an
  owner-bound path whose id was created through Titanos by the caller (an OpenAI file id the key uploaded) reaches the
  provider without the header.

Rules: the caller needs the `native` scope plus `cloud` or `cloud:<upstream>` (`cloud` does not imply `native`);
endpoints that keep provider-side state (files, batches, vector stores, conversations, caches, voices) also need
`cloud:store` and the upstream's `data_policy.allow_store`; account-wide listings, fine-tuning and ephemeral token minting
need `broker:admin` for keys. Ids created through Titanos are bound to the calling key (another key gets 404). The
request body, query and allowlisted headers are forwarded untouched, with one exception: a model named as a Titanos ref
(`model@<upstream>` or an alias of it) in the body's `model`, an Anthropic batch's `requests[].params.model` or a
`{model}` path segment reaches the provider as its own id, and a ref of another upstream (or, on a bare path, a local
model name) is refused 400 before egress. `OpenAI-Organization` and `OpenAI-Project` are forwarded only for
`broker:admin` principals (and the keyless loopback owner), because they choose which organization or project the shared
credential acts for. Encoded separators (`%2F`, `%5C`, `%3F`, `%23`, `%2E%2E`), dot segments and empty segments (`//`)
inside a path parameter (checked again after a `{model}` segment was rewritten, so `..@<upstream>` is refused too) and a
percent-encoded `/native/<upstream>` prefix are refused 400, so the allowlisted row is always the path sent. A request
refused before anything is sent (these checks, an egress refusal such as 502 `egress_refused`, the deadline, the
upstream's concurrency slots) is never counted by the upstream's breaker, and neither is an answer refused because of
which provider object the caller asked for (a redirect, 502 `upstream_redirect_refused`, or a declared size over
`TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES`, 502 `upstream_response_too_large`, such as the key's own large file, video
content or dubbing audio); as on the typed surfaces, only an answer the provider got wrong (502
`upstream_bad_response`), a provider 5xx, a network error or a timeout counts. Nor can a key take the shared
credential offline by its choice of endpoint: a provider 401 marks the credential failed at once only when it carries
the provider's key-rejection code (OpenAI-compatible `invalid_api_key`, Anthropic `authentication_error`, Gemini
`API_KEY_INVALID`, the ElevenLabs key-rejection codes, Ollama `unauthorized`); OpenAI's 401 for a restricted or
project-limited key on an endpoint outside its permissions ("insufficient permissions ... Missing scopes") never marks
it; and any other 401 on a native or typed call is checked against the upstream's own probe endpoint (its models list,
at most one check per credential a minute), whose answer decides. One key therefore cannot take an
upstream offline for every other caller. The
upstream's own credential is attached only for its origin, the inbound `Authorization` and vendor key headers never
leave, the answer keeps the provider's status and body (errors scrubbed; a provider 401 becomes 502
`upstream_auth_failed`, except an ElevenLabs `quota_exceeded` 401, which is answered 402 with the scrubbed provider
body), and answers are capped by `TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES`. A rewritten Gemini resumable-upload URL
carries an upload capability (`titanos_upload`, 256 random bits bound to the caller, the upstream and that upload
session, spent at `finalize`, at most 24 hours, kept in memory), because google-genai sends its upload chunks without
any key; it authorizes POST and PUT on that session's path only. Each call records a decision with surface `native`;
state-changing calls and every WebSocket session are audited (`native.call`) before anything is sent. A budgeted key
needs a price row matching `native@<upstream>`: a character-billed provider (ElevenLabs) is reserved in characters (the
request body size), so a `per_1k_chars` row prices it and the provider's `character-cost` header settles it; for the
other providers the reservation is token-shaped (`input_per_mtok`) and stands as the spend because their usage is not
parsed. A 429 on the native lane of OpenAI, Anthropic or Gemini cools down the native lane only, never the typed
surfaces' models. Managed role refuses native calls unless delegated, and a delegated bare path must name its upstream.

A `switchyard` upstream offers its responses compact and input_tokens, decision and health endpoints under
`/native/<upstream>/` only (no bare paths: they would be ambiguous next to an OpenAI upstream); its stats, stats reset,
routing session stats and metrics need `broker:admin`. <!-- matrix: feature:native.switchyard -->

ElevenLabs with one key: the stock ElevenLabs SDK pointed at Titanos (`base_url=http://127.0.0.1:11435`, the `tk_` key as
its API key, sent as `xi-api-key`) reaches the typed text-to-speech, streaming text-to-speech and speech-to-text surfaces
and every allowlisted public endpoint on its own path: timestamps (`/with-timestamps`, `/stream/with-timestamps`),
dialogue, sound effects, voice changer, audio isolation, forced alignment, music, voice search (`/v2/voices`), voice
settings, cloning and design, dubbing, pronunciation dictionaries, history and the subscription (account rows need
`broker:admin`), single-use tokens (`broker:admin`) and the stream-input and realtime speech-to-text WebSockets.
- Bare model ids: on the ElevenLabs surfaces a model id such as `eleven_multilingual_v2` or `scribe_v1` resolves on the
  single enabled ElevenLabs upstream the caller may see that lists it (a live catalog adds `scribe_v1` and `scribe_v2`,
  which the provider's model list omits); with several such upstreams use `<model>@<upstream>`.
- Text-to-speech without `model_id` (convert and stream) goes to that upstream with the provider's default
  (`eleven_multilingual_v2`, decision note `model_absent:provider_default`) when no single local speech model would
  answer it; otherwise the local route answers as before.
- Every body field, `output_format`, `optimize_streaming_latency` and `enable_logging` are forwarded;
  `history-item-id`, `x-character-count` and the concurrency headers come back. Speech-to-text forwards every form field
  (repeated `keyterms` included) and `enable_logging`, never a single-use `token` query; `webhook=true` answers 202 and
  usage settles on the provider's `audio_duration_secs` (multichannel answers included).
- `GET /v1/models` sent with `xi-api-key` and no `Authorization` header is the provider's own model list (native scope);
  an OpenAI client keeps Titanos' catalog there.
- ElevenLabs also answers 401 for an exhausted quota (`quota_exceeded`, answered 503 `upstream_unavailable` on the typed
  surfaces and 402 with the scrubbed provider body on the native lane) and for a
  restricted-scope key without one endpoint's permission (`missing_permissions`, 502 `upstream_auth_failed`); neither
  takes the shared credential offline, and such a key still validates.

Gemini with one key: the stock google-genai SDK pointed at Titanos (`base_url=http://127.0.0.1:11435`, the `tk_` key as
its API key, sent as `x-goog-api-key`) reaches the typed surfaces and every allowlisted public endpoint on its own path.
A `?key=` query credential is refused (401 `query_credential_refused`), so a curl example copied from Google's guides
must send the `x-goog-api-key` header instead.
- `generateContent`, `streamGenerateContent` (`alt=sse`), `embedContent`, `batchEmbedContents` and `countTokens` forward
  every field. The REST API takes both field spellings (`generationConfig` and `generation_config`), and so does Titanos:
  a budgeted key's output cap and the advisor's reasoning mode go into the config object the body already uses. The
  thinking setting follows the model: `thinkingLevel` on gemini-3 and later (Pro models have no MEDIUM, so medium maps
  to HIGH), `thinkingBudget` on 2.5, and nothing on TTS, image-output and 2.0 models. A stream frame may be one
  multi-MiB image line; the first-frame hold and the usage tee read it whole.
- OpenAI-shaped bridges through `generateContent`: `/v1/audio/speech` to a TTS model (prebuilt voice names,
  `instructions` as the spoken style prefix, `wav` and `pcm`; any other format is answered as WAV and listed in
  `X-Titanos-Dropped`), `/v1/images/generations` and `/v1/images/edits` to an image model (`size` becomes the nearest
  aspect ratio and a 2K or 4K image size; `n` above 1, masks, `url` output and streaming are refused before egress), and
  `/v1/audio/transcriptions` and `/v1/audio/translations` to a model with audio understanding (`json`, `text` and
  `verbose_json` without segments; `srt` and `vtt` need timestamps and are refused). Studio speech and image jobs use
  the same bridges.
- Chat translations keep what the model makes: generated images arrive as `message.images` (data URLs) on chat
  completions and as `image_generation_call` items on Responses; code execution arrives as fenced text; grounding
  metadata and anything else a protocol cannot carry is listed in `X-Titanos-Dropped`. A chat request to a TTS model is
  refused before egress. Audio input parts (`input_audio`) travel as inline audio.
- The data policy counts Gemini file references (`fileData`), video and PDF parts as image (and video as audio too), so
  a text-only upstream policy refuses them.
- Native rows: Files (resumable upload, metadata, `files:register`, and the media download
  `/download/v1beta/files/<id>:download?alt=media`; the `/v1beta/files/<id>:download` form answers with a redirect to
  storage, which the egress never follows, so it is answered 502 `upstream_redirect_refused`), cached contents, batches,
  Interactions (create, also streamed; get, delete, cancel), File Search stores and their documents and operations,
  agents, environments and custom voices (provider-side state: `cloud:store`, owner-bound ids), Imagen `:predict`, Veo
  `:predictLongRunning` and its operations, webhooks, triggers, server-managed credentials and ephemeral `auth_tokens`
  (`broker:admin`), and the Live API (`v1beta` and `v1alpha`) and Lyria RealTime music WebSockets.
  `/native/<upstream>/v1beta/models` is Google's own model list; the bare `/v1beta/models` is Titanos' catalog.
- Catalog: each Gemini model carries its own capabilities (a TTS model offers speech, an image model images, an
  embedding model embeddings; Imagen, Veo and Live models are listed with no typed capability), and with a live catalog
  `/v1beta/models` passes the model's published `supportedGenerationMethods`, token limits and `thinking` through.
- Not served: the deprecated `tunedModels`, `corpora` and `generateAnswer` families, and the OpenAI-compatible
  `/v1beta/openai/` endpoint (the bridges above cover the same OpenAI shapes). Vertex AI is a separate, deferred profile.

Anthropic with one key: the stock Anthropic SDK pointed at Titanos (`base_url=http://127.0.0.1:11435`, the `tk_` key as
its API key, sent as `x-api-key`) reaches Messages, token counting, the model list and every allowlisted public endpoint
on its own path.
- `POST /v1/messages` and `/v1/messages/count_tokens` forward every field and the inbound `anthropic-beta` (server
  tools, MCP servers, citations, prompt caching with either TTL, `output_config`, `context_management`, `fallbacks`,
  `speed`, `inference_geo` ...). A bare model id such as `claude-opus-5-5` resolves on the single enabled Anthropic
  upstream the caller may see that lists it; with several such upstreams use `<model>@<upstream>`.
- Provider-side state named in a request (a `container` id, `custom` skills, a `file_id` in a document, image or
  `container_upload` block, also inside a Message Batch) needs `cloud:store` and the upstream's `data_policy.allow_store`
  (403 before anything is sent); a file uploaded through Titanos by another key is answered 404, as if it did not exist.
  Anthropic files are visible to the whole workspace behind the shared credential, so never hand file ids across keys.
- Translations into Anthropic (OpenAI chat, completions and Responses clients, Gemini and Ollama clients, Studio chat)
  follow the target model: a reasoning effort becomes `thinking: {type: "enabled", budget_tokens}` on 4.5 and older
  models and adaptive thinking plus `output_config.effort` on 4.6 and later (effort alone where thinking cannot be
  turned off; `none` sends `disabled`, or `between_tools` on Sonnet 5.5, where that is allowed). `temperature`, `top_p`
  and `top_k` are dropped on the models that refuse them (Opus 4.7 and later, Sonnet 5 and 5.5, Fable), and `top_p` is
  dropped when a 4.5 model would get both; `tool_choice` `required` or a named function becomes `auto` on Opus 5.5,
  Sonnet 5.5 and Fable 5.1, which refuse forced tool use. Every such change is a decision note and named in
  `X-Titanos-Dropped`. `json_schema` output maps to `output_config.format` (no beta header needed); `json_object` is
  refused. Where thinking is always on, an injected `max_tokens` is at least 16000. Signed thinking blocks (also the
  empty ones of the default `omitted` display) travel as `reasoning_content` / `reasoning_signature` and are replayed in
  tool loops. A model id the table does not know gets the rules of the newest models.
- Anthropic clients reaching other targets: adaptive thinking and `output_config.effort` become the target's reasoning
  effort, `output_config.format` its structured output, and plain-text documents become text.
- `GET /v1/models` and `/v1/models/{id}` sent with `anthropic-version` answer Titanos' catalog in Anthropic's shapes
  (`data`, `has_more`, `first_id`, `last_id`; `limit`, `after_id`, `before_id`), with the provider's `display_name`,
  `max_input_tokens`, `max_tokens` and `capabilities` for Anthropic models once the live catalog has been read.
- Native rows: Message Batches (create, list with `broker:admin`, retrieve, cancel, delete, results as JSONL; the
  `results_url` is rewritten to `/native/<upstream>/...` on this server, so the SDK's `results()` never sends the
  `tk_` key to Anthropic), Files (upload with `expires_in_seconds`, metadata, download of generated files, delete; the
  account-wide list needs `broker:admin`; `GET /v1/files/{id}` sent with the Anthropic SDK's headers and no
  `Authorization` reaches Anthropic, any other client keeps Titanos' artifact route there), Skills and their versions,
  Managed Agents (agents, sessions with their event streams, environments, vaults, deployments, memory stores; they run
  and bill at Anthropic outside Titanos metering, so `broker:admin` only) and the Admin API (`organizations/...`, under
  `/native/<upstream>/` only and with an admin credential on that upstream). `/native/<upstream>/v1/models` is
  Anthropic's own model list.
- Billing details from `usage` are kept: 1-hour cache writes (`cache_creation.ephemeral_1h_input_tokens`), web search
  requests (`server_tool_use.web_search_requests`) and fast mode (`speed`), priced with the row's
  `cache_write_1h_per_mtok`, `per_web_search_request` and `fast_multiplier` (see [Cost and prices](#cost-and-prices)). Code execution container
  time and fallback iterations are not metered separately.

OpenAI and Azure OpenAI with one key: the stock OpenAI SDK pointed at Titanos (`base_url=http://127.0.0.1:11435/v1`,
the `tk_` key as its API key) reaches every typed surface with every argument, and `base_url=.../native/<upstream>` (or
the bare `/v1/...` paths below) reaches the rest of the public API.
- Every body field passes on chat, completions, Responses, embeddings, speech, transcription, translation, images,
  image edits and moderations. The exceptions are named in `X-Titanos-Dropped`: Titanos' local sampling knobs (`top_k`,
  `min_p`, `repeat_penalty`) and VAD gate fields (`vad_enabled`, `vad_model`, `vad_options`), which OpenAI would refuse,
  and `temperature` / `top_p` on reasoning models (`o*`, `gpt-5*`), except `gpt-5-chat*` and requests with reasoning
  effort `none` (GPT-5.1 and later), which keep them. Azure OpenAI deployments of the same models follow the same rules.
- Streaming: `stream=true` on images/generations and images/edits relays the `image_generation.*` / `image_edit.*`
  events (`partial_images`; a whole multi-MiB partial image is one event), on audio/transcriptions the
  `transcript.text.delta` / `transcript.text.done` events (whisper-1 ignores the flag and answers JSON, as documented),
  `stream_format=sse` on audio/speech the `speech.audio.*` events, and `GET /v1/responses/{id}?stream=true&starting_after=N`
  resumes a stored background response. A stream that fails after its first event ends with OpenAI's
  `data: {"error": ...}` frame, which the SDK raises as `APIError`.
- `response_format=url` images are re-hosted as PNG on this server; with `output_format` `jpeg` or `webp`, or with
  `stream=true`, the request is refused before anything is sent (ask for `b64_json`). Image JSON answers (`n` base64
  images) are capped by `TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES`, not the JSON cap.
- A non-streamed call must produce its first byte within the upstream's `timeouts.first_byte` (120 s by default):
  high-effort reasoning, `service_tier: flex` and deep-research calls should stream, or the operator raises that
  timeout on the upstream (`background: true` is the provider's own answer, and needs `store`, below).
- `POST /v1/moderations` without `model` uses the provider's default (`omni-moderation-latest`) on the single enabled
  OpenAI upstream the caller may see that lists it (decision note `model_absent:provider_default`); with several, name
  the model.
- Errors keep the provider's meaning: a plain 403 (organization verification, region, model access) is passed through
  as 403 with the provider's message, `error.type` and `error.code` and never takes the credential offline; a 429 or
  400 `insufficient_quota` (or `billing_hard_limit_reached`, `billing_not_active`) is a billing stop, answered 503
  `upstream_unavailable` without `Retry-After`; 400 and 422 answers keep `error.code`, `error.type` and `error.param`.
- Usage: chat audio tokens, gpt-image input image tokens, gpt-4o-transcribe audio tokens, gpt-4o-mini-tts token usage
  (`stream_format=sse`) and the Responses `web_search_call` items are metered and priced at the row's
  `audio_input_per_mtok`, `audio_output_per_mtok`, `image_input_per_mtok` and `per_web_search_request` (see Cost and
  prices).
- Catalog: each OpenAI (and Azure deployment) model entry carries `titanos.model_capabilities`, the model family's own
  modalities (`gpt-4o-mini-tts`: speech, `whisper-1`: transcription and translation, `gpt-image-1`: images and image
  edits ...), next to the upstream's `capabilities`; routing still uses the upstream's capabilities.
- Native rows (`native` scope): files, uploads, batches, vector stores, conversations, containers, videos, custom voices
  and voice consents (provider-side state: `cloud:store`, owner-bound ids), Responses `compact` and `input_tokens`,
  stored chat completions, fine-tuning (also checkpoint permissions), evals and ChatKit (`broker:admin`), model
  deletion (`broker:admin`), the Realtime WebSocket, and the Realtime client secrets, WebRTC/SIP calls and legacy session
  mints, which let a client reach OpenAI directly past metering (`broker:admin`). Azure OpenAI's classic data plane
  offers files, batches, vector stores, fine-tuning, Responses (pass a preview `api-version` in the query when needed;
  the binding's version is added otherwise) and the Realtime WebSocket (`?deployment=<name>`) under
  `/native/<upstream>/` only. Retired endpoints (`images/variations`, Assistants and Threads) and the organization
  admin API are not allowlisted: 400 `broker_capability_unavailable` before egress.
- A bare OpenAI model id (`gpt-4o-mini`) resolves on the single visible OpenAI upstream that lists it (no alias
  needed); keyless callers need `anonymous_cloud` (connect turns it on for a keyless loopback owner); `store` is forced
  to false and named in `X-Titanos-Dropped` unless the key holds `cloud:store` and the upstream allows storage, and then
  `background` and `conversation` are refused 403 before egress (`previous_response_id` answers 404 naming the rule).

WebSocket endpoints (OpenAI Realtime `/native/<upstream>/realtime`, Gemini Live, the ElevenLabs stream-input and
realtime speech-to-text sockets) are relayed frame by frame through the same checks; the upstream connection is a
minimal RFC 6455 client inside the egress module. Accepting the inbound WebSocket needs uvicorn's WebSocket support:
`websockets` is a runtime dependency of the package (and is named in the frozen build), so a standard install accepts
upgrades; an environment without it (or `wsproto`) answers an upgrade 400 `broker_capability_unavailable` naming the
package, on `/native/...` and on bare WebSocket paths such as the OpenAI SDK's `/v1/realtime`. Keyed mode needs the `Authorization` header (browser WebSockets
cannot set it), keyless loopback refuses cross-site origins, a credential sent inside a client frame is removed for
profiles that list it, and only allowlisted subprotocols are forwarded.

## What leaves the host

Nothing leaves the host while the effective cloud policy is `off`, or for any request the router resolves to a local
target. <!-- matrix: role:standalone\|no_broker_json_no_egress, feature:disabled_lane --> When a request is served by
a cloud or router target, this is what the upstream receives:

| Surface | Sent to the upstream |
|---|---|
| Chat, completions, Responses, Anthropic messages, Gemini generate, Ollama chat and generate | The model id the upstream knows; messages or prompt text, system and developer text, tool definitions, tool calls and tool results; inline images (base64 data URLs; an `https` image URL is passed as a URL to Anthropic and refused for Gemini, Titanos never fetches URLs); sampling parameters after the profile map. `store` is forced to `false` for Responses and chat completions unless the key holds `cloud:store` and the upstream sets `data_policy.allow_store`. |
| Embeddings | The input texts and the operator's or caller's `input_type` (NVIDIA asymmetric models). |
| Speech, ElevenLabs text-to-speech | The text, voice, format and speed. |
| Transcription, translation, ElevenLabs speech-to-text | The uploaded audio file and its form fields, minus Titanos-only fields. |
| Image generation and edits | The prompt; for edits the uploaded image and mask. |
| Moderations, count_tokens | The input text or messages. |
| API Plus cloud stages | Audio for the ASR stage of audio transcription, translation and analysis (only after the local VAD gate; no speech means no egress); the text and fixed prompt for the sentiment and sentiment-composition LLM stages; the text and the versioned `broker-translation-v1` prompt for `/v1/translations`. |
| Asset jobs (`/api/jobs`, task `speech` or `image`) | The speech text or image prompt. |
| Advisors | Request counts, flags (images, tools, audio), approximate token counts and the candidate refs; at most 512 scrubbed characters of the prompt only when the advisor sets `share_excerpt`. Never credentials or upstream URLs. |

Always sent: the upstream's own credential (only to the origin it is bound to), `anthropic-version` and operator betas
for Anthropic, operator-set static headers the profile allows (for example `OpenAI-Organization`), `X-Titanos-Hop` to
self-hosted router and custom profiles (`nim_custom`, `litellm`, `hivemind_gateway`, `pair_facade`,
`custom_openai`) and `X-Request-ID` to `hivemind_gateway`. On passthrough the inbound `anthropic-beta`, `openai-beta`
and `accept` headers are forwarded.

Never sent: the inbound `Authorization`, `x-api-key`, `xi-api-key`, `x-goog-api-key` and cookies; Titanos keys
(`tk_...`) and the bootstrap key; any other `X-Titanos-*` header; local file paths, registry contents or other
upstreams' credentials. <!-- matrix: feature:egress.credential_isolation -->

## Routing, fallback and virtual models

The router runs once per request and decides everything before any egress: hop and loop checks, role switches, name
resolution, the key's visibility (404 before any policy 403), location, effective cloud policy, capability, data
policy, upstream readiness, breakers and admission (rate, concurrency and budget).
`titanos broker explain` and `POST /api/broker/explain` run the same steps as a dry run. <!-- matrix: feature:observability.explain -->

A virtual model expands to up to 16 targets with a strategy:

| Strategy | Order |
|---|---|
| `ordered` | As listed |
| `weighted` | Smooth weighted round-robin over targets whose breaker is closed |
| `cheapest` | Estimated price ascending, unpriced targets last |
| `lowest_latency` | Recent time-to-first-byte, fastest first |
| `advisor` | An advisor may reorder or narrow the deterministic order (see below) |

<!-- matrix: feature:routing.virtual_models -->

```sh
titanos broker virtual set chat-any --capability chat --strategy ordered \
  --target qwen3-8b --target gpt-4o-mini@openai --fallback-on queue_full,model_not_loaded,upstream_unavailable
```

Fallback rules:

- Fallback happens only **before the first byte reaches the client** and only on the typed triggers in `fallback_on`:
  `queue_full`, `rate_limit_error`, `model_not_loaded`, `worker_unavailable`, `model_load_failed`,
  `device_unavailable`, `upstream_rate_limited`, `upstream_unavailable`, `upstream_timeout`, `network`,
  `breaker_open`. A 4xx client error never falls back. <!-- matrix: feature:fallback.before_commit -->
- An upstream stream whose first event is an error counts as a failure before commit, so it can fall back.
- No fallback after an upload has been sent, for `store=true` Responses, for requests with `previous_response_id`,
  for handle-bound follow-ups or for asset jobs.
- `max_attempts` defaults to 3 for chat and embeddings virtual models and 1 for every other capability, and is
  capped at `1 + min(policy.max_fallbacks, TITANOS_BROKER_MAX_FALLBACKS)` (at most 5 fallbacks).
- With cloud policy `explicit` a virtual model keeps only targets of its first target's class (local, cloud or
  router); crossing classes needs `fallback`.

Breakers are kept per upstream and credential: after `TITANOS_BROKER_BREAKER_FAILURES` (3) consecutive server errors,
network errors or timeouts the upstream is skipped for an exponentially growing time with jitter, then one probe or
real request decides. 429 answers start a cooldown from `Retry-After` and 4xx answers are never counted.
<!-- matrix: feature:breakers, feature:e2e.limits_and_breakers -->

Each upstream has `max_concurrency` slots (default 16); when none frees within the first-byte timeout the answer is
429 `concurrency_exceeded`. <!-- matrix: feature:limits.upstream_concurrency -->

## Router upstreams and advisors

**Router upstreams** (`openrouter`, `hf_router`, `litellm`, `hivemind_gateway`, `pair_facade`, `switchyard`) choose a
provider themselves. Titanos delegates and does no balancing; answers carry `X-Titanos-Served-By: router:<id>`. Self-hosted
routers receive `X-Titanos-Hop: <16 hex characters of SHA-256 over this instance's id>`. An inbound request that carries
any hop header (`X-Titanos-Hop` or `X-HiveMind-Peer-Hop`) is restricted to local targets, and one carrying this
instance's own hop value is refused with 508 `loop_detected`. <!-- matrix: role:any\|hops --> A router that drops the hop
header but forwards `X-Request-ID` (Switchyard does) is caught too: while this instance is sending a request to a
router upstream, an inbound request carrying the same `X-Request-ID` is refused 508 `loop_detected` before routing, so
a route that leads back here fails on its first re-entry. A router that drops both headers makes loop detection
best-effort; the fallback cap bounds amplification.

**NVIDIA NeMo Switchyard** (`switchyard`, [SWITCHYARD.md](SWITCHYARD.md)) is a local routing proxy that speaks OpenAI
Chat Completions, OpenAI Responses and Anthropic Messages on one origin (`base_url` `http://127.0.0.1:4000`, no `/v1`).
`titanos broker connect switchyard --no-credential` adds it; clients call `<route-id>@<upstream>`. The row passes
Anthropic Messages and count_tokens through natively instead of re-encoding them as chat, forwards the harness session
headers Switchyard routes on (on this row only), returns the backend Switchyard picked in
`x-model-router-selected-model` (kept as the decision note `router_selected:<model>`), keeps one circuit breaker per
route and forces `store: false` on Responses unless storing is allowed. It sends no credential unless the upstream sets
`auth: bearer` (a proxy in front of Switchyard). <!-- matrix: feature:router.switchyard, feature:breakers.per_route -->
Its live catalog carries each route's tool calling, vision and context window: `GET /v1/models` shows them, and tools or
images sent to a route that declares them false are refused before egress. <!-- matrix: feature:router.catalog_capabilities -->
Switchyard drops `X-Titanos-Hop` towards its own backends, so a deployment whose targets point back at Titanos sets
`extra_headers = { "X-Titanos-Hop" = "switchyard" }` on those clients; the other direction (Titanos as a Switchyard
target, including the identity and cache fields the local routes accept and name in `X-Titanos-Ignored`) is in
[SWITCHYARD.md](SWITCHYARD.md). <!-- matrix: feature:local.compat_fields -->

**Advisors** suggest an order; they never add a target or bypass a scope. Contract `titanos.router-advisor/v1`
(profile `generic-v1`): Titanos POSTs

```json
{"contract": "titanos.router-advisor/v1", "request_id": "...", "surface": "openai.chat", "capability": "chat",
 "requested_model": "chat-any",
 "hints": {"location": "auto", "latency_class": null, "max_cost_usd": null},
 "summary": {"message_count": 3, "approx_input_tokens": 420, "has_images": false, "has_tools": true,
             "has_audio": false, "excerpt": null},
 "candidates": [{"ref": "qwen3-8b", "class": "local", "upstream": null, "capabilities": ["chat"],
                 "breaker": "closed", "price": null, "p50_ttfb_ms": 180}]}
```

and accepts

```json
{"contract": "titanos.router-advisor/v1", "choice": "qwen3-8b", "fallbacks": ["gpt-4o-mini@openai"],
 "reasoning_mode": null, "confidence": 0.8, "explanation": "short reason"}
```

The answer is kept only when the contract string matches and the choice and every fallback are among the candidates.
A slow (default 300 ms, `TITANOS_BROKER_ADVISOR_TIMEOUT_MS`), invalid or failed answer leaves the deterministic order.
`reasoning_mode` maps to the target's reasoning field (`reasoning_effort`, Anthropic thinking shaped for the model: a
thinking budget on 4.5 and older, adaptive thinking and `output_config.effort` on 4.6 and later; or a Gemini
`thinkingBudget`) where the target accepts one. <!-- matrix: feature:advisor, feature:reasoning_injection -->

Other advisor profiles:

- `hivemind-gim-router` calls a HiveMind GIM router's `/gim/llm_router/func_1` and maps its `simple`, `complex` and
  `reasoning` labels to candidate refs through `label_map`, including its fallback chain. It needs the prompt, so it is
  consulted only with `share_excerpt: true`.
- `pending-contract` is a neutral, named slot for an external router or advisor whose wire contract has not been
  published yet (NVIDIA NeMo Switchyard, whose contract is public, has its own `switchyard` profile). The operator names it in private configuration (`advisors.<id> = {"profile": "pending-contract",
  "label": "..."}` or an upstream with profile `pending_contract`). It is listed with status `contract_unknown` and
  is never called; giving it a URL is refused with 409 `advisor_profile_unconfigured`. When the contract is supplied it
  becomes an advisor class or a router profile row, with recorded fixtures, a mock and matrix cells.

Advisors and router upstreams are off in managed role unless the parent delegates a request (and even then only an
explicit target is allowed). <!-- matrix: role:managed\|sealed_no_probes -->

## Keys, scopes, budgets and audit

Without `TITANOS_API_KEY` and without any key record, Titanos stays a keyless loopback server exactly as before:
presented placeholder keys are ignored, only loopback host names are accepted and cross-site browser requests are
refused. <!-- matrix: feature:inbound_auth.keyless, feature:inbound_auth.host_guard --> With the bootstrap key
(`TITANOS_API_KEY`, scope `admin`) or any minted key, every non-public route needs a key. Details, formats and the
full scope list are in [BROKER_SECURITY.md](BROKER_SECURITY.md#inbound-keys).

- Keys are `tk_<id>_<secret>`, shown once at mint, stored as SHA-256 only, and can expire, be revoked or rotated
  without a restart. <!-- matrix: feature:keys.lifecycle -->
- Accepted credential headers (vendor headers also on `/native/*` and on each vendor's bare native paths):
  `Authorization: Bearer` everywhere, `x-api-key` on `/v1/messages*` and `/v1/models*`, `xi-api-key` on the
  ElevenLabs routes and `x-goog-api-key` on `/v1beta/*`. `?key=` is refused with 401 `query_credential_refused` in
  every mode. <!-- matrix: feature:inbound_auth.vendor_headers, feature:inbound_auth.query_key_refused -->
- Scopes are checked before the body is read. Cloud use needs `cloud` or `cloud:<upstream>`; provider-side storage
  needs `cloud:store`. <!-- matrix: feature:inbound_auth.scopes -->
- Per-key limits (`--rpm`, `--tpm`, `--concurrent`) answer 429; budgets (`--usd-per-day`, `--usd-per-month`,
  `--tokens-per-day`, `--max-request-usd`) answer 402 `budget_exceeded` **before** any upstream request.
  <!-- matrix: feature:limits.rate, feature:limits.budgets -->
- In broker role every key with an egress scope needs a budget and an expiry at most 365 days away.
  <!-- matrix: role:broker\|egress_key_rules -->
- Every key, credential, upstream, alias, policy, advisor and price change, every reload, auth failure, budget stop,
  breaker opening and role start is written to a hash-chained audit log under `TITANOS_HOME/broker/audit/`.
  `titanos audit verify` and `GET /api/audit/verify` check the chain. <!-- matrix: feature:audit.chain -->

## Cost and prices

Titanos ships no prices. Prices are operator data in `broker.json` `prices` (or `titanos broker prices import
FILE.json`), as decimal strings; a running server applies a change after a restart or a reload (see [Applying changes
to a running server](#applying-changes-to-a-running-server)):

```json
{"match": "gpt-4o-mini@openai", "currency": "USD", "input_per_mtok": "0.15", "output_per_mtok": "0.60",
 "cached_input_per_mtok": null, "cache_write_per_mtok": null, "per_1k_chars": null, "per_audio_minute": null,
 "per_image": null, "source": "operator", "effective_from": "2026-10-01"}
```

`match` is a glob over refs; the most literal match wins, then the fewest `*`, then the latest `effective_from`.
Cached input falls back to the input price and Anthropic cache writes use `cache_write_per_mtok` when set. Anthropic
rows may also set `cache_write_1h_per_mtok` (1-hour cache writes; the cache-write price when absent),
`per_web_search_request` and `fast_multiplier` (token prices of a fast-mode answer); an answer that reports web searches
or fast mode without the matching price stays unpriced, so the reservation stands as the spend. OpenAI rows may set
`audio_input_per_mtok`, `audio_output_per_mtok` and `image_input_per_mtok`: the audio and image token parts of a usage
(chat audio input and output, gpt-4o-transcribe and gpt-4o-mini-tts audio tokens, gpt-image input images) are billed at
those prices and the rest at the text prices (the text price when absent, so a row without them bills every token as
before); hosted web searches of a Responses call (`web_search_call` output items) use `per_web_search_request`.
<!-- matrix: feature:accounting.pricing -->

Budgets use the same table. Before dispatch the estimated cost (input bytes / 3 as tokens, plus the output cap; an
audio upload's duration, measured from WAV, FLAC or MP3 headers and otherwise its size at 1000 bytes per second; on the
Anthropic Messages surfaces each base64 image counts 4800 tokens, a base64 document its decoded bytes / 3, and the input
never more than 1,000,000 tokens, the largest Claude context window) is reserved; the actual usage settles it afterwards. A budgeted key calling an unpriced target is refused with 402
`budget_unpriced_target`. When a budgeted request on a token-generating surface (chat, completions, Responses,
Anthropic Messages, Gemini `generateContent` and `streamGenerateContent`, Ollama chat and generate) names no output cap,
`TITANOS_BROKER_DEFAULT_MAX_OUTPUT_TOKENS` (4096) is injected in that protocol's own field and stamped
`X-Titanos-Injected: max_tokens`; images, speech, transcription, translation, embeddings, moderations and token
counting never get one. Settlement prices one basis and records it as `price_basis` in the decision and the usage
record: `tokens` when the provider reports token usage and the row has token prices, otherwise the reported
characters, images or audio seconds at the row's unit prices (`audio_seconds:provider`, or `audio_seconds:upload`, the
measured upload, when the provider reports no seconds). Only when nothing can be priced does the reservation stand as
the spend (`reservation`, decision note `cost:reserved_estimate`), so a budgeted key never gets a free or an
unexplained charge. The overshoot of a budget is bounded by the key's
concurrent requests times the per-request cost; see [BROKER_SECURITY.md](BROKER_SECURITY.md#residual-risks). Local
inference costs nothing and is only rate-limited.

## Provenance headers

When the broker is configured, or the caller is a key, every inference answer (including streams and errors) carries:

| Header | Value |
|---|---|
| `X-Request-ID` | The client's value when it is at most 128 safe characters, else a managed parent's trace id, else a new id |
| `X-Titanos-Served-By` | `local:engine`, `local:native`, `cloud:<upstream>` or `router:<upstream>` |
| `X-Titanos-Requested-Model` | The model name the client sent |
| `X-Titanos-Served-Model` | The target ref that answered |
| `X-Titanos-Fallback-Count` | Fallbacks taken before the first byte |
| `X-Titanos-Translation` | `passthrough`, `local`, `native` or `<inbound>-><target>` |
| `X-Titanos-Dropped` | The request fields a translated path (or a parameter map) dropped, comma-separated; every one is also a `dropped:<field>` decision note |
| `X-Titanos-Decision-Id` | The decision record id |
| `X-Titanos-Upstream-Request-ID` | The provider's request id, when it sent one |
| `X-Titanos-Injected` | `max_tokens` when Titanos set an output cap |
| `X-Titanos-Error-Origin` | On errors: `titanos`, `policy`, `upstream`, `network` or `local` |

<!-- matrix: feature:observability.provenance -->

Upstream rate-limit headers (`x-ratelimit-*`, `anthropic-ratelimit-*`) pass through, and so does a `switchyard`
upstream's `x-model-router-selected-model`. A local route that accepted standard client fields without acting on them
(`user`, `metadata`, `prompt_cache_key` and the like) names them in `X-Titanos-Ignored`
([SWITCHYARD.md](SWITCHYARD.md)). <!-- matrix: feature:local.compat_fields --> `X-Titanos-Upstream-URL` is added
only with `TITANOS_BROKER_DEBUG_UPSTREAM_URLS=true` and an admin caller. Internal URLs and peer topology are never
stamped. In passthrough the body's `model` is the upstream's value; in translated mode and for local aliases it is the
requested name, so SDK clients see what they asked for. MCP inference results carry the same facts in `_meta`
(`served_by`, `served_model`, `request_id`, `decision_id`, `fallback_count`).

Request hint headers (never forwarded upstream): `X-Titanos-Upstream` (native passthrough), `X-Titanos-Location`,
`X-Titanos-Target` (scope `route:explicit`),
`X-Titanos-Delegate: cloud` (managed role, scope `delegate`), `X-Titanos-Deadline-Ms` and `X-Request-ID`.

## Error codes

Codes come from one table (`src/titanos/broker/errors.py`); the envelope follows the inbound protocol. Rows marked
"no retry" also carry `x-should-retry: false`, because the OpenAI and Anthropic SDKs otherwise retry 408, 409, 429 and
every 5xx. <!-- matrix: error:no_retry_headers\|openai_compat, error:no_retry_headers\|anthropic, error:no_retry_headers\|gemini, error:no_retry_headers\|elevenlabs, error:no_retry_headers\|ollama_native -->
Anthropic types follow the status (400 `invalid_request_error`, 401 `authentication_error`, 402 and 403
`permission_error`, 404 `not_found_error`, 413 `request_too_large`, 429 `rate_limit_error`, 503 `overloaded_error`,
others `api_error`); Gemini statuses likewise (`INVALID_ARGUMENT`, `UNAUTHENTICATED`, `PERMISSION_DENIED`,
`NOT_FOUND`, `ABORTED`, `RESOURCE_EXHAUSTED`, `CANCELLED`, `UNAVAILABLE`, `DEADLINE_EXCEEDED`, `INTERNAL`).

| Code | HTTP | No retry | When |
|---|---|---|---|
| `model_required` | 400 | | Blank model |
| `invalid_request` | 400 | | A ref over 256 characters, a malformed hint header, or an unparseable body from a key |
| `ambiguous_credentials` | 400 | | Two different credentials in one request |
| `location_unsatisfiable` | 400 | | Nothing survives the location filter (for example `local_only` with only cloud targets) |
| `broker_capability_unavailable` | 400 | | The target cannot serve this surface or field (checked before egress) |
| `provenance_unavailable_for_cloud` | 400 | | `include_provenance` on a cloud translation stage |
| `broker_config_contains_secret` | 400 | | A broker document holds a secret-shaped string |
| `idempotency_key_required` | 400 | | A cloud asset job without `Idempotency-Key` |
| `budget_required` | 400 | | Broker role: a key with an egress scope without budget and expiry |
| `authentication_error` | 401 | | Wrong or missing key (keyed mode), also from loopback |
| `key_expired` | 401 | | The key is past `expires_at` |
| `key_revoked` | 401 | | The key was revoked |
| `query_credential_refused` | 401 | | `?key=` on `/v1beta` |
| `budget_exceeded` | 402 | yes | The reservation does not fit the remaining budget |
| `budget_unpriced_target` | 402 | yes | A budgeted key named a target without a price row |
| `scope_denied` | 403 | yes | The route or hint needs a scope the key lacks |
| `cloud_disabled` | 403 | yes | The effective cloud policy is `off` for a cloud ref the caller could see, or a keyless loopback caller names a cloud model while `policy.anonymous_cloud` is off |
| `role_forbids` | 403 | yes | The role switch table refuses (managed role without delegation); in managed role a reload that would enable egress, credential writes and upstream tests answer 409 with this code |
| `delegation_required` | 403 | yes | Managed role with delegation configured, but the request is not delegated or names no explicit target |
| `transport_insecure` | 403 | yes | Key or credential write over plain HTTP from a non-loopback peer |
| `keyless_mode_refused` | 403 | yes | Key or credential write in keyless mode, or a keyless change of `anonymous_cloud` or `allow_http_lan` |
| `data_policy_denied` | 403 | yes | The upstream's `data_policy` excludes a modality in the body |
| `model_not_found` | 404 | | Unknown model, one the caller may not use, or a model the provider reports unavailable (404) or retired (410) |
| `response_not_found` | 404 | | A Responses id the caller does not own, or unknown |
| `not_found` | 404 | | Other unknown resources |
| `broker_name_conflict` | 409 | yes | An alias or virtual name collides with a local or broker name |
| `advisor_profile_unconfigured` | 409 | yes | A URL on the `pending-contract` slot |
| `last_key_required` | 409 | yes | Revoking the last admin key on a network bind without a bootstrap key |
| `registry_busy` | 409 | | Registry reload while requests (including long cloud streams) are active |
| `request_too_large` | 413 | | Body over `TITANOS_MAX_BODY_BYTES` |
| `credential_validation_failed` | 422 | | The validation probe failed (status class only, scrubbed) |
| `rate_limit_exceeded` | 429 | | The key's RPM or TPM limit |
| `concurrency_exceeded` | 429 | | The key's or upstream's concurrency slots |
| `upstream_rate_limited` | 429 | | The provider answered 429 (its `Retry-After` is kept), or the model is still cooling down from that 429 |
| `auth_throttled` | 429 | | Too many failed credential claims from one address |
| `cancelled` | 499 | | The client disconnected |
| `upstream_auth_failed` | 502 | yes | The provider refused the credential (never surfaced as the client's own 401) |
| `upstream_bad_response` | 502 | | Malformed JSON or SSE, a parser limit, or invalid tool-call JSON |
| `egress_refused` | 502 | yes | Origin mismatch, a refused resolved address or the egress mode |
| `upstream_redirect_refused` | 502 | yes | The provider answered 3xx (redirects are never followed) |
| `upstream_response_too_large` | 502 | yes | The provider's answer exceeded the response cap |
| `stream_interrupted` | 502 | | In-stream terminal type after commit |
| `upstream_unavailable` | 503 | | Provider 5xx or a network error |
| `breaker_open` | 503 | | Every target's breaker is open (`Retry-After` is the earliest half-open) |
| `credential_unavailable` | 503 | yes | Missing env or file value, refused file ACL, or unvalidated after a base URL change |
| `vault_unavailable` | 503 | yes | No usable vault backend for a stored secret |
| `vault_sealed` | 503 | yes | Managed role sealed the vault |
| `classifier_unavailable` | 503 | yes | Moderation on a non-OpenAI or local target |
| `upstream_unconfigured` | 503 | yes | The `pending_contract` slot, or an enabled upstream without a credential |
| `upstream_timeout` | 504 | | The provider did not answer in time |
| `deadline_exceeded` | 504 | | The client or parent deadline expired |
| `loop_detected` | 508 | yes | The inbound hop equals this instance's hop value, or the inbound `X-Request-ID` is one this instance is currently sending to a router upstream |

Provider answers are mapped as follows: 400, 404, 413 and 422 pass through in the inbound envelope after scrubbing; a
provider 410 for a model (410 Gone is how the NVIDIA API catalog answers for a retired model), and a 404 whose provider
code or message is about the model (OpenAI `model_not_found`, Azure `DeploymentNotFound`, Anthropic `not_found_error`
naming the model, Gemini `NOT_FOUND` naming `models/...`, NVIDIA and other OpenAI-compatible messages naming the model)
become 404 `model_not_found` with origin `upstream` and a message saying the provider reports the model unavailable, or
retired for 410, in every inbound envelope; a 404 about anything else (an ElevenLabs `voice_not_found`, a missing file
or response id, a bare 404) stays 404 `not_found` with the provider's scrubbed message. Neither is ever retried,
counted by the breaker or allowed to mark the credential;
401 and a key-rejecting 403 become 502 `upstream_auth_failed`; 408 and timeouts become 504 `upstream_timeout`; 429 becomes 429
`upstream_rate_limited`; 500, 502, 503, 504 and 529 become 503 `upstream_unavailable`; 3xx becomes 502
`upstream_redirect_refused`. <!-- matrix: error:client_4xx_scrubbed\|openai_compat, error:upstream_auth\|openai_compat, error:server_error\|openai_compat, error:redirect_refused\|openai_compat, error:rate_limited\|openai_compat, error:network_and_timeout\|openai_compat -->
A provider 401 marks the credential as failed; a 403 does so only when the provider body says the key is invalid.
On a call a client made (any typed surface or native row, so the client chose the endpoint), a 401 marks it at once
only with the provider's key-rejection code (OpenAI-compatible `invalid_api_key`, Anthropic `authentication_error`,
Gemini `API_KEY_INVALID`, the ElevenLabs key-rejection codes, Ollama `unauthorized`); any other 401 is checked against
the upstream's probe endpoint (its models list, at most once a minute per credential) and marks the credential only if
that answer rejects it too; an upstream whose profile has no probe endpoint is not marked by such a 401. OpenAI's 401
for a restricted or project-limited key outside its permissions ("insufficient permissions ... Missing scopes") never
marks the credential, also on the catalog and probe calls.
ElevenLabs is the exception for 401: only its key-rejection codes (`invalid_api_key`, `needs_authorization` and the like)
mark the credential; `quota_exceeded` becomes 503 `upstream_unavailable` (402 with the provider's scrubbed body on the
native lane) and `missing_permissions` stays 502 without a
mark. A plain 403 in the provider's own envelope that is not a key rejection (OpenAI organization verification,
unsupported region or no access to a model; Anthropic `permission_error`; Gemini `PERMISSION_DENIED`; an ElevenLabs
permission detail) passes through as 403 with the provider's message and code (OpenAI-shaped clients also keep
`error.type`), never a fallback trigger and never a credential mark. While a provider 429 cools a model down (per model
on OpenAI, Anthropic and Gemini; per route on Switchyard), a request for that model answers 429 `upstream_rate_limited`
with the remaining `Retry-After` and zero egress; other models on the same key keep serving. The OpenAI-compatible
family also treats a 429 or 400 whose code or type is
`insufficient_quota`, `billing_hard_limit_reached` or `billing_not_active` is a billing stop like a 402, 503
`upstream_unavailable` without `Retry-After`. OpenAI-shaped clients keep the provider's `error.param` on every passed
through 4xx.

## Observability

| Need | HTTP | CLI | MCP |
|---|---|---|---|
| Summary (role, effective policy, egress mode, vault, upstreams, audit) | `GET /api/broker` | `titanos broker status` | `broker_status` |
| Where would this run? (zero egress) | `POST /api/broker/explain` | `titanos broker explain` | `broker_explain` |
| Recent decisions (metadata, never prompts) | `GET /api/broker/decisions` | `titanos broker decisions` | `broker_decisions` |
| Usage and cost rollups | `GET /api/broker/usage?group_by=key,upstream,model,surface,day` | `titanos broker usage` | `broker_usage` |
| Counters (requests, fallbacks, breaker transitions, time-to-first-byte buckets, egress bytes, auth failures, budget stops) | `GET /api/broker/metrics` | | |
| Catalog the caller may use | `GET /api/broker/catalog?refresh=false` | `titanos broker catalog` | `broker_catalog` |
| Upstream health and breaker | `GET /api/broker/upstreams/{id}/health` | `titanos broker upstream health ID --url U` | `upstream_health` |
| Own principal | `GET /api/keys/self` | `titanos keys whoami --url U` | `whoami` |
| Audit | `GET /api/audit`, `GET /api/audit/verify` | `titanos audit`, `titanos audit verify` | none |

<!-- matrix: feature:observability.decisions, broker.catalog\|catalog, feature:mcp.tools -->

A `broker:read` key sees only its own decisions and usage; `broker:admin` sees all. One decision is recorded per
request, including refusals, in a ring of `TITANOS_BROKER_DECISION_LOG_SIZE` entries (also written to
`TITANOS_HOME/broker/decisions/` as JSONL with `TITANOS_BROKER_DECISION_LOG_FILE=true`). Usage records are written per
UTC day for served requests only; failed upstream calls appear in the decision log, not in usage rows. The server logs
one INFO line per broker request without bodies (`broker request_id=... decision=... key=... surface=... served_by=...
status=... ms=... fallback=...`). `GET /api/info` gains `role`, `orchestrated` and a `broker` block without URLs or
secrets.

## Limits

- **Registry reloads during cloud streams.** A long cloud stream counts as active work, so a registry reload answers 409
  `registry_busy` until it ends.
- **Concurrency.** `TITANOS_LIMIT_CONCURRENCY` (64) bounds all HTTP requests including open streams; broker role warns
  below 64. Upstream slots default to 16.
- **MCP.** Tool schemas stay the strict local schemas: model names are at most 128 characters (use a published alias or
  a short ref), provider-native extra fields cannot be passed, and results are not streamed. The tools'
  `open_world_hint` is fixed when the server starts; a broker reload does not change it.
- **ElevenLabs model listing.** Inbound `GET /v1/models` is the OpenAI catalog; the ElevenLabs list is returned there
  only for a request with `xi-api-key` and no `Authorization` (the ElevenLabs SDK) from a key holding `native`, and
  always at `/native/<upstream>/v1/models`.
- **Retrieval answers** refuse cloud LLMs with a typed 400 and zero upstream calls: exact context accounting and
  checkpoint provenance need a local native model. <!-- matrix-na: plus.retrieval_answer\|cloud -->
- **Local-only stages** (OCR, depth, detection, segmentation, VAD, diarization, pair classification, rerank) answer
  `cloud_only` with 400 `broker_capability_unavailable`. <!-- matrix-na: plus.local_only_stages\|cloud -->
- **Cloud asset jobs** cover speech and image only; video and music stay local.
  <!-- matrix: jobs.create\|cloud:speech, jobs.create\|cloud:image --> <!-- matrix-na: jobs.create\|cloud:video, jobs.create\|cloud:music -->
- **Moderation** has no local classifier: a local target answers 503 `classifier_unavailable`, never `flagged: false`.
  <!-- matrix-na: openai.moderations\|local\|off -->
- **Gaps still open in the matrix**: the macOS Keychain and Linux Secret Service vault backends are untried on the
  Windows host that generated the matrix ([BROKER_SECURITY.md](BROKER_SECURITY.md#credential-storage)), and only nine
  cells are live-pass (OpenAI and the Hugging Face router, 2026-10-02); NVIDIA API catalog inference was refused for
  the account and every other provider is mock-validated only. <!-- matrix-untried: feature:vault.keychain, feature:vault.secret_service --> <!-- matrix-count: live-pass=9 -->
  The Studio panel, `/v1/extensions/infer` with cloud stages and the merged cloud listings of `/api/tags` and
  `/v1beta/models` are now driven by checks. <!-- matrix: feature:studio.providers_panel, plus.extensions_infer\|cloud, ollama.tags\|catalog:cloud, gemini.models\|catalog:cloud -->
- **Overhead.** `python tools/broker_overhead.py` measures the BrokerDispatch cost for local chat in-process against a
  fake engine. On one Windows development host the inactive path added about 0 ms and the active path about 3 ms
  median, about half of it the per-request log line (`--no-request-log` separates it). These are observations from one
  machine, not guarantees.
