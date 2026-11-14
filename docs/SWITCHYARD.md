# NVIDIA NeMo Switchyard and Titanos

[NVIDIA NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) (Apache-2.0) is a routing proxy for LLM calls. A
TOML deployment defines LLM clients, targets and routes, and `switchyard-server` serves OpenAI Chat Completions, OpenAI
Responses and Anthropic Messages on one origin; clients send a route id as the model name. Titanos works with it in
both directions:

- **Titanos -> Switchyard**: a Titanos upstream of profile `switchyard` (a router upstream) sends `<route-id>@<upstream>`
  requests to Switchyard, which picks the backend.
- **Switchyard -> Titanos**: Switchyard targets point at Titanos, so a route can pick a Titanos local model or a
  Titanos-brokered cloud model.

Both directions were run against nemo-switchyard 0.3.0 (the release of 2026-09-22) with the real interop check
`tests/check_switchyard_interop.py` (below). The broker matrix records the mock-validated cells of the `switchyard` row
([BROKER_MATRIX.md](BROKER_MATRIX.md)); [BROKER.md](BROKER.md) is the broker reference.

## Install Switchyard

Use a separate virtual environment (never the Titanos one):

```
python -m venv switchyard-venv
switchyard-venv/bin/python -m pip install nemo-switchyard          # Windows: switchyard-venv\Scripts\python.exe
```

nemo-switchyard 0.3.0 has no extras: `pip install "nemo-switchyard[cli,server]"` installs the same package and warns that
the extras do not exist (they belonged to the 0.1 and 0.2 Python server). 0.3.0 ships the native server two ways:

- the Python host `switchyard_rust.server.Server(config, port)`, which always binds 127.0.0.1:
  ```
  python -c "from switchyard_rust.server import Server; import time; s = Server('routes.toml', port=4000); time.sleep(1e9)"
  ```
- the `switchyard-server` binary: `cargo install --locked switchyard-server`, then
  `switchyard-server --config routes.toml --host 127.0.0.1 --port 4000`.

Switchyard has no inbound authentication and the binary's default bind is `0.0.0.0`. Always bind `127.0.0.1` (or put an
authenticating proxy in front of it): anyone who can reach it can spend whatever keys its TOML references.

## Titanos -> Switchyard (router upstream)

One step, no credential (Switchyard needs none):

```
titanos broker connect switchyard --no-credential                       # base_url http://127.0.0.1:4000
titanos broker connect switchyard --no-credential --base-url http://127.0.0.1:4100 --upstream sy
```

This creates the upstream, enables it, raises `cloud_policy` from `off` to `explicit` and reads the route ids from
Switchyard's `GET /v1/models`. `POST /api/broker/connect {"provider": "switchyard"}` does the same over HTTP (Studio uses
it). The equivalent `broker.json` fragment:

```json
{"version": 1,
 "policy": {"cloud_policy": "explicit"},
 "upstreams": {"switchyard": {"profile": "switchyard", "base_url": "http://127.0.0.1:4000", "enabled": true,
                              "catalog": "live", "label": "NeMo Switchyard (local)"}},
 "models": {"sy-auto": {"target": "switchyard/general@switchyard", "capabilities": ["chat", "tools"]}}}
```

`base_url` is Switchyard's **origin**, without `/v1`: the OpenAI and Anthropic surfaces share it. Use `127.0.0.1`, not
`localhost` (plain http is allowed only to an IP literal). Clients then call `<route-id>@switchyard` (for example
`switchyard/general@switchyard`; refs split on the last `@`) or a published alias such as `sy-auto`.

What the `switchyard` profile does:

| Topic | Behaviour |
|---|---|
| Kind and provenance | `router`: answers carry `X-Titanos-Served-By: router:<upstream>`. Like any cloud or router target it needs `cloud_policy` explicit or fallback, and a key with `cloud` or `cloud:<upstream>` (keyless callers need `policy.anonymous_cloud`). |
| OpenAI Chat Completions, Responses | Passed through with every field; streams pass through (`stream_options.include_usage` is added and the extra usage chunk stripped again when the client did not ask for it). Responses get `store: false` unless the upstream's `data_policy.allow_store` is true and the key holds `cloud:store`. |
| Anthropic Messages, count_tokens | Passed through natively to Switchyard's `/v1/messages` and `/v1/messages/count_tokens` (thinking, `cache_control`, citations, metadata and `anthropic-beta` as sent; `anthropic-version: 2023-06-01` is set). An upstream narrowed with `"surfaces": ["openai.chat"]` translates Anthropic onto chat instead. |
| Other inbound protocols | Gemini, Ollama and `/v1/completions` requests are translated onto Switchyard's chat surface. |
| Auth | None by default; Titanos sends no credential. Set `"auth": "bearer"` plus a credential only when an authenticating proxy fronts Switchyard; a credential with the default auth is refused, so a Titanos key never reaches a proxy that may forward it (`forward_auth`, `fallback_client`). |
| Headers to Switchyard | `X-Titanos-Hop` (loop guard), `X-Request-ID` (Switchyard's routing log correlates with Titanos decisions), `anthropic-beta`, and on this row only the harness session headers Switchyard routes on: `x-switchyard-*`, `x-claude-code-session-id`/`-agent-id`/`-parent-agent-id`, `x-codex-turn-metadata`, `x-codex-parent-thread-id`, `x-openai-subagent`, `x-nemo-relay-*`, `x-dynamo-*`, `session-id`, `x-session-id`, `thread-id`, `x-task-id`, `x-client-request-id` and `user-agent`. Never `Authorization`, `x-api-key`, cookies or `X-Titanos-*` hints. |
| Headers back | `x-model-router-selected-model` (the backend Switchyard picked) is returned and recorded in the decision as `router_selected:<model>`. |
| Catalog | The route ids from `GET /v1/models`, with Switchyard's per-route `tool_calling`, `vision` and context window: `GET /v1/models` shows them under `titanos.capabilities` and `titanos.context_length`, and a request with tools or images to a route that declares them `false` is refused with 400 `broker_capability_unavailable` before anything is sent. |
| Circuit breakers | One per route: a failing route (Switchyard passes its backends' 5xx through) never takes the other routes offline. |
| Usage and prices | Usage is read from the OpenAI or Anthropic answer (and stream) Switchyard returns. A price row matches the ref the client used (`switchyard/general@switchyard`, or `*@switchyard`), so a budgeted key pays the route's price whichever backend served; the backend itself is in the decision note. |
| Errors | Switchyard renders errors in the inbound format; Titanos maps them like any upstream (404 `model_not_found`, 429 with `Retry-After`, 503 `upstream_unavailable`, 504 `upstream_timeout`). |
| Responses follow-ups | Switchyard keeps no responses: `GET`/`DELETE /v1/responses/{id}`, `cancel` and `input_items` are not available on this row (na in the matrix). Use full conversations rather than `previous_response_id`. |

Switchyard's other endpoints are reachable through the [native passthrough](BROKER.md#native-passthrough-and-websockets)
with the `native` scope plus `cloud` or `cloud:<upstream>`; the path is relative to Switchyard's origin and the body is
sent untouched (so the model is the route id itself, not `<route>@<upstream>`):

| Path | Notes |
|---|---|
| `POST /native/<upstream>/v1/responses/compact` | Codex remote compaction; the route needs an OpenAI Responses target |
| `POST /native/<upstream>/v1/responses/input_tokens` | the same condition |
| `POST /native/<upstream>/v1/decision` | which target a route would pick (Switchyard runs its judge calls for it) |
| `GET /native/<upstream>/v1/models`, `GET /native/<upstream>/health` | catalog with Switchyard's own fields, health |
| `GET /native/<upstream>/v1/stats`, `POST /native/<upstream>/v1/stats/reset`, `GET /native/<upstream>/v1/routing/session-stats`, `GET /native/<upstream>/metrics` | operator surfaces: `broker:admin` |

A Switchyard deployment that Titanos calls (`routes.toml`; a noop route needs no backend at all):

```toml
schema_version = 1

[llm_clients.openai]
format = "openai_chat"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"

[targets.weak]
id = "gpt-4o-mini"
llm_client = "openai"

[targets.strong]
id = "claude-sonnet-4-5"
llm_client = "anthropic"

[routes.smoke]
id = "switchyard"
type = "noop"

[routes.general]
id = "switchyard/general"
type = "random"
targets = ["weak", "strong"]
weights = [3, 1]
tool_calling = true
context_window = 128000
```

Here Switchyard holds the provider keys itself. To keep provider keys in Titanos instead, point Switchyard's targets at
Titanos (next section).

## Switchyard -> Titanos (Titanos as a target)

Titanos speaks all three of Switchyard's client formats. A deployment whose targets are a Titanos local model and a
Titanos-brokered cloud model:

```toml
schema_version = 1

# Titanos local models. Any X-Titanos-Hop value keeps a request on Titanos' local models (its 1-hop rule), so a route
# that leads back into Switchyard is refused instead of looping.
[llm_clients.titanos_local]
format = "openai_chat"
base_url = "http://127.0.0.1:11435/v1"
api_key_env = "TITANOS_SY_KEY"
extra_headers = { "X-Titanos-Hop" = "switchyard" }

[llm_clients.titanos_responses]
format = "openai_responses"
base_url = "http://127.0.0.1:11435/v1"
api_key_env = "TITANOS_SY_KEY"
extra_headers = { "X-Titanos-Hop" = "switchyard" }

[llm_clients.titanos_messages]
format = "anthropic_messages"
base_url = "http://127.0.0.1:11435"
api_key_env = "TITANOS_SY_KEY"
extra_headers = { "X-Titanos-Hop" = "switchyard" }

# Titanos brokered cloud refs and aliases: no hop header (a hop-marked request never leaves for the cloud). Titanos
# already applies fallback and budgets, so Switchyard retries are off. A route of this client must never name a target
# that leads back into Switchyard (<route>@switchyard): Titanos refuses such a request on its first re-entry by its
# X-Request-ID (508 loop_detected), but the route is still wrong.
[llm_clients.titanos_cloud]
format = "openai_chat"
base_url = "http://127.0.0.1:11435/v1"
api_key_env = "TITANOS_SY_KEY"
max_retries = 0

[targets.local_small]
id = "qwen3-8b"                     # a Titanos local registry name
llm_client = "titanos_local"

[targets.cloud_strong]
id = "gpt-4o-mini@openai"           # a Titanos qualified ref or published alias
llm_client = "titanos_cloud"

[routes.auto]
id = "titanos-auto"
type = "auto"
capable_target = "cloud_strong"
efficient_target = "local_small"

[routes.local]
id = "titanos-local"
type = "passthrough"
target = "local_small"
```

Run it with `--host 127.0.0.1`. `TITANOS_SY_KEY` is a Titanos key for Switchyard:
`titanos keys create --name switchyard --scope inference --scope models:read --scope cloud:openai --expires 30d` (minting a key
turns on key enforcement for the whole server). A budget on that key (`--usd-per-day 1`; the broker role requires one)
needs an operator price row for every cloud target first, here `gpt-4o-mini@openai` (`titanos broker prices import`,
see [BROKER.md](BROKER.md#cost-and-prices)): a budgeted key calling an unpriced target answers 402
`budget_unpriced_target`. A running server applies imported prices after a restart, or after
`titanos broker reload --url URL` run with a `broker:admin` key in `TITANOS_API_KEY` (the Switchyard key lacks that
scope; see [BROKER.md](BROKER.md#applying-changes-to-a-running-server)). Without keys (keyless loopback), drop `api_key_env` and allow keyless
cloud use with `titanos broker policy set --anonymous-cloud true` if the cloud targets are wanted.

Agent harnesses behind Switchyard send identity, cache and storage hints that local models have no use for. The local
routes accept these fields, ignore them and name them in the `X-Titanos-Ignored` response header (nothing is dropped
silently):

| Route | Accepted and ignored | Typed 400 `unsupported_capability` |
|---|---|---|
| `POST /v1/chat/completions` | `metadata`, `user`, `store: false`, `prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`, `service_tier`, `reasoning_effort`, `verbosity` (`top_logprobs` is a schema field: honoured on llama-cpp models, `0` is served, a live value on an in-process backend is a typed 400; see [OPTION_PARITY.md](OPTION_PARITY.md)) | `store: true` |
| `POST /v1/responses` | `include`, `reasoning`, `truncation`, `user`, `prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`, `service_tier`, `top_logprobs`, `text.verbosity` (`store: false` and `metadata` are part of the schema) | `store: true`, `previous_response_id`, `background: true` |
| `POST /v1/messages` | `metadata`, `service_tier`, `thinking`, `inference_geo`, `cache_control`, `context_management`, `output_config` without `format` (block-level `cache_control`, which Switchyard adds, is ignored as before) | `container`, `output_config.format` |

Any other unknown field still answers 422, as before (except on a managed instance with `TITANOS_UNKNOWN_FIELDS=ignore`,
which names it in `X-Titanos-Ignored` on the chat and responses routes; [TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md)). Brokered cloud refs pass every field through to the provider.
Anthropic `count_tokens` on a local target works for llama-cpp models only. Switchyard's
`/v1/responses/input_tokens` and `/v1/responses/compact` have no local implementation in Titanos; route them to an
OpenAI Responses backend.

## Loops

Titanos sends `X-Titanos-Hop: <16 hex characters>` to Switchyard, but Switchyard forwards only `x-request-id` to its
backends, so the header does not come back by itself. Give every Switchyard client that points at Titanos' local models
`extra_headers = { "X-Titanos-Hop" = "switchyard" }`: such a request stays on Titanos' local models, and a route whose
target leads back into Switchyard (`<route>@switchyard`) is refused with 400 `location_unsatisfiable` after one pass.
Clients that point at brokered cloud refs must not carry the header, or Titanos keeps them local and refuses the cloud
target. A request that comes back with this instance's own hop value is refused with 508 `loop_detected`.

Switchyard does forward `x-request-id`, and Titanos sends its request id in that header. While Titanos is waiting for
Switchyard on a request, an inbound request with the same `X-Request-ID` is refused with 508 `loop_detected` before it is
routed, so a cloud-client route (no hop header) whose target resolves back to `<route>@switchyard`, directly or through
an alias or fallback, fails on its first re-entry instead of recursing until the upstream's concurrency slots are
exhausted. This needs the request id to survive the round trip; a proxy between the two that rewrites `x-request-id`
makes loop detection best-effort again, so keep routes that point at Titanos' cloud refs free of Switchyard targets.

## Checks

- `tests/check_broker_switchyard.py` (base tier): the profile row, chat, Responses and Anthropic passthrough, translation,
  catalog capabilities, per-route breakers, the native endpoints, hops, connect and the local fields above, against the
  loopback mock `tests/broker_mocks/switchyard_mock.py` (wire shapes transcribed from a switchyard-server 0.3.0 run).
- `tests/check_switchyard_interop.py` (manual tier): a real switchyard-server and a real `titanos serve` on loopback, both
  directions and the loop, with no provider key:
  ```
  python tests/check_switchyard_interop.py --switchyard-python /path/to/switchyard-venv/python \
      --fixture-models /path/to/fixture/models
  ```
  Without the Switchyard interpreter every case is untried (never a pass); `--strict` turns untried into a failure.
