# Topology and roles

<!-- claims-legend -->
Titanos runs as one process that serves its own models and, optionally, brokers requests to cloud upstreams
([BROKER.md](BROKER.md)). Inside a larger system it plays the role it is given and nothing more. This page covers the
three roles, what each one switches on or off, how a parent orchestrator (NVIDIA PAIR, HiveMind) drives a managed
Titanos, and port choices. Statements that something is proven carry [BROKER_MATRIX.md](BROKER_MATRIX.md) cell ids in
HTML comments in this file's source.

## The rule that never changes

In every role Titanos **never** discovers peers, fans requests out, aggregates other hosts' catalogs, prewarms,
scatters, federates or load-balances other servers, retries on another host, takes over a port or creates a facade.
Inference never triggers a model download (only the explicit `/api/acquisitions` workflow does). The optional
client-side `titanos.pool.Pool` stays on the client ([MULTIHOST_CLIENT.md](MULTIHOST_CLIENT.md)). A broker-role Titanos
is just another server a client can point at. <!-- matrix: role:any\|switch_table -->

## Choosing a role

`titanos serve --role standalone|broker|managed` or `TITANOS_ROLE`. The role is fixed at start; changing it needs a
restart and is never possible over HTTP, and a reload that would enable egress in managed role without delegation is
refused. <!-- matrix: role:managed\|reload_refusal -->

| Role | Use it for |
|---|---|
| `standalone` (default) | One operator's machine. Local models; cloud only if the operator configures it. |
| `broker` | A shared Titanos that re-serves operator-held provider credentials to several scoped keys. Stricter start rules. |
| `managed` | A Titanos started and supervised by an orchestrator (PAIR, HiveMind). It holds no usable provider keys by default and makes no inference, catalog, probe or advisor egress of its own. Operator-initiated model acquisition is the one exception: `/api/acquisitions*` and the model-provider routes (`/api/model-providers/{provider}/search`, `/inspect`, `/import`) reach the model hubs in every role, for callers holding `models:read` or `models:write`. |

## Switch table

| Switch | standalone | broker | managed |
|---|---|---|---|
| Local inference | on | on | on |
| Cloud egress | when the effective cloud policy is `explicit` or `fallback`, the upstream is enabled and its credential validated | same | **off**: the vault is sealed and the egress client refuses every purpose; per request on only when all five delegation conditions below hold |
| Cloud fallback from local | only with cloud policy `fallback` | same | off: typed errors instead (429 or 503 with `Retry-After`), the parent retries |
| Router upstreams | on if configured | on | off unless delegated |
| Advisors | on if configured | on | off |
| Health probes, live catalogs, credential validation, upstream test | on, lazily | on, lazily | off (the purpose is refused) |
| Anonymous (keyless) cloud | `policy.anonymous_cloud`, default false | impossible | off |
| Inbound keys | optional; enforced as soon as any key exists | **required**: a bootstrap key; serve refuses keyless; egress keys need a budget and an expiry | optional; a keyless loopback parent is allowed; a network bind needs keys |
| Network bind | needs enforced keys | also needs native TLS (`TITANOS_TLS_CERTFILE`, `TITANOS_TLS_KEYFILE`) | default `127.0.0.1`; a network bind needs enforced keys |
| Port | 11435 by default | 11435 by default | an explicit `--port` or `TITANOS_PORT` is required; 11434, 11435, 1234, 1235, 8000 and 8001 are refused |
| Parent headers | honoured if present | honoured if present | honoured and authoritative (`local_only` placement fails closed) |
| Hop guard and 508 loop detection | on | on | on |
| `GET /v1/health/ready` (public) | on | on | on |
| `GET /api/version` (public) | absent | absent | on |
| Audit log | on when the broker is configured | required, writable, chain verified at start | on |
| Vault | open | open; `sealed-file` needs an owner-only passphrase file | sealed unless `TITANOS_MANAGED_DELEGATION=parent` |

<!-- matrix: role:any\|switch_table, role:standalone\|cloud_policy, role:standalone\|anonymous_cloud, role:standalone\|no_broker_json_no_egress, role:broker\|serve_preconditions, role:broker\|egress_key_rules, role:managed\|sealed_no_probes, role:managed\|ports, role:managed\|public_routes, role:any\|hops -->

`GET /api/info` reports `role`, `orchestrated` (true in managed role) and a `broker` block; `standalone: true` keeps its
documented meaning ("runs without an orchestrator dependency") in every role, so existing adoption checks do not
break. `GET /healthz` and `GET /v1/health/ready` report the role too.

### Egress purposes in managed role

Every outbound call carries a purpose, and the egress client re-checks the role switches for that purpose before it
opens a socket. In managed role:

| Purpose | Allowed |
|---|---|
| `inference` | only for a delegated request (below) |
| `probe`, `validate`, `test` | never |
| `catalog` | never |
| `advisor` | never |

This holds for upstreams that need no credential as well, so a keyless managed Titanos makes **zero** egress.
Credential writes, credential validation and upstream tests answer 409 `role_forbids`; key writes are allowed.
At start a managed instance that finds enabled upstreams logs once: `managed role: N configured upstreams ignored; set
TITANOS_MANAGED_DELEGATION=parent to allow parent-delegated use`. <!-- matrix: role:managed\|egress_refused, feature:e2e.managed_role -->

## Managed headers and delegation

A managed Titanos accepts requests with or without parent headers:

| Header | Effect |
|---|---|
| `X-Titanos-Deadline-Ms` (relative milliseconds) | The request's deadline (capped by `TITANOS_CLIENT_TIMEOUT`); expiry answers 504 `deadline_exceeded`, and a committed cloud stream ends with the protocol's terminal frame. Other parent formats (`X-HiveMind-Deadline`) are logged once and ignored until a recorded fixture pins them. `TITANOS_MANAGED_DEADLINE_HEADERS` lists the accepted names. |
| `X-Request-ID`, `traceparent`, `X-HiveMind-Trace` | The first valid one (`TITANOS_MANAGED_TRACE_HEADERS`) is echoed as `X-Request-ID` and recorded in decisions and logs |
| `X-Titanos-Location`, `X-HiveMind-Placement` | `local_only` is authoritative and fails closed |
| `X-Titanos-Attempt`, `X-HiveMind-Attempt` | Recorded only |
| `X-Titanos-Hop`, `X-HiveMind-Peer-Hop` | The request stays local; this instance's own hop value answers 508 `loop_detected` |

<!-- matrix: role:managed\|parent_headers, role:any\|hops -->

**Unknown request fields.** With `TITANOS_UNKNOWN_FIELDS=ignore` a managed instance strips unknown top-level fields
on `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/api/chat`, `/api/generate`,
`/api/embed` and `/api/embeddings` and names them in `X-Titanos-Ignored` instead of answering 422. Unknown nested
fields and wrong-typed known fields stay errors; the default (`strict`) and every other role keep the 422.

**Delegation.** A managed instance may reach a cloud upstream for one request only when all five hold:

1. the operator set `TITANOS_MANAGED_DELEGATION=parent` (then the vault is not sealed);
2. the caller is an authenticated key holding the `delegate` scope;
3. the request carries `X-Titanos-Delegate: cloud`;
4. a local credential exists for the upstream;
5. the target is explicit: a qualified ref in the body, or `X-Titanos-Target` naming one target.

Delegation never enables fallback, probes, catalogs or advisors, and it never raises the configured cloud policy.
Without delegation configured a cloud request answers 403 `role_forbids`; with delegation configured but a request that
is not delegated, or that names an alias, a virtual model or a compatibility suffix in the body, it answers 403
`delegation_required`. <!-- matrix: role:managed\|delegation -->

## NVIDIA PAIR

`titanos pair-manifest` writes PAIR's engine-manager manifest ([PAIR.md](PAIR.md)). In this release its runtime args
gain `--role managed`:

```text
--config <registry> serve --role managed --host {host} --port {port}
```

The identity block (`{"path": "/api/info", "name": "Titanos", "standalone": true}`) is unchanged; readiness and health
are `GET /healthz`, with eight actions. Ports: address (facade) 11440 and engine base 11441 (2026-11-12).
It refuses 11434, 11435 and 11437. **This changed manifest is untried against PAIR** until the manual
`tests/check_pair.py` lane is re-run; until then do not claim the new manifest compatible. <!-- matrix: role:managed\|pair_manifest -->

Under PAIR the default is a managed engine with delegation off: it serves local models and makes no egress. PAIR's
engine-manager action schema carries no authentication headers, so a PAIR-driven call can only be delegated if the
parent forwards a key with the `delegate` scope. Whether a PAIR deployment should instead use a separate broker-role
Titanos for cloud work is a per-deployment choice.

## HiveMind

`titanos hivemind-manifest --port P --job-type JT [--output FILE] [--register-with-gateway URL]` writes a HiveMind
engine-manager manifest for a managed Titanos:

- It uses only the placeholders `{bind}`, `{port}`, `{bin}` and `{cli}`; HiveMind's loader rejects unknown tokens such
  as `{host}`.
- Runtime args: `--config <registry> serve --role managed --host {bind} --port {port}`; the port must not be a reserved
  port.
- Readiness and health: `GET /v1/health/ready`. Identity: `GET /api/info` with `name: Titanos`.
- Actions: `list_models` (`GET /v1/models`), `loaded_models` (`GET /api/models/loaded`), `server_info`
  (`GET /api/info`) and `chat` (`POST /v1/chat/completions`).
- HiveMind's gateway (`menta_hli`) adds Ollama's `options` object (`{"num_ctx": 4096}` by default) to every chat request
  it sends a non-cloud backend. Titanos tolerates it on `/v1/chat/completions` and `/v1/completions`, where `options` is a
  foreign extension: on a llama-cpp model `num_ctx` sizes the runtime (started, or restarted once idle, with the next power
  of two per slot up to the largest per-slot context, [NATIVE.md](NATIVE.md)), so a GGUF model registered below 4096 per
  slot (Titanos's default registration is 8192 over 4 slots, 2,048 per slot) serves HiveMind's 4096 at 4096; above the
  largest per-slot context, and on the other backends, it is not honoured and is named in `X-Titanos-Ignored`, never a
  400 (`/api/chat` answers 400 `context_limit` there). Every other `options` key is named as `options.<key>`, and nothing in `options` is a
  422 ([PROTOCOL_COMPATIBILITY.md](PROTOCOL_COMPATIBILITY.md)). Until 2026-10-06 the request was refused 422
  `body.options: Extra inputs are not permitted`, which HiveMind reported to its client as 503 `model_downloading` (the
  2026-10-05 smoke).
- The integration block is disabled by default (`enabled_by_default: false`). <!-- matrix: role:managed\|hivemind_manifest -->
- HiveMind's engine manager does not read that block. Its manifest struct (`struct Manifest` in the Rust engine
  manager's `src/manifest.rs`, read on 2026-10-05 at its commit `8ee2d458`) has the fields `engine`, `display_name`,
  `manifest_version`, `platforms`, `actions`, `detect`, `install`, `uninstall` and `runtime`, and no
  `integration` field; it is deserialised without a deny-unknown-fields rule, so the whole `integration` block
  (`job_type`, `enabled_by_default`, `identity`) is dropped silently on load. `enabled_by_default` therefore enables
  or disables nothing on the HiveMind side, and the identity route in the block is not what HiveMind probes; the
  block is informative for operators and for Titanos' own tests; whether and when HiveMind starts the engine is not
  decided by this field.

**Gateway registration (optional, off by default).** `serve --role managed --register-with-gateway URL --job-type JT`
(also added to the manifest's args when you pass `--register-with-gateway` to `hivemind-manifest`) sends one
`POST <gateway>/register-service` with `{"job_type": JT, "port": P}` after this server answers its readiness route. The
job type must already exist on the gateway. This is the only parent-directed outbound call Titanos makes, it is
explicit operator configuration, and it is not retried after an answer. The URL may not carry credentials, a query or a
fragment. The request carries `Authorization: Bearer <TITANOS_API_KEY>` when that variable is set (a gateway on
another host needs one) and no credential otherwise. `titanos hivemind register-service GATEWAY --job-type JT
[--port P] [--url U] [--key-env VAR]` sends the same registration for an already running server, from a shell on the
Titanos host or on the gateway host: the bearer comes from `TITANOS_API_KEY`, or from the variable `--key-env` names
when the gateway's key differs (the option takes a variable name, never a secret value); the command probes
`<U>/v1/health/ready` first and exits 1 when the gateway does not answer 2xx ([OPERATIONS.md](OPERATIONS.md#hivemind-registration)).

**The `inference_microservices` job type.** On the HiveMind side, define the job type that the registration names
before starting Titanos. Its exact schema belongs to your HiveMind version; the Titanos side needs only:

```text
job type name:  the value passed to --job-type (1-64 characters [A-Za-z0-9._-])
service port:   the managed Titanos port (explicit, not reserved)
health:         GET /v1/health/ready on that port
protocol:       OpenAI-compatible HTTP (/v1/...)
```

## Orchestrator gateways as upstreams

The other direction also works: a standalone or broker-role Titanos can use a HiveMind gateway or a PAIR facade as a
router upstream (profiles `hivemind_gateway` and `pair_facade`). Titanos then sends no HiveMind cloud-permission or
classification headers, so the gateway's own policy decides; it stamps `X-Titanos-Hop`, forwards `X-Request-ID` to
the HiveMind gateway, and plain HTTP to a LAN address needs `policy.allow_http_lan`. These upstreams are off in managed
role. Loop detection through a gateway that does not echo the hop header is best effort; the fallback cap (at most 5)
bounds amplification. <!-- matrix: openai.chat\|hivemind_gateway\|off, openai.chat\|pair_facade\|on, role:any\|hops -->
Two settings the 2026-10-05 HiveMind run needed: the `hivemind_gateway` `base_url` must end in `/v1` (the profile appends
`/chat/completions` and the other OpenAI paths to it, so `http://host:port/v1`, not the origin), and a router upstream
is a cloud-class target for the policy: while `policy.cloud_policy` is `off` (the default) the broker answers 403
`cloud_disabled` for it, so raise the policy with `titanos broker policy set --cloud-policy explicit` (bounded by
`TITANOS_BROKER_CLOUD_CEILING`) before routing through the gateway.

## Ports

- Standalone and broker roles default to 11435. On a host that also runs HiveMind, its Ollama-compatible facade may
  already use 11435; pick another port with `--port` or `TITANOS_PORT`. Titanos binds exactly the requested port and
  never substitutes another one ([CONFIGURATION.md](CONFIGURATION.md#bind-failures)).
- Managed role requires an explicit port and refuses 11434 (Ollama), 11435 (standalone Titanos), 1234 and 1235
  (LM Studio) and 8000 and 8001 (vLLM). <!-- matrix: role:managed\|ports -->
- PAIR's manifest uses its own address (11440) and engine base (11441) ports.
