# Titanos engine boundary for PAIR

This describes the current implemented interface, with source-based provenance.
It is not a claim that PAIR routes every Titanos modality. The PAIR owner must
qualify its current plugin with a real routed request. Titanos runs independently
and does not require PAIR or HiveMind; PAIR owns fleet orchestration.

## Current discovery and lifecycle

Agreed integration port contract (NVIDIA LAN head, 2026-11-12, replacing the
September 21 pair facade `11436` / backend `11437`): address (facade) `11440`,
engine (managed backend search base) `11441`, persisted facade file
`titanos-proxy-port.json`, participation disabled by default. The generated
manifest must bind the actual selected backend port; PAIR may select the next
free owned port. Never use `11434`/`11435` (Ollama, managed Ollama) or `11437`
(another owner's Ollama); the manifest refuses them. Facade and backend are separate. Ordinary standalone
Titanos defaults remain compatible. This is the coordinated implementation
contract; it is not yet a claim of a built/routed PAIR adapter.

| Operation | Existing interface | Meaning |
|---|---|---|
| Process health | `GET /healthz` | Process is available; no resident model required. |
| Engine discovery | `GET /api/info` | Name/version, standalone flag, interfaces, configured tasks/models, job contracts and settings. |
| Schema | `GET /openapi.json` | Actual typed HTTP interface for this executable. |
| Catalog | `GET /v1/models` | Registered models; configured capability is not hardware or quality qualification. |
| Model metadata | `GET /v1/models/{name}` | One registered model's catalog entry. |
| Loaded inventory | `GET /api/models/loaded` | Currently resident models. |
| Readiness | `GET /readyz` | 200 with loaded models; 503 when none are resident is normal idle state. |
| Load/unload | `POST /api/models/load`, `POST /api/models/unload` | JSON `model`, optional assigned `worker`; explicit lifecycle. |
| Passive capabilities | `POST /api/models/capabilities` | JSON `model`; declared supported operations without loading. |
| Runtime details | `POST /api/models/details` | May load the model; do not poll as passive discovery. |
| Devices | `GET /api/devices?backend=...` | Cached `nvidia`, `pytorch`, `openvino`, `onnx` or `system` inventory; inventory is not inference qualification. |
| Workers | `GET /api/workers` | Actual configured worker observations. |
| Agent access | Streamable HTTP `/mcp/` and `titanos mcp` stdio | Discover tools/resources/prompts at runtime. |

Source: `src/titanos/server.py`, `mcp_server.py`, `engine.py` and the generated
OpenAPI schema. Use actual backend/device metadata and checkpoint provenance;
never infer placement from a model name or hardware vendor alone.

## Inference and jobs

Chat, Responses, embeddings, image, audio, video and typed extension routes are
advertised through the executable's schema and model capabilities. Route only a
modality PAIR itself implements and has tested. The compatibility inventory is
documented in `PROTOCOL_COMPATIBILITY.md`.

`POST /api/jobs` runs supported image/speech/video asset work and supports the
`Idempotency-Key` header. `GET /api/jobs` and `GET /api/jobs/{id}` expose status;
`DELETE /api/jobs/{id}` requests cancellation. Fetch the completed result through
the returned route. A cancellation acknowledgement is not proof that device work
already stopped. Treat failures and model-quality failures separately.

## Authentication, model access and acquisition

HTTP uses `Authorization: Bearer <key>` when keys are configured: either the
bootstrap `TITANOS_API_KEY` or, preferably, a scoped member key minted for PAIR
(`titanos keys create --name pair --scope inference --scope models:read
--scope jobs`). A keyless loopback instance remains valid for a PAIR parent on
the same host. Non-loopback binding requires authentication. Studio's static
shell and health are public; model, schema and inference operations remain
protected. Keep credentials local to the configured transport; do not include
them in generated manifests, logs, discovery announcements or media. This is a
single-operator service boundary with scoped keys, not multi-tenant isolation.

The PAIR-managed engine runs with `--role managed`. In that role Titanos makes no
outbound provider call of its own unless the operator sets
`TITANOS_MANAGED_DELEGATION=parent` and the parent sends a delegated request with
a `delegate`-scoped key ([TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md)). `standalone: true`
in `/api/info` keeps its meaning (no orchestrator dependency) in every role.

The generated `pair-manifest` is disabled by default and separates PAIR's
Titanos address (`11440`) from its managed Titanos engine base (`11441`) through
`titanos-proxy-port.json`. It never silently reuses Ollama ports `11434`,
`11435` or `11437`. Health is process liveness; PAIR must also read `/api/info` and verify
`name: Titanos` plus the advertised contracts before adoption. That identity
check is unchanged; `/api/info` additionally reports `role: managed` and
`orchestrated: true`. The manifest's runtime arguments now include
`--role managed`; this changed manifest is untried against PAIR until the
manual `tests/check_pair.py` lane is re-run.

Model discovery and setup use a separate schema from inference jobs:

| Operation | HTTP |
|---|---|
| Providers | `GET /api/model-providers` |
| Task search | `GET /api/model-providers/{provider}/search?task=...` |
| Inspect exact source | `POST /api/model-providers/inspect` |
| Import model page/link | `POST /api/model-providers/import` |
| Immutable plan | `POST /api/acquisitions/plan` |
| Submit/list/get | `POST/GET /api/acquisitions`, `GET /api/acquisitions/{id}` |
| Cancel/retry | `DELETE /api/acquisitions/{id}`, `POST /api/acquisitions/{id}/retry` |
| Atomic register/load | `POST /api/acquisitions/{id}/activate` |
| Compatible native profiles | `GET /api/runtime-profiles` |
| Native runtime plan/setup | `POST /api/runtime-setups/plan`, `POST /api/runtime-setups` |
| Runtime status/cancel/retry | `GET/DELETE /api/runtime-setups/{id}`, `POST /api/runtime-setups/{id}/retry` |

Acquisition records bind provider, source, requested and resolved revision,
selected files, bytes/digests, runtime/device recommendation, progress, license
evidence and activation. Failed/cancelled/interrupted work is never registered.
PAIR may surface and invoke these Titanos-owned operations; it must not translate
them into `/api/jobs` or claim that provider metadata alone is runnable.

Hugging Face and checksum-bound GitHub Sherpa release assets are implemented
providers. General HTTPS model pages can be retained as metadata, but only a
recognized provider/format can create an executable acquisition plan.

Gated access and model licenses are distinct from the application's license.
Keep immutable revision and declared license/access metadata. Neither routing
nor acquisition can enable remote executable model code or silently opt into
trusted pickle. Weights stay external and registrations may change at runtime
through the existing safe registry lifecycle.

## Fleet ownership boundary

PAIR owns host discovery, trust, scheduling, facade routing and node affinity.
Titanos owns each node's providers, acquisition records, runtime/device choice,
registry, model residency, inference jobs and artifacts. PAIR must retain the
chosen owner for asynchronous job status/cancel/artifact operations and mounted
MCP sessions; it must not reroute a handle-less follow-up to another node.
Capability metadata must retain Titanos task/status and distinguish registered,
loaded, unavailable and hardware-qualified states.

Runtime profile catalogs and executable targets remain Titanos server policy.
PAIR may choose only a configured profile name and must retain the owning node
for runtime setup status/cancellation just as it does for model acquisition and
inference jobs.
