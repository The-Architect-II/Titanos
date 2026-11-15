# MCP interface

Titanos exposes its inference framework through the official Model Context
Protocol Python SDK, pinned to `mcp==2.2.0`. An MCP client uses the same registry,
model lifecycle, request validation, jobs and artifacts as the HTTP API. HiveMind
is not required.

MCP is included in the base Titanos installation and bundled application. Start
`titanos serve` to expose it alongside inference; no separate MCP service or
numerical runtime is required for protocol discovery.

## Connect

The server's Streamable HTTP endpoint is `/mcp/`, on the same port as inference:
`http://127.0.0.1:11435/mcp/`. Supply `Authorization: Bearer <key>` when
a key is configured (`TITANOS_API_KEY` or a minted `tk_` key). MCP is not a
separate permission system: the transport needs the `mcp` scope, and every tool
call runs as the key of its own outer `/mcp` request, so each tool's route
checks that key's scopes (`inference:chat` for `chat`, `broker:read` for
`broker_status`, `models:write` for lifecycle tools, and so on). Mounted tools
never fall back to the server's own key: a key that holds only `mcp` gets
`scope_denied` from admin tools. Two clients with different keys on one session
each keep their own key.

For hosts that launch stdio servers, configure the installed `titanos` executable
with these arguments:

```json
{
  "command": "titanos",
  "args": ["mcp", "--transport", "stdio", "--url", "http://127.0.0.1:11435"]
}
```

Use the executable's absolute path if it is not on the host application's PATH.
The stdio process connects to an already-running Titanos server. It does not
launch a second inference runtime or duplicate resident models. Set
`TITANOS_API_KEY` in its environment when serving requires a key; do not place
credentials in arguments. `TITANOS_URL` supplies the default backend URL.

The SDK handles protocol negotiation, stdio, JSON-RPC, Streamable HTTP,
initialization, resource access, tool dispatch and prompts. Titanos does not
implement a lookalike protocol or use HTTP SSE as a substitute for MCP.

## Tools and resources

| Purpose | MCP tools |
|---|---|
| Discovery | `server_info`, `list_models`, `model_capabilities`, `list_devices`, `list_workers`, `list_runtime_profiles`, `list_runtime_setups` |
| Model sources | `list_model_providers`, `discover_models`, `inspect_model_source`, `import_model_page`, `plan_model_acquisition` |
| Model acquisition | `submit_model_acquisition`, `list_model_acquisitions`, `get_model_acquisition`, `cancel_model_acquisition`, `retry_model_acquisition`, `activate_model_acquisition` |
| Native runtime setup | `plan_runtime_setup`, `prepare_runtime`, `get_runtime_setup`, `cancel_runtime_setup`, `retry_runtime_setup` |
| Stable local assignment | `list_assignment_devices`, `get_model_assignment`, `save_model_assignment` |
| Model lifecycle | `load_model`, `unload_model`, `model_details`, `register_model`, `reload_registry`, `unregister_model` |
| Text and vision | `chat`, `complete`, `responses` |
| Embeddings, retrieval and prepared graphs | `embed`, `rerank`, `retrieval_answer`, `classify_pairs`, `infer_tensor` |
| Audio | `speak`, `transcribe`, `translate_audio`, `vad`, `diarize`, `analyze_audio` |
| Extended text | `translate`, `sentiment`, `compose_sentiment` |
| Images and video | `generate_image`, `edit_image`, `estimate_depth`, `segment_image`, `generate_video` |
| Typed envelope | `extension_infer`: one typed envelope over the 15 extension operations (`translation`, `ocr`, `detection`, `depth`, `segmentation`, `audio_translation`, `audio_transcription`, `vad`, `diarization`, `audio_analysis`, `sentiment`, `sentiment_compose`, `rerank`, `retrieval_answer`, `pair_classification`); OCR and detection are reached only this way |
| Persistent assets | `submit_job` (image, speech, video, music), `list_jobs`, `get_job`, `cancel_job`, `read_artifact` |
| Other wire shapes | `messages` (Anthropic Messages, non-streaming), `count_tokens`, `moderate` |
| Cloud broker (read and probe) | `broker_status`, `broker_catalog`, `broker_explain`, `broker_decisions`, `broker_usage`, `list_upstreams`, `upstream_health`, `test_upstream`, `whoami` |

Seventy-two tools in total; `tools/list` is authoritative for names, argument schemas and annotations. The binding of each
tool to its HTTP route is generated into [ROUTE_CONTRACT_MATRIX.md](ROUTE_CONTRACT_MATRIX.md).

The existing `responses` tool also serves the stateless local text/image adapter.
It preserves omitted tuning so model defaults apply. Native llama.cpp retains
tool/structured-output support; the local fallback returns text. Use HTTP for
Responses SSE streaming. See [Responses](RESPONSES.md) for exact boundaries.

Inference tools use typed Pydantic request schemas derived from the HTTP
contracts. Call `tools/list` to discover exact argument names, bounds and supported
values. For example:

```json
{
  "name": "chat",
  "arguments": {
    "payload": {
      "model": "chat",
      "messages": [{"role": "user", "content": "Explain local inference."}],
      "max_tokens": 64
    }
  }
}
```

The `chat` payload also accepts `require_loaded: true`, forwarding the strict
loaded-only [chat contract](PROTOCOL_COMPATIBILITY.md#chat-without-loading-a-model)
through the same HTTP route. It defaults to false; other MCP inference tools do
not acquire that guarantee.

Tools return structured JSON, except `speak`, which returns an MCP audio content
block, and `read_artifact`, which returns an embedded binary resource. Errors
remain explicit MCP tool errors. A valid MCP call does not imply the chosen model
supports the requested architecture, task or hardware.

Static resources are `titanos://info`, `titanos://models`, `titanos://jobs`, `titanos://acquisitions`,
`titanos://extensions` and `titanos://schema`. Resource templates expose model capabilities, job status and
acquisition status, plus generated `.png`, `.wav` and `.mp4` artifacts. Discover their exact URI forms with
`resources/templates/list`. For model IDs containing slashes, the
`model_capabilities` tool avoids URI-encoding ambiguity.

The `create_asset` prompt describes discovery, model selection, job submission,
status inspection and artifact retrieval. `diagnose_model` describes an evidence
based setup/inference diagnosis. They are reusable prompts; requesting a prompt
does not execute its suggested actions.

## Cloud broker tools and provenance

When the operator configured the optional [cloud broker](BROKER.md), a named model
may be served by a cloud provider; the server instructions say so. Call
`broker_explain` to see where a request would run (zero egress) and
`broker_status` for the broker summary. Inference tool results then carry
`_meta` with `served_by`, `served_model`, `request_id`, `decision_id` and
`fallback_count`, copied from the HTTP provenance headers.

- `broker_decisions` and `broker_usage` show the caller's own records unless the
  key holds `broker:admin`. `whoami` returns the caller's scopes, constraints,
  limits, remaining budget and expiry.
- `test_upstream` makes one probe call to an upstream (needs `broker:admin`;
  refused in managed role). `moderate` and `test_upstream` are marked open-world.
- No MCP tool sets, rotates or deletes credentials, mints, rotates or revokes
  keys, or changes policy, upstreams, aliases, prices or advisors: agents never
  handle secrets or widen egress. Use the CLI, the HTTP API or Studio for those.
- Tool schemas stay the strict local schemas: model names are at most 128
  characters (use a published alias or a short qualified ref) and
  provider-native extra fields cannot be passed. Results are not streamed.
- The inference tools' `open_world_hint` is fixed when the server starts: true
  unless the role is managed without delegation or `broker.json` defines no
  upstream. A broker reload does not change it.

## Asset workflow and limits

For a long image, speech, video or music request, use `submit_job` and save its returned
ID immediately. Supply an idempotency key when retrying a submission whose
response may have been lost. While its job record is retained, reusing the same
key and body identifies the same job; changing the body with that key is an error.
After the record is reclaimed, that key can create new work. Inspect with `get_job`, then
retrieve its artifact with `read_artifact` or a `titanos://artifacts/...` resource.

Audio/image uploads are inline base64. Image bytes use
`TITANOS_MAX_INPUT_IMAGE_BYTES`; audio and inline artifact transfers use
`TITANOS_MAX_BODY_BYTES`. They never
name local server files or cause remote media downloads. Artifact reads accept
only Titanos-generated filenames. Download larger artifacts from the authenticated
HTTP files route. Artifact retention uses `TITANOS_ARTIFACT_TTL_SECONDS` (24 hours
by default); MCP does not introduce a second asset store. Transport body limits
and request timeouts also follow the server settings.

MCP chat calls return one complete inference result. Use HTTP SSE when individual
generation tokens are required. Tool calls, structured outputs and Responses
remain backend-specific. Unsupported options and models fail explicitly.

## Embedding in an ASGI application

`titanos.mcp_server.create_mcp(base_url=None, api_key=None, app=None, broker=None)`
returns an official `MCPServer`. Passing the existing Titanos ASGI app makes tool
calls pass through that application's HTTP routes in-process. Calls that arrive
over a mounted `/mcp` run as the outer request's principal; a direct in-process
call with no HTTP request (an embedding application calling the server object)
keeps the `api_key` given to `create_mcp`. Calling
`mcp.streamable_http_app(streamable_http_path="/", max_request_body_size=64 *
1024 * 1024)` creates the sub-application mounted at `/mcp`.

The owning application's lifespan must enter `mcp.session_manager.run()` once;
Starlette does not run the lifespan of a mounted application. Keep the server's
authentication middleware outside the mount. Preserve the SDK's transport host
and origin checks when configuring a deployment hostname. These requirements
follow the [official ASGI integration documentation](https://py.sdk.modelcontextprotocol.io/run/asgi/).

## Validation

Run `python tests/check_mcp.py`; the MCP SDK is a base dependency.
`python tests/check_broker_mcp.py` checks the 72-tool list, principal
propagation (two keys on one session) and the broker tools, and
`python tests/check_broker_e2e.py` drives a real server with the official client
and a scoped key. The check
uses the official SDK client over real loopback HTTP and a real stdio subprocess.
It exercises modern protocol negotiation and the older initialize handshake,
tool/resource/prompt discovery, typed calls, media transfer, jobs, idempotency,
authentication and explicit errors.

The model fixture is deliberately fake. These checks prove MCP/HTTP/control
integration, not model quality, accelerator execution or useful asset quality.
Real-model and hardware qualification is recorded separately. SDK integration
uses the [official server and client documentation](https://py.sdk.modelcontextprotocol.io/).
The generated [route-to-typed-contract matrix](ROUTE_CONTRACT_MATRIX.md) lists
every MCP tool with its request and response models; the [protocol parity and
deviation map](PROTOCOL_DEVIATIONS.md) records upstream field handling.
