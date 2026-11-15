# Stateless Responses

`POST /v1/responses` serves Titanos's local chat and vision backends through a
stateless adapter over the same engine as `/v1/chat/completions`. Requests for
native llama.cpp models pass through to the native server, which keeps its own
tool and structured-output support. Protocol support is not model or hardware qualification.

The local adapter reuses model loading, request/model/process defaults, limits,
generation, usage and cancellation from `/v1/chat/completions`. It creates one
assistant message containing `output_text`; the official Python SDK exposes that
text through `response.output_text`. Send `POST /api/models/capabilities` with the registered model name to inspect its
`responses` capability record.

```python
from openai import OpenAI

# Use your configured Titanos key if authentication is enabled.
client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="local")
response = client.responses.create(
    model="chat",
    input="Explain local inference briefly.",
    instructions="Be concise.",
    max_output_tokens=128,
    temperature=0.3,
    store=False,
)
print(response.output_text)
```

Supported local inputs are a string or `system`/`user`/`assistant` messages with
text or typed `input_text` parts. Vision models also accept `input_image` parts
with inline base64 `image_url` and `detail="auto"`. Remote URLs and uploaded file
IDs are not fetched. Previous assistant `output_text` messages can be replayed
with their IDs/status and empty annotations/logprobs; the SDK's nullable
`phase=None` bookkeeping is accepted. Nonempty phase/annotation semantics are
not implemented. Image decoding still belongs to the selected vision backend.

Explicit `max_output_tokens`, temperature and top_p override model defaults.
Omitted controls retain the shared model/default policy. The token-limit aliases
`max_tokens`, `max_completion_tokens` and `max_output_tokens` resolve to the
backend's `max_new_tokens`; API aliases are not forwarded as library kwargs.
Unlike the hosted API, this local adapter permits output limits down to one
token. Optional metadata allows 16 string pairs with 64-character keys and
512-character values and is returned with that response.

The local fallback rejects tools, tool results, reasoning/reference items,
developer roles, structured JSON formats, stored response IDs
(`previous_response_id` and `store: true` answer 400), conversations and
background execution (`background: true` answers 400); the tool, tool-result
and structured-output refusals name the model, its backend and the missing
capability (`unsupported_capability`), and a llama-cpp model serves them through
the native server ([PROTOCOL_COMPATIBILITY.md](PROTOCOL_COMPATIBILITY.md#agent-frameworks-on-local-models)). The hosted-platform
hints `include`, `reasoning`, `truncation`, `user`, `service_tier`,
`prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`,
`top_logprobs` and `text.verbosity` are accepted, ignored and named in the
`X-Titanos-Ignored` response header ([PROTOCOL_COMPATIBILITY.md](PROTOCOL_COMPATIBILITY.md#agent-harness-fields-on-the-local-routes)). Explicit empty/default forms
(`tools=[]`, tool_choice `auto` or `none`, `parallel_tool_calls=False`, and text
format `text`) have their ordinary text-only meaning. No unsupported request is
reported as a successful tool or state operation. Native llama.cpp capabilities
remain separate.

With `stream=True`, SSE carries matching event/type fields, increasing sequence
numbers, stable response/message IDs and output/content indexes. Creation and
item/content events precede incremental text; completed text/items and the final
response follow. On a llama-cpp model the llama.cpp server's own events are forwarded, completed with the sequence
numbers, output/content indexes and the full `response.created` object that the stock OpenAI SDK's `responses.stream()`
helper requires (b10809 sends `{id, object, status}` only; `responses.native_stream`). A token limit emits
`response.incomplete` with reason `max_output_tokens`. Backend failure or missing completion emits `response.failed`,
preserving partial text. EOF is never manufactured into success. Disconnecting closes the existing generation stream and propagates its cancellation signal, including while the producer's queue is full.

Reported input/output usage comes from the backend, including a usage-only tail
after the finish chunk. Missing counts remain null. Cache/write/reasoning detail
counts are zero because this adapter does not separately account for those token
categories; this is not a claim about a model's internal reasoning. All reported
tokens remain counted as ordinary input/output. No hosted service tier, response
store, or cache behavior is invented.

The official OpenAI Python SDK 2.54.0 was exercised against real loopback HTTP
with strict response validation, typed SSE, output replay and
`responses.stream(...).get_final_response()` for completed output. That SDK's
high-level final-response helper does not finalize incomplete/failed responses;
consume their typed terminal events directly. The existing MCP `responses` tool
uses the same endpoint and was exercised through HTTP and stdio. These protocol
checks use a controlled inference seam; trained-model checks are separate.

References: [create response](https://developers.openai.com/api/reference/python/resources/responses/methods/create),
[streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events).
