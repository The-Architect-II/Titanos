"""Agent-framework capabilities on local models: typed refusals that name the backend, and the Anthropic tool stream.

Tool calling, tool-call history and structured output reach a model only on the llama-cpp native backend (the llama.cpp
server parses the model's tool template and applies its grammars). The in-process backends (transformers, openvino,
onnx, ...) have no such implementation, so a request that needs one is refused before inference with a 400
`unsupported_capability` whose message names the model, its backend, the capability the backend lacks and the route,
instead of a generic hint. Nothing is dropped silently.

The Anthropic Messages stream translator turns OpenAI chat-completion chunks (text deltas and incremental tool_calls
fragments, as the llama.cpp server emits them) into Anthropic SSE events: text blocks as text_delta, tool calls as
tool_use blocks with input_json_delta fragments, stop_reason tool_use when a tool was called.
"""
import json
import uuid

from .protocols import events

CHAT_CAPABILITIES = (('tools', 'tool calling (tools)'), ('tool_choice', 'tool_choice'),
                     ('parallel_tool_calls', 'parallel_tool_calls'), ('response_format', 'structured output (response_format)'))
RESPONSES_CAPABILITIES = (('tools', 'tool calling (tools)'), ('tool_choice', 'tool_choice other than auto or none'),
                          ('parallel_tool_calls', 'parallel_tool_calls'), ('text', 'structured output (text.format)'))


def backend_name(spec):
    return getattr(spec, 'backend', None) or 'unknown'


def refusal(model, spec, capabilities, route):
    """The 400 message: model, backend, the missing capabilities and the route the client used."""
    missing = ' and '.join(capabilities) if capabilities else 'agent requests'
    return (f"Model '{model}' runs on the {backend_name(spec)} backend, which does not implement {missing} on {route}; "
            'register a llama-cpp (GGUF) model for tool calling, tool-call history and structured output')


def chat_capabilities(data):
    """The capabilities an OpenAI chat body asks of the backend (fields present and tool-role or tool_calls history)."""
    found = [label for key, label in CHAT_CAPABILITIES if key in data]
    messages = data.get('messages') or []
    if any(message.get('role') == 'tool' or message.get('tool_calls') for message in messages):
        found.append('tool-call history (tool_calls and role tool messages)')
    if any(message.get('role') == 'developer' for message in messages):
        found.append('the developer role')
    return found


def responses_capabilities(payload):
    """The capabilities a Responses body asks of the backend; mirrors the refusals of responses.chat_request."""
    found = []
    if payload.tools:
        found.append(RESPONSES_CAPABILITIES[0][1])
    if payload.tool_choice not in (None, 'auto', 'none'):
        found.append(RESPONSES_CAPABILITIES[1][1])
    if payload.parallel_tool_calls:
        found.append(RESPONSES_CAPABILITIES[2][1])
    if payload.text not in (None, {}, {'format': {'type': 'text'}}):
        found.append(RESPONSES_CAPABILITIES[3][1])
    items = payload.input if isinstance(payload.input, list) else []
    if any(isinstance(item, dict) and item.get('type') in ('function_call', 'function_call_output') for item in items):
        found.append('tool-call history items (function_call and function_call_output)')
    return found


def _event(name, data):
    return 'event: ' + name + '\ndata: ' + json.dumps({'type': name, **data}) + '\n\n'


def _start(index, block):
    return _event('content_block_start', {'index': index, 'content_block': block})


def _stop(index):
    return _event('content_block_stop', {'index': index})


async def anthropic_stream(result, ident, model):
    """Anthropic SSE from an OpenAI chat-completion chunk stream (text and tool_calls deltas).

    Content blocks are opened in order: a text block on the first text delta, a tool_use block on the first fragment of
    each tool call (keyed by the OpenAI tool index; the id and name arrive in that first fragment on llama.cpp). The
    open block is stopped when the next block starts and at the end. A fragment for an already stopped block (parallel
    calls interleaved by the backend) is still delivered as an input_json_delta for that index, never dropped. A stream
    with no content ends with one empty text block, so a text-only consumer always finds a text block.
    """
    yield _event('message_start', {'message': {'id': ident, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [],
                                               'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})
    index = -1
    open_block = None  # ('text', None) or ('tool', openai tool index)
    tools = {}  # openai tool index -> anthropic block index
    reason, usage, called = 'end_turn', {}, False

    def close():
        nonlocal open_block
        if open_block is None:
            return None
        current = index if open_block[0] == 'text' else tools[open_block[1]]
        open_block = None
        return _stop(current)

    async for item in events(result):
        if item.get('usage'):
            usage = item['usage']
        if 'error' in item:
            yield _event('error', {'error': item['error']})
            return
        for choice in item.get('choices', []):
            delta = choice.get('delta') or {}
            text = delta.get('content')
            if text:
                if open_block is None or open_block[0] != 'text':
                    stop = close()
                    if stop:
                        yield stop
                    index += 1
                    open_block = ('text', None)
                    yield _start(index, {'type': 'text', 'text': ''})
                yield _event('content_block_delta', {'index': index, 'delta': {'type': 'text_delta', 'text': text}})
            for call in delta.get('tool_calls') or []:
                if not isinstance(call, dict):
                    continue
                key = call.get('index', len(tools))
                function = call.get('function') or {}
                if key not in tools:
                    stop = close()
                    if stop:
                        yield stop
                    index += 1
                    tools[key] = index
                    open_block = ('tool', key)
                    called = True
                    yield _start(index, {'type': 'tool_use', 'id': call.get('id') or 'call_' + uuid.uuid4().hex,
                                         'name': function.get('name') or '', 'input': {}})
                fragment = function.get('arguments')
                if fragment:
                    yield _event('content_block_delta', {'index': tools[key], 'delta': {'type': 'input_json_delta', 'partial_json': fragment}})
            finish = choice.get('finish_reason')
            if finish == 'length':
                reason = 'max_tokens'
            elif finish == 'tool_calls':
                reason = 'tool_use'
    if index < 0:
        index = 0
        open_block = ('text', None)
        yield _start(0, {'type': 'text', 'text': ''})
    stop = close()
    if stop:
        yield stop
    if called and reason == 'end_turn':
        reason = 'tool_use'
    yield _event('message_delta', {'delta': {'stop_reason': reason, 'stop_sequence': None},
                                   'usage': {'input_tokens': usage.get('prompt_tokens', 0), 'output_tokens': usage.get('completion_tokens', 0)}})
    yield _event('message_stop', {})


__all__ = ['CHAT_CAPABILITIES', 'RESPONSES_CAPABILITIES', 'anthropic_stream', 'backend_name', 'chat_capabilities',
           'refusal', 'responses_capabilities']
