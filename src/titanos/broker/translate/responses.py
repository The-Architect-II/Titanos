"""OpenAI Responses codec for non-OpenAI targets and for OpenAI-compatible upstreams without /responses.

decode_request maps input (a string or items: message, function_call, function_call_output, reasoning), instructions,
function tools, tool_choice, max_output_tokens, sampling, text.format (json_schema / json_object) and reasoning.effort to
the IR. Stateful features cannot be translated: previous_response_id is refused (400) and store=true is answered without
storing (note). Built-in tools (web_search, file_search, ...) are refused.

encode_stream emits the full event family with an increasing sequence_number, as the local responses.py does:
response.created, response.in_progress, then per output item response.output_item.added, response.content_part.added,
response.output_text.delta*, response.output_text.done, response.content_part.done, response.output_item.done; function
calls use an output item of type function_call with response.function_call_arguments.delta* and .done; reasoning uses a
reasoning item with response.reasoning_text.delta* and .done. The stream ends with response.completed,
response.incomplete (max tokens or content filter) or response.failed.
"""
from __future__ import annotations

import json
import time
import uuid

from ..envelopes import stream_terminal
from ..sse import write_json_sse
from ..types import Usage
from .common import StreamTail, closes_source, error_code, refuse
from .ir import (BlockStop, ErrorEvent, IRMessage, IRReasoning, IRRequest, IRResponse, Start, Text, TextDelta, Thinking,
                 ThinkingDelta, ToolArgsDelta, ToolCallStart, ToolResult, ToolUse)
from .openai_chat import content_parts, decode_tool_choice, decode_tools

_KNOWN = frozenset({'model', 'input', 'instructions', 'tools', 'tool_choice', 'parallel_tool_calls', 'max_output_tokens',
                    'temperature', 'top_p', 'text', 'reasoning', 'stream', 'store', 'metadata', 'user', 'truncation',
                    'previous_response_id', 'include', 'background', 'service_tier', 'worker', 'location', 'timeout'})


def _note(notes, text):
    if notes is not None and text not in notes:
        notes.append(text)


def decode_request(body: dict, *, notes: list | None = None) -> IRRequest:
    """IRRequest from a /v1/responses body (translated paths only)."""
    if not isinstance(body, dict):
        raise refuse('The request body must be a JSON object', 'invalid_request')
    if body.get('previous_response_id'):
        raise refuse('previous_response_id needs a stateful OpenAI upstream; it cannot be translated')
    if body.get('background'):
        raise refuse('background responses need an OpenAI upstream')
    if body.get('store'):
        _note(notes, 'not_stored:store')
    raw = body.get('input')
    items = [{'type': 'message', 'role': 'user', 'content': raw}] if isinstance(raw, str) else raw
    if not isinstance(items, list) or not items:
        raise refuse('input must be a string or a non-empty list of items', 'invalid_request')
    system = []
    if isinstance(body.get('instructions'), str) and body['instructions']:
        system.append(Text(body['instructions']))
    messages: list[IRMessage] = []

    def add(role, part):
        if messages and messages[-1].role == role and role != 'user':
            messages[-1] = IRMessage(role, list(messages[-1].parts) + [part])
        else:
            messages.append(IRMessage(role, [part]))

    for item in items:
        if not isinstance(item, dict):
            raise refuse('input items must be objects', 'invalid_request')
        kind = item.get('type', 'message')
        if kind == 'message':
            role = item.get('role')
            parts = content_parts(item.get('content'), role=role or 'user')
            if role in ('system', 'developer'):
                system.extend(p for p in parts if isinstance(p, Text))
            elif role in ('user', 'assistant'):
                messages.append(IRMessage(role, parts))
            else:
                raise refuse(f'input role {role!r} is not supported', 'invalid_request')
        elif kind == 'function_call':
            add('assistant', ToolUse(id=str(item.get('call_id')), name=str(item.get('name')),
                                     input_json=item.get('arguments') or '{}'))
        elif kind == 'function_call_output':
            output = item.get('output')
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            add('tool', ToolResult(tool_use_id=str(item.get('call_id')), parts=[Text(text)]))
        elif kind == 'reasoning':
            texts = [c.get('text', '') for c in item.get('content') or [] if isinstance(c, dict)]
            if any(texts) or item.get('encrypted_content'):
                add('assistant', Thinking(''.join(texts), item.get('encrypted_content')))
        else:
            raise refuse(f'input item type {kind!r} cannot be translated for this target')
    for tool in body.get('tools') or []:
        if not isinstance(tool, dict) or tool.get('type') != 'function':
            raise refuse('Only function tools can be translated for this target')
    text_config = body.get('text') or {}
    fmt = text_config.get('format') if isinstance(text_config, dict) else None
    response_format = None
    if isinstance(fmt, dict) and fmt.get('type') == 'json_schema':
        response_format = {'type': 'json_schema', 'json_schema': {
            'name': fmt.get('name') or 'output', 'schema': fmt.get('schema') or {},
            **({'strict': fmt['strict']} if 'strict' in fmt else {})}}
    elif isinstance(fmt, dict) and fmt.get('type') == 'json_object':
        response_format = {'type': 'json_object'}
    reasoning = body.get('reasoning') or {}
    for name in body:
        if name not in _KNOWN:
            _note(notes, f'dropped:{name}')
    for name in ('metadata', 'user', 'include', 'truncation', 'service_tier'):
        if body.get(name):
            _note(notes, f'dropped:{name}')
    extras = {}
    if body.get('parallel_tool_calls') is not None:
        extras['parallel_tool_calls'] = bool(body['parallel_tool_calls'])
    sampling = {k: body[k] for k in ('temperature', 'top_p') if body.get(k) is not None}
    return IRRequest(kind='chat', model=str(body.get('model') or ''), system=system, messages=messages,
                     tools=decode_tools(body.get('tools')), tool_choice=decode_tool_choice(body.get('tool_choice')),
                     sampling=sampling,
                     max_output_tokens=body['max_output_tokens'] if isinstance(body.get('max_output_tokens'), int) else None,
                     response_format=response_format,
                     reasoning=IRReasoning(effort=reasoning['effort']) if reasoning.get('effort') else None,
                     stream=bool(body.get('stream')), extras_by_protocol={'openai': extras, 'notes': list(notes or [])})


def response_object(*, model: str, request: dict | None = None, response_id: str | None = None,
                    created_at: int | None = None) -> dict:
    """The base response object the encoders fill (status in_progress, empty output)."""
    request = request or {}
    text = request.get('text') if isinstance(request.get('text'), dict) else {'format': {'type': 'text'}}
    return {'id': response_id or 'resp_' + uuid.uuid4().hex, 'object': 'response',
            'created_at': int(time.time()) if created_at is None else created_at, 'status': 'in_progress',
            'background': False, 'error': None, 'incomplete_details': None,
            'instructions': request.get('instructions'), 'max_output_tokens': request.get('max_output_tokens'),
            'model': model, 'output': [], 'parallel_tool_calls': request.get('parallel_tool_calls', True),
            'previous_response_id': None, 'reasoning': {'effort': (request.get('reasoning') or {}).get('effort'),
                                                        'summary': None},
            'store': False, 'temperature': request.get('temperature'), 'text': text,
            'tool_choice': request.get('tool_choice', 'auto'), 'tools': request.get('tools') or [],
            'top_p': request.get('top_p'), 'truncation': 'disabled', 'usage': None, 'user': None, 'metadata': {}}


def usage_object(usage: Usage | None) -> dict:
    usage = usage or Usage()
    inp, out = usage.input_tokens or 0, usage.output_tokens or 0
    return {'input_tokens': inp, 'input_tokens_details': {'cached_tokens': usage.cached_input_tokens or 0},
            'output_tokens': out, 'output_tokens_details': {'reasoning_tokens': usage.reasoning_tokens or 0},
            'total_tokens': inp + out}


def _final(response: dict, output: list, stop: str | None, usage: Usage | None) -> dict:
    final = dict(response)
    final['output'] = output
    final['usage'] = usage_object(usage)
    if stop == 'max_tokens':
        final['status'], final['incomplete_details'] = 'incomplete', {'reason': 'max_output_tokens'}
    elif stop == 'content_filter':
        final['status'], final['incomplete_details'] = 'incomplete', {'reason': 'content_filter'}
    else:
        final['status'] = 'completed'
    return final


def _message_item(item_id: str, text: str, status: str) -> dict:
    return {'id': item_id, 'type': 'message', 'status': status, 'role': 'assistant',
            'content': [{'type': 'output_text', 'annotations': [], 'logprobs': [], 'text': text}] if status != 'in_progress'
            else []}


def _part(text: str) -> dict:
    return {'type': 'output_text', 'annotations': [], 'logprobs': [], 'text': text}


def encode_response(resp: IRResponse, response_obj: dict) -> dict:
    """A complete response object from an IRResponse."""
    output = []
    suffix = uuid.uuid4().hex[:24]
    thinking = [p for p in resp.content if isinstance(p, Thinking)]
    if thinking:
        item = {'id': f'rs_{suffix}', 'type': 'reasoning', 'summary': [],
                'content': [{'type': 'reasoning_text', 'text': ''.join(p.text for p in thinking)}]}
        signature = next((p.signature for p in thinking if p.signature), None)
        if signature:
            item['encrypted_content'] = signature
        output.append(item)
    text = ''.join(p.text for p in resp.content if isinstance(p, Text))
    if text or not any(isinstance(p, ToolUse) for p in resp.content):
        output.append(_message_item(f'msg_{suffix}', text, 'completed'))
    for index, call in enumerate(p for p in resp.content if isinstance(p, ToolUse)):
        output.append({'id': f'fc_{suffix}{index}', 'type': 'function_call', 'status': 'completed',
                       'arguments': call.input_json or '{}', 'call_id': call.id, 'name': call.name})
    return _final(response_obj, output, resp.stop, resp.usage)


@closes_source
async def encode_stream(events, response_obj: dict):
    """The Responses SSE event family with an increasing sequence_number."""
    sequence = 0
    response = dict(response_obj)
    tail = StreamTail()
    output: list = []
    suffix = uuid.uuid4().hex[:24]
    blocks: dict[int, dict] = {}  # IR index -> {kind, output_index, item_id, text | name, call_id, args}
    created = False

    def event(kind, **fields):
        nonlocal sequence
        frame = write_json_sse({'type': kind, 'sequence_number': sequence, **fields}, kind)
        sequence += 1
        return frame

    def lifecycle():
        return [event('response.created', response=response), event('response.in_progress', response=response)]

    def close(index):
        block = blocks.pop(index, None)
        if block is None:
            return []
        frames = []
        position = block['output_index']
        if block['kind'] == 'text':
            item = _message_item(block['item_id'], block['text'], 'completed')
            ids = {'item_id': block['item_id'], 'output_index': position, 'content_index': 0}
            frames.append(event('response.output_text.done', **ids, text=block['text'], logprobs=[]))
            frames.append(event('response.content_part.done', **ids, part=_part(block['text'])))
        elif block['kind'] == 'reasoning':
            item = {'id': block['item_id'], 'type': 'reasoning', 'summary': [],
                    'content': [{'type': 'reasoning_text', 'text': block['text']}]}
            if block.get('signature'):
                item['encrypted_content'] = block['signature']
            frames.append(event('response.reasoning_text.done', item_id=block['item_id'], output_index=position,
                                content_index=0, text=block['text']))
        else:
            args = block['args'] or '{}'  # a call that takes no arguments streams none (Anthropic input_json_delta '')
            item = {'id': block['item_id'], 'type': 'function_call', 'status': 'completed', 'arguments': args,
                    'call_id': block['call_id'], 'name': block['name']}
            frames.append(event('response.function_call_arguments.done', item_id=block['item_id'],
                                output_index=position, arguments=args))
        output[position] = item
        frames.append(event('response.output_item.done', output_index=position, item=item))
        return frames

    def open_block(index, kind, **data):
        frames = []
        for other in [i for i, b in blocks.items() if b['kind'] != 'tool' or kind != 'tool']:
            if other != index:
                frames.extend(close(other))
        position = len(output)
        item_id = {'text': 'msg_', 'reasoning': 'rs_', 'tool': 'fc_'}[kind] + suffix + str(position)
        block = {'kind': kind, 'output_index': position, 'item_id': item_id, 'text': '', 'args': '', **data}
        blocks[index] = block
        if kind == 'text':
            item = _message_item(item_id, '', 'in_progress')
            output.append(item)
            frames.append(event('response.output_item.added', output_index=position, item=item))
            frames.append(event('response.content_part.added', item_id=item_id, output_index=position, content_index=0,
                                part=_part('')))
        elif kind == 'reasoning':
            item = {'id': item_id, 'type': 'reasoning', 'summary': [], 'content': []}
            output.append(item)
            frames.append(event('response.output_item.added', output_index=position, item=item))
        else:
            item = {'id': item_id, 'type': 'function_call', 'status': 'in_progress', 'arguments': '',
                    'call_id': data['call_id'], 'name': data['name']}
            output.append(item)
            frames.append(event('response.output_item.added', output_index=position, item=item))
        return frames

    async for ev in events:
        if not created:
            if isinstance(ev, ErrorEvent):
                tail.take(ev)
                break
            created = True
            for frame in lifecycle():
                yield frame
        if isinstance(ev, Start):
            continue
        if isinstance(ev, TextDelta):
            if ev.index not in blocks or blocks[ev.index]['kind'] != 'text':
                for frame in open_block(ev.index, 'text'):
                    yield frame
            block = blocks[ev.index]
            if ev.text:
                block['text'] += ev.text
                yield event('response.output_text.delta', item_id=block['item_id'], output_index=block['output_index'],
                            content_index=0, delta=ev.text, logprobs=[])
        elif isinstance(ev, ThinkingDelta):
            if ev.index not in blocks or blocks[ev.index]['kind'] != 'reasoning':
                for frame in open_block(ev.index, 'reasoning'):
                    yield frame
            block = blocks[ev.index]
            if ev.signature:
                block['signature'] = ev.signature
            if ev.text:
                block['text'] += ev.text
                yield event('response.reasoning_text.delta', item_id=block['item_id'],
                            output_index=block['output_index'], content_index=0, delta=ev.text)
        elif isinstance(ev, ToolCallStart):
            for frame in open_block(ev.index, 'tool', call_id=ev.id, name=ev.name):
                yield frame
        elif isinstance(ev, ToolArgsDelta):
            block = blocks.get(ev.index)
            if block is not None and ev.partial_json:
                block['args'] += ev.partial_json
                yield event('response.function_call_arguments.delta', item_id=block['item_id'],
                            output_index=block['output_index'], delta=ev.partial_json)
        elif isinstance(ev, BlockStop):
            for frame in close(ev.index):
                yield frame
        elif tail.take(ev) and isinstance(ev, ErrorEvent):
            break
    if tail.error is not None:
        if not created:
            for frame in lifecycle():
                yield frame
        failed = dict(response, output=output)
        yield stream_terminal('responses', error_code(tail.error.code), tail.error.message, sequence_number=sequence,
                              response=failed)
        return
    if not created:
        for frame in lifecycle():
            yield frame
    for index in sorted(blocks, key=lambda i: blocks[i]['output_index']):
        for frame in close(index):
            yield frame
    final = _final(response, output, tail.stop, tail.usage)
    yield event('response.' + final['status'], response=final)


__all__ = ['decode_request', 'encode_response', 'encode_stream', 'response_object', 'usage_object']
