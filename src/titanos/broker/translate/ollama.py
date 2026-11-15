"""Ollama /api/chat, /api/generate and /api/embed codec.

It serves Ollama clients reaching cloud targets and decodes Ollama Cloud NDJSON for other inbound protocols.
- Requests: messages (system, user, assistant, tool) with images (base64 list), tool_calls (arguments are objects, no
  ids: ids are synthesized as call_<index>_<8hex> and tool messages are matched back by tool_name or order), thinking;
  tools (OpenAI-shaped function tools); format ('json' or a JSON schema); think; options mapped to sampling: num_predict
  becomes the max tokens, temperature, top_p, top_k, seed and stop are carried over, other options and keep_alive are
  dropped with a note.
- Output: NDJSON lines with done:false, then a done:true line with done_reason, prompt_eval_count and eval_count. Tool-call
  arguments are emitted as complete objects (buffered until the call's BlockStop; documented exception to the one-frame
  rule); invalid buffered arguments end the stream with an {"error": ...} line.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ..envelopes import stream_terminal
from ..sse import ollama_usage
from ..types import Usage
from .common import (StreamTail, b64, bad_upstream, closes_source, error_code, refuse, request_args, stream_error_event,
                     synth_tool_id, unb64)
from .ir import (BlockStop, ErrorEvent, Image, IRMessage, IRReasoning, IRRequest, IRResponse, Start, Stop, Text, TextDelta,
                 Thinking, ThinkingDelta, ToolArgsDelta, ToolCallStart, ToolResult, ToolUse, UsageEvent)
from .openai_chat import decode_tools, encode_tools

OPTION_MAP = {'temperature': 'temperature', 'top_p': 'top_p', 'top_k': 'top_k', 'seed': 'seed',
              'presence_penalty': 'presence_penalty', 'frequency_penalty': 'frequency_penalty'}
DONE_FROM = {'stop': 'end_turn', 'length': 'max_tokens'}
DONE_TO = {'end_turn': 'stop', 'stop_sequence': 'stop', 'tool_use': 'stop', 'max_tokens': 'length',
           'content_filter': 'stop'}
_CHAT_KNOWN = frozenset({'model', 'messages', 'tools', 'format', 'options', 'stream', 'keep_alive', 'think', 'worker',
                         'location', 'timeout'})
_GENERATE_KNOWN = frozenset({'model', 'prompt', 'system', 'images', 'format', 'options', 'stream', 'keep_alive', 'think',
                             'suffix', 'raw', 'template', 'context', 'worker', 'location', 'timeout'})


def _note(notes, text):
    if notes is not None and text not in notes:
        notes.append(text)


def _now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _images(values) -> list:
    out = []
    for value in values or []:
        if not isinstance(value, str):
            raise refuse('images must be base64 strings', 'invalid_request')
        data = unb64(value)
        media = 'image/png' if data[:8] == b'\x89PNG\r\n\x1a\n' else 'image/jpeg' if data[:2] == b'\xff\xd8' else \
            'image/webp' if data[8:12] == b'WEBP' else 'image/gif' if data[:4] == b'GIF8' else 'image/png'
        out.append(Image(media_type=media, data=data))
    return out


def _format(value, notes) -> dict | None:
    if value in (None, ''):
        return None
    if value == 'json':
        return {'type': 'json_object'}
    if isinstance(value, dict):
        return {'type': 'json_schema', 'json_schema': {'name': 'output', 'schema': value}}
    raise refuse('format must be "json" or a JSON schema', 'invalid_request')


def _options(body: dict, notes) -> tuple[dict, int | None, list | None]:
    options = body.get('options') or {}
    if not isinstance(options, dict):
        raise refuse('options must be an object', 'invalid_request')
    sampling, max_tokens, stop = {}, None, None
    for name, value in options.items():
        if name == 'num_predict':
            max_tokens = value if isinstance(value, int) and value > 0 else None
        elif name == 'stop':
            stop = [value] if isinstance(value, str) else list(value or [])
        elif name in OPTION_MAP:
            sampling[OPTION_MAP[name]] = value
        else:
            _note(notes, f'dropped:options.{name}')
    if body.get('keep_alive') is not None:
        _note(notes, 'dropped:keep_alive')
    return sampling, max_tokens, stop


def _think(value) -> IRReasoning | None:
    if value is True:
        return IRReasoning(enabled=True)
    if isinstance(value, str) and value in ('low', 'medium', 'high'):
        return IRReasoning(enabled=True, effort=value)
    return None


def decode_request(body: dict, *, generate: bool, notes: list | None = None) -> IRRequest:
    """IRRequest from an /api/chat (generate=False) or /api/generate (generate=True) body."""
    if not isinstance(body, dict):
        raise refuse('The request body must be a JSON object', 'invalid_request')
    sampling, max_tokens, stop = _options(body, notes)
    for name in body:
        if name not in (_GENERATE_KNOWN if generate else _CHAT_KNOWN):
            _note(notes, f'dropped:{name}')
    system, messages = [], []
    if generate:
        if body.get('raw') or body.get('template') or body.get('suffix') or body.get('context'):
            raise refuse('raw, template, suffix and context cannot be translated for this target')
        if isinstance(body.get('system'), str) and body['system']:
            system.append(Text(body['system']))
        prompt = body.get('prompt')
        if not isinstance(prompt, str):
            raise refuse('prompt is required', 'invalid_request')
        messages.append(IRMessage('user', ([Text(prompt)] if prompt else []) + _images(body.get('images'))))
        tools, choice = [], None
    else:
        ids: dict[str, list] = {}
        order: list[str] = []
        counter = 0
        for message in body.get('messages') or []:
            role = message.get('role')
            content = message.get('content') or ''
            if role == 'system':
                system.append(Text(content))
            elif role == 'user':
                messages.append(IRMessage('user', ([Text(content)] if content else []) + _images(message.get('images'))))
            elif role == 'assistant':
                parts = []
                if message.get('thinking'):
                    parts.append(Thinking(message['thinking']))
                if content:
                    parts.append(Text(content))
                for call in message.get('tool_calls') or []:
                    fn = call.get('function') or {}
                    name, args = str(fn.get('name') or ''), fn.get('arguments') or {}
                    call_id = call.get('id') or synth_tool_id(counter, name, args)
                    counter += 1
                    ids.setdefault(name, []).append(call_id)
                    order.append(call_id)
                    parts.append(ToolUse(call_id, name, args if isinstance(args, str) else json.dumps(
                        args, ensure_ascii=False, separators=(',', ':'))))
                messages.append(IRMessage('assistant', parts))
            elif role == 'tool':
                name = message.get('tool_name') or message.get('name')
                queue = ids.get(name) if name else None
                if queue:
                    call_id = queue.pop(0)
                    order.remove(call_id)
                elif order:
                    call_id = order.pop(0)
                else:
                    raise refuse('A tool message has no matching tool call', 'invalid_request')
                part = ToolResult(call_id, [Text(content)] if content else [])
                if messages and messages[-1].role == 'tool':
                    messages[-1] = IRMessage('tool', list(messages[-1].parts) + [part])
                else:
                    messages.append(IRMessage('tool', [part]))
            else:
                raise refuse(f'message role {role!r} is not supported', 'invalid_request')
        tools = decode_tools(body.get('tools'))
        choice = None
    return IRRequest(kind='chat', model=str(body.get('model') or ''), system=system, messages=messages, tools=tools,
                     tool_choice=choice, sampling=sampling, max_output_tokens=max_tokens, stop=stop,
                     response_format=_format(body.get('format'), notes), reasoning=_think(body.get('think')),
                     stream=body.get('stream', True) is not False,
                     extras_by_protocol={'ollama': {'generate': generate}, 'notes': list(notes or [])})


def encode_request(ir: IRRequest, profile=None, *, generate: bool = False, notes: list | None = None) -> dict:
    """An Ollama /api/chat body (or /api/generate for a single-prompt request when generate=True)."""
    options = {}
    for name, value in ir.sampling.items():
        if name in OPTION_MAP.values():
            options[name] = value
    if ir.max_output_tokens is not None:
        options['num_predict'] = ir.max_output_tokens
    if ir.stop:
        options['stop'] = list(ir.stop)
    body: dict = {'model': ir.model}
    if generate:
        user = ir.messages[-1] if ir.messages else IRMessage('user', [])
        body['prompt'] = ''.join(p.text for p in user.parts if isinstance(p, Text))
        if ir.system:
            body['system'] = '\n\n'.join(p.text for p in ir.system if isinstance(p, Text))
        images = [b64(p.data) for p in user.parts if isinstance(p, Image) and p.data is not None]
        if images:
            body['images'] = images
    else:
        messages = []
        names: dict[str, str] = {}
        if ir.system:
            messages.append({'role': 'system', 'content': '\n\n'.join(p.text for p in ir.system if isinstance(p, Text))})
        for message in ir.messages:
            results = [p for p in message.parts if isinstance(p, ToolResult)]
            for part in results:
                messages.append({'role': 'tool', 'content': ''.join(p.text for p in part.parts if isinstance(p, Text)),
                                 **({'tool_name': names[part.tool_use_id]} if part.tool_use_id in names else {})})
            rest = [p for p in message.parts if not isinstance(p, ToolResult)]
            if message.role == 'tool' or (results and not rest):
                continue
            out: dict = {'role': message.role, 'content': ''.join(p.text for p in rest if isinstance(p, Text))}
            images = []
            for part in rest:
                if isinstance(part, Image):
                    if part.data is None:
                        raise refuse('Image URLs cannot be sent to Ollama; send the image inline (base64)')
                    images.append(b64(part.data))
                elif isinstance(part, Thinking) and part.text:
                    out['thinking'] = out.get('thinking', '') + part.text
                elif isinstance(part, ToolUse):
                    names[part.id] = part.name
                    out.setdefault('tool_calls', []).append({'function': {'name': part.name,
                                                                          'arguments': request_args(part.input_json)}})
            if images:
                out['images'] = images
            messages.append(out)
        body['messages'] = messages
        if ir.tools:
            body['tools'] = encode_tools(ir.tools)
        if ir.tool_choice is not None and ir.tool_choice.mode not in ('auto',):
            _note(notes, 'dropped:tool_choice')
    if ir.response_format is not None:
        kind = ir.response_format.get('type')
        body['format'] = 'json' if kind == 'json_object' else (ir.response_format.get('json_schema') or {}).get('schema')
    if ir.reasoning is not None:
        body['think'] = ir.reasoning.effort if ir.reasoning.effort in ('low', 'medium', 'high') else True
    if options:
        body['options'] = options
    body['stream'] = bool(ir.stream)
    return body


# ---------------------------------------------------------------------------------------------------------------------
# responses

def decode_response(body: dict) -> IRResponse:
    if not isinstance(body, dict):
        raise bad_upstream('The upstream Ollama response is not an object')
    if 'error' in body:
        raise bad_upstream('The upstream Ollama response is an error')
    parts = []
    message = body.get('message') or {}
    if message.get('thinking') or body.get('thinking'):
        parts.append(Thinking(message.get('thinking') or body.get('thinking')))
    text = message.get('content') if 'message' in body else body.get('response')
    if text:
        parts.append(Text(text))
    for index, call in enumerate(message.get('tool_calls') or []):
        fn = call.get('function') or {}
        args = fn.get('arguments') or {}
        parts.append(ToolUse(call.get('id') or synth_tool_id(index, fn.get('name', ''), args), str(fn.get('name') or ''),
                             args if isinstance(args, str) else json.dumps(args, ensure_ascii=False,
                                                                           separators=(',', ':'))))
    called = any(isinstance(p, ToolUse) for p in parts)
    stop = 'tool_use' if called else DONE_FROM.get(body.get('done_reason'), 'end_turn')
    return IRResponse(id=uuid.uuid4().hex, model=str(body.get('model') or ''), content=parts, stop=stop,
                      usage=ollama_usage(dict(body, done=True)) or Usage())


def _done_fields(stop: str | None, usage: Usage | None) -> dict:
    usage = usage or Usage()
    return {'done_reason': DONE_TO.get(stop or 'end_turn', 'stop'), 'done': True, 'total_duration': 0,
            'load_duration': 0, 'prompt_eval_count': usage.input_tokens or 0, 'prompt_eval_duration': 0,
            'eval_count': usage.output_tokens or 0, 'eval_duration': 0}


def encode_response(resp: IRResponse, *, model_name: str, generate: bool) -> dict:
    text = ''.join(p.text for p in resp.content if isinstance(p, Text))
    thinking = ''.join(p.text for p in resp.content if isinstance(p, Thinking))
    out: dict = {'model': model_name, 'created_at': _now()}
    if generate:
        out['response'] = text
        if thinking:
            out['thinking'] = thinking
    else:
        message: dict = {'role': 'assistant', 'content': text}
        if thinking:
            message['thinking'] = thinking
        calls = [p for p in resp.content if isinstance(p, ToolUse)]
        if calls:
            message['tool_calls'] = [{'function': {'name': c.name, 'arguments': json.loads(c.input_json or '{}')}}
                                     for c in calls]
        out['message'] = message
    out.update(_done_fields(resp.stop, resp.usage))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# streams

@closes_source
async def decode_stream(lines):
    """IR events from Ollama NDJSON lines (chat or generate)."""
    started = False
    text_index = thought_index = None
    next_index = 0
    called = False
    counter = 0
    async for line in lines:
        if not isinstance(line, dict):
            raise bad_upstream('The upstream stream sent a line that is not an object')
        if 'error' in line:
            yield stream_error_event({'error': {'message': line['error'], 'type': 'api_error'}}, adapter='ollama_native')
            return
        if not started:
            started = True
            yield Start(id='ollama-' + uuid.uuid4().hex, model=str(line.get('model') or ''))
        message = line.get('message') if isinstance(line.get('message'), dict) else {}
        thinking = message.get('thinking') or line.get('thinking')
        if thinking:
            if thought_index is None:
                thought_index, next_index = next_index, next_index + 1
            yield ThinkingDelta(thought_index, thinking)
        text = message.get('content') if 'message' in line else line.get('response')
        if text:
            if thought_index is not None:
                yield BlockStop(thought_index)
                thought_index = None
            if text_index is None:
                text_index, next_index = next_index, next_index + 1
            yield TextDelta(text_index, text)
        for call in message.get('tool_calls') or []:
            for open_index in (thought_index, text_index):
                if open_index is not None:
                    yield BlockStop(open_index)
            thought_index = text_index = None
            fn = call.get('function') or {}
            name, args = str(fn.get('name') or ''), fn.get('arguments') or {}
            index, next_index = next_index, next_index + 1
            yield ToolCallStart(index, call.get('id') or synth_tool_id(counter, name, args), name)
            counter += 1
            called = True
            yield ToolArgsDelta(index, args if isinstance(args, str) else json.dumps(args, ensure_ascii=False,
                                                                                    separators=(',', ':')))
            yield BlockStop(index)
        if line.get('done'):
            for open_index in (thought_index, text_index):
                if open_index is not None:
                    yield BlockStop(open_index)
            thought_index = text_index = None
            yield Stop('tool_use' if called else DONE_FROM.get(line.get('done_reason'), 'end_turn'))
            usage = ollama_usage(line)
            if usage is not None:
                yield UsageEvent(usage)
            return
    for open_index in (thought_index, text_index):
        if open_index is not None:
            yield BlockStop(open_index)


def _line(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')


@closes_source
async def encode_ndjson_stream(events, *, model_name: str, generate: bool):
    """NDJSON lines: one per text (or thinking) delta, a tool_calls line per completed call, then the done:true line."""
    tail = StreamTail()
    tools: dict[int, list] = {}

    def line(**fields):
        base = {'model': model_name, 'created_at': _now()}
        base.update(fields)
        return _line(base)

    async for event in events:
        if isinstance(event, TextDelta) and event.text:
            if generate:
                yield line(response=event.text, done=False)
            else:
                yield line(message={'role': 'assistant', 'content': event.text}, done=False)
        elif isinstance(event, ThinkingDelta) and event.text:
            if generate:
                yield line(response='', thinking=event.text, done=False)
            else:
                yield line(message={'role': 'assistant', 'content': '', 'thinking': event.text}, done=False)
        elif isinstance(event, ToolCallStart):
            tools[event.index] = [event.name, []]
        elif isinstance(event, ToolArgsDelta):
            if event.index in tools:
                tools[event.index][1].append(event.partial_json)
        elif isinstance(event, BlockStop):
            if event.index in tools:
                name, fragments = tools.pop(event.index)
                text = ''.join(fragments)
                try:
                    args = json.loads(text) if text.strip() else {}
                except ValueError:
                    yield stream_terminal('ollama', 'upstream_bad_response', 'Tool-call arguments are not valid JSON')
                    return
                if generate:
                    continue
                yield line(message={'role': 'assistant', 'content': '',
                                    'tool_calls': [{'function': {'name': name, 'arguments': args}}]}, done=False)
        elif tail.take(event) and isinstance(event, ErrorEvent):
            yield stream_terminal('ollama', error_code(event.code), event.message)
            return
    done = _done_fields(tail.stop, tail.usage)
    if generate:
        yield line(response='', **done)
    else:
        yield line(message={'role': 'assistant', 'content': ''}, **done)


# ---------------------------------------------------------------------------------------------------------------------
# embeddings

def decode_embed_request(body: dict, *, notes: list | None = None) -> IRRequest:
    """IRRequest (kind embed) from /api/embed ({input: str | [str]}) or the legacy /api/embeddings ({prompt})."""
    if not isinstance(body, dict):
        raise refuse('The request body must be a JSON object', 'invalid_request')
    value = body.get('input', body.get('prompt'))
    inputs = [value] if isinstance(value, str) else value
    if not isinstance(inputs, list) or not inputs or not all(isinstance(v, str) for v in inputs):
        raise refuse('input must be a string or a list of strings', 'invalid_request')
    embed = {}
    if body.get('dimensions') is not None:
        embed['dimensions'] = body['dimensions']
    if body.get('truncate') is not None:
        embed['truncate'] = body['truncate']
    if body.get('keep_alive') is not None or body.get('options'):
        _note(notes, 'dropped:keep_alive/options')
    return IRRequest(kind='embed', model=str(body.get('model') or ''), inputs=inputs,
                     extras_by_protocol={'embed': embed, 'legacy': 'prompt' in body and 'input' not in body,
                                         'notes': list(notes or [])})


def embed_request(ir: IRRequest) -> dict:
    embed = ir.extras_by_protocol.get('embed') or {}
    body = {'model': ir.model, 'input': list(ir.inputs or [])}
    for name in ('dimensions', 'truncate'):
        if embed.get(name) is not None:
            body[name] = embed[name]
    return body


def embed_response(body: dict) -> dict:
    """{'vectors': [[float]], 'usage': Usage} from an /api/embed response."""
    if not isinstance(body, dict) or not isinstance(body.get('embeddings'), list):
        raise bad_upstream('The upstream embed response has no embeddings')
    return {'vectors': body['embeddings'], 'usage': Usage(input_tokens=body.get('prompt_eval_count'))}


def encode_embed_response(vectors, usage: Usage | None, *, model_name: str, legacy: bool = False) -> dict:
    if legacy:
        return {'embedding': list(vectors[0])}
    return {'model': model_name, 'embeddings': [list(v) for v in vectors], 'total_duration': 0, 'load_duration': 0,
            'prompt_eval_count': (usage.input_tokens if usage else None) or 0}


__all__ = ['DONE_FROM', 'DONE_TO', 'OPTION_MAP', 'decode_embed_request', 'decode_request', 'decode_response',
           'decode_stream', 'embed_request', 'embed_response', 'encode_embed_response', 'encode_ndjson_stream',
           'encode_request', 'encode_response']
