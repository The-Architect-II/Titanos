"""OpenAI chat-completions and completions codec (plus the embeddings mapping).

Requests: system and developer messages are hoisted into IRRequest.system; user content parts (text, image_url with a
base64 data: URL or an https URL, input_audio) become IR parts; assistant tool_calls become ToolUse with their ids kept;
tool messages become ToolResult; reasoning_content becomes Thinking (the Titanos extension field reasoning_signature carries
an Anthropic thinking signature through OpenAI-shaped clients). n>1, logprobs, top_logprobs and logit_bias are refused on
every translated path (400 broker_capability_unavailable); other unknown fields are dropped with a note, never forwarded.

Streams: decode_stream turns chat.completion.chunk (and text_completion) frames into IR events; encode_stream emits
chat.completion.chunk frames per IR event (no buffering), a finish chunk, the usage chunk (choices []) when the client
asked for include_usage, then data: [DONE], which Client._stream requires. An ErrorEvent becomes the in-stream error frame
{"error": {...}} followed by [DONE].
"""
from __future__ import annotations

import base64
import mimetypes
import struct
import time
import uuid

from ..envelopes import stream_terminal
from ..sse import SSEFrame, openai_usage, write_json_sse
from ..types import Usage
from .common import (FROM_OPENAI_FINISH, OPENAI_FINISH, StreamTail, args_text, bad_upstream, closes_source, data_url,
                     error_code, parse_data_url, refuse, stream_error_event, synth_tool_id)
from .ir import (Audio, BlockStop, ErrorEvent, Image, IRMessage, IRReasoning, IRRequest, IRResponse, IRTool, IRToolChoice,
                 Start, Stop, Text, TextDelta, Thinking, ThinkingDelta, ToolArgsDelta, ToolCallStart, ToolResult, ToolUse,
                 UsageEvent)

TITANOS_ONLY = frozenset({'worker', 'location', 'timeout'})
SAMPLING_FIELDS = ('temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 'seed', 'top_k')
# accepted without a mapping: dropped with a note on translated paths
DROPPED_WITH_NOTE = frozenset({'user', 'metadata', 'store', 'service_tier', 'prediction', 'web_search_options',
                               'safety_identifier', 'prompt_cache_key', 'verbosity', 'functions', 'function_call'})
_CHAT_FIELDS = frozenset({'model', 'messages', 'tools', 'tool_choice', 'parallel_tool_calls', 'max_tokens',
                          'max_completion_tokens', 'stop', 'response_format', 'reasoning_effort', 'stream', 'stream_options',
                          'n', 'logprobs', 'top_logprobs', 'logit_bias', 'modalities', 'audio', *SAMPLING_FIELDS})
_COMPLETION_FIELDS = frozenset({'model', 'prompt', 'max_tokens', 'stop', 'stream', 'stream_options', 'n', 'logprobs',
                                'logit_bias', 'echo', 'best_of', 'suffix', *SAMPLING_FIELDS})


def _note(notes, text):
    if notes is not None and text not in notes:
        notes.append(text)


def refuse_unsupported(body: dict, fields=('n', 'logprobs', 'top_logprobs', 'logit_bias', 'best_of', 'echo')) -> None:
    """n>1, logprobs, top_logprobs and logit_bias cannot be translated faithfully: 400 before egress."""
    if isinstance(body.get('n'), int) and body['n'] > 1:
        raise refuse('n > 1 is not supported for this target')
    for name in fields:
        if name == 'n':
            continue
        value = body.get(name)
        if name == 'best_of' and isinstance(value, int) and value <= 1:
            continue
        if value not in (None, False, {}, [], 0) or (name == 'top_logprobs' and value is not None):
            raise refuse(f'{name} is not supported for this target')


def _sampling(body: dict) -> dict:
    return {name: body[name] for name in SAMPLING_FIELDS if body.get(name) is not None}


def _stop_list(value) -> list | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise refuse('stop must be a string or a list of strings', 'invalid_request')


def content_parts(content, *, role: str = 'user') -> list:
    """IR parts of an OpenAI message content (a string or a list of content parts)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [Text(content)] if content else []
    if not isinstance(content, list):
        raise refuse('message content must be a string or a list of parts', 'invalid_request')
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise refuse('message content parts must be objects', 'invalid_request')
        kind = part.get('type')
        if kind in ('text', 'input_text', 'output_text', 'refusal'):
            text = part.get('text') if kind != 'refusal' else part.get('refusal')
            if isinstance(text, str) and text:
                parts.append(Text(text))
        elif kind in ('image_url', 'input_image'):
            image = part.get('image_url')
            url = image.get('url') if isinstance(image, dict) else image
            if not isinstance(url, str) or not url:
                raise refuse('image_url needs a url', 'invalid_request')
            parts.append(image_part(url))
        elif kind == 'input_audio':
            audio = part.get('input_audio') or {}
            fmt = audio.get('format') or 'wav'
            try:
                data = base64.b64decode(audio.get('data') or '', validate=True)
            except (ValueError, TypeError):
                raise refuse('input_audio data is not valid base64', 'invalid_request') from None
            parts.append(Audio(media_type='audio/' + ('mpeg' if fmt == 'mp3' else fmt), data=data))
        else:
            raise refuse(f'content part type {kind!r} is not supported for this target')
    return parts


def image_part(url: str) -> Image:
    inline = parse_data_url(url)
    if inline is not None:
        return Image(media_type=inline[0], data=inline[1])
    if not url.startswith('https://'):
        raise refuse('Image URLs must be https or base64 data: URLs', 'invalid_request')
    return Image(media_type=mimetypes.guess_type(url.split('?', 1)[0])[0] or 'image/*', url=url)


def decode_tools(tools) -> list:
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get('type', 'function') != 'function':
            raise refuse('Only function tools can be translated for this target')
        fn = tool.get('function') if 'function' in tool else tool  # Responses tools are flat
        if not isinstance(fn, dict) or not isinstance(fn.get('name'), str):
            raise refuse('A function tool needs a name', 'invalid_request')
        out.append(IRTool(name=fn['name'], description=fn.get('description'),
                          parameters=fn.get('parameters') or {'type': 'object', 'properties': {}}))
    return out


def decode_tool_choice(value) -> IRToolChoice | None:
    if value is None:
        return None
    if value in ('auto', 'none', 'required'):
        return IRToolChoice(mode=value)
    if isinstance(value, dict):
        fn = value.get('function') if isinstance(value.get('function'), dict) else value
        if value.get('type') == 'function' and isinstance(fn.get('name'), str):
            return IRToolChoice(mode='tool', name=fn['name'])
        if value.get('type') in ('auto', 'none', 'required'):
            return IRToolChoice(mode=value['type'])
    raise refuse('tool_choice is not supported for this target')


def decode_messages(messages, *, notes=None) -> tuple[list, list]:
    """(system parts, IR messages) from OpenAI chat messages."""
    if not isinstance(messages, list) or not messages:
        raise refuse('messages must be a non-empty list', 'invalid_request')
    system, out = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise refuse('messages must be objects', 'invalid_request')
        role = message.get('role')
        if role in ('system', 'developer'):
            parts = content_parts(message.get('content'))
            if any(not isinstance(p, Text) for p in parts):
                raise refuse('system messages carry text only')
            system.extend(parts)
        elif role == 'user':
            out.append(IRMessage(role='user', parts=content_parts(message.get('content'))))
        elif role == 'assistant':
            parts = []
            reasoning = message.get('reasoning_content')
            signature = message.get('reasoning_signature')
            signature = signature if isinstance(signature, str) and signature else None
            if isinstance(reasoning, str) and reasoning:
                parts.append(Thinking(text=reasoning, signature=signature))
            elif signature:  # a signature-only thinking (a Gemini thoughtSignature carried on a tool call)
                parts.append(Thinking(text='', signature=signature))
            parts.extend(content_parts(message.get('content'), role='assistant'))
            for call in message.get('tool_calls') or []:
                if not isinstance(call, dict) or call.get('type', 'function') != 'function':
                    raise refuse('Only function tool calls can be translated')
                fn = call.get('function') or {}
                if not isinstance(fn.get('name'), str) or not isinstance(call.get('id'), str):
                    raise refuse('A tool call needs an id and a function name', 'invalid_request')
                parts.append(ToolUse(id=call['id'], name=fn['name'], input_json=args_text(fn.get('arguments') or '{}')))
            if message.get('function_call') is not None:
                raise refuse('The legacy function_call message field is not supported; use tool_calls')
            out.append(IRMessage(role='assistant', parts=parts))
        elif role == 'tool':
            if not isinstance(message.get('tool_call_id'), str):
                raise refuse('A tool message needs tool_call_id', 'invalid_request')
            out.append(IRMessage(role='tool', parts=[ToolResult(tool_use_id=message['tool_call_id'],
                                                                parts=content_parts(message.get('content')))]))
        else:
            raise refuse(f'message role {role!r} is not supported for this target')
    return system, out


def decode_request(body: dict, *, notes: list | None = None) -> IRRequest:
    """IRRequest (kind chat) from a /v1/chat/completions body."""
    if not isinstance(body, dict):
        raise refuse('The request body must be a JSON object', 'invalid_request')
    refuse_unsupported(body)
    if body.get('modalities') and 'audio' in body.get('modalities') or body.get('audio') is not None:
        raise refuse('Audio output is not supported for this target')
    system, messages = decode_messages(body.get('messages'), notes=notes)
    for name in body:
        if name in TITANOS_ONLY or name in _CHAT_FIELDS:
            continue
        _note(notes, f'dropped:{name}')
    options = body.get('stream_options') or {}
    effort = body.get('reasoning_effort')
    extras = {'include_usage': bool(options.get('include_usage')) if isinstance(options, dict) else False}
    if body.get('parallel_tool_calls') is not None:
        extras['parallel_tool_calls'] = bool(body['parallel_tool_calls'])
    max_tokens = body.get('max_completion_tokens', body.get('max_tokens'))
    return IRRequest(kind='chat', model=str(body.get('model') or ''), system=system, messages=messages,
                     tools=decode_tools(body.get('tools')), tool_choice=decode_tool_choice(body.get('tool_choice')),
                     sampling=_sampling(body), max_output_tokens=max_tokens if isinstance(max_tokens, int) else None,
                     stop=_stop_list(body.get('stop')), response_format=body.get('response_format'),
                     reasoning=IRReasoning(enabled=True, effort=effort) if isinstance(effort, str) else None,
                     stream=bool(body.get('stream')), extras_by_protocol={'openai': extras, 'notes': list(notes or [])})


def decode_completion_request(body: dict, *, notes: list | None = None) -> IRRequest:
    """IRRequest from a /v1/completions body: the prompt becomes one user message (openai.completions -> openai.chat)."""
    if not isinstance(body, dict):
        raise refuse('The request body must be a JSON object', 'invalid_request')
    refuse_unsupported(body)
    prompt = body.get('prompt')
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
        prompt = prompt[0]
    if not isinstance(prompt, str):
        raise refuse('Only a single text prompt can be translated to a chat target')
    if body.get('suffix'):
        raise refuse('suffix is not supported for this target')
    for name in body:
        if name not in TITANOS_ONLY and name not in _COMPLETION_FIELDS:
            _note(notes, f'dropped:{name}')
    _note(notes, 'translated:openai.completions->openai.chat')
    options = body.get('stream_options') or {}
    return IRRequest(kind='chat', model=str(body.get('model') or ''), messages=[IRMessage('user', [Text(prompt)])],
                     sampling=_sampling(body),
                     max_output_tokens=body['max_tokens'] if isinstance(body.get('max_tokens'), int) else None,
                     stop=_stop_list(body.get('stop')), stream=bool(body.get('stream')),
                     extras_by_protocol={'openai': {'completion': True, 'include_usage': bool(
                         options.get('include_usage')) if isinstance(options, dict) else False},
                         'notes': list(notes or [])})


# ---------------------------------------------------------------------------------------------------------------------
# encode (IR -> an OpenAI-compatible upstream request)

def _encode_content(parts, *, notes) -> str | list:
    if all(isinstance(p, Text) for p in parts):
        return ''.join(p.text for p in parts)
    out = []
    for part in parts:
        if isinstance(part, Text):
            out.append({'type': 'text', 'text': part.text})
        elif isinstance(part, Image):
            url = part.url if part.url is not None else data_url(part.media_type, part.data or b'')
            out.append({'type': 'image_url', 'image_url': {'url': url}})
        elif isinstance(part, Audio):
            fmt = part.media_type.split('/', 1)[-1]
            out.append({'type': 'input_audio', 'input_audio': {'data': base64.b64encode(part.data).decode('ascii'),
                                                               'format': 'mp3' if fmt == 'mpeg' else fmt}})
        elif isinstance(part, Thinking):
            _note(notes, 'dropped:thinking')
        else:
            raise refuse(f'{type(part).__name__} content cannot be sent to this target')
    return out


def encode_messages(ir: IRRequest, *, notes=None, reasoning_roundtrip: bool = False) -> list:
    """OpenAI chat messages. reasoning_roundtrip (profile extras reasoning_content_roundtrip, DeepSeek thinking mode): an
    assistant turn's thinking is sent back as reasoning_content instead of being dropped (DeepSeek answers 400 in a tool
    loop when it is missing)."""
    messages = []
    if ir.system:
        messages.append({'role': 'system', 'content': ''.join(p.text for p in ir.system if isinstance(p, Text))})
    for message in ir.messages:
        if message.role == 'tool':
            for part in message.parts:
                if not isinstance(part, ToolResult):
                    raise refuse('tool messages carry tool results only')
                text = ''.join(p.text for p in part.parts if isinstance(p, Text))
                if any(not isinstance(p, Text) for p in part.parts):
                    _note(notes, 'dropped:tool_result_media')
                messages.append({'role': 'tool', 'tool_call_id': part.tool_use_id, 'content': text})
            continue
        if message.role == 'user':
            results = [p for p in message.parts if isinstance(p, ToolResult)]
            for part in results:  # a user turn that carries tool results (Anthropic shape)
                messages.append({'role': 'tool', 'tool_call_id': part.tool_use_id,
                                 'content': ''.join(p.text for p in part.parts if isinstance(p, Text))})
            rest = [p for p in message.parts if not isinstance(p, ToolResult)]
            if rest or not results:
                messages.append({'role': 'user', 'content': _encode_content(rest, notes=notes)})
            continue
        calls = [p for p in message.parts if isinstance(p, ToolUse)]
        body = [p for p in message.parts if not isinstance(p, ToolUse)]
        thinking = [p for p in body if isinstance(p, Thinking)] if reasoning_roundtrip else []
        if thinking:
            body = [p for p in body if not isinstance(p, Thinking)]
        out = {'role': 'assistant', 'content': _encode_content(body, notes=notes) or None}
        if thinking:
            text = ''.join(p.text for p in thinking)
            if text:
                out['reasoning_content'] = text
            else:
                _note(notes, 'dropped:thinking')  # a signature-only thinking block carries no reasoning text
        if calls:
            out['tool_calls'] = [{'id': c.id, 'type': 'function',
                                  'function': {'name': c.name, 'arguments': c.input_json or '{}'}} for c in calls]
        messages.append(out)
    return messages


def encode_tools(tools) -> list:
    return [{'type': 'function', 'function': {k: v for k, v in (('name', t.name), ('description', t.description),
                                                                 ('parameters', t.parameters)) if v is not None}}
            for t in tools]


def encode_tool_choice(choice: IRToolChoice | None):
    if choice is None:
        return None
    if choice.mode == 'tool':
        return {'type': 'function', 'function': {'name': choice.name}}
    return choice.mode


def encode_request(ir: IRRequest, profile=None, *, notes: list | None = None) -> dict:
    """An OpenAI chat-completions body for an OpenAI-compatible target (the adapter applies the profile ParamMap)."""
    roundtrip = bool(getattr(profile, 'extras', None) and profile.extras.get('reasoning_content_roundtrip'))
    body: dict = {'model': ir.model, 'messages': encode_messages(ir, notes=notes, reasoning_roundtrip=roundtrip)}
    if ir.tools:
        body['tools'] = encode_tools(ir.tools)
    choice = encode_tool_choice(ir.tool_choice)
    if choice is not None:
        body['tool_choice'] = choice
    parallel = (ir.extras_by_protocol.get('openai') or {}).get('parallel_tool_calls')
    if parallel is not None and ir.tools:
        body['parallel_tool_calls'] = parallel
    for name, value in ir.sampling.items():
        if name == 'top_k':
            _note(notes, 'dropped:top_k')
            continue
        body[name] = value
    if ir.max_output_tokens is not None:
        body['max_tokens'] = ir.max_output_tokens
    if ir.stop:
        body['stop'] = list(ir.stop)
    if ir.response_format is not None:
        kind = ir.response_format.get('type')
        features = getattr(profile, 'features', None)
        if kind == 'json_schema' and profile is not None and not getattr(profile, 'operator_surfaces', False) \
                and 'json_schema' not in (features or ()):
            raise refuse(f'json_schema output is not supported by {getattr(profile, "id", "this target")}')
        body['response_format'] = ir.response_format
    if ir.reasoning is not None and ir.reasoning.effort and (profile is None or 'reasoning' in profile.features):
        body['reasoning_effort'] = ir.reasoning.effort
    elif ir.reasoning is not None:
        _note(notes, 'dropped:reasoning')
    if ir.stream:
        body['stream'] = True
    return body


# ---------------------------------------------------------------------------------------------------------------------
# responses

def usage_dict(usage: Usage | None) -> dict:
    usage = usage or Usage()
    prompt, completion = usage.input_tokens or 0, usage.output_tokens or 0
    out = {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion}
    if usage.cached_input_tokens is not None:
        out['prompt_tokens_details'] = {'cached_tokens': usage.cached_input_tokens}
    if usage.reasoning_tokens is not None:
        out['completion_tokens_details'] = {'reasoning_tokens': usage.reasoning_tokens}
    return out


def decode_response(body: dict) -> IRResponse:
    """IRResponse from a chat.completion (or text_completion) body."""
    if not isinstance(body, dict) or not isinstance(body.get('choices'), list) or not body['choices']:
        raise bad_upstream('The upstream chat response has no choices')
    choice = body['choices'][0]
    parts = []
    if body.get('object') == 'text_completion' or 'text' in choice and 'message' not in choice:
        if isinstance(choice.get('text'), str) and choice['text']:
            parts.append(Text(choice['text']))
    else:
        message = choice.get('message') or {}
        reasoning = message.get('reasoning_content') or message.get('reasoning')
        if isinstance(reasoning, str) and reasoning:
            parts.append(Thinking(reasoning, message.get('reasoning_signature')))
        if isinstance(message.get('content'), str) and message['content']:
            parts.append(Text(message['content']))
        elif isinstance(message.get('content'), list):
            parts.extend(content_parts(message['content'], role='assistant'))
        for index, call in enumerate(message.get('tool_calls') or []):
            fn = call.get('function') or {}
            parts.append(ToolUse(id=call.get('id') or synth_tool_id(index, fn.get('name', ''), fn.get('arguments')),
                                 name=fn.get('name', ''), input_json=args_text(fn.get('arguments') or '{}')))
    extras = {k: body[k] for k in ('citations', 'search_results') if k in body}
    return IRResponse(id=str(body.get('id') or ''), model=str(body.get('model') or ''), content=parts,
                      stop=FROM_OPENAI_FINISH.get(choice.get('finish_reason'), 'end_turn'),
                      usage=openai_usage(body.get('usage')) or Usage(), extras=extras)


def encode_response(resp: IRResponse, *, model_name: str, completion: bool = False) -> dict:
    created = int(time.time())
    text = ''.join(p.text for p in resp.content if isinstance(p, Text))
    finish = OPENAI_FINISH.get(resp.stop, 'stop')
    if completion:
        return {'id': resp.id or 'cmpl-' + uuid.uuid4().hex, 'object': 'text_completion', 'created': created,
                'model': model_name, 'choices': [{'text': text, 'index': 0, 'logprobs': None, 'finish_reason': finish}],
                'usage': usage_dict(resp.usage)}
    message: dict = {'role': 'assistant', 'content': text or None}
    thinking = [p for p in resp.content if isinstance(p, Thinking)]
    if thinking:
        message['reasoning_content'] = ''.join(p.text for p in thinking)
        signature = next((p.signature for p in reversed(thinking) if p.signature), None)
        if signature:
            message['reasoning_signature'] = signature
    calls = [p for p in resp.content if isinstance(p, ToolUse)]
    if calls:
        message['tool_calls'] = [{'id': c.id, 'type': 'function', 'function': {'name': c.name,
                                                                               'arguments': c.input_json or '{}'}}
                                 for c in calls]
    message['refusal'] = None
    return {'id': resp.id or 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion', 'created': created,
            'model': model_name, 'choices': [{'index': 0, 'message': message, 'logprobs': None, 'finish_reason': finish}],
            'usage': usage_dict(resp.usage)}


# ---------------------------------------------------------------------------------------------------------------------
# streams

class _Blocks:
    """IR block bookkeeping for decoders: one open text/thinking block at a time, tool calls each in their own block."""

    def __init__(self):
        self.next = 0
        self.current: tuple[str, int] | None = None
        self.open: list[int] = []

    def enter(self, kind: str) -> tuple[list, int]:
        """(BlockStop events to emit, index) for content of kind (text | thinking | tool)."""
        if self.current is not None and self.current[0] == kind and kind != 'tool':
            return [], self.current[1]
        stops = self.close_current()
        index = self.next
        self.next += 1
        self.current = (kind, index)
        self.open.append(index)
        return stops, index

    def close_current(self) -> list:
        if self.current is None:
            return []
        index = self.current[1]
        self.current = None
        if index in self.open:
            self.open.remove(index)
            return [BlockStop(index)]
        return []

    def close_all(self) -> list:
        self.current = None
        stops = [BlockStop(i) for i in sorted(self.open)]
        self.open = []
        return stops


def _frame_json(frame):
    if isinstance(frame, SSEFrame):
        if frame.data == '[DONE]':
            return None
        return frame.json()
    return frame


@closes_source
async def decode_stream(frames, *, adapter: str = 'openai_compat'):
    """IR events from chat.completion.chunk / text_completion SSE frames (SSEFrame or already-parsed dicts)."""
    started = False
    blocks = _Blocks()
    tools: dict[int, int] = {}
    async for frame in frames:
        if isinstance(frame, SSEFrame) and frame.data == '[DONE]':
            break
        data = _frame_json(frame)
        if not isinstance(data, dict):
            raise bad_upstream('The upstream stream sent a frame that is not an object')
        if 'error' in data:
            for stop in blocks.close_all():
                yield stop
            yield stream_error_event(data, adapter=adapter)
            return
        choices = data.get('choices') or []
        if not started and (data.get('id') or choices):
            started = True
            yield Start(id=str(data.get('id') or 'chatcmpl-' + uuid.uuid4().hex), model=str(data.get('model') or ''))
        for choice in choices[:1]:
            if not isinstance(choice, dict):
                continue
            delta = choice.get('delta') if isinstance(choice.get('delta'), dict) else {}
            reasoning = delta.get('reasoning_content') or delta.get('reasoning')
            if isinstance(reasoning, str) and reasoning:
                stops, index = blocks.enter('thinking')
                for event in stops:
                    yield event
                yield ThinkingDelta(index, reasoning)
            if isinstance(delta.get('reasoning_signature'), str):
                stops, index = blocks.enter('thinking')
                for event in stops:
                    yield event
                yield ThinkingDelta(index, '', delta['reasoning_signature'])
            text = delta.get('content') if 'delta' in choice else choice.get('text')
            if isinstance(text, str) and text:
                stops, index = blocks.enter('text')
                for event in stops:
                    yield event
                yield TextDelta(index, text)
            for call in delta.get('tool_calls') or []:
                if not isinstance(call, dict):
                    continue
                position = call.get('index', 0)
                fn = call.get('function') or {}
                if position not in tools:
                    stops, index = blocks.enter('tool')
                    for event in stops:
                        yield event
                    tools[position] = index
                    yield ToolCallStart(index, str(call.get('id') or synth_tool_id(position, fn.get('name', ''), '')),
                                        str(fn.get('name') or ''))
                if isinstance(fn.get('arguments'), str) and fn['arguments']:
                    yield ToolArgsDelta(tools[position], fn['arguments'])
            if choice.get('finish_reason'):
                for event in blocks.close_all():
                    yield event
                yield Stop(FROM_OPENAI_FINISH.get(choice['finish_reason'], 'end_turn'))
        if isinstance(data.get('usage'), dict):
            yield UsageEvent(openai_usage(data['usage']))
    for event in blocks.close_all():
        yield event


def _chunk(rid, created, model, delta=None, finish=None, *, completion=False, text=None):
    if completion:
        return {'id': rid, 'object': 'text_completion', 'created': created, 'model': model,
                'choices': [{'text': text or '', 'index': 0, 'logprobs': None, 'finish_reason': finish}]}
    return {'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': model,
            'choices': [{'index': 0, 'delta': delta or {}, 'logprobs': None, 'finish_reason': finish}]}


@closes_source
async def encode_stream(events, *, model_name: str, include_usage: bool, completion: bool = False):
    """chat.completion.chunk (or text_completion) SSE bytes per IR event, ending with data: [DONE]."""
    created = int(time.time())
    rid = ('cmpl-' if completion else 'chatcmpl-') + uuid.uuid4().hex
    tail = StreamTail()
    tool_slot: dict[int, int] = {}
    tool_args: set[int] = set()  # tool blocks that streamed at least one argument fragment

    def empty_args(index):
        # a call that takes no arguments streams none (Anthropic: input {} plus input_json_delta ''); clients
        # accumulate the fragments and json.loads them, so the call gets '{}' as the non-streamed path does
        tool_args.add(index)
        return write_json_sse(_chunk(rid, created, model_name, {'tool_calls': [
            {'index': tool_slot[index], 'function': {'arguments': '{}'}}]}))

    async for event in events:
        if isinstance(event, Start):
            rid = event.id or rid
            if not completion:
                yield write_json_sse(_chunk(rid, created, model_name, {'role': 'assistant', 'content': ''}))
        elif isinstance(event, TextDelta):
            if completion:
                yield write_json_sse(_chunk(rid, created, model_name, completion=True, text=event.text))
            else:
                yield write_json_sse(_chunk(rid, created, model_name, {'content': event.text}))
        elif isinstance(event, ThinkingDelta):
            if completion:
                continue
            delta = {}
            if event.text:
                delta['reasoning_content'] = event.text
            if event.signature:
                delta['reasoning_signature'] = event.signature
            if delta:
                yield write_json_sse(_chunk(rid, created, model_name, delta))
        elif isinstance(event, ToolCallStart):
            if completion:
                raise bad_upstream('A completions stream cannot carry tool calls')
            slot = tool_slot.setdefault(event.index, len(tool_slot))
            yield write_json_sse(_chunk(rid, created, model_name, {'tool_calls': [
                {'index': slot, 'id': event.id, 'type': 'function', 'function': {'name': event.name, 'arguments': ''}}]}))
        elif isinstance(event, ToolArgsDelta):
            if not event.partial_json:
                continue
            slot = tool_slot.get(event.index, 0)
            tool_args.add(event.index)
            yield write_json_sse(_chunk(rid, created, model_name, {'tool_calls': [
                {'index': slot, 'function': {'arguments': event.partial_json}}]}))
        elif isinstance(event, BlockStop):
            if event.index in tool_slot and event.index not in tool_args:
                yield empty_args(event.index)
            continue
        elif tail.take(event) and isinstance(event, ErrorEvent):
            yield stream_terminal('openai', error_code(event.code), event.message)
            return
    for index in [i for i in tool_slot if i not in tool_args]:
        yield empty_args(index)
    finish = OPENAI_FINISH.get(tail.stop or 'end_turn', 'stop')
    yield write_json_sse(_chunk(rid, created, model_name, {}, finish, completion=completion))
    if include_usage:
        yield write_json_sse({'id': rid, 'object': 'text_completion' if completion else 'chat.completion.chunk',
                              'created': created, 'model': model_name, 'choices': [], 'usage': usage_dict(tail.usage)})
    yield b'data: [DONE]\n\n'


# ---------------------------------------------------------------------------------------------------------------------
# embeddings

def decode_embed_request(body: dict, *, notes: list | None = None) -> IRRequest:
    """IRRequest (kind embed) from a /v1/embeddings body (string or list of strings; token arrays are refused)."""
    if not isinstance(body, dict):
        raise refuse('The request body must be a JSON object', 'invalid_request')
    value = body.get('input')
    inputs = [value] if isinstance(value, str) else value
    if not isinstance(inputs, list) or not inputs or not all(isinstance(v, str) for v in inputs):
        raise refuse('Only text inputs (a string or a list of strings) can be translated for this target')
    embed = {k: body[k] for k in ('dimensions', 'encoding_format', 'input_type', 'truncate') if body.get(k) is not None}
    for name in body:
        if name not in TITANOS_ONLY and name not in ('model', 'input', 'dimensions', 'encoding_format', 'input_type',
                                                     'truncate', 'user'):
            _note(notes, f'dropped:{name}')
    return IRRequest(kind='embed', model=str(body.get('model') or ''), inputs=inputs,
                     extras_by_protocol={'embed': embed, 'notes': list(notes or [])})


def encode_embed_request(ir: IRRequest) -> dict:
    embed = ir.extras_by_protocol.get('embed') or {}
    body = {'model': ir.model, 'input': list(ir.inputs or [])}
    for name in ('dimensions', 'input_type', 'truncate'):
        if embed.get(name) is not None:
            body[name] = embed[name]
    return body


def decode_embed_response(body: dict) -> dict:
    """{'vectors': [[float]], 'usage': Usage} from an OpenAI embeddings response (float or base64 vectors)."""
    if not isinstance(body, dict) or not isinstance(body.get('data'), list):
        raise bad_upstream('The upstream embeddings response has no data')
    rows = sorted(body['data'], key=lambda row: row.get('index', 0))
    vectors = []
    for row in rows:
        value = row.get('embedding')
        if isinstance(value, str):
            raw = base64.b64decode(value)
            value = list(struct.unpack('<%df' % (len(raw) // 4), raw))
        vectors.append(value)
    usage = body.get('usage') or {}
    return {'vectors': vectors, 'usage': Usage(input_tokens=usage.get('prompt_tokens'))}


def encode_embed_response(vectors, usage: Usage | None, *, model_name: str, encoding_format: str = 'float') -> dict:
    data = []
    for index, vector in enumerate(vectors):
        value = vector
        if encoding_format == 'base64':
            value = base64.b64encode(struct.pack('<%df' % len(vector), *vector)).decode('ascii')
        data.append({'object': 'embedding', 'index': index, 'embedding': value})
    tokens = (usage.input_tokens if usage else None) or 0
    return {'object': 'list', 'data': data, 'model': model_name,
            'usage': {'prompt_tokens': tokens, 'total_tokens': tokens}}


__all__ = ['DROPPED_WITH_NOTE', 'content_parts', 'decode_completion_request', 'decode_embed_request',
           'decode_embed_response', 'decode_messages', 'decode_request', 'decode_response', 'decode_stream',
           'decode_tool_choice', 'decode_tools', 'encode_embed_request', 'encode_embed_response', 'encode_messages',
           'encode_request', 'encode_response', 'encode_stream', 'encode_tool_choice', 'encode_tools', 'image_part',
           'refuse_unsupported', 'usage_dict']
