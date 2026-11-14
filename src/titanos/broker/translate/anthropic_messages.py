"""Anthropic Messages codec.

- tool_use blocks and input_json_delta map to tool calls with their ids kept; tool_result blocks map to IR tool results.
- thinking blocks map to and from reasoning (Thinking / ThinkingDelta), with signature_delta kept; redacted_thinking cannot
  be represented in other protocols and is dropped with a note.
- tool_choice: auto / any / tool / none <-> auto / required / function / none; disable_parallel_tool_use <->
  parallel_tool_calls=false.
- image blocks with a base64 or url source are preserved.
- stop_reason and usage are mapped (input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens),
  normalized to the IR meaning where input_tokens includes the cached input (sse.anthropic_usage).
- Every encoded body is shaped by the target model's row in translate/anthropic_models.py (encode_request(model=...)):
  a requested reasoning effort becomes {type: enabled, budget_tokens} on budget-mode models (4.5 and older) and
  {type: adaptive} plus output_config.effort on 4.6 and later (effort alone where thinking is always on; minimal -> low,
  a level the model lacks -> the nearest lower one); temperature, top_p and top_k are dropped (noted) on models that
  refuse them, and top_p is dropped when a 4.5 model would get both; tool_choice any / tool degrades to auto (noted) on
  models that refuse forced tool use. Without a model (pure codec use) the 4.5-era rules apply.
- Thinking plus a tool loop: on budget-mode models Anthropic requires the assistant turn that holds the pending tool_use
  to start with a signed thinking block when thinking is enabled. Clients of other protocols never hold Anthropic
  signatures, so when that turn has no signed thinking block, thinking is not enabled for the request (note
  'dropped:thinking (unsigned tool turn)'). A thinking block with a signature is always replayed, also when its text is
  empty (the default 'omitted' display of 4.7 and later models).
- On encode, IR tool messages are merged into ONE user turn of tool_result blocks and consecutive same-role turns are
  merged. max_tokens is ALWAYS set: the client's cap, otherwise default_max_tokens (settings.broker_default_max_output_tokens;
  at least ALWAYS_THINKING_MIN_MAX_TOKENS for models whose thinking is always on), recorded as the note
  'injected:max_tokens'. json_schema maps to structured outputs (output_config.format, GA: no beta header) on every model
  that supports them; otherwise 400 broker_capability_unavailable (never emulated); json_object is refused. When Titanos
  enables thinking on a 4.6 or older model, temperature, top_p and top_k are dropped with a note.
- On decode (an Anthropic client reaching another protocol), thinking {type: enabled | adaptive} and output_config.effort
  become the IR reasoning, output_config.format (and the deprecated output_format) the response format, and document
  blocks whose source is plain text become text (noted 'converted:document').

SSE event names (event:) and delta types (delta.type) are distinct vocabularies: events message_start, content_block_start,
content_block_delta, content_block_stop, message_delta, message_stop, ping, error; delta types text_delta,
input_json_delta, thinking_delta, signature_delta.
"""
from __future__ import annotations

import json
import uuid

from ..envelopes import stream_terminal
from ..sse import SSEFrame, anthropic_usage, write_json_sse
from ..types import Usage
from . import anthropic_models as models
from .common import (StreamTail, b64, bad_upstream, closes_source, merge_roles, refuse, request_args,
                     stream_error_event, unb64)
from .ir import (BlockStop, ErrorEvent, Image, IRMessage, IRReasoning, IRRequest, IRResponse, IRTool, IRToolChoice, Start,
                 Stop, Text, TextDelta, Thinking, ThinkingDelta, ToolArgsDelta, ToolCallStart, ToolResult, ToolUse,
                 UsageEvent)

STOP_FROM_ANTHROPIC = {'end_turn': 'end_turn', 'max_tokens': 'max_tokens', 'tool_use': 'tool_use',
                       'stop_sequence': 'stop_sequence', 'refusal': 'content_filter', 'pause_turn': 'end_turn',
                       'model_context_window_exceeded': 'max_tokens'}
STOP_TO_ANTHROPIC = {'end_turn': 'end_turn', 'max_tokens': 'max_tokens', 'tool_use': 'tool_use',
                     'stop_sequence': 'stop_sequence', 'content_filter': 'refusal'}
_KNOWN = frozenset({'model', 'messages', 'system', 'max_tokens', 'stop_sequences', 'temperature', 'top_p', 'top_k',
                    'tools', 'tool_choice', 'thinking', 'stream', 'metadata', 'output_format', 'output_config',
                    'worker', 'location', 'timeout'})
EFFORT_BUDGET = models.EFFORT_BUDGET
MIN_THINKING_BUDGET = models.MIN_THINKING_BUDGET
# encode_request without a model (pure codec use, golden tests): the 4.5-era rules (budget thinking, sampling allowed)
UNSPECIFIED_RULES = models.ModelRules('unspecified', 'budget', (), 'allowed', True, True)
# the documented Gemini validator-skip signature is never an Anthropic signature
_FOREIGN_SIGNATURES = frozenset({'skip_thought_signature_validator'})


def _note(notes, text):
    if notes is not None and text not in notes:
        notes.append(text)


def _blocks(content) -> list:
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}] if content else []
    if not isinstance(content, list):
        raise refuse('content must be a string or a list of blocks', 'invalid_request')
    return content


def _image(block: dict) -> Image:
    source = block.get('source') or {}
    if source.get('type') == 'base64':
        return Image(media_type=source.get('media_type') or 'image/png', data=unb64(source.get('data') or ''))
    if source.get('type') == 'url' and isinstance(source.get('url'), str):
        return Image(media_type='image/*', url=source['url'])
    raise refuse('Only base64 and url image sources can be translated')


def _result_parts(content) -> list:
    if content is None:
        return []
    if isinstance(content, str):
        return [Text(content)] if content else []
    parts = []
    for block in content:
        if block.get('type') == 'text':
            parts.append(Text(block.get('text') or ''))
        elif block.get('type') == 'image':
            parts.append(_image(block))
        else:
            raise refuse(f'tool_result block {block.get("type")!r} cannot be translated')
    return parts


def decode_request(body: dict, *, notes: list | None = None) -> IRRequest:
    """IRRequest from a /v1/messages body."""
    if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
        raise refuse('The request body needs messages', 'invalid_request')
    system = []
    raw_system = body.get('system')
    for block in _blocks(raw_system) if raw_system is not None else []:
        if block.get('type') != 'text':
            raise refuse('system carries text blocks only')
        system.append(Text(block.get('text') or ''))
        if block.get('cache_control'):
            _note(notes, 'dropped:cache_control')
    messages = []
    for message in body['messages']:
        role = message.get('role') if isinstance(message, dict) else None
        if role not in ('user', 'assistant'):
            raise refuse('message roles are user and assistant', 'invalid_request')
        results, parts = [], []
        for block in _blocks(message.get('content')):
            kind = block.get('type')
            if block.get('cache_control'):
                _note(notes, 'dropped:cache_control')
            if kind == 'text':
                if block.get('text'):
                    parts.append(Text(block['text']))
            elif kind == 'image':
                parts.append(_image(block))
            elif kind == 'tool_use' and role == 'assistant':
                parts.append(ToolUse(id=str(block.get('id')), name=str(block.get('name')),
                                     input_json=json.dumps(block.get('input') or {}, ensure_ascii=False,
                                                           separators=(',', ':'))))
            elif kind == 'tool_result' and role == 'user':
                results.append(ToolResult(tool_use_id=str(block.get('tool_use_id')),
                                          parts=_result_parts(block.get('content')), is_error=bool(block.get('is_error'))))
            elif kind == 'thinking' and role == 'assistant':
                parts.append(Thinking(text=block.get('thinking') or '', signature=block.get('signature')))
            elif kind == 'redacted_thinking':
                _note(notes, 'dropped:redacted_thinking')
            elif kind == 'document' and _document_text(block) is not None:
                parts.append(Text(_document_text(block)))
                _note(notes, 'converted:document')
            else:
                raise refuse(f'content block {kind!r} cannot be translated for this target')
        if results:
            messages.append(IRMessage(role='tool', parts=results))
        if parts or not results:
            messages.append(IRMessage(role=role, parts=parts))
    tools = []
    for tool in body.get('tools') or []:
        if tool.get('type') not in (None, 'custom') or 'input_schema' not in tool:
            raise refuse('Only client tools with an input_schema can be translated')
        tools.append(IRTool(name=tool['name'], description=tool.get('description'), parameters=tool['input_schema']))
    choice, parallel = None, None
    raw_choice = body.get('tool_choice')
    if isinstance(raw_choice, dict):
        mode = {'auto': 'auto', 'any': 'required', 'none': 'none', 'tool': 'tool'}.get(raw_choice.get('type'))
        if mode is None:
            raise refuse('tool_choice is not supported', 'invalid_request')
        choice = IRToolChoice(mode=mode, name=raw_choice.get('name') if mode == 'tool' else None)
        if raw_choice.get('disable_parallel_tool_use'):
            parallel = False
    thinking = body.get('thinking') if isinstance(body.get('thinking'), dict) else {}
    config = body.get('output_config') if isinstance(body.get('output_config'), dict) else {}
    effort = config.get('effort') if isinstance(config.get('effort'), str) else None
    reasoning = None
    if thinking.get('type') == 'enabled':
        reasoning = IRReasoning(enabled=True, effort=effort, budget_tokens=thinking.get('budget_tokens'))
    elif thinking.get('type') == 'adaptive' or (
            effort is not None and thinking.get('type') not in ('disabled', 'between_tools')):
        reasoning = IRReasoning(enabled=True, effort=effort)
    for name in config:
        if name not in ('effort', 'format'):
            _note(notes, f'dropped:output_config.{name}')
    response_format = None
    output = config.get('format') if isinstance(config.get('format'), dict) else body.get('output_format')
    if isinstance(output, dict) and output.get('type') == 'json_schema':
        response_format = {'type': 'json_schema', 'json_schema': {'name': 'output', 'schema': output.get('schema') or {},
                                                                  'strict': True}}
    sampling = {k: body[k] for k in ('temperature', 'top_p', 'top_k') if body.get(k) is not None}
    for name in body:
        if name not in _KNOWN:
            _note(notes, f'dropped:{name}')
    if body.get('metadata'):
        _note(notes, 'dropped:metadata')
    extras = {}
    if parallel is not None:
        extras['parallel_tool_calls'] = parallel
    return IRRequest(kind='chat', model=str(body.get('model') or ''), system=system, messages=messages, tools=tools,
                     tool_choice=choice, sampling=sampling,
                     max_output_tokens=body.get('max_tokens') if isinstance(body.get('max_tokens'), int) else None,
                     stop=body.get('stop_sequences'), response_format=response_format, reasoning=reasoning,
                     stream=bool(body.get('stream')), extras_by_protocol={'openai': extras, 'notes': list(notes or [])})


def _document_text(block: dict) -> str | None:
    """The text of a document block whose source is plain text (or a content list of text blocks), else None."""
    source = block.get('source') if isinstance(block.get('source'), dict) else {}
    if source.get('type') == 'text' and isinstance(source.get('data'), str):
        return source['data']
    content = source.get('content')
    if source.get('type') == 'content' and isinstance(content, list) and content and all(
            isinstance(c, dict) and c.get('type') == 'text' for c in content):
        return '\n\n'.join(str(c.get('text') or '') for c in content)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# encode (IR -> Anthropic upstream request)

def _encode_part(part, *, notes) -> dict | None:
    if isinstance(part, Text):
        return {'type': 'text', 'text': part.text} if part.text else None
    if isinstance(part, Image):
        if part.data is not None:
            return {'type': 'image', 'source': {'type': 'base64', 'media_type': part.media_type, 'data': b64(part.data)}}
        return {'type': 'image', 'source': {'type': 'url', 'url': part.url}}
    if isinstance(part, ToolUse):
        return {'type': 'tool_use', 'id': part.id, 'name': part.name, 'input': request_args(part.input_json)}
    if isinstance(part, Thinking):
        if part.signature and part.signature not in _FOREIGN_SIGNATURES:
            # an empty text is the 'omitted' display of 4.7 and later models: the signed block must still be replayed
            return {'type': 'thinking', 'thinking': part.text or '', 'signature': part.signature}
        _note(notes, 'dropped:thinking_without_signature')
        return None
    if isinstance(part, ToolResult):
        content = [b for b in (_encode_part(p, notes=notes) for p in part.parts) if b is not None]
        block = {'type': 'tool_result', 'tool_use_id': part.tool_use_id, 'content': content}
        if part.is_error:
            block['is_error'] = True
        return block
    raise refuse(f'{type(part).__name__} content cannot be sent to Anthropic')


def encode_messages(ir: IRRequest, *, notes=None) -> list:
    turns = []
    for message in ir.messages:
        role = 'user' if message.role == 'tool' else message.role
        turns.append(IRMessage(role=role, parts=list(message.parts)))
    out = []
    for message in merge_roles(turns):
        parts = list(message.parts)
        if message.role == 'user':  # tool_result blocks must lead the user turn
            parts = [p for p in parts if isinstance(p, ToolResult)] + [p for p in parts if not isinstance(p, ToolResult)]
        blocks = [b for b in (_encode_part(p, notes=notes) for p in parts) if b is not None]
        if not blocks:
            blocks = [{'type': 'text', 'text': '.'}] if message.role == 'user' else []
            if not blocks:
                continue
        out.append({'role': message.role, 'content': blocks})
    return out


def _unsigned_tool_turn(messages: list) -> bool:
    """Whether the last assistant turn carries tool_use without a leading (signed) thinking block: Anthropic refuses such
    a request with thinking enabled (the tool loop in progress must keep the turn's thinking)."""
    for message in reversed(messages):
        if message.get('role') != 'assistant':
            continue
        blocks = message.get('content') or []
        if not any(isinstance(b, dict) and b.get('type') == 'tool_use' for b in blocks):
            return False
        first = blocks[0] if blocks and isinstance(blocks[0], dict) else {}
        return first.get('type') not in ('thinking', 'redacted_thinking')
    return False


def encode_request(ir: IRRequest, profile=None, *, default_max_tokens: int, structured_outputs: bool = False,
                   notes: list | None = None, model: str | None = None) -> dict:
    """A /v1/messages body shaped for `model` (the upstream model id; anthropic_models.rules_for). structured_outputs is
    kept for callers of the beta era: output_config.format is GA and needs no beta."""
    rules = models.rules_for(model) if model else UNSPECIFIED_RULES
    if ir.response_format is not None:
        kind = ir.response_format.get('type')
        if kind == 'json_object':
            raise refuse('json_object output is not supported by Anthropic (use json_schema)')
        if kind == 'json_schema' and not rules.structured_outputs:
            raise refuse('json_schema output (structured outputs) is not supported by this Claude model')
    body: dict = {'model': ir.model}
    if ir.max_output_tokens is not None:
        body['max_tokens'] = ir.max_output_tokens
    else:
        body['max_tokens'] = int(default_max_tokens)
        if rules.thinking == 'always':
            body['max_tokens'] = max(body['max_tokens'], models.ALWAYS_THINKING_MIN_MAX_TOKENS)
        _note(notes, 'injected:max_tokens')
    if ir.system:
        body['system'] = '\n\n'.join(p.text for p in ir.system if isinstance(p, Text))
    body['messages'] = encode_messages(ir, notes=notes)
    if ir.tools:
        body['tools'] = [{'name': t.name, **({'description': t.description} if t.description else {}),
                          'input_schema': t.parameters or {'type': 'object', 'properties': {}}} for t in ir.tools]
    parallel = (ir.extras_by_protocol.get('openai') or {}).get('parallel_tool_calls')
    forced = False
    if ir.tool_choice is not None or (parallel is False and ir.tools):
        mode = ir.tool_choice.mode if ir.tool_choice is not None else 'auto'
        choice = {'type': {'auto': 'auto', 'required': 'any', 'none': 'none', 'tool': 'tool'}[mode]}
        if mode == 'tool':
            choice['name'] = ir.tool_choice.name
        if parallel is False and mode != 'none':
            choice['disable_parallel_tool_use'] = True
        choice = models.shape_tool_choice(rules, choice, notes=notes)
        forced = choice['type'] in ('any', 'tool')
        body['tool_choice'] = choice
    plan = models.plan_thinking(rules, ir.reasoning, max_tokens=body['max_tokens'], forced_tool=forced,
                                unsigned_tool_turn=_unsigned_tool_turn(body['messages']), notes=notes)
    if plan.thinking is not None:
        body['thinking'] = plan.thinking
    body.update(models.shape_sampling(rules, ir.sampling, thinking_on=plan.thinking_on, notes=notes))
    if ir.stop:
        body['stop_sequences'] = list(ir.stop)
    config = {}
    if plan.effort is not None:
        config['effort'] = plan.effort
    if ir.response_format is not None and ir.response_format.get('type') == 'json_schema':
        schema = (ir.response_format.get('json_schema') or {}).get('schema') or {}
        config['format'] = {'type': 'json_schema', 'schema': schema}
    if config:
        body['output_config'] = config
    if ir.stream:
        body['stream'] = True
    return body

# ---------------------------------------------------------------------------------------------------------------------
# responses

def _usage_out(usage: Usage | None) -> dict:
    """Anthropic usage from an IR Usage: Anthropic's input_tokens excludes the cache reads and writes (they are
    additive), so input_tokens = IR input - cached - cache writes, cache_read_input_tokens = cached and
    cache_creation_input_tokens = cache writes (sse.anthropic_usage is the inverse)."""
    usage = usage or Usage()
    cached = usage.cached_input_tokens or 0
    written = usage.cache_write_tokens or 0
    out = {'input_tokens': max(0, (usage.input_tokens or 0) - cached - written), 'output_tokens': usage.output_tokens or 0}
    if usage.cached_input_tokens is not None:
        out['cache_read_input_tokens'] = usage.cached_input_tokens
    if usage.cache_write_tokens is not None:
        out['cache_creation_input_tokens'] = usage.cache_write_tokens
    return out


def decode_response(body: dict) -> IRResponse:
    if not isinstance(body, dict) or body.get('type') != 'message' or not isinstance(body.get('content'), list):
        raise bad_upstream('The upstream message response is malformed')
    parts = []
    for block in body['content']:
        kind = block.get('type')
        if kind == 'text':
            parts.append(Text(block.get('text') or ''))
        elif kind == 'tool_use':
            parts.append(ToolUse(id=block.get('id', ''), name=block.get('name', ''),
                                 input_json=json.dumps(block.get('input') or {}, ensure_ascii=False,
                                                       separators=(',', ':'))))
        elif kind == 'thinking':
            parts.append(Thinking(block.get('thinking') or '', block.get('signature')))
    return IRResponse(id=str(body.get('id') or ''), model=str(body.get('model') or ''), content=parts,
                      stop=STOP_FROM_ANTHROPIC.get(body.get('stop_reason'), 'end_turn'),
                      usage=anthropic_usage(body.get('usage')) or Usage())


def encode_response(resp: IRResponse, *, model_name: str) -> dict:
    content = []
    for part in resp.content:
        if isinstance(part, Text):
            content.append({'type': 'text', 'text': part.text})
        elif isinstance(part, ToolUse):
            try:
                value = json.loads(part.input_json or '{}')
            except ValueError:
                raise bad_upstream('Tool-call arguments are not valid JSON') from None
            content.append({'type': 'tool_use', 'id': part.id, 'name': part.name, 'input': value})
        elif isinstance(part, Thinking):
            content.append({'type': 'thinking', 'thinking': part.text, 'signature': part.signature or ''})
    rid = resp.id if resp.id.startswith('msg_') else 'msg_' + (resp.id or uuid.uuid4().hex)
    return {'id': rid, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': content,
            'stop_reason': STOP_TO_ANTHROPIC.get(resp.stop, 'end_turn'), 'stop_sequence': None,
            'usage': _usage_out(resp.usage)}


# ---------------------------------------------------------------------------------------------------------------------
# streams

@closes_source
async def decode_stream(frames):
    """IR events from Anthropic SSE frames."""
    usage = None
    async for frame in frames:
        data = frame.json() if isinstance(frame, SSEFrame) else frame
        if not isinstance(data, dict):
            raise bad_upstream('The upstream stream sent a frame that is not an object')
        kind = data.get('type') or (frame.event if isinstance(frame, SSEFrame) else None)
        if kind == 'message_start':
            message = data.get('message') or {}
            usage = anthropic_usage(message.get('usage'), usage)
            yield Start(id=str(message.get('id') or ''), model=str(message.get('model') or ''))
        elif kind == 'content_block_start':
            index, block = data.get('index', 0), data.get('content_block') or {}
            if block.get('type') == 'tool_use':
                yield ToolCallStart(index, str(block.get('id') or ''), str(block.get('name') or ''))
            elif block.get('type') == 'text' and block.get('text'):
                yield TextDelta(index, block['text'])
        elif kind == 'content_block_delta':
            index, delta = data.get('index', 0), data.get('delta') or {}
            dtype = delta.get('type')
            if dtype == 'text_delta':
                yield TextDelta(index, delta.get('text') or '')
            elif dtype == 'input_json_delta':
                if delta.get('partial_json'):
                    yield ToolArgsDelta(index, delta['partial_json'])
            elif dtype == 'thinking_delta':
                yield ThinkingDelta(index, delta.get('thinking') or '')
            elif dtype == 'signature_delta':
                yield ThinkingDelta(index, '', delta.get('signature'))
        elif kind == 'content_block_stop':
            yield BlockStop(data.get('index', 0))
        elif kind == 'message_delta':
            usage = anthropic_usage(data.get('usage'), usage)
            reason = (data.get('delta') or {}).get('stop_reason')
            if reason:
                yield Stop(STOP_FROM_ANTHROPIC.get(reason, 'end_turn'))
            if usage is not None:
                yield UsageEvent(usage)
        elif kind == 'message_stop':
            return
        elif kind == 'error':
            yield stream_error_event(data, adapter='anthropic')
            return
        # ping and unknown event types: nothing


@closes_source
async def encode_stream(events, *, model_name: str):
    """Anthropic SSE bytes per IR event: message_start, content_block_start/delta/stop around each text, thinking and
    tool run (sequential indices), message_delta (stop_reason and usage) and message_stop."""
    tail = StreamTail()
    mapping: dict[int, int] = {}
    kinds: dict[int, str] = {}
    open_block: int | None = None
    counter = 0
    started = False

    def start_frame(rid):
        return write_json_sse({'type': 'message_start', 'message': {
            'id': rid, 'type': 'message', 'role': 'assistant', 'model': model_name, 'content': [], 'stop_reason': None,
            'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, 'message_start')

    def stop_frame(index):
        return write_json_sse({'type': 'content_block_stop', 'index': index}, 'content_block_stop')

    async for event in events:
        if isinstance(event, Start):
            if not started:
                started = True
                rid = event.id if event.id.startswith('msg_') else 'msg_' + (event.id or uuid.uuid4().hex)
                yield start_frame(rid)
            continue
        if not started and not isinstance(event, (Stop, UsageEvent, ErrorEvent)):
            started = True
            yield start_frame('msg_' + uuid.uuid4().hex)
        if isinstance(event, (TextDelta, ThinkingDelta, ToolCallStart)):
            kind = 'text' if isinstance(event, TextDelta) else 'thinking' if isinstance(event, ThinkingDelta) else 'tool'
            if event.index not in mapping or kinds.get(event.index) != kind:
                if open_block is not None:
                    yield stop_frame(open_block)
                mapping[event.index], kinds[event.index] = counter, kind
                open_block = counter
                counter += 1
                block = ({'type': 'text', 'text': ''} if kind == 'text' else
                         {'type': 'thinking', 'thinking': ''} if kind == 'thinking' else
                         {'type': 'tool_use', 'id': event.id, 'name': event.name, 'input': {}})
                yield write_json_sse({'type': 'content_block_start', 'index': mapping[event.index],
                                      'content_block': block}, 'content_block_start')
            index = mapping[event.index]
            if isinstance(event, TextDelta) and event.text:
                yield write_json_sse({'type': 'content_block_delta', 'index': index,
                                      'delta': {'type': 'text_delta', 'text': event.text}}, 'content_block_delta')
            elif isinstance(event, ThinkingDelta):
                if event.text:
                    yield write_json_sse({'type': 'content_block_delta', 'index': index,
                                          'delta': {'type': 'thinking_delta', 'thinking': event.text}},
                                         'content_block_delta')
                if event.signature:
                    yield write_json_sse({'type': 'content_block_delta', 'index': index,
                                          'delta': {'type': 'signature_delta', 'signature': event.signature}},
                                         'content_block_delta')
        elif isinstance(event, ToolArgsDelta):
            if event.index in mapping and event.partial_json:
                yield write_json_sse({'type': 'content_block_delta', 'index': mapping[event.index],
                                      'delta': {'type': 'input_json_delta', 'partial_json': event.partial_json}},
                                     'content_block_delta')
        elif isinstance(event, BlockStop):
            if event.index in mapping and mapping[event.index] == open_block:
                yield stop_frame(open_block)
                open_block = None
        elif tail.take(event) and isinstance(event, ErrorEvent):
            yield stream_terminal('anthropic', event.code, event.message)
            return
    if not started:
        yield start_frame('msg_' + uuid.uuid4().hex)
    if open_block is not None:
        yield stop_frame(open_block)
    usage = _usage_out(tail.usage)
    yield write_json_sse({'type': 'message_delta', 'delta': {'stop_reason': STOP_TO_ANTHROPIC.get(
        tail.stop or 'end_turn', 'end_turn'), 'stop_sequence': None}, 'usage': usage}, 'message_delta')
    yield write_json_sse({'type': 'message_stop'}, 'message_stop')


__all__ = ['EFFORT_BUDGET', 'STOP_FROM_ANTHROPIC', 'STOP_TO_ANTHROPIC', 'decode_request', 'decode_response',
           'decode_stream', 'encode_messages', 'encode_request', 'encode_response', 'encode_stream']
