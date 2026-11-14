"""Stateless Responses text/image transport over the existing chat engine."""
import base64
import binascii
import json
import time
import uuid

from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .engine import EngineError
from .protocols import events
from .settings import current_settings


def unsupported(message):
    raise EngineError(message, 400, 'unsupported_capability')


def chat_request(payload, spec=None):
    """Preserve omitted tuning so the shared model-default policy still applies.

    spec is the registered ModelSpec of payload.model (None when unknown): a request that needs tool calling, tool-call
    history or structured output is refused with a 400 that names the model's backend and the missing capability
    (agent_capability.refusal); those reach a model only on the llama-cpp native path, which never enters here."""
    from .agent_capability import refusal, responses_capabilities
    from .server import Chat
    missing = responses_capabilities(payload)
    if missing:
        unsupported(refusal(payload.model, spec, missing, '/v1/responses'))

    items = [{'role': 'user', 'content': payload.input}] if isinstance(payload.input, str) else payload.input
    settings = current_settings()
    if len(items) + int(payload.instructions is not None) > settings.max_messages:
        raise EngineError('Responses input exceeds the configured message count')
    messages = []
    if payload.instructions is not None:
        messages.append({'role': 'system', 'content': payload.instructions})
    for item in items:
        if not isinstance(item, dict) or item.get('type', 'message') != 'message':
            unsupported('Local Responses input accepts messages, not tool/state/reference items')
        if set(item) - {'type', 'role', 'content', 'id', 'status', 'phase'}:
            unsupported('Unsupported Responses message fields')
        if item.get('phase') is not None:
            unsupported('Local Responses does not implement assistant phase semantics')
        role = item.get('role')
        if role not in ('system', 'user', 'assistant'):
            unsupported('Local Responses accepts system, user and assistant roles')
        if 'id' in item and (not isinstance(item['id'], str) or len(item['id']) > 256):
            raise EngineError('Message id must be a string of at most 256 characters')
        if 'status' in item and item['status'] not in ('completed', 'incomplete', 'in_progress'):
            raise EngineError('Invalid replayed message status')
        content = item.get('content')
        if isinstance(content, list):
            converted = []
            for part in content:
                if not isinstance(part, dict):
                    raise EngineError('Message content requires typed objects')
                kind = part.get('type')
                if kind in ('input_text', 'output_text'):
                    allowed = {'type', 'text'}
                    if kind == 'output_text':
                        if role != 'assistant':
                            unsupported('output_text is accepted only for assistant history')
                        allowed |= {'annotations', 'logprobs'}
                        # Empty output bookkeeping can be replayed without losing
                        # model context; richer hosted annotations are not supported.
                        if part.get('annotations', []) != [] or part.get('logprobs') not in (None, []):
                            unsupported('Annotated or logprob-bearing output history is not supported')
                    if set(part) - allowed or not isinstance(part.get('text'), str):
                        raise EngineError('Text content requires a string text field')
                    converted.append({'type': 'text', 'text': part['text']})
                elif kind == 'input_image':
                    if set(part) - {'type', 'image_url', 'detail'}:
                        unsupported('Images require inline image_url data; file IDs are unsupported')
                    url = part.get('image_url')
                    if not isinstance(url, str) or not url.startswith('data:image/') or ';base64,' not in url:
                        unsupported('Images require inline base64 data URLs; remote URLs are not fetched')
                    if len(url) > 4*((settings.max_input_image_bytes+2)//3)+128:
                        raise EngineError('Image exceeds the configured byte limit')
                    try:
                        raw = base64.b64decode(url.split(';base64,', 1)[1], validate=True)
                        if not raw or len(raw) > settings.max_input_image_bytes:
                            raise ValueError('Empty image')
                    except (ValueError, binascii.Error) as exc:
                        raise EngineError('Invalid inline image encoding') from exc
                    if part.get('detail', 'auto') != 'auto':
                        unsupported('Local Responses supports image detail auto only')
                    converted.append({'type': 'image_url', 'image_url': {'url': url, 'detail': 'auto'}})
                else:
                    unsupported(f'Unsupported Responses content type: {kind}')
            content = converted
        elif not isinstance(content, str):
            raise EngineError('Message content must be text or typed content parts')
        messages.append({'role': role, 'content': content})
    values = {'model': payload.model, 'messages': messages}
    supplied = payload.model_dump(exclude_unset=True, exclude_none=True)
    for key in ('worker', 'temperature', 'top_p', 'stream'):
        if key in supplied:
            values[key] = supplied[key]
    if 'max_output_tokens' in supplied:
        values['max_tokens'] = supplied['max_output_tokens']
    if payload.stream:
        values['stream_options'] = {'include_usage': True}
    try:
        return Chat.model_validate(values)
    except ValidationError as exc:
        raise EngineError(str(exc)) from exc


def make_response(payload, chat):
    return {'id': 'resp_' + uuid.uuid4().hex, 'object': 'response', 'created_at': int(time.time()),
            'status': 'in_progress', 'error': None, 'incomplete_details': None, 'output': [],
            'model': payload.model, 'instructions': payload.instructions,
            'max_output_tokens': chat.max_completion_tokens or chat.max_tokens,
            'temperature': chat.temperature, 'top_p': chat.top_p, 'store': False,
            'tools': [], 'tool_choice': payload.tool_choice or 'auto', 'parallel_tool_calls': False,
            'text': {'format': {'type': 'text'}}, 'metadata': payload.metadata or {}, 'usage': None}


def _usage(value):
    if not isinstance(value, dict):
        return None
    incoming, outgoing = value.get('prompt_tokens'), value.get('completion_tokens')
    if any(type(n) is not int or n < 0 for n in (incoming, outgoing)):
        return None
    total = value.get('total_tokens', incoming + outgoing)
    if type(total) is not int or total != incoming + outgoing:
        raise ValueError('Backend returned inconsistent token usage')
    # This adapter reports all tokens as ordinary input/output. It has no separate
    # prompt-cache or hidden-reasoning accounting; absent overall counts stay null.
    return {'input_tokens': incoming, 'output_tokens': outgoing, 'total_tokens': total,
            'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
            'output_tokens_details': {'reasoning_tokens': 0}}


def _part(text):
    return {'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []}


def _message(ident, text, status):
    return {'id': ident, 'type': 'message', 'role': 'assistant', 'status': status, 'content': [_part(text)]}


def _finish(response, ident, text, reason, usage):
    if reason not in {'stop', 'length', 'content_filter'}:
        raise ValueError('Backend ended without a supported finish reason')
    complete = reason == 'stop'
    return {**response, 'status': 'completed' if complete else 'incomplete',
            'incomplete_details': None if complete else {'reason': 'max_output_tokens' if reason == 'length' else reason},
            'output': [_message(ident, text, 'completed' if complete else 'incomplete')], 'usage': _usage(usage)}


def adapt_response(result, response):
    """Return normal JSON or an incrementally produced, correctly terminated SSE stream."""
    ident = 'msg_' + uuid.uuid4().hex
    if not isinstance(result, StreamingResponse):
        try:
            choices = result['choices']
            if len(choices) != 1 or choices[0]['message'].get('tool_calls'):
                raise ValueError('Expected one text-only backend choice')
            text = choices[0]['message']['content']
            if not isinstance(text, str):
                raise ValueError('Backend message content must be text')
            return _finish(response, ident, text, choices[0].get('finish_reason'), result.get('usage'))
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            raise EngineError('Invalid chat backend response: ' + str(exc), 502, 'invalid_backend_response') from exc

    async def stream():
        sequence = 0
        text, reason, usage = '', None, None
        parsed = events(result)
        started = False

        def event(kind, **fields):
            nonlocal sequence
            body = {'type': kind, 'sequence_number': sequence, **fields}
            sequence += 1
            return 'event: ' + kind + '\ndata: ' + json.dumps(body, ensure_ascii=False) + '\n\n'

        indexes = {'item_id': ident, 'output_index': 0, 'content_index': 0}
        try:
            yield event('response.created', response=response)
            yield event('response.in_progress', response=response)
            item = _message(ident, '', 'in_progress')
            yield event('response.output_item.added', output_index=0, item={**item, 'content': []})
            yield event('response.content_part.added', **indexes, part=_part(''))
            try:
                started = True
                async for chunk in parsed:
                    if not isinstance(chunk, dict):
                        raise ValueError('Backend stream chunk must be an object')
                    if 'error' in chunk:
                        err = chunk['error']
                        raise ValueError(err.get('message', 'Inference failed') if isinstance(err, dict) else str(err))
                    if chunk.get('usage') is not None:
                        usage = chunk['usage']
                    choices = chunk.get('choices', [])
                    if not isinstance(choices, list) or len(choices) > 1:
                        raise ValueError('Expected one streamed backend choice')
                    for choice in choices:
                        if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                            raise ValueError('Invalid streamed choice index')
                        delta = choice.get('delta', {})
                        if not isinstance(delta, dict) or set(delta) - {'role', 'content'}:
                            raise ValueError('Backend streamed unsupported content')
                        if delta.get('role', 'assistant') != 'assistant':
                            raise ValueError('Backend streamed a non-assistant role')
                        value = delta.get('content')
                        if value is not None and not isinstance(value, str):
                            raise ValueError('Backend text delta must be a string')
                        if reason is not None:
                            raise ValueError('Backend emitted another choice after its finish event')
                        if value:
                            text += value
                            yield event('response.output_text.delta', **indexes, delta=value, logprobs=[])
                        if choice.get('finish_reason') is not None:
                            reason = choice['finish_reason']
                final = _finish(response, ident, text, reason, usage)
            except Exception as exc:
                failed = {**response, 'status': 'failed', 'output': [_message(ident, text, 'incomplete')],
                          'error': {'code': 'server_error', 'message': str(exc)}}
                yield event('response.failed', response=failed)
                return
            yield event('response.output_text.done', **indexes, text=text, logprobs=[])
            yield event('response.content_part.done', **indexes, part=_part(text))
            yield event('response.output_item.done', output_index=0, item=final['output'][0])
            yield event('response.' + final['status'], response=final)
        finally:
            # The lifecycle precedes backend iteration. A disconnect after created
            # must close the source even when the parsed generator never started.
            if started:
                await parsed.aclose()
            elif hasattr(result.body_iterator, 'aclose'):
                await result.body_iterator.aclose()

    return StreamingResponse(stream(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


NATIVE_RESPONSE_SKELETON = {'object': 'response', 'status': 'in_progress', 'error': None, 'incomplete_details': None,
                            'instructions': None, 'metadata': {}, 'output': [], 'parallel_tool_calls': True, 'temperature': None,
                            'tool_choice': 'auto', 'tools': [], 'top_p': None, 'usage': None}
# events scoped to one content part of a message item: they need output_index and content_index
PART_EVENTS = ('response.content_part.', 'response.output_text.', 'response.refusal.')


def native_stream(result, model):
    """The llama.cpp server's /v1/responses SSE completed for the stock OpenAI SDK's stream helper (client.responses.stream,
    openai/lib/streaming/responses/_responses.py accumulate_event). b10809 sends response.created and response.in_progress
    with a response of only {id, object, status} (no output list, created_at or model), no sequence_number on any event and
    no output_index / content_index on the item and part events; the accumulator appends to response.output (None -> crash)
    and indexes output[output_index].content[content_index]. Every data frame is re-emitted with those fields filled: the
    sequence number counts the frames, output_index follows the order of response.output_item.added, content_index the
    order of response.content_part.added within the item, and a response object is completed from NATIVE_RESPONSE_SKELETON
    with created_at and the requested model; whatever llama-server sent is kept as sent (setdefault). A non-streaming
    Response passes through untouched; so do the upstream's headers (X-Titanos-Ignored, Cache-Control)."""
    result = as_stream(result)
    if not isinstance(result, StreamingResponse):
        return result
    headers = {k: v for k, v in result.headers.items() if k.lower() not in ('content-type', 'content-length')}

    async def stream():
        sequence, items, parts = 0, {}, {}
        skeleton = {**NATIVE_RESPONSE_SKELETON, 'created_at': int(time.time()), 'model': model}
        parsed = events(result)
        started = False
        try:
            started = True
            async for event in parsed:
                if not isinstance(event, dict) or not isinstance(event.get('type'), str):
                    yield 'data: ' + json.dumps(event, ensure_ascii=False) + '\n\n'  # untyped: passed on, never dropped
                    continue
                kind = event['type']
                event.setdefault('sequence_number', sequence)
                sequence += 1
                if isinstance(event.get('response'), dict):
                    event['response'] = {**skeleton, **event['response']}
                item = event.get('item')
                item_id = item.get('id') if isinstance(item, dict) else event.get('item_id')
                if kind == 'response.output_item.added' and item_id is not None and item_id not in items:
                    items[item_id] = len(items)
                if item_id in items:
                    event.setdefault('output_index', items[item_id])
                    if kind == 'response.content_part.added':
                        parts[item_id] = parts.get(item_id, 0) + 1
                    if kind.startswith(PART_EVENTS):
                        event.setdefault('content_index', max(parts.get(item_id, 1) - 1, 0))
                yield 'event: ' + kind + '\ndata: ' + json.dumps(event, ensure_ascii=False) + '\n\n'
        finally:
            if started:
                await parsed.aclose()
            elif hasattr(result.body_iterator, 'aclose'):
                await result.body_iterator.aclose()

    return StreamingResponse(stream(), media_type='text/event-stream', headers=headers)


def as_stream(result):
    """A buffered text/event-stream Response (a test seam, or a proxy that read the body) as a StreamingResponse, so the SSE
    rewrites of native_stream and openai_params.legacy_logprobs see one shape; anything else is returned as it is."""
    from fastapi.responses import Response
    if (isinstance(result, Response) and not isinstance(result, StreamingResponse)
            and (result.headers.get('content-type') or '').startswith('text/event-stream')):
        headers = {k: v for k, v in result.headers.items() if k.lower() not in ('content-type', 'content-length')}
        return StreamingResponse(iter([result.body]), status_code=result.status_code, media_type='text/event-stream', headers=headers)
    return result
