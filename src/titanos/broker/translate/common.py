"""Helpers shared by the protocol codecs: capability refusals, data: URLs, deterministic tool-call ids, stop-reason and
usage maps, and the IR stream bookkeeping every encoder needs. Data only; no I/O."""
from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..errors import BrokerError
from ..sse import ManagedStream
from ..types import Usage
from .ir import ErrorEvent, IRMessage, Stop, UsageEvent

_DATA_URL = re.compile(r'^data:(?P<type>[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+)?(?P<params>(?:;[^,;]*)*?)(?P<b64>;base64)?,(?P<data>.*)$',
                       re.DOTALL)

# IR stop reason <-> OpenAI finish_reason
OPENAI_FINISH = {'end_turn': 'stop', 'stop_sequence': 'stop', 'max_tokens': 'length', 'tool_use': 'tool_calls',
                 'content_filter': 'content_filter'}
FROM_OPENAI_FINISH = {'stop': 'end_turn', 'length': 'max_tokens', 'tool_calls': 'tool_use', 'function_call': 'tool_use',
                      'content_filter': 'content_filter'}


def refuse(message: str, code: str = 'broker_capability_unavailable') -> BrokerError:
    """A 400 raised before egress (the target cannot represent the request faithfully)."""
    return BrokerError(message, code, origin='titanos')


def bad_upstream(message: str) -> BrokerError:
    return BrokerError(message, 'upstream_bad_response', origin='upstream')


def parse_data_url(url: str) -> tuple[str, bytes] | None:
    """(media type, bytes) of a base64 data: URL, None for any other URL; 400 for a malformed data: URL."""
    if not isinstance(url, str) or not url.startswith('data:'):
        return None
    match = _DATA_URL.match(url)
    if not match or not match.group('b64'):
        raise refuse('Only base64 data: URLs are accepted for inline media', 'invalid_request')
    try:
        data = base64.b64decode(match.group('data'), validate=True)
    except (binascii.Error, ValueError):
        raise refuse('The data: URL is not valid base64', 'invalid_request') from None
    return (match.group('type') or 'application/octet-stream').lower(), data


def data_url(media_type: str, data: bytes) -> str:
    return f'data:{media_type};base64,' + base64.b64encode(data).decode('ascii')


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


def unb64(text: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError, TypeError):
        raise refuse('Inline media is not valid base64', 'invalid_request') from None


def synth_tool_id(index: int, name: str, args) -> str:
    """'call_<index>_<8hex>': a deterministic id for protocols whose tool calls carry none (Gemini, Ollama)."""
    text = args if isinstance(args, str) else json.dumps(args, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    digest = hashlib.sha256(f'{index}:{name}:{text}'.encode('utf-8')).hexdigest()[:8]
    return f'call_{index}_{digest}'


def json_args(text: str):
    """Tool-call argument text parsed to an object (empty text is {}); upstream_bad_response when it is not JSON."""
    if text is None or not str(text).strip():
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        raise bad_upstream('Tool-call arguments are not valid JSON') from None
    return value


def request_args(text: str):
    """Client-sent tool-call arguments parsed to an object; 400 when they are not JSON."""
    try:
        return json_args(text)
    except BrokerError:
        raise refuse('Tool-call arguments are not valid JSON', 'invalid_request') from None


def args_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False, separators=(',', ':'))


def merge_usage(base: Usage | None, new: Usage | None) -> Usage | None:
    """new's non-None fields override base's."""
    if new is None:
        return base
    if base is None:
        return new
    names = ('input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_tokens', 'characters', 'audio_seconds',
             'images', 'cache_write_tokens', 'audio_input_tokens', 'audio_output_tokens', 'image_input_tokens',
             'cache_write_1h_tokens', 'web_search_requests', 'fast')
    values = {n: getattr(new, n) if getattr(new, n) is not None else getattr(base, n) for n in names}
    return Usage(**values, estimated=new.estimated and base.estimated)


def text_of(parts) -> str:
    """The concatenated text of Text parts (other parts ignored)."""
    from .ir import Text
    return ''.join(p.text for p in parts if isinstance(p, Text))


def merge_roles(messages: list[IRMessage]) -> list[IRMessage]:
    """Consecutive same-role messages merged (Anthropic and Gemini require alternation)."""
    out: list[IRMessage] = []
    for message in messages:
        if out and out[-1].role == message.role:
            out[-1] = IRMessage(role=message.role, parts=list(out[-1].parts) + list(message.parts))
        else:
            out.append(message)
    return out


@dataclass
class StreamTail:
    """What an encoder collects for the end of a stream: the stop reason, the merged usage and an error."""
    stop: str | None = None
    usage: Usage | None = None
    error: ErrorEvent | None = None
    notes: list = field(default_factory=list)

    def take(self, event) -> bool:
        """Absorb Stop / UsageEvent / ErrorEvent; True when the event was one of them."""
        if isinstance(event, Stop):
            self.stop = event.reason
            return True
        if isinstance(event, UsageEvent):
            self.usage = merge_usage(self.usage, event.usage)
            return True
        if isinstance(event, ErrorEvent):
            self.error = event
            return True
        return False


def error_code(code: str) -> str:
    """A broker in-stream terminal code for an ErrorEvent code."""
    return code if code in ('upstream_unavailable', 'upstream_timeout', 'upstream_rate_limited', 'upstream_bad_response',
                            'upstream_auth_failed') else 'upstream_unavailable'


def stream_error_event(payload, *, adapter: str) -> ErrorEvent:
    """ErrorEvent for an in-stream provider error (code from upstream_errors' table)."""
    from .upstream_errors import broker_code, classify_stream_error
    err = classify_stream_error(payload, adapter=adapter)
    return ErrorEvent(code=broker_code(err)[0], message=err.message)


async def aclose(iterator) -> None:
    close = getattr(iterator, 'aclose', None)
    if close is not None:
        try:
            await close()
        except Exception:  # pragma: no cover - closing a broken stream must not mask the original outcome
            pass


def closes_source(fn):
    """Decorate an async-generator codec whose first argument is its source iterator: when the codec finishes, fails or is
    closed, the source is closed too, so a whole decode/encode chain unwinds eagerly (never by the loop's finalizer). The
    wrapper returns a ManagedStream, so closing it before its first item still closes the source."""
    async def run(source, args, kwargs):
        inner = fn(source, *args, **kwargs)
        try:
            async for item in inner:
                yield item
        finally:
            await aclose(inner)
            await aclose(source)

    @functools.wraps(fn)
    def wrapper(source, *args, **kwargs):
        async def close_source():
            await aclose(source)
        return ManagedStream(run(source, args, kwargs), close_source)
    return wrapper


async def aiter_list(values) -> AsyncIterator:
    for value in values:
        yield value


__all__ = ['FROM_OPENAI_FINISH', 'OPENAI_FINISH', 'StreamTail', 'aclose', 'aiter_list', 'args_text', 'b64', 'bad_upstream',
           'closes_source', 'data_url', 'error_code', 'json_args', 'merge_roles', 'merge_usage', 'parse_data_url', 'refuse',
           'request_args', 'stream_error_event', 'synth_tool_id', 'text_of', 'unb64']
