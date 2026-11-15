"""Protocol-native error envelopes and in-stream terminal frames for broker-originated errors.

- OpenAI {error:{message,type,param,code}} (the shape of server.error)
- Anthropic {type:'error', error:{type,message}}
- Gemini {error:{code,message,status}}
- ElevenLabs {detail:{status,message}}
- Ollama {error:'...'}
Every broker error carries X-Titanos-Error-Origin; Retry-After when known; x-should-retry: false for no_retry codes.
Local-target errors are never re-rendered here (byte identity). PROTOCOL_BY_PREFIX also sets scope['titanos.protocol'],
so a later release can move local /v1/messages and /v1/text-to-speech errors to native envelopes on a schedule.
"""
from __future__ import annotations

import json

from .errors import ERROR_CODES, anthropic_type, error_headers, gemini_status
from .redact import scrub

PROTOCOLS = ('openai', 'anthropic', 'gemini', 'elevenlabs', 'ollama')
# Segment-boundary prefixes, longest first; anything else is OpenAI-shaped (the Titanos native default).
PROTOCOL_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ('/v1/text-to-speech', 'elevenlabs'),
    ('/v1/speech-to-text', 'elevenlabs'),
    ('/api/embeddings', 'ollama'),
    ('/api/generate', 'ollama'),
    ('/v1/messages', 'anthropic'),
    ('/v1/voices', 'elevenlabs'),
    ('/api/embed', 'ollama'),
    ('/api/chat', 'ollama'),
    ('/api/tags', 'ollama'),
    ('/v1beta', 'gemini'),
    ('/api/ps', 'ollama'),
)


def protocol_for(path: str) -> str:
    for prefix, protocol in PROTOCOL_BY_PREFIX:
        if path == prefix or path.startswith(prefix + '/'):
            return protocol
    return 'openai'


def body(protocol: str, status: int, code: str, message: str) -> dict:
    """The protocol-native error document. The message is scrubbed here too (defence in depth: every error body and
    stream frame passes this choke point; an already clean message is unchanged)."""
    message = scrub(message)
    if protocol == 'openai':
        return {'error': {'message': message, 'type': code, 'param': None, 'code': code}}
    if protocol == 'anthropic':
        row = ERROR_CODES.get(code)
        kind = row.anthropic_type if row is not None and row.status == status else anthropic_type(status)
        return {'type': 'error', 'error': {'type': kind, 'message': message}}
    if protocol == 'gemini':
        row = ERROR_CODES.get(code)
        value = row.gemini_status if row is not None and row.status == status else gemini_status(status)
        return {'error': {'code': status, 'message': message, 'status': value}}
    if protocol == 'elevenlabs':
        return {'detail': {'status': code, 'message': message}}
    if protocol == 'ollama':
        return {'error': message}
    raise ValueError(f'Unknown protocol: {protocol!r}')


def render(protocol: str, status: int, code: str, message: str, *, retry_after: float | None = None,
           origin: str = 'titanos', extra_headers: dict | None = None):
    """A starlette JSONResponse in the inbound protocol's envelope with the broker error headers."""
    from starlette.responses import JSONResponse
    return JSONResponse(body(protocol, status, code, message), status_code=status,
                        headers=error_headers(code, origin=origin, retry_after=retry_after, extra=extra_headers))


def render_error(protocol: str, exc):
    """render() for a BrokerError."""
    return render(protocol, exc.status, exc.code, str(exc), retry_after=exc.retry_after, origin=exc.origin,
                  extra_headers=exc.headers)


def _sse(data: dict, event: str | None = None) -> bytes:
    text = json.dumps(data, ensure_ascii=False)
    return ((f'event: {event}\n' if event else '') + f'data: {text}\n\n').encode('utf-8')


def stream_terminal(protocol: str, code: str, message: str, *, sequence_number: int = 0,
                    response: dict | None = None) -> bytes:
    """The in-protocol terminal frame after commit (code: upstream_unavailable | upstream_timeout | ...).

    - openai: data: {"error":{"message","type":"stream_interrupted","code":code}} then data: [DONE]
    - responses: event response.failed (the Responses SSE family; pass the running sequence_number and response object)
    - anthropic: event: error {type:'error', error:{type:'overloaded_error'|'timeout_error'|'api_error', message}}
      (timeout_error, Anthropic's own 504 type, for deadline_exceeded, so a client can tell an expired deadline apart)
    - gemini: a final {error:{code,message,status}} chunk
    - ollama: an {"error": "..."} NDJSON line
    - elevenlabs (binary audio): b'' - the caller aborts the connection without a terminating chunk."""
    status = ERROR_CODES[code].status if code in ERROR_CODES else 502
    message = scrub(message)  # defence in depth, as in body()
    if protocol == 'openai':
        return (_sse({'error': {'message': message, 'type': 'stream_interrupted', 'code': code}})
                + b'data: [DONE]\n\n')
    if protocol == 'responses':
        failed = dict(response or {})
        failed.update({'status': 'failed', 'error': {'code': 'server_error', 'message': message}})
        return _sse({'type': 'response.failed', 'sequence_number': sequence_number, 'response': failed},
                    event='response.failed')
    if protocol == 'anthropic':
        kind = {'upstream_unavailable': 'overloaded_error', 'deadline_exceeded': 'timeout_error'}.get(code, 'api_error')
        return _sse({'type': 'error', 'error': {'type': kind, 'message': message}}, event='error')
    if protocol == 'gemini':
        return _sse({'error': {'code': status, 'message': message, 'status': gemini_status(status)}})
    if protocol == 'ollama':
        return (json.dumps({'error': message}, ensure_ascii=False) + '\n').encode('utf-8')
    if protocol == 'elevenlabs':
        return b''
    raise ValueError(f'Unknown protocol: {protocol!r}')


async def asgi_error(scope, send, protocol: str, status: int, code: str, message: str, **kw) -> None:
    """Send a complete error response over ASGI (for middleware outside FastAPI's exception handling)."""
    response = render(protocol, status, code, message, **kw)

    async def receive():  # pragma: no cover - JSONResponse never reads the request
        return {'type': 'http.disconnect'}
    await response(scope, receive, send)


__all__ = ['PROTOCOLS', 'PROTOCOL_BY_PREFIX', 'asgi_error', 'body', 'protocol_for', 'render', 'render_error',
           'stream_terminal']
