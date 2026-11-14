"""Inbound request handling for BrokerDispatch (helper module): the bounded body peek, body summaries, the local body
rewrites, injected output caps, and the inbound codecs that drive translated cloud calls and in-process local bridges.

- peek(): the model and stream flag from the already-buffered body (Guard bounds it): a JSON parse, or a multipart part
  parse with starlette's MultiPartParser over the buffered bytes; path parameters for Gemini actions. Nothing is read from
  the client here.
- summarize(): types.BodySummary (approximate input tokens = ceil(body bytes / 3), output cap, images, tools, audio, store,
  previous_response_id).
- rewrite_json_model / rewrite_multipart_field: the alias/virtual body rewrite for local targets (a JSON round trip; the
  multipart body keeps its boundary and every other byte).
- inject_output_cap: the injected max_tokens for budgeted keys in each protocol's own field (token-generating
  surfaces only).
- translated_call(): decode the inbound body into the IR, invoke the target adapter, and encode the answer back in the
  inbound protocol (chat, embeddings, speech and transcription), with the first IR event held before commit.
- local bridge: Gemini and ElevenLabs speech-to-text requests for local models go through the unchanged inner OpenAI route
  with an encoded OpenAI body (L* cells), and the answer is encoded back.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from .errors import BrokerError
from .limits import estimate_audio_seconds, estimate_input_tokens, measure_audio_seconds
from .sse import KEEPALIVE  # noqa: F401  (re-exported for dispatch)
from .surfaces import TOKEN_OUTPUT_SURFACES
from .translate import anthropic_messages, audio, gemini, ollama, openai_chat, responses
from .translate.common import aclose, merge_usage, refuse
from .translate.ir import IRReasoning, IRResponse, Text, UsageEvent
from .types import BodySummary, Usage

CHAT_SURFACES = frozenset({'openai.chat', 'openai.completions', 'openai.responses', 'anthropic.messages',
                           'gemini.generateContent', 'gemini.streamGenerateContent', 'ollama.chat', 'ollama.generate'})
EMBED_SURFACES = frozenset({'openai.embeddings', 'gemini.embedContent', 'gemini.batchEmbedContents', 'ollama.embed',
                            'ollama.embeddings'})
# inbound surfaces whose local route does not exist: local targets are reached through the inner OpenAI route
LOCAL_BRIDGE = {'gemini.generateContent': '/v1/chat/completions', 'gemini.streamGenerateContent': '/v1/chat/completions',
                'gemini.embedContent': '/v1/embeddings', 'gemini.batchEmbedContents': '/v1/embeddings',
                'elevenlabs.stt': '/v1/audio/transcriptions'}
OUTPUT_CAP_FIELDS = {'openai.chat': 'max_tokens', 'openai.completions': 'max_tokens',
                     'openai.responses': 'max_output_tokens', 'anthropic.messages': 'max_tokens'}
_BOUNDARY = re.compile(r'boundary="?([^";]+)"?', re.IGNORECASE)
_DISPOSITION_NAME = re.compile(rb'name="([^"]*)"', re.IGNORECASE)


@dataclass
class Peek:
    model: str | None
    stream: bool
    json: dict | None = None
    form: list | None = None
    summary: BodySummary = field(default_factory=BodySummary)
    parse_error: bool = False
    location: str | None = None


def header(headers, name: str) -> str | None:
    """A header value from ASGI header pairs or a mapping (case-insensitive)."""
    name_b = name.lower().encode('latin-1')
    items = headers.items() if hasattr(headers, 'items') else headers
    for key, value in items:
        key_b = key if isinstance(key, bytes) else str(key).encode('latin-1')
        if key_b.lower() == name_b:
            return value.decode('latin-1') if isinstance(value, bytes) else str(value)
    return None


async def parse_multipart(body: bytes, content_type: str) -> list:
    """(name, value) pairs; file values are (filename, bytes, content_type) tuples."""
    from starlette.datastructures import Headers
    from starlette.formparsers import MultiPartParser

    async def stream():
        yield body
        yield b''
    parser = MultiPartParser(Headers({'content-type': content_type}), stream())
    form = await parser.parse()
    out = []
    try:
        for name, value in form.multi_items():
            if isinstance(value, str):
                out.append((name, value))
            else:
                data = await value.read()
                out.append((name, (value.filename or 'file', data, value.content_type or 'application/octet-stream')))
    finally:
        await form.close()
    return out


def _first_int(*values):
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def output_cap(surface_name: str, body: dict | None) -> int | None:
    if not isinstance(body, dict):
        return None
    if surface_name.startswith('gemini.'):
        return gemini.output_cap(body)  # generationConfig or generation_config, either field spelling
    if surface_name.startswith('ollama.'):
        options = body.get('options') or {}
        return _first_int(options.get('num_predict')) if isinstance(options, dict) else None
    return _first_int(body.get('max_completion_tokens'), body.get('max_tokens'), body.get('max_output_tokens'))


def _walk(value, depth=0):
    if depth > 12:
        return
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item, depth + 1)


def _media(body) -> tuple[bool, bool]:
    images = audio_seen = False
    for node in _walk(body):
        if not isinstance(node, dict):
            continue
        kind = node.get('type')
        if kind in ('image_url', 'input_image', 'image') or 'image_url' in node:
            images = True
        if kind in ('input_audio', 'audio') or 'input_audio' in node:
            audio_seen = True
        # Gemini media parts in either spelling: inline data and file references (fileData file_uri). Video carries
        # pictures and sound and a PDF is read with vision, so they count as image (and video as audio too); a file
        # reference without a media type counts as both, so a text-only data policy never lets an unclassified file pass
        for key in ('inlineData', 'inline_data', 'fileData', 'file_data'):
            inline = node.get(key)
            if not isinstance(inline, dict):
                continue
            mime = str(inline.get('mimeType') or inline.get('mime_type') or '').lower()
            unknown = not mime or mime == 'application/octet-stream'
            images = images or mime.startswith(('image/', 'video/', 'application/pdf')) or unknown
            audio_seen = audio_seen or mime.startswith(('audio/', 'video/')) or unknown
        if isinstance(node.get('images'), list) and node['images']:
            images = True
    return images, audio_seen


def summarize(surface, body: dict | None, form: list | None, raw_length: int) -> BodySummary:
    has_images, has_audio = _media(body) if isinstance(body, dict) else (False, False)
    tokens = estimate_input_tokens(raw_length)
    if surface.protocol == 'anthropic' and isinstance(body, dict):
        # base64 images at a per-image allowance, documents at their decoded size, capped at the largest context window
        from .translate.anthropic_models import estimate_input_tokens as anthropic_estimate
        tokens = anthropic_estimate(body, raw_length)
    audio_seconds = audio_measured = None
    if form is not None:
        files = [v for _, v in form if isinstance(v, tuple)]
        text = sum(len(str(v)) for _, v in form if not isinstance(v, tuple))
        tokens = estimate_input_tokens(text)
        if surface.capability == 'transcription':
            has_audio = bool(files) or has_audio
            if files:
                uploads = [f[1] if len(f) > 1 else b'' for f in files]
                audio_seconds = sum(estimate_audio_seconds(data) for data in uploads)
                measured = [measure_audio_seconds(data) for data in uploads]
                if all(value is not None for value in measured):
                    audio_measured = sum(measured)
        if surface.capability == 'images':
            has_images = bool(files) or has_images
    stream = False
    if isinstance(body, dict):
        stream = bool(body.get('stream', surface.stream_default))
    tools = isinstance(body, dict) and bool(body.get('tools') or body.get('functions'))
    store = isinstance(body, dict) and body.get('store') is True
    previous = body.get('previous_response_id') if isinstance(body, dict) else None
    return BodySummary(stream=stream, approx_input_tokens=tokens, max_output_tokens=output_cap(surface.name, body),
                       has_images=has_images, has_tools=bool(tools), has_audio=has_audio, store=bool(store),
                       previous_response_id=previous if isinstance(previous, str) and previous else None,
                       bytes=raw_length, audio_seconds=audio_seconds, audio_measured=audio_measured)


def _stream_flag(surface, params: dict, body: dict | None, query: dict) -> bool:
    source = surface.stream_from
    if source == 'json:stream':
        if isinstance(body, dict) and 'stream' in body:
            return bool(body.get('stream'))
        return surface.stream_default
    if source == 'path:/stream':
        return True
    if source == 'path:streamGenerateContent':
        return True
    return False


async def peek(surface, params: dict, body: bytes, headers, query: dict) -> Peek:
    """The model and stream flag without touching the client: the buffered body only."""
    source = surface.model_from
    document = None
    form = None
    parse_error = False
    if surface.body == 'json':
        if body.strip():
            try:  # the raw bytes, as the local routes parse them (json.loads detects a BOM and UTF-16/32)
                document = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                parse_error = True
            if document is not None and not isinstance(document, dict):
                parse_error, document = True, None
        else:
            document = {}
    elif surface.body == 'multipart':
        content_type = header(headers, 'content-type') or ''
        if 'multipart/form-data' not in content_type.lower():
            parse_error = True
        else:
            try:
                form = await parse_multipart(body, content_type)
            except Exception:  # malformed multipart: the local route answers as it does today
                parse_error = True
    model = None
    if source in ('json:model', 'json:model_id') and isinstance(document, dict):
        value = document.get(source.split(':', 1)[1])
        model = value if isinstance(value, str) else (None if value is None else str(value))
    elif source in ('form:model', 'form:model_id') and form is not None:
        name = source.split(':', 1)[1]
        model = next((v for k, v in form if k == name and isinstance(v, str)), None)
    elif source == 'path:model_action':
        model = params.get('model')
        if isinstance(model, str) and model.startswith('models/'):
            model = model[len('models/'):]
    location = document.get('location') if isinstance(document, dict) and isinstance(document.get('location'), str) \
        else None
    summary = summarize(surface, document, form, len(body))
    stream = _stream_flag(surface, params, document, query)
    summary = replace(summary, stream=stream)
    return Peek(model=model, stream=stream, json=document, form=form, summary=summary, parse_error=parse_error,
                location=location)


def advisor_input(surface, document: dict | None) -> dict:
    """{message_count, excerpt}: the last user text of the request (the router scrubs and bounds it)."""
    if not isinstance(document, dict):
        return {'message_count': 0, 'excerpt': None}
    messages = document.get('messages') or document.get('contents') or document.get('input')
    if isinstance(messages, str):
        return {'message_count': 1, 'excerpt': messages}
    if not isinstance(messages, list):
        prompt = document.get('prompt')
        return {'message_count': 1 if prompt else 0, 'excerpt': prompt if isinstance(prompt, str) else None}
    text = None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get('role') not in ('user', None):
            continue
        content = message.get('content', message.get('parts'))
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = ' '.join(str(p.get('text')) for p in content if isinstance(p, dict) and p.get('text'))
        if text:
            break
    return {'message_count': len(messages), 'excerpt': text}


# ---------------------------------------------------------------------------------------------------------------------
# body rewrites

def rewrite_json_model(document: dict, field_name: str, value: str) -> bytes:
    out = dict(document)
    out[field_name] = value
    return json.dumps(out, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def multipart_boundary(content_type: str) -> bytes | None:
    found = _BOUNDARY.search(content_type or '')
    return found.group(1).encode('latin-1') if found else None


def rewrite_multipart_field(body: bytes, boundary: bytes, name: str, value: str) -> bytes:
    """The multipart body with the value of the (non-file) field `name` replaced; the boundary and every other byte are
    kept. Unchanged when the field is absent."""
    delimiter = b'--' + boundary
    pieces = body.split(delimiter)
    out = [pieces[0]]
    for piece in pieces[1:]:
        head, sep, rest = piece.partition(b'\r\n\r\n')
        if sep:
            found = _DISPOSITION_NAME.search(head)
            if found and found.group(1).decode('latin-1') == name and b'filename=' not in head.lower():
                # the content ends right before the CRLF that precedes the next delimiter
                trailing = b'\r\n' if rest.endswith(b'\r\n') else b''
                piece = head + sep + value.encode('utf-8') + trailing
        out.append(piece)
    return delimiter.join(out)


def set_content_length(headers: list, length: int) -> list:
    out = [(k, v) for k, v in headers if k.lower() != b'content-length']
    out.append((b'content-length', str(length).encode('latin-1')))
    return out


def inject_output_cap(surface_name: str, document: dict, value: int) -> dict:
    """The body with the output cap set in the protocol's own field (only when the request had none). Only a
    token-generating surface (surfaces.TOKEN_OUTPUT_SURFACES) has such a field: any other surface (images, audio,
    embeddings, moderations, token counters) gets its body back unchanged, never a guessed max_tokens."""
    out = dict(document)
    if surface_name not in TOKEN_OUTPUT_SURFACES:
        return out
    if surface_name.startswith('gemini.'):
        out = gemini.with_output_cap(out, value)  # inside the config object the body already uses
    elif surface_name.startswith('ollama.'):
        options = dict(out.get('options') or {})
        options.setdefault('num_predict', value)
        out['options'] = options
    else:
        name = OUTPUT_CAP_FIELDS[surface_name]
        if output_cap(surface_name, out) is None:
            out[name] = value
    return out


# ---------------------------------------------------------------------------------------------------------------------
# translated calls

@dataclass
class Produced:
    """A ready answer: commit sends the start with content_type and streams body."""
    status: int
    content_type: str
    body: AsyncIterator[bytes]
    stream: bool
    usage: Callable[[], Any]  # returns Usage | None, or an awaitable of it
    aclose: Callable[[], Any]
    upstream_request_id: str | None = None
    headers: dict = field(default_factory=dict)
    sse_protocol: str | None = None  # openai | anthropic | responses | gemini | ollama: keepalive and terminal frames


async def _once(data: bytes):
    yield data


async def _noop():
    return None


def decode_chat(surface_name: str, document: dict, *, requested: str, stream: bool, notes: list):
    if surface_name == 'openai.chat':
        return openai_chat.decode_request(document, notes=notes)
    if surface_name == 'openai.completions':
        return openai_chat.decode_completion_request(document, notes=notes)
    if surface_name == 'openai.responses':
        return responses.decode_request(document, notes=notes)
    if surface_name == 'anthropic.messages':
        return anthropic_messages.decode_request(document, notes=notes)
    if surface_name in ('gemini.generateContent', 'gemini.streamGenerateContent'):
        return gemini.decode_request(document, model='models/' + requested, stream=stream, notes=notes)
    if surface_name in ('ollama.chat', 'ollama.generate'):
        return ollama.decode_request(document, generate=surface_name == 'ollama.generate', notes=notes)
    raise refuse(f'{surface_name} cannot be translated')


def _include_usage(document: dict) -> bool:
    options = document.get('stream_options') if isinstance(document, dict) else None
    return bool(isinstance(options, dict) and options.get('include_usage'))


def encode_chat_stream(surface_name: str, events, *, requested: str, document: dict):
    """(encoded byte stream, content type, sse protocol)."""
    if surface_name == 'openai.chat':
        return (openai_chat.encode_stream(events, model_name=requested, include_usage=_include_usage(document)),
                'text/event-stream', 'openai')
    if surface_name == 'openai.completions':
        return (openai_chat.encode_stream(events, model_name=requested, include_usage=_include_usage(document),
                                          completion=True), 'text/event-stream', 'openai')
    if surface_name == 'openai.responses':
        return (responses.encode_stream(events, responses.response_object(model=requested, request=document)),
                'text/event-stream', 'responses')
    if surface_name == 'anthropic.messages':
        return anthropic_messages.encode_stream(events, model_name=requested), 'text/event-stream', 'anthropic'
    if surface_name.startswith('gemini.'):
        return gemini.encode_stream(events, model_name=requested), 'text/event-stream', 'gemini'
    if surface_name in ('ollama.chat', 'ollama.generate'):
        return (ollama.encode_ndjson_stream(events, model_name=requested, generate=surface_name == 'ollama.generate'),
                'application/x-ndjson', 'ollama')
    raise refuse(f'{surface_name} cannot be translated')


def encode_chat_response(surface_name: str, resp: IRResponse, *, requested: str, document: dict) -> dict:
    if surface_name == 'openai.chat':
        return openai_chat.encode_response(resp, model_name=requested)
    if surface_name == 'openai.completions':
        return openai_chat.encode_response(resp, model_name=requested, completion=True)
    if surface_name == 'openai.responses':
        return responses.encode_response(resp, responses.response_object(model=requested, request=document))
    if surface_name == 'anthropic.messages':
        return anthropic_messages.encode_response(resp, model_name=requested)
    if surface_name.startswith('gemini.'):
        return gemini.encode_response(resp, model_name=requested)
    if surface_name in ('ollama.chat', 'ollama.generate'):
        return ollama.encode_response(resp, model_name=requested, generate=surface_name == 'ollama.generate')
    raise refuse(f'{surface_name} cannot be translated')


def decode_embed(surface_name: str, document: dict, *, requested: str, notes: list):
    if surface_name == 'openai.embeddings':
        return openai_chat.decode_embed_request(document, notes=notes)
    if surface_name in ('gemini.embedContent', 'gemini.batchEmbedContents'):
        return gemini.decode_embed_request(document, batch=surface_name == 'gemini.batchEmbedContents',
                                           model='models/' + requested, notes=notes)
    if surface_name in ('ollama.embed', 'ollama.embeddings'):
        return ollama.decode_embed_request(document, notes=notes)
    raise refuse(f'{surface_name} cannot be translated')


def encode_embed(surface_name: str, vectors, usage, *, requested: str, document: dict) -> dict:
    if surface_name == 'openai.embeddings':
        return openai_chat.encode_embed_response(vectors, usage, model_name=requested,
                                                 encoding_format=document.get('encoding_format') or 'float')
    if surface_name in ('gemini.embedContent', 'gemini.batchEmbedContents'):
        return gemini.encode_embed_response(vectors, batch=surface_name == 'gemini.batchEmbedContents')
    if surface_name in ('ollama.embed', 'ollama.embeddings'):
        return ollama.encode_embed_response(vectors, usage, model_name=requested, legacy=surface_name == 'ollama.embeddings')
    raise refuse(f'{surface_name} cannot be translated')


def attach_media(surface_name: str, resp: IRResponse, encoded: dict, notes: list) -> dict:
    """Generated images and audio of a translated answer (a Gemini image or audio model) in the inbound protocol:
    openai.chat gets message.images ([{type: image_url, image_url: {url: data URL}}], the convention OpenRouter and
    other OpenAI-compatible gateways use), openai.responses an image_generation_call output item per image, Gemini
    inbound keeps inlineData parts (encode_response); every other protocol has no field for them, so they are noted
    (dropped:image_output, dropped:audio_output) instead of disappearing silently."""
    from .translate.common import b64, data_url
    from .translate.ir import Audio, Image
    images = [p for p in resp.content if isinstance(p, Image) and p.data is not None]
    sounds = [p for p in resp.content if isinstance(p, Audio)]
    if not images and not sounds or surface_name.startswith('gemini.'):
        return encoded
    if images and surface_name == 'openai.chat' and isinstance(encoded.get('choices'), list) and encoded['choices']:
        message = encoded['choices'][0].get('message')
        if isinstance(message, dict):
            message['images'] = [{'type': 'image_url', 'image_url': {'url': data_url(p.media_type, p.data)}}
                                 for p in images]
            images = []
    elif images and surface_name == 'openai.responses' and isinstance(encoded.get('output'), list):
        for index, part in enumerate(images):
            encoded['output'].append({'id': f'ig_{index}_{(encoded.get("id") or "resp")[-12:]}',
                                      'type': 'image_generation_call', 'status': 'completed',
                                      'result': b64(part.data), 'output_format': part.media_type.split('/', 1)[-1]})
        images = []
    if images and 'dropped:image_output' not in notes:
        notes.append('dropped:image_output')
    if sounds and 'dropped:audio_output' not in notes:
        notes.append('dropped:audio_output')
    return encoded


class UsageTap:
    """Accumulates UsageEvent values flowing through an IR event stream."""

    def __init__(self):
        self.usage: Usage | None = None

    def wrap(self, events):
        from .sse import ManagedStream

        async def run():
            try:
                async for event in events:
                    if isinstance(event, UsageEvent):
                        self.usage = merge_usage(self.usage, event.usage)
                    yield event
            finally:
                await aclose(events)

        async def close():
            await aclose(events)
        return ManagedStream(run(), close)


def json_bytes(document) -> bytes:
    return json.dumps(document, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def _with_output_cap(surface_name: str, ir, injected: int | None):
    """The IR with the injected output cap, on token-generating surfaces only (the passthrough rule)."""
    if (injected is not None and surface_name in TOKEN_OUTPUT_SURFACES
            and getattr(ir, 'max_output_tokens', None) is None):
        return replace(ir, max_output_tokens=injected)
    return ir


def _with_reasoning(ir, mode: str | None, profile, notes: list):
    """R19: the advisor's reasoning_mode as the IR reasoning when the request set none and the target supports it
    ('thinking' profiles get a budget: Anthropic thinking, Gemini thinkingBudget; 'reasoning' profiles an effort)."""
    if mode is None or ir.reasoning is not None or profile is None:
        return ir
    features = getattr(profile, 'features', frozenset())
    if 'thinking' in features:
        reasoning = IRReasoning(enabled=True, effort=mode, budget_tokens=anthropic_messages.EFFORT_BUDGET.get(mode, 4096))
    elif 'reasoning' in features:
        reasoning = IRReasoning(enabled=True, effort=mode)
    else:
        return ir
    notes.append('injected:reasoning')
    return replace(ir, reasoning=reasoning)


async def translated_call(adapter, grant, egress, surface, peek_: Peek, *, requested: str, query: dict, params: dict,
                          notes: list, inject: dict | None, voice_maps, injected_max_tokens: int | None,
                          reasoning_mode: str | None = None) -> Produced:
    """Decode the inbound request, invoke the adapter and encode the answer in the inbound protocol. Raises
    UpstreamFailure / BrokerError before anything is committed (the first IR event of a stream is held).
    reasoning_mode: the advisor's mode (R19), applied when the request sets no reasoning and the target supports it."""
    name = surface.name
    document = peek_.json if isinstance(peek_.json, dict) else {}
    media = getattr(adapter, 'media_call', None)
    if media is not None and name in getattr(adapter, 'media_surfaces', lambda: frozenset())():
        # a provider-owned typed bridge (Gemini speech, images and transcription through generateContent)
        return await media(grant, egress, name, document, list(peek_.form or []) if peek_.form is not None else None,
                           notes=notes)
    if name in CHAT_SURFACES:
        if name == 'gemini.streamGenerateContent' and (query or {}).get('alt') != 'sse':
            raise refuse('A translated Gemini stream needs alt=sse')
        ir = decode_chat(name, document, requested=requested, stream=peek_.stream, notes=notes)
        ir = _with_reasoning(ir, reasoning_mode, getattr(adapter, 'profile', None), notes)
        ir = replace(_with_output_cap(name, ir, injected_max_tokens), stream=peek_.stream)
        outcome = await adapter.invoke(grant, egress, ir, notes=notes, inject=inject)
        if peek_.stream:
            events = await adapter.peek_events(outcome)
            tap = UsageTap()
            body, content_type, protocol = encode_chat_stream(name, tap.wrap(events), requested=requested,
                                                              document=document)
            return Produced(200, content_type, body, True, lambda: tap.usage, lambda: aclose(body),
                            sse_protocol=protocol)
        encoded = attach_media(name, outcome, encode_chat_response(name, outcome, requested=requested,
                                                                   document=document), notes)
        data = json_bytes(encoded)
        return Produced(200, 'application/json', _once(data), False, lambda: outcome.usage, _noop)
    if name in EMBED_SURFACES:
        ir = decode_embed(name, document, requested=requested, notes=notes)
        resp = await adapter.invoke(grant, egress, ir, notes=notes, inject=inject)
        vectors = (resp.extras or {}).get('vectors') or []
        data = json_bytes(encode_embed(name, vectors, resp.usage, requested=requested, document=document))
        return Produced(200, 'application/json', _once(data), False, lambda: resp.usage, _noop)
    if name == 'openai.audio.speech':
        voice, body, xi_query, wrap = audio.speech_openai_to_elevenlabs(document, voice_maps, notes=notes)
        result = await adapter.tts(grant, egress, voice, body, xi_query, stream=True)
        stream = audio.wrap_wav_stream(result.body, audio.pcm_rate(xi_query.get('output_format'))) if wrap \
            else result.body
        return Produced(result.status, 'audio/wav' if wrap else result.content_type, stream, True,
                        lambda: result.usage, result.aclose, result.upstream_request_id, sse_protocol=None)
    if name in ('elevenlabs.tts', 'elevenlabs.tts_stream'):
        body = audio.speech_elevenlabs_to_openai(document, query or {}, voice_maps, voice_id=params.get('voice_id', ''),
                                                 notes=notes)
        result = await adapter.speech(grant, egress, body, stream=True)
        return Produced(result.status, result.content_type, result.body, True, lambda: result.usage, result.aclose,
                        result.upstream_request_id)
    if name in ('openai.audio.transcriptions', 'openai.audio.translations'):
        if name == 'openai.audio.translations':
            raise refuse('Audio translation is not available from this target')
        form = list(peek_.form or [])
        fmt = next((v for k, v in form if k == 'response_format'), None) or 'json'
        answer = await adapter.stt(grant, egress, audio.stt_openai_to_elevenlabs(form, notes=notes))
        out = audio.stt_elevenlabs_to_openai(answer, fmt)
        usage = adapter.usage_from_body(_surface('elevenlabs.stt'), answer, {})
        if isinstance(out, str):
            content_type = 'text/plain; charset=utf-8' if fmt == 'text' else ('text/vtt' if fmt == 'vtt'
                                                                               else 'application/x-subrip')
            return Produced(200, content_type, _once(out.encode('utf-8')), False, lambda: usage, _noop)
        return Produced(200, 'application/json', _once(json_bytes(out)), False, lambda: usage, _noop)
    if name == 'elevenlabs.stt':
        fields, _fmt = audio.stt_elevenlabs_form_to_openai(list(peek_.form or []), notes=notes)
        answer = await adapter.transcribe(grant, egress, fields)
        out = audio.stt_openai_to_elevenlabs_body(answer)
        usage = adapter.usage_from_body(_surface('openai.audio.transcriptions'), answer, {}) if isinstance(answer, dict) \
            else None
        return Produced(200, 'application/json', _once(json_bytes(out)), False, lambda: usage, _noop)
    raise refuse(f'{name} cannot be translated for this target')


def _surface(name: str):
    from .surfaces import SURFACES
    return SURFACES[name]


# ---------------------------------------------------------------------------------------------------------------------
# local bridge (L*): inbound protocols without a local route reach local models through the inner OpenAI route

def bridge_request(surface, peek_: Peek, *, local_name: str, notes: list) -> tuple[str, bytes, str]:
    """(path, body, content type) of the inner OpenAI request for a bridged local target."""
    name = surface.name
    document = peek_.json if isinstance(peek_.json, dict) else {}
    if name in ('gemini.generateContent', 'gemini.streamGenerateContent'):
        ir = decode_chat(name, document, requested=local_name, stream=peek_.stream, notes=notes)
        body = openai_chat.encode_request(replace(ir, stream=peek_.stream), None, notes=notes)
        body['model'] = local_name
        if peek_.stream:
            body['stream'] = True
            body['stream_options'] = {'include_usage': True}
        else:
            body.pop('stream', None)
        return LOCAL_BRIDGE[name], json_bytes(body), 'application/json'
    if name in ('gemini.embedContent', 'gemini.batchEmbedContents'):
        ir = decode_embed(name, document, requested=local_name, notes=notes)
        body = openai_chat.encode_embed_request(ir)
        body['model'] = local_name
        return LOCAL_BRIDGE[name], json_bytes(body), 'application/json'
    if name == 'elevenlabs.stt':
        fields, _ = audio.stt_elevenlabs_form_to_openai(list(peek_.form or []), notes=notes)
        fields = [(k, local_name if k == 'model' else v) for k, v in fields]
        boundary = 'titanos-bridge-boundary'
        chunks = []
        for key, value in fields:
            if isinstance(value, tuple):
                filename, data, content_type = value
                chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; '
                              f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode('utf-8') + data
                              + b'\r\n')
            else:
                chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
                              .encode('utf-8'))
        chunks.append(f'--{boundary}--\r\n'.encode('utf-8'))
        return LOCAL_BRIDGE[name], b''.join(chunks), f'multipart/form-data; boundary={boundary}'
    raise refuse(f'{name} has no local bridge')


def bridge_response(surface, status: int, body: bytes, *, requested: str, peek_: Peek) -> tuple[int, str, bytes]:
    """(status, content type, body) in the inbound protocol from a complete inner OpenAI answer. Errors pass through."""
    if status >= 400:
        return status, 'application/json', body
    name = surface.name
    try:
        document = json.loads(body.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        raise BrokerError('The local route answered with a body that is not JSON', 'upstream_bad_response',
                          origin='local') from None
    if name in ('gemini.generateContent', 'gemini.streamGenerateContent'):
        resp = openai_chat.decode_response(document)
        return 200, 'application/json', json_bytes(gemini.encode_response(resp, model_name=requested))
    if name in ('gemini.embedContent', 'gemini.batchEmbedContents'):
        decoded = openai_chat.decode_embed_response(document)
        return 200, 'application/json', json_bytes(gemini.encode_embed_response(
            decoded['vectors'], batch=name == 'gemini.batchEmbedContents'))
    if name == 'elevenlabs.stt':
        return 200, 'application/json', json_bytes(audio.stt_openai_to_elevenlabs_body(document))
    return status, 'application/json', body


def bridge_stream(surface, chunks, *, requested: str):
    """The inbound-protocol stream (Gemini alt=sse) from the inner OpenAI chat stream bytes."""
    from .sse import parse_sse
    events = openai_chat.decode_stream(parse_sse(chunks), adapter='openai_compat')
    return gemini.encode_stream(events, model_name=requested)


def ir_text(resp: IRResponse) -> str:
    return ''.join(p.text for p in resp.content if isinstance(p, Text))


__all__ = ['CHAT_SURFACES', 'EMBED_SURFACES', 'LOCAL_BRIDGE', 'OUTPUT_CAP_FIELDS', 'Peek', 'Produced', 'UsageTap',
           'advisor_input', 'attach_media', 'bridge_request', 'bridge_response', 'bridge_stream', 'decode_chat', 'decode_embed',
           'encode_chat_response', 'encode_chat_stream', 'encode_embed', 'header', 'inject_output_cap', 'ir_text',
           'json_bytes', 'multipart_boundary', 'output_cap', 'parse_multipart', 'peek', 'rewrite_json_model',
           'rewrite_multipart_field', 'set_content_length', 'summarize', 'translated_call']
