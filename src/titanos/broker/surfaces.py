"""Declarative table of every broker-visible inference, stage and catalog surface.

BrokerDispatch matches requests against it; tools/build_route_contract_matrix.py and tools/build_broker_matrix.py read it as
a fixed point. Path templates use the FastAPI spelling: '{name}' is one path segment, '{name:path}' may span segments, and a
literal suffix after a ':path' parameter (the Gemini ':generateContent' actions) splits on the LAST ':'.

Fields:
- protocol: the inbound envelope family (openai, anthropic, gemini, elevenlabs, ollama).
- capability: what a target must support (chat, embeddings, speech, transcription, images, moderation; 'asset' for jobs).
- body: json | multipart | none. model_from: where the model name is read. model_absent: 'refuse' (400 model_required) or
  'local_passthrough' (the unchanged local route answers; elevenlabs.tts without model_id keeps today's single-speech default).
- stream_from: where streaming is selected; stream_default: the protocol's default when the field is absent (Ollama streams
  by default).
- replay_safe: a failed attempt that already sent request bytes may be retried on the next target.
- scope: the route scope (scopes.ROUTE_SCOPES agrees). handle_bound: answered only by the upstream owning a handle.
- kind: intercept (BrokerDispatch answers before FastAPI parses), stage (resolved in-process by StageRouter/CloudInvoker),
  catalog (merged inside its handler).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PROTOCOLS = ('openai', 'anthropic', 'gemini', 'elevenlabs', 'ollama')
MODEL_FROM = frozenset({'json:model', 'json:model_id', 'form:model', 'form:model_id', 'path:model', 'path:model_action',
                        'none'})
STREAM_FROM = frozenset({'json:stream', 'path:/stream', 'path:streamGenerateContent', 'none'})
# The token-generating surfaces: the only ones with an output cap (max_tokens, max_output_tokens, maxOutputTokens,
# num_predict). A budgeted key's default cap is injected here and nowhere else: images, audio speech, transcription and
# translation, embeddings, moderations and the token counters (count_tokens, countTokens) never get one.
TOKEN_OUTPUT_SURFACES = frozenset({'openai.chat', 'openai.completions', 'openai.responses', 'anthropic.messages',
                                   'gemini.generateContent', 'gemini.streamGenerateContent', 'ollama.chat',
                                   'ollama.generate'})


@dataclass(frozen=True)
class Surface:
    name: str
    method: str
    path: str
    protocol: str
    capability: str
    body: Literal['json', 'multipart', 'none']
    model_from: str
    model_absent: Literal['refuse', 'local_passthrough']
    stream_from: str
    replay_safe: bool
    scope: str
    handle_bound: bool
    kind: Literal['intercept', 'stage', 'catalog']
    stream_default: bool = False


def _s(name, method, path, protocol, capability, body, model_from, scope, *, kind='intercept', stream_from='none',
       replay_safe=False, model_absent='refuse', handle_bound=False, stream_default=False):
    return Surface(name, method, path, protocol, capability, body, model_from, model_absent, stream_from, replay_safe,
                   scope, handle_bound, kind, stream_default)


_ROWS = (
    # OpenAI-compatible
    _s('openai.chat', 'POST', '/v1/chat/completions', 'openai', 'chat', 'json', 'json:model', 'inference:chat',
       stream_from='json:stream', replay_safe=True),
    _s('openai.completions', 'POST', '/v1/completions', 'openai', 'chat', 'json', 'json:model', 'inference:chat',
       stream_from='json:stream', replay_safe=True),
    _s('openai.responses', 'POST', '/v1/responses', 'openai', 'responses', 'json', 'json:model', 'inference:chat',
       stream_from='json:stream', replay_safe=True),
    _s('openai.responses.retrieve', 'GET', '/v1/responses/{response_id}', 'openai', 'responses', 'none', 'none',
       'inference:chat', handle_bound=True),
    _s('openai.responses.delete', 'DELETE', '/v1/responses/{response_id}', 'openai', 'responses', 'none', 'none',
       'inference:chat', handle_bound=True),
    _s('openai.responses.cancel', 'POST', '/v1/responses/{response_id}/cancel', 'openai', 'responses', 'none', 'none',
       'inference:chat', handle_bound=True),
    _s('openai.responses.input_items', 'GET', '/v1/responses/{response_id}/input_items', 'openai', 'responses', 'none',
       'none', 'inference:chat', handle_bound=True),
    _s('openai.embeddings', 'POST', '/v1/embeddings', 'openai', 'embeddings', 'json', 'json:model', 'inference:embeddings',
       replay_safe=True),
    _s('openai.audio.speech', 'POST', '/v1/audio/speech', 'openai', 'speech', 'json', 'json:model', 'inference:speech'),
    _s('openai.audio.transcriptions', 'POST', '/v1/audio/transcriptions', 'openai', 'transcription', 'multipart',
       'form:model', 'inference:transcription'),
    _s('openai.audio.translations', 'POST', '/v1/audio/translations', 'openai', 'transcription', 'multipart', 'form:model',
       'inference:transcription'),
    _s('openai.images.generations', 'POST', '/v1/images/generations', 'openai', 'images', 'json', 'json:model',
       'inference:images'),
    _s('openai.images.edits', 'POST', '/v1/images/edits', 'openai', 'images', 'multipart', 'form:model', 'inference:images'),
    _s('openai.moderations', 'POST', '/v1/moderations', 'openai', 'moderation', 'json', 'json:model',
       'inference:moderation', replay_safe=True),
    # The de-facto rerank shape (query, documents, top_n, return_documents -> results[index, relevance_score]) that
    # Together, Fireworks, OpenRouter, LiteLLM and Titanos' own API Plus route speak.
    _s('openai.rerank', 'POST', '/v1/rerank', 'openai', 'rerank', 'json', 'json:model', 'inference:plus',
       replay_safe=True),
    # Anthropic
    _s('anthropic.messages', 'POST', '/v1/messages', 'anthropic', 'chat', 'json', 'json:model', 'inference:chat',
       stream_from='json:stream', replay_safe=True),
    _s('anthropic.count_tokens', 'POST', '/v1/messages/count_tokens', 'anthropic', 'chat', 'json', 'json:model',
       'inference:chat', replay_safe=True),
    # Gemini
    _s('gemini.generateContent', 'POST', '/v1beta/models/{model:path}:generateContent', 'gemini', 'chat', 'json',
       'path:model_action', 'inference:chat', replay_safe=True),
    _s('gemini.streamGenerateContent', 'POST', '/v1beta/models/{model:path}:streamGenerateContent', 'gemini', 'chat',
       'json', 'path:model_action', 'inference:chat', stream_from='path:streamGenerateContent', replay_safe=True),
    _s('gemini.embedContent', 'POST', '/v1beta/models/{model:path}:embedContent', 'gemini', 'embeddings', 'json',
       'path:model_action', 'inference:embeddings', replay_safe=True),
    _s('gemini.batchEmbedContents', 'POST', '/v1beta/models/{model:path}:batchEmbedContents', 'gemini', 'embeddings',
       'json', 'path:model_action', 'inference:embeddings', replay_safe=True),
    _s('gemini.countTokens', 'POST', '/v1beta/models/{model:path}:countTokens', 'gemini', 'chat', 'json',
       'path:model_action', 'inference:chat', replay_safe=True),
    # Ollama (streams by default)
    _s('ollama.chat', 'POST', '/api/chat', 'ollama', 'chat', 'json', 'json:model', 'inference:chat',
       stream_from='json:stream', replay_safe=True, stream_default=True),
    _s('ollama.generate', 'POST', '/api/generate', 'ollama', 'chat', 'json', 'json:model', 'inference:chat',
       stream_from='json:stream', replay_safe=True, stream_default=True),
    _s('ollama.embed', 'POST', '/api/embed', 'ollama', 'embeddings', 'json', 'json:model', 'inference:embeddings',
       replay_safe=True),
    _s('ollama.embeddings', 'POST', '/api/embeddings', 'ollama', 'embeddings', 'json', 'json:model',
       'inference:embeddings', replay_safe=True),
    # ElevenLabs
    _s('elevenlabs.tts', 'POST', '/v1/text-to-speech/{voice_id}', 'elevenlabs', 'speech', 'json', 'json:model_id',
       'inference:speech', model_absent='local_passthrough'),
    _s('elevenlabs.tts_stream', 'POST', '/v1/text-to-speech/{voice_id}/stream', 'elevenlabs', 'speech', 'json',
       'json:model_id', 'inference:speech', stream_from='path:/stream', stream_default=True),
    _s('elevenlabs.stt', 'POST', '/v1/speech-to-text', 'elevenlabs', 'transcription', 'multipart', 'form:model_id',
       'inference:transcription'),
    # Catalogs (merged in their handlers)
    _s('openai.models', 'GET', '/v1/models', 'openai', 'catalog', 'none', 'none', 'models:read', kind='catalog'),
    _s('openai.model', 'GET', '/v1/models/{name:path}', 'openai', 'catalog', 'none', 'path:model', 'models:read',
       kind='catalog'),
    _s('ollama.tags', 'GET', '/api/tags', 'ollama', 'catalog', 'none', 'none', 'models:read', kind='catalog'),
    _s('ollama.ps', 'GET', '/api/ps', 'ollama', 'catalog', 'none', 'none', 'models:read', kind='catalog'),
    _s('gemini.models', 'GET', '/v1beta/models', 'gemini', 'catalog', 'none', 'none', 'models:read', kind='catalog'),
    _s('gemini.model', 'GET', '/v1beta/models/{model:path}', 'gemini', 'catalog', 'none', 'path:model', 'models:read',
       kind='catalog'),
    _s('elevenlabs.voices', 'GET', '/v1/voices', 'elevenlabs', 'catalog', 'none', 'none', 'models:read', kind='catalog'),
    _s('elevenlabs.voice', 'GET', '/v1/voices/{voice_id}', 'elevenlabs', 'catalog', 'none', 'none', 'models:read',
       kind='catalog'),
    # Stages (resolved in-process; body.location selects placement, default local_only)
    _s('plus.translations', 'POST', '/v1/translations', 'openai', 'chat', 'json', 'json:model', 'inference:plus',
       kind='stage'),
    _s('plus.audio_analysis', 'POST', '/v1/audio/analysis', 'openai', 'transcription', 'json', 'json:model',
       'inference:plus', kind='stage'),
    _s('plus.sentiment', 'POST', '/v1/sentiment', 'openai', 'chat', 'json', 'json:model', 'inference:plus', kind='stage'),
    _s('plus.sentiment_compose', 'POST', '/v1/sentiment/compose', 'openai', 'chat', 'json', 'json:model',
       'inference:plus', kind='stage'),
    _s('plus.retrieval_answer', 'POST', '/v1/retrieval/answer', 'openai', 'chat', 'json', 'json:model', 'inference:plus',
       kind='stage'),
    _s('plus.extensions_infer', 'POST', '/v1/extensions/infer', 'openai', 'chat', 'json', 'json:model', 'inference:plus',
       kind='stage'),
    _s('jobs.create', 'POST', '/api/jobs', 'openai', 'asset', 'json', 'json:model', 'jobs', kind='stage'),
)

SURFACES: dict[str, Surface] = {row.name: row for row in _ROWS}
assert len(SURFACES) == len(_ROWS), 'duplicate surface name'


def template_regex(template: str) -> re.Pattern:
    """Compile a FastAPI-style path template: '{x}' one segment, '{x:path}' greedy across segments."""
    out, position = ['^'], 0
    for match in re.finditer(r'\{([A-Za-z_][A-Za-z0-9_]*)(:path)?\}', template):
        out.append(re.escape(template[position:match.start()]))
        out.append(f'(?P<{match.group(1)}>.+)' if match.group(2) else f'(?P<{match.group(1)}>[^/]+)')
        position = match.end()
    out.append(re.escape(template[position:]) + '$')
    return re.compile(''.join(out))


def literal_weight(template: str) -> int:
    """Specificity: characters outside parameters (a literal route beats a parameterised one that also matches)."""
    return len(re.sub(r'\{[^}]*\}', '', template))


_COMPILED = tuple(sorted(((row, template_regex(row.path)) for row in _ROWS), key=lambda item: -literal_weight(item[0].path)))


def match(method: str, path: str) -> tuple[Surface, dict[str, str]] | None:
    """The surface for (method, path) with its path parameters, or None. Exact paths only (no trailing-slash folding)."""
    method = method.upper()
    for row, pattern in _COMPILED:
        if row.method != method:
            continue
        found = pattern.match(path)
        if found:
            return row, found.groupdict()
    return None


def by_kind(kind: str) -> tuple[Surface, ...]:
    return tuple(row for row in _ROWS if row.kind == kind)


__all__ = ['MODEL_FROM', 'PROTOCOLS', 'STREAM_FROM', 'SURFACES', 'Surface', 'TOKEN_OUTPUT_SURFACES', 'by_kind',
           'literal_weight', 'match', 'template_regex']
