"""Standard client fields the local OpenAI and Anthropic routes accept without acting on them.

Agent harnesses send identity, prompt-cache, storage and reasoning hints on every call: Claude Code sends Anthropic
metadata.user_id; Codex sends store, include, reasoning, prompt_cache_key and text.verbosity; routers
carry them through when they translate between formats or apply a target's reasoning_effort; HiveMind's gateway adds
Ollama's options object ({"num_ctx": N}) to every chat request it sends a non-cloud backend. The local models have no use
for them, but refusing them with 422 breaks those clients when Titanos serves a local model. Each such field is therefore
checked for its documented JSON type, removed before the strict local schema validates the rest, and named in the
X-Titanos-Ignored response header, so nothing is dropped silently. options.num_ctx is the one key with a local meaning:
context_option leaves it unnamed on a llama-cpp model, whose runtime is sized to it up to the model's largest per-slot
context, and names it above that and on the other backends; every other options key is options.<key>.

Fields whose meaning the local routes cannot honour are refused with a typed 400 instead of being ignored: store=true (no
local response or completion store), previous_response_id and background (local Responses are stateless and synchronous),
Anthropic container (no code-execution containers), output_config.format (structured output on the local Anthropic route)
and embeddings dimensions (no backend truncates to a requested size: llama.cpp b10809 ignores the field). Everything else
keeps today's strict validation (unknown fields: 422; unknown_fields names them instead in managed role with
TITANOS_UNKNOWN_FIELDS=ignore); the llama.cpp sampling fields are schema fields (titanos.openai_params).
"""
from __future__ import annotations

from collections.abc import Mapping

from .engine import EngineError

IGNORED_HEADER = 'X-Titanos-Ignored'
MAX_TEXT = 4096

# field -> accepted JSON type(s); None means any JSON value
_IDENTITY = {'user': str, 'safety_identifier': str, 'prompt_cache_key': str, 'prompt_cache_retention': str,
             'service_tier': str}
CHAT = {'metadata': dict, 'store': bool, 'reasoning_effort': str, 'verbosity': str, 'options': dict, **_IDENTITY}
COMPLETION = {'options': dict, **_IDENTITY}
RESPONSES = {'include': list, 'reasoning': dict, 'truncation': str, 'top_logprobs': int, **_IDENTITY}
ANTHROPIC = {'metadata': dict, 'service_tier': str, 'thinking': dict, 'inference_geo': str, 'cache_control': dict,
             'context_management': dict, 'output_config': dict}
EMBEDDINGS = {'user': str}
KINDS = {'chat': CHAT, 'completion': COMPLETION, 'responses': RESPONSES, 'anthropic': ANTHROPIC, 'embeddings': EMBEDDINGS}
# fields whose unservable values answer a typed 400 (store true, background true, output_config.format: unsupported_capability; a
# non-object options or a non-positive-integer options.num_ctx; any previous_response_id, container or dimensions)
REFUSED = {'chat': frozenset({'store', 'options'}), 'completion': frozenset({'options'}), 'responses': frozenset({'store', 'previous_response_id', 'background'}),
           'anthropic': frozenset({'container', 'output_config'}), 'embeddings': frozenset({'dimensions'})}


def _typed(name: str, value, kind) -> None:
    if value is None:
        return
    if kind is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise EngineError(f'{name} must be an integer')
    if kind is not int and not isinstance(value, kind):
        raise EngineError(f'{name} must be a JSON {"object" if kind is dict else kind.__name__}')
    if isinstance(value, str) and len(value) > MAX_TEXT:
        raise EngineError(f'{name} exceeds {MAX_TEXT} characters')


def _unsupported(message: str):
    raise EngineError(message, 400, 'unsupported_capability')


def split(kind: str, data):
    """(data without the accepted fields, ignored field names in request order, options.num_ctx or None). data that is not a
    JSON object is returned unchanged (the strict schema reports it); kind None (a schema without compat fields) returns it
    as is. num_ctx is carried for the chat and completions routes, whose handler checks it against the model's backend
    (context_option) after the schema has validated the rest. Raises EngineError 400 for a wrong type or an unservable value."""
    if kind is None or not isinstance(data, Mapping):
        return data, (), None
    table = KINDS[kind]
    out = dict(data)
    ignored = []
    num_ctx = None
    if kind == 'embeddings':
        if out.get('dimensions') is not None:
            _unsupported('dimensions is not available on local embedding models: no backend truncates to a requested size '
                         '(llama.cpp b10809 ignores the field); omit dimensions and use the model\'s native vector size')
        if 'dimensions' in out:
            out.pop('dimensions')
            ignored.append('dimensions')
    if kind == 'chat' or kind == 'responses':
        if out.get('store') is True:
            _unsupported('store=true is not available on local models (no local response store); send store=false')
    if kind == 'responses':
        if out.get('previous_response_id') is not None:
            _unsupported('Local Responses are stateless: send the whole conversation in input instead of '
                         'previous_response_id')
        if out.get('background'):
            _unsupported('Local Responses run synchronously; background=true is not available')
        if 'background' in out:
            out.pop('background')
            ignored.append('background')
        text = out.get('text')
        if isinstance(text, dict) and 'verbosity' in text:
            _typed('text.verbosity', text['verbosity'], str)
            out['text'] = {k: v for k, v in text.items() if k != 'verbosity'} or None
            if out['text'] is None:
                out.pop('text')
            ignored.append('text.verbosity')
    if kind == 'anthropic':
        if out.get('container') is not None:
            _unsupported('Code-execution containers are not available on local models')
        config = out.get('output_config')
        if isinstance(config, dict) and config.get('format') is not None:
            _unsupported('output_config.format (structured output) is not available on the local Anthropic route; '
                         'use /v1/chat/completions with response_format on a llama-cpp model')
        if 'container' in out:
            out.pop('container')
            ignored.append('container')
    if kind in ('chat', 'completion') and 'options' in out:
        # HiveMind's Ollama options object: num_ctx is kept for the route's context check (context_option), the other keys
        # are named as options.<key>; a null options is named whole, as a null compat field is below
        options = out.pop('options')
        _typed('options', options, dict)
        if options is None:
            ignored.append('options')
        for name, value in (options or {}).items():
            if name != 'num_ctx':
                ignored.append(f'options.{name}')
            elif value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise EngineError('options.num_ctx must be a positive integer')
                num_ctx = value
    for name, expected in table.items():
        if name not in out:
            continue
        value = out.pop(name)
        _typed(name, value, expected)
        if name == 'metadata' and isinstance(value, dict) and (
                len(value) > 16 or any(not isinstance(k, str) or len(k) > 64 for k in value)):
            raise EngineError('metadata accepts at most 16 keys of at most 64 characters')
        ignored.append(name)
    return out, tuple(ignored), num_ctx


def context_option(spec, model: str, num_ctx) -> list:
    """The names X-Titanos-Ignored gets for options.num_ctx on the OpenAI chat and completions routes, where options is a
    foreign extension (Ollama's, sent by HiveMind's gateway with every request) rather than a native field. A llama-cpp
    model honours it: the leased runtime starts, or restarts once its requests have drained, with a per-slot context of
    the next power of two at or above num_ctx (titanos.native_context; server.generate passes num_ctx on to NativePool),
    so nothing is named up to the largest per-slot context (native.max_slot_context, TITANOS_NATIVE_MAX_SLOT_CONTEXT or
    the GGUF trained context_length). A larger num_ctx is named and not honoured (the chat and completion handlers then
    leave the runtime at its size), never a 400: options is foreign here, and protocols.context_window keeps the 400
    context_limit for /api/chat, where num_ctx is a native field. The other backends have no context option (the
    checkpoint's own length bounds prompt plus max_tokens), so options.num_ctx is named there. A non-object options or a non-positive-integer num_ctx is still a
    typed 400 (split). spec None (an unregistered model) names nothing: the route's own 404 follows. model names the
    model in the refusal and keeps the two routes' callers (the chat and completion handlers in server.py) in one shape
    with the Ollama closure protocols.context_window."""
    if num_ctx is None or spec is None:
        return []
    if spec.backend == 'llama-cpp' and spec.native is not None and _sized(spec, num_ctx):
        return []
    return ['options.num_ctx']


def header_value(names) -> str | None:
    return ', '.join(dict.fromkeys(names)) if names else None


def attach(result, response, names):
    """Name the ignored fields on the answer: a returned Response gets the header itself, a returned dict through
    FastAPI's injected response object."""
    value = header_value(names)
    if not value:
        return result
    from fastapi.responses import Response
    if isinstance(result, Response):
        result.headers[IGNORED_HEADER] = value
    elif response is not None:
        response.headers[IGNORED_HEADER] = value
    return result


def sized_context(spec, payload) -> list:
    """context_option for a parsed chat or completion payload; a named options.num_ctx is cleared from the payload, so
    server.generate does not size the native runtime to it."""
    names = context_option(spec, payload.model, payload._num_ctx)
    if names:
        payload._num_ctx = None
    return names


LENIENT_KINDS = frozenset({'chat', 'completion', 'responses', 'embeddings', 'ollama'})


def unknown_fields(kind, cls, data, ignored=()):
    """(data, ignored) with the unknown top-level fields stripped and appended to ignored, only under TITANOS_ROLE=managed
    with TITANOS_UNKNOWN_FIELDS=ignore and on the OpenAI and Ollama JSON schemas (LENIENT_KINDS); otherwise data is returned
    unchanged and the strict schema answers 422. Unknown nested fields and wrong-typed known fields stay errors."""
    from .settings import current_settings
    settings = current_settings()
    if kind not in LENIENT_KINDS or not isinstance(data, Mapping) or settings.role != 'managed' or settings.unknown_fields != 'ignore':
        return data, tuple(ignored)
    known = set()
    for name, field in cls.model_fields.items():
        known.update(alias for alias in (name, field.alias, field.validation_alias) if isinstance(alias, str))
    extra = [name for name in data if name not in known]
    if not extra:
        return data, tuple(ignored)
    return {k: v for k, v in data.items() if k in known}, (*ignored, *extra)


__all__ = ['ANTHROPIC', 'CHAT', 'COMPLETION', 'EMBEDDINGS', 'IGNORED_HEADER', 'KINDS', 'LENIENT_KINDS', 'REFUSED', 'RESPONSES',
           'attach', 'context_option', 'header_value', 'sized_context', 'split', 'unknown_fields']


def _sized(spec, num_ctx):
    """True when a llama-cpp runtime can be sized to options.num_ctx (titanos.native_context): num_ctx is at most the
    model's largest per-slot context."""
    from .native_context import max_slot_context, registered_slot_context
    return num_ctx <= registered_slot_context(spec.native) or num_ctx <= max_slot_context(spec)
