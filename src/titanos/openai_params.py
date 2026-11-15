"""Standard OpenAI request fields on local models: forwarded where the backend implements them, typed 400s where it does not.

Stock OpenAI SDKs and agent frameworks send logprobs/top_logprobs, logit_bias, presence_penalty, frequency_penalty, n, seed,
stop arrays and response_format json_schema on chat and completions, dimensions/encoding_format/user on embeddings and
OpenAI voice names and speed on speech. The managed llama.cpp server (b10809, verified against the CPU binary) honours
every generation field of that list, so a llama-cpp model's request reaches llama-server with them intact (native.proxy).
The in-process backends implement a subset:

- transformers: seed (torch.manual_seed), repeat_penalty (repetition_penalty) and logit_bias (sequence_bias over single
  token ids) are mapped; logprobs, top_logprobs, presence_penalty, frequency_penalty and n >= 2 have no generation-config
  equivalent and answer 400 unsupported_capability naming the field;
- openvino-genai: the portable options only (genai.py); a logit_bias is refused by its own field check;
- embeddings: dimensions is refused on every backend (llama.cpp b10809 ignores it and no local model declares Matryoshka
  truncation), user is accepted and named in X-Titanos-Ignored (request_compat); a GGUF embedding model is proxied to
  llama-server /v1/embeddings, which does not implement the Titanos-only max_tokens/truncate controls (typed 400);
- speech: a client voice name is mapped through the model's voice_map (ModelSpec) to the backend voice id; speed is
  honoured by sherpa-onnx and refused with a 400 naming it by the transformers, xtts and qwen3-tts backends.

Inert values (logprobs false, top_logprobs 0, penalties 0, logit_bias {}, n 1) are the OpenAI defaults and are removed
before the local engine sees them; nothing is dropped silently.
"""
from .engine import EngineError

# fields the llama.cpp server implements that no in-process generation backend does (local requests: typed 400)
NATIVE_ONLY = ('logprobs', 'top_logprobs', 'presence_penalty', 'frequency_penalty')
# the OpenAI default of each field: a request carrying it asks for nothing and is served
INERT = {'logprobs': (None, False), 'top_logprobs': (None, 0), 'presence_penalty': (None, 0), 'frequency_penalty': (None, 0),
         'n': (None, 1), 'logit_bias': (None, {})}
GENERATION_ROUTES = {'chat', 'vision'}


def _inert(name, value):
    return any(value is item or (type(value) is type(item) or isinstance(value, (int, float)) and isinstance(item, (int, float))
                                 and not isinstance(value, bool)) and value == item for item in INERT[name])


def unsupported(name, spec, detail):
    """400 unsupported_capability naming the field, the backend (a ModelSpec or a backend name) and what to do instead."""
    backend = spec if isinstance(spec, str) else getattr(spec, 'backend', None) or 'local'
    raise EngineError(f'{name} is not available on this {backend} model: {detail}', 400, 'unsupported_capability')


def local_generation(data, spec):
    """The in-process engine's request data: inert fields removed, a live value of a field only llama.cpp implements is a
    typed 400 naming the field (and the model's backend). logit_bias stays for the transformers mapping."""
    out = dict(data)
    for name in (*NATIVE_ONLY, 'n', 'logit_bias'):
        if name not in out:
            continue
        if _inert(name, out[name]):
            out.pop(name)
        elif name == 'n':
            unsupported('n', spec, f'one choice per request on this backend (n={out[name]} requested); llama-cpp models return n choices')
        elif name != 'logit_bias':
            unsupported(name, spec, 'the backend has no generation-config equivalent; llama-cpp models honour it (llama.cpp server)')
    return out


def native_generation(native, model):
    """A llama-cpp model on a generation route: True for chat/vision models; a GGUF embedding model answers 400 instead of
    reaching llama-server's chat route."""
    spec = native.models.get(model)
    if spec is not None and spec.task not in GENERATION_ROUTES:
        raise EngineError(f'{model} is a llama-cpp {spec.task} model; use /v1/embeddings (or Ollama /api/embed), not a generation route',
                          400, 'unsupported_capability')
    return True


def native_embeddings(payload):
    """The llama-server /v1/embeddings body for a GGUF embedding model: model, worker (lease only), input and
    encoding_format. The Titanos-only truncation controls have no native equivalent and are refused by name."""
    for name in ('max_tokens', 'truncate'):
        if getattr(payload, name, None) is not None:
            unsupported(name, 'llama-cpp', 'llama-cpp embedding models bound input by the registered native.context_size; omit it')
    data = payload.model_dump()
    return {key: data[key] for key in ('model', 'worker', 'input', 'encoding_format') if data.get(key) is not None}


def transformers_bias(data):
    """generate() kwargs for logit_bias on transformers: sequence_bias over single-token sequences, token ids as keys like
    OpenAI (a non-integer key is a ValueError, answered 400 by the engine)."""
    bias = data.get('logit_bias')
    if not bias:
        return {}
    try:
        return {'sequence_bias': [[[int(token)], float(value)] for token, value in bias.items()]}
    except (TypeError, ValueError):
        raise ValueError('logit_bias keys must be token ids (integers as strings) for this transformers model') from None


def speech_voice(spec, data):
    """The request with a client voice name (alloy, nova, ...) replaced by the model's own voice id through spec.voice_map;
    names outside the map pass through unchanged for the backend's own check."""
    mapping = getattr(spec, 'voice_map', None) or {}
    voice = data.get('voice')
    if voice in mapping:
        return {**data, 'voice': mapping[voice]}
    return data


def speech_speed(data, backend):
    """Backends that synthesise at natural speed only: any other speed is a ValueError naming the field (400)."""
    speed = data.get('speed', 1.0)
    if speed is not None and speed != 1.0:
        raise ValueError(f'speed {speed} is not available on this {backend} speech model: it synthesises at natural speed 1.0 '
                         '(sherpa-onnx speech models honour speed)')


def legacy_logprobs(result, task, data):
    """The llama.cpp server answers /v1/completions logprobs in its chat shape, choices[].logprobs.content[] (token, logprob,
    top_logprobs[] objects); the OpenAI legacy completions shape the stock SDK types is {tokens, token_logprobs, top_logprobs
    (one token -> logprob map per position), text_offset}. The forwarded body, or every SSE chunk, is rewritten to that shape
    (text_offset counts characters from the start of the completion, across chunks) so a client that asked for logprobs
    finds them in the fields it reads; a chat request, a completion without logprobs and every other field pass untouched."""
    if task != 'completion' or not data.get('logprobs'):
        return result
    import json
    from fastapi.responses import Response, StreamingResponse
    from .responses import as_stream
    result = as_stream(result)
    headers = {k: v for k, v in result.headers.items() if k.lower() not in ('content-type', 'content-length')}
    if isinstance(result, StreamingResponse):
        source = result.body_iterator

        async def stream():
            import codecs
            decoder, pending, offsets = codecs.getincrementaldecoder('utf-8')(), '', {}
            try:
                async for chunk in source:
                    pending += decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
                    while '\n' in pending:
                        line, pending = pending.split('\n', 1)
                        yield (_legacy_line(line, offsets) + '\n').encode('utf-8')
                pending += decoder.decode(b'', final=True)
                if pending:
                    yield _legacy_line(pending, offsets).encode('utf-8')
            finally:
                if hasattr(source, 'aclose'):
                    await source.aclose()
        return StreamingResponse(stream(), media_type='text/event-stream', headers=headers)
    if isinstance(result, Response) and 200 <= result.status_code < 300:
        try:
            body = json.loads(result.body)
        except ValueError:
            return result
        if isinstance(body, dict) and _legacy_body(body, {}):
            return Response(json.dumps(body, ensure_ascii=False), status_code=result.status_code, headers=headers,
                            media_type=result.headers.get('content-type', 'application/json'))
    return result


def _legacy_line(line, offsets):
    """One SSE line: a data frame whose JSON carries chat-shaped logprobs is rewritten; anything else is returned as is."""
    import json
    text = line[:-1] if line.endswith('\r') else line
    if not text.startswith('data:'):
        return line
    value = text[5:].strip()
    if not value or value == '[DONE]':
        return line
    try:
        body = json.loads(value)
    except ValueError:
        return line
    if not (isinstance(body, dict) and _legacy_body(body, offsets)):
        return line
    return 'data: ' + json.dumps(body, ensure_ascii=False) + ('\r' if line.endswith('\r') else '')


def _legacy_body(body, offsets):
    """Rewrite every choice's chat-shaped logprobs in place; offsets (per choice index) carries text_offset across chunks."""
    changed = False
    for choice in body.get('choices') or []:
        if not isinstance(choice, dict):
            continue
        logprobs = choice.get('logprobs')
        content = logprobs.get('content') if isinstance(logprobs, dict) else None
        if not isinstance(content, list):
            continue
        index = choice.get('index', 0)
        offset = offsets.get(index, 0)
        legacy = {'tokens': [], 'token_logprobs': [], 'top_logprobs': [], 'text_offset': []}
        for entry in content:
            if not isinstance(entry, dict):
                continue
            token = entry.get('token') if isinstance(entry.get('token'), str) else ''
            legacy['tokens'].append(token)
            legacy['token_logprobs'].append(entry.get('logprob'))
            legacy['top_logprobs'].append({alt['token']: alt.get('logprob') for alt in entry.get('top_logprobs') or []
                                           if isinstance(alt, dict) and isinstance(alt.get('token'), str)})
            legacy['text_offset'].append(offset)
            offset += len(token)
        offsets[index] = offset
        choice['logprobs'] = legacy
        changed = True
    return changed


__all__ = ['INERT', 'NATIVE_ONLY', 'legacy_logprobs', 'local_generation', 'native_embeddings', 'native_generation', 'speech_speed',
           'speech_voice', 'transformers_bias', 'unsupported']
