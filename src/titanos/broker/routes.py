"""Inference and public routes the app gains with the broker (register_broker_routes).

Every route here is a typed answer when no target exists, never a 501 stub. BrokerDispatch answers the broker-only
surfaces first (dispatch.BROKER_ONLY_SURFACES, even while the broker is inactive): cloud targets through the adapters,
local targets through the bridges or the local routes below. The handlers here therefore answer:
- POST /v1/moderations: a local model gives 503 classifier_unavailable, an unknown one 404 model_not_found (a cloud
  openai target is passed through by BrokerDispatch).
- GET/DELETE /v1/responses/{id}, POST .../cancel, GET .../input_items: handle-bound; BrokerDispatch serves the owner
  (or an admin); reaching the route means no binding: 404 response_not_found.
- POST /v1/messages/count_tokens (the local route of anthropic.count_tokens, R22): a local llama-cpp model is counted
  with its own template and tokenizer (NativePool.count_tokens: /apply-template then /tokenize); any other local model
  gives 400 broker_capability_unavailable and an unknown name 404 model_not_found.
- POST /v1/text-to-speech/{voice_id}/stream: the local ElevenLabs text-to-speech route (the WAV answer as it comes).
- POST /v1/speech-to-text: BrokerDispatch bridges local models through /v1/audio/transcriptions; the route itself
  answers a body that could not be read (400 invalid_request).
- GET /v1/voices and /v1/voices/{voice_id}: the voices of the registry's speech models (ElevenLabs shape), followed by
  the upstream voices of the ElevenLabs upstreams the caller may see (broker.cloud_voices: cloud:<id>, fetched with a
  catalog grant and cached; a local voice id wins a collision). Local rows follow a key's model globs and target
  classes (broker.visible_local).
- GET /v1beta/models and /v1beta/models/{model}: local chat and embedding models plus the broker entries the caller may
  use (Gemini shape). POST /v1beta/models/{model}:{action} splits on the LAST ':' against the allowlist
  generateContent | streamGenerateContent | embedContent | batchEmbedContents | countTokens; anything else 404. A local
  countTokens on a llama-cpp model is counted natively ({totalTokens}).
- GET /v1/health/ready (public): readiness.response: 200 {status:'ready', version, role, models, checks} or 503 {status:'not_ready', reason, message, ...} (titanos/readiness.py).
- GET /api/version (registered, and public, only in managed role): {name:'Titanos', version, role}.
Errors are BrokerErrors rendered in the inbound protocol's envelope by the app's BrokerError handler (server.py).
"""
from __future__ import annotations

import json

from fastapi import Request

from .errors import BrokerError
from .profiles import PROFILES

GEMINI_ACTIONS = ('generateContent', 'streamGenerateContent', 'embedContent', 'batchEmbedContents', 'countTokens')
# public model metadata a Gemini upstream publishes, passed through on GET /v1beta/models for its models
GEMINI_META_KEYS = ('inputTokenLimit', 'outputTokenLimit', 'thinking', 'temperature', 'maxTemperature', 'topP', 'topK')
CHAT_TASKS = frozenset({'chat', 'vision'})


def _json_body(raw: bytes) -> dict:
    try:
        document = json.loads(raw) if raw.strip() else {}
    except (ValueError, UnicodeDecodeError):
        raise BrokerError('The request body must be a JSON object', 'invalid_request') from None
    if not isinstance(document, dict):
        raise BrokerError('The request body must be a JSON object', 'invalid_request')
    return document


def register_broker_routes(app, broker, *, engine, native_getter, config, gate=None) -> None:
    """Attach the broker's inference and public routes. engine: the EngineSlot; native_getter(): the live NativePool;
    config: the registry (mutated in place on reload); gate: the server's reload gate dict (readiness.report)."""
    from .. import __version__, readiness
    from .translate import anthropic_messages, gemini, openai_chat

    role = broker.settings.role

    def local_spec(name):
        return config.models.get(name) if isinstance(name, str) else None

    def native_has(name) -> bool:
        native = native_getter()
        return native is not None and name in getattr(native, 'models', {})

    def principal_of(request: Request):
        return request.scope.get('titanos.principal')

    async def native_count(model: str, messages: list, tools) -> int:
        return await native_getter().count_tokens(model, messages, tools=tools or None)

    def local_or_missing(model, what: str):
        if not isinstance(model, str) or not model.strip():
            raise BrokerError('model is required', 'model_required')
        if local_spec(model) is None:
            raise BrokerError(f'Model {model!r} not found', 'model_not_found')
        raise BrokerError(f'{what} is not available for local model {model!r}', 'broker_capability_unavailable')

    # -- OpenAI ------------------------------------------------------------------------------------------------------
    @app.post('/v1/moderations', tags=['broker'])
    async def moderations(request: Request):
        document = _json_body(await request.body())
        model = document.get('model')
        if not isinstance(model, str) or not model.strip():
            raise BrokerError('model is required', 'model_required')
        if local_spec(model) is not None:
            raise BrokerError('Moderation needs a cloud classifier target', 'classifier_unavailable')
        raise BrokerError(f'Model {model!r} not found', 'model_not_found')

    def response_not_found():
        return BrokerError('Response not found', 'response_not_found', origin='policy')

    @app.get('/v1/responses/{response_id}', tags=['broker'])
    def response_retrieve(response_id: str):
        raise response_not_found()

    @app.delete('/v1/responses/{response_id}', tags=['broker'])
    def response_delete(response_id: str):
        raise response_not_found()

    @app.post('/v1/responses/{response_id}/cancel', tags=['broker'])
    def response_cancel(response_id: str):
        raise response_not_found()

    @app.get('/v1/responses/{response_id}/input_items', tags=['broker'])
    def response_input_items(response_id: str):
        raise response_not_found()

    # -- Anthropic ---------------------------------------------------------------------------------------------------
    @app.post('/v1/messages/count_tokens', tags=['broker'])
    async def anthropic_count_tokens(request: Request):
        document = _json_body(await request.body())
        model = document.get('model')
        if isinstance(model, str) and native_has(model):
            ir = anthropic_messages.decode_request({**document, 'max_tokens': document.get('max_tokens') or 1},
                                                   notes=[])
            body = openai_chat.encode_request(ir, None, notes=[])
            count = await native_count(model, body['messages'], body.get('tools'))
            return {'input_tokens': count}
        local_or_missing(model, 'Token counting')

    # -- ElevenLabs --------------------------------------------------------------------------------------------------
    @app.post('/v1/text-to-speech/{voice_id}/stream', tags=['broker'])
    async def elevenlabs_speech_stream(voice_id: str, request: Request):
        handler = getattr(app.state, 'elevenlabs_speech', None)
        if handler is None:
            raise BrokerError('Local text-to-speech is unavailable', 'broker_capability_unavailable')
        return await handler(voice_id, request)

    @app.post('/v1/speech-to-text', tags=['broker'])
    async def elevenlabs_speech_to_text(request: Request):
        raise BrokerError('Send multipart/form-data with model_id and file', 'invalid_request')

    def voices(principal) -> list[dict]:
        out = []
        for name, spec in sorted(config.models.items()):
            if spec.task != 'speech' or not broker.visible_local(principal, name):
                continue
            for voice in sorted(set(spec.voices) | {'default'}):
                out.append({'voice_id': voice, 'name': voice, 'category': 'local',
                            'labels': {'model_id': name}, 'available_for_tiers': [], 'settings': None})
        return out

    async def all_voices(principal) -> list[dict]:
        local = voices(principal)
        seen = {voice['voice_id'] for voice in local}
        cloud = [voice for voice in await broker.cloud_voices(principal) if voice['voice_id'] not in seen]
        return local + cloud

    @app.get('/v1/voices', tags=['broker'])
    async def voice_list(request: Request):
        return {'voices': await all_voices(principal_of(request))}

    @app.get('/v1/voices/{voice_id}', tags=['broker'])
    async def voice_get(voice_id: str, request: Request):
        for voice in await all_voices(principal_of(request)):
            if voice['voice_id'] == voice_id:
                return voice
        raise BrokerError(f'Voice {voice_id!r} not found', 'not_found')

    # -- Gemini ------------------------------------------------------------------------------------------------------
    def gemini_entry(name: str, methods: list, *, remote: bool = False, owner: str | None = None) -> dict:
        entry = {'name': 'models/' + name, 'baseModelId': name, 'version': '1', 'displayName': name,
                 'description': f'Titanos {"broker" if remote else "local"} model', 'supportedGenerationMethods': methods}
        if remote:
            entry['titanos'] = {'remote': True, 'owned_by': owner}
        return entry

    def gemini_models(principal) -> list[dict]:
        out = []
        for name, spec in sorted(config.models.items()):
            if not broker.visible_local(principal, name):
                continue  # a key sees the local models its constraints allow (keyless and bootstrap: all)
            if spec.task in CHAT_TASKS:
                out.append(gemini_entry(name, ['generateContent', 'streamGenerateContent', 'countTokens']))
            elif spec.task == 'embeddings':
                out.append(gemini_entry(name, ['embedContent', 'batchEmbedContents']))
        for item in broker.catalog_for(principal):
            capabilities = set((item.get('titanos') or {}).get('capabilities') or ())
            if (item.get('titanos') or {}).get('class') == 'virtual':
                capabilities = {(item.get('titanos') or {}).get('capability')}
            methods = []
            if 'chat' in capabilities:
                methods += ['generateContent', 'streamGenerateContent', 'countTokens']
            if 'embeddings' in capabilities:
                methods += ['embedContent', 'batchEmbedContents']
            meta = gemini_meta(item)
            if meta.get('supportedGenerationMethods'):
                # a Gemini upstream model: its own published methods (the ones without a typed surface, such as
                # predict or batchGenerateContent, are served by the native passthrough) and limits
                methods = list(meta['supportedGenerationMethods'])
            if methods:
                entry = gemini_entry(item['id'], methods, remote=True, owner=item.get('owned_by'))
                for key in GEMINI_META_KEYS:
                    if key in meta:
                        entry[key] = meta[key]
                out.append(entry)
        return out

    def gemini_meta(item: dict) -> dict:
        """The live-catalog metadata of a broker entry whose target is a model of a Gemini upstream ({} otherwise)."""
        try:
            catalog = broker.snapshot().catalog
            target = catalog.lookup(item['id'])
        except Exception:
            return {}
        upstream_id = getattr(target, 'upstream_id', None)
        upstream = catalog.config.upstreams.get(upstream_id) if upstream_id else None
        if upstream is None or PROFILES[upstream.profile].adapter != 'gemini':
            return {}
        return catalog.model_meta(upstream_id, target.upstream_model)

    @app.get('/v1beta/models', tags=['broker'])
    def gemini_model_list(request: Request):
        return {'models': gemini_models(principal_of(request))}

    @app.get('/v1beta/models/{model:path}', tags=['broker'])
    def gemini_model_get(model: str, request: Request):
        name = model[len('models/'):] if model.startswith('models/') else model
        principal = principal_of(request)
        entries = gemini_models(principal)
        for entry in entries:
            if entry['baseModelId'] == name:
                return entry
        # google-genai's models.get('gemini-2.5-flash'): a bare provider id resolves exactly as generateContent resolves
        # it (local models and aliases were matched above; then the single visible Gemini upstream that lists it)
        target = None
        if broker.active(principal):
            try:
                found = broker.snapshot().catalog.resolve_vendor(name, 'gemini', principal or broker.default_principal())
            except BrokerError:
                found = None
            target = getattr(found, 'target', None)
        for entry in entries if target is not None else ():
            if entry['baseModelId'] == target.ref:
                return {**entry, 'name': 'models/' + name, 'baseModelId': name,
                        'titanos': {**(entry.get('titanos') or {}), 'ref': target.ref}}
        raise BrokerError(f'Model {name!r} not found', 'model_not_found')

    @app.post('/v1beta/models/{model_action:path}', tags=['broker'])
    async def gemini_model_action(model_action: str, request: Request):
        model, sep, action = model_action.rpartition(':')
        if not sep or action not in GEMINI_ACTIONS or not model:
            raise BrokerError(f'Unknown model action {action or model_action!r}', 'not_found')
        if model.startswith('models/'):
            model = model[len('models/'):]
        document = _json_body(await request.body())
        if action == 'countTokens' and native_has(model):
            request_body = document.get('generateContentRequest') if isinstance(
                document.get('generateContentRequest'), dict) else document
            ir = gemini.decode_request(request_body, model=model, notes=[])
            body = openai_chat.encode_request(ir, None, notes=[])
            return {'totalTokens': await native_count(model, body['messages'], body.get('tools'))}
        if action == 'countTokens':
            local_or_missing(model, 'Token counting')
        raise BrokerError('The request could not be routed; check the model name and body', 'invalid_request')

    # -- public ------------------------------------------------------------------------------------------------------
    @app.get('/v1/health/ready', tags=['health'])
    def health_ready():
        return readiness.response(broker, gate=gate, config=config, engine=engine, role=role)

    if role == 'managed':
        @app.get('/api/version', tags=['health'])
        def api_version():
            return {'name': 'Titanos', 'version': __version__, 'role': role}


__all__ = ['GEMINI_ACTIONS', 'register_broker_routes']
