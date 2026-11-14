"""Gemini (Generative Language API v1beta) adapter.

models/{model}:generateContent, :streamGenerateContent?alt=sse, :embedContent, :batchEmbedContents, :countTokens and the
paginated GET models (pageToken), authenticated with the x-goog-api-key header only (EgressClient; a ?key= query credential
is never sent). Model names 'models/<id>' are normalized. On passthrough the body is forwarded after a JSON round trip
(batchEmbedContents requests[].model and countTokens generateContentRequest.model are rewritten too); an inbound
streamGenerateContent without alt=sse gets the JSON-array stream, passed through as bytes.

Streams: one alt=sse data line is a whole GenerateContentResponse, so an image-output or audio frame can be a single
base64 line of several MiB. The held first frame (pre-commit error check), the usage tee (sse.LARGE_LINE_PROTOCOLS) and
the translated decode therefore accept lines up to sse.MEDIA_MAX_LINE (the egress media cap bounds the bytes anyway).
A non-streamed generateContent answer can carry whole images or audio as base64, so it is read under the media cap
(EgressClient.send media_cap=True), not the 16 MiB JSON cap.

Typed bridges (MEDIA_SURFACES, codecs in translate.gemini_media): /v1/audio/speech to a TTS model, /v1/images/generations
and /v1/images/edits to an image-output model, /v1/audio/transcriptions and /v1/audio/translations to a model with audio
understanding, all through generateContent. inbound.translated_call hands these surfaces to media_call().

Per-model capabilities (model_capabilities): the catalog narrows the upstream-wide capabilities by model family, so a
TTS model offers speech (and chat, because the native generateContent serves it), an image model images and chat, an
embedding model embeddings only, and Imagen, Veo, Live and AQA models nothing typed (the native passthrough reaches
them). list_models_meta keeps each model's public metadata (supportedGenerationMethods, token limits, thinking) for the
catalog and GET /v1beta/models.
"""
from __future__ import annotations

import json

from ..sse import MEDIA_MAX_LINE, gemini_usage, parse_sse
from ..surfaces import Surface
from ..translate import gemini, gemini_media
from ..translate.common import bad_upstream, refuse
from ..translate.ir import IRRequest, IRResponse
from ..types import EgressGrant, Usage
from .base import RawRequest, UpstreamResult
from .common import CATALOG_MAX_PAGES, AdapterBase, json_body, strip_titanos

TRANSLATED = frozenset({'openai.chat', 'openai.completions', 'openai.responses', 'anthropic.messages', 'ollama.chat',
                        'ollama.generate', 'openai.embeddings', 'ollama.embed', 'ollama.embeddings'})
MEDIA_SURFACES = frozenset({'openai.audio.speech', 'openai.images.generations', 'openai.images.edits',
                            'openai.audio.transcriptions', 'openai.audio.translations'})
# Capabilities per model family (None: the upstream-wide set minus the media-only capabilities).
_MEDIA_CAPS = frozenset({'speech', 'images'})
_FAMILY_CAPS = {
    'tts': frozenset({'chat', 'speech'}),
    'image': frozenset({'chat', 'vision', 'images', 'responses'}),
    'embedding': frozenset({'embeddings'}),
    'imagen': frozenset(),
    'veo': frozenset(),
    'live': frozenset(),
    'aqa': frozenset(),
}
_META_KEYS = ('displayName', 'description', 'version', 'inputTokenLimit', 'outputTokenLimit',
              'supportedGenerationMethods', 'thinking', 'temperature', 'maxTemperature', 'topP', 'topK')


class GeminiAdapter(AdapterBase):
    adapter_name = 'gemini'
    native_protocol = 'gemini'

    def supports(self, profile, surface: Surface) -> bool:
        return surface.name in profile.surfaces or surface.name in TRANSLATED or surface.name in MEDIA_SURFACES

    # streams -------------------------------------------------------------------------------------------------------------
    async def sse_frames(self, response, protocol: str = 'gemini'):
        """Parsed alt=sse frames after the error-first check, with the media line limit (translated streams)."""
        _, replay = await self.held(response, protocol)
        return parse_sse(replay, max_line=MEDIA_MAX_LINE)

    # passthrough ---------------------------------------------------------------------------------------------------------
    async def passthrough(self, grant: EgressGrant, egress, surface: Surface, raw: RawRequest, *,
                          notes: list | None = None, inject: dict | None = None,
                          allow_store: bool = False) -> UpstreamResult:
        name = surface.name
        if name not in self.profile.surfaces:
            raise refuse(f'gemini does not offer {name}')
        model = gemini.normalize_model(self.model_for(grant))
        body = strip_titanos(json_body(raw))
        if name == 'gemini.batchEmbedContents':
            body['requests'] = [dict(r, model='models/' + model) if isinstance(r, dict) else r
                                for r in body.get('requests') or []]
        if name == 'gemini.countTokens':
            for key in ('generateContentRequest', 'generate_content_request'):
                if isinstance(body.get(key), dict):
                    body[key] = dict(body[key], model='models/' + model)
        path = self.profile.surfaces[name]
        stream = name == 'gemini.streamGenerateContent'
        sse = stream and (raw.query or {}).get('alt') == 'sse'
        if stream and not sse:
            path = path.split('?', 1)[0]
        # a generateContent answer may carry whole images or audio as base64: the media cap, not the JSON cap
        response = await self.send(grant, egress, 'POST', path, path_params={'model': model}, json=body,
                                   headers=raw.headers, stream=stream,
                                   **({'media_cap': True} if name == 'gemini.generateContent' else {}))
        if sse:
            return await self.stream_result(response, 'gemini')
        if stream:
            return await self.stream_result(response, 'binary')
        if name in ('gemini.embedContent', 'gemini.batchEmbedContents'):  # no counts in the answer: estimate
            return await self.body_result(response, surface, usage=gemini.estimate_embed_usage(gemini.embed_texts(body)))
        return await self.body_result(response, surface)

    def _path(self, name: str) -> str:
        if name not in self.profile.surfaces:
            raise refuse(f'gemini does not offer {name}')
        return self.profile.surfaces[name]

    # translated chat and embeddings --------------------------------------------------------------------------------------
    async def invoke(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                     inject: dict | None = None):
        model = gemini.normalize_model(self.model_for(grant, ir.model))
        if ir.kind == 'embed':
            batch = len(ir.inputs or []) > 1
            name = 'gemini.batchEmbedContents' if batch else 'gemini.embedContent'
            request = IRRequest(**{**{f: getattr(ir, f) for f in ir.__dataclass_fields__}, 'model': model})
            response = await self.send(grant, egress, 'POST', self._path(name), path_params={'model': model},
                                       json=gemini.embed_request(request, batch=batch))
            decoded = gemini.embed_response(self.parse_json(response), batch=batch, inputs=ir.inputs)
            return IRResponse(id='', model=model, usage=decoded['usage'], extras={'vectors': decoded['vectors']})
        if ir.kind != 'chat':
            raise refuse(f'{ir.kind} requests are not available from gemini')
        body = gemini.encode_request(ir, self.profile, notes=notes, model=model)
        if ir.stream:
            response = await self.send(grant, egress, 'POST', self._path('gemini.streamGenerateContent'),
                                       path_params={'model': model}, json=body, stream=True)
            return gemini.decode_stream(await self.sse_frames(response, 'gemini'), notes=notes)
        response = await self.send(grant, egress, 'POST', self._path('gemini.generateContent'),
                                   path_params={'model': model}, json=body, media_cap=True)
        return gemini.decode_response(self.parse_json(response), notes=notes)

    def invoke_sync(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                    inject: dict | None = None) -> IRResponse:
        model = gemini.normalize_model(self.model_for(grant, ir.model))
        if ir.kind == 'embed':
            batch = len(ir.inputs or []) > 1
            name = 'gemini.batchEmbedContents' if batch else 'gemini.embedContent'
            request = IRRequest(**{**{f: getattr(ir, f) for f in ir.__dataclass_fields__}, 'model': model})
            response = self.send_sync(grant, egress, 'POST', self._path(name), path_params={'model': model},
                                      json=gemini.embed_request(request, batch=batch))
            decoded = gemini.embed_response(self.parse_json(response), batch=batch, inputs=ir.inputs)
            return IRResponse(id='', model=model, usage=decoded['usage'], extras={'vectors': decoded['vectors']})
        if ir.kind != 'chat':
            raise refuse(f'{ir.kind} requests are not available from gemini')
        body = gemini.encode_request(ir, self.profile, notes=notes, model=model)
        response = self.send_sync(grant, egress, 'POST', self._path('gemini.generateContent'),
                                  path_params={'model': model}, json=body)
        return gemini.decode_response(self.parse_json(response), notes=notes)

    # typed media bridges -------------------------------------------------------------------------------------------------
    @staticmethod
    def media_surfaces() -> frozenset[str]:
        """The surfaces inbound.translated_call hands to media_call."""
        return MEDIA_SURFACES

    def _media_family(self, name: str, model: str) -> None:
        """Refuse before egress when the model's family cannot serve the media surface."""
        family = gemini.model_family(model)
        if name == 'openai.audio.speech' and family != 'tts':
            raise refuse(f'{model} is not a Gemini text-to-speech model')
        if name in ('openai.images.generations', 'openai.images.edits') and family != 'image':
            raise refuse(f'{model} is not a Gemini image model (Imagen is reachable through the native :predict path)')
        if name in ('openai.audio.transcriptions', 'openai.audio.translations') and family != 'gemini':
            raise refuse(f'{model} does not transcribe audio')

    def media_body(self, name: str, document: dict | None, form: list | None, model: str,
                   notes: list | None) -> tuple[dict, dict]:
        """(generateContent body, context) for one media surface; refusals happen here, before egress."""
        self._media_family(name, model)
        if name == 'openai.audio.speech':
            body, kind = gemini_media.speech_request(document or {}, notes=notes)
            return body, {'kind': kind, 'characters': len((document or {}).get('input') or '')}
        if name == 'openai.images.generations':
            return gemini_media.image_request(document or {}, notes=notes), {}
        if name == 'openai.images.edits':
            fields: dict = {}
            images = []
            for key, value in form or []:
                if isinstance(value, tuple):
                    if key in ('image', 'image[]'):
                        import mimetypes
                        filename, data = value[0], value[1]
                        mime = str(value[2]) if len(value) > 2 and str(value[2]).startswith('image/') else \
                            (mimetypes.guess_type(filename or '')[0] or 'image/png')
                        images.append((mime if mime.startswith('image/') else 'image/png', data))
                    elif key == 'mask':
                        fields['mask'] = True
                    continue
                fields.setdefault(key, value)
            return gemini_media.image_request(fields, images=images, notes=notes, edit=True), {}
        translate = name == 'openai.audio.translations'
        body, fmt, data = gemini_media.transcription_request(list(form or []), translate=translate, notes=notes)
        language = next((v for k, v in form or [] if k == 'language' and isinstance(v, str)), None)
        return body, {'format': fmt, 'audio': data, 'language': language, 'translate': translate}

    def media_answer(self, name: str, document, context: dict):
        """(status, content type, body bytes, Usage) of the OpenAI answer from the generateContent answer."""
        from ..limits import measure_audio_seconds
        from ..translate import audio as audio_codec
        if name == 'openai.audio.speech':
            pcm, rate = gemini_media.speech_audio(document)
            usage = gemini_media.usage_of(document, characters=context.get('characters'),
                                          audio_seconds=gemini_media.pcm_seconds(pcm, rate))
            if context.get('kind') == 'pcm':
                return 200, 'audio/pcm', pcm, usage
            return 200, 'audio/wav', audio_codec.pcm_to_wav(pcm, rate), usage
        if name in ('openai.images.generations', 'openai.images.edits'):
            images, _ = gemini_media.image_outputs(document)
            usage = gemini_media.usage_of(document, images=len(images))
            out = gemini_media.images_response(document, usage=usage)
            return 200, 'application/json', json.dumps(out, separators=(',', ':')).encode('utf-8'), usage
        text = gemini_media.transcription_text(document)
        duration = measure_audio_seconds(context.get('audio') or b'')
        usage = gemini_media.usage_of(document, audio_seconds=duration)
        out = gemini_media.transcription_response(text, context['format'], language=context.get('language'),
                                                  duration=duration, translate=context.get('translate', False))
        if isinstance(out, str):
            return 200, 'text/plain; charset=utf-8', out.encode('utf-8'), usage
        return 200, 'application/json', json.dumps(out, ensure_ascii=False, separators=(',', ':')).encode('utf-8'), usage

    async def media_call(self, grant: EgressGrant, egress, name: str, document: dict | None, form: list | None, *,
                         notes: list | None = None):
        """inbound.translated_call for MEDIA_SURFACES: refuse before egress, one generateContent call, the OpenAI answer
        as an inbound.Produced (never streamed: Gemini returns the whole media in one answer)."""
        from ..inbound import Produced
        model = gemini.normalize_model(self.model_for(grant))
        body, context = self.media_body(name, document, form, model, notes)
        response = await self.send(grant, egress, 'POST', self._path('gemini.generateContent'),
                                   path_params={'model': model}, json=body, media_cap=True)
        status, content_type, data, usage = self.media_answer(name, self.parse_json(response), context)

        async def once():
            yield data

        async def nothing():
            return None
        from .common import request_id
        return Produced(status, content_type, once(), False, lambda: usage, nothing,
                        request_id(self.profile, response.headers))

    def media_sync(self, grant: EgressGrant, send, name: str, document: dict | None, form: list | None, *,
                   notes: list | None = None):
        """The synchronous form for CloudInvoker (jobs and API Plus): send(method, path, **kw) -> (response, bytes) is
        the stage's own cancellable sender. Returns (status, content type, body bytes, Usage)."""
        model = gemini.normalize_model(self.model_for(grant))
        body, context = self.media_body(name, document, form, model, notes)
        _, payload = send('POST', self._path('gemini.generateContent'), path_params={'model': model}, json=body)
        try:
            document_out = json.loads(payload.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise bad_upstream('The upstream answered with a body that is not JSON') from None
        return self.media_answer(name, document_out, context)

    # catalogs --------------------------------------------------------------------------------------------------------
    async def list_models_meta(self, grant: EgressGrant, egress) -> tuple[list[str], dict]:
        """(model ids, {model id: public metadata}) from the paginated GET models."""
        out, meta, token = [], {}, None
        for _ in range(CATALOG_MAX_PAGES):
            params = {'pageSize': '1000'}
            if token:
                params[self.profile.extras.get('catalog_page_token', 'pageToken')] = token
            response = await self.send(grant, egress, 'GET', self.profile.catalog_path, params=params)
            page = self.parse_json(response)
            if not isinstance(page, dict) or not isinstance(page.get('models', []), list):
                raise bad_upstream('The upstream catalog has no models list')
            for row in page.get('models') or []:
                if not isinstance(row, dict) or not isinstance(row.get('name'), str):
                    continue
                model = gemini.normalize_model(row['name'])
                out.append(model)
                info = {key: row[key] for key in _META_KEYS if key in row}
                if isinstance(row.get('inputTokenLimit'), int):
                    info['context_length'] = row['inputTokenLimit']
                meta[model] = info
            token = page.get('nextPageToken')
            if not token:
                break
        return out, meta

    async def list_models(self, grant: EgressGrant, egress) -> list[str]:
        return (await self.list_models_meta(grant, egress))[0]

    @staticmethod
    def model_capabilities(model: str | None, caps: frozenset, meta: dict | None = None) -> frozenset:
        """The capabilities one Gemini model offers out of the upstream-wide set (see the module docstring). A model
        whose published supportedGenerationMethods lack generateContent loses chat; one without embedContent loses
        embeddings."""
        family = gemini.model_family(model)
        if family in _FAMILY_CAPS:
            narrowed = set(_FAMILY_CAPS[family]) & set(caps)
            if family in ('tts', 'image'):
                narrowed |= {'chat'} & set(caps)
        else:
            narrowed = set(caps) - _MEDIA_CAPS - {'embeddings'}
            if family == 'gemma':
                narrowed -= {'transcription'}
        methods = (meta or {}).get('supportedGenerationMethods')
        if isinstance(methods, list) and methods:
            if 'generateContent' not in methods:
                narrowed -= {'chat', 'tools', 'vision', 'responses', 'transcription', 'speech', 'images'}
            if 'embedContent' not in methods and 'batchEmbedContents' not in methods:
                narrowed -= {'embeddings'}
        if 'chat' not in narrowed:
            narrowed -= {'tools', 'vision', 'responses'}
        return frozenset(narrowed)

    def usage_from_body(self, surface: Surface, body, headers) -> Usage | None:
        if not isinstance(body, dict):
            return None
        if surface.name in ('gemini.embedContent', 'gemini.batchEmbedContents'):
            return Usage(estimated=True)  # passthrough estimates from the request (see passthrough)
        if surface.name == 'gemini.countTokens':
            return Usage(input_tokens=0, output_tokens=0)  # token counting is not billed
        return gemini_usage(body.get('usageMetadata'))


__all__ = ['GeminiAdapter', 'MEDIA_SURFACES', 'TRANSLATED']
