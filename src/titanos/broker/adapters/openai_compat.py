"""One adapter for the whole OpenAI-compatible family (OpenAI, Azure OpenAI, xAI, Groq, Together, Mistral, DeepSeek,
Fireworks, Perplexity, Cerebras, SambaNova, OpenRouter, the Hugging Face router, LiteLLM, HiveMind and custom endpoints),
driven by profile rows.

- Paths: profile.surfaces[surface] relative to the upstream base_url (upstream.surfaces may narrow operator rows). Azure:
  the model maps to a deployment through upstream.azure.deployments (unknown model: 404 model_not_found); EgressClient adds
  api-version and the api-key header.
- rewrite_body: the model is set to the upstream model; Titanos-only fields (worker, location, timeout) are stripped;
  published-alias inject defaults apply; the ParamMap refuses (400 before egress), renames, drops (noted), allow-lists and
  injects per model glob (reasoning models: max_tokens -> max_completion_tokens, temperature and top_p dropped; Mistral:
  seed -> random_seed; Groq: logprobs, top_logprobs, logit_bias and n>1 refused). Store is forced false on OpenAI
  Responses (and chat-completions store=true is turned off) unless the caller allows storing. Asymmetric embedding
  models (the row's asymmetric_embedding_globs) need input_type (client value or alias inject; never a silent default). Models matching the profile's
  image_b64_only_globs (OpenAI: every image model since the 2026-10-02 live observation; Azure OpenAI: gpt-image-*
  deployments, as the Azure mock answers; they always answer b64_json and refuse the field) never receive response_format on images.generations or images.edits (JSON body or multipart form):
  any client value is dropped and noted (dispatch re-hosts the base64 answer when the client asked for url).
- Azure OpenAI reasoning deployments (o*, gpt-5*) reject max_tokens, temperature and top_p exactly as OpenAI does, but the
  azure_openai profile row has no ParamMap of its own: its rename and drop rules are borrowed from the openai row
  (PARAMS_FROM), keyed on the upstream model name that maps to the deployment.
- tool_id_rule 'alnum9' (Mistral): foreign tool-call ids are remapped deterministically to 9 alphanumerics in requests and
  restored in non-streamed answers (ToolIdMap).
- include_usage: stream_options.include_usage=true is injected on chat and completions streams for profiles that report
  usage that way; the injected final usage chunk is stripped downstream when the client did not ask for it.
- Native response extensions (citations, search_results, reasoning_content, provider fields) pass through unchanged on
  passthrough paths.
- OpenAI media streams (sse 'openai_media'): images/generations and images/edits with stream=true, audio/transcriptions
  with stream=true (form field) and audio/speech with stream_format=sse are sent with egress stream=True and relayed
  through the held first frame and the usage tee (image, transcription and speech token usage); a non-stream answer to a
  stream request (whisper-1 ignores the flag) is returned whole. GET /responses/{id}?stream=true resumes the event
  stream. Image JSON answers use the media cap (n base64 images). Titanos' VAD gate form fields (vad_enabled, vad_model,
  vad_options) are dropped with a note on cloud transcription. The reasoning-model temperature/top_p drop is lifted by the
  row's sampling_kept_globs and sampling_kept_effort (OpenAI: gpt-5-chat*, reasoning effort 'none'; Azure deployments
  through PARAMS_FROM). Usage: chat audio tokens, image and audio input tokens, and Responses web_search_call items.
- Catalog: {data: [...]} or a bare JSON array; static (upstream.models) for rows without a models endpoint.
- Family rules read from profile extras (the comment above the xai row in profiles.py lists them): reasoning_drop on
  reasoning models (xAI), reasoning_effort_globs (a translated or advisor reasoning_effort reaches only those models),
  stores_responses (store forced false unless the caller may store), image_response_format_values (Together base64),
  reasoning_content_roundtrip (DeepSeek: translated assistant thinking is sent back as reasoning_content).
"""
from __future__ import annotations

import hashlib
import json
import re

from ..errors import BrokerError
from ..profiles import image_b64_only
from ..sse import openai_media_usage, openai_usage, responses_object_usage
from ..surfaces import Surface
from ..translate import openai_chat
from ..translate.common import refuse
from ..translate.ir import Audio, IRRequest, IRResponse, Text
from ..types import EgressGrant, Usage
from .base import RawRequest, UpstreamResult
from .common import (AdapterBase, json_body, matches, split_form, strip_titanos, surface as surface_row)

ALNUM9 = re.compile(r'^[A-Za-z0-9]{9}$')
_BASE62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
STREAM_PROTOCOL = {'openai.chat': 'openai', 'openai.completions': 'openai', 'openai.responses': 'responses'}
INCLUDE_USAGE_SURFACES = frozenset({'openai.chat', 'openai.completions'})
STORE_PROFILES = frozenset({'openai', 'azure_openai'})  # plus every row whose extras set stores_responses
# rows that serve the same models but carry no ParamMap of their own
PARAMS_FROM = {'azure_openai': 'openai', 'azure_openai_v1': 'openai'}
CLOUD_REFUSED_FORM_FIELDS = frozenset({'translation_model', 'source_language', 'target_language'})
# Titanos' local VAD gate fields (audio_gating.GATE_FIELDS) on transcription forms: meaningless to a provider, so they
# are dropped with a note (X-Titanos-Dropped) instead of reaching it as unknown parameters
CLOUD_DROPPED_FORM_FIELDS = frozenset({'vad_enabled', 'vad_model', 'vad_options'})
# OpenAI media event streams (sse 'openai_media'): images and image edits with stream=true, transcriptions with
# stream=true (whisper-1 ignores the flag and answers JSON), speech with stream_format=sse (detected by content type)
MEDIA_STREAM_SURFACES = frozenset({'openai.images.generations', 'openai.images.edits', 'openai.audio.transcriptions'})
# large JSON answers (n base64 images) are capped by the media cap, not the JSON cap
MEDIA_JSON_SURFACES = frozenset({'openai.images.generations', 'openai.images.edits'})
SAMPLING_FIELDS = frozenset({'temperature', 'top_p', 'logprobs', 'top_logprobs'})
TRANSLATED_SURFACES = {
    # inbound surface -> native surface it is translated onto
    'openai.completions': 'openai.chat', 'openai.responses': 'openai.chat', 'anthropic.messages': 'openai.chat',
    'gemini.generateContent': 'openai.chat', 'gemini.streamGenerateContent': 'openai.chat', 'ollama.chat': 'openai.chat',
    'ollama.generate': 'openai.chat', 'gemini.embedContent': 'openai.embeddings',
    'gemini.batchEmbedContents': 'openai.embeddings', 'ollama.embed': 'openai.embeddings',
    'ollama.embeddings': 'openai.embeddings', 'elevenlabs.tts': 'openai.audio.speech',
    'elevenlabs.tts_stream': 'openai.audio.speech', 'elevenlabs.stt': 'openai.audio.transcriptions',
}


class ToolIdMap:
    """Deterministic per-request mapping of foreign tool-call ids to 9 alphanumerics (and back)."""

    def __init__(self):
        self.forward: dict[str, str] = {}
        self.backward: dict[str, str] = {}

    def to_upstream(self, value):
        if not isinstance(value, str) or ALNUM9.match(value):
            return value
        if value not in self.forward:
            number = int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest()[:8], 'big')
            digits = []
            for _ in range(9):
                number, rest = divmod(number, 62)
                digits.append(_BASE62[rest])
            mapped = ''.join(digits)
            self.forward[value], self.backward[mapped] = mapped, value
        return self.forward[value]

    def to_client(self, value):
        return self.backward.get(value, value) if isinstance(value, str) else value

    def rewrite_request(self, body: dict) -> dict:
        messages = []
        for message in body.get('messages') or []:
            if not isinstance(message, dict):
                messages.append(message)
                continue
            message = dict(message)
            if message.get('tool_calls'):
                message['tool_calls'] = [dict(call, id=self.to_upstream(call.get('id'))) if isinstance(call, dict)
                                         else call for call in message['tool_calls']]
            if message.get('role') == 'tool' and 'tool_call_id' in message:
                message['tool_call_id'] = self.to_upstream(message['tool_call_id'])
            messages.append(message)
        if 'messages' in body:
            body = dict(body, messages=messages)
        return body

    def restore(self, value):
        """Mapped ids put back in a response document (tool_calls[].id and tool_call_id)."""
        if not self.backward:
            return value
        if isinstance(value, list):
            return [self.restore(v) for v in value]
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if key in ('id', 'tool_call_id') and isinstance(item, str) and item in self.backward:
                    out[key] = self.backward[item]
                else:
                    out[key] = self.restore(item)
            return out
        return value


def _refused(body: dict, field: str) -> bool:
    if '>' in field:
        name, _, limit = field.partition('>')
        value = body.get(name)
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > float(limit)
    return body.get(field) not in (None, False, [], {})


def effective_params(profile):
    """The profile's ParamMap; for a row in PARAMS_FROM without rename/drop rules, the donor row's rename and drop
    rules (the row's own refuse/allow/inject/tool_id_rule stay)."""
    params = profile.params
    donor_id = PARAMS_FROM.get(profile.id)
    if donor_id is None or params.rename or params.drop:
        return params
    from dataclasses import replace
    from ..profiles import PROFILES
    donor = PROFILES[donor_id].params
    return replace(params, rename=donor.rename, drop=donor.drop)


def _row_extras(profile):
    """The row's extras, or for a PARAMS_FROM row (Azure OpenAI deployments of OpenAI models) the donor row's extras."""
    donor_id = PARAMS_FROM.get(profile.id)
    if donor_id is None:
        return profile.extras
    from ..profiles import PROFILES
    return PROFILES[donor_id].extras


def sampling_kept(profile, model: str | None, body: dict) -> bool:
    """Whether a reasoning-model sampling drop (temperature, top_p, logprobs, top_logprobs) is lifted for this request:
    the model matches the row's sampling_kept_globs (OpenAI gpt-5-chat*: chat models that accept sampling), or the row
    sets sampling_kept_effort and the request asks for that reasoning effort (OpenAI GPT-5.1 and later accept sampling
    parameters with reasoning effort 'none'; the provider answers 400 itself for a model that has no such effort)."""
    extras = _row_extras(profile)
    if any(matches(model or '', glob) for glob in extras.get('sampling_kept_globs') or ()):
        return True
    effort = extras.get('sampling_kept_effort')
    if effort is None:
        return False
    reasoning = body.get('reasoning') if isinstance(body.get('reasoning'), dict) else {}
    return body.get('reasoning_effort') == effort or reasoning.get('effort') == effort


def stores_responses(profile) -> bool:
    """Whether the provider stores Responses (and stored chat completions) by default: store is then forced false unless
    the caller may store (R1)."""
    return profile.id in STORE_PROFILES or bool(profile.extras.get('stores_responses'))


def is_reasoning_model(profile, model: str | None) -> bool:
    """A reasoning model of a row that declares reasoning_globs (and is not excluded by non_reasoning_globs)."""
    globs = profile.extras.get('reasoning_globs') or ()
    if not any(matches(model or '', glob) for glob in globs):
        return False
    return not any(matches(model or '', glob) for glob in profile.extras.get('non_reasoning_globs') or ())


def accepts_reasoning_effort(profile, model: str | None) -> bool:
    """Whether a Titanos-added reasoning_effort may be sent to this model: the row declares the 'reasoning' feature and,
    when it lists reasoning_effort_globs, the model matches one of them."""
    if 'reasoning' not in profile.features:
        return False
    globs = profile.extras.get('reasoning_effort_globs')
    return globs is None or any(matches(model or '', glob) for glob in globs)


def rewrite_body(profile, upstream, surface, body: dict, upstream_model: str, *, inject: dict | None = None,
                 allow_store: bool = False, id_map: ToolIdMap | None = None) -> tuple[dict, list[str]]:
    """The upstream request body for a passthrough or translated call, and the decision notes. Raises BrokerError 400
    broker_capability_unavailable for a ParamMap refusal or a missing input_type, before any egress."""
    name = surface.name if isinstance(surface, Surface) else str(surface)
    notes: list[str] = []
    out = strip_titanos(dict(body))
    out['model'] = upstream_model
    for key, value in (inject or {}).items():
        if key not in out:
            out[key] = value
            notes.append(f'injected:{key}')
    params = effective_params(profile)
    for glob, fields in params.refuse.items():
        if matches(upstream_model, glob):
            for field in fields:
                if _refused(out, field):
                    raise refuse(f'{field.partition(">")[0]} is not supported by {profile.id}')
    for glob, mapping in params.rename.items():
        if matches(upstream_model, glob):
            for source, target in mapping.items():
                if source in out:
                    value = out.pop(source)
                    if target not in out:
                        out[target] = value
                    notes.append(f'renamed:{source}->{target}')
    keep_sampling = sampling_kept(profile, upstream_model, out)
    for glob, fields in params.drop.items():
        if matches(upstream_model, glob):
            for field in fields:
                if field in out and not (keep_sampling and field in SAMPLING_FIELDS):
                    out.pop(field)
                    notes.append(f'dropped:{field}')
    if params.allow is not None:
        for field in [f for f in out if f not in params.allow and f != 'model']:
            out.pop(field)
            notes.append(f'dropped:{field}')
    for glob, values in params.inject.items():
        if matches(upstream_model, glob):
            for key, value in values.items():
                out.setdefault(key, value)
    if name in ('openai.chat', 'openai.completions', 'openai.responses') and is_reasoning_model(profile, upstream_model):
        for field in profile.extras.get('reasoning_drop') or ():
            if field in out:
                out.pop(field)
                notes.append(f'dropped:{field}')
    if (stores_responses(profile) or profile.extras.get('force_store_false')) and not allow_store:
        if name == 'openai.responses':
            for field in ('background', 'conversation'):
                if out.get(field) not in (None, False):
                    # a background response or a conversation lives at the provider: without storage it would be
                    # billed but could never be polled, cancelled or continued through Titanos
                    raise BrokerError(f'{field} keeps the response at the provider and needs a key holding cloud:store '
                                      'on an upstream with data_policy.allow_store', 'scope_denied', origin='policy')
        if name == 'openai.responses' and out.get('store', True) is not False:
            if out.get('store') is True:
                notes.append('dropped:store')  # the client asked for store: true (X-Titanos-Dropped names it)
            out['store'] = False
            notes.append('forced:store=false')
        elif name == 'openai.chat' and out.get('store'):
            out['store'] = False
            notes.append('dropped:store')
            notes.append('forced:store=false')
    if name == 'openai.embeddings':
        globs = profile.extras.get('asymmetric_embedding_globs') or ()
        if any(matches(upstream_model, glob) for glob in globs):
            allowed = tuple(profile.extras.get('input_type_values') or ('query', 'passage'))
            value = out.get('input_type')
            if value is None:
                raise refuse(f'input_type required for {upstream_model} (one of: {", ".join(allowed)})')
            if value not in allowed:
                raise refuse(f'input_type must be one of: {", ".join(allowed)}', 'invalid_request')
    if name in ('openai.images.generations', 'openai.images.edits') and 'response_format' in out \
            and image_b64_only(profile, upstream_model):
        out.pop('response_format')
        notes.append('dropped:response_format')
    values = profile.extras.get('image_response_format_values') or {}
    if name in ('openai.images.generations', 'openai.images.edits') and out.get('response_format') in values:
        out['response_format'] = values[out['response_format']]  # Together: b64_json -> base64
        notes.append('mapped:response_format')
    if params.tool_id_rule == 'alnum9' and name == 'openai.chat':
        mapper = id_map if id_map is not None else ToolIdMap()
        rewritten = mapper.rewrite_request(out)
        if mapper.forward:
            notes.append('remapped:tool_call_ids')
        out = rewritten
    return out, notes


def inject_include_usage(profile, body: dict) -> tuple[dict, bool]:
    """(body, strip_downstream): stream_options.include_usage is injected on streams of profiles that report usage that
    way; strip_downstream is True when the client did not ask for it (the final usage chunk is then removed)."""
    if not profile.include_usage or not body.get('stream'):
        return body, False
    options = body.get('stream_options') if isinstance(body.get('stream_options'), dict) else {}
    if options.get('include_usage'):
        return body, False
    return dict(body, stream_options=dict(options, include_usage=True)), True


class OpenAICompatAdapter(AdapterBase):
    adapter_name = 'openai_compat'
    native_protocol = 'openai'

    def supports(self, profile, surface: Surface) -> bool:
        name = surface.name
        if name in self.native_surfaces() and name in profile.surfaces:
            return True
        native = TRANSLATED_SURFACES.get(name)
        return native is not None and native in self.native_surfaces() and native in profile.surfaces

    # routing -----------------------------------------------------------------------------------------------------------
    def route(self, name: str, model: str | None, path_params: dict | None = None) -> tuple[str, dict]:
        if not self.has(name):
            raise refuse(f'{self.profile.id} does not offer {name}')
        params = {k: v for k, v in (path_params or {}).items() if '{' + k + '}' in self.profile.surfaces[name]}
        if self.profile.id == 'azure_openai':
            azure = getattr(self.upstream, 'azure', None)
            deployments = getattr(azure, 'deployments', None) or {}
            if model not in deployments:
                raise BrokerError(f'Model {model!r} has no Azure deployment on this upstream', 'model_not_found')
            params['deployment'] = deployments[model]
        return self.profile.surfaces[name], params

    # passthrough ---------------------------------------------------------------------------------------------------------
    async def passthrough(self, grant: EgressGrant, egress, surface: Surface, raw: RawRequest, *,
                          notes: list | None = None, inject: dict | None = None,
                          allow_store: bool = False) -> UpstreamResult:
        name = surface.name
        if surface.handle_bound:  # retrieve / delete / cancel / input_items on the owning upstream
            path, params = self.route(name, None, raw.path_params)
            query = dict(raw.query or {})
            # GET /responses/{id}?stream=true resumes a background response's event stream (starting_after)
            stream = name == 'openai.responses.retrieve' and str(query.get('stream', '')).lower() in ('true', '1')
            response = await self.send(grant, egress, surface.method, path, path_params=params,
                                       params=query or None, headers=raw.headers, stream=stream)
            if stream:
                return await self._stream_or_body(response, surface, 'responses')
            return await self.body_result(response, surface)
        model = self.model_for(grant)
        path, params = self.route(name, model, raw.path_params)
        if surface.body == 'multipart':
            fields = []
            strip_format = name in ('openai.images.generations', 'openai.images.edits') and \
                image_b64_only(self.profile, model)
            stream = False
            for field, value in raw.form or []:
                if field in CLOUD_REFUSED_FORM_FIELDS:
                    raise refuse(f'{field} is a Titanos-only field and cannot be sent to a cloud target')
                if strip_format and field == 'response_format':
                    if notes is not None and 'dropped:response_format' not in notes:
                        notes.append('dropped:response_format')
                    continue
                if field in CLOUD_DROPPED_FORM_FIELDS:
                    if notes is not None and f'dropped:{field}' not in notes:
                        notes.append(f'dropped:{field}')
                    continue
                if field == 'stream' and name in MEDIA_STREAM_SURFACES and str(value).strip().lower() in ('true', '1'):
                    stream = True
                fields.append((field, model if field == 'model' else value))
            data, files = split_form(fields)
            response = await self.send(grant, egress, 'POST', path, path_params=params, data=data, files=files,
                                       headers=raw.headers, stream=stream,
                                       **({'media_cap': True} if name in MEDIA_JSON_SURFACES else {}))
            if stream:
                return await self._stream_or_body(response, surface, 'openai_media')
            return await self.body_result(response, surface)
        body = json_body(raw)
        id_map = ToolIdMap()
        body, rewrite_notes = rewrite_body(self.profile, self.upstream, surface, body, model, inject=inject,
                                           allow_store=allow_store, id_map=id_map)
        strip = False
        if name in INCLUDE_USAGE_SURFACES:
            body, strip = inject_include_usage(self.profile, body)
            if strip:
                rewrite_notes.append('injected:stream_options.include_usage')
        if notes is not None:
            notes.extend(n for n in rewrite_notes if n not in notes)
        if name == 'openai.audio.speech':
            characters = len(body.get('input') or '') if isinstance(body.get('input'), str) else None
            response = await self.send(grant, egress, 'POST', path, path_params=params, json=body,
                                       headers=raw.headers, stream=True)
            # stream_format=sse answers speech.audio.* events (token usage in speech.audio.done); otherwise raw audio
            protocol = 'openai_media' if 'event-stream' in response.headers.get('content-type', '') else 'binary'
            return await self.stream_result(response, protocol, usage=Usage(characters=characters, estimated=True))
        if name in MEDIA_STREAM_SURFACES and body.get('stream'):
            response = await self.send(grant, egress, 'POST', path, path_params=params, json=body,
                                       headers=raw.headers, stream=True, media_cap=True)
            return await self._stream_or_body(response, surface, 'openai_media')
        stream = bool(body.get('stream')) and name in STREAM_PROTOCOL
        response = await self.send(grant, egress, 'POST', path, path_params=params, json=body, headers=raw.headers,
                                   stream=stream, **({'media_cap': True} if name in MEDIA_JSON_SURFACES else {}))
        if stream:
            return await self.stream_result(response, STREAM_PROTOCOL[name], strip_usage=strip)
        if id_map.backward and 'json' in response.headers.get('content-type', ''):
            restored = id_map.restore(self.parse_json(response))
            content = json.dumps(restored, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            return await self.body_result(response, surface, content=content)
        return await self.body_result(response, surface)

    async def _stream_or_body(self, response, surface: Surface, protocol: str) -> UpstreamResult:
        """A streamed request's answer: an event stream goes through the held first frame and the usage tee (protocol
        'openai_media' or 'responses'); any other answer (whisper-1 ignores stream=true and answers JSON) is read within
        the cap and returned whole."""
        if 'event-stream' in response.headers.get('content-type', ''):
            return await self.stream_result(response, protocol)
        await response.aread()
        return await self.body_result(response, surface)

    # translated calls ----------------------------------------------------------------------------------------------------
    def _chat_body(self, ir: IRRequest, model: str, notes, *, inject=None) -> tuple[dict, str, dict]:
        body = openai_chat.encode_request(ir, self.profile, notes=notes)
        body, extra = rewrite_body(self.profile, self.upstream, surface_row('openai.chat'), body, model, inject=inject)
        if 'reasoning_effort' in body and not accepts_reasoning_effort(self.profile, model):
            body.pop('reasoning_effort')  # a translated reasoning request on a model that refuses the field
            extra.append('dropped:reasoning_effort')
        if notes is not None:
            notes.extend(n for n in extra if n not in notes)
        body, _ = inject_include_usage(self.profile, body)
        path, params = self.route('openai.chat', model)
        return body, path, params

    async def invoke(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                     inject: dict | None = None):
        model = self.model_for(grant, ir.model)
        if ir.kind == 'embed':
            return await self._embed(grant, egress, ir, model, notes, inject)
        if ir.kind == 'speech':
            body = self.speech_body(ir, model)
            path, params = self.route('openai.audio.speech', model)
            response = await self.send(grant, egress, 'POST', path, path_params=params, json=body)
            return IRResponse(id='', model=model, content=[Audio(response.headers.get('content-type', 'audio/mpeg'),
                                                                 response.content)],
                              usage=Usage(characters=len(body['input']), estimated=True))
        if ir.kind != 'chat':
            raise refuse(f'{ir.kind} requests are not translated for {self.profile.id}')
        body, path, params = self._chat_body(ir, model, notes, inject=inject)
        response = await self.send(grant, egress, 'POST', path, path_params=params, json=body, stream=ir.stream)
        if ir.stream:
            frames = await self.sse_frames(response, 'openai')
            return openai_chat.decode_stream(frames)
        return openai_chat.decode_response(self.parse_json(response))

    def invoke_sync(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                    inject: dict | None = None) -> IRResponse:
        model = self.model_for(grant, ir.model)
        if ir.kind == 'embed':
            body, path, params = self._embed_body(ir, model, notes, inject)
            response = self.send_sync(grant, egress, 'POST', path, path_params=params, json=body)
            return self._embed_response(self.parse_json(response), model)
        if ir.kind != 'chat':
            raise refuse(f'{ir.kind} requests are not translated for {self.profile.id} on worker threads')
        plain = IRRequest(**{**{f: getattr(ir, f) for f in ir.__dataclass_fields__}, 'stream': False})
        body, path, params = self._chat_body(plain, model, notes, inject=inject)
        response = self.send_sync(grant, egress, 'POST', path, path_params=params, json=body)
        return openai_chat.decode_response(self.parse_json(response))

    def _embed_body(self, ir: IRRequest, model: str, notes, inject) -> tuple[dict, str, dict]:
        body = openai_chat.encode_embed_request(ir)
        body, extra = rewrite_body(self.profile, self.upstream, surface_row('openai.embeddings'), body, model,
                                   inject=inject)
        if notes is not None:
            notes.extend(n for n in extra if n not in notes)
        path, params = self.route('openai.embeddings', model)
        return body, path, params

    @staticmethod
    def _embed_response(document: dict, model: str) -> IRResponse:
        decoded = openai_chat.decode_embed_response(document)
        return IRResponse(id='', model=model, usage=decoded['usage'] or Usage(),
                          extras={'vectors': decoded['vectors']})

    async def _embed(self, grant, egress, ir, model, notes, inject) -> IRResponse:
        body, path, params = self._embed_body(ir, model, notes, inject)
        response = await self.send(grant, egress, 'POST', path, path_params=params, json=body)
        return self._embed_response(self.parse_json(response), model)

    # audio (elevenlabs inbound -> OpenAI-compatible targets) ---------------------------------------------------------
    @staticmethod
    def speech_body(ir: IRRequest, model: str) -> dict:
        text = ''.join(p.text for m in ir.messages for p in m.parts if isinstance(p, Text))
        body = {'model': model, 'input': text, 'voice': ir.voice, 'response_format': ir.audio_format or 'mp3'}
        if ir.sampling.get('speed') is not None:
            body['speed'] = ir.sampling['speed']
        return body

    async def speech(self, grant: EgressGrant, egress, body: dict, *, stream: bool = True) -> UpstreamResult:
        """POST /audio/speech with an already-mapped OpenAI body (binary audio; streamed)."""
        model = self.model_for(grant, body.get('model'))
        body, _ = rewrite_body(self.profile, self.upstream, surface_row('openai.audio.speech'), body, model)
        path, params = self.route('openai.audio.speech', model)
        response = await self.send(grant, egress, 'POST', path, path_params=params, json=body, stream=stream)
        usage = Usage(characters=len(body.get('input') or ''), estimated=True)
        if stream:
            return await self.stream_result(response, 'binary', usage=usage)
        return await self.body_result(response, surface_row('openai.audio.speech'), usage=usage)

    async def transcribe(self, grant: EgressGrant, egress, form: list):
        """POST /audio/transcriptions with already-mapped OpenAI form fields; the parsed answer (dict, or str for text)."""
        model = self.model_for(grant, next((v for k, v in form if k == 'model'), None))
        path, params = self.route('openai.audio.transcriptions', model)
        data, files = split_form([(k, model if k == 'model' else v) for k, v in form])
        response = await self.send(grant, egress, 'POST', path, path_params=params, data=data, files=files)
        if 'json' in response.headers.get('content-type', ''):
            return self.parse_json(response)
        return response.content.decode('utf-8', 'replace')

    # catalogs ----------------------------------------------------------------------------------------------------------
    async def list_models(self, grant: EgressGrant, egress) -> list[str]:
        if not self.profile.catalog_path:
            return self.static_models()
        response = await self.send(grant, egress, 'GET', self.profile.catalog_path)
        return parse_catalog(self.parse_json(response))

    def usage_from_body(self, surface: Surface, body, headers) -> Usage | None:
        if not isinstance(body, dict):
            return None
        name = surface.name
        if name == 'openai.responses':
            return responses_object_usage(body)  # tokens plus hosted web_search_call items
        if name in ('openai.images.generations', 'openai.images.edits'):
            # gpt-image token usage with its text/image input split (image_input_tokens priced on their own)
            return openai_media_usage(body.get('usage'), images=len(body.get('data') or []))
        if name in ('openai.audio.transcriptions', 'openai.audio.translations'):
            usage = body.get('usage')
            if isinstance(usage, dict) and usage.get('type') == 'duration':
                return Usage(audio_seconds=usage.get('seconds'))
            if isinstance(usage, dict) and usage.get('prompt_audio_seconds') is not None:  # Mistral Voxtral
                return Usage(audio_seconds=usage.get('prompt_audio_seconds'), input_tokens=usage.get('prompt_tokens'),
                             output_tokens=usage.get('completion_tokens'))
            if isinstance(usage, dict) and 'input_tokens' in usage:  # OpenAI token usage (audio/text input split)
                return openai_media_usage(usage)
            if isinstance(usage, dict):  # chat-style prompt/completion names on other providers
                return Usage(input_tokens=usage.get('input_tokens', usage.get('prompt_tokens')),
                             output_tokens=usage.get('output_tokens', usage.get('completion_tokens')))
            return Usage(audio_seconds=body.get('duration')) if body.get('duration') is not None else None
        return openai_usage(body.get('usage'))


def parse_catalog(document) -> list[str]:
    """Model ids from {data: [...]} or a bare JSON array (of objects with id, or of strings)."""
    rows = document.get('data') if isinstance(document, dict) else document
    if not isinstance(rows, list):
        from ..translate.common import bad_upstream
        raise bad_upstream('The upstream catalog is neither a list nor {data: [...]}')
    out = []
    for row in rows:
        value = row.get('id') if isinstance(row, dict) else row
        if isinstance(value, str) and value:
            out.append(value)
    return out


__all__ = ['ALNUM9', 'CLOUD_REFUSED_FORM_FIELDS', 'INCLUDE_USAGE_SURFACES', 'OpenAICompatAdapter', 'ToolIdMap',
           'TRANSLATED_SURFACES', 'inject_include_usage', 'parse_catalog', 'rewrite_body']
