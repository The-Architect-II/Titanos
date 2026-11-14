"""ElevenLabs adapter (xi-api-key, set by EgressClient).

- text-to-speech POST /v1/text-to-speech/{voice_id} and /stream, output_format as a query parameter (default
  mp3_44100_128, every format passed through; optimize_streaming_latency and enable_logging too); every body field is
  forwarded (only Titanos' worker, location and timeout are stripped); audio is binary passthrough (the first chunk is
  held for the commit). The provider's history-item-id, x-character-count and concurrency headers are returned (profile
  extras response_headers).
- speech-to-text POST /v1/speech-to-text (multipart: every form field, repeated fields kept; the enable_logging query
  parameter; never a single-use token in the query). A webhook=true request answers 202 with the provider's JSON.
- GET /v1/voices and /v1/voices/{voice_id}; GET /v1/models for the catalog (plus extras catalog_extra_models, the scribe
  speech-to-text models GET /v1/models does not list). The ElevenLabs SDK's own GET /v1/models (xi-api-key, no
  Authorization) reaches the provider list through the native passthrough row (vendor_over_listed).
- every other public endpoint (timestamps, dialogue, sound effects, voice changer, isolation, alignment, music, voice
  search, design and library, history, dubbing, dictionaries, the WebSockets) is a native passthrough row
  (profiles._ELEVENLABS_NATIVE); native_usage() reads its character-cost header so a budget settles at the real cost.
Usage: the character-cost response header when present, otherwise the request's character count (estimated).
Speech-to-text usage: audio_duration_secs when the provider reports it (multichannel answers too), else the last word's
end (estimated).
Probe and validation: a 401 whose code is not a key rejection (missing_permissions on a restricted-scope key,
quota_exceeded) still proves the key is genuine, so the credential validates (probe code key_restricted or
quota_exceeded) and is never marked failed (translate.upstream_errors.ELEVENLABS_KEY_REJECTED).
"""
from __future__ import annotations

import time

import httpx

from ..errors import BrokerError
from ..surfaces import Surface
from ..translate.common import refuse
from ..translate.ir import Audio, IRRequest, IRResponse, Text
from ..translate.upstream_errors import ELEVENLABS_KEY_REJECTED, broker_code, elevenlabs_code, network_error
from ..types import EgressGrant, Usage
from .base import ProbeResult, RawRequest, UpstreamResult
from .common import AdapterBase, json_body, split_form, strip_titanos

TRANSLATED = frozenset({'openai.audio.speech', 'openai.audio.transcriptions'})
TTS_QUERY = frozenset({'output_format', 'optimize_streaming_latency', 'enable_logging'})
# speech-to-text query parameters forwarded (the 'token' parameter is a single-use credential and never forwarded)
STT_QUERY = frozenset({'enable_logging'})


def character_usage(headers, text: str | None) -> Usage:
    value = headers.get('character-cost') if headers is not None else None
    try:
        return Usage(characters=int(value), estimated=False)
    except (TypeError, ValueError):
        return Usage(characters=len(text or ''), estimated=True)


class ElevenLabsAdapter(AdapterBase):
    adapter_name = 'elevenlabs'
    native_protocol = 'elevenlabs'

    def supports(self, profile, surface: Surface) -> bool:
        return surface.name in profile.surfaces or surface.name in TRANSLATED

    async def passthrough(self, grant: EgressGrant, egress, surface: Surface, raw: RawRequest, *,
                          notes: list | None = None, inject: dict | None = None,
                          allow_store: bool = False) -> UpstreamResult:
        name = surface.name
        if name not in self.profile.surfaces:
            raise refuse(f'elevenlabs does not offer {name}')
        path = self.profile.surfaces[name]
        params = {k: v for k, v in (raw.path_params or {}).items() if '{' + k + '}' in path}
        if name in ('elevenlabs.voices', 'elevenlabs.voice'):
            response = await self.send(grant, egress, 'GET', path, path_params=params,
                                       params=dict(raw.query or {}) or None)
            return await self.body_result(response, surface, usage=Usage())
        model = self.model_for(grant)
        if name == 'elevenlabs.stt':
            fields = [(k, model if k == 'model_id' else v) for k, v in raw.form or []]
            data, files = split_form(fields)
            query = {k: v for k, v in (raw.query or {}).items() if k in STT_QUERY}
            response = await self.send(grant, egress, 'POST', path, data=data, files=files, params=query or None)
            return await self.body_result(response, surface)
        body = strip_titanos(json_body(raw))
        body['model_id'] = model
        query = {k: v for k, v in (raw.query or {}).items() if k in TTS_QUERY}
        query.setdefault('output_format', self.profile.extras.get('default_output_format', 'mp3_44100_128'))
        return await self.tts(grant, egress, params.get('voice_id', ''), body, query,
                              stream=name == 'elevenlabs.tts_stream')

    async def tts(self, grant: EgressGrant, egress, voice_id: str, body: dict, query: dict, *,
                  stream: bool) -> UpstreamResult:
        """Text-to-speech (binary audio, streamed through with the first chunk held)."""
        if not voice_id:
            raise refuse('voice_id is required', 'invalid_request')
        name = 'elevenlabs.tts_stream' if stream else 'elevenlabs.tts'
        model = self.model_for(grant, body.get('model_id'))
        body = dict(body, model_id=model)
        response = await self.send(grant, egress, 'POST', self.profile.surfaces[name], path_params={'voice_id': voice_id},
                                   params=query or None, json=body, stream=True)
        return await self.stream_result(response, 'binary', usage=character_usage(response.headers, body.get('text')))

    async def stt(self, grant: EgressGrant, egress, form: list) -> dict:
        """Speech-to-text with ElevenLabs form fields (model_id rewritten to the upstream model)."""
        model = self.model_for(grant, next((v for k, v in form if k == 'model_id'), None))
        data, files = split_form([(k, model if k == 'model_id' else v) for k, v in form])
        response = await self.send(grant, egress, 'POST', self.profile.surfaces['elevenlabs.stt'], data=data,
                                   files=files)
        return self.parse_json(response)

    async def invoke(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                     inject: dict | None = None):
        if ir.kind == 'speech':
            text = ''.join(p.text for m in ir.messages for p in m.parts if isinstance(p, Text))
            body = {'text': text}
            if ir.sampling.get('speed') is not None:
                body['voice_settings'] = {'speed': ir.sampling['speed']}
            result = await self.tts(grant, egress, ir.voice or '', body,
                                    {'output_format': ir.audio_format or 'mp3_44100_128'}, stream=False)
            chunks = [chunk async for chunk in result.body]
            await result.aclose()
            return IRResponse(id='', model=self.model_for(grant), content=[Audio(result.content_type, b''.join(chunks))],
                              usage=await result.usage)
        if ir.kind == 'transcribe':
            form = [('model_id', self.model_for(grant, ir.model)),
                    ('file', ('audio', ir.audio or b'', ir.audio_format or 'application/octet-stream'))]
            answer = await self.stt(grant, egress, form)
            return IRResponse(id='', model=self.model_for(grant), content=[Text(answer.get('text') or '')],
                              extras={'body': answer})
        raise refuse(f'{ir.kind} requests are not available from elevenlabs')

    def invoke_sync(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                    inject: dict | None = None) -> IRResponse:
        model = self.model_for(grant, ir.model)
        if ir.kind == 'speech':
            text = ''.join(p.text for m in ir.messages for p in m.parts if isinstance(p, Text))
            body = {'text': text, 'model_id': model}
            if ir.sampling.get('speed') is not None:
                body['voice_settings'] = {'speed': ir.sampling['speed']}
            response = self.send_sync(grant, egress, 'POST', self.profile.surfaces['elevenlabs.tts'],
                                      path_params={'voice_id': ir.voice or ''},
                                      params={'output_format': ir.audio_format or 'mp3_44100_128'}, json=body)
            return IRResponse(id='', model=model,
                              content=[Audio(response.headers.get('content-type', 'audio/mpeg'), response.content)],
                              usage=character_usage(response.headers, text))
        if ir.kind == 'transcribe':
            data, files = split_form([('model_id', model),
                                      ('file', ('audio', ir.audio or b'', ir.audio_format or 'application/octet-stream'))])
            response = self.send_sync(grant, egress, 'POST', self.profile.surfaces['elevenlabs.stt'], data=data,
                                      files=files)
            answer = self.parse_json(response)
            return IRResponse(id='', model=model, content=[Text(answer.get('text') or '')], extras={'body': answer})
        raise refuse(f'{ir.kind} requests are not available from elevenlabs')

    async def list_models(self, grant: EgressGrant, egress) -> list[str]:
        response = await self.send(grant, egress, 'GET', self.profile.catalog_path)
        rows = self.parse_json(response)
        rows = rows.get('models') if isinstance(rows, dict) else rows
        models = [row['model_id'] for row in rows or [] if isinstance(row, dict) and isinstance(row.get('model_id'), str)]
        # usable models the listing omits (the scribe speech-to-text models), after the provider's own rows
        models.extend(m for m in self.profile.extras.get('catalog_extra_models') or () if m not in models)
        return models

    async def probe(self, grant: EgressGrant, egress) -> ProbeResult:
        """GET /v1/models (purpose validate, probe or test). A 401 that does not reject the key itself (a restricted-scope
        key without models_read answers missing_permissions; an exhausted account quota_exceeded) still proves the key
        is genuine: ok, code key_restricted or quota_exceeded."""
        path = self.profile.probe_path
        if not path:
            return ProbeResult(ok=True, status=None, latency_ms=None, code='skipped')
        start = time.perf_counter()
        try:
            response = await egress.send(grant, self.upstream, self.profile, 'GET', path)
        except httpx.HTTPError as exc:
            return ProbeResult(ok=False, status=None, latency_ms=None, code=network_error(exc).kind)
        except BrokerError as exc:
            return ProbeResult(ok=False, status=None, latency_ms=None, code=exc.code)
        latency = (time.perf_counter() - start) * 1000.0
        status = response.status_code
        if 200 <= status < 300:
            return ProbeResult(ok=True, status=status, latency_ms=latency)
        if status == 401:
            try:
                document = response.json()
            except ValueError:
                document = None
            detail = document.get('detail') if isinstance(document, dict) else None
            code = elevenlabs_code(detail) if isinstance(detail, dict) else None
            if code is not None and code not in ELEVENLABS_KEY_REJECTED:
                return ProbeResult(ok=True, status=status, latency_ms=latency,
                                   code='quota_exceeded' if 'quota' in code or 'credit' in code else 'key_restricted')
        err = self.map_error(status, response.headers, response.content)
        return ProbeResult(ok=False, status=status, latency_ms=latency, code=broker_code(err)[0])

    @staticmethod
    def native_usage(route, headers) -> Usage | None:
        """Usage of a native passthrough answer (dialogue, sound effects, voice changer, timestamps ...): the provider's
        character-cost header (its credit charge), or None (the reservation stands)."""
        value = headers.get('character-cost') if headers is not None else None
        try:
            return Usage(characters=int(value), estimated=False) if value is not None else None
        except (TypeError, ValueError):
            return None

    async def list_voices(self, grant: EgressGrant, egress) -> list[dict]:
        """The upstream voice library (GET /v1/voices, purpose catalog): the voice objects as the provider sends them."""
        response = await self.send(grant, egress, 'GET', self.profile.surfaces['elevenlabs.voices'])
        rows = self.parse_json(response)
        rows = rows.get('voices') if isinstance(rows, dict) else rows
        return [row for row in rows or [] if isinstance(row, dict) and isinstance(row.get('voice_id'), str)
                and row['voice_id']]

    @staticmethod
    def model_capabilities(model: str | None, caps: frozenset, meta: dict | None = None) -> frozenset:
        """The capabilities one ElevenLabs model offers out of the upstream-wide set: scribe_* models transcribe, every
        other eleven_* model speaks; an unknown family keeps the upstream's set (never refused for a new name)."""
        name = (model or '').lower()
        if name.startswith('scribe'):
            return frozenset(caps) - {'speech'}
        if name.startswith('eleven_'):
            return frozenset(caps) - {'transcription', 'translation'}
        return frozenset(caps)

    def usage_from_body(self, surface: Surface, body, headers) -> Usage | None:
        if surface.name == 'elevenlabs.stt' and isinstance(body, dict):
            return stt_usage(body)
        return None


def stt_usage(body: dict) -> Usage | None:
    """Speech-to-text usage: audio_duration_secs (single and multichannel answers) when reported, else the latest word
    end over the transcript or every channel's transcripts[] (estimated); None for a 202 webhook acknowledgement."""
    duration = body.get('audio_duration_secs')
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0:
        return Usage(audio_seconds=float(duration), estimated=False)
    channels = [body] + [t for t in body.get('transcripts') or [] if isinstance(t, dict)]
    ends = []
    for channel in channels:
        for word in channel.get('words') or []:
            if isinstance(word, dict) and isinstance(word.get('end'), (int, float)) and not isinstance(word['end'], bool):
                ends.append(float(word['end']))
    if not ends and 'transcription_id' in body and 'text' not in body and not body.get('transcripts'):
        return None  # webhook acknowledgement: the transcript is delivered later
    return Usage(audio_seconds=max(ends, default=None), estimated=True)


__all__ = ['ElevenLabsAdapter', 'STT_QUERY', 'TRANSLATED', 'character_usage', 'stt_usage']
