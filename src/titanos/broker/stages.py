"""Thread-side cloud reach for jobs and API Plus, and the versioned text-translation prompt.

- StageRouter.resolve_stage(name, task, *, location, principal) -> StageTarget. A local registry name keeps its ModelSpec;
  a cloud target gets a read-only CloudSpecView (task, backend='cloud', defaults={}, voices={}, device='remote'), so the
  existing spec.backend and spec.defaults reads in API Plus keep working. Placement follows the broker policy with the API
  Plus default location local_only (api_plus.py:44); local-only stage tasks (OCR, VAD, detection, ...) asked for
  cloud_only give 400 broker_capability_unavailable; unknown names 404 model_not_found.
- GatewayRouter(native, broker, principal, location): candidates() and complete() with NativePool.complete's signature
  (native.py:515) including require_loaded. Local models go to the NativePool unchanged; cloud models are validated by
  the router (policy, scopes) and completed through CloudInvoker on a worker thread. The return shape is NativePool's:
  {text, finish_reason, usage, native_response, provenance, native_runtime}.
- CloudInvoker.infer_sync(target, task, data, *, cancel, deadline, principal): synchronous egress on the caller's worker
  thread, never inside Engine.lock or the NativePool. speech -> (WAV bytes, 'audio/wav') (OpenAI response_format wav;
  ElevenLabs pcm_24000 wrapped in a WAV header); image -> [PNG bytes] (any non-PNG upstream output gives 400
  broker_capability_unavailable, and a profile that declares no PNG output is refused before egress); transcription -> an
  ASRMetadata-shaped dict; translation and chat -> {text, finish_reason, usage}; embeddings -> {data, usage}. The cancel
  object (anything with is_set()) is polled before the call and between body chunks; the deadline bounds every call
  (504 deadline_exceeded). Each call is admitted by the Limiter (budgets included) and settled with its usage.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import struct
import time
import uuid
import wave
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

from . import context as broker_context
from .adapters.common import split_form
from .errors import BrokerError
from .limits import Estimate, estimate_audio_seconds, measure_audio_seconds
from .profiles import PROFILES, image_b64_only
from .surfaces import SURFACES
from .translate import audio, openai_chat
from .translate.ir import IRMessage, Text
from .translate.upstream_errors import UpstreamFailure, model_failure, network_error
from .types import Attempt, BodySummary, Hints, RouteContext, Served, Target

log = logging.getLogger('titanos.broker')

TRANSLATION_PROMPTS = {
    'broker-translation-v1': (
        'You are a translation engine. Translate the user text from {source} to {target}. Preserve meaning, tone, '
        'formatting, numbers and names. Answer with the translation only: no notes, no quotes, no explanations.'),
}
TRANSLATION_PROMPT_VERSION = 'broker-translation-v1'
CLOUD_TASKS = {'speech': 'openai.audio.speech', 'image': 'openai.images.generations',
               'transcription': 'openai.audio.transcriptions', 'translation': 'openai.chat', 'chat': 'openai.chat',
               'vision': 'openai.chat', 'embeddings': 'openai.embeddings'}
INVOKER_TASKS = frozenset({'speech', 'image', 'transcription', 'translation', 'chat', 'embeddings'})
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
DEFAULT_RATE = 16000


@dataclass(frozen=True)
class CloudSpecView:
    """A read-only ModelSpec-shaped view of a cloud target for API Plus stage code."""
    task: str
    backend: str = 'cloud'
    defaults: Any = field(default_factory=lambda: MappingProxyType({}))
    voices: Any = field(default_factory=lambda: MappingProxyType({}))
    device: str = 'remote'
    target: Target | None = None
    path: str = ''
    source_repo: str | None = None
    source_revision: str | None = None
    native: Any = None
    dtype: str = 'auto'
    parallelism: str = 'single'
    allow_pickle: bool = False
    trust_remote_code: bool = False
    allow_cpu_fallback: bool = False
    load_options: Any = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class StageTarget:
    kind: Literal['local', 'cloud']
    spec: Any
    target: Target | None
    backend: str
    task: str


def _cancelled(cancel) -> bool:
    if cancel is None:
        return False
    check = getattr(cancel, 'is_set', None)
    if callable(check):
        return bool(check())
    return bool(cancel() if callable(cancel) else cancel)


def _check(cancel, deadline: float | None) -> None:
    if _cancelled(cancel):
        raise BrokerError('The caller cancelled the request', 'cancelled')
    if deadline is not None and time.monotonic() >= deadline:
        raise BrokerError('The request deadline passed', 'deadline_exceeded')


def _stage_surface(task: str):
    name = CLOUD_TASKS.get(task)
    if name is None:
        return None
    return replace(SURFACES[name], scope='inference:plus', kind='stage')


def _cloud_location(principal) -> str:
    """The location a resolved cloud target is re-checked under: the first of auto, cloud_only and prefer_local the
    principal may use (a request header still takes precedence in effective_policy)."""
    allowed = getattr(getattr(principal, 'constraints', None), 'locations', None)
    for location in ('auto', 'cloud_only', 'prefer_local'):
        if allowed is None or location in allowed:
            return location
    return 'local_only'


class StageRouter:
    """Resolves the model of an API Plus stage or asset job (registry lookups on cloud-eligible stage paths)."""

    def __init__(self, broker):
        self.broker = broker

    def _local(self, name: str, task: str) -> StageTarget | None:
        config = self.broker.local_config()
        spec = (getattr(config, 'models', None) or {}).get(name) if config is not None else None
        if spec is None:
            return None
        return StageTarget(kind='local', spec=spec, target=None, backend=spec.backend, task=task)

    def plan(self, name: str, task: str, *, location: str | None, principal):
        surface = _stage_surface(task)
        if surface is None:
            raise BrokerError(f'{task} stages run on local models only', 'broker_capability_unavailable')
        principal = principal or broker_context.current_principal() or self.broker.default_principal()
        hints = broker_context.current_hints() or Hints(request_id=uuid.uuid4().hex)
        ctx = RouteContext(surface=surface, requested_model=name, principal=principal, hints=hints,
                           body=BodySummary())
        return self.broker.snapshot().router.resolve(ctx, body_location=location or 'local_only'), ctx

    def resolve_stage(self, name: str, task: str, *, location: str | None = 'local_only', principal=None) -> StageTarget:
        local = self._local(name, task)
        if task not in CLOUD_TASKS:
            if location == 'cloud_only':
                raise BrokerError(f'{task} stages run on local models only', 'broker_capability_unavailable')
            if local is None:
                raise BrokerError(f'Model {name!r} not found', 'model_not_found')
            return local
        principal = principal or broker_context.current_principal()
        if not self.broker.active(principal):
            if location == 'cloud_only':
                raise BrokerError('No cloud target is configured', 'location_unsatisfiable', origin='policy')
            if local is None:
                raise BrokerError(f'Model {name!r} not found', 'model_not_found')
            return local
        plan, _ = self.plan(name, task, location=location, principal=principal)
        first = plan.chain[0]
        if first.klass == 'local':
            found = self._local(first.local_name or first.ref, task)
            if found is None:
                raise BrokerError(f'Model {name!r} not found', 'model_not_found')
            return found
        return StageTarget(kind='cloud', spec=CloudSpecView(task=task, target=first), target=first, backend='cloud',
                           task=task)


class GatewayRouter:
    """NativePool's candidates()/complete() for API Plus, with cloud targets behind the broker policy."""

    def __init__(self, native, broker, principal, location):
        self.native = native
        self.broker = broker
        self.principal = principal
        self.location = location

    def _stage(self, model: str) -> StageTarget:
        return self.broker.stage_router().resolve_stage(model, 'chat', location=self.location, principal=self.principal)

    def candidates(self, model, worker=None):
        stage = self._stage(model)
        if stage.kind == 'local':
            if self.native is None:
                raise BrokerError('Native gateway is unavailable', 'upstream_unconfigured')
            return self.native.candidates(model, worker)
        if worker is not None:
            raise BrokerError('worker selects a local worker; it cannot be used with a cloud target', 'invalid_request')
        return [stage.target]

    async def complete(self, data, *, deadline=None, cancel=None, context_max_tokens=None, require_loaded=False):
        if type(require_loaded) is not bool:
            raise BrokerError('require_loaded must be a boolean', 'invalid_request')
        model = data.get('model') if isinstance(data, dict) else None
        stage = self._stage(model)
        if stage.kind == 'local':
            if self.native is None:
                raise BrokerError('Native gateway is unavailable', 'upstream_unconfigured')
            return await self.native.complete(data, deadline=deadline, cancel=cancel,
                                              context_max_tokens=context_max_tokens, require_loaded=require_loaded)
        if context_max_tokens is not None:
            raise BrokerError('Exact context accounting needs a local native model', 'broker_capability_unavailable')
        invoker = self.broker.invoker()
        return await asyncio.to_thread(invoker.infer_sync, stage.target, 'chat', data, cancel=cancel,
                                       deadline=deadline, principal=self.principal)


def _wav_info(data: bytes) -> tuple[int, float] | None:
    """(sample_rate, duration) of a PCM WAV, or None."""
    if not data.startswith(b'RIFF') or data[8:12] != b'WAVE':
        return None
    try:
        with wave.open(io.BytesIO(data)) as reader:
            rate = reader.getframerate()
            frames = reader.getnframes()
            if rate > 0 and 0 < frames < 0x7FFFFFFF:
                return rate, frames / rate
            return rate, 0.0
    except (wave.Error, EOFError, struct.error):
        return None


def asr_metadata(verbose: dict, *, task: str, audio_bytes: bytes | None, requested_language: str | None) -> dict:
    """An ASRMetadata-shaped dict from an OpenAI verbose_json transcription answer."""
    text = verbose.get('text') if isinstance(verbose, dict) else None
    if not isinstance(text, str):
        raise BrokerError('The upstream transcription answer has no text', 'upstream_bad_response', origin='upstream')
    info = _wav_info(audio_bytes or b'')
    rate = info[0] if info else DEFAULT_RATE
    duration = verbose.get('duration')
    if not isinstance(duration, (int, float)) or duration <= 0:
        duration = info[1] if info and info[1] > 0 else 0.0
    words_in = [w for w in verbose.get('words') or [] if isinstance(w, dict)]
    segments_in = [s for s in verbose.get('segments') or [] if isinstance(s, dict)]
    if duration <= 0:
        ends = [float(w.get('end') or 0) for w in words_in] + [float(s.get('end') or 0) for s in segments_in]
        duration = max(ends, default=0.0)
    duration = float(duration) if duration > 0 else 1.0 / rate
    segments = [{'id': i, 'start': float(s['start']) if s.get('start') is not None else None,
                 'end': float(s['end']) if s.get('end') is not None else None, 'text': str(s.get('text', ''))}
                for i, s in enumerate(segments_in)]
    words = [{'id': i, 'start': float(w['start']) if w.get('start') is not None else None,
              'end': float(w['end']) if w.get('end') is not None else None, 'word': str(w.get('word', ''))}
             for i, w in enumerate(words_in)]
    language = verbose.get('language') if isinstance(verbose.get('language'), str) and verbose.get('language') \
        else None
    return {'task': 'translate' if task == 'translation' else 'transcribe', 'text': text, 'duration': duration,
            'sample_rate': rate, 'decoded_samples': max(1, int(round(duration * rate))), 'language': language,
            'language_source': 'backend' if language else 'unavailable', 'requested_language': requested_language,
            'segments': segments or None, 'words': words or None,
            'timestamps': {'segments': bool(segments), 'words': bool(words)}, 'quality': {}, 'backend': 'cloud'}


class CloudInvoker:
    """Non-streaming cloud calls on worker threads (asset jobs and API Plus stages)."""

    def __init__(self, broker):
        self.broker = broker
        self.calls = 0

    # -- plumbing --------------------------------------------------------------------------------------------------------
    def _send(self, adapter, grant, method: str, path: str, *, cancel, deadline, stream: bool = False, **kw):
        _check(cancel, deadline)
        egress = self.broker.egress
        try:
            response = egress.send_sync(grant, adapter.upstream, adapter.profile, method, path, stream=stream, **kw)
        except Exception as exc:
            import httpx
            if isinstance(exc, httpx.HTTPError):
                raise UpstreamFailure(network_error(exc)) from None
            raise
        if not 200 <= response.status_code < 300:
            try:
                body = response.read()
            finally:
                response.close()
            raise UpstreamFailure(adapter.map_error(response.status_code, response.headers, body))
        if not stream:
            return response, response.content
        chunks = []
        try:
            for chunk in response.iter_raw():
                if _cancelled(cancel):
                    raise BrokerError('The caller cancelled the request', 'cancelled')
                if deadline is not None and time.monotonic() >= deadline:
                    raise BrokerError('The request deadline passed', 'deadline_exceeded')
                chunks.append(chunk)
        finally:
            response.close()
        return response, b''.join(chunks)

    @staticmethod
    def _json(data: bytes):
        try:
            return json.loads(data.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise BrokerError('The upstream answered with a body that is not JSON', 'upstream_bad_response',
                              origin='upstream') from None

    @staticmethod
    def _upload(data: dict) -> bytes | None:
        raw = data.get('audio', data.get('file'))
        if isinstance(raw, tuple):
            raw = raw[1] if len(raw) > 1 else b''
        return bytes(raw) if isinstance(raw, (bytes, bytearray)) else None

    def _upload_seconds(self, task: str, data: dict) -> float | None:
        """The measured duration of a transcription upload (limits.measure_audio_seconds), else None."""
        raw = self._upload(data) if task == 'transcription' else None
        return measure_audio_seconds(raw) if raw else None

    def _estimate(self, task: str, data: dict) -> Estimate:
        if task in ('chat', 'translation'):
            size = len(json.dumps(data, default=str))
            cap = data.get('max_tokens') or data.get('max_completion_tokens')
            return Estimate(input_tokens=-(-size // 3), max_output_tokens=cap if isinstance(cap, int) else None)
        if task == 'embeddings':
            return Estimate(input_tokens=-(-len(json.dumps(data.get('input'), default=str)) // 3), max_output_tokens=0,
                            token_output=False)
        if task == 'speech':
            text = data.get('input') or data.get('text') or ''
            return Estimate(max_output_tokens=0, characters=len(text) if isinstance(text, str) else 0,
                            token_output=False)
        if task == 'image':
            return Estimate(max_output_tokens=0, images=int(data.get('n') or 1), token_output=False)
        if task == 'transcription':
            return Estimate(max_output_tokens=0, audio_seconds=estimate_audio_seconds(self._upload(data) or b''),
                            token_output=False)
        return Estimate(max_output_tokens=0, token_output=False)

    # -- API -------------------------------------------------------------------------------------------------------------
    def infer_sync(self, target: Target, task: str, data: dict, *, cancel=None, deadline: float | None = None,
                   principal=None, admission=None, decision_id: str | None = None):
        """admission: a reservation taken earlier (a cloud asset job reserves its budget at submit, server.py); it is
        settled or released here instead of admitting again. decision_id: the Decision id recorded on the job."""
        if task not in INVOKER_TASKS:
            raise BrokerError(f'{task} is not available from cloud targets', 'broker_capability_unavailable')
        if target is None or target.klass == 'local':
            raise BrokerError('CloudInvoker serves cloud targets only', 'invalid_request')
        broker = self.broker
        snapshot = broker.snapshot()
        principal = principal or broker_context.current_principal() or broker.default_principal()
        upstream = snapshot.config.upstreams.get(target.upstream_id)
        adapter = snapshot.adapters.get(target.upstream_id)
        if upstream is None or adapter is None:
            raise BrokerError('The upstream is not configured', 'upstream_unconfigured')
        problem = snapshot.catalog.readiness(target.upstream_id)
        if problem is not None:
            raise BrokerError(f'Model {target.ref!r} not found', 'model_not_found')
        now = time.monotonic()
        limit = now + float(broker.settings.client_timeout)
        deadline = min(deadline, limit) if deadline is not None else limit
        _check(cancel, deadline)
        hints = broker_context.current_hints() or Hints(request_id=uuid.uuid4().hex)
        surface = _stage_surface(task)
        ctx = RouteContext(surface=surface, requested_model=target.ref, principal=principal, hints=hints,
                           body=BodySummary())
        # the target may have been resolved long ago (a queued job): the principal and the CURRENT snapshot's policy
        # are checked again before anything is admitted or sent
        snapshot.router.recheck(ctx, target, body_location=_cloud_location(principal))
        decision = snapshot.router.new_decision(ctx)
        if decision_id:
            decision.decision_id = decision_id
        decision.chain = [target.ref]
        if admission is None:
            admission = broker.limiter.admit(principal, target, self._estimate(task, data))
        breaker = broker.breakers.get(target.upstream_id, upstream.credential)
        permit = breaker.allow()
        started = time.perf_counter()
        usage = None
        outcome = 'success'
        status = 200
        try:
            if not permit:
                raise BrokerError('The upstream is temporarily unavailable', 'breaker_open',
                                  retry_after=permit.retry_after)
            grant = snapshot.issuer.issue('inference', plan_target=target, upstream_id=target.upstream_id,
                                          decision_id=decision.decision_id, request_id=hints.request_id,
                                          delegated=hints.delegated, reservation=admission, deadline=deadline)
            self.calls += 1
            injected = admission.injected_max_tokens if admission is not None else None
            result, usage = self._call(adapter, grant, target, task, data, cancel=cancel, deadline=deadline,
                                       injected=injected, voice_maps=snapshot.config.voice_maps)
            breaker.record('success')
            return result
        except UpstreamFailure as failure:
            breaker.record(failure.upstream.kind, retry_after=failure.upstream.retry_after)
            # a provider 410, or a 404 about the model: 404 model_not_found
            mapped = model_failure(failure, target.ref, upstream_model=target.upstream_model)
            outcome, status = failure.upstream.kind, mapped.status
            if mapped is not failure:
                raise mapped from None
            raise
        except BrokerError as exc:
            if permit:
                breaker.record('cancelled' if exc.code == 'cancelled' else 'refused')
            outcome, status = ('cancelled' if exc.code == 'cancelled' else 'refused'), exc.status
            raise
        except BaseException:  # an unexpected failure must not keep a half-open probe slot taken
            if permit:
                breaker.record('refused')
            outcome, status = 'error', 500
            raise
        finally:
            ms = (time.perf_counter() - started) * 1000.0
            decision.attempts.append(Attempt(ref=target.ref, outcome=outcome, status=status, ms_to_first_byte=ms,
                                             request_bytes_sent=outcome != 'refused'))
            decision.served = Served(ref=target.ref, klass=target.klass, upstream=target.upstream_id,
                                     upstream_model=target.upstream_model) if outcome == 'success' else None
            decision.translation = f'stage:{task}'
            decision.usage = usage
            cost = basis = None
            if outcome == 'success':
                priced = snapshot.prices.price(target.ref, usage, upload_audio_seconds=self._upload_seconds(task, data))
                cost, basis = priced.cost, priced.basis
            from .pricing import RESERVATION, money
            decision.latency_ms = round(ms, 3)
            try:
                if admission is not None:
                    if outcome == 'success':
                        admission.settle(usage, cost)
                        if cost is None and admission.charged is not None and admission.price is not None:
                            # nothing priceable: the reservation is the spend (fail closed), with its basis recorded
                            cost, basis = admission.charged, RESERVATION
                            decision.notes.append('cost:reserved_estimate')
                    else:
                        admission.release()
                decision.cost_usd = money(cost) if cost is not None else None
                decision.price_basis = basis
                broker.decisions.append(decision)
            except Exception:
                decision.cost_usd = money(cost) if cost is not None else None
                decision.price_basis = basis
                log.warning('broker: stage bookkeeping failed', exc_info=True)

    # -- per task --------------------------------------------------------------------------------------------------------
    def _call(self, adapter, grant, target: Target, task: str, data: dict, *, cancel, deadline, injected, voice_maps):
        name = adapter.adapter_name
        model = target.upstream_model
        if task == 'speech':
            return self._speech(adapter, grant, model, data, cancel=cancel, deadline=deadline, voice_maps=voice_maps)
        if task == 'image':
            return self._image(adapter, grant, model, data, cancel=cancel, deadline=deadline)
        if task == 'transcription':
            return self._transcription(adapter, grant, model, data, cancel=cancel, deadline=deadline)
        if task in ('chat', 'translation'):
            body = dict(data)
            if task == 'translation':
                body = self._translation_body(data)
            body['stream'] = False
            if injected is not None and body.get('max_tokens') is None and body.get('max_completion_tokens') is None:
                body['max_tokens'] = injected
            ir = openai_chat.decode_request(body)
            _check(cancel, deadline)
            resp = adapter.invoke_sync(grant, self.broker.egress, ir)
            _check(cancel, deadline)
            document = openai_chat.encode_response(resp, model_name=target.ref)
            choice = (document.get('choices') or [{}])[0]
            text = ''.join(p.text for p in resp.content if isinstance(p, Text))
            result = {'text': text, 'finish_reason': choice.get('finish_reason'),
                      'usage': openai_chat.usage_dict(resp.usage)}
            if task == 'translation':
                result['prompt_version'] = TRANSLATION_PROMPT_VERSION
            else:
                result.update({'native_response': document,
                               'provenance': {'backend': 'cloud', 'upstream': target.upstream_id, 'model': model,
                                              'served_ref': target.ref, 'adapter': name},
                               'native_runtime': None})
            return result, resp.usage
        if task == 'embeddings':
            value = data.get('input')
            body = {'model': model, 'input': value}
            for key in ('dimensions', 'input_type', 'encoding_format'):
                if data.get(key) is not None:
                    body[key] = data[key]
            ir = openai_chat.decode_embed_request(body)
            _check(cancel, deadline)
            resp = adapter.invoke_sync(grant, self.broker.egress, ir)
            vectors = (resp.extras or {}).get('vectors') or []
            tokens = resp.usage.input_tokens or 0
            return ({'data': [{'index': i, 'embedding': list(v)} for i, v in enumerate(vectors)],
                     'usage': {'prompt_tokens': tokens, 'total_tokens': tokens}}, resp.usage)
        raise BrokerError(f'{task} is not available from cloud targets', 'broker_capability_unavailable')

    @staticmethod
    def _translation_body(data: dict) -> dict:
        text = data.get('text')
        if not isinstance(text, str) or not text:
            raise BrokerError('text is required', 'invalid_request')
        source = data.get('source_language') or 'the detected source language'
        target = data.get('target_language')
        if not target:
            raise BrokerError('target_language is required', 'invalid_request')
        prompt = TRANSLATION_PROMPTS[TRANSLATION_PROMPT_VERSION].format(source=source, target=target)
        body = {'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': text}],
                'temperature': 0}
        if isinstance(data.get('max_tokens'), int):
            body['max_tokens'] = data['max_tokens']
        return body

    def _speech(self, adapter, grant, model: str, data: dict, *, cancel, deadline, voice_maps):
        text = data.get('input') if isinstance(data.get('input'), str) else data.get('text')
        if not isinstance(text, str) or not text:
            raise BrokerError('input text is required', 'invalid_request')
        from .types import Usage
        if adapter.adapter_name == 'gemini':  # a Gemini TTS model through generateContent (translate.gemini_media)
            body = {'input': text, 'response_format': 'wav', 'voice': data.get('voice')}
            if data.get('speed') is not None:
                body['speed'] = data['speed']
            _, _, wav, usage = adapter.media_sync(grant, self._sender(adapter, grant, cancel, deadline),
                                                  'openai.audio.speech', body, None)
            return (wav, 'audio/wav'), usage
        if adapter.adapter_name == 'elevenlabs':
            voice = data.get('voice') or ''
            mapped = audio._voice_map(voice_maps, 'openai_to_elevenlabs').get(voice, voice)
            if not mapped:
                raise BrokerError('voice is required for ElevenLabs speech', 'invalid_request')
            body = {'text': text, 'model_id': model}
            if data.get('speed') is not None:
                body['voice_settings'] = {'speed': float(data['speed'])}
            path = adapter.profile.surfaces['elevenlabs.tts_stream']
            response, pcm = self._send(adapter, grant, 'POST', path, cancel=cancel, deadline=deadline, stream=True,
                                       path_params={'voice_id': mapped}, params={'output_format': 'pcm_24000'},
                                       json=body)
            from .adapters.elevenlabs import character_usage
            return (audio.pcm_to_wav(pcm, audio.PCM_RATE), 'audio/wav'), character_usage(response.headers, text)
        if adapter.adapter_name != 'openai_compat':
            raise BrokerError(f'{adapter.profile.id} does not offer speech', 'broker_capability_unavailable')
        path, params = adapter.route('openai.audio.speech', model)
        body = {'model': model, 'input': text, 'voice': data.get('voice') or 'alloy', 'response_format': 'wav'}
        if data.get('speed') is not None:
            body['speed'] = data['speed']
        response, payload = self._send(adapter, grant, 'POST', path, cancel=cancel, deadline=deadline, stream=True,
                                       path_params=params, json=body)
        if not payload.startswith(b'RIFF'):
            raise BrokerError('The upstream speech answer is not WAV', 'upstream_bad_response', origin='upstream')
        return (payload, 'audio/wav'), Usage(characters=len(text), estimated=True)

    def _sender(self, adapter, grant, cancel, deadline):
        """send(method, path, **kw) -> (response, bytes) through _send (cancel and deadline checks) for adapters that
        build their own requests (GeminiAdapter.media_sync)."""
        def send(method, path, **kw):
            return self._send(adapter, grant, method, path, cancel=cancel, deadline=deadline, **kw)
        return send

    def _gemini_image(self, adapter, grant, data: dict, *, cancel, deadline):
        """[PNG bytes] from a Gemini image model (a JPEG or WebP answer is converted to PNG when Pillow is present,
        else refused like any non-PNG output)."""
        body = {k: data[k] for k in ('prompt', 'size', 'n') if data.get(k) is not None}
        _, _, payload, usage = adapter.media_sync(grant, self._sender(adapter, grant, cancel, deadline),
                                                  'openai.images.generations', body, None)
        images = []
        for item in self._json(payload).get('data') or []:
            image = base64.b64decode(item['b64_json'])
            if not image.startswith(PNG_MAGIC):
                try:
                    from PIL import Image as PILImage
                    buffer = io.BytesIO()
                    PILImage.open(io.BytesIO(image)).save(buffer, format='PNG')
                    image = buffer.getvalue()
                except Exception:
                    raise BrokerError('The upstream image is not a PNG', 'broker_capability_unavailable') from None
            images.append(image)
        if not images:
            raise BrokerError('The upstream answered without images', 'upstream_bad_response', origin='upstream')
        return images, usage

    def _image(self, adapter, grant, model: str, data: dict, *, cancel, deadline):
        profile = adapter.profile
        if adapter.adapter_name == 'gemini':
            return self._gemini_image(adapter, grant, data, cancel=cancel, deadline=deadline)
        if adapter.adapter_name != 'openai_compat':
            raise BrokerError(f'{profile.id} does not offer image generation', 'broker_capability_unavailable')
        if profile.image_formats and 'png' not in profile.image_formats:
            raise BrokerError(f'{profile.id} returns {", ".join(sorted(profile.image_formats))} images; this stage '
                              'needs PNG', 'broker_capability_unavailable')
        prompt = data.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            raise BrokerError('prompt is required', 'invalid_request')
        body = {'model': model, 'prompt': prompt}
        for key in ('n', 'size', 'quality', 'style', 'background'):
            if data.get(key) is not None:
                body[key] = data[key]
        if not image_b64_only(profile, model):  # a b64-only model (every OpenAI image model) is never sent the field
            # the provider's spelling of base64 (Together: 'base64'; profile extras image_response_format_values)
            body['response_format'] = (profile.extras.get('image_response_format_values') or {}).get('b64_json',
                                                                                                    'b64_json')
        for key, value in (profile.extras.get('image_png_request') or {}).items():
            body.setdefault(key, value)  # a provider that defaults to JPEG is asked for PNG (Together output_format)
        path, params = adapter.route('openai.images.generations', model)
        response, payload = self._send(adapter, grant, 'POST', path, cancel=cancel, deadline=deadline,
                                       path_params=params, json=body)
        document = self._json(payload)
        images = []
        for item in (document.get('data') if isinstance(document, dict) else None) or []:
            raw = item.get('b64_json') if isinstance(item, dict) else None
            if not isinstance(raw, str):
                raise BrokerError('The upstream answered without image bytes', 'broker_capability_unavailable')
            try:
                image = base64.b64decode(raw)
            except (binascii.Error, ValueError):
                raise BrokerError('The upstream image is not valid base64', 'upstream_bad_response',
                                  origin='upstream') from None
            if not image.startswith(PNG_MAGIC):
                raise BrokerError('The upstream image is not a PNG', 'broker_capability_unavailable')
            images.append(image)
        if not images:
            raise BrokerError('The upstream answered without images', 'upstream_bad_response', origin='upstream')
        from .types import Usage
        return images, Usage(images=len(images))

    def _transcription(self, adapter, grant, model: str, data: dict, *, cancel, deadline):
        raw = data.get('audio', data.get('file'))
        filename, content_type = data.get('filename') or 'audio.wav', data.get('content_type') or 'audio/wav'
        if isinstance(raw, tuple):
            filename, raw, content_type = raw[0], raw[1], raw[2] if len(raw) > 2 else content_type
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            raise BrokerError('audio bytes are required', 'invalid_request')
        raw = bytes(raw)
        language = data.get('language')
        task = 'translation' if data.get('task') == 'translate' else 'transcription'
        if adapter.adapter_name == 'elevenlabs':
            if task == 'translation':
                raise BrokerError('Audio translation is not available from ElevenLabs', 'broker_capability_unavailable')
            form = [('model', model), ('file', (filename, raw, content_type))]
            if language:
                form.append(('language', language))
            fields = audio.stt_openai_to_elevenlabs(form)
            fields = [(k, model if k == 'model_id' else v) for k, v in fields]
            data_fields, files = split_form(fields)
            response, payload = self._send(adapter, grant, 'POST', adapter.profile.surfaces['elevenlabs.stt'],
                                           cancel=cancel, deadline=deadline, data=data_fields, files=files)
            answer = self._json(payload)
            verbose = audio.stt_elevenlabs_to_openai(answer, 'verbose_json')
        elif adapter.adapter_name == 'gemini':  # audio understanding through generateContent (no timestamps)
            surface = 'openai.audio.translations' if task == 'translation' else 'openai.audio.transcriptions'
            form = [('model', model), ('file', (filename, raw, content_type)), ('response_format', 'verbose_json')]
            if language and task != 'translation':
                form.append(('language', language))
            _, _, payload, gem_usage = adapter.media_sync(grant, self._sender(adapter, grant, cancel, deadline),
                                                          surface, None, form)
            verbose = self._json(payload)
            verbose.pop('language', None)  # Gemini detects no language: the request's own stays requested_language
            result = asr_metadata(verbose, task=task, audio_bytes=raw, requested_language=language)
            return result, gem_usage
        elif adapter.adapter_name == 'openai_compat':
            surface = 'openai.audio.translations' if task == 'translation' else 'openai.audio.transcriptions'
            path, params = adapter.route(surface, model)
            form = [('model', model), ('file', (filename, raw, content_type)), ('response_format', 'verbose_json'),
                    ('timestamp_granularities[]', 'word'), ('timestamp_granularities[]', 'segment')]
            if language and task != 'translation':
                form.append(('language', language))
            data_fields, files = split_form(form)
            response, payload = self._send(adapter, grant, 'POST', path, cancel=cancel, deadline=deadline,
                                           path_params=params, data=data_fields, files=files)
            verbose = self._json(payload)
        else:
            raise BrokerError(f'{adapter.profile.id} does not offer transcription', 'broker_capability_unavailable')
        result = asr_metadata(verbose, task=task, audio_bytes=raw, requested_language=language)
        from .types import Usage
        reported = verbose.get('usage') if isinstance(verbose.get('usage'), dict) else {}
        tokens = {'input_tokens': reported.get('input_tokens'), 'output_tokens': reported.get('output_tokens')} \
            if reported.get('type', 'tokens') == 'tokens' else {}
        if isinstance(verbose.get('duration'), (int, float)) and verbose['duration'] > 0:
            return result, Usage(audio_seconds=result['duration'], **tokens)
        if reported.get('type') == 'duration' and isinstance(reported.get('seconds'), (int, float)):
            return result, Usage(audio_seconds=reported['seconds'])
        # no upstream duration: the token usage when reported; the settlement prices a per-minute row by the measured
        # upload duration (basis audio_seconds:upload) and otherwise keeps the reservation (basis reservation)
        return result, (Usage(**tokens) if any(v is not None for v in tokens.values()) else None)


__all__ = ['CLOUD_TASKS', 'CloudInvoker', 'CloudSpecView', 'GatewayRouter', 'StageRouter', 'StageTarget',
           'TRANSLATION_PROMPTS', 'TRANSLATION_PROMPT_VERSION', 'asr_metadata']
