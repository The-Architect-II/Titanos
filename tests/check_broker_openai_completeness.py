"""OpenAI and Azure OpenAI completeness (owner direction 2026-10-02: a provider key "just works", every argument, every
modality), over the integrated app (create_app: Guard, BrokerDispatch) against the WP4 OpenAI loopback mock extended
with the public streaming and error wire formats, plus unit checks of the shared pieces (error classification, usage
parsing, pricing, parameter maps, the profile rows). The netguard is installed; every integrated test ends with zero
non-loopback connection attempts. Synthetic sentinel credentials only.

- error envelopes: a plain OpenAI 403 (organization verification, region, model access) passes through as 403 with the
  provider's message and type and is never a fallback trigger or a credential mark; a 429 insufficient_quota is a
  billing stop (503 upstream_unavailable without Retry-After, the provider's message kept), not a retryable rate limit;
  a 400 keeps the provider's error.code, error.type and error.param;
- streaming fidelity: images/generations and images/edits with stream=true relay image_generation.* / image_edit.*
  events (a whole partial image larger than the 1 MiB SSE line limit included) and settle image token usage;
  audio/transcriptions with stream=true relays transcript.text.* events (whisper-1 answers JSON); audio/speech with
  stream_format=sse relays speech.audio.* events with token usage; GET /v1/responses/{id}?stream=true resumes the stored
  response's event stream;
- arguments: Titanos' local VAD gate fields and local sampling knobs (top_k, min_p, repeat_penalty) are dropped with
  X-Titanos-Dropped instead of reaching OpenAI as unknown parameters; the reasoning-model temperature/top_p drop is
  lifted for gpt-5-chat* and for reasoning effort 'none' (also on Azure deployments of the same models);
  response_format=url that cannot be re-hosted (output_format jpeg or webp, or stream=true) is refused before egress;
  images JSON answers use the media cap; moderations without a model use the provider's default
  (omni-moderation-latest) on the single visible OpenAI upstream, and stay 400 model_required with two;
- usage and pricing: chat audio tokens, image input tokens and transcription audio tokens are parsed and priced at
  their own PriceRow rates (text price when absent);
- native rows: the Realtime calls and session mints, ChatKit and evals need broker:admin; custom voice consents are
  stored and owner-bound; retired images/variations stays outside the allowlist (400 before egress);
- catalog: OpenAI entries carry titanos.model_capabilities per model family;
- storage: for a key without cloud:store a client's explicit store: true (chat and Responses) is forced off AND named in
  X-Titanos-Dropped; background: true and conversation are refused 403 scope_denied before egress (they would be billed
  but could never be polled, cancelled or continued); a key holding cloud:store on an allow_store upstream keeps them;
- bare model ids: the stock SDK's gpt-4o-mini (no @upstream, no alias) resolves on the single visible OpenAI upstream;
- per-model rate limits: a 429 on one model leaves another model on the same key serving, and the cooling model itself
  answers 429 upstream_rate_limited (not 503) with zero egress.
"""
from __future__ import annotations

import base64
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import OpenAIMock, netguard  # noqa: E402
from broker_mocks.fixtures import load  # noqa: E402
from broker_mocks.native_routes import clear_native, install_openai_native  # noqa: E402

from titanos.broker import profiles  # noqa: E402
from titanos.broker.adapters.openai_compat import rewrite_body, sampling_kept  # noqa: E402
from titanos.broker.pricing import PriceTable  # noqa: E402
from titanos.broker.sse import MAX_LINE, UsageObserver, openai_usage, parse_sse  # noqa: E402
from titanos.broker.surfaces import SURFACES  # noqa: E402
from titanos.broker.translate import upstream_errors as ue  # noqa: E402
from titanos.broker.types import Usage  # noqa: E402

BOOT = 'openai-complete-bootstrap-0001'
MOCKS: dict = {}
MODELS = ['gpt-4o-mini', 'gpt-4.1-mini', 'gpt-image-1', 'gpt-4o-mini-transcribe', 'whisper-1', 'gpt-4o-mini-tts',
          'omni-moderation-latest', 'gpt-5', 'gpt-5.1', 'gpt-5-chat-latest', 'gpt-4o-audio-preview']
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mM4UaEBAAN0AWn1BwN7AAAAAElFTkSuQmCC')


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['openai2'] = OpenAIMock()
    for mock in MOCKS.values():
        install_openai_native(mock)
        mock.start()


def tearDownModule():
    for mock in MOCKS.values():
        mock.stop()


def bearer(token: str) -> dict:
    return {'Authorization': 'Bearer ' + token}


def document(*, second: bool = False) -> dict:
    upstreams = {'oai': W.S.upstream(MOCKS['openai'], 'openai', credential='oai-key', models=MODELS,
                                     data_policy={'allow_store': True})}
    if second:
        upstreams['oai2'] = W.S.upstream(MOCKS['openai2'], 'openai', credential='oai2-key', models=MODELS)
    return {'policy': {'cloud_policy': 'explicit'}, 'upstreams': upstreams,
            'prices': [{'match': 'gpt-4o-audio-preview@oai', 'input_per_mtok': '2.50', 'output_per_mtok': '10',
                        'audio_output_per_mtok': '80'}]}


def sse_bodies(text: str) -> list[tuple[str, dict | str]]:
    out = []
    for block in text.replace('\r\n', '\n').split('\n\n'):
        event, data = '', []
        for line in block.split('\n'):
            if line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                data.append(line[5:].lstrip())
        if data:
            joined = '\n'.join(data)
            out.append((event, joined if joined == '[DONE]' else json.loads(joined)))
    return out


class Base(unittest.TestCase):
    doc_kw: dict = {}
    settings_kw: dict = {}

    def setUp(self):
        netguard.reset_attempts()
        for mock in MOCKS.values():
            mock.reset()
            clear_native(mock)
        self.full = W.FullApp(document(**self.doc_kw), api_key=BOOT, settings_kw=self.settings_kw)
        self.http = self.full.http
        self.mock = MOCKS['openai']

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def key(self, *scopes, **kw) -> str:
        return self.full.mint(*scopes, name='k' + str(len(scopes)), **kw)[1]

    def post(self, path, token=BOOT, **kw):
        headers = {**bearer(token), **kw.pop('headers', {})}
        return self.http.post(path, headers=headers, **kw)

    def decision(self, answer) -> dict:
        principal = self.full.broker.authenticator.bootstrap_principal()
        rows = self.full.broker.decisions.query(limit=20, principal=principal)
        wanted = answer.headers.get('x-titanos-decision-id')
        found = [row for row in rows if row.get('decision_id') == wanted]
        self.assertTrue(found, rows[:1])
        return found[0]

    def last_json(self) -> dict:
        return json.loads(self.mock.hits[-1].body.decode('utf-8'))


# ---------------------------------------------------------------------------------------------------------------------
# error envelopes

class ErrorEnvelopes(Base):
    def test_plain_403_passes_through_with_the_provider_message(self):
        self.mock.respond_with('openai.images.generations', 'error_403_verification')
        answer = self.post('/v1/images/generations', json={'model': 'gpt-image-1@oai', 'prompt': 'A red square'})
        self.assertEqual(answer.status_code, 403, answer.text)
        error = answer.json()['error']
        self.assertIn('must be verified', error['message'])
        self.assertEqual(error['type'], 'invalid_request_error')
        self.assertEqual(answer.headers['x-titanos-error-origin'], 'upstream')
        self.assertEqual(len(self.mock.hits), 1)
        # the credential is still usable: the next call reaches the provider
        self.assertEqual(self.post('/v1/chat/completions', json={'model': 'gpt-4o-mini@oai', 'messages': [
            {'role': 'user', 'content': 'Hello!'}]}).status_code, 200)

    def test_insufficient_quota_is_a_billing_stop_not_a_rate_limit(self):
        self.mock.respond_with('openai.chat', 'error_429_insufficient_quota')
        answer = self.post('/v1/chat/completions', json={'model': 'gpt-4o-mini@oai', 'messages': [
            {'role': 'user', 'content': 'Hello!'}]})
        self.assertEqual(answer.status_code, 503, answer.text)
        self.assertEqual(answer.json()['error']['code'], 'upstream_unavailable')
        self.assertIn('exceeded your current quota', answer.json()['error']['message'])
        self.assertNotIn('retry-after', answer.headers)

    def test_400_keeps_code_type_and_param(self):
        self.mock.respond_with('openai.chat', 'error_400_param')
        answer = self.post('/v1/chat/completions', json={'model': 'gpt-4o-mini@oai', 'messages': [
            {'role': 'user', 'content': 'Hello!'}]})
        self.assertEqual(answer.status_code, 400)
        self.assertEqual(answer.json()['error'], {
            'message': "Invalid type for 'messages[0].content': expected one of a string or array of objects, but got "
                       'an integer instead.', 'type': 'invalid_request_error', 'param': 'messages[0].content',
            'code': 'invalid_type'})


class ErrorClassification(unittest.TestCase):
    def body(self, status, **error):
        return json.dumps({'error': {'message': 'm', 'type': None, 'param': None, 'code': None, **error}}).encode()

    def test_table(self):
        plain = ue.classify(403, {}, self.body(403, code='unsupported_country_region_territory',
                                               type='request_forbidden'), adapter='openai_compat')
        self.assertEqual((plain.kind, ue.broker_code(plain), ue.is_passthrough(plain)),
                         ('client_error', ('invalid_request', 403), True))
        key = ue.classify(403, {}, self.body(403, code='invalid_api_key'), adapter='openai_compat')
        self.assertEqual(key.kind, 'auth')
        nvidia = ue.classify(403, {}, json.dumps({'status': 403, 'title': 'Forbidden',
                                                  'detail': 'Authorization failed'}).encode(), adapter='openai_compat')
        self.assertEqual(nvidia.kind, 'auth')  # not an OpenAI envelope: the documented 502 upstream_auth_failed
        anthropic = ue.classify(403, {}, json.dumps({'type': 'error', 'error': {'type': 'permission_error',
                                                                                'message': 'm'}}).encode(),
                                adapter='anthropic')
        self.assertEqual(anthropic.kind, 'client_error')  # the vendor envelopes pass a plain 403 through as well
        anthropic_key = ue.classify(403, {}, json.dumps({'type': 'error', 'error': {
            'type': 'authentication_error', 'message': 'm'}}).encode(), adapter='anthropic')
        self.assertEqual(anthropic_key.kind, 'auth')
        for status, fields in ((429, {'code': 'insufficient_quota', 'type': 'insufficient_quota'}),
                               (400, {'code': 'billing_hard_limit_reached'}), (429, {'type': 'insufficient_quota'})):
            quota = ue.classify(status, {'retry-after': '20'}, self.body(status, **fields), adapter='openai_compat')
            self.assertEqual((quota.kind, quota.retry_after, ue.broker_code(quota)),
                             ('server_error', None, ('upstream_unavailable', 503)))
        limited = ue.classify(429, {'retry-after': '3'}, self.body(429, code='rate_limit_exceeded'),
                              adapter='openai_compat')
        self.assertEqual((limited.kind, limited.retry_after), ('rate_limited', 3.0))

    def test_param_and_type_are_plain_identifiers_only(self):
        err = ue.classify(400, {}, self.body(400, param='tools[0].function.parameters', type='invalid_request_error'),
                          adapter='openai_compat')
        self.assertEqual((err.provider_param, err.provider_type), ('tools[0].function.parameters',
                                                                   'invalid_request_error'))
        odd = ue.classify(400, {}, self.body(400, param='a b <script>', type={'x': 1}), adapter='openai_compat')
        self.assertEqual((odd.provider_param, odd.provider_type), (None, None))
        rendered = json.loads(ue.passthrough_4xx(err, 'anthropic').body)
        self.assertEqual(rendered['error']['type'], 'invalid_request_error')  # Anthropic types follow the status


# ---------------------------------------------------------------------------------------------------------------------
# streaming fidelity

class MediaCap(Base):
    settings_kw = {'broker_max_upstream_json_bytes': 1024}

    def test_images_json_uses_the_media_cap(self):
        # n=10 base64 images: over the 1 KiB JSON cap of this server, inside the media cap
        answer = self.post('/v1/images/generations', json={'model': 'gpt-image-1@oai', 'prompt': 'x', 'n': 10})
        self.assertEqual(answer.status_code, 200, answer.text[:300])
        self.assertEqual(len(answer.json()['data']), 10)
        self.assertGreater(len(answer.content), 1024)


class MediaStreams(Base):
    def test_image_generation_stream_is_relayed_and_metered(self):
        answer = self.post('/v1/images/generations', json={'model': 'gpt-image-1@oai', 'prompt': 'A red square',
                                                            'stream': True, 'partial_images': 1})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertTrue(answer.headers['content-type'].startswith('text/event-stream'))
        fixture = load('openai', 'images_stream')
        self.assertEqual(answer.content, fixture.render({'model': 'gpt-image-1'}))
        self.assertEqual([e for e, _ in sse_bodies(answer.text)],
                         ['image_generation.partial_image', 'image_generation.completed'])
        usage = self.decision(answer)['usage']
        self.assertEqual((usage['images'], usage['input_tokens'], usage['output_tokens']), (1, 12, 272))
        self.assertTrue(self.last_json()['stream'])

    def test_image_stream_failures(self):
        # an error-first stream is a pre-commit failure (typed 503), a mid-stream disconnect ends with the OpenAI
        # terminal frame data:{error} (the SDK raises APIError on it)
        body = {'model': 'gpt-image-1@oai', 'prompt': 'A red square', 'stream': True, 'partial_images': 1}
        self.mock.faults.push('error_first_event')
        first = self.post('/v1/images/generations', json=body)
        self.assertEqual(first.status_code, 503, first.text)
        self.mock.faults.push('disconnect_after_frames', frames=1)
        cut = self.post('/v1/images/generations', json=body)
        self.assertEqual(cut.status_code, 200)
        events = sse_bodies(cut.text)
        self.assertEqual(events[0][1]['type'], 'image_generation.partial_image')
        self.assertEqual(events[-2][1]['error']['type'], 'stream_interrupted')
        self.assertEqual(events[-1][1], '[DONE]')

    def test_image_edit_stream_is_relayed(self):
        answer = self.post('/v1/images/edits', files={'image': ('square.png', PNG, 'image/png')},
                           data={'model': 'gpt-image-1@oai', 'prompt': 'Make it blue', 'stream': 'true',
                                 'partial_images': '1'})
        self.assertEqual(answer.status_code, 200, answer.text)
        events = sse_bodies(answer.text)
        self.assertEqual([e for e, _ in events], ['image_edit.partial_image', 'image_edit.completed'])
        usage = self.decision(answer)['usage']
        self.assertEqual((usage['images'], usage['image_input_tokens']), (1, 323))
        self.assertIn(b'name="stream"', self.mock.hits[-1].body)

    def test_a_partial_image_over_the_sse_line_limit_still_commits(self):
        big = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\0' * (MAX_LINE + 4096)).decode()
        frames = [{'type': 'image_generation.partial_image', 'b64_json': big, 'partial_image_index': 0},
                  {'type': 'image_generation.completed', 'b64_json': big,
                   'usage': {'input_tokens': 5, 'output_tokens': 7, 'total_tokens': 12}}]
        payload = ''.join(f'event: {f["type"]}\ndata: {json.dumps(f)}\n\n' for f in frames).encode()
        from broker_mocks.server import Reply, Route

        async def large(request):
            return Reply(200, [('content-type', 'text/event-stream')], payload)
        route = Route('POST', __import__('re').compile('^/v1/images/generations$'), large, 'openai.images.generations',
                      True)
        self.mock.routes.insert(0, route)
        try:
            answer = self.post('/v1/images/generations', json={'model': 'gpt-image-1@oai', 'prompt': 'x',
                                                                'stream': True})
        finally:
            self.mock.routes.remove(route)
        self.assertEqual(answer.status_code, 200, answer.text[:300])
        self.assertEqual(answer.content, payload)
        self.assertEqual(self.decision(answer)['usage']['output_tokens'], 7)

    def test_transcription_stream_and_whisper_json(self):
        answer = self.post('/v1/audio/transcriptions', files={'file': ('hello.wav', W.WAV, 'audio/wav')},
                           data={'model': 'gpt-4o-mini-transcribe@oai', 'stream': 'true'})
        self.assertEqual(answer.status_code, 200, answer.text)
        events = sse_bodies(answer.text)
        self.assertEqual([body['type'] for _, body in events],
                         ['transcript.text.delta', 'transcript.text.delta', 'transcript.text.done'])
        usage = self.decision(answer)['usage']
        self.assertEqual((usage['input_tokens'], usage['output_tokens'], usage['audio_input_tokens']), (22, 4, 22))
        whisper = self.post('/v1/audio/transcriptions', files={'file': ('hello.wav', W.WAV, 'audio/wav')},
                            data={'model': 'whisper-1@oai', 'stream': 'true'})
        self.assertEqual(whisper.status_code, 200, whisper.text)
        self.assertEqual(whisper.json()['text'], 'Hello world.')

    def test_speech_sse_is_relayed_with_token_usage(self):
        answer = self.post('/v1/audio/speech', json={'model': 'gpt-4o-mini-tts@oai', 'input': 'Hello world.',
                                                      'voice': 'alloy', 'stream_format': 'sse'})
        self.assertEqual(answer.status_code, 200, answer.text)
        events = sse_bodies(answer.text)
        self.assertEqual(events[-1][1]['type'], 'speech.audio.done')
        usage = self.decision(answer)['usage']
        self.assertEqual((usage['input_tokens'], usage['output_tokens']), (14, 101))

    def test_resumed_response_stream(self):
        token = self.key('inference', 'cloud', 'cloud:store')
        created = self.post('/v1/responses', token, json={'model': 'gpt-4.1-mini@oai', 'input': 'Hello!',
                                                           'store': True, 'background': True})
        self.assertEqual(created.status_code, 200, created.text)
        response_id = created.json()['id']
        answer = self.http.get(f'/v1/responses/{response_id}', params={'stream': 'true', 'starting_after': '2'},
                               headers=bearer(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertTrue(answer.headers['content-type'].startswith('text/event-stream'))
        self.assertEqual(sse_bodies(answer.text)[-1][1]['type'], 'response.completed')
        self.assertEqual(self.mock.hits[-1].query_params.get('stream'), 'true')
        self.assertEqual(self.mock.hits[-1].query_params.get('starting_after'), '2')


class StreamUnits(unittest.TestCase):
    def test_usage_observer_media_family(self):
        observer = UsageObserver('openai_media')
        from titanos.broker.sse import SSEFrame
        observer.sse(SSEFrame('image_generation.completed', json.dumps({
            'type': 'image_generation.completed', 'usage': {'input_tokens': 50, 'output_tokens': 272,
                                                            'input_tokens_details': {'image_tokens': 40,
                                                                                     'text_tokens': 10}}}), b''))
        self.assertEqual((observer.usage.images, observer.usage.input_tokens, observer.usage.image_input_tokens),
                         (1, 50, 40))
        observer = UsageObserver('openai_media')
        observer.sse(SSEFrame(None, json.dumps({'type': 'transcript.text.done', 'usage': {
            'type': 'duration', 'seconds': 3.5}}), b''))
        self.assertEqual(observer.usage.audio_seconds, 3.5)

    def test_responses_web_search_calls_are_counted(self):
        from titanos.broker.sse import SSEFrame, responses_object_usage
        response = {'output': [{'type': 'web_search_call', 'id': 'ws_1', 'status': 'completed'},
                               {'type': 'web_search_call', 'id': 'ws_2', 'status': 'completed'},
                               {'type': 'message', 'content': []}],
                    'usage': {'input_tokens': 10, 'output_tokens': 5}}
        self.assertEqual(responses_object_usage(response).web_search_requests, 2)
        observer = UsageObserver('responses')
        observer.sse(SSEFrame('response.completed', json.dumps({'type': 'response.completed',
                                                                 'response': response}), b''))
        self.assertEqual((observer.usage.input_tokens, observer.usage.web_search_requests), (10, 2))
        self.assertIsNone(responses_object_usage({'usage': {'input_tokens': 1}}).web_search_requests)

    def test_chat_audio_usage(self):
        usage = openai_usage(load('openai', 'chat_audio').body({'model': 'gpt-4o-audio-preview'})['usage'])
        self.assertEqual((usage.output_tokens, usage.audio_output_tokens, usage.audio_input_tokens), (60, 52, None))


# ---------------------------------------------------------------------------------------------------------------------
# arguments

class Arguments(Base):
    def chat(self, model, **body):
        return self.post('/v1/chat/completions', json={'model': model, 'messages': [
            {'role': 'user', 'content': 'Hello!'}], **body})

    def test_local_sampling_knobs_are_dropped_with_a_header(self):
        answer = self.chat('gpt-4o-mini@oai', top_k=40, min_p=0.05, repeat_penalty=1.1, temperature=0.2)
        self.assertEqual(answer.status_code, 200, answer.text)
        sent = self.last_json()
        self.assertFalse({'top_k', 'min_p', 'repeat_penalty'} & set(sent))
        self.assertEqual(sent['temperature'], 0.2)
        self.assertEqual(set(answer.headers['x-titanos-dropped'].split(', ')), {'top_k', 'min_p', 'repeat_penalty'})

    def test_reasoning_sampling_rules(self):
        self.assertEqual(self.chat('gpt-5@oai', temperature=0.3, top_p=0.9).status_code, 200)
        self.assertFalse({'temperature', 'top_p'} & set(self.last_json()))
        self.assertEqual(self.chat('gpt-5.1@oai', temperature=0.3, reasoning_effort='none').status_code, 200)
        self.assertEqual(self.last_json()['temperature'], 0.3)
        self.assertEqual(self.chat('gpt-5-chat-latest@oai', temperature=0.3, max_tokens=50).status_code, 200)
        sent = self.last_json()
        self.assertEqual((sent['temperature'], sent['max_completion_tokens']), (0.3, 50))
        answer = self.post('/v1/responses', json={'model': 'gpt-5.1@oai', 'input': 'Hello!', 'temperature': 0.4,
                                                  'reasoning': {'effort': 'none'}})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(self.last_json()['temperature'], 0.4)

    def test_vad_fields_are_dropped_on_cloud_transcription(self):
        answer = self.post('/v1/audio/transcriptions', files={'file': ('hello.wav', W.WAV, 'audio/wav')},
                           data={'model': 'whisper-1@oai', 'vad_enabled': 'true', 'vad_model': 'silero',
                                 'language': 'en'})
        self.assertEqual(answer.status_code, 200, answer.text)
        body = self.mock.hits[-1].body
        self.assertNotIn(b'vad_enabled', body)
        self.assertNotIn(b'vad_model', body)
        self.assertIn(b'name="language"', body)
        self.assertEqual(set(answer.headers['x-titanos-dropped'].split(', ')), {'vad_enabled', 'vad_model'})

    def test_unhostable_url_requests_are_refused_before_egress(self):
        for extra in ({'output_format': 'jpeg'}, {'output_format': 'webp'}, {'stream': True}):
            with self.subTest(**extra):
                answer = self.post('/v1/images/generations', json={'model': 'gpt-image-1@oai', 'prompt': 'x',
                                                                    'response_format': 'url', **extra})
                self.assertEqual(answer.status_code, 400, answer.text)
                self.assertEqual(answer.json()['error']['code'], 'broker_capability_unavailable')
        self.assertEqual(self.mock.hits, [])
        png = self.post('/v1/images/generations', json={'model': 'gpt-image-1@oai', 'prompt': 'x',
                                                         'response_format': 'url', 'output_format': 'png'})
        self.assertEqual(png.status_code, 200, png.text)
        self.assertIn('/v1/files/', png.json()['data'][0]['url'])

    def test_moderation_without_model_uses_the_provider_default(self):
        answer = self.post('/v1/moderations', json={'input': 'I like turtles.'})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(self.last_json()['model'], 'omni-moderation-latest')
        self.assertEqual(answer.headers['x-titanos-served-model'], 'omni-moderation-latest@oai')
        self.assertIn('model_absent:provider_default', self.decision(answer)['notes'])


class ArgumentsTwoUpstreams(Base):
    doc_kw = {'second': True}

    def test_moderation_without_model_stays_model_required_when_ambiguous(self):
        answer = self.post('/v1/moderations', json={'input': 'I like turtles.'})
        self.assertEqual(answer.status_code, 400, answer.text)
        self.assertEqual(answer.json()['error']['code'], 'model_required')
        self.assertEqual(self.mock.hits + MOCKS['openai2'].hits, [])


class Storage(Base):
    def test_store_is_named_when_forced_off_and_background_is_refused(self):
        plain = self.key('inference', 'cloud')
        chat = self.post('/v1/chat/completions', plain, json={'model': 'gpt-4o-mini@oai', 'store': True,
                                                              'messages': [{'role': 'user', 'content': 'hi'}]})
        self.assertEqual(chat.status_code, 200, chat.text)
        self.assertIs(self.last_json()['store'], False)
        self.assertIn('store', chat.headers['x-titanos-dropped'].split(', '))
        responses = self.post('/v1/responses', plain, json={'model': 'gpt-4o-mini@oai', 'input': 'hi', 'store': True})
        self.assertEqual(responses.status_code, 200, responses.text)
        self.assertIn('store', responses.headers['x-titanos-dropped'].split(', '))
        hits = len(self.mock.hits)
        for field, value in (('background', True), ('conversation', 'conv_1')):
            with self.subTest(field=field):
                refused = self.post('/v1/responses', plain, json={'model': 'gpt-4o-mini@oai', 'input': 'hi',
                                                                  field: value})
                self.assertEqual((refused.status_code, refused.json()['error']['code']), (403, 'scope_denied'),
                                 refused.text)
                self.assertIn('cloud:store', refused.json()['error']['message'])
        self.assertEqual(len(self.mock.hits), hits)
        stored = self.key('inference', 'cloud', 'cloud:store')
        kept = self.post('/v1/responses', stored, json={'model': 'gpt-4o-mini@oai', 'input': 'hi', 'background': True})
        self.assertEqual(kept.status_code, 200, kept.text)
        self.assertIs(self.last_json()['background'], True)
        self.assertNotIn('store', (kept.headers.get('x-titanos-dropped') or '').split(', '))

    def test_bare_model_id_and_per_model_rate_limit(self):
        token = self.key('inference', 'cloud')
        bare = self.post('/v1/chat/completions', token, json={'model': 'gpt-4o-mini', 'messages': [
            {'role': 'user', 'content': 'hi'}]})
        self.assertEqual(bare.status_code, 200, bare.text)
        self.assertEqual(bare.headers['x-titanos-served-model'], 'gpt-4o-mini@oai')
        self.mock.faults.push('status', status=429, retry_after=9)
        limited = self.post('/v1/chat/completions', token, json={'model': 'gpt-4o-mini', 'messages': [
            {'role': 'user', 'content': 'hi'}]})
        self.assertEqual(limited.status_code, 429, limited.text)
        hits = len(self.mock.hits)
        held = self.post('/v1/chat/completions', token, json={'model': 'gpt-4o-mini@oai', 'messages': [
            {'role': 'user', 'content': 'hi'}]})
        self.assertEqual((held.status_code, held.json()['error']['code']), (429, 'upstream_rate_limited'), held.text)
        self.assertEqual(len(self.mock.hits), hits)
        other = self.post('/v1/chat/completions', token, json={'model': 'gpt-4.1-mini@oai', 'messages': [
            {'role': 'user', 'content': 'hi'}]})
        self.assertEqual(other.status_code, 200, other.text)  # another model on the same key keeps serving


class RewriteUnits(unittest.TestCase):
    def test_azure_deployments_follow_the_openai_rules(self):
        azure = profiles.PROFILES['azure_openai']
        chat = SURFACES['openai.chat']
        body = {'messages': [], 'temperature': 0.2, 'top_k': 5, 'max_tokens': 9}
        out, notes = rewrite_body(azure, None, chat, body, 'gpt-5')
        self.assertNotIn('temperature', out)
        self.assertNotIn('top_k', out)
        self.assertEqual(out['max_completion_tokens'], 9)
        out, _ = rewrite_body(azure, None, chat, body, 'gpt-5-chat-latest')
        self.assertEqual(out['temperature'], 0.2)
        self.assertTrue(sampling_kept(azure, 'gpt-5.2', {'reasoning_effort': 'none'}))
        self.assertFalse(sampling_kept(azure, 'gpt-5.2', {'reasoning_effort': 'low'}))
        out, _ = rewrite_body(azure, None, chat, dict(body, reasoning_effort='none'), 'gpt-5.2')
        self.assertEqual(out['temperature'], 0.2)
        self.assertIn('dropped:top_k', notes)

    def test_rows(self):
        row = profiles.PROFILES['openai']
        native = {(r.method, r.path): r for r in row.native.routes}
        for path in ('/realtime/calls', '/realtime/sessions', '/realtime/transcription_sessions', '/chatkit/sessions',
                     '/realtime/calls/{call_id}/hangup'):
            self.assertTrue(native[('POST', path)].admin, path)
        self.assertTrue(native[('GET', '/evals')].admin)
        consent = native[('POST', '/audio/voice_consents')]
        self.assertEqual((consent.store, consent.owner), (True, 'create'))
        self.assertNotIn(('POST', '/images/variations'), native)
        self.assertEqual(profiles.model_capabilities(row, 'gpt-4o-mini-tts'), ('speech',))
        self.assertEqual(profiles.model_capabilities(row, 'gpt-4o-transcribe'), ('transcription',))
        self.assertIn('images', profiles.model_capabilities(row, 'gpt-image-1'))
        self.assertIn('chat', profiles.model_capabilities(row, 'gpt-4.1-mini'))
        self.assertIsNone(profiles.model_capabilities(row, 'some-future-model'))
        self.assertEqual(row.extras['default_models']['moderation'], 'omni-moderation-latest')


class PricingUnits(unittest.TestCase):
    def test_modality_tokens_at_their_own_prices(self):
        table = PriceTable([{'match': 'gpt-4o-audio-preview@oai', 'input_per_mtok': '2.50', 'output_per_mtok': '10',
                             'audio_input_per_mtok': '40', 'audio_output_per_mtok': '80'},
                            {'match': 'gpt-image-1@oai', 'input_per_mtok': '5', 'output_per_mtok': '40',
                             'image_input_per_mtok': '10'},
                            {'match': 'plain@oai', 'input_per_mtok': '1', 'output_per_mtok': '2'}])
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, audio_input_tokens=400_000,
                      audio_output_tokens=500_000)
        priced = table.price('gpt-4o-audio-preview@oai', usage)
        # text in 0.6M x 2.50 + audio in 0.4M x 40 + text out 0.5M x 10 + audio out 0.5M x 80
        self.assertEqual((priced.cost, priced.basis), (Decimal('1.5') + 16 + 5 + 40, 'tokens'))
        image = table.price('gpt-image-1@oai', Usage(input_tokens=1_000_000, output_tokens=0,
                                                     image_input_tokens=500_000, images=1))
        self.assertEqual(image.cost, Decimal('2.5') + 5)
        # a row without modality prices bills every token at the text price (unchanged behaviour)
        self.assertEqual(table.price('plain@oai', usage).cost, Decimal(3))


# ---------------------------------------------------------------------------------------------------------------------
# native rows and catalog

class NativeRows(Base):
    def test_admin_rows_need_broker_admin(self):
        plain = self.key('native', 'cloud', 'cloud:store', 'inference')
        for path in ('/native/oai/realtime/calls', '/native/oai/chatkit/sessions', '/native/oai/realtime/sessions'):
            with self.subTest(path=path):
                refused = self.post(path, plain, json={'session': {'type': 'realtime', 'model': 'gpt-realtime'}})
                self.assertEqual(refused.status_code, 403, refused.text)
        self.assertEqual(self.mock.hits, [])
        admin = self.key('native', 'cloud', 'broker:admin')
        reached = self.post('/native/oai/realtime/calls', admin, content=b'v=0\r\n', headers={
            'Content-Type': 'application/sdp'})
        self.assertNotEqual(reached.status_code, 403, reached.text)
        self.assertEqual((self.mock.hits[-1].method, self.mock.hits[-1].path), ('POST', '/v1/realtime/calls'))
        self.assertEqual(self.mock.hits[-1].body, b'v=0\r\n')

    def test_voice_consents_are_stored_rows_and_variations_stay_out(self):
        plain = self.key('native', 'cloud', 'inference')
        refused = self.post('/native/oai/audio/voice_consents', plain, files={
            'recording': ('consent.wav', W.WAV, 'audio/wav')}, data={'name': 'me', 'language': 'en-US'})
        self.assertEqual(refused.status_code, 403, refused.text)  # provider-side state needs cloud:store
        storing = self.key('native', 'cloud', 'cloud:store', 'inference')
        self.post('/native/oai/audio/voice_consents', storing, files={
            'recording': ('consent.wav', W.WAV, 'audio/wav')}, data={'name': 'me', 'language': 'en-US'})
        self.assertEqual(self.mock.hits[-1].path, '/v1/audio/voice_consents')
        retired = self.post('/native/oai/images/variations', storing, files={'image': ('a.png', PNG, 'image/png')})
        self.assertEqual(retired.status_code, 400, retired.text)
        self.assertEqual(retired.json()['error']['code'], 'broker_capability_unavailable')
        self.assertEqual(self.mock.hits[-1].path, '/v1/audio/voice_consents')


class AzureLane(unittest.TestCase):
    """Azure OpenAI (classic data plane) through the real EgressClient and the test-only Azure host transport (R12)."""

    def setUp(self):
        netguard.reset_attempts()
        import broker_adapter_support as A
        self.A = A
        self.targets = A.start_targets(('AZ',))
        self.target = self.targets['AZ']
        install_openai_native(self.target.mock)

    def tearDown(self):
        clear_native(self.target.mock)
        self.A.stop_targets(self.targets)
        self.assertEqual(netguard.attempts(), [])

    def test_native_files_reach_the_resource_with_api_version(self):
        import asyncio
        from titanos.broker.dispatch import egress_options
        spec = profiles.PROFILES['azure_openai'].native
        self.assertIsNone(spec.bare_prefix)
        self.assertIn(('POST', '/files'), {(r.method, r.path) for r in spec.routes})
        self.assertIn(('WS', '/realtime'), {(r.method, r.path) for r in spec.routes})
        self.assertNotIn(('POST', '/realtime/client_secrets'), {(r.method, r.path) for r in spec.routes})

        async def run():
            async with self.target.egress() as egress:
                response = await egress.send(self.target.grant(), self.target.upstream, self.target.profile, 'POST',
                                             '/files', files=[('file', ('b.jsonl', b'{"a":1}\n', 'application/jsonl'))],
                                             data={'purpose': 'batch'}, native=egress_options(spec), params=[])
                return response.status_code, response.json()
        status, body = asyncio.run(run())
        self.assertEqual(status, 200, body)
        hit = self.target.mock.hits[-1]
        self.assertEqual(hit.path, '/openai/files')
        self.assertEqual(hit.query_params.get('api-version'), self.A.AZURE_API_VERSION)
        self.assertEqual(hit.credentials, [('header:api-key', '<sentinel:azure_openai>')])

    def test_azure_error_shapes(self):
        verify = ue.classify(403, {}, json.dumps({'error': {'code': '403', 'message': 'Access to this model is '
                                                            'restricted for this resource.'}}).encode(),
                             adapter='openai_compat')
        self.assertEqual(ue.broker_code(verify), ('invalid_request', 403))
        denied = ue.classify(401, {}, json.dumps({'error': {'code': '401', 'message': 'Access denied'}}).encode(),
                             adapter='openai_compat')
        self.assertEqual(ue.broker_code(denied), ('upstream_auth_failed', 502))


class Catalog(Base):
    def test_entries_carry_model_capabilities(self):
        answer = self.http.get('/v1/models', headers=bearer(BOOT))
        self.assertEqual(answer.status_code, 200, answer.text)
        entries = {row['id']: row for row in answer.json()['data'] if row.get('remote')}
        self.assertEqual(entries['gpt-4o-mini-tts@oai']['titanos']['model_capabilities'], ['speech'])
        self.assertEqual(entries['whisper-1@oai']['titanos']['model_capabilities'], ['transcription', 'translation'])
        self.assertIn('images', entries['gpt-image-1@oai']['titanos']['model_capabilities'])


if __name__ == '__main__':
    unittest.main()
