"""ElevenLabs completeness (2026-10-02 owner direction: a user with an ElevenLabs key gets every modality and argument).

Over the integrated app (create_app: Guard, BrokerDispatch) against the extended WP4 ElevenLabsMock, with the netguard
installed (every test ends with zero non-loopback connection attempts); synthetic sentinel credentials only.

Unit level:
- error mapping (translate.upstream_errors): an ElevenLabs 401 marks the shared credential failed only for a key-rejection
  code (invalid_api_key, needs_authorization ...) or a 401 without a readable code; 401 quota_exceeded is a provider billing
  state (503 upstream_unavailable, never marked); 401 missing_permissions (a restricted-scope key) is 502
  upstream_auth_failed but never marked; the current envelope's detail.code is read before the legacy detail.status;
- the probe: a restricted-scope key (401 missing_permissions on GET /v1/models) or an exhausted one validates;
- speech translation: an unmapped voice that looks like an ElevenLabs voice id passes as that id (OpenAI built-in voice
  names are still refused without a mapping);
- speech-to-text usage: audio_duration_secs (single and multichannel), the last word end otherwise, None for a webhook
  acknowledgement.
Through the app:
- the ElevenLabs SDK's bare model ids (eleven_multilingual_v2, scribe_v1) resolve on the single visible ElevenLabs
  upstream (resolution 'vendor'); with two such upstreams they stay 404 model_not_found;
- text-to-speech forwards every body field and the allowlisted query, returns history-item-id and the concurrency
  headers (never an unlisted provider header);
- a text-to-speech request without model_id (convert and stream) reaches the provider with its default model when no
  single local speech model exists; with one, the local route still answers (R16);
- speech-to-text forwards every form field (repeated keyterms kept) and enable_logging, never a single-use token query;
  multichannel answers settle on audio_duration_secs; webhook=true answers 202;
- a 401 quota_exceeded answers 503 and leaves the credential usable; 401 missing_permissions answers 502 and leaves it
  usable; 401 invalid_api_key marks it (the upstream is then unvalidated);
- native passthrough (bare public paths with the SDK's xi-api-key): GET /v1/models with xi-api-key and no Authorization
  is the provider's own model list, with Authorization it stays Titanos' catalog; dialogue (binary and with timestamps),
  sound effects (provider validation passed through), voice changer and audio isolation (multipart), forced alignment,
  music (song-id header) and its plan, voice search, voice settings, voice cloning and dubbing (cloud:store, owner-bound:
  another key gets 404), the subscription (broker:admin only), history (broker:admin only), and the provider's
  character-cost settles the native decision's usage;
- the OpenAI /v1/audio/speech route to ElevenLabs with a raw ElevenLabs voice id.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
import httpx  # noqa: E402
from broker_mocks import ElevenLabsMock, SENTINELS, netguard  # noqa: E402
from broker_mocks.fixtures import load  # noqa: E402

from titanos.broker.adapters.elevenlabs import ElevenLabsAdapter, stt_usage  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.profiles import PROFILES  # noqa: E402
from titanos.broker.translate import audio  # noqa: E402
from titanos.broker.translate.upstream_errors import (broker_code, classify, credential_failure,  # noqa: E402
                                                      provider_fields)

BOOT = 'elevenlabs-check-bootstrap-0001'
VOICE = '21m00Tcm4TlvDq8ikWAM'
MOCKS: dict = {}
INFER = ('inference', 'cloud')
NATIVE = ('native', 'cloud', 'cloud:store', 'models:read')


def setUpModule():
    MOCKS['xi'] = ElevenLabsMock()
    MOCKS['xi2'] = ElevenLabsMock()
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def body_of(fixture: str):
    return load('elevenlabs', fixture).body({})


def xi(token: str) -> dict:
    return {'xi-api-key': token}


# -- unit level ---------------------------------------------------------------------------------------------------------
class ErrorMapping(unittest.TestCase):
    def raw(self, fixture):
        return json.dumps(body_of(fixture)).encode('utf-8')

    def test_401_codes_decide_the_credential_mark(self):
        quota, permissions = self.raw('error_401_quota'), self.raw('error_401_permissions')
        invalid = self.raw('error_401_invalid')
        self.assertFalse(credential_failure(401, quota, adapter='elevenlabs'))
        self.assertFalse(credential_failure(401, permissions, adapter='elevenlabs'))
        self.assertTrue(credential_failure(401, invalid, adapter='elevenlabs'))
        self.assertTrue(credential_failure(401, b'', adapter='elevenlabs'))
        self.assertTrue(credential_failure(401, b'{"detail": {"code": "invalid_api_key", "message": "x"}}',
                                           adapter='elevenlabs'))
        self.assertTrue(credential_failure(403, b'{"detail": {"code": "invalid_api_key", "message": "x"}}',
                                           adapter='elevenlabs'))
        self.assertFalse(credential_failure(403, b'{"detail": {"code": "voice_access_denied", "message": "x"}}',
                                            adapter='elevenlabs'))
        self.assertTrue(credential_failure(401, permissions, adapter='openai_compat'))  # other providers unchanged

    def test_401_quota_is_unavailable_and_permissions_is_auth(self):
        quota = classify(401, {}, self.raw('error_401_quota'), adapter='elevenlabs')
        self.assertEqual((quota.kind, quota.provider_code), ('server_error', 'quota_exceeded'))
        self.assertEqual(broker_code(quota), ('upstream_unavailable', 503))
        self.assertIn('exceeds your quota', quota.message)
        permissions = classify(401, {}, self.raw('error_401_permissions'), adapter='elevenlabs')
        self.assertEqual((permissions.kind, broker_code(permissions)), ('auth', ('upstream_auth_failed', 502)))
        credits = classify(402, {}, b'{"detail": {"code": "insufficient_credits", "message": "x"}}',
                           adapter='elevenlabs')
        self.assertEqual(broker_code(credits), ('upstream_unavailable', 503))

    def test_current_envelope_code_is_read_first(self):
        code, message = provider_fields(self.raw('error_envelope_v2'), 'elevenlabs')
        self.assertEqual((code, message), ('invalid_voice_settings', 'The voice settings contain invalid values'))
        err = classify(400, {}, self.raw('error_envelope_v2'), adapter='elevenlabs')
        self.assertEqual(broker_code(err), ('invalid_request', 400))


class _Response:
    def __init__(self, status, document):
        self.status_code = status
        self.headers = httpx.Headers({'content-type': 'application/json'})
        self.content = json.dumps(document).encode('utf-8')

    def json(self):
        return json.loads(self.content)


class _Egress:
    def __init__(self, status, document):
        self.answer = _Response(status, document)

    async def send(self, *args, **kw):
        return self.answer


class Probe(unittest.TestCase):
    def probe(self, status, document):
        import asyncio
        adapter = ElevenLabsAdapter(PROFILES['elevenlabs'], None)
        return asyncio.run(adapter.probe(None, _Egress(status, document)))

    def test_restricted_or_exhausted_keys_validate(self):
        restricted = self.probe(401, body_of('error_401_permissions'))
        self.assertEqual((restricted.ok, restricted.code), (True, 'key_restricted'))
        exhausted = self.probe(401, body_of('error_401_quota'))
        self.assertEqual((exhausted.ok, exhausted.code), (True, 'quota_exceeded'))
        invalid = self.probe(401, body_of('error_401_invalid'))
        self.assertEqual((invalid.ok, invalid.code), (False, 'upstream_auth_failed'))
        self.assertTrue(self.probe(200, body_of('models')).ok)


class Translation(unittest.TestCase):
    def test_raw_voice_id_passes_and_openai_names_still_need_a_mapping(self):
        notes = []
        voice, body, query, wrap = audio.speech_openai_to_elevenlabs({'input': 'Hi', 'voice': VOICE}, {}, notes=notes)
        self.assertEqual((voice, body['text'], query['output_format'], wrap), (VOICE, 'Hi', 'mp3_44100_128', False))
        self.assertIn('voice:elevenlabs_id', notes)
        for name in ('nova', 'alloy', 'a b', 'x'):
            with self.subTest(voice=name), self.assertRaises(BrokerError):
                audio.speech_openai_to_elevenlabs({'input': 'Hi', 'voice': name}, {})
        mapped, _, _, _ = audio.speech_openai_to_elevenlabs({'input': 'Hi', 'voice': 'alloy'}, {'alloy': VOICE})
        self.assertEqual(mapped, VOICE)

    def test_speech_to_text_usage(self):
        multi = stt_usage(body_of('stt_multichannel'))
        self.assertEqual((multi.audio_seconds, multi.estimated), (1.25, False))
        single = stt_usage(load('elevenlabs', 'stt').body({'model_id': 'scribe_v1'}))
        self.assertEqual((single.audio_seconds, single.estimated), (0.96, True))
        self.assertIsNone(stt_usage(body_of('stt_webhook')))


class Allowlist(unittest.TestCase):
    """The native rows cover the public voice and audio API the official SDK calls (paths as the SDK spells them)."""
    COVERED = [('POST', '/v1/text-to-speech/v/with-timestamps'), ('POST', '/v1/text-to-speech/v/stream/with-timestamps'),
               ('POST', '/v1/text-to-dialogue'), ('POST', '/v1/text-to-dialogue/stream/with-timestamps'),
               ('POST', '/v1/sound-generation'), ('POST', '/v1/speech-to-speech/v/stream'),
               ('POST', '/v1/audio-isolation/stream'), ('GET', '/v1/audio-isolation/history'),
               ('POST', '/v1/forced-alignment'), ('POST', '/v1/music/detailed/stream'),
               ('POST', '/v1/music/stem-separation'), ('POST', '/v1/music/video-to-music'), ('POST', '/v1/music/upload'),
               ('GET', '/v1/music/finetunes'), ('GET', '/v2/voices'), ('GET', '/v1/voices/accents'),
               ('POST', '/v1/similar-voices'), ('GET', '/v1/shared-voices'), ('POST', '/v1/voices/add'),
               ('POST', '/v1/voices/pvc'), ('POST', '/v1/voices/pvc/v/samples'), ('POST', '/v1/voices/pvc/v/train'),
               ('POST', '/v1/text-to-voice/design'), ('POST', '/v1/text-to-voice/create-previews'),
               ('POST', '/v1/text-to-voice/v/remix'), ('GET', '/v1/models'), ('GET', '/v1/user/subscription'),
               ('GET', '/v1/usage/character-stats'), ('POST', '/v1/single-use-token/tts_websocket'),
               ('GET', '/v1/history'), ('POST', '/v1/history/download'),
               ('GET', '/v1/speech-to-text/transcripts/t'), ('POST', '/v1/dubbing'), ('GET', '/v1/dubbing/d'),
               ('GET', '/v1/dubbing/d/transcripts/es/format/srt'), ('POST', '/v1/dubbing/resource/d/translate'),
               ('POST', '/v1/pronunciation-dictionaries/add-from-rules'),
               ('POST', '/v1/pronunciation-dictionaries/p/set-rules'),
               ('GET', '/v1/pronunciation-dictionaries/p/v/download'),
               ('WS', '/v1/text-to-speech/v/stream-input'), ('WS', '/v1/text-to-speech/v/multi-stream-input'),
               ('WS', '/v1/speech-to-text/realtime')]

    def test_sdk_paths_are_allowlisted_with_the_right_gates(self):
        from titanos.broker import native
        for method, path in self.COVERED:
            with self.subTest(method=method, path=path):
                self.assertIsNotNone(native.match_route('elevenlabs', method, path))
        train, params = native.match_route('elevenlabs', 'POST', '/v1/voices/pvc/v1x/train')
        self.assertEqual((train.store, train.owner, native.owner_param(train, params)), (True, 'bound', 'v1x'))
        resource, params = native.match_route('elevenlabs', 'POST', '/v1/dubbing/resource/dub1/translate')
        self.assertEqual(native.owner_param(resource, params), 'dub1')
        for method, path in (('GET', '/v1/user/subscription'), ('POST', '/v1/single-use-token/realtime_scribe'),
                             ('GET', '/v1/history'), ('GET', '/v1/music/finetunes')):
            row, _ = native.match_route('elevenlabs', method, path)
            self.assertTrue(row.admin or row.owner == 'list', (method, path))
        for method, path in (('GET', '/v1/convai/agents'), ('POST', '/v1/workspace/invites/add'),
                             ('DELETE', '/v1/models')):
            self.assertIsNone(native.match_route('elevenlabs', method, path), (method, path))


# -- through the app ----------------------------------------------------------------------------------------------------
MODELS = ['eleven_multilingual_v2', 'eleven_flash_v2_5', 'eleven_v3', 'scribe_v1']


def document(*, second: bool = False, allow_store: bool = True) -> dict:
    upstreams = {'xi': W.S.upstream(MOCKS['xi'], 'elevenlabs', credential='xi-key', models=MODELS,
                                    data_policy={'allow_store': allow_store})}
    if second:
        upstreams['xi2'] = W.S.upstream(MOCKS['xi2'], 'elevenlabs', credential='xi2-key', models=MODELS)
    return {'policy': {'cloud_policy': 'explicit'}, 'upstreams': upstreams,
            'voice_maps': {'openai_to_elevenlabs': {'alloy': VOICE}}}


class Base(unittest.TestCase):
    doc_kw: dict = {}
    local_models: dict = {}

    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
        self.full = W.FullApp(document(**self.doc_kw), api_key=BOOT, models=self.local_models)
        self.http = self.full.http

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def key(self, *scopes, **kw) -> str:
        return self.full.mint(*scopes, name='k' + str(len(scopes)) + str(len(kw)) + str(id(scopes) % 997), **kw)[1]

    def decisions(self, limit=5):
        return self.full.broker.decisions.query(limit=limit,
                                                principal=self.full.broker.authenticator.bootstrap_principal())

    def readiness(self, upstream='xi'):
        return self.full.broker.snapshot().catalog.readiness(upstream)


class TypedSurfaces(Base):
    def test_bare_model_id_and_every_tts_argument(self):
        token = self.key(*INFER)
        body = {'text': 'Hello there.', 'model_id': 'eleven_multilingual_v2', 'language_code': 'en',
                'voice_settings': {'stability': 0.4, 'similarity_boost': 0.8, 'style': 0.1, 'use_speaker_boost': True,
                                   'speed': 1.05},
                'pronunciation_dictionary_locators': [{'pronunciation_dictionary_id': 'pd1', 'version_id': 'v1'}],
                'seed': 42, 'previous_text': 'Before.', 'next_text': 'After.', 'previous_request_ids': ['r0'],
                'apply_text_normalization': 'on', 'apply_language_text_normalization': False, 'use_pvc_as_ivc': False,
                'future_field': {'kept': True}}
        answer = self.http.post(f'/v1/text-to-speech/{VOICE}?output_format=mp3_22050_32&optimize_streaming_latency=2'
                                '&enable_logging=false', json=body, headers=xi(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertTrue(answer.headers['content-type'].startswith('audio/mpeg'))
        self.assertEqual(answer.headers['x-titanos-served-by'], 'cloud:xi')
        self.assertEqual(answer.headers.get('history-item-id'), 'histmocktts0001')
        self.assertEqual(answer.headers.get('maximum-concurrent-requests'), '5')
        self.assertNotIn('x-mock-private', answer.headers)
        hit = MOCKS['xi'].hits[-1]
        self.assertEqual(hit.json(), body)
        self.assertEqual(hit.query_params.get('output_format'), 'mp3_22050_32')
        self.assertEqual(hit.query_params.get('optimize_streaming_latency'), '2')
        self.assertEqual(hit.query_params.get('enable_logging'), 'false')
        self.assertEqual(hit.credential, '<sentinel:elevenlabs>')
        self.assertFalse(hit.presented(token))
        stream = self.http.post(f'/v1/text-to-speech/{VOICE}/stream', json={'text': 'Hi', 'model_id': 'eleven_v3'},
                                headers=xi(token))
        self.assertEqual(stream.status_code, 200, stream.text)
        self.assertEqual(MOCKS['xi'].hits[-1].json()['model_id'], 'eleven_v3')
        unknown = self.http.post(f'/v1/text-to-speech/{VOICE}', json={'text': 'Hi', 'model_id': 'eleven_nope'},
                                 headers=xi(token))
        self.assertEqual(unknown.status_code, 404, unknown.text)

    def test_tts_without_model_uses_the_provider_default(self):
        token = self.key(*INFER)
        for path in (f'/v1/text-to-speech/{VOICE}', f'/v1/text-to-speech/{VOICE}/stream'):
            with self.subTest(path=path):
                answer = self.http.post(path, json={'text': 'No model named.'}, headers=xi(token))
                self.assertEqual(answer.status_code, 200, answer.text)
                self.assertEqual(MOCKS['xi'].hits[-1].json()['model_id'], 'eleven_multilingual_v2')
                self.assertIn('model_absent:provider_default', self.decisions(1)[0]['notes'])

    def test_speech_to_text_forwards_every_field(self):
        token = self.key(*INFER)
        files = {'file': ('stereo.wav', W.WAV, 'audio/wav')}
        data = {'model_id': 'scribe_v1', 'use_multi_channel': 'true', 'diarize': 'true', 'num_speakers': '2',
                'tag_audio_events': 'false', 'timestamps_granularity': 'character', 'keyterms': ['Titanos', 'Vega'],
                'entity_detection': 'pii'}
        answer = self.http.post('/v1/speech-to-text?enable_logging=false&token=single-use-0001', data=data,
                                files=files, headers=xi(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()['transcripts'][1]['text'], 'Hi.')
        hit = MOCKS['xi'].hits[-1]
        self.assertEqual(hit.query_params.get('enable_logging'), 'false')
        self.assertNotIn('token', hit.query_params)
        for name in ('use_multi_channel', 'diarize', 'num_speakers', 'timestamps_granularity', 'entity_detection'):
            self.assertIn(f'name="{name}"'.encode(), hit.body, name)
        self.assertEqual(hit.body.count(b'name="keyterms"'), 2)
        usage = self.decisions(1)[0]['usage']
        self.assertEqual((usage['audio_seconds'], usage['estimated']), (1.25, False))
        webhook = self.http.post('/v1/speech-to-text', data={'model_id': 'scribe_v1', 'webhook': 'true',
                                                             'source_url': 'https://example.invalid/clip.mp4'},
                                 files={'unused': ('x.txt', b'x', 'text/plain')}, headers=xi(token))
        self.assertEqual(webhook.status_code, 202, webhook.text)
        self.assertEqual(webhook.json()['transcription_id'], 'transcriptmock0002')

    def test_quota_and_permission_401s_keep_the_credential(self):
        token = self.key(*INFER)
        request = dict(json={'text': 'Hi', 'model_id': 'eleven_multilingual_v2'}, headers=xi(token))
        MOCKS['xi'].faults.push('status', status=401, code='quota_exceeded', message='This request exceeds your quota.')
        quota = self.http.post(f'/v1/text-to-speech/{VOICE}', **request)
        self.assertEqual(quota.status_code, 503, quota.text)
        self.assertIsNone(self.readiness())
        MOCKS['xi'].faults.push('status', status=401, code='missing_permissions',
                                message='The API key you used is missing the permission text_to_speech.')
        denied = self.http.post(f'/v1/text-to-speech/{VOICE}', **request)
        self.assertEqual(denied.status_code, 502, denied.text)
        self.assertIsNone(self.readiness())
        self.assertEqual(self.http.post(f'/v1/text-to-speech/{VOICE}', **request).status_code, 200)
        MOCKS['xi'].faults.push('status', status=401, code='invalid_api_key', message='Invalid API key')
        rejected = self.http.post(f'/v1/text-to-speech/{VOICE}', **request)
        self.assertEqual(rejected.status_code, 502, rejected.text)
        self.assertIsNotNone(self.readiness())

    def test_openai_speech_with_a_raw_elevenlabs_voice(self):
        token = self.key(*INFER)
        answer = self.http.post('/v1/audio/speech', json={'model': 'eleven_flash_v2_5@xi', 'input': 'Hi',
                                                          'voice': 'mockvoice00000000002'},
                                headers=W.bearer(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertTrue(MOCKS['xi'].hits[-1].path.startswith('/v1/text-to-speech/mockvoice00000000002'),
                        MOCKS['xi'].hits[-1].path)


class TwoUpstreams(Base):
    doc_kw = {'second': True}

    def test_bare_ids_are_ambiguous(self):
        token = self.key(*INFER)
        answer = self.http.post(f'/v1/text-to-speech/{VOICE}', json={'text': 'Hi', 'model_id': 'eleven_v3'},
                                headers=xi(token))
        self.assertEqual(answer.status_code, 404, answer.text)
        qualified = self.http.post(f'/v1/text-to-speech/{VOICE}', json={'text': 'Hi', 'model_id': 'eleven_v3@xi2'},
                                   headers=xi(token))
        self.assertEqual(qualified.status_code, 200, qualified.text)
        self.assertEqual(len(MOCKS['xi2'].hits), 1)


class LocalSpeechKeepsPrecedence(Base):
    local_models = {'kokoro': 'speech'}

    def test_model_absent_goes_to_the_local_route(self):
        answer = self.http.post(f'/v1/text-to-speech/{VOICE}', json={'text': 'Local please.'},
                                headers=W.bearer(BOOT))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(MOCKS['xi'].hits, [])
        self.assertTrue(any(call[1] == 'speech' for call in self.full.engine.calls))


class NativeLane(Base):
    def test_models_listing_follows_the_sdk_header(self):
        token = self.key(*NATIVE)
        vendor = self.http.get('/v1/models', headers=xi(token))
        self.assertEqual(vendor.status_code, 200, vendor.text)
        self.assertIsInstance(vendor.json(), list)
        self.assertEqual(vendor.json()[0]['model_id'], 'eleven_multilingual_v2')
        self.assertEqual(MOCKS['xi'].hits[-1].path, '/v1/models')
        before = len(MOCKS['xi'].hits)
        titanos = self.http.get('/v1/models', headers=W.bearer(token))
        self.assertEqual(titanos.status_code, 200, titanos.text)
        self.assertEqual(titanos.json()['object'], 'list')
        self.assertEqual(len(MOCKS['xi'].hits), before)

    def test_generation_endpoints(self):
        token = self.key(*NATIVE)
        inputs = [{'text': 'Hi.', 'voice_id': VOICE}, {'text': 'Yo there.', 'voice_id': 'mockvoice00000000002'}]
        dialogue = self.http.post('/v1/text-to-dialogue?output_format=mp3_44100_128',
                                  json={'inputs': inputs, 'model_id': 'eleven_v3', 'settings': {'stability': 0.5},
                                        'seed': 3}, headers=xi(token))
        self.assertEqual(dialogue.status_code, 200, dialogue.text)
        self.assertTrue(dialogue.headers['content-type'].startswith('audio/mpeg'))
        self.assertEqual(MOCKS['xi'].hits[-1].json()['settings'], {'stability': 0.5})
        native = self.decisions(1)[0]
        self.assertEqual((native['surface'], native['usage']['characters']), ('native', 12))
        stamps = self.http.post('/v1/text-to-dialogue/with-timestamps', json={'inputs': inputs}, headers=xi(token))
        self.assertEqual(stamps.json()['voice_segments'][1]['dialogue_input_index'], 1)
        tts_stamps = self.http.post(f'/v1/text-to-speech/{VOICE}/with-timestamps', json={'text': 'Hi.'},
                                    headers=xi(token))
        self.assertEqual(tts_stamps.json()['alignment']['characters'], ['H', 'i', '.'])
        lines = self.http.post(f'/v1/text-to-speech/{VOICE}/stream/with-timestamps', json={'text': 'Hi.'},
                               headers=xi(token))
        self.assertEqual([json.loads(line)['alignment']['characters'] for line in lines.text.splitlines()],
                         [['H', 'i'], ['.']])
        sound = self.http.post('/v1/sound-generation', json={'text': 'rain on a tin roof', 'duration_seconds': 3.5,
                                                             'prompt_influence': 0.4, 'loop': True,
                                                             'model_id': 'eleven_text_to_sound_v2'},
                               headers=xi(token))
        self.assertEqual(sound.status_code, 200, sound.text)
        self.assertEqual(MOCKS['xi'].hits[-1].json()['loop'], True)
        bad = self.http.post('/v1/sound-generation', json={'text': 'x', 'duration_seconds': 60}, headers=xi(token))
        self.assertEqual(bad.status_code, 400, bad.text)
        changer = self.http.post(f'/v1/speech-to-speech/{VOICE}', data={'model_id': 'eleven_multilingual_sts_v2',
                                                                        'remove_background_noise': 'true'},
                                 files={'audio': ('in.wav', W.WAV, 'audio/wav')}, headers=xi(token))
        self.assertEqual(changer.status_code, 200, changer.text)
        isolation = self.http.post('/v1/audio-isolation', files={'audio': ('in.wav', W.WAV, 'audio/wav')},
                                   headers=xi(token))
        self.assertEqual(isolation.status_code, 200, isolation.text)
        aligned = self.http.post('/v1/forced-alignment', data={'text': 'Hi'},
                                 files={'file': ('in.wav', W.WAV, 'audio/wav')}, headers=xi(token))
        self.assertEqual(aligned.json()['words'][0]['text'], 'Hi')
        music = self.http.post('/v1/music', json={'prompt': 'calm piano', 'music_length_ms': 10000,
                                                  'force_instrumental': True}, headers=xi(token))
        self.assertEqual(music.status_code, 200, music.text)
        self.assertTrue(music.headers.get('song-id', '').startswith('songmock'))
        plan = self.http.post('/v1/music/plan', json={'prompt': 'calm piano'}, headers=xi(token))
        self.assertEqual(plan.json()['sections'][0]['section_name'], 'Intro')

    def test_voices_and_owned_objects(self):
        owner, other = self.key(*NATIVE), self.key(*NATIVE)
        search = self.http.get('/v2/voices?page_size=2&search=calm&voice_type=default', headers=xi(owner))
        self.assertEqual(search.json()['next_page_token'], 'page-token-0002')
        self.assertEqual(MOCKS['xi'].hits[-1].query_params.get('voice_type'), 'default')
        settings = self.http.get(f'/v1/voices/{VOICE}/settings', headers=xi(owner))
        self.assertEqual(settings.json()['similarity_boost'], 0.75)
        clone = self.http.post('/v1/voices/add', data={'name': 'Narrator clone'},
                               files={'files': ('s.wav', W.WAV, 'audio/wav')}, headers=xi(owner))
        self.assertEqual(clone.status_code, 200, clone.text)
        voice_id = clone.json()['voice_id']
        before = len(MOCKS['xi'].hits)
        foreign = self.http.delete(f'/v1/voices/{voice_id}', headers=xi(other))
        self.assertEqual(foreign.status_code, 404, foreign.text)
        self.assertEqual(len(MOCKS['xi'].hits), before)
        self.assertEqual(self.http.delete(f'/v1/voices/{voice_id}', headers=xi(owner)).status_code, 200)
        dub = self.http.post('/v1/dubbing', data={'target_lang': 'es', 'source_lang': 'en'},
                             files={'file': ('c.wav', W.WAV, 'audio/wav')}, headers=xi(owner))
        dubbing_id = dub.json()['dubbing_id']
        self.assertEqual(self.http.get(f'/v1/dubbing/{dubbing_id}', headers=xi(owner)).json()['status'], 'dubbed')
        self.assertEqual(self.http.get(f'/v1/dubbing/{dubbing_id}', headers=xi(other)).status_code, 404)
        no_store = self.key('native', 'cloud')
        refused = self.http.post('/v1/voices/add', data={'name': 'x'}, files={'files': ('s.wav', W.WAV, 'audio/wav')},
                                 headers=xi(no_store))
        self.assertEqual(refused.status_code, 403, refused.text)

    def test_unlisted_paths_are_refused_before_egress(self):
        token = self.key(*NATIVE)
        before = len(MOCKS['xi'].hits)
        answer = self.http.get('/native/xi/v1/convai/agents', headers=xi(token))
        self.assertEqual(answer.status_code, 400, answer.text)
        self.assertEqual(answer.json()['detail']['status'], 'broker_capability_unavailable')
        self.assertEqual(len(MOCKS['xi'].hits), before)

    def test_account_endpoints_need_broker_admin(self):
        token = self.key(*NATIVE)
        self.assertEqual(self.http.get('/v1/user/subscription', headers=xi(token)).status_code, 403)
        self.assertEqual(self.http.get('/v1/history', headers=xi(token)).status_code, 403)
        self.assertEqual(self.http.post('/v1/single-use-token/tts_websocket', headers=xi(token)).status_code, 403)
        admin = self.http.get('/v1/user/subscription', headers=W.bearer(BOOT))
        self.assertEqual(admin.status_code, 200, admin.text)
        self.assertEqual(admin.json()['character_limit'], 10000)
        self.assertEqual(self.http.get('/v1/history?page_size=1', headers=W.bearer(BOOT)).json()['history'][0]
                         ['history_item_id'], 'hist0001')



class ModelCapabilities(unittest.TestCase):
    def test_speech_and_transcription_models_are_told_apart(self):
        from titanos.broker.adapters.elevenlabs import ElevenLabsAdapter
        caps = frozenset({'speech', 'transcription'})
        self.assertEqual(ElevenLabsAdapter.model_capabilities('eleven_multilingual_v2', caps), {'speech'})
        self.assertEqual(ElevenLabsAdapter.model_capabilities('scribe_v1', caps), {'transcription'})
        self.assertEqual(ElevenLabsAdapter.model_capabilities('some_future_model', caps), caps)

if __name__ == '__main__':
    unittest.main(verbosity=2)
