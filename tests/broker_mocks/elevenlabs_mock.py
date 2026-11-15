"""ElevenLabs loopback mock.

Routes (api_base = base_url): POST /v1/text-to-speech/{voice_id} and /v1/text-to-speech/{voice_id}/stream (binary audio,
chunked, output_format query defaulting to mp3_44100_128, character-cost and request-id headers), POST /v1/speech-to-text
(multipart file | source_url | cloud_storage_url plus model_id; text, language_code and words[]; use_multi_channel=true
answers transcripts[] with audio_duration_secs; webhook=true answers 202 with a transcription id), GET /v1/voices,
GET /v1/voices/{voice_id}, GET /v1/models (a bare JSON array).

Completeness round (2026-10-02), the public endpoints the native passthrough reaches: POST
/v1/text-to-speech/{voice_id}/with-timestamps (JSON audio_base64 plus alignment) and /stream/with-timestamps (newline-
delimited JSON chunks); POST /v1/text-to-dialogue and /stream (binary), /with-timestamps (JSON with voice_segments) and
/stream/with-timestamps; POST /v1/sound-generation (binary; duration_seconds 0.5-30, prompt_influence 0-1); POST
/v1/speech-to-speech/{voice_id} and /stream and POST /v1/audio-isolation and /stream (multipart audio in, binary out);
POST /v1/forced-alignment (multipart file plus text); POST /v1/music and /v1/music/stream (binary with a song-id header;
exactly one of prompt or composition_plan) and /v1/music/plan (JSON); GET /v2/voices; GET /v1/voices/{voice_id}/settings
and /v1/voices/settings/default; POST /v1/voices/add (multipart name plus files; voice ids clonedvoice000000001, ...)
and DELETE /v1/voices/{voice_id}; POST /v1/text-to-voice/design and POST /v1/text-to-voice; GET /v1/user/subscription;
GET /v1/history; POST /v1/dubbing (dubmock0001, ...), GET and DELETE /v1/dubbing/{dubbing_id}; POST
/v1/pronunciation-dictionaries/add-from-rules (pdictmock0001, ...). Generated audio carries character-cost and
history-item-id headers. Routes added after construction (the tests/broker_mocks/native_routes.py installers) are matched
before these.

Auth: xi-api-key: <SENTINELS['elevenlabs']>; otherwise 401 {"detail": {"status": ..., "message": ...}}. A status fault with
code quota_exceeded or missing_permissions answers the same legacy envelope ElevenLabs sends for an exhausted quota or a
restricted-scope key (fixtures error_401_quota and error_401_permissions; error_envelope_v2 is the current documented
envelope with detail.type, code, message, status, request_id and param).
Validation errors use the FastAPI-shaped {"detail": [{"type", "loc", "msg", "input"}]} with 422.

output_format: the mp3_* and pcm_* families are served (fixture variants mp3 and pcm); any other value is refused with 400
invalid_output_format (ulaw, alaw and opus are real formats this mock does not serve).
"""
from __future__ import annotations

import json
import re

from .fixtures import load
from .server import BaseMock, MockRequest, Reply, json_reply, refuse

OUTPUT_FORMAT = re.compile(r'^(mp3_(22050_32|44100_(32|64|96|128|192))|pcm_(8000|16000|22050|24000|44100|48000))$')
DEFAULT_OUTPUT_FORMAT = 'mp3_44100_128'


class _Bad(Exception):
    """A refusal raised by a request helper; the route answers its reply."""

    def __init__(self, reply: Reply):
        super().__init__('mock refusal')
        self.reply = reply


def _true(value) -> bool:
    return isinstance(value, str) and value.strip().lower() in ('true', '1', 'yes')


def _guarded(handler):
    async def run(self, request: MockRequest) -> Reply:
        try:
            return await handler(self, request)
        except _Bad as bad:
            return bad.reply
    run.__name__ = handler.__name__
    return run


class ElevenLabsMock(BaseMock):
    PROVIDER = 'elevenlabs'
    PREFIX = ''

    def __init__(self, *, credential: str | None = None):
        super().__init__(credential=credential)
        self._sealed = False
        self.voices_added: dict = {}
        self.dubbings: dict = {}
        self.dictionaries: dict = {}
        self.route('POST', r'/v1/text-to-speech/(?P<voice_id>[^/]+)', self.tts, 'elevenlabs.tts')
        self.route('POST', r'/v1/text-to-speech/(?P<voice_id>[^/]+)/stream', self.tts, 'elevenlabs.tts_stream')
        self.route('POST', r'/v1/speech-to-text', self.stt, 'elevenlabs.stt')
        self.route('GET', r'/v1/voices', self.voices, 'elevenlabs.voices')
        self.route('GET', r'/v1/voices/(?P<voice_id>[^/]+)', self.voice, 'elevenlabs.voice')
        self.route('GET', r'/v1/models', self.models, 'elevenlabs.models')
        # completeness round: the endpoints the native passthrough reaches
        voice = r'(?P<voice_id>[^/]+)'
        self.route('POST', rf'/v1/text-to-speech/{voice}/with-timestamps', self.tts_timestamps, 'native:tts')
        self.route('POST', rf'/v1/text-to-speech/{voice}/stream/with-timestamps', self.tts_timestamps_stream,
                   'native:tts')
        self.route('POST', r'/v1/text-to-dialogue', self.dialogue, 'native:dialogue')
        self.route('POST', r'/v1/text-to-dialogue/stream', self.dialogue, 'native:dialogue')
        self.route('POST', r'/v1/text-to-dialogue/with-timestamps', self.dialogue_timestamps, 'native:dialogue')
        self.route('POST', r'/v1/text-to-dialogue/stream/with-timestamps', self.dialogue_timestamps_stream,
                   'native:dialogue')
        self.route('POST', r'/v1/sound-generation', self.sound, 'native:sound_effects')
        self.route('POST', rf'/v1/speech-to-speech/{voice}', self.audio_in_audio_out, 'native:voice_changer')
        self.route('POST', rf'/v1/speech-to-speech/{voice}/stream', self.audio_in_audio_out, 'native:voice_changer')
        self.route('POST', r'/v1/audio-isolation', self.audio_in_audio_out, 'native:audio_isolation')
        self.route('POST', r'/v1/audio-isolation/stream', self.audio_in_audio_out, 'native:audio_isolation')
        self.route('POST', r'/v1/forced-alignment', self.forced_alignment, 'native:forced_alignment')
        self.route('POST', r'/v1/music', self.music, 'native:music')
        self.route('POST', r'/v1/music/stream', self.music, 'native:music')
        self.route('POST', r'/v1/music/plan', self.music_plan, 'native:music')
        self.route('GET', r'/v2/voices', self.voices_v2, 'native:voices')
        self.route('GET', r'/v1/voices/settings/default', self.voice_settings, 'native:voices')
        self.route('GET', rf'/v1/voices/{voice}/settings', self.voice_settings, 'native:voices')
        self.route('POST', r'/v1/voices/add', self.voice_add, 'native:voices')
        self.route('DELETE', rf'/v1/voices/{voice}', self.voice_delete, 'native:voices')
        self.route('POST', r'/v1/text-to-voice/design', self.voice_design, 'native:voice_design')
        self.route('POST', r'/v1/text-to-voice', self.voice_create, 'native:voice_design')
        self.route('GET', r'/v1/user/subscription', self.subscription, 'native:account')
        self.route('GET', r'/v1/history', self.history, 'native:history')
        self.route('POST', r'/v1/dubbing', self.dubbing_create, 'native:dubbing')
        self.route('GET', r'/v1/dubbing/(?P<dubbing_id>[^/]+)', self.dubbing_get, 'native:dubbing')
        self.route('DELETE', r'/v1/dubbing/(?P<dubbing_id>[^/]+)', self.dubbing_delete, 'native:dubbing')
        self.route('POST', r'/v1/pronunciation-dictionaries/add-from-rules', self.dictionary_from_rules,
                   'native:pronunciation')
        self._sealed = True

    def route(self, method: str, path_regex: str, handler, surface: str | None = None, *, auth: bool = True):
        """Routes added after construction (the native_routes installers) are matched before the built-in ones."""
        super().route(method, path_regex, handler, surface, auth=auth)
        if getattr(self, '_sealed', False):
            self.routes.insert(0, self.routes.pop())

    def reset(self) -> None:
        super().reset()
        self.voices_added = {}
        self.dubbings = {}
        self.dictionaries = {}

    def error_reply(self, status: int, message: str, code: str | None = None) -> Reply:
        return json_reply({'detail': {'status': code or {401: 'invalid_api_key', 404: 'not_found', 429: 'too_many_requests',
                                                         500: 'internal_error', 503: 'service_unavailable'}.get(status, 'error'),
                                      'message': message}}, status)

    def auth_failure(self, request: MockRequest) -> Reply | None:
        token = request.headers.get('xi-api-key')
        if token is None:
            return self.error_reply(401, 'Neither authorization header nor xi-api-key received, please provide one.',
                                    'needs_authorization')
        if token.strip() != self.credential:
            return self.error_reply(401, 'Invalid API key', 'invalid_api_key')
        return None

    def stream_error_frame(self, message: str = '') -> bytes:
        return b''  # binary audio has no in-band error event

    def malformed_frame(self, stream_kind: str) -> bytes:
        return b'\x00not-audio\x00'

    @staticmethod
    def _missing(field: str, value=None) -> Reply:
        return json_reply({'detail': [{'type': 'missing', 'loc': ['body', field], 'msg': 'Field required',
                                       'input': value}]}, 422)

    async def tts(self, request: MockRequest) -> Reply:
        try:
            body = json.loads(request.body.decode('utf-8')) if request.body else None
        except (ValueError, UnicodeDecodeError):
            return json_reply({'detail': [{'type': 'json_invalid', 'loc': ['body'], 'msg': 'JSON decode error',
                                           'input': {}}]}, 422)
        if not isinstance(body, dict) or not body.get('text'):
            return self._missing('text', body)
        fmt = request.q('output_format') or DEFAULT_OUTPUT_FORMAT
        if not OUTPUT_FORMAT.match(fmt):
            return self.error_reply(400, f'Invalid output format: {fmt}', 'invalid_output_format')
        variant = 'mp3' if fmt.startswith('mp3') else 'pcm'
        return self.fixture_reply('tts', {}, variant=variant,
                                  extra_headers=[('character-cost', str(len(body['text']))),
                                                 ('history-item-id', 'histmocktts0001'),
                                                 ('current-concurrent-requests', '1'),
                                                 ('maximum-concurrent-requests', '5'),
                                                 ('x-mock-private', 'never-forwarded')])

    async def stt(self, request: MockRequest) -> Reply:
        try:
            form = await request.form()
        except Exception:
            return json_reply({'detail': [{'type': 'value_error', 'loc': ['body'], 'msg': 'Invalid multipart body',
                                           'input': None}]}, 422)
        if not form.get('model_id'):
            return self._missing('model_id')
        sources = [name for name in ('file', 'source_url', 'cloud_storage_url') if form.get(name) is not None]
        if not sources:
            return json_reply({'detail': {'status': 'invalid_parameters',
                                          'message': 'Must provide either file or cloud_storage_url parameter.'}}, 400)
        if len(sources) > 1:
            return json_reply({'detail': {'status': 'invalid_parameters',
                                          'message': 'Provide only one of file, source_url or cloud_storage_url.'}}, 400)
        if _true(form.get('webhook')):
            return self.fixture_reply('stt_webhook', {})
        if _true(form.get('use_multi_channel')):
            return self.fixture_reply('stt_multichannel', {})
        return self.fixture_reply('stt', {'model_id': str(form.get('model_id'))})

    async def voices(self, request: MockRequest) -> Reply:
        return self.fixture_reply('voices', {})

    async def voice(self, request: MockRequest) -> Reply:
        voice_id = request.path_params['voice_id']
        for item in load('elevenlabs', 'voices').body({})['voices']:
            if item['voice_id'] == voice_id:
                return json_reply(item)
        return self.error_reply(404, f'A voice with the voice_id {voice_id} was not found.', 'voice_not_found')

    async def models(self, request: MockRequest) -> Reply:
        return self.fixture_reply('models', {})

    # -- completeness round ----------------------------------------------------------------------------------------------
    def _json_body(self, request: MockRequest) -> dict:
        try:
            body = json.loads(request.body.decode('utf-8')) if request.body else None
        except (ValueError, UnicodeDecodeError):
            raise _Bad(json_reply({'detail': [{'type': 'json_invalid', 'loc': ['body'], 'msg': 'JSON decode error',
                                               'input': {}}]}, 422)) from None
        if not isinstance(body, dict):
            raise _Bad(self._missing('body', body))
        return body

    def _format(self, request: MockRequest) -> str:
        fmt = request.q('output_format') or DEFAULT_OUTPUT_FORMAT
        if not OUTPUT_FORMAT.match(fmt):
            raise _Bad(self.error_reply(400, f'Invalid output format: {fmt}', 'invalid_output_format'))
        return fmt

    def _audio(self, fmt: str, characters: int, *headers) -> Reply:
        variant = 'mp3' if fmt.startswith('mp3') else 'pcm'
        return self.fixture_reply('tts', {}, variant=variant,
                                  extra_headers=[('character-cost', str(characters)),
                                                 ('history-item-id', f'histmock{self.next_id():04d}'), *headers])

    def _dialogue_characters(self, body: dict) -> int:
        inputs = body.get('inputs')
        if not isinstance(inputs, list) or not inputs or not all(
                isinstance(item, dict) and item.get('text') and item.get('voice_id') for item in inputs):
            raise _Bad(self._missing('inputs', inputs))
        return sum(len(item['text']) for item in inputs)

    @_guarded
    async def tts_timestamps(self, request: MockRequest) -> Reply:
        body = self._json_body(request)
        if not body.get('text'):
            return self._missing('text', body)
        self._format(request)
        return self.fixture_reply('tts_with_timestamps', {}, extra_headers=[('character-cost', str(len(body['text'])))])

    @_guarded
    async def tts_timestamps_stream(self, request: MockRequest) -> Reply:
        body = self._json_body(request)
        if not body.get('text'):
            return self._missing('text', body)
        self._format(request)
        return self.fixture_reply('tts_stream_with_timestamps', {},
                                  extra_headers=[('character-cost', str(len(body['text'])))])

    @_guarded
    async def dialogue(self, request: MockRequest) -> Reply:
        characters = self._dialogue_characters(self._json_body(request))
        return self._audio(self._format(request), characters)

    @_guarded
    async def dialogue_timestamps(self, request: MockRequest) -> Reply:
        characters = self._dialogue_characters(self._json_body(request))
        self._format(request)
        return self.fixture_reply('dialogue_with_timestamps', {}, extra_headers=[('character-cost', str(characters))])

    @_guarded
    async def dialogue_timestamps_stream(self, request: MockRequest) -> Reply:
        characters = self._dialogue_characters(self._json_body(request))
        self._format(request)
        return self.fixture_reply('tts_stream_with_timestamps', {}, extra_headers=[('character-cost', str(characters))])

    @_guarded
    async def sound(self, request: MockRequest) -> Reply:
        body = self._json_body(request)
        if not body.get('text'):
            return self._missing('text', body)
        duration = body.get('duration_seconds')
        if duration is not None and not (isinstance(duration, (int, float)) and 0.5 <= duration <= 30):
            return self.error_reply(400, 'duration_seconds must be between 0.5 and 30', 'invalid_parameters')
        influence = body.get('prompt_influence')
        if influence is not None and not (isinstance(influence, (int, float)) and 0 <= influence <= 1):
            return self.error_reply(400, 'prompt_influence must be between 0 and 1', 'invalid_parameters')
        return self._audio(self._format(request), len(body['text']))

    @_guarded
    async def audio_in_audio_out(self, request: MockRequest) -> Reply:
        try:
            form = await request.form()
        except Exception:
            return self._missing('audio')
        upload = form.get('audio')
        if upload is None or isinstance(upload, str):
            return self._missing('audio')
        data = await upload.read()
        return self._audio(self._format(request), max(1, len(data) // 100))

    async def forced_alignment(self, request: MockRequest) -> Reply:
        form = await request.form()
        if form.get('file') is None or isinstance(form.get('file'), str):
            return self._missing('file')
        if not form.get('text'):
            return self._missing('text')
        return self.fixture_reply('forced_alignment', {})

    @_guarded
    async def music(self, request: MockRequest) -> Reply:
        body = self._json_body(request)
        if bool(body.get('prompt')) == bool(body.get('composition_plan')):
            return self.error_reply(400, 'Provide exactly one of prompt or composition_plan', 'invalid_parameters')
        length = body.get('music_length_ms')
        if length is not None and not (isinstance(length, int) and 3000 <= length <= 300000):
            return self.error_reply(400, 'music_length_ms must be between 3000 and 300000', 'invalid_parameters')
        return self._audio(self._format(request), 0, ('song-id', f'songmock{self.next_id():04d}'))

    @_guarded
    async def music_plan(self, request: MockRequest) -> Reply:
        body = self._json_body(request)
        if not body.get('prompt'):
            return self._missing('prompt', body)
        return self.fixture_reply('music_plan', {})

    async def voices_v2(self, request: MockRequest) -> Reply:
        size = request.q('page_size')
        if size is not None and not (size.isdigit() and 1 <= int(size) <= 100):
            return self.error_reply(400, 'page_size must be between 1 and 100', 'invalid_page_size')
        return self.fixture_reply('voices_v2', {})

    async def voice_settings(self, request: MockRequest) -> Reply:
        return self.fixture_reply('voice_settings', {})

    async def voice_add(self, request: MockRequest) -> Reply:
        form = await request.form()
        files = [item for item in form.getlist('files') if not isinstance(item, str)]
        if not form.get('name'):
            return self._missing('name')
        if not files:
            return self._missing('files')
        voice_id = f'clonedvoice{self.next_id():09d}'
        self.voices_added[voice_id] = str(form.get('name'))
        return self.fixture_reply('voice_add', {'voice_id': voice_id})

    async def voice_delete(self, request: MockRequest) -> Reply:
        voice_id = request.path_params['voice_id']
        if self.voices_added.pop(voice_id, None) is None:
            return self.error_reply(404, f'A voice with the voice_id {voice_id} was not found.', 'voice_not_found')
        return json_reply({'status': 'ok'})

    @_guarded
    async def voice_design(self, request: MockRequest) -> Reply:
        body = self._json_body(request)
        if not body.get('voice_description'):
            return self._missing('voice_description', body)
        return self.fixture_reply('voice_design', {})

    @_guarded
    async def voice_create(self, request: MockRequest) -> Reply:
        body = self._json_body(request)
        for name in ('voice_name', 'voice_description', 'generated_voice_id'):
            if not body.get(name):
                return self._missing(name, body)
        voice_id = f'designedvoice{self.next_id():07d}'
        self.voices_added[voice_id] = body['voice_name']
        return json_reply({'voice_id': voice_id, 'name': body['voice_name'], 'category': 'generated',
                           'description': body['voice_description'], 'labels': {}})

    async def subscription(self, request: MockRequest) -> Reply:
        return self.fixture_reply('user_subscription', {})

    async def history(self, request: MockRequest) -> Reply:
        return self.fixture_reply('history', {})

    async def dubbing_create(self, request: MockRequest) -> Reply:
        form = await request.form()
        if not form.get('target_lang'):
            return self._missing('target_lang')
        if form.get('file') is None and not form.get('source_url'):
            return self.error_reply(400, 'Provide a file or a source_url', 'invalid_parameters')
        dubbing_id = f'dubmock{self.next_id():04d}'
        self.dubbings[dubbing_id] = str(form.get('target_lang'))
        return self.fixture_reply('dubbing_create', {'dubbing_id': dubbing_id})

    async def dubbing_get(self, request: MockRequest) -> Reply:
        dubbing_id = request.path_params['dubbing_id']
        if dubbing_id not in self.dubbings:
            return self.error_reply(404, f'Dubbing {dubbing_id} not found', 'dubbing_not_found')
        return self.fixture_reply('dubbing_get', {'dubbing_id': dubbing_id})

    async def dubbing_delete(self, request: MockRequest) -> Reply:
        if self.dubbings.pop(request.path_params['dubbing_id'], None) is None:
            return self.error_reply(404, 'Dubbing not found', 'dubbing_not_found')
        return json_reply({'status': 'ok'})

    @_guarded
    async def dictionary_from_rules(self, request: MockRequest) -> Reply:
        body = self._json_body(request)
        if not body.get('name') or not isinstance(body.get('rules'), list):
            return self._missing('rules', body)
        dictionary_id = f'pdictmock{self.next_id():04d}'
        self.dictionaries[dictionary_id] = body['name']
        return self.fixture_reply('pronunciation_add_rules', {'dictionary_id': dictionary_id})


__all__ = ['DEFAULT_OUTPUT_FORMAT', 'ElevenLabsMock', 'OUTPUT_FORMAT']
