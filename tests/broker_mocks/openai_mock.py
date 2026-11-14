"""OpenAI-protocol loopback mock (profile openai and the openai_compat rows), with an xAI-like variant (JPEG image output)
and a Mistral-like variant (9-character alphanumeric tool-call ids enforced, Mistral error envelopes).

Surfaces (relative to api_base = base_url + '/v1'): chat/completions (stream and non-stream, tools, tool_calls deltas, the
include_usage chunk with choices:[]), completions, responses (create with the full stream event family and
sequence_number, retrieve, delete, cancel, input_items, previous_response_id), embeddings (float and base64), audio/speech
(binary, chunked, per response_format), audio/transcriptions and audio/translations (multipart; json, text, verbose_json),
images/generations and images/edits, moderations, models. Auth: Authorization: Bearer <SENTINELS[provider]>; otherwise
OpenAI's native 401 envelope.

Images model the current OpenAI API (live observation 2026-10-02: dall-e-2 with response_format b64_json answered 400
unknown_parameter, as gpt-image-* already did): every model refuses response_format on images/generations and
images/edits with 400 {"error": {"message": "Unknown parameter: 'response_format'.", "type": "invalid_request_error",
"param": "response_format", "code": "unknown_parameter"}} and answers b64_json PNG payloads by default. The pre-2026-10
behaviour (dall-e answers url by default and accepts response_format b64_json or url; gpt-image-* refuses the field) is
kept for OpenAI-compatible upstreams that still accept the field: AzureMock (Azure deployments) and
OpenAIMock(images_accept_response_format=True). The xAI variant accepts url and b64_json (JPEG payloads).

Answers replay tests/fixtures/providers/openai/* (variant fixtures override by case name). mock.respond_with(surface, case)
queues a fixture case for the next request on that surface.

Media streams (2026-10-02, the public streaming references): images/generations and images/edits with stream=true answer
image_generation.* / image_edit.* events (images_stream, images_edits_stream); audio/transcriptions with stream=true
answers transcript.text.delta / transcript.text.done events (transcriptions_stream) except on whisper-1, which ignores
the flag and answers JSON as the provider documents; audio/speech with stream_format=sse answers speech.audio.delta /
speech.audio.done events (speech_sse); GET /responses/{id}?stream=true replays the stored response's event stream.
Error envelopes for the error-mapping checks are fixture cases selected with respond_with (error_403_verification,
error_429_insufficient_quota, error_400_param).
"""
from __future__ import annotations

import base64
import re
import struct
from pathlib import Path

from .fixtures import FIXTURES_DIR, dumps, load, substitute
from .server import BaseMock, MockRequest, Reply, json_reply, refuse

SPEECH_FORMATS = ('mp3', 'opus', 'aac', 'flac', 'wav', 'pcm')
TRANSCRIPTION_FORMATS = ('json', 'text', 'verbose_json')
_TRUE = ('true', '1')


def _flag(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in _TRUE)
ALNUM9 = re.compile(r'^[A-Za-z0-9]{9}$')
ALL_SURFACES = ('openai.chat', 'openai.completions', 'openai.responses', 'openai.responses.retrieve',
                'openai.responses.delete', 'openai.responses.cancel', 'openai.responses.input_items', 'openai.embeddings',
                'openai.audio.speech', 'openai.audio.transcriptions', 'openai.audio.translations',
                'openai.images.generations', 'openai.images.edits', 'openai.moderations', 'openai.models')
_ERROR_TYPES = {400: 'invalid_request_error', 401: 'invalid_request_error', 403: 'permission_error',
                404: 'invalid_request_error', 409: 'invalid_request_error', 413: 'invalid_request_error',
                422: 'invalid_request_error', 429: 'requests', 500: 'server_error', 502: 'server_error',
                503: 'server_error', 529: 'server_error'}


def mask_key(token: str) -> str:
    """OpenAI's echo of a rejected key: the first 8 and last 4 characters around asterisks."""
    if len(token) <= 12:
        return token[:2] + '*' * max(0, len(token) - 2)
    return token[:8] + '*' * (len(token) - 12) + token[-4:]


class OpenAIMock(BaseMock):
    PROVIDER = 'openai'
    PREFIX = '/v1'
    VARIANTS = ('openai', 'xai', 'mistral')

    # the current OpenAI Images API refuses response_format for every model (module docstring); True models an
    # OpenAI-compatible upstream with the pre-2026-10 behaviour
    IMAGES_ACCEPT_RESPONSE_FORMAT = False

    def __init__(self, *, variant: str = 'openai', credential: str | None = None, models: list[str] | None = None,
                 known_models: list[str] | None = None, surfaces=None, prefix: str | None = None,
                 require_auth: bool = True, images_accept_response_format: bool | None = None):
        if variant not in self.VARIANTS:
            raise ValueError(f'variant must be one of {self.VARIANTS}')
        self.variant = variant
        if images_accept_response_format is not None:
            self.IMAGES_ACCEPT_RESPONSE_FORMAT = images_accept_response_format
        if variant != 'openai':
            self.PROVIDER = variant
        if prefix is not None:
            self.PREFIX = prefix
        super().__init__(credential=credential)
        self.require_auth = require_auth
        self.models = models
        self.known_models = set(known_models) if known_models else None
        self.overrides: dict[str, list[str]] = {}
        self.responses: dict[str, dict] = {}
        self.enabled = set(surfaces) if surfaces is not None else set(ALL_SURFACES)
        self._register()

    # routing -------------------------------------------------------------------------------------------------------
    def _register(self) -> None:
        p = re.escape(self.PREFIX)
        rid = r'(?P<response_id>[^/]+)'
        table = [
            ('POST', p + r'/chat/completions', self.chat, 'openai.chat'),
            ('POST', p + r'/completions', self.completions, 'openai.completions'),
            ('POST', p + r'/responses', self.responses_create, 'openai.responses'),
            ('GET', p + r'/responses/' + rid, self.responses_retrieve, 'openai.responses.retrieve'),
            ('DELETE', p + r'/responses/' + rid, self.responses_delete, 'openai.responses.delete'),
            ('POST', p + r'/responses/' + rid + r'/cancel', self.responses_cancel, 'openai.responses.cancel'),
            ('GET', p + r'/responses/' + rid + r'/input_items', self.responses_input_items, 'openai.responses.input_items'),
            ('POST', p + r'/embeddings', self.embeddings, 'openai.embeddings'),
            ('POST', p + r'/audio/speech', self.speech, 'openai.audio.speech'),
            ('POST', p + r'/audio/transcriptions', self.transcriptions, 'openai.audio.transcriptions'),
            ('POST', p + r'/audio/translations', self.translations, 'openai.audio.translations'),
            ('POST', p + r'/images/generations', self.images_generations, 'openai.images.generations'),
            ('POST', p + r'/images/edits', self.images_edits, 'openai.images.edits'),
            ('POST', p + r'/moderations', self.moderations, 'openai.moderations'),
            ('GET', p + r'/models', self.models_list, 'openai.models'),
            ('GET', p + r'/models/(?P<model>.+)', self.models_get, 'openai.models'),
        ]
        for method, regex, handler, surface in table:
            if surface in self.enabled:
                self.route(method, regex, handler, surface)

    def reset(self) -> None:
        super().reset()
        self.responses.clear()
        self.overrides.clear()

    def respond_with(self, surface: str, case: str) -> None:
        """Answer the next request on `surface` with fixture `case` (FIFO)."""
        self.overrides.setdefault(surface, []).append(case)

    def _case(self, surface: str, default: str) -> str:
        queue = self.overrides.get(surface)
        return queue.pop(0) if queue else default

    def fixture_provider_for(self, case: str) -> str:
        if self.variant != 'openai' and (FIXTURES_DIR / self.variant / (case + '.jsonl')).is_file():
            return self.variant
        return 'openai'

    def reply(self, case: str, variables: dict | None = None, **kw) -> Reply:
        return self.fixture_reply(case, variables, provider=kw.pop('provider', None) or self.fixture_provider_for(case),
                                  **kw)

    # provider envelopes -----------------------------------------------------------------------------------------------
    def error_reply(self, status: int, message: str, code: str | None = None, *, param: str | None = None,
                    type_: str | None = None) -> Reply:
        if self.variant == 'mistral':
            return json_reply({'object': 'error', 'message': message, 'type': type_ or _ERROR_TYPES.get(status, 'api_error'),
                               'param': param, 'code': code}, status)
        return json_reply({'error': {'message': message, 'type': type_ or _ERROR_TYPES.get(status, 'server_error'),
                                     'param': param, 'code': code}}, status)

    def auth_failure(self, request: MockRequest) -> Reply | None:
        token = self.presented_token(request, 'authorization')
        if token is None:
            if not self.require_auth:
                return None
            if self.variant == 'mistral':
                return json_reply({'message': 'Unauthorized', 'request_id': 'mock-unauthorized'}, 401)
            return self.error_reply(401, "You didn't provide an API key. You need to provide your API key in an "
                                         'Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).')
        if token != self.credential:
            if self.variant == 'mistral':
                return json_reply({'message': 'Unauthorized', 'request_id': 'mock-unauthorized'}, 401)
            return self.error_reply(401, f'Incorrect API key provided: {mask_key(token)}.', 'invalid_api_key')
        return None

    def stream_error_frame(self, message: str = 'The server had an error while processing your request. Sorry about that!') -> bytes:
        payload = {'error': {'message': message, 'type': 'server_error', 'param': None, 'code': None}}
        return ('data: ' + dumps(payload) + '\n\n').encode('utf-8')

    # request helpers ---------------------------------------------------------------------------------------------------
    def json_body(self, request: MockRequest) -> dict:
        try:
            body = request.json()
        except (ValueError, UnicodeDecodeError):
            refuse(self.error_reply(400, 'We could not parse the JSON body of your request. (HINT: This likely means you '
                                         "aren't using your HTTP library correctly.)"))
        if not isinstance(body, dict):
            refuse(self.error_reply(400, 'The request body must be a JSON object.'))
        return body

    def request_model(self, request: MockRequest, body) -> str:
        model = body.get('model') if body is not None else None
        if not model or not isinstance(model, str):
            refuse(self.error_reply(400, 'you must provide a model parameter', param='model'))
        if self.known_models is not None and model not in self.known_models:
            refuse(self.error_reply(404, f'The model `{model}` does not exist or you do not have access to it.',
                                    'model_not_found'))
        return model

    async def form_fields(self, request: MockRequest) -> dict:
        try:
            form = await request.form()
        except Exception:
            refuse(self.error_reply(400, 'Could not parse multipart form'))
        return form

    def _require(self, body: dict, field: str) -> None:
        if body.get(field) in (None, '', []):
            refuse(self.error_reply(400, f"Missing required parameter: '{field}'.", 'missing_required_parameter',
                                    param=field))

    def _check_tool_ids(self, body: dict) -> None:
        """Mistral-like: every tool-call id in the request must be exactly 9 alphanumerics."""
        for message in body.get('messages') or []:
            ids = [call.get('id') for call in (message.get('tool_calls') or []) if isinstance(call, dict)]
            if message.get('role') == 'tool':
                ids.append(message.get('tool_call_id'))
            for value in ids:
                if not isinstance(value, str) or not ALNUM9.match(value):
                    refuse(self.error_reply(422, f'Tool call id was {value} but must be a-z, A-Z, 0-9, with a length of 9.',
                                            type_='invalid_request_message_order'))

    # chat and completions ------------------------------------------------------------------------------------------------
    async def chat(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        self._require(body, 'messages')
        if self.variant == 'mistral':
            self._check_tool_ids(body)
        stream = bool(body.get('stream'))
        tools = bool(body.get('tools')) and body.get('tool_choice') != 'none'
        include_usage = bool((body.get('stream_options') or {}).get('include_usage')) if stream else False
        default = ('chat_stream_tools' if tools else 'chat_stream') if stream else ('chat_tools' if tools else 'chat_basic')
        if include_usage:
            default += '_usage'
        return self.reply(self._case('openai.chat', default), {'model': model, 'include_usage': include_usage})

    async def completions(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        self._require(body, 'prompt')
        stream = bool(body.get('stream'))
        include_usage = bool((body.get('stream_options') or {}).get('include_usage')) if stream else False
        default = 'completions_stream' if stream else 'completions'
        return self.reply(self._case('openai.completions', default), {'model': model, 'include_usage': include_usage})

    # responses ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _input_items(response_id: str, value) -> list:
        if isinstance(value, str):
            value = [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': value}]}]
        items = []
        for index, item in enumerate(value or []):
            item = dict(item) if isinstance(item, dict) else {'type': 'message', 'role': 'user', 'content': str(item)}
            item.setdefault('id', f'msg_in_{response_id[5:]}_{index}')
            item.setdefault('status', 'completed')
            items.append(item)
        return items

    async def responses_create(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        self._require(body, 'input')
        previous = body.get('previous_response_id')
        if previous is not None and previous not in self.responses:
            return self.error_reply(400, f"Previous response with id '{previous}' not found.",
                                    'previous_response_not_found', param='previous_response_id')
        response_id = f'resp_mock{self.next_id():04d}'
        store = body.get('store', True) is not False
        background = bool(body.get('background'))
        stream = bool(body.get('stream'))
        tools = bool(body.get('tools')) and body.get('tool_choice') != 'none'
        default = ('responses_stream_tools' if tools else 'responses_stream') if stream else \
            ('responses_tools' if tools else 'responses_create')
        case = self._case('openai.responses', default)
        variables = {'model': model, 'response_id': response_id, 'previous_response_id': previous, 'store': store,
                     'background': background}
        fx = load(self.fixture_provider_for(case), case)
        if fx.kind == 'json':
            final = fx.body(variables)
        else:  # the response object of the terminal response.* event
            final = None
            for row in fx.rows:
                data = row.get('data')
                if isinstance(data, dict) and data.get('type') in ('response.completed', 'response.incomplete',
                                                                    'response.failed'):
                    final = substitute(data['response'], variables)
        if store and final is not None:
            stored = dict(final)
            if background:
                stored['status'] = 'in_progress'
            self.responses[response_id] = {'object': stored, 'input': self._input_items(response_id, body['input']),
                                           'background': background}
        if background and not stream:
            obj = dict(final)
            obj['status'] = 'in_progress'
            return json_reply(obj, 200, [('x-request-id', 'req_mock_background')])
        return self.reply(case, variables)

    def _stored(self, request: MockRequest) -> dict:
        response_id = request.path_params['response_id']
        entry = self.responses.get(response_id)
        if entry is None:
            refuse(self.error_reply(404, f"Response with id '{response_id}' not found.", param='response_id'))
        return entry

    async def responses_retrieve(self, request: MockRequest) -> Reply:
        entry = self._stored(request)
        if _flag(request.q('stream')):
            stored = entry['object']
            return self.reply('responses_stream', {'model': stored.get('model'), 'response_id': stored.get('id'),
                                                   'previous_response_id': stored.get('previous_response_id'),
                                                   'store': True, 'background': entry['background']})
        return json_reply(entry['object'])

    async def responses_delete(self, request: MockRequest) -> Reply:
        self._stored(request)
        response_id = request.path_params['response_id']
        del self.responses[response_id]
        return json_reply({'id': response_id, 'object': 'response', 'deleted': True})

    async def responses_cancel(self, request: MockRequest) -> Reply:
        entry = self._stored(request)
        if not entry['background']:
            return self.error_reply(400, 'Only background responses can be cancelled.')
        entry['object'] = dict(entry['object'], status='cancelled')
        return json_reply(entry['object'])

    async def responses_input_items(self, request: MockRequest) -> Reply:
        items = self._stored(request)['input']
        order = request.q('order', 'desc')
        data = list(reversed(items)) if order == 'desc' else list(items)
        return json_reply({'object': 'list', 'data': data, 'first_id': data[0]['id'] if data else None,
                           'last_id': data[-1]['id'] if data else None, 'has_more': False})

    # embeddings and moderations ---------------------------------------------------------------------------------------------
    async def embeddings(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        self._require(body, 'input')
        value = body['input']
        count = len(value) if isinstance(value, list) and value and not isinstance(value[0], int) else 1
        fx = load(self.fixture_provider_for('embeddings'), self._case('openai.embeddings', 'embeddings'))
        variables = {'model': model}
        out = fx.body(variables)
        template = out['data']
        out['data'] = [dict(template[i % len(template)], index=i) for i in range(count)]
        if body.get('encoding_format') == 'base64':
            for item in out['data']:
                item['embedding'] = base64.b64encode(struct.pack('<%df' % len(item['embedding']),
                                                                 *item['embedding'])).decode('ascii')
        if count == 1 and body.get('encoding_format') != 'base64':
            return self.reply(fx.case, variables)
        return json_reply(out, 200, [(k, v) for k, v in fx.headers.items() if k != 'content-type'])

    async def moderations(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        self._require(body, 'input')
        model = body.get('model') or 'omni-moderation-latest'
        value = body['input']
        count = len(value) if isinstance(value, list) and value and isinstance(value[0], str) else 1
        fx = load('openai', 'moderations')
        if count == 1:
            return self.reply('moderations', {'model': model})
        out = fx.body({'model': model})
        out['results'] = [out['results'][0] for _ in range(count)]
        return json_reply(out)

    # audio -----------------------------------------------------------------------------------------------------------------
    async def speech(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        self.request_model(request, body)
        self._require(body, 'input')
        self._require(body, 'voice')
        fmt = body.get('response_format', 'mp3')
        if fmt not in SPEECH_FORMATS:
            return self.error_reply(400, f"Invalid value: '{fmt}'. Supported values are: {', '.join(SPEECH_FORMATS)}.",
                                    'invalid_value', param='response_format')
        if body.get('stream_format') == 'sse':
            return self.reply(self._case('openai.audio.speech', 'speech_sse'), {})
        return self.reply(self._case('openai.audio.speech', 'speech'), {}, variant=fmt)

    async def _audio(self, request: MockRequest, task: str) -> Reply:
        form = await self.form_fields(request)
        if form.get('file') is None:
            return self.error_reply(400, "Missing required parameter: 'file'.", 'missing_required_parameter', param='file')
        model = form.get('model')
        if not model:
            return self.error_reply(400, "Missing required parameter: 'model'.", 'missing_required_parameter',
                                    param='model')
        if task == 'transcriptions' and _flag(form.get('stream')) and str(model) != 'whisper-1':
            return self.reply(self._case('openai.audio.transcriptions', 'transcriptions_stream'), {'model': str(model)})
        fmt = form.get('response_format') or 'json'
        if fmt not in TRANSCRIPTION_FORMATS:
            return self.error_reply(400, f"mock: response_format '{fmt}' is not implemented", param='response_format')
        surface = 'openai.audio.' + task
        return self.reply(self._case(surface, f'{task}_{fmt}'), {'model': str(model)})

    async def transcriptions(self, request: MockRequest) -> Reply:
        return await self._audio(request, 'transcriptions')

    async def translations(self, request: MockRequest) -> Reply:
        return await self._audio(request, 'translations')

    # images ----------------------------------------------------------------------------------------------------------------
    def _image_case(self, model: str, response_format) -> str:
        if self.variant == 'xai':
            return 'images_url' if response_format == 'url' else 'images_b64'
        if model.startswith('gpt-image') or not self.IMAGES_ACCEPT_RESPONSE_FORMAT:
            if response_format is not None:
                refuse(self.error_reply(400, "Unknown parameter: 'response_format'.", 'unknown_parameter',
                                        param='response_format'))
            return 'images_gpt_image' if model.startswith('gpt-image') else 'images_dalle_b64'
        return 'images_dalle_b64' if response_format == 'b64_json' else 'images_dalle_url'

    def _images(self, surface: str, case: str, model: str, n) -> Reply:
        try:
            count = max(1, min(10, int(n or 1)))
        except (TypeError, ValueError):
            refuse(self.error_reply(400, "Invalid type for 'n'.", param='n'))
        case = self._case(surface, case)
        provider = self.fixture_provider_for(case)
        if count == 1:
            return self.reply(case, {'model': model})
        fx = load(provider, case)
        out = fx.body({'model': model})
        out['data'] = [out['data'][0] for _ in range(count)]
        return json_reply(out)

    async def images_generations(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = body.get('model') or 'dall-e-2'
        if self.known_models is not None and model not in self.known_models:
            self.request_model(request, body)
        self._require(body, 'prompt')
        case = self._image_case(model, body.get('response_format'))
        if _flag(body.get('stream')) and self.variant == 'openai':
            return self.reply(self._case('openai.images.generations', 'images_stream'), {'model': model})
        return self._images('openai.images.generations', case, model, body.get('n'))

    async def images_edits(self, request: MockRequest) -> Reply:
        form = await self.form_fields(request)
        if form.get('image') is None and form.get('image[]') is None:
            return self.error_reply(400, "Missing required parameter: 'image'.", 'missing_required_parameter', param='image')
        if not form.get('prompt'):
            return self.error_reply(400, "Missing required parameter: 'prompt'.", 'missing_required_parameter',
                                    param='prompt')
        model = str(form.get('model') or 'dall-e-2')
        case = self._image_case(model, form.get('response_format'))  # refuses response_format exactly as generations
        if _flag(form.get('stream')) and self.variant == 'openai':
            return self.reply(self._case('openai.images.edits', 'images_edits_stream'), {'model': model})
        return self._images('openai.images.edits', 'images_edits' if model.startswith('gpt-image') else case, model,
                            form.get('n'))

    # catalog ---------------------------------------------------------------------------------------------------------------
    def _catalog(self) -> dict:
        fx = load(self.fixture_provider_for('models'), 'models')
        out = fx.body({})
        if self.models is not None:
            template = out['data'][0]
            out['data'] = [dict(template, id=model) for model in self.models]
        return out

    async def models_list(self, request: MockRequest) -> Reply:
        if self.models is None:
            return self.reply('models', {})
        return json_reply(self._catalog())

    async def models_get(self, request: MockRequest) -> Reply:
        name = request.path_params['model']
        for item in self._catalog()['data']:
            if item['id'] == name:
                return json_reply(item)
        return self.error_reply(404, f"The model '{name}' does not exist", 'model_not_found', param='model')


__all__ = ['ALL_SURFACES', 'OpenAIMock', 'SPEECH_FORMATS', 'TRANSCRIPTION_FORMATS', 'mask_key']
