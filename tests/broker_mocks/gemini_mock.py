"""Gemini (Generative Language API v1beta) loopback mock.

Routes (api_base = base_url + '/v1beta'): POST models/{model}:generateContent, :streamGenerateContent (alt=sse gives SSE
'data:' frames, each a whole GenerateContentResponse, no event names and no [DONE]; without alt=sse the chunks come back as
one JSON array, as the API does), :embedContent, :batchEmbedContents, :countTokens; GET models (paginated with
pageSize/pageToken, nextPageToken) and GET models/{model}.

Auth: x-goog-api-key: <SENTINELS['gemini']>. Native answers follow the API's google.rpc.Status envelope
{"error": {"code", "message", "status", "details"?}}: a wrong key gives 400 INVALID_ARGUMENT with ErrorInfo reason
API_KEY_INVALID and a missing key 403 PERMISSION_DENIED, which is what the API returns for API keys (it has no 401 for
them). A ?key= query credential is refused by this mock with 401 UNAUTHENTICATED: the broker must never send one.

Completeness cases (respond_with(surface, case), fixtures transcribed from the public guides): generate_content_image
(an image model's text plus inlineData image/png), generate_content_tts (inlineData audio/L16;codec=pcm;rate=24000),
generate_content_transcript (audio understanding), generate_content_code_execution (executableCode,
codeExecutionResult, groundingMetadata), stream_generate_content_image (an image frame whose base64 comes from the
image_b64 variable, so a test can make it larger than 1 MiB through mock.extra_vars) and models_families (a catalog with
TTS, image, embedding, Imagen and Veo models). install_completeness_routes() adds the native-only endpoints the
completeness checks reach: the file media download (/download/v1beta/files/{id}:download, and the redirecting
/v1beta/files/{id}:download form), Interactions (create, also streamed; get; delete; cancel), File Search stores (create,
list, get, :importFile, documents), agents, webhooks and the ephemeral auth_tokens.

GeminiMock(sse_crlf=True) frames alt=sse answers (and the in-stream error frame) with CRLF line endings instead of LF. The
public endpoint is reported to frame its SSE with CRLF; the fixtures (transcribed offline) use LF, so streaming tests
should run both framings until a live recording pins it.
"""
from __future__ import annotations

import json

from .fixtures import dumps, load, substitute
from .server import BaseMock, MockRequest, Reply, json_reply, refuse

_STATUS = {400: 'INVALID_ARGUMENT', 401: 'UNAUTHENTICATED', 403: 'PERMISSION_DENIED', 404: 'NOT_FOUND', 409: 'ABORTED',
           429: 'RESOURCE_EXHAUSTED', 499: 'CANCELLED', 500: 'INTERNAL', 502: 'UNAVAILABLE', 503: 'UNAVAILABLE',
           504: 'DEADLINE_EXCEEDED', 529: 'UNAVAILABLE'}
ACTIONS = ('generateContent', 'streamGenerateContent', 'embedContent', 'batchEmbedContents', 'countTokens')
# default values of the completeness fixtures' variables (a 1x1 PNG; tests override image_b64 through mock.extra_vars)
DEFAULT_VARS = {'image_b64': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='}


def count_total_tokens(body: dict) -> int:
    """The mock's deterministic token estimate over contents (and systemInstruction): one per 4 characters of compact JSON."""
    return max(1, len(dumps({'contents': body.get('contents'), 'systemInstruction': body.get('systemInstruction')})) // 4)


class GeminiMock(BaseMock):
    PROVIDER = 'gemini'
    PREFIX = '/v1beta'

    def __init__(self, *, credential: str | None = None, sse_crlf: bool = False):
        super().__init__(credential=credential)
        self.sse_crlf = sse_crlf
        self.overrides: dict[str, list[str]] = {}
        self.extra_vars: dict = dict(DEFAULT_VARS)
        self.native_objects: dict = {}
        for action in ACTIONS:  # one route per action, so hits and faults carry the surface name
            self.route('POST', r'/v1beta/models/(?P<model>[^/:]+):(?P<action>' + action + ')', self.action,
                       'gemini.' + action)
        self.route('GET', r'/v1beta/models', self.models_list, 'gemini.models')
        self.route('GET', r'/v1beta/models/(?P<model>[^/:]+)', self.models_get, 'gemini.model')

    def reset(self) -> None:
        super().reset()
        self.overrides.clear()
        self.extra_vars = dict(DEFAULT_VARS)
        self.native_objects = {}

    def respond_with(self, surface: str, case: str) -> None:
        self.overrides.setdefault(surface, []).append(case)

    def _case(self, surface: str, default: str) -> str:
        queue = self.overrides.get(surface)
        return queue.pop(0) if queue else default

    def error_reply(self, status: int, message: str, code: str | None = None) -> Reply:
        return json_reply({'error': {'code': status, 'message': message, 'status': code or _STATUS.get(status, 'UNKNOWN')}},
                          status)

    def auth_failure(self, request: MockRequest) -> Reply | None:
        if any(name == 'key' for name, _ in request.query):
            return self.error_reply(401, 'mock: query-string API keys are refused; send the x-goog-api-key header',
                                    'UNAUTHENTICATED')
        token = request.headers.get('x-goog-api-key')
        if token is None:
            return self.error_reply(403, "Method doesn't allow unregistered callers (callers without established "
                                         'identity). Please use API Key or other form of API consumer identity to call '
                                         'this API.', 'PERMISSION_DENIED')
        if token.strip() != self.credential:
            return json_reply({'error': {'code': 400, 'message': 'API key not valid. Please pass a valid API key.',
                                         'status': 'INVALID_ARGUMENT',
                                         'details': [{'@type': 'type.googleapis.com/google.rpc.ErrorInfo',
                                                      'reason': 'API_KEY_INVALID', 'domain': 'googleapis.com',
                                                      'metadata': {'service': 'generativelanguage.googleapis.com'}}]}},
                              400)
        return None

    def stream_error_frame(self, message: str = 'The model is overloaded. Please try again later.') -> bytes:
        frame = ('data: ' + dumps({'error': {'code': 503, 'message': message, 'status': 'UNAVAILABLE'}}) + '\n\n').encode()
        return self._framing(frame)

    def _framing(self, frame: bytes) -> bytes:
        return frame.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n') if self.sse_crlf else frame

    def _body(self, request: MockRequest) -> dict:
        try:
            body = json.loads(request.body.decode('utf-8')) if request.body else {}
        except (ValueError, UnicodeDecodeError):
            refuse(self.error_reply(400, 'Invalid JSON payload received.'))
        if not isinstance(body, dict):
            refuse(self.error_reply(400, 'Invalid JSON payload received. Expected an object.'))
        return body

    @staticmethod
    def _has_tools(body: dict) -> bool:
        return any(tool.get('functionDeclarations') for tool in body.get('tools') or [] if isinstance(tool, dict))

    async def action(self, request: MockRequest) -> Reply:
        model, action = request.path_params['model'], request.path_params['action']
        body = self._body(request)
        variables = {**self.extra_vars, 'model': model}
        if action in ('generateContent', 'streamGenerateContent', 'countTokens'):
            if not body.get('contents') and not (action == 'countTokens' and body.get('generateContentRequest')):
                return self.error_reply(400, '* GenerateContentRequest.contents: contents is not specified')
        if action == 'generateContent':
            default = 'generate_content_function_call' if self._has_tools(body) else 'generate_content'
            return self.fixture_reply(self._case('gemini.generateContent', default), variables)
        if action == 'streamGenerateContent':
            default = 'stream_generate_content_function_call' if self._has_tools(body) else 'stream_generate_content'
            case = self._case('gemini.streamGenerateContent', default)
            if request.q('alt') == 'sse':
                reply = self.fixture_reply(case, variables)
                if reply.frames is not None:
                    reply.frames = [self._framing(frame) for frame in reply.frames]
                return reply
            fx = load('gemini', case)
            chunks = [dumps(substitute(row['data'], variables)) for row in fx.rows if isinstance(row.get('data'), dict)]
            return Reply(200, [('content-type', 'application/json; charset=UTF-8')],
                         ('[' + ',\r\n'.join(chunks) + ']').encode('utf-8'))
        if action == 'embedContent':
            if not body.get('content'):
                return self.error_reply(400, '* EmbedContentRequest.content: content is not specified')
            return self.fixture_reply(self._case('gemini.embedContent', 'embed_content'), variables)
        if action == 'batchEmbedContents':
            requests = body.get('requests')
            if not isinstance(requests, list) or not requests:
                return self.error_reply(400, '* BatchEmbedContentsRequest.requests: requests is not specified')
            fx = load('gemini', 'batch_embed_contents')
            template = fx.body(variables)['embeddings']
            if len(requests) == len(template):
                return self.fixture_reply('batch_embed_contents', variables)
            return json_reply({'embeddings': [template[i % len(template)] for i in range(len(requests))]})
        return self.fixture_reply('count_tokens', {'total_tokens': count_total_tokens(
            body.get('generateContentRequest') or body)})

    def _catalog(self) -> list:
        first = load('gemini', 'models_page1').body({})
        second = load('gemini', 'models_page2').body({})
        return first['models'] + second['models']

    async def models_list(self, request: MockRequest) -> Reply:
        case = self._case('gemini.models', '')
        if case:
            return self.fixture_reply(case, {})
        try:
            size = int(request.q('pageSize') or 50)
        except ValueError:
            return self.error_reply(400, 'Invalid pageSize')
        token = request.q('pageToken') or ''
        try:
            offset = int(token.rsplit('-', 1)[1]) if token else 0
        except (IndexError, ValueError):
            return self.error_reply(400, 'Invalid page token')
        models = self._catalog()
        page = models[offset:offset + max(1, size)]
        out = {'models': page}
        if offset + len(page) < len(models):
            out['nextPageToken'] = f'mock-page-{offset + len(page)}'
        if size == 2 and offset == 0:
            return self.fixture_reply('models_page1', {})
        if size == 2 and offset == 2 and len(models) == 3:
            return self.fixture_reply('models_page2', {})
        return json_reply(out)

    async def models_get(self, request: MockRequest) -> Reply:
        name = 'models/' + request.path_params['model']
        for item in self._catalog():
            if item['name'] == name:
                return json_reply(item)
        return self.error_reply(404, f'models/{request.path_params["model"]} is not found for API version v1beta, or is '
                                     'not supported for generateContent.')


    # -- native-only endpoints of the completeness round ------------------------------------------------------------
    def install_completeness_routes(self) -> None:
        """Download, Interactions, File Search, agents, webhooks and auth_tokens (state in self.native_objects)."""

        def created(kind: str, prefix: str, extra: dict | None = None) -> dict:
            number = sum(1 for key in self.native_objects if key.startswith(kind + ':')) + 1
            record = {'name': f'{prefix}/mock{kind}{number:04d}', **(extra or {})}
            self.native_objects[kind + ':' + record['name'].rsplit('/', 1)[-1]] = record
            return record

        async def download(request: MockRequest) -> Reply:
            file_id = request.path_params['file_id']
            if request.q('alt') != 'media':
                return self.error_reply(400, 'alt=media is required for a download')
            return Reply(200, [('content-type', 'application/jsonl'),
                               ('content-disposition', f'attachment; filename="{file_id}.jsonl"')],
                         frames=[b'{"key":"r1","response":{}}\n', b'{"key":"r2","response":{}}\n'],
                         stream_kind='binary')

        async def redirecting(request: MockRequest) -> Reply:
            return Reply(302, [('location', 'https://example.invalid/storage/' + request.path_params['file_id'])], b'')

        async def interaction_create(request: MockRequest) -> Reply:
            body = self._body(request)
            number = sum(1 for key in self.native_objects if key.startswith('interaction:')) + 1
            record = {'id': f'mock-interaction-{number:04d}', 'object': 'interaction', 'status': 'completed',
                      'model': body.get('model'), 'outputs': [{'type': 'text', 'text': 'Hello from interactions.'}],
                      'usage': {'total_input_tokens': 4, 'total_output_tokens': 5, 'total_tokens': 9}}
            self.native_objects['interaction:' + record['id']] = record
            if body.get('stream'):
                frames = [('event: interaction.created\ndata: ' + dumps({'event_type': 'interaction.created',
                                                                        'interaction': {'id': record['id']}})
                           + '\n\n').encode('utf-8'),
                          ('event: interaction.completed\ndata: ' + dumps({'event_type': 'interaction.completed',
                                                                          'interaction': record}) + '\n\n')
                          .encode('utf-8')]
                return Reply(200, [('content-type', 'text/event-stream')], frames=frames, stream_kind='sse')
            return json_reply(record)

        async def interaction_get(request: MockRequest) -> Reply:
            key = 'interaction:' + request.path_params['interaction_id']
            found = self.native_objects.get(key)
            if found is None:
                return self.error_reply(404, 'Interaction not found')
            if request.method == 'DELETE':
                self.native_objects.pop(key)
                return json_reply({})
            return json_reply(found)

        async def interaction_cancel(request: MockRequest) -> Reply:
            found = self.native_objects.get('interaction:' + request.path_params['interaction_id'])
            return json_reply(dict(found, status='cancelled')) if found else self.error_reply(404, 'Interaction not found')

        async def store_create(request: MockRequest) -> Reply:
            body = self._body(request)
            return json_reply(created('store', 'fileSearchStores', {'displayName': body.get('displayName')}))

        async def store_list(request: MockRequest) -> Reply:
            return json_reply({'fileSearchStores': [v for k, v in self.native_objects.items() if k.startswith('store:')]})

        async def store_get(request: MockRequest) -> Reply:
            store_id, _, action = request.path_params['store'].partition(':')
            found = self.native_objects.get('store:' + store_id)
            if found is None:
                return self.error_reply(404, 'File search store not found')
            if request.method == 'POST' and action == 'importFile':
                return json_reply({'name': f"{found['name']}/operations/mock-import-0001", 'done': False})
            if request.method == 'DELETE':
                self.native_objects.pop('store:' + store_id)
                return json_reply({})
            return json_reply(found)

        async def documents(request: MockRequest) -> Reply:
            return json_reply({'documents': [{'name': f"fileSearchStores/{request.path_params['store']}/documents/d1"}]})

        async def agent_create(request: MockRequest) -> Reply:
            return json_reply(created('agent', 'agents', {'displayName': self._body(request).get('displayName')}))

        async def webhook_list(request: MockRequest) -> Reply:
            return json_reply({'webhooks': []})

        async def auth_token(request: MockRequest) -> Reply:
            body = self._body(request)
            return json_reply({'name': 'auth_tokens/mock-ephemeral-0001', 'uses': body.get('uses', 1)})

        self.route('GET', r'/download/v1beta/files/(?P<file_id>[^/:]+):download', download, 'native:files')
        self.route('GET', r'/v1beta/files/(?P<file_id>[^/:]+):download', redirecting, 'native:files')
        self.route('POST', r'/v1beta/interactions', interaction_create, 'native:interactions')
        self.route('GET', r'/v1beta/interactions/(?P<interaction_id>[^/]+)', interaction_get, 'native:interactions')
        self.route('DELETE', r'/v1beta/interactions/(?P<interaction_id>[^/]+)', interaction_get, 'native:interactions')
        self.route('POST', r'/v1beta/interactions/(?P<interaction_id>[^/]+)/cancel', interaction_cancel,
                   'native:interactions')
        self.route('POST', r'/v1beta/fileSearchStores', store_create, 'native:file_search')
        self.route('GET', r'/v1beta/fileSearchStores', store_list, 'native:file_search')
        self.route('GET', r'/v1beta/fileSearchStores/(?P<store>[^/]+)/documents', documents, 'native:file_search')
        for method in ('GET', 'POST', 'DELETE'):
            self.route(method, r'/v1beta/fileSearchStores/(?P<store>[^/]+)', store_get, 'native:file_search')
        self.route('POST', r'/v1beta/agents', agent_create, 'native:agents')
        self.route('GET', r'/v1beta/webhooks', webhook_list, 'native:webhooks')
        self.route('POST', r'/v1alpha/auth_tokens', auth_token, 'native:auth_tokens')


__all__ = ['ACTIONS', 'DEFAULT_VARS', 'GeminiMock', 'count_total_tokens']
