"""Ollama (cloud, native API) loopback mock.

Routes (api_base = base_url): POST /api/chat and /api/generate (NDJSON streams by default, application/x-ndjson; a single
JSON object with "stream": false), POST /api/embed (and the legacy /api/embeddings), GET /api/tags. Auth: Authorization:
Bearer <SENTINELS['ollama_cloud']>; otherwise 401 {"error": "unauthorized"}. Errors are {"error": "<text>"}, and an in-stream
error is an {"error": ...} NDJSON line. Model ids with ':' pass verbatim.
"""
from __future__ import annotations

import json

from .fixtures import dumps, load
from .server import BaseMock, MockRequest, Reply, refuse


class OllamaCloudMock(BaseMock):
    PROVIDER = 'ollama_cloud'
    PREFIX = ''

    def __init__(self, *, credential: str | None = None, known_models: list[str] | None = None):
        super().__init__(credential=credential)
        self.known_models = set(known_models) if known_models else None
        self.overrides: dict[str, list[str]] = {}
        self.route('POST', r'/api/chat', self.chat, 'ollama.chat')
        self.route('POST', r'/api/generate', self.generate, 'ollama.generate')
        self.route('POST', r'/api/embed', self.embed, 'ollama.embed')
        self.route('POST', r'/api/embeddings', self.embeddings_legacy, 'ollama.embeddings')
        self.route('GET', r'/api/tags', self.tags, 'ollama.tags')

    def fixture_provider(self) -> str:
        return 'ollama'

    def reset(self) -> None:
        super().reset()
        self.overrides.clear()

    def respond_with(self, surface: str, case: str) -> None:
        self.overrides.setdefault(surface, []).append(case)

    def _case(self, surface: str, default: str) -> str:
        queue = self.overrides.get(surface)
        return queue.pop(0) if queue else default

    @staticmethod
    def _json(body, status: int = 200) -> Reply:
        return Reply(status, [('content-type', 'application/json; charset=utf-8')], dumps(body).encode('utf-8'))

    def error_reply(self, status: int, message: str, code: str | None = None) -> Reply:
        return self._json({'error': message}, status)

    def auth_failure(self, request: MockRequest) -> Reply | None:
        token = self.presented_token(request, 'authorization')
        if token is None or token != self.credential:
            return self.error_reply(401, 'unauthorized')
        return None

    def stream_error_frame(self, message: str = 'an error was encountered while running the model') -> bytes:
        return (dumps({'error': message}) + '\n').encode('utf-8')

    def _body(self, request: MockRequest) -> dict:
        try:
            body = json.loads(request.body.decode('utf-8')) if request.body else None
        except (ValueError, UnicodeDecodeError):
            refuse(self.error_reply(400, 'invalid character in request body'))
        if not isinstance(body, dict):
            refuse(self.error_reply(400, 'missing request body'))
        if not body.get('model'):
            refuse(self.error_reply(400, 'model is required'))
        if self.known_models is not None and body['model'] not in self.known_models:
            refuse(self.error_reply(404, f"model '{body['model']}' not found"))
        return body

    async def chat(self, request: MockRequest) -> Reply:
        body = self._body(request)
        stream = body.get('stream', True) is not False
        tools = bool(body.get('tools'))
        default = ('chat_stream_tools' if tools else 'chat_stream') if stream else ('chat_tools' if tools else 'chat')
        return self.fixture_reply(self._case('ollama.chat', default), {'model': body['model']})

    async def generate(self, request: MockRequest) -> Reply:
        body = self._body(request)
        stream = body.get('stream', True) is not False
        return self.fixture_reply(self._case('ollama.generate', 'generate_stream' if stream else 'generate'),
                                  {'model': body['model']})

    async def embed(self, request: MockRequest) -> Reply:
        body = self._body(request)
        value = body.get('input')
        if value in (None, '', []):
            return self.error_reply(400, 'input is required')
        count = len(value) if isinstance(value, list) else 1
        out = load('ollama', 'embed').body({'model': body['model']})
        if count == len(out['embeddings']):
            return self.fixture_reply('embed', {'model': body['model']})
        out['embeddings'] = [out['embeddings'][i % len(out['embeddings'])] for i in range(count)]
        return self._json(out)

    async def embeddings_legacy(self, request: MockRequest) -> Reply:
        body = self._body(request)
        vector = load('ollama', 'embed').body({'model': body['model']})['embeddings'][0]
        return self._json({'embedding': vector})

    async def tags(self, request: MockRequest) -> Reply:
        return self.fixture_reply('tags', {})


__all__ = ['OllamaCloudMock']
