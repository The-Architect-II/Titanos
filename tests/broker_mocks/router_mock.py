"""Router-as-upstream loopback mocks: RouterUpstreamMock(kind='hivemind' | 'pair' | 'openrouter').

- hivemind: a HiveMind-gateway-like OpenAI subset under /v1 (chat, completions, embeddings, audio speech and transcriptions,
  image generations, models) plus GET /v1/health/ready; the bearer key is optional (a presented one must be the
  sentinel). It records every inbound X-Titanos-Hop value in mock.hops (loop-detection assertions).
- pair: a PAIR-facade-like OpenAI subset under /v1 (chat, completions, embeddings, models); no auth on loopback (a presented
  key must be the sentinel).
- openrouter: an OpenRouter-like router under /api/v1 (chat, completions, models); a bearer key is required, OpenRouter's
  {"error": {"message", "code"}} 401 envelope; answers name the routed provider and streams open with the
  ': OPENROUTER PROCESSING' SSE comment (tests/fixtures/providers/openrouter/*).
"""
from __future__ import annotations

from .fixtures import FIXTURES_DIR
from .openai_mock import OpenAIMock
from .server import MockRequest, Reply, json_reply

KINDS = {
    'hivemind': ('hivemind_gateway', '/v1', False,
                 ('openai.chat', 'openai.completions', 'openai.embeddings', 'openai.audio.speech',
                  'openai.audio.transcriptions', 'openai.images.generations', 'openai.models')),
    'pair': ('pair_facade', '/v1', False, ('openai.chat', 'openai.completions', 'openai.embeddings', 'openai.models')),
    'openrouter': ('openrouter', '/api/v1', True, ('openai.chat', 'openai.completions', 'openai.models')),
}


class RouterUpstreamMock(OpenAIMock):
    def __init__(self, kind: str = 'hivemind', *, credential: str | None = None, require_auth: bool | None = None,
                 models: list[str] | None = None):
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {sorted(KINDS)}')
        provider, prefix, auth_required, surfaces = KINDS[kind]
        self.kind = kind
        self.PROVIDER = provider
        super().__init__(credential=credential, models=models, surfaces=surfaces, prefix=prefix,
                         require_auth=auth_required if require_auth is None else require_auth)
        if kind == 'hivemind':
            self.route('GET', r'/v1/health/ready', self.health_ready, 'router.health', auth=False)

    @property
    def hops(self) -> list[str]:
        return [hit.headers['x-titanos-hop'] for hit in self.hits if 'x-titanos-hop' in hit.headers]

    def fixture_provider_for(self, case: str) -> str:
        if self.kind == 'openrouter' and (FIXTURES_DIR / 'openrouter' / (case + '.jsonl')).is_file():
            return 'openrouter'
        return 'openai'

    def auth_failure(self, request: MockRequest) -> Reply | None:
        if self.kind != 'openrouter':
            return super().auth_failure(request)
        token = self.presented_token(request, 'authorization')
        if token is None:
            return json_reply({'error': {'message': 'No auth credentials found', 'code': 401}}, 401)
        if token != self.credential:
            return json_reply({'error': {'message': 'User not found.', 'code': 401}}, 401)
        return None

    def error_reply(self, status: int, message: str, code: str | None = None, *, param: str | None = None,
                    type_: str | None = None) -> Reply:
        if self.kind == 'openrouter':
            return json_reply({'error': {'message': message, 'code': status, 'metadata': {}}}, status)
        return super().error_reply(status, message, code, param=param, type_=type_)

    async def health_ready(self, request: MockRequest) -> Reply:
        return json_reply({'status': 'ready'})


__all__ = ['KINDS', 'RouterUpstreamMock']
