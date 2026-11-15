"""Router-advisor loopback mocks: AdvisorMock(profile='generic-v1' | 'gim').

generic-v1 speaks titanos.router-advisor/v1 at POST /v1/advise (mock.url is the advisor url to configure): the request
contract string must match (400 otherwise); the answer's choice is mock.choice when it is a candidate ref, else the first
candidate, and fallbacks are the next two candidate refs. mock.answer, when set, is returned verbatim instead (invalid,
foreign or malformed answers for advisor validation tests). Bearer is optional; a presented key must be
SENTINELS['advisor'].

gim speaks the GIM router's func_1 envelope only as far as the advisor contract tests need it (mock.url is the base url;
the broker posts {url}/gim/llm_router/func_1): {messages, available_models, routing_preference, system_context} in,
{"ok": true, "result": {recommended_model, reasoning_mode, confidence, route_reason, scores, fallback_chain,
classification, latency_ms}} out. The answer is the recorded field-by-field fixture (advisor/gim_func_1, transcribed from
gim_api.py:670-686 inside the :694-727 envelope); no routing decision is computed here. When the request names
available_models, the fixture's preference order (recommended_model, then fallback_chain) is narrowed to them (the first
present is recommended, the rest stay in order) and scores keep only those models. Overrides for tests: mock.recommend
(recommended_model, when available), mock.fallback_chain, mock.reasoning_mode, mock.route_reason and mock.result (a
result dict returned verbatim). Invalid JSON gives 400 {"ok": false, "error": "Invalid JSON"} (gim_api.py:700).
GET /gim/llm_router/healthcheck/fast answers a synthetic healthy status. No auth.
Hits record the request body, so tests can assert that no prompt text reached the advisor.
"""
from __future__ import annotations

import json

from .fixtures import load
from .server import BaseMock, MockRequest, Reply, json_reply

CONTRACT = 'titanos.router-advisor/v1'


class AdvisorMock(BaseMock):
    PROVIDER = 'advisor'
    PREFIX = ''
    PROFILES = ('generic-v1', 'gim')

    def __init__(self, profile: str = 'generic-v1', *, credential: str | None = None):
        if profile not in self.PROFILES:
            raise ValueError(f'profile must be one of {self.PROFILES}')
        super().__init__(credential=credential)
        self.profile = profile
        self.choice: str | None = None
        self.answer: dict | None = None
        self.recommend: str | None = None
        self.fallback_chain: list[str] | None = None
        self.reasoning_mode: str | None = None
        self.route_reason: str | None = None
        self.result: dict | None = None
        if profile == 'generic-v1':
            self.route('POST', r'/v1/advise', self.advise, 'advisor.generic-v1')
        else:
            self.route('POST', r'/gim/llm_router/func_1', self.gim, 'advisor.gim', auth=False)
            self.route('GET', r'/gim/llm_router/healthcheck/fast', self.gim_health, 'advisor.gim.health', auth=False)

    @property
    def url(self) -> str:
        return self.base_url + ('/v1/advise' if self.profile == 'generic-v1' else '')

    def reset(self) -> None:
        super().reset()
        self.choice = self.answer = self.recommend = None
        self.fallback_chain = self.reasoning_mode = self.route_reason = self.result = None

    def error_reply(self, status: int, message: str, code: str | None = None) -> Reply:
        if self.profile == 'gim':
            return json_reply({'ok': False, 'error': message}, status)
        return json_reply({'error': {'message': message, 'code': code}}, status)

    def auth_failure(self, request: MockRequest) -> Reply | None:
        token = self.presented_token(request, 'authorization')
        if token is not None and token != self.credential:
            return self.error_reply(401, 'invalid advisor credential', 'unauthorized')
        return None

    async def advise(self, request: MockRequest) -> Reply:
        if self.answer is not None:
            return json_reply(self.answer)
        try:
            body = json.loads(request.body.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return self.error_reply(400, 'invalid JSON', 'invalid_request')
        if not isinstance(body, dict) or body.get('contract') != CONTRACT:
            return self.error_reply(400, f'contract must be {CONTRACT}', 'contract_mismatch')
        refs = [c.get('ref') for c in body.get('candidates') or [] if isinstance(c, dict) and c.get('ref')]
        if not refs:
            return self.error_reply(400, 'no candidates', 'invalid_request')
        choice = self.choice if self.choice in refs else refs[0]
        fallbacks = [ref for ref in refs if ref != choice][:2]
        return self.fixture_reply('generic_v1', {'choice': choice, 'fallbacks': fallbacks})

    async def gim(self, request: MockRequest) -> Reply:
        try:
            data = json.loads(request.body.decode('utf-8'))
            if not isinstance(data, dict):
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            return json_reply({'ok': False, 'error': 'Invalid JSON'}, 400)
        if self.result is not None:
            return json_reply({'ok': True, 'result': self.result})
        fixture = load('advisor', 'gim_func_1')
        recorded = fixture.body({})['result']
        result = json.loads(json.dumps(recorded))
        available = data.get('available_models')
        if isinstance(available, list) and available:
            order = [m for m in [recorded['recommended_model'], *recorded['fallback_chain'], *available]
                     if m in available]
            order = list(dict.fromkeys(order))
            result['recommended_model'] = order[0]
            result['fallback_chain'] = order[1:4]
            result['scores'] = {m: recorded['scores'].get(m, 0.0) for m in available}
        if self.recommend is not None and (not isinstance(available, list) or self.recommend in available):
            rest = [m for m in [result['recommended_model'], *result['fallback_chain']] if m != self.recommend]
            result['recommended_model'] = self.recommend
            result['fallback_chain'] = rest[:3]
        if self.fallback_chain is not None:
            result['fallback_chain'] = list(self.fallback_chain)
        if self.reasoning_mode is not None:
            result['reasoning_mode'] = self.reasoning_mode
        if self.route_reason is not None:
            result['route_reason'] = self.route_reason
        result['confidence'] = result['scores'].get(result['recommended_model'], 0.0)
        if result == recorded:
            return self.fixture_reply('gim_func_1', {})
        return json_reply({'ok': True, 'result': result})

    async def gim_health(self, request: MockRequest) -> Reply:
        return json_reply({'status': 'healthy', 'models': {'reasoning_router': False, 'arch_router': False,
                                                           'complexity_heuristic': True},
                           'device': 'cpu', 'models_loaded': 1, 'models_total': 3, 'route_classifier_disabled': True})


__all__ = ['AdvisorMock', 'CONTRACT']
