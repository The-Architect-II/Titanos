"""Azure OpenAI loopback mock: /openai/deployments/{deployment}/{surface path}?api-version=<version> with the api-key header.

api_base = base_url + '/openai' stands in for https://{resource}.{endpoint_host}/openai. The request body may omit model;
the deployment map gives the underlying model. Native answers:
- api-key missing or wrong: 401 {"error": {"code": "401", "message": "Access denied due to ..."}};
- api-version missing (or not in api_versions when given): 404 {"error": {"code": "404", "message": "Resource not found"}};
- an unknown deployment: 404 {"error": {"code": "DeploymentNotFound", ...}}.
Chat answers carry Azure's content-filter fields, and a chat stream starts with the prompt_filter_results chunk whose choices
is [] and whose model is '' (tests/fixtures/providers/azure_openai/*); the other surfaces replay the OpenAI fixtures.
Image deployments keep the pre-2026-10 OpenAI behaviour (dall-e accepts response_format and answers url by default;
gpt-image-* refuses the field and answers base64; the azure_openai profile row therefore lists image_b64_only_globs
('gpt-image-*',)): the 2026-10-02 live observation that OpenAI refuses response_format for dall-e was made
against api.openai.com, not Azure.
"""
from __future__ import annotations

import re

from .fixtures import FIXTURES_DIR
from .openai_mock import OpenAIMock
from .server import MockRequest, Reply, json_reply, refuse

AZURE_SURFACES = {
    'chat/completions': 'openai.chat',
    'completions': 'openai.completions',
    'embeddings': 'openai.embeddings',
    'audio/speech': 'openai.audio.speech',
    'audio/transcriptions': 'openai.audio.transcriptions',
    'audio/translations': 'openai.audio.translations',
    'images/generations': 'openai.images.generations',
}
DEFAULT_DEPLOYMENTS = {
    'chat-dep': 'gpt-4o-mini',
    'instruct-dep': 'gpt-35-turbo-instruct',
    'embed-dep': 'text-embedding-3-small',
    'tts-dep': 'tts-1',
    'whisper-dep': 'whisper-1',
    'image-dep': 'dall-e-3',
}


class AzureMock(OpenAIMock):
    PROVIDER = 'azure_openai'
    PREFIX = '/openai'
    IMAGES_ACCEPT_RESPONSE_FORMAT = True

    def __init__(self, *, deployments: dict | None = None, api_versions=None, credential: str | None = None):
        self.deployments = dict(DEFAULT_DEPLOYMENTS if deployments is None else deployments)
        self.api_versions = set(api_versions) if api_versions else None
        super().__init__(credential=credential)

    def _register(self) -> None:
        handlers = {'openai.chat': self.chat, 'openai.completions': self.completions, 'openai.embeddings': self.embeddings,
                    'openai.audio.speech': self.speech, 'openai.audio.transcriptions': self.transcriptions,
                    'openai.audio.translations': self.translations,
                    'openai.images.generations': self.images_generations}
        for suffix, surface in AZURE_SURFACES.items():
            self.route('POST', r'/openai/deployments/(?P<deployment>[^/]+)/' + re.escape(suffix), handlers[surface],
                       surface)

    def fixture_provider_for(self, case: str) -> str:
        return 'azure_openai' if (FIXTURES_DIR / 'azure_openai' / (case + '.jsonl')).is_file() else 'openai'

    def auth_failure(self, request: MockRequest) -> Reply | None:
        token = request.headers.get('api-key')
        if token is None:
            return json_reply({'error': {'code': '401', 'message': 'Access denied due to missing subscription key. Make '
                                         'sure to include subscription key when making requests to an API.'}}, 401)
        if token.strip() != self.credential:
            return json_reply({'error': {'code': '401', 'message': 'Access denied due to invalid subscription key or wrong '
                                         'API endpoint. Make sure to provide a valid key for an active subscription and use '
                                         'a correct regional API endpoint for your resource.'}}, 401)
        version = request.q('api-version')
        if not version or (self.api_versions is not None and version not in self.api_versions):
            return json_reply({'error': {'code': '404', 'message': 'Resource not found'}}, 404)
        # data-plane routes outside /deployments (files, batches: native_routes.install_openai_native under PREFIX)
        if 'deployment' in request.path_params and request.path_params['deployment'] not in self.deployments:
            return json_reply({'error': {'code': 'DeploymentNotFound', 'message': 'The API deployment for this resource '
                                         'does not exist. If you created the deployment within the last 5 minutes, please '
                                         'wait a moment and try again.'}}, 404)
        return None

    def error_reply(self, status: int, message: str, code: str | None = None, *, param: str | None = None,
                    type_: str | None = None) -> Reply:
        return json_reply({'error': {'code': code or str(status), 'message': message, 'param': param,
                                     'type': type_ or ('invalid_request_error' if status < 500 else 'server_error')}},
                          status)

    def request_model(self, request: MockRequest, body) -> str:
        return self.deployments[request.path_params['deployment']]

    async def _audio(self, request: MockRequest, task: str) -> Reply:
        form = await self.form_fields(request)
        if form.get('file') is None:
            return self.error_reply(400, "Missing required parameter: 'file'.", param='file')
        fmt = form.get('response_format') or 'json'
        if fmt not in ('json', 'text', 'verbose_json'):
            return self.error_reply(400, f"mock: response_format '{fmt}' is not implemented", param='response_format')
        return self.reply(self._case('openai.audio.' + task, f'{task}_{fmt}'), {'model': self.request_model(request, None)})

    async def images_generations(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        self._require(body, 'prompt')
        model = self.request_model(request, body)
        # no response_format: a dall-e deployment answers url (its default); a gpt-image deployment answers base64
        return self._images('openai.images.generations', self._image_case(model, body.get('response_format')),
                            model, body.get('n'))


__all__ = ['AZURE_SURFACES', 'AzureMock', 'DEFAULT_DEPLOYMENTS']
