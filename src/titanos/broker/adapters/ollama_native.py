"""Ollama Cloud native adapter: /api/chat, /api/generate and /api/embed (NDJSON streams, streaming by default) plus
/api/tags, with Bearer auth set by EgressClient. Model ids with ':' pass verbatim. It talks only to the configured cloud
upstream and never touches a local Ollama store."""
from __future__ import annotations

from ..sse import ollama_usage
from ..surfaces import Surface
from ..translate import ollama
from ..translate.common import bad_upstream, refuse
from ..translate.ir import IRRequest, IRResponse
from ..types import EgressGrant, Usage
from .base import RawRequest, UpstreamResult
from .common import AdapterBase, json_body, strip_titanos

TRANSLATED = frozenset({'openai.chat', 'openai.completions', 'openai.responses', 'anthropic.messages',
                        'gemini.generateContent', 'gemini.streamGenerateContent', 'openai.embeddings',
                        'gemini.embedContent', 'gemini.batchEmbedContents'})


class OllamaNativeAdapter(AdapterBase):
    adapter_name = 'ollama_native'
    native_protocol = 'ollama'

    def supports(self, profile, surface: Surface) -> bool:
        if surface.name == 'ollama.embeddings':
            return 'ollama.embed' in profile.surfaces
        return surface.name in profile.surfaces or surface.name in TRANSLATED

    async def passthrough(self, grant: EgressGrant, egress, surface: Surface, raw: RawRequest, *,
                          notes: list | None = None, inject: dict | None = None,
                          allow_store: bool = False) -> UpstreamResult:
        name = surface.name
        if name not in self.profile.surfaces:
            raise refuse(f'ollama_cloud does not offer {name}')
        body = strip_titanos(json_body(raw))
        body['model'] = self.model_for(grant)
        for key, value in (inject or {}).items():
            body.setdefault(key, value)
        stream = name in ('ollama.chat', 'ollama.generate') and body.get('stream', True) is not False
        response = await self.send(grant, egress, 'POST', self.profile.surfaces[name], json=body, stream=stream)
        if stream:
            return await self.stream_result(response, 'ollama')
        return await self.body_result(response, surface)

    async def invoke(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                     inject: dict | None = None):
        model = self.model_for(grant, ir.model)
        if ir.kind == 'embed':
            response = await self.send(grant, egress, 'POST', self.profile.surfaces['ollama.embed'],
                                       json=dict(ollama.embed_request(ir), model=model))
            decoded = ollama.embed_response(self.parse_json(response))
            return IRResponse(id='', model=model, usage=decoded['usage'], extras={'vectors': decoded['vectors']})
        if ir.kind != 'chat':
            raise refuse(f'{ir.kind} requests are not available from ollama_cloud')
        body = dict(ollama.encode_request(ir, self.profile, notes=notes), model=model)
        response = await self.send(grant, egress, 'POST', self.profile.surfaces['ollama.chat'], json=body,
                                   stream=ir.stream)
        if ir.stream:
            return ollama.decode_stream(await self.ndjson_lines(response))
        return ollama.decode_response(self.parse_json(response))

    def invoke_sync(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                    inject: dict | None = None) -> IRResponse:
        model = self.model_for(grant, ir.model)
        if ir.kind == 'embed':
            response = self.send_sync(grant, egress, 'POST', self.profile.surfaces['ollama.embed'],
                                      json=dict(ollama.embed_request(ir), model=model))
            decoded = ollama.embed_response(self.parse_json(response))
            return IRResponse(id='', model=model, usage=decoded['usage'], extras={'vectors': decoded['vectors']})
        if ir.kind != 'chat':
            raise refuse(f'{ir.kind} requests are not available from ollama_cloud')
        body = dict(ollama.encode_request(ir, self.profile, notes=notes), model=model, stream=False)
        response = self.send_sync(grant, egress, 'POST', self.profile.surfaces['ollama.chat'], json=body)
        return ollama.decode_response(self.parse_json(response))

    async def list_models(self, grant: EgressGrant, egress) -> list[str]:
        response = await self.send(grant, egress, 'GET', self.profile.catalog_path)
        page = self.parse_json(response)
        if not isinstance(page, dict) or not isinstance(page.get('models'), list):
            raise bad_upstream('The upstream tags answer has no models list')
        return [row.get('name') or row.get('model') for row in page['models']
                if isinstance(row, dict) and (row.get('name') or row.get('model'))]

    def usage_from_body(self, surface: Surface, body, headers) -> Usage | None:
        if not isinstance(body, dict):
            return None
        if surface.name in ('ollama.embed', 'ollama.embeddings'):
            return Usage(input_tokens=body.get('prompt_eval_count'))
        return ollama_usage(dict(body, done=True))


__all__ = ['OllamaNativeAdapter', 'TRANSLATED']
