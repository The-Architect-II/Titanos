"""Anthropic Messages adapter.

- Native passthrough of /v1/messages and /v1/messages/count_tokens: the model is rewritten by a JSON round trip
  (semantically faithful; no byte substitution) and Titanos-only fields are stripped. EgressClient sets x-api-key,
  anthropic-version (upstream.anthropic_version, else the profile default) and anthropic-beta: the inbound header is
  forwarded verbatim on passthrough and merged with upstream.anthropic_betas; translated calls send no inbound headers.
- Streams hold the first event: a 200 stream whose first event is error (overloaded_error, api_error) is a pre-commit
  upstream failure.
- Translated calls encode through translate.anthropic_messages for the upstream model (translate/anthropic_models.py
  decides thinking, effort, sampling, forced tool choice and structured outputs per model) with max_tokens always set
  (the client's cap or default_max_tokens, noted 'injected:max_tokens'); json_schema maps to output_config.format (GA,
  no beta header).
- Catalog from /v1/models (paged with after_id); list_models_meta keeps each entry's display_name, created_at,
  max_input_tokens (context_length), max_tokens and capabilities (vision from image_input) for the catalog metadata and
  the Anthropic-shaped /v1/models; count_tokens for translated callers.
- Provider-side state named in a passthrough body (a container id, custom skills, file_id sources) is checked before
  egress by broker/anthropic_state.py (cloud:store, data_policy.allow_store, owner binding).
"""
from __future__ import annotations

from ..sse import anthropic_usage
from ..surfaces import Surface
from ..translate import anthropic_messages
from ..translate.common import refuse
from ..translate.ir import IRRequest, IRResponse
from ..types import EgressGrant, Usage
from .base import RawRequest, UpstreamResult
from .common import CATALOG_MAX_PAGES, AdapterBase, json_body, strip_titanos

TRANSLATED = frozenset({'openai.chat', 'openai.completions', 'openai.responses', 'gemini.generateContent',
                        'gemini.streamGenerateContent', 'ollama.chat', 'ollama.generate'})
# the provider's model-object fields kept from the live catalog (no secrets; served by the Anthropic-shaped /v1/models)
MODEL_INFO_FIELDS = ('display_name', 'created_at', 'max_input_tokens', 'max_tokens', 'capabilities')


class AnthropicAdapter(AdapterBase):
    adapter_name = 'anthropic'
    native_protocol = 'anthropic'

    def supports(self, profile, surface: Surface) -> bool:
        return surface.name in profile.surfaces or surface.name in TRANSLATED

    def structured_outputs(self) -> bool:
        beta = self.profile.extras.get('structured_outputs_beta')
        return bool(beta) and beta in (getattr(self.upstream, 'anthropic_betas', None) or ())

    async def passthrough(self, grant: EgressGrant, egress, surface: Surface, raw: RawRequest, *,
                          notes: list | None = None, inject: dict | None = None,
                          allow_store: bool = False) -> UpstreamResult:
        if surface.name not in self.profile.surfaces:
            raise refuse(f'anthropic does not offer {surface.name}')
        model = self.model_for(grant)
        body = strip_titanos(json_body(raw))
        body['model'] = model
        for key, value in (inject or {}).items():
            body.setdefault(key, value)
        stream = surface.name == 'anthropic.messages' and bool(body.get('stream'))
        response = await self.send(grant, egress, 'POST', self.profile.surfaces[surface.name], json=body,
                                   headers=raw.headers, stream=stream)
        if stream:
            return await self.stream_result(response, 'anthropic')
        return await self.body_result(response, surface)

    def request_body(self, ir: IRRequest, model: str, notes) -> dict:
        body = anthropic_messages.encode_request(ir, self.profile, default_max_tokens=self.default_max_tokens,
                                                 structured_outputs=self.structured_outputs(), notes=notes, model=model)
        body['model'] = model
        return body

    async def invoke(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                     inject: dict | None = None):
        if ir.kind != 'chat':
            raise refuse(f'{ir.kind} requests are not available from anthropic')
        model = self.model_for(grant, ir.model)
        body = self.request_body(ir, model, notes)
        response = await self.send(grant, egress, 'POST', self.profile.surfaces['anthropic.messages'], json=body,
                                   stream=ir.stream)
        if ir.stream:
            return anthropic_messages.decode_stream(await self.sse_frames(response, 'anthropic'))
        return anthropic_messages.decode_response(self.parse_json(response))

    def invoke_sync(self, grant: EgressGrant, egress, ir: IRRequest, *, notes: list | None = None,
                    inject: dict | None = None) -> IRResponse:
        if ir.kind != 'chat':
            raise refuse(f'{ir.kind} requests are not available from anthropic')
        model = self.model_for(grant, ir.model)
        body = self.request_body(ir, model, notes)
        body.pop('stream', None)
        response = self.send_sync(grant, egress, 'POST', self.profile.surfaces['anthropic.messages'], json=body)
        return anthropic_messages.decode_response(self.parse_json(response))

    async def count_tokens(self, grant: EgressGrant, egress, ir: IRRequest) -> int:
        """input_tokens for a translated request (POST /v1/messages/count_tokens)."""
        model = self.model_for(grant, ir.model)
        body = self.request_body(ir, model, None)
        for name in ('max_tokens', 'stream', 'temperature', 'top_p', 'top_k', 'stop_sequences'):
            body.pop(name, None)
        response = await self.send(grant, egress, 'POST', self.profile.surfaces['anthropic.count_tokens'], json=body)
        value = self.parse_json(response).get('input_tokens')
        if not isinstance(value, int):
            from ..translate.common import bad_upstream
            raise bad_upstream('The upstream count_tokens answer has no input_tokens')
        return value

    async def list_models(self, grant: EgressGrant, egress) -> list[str]:
        return (await self.list_models_meta(grant, egress))[0]

    async def list_models_meta(self, grant: EgressGrant, egress) -> tuple[list[str], dict]:
        """(model ids, {id: meta}) from the paged /v1/models: meta carries context_length (max_input_tokens),
        max_output_tokens, vision (capabilities.image_input) and the provider's own model-object fields
        (MODEL_INFO_FIELDS under 'anthropic') for the Anthropic-shaped catalog."""
        out, meta, after = [], {}, None
        for _ in range(CATALOG_MAX_PAGES):
            params = {'limit': '1000'}
            if after:
                params['after_id'] = after
            response = await self.send(grant, egress, 'GET', self.profile.catalog_path, params=params)
            page = self.parse_json(response)
            rows = page.get('data') if isinstance(page, dict) else None
            if not isinstance(rows, list):
                from ..translate.common import bad_upstream
                raise bad_upstream('The upstream catalog has no data list')
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get('id'), str):
                    out.append(row['id'])
                    meta[row['id']] = model_meta(row)
            if not page.get('has_more') or not page.get('last_id'):
                break
            after = page['last_id']
        return out, meta

    def usage_from_body(self, surface: Surface, body, headers) -> Usage | None:
        if not isinstance(body, dict):
            return None
        if surface.name == 'anthropic.count_tokens':
            return Usage(input_tokens=0, output_tokens=0)  # token counting is not billed
        return anthropic_usage(body.get('usage'))


def model_meta(row: dict) -> dict:
    """The catalog metadata of one Anthropic model object (see AnthropicAdapter.list_models_meta)."""
    out: dict = {'anthropic': {k: row[k] for k in MODEL_INFO_FIELDS if k in row}}
    if isinstance(row.get('max_input_tokens'), int):
        out['context_length'] = row['max_input_tokens']
    if isinstance(row.get('max_tokens'), int):
        out['max_output_tokens'] = row['max_tokens']
    image = (row.get('capabilities') or {}).get('image_input') if isinstance(row.get('capabilities'), dict) else None
    if isinstance(image, dict) and isinstance(image.get('supported'), bool):
        out['vision'] = image['supported']
    return out


__all__ = ['AnthropicAdapter', 'MODEL_INFO_FIELDS', 'TRANSLATED', 'model_meta']
