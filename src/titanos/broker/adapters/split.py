"""SplitProtocolAdapter: an OpenAI-compatible row that speaks another wire family natively on some surfaces.

A router such as NVIDIA NeMo Switchyard serves OpenAI Chat Completions, OpenAI Responses and Anthropic Messages on one
origin. The row's extras.surface_adapters ({surface name: adapter family}) names the surfaces that must not be re-encoded
as OpenAI chat: each passes through an inner adapter of that family built over a derived profile row (same id, kind,
base_url, auth, catalog and headers; only the adapter family and the surface map narrowed), so anthropic.messages keeps
thinking blocks, cache_control, citations and anthropic-beta exactly as the client sent them. Every other surface, every
translated inbound protocol (Gemini, Ollama, completions: onto openai.chat), the catalog and the probe stay on the
OpenAI-compatible path. An operator who narrows upstream.surfaces to exclude such a surface gets the OpenAI translation
again (protocol_for answers 'openai').

list_models_meta() reads the per-model capability fields a router publishes on its catalog (tool_calling, vision,
context window; extras.catalog_capabilities) so the catalog can refuse a tool or image request before egress and Studio
can show them.
"""
from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from ..surfaces import Surface
from ..types import EgressGrant
from .anthropic import AnthropicAdapter
from .base import RawRequest, UpstreamResult
from .openai_compat import OpenAICompatAdapter, parse_catalog

INNER = {'anthropic': AnthropicAdapter}


def catalog_meta(document) -> dict[str, dict]:
    """{model id: {tool_calling, vision, context_length, streaming}} from an OpenAI-shaped catalog whose rows carry a
    capabilities object (Switchyard: capabilities.tool_calling / vision / context_window / streaming, or a top-level
    context_length). Unknown values stay None; only booleans and positive integers are kept."""
    rows = document.get('data') if isinstance(document, dict) else document
    out: dict[str, dict] = {}
    for row in rows if isinstance(rows, list) else ():
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
            continue
        caps = row.get('capabilities') if isinstance(row.get('capabilities'), dict) else {}

        def flag(name):
            value = caps.get(name, row.get(name))
            return value if isinstance(value, bool) else None
        context = row.get('context_length', caps.get('context_window', caps.get('context_length')))
        if not (isinstance(context, int) and not isinstance(context, bool) and context > 0):
            context = None
        out[row['id']] = {'tool_calling': flag('tool_calling'), 'vision': flag('vision'),
                          'streaming': flag('streaming'), 'context_length': context}
    return out


class SplitProtocolAdapter(OpenAICompatAdapter):
    """See the module docstring."""

    def __init__(self, profile, upstream=None, **kw):
        super().__init__(profile, upstream, **kw)
        self.inner: dict[str, object] = {}
        families: dict[str, list[str]] = {}
        for name, family in (profile.extras.get('surface_adapters') or {}).items():
            families.setdefault(family, []).append(name)
        for family, names in families.items():
            cls = INNER.get(family)
            if cls is None:
                raise ValueError(f'{profile.id}: no inner adapter for {family}')
            row = replace(profile, adapter=family,
                          surfaces=MappingProxyType({name: profile.surfaces[name] for name in names}))
            adapter = cls(row, upstream, **kw)
            for name in names:
                self.inner[name] = adapter

    def _inner(self, name: str):
        adapter = self.inner.get(name)
        return adapter if adapter is not None and self.has(name) else None

    def protocol_for(self, name: str) -> str:
        """The wire family that serves surface `name` natively on this upstream."""
        adapter = self._inner(name)
        return adapter.native_protocol if adapter is not None else self.native_protocol

    async def passthrough(self, grant: EgressGrant, egress, surface: Surface, raw: RawRequest, *,
                          notes: list | None = None, inject: dict | None = None,
                          allow_store: bool = False) -> UpstreamResult:
        adapter = self._inner(surface.name)
        if adapter is not None:
            return await adapter.passthrough(grant, egress, surface, raw, notes=notes, inject=inject,
                                             allow_store=allow_store)
        return await super().passthrough(grant, egress, surface, raw, notes=notes, inject=inject,
                                         allow_store=allow_store)

    async def count_tokens(self, grant: EgressGrant, egress, ir) -> int:
        """input_tokens through the Anthropic count_tokens surface (translated callers)."""
        adapter = self._inner('anthropic.count_tokens')
        if adapter is None:
            from ..translate.common import refuse
            raise refuse(f'{self.profile.id} does not offer anthropic.count_tokens')
        return await adapter.count_tokens(grant, egress, ir)

    def usage_from_body(self, surface: Surface, body, headers):
        adapter = self.inner.get(surface.name)
        if adapter is not None:
            return adapter.usage_from_body(surface, body, headers)
        return super().usage_from_body(surface, body, headers)

    async def list_models_meta(self, grant: EgressGrant, egress) -> tuple[list[str], dict[str, dict]]:
        """(model ids, per-model capability metadata) from one catalog call."""
        if not self.profile.catalog_path:
            return self.static_models(), {}
        response = await self.send(grant, egress, 'GET', self.profile.catalog_path)
        document = self.parse_json(response)
        meta = catalog_meta(document) if self.profile.extras.get('catalog_capabilities') else {}
        return parse_catalog(document), meta


__all__ = ['INNER', 'SplitProtocolAdapter', 'catalog_meta']
