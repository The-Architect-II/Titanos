"""Broker MCP tools: status, catalog, explain, decisions, usage, upstreams, health, test, messages, count_tokens,
moderate and whoami (register_mcp, called from mcp_server.create_mcp next to runtime_api and assignment_api).

Every tool goes through the same in-process request() as the other tools, so a mounted call runs as the outer
request's principal and each inner route enforces its own scope (broker:read, broker:admin, inference:chat, ...).

Deliberately absent: no MCP tool sets, rotates or deletes credentials, mints, rotates or revokes keys, or changes
policy, upstreams, aliases, prices or advisors. Agents never handle secrets or widen egress; the CLI, the HTTP API and
Studio cover those for humans. No tool accepts or returns a secret value: the inputs below are model names, ids,
messages and query filters only, and the answers are the HTTP routes' metadata (which never carry secrets).

MCP tool schemas stay strict and local-sized: model names are at most 128 characters, so cloud calls over MCP take
published aliases or short qualified refs, and provider-native extra fields cannot be passed (documented)."""
from typing import Annotated, Any, Literal
from urllib.parse import quote

from mcp.server.mcpserver import Context  # this module is imported only by mcp_server (the mcp extra is installed)
from mcp.types import ToolAnnotations
from pydantic import Field

from ..protocols import Anthropic

UpstreamId = Annotated[str, Field(pattern=r'^[a-z][a-z0-9_-]{0,31}$')]
ModelName = Annotated[str, Field(min_length=1, max_length=128)]
DayStamp = Annotated[str, Field(pattern=r'^[0-9]{4}-[0-9]{2}-[0-9]{2}([T ][0-9:.]+Z?)?$')]
UsageGroups = Annotated[str, Field(pattern=r'^(key|upstream|model|surface|day)(,(key|upstream|model|surface|day)){0,4}$')]
TOOL_NAMES = ('broker_status', 'broker_catalog', 'broker_explain', 'broker_decisions', 'broker_usage', 'list_upstreams',
              'upstream_health', 'test_upstream', 'messages', 'count_tokens', 'moderate', 'whoami')


class AnthropicMessagesCall(Anthropic):
    """The local strict Anthropic Messages body (protocols.Anthropic), non-streaming, with a 128-character model."""
    model: ModelName
    stream: Literal[False] = False


def register_mcp(mcp, json_request, *, request=None, open_world: bool = True) -> None:
    """Register the broker tools. json_request(method, route, *, ctx=None, **kwargs) and request() are mcp_server's
    ctx-aware helpers (request is required: the inference tools read provenance headers); open_world is the static open_world_hint of the inference tools (mcp_server.open_world_hint)."""
    from ..mcp_server import with_provenance

    read = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    infer = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=open_world)
    probe = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True)

    @mcp.tool(annotations=read)
    async def broker_status(ctx: Context = None) -> dict[str, Any]:
        """Broker summary: role, effective cloud policy, egress mode, vault backend and sealed state, upstream counts, virtual models and audit state. No URLs or secrets."""
        return await json_request('GET', '/api/broker', ctx=ctx)

    @mcp.tool(annotations=read)
    async def broker_catalog(refresh: bool = False, ctx: Context = None) -> dict[str, Any]:
        """Cloud, router and virtual models this caller may use, with each upstream's catalog source. refresh re-reads stale live catalogs of upstreams the caller may see."""
        return await json_request('GET', '/api/broker/catalog', ctx=ctx, params={'refresh': 'true' if refresh else 'false'})

    @mcp.tool(annotations=read)
    async def broker_explain(surface: Annotated[str, Field(min_length=1, max_length=64)] = 'openai.chat',
                             body: dict[str, Any] | None = None,
                             headers: dict[str, str] | None = None, ctx: Context = None) -> dict[str, Any]:
        """Dry-run routing for a request body on a surface (for example openai.chat): where the model would run, which candidates were excluded and why. Zero egress; nothing is sent upstream."""
        return await json_request('POST', '/api/broker/explain', ctx=ctx,
                                  json={'surface': surface, 'body': body or {}, 'headers': headers or {}})

    @mcp.tool(annotations=read)
    async def broker_decisions(limit: Annotated[int, Field(ge=1, le=1000)] = 100, model: ModelName | None = None,
                               ctx: Context = None) -> dict[str, Any]:
        """Recent routing decisions (metadata only, never prompts or outputs): own key unless the caller holds broker:admin."""
        params = {'limit': limit, **({'model': model} if model is not None else {})}
        return await json_request('GET', '/api/broker/decisions', ctx=ctx, params=params)

    @mcp.tool(annotations=read)
    async def broker_usage(group_by: UsageGroups = 'key,upstream,model,day', since: DayStamp | None = None,
                           until: DayStamp | None = None, ctx: Context = None) -> dict[str, Any]:
        """Usage and cost rollups grouped by key, upstream, model, surface or day; own key unless the caller holds broker:admin."""
        params = {'group_by': group_by, **({'since': since} if since else {}), **({'until': until} if until else {})}
        return await json_request('GET', '/api/broker/usage', ctx=ctx, params=params)

    @mcp.tool(annotations=read)
    async def list_upstreams(ctx: Context = None) -> dict[str, Any]:
        """Configured broker upstreams with readiness, credential validation state (never the credential) and catalog status."""
        return await json_request('GET', '/api/broker/upstreams', ctx=ctx)

    @mcp.tool(annotations=read)
    async def upstream_health(id: UpstreamId, ctx: Context = None) -> dict[str, Any]:
        """One upstream's breaker state, recent outcomes and last seen rate-limit headroom."""
        return await json_request('GET', '/api/broker/upstreams/' + quote(id, safe='') + '/health', ctx=ctx)

    @mcp.tool(annotations=probe)
    async def test_upstream(id: UpstreamId, ctx: Context = None) -> dict[str, Any]:
        """Make one probe call to an upstream (needs broker:admin; refused in managed role). Reports ok, status, latency and an error code."""
        return await json_request('POST', '/api/broker/upstreams/' + quote(id, safe='') + '/test', ctx=ctx)

    @mcp.tool(annotations=infer)
    async def messages(payload: AnthropicMessagesCall, ctx: Context = None) -> dict[str, Any]:
        """Anthropic Messages (non-streaming) through the broker: a local model, a published alias or a short qualified ref. Read _meta.served_by for where it ran. Use HTTP for streaming."""
        response = await request('POST', '/v1/messages', ctx=ctx, json=payload.model_dump(mode='json', exclude_unset=True))
        return with_provenance(response.json(), response)

    @mcp.tool(annotations=infer)
    async def count_tokens(model: ModelName, messages: Annotated[list[dict[str, Any]], Field(min_length=1)],
                           ctx: Context = None) -> dict[str, Any]:
        """Count the input tokens of Anthropic-style messages for a model (an Anthropic upstream, or a local llama-cpp model counted natively)."""
        response = await request('POST', '/v1/messages/count_tokens', ctx=ctx, json={'model': model, 'messages': messages})
        return with_provenance(response.json(), response)

    @mcp.tool(annotations=probe)
    async def moderate(model: ModelName, input: str | list[str], ctx: Context = None) -> dict[str, Any]:
        """Classify text with a cloud moderation model (OpenAI moderations shape). Local models cannot moderate (classifier_unavailable)."""
        response = await request('POST', '/v1/moderations', ctx=ctx, json={'model': model, 'input': input})
        return with_provenance(response.json(), response)

    @mcp.tool(annotations=read)
    async def whoami(ctx: Context = None) -> dict[str, Any]:
        """The caller's own principal: id, kind, scopes, constraints, limits, remaining budget and expiry (never a secret)."""
        return await json_request('GET', '/api/keys/self', ctx=ctx)


__all__ = ['AnthropicMessagesCall', 'TOOL_NAMES', 'register_mcp']
