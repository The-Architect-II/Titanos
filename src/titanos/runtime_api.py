"""Existing HTTP/MCP setup surfaces for a server-owned native runtime catalog."""
import asyncio
from typing import Annotated, Any

from fastapi import HTTPException
from filelock import Timeout
from pydantic import BaseModel, ConfigDict, Field


class RuntimeProfile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    profile: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]*$', max_length=80)


RuntimeId = Annotated[str, Field(pattern=r'^[a-f0-9]{32}$')]


def register_http(app, setup):
    """Register inside Titanos's existing Guard; owner closes setup at shutdown."""
    async def call(method, *args):
        try:
            return await asyncio.to_thread(method, *args)
        except FileNotFoundError as exc:
            raise HTTPException(404, 'Runtime setup or configured catalog not found') from exc
        except Timeout as exc:
            raise HTTPException(409, 'Another operation owns this runtime installation') from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/api/runtime-setups/plan')
    async def runtime_plan(payload: RuntimeProfile):
        return await call(setup.plan, payload.profile)

    @app.post('/api/runtime-setups', status_code=202)
    async def runtime_prepare(payload: RuntimeProfile):
        return await call(setup.submit, payload.profile)

    @app.get('/api/runtime-setups')
    async def runtime_list():
        return await call(setup.list)

    @app.get('/api/runtime-setups/{identifier}')
    async def runtime_status(identifier: RuntimeId):
        return await call(setup.get, identifier)

    @app.delete('/api/runtime-setups/{identifier}')
    async def runtime_cancel(identifier: RuntimeId):
        return await call(setup.cancel, identifier)

    @app.post('/api/runtime-setups/{identifier}/retry', status_code=202)
    async def runtime_retry(identifier: RuntimeId):
        return await call(setup.retry, identifier)


def register_mcp(mcp, json_request):
    """Use the same internal HTTP transport as existing tools; each tool passes its Context so a mounted call runs as
    the outer request's principal (mcp_server.request). mcp is imported here: the HTTP half needs no MCP install."""
    from mcp.server.mcpserver import Context
    read = {'readOnlyHint': True, 'openWorldHint': False}
    write = {'readOnlyHint': False, 'openWorldHint': True}

    @mcp.tool(annotations=read)
    async def plan_runtime_setup(payload: RuntimeProfile, ctx: Context = None) -> dict[str, Any]:
        """Plan a configured native runtime profile without installing or loading it."""
        return await json_request('POST', '/api/runtime-setups/plan', ctx=ctx, json=payload.model_dump())

    @mcp.tool(annotations=write)
    async def prepare_runtime(payload: RuntimeProfile, ctx: Context = None) -> dict[str, Any]:
        """Acquire and verify a pinned native runtime; preserve the returned setup ID."""
        return await json_request('POST', '/api/runtime-setups', ctx=ctx, json=payload.model_dump())

    @mcp.tool(annotations=read)
    async def get_runtime_setup(identifier: RuntimeId, ctx: Context = None) -> dict[str, Any]:
        """Read durable runtime preparation progress, result and failure details."""
        return await json_request('GET', '/api/runtime-setups/' + identifier, ctx=ctx)

    @mcp.tool(annotations=write)
    async def cancel_runtime_setup(identifier: RuntimeId, ctx: Context = None) -> dict[str, Any]:
        """Cancel runtime acquisition/preparation; installed runtimes remain intact."""
        return await json_request('DELETE', '/api/runtime-setups/' + identifier, ctx=ctx)

    @mcp.tool(annotations=write)
    async def retry_runtime_setup(identifier: RuntimeId, ctx: Context = None) -> dict[str, Any]:
        """Retry a failed/cancelled/interrupted runtime plan, preserving its identity."""
        return await json_request('POST', '/api/runtime-setups/' + identifier + '/retry', ctx=ctx)
