"""MCP bridge for Titanos-owned stable local worker assignments."""
from typing import Any
from urllib.parse import quote

from mcp.types import ToolAnnotations

from .server import AssignmentRequest


def register_mcp(mcp,json_request):
    from mcp.server.mcpserver import Context  # each tool runs as the outer request's principal
    read=ToolAnnotations(read_only_hint=True,open_world_hint=False)
    change=ToolAnnotations(read_only_hint=False,destructive_hint=False,idempotent_hint=True,open_world_hint=False)

    @mcp.tool(annotations=read)
    async def list_assignment_devices(model:str|None=None,refresh:bool=False,ctx:Context=None)->dict[str,Any]:
        """Inspect installed devices, runtime visibility and optional model compatibility; enumeration is not execution proof."""
        params={**({'model':model} if model is not None else {}),**({'refresh':'true'} if refresh else {})}
        return await json_request('GET','/api/assignments/devices',ctx=ctx,**({'params':params} if params else {}))

    @mcp.tool(annotations=read)
    async def get_model_assignment(model:str,ctx:Context=None)->dict[str,Any]:
        """Read persisted local worker groups, execution mode and replica count for one model."""
        return await json_request('GET','/api/assignments/'+quote(model,safe=''),ctx=ctx)

    @mcp.tool(annotations=change)
    async def save_model_assignment(payload:AssignmentRequest,ctx:Context=None)->dict[str,Any]:
        """Persist stable device groups through the idle registry reload; missing devices fail instead of falling back."""
        return await json_request('POST','/api/assignments',ctx=ctx,json=payload.model_dump(mode='json'))
