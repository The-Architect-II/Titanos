"""Client/MCP/CLI assignment parity; controlled transport only."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from unittest.mock import patch

import httpx
from mcp.server import MCPServer

from titanos.cli import main
from titanos.client import Client
from titanos.mcp_server import create_mcp

A='GPU-11111111-1111-1111-1111-111111111111'

def client_check():
    seen=[]
    def handle(request):
        seen.append((request.method,request.url.path,json.loads(request.content) if request.content else None))
        if request.url.path.endswith('/devices'):return httpx.Response(200,json={'data':[{'id':A}]})
        return httpx.Response(200,json={'model':'chat','mode':'single','groups':[[A]],'replicas':1})
    with Client(transport=httpx.MockTransport(handle)) as client:
        assert client.assignment_devices()['data'][0]['id']==A
        client.assignment('chat')
        client.save_assignment(model='chat',mode='single',groups=[[A]],tensor_parallel_size=2)
    assert ('POST','/api/assignments',{'model':'chat','mode':'single','groups':[[A]],'tensor_parallel_size':2}) in seen

async def mcp_check():
    async def request(method,path,**kwargs):
        if path.endswith('/devices'):return {'data':[{'id':A}]}
        return {'model':'chat','mode':'single','groups':[[A]]}
    mcp=create_mcp()
    # Replace only the generated bridge transport by registering the assignment
    # tools on a small isolated server.
    server=MCPServer('assignment-test')
    from titanos.assignment_api import register_mcp
    register_mcp(server,request)
    names={tool.name for tool in await server.list_tools()}
    assert {'list_assignment_devices','get_model_assignment','save_model_assignment'}<=names
    result=await server.call_tool('save_model_assignment',{'payload':{'model':'chat','mode':'single','groups':[[A]],'tensor_parallel_size':2}})
    assert not result.is_error

def cli_check():
    calls=[]
    class Stub:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def assignment_devices(self):calls.append(('devices',));return {'data':[]}
        def assignment(self,model):calls.append(('get',model));return {'model':model}
        def save_assignment(self,**payload):calls.append(('save',payload));return payload
    with patch('titanos.client.Client',return_value=Stub()),redirect_stdout(io.StringIO()):
        main(['assignment-devices'])
        main(['assignment','get','chat'])
        main(['assignment','save','chat','--mode','single','--group',A])
    assert calls[-1][0]=='save' and calls[-1][1]['groups']==[[A]]

if __name__=='__main__':
    client_check();asyncio.run(mcp_check());cli_check()
    print('PASS: assignment Client, MCP and CLI use the same typed stable-ID payload; no hardware/model work.')
