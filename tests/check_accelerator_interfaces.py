"""Client/CLI/MCP parity and frozen-shell-compatible UUID configuration, no drivers."""
import asyncio
import builtins
from contextlib import redirect_stdout,redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from mcp.server import MCPServer
from titanos.assignment_api import register_mcp
from titanos.cli import main
from titanos.client import Client
from titanos.config import Config,ModelSpec,read_config,write_config

A='GPU-11111111-1111-1111-1111-111111111111'
B='GPU-22222222-2222-2222-2222-222222222222'
ROWS=[{'uuid':A,'index':0,'name':'Card A','capability':12.0},{'uuid':B,'index':1,'name':'Card B','capability':12.0}]

class Checks(unittest.TestCase):
    def test_client_model_refresh_policy_and_profile_fields(self):
        seen=[]
        def route(request):
            seen.append((request.method,request.url.path,dict(request.url.params),json.loads(request.content) if request.content else None))
            return httpx.Response(200,json={'data':[]})
        with Client(transport=httpx.MockTransport(route)) as client:
            client.assignment_devices(model='m',refresh=True)
            client.save_assignment(model='m',groups=[],policy='auto')
            client.activate_acquisition('a'*32,load=False,runtime_profile='controlled-cuda')
        self.assertEqual(seen[0][2],{'model':'m','refresh':'true'})
        self.assertEqual(seen[1][3]['policy'],'auto')
        self.assertEqual(seen[2][3]['runtime_profile'],'controlled-cuda')

    def test_mcp_model_refresh_fields(self):
        seen=[]
        async def request(method,path,**kwargs):seen.append((method,path,kwargs));return {'data':[]}
        server=MCPServer('accelerator-contract');register_mcp(server,request)
        async def run():
            result=await server.call_tool('list_assignment_devices',{'model':'m','refresh':True})
            self.assertFalse(result.is_error)
        asyncio.run(run())
        self.assertEqual(seen[0][2]['params'],{'model':'m','refresh':'true'})

    def native_fixture(self,root):
        path=root/'registry.json'
        write_config(path,Config(models={'m':ModelSpec(path=str(root/'model'),task='chat',backend='llama-cpp',device='auto',native={'model_file':'model.gguf'})}))
        return path

    def no_torch(self):
        original=builtins.__import__
        def imported(name,*args,**kwargs):
            if name.split('.')[0]=='torch':raise AssertionError('Native worker configuration imported Torch')
            return original(name,*args,**kwargs)
        return patch('builtins.__import__',side_effect=imported)

    def test_full_uuid_native_worker_does_not_require_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);registry=self.native_fixture(root)
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':B}),patch('titanos.devices.scan',return_value=ROWS),self.no_torch(),redirect_stdout(io.StringIO()):
                main(['--config',str(registry),'worker','worker','--models','m','--gpus',B])
            self.assertEqual(read_config(registry).workers['worker'].gpus,[B])

    def test_native_numeric_cuda_index_binds_current_runtime_uuid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);registry=self.native_fixture(root);executable=root/'runtime/llama-server.exe'
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch('titanos.devices.scan',return_value=ROWS),\
                 patch('titanos.native_install.executable_path',return_value=executable),\
                 patch('titanos.devices.probe_visible_cuda',return_value=[{'uuid':B,'runtime_address':'cuda:0'}]) as probe,\
                 self.no_torch(),redirect_stdout(io.StringIO()):
                main(['--config',str(registry),'worker','worker','--models','m','--gpus','0'])
            self.assertEqual(read_config(registry).workers['worker'].gpus,[B])
            self.assertEqual(probe.call_args.args,(executable.parent,))

    def test_masked_uuid_does_not_mutate_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);registry=self.native_fixture(root);before=registry.read_bytes()
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch('titanos.devices.scan',return_value=ROWS),self.no_torch(),redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:main(['--config',str(registry),'worker','worker','--models','m','--gpus',A])
                self.assertEqual(stopped.exception.code,1)
            self.assertEqual(registry.read_bytes(),before)

    def test_cli_activation_and_assignment_policy_forwarded(self):
        calls=[]
        class Stub:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def activate_acquisition(self,identifier,**kwargs):calls.append(('activate',identifier,kwargs));return {'status':'active'}
            def assignment_devices(self,**kwargs):calls.append(('devices',kwargs));return {'data':[]}
            def save_assignment(self,**kwargs):calls.append(('assignment',kwargs));return kwargs
        with patch('titanos.client.Client',return_value=Stub()),redirect_stdout(io.StringIO()):
            main(['acquisitions','activate','a'*32,'--no-load','--runtime-profile','controlled-cuda'])
            main(['assignment-devices','--model','m','--refresh'])
            main(['assignment','save','m','--policy','cpu'])
        self.assertEqual(calls[0][2]['runtime_profile'],'controlled-cuda')
        self.assertEqual(calls[1][1],{'model':'m','refresh':True})
        self.assertEqual(calls[2][1]['policy'],'cpu')


if __name__=='__main__':unittest.main(verbosity=2)
