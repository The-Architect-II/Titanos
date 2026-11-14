"""Public HTTP/MCP ONNX selectors preserve validation without running a model."""
import asyncio,json,tempfile
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError
from titanos.config import Config,read_config,write_config
from titanos.server import create_app


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);checkpoint=root/'model';checkpoint.mkdir();registry=root/'models.json';cfg=Config();write_config(registry,cfg)
        app=create_app(cfg,output_dir=root/'output',registry_path=registry)
        spec={'path':str(checkpoint),'task':'image','backend':'onnx-diffusers','device':'dml','allow_cpu_fallback':True,
              'load_options':{'device_id':0,'adapter_luid':'00000000:000190ef'},'defaults':{'steps':30,'seed':303}}
        with patch('titanos.engine.Engine.load',side_effect=AssertionError('Registration loaded a model')) as load,TestClient(app,base_url='http://127.0.0.1') as client:
            response=client.post('/api/models/register',json={'name':'image','spec':spec});assert response.status_code==200,response.text
            assert read_config(registry).models['image'].load_options==spec['load_options']
            capability=client.post('/api/models/capabilities',json={'model':'image'}).json();assert capability['instruction_edit'] is None and capability['image_edit'] is False
            async def replace(payload):return await app.state.mcp.call_tool('register_model',{'payload':payload})
            updated={**spec,'allow_cpu_fallback':False,'load_options':{'device_id':2,'adapter_luid':'00000001:000190ef'}}
            reply=asyncio.run(replace({'name':'image','replace':True,'spec':updated}));assert not reply.is_error
            assert read_config(registry).models['image'].load_options==updated['load_options'] and cfg.models['image'].allow_cpu_fallback is False
            for value in (True,False):
                diagnostic={**updated,'load_options':{**updated['load_options'],'disable_metacommands':value}}
                body={'name':'image','replace':True,'spec':diagnostic}
                response=client.post('/api/models/register',json=body);assert response.status_code==200,response.text
                assert read_config(registry).models['image'].load_options['disable_metacommands'] is value
                reply=asyncio.run(replace(body));assert not reply.is_error
                assert read_config(registry).models['image'].load_options['disable_metacommands'] is value
                capability=client.post('/api/models/capabilities',json={'model':'image'}).json()
                assert capability['load_controls']['disable_metacommands']['type']=='boolean' and capability['qualified'] is None
            # Replacing with omission restores an omitted field, not an invented false override.
            reply=asyncio.run(replace({'name':'image','replace':True,'spec':updated}));assert not reply.is_error
            assert 'disable_metacommands' not in read_config(registry).models['image'].load_options
            for change in [
                {'load_options':{**updated['load_options'],'profile_dir':str(root/'profile-output')}},
                {'load_options':{**updated['load_options'],'device_id':True}},
                {'load_options':{**updated['load_options'],'adapter_luid':'UPPERCASE:NOTVALID'}},
                {'load_options':{**updated['load_options'],'custom_pipeline':'outside'}},
                {'device':'cpu'},
                {'defaults':{'callback':'not-a-public-operation'}},
            ]+[{'load_options':{**updated['load_options'],'disable_metacommands':value}} for value in (0,1,'true','false',None,{},[])]:
                before=registry.read_bytes();body={'name':'image','replace':True,'spec':{**updated,**change}}
                response=client.post('/api/models/register',json=body);assert 400<=response.status_code<500,response.text
                assert registry.read_bytes()==before
                try:reply=asyncio.run(replace(body))
                except ToolError:pass
                else:assert reply.is_error
                assert registry.read_bytes()==before
            assert not (root/'profile-output').exists() and load.call_count==0
    print('PASS HTTP/MCP data-only DXGI registration, exact replacement, discovery and atomic rejection; no model loads or profile-directory creation')


if __name__=='__main__':main()
