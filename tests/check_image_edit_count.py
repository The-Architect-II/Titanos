"""Actual MCP -> Titanos HTTP count/default/limit checks; inference is a double."""
import asyncio
import base64
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, PropertyMock

from PIL import Image
from fastapi.testclient import TestClient
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from titanos import cli
from titanos.client import Client
from titanos.config import Config, ModelSpec
from titanos.server import create_app
from titanos.settings import Settings


def png(mask=False):
    with Image.new('RGBA' if mask else 'RGB',(32,32),(7,11,13,255) if mask else (7,11,13)) as im:
        if mask:im.paste((0,0,0,0),(0,0,16,32))
        stream=io.BytesIO();im.save(stream,format='PNG');return stream.getvalue()


class Checks(unittest.TestCase):
    def test_actual_tool_to_http_count_and_model_default_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            calls=[]
            app=create_app(Config(models={'asset':ModelSpec(path='unused',backend='diffusers',task='image',
                defaults={'n':3,'size':'32x32'})}),output_dir=Path(directory)/'out',settings=Settings(max_generated_images=3))
            def infer(name,task,data,emit=None,cancel=None):
                calls.append((task,data['n'],'mask_image' in data))
                return [Image.new('RGB',(data['width'],data['height']),(19,23,29)) for _ in range(data['n'])]
            app.state.engine.target.infer=infer
            raw=png();mask=png(True)
            base={'model':'asset','prompt':'edit','image_base64':base64.b64encode(raw).decode()}
            async def invoke():
                schema=next(tool.input_schema for tool in await app.state.mcp.list_tools() if tool.name=='edit_image')
                self.assertIn('n',schema['properties']);self.assertNotIn('n',schema.get('required',[]))
                self.assertIn('mask_base64',schema['properties']);self.assertIn('preserve_unmasked',schema['properties'])
                for masked in (False,True):
                    arguments={**base,**({'mask_base64':base64.b64encode(mask).decode(),'preserve_unmasked':False} if masked else {})}
                    for extra,expected in [({'n':2},2),({},3)]:
                        result=await app.state.mcp.call_tool('edit_image',{**arguments,**extra})
                        self.assertFalse(result.is_error)
                        self.assertEqual(len(result.structured_content['data']),expected)
                        self.assertEqual(calls[-1],('image-edit',expected,masked))
                        if masked:self.assertFalse(result.structured_content['masking']['preserve_unmasked'])
                before=len(calls)
                for value in (True,False,0,-1,2.0,'2',4):
                    with self.subTest(n=value):
                        try:
                            result=await app.state.mcp.call_tool('edit_image',{**base,'n':value})
                        except ToolError as exc:
                            self.assertIn('n',str(exc))
                            if value == 4:self.assertIn('MAX_GENERATED_IMAGES',str(exc))
                        else:
                            self.assertTrue(result.is_error)
                        self.assertEqual(len(calls),before,'Invalid n must not reach inference')
            with TestClient(app,base_url='http://127.0.0.1') as http:
                with patch.object(Context,'request_context',new_callable=PropertyMock,return_value=NS(request=None)):
                    asyncio.run(invoke())
                before=len(calls)
                for value in ('0','true','4',''):
                    response=http.post('/v1/images/edits',data={'model':'asset','prompt':'x','n':value},files={'image':('source.png',raw)})
                    self.assertIn(response.status_code,(400,422));self.assertEqual(len(calls),before)
                response=http.post('/v1/images/edits',data={'model':'asset','prompt':'x'},
                    files=[('image',('source.png',raw)),('n',(None,'1')),('n',(None,'2'))])
                self.assertEqual(response.status_code,400);self.assertEqual(len(calls),before)

    def test_cli_edit_count_and_omission(self):
        for arguments,expected in [(['--n','2'],{'n':2}),([], {})]:
            with patch.object(Client,'edit',return_value={'created':1,'data':[]}) as edit,redirect_stdout(io.StringIO()):
                cli.main(['edit','asset','edit','--image','source.png',*arguments])
            self.assertEqual(edit.call_args.args,('asset',Path('source.png'),'edit'))
            self.assertEqual(edit.call_args.kwargs,expected)


if __name__=='__main__':unittest.main(verbosity=2)
