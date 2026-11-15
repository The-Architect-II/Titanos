"""Model-free mask transport checks: client/CLI, actual MCP dispatch, and file ownership."""

import asyncio
import base64
from contextlib import redirect_stdout
from email.parser import BytesParser
from email.policy import default
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import PropertyMock, patch

import httpx
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from titanos import cli
from titanos.client import Client, TitanosError
from titanos.mcp_server import create_mcp


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==')
INLINE = base64.b64encode(PNG).decode('ascii')
RESULT = {'created': 1, 'data': [{'b64_json': INLINE}]}


def multipart(request):
    message = BytesParser(policy=default).parsebytes(
        ('Content-Type: ' + request.headers['content-type'] + '\r\n\r\n').encode() + request.content)
    return {part.get_param('name', header='content-disposition'): part.get_payload(decode=True)
            for part in message.iter_parts()}


class MaskInterfaces(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'image.png'
        self.mask = self.root / 'mask.png'
        self.image.write_bytes(PNG)
        self.mask.write_bytes(PNG)

    def test_client_multipart_mask_and_inline_omission_false_and_exclusion(self):
        seen = []

        def handle(request):
            self.assertEqual((request.method, request.url.path), ('POST', '/v1/images/edits'))
            seen.append(multipart(request))
            return httpx.Response(200, json=RESULT)

        with Client('http://127.0.0.1', transport=httpx.MockTransport(handle)) as client:
            self.assertEqual(client.edit('asset', self.image, 'edit', mask=self.mask,
                                         preserve_unmasked=False, seed=0), RESULT)
            self.assertEqual(seen[-1], {'model': b'asset', 'prompt': b'edit', 'image': PNG,
                                       'mask': PNG, 'preserve_unmasked': b'false', 'seed': b'0'})
            client.edit('asset', self.image, 'edit', mask_base64=INLINE, preserve_unmasked=True)
            self.assertEqual(seen[-1]['mask_base64'], INLINE.encode())
            self.assertEqual(seen[-1]['preserve_unmasked'], b'true')
            self.assertNotIn('mask', seen[-1])
            client.edit('asset', self.image, 'edit')
            self.assertEqual(set(seen[-1]), {'model', 'prompt', 'image'})
            with patch.object(Path, 'open') as opened, self.assertRaisesRegex(ValueError, 'not both'):
                client.edit('asset', self.image, 'edit', mask=self.mask, mask_base64=INLINE)
            opened.assert_not_called()
            self.assertEqual(len(seen), 3)

    def test_client_closes_both_files_on_success_and_failures(self):
        original_open = Path.open
        opened = []

        def track(path, *args, **kwargs):
            file = original_open(path, *args, **kwargs)
            opened.append(file)
            return file

        for outcome in ('success', 'network', 'server', 'json'):
            opened.clear()

            def handle(request):
                self.assertEqual(len(opened), 2)
                self.assertTrue(all(not file.closed for file in opened))
                if outcome == 'network':
                    raise httpx.ReadError('fixture connection lost', request=request)
                if outcome == 'server':
                    return httpx.Response(422, json={'error': {'message': 'fixture rejection'}})
                if outcome == 'json':
                    return httpx.Response(200, text='invalid JSON')
                return httpx.Response(200, json=RESULT)

            with self.subTest(outcome=outcome), Client(transport=httpx.MockTransport(handle)) as client:
                with patch.object(Path, 'open', track):
                    if outcome == 'success':
                        client.edit('asset', self.image, 'edit', mask=self.mask)
                    else:
                        with self.assertRaises(TitanosError if outcome != 'json' else ValueError):
                            client.edit('asset', self.image, 'edit', mask=self.mask)
                self.assertEqual(len(opened), 2)
                self.assertTrue(all(file.closed for file in opened))

        opened.clear()
        with Client() as client, patch.object(Path, 'open', track), self.assertRaises(FileNotFoundError):
            client.edit('asset', self.image, 'edit', mask=self.root / 'missing.png')
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)

    def test_cli_boolean_options_and_omission(self):
        for flags, expected in (([], {}), (['--mask', str(self.mask)], {'mask': self.mask}),
                                (['--mask', str(self.mask), '--preserve-unmasked'],
                                 {'mask': self.mask, 'preserve_unmasked': True}),
                                (['--mask', str(self.mask), '--no-preserve-unmasked'],
                                 {'mask': self.mask, 'preserve_unmasked': False})):
            with self.subTest(flags=flags), patch.object(Client, 'edit', return_value=RESULT) as edit:
                with redirect_stdout(io.StringIO()):
                    cli.main(['edit', 'asset', 'edit', '--image', str(self.image), *flags])
                self.assertEqual(edit.call_args.args, ('asset', self.image, 'edit'))
                self.assertEqual(edit.call_args.kwargs, expected)

    def test_actual_mcp_dispatch_advertises_and_forwards_mask_rejects_aliases(self):
        seen = []

        async def receive(request: Request):
            async with request.form() as form:
                fields = {key: value for key, value in form.items() if isinstance(value, str)}
                files = {key: await value.read() for key, value in form.items() if not isinstance(value, str)}
                seen.append((fields, files))
            return JSONResponse(RESULT)

        app = Starlette(routes=[Route('/v1/images/edits', receive, methods=['POST'])])
        mcp = create_mcp(app=app)
        base = {'model': 'asset', 'prompt': 'edit', 'image_base64': INLINE}

        async def invoke():
            schema = next(tool.input_schema for tool in await mcp.list_tools() if tool.name == 'edit_image')
            self.assertIn('mask_base64', schema['properties'])
            self.assertIn('preserve_unmasked', schema['properties'])
            self.assertNotIn('mask', schema['properties'])
            self.assertNotIn('mask_image', schema['properties'])
            result = await mcp.call_tool('edit_image', {**base, 'mask_base64': INLINE, 'preserve_unmasked': False})
            self.assertFalse(result.is_error)
            self.assertEqual(seen[-1], ({'model': 'asset', 'prompt': 'edit', 'preserve_unmasked': 'false'},
                                        {'image': PNG, 'mask': PNG}))
            result = await mcp.call_tool('edit_image', base)
            self.assertFalse(result.is_error)
            self.assertEqual(seen[-1], ({'model': 'asset', 'prompt': 'edit'}, {'image': PNG}))
            for options in ({'mask': INLINE}, {'mask_image': INLINE},
                            {'mask': INLINE, 'mask_base64': INLINE}, {'mask_base64': 'not base64!'}):
                with self.subTest(options=list(options)):
                    try:
                        rejected = await mcp.call_tool('edit_image', {**base, **options})
                    except ToolError as exc:
                        self.assertRegex(str(exc), 'mask|base64')
                    else:
                        self.assertTrue(rejected.is_error)
                    self.assertEqual(len(seen), 2, 'Invalid masks must not produce an unmasked HTTP edit')

        with patch.object(Context, 'request_context', new_callable=PropertyMock,
                          return_value=SimpleNamespace(request=None)):
            asyncio.run(invoke())


if __name__ == '__main__':
    unittest.main(verbosity=2)
