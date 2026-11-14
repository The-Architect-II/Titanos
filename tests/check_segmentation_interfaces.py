"""Actual CLI/client/MCP serialization with a controlled HTTP response; no model runtime."""
import asyncio
import base64
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import httpx
from PIL import Image
from mcp.server.mcpserver.exceptions import ToolError

from titanos.cli import main as cli
from titanos.client import Client, image_part
from titanos.mcp_server import create_mcp
from titanos.segmentation import SegmentationResult


POINTS = [[1.25, 2.75], [7.5, 3.25]]
LABELS = [1, 0]
BOX = [0.5, 0.25, 14.25, 6.75]


def client_and_cli(result, image, folder):
    calls = []

    def handle(request):
        calls.append((request.method, request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=result)

    transport = httpx.MockTransport(handle)
    common = {'model': 'sam', 'image_data_url': image_part(image)['image_url']['url']}
    with Client(transport=transport) as client:
        value = client.segment('sam', image, points=POINTS, point_labels=LABELS,
                               multimask=False, mask_threshold=0, timeout=500, worker='cpu')
        assert isinstance(value, SegmentationResult)
        assert calls[-1] == ('POST', '/v1/images/segment', {
            **common, 'points': POINTS, 'point_labels': LABELS,
            'multimask': False, 'mask_threshold': 0, 'timeout': 500, 'worker': 'cpu'})
        assert calls[-1][2]['multimask'] is False
        client.segment('sam', image, box=BOX)
        assert calls[-1][2] == {**common, 'box': BOX}, 'Omitted policy must remain omitted'

    points_file = folder / 'points.json'; points_file.write_text(json.dumps(POINTS), encoding='utf-8-sig')
    labels_file = folder / 'labels.json'; labels_file.write_text(json.dumps(LABELS), encoding='utf-8-sig')
    options_file = folder / 'options.json'; options_file.write_text('{"location":"local_only"}', encoding='utf-8')
    stdout = io.StringIO()
    # Patch only the transport factory: the real parser, command handler and Client.segment execute.
    with patch('titanos.client.Client', side_effect=lambda *args, **kwargs: Client(*args, transport=transport, **kwargs)), \
            redirect_stdout(stdout):
        cli(['segment', 'sam', str(image), '--points', str(points_file), '--point-labels', str(labels_file),
             '--no-multimask', '--mask-threshold', '0', '--deadline-ms', '500', '--worker', 'cpu',
             '--options', str(options_file)])
        assert json.loads(stdout.getvalue()) == SegmentationResult.model_validate(result).model_dump(mode='json')
        assert calls[-1][2] == {**common, 'points': POINTS, 'point_labels': LABELS,
            'multimask': False, 'mask_threshold': 0, 'timeout': 500, 'worker': 'cpu', 'location': 'local_only'}
        assert calls[-1][2]['multimask'] is False
        cli(['segment', 'sam', str(image), '--box', *map(str, BOX)])
        assert calls[-1][2] == {**common, 'box': BOX}
        cli(['segment', 'sam', str(image), '--box', *map(str, BOX), '--multimask', '--mask-threshold', '-1.125'])
        assert calls[-1][2] == {**common, 'box': BOX, 'multimask': True, 'mask_threshold': -1.125}
        before = len(calls)
        for arguments in [[], ['--points', str(points_file)],
                          ['--box', *map(str, BOX), '--point-labels', str(labels_file)],
                          ['--box', *map(str, BOX), '--points', str(points_file)]]:
            with redirect_stderr(io.StringIO()):
                try: cli(['segment', 'sam', str(image), *arguments])
                except SystemExit as error: assert error.code in {1, 2}
                else: raise AssertionError('Incomplete or conflicting prompt reached HTTP')
        assert len(calls) == before

    with Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))) as client:
        try: client.segment('sam', image, box=BOX)
        except ValueError: pass
        else: raise AssertionError('Python accepted an invalid typed result')


async def mcp_calls(result, png):
    calls = []

    async def app(scope, receive, send):
        body = b''
        while True:
            message = await receive(); body += message.get('body', b'')
            if not message.get('more_body'): break
        calls.append((scope['method'], scope['path'], json.loads(body)))
        await send({'type': 'http.response.start', 'status': 200,
                    'headers': [(b'content-type', b'application/json')]})
        await send({'type': 'http.response.body', 'body': json.dumps(result).encode()})

    mcp = create_mcp(app=app)
    tool = next(tool for tool in await mcp.list_tools() if tool.name == 'segment_image')
    schema = tool.input_schema
    request_schema = schema['$defs']['SegmentationRequest']
    assert {'box', 'points', 'point_labels', 'multimask', 'mask_threshold', 'timeout'} <= set(request_schema['properties'])
    assert tool.output_schema and 'predicted_iou' in json.dumps(tool.output_schema)
    common = {'model': 'sam', 'image_base64': base64.b64encode(png).decode('ascii')}
    for options in [{'box': BOX}, {'points': POINTS, 'point_labels': LABELS,
            'multimask': False, 'mask_threshold': 0, 'timeout': 500, 'worker': 'cpu'}]:
        payload = {**common, **options}
        reply = await mcp.call_tool('segment_image', {'payload': payload})
        assert not reply.is_error, reply
        data = reply.structured_content; data = data.get('result', data)
        assert SegmentationResult.model_validate(data).model_dump(mode='json') == SegmentationResult.model_validate(result).model_dump(mode='json')
        assert calls[-1] == ('POST', '/v1/images/segment', payload)
        if 'multimask' in options: assert calls[-1][2]['multimask'] is False
    before = len(calls)
    for values in [{'box': BOX, 'points': POINTS, 'point_labels': LABELS},
                   {'points': POINTS}, {'box': BOX, 'multimask': 'false'},
                   {'box': BOX, 'mask_threshold': True}, {'box': BOX, 'unknown': 1}]:
        try: reply = await mcp.call_tool('segment_image', {'payload': {**common, **values}})
        except ToolError: pass
        else: assert reply.is_error, 'Invalid typed MCP input reached HTTP'
    assert len(calls) == before


def main():
    registration_help = io.StringIO()
    with redirect_stdout(registration_help):
        try: cli(['add', '--help'])
        except SystemExit as error: assert error.code == 0
        else: raise AssertionError('Registration help should exit before changing the registry')
    assert 'image-segmentation' in registration_help.getvalue()
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        with Image.new('RGB', (16, 8), (19, 71, 143)) as original:
            stream = io.BytesIO(); original.save(stream, format='PNG'); png = stream.getvalue()
            input_hash = hashlib.sha256(original.tobytes()).hexdigest()
        image = folder / 'image.png'; image.write_bytes(png)
        with Image.new('RGBA', (16, 8), (0, 0, 0, 0)) as mask:
            stream = io.BytesIO(); mask.save(stream, format='PNG'); mask_png = stream.getvalue()
        result = {
            'model': 'sam', 'task': 'image-segmentation', 'width': 16, 'height': 8,
            'input_rgb_sha256': input_hash, 'multimask': False, 'mask_threshold': 0,
            'prompt': {'box': BOX, 'multimask': False, 'mask_threshold': 0},
            'model_output_shape': [1, 1, 1, 4, 4], 'model_class': 'ControlledModel',
            'processor_class': 'ControlledProcessor', 'device': 'cpu',
            'masks': [{'index': 0, 'native_mask_token_index': 0, 'predicted_iou': 0.75,
                       'png_base64': base64.b64encode(mask_png).decode('ascii'),
                       'png_sha256': hashlib.sha256(mask_png).hexdigest(), 'selected_pixels': 128}],
        }
        result = SegmentationResult.model_validate(result).model_dump(mode='json')
        client_and_cli(result, image, folder)
        asyncio.run(mcp_calls(result, png))
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    print('PASS: real Python/CLI/MCP serialization preserves coordinates, labels, omission, false, zero and deadlines; typed errors reject before HTTP; no model runtime')


if __name__ == '__main__': main()
