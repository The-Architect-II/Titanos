"""Rerank caller cancellation and executor wait; no numerical model is loaded."""
import asyncio
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

import httpx
from titanos.config import Config, ModelSpec
from titanos.server import create_app


async def check():
    for mode in ['http_disconnect', 'mcp_cancel', 'mcp_extension_cancel', 'queued_deadline']:
        with tempfile.TemporaryDirectory() as folder:
            app = create_app(Config(models={'ranker': ModelSpec(path=folder, task='rerank', device='cpu')}),
                             output_dir=Path(folder))
            started, observed = threading.Event(), threading.Event()
            calls = []

            def infer(model, task, data, emit=None, cancel=None):
                calls.append((model, task))
                started.set()
                assert cancel.wait(2), 'Caller cancellation did not reach the backend'
                observed.set()
                return {}  # Cancellation must win over validation of this late response.

            app.state.engine.target.infer = infer
            payload = {'model': 'ranker', 'query': 'development query', 'documents': ['development document'], 'timeout': 5000}
            try:
                if mode == 'http_disconnect':
                    first = True
                    sent = []

                    async def receive():
                        nonlocal first
                        if first:
                            first = False
                            return {'type': 'http.request', 'body': json.dumps(payload).encode(), 'more_body': False}
                        while not started.is_set():
                            await asyncio.sleep(.005)
                        return {'type': 'http.disconnect'}

                    async def send(message):
                        sent.append(message)

                    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST',
                             'scheme': 'http', 'path': '/v1/rerank', 'raw_path': b'/v1/rerank', 'root_path': '',
                             'query_string': b'', 'client': ('127.0.0.1', 50000), 'server': ('127.0.0.1', 80),
                             'headers': [(b'host', b'127.0.0.1'), (b'content-type', b'application/json')]}
                    await asyncio.wait_for(app(scope, receive, send), 3)
                    body = json.loads(b''.join(m.get('body', b'') for m in sent))
                    assert body['error']['code'] == 'cancelled'
                elif mode.startswith('mcp'):
                    tool = 'rerank'
                    if mode == 'mcp_extension_cancel':
                        tool = 'extension_infer'
                        payload['operation'] = 'rerank'
                    pending = asyncio.create_task(app.state.mcp.call_tool(tool, {'payload': payload}))
                    assert await asyncio.to_thread(started.wait, 1)
                    pending.cancel()
                    try:
                        await asyncio.wait_for(pending, 3)
                    except asyncio.CancelledError:
                        pass
                    else:
                        raise AssertionError('MCP cancellation was swallowed')
                else:
                    actual = asyncio.to_thread

                    async def delayed(function, *args, **kwargs):
                        await asyncio.sleep(.03)
                        return await actual(function, *args, **kwargs)

                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1') as client:
                        with patch('titanos.api_plus.asyncio.to_thread', side_effect=delayed):
                            for route, fields in [('/v1/rerank', {}), ('/v1/extensions/infer', {'operation': 'rerank'})]:
                                response = await client.post(route, json={**payload, **fields, 'timeout': 5})
                                assert response.status_code == 504 and response.json()['error']['code'] == 'deadline_exceeded'
                if mode == 'queued_deadline':
                    assert calls == []
                else:
                    assert calls == [('ranker', 'rerank')] and observed.is_set()
            finally:
                await app.state.native.close()
                await asyncio.to_thread(app.state.engine.close)
    print('PASS: rerank HTTP/MCP caller cancellation and shared queue deadline; no numerical inference')


if __name__ == '__main__':
    asyncio.run(check())
