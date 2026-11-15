"""Actual loopback HTTP with the official OpenAI and MCP clients; synthetic inference.

Optional test dependency: python -m pip install openai==2.54.0
No hosted API calls, credentials, model loads or accelerator work are used.
"""
import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from importlib.metadata import version

import httpx
from openai import OpenAI
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
import uvicorn

from titanos.config import Config, ModelSpec
from titanos.server import create_app
from titanos.settings import Settings


def text_of(value):
    return ''.join(part['text'] for item in value['output'] if item['type'] == 'message'
                   for part in item['content'] if part['type'] == 'output_text')


def tool_data(result):
    assert not result.is_error, result
    if result.structured_content is not None:
        data = result.structured_content
        return data.get('result', data)
    return json.loads(next(c.text for c in result.content if getattr(c, 'type', '') == 'text'))


def main():
    seen = []
    outcome = {'finish_reason': 'stop', 'raise': False}
    text = 'Hello caf\u00e9'

    def infer(name, task, data, emit=None, cancel=None):
        seen.append(data)
        if emit:
            emit('Hello ')
            time.sleep(.02)
            emit('caf\u00e9')
        if outcome['raise']:
            raise RuntimeError('Synthetic SDK test failure')
        return {'text': text, 'finish_reason': outcome['finish_reason'],
                'usage': {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}}

    with tempfile.TemporaryDirectory(prefix='titanos-sdk-') as folder:
        root = Path(folder)
        cfg = Config(models={'fixture': ModelSpec(path=folder, task='chat', defaults={'max_output_tokens': 41})})
        settings = Settings(log_level='warning')
        app = create_app(cfg, api_key='local-test-key', output_dir=root/'outputs', settings=settings)
        app.state.engine.target.infer = infer
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        sock.listen()
        port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        server = uvicorn.Server(uvicorn.Config(app, log_level='warning', lifespan='on'))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
        thread.start()
        checks = []
        try:
            deadline = time.monotonic()+5
            while not server.started:
                assert thread.is_alive() and time.monotonic()<deadline, 'Test server did not start'
                time.sleep(.01)
            with OpenAI(base_url=url+'/v1', api_key='local-test-key', max_retries=0,
                        _strict_response_validation=True, http_client=httpx.Client(trust_env=False)) as client:
                response = client.responses.create(model='fixture', input='Hello', store=False, metadata={'test': 'SDK'})
                assert response.output_text == text and response.status == 'completed'
                assert response.max_output_tokens == 41 and response.metadata == {'test': 'SDK'}
                assert response.usage.input_tokens == 7 and response.usage.input_tokens_details.cache_write_tokens == 0
                assert response.usage.output_tokens == 3 and response.usage.total_tokens == 10
                checks.append('OpenAI strict typed nonstream response, output_text, metadata and actual usage')

                replay = [item.model_dump(mode='json') for item in response.output]
                replay.append({'role': 'user', 'content': 'Again'})
                response = client.responses.create(model='fixture', input=replay, max_output_tokens=17)
                assert response.output_text == text and seen[-1]['max_tokens'] == 17
                checks.append('Official SDK output items replay as stateless history')

                with client.responses.create(model='fixture', input='Hello', stream=True) as stream:
                    events = list(stream)
                assert events[0].type == 'response.created' and events[-1].type == 'response.completed'
                assert ''.join(e.delta for e in events if e.type == 'response.output_text.delta') == text
                assert events[-1].response.output_text == text and events[-1].response.usage.total_tokens == 10
                checks.append('Official typed SSE iteration with content and final usage')

                with client.responses.stream(model='fixture', input='Hello') as stream:
                    deltas = [e.delta for e in stream if e.type == 'response.output_text.delta']
                    final = stream.get_final_response()
                assert ''.join(deltas) == final.output_text == text
                checks.append('Official high-level typed deltas and get_final_response for completed output')

                for reason, event_name in [('length', 'response.incomplete'), ('stop', 'response.failed')]:
                    outcome.update(finish_reason=reason, **{'raise': event_name.endswith('failed')})
                    with client.responses.create(model='fixture', input='Hello', stream=True) as stream:
                        events = list(stream)
                    assert events[-1].type == event_name
                    assert not any(e.type == 'response.completed' for e in events)
                    if reason == 'length':
                        assert events[-1].response.incomplete_details.reason == 'max_output_tokens'
                    else:
                        assert events[-1].response.error.code == 'server_error'
                checks.append('Official typed incomplete/failed events never masquerade as completion')
            outcome.update(finish_reason='stop', **{'raise': False})

            async def mcp_checks():
                headers = {'Authorization': 'Bearer local-test-key'}
                async with httpx.AsyncClient(headers=headers, trust_env=False) as client:
                    async with streamable_http_client(url+'/mcp/', http_client=client) as streams:
                        async with ClientSession(streams[0], streams[1]) as session:
                            await session.initialize()
                            result = tool_data(await session.call_tool('responses', {'payload': {'model': 'fixture', 'input': 'Hello'}}))
                            assert text_of(result) == text and result['max_output_tokens'] == 41
                env = {**os.environ, 'TITANOS_HOME': str(root/'stdio'), 'TITANOS_API_KEY': 'local-test-key'}
                params = StdioServerParameters(command=sys.executable, args=['-m', 'titanos', 'mcp', '--url', url], env=env)
                async with stdio_client(params) as streams:
                    async with ClientSession(streams[0], streams[1]) as session:
                        await session.initialize()
                        result = tool_data(await session.call_tool('responses', {'payload': {'model': 'fixture', 'input': 'Hello', 'max_output_tokens': 19}}))
                        assert text_of(result) == text and result['max_output_tokens'] == 19
            asyncio.run(mcp_checks())
            checks.append('Actual MCP HTTP and stdio responses tool on a non-native model, preserving tuning')
        finally:
            server.should_exit = True
            thread.join(5)
            sock.close()
            assert not thread.is_alive(), 'Temporary test server did not stop'
    print(json.dumps({'passed': True, 'openai_sdk': version('openai'), 'mcp_sdk': version('mcp'),
                      'checks': checks, 'server_stopped': True,
                      'scope': 'Actual loopback SDK/protocol integration; synthetic inference, no accelerator/model/hostedAPI work'}))


if __name__ == '__main__':
    main()
