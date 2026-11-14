"""Official SDK protocol checks over real HTTP and stdio, with fake inference."""

import asyncio
import base64
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import wave
from urllib.parse import urlsplit

import httpx
import httpx2
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from titanos.config import Config, ModelSpec, write_config
from titanos.mcp_server import create_mcp
from titanos.server import create_app

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==')
VIDEO = b'\x00\x00\x00\x18ftypmp42' + bytes(256)


class FakeImage:
    def close(self):
        self.closed = True

    def save(self, target, **kwargs):
        if hasattr(target, 'write'):
            target.write(PNG)
        else:
            Path(target).write_bytes(PNG)


class FakeModel:
    def __init__(self, spec):
        self.spec = spec

    def details(self):
        return {'backend': 'test-double', 'device': 'cpu'}

    def infer(self, task, data, emit=None, cancel=None):
        if task == 'embeddings':
            return {'vectors': [[1.0, 0.0] for _ in data['input']], 'usage': {'prompt_tokens': 1, 'total_tokens': 1}}
        if task == 'transcription':
            return 'fixture transcription'
        if task == 'speech':
            output = io.BytesIO()
            with wave.open(output, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b'\x00\x00' * 800)
            return output.getvalue()
        if task in {'image', 'image-edit'}:
            return [FakeImage()]
        if task == 'video':
            return [FakeImage()] * data['frames']
        return {'text': 'fixture reply', 'finish_reason': 'stop',
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


def serve(port, root):
    import titanos.backends
    import titanos.media
    import uvicorn
    titanos.backends.LoadedModel = FakeModel
    titanos.backends.image_input = lambda value: FakeImage()
    titanos.media.write_video = lambda frames, path, fps: Path(path).write_bytes(VIDEO)
    cfg = Config(models={task: ModelSpec(path=root, task=task,
                  backend='diffusers' if task in {'image', 'video'} else 'transformers')
                  for task in ['chat', 'vision', 'embeddings', 'speech', 'transcription', 'image', 'video', 'translation']})
    cfg.models['image'].defaults = {'response_format': 'url'}
    registry = Path(root) / 'models.json'
    write_config(registry, cfg)
    app = create_app(cfg, api_key=os.environ['TITANOS_API_KEY'], output_dir=Path(root) / 'outputs', registry_path=registry)
    if not any(getattr(route, 'path', '') == '/mcp' for route in app.routes):
        mcp = create_mcp(api_key=os.environ['TITANOS_API_KEY'], app=app)
        mcp_app = mcp.streamable_http_app(streamable_http_path='/', max_request_body_size=64 * 1024 * 1024)
        original = app.router.lifespan_context
        @asynccontextmanager
        async def lifespan(owner):
            async with original(owner), mcp.session_manager.run():
                yield
        app.router.lifespan_context = lifespan
        app.mount('/mcp', mcp_app)
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='error')


async def exercise(client, transport, expected_origin):
    assert client.server_info.name == 'Titanos'
    assert client.server_capabilities.tools is not None
    assert client.server_capabilities.resources is not None
    assert client.server_capabilities.prompts is not None
    tools = (await client.list_tools()).tools
    names = {tool.name for tool in tools}
    required = {'server_info', 'list_models', 'model_capabilities', 'load_model', 'unload_model',
                'model_details', 'register_model', 'reload_registry', 'unregister_model',
                'list_devices', 'list_workers', 'chat', 'complete', 'responses',
                'embed', 'speak', 'transcribe', 'generate_image', 'edit_image', 'generate_video',
                'infer_tensor', 'extension_infer', 'submit_job', 'list_jobs', 'get_job', 'cancel_job', 'read_artifact'}
    assert required <= names, required - names
    assert all(tool.input_schema.get('type') == 'object' for tool in tools)
    resources = {str(resource.uri) for resource in (await client.list_resources()).resources}
    assert {'titanos://info', 'titanos://models', 'titanos://jobs', 'titanos://schema'} <= resources
    assert len((await client.list_resource_templates()).resource_templates) >= 5
    assert len((await client.list_prompts()).prompts) >= 2
    assert (await client.get_prompt('create_asset', {'description': 'a red square'})).messages
    models = await client.read_resource('titanos://models')
    assert any(item['id'] == 'chat' for item in json.loads(models.contents[0].text)['data'])
    async def call(name, arguments=None):
        result = await client.call_tool(name, arguments or {})
        assert not result.is_error, (name, result.content)
        return result
    await call('server_info')
    await call('list_devices', {'backend': 'system'})
    await call('list_workers')
    await client.read_resource('titanos://extensions')
    extension = await call('extension_infer', {'payload': {'operation': 'translation', 'model': 'translation',
                            'text': 'hello', 'target_language': 'fra_Latn'}})
    assert extension.structured_content['text'] == 'fixture reply'
    await call('model_capabilities', {'model': 'chat'})
    await call('load_model', {'model': 'chat'})
    await call('model_details', {'model': 'chat'})
    result = await call('chat', {'payload': {'model': 'chat', 'messages': [{'role': 'user', 'content': 'hello'}], 'max_completion_tokens': 8}})
    assert result.structured_content['choices'][0]['message']['content'] == 'fixture reply'
    await call('complete', {'payload': {'model': 'chat', 'prompt': 'hello'}})
    result = await call('responses', {'payload': {'model': 'chat', 'input': 'hello'}})
    response = result.structured_content
    assert response['object'] == 'response' and response['status'] == 'completed'
    assert response['output'][0]['content'][0]['text'] == 'fixture reply'
    result = await call('embed', {'payload': {'model': 'embeddings', 'input': 'hello'}})
    assert result.structured_content['data'][0]['embedding'] == [1.0, 0.0]
    result = await call('speak', {'payload': {'model': 'speech', 'input': 'hello'}})
    assert result.content[0].type == 'audio'
    audio = result.content[0].data
    result = await call('transcribe', {'model': 'transcription', 'audio_base64': audio})
    assert result.structured_content['text'] == 'fixture transcription'
    result = await call('generate_image', {'payload': {'model': 'image', 'prompt': 'red square', 'response_format': 'url'}})
    image_url = result.structured_content['data'][0]['url']
    assert urlsplit(image_url).netloc == urlsplit(expected_origin).netloc, image_url
    async with httpx.AsyncClient(headers={'Authorization': 'Bearer protocol-test-only'}, trust_env=False) as download:
        image_response = await download.get(image_url)
        assert image_response.status_code == 200 and image_response.content == PNG
    artifact = image_url.rsplit('/', 1)[1]
    result = await call('read_artifact', {'name': artifact})
    assert base64.b64decode(result.content[0].resource.blob) == PNG
    result = await client.read_resource('titanos://artifacts/' + artifact)
    assert base64.b64decode(result.contents[0].blob) == PNG
    # Every managed-artifact producer must preserve the caller's origin,
    # including image edits that switch to URLs through local model defaults.
    async def verify_asset(result, expected):
        url = result['data'][0]['url'] if 'data' in result else result['url']
        assert (urlsplit(url).scheme, urlsplit(url).netloc) == (urlsplit(expected_origin).scheme, urlsplit(expected_origin).netloc), url
        async with httpx.AsyncClient(headers={'Authorization': 'Bearer protocol-test-only'}, trust_env=False) as download:
            response = await download.get(url)
            assert response.status_code == 200 and response.content == expected
    for tool, arguments, expected in [
        ('generate_image', {'payload': {'model': 'image', 'prompt': 'cube'}}, PNG),
        ('edit_image', {'model': 'image', 'image_base64': base64.b64encode(PNG).decode(), 'prompt': 'make blue'}, PNG),
        ('generate_video', {'payload': {'model': 'video', 'prompt': 'cube', 'size': '64x64', 'frames': 5}}, VIDEO),
    ]:
        await verify_asset((await call(tool, arguments)).structured_content, expected)
    submitted = await call('submit_job', {'payload': {'task': 'speech', 'model': 'speech', 'input': 'hello'}, 'idempotency_key': 'protocol-' + transport})
    identifier = submitted.structured_content['id']
    same = await call('submit_job', {'payload': {'task': 'speech', 'model': 'speech', 'input': 'hello'}, 'idempotency_key': 'protocol-' + transport})
    assert same.structured_content['id'] == identifier
    deadline = time.monotonic() + 10
    while True:
        job = (await call('get_job', {'identifier': identifier})).structured_content
        if job['status'] == 'completed':
            break
        assert job['status'] in {'queued', 'running'} and time.monotonic() < deadline, job
        await asyncio.sleep(0.02)
    await call('list_jobs')
    assert urlsplit(job['result']['url']).netloc == urlsplit(expected_origin).netloc, job
    async with httpx.AsyncClient(headers={'Authorization': 'Bearer protocol-test-only'}, trust_env=False) as download:
        audio_response = await download.get(job['result']['url'])
        assert audio_response.status_code == 200 and audio_response.content.startswith(b'RIFF')
    await client.read_resource('titanos://jobs/' + identifier)
    await call('cancel_job', {'identifier': identifier})
    for task, expected in [('image', PNG), ('video', VIDEO)]:
        payload = {'task': task, 'model': task, 'prompt': 'cube'}
        if task == 'video':
            payload.update(size='64x64', frames=5)
        record = (await call('submit_job', {'payload': payload})).structured_content
        deadline = time.monotonic() + 10
        while record['status'] != 'completed':
            assert record['status'] in {'queued', 'running'} and time.monotonic() < deadline, record
            await asyncio.sleep(0.01)
            record = (await call('get_job', {'identifier': record['id']})).structured_content
        await verify_asset(record['result'], expected)
    await call('unload_model', {'model': 'speech'})
    temporary_model = 'registered-' + transport
    await call('register_model', {'payload': {'name': temporary_model, 'spec': {'path': os.environ['TITANOS_MCP_TEST_ROOT'], 'task': 'chat'}}})
    refreshed = (await call('list_models')).structured_content
    assert temporary_model in {row['id'] for row in refreshed['data']}
    await call('reload_registry')
    await call('unregister_model', {'model': temporary_model})
    assert temporary_model not in {row['id'] for row in (await call('list_models')).structured_content['data']}
    # Errors remain MCP tool results, and disallowed input never becomes file access.
    for name, args in [('read_artifact', {'name': '../models.json'}),
                       ('transcribe', {'model': 'transcription', 'audio_base64': 'not-base64'}),
                       ('chat', {'payload': {'model': 'missing', 'messages': [{'role': 'user', 'content': 'hello'}]}}),
                       ('chat', {'payload': {'model': 'chat', 'messages': [{'role': 'user', 'content': 'hello'}], 'stream': True}}),
                       ('responses', {'payload': {'model': 'chat', 'input': 'hello', 'stream': True}})]:
        assert (await client.call_tool(name, args)).is_error, name
    return {'transport': transport, 'protocol': client.protocol_version, 'tools': len(tools),
            'resource_count': len(resources), 'fake_model_calls': True, 'passed': True}


async def protocols(url, env):
    results = []
    async with httpx2.AsyncClient(headers={'Authorization': 'Bearer ' + env['TITANOS_API_KEY']}, trust_env=False) as http:
        async with Client(streamable_http_client(url + '/mcp/', http_client=http), read_timeout_seconds=30) as client:
            results.append(await exercise(client, 'streamable-http', url))
        async with Client(streamable_http_client(url + '/mcp/', http_client=http), mode='legacy', read_timeout_seconds=30) as client:
            results.append(await exercise(client, 'streamable-http-legacy', url))
    params = StdioServerParameters(command=sys.executable, args=['-m', 'titanos.mcp_server', '--url', url], env=env)
    # Force the classic initialize handshake as well as the modern HTTP discovery.
    async with Client(params, mode='legacy', read_timeout_seconds=30) as client:
        results.append(await exercise(client, 'stdio', url))
    return results


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--serve':
        serve(int(sys.argv[2]), sys.argv[3])
        return
    with tempfile.TemporaryDirectory(prefix='titanos-mcp-') as temp, socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
        reservation.close()
        env = {**os.environ, 'TITANOS_API_KEY': 'protocol-test-only', 'TITANOS_HOME': temp}
        url = f'http://127.0.0.1:{port}'
        log = open(Path(temp) / 'server.log', 'w+', encoding='utf-8')
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--serve', str(port), temp],
                                   env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    if httpx.get(url + '/healthz', trust_env=False, timeout=0.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if process.poll() is not None or time.monotonic() > deadline:
                    log.seek(0)
                    raise AssertionError(log.read())
                time.sleep(0.05)
            assert httpx.post(url + '/mcp/', json={}, trust_env=False).status_code == 401
            previous_root = os.environ.get('TITANOS_MCP_TEST_ROOT')
            os.environ['TITANOS_MCP_TEST_ROOT'] = temp
            try:
                results = asyncio.run(protocols(url, env))
            finally:
                if previous_root is None:
                    os.environ.pop('TITANOS_MCP_TEST_ROOT', None)
                else:
                    os.environ['TITANOS_MCP_TEST_ROOT'] = previous_root
        finally:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                process.terminate()
            process.wait(timeout=10)
            log.close()
            # Windows venv launchers can exit just before inherited handles close.
            # Retry only known files in this freshly created task directory.
            for filename in [Path(temp) / 'server.log', Path(temp) / 'outputs' / 'jobs' / 'owner.lock']:
                for attempt in range(60):
                    try:
                        filename.unlink(missing_ok=True)
                        break
                    except PermissionError:
                        if attempt == 59:
                            raise
                        time.sleep(0.05)
    print(json.dumps({'passed': True, 'checks': results, 'scope': 'Real MCP HTTP/stdio and Titanos HTTP contracts with fake model inference; no model quality or accelerator qualification'}))


if __name__ == '__main__':
    main()
