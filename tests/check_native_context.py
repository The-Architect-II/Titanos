"""Per-slot context sized from options.num_ctx (titanos.native_context, NativePool.lease, the chat and Ollama routes).

Covers: the next power of two at or above num_ctx, capped at the largest per-slot context (native.max_slot_context, then
TITANOS_NATIVE_MAX_SLOT_CONTEXT, then the GGUF trained context_length, never below the registered context_size over slots);
the typed 400 context_limit above it on /api/chat (named on the OpenAI routes); SlotContextOptions (slot_context sets context_size, mismatches refused, unset fields
left out of a saved registration); NativePool.lease starting a cold runtime at the needed size, restarting an idle one,
waiting for a busy one to drain first, refusing under require_loaded, keeping the previous size after a failed start and
forgetting it on unload; --ctx-size = per-slot context x slots in the llama-server argv; and server.generate handing
options.num_ctx from /v1/chat/completions and /api/chat to the native pool. Synthetic GGUF headers, fake processes and
a synthetic proxy seam: no model, llama-server, GPU or network."""
import asyncio
import contextlib
import json
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fastapi.responses import Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from titanos import native  # noqa: E402
from titanos.config import Config, ModelSpec, NativeOptions, read_config, write_config  # noqa: E402
from titanos.engine import EngineError  # noqa: E402
from titanos.native import NativePool  # noqa: E402
from titanos.native_context import max_slot_context, needed_slot_context, next_power_of_two  # noqa: E402
from titanos.server import create_app  # noqa: E402
from titanos.settings import Settings, _active_settings  # noqa: E402


def gguf(path, context_length):
    def text(value):
        return struct.pack('<Q', len(value)) + value.encode()
    pairs = [('general.architecture', 8, text('qwen2')), ('qwen2.context_length', 4, struct.pack('<I', context_length)),
             ('qwen2.block_count', 4, struct.pack('<I', 2))]
    body = b''.join(text(key) + struct.pack('<I', kind) + value for key, kind, value in pairs)
    Path(path).write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, len(pairs)) + body + b'\0' * 16)


def refused(call, fragment):
    try:
        call()
    except EngineError as exc:
        assert exc.status == 400 and exc.code == 'context_limit' and fragment in str(exc), (exc.status, exc.code, str(exc))
        return
    raise AssertionError('no context_limit refusal')


def check_sizing(folder):
    gguf(folder / 'model.gguf', 32768)
    spec = ModelSpec(path=str(folder), task='chat', backend='llama-cpp', native=NativeOptions(model_file='model.gguf',
                                                                                           context_size=8192, slots=4))
    settings = Settings()
    assert [next_power_of_two(v) for v in (1, 2, 3, 2049, 4096, 4097)] == [1, 2, 4, 4096, 4096, 8192]
    assert max_slot_context(spec, settings) == 32768  # the GGUF trained context
    assert needed_slot_context(spec, None, 'm', settings) is None and needed_slot_context(spec, 2048, 'm', settings) is None
    assert needed_slot_context(spec, 2049, 'm', settings) == 4096 and needed_slot_context(spec, 3500, 'm', settings) == 4096
    assert needed_slot_context(spec, 8192, 'm', settings) == 8192 and needed_slot_context(spec, 32768, 'm', settings) == 32768
    refused(lambda: needed_slot_context(spec, 32769, 'm', settings), 'options.num_ctx 32769 exceeds the largest per-slot context 32768 of m')
    # the setting caps below the trained context; a registration caps below the setting; a cap that is not a power of two is used
    assert max_slot_context(spec, Settings(native_max_slot_context=12000)) == 12000
    assert needed_slot_context(spec, 9000, 'm', Settings(native_max_slot_context=12000)) == 12000
    capped = spec.model_copy(update={'native': spec.native.model_copy(update={'max_slot_context': 4096})})
    assert max_slot_context(capped, Settings(native_max_slot_context=12000)) == 4096
    refused(lambda: needed_slot_context(capped, 4097, 'm', settings), 'largest per-slot context 4096')
    # no readable header: the registered per-slot context is the maximum (nothing grows silently)
    bare = ModelSpec(path=str(folder), task='chat', backend='llama-cpp', native=NativeOptions(model_file='absent.gguf',
                                                                                           context_size=2048, slots=2))
    assert max_slot_context(bare, settings) == 1024
    refused(lambda: needed_slot_context(bare, 1025, 'm', settings), 'largest per-slot context 1024')
    for value in (0, 100, 511):
        try:
            Settings(native_max_slot_context=value)
        except ValueError:
            assert value, 'zero is the GGUF default'
        else:
            assert value == 0, value


def check_registration(folder):
    options = NativeOptions(model_file='m.gguf', slot_context=4096, slots=3)
    assert options.context_size == 12288 and options.slot_context == 4096
    assert NativeOptions(model_file='m.gguf', slot_context=1024, slots=2, context_size=2048).context_size == 2048
    for bad in ({'slot_context': 1024, 'slots': 2, 'context_size': 4096}, {'context_size': 8192, 'slots': 4, 'max_slot_context': 1024},
                {'slot_context': 128}):
        try:
            NativeOptions(model_file='m.gguf', **bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} was accepted')
    path = folder / 'models.json'
    cfg = Config(models={'plain': ModelSpec(path=str(folder), task='chat', backend='llama-cpp', native=NativeOptions(model_file='m.gguf')),
                         'sized': ModelSpec(path=str(folder), task='chat', backend='llama-cpp',
                                            native=NativeOptions(model_file='m.gguf', slot_context=2048, slots=2, max_slot_context=8192))})
    write_config(path, cfg)
    saved = json.loads(path.read_text(encoding='utf-8'))['models']
    assert 'slot_context' not in saved['plain']['native'] and 'max_slot_context' not in saved['plain']['native'], saved['plain']
    assert saved['sized']['native']['slot_context'] == 2048 and saved['sized']['native']['context_size'] == 4096
    assert read_config(path).models['sized'].native.max_slot_context == 8192


class Process:
    def poll(self):
        return None


def pool(folder, warm, trained=16384):
    gguf(folder / 'pool.gguf', trained)
    cfg = Config(models={'m': ModelSpec(path=str(folder), task='chat', backend='llama-cpp',
                                        native=NativeOptions(model_file='pool.gguf', context_size=4096, slots=2))})
    value = NativePool(cfg, Settings(native_queue_timeout=1.0))
    worker = value.workers[0]
    worker.events = []
    worker.process, worker.ready, worker.client = (Process(), True, NS(is_closed=False)) if warm else (None, False, None)
    worker.fail_next = False

    async def start():
        worker.events.append(('start', worker.slot_context))
        if worker.fail_next:
            worker.fail_next = False
            raise EngineError('synthetic load failure', 503, 'model_load_failed')
        if worker.process is None:
            worker.process, worker.ready, worker.client = Process(), True, NS(is_closed=False)

    async def stop():
        worker.events.append(('stop', worker.slot_context))
        worker.process, worker.ready, worker.client = None, False, None
    worker.start, worker.stop = start, stop
    return value, worker


async def lease_once(value, **kwargs):
    async with value.lease('m', **kwargs) as selected:
        return selected


def check_pool(folder):
    async def scenario():
        value, worker = pool(folder, warm=False)
        await lease_once(value, num_ctx=8192)  # cold: starts at the needed size
        assert worker.events == [('start', 8192)] and native.slot_context(worker) == 8192, worker.events
        await lease_once(value, num_ctx=4096)  # covered: no restart
        assert worker.events[-1] == ('start', 8192) and len(worker.events) == 2, worker.events

        value, worker = pool(folder, warm=True)  # warm at the registered 2048 per slot, idle
        await lease_once(value, num_ctx=3500)
        assert worker.events == [('stop', None), ('start', 4096)], worker.events

        value, worker = pool(folder, warm=True)  # busy: the resize waits for the in-flight request
        hold, release = asyncio.Event(), asyncio.Event()

        async def busy():
            async with value.lease('m'):
                hold.set()
                await release.wait()
        first = asyncio.create_task(busy())
        await hold.wait()
        second = asyncio.create_task(lease_once(value, num_ctx=16384))
        for _ in range(50):
            await asyncio.sleep(0)
        assert worker.events == [('start', None)], worker.events  # the running request's no-op start; nothing stopped under it
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 2)
        assert worker.events == [('start', None), ('stop', None), ('start', 16384)], worker.events

        value, worker = pool(folder, warm=True)  # require_loaded never restarts
        try:
            await lease_once(value, num_ctx=4096, require_loaded=True)
        except EngineError as exc:
            assert exc.status == 400 and exc.code == 'context_limit' and 'without require_loaded' in str(exc), str(exc)
        else:
            raise AssertionError('a loaded runtime was resized under require_loaded')
        await lease_once(value, num_ctx=2048, require_loaded=True)  # within the running size

        value, worker = pool(folder, warm=True)  # above the maximum: refused before anything is queued or stopped
        try:
            await lease_once(value, num_ctx=16385)
        except EngineError as exc:
            assert exc.code == 'context_limit' and value.waiting == 0 and worker.events == [], (exc.code, worker.events)
        else:
            raise AssertionError('num_ctx above the trained context was accepted')

        value, worker = pool(folder, warm=False)  # a failed start keeps the previous size for the next start
        worker.fail_next = True
        try:
            await lease_once(value, num_ctx=8192)
        except EngineError:
            pass
        assert worker.slot_context is None and native.slot_context(worker) == 2048, worker.slot_context
        await lease_once(value, num_ctx=8192)
        await value.unload('m')  # unload forgets the size
        assert worker.slot_context is None, worker.slot_context
    asyncio.run(scenario())


def check_argv(folder):
    gguf(folder / 'argv.gguf', 32768)
    executable = folder / 'llama-server.exe'
    executable.write_bytes(b'inert executable bytes')
    spec = ModelSpec(path=str(folder), task='chat', backend='llama-cpp', device='cpu',
                     native=NativeOptions(model_file='argv.gguf', context_size=8192, slots=4))
    placement = {'resolved_device': 'cpu', 'resolved_gpu_layers': 0, 'native_devices': ['none'], 'verified_devices': []}
    worker = native.NativeWorker('w', 'm', spec)
    worker.slot_context = 8192
    captured = []

    class Running:
        stdin = NS(close=lambda: None)

        def poll(self):
            return None

        def wait(self, *args, **kwargs):
            return 0

    def start(argv, **kwargs):
        captured.append(argv)
        Path(argv[argv.index('titanos.native_process') + 1]).write_text('4242')
        return Running()
    with patch.object(native, 'executable_path', return_value=executable), \
         patch.object(native, 'resolve_native_placement', return_value=placement), \
         patch.object(native.subprocess, 'Popen', side_effect=start), \
         patch.object(native.httpx, 'Client', return_value=contextlib.nullcontext(NS(get=lambda url: NS(status_code=200)))):
        worker.start_process()
    argv = captured[0]
    assert argv[argv.index('--ctx-size') + 1] == '32768' and argv[argv.index('--parallel') + 1] == '4', argv
    worker.process = None
    assert worker.status()['slot_context'] == 8192
    worker.temp.cleanup()


def check_routes(folder):
    gguf(folder / 'route.gguf', 8192)
    cfg = Config(models={'native': ModelSpec(path=str(folder), task='chat', backend='llama-cpp',
                                             native=NativeOptions(model_file='route.gguf', context_size=2048, slots=2))})
    app = create_app(cfg, output_dir=folder / 'outputs')
    seen = []

    async def proxy(request, route, data, **kwargs):
        seen.append((route, kwargs.get('num_ctx')))
        return Response(json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
                                    'usage': {'prompt_tokens': 3, 'completion_tokens': 1}}), media_type='application/json')
    app.state.native.proxy = proxy
    body = {'model': 'native', 'messages': [{'role': 'user', 'content': 'Hi'}]}
    with TestClient(app, base_url='http://127.0.0.1') as client:
        r = client.post('/v1/chat/completions', json={**body, 'options': {'num_ctx': 4096}})
        assert r.status_code == 200 and 'x-titanos-ignored' not in r.headers and seen[-1] == ('/v1/chat/completions', 4096), (r.text, seen)
        r = client.post('/v1/completions', json={'model': 'native', 'prompt': 'Hi', 'options': {'num_ctx': 8192}})
        assert r.status_code == 200 and seen[-1] == ('/v1/completions', 8192), (r.text, seen)
        r = client.post('/api/chat', json={**body, 'stream': False, 'options': {'num_ctx': 6000}})
        assert r.status_code == 200 and 'x-titanos-dropped' not in r.headers and seen[-1] == ('/v1/chat/completions', 6000), (r.text, seen)
        # above the maximum: named and not honoured on the OpenAI routes (options is foreign there), 400 on /api/chat
        r = client.post('/v1/chat/completions', json={**body, 'options': {'num_ctx': 8193}})
        assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'options.num_ctx' and seen[-1] == ('/v1/chat/completions', None), (r.text, seen)
        count = len(seen)
        r = client.post('/api/chat', json={**body, 'stream': False, 'options': {'num_ctx': 8193}})
        assert r.status_code == 400 and r.json()['error']['code'] == 'context_limit', (r.status_code, r.text)
        assert 'largest per-slot context 8192' in r.json()['error']['message'], r.text
        assert len(seen) == count, 'the refusal happens before the native pool is asked'
        r = client.post('/v1/chat/completions', json=body)
        assert r.status_code == 200 and seen[-1] == ('/v1/chat/completions', None), seen


def main():
    token = _active_settings.set(Settings())
    try:
        with tempfile.TemporaryDirectory(prefix='titanos-context-') as name:
            folder = Path(name)
            check_sizing(folder)
            check_registration(folder)
            check_pool(folder)
            check_argv(folder)
            check_routes(folder)
    finally:
        _active_settings.reset(token)
    print('{"check": "native_context", "result": "pass", "scope": "synthetic GGUF headers, fake runtimes and a proxy seam; '
          'no model, llama-server, GPU or network"}')


if __name__ == '__main__':
    main()
