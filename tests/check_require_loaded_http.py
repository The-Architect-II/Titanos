"""Actual in-process HTTP/ASGI loaded-only seams; fake providers, no listeners."""
from pathlib import Path
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.abc
import inspect
import json
import logging
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'src'))
PROVIDER_ATTEMPTS = []
EVIDENCE = None


class NoProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'diffusers', 'openvino',
                'openvino_genai', 'onnxruntime', 'qwen_tts', 'mlx', 'faster_whisper',
                'sentence_transformers'}:
            PROVIDER_ATTEMPTS.append(fullname)
            raise AssertionError('Real provider import forbidden: ' + fullname)


sys.meta_path.insert(0, NoProviders())
import httpx
from titanos import server, native
from titanos.client import Client
from titanos.config import Config, ModelSpec, NativeOptions
from titanos.engine import EngineError
from titanos.mcp_server import ChatCall
from titanos.settings import Settings


def payload(**options):
    return {'model': 'm', 'messages': [{'role': 'user', 'content': 'inert'}], **options}


class Exchange:
    """Real ASGI calls with explicit receive/send barriers, never a socket."""
    def __init__(self, app, data, *, path='/v1/chat/completions', hold_final=False,
                 fail_send=None, timeline=None):
        self.app, self.data = app, data
        self.messages = []
        self.disconnect = asyncio.Event()
        self.receive_waiting = asyncio.Event()
        self.disconnect_delivered = asyncio.Event()
        self.body_started = asyncio.Event()
        self.final_entered = asyncio.Event()
        self.final_release = asyncio.Event()
        if not hold_final:
            self.final_release.set()
        self.fail_send = fail_send
        self.timeline = timeline if timeline is not None else []
        self.scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
                      'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
                      'path': path, 'raw_path': path.encode(), 'query_string': b'',
                      'root_path': '', 'headers': [(b'host', b'localhost'),
                      (b'content-type', b'application/json')],
                      'client': ('in-process', 1), 'server': ('localhost', 80)}
        self.sent_request = False
        self.task = None

    async def receive(self):
        if not self.sent_request:
            self.sent_request = True
            return {'type': 'http.request', 'body': json.dumps(self.data).encode(),
                    'more_body': False}
        self.receive_waiting.set()
        await self.disconnect.wait()
        self.disconnect_delivered.set()
        return {'type': 'http.disconnect'}

    async def send(self, message):
        terminal = message['type'] == 'http.response.body' and not message.get('more_body', False)
        if terminal:
            self.final_entered.set()
            await self.final_release.wait()
        if (self.fail_send == 'headers' and message['type'] == 'http.response.start') or (
                self.fail_send == 'final' and terminal):
            raise OSError('inert send failure: ' + self.fail_send)
        self.messages.append(dict(message))
        if message['type'] == 'http.response.body': self.body_started.set()
        if terminal:
            self.timeline.append('final_sent')

    def start(self):
        self.task = asyncio.create_task(self.app(self.scope, self.receive, self.send))
        return self

    @property
    def statuses(self):
        return [m['status'] for m in self.messages if m['type'] == 'http.response.start']

    @property
    def body(self):
        return b''.join(m.get('body', b'') for m in self.messages)


class RequireLoadedHTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='titanos-loaded-http-')
        self.directory = Path(self.temp.name)
        self.loop = asyncio.get_running_loop()
        self.counts = {'construct': 0, 'infer': 0, 'close': 0}
        self.inputs, self.timeline, self.exchanges = [], [], []
        self.provider_entered = asyncio.Event()
        self.provider_release = threading.Event(); self.provider_release.set()
        self.provider_cancel = None
        self.chunks = 2
        self.patches = []
        owner = self

        class FakeProvider:
            runtime_fault = None
            def __init__(self, spec):
                owner.counts['construct'] += 1
            def infer(self, task, data, emit=None, cancel=None):
                owner.counts['infer'] += 1
                owner.inputs.append(dict(data)); owner.provider_cancel = cancel
                owner.loop.call_soon_threadsafe(owner.provider_entered.set)
                if not owner.provider_release.wait(4):
                    raise TimeoutError('test provider release absent')
                if cancel is not None and cancel.is_set():
                    raise EngineError('inert cancelled', 499, 'cancelled')
                for _ in range(owner.chunks):
                    if emit: emit('x')
                owner.timeline.append('provider_returned')
                return {'text': 'xx', 'finish_reason': 'stop',
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}}
            def close(self):
                owner.counts['close'] += 1; owner.timeline.append('provider_closed')

        backend = types.ModuleType('titanos.backends'); backend.LoadedModel = FakeProvider
        self.backend_patch = patch.dict(sys.modules, {'titanos.backends': backend})
        self.backend_patch.start()
        self.cfg = Config(models={'m': ModelSpec(path=str(self.directory), task='chat', device='cpu')})
        self.app = server.create_app(self.cfg, output_dir=self.directory / 'outputs', settings=Settings())
        self.engine = self.app.state.engine.target
        self.gate = next(m.kwargs['gate'] for m in self.app.user_middleware if m.cls is server.Guard)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://localhost')

    async def asyncTearDown(self):
        self.provider_release.set()
        for e in self.exchanges:
            e.final_release.set(); e.disconnect.set()
        if self.exchanges:
            await asyncio.wait_for(asyncio.gather(*(e.task for e in self.exchanges), return_exceptions=True), 5)
        await self.client.aclose()
        for item in reversed(self.patches): item.stop()
        # Native workers used below have no real process/client resources.
        self.app.state.native.workers = []
        await self.app.state.native.close()
        self.engine.close()
        self.backend_patch.stop(); self.temp.cleanup()
        self.assertEqual(PROVIDER_ATTEMPTS, [])

    def exchange(self, data, **kw):
        e = Exchange(self.app, data, timeline=self.timeline, **kw).start()
        self.exchanges.append(e)
        return e

    async def finished(self, exchange):
        return await asyncio.wait_for(asyncio.shield(exchange.task), 3)

    def warm(self):
        self.engine.load('m')
        self.counts.update(construct=0, infer=0, close=0)

    def assert_released(self):
        acquired = self.engine.lock.acquire(blocking=False)
        self.assertTrue(acquired, 'Engine ownership leaked')
        if acquired: self.engine.lock.release()
        self.assertEqual(self.gate['active'], 0)

    async def test_strict_bool_schema_and_chat_only_scope(self):
        for value in (None, 'true', 'false', 0, 1, [], {}):
            with self.subTest(value=value):
                r = await self.client.post('/v1/chat/completions', json=payload(require_loaded=value))
                self.assertEqual(r.status_code, 422)
        for route, body in [('/v1/completions', {'model': 'm', 'prompt': 'x'}),
                            ('/v1/responses', {'model': 'm', 'input': 'x'}),
                            ('/v1/embeddings', {'model': 'm', 'input': 'x'})]:
            with self.subTest(route=route):
                r = await self.client.post(route, json={**body, 'require_loaded': True})
                self.assertEqual(r.status_code, 422)
        self.assertEqual(self.counts['construct'], 0)
        self.assertEqual(self.counts['infer'], 0)

    async def test_false_absent_and_model_default_exclusion(self):
        self.cfg.models['m'].defaults['require_loaded'] = True
        for options in ({}, {'require_loaded': False}):
            with self.subTest(options=options):
                self.engine.unload('m'); self.counts.update(construct=0, infer=0)
                r = await self.client.post('/v1/chat/completions', json=payload(**options))
                self.assertEqual(r.status_code, 200)
                self.assertEqual((self.counts['construct'], self.counts['infer']), (1, 1))
                self.assertNotIn('require_loaded', self.inputs[-1])
                self.assert_released()

    async def test_cold_is_409_before_success_headers_stream_and_plain(self):
        for stream in (False, True):
            with self.subTest(stream=stream), patch.object(self.engine, '_load', side_effect=AssertionError('strict called _load')):
                e = self.exchange(payload(require_loaded=True, stream=stream))
                await self.finished(e)
                self.assertEqual(e.statuses, [409])
                self.assertEqual(json.loads(e.body)['error']['code'], 'model_not_loaded')
                self.assertEqual((self.counts['construct'], self.counts['infer']), (0, 0))
                self.assert_released()

    async def test_warm_true_bypasses_load_and_strips_provider_flag(self):
        self.warm()
        for stream in (False, True):
            with self.subTest(stream=stream), patch.object(self.engine, '_load', side_effect=AssertionError('strict called _load')):
                e = self.exchange(payload(require_loaded=True, stream=stream))
                await self.finished(e); self.assertEqual(e.statuses, [200])
                if stream: self.assertIn(b'data: [DONE]', e.body)
                self.assertNotIn('require_loaded', self.inputs[-1])
                self.assert_released()
        self.assertEqual(self.counts['construct'], 0)

    async def test_unload_after_snapshot_before_request_refuses(self):
        self.warm(); self.assertEqual(self.engine.catalog(True)['data'][0]['id'], 'm')
        self.engine.unload('m')
        r = await self.client.post('/v1/chat/completions', json=payload(require_loaded=True))
        self.assertEqual(r.status_code, 409); self.assertEqual(self.counts['infer'], 0)
        self.assertEqual(self.counts['construct'], 0); self.assert_released()

    async def delayed_send(self, stream):
        self.warm(); e = self.exchange(payload(require_loaded=True, stream=stream), hold_final=True)
        await asyncio.wait_for(e.final_entered.wait(), 2)
        self.assertIn('provider_returned', self.timeline)
        acquired = self.engine.lock.acquire(blocking=False)
        if acquired: self.engine.lock.release()
        self.assertFalse(acquired, 'released before final ASGI send')
        self.assertEqual(self.gate['active'], 1)
        done = self.loop.create_future()
        def unload():
            try: self.engine.unload('m')
            except BaseException as exc: self.loop.call_soon_threadsafe(done.set_exception, exc)
            else: self.loop.call_soon_threadsafe(done.set_result, None)
        thread = threading.Thread(target=unload)
        thread.start()
        try:
            self.assertFalse(done.done()); self.assertEqual(self.counts['close'], 0)
        finally:
            e.final_release.set()
        await self.finished(e); await asyncio.wait_for(done, 2); thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertLess(self.timeline.index('final_sent'), self.timeline.index('provider_closed'))
        self.assert_released()

    async def test_nonstream_retains_owner_through_final_send(self):
        await self.delayed_send(False)

    async def test_stream_retains_owner_through_final_send(self):
        await self.delayed_send(True)

    async def test_never_entered_stream_still_finalizes(self):
        self.warm(); entered = asyncio.Event()
        class NeverEntered(server.StreamingResponse):
            async def __call__(self, scope, receive, send):
                entered.set(); raise OSError('inert iterator never entered')
        with patch.object(server, 'StreamingResponse', NeverEntered):
            e = self.exchange(payload(require_loaded=True, stream=True))
            with self.assertRaisesRegex(OSError, 'never entered'): await self.finished(e)
        self.assertTrue(entered.is_set()); self.assert_released()

    async def test_send_errors_release_and_keep_primary(self):
        self.warm()
        for stream in (False, True):
            for point in ('headers', 'final'):
                with self.subTest(stream=stream, point=point):
                    e = self.exchange(payload(require_loaded=True, stream=stream), fail_send=point)
                    with self.assertRaises(BaseException) as caught: await self.finished(e)
                    self.assertIn('inert send failure', str(caught.exception))
                    self.assert_released()

    async def test_disconnect_while_waiting_for_engine_prevents_inference(self):
        self.warm(); self.engine.lock.acquire()
        entered = asyncio.Event(); cancelled = asyncio.Event(); actual = self.engine.infer
        def observed(*args, **kw):
            cancel = args[4]
            original_set = cancel.set
            def set_cancel():
                original_set(); self.loop.call_soon_threadsafe(cancelled.set)
            cancel.set = set_cancel
            self.loop.call_soon_threadsafe(entered.set)
            return actual(*args, **kw)
        try:
            with patch.object(self.engine, 'infer', observed):
                e = self.exchange(payload(require_loaded=True, stream=True))
                await asyncio.wait_for(entered.wait(), 2)
                e.disconnect.set(); await asyncio.wait_for(cancelled.wait(), 2)
                self.assertEqual(self.counts['infer'], 0)
        finally:
            self.engine.lock.release()
        await self.finished(e)
        self.assertNotIn(200, e.statuses); self.assertEqual(self.counts['infer'], 0)
        self.assert_released()

    async def test_repeated_asgi_cancellation_settles_actual_provider(self):
        self.warm(); self.provider_release.clear()
        e = self.exchange(payload(require_loaded=True))
        await asyncio.wait_for(self.provider_entered.wait(), 2)
        e.task.cancel('first inert cancellation')
        await asyncio.sleep(0)
        self.assertFalse(e.task.done()); self.assertEqual(self.gate['active'], 1)
        e.task.cancel('second inert cancellation')
        self.provider_release.set()
        with self.assertRaises(asyncio.CancelledError): await self.finished(e)
        self.assert_released()

    async def test_cancelled_execution_wrapper_is_not_thread_completion(self):
        self.warm(); self.provider_release.clear()
        e = self.exchange(payload(require_loaded=True))
        await asyncio.wait_for(self.provider_entered.wait(), 2)
        cleanup = e.scope['titanos.cleanup'][0]
        wrapper = inspect.getclosurevars(cleanup).nonlocals['worker']
        self.assertIsInstance(wrapper, asyncio.Task)
        wrapper.cancel('inert execution-wrapper cancellation')
        self.assertFalse(self.provider_release.is_set())
        entered, exited = asyncio.Event(), asyncio.Event()
        async def observed_cleanup():
            entered.set()
            try: await cleanup()
            finally: exited.set()
        e.scope['titanos.cleanup'][0] = observed_cleanup
        try:
            e.task.cancel('inert ASGI cancellation after wrapper cancellation')
            await asyncio.wait_for(entered.wait(), 2)
            self.assertFalse(exited.is_set(), 'cancelled wrapper was treated as actual thread completion')
            self.assertEqual(self.gate['active'], 1)
        finally:
            self.provider_release.set()
            try: await self.finished(e)
            except asyncio.CancelledError: pass
        self.assert_released()

    async def test_cancelled_queued_execution_wrapper_never_infers_and_settles(self):
        self.warm()
        self.loop.set_default_executor(ThreadPoolExecutor(max_workers=1, thread_name_prefix='loaded-only-queued-cancel'))
        blocker_started = asyncio.Event(); unblock = threading.Event()
        def block_executor():
            self.loop.call_soon_threadsafe(blocker_started.set)
            if not unblock.wait(4): raise TimeoutError('test executor release absent')
        blocker = self.loop.run_in_executor(None, block_executor)
        await asyncio.wait_for(blocker_started.wait(), 2)
        e = self.exchange(payload(require_loaded=True))
        entered, exited = asyncio.Event(), asyncio.Event()
        try:
            await asyncio.wait_for(e.receive_waiting.wait(), 2)
            cleanup = e.scope['titanos.cleanup'][0]
            wrapper = inspect.getclosurevars(cleanup).nonlocals['worker']
            async def observed_cleanup():
                entered.set()
                try: await cleanup()
                finally: exited.set()
            e.scope['titanos.cleanup'][0] = observed_cleanup
            wrapper.cancel('inert queued wrapper cancellation')
            await asyncio.wait_for(entered.wait(), 2)
            self.assertFalse(exited.is_set(), 'queued actual operation was abandoned')
            self.assertEqual(self.counts['infer'], 0)
            self.assertEqual(self.gate['active'], 1)
        finally:
            unblock.set(); await asyncio.wait_for(blocker, 2)
            try: await self.finished(e)
            except asyncio.CancelledError: pass
        self.assertNotIn(200, e.statuses); self.assertEqual(self.counts['infer'], 0)
        self.assertEqual(self.counts['construct'], 0); self.assert_released()

    async def test_execution_wrapper_cancelled_before_dispatch_refuses_without_hang(self):
        self.warm(); original = asyncio.create_task; intercepted = []
        def cancel_owned_before_start(coro, *args, **kwargs):
            task = original(coro, *args, **kwargs)
            if coro.__qualname__.endswith('generate.<locals>.execute'):
                intercepted.append(task); task.cancel('inert predispatch cancellation')
            return task
        with patch.object(asyncio, 'create_task', cancel_owned_before_start):
            e = self.exchange(payload(require_loaded=True))
            try: await self.finished(e)
            except asyncio.CancelledError: pass
        self.assertEqual(len(intercepted), 1)
        self.assertEqual(e.statuses, [499])
        self.assertEqual((self.counts['construct'], self.counts['infer']), (0, 0))
        self.assert_released()

    async def test_execution_wrapper_cancelled_after_stream_headers_wakes_consumer(self):
        self.warm(); self.provider_release.clear()
        e = self.exchange(payload(require_loaded=True, stream=True))
        await asyncio.wait_for(e.body_started.wait(), 2)
        self.assertEqual(e.statuses, [200])
        cleanup = e.scope['titanos.cleanup'][0]
        wrapper = inspect.getclosurevars(cleanup).nonlocals['worker']
        try:
            wrapper.cancel('inert execution cancellation after stream headers')
            self.provider_release.set()
            try: await self.finished(e)
            except asyncio.CancelledError: pass
        finally:
            # This is only bounded failed-test cleanup, not the success trigger.
            self.provider_release.set(); e.disconnect.set()
            await asyncio.wait_for(asyncio.gather(e.task, return_exceptions=True), 3)
        self.assert_released()

    async def test_single_executor_thread_drains_many_chunks_and_finalizer(self):
        self.warm(); self.chunks = 70
        self.loop.set_default_executor(ThreadPoolExecutor(max_workers=1, thread_name_prefix='loaded-only-test'))
        for stream in (False, True):
            with self.subTest(stream=stream):
                e = self.exchange(payload(require_loaded=True, stream=stream))
                await self.finished(e); self.assertEqual(e.statuses, [200]); self.assert_released()
                if stream: self.assertIn(b'data: [DONE]', e.body)

    async def test_single_executor_active_owner_and_queued_second_request_progress(self):
        self.warm(); self.chunks = 70
        self.loop.set_default_executor(ThreadPoolExecutor(max_workers=1, thread_name_prefix='loaded-only-queued-test'))
        first = self.exchange(payload(require_loaded=True, stream=True), hold_final=True)
        await asyncio.wait_for(first.final_entered.wait(), 2)
        second = self.exchange(payload(require_loaded=True, stream=True))
        await asyncio.wait_for(second.receive_waiting.wait(), 2)
        self.assertEqual(self.gate['active'], 2)
        self.assertEqual(self.counts['infer'], 1)
        second.disconnect.set()
        await asyncio.wait_for(second.disconnect_delivered.wait(), 2)
        # Releasing the first ASGI send must need no free execution thread.
        first.final_release.set()
        await self.finished(first); await self.finished(second)
        self.assertEqual(first.statuses, [200]); self.assertNotIn(200, second.statuses)
        self.assertEqual(self.counts['infer'], 1); self.assertEqual(self.counts['construct'], 0)
        self.assert_released()

    async def native_fixture(self):
        await self.client.aclose(); self.engine.close()
        self.cfg = Config(models={'m': ModelSpec(path=str(self.directory), task='chat', device='cpu',
                backend='llama-cpp', native=NativeOptions(model_file='inert-not-read.gguf'))})
        self.app = server.create_app(self.cfg, output_dir=self.directory/'native-outputs', settings=Settings())
        self.engine = self.app.state.engine.target
        self.gate = next(m.kwargs['gate'] for m in self.app.user_middleware if m.cls is server.Guard)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://localhost')
        owner = self
        self.native_inputs, self.native_responses = [], []
        self.native_close_error = False
        self.native_starts = self.native_stops = 0

        class FakeResponse:
            is_success = True
            status_code = 200
            headers = {'content-type': 'application/json'}
            def __init__(self): self.close_calls = 0
            async def aread(self): return b'{"choices":[{"message":{"content":"inert"}}]}'
            async def aiter_bytes(self):
                yield b'data: {"choices":[]}\n\n'
                yield b'data: [DONE]\n\n'
            async def aclose(self):
                self.close_calls += 1
                if owner.native_close_error: raise OSError('inert upstream close failure')

        class FakeClient:
            is_closed = False
            def build_request(self, method, route, json):
                owner.native_inputs.append(dict(json)); return object()
            async def send(self, request, stream):
                response = FakeResponse(); owner.native_responses.append(response); return response

        w = self.app.state.native.workers[0]
        w.process = types.SimpleNamespace(poll=lambda: None)
        w.client = FakeClient(); w.ready = True
        async def forbidden_start():
            owner.native_starts += 1
            raise AssertionError('strict native called start')
        async def fake_stop(): owner.native_stops += 1
        for item in [patch.object(w, 'start', forbidden_start), patch.object(w, 'stop', fake_stop)]:
            item.start(); self.patches.append(item)
        return w

    def assert_native_released(self, w):
        self.assertEqual((w.active, w.pending, self.app.state.native.waiting), (0, 0, 0))
        self.assertEqual(self.native_starts, 0)
        self.assert_released()

    async def test_native_cold_refusal_precedes_success_headers(self):
        w = await self.native_fixture(); w.process = None; w.ready = False
        for stream in (False, True):
            with self.subTest(stream=stream):
                e = self.exchange(payload(require_loaded=True, stream=stream))
                await self.finished(e)
                self.assertEqual(e.statuses, [409])
                self.assertEqual(json.loads(e.body)['error']['code'], 'model_not_loaded')
                self.assertEqual(self.native_inputs, [])
                self.assertEqual(self.native_responses, [])
                self.assert_native_released(w)

    async def test_native_plain_and_stream_lease_survives_final_send(self):
        w = await self.native_fixture()
        for stream in (False, True):
            with self.subTest(stream=stream):
                e = self.exchange(payload(require_loaded=True, stream=stream), hold_final=True)
                await asyncio.wait_for(e.final_entered.wait(), 2)
                self.assertEqual(w.active, 1)
                self.assertEqual(self.native_responses[-1].close_calls, 0)
                with self.assertRaises(EngineError) as caught: await self.app.state.native.unload('m')
                self.assertEqual(caught.exception.code, 'model_busy'); self.assertEqual(self.native_stops, 0)
                e.final_release.set(); await self.finished(e)
                self.assertEqual(e.statuses, [200]); self.assertNotIn('require_loaded', self.native_inputs[-1])
                self.assertEqual(self.native_responses[-1].close_calls, 1)
                self.assertNotIn('titanos.cleanup', e.scope); self.assert_native_released(w)

    async def test_native_never_entered_stream_guard_finalizes(self):
        w = await self.native_fixture()
        class NeverEntered(native.StreamingResponse):
            async def __call__(self, scope, receive, send):
                raise OSError('inert native iterator never entered')
        with patch.object(native, 'StreamingResponse', NeverEntered):
            e = self.exchange(payload(require_loaded=True, stream=True))
            with self.assertRaisesRegex(OSError, 'never entered'): await self.finished(e)
        self.assertEqual(self.native_responses[-1].close_calls, 1); self.assert_native_released(w)

    async def test_native_close_error_unwinds_lease_preserves_asgi_primary(self):
        w = await self.native_fixture(); self.native_close_error = True
        for point in (None, 'headers'):
            with self.subTest(point=point):
                e = self.exchange(payload(require_loaded=True, stream=True), fail_send=point)
                with self.assertRaises(BaseException) as caught: await self.finished(e)
                if point:
                    self.assertIn('inert send failure', str(caught.exception))
                    self.assertIn('upstream close failure', str(getattr(caught.exception, '__notes__', [])))
                else:
                    self.assertIn('upstream close failure', str(caught.exception))
                self.assertEqual(self.native_responses[-1].close_calls, 1); self.assert_native_released(w)

    async def test_capability_mcp_inherited_route_and_python_client_forwarding(self):
        r = await self.client.get('/api/info')
        self.assertEqual(r.json()['inference_contracts']['require_loaded'],
                         {'revision': 1, 'routes': ['/v1/chat/completions']})
        self.assertIs(ChatCall.model_validate(payload(require_loaded=True)).require_loaded, True)
        self.warm()
        result = await self.app.state.mcp.call_tool('chat', {'payload': payload(require_loaded=True)})
        self.assertFalse(result.is_error); self.assertEqual(self.counts['infer'], 1)
        self.assertNotIn('require_loaded', self.inputs[-1]); self.assert_released()
        requests = []
        def handler(request):
            requests.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, json={'fake': True})
        with Client('http://in-process.invalid', api_key='', transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(client.chat('m', [{'role': 'user', 'content': 'x'}], require_loaded=True), {'fake': True})
        self.assertEqual(requests[0][0], 'http://in-process.invalid/v1/chat/completions')
        self.assertIs(requests[0][1]['require_loaded'], True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--case', action='append', help='Run only named test methods for an affected regression')
    args = parser.parse_args()
    if args.evidence:
        EVIDENCE = args.evidence.resolve(); EVIDENCE.mkdir(exist_ok=False, parents=True)
    logging.disable(logging.CRITICAL)
    suite = (unittest.TestSuite(RequireLoadedHTTP(n) for n in args.case) if args.case else
             unittest.defaultTestLoader.loadTestsFromTestCase(RequireLoadedHTTP))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors),
               'provider_import_attempts': PROVIDER_ATTEMPTS,
               'source_hashes': {n: hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in
                   ['src/titanos/server.py', 'src/titanos/engine.py', 'src/titanos/replicas.py',
                    'src/titanos/native.py', 'src/titanos/request_defaults.py',
                    'tests/check_require_loaded_http.py']},
               'scope': 'Actual create_app, Guard, HTTP/ASGI, direct Engine and NativePool with fake provider/client/process identity; MCP in-process tool call and Python MockTransport; no listener, process start, real provider/model, device or network.',
               'limitations': 'Native/replica authoritative admission unit coverage is owned separately; source tests are not native runtime qualification.'}
    if EVIDENCE: (EVIDENCE/'RESULT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
