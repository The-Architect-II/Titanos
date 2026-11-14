"""Stateless Responses behavior through fake local engines; no weights or network."""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import threading
import types
from unittest.mock import patch

from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient

from titanos.config import Config, ModelSpec, NativeOptions
from titanos.server import create_app
from titanos.settings import Settings


IMAGE = ('data:image/png;base64,'
         'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aE1cAAAAASUVORK5CYII=')
USAGE = {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}


def text_of(response):
    return ''.join(part['text'] for item in response['output'] if item['type'] == 'message'
                   for part in item['content'] if part['type'] == 'output_text')


def sse_rows(body):
    rows = []
    for frame in body.replace('\r\n', '\n').split('\n\n'):
        if not frame.strip():
            continue
        lines = frame.splitlines()
        values = [line[5:].lstrip() for line in lines if line.startswith('data:')]
        assert values and values != ['[DONE]'], frame
        value = json.loads('\n'.join(values))
        names = [line[6:].strip() for line in lines if line.startswith('event:')]
        assert names == [value['type']], frame
        rows.append(value)
    assert rows, body
    return rows


def check_response(body, text='Hello caf\u00e9', status='completed', reason=None):
    assert body['object'] == 'response' and body['id'] and body['model']
    assert isinstance(body['created_at'], (float, int)) and body['created_at'] > 0
    assert body['status'] == status and text_of(body) == text, body
    assert len(body['output']) == 1 and body['output'][0]['role'] == 'assistant'
    item = body['output'][0]
    assert item['id'] and item['id'] != body['id'] and item['status'] == status
    assert item['content'][0]['type'] == 'output_text'
    assert item['content'][0]['annotations'] == []
    assert body['error'] is None
    assert body['incomplete_details'] == ({'reason': reason} if reason else None)
    assert body['store'] is False
    usage = body['usage']
    assert (usage['input_tokens'], usage['output_tokens'], usage['total_tokens']) == (7, 3, 10), usage


def check_stream(rows, text='Hello caf\u00e9', status='completed', reason=None):
    kinds = [row['type'] for row in rows]
    assert kinds[:4] == ['response.created', 'response.in_progress',
                         'response.output_item.added', 'response.content_part.added'], kinds
    assert kinds[-4:] == ['response.output_text.done', 'response.content_part.done',
                          'response.output_item.done', 'response.' + status], kinds
    assert kinds.count('response.' + status) == 1
    assert sum(kind in {'response.completed', 'response.incomplete', 'response.failed'} for kind in kinds) == 1
    sequence = [row['sequence_number'] for row in rows]
    assert all(type(number) is int for number in sequence)
    assert sequence == list(range(sequence[0], sequence[0] + len(sequence))), sequence
    response = rows[-1]['response']
    check_response(response, text, status, reason)
    for row in rows:
        if 'response' in row:
            assert row['response']['id'] == response['id']
            assert row['response']['model'] == response['model']
            assert row['response']['created_at'] == response['created_at']
        if 'output_index' in row:
            assert row['output_index'] == 0
        if 'content_index' in row:
            assert row['content_index'] == 0
        if 'item_id' in row:
            assert row['item_id'] == response['output'][0]['id']
    assert rows[0]['response']['status'] == 'in_progress'
    assert rows[0]['response']['usage'] is None
    assert rows[2]['item']['status'] == 'in_progress' and rows[2]['item']['content'] == []
    assert rows[2]['item']['id'] == response['output'][0]['id']
    assert rows[3]['part']['text'] == ''
    assert ''.join(row['delta'] for row in rows if row['type'] == 'response.output_text.delta') == text
    assert rows[-4]['text'] == rows[-3]['part']['text'] == text
    assert rows[-2]['item'] == response['output'][0]


def contracts():
    seen = []
    outcome = {'text': 'Hello caf\u00e9', 'finish_reason': 'stop'}

    def infer(name, task, data, emit=None, cancel=None):
        seen.append((name, task, data))
        if outcome.get('error_before'):
            raise RuntimeError('Synthetic generation failure')
        if emit:
            for piece in [outcome['text'][:5], outcome['text'][5:]]:
                if piece:
                    emit(piece)
            if outcome.get('error_after'):
                raise RuntimeError('Synthetic generation failure after output')
        return {**outcome, 'usage': dict(USAGE)}

    settings = Settings(max_tokens=256, default_max_tokens=64, max_messages=8,
                        max_text_chars=128, max_input_image_bytes=256)
    with tempfile.TemporaryDirectory(prefix='titanos-responses-') as temp:
        cfg = Config(models={
            'chat': ModelSpec(path=temp, task='chat'),
            'vision': ModelSpec(path=temp, task='vision'),
            'tuned': ModelSpec(path=temp, task='chat', defaults={
                'max_new_tokens': 37, 'temperature': .25, 'top_p': .7}),
            'completion_default': ModelSpec(path=temp, task='chat', defaults={'max_completion_tokens': 43}),
            'responses_default': ModelSpec(path=temp, task='chat', defaults={'max_output_tokens': 41}),
            'embed': ModelSpec(path=temp, task='embeddings'),
        })
        app = create_app(cfg, settings=settings, output_dir=Path(temp) / 'outputs')
        app.state.engine.target.infer = infer
        with TestClient(app, base_url='http://localhost') as client:
            def post(**fields):
                return client.post('/v1/responses', json={'model': 'chat', 'input': 'Hi', **fields})

            def check_capability(model, inputs, native=False):
                count = len(seen)
                # Capability reporting must not inspect/load a real checkpoint.
                # capabilities.describe no longer probes installed modules (passive inspection moved to model_inspection); the guard
                # that matters is that reporting never loads a checkpoint.
                with patch('titanos.engine.Engine._load', side_effect=AssertionError('capabilities must not load a model')):
                    result = client.post('/api/models/capabilities', json={'model': model})
                assert result.status_code == 200 and len(seen) == count, result.text
                capability = result.json()['responses']
                expected = {'implemented': bool(inputs), 'inputs': inputs, 'store': False,
                            'tools': native, 'structured_output': native, 'qualification': None}
                assert all(capability[key] == value for key, value in expected.items()), capability

            assert client.get('/api/info').json()['interfaces']['responses'] == '/v1/responses'
            for model, inputs in [('chat', ['text']), ('vision', ['text', 'inline_image']), ('embed', [])]:
                check_capability(model, inputs)

            normal = post(instructions='Be brief', max_output_tokens=23, temperature=0,
                          top_p=.6, worker='chosen', metadata={'case': 'normal'})
            assert normal.status_code == 200, normal.text
            first = normal.json()
            check_response(first)
            assert first['metadata'] == {'case': 'normal'}
            data = seen[-1][2]
            assert seen[-1][:2] == ('chat', 'chat')
            assert data['messages'] == [{'role': 'system', 'content': 'Be brief'},
                                        {'role': 'user', 'content': 'Hi'}]
            assert (data['max_tokens'], data['temperature'], data['top_p'], data['worker']) == (23, 0, .6, 'chosen')

            history = [{'role': 'user', 'content': 'First'}, first['output'][0],
                       {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Then?'}]}]
            replay = post(input=history)
            assert replay.status_code == 200, replay.text
            messages = seen[-1][2]['messages']
            assert [message['role'] for message in messages] == ['user', 'assistant', 'user']
            def message_text(message):
                content = message['content']
                return content if isinstance(content, str) else ''.join(part['text'] for part in content if part['type'] == 'text')
            assert [message_text(message) for message in messages] == ['First', 'Hello caf\u00e9', 'Then?']
            replay = post(input=[history[0], {**first['output'][0], 'phase': None}, history[-1]])
            assert replay.status_code == 200, replay.text
            assert [message_text(message) for message in seen[-1][2]['messages']] == ['First', 'Hello caf\u00e9', 'Then?']
            nullable_logprobs = {**first['output'][0], 'content': [{**first['output'][0]['content'][0], 'logprobs': None}]}
            replay = post(input=[history[0], nullable_logprobs, history[-1]])
            assert replay.status_code == 200, replay.text

            parts = [{'type': 'input_text', 'text': 'Before'}, {'type': 'input_image', 'image_url': IMAGE},
                     {'type': 'input_text', 'text': 'After'}]
            vision = post(model='vision', input=[{'role': 'user', 'content': parts}])
            assert vision.status_code == 200, vision.text
            assert seen[-1][2]['messages'][0]['content'] == [
                {'type': 'text', 'text': 'Before'}, {'type': 'image_url', 'image_url': {'url': IMAGE, 'detail': 'auto'}},
                {'type': 'text', 'text': 'After'}]
            count = len(seen)
            text_only = post(input=[{'role': 'user', 'content': parts}])
            assert text_only.status_code == 400 and len(seen) == count, text_only.text
            assert text_only.json()['error']['type'] == 'unsupported_capability'

            for overrides, expected in [({}, (37, .25, .7)),
                                        ({'max_output_tokens': 19, 'temperature': 0, 'top_p': .9}, (19, 0, .9))]:
                result = post(model='tuned', **overrides)
                assert result.status_code == 200, result.text
                data = seen[-1][2]
                assert (data['max_tokens'], data['temperature'], data['top_p']) == expected, data
                body = result.json()
                assert (body['max_output_tokens'], body['temperature'], body['top_p']) == expected, body
            for model, default in [('completion_default', 43), ('responses_default', 41)]:
                for fields, expected in [({}, default), ({'max_output_tokens': 17}, 17)]:
                    result = post(model=model, **fields)
                    assert result.status_code == 200, result.text
                    assert seen[-1][2]['max_tokens'] == expected, seen[-1]
                    assert result.json()['max_output_tokens'] == expected, result.text

            for noops in [{}, {'tools': [], 'tool_choice': 'auto', 'parallel_tool_calls': False, 'text': {}},
                          {'tools': None, 'tool_choice': 'none', 'text': {'format': {'type': 'text'}}}]:
                result = post(**noops)
                assert result.status_code == 200, result.text
            max_metadata = {('k' * 62) + str(i): 'v' * 512 for i in range(16)}
            result = post(metadata=max_metadata)
            assert result.status_code == 200 and result.json()['metadata'] == max_metadata, result.text

            invalid = [
                {'input': []}, {'input': [{}]}, {'input': [{'role': 'user'}]},
                {'input': [{'role': 'user', 'content': {'text': 'bad'}}]},
                {'input': [{'role': 'user', 'content': [None]}]},
                {'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 42}]}]},
                {'input': [{'role': 'developer', 'content': 'Unsupported hierarchy'}]},
                {'input': [{**first['output'][0], 'phase': 'analysis'}]},
                {'input': [{**first['output'][0], 'phase': {}}]},
                {'input': [{'type': 'function_call_output', 'call_id': 'c', 'output': 'x'}]},
                {'input': [{'type': 'reasoning', 'summary': []}]},
                {'input': [{'type': 'item_reference', 'id': 'msg_previous'}]},
                {'tools': [{'type': 'function', 'name': 'f', 'parameters': {}}]},
                {'tool_choice': 'required'}, {'tool_choice': {'type': 'function', 'name': 'f'}},
                {'parallel_tool_calls': True}, {'text': {'format': {'type': 'json_schema', 'schema': {}}}},
                {'text': {'verbosity': 7}}, {'store': True}, {'previous_response_id': 'resp_old'},
                {'conversation': 'conv_old'}, {'background': True}, {'max_output_tokens': 257},
                {'input': 'x' * 129}, {'instructions': 'x' * 129},
                {'input': [{'role': 'user', 'content': 'x'}] * 9},
                {'instructions': 'system', 'input': [{'role': 'user', 'content': 'x'}] * 8},
                {'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'x' * 129}]}]},
                {'metadata': {str(i): 'v' for i in range(17)}}, {'metadata': {'k' * 65: 'v'}},
                {'metadata': {'k': 'v' * 513}}, {'metadata': {'k': 42}},
            ]
            for untyped in [[], {}]:
                invalid.extend([
                    {'input': [{'role': untyped, 'content': 'bad'}]},
                    {'input': [{'role': 'assistant', 'status': untyped, 'content': 'bad'}]},
                    {'input': [{'role': 'user', 'content': [{'type': untyped, 'text': 'bad'}]}]},
                ])
            for field, value in [('annotations', None), ('annotations', {}), ('annotations', [{'type': 'url_citation'}]),
                                 ('logprobs', {}), ('logprobs', ''), ('logprobs', [{'token': 'x'}])]:
                invalid.append({'input': [{**first['output'][0],
                                          'content': [{**first['output'][0]['content'][0], field: value}]}]})
            for part in [{'type': 'input_file', 'file_id': 'file_x'},
                         {'type': 'input_image', 'file_id': 'file_x'},
                         {'type': 'input_image', 'image_url': IMAGE, 'detail': 'high'},
                         {'type': 'input_image', 'image_url': {'url': IMAGE}},
                         *[{'type': 'input_image', 'image_url': 'data:image/png;base64,' + encoded}
                           for encoded in ['', 'not-base64', base64.b64encode(b'x' * 257).decode()]]]:
                invalid.append({'model': 'vision', 'input': [{'role': 'user', 'content': [part]}]})
            for payload in invalid:
                count = len(seen)
                result = post(**payload)
                assert result.status_code in {400, 422}, (payload, result.status_code, result.text)
                assert 'error' in result.json() and len(seen) == count, (payload, result.text)
            count = len(seen)
            with patch('titanos.responses.base64.b64decode') as decode:
                result = post(model='vision', input=[{'role': 'user', 'content': [
                    {'type': 'input_image', 'image_url': 'data:image/png;base64,' + 'A' * 600}]}])
            assert result.status_code == 400 and len(seen) == count, result.text
            decode.assert_not_called()
            for model, status in [('unknown', 404), ('embed', 400)]:
                count = len(seen)
                result = post(model=model)
                assert result.status_code == status and len(seen) == count, result.text

            for finish, status, reason in [('stop', 'completed', None),
                                           ('length', 'incomplete', 'max_output_tokens'),
                                           ('content_filter', 'incomplete', 'content_filter')]:
                outcome['finish_reason'] = finish
                result = post()
                assert result.status_code == 200, result.text
                check_response(result.json(), status=status, reason=reason)
                streamed = post(stream=True)
                assert streamed.status_code == 200 and streamed.headers['content-type'].startswith('text/event-stream')
                check_stream(sse_rows(streamed.text), status=status, reason=reason)
                assert seen[-1][2]['stream_options']['include_usage'] is True
            outcome.update(text='', finish_reason='stop')
            check_stream(sse_rows(post(stream=True).text), text='')
            outcome['text'] = 'Hello caf\u00e9'
            for error in ['error_before', 'error_after']:
                outcome[error] = True
                rows = sse_rows(post(stream=True).text)
                kinds = [row['type'] for row in rows]
                assert kinds[-1] == 'response.failed' and kinds.count('response.failed') == 1, rows
                assert 'response.completed' not in kinds and 'response.incomplete' not in kinds
                failed = rows[-1]['response']
                assert failed['status'] == 'failed' and failed['error']['message'], failed
                if error == 'error_after':
                    assert text_of(failed) == 'Hello caf\u00e9'
                del outcome[error]

            native_calls = []
            native_body = {'id': 'resp_native', 'object': 'response', 'output': [], 'custom_native_field': 'preserved'}
            native_events = native_llama_events();native_sse = ''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n' for e in native_events)
            cfg.models['native'] = ModelSpec(path=temp, task='chat', backend='llama-cpp', device='cpu',
                                             native=NativeOptions(model_file='unused.gguf', gpu_layers=0))
            app.state.native.models['native'] = cfg.models['native']
            check_capability('native', ['text'], native=True)
            async def native_proxy(request, route, data):
                native_calls.append((route, data))
                if data['input'] == 'fail':
                    return Response('{"error":{"message":"native unavailable"}}', status_code=503,
                                    media_type='application/json', headers={'X-Native': 'yes'})
                return Response(native_sse if data.get('stream') else json.dumps(native_body),
                                media_type='text/event-stream' if data.get('stream') else 'application/json',
                                headers={'X-Native': 'yes'})
            app.state.native.proxy = native_proxy
            for stream in [False, True]:
                request = {'model': 'native', 'input': 'native input', 'stream': stream,
                           'tools': [{'type': 'function', 'name': 'f', 'parameters': {}}], 'tool_choice': 'required',
                           'parallel_tool_calls': True, 'max_output_tokens': 29, 'temperature': .4, 'top_p': .8}
                count = len(seen)
                result = client.post('/v1/responses', json=request)
                assert result.status_code == 200 and result.headers['X-Native'] == 'yes', result.text
                assert (check_native_stream(result.text, native_events) if stream else result.json() == native_body)
                route, forwarded = native_calls[-1]
                assert route == '/v1/responses' and all(forwarded[key] == value for key, value in request.items())
                assert len(seen) == count
                result = post(model='native', input='fail', stream=stream)
                assert result.status_code == 503 and result.headers['X-Native'] == 'yes', result.text
    return ['text/history/images, nullable SDK phase and metadata', 'model defaults and request overrides',
            'Responses discovery/capabilities and text-only image rejection before inference',
            'pre-inference malformed/unsupported/limit rejection', 'native passthrough and upstream errors',
            'SSE lifecycle, IDs, sequence, complete/incomplete, usage tail and generation failure']


async def stream_contracts():
    from titanos.responses import adapt_response, chat_request, make_response
    from titanos.server import Responses
    from titanos.engine import EngineError

    payload = Responses.model_validate({'model': 'chat', 'input': 'Hi', 'stream': True})

    for malformed in [{'choices': [{'message': None}]}, {'choices': [{'message': 'bad'}]},
                      {'choices': [{'message': []}]}, {'choices': [None]}, {'choices': None},
                      {'choices': [{'message': {'content': 'x'}, 'finish_reason': []}]}]:
        try:
            adapt_response(malformed, make_response(payload, chat_request(payload)))
        except EngineError as error:
            assert error.status == 502 and error.code == 'invalid_backend_response', error
        else:
            raise AssertionError(f'Malformed backend response was accepted: {malformed}')

    def adapt(body):
        return adapt_response(StreamingResponse(body, media_type='text/event-stream'),
                              make_response(payload, chat_request(payload)))

    def frame(value):
        return ('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode()

    def delta(text):
        return {'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}

    finish = {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}

    class Chunks:
        # An explicit aclose receipt also observes a close before first iteration.
        def __init__(self, values):
            self.values = iter(values)
            self.reads = 0
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.reads += 1
            try:
                return next(self.values)
            except StopIteration:
                raise StopAsyncIteration from None

        async def aclose(self):
            self.closed = True

    async def consume(body):
        adapted = adapt(body)
        assert isinstance(adapted, StreamingResponse)
        values = []
        async for piece in adapted.body_iterator:
            values.append(piece.decode() if isinstance(piece, bytes) else piece)
        assert body.closed, 'The source stream was left open'
        return sse_rows(''.join(values))

    raw = frame(delta('Hello caf\u00e9'))
    split = raw.index(b'\xc3') + 1
    source = Chunks([raw[:split], raw[split:], frame(finish), frame({'choices': [], 'usage': USAGE}),
                     b'data: [DONE]\n\n'])
    rows = await consume(source)
    check_stream(rows)
    assert source.reads >= 5, 'The usage-only tail was not consumed'

    for tail in [[], [b'data: [DONE]\n\n'], [b'data: {malformed}\n\n'],
                 [frame(['not', 'a', 'chunk'])], [frame(delta(42))],
                 [frame({'error': {'message': 'fixture failure', 'type': 'inference_error'}})],
                 [frame({'choices': [{'delta': {}, 'finish_reason': 'unknown'}]})],
                 [frame({'choices': [{'delta': {'tool_calls': [{'id': 'call_x'}]}, 'finish_reason': 'tool_calls'}]})]]:
        rows = await consume(Chunks([frame(delta('partial')), *tail]))
        kinds = [row['type'] for row in rows]
        assert kinds[-1] == 'response.failed' and kinds.count('response.failed') == 1, rows
        assert 'response.completed' not in kinds and 'response.incomplete' not in kinds, rows
        terminal = rows[-1]['response']
        assert terminal['status'] == 'failed' and terminal['error']['message'], terminal
        assert text_of(terminal) == 'partial' and terminal['usage'] is None, terminal

    missing_usage = await consume(Chunks([frame(delta('partial')), frame(finish)]))
    assert missing_usage[-1]['type'] == 'response.completed'
    assert missing_usage[-1]['response']['usage'] is None, 'Absent usage must not become measured zeros'

    for after_text in [False, True]:
        source = Chunks([frame(delta('partial')), frame(finish)])
        iterator = adapt(source).body_iterator
        try:
            for _ in range(12):
                row = sse_rows(await anext(iterator))[0]
                if not after_text or row['type'] == 'response.output_text.delta':
                    break
            else:
                raise AssertionError('No output text delta appeared')
        finally:
            await iterator.aclose()
        assert source.closed, 'Early client close must reach the backend body iterator'
        assert source.reads == (1 if after_text else 0), source.reads
    return ['fragmented UTF-8, usage-only tail, malformed/failed/EOF source handling',
            'underlying iterator close before backend iteration and after text']


async def generation_cancellation():
    """Exercise actual route -> generate -> thread cleanup, without an ASGI buffer."""
    from titanos.server import Responses
    for mode in ['close', 'task_cancel', 'queue_pressure']:
        started, cancelled, stopped = threading.Event(), threading.Event(), threading.Event()
        filled = threading.Event()

        def infer(name, task, data, emit=None, cancel=None):
            assert emit is not None and cancel is not None
            started.set()
            try:
                emit('partial')
                if mode == 'queue_pressure':
                    for index in range(100):
                        if index == 32:
                            filled.set()
                        emit(' more')
                if cancel.wait(3):
                    cancelled.set()
                else:
                    raise AssertionError('Cancellation did not reach the generation thread')
                return {'text': 'partial', 'finish_reason': 'stop', 'usage': dict(USAGE)}
            finally:
                stopped.set()

        with tempfile.TemporaryDirectory(prefix='titanos-responses-cancel-') as temp:
            app = create_app(Config(models={'chat': ModelSpec(path=temp, task='chat')}), output_dir=Path(temp))
            app.state.engine.target.infer = infer
            route = next(route for route in app.routes if getattr(route, 'path', None) == '/v1/responses')
            request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/responses', 'headers': []})
            payload = Responses.model_validate({'model': 'chat', 'input': 'Hi', 'stream': True})
            try:
                result = await route.endpoint(payload, request)
                iterator = result.body_iterator
                try:
                    async with asyncio.timeout(2):
                        async for piece in iterator:
                            if sse_rows(piece)[0]['type'] == 'response.output_text.delta':
                                break
                    if mode == 'queue_pressure':
                        assert await asyncio.to_thread(filled.wait, 1), 'Producer never filled its queue'
                    elif mode == 'task_cancel':
                        pending = asyncio.create_task(anext(iterator))
                        await asyncio.sleep(0)  # Enter the next source read before cancellation.
                        pending.cancel()
                        try:
                            await asyncio.wait_for(pending, 2)
                        except asyncio.CancelledError:
                            pass
                        else:
                            raise AssertionError('Task cancellation was swallowed')
                        assert cancelled.is_set() and stopped.is_set(), 'Cancellation did not join the producer'
                finally:
                    await asyncio.wait_for(iterator.aclose(), 2)
                assert started.is_set() and cancelled.is_set() and stopped.is_set(), mode
            finally:
                await app.state.native.close()
                await asyncio.to_thread(app.state.engine.close)
    return ['actual generation thread exits on close, pending-read cancellation, and full-queue cancellation']


def backend_token_aliases():
    """Exercise real argument-building methods with synthetic library boundaries."""
    from titanos.backends import LoadedModel
    from titanos.genai import GenAIModel
    captured = {}

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def __call__(self, text, **kwargs):
            return Inputs(input_ids=types.SimpleNamespace(shape=(1, 2)))

        def encode(self, text):
            return types.SimpleNamespace(input_ids=types.SimpleNamespace(shape=(1, 2)))

        def decode(self, tokens, **kwargs):
            return 'synthetic'

    def transformers_generate(**kwargs):
        captured['transformers'] = kwargs
        return [[0, 1, 2]]

    def genai_generate(prompts, **kwargs):
        captured['genai'] = kwargs
        return types.SimpleNamespace(texts=['synthetic'], finish_reasons=['STOP'],
                                     perf_metrics=types.SimpleNamespace(get_num_input_tokens=lambda: 2,
                                                                       get_num_generated_tokens=lambda: 1))

    stubs = {
        'torch': types.SimpleNamespace(is_floating_point=lambda value: False),
        'transformers': types.SimpleNamespace(TextStreamer=type('TextStreamer', (), {}),
                                              StoppingCriteria=type('StoppingCriteria', (), {}),
                                              StoppingCriteriaList=list),
        'openvino_genai': types.SimpleNamespace(GenerationFinishReason=types.SimpleNamespace(LENGTH='LENGTH'),
            GenerationConfig=lambda: types.SimpleNamespace(**{key: None for key in
                ['max_new_tokens','do_sample','apply_chat_template','num_return_sequences','echo','temperature','top_p','top_k']})),
    }
    defaults = {'max_tokens': 11, 'max_completion_tokens': 13, 'max_output_tokens': 17, 'top_k': 7}
    spec = ModelSpec(path='unused', task='chat', defaults=defaults)
    data = {'prompt': 'synthetic', 'max_tokens': 19, 'temperature': .3, 'top_p': .6}
    with patch.dict('sys.modules', stubs):
        loaded = object.__new__(LoadedModel)
        loaded.spec, loaded.processor, loaded.device, loaded.tensor_parallel = spec, Tokenizer(), 'cpu', False
        loaded.model = types.SimpleNamespace(config=types.SimpleNamespace(max_position_embeddings=1024),
                                             generate=transformers_generate)
        assert loaded.chat(data, None, threading.Event())['text'] == 'synthetic'
        genai = object.__new__(GenAIModel)
        genai.spec, genai.tokenizer, genai.context = spec, Tokenizer(), 1024
        genai.model = types.SimpleNamespace(generate=genai_generate)
        assert genai.infer('chat', data)['text'] == 'synthetic'
    for backend, kwargs in captured.items():
        assert not {'max_tokens', 'max_completion_tokens', 'max_output_tokens'}.intersection(kwargs), (backend, kwargs)
        assert (kwargs['max_new_tokens'], kwargs['top_k'], kwargs['temperature'], kwargs['top_p']) == (19, 7, .3, .6)
    assert set(captured) == {'transformers', 'genai'}
    return ['real Transformers/GenAI argument builders strip API token aliases and preserve resolved tuning']


def main():
    checks = contracts()
    checks.extend(asyncio.run(stream_contracts()))
    checks.extend(asyncio.run(generation_cancellation()))
    checks.extend(backend_token_aliases())
    print(json.dumps({'passed': True, 'checks': checks,
                      'scope': 'Synthetic protocol and cancellation checks; no model or hardware inference'}))


def native_llama_events():
    """What the llama.cpp server (b10809) streams for /v1/responses: response.created/in_progress carry only {id, object, status},
    no event has a sequence_number, the item and part events have no output_index/content_index; a text message item is
    followed here by a function_call item (captured from the CPU binary, 2026-10-06)."""
    text = {'type': 'output_text', 'annotations': [], 'logprobs': [], 'text': 'Hi'}
    call = {'id': 'fc_n', 'type': 'function_call', 'status': 'completed', 'arguments': '{}', 'call_id': 'call_n', 'name': 'f'}
    return [{'type': 'response.created', 'response': {'id': 'resp_native', 'object': 'response', 'status': 'in_progress'}},
            {'type': 'response.in_progress', 'response': {'id': 'resp_native', 'object': 'response', 'status': 'in_progress'}},
            {'type': 'response.output_item.added', 'item': {'content': [], 'id': 'msg_n', 'role': 'assistant', 'status': 'in_progress', 'type': 'message'}},
            {'type': 'response.content_part.added', 'item_id': 'msg_n', 'part': {'type': 'output_text', 'text': ''}},
            {'type': 'response.output_text.delta', 'item_id': 'msg_n', 'delta': 'Hi'},
            {'type': 'response.output_text.done', 'item_id': 'msg_n', 'text': 'Hi'},
            {'type': 'response.content_part.done', 'item_id': 'msg_n', 'part': text},
            {'type': 'response.output_item.done', 'item': {'type': 'message', 'status': 'completed', 'id': 'msg_n', 'content': [text], 'role': 'assistant'}},
            {'type': 'response.output_item.added', 'item': {**call, 'arguments': '', 'status': 'in_progress'}},
            {'type': 'response.function_call_arguments.delta', 'delta': '{}', 'item_id': 'fc_n'},
            {'type': 'response.output_item.done', 'item': call},
            {'type': 'response.completed', 'response': {'id': 'resp_native', 'object': 'response', 'created_at': 1, 'status': 'completed', 'model': 'x.gguf',
                                                        'output': [{'type': 'message', 'status': 'completed', 'id': 'msg_n', 'content': [text], 'role': 'assistant'}, call],
                                                        'usage': {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5}}, 'timings': {'predicted_n': 2}}]


def check_native_stream(text, sent):
    """responses.native_stream over the frames above: every frame re-emitted in order with a sequence number, the response
    objects completed (output list, created_at, the requested model) without overwriting what llama-server sent, output_index
    from the order of output_item.added, content_index from the parts of each message item; the stock OpenAI SDK's
    responses.stream() helper indexes exactly these fields. Returns True (asserts on failure)."""
    lines = text.split('\n')
    rows = [json.loads(line[5:]) for line in lines if line.startswith('data:')]
    assert [row['type'] for row in rows] == [event['type'] for event in sent], [row['type'] for row in rows]
    assert [line for line in lines if line.startswith('event:')] == ['event: ' + event['type'] for event in sent]
    assert [row['sequence_number'] for row in rows] == list(range(len(sent))), [row.get('sequence_number') for row in rows]
    created = rows[0]['response']
    assert created['output'] == [] and isinstance(created['created_at'], int) and created['model'] == 'native', created
    assert created['id'] == 'resp_native' and created['status'] == 'in_progress' and created['object'] == 'response', created
    assert 'error' in created and 'incomplete_details' in created and 'usage' in created, sorted(created)
    by_type = {}
    for row in rows:
        by_type.setdefault(row['type'], []).append(row)
    assert [row['output_index'] for row in by_type['response.output_item.added']] == [0, 1]
    assert [row['output_index'] for row in by_type['response.output_item.done']] == [0, 1]
    assert (by_type['response.content_part.added'][0]['output_index'], by_type['response.content_part.added'][0]['content_index']) == (0, 0)
    assert (by_type['response.output_text.delta'][0]['output_index'], by_type['response.output_text.delta'][0]['content_index']) == (0, 0)
    assert by_type['response.content_part.done'][0]['content_index'] == 0 and by_type['response.output_text.done'][0]['content_index'] == 0
    delta = by_type['response.function_call_arguments.delta'][0]
    assert delta['output_index'] == 1 and 'content_index' not in delta and delta['delta'] == '{}', delta
    completed = by_type['response.completed'][0]
    assert completed['response']['model'] == 'x.gguf' and completed['response']['created_at'] == 1, 'what llama-server sent is kept'
    assert len(completed['response']['output']) == 2 and completed['response']['usage']['total_tokens'] == 5 and completed['timings'] == {'predicted_n': 2}
    assert completed['response']['status'] == 'completed' and completed['response']['tool_choice'] == 'auto'
    return True


if __name__ == '__main__':
    main()
