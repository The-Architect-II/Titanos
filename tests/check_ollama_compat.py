"""The Ollama client surface over a synthetic local inference seam: GET /api/version outside managed role, POST /api/show
from the registry and the passive model inspection, keep_alive (0 unloads after the answer, negative is resident, a
duration is named in X-Titanos-Dropped, wrong values are 400), options.num_ctx (named in X-Titanos-Dropped on backends
without a context option; on llama-cpp models honoured up to the largest per-slot context, 400 context_limit above it)
and the typed 400 refusals of /api/pull, /api/create, /api/copy and /api/delete that name Titanos's acquisition path; the
Ollama tool loop (ids synthesized, results bound by tool_name, streamed argument fragments gathered into whole tool_calls)
and X-Titanos-Ignored on /api/embed and /api/embeddings; HiveMind's Ollama options object on the OpenAI chat and completions
routes (its exact 2026-10-05 body shape: 200 with options.num_ctx named in X-Titanos-Ignored on transformers, unnamed up to the
largest per-slot native context and named above it where /api/chat answers 400 context_limit, every other key named, never a 422).
No model weights, no network, no process; the native proxy is a synthetic seam as in tests/check_protocols.py."""
import json
import sys
import tempfile
from pathlib import Path

from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from titanos import __version__  # noqa: E402
from titanos.config import Config, ModelSpec, NativeOptions  # noqa: E402
from titanos.engine import EngineError  # noqa: E402
from titanos.protocols import keep_alive_policy  # noqa: E402
from titanos.server import create_app  # noqa: E402


def policy_table():
    table = ((None, None), (0, 'unload'), ('0', 'unload'), ('0s', 'unload'), (0.0, 'unload'), (-1, 'resident'), ('-1', 'resident'),
             ('-1m', 'resident'), (300, 'dropped'), ('5m', 'dropped'), ('1h30m', 'dropped'), ('1.5h', 'dropped'), (' 10s ', 'dropped'))
    for value, expected in table:
        assert keep_alive_policy(value) == expected, (value, keep_alive_policy(value))
    for bad in (True, 'soon', '5 minutes', [], {}, 'm', '5x', '', '-'):
        try:
            keep_alive_policy(bad)
        except EngineError as exc:
            assert exc.status == 400 and 'keep_alive' in str(exc), (bad, exc.status, str(exc))
        else:
            raise AssertionError(f'keep_alive {bad!r} was accepted')
    return len(table)


def main():
    checks = ['keep_alive_policy table (%d values, 9 refusals)' % policy_table()]
    seen = []

    def infer(name, task, data, emit=None, cancel=None):
        seen.append((name, task, data))
        if task == 'embeddings':
            return {'vectors': [[.1, .2] for _ in data['input']], 'usage': {'prompt_tokens': 2, 'total_tokens': 2}}
        if emit:
            emit('Hello'); emit(' world')
        return {'text': 'Hello world', 'finish_reason': 'stop', 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}

    with tempfile.TemporaryDirectory() as temp:
        Path(temp, 'config.json').write_text(json.dumps({'architectures': ['LlamaForCausalLM'], 'model_type': 'llama'}), encoding='utf-8')
        cfg = Config(models={'chat': ModelSpec(path=temp, task='chat', defaults={'max_tokens': 64}),
                             'embed': ModelSpec(path=temp, task='embeddings'),
                             'native': ModelSpec(path=temp, task='chat', backend='llama-cpp',
                                                 native=NativeOptions(model_file='model.gguf', context_size=2048, slots=2)),
                             'gembed': ModelSpec(path=temp, task='embeddings', backend='llama-cpp',
                                                 native=NativeOptions(model_file='embed.gguf', context_size=512, slots=1))})
        app = create_app(cfg, output_dir=Path(temp) / 'outputs')
        app.state.engine.target.infer = infer
        unloaded = []
        original_unload = app.state.engine.target.unload

        def unload(name=None, worker=None):
            unloaded.append(name)
            return original_unload(name, worker)
        app.state.engine.target.unload = unload
        with TestClient(app, base_url='http://localhost') as c:
            # GET /api/version: registered outside managed role with the managed role's body; Ollama clients read `version`
            r = c.get('/api/version')
            assert r.status_code == 200 and r.json() == {'name': 'Titanos', 'version': __version__, 'role': 'standalone'}, r.text
            checks.append('GET /api/version in standalone role')

            # GET /api/ps: Titanos catalog rows plus Ollama's name/model (= id), so a stock client sees what is resident
            assert c.get('/api/ps').json()['models'] == []
            app.state.engine.states['chat'] = 'loaded'
            rows = c.get('/api/ps').json()['models']
            assert [(row['name'], row['model'], row['id'], row['status']) for row in rows] == [('chat', 'chat', 'chat', 'loaded')], rows
            app.state.engine.states['chat'] = 'unloaded'
            checks.append('GET /api/ps rows carry name and model')

            # POST /api/show: registry + passive inspection, Ollama's keys, no load
            r = c.post('/api/show', json={'model': 'chat'})
            assert r.status_code == 200, r.text
            doc = r.json()
            assert doc['details'] == {'parent_model': '', 'format': 'transformers', 'family': 'chat', 'families': ['chat'], 'parameter_size': '', 'quantization_level': ''}, doc['details']
            assert doc['capabilities'] == ['completion'] and doc['model_info']['general.architecture'] == 'LlamaForCausalLM', doc
            assert doc['parameters'] == 'num_predict 64' and doc['template'] == '' and 'models/register' in doc['modelfile'], doc
            assert doc['titanos']['status'] == 'unloaded' and doc['titanos']['setup']['scope'] == 'passive_local_files_and_json', doc['titanos']
            assert seen == [] and unloaded == [], 'show must not load or unload'
            r = c.post('/api/show', json={'name': 'native', 'verbose': True})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'verbose', (r.status_code, r.headers, r.text)
            assert r.json()['parameters'] == 'num_ctx 2048' and r.json()['model_info']['llama-cpp.slots'] == 2 and r.json()['details']['format'] == 'llama-cpp', r.text
            assert c.post('/api/show', json={'model': 'embed'}).json()['capabilities'] == ['embedding']
            assert c.post('/api/show', json={'model': 'missing'}).status_code == 404
            r = c.post('/api/show', json={})
            assert r.status_code == 400 and r.json()['error']['message'].startswith('model is required'), r.text
            assert c.post('/api/show', json={'model': 'chat', 'system': 'x'}).status_code == 400
            assert c.post('/api/show', json={'model': 'chat', 'verbose': 'yes'}).status_code == 400
            checks.append('POST /api/show (transformers, llama-cpp, embeddings, 404, typed 400s)')

            # Ollama library routes: typed refusals naming the field and Titanos's acquisition path; nothing changes
            for method, route, body, field, path in (('POST', '/api/pull', {'model': 'llama3'}, 'model', 'POST /api/models/register'),
                                                     ('POST', '/api/create', {'model': 'x', 'from': 'chat'}, 'model', 'POST /api/models/register'),
                                                     ('POST', '/api/copy', {'source': 'chat', 'destination': 'chat2'}, 'destination', 'POST /api/models/register'),
                                                     ('DELETE', '/api/delete', {'model': 'chat'}, 'model', 'DELETE /api/models/registrations/{name}')):
                r = c.request(method, route, json=body)
                assert r.status_code == 400, (route, r.status_code, r.text)
                err = r.json()['error']
                assert err['code'] == 'unsupported_capability' and err['param'] == field and route in err['message'], (route, err)
                assert path in err['message'] and '/api/acquisitions' in err['message'], (route, err['message'])
            assert c.post('/api/pull').status_code == 400  # no body at all
            assert c.post('/api/pull', content=b'[1]', headers={'content-type': 'application/json'}).status_code == 400
            assert sorted(m['name'] for m in c.get('/api/tags').json()['models']) == ['chat', 'embed', 'gembed', 'native']
            checks.append('typed refusals on /api/pull, /api/create, /api/copy, /api/delete')

            # keep_alive on /api/chat and /api/generate
            body = {'model': 'chat', 'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': False}
            r = c.post('/api/chat', json={**body, 'keep_alive': '5m'})
            assert r.status_code == 200 and r.headers['x-titanos-dropped'] == 'keep_alive', (r.status_code, r.headers, r.text)
            assert 'keep_alive' not in seen[-1][2] and unloaded == []
            r = c.post('/api/chat', json={**body, 'keep_alive': -1})
            assert r.status_code == 200 and 'x-titanos-dropped' not in r.headers and unloaded == [], r.headers
            r = c.post('/api/chat', json={**body, 'keep_alive': 0})
            assert r.status_code == 200 and r.json()['message']['content'] == 'Hello world' and 'x-titanos-dropped' not in r.headers, r.text
            assert unloaded == ['chat'], 'keep_alive 0 unloads after the answer'
            r = c.post('/api/generate', json={'model': 'chat', 'prompt': 'Hi', 'stream': False, 'keep_alive': '0s'})
            assert r.status_code == 200 and r.json()['response'] == 'Hello world' and unloaded == ['chat', 'chat'], r.text
            r = c.post('/api/chat', json={**body, 'stream': True, 'keep_alive': '1h'})
            assert r.status_code == 200 and r.headers['x-titanos-dropped'] == 'keep_alive', r.headers
            rows = [json.loads(line) for line in r.text.splitlines()]
            assert rows[-1]['done'] and len(unloaded) == 2
            r = c.post('/api/chat', json={**body, 'stream': True, 'keep_alive': 0})
            rows = [json.loads(line) for line in r.text.splitlines()]
            assert ''.join(row['message']['content'] for row in rows) == 'Hello world' and rows[-1]['done'], rows
            assert unloaded == ['chat'] * 3, 'streaming keep_alive 0 unloads after the last frame'
            r = c.post('/api/chat', json={**body, 'keep_alive': 'soon'})
            assert r.status_code == 400 and 'keep_alive' in r.json()['error']['message'], r.text
            assert c.post('/api/chat', json={**body, 'keep_alive': [0]}).status_code == 422  # not one of Ollama's types
            checks.append('keep_alive on /api/chat and /api/generate (duration named, negative resident, 0 unloads, 400)')

            # keep_alive on the embedding routes; options stays refused there (unchanged)
            r = c.post('/api/embed', json={'model': 'embed', 'input': ['one', 'two'], 'keep_alive': '10m'})
            assert r.status_code == 200 and len(r.json()['embeddings']) == 2 and r.headers['x-titanos-dropped'] == 'keep_alive', r.text
            assert 'keep_alive' not in seen[-1][2] and unloaded == ['chat'] * 3
            r = c.post('/api/embed', json={'model': 'embed', 'input': 'one', 'keep_alive': 0})
            assert r.status_code == 200 and unloaded[-1] == 'embed', r.text
            r = c.post('/api/embeddings', json={'model': 'embed', 'prompt': 'one', 'keep_alive': '0'})
            assert r.status_code == 200 and r.json()['embedding'] == [.1, .2] and unloaded == ['chat'] * 3 + ['embed'] * 2, r.text
            assert c.post('/api/embed', json={'model': 'embed', 'input': 'one', 'keep_alive': 'x'}).status_code == 400
            assert c.post('/api/embed', json={'model': 'embed', 'input': 'one', 'options': {}}).status_code == 400
            checks.append('keep_alive on /api/embed and /api/embeddings')

            # options.num_ctx: named on transformers (the checkpoint's own context bounds prompt + max_tokens)
            r = c.post('/api/chat', json={**body, 'options': {'num_ctx': 4096, 'num_predict': 5}})
            assert r.status_code == 200 and r.headers['x-titanos-dropped'] == 'options.num_ctx', (r.status_code, r.headers, r.text)
            assert 'num_ctx' not in seen[-1][2] and seen[-1][2]['max_tokens'] == 5, seen[-1][2]
            r = c.post('/api/chat', json={**body, 'options': {'num_ctx': 4096}, 'keep_alive': '5m'})
            assert r.headers['x-titanos-dropped'] == 'keep_alive, options.num_ctx', r.headers
            r = c.post('/api/generate', json={'model': 'chat', 'prompt': 'Hi', 'stream': True, 'options': {'num_ctx': 8}})
            assert r.status_code == 200 and r.headers['x-titanos-dropped'] == 'options.num_ctx', r.headers
            assert c.post('/api/chat', json={**body, 'options': {'num_ctx': 0}}).status_code == 422  # ge=1
            assert c.post('/api/chat', json={**body, 'options': {'num_gpu': 1}}).status_code == 422  # other load options stay refused
            # options.num_ctx against the largest per-slot native context (no GGUF header here: the registered 2048 over 2 slots = 1024)
            proxied = []

            async def native_proxy(request, route, data, **kwargs):
                proxied.append(data)
                return Response(json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'Native answer'}, 'finish_reason': 'stop'}],
                                            'usage': {'prompt_tokens': 7, 'completion_tokens': 4}}), media_type='application/json')
            app.state.native.proxy = native_proxy
            native_body = {'model': 'native', 'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': False}
            r = c.post('/api/chat', json={**native_body, 'options': {'num_ctx': 1024}})
            assert r.status_code == 200 and 'x-titanos-dropped' not in r.headers and 'num_ctx' not in proxied[-1], (r.headers, r.text, proxied)
            r = c.post('/api/chat', json={**native_body, 'options': {'num_ctx': 1025}})
            assert r.status_code == 400, r.text
            err = r.json()['error']
            assert err['code'] == 'context_limit' and 'options.num_ctx 1025' in err['message'] and 'largest per-slot context 1024' in err['message'] and 'native.context_size' in err['message'], err
            assert len(proxied) == 1, 'the refusal happens before the native runtime is asked'
            r = c.post('/api/chat', json={**native_body, 'keep_alive': 0})
            assert r.status_code == 200 and unloaded[-1] == 'embed', r.text  # the native pool, not the engine, unloads a native model; not loaded is a no-op
            checks.append('options.num_ctx (named on transformers; admitted or 400 context_limit against the largest per-slot native context)')

            # HiveMind's gateway sends Ollama's options object with every OpenAI chat request to a non-cloud backend; this is the
            # exact body of the 2026-10-05 smoke, which answered 422 "body.options: Extra inputs are not permitted" at fd5638a
            hivemind = {'model': 'chat', 'messages': [{'role': 'user', 'content': 'Say hello in five words.'}], 'max_tokens': 32, 'temperature': 0,
                        'options': {'num_ctx': 4096}}
            r = c.post('/v1/chat/completions', json=hivemind)
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'options.num_ctx', (r.status_code, r.headers, r.text)
            assert 'options' not in seen[-1][2] and 'num_ctx' not in seen[-1][2] and seen[-1][2]['max_tokens'] == 32, seen[-1][2]
            r = c.post('/v1/chat/completions', json={**hivemind, 'stream': True})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'options.num_ctx' and r.text.startswith('data:'), (r.headers, r.text[:80])
            r = c.post('/v1/completions', json={'model': 'chat', 'prompt': 'Hi', 'options': {'num_ctx': 4096}})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'options.num_ctx', (r.status_code, r.headers, r.text)
            # every other options key is named as options.<key> (not mapped, not a 422), beside the compat fields
            r = c.post('/v1/chat/completions', json={**hivemind, 'options': {'num_ctx': 4096, 'num_gpu': 1, 'temperature': 0.9}, 'user': 'u'})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'options.num_ctx, options.num_gpu, options.temperature, user', r.headers
            assert seen[-1][2]['temperature'] == 0, seen[-1][2]  # options.temperature is named, never applied over the request's own field
            r = c.post('/v1/chat/completions', json={**hivemind, 'options': None})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'options', r.headers
            r = c.post('/v1/chat/completions', json={**hivemind, 'options': {}})
            assert r.status_code == 200 and 'x-titanos-ignored' not in r.headers, r.headers
            for options, fragment in (('4096', 'options must be a JSON object'), ({'num_ctx': 'big'}, 'options.num_ctx must be a positive integer'),
                                      ({'num_ctx': 0}, 'options.num_ctx must be a positive integer'), ({'num_ctx': True}, 'options.num_ctx must be a positive integer')):
                r = c.post('/v1/chat/completions', json={**hivemind, 'options': options})
                assert r.status_code == 400 and fragment in r.json()['error']['message'], (options, r.status_code, r.text)
            # against the largest per-slot native context (1024 here, without a GGUF header): within it nothing is named and the runtime is
            # sized to it (tests/check_native_context.py); above it named and not honoured, where /api/chat answers 400 context_limit; options never reaches llama-server
            r = c.post('/v1/chat/completions', json={**hivemind, 'model': 'native', 'options': {'num_ctx': 1024}})
            assert r.status_code == 200 and 'x-titanos-ignored' not in r.headers and 'options' not in proxied[-1] and 'num_ctx' not in proxied[-1], (r.headers, proxied[-1])
            count = len(proxied)
            r = c.post('/v1/chat/completions', json={**hivemind, 'model': 'native'})  # HiveMind's default num_ctx 4096 above the 1024 maximum
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'options.num_ctx', (r.status_code, r.headers, r.text)
            assert len(proxied) == count + 1 and 'options' not in proxied[-1] and 'num_ctx' not in proxied[-1], proxied[-1]
            ollama = c.post('/api/chat', json={**native_body, 'options': {'num_ctx': 4096}})
            assert ollama.status_code == 400 and ollama.json()['error']['code'] == 'context_limit' and 'options.num_ctx 4096' in ollama.json()['error']['message'], ollama.text
            assert len(proxied) == count + 1, 'the Ollama refusal happens before the native runtime is asked'
            checks.append("HiveMind's options object on /v1/chat/completions and /v1/completions (num_ctx unnamed up to the largest per-slot native context, named above it where /api/chat is 400 context_limit; other keys named; typed 400s, no 422)")

            # the Ollama tool loop in Ollama's own shape: assistant tool_calls without ids, tool results with tool_name and no
            # tool_call_id; Titanos synthesizes the ids and binds each result to the pending call of that name before llama-server sees it
            tool = {'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}}
            clock = {'type': 'function', 'function': {'name': 'get_time', 'parameters': {'type': 'object', 'properties': {}}}}
            call = lambda name, **arguments: {'function': {'name': name, 'arguments': arguments}}  # noqa: E731
            loop = [{'role': 'user', 'content': 'Weather in Paris?'}, {'role': 'assistant', 'content': '', 'tool_calls': [call('get_weather', city='Paris')]},
                    {'role': 'tool', 'content': 'Sunny, 22 C', 'tool_name': 'get_weather'}]
            r = c.post('/api/chat', json={**native_body, 'messages': loop, 'tools': [tool]})
            assert r.status_code == 200 and r.json()['message']['content'] == 'Native answer', r.text
            sent = proxied[-1]['messages']
            assert sent[1]['tool_calls'][0]['id'].startswith('call_') and sent[1]['tool_calls'][0]['type'] == 'function', sent[1]
            assert json.loads(sent[1]['tool_calls'][0]['function']['arguments']) == {'city': 'Paris'}, sent[1]
            assert sent[2] == {'role': 'tool', 'content': 'Sunny, 22 C', 'name': 'get_weather', 'tool_call_id': sent[1]['tool_calls'][0]['id']}, sent[2]
            # two calls answered out of order: tool_name picks the call; a result without tool_name takes the next unanswered call
            two = [loop[0], {'role': 'assistant', 'content': '', 'tool_calls': [call('get_weather', city='Paris'), call('get_time')]},
                   {'role': 'tool', 'content': '12:00', 'tool_name': 'get_time'}, {'role': 'tool', 'content': 'Sunny'}]
            r = c.post('/api/chat', json={**native_body, 'messages': two, 'tools': [tool, clock]})
            assert r.status_code == 200, r.text
            sent = proxied[-1]['messages']
            ids = [item['id'] for item in sent[1]['tool_calls']]
            assert len(set(ids)) == 2 and sent[2]['tool_call_id'] == ids[1] and sent[3]['tool_call_id'] == ids[0] and 'name' not in sent[3], sent
            # an explicit tool_call_id (the OpenAI shape over the Ollama route) is kept
            kept = [loop[0], {'role': 'assistant', 'content': '', 'tool_calls': [{**call('get_weather', city='Paris'), 'id': 'call_given'}]},
                    {'role': 'tool', 'content': 'Sunny', 'tool_call_id': 'call_given'}]
            r = c.post('/api/chat', json={**native_body, 'messages': kept, 'tools': [tool]})
            assert r.status_code == 200 and proxied[-1]['messages'][2]['tool_call_id'] == 'call_given', (r.text, proxied[-1]['messages'])
            # a tool result with nothing to answer is a typed 400 naming the message, never a schema dump
            sent_before = len(proxied)
            r = c.post('/api/chat', json={**native_body, 'messages': [loop[0], loop[2]], 'tools': [tool]})
            err = r.json().get('error', {})
            assert r.status_code == 400 and err.get('type') == 'invalid_request_error' and err.get('message', '').startswith("messages[1]: the tool result for tool_name 'get_weather'"), r.text
            r = c.post('/api/chat', json={**native_body, 'messages': [*loop, {'role': 'tool', 'content': 'again', 'tool_name': 'get_weather'}], 'tools': [tool]})
            assert r.status_code == 400 and r.json()['error']['message'].startswith('messages[3]:'), r.text
            assert len(proxied) == sent_before, 'the refusals happen before the native runtime is asked'
            checks.append('Ollama tool loop (ids synthesized, results bound by tool_name or order, explicit tool_call_id kept, orphan result 400)')

            # streaming tool calls in Ollama's shape: llama-server's OpenAI argument fragments (one per chunk, partial strings) become
            # one whole tool_calls frame with object arguments ahead of the final frame; text deltas pass through as they arrive
            def delta(**fields):
                return {'choices': [{'index': 0, 'delta': {'role': 'assistant', **fields}, 'finish_reason': None}]}

            def fragment(index, **fn):
                return {'index': index, 'function': fn}
            streamed = {}

            async def native_stream(request, route, data, **kwargs):
                proxied.append(data)
                return StreamingResponse((f'data: {json.dumps(chunk)}\n\n' for chunk in streamed['frames']), media_type='text/event-stream')
            app.state.native.proxy = native_stream
            streamed['frames'] = [delta(content=''), delta(content='', tool_calls=[{'index': 0, 'id': 'call_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{'}}]),
                                  delta(tool_calls=[fragment(0, arguments='"city": "Paris"')]), delta(tool_calls=[fragment(0, arguments='}')]),
                                  {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]}, {'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': 9}}]
            r = c.post('/api/chat', json={**native_body, 'stream': True, 'tools': [tool]})
            assert r.status_code == 200, r.text
            rows = [json.loads(line) for line in r.text.splitlines()]
            with_calls = [row for row in rows if row['message'].get('tool_calls')]
            assert len(with_calls) == 1 and with_calls[0]['message']['tool_calls'] == [{'function': {'name': 'get_weather', 'arguments': {'city': 'Paris'}}}], rows
            assert not with_calls[0]['done'] and rows[-1]['done'] and rows[-1]['done_reason'] == 'tool_calls' and rows[-1]['eval_count'] == 9, rows[-1]
            assert rows[0]['message'] == {'role': 'assistant', 'content': ''} and all(isinstance(row['message'].get('content'), str) for row in rows), rows
            # two calls, text between the fragments, and a last fragment travelling with the finish frame
            streamed['frames'] = [delta(content='Checking'), delta(tool_calls=[fragment(0, name='get_weather', arguments='{"city": "Pa')]), delta(content=' now'),
                                  delta(tool_calls=[fragment(1, name='get_time', arguments='{}')]),
                                  {'choices': [{'index': 0, 'delta': {'tool_calls': [fragment(0, arguments='ris"}')]}, 'finish_reason': 'tool_calls'}]}]
            rows = [json.loads(line) for line in c.post('/api/chat', json={**native_body, 'stream': True, 'tools': [tool, clock]}).text.splitlines()]
            assert ''.join(row['message']['content'] for row in rows) == 'Checking now', rows
            calls_out = [row['message']['tool_calls'] for row in rows if row['message'].get('tool_calls')]
            assert calls_out == [[{'function': {'name': 'get_weather', 'arguments': {'city': 'Paris'}}}, {'function': {'name': 'get_time', 'arguments': {}}}]], calls_out
            assert rows[-1]['done'] and rows[-1]['done_reason'] == 'tool_calls' and 'tool_calls' not in rows[-1]['message'], rows[-1]
            # argument text that never became JSON is one error frame in Ollama's string envelope, never a fragment
            streamed['frames'] = [delta(tool_calls=[fragment(0, name='get_weather', arguments='{"city": ')]), {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]}]
            rows = [json.loads(line) for line in c.post('/api/chat', json={**native_body, 'stream': True, 'tools': [tool]}).text.splitlines()]
            assert any(str(row.get('error', '')).startswith('Model returned invalid tool arguments') for row in rows), rows
            assert not any(row.get('message', {}).get('tool_calls') for row in rows), rows
            checks.append('streaming /api/chat tool_calls (fragments gathered into whole calls with object arguments; text deltas pass; invalid arguments are one error frame)')

            # X-Titanos-Ignored on the Ollama embedding routes: `user` (and dimensions null) are removed from the request and named on
            # the answer as on /v1/embeddings, for the local embedding model and for a GGUF embedding model proxied to llama-server
            r = c.post('/api/embed', json={'model': 'embed', 'input': 'a', 'user': 'u'})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'user' and 'user' not in seen[-1][2], (r.headers, seen[-1])
            r = c.post('/api/embeddings', json={'model': 'embed', 'prompt': 'a', 'user': 'u', 'keep_alive': '5m'})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'user' and r.headers['x-titanos-dropped'] == 'keep_alive', r.headers
            r = c.post('/api/embed', json={'model': 'embed', 'input': 'a', 'dimensions': None})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'dimensions', r.headers
            assert c.post('/api/embed', json={'model': 'embed', 'input': 'a', 'dimensions': 8}).status_code == 400
            assert 'x-titanos-ignored' not in c.post('/api/embed', json={'model': 'embed', 'input': 'a'}).headers

            async def native_embed(request, route, data, **kwargs):
                proxied.append(data)
                return Response(json.dumps({'object': 'list', 'model': data['model'], 'usage': {'prompt_tokens': 1, 'total_tokens': 1},
                                            'data': [{'object': 'embedding', 'index': 0, 'embedding': [1.0, 2.0]}]}), media_type='application/json')
            app.state.native.proxy = native_embed
            r = c.post('/api/embed', json={'model': 'gembed', 'input': 'a', 'user': 'u'})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'user' and r.json()['embeddings'] == [[1.0, 2.0]], (r.headers, r.text)
            assert proxied[-1]['model'] == 'gembed' and 'user' not in proxied[-1], proxied[-1]
            r = c.post('/api/embeddings', json={'model': 'gembed', 'prompt': 'a', 'user': 'u'})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'user' and r.json()['embedding'] == [1.0, 2.0], (r.headers, r.text)
            checks.append('X-Titanos-Ignored (user, dimensions null) on /api/embed and /api/embeddings for local and GGUF embedding models')
    print(json.dumps({'passed': True, 'checks': checks,
                      'scope': 'synthetic inference seam; the stock Ollama client against a real serve is tests/check_ollama_sdk.py'}))


if __name__ == '__main__':
    main()
