"""Agent capabilities on local models over a synthetic native (llama.cpp) seam and a synthetic in-process engine.

- Anthropic /v1/messages with stream=true and tools on a llama-cpp model streams tool_use content blocks with
  input_json_delta fragments and stop_reason tool_use (text and tool blocks in order, parallel calls, empty output,
  a backend error), and maps tool_choice.disable_parallel_tool_use to parallel_tool_calls=false.
- OpenAI /v1/chat/completions on a llama-cpp model forwards tools, tool_choice, parallel_tool_calls and the tool-call
  history unchanged.
- transformers/openvino-genai models refuse tool calling, tool-call history and structured output with a typed 400
  unsupported_capability that names the model, its backend, the missing capability and the route the client used, on
  the OpenAI chat, Ollama, Anthropic and Responses routes.
No model weights, no network, no process."""
import json
import tempfile
from pathlib import Path

from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient

from titanos.config import Config, ModelSpec, NativeOptions
from titanos.server import create_app


def chunk(delta=None, finish=None, usage=None):
    body = {'id': 'chatcmpl-x', 'object': 'chat.completion.chunk', 'model': 'native', 'choices': []}
    if delta is not None or finish is not None:
        body['choices'] = [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}]
    if usage is not None:
        body['usage'] = usage
    return 'data: ' + json.dumps(body) + '\n\n'


def tool(index, arguments, ident=None, name=None):
    call = {'index': index, 'function': {'arguments': arguments}}
    if ident:
        call.update(id=ident, type='function')
        call['function']['name'] = name
    return {'tool_calls': [call]}


USAGE = {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}
# what the synthetic llama.cpp server streams for each user prompt
SCRIPTS = {
    'tools': [chunk({'role': 'assistant', 'content': None}), chunk({'content': 'Let me check.'}),
              chunk(tool(0, '{"ci', 'call_a', 'get_weather')), chunk(tool(0, 'ty": "Paris"}')),
              chunk(tool(1, '', 'call_b', 'get_time')), chunk(tool(1, '{}')),
              chunk(finish='tool_calls'), chunk(usage=USAGE), 'data: [DONE]\n\n'],
    'tools-stop': [chunk(tool(0, '{"city": "Oslo"}', 'call_c', 'get_weather')), chunk(finish='stop', usage=USAGE), 'data: [DONE]\n\n'],
    'text': [chunk({'role': 'assistant', 'content': ''}), chunk({'content': 'Hello'}), chunk({'content': ' world'}),
             chunk(finish='stop'), chunk(usage=USAGE), 'data: [DONE]\n\n'],
    'long': [chunk({'content': 'Hel'}), chunk(finish='length', usage=USAGE), 'data: [DONE]\n\n'],
    'empty': [chunk(finish='stop', usage=USAGE), 'data: [DONE]\n\n'],
    'error': [chunk({'content': 'Hel'}), 'data: ' + json.dumps({'error': {'message': 'slot lost', 'type': 'server_error'}}) + '\n\n'],
}


def sse(text):
    rows, event = [], None
    for line in text.splitlines():
        if line.startswith('event: '):
            event = line[7:]
        elif line.startswith('data: '):
            data = json.loads(line[6:])
            assert data['type'] == event, (event, data)
            rows.append(data)
    return rows


def blocks(rows):
    """[(index, content_block, concatenated deltas)] in start order; asserts each block stops before the next starts."""
    out, open_index = [], None
    for row in rows:
        if row['type'] == 'content_block_start':
            assert open_index is None, ('block started while another is open', row)
            assert row['index'] == len(out), row
            out.append([row['index'], row['content_block'], ''])
            open_index = row['index']
        elif row['type'] == 'content_block_delta':
            delta = row['delta']
            out[row['index']][2] += delta['text'] if delta['type'] == 'text_delta' else delta['partial_json']
        elif row['type'] == 'content_block_stop':
            assert open_index == row['index'], row
            open_index = None
    assert open_index is None, 'a content block was never stopped'
    return out


def main():
    forwarded = []
    engine_calls = []

    def infer(name, task, data, emit=None, cancel=None):
        engine_calls.append((name, task))
        return {'text': 'Hello', 'finish_reason': 'stop', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}

    with tempfile.TemporaryDirectory() as temp:
        cfg = Config(models={
            'chat': ModelSpec(path=temp, task='chat'),
            'ov': ModelSpec(path=temp, task='chat', backend='openvino-genai'),
            'native': ModelSpec(path=temp, task='chat', backend='llama-cpp', device='cpu',
                                native=NativeOptions(model_file='unused.gguf', gpu_layers=0))})
        app = create_app(cfg, output_dir=Path(temp) / 'outputs')
        app.state.engine.target.infer = infer
        app.state.native.models['native'] = cfg.models['native']

        async def native_proxy(request, route, data):
            forwarded.append((route, data))
            prompt = data['messages'][-1]['content']
            if isinstance(prompt, list):
                prompt = prompt[0]['text']
            if data.get('stream'):
                frames = SCRIPTS[prompt]

                async def body():
                    for frame in frames:
                        yield frame.encode()
                return StreamingResponse(body(), media_type='text/event-stream')
            return Response(json.dumps({'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
                                        'usage': USAGE}), media_type='application/json')
        app.state.native.proxy = native_proxy

        tools = [{'name': 'get_weather', 'description': 'Weather by city', 'input_schema': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}},
                 {'name': 'get_time', 'input_schema': {'type': 'object', 'properties': {}}}]
        with TestClient(app, base_url='http://localhost') as c:
            def messages(prompt, **extra):
                return c.post('/v1/messages', json={'model': 'native', 'max_tokens': 64, 'stream': True,
                                                    'messages': [{'role': 'user', 'content': prompt}], **extra})

            # 1. streaming tool calls: text block, two tool_use blocks with incremental JSON, stop_reason tool_use
            r = messages('tools', tools=tools, tool_choice={'type': 'any', 'disable_parallel_tool_use': True})
            assert r.status_code == 200 and r.headers['content-type'].startswith('text/event-stream'), r.text
            route, data = forwarded[-1]
            assert route == '/v1/chat/completions' and data['stream'] is True and data['stream_options'] == {'include_usage': True}
            assert data['tools'] == [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather by city', 'parameters': tools[0]['input_schema']}},
                                     {'type': 'function', 'function': {'name': 'get_time', 'description': '', 'parameters': tools[1]['input_schema']}}], data['tools']
            assert data['tool_choice'] == 'required' and data['parallel_tool_calls'] is False, data
            rows = sse(r.text)
            assert [row['type'] for row in rows][:1] == ['message_start'] and rows[-1]['type'] == 'message_stop', rows
            assert rows[0]['message']['content'] == [] and rows[0]['message']['stop_reason'] is None
            parts = blocks(rows)
            assert [(index, block['type']) for index, block, _ in parts] == [(0, 'text'), (1, 'tool_use'), (2, 'tool_use')], parts
            assert parts[0][2] == 'Let me check.'
            assert parts[1][1] == {'type': 'tool_use', 'id': 'call_a', 'name': 'get_weather', 'input': {}} and json.loads(parts[1][2]) == {'city': 'Paris'}, parts[1]
            assert parts[2][1] == {'type': 'tool_use', 'id': 'call_b', 'name': 'get_time', 'input': {}} and json.loads(parts[2][2]) == {}, parts[2]
            kinds = [row['delta']['type'] for row in rows if row['type'] == 'content_block_delta']
            assert kinds == ['text_delta', 'input_json_delta', 'input_json_delta', 'input_json_delta'], kinds
            final = rows[-2]
            assert final['type'] == 'message_delta' and final['delta'] == {'stop_reason': 'tool_use', 'stop_sequence': None}, final
            assert final['usage'] == {'input_tokens': 11, 'output_tokens': 7}, final
            # a tool call the backend finished with stop is still a tool_use turn for the client loop
            parts = blocks(sse(messages('tools-stop', tools=tools).text))
            assert len(parts) == 1 and parts[0][1]['name'] == 'get_weather' and json.loads(parts[0][2]) == {'city': 'Oslo'}
            assert sse(messages('tools-stop', tools=tools).text)[-2]['delta']['stop_reason'] == 'tool_use'
            assert 'parallel_tool_calls' not in forwarded[-1][1], 'parallel_tool_calls is set only from disable_parallel_tool_use'
            # tool_choice.tool maps to a forced function
            messages('tools-stop', tools=tools, tool_choice={'type': 'tool', 'name': 'get_weather'})
            assert forwarded[-1][1]['tool_choice'] == {'type': 'function', 'function': {'name': 'get_weather'}}

            # 2. text-only, max_tokens, empty and failed streams keep their shapes
            rows = sse(messages('text', tools=tools).text)
            parts = blocks(rows)
            assert [(index, block, text) for index, block, text in parts] == [(0, {'type': 'text', 'text': ''}, 'Hello world')], parts
            assert rows[-2]['delta']['stop_reason'] == 'end_turn' and rows[-2]['usage']['output_tokens'] == 7
            rows = sse(messages('long').text)
            assert blocks(rows)[0][2] == 'Hel' and rows[-2]['delta']['stop_reason'] == 'max_tokens', rows
            rows = sse(messages('empty').text)
            assert blocks(rows) == [[0, {'type': 'text', 'text': ''}, '']] and rows[-2]['delta']['stop_reason'] == 'end_turn', rows
            rows = sse(messages('error').text)
            assert rows[-1]['type'] == 'error' and rows[-1]['error']['message'] == 'slot lost' and not any(row['type'] == 'message_stop' for row in rows), rows

            # 3. OpenAI chat tools/tool_choice/parallel_tool_calls and the tool-call history round-trip to llama.cpp unchanged
            history = [{'role': 'user', 'content': 'weather?'},
                       {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_a', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'}}]},
                       {'role': 'tool', 'tool_call_id': 'call_a', 'content': 'Sunny'}]
            openai_tools = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': tools[0]['input_schema']}}]
            body = {'model': 'native', 'messages': history, 'tools': openai_tools, 'tool_choice': 'auto', 'parallel_tool_calls': True, 'max_tokens': 32}
            r = c.post('/v1/chat/completions', json=body)
            assert r.status_code == 200, r.text
            route, data = forwarded[-1]
            assert route == '/v1/chat/completions' and all(data[key] == value for key, value in body.items() if key != 'messages'), data
            # the schema dump omits null fields (the assistant tool-call message's content: null), nothing else changes
            assert data['messages'] == [{k: v for k, v in message.items() if v is not None} for message in history], data['messages']
            assert 'require_loaded' not in data

            # 4. typed refusals name the model, backend, capability and route on in-process backends
            def refused(route, body, *needles):
                r = c.post(route, json=body)
                assert r.status_code == 400, (route, r.status_code, r.text)
                error = r.json()['error']
                assert error.get('code') == 'unsupported_capability', error
                message = error['message']
                for needle in needles + (route,):
                    assert needle in message, (needle, message)
                return message

            chat = {'model': 'chat', 'messages': [{'role': 'user', 'content': 'Hi'}]}
            message = refused('/v1/chat/completions', {**chat, 'tools': openai_tools}, "Model 'chat'", 'transformers backend', 'tool calling (tools)')
            assert 'llama-cpp' in message, message
            refused('/v1/chat/completions', {**chat, 'tool_choice': 'auto'}, 'transformers backend', 'tool_choice')
            refused('/v1/chat/completions', {**chat, 'parallel_tool_calls': False}, 'parallel_tool_calls')
            refused('/v1/chat/completions', {**chat, 'response_format': {'type': 'json_object'}}, 'structured output (response_format)')
            refused('/v1/chat/completions', {**chat, 'messages': history}, 'tool-call history (tool_calls and role tool messages)')
            refused('/v1/chat/completions', {**chat, 'messages': [{'role': 'developer', 'content': 'Be brief'}, *chat['messages']]}, 'the developer role')
            refused('/v1/chat/completions', {'model': 'ov', 'messages': chat['messages'], 'tools': openai_tools}, "Model 'ov'", 'openvino-genai backend')
            refused('/api/chat', {**chat, 'stream': False, 'tools': openai_tools}, 'transformers backend', 'tool calling (tools)')
            refused('/v1/messages', {'model': 'ov', 'max_tokens': 8, 'messages': chat['messages'], 'tools': tools}, "Model 'ov'", 'openvino-genai backend', 'tool calling (tools)')
            refused('/v1/messages', {'model': 'ov', 'max_tokens': 8, 'messages': chat['messages'], 'tools': tools, 'stream': True}, 'openvino-genai backend')
            responses = {'model': 'chat', 'input': 'Hi'}
            refused('/v1/responses', {**responses, 'tools': [{'type': 'function', 'name': 'get_weather', 'parameters': {}}]}, "Model 'chat'", 'transformers backend', 'tool calling (tools)')
            refused('/v1/responses', {**responses, 'tool_choice': 'required'}, 'tool_choice other than auto or none')
            refused('/v1/responses', {**responses, 'parallel_tool_calls': True}, 'parallel_tool_calls')
            refused('/v1/responses', {**responses, 'text': {'format': {'type': 'json_schema', 'name': 'x', 'schema': {}}}}, 'structured output (text.format)')
            refused('/v1/responses', {**responses, 'input': [{'type': 'function_call_output', 'call_id': 'c', 'output': 'x'}]}, 'tool-call history items (function_call and function_call_output)')
            refused('/v1/responses', {'model': 'ov', 'input': 'Hi', 'tools': [{'type': 'function', 'name': 'f', 'parameters': {}}]}, 'openvino-genai backend')
            # every refusal above happened before inference; the ordinary forms stay served by the in-process engine
            assert engine_calls == [], engine_calls
            r = c.post('/v1/responses', json={**responses, 'tools': [], 'tool_choice': 'auto', 'parallel_tool_calls': False})
            assert r.status_code == 200 and engine_calls == [('chat', 'chat')], (r.status_code, r.text, engine_calls)
            assert not any(route == '/v1/chat/completions' and data['model'] != 'native' for route, data in forwarded)
    print(json.dumps({'passed': True, 'checks': [
        'Anthropic streaming tool_use blocks with input_json_delta, stop_reason tool_use, disable_parallel_tool_use mapping',
        'Anthropic text/max_tokens/empty/error streams keep their shapes',
        'OpenAI chat tools, tool_choice, parallel_tool_calls and tool-call history forwarded to llama.cpp unchanged',
        'typed 400 naming model, backend, capability and route on chat, Ollama, Anthropic and Responses routes'],
        'scope': 'synthetic native seam; real llama.cpp agent loops are tests/check_agent_loops.py (manual tier)'}))


if __name__ == '__main__':
    main()
