"""Real agent loops against a real `titanos serve` on the llama.cpp CPU binary (manual tier: needs a GGUF).

Server side (this interpreter, src importable): a temporary TITANOS_HOME and registry with one llama-cpp chat model
(device cpu, gpu_layers 0), `python -m titanos --config <registry> serve` on one loopback port, then the client under
--sdk-python (default: this interpreter; an interpreter with the stock openai and anthropic packages and, for the
framework cases, openai-agents), then a graceful stop. The client imports nothing from titanos.

Client cases (pass, fail or untried; untried is never a pass):
  openai.chat.tools            stock OpenAI SDK: tools -> tool_calls -> role tool -> final text (nonstream)
  openai.chat.tools.stream     the same loop streaming: incremental tool_call arguments reconstruct valid JSON
  openai.chat.tool_choice      tool_choice {type: function} forces the named tool; parallel_tool_calls true/false accepted
  openai.chat.json_schema      response_format json_schema answers valid JSON of the schema (llama.cpp grammar)
  openai.responses.tools       stock OpenAI SDK Responses: tools -> function_call -> function_call_output -> message
  openai.responses.tools.stream  the same loop streaming: output_item.added function_call, arguments deltas, completed
  anthropic.messages.tools     stock Anthropic SDK: tools -> tool_use -> tool_result -> final text (nonstream)
  anthropic.messages.tools.stream  messages.stream(): tool_use block with input from input_json_delta, stop_reason tool_use
  agents.chat_completions      openai-agents Agent + function_tool over OpenAIChatCompletionsModel pointed at Titanos
  agents.responses             the same agent over OpenAIResponsesModel (the framework's default Responses path)
Engine status is whether the loop ran with valid shapes; whether the small model picks the right city is reported
separately as model quality, never as an engine failure. No hosted API is reached: every client is pointed at the
loopback serve with a placeholder key and the framework's tracing is disabled.

    python tests/check_agent_loops.py --llama-server <llama-server(.exe)> --gguf <model.gguf> --sdk-python <python> --report out.json
Output: one JSON document on stdout (also --report). Exit 1 on any fail, or with --strict on any untried.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEATHER = {'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'City name'}},
           'required': ['city'], 'additionalProperties': False}
QUESTION = 'What is the weather in Paris right now? Call the get_weather tool, then answer in one sentence.'


def weather(city):
    return f'Sunny, 22 degrees Celsius in {city}.'


# ----------------------------------------------------------------------------------------------------------------------
# client side: runs under the SDK interpreter; imports only the SDKs


class Untried(Exception):
    """A case that could not be tried here (a missing SDK); reported as untried, never as a pass."""


def quality(arguments, final_text):
    city = (arguments or {}).get('city')
    return {'tool_arguments': arguments, 'city_is_paris': isinstance(city, str) and city.strip().lower() == 'paris',
            'answer_mentions_22': '22' in (final_text or '')}


def client_main(base_url, model, report_path):
    results = []

    def record(case_id, function):
        started = time.monotonic()
        try:
            detail = function()
            results.append({'id': case_id, 'state': 'pass', 'seconds': round(time.monotonic() - started, 2), **(detail or {})})
        except Untried as exc:
            results.append({'id': case_id, 'state': 'untried', 'detail': str(exc)})
        except Exception as exc:  # reported per case; the orchestrator turns it into a failure
            results.append({'id': case_id, 'state': 'fail', 'seconds': round(time.monotonic() - started, 2),
                            'detail': f'{type(exc).__name__}: {exc}'[:1500]})

    def expect(condition, detail):
        if not condition:
            raise AssertionError(json.dumps(detail, default=str)[:1500])

    versions = {}
    try:
        import openai
        from importlib.metadata import version
        versions['openai'] = version('openai')
    except ImportError as exc:
        openai = None
        versions['openai'] = f'missing ({exc})'
    try:
        import anthropic
        from importlib.metadata import version
        versions['anthropic'] = version('anthropic')
    except ImportError as exc:
        anthropic = None
        versions['anthropic'] = f'missing ({exc})'
    try:
        from importlib.metadata import version
        versions['openai-agents'] = version('openai-agents')
    except Exception as exc:
        versions['openai-agents'] = f'missing ({exc})'

    os.environ.setdefault('OPENAI_API_KEY', 'sk-local-titanos-placeholder')
    chat_tool = {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city', 'parameters': WEATHER}}
    messages = [{'role': 'user', 'content': QUESTION}]

    def openai_client():
        if openai is None:
            raise Untried('the openai package is not importable here')
        return openai.OpenAI(base_url=base_url + '/v1', api_key='sk-local-titanos-placeholder', timeout=300, max_retries=0)

    def chat_tools():
        client = openai_client()
        first = client.chat.completions.create(model=model, messages=messages, tools=[chat_tool], tool_choice='auto', temperature=0, max_tokens=96)
        choice = first.choices[0]
        calls = choice.message.tool_calls or []
        expect(len(calls) >= 1 and choice.finish_reason == 'tool_calls', {'finish_reason': choice.finish_reason, 'message': choice.message.model_dump()})
        call = calls[0]
        arguments = json.loads(call.function.arguments)
        expect(call.id and call.type == 'function' and call.function.name == 'get_weather' and isinstance(arguments, dict), call.model_dump())
        follow = messages + [choice.message.model_dump(exclude_none=True),
                             {'role': 'tool', 'tool_call_id': call.id, 'content': weather(arguments.get('city', '?'))}]
        second = client.chat.completions.create(model=model, messages=follow, tools=[chat_tool], temperature=0, max_tokens=64)
        text = second.choices[0].message.content or ''
        expect(text.strip() and second.choices[0].finish_reason in ('stop', 'length') and not second.choices[0].message.tool_calls, second.model_dump())
        return {'quality': quality(arguments, text), 'final': text[:200], 'usage': first.usage.model_dump() if first.usage else None}

    def chat_tools_stream():
        client = openai_client()
        pieces, finish, chunks = {}, None, 0
        with client.chat.completions.create(model=model, messages=messages, tools=[chat_tool], tool_choice='required', temperature=0, max_tokens=96, stream=True) as stream:
            for chunk in stream:
                chunks += 1
                for choice in chunk.choices:
                    for call in choice.delta.tool_calls or []:
                        slot = pieces.setdefault(call.index, {'id': None, 'name': None, 'arguments': ''})
                        if call.id:
                            slot['id'] = call.id
                        if call.function and call.function.name:
                            slot['name'] = call.function.name
                        if call.function and call.function.arguments:
                            slot['arguments'] += call.function.arguments
                    if choice.finish_reason:
                        finish = choice.finish_reason
        expect(pieces and finish == 'tool_calls', {'pieces': pieces, 'finish': finish, 'chunks': chunks})
        slot = pieces[min(pieces)]
        arguments = json.loads(slot['arguments'])
        expect(slot['id'] and slot['name'] == 'get_weather' and isinstance(arguments, dict), slot)
        follow = messages + [{'role': 'assistant', 'content': None, 'tool_calls': [{'id': slot['id'], 'type': 'function', 'function': {'name': slot['name'], 'arguments': slot['arguments']}}]},
                             {'role': 'tool', 'tool_call_id': slot['id'], 'content': weather(arguments.get('city', '?'))}]
        text = ''
        with client.chat.completions.create(model=model, messages=follow, tools=[chat_tool], temperature=0, max_tokens=64, stream=True) as stream:
            for chunk in stream:
                for choice in chunk.choices:
                    text += choice.delta.content or ''
        expect(text.strip(), {'text': text})
        return {'quality': quality(arguments, text), 'final': text[:200], 'stream_chunks': chunks}

    def chat_tool_choice():
        client = openai_client()
        seen = {}
        for parallel in (True, False):
            raw = client.chat.completions.with_raw_response.create(
                model=model, messages=messages, tools=[chat_tool], tool_choice={'type': 'function', 'function': {'name': 'get_weather'}},
                parallel_tool_calls=parallel, temperature=0, max_tokens=96)
            completion = raw.parse()
            calls = completion.choices[0].message.tool_calls or []
            expect(len(calls) >= 1 and calls[0].function.name == 'get_weather', completion.model_dump())
            expect('x-titanos-dropped' not in {k.lower() for k in raw.headers} and 'x-titanos-ignored' not in {k.lower() for k in raw.headers}, dict(raw.headers))
            seen[str(parallel)] = json.loads(calls[0].function.arguments)
        return {'arguments_by_parallel_tool_calls': seen}

    def chat_json_schema():
        client = openai_client()
        schema = {'type': 'object', 'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}}, 'required': ['city', 'country'], 'additionalProperties': False}
        completion = client.chat.completions.create(
            model=model, messages=[{'role': 'user', 'content': 'Give the city Paris and its country as JSON.'}],
            response_format={'type': 'json_schema', 'json_schema': {'name': 'place', 'schema': schema, 'strict': True}}, temperature=0, max_tokens=64)
        text = completion.choices[0].message.content
        value = json.loads(text)
        expect(isinstance(value, dict) and set(value) == {'city', 'country'} and all(isinstance(v, str) for v in value.values()), {'text': text})
        return {'value': value}

    responses_tool = {'type': 'function', 'name': 'get_weather', 'description': 'Current weather for a city', 'parameters': WEATHER}

    def responses_tools():
        client = openai_client()
        first = client.responses.create(model=model, input=QUESTION, tools=[responses_tool], tool_choice='auto', temperature=0, max_output_tokens=96)
        calls = [item for item in first.output if item.type == 'function_call']
        expect(calls and first.status == 'completed', first.model_dump())
        call = calls[0]
        arguments = json.loads(call.arguments)
        expect(call.call_id and call.name == 'get_weather' and isinstance(arguments, dict), call.model_dump())
        items = [{'role': 'user', 'content': QUESTION}, call.model_dump(exclude_none=True),
                 {'type': 'function_call_output', 'call_id': call.call_id, 'output': weather(arguments.get('city', '?'))}]
        second = client.responses.create(model=model, input=items, tools=[responses_tool], temperature=0, max_output_tokens=64)
        text = second.output_text
        expect(text.strip() and not [item for item in second.output if item.type == 'function_call'], second.model_dump())
        return {'quality': quality(arguments, text), 'final': text[:200]}

    def responses_tools_stream():
        client = openai_client()
        kinds, added, arguments_text, done_args, final = [], None, '', None, None
        with client.responses.create(model=model, input=QUESTION, tools=[responses_tool], tool_choice='required', temperature=0, max_output_tokens=96, stream=True) as stream:
            for event in stream:
                kinds.append(event.type)
                if event.type == 'response.output_item.added' and event.item.type == 'function_call':
                    added = event.item
                elif event.type == 'response.function_call_arguments.delta':
                    arguments_text += event.delta
                elif event.type == 'response.function_call_arguments.done':
                    done_args = event.arguments
                elif event.type == 'response.completed':
                    final = event.response
        expect(added is not None and final is not None, {'kinds': kinds[:40]})
        arguments = json.loads(arguments_text or done_args or '{}')
        expect(added.name == 'get_weather' and isinstance(arguments, dict) and arguments, {'added': added.model_dump(), 'arguments_text': arguments_text, 'done': done_args})
        if done_args is not None:
            expect(json.loads(done_args) == arguments, {'arguments_text': arguments_text, 'done': done_args})
        call = [item for item in final.output if item.type == 'function_call']
        expect(call and call[0].call_id, final.model_dump())
        items = [{'role': 'user', 'content': QUESTION}, call[0].model_dump(exclude_none=True),
                 {'type': 'function_call_output', 'call_id': call[0].call_id, 'output': weather(arguments.get('city', '?'))}]
        text = ''
        with client.responses.create(model=model, input=items, tools=[responses_tool], temperature=0, max_output_tokens=64, stream=True) as stream:
            for event in stream:
                if event.type == 'response.output_text.delta':
                    text += event.delta
        expect(text.strip(), {'text': text})
        return {'quality': quality(arguments, text), 'final': text[:200], 'event_kinds': sorted(set(kinds))}

    anthropic_tool = {'name': 'get_weather', 'description': 'Current weather for a city', 'input_schema': WEATHER}
    SAMPLING = {'temperature': 0}

    def anthropic_client():
        if anthropic is None:
            raise Untried('the anthropic package is not importable here')
        return anthropic.Anthropic(base_url=base_url, api_key='local-titanos-placeholder', timeout=300, max_retries=0)

    def anthropic_tools():
        client = anthropic_client()
        # anthropic 1.x dropped the temperature keyword from messages.create; the stock extra_body carries it to the route
        first = client.messages.create(model=model, max_tokens=96, messages=messages, tools=[anthropic_tool], tool_choice={'type': 'any'}, extra_body=SAMPLING)
        uses = [block for block in first.content if block.type == 'tool_use']
        expect(uses and first.stop_reason == 'tool_use', first.model_dump())
        use = uses[0]
        expect(use.id and use.name == 'get_weather' and isinstance(use.input, dict), use.model_dump())
        follow = messages + [{'role': 'assistant', 'content': [block.model_dump(exclude_none=True) for block in first.content]},
                             {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': use.id, 'content': weather(use.input.get('city', '?'))}]}]
        second = client.messages.create(model=model, max_tokens=64, messages=follow, tools=[anthropic_tool], extra_body=SAMPLING)
        text = ''.join(block.text for block in second.content if block.type == 'text')
        expect(text.strip() and second.stop_reason in ('end_turn', 'max_tokens'), second.model_dump())
        return {'quality': quality(use.input, text), 'final': text[:200], 'usage': first.usage.model_dump()}

    def anthropic_tools_stream():
        client = anthropic_client()
        kinds = []
        with client.messages.stream(model=model, max_tokens=96, messages=messages, tools=[anthropic_tool], tool_choice={'type': 'any'}, extra_body=SAMPLING) as stream:
            for event in stream:
                kinds.append(event.type)
            first = stream.get_final_message()
        uses = [block for block in first.content if block.type == 'tool_use']
        expect(uses and first.stop_reason == 'tool_use' and 'input_json' in kinds, {'message': first.model_dump(), 'kinds': kinds[:40]})
        use = uses[0]
        expect(use.id and use.name == 'get_weather' and isinstance(use.input, dict) and use.input, use.model_dump())
        follow = messages + [{'role': 'assistant', 'content': [block.model_dump(exclude_none=True) for block in first.content]},
                             {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': use.id, 'content': weather(use.input.get('city', '?'))}]}]
        text = ''
        with client.messages.stream(model=model, max_tokens=64, messages=follow, tools=[anthropic_tool], extra_body=SAMPLING) as stream:
            for piece in stream.text_stream:
                text += piece
            second = stream.get_final_message()
        expect(text.strip() and second.stop_reason in ('end_turn', 'max_tokens'), second.model_dump())
        return {'quality': quality(use.input, text), 'final': text[:200], 'sdk_event_kinds': sorted(set(kinds)),
                'output_tokens': first.usage.output_tokens}

    def agents_case(use_responses):
        try:
            from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
            from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
        except ImportError as exc:
            raise Untried(f'the openai-agents package is not importable here ({exc})')
        if openai is None:
            raise Untried('the openai package is not importable here')
        set_tracing_disabled(True)  # no trace export to the hosted platform
        calls = []

        @function_tool
        def get_weather(city: str) -> str:
            """Current weather for a city."""
            calls.append(city)
            return weather(city)

        client = openai.AsyncOpenAI(base_url=base_url + '/v1', api_key='sk-local-titanos-placeholder', timeout=300, max_retries=0)
        model_object = (OpenAIResponsesModel if use_responses else OpenAIChatCompletionsModel)(model=model, openai_client=client)
        # instructions=None: with a system message in the prompt, llama.cpp b10809 with the Qwen2.5-0.5B template answered
        # text to tool_choice auto and required alike (probe 2026-10-05); the same request without a system message calls the
        # tool. A backend/template outcome, not a route one: Titanos forwards the body unchanged. tool_choice required is set
        # for the first turn; the framework resets it to auto after a tool call so the second turn answers in text.
        agent = Agent(name='weather-agent', instructions=None, tools=[get_weather], model=model_object,
                      model_settings=ModelSettings(temperature=0, max_tokens=96, tool_choice='required'))
        import asyncio
        result = asyncio.run(Runner.run(agent, QUESTION, max_turns=4))
        text = str(result.final_output or '')
        expect(calls and text.strip(), {'calls': calls, 'final': text, 'items': [type(item).__name__ for item in result.new_items]})
        return {'quality': quality({'city': calls[0]}, text), 'final': text[:200], 'tool_calls': calls,
                'run_items': [type(item).__name__ for item in result.new_items]}

    record('openai.chat.tools', chat_tools)
    record('openai.chat.tools.stream', chat_tools_stream)
    record('openai.chat.tool_choice', chat_tool_choice)
    record('openai.chat.json_schema', chat_json_schema)
    record('openai.responses.tools', responses_tools)
    record('openai.responses.tools.stream', responses_tools_stream)
    record('anthropic.messages.tools', anthropic_tools)
    record('anthropic.messages.tools.stream', anthropic_tools_stream)
    record('agents.chat_completions', lambda: agents_case(False))
    record('agents.responses', lambda: agents_case(True))
    report = {'python': sys.version.split()[0], 'sdk_versions': versions, 'cases': results}
    Path(report_path).write_text(json.dumps(report, indent=1), encoding='utf-8')
    return 0


# ----------------------------------------------------------------------------------------------------------------------
# server side


BIND_REFUSED = 'error while attempting to bind'  # asyncio's OSError text, which uvicorn logs before it exits


def free_port(port_range):
    if port_range:
        low, high = (int(v) for v in port_range.split('-'))
        for candidate in range(low, high + 1):
            with socket.socket() as sock:
                try:
                    sock.bind(('127.0.0.1', candidate))
                except OSError:
                    continue
                return candidate
        raise SystemExit(f'no free port in {port_range}')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--client', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--base-url', help=argparse.SUPPRESS)
    parser.add_argument('--model', default='agent', help='registered model name (default agent)')
    parser.add_argument('--llama-server', type=Path, default=os.environ.get('TITANOS_LLAMA_SERVER'), help='llama-server executable (llama.cpp)')
    parser.add_argument('--gguf', type=Path, default=os.environ.get('TITANOS_AGENT_GGUF'), help='GGUF chat model with a tool-calling template (Qwen2.5-0.5B-Instruct)')
    parser.add_argument('--sdk-python', default=os.environ.get('TITANOS_SDK_PYTHON_AGENTS') or sys.executable, help='interpreter with openai, anthropic and openai-agents')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--port-range', default=None, help='lo-hi: pick the first free loopback port in the range')
    parser.add_argument('--report', type=Path, default=None)
    parser.add_argument('--strict', action='store_true', help='untried cases also fail')
    args = parser.parse_args()
    if args.client:
        raise SystemExit(client_main(args.base_url, args.model, args.report))
    if args.llama_server is None or args.gguf is None:
        raise SystemExit('--llama-server and --gguf are required (manual tier: a llama.cpp binary and a GGUF)')
    if not args.llama_server.is_file() or not args.gguf.is_file():
        raise SystemExit(f'missing input: {args.llama_server} / {args.gguf}')

    import tempfile

    import httpx

    from titanos.config import Config, ModelSpec, NativeOptions, write_config

    spec = ModelSpec(path=str(args.gguf.parent), task='chat', backend='llama-cpp', device='cpu',
                     native=NativeOptions(executable=str(args.llama_server), model_file=args.gguf.name, context_size=4096,
                                          slots=2, threads=args.threads, gpu_layers=0))
    report = {'lane': 'agent_loops', 'model': spec.model_dump(), 'cases': [], 'success': False}
    with tempfile.TemporaryDirectory(prefix='agent-loops-') as tmp:
        root = Path(tmp)
        registry = root / 'registry.json'
        write_config(registry, Config(models={args.model: spec}))
        port = args.port or free_port(args.port_range)
        url = f'http://127.0.0.1:{port}'
        env = {**os.environ, 'TITANOS_HOME': str(root / 'home'), 'HF_HUB_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '',
               'PYTHONPATH': os.pathsep.join(filter(None, [str(ROOT / 'src'), os.environ.get('PYTHONPATH')])), 'PYTHONIOENCODING': 'utf-8'}
        env.pop('TITANOS_API_KEY', None)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0

        def launch(port):
            log = (root / 'server.log').open('wb')
            proc = subprocess.Popen([sys.executable, '-B', '-m', 'titanos', '--config', str(registry), 'serve', '--host', '127.0.0.1', '--port', str(port)],
                                    cwd=tmp, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, start_new_session=sys.platform != 'win32')
            return proc, log
        started = time.monotonic()
        proc, log = launch(port)
        # a chosen port another socket took after free_port closed its probe (uvicorn's bind error) moves to a fresh one
        rebinds = 0 if args.port else 2
        client_report = root / 'client.json'
        try:
            with httpx.Client(timeout=10, trust_env=False) as c:
                while True:
                    try:
                        if c.get(url + '/healthz').status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if proc.poll() is not None and rebinds and BIND_REFUSED in (root / 'server.log').read_text(errors='replace'):
                        log.close()
                        rebinds -= 1
                        port = free_port(args.port_range)
                        url = f'http://127.0.0.1:{port}'
                        proc, log = launch(port)
                        continue
                    if proc.poll() is not None or time.monotonic() - started > 120:
                        log.close()
                        raise SystemExit('titanos serve did not start:\n' + (root / 'server.log').read_text(errors='replace')[-3000:])
                    time.sleep(.2)
                info = c.get(url + '/api/info').json()
                report['server'] = {'url': url, 'version': info.get('version'), 'port': port}
            command = [args.sdk_python, '-B', str(Path(__file__).resolve()), '--client', '--base-url', url, '--model', args.model, '--report', str(client_report)]
            client_env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
            client_env['OPENAI_API_KEY'] = 'sk-local-titanos-placeholder'
            run = subprocess.run(command, env=client_env, capture_output=True, text=True, timeout=1800)
            report['client'] = {'python': args.sdk_python, 'exit_code': run.returncode, 'stderr_tail': run.stderr[-2000:]}
            if client_report.exists():
                report.update(json.loads(client_report.read_text(encoding='utf-8')))
            else:
                report['cases'] = [{'id': 'client', 'state': 'fail', 'detail': 'the client wrote no report: ' + run.stderr[-1500:]}]
            with httpx.Client(base_url=url, timeout=10, trust_env=False) as c:
                workers = c.get('/api/workers').json().get('data', [])
                report['server']['native_workers'] = [{k: w.get(k) for k in ('id', 'running', 'loaded', 'completed_requests', 'last_error', 'resolved_device')} for w in workers]
        finally:
            if proc.poll() is None:
                import signal
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == 'win32' else signal.SIGINT)
                    proc.wait(60)
                except Exception:
                    proc.kill()
                    proc.wait(60)
            report['server_exit_code'] = proc.returncode
            log.close()
            report['server_log_tail'] = (root / 'server.log').read_text(errors='replace')[-1500:]
    states = {case['state'] for case in report['cases']}
    report['counts'] = {state: sum(case['state'] == state for case in report['cases']) for state in ('pass', 'fail', 'untried')}
    report['success'] = 'fail' not in states and (not args.strict or 'untried' not in states)
    report['model_quality'] = {case['id']: case.get('quality') for case in report['cases'] if case.get('quality')}
    text = json.dumps(report, indent=1, default=str)
    if args.report:
        args.report.write_text(text, encoding='utf-8')
    print(text)
    raise SystemExit(0 if report['success'] else 1)


if __name__ == '__main__':
    main()
