"""The stock Ollama Python client (pip install ollama) against a real `titanos serve` in keyless loopback mode.

Server side (this interpreter, src importable): a temporary TITANOS_HOME and registry, `python -m titanos --config <tmp>/models.json
serve` on a free loopback port in 49900..49949, stopped at the end. The registry always holds `registry-only`, a chat entry over a
directory with a config.json and no weights, which /api/show describes without loading. TITANOS_SDK_FIXTURE_MODELS (the
tests/check_broker_sdk.py convention: chat/ and embeddings/ directories with a config.json) adds `local-chat` and `local-embed` on
CPU, and TITANOS_OLLAMA_SDK_GGUF plus TITANOS_OLLAMA_SDK_LLAMA_SERVER (a GGUF file and a llama-server executable) add `local-gguf`,
a llama-cpp model with native.context_size 2048 on one CPU slot.

Client side (TITANOS_SDK_PYTHON_OLLAMA, then TITANOS_SDK_PYTHON_EXT, default this interpreter; imports only `ollama` and the standard
library): version, list, ps, show, the refused library routes (pull, create, copy, delete), and with the optional models chat,
generate, embed and streaming with keep_alive (a duration, -1 and 0, whose unload /api/ps confirms), options.num_ctx (admitted
within the native per-slot context, 400 above it) and, on the GGUF model, the tool loop in the client's own shape (an id-less
ToolCall answered with a `role: tool` + `tool_name` message; a result that answers no call is 400), nonstream and with stream=True
(the call arrives as one whole ToolCall with dict arguments, not as argument fragments). A missing `ollama` package
reports untried (exit 0), never a pass; every case
that ran must pass. Nothing leaves loopback; no provider is called."""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT_RANGE = range(49900, 49950)
BIND_REFUSED = 'error while attempting to bind'  # asyncio's OSError text, which uvicorn logs before it exits
FIXTURE_ENV = 'TITANOS_SDK_FIXTURE_MODELS'
GGUF_ENV = 'TITANOS_OLLAMA_SDK_GGUF'
LLAMA_ENV = 'TITANOS_OLLAMA_SDK_LLAMA_SERVER'

CLIENT = r'''
import json, os, sys, urllib.request
report = {'cases': {}, 'untried': {}}
try:
    import ollama
    from ollama import Client, ResponseError
except ImportError as exc:
    print(json.dumps({'untried': {'all': 'ollama is not installed: ' + str(exc)}, 'cases': {}}))
    sys.exit(0)
from importlib.metadata import version
report['ollama'] = version('ollama')
url = os.environ['TITANOS_OLLAMA_SDK_URL']
models = json.loads(os.environ['TITANOS_OLLAMA_SDK_MODELS'])
client = Client(host=url, timeout=600)


def case(name, fn):
    try:
        fn()
        report['cases'][name] = 'pass'
    except Exception as exc:  # reported per case; the orchestrator turns it into a failure
        report['cases'][name] = 'fail: ' + type(exc).__name__ + ': ' + str(exc)[:400]


def refused(call, field, *needles):
    try:
        call()
    except ResponseError as exc:
        text = json.dumps(exc.error) if not isinstance(exc.error, str) else exc.error
        assert exc.status_code == 400, (exc.status_code, text)
        err = exc.error if isinstance(exc.error, dict) else {}
        assert err.get('code') == 'unsupported_capability' and err.get('param') == field, text
        for needle in needles:
            assert needle in err.get('message', ''), (needle, text)
        return
    raise AssertionError('the library route was not refused')


def loaded_names():
    return {m.model or m.name for m in client.ps().models}


def version_route():
    with urllib.request.urlopen(url + '/api/version', timeout=30) as reply:
        doc = json.loads(reply.read())
    assert doc['version'] and doc['name'] == 'Titanos' and doc['role'] == 'standalone', doc


def list_route():
    names = {m.model for m in client.list().models}
    assert {'registry-only', *models}.issubset(names), names


def ps_route():
    assert 'registry-only' not in loaded_names()


def show_route():
    doc = client.show('registry-only')
    assert doc.details.format == 'transformers' and doc.details.family == 'chat' and doc.details.families == ['chat'], doc.details
    assert doc.capabilities == ['completion'] and doc.modelinfo['titanos.task'] == 'chat' and doc.modelinfo['general.architecture'] == 'LlamaForCausalLM', doc.modelinfo
    assert doc.template == '' and '/api/models/register' in doc.modelfile and doc.parameters == 'num_predict 32', (doc.template, doc.parameters)
    assert 'registry-only' not in loaded_names(), 'show must not load'
    try:
        client.show('no-such-model')
    except ResponseError as exc:
        assert exc.status_code == 404, exc.status_code
    else:
        raise AssertionError('unknown model was shown')


def pull_route():
    refused(lambda: client.pull('llama3'), 'model', '/api/pull', 'POST /api/acquisitions', 'POST /api/models/register')
    refused(lambda: list(client.pull('llama3', stream=True)), 'model', '/api/pull')


def create_route():
    refused(lambda: client.create(model='derived', from_='registry-only'), 'model', '/api/create', 'POST /api/models/register')


def copy_route():
    refused(lambda: client.copy('registry-only', 'copy-of'), 'destination', '/api/copy', 'POST /api/models/register')


def delete_route():
    refused(lambda: client.delete('registry-only'), 'model', '/api/delete', 'DELETE /api/models/registrations/{name}')
    assert 'registry-only' in {m.model for m in client.list().models}, 'nothing may be deleted'


def chat_fixture():
    name = models['local-chat']
    answer = client.chat(model=name, messages=[{'role': 'user', 'content': 'Say hello.'}], options={'num_ctx': 2048, 'num_predict': 8, 'temperature': 0}, keep_alive='5m')
    assert answer.done and isinstance(answer.message.content, str) and answer.eval_count >= 1, answer
    assert name in loaded_names(), 'keep_alive 5m keeps the model resident (no timer)'
    text = client.generate(model=name, prompt='Hello', options={'num_predict': 4, 'temperature': 0}, keep_alive=-1)
    assert text.done and isinstance(text.response, str), text
    assert name in loaded_names()
    parts = list(client.chat(model=name, messages=[{'role': 'user', 'content': 'Say hello.'}], options={'num_predict': 6, 'temperature': 0}, stream=True, keep_alive=0))
    assert parts and parts[-1].done and all(isinstance(p.message.content, str) for p in parts), parts[-1]
    assert name not in loaded_names(), 'keep_alive 0 unloads after the last frame'


def embed_fixture():
    name = models['local-embed']
    vectors = client.embed(model=name, input=['one', 'two'], keep_alive=0)
    assert len(vectors.embeddings) == 2 and len(vectors.embeddings[0]) > 1, vectors
    assert name not in loaded_names(), 'keep_alive 0 unloads after the answer'
    legacy = client.embeddings(model=name, prompt='one', keep_alive='10m')
    assert len(legacy.embedding) == len(vectors.embeddings[0]), legacy
    assert name in loaded_names(), 'keep_alive 10m: resident, no timer'


def native_gguf():
    name = models['local-gguf']
    doc = client.show(name)
    assert doc.details.format == 'llama-cpp' and doc.parameters.startswith('num_ctx 2048') and 'tools' in doc.capabilities, (doc.details, doc.parameters, doc.capabilities)
    answer = client.chat(model=name, messages=[{'role': 'user', 'content': 'Say hello.'}], options={'num_ctx': 1024, 'num_predict': 8, 'temperature': 0}, keep_alive='1h')
    assert answer.done and isinstance(answer.message.content, str) and answer.eval_count >= 1, answer
    assert name in loaded_names()
    try:
        client.chat(model=name, messages=[{'role': 'user', 'content': 'Hi'}], options={'num_ctx': 100000})
    except ResponseError as exc:
        err = exc.error if isinstance(exc.error, dict) else {}
        assert exc.status_code == 400 and err.get('code') == 'context_limit' and 'options.num_ctx 100000' in err.get('message', ''), (exc.status_code, exc.error)
    else:
        raise AssertionError('num_ctx above the native context was not refused')
    text = client.generate(model=name, prompt='Hello', options={'num_predict': 4, 'temperature': 0}, keep_alive=0)
    assert text.done and isinstance(text.response, str), text
    assert name not in loaded_names(), 'keep_alive 0 stops the native worker after the answer'


def native_tool_loop():
    # the tool loop in the stock client's own shape: the ToolCall it receives has no id and the result it can send carries tool_name
    name = models['local-gguf']
    tool = {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
                                             'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}
    history = [{'role': 'user', 'content': 'What is the weather in Paris? Call get_weather.'}]
    first = client.chat(model=name, messages=history, tools=[tool], options={'num_predict': 48, 'temperature': 0}, keep_alive='5m')
    calls = first.message.tool_calls or []
    assert first.done_reason == 'tool_calls' and len(calls) == 1 and calls[0].function.name == 'get_weather', (first.done_reason, first.message)
    assert isinstance(calls[0].function.arguments, dict) and 'city' in calls[0].function.arguments, calls[0].function.arguments
    follow = client.chat(model=name, messages=[*history, first.message, {'role': 'tool', 'content': 'Sunny, 22 C in Paris', 'tool_name': 'get_weather'}],
                         tools=[tool], options={'num_predict': 48, 'temperature': 0}, keep_alive=0)
    assert follow.done and isinstance(follow.message.content, str) and follow.message.content.strip(), follow.message
    try:
        client.chat(model=name, messages=[*history, {'role': 'tool', 'content': 'Sunny', 'tool_name': 'get_weather'}], tools=[tool], options={'num_predict': 8})
    except ResponseError as exc:
        err = exc.error if isinstance(exc.error, dict) else {}
        assert exc.status_code == 400 and err.get('message', '').startswith("messages[1]: the tool result for tool_name 'get_weather'"), (exc.status_code, exc.error)
    else:
        raise AssertionError('a tool result that answers no tool call was accepted')


def native_tool_loop_stream():
    # the same first turn with stream=True (the default of Ollama UIs and agents): the client validates every frame, so the call must
    # arrive as one whole ToolCall with dict arguments, never as the OpenAI argument fragments llama-server streams
    name = models['local-gguf']
    tool = {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Current weather for a city',
                                             'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}
    parts = list(client.chat(model=name, messages=[{'role': 'user', 'content': 'What is the weather in Paris? Call get_weather.'}], tools=[tool],
                             options={'num_predict': 48, 'temperature': 0}, keep_alive=0, stream=True))
    calls = [part.message.tool_calls for part in parts if part.message.tool_calls]
    assert parts and parts[-1].done and parts[-1].done_reason == 'tool_calls', (len(parts), parts[-1].done if parts else None, parts[-1].done_reason if parts else None)
    assert len(calls) == 1 and len(calls[0]) == 1 and calls[0][0].function.name == 'get_weather', calls
    assert isinstance(calls[0][0].function.arguments, dict) and 'city' in calls[0][0].function.arguments, calls[0][0].function.arguments
    assert all(isinstance(part.message.content, str) for part in parts)


for label, fn in (('version', version_route), ('list', list_route), ('ps', ps_route), ('show', show_route), ('pull', pull_route),
                  ('create', create_route), ('copy', copy_route), ('delete', delete_route)):
    case(label, fn)
for label, key, fn in (('chat+generate+stream keep_alive (fixture)', 'local-chat', chat_fixture), ('embed keep_alive (fixture)', 'local-embed', embed_fixture),
                       ('llama-cpp num_ctx/keep_alive (gguf)', 'local-gguf', native_gguf), ('llama-cpp tool loop (gguf)', 'local-gguf', native_tool_loop),
                       ('llama-cpp streaming tool loop (gguf)', 'local-gguf', native_tool_loop_stream)):
    if key in models:
        case(label, fn)
    else:
        report['untried'][label] = key + ' is not registered (set ' + ('TITANOS_SDK_FIXTURE_MODELS' if key != 'local-gguf' else 'TITANOS_OLLAMA_SDK_GGUF and TITANOS_OLLAMA_SDK_LLAMA_SERVER') + ')'
print(json.dumps(report))
'''


def scrubbed_env() -> dict:
    out = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper.startswith('TITANOS_') or upper.endswith('_BASE_URL') or 'PROXY' in upper or 'API_KEY' in upper or 'TOKEN' in upper \
                or 'SECRET' in upper or upper in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
            continue
        out[name] = value
    return out


def free_port() -> int:
    for port in PORT_RANGE:
        sock = socket.socket()
        try:
            sock.bind(('127.0.0.1', port))
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError(f'no free loopback port in {PORT_RANGE.start}..{PORT_RANGE.stop - 1}')


def registry(root: Path) -> dict:
    only = root / 'registry-only'
    only.mkdir()
    (only / 'config.json').write_text(json.dumps({'architectures': ['LlamaForCausalLM'], 'model_type': 'llama'}), encoding='utf-8')
    models = {'registry-only': {'path': str(only), 'task': 'chat', 'defaults': {'max_tokens': 32}}}
    base = os.environ.get(FIXTURE_ENV, '').strip()
    if base:
        for name, folder, task in (('local-chat', 'chat', 'chat'), ('local-embed', 'embeddings', 'embeddings')):
            if (Path(base) / folder / 'config.json').is_file():
                models[name] = {'path': str(Path(base) / folder), 'task': task, 'device': 'cpu'}
    gguf, server = os.environ.get(GGUF_ENV, '').strip(), os.environ.get(LLAMA_ENV, '').strip()
    if gguf and server and Path(gguf).is_file() and Path(server).is_file():
        models['local-gguf'] = {'path': str(Path(gguf).parent), 'task': 'chat', 'backend': 'llama-cpp', 'device': 'cpu',
                                'native': {'executable': server, 'model_file': Path(gguf).name, 'context_size': 2048, 'slots': 1, 'gpu_layers': 0, 'threads': 4}}
    return models


class Serve:
    def __init__(self, root: Path, models: dict):
        self.root, self.models, self.process = root, models, None
        self.home = root / 'home'
        self.home.mkdir()
        self.log_path = root / 'serve.log'
        (root / 'models.json').write_text(json.dumps({'models': models}, indent=2), encoding='utf-8')

    def env(self) -> dict:
        env = scrubbed_env()
        env.update({'TITANOS_HOME': str(self.home), 'TITANOS_BROKER_EGRESS': 'loopback-only', 'PYTHONPATH': str(ROOT / 'src'),
                    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                    'HF_HUB_DISABLE_TELEMETRY': '1', 'CUDA_VISIBLE_DEVICES': ''})
        return env

    def start(self, attempts: int = 3) -> str:
        """serve on a free port; a port another socket took after free_port closed its probe (uvicorn's bind error)
        moves to a fresh port, up to attempts times."""
        import urllib.request
        import urllib.error
        self.port = free_port()
        url = f'http://127.0.0.1:{self.port}'
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        self._log = open(self.log_path, 'wb')
        self.process = subprocess.Popen([sys.executable, '-B', '-m', 'titanos', '--config', str(self.root / 'models.json'), 'serve', '--host', '127.0.0.1',
                                         '--port', str(self.port), '--log-level', 'warning'], cwd=str(self.root), env=self.env(), stdout=self._log,
                                        stderr=subprocess.STDOUT, creationflags=flags, start_new_session=sys.platform != 'win32')
        deadline = time.monotonic() + 180
        while True:
            try:
                with urllib.request.urlopen(url + '/healthz', timeout=5) as reply:
                    if reply.status == 200:
                        return url
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if self.process.poll() is not None and attempts > 1 and BIND_REFUSED in self.log():
                self._log.close()
                return self.start(attempts - 1)
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('titanos serve did not start: ' + self.log()[-1500:])
            time.sleep(0.2)

    def log(self) -> str:
        try:
            return self.log_path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return ''

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == 'win32' else signal.SIGINT)
                self.process.wait(timeout=60)
            except (subprocess.TimeoutExpired, OSError, ValueError):
                self.process.kill()
                self.process.wait(timeout=30)
        self._log.close()


def run_client(url: str, models: dict) -> dict:
    python = os.environ.get('TITANOS_SDK_PYTHON_OLLAMA') or os.environ.get('TITANOS_SDK_PYTHON_EXT') or sys.executable
    env = scrubbed_env()
    env.update({'TITANOS_OLLAMA_SDK_URL': url, 'TITANOS_OLLAMA_SDK_MODELS': json.dumps({k: k for k in models if k != 'registry-only'}),
                'NO_PROXY': '127.0.0.1,localhost', 'PYTHONIOENCODING': 'utf-8'})
    result = subprocess.run([python, '-c', CLIENT], env=env, capture_output=True, text=True, encoding='utf-8', timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f'client failed ({result.returncode}): {result.stderr[-2000:]}')
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix='titanos-ollama-sdk-'))
    try:
        models = registry(root)
        serve = Serve(root, models)
        try:
            url = serve.start()
            report = run_client(url, models)
        finally:
            serve.stop()
        failures = {k: v for k, v in report['cases'].items() if v != 'pass'}
        out = {'passed': not failures and bool(report['cases']), 'ollama': report.get('ollama'), 'models': sorted(models), 'port': serve.port,
               'cases': report['cases'], 'untried': report['untried'],
               'scope': 'stock ollama client against a real keyless serve; the in-process surface is tests/check_ollama_compat.py'}
        if not report['cases']:
            out['passed'] = None
            print(json.dumps(out))
            return 0
        print(json.dumps(out))
        if failures:
            sys.stderr.write(serve.log()[-3000:])
            return 1
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
