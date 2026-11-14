"""Real interop with NVIDIA NeMo Switchyard (github.com/NVIDIA-NeMo/Switchyard, Apache-2.0), both directions, on loopback
only: a REAL switchyard-server (the native Rust server of the nemo-switchyard package, started through its Python host
switchyard_rust.server.Server, which binds 127.0.0.1) and a REAL `python -m titanos serve` subprocess (the harness of
tests/check_broker_sdk.py: temporary TITANOS_HOME, the WP4 loopback mocks, the netguard sitecustomize, a bootstrap key,
minted tk_ keys and env: credential references to synthetic sentinels). No provider key exists anywhere in this lane.

Usage (manual tier; never in CI):
    python tests/check_switchyard_interop.py --switchyard-python /path/to/switchyard-venv/python \
        --fixture-models /path/to/fixture/models [--strict]
--switchyard-python (or TITANOS_SWITCHYARD_PYTHON) names an interpreter where `pip install nemo-switchyard` succeeded
(a separate venv; never the project venv). --fixture-models (or TITANOS_SDK_FIXTURE_MODELS) names a directory whose chat/
folder holds a small CPU transformers chat checkpoint (served as the local model `local-chat`). Without the Switchyard
interpreter every case is untried; without the chat fixture the cases that need a local model are untried. Untried is
never a pass; --strict exits 1 on any untried case.

The Switchyard deployment (TOML, written into the run directory) targets Titanos only:
- titanos_local (openai_chat), titanos_responses (openai_responses) and titanos_messages (anthropic_messages): Titanos'
  local model through each of its three inbound shapes, with extra_headers X-Titanos-Hop = "switchyard" so Titanos keeps
  those requests local (its 1-hop rule) and a loop through Switchyard is refused instead of repeated;
- titanos_cloud (openai_chat, no hop header): Titanos' brokered cloud ref gpt-4o-mini@openai, which Titanos sends to the
  loopback OpenAI mock with the mock's synthetic sentinel (Titanos' own credential binding; Switchyard holds only a
  Titanos tk_ key, through api_key_env);
- routes: switchyard (noop), titanos-local, titanos-local-responses, titanos-local-messages, titanos-cloud and
  titanos-loop (whose target is Titanos' own model titanos-loop@sy, i.e. back through Switchyard) and titanos-loop-cloud
  (the same loop through the cloud client, which carries no hop header).
Before Switchyard starts, the harness asserts that every base_url in the TOML is http://127.0.0.1 (the Rust process is
outside the Python netguard) and gives it a scrubbed environment holding only the Titanos key it needs.

Titanos' broker.json gains the upstream `sy` of profile switchyard (base_url http://127.0.0.1:<port>, no credential,
catalog live), so `<route>@sy` names a Switchyard route.

Cases:
(b) Switchyard -> Titanos: OpenAI chat (plain and streamed) to the local model; Anthropic Messages into Switchyard
    (Claude Code shape with metadata.user_id), translated by Switchyard onto Titanos' chat route; Anthropic Messages to
    Titanos' own /v1/messages; Codex-shaped Responses (store false, include, reasoning, prompt_cache_key) to Titanos'
    /v1/responses; OpenAI chat and Anthropic Messages to the brokered cloud ref (the OpenAI mock receives exactly its
    sentinel, never the tk_ key). Each answer names the selected backend in x-model-router-selected-model.
(a) Titanos -> Switchyard: the live catalog lists the route ids as <route>@sy with Switchyard's capabilities; OpenAI chat,
    Anthropic Messages (passed through natively) and Responses to switchyard@sy; chat to titanos-local@sy (Titanos ->
    Switchyard -> Titanos local) and titanos-cloud@sy; the loop titanos-loop@sy ends with Titanos' 400
    location_unsatisfiable after exactly one pass through Switchyard; the hop-less loop titanos-loop-cloud@sy is refused
    on its first re-entry (Titanos recognizes its own X-Request-ID heading to a router again: 508 loop_detected) in
    seconds, with no provider call. A Codex-shaped answer that ends at its 16-token cap ('incomplete',
    max_output_tokens) counts as served.
Lane checks: zero non-loopback connection attempts (harness and server), every mock hit carries exactly its own sentinel,
no secret value in any output, the TOML loopback assertion, and a clean shutdown of both servers.
Output: one JSON document on stdout. Exit 1 on any fail (or, with --strict, any untried).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests'
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(ROOT / 'src'))

PYTHON_ENV = 'TITANOS_SWITCHYARD_PYTHON'
FIXTURE_ENV = 'TITANOS_SDK_FIXTURE_MODELS'
KEY_ENV = 'TITANOS_SY_KEY'
HOP_VALUE = 'switchyard'
CHAT = [{'role': 'user', 'content': 'Say hello in three words.'}]

RUNNER = r'''
import sys, time, pathlib
from switchyard_rust.server import Server
config, port, stop = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
with Server(config, port=port) as server:
    print('READY ' + server.base_url, flush=True)
    while not stop.exists():
        time.sleep(0.05)
print('STOPPED', flush=True)
'''

TOML = '''schema_version = 1

[llm_clients.titanos_local]
format = "openai_chat"
base_url = "{titanos}/v1"
api_key_env = "{key_env}"
extra_headers = {{ "X-Titanos-Hop" = "{hop}" }}
max_retries = 0
timeout_ms = 300000

[llm_clients.titanos_responses]
format = "openai_responses"
base_url = "{titanos}/v1"
api_key_env = "{key_env}"
extra_headers = {{ "X-Titanos-Hop" = "{hop}" }}
max_retries = 0
timeout_ms = 300000

[llm_clients.titanos_messages]
format = "anthropic_messages"
base_url = "{titanos}"
api_key_env = "{key_env}"
extra_headers = {{ "X-Titanos-Hop" = "{hop}" }}
max_retries = 0
timeout_ms = 300000

[llm_clients.titanos_cloud]
format = "openai_chat"
base_url = "{titanos}/v1"
api_key_env = "{key_env}"
max_retries = 0
timeout_ms = 120000

[targets.local_chat]
id = "local-chat"
llm_client = "titanos_local"

[targets.local_responses]
id = "local-chat"
llm_client = "titanos_responses"

[targets.local_messages]
id = "local-chat"
llm_client = "titanos_messages"

[targets.cloud_mini]
id = "gpt-4o-mini@openai"
llm_client = "titanos_cloud"

[targets.loop_back]
id = "titanos-loop@sy"
llm_client = "titanos_local"

[targets.loop_back_cloud]
id = "titanos-loop-cloud@sy"
llm_client = "titanos_cloud"

[routes.smoke]
id = "switchyard"
type = "noop"

[routes.local]
id = "titanos-local"
type = "passthrough"
target = "local_chat"
tool_calling = true
vision = false
context_window = 8192

[routes.local_responses]
id = "titanos-local-responses"
type = "passthrough"
target = "local_responses"

[routes.local_messages]
id = "titanos-local-messages"
type = "passthrough"
target = "local_messages"

[routes.cloud]
id = "titanos-cloud"
type = "passthrough"
target = "cloud_mini"

[routes.loop]
id = "titanos-loop"
type = "passthrough"
target = "loop_back"

[routes.loop_cloud]
id = "titanos-loop-cloud"
type = "passthrough"
target = "loop_back_cloud"
'''


class Untried(Exception):
    pass


def case(cases: list, case_id: str, fn, *, needs_local: bool = False, local: bool = True) -> None:
    if needs_local and not local:
        cases.append({'id': case_id, 'state': 'untried', 'detail': 'no chat fixture model (--fixture-models)'})
        return
    try:
        detail = fn()
        cases.append({'id': case_id, 'state': 'pass', 'detail': detail})
    except Untried as exc:
        cases.append({'id': case_id, 'state': 'untried', 'detail': str(exc)})
    except AssertionError as exc:
        cases.append({'id': case_id, 'state': 'fail', 'detail': str(exc)[:600]})
    except Exception as exc:  # a harness or transport error is a failed case, never a pass
        cases.append({'id': case_id, 'state': 'fail', 'detail': f'{type(exc).__name__}: {str(exc)[:500]}'})


def expect(condition, message) -> None:
    if not condition:
        raise AssertionError(message)


def loopback_only(toml_text: str) -> list[str]:
    """Every base_url of the deployment, asserted to be http://127.0.0.1:<port>[/...]."""
    urls = re.findall(r'base_url\s*=\s*"([^"]+)"', toml_text)
    bad = [url for url in urls if not re.fullmatch(r'http://127\.0\.0\.1:\d+(/.*)?', url)]
    if bad or not urls:
        raise AssertionError(f'non-loopback base_url in the Switchyard deployment: {bad}')
    return urls


def switchyard_python(argv: list[str]) -> str | None:
    if '--switchyard-python' in argv:
        return argv[argv.index('--switchyard-python') + 1]
    return os.environ.get(PYTHON_ENV, '').strip() or None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    strict = '--strict' in argv
    if '--fixture-models' in argv:
        os.environ[FIXTURE_ENV] = argv[argv.index('--fixture-models') + 1]
    python = switchyard_python(argv)
    probe = None
    if python:
        from check_broker_sdk import _scrubbed_env
        probe = subprocess.run([python, '-c', 'import switchyard_rust.server, importlib.metadata as m; '
                                'print(m.version("nemo-switchyard"))'], capture_output=True, text=True,
                               env=_scrubbed_env(), timeout=120)
    if not python or probe is None or probe.returncode:
        detail = (f'{PYTHON_ENV} / --switchyard-python names no interpreter with nemo-switchyard'
                  if not python else 'switchyard_rust is not importable by the named interpreter')
        cases = [{'id': 'switchyard.interop', 'state': 'untried', 'detail': detail}]
        print(json.dumps({'lane': 'switchyard-interop', 'result': 'untried', 'counts': {'untried': 1},
                          'cases': cases}, indent=1))
        return 1 if strict else 0
    version = probe.stdout.strip()
    return run(python, version, strict)


def run(python: str, version: str, strict: bool) -> int:
    import httpx
    from check_broker_sdk import Stack, _port, _scrubbed_env

    sy_port = _port()

    class InteropStack(Stack):
        def _write_home(self):
            super()._write_home()
            from titanos.broker.config import BrokerConfig, read_broker_config, write_broker_config
            cfg = read_broker_config(self.home / 'broker.json')
            document = cfg.model_dump(mode='json', exclude_unset=True)
            document['upstreams']['sy'] = {'profile': 'switchyard', 'base_url': f'http://127.0.0.1:{sy_port}',
                                           'enabled': True, 'catalog': 'live', 'label': 'NeMo Switchyard (local)'}
            write_broker_config(self.home / 'broker.json', BrokerConfig.model_validate(document))

    cases: list = []
    checks: list = []
    stack = InteropStack()
    sy_proc = None
    sy_log = None
    stop_file = None
    try:
        stack.start()
        local = 'local-chat' in stack.local_specs
        token = stack.tokens['sdk']
        titanos = stack.url
        toml_text = TOML.format(titanos=titanos, key_env=KEY_ENV, hop=HOP_VALUE)
        urls = loopback_only(toml_text)
        checks.append({'id': 'switchyard.toml_loopback_only', 'state': 'pass', 'detail': sorted(set(urls))})
        config = stack.root / 'titanos-targets.toml'
        config.write_text(toml_text, encoding='utf-8')
        stop_file = stack.root / 'switchyard.stop'
        env = _scrubbed_env()
        env[KEY_ENV] = token
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['RUST_LOG'] = 'warn'
        sy_log = open(stack.root / 'switchyard.log', 'wb')
        sy_proc = subprocess.Popen([python, '-c', RUNNER, str(config), str(sy_port), str(stop_file)],
                                   cwd=str(stack.root), env=env, stdout=subprocess.PIPE, stderr=sy_log)
        ready = sy_proc.stdout.readline().decode('utf-8', 'replace').strip()
        if not ready.startswith('READY '):
            raise RuntimeError('switchyard-server did not start: ' + ready + (stack.root / 'switchyard.log').read_text(
                encoding='utf-8', errors='replace')[-1500:])
        sy = ready.split(' ', 1)[1].rstrip('/')
        expect(sy == f'http://127.0.0.1:{sy_port}', f'switchyard bound {sy}')
        boot = {'Authorization': 'Bearer ' + stack.BOOT}
        user = {'Authorization': 'Bearer ' + token}
        timeout = httpx.Timeout(600.0, connect=10.0)
        with httpx.Client(trust_env=False, timeout=timeout) as http:
            refreshed = http.get(titanos + '/api/broker/catalog?refresh=true', headers=boot)
            expect(refreshed.status_code == 200, f'catalog refresh {refreshed.status_code} {refreshed.text[:300]}')
            openai_mock = stack.mocks['openai']

            # ---- (b) Switchyard -> Titanos ---------------------------------------------------------------------
            def b_chat():
                answer = http.post(sy + '/v1/chat/completions', json={'model': 'titanos-local', 'messages': CHAT,
                                                                      'max_tokens': 16, 'temperature': 0})
                expect(answer.status_code == 200, f'{answer.status_code} {answer.text[:400]}')
                body = answer.json()
                text = body['choices'][0]['message']['content']
                expect(isinstance(text, str) and text.strip(), 'empty completion')
                expect(answer.headers.get('x-model-router-selected-model') == 'local-chat', dict(answer.headers))
                return {'selected': 'local-chat', 'completion_chars': len(text), 'usage': body.get('usage')}

            def b_chat_stream():
                with http.stream('POST', sy + '/v1/chat/completions', json={
                        'model': 'titanos-local', 'messages': CHAT, 'max_tokens': 12, 'stream': True,
                        'stream_options': {'include_usage': True}}) as answer:
                    expect(answer.status_code == 200, f'{answer.status_code}')
                    text = b''.join(answer.iter_bytes()).decode('utf-8', 'replace')
                frames = [line for line in text.splitlines() if line.startswith('data: ')]
                expect(frames and frames[-1] == 'data: [DONE]', text[-300:])
                expect(any('"usage"' in frame for frame in frames), 'no usage chunk')
                return {'frames': len(frames)}

            def b_claude_code_shape():
                answer = http.post(sy + '/v1/messages', json={
                    'model': 'titanos-local', 'max_tokens': 16, 'metadata': {'user_id': 'user_interop_session_0001'},
                    'system': [{'type': 'text', 'text': 'Answer briefly.'}],
                    'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Say hello.'}]}]},
                    headers={'anthropic-version': '2023-06-01', 'x-claude-code-session-id': 'interop-session'})
                expect(answer.status_code == 200, f'{answer.status_code} {answer.text[:400]}')
                body = answer.json()
                expect(body.get('type') == 'message' and body['content'], body)
                return {'stop_reason': body.get('stop_reason'), 'usage': body.get('usage')}

            def b_messages_to_titanos_messages():
                answer = http.post(sy + '/v1/messages', json={
                    'model': 'titanos-local-messages', 'max_tokens': 16, 'metadata': {'user_id': 'u-1'},
                    'messages': [{'role': 'user', 'content': 'Say hello.'}]},
                    headers={'anthropic-version': '2023-06-01'})
                expect(answer.status_code == 200, f'{answer.status_code} {answer.text[:400]}')
                body = answer.json()
                expect(body.get('type') == 'message' and body['content'][0]['text'].strip(), body)
                return {'model': body.get('model')}

            def b_codex_responses():
                answer = http.post(sy + '/v1/responses', json={
                    'model': 'titanos-local-responses', 'input': 'Say hello.', 'max_output_tokens': 16,
                    'store': False, 'include': ['reasoning.encrypted_content'], 'reasoning': {'effort': 'low'},
                    'prompt_cache_key': 'interop-cache-key'})
                expect(answer.status_code == 200, f'{answer.status_code} {answer.text[:400]}')
                body = answer.json()
                # 16 output tokens may end the sampled answer early: 'incomplete' for max_output_tokens is a valid answer
                finished = body.get('status') == 'completed' or (
                    body.get('status') == 'incomplete'
                    and (body.get('incomplete_details') or {}).get('reason') == 'max_output_tokens')
                expect(body.get('object') == 'response' and finished, body)
                return {'output_items': len(body.get('output') or [])}

            def b_cloud_ref():
                before = len(openai_mock.hits)
                answer = http.post(sy + '/v1/chat/completions', json={'model': 'titanos-cloud', 'messages': CHAT})
                expect(answer.status_code == 200, f'{answer.status_code} {answer.text[:400]}')
                hits = openai_mock.hits[before:]
                expect(len(hits) == 1 and hits[0].path == '/v1/chat/completions', [h.path for h in hits])
                expect(hits[0].credential == 'Bearer <sentinel:openai>', hits[0].credential)
                expect(not hits[0].presented(token), 'the Titanos key reached the provider mock')
                expect(hits[0].json()['model'] == 'gpt-4o-mini', hits[0].json().get('model'))
                expect(answer.headers.get('x-model-router-selected-model') == 'gpt-4o-mini@openai',
                       answer.headers.get('x-model-router-selected-model'))
                anthropic_in = http.post(sy + '/v1/messages', json={'model': 'titanos-cloud', 'max_tokens': 32,
                                                                    'messages': [{'role': 'user', 'content': 'hi'}]},
                                         headers={'anthropic-version': '2023-06-01'})
                expect(anthropic_in.status_code == 200 and anthropic_in.json().get('type') == 'message',
                       f'{anthropic_in.status_code} {anthropic_in.text[:300]}')
                return {'mock_hits': len(openai_mock.hits) - before}

            case(cases, 'b.switchyard_to_titanos.local.openai_chat', b_chat, needs_local=True, local=local)
            case(cases, 'b.switchyard_to_titanos.local.openai_chat_stream', b_chat_stream, needs_local=True,
                 local=local)
            case(cases, 'b.switchyard_to_titanos.local.claude_code_shape', b_claude_code_shape, needs_local=True,
                 local=local)
            case(cases, 'b.switchyard_to_titanos.local.anthropic_messages', b_messages_to_titanos_messages,
                 needs_local=True, local=local)
            case(cases, 'b.switchyard_to_titanos.local.codex_responses', b_codex_responses, needs_local=True,
                 local=local)
            case(cases, 'b.switchyard_to_titanos.brokered_cloud_ref', b_cloud_ref)

            # ---- (a) Titanos -> Switchyard ---------------------------------------------------------------------
            def a_catalog():
                rows = {row['id']: row for row in http.get(titanos + '/v1/models', headers=user).json()['data']
                        if row.get('owned_by') == 'sy'}
                wanted = {'switchyard@sy', 'titanos-local@sy', 'titanos-cloud@sy', 'titanos-loop@sy'}
                expect(wanted <= set(rows), sorted(rows))
                info = rows['titanos-local@sy']['titanos']
                expect(info['class'] == 'router' and 'tools' in info['capabilities']
                       and 'vision' not in info['capabilities'] and info.get('context_length') == 8192, info)
                return {'routes': sorted(rows)}

            def a_noop_all_shapes():
                chat = http.post(titanos + '/v1/chat/completions', json={'model': 'switchyard@sy', 'messages': CHAT},
                                 headers=user)
                expect(chat.status_code == 200 and chat.json()['choices'][0]['message']['content'] == 'OK',
                       f'{chat.status_code} {chat.text[:300]}')
                expect(chat.headers.get('x-titanos-served-by') == 'router:sy', chat.headers.get('x-titanos-served-by'))
                expect(chat.headers.get('x-model-router-selected-model') == 'switchyard', dict(chat.headers))
                messages = http.post(titanos + '/v1/messages', json={
                    'model': 'switchyard@sy', 'max_tokens': 16, 'metadata': {'user_id': 'u'},
                    'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi',
                                                               'cache_control': {'type': 'ephemeral'}}]}]},
                    headers={'x-api-key': token, 'anthropic-version': '2023-06-01'})
                expect(messages.status_code == 200 and messages.json()['content'][0]['text'] == 'OK',
                       f'{messages.status_code} {messages.text[:300]}')
                expect(messages.headers.get('x-titanos-translation') == 'passthrough',
                       messages.headers.get('x-titanos-translation'))
                responses = http.post(titanos + '/v1/responses', json={'model': 'switchyard@sy', 'input': 'hi'},
                                      headers=user)
                expect(responses.status_code == 200 and responses.json()['status'] == 'completed',
                       f'{responses.status_code} {responses.text[:300]}')
                with http.stream('POST', titanos + '/v1/chat/completions', json={
                        'model': 'switchyard@sy', 'messages': CHAT, 'stream': True}, headers=user) as streamed:
                    text = b''.join(streamed.iter_bytes()).decode('utf-8', 'replace')
                expect(text.rstrip().endswith('data: [DONE]') and '"usage"' not in text, text[-300:])
                return {'shapes': ['openai.chat', 'openai.chat stream', 'anthropic.messages', 'openai.responses']}

            def a_through_to_local():
                answer = http.post(titanos + '/v1/chat/completions', json={
                    'model': 'titanos-local@sy', 'messages': CHAT, 'max_tokens': 12}, headers=user)
                expect(answer.status_code == 200, f'{answer.status_code} {answer.text[:400]}')
                expect(answer.headers.get('x-titanos-served-by') == 'router:sy', dict(answer.headers))
                expect(answer.headers.get('x-model-router-selected-model') == 'local-chat', dict(answer.headers))
                return {'completion_chars': len(answer.json()['choices'][0]['message']['content'] or '')}

            def a_through_to_cloud():
                before = len(openai_mock.hits)
                answer = http.post(titanos + '/v1/chat/completions', json={'model': 'titanos-cloud@sy',
                                                                          'messages': CHAT}, headers=user)
                expect(answer.status_code == 200, f'{answer.status_code} {answer.text[:400]}')
                expect(len(openai_mock.hits) - before == 1, 'expected exactly one provider call')
                return {'served_by': answer.headers.get('x-titanos-served-by')}

            def a_loop_is_refused():
                before = len(openai_mock.hits)
                answer = http.post(titanos + '/v1/chat/completions', json={'model': 'titanos-loop@sy',
                                                                          'messages': CHAT}, headers=user)
                expect(answer.status_code == 400, f'{answer.status_code} {answer.text[:400]}')
                expect('location_unsatisfiable' in answer.text, answer.text[:400])
                expect(len(openai_mock.hits) == before, 'the loop reached a provider')
                stats = http.get(sy + '/v1/stats').json()
                return {'switchyard_requests': stats.get('total_requests'), 'status': answer.status_code}

            case(cases, 'a.titanos_to_switchyard.catalog_live', a_catalog)
            case(cases, 'a.titanos_to_switchyard.noop_openai_anthropic_responses', a_noop_all_shapes)
            case(cases, 'a.titanos_to_switchyard_to_titanos.local', a_through_to_local, needs_local=True, local=local)
            case(cases, 'a.titanos_to_switchyard_to_titanos.brokered_cloud', a_through_to_cloud)
            def a_loop_without_hop_header_is_refused():
                # a cloud-client route (no X-Titanos-Hop) that points back into Switchyard: Titanos sees its own
                # X-Request-ID coming back on its way to a router again and refuses it (508) on the first re-entry,
                # instead of recursing until the upstream's concurrency slots are exhausted
                before = len(openai_mock.hits)
                started = time.monotonic()
                answer = http.post(titanos + '/v1/chat/completions', json={'model': 'titanos-loop-cloud@sy',
                                                                          'messages': CHAT},
                                   headers={**user, 'X-Request-ID': 'interop-loop-0001'})
                elapsed = time.monotonic() - started
                expect(answer.status_code != 200, f'{answer.status_code} {answer.text[:400]}')
                expect(elapsed < 60, f'the loop took {elapsed:.1f}s (slot exhaustion, not a refusal)')
                expect(len(openai_mock.hits) == before, 'the loop reached a provider')
                return {'status': answer.status_code, 'seconds': round(elapsed, 2), 'body': answer.text[:200]}

            case(cases, 'a.loop_through_switchyard_is_refused', a_loop_is_refused)
            case(cases, 'a.loop_without_hop_header_is_refused', a_loop_without_hop_header_is_refused)
    except Exception as exc:
        cases.append({'id': 'switchyard.interop.harness', 'state': 'fail',
                      'detail': f'{type(exc).__name__}: {str(exc)[:800]}'})
    finally:
        if stop_file is not None:
            stop_file.write_text('stop', encoding='utf-8')
        sy_clean = None
        if sy_proc is not None:
            try:
                sy_proc.wait(30)
                sy_clean = sy_proc.returncode == 0
            except subprocess.TimeoutExpired:
                sy_proc.kill()
                sy_proc.wait(30)
                sy_clean = False
        if sy_log is not None:
            sy_log.close()
        stack.stop()
    if sy_proc is not None:
        checks.append({'id': 'switchyard.clean_shutdown', 'state': 'pass' if sy_clean else 'fail',
                       'detail': f'exit code {sy_proc.returncode}'})
    attempts = stack.blocked_attempts()
    checks.append({'id': 'netguard.zero_non_loopback', 'state': 'fail' if attempts else 'pass',
                   'detail': attempts[:5] if attempts else 'harness and titanos serve made no non-loopback attempt'})
    problems = stack.hit_problems()
    checks.append({'id': 'mocks.credentials_exact', 'state': 'fail' if problems else 'pass',
                   'detail': problems[:5] if problems else {uid: len(m.hits) for uid, m in stack.mocks.items()}})
    logs = {'server_output': stack.server_log()}
    sy_path = stack.root / 'switchyard.log'
    if sy_path.is_file():
        logs['switchyard_output'] = sy_path.read_text(encoding='utf-8', errors='replace')
    logs['report'] = json.dumps(cases)
    leaked = [f'{label} in {where}' for where, text in logs.items() for label, secret in stack.named_secrets().items()
              if secret in text]
    checks.append({'id': 'outputs.no_secret_values', 'state': 'fail' if leaked else 'pass',
                   'detail': sorted(set(leaked)) or 'no sentinel, tk_ token or bootstrap value in any output'})
    checks.append({'id': 'serve.clean_shutdown', 'state': 'pass' if stack.clean_shutdown else 'fail',
                   'detail': f'exit code {stack.process.returncode if stack.process else None}'})
    everything = cases + checks
    states = {state: sum(1 for c in everything if c['state'] == state) for state in ('pass', 'fail', 'untried')}
    result = 'fail' if states['fail'] else 'untried' if states['untried'] else 'pass'
    print(json.dumps({'lane': 'switchyard-interop', 'result': result, 'counts': states,
                      'switchyard': {'package': 'nemo-switchyard', 'version': version},
                      'local_models': sorted(stack.local_specs), 'cases': everything,
                      'scope': 'a real switchyard-server and a real titanos serve on loopback, the WP4 OpenAI mock '
                               'and local fixtures; no provider key'}, indent=1))
    for item in everything:
        if item['state'] != 'pass':
            print(f"{item['state'].upper()}: {item['id']}: {item['detail']}", file=sys.stderr)
    stack.cleanup()
    return 1 if states['fail'] or (strict and states['untried']) else 0


if __name__ == '__main__':
    sys.exit(main())
