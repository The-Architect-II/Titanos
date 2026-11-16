"""TITANOS_HEALTH_PORT: health routes on their own listener, outside uvicorn's concurrency cap.

A real uvicorn pair (titanos.health_listener.server) on two free loopback ports, over a small ASGI application: with the
main server's limit_concurrency at 2 (uvicorn counts the open connections) and one slow request open, the main port answers 503 while the health port
answers /healthz, /readyz and /v1/health/ready from the same application and 404 to any other path; stopping the main
server stops the health server. Also: health_app routing at the ASGI level, the setting's bounds, and `titanos serve`
choosing the runner only when the port is set (uvicorn replaced, nothing binds). Loopback only, ephemeral ports chosen by
the operating system; no model, GPU or outside network."""
import asyncio
import functools
import json
import logging
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from titanos import health_listener  # noqa: E402
from titanos.cli import main  # noqa: E402
from titanos.config import Config, write_config  # noqa: E402
from titanos.settings import Settings  # noqa: E402


def bound_socket():
    """A loopback socket bound to a free port and handed to uvicorn as it is: closing it and binding the port again
    lets another socket take the port in between (Windows then refuses uvicorn's bind with WinError 10013)."""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    return sock


def application(release):
    async def app(scope, receive, send):
        if scope['type'] != 'http':
            return
        if scope['path'] == '/slow':
            while not release.is_set():
                await asyncio.sleep(0.02)
        body = json.dumps({'path': scope['path'], 'server_port': scope['server'][1]}).encode()
        await send({'type': 'http.response.start', 'status': 200, 'headers': [(b'content-type', b'application/json')]})
        await send({'type': 'http.response.body', 'body': body})
    return app


def check_routing():
    calls, sent = [], []

    async def app(scope, receive, send):
        calls.append(scope['path'])

    async def send(message):
        sent.append(message)

    async def scenario():
        health = health_listener.health_app(app)
        for path in ('/healthz', '/readyz', '/v1/health/ready'):
            await health({'type': 'http', 'path': path}, None, send)
        await health({'type': 'http', 'path': '/v1/chat/completions'}, None, send)
        await health({'type': 'websocket', 'path': '/healthz'}, None, send)
    asyncio.run(scenario())
    assert calls == ['/healthz', '/readyz', '/v1/health/ready'], calls
    assert sent[0]['status'] == 404 and json.loads(sent[1]['body'])['error']['code'] == 'not_found', sent
    assert sent[2] == {'type': 'websocket.close', 'code': 1008}, sent


def check_live():
    release = threading.Event()
    main_sock, health_sock = bound_socket(), bound_socket()
    main_port, health_port = main_sock.getsockname()[1], health_sock.getsockname()[1]
    pair = health_listener.server(application(release), health_port, host='127.0.0.1', port=main_port, limit_concurrency=2,
                                  log_level='warning', timeout_keep_alive=5)
    assert logging.getLogger('uvicorn.access').handlers, 'building the health server leaves access logging configured'
    pair.health.serve = functools.partial(pair.health.serve, sockets=[health_sock])
    thread = threading.Thread(target=pair.run, kwargs={'sockets': [main_sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not (pair.started and pair.health.started) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pair.started and pair.health.started, 'servers did not start'
    with httpx.Client(timeout=10, trust_env=False) as client:
        slow = threading.Thread(target=lambda: client.get(f'http://127.0.0.1:{main_port}/slow'), daemon=True)
        slow.start()
        time.sleep(0.5)
        with httpx.Client(timeout=10, trust_env=False) as other:
            busy = other.get(f'http://127.0.0.1:{main_port}/healthz')
            assert busy.status_code == 503, busy.status_code  # the cap refuses the main port's own health route
            for path in ('/healthz', '/readyz', '/v1/health/ready'):
                answer = other.get(f'http://127.0.0.1:{health_port}{path}')
                assert answer.status_code == 200 and answer.json() == {'path': path, 'server_port': health_port}, answer.text
            refused = other.get(f'http://127.0.0.1:{health_port}/slow')
            assert refused.status_code == 404, refused.status_code
        release.set()
        slow.join(10)
    pair.should_exit = True
    thread.join(20)
    assert not thread.is_alive(), 'the server pair did not stop'
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(('127.0.0.1', health_port)) != 0, 'the health port still accepts after the main server stopped'


def check_setting_and_serve():
    assert Settings().health_port == 0 and Settings(health_port=11448).health_port == 11448
    for bad in ({'health_port': 70000}, {'health_port': 11435, 'port': 11435}, {'health_port': -1}):
        try:
            Settings(**bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} accepted')
    with tempfile.TemporaryDirectory() as tmp:
        registry = Path(tmp) / 'models.json'
        write_config(registry, Config())
        clean = {key: value for key, value in os.environ.items() if not key.startswith('TITANOS_')}
        app_factory, plain_run, paired_run = Mock(return_value='fixture-app'), Mock(), Mock()
        runner = Mock(return_value=paired_run)
        with patch.dict(os.environ, {**clean, 'TITANOS_HOME': tmp}, clear=True), \
             patch.dict('sys.modules', {'titanos.server': SimpleNamespace(create_app=app_factory)}), \
             patch('uvicorn.run', plain_run), patch.object(health_listener, 'runner', runner):
            main(['--config', str(registry), 'serve', '--port', '12001'])
            assert plain_run.called and not runner.called
            plain_run.reset_mock()
            os.environ['TITANOS_HEALTH_PORT'] = '12002'
            main(['--config', str(registry), 'serve', '--port', '12001'])
            assert runner.call_args.args == (12002,) and not plain_run.called, runner.call_args
            kwargs = paired_run.call_args.kwargs
            assert paired_run.call_args.args == ('fixture-app',) and kwargs['port'] == 12001 and kwargs['limit_concurrency'] == 64, kwargs


def main_check():
    check_routing()
    check_setting_and_serve()
    check_live()
    print('{"check": "health_listener", "result": "pass", "scope": "a real uvicorn pair on two loopback ephemeral ports over a '
          'small ASGI app; no model or GPU"}')


if __name__ == '__main__':
    main_check()
