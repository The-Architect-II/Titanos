"""OPS (b): gateway registration carries a bearer (cli.register_with_gateway, `titanos hivemind register-service`).

- register_with_gateway(..., key=K) sends Authorization: Bearer K on POST <gateway>/register-service; without a key the
  request carries no Authorization header (unchanged loopback behaviour);
- `titanos hivemind register-service GATEWAY --job-type JT [--port P] [--url U] [--key-env VAR]`: the bearer comes from
  TITANOS_API_KEY when set, from the variable --key-env names when given (a set variable is required then), and is
  absent otherwise; the printed result says bearer_sent and never the secret; a non-2xx gateway answer or an unready
  server is exit 1 with a message; --key (a secret on the command line) is not an option (exit 2, value not echoed);
- serve --register-with-gateway hands the registration thread TITANOS_API_KEY as the bearer.
A loopback gateway stand-in records the request; the netguard asserts zero non-loopback connection attempts.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support  # noqa: E402,F401  (installs the netguard)
from broker_mocks import netguard  # noqa: E402
from titanos import cli  # noqa: E402

SECRET = 'tk_gateway_fixture_secret_0001'
OTHER = 'hive_fixture_other_secret_0002'


class ExclusiveServer(ThreadingHTTPServer):
    allow_reuse_address = False  # on Windows SO_REUSEADDR lets the bind succeed on a port another socket holds


def bind_in_window(handler) -> ThreadingHTTPServer:
    """The server bound to the first free loopback port in the lane's 49900-49949 window (binding is the probe: closing
    a probe socket and binding the port again lets another socket take it in between)."""
    for port in range(49900, 49950):
        try:
            return ExclusiveServer(('127.0.0.1', port), handler)
        except OSError:
            continue
    raise AssertionError('no free port in 49900-49949')


class Gateway:
    """Records POST /register-service (headers and JSON body); answers GET /v1/health/ready 200 so it can stand in for
    the local readiness route too. status: the answer to the registration."""

    def __init__(self):
        self.requests: list[dict] = []
        self.status = 200
        self.ready = True
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                code = 200 if owner.ready and self.path == '/v1/health/ready' else 503
                self._answer(code, {'status': 'ready' if code == 200 else 'not_ready'})

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or 0)
                body = self.rfile.read(length)
                owner.requests.append({'path': self.path, 'authorization': self.headers.get('Authorization'),
                                       'body': json.loads(body or b'{}')})
                self._answer(owner.status, {'ok': owner.status < 300})

            def _answer(self, code, document):
                payload = json.dumps(document).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = bind_in_window(Handler)
        self.port = self.server.server_address[1]
        self.url = f'http://127.0.0.1:{self.port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


GATEWAY: dict = {}


def setUpModule():
    GATEWAY['g'] = Gateway().start()


def tearDownModule():
    GATEWAY['g'].stop()


def run(argv, *, env=None, clear=('TITANOS_API_KEY', 'TITANOS_URL', 'HIVE_KEY')):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with mock.patch.dict(os.environ, env or {}):
        for name in clear:
            if name not in (env or {}):
                os.environ.pop(name, None)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                cli.main(argv)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


class Guarded(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        self.gateway = GATEWAY['g']
        self.gateway.requests.clear()
        self.gateway.status = 200
        self.gateway.ready = True

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')


class Function(Guarded):
    def test_key_becomes_the_bearer(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            status = cli.register_with_gateway(self.gateway.url, 'jt', 4242, self.gateway.url + '/v1/health/ready',
                                               attempts=5, key=SECRET)
        self.assertEqual(status, 200)
        self.assertEqual(self.gateway.requests, [{'path': '/register-service', 'authorization': 'Bearer ' + SECRET,
                                                  'body': {'job_type': 'jt', 'port': 4242}}])
        self.assertIn('bearer sent', err.getvalue())
        self.assertNotIn(SECRET, err.getvalue())

    def test_no_key_sends_no_authorization(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            status = cli.register_with_gateway(self.gateway.url, 'jt', 4242, self.gateway.url + '/v1/health/ready',
                                               attempts=5)
        self.assertEqual(status, 200)
        self.assertIsNone(self.gateway.requests[0]['authorization'])
        self.assertIn('no bearer', err.getvalue())

    def test_unready_server_registers_nothing(self):
        self.gateway.ready = False
        with contextlib.redirect_stderr(io.StringIO()) as err:
            status = cli.register_with_gateway(self.gateway.url, 'jt', 4242, self.gateway.url + '/v1/health/ready',
                                               attempts=2, key=SECRET)
        self.assertIsNone(status)
        self.assertEqual(self.gateway.requests, [])
        self.assertIn('never became ready', err.getvalue())


class Subcommand(Guarded):
    def registered(self, code, out, err):
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual(len(self.gateway.requests), 1, self.gateway.requests)
        return result, self.gateway.requests[0]

    def test_titanos_api_key_is_the_default_bearer(self):
        code, out, err = run(['hivemind', 'register-service', self.gateway.url, '--job-type', 'jt', '--url',
                              self.gateway.url, '--port', '4242', '--attempts', '5'], env={'TITANOS_API_KEY': SECRET})
        result, request = self.registered(code, out, err)
        self.assertEqual(request['authorization'], 'Bearer ' + SECRET)
        self.assertEqual(request['body'], {'job_type': 'jt', 'port': 4242})
        self.assertEqual((result['bearer_sent'], result['status'], result['port'], result['job_type']),
                         (True, 200, 4242, 'jt'))
        self.assertNotIn(SECRET, out + err)

    def test_key_env_names_another_variable(self):
        code, out, err = run(['hivemind', 'register-service', self.gateway.url, '--job-type', 'jt', '--url',
                              self.gateway.url, '--key-env', 'HIVE_KEY', '--attempts', '5'],
                             env={'TITANOS_API_KEY': SECRET, 'HIVE_KEY': OTHER})
        result, request = self.registered(code, out, err)
        self.assertEqual(request['authorization'], 'Bearer ' + OTHER)
        self.assertEqual(result['port'], self.gateway.port)  # defaults to the --url port
        self.assertNotIn(OTHER, out + err)

    def test_key_env_must_be_set(self):
        code, out, err = run(['hivemind', 'register-service', self.gateway.url, '--job-type', 'jt', '--url',
                              self.gateway.url, '--key-env', 'HIVE_KEY', '--attempts', '5'])
        self.assertEqual(code, 1)
        self.assertIn('HIVE_KEY', err)
        self.assertIn('not set', err)
        self.assertEqual(self.gateway.requests, [])

    def test_without_any_key_no_bearer(self):
        code, out, err = run(['hivemind', 'register-service', self.gateway.url, '--job-type', 'jt', '--url',
                              self.gateway.url, '--attempts', '5'])
        result, request = self.registered(code, out, err)
        self.assertIsNone(request['authorization'])
        self.assertFalse(result['bearer_sent'])

    def test_rejected_registration_is_exit_1(self):
        self.gateway.status = 401
        code, out, err = run(['hivemind', 'register-service', self.gateway.url, '--job-type', 'jt', '--url',
                              self.gateway.url, '--attempts', '5'], env={'TITANOS_API_KEY': SECRET})
        self.assertEqual(code, 1)
        self.assertIn('HTTP 401', err)
        self.assertEqual(json.loads(out)['status'], 401)

    def test_refusals(self):
        code, _, err = run(['hivemind', 'register-service', self.gateway.url, '--job-type', 'bad type', '--url',
                            self.gateway.url])
        self.assertEqual(code, 1)
        self.assertIn('--job-type', err)
        code, _, err = run(['hivemind', 'register-service', 'ftp://x', '--job-type', 'jt'])
        self.assertEqual(code, 2)
        code, _, err = run(['hivemind', 'register-service', self.gateway.url, '--job-type', 'jt', '--key', SECRET])
        self.assertEqual(code, 2)  # no option takes a secret value
        self.assertIn('unrecognized arguments', err)
        self.assertNotIn(SECRET, err)
        self.assertEqual(self.gateway.requests, [])


class ServeHandsOverTheKey(Guarded):
    def test_serve_passes_titanos_api_key_to_the_registration(self):
        recorded = {}

        def fake_register(url, job_type, port, health_url, *, attempts=300, key=None):
            recorded.update(url=url, job_type=job_type, port=port, health_url=health_url, key=key)

        class ImmediateThread:
            def __init__(self, target, args=(), kwargs=None, daemon=None):
                self._run = lambda: target(*args, **(kwargs or {}))

            def start(self):
                self._run()

        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / 'models.json'
            from titanos.config import Config, write_config
            write_config(registry, Config())
            env = {'TITANOS_HOME': tmp, 'TITANOS_API_KEY': SECRET}
            with mock.patch.dict('sys.modules', {'titanos.server': SimpleNamespace(create_app=mock.Mock(return_value='app'))}), \
                    mock.patch('uvicorn.run', mock.Mock()), mock.patch('threading.Thread', ImmediateThread), \
                    mock.patch.object(cli, 'register_with_gateway', fake_register):
                code, out, err = run(['--config', str(registry), 'serve', '--role', 'managed', '--port', '49949',
                                      '--register-with-gateway', self.gateway.url, '--job-type', 'jt'], env=env,
                                     clear=('TITANOS_ROLE', 'TITANOS_PORT', 'TITANOS_URL'))
        self.assertEqual(code, 0, err)
        self.assertEqual((recorded['key'], recorded['job_type'], recorded['port']), (SECRET, 'jt', 49949))
        self.assertTrue(recorded['health_url'].endswith(':49949/v1/health/ready'))
        self.assertNotIn(SECRET, out + err)


if __name__ == '__main__':
    unittest.main(verbosity=2)
