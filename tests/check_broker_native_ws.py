"""R32 native WebSocket relay over the integrated app, against loopback RFC 6455 mocks (tests/broker_mocks/ws_mock.py;
the client half is egress.UpstreamWebSocket; the inbound half is the ASGI WebSocket the Starlette test client drives,
and, in the RealServe lane, a real uvicorn server accepting a real upgrade from the websockets client library). The
netguard is installed; every test ends with zero non-loopback connection attempts. Synthetic sentinel credentials only.

- OpenAI Realtime through /native/<upstream>/realtime: the handshake reaches /v1/realtime with the query kept, the
  upstream's own credential (never the client's tk_ key), the 'realtime' subprotocol forwarded and selected and a
  subprotocol carrying a key dropped; client frames are masked; the upstream's ping is answered; text, binary and
  fragmented messages relay both ways; the upstream's close (1000) reaches the client; a native.call audit record
  (event websocket) and a Decision (surface native) are written;
- ElevenLabs stream-input on its bare public path with the xi-api-key header: a key sent inside the first JSON frame is
  removed before it reaches the provider;
- refusals before any upstream connection, as HTTP denial answers: no credential in keyed mode (401), a key without the
  native scope (403 scope_denied); a provider that refuses the handshake with 401 becomes 502 upstream_auth_failed;
- keyless loopback: a cross-site Origin is refused (403), a same-origin client is relayed;
- an HTTP request to a WebSocket-only row answers 400 broker_capability_unavailable, naming the missing websockets or
  wsproto package when it carried Upgrade: websocket (uvicorn accepts WebSocket upgrades only with one of them), on the
  /native path and on the bare OpenAI path /v1/realtime (the OpenAI SDK derives its Realtime URL from base_url /v1);
- RealServe: uvicorn (the server `titanos serve` runs) with its websockets protocol accepts a real upgrade on the bare
  /v1/realtime path and on /native/<upstream>/realtime and relays frames both ways with the upstream's own credential.
  websockets is a declared runtime dependency (pyproject), so CI always runs this lane; an interpreter without it
  reports the lane untried (skipped), never a pass.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import SENTINELS, netguard  # noqa: E402
from broker_mocks.ws_mock import WebSocketMock  # noqa: E402
from starlette.testclient import WebSocketDenialResponse  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

BOOT = 'native-ws-bootstrap-0001'
LOOPBACK = 'ws://127.0.0.1:8000'
MOCKS: dict = {}


def setUpModule():
    MOCKS['openai'] = WebSocketMock()
    MOCKS['elevenlabs'] = WebSocketMock(credential_header='xi-api-key', credential=SENTINELS['elevenlabs'],
                                        subprotocols=())
    MOCKS['refusing'] = WebSocketMock(credential='Bearer something-else-entirely')
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def document(*, anonymous_cloud: bool = False) -> dict:
    return {'policy': {'cloud_policy': 'explicit', 'anonymous_cloud': anonymous_cloud},
            'upstreams': {
                'oai': {'profile': 'openai', 'base_url': MOCKS['openai'].base_url + '/v1', 'credential': 'oai-key',
                        'enabled': True, 'models': ['gpt-realtime']},
                'xi': {'profile': 'elevenlabs', 'base_url': MOCKS['elevenlabs'].base_url, 'credential': 'xi-key',
                       'enabled': True, 'models': ['eleven_flash_v2_5']},
                'bad': {'profile': 'openai', 'base_url': MOCKS['refusing'].base_url + '/v1', 'credential': 'bad-key',
                        'enabled': True, 'models': ['gpt-realtime']}}}


class Base(unittest.TestCase):
    keyed = True
    anonymous_cloud = False

    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.hits.clear()
        self.full = W.FullApp(document(anonymous_cloud=self.anonymous_cloud), api_key=BOOT if self.keyed else None)
        self.http = self.full.http

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def key(self, *scopes) -> str:
        return self.full.mint(*scopes, name='ws' + str(len(scopes)))[1]


class Relay(Base):
    def test_openai_realtime_relay(self):
        token = self.key('native', 'cloud')
        with self.http.websocket_connect('/native/oai/realtime?model=gpt-realtime',
                                         headers={'Authorization': 'Bearer ' + token, 'OpenAI-Beta': 'realtime=v1'},
                                         subprotocols=['realtime', 'openai-insecure-api-key.sk-test-not-a-key']) as ws:
            self.assertEqual(ws.accepted_subprotocol, 'realtime')
            created = ws.receive_json()
            self.assertEqual((created['type'], created['path']), ('session.created', '/v1/realtime'))
            self.assertEqual(created['query'], [['model', 'gpt-realtime']])
            ws.send_json({'type': 'session.update', 'session': {'voice': 'alloy'}})
            self.assertEqual(ws.receive_json(), {'type': 'echo', 'data': {'type': 'session.update',
                                                                          'session': {'voice': 'alloy'}}})
            ws.send_bytes(b'\x00\x01\x02pcm')
            self.assertEqual(ws.receive_bytes(), b'\x00\x01\x02pcm')
            ws.send_text('big-message')
            big = ws.receive_json()
            self.assertEqual((big['type'], len(big['data'])), ('big', 300))
            ws.send_text('close-me')
            with self.assertRaises(WebSocketDisconnect) as closed:
                ws.receive_text()
            self.assertEqual(closed.exception.code, 1000)
        hit = MOCKS['openai'].hits[-1]
        self.assertEqual(hit.credentials, [('header:authorization', 'Bearer <sentinel:openai>')])
        self.assertEqual(hit.subprotocol, 'realtime')
        self.assertNotIn('insecure', hit.headers.get('sec-websocket-protocol', ''))
        self.assertEqual(hit.headers.get('openai-beta'), 'realtime=v1')
        self.assertFalse(hit.presented(token))
        self.assertFalse(any(name.startswith('x-titanos') for name in hit.headers))
        self.assertGreaterEqual(hit.pongs, 1)
        self.assertEqual(hit.unmasked, 0)
        audit = self.full.broker.audit.query(kind='native.call', limit=10)
        self.assertTrue(any(row['detail'].get('event') == 'websocket' for row in audit), audit)
        decisions = self.full.broker.decisions.query(limit=5,
                                                     principal=self.full.broker.authenticator.bootstrap_principal())
        self.assertEqual((decisions[0]['surface'], decisions[0]['translation']), ('native', 'native'))

    def test_elevenlabs_stream_input_strips_a_key_in_the_first_frame(self):
        token = self.key('native', 'cloud')
        with self.http.websocket_connect('/v1/text-to-speech/voice0001/stream-input?model_id=eleven_flash_v2_5',
                                         headers={'xi-api-key': token}) as ws:
            ws.receive_json()
            ws.send_json({'text': ' ', 'xi_api_key': 'sk-test-in-frame-not-a-key', 'voice_settings': {'speed': 1}})
            echoed = ws.receive_json()
            self.assertEqual(echoed['data'], {'text': ' ', 'voice_settings': {'speed': 1}})
        hit = MOCKS['elevenlabs'].hits[-1]
        self.assertEqual(hit.credentials, [('header:xi-api-key', '<sentinel:elevenlabs>')])
        self.assertNotIn('sk-test-in-frame', json.dumps(hit.frames))
        self.assertEqual(hit.path, '/v1/text-to-speech/voice0001/stream-input')


class Refusals(Base):
    def test_denials_before_any_upstream_connection(self):
        plain = self.key('inference', 'cloud')
        for headers, status, code in (({}, 401, 'authentication_error'),
                                      ({'Authorization': 'Bearer ' + plain}, 403, 'scope_denied')):
            with self.subTest(status=status):
                with self.assertRaises(WebSocketDenialResponse) as denied:
                    with self.http.websocket_connect('/native/oai/realtime', headers=headers):
                        pass
                self.assertEqual(denied.exception.status_code, status)
                self.assertEqual(denied.exception.json()['error']['code'], code)
        self.assertEqual(MOCKS['openai'].hits, [])

    def test_provider_handshake_401(self):
        token = self.key('native', 'cloud')
        with self.assertRaises(WebSocketDenialResponse) as denied:
            with self.http.websocket_connect('/native/bad/realtime', headers={'Authorization': 'Bearer ' + token}):
                pass
        self.assertEqual((denied.exception.status_code, denied.exception.json()['error']['code']),
                         (502, 'upstream_auth_failed'))
        self.assertEqual(MOCKS['refusing'].hits[-1].status, 401)

    def test_transport_failures_opening_the_upstream_are_503_and_counted_once(self):
        """A connect error, reset or httpcore failure while opening the upstream WebSocket is 503 upstream_unavailable
        (the relay's except clause must be a flat tuple: CPython 3.11+ raises TypeError at match time for a nested one,
        which left a half-open breaker's probe slot taken for good)."""
        from unittest.mock import AsyncMock, patch
        import httpcore
        from titanos.broker.egress import EgressClient
        token = self.key('native', 'cloud')
        for failure in (ConnectionRefusedError('refused'), httpcore.ConnectError('no route'), OSError('reset')):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(EgressClient, 'open_websocket', new=AsyncMock(side_effect=failure)):
                    with self.assertRaises(WebSocketDenialResponse) as denied:
                        with self.http.websocket_connect('/native/oai/realtime',
                                                         headers={'Authorization': 'Bearer ' + token}):
                            pass
                self.assertEqual((denied.exception.status_code, denied.exception.json()['error']['code']),
                                 (503, 'upstream_unavailable'))
        self.assertEqual(MOCKS['openai'].hits, [])

    def test_http_request_to_a_websocket_row(self):
        token = self.key('native', 'cloud')
        plain = self.http.get('/native/oai/realtime', headers={'Authorization': 'Bearer ' + token})
        self.assertEqual((plain.status_code, plain.json()['error']['code']), (400, 'broker_capability_unavailable'))
        self.assertIn('WebSocket client', plain.json()['error']['message'])
        upgrade = self.http.get('/native/oai/realtime', headers={'Authorization': 'Bearer ' + token,
                                                                  'Upgrade': 'websocket', 'Connection': 'Upgrade'})
        self.assertEqual(upgrade.status_code, 400)
        bare = self.http.get('/v1/realtime?model=gpt-realtime', headers={'Authorization': 'Bearer ' + token,
                                                                          'Upgrade': 'websocket',
                                                                          'Connection': 'Upgrade',
                                                                          'X-Titanos-Upstream': 'oai'})
        self.assertEqual((bare.status_code, bare.json()['error']['code']), (400, 'broker_capability_unavailable'),
                         bare.text)
        from titanos.broker.native import websocket_support
        if websocket_support() is None:
            self.assertIn('websockets or wsproto', upgrade.json()['error']['message'])
            self.assertIn('websockets or wsproto', bare.json()['error']['message'])
        self.assertEqual(MOCKS['openai'].hits, [])


class Keyless(Base):
    keyed = False
    anonymous_cloud = True

    def test_cross_site_origin_refused_same_origin_relayed(self):
        with self.assertRaises(WebSocketDenialResponse) as denied:
            # the test client's WebSocket URL defaults to ws://testserver; keyless mode accepts loopback Host names only
            with self.http.websocket_connect(LOOPBACK + '/native/oai/realtime', headers={'Origin': 'http://evil.example'}):
                pass
        self.assertEqual(denied.exception.status_code, 403)
        self.assertEqual(MOCKS['openai'].hits, [])
        with self.http.websocket_connect(LOOPBACK + '/native/oai/realtime',
                                         headers={'Origin': 'http://127.0.0.1:8000'}) as ws:
            self.assertEqual(ws.receive_json()['type'], 'session.created')
        self.assertEqual(MOCKS['openai'].hits[-1].credentials, [('header:authorization', 'Bearer <sentinel:openai>')])


def _bound_socket() -> socket.socket:
    """A loopback socket bound to a free port and handed to uvicorn as it is: closing it and binding the port again
    lets another socket take the port in between (Windows then refuses uvicorn's bind with WinError 10013)."""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    return sock


class RealServe(unittest.TestCase):
    """A real uvicorn server (as `titanos serve` runs it) accepting real WebSocket upgrades."""

    def setUp(self):
        from titanos.broker.native import websocket_support
        try:
            from websockets.sync.client import connect  # noqa: F401
        except ImportError:
            connect = None
        if websocket_support() != 'websockets' or connect is None:
            self.skipTest('untried: the websockets package is not importable here (it is a runtime dependency: '
                          'pip install .)')
        import uvicorn
        netguard.reset_attempts()
        MOCKS['openai'].hits.clear()
        self.full = W.FullApp(document(), api_key=BOOT)
        self.token = self.full.mint('native', 'cloud', name='realserve')[1]
        self.sock = _bound_socket()
        self.port = self.sock.getsockname()[1]
        config = uvicorn.Config(self.full.app, host='127.0.0.1', port=self.port, ws='websockets', lifespan='off',
                                log_level='warning', proxy_headers=False)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, kwargs={'sockets': [self.sock]}, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 20
        while not self.server.started and time.monotonic() < deadline and self.thread.is_alive():
            time.sleep(0.05)
        self.assertTrue(self.server.started, 'uvicorn did not start')

    def tearDown(self):
        server = getattr(self, 'server', None)
        if server is not None:
            server.should_exit = True
            self.thread.join(timeout=20)
            self.sock.close()
            self.full.close()
            self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def relay(self, path: str) -> None:
        from websockets.sync.client import connect
        url = f'ws://127.0.0.1:{self.port}{path}'
        with connect(url, additional_headers={'Authorization': 'Bearer ' + self.token},
                     subprotocols=['realtime'], open_timeout=20) as ws:
            self.assertEqual(ws.subprotocol, 'realtime')
            created = json.loads(ws.recv(timeout=20))
            self.assertEqual((created['type'], created['path']), ('session.created', '/v1/realtime'))
            ws.send(json.dumps({'type': 'session.update'}))
            self.assertEqual(json.loads(ws.recv(timeout=20))['data'], {'type': 'session.update'})
            ws.send(b'\x01\x02pcm')
            self.assertEqual(ws.recv(timeout=20), b'\x01\x02pcm')
        hit = MOCKS['openai'].hits[-1]
        self.assertEqual(hit.credentials, [('header:authorization', 'Bearer <sentinel:openai>')])
        self.assertFalse(hit.presented(self.token))

    def test_bare_openai_realtime_path(self):
        # the OpenAI SDK's realtime URL for base_url http://host:port/v1; two openai upstreams here, so the header names one
        from websockets.sync.client import connect
        url = f'ws://127.0.0.1:{self.port}/v1/realtime?model=gpt-realtime'
        with connect(url, additional_headers={'Authorization': 'Bearer ' + self.token, 'X-Titanos-Upstream': 'oai'},
                     subprotocols=['realtime'], open_timeout=20) as ws:
            created = json.loads(ws.recv(timeout=20))
            self.assertEqual((created['type'], created['query']), ('session.created', [['model', 'gpt-realtime']]))
        self.assertEqual(MOCKS['openai'].hits[-1].credentials, [('header:authorization', 'Bearer <sentinel:openai>')])

    def test_native_prefix_realtime(self):
        self.relay('/native/oai/realtime?model=gpt-realtime')


if __name__ == '__main__':
    unittest.main()
