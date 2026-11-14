"""WP7 disabled-lane regression: with the broker installed but not in use, the existing inference and catalog routes
answer exactly as before the broker existed.

The reference capture is the same app with BrokerDispatch not installed and TrustedHostMiddleware in place of
KeylessHostGuard (and the Guard's original single bearer compare). uuid4 and time.time (and the Ollama clock) are
patched to the same deterministic sequences in both captures, and a recording fake engine answers every request.

(a) no broker.json: status, headers and body are identical for chat (stream and non-stream), completions, responses,
    embeddings, speech, /v1/messages, /api/chat, /v1/text-to-speech, /v1/models and /api/tags, and the invalid-Host
    refusal is byte-identical;
(b) broker.json with only a cloud upstream enabled and cloud_policy off: bodies are identical; headers differ only by
    the added X-Titanos-* and X-Request-ID provenance headers (the local route still answers);
/api/info differs only by the documented additive keys (role, orchestrated, broker) and /healthz by role.
No connection leaves loopback (the cloud upstream is never contacted).
"""
from __future__ import annotations

import itertools
import sys
import time as real_time
import unittest
import uuid as real_uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import netguard, unused_loopback_url  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

LOCAL = {'chat-local': 'chat', 'embed-local': 'embeddings', 'voice': 'speech'}
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
REQUESTS = [
    ('POST', '/v1/chat/completions', {'model': 'chat-local', 'messages': CHAT}),
    ('POST', '/v1/chat/completions', {'model': 'chat-local', 'messages': CHAT, 'stream': True,
                                      'stream_options': {'include_usage': True}}),
    ('POST', '/v1/completions', {'model': 'chat-local', 'prompt': 'Say hello.'}),
    ('POST', '/v1/responses', {'model': 'chat-local', 'input': 'Say hello.'}),
    ('POST', '/v1/embeddings', {'model': 'embed-local', 'input': ['one', 'two']}),
    ('POST', '/v1/audio/speech', {'model': 'voice', 'input': 'Hello.'}),
    ('POST', '/v1/messages', {'model': 'chat-local', 'max_tokens': 16, 'messages': CHAT}),
    ('POST', '/v1/messages', {'model': 'chat-local', 'max_tokens': 16, 'messages': CHAT, 'stream': True}),
    ('POST', '/api/chat', {'model': 'chat-local', 'messages': CHAT, 'stream': False}),
    ('POST', '/api/chat', {'model': 'chat-local', 'messages': CHAT}),
    ('POST', '/v1/text-to-speech/default?output_format=wav', {'text': 'Hello.', 'model_id': 'voice'}),
    ('POST', '/v1/text-to-speech/default?output_format=wav', {'text': 'Hello.'}),
    ('POST', '/v1/chat/completions', {'model': 'missing', 'messages': CHAT}),
    ('GET', '/v1/models', None),
    ('GET', '/api/tags', None),
]
PRE_BROKER_INFO = {'name', 'version', 'standalone', 'inference_contracts', 'interfaces', 'model_setup', 'tasks', 'models',
                   'jobs', 'settings', 'notes'}
# in the configured lane an unknown name is answered by the broker's router (404 model_not_found with its own message)
ROUTER_ANSWERED = {('POST', '/v1/chat/completions', 'missing')}
PROVENANCE = {'x-request-id', 'x-titanos-decision-id', 'x-titanos-fallback-count', 'x-titanos-requested-model',
              'x-titanos-served-by', 'x-titanos-served-model', 'x-titanos-translation'}


class _Clock:
    """A module-local `time` with a deterministic time() (monotonic and the rest stay real)."""

    def __init__(self):
        self._ticks = itertools.count(1_790_000_000)

    def time(self):
        return float(next(self._ticks))

    def __getattr__(self, name):
        return getattr(real_time, name)


class _Ids:
    def __init__(self):
        self._n = itertools.count(1)

    def uuid4(self):
        return real_uuid.UUID(int=next(self._n))

    def __getattr__(self, name):
        return getattr(real_uuid, name)


class _FixedDatetime:
    @staticmethod
    def now(tz=None):
        return datetime(2026, 10, 1, tzinfo=timezone.utc)


def reference(app) -> None:
    """The pre-broker middleware: Guard with the single bearer compare, TrustedHostMiddleware, no BrokerDispatch."""
    from titanos.server import Guard
    guard = next(m for m in app.user_middleware if m.cls is Guard)
    kwargs = dict(guard.kwargs)
    kwargs.pop('auth', None)
    kwargs.pop('public_routes', None)
    app.user_middleware = [Middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]']),
                           Middleware(Guard, **kwargs)]
    app.middleware_stack = None


def capture(document, *, legacy: bool):
    full = W.FullApp(document, models=LOCAL)
    try:
        if legacy:
            reference(full.app)
        full.app.state.engine.target.started = 1_790_000_000  # the catalog's created stamp (engine construction time)
        client = full.http
        out = []
        ids, clock, ollama_clock = _Ids(), _Clock(), SimpleNamespace(monotonic_ns=itertools.count(1000, 1000).__next__)
        with mock.patch('titanos.server.uuid', ids), mock.patch('titanos.responses.uuid', ids), \
                mock.patch('titanos.protocols.uuid', ids), mock.patch('titanos.server.time', clock), \
                mock.patch('titanos.responses.time', clock), mock.patch('titanos.protocols.time', ollama_clock), \
                mock.patch('titanos.protocols.datetime', _FixedDatetime):
            for method, path, body in REQUESTS:
                reply = client.request(method, path, json=body)
                out.append((method, path, reply.status_code, dict(reply.headers), reply.content))
            bad = client.get('/v1/models', headers={'Host': 'evil.example'})
            out.append(('GET', 'invalid-host', bad.status_code, dict(bad.headers), bad.content))
            info = client.get('/api/info').json()
            health = client.get('/healthz').json()
        return out, info, health, full.engine.calls
    finally:
        full.close()


class DisabledLane(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def test_a_no_broker_json_is_byte_identical(self):
        before, info_before, health_before, calls_before = capture(None, legacy=True)
        after, info_after, health_after, calls_after = capture(None, legacy=False)
        self.assertEqual(len(before), len(after))
        for old, new in zip(before, after):
            with self.subTest(request=old[:2]):
                self.assertEqual(old, new)
        self.assertEqual(before[-1][2:], (400, {'content-length': '19', 'content-type': 'text/plain; charset=utf-8'},
                                          b'Invalid host header'))
        self.assertEqual(calls_before, calls_after)
        # the pre-broker /api/info keys, unchanged, plus exactly the documented additive keys
        self.assertEqual(set(info_after) - PRE_BROKER_INFO, {'role', 'orchestrated', 'broker'})
        self.assertTrue(PRE_BROKER_INFO <= set(info_after))
        self.assertEqual(info_after['standalone'], True)
        self.assertEqual({k: v for k, v in info_after.items() if k in PRE_BROKER_INFO},
                         {k: v for k, v in info_before.items() if k in PRE_BROKER_INFO})
        self.assertEqual((info_after['role'], info_after['orchestrated'], info_after['broker']['configured']),
                         ('standalone', False, False))
        self.assertNotIn('url', str(info_after['broker']).lower())
        self.assertEqual(health_after, {**health_before, 'role': 'standalone'})

    def test_b_cloud_upstream_with_policy_off_adds_only_provenance_headers(self):
        document = {'policy': {'cloud_policy': 'off'},
                    'upstreams': {'openai': {'profile': 'openai', 'base_url': unused_loopback_url() + '/v1',
                                             'enabled': True, 'models': ['gpt-4o-mini']}}}
        before, _, _, calls_before = capture(None, legacy=True)
        after, info, _, calls_after = capture(document, legacy=False)
        self.assertTrue(info['broker']['configured'])
        for (method, path, body), old, new in zip(REQUESTS, before[:-1], after[:-1]):
            if (method, path, (body or {}).get('model')) in ROUTER_ANSWERED:
                self.assertEqual((new[2], new[3].get('x-titanos-error-origin')), (404, 'titanos'))
                continue
            with self.subTest(request=old[:2]):
                self.assertEqual((old[2], old[4]), (new[2], new[4]))
                added = {k for k in new[3] if k not in old[3]}
                self.assertTrue(added <= PROVENANCE, added)
                self.assertEqual({k: v for k, v in new[3].items() if k not in PROVENANCE},
                                 {k: v for k, v in old[3].items() if k not in PROVENANCE})
        inference = [row for row in after if row[1] not in ('/v1/models', '/api/tags')]
        self.assertTrue(all('x-request-id' in row[3] for row in inference if row[2] == 200))
        self.assertEqual(calls_before, calls_after)


if __name__ == '__main__':
    unittest.main(verbosity=2)
