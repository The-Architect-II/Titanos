"""Local models as the target of an agent harness or a router, over the integrated app (create_app: KeylessHostGuard,
Guard + Authenticator, BrokerDispatch). The netguard is installed and every test ends with zero non-loopback connection
attempts. Synthetic credentials only.

- the standard identity, cache, storage and reasoning fields Claude Code and Codex send (and routers carry through) are
  accepted on /v1/chat/completions, /v1/responses and /v1/messages and named in X-Titanos-Ignored; store=true,
  previous_response_id, background, Anthropic container and output_config.format answer a typed 400; unknown fields
  still answer 422;
- a request that re-enters with an X-Request-ID this instance is currently sending to a router upstream is served when
  it targets a local model (it never goes to a router again), and refused 508 loop_detected with zero egress when it
  would go to a router upstream again (a HiveMind-gateway-like router mock that forwards X-Request-ID).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import RouterUpstreamMock, netguard  # noqa: E402

BOOT = 'local-compat-check-bootstrap-0001'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
MOCKS: dict = {}


def setUpModule():
    MOCKS['hive'] = RouterUpstreamMock('hivemind')
    MOCKS['hive'].start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


class RouterLoop(unittest.TestCase):
    """A router upstream that drops X-Titanos-Hop but forwards X-Request-ID cannot loop a request back here."""

    def setUp(self):
        netguard.reset_attempts()
        MOCKS['hive'].reset()
        doc = {'policy': {'cloud_policy': 'explicit'},
               'upstreams': {'hive': {'profile': 'hivemind_gateway', 'base_url': MOCKS['hive'].api_base, 'enabled': True,
                                      'credential': 'hive-key', 'models': ['hive-chat']}}}
        self.full = W.FullApp(doc, api_key=BOOT, creds={'hive': 'hivemind_gateway'})
        self.http = self.full.http
        self.token = self.full.mint('inference', 'cloud', name='loop')[1]

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def chat(self, request_id):
        return self.http.post('/v1/chat/completions', json={'model': 'hive-chat@hive', 'messages': CHAT},
                              headers={'Authorization': 'Bearer ' + self.token, 'X-Request-ID': request_id})

    def test_a_request_sent_back_by_the_router_is_a_loop(self):
        from unittest import mock as umock
        from titanos.broker.router import ROUTER_INFLIGHT
        held = []
        real_add = ROUTER_INFLIGHT.add

        def add(request_id):
            held.append(request_id)
            real_add(request_id)
        with umock.patch.object(ROUTER_INFLIGHT, 'add', side_effect=add):
            outer = self.chat('outer-request-0001')
        self.assertEqual(outer.status_code, 200, outer.text)
        self.assertEqual(held, ['outer-request-0001'])  # held while the router upstream served it
        self.assertNotIn('outer-request-0001', ROUTER_INFLIGHT)  # released when the request ended
        self.assertEqual(MOCKS['hive'].hits[-1].headers.get('x-request-id'), 'outer-request-0001')
        # the router sends the same request back (no hop header, same X-Request-ID): refused on re-entry
        ROUTER_INFLIGHT.add('inflight-request-0002')
        try:
            before = len(MOCKS['hive'].hits)
            back = self.chat('inflight-request-0002')
            self.assertEqual((back.status_code, back.json()['error']['code']), (508, 'loop_detected'), back.text)
            self.assertEqual(len(MOCKS['hive'].hits), before)
        finally:
            ROUTER_INFLIGHT.remove('inflight-request-0002')


class LocalTarget(unittest.TestCase):
    """A local model as the target (keyless loopback, as a local router calls it)."""

    def setUp(self):
        netguard.reset_attempts()
        self.full = W.FullApp({'policy': {'cloud_policy': 'explicit'}, 'upstreams': {}}, api_key=None,
                              models={'local-chat': 'chat'})
        self.http = self.full.http

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def post(self, path, body, **headers):
        return self.http.post(path, json=body, headers=headers)

    def test_chat_accepts_the_standard_fields_and_names_them(self):
        answer = self.post('/v1/chat/completions', {
            'model': 'local-chat', 'messages': CHAT, 'metadata': {'user_id': 'u-1'}, 'user': 'u-1', 'store': False,
            'prompt_cache_key': 'pk', 'prompt_cache_retention': '24h', 'safety_identifier': 's', 'service_tier': 'auto',
            'top_logprobs': 0, 'reasoning_effort': 'low', 'stream_options': {'include_usage': True}})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()['choices'][0]['message']['content'], 'local hello')
        # top_logprobs is a schema field since the OpenAI-params work (0 is inert and served; a live value on a transformers model is a typed 400)
        self.assertEqual(answer.headers['x-titanos-ignored'], 'metadata, store, reasoning_effort, user, '
                         'safety_identifier, prompt_cache_key, prompt_cache_retention, service_tier')
        name, task, data = self.full.engine.calls[-1]
        self.assertFalse({'metadata', 'user', 'prompt_cache_key', 'reasoning_effort'} & set(data))
        streamed = self.post('/v1/chat/completions', {'model': 'local-chat', 'messages': CHAT, 'stream': True,
                                                      'metadata': {'user_id': 'u-1'}})
        self.assertEqual(streamed.status_code, 200, streamed.text)
        self.assertEqual(streamed.headers['x-titanos-ignored'], 'metadata')
        plain = self.post('/v1/chat/completions', {'model': 'local-chat', 'messages': CHAT})
        self.assertNotIn('x-titanos-ignored', plain.headers)

    def test_a_router_reentry_to_a_local_model_is_served(self):
        # a router route whose target is this instance's local model re-enters with the outer X-Request-ID: not a
        # loop (it never goes to a router again), so it is served
        from titanos.broker.router import ROUTER_INFLIGHT
        ROUTER_INFLIGHT.add('reentry-request-0003')
        try:
            answer = self.post('/v1/chat/completions', {'model': 'local-chat', 'messages': CHAT},
                               **{'X-Request-ID': 'reentry-request-0003'})
            self.assertEqual(answer.status_code, 200, answer.text)
        finally:
            ROUTER_INFLIGHT.remove('reentry-request-0003')

    def test_refusals_stay_typed(self):
        cases = [('/v1/chat/completions', {'model': 'local-chat', 'messages': CHAT, 'store': True}, 400),
                 ('/v1/chat/completions', {'model': 'local-chat', 'messages': CHAT, 'user': 5}, 400),
                 ('/v1/chat/completions', {'model': 'local-chat', 'messages': CHAT, 'bogus': 1}, 422),
                 ('/v1/responses', {'model': 'local-chat', 'input': 'hi', 'previous_response_id': 'resp_1'}, 400),
                 ('/v1/responses', {'model': 'local-chat', 'input': 'hi', 'background': True}, 400),
                 ('/v1/responses', {'model': 'local-chat', 'input': 'hi', 'store': True}, 400),
                 ('/v1/messages', {'model': 'local-chat', 'max_tokens': 8, 'messages': CHAT,
                                   'container': 'c-1'}, 400),
                 ('/v1/messages', {'model': 'local-chat', 'max_tokens': 8, 'messages': CHAT,
                                   'output_config': {'format': {'type': 'json_schema', 'schema': {}}}}, 400),
                 ('/v1/messages', {'model': 'local-chat', 'max_tokens': 8, 'messages': CHAT, 'bogus': 1}, 422)]
        for path, body, status in cases:
            with self.subTest(path=path, body=sorted(body)):
                answer = self.post(path, body)
                self.assertEqual(answer.status_code, status, answer.text)
                if status == 400 and 'user' not in body:
                    self.assertEqual(answer.json()['error']['code'], 'unsupported_capability')

    def test_responses_and_messages_accept_the_codex_and_claude_code_fields(self):
        answer = self.post('/v1/responses', {
            'model': 'local-chat', 'input': 'hi', 'store': False, 'include': ['reasoning.encrypted_content'],
            'reasoning': {'effort': 'low', 'summary': 'auto'}, 'text': {'verbosity': 'low'}, 'prompt_cache_key': 'pk',
            'truncation': 'disabled', 'user': 'u-1', 'service_tier': 'auto'})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()['object'], 'response')
        self.assertEqual(answer.headers['x-titanos-ignored'],
                         'text.verbosity, include, reasoning, truncation, user, prompt_cache_key, service_tier')
        message = self.post('/v1/messages', {
            'model': 'local-chat', 'max_tokens': 16, 'metadata': {'user_id': 'cc-user'}, 'service_tier': 'auto',
            'thinking': {'type': 'disabled'}, 'output_config': {'effort': 'low'},
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi',
                                                       'cache_control': {'type': 'ephemeral'}}]}]},
            **{'anthropic-version': '2023-06-01'})
        self.assertEqual(message.status_code, 200, message.text)
        self.assertEqual(message.json()['type'], 'message')
        self.assertEqual(message.headers['x-titanos-ignored'], 'metadata, service_tier, thinking, output_config')


if __name__ == '__main__':
    from broker_adapter_support import main
    main({})
