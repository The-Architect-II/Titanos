"""NVIDIA NeMo Switchyard integration, both directions, over the integrated app (create_app: KeylessHostGuard, Guard +
Authenticator, BrokerDispatch) against tests/broker_mocks/switchyard_mock.py (wire shapes transcribed from a real
switchyard-server 0.3.0 run). The netguard is installed and every test ends with zero non-loopback connection attempts.
Synthetic credentials only. The real-process interop (a real Switchyard and a real titanos serve) is
tests/check_switchyard_interop.py.

Titanos -> Switchyard (profile switchyard, kind router):
- the profile row: origin base_url http://127.0.0.1:4000 (no /v1), auth none by default (a credential needs auth bearer;
  broker.json with a credential and the default auth is refused), catalog and probe GET /v1/models, the hop header, the
  request id, the harness session headers on its row only; 127.0.0.1 is not a provider hostname;
- chat and Responses pass through (every field, streaming, include_usage injected and stripped back), the answer carries
  router:<id> provenance and the backend Switchyard selected (x-model-router-selected-model, recorded as a decision note);
  Responses get store=false unless the upstream allows store and the key holds cloud:store;
- Anthropic Messages and count_tokens pass through natively (thinking, cache_control, metadata and anthropic-beta reach
  Switchyard as sent; anthropic-version is set), streams keep their event lines; an upstream narrowed to openai.chat
  translates Anthropic onto chat again; Gemini and Ollama inbound are translated onto chat;
- the live catalog keeps the per-route tool_calling, vision and context window: tools or images to a route declaring
  them false are refused before egress; Studio sees capabilities and context_length;
- errors: an unknown route is 404 model_not_found, 429 keeps Retry-After, 503 is upstream_unavailable;
- the native lane reaches /v1/responses/compact, /v1/responses/input_tokens, /v1/decision and /health; the operator
  surfaces (stats, stats reset, metrics) need broker:admin;
- loops: an inbound hop keeps the request local (400 location_unsatisfiable, zero egress) and this instance's own hop is
  508 loop_detected; Switchyard drops X-Titanos-Hop but forwards X-Request-ID, so a request whose X-Request-ID this
  instance is currently sending to a router upstream is 508 loop_detected as well when it would go to a router again
  (zero egress), a re-entry to a local model is served, and the id is released when the outer request ends;
- connect: POST /api/broker/connect {provider: switchyard} creates the upstream without a credential and discovers the
  route ids.

Switchyard -> Titanos (Titanos as a Switchyard target, local models):
- the standard identity, cache, storage and reasoning fields Claude Code, Codex and Switchyard send are accepted on
  /v1/chat/completions, /v1/responses and /v1/messages and named in X-Titanos-Ignored; store=true,
  previous_response_id, background, Anthropic container and output_config.format answer a typed 400; unknown fields
  still answer 422.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import OpenAIMock, SENTINELS, netguard  # noqa: E402
from broker_mocks.switchyard_mock import SwitchyardMock  # noqa: E402

BOOT = 'switchyard-check-bootstrap-0001'
MOCKS: dict = {}
SESSION = {'X-Switchyard-Session-Id': 'sess-0001', 'X-Claude-Code-Session-Id': 'cc-0001',
           'X-Codex-Turn-Metadata': '{"session_id":"cx-1"}', 'X-OpenAI-Subagent': 'review', 'Session-Id': 'gen-1'}
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
TOOL = {'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {}}}}
IMAGE = [{'role': 'user', 'content': [{'type': 'text', 'text': 'what is it?'},
                                      {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}}]}]


def setUpModule():
    MOCKS['sy'] = SwitchyardMock()
    MOCKS['sy_bearer'] = SwitchyardMock(credential=SENTINELS['openai'], require_auth=True)
    MOCKS['openai'] = OpenAIMock()
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def document(*, surfaces=None, allow_store=False, anonymous_cloud=False, bearer_upstream=False) -> dict:
    sy = {'profile': 'switchyard', 'base_url': MOCKS['sy'].base_url, 'enabled': True, 'catalog': 'live'}
    if surfaces is not None:
        sy['surfaces'] = surfaces
    if allow_store:
        sy['data_policy'] = {'allow_store': True}
    upstreams = {'sy': sy}
    if bearer_upstream:
        upstreams['syb'] = {'profile': 'switchyard', 'base_url': MOCKS['sy_bearer'].base_url, 'enabled': True,
                            'auth': 'bearer', 'credential': 'syb-key', 'models': ['switchyard/general']}
    return {'policy': {'cloud_policy': 'explicit', 'anonymous_cloud': anonymous_cloud}, 'upstreams': upstreams}


def bearer(token: str) -> dict:
    return {'Authorization': 'Bearer ' + token}


class Base(unittest.TestCase):
    keyed = True
    doc_kw: dict = {}
    models: dict | None = None

    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
        self.full = W.FullApp(document(**self.doc_kw), api_key=BOOT if self.keyed else None, models=self.models,
                              creds={'syb': 'openai'})
        self.http = self.full.http
        if self.keyed:
            self.token = self.full.mint('inference', 'cloud', 'models:read', 'native', name='sy-user')[1]
        self.refresh()

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def refresh(self):
        if self.keyed:
            answer = self.http.get('/api/broker/catalog?refresh=true', headers=bearer(BOOT))
            self.assertEqual(answer.status_code, 200, answer.text)

    def auth(self) -> dict:
        return bearer(self.token) if self.keyed else {}

    def chat(self, model='switchyard/general@sy', **body):
        return self.http.post('/v1/chat/completions', json={'model': model, 'messages': CHAT, **body},
                              headers={**self.auth(), **SESSION})

    def decision(self, answer) -> dict:
        principal = self.full.broker.authenticator.bootstrap_principal()
        found = [d for d in self.full.broker.decisions.query(limit=50, principal=principal)
                 if d.get('decision_id') == answer.headers.get('x-titanos-decision-id')]
        self.assertTrue(found, 'no decision recorded')
        return found[0]

    def sy_hits(self, surface=None):
        return MOCKS['sy'].hits_for(surface)


class ProfileRow(unittest.TestCase):
    def test_row_and_config_rules(self):
        from titanos.broker import profiles
        from titanos.broker.config import Upstream
        row = profiles.profile('switchyard')
        self.assertEqual((row.kind, row.adapter, row.auth, row.auth_optional, row.default_base_url),
                         ('router', 'openai_compat', 'none', True, 'http://127.0.0.1:4000'))
        self.assertEqual(dict(row.surfaces), {'openai.chat': '/v1/chat/completions', 'openai.responses': '/v1/responses',
                                              'anthropic.messages': '/v1/messages',
                                              'anthropic.count_tokens': '/v1/messages/count_tokens'})
        self.assertEqual((row.catalog_path, row.probe_path), ('/v1/models', '/v1/models'))
        self.assertTrue(row.hop_header and row.extras['forward_request_id'] and row.operator_surfaces)
        self.assertEqual(dict(row.extras['surface_adapters']),
                         {'anthropic.messages': 'anthropic', 'anthropic.count_tokens': 'anthropic'})
        self.assertIn('x-switchyard-session-id', row.extras['inbound_headers'])
        self.assertFalse({'authorization', 'x-api-key', 'cookie'} & set(row.extras['inbound_headers']))
        self.assertFalse(any(h.startswith('x-titanos-') for h in row.extras['inbound_headers']))
        self.assertEqual(row.env_names, ())
        self.assertNotIn('127.0.0.1', profiles.hostnames())
        self.assertIn('/v1/decision', {r.path for r in row.native.routes})
        self.assertTrue(all(r.admin for r in row.native.routes if r.family == 'operator' and r.path != '/health'))
        self.assertEqual(Upstream(profile='switchyard').effective_base_url(), 'http://127.0.0.1:4000')
        self.assertEqual(Upstream(profile='switchyard').effective_auth(), 'none')
        with self.assertRaises(ValueError):  # a credential needs auth bearer
            Upstream(profile='switchyard', credential='sy-key')
        self.assertEqual(Upstream(profile='switchyard', auth='bearer', credential='sy-key').effective_auth(), 'bearer')
        with self.assertRaises(ValueError):
            Upstream(profile='switchyard', auth='azure_api_key')

    def test_catalog_metadata_parser(self):
        from titanos.broker.adapters.split import catalog_meta
        meta = catalog_meta({'data': [
            {'id': 'a', 'capabilities': {'tool_calling': True, 'vision': None, 'context_window': 32768}},
            {'id': 'b', 'context_length': 8192, 'capabilities': {'vision': False}},
            {'id': 'c', 'capabilities': {'tool_calling': 'yes', 'context_window': -1}}, {'nope': 1}]})
        self.assertEqual(meta['a'], {'tool_calling': True, 'vision': None, 'streaming': None, 'context_length': 32768})
        self.assertEqual(meta['b']['context_length'], 8192)
        self.assertIs(meta['b']['vision'], False)
        self.assertEqual(meta['c'], {'tool_calling': None, 'vision': None, 'streaming': None, 'context_length': None})
        self.assertEqual(sorted(meta), ['a', 'b', 'c'])


class OpenAISurfaces(Base):
    def test_chat_passthrough_headers_and_provenance(self):
        answer = self.http.post('/v1/chat/completions', json={
            'model': 'switchyard/general@sy', 'messages': CHAT, 'metadata': {'user_id': 'u-1'}, 'user': 'u-1',
            'prompt_cache_key': 'pk', 'reasoning_effort': 'low', 'worker': 'w1'},
            headers={**self.auth(), **SESSION, 'X-Request-ID': 'req-sy-0001', 'Cookie': 'a=b',
                     'X-Titanos-Location': 'auto', 'User-Agent': 'claude-cli/2.1.200'})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()['model'], 'mock-weak')
        self.assertEqual(answer.headers['x-titanos-served-by'], 'router:sy')
        self.assertEqual(answer.headers['x-titanos-translation'], 'passthrough')
        self.assertEqual(answer.headers['x-model-router-selected-model'], 'mock-weak')
        hit = self.sy_hits('openai.chat')[-1]
        self.assertEqual((hit.method, hit.path), ('POST', '/v1/chat/completions'))
        self.assertEqual(hit.credentials, [])  # auth none: no credential of any kind, never the client's tk_ key
        self.assertFalse(hit.presented(self.token))
        self.assertEqual(hit.headers['x-titanos-hop'], self.full.broker.instance_hop())
        self.assertEqual(hit.headers['x-request-id'], 'req-sy-0001')
        for name, value in SESSION.items():
            self.assertEqual(hit.headers[name.lower()], value)
        self.assertEqual(hit.headers['user-agent'], 'claude-cli/2.1.200')
        self.assertNotIn('cookie', hit.headers)
        self.assertFalse([h for h in hit.headers if h.startswith('x-titanos-') and h != 'x-titanos-hop'])
        sent = hit.json()
        self.assertEqual(sent['model'], 'switchyard/general')
        self.assertEqual((sent['metadata'], sent['user'], sent['prompt_cache_key'], sent['reasoning_effort']),
                         ({'user_id': 'u-1'}, 'u-1', 'pk', 'low'))
        self.assertNotIn('worker', sent)
        self.assertIn('router_selected:mock-weak', self.decision(answer)['notes'])

    def test_chat_stream_include_usage_injected_and_stripped(self):
        with self.http.stream('POST', '/v1/chat/completions', json={'model': 'switchyard/general@sy', 'messages': CHAT,
                                                                    'stream': True}, headers=self.auth()) as answer:
            self.assertEqual(answer.status_code, 200)
            body = b''.join(answer.iter_bytes()).decode()
        self.assertIn('"content":"Hi"', body.replace(' ', ''))
        self.assertTrue(body.rstrip().endswith('data: [DONE]'), body[-200:])
        self.assertNotIn('"usage"', body)  # the client did not ask for usage: the injected chunk is stripped
        self.assertEqual(self.sy_hits('openai.chat')[-1].json()['stream_options'], {'include_usage': True})
        with self.http.stream('POST', '/v1/chat/completions', json={
                'model': 'switchyard/general@sy', 'messages': CHAT, 'stream': True,
                'stream_options': {'include_usage': True}}, headers=self.auth()) as answer:
            body = b''.join(answer.iter_bytes()).decode()
        self.assertIn('"usage"', body)

    def test_responses_store_forced_false_and_stream_events(self):
        answer = self.http.post('/v1/responses', json={'model': 'switchyard/codex@sy', 'input': 'hi',
                                                       'include': ['reasoning.encrypted_content']}, headers=self.auth())
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()['model'], 'mock-codex')
        sent = self.sy_hits('openai.responses')[-1].json()
        self.assertIs(sent['store'], False)
        self.assertEqual(sent['include'], ['reasoning.encrypted_content'])
        self.assertIn('forced:store=false', self.decision(answer)['notes'])
        with self.http.stream('POST', '/v1/responses', json={'model': 'switchyard/codex@sy', 'input': 'hi',
                                                             'stream': True}, headers=self.auth()) as streamed:
            text = b''.join(streamed.iter_bytes()).decode()
        events = [line[len('event: '):] for line in text.splitlines() if line.startswith('event: ')]
        self.assertEqual(events[0], 'response.created')
        self.assertEqual(events[-1], 'response.completed')
        self.assertIn('response.output_text.delta', events)

    def test_errors_keep_their_meaning(self):
        missing = self.chat('no-such-route@sy')
        self.assertEqual(missing.status_code, 404, missing.text)  # allow_any router: Switchyard answers, mapped
        self.assertEqual(missing.json()['error']['code'], 'model_not_found')
        MOCKS['sy'].faults.push('status', status=429, retry_after=7)
        limited = self.chat()
        self.assertEqual(limited.status_code, 429, limited.text)
        self.assertEqual(limited.headers.get('retry-after'), '7')
        before = len(MOCKS['sy'].hits)
        held = self.chat()  # that route's breaker now waits out Retry-After, with zero egress, answered as a 429
        self.assertEqual((held.status_code, held.json()['error']['code']), (429, 'upstream_rate_limited'))
        self.assertIsNotNone(held.headers.get('retry-after'))
        self.assertEqual(len(MOCKS['sy'].hits), before)
        other = self.chat('switchyard/codex@sy')  # one breaker per route: the other routes keep serving
        self.assertEqual(other.status_code, 200, other.text)
        MOCKS['sy'].faults.push('status', status=503, code='temporarily_unavailable')
        busy = self.chat('switchyard/claude@sy')
        self.assertEqual(busy.status_code, 503, busy.text)
        self.assertEqual(busy.json()['error']['code'], 'upstream_unavailable')
        listing = {row['id']: row['titanos']['status'] for row in
                   self.http.get('/v1/models', headers=self.auth()).json()['data'] if row.get('owned_by') == 'sy'}
        self.assertEqual((listing['switchyard/general@sy'], listing['switchyard/codex@sy']),
                         ('breaker_open', 'available'))
        cooling = {s.get('model'): s['retry_after'] for s in self.full.broker.breakers.snapshot() if s['upstream'] == 'sy'}
        self.assertGreater(cooling['switchyard/general'], 0)  # the snapshot names the route whose breaker waits
        self.assertEqual(cooling.get('switchyard/codex', 0), 0)


class StoreAllowed(Base):
    doc_kw = {'allow_store': True}

    def test_store_passes_with_allow_store_and_the_scope(self):
        token = self.full.mint('inference', 'cloud', 'cloud:store', name='storer')[1]
        answer = self.http.post('/v1/responses', json={'model': 'switchyard/codex@sy', 'input': 'hi', 'store': True},
                                headers=bearer(token))
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertIs(self.sy_hits('openai.responses')[-1].json()['store'], True)


class AnthropicSurfaces(Base):
    def test_messages_pass_through_natively(self):
        body = {'model': 'switchyard/claude@sy', 'max_tokens': 64, 'metadata': {'user_id': 'u-1'},
                'thinking': {'type': 'enabled', 'budget_tokens': 1024}, 'system': [
                    {'type': 'text', 'text': 'Be brief.', 'cache_control': {'type': 'ephemeral'}}],
                'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi',
                                                           'cache_control': {'type': 'ephemeral'}}]}]}
        answer = self.http.post('/v1/messages', json=body, headers={
            'x-api-key': self.token, 'anthropic-version': '2023-06-01', 'anthropic-beta': 'interleaved-thinking-2025-05-14',
            **SESSION})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()['type'], 'message')
        self.assertEqual(answer.json()['model'], 'mock-strong')
        self.assertEqual(answer.headers['x-titanos-translation'], 'passthrough')
        self.assertEqual(answer.headers['x-model-router-selected-model'], 'mock-strong')
        hit = self.sy_hits('anthropic.messages')[-1]
        self.assertEqual(hit.path, '/v1/messages')
        self.assertEqual(hit.credentials, [])  # the client's x-api-key (a tk_ key) never leaves
        self.assertFalse(hit.presented(self.token))
        self.assertEqual(hit.headers['anthropic-version'], '2023-06-01')
        self.assertEqual(hit.headers['anthropic-beta'], 'interleaved-thinking-2025-05-14')
        self.assertEqual(hit.headers['x-switchyard-session-id'], 'sess-0001')
        self.assertEqual(hit.headers['x-titanos-hop'], self.full.broker.instance_hop())
        sent = hit.json()
        self.assertEqual(sent, dict(body, model='switchyard/claude'))  # every field as sent, nothing re-encoded
        self.assertEqual(self.sy_hits('openai.chat'), [])

    def test_messages_stream_and_count_tokens(self):
        with self.http.stream('POST', '/v1/messages', json={'model': 'switchyard/claude@sy', 'max_tokens': 32,
                                                            'stream': True, 'messages': CHAT},
                              headers={'x-api-key': self.token}) as answer:
            self.assertEqual(answer.status_code, 200)
            text = b''.join(answer.iter_bytes()).decode()
        events = [line[len('event: '):] for line in text.splitlines() if line.startswith('event: ')]
        self.assertEqual(events, ['message_start', 'content_block_start', 'content_block_delta', 'content_block_stop',
                                  'message_delta', 'message_stop'])
        counted = self.http.post('/v1/messages/count_tokens', json={'model': 'switchyard/claude@sy', 'messages': CHAT},
                                 headers={'x-api-key': self.token})
        self.assertEqual((counted.status_code, counted.json()), (200, {'input_tokens': 7}))
        self.assertEqual(self.sy_hits('anthropic.count_tokens')[-1].path, '/v1/messages/count_tokens')
        refused = self.http.post('/v1/messages/count_tokens', json={'model': 'switchyard/general@sy',
                                                                    'messages': CHAT}, headers={'x-api-key': self.token})
        self.assertEqual(refused.status_code, 400, refused.text)
        self.assertEqual(refused.json()['type'], 'error')  # the Anthropic envelope, as Switchyard rendered it

    def test_other_protocols_translate_onto_chat(self):
        answer = self.http.post('/api/chat', json={'model': 'switchyard/general@sy', 'messages': CHAT, 'stream': False},
                                headers=self.auth())
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.headers['x-titanos-translation'], 'ollama->openai')
        self.assertEqual(answer.json()['message']['content'], 'Hi')
        gemini = self.http.post('/v1beta/models/switchyard/general@sy:generateContent',
                                json={'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]}, headers=self.auth())
        self.assertEqual(gemini.status_code, 200, gemini.text)
        self.assertEqual(gemini.headers['x-titanos-translation'], 'gemini->openai')
        self.assertEqual(len(self.sy_hits('openai.chat')), 2)


class NarrowedSurfaces(Base):
    doc_kw = {'surfaces': ['openai.chat']}

    def test_anthropic_is_translated_when_the_operator_narrows_the_row(self):
        answer = self.http.post('/v1/messages', json={'model': 'switchyard/general@sy', 'max_tokens': 32,
                                                      'messages': CHAT}, headers={'x-api-key': self.token})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.headers['x-titanos-translation'], 'anthropic->openai')
        self.assertEqual(self.sy_hits('anthropic.messages'), [])
        self.assertEqual(len(self.sy_hits('openai.chat')), 1)
        refused = self.http.post('/v1/responses', json={'model': 'switchyard/general@sy', 'input': 'hi'},
                                 headers=self.auth())
        self.assertEqual(refused.status_code, 200, refused.text)  # Responses translated onto chat as well
        self.assertEqual(refused.headers['x-titanos-translation'], 'openai->openai')


class Catalog(Base):
    def test_route_capabilities_reach_the_catalog_and_refuse_before_egress(self):
        listing = self.http.get('/v1/models', headers=bearer(self.token)).json()['data']
        rows = {row['id']: row for row in listing if row.get('owned_by') == 'sy'}
        self.assertEqual(sorted(rows), ['switchyard/claude@sy', 'switchyard/codex@sy', 'switchyard/general@sy',
                                        'switchyard@sy'])
        general = rows['switchyard/general@sy']['titanos']
        self.assertEqual(general['class'], 'router')
        self.assertIn('tools', general['capabilities'])
        self.assertNotIn('vision', general['capabilities'])
        self.assertEqual(general['context_length'], 32768)
        self.assertIn('vision', rows['switchyard/codex@sy']['titanos']['capabilities'])
        self.assertNotIn('context_length', rows['switchyard@sy']['titanos'])
        before = len(MOCKS['sy'].hits)
        image = self.http.post('/v1/chat/completions', json={'model': 'switchyard/general@sy', 'messages': IMAGE},
                               headers=self.auth())
        self.assertEqual(image.status_code, 400, image.text)
        self.assertEqual(image.json()['error']['code'], 'broker_capability_unavailable')
        self.assertEqual(len(MOCKS['sy'].hits), before)  # refused with zero egress
        routes = dict(MOCKS['sy'].routes_table)
        try:
            MOCKS['sy'].routes_table['switchyard/general'] = ('mock-weak', False, False, 32768, False, False)
            self.refresh()
            tools = self.http.post('/v1/chat/completions', json={'model': 'switchyard/general@sy', 'messages': CHAT,
                                                                 'tools': [TOOL]}, headers=self.auth())
            self.assertEqual(tools.status_code, 400, tools.text)
            self.assertEqual(tools.json()['error']['code'], 'broker_capability_unavailable')
        finally:
            MOCKS['sy'].routes_table = routes
        fine = self.http.post('/v1/chat/completions', json={'model': 'switchyard/codex@sy', 'messages': IMAGE},
                              headers=self.auth())
        self.assertEqual(fine.status_code, 200, fine.text)


class NativeLane(Base):
    def test_auxiliary_and_operator_endpoints(self):
        compact = self.http.post('/native/sy/v1/responses/compact', json={'model': 'switchyard/codex', 'input': 'hi'},
                                 headers={**self.auth(), **SESSION})
        self.assertEqual(compact.status_code, 200, compact.text)
        self.assertEqual(compact.json()['object'], 'response.compaction')
        self.assertEqual(compact.headers['x-model-router-selected-model'], 'mock-codex')
        hit = self.sy_hits('switchyard.compact')[-1]
        self.assertEqual(hit.headers['x-switchyard-session-id'], 'sess-0001')
        self.assertEqual(hit.credentials, [])
        tokens = self.http.post('/native/sy/v1/responses/input_tokens', json={'model': 'switchyard/codex',
                                                                             'input': 'hi'}, headers=self.auth())
        self.assertEqual(tokens.json(), {'object': 'response.input_tokens', 'input_tokens': 11})
        unsupported = self.http.post('/native/sy/v1/responses/compact', json={'model': 'switchyard/general',
                                                                             'input': 'hi'}, headers=self.auth())
        self.assertEqual(unsupported.status_code, 400)
        self.assertEqual(unsupported.json()['error']['code'], 'responses_auxiliary_unsupported')
        decision = self.http.post('/native/sy/v1/decision', json={
            'input_format': 'openai_chat', 'request': {'model': 'switchyard/general', 'messages': CHAT}},
            headers=self.auth())
        self.assertEqual(decision.json()['selected']['model'], 'mock-weak')
        self.assertEqual(self.http.get('/native/sy/health', headers=self.auth()).json(), {'status': 'ok'})
        before = len(MOCKS['sy'].hits)
        for method, path in (('GET', '/native/sy/v1/stats'), ('POST', '/native/sy/v1/stats/reset'),
                             ('GET', '/native/sy/metrics')):
            refused = self.http.request(method, path, headers=self.auth())
            self.assertEqual(refused.status_code, 403, (path, refused.text))
        self.assertEqual(len(MOCKS['sy'].hits), before)
        self.assertEqual(self.http.get('/native/sy/v1/stats', headers=bearer(BOOT)).status_code, 200)
        self.assertIn('switchyard_build_info', self.http.get('/native/sy/metrics', headers=bearer(BOOT)).text)
        reset = self.http.post('/native/sy/v1/stats/reset', headers=bearer(BOOT))
        self.assertEqual((reset.status_code, MOCKS['sy'].reset_count), (200, 1))
        refused = self.http.post('/native/sy/v1/not-allowlisted', json={}, headers=self.auth())
        self.assertEqual(refused.status_code, 400)


class Hops(Base):
    def test_inbound_hop_stays_local_and_own_hop_is_a_loop(self):
        before = len(MOCKS['sy'].hits)
        foreign = self.http.post('/v1/chat/completions', json={'model': 'switchyard/general@sy', 'messages': CHAT},
                                 headers={**self.auth(), 'X-Titanos-Hop': 'switchyard'})
        self.assertEqual(foreign.status_code, 400, foreign.text)
        self.assertEqual(foreign.json()['error']['code'], 'location_unsatisfiable')
        own = self.http.post('/v1/chat/completions', json={'model': 'switchyard/general@sy', 'messages': CHAT},
                             headers={**self.auth(), 'X-Titanos-Hop': self.full.broker.instance_hop()})
        self.assertEqual(own.status_code, 508, own.text)
        self.assertEqual(own.json()['error']['code'], 'loop_detected')
        self.assertEqual(len(MOCKS['sy'].hits), before)

    def test_a_request_sent_back_by_the_router_is_a_loop(self):
        from unittest import mock as umock
        from titanos.broker.router import ROUTER_INFLIGHT
        held = []
        real_add = ROUTER_INFLIGHT.add

        def add(request_id):
            held.append(request_id)
            real_add(request_id)
        with umock.patch.object(ROUTER_INFLIGHT, 'add', side_effect=add):
            outer = self.http.post('/v1/chat/completions', json={'model': 'switchyard/general@sy', 'messages': CHAT},
                                   headers={**self.auth(), 'X-Request-ID': 'outer-request-0001'})
        self.assertEqual(outer.status_code, 200, outer.text)
        self.assertEqual(held, ['outer-request-0001'])  # held while the router upstream served it
        self.assertNotIn('outer-request-0001', ROUTER_INFLIGHT)  # released when the request ended
        self.assertEqual(MOCKS['sy'].hits_for('openai.chat')[-1].headers.get('x-request-id'), 'outer-request-0001')
        # the router sends the same request back (no hop header, same X-Request-ID): refused on re-entry
        ROUTER_INFLIGHT.add('inflight-request-0002')
        try:
            before = len(MOCKS['sy'].hits)
            back = self.http.post('/v1/chat/completions', json={'model': 'switchyard/general@sy', 'messages': CHAT},
                                  headers={**self.auth(), 'X-Request-ID': 'inflight-request-0002'})
            self.assertEqual((back.status_code, back.json()['error']['code']), (508, 'loop_detected'), back.text)
            self.assertEqual(len(MOCKS['sy'].hits), before)
        finally:
            ROUTER_INFLIGHT.remove('inflight-request-0002')


class BearerUpstream(Base):
    doc_kw = {'bearer_upstream': True}

    def test_auth_bearer_sends_the_bound_credential_only_there(self):
        answer = self.chat('switchyard/general@syb')
        self.assertEqual(answer.status_code, 200, answer.text)
        hit = MOCKS['sy_bearer'].hits_for('openai.chat')[-1]
        self.assertEqual(hit.credential, 'Bearer <sentinel:openai>')
        self.assertFalse(hit.presented(self.token))
        self.assertEqual(self.sy_hits('openai.chat'), [])
        from titanos.broker.config import BrokerConfig
        with self.assertRaises(ValueError):
            BrokerConfig.model_validate({'upstreams': {'x': {'profile': 'switchyard', 'credential': 'x-key'}}})


class Connect(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        MOCKS['sy'].reset()
        self.full = W.FullApp({'policy': {'cloud_policy': 'off'}, 'upstreams': {}}, api_key=BOOT)

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def test_connect_creates_the_router_without_a_credential(self):
        answer = self.full.http.post('/api/broker/connect', json={'provider': 'switchyard',
                                                                  'base_url': MOCKS['sy'].base_url},
                                     headers=bearer(BOOT))
        self.assertEqual(answer.status_code, 200, answer.text)
        data = answer.json()
        self.assertEqual((data['upstream']['id'], data['upstream']['profile'], data['upstream']['credential']),
                         ('switchyard', 'switchyard', None))
        self.assertTrue(data['upstream']['enabled'])
        self.assertEqual(data['validation'], 'not_needed')
        self.assertEqual(data['cloud_policy']['file'], 'explicit')
        self.assertEqual(sorted(data['models']), ['switchyard', 'switchyard/claude', 'switchyard/codex',
                                                 'switchyard/general'])
        hits = MOCKS['sy'].hits_for('switchyard.models')
        self.assertTrue(hits and all(hit.credentials == [] for hit in hits))


class LocalTarget(Base):
    """Switchyard -> Titanos: a local model as a Switchyard target (keyless loopback, as switchyard-server calls it)."""
    keyed = False
    models = {'local-chat': 'chat'}

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
        # a Switchyard route whose target is this instance's local model re-enters with the outer X-Request-ID: not a
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


CELLS = {
    'openai.chat|SY|P': 'OpenAISurfaces.test_chat_passthrough_headers_and_provenance',
    'openai.chat|SY|P|stream': 'OpenAISurfaces.test_chat_stream_include_usage_injected_and_stripped',
    'openai.responses|SY|P': 'OpenAISurfaces.test_responses_store_forced_false_and_stream_events',
    'openai.responses|SY|P|stream': 'OpenAISurfaces.test_responses_store_forced_false_and_stream_events',
    'anthropic.messages|SY|P': 'AnthropicSurfaces.test_messages_pass_through_natively',
    'anthropic.messages|SY|P|stream': 'AnthropicSurfaces.test_messages_stream_and_count_tokens',
    'anthropic.count_tokens|SY|P': 'AnthropicSurfaces.test_messages_stream_and_count_tokens',
}

if __name__ == '__main__':
    from broker_adapter_support import main
    main(CELLS)
