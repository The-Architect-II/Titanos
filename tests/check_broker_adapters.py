"""Broker provider adapters (WP5): contract tests per profile row against the WP4 loopback mocks.

Covers, for every adapter: the auth header style (and never a query credential), paths and queries, the model rewrite,
Titanos-only field stripping, the ParamMap (reasoning-model rewrite, Mistral random_seed and drops, Groq refusals before
egress, NVIDIA input_type), Mistral tool-id remapping round trip, include_usage injection, catalog parsing ({data} and
bare arrays, Anthropic and Gemini paging), probes, error mapping, usage extraction (ElevenLabs character-cost included)
and invoke_sync parity with the async invoke. Every passthrough (P) cell of the translation matrix has a test here.

Fidelity: the exact public URLs, auth headers and required headers of the live-lane rows (OpenAI, the NVIDIA API catalog
and the Hugging Face router) and of Anthropic, Gemini, ElevenLabs, Ollama and Azure are asserted on the request the
EgressClient would send (captured by an httpx MockTransport; nothing leaves the process). Azure reaches the loopback
AzureMock through the test-only AzureHostTransport (ratified R12). R11: Upstream.auth selects custom_openai's scheme.
Source assertion: no adapter or codec imports config, vault or os.environ.
"""
import ast
import dataclasses
import json
import unittest
from pathlib import Path

import broker_adapter_support as S  # installs the netguard first
from broker_mocks import SENTINELS, OpenAIMock, netguard

import httpx

from titanos.broker.adapters.base import RawRequest
from titanos.broker.adapters.common import AdapterBase
from titanos.broker.adapters.elevenlabs import character_usage
from titanos.broker.adapters.gemini import GeminiAdapter
from titanos.broker.adapters.openai_compat import (INCLUDE_USAGE_SURFACES, OpenAICompatAdapter, ToolIdMap,
                                                   inject_include_usage, parse_catalog, rewrite_body)
from titanos.broker.config import Upstream
from titanos.broker.errors import BrokerError
from titanos.broker.profiles import AUTH_TEMPLATES, PROFILES
from titanos.broker.surfaces import SURFACES
from titanos.broker.translate import anthropic_messages, openai_chat
from titanos.broker.translate import gemini as gemini_codec
from titanos.broker.translate.ir import IRMessage, IRRequest, Text
from titanos.broker.translate.upstream_errors import UpstreamFailure
from titanos.broker.types import Usage

ROOT = Path(__file__).resolve().parents[1]
BROKER = ROOT / 'src' / 'titanos' / 'broker'
WAV = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data' \
      b'\x00\x00\x00\x00'


def raw(body=None, *, form=None, path_params=None, query=None, headers=None, method='POST'):
    return RawRequest(method=method, path_params=path_params or {}, query=query or {}, headers=headers or {}, json=body,
                      form=form)


async def drain(result):
    chunks = [chunk async for chunk in result.body]
    await result.aclose()
    return b''.join(chunks), await result.usage


class Case(unittest.TestCase):
    columns = ('OAI', 'AZ', 'ANT', 'GEM', 'OLC', 'XI')

    @classmethod
    def setUpClass(cls):
        cls.targets = S.start_targets(cls.columns)

    @classmethod
    def tearDownClass(cls):
        S.stop_targets(cls.targets)

    def setUp(self):
        netguard.reset_attempts()
        for target in self.targets.values():
            target.mock.reset()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def passthrough(self, column, surface, raw_request, *, model=None, notes=None, **kw):
        target = self.targets[column]

        async def go():
            async with target.egress() as egress:
                result = await target.adapter.passthrough(target.grant(model), egress, SURFACES[surface], raw_request,
                                                          notes=notes, **kw)
                data, usage = await drain(result)
                return result, data, usage
        return S.run(go())

    def sync_egress(self, target):
        client = target.egress().client
        self.addCleanup(client.close)
        return client

    def last(self, column):
        hits = self.targets[column].mock.hits
        self.assertTrue(hits, 'the mock saw no request')
        return hits[-1]


class OpenAIPassthrough(Case):
    columns = ('OAI',)

    def test_chat_auth_path_model_rewrite_and_stripping(self):
        notes = []
        body = {'model': 'my-alias', 'messages': [{'role': 'user', 'content': 'hi'}], 'worker': 'gpu0',
                'location': 'cloud_only', 'timeout': 5, 'logprobs': True}
        result, data, usage = self.passthrough('OAI', 'openai.chat', raw(body, headers={'openai-beta': 'x'}),
                                               notes=notes)
        hit = self.last('OAI')
        self.assertEqual((hit.method, hit.path), ('POST', '/v1/chat/completions'))
        self.assertEqual(hit.credential, 'Bearer <sentinel:openai>')
        self.assertEqual(hit.query_pairs, [])
        sent = hit.json()
        self.assertEqual(sent['model'], 'gpt-4o-mini')
        self.assertTrue(sent['logprobs'])  # native fields pass through on P paths
        self.assertFalse({'worker', 'location', 'timeout'} & set(sent))
        self.assertEqual(hit.headers.get('openai-beta'), 'x')
        self.assertEqual(json.loads(data)['choices'][0]['message']['content'], 'Hello! How can I assist you today?')
        self.assertEqual((usage.input_tokens, usage.output_tokens), (19, 10))
        self.assertEqual(result.upstream_request_id, None if 'x-request-id' not in result.headers
                         else result.headers['x-request-id'])

    def test_chat_stream_include_usage_injected_and_stripped(self):
        body = {'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True}
        notes = []
        result, data, usage = self.passthrough('OAI', 'openai.chat', raw(body), notes=notes)
        self.assertEqual(self.last('OAI').json()['stream_options'], {'include_usage': True})
        self.assertIn('injected:stream_options.include_usage', notes)
        events = S.sse_events(data)
        self.assertEqual(events[-1][1], '[DONE]')
        self.assertFalse(any(json.loads(p).get('choices') == [] for _, p in events[:-1]))
        self.assertIsNotNone(usage.input_tokens)
        self.assertIn('text/event-stream', result.content_type)
        # a client that asked for usage keeps the usage chunk
        body['stream_options'] = {'include_usage': True}
        _, data, _ = self.passthrough('OAI', 'openai.chat', raw(body))
        self.assertTrue(any(json.loads(p).get('choices') == [] for _, p in S.sse_events(data)[:-1]))

    def test_completions(self):
        _, data, usage = self.passthrough('OAI', 'openai.completions', raw({'model': 'a', 'prompt': 'Say this'}),
                                          model='gpt-3.5-turbo-instruct')
        self.assertEqual(self.last('OAI').path, '/v1/completions')
        self.assertEqual(json.loads(data)['object'], 'text_completion')
        self.assertEqual(usage.output_tokens, 7)

    def test_completions_and_responses_streams_pass_through(self):
        """Passthrough streams of the two other json:stream surfaces: completions gets include_usage injected and its
        usage chunk stripped again (the client did not ask), ending with [DONE]; Responses streams its typed event
        family with store forced false and increasing sequence numbers."""
        notes = []
        result, data, usage = self.passthrough('OAI', 'openai.completions',
                                               raw({'model': 'a', 'prompt': 'Say this', 'stream': True}),
                                               model='gpt-3.5-turbo-instruct', notes=notes)
        sent = self.last('OAI').json()
        self.assertEqual((sent['stream'], sent['stream_options']), (True, {'include_usage': True}))
        self.assertIn('injected:stream_options.include_usage', notes)
        events = S.sse_events(data)
        self.assertEqual(events[-1][1], '[DONE]')
        self.assertTrue(all(json.loads(p)['object'] == 'text_completion' for _, p in events[:-1]))
        self.assertFalse(any(json.loads(p).get('choices') == [] for _, p in events[:-1]))
        self.assertIsNotNone(usage.output_tokens)
        self.assertIn('text/event-stream', result.content_type)
        notes = []
        result, data, usage = self.passthrough('OAI', 'openai.responses',
                                               raw({'model': 'a', 'input': 'hi', 'stream': True}), notes=notes)
        self.assertIs(self.last('OAI').json()['store'], False)
        events = S.sse_events(data)
        names = [name for name, _ in events]
        self.assertEqual((names[0], names[-1]), ('response.created', 'response.completed'))
        sequence = [json.loads(p)['sequence_number'] for _, p in events]
        self.assertEqual(sequence, sorted(sequence))
        self.assertIsNotNone(usage.input_tokens)

    def test_responses_store_forced_false_and_handle_bound_routes(self):
        notes = []
        _, data, usage = self.passthrough('OAI', 'openai.responses', raw({'model': 'a', 'input': 'hi'}), notes=notes)
        self.assertIs(self.last('OAI').json()['store'], False)
        self.assertIn('forced:store=false', notes)
        self.assertEqual(usage.input_tokens is not None, True)
        # allowed to store (cloud:store and data_policy.allow_store): the handle routes reach the owning upstream
        _, data, _ = self.passthrough('OAI', 'openai.responses', raw({'model': 'a', 'input': 'hi', 'background': True}),
                                      allow_store=True)
        response_id = json.loads(data)['id']
        self.assertNotIn('store', self.last('OAI').json())
        for surface, method in (('openai.responses.retrieve', 'GET'), ('openai.responses.input_items', 'GET'),
                                ('openai.responses.cancel', 'POST'), ('openai.responses.delete', 'DELETE')):
            with self.subTest(surface=surface):
                result, data, _ = self.passthrough('OAI', surface, raw(None, method=method,
                                                                       path_params={'response_id': response_id}))
                hit = self.last('OAI')
                self.assertEqual(hit.method, method)
                self.assertTrue(hit.path.startswith('/v1/responses/' + response_id), hit.path)
                self.assertEqual(result.status, 200)

    def test_embeddings_moderations_and_images(self):
        _, data, usage = self.passthrough('OAI', 'openai.embeddings', raw({'model': 'a', 'input': 'x'}),
                                          model='text-embedding-3-small')
        self.assertEqual(json.loads(data)['data'][0]['object'], 'embedding')
        self.assertEqual(usage.input_tokens, 8)
        _, data, _ = self.passthrough('OAI', 'openai.moderations', raw({'model': 'a', 'input': 'x'}),
                                      model='omni-moderation-latest')
        self.assertIn('results', json.loads(data))
        notes = []
        _, data, usage = self.passthrough('OAI', 'openai.images.generations',
                                          raw({'model': 'a', 'prompt': 'cat', 'response_format': 'b64_json'}),
                                          model='gpt-image-1', notes=notes)
        self.assertNotIn('response_format', self.last('OAI').json())  # gpt-image-* takes no response_format
        self.assertIn('dropped:response_format', notes)
        self.assertEqual(usage.images, 1)
        # live 2026-10-02: dall-e models refuse response_format too (image_b64_only_globs '*'); any value is dropped
        for model in ('dall-e-2', 'dall-e-3'):
            for fmt in ('b64_json', 'url'):
                with self.subTest(model=model, response_format=fmt):
                    notes = []
                    _, data, _ = self.passthrough('OAI', 'openai.images.generations',
                                                  raw({'model': 'a', 'prompt': 'cat', 'response_format': fmt}),
                                                  model=model, notes=notes)
                    self.assertEqual(self.last('OAI').json(), {'model': model, 'prompt': 'cat'})
                    self.assertIn('dropped:response_format', notes)
                    self.assertIn('b64_json', json.loads(data)['data'][0])
        for model in ('gpt-image-1', 'dall-e-2'):
            with self.subTest(surface='openai.images.edits', model=model):
                notes = []
                _, data, _ = self.passthrough('OAI', 'openai.images.edits',
                                              raw(form=[('model', 'a'), ('prompt', 'hat'), ('response_format', 'url'),
                                                        ('image', ('cat.png', b'\x89PNG\r\n\x1a\nmock', 'image/png'))]),
                                              model=model, notes=notes)
                hit = self.last('OAI')
                self.assertEqual(hit.path, '/v1/images/edits')
                self.assertIn(model.encode(), hit.body)
                self.assertNotIn(b'response_format', hit.body)
                self.assertIn('dropped:response_format', notes)
                self.assertIn('data', json.loads(data))

    def test_speech_binary_stream_and_audio_forms(self):
        result, data, usage = self.passthrough('OAI', 'openai.audio.speech',
                                               raw({'model': 'a', 'input': 'hello there', 'voice': 'alloy',
                                                    'response_format': 'wav'}), model='tts-1')
        self.assertTrue(data.startswith(b'RIFF'))
        self.assertEqual((usage.characters, usage.estimated), (11, True))
        self.assertTrue(result.content_type.startswith('audio/'))
        form = [('model', 'a'), ('response_format', 'verbose_json'), ('file', ('a.wav', WAV, 'audio/wav'))]
        for surface, path in (('openai.audio.transcriptions', '/v1/audio/transcriptions'),
                              ('openai.audio.translations', '/v1/audio/translations')):
            with self.subTest(surface=surface):
                _, data, usage = self.passthrough('OAI', surface, raw(form=form + [('worker', 'gpu0')]), model='whisper-1')
                hit = self.last('OAI')
                self.assertEqual(hit.path, path)
                self.assertIn(b'whisper-1', hit.body)
                self.assertNotIn(b'gpu0', hit.body)
                self.assertIn('text', json.loads(data))
                self.assertEqual(usage.audio_seconds, 1.0)
        with self.assertRaises(BrokerError) as ctx:
            self.passthrough('OAI', 'openai.audio.translations', raw(form=form + [('target_language', 'de')]),
                             model='whisper-1')
        self.assertEqual(ctx.exception.code, 'broker_capability_unavailable')

    def test_reasoning_model_rewrite(self):
        notes = []
        body = {'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 50, 'temperature': 0.2,
                'top_p': 0.9}
        self.passthrough('OAI', 'openai.chat', raw(body), model='o3-mini', notes=notes)
        sent = self.last('OAI').json()
        self.assertEqual(sent['max_completion_tokens'], 50)
        self.assertFalse({'max_tokens', 'temperature', 'top_p'} & set(sent))
        self.assertIn('renamed:max_tokens->max_completion_tokens', notes)
        self.assertIn('dropped:temperature', notes)
        self.passthrough('OAI', 'openai.chat', raw(dict(body)), model='gpt-4o-mini')
        self.assertEqual(self.last('OAI').json()['temperature'], 0.2)

    def test_catalog_probe_and_errors(self):
        target = self.targets['OAI']

        async def go():
            async with target.egress() as egress:
                models = await target.adapter.list_models(target.grant(purpose='catalog'), egress)
                probe = await target.adapter.probe(target.grant(purpose='probe'), egress)
                target.mock.faults.push('status', status=503)
                down = await target.adapter.probe(target.grant(purpose='probe'), egress)
                return models, probe, down
        models, probe, down = S.run(go())
        self.assertIn('gpt-4o-mini', models)
        self.assertTrue(probe.ok and probe.status == 200 and probe.latency_ms is not None)
        self.assertEqual((down.ok, down.status, down.code), (False, 503, 'upstream_unavailable'))
        self.assertEqual(self.last('OAI').path, '/v1/models')
        err = target.adapter.map_error(429, {'retry-after': '3'}, b'{"error":{"message":"slow","code":"rate"}}')
        self.assertEqual((err.kind, err.retry_after, err.provider_code), ('rate_limited', 3.0, 'rate'))
        target.mock.faults.push('status', status=429)
        with self.assertRaises(UpstreamFailure) as ctx:
            self.passthrough('OAI', 'openai.chat', raw({'model': 'a', 'messages': [{'role': 'user', 'content': 'x'}]}))
        self.assertEqual(ctx.exception.code, 'upstream_rate_limited')

    def test_invoke_sync_parity(self):
        target = self.targets['OAI']
        ir = openai_chat.decode_request({'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}]})

        async def go():
            async with target.egress() as egress:
                return await target.adapter.invoke(target.grant(), egress, ir)
        async_resp = S.run(go())
        sync_resp = target.adapter.invoke_sync(target.grant(), self.sync_egress(target), ir)
        self.assertEqual((async_resp.content, async_resp.stop, async_resp.usage),
                         (sync_resp.content, sync_resp.stop, sync_resp.usage))
        embed = openai_chat.decode_embed_request({'model': 'a', 'input': ['x', 'y']})
        vectors = target.adapter.invoke_sync(target.grant('text-embedding-3-small'), self.sync_egress(target), embed)
        self.assertEqual(len(vectors.extras['vectors']), 2)


class ProfileRows(unittest.TestCase):
    """ParamMap rows, Mistral, Groq, NVIDIA, Together and Perplexity against an OpenAI-protocol mock."""

    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    def row(self, profile_id, mock, model):
        upstream = Upstream(profile=profile_id, base_url=mock.api_base, enabled=True, credential='cred')
        adapter = OpenAICompatAdapter(PROFILES[profile_id], upstream)
        handle = S.handle(upstream.origin(allow_http_lan=False), profile_id, mock.PROVIDER if mock.PROVIDER in SENTINELS
                          else 'openai')
        return adapter, S.grant(upstream, profile_id, model, handle)

    def call(self, adapter, grant, surface, body, **kw):
        async def go():
            async with S.Egress() as egress:
                result = await adapter.passthrough(grant, egress, SURFACES[surface], raw(body), **kw)
                return await drain(result)
        return S.run(go())

    def test_mistral_param_map_and_tool_id_round_trip(self):
        with OpenAIMock(variant='mistral') as mock:
            adapter, grant = self.row('mistral', mock, 'mistral-large-latest')
            foreign = 'call_7f3a9c2e1b4d'
            body = {'model': 'a', 'seed': 7, 'logit_bias': {'1': 2}, 'parallel_tool_calls': False, 'safe_prompt': True,
                    'tools': [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {}}}],
                    'messages': [{'role': 'user', 'content': 'weather?'},
                                 {'role': 'assistant', 'content': None, 'tool_calls': [
                                     {'id': foreign, 'type': 'function',
                                      'function': {'name': 'get_weather', 'arguments': '{"location":"Paris"}'}}]},
                                 {'role': 'tool', 'tool_call_id': foreign, 'content': '18C'}]}
            notes = []
            data, _ = self.call(adapter, grant, 'openai.chat', body, notes=notes)
            sent = mock.hits[-1].json()
            self.assertEqual(sent['random_seed'], 7)
            self.assertNotIn('seed', sent)
            self.assertNotIn('logit_bias', sent)
            self.assertIs(sent['parallel_tool_calls'], False)
            self.assertIs(sent['safe_prompt'], True)
            mapped = sent['messages'][1]['tool_calls'][0]['id']
            self.assertRegex(mapped, r'^[A-Za-z0-9]{9}$')
            self.assertEqual(sent['messages'][2]['tool_call_id'], mapped)
            self.assertIn('remapped:tool_call_ids', notes)
            self.assertNotIn('stream_options', sent)  # mistral reports usage without include_usage
            self.assertEqual(json.loads(data)['choices'][0]['message']['tool_calls'][0]['id'], 'D681PevKs')
            ids = ToolIdMap()
            self.assertEqual(ids.to_client(ids.to_upstream(foreign)), foreign)
            self.assertEqual(ids.to_upstream(foreign), mapped)  # deterministic per value
            self.assertEqual(ids.to_upstream('D681PevKs'), 'D681PevKs')
            self.assertEqual(ids.restore({'tool_calls': [{'id': mapped}]}), {'tool_calls': [{'id': foreign}]})
            # without the remap the Mistral-like mock refuses the foreign id (422)
            with self.assertRaises(UpstreamFailure) as ctx:
                S.run(_send_plain(mock, body))
            self.assertEqual(ctx.exception.upstream.status, 422)

    def test_groq_refusals_happen_before_egress(self):
        with OpenAIMock() as mock:
            adapter, grant = self.row('groq', mock, 'llama-3.3-70b-versatile')
            for extra in ({'logprobs': True}, {'top_logprobs': 2}, {'logit_bias': {'1': 1}}, {'n': 2}):
                with self.subTest(extra=extra):
                    with self.assertRaises(BrokerError) as ctx:
                        self.call(adapter, grant, 'openai.chat',
                                  dict({'model': 'a', 'messages': [{'role': 'user', 'content': 'x'}]}, **extra))
                    self.assertEqual((ctx.exception.code, ctx.exception.status), ('broker_capability_unavailable', 400))
            self.assertEqual(mock.hits, [])
            self.call(adapter, grant, 'openai.chat', {'model': 'a', 'n': 1, 'messages': [{'role': 'user', 'content': 'x'}]})
            self.assertEqual(len(mock.hits), 1)

    def test_nvidia_input_type_never_defaulted(self):
        with OpenAIMock() as mock:
            adapter, grant = self.row('nvidia_api_catalog', mock, 'nvidia/nv-embedqa-e5-v5')
            with self.assertRaises(BrokerError) as ctx:
                self.call(adapter, grant, 'openai.embeddings', {'model': 'a', 'input': 'x'})
            self.assertEqual(ctx.exception.code, 'broker_capability_unavailable')
            self.assertIn('input_type required', str(ctx.exception))
            self.assertEqual(mock.hits, [])
            self.call(adapter, grant, 'openai.embeddings', {'model': 'a', 'input': 'x', 'input_type': 'passage',
                                                            'truncate': 'END'})
            self.assertEqual((mock.hits[-1].json()['input_type'], mock.hits[-1].json()['truncate']), ('passage', 'END'))
            notes = []
            self.call(adapter, grant, 'openai.embeddings', {'model': 'a', 'input': 'x'}, inject={'input_type': 'query'},
                      notes=notes)
            self.assertEqual(mock.hits[-1].json()['input_type'], 'query')
            with self.assertRaises(BrokerError):
                self.call(adapter, grant, 'openai.embeddings', {'model': 'a', 'input': 'x', 'input_type': 'document'})
            plain, plain_grant = self.row('nvidia_api_catalog', mock, 'nvidia/nv-embed-v1')
            self.call(plain, plain_grant, 'openai.embeddings', {'model': 'a', 'input': 'x'})  # symmetric: no input_type

    def test_catalog_shapes(self):
        self.assertEqual(parse_catalog({'object': 'list', 'data': [{'id': 'a'}, {'id': 'b'}]}), ['a', 'b'])
        self.assertEqual(parse_catalog([{'id': 'org/m1', 'object': 'model'}, {'id': 'org/m2'}]), ['org/m1', 'org/m2'])
        self.assertEqual(parse_catalog(['x', 'y']), ['x', 'y'])
        with self.assertRaises(BrokerError):
            parse_catalog({'models': []})
        perplexity = OpenAICompatAdapter(PROFILES['perplexity'], Upstream(profile='perplexity', models=['sonar']))

        async def static():
            return await perplexity.list_models(None, None), await perplexity.probe(None, None)
        models, probe = S.run(static())
        self.assertEqual(models, ['sonar'])
        self.assertEqual((probe.ok, probe.code), (True, 'skipped'))

    def test_include_usage_rules(self):
        body = {'stream': True}
        self.assertEqual(inject_include_usage(PROFILES['openai'], body), ({'stream': True, 'stream_options':
                                                                            {'include_usage': True}}, True))
        self.assertEqual(inject_include_usage(PROFILES['openai'], {'stream': False}), ({'stream': False}, False))
        asked = {'stream': True, 'stream_options': {'include_usage': True}}
        self.assertEqual(inject_include_usage(PROFILES['openai'], asked), (asked, False))
        self.assertEqual(inject_include_usage(PROFILES['mistral'], body), (body, False))

    def test_rewrite_body_unit(self):
        out, notes = rewrite_body(PROFILES['openai'], None, 'openai.chat',
                                  {'model': 'x', 'store': True, 'worker': 'w', 'messages': []}, 'gpt-4o')
        self.assertEqual(out, {'model': 'gpt-4o', 'store': False, 'messages': []})
        self.assertIn('forced:store=false', notes)
        out, _ = rewrite_body(PROFILES['openai'], None, 'openai.chat', {'model': 'x', 'store': True}, 'gpt-4o',
                              allow_store=True)
        self.assertIs(out['store'], True)
        out, _ = rewrite_body(PROFILES['together'], None, 'openai.chat', {'model': 'x', 'store': True}, 'org/m')
        self.assertIs(out['store'], True)  # store is OpenAI's field; other rows see the client's native field


async def _send_plain(mock, body):
    """The same body sent without the adapter's remap (to prove the mock enforces Mistral's id rule)."""
    upstream = Upstream(profile='custom_openai', base_url=mock.api_base)
    adapter = OpenAICompatAdapter(PROFILES['custom_openai'], upstream)
    handle = S.handle(upstream.origin(allow_http_lan=False), 'custom_openai', 'mistral')
    async with S.Egress() as egress:
        result = await adapter.passthrough(S.grant(upstream, 'custom_openai', 'mistral-large-latest', handle), egress,
                                           SURFACES['openai.chat'], raw(body))
        await drain(result)


class AzurePassthrough(Case):
    columns = ('AZ',)

    def test_azure_surfaces_through_the_exact_host(self):
        target = self.targets['AZ']
        cases = [
            ('openai.chat', {'messages': [{'role': 'user', 'content': 'hi'}]}, 'gpt-4o-mini', 'chat-dep',
             'chat/completions'),
            ('openai.completions', {'prompt': 'Say this'}, 'gpt-35-turbo-instruct', 'instruct-dep', 'completions'),
            ('openai.embeddings', {'input': 'x'}, 'text-embedding-3-small', 'embed-dep', 'embeddings'),
            ('openai.audio.speech', {'input': 'hello', 'voice': 'alloy'}, 'tts-1', 'tts-dep', 'audio/speech'),
            ('openai.images.generations', {'prompt': 'cat'}, 'dall-e-3', 'image-dep', 'images/generations'),
        ]
        for surface, body, model, deployment, suffix in cases:
            with self.subTest(surface=surface):
                _, data, _ = self.passthrough('AZ', surface, raw(dict(body, model='alias')), model=model)
                hit = self.last('AZ')
                self.assertEqual(hit.path, f'/openai/deployments/{deployment}/{suffix}')
                self.assertEqual(hit.query_params, {'api-version': S.AZURE_API_VERSION})
                self.assertEqual(hit.credential, '<sentinel:azure_openai>')
                self.assertNotIn('authorization', hit.headers)
                self.assertTrue(data)
        self.assertTrue(all(url.host == f'{S.AZURE_RESOURCE}.openai.azure.com' for url in target.transport.seen))
        form = [('model', 'a'), ('file', ('a.wav', WAV, 'audio/wav'))]
        for surface, suffix in (('openai.audio.transcriptions', 'audio/transcriptions'),
                                ('openai.audio.translations', 'audio/translations')):
            with self.subTest(surface=surface):
                _, data, _ = self.passthrough('AZ', surface, raw(form=form), model='whisper-1')
                self.assertEqual(self.last('AZ').path, f'/openai/deployments/whisper-dep/{suffix}')
                self.assertIn('text', json.loads(data))

    def test_azure_unknown_model_is_404_before_egress(self):
        with self.assertRaises(BrokerError) as ctx:
            self.passthrough('AZ', 'openai.chat', raw({'messages': []}), model='gpt-unknown')
        self.assertEqual((ctx.exception.code, ctx.exception.status), ('model_not_found', 404))
        self.assertEqual(self.targets['AZ'].mock.hits, [])

    def test_azure_stream_with_prompt_filter_chunk(self):
        self.targets['AZ'].mock.respond_with('openai.chat', 'chat_stream')  # Azure's own stream fixture
        _, data, usage = self.passthrough('AZ', 'openai.chat', raw({'messages': [{'role': 'user', 'content': 'hi'}],
                                                                     'stream': True}))
        events = S.sse_events(data)
        self.assertIn('prompt_filter_results', json.loads(events[0][1]))  # passed through unchanged
        self.assertEqual(events[-1][1], '[DONE]')

    def test_azure_completions_stream(self):
        _, data, usage = self.passthrough('AZ', 'openai.completions', raw({'prompt': 'Say this', 'stream': True}),
                                          model='gpt-35-turbo-instruct')
        hit = self.last('AZ')
        self.assertTrue(hit.path.endswith('/deployments/instruct-dep/completions'), hit.path)
        self.assertEqual(hit.json()['stream_options'], {'include_usage': True})
        self.assertEqual(hit.credential, '<sentinel:azure_openai>')
        events = S.sse_events(data)
        self.assertEqual(events[-1][1], '[DONE]')
        self.assertTrue(any(json.loads(p).get('choices') for _, p in events[:-1]))

    def test_azure_reasoning_deployment_translated(self):
        # Azure has no /responses surface, so Responses and Anthropic clients reach o-series / gpt-5 deployments only
        # through the translated chat path, which must use the reasoning-model ParamMap of the openai row.
        target = self.targets['AZ']
        target.mock.deployments['reason-dep'] = 'o3-mini'  # this test's deployment on the running mock
        self.addCleanup(target.mock.deployments.pop, 'reason-dep', None)
        upstream = Upstream(profile='azure_openai', enabled=True, credential='cred',
                            azure={'resource': S.AZURE_RESOURCE, 'api_version': S.AZURE_API_VERSION,
                                   'deployments': {'o3-mini': 'reason-dep'}})
        reasoning = S.MockTarget('AZ', target.mock, 'azure_openai', 'o3-mini', 'azure_openai', upstream=upstream,
                                 transport=target.transport)
        ir = anthropic_messages.decode_request({'model': 'alias', 'max_tokens': 1000, 'temperature': 0.5,
                                                'messages': [{'role': 'user', 'content': 'hi'}]})
        notes = []

        async def go():
            async with reasoning.egress() as egress:
                return await reasoning.adapter.invoke(reasoning.grant(), egress, ir, notes=notes)
        response = S.run(go())
        hit = self.last('AZ')
        self.assertEqual(hit.path, '/openai/deployments/reason-dep/chat/completions')
        sent = hit.json()
        self.assertEqual(sent.get('max_completion_tokens'), 1000)
        self.assertNotIn('max_tokens', sent)
        self.assertNotIn('temperature', sent)
        self.assertIn('renamed:max_tokens->max_completion_tokens', notes)
        self.assertIn('dropped:temperature', notes)
        self.assertTrue(response.content)
        # a non-reasoning deployment keeps max_tokens and temperature
        body, extra = rewrite_body(PROFILES['azure_openai'], upstream, SURFACES['openai.chat'],
                                   {'max_tokens': 5, 'temperature': 0.2}, 'gpt-4o-mini')
        self.assertEqual((body.get('max_tokens'), body.get('temperature'), extra), (5, 0.2, []))

    def test_azure_gpt_image_deployment_never_gets_response_format(self):
        # azure_openai image_b64_only_globs ('gpt-image-*',): a gpt-image deployment answers base64 and refuses the field
        # (the Azure mock answers 400 unknown_parameter, as Azure does); a dall-e deployment keeps receiving it.
        target = self.targets['AZ']
        target.mock.deployments['gpt-image-dep'] = 'gpt-image-1'  # this test's deployment on the running mock
        self.addCleanup(target.mock.deployments.pop, 'gpt-image-dep', None)
        self.assertEqual(PROFILES['azure_openai'].extras.get('image_b64_only_globs'), ('gpt-image-*',))
        upstream = Upstream(profile='azure_openai', enabled=True, credential='cred',
                            azure={'resource': S.AZURE_RESOURCE, 'api_version': S.AZURE_API_VERSION,
                                   'deployments': {'gpt-image-1': 'gpt-image-dep', 'dall-e-3': 'image-dep'}})
        image = S.MockTarget('AZ', target.mock, 'azure_openai', 'gpt-image-1', 'azure_openai', upstream=upstream,
                             transport=target.transport)
        for fmt in ('b64_json', 'url'):
            with self.subTest(response_format=fmt):
                notes = []

                async def go():
                    async with image.egress() as egress:
                        result = await image.adapter.passthrough(
                            image.grant(), egress, SURFACES['openai.images.generations'],
                            raw({'model': 'alias', 'prompt': 'cat', 'response_format': fmt}), notes=notes)
                        return result, *(await drain(result))
                result, data, usage = S.run(go())
                hit = self.last('AZ')
                self.assertEqual(hit.path, '/openai/deployments/gpt-image-dep/images/generations')
                self.assertNotIn('response_format', hit.json())
                self.assertIn('dropped:response_format', notes)
                self.assertEqual(result.status, 200)
                self.assertIn('b64_json', json.loads(data)['data'][0])
        # the dall-e deployment keeps the field (pre-2026-10 OpenAI behaviour on Azure)
        body, extra = rewrite_body(PROFILES['azure_openai'], upstream, SURFACES['openai.images.generations'],
                                   {'prompt': 'cat', 'response_format': 'b64_json'}, 'dall-e-3')
        self.assertEqual((body.get('response_format'), extra), ('b64_json', []))
        body, extra = rewrite_body(PROFILES['azure_openai'], upstream, SURFACES['openai.images.generations'],
                                   {'prompt': 'cat', 'response_format': 'b64_json'}, 'gpt-image-1')
        self.assertNotIn('response_format', body)
        self.assertIn('dropped:response_format', extra)


class AnthropicAdapterRow(Case):
    columns = ('ANT',)

    def test_messages_headers_model_and_betas(self):
        target = self.targets['ANT']
        target.upstream = Upstream(profile='anthropic', base_url=target.mock.api_base, credential='cred',
                                   anthropic_betas=['prompt-caching-2024-07-31'])
        target.adapter.upstream = target.upstream
        body = {'model': 'alias', 'max_tokens': 32, 'messages': [{'role': 'user', 'content': 'hi'}], 'location': 'x'}
        result, data, usage = self.passthrough('ANT', 'anthropic.messages',
                                               raw(body, headers={'anthropic-beta': 'files-api-2025-04-14'}))
        hit = self.last('ANT')
        self.assertEqual((hit.path, hit.credential), ('/v1/messages', '<sentinel:anthropic>'))
        self.assertEqual(hit.headers['anthropic-version'], '2023-06-01')
        self.assertEqual(hit.headers['anthropic-beta'], 'files-api-2025-04-14,prompt-caching-2024-07-31')
        self.assertNotIn('authorization', hit.headers)
        self.assertEqual(hit.json()['model'], 'claude-sonnet-4-5-20250929')
        self.assertNotIn('location', hit.json())
        self.assertEqual(json.loads(data)['type'], 'message')
        self.assertEqual((usage.input_tokens, usage.output_tokens), (12, 10))
        self.assertEqual(result.upstream_request_id, result.headers.get('request-id'))

    def test_stream_count_tokens_and_catalog(self):
        _, data, usage = self.passthrough('ANT', 'anthropic.messages', raw(
            {'model': 'a', 'max_tokens': 32, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}))
        names = [e for e, _ in S.sse_events(data)]
        self.assertEqual((names[0], names[-1]), ('message_start', 'message_stop'))
        self.assertEqual((usage.input_tokens, usage.output_tokens), (12, 10))
        _, data, usage = self.passthrough('ANT', 'anthropic.count_tokens',
                                          raw({'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}]}))
        self.assertGreater(json.loads(data)['input_tokens'], 0)
        self.assertEqual(self.last('ANT').path, '/v1/messages/count_tokens')
        self.assertEqual((usage.input_tokens, usage.output_tokens), (0, 0))
        target = self.targets['ANT']

        async def go():
            async with target.egress() as egress:
                models = await target.adapter.list_models(target.grant(purpose='catalog'), egress)
                ir = openai_chat.decode_request({'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}]})
                count = await target.adapter.count_tokens(target.grant(), egress, ir)
                return models, count
        models, count = S.run(go())
        self.assertIn('claude-sonnet-4-5-20250929', models)
        self.assertGreater(count, 0)

    def test_translated_max_tokens_and_structured_outputs(self):
        target = self.targets['ANT']
        ir = openai_chat.decode_request({'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}],
                                         'response_format': {'type': 'json_schema', 'json_schema': {
                                             'name': 'x', 'schema': {'type': 'object'}}}})
        # structured outputs are GA: output_config.format without any beta header
        target.adapter.invoke_sync(target.grant(), self.sync_egress(target), ir)
        sent = target.mock.hits[-1]
        self.assertEqual(sent.json()['output_config'], {'format': {'type': 'json_schema', 'schema': {'type': 'object'}}})
        self.assertNotIn('output_format', sent.json())
        self.assertNotIn('structured-outputs', sent.headers.get('anthropic-beta', ''))
        # an operator-listed beta is still sent (harmless), and the default cap is injected
        beta = PROFILES['anthropic'].extras['structured_outputs_beta']
        adapter = type(target.adapter)(PROFILES['anthropic'], Upstream(profile='anthropic', base_url=target.mock.api_base,
                                                                       anthropic_betas=[beta]), default_max_tokens=777)
        notes = []
        resp = adapter.invoke_sync(target.grant(), self.sync_egress(target), ir, notes=notes)
        sent = target.mock.hits[-1]
        self.assertEqual(sent.json()['max_tokens'], 777)
        self.assertIn('injected:max_tokens', notes)
        self.assertEqual(sent.json()['output_config']['format'], {'type': 'json_schema', 'schema': {'type': 'object'}})
        self.assertEqual(sent.headers['anthropic-beta'], beta)
        self.assertEqual(resp.stop, 'end_turn')

    def test_invoke_sync_parity(self):
        target = self.targets['ANT']
        ir = openai_chat.decode_request({'model': 'a', 'max_tokens': 20, 'messages': [{'role': 'user', 'content': 'x'}],
                                         'tools': [{'type': 'function', 'function': {'name': 'get_weather'}}]})

        async def go():
            async with target.egress() as egress:
                return await target.adapter.invoke(target.grant(), egress, ir)
        a = S.run(go())
        b = target.adapter.invoke_sync(target.grant(), self.sync_egress(target), ir)
        self.assertEqual((a.content, a.stop, a.usage), (b.content, b.stop, b.usage))
        self.assertEqual(a.stop, 'tool_use')


class GeminiAdapterRow(Case):
    columns = ('GEM',)

    def test_generate_stream_embed_count_paths_and_header_auth(self):
        cases = [('gemini.generateContent', {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                  '/v1beta/models/gemini-2.5-flash:generateContent', {}),
                 ('gemini.streamGenerateContent', {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                  '/v1beta/models/gemini-2.5-flash:streamGenerateContent', {'alt': 'sse'}),
                 ('gemini.embedContent', {'content': {'parts': [{'text': 'x'}]}},
                  '/v1beta/models/gemini-2.5-flash:embedContent', {}),
                 ('gemini.batchEmbedContents', {'requests': [{'model': 'models/alias', 'content': {'parts': [{'text': 'a'}]}},
                                                             {'model': 'models/alias', 'content': {'parts': [{'text': 'b'}]}}]},
                  '/v1beta/models/gemini-2.5-flash:batchEmbedContents', {}),
                 ('gemini.countTokens', {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                  '/v1beta/models/gemini-2.5-flash:countTokens', {})]
        for surface, body, path, query in cases:
            with self.subTest(surface=surface):
                _, data, usage = self.passthrough('GEM', surface, raw(body, path_params={'model': 'models/alias'},
                                                                      query={'alt': 'sse'} if query else {}),
                                                  model='models/gemini-2.5-flash')
                hit = self.last('GEM')
                self.assertEqual((hit.path, hit.query_params), (path, query))
                self.assertEqual(hit.credential, '<sentinel:gemini>')
                self.assertNotIn('key', hit.query_params)
                self.assertTrue(data)
        self.assertEqual([r['model'] for r in self.last('GEM').json().get('requests', [])], [])
        batch = [h for h in self.targets['GEM'].mock.hits if h.path.endswith('batchEmbedContents')][0]
        self.assertEqual({r['model'] for r in batch.json()['requests']}, {'models/gemini-2.5-flash'})

    def test_stream_without_alt_sse_passes_the_json_array(self):
        _, data, _ = self.passthrough('GEM', 'gemini.streamGenerateContent',
                                      raw({'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                                          query={'alt': 'json'}))
        self.assertEqual(self.last('GEM').query_params, {})
        self.assertTrue(data.startswith(b'['))

    def test_paged_catalog_and_usage(self):
        target = self.targets['GEM']

        class SmallPages(GeminiAdapter):
            pass
        adapter = SmallPages(target.profile, target.upstream)

        async def go():
            async with target.egress() as egress:
                return await adapter.list_models(target.grant(purpose='catalog'), egress)
        models = S.run(go())
        self.assertEqual(len(models), 3)
        self.assertTrue(all(not m.startswith('models/') for m in models))
        self.assertEqual(self.last('GEM').query_params.get('pageSize'), '1000')
        usage = adapter.usage_from_body(SURFACES['gemini.generateContent'],
                                        {'usageMetadata': {'promptTokenCount': 4, 'candidatesTokenCount': 9,
                                                           'thoughtsTokenCount': 3}}, {})
        # thinking is billed as output: the IR's output_tokens = candidates + thoughts
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.reasoning_tokens), (4, 12, 3))

    def test_gemini3_tool_loop_signature_and_embedding_usage(self):
        target = self.targets['GEM']
        follow_up = {'model': 'alias', 'messages': [
            {'role': 'user', 'content': 'weather?'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city":"Paris"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"t":20}'}],
            'tools': [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {
                'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]}
        for model, expected in (('gemini-3-pro-preview', gemini_codec.DUMMY_THOUGHT_SIGNATURE),
                                ('gemini-2.5-flash', None)):
            with self.subTest(model=model):
                notes = []

                async def go():
                    async with target.egress() as egress:
                        return await target.adapter.invoke(target.grant(model), egress,
                                                           openai_chat.decode_request(follow_up), notes=notes)
                S.run(go())
                part = self.last('GEM').json()['contents'][1]['parts'][0]
                self.assertEqual(part.get('thoughtSignature'), expected)
        # embeddings carry no counts upstream: usage is estimated from the input text, never zero
        _, _, usage = self.passthrough('GEM', 'gemini.embedContent',
                                       raw({'content': {'parts': [{'text': 'abcdefgh'}]}}))
        self.assertEqual(usage, Usage(input_tokens=2, estimated=True))

        async def embed():
            async with target.egress() as egress:
                return await target.adapter.invoke(target.grant(), egress, IRRequest(kind='embed', model='alias',
                                                                                      inputs=['abcd', 'efgh']))
        self.assertEqual(S.run(embed()).usage, Usage(input_tokens=2, estimated=True))

    def test_paging_follows_next_page_token(self):
        target = self.targets['GEM']
        original = dict(target.profile.extras)
        calls = []

        class Paging(GeminiAdapter):
            async def send(self, grant, egress, method, path, *, stream=False, **kw):
                params = dict(kw.pop('params') or {})
                params['pageSize'] = '2'
                calls.append(params)
                return await super().send(grant, egress, method, path, stream=stream, params=params, **kw)
        adapter = Paging(target.profile, target.upstream)

        async def go():
            async with target.egress() as egress:
                return await adapter.list_models(target.grant(purpose='catalog'), egress)
        models = S.run(go())
        self.assertEqual(len(models), 3)
        self.assertEqual(len(calls), 2)
        self.assertIn('pageToken', calls[1])
        self.assertEqual(dict(target.profile.extras), original)

    def test_invoke_sync_parity(self):
        target = self.targets['GEM']
        ir = openai_chat.decode_request({'model': 'a', 'messages': [{'role': 'user', 'content': 'x'}],
                                         'tools': [{'type': 'function', 'function': {'name': 'get_weather'}}]})

        async def go():
            async with target.egress() as egress:
                return await target.adapter.invoke(target.grant(), egress, ir)
        a = S.run(go())
        b = target.adapter.invoke_sync(target.grant(), self.sync_egress(target), ir)
        self.assertEqual((a.content, a.stop, a.usage), (b.content, b.stop, b.usage))
        self.assertEqual(a.stop, 'tool_use')
        self.assertRegex(a.content[0].id, r'^call_0_[0-9a-f]{8}$')


class ElevenLabsAdapterRow(Case):
    columns = ('XI',)
    VOICE = '21m00Tcm4TlvDq8ikWAM'

    def test_tts_stream_stt_voices(self):
        result, data, usage = self.passthrough('XI', 'elevenlabs.tts', raw({'text': 'hello', 'model_id': 'alias'},
                                                                           path_params={'voice_id': self.VOICE}))
        hit = self.last('XI')
        self.assertEqual((hit.path, hit.query_params), (f'/v1/text-to-speech/{self.VOICE}',
                                                        {'output_format': 'mp3_44100_128'}))
        self.assertEqual(hit.credential, '<sentinel:elevenlabs>')
        self.assertEqual(hit.json()['model_id'], 'eleven_multilingual_v2')
        self.assertEqual((usage.characters, usage.estimated), (5, False))  # the character-cost header
        self.assertTrue(result.content_type.startswith('audio/'))
        _, stream, usage = self.passthrough('XI', 'elevenlabs.tts_stream',
                                            raw({'text': 'hello there', 'model_id': 'a'},
                                                path_params={'voice_id': self.VOICE},
                                                query={'output_format': 'pcm_24000', 'enable_logging': 'false'}))
        hit = self.last('XI')
        self.assertEqual(hit.path, f'/v1/text-to-speech/{self.VOICE}/stream')
        self.assertEqual(hit.query_params, {'output_format': 'pcm_24000', 'enable_logging': 'false'})
        self.assertEqual(usage.characters, 11)
        expected = b''.join(self.targets['XI'].mock.fixture('tts').frames({}, variant='pcm'))
        self.assertEqual(stream, expected)
        _, data, usage = self.passthrough('XI', 'elevenlabs.stt', raw(form=[('model_id', 'a'),
                                                                            ('file', ('a.wav', WAV, 'audio/wav'))]),
                                          model='scribe_v1')
        self.assertIn(b'scribe_v1', self.last('XI').body)
        self.assertEqual(json.loads(data)['text'], 'Hello world.')
        self.assertAlmostEqual(usage.audio_seconds, 0.96)
        _, data, _ = self.passthrough('XI', 'elevenlabs.voices', raw(None, method='GET'))
        self.assertIn('voices', json.loads(data))
        _, data, _ = self.passthrough('XI', 'elevenlabs.voice', raw(None, method='GET',
                                                                    path_params={'voice_id': self.VOICE}))
        self.assertEqual(json.loads(data)['voice_id'], self.VOICE)

    def test_character_cost_fallback_and_catalog(self):
        self.assertEqual(character_usage({}, 'abcd'), Usage(characters=4, estimated=True))
        self.assertEqual(character_usage({'character-cost': '9'}, 'abcd'), Usage(characters=9))
        target = self.targets['XI']

        async def go():
            async with target.egress() as egress:
                return await target.adapter.list_models(target.grant(purpose='catalog'), egress)
        self.assertIn('eleven_multilingual_v2', S.run(go()))

    def test_invoke_sync_parity(self):
        target = self.targets['XI']
        ir = IRRequest(kind='speech', model='m', messages=[IRMessage('user', [Text('hello')])], voice=self.VOICE,
                       audio_format='mp3_44100_128')

        async def go():
            async with target.egress() as egress:
                return await target.adapter.invoke(target.grant(), egress, ir)
        a = S.run(go())
        b = target.adapter.invoke_sync(target.grant(), self.sync_egress(target), ir)
        self.assertEqual(a.content, b.content)
        self.assertEqual(a.usage.characters, b.usage.characters)


class OllamaAdapterRow(Case):
    columns = ('OLC',)

    def test_chat_generate_embed_tags(self):
        _, data, usage = self.passthrough('OLC', 'ollama.chat', raw({'model': 'a', 'messages': [
            {'role': 'user', 'content': 'hi'}]}))
        hit = self.last('OLC')
        self.assertEqual((hit.path, hit.credential), ('/api/chat', 'Bearer <sentinel:ollama_cloud>'))
        self.assertEqual(hit.json()['model'], 'gpt-oss:120b')
        lines = S.ndjson_lines(data)
        self.assertTrue(lines[-1]['done'])
        self.assertEqual((usage.input_tokens, usage.output_tokens), (26, 12))
        _, data, usage = self.passthrough('OLC', 'ollama.generate', raw({'model': 'a', 'prompt': 'x', 'stream': False}))
        self.assertEqual(self.last('OLC').path, '/api/generate')
        self.assertTrue(json.loads(data)['done'])
        _, data, usage = self.passthrough('OLC', 'ollama.embed', raw({'model': 'a', 'input': 'x'}),
                                          model='embeddinggemma')
        self.assertIn('embeddings', json.loads(data))
        self.assertEqual(usage.input_tokens, 8)
        target = self.targets['OLC']

        async def go():
            async with target.egress() as egress:
                return await target.adapter.list_models(target.grant(purpose='catalog'), egress)
        self.assertIn('gpt-oss:120b', S.run(go()))

    def test_chat_and_generate_streams(self):
        """NDJSON passthrough streams (Ollama streams by default; an explicit stream:true is the same): one JSON object
        per line, the last with done:true and the usage counts."""
        for surface, body in (('ollama.chat', {'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}],
                                               'stream': True}),
                              ('ollama.generate', {'model': 'a', 'prompt': 'hi', 'stream': True})):
            with self.subTest(surface=surface):
                result, data, usage = self.passthrough('OLC', surface, raw(body))
                hit = self.last('OLC')
                self.assertEqual((hit.path, hit.json()['stream'], hit.json()['model']),
                                 ('/' + surface.replace('.', '/', 1).replace('ollama/', 'api/'), True, 'gpt-oss:120b'))
                lines = S.ndjson_lines(data)
                self.assertGreater(len(lines), 1)
                self.assertTrue(lines[-1]['done'])
                self.assertTrue(all(('message' in l) == (surface == 'ollama.chat') for l in lines[:-1]))
                self.assertIsNotNone(usage.output_tokens)

    def test_invoke_sync_parity(self):
        target = self.targets['OLC']
        ir = anthropic_messages.decode_request({'model': 'a', 'max_tokens': 20,
                                                'messages': [{'role': 'user', 'content': 'x'}]})

        async def go():
            async with target.egress() as egress:
                return await target.adapter.invoke(target.grant(), egress, ir)
        a = S.run(go())
        b = target.adapter.invoke_sync(target.grant(), self.sync_egress(target), ir)
        self.assertEqual((a.content, a.stop, a.usage), (b.content, b.stop, b.usage))


class LiveRowFidelity(unittest.TestCase):
    """The exact request the EgressClient would send for the live-lane rows and every native adapter, captured by an
    httpx.MockTransport (no socket): URL, auth header form, required headers, body model."""

    def capture(self, profile_id, surface, body, model, *, upstream=None, raw_request=None, headers=None):
        seen = []

        def handler(request):
            seen.append(request)
            if 'stream' in str(request.url) or (isinstance(body, dict) and body.get('stream')):
                return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=b'')
            return httpx.Response(200, headers={'content-type': 'application/json'}, content=b'{}')
        transport = httpx.MockTransport(handler)
        upstream = upstream or Upstream(profile=profile_id, enabled=True, credential='cred')
        profile = PROFILES[profile_id]
        adapter = S.ADAPTERS[profile.adapter](profile, upstream)
        credential = S.handle(upstream.origin(allow_http_lan=False), profile_id, 'openai')

        async def go():
            async with S.Egress(transport=transport, settings=S.Settings()) as egress:
                result = await adapter.passthrough(S.grant(upstream, profile_id, model, credential), egress,
                                                   SURFACES[surface], raw_request or raw(body, headers=headers))
                await drain(result)
        S.run(go())
        self.assertEqual(len(seen), 1)
        return seen[0]

    def test_openai_nvidia_and_hf_router_requests(self):
        chat = {'model': 'alias', 'messages': [{'role': 'user', 'content': 'hi'}]}
        cases = [('openai', 'gpt-4o-mini', 'https://api.openai.com/v1/chat/completions'),
                 ('nvidia_api_catalog', 'meta/llama-3.1-8b-instruct', 'https://integrate.api.nvidia.com/v1/chat/completions'),
                 ('hf_router', 'openai/gpt-oss-20b:cheapest', 'https://router.huggingface.co/v1/chat/completions')]
        for profile_id, model, url in cases:
            with self.subTest(profile=profile_id):
                request = self.capture(profile_id, 'openai.chat', chat, model)
                self.assertEqual(str(request.url), url)
                self.assertEqual(request.headers['authorization'], 'Bearer ' + SENTINELS['openai'])
                self.assertEqual(json.loads(request.content)['model'], model)
                self.assertEqual(request.headers['accept-encoding'], 'identity')
                for name in ('x-api-key', 'api-key', 'x-goog-api-key', 'xi-api-key'):
                    self.assertNotIn(name, request.headers)
        embed = self.capture('nvidia_api_catalog', 'openai.embeddings',
                             {'model': 'a', 'input': ['x'], 'input_type': 'query', 'encoding_format': 'float'},
                             'nvidia/llama-3.2-nv-embedqa-1b-v2')
        self.assertEqual(str(embed.url), 'https://integrate.api.nvidia.com/v1/embeddings')
        self.assertEqual(json.loads(embed.content)['input_type'], 'query')
        responses = self.capture('openai', 'openai.responses', {'model': 'a', 'input': 'hi'}, 'gpt-4.1-mini')
        self.assertEqual(str(responses.url), 'https://api.openai.com/v1/responses')
        self.assertIs(json.loads(responses.content)['store'], False)
        image = self.capture('openai', 'openai.images.generations', {'model': 'a', 'prompt': 'cat', 'size': '1024x1024'},
                             'gpt-image-1')
        self.assertEqual(str(image.url), 'https://api.openai.com/v1/images/generations')
        stream = self.capture('openai', 'openai.chat', dict(chat, stream=True), 'gpt-4o-mini')
        self.assertEqual(json.loads(stream.content)['stream_options'], {'include_usage': True})
        hf = self.capture('hf_router', 'openai.chat', dict(chat, stream=True), 'meta-llama/Llama-3.1-8B-Instruct')
        self.assertNotIn('stream_options', json.loads(hf.content))

    def test_native_adapter_requests(self):
        request = self.capture('anthropic', 'anthropic.messages',
                               {'model': 'a', 'max_tokens': 8, 'messages': [{'role': 'user', 'content': 'hi'}]},
                               'claude-haiku-4-5-20251001')
        self.assertEqual(str(request.url), 'https://api.anthropic.com/v1/messages')
        self.assertEqual(request.headers['x-api-key'], SENTINELS['openai'])
        self.assertEqual(request.headers['anthropic-version'], '2023-06-01')
        self.assertNotIn('authorization', request.headers)
        request = self.capture('gemini', 'gemini.streamGenerateContent',
                               {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]}, 'models/gemini-2.5-flash',
                               raw_request=raw({'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                                               query={'alt': 'sse'}))
        self.assertEqual(str(request.url), 'https://generativelanguage.googleapis.com/v1beta/models/'
                                           'gemini-2.5-flash:streamGenerateContent?alt=sse')
        self.assertEqual(request.headers['x-goog-api-key'], SENTINELS['openai'])
        self.assertNotIn('key', request.url.params)
        request = self.capture('elevenlabs', 'elevenlabs.tts', None, 'eleven_flash_v2_5',
                               raw_request=raw({'text': 'hi', 'model_id': 'x'}, path_params={'voice_id': 'v123'}))
        self.assertEqual(str(request.url), 'https://api.elevenlabs.io/v1/text-to-speech/v123?output_format=mp3_44100_128')
        self.assertEqual(request.headers['xi-api-key'], SENTINELS['openai'])
        request = self.capture('ollama_cloud', 'ollama.chat', {'model': 'a', 'messages': [{'role': 'user', 'content': 'x'}],
                                                               'stream': False}, 'gpt-oss:120b')
        self.assertEqual(str(request.url), 'https://ollama.com/api/chat')
        self.assertEqual(request.headers['authorization'], 'Bearer ' + SENTINELS['openai'])
        azure = Upstream(profile='azure_openai', credential='cred',
                         azure={'resource': 'contoso', 'endpoint_host': 'cognitiveservices.azure.com',
                                'api_version': '2024-10-21', 'deployments': {'gpt-4o': 'prod-4o'}})
        request = self.capture('azure_openai', 'openai.chat', {'messages': [{'role': 'user', 'content': 'hi'}]}, 'gpt-4o',
                               upstream=azure)
        self.assertEqual(str(request.url), 'https://contoso.cognitiveservices.azure.com/openai/deployments/prod-4o/'
                                           'chat/completions?api-version=2024-10-21')
        self.assertEqual(request.headers['api-key'], SENTINELS['openai'])


class EveryProfileRow(unittest.TestCase):
    """Every profile row, not only the reference row of its adapter column, through its own adapter code: the request the
    EgressClient would send for each native surface of the row, captured by an httpx.MockTransport (no socket). Asserted
    per row and surface: scheme, host and path from the row's default base URL (an operator URL for rows without one,
    the Azure template for azure_openai), the row's auth header form and no other credential header, the configured
    static headers, the hop header on hop rows only, X-Request-ID only where the row forwards it, the rewritten model,
    and (stream variant) include_usage injected exactly on the rows that declare it. Plus the row's catalog parse (or
    its static list without a request) and usage extraction from a response body. The matrix cites these tests as the
    profile-row proof of every passthrough cell."""

    OPERATOR_BASE = 'https://gateway.example.invalid/v1'
    HOP = 'feedfacecafebeef'
    MODEL = 'row-model'
    CREDENTIAL_HEADERS = ('authorization', 'x-api-key', 'api-key', 'x-goog-api-key', 'xi-api-key')
    PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 32

    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    def upstream(self, row):
        headers = {name: f'test-{name.lower()}' for name in row.static_headers}
        if row.id == 'azure_openai':
            return Upstream(profile=row.id, enabled=True, credential='cred', headers=headers, models=[self.MODEL],
                            azure={'resource': 'contoso', 'api_version': '2024-10-21',
                                   'deployments': {self.MODEL: 'row-deployment'}})
        base = None if row.default_base_url else self.OPERATOR_BASE
        return Upstream(profile=row.id, enabled=True, credential='cred', base_url=base, headers=headers,
                        models=[self.MODEL])

    def request_for(self, surface, stream):
        """(RawRequest, JSON body or None) for one native surface."""
        chat = [{'role': 'user', 'content': 'hi'}]
        wav = ('a.wav', WAV, 'audio/wav')
        handle_id = {'response_id': 'resp_row1'}
        bodies = {
            'openai.chat': {'model': 'alias', 'messages': chat},
            'openai.completions': {'model': 'alias', 'prompt': 'hi'},
            'openai.responses': {'model': 'alias', 'input': 'hi'},
            'openai.embeddings': {'model': 'alias', 'input': 'hi'},
            'openai.audio.speech': {'model': 'alias', 'input': 'hi', 'voice': 'alloy'},
            'openai.images.generations': {'model': 'alias', 'prompt': 'cat'},
            'openai.moderations': {'model': 'alias', 'input': 'hi'},
            'openai.rerank': {'model': 'alias', 'query': 'q', 'documents': ['a', 'b']},
            'anthropic.messages': {'model': 'alias', 'max_tokens': 8, 'messages': chat},
            'anthropic.count_tokens': {'model': 'alias', 'messages': chat},
            'gemini.generateContent': {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
            'gemini.streamGenerateContent': {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
            'gemini.embedContent': {'content': {'parts': [{'text': 'hi'}]}},
            'gemini.batchEmbedContents': {'requests': [{'model': 'models/alias', 'content': {'parts': [{'text': 'hi'}]}}]},
            'gemini.countTokens': {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
            'elevenlabs.tts': {'text': 'hi', 'model_id': 'alias'},
            'elevenlabs.tts_stream': {'text': 'hi', 'model_id': 'alias'},
            'ollama.chat': {'model': 'alias', 'messages': chat},
            'ollama.generate': {'model': 'alias', 'prompt': 'hi'},
            'ollama.embed': {'model': 'alias', 'input': 'hi'},
        }
        forms = {
            'openai.audio.transcriptions': [('model', 'alias'), ('file', wav)],
            'openai.audio.translations': [('model', 'alias'), ('file', wav)],
            'openai.images.edits': [('model', 'alias'), ('prompt', 'cat'), ('image', ('a.png', self.PNG, 'image/png'))],
            'elevenlabs.stt': [('model_id', 'alias'), ('file', wav)],
        }
        if surface in ('openai.responses.retrieve', 'openai.responses.input_items'):
            return raw(None, method='GET', path_params=handle_id), None
        if surface == 'openai.responses.delete':
            return raw(None, method='DELETE', path_params=handle_id), None
        if surface == 'openai.responses.cancel':
            return raw({}, path_params=handle_id), None
        if surface == 'elevenlabs.voices':
            return raw(None, method='GET'), None
        if surface == 'elevenlabs.voice':
            return raw(None, method='GET', path_params={'voice_id': 'v123'}), None
        if surface in forms:
            return raw(form=forms[surface]), None
        body = dict(bodies[surface])
        if SURFACES[surface].stream_from == 'json:stream':
            body['stream'] = stream
        params = {'voice_id': 'v123'} if surface.startswith('elevenlabs.tts') else None
        query = {'alt': 'sse'} if surface == 'gemini.streamGenerateContent' else None
        return raw(body, path_params=params, query=query), body

    def capture(self, row, upstream, surface, raw_request, *, stream):
        seen = []

        def handler(request):
            seen.append(request)
            if stream or surface in ('gemini.streamGenerateContent', 'elevenlabs.tts_stream'):
                return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=b'')
            # a NIM-shaped reranker answers rankings (the adapter translates them); every other surface an empty object
            content = b'{"rankings":[]}' if surface == 'openai.rerank' else b'{}'
            return httpx.Response(200, headers={'content-type': 'application/json'}, content=content)
        adapter = S.ADAPTERS[row.adapter](row, upstream)
        credential = S.handle(upstream.origin(allow_http_lan=False), row.id, 'openai')
        grant = dataclasses.replace(S.grant(upstream, row.id, self.MODEL, credential), hop_header=self.HOP)

        async def go():
            async with S.Egress(transport=httpx.MockTransport(handler), settings=S.Settings()) as egress:
                result = await adapter.passthrough(grant, egress, SURFACES[surface], raw_request)
                await drain(result)
        S.run(go())
        self.assertEqual(len(seen), 1, (row.id, surface))
        return seen[0]

    def expected_path(self, row, upstream, surface):
        from urllib.parse import urlsplit
        template = row.surfaces[surface].split('?', 1)[0]
        org, _, name = self.MODEL.partition('/')  # rerank_model_path rows put <org>/<name> in the path
        filled = template.format(deployment='row-deployment', model=self.MODEL, voice_id='v123', org=org,
                                 name=name.replace('.', '_'),
                                 response_id='resp_row1')
        base = urlsplit(upstream.effective_base_url())
        return base.scheme, base.hostname, base.path.rstrip('/') + filled

    def check_row(self, row):
        upstream = self.upstream(row)
        for surface in sorted(row.surfaces):
            variants = (False, True) if SURFACES[surface].stream_from == 'json:stream' else (False,)
            for stream in variants:
                with self.subTest(profile=row.id, surface=surface, stream=stream):
                    raw_request, body = self.request_for(surface, stream)
                    request = self.capture(row, upstream, surface, raw_request, stream=stream)
                    scheme, host, path = self.expected_path(row, upstream, surface)
                    self.assertEqual((request.url.scheme, request.url.host, request.url.path), (scheme, host, path))
                    if row.id == 'azure_openai':
                        self.assertEqual(request.url.params.get('api-version'), '2024-10-21')
                    if surface == 'gemini.streamGenerateContent':
                        self.assertEqual(request.url.params.get('alt'), 'sse')
                    self.assertNotIn('key', request.url.params)
                    template = AUTH_TEMPLATES[row.auth]
                    self.assertIsNotNone(template, row.id)
                    name, value = template
                    self.assertEqual(request.headers[name], value.format(secret=SENTINELS['openai']))
                    for other in self.CREDENTIAL_HEADERS:
                        if other != name.lower():
                            self.assertNotIn(other, request.headers)
                    for header in row.static_headers:
                        self.assertEqual(request.headers[header], f'test-{header.lower()}')
                    self.assertEqual(request.headers.get('x-titanos-hop'), self.HOP if row.hop_header else None)
                    self.assertEqual(request.headers.get('x-request-id'),
                                     'req-wp5' if row.extras.get('forward_request_id') else None)
                    self.assertEqual(request.headers['accept-encoding'], 'identity')
                    content = request.content
                    if body is not None and request.headers.get('content-type', '').startswith('application/json'):
                        sent = json.loads(content)
                        if row.adapter in ('openai_compat', 'anthropic', 'ollama_native') and row.id != 'azure_openai':
                            self.assertEqual(sent.get('model'), self.MODEL)
                        if row.adapter == 'elevenlabs':
                            self.assertEqual(sent['model_id'], self.MODEL)
                        if row.adapter == 'openai_compat' and stream:
                            injected = row.include_usage and surface in INCLUDE_USAGE_SURFACES
                            self.assertEqual(sent.get('stream_options'), {'include_usage': True} if injected else None)
                        self.assertFalse({'worker', 'location', 'timeout'} & set(sent))
                    elif raw_request.form is not None:
                        self.assertIn(self.MODEL.encode(), content)
        self.check_catalog(row, upstream)
        self.check_usage(row)

    def check_catalog(self, row, upstream):
        seen = []
        bodies = {'openai_compat': {'object': 'list', 'data': [{'id': self.MODEL}, {'id': 'other-model'}]},
                  'anthropic': {'data': [{'id': self.MODEL, 'type': 'model'}], 'has_more': False},
                  'gemini': {'models': [{'name': 'models/' + self.MODEL, 'supportedGenerationMethods': ['generateContent']}]},
                  'elevenlabs': [{'model_id': self.MODEL}],
                  'ollama_native': {'models': [{'name': self.MODEL, 'model': self.MODEL}]}}
        document = bodies[row.adapter]
        if row.extras.get('catalog_bare_array'):
            document = document['data']

        def handler(request):
            seen.append(request)
            return httpx.Response(200, headers={'content-type': 'application/json'}, content=json.dumps(document).encode())
        adapter = S.ADAPTERS[row.adapter](row, upstream)
        credential = S.handle(upstream.origin(allow_http_lan=False), row.id, 'openai')

        async def go():
            async with S.Egress(transport=httpx.MockTransport(handler), settings=S.Settings()) as egress:
                return await adapter.list_models(S.grant(upstream, row.id, None, credential, purpose='catalog'), egress)
        with self.subTest(profile=row.id, catalog=row.catalog_path):
            models = S.run(go())
            self.assertIn(self.MODEL, [m.split('/', 1)[1] if m.startswith('models/') else m for m in models])
            if row.catalog_path:
                self.assertEqual(len(seen), 1)
                from urllib.parse import urlsplit
                self.assertEqual(seen[0].url.path, urlsplit(upstream.effective_base_url()).path.rstrip('/') + row.catalog_path)
            else:
                self.assertEqual(seen, [])  # a static list: no catalog request at all

    def check_usage(self, row):
        adapter = S.ADAPTERS[row.adapter](row, self.upstream(row))
        samples = {'openai_compat': ('openai.chat', {'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}),
                   'anthropic': ('anthropic.messages', {'usage': {'input_tokens': 3, 'output_tokens': 2}}),
                   'gemini': ('gemini.generateContent', {'usageMetadata': {'promptTokenCount': 3,
                                                                          'candidatesTokenCount': 2}}),
                   'ollama_native': ('ollama.chat', {'prompt_eval_count': 3, 'eval_count': 2})}
        with self.subTest(profile=row.id, usage=True):
            if row.adapter == 'elevenlabs':
                usage = character_usage({'character-cost': '5'}, 'hello')
                self.assertEqual(usage.characters, 5)
                return
            surface, body = samples[row.adapter]
            if surface not in row.surfaces:
                surface = sorted(row.surfaces)[0]
            usage = adapter.usage_from_body(SURFACES[surface], body, {})
            self.assertEqual((usage.input_tokens, usage.output_tokens), (3, 2))

    def test_openai_compatible_rows(self):
        # split-protocol routers (switchyard: auth none, anthropic.* through an inner Anthropic adapter) have their own
        # row check in tests/check_broker_switchyard.py
        rows = [row for row in PROFILES.values() if row.adapter == 'openai_compat'
                and not row.extras.get('surface_adapters')]
        self.assertEqual(len(rows), 22)
        for row in rows:
            # a row whose rerank path carries the model id needs an <org>/<name> model
            self.MODEL = 'nvidia/row-model-1.5' if row.extras.get('rerank_model_path') else type(self).MODEL
            self.check_row(row)

    def test_native_adapter_rows(self):
        rows = [row for row in PROFILES.values() if row.adapter in ('anthropic', 'gemini', 'elevenlabs', 'ollama_native')]
        self.assertEqual({row.id for row in rows}, {'anthropic', 'gemini', 'elevenlabs', 'ollama_cloud'})
        for row in rows:
            self.check_row(row)


class UpstreamAuthField(unittest.TestCase):
    """R11: Upstream.auth picks custom_openai's scheme (None = the profile default)."""

    def test_schema(self):
        self.assertIsNone(Upstream(profile='custom_openai', base_url='https://gw.example.invalid/v1').auth)
        for value in ('bearer', 'azure_api_key', 'none'):
            upstream = Upstream(profile='custom_openai', base_url='https://gw.example.invalid/v1', auth=value)
            self.assertEqual(upstream.effective_auth(), value)
        self.assertEqual(Upstream(profile='openai').effective_auth(), 'bearer')
        with self.assertRaises(ValueError):
            Upstream(profile='openai', auth='azure_api_key')
        with self.assertRaises(ValueError):
            Upstream(profile='custom_openai', base_url='https://gw.example.invalid/v1', auth='goog')
        with self.assertRaises(ValueError):
            Upstream(profile='custom_openai', base_url='https://gw.example.invalid/v1', auth='none', credential='c')
        from titanos.broker.config import BrokerConfig
        cfg = BrokerConfig.model_validate({'upstreams': {'gw': {'profile': 'custom_openai', 'base_url':
                                                                'https://gw.example.invalid/v1', 'auth': 'azure_api_key',
                                                                'credential': 'gwkey'}}})
        self.assertEqual(cfg.upstreams['gw'].auth, 'azure_api_key')
        self.assertEqual(cfg.model_dump(mode='json')['upstreams']['gw']['auth'], 'azure_api_key')

    def test_egress_uses_the_chosen_scheme(self):
        seen = []
        transport = httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(
            200, headers={'content-type': 'application/json'}, content=b'{}'))
        upstream = Upstream(profile='custom_openai', base_url='https://gw.example.invalid/v1', auth='azure_api_key')
        origin = upstream.origin(allow_http_lan=False)
        adapter = OpenAICompatAdapter(PROFILES['custom_openai'], upstream)

        async def go(template):
            async with S.Egress(transport=transport, settings=S.Settings()) as egress:
                credential = S.handle(origin, 'custom_openai', 'openai', template=template)
                result = await adapter.passthrough(S.grant(upstream, 'custom_openai', 'm', credential), egress,
                                                   SURFACES['openai.chat'], raw({'model': 'x', 'messages': []}))
                await drain(result)
        S.run(go('azure_api_key'))
        self.assertEqual(seen[-1].headers['api-key'], SENTINELS['openai'])
        self.assertNotIn('authorization', seen[-1].headers)
        with self.assertRaises(BrokerError) as ctx:
            S.run(go('bearer'))  # the profile allows bearer, the upstream chose azure_api_key
        self.assertEqual(ctx.exception.code, 'egress_refused')
        self.assertEqual(len(seen), 1)


class SourceAssertions(unittest.TestCase):
    def test_adapters_and_codecs_take_authority_only_from_the_grant(self):
        offenders = {}
        for folder in ('adapters', 'translate'):
            for path in sorted((BROKER / folder).glob('*.py')):
                tree = ast.parse(path.read_text(encoding='utf-8'))
                found = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = ('.' * node.level) + (node.module or '')
                        if module.lstrip('.').split('.')[0] in ('config', 'vault') and node.level >= 2 or \
                                module in ('os', 'titanos.broker.config', 'titanos.broker.vault'):
                            found.append(module)
                        if module.endswith(('.config', '.vault')) or module in ('..config', '..vault'):
                            found.append(module)
                    elif isinstance(node, ast.Import):
                        found.extend(a.name for a in node.names if a.name in ('os', 'titanos.broker.config',
                                                                              'titanos.broker.vault'))
                    elif isinstance(node, ast.Attribute) and node.attr in ('environ', 'getenv'):
                        found.append(node.attr)
                if found:
                    offenders[path.name] = found
        self.assertEqual(offenders, {})

    def test_every_adapter_implements_the_contract(self):
        for name, cls in S.ADAPTERS.items():
            with self.subTest(adapter=name):
                self.assertTrue(issubclass(cls, AdapterBase))
                self.assertFalse(getattr(cls, '__abstractmethods__', None))
                for row in PROFILES.values():
                    if row.adapter == name:
                        cls(row, None)  # constructible for every row it serves
                        for surface in row.surfaces:
                            self.assertTrue(cls(row, None).supports(row, SURFACES[surface]), (row.id, surface))
                with self.assertRaises(ValueError):
                    cls(PROFILES['pending_contract'], None)


CELLS = {
    'openai.chat|OAI|P': 'OpenAIPassthrough.test_chat_auth_path_model_rewrite_and_stripping',
    'openai.chat|OAI|P|stream': 'OpenAIPassthrough.test_chat_stream_include_usage_injected_and_stripped',
    'openai.chat|AZ|P': 'AzurePassthrough.test_azure_surfaces_through_the_exact_host',
    'openai.chat|AZ|P|stream': 'AzurePassthrough.test_azure_stream_with_prompt_filter_chunk',
    'openai.completions|OAI|P': 'OpenAIPassthrough.test_completions',
    'openai.completions|OAI|P|stream': 'OpenAIPassthrough.test_completions_and_responses_streams_pass_through',
    'openai.responses|OAI|P|stream': 'OpenAIPassthrough.test_completions_and_responses_streams_pass_through',
    'openai.completions|AZ|P|stream': 'AzurePassthrough.test_azure_completions_stream',
    'openai.completions|AZ|P': 'AzurePassthrough.test_azure_surfaces_through_the_exact_host',
    'openai.responses|OAI|P': 'OpenAIPassthrough.test_responses_store_forced_false_and_handle_bound_routes',
    'openai.responses.retrieve|OAI|P': 'OpenAIPassthrough.test_responses_store_forced_false_and_handle_bound_routes',
    'openai.responses.delete|OAI|P': 'OpenAIPassthrough.test_responses_store_forced_false_and_handle_bound_routes',
    'openai.responses.cancel|OAI|P': 'OpenAIPassthrough.test_responses_store_forced_false_and_handle_bound_routes',
    'openai.responses.input_items|OAI|P': 'OpenAIPassthrough.test_responses_store_forced_false_and_handle_bound_routes',
    'openai.embeddings|OAI|P': 'OpenAIPassthrough.test_embeddings_moderations_and_images',
    'openai.embeddings|AZ|P': 'AzurePassthrough.test_azure_surfaces_through_the_exact_host',
    'openai.audio.speech|OAI|P': 'OpenAIPassthrough.test_speech_binary_stream_and_audio_forms',
    'openai.audio.speech|AZ|P': 'AzurePassthrough.test_azure_surfaces_through_the_exact_host',
    'openai.audio.transcriptions|OAI|P': 'OpenAIPassthrough.test_speech_binary_stream_and_audio_forms',
    'openai.audio.transcriptions|AZ|P': 'AzurePassthrough.test_azure_surfaces_through_the_exact_host',
    'openai.audio.translations|OAI|P': 'OpenAIPassthrough.test_speech_binary_stream_and_audio_forms',
    'openai.audio.translations|AZ|P': 'AzurePassthrough.test_azure_surfaces_through_the_exact_host',
    'openai.images.generations|OAI|P': 'OpenAIPassthrough.test_embeddings_moderations_and_images',
    'openai.images.edits|OAI|P': 'OpenAIPassthrough.test_embeddings_moderations_and_images',
    'openai.images.generations|AZ|P': 'AzurePassthrough.test_azure_surfaces_through_the_exact_host',
    'openai.moderations|OAI|P': 'OpenAIPassthrough.test_embeddings_moderations_and_images',
    'anthropic.messages|ANT|P': 'AnthropicAdapterRow.test_messages_headers_model_and_betas',
    'anthropic.messages|ANT|P|stream': 'AnthropicAdapterRow.test_stream_count_tokens_and_catalog',
    'anthropic.count_tokens|ANT|P': 'AnthropicAdapterRow.test_stream_count_tokens_and_catalog',
    'gemini.generateContent|GEM|P': 'GeminiAdapterRow.test_generate_stream_embed_count_paths_and_header_auth',
    'gemini.streamGenerateContent|GEM|P': 'GeminiAdapterRow.test_generate_stream_embed_count_paths_and_header_auth',
    'gemini.embedContent|GEM|P': 'GeminiAdapterRow.test_generate_stream_embed_count_paths_and_header_auth',
    'gemini.batchEmbedContents|GEM|P': 'GeminiAdapterRow.test_generate_stream_embed_count_paths_and_header_auth',
    'gemini.countTokens|GEM|P': 'GeminiAdapterRow.test_generate_stream_embed_count_paths_and_header_auth',
    'ollama.chat|OLC|P': 'OllamaAdapterRow.test_chat_generate_embed_tags',
    'ollama.generate|OLC|P': 'OllamaAdapterRow.test_chat_generate_embed_tags',
    'ollama.embed|OLC|P': 'OllamaAdapterRow.test_chat_generate_embed_tags',
    'ollama.chat|OLC|P|stream': 'OllamaAdapterRow.test_chat_and_generate_streams',
    'ollama.generate|OLC|P|stream': 'OllamaAdapterRow.test_chat_and_generate_streams',
    'elevenlabs.tts|XI|P': 'ElevenLabsAdapterRow.test_tts_stream_stt_voices',
    'elevenlabs.tts_stream|XI|P': 'ElevenLabsAdapterRow.test_tts_stream_stt_voices',
    'elevenlabs.stt|XI|P': 'ElevenLabsAdapterRow.test_tts_stream_stt_voices',
    'elevenlabs.voices|XI|P': 'ElevenLabsAdapterRow.test_tts_stream_stt_voices',
}


if __name__ == '__main__':
    S.main(CELLS)
