"""Standard OpenAI request fields on local models (titanos.openai_params): forwarded to llama.cpp, typed 400s elsewhere.

Asserted over the live FastAPI app with a synthetic inference seam and a native.proxy stand-in: the llama.cpp sampling
fields (logprobs, top_logprobs, logit_bias, presence_penalty, frequency_penalty, n, seed, stop arrays, json_schema)
reach the native proxy intact for a llama-cpp model and are refused by name on the in-process backends (inert values
served, logit_bias mapped to transformers sequence_bias); n is bounded by TITANOS_MAX_CHOICES; GGUF embedding models
are proxied to llama-server /v1/embeddings (also through Ollama /api/embed) with user named in X-Titanos-Ignored,
dimensions refused by name and the Titanos-only truncation controls refused by name; speech maps client voice names
through the model's voice_map and accepts speed (sherpa-onnx honours it, other backends refuse it by name); the
registration contract (task embeddings, native.pooling, --embeddings argv) holds. No model, no process, no network.
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from titanos import openai_params, request_compat  # noqa: E402
from titanos.config import Config, ModelSpec  # noqa: E402
from titanos.native import NativeWorker, embedding_arguments  # noqa: E402
from titanos.server import Chat, Completion, Embeddings, Speech, create_app  # noqa: E402
from titanos.settings import Settings, _active_settings  # noqa: E402

CHAT = [{'role': 'user', 'content': 'inert fixture'}]
WAV = b'RIFF' + b'\0' * 40


def native_spec(task='chat', **native):
    return ModelSpec(path='unused-local-fixture', task=task, backend='llama-cpp', device='cpu',
                     native={'model_file': 'fixture.gguf', **native})


class Fixture:
    """The app with a transformers chat/embeddings/speech model and llama-cpp chat and embeddings models; the native proxy
    is a spy that answers OpenAI-shaped bodies and records every forwarded request; the engine never loads anything."""

    def __init__(self, settings=None):
        self.temp = tempfile.TemporaryDirectory(prefix='titanos-openai-params-')
        self.home = tempfile.TemporaryDirectory(prefix='titanos-openai-params-home-')
        self.saved_home = os.environ.get('TITANOS_HOME')
        os.environ['TITANOS_HOME'] = self.home.name  # never the caller's broker.json, keys or vault
        root = Path(self.temp.name)
        self.config = Config(models={
            'local': ModelSpec(path=self.temp.name, task='chat'),
            'embed': ModelSpec(path=self.temp.name, task='embeddings'),
            'voice': ModelSpec(path=self.temp.name, task='speech', voice_map={'alloy': 'default', 'nova': 'narrator'}),
            'gguf': native_spec(), 'gguf-embed': native_spec('embeddings', pooling='mean')})
        self.settings = settings or Settings()
        self.app = create_app(self.config, settings=self.settings, output_dir=root / 'outputs', runtime_catalog=False)
        self.engine_calls, self.proxied = [], []
        self.app.state.engine.target.infer = self.infer
        self.patches = [patch.object(self.app.state.native, 'proxy', new=AsyncMock(side_effect=self.proxy)),
                        patch.object(NativeWorker, 'start', new=AsyncMock(side_effect=AssertionError('Native load forbidden')))]
        for item in self.patches:
            item.start()
        self.client = TestClient(self.app, base_url='http://localhost')
        self.client.__enter__()

    def close(self):
        self.client.__exit__(None, None, None)
        for item in self.patches:
            item.stop()
        if self.saved_home is None:
            os.environ.pop('TITANOS_HOME', None)
        else:
            os.environ['TITANOS_HOME'] = self.saved_home
        self.home.cleanup()
        self.temp.cleanup()

    def infer(self, name, task, data, emit=None, cancel=None, **options):
        self.engine_calls.append((name, task, deepcopy(data)))
        if task == 'embeddings':
            return {'vectors': [[.1, .2] for _ in data['input']], 'usage': {'prompt_tokens': 2, 'total_tokens': 2}}
        if task == 'speech':
            return WAV
        if emit:
            emit('Hello')
        return {'text': 'Hello', 'finish_reason': 'stop', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}

    async def proxy(self, request, route, data, **options):
        self.proxied.append({'route': route, 'data': deepcopy(data), 'options': options})
        if route == '/v1/embeddings':
            texts = data['input'] if isinstance(data['input'], list) else [data['input']]
            return JSONResponse({'object': 'list', 'model': data['model'], 'usage': {'prompt_tokens': 3, 'total_tokens': 3},
                                 'data': [{'object': 'embedding', 'index': i, 'embedding': 'AACAPwAAAEA=' if data.get('encoding_format') == 'base64' else [1.0, 2.0]}
                                          for i, _ in enumerate(texts)]})
        choices = [{'index': i, 'message': {'role': 'assistant', 'content': 'native'}, 'finish_reason': 'stop', 'logprobs': None}
                   for i in range(data.get('n') or 1)]
        if route == '/v1/completions':
            choices = [{'index': i, 'text': 'native', 'finish_reason': 'stop', 'logprobs': None} for i in range(data.get('n') or 1)]
        return JSONResponse({'id': 'x', 'object': 'chat.completion', 'model': data['model'], 'choices': choices,
                             'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})


class OpenAIParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = Fixture()
        cls.c = cls.fixture.client

    @classmethod
    def tearDownClass(cls):
        cls.fixture.close()

    def setUp(self):
        self.fixture.engine_calls.clear()
        self.fixture.proxied.clear()

    def refused(self, response, field, backend='transformers'):
        self.assertEqual(response.status_code, 400, response.text)
        body = response.json()['error']
        self.assertEqual(body['code'], 'unsupported_capability')
        self.assertTrue(body['message'].startswith(field), body['message'])
        self.assertIn(backend, body['message'])

    # --- llama-cpp: the fields reach llama-server untouched -----------------------------------------------------------
    def test_native_chat_forwards_every_llama_cpp_field(self):
        schema = {'type': 'json_schema', 'json_schema': {'name': 'r', 'schema': {'type': 'object', 'properties': {'a': {'type': 'string'}}}}}
        body = {'model': 'gguf', 'messages': CHAT, 'logprobs': True, 'top_logprobs': 2, 'logit_bias': {'15339': -100, 'hello': 5},
                'presence_penalty': 1.5, 'frequency_penalty': -0.5, 'n': 2, 'seed': 7, 'stop': ['\n', 'END'], 'response_format': schema,
                'user': 'agent-7', 'max_tokens': 8}
        r = self.c.post('/v1/chat/completions', json=body)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()['choices']), 2)
        self.assertEqual(r.headers['x-titanos-ignored'], 'user')
        forwarded = self.fixture.proxied[-1]
        self.assertEqual(forwarded['route'], '/v1/chat/completions')
        for name in ('logprobs', 'top_logprobs', 'logit_bias', 'presence_penalty', 'frequency_penalty', 'n', 'seed', 'stop', 'response_format'):
            self.assertEqual(forwarded['data'][name], body[name], name)
        self.assertNotIn('user', forwarded['data'])
        self.assertEqual(self.fixture.engine_calls, [])

    def test_native_completions_forward_integer_logprobs_and_name_user(self):
        r = self.c.post('/v1/completions', json={'model': 'gguf', 'prompt': 'The capital of France is', 'logprobs': 2, 'n': 2,
                                                 'presence_penalty': 1, 'logit_bias': {'13': -100}, 'user': 'u', 'max_tokens': 4})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers['x-titanos-ignored'], 'user')
        forwarded = self.fixture.proxied[-1]
        self.assertEqual(forwarded['route'], '/v1/completions')
        self.assertEqual((forwarded['data']['logprobs'], forwarded['data']['n'], forwarded['data']['presence_penalty'], forwarded['data']['logit_bias']),
                         (2, 2, 1.0, {'13': -100.0}))
        self.assertNotIn('user', forwarded['data'])

    def test_native_completions_logprobs_take_the_legacy_shape(self):
        # llama-server answers completions logprobs in its chat shape (logprobs.content[]); the stock SDK types the legacy shape
        # (tokens / token_logprobs / top_logprobs maps / text_offset), so the forwarded body and every SSE chunk are rewritten
        content = [{'id': 1, 'token': ' Paris', 'bytes': [32], 'logprob': -0.8, 'top_logprobs': [{'id': 1, 'token': ' Paris', 'logprob': -0.8}, {'id': 2, 'token': ' Lyon', 'logprob': -2.5}]},
                   {'id': 3, 'token': '.', 'logprob': -0.6, 'top_logprobs': []}]
        body = {'id': 'x', 'object': 'text_completion', 'model': 'gguf', 'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7},
                'choices': [{'index': 0, 'text': ' Paris.', 'finish_reason': 'stop', 'logprobs': {'content': content}}]}
        legacy = {'tokens': [' Paris', '.'], 'token_logprobs': [-0.8, -0.6], 'top_logprobs': [{' Paris': -0.8, ' Lyon': -2.5}, {}], 'text_offset': [0, 6]}
        native = self.fixture.app.state.native
        with patch.object(native, 'proxy', new=AsyncMock(side_effect=lambda *a, **k: JSONResponse(deepcopy(body), headers={'X-Native': 'yes'}))):
            r = self.c.post('/v1/completions', json={'model': 'gguf', 'prompt': 'The capital of France is', 'logprobs': 2})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()['choices'][0]['logprobs'], legacy)
            self.assertEqual((r.json()['usage'], r.json()['choices'][0]['text'], r.headers.get('x-native')), (body['usage'], ' Paris.', 'yes'))
            # a chat request keeps llama-server's chat shape; a completion that asked for no logprobs is forwarded untouched
            r = self.c.post('/v1/chat/completions', json={'model': 'gguf', 'messages': CHAT, 'logprobs': True})
            self.assertEqual(r.json()['choices'][0]['logprobs'], {'content': content})
            self.assertEqual(self.c.post('/v1/completions', json={'model': 'gguf', 'prompt': 'x'}).json(), body)
        frames = [{'choices': [{'index': 0, 'text': ' Paris', 'finish_reason': None, 'logprobs': {'content': content[:1]}}], 'object': 'text_completion'},
                  {'choices': [{'index': 0, 'text': '.', 'finish_reason': 'stop', 'logprobs': {'content': content[1:]}}], 'object': 'text_completion'}]
        sse = ''.join(f'data: {json.dumps(frame)}\n\n' for frame in frames) + 'data: [DONE]\n\n'
        with patch.object(native, 'proxy', new=AsyncMock(side_effect=lambda *a, **k: StreamingResponse(iter([sse.encode('utf-8')]), media_type='text/event-stream', headers={'X-Native': 'yes'}))):
            r = self.c.post('/v1/completions', json={'model': 'gguf', 'prompt': 'x', 'logprobs': 2, 'stream': True})
            self.assertEqual((r.status_code, r.headers.get('x-native')), (200, 'yes'))
            lines = [line for line in r.text.split('\n') if line.startswith('data:')]
            self.assertEqual(lines[-1], 'data: [DONE]')
            chunks = [json.loads(line[5:]) for line in lines[:-1]]
            self.assertEqual([chunk['choices'][0]['logprobs'] for chunk in chunks],
                             [{'tokens': [' Paris'], 'token_logprobs': [-0.8], 'top_logprobs': [{' Paris': -0.8, ' Lyon': -2.5}], 'text_offset': [0]},
                              {'tokens': ['.'], 'token_logprobs': [-0.6], 'top_logprobs': [{}], 'text_offset': [6]}], 'text_offset continues across chunks')
            self.assertEqual([chunk['choices'][0]['text'] for chunk in chunks], [' Paris', '.'])

    def test_ollama_options_penalties_forward_to_llama_cpp(self):
        r = self.c.post('/api/chat', json={'model': 'gguf', 'messages': CHAT, 'stream': False,
                                           'options': {'presence_penalty': 1.0, 'frequency_penalty': 0.5, 'seed': 3}})
        self.assertEqual(r.status_code, 200, r.text)
        data = self.fixture.proxied[-1]['data']
        self.assertEqual((data['presence_penalty'], data['frequency_penalty'], data['seed']), (1.0, 0.5, 3))

    # --- in-process backends: typed refusals naming the field, inert values served ---------------------------------
    def test_local_backend_refuses_live_values_by_name(self):
        for field, value in (('logprobs', True), ('top_logprobs', 3), ('presence_penalty', .5), ('frequency_penalty', -1), ('n', 2)):
            with self.subTest(field=field):
                self.refused(self.c.post('/v1/chat/completions', json={'model': 'local', 'messages': CHAT, field: value}), field)
        self.refused(self.c.post('/v1/completions', json={'model': 'local', 'prompt': 'x', 'logprobs': 1}), 'logprobs')
        self.refused(self.c.post('/api/chat', json={'model': 'local', 'messages': CHAT, 'stream': False, 'options': {'presence_penalty': 1}}), 'presence_penalty')
        self.assertEqual(self.fixture.engine_calls, [])

    def test_inert_values_are_served_and_not_forwarded_to_the_engine(self):
        r = self.c.post('/v1/chat/completions', json={'model': 'local', 'messages': CHAT, 'logprobs': False, 'top_logprobs': 0,
                                                      'presence_penalty': 0, 'frequency_penalty': 0.0, 'n': 1, 'logit_bias': {}})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['choices'][0]['message']['content'], 'Hello')
        _, _, data = self.fixture.engine_calls[-1]
        self.assertFalse({'logprobs', 'top_logprobs', 'presence_penalty', 'frequency_penalty', 'n', 'logit_bias'} & set(data))

    def test_logit_bias_reaches_transformers_as_sequence_bias(self):
        r = self.c.post('/v1/chat/completions', json={'model': 'local', 'messages': CHAT, 'logit_bias': {'42': -100, '7': 2.5}})
        self.assertEqual(r.status_code, 200, r.text)
        _, _, data = self.fixture.engine_calls[-1]
        self.assertEqual(data['logit_bias'], {'42': -100.0, '7': 2.5})
        self.assertEqual(openai_params.transformers_bias(data), {'sequence_bias': [[[42], -100.0], [[7], 2.5]]})
        self.assertEqual(openai_params.transformers_bias({'logit_bias': {}}), {})
        with self.assertRaisesRegex(ValueError, 'logit_bias'):
            openai_params.transformers_bias({'logit_bias': {'hello': 1}})

    def test_schema_bounds_hold(self):
        for body in ({'presence_penalty': 2.5}, {'frequency_penalty': -2.5}, {'top_logprobs': 21}, {'n': 0},
                     {'logit_bias': {str(i): 1 for i in range(301)}}, {'logit_bias': {'1': 'x'}}):
            with self.subTest(body=body):
                self.assertEqual(self.c.post('/v1/chat/completions', json={'model': 'gguf', 'messages': CHAT, **body}).status_code, 422)
        self.assertEqual(self.c.post('/v1/completions', json={'model': 'gguf', 'prompt': 'x', 'logprobs': 6}).status_code, 422)
        self.assertEqual(self.c.post('/v1/completions', json={'model': 'gguf', 'prompt': 'x', 'logprobs': True}).status_code, 422)

    def test_n_is_bounded_by_max_choices(self):
        token = _active_settings.set(Settings(max_choices=2))
        try:
            with self.assertRaisesRegex(ValidationError, 'TITANOS_MAX_CHOICES=2'):
                Chat(model='gguf', messages=CHAT, n=3)
            self.assertEqual(Chat(model='gguf', messages=CHAT, n=2).n, 2)
            self.assertEqual(Completion(model='gguf', prompt='x', n=2).n, 2)
        finally:
            _active_settings.reset(token)
        bounded = Fixture(Settings(max_choices=2))
        try:
            r = bounded.client.post('/v1/chat/completions', json={'model': 'gguf', 'messages': CHAT, 'n': 3})
            self.assertEqual(r.status_code, 422, r.text)
            self.assertIn('TITANOS_MAX_CHOICES=2', r.json()['error']['message'])
            self.assertEqual(bounded.client.post('/v1/chat/completions', json={'model': 'gguf', 'messages': CHAT, 'n': 2}).status_code, 200)
            self.assertEqual(bounded.proxied[-1]['data']['n'], 2)
        finally:
            bounded.close()

    # --- embeddings ----------------------------------------------------------------------------------------------------
    def test_gguf_embedding_model_is_proxied_to_llama_server(self):
        r = self.c.post('/v1/embeddings', json={'model': 'gguf-embed', 'input': ['a', 'b'], 'encoding_format': 'base64', 'user': 'u'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers['x-titanos-ignored'], 'user')
        self.assertEqual([row['embedding'] for row in r.json()['data']], ['AACAPwAAAEA=', 'AACAPwAAAEA='])
        forwarded = self.fixture.proxied[-1]
        self.assertEqual(forwarded['route'], '/v1/embeddings')
        self.assertEqual(forwarded['data'], {'model': 'gguf-embed', 'input': ['a', 'b'], 'encoding_format': 'base64'})
        self.assertEqual(self.fixture.engine_calls, [])

    def test_gguf_embedding_refusals_name_the_field(self):
        r = self.c.post('/v1/embeddings', json={'model': 'gguf-embed', 'input': 'a', 'dimensions': 8})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertTrue(r.json()['error']['message'].startswith('dimensions'), r.text)
        for field, value in (('max_tokens', 16), ('truncate', True)):
            with self.subTest(field=field):
                self.refused(self.c.post('/v1/embeddings', json={'model': 'gguf-embed', 'input': 'a', field: value}), field, 'llama-cpp')
        self.refused(self.c.post('/v1/chat/completions', json={'model': 'gguf-embed', 'messages': CHAT}), 'gguf-embed is a llama-cpp embeddings model', 'embeddings')
        self.assertEqual(self.fixture.proxied, [])

    def test_ollama_embed_routes_reach_the_gguf_embedding_model(self):
        r = self.c.post('/api/embed', json={'model': 'gguf-embed', 'input': ['a', 'b']})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((r.json()['model'], r.json()['embeddings'], r.json()['prompt_eval_count']), ('gguf-embed', [[1.0, 2.0], [1.0, 2.0]], 3))
        r = self.c.post('/api/embeddings', json={'model': 'gguf-embed', 'prompt': 'a'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['embedding'], [1.0, 2.0])
        self.assertEqual([p['route'] for p in self.fixture.proxied], ['/v1/embeddings', '/v1/embeddings'])
        r = self.c.post('/api/embed', json={'model': 'gguf-embed', 'input': 'a', 'dimensions': 4})
        self.assertEqual(r.status_code, 400, r.text)

    def test_local_embeddings_name_user_and_refuse_dimensions(self):
        r = self.c.post('/v1/embeddings', json={'model': 'embed', 'input': 'a', 'user': 'u'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers['x-titanos-ignored'], 'user')
        self.assertEqual(r.json()['data'][0]['embedding'], [.1, .2])
        self.assertNotIn('user', self.fixture.engine_calls[-1][2])
        r = self.c.post('/v1/embeddings', json={'model': 'embed', 'input': 'a', 'dimensions': 8})
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()['error']['code'], 'unsupported_capability')
        self.assertTrue(r.json()['error']['message'].startswith('dimensions'))
        self.assertEqual(self.c.post('/v1/embeddings', json={'model': 'embed', 'input': 'a', 'dimensions': None}).headers['x-titanos-ignored'], 'dimensions')

    # --- speech --------------------------------------------------------------------------------------------------------
    def test_speech_speed_passes_the_schema_and_voice_names_reach_the_engine(self):
        r = self.c.post('/v1/audio/speech', json={'model': 'voice', 'input': 'Hi', 'voice': 'alloy', 'speed': 1.5})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.content.startswith(b'RIFF'))
        _, task, data = self.fixture.engine_calls[-1]
        self.assertEqual((task, data['voice'], data['speed']), ('speech', 'alloy', 1.5))  # the map is applied inside Engine.infer (below)
        self.assertEqual(self.c.post('/v1/audio/speech', json={'model': 'voice', 'input': 'Hi', 'speed': 5}).status_code, 422)
        self.assertEqual(self.c.post('/v1/audio/speech', json={'model': 'voice', 'input': 'Hi', 'speed': 0.1}).status_code, 422)
        r = self.c.post('/v1/text-to-speech/nova', json={'text': 'Hi', 'model_id': 'voice'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.fixture.engine_calls[-1][2]['voice'], 'nova')
        self.assertEqual(Speech(model='voice', input='Hi', speed=0.25).speed, 0.25)
        with self.assertRaisesRegex(ValueError, 'speed 1.5 is not available on this xtts speech model'):
            openai_params.speech_speed({'speed': 1.5}, 'xtts')
        openai_params.speech_speed({'speed': 1.0}, 'transformers')
        openai_params.speech_speed({}, 'transformers')

    def test_engine_applies_the_voice_map_for_every_speech_caller(self):
        """Engine.infer is the one choke point under the sync route, speech asset jobs, the ElevenLabs shape and MCP."""
        from titanos.engine import Engine
        engine = Engine(self.fixture.config)
        seen = []

        class Runtime:
            def infer(self, task, data, emit=None, cancel=None):
                seen.append((task, dict(data)))
                return WAV
        engine.runtime, engine.current, engine.states['voice'] = Runtime(), 'voice', 'loaded'
        engine._load = lambda name: None
        self.assertEqual(engine.infer('voice', 'speech', {'model': 'voice', 'input': 'Hi', 'voice': 'alloy', 'speed': 1.5}), WAV)
        self.assertEqual(seen[-1][1]['voice'], 'default')
        self.assertEqual(seen[-1][1]['speed'], 1.5)
        engine.infer('voice', 'speech', {'model': 'voice', 'input': 'Hi', 'voice': 'nova'})
        self.assertEqual(seen[-1][1]['voice'], 'narrator')
        engine.infer('voice', 'speech', {'model': 'voice', 'input': 'Hi', 'voice': 'custom'})
        self.assertEqual(seen[-1][1]['voice'], 'custom')  # outside the map: the backend's own check decides
        engine.current, engine.states['local'] = 'local', 'loaded'
        engine.infer('local', 'chat', {'model': 'local', 'messages': CHAT, 'voice': 'alloy'})
        self.assertEqual(seen[-1][1]['voice'], 'alloy')  # the map is for speech only

    def test_voice_map_registration_contract(self):
        with self.assertRaisesRegex(ValueError, 'voice_map applies to speech'):
            ModelSpec(path='x', task='chat', voice_map={'alloy': 'default'})
        with self.assertRaisesRegex(ValueError, 'nonblank'):
            ModelSpec(path='x', task='speech', voice_map={'alloy': ''})
        spec = ModelSpec(path='x', task='speech', backend='sherpa-onnx', voice_map={'alloy': '0', 'echo': '3'})
        self.assertEqual(openai_params.speech_voice(spec, {'voice': 'echo'}), {'voice': '3'})
        self.assertEqual(openai_params.speech_voice(spec, {'voice': 'other'}), {'voice': 'other'})

    # --- registration and runtime contract for GGUF embedding models --------------------------------------------------
    def test_llama_cpp_embeddings_registration_and_argv(self):
        spec = native_spec('embeddings', pooling='mean')
        self.assertEqual(embedding_arguments(spec), ['--embeddings', '--pooling', 'mean'])
        self.assertEqual(embedding_arguments(native_spec('embeddings')), ['--embeddings'])
        self.assertEqual(embedding_arguments(native_spec()), [])
        with self.assertRaisesRegex(ValueError, 'pooling applies to llama-cpp embeddings'):
            native_spec(pooling='mean')
        with self.assertRaisesRegex(ValueError, 'chat, vision or embeddings'):
            native_spec('transcription')
        with self.assertRaises(ValidationError):
            native_spec('embeddings', pooling='average')
        catalog = {row['id']: row for row in self.fixture.app.state.native.catalog()}
        self.assertEqual(catalog['gguf-embed']['capabilities'], ['embeddings'])
        self.assertIn('tools', catalog['gguf']['capabilities'])

    # --- the compat split moved onto Payload: the other routes keep their behaviour ------------------------------------
    def test_compat_tables_and_other_routes(self):
        self.assertNotIn('top_logprobs', request_compat.CHAT)
        self.assertIn('top_logprobs', request_compat.RESPONSES)
        self.assertEqual(request_compat.EMBEDDINGS, {'user': str})
        self.assertIn('dimensions', request_compat.REFUSED['embeddings'])
        self.assertIn('user', request_compat.COMPLETION)
        self.assertEqual(request_compat.split(None, {'a': 1}), ({'a': 1}, (), None))
        self.assertEqual(request_compat.split('chat', {'model': 'm', 'options': {'num_ctx': 4096, 'num_gpu': 1}}), ({'model': 'm'}, ('options.num_gpu',), 4096))
        r = self.c.post('/v1/responses', json={'model': 'local', 'input': 'Hi', 'include': ['x'], 'user': 'u'})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers['x-titanos-ignored'], 'include, user')
        self.assertEqual(self.c.post('/v1/responses', json={'model': 'local', 'input': 'Hi', 'store': True}).status_code, 400)
        r = self.c.post('/v1/chat/completions', json={'model': 'local', 'messages': CHAT, 'metadata': {'k': 'v'}, 'store': False, 'top_logprobs': 0})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers['x-titanos-ignored'], 'metadata, store')
        self.assertEqual(self.c.post('/v1/images/generations', json={'model': 'local', 'prompt': 'x', 'user': 'u'}).status_code, 422)  # no compat kind: strict

    def test_unknown_fields_are_strict_by_default(self):
        self.assertEqual(self.c.post('/v1/chat/completions', json={'model': 'local', 'messages': CHAT, 'x_extra': 1}).status_code, 422)
        self.assertEqual(self.c.post('/api/chat', json={'model': 'local', 'messages': CHAT, 'stream': False, 'x_extra': 1}).status_code, 422)
        with self.assertRaisesRegex(ValueError, 'TITANOS_UNKNOWN_FIELDS'):
            Settings(unknown_fields='lenient')
        with patch.dict(os.environ, {'TITANOS_UNKNOWN_FIELDS': 'ignore'}):
            self.assertEqual(Settings.from_env().unknown_fields, 'ignore')


class UnknownFields(unittest.TestCase):
    """TITANOS_UNKNOWN_FIELDS=ignore: honoured only in managed role, on the OpenAI and Ollama JSON routes."""

    def test_managed_ignore_names_unknown_top_level_fields(self):
        fixture = Fixture(Settings(role='managed', unknown_fields='ignore'))
        try:
            c = fixture.client
            r = c.post('/v1/chat/completions', json={'model': 'local', 'messages': CHAT, 'user': 'u', 'x_extra': 1, 'y': None})
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.headers['x-titanos-ignored'], 'user, x_extra, y')
            for route, body in (('/v1/completions', {'prompt': 'Hi'}), ('/v1/responses', {'input': 'Hi'}), ('/v1/embeddings', {'input': 'Hi'})):
                r = c.post(route, json={'model': 'embed' if 'embed' in route else 'local', **body, 'x_extra': 1})
                self.assertEqual((r.status_code, r.headers.get('x-titanos-ignored')), (200, 'x_extra'), route)
            for route, body in (('/api/chat', {'model': 'local', 'messages': CHAT, 'stream': False}), ('/api/generate', {'model': 'local', 'prompt': 'Hi', 'stream': False}),
                                ('/api/embed', {'model': 'embed', 'input': 'Hi'}), ('/api/embeddings', {'model': 'embed', 'prompt': 'Hi'})):
                r = c.post(route, json={**body, 'x_extra': 1})
                self.assertEqual((r.status_code, r.headers.get('x-titanos-ignored')), (200, 'x_extra'), route + r.text)
            for route, body in (('/v1/chat/completions', {'stream': True}), ('/api/chat', {})):  # streamed answers carry it too
                r = c.post(route, json={'model': 'local', 'messages': CHAT, **body, 'x_extra': 1})
                self.assertEqual((r.status_code, r.headers.get('x-titanos-ignored')), (200, 'x_extra'), route)
            # nested unknown fields and wrong-typed known fields stay errors
            self.assertEqual(c.post('/v1/chat/completions', json={'model': 'local', 'messages': [{**CHAT[0], 'x': 1}]}).status_code, 422)
            self.assertEqual(c.post('/v1/chat/completions', json={'model': 'local', 'messages': CHAT, 'max_tokens': 'many'}).status_code, 422)
            self.assertEqual(c.post('/api/chat', json={'model': 'local', 'messages': CHAT, 'options': {'x': 1}}).status_code, 422)
            # routes outside the OpenAI and Ollama JSON schemas stay strict
            self.assertEqual(c.post('/v1/images/generations', json={'model': 'local', 'prompt': 'x', 'x_extra': 1}).status_code, 422)
        finally:
            fixture.close()

    def test_ignore_outside_managed_role_stays_strict(self):
        for role in ('standalone', 'managed'):
            token = _active_settings.set(Settings(role=role, unknown_fields='ignore' if role == 'standalone' else 'strict'))
            try:
                with self.assertRaises(ValidationError):
                    Chat.model_validate({'model': 'local', 'messages': CHAT, 'x_extra': 1})
            finally:
                _active_settings.reset(token)


if __name__ == '__main__':
    unittest.main(verbosity=1)
