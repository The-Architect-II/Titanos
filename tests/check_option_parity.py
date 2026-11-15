"""Configuration precedence and fail-closed native controls; no model or process."""
import asyncio
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pydantic import ValidationError
from titanos.config import Config, ModelSpec, WorkerSpec
from titanos.engine import EngineError
from titanos.native import NativePool, NativeWorker
from titanos.protocols import Options, Ollama, register_protocols
from titanos.request_defaults import model_defaults
from titanos.server import Chat, Completion, Responses, create_app
from titanos.settings import Settings, _active_settings


def native_spec(defaults=None):
    return ModelSpec(path='unused-local-fixture', task='chat', backend='llama-cpp', device='cpu',
                     native={'model_file': 'fixture.gguf'}, defaults=defaults or {})


class OptionParity(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(default_max_tokens=300, max_tokens=32768,
                                 default_temperature=.6, default_top_p=.9, worker_timeout=17)
        token = _active_settings.set(self.settings)
        self.addCleanup(_active_settings.reset, token)

    def chat(self, **values):
        return Chat(model='model', messages=[{'role': 'user', 'content': 'inert fixture'}], **values)

    def native_responses(self, defaults, **body):
        import httpx
        from fastapi.responses import JSONResponse, StreamingResponse
        async def exchange():
            with tempfile.TemporaryDirectory(prefix='titanos-response-defaults-') as directory:
                spec = native_spec(defaults)
                app = create_app(Config(models={'model': spec}), settings=self.settings,
                                 output_dir=Path(directory) / 'outputs', runtime_catalog=False)
                observed = []
                async def proxy(request, route, data, **options):
                    observed.append({'route': route, 'data': deepcopy(data), 'options': options})
                    if data.get('stream'):
                        async def events():
                            yield b'event: response.completed\ndata: {"type":"response.completed","fixture":true}\n\n'
                        return StreamingResponse(events(), media_type='text/event-stream')
                    return JSONResponse({'fixture': True, 'forwarded': data}, status_code=201)
                try:
                    with patch.object(app.state.native, 'proxy', new=AsyncMock(side_effect=proxy)), \
                         patch.object(NativeWorker, 'start', new=AsyncMock(side_effect=AssertionError('Native load forbidden'))) as start, \
                         patch.object(app.state.engine.target, 'infer', side_effect=AssertionError('Model inference forbidden')) as infer:
                        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
                            response = await client.post('/v1/responses', json={'model': 'model', 'input': 'inert fixture', **body})
                        start.assert_not_called(); infer.assert_not_called()
                        self.assertTrue(all(worker.process is None for worker in app.state.native.workers))
                    return response, observed
                finally:
                    await app.state.native.close()
                    app.state.engine.target.close()
        return asyncio.run(exchange())

    def test_native_responses_applies_registered_defaults(self):
        response, observed = self.native_responses({'max_tokens': 71, 'temperature': .3, 'top_p': .8})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(observed[0]['route'], '/v1/responses')
        data = observed[0]['data']
        self.assertEqual((data['max_output_tokens'], data['temperature'], data['top_p']), (71, .3, .8))

    def test_native_responses_rejects_unsupported_default_before_proxy(self):
        response, observed = self.native_responses({'repeat_last_n': 128})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()['error']['code'], 'unsupported_capability')
        self.assertEqual(observed, [])

    def test_native_responses_aliases_and_explicit_request_precedence(self):
        for alias in ('max_tokens', 'max_new_tokens', 'max_output_tokens', 'max_completion_tokens'):
            defaults = {alias: 71, 'temperature': .3, 'top_p': .8}
            for values, expected in (({}, (71, .3, .8)),
                                     ({'max_output_tokens': 13, 'temperature': 0, 'top_p': .4}, (13, 0, .4))):
                with self.subTest(alias=alias, values=values):
                    response, observed = self.native_responses(defaults, **values)
                    self.assertEqual(response.status_code, 201, response.text)
                    data = observed[0]['data']
                    self.assertEqual((data['max_output_tokens'], data['temperature'], data['top_p']), expected)
                    self.assertFalse({'max_tokens', 'max_new_tokens', 'max_completion_tokens'} & set(data))
                    self.assertEqual(defaults, {alias: 71, 'temperature': .3, 'top_p': .8})

    def test_native_responses_false_and_optional_null_do_not_inherit(self):
        defaults = {'max_completion_tokens': None, 'parallel_tool_calls': True, 'stream': True,
                    'tools': [{'type': 'function', 'name': 'fixture', 'parameters': {}}],
                    'tool_choice': 'auto', 'text': {'format': {'type': 'json_object'}},
                    'metadata': {'source': 'model'}, 'instructions': 'model instruction'}
        response, observed = self.native_responses(defaults, parallel_tool_calls=False, stream=False,
            tools=None, tool_choice=None, text=None, metadata=None, instructions=None, store=False)
        self.assertEqual(response.status_code, 201, response.text)
        data = observed[0]['data']
        self.assertEqual(data['max_output_tokens'], 300)
        self.assertIs(data['parallel_tool_calls'], False)
        self.assertIs(data['stream'], False); self.assertIs(data['store'], False)
        self.assertFalse({'tools', 'tool_choice', 'text', 'metadata', 'instructions'} & set(data))
        for field in ('max_output_tokens', 'temperature', 'top_p'):
            with self.subTest(field=field):
                response, observed = self.native_responses({'max_tokens': 71, 'temperature': .3, 'top_p': .8}, **{field: None})
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(observed, [])

    def test_native_responses_validate_effective_limits_and_protocol_defaults(self):
        for defaults in ({'max_tokens': 32769}, {'max_output_tokens': 0}, {'temperature': float('nan')}, {'top_p': 0}):
            with self.subTest(defaults=defaults):
                response, observed = self.native_responses(defaults)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(observed, [])
        response, observed = self.native_responses({'max_completion_tokens': 16384})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(observed[0]['data']['max_output_tokens'], 16384)
        # An explicit valid request overrides an unused out-of-range model budget.
        response, observed = self.native_responses({'max_tokens': 32769}, max_output_tokens=11)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(observed[0]['data']['max_output_tokens'], 11)
        response, observed = self.native_responses({'top_k': 40})
        self.assertEqual(response.status_code, 400, response.text)  # Chat-only field is not added to Responses.
        self.assertEqual(observed, [])

    def test_native_responses_preserves_native_tool_text_store_and_stream_envelope(self):
        body = {
            'input': [{'type': 'message', 'role': 'developer', 'content': [{'type': 'input_text', 'text': 'inert instruction'}]},
                      {'type': 'function_call_output', 'call_id': 'call_fixture', 'output': 'inert tool result'}],
            'instructions': 'Retain the typed native envelope', 'max_output_tokens': 23,
            'tools': [{'type': 'function', 'name': 'fixture', 'parameters': {'type': 'object'}}],
            'tool_choice': {'type': 'function', 'name': 'fixture'}, 'parallel_tool_calls': False,
            'text': {'format': {'type': 'json_schema', 'name': 'fixture', 'schema': {'type': 'object'}}},
            'metadata': {'case': 'native-stream-envelope'}, 'store': False, 'stream': True,
        }
        response, observed = self.native_responses({'max_tokens': 71, 'temperature': .3}, **body)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers['content-type'].startswith('text/event-stream'))
        self.assertEqual(response.content, b'event: response.completed\ndata: {"type": "response.completed", "fixture": true, "sequence_number": 0}\n\n')  # native_stream: the envelope kept, the sequence number added
        self.assertEqual(observed[0]['route'], '/v1/responses')
        for key, value in body.items():
            self.assertEqual(observed[0]['data'][key], value)
        self.assertEqual(observed[0]['data']['temperature'], .3)

    def test_request_model_environment_precedence(self):
        spec = native_spec({'max_tokens': 80, 'temperature': .3, 'top_p': .8})
        base = model_defaults(self.chat(), native_spec())
        self.assertEqual((base.max_tokens, base.temperature, base.top_p), (300, .6, .9))
        inherited = model_defaults(self.chat(), spec)
        self.assertEqual((inherited.max_tokens, inherited.temperature, inherited.top_p), (80, .3, .8))
        explicit = model_defaults(self.chat(max_tokens=32, temperature=0, top_p=.4), spec)
        self.assertEqual((explicit.max_tokens, explicit.temperature, explicit.top_p), (32, 0, .4))
        self.assertEqual(spec.defaults, {'max_tokens': 80, 'temperature': .3, 'top_p': .8})

    def test_completion_model_token_aliases(self):
        for alias in ('max_tokens', 'max_new_tokens', 'max_output_tokens', 'max_completion_tokens'):
            with self.subTest(alias=alias):
                result = model_defaults(Completion(model='model', prompt='fixture'), native_spec({alias: 71}))
                self.assertEqual(result.max_tokens, 71)

    def test_request_alias_overrides_either_model_alias(self):
        for key in ('max_tokens', 'max_completion_tokens'):
            for defaults in ({'max_tokens': 71}, {'max_completion_tokens': 73}):
                with self.subTest(key=key, defaults=defaults):
                    result = model_defaults(self.chat(**{key: 13}), native_spec(defaults))
                    effective = result.max_completion_tokens if result.max_completion_tokens is not None else result.max_tokens
                    self.assertEqual(effective, 13)

    def test_explicit_zero_false_and_null_remain_explicit(self):
        defaults = {'temperature': .7, 'top_k': 20, 'min_p': .2, 'seed': 9,
                    'parallel_tool_calls': True, 'stop': ['END'], 'max_tokens': 71}
        result = model_defaults(self.chat(temperature=0, top_k=0, min_p=0, seed=0,
                                         parallel_tool_calls=False, stop=None,
                                         max_completion_tokens=None), native_spec(defaults))
        self.assertEqual((result.temperature, result.top_k, result.min_p, result.seed), (0, 0, 0, 0))
        self.assertIs(result.parallel_tool_calls, False)
        self.assertIsNone(result.stop)
        # Existing optional-null contract suppresses the model's token default.
        self.assertEqual(result.max_tokens, 300)
        with self.assertRaises(ValidationError): self.chat(max_tokens=None)

    def test_null_completion_alias_default_stays_unspecified(self):
        result = model_defaults(Completion(model='model', prompt='fixture'), native_spec({'max_completion_tokens': None}))
        self.assertEqual(result.max_tokens, 300)

    def test_unknown_native_defaults_fail_instead_of_disappearing(self):
        for key, value in [('repeat_last_n', 128), ('typical_p', .9),
                           ('chat_template_kwargs', {'enable_thinking': False}), ('misspelled_temperature', .1)]:
            with self.subTest(key=key):
                with self.assertRaisesRegex(EngineError, key) as error:
                    model_defaults(self.chat(), native_spec({key: value}))
                self.assertEqual(error.exception.code, 'unsupported_capability')

    def test_internal_native_completion_refuses_before_worker_selection(self):
        pool = NativePool.__new__(NativePool)
        pool.models = {'model': native_spec({'repeat_last_n': 128})}
        pool.candidates = Mock(side_effect=AssertionError('Unsupported option reached worker selection'))
        with self.assertRaisesRegex(EngineError, 'repeat_last_n'):
            asyncio.run(pool.complete({'model': 'model', 'messages': [{'role': 'user', 'content': 'fixture'}]}))
        pool.candidates.assert_not_called()

    def test_adapter_metadata_reserved_fields_and_other_backends_preserved(self):
        defaults = {'min_score': .6, 'min_margin': .1, 'sentiment_prompt_version': 'label-v2',
                    'sentiment_decoding': 'native_json_schema', 'retrieval_prompt_version': 'retrieval-answer-v2',
                    'require_loaded': True, 'worker': 'not-inherited', 'model': 'not-inherited', 'task': 'not-inherited'}
        result = model_defaults(self.chat(), native_spec(defaults))
        self.assertFalse(result.require_loaded); self.assertIsNone(result.worker); self.assertEqual(result.model, 'model')
        transformer = ModelSpec(path='unused-local-fixture', task='chat', defaults={'repetition_penalty': 1.2})
        model_defaults(self.chat(), transformer)
        self.assertEqual(transformer.defaults, {'repetition_penalty': 1.2})

    def test_configured_ceiling_is_raiseable_and_bad_values_reject(self):
        self.assertEqual(model_defaults(self.chat(), native_spec({'max_tokens': 16384})).max_tokens, 16384)
        with self.assertRaisesRegex(EngineError, 'current request limits'):
            model_defaults(self.chat(), native_spec({'max_tokens': 32769}))
        options = Options(num_predict=16384, temperature=0, top_k=0, min_p=0, seed=0)
        self.assertEqual(options.num_predict, 16384)
        self.assertEqual(Options(num_ctx=8192).num_ctx, 8192)  # accepted; admitted against the native context by protocols.context_window
        for values in ({'num_predict': -1}, {'num_ctx': 0}, {'repeat_last_n': 128}):
            with self.subTest(values=values), self.assertRaises(ValidationError): Options(**values)

    def test_native_worker_timeout_uses_settings_then_explicit_worker(self):
        implicit = NativePool(Config(models={'model': native_spec()}), settings=self.settings)
        self.assertEqual(implicit.workers[0].timeout, 17)
        self.assertIsNone(implicit.workers[0].process)
        assigned = NativePool(Config(models={'model': native_spec()}, workers={
            'worker': WorkerSpec(models=['model'], device='cpu', timeout=23)}), settings=self.settings)
        self.assertEqual(assigned.workers[0].timeout, 23)
        self.assertIsNone(assigned.workers[0].process)

    def test_actual_ollama_generate_bridge_uses_completion_default_and_override(self):
        routes = {}
        class Routes:
            def __getattr__(self, method):
                def register(path, **kwargs):
                    def save(function):
                        routes[(method, path)] = function
                        return function
                    return save
                return register
        spec = native_spec({'max_completion_tokens': 71})
        observed = []
        async def generate(payload, task, request):
            observed.append((task, model_defaults(payload, spec).max_tokens))
            return {'choices': [{'text': 'inert fixture', 'finish_reason': 'stop'}], 'usage': {}}
        register_protocols(Routes(), Config(models={'model': spec}), None, generate, None, None)
        route = routes[('post', '/api/generate')]
        for options in ({}, {'num_predict': 13}):
            asyncio.run(route(Ollama(model='model', prompt='fixture', stream=False, options=options), None))
        self.assertEqual(observed, [('completion', 71), ('completion', 13)])


if __name__ == '__main__': unittest.main(verbosity=2)
