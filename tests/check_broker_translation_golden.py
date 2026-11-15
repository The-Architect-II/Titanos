"""Broker canonical translation (WP5): golden round trips through translate/ir.py, both directions, streaming and not.

Pure codec goldens:
- openai.chat <-> anthropic, openai.chat <-> gemini, openai.chat <-> ollama; responses -> chat, anthropic and gemini;
  anthropic -> openai, gemini -> openai, ollama -> openai and anthropic;
- tool ids kept (and synthesized deterministically for Gemini and Ollama), every tool_choice shape, several tool results
  merged into ONE Anthropic user turn, consecutive same-role turns merged;
- image parts (base64 and https url; Gemini refuses https urls);
- thinking <-> reasoning_content with the signature kept;
- stop reasons and usage, with each provider's token meaning normalized to the IR (OpenAI) meaning: Anthropic cache reads
  and cache writes are added to input_tokens, Gemini thoughtsTokenCount is added to output_tokens, and both inverse
  encoders restore the provider meaning; the settled cost follows;
- tool loops: a Gemini 3 thought signature survives an OpenAI-chat client's echo and a standard client gets the
  documented validator-skip signature; Anthropic thinking is not enabled over an unsigned tool turn; a tool call that
  takes no arguments streams '{}' to OpenAI chat and Responses clients;
- Gemini schema reduction inlines local $ref/$defs, maps const and drops non-string enums;
- Anthropic max_tokens injected when absent (noted), json_schema mapped to output_config.format (GA) and refused for a
  model without structured outputs, json_object refused; Gemini responseMimeType set with responseSchema;
- n>1, logprobs and logit_bias refused on translated paths;
- in-protocol error events and terminal frames;
- audio: openai.speech <-> elevenlabs.tts (voice maps, formats, speed range, WAV wrapping) and openai.transcriptions <->
  elevenlabs.stt (language codes, verbose_json segments from words, srt/vtt).
End-to-end translated (T) cells: every T cell of the translation matrix is driven through the target adapter against its
WP4 mock (decode inbound -> IR -> adapter -> IR -> encode inbound). Loopback only; synthetic values only.
"""
import json
import struct
import unittest

import broker_adapter_support as S  # installs the netguard first
from broker_mocks import OpenAIMock, load_fixture, netguard

from titanos.broker.config import Upstream
from titanos.broker.envelopes import stream_terminal
from titanos.broker.errors import BrokerError
from titanos.broker.profiles import PROFILES
from titanos.broker.sse import SSEFrame
from titanos.broker.translate import anthropic_messages as ant
from titanos.broker.translate import audio, gemini, ollama, openai_chat, responses
from titanos.broker.translate.common import aiter_list, synth_tool_id
from titanos.broker.translate.ir import (BlockStop, ErrorEvent, IRResponse, Start, Stop, Text, TextDelta, Thinking,
                                         ThinkingDelta, ToolArgsDelta, ToolCallStart, ToolUse, UsageEvent)
from decimal import Decimal

from titanos.broker.config import PriceRow
from titanos.broker.pricing import PriceTable
from titanos.broker.sse import anthropic_usage, gemini_usage
from titanos.broker.types import Usage

PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 8
PNG_URL = 'data:image/png;base64,iVBORw0KGgoAAAAAAAAAAA=='
WEATHER = {'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather by city',
                                            'parameters': {'type': 'object', 'properties': {'location': {'type': 'string'}},
                                                           'required': ['location'], 'additionalProperties': False}}}
CONVERSATION = {
    'model': 'alias', 'max_tokens': 300, 'temperature': 0.5,
    'messages': [
        {'role': 'system', 'content': 'Be brief.'},
        {'role': 'developer', 'content': 'Use metric units.'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Weather in Paris and Rome?'},
                                     {'type': 'image_url', 'image_url': {'url': PNG_URL}},
                                     {'type': 'image_url', 'image_url': {'url': 'https://img.example.invalid/a.png'}}]},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_paris01', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"location":"Paris"}'}},
            {'id': 'call_rome002', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"location":"Rome"}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_paris01', 'content': '18C'},
        {'role': 'tool', 'tool_call_id': 'call_rome002', 'content': '24C'},
        {'role': 'user', 'content': 'Thanks.'},
        {'role': 'user', 'content': 'Which is warmer?'}],
    'tools': [WEATHER], 'tool_choice': 'auto'}

TEXT_ONLY = dict(CONVERSATION, messages=[m for m in CONVERSATION['messages'] if not isinstance(m.get('content'), list)])


async def collect(aiter):
    return [item async for item in aiter]


def run(coro):
    return S.run(coro)


def events_of(frames):
    return [SSEFrame(None, json.dumps(f), b'') if isinstance(f, dict) else f for f in frames]


def fixture_frames(provider, case, variables=None):
    fx = load_fixture(provider, case)
    out = []
    for raw in fx.frames(variables or {'model': 'm'}):
        out.extend(f for _, f in S.SSESplitter().feed(raw) if f is not None) if fx.kind == 'sse' else \
            out.append(json.loads(raw))
    return out


class OpenAIAnthropic(unittest.TestCase):
    def test_request_openai_to_anthropic(self):
        notes = []
        ir = openai_chat.decode_request(CONVERSATION)
        body = ant.encode_request(ir, PROFILES['anthropic'], default_max_tokens=4096, notes=notes)
        self.assertEqual(body['system'], 'Be brief.\n\nUse metric units.')
        self.assertEqual(body['max_tokens'], 300)
        self.assertNotIn('injected:max_tokens', notes)
        roles = [m['role'] for m in body['messages']]
        self.assertEqual(roles, ['user', 'assistant', 'user'])  # alternation; tool results and both user turns merged
        first = body['messages'][0]['content']
        self.assertEqual(first[1], {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                                                'data': 'iVBORw0KGgoAAAAAAAAAAA=='}})
        self.assertEqual(first[2], {'type': 'image', 'source': {'type': 'url', 'url': 'https://img.example.invalid/a.png'}})
        calls = body['messages'][1]['content']
        self.assertEqual([(c['type'], c['id'], c['input']) for c in calls],
                         [('tool_use', 'call_paris01', {'location': 'Paris'}), ('tool_use', 'call_rome002', {'location': 'Rome'})])
        last = body['messages'][2]['content']
        self.assertEqual([b['type'] for b in last], ['tool_result', 'tool_result', 'text', 'text'])
        self.assertEqual([b['tool_use_id'] for b in last[:2]], ['call_paris01', 'call_rome002'])
        self.assertEqual(last[0]['content'], [{'type': 'text', 'text': '18C'}])
        self.assertEqual(body['tools'][0]['input_schema']['required'], ['location'])
        self.assertEqual(body['tool_choice'], {'type': 'auto'})
        self.assertEqual(body['temperature'], 0.5)

    def test_tool_choice_shapes(self):
        expected = {'auto': {'type': 'auto'}, 'required': {'type': 'any'}, 'none': {'type': 'none'},
                    'fn': {'type': 'tool', 'name': 'get_weather'}}
        gemini_expected = {'auto': {'mode': 'AUTO'}, 'required': {'mode': 'ANY'}, 'none': {'mode': 'NONE'},
                           'fn': {'mode': 'ANY', 'allowedFunctionNames': ['get_weather']}}
        for key, choice in (('auto', 'auto'), ('required', 'required'), ('none', 'none'),
                            ('fn', {'type': 'function', 'function': {'name': 'get_weather'}})):
            with self.subTest(choice=key):
                ir = openai_chat.decode_request(dict(TEXT_ONLY, tool_choice=choice))
                self.assertEqual(ant.encode_request(ir, None, default_max_tokens=10)['tool_choice'], expected[key])
                self.assertEqual(gemini.encode_request(ir)['toolConfig']['functionCallingConfig'], gemini_expected[key])
                back = ant.decode_request(dict(ant.encode_request(ir, None, default_max_tokens=10), model='m'))
                self.assertEqual(openai_chat.encode_request(back)['tool_choice'], choice)
                back = gemini.decode_request(gemini.encode_request(ir), model='models/m')
                self.assertEqual(openai_chat.encode_request(back)['tool_choice'], choice)
        ir = openai_chat.decode_request(dict(TEXT_ONLY, parallel_tool_calls=False))
        self.assertEqual(ant.encode_request(ir, None, default_max_tokens=10)['tool_choice'],
                         {'type': 'auto', 'disable_parallel_tool_use': True})
        back = ant.decode_request(dict(ant.encode_request(ir, None, default_max_tokens=10), model='m'))
        self.assertIs(openai_chat.encode_request(back)['parallel_tool_calls'], False)

    def test_max_tokens_injection_and_structured_outputs(self):
        body = {'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}]}
        notes = []
        out = ant.encode_request(openai_chat.decode_request(body), None, default_max_tokens=4096, notes=notes)
        self.assertEqual(out['max_tokens'], 4096)
        self.assertIn('injected:max_tokens', notes)
        cap = ant.encode_request(openai_chat.decode_request(dict(body, max_completion_tokens=99)), None,
                                 default_max_tokens=4096)
        self.assertEqual(cap['max_tokens'], 99)
        schema = {'type': 'json_schema', 'json_schema': {'name': 'w', 'schema': {'type': 'object'}}}
        ir = openai_chat.decode_request(dict(body, response_format=schema))
        # structured outputs are GA (output_config.format, no beta); a model without them refuses (never emulated)
        with self.assertRaises(BrokerError) as ctx:
            ant.encode_request(ir, None, default_max_tokens=10, model='claude-sonnet-4-20250514')
        self.assertEqual((ctx.exception.code, ctx.exception.status), ('broker_capability_unavailable', 400))
        mapped = ant.encode_request(ir, None, default_max_tokens=10)
        self.assertNotIn('output_format', mapped)
        self.assertEqual(mapped['output_config'], {'format': {'type': 'json_schema', 'schema': {'type': 'object'}}})
        self.assertEqual(ant.decode_request(dict(mapped, model='m')).response_format['json_schema']['schema'],
                         {'type': 'object'})
        with self.assertRaises(BrokerError):
            ant.encode_request(openai_chat.decode_request(dict(body, response_format={'type': 'json_object'})), None,
                               default_max_tokens=10, structured_outputs=True)

    def test_refused_fields_on_translated_paths(self):
        body = {'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}]}
        for extra in ({'n': 2}, {'logprobs': True}, {'top_logprobs': 3}, {'logit_bias': {'50256': -100}}):
            with self.subTest(extra=extra):
                with self.assertRaises(BrokerError) as ctx:
                    openai_chat.decode_request(dict(body, **extra))
                self.assertEqual(ctx.exception.code, 'broker_capability_unavailable')
        notes = []
        ir = openai_chat.decode_request(dict(body, user='u1', metadata={'a': 'b'}, worker='w', frobnicate=1), notes=notes)
        self.assertIn('dropped:frobnicate', notes)
        self.assertIn('dropped:user', notes)
        self.assertNotIn('dropped:worker', notes)
        self.assertNotIn('frobnicate', json.dumps(ant.encode_request(ir, None, default_max_tokens=5)))

    def test_thinking_and_reasoning_signature_round_trip(self):
        anthropic_body = load_fixture('anthropic', 'messages_thinking').body({'model': 'claude-x'})
        resp = ant.decode_response(anthropic_body)
        openai = openai_chat.encode_response(resp, model_name='alias')
        message = openai['choices'][0]['message']
        self.assertEqual(message['reasoning_content'], 'The user greets me; answer briefly.')
        self.assertEqual(message['reasoning_signature'], 'mock-signature-0001')
        self.assertEqual(message['content'], 'Hello! How can I help you today?')
        follow_up = {'model': 'a', 'messages': [{'role': 'user', 'content': 'hi'}, message,
                                                {'role': 'user', 'content': 'and?'}],
                     'reasoning_effort': 'low'}
        body = ant.encode_request(openai_chat.decode_request(follow_up), None, default_max_tokens=8000)
        self.assertEqual(body['messages'][1]['content'][0], {'type': 'thinking', 'thinking':
                                                             'The user greets me; answer briefly.',
                                                             'signature': 'mock-signature-0001'})
        self.assertEqual(body['thinking'], {'type': 'enabled', 'budget_tokens': 1024})
        # thinking turns sampling off (noted)
        notes = []
        body = ant.encode_request(openai_chat.decode_request(dict(follow_up, temperature=0.3)), None,
                                  default_max_tokens=8000, notes=notes)
        self.assertNotIn('temperature', body)
        self.assertIn('dropped:temperature (thinking)', notes)
        # an OpenAI-compatible target never receives reasoning_content back
        notes = []
        oai = openai_chat.encode_request(ant.decode_request(dict(body, model='m')), PROFILES['openai'], notes=notes)
        self.assertNotIn('reasoning_content', json.dumps(oai))
        self.assertIn('dropped:thinking', notes)

    def test_response_stop_reasons_and_usage(self):
        tools = ant.decode_response(load_fixture('anthropic', 'messages_tools').body({'model': 'm'}))
        doc = openai_chat.encode_response(tools, model_name='alias')
        self.assertEqual(doc['choices'][0]['finish_reason'], 'tool_calls')
        call = doc['choices'][0]['message']['tool_calls'][0]
        self.assertEqual((call['id'], call['function']['name'], json.loads(call['function']['arguments'])),
                         ('toolu_mock0001', 'get_weather', {'location': 'Paris'}))
        self.assertEqual(doc['usage']['prompt_tokens'], tools.usage.input_tokens)
        back = ant.encode_response(openai_chat.decode_response(doc), model_name='claude-x')
        self.assertEqual(back['stop_reason'], 'tool_use')
        self.assertEqual(back['content'][1], {'type': 'tool_use', 'id': 'toolu_mock0001', 'name': 'get_weather',
                                              'input': {'location': 'Paris'}})
        for finish, stop, anthropic_stop in (('stop', 'end_turn', 'end_turn'), ('length', 'max_tokens', 'max_tokens'),
                                             ('tool_calls', 'tool_use', 'tool_use'),
                                             ('content_filter', 'content_filter', 'refusal')):
            with self.subTest(finish=finish):
                resp = openai_chat.decode_response({'id': 'x', 'choices': [{'index': 0, 'message': {'content': 'a'},
                                                                            'finish_reason': finish}],
                                                    'usage': {'prompt_tokens': 3, 'completion_tokens': 4,
                                                              'prompt_tokens_details': {'cached_tokens': 2},
                                                              'completion_tokens_details': {'reasoning_tokens': 1}}})
                self.assertEqual(resp.stop, stop)
                self.assertEqual(resp.usage, Usage(input_tokens=3, output_tokens=4, cached_input_tokens=2,
                                                   reasoning_tokens=1))
                anth = ant.encode_response(resp, model_name='m')
                self.assertEqual(anth['stop_reason'], anthropic_stop)
                # Anthropic input_tokens excludes the cache reads (additive): 3 total - 2 cached
                self.assertEqual(anth['usage'], {'input_tokens': 1, 'output_tokens': 4, 'cache_read_input_tokens': 2})
        self.assertEqual(ant.decode_response(dict(load_fixture('anthropic', 'messages').body({'model': 'm'}),
                                                  stop_reason='stop_sequence')).stop, 'stop_sequence')

    def test_anthropic_request_to_openai(self):
        body = {'model': 'claude', 'max_tokens': 50, 'system': [{'type': 'text', 'text': 'Be brief.',
                                                                 'cache_control': {'type': 'ephemeral'}}],
                'tools': [{'name': 'get_weather', 'description': 'w', 'input_schema': {'type': 'object'}}],
                'tool_choice': {'type': 'any'}, 'stop_sequences': ['END'],
                'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Weather?'},
                                                          {'type': 'image', 'source': {'type': 'base64',
                                                                                       'media_type': 'image/png',
                                                                                       'data': 'iVBORw0KGgo='}}]},
                             {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'toolu_01', 'name': 'get_weather',
                                                                'input': {'location': 'Paris'}}]},
                             {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_01',
                                                           'content': '18C'}, {'type': 'text', 'text': 'Thanks'}]}]}
        notes = []
        out = openai_chat.encode_request(ant.decode_request(body, notes=notes), PROFILES['openai'], notes=notes)
        self.assertIn('dropped:cache_control', notes)
        self.assertEqual(out['messages'][0], {'role': 'system', 'content': 'Be brief.'})
        self.assertEqual(out['messages'][1]['content'][1], {'type': 'image_url',
                                                            'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}})
        self.assertEqual(out['messages'][2]['tool_calls'][0]['id'], 'toolu_01')
        self.assertEqual(out['messages'][3], {'role': 'tool', 'tool_call_id': 'toolu_01', 'content': '18C'})
        self.assertEqual(out['messages'][4], {'role': 'user', 'content': 'Thanks'})
        self.assertEqual((out['tool_choice'], out['stop'], out['max_tokens']), ('required', ['END'], 50))


class OpenAIGemini(unittest.TestCase):
    def test_request_openai_to_gemini(self):
        messages = [m for m in CONVERSATION['messages'] if m['role'] in ('system', 'developer')]
        messages.append({'role': 'user', 'content': 'Weather in Paris and Rome?'})
        messages.extend(m for m in CONVERSATION['messages'] if m['role'] in ('assistant', 'tool'))
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': 'Which is warmer?'},
                                                     {'type': 'image_url', 'image_url': {'url': PNG_URL}}]})
        body = dict(CONVERSATION, messages=messages, stop='END', seed=7,
                    response_format={'type': 'json_schema', 'json_schema': {'name': 'w', 'strict': True, 'schema': {
                        'type': 'object', 'properties': {'city': {'type': ['string', 'null']}}, 'required': ['city'],
                        'additionalProperties': False}}})
        notes = []
        out = gemini.encode_request(openai_chat.decode_request(body), notes=notes)
        self.assertEqual(out['systemInstruction'], {'parts': [{'text': 'Be brief.'}, {'text': 'Use metric units.'}]})
        self.assertEqual([c['role'] for c in out['contents']], ['user', 'model', 'user'])
        calls = out['contents'][1]['parts']
        self.assertEqual(calls, [{'functionCall': {'name': 'get_weather', 'args': {'location': 'Paris'}}},
                                 {'functionCall': {'name': 'get_weather', 'args': {'location': 'Rome'}}}])
        last = out['contents'][2]['parts']
        self.assertEqual(last[:2], [{'functionResponse': {'name': 'get_weather', 'response': {'output': '18C'}}},
                                    {'functionResponse': {'name': 'get_weather', 'response': {'output': '24C'}}}])
        self.assertEqual(last[2:], [{'text': 'Which is warmer?'},
                                    {'inlineData': {'mimeType': 'image/png', 'data': 'iVBORw0KGgoAAAAAAAAAAA=='}}])
        gen = out['generationConfig']
        self.assertEqual(gen['responseMimeType'], 'application/json')
        self.assertEqual(gen['responseSchema'], {'type': 'OBJECT', 'properties': {'city': {'type': 'STRING',
                                                                                           'nullable': True}},
                                                 'required': ['city']})
        self.assertIn('dropped:schema.additionalProperties', notes)
        self.assertEqual((gen['maxOutputTokens'], gen['temperature'], gen['stopSequences'], gen['seed']),
                         (300, 0.5, ['END'], 7))
        decl = out['tools'][0]['functionDeclarations'][0]
        self.assertEqual(decl['parameters']['type'], 'OBJECT')
        with self.assertRaises(BrokerError) as ctx:
            gemini.encode_request(openai_chat.decode_request(CONVERSATION))  # carries an https image url
        self.assertEqual((ctx.exception.code, ctx.exception.status), ('broker_capability_unavailable', 400))
        json_object = gemini.encode_request(openai_chat.decode_request(
            {'model': 'a', 'messages': [{'role': 'user', 'content': 'x'}], 'response_format': {'type': 'json_object'}}))
        self.assertEqual(json_object['generationConfig'], {'responseMimeType': 'application/json'})

    def test_gemini_response_ids_and_back(self):
        body = load_fixture('gemini', 'generate_content_function_call').body({'model': 'gemini-x'})
        resp = gemini.decode_response(body)
        call = resp.content[0]
        expected = synth_tool_id(0, 'get_weather', {'location': 'Paris'})
        self.assertEqual(call.id, expected)
        self.assertRegex(expected, r'^call_0_[0-9a-f]{8}$')
        self.assertEqual(gemini.decode_response(body).content[0].id, expected)  # deterministic
        doc = openai_chat.encode_response(resp, model_name='alias')
        self.assertEqual(doc['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(doc['usage'], {'prompt_tokens': 4, 'completion_tokens': 5, 'total_tokens': 9})
        follow = {'model': 'a', 'messages': [{'role': 'user', 'content': 'w?'}, doc['choices'][0]['message'],
                                             {'role': 'tool', 'tool_call_id': expected, 'content': '{"temp": 18}'}]}
        out = gemini.encode_request(openai_chat.decode_request(follow))
        self.assertEqual(out['contents'][2]['parts'][0], {'functionResponse': {'name': 'get_weather',
                                                                               'response': {'temp': 18}}})
        for reason, stop in (('STOP', 'end_turn'), ('MAX_TOKENS', 'max_tokens'), ('SAFETY', 'content_filter'),
                             ('RECITATION', 'content_filter')):
            text = {'candidates': [{'content': {'parts': [{'text': 'x'}], 'role': 'model'}, 'finishReason': reason}]}
            self.assertEqual(gemini.decode_response(text).stop, stop)
        encoded = gemini.encode_response(IRResponse(id='r', model='m', content=[Text('x')], stop='max_tokens',
                                                    usage=Usage(input_tokens=1, output_tokens=2)), model_name='alias')
        self.assertEqual(encoded['candidates'][0]['finishReason'], 'MAX_TOKENS')
        self.assertEqual(encoded['usageMetadata'], {'promptTokenCount': 1, 'candidatesTokenCount': 2,
                                                    'totalTokenCount': 3})

    def test_gemini_request_to_openai(self):
        body = {'systemInstruction': {'parts': [{'text': 'Be brief.'}]},
                'contents': [{'role': 'user', 'parts': [{'text': 'Weather?'}, {'inlineData': {'mimeType': 'image/png',
                                                                                              'data': 'iVBORw0KGgo='}}]},
                             {'role': 'model', 'parts': [{'functionCall': {'name': 'get_weather',
                                                                           'args': {'location': 'Paris'}}}]},
                             {'role': 'user', 'parts': [{'functionResponse': {'name': 'get_weather',
                                                                              'response': {'output': '18C'}}}]}],
                'tools': [{'functionDeclarations': [{'name': 'get_weather', 'parameters': {
                    'type': 'OBJECT', 'properties': {'location': {'type': 'STRING'}}}}]}],
                'toolConfig': {'functionCallingConfig': {'mode': 'ANY', 'allowedFunctionNames': ['get_weather']}},
                'generationConfig': {'maxOutputTokens': 64, 'topK': 3, 'responseMimeType': 'application/json',
                                     'responseSchema': {'type': 'OBJECT', 'properties': {'t': {'type': 'NUMBER'}}}}}
        ir = gemini.decode_request(body, model='models/gemini-2.5-flash')
        self.assertEqual(ir.model, 'gemini-2.5-flash')
        out = openai_chat.encode_request(ir, PROFILES['openai'])
        call_id = out['messages'][2]['tool_calls'][0]['id']
        self.assertRegex(call_id, r'^call_0_[0-9a-f]{8}$')
        self.assertEqual(out['messages'][3], {'role': 'tool', 'tool_call_id': call_id, 'content': '18C'})
        self.assertEqual(out['tool_choice'], {'type': 'function', 'function': {'name': 'get_weather'}})
        self.assertEqual(out['tools'][0]['function']['parameters'], {'type': 'object', 'properties': {
            'location': {'type': 'string'}}})
        self.assertEqual(out['response_format']['json_schema']['schema'], {'type': 'object', 'properties': {
            't': {'type': 'number'}}})
        self.assertEqual(out['max_tokens'], 64)
        with self.assertRaises(BrokerError):
            gemini.decode_request(dict(body, generationConfig={'candidateCount': 2}), model='models/m')


class OllamaCodec(unittest.TestCase):
    def test_openai_to_ollama_and_back(self):
        body = dict(CONVERSATION, messages=[m for m in CONVERSATION['messages'] if not (
            m['role'] == 'user' and isinstance(m['content'], list))], seed=3, stop=['X'])
        out = ollama.encode_request(openai_chat.decode_request(body))
        self.assertEqual(out['messages'][0], {'role': 'system', 'content': 'Be brief.\n\nUse metric units.'})
        self.assertEqual(out['messages'][1]['tool_calls'][0], {'function': {'name': 'get_weather',
                                                                            'arguments': {'location': 'Paris'}}})
        self.assertEqual(out['messages'][2], {'role': 'tool', 'content': '18C', 'tool_name': 'get_weather'})
        self.assertEqual(out['options'], {'temperature': 0.5, 'seed': 3, 'num_predict': 300, 'stop': ['X']})
        self.assertEqual(out['tools'][0]['function']['name'], 'get_weather')
        self.assertIs(out['stream'], False)
        resp = ollama.decode_response(load_fixture('ollama', 'chat_tools').body({'model': 'm'}))
        self.assertEqual(resp.stop, 'tool_use')
        self.assertEqual(resp.usage, Usage(input_tokens=26, output_tokens=12))
        doc = openai_chat.encode_response(resp, model_name='alias')
        self.assertEqual(doc['choices'][0]['message']['tool_calls'][0]['function']['arguments'], '{"location":"Paris"}')

    def test_ollama_to_openai_and_anthropic(self):
        body = {'model': 'llama3', 'keep_alive': '5m', 'format': 'json',
                'options': {'num_predict': 50, 'temperature': 0.1, 'top_k': 5, 'mirostat': 1},
                'messages': [{'role': 'system', 'content': 'sys'},
                             {'role': 'user', 'content': 'hi', 'images': ['iVBORw0KGgo=']},
                             {'role': 'assistant', 'content': '', 'tool_calls': [
                                 {'function': {'name': 'get_weather', 'arguments': {'location': 'Paris'}}}]},
                             {'role': 'tool', 'content': '18C', 'tool_name': 'get_weather'}]}
        notes = []
        ir = ollama.decode_request(body, generate=False, notes=notes)
        self.assertIn('dropped:keep_alive', notes)
        self.assertIn('dropped:options.mirostat', notes)
        out = openai_chat.encode_request(ir, PROFILES['openai'])
        self.assertEqual(out['max_tokens'], 50)
        self.assertEqual(out['response_format'], {'type': 'json_object'})
        call_id = out['messages'][2]['tool_calls'][0]['id']
        self.assertEqual(out['messages'][3]['tool_call_id'], call_id)
        self.assertEqual(out['messages'][1]['content'][1]['image_url']['url'], 'data:image/png;base64,iVBORw0KGgo=')
        with self.assertRaises(BrokerError):  # format json -> json_object, which Anthropic cannot honour
            ant.encode_request(ir, None, default_max_tokens=10)
        ir = ollama.decode_request(dict(body, format=None), generate=False)
        anth = ant.encode_request(ir, None, default_max_tokens=10)
        self.assertEqual(anth['messages'][2]['content'][0]['tool_use_id'], call_id)
        self.assertEqual(anth['top_k'], 5)
        generate = ollama.decode_request({'model': 'm', 'prompt': 'Why?', 'system': 'sys', 'stream': False},
                                         generate=True)
        self.assertEqual(openai_chat.encode_request(generate)['messages'], [{'role': 'system', 'content': 'sys'},
                                                                            {'role': 'user', 'content': 'Why?'}])
        encoded = ollama.encode_response(IRResponse(id='x', model='m', content=[Text('ok')], stop='max_tokens',
                                                    usage=Usage(input_tokens=2, output_tokens=3)),
                                         model_name='alias', generate=True)
        self.assertEqual((encoded['response'], encoded['done'], encoded['done_reason'], encoded['prompt_eval_count'],
                          encoded['eval_count']), ('ok', True, 'length', 2, 3))


class ResponsesCodec(unittest.TestCase):
    BODY = {'model': 'alias', 'instructions': 'Be brief.', 'max_output_tokens': 80,
            'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Weather?'},
                                                                      {'type': 'input_image', 'image_url': PNG_URL}]},
                      {'type': 'function_call', 'call_id': 'call_abc', 'name': 'get_weather',
                       'arguments': '{"location":"Paris"}'},
                      {'type': 'function_call_output', 'call_id': 'call_abc', 'output': '18C'}],
            'tools': [{'type': 'function', 'name': 'get_weather', 'parameters': {'type': 'object'}}],
            'tool_choice': {'type': 'function', 'name': 'get_weather'},
            'text': {'format': {'type': 'json_schema', 'name': 'w', 'schema': {'type': 'object'}, 'strict': True}}}

    def test_responses_to_chat_anthropic_and_gemini(self):
        ir = responses.decode_request(self.BODY)
        chat = openai_chat.encode_request(ir, PROFILES['openai'])
        self.assertEqual(chat['messages'][0], {'role': 'system', 'content': 'Be brief.'})
        self.assertEqual(chat['messages'][2]['tool_calls'][0]['id'], 'call_abc')
        self.assertEqual(chat['messages'][3], {'role': 'tool', 'tool_call_id': 'call_abc', 'content': '18C'})
        self.assertEqual(chat['tool_choice'], {'type': 'function', 'function': {'name': 'get_weather'}})
        self.assertEqual(chat['response_format']['type'], 'json_schema')
        self.assertEqual(chat['max_tokens'], 80)
        anth = ant.encode_request(ir, None, default_max_tokens=10, structured_outputs=True)
        self.assertEqual(anth['messages'][2]['content'][0]['type'], 'tool_result')
        self.assertEqual(anth['tool_choice'], {'type': 'tool', 'name': 'get_weather'})
        gem = gemini.encode_request(ir)
        self.assertEqual(gem['contents'][2]['parts'][0]['functionResponse']['name'], 'get_weather')
        self.assertEqual(gem['generationConfig']['responseSchema'], {'type': 'OBJECT'})
        with self.assertRaises(BrokerError):
            responses.decode_request(dict(self.BODY, previous_response_id='resp_1'))
        with self.assertRaises(BrokerError):
            responses.decode_request(dict(self.BODY, tools=[{'type': 'web_search'}]))
        notes = []
        responses.decode_request(dict(self.BODY, store=True), notes=notes)
        self.assertIn('not_stored:store', notes)

    def test_encode_response_object(self):
        resp = IRResponse(id='x', model='m', content=[Thinking('hmm', 'sig-1'), Text('Hi'),
                                                      ToolUse('call_1', 'get_weather', '{}')],
                          stop='tool_use', usage=Usage(input_tokens=5, output_tokens=6, reasoning_tokens=2))
        doc = responses.encode_response(resp, responses.response_object(model='alias'))
        self.assertEqual(doc['status'], 'completed')
        self.assertEqual([item['type'] for item in doc['output']], ['reasoning', 'message', 'function_call'])
        self.assertEqual(doc['output'][0]['encrypted_content'], 'sig-1')
        self.assertEqual(doc['output'][2]['call_id'], 'call_1')
        self.assertEqual(doc['usage']['output_tokens_details'], {'reasoning_tokens': 2})
        limited = responses.encode_response(IRResponse(id='x', model='m', content=[Text('a')], stop='max_tokens'),
                                            responses.response_object(model='alias'))
        self.assertEqual((limited['status'], limited['incomplete_details']), ('incomplete',
                                                                              {'reason': 'max_output_tokens'}))
        back = responses.decode_request({'model': 'a', 'input': [doc['output'][0], {'type': 'message', 'role': 'user',
                                                                                     'content': 'next'}]})
        self.assertEqual(back.messages[0].parts[0], Thinking('hmm', 'sig-1'))


class StreamGoldens(unittest.TestCase):
    """Fixture streams decoded and re-encoded (no network): event order, tool ids, usage, terminal frames."""

    def test_anthropic_stream_to_openai_chunks(self):
        frames = fixture_frames('anthropic', 'messages_stream_tools')
        events = run(collect(ant.decode_stream(aiter_list(frames))))
        self.assertIsInstance(events[0], Start)
        self.assertIn(ToolCallStart(1, 'toolu_mock0001', 'get_weather'), events)
        self.assertEqual([e for e in events if isinstance(e, Stop)], [Stop('tool_use')])
        self.assertEqual([e for e in events if isinstance(e, UsageEvent)][-1].usage.output_tokens, 40)
        data = b''.join(run(collect(openai_chat.encode_stream(aiter_list(events), model_name='alias',
                                                              include_usage=True))))
        chunks = [json.loads(p) for _, p in S.sse_events(data)[:-1]]
        self.assertEqual(S.sse_events(data)[-1][1], '[DONE]')
        self.assertEqual(chunks[0]['choices'][0]['delta'], {'role': 'assistant', 'content': ''})
        calls = [c['choices'][0]['delta']['tool_calls'][0] for c in chunks if c['choices'] and
                 'tool_calls' in c['choices'][0]['delta']]
        self.assertEqual(calls[0], {'index': 0, 'id': 'toolu_mock0001', 'type': 'function',
                                    'function': {'name': 'get_weather', 'arguments': ''}})
        self.assertEqual(json.loads(''.join(c['function']['arguments'] for c in calls[1:])), {'location': 'Paris'})
        self.assertEqual(chunks[-2]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(chunks[-1]['choices'], [])
        self.assertEqual(chunks[-1]['usage']['prompt_tokens'], 12)

    def test_thinking_stream_keeps_the_signature(self):
        events = run(collect(ant.decode_stream(aiter_list(fixture_frames('anthropic', 'messages_stream_thinking')))))
        self.assertIn(ThinkingDelta(0, '', 'mock-signature-0001'), events)
        data = b''.join(run(collect(openai_chat.encode_stream(aiter_list(events), model_name='a', include_usage=False))))
        deltas = [json.loads(p)['choices'][0]['delta'] for _, p in S.sse_events(data)[:-1]]
        self.assertIn({'reasoning_signature': 'mock-signature-0001'}, deltas)
        frames = [f for _, f in S.SSESplitter().feed(data) if f is not None]
        reencoded = b''.join(run(collect(ant.encode_stream(openai_chat.decode_stream(aiter_list(frames)),
                                                           model_name='claude'))))
        types = [(json.loads(p).get('delta') or {}).get('type') for e, p in S.sse_events(reencoded)
                 if e == 'content_block_delta']
        self.assertIn('signature_delta', types)
        self.assertIn('thinking_delta', types)

    def test_openai_stream_to_anthropic_events(self):
        frames = fixture_frames('openai', 'chat_stream_tools_usage')
        data = b''.join(run(collect(ant.encode_stream(openai_chat.decode_stream(aiter_list(frames)),
                                                      model_name='claude-x'))))
        events = S.sse_events(data)
        names = [e for e, _ in events]
        self.assertEqual(names[0], 'message_start')
        self.assertEqual(names[-2:], ['message_delta', 'message_stop'])
        starts = [json.loads(p)['content_block'] for e, p in events if e == 'content_block_start']
        self.assertEqual(starts, [{'type': 'tool_use', 'id': 'call_mock0001abc', 'name': 'get_weather', 'input': {}}])
        partial = ''.join(json.loads(p)['delta']['partial_json'] for e, p in events if e == 'content_block_delta')
        self.assertEqual(json.loads(partial), {'location': 'Paris'})
        delta = json.loads(events[-2][1])
        self.assertEqual(delta['delta']['stop_reason'], 'tool_use')
        self.assertEqual(delta['usage'], {'input_tokens': 19, 'output_tokens': 10, 'cache_read_input_tokens': 0})
        self.assertEqual(names.count('content_block_start'), names.count('content_block_stop'))

    def test_gemini_and_ollama_streams(self):
        frames = fixture_frames('gemini', 'stream_generate_content_function_call')
        events = run(collect(gemini.decode_stream(aiter_list(frames))))
        start = next(e for e in events if isinstance(e, ToolCallStart))
        self.assertRegex(start.id, r'^call_0_[0-9a-f]{8}$')
        self.assertEqual([type(e).__name__ for e in events[1:4]], ['ToolCallStart', 'ToolArgsDelta', 'BlockStop'])
        self.assertIn(Stop('tool_use'), events)
        lines = [json.loads(line) for line in load_fixture('ollama', 'chat_stream').frames({'model': 'm'})]
        events = run(collect(ollama.decode_stream(aiter_list(lines))))
        self.assertEqual(''.join(e.text for e in events if isinstance(e, TextDelta)), 'Hello! How can I help you today?')
        self.assertEqual([e for e in events if isinstance(e, UsageEvent)][0].usage, Usage(input_tokens=26,
                                                                                         output_tokens=12))
        ndjson = b''.join(run(collect(ollama.encode_ndjson_stream(aiter_list(events), model_name='alias',
                                                                  generate=False))))
        out = S.ndjson_lines(ndjson)
        self.assertEqual(out[-1]['done'], True)
        self.assertEqual((out[-1]['prompt_eval_count'], out[-1]['eval_count']), (26, 12))
        gem = b''.join(run(collect(gemini.encode_stream(aiter_list(events), model_name='alias'))))
        last = json.loads(S.sse_events(gem)[-1][1])
        self.assertEqual(last['candidates'][0]['finishReason'], 'STOP')
        self.assertEqual(last['usageMetadata']['promptTokenCount'], 26)

    def test_error_events_and_terminal_frames(self):
        events = [Start('x', 'm'), TextDelta(0, 'Hi'), ErrorEvent('upstream_unavailable', 'Overloaded')]
        openai = b''.join(run(collect(openai_chat.encode_stream(aiter_list(events), model_name='a', include_usage=False))))
        self.assertTrue(openai.endswith(stream_terminal('openai', 'upstream_unavailable', 'Overloaded')))
        self.assertEqual(json.loads(S.sse_events(openai)[-2][1])['error']['type'], 'stream_interrupted')
        anth = b''.join(run(collect(ant.encode_stream(aiter_list(events), model_name='a'))))
        self.assertEqual(S.sse_events(anth)[-1][0], 'error')
        self.assertEqual(json.loads(S.sse_events(anth)[-1][1])['error']['type'], 'overloaded_error')
        gem = b''.join(run(collect(gemini.encode_stream(aiter_list(events), model_name='a'))))
        self.assertEqual(json.loads(S.sse_events(gem)[-1][1])['error']['status'], 'UNAVAILABLE')
        olla = b''.join(run(collect(ollama.encode_ndjson_stream(aiter_list(events), model_name='a', generate=False))))
        self.assertEqual(S.ndjson_lines(olla)[-1], {'error': 'Overloaded'})
        resp = b''.join(run(collect(responses.encode_stream(aiter_list(events), responses.response_object(model='a')))))
        names = [e for e, _ in S.sse_events(resp)]
        self.assertEqual(names[-1], 'response.failed')
        sequence = [json.loads(p)['sequence_number'] for _, p in S.sse_events(resp)]
        self.assertEqual(sequence, list(range(len(sequence))))
        decoded = run(collect(ant.decode_stream(aiter_list(fixture_frames('anthropic', 'messages_stream_error_first')))))
        self.assertEqual(decoded, [ErrorEvent('upstream_unavailable', 'Overloaded')])
        decoded = run(collect(openai_chat.decode_stream(aiter_list([{'error': {'message': 'x', 'type': 'server_error'}}]))))
        self.assertEqual(decoded[-1].code, 'upstream_unavailable')

    def test_tool_arguments_invalid_json_end_the_stream(self):
        events = [Start('x', 'm'), ToolCallStart(0, 'c1', 'f'), ToolArgsDelta(0, '{"a":'), BlockStop(0), Stop('tool_use')]
        gem = b''.join(run(collect(gemini.encode_stream(aiter_list(events), model_name='a'))))
        self.assertIn('error', json.loads(S.sse_events(gem)[-1][1]))
        olla = b''.join(run(collect(ollama.encode_ndjson_stream(aiter_list(events), model_name='a', generate=False))))
        self.assertIn('error', S.ndjson_lines(olla)[-1])


class AudioMaps(unittest.TestCase):
    VOICES = {'openai_to_elevenlabs': {'alloy': '21m00Tcm4TlvDq8ikWAM'},
              'elevenlabs_to_openai': {'21m00Tcm4TlvDq8ikWAM': 'alloy'}}

    def test_speech_openai_to_elevenlabs(self):
        notes = []
        voice, body, query, wrap = audio.speech_openai_to_elevenlabs(
            {'model': 'eleven_multilingual_v2', 'input': 'Hello', 'voice': 'alloy', 'response_format': 'wav',
             'speed': 1.1, 'instructions': 'cheerful'}, self.VOICES['openai_to_elevenlabs'], notes=notes)
        self.assertEqual((voice, query, wrap), ('21m00Tcm4TlvDq8ikWAM', {'output_format': 'pcm_24000'}, True))
        self.assertEqual(body, {'text': 'Hello', 'model_id': 'eleven_multilingual_v2', 'voice_settings': {'speed': 1.1}})
        self.assertIn('dropped:instructions', notes)
        for fmt, target in (('mp3', 'mp3_44100_128'), ('opus', 'opus_48000_128'), ('pcm', 'pcm_24000')):
            _, _, query, wrap = audio.speech_openai_to_elevenlabs({'input': 'x', 'voice': 'alloy', 'response_format': fmt},
                                                                  self.VOICES['openai_to_elevenlabs'])
            self.assertEqual((query['output_format'], wrap), (target, False))
        for bad in ({'response_format': 'aac'}, {'response_format': 'flac'}, {'speed': 0.5}, {'speed': 1.5},
                    {'voice': 'nova'}):
            with self.subTest(bad=bad):
                with self.assertRaises(BrokerError) as ctx:
                    audio.speech_openai_to_elevenlabs(dict({'input': 'x', 'voice': 'alloy'}, **bad),
                                                      self.VOICES['openai_to_elevenlabs'])
                self.assertEqual(ctx.exception.status, 400)

    def test_speech_elevenlabs_to_openai(self):
        body = audio.speech_elevenlabs_to_openai({'text': 'Hi', 'model_id': 'tts-1', 'voice_settings': {'speed': 0.9}},
                                                 {'output_format': 'pcm_24000'}, self.VOICES['elevenlabs_to_openai'],
                                                 voice_id='21m00Tcm4TlvDq8ikWAM')
        self.assertEqual(body, {'model': 'tts-1', 'input': 'Hi', 'voice': 'alloy', 'response_format': 'pcm', 'speed': 0.9})
        for fmt, expected in (('mp3_22050_32', 'mp3'), ('opus_48000_64', 'opus')):
            self.assertEqual(audio.speech_elevenlabs_to_openai({'text': 'x'}, {'output_format': fmt},
                                                               self.VOICES['elevenlabs_to_openai'],
                                                               voice_id='21m00Tcm4TlvDq8ikWAM')['response_format'],
                             expected)
        with self.assertRaises(BrokerError):
            audio.speech_elevenlabs_to_openai({'text': 'x'}, {'output_format': 'pcm_16000'},
                                              self.VOICES['elevenlabs_to_openai'], voice_id='21m00Tcm4TlvDq8ikWAM')
        with self.assertRaises(BrokerError):
            audio.speech_elevenlabs_to_openai({'text': 'x'}, {}, self.VOICES['elevenlabs_to_openai'], voice_id='other')

    def test_pcm_to_wav(self):
        pcm = struct.pack('<4h', 0, 1000, -1000, 0)
        wav = audio.pcm_to_wav(pcm, 24000)
        self.assertEqual(wav[:4], b'RIFF')
        self.assertEqual(struct.unpack('<I', wav[4:8])[0], 36 + len(pcm))
        self.assertEqual(struct.unpack('<IHHIIHH', wav[16:36]), (16, 1, 1, 24000, 48000, 2, 16))
        self.assertEqual(struct.unpack('<I', wav[40:44])[0], len(pcm))
        self.assertEqual(wav[44:], pcm)
        import io
        import wave
        with wave.open(io.BytesIO(wav)) as reader:
            self.assertEqual((reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getnframes()),
                             (24000, 1, 2, 4))

    def test_transcription_maps(self):
        form = [('file', ('a.wav', b'RIFF', 'audio/wav')), ('model', 'scribe_v1'), ('language', 'en'),
                ('response_format', 'verbose_json'), ('prompt', 'names'), ('worker', 'gpu0')]
        notes = []
        out = audio.stt_openai_to_elevenlabs(form, notes=notes)
        self.assertEqual(out, [('model_id', 'scribe_v1'), ('file', ('a.wav', b'RIFF', 'audio/wav')),
                               ('language_code', 'eng'), ('timestamps_granularity', 'word')])
        self.assertIn('dropped:prompt', notes)
        answer = load_fixture('elevenlabs', 'stt').body({'model_id': 'scribe_v1'})
        verbose = audio.stt_elevenlabs_to_openai(answer, 'verbose_json')
        self.assertEqual((verbose['task'], verbose['language'], verbose['text'], verbose['duration']),
                         ('transcribe', 'english', 'Hello world.', 0.96))
        self.assertEqual(len(verbose['segments']), 1)
        self.assertEqual((verbose['segments'][0]['start'], verbose['segments'][0]['end'], verbose['segments'][0]['text']),
                         (0.0, 0.96, ' Hello world.'))
        self.assertEqual([w['word'] for w in verbose['words']], ['Hello', 'world.'])
        self.assertEqual(audio.stt_elevenlabs_to_openai(answer, 'json'), {'text': 'Hello world.'})
        self.assertEqual(audio.stt_elevenlabs_to_openai(answer, 'text'), 'Hello world.')
        self.assertEqual(audio.stt_elevenlabs_to_openai(answer, 'srt'), '1\n00:00:00,000 --> 00:00:00,960\nHello world.\n\n')
        self.assertTrue(audio.stt_elevenlabs_to_openai(answer, 'vtt').startswith('WEBVTT\n\n00:00:00.000 --> '))
        fields, fmt = audio.stt_elevenlabs_form_to_openai([('model_id', 'whisper-1'), ('language_code', 'eng'),
                                                           ('file', ('a.wav', b'RIFF', 'audio/wav'))])
        self.assertEqual(fmt, 'verbose_json')
        self.assertIn(('language', 'en'), fields)
        self.assertIn(('model', 'whisper-1'), fields)
        openai_answer = load_fixture('openai', 'transcriptions_verbose_json').body({'model': 'whisper-1'})
        back = audio.stt_openai_to_elevenlabs_body(openai_answer)
        self.assertEqual((back['language_code'], back['text']), ('eng', 'Hello world.'))


class TranslatedCells(unittest.TestCase):
    """Every T cell of the matrix, end to end through the target adapter and its WP4 mock."""

    @classmethod
    def setUpClass(cls):
        cls.targets = S.start_targets(('OAI', 'AZ', 'ANT', 'GEM', 'OLC', 'XI'))
        cls.chat_only = OpenAIMock()  # an OpenAI-compatible row without /completions or /responses (xai)
        cls.chat_only.start()
        upstream = Upstream(profile='xai', base_url=cls.chat_only.api_base, credential='cred')
        cls.targets['OAI-chat'] = S.MockTarget('OAI-chat', cls.chat_only, 'xai', 'grok-4', 'openai', upstream=upstream)

    @classmethod
    def tearDownClass(cls):
        S.stop_targets(cls.targets)

    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    def cell(self, inbound, column):
        for stream in (False, True):
            with self.subTest(inbound=inbound, target=column, stream=stream):
                out = S.run(S.translate(self.targets[column], inbound, stream=stream))
                text = S.stream_text(inbound, out) if stream else S.response_text(inbound, out)
                self.assertTrue(text.startswith('Hello'), text)

    def test_openai_chat_cells(self):
        for column in ('ANT', 'GEM', 'OLC'):
            self.cell('openai.chat', column)

    def test_openai_completions_cells(self):
        for column in ('OAI-chat', 'ANT', 'GEM', 'OLC'):
            self.cell('openai.completions', column)
        self.assertEqual(self.targets['OAI-chat'].mock.hits[-1].path, '/v1/chat/completions')

    def test_openai_responses_cells(self):
        for column in ('OAI-chat', 'AZ', 'ANT', 'GEM', 'OLC'):
            self.cell('openai.responses', column)

    def test_anthropic_messages_cells(self):
        for column in ('OAI', 'AZ', 'GEM', 'OLC'):
            self.cell('anthropic.messages', column)
        sent = self.targets['OAI'].mock.hits[-1].json()
        self.assertEqual(sent['messages'][0], {'role': 'system', 'content': 'Be brief.'})

    def test_gemini_generate_cells(self):
        for column in ('OAI', 'AZ', 'ANT', 'OLC'):
            self.cell('gemini.generateContent', column)
        self.assertEqual(self.targets['ANT'].mock.hits[-1].json()['max_tokens'], 64)

    def test_ollama_cells(self):
        for inbound in ('ollama.chat', 'ollama.generate'):
            for column in ('OAI', 'AZ', 'ANT', 'GEM'):
                self.cell(inbound, column)
        sent = self.targets['ANT'].mock.hits[-1].json()
        self.assertEqual(sent['max_tokens'], 1024)  # injected: Ollama sent no num_predict (default_max_tokens)

    def embed(self, column, ir, *, model=None):
        target = self.targets[column]

        async def go():
            async with target.egress() as egress:
                return await target.adapter.invoke(target.grant(model), egress, ir)
        return S.run(go())

    def test_embedding_cells(self):
        openai_ir = openai_chat.decode_embed_request({'model': 'a', 'input': ['one', 'two'], 'dimensions': 3})
        for column, model in (('GEM', 'gemini-embedding-001'), ('OLC', 'embeddinggemma')):
            with self.subTest(inbound='openai.embeddings', target=column):
                resp = self.embed(column, openai_ir, model=model)
                doc = openai_chat.encode_embed_response(resp.extras['vectors'], resp.usage, model_name='alias')
                self.assertEqual([d['index'] for d in doc['data']], [0, 1])
        sent = self.targets['GEM'].mock.hits[-1]
        self.assertTrue(sent.path.endswith(':batchEmbedContents'))
        self.assertEqual(sent.json()['requests'][0]['outputDimensionality'], 3)
        for batch in (False, True):
            gem_body = ({'requests': [{'model': 'models/a', 'content': {'parts': [{'text': 'x'}]}}] * 2} if batch else
                        {'content': {'parts': [{'text': 'x'}]}, 'outputDimensionality': 8})
            ir = gemini.decode_embed_request(gem_body, batch=batch, model='models/alias')
            for column, model in (('OAI', 'text-embedding-3-small'), ('AZ', 'text-embedding-3-small'),
                                  ('OLC', 'embeddinggemma')):
                with self.subTest(inbound='gemini.embed', batch=batch, target=column):
                    resp = self.embed(column, ir, model=model)
                    doc = gemini.encode_embed_response(resp.extras['vectors'], batch=batch)
                    self.assertTrue(doc['embeddings'] if batch else doc['embedding']['values'])
        ollama_ir = ollama.decode_embed_request({'model': 'a', 'input': ['x', 'y']})
        for column, model in (('OAI', 'text-embedding-3-small'), ('AZ', 'text-embedding-3-small'),
                              ('GEM', 'gemini-embedding-001')):
            with self.subTest(inbound='ollama.embed', target=column):
                resp = self.embed(column, ollama_ir, model=model)
                doc = ollama.encode_embed_response(resp.extras['vectors'], resp.usage, model_name='alias')
                self.assertEqual(len(doc['embeddings']), 2)

    def test_legacy_ollama_embedding_cells(self):
        """The legacy Ollama /api/embeddings shape ({prompt} in, {embedding} out) on every embedding target: translated
        to the OpenAI-compatible, Azure and Gemini adapters, and served by Ollama Cloud's native /api/embed route."""
        legacy = ollama.decode_embed_request({'model': 'a', 'prompt': 'one'})
        self.assertTrue(legacy.extras_by_protocol['legacy'])
        for column, model in (('OAI', 'text-embedding-3-small'), ('AZ', 'text-embedding-3-small'),
                              ('GEM', 'gemini-embedding-001'), ('OLC', 'embeddinggemma')):
            with self.subTest(target=column):
                resp = self.embed(column, legacy, model=model)
                doc = ollama.encode_embed_response(resp.extras['vectors'], resp.usage, model_name='alias', legacy=True)
                self.assertEqual(list(doc), ['embedding'])
                self.assertTrue(doc['embedding'] and all(isinstance(v, float) for v in doc['embedding']))
                hit = self.targets[column].mock.hits[-1]
                sent = hit.json()
                if column == 'OLC':
                    self.assertEqual((hit.path, sent['input']), ('/api/embed', ['one']))
                elif column == 'GEM':
                    self.assertTrue(hit.path.endswith((':embedContent', ':batchEmbedContents')), hit.path)
                else:
                    self.assertTrue(hit.path.endswith('/embeddings'))
                    self.assertEqual(sent['input'], ['one'])

    def test_audio_cells(self):
        xi, voices = self.targets['XI'], AudioMaps.VOICES
        voice, body, query, wrap = audio.speech_openai_to_elevenlabs(
            {'model': 'alias', 'input': 'Hello', 'voice': 'alloy', 'response_format': 'wav'},
            voices['openai_to_elevenlabs'])

        async def speech():
            async with xi.egress() as egress:
                result = await xi.adapter.tts(xi.grant(), egress, voice, body, query, stream=True)
                chunks = [c async for c in (audio.wrap_wav_stream(result.body) if wrap else result.body)]
                await result.aclose()
                return b''.join(chunks)
        data = S.run(speech())
        self.assertEqual(data[:4], b'RIFF')
        self.assertEqual(struct.unpack('<I', data[40:44])[0], 0xFFFFFFFF)
        self.assertEqual(xi.mock.hits[-1].query_params['output_format'], 'pcm_24000')

        async def stt():
            async with xi.egress() as egress:
                return await xi.adapter.stt(xi.grant('scribe_v1'), egress, audio.stt_openai_to_elevenlabs(
                    [('file', ('a.wav', b'RIFF', 'audio/wav')), ('model', 'alias'), ('language', 'en')]))
        answer = audio.stt_elevenlabs_to_openai(S.run(stt()), 'verbose_json')
        self.assertEqual(answer['text'], 'Hello world.')
        self.assertEqual(xi.mock.hits[-1].surface, 'elevenlabs.stt')
        for column in ('OAI', 'AZ'):
            target = self.targets[column]
            openai_body = audio.speech_elevenlabs_to_openai({'text': 'Hello', 'model_id': 'alias'},
                                                            {'output_format': 'mp3_44100_128'},
                                                            voices['elevenlabs_to_openai'], voice_id=voice)
            for stream in (False, True):
                with self.subTest(inbound='elevenlabs.tts', target=column, stream=stream):
                    async def tts():
                        async with target.egress() as egress:
                            result = await target.adapter.speech(target.grant('tts-1'), egress, openai_body,
                                                                 stream=stream)
                            chunks = [c async for c in result.body]
                            await result.aclose()
                            return b''.join(chunks), await result.usage
                    data, usage = S.run(tts())
                    self.assertTrue(data)
                    self.assertEqual(usage.characters, 5)
            with self.subTest(inbound='elevenlabs.stt', target=column):
                fields, fmt = audio.stt_elevenlabs_form_to_openai([('model_id', 'alias'),
                                                                   ('file', ('a.wav', b'RIFF', 'audio/wav'))])

                async def transcribe():
                    async with target.egress() as egress:
                        return await target.adapter.transcribe(target.grant('whisper-1'), egress, fields)
                body_out = audio.stt_openai_to_elevenlabs_body(S.run(transcribe()))
                self.assertEqual(body_out['text'], 'Hello world.')


async def _collect(aiter):
    return [item async for item in aiter]


def _openai_args(data: bytes) -> dict:
    """Accumulated tool-call arguments per slot from OpenAI chat chunks."""
    args: dict = {}
    for _, payload in S.sse_events(data):
        if payload == '[DONE]':
            continue
        for choice in json.loads(payload).get('choices') or []:
            for call in (choice.get('delta') or {}).get('tool_calls') or []:
                args[call['index']] = args.get(call['index'], '') + ((call.get('function') or {}).get('arguments') or '')
    return args


class UsageMeanings(unittest.TestCase):
    PRICES = PriceTable([PriceRow(match='m', input_per_mtok='3', output_per_mtok='15', cached_input_per_mtok='0.3')])

    def test_anthropic_cache_fields_both_directions(self):
        usage = anthropic_usage({'input_tokens': 2000, 'cache_read_input_tokens': 1500,
                                 'cache_creation_input_tokens': 10000, 'output_tokens': 50})
        self.assertEqual(usage, Usage(input_tokens=13500, output_tokens=50, cached_input_tokens=1500,
                                      cache_write_tokens=10000))
        # 12000 fresh (uncached + cache writes, no cache-write price on the row) at 3, 1500 cached at 0.3, 50 output at 15
        expected = (Decimal(12000) * 3 + Decimal(1500) * Decimal('0.3') + Decimal(50) * 15) / Decimal(1_000_000)
        self.assertEqual(self.PRICES.cost('m', usage), expected)
        # R18: a cache-write price bills the 10000 written tokens at their own rate (2000 fresh at 3)
        priced = PriceTable([PriceRow(match='m', input_per_mtok='3', output_per_mtok='15', cached_input_per_mtok='0.3',
                                      cache_write_per_mtok='3.75')])
        expected = (Decimal(2000) * 3 + Decimal(10000) * Decimal('3.75') + Decimal(1500) * Decimal('0.3')
                    + Decimal(50) * 15) / Decimal(1_000_000)
        self.assertEqual(priced.cost('m', usage), expected)
        self.assertEqual(ant._usage_out(usage), {'input_tokens': 2000, 'output_tokens': 50,
                                                 'cache_read_input_tokens': 1500, 'cache_creation_input_tokens': 10000})
        view = openai_chat.usage_dict(usage)
        self.assertGreaterEqual(view['prompt_tokens'], view['prompt_tokens_details']['cached_tokens'])
        # decode_response and the inverse encoder
        resp = ant.decode_response({'type': 'message', 'id': 'msg_1', 'model': 'm', 'content': [],
                                    'stop_reason': 'end_turn',
                                    'usage': {'input_tokens': 20, 'cache_read_input_tokens': 5000, 'output_tokens': 7}})
        self.assertEqual((resp.usage.input_tokens, resp.usage.cached_input_tokens), (5020, 5000))
        self.assertEqual(ant.encode_response(resp, model_name='m')['usage'],
                         {'input_tokens': 20, 'output_tokens': 7, 'cache_read_input_tokens': 5000})
        # streams: message_start carries the input fields, message_delta only the (cumulative) output
        start = anthropic_usage({'input_tokens': 20, 'cache_read_input_tokens': 5000, 'cache_creation_input_tokens': 30,
                                 'output_tokens': 1})
        final = anthropic_usage({'output_tokens': 9}, start)
        self.assertEqual(final, Usage(input_tokens=5050, output_tokens=9, cached_input_tokens=5000, cache_write_tokens=30))
        # a delta that repeats only input_tokens never lowers the input already seen
        self.assertEqual(anthropic_usage({'input_tokens': 20, 'output_tokens': 9}, start).input_tokens, 5050)

    def test_anthropic_stream_usage_reaches_openai_clients(self):
        frames = [
            {'type': 'message_start', 'message': {'id': 'msg_1', 'model': 'm', 'usage': {
                'input_tokens': 20, 'cache_read_input_tokens': 5000, 'cache_creation_input_tokens': 0,
                'output_tokens': 1}}},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'hi'}},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 4}},
            {'type': 'message_stop'}]
        out = b''.join(S.run(_collect(openai_chat.encode_stream(ant.decode_stream(aiter_list(frames)),
                                                                model_name='m', include_usage=True))))
        usage = [json.loads(p)['usage'] for _, p in S.sse_events(out) if p != '[DONE]' and '"usage"' in p][-1]
        self.assertEqual((usage['prompt_tokens'], usage['prompt_tokens_details']['cached_tokens'],
                          usage['completion_tokens']), (5020, 5000, 4))

    def test_gemini_thoughts_are_output(self):
        usage = gemini_usage({'promptTokenCount': 10, 'candidatesTokenCount': 100, 'thoughtsTokenCount': 5000,
                              'cachedContentTokenCount': 4, 'totalTokenCount': 5110})
        self.assertEqual(usage, Usage(input_tokens=10, output_tokens=5100, cached_input_tokens=4,
                                      reasoning_tokens=5000))
        view = openai_chat.usage_dict(usage)
        self.assertGreaterEqual(view['completion_tokens'], view['completion_tokens_details']['reasoning_tokens'])
        self.assertEqual(gemini_usage({'promptTokenCount': 3, 'toolUsePromptTokenCount': 2,
                                       'candidatesTokenCount': 1}).input_tokens, 5)
        # the inverse: an OpenAI reasoning answer (output includes reasoning) to a Gemini client, never double-counted
        resp = openai_chat.decode_response({'id': 'x', 'choices': [{'index': 0, 'message': {'content': 'a'},
                                                                    'finish_reason': 'stop'}],
                                            'usage': {'prompt_tokens': 10, 'completion_tokens': 905,
                                                      'completion_tokens_details': {'reasoning_tokens': 900}}})
        meta = gemini.encode_response(resp, model_name='m')['usageMetadata']
        self.assertEqual(meta, {'promptTokenCount': 10, 'candidatesTokenCount': 5, 'thoughtsTokenCount': 900,
                                'totalTokenCount': 915})
        self.assertEqual(gemini_usage(meta), resp.usage)
        resp = gemini.decode_response({'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'a'}]},
                                                       'finishReason': 'STOP'}],
                                       'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 5,
                                                         'thoughtsTokenCount': 900}})
        self.assertEqual(resp.usage.output_tokens, 905)

    def test_gemini_embedding_usage_is_estimated(self):
        decoded = gemini.embed_response({'embedding': {'values': [0.1, 0.2]}}, batch=False, inputs=['abcdefgh', 'x'])
        self.assertEqual(decoded['usage'], Usage(input_tokens=3, estimated=True))
        self.assertEqual(gemini.embed_texts({'requests': [{'content': {'parts': [{'text': 'ab'}, {'text': 'c'}]}}]}),
                         ['ab', 'c'])


class ToolLoops(unittest.TestCase):
    TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {
        'type': 'object', 'properties': {'city': {'type': 'string'}}}}}]

    def gemini_turn(self, signature='SIG123'):
        part = {'functionCall': {'name': 'get_weather', 'args': {'city': 'Paris'}}}
        if signature:
            part['thoughtSignature'] = signature
        return gemini.decode_response({'candidates': [{'content': {'role': 'model', 'parts': [part]},
                                                       'finishReason': 'STOP'}],
                                       'usageMetadata': {'promptTokenCount': 1, 'candidatesTokenCount': 1}})

    def follow_up(self, message):
        return {'model': 'g', 'tools': self.TOOLS, 'messages': [
            {'role': 'user', 'content': 'weather?'}, message,
            {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'], 'content': '{"t":20}'}]}

    def test_gemini3_signature_survives_an_openai_chat_tool_loop(self):
        message = openai_chat.encode_response(self.gemini_turn(), model_name='gemini-3-pro')['choices'][0]['message']
        self.assertEqual((message['reasoning_content'], message['reasoning_signature']), ('', 'SIG123'))
        echoed = {k: v for k, v in message.items() if k != 'refusal'}  # a client that echoes the extension fields
        notes = []
        body = gemini.encode_request(openai_chat.decode_request(self.follow_up(echoed)), notes=notes,
                                     model='gemini-3-pro-preview')
        self.assertEqual(body['contents'][1]['parts'][0]['thoughtSignature'], 'SIG123')
        self.assertNotIn('injected:thoughtSignature', notes)

    def test_standard_client_gets_the_documented_skip_signature_on_gemini3(self):
        message = openai_chat.encode_response(self.gemini_turn(), model_name='gemini-3-pro')['choices'][0]['message']
        standard = {'role': 'assistant', 'content': None, 'tool_calls': message['tool_calls']}  # no extension fields
        for model, expected in (('models/gemini-3-pro-preview', gemini.DUMMY_THOUGHT_SIGNATURE),
                                ('gemini-2.5-flash', None)):
            with self.subTest(model=model):
                notes = []
                body = gemini.encode_request(openai_chat.decode_request(self.follow_up(standard)), notes=notes,
                                             model=model)
                part = body['contents'][1]['parts'][0]
                self.assertEqual(part.get('thoughtSignature'), expected)
                self.assertEqual('injected:thoughtSignature' in notes, expected is not None)

    def test_anthropic_thinking_is_not_enabled_over_an_unsigned_tool_turn(self):
        body = {'model': 'c', 'reasoning_effort': 'high', 'max_tokens': 20000, 'tools': self.TOOLS, 'messages': [
            {'role': 'user', 'content': 'weather?'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'toolu_1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'sunny'}]}
        notes = []
        out = ant.encode_request(openai_chat.decode_request(body, notes=notes), PROFILES['anthropic'],
                                 default_max_tokens=4096, notes=notes)
        self.assertNotIn('thinking', out)
        self.assertIn('dropped:thinking (unsigned tool turn)', notes)
        # a first turn (no tool loop yet) keeps thinking
        first = dict(body, messages=body['messages'][:1])
        self.assertEqual(ant.encode_request(openai_chat.decode_request(first), PROFILES['anthropic'],
                                            default_max_tokens=4096)['thinking']['type'], 'enabled')
        # an Anthropic client's signed thinking keeps it enabled through the loop
        signed = {'model': 'c', 'max_tokens': 20000, 'thinking': {'type': 'enabled', 'budget_tokens': 2048},
                  'tools': [{'name': 'get_weather', 'input_schema': {'type': 'object', 'properties': {}}}],
                  'messages': [{'role': 'user', 'content': 'weather?'},
                               {'role': 'assistant', 'content': [
                                   {'type': 'thinking', 'thinking': 'Use the tool.', 'signature': 'sig-1'},
                                   {'type': 'tool_use', 'id': 'toolu_1', 'name': 'get_weather', 'input': {}}]},
                               {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1',
                                                             'content': 'sunny'}]}]}
        out = ant.encode_request(ant.decode_request(signed), PROFILES['anthropic'], default_max_tokens=4096)
        self.assertEqual(out['thinking'], {'type': 'enabled', 'budget_tokens': 2048})
        self.assertEqual(out['messages'][1]['content'][0]['type'], 'thinking')

    def test_no_argument_tool_call_streams_an_empty_object(self):
        frames = [
            {'type': 'message_start', 'message': {'id': 'msg_1', 'model': 'c', 'usage': {'input_tokens': 5,
                                                                                           'output_tokens': 1}}},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'tool_use', 'id': 'toolu_9',
                                                                          'name': 'now', 'input': {}}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'input_json_delta', 'partial_json': ''}},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}, 'usage': {'output_tokens': 3}},
            {'type': 'message_stop'}]
        out = b''.join(S.run(_collect(openai_chat.encode_stream(ant.decode_stream(aiter_list(frames)),
                                                                model_name='c', include_usage=False))))
        self.assertEqual(_openai_args(out), {0: '{}'})
        self.assertEqual(json.loads(_openai_args(out)[0]), {})
        events = S.run(_collect(responses.encode_stream(ant.decode_stream(aiter_list(frames)),
                                                        responses.response_object(model='c'))))
        done = [json.loads(p) for e, p in S.sse_events(b''.join(events))
                if e == 'response.function_call_arguments.done']
        self.assertEqual(done[0]['arguments'], '{}')
        # a call that does stream its arguments is unchanged
        with_args = frames[:2] + [{'type': 'content_block_delta', 'index': 0,
                                   'delta': {'type': 'input_json_delta', 'partial_json': '{"a":1}'}}] + frames[3:]
        out = b''.join(S.run(_collect(openai_chat.encode_stream(ant.decode_stream(aiter_list(with_args)),
                                                                model_name='c', include_usage=False))))
        self.assertEqual(_openai_args(out), {0: '{"a":1}'})


class GeminiSchema(unittest.TestCase):
    def test_refs_const_and_enums(self):
        schema = {'type': 'object', '$defs': {
            'Unit': {'type': 'string', 'enum': ['c', 'f']},
            'Node': {'type': 'object', 'properties': {'child': {'$ref': '#/$defs/Node'}}}},
            'properties': {'unit': {'$ref': '#/$defs/Unit', 'description': 'unit'},
                           'kind': {'const': 'weather'},
                           'level': {'type': 'integer', 'enum': [1, 2, 3]},
                           'tree': {'$ref': '#/$defs/Node'}},
            'required': ['unit']}
        notes = []
        out = gemini.to_gemini_schema(schema, notes=notes)
        self.assertNotIn('$defs', out)
        self.assertEqual(out['properties']['unit'], {'type': 'STRING', 'enum': ['c', 'f'], 'description': 'unit'})
        self.assertEqual(out['properties']['kind'], {'type': 'STRING', 'enum': ['weather']})
        self.assertEqual(out['properties']['level'], {'type': 'INTEGER'})
        self.assertEqual(out['properties']['tree']['type'], 'OBJECT')
        self.assertEqual(out['properties']['tree']['properties']['child'], {'type': 'OBJECT'})  # recursion cut here
        self.assertIn('dropped:schema.$ref', notes)
        self.assertIn('dropped:schema.enum (non-string values)', notes)
        self.assertNotIn('$ref', json.dumps(out))
        tools = gemini.encode_request(openai_chat.decode_request({
            'model': 'g', 'messages': [{'role': 'user', 'content': 'x'}],
            'tools': [{'type': 'function', 'function': {'name': 'f', 'parameters': schema}}]}))['tools']
        self.assertNotIn('$ref', json.dumps(tools))


CELLS = {
    **{f'openai.chat|{c}|T': 'TranslatedCells.test_openai_chat_cells' for c in ('ANT', 'GEM', 'OLC')},
    **{f'openai.completions|{c}|T': 'TranslatedCells.test_openai_completions_cells'
       for c in ('OAI', 'ANT', 'GEM', 'OLC')},
    **{f'openai.responses|{c}|T': 'TranslatedCells.test_openai_responses_cells' for c in ('OAI', 'AZ', 'ANT', 'GEM', 'OLC')},
    **{f'anthropic.messages|{c}|T': 'TranslatedCells.test_anthropic_messages_cells' for c in ('OAI', 'AZ', 'GEM', 'OLC')},
    **{f'gemini.generateContent|{c}|T': 'TranslatedCells.test_gemini_generate_cells' for c in ('OAI', 'AZ', 'ANT', 'OLC')},
    **{f'gemini.streamGenerateContent|{c}|T': 'TranslatedCells.test_gemini_generate_cells'
       for c in ('OAI', 'AZ', 'ANT', 'OLC')},
    **{f'ollama.{s}|{c}|T': 'TranslatedCells.test_ollama_cells' for s in ('chat', 'generate')
       for c in ('OAI', 'AZ', 'ANT', 'GEM')},
    **{f'openai.embeddings|{c}|T': 'TranslatedCells.test_embedding_cells' for c in ('GEM', 'OLC')},
    **{f'gemini.{s}|{c}|T': 'TranslatedCells.test_embedding_cells' for s in ('embedContent', 'batchEmbedContents')
       for c in ('OAI', 'AZ', 'OLC')},
    **{f'ollama.embed|{c}|T': 'TranslatedCells.test_embedding_cells' for c in ('OAI', 'AZ', 'GEM')},
    **{f'ollama.embeddings|{c}|T': 'TranslatedCells.test_legacy_ollama_embedding_cells' for c in ('OAI', 'AZ', 'GEM')},
    'ollama.embeddings|OLC|P': 'TranslatedCells.test_legacy_ollama_embedding_cells',
    'openai.audio.speech|XI|T': 'TranslatedCells.test_audio_cells',
    'openai.audio.transcriptions|XI|T': 'TranslatedCells.test_audio_cells',
    **{f'elevenlabs.{s}|{c}|T': 'TranslatedCells.test_audio_cells' for s in ('tts', 'tts_stream', 'stt')
       for c in ('OAI', 'AZ')},
    'golden:openai.chat<->anthropic': 'OpenAIAnthropic.test_request_openai_to_anthropic',
    'golden:anthropic->openai': 'OpenAIAnthropic.test_anthropic_request_to_openai',
    'golden:openai.chat<->gemini': 'OpenAIGemini.test_request_openai_to_gemini',
    'golden:gemini->openai': 'OpenAIGemini.test_gemini_request_to_openai',
    'golden:openai.chat<->ollama': 'OllamaCodec.test_openai_to_ollama_and_back',
    'golden:ollama->openai,anthropic': 'OllamaCodec.test_ollama_to_openai_and_anthropic',
    'golden:responses->chat,anthropic,gemini': 'ResponsesCodec.test_responses_to_chat_anthropic_and_gemini',
    'golden:speech<->tts': 'AudioMaps.test_speech_openai_to_elevenlabs',
    'golden:transcriptions<->stt': 'AudioMaps.test_transcription_maps',
}


if __name__ == '__main__':
    S.main(CELLS)
