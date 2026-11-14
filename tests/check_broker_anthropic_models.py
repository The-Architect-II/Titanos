"""Anthropic per-model request rules (translate/anthropic_models.py) and the Anthropic codec fixes of the completeness round.

Pure checks (no server): every body Titanos encodes for an Anthropic upstream is shaped for the upstream model and is
accepted by the WP4 mock's independent transcription of the provider's model rules (broker_mocks.anthropic_mock.
model_rule_error), for every model family:
- reasoning_effort becomes {type: enabled, budget_tokens} on 4.5 and older, {type: adaptive} (display summarized on 4.7+)
  plus output_config.effort on 4.6 and later, and effort alone where thinking is always on; minimal -> low, xhigh -> high
  on 4.6, 'none' turns thinking off where it can be (disabled / between_tools) and is noted where it cannot;
- temperature, top_p and top_k are dropped (noted) on models that refuse them, top_p alone is dropped when a 4.5 model
  would get both, and they stay on 4.6 and older without thinking;
- tool_choice required / a named function degrades to auto (noted) on Opus 5.5, Sonnet 5.5 and Fable 5.1, and stays
  forced elsewhere (thinking is then not enabled where forced tool use excludes it);
- json_schema maps to output_config.format (GA, no beta) and is refused for a model without structured outputs;
- the injected max_tokens is at least 16000 where thinking is always on;
- a signed thinking block with empty text (the omitted display) is replayed, a signature-only Gemini dummy is not;
- decode (Anthropic client to another target): output_config.format, output_config.effort and adaptive thinking reach
  the IR; plain-text document blocks become text (noted); other output_config fields are noted as dropped;
- R19 on the native passthrough (dispatch.inject_reasoning with the upstream model): the budget form on 4.5, adaptive
  thinking plus effort on 4.7, effort only on Opus 5.5; never over a client's own thinking or effort;
- the reservation estimate of an Anthropic body: base64 images at a fixed allowance, documents at their decoded size,
  never above the largest context window;
- usage: cache_creation.ephemeral_1h_input_tokens, server_tool_use.web_search_requests and speed are parsed (also across
  a stream's message_start and message_delta) and priced (1-hour cache writes, per-search fee, fast-mode multiplier;
  each missing price leaves the answer unpriced);
- the catalog metadata of an Anthropic model object (context_length, max_output_tokens, vision);
- provider-side state in a Messages body (container, custom skills, file ids) is recognised for the store gate.
"""
import base64
import json
import sys
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from broker_mocks import netguard  # noqa: E402
from broker_mocks.anthropic_mock import model_rule_error  # noqa: E402
from broker_mocks.fixtures import load  # noqa: E402

netguard.install()

from titanos.broker import anthropic_state  # noqa: E402
from titanos.broker.adapters.anthropic import model_meta  # noqa: E402
from titanos.broker.dispatch import inject_reasoning  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.pricing import PriceTable  # noqa: E402
from titanos.broker.profiles import PROFILES  # noqa: E402
from titanos.broker.sse import anthropic_usage  # noqa: E402
from titanos.broker.translate import anthropic_messages as ant  # noqa: E402
from titanos.broker.translate import anthropic_models as rules  # noqa: E402
from titanos.broker.translate import openai_chat  # noqa: E402
from titanos.broker.translate.ir import IRReasoning, Thinking  # noqa: E402

FAMILIES = {
    'claude-fable-5-1': 'always', 'claude-mythos-5-1': 'always', 'claude-fable-5': 'always',
    'claude-opus-5-5': 'always', 'claude-sonnet-5-5': 'always', 'claude-opus-5': 'default_on',
    'claude-sonnet-5': 'default_on', 'claude-opus-4-8': 'adaptive', 'claude-opus-4-7': 'adaptive',
    'claude-opus-4-6': 'adaptive', 'claude-sonnet-4-6': 'adaptive', 'claude-opus-4-5-20251101': 'budget',
    'claude-sonnet-4-5-20250929': 'budget', 'claude-haiku-4-5-20251001': 'budget', 'claude-opus-4-1-20250805': 'budget',
    'claude-sonnet-4-20250514': 'budget', 'claude-3-haiku-20240307': 'none', 'claude-opus-9-9': 'always',
}
WEATHER = {'type': 'function', 'function': {'name': 'get_weather', 'parameters': {
    'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}


def encode(body, model, notes=None, default_max_tokens=4096):
    notes = [] if notes is None else notes
    ir = openai_chat.decode_request(dict({'model': 'alias', 'messages': [{'role': 'user', 'content': 'hi'}]}, **body),
                                    notes=notes)
    out = ant.encode_request(ir, PROFILES['anthropic'], default_max_tokens=default_max_tokens, notes=notes, model=model)
    out['model'] = model
    return out


class ModelTable(unittest.TestCase):
    def test_every_family_resolves(self):
        for model, thinking in FAMILIES.items():
            with self.subTest(model=model):
                self.assertEqual(rules.rules_for(model).thinking, thinking)
        self.assertEqual(rules.rules_for('anthropic/claude-opus-5-5').family, 'opus-5.5')
        self.assertIs(rules.rules_for(None), rules.DEFAULT_RULES)
        self.assertFalse(rules.rules_for('claude-sonnet-5-5').forced_tool_choice)
        self.assertTrue(rules.rules_for('claude-fable-5').forced_tool_choice)

    def test_effort_mapping(self):
        r46 = rules.rules_for('claude-opus-4-6')
        self.assertEqual(rules.effort_for(r46, 'xhigh'), 'high')
        self.assertEqual(rules.effort_for(r46, 'max'), 'max')
        self.assertEqual(rules.effort_for(rules.rules_for('claude-opus-5-5'), 'minimal'), 'low')
        self.assertIsNone(rules.effort_for(rules.rules_for('claude-haiku-4-5'), 'high'))
        self.assertEqual([rules.effort_from_budget(b) for b in (1024, 4096, 16384, 65536, -1)],
                         ['low', 'medium', 'high', 'xhigh', None])


class TranslatedBodies(unittest.TestCase):
    def test_every_family_gets_a_body_the_model_accepts(self):
        cases = [
            {'reasoning_effort': 'high', 'temperature': 0.3, 'top_p': 0.9},
            {'reasoning_effort': 'minimal'},
            {'reasoning_effort': 'xhigh', 'max_completion_tokens': 30000},
            {'reasoning_effort': 'none'},
            {'temperature': 0.2, 'top_p': 0.5, 'top_k': 5},
            {'tools': [WEATHER], 'tool_choice': 'required', 'reasoning_effort': 'medium'},
            {'tools': [WEATHER], 'tool_choice': {'type': 'function', 'function': {'name': 'get_weather'}},
             'parallel_tool_calls': False},
            {'response_format': {'type': 'json_schema', 'json_schema': {'name': 'w', 'schema': {
                'type': 'object', 'properties': {}, 'additionalProperties': False}}}},
        ]
        for model in FAMILIES:
            for case in cases:
                with self.subTest(model=model, case=case):
                    notes = []
                    try:
                        body = encode(case, model, notes)
                    except BrokerError as exc:
                        # only a model without structured outputs refuses, and with a typed 400
                        self.assertIn('response_format', case)
                        self.assertEqual(exc.code, 'broker_capability_unavailable')
                        self.assertFalse(rules.rules_for(model).structured_outputs)
                        continue
                    self.assertIsNone(model_rule_error(body), (body, notes))
                    self.assertNotIn('output_format', body)

    def test_current_models_lose_what_they_refuse_with_notes(self):
        notes = []
        body = encode({'reasoning_effort': 'high', 'temperature': 0.7, 'top_k': 3, 'tools': [WEATHER],
                       'tool_choice': 'required'}, 'claude-opus-5-5', notes)
        self.assertNotIn('temperature', body)
        self.assertNotIn('top_k', body)
        self.assertNotIn('budget_tokens', json.dumps(body))
        self.assertEqual(body['tool_choice'], {'type': 'auto'})
        self.assertEqual(body['output_config'], {'effort': 'high'})
        self.assertIn('dropped:temperature (the model refuses sampling parameters)', notes)
        self.assertIn('degraded:tool_choice=auto (the model refuses forced tool use)', notes)
        body = encode({'reasoning_effort': 'low'}, 'claude-opus-4-8')
        self.assertEqual(body['thinking'], {'type': 'adaptive', 'display': 'summarized'})
        self.assertEqual(body['output_config'], {'effort': 'low'})
        body = encode({'reasoning_effort': 'medium'}, 'claude-opus-4-6')
        self.assertEqual(body['thinking'], {'type': 'adaptive'})
        body = encode({'reasoning_effort': 'none'}, 'claude-sonnet-5-5')
        self.assertEqual(body['thinking'], {'type': 'between_tools'})
        body = encode({'reasoning_effort': 'none'}, 'claude-opus-5')
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        notes = []
        body = encode({'reasoning_effort': 'none'}, 'claude-opus-5-5', notes)
        self.assertNotIn('thinking', body)
        self.assertIn('kept:thinking (always on for this model)', notes)

    def test_older_models_keep_their_forms(self):
        body = encode({'reasoning_effort': 'medium', 'max_completion_tokens': 9000}, 'claude-sonnet-4-5-20250929')
        self.assertEqual(body['thinking'], {'type': 'enabled', 'budget_tokens': 4096})
        self.assertNotIn('output_config', body)
        notes = []
        body = encode({'temperature': 0.2, 'top_p': 0.5}, 'claude-haiku-4-5-20251001', notes)
        self.assertEqual((body.get('temperature'), body.get('top_p')), (0.2, None))
        self.assertIn('dropped:top_p (the model takes temperature or top_p, not both)', notes)
        body = encode({'temperature': 0.2, 'top_p': 0.5}, 'claude-opus-4-6')
        self.assertEqual((body['temperature'], body['top_p']), (0.2, 0.5))
        notes = []
        body = encode({'tools': [WEATHER], 'tool_choice': 'required', 'reasoning_effort': 'high'},
                      'claude-sonnet-4-5-20250929', notes)
        self.assertEqual(body['tool_choice']['type'], 'any')
        self.assertNotIn('thinking', body)
        self.assertIn('dropped:thinking (forced tool_choice)', notes)
        notes = []
        body = encode({'reasoning_effort': 'high'}, 'claude-3-haiku-20240307', notes)
        self.assertNotIn('thinking', body)
        self.assertIn('dropped:reasoning (the model has no extended thinking)', notes)

    def test_structured_outputs_and_default_cap(self):
        schema = {'type': 'json_schema', 'json_schema': {'name': 'w', 'schema': {'type': 'object'}}}
        body = encode({'response_format': schema}, 'claude-opus-5-5')
        self.assertEqual(body['output_config'], {'format': {'type': 'json_schema', 'schema': {'type': 'object'}}})
        with self.assertRaises(BrokerError) as ctx:
            encode({'response_format': schema}, 'claude-sonnet-4-20250514')
        self.assertEqual(ctx.exception.status, 400)
        with self.assertRaises(BrokerError):
            encode({'response_format': {'type': 'json_object'}}, 'claude-opus-5-5')
        notes = []
        self.assertEqual(encode({}, 'claude-opus-5-5', notes)['max_tokens'], rules.ALWAYS_THINKING_MIN_MAX_TOKENS)
        self.assertIn('injected:max_tokens', notes)
        self.assertEqual(encode({}, 'claude-sonnet-4-5-20250929')['max_tokens'], 4096)
        self.assertEqual(encode({'max_completion_tokens': 50}, 'claude-opus-5-5')['max_tokens'], 50)

    def test_omitted_thinking_round_trip(self):
        resp = ant.decode_response(load('anthropic', 'messages_thinking_omitted').body({'model': 'claude-opus-5-5'}))
        message = openai_chat.encode_response(resp, model_name='alias')['choices'][0]['message']
        self.assertEqual(message['reasoning_signature'], 'mock-signature-omitted-0001')
        follow = {'tools': [WEATHER], 'messages': [
            {'role': 'user', 'content': 'weather in Paris?'}, message,
            {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'], 'content': 'sunny'}]}
        body = encode(follow, 'claude-opus-5-5')
        first = body['messages'][1]['content'][0]
        self.assertEqual(first, {'type': 'thinking', 'thinking': '', 'signature': 'mock-signature-omitted-0001'})
        self.assertIsNone(model_rule_error(body))
        # the budget form over the same loop keeps thinking because the signed block leads the turn
        body = encode(dict(follow, reasoning_effort='low', max_completion_tokens=5000), 'claude-sonnet-4-5-20250929')
        self.assertEqual(body['thinking']['type'], 'enabled')
        self.assertIsNone(model_rule_error(body))
        notes = []
        part = ant._encode_part(Thinking('', 'skip_thought_signature_validator'), notes=notes)
        self.assertIsNone(part)
        self.assertIn('dropped:thinking_without_signature', notes)


class DecodeAnthropic(unittest.TestCase):
    def test_output_config_adaptive_thinking_and_documents(self):
        notes = []
        ir = ant.decode_request({
            'model': 'm', 'max_tokens': 100, 'thinking': {'type': 'adaptive'},
            'output_config': {'effort': 'xhigh', 'format': {'type': 'json_schema', 'schema': {'type': 'object'}},
                              'task_budget': {'type': 'tokens', 'total': 20000}},
            'messages': [{'role': 'user', 'content': [
                {'type': 'document', 'source': {'type': 'text', 'media_type': 'text/plain', 'data': 'The sky.'}},
                {'type': 'text', 'text': 'Color?'}]}]}, notes=notes)
        self.assertEqual((ir.reasoning.enabled, ir.reasoning.effort), (True, 'xhigh'))
        self.assertEqual(ir.response_format['json_schema']['schema'], {'type': 'object'})
        self.assertEqual(ir.messages[0].parts[0].text, 'The sky.')
        self.assertIn('converted:document', notes)
        self.assertIn('dropped:output_config.task_budget', notes)
        self.assertNotIn('dropped:output_config', notes)
        ir = ant.decode_request({'model': 'm', 'max_tokens': 5, 'output_config': {'effort': 'low'},
                                 'messages': [{'role': 'user', 'content': 'x'}]})
        self.assertEqual(ir.reasoning.effort, 'low')
        ir = ant.decode_request({'model': 'm', 'max_tokens': 5, 'thinking': {'type': 'disabled'},
                                 'output_config': {'effort': 'low'}, 'messages': [{'role': 'user', 'content': 'x'}]})
        self.assertIsNone(ir.reasoning)
        oai = openai_chat.encode_request(ant.decode_request({
            'model': 'm', 'max_tokens': 5, 'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'},
            'messages': [{'role': 'user', 'content': 'x'}]}), PROFILES['openai'])
        self.assertEqual(oai.get('reasoning_effort'), 'high')
        with self.assertRaises(BrokerError):  # a file-backed document cannot leave Anthropic
            ant.decode_request({'model': 'm', 'max_tokens': 5, 'messages': [{'role': 'user', 'content': [
                {'type': 'document', 'source': {'type': 'file', 'file_id': 'file_x'}}]}]})


class AdvisorInjection(unittest.TestCase):
    BODY = {'model': 'x', 'max_tokens': 8000, 'messages': [{'role': 'user', 'content': 'hi'}]}

    def inject(self, model, body=None):
        notes = []
        out = inject_reasoning('anthropic.messages', dict(body or self.BODY), 'high', PROFILES['anthropic'].features,
                               notes, model=model)
        return out, notes

    def test_shaped_per_model(self):
        out, notes = self.inject('claude-sonnet-4-5-20250929')
        self.assertEqual(out['thinking'], {'type': 'enabled', 'budget_tokens': 7999})
        out, notes = self.inject('claude-opus-4-7')
        self.assertEqual(out['thinking'], {'type': 'adaptive', 'display': 'summarized'})
        self.assertEqual(out['output_config'], {'effort': 'high'})
        self.assertEqual(notes, ['injected:thinking', 'injected:effort'])
        out, notes = self.inject('claude-opus-5-5')
        self.assertEqual(out['thinking'], {'type': 'adaptive', 'display': 'summarized'})
        for model in ('claude-sonnet-4-5-20250929', 'claude-opus-4-7', 'claude-opus-5-5', 'claude-fable-5-1'):
            self.assertIsNone(model_rule_error(dict(self.inject(model)[0], model=model)), model)
        # the client's own choices win
        out, notes = self.inject('claude-opus-5-5', dict(self.BODY, output_config={'effort': 'low'}))
        self.assertEqual((out['output_config'], notes), ({'effort': 'low'}, []))
        out, notes = self.inject('claude-opus-4-7', dict(self.BODY, thinking={'type': 'disabled'}))
        self.assertEqual((out['thinking'], notes), ({'type': 'disabled'}, []))
        # without a model (pure use) the budget form stays
        notes = []
        out = inject_reasoning('anthropic.messages', dict(self.BODY), 'high', PROFILES['anthropic'].features, notes)
        self.assertEqual(out['thinking']['type'], 'enabled')


class EstimateUsageAndPrices(unittest.TestCase):
    def test_estimate(self):
        image = base64.b64encode(b'\x89PNG' + b'\0' * 300_000).decode()
        pdf = base64.b64encode(b'%PDF' + b'\0' * 30_000).decode()
        body = {'model': 'm', 'max_tokens': 5, 'messages': [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': image}},
            {'type': 'document', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': pdf}},
            {'type': 'text', 'text': 'hi'}]}]}
        raw = len(json.dumps(body))
        estimate = rules.estimate_input_tokens(body, raw)
        self.assertLess(estimate, raw // 3)
        self.assertGreaterEqual(estimate, rules.IMAGE_TOKEN_ALLOWANCE + 30_000 // 3)
        self.assertEqual(rules.estimate_input_tokens({}, 30_000_000), rules.MAX_CONTEXT_TOKENS)

    def test_usage_details_and_prices(self):
        body = load('anthropic', 'messages_server_tools').body({'model': 'claude-opus-5-5'})
        usage = anthropic_usage(body['usage'])
        self.assertEqual((usage.input_tokens, usage.cached_input_tokens, usage.cache_write_tokens),
                         (3500, 500, 2000))
        self.assertEqual((usage.cache_write_1h_tokens, usage.web_search_requests, usage.fast), (1500, 2, True))
        # a stream: message_start carries the input, message_delta the cumulative server tool use
        start = anthropic_usage({'input_tokens': 10, 'output_tokens': 1})
        delta = anthropic_usage({'output_tokens': 20, 'server_tool_use': {'web_search_requests': 1}}, start)
        self.assertEqual((delta.input_tokens, delta.output_tokens, delta.web_search_requests), (10, 20, 1))
        full = {'match': 'claude-opus-5-5@*', 'input_per_mtok': '4', 'output_per_mtok': '20',
                'cached_input_per_mtok': '0.4', 'cache_write_per_mtok': '5', 'cache_write_1h_per_mtok': '8',
                'per_web_search_request': '0.01', 'fast_multiplier': '2'}
        priced = PriceTable([full]).price('claude-opus-5-5@ant', usage)
        tokens = (Decimal(1000) * 4 + Decimal(500) * Decimal('0.4') + Decimal(500) * 5 + Decimal(1500) * 8
                  + Decimal(300) * 20) / Decimal(1_000_000)
        self.assertEqual(priced.basis, 'tokens')
        self.assertEqual(priced.cost, tokens * 2 + Decimal('0.02'))
        for missing in ('per_web_search_request', 'fast_multiplier'):
            row = {k: v for k, v in full.items() if k != missing}
            self.assertIsNone(PriceTable([row]).price('claude-opus-5-5@ant', usage).cost, missing)
        row = {k: v for k, v in full.items() if k != 'cache_write_1h_per_mtok'}
        cheaper = PriceTable([row]).price('claude-opus-5-5@ant', usage).cost
        self.assertEqual(cheaper, (tokens - Decimal(1500) * 3 / Decimal(1_000_000)) * 2 + Decimal('0.02'))

    def test_model_meta(self):
        row = load('anthropic', 'models_get').body({})
        meta = model_meta(row)
        self.assertEqual((meta['context_length'], meta['max_output_tokens'], meta['vision']), (200000, 64000, True))
        self.assertEqual(meta['anthropic']['display_name'], 'Claude Sonnet 4.5')
        self.assertEqual(model_meta({'id': 'x'}), {'anthropic': {}})


class ProviderState(unittest.TestCase):
    def test_references(self):
        file_doc = {'role': 'user', 'content': [
            {'type': 'document', 'source': {'type': 'file', 'file_id': 'file_a'}},
            {'type': 'image', 'source': {'type': 'file', 'file_id': 'file_b'}},
            {'type': 'container_upload', 'file_id': 'file_c'}]}
        self.assertEqual(anthropic_state.file_ids({'messages': [file_doc]}), ['file_a', 'file_b', 'file_c'])
        self.assertEqual(anthropic_state.state_references({'container': 'container_1', 'messages': []}),
                         ['container'])
        self.assertEqual(anthropic_state.state_references({'container': {'skills': [
            {'type': 'anthropic', 'skill_id': 'pptx'}]}, 'messages': []}), [])
        self.assertEqual(anthropic_state.state_references({'container': {'skills': [
            {'type': 'custom', 'skill_id': 'skill_1'}]}, 'messages': [file_doc]}), ['skills', 'file'])
        self.assertEqual(anthropic_state.state_references({'messages': [{'role': 'user', 'content': 'x'}]}), [])
        with self.assertRaises(BrokerError) as ctx:
            anthropic_state.check({'container': 'c'}, principal=None, allow_store=False, store_scope=False,
                                  handles=None, upstream_id='ant', role='standalone')
        self.assertEqual((ctx.exception.status, ctx.exception.code), (403, 'scope_denied'))
        with self.assertRaises(BrokerError) as ctx:
            anthropic_state.check({'container': 'c'}, principal=None, allow_store=False, store_scope=True,
                                  handles=None, upstream_id='ant', role='standalone')
        self.assertEqual(ctx.exception.code, 'data_policy_denied')
        self.assertEqual(anthropic_state.batch_params({'requests': [{'custom_id': 'a', 'params': {'x': 1}}, 'bad']}),
                         [{'x': 1}])
        self.assertIsInstance(IRReasoning(), IRReasoning)


if __name__ == '__main__':
    unittest.main()
