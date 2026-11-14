"""A model's coherent wrong answer must not fail an engine contract check."""
import json
import unittest
from check_api_plus import text_observation
from check_native import weather_arguments, response_terminal


class QualityBoundary(unittest.TestCase):
    def test_coherent_wrong_translation_keeps_engine_success_and_quality_failure(self):
        body = {'text': 'The house is wonderful.'}  # Valid output, wrong target language.
        result = text_observation(body, .1, 'French translation contains maison', False)
        self.assertEqual(result['engine_status'], 'passed')
        self.assertEqual(result['model_quality']['status'], 'failed')
        self.assertIs(result['response'], body)

    def test_malformed_or_empty_output_still_fails_engine_contract(self):
        for body in ({}, {'text': None}, {'text': ['words']}, {'text': '  '}):
            with self.subTest(body=body), self.assertRaises(AssertionError):
                text_observation(body, .1, 'quality is not a schema check', True)

    def test_wrong_tool_city_is_quality_failure_but_valid_tool_schema(self):
        result = weather_arguments('{"city":"London"}')
        self.assertEqual(result['engine_status'], 'passed')
        self.assertEqual(result['model_quality']['status'], 'failed')
        self.assertEqual(result['arguments'], {'city': 'London'})

    def test_wrong_tool_argument_schema_still_fails(self):
        for raw in ('{}', '{"city":42}', '{"city":"Paris","extra":true}'):
            with self.subTest(raw=raw), self.assertRaises(AssertionError):
                weather_arguments(raw)

    def test_requested_token_cap_is_a_valid_stream_termination(self):
        event = {'type': 'response.incomplete', 'response': {'status': 'incomplete',
                 'usage': {'output_tokens': 20}, 'incomplete_details': {'reason': 'max_output_tokens'}}}
        self.assertEqual(response_terminal(['data: '+json.dumps(event)], 20)['status'], 'incomplete')
        event['response']['usage']['output_tokens'] = 21
        with self.assertRaises(AssertionError):
            response_terminal(['data: '+json.dumps(event)], 20)


if __name__ == '__main__':
    unittest.main()
