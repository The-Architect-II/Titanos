"""Provider wire fixtures (WP4): every tests/fixtures/providers/<provider>/<case>.jsonl has the header row {source_url,
retrieved, transcriber_note} and well-formed rows; every fixture that carries an example request is sent to a fresh loopback
mock of its provider, and the answer matches the fixture byte for byte (status, content type, fixture headers, body; binary
fixtures per variant). The structure is checked against the provider protocols the broker relies on: Anthropic SSE event
names and content_block_delta delta types (all eight event names and all four delta types appear, in protocol order), the
Responses event family with increasing sequence_number, OpenAI chat streams ending with [DONE] and the include_usage chunk
(choices:[]) only when requested, Gemini SSE frames without event names or [DONE], Ollama NDJSON ending with done:true, and
the GIM advisor fields of gim_api.py:670-686.

Loopback only (netguard installed); synthetic credentials only.
"""
import base64
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from broker_mocks import (AdvisorMock, AnthropicMock, AzureMock, ElevenLabsMock, GeminiMock, OllamaCloudMock,  # noqa: E402
                          OpenAIMock, RouterUpstreamMock, SENTINELS, all_fixtures, netguard, parse_sse)
from broker_mocks.compat_mock import CompatMock  # noqa: E402
from broker_mocks.fixtures import FIXTURES_DIR, HEADER_KEYS, KINDS  # noqa: E402

import httpx  # noqa: E402

netguard.install()
WRONG = SENTINELS['wrong']
ANTHROPIC_EVENTS = {'message_start', 'content_block_start', 'content_block_delta', 'content_block_stop', 'message_delta',
                    'message_stop', 'ping', 'error'}
ANTHROPIC_DELTAS = {'text_delta', 'input_json_delta', 'thinking_delta', 'signature_delta'}
RESPONSES_TEXT_FAMILY = ['response.created', 'response.in_progress', 'response.output_item.added',
                         'response.content_part.added', 'response.output_text.delta', 'response.output_text.done',
                         'response.content_part.done', 'response.output_item.done', 'response.completed']
RESPONSES_TOOL_FAMILY = ['response.created', 'response.in_progress', 'response.output_item.added',
                         'response.function_call_arguments.delta', 'response.function_call_arguments.done',
                         'response.output_item.done', 'response.completed']
# OpenAI media event streams (case -> type of the final event): no [DONE], usage on the final event
OPENAI_MEDIA_STREAMS = {'images_stream': 'image_generation.completed', 'images_edits_stream': 'image_edit.completed',
                        'transcriptions_stream': 'transcript.text.done', 'speech_sse': 'speech.audio.done'}
# provider directory -> (mock factory, credential style, sentinel name)
MOCKS = {
    'openai': (OpenAIMock, 'bearer', 'openai'),
    'azure_openai': (AzureMock, 'api-key', 'azure_openai'),
    # the OpenAI-compatible family rows (broker_mocks.compat_mock: a superset of the OpenAIMock variants)
    'xai': (lambda: CompatMock('xai'), 'bearer', 'xai'),
    'mistral': (lambda: CompatMock('mistral'), 'bearer', 'mistral'),
    'groq': (lambda: CompatMock('groq'), 'bearer', 'groq'),
    'together': (lambda: CompatMock('together'), 'bearer', 'together'),
    'fireworks': (lambda: CompatMock('fireworks'), 'bearer', 'fireworks'),
    'deepseek': (lambda: CompatMock('deepseek'), 'bearer', 'deepseek'),
    'perplexity': (lambda: CompatMock('perplexity'), 'bearer', 'perplexity'),
    'anthropic': (AnthropicMock, 'anthropic', 'anthropic'),
    'gemini': (GeminiMock, 'x-goog-api-key', 'gemini'),
    'elevenlabs': (ElevenLabsMock, 'xi-api-key', 'elevenlabs'),
    'ollama': (OllamaCloudMock, 'bearer', 'ollama_cloud'),
    'openrouter': (lambda: RouterUpstreamMock('openrouter'), 'bearer', 'openrouter'),
    'advisor': (None, 'none', 'advisor'),
}


def auth_headers(style, sentinel, mode):
    """Credential headers for a fixture request; mode is valid (default), missing, wrong or no_version."""
    if style == 'none' or mode == 'missing':
        return {'anthropic-version': '2023-06-01'} if style == 'anthropic' else {}
    token = WRONG if mode == 'wrong' else SENTINELS[sentinel]
    if style == 'bearer':
        return {'authorization': 'Bearer ' + token}
    if style == 'anthropic':
        headers = {'x-api-key': token}
        if mode != 'no_version':
            headers['anthropic-version'] = '2023-06-01'
        return headers
    return {style: token}


def send(http, mock, style, sentinel, request, patch=None):
    request = dict(request)
    patch = patch or {}
    kwargs = {'headers': auth_headers(style, sentinel, request.get('auth', 'valid'))}
    query = dict(request.get('query') or {})
    query.update(patch.get('query') or {})
    if query:
        kwargs['params'] = query
    if 'json' in request:
        kwargs['json'] = dict(request['json'], **(patch.get('json') or {}))
    if 'form' in request:
        kwargs['data'] = request['form']
    if 'files' in request:
        kwargs['files'] = {name: (spec['filename'], base64.b64decode(spec['b64']), spec['content_type'])
                           for name, spec in request['files'].items()}
    return http.request(request['method'], mock.base_url + request['path'], **kwargs)


class FixtureFiles(unittest.TestCase):
    def test_every_fixture_is_well_formed(self):
        fixtures = list(all_fixtures())
        self.assertGreaterEqual(len(fixtures), 60)
        self.assertEqual({fx.provider for fx in fixtures}, set(MOCKS))
        for fx in fixtures:
            with self.subTest(fixture=f'{fx.provider}/{fx.case}'):
                for key in HEADER_KEYS:
                    self.assertTrue(fx.header.get(key), key)
                self.assertRegex(fx.header['retrieved'], r'^\d{4}-\d{2}-\d{2}$')
                self.assertRegex(fx.header['source_url'], r'^(https://|urn:|source:)')
                self.assertIn(fx.kind, KINDS)
                self.assertTrue(re.fullmatch(r'[a-z0-9_]+', fx.case))
                for name in fx.headers:
                    self.assertEqual(name, name.lower())
                    self.assertNotIn(name, {'authorization', 'x-api-key', 'xi-api-key', 'api-key', 'x-goog-api-key',
                                            'cookie', 'set-cookie'})
                raw = fx.path.read_bytes()
                # LF as written; a core.autocrlf checkout may turn every line ending into CRLF, never a bare CR
                self.assertEqual(raw.replace(b'\r\n', b'\n').count(b'\r'), 0)
                self.assertTrue(raw.endswith(b'\n'))
                if fx.kind == 'sse':
                    for row in fx.rows:
                        self.assertTrue('comment' in row or 'data' in row, row)
                if fx.kind == 'binary':
                    for row in fx.rows:
                        self.assertEqual(set(row), {'variant', 'content_type', 'chunk_b64'})
                fx.render(dict(fx.header.get('vars') or {}))  # every placeholder has a value

    def test_fixture_dir_holds_only_jsonl(self):
        for path in FIXTURES_DIR.rglob('*'):
            if path.is_file():
                self.assertEqual(path.suffix, '.jsonl', path.name)


class MockMatchesFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = httpx.Client(timeout=15.0, trust_env=False)
        cls.mocks = {}

    @classmethod
    def tearDownClass(cls):
        cls.http.close()
        for mock in cls.mocks.values():
            mock.stop()

    def mock_for(self, fx):
        key = fx.provider if fx.provider != 'advisor' else ('advisor-gim' if fx.case.startswith('gim') else 'advisor-generic')
        if key not in self.mocks:
            if key == 'advisor-gim':
                mock = AdvisorMock('gim')
            elif key == 'advisor-generic':
                mock = AdvisorMock('generic-v1')
            else:
                mock = MOCKS[fx.provider][0]()
            mock.start()
            self.mocks[key] = mock
        mock = self.mocks[key]
        mock.reset()
        return mock

    def check_reply(self, fx, reply, variables, variant=None):
        self.assertEqual(reply.status_code, fx.status, reply.text[:300])
        expected_type = fx.variants()[variant] if variant else fx.headers.get('content-type')
        self.assertEqual(reply.headers.get('content-type'), expected_type)
        for name, value in fx.headers.items():
            if name != 'content-type':
                self.assertEqual(reply.headers.get(name), value, name)
        self.assertEqual(reply.content, fx.render(variables, variant=variant))

    def test_every_fixture_request_replays_byte_for_byte(self):
        replayed = 0
        for fx in all_fixtures():
            request = fx.header.get('request')
            if request is None:
                continue
            with self.subTest(fixture=f'{fx.provider}/{fx.case}'):
                mock = self.mock_for(fx)
                _, style, sentinel = MOCKS[fx.provider]
                variables = dict(fx.header.get('vars') or {})
                if fx.kind == 'binary':
                    for variant, patch in fx.header['variant_requests'].items():
                        self.check_reply(fx, send(self.http, mock, style, sentinel, request, patch), variables, variant)
                    self.assertEqual(set(fx.header['variant_requests']), set(fx.variants()))
                else:
                    select = fx.header.get('select')
                    if select:
                        mock.respond_with(select['surface'], select['case'])
                    self.check_reply(fx, send(self.http, mock, style, sentinel, request), variables)
                hit = mock.hits[-1]
                for _, spelled in hit.credentials:
                    self.assertRegex(spelled, r'<(sentinel:[a-z_]+|foreign:[0-9a-f]{12})>')
                replayed += 1
        self.assertGreaterEqual(replayed, 60)


class ProtocolStructure(unittest.TestCase):
    def fixtures(self, provider, kind=None):
        return [fx for fx in all_fixtures() if fx.provider == provider and (kind is None or fx.kind == kind)]

    def test_anthropic_event_names_delta_types_and_order(self):
        names, deltas, blocks = set(), set(), set()
        for fx in self.fixtures('anthropic', 'sse'):
            events = fx.event_names()
            names.update(events)
            deltas.update(fx.delta_types())
            for row in fx.rows:
                data = row['data']
                self.assertEqual(row['event'], data['type'])  # event name equals data.type
                if data['type'] == 'content_block_start':
                    blocks.add(data['content_block']['type'])
            if events == ['error']:
                continue
            self.assertEqual(events[0], 'message_start')
            self.assertEqual(events[-2:], ['message_delta', 'message_stop'])
            open_blocks = set()
            for row in fx.rows:
                data = row['data']
                if data['type'] == 'content_block_start':
                    self.assertNotIn(data['index'], open_blocks)
                    open_blocks.add(data['index'])
                elif data['type'] == 'content_block_delta':
                    self.assertIn(data['index'], open_blocks)
                elif data['type'] == 'content_block_stop':
                    open_blocks.remove(data['index'])
            self.assertEqual(open_blocks, set())
        self.assertEqual(names, ANTHROPIC_EVENTS)
        self.assertEqual(deltas, ANTHROPIC_DELTAS)
        self.assertEqual(blocks, {'text', 'tool_use', 'thinking'})
        thinking = [fx for fx in self.fixtures('anthropic', 'sse') if fx.case == 'messages_stream_thinking'][0].delta_types()
        self.assertLess(thinking.index('thinking_delta'), thinking.index('signature_delta'))

    def test_responses_event_family(self):
        for case, family in (('responses_stream', RESPONSES_TEXT_FAMILY), ('responses_stream_tools', RESPONSES_TOOL_FAMILY)):
            fx = [f for f in self.fixtures('openai', 'sse') if f.case == case][0]
            names = fx.event_names()
            collapsed = [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]
            self.assertEqual(collapsed, family)
            self.assertEqual([row['data']['sequence_number'] for row in fx.rows], list(range(len(fx.rows))))
            self.assertEqual(names, fx.data_types())

    def test_openai_chat_streams(self):
        for fx in self.fixtures('openai', 'sse') + self.fixtures('azure_openai', 'sse') + self.fixtures('mistral', 'sse') \
                + self.fixtures('openrouter', 'sse'):
            if fx.case.startswith('responses') or fx.case in OPENAI_MEDIA_STREAMS:
                continue
            with self.subTest(fixture=f'{fx.provider}/{fx.case}'):
                variables = dict(fx.header.get('vars') or {})
                frames = parse_sse(fx.render(variables))
                self.assertEqual(frames[-1], ('', '[DONE]'))
                bodies = [json.loads(data) for _, data in frames[:-1]]
                usage_chunks = [b for b in bodies if b.get('choices') == [] and 'usage' in b]
                wants_usage = variables.get('include_usage') or fx.case.endswith('_usage')
                if fx.provider == 'openai':
                    self.assertEqual(len(usage_chunks), 1 if wants_usage else 0)
                    if usage_chunks:
                        self.assertIs(bodies[-1], usage_chunks[0])
                calls = [c for b in bodies for ch in b.get('choices', []) for c in ch.get('delta', {}).get('tool_calls', [])]
                if calls:
                    self.assertTrue({'id', 'type', 'function'} <= set(calls[0]) or fx.provider == 'mistral')
                    self.assertIn('name', calls[0]['function'])

    def test_openai_media_streams(self):
        """The OpenAI media event streams end with their done/completed event (no [DONE]); image events carry the
        SSE event name equal to type and PNG payloads; the final event carries usage."""
        found = {fx.case: fx for fx in self.fixtures('openai', 'sse') if fx.case in OPENAI_MEDIA_STREAMS}
        self.assertEqual(set(found), set(OPENAI_MEDIA_STREAMS))
        for case, last in OPENAI_MEDIA_STREAMS.items():
            fx = found[case]
            with self.subTest(fixture=case):
                frames = parse_sse(fx.render(dict(fx.header.get('vars') or {})))
                bodies = [json.loads(data) for _, data in frames]
                self.assertEqual(bodies[-1]['type'], last)
                self.assertIn('usage', bodies[-1])
                self.assertNotIn(('', '[DONE]'), frames)
                for event, body in zip((e for e, _ in frames), bodies):
                    if case.startswith('images'):
                        self.assertEqual(event, body['type'])
                        self.assertEqual(base64.b64decode(body['b64_json'])[:8], b'\x89PNG\r\n\x1a\n')

    def test_gemini_and_ollama_framing(self):
        for fx in self.fixtures('gemini', 'sse'):
            self.assertEqual(set(fx.event_names()), {''})
            for row in fx.rows:
                self.assertIn('candidates', row['data'])
            self.assertIn('finishReason', fx.rows[-1]['data']['candidates'][0])
            self.assertIn('usageMetadata', fx.rows[-1]['data'])
        for fx in self.fixtures('ollama', 'ndjson'):
            lines = [row['line'] for row in fx.rows]
            self.assertTrue(lines[-1]['done'])
            self.assertTrue(all(not line['done'] for line in lines[:-1]))

    def test_mistral_tool_ids_and_png_versus_jpeg(self):
        for fx in self.fixtures('mistral'):
            text = fx.path.read_text(encoding='utf-8')
            for call_id in re.findall(r'"id": "([^"]+)", "function"', text):
                self.assertRegex(call_id, r'^[A-Za-z0-9]{9}$')
        for fx in self.fixtures('openai', 'json'):
            for item in (fx.body(dict(fx.header.get('vars') or {})).get('data') or []) if fx.case.startswith('images') else []:
                if 'b64_json' in item:
                    self.assertEqual(base64.b64decode(item['b64_json'])[:8], b'\x89PNG\r\n\x1a\n')
        for fx in self.fixtures('xai', 'json'):
            for item in fx.body(dict(fx.header.get('vars') or {}))['data']:
                if 'b64_json' in item:
                    self.assertEqual(base64.b64decode(item['b64_json'])[:3], b'\xff\xd8\xff')

    def test_gim_advisor_fields(self):
        fx = [f for f in self.fixtures('advisor') if f.case == 'gim_func_1'][0]
        body = fx.body({})
        self.assertEqual(set(body), {'ok', 'result'})
        result = body['result']
        self.assertEqual(list(result), ['recommended_model', 'reasoning_mode', 'confidence', 'route_reason', 'scores',
                                        'fallback_chain', 'classification', 'latency_ms'])
        self.assertEqual(set(result['classification']), {'strength', 'reasoning', 'domain'})
        self.assertLessEqual(len(result['fallback_chain']), 3)
        self.assertNotIn(result['recommended_model'], result['fallback_chain'])
        self.assertEqual(result['confidence'], result['scores'][result['recommended_model']])
        self.assertIn(result['reasoning_mode'], ('on', 'off'))


if __name__ == '__main__':
    unittest.main(verbosity=1)
