"""Gemini provider completeness (2026-10-02 round): typed bridges, field spellings, thinking per model, media in answers,
large stream frames, per-model catalog capabilities, the native rows the round added, and the jobs lane.

Against the WP4 GeminiMock (tests/broker_mocks/gemini_mock.py, fixtures transcribed from the public reference with
synthetic values) through the integrated app (create_app: Guard, BrokerDispatch); the netguard is installed and every
test ends with zero non-loopback connection attempts. Synthetic sentinel credentials only.

- codec: snake_case request fields (generation_config, inline_data, function_call ...) decode like lowerCamelCase; the
  output cap and the advisor's thinking land in the config object the body already uses (never a second one); thinking
  maps per model (thinkingLevel on gemini-3, thinkingBudget on 2.5, nothing for TTS, image and 2.0 models, every change
  noted); json_schema travels lossless as responseJsonSchema on 2.5 and later; audio parts and IR Audio round trip;
  answers keep images and audio, code execution becomes fenced text and grounding is noted; the data policy classifies
  fileData, video and PDF parts; per-model capabilities by family and published methods;
- bridges (cells openai.audio.speech, openai.images.generations, openai.images.edits, openai.audio.transcriptions and
  openai.audio.translations |GEM|T): the exact generateContent bodies, the OpenAI answers, the X-Titanos-Dropped notes and
  the refusals before egress (zero upstream hits);
- a chat translation to an image model returns message.images; one to a TTS model is refused before egress;
- an image frame over 1 MiB streams through the passthrough and the translated stream;
- a budgeted key's cap goes into generation_config when the body uses that spelling;
- the live catalog: TTS, image, embedding, Imagen and Veo models carry their own capabilities in /v1/models, and
  /v1beta/models passes the upstream's supportedGenerationMethods and token limits through;
- native rows: the /download media path, Interactions (create, also streamed; owner-bound get), File Search stores and
  :importFile, account-wide webhooks and auth_tokens need broker:admin, and the redirecting download form is refused;
- jobs: a speech job on a Gemini TTS model gives a WAV artifact, an image job on an image model a PNG artifact;
- bare model ids: google-genai puts the provider's own id in the path (models/gemini-2.5-flash:generateContent) and can
  send nothing else, so a bare id resolves on the single visible Gemini upstream (profile extras bare_model_ids), and
  the catalog reports titanos.model_capabilities for every Gemini entry.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_adapter_support as A  # noqa: E402  (installs the netguard; main() records the matrix cells)
import broker_wp7_support as W  # noqa: E402
from broker_mocks import GeminiMock, netguard  # noqa: E402

from titanos.broker import inbound  # noqa: E402
from titanos.broker.adapters.gemini import GeminiAdapter  # noqa: E402
from titanos.broker.dispatch import inject_reasoning  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.policy import _modalities  # noqa: E402
from titanos.broker.profiles import PROFILES  # noqa: E402
from titanos.broker.sse import gemini_usage  # noqa: E402
from titanos.broker.surfaces import SURFACES  # noqa: E402
from titanos.broker.translate import gemini, gemini_media, openai_chat  # noqa: E402
from titanos.broker.translate.common import aiter_list  # noqa: E402
from titanos.broker.translate.ir import Audio, TextDelta  # noqa: E402

BOOT = 'gemini-completeness-boot-0001'
MOCKS: dict = {}
PNG = base64.b64decode(GeminiMock().extra_vars['image_b64'])
MODELS = ['gemini-2.5-flash', 'gemini-3-pro-preview', 'gemini-2.5-flash-preview-tts', 'gemini-2.5-flash-image',
          'gemini-embedding-001']


def setUpModule():
    MOCKS['gemini'] = GeminiMock()
    MOCKS['gemini'].install_completeness_routes()
    MOCKS['gemini'].start()


def tearDownModule():
    MOCKS['gemini'].stop()


def document(*, live: bool = False) -> dict:
    upstream = {'profile': 'gemini', 'base_url': MOCKS['gemini'].api_base, 'credential': 'gem-key', 'enabled': True,
                'models': list(MODELS), 'data_policy': {'allow_store': True}}
    if live:
        upstream['catalog'] = 'live'
    return {'policy': {'cloud_policy': 'explicit'}, 'upstreams': {'gem': upstream},
            'prices': [{'match': '*@gem', 'input_per_mtok': '1', 'output_per_mtok': '1'}]}


def bearer(token: str) -> dict:
    return {'Authorization': 'Bearer ' + token}


def wav_bytes(seconds: float = 0.25, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b'\x00\x00' * int(seconds * rate))
    return buffer.getvalue()


# ---------------------------------------------------------------------------------------------------------------------
# codec (no network)

class Codec(unittest.TestCase):
    def test_snake_case_request_decodes_like_camel_case(self):
        audio = base64.b64encode(b'RIFFmock').decode()
        snake = {'contents': [{'role': 'user', 'parts': [{'text': 'hi'},
                                                         {'inline_data': {'mime_type': 'audio/wav', 'data': audio}}]},
                              {'role': 'model', 'parts': [{'function_call': {'name': 'f', 'args': {'snake_arg': 1}}}]},
                              {'role': 'user', 'parts': [{'function_response': {'name': 'f',
                                                                                'response': {'output': 'ok'}}}]}],
                 'system_instruction': {'parts': [{'text': 'Be brief.'}]},
                 'tools': [{'function_declarations': [{'name': 'f', 'parameters': {
                     'type': 'OBJECT', 'properties': {'snake_arg': {'type': 'INTEGER'}}}}]}],
                 'tool_config': {'function_calling_config': {'mode': 'ANY', 'allowed_function_names': ['f']}},
                 'generation_config': {'max_output_tokens': 64, 'temperature': 0.2,
                                       'thinking_config': {'thinking_level': 'low'}},
                 'safety_settings': [{'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'}]}
        notes: list = []
        ir = gemini.decode_request(snake, model='models/m', notes=notes)
        self.assertEqual(ir.max_output_tokens, 64)
        self.assertEqual(ir.sampling, {'temperature': 0.2})
        self.assertEqual((ir.tool_choice.mode, ir.tool_choice.name), ('tool', 'f'))
        self.assertEqual(ir.reasoning.effort, 'low')
        self.assertIsInstance(ir.messages[0].parts[1], Audio)
        self.assertEqual(ir.tools[0].parameters['properties'], {'snake_arg': {'type': 'integer'}})  # names kept
        self.assertEqual(ir.messages[1].parts[0].input_json, '{"snake_arg":1}')  # user data never renamed
        self.assertIn('dropped:safetySettings', notes)  # nothing ignored silently any more
        camel = gemini.camelize(snake)
        self.assertEqual(camel['generationConfig']['maxOutputTokens'], 64)
        self.assertNotIn('generation_config', camel)

    def test_output_cap_and_thinking_follow_the_body_spelling_and_model(self):
        snake = {'contents': [], 'generation_config': {'temperature': 1}}
        self.assertIsNone(inbound.output_cap('gemini.generateContent', snake))
        capped = inbound.inject_output_cap('gemini.generateContent', snake, 77)
        self.assertEqual(capped, {'contents': [], 'generation_config': {'temperature': 1, 'max_output_tokens': 77}})
        self.assertNotIn('generationConfig', capped)
        own = {'contents': [], 'generation_config': {'max_output_tokens': 5}}
        self.assertEqual(inbound.output_cap('gemini.generateContent', own), 5)
        self.assertEqual(inbound.inject_output_cap('gemini.generateContent', own, 77), own)
        notes: list = []
        out = inject_reasoning('gemini.generateContent', snake, 'medium', frozenset({'thinking'}), notes,
                               model='gemini-3-pro-preview')
        self.assertEqual(out['generation_config']['thinking_config'], {'thinkingLevel': 'HIGH'})
        self.assertNotIn('generationConfig', out)
        out = inject_reasoning('gemini.generateContent', {'contents': []}, 'low', frozenset({'thinking'}), notes,
                               model='gemini-2.5-flash')
        self.assertEqual(out['generationConfig']['thinkingConfig'], {'thinkingBudget': 1024})
        for model in ('gemini-2.5-flash-preview-tts', 'gemini-2.5-flash-image', 'gemini-2.0-flash'):
            with self.subTest(model=model):
                self.assertEqual(inject_reasoning('gemini.generateContent', {'contents': []}, 'high',
                                                  frozenset({'thinking'}), [], model=model), {'contents': []})
        self.assertEqual(notes, ['injected:thinkingLevel', 'injected:thinkingBudget'])
        already = {'contents': [], 'generationConfig': {'thinking_config': {'thinking_budget': 0}}}
        self.assertEqual(inject_reasoning('gemini.generateContent', already, 'high', frozenset({'thinking'}), [],
                                          model='gemini-2.5-flash'), already)

    def test_reasoning_maps_per_model_family(self):
        chat = {'model': 'x', 'messages': [{'role': 'user', 'content': 'hi'}], 'reasoning_effort': 'medium'}
        cases = (('gemini-3-pro-preview', {'thinkingLevel': 'HIGH', 'includeThoughts': True}, 'mapped:reasoning_effort'),
                 ('gemini-3-flash-preview', {'thinkingLevel': 'MEDIUM', 'includeThoughts': True}, None),
                 ('gemini-2.5-flash', {'thinkingBudget': 4096, 'includeThoughts': True}, 'mapped:reasoning_effort'),
                 ('gemini-2.0-flash', None, 'dropped:reasoning_effort'),
                 ('gemini-2.5-flash-image', None, 'dropped:reasoning_effort'))
        for model, expected, note in cases:
            with self.subTest(model=model):
                notes: list = []
                body = gemini.encode_request(openai_chat.decode_request(chat), PROFILES['gemini'], notes=notes,
                                             model=model)
                self.assertEqual((body.get('generationConfig') or {}).get('thinkingConfig'), expected)
                if note:
                    self.assertIn(note, notes)
        none = gemini.encode_request(openai_chat.decode_request(dict(chat, reasoning_effort='none')), notes=[],
                                     model='gemini-2.5-pro')
        self.assertEqual(none['generationConfig']['thinkingConfig'], {'thinkingBudget': 128, 'includeThoughts': True})
        for model in ('gemini-2.5-flash-preview-tts', 'gemini-embedding-001', 'imagen-4.0-generate-001'):
            with self.subTest(refused=model):
                with self.assertRaises(BrokerError) as ctx:
                    gemini.encode_request(openai_chat.decode_request(chat), notes=[], model=model)
                self.assertEqual(ctx.exception.code, 'broker_capability_unavailable')

    def test_lossless_json_schema_on_new_models(self):
        schema = {'type': 'object', 'properties': {'n': {'type': 'integer', 'enum': [1, 2]}},
                  'additionalProperties': False, 'required': ['n']}
        chat = openai_chat.decode_request({'model': 'x', 'messages': [{'role': 'user', 'content': 'hi'}],
                                           'response_format': {'type': 'json_schema',
                                                               'json_schema': {'name': 'o', 'schema': schema}},
                                           'tools': [{'type': 'function', 'function': {'name': 't',
                                                                                       'parameters': schema}}]})
        notes: list = []
        body = gemini.encode_request(chat, notes=notes, model='gemini-2.5-flash')
        self.assertEqual(body['generationConfig']['responseJsonSchema'], schema)
        self.assertNotIn('responseSchema', body['generationConfig'])
        self.assertEqual(body['tools'][0]['functionDeclarations'][0]['parametersJsonSchema'], schema)
        self.assertFalse([n for n in notes if n.startswith('dropped:schema')])
        old: list = []
        body = gemini.encode_request(chat, notes=old, model='gemini-2.0-flash')
        self.assertEqual(body['generationConfig']['responseSchema']['type'], 'OBJECT')
        self.assertIn('dropped:schema.additionalProperties', old)
        simple = {'type': 'object', 'properties': {'a': {'type': 'string'}}}
        plain = gemini.encode_request(openai_chat.decode_request({
            'model': 'x', 'messages': [{'role': 'user', 'content': 'hi'}],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'o', 'schema': simple}}}),
            notes=[], model='gemini-2.5-flash')
        self.assertIn('responseSchema', plain['generationConfig'])  # nothing lost: the reduced form stays

    def test_audio_input_round_trips_and_answers_keep_media(self):
        wav = wav_bytes(0.01)
        ir = openai_chat.decode_request({'model': 'x', 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'transcribe'},
            {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(wav).decode(), 'format': 'mp3'}}]}]})
        part = gemini.encode_request(ir, notes=[], model='gemini-2.5-flash')['contents'][0]['parts'][1]
        self.assertEqual(part['inlineData']['mimeType'], 'audio/mp3')
        answer = {'candidates': [{'content': {'parts': [
            {'text': 'Here.'}, {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(PNG).decode()}},
            {'executableCode': {'language': 'PYTHON', 'code': 'print(1)'}},
            {'codeExecutionResult': {'outcome': 'OUTCOME_OK', 'output': '1'}},
            {'inlineData': {'mimeType': 'audio/L16;codec=pcm;rate=24000', 'data': 'AAA='}}]},
            'finishReason': 'STOP', 'groundingMetadata': {'webSearchQueries': ['q']}}],
            'usageMetadata': {'promptTokenCount': 3, 'candidatesTokenCount': 5,
                              'promptTokensDetails': [{'modality': 'AUDIO', 'tokenCount': 2}],
                              'candidatesTokensDetails': [{'modality': 'AUDIO', 'tokenCount': 4}]}}
        notes: list = []
        resp = gemini.decode_response(answer, notes=notes)
        kinds = [type(p).__name__ for p in resp.content]
        self.assertEqual(kinds, ['Text', 'Image', 'Text', 'Text', 'Audio'])
        self.assertIn('```python\nprint(1)\n```', resp.content[2].text)
        self.assertEqual(sorted(notes), ['converted:executableCode', 'dropped:groundingMetadata'])
        self.assertEqual((resp.usage.audio_input_tokens, resp.usage.audio_output_tokens), (2, 4))
        chat = inbound.attach_media('openai.chat', resp, openai_chat.encode_response(resp, model_name='m'), [])
        self.assertTrue(chat['choices'][0]['message']['images'][0]['image_url']['url'].startswith('data:image/png;'))
        dropped: list = []
        inbound.attach_media('anthropic.messages', resp, {}, dropped)
        self.assertEqual(dropped, ['dropped:image_output', 'dropped:audio_output'])
        native = gemini.encode_response(resp, model_name='m')['candidates'][0]['content']['parts']
        self.assertTrue(any('inlineData' in p for p in native))

    def test_stream_decode_notes_media_and_streams_code(self):
        frames = [{'candidates': [{'content': {'parts': [{'inlineData': {'mimeType': 'image/png', 'data': 'AA=='}}]}}]},
                  {'candidates': [{'content': {'parts': [{'executableCode': {'language': 'PYTHON', 'code': 'x'}}]},
                                   'finishReason': 'STOP', 'groundingMetadata': {'a': 1}}],
                   'usageMetadata': {'promptTokenCount': 1, 'candidatesTokenCount': 2}}]
        notes: list = []

        async def go():
            return [event async for event in gemini.decode_stream(aiter_list(frames), notes=notes)]
        events = A.run(go())
        self.assertTrue(any(isinstance(e, TextDelta) and '```python' in e.text for e in events))
        self.assertEqual(sorted(notes), ['converted:executableCode', 'dropped:groundingMetadata',
                                         'dropped:image_output'])

    def test_data_policy_classifies_gemini_media(self):
        surface = SURFACES['gemini.generateContent']
        cases = ((({'fileData': {'mimeType': 'video/mp4', 'fileUri': 'files/a'}}), {'text', 'image', 'audio'}),
                 (({'file_data': {'file_uri': 'files/a'}}), {'text', 'image', 'audio'}),
                 (({'inline_data': {'mime_type': 'application/pdf', 'data': 'AA=='}}), {'text', 'image'}),
                 (({'inlineData': {'mimeType': 'audio/wav', 'data': 'AA=='}}), {'text', 'audio'}),
                 (({'text': 'plain'}), {'text'}))
        for part, needed in cases:
            with self.subTest(part=part):
                body = {'contents': [{'role': 'user', 'parts': [part]}]}
                self.assertEqual(_modalities(inbound.summarize(surface, body, None, 10)), needed)

    def test_model_capabilities_by_family_and_published_methods(self):
        caps = frozenset({'chat', 'tools', 'vision', 'responses', 'embeddings', 'speech', 'images', 'transcription'})
        narrow = GeminiAdapter.model_capabilities
        self.assertEqual(narrow('gemini-2.5-flash', caps), {'chat', 'tools', 'vision', 'responses', 'transcription'})
        self.assertEqual(narrow('gemini-2.5-flash-preview-tts', caps), {'chat', 'speech'})
        self.assertEqual(narrow('gemini-2.5-flash-image', caps), {'chat', 'vision', 'images', 'responses'})
        self.assertEqual(narrow('gemini-embedding-001', caps), {'embeddings'})
        self.assertEqual(narrow('imagen-4.0-generate-001', caps), frozenset())
        self.assertEqual(narrow('veo-3.1-generate-preview', caps), frozenset())
        self.assertEqual(narrow('gemini-2.5-flash-native-audio-preview', caps), frozenset())
        self.assertEqual(narrow('some-new-model', caps, {'supportedGenerationMethods': ['embedContent']}), frozenset())
        self.assertEqual(gemini.thinking_mode('gemini-flash-latest'), 'budget')

    def test_media_codecs(self):
        self.assertEqual(gemini_media.aspect_ratio('1792x1024'), ('16:9', None))
        self.assertEqual(gemini_media.aspect_ratio('4096x4096'), ('1:1', '4K'))
        self.assertEqual(gemini_media.aspect_ratio('auto'), (None, None))
        self.assertEqual(gemini_media.voice_name('puck'), 'Puck')
        self.assertIsNone(gemini_media.voice_name('alloy'))
        with self.assertRaises(BrokerError):
            gemini_media.aspect_ratio('wide')
        pcm, rate = gemini_media.speech_audio({'candidates': [{'content': {'parts': [
            {'inlineData': {'mimeType': 'audio/L16;codec=pcm;rate=16000', 'data': 'AAABAA=='}}]}}]})
        self.assertEqual((pcm, rate), (b'\x00\x00\x01\x00', 16000))
        with self.assertRaises(BrokerError) as ctx:
            gemini_media.image_outputs({'candidates': [{'content': {'parts': []}, 'finishReason': 'IMAGE_SAFETY'}]})
        self.assertEqual(ctx.exception.code, 'invalid_request')
        self.assertEqual(gemini_usage({'promptTokenCount': 2, 'promptTokensDetails': [
            {'modality': 'IMAGE', 'tokenCount': 2}]}).image_input_tokens, 2)

    def test_stage_transcription_through_gemini(self):
        from types import SimpleNamespace

        from broker_mocks.fixtures import load
        from titanos.broker.stages import CloudInvoker
        answer = json.dumps(load('gemini', 'generate_content_transcript').body({'model': 'm'})).encode()
        sent = []

        class Invoker(CloudInvoker):
            def _send(self, adapter, grant, method, path, *, cancel, deadline, stream=False, **kw):
                sent.append((method, path, kw))
                return None, answer
        adapter = GeminiAdapter(PROFILES['gemini'], None)
        grant = SimpleNamespace(target=SimpleNamespace(upstream_model='gemini-2.5-flash'))
        audio = wav_bytes(0.5)
        result, usage = Invoker(None)._transcription(adapter, grant, 'gemini-2.5-flash',
                                                     {'audio': audio, 'language': 'en'}, cancel=None, deadline=None)
        self.assertEqual((result['text'], result['duration'], result['language']),
                         ('Hello from the mock transcript.', 0.5, None))
        self.assertEqual(usage.input_tokens, 120)
        method, path, kw = sent[0]
        self.assertEqual((method, path, kw['path_params']), ('POST', '/models/{model}:generateContent',
                                                             {'model': 'gemini-2.5-flash'}))
        self.assertIn('language with code en', kw['json']['contents'][0]['parts'][0]['text'])


# ---------------------------------------------------------------------------------------------------------------------
# through the integrated app

class Base(unittest.TestCase):
    live = False
    settings_kw: dict = {}

    def setUp(self):
        netguard.reset_attempts()
        self.mock = MOCKS['gemini']
        self.mock.reset()
        self.full = W.FullApp(document(live=self.live), api_key=BOOT, settings_kw=self.settings_kw)
        self.http = self.full.http
        _, self.user = self.full.mint('inference', 'cloud', name='user')

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def last(self):
        return self.mock.hits[-1]

    def hits(self) -> int:
        return len(self.mock.hits)


class Bridges(Base):
    def test_image_generation_through_an_image_model(self):
        self.mock.respond_with('gemini.generateContent', 'generate_content_image')
        answer = self.http.post('/v1/images/generations', headers=bearer(self.user), json={
            'model': 'gemini-2.5-flash-image@gem', 'prompt': 'A small red square', 'size': '1536x1024',
            'quality': 'high', 'response_format': 'b64_json'})
        self.assertEqual(answer.status_code, 200, answer.text[:400])
        data = answer.json()
        self.assertEqual(base64.b64decode(data['data'][0]['b64_json']), PNG)
        self.assertEqual(data['data'][0]['revised_prompt'], 'Here is a small red square.')
        self.assertEqual(data['usage']['output_tokens'], 1297)
        hit = self.last()
        self.assertEqual(hit.path, '/v1beta/models/gemini-2.5-flash-image:generateContent')
        self.assertEqual(hit.credential, '<sentinel:gemini>')
        self.assertEqual(hit.json(), {'contents': [{'role': 'user', 'parts': [{'text': 'A small red square'}]}],
                                      'generationConfig': {'responseModalities': ['TEXT', 'IMAGE'],
                                                           'imageConfig': {'aspectRatio': '3:2'}}})
        self.assertIn('quality', answer.headers.get('x-titanos-dropped', ''))

    def test_image_edit_through_an_image_model(self):
        self.mock.respond_with('gemini.generateContent', 'generate_content_image')
        answer = self.http.post('/v1/images/edits', headers=bearer(self.user),
                                data={'model': 'gemini-2.5-flash-image@gem', 'prompt': 'Make it blue'},
                                files={'image': ('in.png', PNG, 'image/png')})
        self.assertEqual(answer.status_code, 200, answer.text[:400])
        self.assertEqual(base64.b64decode(answer.json()['data'][0]['b64_json']), PNG)
        parts = self.last().json()['contents'][0]['parts']
        self.assertEqual(parts[0], {'text': 'Make it blue'})
        self.assertEqual(parts[1], {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(PNG).decode()}})

    def test_speech_through_a_tts_model(self):
        self.mock.respond_with('gemini.generateContent', 'generate_content_tts')
        answer = self.http.post('/v1/audio/speech', headers=bearer(self.user), json={
            'model': 'gemini-2.5-flash-preview-tts@gem', 'input': 'Have a wonderful day!', 'voice': 'puck',
            'instructions': 'Say cheerfully', 'response_format': 'mp3'})
        self.assertEqual(answer.status_code, 200, answer.text[:400])
        self.assertEqual(answer.headers['content-type'], 'audio/wav')
        with wave.open(io.BytesIO(answer.content), 'rb') as reader:
            self.assertEqual((reader.getframerate(), reader.getnchannels(), reader.getsampwidth()), (24000, 1, 2))
            self.assertEqual(reader.readframes(4), b'\x00\x00\x01\x00\x02\x00\x03\x00')
        self.assertEqual(self.last().json(), {
            'contents': [{'role': 'user', 'parts': [{'text': 'Say cheerfully: Have a wonderful day!'}]}],
            'generationConfig': {'responseModalities': ['AUDIO'], 'speechConfig': {
                'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Puck'}}}}})
        self.assertIn('response_format', answer.headers.get('x-titanos-dropped', ''))
        self.mock.respond_with('gemini.generateContent', 'generate_content_tts')
        pcm = self.http.post('/v1/audio/speech', headers=bearer(self.user), json={
            'model': 'gemini-2.5-flash-preview-tts@gem', 'input': 'Hi', 'voice': 'Kore', 'response_format': 'pcm'})
        self.assertEqual((pcm.status_code, pcm.content), (200, b'\x00\x00\x01\x00\x02\x00\x03\x00'))

    def test_transcription_and_translation_through_audio_understanding(self):
        audio = wav_bytes(0.5)
        for path, fmt, check in (('/v1/audio/transcriptions', 'json', lambda r: r.json()['text']),
                                 ('/v1/audio/transcriptions', 'text', lambda r: r.text),
                                 ('/v1/audio/transcriptions', 'verbose_json', lambda r: r.json()['text']),
                                 ('/v1/audio/translations', 'json', lambda r: r.json()['text'])):
            with self.subTest(path=path, fmt=fmt):
                self.mock.respond_with('gemini.generateContent', 'generate_content_transcript')
                answer = self.http.post(path, headers=bearer(self.user),
                                        data={'model': 'gemini-2.5-flash@gem', 'response_format': fmt,
                                              'language': 'en', 'prompt': 'Titanos', 'temperature': '0'},
                                        files={'file': ('clip.wav', audio, 'audio/wav')})
                self.assertEqual(answer.status_code, 200, answer.text[:400])
                self.assertEqual(check(answer), 'Hello from the mock transcript.')
                sent = self.last().json()
                instruction, media = sent['contents'][0]['parts']
                self.assertEqual(media['inlineData']['mimeType'], 'audio/wav')
                self.assertEqual(base64.b64decode(media['inlineData']['data']), audio)
                self.assertIn('Titanos', instruction['text'])
                self.assertIn('English' if 'translations' in path else 'transcript', instruction['text'])
                self.assertEqual(sent['generationConfig'], {'temperature': 0.0})
                if fmt == 'verbose_json':
                    self.assertEqual(answer.json()['duration'], 0.5)

    def test_refusals_before_egress(self):
        before = self.hits()
        cases = (('/v1/images/generations', {'model': 'gemini-2.5-flash-image@gem', 'prompt': 'x', 'n': 2}, None),
                 ('/v1/images/generations', {'model': 'gemini-2.5-flash-image@gem', 'prompt': 'x',
                                             'response_format': 'url'}, None),
                 ('/v1/audio/speech', {'model': 'gemini-2.5-flash-preview-tts@gem', 'input': 'x',
                                       'stream_format': 'sse'}, None),
                 ('/v1/chat/completions', {'model': 'gemini-2.5-flash-preview-tts@gem',
                                           'messages': [{'role': 'user', 'content': 'hi'}]}, None),
                 ('/v1/audio/transcriptions', {'model': 'gemini-2.5-flash@gem', 'response_format': 'srt'},
                  {'file': ('clip.wav', wav_bytes(0.1), 'audio/wav')}),
                 ('/v1/images/edits', {'model': 'gemini-2.5-flash-image@gem', 'prompt': 'x'},
                  {'image': ('a.png', PNG, 'image/png'), 'mask': ('m.png', PNG, 'image/png')}))
        for path, body, files in cases:
            with self.subTest(path=path, body=body):
                if files:
                    answer = self.http.post(path, headers=bearer(self.user), data=body, files=files)
                else:
                    answer = self.http.post(path, headers=bearer(self.user), json=body)
                self.assertEqual(answer.status_code, 400, answer.text[:300])
                self.assertEqual(answer.json()['error']['code'], 'broker_capability_unavailable')
        # a model of the wrong family is not even a candidate for the surface (capability filter, zero hits)
        wrong = self.http.post('/v1/audio/speech', headers=bearer(self.user),
                               json={'model': 'gemini-2.5-flash@gem', 'input': 'x'})
        self.assertEqual(wrong.status_code, 400, wrong.text[:300])
        self.assertEqual(self.hits(), before)

    def test_chat_answer_carries_generated_images(self):
        self.mock.respond_with('gemini.generateContent', 'generate_content_image')
        answer = self.http.post('/v1/chat/completions', headers=bearer(self.user), json={
            'model': 'gemini-2.5-flash-image@gem', 'messages': [{'role': 'user', 'content': 'draw'}]})
        self.assertEqual(answer.status_code, 200, answer.text[:400])
        message = answer.json()['choices'][0]['message']
        self.assertEqual(message['content'], 'Here is a small red square.')
        self.assertEqual(message['images'][0]['image_url']['url'],
                         'data:image/png;base64,' + base64.b64encode(PNG).decode())

    def test_large_first_image_frame_streams_through(self):
        big = base64.b64encode(PNG + b'\x00' * (1536 * 1024)).decode()  # one data line well over 1 MiB
        self.mock.extra_vars['image_b64'] = big
        self.mock.respond_with('gemini.streamGenerateContent', 'stream_generate_content_image')
        body = {'contents': [{'role': 'user', 'parts': [{'text': 'A small red square'}]}],
                'generationConfig': {'responseModalities': ['TEXT', 'IMAGE']}}
        answer = self.http.post('/v1beta/models/gemini-2.5-flash-image@gem:streamGenerateContent?alt=sse',
                                headers={'x-goog-api-key': self.user}, json=body)
        self.assertEqual(answer.status_code, 200, answer.text[:400])
        self.assertIn(big.encode(), answer.content)
        self.assertIn(b'"finishReason":"STOP"', answer.content)
        self.mock.respond_with('gemini.streamGenerateContent', 'stream_generate_content_image')
        chat = self.http.post('/v1/chat/completions', headers=bearer(self.user), json={
            'model': 'gemini-2.5-flash-image@gem', 'stream': True,
            'messages': [{'role': 'user', 'content': 'draw'}]})
        self.assertEqual(chat.status_code, 200, chat.text[:400])
        self.assertIn('Here is the square.', chat.text)
        self.assertNotIn('upstream_bad_response', chat.text)

    def test_budgeted_snake_case_body_gets_one_cap(self):
        _, budgeted = self.full.mint('inference', 'cloud', name='budgeted', budget={'usd_per_day': '1000'})
        body = {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}], 'generation_config': {'temperature': 0.5}}
        answer = self.http.post('/v1beta/models/gemini-2.5-flash@gem:generateContent',
                                headers={'x-goog-api-key': budgeted}, json=body)
        self.assertEqual(answer.status_code, 200, answer.text[:400])
        sent = self.last().json()
        self.assertNotIn('generationConfig', sent)
        self.assertEqual(sent['generation_config']['temperature'], 0.5)
        self.assertEqual(sent['generation_config']['max_output_tokens'],
                         self.full.settings.broker_default_max_output_tokens)
        own = dict(body, generation_config={'max_output_tokens': 9})
        self.http.post('/v1beta/models/gemini-2.5-flash@gem:generateContent', headers={'x-goog-api-key': budgeted},
                       json=own)
        self.assertEqual(self.last().json()['generation_config'], {'max_output_tokens': 9})


class MediaCap(Base):
    settings_kw = {'broker_max_upstream_json_bytes': 4096}

    def test_generate_content_answers_use_the_media_cap(self):
        big = base64.b64encode(PNG + b'\x00' * 8192).decode()  # a JSON answer above the JSON cap
        for path, body, headers in (
                ('/v1beta/models/gemini-2.5-flash-image@gem:generateContent',
                 {'contents': [{'role': 'user', 'parts': [{'text': 'draw'}]}]}, {'x-goog-api-key': BOOT}),
                ('/v1/images/generations', {'model': 'gemini-2.5-flash-image@gem', 'prompt': 'draw'}, bearer(BOOT))):
            with self.subTest(path=path):
                self.mock.extra_vars['image_b64'] = big
                self.mock.respond_with('gemini.generateContent', 'generate_content_image')
                answer = self.http.post(path, headers=headers, json=body)
                self.assertEqual(answer.status_code, 200, answer.text[:300])
                self.assertIn(big[:64], answer.text)


class LiveCatalog(Base):
    live = True

    def test_capabilities_and_metadata_per_model(self):
        for _ in range(4):
            self.mock.respond_with('gemini.models', 'models_families')
        refreshed = self.http.get('/api/broker/catalog?refresh=true', headers=bearer(BOOT))
        self.assertEqual(refreshed.status_code, 200, refreshed.text[:300])
        listing = self.http.get('/v1/models', headers=bearer(BOOT)).json()['data']
        caps = {row['id']: set(row['titanos']['capabilities']) for row in listing if row.get('remote')}
        self.assertEqual(caps['gemini-2.5-flash-preview-tts@gem'], {'chat', 'speech'})
        self.assertIn('images', caps['gemini-2.5-flash-image@gem'])
        self.assertEqual(caps['gemini-embedding-001@gem'], {'embeddings'})
        self.assertEqual(caps['imagen-4.0-generate-001@gem'], set())
        self.assertNotIn('speech', caps['gemini-2.5-flash@gem'])
        self.assertNotIn('embeddings', caps['gemini-2.5-flash@gem'])
        self.assertEqual({row['id']: row['titanos'].get('context_length') for row in listing
                          if row['id'] == 'gemini-2.5-flash@gem'}, {'gemini-2.5-flash@gem': 1048576})
        models = {row['baseModelId']: row for row in
                  self.http.get('/v1beta/models', headers={'x-goog-api-key': BOOT}).json()['models']}
        self.assertEqual(models['imagen-4.0-generate-001@gem']['supportedGenerationMethods'], ['predict'])
        flash = models['gemini-2.5-flash@gem']
        self.assertEqual((flash['inputTokenLimit'], flash['outputTokenLimit'], flash['thinking']),
                         (1048576, 65536, True))
        self.assertIn('batchGenerateContent', flash['supportedGenerationMethods'])


class BareIds(Base):
    def test_google_genai_bare_model_ids(self):
        answer = self.http.post('/v1beta/models/gemini-2.5-flash:generateContent',
                                json={'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                                headers={'x-goog-api-key': self.user})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.headers['x-titanos-served-model'], 'gemini-2.5-flash@gem')
        self.assertIn('/models/gemini-2.5-flash:generateContent', self.last().path)
        counted = self.http.post('/v1beta/models/gemini-2.5-flash:countTokens',
                                 json={'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                                 headers={'x-goog-api-key': self.user})
        self.assertEqual(counted.status_code, 200, counted.text)
        listing = self.http.get('/v1/models', headers=bearer(BOOT)).json()['data']
        hints = {row['id']: row['titanos'].get('model_capabilities') for row in listing if row.get('remote')}
        self.assertIn('chat', hints['gemini-2.5-flash@gem'])


class NativeRows(Base):
    def test_download_interactions_and_file_search(self):
        _, owner = self.full.mint('native', 'cloud', 'cloud:store', name='owner')
        _, other = self.full.mint('native', 'cloud', 'cloud:store', name='other')
        download = self.http.get('/download/v1beta/files/batch0001:download?alt=media',
                                 headers={'x-goog-api-key': BOOT})
        self.assertEqual(download.status_code, 200, download.text[:300])
        self.assertEqual(download.content, b'{"key":"r1","response":{}}\n{"key":"r2","response":{}}\n')
        self.assertIn('attachment', download.headers.get('content-disposition', ''))
        self.assertEqual(self.last().credential, '<sentinel:gemini>')
        self.assertEqual(self.last().query_params, {'alt': 'media'})
        self.assertEqual(self.http.get('/download/v1beta/files/batch0001:download?alt=media',
                                       headers={'x-goog-api-key': owner}).status_code, 404)  # not bound to owner
        created = self.http.post('/v1beta/interactions', headers={'x-goog-api-key': owner},
                                 json={'model': 'gemini-2.5-flash', 'input': 'hi', 'store': True})
        self.assertEqual(created.status_code, 200, created.text[:300])
        interaction = created.json()['id']
        self.assertEqual(self.http.get('/v1beta/interactions/' + interaction,
                                       headers={'x-goog-api-key': owner}).json()['id'], interaction)
        self.assertEqual(self.http.get('/v1beta/interactions/' + interaction,
                                       headers={'x-goog-api-key': other}).status_code, 404)
        streamed = self.http.post('/native/gem/v1beta/interactions', headers={'x-goog-api-key': owner},
                                  json={'model': 'gemini-2.5-flash', 'input': 'hi', 'stream': True})
        self.assertEqual(streamed.status_code, 200, streamed.text[:300])
        self.assertIn('event: interaction.completed', streamed.text)
        store = self.http.post('/v1beta/fileSearchStores', headers={'x-goog-api-key': owner},
                               json={'displayName': 'docs'})
        self.assertEqual(store.status_code, 200, store.text[:300])
        name = store.json()['name']
        imported = self.http.post(f'/v1beta/{name}:importFile', headers={'x-goog-api-key': owner},
                                  json={'fileName': 'files/abc'})
        self.assertEqual(imported.status_code, 200, imported.text[:300])
        self.assertIn('/operations/', imported.json()['name'])
        self.assertEqual(self.http.post(f'/v1beta/{name}:importFile', headers={'x-goog-api-key': other},
                                        json={'fileName': 'files/abc'}).status_code, 404)

    def test_account_wide_rows_need_admin_and_redirects_are_refused(self):
        _, plain = self.full.mint('native', 'cloud', 'cloud:store', name='plain')
        before = self.hits()
        for method, path, body in (('GET', '/v1beta/webhooks', None), ('POST', '/v1alpha/auth_tokens', {'uses': 1}),
                                   ('GET', '/v1beta/fileSearchStores', None)):
            with self.subTest(path=path):
                answer = self.http.request(method, path, headers={'x-goog-api-key': plain}, json=body)
                error = answer.json()['error']  # the Gemini google.rpc.Status envelope
                self.assertEqual((answer.status_code, error['status']), (403, 'PERMISSION_DENIED'))
                self.assertIn('broker:admin', error['message'])
        self.assertEqual(self.hits(), before)
        token = self.http.post('/v1alpha/auth_tokens', headers={'x-goog-api-key': BOOT}, json={'uses': 1})
        self.assertEqual((token.status_code, token.json()['name']), (200, 'auth_tokens/mock-ephemeral-0001'))
        self.assertEqual(self.http.get('/v1beta/webhooks', headers={'x-goog-api-key': BOOT}).status_code, 200)
        moved = self.http.get('/v1beta/files/abc:download?alt=media', headers={'x-goog-api-key': BOOT})
        self.assertEqual(moved.status_code, 502)
        self.assertIn('redirect', moved.json()['error']['message'])


class Jobs(Base):
    def submit(self, body: dict, key: str) -> dict:
        _, token = self.full.mint('jobs', 'inference', 'cloud', name='jobs-' + key)
        accepted = self.http.post('/api/jobs', json=body, headers={**bearer(token), 'Idempotency-Key': key})
        self.assertEqual(accepted.status_code, 202, accepted.text[:300])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            record = self.http.get('/api/jobs/' + accepted.json()['id'], headers=bearer(token)).json()
            if record['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
                return record
            time.sleep(0.05)
        self.fail('job did not finish')

    def test_speech_and_image_jobs_on_gemini_models(self):
        self.mock.respond_with('gemini.generateContent', 'generate_content_tts')
        record = self.submit({'task': 'speech', 'model': 'gemini-2.5-flash-preview-tts@gem', 'input': 'Hello there.',
                              'voice': 'Kore'}, 'speech-1')
        self.assertEqual(record['status'], 'completed', record)
        name = record['result']['url'].rsplit('/', 1)[-1]
        self.assertTrue(self.http.get('/v1/files/' + name, headers=bearer(BOOT)).content.startswith(b'RIFF'))
        self.assertEqual(self.last().json()['generationConfig']['speechConfig']['voiceConfig'],
                         {'prebuiltVoiceConfig': {'voiceName': 'Kore'}})
        self.mock.respond_with('gemini.generateContent', 'generate_content_image')
        record = self.submit({'task': 'image', 'model': 'gemini-2.5-flash-image@gem', 'prompt': 'a cat'}, 'image-1')
        self.assertEqual(record['status'], 'completed', record)
        name = record['result']['data'][0]['url'].rsplit('/', 1)[-1]
        self.assertEqual(self.http.get('/v1/files/' + name, headers=bearer(BOOT)).content, PNG)


CELLS = {
    'openai.images.generations|GEM|T': 'Bridges.test_image_generation_through_an_image_model',
    'openai.images.edits|GEM|T': 'Bridges.test_image_edit_through_an_image_model',
    'openai.audio.speech|GEM|T': 'Bridges.test_speech_through_a_tts_model',
    'openai.audio.transcriptions|GEM|T': 'Bridges.test_transcription_and_translation_through_audio_understanding',
    'openai.audio.translations|GEM|T': 'Bridges.test_transcription_and_translation_through_audio_understanding',
}


if __name__ == '__main__':
    A.main(CELLS)
