"""The OpenAI-compatible family beyond the reference OpenAI row (xAI, Groq, Together, Mistral, DeepSeek, Fireworks,
Perplexity, Cerebras, OpenRouter, Azure OpenAI v1) over the integrated
app (create_app: Guard, KeylessHostGuard, BrokerDispatch) against broker_mocks.compat_mock.CompatMock, whose answers
replay fixtures transcribed from each provider's public reference (synthetic values). The netguard is installed and
every integrated test ends with zero non-loopback connection attempts; synthetic sentinel credentials only.

- typed surfaces the rows now serve natively: Groq text-to-speech and Responses, Together speech, transcription and
  rerank, Mistral Voxtral transcription (audio seconds metered), OpenRouter embeddings (usage.cost passed through),
  xAI and Fireworks Responses (store forced false unless the caller may store; the stored xAI response is retrieved
  through its handle), the rerank surface on Together (results shape) and Fireworks (data shape), and a local rerank model still answered by the local route through the active broker;
- images: xAI and Together answer their own image URLs, so a url or default request is passed through (never
  re-hosted, never refused); Together's base64 is spelled 'base64' (b64_json mapped, noted);
- arguments: xAI reasoning models never receive presence_penalty, frequency_penalty or stop (dropped, named in
  X-Titanos-Dropped; a non-reasoning model keeps them); local-engine knobs (require_loaded, VAD fields) never reach a
  provider; repeat_penalty is renamed repetition_penalty where the provider takes it (Together) and dropped with
  top_k and min_p where it would be a 400 (Groq); a translated reasoning effort reaches only the models of the row's
  reasoning_effort_globs (Cerebras gpt-oss yes, llama no);
- DeepSeek thinking mode: an Anthropic Messages tool loop translated onto DeepSeek sends each assistant turn's thinking
  back as reasoning_content (the mock answers 400 otherwise, as DeepSeek does); Perplexity citations and search_results
  pass through; Perplexity Search and Groq Files through the native lane;
- asset jobs (Studio's /api/jobs): a Together image job asks for base64 PNG (Together's spelling, its JPEG default
  overridden) and yields an owner-bound .png; a Groq speech job yields a .wav;
- errors: Groq 498 (flex capacity) is 429 upstream_rate_limited.
Plus in-process unit cases for the profile data (extras, native rows, env names), rewrite_body,
accepts_reasoning_effort and the DeepSeek reasoning_content round trip encoder.
"""
from __future__ import annotations

import base64
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import netguard  # noqa: E402
from broker_mocks.compat_mock import CompatMock  # noqa: E402

from titanos.broker import native as native_lane  # noqa: E402
from titanos.broker.adapters.openai_compat import (accepts_reasoning_effort, is_reasoning_model,  # noqa: E402
                                                   rewrite_body, stores_responses)
from titanos.broker.profiles import PROFILES, TITANOS_ONLY_FIELDS, env_names  # noqa: E402
from titanos.broker.surfaces import SURFACES  # noqa: E402
from titanos.broker.translate import openai_chat  # noqa: E402
from titanos.broker.translate.ir import IRMessage, IRRequest, Text, Thinking, ToolResult, ToolUse  # noqa: E402

BOOT = 'compat-check-bootstrap-0001'
MOCKS: dict = {}
# upstream id -> (mock variant, profile, models, sentinel name)
UPSTREAMS = {
    'groq': ('groq', 'groq', ['openai/gpt-oss-20b', 'llama-3.3-70b-versatile', 'playai-tts'], 'groq'),
    'tog': ('together', 'together', ['black-forest-labs/FLUX.1-schnell', 'Salesforce/Llama-Rank-V1',
                                     'meta-llama/Llama-3.3-70B-Instruct-Turbo', 'cartesia/sonic',
                                     'openai/whisper-large-v3'], 'together'),
    'fw': ('fireworks', 'fireworks', ['accounts/fireworks/models/qwen3-reranker-8b',
                                      'accounts/fireworks/models/gpt-oss-20b'], 'fireworks'),
    'ds': ('deepseek', 'deepseek', ['deepseek-reasoner'], 'deepseek'),
    'pplx': ('perplexity', 'perplexity', ['sonar'], 'perplexity'),
    'mis': ('mistral', 'mistral', ['voxtral-mini-latest', 'mistral-small-latest'], 'mistral'),
    'xai': ('xai', 'xai', ['grok-4', 'grok-4-fast-non-reasoning'], 'xai'),
    'or': ('openrouter', 'openrouter', ['openai/text-embedding-3-small'], 'openrouter'),
    'cer': ('cerebras', 'cerebras', ['gpt-oss-120b', 'llama3.1-8b'], 'cerebras'),
}
CHAT = [{'role': 'user', 'content': 'hi'}]
WAV = W.WAV


def setUpModule():
    for uid, (variant, *_rest) in UPSTREAMS.items():
        MOCKS[uid] = CompatMock(variant)
        MOCKS[uid].start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def document() -> dict:
    upstreams = {}
    for uid, (_variant, profile, models, _sentinel) in UPSTREAMS.items():
        upstreams[uid] = {'profile': profile, 'base_url': MOCKS[uid].api_base, 'credential': uid + '-key',
                          'enabled': True, 'models': models, 'data_policy': {'allow_store': True}}
    return {'policy': {'cloud_policy': 'explicit'}, 'upstreams': upstreams}


def bearer(token: str) -> dict:
    return {'Authorization': 'Bearer ' + token}


class Base(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
        creds = {uid: sentinel for uid, (_v, _p, _m, sentinel) in UPSTREAMS.items()}
        self.full = W.FullApp(document(), api_key=BOOT, creds=creds, models={'local-rerank': 'rerank'})
        self.http = self.full.http
        self.store_key = self.full.mint('inference', 'cloud', 'cloud:store', name='store')[1]
        self.plain_key = self.full.mint('inference', 'cloud', name='plain')[1]

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def post(self, path, body, token=None, **kw):
        return self.http.post(path, json=body, headers=bearer(token or self.store_key), **kw)

    def hit(self, uid):
        return MOCKS[uid].hits[-1]


class TypedSurfaces(Base):
    def test_groq_speech_and_responses_are_native(self):
        speech = self.post('/v1/audio/speech', {'model': 'playai-tts@groq', 'input': 'Hello', 'voice': 'Fritz-PlayAI',
                                                'response_format': 'wav'})
        self.assertEqual(speech.status_code, 200, speech.text[:300])
        self.assertTrue(speech.content.startswith(b'RIFF'))
        hit = self.hit('groq')
        self.assertEqual((hit.path, hit.credential), ('/openai/v1/audio/speech', 'Bearer <sentinel:groq>'))
        self.assertEqual(hit.json()['model'], 'playai-tts')
        answer = self.post('/v1/responses', {'model': 'openai/gpt-oss-20b@groq', 'input': 'Hello!',
                                             'reasoning': {'effort': 'low'}})
        self.assertEqual(answer.status_code, 200, answer.text[:300])
        self.assertEqual(answer.headers['x-titanos-translation'], 'passthrough')
        self.assertEqual(answer.json()['output'][-1]['content'][0]['text'], 'Hello! How can I help?')
        sent = self.hit('groq').json()
        self.assertEqual((sent['model'], sent['reasoning']), ('openai/gpt-oss-20b', {'effort': 'low'}))
        self.assertNotIn('store', sent)  # Groq stores nothing: the field is never forced onto it

    def test_together_speech_transcription_and_rerank(self):
        speech = self.post('/v1/audio/speech', {'model': 'cartesia/sonic@tog', 'input': 'Hello', 'voice': 'v1'})
        self.assertEqual(speech.status_code, 200, speech.text[:300])
        self.assertEqual(speech.headers['content-type'], 'audio/mpeg')
        text = self.http.post('/v1/audio/transcriptions', data={'model': 'openai/whisper-large-v3@tog'},
                              files={'file': ('a.wav', WAV, 'audio/wav')}, headers=bearer(self.store_key))
        self.assertEqual((text.status_code, text.json()), (200, {'text': 'Hello world.'}))
        self.assertEqual(self.hit('tog').path, '/v1/audio/transcriptions')
        ranked = self.post('/v1/rerank', {'model': 'Salesforce/Llama-Rank-V1@tog', 'query': 'gpu',
                                          'documents': ['bananas', 'a gpu'], 'top_n': 2, 'rank_fields': ['text']})
        self.assertEqual(ranked.status_code, 200, ranked.text[:300])
        self.assertEqual([r['index'] for r in ranked.json()['results']], [1, 0])
        self.assertEqual(ranked.headers['x-titanos-served-by'], 'cloud:tog')
        sent = self.hit('tog').json()
        self.assertEqual(sent, {'model': 'Salesforce/Llama-Rank-V1', 'query': 'gpu', 'documents': ['bananas', 'a gpu'],
                                'top_n': 2, 'rank_fields': ['text']})  # every argument forwarded

    def test_fireworks_rerank_and_responses_store_rule(self):
        ranked = self.post('/v1/rerank', {'model': 'accounts/fireworks/models/qwen3-reranker-8b@fw', 'query': 'ML',
                                          'documents': ['Machine learning is a subset of AI.', 'Sunny.'],
                                          'task': 'Rank by relevance'})
        self.assertEqual(ranked.status_code, 200, ranked.text[:300])
        self.assertEqual(ranked.json()['data'][0]['relevance_score'], 0.95)  # the provider's own shape
        self.assertEqual(self.hit('fw').path, '/inference/v1/rerank')
        self.assertEqual(self.hit('fw').json()['task'], 'Rank by relevance')
        plain = self.post('/v1/responses', {'model': 'accounts/fireworks/models/gpt-oss-20b@fw', 'input': 'Hi'},
                          token=self.plain_key)
        self.assertEqual(plain.status_code, 200, plain.text[:300])
        self.assertIs(self.hit('fw').json()['store'], False)  # a key without cloud:store never stores
        stored = self.post('/v1/responses', {'model': 'accounts/fireworks/models/gpt-oss-20b@fw', 'input': 'Hi'})
        self.assertEqual(stored.status_code, 200, stored.text[:300])
        self.assertNotIn('store', self.hit('fw').json())

    def test_xai_responses_store_and_handle_retrieval(self):
        created = self.post('/v1/responses', {'model': 'grok-4@xai', 'input': 'Hi'})
        self.assertEqual(created.status_code, 200, created.text[:300])
        response_id = created.json()['id']
        again = self.http.get(f'/v1/responses/{response_id}', headers=bearer(self.store_key))
        self.assertEqual(again.status_code, 200, again.text[:300])
        self.assertEqual(self.hit('xai').path, f'/v1/responses/{response_id}')
        self.post('/v1/responses', {'model': 'grok-4@xai', 'input': 'Hi'}, token=self.plain_key)
        self.assertIs(self.hit('xai').json()['store'], False)

    def test_local_rerank_model_still_reaches_the_local_route(self):
        answer = self.post('/v1/rerank', {'model': 'local-rerank', 'query': 'q', 'documents': ['a', 'b']},
                           token=BOOT)
        self.assertEqual(answer.headers.get('x-titanos-served-by'), 'local:engine')
        self.assertTrue(any(call[1] == 'rerank' for call in self.full.engine.calls), self.full.engine.calls)
        self.assertEqual(sum(len(m.hits) for m in MOCKS.values()), 0)

    def test_mistral_voxtral_transcription_and_openrouter_embeddings(self):
        text = self.http.post('/v1/audio/transcriptions', data={'model': 'voxtral-mini-latest@mis', 'language': 'en'},
                              files={'file': ('a.wav', WAV, 'audio/wav')}, headers=bearer(self.store_key))
        self.assertEqual(text.status_code, 200, text.text[:300])
        self.assertEqual(text.json()['usage']['prompt_audio_seconds'], 2)
        self.assertEqual(self.hit('mis').path, '/v1/audio/transcriptions')
        vectors = self.post('/v1/embeddings', {'model': 'openai/text-embedding-3-small@or', 'input': 'hi',
                                               'provider': {'order': ['openai']}})
        self.assertEqual(vectors.status_code, 200, vectors.text[:300])
        self.assertEqual(vectors.json()['usage']['cost'], 0.0001)
        self.assertEqual(self.hit('or').path, '/api/v1/embeddings')
        self.assertEqual(self.hit('or').json()['provider'], {'order': ['openai']})


class Images(Base):
    def test_url_native_providers_answer_their_own_urls(self):
        url = self.post('/v1/images/generations', {'model': 'black-forest-labs/FLUX.1-schnell@tog', 'prompt': 'cat'})
        self.assertEqual(url.status_code, 200, url.text[:300])
        self.assertEqual(url.json()['data'][0]['url'], 'https://example.invalid/together/mock-0001.jpg')
        self.assertNotIn('response_format', self.hit('tog').json())
        b64 = self.post('/v1/images/generations', {'model': 'black-forest-labs/FLUX.1-schnell@tog', 'prompt': 'cat',
                                                   'response_format': 'b64_json', 'steps': 4})
        self.assertEqual(b64.status_code, 200, b64.text[:300])
        self.assertEqual(base64.b64decode(b64.json()['data'][0]['b64_json'])[:3], b'\xff\xd8\xff')
        sent = self.hit('tog').json()
        self.assertEqual((sent['response_format'], sent['steps']), ('base64', 4))
        xai = self.post('/v1/images/generations', {'model': 'grok-4@xai', 'prompt': 'cat', 'response_format': 'url'})
        self.assertEqual(xai.status_code, 200, xai.text[:300])
        self.assertTrue(xai.json()['data'][0]['url'].startswith('https://example.invalid/xai/'))


class Arguments(Base):
    def test_xai_reasoning_models_never_get_penalties_or_stop(self):
        answer = self.post('/v1/chat/completions', {'model': 'grok-4@xai', 'messages': CHAT, 'presence_penalty': 0.5,
                                                    'frequency_penalty': 0.5, 'stop': ['END'],
                                                    'search_parameters': {'mode': 'auto'}})
        self.assertEqual(answer.status_code, 200, answer.text[:300])
        dropped = answer.headers.get('x-titanos-dropped', '')
        for name in ('presence_penalty', 'frequency_penalty', 'stop'):
            self.assertIn(name, dropped)
            self.assertNotIn(name, self.hit('xai').json())
        self.assertEqual(self.hit('xai').json()['search_parameters'], {'mode': 'auto'})
        keep = self.post('/v1/chat/completions', {'model': 'grok-4-fast-non-reasoning@xai', 'messages': CHAT,
                                                  'stop': ['END']})
        self.assertEqual(keep.status_code, 200, keep.text[:300])
        self.assertEqual(self.hit('xai').json()['stop'], ['END'])

    def test_local_engine_knobs_are_mapped_or_dropped(self):
        knobs = {'repeat_penalty': 1.1, 'top_k': 40, 'min_p': 0.05, 'require_loaded': True, 'include_provenance': True}
        groq = self.post('/v1/chat/completions', dict({'model': 'llama-3.3-70b-versatile@groq', 'messages': CHAT},
                                                      **knobs))
        self.assertEqual(groq.status_code, 200, groq.text[:300])
        self.assertFalse(set(knobs) & set(self.hit('groq').json()), self.hit('groq').json())
        for name in ('repeat_penalty', 'top_k', 'min_p'):
            self.assertIn(name, groq.headers.get('x-titanos-dropped', ''))
        tog = self.post('/v1/chat/completions', dict({'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo@tog',
                                                      'messages': CHAT}, **knobs))
        self.assertEqual(tog.status_code, 200, tog.text[:300])
        sent = self.hit('tog').json()
        self.assertEqual((sent['repetition_penalty'], sent['top_k'], sent['min_p']), (1.1, 40, 0.05))
        self.assertFalse({'repeat_penalty', 'require_loaded', 'include_provenance'} & set(sent))

    def test_translated_reasoning_effort_only_reaches_models_that_take_it(self):
        for model, expected in (('gpt-oss-120b@cer', 'high'), ('llama3.1-8b@cer', None)):
            answer = self.post('/v1/responses', {'model': model, 'input': 'Hi', 'reasoning': {'effort': 'high'}})
            self.assertEqual(answer.status_code, 200, answer.text[:300])
            self.assertNotEqual(answer.headers['x-titanos-translation'], 'passthrough')  # Responses onto chat
            self.assertEqual(self.hit('cer').json().get('reasoning_effort'), expected, model)

    def test_deepseek_tool_loop_sends_reasoning_content_back(self):
        answer = self.http.post('/v1/messages', json={
            'model': 'deepseek-reasoner@ds', 'max_tokens': 64,
            'tools': [{'name': 'f', 'description': 'd', 'input_schema': {'type': 'object', 'properties': {}}}],
            'messages': [{'role': 'user', 'content': 'call f'},
                         {'role': 'assistant', 'content': [
                             {'type': 'thinking', 'thinking': 'I should call f.', 'signature': 'sig-mock'},
                             {'type': 'tool_use', 'id': 'toolu_1', 'name': 'f', 'input': {}}]},
                         {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1',
                                                       'content': '42'}]}]},
            headers={'x-api-key': self.store_key, 'anthropic-version': '2023-06-01'})
        self.assertEqual(answer.status_code, 200, answer.text[:300])
        assistant = [m for m in self.hit('ds').json()['messages'] if m['role'] == 'assistant']
        self.assertEqual(assistant[0]['reasoning_content'], 'I should call f.')
        self.assertEqual(answer.json()['content'][0]['type'], 'thinking')

    def test_perplexity_extensions_pass_through_and_native_search(self):
        chat = self.post('/v1/chat/completions', {'model': 'sonar@pplx', 'messages': CHAT,
                                                  'search_domain_filter': ['example.invalid'],
                                                  'web_search_options': {'search_context_size': 'low'}})
        self.assertEqual(chat.status_code, 200, chat.text[:300])
        self.assertEqual(chat.json()['citations'], ['https://example.invalid/pplx/one'])
        self.assertEqual(chat.json()['usage']['cost']['total_cost'], 0.005)
        self.assertEqual(self.hit('pplx').json()['search_domain_filter'], ['example.invalid'])
        token = self.full.mint('native', 'cloud', name='native')[1]
        search = self.http.post('/native/pplx/search', json={'query': 'gpu', 'max_results': 1}, headers=bearer(token))
        self.assertEqual(search.status_code, 200, search.text[:300])
        self.assertEqual(self.hit('pplx').path, '/search')

    def test_groq_files_through_the_native_lane(self):
        token = self.full.mint('native', 'cloud', 'cloud:store', name='files')[1]
        upload = self.http.post('/native/groq/files', files={'file': ('b.jsonl', b'{"custom_id":"a"}\n',
                                                                      'application/jsonl')},
                                data={'purpose': 'batch'}, headers=bearer(token))
        self.assertEqual(upload.status_code, 200, upload.text[:300])
        self.assertEqual((self.hit('groq').path, upload.json()['object']), ('/openai/v1/files', 'file'))


class AssetJobs(Base):
    def wait(self, job_id, token) -> dict:
        import time
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            record = self.http.get(f'/api/jobs/{job_id}', headers=bearer(token)).json()
            if record['status'] in ('completed', 'failed', 'cancelled', 'interrupted'):
                return record
            time.sleep(0.05)
        self.fail('job did not finish')

    def test_together_image_and_groq_speech_jobs(self):
        """Studio's image and speech jobs reach the family rows: Together is asked for base64 PNG (its own spelling and
        its JPEG default overridden), Groq for WAV."""
        token = self.full.mint('jobs', 'inference', 'cloud', name='jobs')[1]
        for body, uid, suffix, magic in (
                ({'task': 'image', 'model': 'black-forest-labs/FLUX.1-schnell@tog', 'prompt': 'a cat'}, 'tog', '.png',
                 b'\x89PNG'),
                ({'task': 'speech', 'model': 'playai-tts@groq', 'input': 'Hello there.', 'voice': 'Fritz-PlayAI'},
                 'groq', '.wav', b'RIFF')):
            with self.subTest(model=body['model']):
                accepted = self.http.post('/api/jobs', json=body,
                                          headers={**bearer(token), 'Idempotency-Key': 'idem-' + uid})
                self.assertEqual(accepted.status_code, 202, accepted.text[:300])
                record = self.wait(accepted.json()['id'], token)
                self.assertEqual(record['status'], 'completed', record)
                url = record['result']['url'] if body['task'] == 'speech' else record['result']['data'][0]['url']
                self.assertTrue(url.endswith(suffix), url)
                fetched = self.http.get('/v1/files/' + url.rsplit('/', 1)[-1], headers=bearer(token))
                self.assertTrue(fetched.content.startswith(magic))
        sent = [h.json() for h in MOCKS['tog'].hits if h.path == '/v1/images/generations'][-1]
        self.assertEqual((sent['response_format'], sent['output_format']), ('base64', 'png'))
        self.assertEqual([h.json() for h in MOCKS['groq'].hits][-1]['response_format'], 'wav')


class Errors(Base):
    def test_groq_498_is_a_rate_limit(self):
        MOCKS['groq'].faults.push('status', status=498, message='Flex tier capacity exceeded', code='capacity_exceeded',
                                  retry_after=2)
        answer = self.post('/v1/chat/completions', {'model': 'llama-3.3-70b-versatile@groq', 'messages': CHAT,
                                                    'service_tier': 'flex'})
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (429, 'upstream_rate_limited'))
        self.assertEqual(answer.headers.get('retry-after'), '2')

class ProfileData(unittest.TestCase):
    """In-process: the profile rows, rewrite_body and the codecs (no socket)."""

    def test_rows_declare_their_surfaces_extras_and_env_names(self):
        rows = {pid: PROFILES[pid] for pid in ('xai', 'groq', 'together', 'mistral', 'deepseek', 'fireworks',
                                               'fireworks_audio', 'perplexity', 'cerebras', 'sambanova',
                                               'openrouter', 'hf_router', 'litellm', 'azure_openai_v1')}
        self.assertIn('openai.audio.speech', rows['groq'].surfaces)
        self.assertLessEqual({'openai.audio.speech', 'openai.audio.transcriptions', 'openai.audio.translations',
                              'openai.rerank'}, set(rows['together'].surfaces))
        self.assertIn('openai.audio.transcriptions', rows['mistral'].surfaces)
        self.assertNotIn('openai.audio.speech', rows['mistral'].surfaces)  # Mistral speech is not the OpenAI shape
        self.assertIn('openai.embeddings', rows['openrouter'].surfaces)
        for pid in ('xai', 'groq', 'openrouter', 'fireworks', 'hf_router', 'litellm', 'azure_openai_v1'):
            self.assertIn('openai.responses', rows[pid].surfaces, pid)
        self.assertNotIn('api-version', str(rows['azure_openai_v1'].surfaces))
        self.assertEqual(rows['azure_openai_v1'].auth, 'azure_api_key')
        self.assertTrue(stores_responses(rows['xai']) and stores_responses(rows['azure_openai_v1']))
        self.assertFalse(stores_responses(rows['groq']))
        self.assertEqual(env_names()['fireworks_audio'], ('FIREWORKS_API_KEY',))
        self.assertIn('image', rows['perplexity'].data_modalities)
        self.assertIn('audio', rows['mistral'].data_modalities)
        self.assertLessEqual({'require_loaded', 'vad_enabled', 'vad_model'}, TITANOS_ONLY_FIELDS)
        self.assertEqual(SURFACES['openai.rerank'].scope, 'inference:plus')

    def test_native_rows_cover_the_endpoints_without_a_typed_surface(self):
        cases = [('mistral', 'POST', '/fim/completions'), ('mistral', 'POST', '/ocr'), ('mistral', 'GET', '/audio/voices'),
                 ('mistral', 'POST', '/batch/jobs'), ('mistral', 'POST', '/conversations'),
                 ('together', 'POST', '/files/upload'), ('together', 'POST', '/batches'),
                 ('groq', 'POST', '/files'), ('groq', 'POST', '/batches'),
                 ('xai', 'POST', '/images/edits'), ('xai', 'POST', '/videos/generations'),
                 ('xai', 'POST', '/tokenize-text'), ('perplexity', 'POST', '/search'),
                 ('fireworks', 'POST', '/inference/v1/workflows/accounts/fireworks/models/flux-1-dev-fp8/text_to_image'),
                 ('deepseek', 'POST', '/beta/completions'), ('openrouter', 'GET', '/key'),
                 ('azure_openai_v1', 'POST', '/files'), ('azure_openai_v1', 'POST', '/batches')]
        for pid, method, rel in cases:
            self.assertIsNotNone(native_lane.match_route(pid, method, rel), (pid, method, rel))
        # typed surfaces left the native allowlists (one lane per endpoint)
        self.assertIsNone(native_lane.match_route('xai', 'POST', '/responses'))
        self.assertIsNone(native_lane.match_route('together', 'POST', '/rerank'))
        # an OpenAI SDK's bare POST /v1/files never becomes ambiguous because Mistral or Together is configured
        self.assertFalse(any(route.bare for route in PROFILES['mistral'].native.routes if route.family == 'files'))

    def test_rewrite_body_rules(self):
        xai = PROFILES['xai']
        out, notes = rewrite_body(xai, None, 'openai.chat', {'model': 'x', 'stop': 'X', 'presence_penalty': 1,
                                                             'top_k': 3}, 'grok-3-mini')
        self.assertEqual(out, {'model': 'grok-3-mini'})
        self.assertEqual(set(notes), {'dropped:stop', 'dropped:presence_penalty', 'dropped:top_k'})
        self.assertTrue(is_reasoning_model(xai, 'grok-4-0709'))
        self.assertFalse(is_reasoning_model(xai, 'grok-4-fast-non-reasoning'))
        out, notes = rewrite_body(PROFILES['together'], None, 'openai.images.generations',
                                  {'model': 'x', 'prompt': 'p', 'response_format': 'b64_json'}, 'flux')
        self.assertEqual((out['response_format'], notes), ('base64', ['mapped:response_format']))
        out, notes = rewrite_body(PROFILES['azure_openai_v1'], None, 'openai.responses', {'model': 'x', 'input': 'i'},
                                  'my-gpt-5-deployment')
        self.assertEqual((out['store'], 'forced:store=false' in notes), (False, True))
        out, _ = rewrite_body(PROFILES['azure_openai_v1'], None, 'openai.chat', {'model': 'x', 'max_tokens': 5},
                              'gpt-5-mini')  # the OpenAI row's reasoning-model rules apply to v1 deployments too
        self.assertEqual(out.get('max_completion_tokens'), 5)

    def test_reasoning_effort_globs(self):
        self.assertTrue(accepts_reasoning_effort(PROFILES['xai'], 'grok-4.5'))
        self.assertTrue(accepts_reasoning_effort(PROFILES['xai'], 'grok-3-mini'))
        self.assertFalse(accepts_reasoning_effort(PROFILES['xai'], 'grok-4'))
        self.assertTrue(accepts_reasoning_effort(PROFILES['groq'], 'openai/gpt-oss-120b'))
        self.assertFalse(accepts_reasoning_effort(PROFILES['groq'], 'llama-3.3-70b-versatile'))
        self.assertTrue(accepts_reasoning_effort(PROFILES['deepseek'], 'deepseek-reasoner'))  # no globs: every model
        self.assertFalse(accepts_reasoning_effort(PROFILES['hf_router'], 'any'))  # no 'reasoning' feature

    def test_reasoning_content_round_trip_is_per_profile(self):
        ir = IRRequest(kind='chat', model='m', messages=[
            IRMessage('user', [Text('hi')]),
            IRMessage('assistant', [Thinking('plan', 'sig'), ToolUse(id='t1', name='f', input_json='{}')]),
            IRMessage('user', [ToolResult(tool_use_id='t1', parts=[Text('42')])])])
        notes = []
        deepseek = openai_chat.encode_request(ir, PROFILES['deepseek'], notes=notes)
        self.assertEqual(deepseek['messages'][1]['reasoning_content'], 'plan')
        self.assertNotIn('dropped:thinking', notes)
        notes = []
        groq = openai_chat.encode_request(ir, PROFILES['groq'], notes=notes)
        self.assertNotIn('reasoning_content', groq['messages'][1])
        self.assertIn('dropped:thinking', notes)


if __name__ == '__main__':
    unittest.main()
