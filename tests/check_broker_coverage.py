"""Broker coverage over the integrated app with BrokerDispatch active (create_app, a keyed server, broker.json pointing
at the WP4 loopback mocks; the netguard is installed and every case ends with zero non-loopback attempts).

It drives the support-matrix cells no other check reached:
- local targets through an ACTIVE broker (a minted key and a configured broker.json, so BrokerDispatch inspects every
  request): image generations and edits (a virtual model rewriting the multipart field to the local name), audio
  transcriptions and translations (multipart alias rewrite), Ollama generate (both stream modes), embed and the legacy
  /api/embeddings, Gemini embedContent and batchEmbedContents (bridged to /v1/embeddings) and a native countTokens,
  completions and Responses streams, and the ElevenLabs /stream route; each answer is served by local:engine and the
  fake engine sees the local name;
- the legacy Ollama /api/embeddings shape translated to a cloud OpenAI target ({prompt} in, {embedding} out);
- catalogs: GET /v1/models/{qualified ref}, /api/tags and /v1beta/models merging the broker entries the key may see,
  /api/ps listing loaded native models only (cloud targets are never loaded), and GET /v1/voices merging the upstream
  ElevenLabs voices for a key holding cloud:<id> (fetched once per TTL with a catalog grant; a key without the scope
  sees local voices only and causes no upstream request), plus GET /v1/voices/{voice_id} for local and upstream voices;
- the R15 deadline terminal frame on the Responses, Gemini and Ollama streams, and the ElevenLabs binary stream ended
  without a terminating chunk;
- inbound authentication refusals in the ElevenLabs and Ollama envelopes;
- upstream 429 from the ElevenLabs and Ollama Cloud adapters: 429 upstream_rate_limited with Retry-After, in the
  inbound protocol's envelope.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import ElevenLabsMock, GeminiMock, OllamaCloudMock, OpenAIMock, netguard  # noqa: E402

BOOT = 'wp9-coverage-bootstrap-0001'
GPT, GEMINI, OLLAMA, ELEVEN = 'gpt-4o-mini@openai', 'gemini-2.5-flash@gemini', 'gpt-oss:120b@olc', \
    'eleven_multilingual_v2@xi'
UPSTREAM_VOICE = '21m00Tcm4TlvDq8ikWAM'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
LOCAL = {'chat-local': 'chat', 'voice': 'speech', 'embed-local': 'embeddings', 'asr-local': 'transcription',
         'img-local': ('image', 'diffusers')}
MOCKS: dict = {}


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['gemini'] = GeminiMock()
    MOCKS['olc'] = OllamaCloudMock()
    MOCKS['xi'] = ElevenLabsMock()
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def document() -> dict:
    m = MOCKS
    return {
        'policy': {'cloud_policy': 'explicit'},
        'upstreams': {
            'openai': W.S.upstream(m['openai'], 'openai', credential='oa',
                                   models=['gpt-4o-mini', 'text-embedding-3-small']),
            'gemini': W.S.upstream(m['gemini'], 'gemini', credential='gm', models=['gemini-2.5-flash']),
            'olc': W.S.upstream(m['olc'], 'ollama_cloud', credential='ol', models=['gpt-oss:120b']),
            'xi': W.S.upstream(m['xi'], 'elevenlabs', credential='xk', models=['eleven_multilingual_v2']),
        },
        'virtual_models': {
            'img-vm': {'capability': 'images', 'targets': [{'ref': 'img-local'}]},
            'stt-vm': {'capability': 'transcription', 'targets': [{'ref': 'asr-local'}]},
        },
    }


def png_bytes() -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (8, 8), (40, 80, 120)).save(buffer, format='PNG')
    return buffer.getvalue()


class FakeNativeClient:
    """The leased llama-server client: /apply-template then /tokenize (one token per word)."""

    def __init__(self, seen: list):
        self.seen = seen

    async def post(self, route, json=None):
        import httpx
        self.seen.append(route)
        if route == '/apply-template':
            text = ''.join(m.get('content') or '' for m in json['messages'] if isinstance(m.get('content'), str))
            payload = {'prompt': '<|start|>' + text + '<|end|>'}
        else:
            payload = {'tokens': list(range(len(json['content'].split())))}
        return httpx.Response(200, json=payload)


class Base(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
            item.frame_delay = 0
        self.full = W.FullApp(document(), models=LOCAL, api_key=BOOT)
        self.c = self.full.http
        engine = self.full.engine

        def infer(name, task, data, emit=None, cancel=None, **kw):  # the recording fake, plus PIL images
            if task in ('image', 'image-edit'):
                from PIL import Image
                engine.calls.append((name, task, {k: v for k, v in data.items() if not isinstance(v, bytes)}))
                return [Image.new('RGB', (4, 4))]
            return engine(name, task, data, emit=emit, cancel=cancel, **kw)
        self.full.app.state.engine.target.infer = infer
        self.engine = engine
        self.key_id, self.key = self.full.mint('inference', 'cloud', 'models:read', name='cloud')

    def tearDown(self):
        for item in MOCKS.values():
            item.frame_delay = 0
            item.faults.clear()
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def hits(self) -> int:
        return W.S.total_hits(*MOCKS.values())

    def post(self, path, body, headers=None, **kw):
        return self.c.post(path, json=body, headers=headers or W.bearer(self.key), **kw)

    def local(self, reply, model, task=None):
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        self.assertTrue(reply.headers['x-titanos-served-by'].startswith('local:'), reply.headers)
        self.assertEqual(self.engine.calls[-1][0], model)
        if task is not None:
            self.assertEqual(self.engine.calls[-1][1], task)


class LocalThroughActiveBroker(Base):
    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'local image outputs are PIL images: needs the pillow extra')
    def test_images_generations_and_edits_reach_the_local_route(self):
        generated = self.post('/v1/images/generations', {'model': 'img-local', 'prompt': 'a cat'})
        self.local(generated, 'img-local', 'image')
        self.assertTrue(generated.json()['data'][0]['b64_json'])
        edited = self.c.post('/v1/images/edits', headers=W.bearer(self.key),
                             files={'image': ('a.png', png_bytes(), 'image/png')},
                             data={'model': 'img-vm', 'prompt': 'a hat'})
        self.local(edited, 'img-local')
        self.assertEqual(edited.headers['x-titanos-requested-model'], 'img-vm')
        self.assertEqual(edited.headers['x-titanos-served-model'], 'img-local')
        self.assertEqual(self.hits(), 0)

    def test_audio_transcriptions_and_translations_rewrite_the_multipart_alias(self):
        wav = W.WAV + bytes(3200)
        for path in ('/v1/audio/transcriptions', '/v1/audio/translations'):
            with self.subTest(path=path):
                reply = self.c.post(path, headers=W.bearer(self.key), files={'file': ('a.wav', wav, 'audio/wav')},
                                    data={'model': 'stt-vm'})
                self.local(reply, 'asr-local', 'transcription')
                self.assertEqual(reply.json()['text'], 'local transcript')
                self.assertEqual(self.engine.calls[-1][2]['audio'], wav)  # the upload survives the rewrite intact
                self.assertEqual(reply.headers['x-titanos-served-model'], 'asr-local')
        self.assertEqual(self.hits(), 0)

    def test_ollama_generate_embed_and_legacy_embeddings(self):
        whole = self.post('/api/generate', {'model': 'chat-local', 'prompt': 'hi', 'stream': False})
        self.local(whole, 'chat-local')
        self.assertEqual((whole.json()['response'], whole.json()['done']), ('local hello', True))
        streamed = self.post('/api/generate', {'model': 'chat-local', 'prompt': 'hi'})
        self.local(streamed, 'chat-local')
        lines = [json.loads(line) for line in streamed.text.splitlines() if line.strip()]
        self.assertGreater(len(lines), 1)
        self.assertEqual(''.join(line.get('response', '') for line in lines), 'local hello')
        self.assertTrue(lines[-1]['done'])
        embed = self.post('/api/embed', {'model': 'embed-local', 'input': 'hi'})
        self.local(embed, 'embed-local', 'embeddings')
        self.assertEqual(embed.json()['embeddings'], [[0.25, 0.5]])
        legacy = self.post('/api/embeddings', {'model': 'embed-local', 'prompt': 'hi'})
        self.local(legacy, 'embed-local', 'embeddings')
        self.assertEqual(legacy.json(), {'embedding': [0.25, 0.5]})
        self.assertEqual(self.hits(), 0)

    def test_gemini_embeddings_bridge_and_native_count_tokens(self):
        single = self.post('/v1beta/models/embed-local:embedContent', {'content': {'parts': [{'text': 'hi'}]}},
                           headers={'x-goog-api-key': self.key})
        self.local(single, 'embed-local', 'embeddings')
        self.assertEqual(single.json(), {'embedding': {'values': [0.25, 0.5]}})
        batch = self.post('/v1beta/models/embed-local:batchEmbedContents',
                          {'requests': [{'content': {'parts': [{'text': 'a'}]}}, {'content': {'parts': [{'text': 'b'}]}}]},
                          headers={'x-goog-api-key': self.key})
        self.local(batch, 'embed-local', 'embeddings')
        self.assertEqual(len(batch.json()['embeddings']), 2)
        from titanos.config import ModelSpec
        seen: list = []
        spec = ModelSpec.model_construct(path=str(self.full.root), task='chat', backend='llama-cpp', defaults={},
                                         voices={}, device='auto')
        self.full.config.models['native-chat'] = spec
        native = self.full.app.state.native
        native.models['native-chat'] = spec

        @asynccontextmanager
        async def lease(model, worker=None, *, require_loaded=False):
            yield SimpleNamespace(client=FakeNativeClient(seen))
        native.lease = lease
        counted = self.post('/v1beta/models/native-chat:countTokens',
                            {'contents': [{'role': 'user', 'parts': [{'text': 'one two three'}]}]},
                            headers={'x-goog-api-key': self.key})
        self.assertEqual(counted.status_code, 200, counted.text[:300])
        self.assertEqual(counted.json(), {'totalTokens': 3})
        self.assertEqual(seen, ['/apply-template', '/tokenize'])
        self.assertEqual(self.hits(), 0)

    def test_completions_and_responses_streams(self):
        completions = self.post('/v1/completions', {'model': 'chat-local', 'prompt': 'hi', 'stream': True})
        self.local(completions, 'chat-local')
        frames = [block for block in completions.text.split('\n\n') if block.strip()]
        self.assertEqual(frames[-1].strip(), 'data: [DONE]')
        self.assertEqual(json.loads(frames[0][len('data: '):])['object'], 'text_completion')
        responses = self.post('/v1/responses', {'model': 'chat-local', 'input': 'hi', 'stream': True})
        self.local(responses, 'chat-local')
        events = [block.split('\n', 1)[0] for block in responses.text.split('\n\n') if block.strip()]
        self.assertEqual((events[0], events[-1]), ('event: response.created', 'event: response.completed'))
        self.assertEqual(self.hits(), 0)

    def test_elevenlabs_stream_route_answers_locally(self):
        reply = self.c.post('/v1/text-to-speech/default/stream', json={'text': 'hi', 'model_id': 'voice'},
                            headers={'xi-api-key': self.key})
        self.local(reply, 'voice', 'speech')
        self.assertTrue(reply.content.startswith(b'RIFF'))
        self.assertEqual(self.hits(), 0)


class CloudLegacyEmbeddings(Base):
    def test_legacy_ollama_embeddings_translated_to_a_cloud_target(self):
        reply = self.post('/api/embeddings', {'model': 'text-embedding-3-small@openai', 'prompt': 'hi'})
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        self.assertEqual(reply.headers['x-titanos-served-by'], 'cloud:openai')
        self.assertEqual(reply.headers['x-titanos-translation'], 'ollama->openai')
        self.assertEqual(list(reply.json()), ['embedding'])
        hit = MOCKS['openai'].hits[-1]
        self.assertEqual((hit.surface, hit.json()['input']), ('openai.embeddings', ['hi']))


class Catalogs(Base):
    def test_qualified_model_entry_tags_and_v1beta_merge_the_broker_catalog(self):
        entry = self.c.get('/v1/models/' + GPT, headers=W.bearer(self.key))
        self.assertEqual(entry.status_code, 200, entry.text[:300])
        self.assertEqual((entry.json()['id'], entry.json()['owned_by']), (GPT, 'openai'))
        self.assertEqual(self.c.get('/v1/models/nope@openai', headers=W.bearer(self.key)).status_code, 404)
        tags = {row['name'] for row in self.c.get('/api/tags', headers=W.bearer(self.key)).json()['models']}
        self.assertTrue({'chat-local', GPT, OLLAMA} <= tags, tags)
        beta = {row['baseModelId']: row for row in self.c.get('/v1beta/models',
                                                             headers={'x-goog-api-key': self.key}).json()['models']}
        self.assertIn('chat-local', beta)
        self.assertTrue(beta[GEMINI]['titanos']['remote'])
        self.assertIn('generateContent', beta[GEMINI]['supportedGenerationMethods'])
        _, local_only = self.full.mint('inference', 'models:read', name='local-only',
                                       constraints={'target_classes': ['local']})
        narrow = {row['name'] for row in self.c.get('/api/tags', headers=W.bearer(local_only)).json()['models']}
        self.assertNotIn(GPT, narrow)
        self.assertEqual(self.hits(), 0)  # static catalogs: listing never contacts an upstream

    def test_ps_lists_loaded_native_models_only(self):
        reply = self.c.get('/api/ps', headers=W.bearer(self.key))
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.json(), {'models': []})  # nothing loaded; cloud targets are never 'loaded'
        self.assertEqual(self.hits(), 0)

    def test_voices_merge_upstream_voices_for_keys_that_see_the_upstream(self):
        voice_hits = lambda: [h for h in MOCKS['xi'].hits if h.path == '/v1/voices']  # noqa: E731
        _, local_only = self.full.mint('models:read', 'inference', name='no-cloud')
        listed = self.c.get('/v1/voices', headers={'xi-api-key': local_only}).json()['voices']
        self.assertEqual([v['voice_id'] for v in listed], ['default'])
        self.assertEqual(voice_hits(), [])  # a key without cloud:xi never causes an upstream request
        listed = self.c.get('/v1/voices', headers={'xi-api-key': self.key}).json()['voices']
        ids = [v['voice_id'] for v in listed]
        self.assertEqual(ids[0], 'default')
        self.assertIn(UPSTREAM_VOICE, ids)
        upstream = next(v for v in listed if v['voice_id'] == UPSTREAM_VOICE)
        self.assertEqual((upstream['name'], upstream['titanos']), ('Rachel', {'remote': True, 'upstream': 'xi'}))
        self.assertEqual(len(voice_hits()), 1)
        self.assertEqual(voice_hits()[0].credential, '<sentinel:elevenlabs>')
        one = self.c.get('/v1/voices/' + UPSTREAM_VOICE, headers={'xi-api-key': self.key})
        self.assertEqual((one.status_code, one.json()['voice_id']), (200, UPSTREAM_VOICE))
        local = self.c.get('/v1/voices/default', headers={'xi-api-key': self.key})
        self.assertEqual((local.status_code, local.json()['category']), (200, 'local'))
        missing = self.c.get('/v1/voices/nope', headers={'xi-api-key': self.key})
        self.assertEqual(missing.status_code, 404)
        self.assertIn('detail', missing.json())
        self.assertEqual(len(voice_hits()), 1)  # cached for the catalog TTL
        self.assertNotIn(self.key, json.dumps(listed))


class Envelopes(Base):
    def deadline(self, mock, path, body, headers):
        MOCKS[mock].frame_delay = 0.4
        try:
            started = time.monotonic()
            reply = self.c.post(path, json=body, headers={**headers, 'X-Titanos-Deadline-Ms': '700'})
        finally:
            MOCKS[mock].frame_delay = 0
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        return reply

    def test_deadline_terminal_frames(self):
        """R15 on the Responses, Gemini and Ollama streams: the committed stream ends with the protocol's terminal
        frame naming the expired deadline."""
        bearer = W.bearer(self.key)
        responses = self.deadline('openai', '/v1/responses', {'model': GPT, 'input': 'hi', 'stream': True}, bearer)
        last = [block for block in responses.text.split('\n\n') if block.strip()][-1]
        self.assertTrue(last.startswith('event: response.failed'), last)
        failed = json.loads(last.split('data: ', 1)[1])
        self.assertEqual(failed['response']['status'], 'failed')
        self.assertIn('deadline', failed['response']['error']['message'])
        gemini = self.deadline('gemini', f'/v1beta/models/{GEMINI}:streamGenerateContent?alt=sse',
                               {'contents': [{'role': 'user', 'parts': [{'text': 'hi'}]}]},
                               {'x-goog-api-key': self.key})
        error = json.loads([b for b in gemini.text.split('\n\n') if b.strip()][-1].split('data: ', 1)[1])['error']
        self.assertEqual((error['code'], error['status']), (504, 'DEADLINE_EXCEEDED'))
        ollama = self.deadline('olc', '/api/chat', {'model': OLLAMA, 'messages': CHAT, 'stream': True}, bearer)
        lines = [json.loads(line) for line in ollama.text.splitlines() if line.strip()]
        self.assertIn('deadline', lines[-1]['error'])
        self.assertNotIn('done', lines[-1])

    def test_elevenlabs_binary_stream_ends_without_a_terminating_chunk(self):
        """Binary audio has no error frame: on the deadline the body stops early (no terminating chunk) and the
        connection closes."""
        MOCKS['xi'].frame_delay = 0.4
        try:
            started = time.monotonic()
            # the in-process client surfaces the aborted response body as the dispatcher's StreamAborted
            with self.assertRaises(Exception) as aborted:
                self.c.post(f'/v1/text-to-speech/{UPSTREAM_VOICE}/stream',
                            json={'text': 'Hello there', 'model_id': ELEVEN},
                            headers={'xi-api-key': self.key, 'X-Titanos-Deadline-Ms': '700'})
        finally:
            MOCKS['xi'].frame_delay = 0
        self.assertEqual(type(aborted.exception).__name__, 'StreamAborted')
        self.assertIn('deadline', str(aborted.exception))
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(MOCKS['xi'].hits[-1].surface, 'elevenlabs.tts_stream')

    def test_auth_refusals_in_the_elevenlabs_and_ollama_envelopes(self):
        wrong = 'tk_' + 'z' * 10 + '_' + 'y' * 40
        eleven = self.c.post('/v1/text-to-speech/default', json={'text': 'hi', 'model_id': 'voice'},
                             headers={'xi-api-key': wrong})
        self.assertEqual(eleven.status_code, 401)
        self.assertIsInstance(eleven.json()['detail']['message'], str)
        self.assertIn('status', eleven.json()['detail'])
        ollama = self.c.post('/api/chat', json={'model': 'chat-local', 'messages': CHAT, 'stream': False},
                             headers=W.bearer(wrong))
        self.assertEqual(ollama.status_code, 401)
        self.assertIsInstance(ollama.json()['error'], str)
        for reply in (eleven, ollama):
            self.assertNotIn(wrong, reply.text)
        self.assertEqual(self.hits(), 0)

    def test_upstream_rate_limits_from_elevenlabs_and_ollama(self):
        MOCKS['xi'].faults.push('status', surface='elevenlabs.tts', status=429, retry_after=7)
        eleven = self.c.post(f'/v1/text-to-speech/{UPSTREAM_VOICE}', json={'text': 'hi', 'model_id': ELEVEN},
                             headers={'xi-api-key': self.key})
        self.assertEqual(eleven.status_code, 429, eleven.text[:300])
        self.assertEqual(eleven.headers['retry-after'], '7')
        self.assertIn('detail', eleven.json())
        MOCKS['olc'].faults.push('status', surface='ollama.chat', status=429, retry_after=5)
        ollama = self.post('/api/chat', {'model': OLLAMA, 'messages': CHAT, 'stream': False})
        self.assertEqual(ollama.status_code, 429, ollama.text[:300])
        self.assertEqual(ollama.headers['retry-after'], '5')
        self.assertIsInstance(ollama.json()['error'], str)
        for reply in (eleven, ollama):
            self.assertEqual(reply.headers['x-titanos-error-origin'], 'upstream')
        self.assertEqual((len(MOCKS['xi'].hits), len(MOCKS['olc'].hits)), (1, 1))  # never retried


if __name__ == '__main__':
    unittest.main(verbosity=2)
