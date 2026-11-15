"""WP7 API Plus stages with broker placement (StageRouter, GatewayRouter, CloudInvoker) over the integrated app, with the
WP4 loopback mocks (netguard installed; zero non-loopback attempts).

- location semantics per operation: the default local_only refuses a cloud ref (400 location_unsatisfiable) before any
  hit, auto reaches it; local-only stages (rerank, VAD) refuse cloud_only (400 broker_capability_unavailable); a local
  model asked for cloud_only is location_unsatisfiable;
- cloud ASR through the OpenAI and ElevenLabs mocks runs only after the local VAD gate: no_speech gives zero hits;
- the LLM of sentiment and sentiment_compose goes to the cloud through GatewayRouter (decoding resolves to validate);
- text translation goes to cloud chat with the versioned broker-translation-v1 prompt; include_provenance with a cloud
  target is 400 provenance_unavailable_for_cloud;
- a retrieval answer needs a local llama-cpp model (exact native context accounting): a cloud LLM is refused typed;
- /v1/extensions reports runtime 'standalone-local' without a broker.json and 'local+broker' with its cloud stages.
"""
from __future__ import annotations

import base64
import hashlib
import io
import sys
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import ElevenLabsMock, OpenAIMock, netguard  # noqa: E402

GPT = 'gpt-4o-mini@openai'
BOOT = 'wp7-plus-bootstrap-0001'
MOCKS: dict = {}
LOCAL = {'nmt-local': 'translation', 'vad-local': 'vad', 'cls-local': 'sentiment', 'rerank-local': 'rerank'}


def wav_bytes(seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b'\x00\x00' * int(16000 * seconds))
    return buffer.getvalue()


AUDIO = base64.b64encode(wav_bytes()).decode()


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['elevenlabs'] = ElevenLabsMock()
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def document() -> dict:
    return {'policy': {'cloud_policy': 'explicit'},
            'upstreams': {
                'openai': W.S.upstream(MOCKS['openai'], 'openai', credential='oa', models=['gpt-4o-mini', 'whisper-1']),
                'elevenlabs': W.S.upstream(MOCKS['elevenlabs'], 'elevenlabs', credential='xi', models=['scribe_v1']),
            }}


class StageEngine(W.FakeEngine):
    speech = True

    def __call__(self, name, task, data, emit=None, cancel=None, **kw):
        if task == 'vad':
            self.calls.append((name, task, data))
            return {'model': name, 'task': 'vad', 'sample_rate': 16000, 'duration': 1.0, 'decoded_samples': 16000,
                    'frame_samples': 512, 'context_samples': 64, 'state_reset': 'per_request',
                    'score_kind': 'speech_probability_uncalibrated', 'frames': None,
                    'segments': [{'start': .2, 'end': .8, 'start_sample': 3200, 'end_sample': 12800}]
                    if self.speech else [],
                    'parameters': {'threshold': .5}, 'input_sha256': hashlib.sha256(data['audio']).hexdigest(),
                    'provenance': {'backend': 'onnx', 'device': 'cpu', 'checkpoint_digest': 'a' * 64}}
        if task == 'sentiment':
            self.calls.append((name, task, data))
            return {'scores': {'negative': .05, 'neutral': .05, 'positive': .9}, 'score_kind': 'softmax_uncalibrated',
                    'input_tokens': 8, 'provenance': {'backend': 'transformers', 'device': 'cpu', 'repository': None,
                                                      'revision': None, 'checkpoint_digest': 'c' * 64,
                                                      'label_mapping': {'POSITIVE': 'positive'}}}
        return super().__call__(name, task, data, emit, cancel, **kw)


class Case(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
        self.full = W.FullApp(document(), models=LOCAL, api_key=BOOT)
        self.engine = StageEngine()
        self.full.app.state.engine.target.infer = self.engine
        self.c = self.full.http
        _, self.token = self.full.mint('inference', 'cloud', name='plus')

    def tearDown(self):
        self.full.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def hits(self) -> int:
        return W.S.total_hits(*MOCKS.values())

    def post(self, path, body):
        return self.c.post(path, json=body, headers=W.bearer(self.token))


class Locations(Case):
    def test_location_semantics_per_operation(self):
        body = {'model': GPT, 'text': 'Bonjour', 'source_language': 'fra_Latn', 'target_language': 'eng_Latn'}
        refused = self.post('/v1/translations', body)
        self.assertEqual((refused.status_code, refused.json()['error']['code']), (400, 'location_unsatisfiable'))
        self.assertEqual(self.hits(), 0)
        cloud = self.post('/v1/translations', {**body, 'location': 'auto'})
        self.assertEqual(cloud.status_code, 200, cloud.text[:300])
        sent = MOCKS['openai'].hits[-1].json()
        self.assertIn('translation engine', sent['messages'][0]['content'])
        self.assertEqual(sent['messages'][1]['content'], 'Bonjour')
        self.assertEqual(cloud.json()['text'], 'Hello! How can I assist you today?')
        provenance = self.post('/v1/translations', {**body, 'location': 'auto', 'include_provenance': True})
        self.assertEqual((provenance.status_code, provenance.json()['error']['code']),
                         (400, 'provenance_unavailable_for_cloud'))
        local_cloud_only = self.post('/v1/translations', {**body, 'model': 'nmt-local', 'location': 'cloud_only'})
        self.assertEqual((local_cloud_only.status_code, local_cloud_only.json()['error']['code']),
                         (400, 'location_unsatisfiable'))
        local = self.post('/v1/translations', {**body, 'model': 'nmt-local'})
        self.assertEqual((local.status_code, local.json()['text']), (200, 'local translation'))
        rerank = self.post('/v1/rerank', {'model': 'rerank-local', 'query': 'q', 'documents': ['a'],
                                          'location': 'cloud_only'})
        self.assertEqual((rerank.status_code, rerank.json()['error']['code']), (400, 'broker_capability_unavailable'))
        vad = self.post('/v1/audio/vad', {'model': 'vad-local', 'audio_base64': AUDIO, 'location': 'cloud_only'})
        self.assertEqual((vad.status_code, vad.json()['error']['code']), (400, 'broker_capability_unavailable'))
        self.assertEqual(len(MOCKS['openai'].hits), 1)


class CloudASR(Case):
    def transcribe(self, model):
        return self.post('/v1/extensions/infer', {'operation': 'audio_transcription', 'model': model, 'location': 'auto',
                                                  'audio_base64': AUDIO, 'vad_enabled': True, 'vad_model': 'vad-local',
                                                  'response_format': 'verbose_json'})

    def test_cloud_asr_after_the_local_vad_gate(self):
        for model, mock in (('whisper-1@openai', MOCKS['openai']), ('scribe_v1@elevenlabs', MOCKS['elevenlabs'])):
            with self.subTest(model=model):
                reply = self.transcribe(model)
                self.assertEqual(reply.status_code, 200, reply.text[:300])
                result = reply.json()
                self.assertEqual(result['backend'], 'cloud')
                self.assertTrue(result['vad_gate']['asr_executed'])
                self.assertTrue(result['text'])
                self.assertEqual(self.engine.calls[-1][1], 'vad')
                self.assertTrue(mock.hits)
        self.engine.speech = False
        before = self.hits()
        silent = self.transcribe('whisper-1@openai')
        self.assertEqual(silent.status_code, 200, silent.text[:300])
        self.assertEqual((silent.json()['text'], silent.json()['vad_gate']['decision']), ('', 'no_speech_detected'))
        self.assertEqual(self.hits(), before)


class CloudLLM(Case):
    def test_sentiment_and_compose_reach_the_cloud_llm_through_gateway_router(self):
        reply = self.post('/v1/sentiment', {'model': GPT, 'text': 'I love it.', 'location': 'auto'})
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        sent = MOCKS['openai'].hits[-1].json()
        self.assertEqual(sent['messages'][0]['role'], 'system')
        self.assertNotIn('response_format', sent)  # decoding resolved to validate (no native JSON schema)
        self.assertEqual(reply.json()['model'], GPT)
        before = len(MOCKS['openai'].hits)
        compose = self.post('/v1/sentiment/compose', {'classifier_model': 'cls-local', 'llm_model': GPT,
                                                      'text': 'I love it.', 'location': 'auto'})
        self.assertEqual(compose.status_code, 200, compose.text[:300])
        result = compose.json()
        self.assertEqual((result['classifier']['status'], result['llm']['status']), ('completed', 'completed'))
        self.assertEqual(result['llm']['model'], GPT)
        self.assertEqual(len(MOCKS['openai'].hits), before + 1)
        self.assertEqual(self.engine.calls[-1][1], 'sentiment')
        local_only = self.post('/v1/sentiment', {'model': GPT, 'text': 'I love it.'})
        self.assertEqual((local_only.status_code, local_only.json()['error']['code']), (400, 'location_unsatisfiable'))

    def test_retrieval_answer_needs_a_local_native_llm(self):
        before = self.hits()
        body = {'query': 'q', 'documents': [{'id': 'd1', 'text': 'alpha'}], 'embedding_model': 'nmt-local',
                'rerank_model': 'rerank-local', 'llm_model': GPT}
        default = self.post('/v1/retrieval/answer', body)  # the API Plus default placement is local_only
        self.assertEqual((default.status_code, default.json()['error']['code']), (400, 'location_unsatisfiable'))
        placed = self.c.post('/v1/retrieval/answer', json=body,
                             headers={**W.bearer(self.token), 'X-Titanos-Location': 'auto'})
        self.assertEqual((placed.status_code, placed.json()['error']['code']), (400, 'broker_capability_unavailable'))
        self.assertEqual(self.hits(), before)


class Extensions(unittest.TestCase):
    def test_runtime_reporting(self):
        netguard.reset_attempts()
        plain = W.FullApp(None, models=LOCAL)
        try:
            data = plain.http.get('/v1/extensions').json()
            self.assertEqual(data['runtime'], 'standalone-local')
            self.assertNotIn('cloud_stages', data)
        finally:
            plain.close()
        MOCKS['openai'].reset()
        configured = W.FullApp(document(), models=LOCAL)
        try:
            data = configured.http.get('/v1/extensions').json()
            self.assertEqual(data['runtime'], 'local+broker')
            self.assertIn('translation', data['cloud_stages'])
            self.assertIn('cloud_only', data['locations'])
        finally:
            configured.close()
        self.assertEqual(netguard.attempts(), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
