"""Thread-side cloud reach (WP6): GatewayRouter, StageRouter with CloudSpecView, and CloudInvoker against the WP4 loopback
mocks (netguard installed; zero non-loopback attempts).

- GatewayRouter.candidates and complete for cloud and local targets (NativePool.complete's signature, require_loaded
  accepted and forwarded);
- StageRouter returns a read-only CloudSpecView, so spec.backend and spec.defaults reads keep working;
- CloudInvoker: speech returns WAV (OpenAI wav, ElevenLabs pcm wrapped); image returns PNG bytes and refuses JPEG
  (before egress for a JPEG-only profile, after it for a JPEG answer); transcription returns an ASRMetadata-shaped dict;
  chat, translation and embeddings shapes; cancel and deadline honoured;
- Engine.lock is never acquired (an instrumented lock).
"""
from __future__ import annotations

import asyncio
import dataclasses
import io
import struct
import threading
import time
import unittest
import wave

import broker_wp6_support as S
from broker_mocks import ElevenLabsMock, OpenAIMock, netguard

from titanos.audio_contract import ASRMetadata
from titanos.broker.errors import BrokerError
from titanos.broker.stages import TRANSLATION_PROMPTS, CloudSpecView, GatewayRouter

MOCKS: dict = {}
CLOUD = S.key('inference', 'cloud', id='kstages0001')
VOICE = '21m00Tcm4TlvDq8ikWAM'


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['xi'] = ElevenLabsMock()
    MOCKS['xai'] = OpenAIMock(variant='xai')
    for item in MOCKS.values():
        item.start()


def tearDownModule():
    for item in MOCKS.values():
        item.stop()


def wav_bytes(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b'\x00\x00' * int(seconds * rate))
    return buffer.getvalue()


class RecordingLock:
    def __init__(self):
        self.acquired = 0
        self._lock = threading.Lock()

    def acquire(self, *args, **kw):
        self.acquired += 1
        return self._lock.acquire(*args, **kw)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class FakeEngine:
    def __init__(self):
        self.lock = RecordingLock()

    def infer(self, *args, **kw):  # pragma: no cover - must never be reached by cloud work
        with self.lock:
            raise AssertionError('the engine was used for a cloud target')


class FakeNative:
    def __init__(self):
        self.calls = []

    def candidates(self, model, worker=None):
        self.calls.append(('candidates', model, worker))
        return ['native-worker-0']

    async def complete(self, data, *, deadline=None, cancel=None, context_max_tokens=None, require_loaded=False):
        self.calls.append(('complete', data['model'], require_loaded))
        return {'text': 'native text', 'finish_reason': 'stop', 'usage': {}, 'native_response': {},
                'provenance': {'backend': 'llama-cpp'}, 'native_runtime': {}}


def document() -> dict:
    m = MOCKS
    return {'policy': {'cloud_policy': 'fallback'},
            'upstreams': {
                'openai': S.upstream(m['openai'], 'openai', credential='oa',
                                     models=['tts-1', 'dall-e-3', 'whisper-1', 'gpt-4o-mini', 'text-embedding-3-small']),
                'xi': S.upstream(m['xi'], 'elevenlabs', credential='xi', models=['eleven_multilingual_v2', 'scribe_v1']),
                'xai': S.upstream(m['xai'], 'xai', credential='xa', models=['grok-2-image']),
                'xjpeg': S.upstream(m['xai'], 'custom_openai', credential='xj', models=['grok-2-image'])},
            'voice_maps': {'openai_to_elevenlabs': {'alloy': VOICE}}}


class Case(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        for item in MOCKS.values():
            item.reset()
        self.engine = FakeEngine()
        self.native = FakeNative()
        self.lab = S.Lab(document(), local={'qwen-local': ('chat', 'llama-cpp'), 'vad-local': 'vad',
                                            'tts-local': 'speech'},
                         creds={'xai': 'xai', 'xjpeg': 'xai'}, engine_slot={'engine': self.engine})
        self.broker = self.lab.broker
        self.invoker = self.broker.invoker()

    def tearDown(self):
        for item in MOCKS.values():
            item.gate.set()
        self.assertEqual(self.engine.lock.acquired, 0, 'Engine.lock was acquired by cloud work')
        self.lab.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def target(self, ref):
        return self.broker.stage_router().resolve_stage(ref, self.task_for(ref), location='auto',
                                                        principal=CLOUD).target

    @staticmethod
    def task_for(ref):
        return {'tts-1': 'speech', 'eleven_multilingual_v2': 'speech', 'dall-e-3': 'image', 'grok-2-image': 'image',
                'whisper-1': 'transcription', 'scribe_v1': 'transcription', 'text-embedding-3-small': 'embeddings'
                }.get(ref.split('@')[0], 'chat')

    def infer(self, ref, task, data, **kw):
        kw.setdefault('principal', CLOUD)
        return self.invoker.infer_sync(self.target(ref), task, data, **kw)

    def code(self, expected, fn, *args, **kw):
        with self.assertRaises(BrokerError) as caught:
            fn(*args, **kw)
        self.assertEqual(caught.exception.code, expected, str(caught.exception))
        return caught.exception


class Gateway(Case):
    def test_candidates_and_complete_for_cloud_and_local(self):
        gw = self.broker.gateway(self.native, CLOUD, 'auto')
        self.assertIsInstance(gw, GatewayRouter)
        self.assertEqual(gw.candidates('qwen-local', 'w1'), ['native-worker-0'])
        self.assertEqual(self.native.calls[-1], ('candidates', 'qwen-local', 'w1'))
        cloud = gw.candidates('gpt-4o-mini@openai')
        self.assertEqual([t.ref for t in cloud], ['gpt-4o-mini@openai'])
        self.code('invalid_request', gw.candidates, 'gpt-4o-mini@openai', 'w1')
        self.code('model_not_found', self.broker.gateway(self.native, S.key('inference'), 'auto').candidates,
                  'gpt-4o-mini@openai')
        self.code('location_unsatisfiable', self.broker.gateway(self.native, CLOUD, 'local_only').candidates,
                  'gpt-4o-mini@openai')
        self.assertEqual(MOCKS['openai'].hits, [])

        async def local():
            return await gw.complete({'model': 'qwen-local', 'messages': []}, require_loaded=True)
        self.assertEqual(asyncio.run(local())['text'], 'native text')
        self.assertEqual(self.native.calls[-1], ('complete', 'qwen-local', True))

        async def cloud_call(**kw):
            try:
                return await gw.complete({'model': 'gpt-4o-mini@openai',
                                          'messages': [{'role': 'user', 'content': 'hi'}]}, **kw)
            finally:
                await self.broker.close()
        result = asyncio.run(cloud_call(require_loaded=True, deadline=time.monotonic() + 30))
        self.assertEqual(set(result), {'text', 'finish_reason', 'usage', 'native_response', 'provenance',
                                       'native_runtime'})
        self.assertTrue(result['text'])
        self.assertEqual((result['provenance']['backend'], result['provenance']['upstream']), ('cloud', 'openai'))
        self.assertEqual(result['finish_reason'], 'stop')
        with self.assertRaises(BrokerError) as caught:
            asyncio.run(cloud_call(context_max_tokens=100))
        self.assertEqual(caught.exception.code, 'broker_capability_unavailable')


class Stages(Case):
    def test_cloud_spec_view_keeps_spec_reads_working(self):
        router = self.broker.stage_router()
        stage = router.resolve_stage('gpt-4o-mini@openai', 'translation', location='auto', principal=CLOUD)
        self.assertEqual((stage.kind, stage.backend, stage.task), ('cloud', 'cloud', 'translation'))
        spec = stage.spec
        self.assertIsInstance(spec, CloudSpecView)
        self.assertEqual((spec.backend, spec.device, spec.task), ('cloud', 'remote', 'translation'))
        self.assertIsNone(spec.defaults.get('source_language'))
        self.assertEqual(dict(spec.voices), {})
        with self.assertRaises(TypeError):
            spec.defaults['source_language'] = 'en'
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.backend = 'transformers'
        local = router.resolve_stage('qwen-local', 'chat', location='auto', principal=CLOUD)
        self.assertEqual((local.kind, local.spec.backend), ('local', 'llama-cpp'))
        self.assertEqual(router.resolve_stage('vad-local', 'vad', principal=CLOUD).kind, 'local')
        self.code('broker_capability_unavailable', router.resolve_stage, 'vad-local', 'vad', location='cloud_only',
                  principal=CLOUD)
        self.code('model_not_found', router.resolve_stage, 'nope', 'chat', location='auto', principal=CLOUD)
        self.code('location_unsatisfiable', router.resolve_stage, 'gpt-4o-mini@openai', 'chat', principal=CLOUD)
        self.code('model_not_found', router.resolve_stage, 'gpt-4o-mini@openai', 'chat', location='auto',
                  principal=S.key('inference:plus'))
        self.assertEqual(MOCKS['openai'].hits, [])


class Invoker(Case):
    def test_speech_returns_wav(self):
        audio, kind = self.infer('tts-1@openai', 'speech', {'input': 'Hello there', 'voice': 'alloy'})
        self.assertEqual((audio[:4], audio[8:12], kind), (b'RIFF', b'WAVE', 'audio/wav'))
        self.assertEqual(MOCKS['openai'].hits[-1].json()['response_format'], 'wav')
        audio, kind = self.infer('eleven_multilingual_v2@xi', 'speech', {'input': 'Hello there', 'voice': 'alloy'})
        self.assertEqual((audio[:4], kind), (b'RIFF', 'audio/wav'))
        self.assertEqual(struct.unpack('<I', audio[24:28])[0], 24000)
        self.assertEqual(struct.unpack('<I', audio[40:44])[0], len(audio) - 44)
        hit = MOCKS['xi'].hits[-1]
        self.assertEqual(hit.query_params['output_format'], 'pcm_24000')
        self.assertIn(VOICE, hit.path)

    def test_image_returns_png_and_refuses_jpeg(self):
        images = self.infer('dall-e-3@openai', 'image', {'prompt': 'a cat'})
        self.assertTrue(images and all(i.startswith(b'\x89PNG\r\n\x1a\n') for i in images))
        # every OpenAI image model refuses response_format (live 2026-10-02) and answers base64: never sent
        self.assertNotIn('response_format', MOCKS['openai'].hits[-1].json())
        self.code('broker_capability_unavailable', self.infer, 'grok-2-image@xai', 'image', {'prompt': 'a cat'})
        self.assertEqual(MOCKS['xai'].hits, [])  # a JPEG-only profile is refused before egress
        self.code('broker_capability_unavailable', self.infer, 'grok-2-image@xjpeg', 'image', {'prompt': 'a cat'})
        self.assertEqual(len(MOCKS['xai'].hits), 1)  # a JPEG answer is refused after it

    def test_transcription_returns_an_asr_shaped_dict(self):
        audio = wav_bytes(0.5)
        for ref in ('whisper-1@openai', 'scribe_v1@xi'):
            with self.subTest(ref=ref):
                result = self.infer(ref, 'transcription', {'audio': audio, 'language': 'en', 'filename': 'a.wav'})
                model = ASRMetadata.model_validate(result)
                self.assertTrue(model.text)
                self.assertEqual((model.backend, model.task), ('cloud', 'transcribe'))
                self.assertGreater(model.duration, 0)

    def test_chat_translation_and_embeddings(self):
        chat = self.infer('gpt-4o-mini@openai', 'chat', {'messages': [{'role': 'user', 'content': 'hi'}]})
        self.assertTrue(chat['text'])
        self.assertIn('prompt_tokens', chat['usage'])
        translated = self.infer('gpt-4o-mini@openai', 'translation', {'text': 'Hallo', 'source_language': 'German',
                                                                      'target_language': 'English'})
        self.assertEqual(translated['prompt_version'], 'broker-translation-v1')
        self.assertIn('broker-translation-v1', TRANSLATION_PROMPTS)
        sent = MOCKS['openai'].hits[-1].json()
        self.assertEqual(sent['messages'][0]['role'], 'system')
        self.assertIn('German', sent['messages'][0]['content'])
        self.assertEqual(sent['messages'][1]['content'], 'Hallo')
        vectors = self.infer('text-embedding-3-small@openai', 'embeddings', {'input': ['a', 'b']})
        self.assertEqual([row['index'] for row in vectors['data']], [0, 1])
        self.assertIn('prompt_tokens', vectors['usage'])
        decisions = self.broker.decisions.query(principal=S.BOOTSTRAP, limit=10)
        self.assertTrue(all(d['translation'].startswith('stage:') for d in decisions))

    def test_cancel_and_deadline_are_honoured(self):
        cancel = threading.Event()
        cancel.set()
        error = self.code('cancelled', self.infer, 'tts-1@openai', 'speech', {'input': 'x', 'voice': 'alloy'},
                          cancel=cancel)
        self.assertEqual(error.status, 499)
        self.code('deadline_exceeded', self.infer, 'tts-1@openai', 'speech', {'input': 'x', 'voice': 'alloy'},
                  deadline=time.monotonic() - 1)
        self.assertEqual(MOCKS['openai'].hits, [])
        # cancelled between body chunks
        MOCKS['openai'].hold(frame=2)
        late = threading.Event()
        timer = threading.Timer(0.3, late.set)
        release = threading.Timer(1.5, MOCKS['openai'].gate.set)
        timer.start()
        release.start()
        try:
            self.code('cancelled', self.infer, 'tts-1@openai', 'speech', {'input': 'x', 'voice': 'alloy'},
                      cancel=late)
        finally:
            timer.cancel()
            release.cancel()
            MOCKS['openai'].gate.set()
        # a slow first byte beyond the deadline
        MOCKS['openai'].faults.push('slow_first_byte', delay=2.0)
        started = time.monotonic()
        with self.assertRaises(BrokerError) as caught:
            self.infer('gpt-4o-mini@openai', 'chat', {'messages': [{'role': 'user', 'content': 'hi'}]},
                       deadline=time.monotonic() + 0.4)
        self.assertIn(caught.exception.code, ('upstream_timeout', 'deadline_exceeded'))
        self.assertLess(time.monotonic() - started, 1.8)

    def test_infer_sync_rechecks_the_principal_and_the_current_policy(self):
        """A target resolved earlier (a queued job) is checked again against the principal as it is now and the
        current snapshot: a key without the cloud scope, a local_only key, or a reload to cloud_policy off is refused
        with zero hits."""
        target = self.target('gpt-4o-mini@openai')
        data = {'messages': [{'role': 'user', 'content': 'hi'}]}
        self.code('model_not_found', self.invoker.infer_sync, target, 'chat', data, principal=S.key('inference'))
        self.code('location_unsatisfiable', self.invoker.infer_sync, target, 'chat', data,
                  principal=S.key('inference', 'cloud', locations=('local_only',)))
        self.lab.rewrite(dict(document(), policy={'cloud_policy': 'off'}))
        self.broker.reload()
        self.code('cloud_disabled', self.invoker.infer_sync, target, 'chat', data, principal=CLOUD)
        self.assertEqual(MOCKS['openai'].hits, [])
        self.lab.rewrite(document())
        self.broker.reload()
        self.assertTrue(self.invoker.infer_sync(target, 'chat', data, principal=CLOUD)['text'])
        self.assertEqual(len(MOCKS['openai'].hits), 1)

    def test_the_engine_lock_is_never_acquired(self):
        self.infer('tts-1@openai', 'speech', {'input': 'x', 'voice': 'alloy'})
        self.infer('dall-e-3@openai', 'image', {'prompt': 'x'})
        self.infer('whisper-1@openai', 'transcription', {'audio': wav_bytes(0.2)})
        self.infer('gpt-4o-mini@openai', 'chat', {'messages': [{'role': 'user', 'content': 'hi'}]})
        self.assertEqual(self.engine.lock.acquired, 0)
        self.code('invalid_request', self.invoker.infer_sync, self.broker.stage_router().resolve_stage(
            'qwen-local', 'chat', principal=CLOUD).target, 'chat', {})


if __name__ == '__main__':
    unittest.main(verbosity=1)
