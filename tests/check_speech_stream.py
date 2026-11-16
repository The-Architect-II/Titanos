"""Speech streaming: POST /v1/audio/speech with stream_format audio sends gim_api's streaming WAV header, then each PCM
chunk as the backend emits it. A fake runtime decides what is emitted; real uvicorn carries the transfer-level cases
(chunk timing, aborts after audio, client disconnects), which TestClient cannot show because it buffers bodies.
"""
import http.client
import io
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
HOME = tempfile.TemporaryDirectory(prefix='titanos-speech-stream-home-', ignore_cleanup_errors=True)
os.environ['TITANOS_HOME'] = HOME.name  # the keyed app's broker state stays out of ~/.titanos

import uvicorn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from titanos import server  # noqa: E402
from titanos.config import Config, ModelSpec  # noqa: E402
from titanos.engine import EngineError  # noqa: E402
from titanos.settings import Settings  # noqa: E402
from titanos.speech_stream import wav_header  # noqa: E402

RATE = 24000
RECORD = {'sample_rate': RATE, 'channels': 1, 'sample_width': 2}
CHUNKS = [bytes([index]) * 960 for index in (1, 2, 3)]
KEY = 'speech-stream-check-0001'
HEADER = wav_header(RATE)


def wav_bytes(frames):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(frames)
    return buffer.getvalue()


WAV = wav_bytes(b''.join(CHUNKS))


class SpeechRuntime:
    """The engine's runtime; the request text picks what it emits and how it ends."""

    def __init__(self):
        self.calls = []
        self.reset()

    def reset(self):
        self.started, self.cancelled, self.gate = threading.Event(), threading.Event(), threading.Event()

    def until_cancelled(self, cancel):
        if cancel is not None and cancel.wait(10):
            self.cancelled.set()
            raise EngineError('Request cancelled', 499, 'cancelled')
        raise AssertionError('the synthesis was never cancelled')

    def infer(self, task, data, emit=None, cancel=None):
        assert task == 'speech'
        self.calls.append(dict(data))
        mode = data['input']
        if mode == 'refuse':
            raise ValueError('voice host cannot say that')
        if emit is None or mode == 'whole':
            return WAV
        if mode == 'stall':
            self.started.set()
            self.until_cancelled(cancel)
        if mode == 'bad record':
            emit({**RECORD, 'sample_width': 4})
            return WAV
        emit(RECORD)
        emit(CHUNKS[0])
        if mode == 'odd':
            emit(b'\x00' * 3)
        elif mode == 'gated':
            self.started.set()
            if not self.gate.wait(10):
                raise AssertionError('the gate never opened')
        elif mode == 'hang':
            self.started.set()
            self.until_cancelled(cancel)
        elif mode == 'abort':
            emit(CHUNKS[1])
            raise RuntimeError('decoder fault')
        for chunk in CHUNKS[1:]:
            emit(chunk)
        return WAV


class Fixture:
    def __init__(self, api_key=None):
        self.temp = tempfile.TemporaryDirectory(prefix='titanos-speech-stream-', ignore_cleanup_errors=True)
        scratch = Path(self.temp.name)
        voice = scratch / 'host.wav'
        voice.write_bytes(wav_bytes(b'\x00\x01' * RATE))
        self.config = Config(models={
            'qwen': ModelSpec(path=str(scratch), task='speech', backend='qwen3-tts', voices={'host': str(voice)},
                              defaults={'temperature': 0.9, 'seed': 3}),
            'plain': ModelSpec(path=str(scratch), task='speech')})
        self.app = server.create_app(self.config, api_key=api_key, output_dir=scratch / 'out', settings=Settings(),
                                     runtime_catalog=False)
        self.runtime = SpeechRuntime()
        self.bind('qwen')

    def bind(self, name):
        engine = self.app.state.engine.target
        engine.runtime, engine.current, engine.states[name] = self.runtime, name, 'loaded'


def speech(**fields):
    return {'model': 'qwen', 'input': 'ok', 'voice': 'host', **fields}


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = Fixture()
        cls.client = TestClient(cls.fixture.app, base_url='http://127.0.0.1:8000')
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.fixture.temp.cleanup()

    def post(self, model='qwen', path='/v1/audio/speech', **fields):
        self.fixture.bind(model)
        return self.client.post(path, json=speech(model=model, **fields))

    def failed(self, response, status, text, code=None):
        self.assertEqual(response.status_code, status, response.text)
        error = response.json()['error']
        self.assertIn(text, error['message'])
        if code:
            self.assertEqual(error['code'], code)

    def test_header_is_gim_streaming_header(self):
        self.assertEqual(struct.unpack('<4sI4s4sIHHIIHH4sI', HEADER),
                         (b'RIFF', 0x7FFFFFFF, b'WAVE', b'fmt ', 16, 1, 1, RATE, 2 * RATE, 2, 16, b'data', 0x7FFFFFFF))

    def test_complete_and_streamed_bodies(self):
        whole = self.post()
        self.assertEqual((whole.status_code, whole.content, whole.headers['content-type']), (200, WAV, 'audio/wav'))
        pcm = self.post(response_format='pcm')
        self.assertEqual((pcm.content, pcm.headers['x-sample-rate'], pcm.headers['x-channels']),
                         (b''.join(CHUNKS), '24000', '1'))
        streamed = self.post(stream_format='audio')
        self.assertEqual(streamed.content, HEADER + b''.join(CHUNKS))
        self.assertEqual((streamed.headers['x-accel-buffering'], streamed.headers['x-bits-per-sample']), ('no', '16'))
        raw = self.post(stream_format='audio', response_format='pcm')
        self.assertEqual((raw.content, raw.headers['content-type']), (b''.join(CHUNKS), 'audio/pcm'))
        self.assertEqual(self.post(stream_format='audio', input='whole').content, WAV)  # no record: the whole body

    def test_engine_request(self):
        self.post(stream_format='audio')
        self.assertEqual(self.fixture.runtime.calls[-1], {'model': 'qwen', 'worker': None, 'seed': 3, 'input': 'ok',
                                                          'voice': 'host', 'response_format': 'wav',
                                                          'language': None, 'speed': 1.0})
        self.post(stream_format='audio', temperature=0.7, frame_budget=400)
        self.assertEqual({key: self.fixture.runtime.calls[-1].get(key) for key in ('temperature', 'frame_budget')},
                         {'temperature': 0.7, 'frame_budget': 400})

    def test_errors_before_audio_are_json(self):
        for fields in ({}, {'stream_format': 'audio'}):
            self.failed(self.post(input='refuse', **fields), 400, 'voice host cannot say that')
            self.failed(self.post('plain', temperature=0.7, **fields), 400, 'apply to qwen3-tts speech models only',
                        'unsupported_capability')
        self.failed(self.post(stream_format='audio', input='bad record'), 500, 'no valid stream record',
                    'inference_error')
        self.failed(self.post(stream_format='sse'), 400, 'relayed for cloud speech models only',
                    'unsupported_capability')
        for fields in ({'temperature': 3}, {'top_k': 1.5}, {'temperature': True}, {'frame_budget': 10},
                       {'response_format': 'mp3'}, {'stream_format': 'chunks'}):
            self.assertEqual(self.post(**fields).status_code, 422, fields)

    def test_jobs_and_elevenlabs_keep_whole_wav(self):
        refused = self.client.post('/api/jobs', json={'task': 'speech', **speech(stream_format='audio')})
        self.failed(refused, 422, 'A speech job writes a WAV file')
        self.fixture.bind('qwen')
        created = self.client.post('/api/jobs', json={'task': 'speech', **speech(input='whole')})
        self.assertEqual(created.status_code, 202, created.text)
        deadline = time.monotonic() + 30
        while (job := self.client.get(f"/api/jobs/{created.json()['id']}").json())['status'] not in (
                'completed', 'failed', 'cancelled', 'interrupted'):
            self.assertLess(time.monotonic(), deadline, job)
            time.sleep(0.1)
        self.assertEqual(job['status'], 'completed', job)
        self.assertNotIn('stream_format', self.fixture.runtime.calls[-1])
        self.assertEqual(self.fixture.runtime.calls[-1]['response_format'], 'wav')
        eleven = self.client.post('/v1/text-to-speech/host', json={'text': 'Hi.', 'model_id': 'qwen'})
        self.assertEqual((eleven.status_code, eleven.content), (200, WAV), eleven.text)
        self.failed(self.client.post('/v1/text-to-speech/host?output_format=mp3_44100',
                                     json={'text': 'Hi.', 'model_id': 'qwen'}), 400, 'output_format')


class Transfer(unittest.TestCase):
    """A keyed app on real uvicorn. The server starts before any assertLogs: uvicorn.Config runs dictConfig."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = Fixture(api_key=KEY)
        # the bound socket goes to uvicorn as it is: closing it and binding the port again lets another socket take it
        cls.sock = socket.socket()
        cls.sock.bind(('127.0.0.1', 0))
        cls.port = cls.sock.getsockname()[1]
        cls.server = uvicorn.Server(uvicorn.Config(cls.fixture.app, host='127.0.0.1', port=cls.port,
                                                   log_level='warning', lifespan='on'))
        cls.thread = threading.Thread(target=cls.server.run, kwargs={'sockets': [cls.sock]}, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 30
        while not cls.server.started:
            assert time.monotonic() < deadline and cls.thread.is_alive(), 'uvicorn did not start'
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(15)
        cls.sock.close()
        cls.fixture.temp.cleanup()

    def setUp(self):
        self.fixture.runtime.reset()

    def tearDown(self):
        connection, response = self.open('ok')  # every case leaves the server serving
        self.assertEqual(response.read(), HEADER + b''.join(CHUNKS))
        connection.close()

    def open(self, text, headers=True, key=KEY):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=20)
        connection.request('POST', '/v1/audio/speech', json.dumps(speech(input=text, stream_format='audio')),
                           {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
        return (connection, connection.getresponse()) if headers else (connection, None)

    def test_key_is_required(self):
        connection, response = self.open('ok', key='wrong-key-0000')
        self.assertEqual(response.status, 401, response.read())
        connection.close()

    def test_chunks_arrive_as_made(self):
        connection, response = self.open('gated')
        try:
            self.assertEqual(response.getheader('Transfer-Encoding'), 'chunked')
            self.assertEqual(response.read(len(HEADER) + len(CHUNKS[0])), HEADER + CHUNKS[0])  # before the gate opens
            self.assertTrue(self.fixture.runtime.started.is_set())
        finally:
            self.fixture.runtime.gate.set()
        self.assertEqual(response.read(), b''.join(CHUNKS[1:]))
        connection.close()

    def aborted(self, text, sent, warning):
        with self.assertLogs('titanos.speech_stream', 'WARNING') as stream_log, \
                self.assertLogs(level='ERROR') as root_log, self.assertLogs('uvicorn.error', 'ERROR') as uvicorn_log:
            connection, response = self.open(text)
            self.assertEqual(response.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as caught:
                response.read()  # one read: IncompleteRead.partial holds this call's bytes
            connection.close()
        self.assertEqual(caught.exception.partial, sent)
        self.assertIn(warning, '\n'.join(stream_log.output))
        self.assertIn('Unhandled Titanos request', '\n'.join(root_log.output))
        self.assertIn('Exception in ASGI application', '\n'.join(uvicorn_log.output))

    def test_failure_after_audio_aborts_the_transfer(self):
        self.aborted('abort', HEADER + CHUNKS[0] + CHUNKS[1], 'Inference failed: decoder fault')

    def test_partial_frame_aborts_the_transfer(self):
        self.aborted('odd', HEADER + CHUNKS[0], 'partial PCM frame')

    def test_disconnect_before_audio_cancels(self):
        connection, _ = self.open('stall', headers=False)
        self.assertTrue(self.fixture.runtime.started.wait(10))
        connection.close()
        self.assertTrue(self.fixture.runtime.cancelled.wait(10), 'a stalled synthesis outlived its client')

    def test_disconnect_mid_body_cancels(self):
        connection, response = self.open('hang')
        self.assertEqual(response.read(len(HEADER) + len(CHUNKS[0])), HEADER + CHUNKS[0])
        self.assertTrue(self.fixture.runtime.started.wait(10))
        response.close()
        connection.close()
        self.assertTrue(self.fixture.runtime.cancelled.wait(10), 'a streaming synthesis outlived its client')


if __name__ == '__main__':
    unittest.main(verbosity=1)
