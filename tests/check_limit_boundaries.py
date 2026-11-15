"""Just-under / just-over boundaries of the configurable limits (FAIL-02).

Every limit is lowered to a small declared value through Settings and driven in-process: fastapi TestClient over
create_app(Config(...)) with inert fixtures, a raw ASGI driver for the Guard's framing paths, and direct unit calls
for validators that need no route. The engine runtime is a recording stand-in bound as the resident runtime, so
Engine.infer's real validation and ValueError -> 400 conversion (engine.py) run while no model is loaded, no device is
touched and no server listener is started (the only sockets are asyncio's loopback self-pipe for the TestClient
portal and urllib3's IPv6 probe when titanos.server imports). Nothing forks: the FFmpeg child that media.read_audio
starts is replaced by an in-process stand-in (FakePopen) that decodes the PCM16 fixture with the wave module and
honours the command's -t, -ac and -ar arguments, so read_audio's own byte, duration and sample checks stay real; the
real FFmpeg decode stays covered by tests/check_media.py and tests/check_audio_decode.py. Every body stays far below
8 MiB (the largest is the 1 MiB multipart part) and every declared limit is small (the music reference-audio bound is
derived from max_body_bytes=3000, not the 64 MiB default).

Per limit: the just-under value is accepted (2xx, or the documented downstream typed error that is not the limit
refusal) and, where observable, reaches the recording runtime with its full declared size (no silent clipping); the
just-over value answers the typed refusal read from the source (exact status and code). Most refusals happen before
the runtime (schema or Guard). Two happen inside the runtime step, where the adapter's own helper refuses and
Engine.infer converts the ValueError to 400: tensor element counts (compiled.py calls tensor_payload) and decoded
audio duration/samples (asr.py calls media.read_audio). The stand-in calls those real helpers and records the
refusal, so those tests assert both that the call was admitted and that the helper refused it.

Requires numpy and pillow (installed by the numpy/pillow CI step); nothing beyond the base dependencies otherwise.

Limits that exist but are not enforced the Titanos way are kept visible as expected failures, not papered over:
- Starlette's default 1 MiB multipart part size answers 400 outside the Titanos error envelope (request.form() is
  called without arguments; server.py, api_plus.py);
- ExtensionPayload.timeout (api_plus.py:43) has no upper bound, so a caller can extend a deadline past
  TITANOS_CLIENT_TIMEOUT (a 10**12 ms value also makes the per-run threading.Timer raise OverflowError in its thread).

Bounded elsewhere and not repeated here (enforcing source; existing test): the Guard's body-read wait 503
server_busy and body-read deadline 408 request_timeout (server.py Guard; tests/check_security_public_routes.py);
VideoRequest.input_reference data-URL bound and pipeline_options size (server.py; tests/check_i2v.py); retrieval,
rerank and pair document/label counts and model-page fetch ceilings (tests/check_retrieval.py,
tests/check_pair_classification.py, tests/check_model_inspection.py); native queue
(tests/check_native_queue_settings.py); depth/segmentation budgets, diarization/VAD/rerank/sentiment token limits
and OCR budgets (their own check_*.py). Not covered anywhere in-process: the API Plus transcribe workflow's VAD
timing controls and post-decode budget checks (api_plus.py) need VAD and diarization model stand-ins; the API Plus
multipart file byte bound (api_plus.py, file.read(max_body_bytes + 1)) is shadowed over HTTP by the Guard's
max_body_bytes 413 covered here; uvicorn limits, the MCP body cap, artifact TTL, acquisition_max_bytes and
worker/native startup timeouts are process-level settings outside an in-process client.
"""
from pathlib import Path
import base64
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as exc:  # base dependencies do not include them; CI installs numpy and pillow first
    raise SystemExit(f'check_limit_boundaries needs numpy and pillow (the numpy/pillow CI install step): {exc}')
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from titanos import json_depth, server  # noqa: E402
from titanos import media  # noqa: E402
from titanos.config import Config, ModelSpec, WorkerSpec  # noqa: E402
from titanos.engine import EngineError  # noqa: E402
from titanos.settings import Settings, _active_settings  # noqa: E402

RATE = 16000
EMPTY = {'type': 'http.request', 'body': b'', 'more_body': False}


def png_bytes(width, height):
    out = io.BytesIO()
    Image.new('RGB', (width, height)).save(out, format='PNG')
    return out.getvalue()


def wav_bytes(frames, rate=RATE, channels=1):
    """Synthetic PCM16 WAV with exactly `frames` frames per channel (tiny: 2 bytes per sample)."""
    out = io.BytesIO()
    with wave.open(out, 'wb') as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b'\x10\x00' * (frames * channels))
    return out.getvalue()


PNG_8x8 = png_bytes(8, 8)  # 64 pixels; the byte limit of the small app is set to exactly this length
SMALL = Settings(
    max_body_bytes=4000,
    max_tokens=16, default_max_tokens=8, embedding_max_tokens=8,
    max_messages=3, max_tools=2, max_images=2, max_stop_sequences=4, max_stop_chars=6,
    embedding_max_batch=3, max_text_chars=20, max_speech_chars=10, max_prompt_chars=12,
    max_image_dimension=64, max_image_pixels=64 * 48, default_image_size='32x32',
    max_steps=5, default_steps=2, max_generated_images=2,
    max_input_image_bytes=len(PNG_8x8), max_input_image_pixels=64,
    max_tensor_elements=4,
    queue_capacity=1,
    max_audio_seconds=.01, diarization_min_duration_on=.001, diarization_min_duration_off=.001,
    max_video_frames=9, default_video_frames=4, max_video_fps=12, default_video_fps=6,
)
DATA_URL_LIMIT = 4 * ((SMALL.max_input_image_bytes + 2) // 3) + 128  # server.py Message.tool_message bound


class Runtime:
    """Recording resident runtime: Engine.infer's validation and ValueError handling stay real, nothing is loaded.

    `calls` records every admitted call; `refused` records the calls the adapter-side helper refused after admission
    (tensor_payload, media.read_audio), which is where the tensor and decoded-audio limits live in the real adapters."""

    def __init__(self):
        self.calls = []
        self.refused = []
        self.hold = None  # when set to an Event, speech blocks until it is released (asset-queue test)

    def infer(self, task, data, emit=None, cancel=None):
        self.calls.append((task, data))
        try:
            return self._infer(task, data)
        except ValueError as exc:
            self.refused.append((task, str(exc)))
            raise

    def _infer(self, task, data):
        if task == 'embeddings':
            return {'vectors': [[.1, .2] for _ in data['input']], 'usage': {'prompt_tokens': 2, 'total_tokens': 2}}
        if task == 'speech':
            if self.hold is not None:
                assert self.hold.wait(10), 'asset-queue fixture was not released'
            return media.write_wav(np.zeros(16, dtype=np.float32), RATE)
        if task in ('image', 'image-edit'):
            return [Image.new('RGB', (data['width'], data['height'])) for _ in range(data['n'])]
        if task == 'tensor':
            from titanos.compiled import tensor_payload  # the compiled runtime's own validator (compiled.py)
            values = {k: tensor_payload(v) for k, v in data['inputs'].items()}
            return {'outputs': {k: {'dtype': v.dtype.name, 'shape': list(v.shape), 'data': v.reshape(-1).tolist()}
                                for k, v in values.items()}}
        if task == 'transcription':
            decoded = media.read_audio(data['audio'], RATE)  # the ASR adapters' decode path (asr.py -> media.py)
            return f'{len(decoded)} samples'
        if task == 'music':
            return {'wav': media.write_wav(np.zeros(160, dtype=np.float32), RATE)}
        return {'text': 'ok', 'finish_reason': 'stop', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}

    def close(self):
        pass


FFMPEG_MARKER = 'ffmpeg-stand-in'
DECODE_REFUSAL = 'Audio is empty, non-finite, or exceeds configured duration/sample limits'


class FakePopen:
    """Stand-in for the FFmpeg child media.read_audio starts: decodes the PCM16 WAV on stdin with the wave module,
    downmixes to mono (-ac 1) and trims to -t seconds as FFmpeg would, and answers f32le on stdout. No process starts;
    read_audio's byte, duration and sample checks around it are the real ones."""

    def __init__(self, command, stdin=None, stdout=None, stderr=None, **kwargs):
        assert command[0] == FFMPEG_MARKER, command
        options = {command[i]: command[i + 1] for i in range(1, len(command) - 1) if command[i].startswith('-')}
        assert (options['-ac'], options['-f'], options['-acodec']) == ('1', 'f32le', 'pcm_f32le'), command
        rate, seconds = int(options['-ar']), float(options['-t'])
        with wave.open(io.BytesIO(stdin.read()), 'rb') as reader:
            assert reader.getframerate() == rate and reader.getsampwidth() == 2, 'fixture WAVs are PCM16 at the requested rate'
            frames = np.frombuffer(reader.readframes(reader.getnframes()), dtype='<i2').reshape(-1, reader.getnchannels())
        mono = frames.astype(np.float32).mean(axis=1) / 32768
        self.output = mono[:int(round(seconds * rate))].astype('<f4').tobytes()
        self.returncode = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def communicate(self, timeout=None):
        return self.output, b''

    def kill(self):
        pass


@contextlib.contextmanager
def fake_ffmpeg():
    """media.read_audio without a child process or the imageio_ffmpeg package: a stub module names the marker
    executable and titanos.media's `subprocess` global is swapped for a namespace whose Popen is FakePopen. Scoped to
    media.py's globals, so nothing else in the process sees it."""
    stub = types.ModuleType('imageio_ffmpeg')
    stub.get_ffmpeg_exe = lambda: FFMPEG_MARKER
    namespace = types.SimpleNamespace(Popen=FakePopen, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired,
                                      CREATE_NO_WINDOW=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    with patch.dict(sys.modules, {'imageio_ffmpeg': stub}), patch.object(media, 'subprocess', namespace):
        yield


def make_scope(method, path, headers=(), host='127.0.0.1:8000'):
    raw = [(b'host', host.encode('latin-1'))] + [(k.lower().encode('latin-1'), v.encode('latin-1')) for k, v in headers]
    return {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method, 'scheme': 'http',
            'path': path, 'raw_path': path.encode(), 'query_string': b'', 'root_path': '', 'headers': raw,
            'client': ('127.0.0.1', 50000), 'server': ('127.0.0.1', 8000)}


class Recorder:
    def __init__(self, chunks):
        self.calls = 0
        self.chunks = list(chunks)

    async def __call__(self):
        self.calls += 1
        return self.chunks.pop(0) if self.chunks else {'type': 'http.disconnect'}


async def call(app, scope, receive):
    sent = []

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    start = next(m for m in sent if m['type'] == 'http.response.start')
    headers = {k.decode('latin-1').lower(): v.decode('latin-1') for k, v in start['headers']}
    body = b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body')
    return start['status'], headers, body


def within(settings):
    """Context for direct validator calls: current_settings() reads the ContextVar (settings.py)."""
    class Scope:
        def __enter__(self):
            self.token = _active_settings.set(settings)

        def __exit__(self, *exc):
            _active_settings.reset(self.token)
    return Scope()


class Fixture:
    """One app per settings profile; the runtime is bound as resident for the model under test."""

    def __init__(self, models, settings, raise_server_exceptions=True):
        self.tmp = tempfile.TemporaryDirectory(prefix='titanos-limit-boundaries-')
        self.config = Config(models={name: ModelSpec(path=self.tmp.name, **spec) for name, spec in models.items()})
        self.app = server.create_app(self.config, output_dir=Path(self.tmp.name) / 'out', settings=settings, runtime_catalog=False)
        self.runtime = Runtime()
        self.engine = self.app.state.engine.target
        self.engine.runtime = self.runtime
        self.client = TestClient(self.app, base_url='http://127.0.0.1:8000', raise_server_exceptions=raise_server_exceptions)
        self.client.__enter__()

    def bind(self, name):
        self.engine.current = name
        self.engine.states[name] = 'loaded'

    def close(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()


class LimitBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small = Fixture({'chat': {'task': 'chat'}, 'vision': {'task': 'vision'}, 'embed': {'task': 'embeddings'},
                             'voice': {'task': 'speech'}, 'paint': {'task': 'image', 'backend': 'diffusers'},
                             'tens': {'task': 'tensor', 'backend': 'onnx'}, 'asr': {'task': 'transcription'}}, SMALL)
        # Process defaults: the Starlette multipart bound, JSON nesting and the music hard schema range need no lowering.
        cls.wide = Fixture({'embed': {'task': 'embeddings'}, 'voice': {'task': 'speech'}, 'asr': {'task': 'transcription'},
                            'music': {'task': 'music', 'backend': 'ace-step'}}, Settings(), raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls):
        cls.small.close()
        cls.wide.close()

    def setUp(self):
        self.mark = len(self.small.runtime.calls)
        self.wide_mark = len(self.wide.runtime.calls)

    # -- helpers -------------------------------------------------------------------------------------------------
    def calls(self, fixture=None):
        fixture = fixture or self.small
        mark = self.mark if fixture is self.small else self.wide_mark
        return fixture.runtime.calls[mark:]

    def assert_refused(self, reply, status, code, fragment=None):
        self.assertEqual(reply.status_code, status, reply.text)
        body = reply.json()
        self.assertEqual(body['error']['code'], code, body)
        if fragment:
            self.assertIn(fragment, body['error']['message'], body)
        return body

    def chat(self, **fields):
        self.small.bind('chat')
        payload = {'model': 'chat', 'messages': [{'role': 'user', 'content': 'hi'}]}
        payload.update(fields)
        return self.small.client.post('/v1/chat/completions', json=payload)

    def wait_job(self, fixture, identifier, seconds=10):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            record = fixture.client.get('/api/jobs/' + identifier).json()
            if record['status'] in {'completed', 'failed', 'cancelled'}:
                return record
            time.sleep(.02)
        self.fail('job did not settle: ' + json.dumps(record))

    # -- request body bytes: Guard (server.py Guard.__call__) --------------------------------------------------------
    def test_body_bytes_declared_length(self):
        """max_body_bytes: Content-Length == limit reaches the route with the full body; limit + 1 is 413 before receive."""
        fixture, limit = self.small, SMALL.max_body_bytes
        fixture.bind('embed')
        text = 't' * SMALL.max_text_chars
        body = json.dumps({'model': 'embed', 'input': [text]}).encode()
        body = body + b' ' * (limit - len(body))  # JSON tolerates trailing whitespace: exactly the limit
        self.assertEqual(len(body), limit)
        receive = Recorder([{'type': 'http.request', 'body': body, 'more_body': False}])
        scope = make_scope('POST', '/v1/embeddings', [('content-type', 'application/json'), ('content-length', str(limit))])
        status, headers, reply = fixture.client.portal.call(call, fixture.app, scope, receive)
        self.assertEqual(status, 200, reply)
        self.assertEqual(len(json.loads(reply)['data']), 1)
        self.assertEqual(self.calls()[-1][1]['input'], [text])  # the full text arrived; nothing was clipped
        over = body + b' '
        receive = Recorder([{'type': 'http.request', 'body': over, 'more_body': False}])
        scope = make_scope('POST', '/v1/embeddings', [('content-type', 'application/json'), ('content-length', str(limit + 1))])
        status, headers, reply = fixture.client.portal.call(call, fixture.app, scope, receive)
        self.assertEqual(status, 413, reply)
        self.assertEqual(json.loads(reply)['error']['code'], 'request_too_large')
        self.assertEqual(headers.get('connection'), 'close')
        self.assertEqual(receive.calls, 0)  # framing alone: not one byte was received
        self.assertEqual(len(self.calls()), 1)

    def test_body_bytes_chunked_accumulation(self):
        """max_body_bytes without Content-Length: chunks totalling the limit pass; one byte more is 413 after the read."""
        fixture, limit = self.small, SMALL.max_body_bytes
        fixture.bind('embed')
        text = 'u' * SMALL.max_text_chars
        body = json.dumps({'model': 'embed', 'input': [text]}).encode()
        body = body + b' ' * (limit - len(body))
        half = len(body) // 2
        chunks = [{'type': 'http.request', 'body': body[:half], 'more_body': True},
                  {'type': 'http.request', 'body': body[half:], 'more_body': False}]
        scope = make_scope('POST', '/v1/embeddings', [('content-type', 'application/json'), ('transfer-encoding', 'chunked')])
        receive = Recorder(chunks)
        status, headers, reply = fixture.client.portal.call(call, fixture.app, scope, receive)
        self.assertEqual(status, 200, reply)
        self.assertEqual(receive.calls, 2)
        self.assertEqual(self.calls()[-1][1]['input'], [text])
        over = [{'type': 'http.request', 'body': body[:half], 'more_body': True},
                {'type': 'http.request', 'body': body[half:] + b' ', 'more_body': True},
                {'type': 'http.request', 'body': b'never read', 'more_body': False}]
        receive = Recorder(over)
        status, headers, reply = fixture.client.portal.call(call, fixture.app, scope, receive)
        self.assertEqual(status, 413, reply)
        self.assertEqual(json.loads(reply)['error']['code'], 'request_too_large')
        self.assertEqual(receive.calls, 2)  # the read stopped at the first chunk that crossed the limit
        self.assertEqual(len(self.calls()), 1)

    # -- multipart: Starlette's default part size, not a Titanos setting -----------------------------------------------
    def test_multipart_field_size_default_bound(self):
        """A non-file part of exactly 1 MiB is parsed whole (the next typed refusal is the ASR prompt bound); 1 MiB + 1
        is Starlette's own 400 outside the Titanos envelope (request.form() takes the 1024 KB default)."""
        fixture = self.wide
        fixture.bind('asr')
        size = 1024 * 1024
        files = {'file': ('fixture.wav', wav_bytes(160), 'audio/wav')}
        reply = fixture.client.post('/v1/audio/transcriptions', data={'model': 'asr', 'prompt': 'x' * size}, files=files)
        self.assert_refused(reply, 400, 'invalid_request_error', 'TITANOS_MAX_PROMPT_CHARS')  # whole part reached ASRRequest
        reply = fixture.client.post('/v1/audio/transcriptions', data={'model': 'asr', 'prompt': 'x' * (size + 1)}, files=files)
        self.assertEqual(reply.status_code, 400, reply.text)
        self.assertEqual(reply.json(), {'detail': 'Part exceeded maximum size of 1024KB.'})
        self.assertEqual(self.calls(fixture), [])

    @unittest.expectedFailure
    def test_multipart_field_size_uses_titanos_envelope(self):
        """UNENFORCED (envelope): the oversize-part refusal is Starlette's {"detail": ...}, not {"error": {code}}."""
        fixture = self.wide
        fixture.bind('asr')
        reply = fixture.client.post('/v1/audio/transcriptions', data={'model': 'asr', 'prompt': 'x' * (1024 * 1024 + 1)},
                                    files={'file': ('fixture.wav', wav_bytes(160), 'audio/wav')})
        self.assertIn('error', reply.json(), 'multipart part-size refusal is outside the Titanos error envelope')

    # -- JSON nesting: no declared limit -----------------------------------------------------------------------------
    def test_json_nesting_shallow_is_typed(self):
        fixture = self.wide
        fixture.bind('embed')
        reply = fixture.client.post('/v1/embeddings', content=('[' * 100 + ']' * 100).encode(),
                                    headers={'content-type': 'application/json'})
        self.assert_refused(reply, 422, 'invalid_request_error')
        self.assertEqual(self.calls(fixture), [])

    def test_json_nesting_beyond_limit_is_refused_by_the_guard(self):
        """The Guard refuses JSON nested deeper than titanos.json_depth.JSON_MAX_DEPTH (256) with a typed 400 before any route parses
        it, on request.json() routes (/api/embed, protocols.py) and pydantic-bound routes alike. CPython's own limit is
        platform-dependent (a RecursionError 500 at depth ~3000 on Windows, accepted on Linux), so 5000 is refused
        everywhere now; depth 256 still reaches the route."""
        fixture = self.wide
        fixture.bind('embed')
        for path in ('/api/embed', '/v1/embeddings'):
            for opener, closer in (('[', ']'), ('{"a":', '}')):
                body = (opener * 5000 + '1' + closer * 5000).encode()
                reply = fixture.client.post(path, content=body, headers={'content-type': 'application/json'})
                self.assert_refused(reply, 400, 'invalid_request_error', 'JSON nesting exceeds 256 levels')
            limit = json_depth.JSON_MAX_DEPTH
            reply = fixture.client.post(path, content=('[' * limit + ']' * limit).encode(),
                                        headers={'content-type': 'application/json'})
            self.assertNotIn('JSON nesting', reply.text)
            self.assertLess(reply.status_code, 500, reply.text)
        self.assertEqual(self.calls(fixture), [])

    def test_json_depth_scanner(self):
        deep = json_depth.json_too_deep
        limit = json_depth.JSON_MAX_DEPTH
        self.assertFalse(deep(b'[' * limit + b']' * limit))
        self.assertTrue(deep(b'[' * (limit + 1) + b']' * (limit + 1)))
        self.assertTrue(deep(bytearray(b' \r\n\t' + b'{"k":' * (limit + 1) + b'0' + b'}' * (limit + 1))))
        self.assertFalse(deep(json.dumps({'s': '[' * 5000, 'e': '\\"[[[' * 400}).encode()), 'brackets inside strings')
        self.assertFalse(deep(json.dumps([[i, i] for i in range(50000)]).encode()), 'many shallow brackets')
        self.assertFalse(deep(b'--boundary\r\n' + b'[' * 5000), 'not a JSON document')
        self.assertFalse(deep(b''))

    # -- tokens, prompt, text, speech, messages, tools, stops, images: Payload bounds (server.py) ---------------------
    def test_max_tokens_ceiling(self):
        reply = self.chat(max_tokens=SMALL.max_tokens)
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(self.calls()[-1][1]['max_tokens'], SMALL.max_tokens)
        self.assert_refused(self.chat(max_tokens=SMALL.max_tokens + 1), 422, 'invalid_request_error',
                            f'max_tokens exceeds TITANOS_MAX_TOKENS={SMALL.max_tokens}')
        self.assert_refused(self.chat(max_completion_tokens=SMALL.max_tokens + 1), 422, 'invalid_request_error', 'TITANOS_MAX_TOKENS')
        self.assertEqual(len(self.calls()), 1)

    def test_prompt_chars(self):
        self.small.bind('chat')
        reply = self.small.client.post('/v1/completions', json={'model': 'chat', 'prompt': 'p' * SMALL.max_prompt_chars})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(len(self.calls()[-1][1]['prompt']), SMALL.max_prompt_chars)
        reply = self.small.client.post('/v1/completions', json={'model': 'chat', 'prompt': 'p' * (SMALL.max_prompt_chars + 1)})
        self.assert_refused(reply, 422, 'invalid_request_error', f'prompt exceeds TITANOS_MAX_PROMPT_CHARS={SMALL.max_prompt_chars}')
        self.assertEqual(len(self.calls()), 1)

    def test_text_chars_message_textpart_instructions(self):
        limit = SMALL.max_text_chars
        reply = self.chat(messages=[{'role': 'user', 'content': 'm' * limit}])
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(len(self.calls()[-1][1]['messages'][0]['content']), limit)
        self.assert_refused(self.chat(messages=[{'role': 'user', 'content': 'm' * (limit + 1)}]), 422, 'invalid_request_error',
                            'Message exceeds configured text limit')
        reply = self.chat(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'w' * limit}]}])
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assert_refused(self.chat(messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'w' * (limit + 1)}]}]),
                            422, 'invalid_request_error', 'Text part exceeds configured text limit')
        responses = {'model': 'chat', 'input': 'hi'}
        reply = self.small.client.post('/v1/responses', json={**responses, 'instructions': 'i' * limit})
        self.assertEqual(reply.status_code, 200, reply.text)
        reply = self.small.client.post('/v1/responses', json={**responses, 'instructions': 'i' * (limit + 1)})
        self.assert_refused(reply, 422, 'invalid_request_error', f'instructions exceeds TITANOS_MAX_TEXT_CHARS={limit}')
        self.assertEqual(len(self.calls()), 3)

    def test_speech_chars(self):
        self.small.bind('voice')
        limit = SMALL.max_speech_chars
        reply = self.small.client.post('/v1/audio/speech', json={'model': 'voice', 'input': 's' * limit})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertTrue(reply.content.startswith(b'RIFF'))
        self.assertEqual(len(self.calls()[-1][1]['input']), limit)
        reply = self.small.client.post('/v1/audio/speech', json={'model': 'voice', 'input': 's' * (limit + 1)})
        self.assert_refused(reply, 422, 'invalid_request_error', 'input exceeds configured text limit')
        self.assertEqual(len(self.calls()), 1)

    def test_messages_count(self):
        limit = SMALL.max_messages
        messages = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'm{i}'} for i in range(limit)]
        reply = self.chat(messages=messages)
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(len(self.calls()[-1][1]['messages']), limit)
        messages.append({'role': 'user', 'content': 'one more'})
        self.assert_refused(self.chat(messages=messages), 422, 'invalid_request_error', f'messages exceeds TITANOS_MAX_MESSAGES={limit}')
        self.assertEqual(len(self.calls()), 1)

    def test_tools_and_tool_calls(self):
        """Just-under tools pass the bound and meet the adapter's documented 400 unsupported_capability (not the limit);
        one more is the 422 limit refusal before any adapter check."""
        limit = SMALL.max_tools
        tool = {'type': 'function', 'function': {'name': 'f', 'parameters': {}}}
        self.assert_refused(self.chat(tools=[tool] * limit), 400, 'unsupported_capability')
        self.assert_refused(self.chat(tools=[tool] * (limit + 1)), 422, 'invalid_request_error', f'tools exceeds TITANOS_MAX_TOOLS={limit}')
        call_ = {'id': 'c', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}
        assistant = {'role': 'assistant', 'content': None, 'tool_calls': [call_] * limit}
        self.assert_refused(self.chat(messages=[{'role': 'user', 'content': 'hi'}, assistant]), 400, 'unsupported_capability')
        assistant = {**assistant, 'tool_calls': [call_] * (limit + 1)}
        self.assert_refused(self.chat(messages=[{'role': 'user', 'content': 'hi'}, assistant]), 422, 'invalid_request_error',
                            'Message tool calls exceed configured limit')
        self.assertEqual(self.calls(), [])

    def test_stop_sequences_and_chars(self):
        count, chars = SMALL.max_stop_sequences, SMALL.max_stop_chars
        stops = [chr(ord('a') + i) * chars for i in range(count)]
        reply = self.chat(stop=stops)
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(self.calls()[-1][1]['stop'], stops)
        self.assert_refused(self.chat(stop=stops + ['z']), 422, 'invalid_request_error', f'at most {count} nonempty strings')
        self.assert_refused(self.chat(stop='y' * (chars + 1)), 422, 'invalid_request_error', f'each at most {chars} characters')
        self.assertEqual(len(self.calls()), 1)

    def test_images_per_chat_and_data_url_length(self):
        self.small.bind('vision')
        url = 'data:image/png;base64,' + base64.b64encode(PNG_8x8).decode()
        part = {'type': 'image_url', 'image_url': {'url': url}}

        def vision(parts):
            return self.small.client.post('/v1/chat/completions', json={'model': 'vision', 'messages': [{'role': 'user', 'content': parts}]})
        reply = vision([part] * SMALL.max_images)
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(len(self.calls()[-1][1]['messages'][0]['content']), SMALL.max_images)
        self.assert_refused(vision([part] * (SMALL.max_images + 1)), 422, 'invalid_request_error',
                            'no more than the configured image count')
        padded = url + 'A' * (DATA_URL_LIMIT - len(url))  # exactly the max_input_image_bytes-derived URL bound
        self.assertEqual(len(padded), DATA_URL_LIMIT)
        reply = vision([{'type': 'image_url', 'image_url': {'url': padded}}])
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(len(self.calls()[-1][1]['messages'][0]['content'][0]['image_url']['url']), DATA_URL_LIMIT)
        self.assert_refused(vision([{'type': 'image_url', 'image_url': {'url': padded + 'A'}}]), 422, 'invalid_request_error',
                            'Image data URL exceeds configured image limit')
        self.assertEqual(len(self.calls()), 2)

    def test_responses_metadata_hard_bound(self):
        self.small.bind('chat')
        body = {'model': 'chat', 'input': 'hi'}
        reply = self.small.client.post('/v1/responses', json={**body, 'metadata': {f'k{i}': 'v' for i in range(16)}})
        self.assertEqual(reply.status_code, 200, reply.text)
        reply = self.small.client.post('/v1/responses', json={**body, 'metadata': {f'k{i}': 'v' for i in range(17)}})
        self.assert_refused(reply, 422, 'invalid_request_error', 'at most 16 pairs')
        reply = self.small.client.post('/v1/responses', json={**body, 'metadata': {'k' * 64: 'v' * 512}})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assert_refused(self.small.client.post('/v1/responses', json={**body, 'metadata': {'k' * 65: 'v'}}), 422,
                            'invalid_request_error', 'keys up to 64')
        self.assert_refused(self.small.client.post('/v1/responses', json={**body, 'metadata': {'k': 'v' * 513}}), 422,
                            'invalid_request_error', 'values up to 512')
        self.assertEqual(len(self.calls()), 2)

    # -- embeddings -----------------------------------------------------------------------------------------------
    def test_embedding_batch_and_text(self):
        fixture, limit = self.small, SMALL.embedding_max_batch
        fixture.bind('embed')
        texts = [f'text{i}' for i in range(limit)]
        reply = fixture.client.post('/v1/embeddings', json={'model': 'embed', 'input': texts})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(len(reply.json()['data']), limit)
        self.assertEqual(self.calls()[-1][1]['input'], texts)
        reply = fixture.client.post('/v1/embeddings', json={'model': 'embed', 'input': texts + ['extra']})
        self.assert_refused(reply, 422, 'invalid_request_error', 'Embedding batch exceeds configured count or text length')
        reply = fixture.client.post('/v1/embeddings', json={'model': 'embed', 'input': 'e' * SMALL.max_text_chars})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(self.calls()[-1][1]['input'], ['e' * SMALL.max_text_chars])
        # A string input meets Payload's own input bound first; a list element meets bounded_batch (server.py).
        reply = fixture.client.post('/v1/embeddings', json={'model': 'embed', 'input': 'e' * (SMALL.max_text_chars + 1)})
        self.assert_refused(reply, 422, 'invalid_request_error', 'input exceeds configured text limit')
        reply = fixture.client.post('/v1/embeddings', json={'model': 'embed', 'input': ['e' * (SMALL.max_text_chars + 1)]})
        self.assert_refused(reply, 422, 'invalid_request_error', 'Embedding batch exceeds configured count or text length')
        self.assertEqual(len(self.calls()), 2)

    def test_embedding_max_tokens_is_bounded_by_max_tokens_only(self):
        """embedding_max_tokens is documented as the default sequence bound (docs/CONFIGURATION.md), not a request
        ceiling: a request max_tokens above it is accepted unchanged; the ceiling that refuses is TITANOS_MAX_TOKENS."""
        fixture = self.small
        fixture.bind('embed')
        reply = fixture.client.post('/v1/embeddings', json={'model': 'embed', 'input': ['a'], 'max_tokens': SMALL.embedding_max_tokens + 1})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(self.calls()[-1][1]['max_tokens'], SMALL.embedding_max_tokens + 1)
        reply = fixture.client.post('/v1/embeddings', json={'model': 'embed', 'input': ['a'], 'max_tokens': SMALL.max_tokens + 1})
        self.assert_refused(reply, 422, 'invalid_request_error', f'max_tokens exceeds TITANOS_MAX_TOKENS={SMALL.max_tokens}')
        self.assertEqual(len(self.calls()), 1)

    # -- audio: bytes, duration, samples, channels (media.read_audio; ASR adapters decode through it) ------------------
    def test_audio_duration_samples_channels_unit(self):
        """read_audio's own checks with the FFmpeg child replaced in-process (FakePopen honours -t, -ac, -ar)."""
        limit = int(SMALL.max_audio_seconds * RATE)  # 160 samples
        with within(SMALL), fake_ffmpeg():
            decoded = media.read_audio(wav_bytes(limit), RATE)
            self.assertEqual(len(decoded), limit)  # every declared sample, none clipped
            with self.assertRaisesRegex(ValueError, DECODE_REFUSAL):
                media.read_audio(wav_bytes(limit + 1), RATE)
            # Channels are not a limit: stereo is downmixed to mono at the requested rate, frame count preserved.
            self.assertEqual(len(media.read_audio(wav_bytes(limit, channels=2), RATE)), limit)
            with self.assertRaisesRegex(ValueError, 'TITANOS_MAX_BODY_BYTES'):
                media.read_audio(b'R' * (SMALL.max_body_bytes + 1), RATE)  # refused before any decode
        with within(Settings(max_audio_seconds=1, max_audio_samples=100, diarization_min_duration_on=.001,
                             diarization_min_duration_off=.001)), fake_ffmpeg():
            self.assertEqual(len(media.read_audio(wav_bytes(100), RATE)), 100)
            with self.assertRaisesRegex(ValueError, DECODE_REFUSAL):
                media.read_audio(wav_bytes(101), RATE)

    def test_audio_duration_over_http(self):
        """This limit lives inside the runtime step: the ASR adapter decodes through media.read_audio (asr.py), whose
        ValueError Engine.infer converts to 400 invalid_request_error (engine.py). The over-limit upload is therefore
        admitted to the runtime and refused there; the accepted upload is decoded with its full sample count."""
        fixture, limit = self.small, int(SMALL.max_audio_seconds * RATE)
        fixture.bind('asr')
        with fake_ffmpeg():
            reply = fixture.client.post('/v1/audio/transcriptions', data={'model': 'asr'},
                                        files={'file': ('fixture.wav', wav_bytes(limit), 'audio/wav')})
            self.assertEqual(reply.status_code, 200, reply.text)
            self.assertEqual(reply.json()['text'], f'{limit} samples')
            refused = len(fixture.runtime.refused)
            reply = fixture.client.post('/v1/audio/transcriptions', data={'model': 'asr'},
                                        files={'file': ('fixture.wav', wav_bytes(limit + 1), 'audio/wav')})
            self.assert_refused(reply, 400, 'invalid_request_error', DECODE_REFUSAL)
            self.assertEqual(fixture.runtime.refused[refused:], [('transcription', DECODE_REFUSAL)])  # refused in the runtime step
            reply = fixture.client.post('/v1/audio/transcriptions', data={'model': 'asr'},
                                        files={'file': ('fixture.wav', wav_bytes(limit, channels=2), 'audio/wav')})
            self.assertEqual(reply.json()['text'], f'{limit} samples')  # stereo downmix, frame count preserved
        self.assertEqual(len(self.calls()), 3)  # 160 accepted, 161 admitted then refused by read_audio, stereo accepted

    def test_inline_audio_bytes_unit(self):
        """_inline_bytes (api_plus.py) reads the process environment: an inline payload above TITANOS_MAX_BODY_BYTES is
        413 request_too_large; exactly the limit decodes whole."""
        import os
        from unittest.mock import patch
        from titanos.api_plus import _inline_bytes
        with patch.dict(os.environ, {'TITANOS_MAX_BODY_BYTES': '96'}):
            raw = _inline_bytes(base64.b64encode(b'a' * 96).decode(), 'audio')
            self.assertEqual(len(raw), 96)
            with self.assertRaises(EngineError) as caught:
                _inline_bytes(base64.b64encode(b'a' * 97).decode(), 'audio')
            self.assertEqual((caught.exception.status, caught.exception.code), (413, 'request_too_large'))

    # -- images: generation dimensions/pixels/steps/count; input bytes/pixels on the edit route ----------------------
    def test_image_dimension_pixels_steps_count(self):
        fixture = self.small
        fixture.bind('paint')
        base = {'model': 'paint', 'prompt': 'p'}
        size = f'{SMALL.max_image_dimension}x{SMALL.max_image_pixels // SMALL.max_image_dimension}'  # 64x48: both bounds met
        reply = fixture.client.post('/v1/images/generations', json={**base, 'size': size, 'steps': SMALL.max_steps, 'n': SMALL.max_generated_images})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(len(reply.json()['data']), SMALL.max_generated_images)
        seen = self.calls()[-1][1]
        self.assertEqual((seen['width'], seen['height'], seen['steps'], seen['n']), (64, 48, SMALL.max_steps, SMALL.max_generated_images))
        for over in ('72x48', '64x56'):  # dimension over 64; pixels 3584 over 3072 while each side is within 64
            self.assert_refused(fixture.client.post('/v1/images/generations', json={**base, 'size': over}), 422, 'invalid_request_error',
                                'within configured dimension/pixel limits')
        self.assert_refused(fixture.client.post('/v1/images/generations', json={**base, 'steps': SMALL.max_steps + 1}), 422,
                            'invalid_request_error', f'steps exceeds TITANOS_MAX_STEPS={SMALL.max_steps}')
        self.assert_refused(fixture.client.post('/v1/images/generations', json={**base, 'n': SMALL.max_generated_images + 1}), 422,
                            'invalid_request_error', f'n exceeds TITANOS_MAX_GENERATED_IMAGES={SMALL.max_generated_images}')
        # The multipart edit route converts the same ImageRequest ValueError to 400 (server.py edit()).
        reply = fixture.client.post('/v1/images/edits', data={'model': 'paint', 'prompt': 'p', 'n': str(SMALL.max_generated_images + 1)},
                                    files={'image': ('source.png', PNG_8x8, 'image/png')})
        self.assert_refused(reply, 400, 'invalid_request_error', 'TITANOS_MAX_GENERATED_IMAGES')
        self.assertEqual(len(self.calls()), 1)

    def test_input_image_bytes_and_pixels_on_edit(self):
        fixture = self.small
        fixture.bind('paint')
        form = {'model': 'paint', 'prompt': 'p'}
        reply = fixture.client.post('/v1/images/edits', data=form, files={'image': ('source.png', PNG_8x8, 'image/png')})
        self.assertEqual(reply.status_code, 200, reply.text)  # exactly max_input_image_bytes and exactly 64 pixels
        seen = self.calls()[-1]
        self.assertEqual(seen[0], 'image-edit')
        self.assertEqual(seen[1]['image'].size, (8, 8))
        reply = fixture.client.post('/v1/images/edits', data=form, files={'image': ('source.png', PNG_8x8 + b'\0', 'image/png')})
        self.assert_refused(reply, 413, 'request_too_large', 'Source exceeds the configured image byte limit')
        reply = fixture.client.post('/v1/images/edits', data=form, files={'image': ('source.png', png_bytes(9, 8), 'image/png')})
        self.assertLessEqual(len(png_bytes(9, 8)), SMALL.max_input_image_bytes, 'fixture must exercise pixels, not bytes')
        self.assert_refused(reply, 400, 'invalid_request_error', 'TITANOS_MAX_INPUT_IMAGE_PIXELS')
        self.assertEqual(len(self.calls()), 1)

    # -- tensors (compiled.py tensor_payload through Engine.infer; TensorRequest schema) -----------------------------
    def test_tensor_elements_rank_and_input_count(self):
        fixture, limit = self.small, SMALL.max_tensor_elements
        fixture.bind('tens')
        tensor = {'dtype': 'float32', 'shape': [2, 2], 'data': [1, 2, 3, 4]}
        reply = fixture.client.post('/api/infer', json={'model': 'tens', 'inputs': {'x': tensor}})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(reply.json()['outputs']['x'], {'dtype': 'float32', 'shape': [2, 2], 'data': [1.0, 2.0, 3.0, 4.0]})
        over = {'dtype': 'float32', 'shape': [limit + 1], 'data': [1] * (limit + 1)}
        refused = len(fixture.runtime.refused)
        reply = fixture.client.post('/api/infer', json={'model': 'tens', 'inputs': {'x': over}})
        self.assert_refused(reply, 400, 'invalid_request_error', f'contain at most {limit} elements')
        # This limit lives inside the runtime step (compiled.py calls tensor_payload): the call is admitted, the
        # compiled runtime's own validator refuses it and Engine.infer converts the ValueError to 400.
        self.assertEqual(len(fixture.runtime.refused), refused + 1)
        self.assertIn(f'contain at most {limit} elements', fixture.runtime.refused[-1][1])
        one = {'dtype': 'float32', 'shape': [1], 'data': [1]}
        reply = fixture.client.post('/api/infer', json={'model': 'tens', 'inputs': {f'x{i}': one for i in range(64)}})
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(len(self.calls()[-1][1]['inputs']), 64)
        reply = fixture.client.post('/api/infer', json={'model': 'tens', 'inputs': {f'x{i}': one for i in range(65)}})
        self.assert_refused(reply, 422, 'invalid_request_error', 'at most 64 items')
        from titanos.compiled import tensor_payload
        with within(SMALL):
            self.assertEqual(tensor_payload({'dtype': 'int8', 'shape': [1] * 8, 'data': [7]}).shape, (1,) * 8)
            with self.assertRaisesRegex(ValueError, 'at most eight'):
                tensor_payload({'dtype': 'int8', 'shape': [1] * 9, 'data': [7]})
        # [2,2] accepted, [5] admitted then refused by tensor_payload, 64 inputs accepted; only the 65-inputs 422 is
        # a schema refusal before the runtime.
        self.assertEqual(len(self.calls()), 3)

    # -- video frames/fps (VideoRequest through Payload bounds) -------------------------------------------------------
    def test_video_frames_and_fps(self):
        """Validator-level (direct VideoRequest under the settings ContextVar): the HTTP 422 wrapping is the same
        Payload path the other routes prove; no video runtime stand-in is built."""
        with within(SMALL):
            request = server.VideoRequest(model='v', prompt='p', frames=SMALL.max_video_frames, fps=SMALL.max_video_fps)
            self.assertEqual((request.frames, request.fps), (SMALL.max_video_frames, SMALL.max_video_fps))
            with self.assertRaisesRegex(ValidationError, f'frames exceeds TITANOS_MAX_VIDEO_FRAMES={SMALL.max_video_frames}'):
                server.VideoRequest(model='v', prompt='p', frames=SMALL.max_video_frames + 1)
            with self.assertRaisesRegex(ValidationError, f'fps exceeds TITANOS_MAX_VIDEO_FPS={SMALL.max_video_fps}'):
                server.VideoRequest(model='v', prompt='p', fps=SMALL.max_video_fps + 1)

    # -- music: duration and output budgets (server.py MusicRequest; media.write_music_wav) ---------------------------
    def test_music_duration_settings_bound(self):
        """Validator-level (direct MusicRequest): the settings-driven duration, prompt/lyrics and reference-audio
        bounds. max_body_bytes is declared small so the reference string is ~4 KB, not the 64 MiB default."""
        settings = Settings(max_audio_seconds=20, max_body_bytes=3000, max_prompt_chars=12, max_text_chars=20)
        with within(settings):
            self.assertEqual(server.MusicRequest(model='m', prompt='p', duration=20).duration, 20)
            with self.assertRaisesRegex(ValidationError, 'Music duration exceeds TITANOS_MAX_AUDIO_SECONDS'):
                server.MusicRequest(model='m', prompt='p', duration=20.5)
            reference = 'A' * (4 * ((settings.max_body_bytes + 2) // 3) + 128)
            self.assertEqual(len(reference), 4128)  # the declared limit keeps the fixture small
            accepted = server.MusicRequest(model='m', prompt='p', duration=20, reference_audio_base64=reference)
            self.assertEqual(len(accepted.reference_audio_base64), len(reference))
            with self.assertRaisesRegex(ValidationError, 'reference audio exceeds the configured upload limit'):
                server.MusicRequest(model='m', prompt='p', duration=20, reference_audio_base64=reference + 'A')
            # Prompt and lyrics text bounds: Payload's own prompt bound (max_prompt_chars) fires first, so the prompt
            # clause of MusicRequest.music_limits (server.py) is shadowed; lyrics (max_text_chars) is music_limits' own.
            self.assertEqual(len(server.MusicRequest(model='m', prompt='p' * 12, duration=20).prompt), 12)
            with self.assertRaisesRegex(ValidationError, 'prompt exceeds TITANOS_MAX_PROMPT_CHARS=12'):
                server.MusicRequest(model='m', prompt='p' * 13, duration=20)
            song = server.MusicRequest(model='m', prompt='p', mode='song', lyrics='l' * 20, duration=20)
            self.assertEqual(len(song.lyrics), 20)
            with self.assertRaisesRegex(ValidationError, 'Music prompt or lyrics exceed configured text limits'):
                server.MusicRequest(model='m', prompt='p', mode='song', lyrics='l' * 21, duration=20)

    def test_music_duration_hard_schema_over_http(self):
        fixture = self.wide
        fixture.bind('music')
        reply = fixture.client.post('/v1/music/generations', json={'model': 'music', 'prompt': 'p', 'duration': 600})
        self.assertEqual(reply.status_code, 202, reply.text)
        record = self.wait_job(fixture, reply.json()['id'])
        self.assertEqual(record['status'], 'completed', record)
        self.assertEqual(self.calls(fixture)[-1][1]['duration'], 600)
        reply = fixture.client.post('/v1/music/generations', json={'model': 'music', 'prompt': 'p', 'duration': 600.5})
        self.assert_refused(reply, 422, 'invalid_request_error', 'less than or equal to 600')
        reply = fixture.client.post('/v1/music/generations', json={'model': 'music', 'prompt': 'p', 'duration': 9.5})
        self.assert_refused(reply, 422, 'invalid_request_error', 'greater than or equal to 10')
        self.assertEqual(len(self.calls(fixture)), 1)

    def test_music_output_budgets_unit(self):
        """Unit-level (direct media.write_music_wav): the three output bounds refuse with a ValueError before any
        encoding; the job path's failed-status wrapping is not observed here."""
        samples = np.zeros((2, 400), dtype=np.float32)  # 800 samples, 400 frames -> 44 + 1600 bytes
        with within(Settings(max_music_samples=800)):
            self.assertEqual(len(media.write_music_wav(samples, 8000)), 1644)
        with within(Settings(max_music_samples=799)):
            with self.assertRaisesRegex(ValueError, 'exceeds configured audio limits'):
                media.write_music_wav(samples, 8000)
        with within(Settings(max_music_output_bytes=1644)):
            self.assertEqual(len(media.write_music_wav(samples, 8000)), 1644)
        with within(Settings(max_music_output_bytes=1643)):
            with self.assertRaisesRegex(ValueError, 'exceeds configured audio limits'):
                media.write_music_wav(samples, 8000)
        # Frames are also bounded by max_audio_seconds * rate (media.py): 400 frames at 8000 Hz is exactly .05 s.
        tiny = dict(diarization_min_duration_on=.001, diarization_min_duration_off=.001)
        with within(Settings(max_audio_seconds=.05, **tiny)):
            self.assertEqual(len(media.write_music_wav(samples, 8000)), 1644)
        with within(Settings(max_audio_seconds=.049, **tiny)):
            with self.assertRaisesRegex(ValueError, 'exceeds configured audio limits'):
                media.write_music_wav(samples, 8000)

    # -- queues: asset jobs (jobs.py) and replica admission (replicas.py) ----------------------------------------------
    def test_asset_queue_capacity(self):
        """queue_capacity=1 with an inert speech job that blocks until released: the second submission is 429
        queue_full without reaching the runtime; after release the slot is reusable."""
        fixture = self.small
        fixture.bind('voice')
        hold = threading.Event()
        fixture.runtime.hold = hold
        try:
            first = fixture.client.post('/api/jobs', json={'model': 'voice', 'task': 'speech', 'input': 'one'})
            self.assertEqual(first.status_code, 202, first.text)
            until = time.monotonic() + 10
            while len(self.calls()) < 1 and time.monotonic() < until:
                time.sleep(.01)
            self.assertEqual(len(self.calls()), 1, 'first job did not start')
            second = fixture.client.post('/api/jobs', json={'model': 'voice', 'task': 'speech', 'input': 'two'})
            self.assert_refused(second, 429, 'queue_full', 'Asset queue is full')
            self.assertEqual(len(self.calls()), 1)
        finally:
            hold.set()
            fixture.runtime.hold = None
        record = self.wait_job(fixture, first.json()['id'])
        self.assertEqual(record['status'], 'completed', record)
        self.assertIn('/v1/files/', record['result']['url'])
        third = fixture.client.post('/api/jobs', json={'model': 'voice', 'task': 'speech', 'input': 'three'})
        self.assertEqual(third.status_code, 202, third.text)
        self.assertEqual(self.wait_job(fixture, third.json()['id'])['status'], 'completed')
        self.assertEqual(len(self.calls()), 2)

    def test_replica_queue_capacity(self):
        from concurrent.futures import ThreadPoolExecutor
        from titanos.replicas import ReplicaEngine
        config = Config(models={'a': ModelSpec(path='.', task='image', backend='diffusers', device='cpu')},
                        workers={'a': WorkerSpec(models=['a'], device='cpu')})
        engine = ReplicaEngine(config, Settings(replica_queue_capacity=1, replica_queue_timeout=5))

        def waiter():
            with engine.acquire('a'):
                return 'ok'
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                with engine.acquire('a'):  # the one worker is busy; no process is spawned by acquire
                    future = pool.submit(waiter)
                    until = time.monotonic() + 5
                    while time.monotonic() < until:
                        with engine.condition:
                            if engine.pending == 1:
                                break
                        time.sleep(.001)
                    self.assertEqual(engine.pending, 1, 'waiter was not admitted to the queue')
                    with self.assertRaises(EngineError) as caught:
                        with engine.acquire('a'):
                            pass
                    self.assertEqual((caught.exception.status, caught.exception.code), (429, 'rate_limit_error'))
                self.assertEqual(future.result(timeout=5), 'ok')
            self.assertEqual(engine.pending, 0)
            self.assertTrue(all(w.process is None for w in engine.workers))
        finally:
            engine.close()

    # -- query and header bounds --------------------------------------------------------------------------------------
    def test_list_limit_and_idempotency_key_length(self):
        fixture = self.wide
        fixture.bind('voice')
        self.assertEqual(fixture.client.get('/api/jobs?limit=100').status_code, 200)
        self.assert_refused(fixture.client.get('/api/jobs?limit=101'), 422, 'invalid_request_error', 'less than or equal to 100')
        body = {'model': 'voice', 'task': 'speech', 'input': 'key'}
        reply = fixture.client.post('/api/jobs', json=body, headers={'Idempotency-Key': 'k' * 128})
        self.assertEqual(reply.status_code, 202, reply.text)
        self.assertEqual(self.wait_job(fixture, reply.json()['id'])['status'], 'completed')
        self.assert_refused(fixture.client.post('/api/jobs', json=body, headers={'Idempotency-Key': 'k' * 129}), 400,
                            'invalid_request_error', '1 to 128 characters')
        self.assertEqual(len(self.calls(fixture)), 1)

    # -- extension deadline ---------------------------------------------------------------------------------------------
    def test_extension_timeout_within_client_timeout(self):
        fixture = self.small
        fixture.bind('asr')
        body = {'operation': 'audio_transcription', 'model': 'asr', 'audio_base64': base64.b64encode(wav_bytes(160)).decode(),
                'timeout': int(Settings().client_timeout * 1000)}
        with fake_ffmpeg():
            reply = fixture.client.post('/v1/extensions/infer', json=body)
        self.assertEqual(reply.status_code, 200, reply.text)
        self.assertEqual(reply.json()['text'], '160 samples')

    @unittest.expectedFailure
    def test_extension_timeout_above_client_timeout_is_refused(self):
        """UNENFORCED: ExtensionPayload.timeout (api_plus.py:43) is gt=0 with no ceiling; client_timeout*1000 + 1 ms
        is accepted and extends the deadline past TITANOS_CLIENT_TIMEOUT."""
        fixture = self.small
        fixture.bind('asr')
        body = {'operation': 'audio_transcription', 'model': 'asr', 'audio_base64': base64.b64encode(wav_bytes(160)).decode(),
                'timeout': int(Settings().client_timeout * 1000) + 1}
        with fake_ffmpeg():
            reply = fixture.client.post('/v1/extensions/infer', json=body)
        self.assertEqual(reply.status_code, 422, 'timeout above TITANOS_CLIENT_TIMEOUT was accepted: ' + reply.text[:120])

    # -- settings-level coherence ---------------------------------------------------------------------------------------
    def test_defaults_may_not_exceed_ceilings(self):
        Settings(max_tokens=16, default_max_tokens=16)
        with self.assertRaisesRegex(ValueError, 'TITANOS_DEFAULT_MAX_TOKENS exceeds TITANOS_MAX_TOKENS'):
            Settings(max_tokens=16, default_max_tokens=17)
        Settings(max_steps=5, default_steps=5)
        with self.assertRaisesRegex(ValueError, 'TITANOS_DEFAULT_STEPS exceeds TITANOS_MAX_STEPS'):
            Settings(max_steps=5, default_steps=6)
        with self.assertRaisesRegex(ValueError, 'within configured dimension/pixel limits'):
            Settings(max_image_dimension=64, max_image_pixels=64 * 48, default_image_size='64x56')


if __name__ == '__main__':
    unittest.main(verbosity=2)
