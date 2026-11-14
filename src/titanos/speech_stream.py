"""Streaming /v1/audio/speech: the synthesis thread's audio as one chunked HTTP body.

The first item from the synthesis decides the response. An error before any audio is an ordinary JSON HTTP error. A
backend without incremental synthesis answers with its complete body (the plain WAV, or its PCM for "pcm"). A streaming
backend first emits its stream record ({sample_rate, channels, sample_width}), then PCM16 bytes. The "wav" form starts
with gim_api TTS Super's streaming header (RIFF and data chunk sizes 0x7FFFFFFF, the length being unknown); "pcm" is the
bare little-endian PCM16, its format in X-Sample-Rate, X-Bits-Per-Sample and X-Channels. A failure after audio was sent
aborts the transfer (the chunked body never terminates cleanly), as gim_api does, so a client cannot mistake a truncated
take for a whole one. A client disconnect cancels the synthesis; the request cleanup waits for its thread.
"""
import asyncio
import io
import logging
import queue
import struct
import threading
import wave
from contextvars import copy_context

from fastapi.responses import Response, StreamingResponse

from .engine import EngineError

STREAM_SIZE = 0x7FFFFFFF  # gim_api's streaming WAV sentinel for the RIFF and data chunk sizes
log = logging.getLogger(__name__)


def wav_header(rate, channels=1, width=2):
    """The 44-byte PCM WAV header of a stream whose length is unknown, as gim_api TTS Super sends it."""
    return struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', STREAM_SIZE, b'WAVE', b'fmt ', 16, 1, channels, rate,
                       rate * channels * width, channels * width, width * 8, b'data', STREAM_SIZE)


def wav_frames(wav):
    """The stream record and the PCM frames of a complete PCM16 WAV body."""
    try:
        with wave.open(io.BytesIO(wav), 'rb') as reader:
            record = {'sample_rate': reader.getframerate(), 'channels': reader.getnchannels(),
                      'sample_width': reader.getsampwidth()}
            frames = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError) as exc:
        raise EngineError(f'The speech backend returned an unreadable WAV: {exc}', 500, 'inference_error') from exc
    if record['sample_width'] != 2:
        raise EngineError('The speech backend returned audio that is not PCM16', 500, 'inference_error')
    return record, frames


def audio_headers(record):
    return {'X-Sample-Rate': str(record['sample_rate']), 'X-Bits-Per-Sample': str(8 * record['sample_width']),
            'X-Channels': str(record['channels']), 'Cache-Control': 'no-store'}


def complete(wav, response_format):
    """A complete synthesis in the requested format: the WAV body unchanged, or its PCM frames for "pcm"."""
    if response_format == 'wav':
        return Response(wav, media_type='audio/wav')
    record, frames = wav_frames(wav)
    return Response(frames, media_type='audio/pcm', headers=audio_headers(record))


def valid_record(value):
    return (isinstance(value, dict) and type(value.get('sample_rate')) is int and value['sample_rate'] > 0
            and value.get('channels') in (1, 2) and value.get('sample_width') == 2)


async def stream(engine, model, data, response_format, request):
    """The synthesis of data as a streamed response; the module docstring says what decides its shape."""
    from .server import _settled
    items, cancel = queue.Queue(maxsize=64), threading.Event()

    def put(item):
        while not cancel.is_set():
            try:
                items.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def run():
        try:
            put(('done', engine.infer(model, 'speech', data, lambda value: put(('audio', value)), cancel)))
        except BaseException as exc:
            put(('error', exc))

    worker = asyncio.get_running_loop().run_in_executor(None, copy_context().run, run)

    async def finish():
        cancel.set()
        await _settled(worker)

    request.scope.setdefault('titanos.cleanup', []).append(finish)

    async def next_item(first):
        while True:
            try:
                return await asyncio.to_thread(items.get, True, 0.25)
            except queue.Empty:
                if worker.done() and items.empty():
                    raise EngineError('Speech synthesis was cancelled', 499, 'cancelled') from None
                if first and await request.is_disconnected():
                    cancel.set()
                    raise EngineError('Request cancelled', 499, 'cancelled') from None

    kind, value = await next_item(True)
    if kind == 'error':
        raise value
    if kind == 'done':
        if isinstance(value, (bytes, bytearray)):
            return complete(bytes(value), response_format)
        raise EngineError('Speech synthesis finished without audio', 500, 'inference_error')
    if not valid_record(value):
        cancel.set()
        raise EngineError('The speech backend streamed no valid stream record', 500, 'inference_error')
    record = value
    frame = record['channels'] * record['sample_width']

    async def body():
        try:
            if response_format == 'wav':
                yield wav_header(record['sample_rate'], record['channels'], record['sample_width'])
            while True:
                kind, value = await next_item(False)
                if kind == 'done':
                    return
                if kind == 'error':
                    raise value
                if not isinstance(value, (bytes, bytearray)) or len(value) % frame:
                    raise EngineError('The speech backend streamed a partial PCM frame', 500, 'inference_error')
                if value:
                    yield bytes(value)
        except Exception as exc:
            log.warning('Speech stream aborted after its response started: %s', exc)
            raise
        finally:
            cancel.set()

    return StreamingResponse(body(), media_type='audio/wav' if response_format == 'wav' else 'audio/pcm',
                             headers={**audio_headers(record), 'X-Accel-Buffering': 'no'})
