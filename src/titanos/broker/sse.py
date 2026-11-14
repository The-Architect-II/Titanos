"""Incremental SSE and NDJSON parsing and writing over raw httpx byte streams.

- SSE framing follows the WHATWG event-stream rules: lines end in CRLF, LF or CR (a CR at a chunk boundary waits for the
  next chunk), 'data:' lines join with '\\n', one optional space after the colon is removed, comment lines (':') and
  unknown fields are ignored, a blank line dispatches the frame, and a frame without data is not dispatched. An incomplete
  trailing frame at end of stream is discarded (parse_sse) but its bytes still pass through tee and first_frame replays.
- Lines are decoded as UTF-8 only once complete, so a multi-byte character split across chunks is never broken.
- Limits: MAX_LINE (1 MiB) per line and MAX_FRAMES (100,000) per stream; exceeding either, or a line that is not valid
  UTF-8, raises BrokerError upstream_bad_response (502) wherever the broker must parse the stream (translation, the
  first-frame hold, include_usage stripping). The observe-only usage tee never fails a passthrough stream on these limits
  (a single base64 image frame may exceed 1 MiB); it stops observing and keeps forwarding the bytes.
- Every stream object returned here (parse_sse, parse_ndjson, tee_usage, first_frame's replay) is a ManagedStream whose
  aclose() closes its source even when it was never iterated, so the upstream response and its egress slot are released
  deterministically when a client goes away before the first byte.
- Usage is normalized to the IR (OpenAI) meaning: input_tokens includes cached input, output_tokens includes reasoning
  (anthropic_usage and gemini_usage document the provider differences).
- tee_usage observes a passthrough stream without altering it: the client receives the upstream chunks byte for byte while
  a parser reads a copy for usage. The one documented exception: an include_usage chunk the broker injected (and the client
  did not ask for) is removed, which re-frames the stream; every other frame is re-emitted byte-identically.
- first_frame reads and holds the first complete frame (SSE frame, NDJSON line, or the first binary chunk, at most 64 KiB)
  so the caller can decide to commit (or fall over on an error-first stream) before the client sees a byte; the returned
  iterator replays every byte read so far, then the rest.
- 'openai_media' is the OpenAI media event-stream family (images/generations and images/edits with stream=true:
  image_generation.* and image_edit.* events whose b64_json frames are megabytes; audio/transcriptions with stream=true:
  transcript.text.*; audio/speech with stream_format=sse: speech.audio.*). Its lines may be up to MEDIA_MAX_LINE, so the
  first-frame hold and the usage tee read whole partial images; usage comes from the *.completed, transcript.text.done
  and speech.audio.done events (and every *.completed event counts one image).
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .errors import BrokerError
from .types import Usage

MAX_LINE = 1024 * 1024
MAX_FRAMES = 100_000
BINARY_FIRST_FRAME = 64 * 1024
KEEPALIVE = b': keepalive\n\n'
SSE_PROTOCOLS = frozenset({'openai', 'anthropic', 'gemini', 'responses', 'openai_media'})
# per-line limit of the OpenAI media streams (a partial or final image is one base64 data line); the egress media cap
# (broker_max_upstream_media_bytes) still bounds the whole answer
MEDIA_MAX_LINE = 64 * 1024 * 1024
MEDIA_PROTOCOLS = frozenset({'openai_media'})


# Gemini alt=sse frames are whole GenerateContentResponse objects: an image-output or TTS frame is one base64 data
# line of several MiB, so the Gemini first-frame hold and usage tee take the media line limit too
LARGE_LINE_PROTOCOLS = MEDIA_PROTOCOLS | frozenset({'gemini'})


def _splitter(protocol: str) -> 'SSESplitter':
    return SSESplitter(max_line=MEDIA_MAX_LINE) if protocol in LARGE_LINE_PROTOCOLS else SSESplitter()
NDJSON_PROTOCOLS = frozenset({'ollama'})
BINARY_PROTOCOLS = frozenset({'elevenlabs', 'binary', 'audio'})


@dataclass(frozen=True)
class SSEFrame:
    event: str | None
    data: str
    raw: bytes

    def json(self):
        """The data parsed as JSON; BrokerError upstream_bad_response when it is not."""
        try:
            return json.loads(self.data)
        except ValueError:
            raise _bad('The upstream sent an event whose data is not JSON') from None


def _bad(message: str) -> BrokerError:
    return BrokerError(message, 'upstream_bad_response', origin='upstream')


class _Block:
    __slots__ = ('raw', 'event', 'data', 'has_data')

    def __init__(self):
        self.raw = bytearray()
        self.event: str | None = None
        self.data: list[str] = []
        self.has_data = False


class SSESplitter:
    """Byte-level SSE splitter: feed(chunk) returns the completed blocks as (raw bytes, SSEFrame | None); None is a block
    that dispatches nothing (comments, keepalives, a block without data). finish() returns the incomplete tail bytes."""

    def __init__(self, *, max_line: int = MAX_LINE, max_frames: int = MAX_FRAMES):
        self.max_line, self.max_frames = max_line, max_frames
        self._buf = bytearray()
        self._block = _Block()
        self.frames = 0

    def feed(self, chunk: bytes) -> list[tuple[bytes, SSEFrame | None]]:
        out: list[tuple[bytes, SSEFrame | None]] = []
        self._buf += chunk
        while True:
            lf, cr = self._buf.find(b'\n'), self._buf.find(b'\r')
            if lf < 0 and cr < 0:
                if len(self._buf) > self.max_line:
                    raise _bad('The upstream sent an event-stream line over the size limit')
                return out
            if cr >= 0 and (lf < 0 or cr < lf):
                if cr == len(self._buf) - 1:
                    if cr > self.max_line:
                        raise _bad('The upstream sent an event-stream line over the size limit')
                    return out  # a CR at the chunk end may be the first half of CRLF
                end, cut = cr, cr + (2 if self._buf[cr + 1:cr + 2] == b'\n' else 1)
            else:
                end, cut = lf, lf + 1
            if end > self.max_line:
                raise _bad('The upstream sent an event-stream line over the size limit')
            line = bytes(self._buf[:end])
            self._block.raw += self._buf[:cut]
            del self._buf[:cut]
            done = self._line(line)
            if done is not None:
                out.append(done)

    def _line(self, line: bytes):
        block = self._block
        if not line:
            self._block = _Block()
            if not block.has_data:
                return bytes(block.raw), None
            self.frames += 1
            if self.frames > self.max_frames:
                raise _bad('The upstream stream exceeded the frame limit')
            return bytes(block.raw), SSEFrame(block.event, '\n'.join(block.data), bytes(block.raw))
        try:
            text = line.decode('utf-8')
        except UnicodeDecodeError:
            raise _bad('The upstream sent an event-stream line that is not UTF-8') from None
        if text.startswith(':'):
            return None
        name, sep, value = text.partition(':')
        if sep and value.startswith(' '):
            value = value[1:]
        if name == 'data':
            block.data.append(value)
            block.has_data = True
        elif name == 'event':
            block.event = value or None
        return None

    def finish(self) -> bytes:
        tail = bytes(self._block.raw) + bytes(self._buf)
        self._block, self._buf = _Block(), bytearray()
        return tail


class NDJSONSplitter:
    """Line splitter for NDJSON: feed(chunk) returns (raw line bytes, parsed dict | None for a blank line)."""

    def __init__(self, *, max_line: int = MAX_LINE, max_frames: int = MAX_FRAMES):
        self.max_line, self.max_frames = max_line, max_frames
        self._buf = bytearray()
        self.frames = 0

    def feed(self, chunk: bytes) -> list[tuple[bytes, dict | None]]:
        out = []
        self._buf += chunk
        while True:
            index = self._buf.find(b'\n')
            if index < 0:
                if len(self._buf) > self.max_line:
                    raise _bad('The upstream sent an NDJSON line over the size limit')
                return out
            if index > self.max_line:
                raise _bad('The upstream sent an NDJSON line over the size limit')
            raw = bytes(self._buf[:index + 1])
            del self._buf[:index + 1]
            out.append((raw, self._parse(raw)))

    def _parse(self, raw: bytes) -> dict | None:
        text = raw.strip()
        if not text:
            return None
        self.frames += 1
        if self.frames > self.max_frames:
            raise _bad('The upstream stream exceeded the frame limit')
        try:
            value = json.loads(text.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise _bad('The upstream sent an NDJSON line that is not JSON') from None
        if not isinstance(value, dict):
            raise _bad('The upstream sent an NDJSON line that is not an object')
        return value

    def finish(self) -> list[tuple[bytes, dict | None]]:
        """The last line when the stream ended without a newline."""
        raw, self._buf = bytes(self._buf), bytearray()
        return [(raw, self._parse(raw))] if raw.strip() else ([(raw, None)] if raw else [])


def parse_sse(aiter, *, max_line: int = MAX_LINE, max_frames: int = MAX_FRAMES) -> AsyncIterator[SSEFrame]:
    """SSE frames from an async iterator of bytes (comments and data-less blocks are skipped). aclose() closes aiter,
    also before the first frame was pulled."""
    async def frames():
        splitter = SSESplitter(max_line=max_line, max_frames=max_frames)
        try:
            async for chunk in aiter:
                for _, frame in splitter.feed(chunk):
                    if frame is not None:
                        yield frame
            splitter.finish()  # an incomplete trailing frame is discarded, as the event-stream rules require
        finally:
            await _aclose(aiter)
    return ManagedStream(frames(), _closer(aiter))


def parse_ndjson(aiter, *, max_line: int = MAX_LINE, max_frames: int = MAX_FRAMES) -> AsyncIterator[dict]:
    """JSON objects, one per non-blank line. aclose() closes aiter, also before the first line was pulled."""
    async def lines():
        splitter = NDJSONSplitter(max_line=max_line, max_frames=max_frames)
        try:
            async for chunk in aiter:
                for _, value in splitter.feed(chunk):
                    if value is not None:
                        yield value
            for _, value in splitter.finish():
                if value is not None:
                    yield value
        finally:
            await _aclose(aiter)
    return ManagedStream(lines(), _closer(aiter))


def write_sse(data: str, event: str | None = None) -> bytes:
    """One SSE frame: 'event: <name>' (when given), one 'data:' line per line of data, then a blank line."""
    head = f'event: {event}\n' if event else ''
    lines = str(data).replace('\r\n', '\n').replace('\r', '\n').split('\n')
    return (head + ''.join(f'data: {line}\n' for line in lines) + '\n').encode('utf-8')


def write_json_sse(value, event: str | None = None) -> bytes:
    return write_sse(json.dumps(value, ensure_ascii=False, separators=(',', ':')), event)


# ---------------------------------------------------------------------------------------------------------------------
# usage observation

def _int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def openai_usage(usage) -> Usage | None:
    """Usage from an OpenAI chat/completions/embeddings usage object (prompt/completion tokens and details)."""
    if not isinstance(usage, dict):
        return None
    prompt = usage.get('prompt_tokens_details') or {}
    completion = usage.get('completion_tokens_details') or {}
    prompt = prompt if isinstance(prompt, dict) else {}
    completion = completion if isinstance(completion, dict) else {}
    return Usage(input_tokens=_int(usage.get('prompt_tokens')), output_tokens=_int(usage.get('completion_tokens')),
                 cached_input_tokens=_int(prompt.get('cached_tokens')),
                 reasoning_tokens=_int(completion.get('reasoning_tokens')),
                 audio_input_tokens=_int(prompt.get('audio_tokens')) or None,
                 audio_output_tokens=_int(completion.get('audio_tokens')) or None,
                 image_input_tokens=_int(prompt.get('image_tokens')) or None)


def openai_media_usage(usage, *, images: int | None = None) -> Usage | None:
    """Usage from an OpenAI image (images API, image_generation.completed), transcription (type tokens or duration) or
    speech (speech.audio.done) usage object: input_tokens_details / input_token_details carry the image, text and audio
    token parts."""
    if not isinstance(usage, dict):
        return Usage(images=images) if images else None
    if usage.get('type') == 'duration':
        seconds = usage.get('seconds')
        return Usage(audio_seconds=seconds if isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                     else None, images=images)
    details = usage.get('input_tokens_details') or usage.get('input_token_details') or {}
    details = details if isinstance(details, dict) else {}
    return Usage(input_tokens=_int(usage.get('input_tokens')), output_tokens=_int(usage.get('output_tokens')),
                 images=images, audio_input_tokens=_int(details.get('audio_tokens')) or None,
                 image_input_tokens=_int(details.get('image_tokens')) or None)


def responses_usage(usage) -> Usage | None:
    if not isinstance(usage, dict):
        return None
    inp = usage.get('input_tokens_details') or {}
    out = usage.get('output_tokens_details') or {}
    return Usage(input_tokens=_int(usage.get('input_tokens')), output_tokens=_int(usage.get('output_tokens')),
                 cached_input_tokens=_int(inp.get('cached_tokens')) if isinstance(inp, dict) else None,
                 reasoning_tokens=_int(out.get('reasoning_tokens')) if isinstance(out, dict) else None)


def responses_object_usage(response) -> Usage | None:
    """Usage of a whole Responses object: its usage plus the hosted web searches it ran (output items of type
    web_search_call, billed per call: Usage.web_search_requests, PriceRow per_web_search_request)."""
    if not isinstance(response, dict):
        return None
    usage = responses_usage(response.get('usage'))
    output = response.get('output') if isinstance(response.get('output'), list) else []
    searches = sum(1 for item in output if isinstance(item, dict) and item.get('type') == 'web_search_call')
    if not searches:
        return usage
    from dataclasses import replace
    return replace(usage or Usage(), web_search_requests=searches)


def anthropic_usage(usage, base: Usage | None = None) -> Usage | None:
    """Usage in the IR (OpenAI) meaning from an Anthropic usage object; later values override base.

    Anthropic's input_tokens counts only the uncached input, and cache_read_input_tokens and cache_creation_input_tokens
    are separate and additive. The IR's input_tokens includes every input token (pricing bills cached_input_tokens at
    the cached price and the rest at the input price), so input_tokens = input + cache_read + cache_creation,
    cached_input_tokens = cache_read and cache_write_tokens = cache_creation (billed at the cache-write price, R18).
    A message_delta that repeats only some of the input fields never lowers the input already seen (the input of one
    message is fixed). Billing details: cache_creation.ephemeral_1h_input_tokens (the 1-hour-TTL part of the cache
    writes) -> cache_write_1h_tokens, server_tool_use.web_search_requests -> web_search_requests, speed 'fast' -> fast."""
    if not isinstance(usage, dict):
        return base
    base = base or Usage()
    uncached = _int(usage.get('input_tokens'))
    read = _int(usage.get('cache_read_input_tokens'))
    write = _int(usage.get('cache_creation_input_tokens'))
    output = _int(usage.get('output_tokens'))
    cached = read if read is not None else base.cached_input_tokens
    if uncached is None and read is None and write is None:
        total = base.input_tokens
    else:
        total = (uncached or 0) + (cached or 0) + (write or 0)
        if base.input_tokens is not None:
            total = max(total, base.input_tokens)
    creation = usage.get('cache_creation') if isinstance(usage.get('cache_creation'), dict) else {}
    write_1h = _int(creation.get('ephemeral_1h_input_tokens'))
    tools = usage.get('server_tool_use') if isinstance(usage.get('server_tool_use'), dict) else {}
    searches = _int(tools.get('web_search_requests'))
    speed = usage.get('speed')
    return Usage(input_tokens=total, output_tokens=output if output is not None else base.output_tokens,
                 cached_input_tokens=cached, reasoning_tokens=base.reasoning_tokens,
                 cache_write_tokens=write if write is not None else base.cache_write_tokens,
                 cache_write_1h_tokens=(write_1h or None) if write_1h is not None else base.cache_write_1h_tokens,
                 web_search_requests=(searches or None) if searches is not None else base.web_search_requests,
                 fast=(speed == 'fast') if isinstance(speed, str) else base.fast)


def gemini_usage(meta) -> Usage | None:
    """Usage in the IR (OpenAI) meaning from a Gemini usageMetadata. promptTokenCount already includes the cached
    content; candidatesTokenCount does NOT include thoughtsTokenCount (thinking is billed as output), so output_tokens =
    candidates + thoughts and reasoning_tokens = thoughts. toolUsePromptTokenCount is billed as input and is added.
    The modality details (promptTokensDetails / candidatesTokensDetails [{modality, tokenCount}]) give the audio input,
    image input and audio output parts that providers price at their own rates."""
    if not isinstance(meta, dict):
        return None
    prompt, tool_prompt = _int(meta.get('promptTokenCount')), _int(meta.get('toolUsePromptTokenCount'))
    candidates, thoughts = _int(meta.get('candidatesTokenCount')), _int(meta.get('thoughtsTokenCount'))
    inp = None if prompt is None and tool_prompt is None else (prompt or 0) + (tool_prompt or 0)
    out = None if candidates is None and thoughts is None else (candidates or 0) + (thoughts or 0)

    def modality(key: str, name: str) -> int | None:
        rows = meta.get(key)
        found = [_int(row.get('tokenCount')) or 0 for row in rows if isinstance(row, dict)
                 and row.get('modality') == name] if isinstance(rows, list) else []
        return sum(found) or None
    return Usage(input_tokens=inp, output_tokens=out, cached_input_tokens=_int(meta.get('cachedContentTokenCount')),
                 reasoning_tokens=thoughts, audio_input_tokens=modality('promptTokensDetails', 'AUDIO'),
                 image_input_tokens=modality('promptTokensDetails', 'IMAGE'),
                 audio_output_tokens=modality('candidatesTokensDetails', 'AUDIO'))


def ollama_usage(line) -> Usage | None:
    if not isinstance(line, dict) or not line.get('done'):
        return None
    return Usage(input_tokens=_int(line.get('prompt_eval_count')), output_tokens=_int(line.get('eval_count')))


def _add(a: int | None, b: int | None) -> int | None:
    return None if a is None and b is None else (a or 0) + (b or 0)


class UsageObserver:
    """Feeds on parsed frames of one protocol and keeps the usage seen so far."""

    def __init__(self, protocol: str):
        self.protocol = protocol
        self.usage: Usage | None = None
        self.images = 0

    def sse(self, frame: SSEFrame) -> bool:
        """Observe one SSE frame; True when it is the injected include_usage chunk (choices [] plus a usage object)."""
        if frame.data == '[DONE]':
            return False
        try:
            data = json.loads(frame.data)
        except ValueError:
            return False
        if not isinstance(data, dict):
            return False
        if self.protocol == 'openai':
            if isinstance(data.get('usage'), dict):
                self.usage = openai_usage(data['usage'])
                return data.get('choices') == []
        elif self.protocol == 'responses':
            response = data.get('response')
            if isinstance(response, dict) and isinstance(response.get('usage'), dict):
                self.usage = responses_object_usage(response)
        elif self.protocol == 'anthropic':
            if data.get('type') == 'message_start' and isinstance(data.get('message'), dict):
                self.usage = anthropic_usage(data['message'].get('usage'), self.usage)
            elif data.get('type') == 'message_delta':
                self.usage = anthropic_usage(data.get('usage'), self.usage)
        elif self.protocol == 'gemini':
            if isinstance(data.get('usageMetadata'), dict):
                self.usage = gemini_usage(data['usageMetadata'])
        elif self.protocol == 'openai_media':
            kind = data.get('type') if isinstance(data.get('type'), str) else ''
            if kind.endswith('.completed') and kind.startswith(('image_generation.', 'image_edit.')):
                self.images += 1
                tokens = openai_media_usage(data.get('usage')) or Usage()
                previous = self.usage or Usage()
                self.usage = Usage(
                    images=self.images,
                    input_tokens=_add(previous.input_tokens, tokens.input_tokens),
                    output_tokens=_add(previous.output_tokens, tokens.output_tokens),
                    image_input_tokens=_add(previous.image_input_tokens, tokens.image_input_tokens))
            elif kind in ('transcript.text.done', 'speech.audio.done'):
                self.usage = openai_media_usage(data.get('usage')) or self.usage
        return False

    def line(self, value: dict | None) -> None:
        if value is not None and value.get('done'):
            self.usage = ollama_usage(value) or self.usage


async def _aclose(iterator) -> None:
    close = getattr(iterator, 'aclose', None)
    if close is not None:
        await close()


class ManagedStream:
    """An async iterator over an async generator whose aclose() also runs the cleanup callables, even when the generator
    was never started (aclose() on an unstarted async generator skips its finally blocks, so a source it would have
    closed stays open until the garbage collector finds it). Every cleanup must be idempotent; each runs even when an
    earlier one fails (the first failure is re-raised)."""
    __slots__ = ('_gen', '_cleanup', '_closed')

    def __init__(self, gen, *cleanup):
        self._gen = gen
        self._cleanup = cleanup
        self._closed = False

    def __aiter__(self):
        return self

    def __anext__(self):
        return self._gen.__anext__()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        failure = None
        for step in (self._gen.aclose, *self._cleanup):
            try:
                await step()
            except BaseException as exc:  # noqa: BLE001 - every cleanup runs; the first failure wins
                failure = failure or exc
        if failure is not None:
            raise failure


def _closer(iterator):
    async def close():
        await _aclose(iterator)
    return close


def _loop_future() -> asyncio.Future:
    return asyncio.get_running_loop().create_future()


def tee_usage(aiter, protocol: str, *, strip_injected_usage: bool) -> tuple[AsyncIterator[bytes], asyncio.Future]:
    """(bytes for the client, future of the observed Usage | None). Without stripping, the client receives the upstream
    chunks unchanged; with strip_injected_usage (OpenAI chat/completions only) the stream is re-framed per frame and the
    injected usage chunk is dropped. The future resolves when the stream ends, is closed (also before its first byte) or
    fails (with what was seen). The returned iterator's aclose() closes aiter even when iteration never started.
    Observe-only (non-stripping) streams never fail on the parser limits: a line over MAX_LINE, a non-UTF-8 line or too
    many frames only stops the observation (the usage seen so far is kept) while the bytes keep flowing unchanged.
    Must be called with a running event loop."""
    future = _loop_future()
    observer = UsageObserver(protocol)

    async def settle():
        if not future.done():
            future.set_result(observer.usage)
        await _aclose(aiter)

    async def binary():
        try:
            async for chunk in aiter:
                yield chunk
        finally:
            await settle()

    async def ndjson():
        splitter = NDJSONSplitter()
        try:
            async for chunk in aiter:
                if splitter is not None:
                    try:
                        for _, value in splitter.feed(chunk):
                            observer.line(value)
                    except BrokerError:
                        splitter = None  # observe-only: stop observing, keep forwarding
                yield chunk
            if splitter is not None:
                try:
                    for _, value in splitter.finish():
                        observer.line(value)
                except BrokerError:
                    pass
        finally:
            await settle()

    async def sse():
        splitter = _splitter(protocol)
        strip = strip_injected_usage and protocol == 'openai'
        observing = True
        try:
            async for chunk in aiter:
                if not strip:
                    if observing:
                        try:
                            for _, frame in splitter.feed(chunk):
                                if frame is not None:
                                    observer.sse(frame)
                        except BrokerError:
                            observing = False  # observe-only: stop observing, keep forwarding
                    yield chunk
                    continue
                blocks = splitter.feed(chunk)
                out = bytearray()
                for raw, frame in blocks:
                    if frame is not None and observer.sse(frame):
                        continue  # the injected include_usage chunk: observed, never forwarded
                    out += raw
                if out:
                    yield bytes(out)
            if strip:
                tail = splitter.finish()
                if tail:
                    yield tail
        finally:
            await settle()

    if protocol in BINARY_PROTOCOLS:
        return ManagedStream(binary(), settle), future
    if protocol in NDJSON_PROTOCOLS:
        return ManagedStream(ndjson(), settle), future
    if protocol in SSE_PROTOCOLS:
        return ManagedStream(sse(), settle), future
    raise ValueError(f'Unknown stream protocol: {protocol!r}')


async def first_frame(aiter, protocol: str):
    """(held first frame, replay iterator). The frame is an SSEFrame (SSE protocols; comment-only blocks before it are held
    too), a dict (NDJSON) or bytes (binary, at most BINARY_FIRST_FRAME); None when the stream ended first. The replay
    iterator yields every byte read so far exactly as received, then the rest of aiter; its aclose() closes aiter even
    when it was never iterated. When reading the first frame fails (a parser limit, an upstream failure, cancellation),
    aiter is closed before the error propagates."""
    iterator = aiter.__aiter__()
    try:
        frame, held = await _read_first(iterator, protocol)
    except BaseException:
        await _aclose(iterator)
        raise

    async def replay():
        try:
            for chunk in held:
                yield chunk
            async for chunk in iterator:
                yield chunk
        finally:
            await _aclose(iterator)

    return frame, ManagedStream(replay(), _closer(iterator))


async def _read_first(iterator, protocol: str):
    held: list[bytes] = []
    frame = None
    if protocol in BINARY_PROTOCOLS:
        async for chunk in iterator:
            if not chunk:
                continue
            if len(chunk) > BINARY_FIRST_FRAME:
                held.extend([chunk[:BINARY_FIRST_FRAME], chunk[BINARY_FIRST_FRAME:]])
                frame = chunk[:BINARY_FIRST_FRAME]
            else:
                held.append(chunk)
                frame = chunk
            break
    else:
        splitter = _splitter(protocol) if protocol in SSE_PROTOCOLS else NDJSONSplitter() if protocol in NDJSON_PROTOCOLS \
            else None
        if splitter is None:
            raise ValueError(f'Unknown stream protocol: {protocol!r}')
        async for chunk in iterator:
            held.append(chunk)
            found = [value for _, value in splitter.feed(chunk) if value is not None]
            if found:
                frame = found[0]
                break
        else:
            if isinstance(splitter, NDJSONSplitter):
                found = [value for _, value in splitter.finish() if value is not None]
                frame = found[0] if found else None
    return frame, held


def is_error_frame(frame, protocol: str) -> bool:
    """Whether a held first frame is an in-protocol error (an error-first stream is a pre-commit upstream failure)."""
    if frame is None:
        return False
    if protocol in NDJSON_PROTOCOLS:
        return isinstance(frame, dict) and 'error' in frame
    if not isinstance(frame, SSEFrame):
        return False
    if frame.event in ('error', 'response.failed'):
        return True
    try:
        data = json.loads(frame.data)
    except ValueError:
        return False
    if not isinstance(data, dict):
        return False
    if protocol in ('anthropic', 'responses'):
        return data.get('type') in ('error', 'response.failed')
    if protocol in MEDIA_PROTOCOLS:
        return data.get('type') == 'error' or 'error' in data
    return 'error' in data


__all__ = ['BINARY_FIRST_FRAME', 'KEEPALIVE', 'MAX_FRAMES', 'MAX_LINE', 'MEDIA_MAX_LINE', 'MEDIA_PROTOCOLS', 'ManagedStream', 'NDJSONSplitter', 'SSEFrame',
           'SSESplitter', 'UsageObserver', 'anthropic_usage', 'first_frame', 'gemini_usage', 'is_error_frame', 'ollama_usage',
           'openai_media_usage', 'openai_usage', 'parse_ndjson', 'parse_sse', 'responses_object_usage', 'responses_usage', 'tee_usage', 'write_json_sse', 'write_sse']
