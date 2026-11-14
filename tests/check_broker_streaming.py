"""Broker streaming engine (WP5): sse.py and the incremental codecs.

- Incrementality proof for every claimed streaming pair: the WP4 mock holds a later upstream frame behind its gate and
  the test asserts that a translated chunk has already reached the reader before the gate is released (the held frame is
  frame 2 wherever the inbound encoder answers the upstream's first event, otherwise the frame right after the first text
  frame, since Gemini, Ollama and completions encoders emit nothing for a bare start event). The L* pair (gemini <- local
  OpenAI SSE) uses an in-process gated source. Gemini and Ollama tool calls are asserted to be emitted at BlockStop.
- OpenAI chat and completions streams end with data: [DONE]; Responses streams carry an increasing sequence_number and the
  full item, part and done events (text and function calls).
- Keepalive comments are ignored by protocols.events (the local SSE consumer) and by parse_sse.
- The include_usage strip: byte-identical except the injected usage chunk, at any chunk boundary.
- Parser limits (line size, frame count), CRLF / CR / split CRLF, partial UTF-8, multi-line data, event names, comments.
- Binary TTS stream passthrough (first chunk held, bytes unchanged) and the streaming WAV header (0xFFFFFFFF sizes).
- first_frame replays every byte, holds comment-only blocks, and flags error-first streams; a mid-stream disconnect is a
  typed upstream failure. Gemini streams are checked with LF and CRLF framing.
- Release: with the garbage collector disabled, the upstream response and its egress concurrency slot are free
  synchronously after an error-first UpstreamFailure, after an UpstreamResult closed before its body was iterated (and
  result.usage is resolved then; checked on an httpx.MockTransport with a streamed body, no socket), after a translated
  IR stream closed before its first event, and when reading the first frame fails.
- The observe-only usage tee forwards a frame over MAX_LINE unchanged (it only stops observing).
Loopback mocks only (netguard installed); synthetic values only.
"""
import asyncio
import gc
import json
import struct
import unittest

import broker_adapter_support as S  # installs the netguard first
from broker_mocks import OpenAIMock, load_fixture, netguard

import httpx

from titanos.broker.adapters.base import RawRequest
from titanos.broker.config import Upstream
from titanos.broker.errors import BrokerError
from titanos.broker.profiles import PROFILES
from titanos.broker.sse import (BINARY_FIRST_FRAME, KEEPALIVE, MAX_FRAMES, MAX_LINE, SSEFrame, first_frame,
                                is_error_frame, parse_ndjson, parse_sse, tee_usage, write_sse)
from titanos.broker.surfaces import SURFACES
from titanos.broker.translate import audio, gemini, ollama, openai_chat, responses
from titanos.broker.translate.ir import (BlockStop, IRMessage, IRRequest, Start, Stop, Text, TextDelta, ToolArgsDelta,
                                        ToolCallStart)
from titanos.broker.translate.upstream_errors import UpstreamFailure
from titanos.broker.types import Usage

WAIT = 10.0


async def chunks(*parts, delay=0.0):
    for part in parts:
        if delay:
            await asyncio.sleep(delay)
        yield part


async def collect(aiter):
    return [item async for item in aiter]


def split(data: bytes, size: int) -> list[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)]


# upstream frame index right after the first text-bearing frame, per target column (mock fixtures)
AFTER_TEXT = {'ANT': 5, 'OAI': 3, 'OAI-chat': 3, 'AZ': 3, 'GEM': 2, 'OLC': 2}
STARTS_ON_FIRST_EVENT = {'openai.chat', 'anthropic.messages', 'openai.responses'}
PAIRS = {
    'openai.chat': ('ANT', 'GEM', 'OLC'),
    'openai.completions': ('OAI-chat', 'ANT', 'GEM', 'OLC'),
    'openai.responses': ('OAI-chat', 'AZ', 'ANT', 'GEM', 'OLC'),
    'anthropic.messages': ('OAI', 'AZ', 'GEM', 'OLC'),
    'gemini.generateContent': ('OAI', 'AZ', 'ANT', 'OLC'),
    'ollama.chat': ('OAI', 'AZ', 'ANT', 'GEM'),
    'ollama.generate': ('OAI', 'AZ', 'ANT', 'GEM'),
}


class Incrementality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets = S.start_targets(('OAI', 'AZ', 'ANT', 'GEM', 'OLC', 'XI'))
        mock = OpenAIMock()
        mock.start()
        upstream = Upstream(profile='xai', base_url=mock.api_base, credential='cred')
        cls.targets['OAI-chat'] = S.MockTarget('OAI-chat', mock, 'xai', 'grok-4', 'openai', upstream=upstream)

    @classmethod
    def tearDownClass(cls):
        S.stop_targets(cls.targets)

    def setUp(self):
        netguard.reset_attempts()
        for target in self.targets.values():
            target.mock.reset()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    async def gated(self, target, hold, produce):
        """Run produce(egress) -> async iterator of output chunks with frame `hold` held; returns (chunks before the
        release, all chunks, frames the mock had sent at the first chunk)."""
        target.mock.hold(frame=hold)
        out: list[bytes] = []
        async with target.egress() as egress:
            stream = await produce(egress)

            async def pump():
                async for chunk in stream:
                    out.append(chunk)
            task = asyncio.ensure_future(pump())
            loop = asyncio.get_running_loop()
            deadline = loop.time() + WAIT
            while not out and not task.done() and loop.time() < deadline:
                await asyncio.sleep(0.005)
            early = list(out)
            gate_closed = not target.mock.gate.is_set()
            sent = target.mock.hits[-1].frames_sent
            target.mock.gate.set()
            await asyncio.wait_for(task, 3 * WAIT)
        self.assertTrue(gate_closed, 'the mock released the held frame before the reader saw a chunk')
        self.assertTrue(early, 'no translated chunk reached the reader while the upstream frame was held')
        self.assertEqual(sent, hold - 1)
        return early, out

    def pair(self, inbound, column):
        target = self.targets[column]
        hold = 2 if inbound in STARTS_ON_FIRST_EVENT else AFTER_TEXT[column]
        driver = S.INBOUND[inbound]

        async def produce(egress):
            ir = driver['decode'](S.REQUESTS[inbound](True))
            events = await target.adapter.invoke(target.grant(), egress, ir)
            return driver['stream'](events, 'alias')
        early, out = S.run(self.gated(target, hold, produce))
        text = S.stream_text(inbound, b''.join(out))
        self.assertTrue(text.startswith('Hello'), text)
        return early, out

    def test_openai_chat_pairs(self):
        for column in PAIRS['openai.chat']:
            with self.subTest(target=column):
                early, out = self.pair('openai.chat', column)
                self.assertEqual(S.sse_events(b''.join(out))[-1][1], '[DONE]')

    def test_openai_completions_pairs(self):
        for column in PAIRS['openai.completions']:
            with self.subTest(target=column):
                early, out = self.pair('openai.completions', column)
                events = S.sse_events(b''.join(out))
                self.assertEqual(events[-1][1], '[DONE]')
                self.assertEqual(json.loads(events[0][1])['object'], 'text_completion')

    def test_responses_pairs(self):
        expected = ['response.created', 'response.in_progress', 'response.output_item.added',
                    'response.content_part.added', 'response.output_text.delta', 'response.output_text.done',
                    'response.content_part.done', 'response.output_item.done', 'response.completed']
        for column in PAIRS['openai.responses']:
            with self.subTest(target=column):
                early, out = self.pair('openai.responses', column)
                events = S.sse_events(b''.join(out))
                names = [e for e, _ in events]
                self.assertEqual([n for i, n in enumerate(names) if i == 0 or n != names[i - 1]], expected)
                sequence = [json.loads(p)['sequence_number'] for _, p in events]
                self.assertEqual(sequence, list(range(len(sequence))))
                self.assertTrue(all(json.loads(p)['type'] == e for e, p in events))
                final = json.loads(events[-1][1])['response']
                self.assertEqual(final['output'][0]['content'][0]['text'], S.stream_text('openai.responses',
                                                                                         b''.join(out)))

    def test_anthropic_pairs(self):
        for column in PAIRS['anthropic.messages']:
            with self.subTest(target=column):
                early, out = self.pair('anthropic.messages', column)
                self.assertEqual(S.sse_events(early[0])[0][0], 'message_start')

    def test_gemini_pairs(self):
        for column in PAIRS['gemini.generateContent']:
            with self.subTest(target=column):
                self.pair('gemini.generateContent', column)

    def test_gemini_from_local_openai_stream(self):
        """L*: a local OpenAI SSE stream (the shape the local route emits) re-encoded as Gemini, incrementally."""
        frames = load_fixture('openai', 'chat_stream').frames({'model': 'local-model'})
        gate = None

        async def local():
            for index, frame in enumerate(frames, start=1):
                if index == 3:
                    await gate.wait()
                yield frame

        async def go():
            nonlocal gate
            gate = asyncio.Event()
            out = []
            encoder = gemini.encode_stream(openai_chat.decode_stream(parse_sse(local())), model_name='alias')

            async def pump():
                async for chunk in encoder:
                    out.append(chunk)
            task = asyncio.ensure_future(pump())
            loop = asyncio.get_running_loop()
            deadline = loop.time() + WAIT
            while not out and loop.time() < deadline:
                await asyncio.sleep(0.005)
            early = list(out)
            gate.set()
            await asyncio.wait_for(task, WAIT)
            return early, out
        early, out = S.run(go())
        self.assertTrue(early)
        self.assertEqual(S.stream_text('gemini.generateContent', b''.join(out)), 'Hello! How can I help?')

    def test_ollama_pairs(self):
        for column in PAIRS['ollama.chat']:
            with self.subTest(target=column):
                early, out = self.pair('ollama.chat', column)
                self.assertEqual(S.ndjson_lines(b''.join(out))[-1]['done'], True)

    def test_ollama_generate_pairs(self):
        for column in PAIRS['ollama.generate']:
            with self.subTest(target=column):
                early, out = self.pair('ollama.generate', column)
                lines = S.ndjson_lines(b''.join(out))
                self.assertEqual(lines[-1]['done'], True)
                self.assertTrue(all('response' in line and 'message' not in line for line in lines))
                self.assertTrue(early[0].strip().startswith(b'{'))

    def test_binary_audio_pairs(self):
        for column in ('OAI', 'AZ'):  # elevenlabs.tts_stream <- OpenAI-compatible speech
            target = self.targets[column]
            with self.subTest(target=column):
                async def produce(egress, target=target):
                    body = audio.speech_elevenlabs_to_openai({'text': 'Hello', 'model_id': 'x'},
                                                             {'output_format': 'mp3_44100_128'},
                                                             {'v1': 'alloy'}, voice_id='v1')
                    result = await target.adapter.speech(target.grant('tts-1'), egress, body, stream=True)
                    return result.body
                early, out = S.run(self.gated(target, 2, produce))
                expected = b''.join(load_fixture('openai', 'speech').frames({}, variant='mp3'))
                self.assertEqual(b''.join(out), expected)
        xi = self.targets['XI']  # openai.speech <- ElevenLabs

        async def produce_xi(egress):
            voice, body, query, wrap = audio.speech_openai_to_elevenlabs({'input': 'Hello', 'voice': 'alloy'},
                                                                         {'alloy': 'v1'})
            result = await xi.adapter.tts(xi.grant(), egress, voice, body, query, stream=True)
            return result.body
        early, out = S.run(self.gated(xi, 2, produce_xi))
        self.assertEqual(b''.join(out), b''.join(load_fixture('elevenlabs', 'tts').frames({}, variant='mp3')))

    def test_tool_call_streams_are_whole_for_gemini_and_ollama(self):
        async def go(encoder_factory):
            queue: asyncio.Queue = asyncio.Queue()

            async def events():
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    yield item
            out = []
            encoder = encoder_factory(events())

            async def pump():
                async for chunk in encoder:
                    out.append(chunk)
            task = asyncio.ensure_future(pump())

            async def settle():
                for _ in range(20):
                    await asyncio.sleep(0)
            for event in (Start('r1', 'm'), TextDelta(0, 'Checking.'), BlockStop(0), ToolCallStart(1, 'call_1', 'get_weather'),
                          ToolArgsDelta(1, '{"location":'), ToolArgsDelta(1, ' "Paris"}')):
                await queue.put(event)
            await settle()
            before_stop = list(out)
            await queue.put(BlockStop(1))
            await settle()
            after_stop = list(out)
            await queue.put(Stop('tool_use'))
            await queue.put(None)
            await asyncio.wait_for(task, WAIT)
            return before_stop, after_stop, out
        before, after, out = S.run(go(lambda ev: gemini.encode_stream(ev, model_name='alias')))
        self.assertEqual(len(before), 1)  # the text chunk only: the tool call is still buffering
        call = json.loads(S.sse_events(after[-1])[0][1])['candidates'][0]['content']['parts'][0]
        self.assertEqual(call, {'functionCall': {'name': 'get_weather', 'args': {'location': 'Paris'}}})
        before, after, out = S.run(go(lambda ev: ollama.encode_ndjson_stream(ev, model_name='alias', generate=False)))
        self.assertEqual(len(before), 1)
        line = json.loads(after[-1])
        self.assertEqual(line['message']['tool_calls'], [{'function': {'name': 'get_weather',
                                                                       'arguments': {'location': 'Paris'}}}])
        self.assertTrue(json.loads(out[-1])['done'])

    def test_responses_function_call_family(self):
        target = self.targets['ANT']
        target.mock.respond_with('anthropic.messages', 'messages_stream_tools')

        async def go():
            async with target.egress() as egress:
                ir = responses.decode_request({'model': 'a', 'input': 'weather?', 'stream': True, 'tools': [
                    {'type': 'function', 'name': 'get_weather', 'parameters': {'type': 'object'}}]})
                events = await target.adapter.invoke(target.grant(), egress, ir)
                return b''.join(await collect(responses.encode_stream(events, responses.response_object(model='a'))))
        events = S.sse_events(S.run(go()))
        names = [e for e, _ in events]
        for name in ('response.output_item.added', 'response.function_call_arguments.delta',
                     'response.function_call_arguments.done', 'response.output_item.done', 'response.completed'):
            self.assertIn(name, names)
        done = [json.loads(p) for e, p in events if e == 'response.function_call_arguments.done'][0]
        self.assertEqual(json.loads(done['arguments']), {'location': 'Paris'})
        final = json.loads(events[-1][1])['response']
        self.assertEqual([item['type'] for item in final['output']], ['message', 'function_call'])
        self.assertEqual(final['output'][1]['call_id'], 'toolu_mock0001')
        sequence = [json.loads(p)['sequence_number'] for _, p in events]
        self.assertEqual(sequence, list(range(len(sequence))))

    def test_mid_stream_disconnect_is_a_typed_failure(self):
        target = self.targets['OAI']
        target.mock.faults.push('disconnect_after_frames', frames=2)

        async def go():
            async with target.egress() as egress:
                result = await target.adapter.passthrough(target.grant(), egress, SURFACES['openai.chat'],
                                                          RawRequest(method='POST', json={
                                                              'model': 'a', 'stream': True,
                                                              'messages': [{'role': 'user', 'content': 'x'}]}))
                try:
                    async for _ in result.body:
                        pass
                finally:
                    await result.aclose()
        with self.assertRaises(UpstreamFailure) as ctx:
            S.run(go())
        self.assertEqual(ctx.exception.upstream.kind, 'network')

    def test_gemini_crlf_framing(self):
        mock_targets = S.start_targets(('GEM',), gemini_crlf=True)
        try:
            out = S.run(S.translate(mock_targets['GEM'], 'openai.chat', stream=True))
            self.assertEqual(S.stream_text('openai.chat', out), 'Hello! How can I help you today?')
            self.assertTrue(mock_targets['GEM'].mock.sse_crlf)
        finally:
            S.stop_targets(mock_targets)


class Keepalive(unittest.TestCase):
    def test_local_protocols_events_ignore_keepalive(self):
        from titanos.protocols import events

        class Body:
            def __init__(self, parts):
                self.body_iterator = chunks(*parts)
        data = (KEEPALIVE + b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n' + KEEPALIVE +
                b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n' + b'data: [DONE]\n\n')
        parsed = S.run(collect(events(Body(split(data, 5)))))
        self.assertEqual([p['choices'][0]['delta']['content'] for p in parsed], ['a', 'b'])
        frames = S.run(collect(parse_sse(chunks(*split(data, 3)))))
        self.assertEqual([f.data for f in frames][-1], '[DONE]')
        self.assertEqual(len(frames), 3)
        self.assertEqual(KEEPALIVE, b': keepalive\n\n')


class Parser(unittest.TestCase):
    def test_framing_rules(self):
        data = (b'event: message_start\r\ndata: {"a":\r\ndata: 1}\r\n\r\n'  # CRLF and multi-line data
                b': comment\n\n'  # comment-only block: not dispatched
                b'event: x\rdata:no-space\r\r'  # CR-only line endings, no space after the colon
                b'data: \xc3\xa9t\xc3\xa9\n\n')  # UTF-8 split across chunks below
        for size in (1, 2, 3, 7, len(data)):
            with self.subTest(chunk=size):
                frames = S.run(collect(parse_sse(chunks(*split(data, size)))))
                self.assertEqual([(f.event, f.data) for f in frames],
                                 [('message_start', '{"a":\n1}'), ('x', 'no-space'), (None, 'été')])
        self.assertEqual(write_sse('a\nb', 'evt'), b'event: evt\ndata: a\ndata: b\n\n')
        incomplete = S.run(collect(parse_sse(chunks(b'data: 1\n\ndata: 2'))))
        self.assertEqual([f.data for f in incomplete], ['1'])  # an unterminated trailing frame is discarded

    def test_limits(self):
        self.assertEqual((MAX_LINE, MAX_FRAMES), (1024 * 1024, 100_000))
        with self.assertRaises(BrokerError) as ctx:
            S.run(collect(parse_sse(chunks(b'data: ' + b'x' * 100), max_line=64)))
        self.assertEqual((ctx.exception.code, ctx.exception.status), ('upstream_bad_response', 502))
        with self.assertRaises(BrokerError):
            S.run(collect(parse_sse(chunks(b'data: ' + b'x' * 100 + b'\n\n'), max_line=64)))
        with self.assertRaises(BrokerError):
            S.run(collect(parse_sse(chunks(b'data: 1\n\n' * 11), max_frames=10)))
        self.assertEqual(len(S.run(collect(parse_sse(chunks(b'data: 1\n\n' * 10), max_frames=10)))), 10)
        with self.assertRaises(BrokerError):
            S.run(collect(parse_sse(chunks(b'data: \xff\xfe\n\n'))))
        with self.assertRaises(BrokerError):
            S.run(collect(parse_ndjson(chunks(b'{"a":1}\n{not json}\n'))))
        with self.assertRaises(BrokerError):
            S.run(collect(parse_ndjson(chunks(b'x' * 100), max_line=64)))
        lines = S.run(collect(parse_ndjson(chunks(b'{"a":', b'1}\n\n{"b":2}'))))
        self.assertEqual(lines, [{'a': 1}, {'b': 2}])
        big = b'data: ' + b'y' * (MAX_LINE - 10) + b'\n\n'  # the default limit admits a frame just under 1 MiB
        self.assertEqual(len(S.run(collect(parse_sse(chunks(*split(big, 65536)))))[0].data), MAX_LINE - 10)

    def test_include_usage_strip_is_byte_identical_otherwise(self):
        fx = load_fixture('openai', 'chat_stream_usage')
        frames = fx.frames({'model': 'gpt-4o-mini', 'include_usage': True})
        upstream = b''.join(frames)
        usage_frame = [f for f in frames if b'"choices":[]' in f]
        self.assertEqual(len(usage_frame), 1)
        expected = upstream.replace(usage_frame[0], b'')
        for size in (1, 5, 64, len(upstream)):
            with self.subTest(chunk=size):
                async def go(strip):
                    body, future = tee_usage(chunks(*split(upstream, size)), 'openai', strip_injected_usage=strip)
                    data = b''.join(await collect(body))
                    return data, await future
                stripped, usage = S.run(go(True))
                self.assertEqual(stripped, expected)
                self.assertEqual((usage.input_tokens, usage.output_tokens), (19, 10))
                kept, usage = S.run(go(False))
                self.assertEqual(kept, upstream)
                self.assertEqual(usage.output_tokens, 10)

    def test_tee_usage_per_protocol(self):
        cases = [('anthropic', load_fixture('anthropic', 'messages_stream').frames({'model': 'm'}),
                  Usage(input_tokens=12, output_tokens=10, cached_input_tokens=0, cache_write_tokens=0)),
                 ('gemini', load_fixture('gemini', 'stream_generate_content').frames({'model': 'm'}),
                  Usage(input_tokens=4, output_tokens=9)),
                 ('ollama', load_fixture('ollama', 'chat_stream').frames({'model': 'm'}),
                  Usage(input_tokens=26, output_tokens=12)),
                 ('responses', load_fixture('openai', 'responses_stream').frames(
                     {'model': 'm', 'response_id': 'resp_1', 'previous_response_id': None, 'store': True,
                      'background': False}), None),
                 ('binary', [b'\x00' * 10, b'\x01' * 10], None)]
        for protocol, frames, expected in cases:
            with self.subTest(protocol=protocol):
                async def go():
                    body, future = tee_usage(chunks(*frames), protocol, strip_injected_usage=False)
                    return b''.join(await collect(body)), await future
                data, usage = S.run(go())
                self.assertEqual(data, b''.join(frames))
                if protocol == 'responses':
                    self.assertIsNotNone(usage.input_tokens)
                else:
                    self.assertEqual(usage, expected)

    def test_first_frame_holds_and_replays(self):
        data = b': OPENROUTER PROCESSING\n\n' + b'data: {"id":"x","choices":[]}\n\n' + b'data: [DONE]\n\n'

        async def go(parts, protocol):
            frame, replay = await first_frame(chunks(*parts), protocol)
            return frame, b''.join(await collect(replay))
        for size in (1, 4, len(data)):
            frame, replayed = S.run(go(split(data, size), 'openai'))
            self.assertEqual(frame.data, '{"id":"x","choices":[]}')
            self.assertEqual(replayed, data)
        frame, replayed = S.run(go([b'{"model":"m","done":false}\n{"done":true}\n'], 'ollama'))
        self.assertEqual(frame, {'model': 'm', 'done': False})
        audio_bytes = b'\x01' * (BINARY_FIRST_FRAME + 100)
        frame, replayed = S.run(go([audio_bytes, b'\x02' * 10], 'binary'))
        self.assertEqual(len(frame), BINARY_FIRST_FRAME)
        self.assertEqual(replayed, audio_bytes + b'\x02' * 10)
        frame, replayed = S.run(go([], 'gemini'))
        self.assertIsNone(frame)
        error = SSEFrame('error', '{"type":"error","error":{"type":"overloaded_error","message":"x"}}', b'')
        self.assertTrue(is_error_frame(error, 'anthropic'))
        self.assertTrue(is_error_frame(SSEFrame(None, '{"error":{"code":503}}', b''), 'gemini'))
        self.assertTrue(is_error_frame(SSEFrame(None, '{"error":{"message":"x"}}', b''), 'openai'))
        self.assertTrue(is_error_frame({'error': 'boom'}, 'ollama'))
        self.assertFalse(is_error_frame(SSEFrame('message_start', '{"type":"message_start"}', b''), 'anthropic'))
        self.assertFalse(is_error_frame(SSEFrame(None, '[DONE]', b''), 'openai'))

    def test_streaming_wav_header(self):
        pcm = struct.pack('<6h', 1, 2, 3, 4, 5, 6)
        streamed = audio.pcm_to_wav(pcm, 24000, streaming=True)
        self.assertEqual(struct.unpack('<I', streamed[4:8])[0], 0xFFFFFFFF)
        self.assertEqual(struct.unpack('<I', streamed[40:44])[0], 0xFFFFFFFF)
        self.assertEqual(streamed[44:], pcm)
        out = S.run(collect(audio.wrap_wav_stream(chunks(pcm[:5], b'', pcm[5:]), 24000)))
        self.assertEqual(out[0], audio.wav_header(24000, None))
        self.assertEqual(b''.join(out[1:]), pcm)
        self.assertEqual(audio.pcm_rate('pcm_16000'), 16000)


class Release(unittest.TestCase):
    """Deterministic release of the upstream response and the egress slot (gc disabled, so nothing is left to the
    collector or the loop's async-generator finalizer)."""

    @classmethod
    def setUpClass(cls):
        cls.targets = S.start_targets(('OAI', 'ANT'))
        for target in cls.targets.values():
            target.upstream = Upstream(profile=target.profile_id, base_url=target.mock.api_base, enabled=True,
                                       credential='cred', max_concurrency=1)
            target.adapter = S.ADAPTERS[target.profile.adapter](target.profile, target.upstream,
                                                                default_max_tokens=256)

    @classmethod
    def tearDownClass(cls):
        S.stop_targets(cls.targets)

    def setUp(self):
        netguard.reset_attempts()
        for target in self.targets.values():
            target.mock.reset()
        gc.disable()

    def tearDown(self):
        gc.enable()
        self.assertEqual(netguard.attempts(), [])

    @staticmethod
    def slot_free(client) -> bool:
        return all(not entry[1].locked() for entry in client._async_slots.values())

    def test_result_closed_before_iteration_resolves_usage_and_frees_the_slot(self):
        # an httpx.MockTransport with a genuinely streamed body (no socket): the client goes away between commit and
        # the first byte, so the body is never iterated
        cases = (('anthropic', 'anthropic.messages', 'anthropic', load_fixture('anthropic', 'messages_stream'),
                  {'model': 'x', 'max_tokens': 16, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}),
                 ('openai', 'openai.chat', 'openai', load_fixture('openai', 'chat_stream_usage'),
                  {'model': 'x', 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}))
        for profile_id, surface, sentinel, fixture, body in cases:
            with self.subTest(profile=profile_id):
                frames = fixture.frames({'model': 'x', 'include_usage': True})

                async def streamed():
                    for frame in frames:
                        yield frame

                def handler(request):
                    return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=streamed())
                upstream = Upstream(profile=profile_id, enabled=True, credential='cred', max_concurrency=1)
                adapter = S.ADAPTERS[PROFILES[profile_id].adapter](PROFILES[profile_id], upstream)
                credential = S.handle(upstream.origin(allow_http_lan=False), profile_id, sentinel)

                async def go():
                    async with S.Egress(transport=httpx.MockTransport(handler), settings=S.Settings()) as client:
                        call = S.grant(upstream, profile_id, 'm', credential)
                        result = await adapter.passthrough(call, client, SURFACES[surface],
                                                           RawRequest(method='POST', json=body))
                        held = self.slot_free(client)
                        await result.aclose()
                        free, done = self.slot_free(client), result.usage.done()
                        again = await adapter.passthrough(call, client, SURFACES[surface],
                                                          RawRequest(method='POST', json=body))
                        data = b''.join([chunk async for chunk in again.body])
                        await again.aclose()
                        return held, free, done, data, await again.usage
                held, free, done, data, usage = S.run(go())
                self.assertFalse(held)
                self.assertTrue(free)
                self.assertTrue(done)
                self.assertTrue(data)
                self.assertIsNotNone(usage.output_tokens)

    def test_error_first_failure_frees_the_slot_synchronously(self):
        target = self.targets['ANT']

        async def go():
            async with target.egress() as client:
                target.mock.respond_with('anthropic.messages', 'messages_stream_error_first')
                kept = None
                try:
                    await target.adapter.passthrough(target.grant(), client, SURFACES['anthropic.messages'],
                                                     RawRequest(method='POST', json={
                                                         'model': 'x', 'max_tokens': 16, 'stream': True,
                                                         'messages': [{'role': 'user', 'content': 'hi'}]}))
                except UpstreamFailure as exc:
                    kept = exc  # a fallback chain keeps the last failure alive
                return kept, self.slot_free(client)
        failure, free = S.run(go())
        self.assertIsInstance(failure, UpstreamFailure)
        self.assertTrue(free)

    def test_translated_stream_closed_before_its_first_event_frees_the_slot(self):
        target = self.targets['ANT']

        async def go():
            async with target.egress() as client:
                ir = IRRequest(kind='chat', model=target.model, messages=[IRMessage('user', [Text('hi')])],
                               stream=True)
                events = await target.adapter.invoke(target.grant(), client, ir)
                before = self.slot_free(client)
                await events.aclose()
                return before, self.slot_free(client)
        before, after = S.run(go())
        self.assertFalse(before)
        self.assertTrue(after)

    def test_first_frame_failure_closes_the_source(self):
        source = BadFirstLine()
        with self.assertRaises(BrokerError):
            S.run(first_frame(source, 'openai'))
        self.assertTrue(source.closed)

    def test_observe_only_tee_forwards_an_over_long_frame(self):
        big = b'data: {"candidates":[{"content":{"parts":[{"inlineData":{"data":"' + b'A' * (MAX_LINE + 10) + \
            b'"}}]}}]}\n\n'
        tail = b'data: {"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1}}\n\n'
        upstream = big + tail

        async def go(stream, protocol):
            body, future = tee_usage(chunks(*split(stream, 65536)), protocol, strip_injected_usage=False)
            return b''.join(await collect(body)), await future
        # Gemini frames are whole GenerateContentResponse objects (an image frame is one multi-MiB line): its tee reads
        # lines up to MEDIA_MAX_LINE, so the usage after a large image frame is still observed
        data, usage = S.run(go(upstream, 'gemini'))
        self.assertEqual(data, upstream)
        self.assertEqual((usage.input_tokens, usage.output_tokens), (3, 1))
        # other protocols keep MAX_LINE: observation stops at the over-long line; the bytes are never cut
        big = b'data: {"choices":[{"delta":{"content":"' + b'A' * (MAX_LINE + 10) + b'"}}]}\n\n'
        tail = b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\ndata: [DONE]\n\n'
        data, usage = S.run(go(big + tail, 'openai'))
        self.assertEqual(data, big + tail)
        self.assertIsNone(usage)


class BadFirstLine:
    # a byte source (with aclose) whose first line is not UTF-8

    def __init__(self):
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return bytes([100, 97, 116, 97, 58, 32, 255, 254, 10, 10])  # 'data: ' + two invalid bytes + blank line

    async def aclose(self):
        self.closed = True


CELLS = {
    **{f'stream:openai.chat<-{c}': 'Incrementality.test_openai_chat_pairs' for c in PAIRS['openai.chat']},
    **{f'stream:openai.completions<-{c}': 'Incrementality.test_openai_completions_pairs'
       for c in PAIRS['openai.completions']},
    **{f'stream:responses<-{c}': 'Incrementality.test_responses_pairs' for c in PAIRS['openai.responses']},
    **{f'stream:anthropic<-{c}': 'Incrementality.test_anthropic_pairs' for c in PAIRS['anthropic.messages']},
    **{f'stream:gemini<-{c}': 'Incrementality.test_gemini_pairs' for c in PAIRS['gemini.generateContent']},
    'stream:gemini<-L*': 'Incrementality.test_gemini_from_local_openai_stream',
    **{f'stream:ollama<-{c}': 'Incrementality.test_ollama_pairs' for c in PAIRS['ollama.chat']},
    **{f'stream:ollama.generate<-{c}': 'Incrementality.test_ollama_generate_pairs' for c in PAIRS['ollama.generate']},
    'stream:elevenlabs.tts_stream<-OAI': 'Incrementality.test_binary_audio_pairs',
    'stream:elevenlabs.tts_stream<-AZ': 'Incrementality.test_binary_audio_pairs',
    'stream:openai.speech<-XI': 'Incrementality.test_binary_audio_pairs',
    'stream:gemini+ollama tool calls at BlockStop': 'Incrementality.test_tool_call_streams_are_whole_for_gemini_and_ollama',
}


if __name__ == '__main__':
    S.main(CELLS)
