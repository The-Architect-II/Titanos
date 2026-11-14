"""Broker SDK lane (sdk tier): the official OpenAI Python SDK and the official MCP client against a REAL
`python -m titanos --config <tmp>/models.json serve` subprocess whose broker.json points at the WP4 loopback mocks.

Processes:
- this script under the project interpreter (src importable): the mocks (tests/broker_mocks, netguard installed in this
  process), a temporary TITANOS_HOME (broker.json, env: credential records bound to each mock's exact origin, refs-only
  vault, an operator price row), the serve subprocess (tests/broker_mocks/netguard_site sitecustomize with
  TITANOS_TEST_NETGUARD=1, TITANOS_BROKER_EGRESS=loopback-only, a bootstrap key through TITANOS_API_KEY; tk_ keys are
  minted through POST /api/keys) and every assertion on what the mocks recorded;
- the SDK client: this same file with --client under the interpreter named by TITANOS_SDK_PYTHON (default: this
  interpreter, as in CI after `python -m pip install -r tests/requirements-sdk.txt`). The client imports only the SDKs
  and the standard library, runs under the same netguard sitecustomize, and changes nothing in the SDK except base_url
  and api_key (plus a request-counting event hook on an SDK-built HTTP client for the retry cases).

Cases (each pass, fail or untried; untried is never a pass):
- cloud through the openai mock: models list and retrieve, chat non-stream with tools and provenance headers, chat
  stream with tools and include_usage (tool-call deltas reassembled, [DONE] consumed by the SDK), a published alias,
  responses non-stream (stored and retrieved) and the typed stream event family (created ... completed, increasing
  sequence_number, function-call items), embeddings (the SDK's default base64 encoding decoded to floats), images
  (b64_json), audio speech (wav, and mp3 through the streaming response), audio transcription, moderations; the bare
  id in models.retrieve('gpt-4o-mini'); videos (create with sora-2 on base_url /v1, which Titanos' local video jobs
  share, then retrieve, download_content and delete of that id);
- the same SDK translated to other upstreams: chat with tools, stream and not, to the Anthropic and Gemini mocks; speech
  and transcription to the ElevenLabs mock (voice_maps in broker.json);
- typed errors surfaced by the SDK: unknown model (NotFoundError), wrong key (AuthenticationError), a key without the
  cloud scope (PermissionDeniedError or NotFoundError, both typed), 402 budget_exceeded and 508 loop_detected with the
  SDK's default retries enabled: each is sent exactly once (x-should-retry: false honoured), and no mock is hit;
- upstream faults through the SDK, one per client run: a mock 400 whose body echoes the presented credential becomes
  BadRequestError with the credential scrubbed (the rest of the body passes through), a stream cut after two frames
  ends with the protocol's terminal error event (the SDK raises), a mock 429 becomes RateLimitError with Retry-After;
  each makes exactly one upstream request;
- the official MCP client over streamable HTTP /mcp with a scoped key: chat to the cloud alias (provenance _meta),
  broker_explain (zero egress) and broker_status;
- local fixture models (TITANOS_SDK_FIXTURE_MODELS may name a directory with chat/, embeddings/, speech/ and
  transcription/ holding small CPU transformers models; each one present is registered; absent ones leave their
  cases untried): chat, chat stream, embeddings, speech and transcription with local provenance.
After the client: the mocks were reached on every surface above, the netguard was active in the server and the client,
zero non-loopback connection attempts (this process, the server and the client), every mock hit carries exactly its own
sentinel credential and never a tk_ token or the bootstrap key, no sentinel, token or bootstrap value appears in the
server's output or the client's report, and the server shuts down gracefully on an interrupt.

Output: one JSON document on stdout. Exit 1 on any fail, or with --strict on any untried.
tests/check_broker_sdk_vendors.py imports the harness (Stack, run_client, summarize) from this file.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests'
NETGUARD_SITE = TESTS / 'broker_mocks' / 'netguard_site'
SDK_PYTHON_ENV = 'TITANOS_SDK_PYTHON'
FIXTURE_ENV = 'TITANOS_SDK_FIXTURE_MODELS'
LOCAL_DIRS = {'local-chat': ('chat', 'chat'), 'local-embed': ('embeddings', 'embeddings'),
              'local-voice': ('speech', 'speech'), 'local-asr': ('transcription', 'transcription')}
OPENAI_SURFACES = ['openai.chat', 'openai.responses', 'openai.responses.retrieve', 'openai.embeddings',
                   'openai.images.generations', 'openai.audio.speech', 'openai.audio.transcriptions',
                   'openai.moderations']
ELEVENLABS_VOICE = '21m00Tcm4TlvDq8ikWAM'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
WEATHER_TOOL = {'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Weather for a city',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}


# =====================================================================================================================
# client side: runs under the SDK interpreter and imports nothing from titanos or the tests

class Untried(Exception):
    """A case that could not be tried here (a missing SDK or fixture); reported as untried, never as a pass."""


def run_cases(cases) -> list:
    out = []
    for case_id, fn in cases:
        started = time.monotonic()
        try:
            detail = fn()
            out.append({'id': case_id, 'state': 'pass', 'detail': detail,
                        'seconds': round(time.monotonic() - started, 2)})
        except Untried as exc:
            out.append({'id': case_id, 'state': 'untried', 'detail': str(exc)})
        except Exception as exc:  # reported per case; the orchestrator turns it into a failure
            out.append({'id': case_id, 'state': 'fail', 'detail': f'{type(exc).__name__}: {str(exc)[:600]}',
                        'seconds': round(time.monotonic() - started, 2)})
    return out


def wav_tone(seconds: float = 0.5, rate: int = 16000) -> bytes:
    import math
    import struct
    count = int(seconds * rate)
    frames = b''.join(struct.pack('<h', int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(count))
    header = (b'RIFF' + struct.pack('<I', 36 + len(frames)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 1, 1, rate,
                                                                                       rate * 2, 2, 16)
              + b'data' + struct.pack('<I', len(frames)))
    return header + frames


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def openai_client_main(phase: str, spec: dict) -> dict:
    try:
        import openai
    except ImportError as exc:
        return {'sdk': {'openai': None}, 'missing': [f'openai ({exc})'], 'cases': []}
    from importlib.metadata import version
    url, tokens = spec['url'], spec['tokens']
    sent: list = []

    def counting_client():
        return openai.DefaultHttpxClient(trust_env=False, event_hooks={'request': [lambda request: sent.append(
            (request.method, request.url.path))]})

    def client(token=None, *, retries=0, counted=False):
        http = counting_client() if counted else openai.DefaultHttpxClient(trust_env=False)
        return openai.OpenAI(base_url=url + '/v1', api_key=token or tokens['sdk'], max_retries=retries, timeout=120,
                             http_client=http)

    def status_of(fn):
        try:
            fn()
        except openai.APIStatusError as exc:
            return exc
        raise AssertionError('the SDK raised nothing')

    cases = []

    if phase == 'main':
        c = client()

        def models():
            ids = [m.id for m in c.models.list()]
            for name in ('gpt-4o-mini@openai', 'claude-sonnet-4-5@anthropic', 'gemini-2.0-flash@gemini', 'mini'):
                expect(name in ids, f'{name} missing from /v1/models: {ids[:20]}')
            one = c.models.retrieve('gpt-4o-mini@openai')
            expect(one.id == 'gpt-4o-mini@openai', one)
            return {'models': len(ids)}

        def chat_tools():
            raw = c.chat.completions.with_raw_response.create(model='gpt-4o-mini@openai', messages=CHAT,
                                                              tools=[WEATHER_TOOL])
            expect(raw.headers.get('x-titanos-served-by') == 'cloud:openai', dict(raw.headers))
            expect(raw.headers.get('x-titanos-served-model') == 'gpt-4o-mini@openai', dict(raw.headers))
            done = raw.parse()
            calls = done.choices[0].message.tool_calls or []
            expect(calls and calls[0].function.name and json.loads(calls[0].function.arguments) is not None, done)
            expect(done.choices[0].finish_reason == 'tool_calls', done.choices[0].finish_reason)
            return {'tool': calls[0].function.name, 'served_by': raw.headers.get('x-titanos-served-by')}

        def chat_plain():
            done = c.chat.completions.create(model='mini', messages=CHAT)
            expect(done.choices[0].message.content, done)
            expect(done.usage and done.usage.total_tokens > 0, done.usage)
            return {'text': done.choices[0].message.content[:40]}

        def chat_stream_tools():
            calls, finish, usage, chunks = {}, None, None, 0
            with c.chat.completions.create(model='gpt-4o-mini@openai', messages=CHAT, tools=[WEATHER_TOOL],
                                           stream=True, stream_options={'include_usage': True}) as stream:
                for chunk in stream:
                    chunks += 1
                    if chunk.usage is not None:
                        usage = chunk.usage
                    for choice in chunk.choices:
                        finish = choice.finish_reason or finish
                        for delta in choice.delta.tool_calls or []:
                            entry = calls.setdefault(delta.index, {'name': '', 'arguments': ''})
                            if delta.function and delta.function.name:
                                entry['name'] += delta.function.name
                            if delta.function and delta.function.arguments:
                                entry['arguments'] += delta.function.arguments
            expect(calls and all(json.loads(v['arguments']) is not None and v['name'] for v in calls.values()), calls)
            expect(finish == 'tool_calls', finish)
            expect(usage is not None and usage.total_tokens > 0, usage)
            return {'chunks': chunks, 'tools': [v['name'] for v in calls.values()]}

        def chat_stream_text():
            with c.chat.completions.create(model='gpt-4o-mini@openai', messages=CHAT, stream=True) as stream:
                text = ''.join(ch.choices[0].delta.content or '' for ch in stream if ch.choices)
            expect(text, 'empty stream text')
            return {'text': text[:40]}

        def responses_plain():
            done = c.responses.create(model='gpt-4o-mini@openai', input='Say hello.', store=False)
            expect(done.status == 'completed' and done.output_text, done)
            stored = c.responses.create(model='gpt-4o-mini@openai', input='Say hello.')
            again = c.responses.retrieve(stored.id)
            expect(again.id == stored.id and again.output_text == stored.output_text, again)
            return {'text': done.output_text[:40], 'stored': stored.id}

        def responses_stream():
            with c.responses.create(model='gpt-4o-mini@openai', input='Say hello.', stream=True, store=False) as stream:
                events = list(stream)
            kinds = [e.type for e in events]
            expect(kinds[0] == 'response.created' and kinds[-1] == 'response.completed', kinds)
            numbers = [e.sequence_number for e in events]
            expect(numbers == sorted(numbers) and len(set(numbers)) == len(numbers), numbers)
            text = ''.join(e.delta for e in events if e.type == 'response.output_text.delta')
            expect(text and events[-1].response.output_text == text, (text, events[-1].response.output_text))
            with c.responses.stream(model='gpt-4o-mini@openai', input='Say hello.', store=False) as stream:
                deltas = [e.delta for e in stream if e.type == 'response.output_text.delta']
                final = stream.get_final_response()
            expect(''.join(deltas) == final.output_text, 'high-level stream text differs from the final response')
            return {'events': len(events), 'families': sorted(set(kinds))}

        def responses_stream_tools():
            tool = {'type': 'function', 'name': 'get_weather', 'description': 'Weather for a city',
                    'parameters': WEATHER_TOOL['function']['parameters']}
            with c.responses.create(model='gpt-4o-mini@openai', input='Weather in Paris?', tools=[tool], stream=True,
                                    store=False) as stream:
                events = list(stream)
            kinds = {e.type for e in events}
            expect('response.function_call_arguments.delta' in kinds or 'response.output_item.added' in kinds, kinds)
            final = events[-1].response
            calls = [item for item in final.output if item.type == 'function_call']
            expect(calls and json.loads(calls[0].arguments) is not None, final.output)
            return {'tool': calls[0].name}

        def embeddings():
            done = c.embeddings.create(model='text-embedding-3-small@openai', input=['one', 'two'])
            expect(len(done.data) == 2 and all(isinstance(x, float) for x in done.data[0].embedding), done)
            plain = c.embeddings.create(model='text-embedding-3-small@openai', input='one', encoding_format='float')
            expect(len(plain.data[0].embedding) == len(done.data[0].embedding), 'float and base64 dimensions differ')
            return {'dimensions': len(done.data[0].embedding)}

        def images():
            done = c.images.generate(model='dall-e-3@openai', prompt='A red square', response_format='b64_json',
                                     size='1024x1024', n=1)
            import base64
            data = base64.b64decode(done.data[0].b64_json)
            expect(data[:8] == b'\x89PNG\r\n\x1a\n', data[:8])
            return {'png_bytes': len(data)}

        def speech():
            raw = c.audio.speech.with_raw_response.create(model='tts-1@openai', voice='alloy', input='Hello there.',
                                                          response_format='wav')
            body = raw.parse().content
            expect(body[:4] == b'RIFF', body[:16])
            expect(raw.headers.get('x-titanos-served-by') == 'cloud:openai', dict(raw.headers))
            with c.audio.speech.with_streaming_response.create(model='tts-1@openai', voice='alloy', input='Hello.',
                                                               response_format='mp3') as streamed:
                parts = list(streamed.iter_bytes())
            expect(sum(len(p) for p in parts) > 0, 'empty streamed speech')
            return {'wav_bytes': len(body), 'mp3_chunks': len(parts)}

        def transcription():
            done = c.audio.transcriptions.create(model='whisper-1@openai', file=('a.wav', wav_tone()))
            expect(isinstance(done.text, str) and done.text, done)
            text = c.audio.transcriptions.create(model='whisper-1@openai', file=('a.wav', wav_tone()),
                                                 response_format='text')
            expect(isinstance(text, str) and text.strip(), repr(text))
            return {'text': done.text[:40]}

        def moderations():
            done = c.moderations.create(model='omni-moderation-latest@openai', input='hello')
            expect(len(done.results) == 1 and isinstance(done.results[0].flagged, bool), done)
            return {'flagged': done.results[0].flagged}

        def translated_chat():
            out = {}
            for model, served in (('claude-sonnet-4-5@anthropic', 'cloud:anthropic'),
                                  ('gemini-2.0-flash@gemini', 'cloud:gemini')):
                raw = c.chat.completions.with_raw_response.create(model=model, messages=CHAT, tools=[WEATHER_TOOL])
                expect(raw.headers.get('x-titanos-served-by') == served, dict(raw.headers))
                calls = raw.parse().choices[0].message.tool_calls or []
                expect(calls and json.loads(calls[0].function.arguments) is not None, raw.parse())
                streamed = {}
                with c.chat.completions.create(model=model, messages=CHAT, tools=[WEATHER_TOOL], stream=True) as stream:
                    for chunk in stream:
                        for choice in chunk.choices:
                            for delta in choice.delta.tool_calls or []:
                                entry = streamed.setdefault(delta.index, '')
                                streamed[delta.index] = entry + ((delta.function and delta.function.arguments) or '')
                expect(streamed and all(json.loads(v) is not None for v in streamed.values()), streamed)
                out[model] = {'tool': calls[0].function.name, 'translation': raw.headers.get('x-titanos-translation')}
            return out

        def translated_audio():
            raw = c.audio.speech.with_raw_response.create(model='eleven_multilingual_v2@elevenlabs', voice='alloy',
                                                          input='Hello there.', response_format='mp3')
            expect(raw.headers.get('x-titanos-served-by') == 'cloud:elevenlabs', dict(raw.headers))
            audio = raw.parse().content
            expect(len(audio) > 0, 'empty translated speech')
            done = c.audio.transcriptions.create(model='scribe_v1@elevenlabs', file=('a.wav', wav_tone()))
            expect(isinstance(done.text, str) and done.text, done)
            return {'speech_bytes': len(audio), 'text': done.text[:40],
                    'translation': raw.headers.get('x-titanos-translation')}

        def models_retrieve_bare():
            one = c.models.retrieve('gpt-4o-mini')  # a bare id resolves as it does for chat (single OpenAI upstream)
            expect(one.id == 'gpt-4o-mini' and one.owned_by == 'openai', one)
            return {'id': one.id, 'owned_by': one.owned_by}

        def videos():
            # base_url .../v1 is also Titanos' local video job route: a sora model goes to OpenAI, ids follow it
            created = c.videos.create(model='sora-2', prompt='A paper boat on a pond.', seconds='4')
            expect(created.id.startswith('video_mock') and created.model == 'sora-2', created)
            again = c.videos.retrieve(created.id)
            expect(again.id == created.id, again)
            content = c.videos.download_content(created.id).read()
            expect(content.startswith(b'\x00\x00\x00\x18ftyp'), content[:16])
            gone = c.videos.delete(created.id)
            expect(gone.deleted is True, gone)
            return {'id': created.id, 'content_bytes': len(content)}

        def mcp():
            return mcp_cases(url, tokens['sdk'])

        cases += [('openai.models.retrieve_bare', models_retrieve_bare), ('openai.videos', videos)]
        cases += [('openai.models', models), ('openai.chat.tools', chat_tools), ('openai.chat.alias', chat_plain),
                  ('openai.chat.stream_tools', chat_stream_tools), ('openai.chat.stream', chat_stream_text),
                  ('openai.responses', responses_plain), ('openai.responses.stream', responses_stream),
                  ('openai.responses.stream_tools', responses_stream_tools), ('openai.embeddings', embeddings),
                  ('openai.images', images), ('openai.audio.speech', speech),
                  ('openai.audio.transcriptions', transcription), ('openai.moderations', moderations),
                  ('openai.translated.chat_tools', translated_chat), ('openai.translated.audio', translated_audio),
                  ('mcp.streamable_http', mcp)]

        local = spec.get('local') or {}

        def need(name):
            if not local.get(name):
                raise Untried(f'no local fixture model {name} (set {FIXTURE_ENV})')

        def local_chat():
            need('local-chat')
            raw = c.chat.completions.with_raw_response.create(model='local-chat', messages=CHAT, max_tokens=8)
            expect(raw.headers.get('x-titanos-served-by', '').startswith('local:'), dict(raw.headers))
            done = raw.parse()
            expect(isinstance(done.choices[0].message.content, str), done)
            with c.chat.completions.create(model='local-chat', messages=CHAT, max_tokens=8, stream=True) as stream:
                pieces = [ch.choices[0].delta.content or '' for ch in stream if ch.choices]
            return {'served_by': raw.headers.get('x-titanos-served-by'), 'stream_chunks': len(pieces)}

        def local_embed():
            need('local-embed')
            done = c.embeddings.create(model='local-embed', input=['one', 'two'])
            expect(len(done.data) == 2 and len(done.data[0].embedding) > 8, done)
            return {'dimensions': len(done.data[0].embedding)}

        def local_speech():
            need('local-voice')
            body = c.audio.speech.create(model='local-voice', voice='default', input='Hello.',
                                         response_format='wav').read()
            expect(body[:4] == b'RIFF', body[:16])
            return {'wav_bytes': len(body)}

        def local_asr():
            need('local-asr')
            done = c.audio.transcriptions.create(model='local-asr', file=('a.wav', wav_tone(1.0)))
            expect(isinstance(done.text, str), done)
            return {'text': done.text[:40]}

        cases += [('local.openai.chat', local_chat), ('local.openai.embeddings', local_embed),
                  ('local.openai.audio.speech', local_speech), ('local.openai.audio.transcriptions', local_asr)]

    elif phase == 'refusals':
        def unknown_model():
            exc = status_of(lambda: client().chat.completions.create(model='no-such-model@openai', messages=CHAT))
            expect(isinstance(exc, openai.NotFoundError), type(exc))
            return {'status': exc.status_code, 'code': (exc.body or {}).get('code')}

        def wrong_key():
            exc = status_of(lambda: client('tk_wrong_' + 'x' * 32).models.list())
            expect(isinstance(exc, openai.AuthenticationError), type(exc))
            return {'status': exc.status_code}

        def no_cloud_scope():
            exc = status_of(lambda: client(tokens['local']).chat.completions.create(model='gpt-4o-mini@openai',
                                                                                   messages=CHAT))
            expect(isinstance(exc, (openai.PermissionDeniedError, openai.NotFoundError)), type(exc))
            return {'status': exc.status_code, 'code': (exc.body or {}).get('code')}

        def budget():
            sent.clear()
            exc = status_of(lambda: client(tokens['budget'], retries=2, counted=True).chat.completions.create(
                model='gpt-4o-mini@openai', messages=CHAT))
            expect(exc.status_code == 402, exc.status_code)
            expect((exc.body or {}).get('code') in ('budget_exceeded', 'budget_unpriced_target'), exc.body)
            expect(exc.response.headers.get('x-should-retry') == 'false', dict(exc.response.headers))
            expect(len(sent) == 1, f'the SDK sent {len(sent)} requests for a 402')
            return {'status': 402, 'code': exc.body.get('code'), 'requests_sent': len(sent)}

        def loop():
            sent.clear()
            exc = status_of(lambda: client(retries=2, counted=True).chat.completions.create(
                model='gpt-4o-mini@openai', messages=CHAT, extra_headers={'X-Titanos-Hop': spec['hop']}))
            expect(exc.status_code == 508, exc.status_code)
            expect((exc.body or {}).get('code') == 'loop_detected', exc.body)
            expect(exc.response.headers.get('x-should-retry') == 'false', dict(exc.response.headers))
            expect(len(sent) == 1, f'the SDK sent {len(sent)} requests for a 508')
            return {'status': 508, 'requests_sent': len(sent)}

        cases += [('errors.not_found', unknown_model), ('errors.authentication', wrong_key),
                  ('errors.scope', no_cloud_scope), ('errors.budget_402_not_retried', budget),
                  ('errors.loop_508_not_retried', loop)]

    elif phase.startswith('upstream.'):
        def rate_limited():
            exc = status_of(lambda: client().chat.completions.create(model='gpt-4o-mini@openai', messages=CHAT))
            expect(isinstance(exc, openai.RateLimitError), type(exc))
            expect(exc.response.headers.get('retry-after'), dict(exc.response.headers))
            return {'status': 429, 'retry_after': exc.response.headers.get('retry-after'),
                    'origin': exc.response.headers.get('x-titanos-error-origin')}

        def bad_request():
            exc = status_of(lambda: client().chat.completions.create(model='gpt-4o-mini@openai', messages=CHAT))
            expect(isinstance(exc, openai.BadRequestError), type(exc))
            text = json.dumps(exc.body) + str(exc) + exc.response.text
            leaked = sorted(name for name, secret in spec['secrets'].items() if secret in text)
            expect(not leaked, f'secret values reached the SDK error: {leaked}')
            return {'status': 400, 'origin': exc.response.headers.get('x-titanos-error-origin'),
                    'body_passed_through': 'provided key' in text and 'trace' in text}

        def cut_stream():
            got = []
            try:
                with client().chat.completions.create(model='gpt-4o-mini@openai', messages=CHAT,
                                                      stream=True) as stream:
                    for chunk in stream:
                        got.append(chunk)
            except openai.APIError as exc:
                return {'chunks_before_error': len(got), 'error': type(exc).__name__,
                        'status': getattr(exc, 'status_code', None), 'message': str(exc)[:200]}
            raise AssertionError(f'the cut stream ended without an SDK error after {len(got)} chunks')

        cases += [{'upstream.400': ('upstream.400_scrubbed', bad_request),
                   'upstream.cut': ('upstream.stream_cut', cut_stream),
                   'upstream.429': ('upstream.429', rate_limited)}[phase]]

    return {'sdk': {'openai': version('openai')}, 'missing': [], 'cases': run_cases(cases)}


def mcp_cases(url: str, token: str) -> dict:
    try:
        from mcp import Client
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
    except ImportError as exc:
        raise Untried(f'the official MCP client is not importable here ({exc})')
    import asyncio

    async def go():
        http = create_mcp_http_client(headers={'Authorization': 'Bearer ' + token})
        async with http:
            async with Client(streamable_http_client(url + '/mcp/', http_client=http), read_timeout_seconds=60) as mcp:
                chat = await mcp.call_tool('chat', {'payload': {'model': 'mini', 'messages': CHAT}})
                explain = await mcp.call_tool('broker_explain', {'surface': 'openai.chat',
                                                                 'body': {'model': 'gpt-4o-mini@openai',
                                                                          'messages': CHAT}})
                status = await mcp.call_tool('broker_status', {})
                return chat, explain, status
    chat, explain, status = asyncio.run(go())
    expect(not chat.is_error, str(chat.content)[:300])
    expect((chat.meta or {}).get('served_by') == 'cloud:openai', chat.meta)
    expect(chat.structured_content.get('object') == 'chat.completion', chat.structured_content)
    expect(not explain.is_error, str(explain.content)[:300])
    expect(not status.is_error, str(status.content)[:300])
    from importlib.metadata import version
    return {'mcp': version('mcp'), 'served_by': chat.meta.get('served_by'),
            'served_model': chat.meta.get('served_model')}


# =====================================================================================================================
# server side: the project interpreter (titanos, the mocks and the netguard)

def _scrubbed_env() -> dict:
    """os.environ without provider credentials, base URLs, proxies or Titanos settings (the children get only what the
    lane sets explicitly)."""
    out = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper.startswith('TITANOS_') or upper.endswith('_BASE_URL') or 'PROXY' in upper or 'API_KEY' in upper \
                or 'TOKEN' in upper or 'SECRET' in upper or upper in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'):
            continue
        out[name] = value
    return out


def _port() -> int:
    """A free loopback port below the Windows dynamic range (49152+), so it never collides with ephemeral ports."""
    rng = random.Random()
    for _ in range(200):
        port = rng.randrange(20000, 45000)
        sock = socket.socket()
        try:
            sock.bind(('127.0.0.1', port))
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError('no free loopback port')


def fixture_models() -> dict:
    """{local model name: ModelSpec dict} for the fixture directories present under TITANOS_SDK_FIXTURE_MODELS."""
    base = os.environ.get(FIXTURE_ENV, '').strip()
    out = {}
    if not base:
        return out
    for name, (folder, task) in LOCAL_DIRS.items():
        path = Path(base) / folder
        if (path / 'config.json').is_file():
            out[name] = {'path': str(path), 'task': task, 'device': 'cpu'}
    return out


class Stack:
    """Mocks + temporary TITANOS_HOME + a real serve subprocess. Use as a context manager."""

    BOOT = 'sdk-lane-bootstrap-' + os.urandom(8).hex()

    def __init__(self, *, local: bool = True):
        self.local_specs = fixture_models() if local else {}
        self.root = Path(tempfile.mkdtemp(prefix='titanos-sdk-lane-'))
        self.home = self.root / 'home'
        self.home.mkdir()
        self.process = None
        self.mocks: dict = {}
        self.tokens: dict = {}
        self.cred_env: dict = {}
        self.log_path = self.root / 'serve.log'
        self.reports = {'server': self.root / 'netguard-server.jsonl', 'client': self.root / 'netguard-client.jsonl'}
        self.clean_shutdown = None

    # setup -----------------------------------------------------------------------------------------------------------
    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self) -> None:
        for entry in (str(ROOT / 'src'), str(TESTS)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        from broker_mocks import AnthropicMock, ElevenLabsMock, GeminiMock, OpenAIMock, SENTINELS, netguard
        netguard.install()
        netguard.reset_attempts()
        self.netguard, self.sentinels = netguard, SENTINELS
        self.mocks = {'openai': OpenAIMock(), 'anthropic': AnthropicMock(), 'gemini': GeminiMock(),
                      'elevenlabs': ElevenLabsMock()}
        from broker_mocks.native_routes import install_gemini_native, install_openai_videos
        install_gemini_native(self.mocks['gemini'])  # the resumable Files upload (native passthrough)
        install_openai_videos(self.mocks['openai'])  # OpenAI videos through base_url /v1 (native passthrough)
        for item in self.mocks.values():
            item.start()
        self._write_home()
        self._serve()
        self._mint()

    def _write_home(self) -> None:
        from titanos.broker.config import BrokerConfig, write_broker_config
        from titanos.broker.router import hop_value
        from titanos.broker.vault.store import Vault
        from titanos.settings import Settings
        m = self.mocks
        models = {'openai': ['gpt-4o-mini', 'text-embedding-3-small', 'tts-1', 'whisper-1', 'dall-e-3',
                             'omni-moderation-latest'],
                  'anthropic': ['claude-sonnet-4-5'], 'gemini': ['gemini-2.0-flash', 'text-embedding-004'],
                  'elevenlabs': ['eleven_multilingual_v2', 'eleven_flash_v2_5', 'scribe_v1']}
        upstreams = {}
        for uid, mock in m.items():
            upstreams[uid] = {'profile': uid, 'base_url': mock.api_base, 'enabled': True, 'credential': 'sdk-' + uid,
                              'models': models[uid]}
        upstreams['openai']['data_policy'] = {'allow_store': True}  # the Responses store/retrieve case
        upstreams['gemini']['data_policy'] = {'allow_store': True}  # google-genai files.upload (native passthrough)
        document = {
            'policy': {'cloud_policy': 'explicit'},
            'upstreams': upstreams,
            'models': {'mini': {'target': 'gpt-4o-mini@openai'}},
            'voice_maps': {'elevenlabs_to_openai': {ELEVENLABS_VOICE: 'alloy'},
                           'openai_to_elevenlabs': {'alloy': ELEVENLABS_VOICE}},
            'prices': [{'match': 'gpt-4o-mini@openai', 'input_per_mtok': '0.15', 'output_per_mtok': '0.60'}],
        }
        cfg = BrokerConfig.model_validate(document)
        write_broker_config(self.home / 'broker.json', cfg)
        self.cred_env = {f'SDK_LANE_CRED_{uid.upper()}': self.sentinels[uid] for uid in m}
        settings = Settings(broker_egress='loopback-only', broker_vault_backend='refs-only')
        vault = Vault(settings, self.home, environ=dict(self.cred_env))
        for uid, upstream in cfg.upstreams.items():
            vault.put_env_ref(upstream.credential, f'SDK_LANE_CRED_{uid.upper()}', upstream_id=uid,
                              origin=upstream.origin(allow_http_lan=False))
            vault.mark_validated(upstream.credential, True, 'skipped', None)
        self.hop = hop_value(vault.instance_id())
        registry = {'models': self.local_specs}
        self.models_json = self.root / 'models.json'
        self.models_json.write_text(json.dumps(registry, indent=2), encoding='utf-8')

    def server_env(self) -> dict:
        env = _scrubbed_env()
        env.update(self.cred_env)
        env.update({'TITANOS_HOME': str(self.home), 'TITANOS_BROKER_EGRESS': 'loopback-only',
                    'TITANOS_BROKER_VAULT_BACKEND': 'refs-only', 'TITANOS_API_KEY': self.BOOT,
                    'TITANOS_TEST_NETGUARD': '1', 'TITANOS_TEST_NETGUARD_REPORT': str(self.reports['server']),
                    'PYTHONPATH': os.pathsep.join([str(NETGUARD_SITE), str(ROOT / 'src')]),
                    'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8', 'HF_HUB_OFFLINE': '1',
                    'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'})
        return env

    def _serve(self) -> None:
        self.port = _port()
        self.url = f'http://127.0.0.1:{self.port}'
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
        self._log = open(self.log_path, 'wb')
        self.process = subprocess.Popen(
            [sys.executable, '-B', '-m', 'titanos', '--config', str(self.models_json), 'serve', '--host', '127.0.0.1',
             '--port', str(self.port), '--log-level', 'warning'],
            cwd=str(self.root), env=self.server_env(), stdout=self._log, stderr=subprocess.STDOUT,
            creationflags=flags, start_new_session=sys.platform != 'win32')
        import httpx
        deadline = time.monotonic() + 120
        with httpx.Client(base_url=self.url, trust_env=False, timeout=5) as http:
            while True:
                try:
                    if http.get('/healthz').status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                if self.process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('titanos serve did not start: ' + self.server_log()[-1500:])
                time.sleep(0.2)

    def _mint(self) -> None:
        import httpx
        specs = {
            'sdk': {'scopes': ['inference', 'cloud', 'cloud:store', 'models:read', 'mcp', 'broker:read', 'native']},
            'local': {'scopes': ['inference', 'models:read']},
            'budget': {'scopes': ['inference', 'cloud'], 'budget': {'tokens_per_day': 1}},
        }
        with httpx.Client(base_url=self.url, trust_env=False, timeout=30,
                          headers={'Authorization': 'Bearer ' + self.BOOT}) as http:
            for name, body in specs.items():
                reply = http.post('/api/keys', json={'name': 'sdk-lane-' + name, **body})
                if reply.status_code != 201:
                    raise RuntimeError(f'minting {name} failed: {reply.status_code} {reply.text[:300]}')
                self.tokens[name] = reply.json()['secret']

    # running the client ----------------------------------------------------------------------------------------------
    def client_env(self) -> dict:
        env = _scrubbed_env()
        env.update({'TITANOS_TEST_NETGUARD': '1', 'TITANOS_TEST_NETGUARD_REPORT': str(self.reports['client']),
                    'PYTHONPATH': str(NETGUARD_SITE), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8'})
        return env

    def spec(self, **extra) -> dict:
        return {'url': self.url, 'tokens': dict(self.tokens), 'hop': self.hop,
                'local': {name: True for name in self.local_specs}, 'secrets': self.named_secrets(), **extra}

    def named_secrets(self) -> dict:
        """{label: value} of every credential this lane hands out: the upstream sentinels held by the server, the
        bootstrap key and each minted token (labels are safe to print). SENTINELS['leak'] is not among them: it is a
        marker the mocks' 400 bodies carry, unknown to the server, so it may pass through with a deterministic 4xx."""
        out = {'sentinel:' + name: self.sentinels[name] for name in self.mocks}
        out['bootstrap'] = self.BOOT
        out.update({'token:' + name: value for name, value in self.tokens.items()})
        return out

    def secrets(self) -> list:
        return sorted(set(self.named_secrets().values()))

    def guard_active(self, python: str, env: dict) -> bool:
        """True when `python` started with `env` loads the netguard sitecustomize and has it installed."""
        probe = subprocess.run([python, '-B', '-c', 'import sys; g = sys.modules.get("titanos_test_netguard"); '
                                'print(bool(g and g.installed()))'], capture_output=True, env=env, timeout=120)
        return probe.stdout.decode('utf-8', errors='replace').strip().endswith('True')

    def surface_check(self, expected: dict) -> dict:
        """Each {upstream id: [mock surface]} was reached at least once (by a 2xx answer)."""
        missing = [f'{uid}:{surface}' for uid, surfaces in expected.items() for surface in surfaces
                   if not any(hit.status and hit.status < 300 for hit in self.mocks[uid].hits_for(surface))]
        reached = {uid: sorted({hit.surface for hit in self.mocks[uid].hits if hit.surface}) for uid in expected}
        return {'id': 'mocks.surfaces_reached', 'state': 'fail' if missing else 'pass',
                'detail': {'missing': missing} if missing else reached}

    def hit_total(self) -> int:
        return sum(len(m.hits) for m in self.mocks.values())

    # teardown and evidence -------------------------------------------------------------------------------------------
    def server_log(self) -> str:
        try:
            return self.log_path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return ''

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            import signal
            try:
                if sys.platform == 'win32':
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.process.send_signal(signal.SIGINT)
                self.process.wait(30)
                # graceful: uvicorn drains, runs the lifespan shutdown, then re-raises the signal it caught; the
                # Windows CRT ends a re-raised SIGBREAK with exit code 3, POSIX a re-raised SIGINT with -2 (or 130)
                self.clean_shutdown = self.process.returncode in ((0, 3) if sys.platform == 'win32' else (0, -2, 130))
            except Exception:
                self.clean_shutdown = False
                self.process.kill()
                self.process.wait(30)
        if getattr(self, '_log', None) is not None:
            self._log.close()
            self._log = None
        for item in self.mocks.values():
            try:
                item.stop()
            except Exception:
                pass

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def blocked_attempts(self) -> list:
        out = list(self.netguard.attempts())
        for path in self.reports.values():
            if path.is_file():
                out += [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        return out

    def hit_problems(self) -> list:
        """Every mock hit carries exactly its own sentinel and never a tk_ token or the bootstrap key."""
        problems = []
        inbound = [self.BOOT, *self.tokens.values()]
        for uid, mock in self.mocks.items():
            own = f'<sentinel:{uid}>'
            for hit in mock.hits:
                spelled = [value for _, value in hit.credentials]
                if not spelled or any(own not in value for value in spelled):
                    problems.append(f'{uid} {hit.path}: credentials {spelled}')
                if any(hit.presented(token) for token in inbound):
                    problems.append(f'{uid} {hit.path}: an inbound Titanos credential was forwarded')
                if hit.duplicate_credentials:
                    problems.append(f'{uid} {hit.path}: duplicate credential headers {hit.duplicate_credentials}')
        return problems


def find_python(env_names: tuple, modules: tuple) -> tuple[str, list]:
    """(interpreter, missing modules): the first env_names entry that is set, else this interpreter."""
    python = next((os.environ[name] for name in env_names if os.environ.get(name, '').strip()), sys.executable)
    missing = []
    for module in modules:
        probe = subprocess.run([python, '-c', f'import {module}'], capture_output=True, env=_scrubbed_env(),
                               timeout=120)
        if probe.returncode:
            missing.append(module)
    return python, missing


def run_client(stack: Stack, python: str, script: Path, phase: str, spec: dict, timeout: float = 900) -> dict:
    proc = subprocess.run([python, '-B', str(script), '--client', phase], input=json.dumps(spec).encode('utf-8'),
                          capture_output=True, env=stack.client_env(), timeout=timeout, cwd=str(stack.root))
    out = proc.stdout.decode('utf-8', errors='replace')
    err = proc.stderr.decode('utf-8', errors='replace')
    try:
        result = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        result = {'cases': [{'id': f'client.{phase}', 'state': 'fail',
                             'detail': f'client exit {proc.returncode}: {(err or out)[-1500:]}'}]}
    result['raw'] = out + err
    return result


LANE_CHECKS = {'openai-sdk': 'check_broker_sdk.py', 'vendor-sdks': 'check_broker_sdk_vendors.py'}


def report_matrix(lane: str, cases: list) -> None:
    """Under the broker matrix run (TITANOS_BROKER_RESULTS set by tools/build_broker_matrix.py --run) append this lane's
    case outcomes, so the matrix can cite them by case id: pass -> passed, fail -> failed, untried -> skipped with the
    reason (never a pass)."""
    target = os.environ.get('TITANOS_BROKER_RESULTS')
    if not target:
        return
    line = {'check': LANE_CHECKS[lane], 'passed': sorted(c['id'] for c in cases if c['state'] == 'pass'),
            'failed': sorted(c['id'] for c in cases if c['state'] == 'fail'),
            'skipped': {c['id']: 'untried: ' + str(c.get('detail') or '')[:150] for c in cases
                        if c['state'] == 'untried'}}
    with open(target, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(line, sort_keys=True) + '\n')


def summarize(lane: str, stack: Stack, results: list, extra_checks: list, strict: bool, sdk: dict,
              client_python: str) -> int:
    """Print the lane report and return the exit code."""
    cases = [case for result in results for case in result.get('cases', [])]
    checks = list(extra_checks)
    guards = {'server': stack.guard_active(sys.executable, stack.server_env()),
              'client': stack.guard_active(client_python, stack.client_env())}
    checks.append({'id': 'netguard.active', 'state': 'pass' if all(guards.values()) else 'fail', 'detail': guards})
    attempts = stack.blocked_attempts()
    checks.append({'id': 'netguard.zero_non_loopback', 'state': 'fail' if attempts else 'pass',
                   'detail': attempts[:5] if attempts else 'server, client and harness made no non-loopback attempt'})
    problems = stack.hit_problems()
    checks.append({'id': 'mocks.credentials_exact', 'state': 'fail' if problems else 'pass',
                   'detail': problems[:5] if problems else {uid: len(m.hits) for uid, m in stack.mocks.items()}})
    texts = {'server_output': stack.server_log(), 'client_output': ''.join(r.get('raw', '') for r in results)}
    leaked = [f'{label} in {where}' for where, text in texts.items() for label, secret in stack.named_secrets().items()
              if secret in text]
    checks.append({'id': 'outputs.no_secret_values', 'state': 'fail' if leaked else 'pass',
                   'detail': sorted(set(leaked)) or 'no sentinel, tk_ token or bootstrap value in any output'})
    code = stack.process.returncode if stack.process else None
    checks.append({'id': 'serve.clean_shutdown', 'state': 'pass' if stack.clean_shutdown else 'fail',
                   'detail': f'exit code {code} after the interrupt' if stack.clean_shutdown
                   else f'exit code {code}: {stack.server_log()[-800:]}'})
    everything = cases + checks
    states = {state: sum(1 for c in everything if c['state'] == state) for state in ('pass', 'fail', 'untried')}
    failed = states['fail'] > 0 or (strict and states['untried'] > 0)
    result = 'fail' if states['fail'] else 'untried' if states['untried'] else 'pass'  # untried is never a pass
    report_matrix(lane, everything)
    print(json.dumps({'lane': lane, 'result': result, 'counts': states, 'sdk': sdk,
                      'local_models': sorted(stack.local_specs), 'cases': everything,
                      'scope': 'a real titanos serve subprocess, loopback mocks and local fixtures; no live provider'},
                     indent=1))
    for case in everything:
        if case['state'] != 'pass':
            print(f"{case['state'].upper()}: {case['id']}: {case['detail']}", file=sys.stderr)
    return 1 if failed else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    strict = '--strict' in argv
    python, missing = find_python((SDK_PYTHON_ENV,), ('openai',))
    if missing:
        cases = [{'id': 'openai.sdk', 'state': 'untried',
                  'detail': f'openai is not importable by {SDK_PYTHON_ENV} or this interpreter'}]
        report_matrix('openai-sdk', cases)
        print(json.dumps({'lane': 'openai-sdk', 'result': 'untried', 'counts': {'untried': 1}, 'cases': cases}))
        return 1 if strict else 0
    stack = Stack()
    try:
        with stack:
            results, checks = [], []
            results.append(run_client(stack, python, Path(__file__), 'main', stack.spec()))
            checks.append(stack.surface_check({'openai': OPENAI_SURFACES, 'anthropic': ['anthropic.messages'],
                                               'gemini': ['gemini.generateContent', 'gemini.streamGenerateContent'],
                                               # translated OpenAI speech streams through ElevenLabs' /stream path
                                               'elevenlabs': ['elevenlabs.tts_stream', 'elevenlabs.stt']}))
            before = stack.hit_total()
            results.append(run_client(stack, python, Path(__file__), 'refusals', stack.spec()))
            added = stack.hit_total() - before
            checks.append({'id': 'refusals.zero_upstream_hits', 'state': 'pass' if added == 0 else 'fail',
                           'detail': f'{added} mock hits during the refusal cases'})
            openai = stack.mocks['openai']
            # one fault per client run, so each SDK call meets exactly its own fault; the 429 goes last because its
            # Retry-After cools the upstream down and a request right after it is refused locally
            for phase, kind, opts in (('upstream.400', 'status', {'status': 400}),
                                      ('upstream.cut', 'disconnect_after_frames', {'frames': 2}),
                                      ('upstream.429', 'status', {'status': 429, 'retry_after': 3})):
                before = len(openai.hits_for('openai.chat'))
                openai.faults.push(kind, surface='openai.chat', **opts)
                results.append(run_client(stack, python, Path(__file__), phase, stack.spec()))
                hits = len(openai.hits_for('openai.chat')) - before
                pending = [f.kind for f in openai.faults.pending()]
                openai.faults.clear()
                checks.append({'id': phase + '.one_upstream_hit',
                               'state': 'pass' if hits == 1 and not pending else 'fail',
                               'detail': f'{hits} openai.chat hit(s) with max_retries=0; unserved faults {pending}'})
        sdk = {}
        for result in results:
            sdk.update(result.get('sdk') or {})
        return summarize('openai-sdk', stack, results, checks, strict, sdk, python)
    finally:
        stack.cleanup()


def client_entry(argv) -> int:
    phase = argv[argv.index('--client') + 1]
    spec = json.loads(sys.stdin.read())
    result = openai_client_main(phase, spec)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    if '--client' in sys.argv:
        sys.exit(client_entry(sys.argv))
    sys.exit(main())
