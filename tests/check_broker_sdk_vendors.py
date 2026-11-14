"""Broker vendor-SDK lane (sdk tier): the official anthropic, google-genai and elevenlabs Python SDKs, unmodified (only
base_url and api_key change), against a REAL `titanos serve` subprocess routed to the WP4 loopback mocks and, where
TITANOS_SDK_FIXTURE_MODELS names them, to local CPU fixture models.

The harness (mocks, temporary TITANOS_HOME, env: credentials bound to the mocks' exact origins, the serve subprocess
under the netguard sitecustomize, minted tk_ keys and the closing assertions) is tests/check_broker_sdk.py's Stack. The
SDK client is this file with --client, run by the interpreter named by TITANOS_SDK_PYTHON_EXT (or
TITANOS_SDK_PYTHON_EXTRA; default: this interpreter, as in CI after
`python -m pip install -r tests/requirements-sdk.txt`).
Each SDK authenticates with its own native header (anthropic x-api-key, google-genai x-goog-api-key, elevenlabs
xi-api-key); the auth cases record which header carried the tk_ token (by name only).

Cases (each pass, fail or untried; an SDK that is not importable makes its cases untried, never a pass):
- anthropic: messages non-stream and stream with tools passed through to the Anthropic mock (tool_use blocks,
  provenance),
  the same SDK translated to an OpenAI upstream (stream and non-stream with tools), count_tokens passed through, typed
  errors (NotFoundError for an unknown model, AuthenticationError for a wrong key) and, locally, messages non-stream and
  stream on the chat fixture plus count_tokens on a transformers model (a typed 400: R22 counts llama-cpp models only);
- google-genai over /v1beta: generateContent with function declarations, streamGenerateContent (alt=sse), embedContent
  and countTokens through the Gemini mock, files.upload (the resumable upload over the native passthrough: the SDK sends
  its upload and finalize chunks to the rewritten session URL without a key, authorized by the upload capability
  Titanos put in it), generateContent and its stream translated to an OpenAI upstream, models.list, models.get with
  the bare id gemini-2.0-flash,
  a typed ClientError for an unknown model and, locally, generateContent, its stream and embedContent;
- elevenlabs: text-to-speech, streaming text-to-speech and speech-to-text through the ElevenLabs mock, the same two
  translated to OpenAI speech and transcription, voices (the local registry's speech voices), a typed ApiError for an
  unknown model and, locally, text-to-speech with and without model_id and speech-to-text.
- refusals through each SDK with its default retry policy: 402 (a budgeted key) and 508 loop_detected (our own
  X-Titanos-Hop) arrive as each SDK's typed error with zero upstream hits; the requests each SDK sent are counted
  (anthropic must send exactly one; an SDK that retries despite x-should-retry: false is reported as a finding).
Afterwards the mocks must have been reached on every surface above, and the shared closing checks hold (netguard,
exact credentials per mock, no secret in any output, clean shutdown).
Exit 1 on any fail, or with --strict on any untried.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_broker_sdk as H  # noqa: E402  (the shared harness; imports only the standard library at module level)

EXT_ENV = ('TITANOS_SDK_PYTHON_EXT', 'TITANOS_SDK_PYTHON_EXTRA')
MODULES = {'anthropic': 'anthropic', 'gemini': 'google.genai', 'elevenlabs': 'elevenlabs'}
VOICE = H.ELEVENLABS_VOICE
TOOL_SCHEMA = {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}
EXPECTED_SURFACES = {
    'anthropic': ['anthropic.messages', 'anthropic.count_tokens'],
    # google-genai's models.embed_content posts :batchEmbedContents on the Gemini API (one request for every content)
    'gemini': ['gemini.generateContent', 'gemini.streamGenerateContent', 'gemini.batchEmbedContents',
               'gemini.countTokens'],
    'elevenlabs': ['elevenlabs.tts', 'elevenlabs.tts_stream', 'elevenlabs.stt'],
    'openai': ['openai.chat', 'openai.audio.speech', 'openai.audio.transcriptions'],
}
expect, Untried = H.expect, H.Untried


# =====================================================================================================================
# client side (the vendor SDK interpreter)

def header_names(token: str, seen: list):
    """An httpx request hook recording the names (never the values) of headers carrying `token`."""
    def hook(request):
        seen.append(sorted(name.lower() for name, value in request.headers.items() if token in value))
    return hook


def anthropic_cases(spec: dict) -> list:
    try:
        import anthropic
    except ImportError as exc:
        return [('anthropic.sdk', _untried(f'anthropic is not importable here ({exc})'))]
    url, token = spec['url'], spec['tokens']['sdk']
    c = anthropic.Anthropic(base_url=url, api_key=token, max_retries=0, timeout=120)
    tools = [{'name': 'get_weather', 'description': 'Weather for a city', 'input_schema': TOOL_SCHEMA}]
    messages = [{'role': 'user', 'content': 'Weather in Paris?'}]
    local = spec.get('local') or {}

    def auth_header():
        seen: list = []
        hooked = anthropic.Anthropic(base_url=url, api_key=token, max_retries=0, timeout=120,
                                     http_client=anthropic.DefaultHttpxClient(event_hooks={'request': [
                                         header_names(token, seen)]}))
        done = hooked.messages.create(model='claude-sonnet-4-5@anthropic', max_tokens=64, messages=messages)
        expect(seen and seen[-1] == ['x-api-key'], seen)
        return {'type': done.type, 'token_header': seen[-1]}

    def passthrough_tools():
        raw = c.messages.with_raw_response.create(model='claude-sonnet-4-5@anthropic', max_tokens=256,
                                                  messages=messages, tools=tools)
        expect(raw.headers.get('x-titanos-served-by') == 'cloud:anthropic', dict(raw.headers))
        done = raw.parse()
        uses = [b for b in done.content if b.type == 'tool_use']
        expect(uses and uses[0].name and isinstance(uses[0].input, dict), done.content)
        expect(done.stop_reason == 'tool_use', done.stop_reason)
        return {'tool': uses[0].name, 'served_by': raw.headers.get('x-titanos-served-by')}

    def passthrough_stream_tools():
        with c.messages.stream(model='claude-sonnet-4-5@anthropic', max_tokens=256, messages=messages,
                               tools=tools) as stream:
            kinds = [event.type for event in stream]
            final = stream.get_final_message()
        uses = [b for b in final.content if b.type == 'tool_use']
        expect(uses and isinstance(uses[0].input, dict) and uses[0].input, final.content)
        expect('message_stop' in kinds or 'message_delta' in kinds, kinds)
        return {'events': len(kinds), 'tool': uses[0].name, 'stop_reason': final.stop_reason}

    def passthrough_text_stream():
        with c.messages.stream(model='claude-sonnet-4-5@anthropic', max_tokens=64,
                               messages=[{'role': 'user', 'content': 'Say hello.'}]) as stream:
            text = ''.join(stream.text_stream)
        expect(text, 'empty text stream')
        return {'text': text[:40]}

    def translated_tools():
        raw = c.messages.with_raw_response.create(model='gpt-4o-mini@openai', max_tokens=256, messages=messages,
                                                  tools=tools)
        expect(raw.headers.get('x-titanos-served-by') == 'cloud:openai', dict(raw.headers))
        done = raw.parse()
        uses = [b for b in done.content if b.type == 'tool_use']
        expect(uses and isinstance(uses[0].input, dict), done.content)
        expect(done.stop_reason == 'tool_use', done.stop_reason)
        return {'tool': uses[0].name}

    def translated_stream_tools():
        with c.messages.stream(model='gpt-4o-mini@openai', max_tokens=256, messages=messages, tools=tools) as stream:
            kinds = [event.type for event in stream]
            final = stream.get_final_message()
        uses = [b for b in final.content if b.type == 'tool_use']
        expect(uses and isinstance(uses[0].input, dict) and uses[0].input, final.content)
        expect(final.stop_reason == 'tool_use', final.stop_reason)
        return {'events': len(kinds), 'tool': uses[0].name}

    def count_tokens():
        done = c.messages.count_tokens(model='claude-sonnet-4-5@anthropic', messages=messages, tools=tools)
        expect(isinstance(done.input_tokens, int) and done.input_tokens > 0, done)
        return {'input_tokens': done.input_tokens}

    def errors():
        try:
            c.messages.create(model='no-such-model@anthropic', max_tokens=16, messages=messages)
            raise AssertionError('unknown model accepted')
        except anthropic.NotFoundError as exc:
            missing = {'status': exc.status_code, 'type': ((exc.body or {}).get('error') or {}).get('type')}
        try:
            anthropic.Anthropic(base_url=url, api_key='tk_wrong_' + 'x' * 32, max_retries=0).messages.create(
                model='claude-sonnet-4-5@anthropic', max_tokens=16, messages=messages)
            raise AssertionError('wrong key accepted')
        except anthropic.AuthenticationError as exc:
            wrong = {'status': exc.status_code, 'type': ((exc.body or {}).get('error') or {}).get('type')}
        return {'not_found': missing, 'authentication': wrong}

    def local_messages():
        if not local.get('local-chat'):
            raise Untried(f'no local chat fixture (set {H.FIXTURE_ENV})')
        raw = c.messages.with_raw_response.create(model='local-chat', max_tokens=8,
                                                  messages=[{'role': 'user', 'content': 'Say hello.'}])
        expect(raw.headers.get('x-titanos-served-by', '').startswith('local:'), dict(raw.headers))
        done = raw.parse()
        expect(done.type == 'message' and done.role == 'assistant', done)
        with c.messages.stream(model='local-chat', max_tokens=8,
                               messages=[{'role': 'user', 'content': 'Say hello.'}]) as stream:
            text = ''.join(stream.text_stream)
            final = stream.get_final_message()
        expect(final.stop_reason in ('end_turn', 'max_tokens'), final.stop_reason)
        return {'served_by': raw.headers.get('x-titanos-served-by'), 'stream_text': text[:40]}

    def local_count_tokens():
        if not local.get('local-chat'):
            raise Untried(f'no local chat fixture (set {H.FIXTURE_ENV})')
        try:
            c.messages.count_tokens(model='local-chat', messages=messages)
        except anthropic.BadRequestError as exc:
            return {'status': exc.status_code, 'type': ((exc.body or {}).get('error') or {}).get('type'),
                    'note': 'count_tokens is native only for llama-cpp models (R22); others get a typed 400'}
        raise AssertionError('count_tokens on a transformers model was answered')

    return [('anthropic.auth_header', auth_header), ('anthropic.messages.tools', passthrough_tools),
            ('anthropic.messages.stream_tools', passthrough_stream_tools),
            ('anthropic.messages.stream_text', passthrough_text_stream),
            ('anthropic.translated_openai.tools', translated_tools),
            ('anthropic.translated_openai.stream_tools', translated_stream_tools),
            ('anthropic.count_tokens', count_tokens), ('anthropic.errors', errors),
            ('local.anthropic.messages', local_messages), ('local.anthropic.count_tokens', local_count_tokens)]


def gemini_cases(spec: dict) -> list:
    try:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types
    except ImportError as exc:
        return [('gemini.sdk', _untried(f'google-genai is not importable here ({exc})'))]
    import httpx
    url, token = spec['url'], spec['tokens']['sdk']
    local = spec.get('local') or {}

    def client(key=None, **http):
        return genai.Client(api_key=key or token, http_options=types.HttpOptions(base_url=url, api_version='v1beta',
                                                                                 **http))
    c = client()
    weather = types.Tool(function_declarations=[types.FunctionDeclaration(
        name='get_weather', description='Weather for a city', parameters_json_schema=TOOL_SCHEMA)])
    config = types.GenerateContentConfig(tools=[weather])

    def auth_header():
        seen: list = []
        hooked = client(httpx_client=httpx.Client(event_hooks={'request': [header_names(token, seen)]}, timeout=120))
        done = hooked.models.generate_content(model='gemini-2.0-flash@gemini', contents='Say hello.')
        expect(seen and seen[-1] == ['x-goog-api-key'], seen)
        return {'text': (done.text or '')[:40], 'token_header': seen[-1]}

    def generate_tools():
        done = c.models.generate_content(model='gemini-2.0-flash@gemini', contents='Weather in Paris?', config=config)
        calls = done.function_calls or []
        expect(calls and calls[0].name == 'get_weather' and isinstance(calls[0].args, dict), done)
        return {'tool': calls[0].name, 'finish': str(done.candidates[0].finish_reason)}

    def generate_stream():
        chunks = list(c.models.generate_content_stream(model='gemini-2.0-flash@gemini', contents='Say hello.'))
        text = ''.join(chunk.text or '' for chunk in chunks)
        expect(len(chunks) >= 1 and text, chunks)
        return {'chunks': len(chunks), 'text': text[:40]}

    def stream_tools():
        chunks = list(c.models.generate_content_stream(model='gemini-2.0-flash@gemini', contents='Weather in Paris?',
                                                        config=config))
        calls = [call for chunk in chunks for call in (chunk.function_calls or [])]
        expect(calls and calls[0].name == 'get_weather', chunks)
        return {'chunks': len(chunks), 'tool': calls[0].name}

    def embed():
        done = c.models.embed_content(model='text-embedding-004@gemini', contents='one')
        values = done.embeddings[0].values
        expect(values and all(isinstance(v, float) for v in values), done)
        return {'dimensions': len(values)}

    def count_tokens():
        done = c.models.count_tokens(model='gemini-2.0-flash@gemini', contents='Weather in Paris?')
        expect(isinstance(done.total_tokens, int) and done.total_tokens > 0, done)
        return {'total_tokens': done.total_tokens}

    def files_upload():
        import os
        import tempfile
        handle, path = tempfile.mkstemp(suffix='.wav')
        try:
            with os.fdopen(handle, 'wb') as out:
                out.write(b'RIFF' + b'\x00' * 40)
            done = c.files.upload(file=path, config=types.UploadFileConfig(mime_type='audio/wav',
                                                                         display_name='clip'))
        finally:
            os.unlink(path)
        expect(isinstance(done.name, str) and done.name.startswith('files/'), done)
        return {'name': done.name, 'state': str(done.state)}

    def translated():
        done = c.models.generate_content(model='gpt-4o-mini@openai', contents='Weather in Paris?', config=config)
        calls = done.function_calls or []
        expect(calls and calls[0].name == 'get_weather' and isinstance(calls[0].args, dict), done)
        chunks = list(c.models.generate_content_stream(model='gpt-4o-mini@openai', contents='Weather in Paris?',
                                                        config=config))
        streamed = [call for chunk in chunks for call in (chunk.function_calls or [])]
        expect(streamed and streamed[0].name == 'get_weather' and isinstance(streamed[0].args, dict), chunks)
        text = ''.join(chunk.text or '' for chunk in c.models.generate_content_stream(model='gpt-4o-mini@openai',
                                                                                      contents='Say hello.'))
        expect(text, 'empty translated text stream')
        return {'tool': calls[0].name, 'stream_chunks': len(chunks), 'text': text[:40]}

    def models():
        names = [m.name for m in c.models.list()]
        expect(any('gemini-2.0-flash@gemini' in n for n in names), names[:20])
        return {'models': len(names)}

    def models_get_bare():
        one = c.models.get(model='gemini-2.0-flash')  # a bare id resolves as generate_content resolves it
        expect(one.name == 'models/gemini-2.0-flash', one)
        return {'name': one.name}

    def errors():
        try:
            c.models.generate_content(model='no-such-model@gemini', contents='hi')
            raise AssertionError('unknown model accepted')
        except genai_errors.ClientError as exc:
            missing = {'code': exc.code, 'status': exc.status}
        expect(missing['code'] == 404, missing)
        wrong_client = client('tk_wrong_' + 'x' * 32)  # keep a reference: a collected genai.Client closes its transport
        try:
            wrong_client.models.generate_content(model='gemini-2.0-flash@gemini', contents='hi')
            raise AssertionError('wrong key accepted')
        except genai_errors.ClientError as exc:
            wrong = {'code': exc.code, 'status': exc.status}
        expect(wrong['code'] in (401, 403), wrong)
        return {'not_found': missing, 'authentication': wrong}

    def local_generate():
        if not local.get('local-chat'):
            raise Untried(f'no local chat fixture (set {H.FIXTURE_ENV})')
        small = types.GenerateContentConfig(max_output_tokens=8)
        done = c.models.generate_content(model='local-chat', contents='Say hello.', config=small)
        expect(done.candidates and isinstance(done.text or '', str), done)
        chunks = list(c.models.generate_content_stream(model='local-chat', contents='Say hello.', config=small))
        expect(chunks, 'no local stream chunks')
        return {'text': (done.text or '')[:40], 'stream_chunks': len(chunks)}

    def local_embed():
        if not local.get('local-embed'):
            raise Untried(f'no local embeddings fixture (set {H.FIXTURE_ENV})')
        done = c.models.embed_content(model='local-embed', contents='one')
        expect(len(done.embeddings[0].values) > 8, done)
        return {'dimensions': len(done.embeddings[0].values)}

    return [('gemini.auth_header', auth_header), ('gemini.generateContent.tools', generate_tools),
            ('gemini.streamGenerateContent', generate_stream), ('gemini.streamGenerateContent.tools', stream_tools),
            ('gemini.embedContent', embed), ('gemini.countTokens', count_tokens), ('gemini.files.upload', files_upload),
            ('gemini.translated_openai', translated), ('gemini.models', models),
            ('gemini.models.get_bare', models_get_bare), ('gemini.errors', errors),
            ('local.gemini.generateContent', local_generate), ('local.gemini.embedContent', local_embed)]


def elevenlabs_cases(spec: dict) -> list:
    try:
        from elevenlabs import ElevenLabs
        from elevenlabs.core.api_error import ApiError
    except ImportError as exc:
        return [('elevenlabs.sdk', _untried(f'elevenlabs is not importable here ({exc})'))]
    import httpx
    url, token = spec['url'], spec['tokens']['sdk']
    local = spec.get('local') or {}
    c = ElevenLabs(api_key=token, base_url=url, timeout=120)

    def auth_header():
        seen: list = []
        hooked = ElevenLabs(api_key=token, base_url=url, httpx_client=httpx.Client(
            event_hooks={'request': [header_names(token, seen)]}, timeout=120))
        audio = b''.join(hooked.text_to_speech.convert(VOICE, text='Hello.',
                                                       model_id='eleven_multilingual_v2@elevenlabs'))
        expect(seen and seen[-1] == ['xi-api-key'], seen)
        return {'bytes': len(audio), 'token_header': seen[-1]}

    def tts():
        audio = b''.join(c.text_to_speech.convert(VOICE, text='Hello there.',
                                                  model_id='eleven_multilingual_v2@elevenlabs',
                                                  output_format='mp3_44100_128'))
        expect(len(audio) > 0, 'empty audio')
        pcm = b''.join(c.text_to_speech.convert(VOICE, text='Hello.', model_id='eleven_flash_v2_5@elevenlabs',
                                                output_format='pcm_16000'))
        expect(len(pcm) > 0, 'empty pcm audio')
        return {'mp3_bytes': len(audio), 'pcm_bytes': len(pcm)}

    def tts_stream():
        chunks = list(c.text_to_speech.stream(VOICE, text='Hello there.', model_id='eleven_multilingual_v2@elevenlabs'))
        expect(chunks and sum(len(x) for x in chunks) > 0, 'empty stream')
        return {'chunks': len(chunks), 'bytes': sum(len(x) for x in chunks)}

    def stt():
        done = c.speech_to_text.convert(model_id='scribe_v1@elevenlabs', file=('a.wav', H.wav_tone(), 'audio/wav'))
        expect(isinstance(done.text, str) and done.text, done)
        return {'text': done.text[:40], 'words': len(getattr(done, 'words', None) or [])}

    def translated():
        audio = b''.join(c.text_to_speech.convert(VOICE, text='Hello there.', model_id='tts-1@openai',
                                                  output_format='mp3_44100_128'))
        expect(len(audio) > 0, 'empty translated audio')
        done = c.speech_to_text.convert(model_id='whisper-1@openai', file=('a.wav', H.wav_tone(), 'audio/wav'))
        expect(isinstance(done.text, str) and done.text, done)
        return {'tts_bytes': len(audio), 'stt_text': done.text[:40]}

    def voices():
        done = c.voices.get_all()
        names = [v.voice_id for v in done.voices]
        if local.get('local-voice'):
            expect(names, 'no local speech voice listed although a local speech model is registered')
        return {'voices': len(names)}

    def errors():
        try:
            b''.join(c.text_to_speech.convert(VOICE, text='Hello.', model_id='no-such-model@elevenlabs'))
            raise AssertionError('unknown model accepted')
        except ApiError as exc:
            missing = exc.status_code
        expect(missing == 404, missing)
        try:
            b''.join(ElevenLabs(api_key='tk_wrong_' + 'x' * 32, base_url=url).text_to_speech.convert(
                VOICE, text='Hello.', model_id='eleven_multilingual_v2@elevenlabs'))
            raise AssertionError('wrong key accepted')
        except ApiError as exc:
            wrong = exc.status_code
        expect(wrong == 401, wrong)
        return {'not_found': missing, 'authentication': wrong}

    def local_tts():
        if not local.get('local-voice'):
            raise Untried(f'no local speech fixture (set {H.FIXTURE_ENV})')
        named = b''.join(c.text_to_speech.convert('default', text='Hello.', model_id='local-voice'))
        expect(named[:4] == b'RIFF', named[:16])
        bare = b''.join(c.text_to_speech.convert('default', text='Hello.'))
        expect(bare[:4] == b'RIFF', bare[:16])
        streamed = b''.join(c.text_to_speech.stream('default', text='Hello.', model_id='local-voice'))
        expect(streamed[:4] == b'RIFF', streamed[:16])
        return {'named_bytes': len(named), 'no_model_id_bytes': len(bare), 'stream_bytes': len(streamed)}

    def local_stt():
        if not local.get('local-asr'):
            raise Untried(f'no local transcription fixture (set {H.FIXTURE_ENV})')
        done = c.speech_to_text.convert(model_id='local-asr', file=('a.wav', H.wav_tone(1.0), 'audio/wav'))
        expect(isinstance(done.text, str), done)
        return {'text': done.text[:40]}

    return [('elevenlabs.auth_header', auth_header), ('elevenlabs.tts', tts), ('elevenlabs.tts_stream', tts_stream),
            ('elevenlabs.stt', stt), ('elevenlabs.translated_openai', translated), ('elevenlabs.voices', voices),
            ('elevenlabs.errors', errors), ('local.elevenlabs.tts', local_tts), ('local.elevenlabs.stt', local_stt)]


def refusal_cases(spec: dict) -> list:
    """402 budget_exceeded and 508 loop_detected (x-should-retry: false) through each SDK with its default retry policy.
    The server answers each before any upstream; the count of requests each SDK sends is recorded. An SDK that retries
    anyway is reported as a finding in the case detail (the refusal costs no upstream call), not as a lane failure."""
    import httpx
    url, tokens, hop = spec['url'], spec['tokens'], spec['hop']
    messages = [{'role': 'user', 'content': 'Say hello.'}]
    out = []

    def counted(seen):
        return {'request': [lambda request: seen.append(request.url.path)]}

    try:
        import anthropic
    except ImportError as exc:
        out.append(('refusals.anthropic', _untried(f'anthropic is not importable here ({exc})')))
    else:
        def anthropic_refusals():
            result = {}
            for label, token, headers, status in (('budget', tokens['budget'], {}, 402),
                                                  ('loop', tokens['sdk'], {'X-Titanos-Hop': hop}, 508)):
                seen: list = []
                client = anthropic.Anthropic(base_url=url, api_key=token, default_headers=headers,
                                             http_client=anthropic.DefaultHttpxClient(event_hooks=counted(seen)))
                try:
                    client.messages.create(model='claude-sonnet-4-5@anthropic', max_tokens=16, messages=messages)
                    raise AssertionError(f'{label}: answered')
                except anthropic.APIStatusError as exc:
                    expect(exc.status_code == status, (label, exc.status_code, exc.body))
                    kind = ((exc.body or {}).get('error') or {}).get('type')
                expect(len(seen) == 1, f'{label}: the anthropic SDK sent {len(seen)} requests')
                result[label] = {'status': status, 'type': kind, 'requests_sent': len(seen)}
            return result
        out.append(('refusals.anthropic', anthropic_refusals))

    try:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types
    except ImportError as exc:
        out.append(('refusals.gemini', _untried(f'google-genai is not importable here ({exc})')))
    else:
        def gemini_refusals():
            result = {}
            for label, token, headers, status in (('budget', tokens['budget'], {}, 402),
                                                  ('loop', tokens['sdk'], {'X-Titanos-Hop': hop}, 508)):
                seen: list = []
                client = genai.Client(api_key=token, http_options=types.HttpOptions(
                    base_url=url, api_version='v1beta', headers=headers,
                    httpx_client=httpx.Client(event_hooks=counted(seen), timeout=60)))
                try:
                    client.models.generate_content(model='gemini-2.0-flash@gemini', contents='hi')
                    raise AssertionError(f'{label}: answered')
                except genai_errors.APIError as exc:
                    expect(exc.code == status, (label, exc.code, exc.status))
                    kind = exc.status
                result[label] = {'status': status, 'rpc_status': kind, 'requests_sent': len(seen)}
                if len(seen) != 1:
                    result[label]['finding'] = 'google-genai retried a refusal marked x-should-retry: false'
            return result
        out.append(('refusals.gemini', gemini_refusals))

    try:
        from elevenlabs import ElevenLabs
        from elevenlabs.core.api_error import ApiError
    except ImportError as exc:
        out.append(('refusals.elevenlabs', _untried(f'elevenlabs is not importable here ({exc})')))
    else:
        def elevenlabs_refusals():
            result = {}
            for label, token, headers, status in (('budget', tokens['budget'], {}, 402),
                                                  ('loop', tokens['sdk'], {'X-Titanos-Hop': hop}, 508)):
                seen: list = []
                client = ElevenLabs(api_key=token, base_url=url, headers=headers,
                                    httpx_client=httpx.Client(event_hooks=counted(seen), timeout=60))
                try:
                    b''.join(client.text_to_speech.convert(VOICE, text='Hello.',
                                                           model_id='eleven_multilingual_v2@elevenlabs'))
                    raise AssertionError(f'{label}: answered')
                except ApiError as exc:
                    expect(exc.status_code == status, (label, exc.status_code, exc.body))
                result[label] = {'status': status, 'requests_sent': len(seen)}
                if len(seen) != 1:
                    result[label]['finding'] = ('the elevenlabs SDK retries every 5xx and ignores x-should-retry: '
                                                'false (fern http_client._should_retry)')
            return result
        out.append(('refusals.elevenlabs', elevenlabs_refusals))
    return out


def _untried(message: str):
    def case():
        raise Untried(message)
    return case


def vendors_client_main(phase: str, spec: dict) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    sdk = {}
    for name in ('anthropic', 'google-genai', 'elevenlabs', 'httpx'):
        try:
            sdk[name] = version(name)
        except PackageNotFoundError:
            sdk[name] = None
    if phase == 'refusals':
        return {'sdk': sdk, 'cases': H.run_cases(refusal_cases(spec))}
    groups = {'anthropic': anthropic_cases, 'gemini': gemini_cases, 'elevenlabs': elevenlabs_cases}
    cases = []
    for name, build in groups.items():
        if phase in ('main', name):
            try:
                cases += build(spec)
            except Exception as exc:  # a client that cannot even be constructed fails its group
                cases.append((name + '.client', _raise(exc)))
    return {'sdk': sdk, 'cases': H.run_cases(cases)}


def _raise(exc):
    def case():
        raise exc
    return case


# =====================================================================================================================
# server side

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    strict = '--strict' in argv
    python, missing = H.find_python(EXT_ENV, tuple(MODULES.values()))
    if len(missing) == len(MODULES):
        cases = [{'id': 'vendor.sdks', 'state': 'untried',
                  'detail': f'none of {sorted(MODULES.values())} is importable by {" / ".join(EXT_ENV)} or this '
                            'interpreter'}]
        H.report_matrix('vendor-sdks', cases)
        print(json.dumps({'lane': 'vendor-sdks', 'result': 'untried', 'counts': {'untried': 1}, 'cases': cases}))
        return 1 if strict else 0
    stack = H.Stack()
    try:
        with stack:
            results = [H.run_client(stack, python, Path(__file__), 'main', stack.spec())]
            expected = {uid: surfaces for uid, surfaces in EXPECTED_SURFACES.items()
                        if uid != 'openai' and MODULES[uid] not in missing}
            present = {uid for uid, module in MODULES.items() if module not in missing}
            expected['openai'] = ((['openai.chat'] if present & {'anthropic', 'gemini'} else [])
                                  + (['openai.audio.speech', 'openai.audio.transcriptions'] if 'elevenlabs' in present
                                     else []))
            checks = [stack.surface_check(expected)]
            before = stack.hit_total()
            results.append(H.run_client(stack, python, Path(__file__), 'refusals', stack.spec()))
            added = stack.hit_total() - before
            checks.append({'id': 'refusals.zero_upstream_hits', 'state': 'pass' if added == 0 else 'fail',
                           'detail': f'{added} mock hits during the 402 and 508 refusals'})
        sdk = {}
        for result in results:
            sdk.update(result.get('sdk') or {})
        return H.summarize('vendor-sdks', stack, results, checks, strict, sdk, python)
    finally:
        stack.cleanup()


def client_entry(argv) -> int:
    phase = argv[argv.index('--client') + 1]
    spec = json.loads(sys.stdin.read())
    print(json.dumps(vendors_client_main(phase, spec)))
    return 0


if __name__ == '__main__':
    if '--client' in sys.argv:
        sys.exit(client_entry(sys.argv))
    sys.exit(main())
