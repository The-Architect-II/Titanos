"""ElevenLabs completeness through the official elevenlabs Python SDK, unmodified (only base_url and api_key change),
against a REAL `titanos serve` subprocess routed to the extended WP4 ElevenLabsMock (sdk tier).

The harness (mocks, temporary TITANOS_HOME, credentials bound to the mocks' exact origins, the serve subprocess under the
netguard sitecustomize, minted tk_ keys) is tests/check_broker_sdk.py's Stack, without local fixture models. This file
mints one more key with the 'native' scope. The SDK client is this file with --client, run by the interpreter named by
TITANOS_SDK_PYTHON_EXT (or TITANOS_SDK_PYTHON_EXTRA; default: this interpreter, as in CI after
`python -m pip install -r tests/requirements-sdk.txt`). Every call authenticates with the SDK's own xi-api-key header.

Cases (each pass, fail or untried; without the elevenlabs SDK every case is untried, never a pass):
- the SDK's bare model ids: text-to-speech, its stream and speech-to-text with eleven_multilingual_v2,
  eleven_flash_v2_5 and scribe_v1 (no '@upstream' suffix), text-to-speech without model_id (the provider default),
  multichannel speech-to-text with ElevenLabs-only fields, and a typed ApiError 404 for an unknown model;
- through the native passthrough on the SDK's own paths: models.list (the provider list, because the SDK sends xi-api-key
  and no Authorization), text-to-speech with timestamps and its stream, dialogue and dialogue with timestamps, sound
  effects, voice changer, audio isolation, forced alignment, music and its composition plan, voice search, voice
  settings and voice design.
Afterwards the mock must have been reached on each of those endpoints, and the shared closing checks hold (netguard on
both sides with zero non-loopback attempts, the exact upstream credential on every mock hit, no secret in any output,
clean serve shutdown). Exit 1 on any fail, or with --strict on any untried.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_broker_sdk as H  # noqa: E402  (the shared harness; imports only the standard library at module level)

EXT_ENV = ('TITANOS_SDK_PYTHON_EXT', 'TITANOS_SDK_PYTHON_EXTRA')
VOICE = H.ELEVENLABS_VOICE
EXPECTED = ['elevenlabs.tts', 'elevenlabs.tts_stream', 'elevenlabs.stt', 'elevenlabs.models', 'native:tts',
            'native:dialogue', 'native:sound_effects', 'native:voice_changer', 'native:audio_isolation',
            'native:forced_alignment', 'native:music', 'native:voices', 'native:voice_design']
expect, Untried = H.expect, H.Untried


# =====================================================================================================================
# client side (the vendor SDK interpreter)

def client_cases(spec: dict) -> list:
    try:
        from elevenlabs import ElevenLabs
        from elevenlabs.core.api_error import ApiError
    except ImportError as exc:
        message = f'elevenlabs is not importable here ({exc})'

        def untried():
            raise Untried(message)
        return [('elevenlabs.sdk', untried)]
    url = spec['url']
    c = ElevenLabs(api_key=spec['tokens']['sdk'], base_url=url, timeout=120)
    n = ElevenLabs(api_key=spec['tokens']['native'], base_url=url, timeout=120)
    wav = H.wav_tone()

    def audio(chunks) -> int:
        size = len(b''.join(chunks))
        expect(size > 0, 'empty audio')
        return size

    def bare_model_ids():
        return {'tts': audio(c.text_to_speech.convert(VOICE, text='Hello there.', model_id='eleven_multilingual_v2',
                                                      output_format='mp3_44100_128',
                                                      voice_settings={'stability': 0.5, 'similarity_boost': 0.7})),
                'stream': audio(c.text_to_speech.stream(VOICE, text='Hello.', model_id='eleven_flash_v2_5')),
                'no_model': audio(c.text_to_speech.convert(VOICE, text='Default model.'))}

    def stt_bare_and_multichannel():
        single = c.speech_to_text.convert(model_id='scribe_v1', file=('a.wav', wav, 'audio/wav'), diarize=True,
                                          tag_audio_events=False)
        expect(getattr(single, 'text', None), single)
        multi = c.speech_to_text.convert(model_id='scribe_v1', file=('a.wav', wav, 'audio/wav'), use_multi_channel=True)
        transcripts = getattr(multi, 'transcripts', None)
        expect(transcripts and len(transcripts) == 2, multi)
        return {'single': single.text, 'channels': len(transcripts)}

    def unknown_model():
        try:
            b''.join(c.text_to_speech.convert(VOICE, text='Hi.', model_id='eleven_no_such_model'))
        except ApiError as exc:
            expect(exc.status_code == 404, exc.status_code)
            return {'status': exc.status_code}
        raise AssertionError('no ApiError for an unknown model')

    def models_list():
        rows = n.models.list()
        ids = [row.model_id for row in rows]
        expect('eleven_multilingual_v2' in ids, ids)
        return {'models': ids}

    def timestamps():
        whole = n.text_to_speech.convert_with_timestamps(VOICE, text='Hi.')
        expect(whole.alignment.characters == ['H', 'i', '.'], whole.alignment)
        chunks = list(n.text_to_speech.stream_with_timestamps(VOICE, text='Hi.'))
        expect(len(chunks) == 2, chunks)
        return {'characters': whole.alignment.characters, 'chunks': len(chunks)}

    def dialogue():
        inputs = [{'text': 'Hi.', 'voice_id': VOICE}, {'text': 'Hello.', 'voice_id': 'mockvoice00000000002'}]
        size = audio(n.text_to_dialogue.convert(inputs=inputs, model_id='eleven_v3'))
        stamped = n.text_to_dialogue.convert_with_timestamps(inputs=inputs)
        expect(len(stamped.voice_segments) == 2, stamped)
        return {'bytes': size, 'segments': len(stamped.voice_segments)}

    def sound_effects():
        return {'bytes': audio(n.text_to_sound_effects.convert(text='rain on a roof', duration_seconds=2.0,
                                                               prompt_influence=0.5))}

    def voice_changer_and_isolation():
        changed = audio(n.speech_to_speech.convert(VOICE, audio=('in.wav', wav, 'audio/wav'),
                                                   model_id='eleven_multilingual_sts_v2'))
        isolated = audio(n.audio_isolation.convert(audio=('in.wav', wav, 'audio/wav')))
        return {'changed': changed, 'isolated': isolated}

    def forced_alignment():
        aligned = n.forced_alignment.create(file=('in.wav', wav, 'audio/wav'), text='Hi')
        expect(aligned.words[0].text == 'Hi', aligned)
        return {'words': len(aligned.words)}

    def music():
        size = audio(n.music.compose(prompt='a calm piano piece', music_length_ms=10000))
        plan = n.music.composition_plan.create(prompt='a calm piano piece', music_length_ms=10000)
        expect(plan.sections and plan.sections[0].section_name == 'Intro', plan)
        return {'bytes': size, 'sections': len(plan.sections)}

    def voices():
        found = n.voices.search(page_size=2, search='calm')
        expect(found.next_page_token == 'page-token-0002', found)
        settings = n.voices.settings.get(VOICE)
        expect(settings.similarity_boost == 0.75, settings)
        design = n.text_to_voice.design(voice_description='A calm, low narrator voice')
        expect(design.previews and design.previews[0].generated_voice_id == 'genvoice0001', design)
        return {'voices': len(found.voices), 'previews': len(design.previews)}

    return [('elevenlabs.bare_model_ids', bare_model_ids), ('elevenlabs.stt_bare_and_multichannel',
                                                             stt_bare_and_multichannel),
            ('elevenlabs.unknown_model', unknown_model), ('elevenlabs.native.models_list', models_list),
            ('elevenlabs.native.timestamps', timestamps), ('elevenlabs.native.dialogue', dialogue),
            ('elevenlabs.native.sound_effects', sound_effects),
            ('elevenlabs.native.voice_changer_and_isolation', voice_changer_and_isolation),
            ('elevenlabs.native.forced_alignment', forced_alignment), ('elevenlabs.native.music', music),
            ('elevenlabs.native.voices', voices)]


def client_main(spec: dict) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    sdk = {}
    for name in ('elevenlabs', 'httpx'):
        try:
            sdk[name] = version(name)
        except PackageNotFoundError:
            sdk[name] = None
    return {'sdk': sdk, 'cases': H.run_cases(client_cases(spec))}


# =====================================================================================================================
# server side

def mint_native(stack) -> str:
    import httpx
    with httpx.Client(base_url=stack.url, trust_env=False, timeout=30,
                      headers={'Authorization': 'Bearer ' + stack.BOOT}) as http:
        reply = http.post('/api/keys', json={'name': 'elevenlabs-sdk-native',
                                             'scopes': ['inference', 'cloud', 'native', 'models:read']})
        if reply.status_code != 201:
            raise RuntimeError(f'minting the native key failed: {reply.status_code} {reply.text[:300]}')
        token = reply.json()['secret']
    stack.tokens['native'] = token
    return token


def closing_checks(stack, results: list, client_python: str) -> list:
    """The shared closing checks of tests/check_broker_sdk.py's summarize (without its matrix report)."""
    checks = []
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
    return checks


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    strict = '--strict' in argv
    python, missing = H.find_python(EXT_ENV, ('elevenlabs',))
    if missing:
        cases = [{'id': 'elevenlabs.sdk', 'state': 'untried',
                  'detail': f'elevenlabs is not importable by {" / ".join(EXT_ENV)} or this interpreter'}]
        print(json.dumps({'lane': 'elevenlabs-sdk', 'result': 'untried', 'counts': {'untried': 1}, 'cases': cases}))
        return 1 if strict else 0
    stack = H.Stack(local=False)
    try:
        with stack:
            mint_native(stack)
            results = [H.run_client(stack, python, Path(__file__), 'main', stack.spec())]
            checks = [stack.surface_check({'elevenlabs': EXPECTED})]
        checks += closing_checks(stack, results, python)
        cases = [case for result in results for case in result.get('cases', [])] + checks
        states = {state: sum(1 for c in cases if c['state'] == state) for state in ('pass', 'fail', 'untried')}
        result = 'fail' if states['fail'] else 'untried' if states['untried'] else 'pass'
        sdk = {}
        for item in results:
            sdk.update(item.get('sdk') or {})
        print(json.dumps({'lane': 'elevenlabs-sdk', 'result': result, 'counts': states, 'sdk': sdk,
                          'cases': cases, 'scope': 'a real titanos serve subprocess, loopback mocks; no live provider'},
                         indent=1, default=str))
        return 1 if states['fail'] or (strict and states['untried']) else 0
    finally:
        stack.cleanup()


def client_entry(argv) -> int:
    spec = json.loads(sys.stdin.read())
    print(json.dumps(client_main(spec), default=str))
    return 0


if __name__ == '__main__':
    if '--client' in sys.argv:
        sys.exit(client_entry(sys.argv))
    sys.exit(main())
