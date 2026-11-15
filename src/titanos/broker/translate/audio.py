"""Audio mapping between OpenAI /audio/speech and /audio/transcriptions and ElevenLabs text-to-speech and speech-to-text.

Speech (OpenAI -> ElevenLabs): voice through voice_maps.openai_to_elevenlabs; an unmapped voice that is not one of
OpenAI's built-in voice names and looks like an ElevenLabs voice id (8-64 letters and digits) is sent as that voice id
(note voice:elevenlabs_id), so a client with only an ElevenLabs key can name its voices; anything else 400. Formats mp3 ->
mp3_44100_128, opus -> opus_48000_128, pcm -> pcm_24000, wav -> pcm_24000 wrapped to WAV here, aac and flac -> 400;
speed -> voice_settings.speed within ElevenLabs' 0.7-1.2 (otherwise 400); instructions is dropped with a note.
Speech (ElevenLabs -> OpenAI): voice_id through voice_maps.elevenlabs_to_openai; output_format mp3_* -> mp3,
pcm_24000 -> pcm, opus_* -> opus, anything else 400 (OpenAI pcm is 24 kHz 16-bit mono only).
Transcription: language <-> language_code; ElevenLabs answers text, language_code and words[]; OpenAI verbose_json
segments are synthesized from the words (split at sentence ends, speaker changes and long pauses); srt and vtt are
rendered from those segments.
PCM -> WAV: a 44-byte RIFF/WAVE header for 16-bit little-endian mono PCM; a streamed WAV uses 0xFFFFFFFF for the RIFF and
data sizes because the length is unknown when the header is sent.
"""
from __future__ import annotations

import re
import struct

from .common import closes_source, refuse

OPENAI_TO_XI_FORMAT = {'mp3': 'mp3_44100_128', 'opus': 'opus_48000_128', 'pcm': 'pcm_24000', 'wav': 'pcm_24000+wav',
                       'aac': None, 'flac': None}
SPEED_RANGE = (0.7, 1.2)
# OpenAI's built-in /audio/speech voice names: never taken for a raw ElevenLabs voice id
OPENAI_VOICES = frozenset({'alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer', 'verse',
                           'marin', 'cedar'})
_XI_VOICE_ID = re.compile(r'[A-Za-z0-9]{8,64}')
PCM_RATE = 24000
STREAMING_SIZE = 0xFFFFFFFF
TRANSCRIPTION_FORMATS = ('json', 'text', 'verbose_json', 'srt', 'vtt')
_PAUSE = 1.0
# ISO 639-3 (ElevenLabs language_code) <-> the English language names OpenAI's verbose_json reports
LANGUAGE_NAMES = {'eng': 'english', 'spa': 'spanish', 'fra': 'french', 'deu': 'german', 'ita': 'italian',
                  'por': 'portuguese', 'nld': 'dutch', 'jpn': 'japanese', 'kor': 'korean', 'zho': 'chinese',
                  'cmn': 'chinese', 'rus': 'russian', 'hin': 'hindi', 'ara': 'arabic', 'pol': 'polish', 'tur': 'turkish',
                  'swe': 'swedish', 'ukr': 'ukrainian'}
ISO1_TO_3 = {'en': 'eng', 'es': 'spa', 'fr': 'fra', 'de': 'deu', 'it': 'ita', 'pt': 'por', 'nl': 'nld', 'ja': 'jpn',
             'ko': 'kor', 'zh': 'zho', 'ru': 'rus', 'hi': 'hin', 'ar': 'ara', 'pl': 'pol', 'tr': 'tur', 'sv': 'swe',
             'uk': 'ukr'}
ISO3_TO_1 = {v: k for k, v in ISO1_TO_3.items()} | {'cmn': 'zh'}


def _note(notes, text):
    if notes is not None and text not in notes:
        notes.append(text)


def _voice_map(voice_map, direction: str) -> dict:
    if voice_map is None:
        return {}
    if isinstance(voice_map, dict):
        return voice_map
    return dict(getattr(voice_map, direction, None) or {})


# ---------------------------------------------------------------------------------------------------------------------
# speech

def speech_openai_to_elevenlabs(body: dict, voice_map, *, notes: list | None = None) -> tuple[str, dict, dict, bool]:
    """(voice_id, JSON body, query, wrap_wav) for ElevenLabs text-to-speech from an OpenAI /audio/speech body.
    voice_map is voice_maps.openai_to_elevenlabs (a dict, or the VoiceMaps object)."""
    if not isinstance(body, dict) or not isinstance(body.get('input'), str) or not body['input']:
        raise refuse('input is required', 'invalid_request')
    voices = _voice_map(voice_map, 'openai_to_elevenlabs')
    voice = body.get('voice')
    if isinstance(voice, dict):
        voice = voice.get('id')
    if (isinstance(voice, str) and voice not in voices and voice.lower() not in OPENAI_VOICES
            and _XI_VOICE_ID.fullmatch(voice)):
        voices = dict(voices, **{voice: voice})  # a raw ElevenLabs voice id
        _note(notes, 'voice:elevenlabs_id')
    if not isinstance(voice, str) or voice not in voices:
        raise refuse(f'Voice {voice!r} has no ElevenLabs mapping (voice_maps.openai_to_elevenlabs)')
    fmt = body.get('response_format') or 'mp3'
    target = OPENAI_TO_XI_FORMAT.get(fmt, None) if fmt in OPENAI_TO_XI_FORMAT else None
    if target is None:
        raise refuse(f'response_format {fmt!r} is not available from ElevenLabs')
    wrap = target.endswith('+wav')
    output_format = target.split('+', 1)[0]
    if body.get('stream_format') not in (None, 'audio'):
        raise refuse('stream_format sse is not available from ElevenLabs')
    out: dict = {'text': body['input']}
    if body.get('model'):
        out['model_id'] = body['model']
    speed = body.get('speed')
    if speed is not None:
        if not isinstance(speed, (int, float)) or not SPEED_RANGE[0] <= float(speed) <= SPEED_RANGE[1]:
            raise refuse(f'speed must be within {SPEED_RANGE[0]}-{SPEED_RANGE[1]} for ElevenLabs')
        out['voice_settings'] = {'speed': float(speed)}
    if body.get('instructions'):
        _note(notes, 'dropped:instructions')
    for name in body:
        if name not in ('model', 'input', 'voice', 'response_format', 'speed', 'instructions', 'stream_format',
                        'worker', 'location', 'timeout'):
            _note(notes, f'dropped:{name}')
    return voices[voice], out, {'output_format': output_format}, wrap


def speech_elevenlabs_to_openai(body: dict, query: dict, voice_map, *, voice_id: str,
                                notes: list | None = None) -> dict:
    """An OpenAI /audio/speech body from an ElevenLabs text-to-speech request (voice_id from the path, output_format from
    the query). voice_map is voice_maps.elevenlabs_to_openai (a dict, or the VoiceMaps object)."""
    if not isinstance(body, dict) or not isinstance(body.get('text'), str) or not body['text']:
        raise refuse('text is required', 'invalid_request')
    voices = _voice_map(voice_map, 'elevenlabs_to_openai')
    if voice_id not in voices:
        raise refuse(f'Voice {voice_id!r} has no OpenAI mapping (voice_maps.elevenlabs_to_openai)')
    output_format = (query or {}).get('output_format') or 'mp3_44100_128'
    if output_format.startswith('mp3_'):
        fmt = 'mp3'
    elif output_format == 'pcm_24000':
        fmt = 'pcm'
    elif output_format.startswith('opus_'):
        fmt = 'opus'
    else:
        raise refuse(f'output_format {output_format!r} is not available from this target')
    out = {'model': body.get('model_id'), 'input': body['text'], 'voice': voices[voice_id], 'response_format': fmt}
    speed = (body.get('voice_settings') or {}).get('speed')
    if speed is not None:
        out['speed'] = speed
    settings = {k for k in (body.get('voice_settings') or {}) if k != 'speed'}
    if settings:
        _note(notes, 'dropped:voice_settings')
    for name in body:
        if name not in ('text', 'model_id', 'voice_settings', 'worker', 'location', 'timeout'):
            _note(notes, f'dropped:{name}')
    return out


def wav_header(rate: int, data_length: int | None, *, channels: int = 1, bits: int = 16) -> bytes:
    """A 44-byte RIFF/WAVE header; data_length None writes the streaming sizes (0xFFFFFFFF)."""
    block = channels * bits // 8
    if data_length is None:
        riff, data = STREAMING_SIZE, STREAMING_SIZE
    else:
        riff, data = 36 + data_length, data_length
    return (b'RIFF' + struct.pack('<I', riff) + b'WAVE' + b'fmt ' + struct.pack('<IHHIIHH', 16, 1, channels, rate,
                                                                                 rate * block, block, bits)
            + b'data' + struct.pack('<I', data))


def pcm_to_wav(pcm: bytes, rate: int, *, streaming: bool = False) -> bytes:
    """16-bit mono PCM wrapped in a WAV header (streaming=True: the 0xFFFFFFFF size form, for the first chunk)."""
    return wav_header(rate, None if streaming else len(pcm)) + pcm


@closes_source
async def wrap_wav_stream(chunks, rate: int = PCM_RATE):
    """A streamed WAV: the streaming header first, then the PCM chunks unchanged (the source is closed with it)."""
    yield wav_header(rate, None)
    async for chunk in chunks:
        if chunk:
            yield chunk


def pcm_rate(output_format: str) -> int:
    match = re.fullmatch(r'pcm_(\d+)', output_format or '')
    return int(match.group(1)) if match else PCM_RATE


# ---------------------------------------------------------------------------------------------------------------------
# transcription

def _field(form, name):
    for key, value in form:
        if key == name:
            return value
    return None


def stt_openai_to_elevenlabs(form: list, *, notes: list | None = None) -> list:
    """ElevenLabs speech-to-text form fields from an OpenAI transcription form. form is a list of (name, value) pairs;
    file values are (filename, bytes, content_type) tuples. The response_format stays with the caller."""
    file = _field(form, 'file')
    if file is None:
        raise refuse('file is required', 'invalid_request')
    model = _field(form, 'model')
    if not model:
        raise refuse('model is required', 'invalid_request')
    out = [('model_id', model), ('file', file)]
    language = _field(form, 'language')
    if language:
        out.append(('language_code', ISO1_TO_3.get(str(language), str(language))))
    granularities = [v for k, v in form if k in ('timestamp_granularities[]', 'timestamp_granularities')]
    out.append(('timestamps_granularity', 'word'))
    for name, _ in form:
        if name in ('prompt', 'temperature', 'chunking_strategy', 'include[]', 'stream'):
            _note(notes, f'dropped:{name}')
        elif name not in ('file', 'model', 'language', 'response_format', 'timestamp_granularities[]',
                          'timestamp_granularities', 'worker', 'location', 'timeout'):
            _note(notes, f'dropped:{name}')
    if granularities and set(granularities) - {'word', 'segment'}:
        raise refuse('timestamp_granularities must be word or segment', 'invalid_request')
    return out


def stt_elevenlabs_form_to_openai(form: list, *, notes: list | None = None) -> tuple[list, str]:
    """(OpenAI transcription form fields, response_format) from an ElevenLabs speech-to-text form; verbose_json with word
    timestamps is requested so the ElevenLabs words[] can be rebuilt. Only the word granularity is asked for: the answer
    is rebuilt from words alone, and Titanos' local transcription route accepts one granularity per request."""
    file = _field(form, 'file')
    if file is None:
        if _field(form, 'cloud_storage_url'):
            raise refuse('cloud_storage_url cannot be translated; upload the file')
        raise refuse('file is required', 'invalid_request')
    model = _field(form, 'model_id')
    if not model:
        raise refuse('model_id is required', 'invalid_request')
    out = [('model', model), ('file', file), ('response_format', 'verbose_json'), ('timestamp_granularities[]', 'word')]
    language = _field(form, 'language_code')
    if language:
        out.append(('language', ISO3_TO_1.get(str(language), str(language))))
    for name, _ in form:
        if name not in ('file', 'model_id', 'language_code', 'worker', 'location', 'timeout'):
            _note(notes, f'dropped:{name}')
    return out, 'verbose_json'


def _segments(words: list) -> list:
    segments, current = [], []

    def flush():
        if not current:
            return
        text = ''.join(w.get('text', '') for w in current).strip()
        spoken = [w for w in current if w.get('type', 'word') == 'word']
        logprobs = [w['logprob'] for w in spoken if isinstance(w.get('logprob'), (int, float))]
        segments.append({'id': len(segments), 'seek': 0, 'start': float(current[0].get('start') or 0.0),
                         'end': float(current[-1].get('end') or 0.0), 'text': ' ' + text if text else '',
                         'tokens': [], 'temperature': 0.0,
                         'avg_logprob': (sum(logprobs) / len(logprobs)) if logprobs else 0.0,
                         'compression_ratio': 0.0, 'no_speech_prob': 0.0})
        current.clear()

    previous = None
    for word in words:
        if word.get('type') == 'audio_event':
            continue
        if previous is not None and word.get('type', 'word') == 'word' and (
                word.get('speaker_id') != previous.get('speaker_id')
                or float(word.get('start') or 0) - float(previous.get('end') or 0) > _PAUSE):
            flush()
        current.append(word)
        if word.get('type', 'word') == 'word':
            previous = word
            if re.search(r'[.!?]["\')\]]*$', word.get('text', '')):
                flush()
    flush()
    return segments


def _clock(seconds: float, separator: str) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}'


def stt_elevenlabs_to_openai(body: dict, response_format: str = 'json') -> dict | str:
    """An OpenAI transcription answer (json, text, verbose_json, srt or vtt) from an ElevenLabs speech-to-text answer."""
    if not isinstance(body, dict) or not isinstance(body.get('text'), str):
        from .common import bad_upstream
        raise bad_upstream('The upstream speech-to-text answer has no text')
    if response_format not in TRANSCRIPTION_FORMATS:
        raise refuse(f'response_format {response_format!r} is not supported', 'invalid_request')
    text = body['text']
    if response_format == 'text':
        return text
    if response_format == 'json':
        return {'text': text}
    words = [w for w in body.get('words') or [] if isinstance(w, dict)]
    segments = _segments(words)
    if response_format == 'srt':
        return ''.join(f'{i + 1}\n{_clock(s["start"], ",")} --> {_clock(s["end"], ",")}\n{s["text"].strip()}\n\n'
                       for i, s in enumerate(segments))
    if response_format == 'vtt':
        return 'WEBVTT\n\n' + ''.join(f'{_clock(s["start"], ".")} --> {_clock(s["end"], ".")}\n{s["text"].strip()}\n\n'
                                      for s in segments)
    code = body.get('language_code') or ''
    duration = max((float(w.get('end') or 0.0) for w in words), default=0.0)
    return {'task': 'transcribe', 'language': LANGUAGE_NAMES.get(code, code), 'duration': duration, 'text': text,
            'segments': segments,
            'words': [{'word': w.get('text', '').strip(), 'start': float(w.get('start') or 0.0),
                       'end': float(w.get('end') or 0.0)} for w in words if w.get('type', 'word') == 'word']}


def stt_openai_to_elevenlabs_body(body) -> dict:
    """An ElevenLabs speech-to-text answer from an OpenAI verbose_json (or json / text) transcription answer."""
    if isinstance(body, str):
        return {'language_code': None, 'language_probability': None, 'text': body, 'words': []}
    if not isinstance(body, dict) or not isinstance(body.get('text'), str):
        from .common import bad_upstream
        raise bad_upstream('The upstream transcription answer has no text')
    language = body.get('language') or ''
    names = {v: k for k, v in LANGUAGE_NAMES.items() if k != 'cmn'}
    code = names.get(language, ISO1_TO_3.get(language, language or None))
    words = []
    for index, word in enumerate(body.get('words') or []):
        if index:
            previous = body['words'][index - 1]
            words.append({'text': ' ', 'start': previous.get('end', 0.0), 'end': word.get('start', 0.0),
                          'type': 'spacing'})
        words.append({'text': word.get('word', ''), 'start': word.get('start', 0.0), 'end': word.get('end', 0.0),
                      'type': 'word'})
    return {'language_code': code, 'language_probability': None, 'text': body['text'], 'words': words}


__all__ = ['ISO1_TO_3', 'ISO3_TO_1', 'LANGUAGE_NAMES', 'OPENAI_TO_XI_FORMAT', 'PCM_RATE', 'SPEED_RANGE', 'STREAMING_SIZE',
           'TRANSCRIPTION_FORMATS', 'pcm_rate', 'pcm_to_wav', 'speech_elevenlabs_to_openai', 'speech_openai_to_elevenlabs',
           'stt_elevenlabs_form_to_openai', 'stt_elevenlabs_to_openai', 'stt_openai_to_elevenlabs',
           'stt_openai_to_elevenlabs_body', 'wav_header', 'wrap_wav_stream']
