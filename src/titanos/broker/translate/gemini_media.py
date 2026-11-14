"""OpenAI media surfaces translated to Gemini generateContent (typed bridges for a Gemini upstream).

- Speech (openai.audio.speech -> a Gemini TTS model): input becomes the user text, voice a prebuilt voiceName
  (generationConfig.responseModalities ['AUDIO'] + speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName); a voice
  that is not a Gemini prebuilt voice uses DEFAULT_VOICE (noted dropped:voice); instructions become the natural-language
  style prefix the TTS models read ('<instructions>: <input>', noted mapped:instructions); speed has no Gemini field
  (noted dropped:speed). The answer's inlineData is 16-bit PCM (audio/L16;codec=pcm;rate=24000) or WAV: wav and pcm are
  answered as asked; any other response_format (mp3, opus, aac, flac) is answered as WAV (noted
  dropped:response_format; Titanos does not transcode). stream_format sse is refused.
- Images (openai.images.generations and openai.images.edits -> a Gemini image model): the prompt (plus the edit's input
  images, inline) with responseModalities ['TEXT', 'IMAGE']; size maps to imageConfig.aspectRatio (the nearest
  supported ratio) and imageSize (2K / 4K for large sizes); n > 1, masks, url output and streaming are refused before
  egress; quality, style, background, moderation, output_format, output_compression, input_fidelity and user have no
  Gemini field (noted dropped:<name>). The answer is OpenAI's {created, data: [{b64_json}], usage}.
- Transcription and translation (openai.audio.transcriptions / translations -> a Gemini model with audio understanding):
  the upload travels inline with a fixed instruction (TRANSCRIBE_PROMPT / TRANSLATE_PROMPT); language and prompt are
  added to the instruction, temperature maps to generationConfig.temperature. response_format json, text and
  verbose_json (text only, segments []) are served; srt and vtt need timestamps Gemini does not return and are refused
  before egress, as is stream=true; timestamp_granularities and include are noted dropped.
Usage comes from usageMetadata (tokens); images also count Usage.images.
"""
from __future__ import annotations

import base64
import binascii
import io
import math
import time
import wave

from ..errors import BrokerError
from ..sse import gemini_usage
from ..types import Usage
from .common import bad_upstream, refuse
from .gemini import FILTER_REASONS

VOICES = ('Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir', 'Leda', 'Orus', 'Aoede', 'Callirrhoe', 'Autonoe', 'Enceladus',
          'Iapetus', 'Umbriel', 'Algieba', 'Despina', 'Erinome', 'Algenib', 'Rasalgethi', 'Laomedeia', 'Achernar',
          'Alnilam', 'Schedar', 'Gacrux', 'Pulcherrima', 'Achird', 'Zubenelgenubi', 'Vindemiatrix', 'Sadachbia',
          'Sadaltager', 'Sulafat')
DEFAULT_VOICE = 'Kore'
PCM_RATE = 24000
ASPECT_RATIOS = ('1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9')
TRANSCRIBE_PROMPT = ('Generate a verbatim transcript of the speech in this audio. Answer with the transcript text only: '
                     'no timestamps, speaker labels, notes or quotes.')
TRANSLATE_PROMPT = ('Translate the speech in this audio into English. Answer with the English translation only: no '
                    'timestamps, notes or quotes.')
_TITANOS = frozenset({'worker', 'location', 'timeout'})
_SPEECH_FIELDS = frozenset({'model', 'input', 'voice', 'response_format', 'speed', 'instructions', 'stream_format'})
_IMAGE_DROPPED = ('quality', 'style', 'background', 'moderation', 'output_format', 'output_compression',
                  'input_fidelity', 'user', 'partial_images')
_IMAGE_FIELDS = frozenset({'model', 'prompt', 'n', 'size', 'response_format', 'stream', 'image', 'image[]', 'mask',
                           *_IMAGE_DROPPED})
_STT_FIELDS = frozenset({'model', 'file', 'language', 'prompt', 'response_format', 'temperature',
                         'timestamp_granularities', 'timestamp_granularities[]', 'include', 'include[]', 'stream',
                         'chunking_strategy'})
_EXTENSION_MIME = {'wav': 'audio/wav', 'mp3': 'audio/mp3', 'mpeg': 'audio/mp3', 'mpga': 'audio/mp3', 'm4a': 'audio/aac',
                   'aac': 'audio/aac', 'ogg': 'audio/ogg', 'oga': 'audio/ogg', 'flac': 'audio/flac', 'aiff': 'audio/aiff',
                   'aif': 'audio/aiff', 'webm': 'audio/webm', 'opus': 'audio/ogg'}


def _note(notes, text):
    if notes is not None and text not in notes:
        notes.append(text)


def _content(parts: list) -> list:
    return [{'role': 'user', 'parts': parts}]


def _candidate_parts(document) -> tuple[list, str | None]:
    """(parts of the first candidate, finishReason or blockReason); a blocked prompt or a filtered answer is a 400."""
    if not isinstance(document, dict):
        raise bad_upstream('The upstream generateContent response is not an object')
    feedback = document.get('promptFeedback') or {}
    candidates = document.get('candidates') or []
    if not candidates:
        if feedback.get('blockReason'):
            raise BrokerError(f'The provider declined the request (blockReason {feedback["blockReason"]})',
                              'invalid_request', origin='upstream')
        raise bad_upstream('The upstream generateContent response has no candidates')
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    parts = (candidate.get('content') or {}).get('parts') or []
    return [p for p in parts if isinstance(p, dict)], candidate.get('finishReason')


def _inline(part: dict) -> tuple[str, bytes] | None:
    inline = part.get('inlineData') or part.get('inline_data')
    if not isinstance(inline, dict):
        return None
    mime = str(inline.get('mimeType') or inline.get('mime_type') or 'application/octet-stream')
    try:
        data = base64.b64decode(inline.get('data') or '', validate=True)
    except (binascii.Error, ValueError, TypeError):
        raise bad_upstream('The upstream inline data is not valid base64') from None
    return mime, data


def _filtered(reason: str | None, what: str):
    if reason in FILTER_REASONS or reason in ('IMAGE_PROHIBITED_CONTENT', 'IMAGE_RECITATION', 'NO_IMAGE', 'IMAGE_OTHER'):
        raise BrokerError(f'The provider returned no {what} (finishReason {reason})', 'invalid_request',
                          origin='upstream')


def usage_of(document, **extra) -> Usage:
    usage = gemini_usage((document or {}).get('usageMetadata') if isinstance(document, dict) else None) or Usage()
    if extra:
        values = {name: getattr(usage, name) for name in ('input_tokens', 'output_tokens', 'cached_input_tokens',
                                                          'reasoning_tokens', 'characters', 'audio_seconds', 'images',
                                                          'cache_write_tokens')}
        values.update(extra)
        usage = Usage(**values, estimated=usage.estimated)
    return usage


# -- speech ---------------------------------------------------------------------------------------------------------
def voice_name(voice) -> str | None:
    """The Gemini prebuilt voice for a requested voice (case-insensitive), or None."""
    if isinstance(voice, dict):
        voice = voice.get('id') or voice.get('name')
    if not isinstance(voice, str):
        return None
    lowered = voice.strip().lower()
    return next((name for name in VOICES if name.lower() == lowered), None)


def speech_request(body: dict, *, notes: list | None = None) -> tuple[dict, str]:
    """(generateContent body, output kind 'wav' | 'pcm') from an OpenAI /v1/audio/speech body."""
    if not isinstance(body, dict) or not isinstance(body.get('input'), str) or not body['input']:
        raise refuse('input is required', 'invalid_request')
    if body.get('stream_format') not in (None, 'audio'):
        raise refuse('stream_format sse is not available from Gemini text-to-speech')
    voice = voice_name(body.get('voice'))
    if voice is None:
        voice = DEFAULT_VOICE
        if body.get('voice') is not None:
            _note(notes, 'dropped:voice')
    fmt = body.get('response_format') or 'mp3'
    kind = fmt if fmt in ('wav', 'pcm') else 'wav'
    if fmt not in ('wav', 'pcm'):
        _note(notes, 'dropped:response_format')
    text = body['input']
    instructions = body.get('instructions')
    if isinstance(instructions, str) and instructions.strip():
        text = f'{instructions.strip()}: {text}'
        _note(notes, 'mapped:instructions')
    if body.get('speed') is not None:
        _note(notes, 'dropped:speed')
    for name in body:
        if name not in _SPEECH_FIELDS and name not in _TITANOS:
            _note(notes, f'dropped:{name}')
    request = {'contents': _content([{'text': text}]),
               'generationConfig': {'responseModalities': ['AUDIO'],
                                    'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': voice}}}}}
    return request, kind


def _rate(mime: str) -> int:
    for piece in mime.split(';')[1:]:
        name, _, value = piece.strip().partition('=')
        if name.lower() == 'rate' and value.strip().isdigit():
            return int(value.strip())
    return PCM_RATE


def speech_audio(document) -> tuple[bytes, int]:
    """(16-bit mono PCM, sample rate) of a TTS answer (inlineData audio/L16 or audio/wav, possibly in several parts)."""
    parts, reason = _candidate_parts(document)
    pcm, rate = bytearray(), None
    for part in parts:
        found = _inline(part)
        if found is None or not found[0].lower().startswith('audio/'):
            continue
        mime, data = found
        kind = mime.split(';', 1)[0].strip().lower()
        if kind in ('audio/wav', 'audio/x-wav', 'audio/wave'):
            try:
                with wave.open(io.BytesIO(data), 'rb') as reader:
                    if reader.getsampwidth() != 2 or reader.getnchannels() != 1:
                        raise bad_upstream('The upstream WAV is not 16-bit mono')
                    rate = rate or reader.getframerate()
                    pcm += reader.readframes(reader.getnframes())
            except (wave.Error, EOFError):
                raise bad_upstream('The upstream WAV could not be read') from None
        elif kind in ('audio/l16', 'audio/pcm'):
            rate = rate or _rate(mime)
            pcm += data
        else:
            raise bad_upstream(f'The upstream answered {kind}, not PCM or WAV')
    if not pcm:
        _filtered(reason, 'audio')
        raise bad_upstream('The upstream answered without audio')
    return bytes(pcm), rate or PCM_RATE


def pcm_seconds(pcm: bytes, rate: int) -> float:
    return round(len(pcm) / 2 / max(1, rate), 3)


# -- images ---------------------------------------------------------------------------------------------------------
def aspect_ratio(size) -> tuple[str | None, str | None]:
    """(aspectRatio, imageSize) for an OpenAI size 'WxH' ('auto' or absent: (None, None)); 400 for a malformed size."""
    if size in (None, '', 'auto'):
        return None, None
    try:
        width, height = (int(v) for v in str(size).lower().split('x', 1))
    except ValueError:
        raise refuse('size must be WIDTHxHEIGHT', 'invalid_request') from None
    if width <= 0 or height <= 0:
        raise refuse('size must be WIDTHxHEIGHT', 'invalid_request')
    target = math.log(width / height)

    def distance(ratio):
        w, h = (int(v) for v in ratio.split(':'))
        return abs(math.log(w / h) - target)
    ratio = min(ASPECT_RATIOS, key=distance)
    longest = max(width, height)
    image_size = '4K' if longest >= 3072 else '2K' if longest >= 2048 else None
    return ratio, image_size


def image_request(body: dict, *, images=(), notes: list | None = None, edit: bool = False) -> dict:
    """A generateContent body for an OpenAI images.generations (JSON) or images.edits (form fields as a dict, input
    images as (mime, bytes) pairs) request."""
    prompt = body.get('prompt')
    if not isinstance(prompt, str) or not prompt:
        raise refuse('prompt is required', 'invalid_request')
    n = body.get('n')
    if n not in (None, '', 1, '1'):
        raise refuse('n > 1 is not available from Gemini image models (one image per request)')
    if body.get('response_format') not in (None, '', 'b64_json'):
        raise refuse('Gemini image models answer base64 only; ask for response_format b64_json')
    if body.get('stream') not in (None, False, 'false', ''):
        raise refuse('Streaming image generation is not available from Gemini image models')
    if body.get('mask') is not None:
        raise refuse('Masked edits are not available from Gemini image models; describe the edit in the prompt')
    if edit and not images:
        raise refuse('image is required', 'invalid_request')
    for name in _IMAGE_DROPPED:
        if body.get(name) not in (None, '', 'auto', 'standard'):
            _note(notes, f'dropped:{name}')
    for name in body:
        if name not in _IMAGE_FIELDS and name not in _TITANOS:
            _note(notes, f'dropped:{name}')
    parts = [{'text': prompt}]
    for mime, data in images:
        parts.append({'inlineData': {'mimeType': mime, 'data': base64.b64encode(data).decode('ascii')}})
    config: dict = {'responseModalities': ['TEXT', 'IMAGE']}
    ratio, image_size = aspect_ratio(body.get('size'))
    image_config = {}
    if ratio:
        image_config['aspectRatio'] = ratio
    if image_size:
        image_config['imageSize'] = image_size
    if image_config:
        config['imageConfig'] = image_config
        _note(notes, 'mapped:size')
    return {'contents': _content(parts), 'generationConfig': config}


def image_outputs(document) -> tuple[list[tuple[str, bytes]], str]:
    """([(mime, bytes)], the answer's text) of an image-model answer; 400 when the provider filtered the image."""
    parts, reason = _candidate_parts(document)
    images, texts = [], []
    for part in parts:
        if part.get('thought'):
            continue
        found = _inline(part)
        if found is not None and found[0].lower().startswith('image/'):
            images.append(found)
        elif isinstance(part.get('text'), str):
            texts.append(part['text'])
    if not images:
        _filtered(reason, 'image')
        raise BrokerError('The image model answered without an image' + (f' (finishReason {reason})' if reason else ''),
                          'upstream_bad_response', origin='upstream')
    return images, ''.join(texts).strip()


def images_response(document, *, usage: Usage) -> dict:
    """The OpenAI images answer {created, data: [{b64_json, revised_prompt?}], output_format, usage}."""
    images, text = image_outputs(document)
    data = []
    for mime, raw in images:
        item = {'b64_json': base64.b64encode(raw).decode('ascii')}
        if text:
            item['revised_prompt'] = text
        data.append(item)
    out = {'created': int(time.time()), 'data': data,
           'output_format': images[0][0].split('/', 1)[-1].split(';', 1)[0].replace('jpg', 'jpeg')}
    tokens_in, tokens_out = usage.input_tokens or 0, usage.output_tokens or 0
    out['usage'] = {'input_tokens': tokens_in, 'output_tokens': tokens_out, 'total_tokens': tokens_in + tokens_out}
    return out


# -- transcription ----------------------------------------------------------------------------------------------------
def audio_mime(filename: str | None, content_type: str | None) -> str:
    kind = (content_type or '').split(';', 1)[0].strip().lower()
    if kind.startswith('audio/') or kind.startswith('video/'):
        return {'audio/mpeg': 'audio/mp3', 'audio/x-wav': 'audio/wav', 'audio/wave': 'audio/wav',
                'audio/x-flac': 'audio/flac', 'audio/x-m4a': 'audio/aac', 'audio/mp4': 'audio/aac'}.get(kind, kind)
    extension = (filename or '').rsplit('.', 1)[-1].lower() if '.' in (filename or '') else ''
    return _EXTENSION_MIME.get(extension, 'audio/wav')


def transcription_request(form: list, *, translate: bool = False, notes: list | None = None) -> tuple[dict, str, bytes]:
    """(generateContent body, response_format, audio bytes) from an OpenAI transcription or translation form."""
    upload = next((v for k, v in form if k == 'file' and isinstance(v, tuple)), None)
    if upload is None or not upload[1]:
        raise refuse('file is required', 'invalid_request')
    fields = {k: v for k, v in form if not isinstance(v, tuple)}
    fmt = fields.get('response_format') or 'json'
    if fmt in ('srt', 'vtt'):
        raise refuse(f'response_format {fmt} needs timestamps, which Gemini does not return; ask for json, text or '
                     'verbose_json')
    if fmt not in ('json', 'text', 'verbose_json'):
        raise refuse(f'response_format {fmt!r} is not supported', 'invalid_request')
    if str(fields.get('stream') or '').lower() in ('true', '1'):
        raise refuse('Streamed transcription is not available from Gemini')
    for name, _ in form:
        if name in ('timestamp_granularities', 'timestamp_granularities[]', 'include', 'include[]',
                    'chunking_strategy'):
            _note(notes, 'dropped:' + name.rstrip('[]'))
        elif name not in _STT_FIELDS and name not in _TITANOS:
            _note(notes, f'dropped:{name}')
    instruction = TRANSLATE_PROMPT if translate else TRANSCRIBE_PROMPT
    if fields.get('language') and not translate:
        instruction += f' The speech is in the language with code {fields["language"]}.'
    if fields.get('prompt'):
        instruction += f' Context and spellings to follow: {fields["prompt"]}'
    filename, data = upload[0], upload[1]
    content_type = upload[2] if len(upload) > 2 else None
    request = {'contents': _content([{'text': instruction},
                                     {'inlineData': {'mimeType': audio_mime(filename, content_type),
                                                     'data': base64.b64encode(data).decode('ascii')}}])}
    config: dict = {}
    if fields.get('temperature') not in (None, ''):
        try:
            config['temperature'] = float(fields['temperature'])
        except ValueError:
            raise refuse('temperature must be a number', 'invalid_request') from None
    if config:
        request['generationConfig'] = config
    return request, fmt, bytes(data)


def transcription_text(document) -> str:
    parts, reason = _candidate_parts(document)
    text = ''.join(p['text'] for p in parts if isinstance(p.get('text'), str) and not p.get('thought')).strip()
    if not text:
        _filtered(reason, 'transcript')
    return text


def transcription_response(text: str, fmt: str, *, language: str | None, duration: float | None,
                           translate: bool = False):
    """The OpenAI answer: {'text'} (json), the text itself (text) or the verbose_json object (no segments)."""
    if fmt == 'text':
        return text
    if fmt == 'verbose_json':
        return {'task': 'translate' if translate else 'transcribe',
                'language': 'english' if translate else (language or 'unknown'),
                'duration': duration if duration is not None else 0.0, 'text': text, 'segments': []}
    return {'text': text}


__all__ = ['ASPECT_RATIOS', 'DEFAULT_VOICE', 'PCM_RATE', 'TRANSCRIBE_PROMPT', 'TRANSLATE_PROMPT', 'VOICES',
           'aspect_ratio', 'audio_mime', 'image_outputs', 'image_request', 'images_response', 'pcm_seconds',
           'speech_audio', 'speech_request', 'transcription_request', 'transcription_response', 'transcription_text',
           'usage_of', 'voice_name']
