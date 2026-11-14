"""ASR fidelity: preserve plain text and disclose only backend-provided alignment."""
import math

from .engine import EngineError
from .media import read_audio
from .settings import current_settings
from .text_inputs import check_cancel

# Whisper accepts a language as a code, a name or a token; these all mean English for an English-only checkpoint.
ENGLISH_HINTS = {'en', 'eng', 'english', '<|en|>', 'en-us', 'en-gb', 'en-au', 'en-ca', 'en-in', 'en-ie', 'en-nz', 'en-za'}


def english_hint(value):
    return str(value).strip().lower().replace('_', '-') in ENGLISH_HINTS


def english_only(pipeline):
    """True for a Whisper checkpoint whose generation config declares is_multilingual=False (for example whisper-tiny.en)."""
    generation = getattr(getattr(pipeline, 'model', None), 'generation_config', None)
    return getattr(generation, 'is_multilingual', None) is False


def transcribe(pipeline, spec, data, cancel=None):
    settings = current_settings()
    verbose = data.get('response_format', 'json') == 'verbose_json'
    granularity = data.get('timestamp_granularities') or []
    if len(granularity) > 1 or any(x not in {'segment', 'word'} for x in granularity):
        raise ValueError('Select one timestamp granularity: segment or word')
    if granularity and not verbose:
        raise ValueError('timestamp_granularities requires response_format=verbose_json')
    kind = getattr(pipeline, 'type', None)
    supported = ['segment', 'word'] if kind == 'seq2seq_whisper' else ['word'] if kind in {'ctc', 'ctc_with_lm'} else []
    if data.get('translation') and granularity:
        raise ValueError('Audio translation timestamps are not supported; transcription alignment cannot describe translated text')
    selected = granularity[0] if granularity else supported[0] if verbose and supported and not data.get('translation') else None
    if selected and selected not in supported:
        raise ValueError('This ASR architecture does not support the requested timestamps')
    kw = dict(spec.defaults)
    from .audio_gating import GATE_FIELDS
    for key in GATE_FIELDS:
        kw.pop(key, None)
    kw.pop('timestamp_granularities', None)
    kw.pop('response_format', None)
    default_prompt=kw.pop('prompt',None)
    prompt=data.get('prompt') if data.get('prompt') is not None else default_prompt
    default_limit = kw.pop('max_tokens', None)
    token_limit = data.get('max_tokens') if data.get('max_tokens') is not None else default_limit
    if token_limit is not None:
        if isinstance(token_limit, bool) or not isinstance(token_limit, int) or not 1 <= token_limit <= settings.max_tokens:
            raise ValueError('max_tokens must be positive and within TITANOS_MAX_TOKENS')
        kw['generate_kwargs'] = dict(kw.get('generate_kwargs', {}), max_new_tokens=token_limit)
    if data.get('translation'):
        if getattr(pipeline.model.config, 'model_type', '') != 'whisper':
            raise ValueError('Direct audio translation requires a Whisper checkpoint; select translation_model for a local ASR + NMT pipeline')
        if getattr(pipeline.model.generation_config, 'is_multilingual', None) is False:
            raise ValueError('English-only Whisper checkpoints cannot translate foreign speech; choose a multilingual Whisper checkpoint')
        kw['generate_kwargs'] = dict(kw.get('generate_kwargs', {}), task='translate')
    if data.get('language'):
        if english_only(pipeline):
            # The checkpoint has no language tokens, so the library rejects any language argument. An English hint
            # is already satisfied and is not forwarded; any other language cannot be honored by this checkpoint.
            if not english_hint(data['language']):
                raise ValueError(f"This English-only Whisper checkpoint transcribes English only; omit language or choose a multilingual checkpoint (requested {data['language']!r})")
        else:
            kw['generate_kwargs'] = dict(kw.get('generate_kwargs', {}), language=data['language'])
    if verbose:
        kw.pop('return_timestamps', None)
        kw.pop('return_language', None)
        if selected:
            kw['return_timestamps'] = True if selected == 'segment' else 'word'
        if kind == 'seq2seq_whisper':
            kw['return_language'] = True
    conditioning=None
    if prompt is not None:
        from .whisper_prompt import prepare
        generate=dict(kw.get('generate_kwargs',{}))
        conditioning=prepare(pipeline,spec,prompt,generate,kw.get('return_timestamps'),cancel)
        conditioning['source']='request' if data.get('prompt') is not None else 'model_default'
        kw['generate_kwargs']=generate
    if cancel is not None and kind in {'seq2seq', 'seq2seq_whisper'}:
        from transformers import StoppingCriteria, StoppingCriteriaList
        class Cancelled(StoppingCriteria):
            def __call__(self, *args, **kwargs):
                return cancel.is_set()
        generate = dict(kw.get('generate_kwargs', {}))
        generate['stopping_criteria'] = StoppingCriteriaList([*generate.get('stopping_criteria', []), Cancelled()])
        kw['generate_kwargs'] = generate
    rate = pipeline.feature_extractor.sampling_rate
    wave = read_audio(data['audio'], rate, cancel)
    check_cancel(cancel)
    if conditioning is not None:
        from .whisper_prompt import check_budget
        check_budget(conditioning,pipeline,kw['generate_kwargs'],len(wave)/rate)
        check_cancel(cancel)
    result = pipeline({'raw': wave, 'sampling_rate': rate}, **kw)
    check_cancel(cancel)
    if not isinstance(result, dict) or not isinstance(result.get('text'), str):
        raise EngineError('ASR returned no text', 502, 'invalid_backend_response')
    if not verbose:
        return result['text']
    chunks = result.get('chunks')
    if chunks is not None and (not isinstance(chunks, list) or any(not isinstance(c, dict) for c in chunks)):
        raise EngineError('ASR returned malformed chunks', 502, 'invalid_backend_response')
    aligned = None
    if selected and chunks is not None:
        aligned = []
        for i, chunk in enumerate(chunks):
            pair = chunk.get('timestamp')
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not isinstance(chunk.get('text'), str):
                raise EngineError('ASR returned malformed alignment', 502, 'invalid_backend_response')
            start, end = pair
            if any(x is not None and (isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0) for x in pair) or (start is not None and end is not None and end < start):
                raise EngineError('ASR returned invalid timestamps', 502, 'invalid_backend_response')
            aligned.append({'id': i, 'start': start, 'end': end, 'text' if selected == 'segment' else 'word': chunk['text']})
    languages = {c['language'] for c in chunks or [] if isinstance(c.get('language'), str)}
    language = next(iter(languages)) if len(languages) == 1 else None
    return {'task': 'translate' if data.get('translation') else 'transcribe', 'text': result['text'],
            'duration': len(wave) / rate, 'sample_rate': rate, 'decoded_samples': len(wave),
            'language': language, 'language_source': 'backend' if language else 'unavailable',
            'requested_language': data.get('language'),
            'segments': aligned if selected == 'segment' else None, 'words': aligned if selected == 'word' else None,
            'timestamps': {'granularity': selected, 'source': 'backend_alignment' if aligned is not None else 'unavailable',
                           'supported': supported if not data.get('translation') else []},
            'quality': {'confidence': None, 'avg_logprob': None, 'no_speech_prob': None,
                        'status': 'unavailable', 'reason': 'The pipeline does not expose these quality measurements'},
            'backend': spec.backend,**({'prompt_conditioning':conditioning} if conditioning is not None else {})}
