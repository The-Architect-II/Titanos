"""Local Qwen3-TTS Base voice cloning; no orchestration-service dependency.

The streaming wrapper revision below is the one used to bind the requested Vega
reference. Speech comes at its native rate with no tempo/pitch processing: whole, and
with an emit callback also chunk by chunk as it is generated. A voice with a registered
transcript (voice_texts) is cloned in context (ICL), any other from its x-vector. The
streaming knobs and request sampling follow gim_api TTS Super (docs/QWEN_TTS.md).
"""
import hashlib
from importlib import metadata
import io
import json
import math
from pathlib import Path
import secrets
import threading
import time
import wave

from .engine import EngineError
from .hardware import select_device
from .media import pcm16, write_wav
from .settings import current_settings

WRAPPER_REVISION = '13ef823ffe76aec4ac7e35d3fec28ecfc8d7427a'
LANGUAGES = {'en': 'english', 'zh': 'chinese', 'ja': 'japanese', 'ko': 'korean',
             'de': 'german', 'fr': 'french', 'ru': 'russian', 'pt': 'portuguese',
             'es': 'spanish', 'it': 'italian'}
# Request sampling as gim_api TTS Super accepts it (its STREAM_SAMPLING_KEYS): kind, lowest, highest
SAMPLING = {'temperature': (float, 0.05, 2.0), 'top_k': (int, 1, 4096), 'top_p': (float, 0.05, 1.0),
            'subtalker_temperature': (float, 0.05, 2.0), 'subtalker_top_k': (int, 1, 4096),
            'subtalker_top_p': (float, 0.05, 1.0)}
SEAM_JOINS = ('continuity', 'qwen')
_seam = threading.local()  # the seam_join of the synthesis running on this thread (install_seams)
_seam_lock = threading.Lock()


def sampling_value(name, value):
    """A request sampling value checked as gim_api TTS Super checks it: a number of the key's kind within its range."""
    kind, low, high = SAMPLING[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    if kind is int and (not math.isfinite(value) or float(value) != int(value)):
        raise ValueError(f'{name} must be a whole number')
    value = kind(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be within {low} and {high}')
    return value


def check_defaults(defaults):
    """Refuse an unknown seam_join and a ramp or overlap that is not a nonnegative whole number."""
    if defaults.get('seam_join', 'continuity') not in SEAM_JOINS:
        raise ValueError(f"Qwen seam_join must be one of {', '.join(SEAM_JOINS)}")
    for name in ('first_emit_frames', 'overlap_samples'):
        value = defaults.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'Qwen {name} must be a nonnegative whole number')


def voice_texts(spec):
    """Each voice's registered reference transcript, stripped; a voice without one is cloned from its x-vector."""
    return {voice: text.strip() for voice, text in (getattr(spec, 'voice_texts', None) or {}).items()}


def reference_audio(path, seconds=5):
    """Match the bound reference preparation: first N seconds, first channel.

    A PCM tuple avoids URL/file fetching inside the third-party wrapper. Restrict
    this adapter's references to PCM16 WAV rather than silently converting them.
    """
    import numpy as np
    path = Path(path)
    with wave.open(str(path), 'rb') as source:
        if source.getsampwidth() != 2 or source.getcomptype() != 'NONE':
            raise ValueError('Qwen voice references must be PCM16 WAV files')
        rate, channels = source.getframerate(), source.getnchannels()
        data = source.readframes(int(rate * seconds))
    if not data:
        raise ValueError('Voice reference is empty')
    pcm = np.frombuffer(data, dtype='<i2').reshape(-1, channels)[:, 0].copy()
    with path.open('rb') as source:
        file_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    return pcm.astype(np.float32) / 32768, rate, {
        'file_sha256': file_hash,
        'sample_sha256': hashlib.sha256(pcm.tobytes()).hexdigest(),
        'seconds': len(pcm) / rate, 'sample_rate': rate, 'channel': 0,
    }


def icl_reference(path, seconds=10):
    """The reference samples gim_api TTS Super gives qwen_tts for an in-context clone (its prepare_ref_audio).

    A file of at most `seconds` is read whole and mixed to mono (librosa.load); a longer one is cut to its first
    channel's first `seconds`, stored as PCM16 and read back, as gim_api's temporary copy is. The native rate is kept;
    qwen_tts resamples. As for x-vector references, only PCM16 WAV files are accepted.
    """
    import librosa
    import numpy as np
    import soundfile as sf
    path = Path(path)
    with wave.open(str(path), 'rb') as source:
        if source.getsampwidth() != 2 or source.getcomptype() != 'NONE':
            raise ValueError('Qwen voice references must be PCM16 WAV files')
    info = sf.info(str(path))
    truncated, source = info.duration > seconds, str(path)
    if truncated:
        data, rate = sf.read(source, frames=int(info.samplerate * seconds), dtype='float32')
        source = io.BytesIO()
        sf.write(source, data[:, 0] if data.ndim > 1 else data, rate, format='WAV')
        source.seek(0)
    audio, rate = librosa.load(source, sr=None, mono=True)
    audio, rate = np.asarray(audio, dtype=np.float32), int(rate)
    if not len(audio):
        raise ValueError('Voice reference is empty')
    with path.open('rb') as handle:
        file_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    return audio, rate, {
        'file_sha256': file_hash, 'sample_sha256': hashlib.sha256(audio.tobytes()).hexdigest(),
        'seconds': len(audio) / rate, 'sample_rate': rate, 'truncated': truncated,
    }


def frame_budget(text, defaults, requested=None):
    """The request's codec-frame cap: its frame_budget when given, else the model's estimate from the text length."""
    ceiling = int(defaults.get('max_frames', 7500))
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 16:
            raise ValueError('frame_budget must be a whole number of at least 16 codec frames')
        if requested > ceiling:
            raise ValueError(f"frame_budget {requested} exceeds this model's max_frames {ceiling}")
        return requested
    cap = max(int(defaults.get('minimum_frames', 240)), math.ceil(len(text) * float(defaults.get('frames_per_character', 3.0))) + 1)
    if cap < 1 or ceiling < 1 or cap > ceiling:
        raise ValueError('Speech exceeds the configured Qwen frame budget; split it into shorter sentences')
    return cap


def streaming(defaults):
    """A model's streaming knobs; unset ones keep this adapter's earlier behaviour (steady 8-frame chunks, no ramp,
    no overlap). An overlap is used only with a known seam_join."""
    join = defaults.get('seam_join', 'continuity')
    return {'emit_every_frames': int(defaults.get('emit_every_frames', 8)),
            'first_emit_frames': int(defaults.get('first_emit_frames', 0)),
            'decode_window_frames': int(defaults.get('decode_window_frames', 80)),
            'overlap_samples': int(defaults.get('overlap_samples', 0)) if join in SEAM_JOINS else 0,
            'seam_join': join}


def seam_continuity_join(prev_tail, new_head):
    """gim_api TTS Super's chunk join, used in place of qwen_tts' _crossfade(prev_tail, new_head).

    qwen_tts blends samples it already sent into the head of the next chunk, so that audio plays twice and the seam
    steps back. This keeps every new sample and fades out, over the overlap, only the new chunk's offset from where
    the sent audio was heading (its last sample plus its last step). The length is unchanged.
    """
    import numpy as np
    n = min(len(prev_tail), len(new_head))
    if n <= 0:
        return new_head
    last = float(prev_tail[-1])
    slope = last - float(prev_tail[-2]) if len(prev_tail) >= 2 else 0.0
    step = (last + slope) - float(new_head[0])
    fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    joined = new_head[:n] + np.float32(step) * fade
    return joined.astype(np.float32, copy=False)


def ramp_stream_chunks(chunks, first_emit_frames, steady_emit_frames):
    """gim_api TTS Super's first-chunk ramp over chunks generated every first_emit_frames codec frames.

    The first chunk goes out alone, the next two together, then as many per batch as make steady_emit_frames. Every
    sample is kept; the remainder is flushed at the end. Chunks pass through unchanged unless 0 < first < steady.
    """
    import numpy as np
    if first_emit_frames <= 0 or first_emit_frames >= steady_emit_frames:
        for chunk in chunks:
            yield chunk
        return
    steady_batch = -(-steady_emit_frames // first_emit_frames)
    pending = []
    batch_target = 1
    for chunk in chunks:
        pending.append(chunk)
        if len(pending) >= batch_target:
            if len(pending) == 1:
                yield pending[0]
            else:
                yield np.concatenate(pending).astype(np.float32, copy=False)
            pending = []
            batch_target = 2 if batch_target == 1 else steady_batch
    if pending:
        if len(pending) == 1:
            yield pending[0]
        else:
            yield np.concatenate(pending).astype(np.float32, copy=False)


def install_seams():
    """Route qwen_tts' chunk join by thread: seam_continuity_join while the thread's _seam.join is "continuity",
    qwen_tts' own _crossfade otherwise. Installed once per process; other syntheses keep qwen_tts' join."""
    from qwen_tts.core.models import modeling_qwen3_tts as module
    with _seam_lock:
        if getattr(module._crossfade, 'titanos_seams', False):
            return
        original = module._crossfade

        def dispatch(prev_tail, new_head):
            if getattr(_seam, 'join', None) == 'continuity':
                _seam.joins += 1
                return seam_continuity_join(prev_tail, new_head)
            return original(prev_tail, new_head)
        dispatch.titanos_seams = True
        module._crossfade = dispatch


class QwenTTSModel:
    def __init__(self, spec):
        import torch
        if spec.task != 'speech' or spec.parallelism != 'single':
            raise ValueError('Qwen3-TTS currently supports single-device speech models')
        root = Path(spec.path)
        required = ['config.json', 'model.safetensors', 'speech_tokenizer/config.json', 'speech_tokenizer/model.safetensors']
        if any(not (root / name).is_file() for name in required):
            raise ValueError('Qwen3-TTS needs the complete local Base checkpoint and speech tokenizer')
        provenance = metadata.distribution('qwen-tts').read_text('direct_url.json')
        revision = json.loads(provenance or '{}').get('vcs_info', {}).get('commit_id')
        if revision != WRAPPER_REVISION:
            raise ValueError('Install the pinned Qwen3-TTS streaming wrapper revision documented by this adapter')
        self.spec, self.device = spec, select_device(spec.device)
        self.wrapper_revision, self.prompts, self.references, self.last_generation = revision, {}, {}, None
        from qwen_tts import Qwen3TTSModel
        dtype = spec.dtype if spec.dtype != 'auto' else ('bfloat16' if self.device.startswith('cuda') else 'float32')
        options = dict(spec.load_options)
        options.setdefault('attn_implementation', 'sdpa')
        options.setdefault('low_cpu_mem_usage', True)
        self.model = Qwen3TTSModel.from_pretrained(str(root), device_map=self.device,
            dtype=getattr(torch, dtype), local_files_only=True, use_safetensors=True, **options)
        self.model.model.eval()
        if self.model.model.tts_model_type != 'base':
            raise ValueError('Voice cloning requires a Qwen3-TTS Base checkpoint')
        if not hasattr(self.model, 'stream_generate_voice_clone'):
            raise ValueError('The selected Qwen wrapper lacks the required streaming voice-clone method')
        self.model.enable_streaming_optimizations(decode_window_frames=int(spec.defaults.get('decode_window_frames', 80)),
            use_compile=False, use_cuda_graphs=False, use_fast_codebook=True,
            compile_codebook_predictor=False, compile_talker=False)

    def details(self):
        import torch
        placement = {'parameter_devices': sorted({str(p.device) for p in self.model.model.parameters()})}
        if self.device.startswith('cuda'):
            placement['gpu_uuid'] = str(torch.cuda.get_device_properties(self.device).uuid)
        texts = voice_texts(self.spec)
        modes = {voice: 'icl' if voice in texts else 'x_vector' for voice in sorted(self.spec.voices)}
        kinds = set(modes.values())
        mode = 'x_vector_only' if kinds <= {'x_vector'} else 'icl' if kinds == {'icl'} else 'mixed'
        text_hashes = {voice: hashlib.sha256(text.encode()).hexdigest() for voice, text in texts.items()}
        return {'backend': 'qwen3-tts', 'device': self.device, 'dtype': str(self.model.model.dtype),
                **placement,
                'model_repository': self.spec.source_repo, 'model_revision': self.spec.source_revision,
                'wrapper_revision': self.wrapper_revision, 'voice_mode': mode, 'voice_modes': modes,
                'voice_text_sha256': text_hashes,
                'registered_voices': sorted(self.spec.voices), 'reference_bindings': self.references,
                'streaming': {**streaming(self.spec.defaults), 'fast_codebook': True, 'torch_compile': False,
                              'cuda_graphs': False},
                'provider': 'Titanos', 'last_generation': self.last_generation}

    def infer(self, task, data, emit=None, cancel=None):
        import numpy as np
        import torch
        if task != 'speech':
            raise ValueError('Qwen3-TTS implements speech only')
        if data.get('speed', 1.0) != 1.0:
            raise ValueError('Qwen speech uses natural speed 1.0; tempo fitting is not supported')
        voice = data.get('voice', 'default')
        if voice not in self.spec.voices:
            raise ValueError(f'Unknown configured voice {voice!r}; reference aliases never fall back')
        text = data['input']
        budget = frame_budget(text, self.spec.defaults, data.get('frame_budget'))
        language = LANGUAGES.get(data.get('language') or 'en', data.get('language') or 'english')
        transcript = voice_texts(self.spec).get(voice)
        mode = 'icl' if transcript else 'x_vector'
        if voice not in self.prompts:
            seconds = float(self.spec.defaults.get('reference_seconds', 10 if transcript else 5))
            if transcript:
                audio, rate, receipt = icl_reference(self.spec.voices[voice], seconds)
                prompt = self.model.create_voice_clone_prompt(ref_audio=(audio, rate), ref_text=transcript,
                                                              x_vector_only_mode=False)
                receipt['ref_text_sha256'] = hashlib.sha256(transcript.encode()).hexdigest()
                if getattr(prompt[0], 'ref_code', None) is not None:
                    code = prompt[0].ref_code.detach().cpu().numpy().tobytes()
                    receipt['ref_code_sha256'] = hashlib.sha256(code).hexdigest()
            else:
                audio, rate, receipt = reference_audio(self.spec.voices[voice], seconds)
                prompt = self.model.create_voice_clone_prompt(ref_audio=(audio, rate), ref_text=None,
                                                              x_vector_only_mode=True)
            receipt['mode'] = mode
            receipt['speaker_embedding_sha256'] = hashlib.sha256(prompt[0].ref_spk_embedding.detach().float().cpu().numpy().tobytes()).hexdigest()
            self.prompts[voice], self.references[voice] = prompt, receipt
        seed = data.get('seed')
        if seed is None:
            seed = self.spec.defaults.get('seed')
        if seed is None:
            seed = secrets.randbits(32)
        torch.manual_seed(seed)
        parameters = {name: self.spec.defaults[name] for name in ['do_sample', 'top_k', 'top_p', 'temperature',
                      'subtalker_dosample', 'subtalker_top_k', 'subtalker_top_p', 'subtalker_temperature']
                      if name in self.spec.defaults}
        parameters.update({name: sampling_value(name, data[name]) for name in SAMPLING if data.get(name) is not None})
        knobs = streaming(self.spec.defaults)
        steady, first = knobs['emit_every_frames'], knobs['first_emit_frames']
        emitted = first if 0 < first < steady else steady
        if knobs['overlap_samples'] and knobs['seam_join'] == 'continuity':
            install_seams()
        deadline = time.monotonic() + current_settings().client_timeout
        chunks, rate, first_audio = [], None, None
        started = time.monotonic()

        def samples(generator):
            nonlocal rate
            for chunk, sample_rate in generator:
                if cancel is not None and cancel.is_set():
                    raise EngineError('Speech cancelled', 499, 'cancelled')
                if time.monotonic() > deadline:
                    raise EngineError('Speech exceeded the configured inference deadline', 504, 'deadline_exceeded')
                if rate is not None and rate != sample_rate:
                    raise ValueError('Qwen stream changed sample rate')
                rate = sample_rate
                value = chunk.detach().float().cpu() if isinstance(chunk, torch.Tensor) else chunk
                yield np.asarray(value, dtype=np.float32).reshape(-1)

        # With emit, the stream record and then each chunk's PCM16 go out as they are made (speech_stream.py).
        # The whole WAV is still returned: replica workers pass an emit even when their caller wants only the WAV.
        generator = pieces = batches = None
        _seam.join, _seam.joins = knobs['seam_join'], 0
        try:
            generator = self.model.stream_generate_voice_clone(text=text, language=language,
                voice_clone_prompt=self.prompts[voice], emit_every_frames=emitted,
                decode_window_frames=knobs['decode_window_frames'], overlap_samples=knobs['overlap_samples'],
                max_frames=budget, **parameters)
            pieces = samples(generator)
            batches = ramp_stream_chunks(pieces, first, steady)
            for value in batches:
                if not len(value):
                    continue
                if not np.isfinite(value).all():
                    raise ValueError('Qwen produced non-finite audio')
                if first_audio is None:
                    first_audio = time.monotonic() - started
                    if emit is not None:
                        emit({'sample_rate': int(rate), 'channels': 1, 'sample_width': 2})
                chunks.append(value)
                if emit is not None:
                    emit(pcm16(value))
        finally:
            joins, _seam.join = _seam.joins, None
            for item in (batches, pieces, generator):
                if item is not None:
                    item.close()
        if not chunks or not rate:
            raise ValueError('Qwen produced no speech samples')
        waveform = np.concatenate(chunks)
        frames = int(len(waveform) / rate * 12.5 + 0.5)
        if frames >= budget - 1:
            raise ValueError('Qwen exhausted its frame budget; truncated audio was rejected')
        self.last_generation = {'provider': 'Titanos', 'backend': 'qwen3-tts', 'voice': voice, 'seed': seed,
            'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'sample_rate': rate, 'samples': len(waveform),
            'duration_seconds': len(waveform) / rate, 'generation_seconds': time.monotonic() - started,
            'first_audio_seconds': first_audio, 'chunks': len(chunks), 'voice_mode': mode,
            'frame_budget': budget, 'produced_frames': frames, 'peak': float(np.max(np.abs(waveform))),
            'sampling': parameters, 'emit_every_frames': emitted, 'first_emit_frames': first,
            'overlap_samples': knobs['overlap_samples'], 'seam_join': knobs['seam_join'], 'continuity_joins': joins,
            'provider_speed': 1.0, 'tempo_factor': 1.0, 'duration_fit': False}
        return write_wav(waveform, rate)

    def close(self):
        self.prompts.clear()
        self.model = None
