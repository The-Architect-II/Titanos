"""Qwen adapter streaming and in-context cloning control checks with a fake model; no numerical inference claims.

gim_api TTS Super is the reference for the request sampling ranges, the frame-budget estimate, the first-chunk ramp,
the seam join and the ICL reference preparation. Its reference preparation (prepare_ref_audio, then qwen_tts' librosa
load of the file it hands over) is rebuilt here from its source and compared sample for sample.
"""
import hashlib
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import wave

import librosa
import numpy as np
import soundfile as sf
import torch
from qwen_tts.core.models import modeling_qwen3_tts as qwen_module

from titanos import qwen_tts_backend as backend
from titanos.config import ModelSpec
from titanos.engine import EngineError
from titanos.qwen_tts_backend import (QwenTTSModel, check_defaults, frame_budget, icl_reference, install_seams,
                                      ramp_stream_chunks, sampling_value, seam_continuity_join, streaming)

RATE = 24000
FRAME = int(RATE / 12.5)  # samples in one codec frame


def refused(call, text=None, kind=ValueError):
    try:
        call()
    except kind as exc:
        assert text is None or text in str(exc), f'{text!r} not in {exc}'
        return exc
    raise AssertionError(f'accepted; expected {kind.__name__} {text!r}')


def write_pcm16(path, samples, rate, width=2):
    samples = np.asarray(samples, dtype='<i2' if width == 2 else 'u1')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())


def gim_reference(path, scratch, max_seconds=10):
    """gim_api TTS Super's prepare_ref_audio (MAX_REF_SECONDS 10), then qwen_tts' load of the path it hands over."""
    info = sf.info(str(path))
    source = str(path)
    if info.duration > max_seconds:
        data, rate = sf.read(source, frames=int(info.samplerate * max_seconds), dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]
        source = str(Path(scratch) / 'ref_trunc_gim.wav')
        sf.write(source, data, rate)
    audio, rate = librosa.load(source, sr=None, mono=True)
    return audio.astype(np.float32), int(rate)


def check_sampling():
    assert sampling_value('temperature', 0.7) == 0.7 and type(sampling_value('temperature', 1)) is float
    assert sampling_value('top_k', 50) == 50 and type(sampling_value('top_k', 50.0)) is int
    assert sampling_value('subtalker_top_p', 1) == 1.0
    for name, value, text in [('temperature', True, 'must be a number'), ('top_k', '50', 'must be a number'),
                              ('top_k', None, 'must be a number'), ('top_k', 1.5, 'whole number'),
                              ('top_k', float('inf'), 'whole number'), ('top_k', float('nan'), 'whole number'),
                              ('temperature', 0.04, 'within 0.05 and 2.0'), ('temperature', 2.5, 'within'),
                              ('temperature', float('nan'), 'within'), ('top_p', 0, 'within'),
                              ('subtalker_top_k', 4097, 'within 1 and 4096'), ('subtalker_temperature', -1, 'within')]:
        refused(lambda: sampling_value(name, value), text)


def check_registry(scratch, voice):
    check_defaults({})
    check_defaults({'seam_join': 'qwen', 'first_emit_frames': 2, 'overlap_samples': 120})
    for bad, text in [({'seam_join': 'blend'}, 'seam_join must be one of continuity, qwen'),
                      ({'first_emit_frames': -1}, 'first_emit_frames'),
                      ({'first_emit_frames': 1.5}, 'first_emit_frames'),
                      ({'overlap_samples': True}, 'overlap_samples')]:
        refused(lambda: check_defaults(bad), text)

    def spec(**fields):
        return ModelSpec(**{'path': str(scratch), 'task': 'speech', 'backend': 'qwen3-tts',
                            'voices': {'host': str(voice)}, **fields})
    assert spec(voice_texts={'host': 'Hello.'}).voice_texts == {'host': 'Hello.'}
    for fields, text in [({'voice_texts': {'guest': 'Hello.'}}, 'registered voice ids'),
                         ({'voice_texts': {'host': '  '}}, 'nonblank reference transcripts'),
                         ({'defaults': {'seam_join': 'blend'}}, 'seam_join'),
                         ({'defaults': {'overlap_samples': -120}}, 'overlap_samples'),
                         ({'backend': 'transformers', 'voice_texts': {'host': 'Hello.'}},
                          'qwen3-tts speech models only')]:
        refused(lambda: spec(**fields), text)


def check_budgets():
    # gim_api TTS Super's estimate: max(240, int(max(1, len(text)) * 3.0) + 1), its 413 above max_frames 7500
    for length in (0, 1, 79, 80, 100, 2499):
        assert frame_budget('a' * length, {}) == max(240, int(max(1, length) * 3.0) + 1), length
    refused(lambda: frame_budget('a' * 2500, {}), 'split it into shorter sentences')
    assert frame_budget('Hi.', {}, 16) == 16 and frame_budget('Hi.', {}, 7500) == 7500
    for requested, text in [(15, 'at least 16'), (True, 'whole number'), (16.0, 'whole number'),
                            (7501, "exceeds this model's max_frames 7500")]:
        refused(lambda: frame_budget('Hi.', {}, requested), text)
    refused(lambda: frame_budget('Hi.', {'max_frames': 2000}, 3000), "max_frames 2000")
    assert streaming({}) == {'emit_every_frames': 8, 'first_emit_frames': 0, 'decode_window_frames': 80,
                             'overlap_samples': 0, 'seam_join': 'continuity'}
    rig = {'emit_every_frames': 8, 'first_emit_frames': 2, 'decode_window_frames': 80, 'overlap_samples': 120,
           'seam_join': 'continuity'}
    assert streaming(rig) == rig


def check_ramp_and_seams():
    chunks = [np.full(2 * FRAME, index, np.float32) for index in range(10)]
    batches = list(ramp_stream_chunks(iter(chunks), 2, 8))
    assert [len(batch) // (2 * FRAME) for batch in batches] == [1, 2, 4, 3]
    np.testing.assert_array_equal(np.concatenate(batches), np.concatenate(chunks))
    for first in (0, 8, 9):
        assert [id(batch) for batch in ramp_stream_chunks(chunks, first, 8)] == [id(chunk) for chunk in chunks]

    sent = np.linspace(0.0, 0.5, 200, dtype=np.float32)  # still rising when the next chunk starts
    tail, head = sent[-120:], np.full(120, -0.25, np.float32)
    joined = seam_continuity_join(tail, head)
    slope = float(sent[-1]) - float(sent[-2])
    assert len(joined) == 120 and joined.dtype == np.float32
    assert abs(joined[0] - (sent[-1] + slope)) < 1e-6 and abs(joined[-1] - head[-1]) < 1e-6
    assert seam_continuity_join(tail[:0], head) is head
    assert abs(seam_continuity_join(tail[-1:], head)[0] - tail[-1]) < 1e-6  # one sent sample: no slope
    original = qwen_module._crossfade
    assert abs(original(tail, head)[0] - tail[0]) < 1e-6  # qwen_tts' join starts over on audio already sent

    assert not getattr(original, 'titanos_seams', False)
    try:
        install_seams()
        dispatch = qwen_module._crossfade
        install_seams()
        assert dispatch.titanos_seams and dispatch is not original and qwen_module._crossfade is dispatch
        backend._seam.join = None
        np.testing.assert_array_equal(dispatch(tail, head), original(tail, head))
        backend._seam.join, backend._seam.joins = 'continuity', 0
        np.testing.assert_array_equal(dispatch(tail, head), seam_continuity_join(tail, head))
        assert backend._seam.joins == 1
        seen = {}
        other = threading.Thread(target=lambda: seen.setdefault('joined', dispatch(tail, head)))
        other.start()
        other.join()
        np.testing.assert_array_equal(seen['joined'], original(tail, head))  # other syntheses keep qwen_tts' join
        backend._seam.join = 'qwen'
        np.testing.assert_array_equal(dispatch(tail, head), original(tail, head))
        assert backend._seam.joins == 1
    finally:
        backend._seam.join = None
        qwen_module._crossfade = original


def check_icl_references(scratch):
    rng = np.random.default_rng(7)
    short = scratch / 'short.wav'  # stereo 4 s: read whole and mixed to mono
    write_pcm16(short, rng.integers(-20000, 20000, (4 * 16000, 2)), 16000)
    long = scratch / 'long.wav'  # stereo 12 s: its first channel's first 10 s
    write_pcm16(long, rng.integers(-20000, 20000, (12 * 22050, 2)), 22050)
    mono = scratch / 'mono.wav'
    write_pcm16(mono, rng.integers(-20000, 20000, 6 * 8000), 8000)
    for path in (short, long, mono):
        audio, rate, receipt = icl_reference(path)
        expected, expected_rate = gim_reference(path, scratch)
        assert rate == expected_rate and audio.dtype == np.float32
        np.testing.assert_array_equal(audio, expected)
        assert receipt['truncated'] is (path == long) and receipt['sample_rate'] == rate
        assert receipt['sample_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
        assert receipt['file_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert icl_reference(long)[2]['seconds'] == 10.0 and icl_reference(short)[2]['seconds'] == 4.0
    eight_bit = scratch / 'eight.wav'
    write_pcm16(eight_bit, np.full(800, 128), 8000, width=1)
    refused(lambda: icl_reference(eight_bit), 'PCM16 WAV')
    return short, long, mono


class FakeQwen:
    """The two qwen_tts calls the adapter makes. Generation yields the given chunks and, for the first `joins` of
    them, calls qwen_tts' chunk join the way its streaming decoder does."""

    def __init__(self, chunks, joins=0):
        self.chunks, self.joins, self.prompts, self.calls = chunks, joins, [], []

    def create_voice_clone_prompt(self, **kwargs):
        self.prompts.append(kwargs)
        code = torch.arange(6) if kwargs['ref_text'] else None
        return [SimpleNamespace(ref_spk_embedding=torch.ones(4), ref_code=code)]

    def stream_generate_voice_clone(self, **kwargs):
        self.calls.append({**kwargs, 'seam_join': getattr(backend._seam, 'join', None)})
        for index, chunk in enumerate(self.chunks):
            if index < self.joins:
                qwen_module._crossfade(np.zeros(4, np.float32), np.ones(4, np.float32))
            yield chunk, RATE


def adapter(fake, voices, texts, defaults):
    model = QwenTTSModel.__new__(QwenTTSModel)
    model.spec = SimpleNamespace(voices=voices, voice_texts=texts, defaults=defaults)
    model.model, model.prompts, model.references, model.last_generation = fake, {}, {}, None
    return model


def check_inference(scratch, short, long, mono):
    tone = [(0.3 * np.sin(np.arange(2 * FRAME) / 9 + index)).astype(np.float32) for index in range(10)]
    fake = FakeQwen(tone)
    model = adapter(fake, {'host': str(long), 'guest': str(mono)}, {'host': '  The host reads this.  '},
                    {'seed': 1, 'temperature': 0.9, 'first_emit_frames': 2, 'emit_every_frames': 8})
    emitted = []
    wav = model.infer('speech', {'input': 'Hello there.', 'voice': 'host'}, emitted.append)
    assert emitted[0] == {'sample_rate': RATE, 'channels': 1, 'sample_width': 2}
    audio = emitted[1:]
    assert len(audio) == 4 and all(type(piece) is bytes for piece in audio)
    with wave.open(io.BytesIO(wav), 'rb') as reader:
        assert reader.getframerate() == RATE and b''.join(audio) == reader.readframes(reader.getnframes())
    call = fake.calls[0]
    assert call['emit_every_frames'] == 2 and call['decode_window_frames'] == 80 and call['overlap_samples'] == 0
    assert call['max_frames'] == 240 and call['temperature'] == 0.9 and 'top_k' not in call
    assert call['language'] == 'english' and call['seam_join'] == 'continuity'
    prompt = fake.prompts[0]
    assert prompt['ref_text'] == 'The host reads this.' and prompt['x_vector_only_mode'] is False
    expected, expected_rate = gim_reference(long, scratch)
    np.testing.assert_array_equal(prompt['ref_audio'][0], expected)
    assert prompt['ref_audio'][1] == expected_rate
    receipt = model.references['host']
    assert receipt['mode'] == 'icl' and receipt['truncated'] is True and receipt['seconds'] == 10.0
    assert receipt['ref_text_sha256'] == hashlib.sha256(b'The host reads this.').hexdigest()
    assert receipt['ref_code_sha256'] == hashlib.sha256(torch.arange(6).numpy().tobytes()).hexdigest()
    generation = model.last_generation
    assert generation['voice_mode'] == 'icl' and generation['chunks'] == 4 and generation['produced_frames'] == 20
    assert generation['frame_budget'] == 240 and generation['sampling'] == {'temperature': 0.9}
    assert generation['emit_every_frames'] == 2 and generation['first_emit_frames'] == 2
    assert generation['continuity_joins'] == 0 and generation['seed'] == 1

    # Request sampling overrides the model's; frame_budget caps the generation; the prompt is cached per voice.
    model.infer('speech', {'input': 'Hello there.', 'voice': 'host', 'language': 'ja', 'temperature': 0.7, 'top_k': 50,
                           'frame_budget': 100})
    call = fake.calls[-1]
    assert call['temperature'] == 0.7 and call['top_k'] == 50 and call['max_frames'] == 100
    assert call['language'] == 'japanese' and len(fake.prompts) == 1
    assert model.last_generation['sampling'] == {'temperature': 0.7, 'top_k': 50}
    refused(lambda: model.infer('speech', {'input': 'Hi.', 'voice': 'host', 'top_k': 0}), 'within 1 and 4096')

    model.infer('speech', {'input': 'Hi.', 'voice': 'guest'})
    prompt = fake.prompts[-1]
    assert prompt['x_vector_only_mode'] is True and prompt['ref_text'] is None
    assert model.references['guest']['mode'] == 'x_vector' and model.references['guest']['seconds'] == 5.0
    assert model.last_generation['voice_mode'] == 'x_vector'

    # 20 frames against a 21-frame cap is gim_api's truncation; the audio was already emitted (the HTTP stream aborts).
    emitted = []
    refused(lambda: model.infer('speech', {'input': 'Hi.', 'voice': 'host', 'frame_budget': 21}, emitted.append),
            'exhausted its frame budget')
    assert len(emitted) == 5
    model.infer('speech', {'input': 'Hi.', 'voice': 'host', 'frame_budget': 22})

    cancel = threading.Event()
    cancel.set()
    exc = refused(lambda: model.infer('speech', {'input': 'Hi.', 'voice': 'host'}, None, cancel), kind=EngineError)
    assert exc.status == 499 and exc.code == 'cancelled'
    broken = adapter(FakeQwen([np.array([0.1, np.nan], np.float32)]), {'guest': str(mono)}, {}, {'seed': 1})
    refused(lambda: broken.infer('speech', {'input': 'Hi.', 'voice': 'guest'}), 'non-finite')

    original = qwen_module._crossfade
    try:
        seams = adapter(FakeQwen(tone, joins=3), {'guest': str(mono)}, {}, {'seed': 1, 'overlap_samples': 120})
        seams.infer('speech', {'input': 'Hi.', 'voice': 'guest'})
        assert seams.model.calls[0]['overlap_samples'] == 120 and seams.model.calls[0]['seam_join'] == 'continuity'
        assert seams.last_generation['continuity_joins'] == 3 and getattr(backend._seam, 'join', None) is None
        qwen = adapter(FakeQwen(tone, joins=2), {'guest': str(mono)}, {},
                       {'seed': 1, 'overlap_samples': 120, 'seam_join': 'qwen'})
        qwen.infer('speech', {'input': 'Hi.', 'voice': 'guest'})
        assert qwen.model.calls[0]['seam_join'] == 'qwen' and qwen.last_generation['continuity_joins'] == 0
    finally:
        qwen_module._crossfade = original


def main():
    with tempfile.TemporaryDirectory() as temp:
        scratch = Path(temp)
        check_sampling()
        short, long, mono = check_icl_references(scratch)
        check_registry(scratch, short)
        check_budgets()
        check_ramp_and_seams()
        check_inference(scratch, short, long, mono)
    print('PASS: gim_api sampling ranges, frame budgets, ramp, seam join and ICL references; streamed PCM equals the '
          'WAV; fake numerical inference only')


if __name__ == '__main__': main()
