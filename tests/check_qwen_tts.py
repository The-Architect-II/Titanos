"""Qwen voice adapter control checks; numerical model inference is a separate gate."""
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import wave

import numpy as np
import torch
from titanos.qwen_tts_backend import QwenTTSModel, frame_budget, reference_audio


def main():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'vega.wav'
        stereo = np.column_stack([np.arange(100, dtype=np.int16), np.arange(100, dtype=np.int16) + 1000])
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(2); output.setsampwidth(2); output.setframerate(20); output.writeframes(stereo.tobytes())
        audio, rate, proof = reference_audio(path, 2)
        assert rate == 20 and len(audio) == 40
        np.testing.assert_array_equal(audio, stereo[:40, 0].astype(np.float32) / 32768)
        assert proof['sample_sha256'] == hashlib.sha256(stereo[:40, 0].tobytes()).hexdigest()
        assert frame_budget('a' * 100, {}) == 301
        try:
            frame_budget('a' * 100, {'max_frames': 240})
            raise AssertionError('Oversized request accepted')
        except ValueError: pass

        class Fake:
            def create_voice_clone_prompt(self, **kwargs):
                assert kwargs['x_vector_only_mode'] is True and kwargs['ref_text'] is None
                return [SimpleNamespace(ref_spk_embedding=torch.ones(4))]
            def stream_generate_voice_clone(self, **kwargs):
                assert kwargs['max_frames'] == 240
                yield np.ones(2400, np.float32) * 0.1, 24000
        model = QwenTTSModel.__new__(QwenTTSModel)
        model.spec = SimpleNamespace(voices={'vega': str(path)}, defaults={'seed': 1})
        model.model, model.prompts, model.references = Fake(), {}, {}
        result = model.infer('speech', {'input': 'Hello.', 'voice': 'vega', 'speed': 1.0})
        with wave.open(io.BytesIO(result), 'rb') as wav:
            assert wav.getnframes() == 2400 and wav.getframerate() == 24000
        assert model.last_generation['provider_speed'] == 1.0 and model.last_generation['tempo_factor'] == 1.0
        assert model.last_generation['duration_fit'] is False
        # A nullable API/model default means unseeded generation; explicit zero
        # is a real seed, and an explicit request overrides the model default.
        for default, request, expected in [(None, None, None), (None, 101, 101), (0, None, 0)]:
            model.spec.defaults = {'seed': default}
            model.infer('speech', {'input': 'Hello.', 'voice': 'vega', 'seed': request})
            selected = model.last_generation['seed']
            assert isinstance(selected, int) and 0 <= selected < 2**32
            assert torch.initial_seed() == selected
            if expected is not None:
                assert selected == expected
        for request in [{'input': 'Hello', 'voice': 'other'}, {'input': 'Hello', 'voice': 'vega', 'speed': 1.1}]:
            try:
                model.infer('speech', request)
                raise AssertionError('Voice substitution or tempo change was accepted')
            except ValueError: pass
    print('PASS: reference first-channel samples, frame budgets, exact voice selection, natural-speed output; fake numerical inference only')


if __name__ == '__main__': main()
