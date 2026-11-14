"""Generated music limits are configurable independently of uploaded audio."""
import io
import unittest
from unittest.mock import patch
import wave

import numpy as np
from titanos.media import write_music_wav
from titanos.settings import Settings


class Checks(unittest.TestCase):
    def test_default_budget_accepts_declared_ten_minute_stereo_size(self):
        settings=Settings.from_env({})
        samples=600*48000*2
        self.assertLessEqual(samples,settings.max_music_samples)
        self.assertLessEqual(44+samples*2,settings.max_music_output_bytes)

    def test_output_independent_of_small_input_limits(self):
        settings=Settings.from_env({'TITANOS_MAX_BODY_BYTES':'64','TITANOS_MAX_AUDIO_SAMPLES':'10',
                                    'TITANOS_MAX_MUSIC_SAMPLES':'800','TITANOS_MAX_MUSIC_OUTPUT_BYTES':'2048'})
        with patch('titanos.media.current_settings',return_value=settings):
            raw=write_music_wav(np.zeros((2,400),dtype=np.float32),8000)
        with wave.open(io.BytesIO(raw)) as reader:
            self.assertEqual((reader.getnchannels(),reader.getnframes()),(2,400))
        self.assertGreater(len(raw),settings.max_body_bytes)

    def test_each_output_limit_rejects_before_encoding(self):
        # The audio-seconds case keeps the unrelated diarization minimum durations within the lowered ceiling;
        # Settings rejects the inconsistent combination before this music bound is reached.
        for limits in [{'max_music_samples':799},{'max_music_output_bytes':1643},
                       {'max_audio_seconds':.04,'diarization_min_duration_on':.01,'diarization_min_duration_off':.01}]:
            with self.subTest(limits=limits),patch('titanos.media.current_settings',return_value=Settings(**limits)):
                with self.assertRaisesRegex(ValueError,'limits'):
                    write_music_wav(np.zeros((2,400),dtype=np.float32),8000)

    def test_request_can_raise_steps_and_lower_duration_limit(self):
        from titanos.server import MusicRequest
        with patch('titanos.server.current_settings',return_value=Settings(max_steps=500)):
            self.assertEqual(MusicRequest(model='music',prompt='quiet instrumental',steps=401).steps,401)
        with patch('titanos.server.current_settings',return_value=Settings(max_audio_seconds=20)):
            with self.assertRaisesRegex(ValueError,'MAX_AUDIO_SECONDS'):
                MusicRequest(model='music',prompt='quiet instrumental',duration=30)


if __name__=='__main__':unittest.main(verbosity=2)
