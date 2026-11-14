"""Real FFmpeg PCM boundaries and the ASR generation-options contract."""
from types import SimpleNamespace
import numpy as np
from titanos.media import read_audio,write_wav
from titanos.settings import Settings,_active_settings
from titanos.backends import LoadedModel
from titanos.config import ModelSpec


def main():
    samples=np.sin(np.arange(800,dtype=np.float32)*.05)*.2
    wav=write_wav(samples,16000)
    assert wav.startswith(b'RIFF')
    decoded=read_audio(wav,16000)
    assert len(decoded)==800 and np.max(np.abs(decoded-samples))<.001
    token=_active_settings.set(Settings(max_audio_seconds=.05,diarization_min_duration_on=.01,diarization_min_duration_off=.01))
    try:
        assert len(read_audio(wav,16000))==800
        for value in [write_wav(np.zeros(801,dtype=np.float32),16000),b'not audio']:
            try:read_audio(value,16000);raise AssertionError('Invalid/over-limit audio accepted')
            except ValueError:pass
    finally:_active_settings.reset(token)
    token=_active_settings.set(Settings(max_audio_samples=400))
    try:
        try:read_audio(wav,16000);raise AssertionError('Sample ceiling ignored')
        except ValueError:pass
    finally:_active_settings.reset(token)
    seen=[]
    class Pipeline:
        feature_extractor=SimpleNamespace(sampling_rate=16000)
        model=SimpleNamespace(config=SimpleNamespace(model_type='whisper'),generation_config=SimpleNamespace(is_multilingual=True))
        # Match the actual pipeline boundary: max_tokens is not a top-level ASR option.
        def __call__(self,audio,generate_kwargs=None):
            seen.append(generate_kwargs);return {'text':'decoded fixture'}
    loaded=LoadedModel.__new__(LoadedModel)
    loaded.spec=ModelSpec(path='not-loaded',task='transcription',defaults={'max_tokens':6})
    loaded.model=Pipeline()
    assert loaded.transcribe({'audio':wav})=='decoded fixture'
    assert seen[-1]['max_new_tokens']==6
    assert loaded.transcribe({'audio':wav,'max_tokens':4,'translation':True})=='decoded fixture'
    assert seen[-1]=={'max_new_tokens':4,'task':'translate'}
    print('PASS: real FFmpeg WAV roundtrip, exact duration/sample ceilings, malformed audio, ASR model/request token precedence and translation task; no model-quality claim')


if __name__=='__main__':main()
