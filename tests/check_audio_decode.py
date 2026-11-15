"""Optional real decoder boundary check; synthetic PCM only, no learned model."""
import io
import wave
from titanos.media import read_audio
from titanos.settings import Settings, _active_settings

def pcm(rate, count):
    target=io.BytesIO()
    with wave.open(target,'wb') as output:
        output.setnchannels(1);output.setsampwidth(2);output.setframerate(rate)
        output.writeframes(b'\0\0'*count)
    return target.getvalue()

for rate in [8000,16000]:
    raw=pcm(rate,rate//10)
    result=read_audio(raw,rate)
    assert len(result)==rate//10 and (result==0).all()
    token=_active_settings.set(Settings(max_audio_seconds=.01,diarization_min_duration_on=.005,diarization_min_duration_off=.005))
    try:
        try:read_audio(raw,rate)
        except ValueError as exc:assert 'limits' in str(exc)
        else:raise AssertionError('Oversize recording silently cut')
    finally:_active_settings.reset(token)
try:read_audio(b'invalid audio',16000)
except ValueError:pass
else:raise AssertionError('Invalid audio accepted')
print('PASS: real FFmpeg PCM decode at8k/16k, exact duration and oversize/invalid rejection. No model.')
