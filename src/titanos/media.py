"""Media conversion with the bundled FFmpeg and Python's PCM WAV writer."""
import io
import subprocess
import time
import tempfile
import wave

from .settings import current_settings


def read_audio(raw, sample_rate, cancel=None):
    import numpy as np
    import imageio_ffmpeg
    settings=current_settings()
    from .text_inputs import check_cancel
    check_cancel(cancel)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError('Decoded sample rate must be a positive integer')
    if not isinstance(raw, bytes) or not raw or len(raw) > settings.max_body_bytes:
        raise ValueError('Audio is empty or exceeds TITANOS_MAX_BODY_BYTES')
    limit=min(int(settings.max_audio_seconds*sample_rate),settings.max_audio_samples)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin',
             '-protocol_whitelist','pipe','-i','pipe:0','-map','0:a:0','-t',str((limit+1)/sample_rate),'-ac','1','-ar',str(sample_rate),
             '-f','f32le','-acodec','pcm_f32le','pipe:1']
    # A regular input handle avoids a blocking Windows communicate() stdin write.
    with tempfile.TemporaryFile() as source:
        source.write(raw)
        source.seek(0)
        with subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_ffmpeg_env(),
                              creationflags=subprocess.CREATE_NO_WINDOW if __import__('os').name=='nt' else 0) as process:
            deadline = time.monotonic() + settings.client_timeout
            try:
                while True:
                    check_cancel(cancel)
                    if time.monotonic() >= deadline:
                        raise ValueError('Audio decoding exceeded TITANOS_CLIENT_TIMEOUT')
                    try:
                        output, _ = process.communicate(timeout=.1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                process.kill()
                process.communicate()
                raise
    check_cancel(cancel)
    if process.returncode or len(output)%4:
        raise ValueError('Unsupported or invalid audio for the installed FFmpeg codecs')
    audio=np.frombuffer(output,dtype='<f4').copy()
    if not len(audio) or len(audio)>limit or not np.isfinite(audio).all():
        raise ValueError('Audio is empty, non-finite, or exceeds configured duration/sample limits')
    return audio


def pcm16(samples):
    """Little-endian PCM16 bytes of samples in [-1, 1]: clipped, scaled by 32767 and rounded, as write_wav stores them."""
    import numpy as np
    return (np.clip(np.asarray(samples),-1,1)*32767).round().astype('<i2').tobytes()


def write_wav(samples, sample_rate):
    import numpy as np
    data=np.asarray(samples).squeeze()
    if data.ndim!=1 or not len(data) or not np.isfinite(data).all() or int(sample_rate)<=0:
        raise ValueError('Speech model must produce finite mono samples and a positive sample rate')
    output=io.BytesIO()
    with wave.open(output,'wb') as writer:
        writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(int(sample_rate));writer.writeframes(pcm16(data))
    return output.getvalue()


def write_music_wav(samples,sample_rate):
    """Encode finite channel-preserving PCM16 WAV; never downmix or invent stereo."""
    import numpy as np
    data=np.asarray(samples,dtype=np.float32)
    if data.ndim==1:data=data[:,None]
    elif data.ndim==2 and data.shape[0] in {1,2} and data.shape[1]>data.shape[0]:data=data.T
    if data.ndim!=2 or data.shape[1] not in {1,2} or not data.shape[0] or not np.isfinite(data).all():
        raise ValueError('Music output must contain finite mono or stereo samples')
    rate=int(sample_rate)
    if rate<8000 or rate>192000:raise ValueError('Music sample rate must be 8000..192000 Hz')
    settings=current_settings()
    if (data.size>settings.max_music_samples or data.shape[0]>int(settings.max_audio_seconds*rate)
            or 44+data.size*2>settings.max_music_output_bytes):
        raise ValueError('Music output exceeds configured audio limits')
    pcm=(np.clip(data,-1,1)*32767).round().astype('<i2')
    output=io.BytesIO()
    with wave.open(output,'wb') as writer:
        writer.setnchannels(data.shape[1]);writer.setsampwidth(2);writer.setframerate(rate);writer.writeframes(pcm.tobytes())
    return output.getvalue()


def write_video(frames,path,fps):
    import numpy as np
    import imageio
    settings=current_settings()
    if not len(frames):raise ValueError('Video model produced no frames')
    with imageio.get_writer(str(path),fps=fps,codec=settings.video_codec,quality=None,
                            bitrate=settings.video_bitrate,macro_block_size=1,pixelformat='yuv420p',
                            ffmpeg_log_level='error') as writer:
        for frame in frames:
            array=np.asarray(frame)
            if np.issubdtype(array.dtype,np.floating):
                if not np.isfinite(array).all():raise ValueError('Video contains non-finite pixels')
                array=(np.clip(array,0,1)*255).round().astype(np.uint8)
            writer.append_data(array)
    if not path.is_file() or path.stat().st_size<128:
        raise ValueError('Video encoder produced no usable output; check TITANOS_VIDEO_CODEC')


def _ffmpeg_env():
    """FFmpeg's environment: the server's, without provider credentials (titanos.child_env.scrubbed_env)."""
    import os
    from .child_env import scrubbed_env
    from .native_process import vault_env_vars
    return scrubbed_env(dict(os.environ), extra_deny=vault_env_vars())
