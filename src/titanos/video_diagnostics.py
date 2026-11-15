"""Opt-in, private model-frame evidence before the video encoder is opened."""
import hashlib
import io
import json
from pathlib import Path
import re
import time

from .text_inputs import check_cancel


class VideoDiagnostics:
    def __init__(self, settings, identifier, model, fps, frame_count):
        if not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('Video diagnostics require a generated artifact identifier')
        self.directory = Path(settings.video_diagnostics_dir).expanduser().resolve() / identifier
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self.directory.mkdir(mode=0o700, exist_ok=False)
        self.limit = settings.video_diagnostics_max_bytes
        self.record = {'schema': 'titanos.video-frames/v1', 'video_id': identifier,
                       'model': model, 'fps': fps, 'expected_frames': frame_count,
                       'codec': settings.video_codec, 'bitrate': settings.video_bitrate,
                       'boundary': 'Model-returned frames before media.write_video normalization or encoding',
                       'format': 'NumPy NPY; load with allow_pickle=False',
                       'retention': 'Private diagnostic files remain until the operator removes them, including failures and cancellations',
                       'status': 'capturing', 'capture_complete': False,
                       'max_frame_file_bytes': self.limit, 'frame_file_bytes': 0,
                       'started_at': time.time(), 'frames': []}
        self._save()

    def _save(self):
        raw = (json.dumps(self.record, indent=2, allow_nan=False) + '\n').encode()
        temporary = self.directory / 'manifest.json.tmp'
        temporary.write_bytes(raw)
        temporary.replace(self.directory / 'manifest.json')

    def capture(self, frames, cancel=None):
        import numpy as np
        for index, frame in enumerate(frames):
            check_cancel(cancel)
            array = np.asarray(frame)
            if array.dtype.hasobject:
                raise ValueError('Video diagnostics do not serialize object arrays')
            if array.nbytes > self.limit - self.record['frame_file_bytes']:
                raise ValueError('Model frames exceed TITANOS_VIDEO_DIAGNOSTICS_MAX_BYTES')
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            raw = buffer.getvalue()
            if len(raw) > self.limit - self.record['frame_file_bytes']:
                raise ValueError('Model frames exceed TITANOS_VIDEO_DIAGNOSTICS_MAX_BYTES')
            check_cancel(cancel)
            name = f'frame-{index:06d}.npy'
            with (self.directory / name).open('xb') as output:
                output.write(raw)
            self.record['frame_file_bytes'] += len(raw)
            self.record['frames'].append({'index': index, 'path': name, 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest(), 'dtype': array.dtype.str,
                'shape': list(array.shape), 'source_type': type(frame).__name__})
            self._save()
            check_cancel(cancel)
        if len(self.record['frames']) != self.record['expected_frames']:
            raise ValueError('Video frame count changed during diagnostics capture')
        self.record.update(status='frames_captured', capture_complete=True)
        self._save()

    def finish(self, status, error=None, video=None):
        self.record.update(status=status, finished_at=time.time())
        if error is not None:
            self.record['error'] = {'type': type(error).__name__, 'message': str(error)[:1000]}
        if video is not None:
            with Path(video).open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            self.record['encoded_video'] = {'name': Path(video).name,
                'bytes': Path(video).stat().st_size, 'sha256': digest}
        self._save()
