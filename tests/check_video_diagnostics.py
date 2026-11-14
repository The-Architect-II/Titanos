"""Lossless model-frame diagnostics and real HTTP/job boundaries; no model or codec."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from titanos.config import Config, ModelSpec
from titanos.engine import EngineError
from titanos.server import create_app
from titanos.settings import Settings
from titanos.video_diagnostics import VideoDiagnostics


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frames = [np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
                  np.linspace(-.2, 1.2, 48, dtype=np.float32).reshape(4, 4, 3)]
        snapshots = [frame.copy() for frame in frames]
        settings = Settings(video_diagnostics_dir=str(root/'diagnostics'))
        capture = VideoDiagnostics(settings, 'a'*32, 'video', 8, 2)
        capture.capture(frames)
        for index, item in enumerate(capture.record['frames']):
            path = capture.directory/item['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
            restored = np.load(path, allow_pickle=False)
            assert restored.dtype == frames[index].dtype
            np.testing.assert_array_equal(restored, snapshots[index])
            np.testing.assert_array_equal(frames[index], snapshots[index])
        try:
            VideoDiagnostics(settings, 'a'*32, 'video', 8, 2)
            raise AssertionError('Existing diagnostics overwritten')
        except FileExistsError:
            pass
        # Preserve NaN/Inf in original array evidence; the encoder's validation remains authoritative.
        invalid = VideoDiagnostics(settings, 'b'*32, 'video', 8, 1)
        invalid.capture([np.array([[[np.nan, np.inf, -np.inf]]], dtype=np.float32)])
        assert np.isnan(np.load(invalid.directory/'frame-000000.npy', allow_pickle=False)[0, 0, 0])
        # The payload cap includes NPY headers, and never truncates a saved frame to make it fit.
        limit = VideoDiagnostics(Settings(video_diagnostics_dir=str(root/'small'), video_diagnostics_max_bytes=49), 'c'*32, 'video', 8, 1)
        try:
            limit.capture([frames[0]])
            raise AssertionError('NPY header exceeded the cap silently')
        except ValueError as error:
            assert 'MAX_BYTES' in str(error)
        assert not list(limit.directory.glob('*.npy'))
        cancel = threading.Event();cancel.set()
        stopped = VideoDiagnostics(settings, 'd'*32, 'video', 8, 2)
        try:
            stopped.capture(frames, cancel)
            raise AssertionError('Cancelled capture proceeded')
        except EngineError as error:
            assert error.status == 499
        assert not list(stopped.directory.glob('*.npy'))
        cancel.clear()
        class InterruptedFrame:
            def __array__(self, dtype=None, copy=None):
                cancel.set()
                return frames[0]
        interrupted = VideoDiagnostics(settings, 'e'*32, 'video', 8, 2)
        try:
            interrupted.capture([frames[0], InterruptedFrame()], cancel)
            raise AssertionError('Cancellation during array capture was ignored')
        except EngineError as error:
            interrupted.finish('cancelled', error=error)
        assert len(list(interrupted.directory.glob('*.npy'))) == 1
        assert interrupted.record['status']=='cancelled' and not interrupted.record['capture_complete']
        assert Settings.from_env({'TITANOS_VIDEO_DIAGNOSTICS_DIR':str(root/'env'), 'TITANOS_VIDEO_DIAGNOSTICS_MAX_BYTES':'1024'}).video_diagnostics_max_bytes == 1024
        try:
            Settings(video_diagnostics_dir='relative')
            raise AssertionError('Relative diagnostics directory accepted')
        except ValueError:
            pass

        # Fake only model output and encoder: actual server owns diagnostics and publication.
        config = Config(models={'video':ModelSpec(path=str(root), task='video', backend='diffusers', device='cpu')})
        payload = {'model':'video', 'prompt':'fixture', 'size':'64x64', 'frames':2, 'fps':8, 'steps':1}
        codec_calls = []
        def writer(returned, output, fps):
            records = list((root/'http-diagnostics').glob('*/manifest.json'))
            latest = next(record for record in records if record.parent.name == output.stem)
            manifest = json.loads(latest.read_text())
            assert manifest['capture_complete'] and len(manifest['frames']) == 2
            assert all((latest.parent/item['path']).is_file() for item in manifest['frames'])
            codec_calls.append(output)
            output.write_bytes(b'encoded-video-placeholder'*10)
        app = create_app(config, output_dir=root/'http', settings=Settings(video_diagnostics_dir=str(root/'http-diagnostics')))
        app.state.engine.target.infer = lambda *args, **kwargs: frames
        with TestClient(app, base_url='http://127.0.0.1') as client, patch('titanos.media.write_video', writer):
            response = client.post('/v1/videos/generations', json=payload)
            assert response.status_code == 200, response.text
            result = response.json();manifest = json.loads((root/'http-diagnostics'/result['id']/'manifest.json').read_text())
            assert manifest['status'] == 'encoded' and manifest['encoded_video']['bytes'] > 128
            assert not any(key.startswith('diagnostic') for key in result)
        # Failed encoding retains complete raw evidence but never publishes an MP4.
        def failed_writer(returned, output, fps):
            output.write_bytes(b'partial');raise ValueError('fixture codec failure')
        app = create_app(config, output_dir=root/'failed', settings=Settings(video_diagnostics_dir=str(root/'failed-diagnostics')))
        app.state.engine.target.infer = lambda *args, **kwargs: frames
        with TestClient(app, base_url='http://127.0.0.1', raise_server_exceptions=False) as client, patch('titanos.media.write_video', failed_writer):
            assert client.post('/v1/videos/generations', json=payload).status_code == 500
        assert not list((root/'failed').glob('*.mp4'))
        record = json.loads(next((root/'failed-diagnostics').glob('*/manifest.json')).read_text())
        assert record['status'] == 'failed' and record['capture_complete'] and len(record['frames']) == 2
        app = create_app(config, output_dir=root/'limited', settings=Settings(video_diagnostics_dir=str(root/'limited-diagnostics'),video_diagnostics_max_bytes=49))
        app.state.engine.target.infer = lambda *args, **kwargs: frames
        with TestClient(app, base_url='http://127.0.0.1', raise_server_exceptions=False) as client, patch('titanos.media.write_video', side_effect=AssertionError('Over-budget capture reached encoder')):
            assert client.post('/v1/videos/generations', json=payload).status_code == 500
        record=json.loads(next((root/'limited-diagnostics').glob('*/manifest.json')).read_text())
        assert record['status']=='failed' and not record['capture_complete'] and record['frames']==[]
        # A disabled diagnostic mode never instantiates the capture or writes a diagnostic directory.
        app = create_app(config, output_dir=root/'ordinary', settings=Settings())
        app.state.engine.target.infer = lambda *args, **kwargs: frames
        with TestClient(app, base_url='http://127.0.0.1') as client, patch('titanos.media.write_video', lambda frames,path,fps:path.write_bytes(b'video'*40)), patch('titanos.video_diagnostics.VideoDiagnostics', side_effect=AssertionError('Disabled diagnostics executed')):
            assert client.post('/v1/videos/generations', json=payload).status_code == 200
        # Cancel while encoding: retain explicitly cancelled diagnostics; job cleanup removes MP4.
        entered, release = threading.Event(), threading.Event()
        def held_writer(frames, path, fps):
            path.write_bytes(b'video'*40);entered.set();assert release.wait(5)
        app = create_app(config, output_dir=root/'cancelled', settings=Settings(video_diagnostics_dir=str(root/'cancelled-diagnostics')))
        app.state.engine.target.infer = lambda *args, **kwargs: frames
        with TestClient(app, base_url='http://127.0.0.1') as client, patch('titanos.media.write_video', held_writer):
            try:
                submitted = client.post('/api/video/jobs', json=payload);assert submitted.status_code == 202
                identifier = submitted.json()['id'];assert entered.wait(5)
                assert client.delete('/api/jobs/'+identifier).status_code == 200
            finally:
                release.set()
            deadline = time.monotonic()+5
            while time.monotonic()<deadline:
                job = client.get('/api/jobs/'+identifier).json()
                if job['status'] in {'completed','failed','cancelled'}:break
                time.sleep(.01)
            assert job['status']=='cancelled' and 'result' not in job
        assert not list((root/'cancelled').glob('*.mp4'))
        record = json.loads(next((root/'cancelled-diagnostics').glob('*/manifest.json')).read_text())
        assert record['status']=='cancelled' and record['capture_complete']
    print('PASS: lossless pre-encoder arrays, dtype/NaN preservation, NPY byte cap, no overwrite, settings, encoder-before-capture prohibition, disabled path, failed/cancelled retention and publication cleanup. No model or codec execution.')


if __name__ == '__main__':
    main()
