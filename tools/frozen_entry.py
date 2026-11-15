"""Executable entry: handle spawned workers and owned helpers before CLI parsing."""
import multiprocessing
import json
import os
from pathlib import Path
import sys


def main():
    multiprocessing.freeze_support()
    ffmpeg = Path(getattr(sys, '_MEIPASS', '')) / 'native' / 'ffmpeg' / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
    if getattr(sys, 'frozen', False) and ffmpeg.is_file():
        os.environ.setdefault('IMAGEIO_FFMPEG_EXE', str(ffmpeg))
        native = json.loads((ffmpeg.parent / 'BUILD.json').read_text(encoding='utf-8'))
        if native.get('video_codec'):
            os.environ.setdefault('TITANOS_VIDEO_CODEC', native['video_codec'])
    if sys.argv[1:2] == ['--titanos-internal']:
        if len(sys.argv) < 3:
            raise SystemExit('Missing Titanos helper module')
        module = sys.argv[2]
        sys.argv = [module, *sys.argv[3:]]
        if module == 'titanos.device_probe':
            from titanos.device_probe import main as helper
        elif module == 'titanos.native_process':
            from titanos.native_process import main as helper
        else:
            raise SystemExit('Unknown Titanos helper module')
        return helper()
    from titanos.cli import main as cli
    return cli()


if __name__ == '__main__':
    sys.exit(main())
