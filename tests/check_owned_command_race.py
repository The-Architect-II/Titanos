"""run_owned's byte budget and the music progress read tolerate a child that renames its files while the parent polls.

The music adapter publishes progress by writing progress.json.tmp and replacing it onto progress.json. run_owned's budget
listed the request directory and then read each file's size, so a .tmp renamed in between failed the whole request with
FileNotFoundError ([WinError 2] on Windows; about 1 in 5 music jobs on 2026-10-21). Covered here: directory_bytes skipping
a file that vanishes or is being deleted between listing and size read (os.stat raising, deterministic) while still
counting a file the child holds open and keeps writing (Windows directory listings lag its size); a real
child publishing progress 1,500 times per job under run_owned with a progress reader, five jobs, no failure; and the music
backend's progress size read returning None for a missing file. No model or GPU; short-lived Python children."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from titanos import owned_command  # noqa: E402
from titanos.music_backend import _file_size  # noqa: E402
from titanos.owned_command import directory_bytes, run_owned  # noqa: E402

CHILD = r'''
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
for step in range(int(sys.argv[2])):
    temporary = root / 'progress.json.tmp'
    temporary.write_text(json.dumps({'stage': 'generate', 'completed': step}), encoding='utf-8')
    for attempt in range(50):
        try:
            os.replace(temporary, root / 'progress.json')
            break
        except PermissionError:
            pass
'''


def check_directory_bytes():
    """A file renamed away or being deleted between the listing and the size read is skipped; other errors surface."""
    real_stat = os.stat
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        (root / 'stdout.log').write_bytes(b'x' * 10)
        (root / 'progress.json').write_bytes(b'x' * 32)
        (root / 'input.png').write_bytes(b'x' * 1000)
        (root / 'sub').mkdir()
        (root / 'progress.json.tmp').write_bytes(b'x' * 5)
        (root / 'result.json.tmp').write_bytes(b'x' * 7)
        assert directory_bytes(root) == 10 + 32 + 5 + 7
        with (root / 'growing.log').open('wb') as growing:  # still open: the size must come from os.stat, not the listing
            growing.write(b'x' * 5000)
            growing.flush()
            assert directory_bytes(root) == 10 + 32 + 5 + 7 + 5000
        (root / 'growing.log').unlink()

        def racing(path, *args, **kwargs):
            name = Path(path).name
            if name == 'progress.json.tmp':
                raise FileNotFoundError(2, 'renamed by the child', os.fspath(path))
            if name == 'result.json.tmp':
                raise PermissionError(13, 'delete pending', os.fspath(path))
            return real_stat(path, *args, **kwargs)
        with patch.object(owned_command.os, 'stat', side_effect=racing):
            assert directory_bytes(root) == 42

        def failing(path, *args, **kwargs):
            if os.fspath(path).endswith('stdout.log'):
                raise OSError(5, 'device error', os.fspath(path))
            return real_stat(path, *args, **kwargs)
        with patch.object(owned_command.os, 'stat', side_effect=failing):
            try:
                directory_bytes(root)
            except OSError as exc:
                assert exc.errno == 5
            else:
                raise AssertionError('an unrelated I/O error was swallowed')


def check_real_race(jobs=5, publications=1500):
    env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'SystemRoot', 'TEMP', 'TMP') if key in os.environ}
    reads = 0
    for _ in range(jobs):
        with tempfile.TemporaryDirectory(prefix='titanos-race-') as name:
            root = Path(name)
            progress = root / 'progress.json'

            def observe():
                nonlocal reads
                if 0 < (_file_size(progress) or 0) <= 65536:
                    try:
                        json.loads(progress.read_text(encoding='utf-8'))
                        reads += 1
                    except (OSError, ValueError):
                        pass
            run_owned([sys.executable, '-c', CHILD, str(root), str(publications)], root, env, lambda: None, 64 * 1024 * 1024,
                      label='Race check', observe=observe)
            assert json.loads(progress.read_text(encoding='utf-8'))['completed'] == publications - 1
    return reads


def check_progress_size():
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / 'progress.json'
        assert _file_size(path) is None
        path.write_text('{}', encoding='utf-8')
        assert _file_size(path) == 2


def main():
    check_directory_bytes()
    check_progress_size()
    reads = check_real_race()
    print(json.dumps({'check': 'owned_command_race', 'result': 'pass', 'jobs': 5, 'publications_per_job': 1500,
                      'progress_reads': reads, 'scope': 'injected stat failures and real short-lived Python children; no model'}))


if __name__ == '__main__':
    main()
