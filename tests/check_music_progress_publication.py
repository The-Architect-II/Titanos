"""Windows-safe progress publication of the ACE adapter (tools/music_runtime/ace_adapter.py).

The trusted parent polls ``progress.json`` with an ordinary ``open()`` while the adapter atomically replaces it. On Windows
that read holds the destination without ``FILE_SHARE_DELETE``, so ``MoveFileEx(REPLACE_EXISTING)`` fails with
``ERROR_ACCESS_DENIED`` (WinError 5) or ``ERROR_SHARING_VIOLATION`` (WinError 32) for the duration of the read; the packaged
G candidate lost a real generation to that window at stage ``loading_dit`` on 27 Sep 2026. These checks hold a reader on the
destination and assert that the adapter's bounded retry publishes anyway, that an exhausted budget or an unrelated error
surfaces unchanged with the temporary file retained, and that failure reporting never masks the primary error.

Scratch files only; no model, device, network, server or child process. The reader-held cases need Windows sharing
semantics and are skipped elsewhere (reported, not failed).
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/music_runtime/ace_adapter.py'
spec = importlib.util.spec_from_file_location('ace_progress_check_subject', RUNNER)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
WINDOWS = os.name == 'nt'


def hold_reader(path, seconds, opened):
    """Mimic the parent's poll: keep the destination open for reading (no FILE_SHARE_DELETE) for a while."""
    with path.open('r', encoding='utf-8') as stream:
        opened.set()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            stream.seek(0)
            stream.read()
            time.sleep(0.005)


class ProgressPublication(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.progress = self.folder / 'progress.json'
        self.progress.write_text(json.dumps({'stage': 'validating_bundle', 'native_cancelled': False}) + '\n', encoding='utf-8')

    def tearDown(self):
        self.temporary.cleanup()

    def test_retry_constants_are_bounded_and_name_the_two_windows_sharing_errors(self):
        self.assertEqual(adapter.REPLACE_RETRY_WINERRORS, frozenset({5, 32}))
        self.assertTrue(1 < adapter.REPLACE_RETRY_ATTEMPTS <= 200)
        self.assertLessEqual(adapter.REPLACE_RETRY_ATTEMPTS * adapter.REPLACE_RETRY_SLEEP_SECONDS, 5.0)

    def test_plain_replace_uses_one_attempt_and_leaves_no_temporary(self):
        adapter.atomic_json(self.progress, {'stage': 'sampling', 'native_cancelled': False})
        self.assertEqual(json.loads(self.progress.read_text(encoding='utf-8'))['stage'], 'sampling')
        self.assertFalse(self.progress.with_name('progress.json.tmp').exists())

    @unittest.skipUnless(WINDOWS, 'Windows destination-handle sharing semantics')
    def test_publishes_while_the_parent_holds_the_destination_open(self):
        opened = threading.Event()
        reader = threading.Thread(target=hold_reader, args=(self.progress, 0.3, opened), daemon=True)
        reader.start()
        self.assertTrue(opened.wait(5))
        attempts = []
        original = adapter.replace_retrying

        def spy(temporary, path):
            used = original(temporary, path)
            attempts.append(used)
            return used

        with patch.object(adapter, 'replace_retrying', spy):
            started = time.monotonic()
            adapter.atomic_json(self.progress, {'stage': 'loading_dit', 'native_cancelled': False})
            elapsed = time.monotonic() - started
        reader.join(5)
        self.assertEqual(json.loads(self.progress.read_text(encoding='utf-8'))['stage'], 'loading_dit')
        self.assertFalse(self.progress.with_name('progress.json.tmp').exists())
        self.assertEqual(len(attempts), 1)
        self.assertGreater(attempts[0], 1, 'the reader-held window must have forced at least one retry')
        self.assertLess(attempts[0], adapter.REPLACE_RETRY_ATTEMPTS)
        self.assertLess(elapsed, adapter.REPLACE_RETRY_ATTEMPTS * adapter.REPLACE_RETRY_SLEEP_SECONDS + 1.0)

    @unittest.skipUnless(WINDOWS, 'Windows destination-handle sharing semantics')
    def test_exhausted_budget_surfaces_the_sharing_error_and_retains_the_temporary(self):
        opened = threading.Event()
        reader = threading.Thread(target=hold_reader, args=(self.progress, 1.5, opened), daemon=True)
        reader.start()
        self.assertTrue(opened.wait(5))
        with patch.object(adapter, 'REPLACE_RETRY_ATTEMPTS', 3), patch.object(adapter, 'REPLACE_RETRY_SLEEP_SECONDS', 0.01):
            with self.assertRaises(PermissionError) as raised:
                adapter.atomic_json(self.progress, {'stage': 'loading_dit', 'native_cancelled': False})
        reader.join(5)
        self.assertIn(raised.exception.winerror, adapter.REPLACE_RETRY_WINERRORS)
        self.assertEqual(json.loads(self.progress.read_text(encoding='utf-8'))['stage'], 'validating_bundle', 'destination unchanged')
        temporary = self.progress.with_name('progress.json.tmp')
        self.assertTrue(temporary.exists(), 'the fully written temporary file is retained for diagnosis')
        self.assertEqual(json.loads(temporary.read_text(encoding='utf-8'))['stage'], 'loading_dit')

    def test_unrelated_permission_errors_surface_immediately_without_retry(self):
        def refuse(self_path, target):
            error = PermissionError(13, 'Permission denied')
            error.winerror = 1  # ERROR_INVALID_FUNCTION: not a sharing condition
            raise error
        with patch.object(adapter.Path, 'replace', refuse), patch.object(adapter.time, 'sleep') as sleep:
            with self.assertRaises(PermissionError):
                adapter.atomic_json(self.progress, {'stage': 'sampling', 'native_cancelled': False})
        sleep.assert_not_called()

    def test_other_oserrors_surface_immediately_without_retry(self):
        with patch.object(adapter.Path, 'replace', side_effect=FileNotFoundError(2, 'gone')), patch.object(adapter.time, 'sleep') as sleep:
            with self.assertRaises(FileNotFoundError):
                adapter.atomic_json(self.progress, {'stage': 'sampling', 'native_cancelled': False})
        sleep.assert_not_called()

    def test_failure_report_publication_errors_do_not_replace_the_primary_error(self):
        result_path = self.folder / 'result.json'
        stages = []
        calls = {'count': 0}

        def flaky_atomic_json(path, value):
            calls['count'] += 1
            error = PermissionError(13, 'Access is denied')
            error.winerror = 5
            raise error

        stderr = io.StringIO()
        with patch.object(adapter, 'atomic_json', flaky_atomic_json), patch.object(adapter.sys, 'stderr', stderr):
            adapter.report_failure(result_path, lambda stage: stages.append(stage) or adapter.atomic_json(self.progress, {}), {'evidence': {}})
        self.assertEqual(calls['count'], 2, 'result and progress publication were both attempted')
        self.assertEqual(stages, ['failed'])
        self.assertEqual(stderr.getvalue().count('failure report not published'), 2)

    def test_main_reraises_the_primary_error_when_the_failure_report_cannot_be_published(self):
        request = self.folder / 'request.json'
        request.write_text('{}\n', encoding='utf-8')
        output, result, progress = (self.folder / name for name in ('output.wav', 'result.json', 'progress.json'))
        progress.unlink()
        argv = ['ace_adapter', '--request', str(request), '--output', str(output), '--result', str(result), '--progress', str(progress)]

        def refuse_publication(path, value):
            error = PermissionError(13, 'Access is denied')
            error.winerror = 5
            raise error

        stderr = io.StringIO()
        with patch.object(adapter.sys, 'argv', argv), patch.object(adapter, 'validate_request', side_effect=ValueError('primary generation refusal')), \
                patch.object(adapter, 'atomic_json', refuse_publication), patch.object(adapter.sys, 'stderr', stderr):
            with self.assertRaises(ValueError) as raised:
                adapter.main()
        self.assertEqual(str(raised.exception), 'primary generation refusal')
        self.assertIn('failure report not published', stderr.getvalue())


if __name__ == '__main__':
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProgressPublication))
    sys.stderr.write(stream.getvalue())
    print(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                      'skipped': len(result.skipped), 'windows_reader_held_cases_ran': WINDOWS,
                      'scope': 'scratch files and one reader thread; no model, device, network, server or child process'}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
