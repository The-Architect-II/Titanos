"""Worker defaults and CLI persistence without model or native process creation."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from pydantic import ValidationError
from titanos.cli import main
from titanos.config import Config, ModelSpec, WorkerSpec
from titanos.settings import Settings, _active_settings, load_settings
from titanos.replicas import Worker
from titanos.engine import EngineError


class Tests(unittest.TestCase):
    def test_fractional_deadline_rejects_late_ready_or_received_results(self):
        for poll_seconds, receive_seconds, accepted in [(.08, 0, False), (.004, .08, False), (.004, .004, True)]:
            clock = [0.0]; waits = []; receives = []; stops = []
            worker = Worker('w', WorkerSpec(models=['model'], timeout=.01), Config(models={'model': ModelSpec(path='unused', task='chat')}))
            worker.process = SimpleNamespace(is_alive=lambda: True, pid=123)
            worker.start = lambda: None
            worker.stop = lambda force=False: stops.append(force)
            worker.cancel = SimpleNamespace(clear=lambda: None, set=lambda: None)
            def poll(seconds):
                waits.append(seconds); clock[0] += poll_seconds; return True
            def receive():
                receives.append(True); clock[0] += receive_seconds
                return 'done', ({'answer': 'fixture'}, {'current_model': 'model'})
            worker.connection = SimpleNamespace(send=lambda message: None, poll=poll, recv=receive)
            with patch('titanos.replicas.time.monotonic', side_effect=lambda: clock[0]):
                if accepted:
                    self.assertEqual(worker.call('infer', 'model', 'chat', {}), {'answer': 'fixture'})
                else:
                    with self.assertRaises(EngineError): worker.call('infer', 'model', 'chat', {})
            self.assertEqual(waits, [.01])
            self.assertEqual(stops, [] if accepted else [True])
            if poll_seconds > .01: self.assertFalse(receives)

    def test_defaults_environment_and_explicit_precedence(self):
        with patch.dict(os.environ, {'TITANOS_WORKER_THREADS': '300', 'TITANOS_WORKER_TIMEOUT': '.25'}):
            self.assertEqual((load_settings().worker_threads, load_settings().worker_timeout), (300, .25))
            default = WorkerSpec(models=['model'])
            explicit = WorkerSpec(models=['model'], threads=2, timeout=172800)
            self.assertEqual((default.threads, default.timeout), (300, .25))
            self.assertEqual((explicit.threads, explicit.timeout), (2, 172800))

    def test_active_application_settings_and_saved_values(self):
        token = _active_settings.set(Settings(worker_threads=12, worker_timeout=1.5))
        try:
            with patch.dict(os.environ, {'TITANOS_WORKER_THREADS': '99', 'TITANOS_WORKER_TIMEOUT': '999'}):
                worker = WorkerSpec(models=['model'])
                self.assertEqual((worker.threads, worker.timeout), (12, 1.5))
                saved = worker.model_dump()
        finally:
            _active_settings.reset(token)
        with patch.dict(os.environ, {'TITANOS_WORKER_THREADS': '8', 'TITANOS_WORKER_TIMEOUT': '60'}):
            restored = WorkerSpec.model_validate(saved)
            self.assertEqual((restored.threads, restored.timeout), (12, 1.5))

    def test_invalid_values_rejected(self):
        for value in [0, -1, True, '4', 4.5]:
            with self.subTest(threads=value), self.assertRaises(ValidationError):
                WorkerSpec(models=['model'], threads=value)
        for value in [0, -1, True, '1', float('nan'), float('inf')]:
            with self.subTest(timeout=value), self.assertRaises(ValidationError):
                WorkerSpec(models=['model'], timeout=value)
        for name, value in [('TITANOS_WORKER_THREADS', '0'), ('TITANOS_WORKER_THREADS', '1.2'), ('TITANOS_WORKER_TIMEOUT', 'nan'), ('TITANOS_WORKER_TIMEOUT', '0')]:
            with patch.dict(os.environ, {name: value}), self.assertRaises(ValueError): load_settings()

    def test_cli_omission_and_explicit_options_persist_effective_values(self):
        with tempfile.TemporaryDirectory(prefix='titanos-worker-settings-') as directory:
            path = Path(directory)/'models.json'
            config = Config(models={'model': ModelSpec(path=directory, task='chat')})
            path.write_text(config.model_dump_json(), encoding='utf-8')
            with patch.dict(os.environ, {'TITANOS_WORKER_THREADS': '300', 'TITANOS_WORKER_TIMEOUT': '.25'}), contextlib.redirect_stdout(io.StringIO()):
                main(['--config', str(path), 'worker', 'w', '--models', 'model'])
            worker = json.loads(path.read_text(encoding='utf-8'))['workers']['w']
            self.assertEqual((worker['threads'], worker['timeout']), (300, .25))
            with patch.dict(os.environ, {'TITANOS_WORKER_THREADS': '300', 'TITANOS_WORKER_TIMEOUT': '.25'}), contextlib.redirect_stdout(io.StringIO()):
                main(['--config', str(path), 'worker', 'w', '--models', 'model', '--threads', '1', '--timeout', '172800', '--replace'])
            worker = json.loads(path.read_text(encoding='utf-8'))['workers']['w']
            self.assertEqual((worker['threads'], worker['timeout']), (1, 172800))


if __name__ == '__main__': unittest.main()
