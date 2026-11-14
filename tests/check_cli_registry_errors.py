"""Local CLI registry error messages are actionable: no listener, no model load, no network.

Runs the real `titanos.cli.main` in-process against a private TITANOS_HOME and checks that option-file and
device-runtime failures name the option, file or missing runtime profile, and that every refused command leaves
the registry bytes unchanged.
"""
import argparse
import builtins
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos import cli  # noqa: E402

EVIDENCE = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Checker(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='titanos-cli-errors-', dir=EVIDENCE)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.home = self.base / 'home'
        self.model = self.base / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}', encoding='utf-8')
        self.environment = patch.dict(os.environ, {'TITANOS_HOME': str(self.home)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.invoke(['init'])

    def invoke(self, argv):
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                cli.main(argv)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
        return code, out.getvalue(), err.getvalue()

    def registry(self):
        return self.home / 'models.json'

    def refused(self, argv, *fragments):
        before = digest(self.registry())
        code, out, err = self.invoke(argv)
        self.assertEqual(code, 1, (argv, err))
        self.assertNotIn('Traceback', err)
        for fragment in fragments:
            self.assertIn(fragment, err, err)
        self.assertEqual(digest(self.registry()), before, 'refused command changed the registry')
        return err

    def test_malformed_defaults_file_names_option_and_file(self):
        bad = self.base / 'defaults.json'
        bad.write_text('{"max_tokens": ', encoding='utf-8')
        err = self.refused(['add', 'm', str(self.model), '--task', 'embeddings', '--device', 'cpu', '--defaults', str(bad)],
                           '--defaults must be a JSON file', 'defaults.json')
        self.assertIn('Expecting value', err)

    def test_missing_native_options_file_names_option_and_file(self):
        missing = self.base / 'absent.json'
        self.refused(['add', 'm', str(self.model), '--task', 'chat', '--backend', 'llama-cpp', '--device', 'cpu',
                      '--model-file', 'x.gguf', '--native-options', str(missing)], '--native-options file could not be read', 'absent.json')

    def test_provider_options_and_voices_are_named_too(self):
        bad = self.base / 'bad.json'
        bad.write_text('[', encoding='utf-8')
        self.refused(['add', 'm', str(self.model), '--task', 'embeddings', '--device', 'cpu', '--provider-options', str(bad)], '--provider-options must be a JSON file')
        self.refused(['add', 'm', str(self.model), '--task', 'speech', '--device', 'cpu', '--voices', str(bad)], '--voices must be a JSON file')

    def test_cuda_and_xpu_without_torch_explain_the_runtime_profile(self):
        real_import = builtins.__import__

        def no_torch(name, *args, **kwargs):
            if name == 'torch' or name.startswith('torch.'):
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, '__import__', side_effect=no_torch):
            for device in ('cuda', 'cuda:1', 'xpu'):
                with self.subTest(device=device):
                    err = self.refused(['add', 'm', str(self.model), '--task', 'embeddings', '--device', device],
                                       'needs the PyTorch runtime profile', repr(device), '--device cpu')
                    self.assertNotIn("titanos: No module named 'torch'", err)

    def test_valid_registration_and_removal_still_work(self):
        defaults = self.base / 'ok.json'
        defaults.write_text(json.dumps({'truncate': True}), encoding='utf-8')
        code, out, err = self.invoke(['add', 'm', str(self.model), '--task', 'embeddings', '--device', 'cpu', '--defaults', str(defaults)])
        self.assertEqual((code, err), (0, ''), err)
        saved = json.loads(self.registry().read_text(encoding='utf-8'))
        self.assertEqual(saved['models']['m']['defaults'], {'truncate': True})
        self.assertEqual(saved['models']['m']['path'], str(self.model.resolve()))
        code, out, err = self.invoke(['remove', 'm'])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(self.registry().read_text(encoding='utf-8'))['models'], {})
        self.assertTrue((self.model / 'config.json').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.evidence:
        EVIDENCE = args.evidence.resolve()
        EVIDENCE.mkdir(exist_ok=False)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checker))
    print(log.getvalue(), end='')
    record = {'passed': result.wasSuccessful(), 'cases': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'source_sha256': digest(ROOT / 'src/titanos/cli.py'), 'checker_sha256': digest(Path(__file__)),
              'scope': 'In-process CLI registry error contracts with a private registry; no listener, model load, GPU or network'}
    if EVIDENCE:
        (EVIDENCE / 'run.log').write_text(log.getvalue(), encoding='utf-8')
        (EVIDENCE / 'RESULT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
