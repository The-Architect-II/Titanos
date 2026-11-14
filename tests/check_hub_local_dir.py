"""`titanos pull` downloads into a plain per-revision directory (no Hugging Face cache symlinks) and registers it, so a
stock Windows account without symlink privilege can pull; --cache-dir moves the base, TITANOS_HOME/models is the default.
Mocked hub, no network."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos import hub  # noqa: E402
from titanos.cli import main  # noqa: E402
from titanos.config import read_config  # noqa: E402

EVIDENCE = None
SHA = 'b' * 40


def info(files=('model.safetensors', 'config.json')):
    return SimpleNamespace(id='owner/model', pipeline_tag='text-generation', library_name='transformers', downloads=1, likes=0, gated=False, sha=SHA,
                           siblings=[SimpleNamespace(rfilename=f) for f in files])


class Checker(unittest.TestCase):
    def pull(self, tmp, extra, files=('model.safetensors', 'config.json'), env=None):
        with patch.object(hub, 'HfApi') as api, patch.object(hub, 'snapshot_download') as download, patch.object(hub, 'get_token', return_value=None), \
                patch.dict(os.environ, env or {}, clear=False):
            api.return_value.model_info.return_value = info(files)
            cfg = Path(tmp) / 'models.json'
            with contextlib.redirect_stdout(io.StringIO()):
                main(['--config', str(cfg), 'pull', 'm', 'owner/model', '--task', 'chat', *extra])
            return download.call_args.kwargs, read_config(cfg).models['m']

    def test_default_is_plain_directory_under_home_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs, spec = self.pull(tmp, [], env={'TITANOS_HOME': tmp})
            expected = Path(tmp).resolve() / 'models' / ('owner--model@' + SHA[:12])
            self.assertEqual(Path(kwargs['local_dir']), expected)
            self.assertNotIn('cache_dir', kwargs)
            self.assertEqual(Path(spec.path), expected)
            self.assertEqual(spec.source_revision, SHA)
            self.assertTrue(expected.parent.is_dir())

    def test_cache_dir_moves_the_base_and_never_uses_cache_symlink_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'elsewhere'
            kwargs, spec = self.pull(tmp, ['--cache-dir', str(base)])
            self.assertEqual(Path(kwargs['local_dir']), base.resolve() / ('owner--model@' + SHA[:12]))
            self.assertNotIn('cache_dir', kwargs)
            self.assertEqual(kwargs['revision'], SHA)
            self.assertEqual(Path(spec.path), base.resolve() / ('owner--model@' + SHA[:12]))

    def test_every_backend_branch_uses_the_plain_directory(self):
        cases = [(['--backend', 'onnx', '--task', 'embeddings'], ('model.onnx', 'config.json')),
                 (['--backend', 'llama-cpp', '--task', 'chat', '--model-file', 'q.gguf'], ('q.gguf', 'config.json')),
                 (['--backend', 'transformers', '--task', 'chat'], ('model.safetensors',))]
        for extra, files in cases:
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(hub, 'HfApi') as api, patch.object(hub, 'snapshot_download') as download, patch.object(hub, 'get_token', return_value=None), \
                        patch.dict(os.environ, {'TITANOS_HOME': tmp}):
                    api.return_value.model_info.return_value = info(files)
                    with contextlib.redirect_stdout(io.StringIO()):
                        main(['--config', str(Path(tmp) / 'models.json'), 'pull', 'm', 'owner/model', *extra])
                    self.assertIn('local_dir', download.call_args.kwargs, extra)
                    self.assertNotIn('cache_dir', download.call_args.kwargs, extra)

    def test_selection_patterns_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The default pull excludes GGUF and pickle weights by naming only the kept siblings (literal patterns).
            kwargs, _ = self.pull(tmp, [], files=('model.safetensors', 'config.json', 'weights.gguf', 'pytorch_model.bin'), env={'TITANOS_HOME': tmp})
            self.assertEqual(kwargs['allow_patterns'], ['config.json', 'model.safetensors'])
            self.assertNotIn('ignore_patterns', kwargs)


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
              'source_sha256': hashlib.sha256((ROOT / 'src/titanos/hub.py').read_bytes()).hexdigest(), 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Mocked hub download; no network, model or symlink creation'}
    if EVIDENCE:
        (EVIDENCE / 'run.log').write_text(log.getvalue(), encoding='utf-8')
        (EVIDENCE / 'RESULT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
