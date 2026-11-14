"""An English `language` hint to an English-only Whisper checkpoint is satisfied without being forwarded (the library
would reject it); a non-English hint to such a checkpoint is refused with a clear message; multilingual checkpoints still
receive the hint. Fake pipelines, no model, no audio decoding, no server."""
import argparse
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos import asr  # noqa: E402

EVIDENCE = None


class FakeWhisper:
    type = 'seq2seq_whisper'

    def __init__(self, multilingual):
        self.model = SimpleNamespace(generation_config=SimpleNamespace(is_multilingual=multilingual), config=SimpleNamespace(model_type='whisper'))
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)
        self.calls = []

    def __call__(self, inputs, **kw):
        self.calls.append(kw)
        return {'text': ' hello world', 'chunks': [{'text': ' hello world', 'timestamp': (0.0, 1.0), 'language': 'english'}]}


SPEC = SimpleNamespace(defaults={}, backend='transformers')


def run(pipeline, data):
    with mock.patch.object(asr, 'read_audio', return_value=np.zeros(16000, dtype=np.float32)):
        return asr.transcribe(pipeline, SPEC, {'audio': b'fake', **data})


class Checker(unittest.TestCase):
    def test_english_hint_is_satisfied_not_forwarded_on_english_only_checkpoint(self):
        for hint in ('en', 'English', 'en-US', 'en_GB', '<|en|>', ' eng '):
            p = FakeWhisper(multilingual=False)
            self.assertEqual(run(p, {'language': hint}), ' hello world', hint)
            self.assertNotIn('language', p.calls[-1].get('generate_kwargs', {}), hint)

    def test_non_english_hint_on_english_only_checkpoint_is_refused_clearly(self):
        p = FakeWhisper(multilingual=False)
        with self.assertRaises(ValueError) as ctx:
            run(p, {'language': 'es'})
        self.assertIn('English-only Whisper checkpoint', str(ctx.exception))
        self.assertIn("'es'", str(ctx.exception))
        self.assertEqual(p.calls, [])

    def test_multilingual_checkpoint_still_receives_the_hint(self):
        p = FakeWhisper(multilingual=True)
        run(p, {'language': 'es'})
        self.assertEqual(p.calls[-1]['generate_kwargs']['language'], 'es')
        p = FakeWhisper(multilingual=True)
        run(p, {'language': 'en'})
        self.assertEqual(p.calls[-1]['generate_kwargs']['language'], 'en')

    def test_no_hint_is_unchanged(self):
        for multilingual in (False, True):
            p = FakeWhisper(multilingual=multilingual)
            run(p, {})
            self.assertNotIn('generate_kwargs', p.calls[-1])

    def test_verbose_result_still_records_the_requested_hint(self):
        p = FakeWhisper(multilingual=False)
        result = run(p, {'language': 'en', 'response_format': 'verbose_json'})
        self.assertEqual(result['requested_language'], 'en')
        self.assertEqual(result['language'], 'english')
        self.assertEqual(result['text'], ' hello world')

    def test_unknown_pipeline_shape_forwards_like_before(self):
        p = FakeWhisper(multilingual=None)
        p.model = SimpleNamespace(config=SimpleNamespace(model_type='wav2vec2'))
        run(p, {'language': 'fr'})
        self.assertEqual(p.calls[-1]['generate_kwargs']['language'], 'fr')


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
              'source_sha256': hashlib.sha256((ROOT / 'src/titanos/asr.py').read_bytes()).hexdigest(), 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Fake Whisper pipelines; no model, audio decoding, server or network'}
    if EVIDENCE:
        (EVIDENCE / 'run.log').write_text(log.getvalue(), encoding='utf-8')
        (EVIDENCE / 'RESULT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
