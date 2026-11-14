"""Engine payload equals its declared closure: modules of other installed distributions are excluded.

Pure metadata doubles; no build, PyInstaller, model or network.
"""
import argparse
import ast
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/build_binary.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
FUNCTION = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'engine_excludes')
namespace = {}
exec('from packaging.utils import canonicalize_name\nimport importlib.metadata as metadata', namespace)
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), 'exec'), namespace)
engine_excludes = namespace['engine_excludes']
EVIDENCE = None

MAPPING = {'torch': ['torch'], 'numpy': ['numpy'], 'onnxruntime': ['onnxruntime'], 'google': ['protobuf'], 'psutil': ['psutil'],
           'win32api': ['pywin32'], 'pythoncom': ['pywin32'], 'titanos': ['titanos-engine'], 'fastapi': ['fastapi'],
           'Cryptography_Vendored': ['cryptography'], 'shared_tests': ['torch', 'fastapi'], '': ['weird']}
CLOSURE = {'titanos-engine', 'fastapi', 'PyWin32', 'cryptography'}


class Checker(unittest.TestCase):
    def test_only_modules_outside_the_closure_are_excluded(self):
        self.assertEqual(engine_excludes(CLOSURE, MAPPING), ['google', 'numpy', 'onnxruntime', 'psutil', 'torch'])

    def test_closure_names_are_matched_canonically_and_shared_modules_are_kept(self):
        excluded = engine_excludes({'pywin32', 'titanos_engine', 'FastAPI', 'cryptography'}, MAPPING)
        self.assertNotIn('win32api', excluded)
        self.assertNotIn('titanos', excluded)
        self.assertNotIn('shared_tests', excluded)  # one of its distributions is in the closure
        self.assertNotIn('', excluded)

    def test_empty_environment_or_full_closure_excludes_nothing(self):
        self.assertEqual(engine_excludes(CLOSURE, {}), [])
        self.assertEqual(engine_excludes({d for ds in MAPPING.values() for d in ds}, MAPPING), [])

    def test_plan_records_excludes_for_engine_only_and_spec_consumes_them(self):
        source = SOURCE.read_text(encoding='utf-8')
        self.assertIn("excluded = engine_excludes(installed) if profile == 'engine' else []", source)
        self.assertIn("'exclude_modules': excluded", source)
        spec = (ROOT / 'tools/titanos.spec').read_text(encoding='utf-8')
        self.assertIn("settings.get('exclude_modules', [])", spec)
        self.assertIn("if settings['profile'] == 'engine':", spec)
        self.assertIn("analysis.datas = [entry for entry in analysis.datas if in_closure(entry)]", spec)

    def test_spec_metadata_filter_keeps_closure_and_drops_foreign_dist_info(self):
        spec = (ROOT / 'tools/titanos.spec').read_text(encoding='utf-8')
        body = spec[spec.index("    from packaging.utils import canonicalize_name"):spec.index("    analysis.datas = [")]
        namespace = {'settings': {'distributions': ['titanos-engine', 'fastapi', 'PyWin32']}}
        exec('\n'.join(line[4:] for line in body.splitlines()), namespace)
        keep = namespace['in_closure']
        self.assertTrue(keep(('titanos_engine-0.5.28.dev0+accelerators1.dist-info\\METADATA', 'x', 'DATA')))
        self.assertTrue(keep(('pywin32-311.dist-info/RECORD', 'x', 'DATA')))
        self.assertTrue(keep(('titanos/web/index.html', 'x', 'DATA')))
        self.assertFalse(keep(('openvino-2026.3.1.dist-info\\METADATA', 'x', 'DATA')))
        self.assertFalse(keep(('torch-2.8.0+cu128.dist-info/METADATA', 'x', 'DATA')))


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
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'engine_excludes with metadata doubles plus plan/spec wiring; no build or PyInstaller execution'}
    if EVIDENCE:
        (EVIDENCE / 'run.log').write_text(log.getvalue(), encoding='utf-8')
        (EVIDENCE / 'RESULT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
