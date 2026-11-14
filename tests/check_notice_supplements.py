"""Build-time attribution supplements: retained license/attribution files are hash-checked and copied into the bundle, the
upstream SBOM member is nested under the shipped component, DEPENDENCIES fields are returned; a tampered catalog is refused.
Fake destination and dist-info; no build, model or network."""
import argparse
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('titanos_build_binary', ROOT / 'tools/build_binary.py')
bb = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ['build_binary.py']
spec.loader.exec_module(bb)
sys.argv = _argv
EVIDENCE = None
CATALOG = json.loads((ROOT / 'tools/licenses/supplements.json').read_text(encoding='utf-8'))


class Checker(unittest.TestCase):
    def fake_bundle(self, tmp, with_sbom=True):
        dest = Path(tmp) / 'bundle'
        dist = dest / '_internal/onnx-1.22.0.dist-info'
        (dist / 'sboms').mkdir(parents=True)
        if with_sbom:
            (dist / 'sboms/sbom.cdx.json').write_text(json.dumps({'components': [{'type': 'library', 'name': 'protobuf', 'version': '25.1', 'bom-ref': 'pkg:generic/protobuf@25.1'}]}), encoding='utf-8')
        return dest, dist

    def test_onnx_supplement_copies_files_and_nests_upstream_sbom(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, dist = self.fake_bundle(tmp)
            files, component, notes = [{'path': 'THIRD_PARTY_LICENSES/onnx/1-LICENSE', 'sha256': 'x'}], {'type': 'library', 'name': 'onnx', 'version': '1.22.0', 'purl': 'pkg:pypi/onnx@1.22.0'}, []
            extra = bb.apply_attribution_supplements(dest, 'onnx', '1.22.0', dist, files, component, notes)
            lic = dest / 'THIRD_PARTY_LICENSES/onnx/3-protobuf-v25.1-LICENSE.txt'
            self.assertTrue(lic.is_file())
            self.assertEqual(hashlib.sha256(lic.read_bytes()).hexdigest(), CATALOG[0]['license_files'][0]['sha256'])
            self.assertEqual(files[-1]['path'], 'THIRD_PARTY_LICENSES/onnx/3-protobuf-v25.1-LICENSE.txt')
            self.assertTrue((dest / 'THIRD_PARTY_SOURCES/onnx/ATTRIBUTION.json').is_file())
            self.assertTrue((dest / 'THIRD_PARTY_SOURCES/onnx/upstream-sbom.cdx.json').is_file())
            self.assertEqual(extra['source_attribution'], 'THIRD_PARTY_SOURCES/onnx/ATTRIBUTION.json')
            self.assertEqual(extra['upstream_sbom'], 'THIRD_PARTY_SOURCES/onnx/upstream-sbom.cdx.json')
            self.assertIn('attribution_note', extra)
            self.assertEqual(component['components'][0]['bom-ref'], 'onnx-vendored:pkg:generic/protobuf@25.1')
            self.assertIn({'name': 'titanos:source-attribution', 'value': 'THIRD_PARTY_SOURCES/onnx/ATTRIBUTION.json'}, component['properties'])
            self.assertTrue(notes and notes[0].startswith('## ONNX attribution supplement'))
            self.assertNotIn('titanos:component-role', json.dumps(component))  # stays a pypi-classified component for the identity gate

    def test_missing_dist_info_member_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, dist = self.fake_bundle(tmp, with_sbom=False)
            component, files = {'type': 'library', 'name': 'onnx', 'version': '1.22.0'}, []
            extra = bb.apply_attribution_supplements(dest, 'onnx', '1.22.0', dist, files, component, [])
            self.assertNotIn('upstream_sbom', extra)
            self.assertNotIn('components', component)
            self.assertEqual(len(files), 1)

    def test_other_distributions_are_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, dist = self.fake_bundle(tmp)
            files, component, notes = [], {'type': 'library', 'name': 'numpy', 'version': '1.26.4'}, []
            self.assertEqual(bb.apply_attribution_supplements(dest, 'numpy', '1.26.4', None, files, component, notes), {})
            self.assertEqual((files, notes), ([], []))
            self.assertFalse((dest / 'THIRD_PARTY_LICENSES').exists())

    def test_tampered_catalog_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest, dist = self.fake_bundle(tmp)
            bad = json.loads(json.dumps(CATALOG))
            bad[0]['license_files'][0]['sha256'] = '0' * 64
            catalog = Path(tmp) / 'supplements.json'
            catalog.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaises(ValueError):
                bb.apply_attribution_supplements(dest, 'onnx', '1.22.0', dist, [], {'name': 'onnx'}, [], catalog=catalog)

    def test_retained_files_match_catalog(self):
        for row in CATALOG:
            for item in row.get('license_files', []) + row.get('attribution_files', []):
                path = ROOT / 'tools/licenses' / item['file']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'], item['file'])


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
              'source_sha256': hashlib.sha256((ROOT / 'tools/build_binary.py').read_bytes()).hexdigest(), 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Fake bundle and dist-info; no build, model or network'}
    if EVIDENCE:
        (EVIDENCE / 'run.log').write_text(log.getvalue(), encoding='utf-8')
        (EVIDENCE / 'RESULT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
