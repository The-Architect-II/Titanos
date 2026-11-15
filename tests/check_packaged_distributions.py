"""Shipped distribution identities/notices over inert files; no runtime imports."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/build_binary.py'
EVIDENCE = None
spec = importlib.util.spec_from_file_location('packaged_distribution_build_contract', SOURCE)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

# The two actual rows from the sealed speech package SBOM (97 shipped distributions): the PyPI wheel and the
# separately attributed native library share the name sherpa-onnx and the version 1.13.8.
SHERPA_PYTHON = {'type': 'library', 'name': 'sherpa-onnx', 'version': '1.13.8',
                 'purl': 'pkg:pypi/sherpa-onnx@1.13.8', 'scope': 'required'}
SHERPA_NATIVE = {'type': 'library', 'name': 'sherpa-onnx', 'version': '1.13.8',
                 'licenses': [{'expression': 'Apache-2.0'}],
                 'properties': [{'name': 'titanos:component-role', 'value': 'optional-sherpa-native'},
                                {'name': 'titanos:source-bundle', 'value': 'THIRD_PARTY_SOURCES/sherpa-onnx-1.13.8'},
                                {'name': 'titanos:linkage', 'value': 'native module/library'}],
                 'externalReferences': [{'type': 'vcs',
                                         'url': 'https://github.com/k2-fsa/sherpa-onnx/tree/11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf'}]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


class PackagedDistributionChecks(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(build, 'validate_packaged_distributions', None)),
                        'Missing validate_packaged_distributions shipping gate')
        self.assertTrue(callable(getattr(build, 'reconcile_packaged_distributions', None)),
                        'Missing reconcile_packaged_distributions collection gate')
        self.temporary = tempfile.TemporaryDirectory(prefix='packaged-distributions-', dir=EVIDENCE or ROOT.parent)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.bundle = self.base / 'bundle'
        (self.bundle / '_internal').mkdir(parents=True)
        self.dependencies = []
        self.components = []
        self.external_runtimes = []

    def inventories(self):
        write_json(self.bundle / 'DEPENDENCIES.json', {'components': self.dependencies,
                                                     'external_runtimes': self.external_runtimes})
        write_json(self.bundle / 'sbom.cdx.json', {'bomFormat': 'CycloneDX', 'specVersion': '1.6',
                                                'components': self.components})

    def distribution(self, name='onnx', version='1.22.0', folder=None, root=None):
        folder = folder or name.replace('-', '_') + '-' + version + '.dist-info'
        directory = (root or self.bundle) / '_internal' / folder
        directory.mkdir(parents=True)
        (directory / 'METADATA').write_text(
            f'Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n'
            'License-Expression: Apache-2.0\nLicense-File: licenses/LICENSE\n\nInert fixture.\n', encoding='utf-8')
        (directory / 'WHEEL').write_text('Wheel-Version: 1.0\nTag: py3-none-any\n', encoding='utf-8')
        (directory / 'licenses').mkdir()
        (directory / 'licenses/LICENSE').write_text('Inert fixture license text.\n', encoding='utf-8')
        (directory / 'RECORD').write_text(
            '\n'.join(f'{folder}/{name},,' for name in ['METADATA', 'WHEEL', 'licenses/LICENSE', 'RECORD']) + '\n',
            encoding='utf-8')
        return directory

    def dependency_row(self, name='onnx', version='1.22.0'):
        notice = self.bundle / 'THIRD_PARTY_LICENSES' / name / 'LICENSE'
        notice.parent.mkdir(parents=True, exist_ok=True)
        notice.write_text('Inert fixture license text.\n', encoding='utf-8')
        row = {'name': name, 'version': version, 'declared_license': 'Apache-2.0',
               'license_files': [{'path': notice.relative_to(self.bundle).as_posix(), 'sha256': digest(notice)}],
               'license_review': 'not_performed'}
        self.dependencies.append(row)
        return row

    def inventory_row(self, name='onnx', version='1.22.0'):
        row = self.dependency_row(name, version)
        self.components.append({'type': 'library', 'name': name, 'version': version,
                                'purl': f'pkg:pypi/{name}@{version}'})
        return row

    def sherpa_distribution(self):
        self.distribution('sherpa-onnx', '1.13.8', folder='sherpa_onnx-1.13.8.dist-info')
        self.dependency_row('sherpa-onnx', '1.13.8')

    def validate(self):
        self.inventories()
        return build.validate_packaged_distributions(self.bundle)

    def test_actual_missing_onnx_inventory_with_native_payload_is_rejected(self):
        self.distribution()
        native = self.bundle / '_internal/onnx/onnx_cpp2py_export.pyd'
        native.parent.mkdir()
        native.write_bytes(b'inert native payload; never loaded')
        self.inventory_row('numpy', '1.26.4')
        with self.assertRaises(ValueError):
            self.validate()

    def test_complete_metadata_receipt_is_json_safe_and_canonically_named(self):
        dist = self.distribution('ONNX.Runtime_Test', folder='onnx_runtime_test-1.22.0.dist-info')
        self.inventory_row('onnx-runtime-test')
        receipt = self.validate()
        rows = json.loads(json.dumps(receipt))['distributions']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['name'], 'onnx-runtime-test')
        self.assertEqual(rows[0]['version'], '1.22.0')
        self.assertEqual(rows[0]['dist_info'], dist.relative_to(self.bundle).as_posix())
        self.assertEqual(rows[0]['metadata_sha256'], digest(dist / 'METADATA'))
        self.assertEqual(receipt, self.validate())

    def test_inventory_is_one_way_not_required_to_match_uncollected_distributions(self):
        self.distribution()
        self.inventory_row()
        self.inventory_row('uncollected-dependency', '3.0')
        receipt = self.validate()
        self.assertEqual([row['name'] for row in receipt['distributions']], ['onnx'])

    def test_missing_dependency_and_sbom_rows_are_both_rejected(self):
        self.distribution()
        row = self.inventory_row()
        component = self.components[0]
        for missing in ['dependency', 'sbom']:
            with self.subTest(missing=missing):
                self.dependencies = [] if missing == 'dependency' else [row]
                self.components = [] if missing == 'sbom' else [component]
                with self.assertRaises(ValueError):
                    self.validate()

    def test_missing_inventory_file_is_a_validation_error(self):
        self.distribution()
        self.inventory_row()
        for missing in ['DEPENDENCIES.json', 'sbom.cdx.json']:
            with self.subTest(missing=missing):
                self.inventories()
                (self.bundle / missing).unlink()
                with self.assertRaises(ValueError):
                    build.validate_packaged_distributions(self.bundle)

    def test_each_inventory_version_must_match_shipped_metadata(self):
        self.distribution()
        self.inventory_row()
        for rows in [self.dependencies, self.components]:
            with self.subTest(inventory='dependencies' if rows is self.dependencies else 'sbom'):
                rows[0]['version'] = '1.21.0'
                with self.assertRaises(ValueError):
                    self.validate()
                rows[0]['version'] = '1.22.0'

    def test_duplicate_canonical_inventory_names_are_rejected(self):
        self.distribution('onnx-runtime', folder='onnx_runtime-1.22.0.dist-info')
        self.inventory_row('onnx-runtime')
        for rows in [self.dependencies, self.components]:
            with self.subTest(inventory='dependencies' if rows is self.dependencies else 'sbom'):
                rows.append({**rows[0], 'name': 'ONNX_runtime'})
                with self.assertRaises(ValueError):
                    self.validate()
                rows.pop()

    def test_duplicate_shipped_identity_is_rejected(self):
        self.distribution()
        self.distribution(version='1.22.1')
        self.inventory_row()
        with self.assertRaises(ValueError):
            self.validate()

    def test_incomplete_second_dist_info_cannot_be_silently_ignored(self):
        self.distribution()
        incomplete = self.bundle / '_internal/onnx-1.22.1.dist-info'
        incomplete.mkdir()
        (incomplete / 'WHEEL').write_text('Wheel-Version: 1.0\n', encoding='utf-8')
        self.inventory_row()
        with self.assertRaises(ValueError):
            self.validate()

    def test_malformed_metadata_is_rejected(self):
        dist = self.distribution()
        self.inventory_row()
        for metadata in [
            '', 'not a metadata header\n',
            'Metadata-Version: 2.3\nVersion: 1.22.0\n',
            'Metadata-Version: 2.3\nName: onnx\n',
            'Metadata-Version: 2.3\nName: onnx\nName: other\nVersion: 1.22.0\n',
            'Metadata-Version: 2.3\nName: onnx\nVersion: 1.22.0\nVersion: 1.21.0\n',
        ]:
            with self.subTest(metadata=metadata):
                (dist / 'METADATA').write_text(metadata, encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.validate()

    def test_notice_tampering_and_missing_file_are_rejected(self):
        self.distribution()
        row = self.inventory_row()
        path = self.bundle / row['license_files'][0]['path']
        path.write_text('changed notice', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.validate()
        path.unlink()
        with self.assertRaises(ValueError):
            self.validate()

    def test_notice_paths_cannot_escape_even_with_matching_hash(self):
        self.distribution()
        row = self.inventory_row()
        outside = self.base / 'outside-LICENSE'
        outside.write_text('Inert fixture license text.\n', encoding='utf-8')
        for path in ['../outside-LICENSE', str(outside),
                     'THIRD_PARTY_LICENSES/../../outside-LICENSE', '..\\outside-LICENSE']:
            with self.subTest(path=path):
                row['license_files'] = [{'path': path, 'sha256': digest(outside)}]
                with self.assertRaises(ValueError):
                    self.validate()

    def test_only_complete_reviewed_exception_allows_absent_notice(self):
        self.distribution()
        row = self.inventory_row()
        row['license_files'] = []
        with self.assertRaises(ValueError):
            self.validate()
        exception = {'reason': 'Fixture decision for absent notice', 'reviewed_by': 'fixture-reviewer',
                     'reference': 'urn:fixture:notice-review:1'}
        for key in exception:
            for value in [None, '', '   ']:
                with self.subTest(key=key, value=value):
                    row['license_exception'] = {**exception, key: value}
                    with self.assertRaises(ValueError):
                        self.validate()
        row['license_exception'] = exception
        self.assertEqual(self.validate()['distributions'][0]['name'], 'onnx')

    def test_reviewed_exception_does_not_hide_tampered_declared_notice(self):
        self.distribution()
        row = self.inventory_row()
        row['license_exception'] = {'reason': 'Fixture', 'reviewed_by': 'reviewer', 'reference': 'urn:fixture:1'}
        (self.bundle / row['license_files'][0]['path']).write_text('tampered', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.validate()

    def test_reconcile_adds_collected_identity_preserves_uncollected_plan_and_is_stable(self):
        self.distribution()
        plan = {'distributions': ['numpy'], 'packages': {'numpy': '1.26.4'}}
        receipt = build.reconcile_packaged_distributions(self.bundle, plan)
        json.dumps(receipt)
        self.assertEqual(set(plan['distributions']), {'onnx', 'numpy'})
        self.assertEqual(plan['packages'], {'numpy': '1.26.4', 'onnx': '1.22.0'})
        before = {key: json.loads(json.dumps(plan[key])) for key in ['distributions', 'packages']}
        second = build.reconcile_packaged_distributions(self.bundle, plan)
        self.assertEqual({key: plan[key] for key in before}, before)
        self.assertEqual(second['added_distributions'], [])

    def test_reconcile_rejects_existing_plan_version_disagreement(self):
        self.distribution()
        plan = {'distributions': ['onnx'], 'packages': {'onnx': '1.21.0'}}
        with self.assertRaises(ValueError):
            build.reconcile_packaged_distributions(self.bundle, plan)

    def test_collected_metadata_supplies_notice_and_sbom_without_installed_fallback(self):
        self.distribution()
        plan = {'version': 'fixture', 'distributions': [], 'packages': {},
                'collect_packages': [], 'dependency_exceptions': {}, 'pyinstaller': {}}
        build.reconcile_packaged_distributions(self.bundle, plan)
        fixture_root = self.base / 'source'
        (fixture_root / 'tools/licenses').mkdir(parents=True)
        write_json(fixture_root / 'tools/licenses/provenance.json', [])
        interpreter = self.base / 'python'
        interpreter.mkdir()
        (interpreter / 'LICENSE.txt').write_text('Inert CPython license fixture.\n', encoding='utf-8')
        with patch.object(build, 'ROOT', fixture_root), \
             patch.object(build, 'sys', types.SimpleNamespace(base_prefix=str(interpreter))), \
             patch.object(build.sysconfig, 'get_path', return_value=str(interpreter)), \
             patch.object(build, 'normalize_packaged_build_metadata'), \
             patch.object(build, 'add_bootloader_metadata'), \
             patch.object(build.metadata, 'distribution', side_effect=AssertionError('Installed metadata fallback forbidden')):
            build.write_notices(self.bundle, plan)
        inventory = json.loads((self.bundle / 'DEPENDENCIES.json').read_text(encoding='utf-8'))
        onnx = next(row for row in inventory['components'] if row['name'] == 'onnx')
        self.assertEqual(onnx['version'], '1.22.0')
        self.assertTrue(onnx['license_files'])
        for notice in onnx['license_files']:
            self.assertEqual(digest(self.bundle / notice['path']), notice['sha256'])
        sbom = json.loads((self.bundle / 'sbom.cdx.json').read_text(encoding='utf-8'))
        self.assertEqual(next(row['version'] for row in sbom['components'] if row['name'] == 'onnx'), '1.22.0')
        self.assertEqual(build.validate_packaged_distributions(self.bundle)['distributions'][0]['name'], 'onnx')

    def test_isolated_runtime_uses_its_own_version_inventory_and_bound_string_notice(self):
        self.distribution()
        self.inventory_row()
        location = 'runtimes/isolated'
        runtime = self.bundle / location
        self.distribution(version='1.19.0', root=runtime)
        notice = runtime / 'licenses/LICENSE'
        notice.parent.mkdir()
        notice.write_text('Isolated runtime license fixture.\n', encoding='utf-8')
        write_json(runtime / 'DEPENDENCIES.json', {'components': [{
            'name': 'onnx', 'version': '1.19.0', 'license_files': ['licenses/LICENSE']}]})
        write_json(runtime / 'runtime.json', {'files': {
            'licenses/LICENSE': {'bytes': notice.stat().st_size, 'sha256': digest(notice)}}})
        self.external_runtimes = [{'location': location,
                                  'dependency_inventory': location + '/DEPENDENCIES.json',
                                  'manifest_sha256': digest(runtime / 'runtime.json')}]
        scoped_component = {'type': 'library', 'name': 'onnx', 'version': '1.19.0',
                            'properties': [{'name': 'titanos:runtime', 'value': location}]}
        self.components.append(scoped_component)
        receipt = self.validate()
        self.assertEqual({row['version'] for row in receipt['distributions']}, {'1.22.0', '1.19.0'})
        self.assertEqual(len(receipt['distributions']), 2)
        self.components.remove(scoped_component)
        with self.assertRaises(ValueError):
            self.validate()
        self.components.append(scoped_component)
        notice.write_text('changed isolated notice', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.validate()

    def test_actual_sherpa_python_and_native_rows_are_distinct_preserved_identities(self):
        self.sherpa_distribution()
        for order in [(SHERPA_PYTHON, SHERPA_NATIVE), (SHERPA_NATIVE, SHERPA_PYTHON)]:
            with self.subTest(first='python' if order[0] is SHERPA_PYTHON else 'native'):
                self.components = [copy.deepcopy(item) for item in order]
                receipt = self.validate()
                self.assertEqual([(row['name'], row['version']) for row in receipt['distributions']],
                                 [('sherpa-onnx', '1.13.8')])
                self.assertEqual(receipt['separately_attributed_components'],
                                 [{'name': 'sherpa-onnx', 'runtime': '', 'version': '1.13.8', 'purl': None,
                                   'role': 'optional-sherpa-native'}])
                self.assertEqual(json.loads(json.dumps(receipt)), receipt)
        # Both records are read as written; the gate never rewrites or drops the native attribution.
        sbom = json.loads((self.bundle / 'sbom.cdx.json').read_text(encoding='utf-8'))
        self.assertEqual(sbom['components'], [SHERPA_NATIVE, SHERPA_PYTHON])

    def test_native_attribution_alone_is_not_the_python_distribution_identity(self):
        self.sherpa_distribution()
        for components in [[SHERPA_NATIVE], [SHERPA_NATIVE, {**SHERPA_NATIVE, 'version': '1.13.9'}]]:
            with self.subTest(rows=len(components)):
                self.components = [copy.deepcopy(item) for item in components]
                with self.assertRaisesRegex(ValueError, 'SBOM missing'):
                    self.validate()

    def test_duplicate_or_conflicting_python_identities_are_rejected_beside_native_attribution(self):
        self.sherpa_distribution()
        duplicate, conflicting, purl = 'SBOM missing, duplicate or version mismatch', 'conflicting Python distribution components', 'PyPI purl differs'
        cases = {
            'duplicate PyPI row': ([SHERPA_PYTHON, SHERPA_PYTHON], duplicate),
            'second PyPI version': ([SHERPA_PYTHON, {**SHERPA_PYTHON, 'version': '1.13.9',
                                                     'purl': 'pkg:pypi/sherpa-onnx@1.13.9'}], duplicate),
            'bare row beside PyPI row': ([SHERPA_PYTHON, {'type': 'library', 'name': 'sherpa-onnx', 'version': '1.13.8'}], conflicting),
            'purl version differs': ([{**SHERPA_PYTHON, 'purl': 'pkg:pypi/sherpa-onnx@1.13.7'}], purl),
            'purl names another distribution': ([{**SHERPA_PYTHON, 'purl': 'pkg:pypi/sherpa-onnx-core@1.13.8'}], purl),
            'component version differs': ([{**SHERPA_PYTHON, 'version': '1.13.9'}], duplicate),
        }
        for label, (components, message) in cases.items():
            with self.subTest(case=label):
                # The native attribution beside these rows must neither mask nor cause the refusal.
                self.components = [copy.deepcopy(item) for item in components] + [copy.deepcopy(SHERPA_NATIVE)]
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()

    def test_native_attribution_and_foreign_ecosystem_purl_do_not_couple_to_the_distribution(self):
        self.sherpa_distribution()
        github = {'type': 'library', 'name': 'sherpa-onnx', 'version': '11afbd0',
                  'purl': 'pkg:github/k2-fsa/sherpa-onnx@11afbd009a7f8c08f4bcf2fc1b265d0df4670fbf'}
        self.components = [copy.deepcopy(SHERPA_PYTHON), {**copy.deepcopy(SHERPA_NATIVE), 'version': '11afbd0'}, github]
        receipt = self.validate()
        self.assertEqual([row['version'] for row in receipt['distributions']], ['1.13.8'])
        self.assertEqual([(row['version'], row['purl'], row['role']) for row in receipt['separately_attributed_components']],
                         [('11afbd0', None, 'optional-sherpa-native'), ('11afbd0', github['purl'], None)])

    def test_encoded_pypi_purl_matches_canonical_identity_and_malformed_purl_is_rejected(self):
        self.distribution('ONNX.Runtime_Test', folder='onnx_runtime_test-1.22.0.dist-info')
        self.dependency_row('onnx-runtime-test')
        self.components = [{'type': 'library', 'name': 'ONNX.Runtime_Test', 'version': '1.22.0',
                            'purl': 'pkg:pypi/ONNX.Runtime_Test@1.22.0?arch=any#sub'}]
        self.assertEqual(self.validate()['distributions'][0]['name'], 'onnx-runtime-test')
        self.components[0]['purl'] = 'pkg:pypi/onnx-runtime-test'
        self.assertEqual(self.validate()['distributions'][0]['version'], '1.22.0')
        self.components[0]['purl'] = 'pkg:PyPI/onnx_runtime_test@1.22.0'  # purl type is case-insensitive
        self.assertEqual(self.validate()['distributions'][0]['version'], '1.22.0')
        self.components[0]['purl'] = 'pkg:pypi/@1.22.0'
        with self.assertRaisesRegex(ValueError, 'Malformed PyPI purl'):
            self.validate()

    def test_isolated_runtime_scope_keeps_root_native_attribution_separate(self):
        self.distribution()
        self.inventory_row()
        location = 'runtimes/isolated'
        runtime = self.bundle / location
        self.distribution('sherpa-onnx', '1.13.8', folder='sherpa_onnx-1.13.8.dist-info', root=runtime)
        notice = runtime / 'licenses/LICENSE'
        notice.parent.mkdir()
        notice.write_text('Isolated runtime license fixture.\n', encoding='utf-8')
        write_json(runtime / 'DEPENDENCIES.json', {'components': [{
            'name': 'sherpa-onnx', 'version': '1.13.8', 'license_files': ['licenses/LICENSE']}]})
        write_json(runtime / 'runtime.json', {'files': {
            'licenses/LICENSE': {'bytes': notice.stat().st_size, 'sha256': digest(notice)}}})
        self.external_runtimes = [{'location': location,
                                  'dependency_inventory': location + '/DEPENDENCIES.json',
                                  'manifest_sha256': digest(runtime / 'runtime.json')}]
        scoped = {'type': 'library', 'name': 'sherpa-onnx', 'version': '1.13.8',
                  'properties': [{'name': 'titanos:runtime', 'value': location}]}
        self.components += [copy.deepcopy(SHERPA_NATIVE), scoped]
        receipt = self.validate()
        self.assertEqual({(row['name'], row['version']) for row in receipt['distributions']},
                         {('onnx', '1.22.0'), ('sherpa-onnx', '1.13.8')})
        self.assertEqual(receipt['separately_attributed_components'], [])
        self.components.remove(scoped)
        with self.assertRaisesRegex(ValueError, 'SBOM missing'):
            self.validate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.evidence:
        EVIDENCE = args.evidence.resolve()
        EVIDENCE.mkdir(exist_ok=False)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PackagedDistributionChecks))
    print(log.getvalue(), end='')
    record = {'passed': result.wasSuccessful(), 'cases': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'source_sha256': digest(SOURCE), 'checker_sha256': digest(Path(__file__)),
              'scope': 'Inert metadata/native-file/notice fixtures only; no framework, native payload, model, GPU, server or network execution'}
    if EVIDENCE:
        (EVIDENCE / 'run.log').write_text(log.getvalue(), encoding='utf-8')
        write_json(EVIDENCE / 'RESULT.json', record)
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
