"""Tiny inert composition: real filesystem/hardlinks/pyc, no application process."""
import hashlib
import importlib.machinery
import importlib.util
import json
import marshal
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import compose_binary as c


class PYZ:
    def __init__(self, entries):
        self.entries = entries
        self.toc = {name: (kind, 0, 0) for name, (kind, _) in entries.items()}
    def extract(self, name): return self.entries[name][1]


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.core, self.full, self.output = [self.root / n for n in ['core', 'full', 'new']]
        self.shared = compile('VALUE = 7', 'shared.py', 'exec')
        hook = marshal.dumps(compile('SAFE_HOOK = True', 'hook.py', 'exec'))
        self.hook = NS(toc={'pyi_rth_fixture': (0, 's')}, extract=lambda n: hook)
        self.archives = {}
        for target, profile, text in [(self.core, 'engine', "STATE = 'new'\n"), (self.full, 'full', "STATE = 'old'\n")]:
            target.mkdir()
            source = target / 'SOURCE/src/titanos/__init__.py'
            source.parent.mkdir(parents=True); source.write_text(text)
            files = {'titanos.exe': b'inert-' + profile.encode(), '_internal/titanos/__init__.py': text.encode(),
                     'THIRD_PARTY_LICENSES/titanos-engine/NOTICE': text.encode(),
                     '_internal/python312.dll': b'inert common native dependency'}
            if profile == 'full':
                files.update({'_internal/runtime/inert.dll': b'not a DLL', '_internal/extdep.py': b'VALUE = 31\n',
                              '_internal/depns/resource.txt': b'namespace data'})
            for name, data in files.items():
                p = target / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
            with zipfile.ZipFile(target / '_internal/base_library.zip', 'w') as z:
                code = compile('BASE = 1', profile + '-source.py', 'exec')
                z.writestr('base.pyc', importlib.util.MAGIC_NUMBER + bytes(12) + marshal.dumps(code))
            c.write(target / 'BUILD.json', {'profile': profile, 'version': 'fixture',
                'python': '.'.join(map(str, sys.version_info[:3])), 'pyinstaller': 'fixture',
                'os': 'windows', 'architecture': 'AMD64',
                'packages': {'titanos-engine': 'fixture', 'dep': '1'}, 'distributions': ['dep'],
                'collect_packages': ['titanos', 'dep'],
                'source_files': {'src/titanos/__init__.py': c.digest(source)}})
            c.write(target / 'DEPENDENCIES.json', {'components': [{'name': 'titanos-engine',
                'version': 'fixture',
                'license_files': [{'path': 'THIRD_PARTY_LICENSES/titanos-engine/NOTICE',
                                  'sha256': c.digest(target / 'THIRD_PARTY_LICENSES/titanos-engine/NOTICE')}]}]})
            c.write(target / 'NATIVE_COMPONENTS.json', {})
            c.write(target / 'sbom.cdx.json', {'bomFormat': 'CycloneDX', 'serialNumber': 'urn:uuid:fixture',
                'metadata': {'component': {'type': 'application', 'name': 'Titanos', 'version': 'fixture'}},
                'components': [{'name': 'titanos-engine', 'version': 'fixture'}, {'name': 'dep', 'version': '1'}]})
            (target / 'THIRD_PARTY_NOTICES.md').write_text('Synthetic fixture notice\n')
            self.seal(target)
            entries = {'titanos': (1, compile(text, '__init__.py', 'exec')), 'shared': (0, self.shared)}
            if profile == 'full':
                entries.update({'olddep': (1, compile('VALUE = 23', 'olddep/__init__.py', 'exec')),
                                'olddep.child': (0, compile('VALUE = 29', 'olddep/child.py', 'exec')),
                                'extdep': (0, compile('VALUE = 31', 'extdep.py', 'exec')), 'depns': (3, None)})
            self.archives[target] = (self.hook, PYZ(entries))
        context = patch.object(c, 'archive', side_effect=lambda path: self.archives[path.parent])
        context.start(); self.addCleanup(context.stop)

    def seal(self, root):
        p = root / 'MANIFEST.json'
        data = {f.relative_to(root).as_posix(): c.pin(f) for f in root.rglob('*') if f.is_file() and f != p}
        p.write_text(json.dumps(data))

    def compose(self):
        return c.assemble(self.core, self.full, self.output,
                          core_manifest_sha256=c.digest(self.core / 'MANIFEST.json'),
                          runtime_manifest_sha256=c.digest(self.full / 'MANIFEST.json'))

    def test_complete_composition_binds_pyc_keeps_current_app_and_preserves_inputs(self):
        before = {str(p): c.digest(p) for root in [self.core, self.full] for p in root.rglob('*') if p.is_file()}
        receipt = self.compose()
        self.assertFalse(receipt['old_application_reused'])
        self.assertEqual((self.output / 'SOURCE/src/titanos/__init__.py').read_text(), "STATE = 'new'\n")
        self.assertTrue((self.output / '_internal/runtime/inert.dll').samefile(self.full / '_internal/runtime/inert.dll'))
        self.assertFalse((self.output / 'titanos.exe').samefile(self.core / 'titanos.exe'))
        self.assertFalse((self.output / 'BUILD.json').samefile(self.core / 'BUILD.json'))
        self.assertEqual(set(receipt['dependency_bytecode']), {'olddep', 'olddep.child'})
        self.assertEqual(receipt['namespace_directories'], ['depns'])
        module_path = self.output / '_internal/olddep/__init__.pyc'
        spec = importlib.util.spec_from_file_location('inert_composition_fixture', module_path,
            loader=importlib.machinery.SourcelessFileLoader('inert_composition_fixture', str(module_path)))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertEqual(module.VALUE, 23)
        self.assertEqual(receipt['dependency_bytecode']['olddep']['full_PYZ_marshaled_sha256'],
            hashlib.sha256(marshal.dumps(self.archives[self.full][1].extract('olddep'))).hexdigest())
        notice = c.read(self.output / 'DEPENDENCIES.json')['components'][0]['license_files'][0]
        self.assertEqual(c.digest(self.output / notice['path']), notice['sha256'])
        self.assertEqual(before, {str(p): c.digest(p) for root in [self.core, self.full] for p in root.rglob('*') if p.is_file()})
        self.assertFalse((self.output / 'BUILD_SMOKE.json').exists())
        self.assertFalse((self.output / 'MANIFEST.json').exists())  # only mandatory cold checks may seal it

    def test_manifest_mismatch_fails_before_new_directory(self):
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            c.assemble(self.core, self.full, self.output, core_manifest_sha256='0'*64,
                       runtime_manifest_sha256=c.digest(self.full / 'MANIFEST.json'))
        self.assertFalse(self.output.exists())

    def test_existing_output_and_noncanonical_paths_are_preserved(self):
        self.output.mkdir(); (self.output / 'keep.txt').write_text('owned elsewhere')
        with self.assertRaisesRegex(ValueError, 'fresh directory'): self.compose()
        self.assertEqual((self.output / 'keep.txt').read_text(), 'owned elsewhere')
        with self.assertRaisesRegex(ValueError, 'parent traversal'):
            c.ordinary(self.core / '..' / 'full')

    def test_shared_dependency_code_mismatch_is_refused(self):
        self.archives[self.full][1].entries['shared'] = (0, compile('VALUE = 8', 'shared.py', 'exec'))
        with self.assertRaisesRegex(ValueError, 'bytecode differs'): self.compose()
        self.assertFalse(self.output.exists())

    def test_unmatched_runtime_hook_is_refused(self):
        container, pyz = self.archives[self.full]
        self.archives[self.full] = (NS(toc={'pyi_rth_other': (0, 's')},
            extract=lambda n: marshal.dumps(compile('OTHER_HOOK = True', 'other.py', 'exec'))), pyz)
        with self.assertRaisesRegex(ValueError, 'hook is absent'): self.compose()
        self.assertFalse(self.output.exists())

    def test_crt_cohort_choice_recorded_and_downgrade_refused(self):
        for root, name, data in [(self.core, 'VCRUNTIME140.dll', b'old CRT'),
                                 (self.full, 'vcruntime140.dll', b'new CRT')]:
            (root / '_internal' / name).write_bytes(data); self.seal(root)
        with patch.object(c, 'crt_version', side_effect=lambda p: (14, 42, 0, 0) if p.is_relative_to(self.core) else (14, 51, 0, 0)):
            receipt = self.compose()
        row = receipt['native_collision_choices']['_internal/vcruntime140.dll']
        self.assertEqual(row['selected'], 'retained full native cohort')
        self.output = self.root / 'downgrade'
        with patch.object(c, 'crt_version', side_effect=lambda p: (14, 51, 0, 0) if p.is_relative_to(self.core) else (14, 42, 0, 0)):
            with self.assertRaisesRegex(ValueError, 'downgrade'): self.compose()
        self.assertFalse(self.output.exists())

    def test_new_application_version_keeps_only_the_pinned_dependency_cohort(self):
        before = {str(p): c.digest(p) for p in self.full.rglob('*') if p.is_file()}
        build = c.read(self.core/'BUILD.json')
        build['version'] = build['packages']['titanos-engine'] = 'next-app'
        (self.core/'BUILD.json').write_text(json.dumps(build))
        deps = c.read(self.core/'DEPENDENCIES.json'); deps['components'][0]['version'] = 'next-app'
        (self.core/'DEPENDENCIES.json').write_text(json.dumps(deps))
        sbom = c.read(self.core/'sbom.cdx.json')
        sbom['metadata']['component']['version'] = sbom['components'][0]['version'] = 'next-app'
        (self.core/'sbom.cdx.json').write_text(json.dumps(sbom)); self.seal(self.core)
        receipt = self.compose()
        self.assertEqual((receipt['application_version'],receipt['retained_runtime_application_version']),('next-app','fixture'))
        output_build=c.read(self.output/'BUILD.json')
        self.assertEqual(output_build['packages'],{'titanos-engine':'next-app','dep':'1'})
        output_sbom=c.read(self.output/'sbom.cdx.json')
        self.assertEqual(output_sbom['metadata']['component']['version'],'next-app')
        self.assertEqual(output_sbom['components'],[{'name':'titanos-engine','version':'next-app'},{'name':'dep','version':'1'}])
        self.assertNotEqual(output_sbom['serialNumber'],'urn:uuid:fixture')
        self.assertEqual(c.read(self.output/'DEPENDENCIES.json')['components'][0]['version'],'next-app')
        self.assertEqual(before,{str(p):c.digest(p) for p in self.full.rglob('*') if p.is_file()})

    def test_cross_version_does_not_relax_dependency_or_extension_compatibility(self):
        for change in [{'packages':{'titanos-engine':'fixture','dep':'2'}}, {'architecture':'ARM64'}, {'python':'3.11.0'}]:
            original=c.read(self.core/'BUILD.json')
            (self.core/'BUILD.json').write_text(json.dumps({**original,**change}));self.seal(self.core)
            with self.assertRaises(ValueError):self.compose()
            self.assertFalse(self.output.exists())
            (self.core/'BUILD.json').write_text(json.dumps(original));self.seal(self.core)


if __name__ == '__main__': unittest.main(verbosity=2)
