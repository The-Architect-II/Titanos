"""Normal builder includes a verified music sidecar before sealing; no binaries run."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('package_fixtures',ROOT/'tests/check_music_package.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
sys.modules['music_package']=fixtures.package
tree=ast.parse((ROOT/'tools/build_binary.py').read_text(encoding='utf-8'))
scope={'Path':Path,'json':json,'shutil':shutil,'digest':lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()}
from packaging.utils import canonicalize_name
scope.update(ROOT=ROOT,canonicalize_name=canonicalize_name)
for name in ('music_runtime_plan','supplement_runtime_licenses','cuda_vendor_component','include_music_runtime'):
    node=next(row for row in tree.body if isinstance(row,ast.FunctionDef) and row.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual build functions','exec'),scope)

RUNTIME_ROW='Isolated ACE-Step runtime'
CUDA_ROW_PREFIX='NVIDIA CUDA runtime libraries'


def fresh_application(fixture,name,native_components):
    """A not-yet-sealed application directory with the inventories include_music_runtime extends."""
    destination=fixture.root/name;destination.mkdir()
    for file_name,value in [('NATIVE_COMPONENTS.json',native_components),('DEPENDENCIES.json',{'components':[]}),('sbom.cdx.json',{'components':[]})]:
        (destination/file_name).write_text(json.dumps(value),encoding='utf-8')
    (destination/'THIRD_PARTY_NOTICES.md').write_text('Original application notices\n',encoding='utf-8')
    return destination


def write_torch_dist_info(fixture,version):
    """An inert torch dist-info in the fixture site-packages, with the license and notice files a torch wheel retains."""
    info=fixture.site/('torch-'+version+'.dist-info')
    fixture.write(info/'METADATA',('Name: torch\nVersion: '+version+'\nLicense: BSD-3-Clause\n').encode('utf-8'))
    fixture.write(info/'licenses/LICENSE',b'fixture torch license text\n')
    fixture.write(info/'licenses/NOTICE',b'fixture torch notice text\n')


def cuda_rows(destination):
    rows=json.loads((destination/'NATIVE_COMPONENTS.json').read_text(encoding='utf-8'))
    return rows,[row for row in rows if row['name'].startswith(CUDA_ROW_PREFIX)]


@unittest.skipUnless(os.name=='nt' or sys.platform.startswith('linux'), 'Music runtime publication currently supports Windows/Linux')
class Checks(unittest.TestCase):
    def test_planned_sidecar_is_copied_bound_and_inventoried(self):
        fixture=fixtures.MusicPackageTests();fixture.setUp()
        try:
            fixture.build();plan={'native_music':scope['music_runtime_plan'](fixture.output)}
            destination=fresh_application(fixture,'new-app',[])
            scope['include_music_runtime'](destination,plan)
            copied=destination/'runtimes/ace-step-1.5'
            self.assertEqual(fixtures.package.verify_runtime(copied),fixtures.package.verify_runtime(fixture.output))
            rows=json.loads((destination/'NATIVE_COMPONENTS.json').read_text())
            native=next(row for row in rows if row['name']==RUNTIME_ROW)
            self.assertNotIn('directory',native)
            self.assertEqual(native['manifest_sha256'],scope['digest'](copied/'runtime.json'))
            self.assertIn('example',{row['name'] for row in json.loads((destination/'sbom.cdx.json').read_text())['components']})
            self.assertFalse((destination/'MANIFEST.json').exists()) # caller seals afterwards
            with self.assertRaises(ValueError):scope['include_music_runtime'](destination,plan)
            other=fixture.root/'sealed';other.mkdir();(other/'MANIFEST.json').write_text('{}')
            with self.assertRaises(ValueError):scope['include_music_runtime'](other,plan)
            manifest=fixture.output/'runtime.json';manifest.write_text(manifest.read_text()+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'changed after build planning'):
                scope['include_music_runtime'](fixture.root/'not-created',plan)
            self.assertFalse((fixture.root/'not-created').exists())
        finally:fixture.tearDown()


@unittest.skipUnless(os.name=='nt' or sys.platform.startswith('linux'), 'Music runtime publication currently supports Windows/Linux')
class CudaVendorScopes(unittest.TestCase):
    """The main runtime's torch wheel and the isolated runtime's torch wheel carry CUDA vendor libraries built against different
    CUDA releases; NATIVE_COMPONENTS.json must keep one scoped row per payload for human vendor-terms review. No legal claim."""
    def test_helper_row_names_scope_location_version_and_build_tag(self):
        row=scope['cuda_vendor_component']('main runtime','2.8.0+cu128','THIRD_PARTY_LICENSES/torch','_internal/torch/lib')
        self.assertEqual(row['name'],'NVIDIA CUDA runtime libraries (main runtime)')
        self.assertEqual((row['scope'],row['location'],row['torch_version'],row['cuda_build_tag'],row['notice_files'],row['review']),
                         ('main runtime','_internal/torch/lib','2.8.0+cu128','cu128','THIRD_PARTY_LICENSES/torch','not_performed'))
        self.assertIn('human vendor-terms review',row['status'])
        self.assertIsNone(scope['cuda_vendor_component']('main runtime','2.8.0','THIRD_PARTY_LICENSES/torch','_internal/torch/lib')['cuda_build_tag'])

    def test_combined_build_keeps_main_cu128_and_isolated_cu130_rows_distinct(self):
        fixture=fixtures.MusicPackageTests();fixture.setUp()
        try:
            write_torch_dist_info(fixture,'2.10.0+cu130')
            fixture.build();plan={'native_music':scope['music_runtime_plan'](fixture.output)}
            runtime_torch=next(row for row in json.loads((fixture.output/'DEPENDENCIES.json').read_text(encoding='utf-8'))['components'] if row['name']=='torch')
            self.assertEqual(runtime_torch['version'],'2.10.0+cu130');self.assertEqual(len(runtime_torch['license_files']),2)
            main=scope['cuda_vendor_component']('main runtime','2.8.0+cu128','THIRD_PARTY_LICENSES/torch','_internal/torch/lib')
            destination=fresh_application(fixture,'combined-app',[main])
            scope['include_music_runtime'](destination,plan)
            rows,cuda=cuda_rows(destination)
            self.assertEqual([row['name'] for row in rows],[main['name'],'NVIDIA CUDA runtime libraries (isolated ACE-Step runtime)',RUNTIME_ROW])
            self.assertEqual(cuda[0],main)
            isolated=cuda[1]
            self.assertEqual((isolated['scope'],isolated['torch_version'],isolated['cuda_build_tag'],isolated['review']),
                             ('isolated ACE-Step runtime','2.10.0+cu130','cu130','not_performed'))
            self.assertEqual(isolated['location'],'runtimes/ace-step-1.5/python/lib/python3.11/site-packages/torch/lib')
            for key in ('scope','location','torch_version','cuda_build_tag','name'):
                self.assertNotEqual(cuda[0][key],cuda[1][key],key)
            self.assertEqual(sorted(isolated['notice_files']),
                             ['runtimes/ace-step-1.5/python/lib/python3.11/site-packages/torch-2.10.0+cu130.dist-info/licenses/LICENSE',
                              'runtimes/ace-step-1.5/python/lib/python3.11/site-packages/torch-2.10.0+cu130.dist-info/licenses/NOTICE'])
            for path in isolated['notice_files']:self.assertTrue((destination/path).is_file(),path)
            self.assertEqual(rows[-1]['manifest_sha256'],scope['digest'](destination/'runtimes/ace-step-1.5/runtime.json'))
            notices=(destination/'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')
            self.assertIn('CUDA payload scoped to the isolated runtime',notices)
            self.assertNotIn('approv',notices.split('## Optional isolated music runtime')[1].split('CUDA payload')[1])
        finally:fixture.tearDown()

    def test_ace_only_build_gets_exactly_one_isolated_cuda_row(self):
        fixture=fixtures.MusicPackageTests();fixture.setUp()
        try:
            write_torch_dist_info(fixture,'2.10.0+cu130')
            fixture.build();plan={'native_music':scope['music_runtime_plan'](fixture.output)}
            destination=fresh_application(fixture,'ace-only-app',[])
            scope['include_music_runtime'](destination,plan)
            rows,cuda=cuda_rows(destination)
            self.assertEqual(len(cuda),1)
            self.assertEqual((cuda[0]['scope'],cuda[0]['cuda_build_tag'],cuda[0]['torch_version']),('isolated ACE-Step runtime','cu130','2.10.0+cu130'))
            self.assertEqual([row['name'] for row in rows],[cuda[0]['name'],RUNTIME_ROW])
        finally:fixture.tearDown()

    def test_runtime_without_cuda_torch_adds_no_cuda_row(self):
        fixture=fixtures.MusicPackageTests();fixture.setUp()
        try:
            write_torch_dist_info(fixture,'2.10.0') # a torch wheel without a +cuNNN build tag carries no CUDA vendor payload
            fixture.build();plan={'native_music':scope['music_runtime_plan'](fixture.output)}
            destination=fresh_application(fixture,'cpu-app',[])
            scope['include_music_runtime'](destination,plan)
            rows,cuda=cuda_rows(destination)
            self.assertEqual(cuda,[])
            self.assertEqual([row['name'] for row in rows],[RUNTIME_ROW])
            self.assertNotIn('CUDA payload',(destination/'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8'))
        finally:fixture.tearDown()


@unittest.skipUnless(os.name=='nt' or sys.platform.startswith('linux'), 'Music runtime publication currently supports Windows/Linux')
class RuntimeLicenseSupplements(unittest.TestCase):
    """Isolated-runtime distributions without inventory notices: shipped 'LICENCE' text or retained upstream text is copied and
    re-hashed; neither -> refusal. Uses the real build_binary module against the inert fixture runtime."""
    def setUp(self):
        sys.path.insert(0, str(ROOT / 'tools'))
        import build_binary
        self.bb = build_binary

    def test_inventory_discovers_british_licence_spelling(self):
        fixture = fixtures.MusicPackageTests(); fixture.setUp()
        try:
            fixture.write(fixture.site / 'brit-3.0.dist-info/METADATA', b'Name: brit' + bytes([10]) + b'Version: 3.0' + bytes([10]) + b'License: MPL-2.0 AND MIT' + bytes([10]))
            fixture.write(fixture.site / 'brit-3.0.dist-info/licenses/LICENCE', b'fixture licence text' + bytes([10]))
            fixture.build()
            rows = json.loads((fixture.output / 'DEPENDENCIES.json').read_text(encoding='utf-8'))['components']
            brit = next(row for row in rows if row['name'] == 'brit')
            self.assertTrue(brit['license_files'] and brit['license_files'][0].endswith('licenses/LICENCE'))
        finally: fixture.tearDown()

    def test_supplement_shipped_retained_and_refusal(self):
        fixture = fixtures.MusicPackageTests(); fixture.setUp()
        try:
            # a distribution whose wheel ships no license text at all
            fixture.write(fixture.site / 'nolic-2.0.dist-info/METADATA', b'Name: nolic' + bytes([10]) + b'Version: 2.0' + bytes([10]) + b'License-Expression: MIT' + bytes([10]))
            fixture.build(); plan = {'native_music': self.bb.music_runtime_plan(fixture.output)}
            runtime = fixture.output
            deps = json.loads((runtime / 'DEPENDENCIES.json').read_text(encoding='utf-8'))['components']
            self.assertFalse(next(row for row in deps if row['name'] == 'nolic')['license_files'])
            # unit: a dist-info that carries only a British-spelled LICENCE but whose inventory row lists nothing -> shipped copy
            unit = fixture.root / 'unit-runtime'; (unit / 'python/Lib/site-packages/brit-3.0.dist-info/licenses').mkdir(parents=True)
            (unit / 'runtime.json').write_text(json.dumps({'layout': {'site_packages': 'python/Lib/site-packages'}}), encoding='utf-8')
            (unit / 'python/Lib/site-packages/brit-3.0.dist-info/licenses/LICENCE').write_bytes(b'fixture licence text' + bytes([10]))
            app = fixture.root / 'unit-app'; app.mkdir()
            records = self.bb.supplement_runtime_licenses(app, unit, 'runtimes/ace-step-1.5', [{'name': 'brit', 'version': '3.0', 'license_files': []}])
            self.assertEqual(records[0]['files'][0]['source'], 'shipped:python/Lib/site-packages/brit-3.0.dist-info/licenses/LICENCE')
            self.assertTrue((app / records[0]['files'][0]['path']).is_file())
            # neither shipped nor retained -> refusal naming the distribution
            destination = fixture.root / 'app-refused'; destination.mkdir()
            for name, value in [('NATIVE_COMPONENTS.json', []), ('DEPENDENCIES.json', {'components': []}), ('sbom.cdx.json', {'components': []})]:
                (destination / name).write_text(json.dumps(value), encoding='utf-8')
            (destination / 'THIRD_PARTY_NOTICES.md').write_text('Original application notices' + chr(10), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'nolic 2.0'):
                self.bb.include_music_runtime(destination, plan)
            # retained upstream text from a provenance catalog -> copied, recorded, hash-bound
            catalog_dir = fixture.root / 'catalog'; catalog_dir.mkdir()
            text = b'retained upstream text' + bytes([10])
            (catalog_dir / 'nolic-2.0-LICENSE.txt').write_bytes(text)
            (catalog_dir / 'provenance.json').write_text(json.dumps([{'file': 'nolic-2.0-LICENSE.txt', 'name': 'nolic', 'version': '2.0', 'url': 'https://example.invalid/LICENSE',
                                                                     'sha256': hashlib.sha256(text).hexdigest()}]), encoding='utf-8')
            app2 = fixture.root / 'app-retained'; app2.mkdir()
            records = self.bb.supplement_runtime_licenses(app2, runtime, 'runtimes/ace-step-1.5', deps, catalog=catalog_dir / 'provenance.json')
            self.assertEqual([r['name'] for r in records], ['nolic'])
            self.assertEqual(records[0]['files'][0]['source'], 'retained:tools/licenses/nolic-2.0-LICENSE.txt')
            copied = app2 / records[0]['files'][0]['path']
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), records[0]['files'][0]['sha256'])
            # tampering with the retained text is refused before copying
            (catalog_dir / 'nolic-2.0-LICENSE.txt').write_bytes(b'changed' + bytes([10]))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.bb.supplement_runtime_licenses(fixture.root / 'app-x', runtime, 'runtimes/ace-step-1.5', deps, catalog=catalog_dir / 'provenance.json')
        finally: fixture.tearDown()

if __name__=='__main__':unittest.main(verbosity=2)
