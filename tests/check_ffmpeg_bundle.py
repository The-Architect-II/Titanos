"""The FFmpeg bundle gates of tools/build_binary.py over inert fixtures: binding a bundle whose BUILD.json carries a `binaries`
mapping or only a `members` inventory (binaries derived from the ELF/PE/Mach-O members), the refusals for a bundle with
neither, the FFmpeg section of THIRD_PARTY_NOTICES.md and the retained LGPL texts under THIRD_PARTY_LICENSES/ffmpeg/, the
member hashes recorded as shipped after the --prefix normalization (bundle hashes kept under normalized_from), and the
packaged-inventory verifier that re-checks all of it. No PyInstaller, executable, server, model or network is involved.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/build_binary.py'
EVIDENCE = None
spec = importlib.util.spec_from_file_location('ffmpeg_bundle_build_contract', SOURCE)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
sys.path.insert(0, str(ROOT / 'tools'))
import native_inventory  # noqa: E402

LICENSE_BYTES = b'GNU LESSER GENERAL PUBLIC LICENSE\nVersion 2.1, February 1999\n\nInert fixture text.\n'
EXECUTABLE = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
# the neutral prefix of the 2026-10-05/06 bundles; a fixture value, nothing is installed there
PREFIX = '/opt/titanos/ffmpeg'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bundle(root, shape, prefix=PREFIX, magic=b'\x7fELF', executables=(EXECUTABLE, 'libavutil.so.61'),
                build_script=True, version_line='9.0.1', **overrides):
    """An inert FFmpeg bundle: executable-image members carrying the configure prefix, the text members an FFmpeg build writes,
    the licence texts, the corresponding-source archive (whose COPYING.LGPLv2.1 equals the bundle's) and a build script.
    `shape` selects what BUILD.json binds the binaries with: 'binaries' (the Windows bundle), 'members' (the Linux bundle),
    'both' or 'neither'."""
    root = Path(root)
    root.mkdir(parents=True)
    for name in executables:
        (root / name).write_bytes(magic + b'\x00' + ('--prefix=' + prefix).encode() + b'\x00inert image bytes for ' + name.encode())
    configuration = '--prefix=' + prefix + ' --disable-gpl --disable-nonfree --disable-static --enable-shared'
    (root / 'version.txt').write_text('ffmpeg version ' + version_line + ' Copyright (c) 2000-2026 the FFmpeg developers\n'
                                      'configuration: ' + configuration + '\nlibavutil      61.  1.101 / 61.  1.101\n', encoding='utf-8')
    (root / 'license-output.txt').write_text('ffmpeg version 9.0.1\n  configuration: ' + configuration + '\n'
                                             'ffmpeg is free software; LGPL fixture\n', encoding='utf-8')
    (root / 'COPYING.LGPLv2.1').write_bytes(LICENSE_BYTES)
    (root / 'LICENSE.md').write_text('# License\n\nMost files in FFmpeg are under the LGPL (fixture overview).\n', encoding='utf-8')
    (root / 'source').mkdir()
    archive = root / 'source/ffmpeg-9.0.1.tar.xz'
    with tarfile.open(archive, 'w:xz') as tar:
        entry = tarfile.TarInfo('ffmpeg-9.0.1/COPYING.LGPLv2.1')
        entry.size = len(LICENSE_BYTES)
        tar.addfile(entry, io.BytesIO(LICENSE_BYTES))
        readme = b'FFmpeg fixture source\n'
        entry = tarfile.TarInfo('ffmpeg-9.0.1/README.md')
        entry.size = len(readme)
        tar.addfile(entry, io.BytesIO(readme))
    if build_script:
        (root / 'source/build_neutral.sh').write_text('#!/bin/sh\n./configure ' + configuration + '\nmake install\n', encoding='utf-8')
    (root / 'source/changes.diff').write_text('# no source changes\n', encoding='utf-8')
    record = {'source_url': 'https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz', 'source_sha256': digest(archive), 'source_changes': 'none',
              'configure': configuration.split(' '), 'license_from_binary': 'LGPL-2.1-or-later', 'external_codec_libraries': [],
              'install_prefix': prefix}
    members = {path.relative_to(root).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
               for path in sorted(root.rglob('*')) if path.is_file()}
    binaries = {name: members[name]['sha256'] for name in executables}
    if shape in ('binaries', 'both'):
        record['binaries'] = binaries
    if shape in ('members', 'both'):
        record['members'] = members
        record['ffmpeg_version'] = '9.0.1'
    record.update(overrides)
    (root / 'BUILD.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


class FfmpegBundleChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ffmpeg-bundle-', dir=EVIDENCE or ROOT.parent)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def bundle(self, shape, name='bundle', **kwargs):
        root = self.base / name
        record = make_bundle(root, shape, **kwargs)
        return root, record

    # ------------------------------------------------------------------------------------------------------ ffmpeg_plan
    def test_binaries_mapping_binds_as_before_and_is_not_derived(self):
        root, record = self.bundle('binaries')
        plan = build.ffmpeg_plan(root)
        self.assertEqual(plan['ffmpeg_binaries'], record['binaries'])
        self.assertFalse(plan['ffmpeg_binaries_derived_from_members'])
        self.assertEqual(Path(plan['ffmpeg_dir']), root.resolve())
        (root / 'libavutil.so.61').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'binary provenance'):
            build.ffmpeg_plan(root)

    def test_members_only_bundle_derives_binaries_from_executable_images(self):
        root, record = self.bundle('members')
        plan = build.ffmpeg_plan(root)
        self.assertTrue(plan['ffmpeg_binaries_derived_from_members'])
        self.assertEqual(plan['ffmpeg_binaries'], {name: record['members'][name]['sha256'] for name in (EXECUTABLE, 'libavutil.so.61')})
        retained = build.ffmpeg_plan(root, retain_inventory=True)
        self.assertEqual(retained['native_ffmpeg']['source_build_binaries'], plan['ffmpeg_binaries'])
        self.assertEqual(retained['native_ffmpeg']['input_files']['COPYING.LGPLv2.1'], hashlib.sha256(LICENSE_BYTES).hexdigest())
        for magic in (b'MZ\x90\x00', b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe'):
            with self.subTest(magic=magic):
                other, other_record = self.bundle('members', name='bundle-' + magic.hex(), magic=magic)
                self.assertEqual(set(build.ffmpeg_plan(other)['ffmpeg_binaries']), {EXECUTABLE, 'libavutil.so.61'})

    def test_both_mappings_must_agree_and_the_declared_one_is_used(self):
        root, record = self.bundle('both')
        plan = build.ffmpeg_plan(root)
        self.assertFalse(plan['ffmpeg_binaries_derived_from_members'])
        self.assertEqual(plan['ffmpeg_binaries'], record['binaries'])

    def test_bundle_with_neither_mapping_is_refused_with_a_typed_message(self):
        root, _ = self.bundle('neither')
        with self.assertRaisesRegex(ValueError, 'neither a binaries mapping nor a members inventory'):
            build.ffmpeg_plan(root)
        with self.assertRaisesRegex(ValueError, 'no binaries mapping and none was derived'):
            native_inventory.finalize_native_inventory(self.base / 'app', root, 'fixture')

    def test_members_inventory_is_checked_against_the_files(self):
        root, record = self.bundle('members')
        (root / 'LICENSE.md').write_text('changed\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'differs from its members inventory: LICENSE.md'):
            build.ffmpeg_plan(root)
        make_bundle(self.base / 'unlisted', 'members')
        (self.base / 'unlisted' / 'stray.txt').write_text('not in the inventory\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'does not list: stray.txt'):
            build.ffmpeg_plan(self.base / 'unlisted')
        make_bundle(self.base / 'text-only', 'members', magic=b'TEXT')
        with self.assertRaisesRegex(ValueError, 'no ELF, PE or Mach-O'):
            build.ffmpeg_plan(self.base / 'text-only')
        make_bundle(self.base / 'empty-members', 'binaries', members={})
        with self.assertRaisesRegex(ValueError, 'members inventory is empty'):
            build.ffmpeg_plan(self.base / 'empty-members')

    def test_native_inventory_accepts_only_a_matching_supplied_mapping(self):
        root, record = self.bundle('binaries')
        app = self.base / 'app'
        shutil.copytree(root, app / '_internal/native/ffmpeg')
        self.assertEqual(native_inventory.finalize_native_inventory(app, root, 'fixture', binaries=record['binaries']), record)
        with self.assertRaisesRegex(ValueError, 'differs from the bundle BUILD.json'):
            native_inventory.finalize_native_inventory(app, root, 'fixture', binaries={EXECUTABLE: 'f' * 64})
        linux, linux_record = self.bundle('members', name='linux')
        shutil.copytree(linux, app / '_internal/native/ffmpeg-linux')
        derived = build.ffmpeg_plan(linux)['ffmpeg_binaries']
        # the Linux-shaped bundle is bound through the derived mapping when its copy sits at the packaged location
        shutil.rmtree(app / '_internal/native/ffmpeg')
        shutil.copytree(linux, app / '_internal/native/ffmpeg')
        record = native_inventory.finalize_native_inventory(app, linux, 'fixture', binaries=derived)
        self.assertEqual(record['binaries'], derived)
        self.assertEqual(record['members'], linux_record['members'])

    # ----------------------------------------------------------------------------------------------------- write_notices
    def package(self, root, plan_extra=None, before=None):
        """Run the real write_notices over a destination that carries the bundle at _internal/native/ffmpeg (as PyInstaller's datas
        place it), with the retained-license catalog and the interpreter licence replaced by fixtures and the bootloader step
        patched out; `before(destination)` edits the packaged copy before the notices run."""
        destination = self.base / 'app'
        (destination / '_internal').mkdir(parents=True)
        shutil.copytree(root, destination / build.FFMPEG_LOCATION)
        if before is not None:
            before(destination)
        plan = {'version': 'fixture', 'distributions': [], 'packages': {}, 'collect_packages': [], 'dependency_exceptions': [],
                'pyinstaller': '6.22.2', **build.ffmpeg_plan(root), **(plan_extra or {})}
        fixture_root = self.base / 'source-tree'
        (fixture_root / 'tools/licenses').mkdir(parents=True, exist_ok=True)
        (fixture_root / 'tools/licenses/provenance.json').write_text('[]\n', encoding='utf-8')
        interpreter = self.base / 'python'
        interpreter.mkdir(exist_ok=True)
        (interpreter / 'LICENSE.txt').write_text('Inert CPython license fixture.\n', encoding='utf-8')
        with patch.object(build, 'ROOT', fixture_root), \
             patch.object(build, 'sys', types.SimpleNamespace(base_prefix=str(interpreter))), \
             patch.object(build.sysconfig, 'get_path', return_value=str(interpreter)), \
             patch.object(build, 'add_bootloader_metadata'), \
             patch.object(build.metadata, 'distribution', side_effect=AssertionError('Installed metadata fallback forbidden')):
            build.write_notices(destination, plan)
        return destination, plan

    def row(self, destination):
        rows = [row for row in json.loads((destination / 'NATIVE_COMPONENTS.json').read_text(encoding='utf-8')) if row['name'] == 'FFmpeg']
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_notices_section_license_copies_and_members_as_shipped(self):
        for shape in ('binaries', 'members', 'both'):
            with self.subTest(shape=shape):
                root, record = self.bundle(shape, name='bundle-' + shape)
                destination, plan = self.package(root)
                row = self.row(destination)
                packaged = destination / build.FFMPEG_LOCATION
                # the text members were normalized and are recorded as shipped, with the bundle hash kept under normalized_from
                for name in ('version.txt', 'license-output.txt'):
                    text = (packaged / name).read_text(encoding='utf-8')
                    self.assertIn('--prefix=<install-prefix>', text)
                    self.assertNotIn(PREFIX, text)
                    member = row['members'][name]
                    self.assertEqual(member['sha256'], digest(packaged / name))
                    self.assertEqual(member['bytes'], (packaged / name).stat().st_size)
                    self.assertEqual(member['normalized_from']['sha256'], digest(root / name))
                    self.assertEqual(member['normalized_from']['bytes'], (root / name).stat().st_size)
                    self.assertNotEqual(member['sha256'], member['normalized_from']['sha256'])
                for name, member in row['members'].items():
                    self.assertEqual(member['sha256'], digest(packaged / name), name)
                    if name not in build.FFMPEG_NORMALIZED_TEXT:
                        self.assertNotIn('normalized_from', member, name)
                        self.assertEqual(member['sha256'], digest(root / name), name)
                self.assertEqual(set(row['members']), {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} - {'BUILD.json'})
                self.assertEqual(row['binaries'], {name: digest(packaged / name) for name in (EXECUTABLE, 'libavutil.so.61')})
                self.assertEqual(row['binaries_derived_from_members'], shape == 'members')
                self.assertEqual(row['version'], '9.0.1')
                self.assertEqual(row['configure'][0], '--prefix=<install-prefix>')
                self.assertEqual(row['install_prefix'], '<install-prefix>')
                self.assertNotIn(PREFIX, json.dumps(row))
                self.assertEqual(row['location'], build.FFMPEG_LOCATION)
                # the packaged BUILD.json is the same record
                copied = json.loads((packaged / 'BUILD.json').read_text(encoding='utf-8'))
                self.assertEqual(copied, {key: value for key, value in row.items() if key not in ('name', 'location')})
                # the LGPL texts are retained under THIRD_PARTY_LICENSES/ffmpeg/ with their hashes
                self.assertEqual([item['path'] for item in row['license_files']],
                                 ['THIRD_PARTY_LICENSES/ffmpeg/COPYING.LGPLv2.1', 'THIRD_PARTY_LICENSES/ffmpeg/LICENSE.md'])
                for item in row['license_files']:
                    self.assertEqual((destination / item['path']).read_bytes(), (root / Path(item['path']).name).read_bytes())
                    self.assertEqual(digest(destination / item['path']), item['sha256'])
                self.assertEqual((destination / 'THIRD_PARTY_LICENSES/ffmpeg/COPYING.LGPLv2.1').read_bytes(), LICENSE_BYTES)
                # the corresponding source and its build script are named with their hashes
                source = row['corresponding_source']
                self.assertEqual(source['archive'], build.FFMPEG_LOCATION + '/source/ffmpeg-9.0.1.tar.xz')
                self.assertEqual(source['archive_sha256'], record['source_sha256'])
                self.assertEqual(source['build_scripts'], {build.FFMPEG_LOCATION + '/source/build_neutral.sh': digest(root / 'source/build_neutral.sh')})
                self.assertEqual(set(source['other_files']), {build.FFMPEG_LOCATION + '/source/changes.diff'})
                # the notices section
                notices = (destination / 'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')
                section = notices[notices.index('\n## FFmpeg\n'):]
                for expected in ('FFmpeg 9.0.1', 'LGPL-2.1-or-later', '`--disable-gpl`, `--disable-nonfree`', 'no GPL-licensed or nonfree FFmpeg component',
                                 'THIRD_PARTY_LICENSES/ffmpeg/COPYING.LGPLv2.1 (sha256 ' + hashlib.sha256(LICENSE_BYTES).hexdigest() + ')',
                                 'THIRD_PARTY_LICENSES/ffmpeg/LICENSE.md (sha256 ' + digest(root / 'LICENSE.md') + ')',
                                 '`' + source['archive'] + '` (sha256 ' + record['source_sha256'] + ', ' + str((root / 'source/ffmpeg-9.0.1.tar.xz').stat().st_size) + ' bytes, from ' + record['source_url'] + ')',
                                 'build_neutral.sh (sha256 ' + digest(root / 'source/build_neutral.sh') + ')', 'changes.diff',
                                 'Written offer', 'three years', 'LGPL-2.1 section 6', 'not a legal review'):
                    self.assertIn(expected, section, expected)
                self.assertNotIn(PREFIX, notices)
                self.assertEqual(row['notice_section'], '## FFmpeg')
                sbom = json.loads((destination / 'sbom.cdx.json').read_text(encoding='utf-8'))
                ffmpeg = next(item for item in sbom['components'] if item['name'] == 'FFmpeg')
                self.assertEqual((ffmpeg['version'], ffmpeg['licenses']), ('9.0.1', [{'expression': 'LGPL-2.1-or-later'}]))
                # the verifier the chain runs re-checks it and reports the row
                receipt = build.validate_packaged_distributions(destination)
                self.assertEqual(receipt['native_ffmpeg'], [{'location': build.FFMPEG_LOCATION, 'binaries': 2, 'members': len(row['members']),
                                                             'license_files': 2, 'binaries_derived_from_members': shape == 'members'}])
                self.assertNotIn('ffmpeg_dir', json.dumps(receipt))
                shutil.rmtree(destination)

    def test_verifier_refuses_the_shapes_the_5aa3311_builds_shipped(self):
        root, record = self.bundle('members')
        destination, plan = self.package(root)
        build.validate_packaged_distributions(destination)
        packaged = destination / build.FFMPEG_LOCATION
        native_path = destination / 'NATIVE_COMPONENTS.json'
        original = native_path.read_bytes()
        notices_path = destination / 'THIRD_PARTY_NOTICES.md'
        notices = notices_path.read_bytes()

        def restore():
            native_path.write_bytes(original)
            notices_path.write_bytes(notices)

        # the Linux dist of 2026-10-06: member hashes recorded before the --prefix normalization
        rows = json.loads(original.decode('utf-8'))
        row = next(item for item in rows if item['name'] == 'FFmpeg')
        row['members']['version.txt'] = {**row['members']['version.txt'], 'sha256': row['members']['version.txt']['normalized_from']['sha256']}
        native_path.write_text(json.dumps(rows), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'member differs from NATIVE_COMPONENTS.json: version.txt'):
            build.validate_packaged_distributions(destination)
        restore()
        # both dists of 2026-10-06: no FFmpeg section, no retained texts, no members as shipped
        rows = json.loads(original.decode('utf-8'))
        row = next(item for item in rows if item['name'] == 'FFmpeg')
        for key, message in (('members', 'lacks its members as shipped'), ('license_files', 'lacks its retained license texts'),
                             ('binaries', 'lacks its binaries mapping'), ('location', 'lacks its packaged location')):
            with self.subTest(missing=key):
                native_path.write_text(json.dumps([{k: v for k, v in row.items() if k != key}]), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, message):
                    build.validate_packaged_distributions(destination)
        restore()
        text = notices.decode('utf-8')
        notices_path.write_text(text.replace('## FFmpeg', '## Media'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'lacks the FFmpeg section'):
            build.validate_packaged_distributions(destination)
        notices_path.write_text(text.replace(row['corresponding_source']['archive'], 'elsewhere'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'lacks the FFmpeg section'):
            build.validate_packaged_distributions(destination)
        restore()
        copied = destination / 'THIRD_PARTY_LICENSES/ffmpeg/LICENSE.md'
        kept = copied.read_bytes()
        copied.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Retained FFmpeg license text differs'):
            build.validate_packaged_distributions(destination)
        copied.unlink()
        with self.assertRaisesRegex(ValueError, 'Unavailable distribution inventory member|Retained FFmpeg license text differs'):
            build.validate_packaged_distributions(destination)
        copied.write_bytes(kept)
        # the shipped directory and the inventories must agree
        (packaged / 'stray.txt').write_text('late addition\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'differs from its members inventory: stray.txt'):
            build.validate_packaged_distributions(destination)
        (packaged / 'stray.txt').unlink()
        shipped = (packaged / 'libavutil.so.61').read_bytes()
        (packaged / 'libavutil.so.61').write_bytes(shipped + b'!')
        with self.assertRaisesRegex(ValueError, 'binary differs from NATIVE_COMPONENTS.json: libavutil.so.61'):
            build.validate_packaged_distributions(destination)
        (packaged / 'libavutil.so.61').write_bytes(shipped)
        build_json = packaged / 'BUILD.json'
        kept = build_json.read_bytes()
        copied_record = json.loads(kept.decode('utf-8'))
        copied_record['binaries'] = dict(copied_record['binaries'], extra='0' * 64)
        build_json.write_text(json.dumps(copied_record), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'BUILD.json disagrees with NATIVE_COMPONENTS.json'):
            build.validate_packaged_distributions(destination)
        build_json.write_bytes(kept)
        self.assertEqual(len(build.validate_packaged_distributions(destination)['native_ffmpeg']), 1)
        # a packaged FFmpeg directory must be bound by an FFmpeg row: an empty or a missing inventory is refused while the
        # directory ships; without the directory the other profiles report no FFmpeg (their shipping gates are unchanged)
        native_path.write_text('[]\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'not bound by an FFmpeg row'):
            build.validate_packaged_distributions(destination)
        native_path.unlink()
        with self.assertRaisesRegex(ValueError, 'not bound by an FFmpeg row'):
            build.validate_packaged_distributions(destination)
        shutil.rmtree(packaged)
        self.assertEqual(build.validate_packaged_distributions(destination)['native_ffmpeg'], [])

    def test_lgpl_claim_requires_the_configuration_that_backs_it(self):
        cases = {'enable-gpl': {'configure': ['--prefix=' + PREFIX, '--enable-gpl', '--disable-nonfree']},
                 'no disable-nonfree': {'configure': ['--prefix=' + PREFIX, '--disable-gpl']},
                 'nonfree true': {'nonfree': True},
                 'external codec': {'external_codec_libraries': ['libx264']}}
        for label, overrides in cases.items():
            with self.subTest(case=label):
                root, _ = self.bundle('binaries', name='bundle-' + label.replace(' ', '-'), **overrides)
                with self.assertRaisesRegex(ValueError, 'does not exclude GPL and nonfree components'):
                    self.package(root)
                shutil.rmtree(self.base / 'app')

    def test_version_source_script_and_packaged_members_are_bound(self):
        root, _ = self.bundle('members', name='declared-version', ffmpeg_version='9.0.2')
        with self.assertRaisesRegex(ValueError, 'ffmpeg_version differs from the corresponding-source archive name'):
            self.package(root)
        shutil.rmtree(self.base / 'app')
        root, _ = self.bundle('binaries', name='announced-version', version_line='9.0.2')
        with self.assertRaisesRegex(ValueError, 'version.txt announces another version'):
            self.package(root)
        shutil.rmtree(self.base / 'app')
        root, _ = self.bundle('binaries', name='no-script', build_script=False)
        with self.assertRaisesRegex(ValueError, 'ships no build script'):
            self.package(root)
        shutil.rmtree(self.base / 'app')
        root, _ = self.bundle('binaries', name='edited-copy')

        def edit_license(destination):
            (destination / build.FFMPEG_LOCATION / 'LICENSE.md').write_text('edited in the packaged copy\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed during packaging: LICENSE.md'):
            self.package(root, before=edit_license)
        shutil.rmtree(self.base / 'app')

        def add_file(destination):
            (destination / build.FFMPEG_LOCATION / 'extra.txt').write_text('not in the bundle\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'carries files the bundle does not: extra.txt'):
            self.package(root, before=add_file)
        shutil.rmtree(self.base / 'app')
        root, _ = self.bundle('binaries', name='changed-input')
        with self.assertRaisesRegex(ValueError, 'inputs changed during build'):
            self.package(root, plan_extra={'ffmpeg_binaries': {EXECUTABLE: '0' * 64}})
        shutil.rmtree(self.base / 'app')
        root, _ = self.bundle('binaries', name='license-differs')
        (root / 'COPYING.LGPLv2.1').write_bytes(b'not the corresponding license\n')
        with self.assertRaisesRegex(ValueError, 'license text differs from corresponding source'):
            self.package(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.evidence:
        EVIDENCE = args.evidence.resolve()
        EVIDENCE.mkdir(exist_ok=False)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FfmpegBundleChecks))
    print(log.getvalue(), end='')
    record = {'passed': result.wasSuccessful(), 'cases': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'source_sha256': digest(SOURCE), 'checker_sha256': digest(Path(__file__)),
              'scope': 'Inert FFmpeg bundle fixtures over the real ffmpeg_plan, write_notices and validate_packaged_distributions; no PyInstaller, executable, server, model or network'}
    if EVIDENCE:
        (EVIDENCE / 'run.log').write_text(log.getvalue(), encoding='utf-8')
        (EVIDENCE / 'RESULT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
