"""Actual source-copy and sdist filters over inert tiny filesystem fixtures.

Only source_package's AST is loaded; no product/build module is imported and no
builder, interpreter payload, model or network operation is invoked.
"""
import argparse
import ast
from copy import deepcopy
from contextlib import chdir
import fnmatch
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from setuptools._distutils.filelist import FileList
from setuptools._distutils.text_file import TextFile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (ROOT / 'MANIFEST.in').read_text(encoding='utf-8')
# The real export policy (excluded repository paths) that source_package reads; the fixture tree carries a copy like MANIFEST.in.
POLICY = (ROOT / 'tools/public_export.json').read_text(encoding='utf-8')
TREE = ast.parse((ROOT / 'tools/build_binary.py').read_text(encoding='utf-8'))
SOURCE_PACKAGE = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'source_package')
PUBLIC = {
    'LICENSE', 'NOTICE', 'README.md', 'CONTRIBUTING.md', 'SECURITY.md', 'CODE_OF_CONDUCT.md',
    'STATUS.md', 'pyproject.toml', 'MANIFEST.in', 'install.py',
    'src/titanos/__init__.py', 'src/titanos/web/app.js', 'src/titanos/web/index.html',
    'src/titanos/web/style.css', 'src/titanos/web/logo.svg', 'src/titanos/data/catalog.json',
    'src/titanos/web/swagger-ui.css', 'src/titanos/web/swagger-ui-bundle.js', 'tools/licenses/vendored_assets.json',
    'tools/licenses/swagger-ui-dist-5.33.0-LICENSE.txt', 'tools/licenses/swagger-ui-dist-5.33.0-PROVENANCE.json',
    'docs/public.md', 'docs/fixture.txt', 'docs/.env.example', 'docs/.env.sample.example',
    'tests/check_synthetic.py', 'tests/fixtures/ordinary.txt',
    'tools/profiles.json', 'tools/native-profiles.json', 'tools/utility.py', 'tools/matrix_kit/ingest.py',
    'tools/licenses/upstream-LICENSE.txt', 'tools/public_export.json',
}
PRIVATE = {
    # historical record paths named by tools/public_export.json excluded_paths
    'docs/evidence/2026-01-01-fixture.md', 'docs/evidence/nested/fixture.json', 'docs/CLUSTER_HANDOFF.md',
    '.env', '.env.local', 'docs/.env.production', 'tools/credentials.json',
    'tests/credentials.local.json', 'docs/secrets.json', 'docs/token.json', 'docs/tokens.json',
    'tools/id_rsa', 'tools/id_ed25519', 'docs/private.key', 'docs/private.pem',
    'docs/weights.safetensors', 'tests/model.gguf', 'tools/export.onnx', 'docs/legacy.pt',
    'docs/legacy.pth', 'docs/legacy.ckpt', 'docs/model.bin', 'tools/payload.exe',
    'tools/payload.dll', 'tools/payload.pyd', 'tools/payload.so', 'tools/payload.so.1',
    'tools/payload.dylib', 'tools/payload.msi', 'tools/runtime.whl',
    'docs/archive.zip', 'docs/archive.7z', 'docs/archive.tar', 'docs/archive.tar.gz',
    'docs/archive.tar.xz', 'docs/archive.tar.bz2', 'docs/archive.tgz', 'docs/operation.log',
    'docs/validation/raw.json', 'tools/validation/gen01/plan.json', 'tools/matrix_kit/legacy_results.json',
    'tools/validation/gen01/controller.py', 'verification/report.json',
    'hardware-window/state.json', 'outputs/result.txt', 'models/metadata.json',
    'weights/metadata.json', 'env/local.py', 'venv/local.py', '.venv/local.py',
    '.cache/local.json', 'build/generated.py', 'dist/generated.py', 'release/receipt.json',
}
NATIVE_DEEP_PRIVATE = {
    'src/titanos/nested/' + directory + '/fixture.txt'
    for directory in ('__pycache__', '.cache', '.pytest_cache', '.mypy_cache', '.ruff_cache',
                      'env', 'venv', '.venv', 'build', 'dist', 'release', 'weights', 'models',
                      'outputs', 'verification', 'hardware-window', '.git', '.ssh')
}


def sdist_names(root, preselected=False):
    """The file set the real MANIFEST.in under `root` selects, through setuptools' own template processing.

    Normal source/package-data defaults precede MANIFEST processing. The preselected mode also exercises explicit exclusions
    when a VCS finder or prior source inventory already selected a payload. tests/check_public_export_hygiene.py reuses this
    over the repository itself, so there is one sdist rule reader."""
    with chdir(root), patch('setuptools._distutils.filelist.log.warning'):
        files = FileList()
        files.findall()
        files.files = list(files.allfiles) if preselected else [
            name for name in files.allfiles
            if Path(name).parts[0] == 'src' or name in {'README.md', 'pyproject.toml', 'MANIFEST.in'}
        ]
        template = TextFile('MANIFEST.in', strip_comments=True, skip_blanks=True, join_lines=True,
                            lstrip_ws=True, rstrip_ws=True)
        try:
            for line in template.readlines():
                files.process_template_line(line)
        finally:
            template.close()
        return {Path(name).as_posix() for name in files.files}


class SourcePackageChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='titanos-source-package-')
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve()
        self.source = self.workspace / 'source'
        self.source.mkdir()
        self.destination = self.workspace / 'app'
        self.destination.mkdir()
        scope = {'ROOT': self.source, 'Path': Path, 'shutil': shutil, 'os': os,
                 'stat': stat, 'fnmatch': fnmatch, 'json': json}
        exec(compile(ast.Module(body=[SOURCE_PACKAGE], type_ignores=[]), 'actual source_package', 'exec'), scope)
        self.copy_source = scope['source_package']

    def populate(self, extra=()):
        for name in sorted(PUBLIC | PRIVATE | set(extra)):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('INERT SOURCE-PACKAGING FIXTURE: ' + name + '\n', encoding='utf-8')
        (self.source / 'MANIFEST.in').write_text(MANIFEST, encoding='utf-8')
        (self.source / 'tools/public_export.json').write_text(POLICY, encoding='utf-8')

    def copied_names(self):
        root = self.destination / 'SOURCE'
        return {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}

    def manifest_names(self, preselected=False):
        return sdist_names(self.source, preselected)

    def assert_public_retained(self, names):
        self.assertFalse(PUBLIC - names, 'Missing public source: ' + repr(sorted(PUBLIC - names)))

    def test_native_copy_retains_public_source_and_excludes_operational_payloads(self):
        self.populate(NATIVE_DEEP_PRIVATE)
        self.copy_source(self.destination)
        names = self.copied_names()
        self.assert_public_retained(names)
        self.assertFalse(names & (PRIVATE | NATIVE_DEEP_PRIVATE),
                         'Private payload copied: ' + repr(sorted(names & (PRIVATE | NATIVE_DEEP_PRIVATE))))
        for name in PUBLIC:
            self.assertEqual((self.destination / 'SOURCE' / name).read_bytes(), (self.source / name).read_bytes())

    def test_actual_manifest_retains_safe_examples_and_excludes_known_payloads(self):
        self.populate()
        for preselected in (False, True):
            with self.subTest(preselected=preselected):
                names = self.manifest_names(preselected)
                self.assert_public_retained(names)
                self.assertFalse(names & PRIVATE, 'Private sdist payload: ' + repr(sorted(names & PRIVATE)))

    def test_native_and_sdist_both_exclude_archived_gen01_controller(self):
        self.populate()
        self.copy_source(self.destination)
        for names in (self.copied_names(), self.manifest_names(True)):
            self.assertFalse(any(name.startswith('tools/validation/gen01/') for name in names))
            self.assertIn('tests/check_synthetic.py', names)

    def test_native_and_sdist_both_exclude_lab_operational_records(self):
        """tools/public_export.json excluded_paths (read by source_package) and the MANIFEST.in prune/exclude lines agree."""
        policy = json.loads(POLICY)
        self.assertEqual(set(policy['excluded_paths']),
                         {'docs/CLUSTER_HANDOFF.md', 'docs/evidence', 'docs/validation', 'tools/validation/gen01'})
        self.populate()
        self.copy_source(self.destination)
        for names in (self.copied_names(), self.manifest_names(True), self.manifest_names(False)):
            self.assertFalse({name for name in names if name.casefold().startswith('docs/evidence/')}, names)
            self.assertNotIn('docs/CLUSTER_HANDOFF.md', names)
            self.assertNotIn('tools/matrix_kit/legacy_results.json', names)
            self.assertIn('docs/public.md', names)
            self.assertIn('tools/public_export.json', names)

    def test_native_copy_refuses_a_policy_without_excluded_paths(self):
        self.populate()
        (self.source / 'tools/public_export.json').write_text(json.dumps({'excluded_paths': []}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'public_export'):
            self.copy_source(self.destination)

    @unittest.skipUnless(os.name == 'nt', 'Windows case-insensitive native source filtering')
    def test_native_windows_mixed_case_payload_suffixes_and_directory_names(self):
        dangerous = {'docs/model.SaFeTeNsOrS', 'tools/program.ExE', 'tools/plugin.DlL',
                     'docs/archive.ZiP', 'docs/runtime.LoG', 'src/titanos/VeRiFiCaTiOn/raw.json',
                     'docs/EVIDENCE/mixed-case.md', 'docs/cluster_handoff.MD'}
        self.populate(dangerous)
        self.copy_source(self.destination)
        self.assertFalse(self.copied_names() & dangerous)

    def test_native_copy_refuses_included_links_and_reparse_points(self):
        self.populate({'src/titanos/linked.py', 'docs/linked-dir/ordinary.txt'})
        original_lstat = Path.lstat
        for name, symlink in (('README.md', True), ('src/titanos/linked.py', True),
                              ('src/titanos/linked.py', False), ('docs/linked-dir', False)):
            offender = self.source / name
            def simulated_lstat(path, *args, **kwargs):
                observed = original_lstat(path, *args, **kwargs)
                if path == offender:
                    return SimpleNamespace(st_mode=stat.S_IFLNK if symlink else observed.st_mode,
                                           st_file_attributes=0x400)
                return observed
            destination = self.workspace / ('link-case-' + str(len(list(self.workspace.iterdir()))))
            destination.mkdir()
            with self.subTest(name=name, symlink=symlink), patch.object(Path, 'lstat', simulated_lstat):
                with self.assertRaisesRegex(ValueError, '(?i)link|reparse'):
                    self.copy_source(destination)

    def test_browser_fixture_default_is_fresh_and_explicit_root_is_preserved(self):
        source = ROOT / 'tests/studio_browser_fixture.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        main = deepcopy(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main'))
        prefix = []
        for statement in main.body:
            if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'fixture'
                   for node in ast.walk(statement)):
                break
            prefix.append(statement)
        else:
            self.fail('Fixture setup boundary was not found')
        main.body = [*prefix, ast.Return(value=ast.Name(id='args', ctx=ast.Load()))]
        module = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
        create = Mock(side_effect=[str(self.workspace / 'fresh-one'), str(self.workspace / 'fresh-two')])
        scope = {'argparse': argparse, 'Path': Path, 'tempfile': SimpleNamespace(mkdtemp=create)}
        exec(compile(module, str(source), 'exec'), scope)
        with patch.object(sys, 'argv', ['studio-browser-fixture']):
            first, second = scope['main'](), scope['main']()
        self.assertNotEqual(first.root, second.root)
        self.assertEqual(create.call_count, 2)
        create.assert_called_with(prefix='titanos-studio-fixture-')
        create.reset_mock()
        explicit = self.workspace / 'explicit'
        with patch.object(sys, 'argv', ['studio-browser-fixture', '--root', str(explicit)]):
            self.assertEqual(scope['main']().root, explicit)
        create.assert_not_called()
        with patch.object(sys, 'argv', ['studio-browser-fixture', '--host', '192.0.2.1']):
            with self.assertRaisesRegex(SystemExit, 'loopback'):
                scope['main']()
        create.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
