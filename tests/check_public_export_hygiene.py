"""Public export hygiene (tools/public_export.json): the source copies that leave this repository, and the tracked tree outside
the excluded records, carry no lab locators.

The export set is computed exactly as a build would compute it: the real tools/build_binary.source_package is called against this
repository into a temporary directory, and the real MANIFEST.in is processed by setuptools (check_source_package.sdist_names).
No rule is copied here. The checks:

- the build-level manifest (build_binary.package_manifest, what main() writes as MANIFEST.json) covers exactly the exported set;
- both copies exclude the historical record paths the policy names (docs/evidence/**, docs/CLUSTER_HANDOFF.md, docs/validation,
  tools/validation/gen01) and the private files (tools/matrix_kit/legacy_results.json, tools/lab_hosts.private.json), and agree
  with each other and with MANIFEST.in's lines;
- the tree kind decides what may exist: in the private tree (the default) the excluded records exist and are tolerated only there;
  in a public tree (TITANOS_TREE_KIND=public, or the marker file the policy names) every excluded path and private file must be
  absent and the whole tracked tree is scanned;
- every UTF-8 text file in either copy, and every tracked file outside the excluded records, is scanned with the policy's generic
  detection patterns (Windows user-profile paths, agent work trees, RFC1918 IPv4 literals, drive-rooted lab records roots, Linux
  home directories, the lab's Linux hostname shape, bare drive-letter pairs, SSH host-key fingerprints, Windows generated machine
  names), with the exporting machine's own name read from the environment at scan time, and, when the private host map is present,
  with the lab's own machine names and hostnames (class private_denylist); a policy-pattern hit is tolerated only when the matched
  text equals an allowlisted fixture literal for that file, a machine-name or private-denylist hit never, and no allowlist literal
  may be of the generated machine-name class or a private name: the fleet machines are published under stable role labels
  (tools/lab_paths.HOST_ROLE_LABELS), and the public matrix views carry zero hits of the generated class without any allowlist.
  Failure messages name the file, line and pattern class, never the matched text;
- coordination-narrative classes (the policy's narrative_patterns and the private map's narrative words) are reported over the
  tracked tree and fail only under TITANOS_HYGIENE_STRICT_NARRATIVE=1;
- a contamination fixture (a public document carrying a synthetic generated machine name and, when the environment or the map
  names one, the exporting machine and a private name) passes the actual source_package and sdist selection and fails the same
  scan in both copies, so the guard is exercised on real export paths, not on an inert fixture alone;
- the committed generated records (docs/execution_matrix.json, docs/roster.json) are fixed points of the generator's
  neutralization rules (tools/lab_paths), so the rebuild and the builder agree; the rules' last defence reduces a generated
  machine name that reaches a generator string to <windows-host> and is spelled identically to the policy's pattern class;
- the role labels are one mapping: every label of tools/lab_paths.HOST_ROLE_LABELS is documented in docs/MATRIX_KIT.md (Host
  labels), has the label form, is left alone by the rules, and every host the private map names maps to one of them.

This is a hygiene guard for lab locators only: not a secret scanner, not a legal or vendor review and not a
release-readiness claim. It reads the repository and writes only inside a temporary directory; no server, model or browser runs.
"""
import ast
import fnmatch
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tests'))
from check_source_package import SOURCE_PACKAGE, sdist_names  # noqa: E402  (the one sdist rule reader; the real source_package AST)
import lab_paths  # noqa: E402
from lab_paths import neutralize_records, neutralize_reference  # noqa: E402

_spec = importlib.util.spec_from_file_location('public_export_build_tool', ROOT / 'tools' / 'build_binary.py')
build_binary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_binary)

POLICY = json.loads((ROOT / 'tools' / 'public_export.json').read_text(encoding='utf-8'))
PATTERNS = {name: re.compile(pattern) for name, pattern in POLICY['detection_patterns'].items()}
NARRATIVE_PATTERNS = {name: re.compile(pattern) for name, pattern in POLICY.get('narrative_patterns', {}).items()}
ALLOWLIST = {entry['path']: set(entry['literals']) for entry in POLICY['fixture_allowlist']}
# The policy file defines the allowlisted literals, so it necessarily carries them (in their JSON-escaped spelling): exactly those
# literals are tolerated there.
ALLOWLIST['tools/public_export.json'] = {form for literal in set().union(*ALLOWLIST.values())
                                         for form in (literal, json.dumps(literal)[1:-1])}
# The exporting machine's own name is read from the environment at scan time and never written into the repository; it is a
# detection class of its own with no allowlist (tools/public_export.json machine_name_policy).
MACHINE_NAMES = {value.strip() for value in (os.environ.get('COMPUTERNAME'), os.environ.get('HOSTNAME'), platform.node())
                 if value and len(value.strip()) >= 3 and value.strip().casefold().split('.')[0] != 'localhost'}
MACHINE_NAME_CLASS = 'exporting_machine_name'
# the class is always built (platform.node() is the fallback) so a scrubbed environment cannot silently drop it; whole-word, any
# ASCII case, hyphen or underscore neighbours count as boundaries
MACHINE_NAME_PATTERN = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in sorted(MACHINE_NAMES)) + r')(?![A-Za-z0-9])',
                                  re.IGNORECASE | re.ASCII) if MACHINE_NAMES else None
GENERATED_CLASS = 'windows_generated_machine_name'
# The private host map (tools/public_export.json private_files): its host keys and deny list are the lab's own names, a hard class
# with no allowlist; its narrative words and phrase keys are reported. Both are empty when the map is absent (CI, a public tree).
PRIVATE_CLASS = 'private_denylist'
PRIVATE_NAMES = lab_paths.private_names()
PRIVATE_PATTERN = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in sorted(PRIVATE_NAMES, key=len, reverse=True)) + r')(?![A-Za-z0-9])',
                             re.IGNORECASE | re.ASCII) if PRIVATE_NAMES else None
NARRATIVE_CLASS = 'coordination_narrative'
NARRATIVE_WORDS = lab_paths.narrative_words()
NARRATIVE_PATTERN = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in sorted(NARRATIVE_WORDS, key=len, reverse=True)) + r')(?![A-Za-z0-9])') if NARRATIVE_WORDS else None
STRICT_NARRATIVE = os.environ.get('TITANOS_HYGIENE_STRICT_NARRATIVE', '').strip() == '1'
PUBLIC_TREE_MARKER = POLICY['tree_kinds'].get('marker', '.titanos-public-tree')


def tree_kind():
    """'public' or 'private' (tools/public_export.json tree_kinds.selection): the environment first, then the marker file, else private."""
    kind = os.environ.get('TITANOS_TREE_KIND', '').strip().casefold()
    if kind in ('public', 'private'):
        return kind
    return 'public' if (ROOT / PUBLIC_TREE_MARKER).exists() else 'private'


TREE_KIND = tree_kind()
PRIVATE_FILES = list(POLICY.get('private_files', []))
NEVER_TRACKED = list(POLICY.get('never_tracked', []))
EXCLUDED_PREFIXES = tuple(path.casefold() + '/' for path in POLICY['excluded_paths'])
EXCLUDED_FILES = {path.casefold() for path in POLICY['excluded_paths']} | {path.casefold() for path in PRIVATE_FILES}


def excluded(name):
    lower = name.casefold()
    return lower in EXCLUDED_FILES or lower.startswith(EXCLUDED_PREFIXES)


def tracked_files():
    """{relative name: path} of every tracked file (git ls-files); without git, or when git sees nothing here (a copy that is not a
    checkout), every file under the root outside .git and caches. Never empty."""
    names = []
    try:
        out = subprocess.run(['git', 'ls-files', '-z'], cwd=str(ROOT), capture_output=True, check=True).stdout.decode('utf-8')
        names = [n for n in out.split('\0') if n]
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        names = []
    if not names:
        skip = {'.git', '__pycache__', '.venv', 'venv', 'build', 'dist', 'release', '.pytest_cache'}
        names = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts[:-1]) & skip)]
    files = {n: ROOT / n for n in names if (ROOT / n).is_file()}
    assert files, 'no files to scan'
    return files


BINARY_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.woff', '.woff2', '.ttf', '.pdf', '.zip', '.gz', '.wav', '.mp3', '.mp4', '.onnx', '.bin', '.pyc'}
SKIPPED = []   # (relative name) of files the scan could not decode as UTF-8; a test asserts they are known binary kinds


def read_text(name, path, skipped):
    try:
        return path.read_bytes().decode('utf-8')
    except UnicodeDecodeError:
        skipped.append(name)
        return None


def scan(files, skipped=None):
    """[(relative name, line, pattern class)] for every hit outside the fixture allowlist, plus every whole-word hit of the exporting
    machine's own name (class exporting_machine_name) and of a private-map name (class private_denylist), neither ever tolerated;
    files that are not UTF-8 text are recorded in `skipped` (default SKIPPED; a test asserts they are known binary kinds, so a UTF-16
    document cannot slip past the scan silently). The tuples carry no matched text."""
    hits = []
    skipped = SKIPPED if skipped is None else skipped
    for name, path in sorted(files.items()):
        text = read_text(name, path, skipped)
        if text is None:
            continue
        allowed = ALLOWLIST.get(name, set())
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                if match.group(0) not in allowed:
                    hits.append((name, text.count('\n', 0, match.start()) + 1, label))
        for pattern, label in ((MACHINE_NAME_PATTERN, MACHINE_NAME_CLASS), (PRIVATE_PATTERN, PRIVATE_CLASS)):
            if pattern is not None:   # the machine-name pattern is None only when even the interpreter reports no node name
                for match in pattern.finditer(text):
                    hits.append((name, text.count('\n', 0, match.start()) + 1, label))
    return hits


def narrative_hits(files):
    """[(relative name, line, class)] for the reported classes: the policy's narrative_patterns by name and the private map's words
    as coordination_narrative. No matched text."""
    hits = []
    for name, path in sorted(files.items()):
        text = read_text(name, path, [])
        if text is None:
            continue
        for label, pattern in NARRATIVE_PATTERNS.items():
            for match in pattern.finditer(text):
                hits.append((name, text.count('\n', 0, match.start()) + 1, label))
        if NARRATIVE_PATTERN is not None:
            for match in NARRATIVE_PATTERN.finditer(text):
                hits.append((name, text.count('\n', 0, match.start()) + 1, NARRATIVE_CLASS))
    return hits


def report(hits):
    """The failure text for a list of hits: file, line and class only."""
    return '; '.join(f'{name}:{line} {kind}' for name, line, kind in hits[:40])


def export_from(tree, destination):
    """Run the real tools/build_binary.source_package (its AST, ROOT rebound to `tree`) into `destination`; returns {name: path}
    of the SOURCE/ copy it wrote."""
    scope = {'ROOT': tree, 'Path': Path, 'shutil': shutil, 'os': os, 'stat': stat, 'fnmatch': fnmatch, 'json': json}
    exec(compile(ast.Module(body=[SOURCE_PACKAGE], type_ignores=[]), 'actual source_package', 'exec'), scope)
    scope['source_package'](destination)
    source = destination / 'SOURCE'
    return {path.relative_to(source).as_posix(): path for path in source.rglob('*') if path.is_file()}


class PublicExportHygiene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='titanos-public-export-')
        built = Path(cls.temporary.name) / 'build'
        built.mkdir()
        build_binary.source_package(built)
        source = built / 'SOURCE'
        cls.built = built
        cls.exported = {path.relative_to(source).as_posix(): path for path in source.rglob('*') if path.is_file()}
        cls.manifest = build_binary.package_manifest(built)
        cls.sdist = {name: ROOT / name for name in sdist_names(ROOT, preselected=False)}
        cls.tracked = tracked_files()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_source_copy_passes_the_builder_distribution_scan(self):
        """The builder's own gate (build_binary.distribution_hygiene: the policy classes plus its builder-local classes, every
        file as bytes, SOURCE/ in the owned zone where every class refuses) over the real source_package copy. The tracked-tree
        scan below does not run the builder-local classes, so this is the check that a hygiene pattern, a planted test hit or a
        documentation example does not match its own text and refuse every future build at the gate (the 2026-10-05 lesson:
        tools/public_export.json spells one of its classes as /Users[/] for exactly that reason)."""
        report = build_binary.distribution_hygiene(self.built)
        self.assertEqual(report['files_scanned'], len(self.exported))
        self.assertEqual(report['zones'], {'owned': len(self.exported), 'third_party': 0}, 'SOURCE/ is wholly the owned zone')
        self.assertEqual(report['refused'], [], 'the builder would refuse this source copy at the gate: ' + build_binary.hygiene_report(report['refused']))
        self.assertEqual(report['reported'], [])

    def test_policy_names_records_that_exist_and_manifest_in_carries_the_same_lines(self):
        historical = set(POLICY.get('historical_exclusions', []))
        self.assertTrue(historical <= set(POLICY['excluded_paths']))
        self.assertIn(TREE_KIND, POLICY['tree_kinds'], 'the tree kind is one the policy describes')
        if TREE_KIND == 'private':
            # the private tree holds the records so the exclusions are exercised, not vacuous
            for path in set(POLICY['excluded_paths']) - historical:
                self.assertTrue((ROOT / path).exists(), path + ' is named by the policy but absent; the exclusion would be untested')
            self.assertTrue(any((ROOT / 'docs' / 'evidence').iterdir()), 'docs/evidence is empty; the exclusion would be untested')
            self.assertTrue((ROOT / 'tools/matrix_kit/legacy_results.json').is_file())
        else:
            # a public tree carries none of them
            for path in list(POLICY['excluded_paths']) + PRIVATE_FILES:
                self.assertFalse((ROOT / path).exists(), path + ' exists in a public tree')
        lines = {line.strip() for line in (ROOT / 'MANIFEST.in').read_text(encoding='utf-8').splitlines()}
        for path in POLICY['excluded_paths']:
            self.assertTrue({'prune ' + path, 'exclude ' + path} & lines, 'MANIFEST.in does not carry ' + path)
        self.assertIn('exclude tools/matrix_kit/legacy_results.json', lines)
        for entry in POLICY['fixture_allowlist']:
            self.assertTrue(entry['reason'].strip(), 'an allowlist entry needs a reason')
            self.assertIn(entry['path'], self.exported, 'allowlisted fixture file is not exported: ' + entry['path'])

    def test_private_map_is_never_tracked_and_never_exported(self):
        # the map that names the lab's machines is the very leak the guard exists for: it is never tracked (any tree kind), git
        # ignores it when it sits at its default path, and neither export copy carries it
        self.assertIn('tools/lab_hosts.private.json', NEVER_TRACKED)
        for name in NEVER_TRACKED:
            self.assertNotIn(name, self.tracked, name + ' is tracked')
            self.assertNotIn(name, self.exported, name + ' is exported')
            self.assertNotIn(name, self.sdist, name + ' is in the sdist')
            if (ROOT / name).exists():
                ignored = subprocess.run(['git', 'check-ignore', '-q', name], cwd=str(ROOT), capture_output=True)
                self.assertEqual(ignored.returncode, 0, name + ' exists but is not gitignored')
        for name in PRIVATE_FILES:
            self.assertNotIn(name, self.exported, name + ' is exported')
            self.assertNotIn(name, self.sdist, name + ' is in the sdist')

    def test_host_role_labels_are_one_mapping(self):
        labels = lab_paths.HOST_ROLE_LABELS
        self.assertTrue(labels)
        kit_doc = (ROOT / 'docs' / 'MATRIX_KIT.md').read_text(encoding='utf-8')
        section = kit_doc.split('## Host labels', 1)
        self.assertEqual(len(section), 2, 'docs/MATRIX_KIT.md documents the labels under a Host labels heading')
        table = section[1].split('\n## ', 1)[0]
        documented = set(re.findall(r'^\| `([a-z0-9-]+)` \|', table, re.M))
        self.assertEqual(documented, set(labels), 'docs/MATRIX_KIT.md Host labels table and tools/lab_paths.HOST_ROLE_LABELS disagree')
        self.assertIn('HOST_ROLE_LABELS', POLICY['parameterization']['host_role_labels'], 'the policy points at the single mapping')
        for label, role in labels.items():
            self.assertRegex(label, lab_paths.ROLE_LABEL_FORM, label)
            self.assertTrue(role.strip(), label + ' needs a hardware role')
            self.assertIsNone(PATTERNS[GENERATED_CLASS].search(label), label)
            if PRIVATE_PATTERN is not None:
                self.assertIsNone(PRIVATE_PATTERN.search(label), label + ' is a private name')
            self.assertEqual(neutralize_reference('fleet-' + label + '-20260924'), 'fleet-' + label + '-20260924')
        # the two labels the earlier passes introduced stay
        self.assertTrue({'turing-laptop-win-x64', 'intel-arc-win-x64-openvino'} <= set(labels))
        # the private map, when present, points only at these labels and never uses a label as a key
        private = lab_paths.PRIVATE
        for name, label in private['hosts'].items():
            self.assertIn(label, labels, 'private map maps a host to an unknown label')
            self.assertNotIn(name, labels, 'a role label is a key of the private map')
        for name in private['hosts']:
            self.assertIsNotNone(PRIVATE_PATTERN and PRIVATE_PATTERN.search(' ' + name + ' '), 'the denylist covers every mapped host')

    def test_detection_pattern_classes_are_present_and_match_canonical_samples(self):
        # the scan must not narrow silently: every class stays declared and each still matches a synthetic sample (neutral parts only)
        self.assertTrue({'windows_user_profile', 'posix_user_profile', 'rfc1918_ipv4', 'lab_drive_titanos', 'lab_drive_codex', 'lab_codex_worktree', 'linux_home',
                         'lab_linux_host', 'drive_letter_pair', 'ssh_hostkey_fingerprint', GENERATED_CLASS} <= set(PATTERNS), sorted(PATTERNS))
        self.assertIn('workflow_id', NARRATIVE_PATTERNS)
        # samples are assembled from parts so this file carries no contiguous literal the scan would flag
        drive, users, lab, codex, home = 'X:', 'Users', 'Titanos', 'Codex', 'home'
        samples = {'windows_user_profile': [drive + '/' + users + '/example/file.txt', drive.lower() + chr(92) + users.lower() + chr(92) + 'example' + chr(92) + 'file.txt'],
                   'rfc1918_ipv4': ['.'.join(['10', '1', '2', '3']), '.'.join(['192', '168', '0', '9']), '.'.join(['172', '16', '5', '5'])],
                   'lab_drive_titanos': [drive + '/' + lab + 'Example/run', drive.lower() + chr(92) + lab.lower() + 'example' + chr(92) + 'run'],
                   'lab_drive_codex': [drive + '/' + codex + '/work', drive.lower() + chr(92) + codex.lower() + chr(92) + 'work'],
                   'lab_codex_worktree': ['<user-profile>/Documents/' + codex + '/2026-01-01/tree/work/x.json', drive + '/' + users + '/example/Documents/' + codex + ')', '<user-profile>/Documents/' + codex + '/)',
                                          # the MSYS, Cygwin and WSL spellings of a work tree under a user profile (the retained lab FFmpeg bundle's configure --prefix)
                                          '--prefix=/c/' + users + '/example/Documents/' + codex + '/2026-01-01/work/install', '/mnt/c/' + users + '/example/Documents/' + codex + ')', '/cygdrive/d/' + users + '/example/Documents/' + codex + '/x'],
                   'posix_user_profile': ['--prefix=/c/' + users + '/example/install', '/mnt/c/' + users + '/example/x', '/cygdrive/e/' + users + '/example', ' /E/' + users + '/example/y'],
                   'linux_home': ['/' + home + '/example/x.json', ' /' + home + '/account', '=/' + home + '/user/.cache'],
                   'lab_linux_host': ['lab-' + 'server-' + '001', 'PROJECT-EXAMPLE-' + 'SERVER-' + '123'],
                   'drive_letter_pair': ['the reachable ' + 'E:/' + 'C:' + ' roots', 'D:/' + 'E:'],
                   'ssh_hostkey_fingerprint': ['SHA256:' + 'A' * 43, 'key SHA256:' + 'a1B2' * 10 + 'xyz' + ' host']}
        samples['macos_user_home'] = ['/' + users + '/example/file.txt', 'path /' + users + '/someone']
        for label, texts in samples.items():
            for text in texts:
                self.assertTrue(PATTERNS[label].search(' ' + text), (label, text))
        self.assertTrue(NARRATIVE_PATTERNS['workflow_id'].search('review wf_' + '0123abcd' + '-1a2'))
        self.assertIsNone(PATTERNS['lab_drive_titanos'].search('http://' + lab.lower() + '.test/x'), 'a URL scheme must not match through its letters')
        self.assertIsNone(PATTERNS['rfc1918_ipv4'].search('.'.join(['11', '0', '0', '1']) + ' and ' + '.'.join(['193', '168', '1', '1'])))
        negatives = {'linux_home': ['<lab-records>/native/' + home + '/x.json', 'some' + home + '/x', '/' + home + '/', 'the ' + home + ' directory'],
                     'lab_linux_host': ['llama-server', 'server 5087', 'server-5087', 'llama-server-b10809', 'standalone_service.py', 'lab-server-12'],
                     'drive_letter_pair': ['C:/x', 'http://a/b', 'E:/models/x'],
                     'ssh_hostkey_fingerprint': ['SHA256:abc', 'sha256 24a474ca51e6', 'SHA256:' + 'A' * 44],
                     'lab_codex_worktree': ['<lab-records>/work/x', 'Documents/' + codex + 'Reader/x', 'tools/' + codex.lower() + '/x'],
                     # a neutral install prefix, a Linux home, a folder named Users deeper than a drive letter, the macOS form (its own class) and
                     # a drive letter under /cygdrive without the Users segment do not match the POSIX user-profile class
                     'posix_user_profile': ['--prefix=/opt/titanos/ffmpeg', '/' + home + '/example/x', '/usr/' + users + '/x', '/' + users + '/example', drive + '/' + users + '/x', '/cygdrive/' + users + '/x', '/mnt/c/' + users + '/']}
        for label, texts in negatives.items():
            for text in texts:
                self.assertIsNone(PATTERNS[label].search(text), (label, text))
        self.assertIsNone(NARRATIVE_PATTERNS['workflow_id'].search('wf_' + 'zz'))

    def test_generated_machine_name_class_positives_and_negatives(self):
        # synthetic names are assembled from parts so this file carries no contiguous generated-form literal; the real machine
        # name is never written here
        pattern = PATTERNS[GENERATED_CLASS]
        desk, lap, win = 'DESK' + 'TOP', 'LAP' + 'TOP', 'W' + 'IN'
        seven, eleven = 'AB12CD3', 'Q1W2E3R4T5Y'
        names = [desk + '-' + seven, lap + '-' + seven, win + '-' + eleven]
        for name in names:
            for form in (name, name.lower(), name.capitalize(), name[:4].lower() + name[4:]):
                for embedded in (form, 'cells on ' + form + ';', '"' + form + '"', 'fleet-' + form + '-20260924/run/receipts',
                                 '(' + form + ')', 'host=' + form + ',', 'run_' + form + '_x', 'receipt_' + form + '.json', chr(92) + chr(92) + form + chr(92) + 'share'):
                    self.assertTrue(pattern.search(embedded), (name, embedded[:4] + '...'))
        negatives = ['desktop', 'Desktop-class', 'win-x64-cpu', 'win-arm64-qnn-npu', lap + '-20260924', 'ROLE-' + lap + '-20260925',
                     win + '-32', desk + '-ABC', desk + '-ABCDEFGH', desk + '_' + seven, desk + '-' + seven + 'Z', 'X' + desk + '-' + seven, desk + '-' + 'ÄB12CD3',
                     'Qwen2.5-0.5B-Instruct', 'all-MiniLM-L6-v2', 'intel-arc-win-x64-openvino', 'sherpa-onnx-1.13.8', desk + '-/' + lap + '-']
        for text in negatives:
            self.assertIsNone(pattern.search(text), text)
        # the policy carries no exception for the class or for a private name: no allowlist literal is of the generated form or a
        # name the private map knows; the message names the file, never the literal
        for entry in POLICY['fixture_allowlist']:
            for literal in entry['literals']:
                self.assertIsNone(pattern.search(literal), entry['path'] + ' allowlists a generated Windows machine name')
                if PRIVATE_PATTERN is not None:
                    self.assertIsNone(PRIVATE_PATTERN.search(literal), entry['path'] + ' allowlists a private name')

    def test_lab_paths_reduces_a_generated_machine_name_to_the_windows_host_label(self):
        # the builder's last defence (tools/lab_paths.neutralize_reference): a generated name that reaches a generator string is
        # reduced to <windows-host>; the class is spelled identically to the policy's, so the two cannot drift. The synthetic name is
        # assembled from parts so this file carries no generated-form literal
        self.assertEqual(lab_paths.WINDOWS_GENERATED_HOST.pattern, POLICY['detection_patterns'][GENERATED_CLASS],
                         'tools/lab_paths and the policy must spell the generated machine-name class identically')
        self.assertEqual(lab_paths.WINDOWS_HOST, '<windows-host>')
        for label in (lab_paths.WINDOWS_HOST, lab_paths.LAB_RECORDS, lab_paths.USER_PROFILE, lab_paths.LINUX_HOME):
            self.assertIn(label, POLICY['parameterization']['labels'], 'the policy documents the label ' + label)
        desk, lap, win = 'DESK' + 'TOP', 'LAP' + 'TOP', 'W' + 'IN'
        drive, lab = 'X:', 'Titanos'
        for name in (desk + '-ZX9Q1W7', lap + '-AB12CD3', win + '-Q1W2E3R4T5Y'):
            for form in (name, name.lower(), name.capitalize()):
                self.assertEqual(neutralize_reference('fleet-' + form + '-20260924'), 'fleet-<windows-host>-20260924')
                self.assertEqual(neutralize_reference('matrix-kit:fleet-' + form + '-dml-20260924'), 'matrix-kit:fleet-<windows-host>-dml-20260924')
                self.assertEqual(neutralize_reference('fleet-' + form + '-20260925-receipts-tiny'), 'fleet-<windows-host>-20260925-receipts-tiny')
                # the path rules and the machine-name rule compose: a run folder named after the machine under a lab records root
                self.assertEqual(neutralize_reference(drive + '/' + lab + 'Example/runs/fleet-' + form + '-20260924/run/receipts/a.json'),
                                 '<lab-records>/runs/fleet-<windows-host>-20260924/run/receipts/a.json')
                self.assertEqual(neutralize_reference('cells on ' + form + '; 28 passes'), 'cells on <windows-host>; 28 passes')
        # the other rules: Linux homes, agent work trees (already behind <user-profile> or still drive-rooted), bare drive-letter pairs
        home, codex, users = 'home', 'Codex', 'Users'
        self.assertEqual(neutralize_reference('local weights /' + home + '/account/models/x'), 'local weights <linux-home>/models/x')
        self.assertEqual(neutralize_reference('<lab-records>/native/' + home + '/native-runtime.json'), '<lab-records>/native/' + home + '/native-runtime.json')
        self.assertEqual(neutralize_reference('<user-profile>/Documents/' + codex + '/2026-01-01/tree-name/work/api/sources.json'), '<lab-records>/work/api/sources.json')
        self.assertEqual(neutralize_reference(drive + '/' + users + '/account/Documents/' + codex + '/2026-01-01/tree-name/work/x.json'), '<lab-records>/work/x.json')
        self.assertEqual(neutralize_reference('(<user-profile>/Documents/' + codex + '/)'), '(<lab-records>/)')
        self.assertEqual(neutralize_reference('under the reachable ' + 'E:/' + 'C:' + ' roots'), 'under the reachable lab records roots')
        # the existing negatives are left unchanged: platform slot labels, dated labels, role labels, model identifiers, product words
        for text in ('win-x64-cpu', 'win-arm64-qnn-npu', 'fleet-turing-laptop-win-x64-20260924', 'fleet-rtx5070-desktop-win-x64-20260925-receipts-detr3',
                     'fleet-intel-arc-win-x64-openvino-20260924', 'matrix-kit:fleet-linux-cuda-server-x64-20260924', 'ROLE-' + lap + '-20260924', lap + '-20260924', desk + '-ABC', desk + '_' + 'ZX9Q1W7',
                     'Qwen2.5-0.5B-Instruct', 'all-MiniLM-L6-v2', 'sherpa-onnx-1.13.8', 'the Turing laptop (RTX 2070 Max-Q, Windows x64)', 'desktop', 'llama-server', 'E:/models/x'):
            self.assertEqual(neutralize_reference(text), text, text)
        self.assertEqual(neutralize_reference(42), 42)
        self.assertEqual(neutralize_records({'a': [None, 3, 'win-x64-cpu']}), {'a': [None, 3, 'win-x64-cpu']})
        # idempotent on the committed matrix (a fixed point of the rules, asserted below; applying them twice changes nothing more)
        data = json.loads((ROOT / 'docs' / 'execution_matrix.json').read_text(encoding='utf-8'))
        once = neutralize_records(data)
        self.assertEqual(neutralize_records(once), once)

    def test_public_matrix_views_carry_no_generated_machine_name_without_any_allowlist(self):
        # the fleet machines once named by a generated name are published under stable role labels: the raw class, with no allowlist
        # consulted, finds nothing in the public matrix views of either copy, and no allowlist entry names one of those views
        pattern = PATTERNS[GENERATED_CLASS]
        views = ('docs/CAPABILITY_MATRIX.md', 'docs/execution_matrix.json', 'docs/EXECUTION_MATRIX.md', 'docs/execution_matrix.csv', 'docs/execution_matrix.html')
        for label, names in (('native SOURCE/', self.exported), ('sdist', self.sdist)):
            # the native copy carries all five; the sdist selects docs by suffix (MANIFEST.in recursive-include docs *.md *.json *.txt),
            # so it must carry the Markdown and JSON views and is scanned over whichever views it carries
            required = views if label == 'native SOURCE/' else tuple(view for view in views if view.endswith(('.md', '.json')))
            for view in required:
                self.assertIn(view, names, label + ' must carry the public matrix view ' + view)
            for view in (view for view in views if view in names):
                text = names[view].read_text(encoding='utf-8')
                lines = [text.count('\n', 0, match.start()) + 1 for match in pattern.finditer(text)]
                self.assertEqual(lines, [], f'{label} {view} carries a generated Windows machine name at lines {lines[:10]}')
        self.assertFalse({entry['path'] for entry in POLICY['fixture_allowlist']} & set(views), 'no public matrix view may carry an allowlist entry')
        # the two capability rows that named the Turing laptop by machine name now name it by role
        rows = {line[:9]: line for line in self.exported['docs/CAPABILITY_MATRIX.md'].read_text(encoding='utf-8').splitlines() if line.startswith('| [E-09')}
        self.assertIn('the Turing laptop (RTX 2070 Max-Q, Windows x64)', rows['| [E-096]'])
        self.assertIn('the Turing laptop', rows['| [E-098]'])

    def test_skipped_files_are_known_binary_kinds(self):
        skipped = []
        scan(self.exported, skipped)
        unexpected = sorted(name for name in skipped if Path(name).suffix.lower() not in BINARY_SUFFIXES)
        self.assertEqual(unexpected, [], 'undecodable files that are not known binary kinds must not be skipped silently')

    def test_manifest_binding_is_what_main_writes(self):
        # the build writes MANIFEST.json from package_manifest(built); assert that in main's source so the walk below is not tautological
        tree = ast.parse((ROOT / 'tools/build_binary.py').read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        source = ast.get_source_segment((ROOT / 'tools/build_binary.py').read_text(encoding='utf-8'), main)
        self.assertRegex(source, r"manifest\s*=\s*package_manifest\(built\)")
        self.assertRegex(source, r"'MANIFEST\.json'\)\.write_text\(json\.dumps\(manifest")

    def test_native_manifest_covers_exactly_the_exported_set(self):
        self.assertTrue(self.exported)
        self.assertEqual({key for key in self.manifest if key.startswith('SOURCE/')},
                         {'SOURCE/' + name for name in self.exported})
        self.assertEqual(set(self.manifest), {'SOURCE/' + name for name in self.exported},
                         'the temporary build holds nothing but the source copy')
        for name, path in self.exported.items():
            self.assertEqual(self.manifest['SOURCE/' + name]['bytes'], path.stat().st_size, name)

    def test_both_copies_exclude_the_lab_operational_records_and_agree(self):
        for label, names in (('native SOURCE/', self.exported), ('sdist', self.sdist)):
            offenders = sorted(name for name in names if excluded(name))
            self.assertFalse(offenders, label + ' includes excluded lab records: ' + repr(offenders[:20]))
            self.assertIn('tools/public_export.json', names, label + ' must carry the policy it was built by')
            self.assertIn('docs/CAPABILITY_MATRIX.md', names, label)
            self.assertIn('MANIFEST.in', names, label)
        for name in ('docs/evidence/2026-09-22-matrix-push.md', 'docs/CLUSTER_HANDOFF.md', 'tools/matrix_kit/legacy_results.json'):
            if TREE_KIND == 'private':
                self.assertTrue((ROOT / name).is_file(), name + ' is present in the default tree kind')
            else:
                self.assertFalse((ROOT / name).exists(), name + ' must be absent from a public tree')
            self.assertNotIn(name, self.exported)
            self.assertNotIn(name, self.sdist)

    def test_no_private_reference_survives_in_either_copy(self):
        for label, names in (('native SOURCE/', self.exported), ('sdist', self.sdist)):
            hits = scan(names)
            self.assertFalse(hits, label + ' carries lab locators (file, line, pattern class): ' + report(hits))

    def test_tracked_tree_outside_excluded_records_carries_no_private_reference(self):
        # hand-written documents are a guarded surface too: every tracked file, minus (in a private tree only) the excluded records
        # and private files, passes the same scan; in a public tree nothing is excluded
        files = self.tracked if TREE_KIND == 'public' else {name: path for name, path in self.tracked.items() if not excluded(name)}
        self.assertTrue(files)
        self.assertTrue(any(name.startswith('docs/') for name in files), 'the tracked set includes the documents')
        skipped = []
        hits = scan(files, skipped)
        unexpected = sorted(name for name in skipped if Path(name).suffix.lower() not in BINARY_SUFFIXES)
        self.assertEqual(unexpected, [], 'undecodable tracked files that are not known binary kinds must not be skipped silently')
        self.assertFalse(hits, f'{TREE_KIND} tree carries lab locators outside the excluded records (file, line, class): ' + report(hits))
        # the reported classes: named and counted per file, never quoted; failing only when asked to
        reported = narrative_hits(files)
        if reported:
            counts = {}
            for name, _line, kind in reported:
                counts[(name, kind)] = counts.get((name, kind), 0) + 1
            summary = '; '.join(f'{name} {kind} x{count}' for (name, kind), count in sorted(counts.items()))
            print(f'\n[{NARRATIVE_CLASS}] reported, not failing (TITANOS_HYGIENE_STRICT_NARRATIVE=1 to fail): {summary}', file=sys.stderr)
            if STRICT_NARRATIVE:
                self.fail('coordination narrative in the tracked tree: ' + summary)

    def test_exporting_machine_name_absent_from_both_copies_and_never_allowlisted(self):
        # the lab computer name is a private identifier; it is read from the environment at scan time, never written into the
        # repository, and no allowlist may tolerate it
        self.assertTrue(MACHINE_NAMES, 'the interpreter reports no node name; the machine-name class cannot be built')
        names = {n.casefold() for n in MACHINE_NAMES}
        for entry in POLICY['fixture_allowlist']:
            for literal in entry['literals']:
                # the message never carries the literal or the names
                self.assertFalse(literal.casefold() in names, entry['path'] + ' allowlists the exporting machine')
        for label, names in (('native SOURCE/', self.exported), ('sdist', self.sdist)):
            hits = [hit for hit in scan(names) if hit[2] == MACHINE_NAME_CLASS]
            self.assertEqual(hits, [], label + ' names the exporting machine (file, line, class): ' + report(hits))

    def test_repaired_capability_summaries_are_neutral(self):
        # the three matrix-kit campaign rows of this workstation (E-090..E-092) name it by role, not by machine name
        text = self.exported['docs/CAPABILITY_MATRIX.md'].read_text(encoding='utf-8')
        rows = [line for line in text.splitlines() if line.startswith('| [E-090]') or line.startswith('| [E-091]') or line.startswith('| [E-092]')]
        self.assertEqual(len(rows), 3, 'the three workstation campaign rows are present in the public document')
        for row in rows:
            # the message names the row, never its text
            self.assertTrue('cells on this workstation;' in row, row[:10] + ' summary does not name the workstation by role')
        self.assertEqual([hit for hit in scan({'docs/CAPABILITY_MATRIX.md': self.exported['docs/CAPABILITY_MATRIX.md']})], [])

    def test_contaminated_public_document_survives_selection_but_fails_hygiene_in_both_copies(self):
        """A public document carrying a machine name passes the actual source_package and the actual sdist selection (correct
        inclusion) and is reported by the scan in both copies with file, line and class only (unsafe content). Exercised on a
        temporary tree built from the real export, never on the repository."""
        synthetic = 'DESK' + 'TOP' + '-' + 'ZX9Q1W7'   # assembled so this file carries no generated-form literal
        fixture = 'docs/PUBLIC_DOC_FIXTURE.md'
        with tempfile.TemporaryDirectory(prefix='titanos-export-contamination-') as temporary:
            tree = Path(temporary) / 'tree'
            for name, path in self.exported.items():
                target = tree / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            lines = ['# Matrix kit campaign', '', 'Summary: 28 pass / 8 fail / 0 n/a cells on ' + synthetic + '; 28 passes verifier-upheld.']
            expected = [(fixture, 3, GENERATED_CLASS)]
            self.assertTrue(MACHINE_NAMES, 'the machine-name branch of this test needs a node name')
            lines.append('Summary: 12 pass / 12 fail cells on ' + sorted(MACHINE_NAMES)[0] + '; 12 passes verifier-upheld.')
            expected.append((fixture, 4, MACHINE_NAME_CLASS))
            if PATTERNS[GENERATED_CLASS].search(sorted(MACHINE_NAMES)[0]):
                expected.append((fixture, 4, GENERATED_CLASS))
            if PRIVATE_PATTERN is not None and PRIVATE_PATTERN.search(sorted(MACHINE_NAMES)[0]):
                expected.append((fixture, 4, PRIVATE_CLASS))
            # a private-map name that is neither of the generated form nor the exporting machine, when the map is present
            private = [n for n in sorted(PRIVATE_NAMES) if not PATTERNS[GENERATED_CLASS].search(n) and n.casefold() not in {m.casefold() for m in MACHINE_NAMES}
                       and not any(p.search(n) for p in PATTERNS.values())]
            if private:
                lines.append('Summary: 7 pass cells on ' + private[0] + '; 7 passes verifier-upheld.')
                expected.append((fixture, 5, PRIVATE_CLASS))
            (tree / fixture).write_text('\n'.join(lines) + '\n', encoding='utf-8')
            # private material that the real filters must drop from both copies, so a copy that skipped the real source_package or
            # MANIFEST.in rules is caught here rather than passing vacuously
            private_files = {'docs/evidence/PRIVATE_FIXTURE.md': 'lab record ' + synthetic, 'docs/PRIVATE_FIXTURE.log': 'operation log ' + synthetic,
                             'tools/matrix_kit/legacy_results.json': '{"' + synthetic + '": 1}', 'docs/CLUSTER_HANDOFF.md': 'handoff ' + synthetic}
            for name, content in private_files.items():
                (tree / name).parent.mkdir(parents=True, exist_ok=True)
                (tree / name).write_text(content + '\n', encoding='utf-8')
            built = Path(temporary) / 'build'
            built.mkdir()
            exported = export_from(tree, built)
            sdist = {name: tree / name for name in sdist_names(tree, preselected=False)}
            for label, names in (('native SOURCE/', exported), ('sdist', sdist)):
                self.assertIn(fixture, names, label + ' selection must include an ordinary public document')
                self.assertIn('docs/CAPABILITY_MATRIX.md', names, label)
                for name in private_files:
                    self.assertNotIn(name, names, label + ' selection must drop the private fixture ' + name)
                hits = scan(names)
                for hit in expected:
                    self.assertIn(hit, hits, label + ' must report the contaminated document (file, line, class)')
                self.assertEqual(sorted(set(hits)), sorted(set(expected)), label + ': the only hits are the fixture lines: ' + report(hits))
                message = report(hits)
                self.assertNotIn(synthetic.casefold(), message.casefold(), 'diagnostics must not carry the matched text')
                for name in list(MACHINE_NAMES) + private:
                    self.assertNotIn(name.casefold(), message.casefold(), 'diagnostics must not carry the machine name')
                self.assertRegex(message, r'^' + re.escape(fixture) + r':\d+ [a-z_]+(; ' + re.escape(fixture) + r':\d+ [a-z_]+)*$')

    def test_allowlist_is_exact(self):
        """The allowlisted fixtures are present as written (so the allowlist cannot silently widen or rot)."""
        for entry in POLICY['fixture_allowlist']:
            text = (ROOT / entry['path']).read_text(encoding='utf-8')
            for literal in entry['literals']:
                self.assertIn(literal, text, f"{entry['path']} no longer carries an allowlisted literal")
                self.assertTrue(any(pattern.search(literal) for pattern in PATTERNS.values()),
                                f"an allowlisted literal in {entry['path']} matches no detection pattern; remove it")

    def test_generated_records_are_fixed_points_of_the_neutralization_rule(self):
        for name in ('docs/execution_matrix.json', 'docs/roster.json'):
            data = json.loads((ROOT / name).read_text(encoding='utf-8'))
            self.assertEqual(neutralize_records(data), data, name + ' carries a reference the generator rule would rewrite')
        matrix = json.loads((ROOT / 'docs' / 'execution_matrix.json').read_text(encoding='utf-8'))
        self.assertIn('lab-neutral references', matrix['basis'], 'the neutralization is recorded in the matrix basis (generator note, or the recorded one-off pass)')
        # every reader family and mode-cell source names its host by a role label or a local label, never by a generated name
        families = {reader for cell in matrix['cells'] for reader in cell.get('readers', [])}
        sources = {cell.get('source') for cell in matrix['mode_cells']} | {other.get('source') for cell in matrix['mode_cells'] for other in cell.get('other_sources', [])}
        for text in families | sources:
            if not text:
                continue
            self.assertIsNone(PATTERNS[GENERATED_CLASS].search(text), 'a reader family or mode-cell source carries a generated machine name')
            if PRIVATE_PATTERN is not None:
                self.assertIsNone(PRIVATE_PATTERN.search(text), 'a reader family or mode-cell source carries a private name')
            if text.startswith(('matrix-kit:fleet-', 'fleet-')):
                self.assertTrue(any(('fleet-' + label + '-') in text for label in lab_paths.HOST_ROLE_LABELS), 'a fleet run is labelled by a role label: ' + text[:60])


class DistributionHygiene(unittest.TestCase):
    """tools/build_binary.distribution_hygiene / verify_distribution_hygiene over a fake unsealed distribution: every file is
    scanned (text, byte strings, zip members); this lab's locators are refused in every zone, generic third-party residue is
    reported and not refused, and the report carries file, class, zone, offset and line but never the matched text. Planted
    strings are assembled from parts so this file carries no contiguous literal the tracked-tree scan or the builder's own gate
    (which also runs over this file inside SOURCE/; PublicExportHygiene.test_source_copy_passes_the_builder_distribution_scan)
    would flag."""
    NAMES = {'machines': ['example-build-host'], 'accounts': ['example-build-account'], 'private': []}
    DRIVE, USERS, LAB, CODEX, HOME = 'E:', 'Users', 'Titanos', 'Codex', 'home'
    LAB_HOST = 'project-' + 'example-' + 'server-' + '001'

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='titanos-dist-hygiene-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.dist = self.base / 'dist'
        self.private = self.base / 'private'
        import random
        noise = random.Random(7).randbytes(3000)   # binary padding: not UTF-8, no anchors by construction below
        noise = noise.replace(b'/', b'_').replace(b'\\', b'_').replace(b':', b'_').replace(b'-', b'_')
        self.noise = noise
        self.write('BUILD.json', b'{"profile": "full", "weights_included": false}\n')
        self.write('SOURCE/README.md', b'# Titanos\n\nA source copy with nothing to flag.\n')
        for name in ('tools/public_export.json', 'tests/check_music_failure_diagnostics.py'):   # the real allowlisted bytes
            self.write('SOURCE/' + name, (ROOT / name).read_bytes())
        self.write('_internal/titanos/web/style.css', b'body { color: black }\n')
        self.write('_internal/libexample.so', noise + b'\x00libexample 1.0\x00' + noise)
        # third-party residue of the kinds found in real builds: reported, never refused
        self.write('_internal/example-1.0.dist-info/METADATA', ('Name: example\nVersion: 1.0\nDescription: cache at /' + self.HOME + '/someone/.cache\n').encode())
        self.write('_internal/sympy-1.0.dist-info/licenses/AUTHORS', ('Alice <alice@example.test>\nBob <bob@' + 'DESKTOP-' + 'ABC1234' + '>\n').encode())
        archive = self.dist / '_internal/base_library.zip'
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, 'w') as zipped:
            zipped.writestr('ntpath.pyc', noise[:200] + ('(' + 'C:' + '/' + self.USERS + '/Barney' + ') == ').encode() + noise[200:400])
            zipped.writestr('clean.pyc', noise[400:900])

    def write(self, name, data):
        path = self.dist / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def scan(self):
        return build_binary.distribution_hygiene(self.dist, names=self.NAMES)

    def refused(self, report, file):
        return sorted({hit['class'] + ('/' + hit['reason'] if hit.get('reason') else '') for hit in report['refused'] if hit['file'] == file})

    def test_clean_distribution_passes_reports_residue_and_writes_the_summaries(self):
        report = self.scan()
        self.assertEqual(report['refused'], [])
        self.assertEqual({(row['file'], row['class']) for row in report['reported']},
                         {('_internal/example-1.0.dist-info/METADATA', 'linux_home'),
                          ('_internal/sympy-1.0.dist-info/licenses/AUTHORS', 'windows_generated_machine_name'),
                          ('_internal/base_library.zip', 'windows_user_profile'),
                          ('_internal/base_library.zip!ntpath.pyc', 'windows_user_profile')})
        self.assertEqual(report['files_scanned'], 9)
        self.assertEqual(report['archive_members_scanned'], 2)
        self.assertEqual(report['zones'], {'owned': 5, 'third_party': 4})
        summary = build_binary.verify_distribution_hygiene(self.dist, self.private)
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['refused_hits'], 0)
        self.assertEqual(summary['reported_third_party_residue'], {'files': 4, 'by_class': {'linux_home': 1, 'windows_generated_machine_name': 1, 'windows_user_profile': 2}})
        self.assertEqual(json.loads((self.dist / 'BUILD_HYGIENE.json').read_text(encoding='utf-8')), summary)
        private = json.loads((self.private / 'distribution-hygiene.json').read_text(encoding='utf-8'))
        self.assertEqual(private['refused'], [])
        self.assertNotIn(str(self.base), json.dumps(summary) + json.dumps(private))
        with self.assertRaisesRegex(ValueError, 'once'):
            build_binary.verify_distribution_hygiene(self.dist, self.private)
        with self.assertRaisesRegex(ValueError, 'outside'):
            build_binary.verify_distribution_hygiene(self.dist, self.dist / 'private')

    def test_pip_provenance_with_a_lab_records_url_is_refused_in_a_runtime(self):
        name = 'runtimes/ace-step-1.5/python/Lib/site-packages/torch-2.10.0.dist-info/direct_url.json'
        self.write(name, ('{"dir_info": {}, "url": "file:///' + self.DRIVE + '/' + self.LAB + 'Frozen-20260922/ext-models/x"}\n').encode())
        report = self.scan()
        self.assertEqual(self.refused(report, name), ['lab_drive_titanos', 'local_file_url'])
        hit = next(hit for hit in report['refused'] if hit['class'] == 'lab_drive_titanos')
        self.assertEqual((hit['zone'], hit['line']), ('owned', 1))
        with self.assertRaisesRegex(ValueError, r'refused 2 hit\(s\) in 1 file\(s\).*direct_url\.json \[owned\] lab_drive_titanos x1 \(first:1\), local_file_url x1'):
            build_binary.verify_distribution_hygiene(self.dist, self.private)
        self.assertFalse((self.dist / 'BUILD_HYGIENE.json').exists())
        self.assertTrue((self.private / 'distribution-hygiene.json').is_file(), 'the full report is retained on refusal')

    def test_lab_build_prefix_inside_native_binaries_is_refused_in_both_zones(self):
        prefix = ('--prefix=/' + self.HOME + '/' + self.LAB_HOST + '/titanos-validation/ffmpeg').encode()
        payload = self.noise + b'\x00' + prefix + b'\x00' + self.noise
        self.write('_internal/native/ffmpeg/libavcodec.so', payload)            # the lab-built bundle: owned zone
        self.write('_internal/libavcodec.so.63', payload)                       # the collected copy: third-party zone
        report = self.scan()
        self.assertEqual(self.refused(report, '_internal/native/ffmpeg/libavcodec.so'), ['lab_linux_host', 'linux_home'])
        self.assertEqual(self.refused(report, '_internal/libavcodec.so.63'), ['lab_linux_host', 'linux_home/denied_account'])
        hit = next(hit for hit in report['refused'] if hit['file'] == '_internal/libavcodec.so.63' and hit['class'] == 'linux_home')
        self.assertEqual(hit['offset'], len(self.noise) + 1 + len(b'--prefix='))
        self.assertIsNone(hit['line'])
        self.assertNotIn(self.LAB_HOST, json.dumps(report))
        self.assertNotIn(self.LAB_HOST, build_binary.hygiene_report(report['refused']))

    def test_building_account_and_machine_are_refused_in_third_party_payload(self):
        worktree = ('C:' + '/' + self.USERS + '/' + self.NAMES['accounts'][0] + '/Documents/' + self.CODEX + '/2026-09-11/work/ffmpeg.c').encode()
        self.write('_internal/avcodec-63.dll', self.noise + worktree + self.noise)
        self.write('_internal/vendor/notes.txt', ('built on ' + self.NAMES['machines'][0] + ' last week\n').encode())
        self.write('_internal/vendor/other.txt', ('built on ' + self.NAMES['machines'][0] + 'x, not the machine\n').encode())
        report = self.scan()
        self.assertEqual(self.refused(report, '_internal/avcodec-63.dll'), ['lab_codex_worktree', 'windows_user_profile/denied_account'])
        self.assertEqual(self.refused(report, '_internal/vendor/notes.txt'), ['denied_name'])
        self.assertEqual(self.refused(report, '_internal/vendor/other.txt'), [])
        self.assertEqual(next(hit['line'] for hit in report['refused'] if hit['file'] == '_internal/vendor/notes.txt'), 1)
        for private in (self.NAMES['accounts'][0], self.NAMES['machines'][0]):
            self.assertNotIn(private, json.dumps(report) + build_binary.hygiene_report(report['refused']))

    def test_foreign_accounts_in_third_party_payload_are_reported_not_refused(self):
        self.write('_internal/libother.so', self.noise + ('/' + self.HOME + '/runner/work/x').encode() + self.noise)
        self.write('_internal/other-1.0.dist-info/METADATA', ('Name: other\nHome: ' + 'C:' + chr(92) + self.USERS + chr(92) + 'runneradmin' + chr(92) + 'x\n').encode())
        # a cargo SBOM's local file URL of a drive root, assembled so this file carries no contiguous match of the builder's class
        self.write('_internal/sbom.cyclonedx.json', ('{"url": "file:' + '///' + self.DRIVE + '/a/x"}\n').encode())
        report = self.scan()
        self.assertEqual(report['refused'], [])
        reported = {(row['file'], row['class']) for row in report['reported']}
        self.assertIn(('_internal/libother.so', 'linux_home'), reported)
        self.assertIn(('_internal/other-1.0.dist-info/METADATA', 'windows_user_profile'), reported)
        self.assertIn(('_internal/sbom.cyclonedx.json', 'local_file_url'), reported)
        # the same account path in the owned zone (the builder's own records) is refused
        self.write('NATIVE_COMPONENTS.json', ('[{"built_at": "/' + self.HOME + '/runner/work/x"}]\n').encode())
        report = self.scan()
        self.assertEqual(self.refused(report, 'NATIVE_COMPONENTS.json'), ['linux_home'])

    def test_posix_spellings_of_a_user_profile_in_a_bundle_text_member_are_refused(self):
        # the retained lab Windows FFmpeg bundle records its configure --prefix as /c/<Users>/<account>/Documents/<agent tree>/...
        # in version.txt and buildconf.txt (MSYS spelling); the WSL and Cygwin spellings are the same class. Assembled from parts.
        account = self.NAMES['accounts'][0]
        prefix = '--prefix=/c/' + self.USERS + '/' + account + '/Documents/' + self.CODEX + '/2026-09-11/work/ffmpeg-build/install'
        self.write('_internal/native/ffmpeg/version.txt', ('ffmpeg version 9.0.1\nconfiguration: ' + prefix + ' --disable-gpl\n').encode())
        self.write('_internal/native/ffmpeg/buildconf.txt', ('  configuration:\n    ' + prefix + '\n    --disable-gpl\n').encode())
        self.write('_internal/other-1.0.dist-info/METADATA', ('Name: other\nBuilt: /mnt/c/' + self.USERS + '/' + account + '/src\n').encode())
        self.write('_internal/libthird.so', self.noise + ('/cygdrive/c/' + self.USERS + '/runneradmin/work').encode() + self.noise)
        report = self.scan()
        self.assertEqual(self.refused(report, '_internal/native/ffmpeg/version.txt'), ['lab_codex_worktree', 'posix_user_profile'])
        self.assertEqual(self.refused(report, '_internal/native/ffmpeg/buildconf.txt'), ['lab_codex_worktree', 'posix_user_profile'])
        self.assertEqual(next(hit['line'] for hit in report['refused'] if hit['file'] == '_internal/native/ffmpeg/buildconf.txt' and hit['class'] == 'posix_user_profile'), 2)
        # third-party zone: the building account refuses, a CI runner's profile is reported
        self.assertEqual(self.refused(report, '_internal/other-1.0.dist-info/METADATA'), ['posix_user_profile/denied_account'])
        self.assertEqual(self.refused(report, '_internal/libthird.so'), [])
        self.assertIn(('_internal/libthird.so', 'posix_user_profile'), {(row['file'], row['class']) for row in report['reported']})
        self.assertNotIn(account, json.dumps(report) + build_binary.hygiene_report(report['refused']))
        with self.assertRaisesRegex(ValueError, r'version\.txt \[owned\] lab_codex_worktree x1 \(first:2\), posix_user_profile x1 \(first:2\)'):
            build_binary.verify_distribution_hygiene(self.dist, self.private)

    def test_utf16le_strings_in_binaries_are_scanned_with_the_same_classes_and_zones(self):
        # wide strings as a Windows DLL carries them: the 2026-10-06 verifier counted 1,647 UTF-16LE copies of an upstream conda
        # builder's user profile in one collected DLL, which the 8-bit scan of that day could not see. Planted from parts, as UTF-16LE.
        def wide(text):
            return text.encode('utf-16-le')
        account = self.NAMES['accounts'][0]
        upstream = 'C:' + chr(92) + self.USERS + chr(92) + 'ci' + chr(92) + '64' + chr(92) + 'miniconda3' + chr(92) + 'envs' + chr(92) + 'numba-ci'
        self.write('_internal/llvmlite/binding/llvmlite.dll', self.noise + b'\x00\x00' + wide(upstream) + b'\x00\x00' + self.noise + wide(upstream) + b'\x00\x00')
        self.write('_internal/native/ffmpeg/avcodec-63.dll', self.noise + wide('C:' + '/' + self.USERS + '/runner/build/ffmpeg.c') + b'\x00\x00' + self.noise)
        self.write('_internal/avutil-61.dll', self.noise + wide('--prefix=/c/' + self.USERS + '/' + account + '/install') + b'\x00\x00' + self.noise)
        self.write('_internal/other.dll', self.noise + wide(self.DRIVE + '/' + self.LAB + 'Frozen/ext/x') + b'\x00\x00' + wide('built on ' + self.NAMES['machines'][0]) + self.noise)
        self.write('_internal/odd.dll', self.noise + b'\x01' + wide('/' + self.HOME + '/' + self.LAB_HOST + '/build') + b'\x00\x00' + self.noise)   # odd alignment
        # a surrogate pair (four bytes, one decoded character) in the same window before the hit: the byte offset must still be exact
        self.write('_internal/pair.dll', self.noise + b'\x00\x00' + wide('\U0001F600 /' + self.HOME + '/' + self.LAB_HOST + '/build') + b'\x00\x00' + self.noise)
        report = self.scan()
        reported = {(row['file'], row['class'], row.get('encoding')): row['count'] for row in report['reported']}
        self.assertEqual(reported[('_internal/llvmlite/binding/llvmlite.dll', 'windows_user_profile', 'utf-16le')], 2)
        self.assertEqual(self.refused(report, '_internal/llvmlite/binding/llvmlite.dll'), [])
        self.assertEqual(self.refused(report, '_internal/native/ffmpeg/avcodec-63.dll'), ['windows_user_profile'])          # owned zone refuses every class
        self.assertEqual(self.refused(report, '_internal/avutil-61.dll'), ['posix_user_profile/denied_account'])            # the building account
        self.assertEqual(self.refused(report, '_internal/other.dll'), ['denied_name', 'lab_drive_titanos'])                  # lab shapes and names
        self.assertEqual(self.refused(report, '_internal/odd.dll'), ['lab_linux_host', 'linux_home/denied_account'])
        self.assertEqual(self.refused(report, '_internal/pair.dll'), ['lab_linux_host', 'linux_home/denied_account'])
        for hit in report['refused']:
            self.assertEqual(hit['encoding'], 'utf-16le', hit)
            self.assertIsNone(hit['line'])
        offset = next(hit['offset'] for hit in report['refused'] if hit['file'] == '_internal/odd.dll' and hit['class'] == 'linux_home')
        self.assertEqual(offset, len(self.noise) + 1)
        offset = next(hit['offset'] for hit in report['refused'] if hit['file'] == '_internal/pair.dll' and hit['class'] == 'linux_home')
        self.assertEqual(offset, len(self.noise) + 2 + 4 + 2)   # two NULs, the four-byte pair, the two-byte space: where '/' begins
        text = json.dumps(report) + build_binary.hygiene_report(report['refused'])
        for private in (account, self.NAMES['machines'][0], self.LAB_HOST):
            self.assertNotIn(private, text)
        self.assertIn('windows_user_profile[utf-16le] x1', build_binary.hygiene_report(report['refused']))
        # with the refused files gone, the residue is counted in BUILD_HYGIENE.json by class and separately for UTF-16LE
        for name in ('_internal/native/ffmpeg/avcodec-63.dll', '_internal/avutil-61.dll', '_internal/other.dll', '_internal/odd.dll', '_internal/pair.dll'):
            (self.dist / name).unlink()
        summary = build_binary.verify_distribution_hygiene(self.dist, self.private)
        self.assertEqual(summary['utf16le'], {'refused_hits': 0, 'reported_third_party_residue': {'files': 1, 'by_class': {'windows_user_profile': 2}}})
        self.assertEqual(summary['reported_third_party_residue']['by_class']['windows_user_profile'], 4)   # the two 8-bit zip-member hits of the fixture and these
        self.assertIn('UTF-16LE', summary['scope'])
        self.assertIn('utf-16le', summary['classes']['encodings'])

    def test_text_only_classes_apply_to_decodable_owned_files(self):
        self.write('DEPENDENCIES.json', ('{"mirror": "http://' + '.'.join(['192', '168', '7', '7']) + '/simple"}\n').encode())
        self.write('_internal/lib.so', self.noise + '.'.join(['192', '168', '7', '7']).encode() + self.noise)   # random-looking bytes: not scanned for numeric shapes
        report = self.scan()
        self.assertEqual(self.refused(report, 'DEPENDENCIES.json'), ['rfc1918_ipv4'])
        self.assertEqual([hit for hit in report['refused'] if hit['file'] == '_internal/lib.so'], [])

    def test_large_files_are_scanned_in_overlapping_chunks_and_a_boundary_match_is_seen_once(self):
        locator = ('/' + self.HOME + '/' + self.LAB_HOST + '/build').encode()
        chunk, overlap, context = 4096, 512, 64
        body = bytearray(self.noise * 5)                    # 15000 bytes: several chunks at the patched sizes
        positions = sorted((chunk - 10, 2 * chunk - overlap + 3, 3 * chunk - context + 1, 11000))   # across each boundary kind
        for position in positions:
            body[position - 1] = 0                           # a NUL before the string, as in a real binary's string table
            body[position:position + len(locator)] = locator
        name = '_internal/native/ffmpeg/libavutil.so'
        self.write(name, bytes(body))
        with patch.object(build_binary, 'HYGIENE_CHUNK', chunk), patch.object(build_binary, 'HYGIENE_OVERLAP', overlap), \
                patch.object(build_binary, 'HYGIENE_CONTEXT', context), patch.object(build_binary, 'HYGIENE_TEXT_LIMIT', 1024):
            report = self.scan()
        offsets = sorted(hit['offset'] for hit in report['refused'] if hit['file'] == name and hit['class'] == 'linux_home')
        self.assertEqual(offsets, positions)
        self.assertEqual(len([hit for hit in report['refused'] if hit['file'] == name and hit['class'] == 'lab_linux_host']), 4)

    def test_zones_and_names(self):
        zone = build_binary.hygiene_zone
        for name in ('BUILD.json', 'SOURCE/src/titanos/server.py', '_internal/titanos/web/index.html', '_internal/native/ffmpeg/ffmpeg.exe',
                     '_internal/torch-2.8.0.dist-info/direct_url.json', 'runtimes/ace-step-1.5/runtime.json', 'runtimes/ace-step-1.5/adapter/adapter.py',
                     'runtimes/ace-step-1.5/python/Lib/site-packages/x.dist-info/direct_url.json'):
            self.assertEqual(zone(name), 'owned', name)
        for name in ('_internal/torch/lib/torch_cpu.dll', '_internal/base_library.zip', 'THIRD_PARTY_LICENSES/sympy/AUTHORS', 'THIRD_PARTY_SOURCES/x/archives/y.zip',
                     'runtimes/ace-step-1.5/python/python.exe', 'runtimes/ace-step-1.5/SOURCE/ace-step-1.5/README.md', 'runtimes/ace-step-1.5/licenses/x'):
            self.assertEqual(zone(name), 'third_party', name)
        environ = {'COMPUTERNAME': 'EXAMPLE-HOST-01', 'HOSTNAME': 'localhost', 'USERNAME': 'example-account', 'TITANOS_HYGIENE_DENY': 'other-account; OTHER-HOST'}
        names = build_binary.hygiene_names(environ)
        self.assertIn('EXAMPLE-HOST-01', names['machines'])
        self.assertIn('OTHER-HOST', names['machines'])
        self.assertNotIn('localhost', names['machines'])
        self.assertTrue({'example-account', 'other-account', 'OTHER-HOST'} <= set(names['accounts']))
        self.assertEqual(set(names['private']), set(lab_paths.private_names()))
        # the policy file is the one statement of the classes the scan relies on
        policy = build_binary.hygiene_policy()
        self.assertTrue(set(build_binary.LOCATOR_ANCHORS) | set(build_binary.TEXT_ONLY_CLASSES) <= set(policy['patterns']))
        self.assertEqual(set(policy['patterns']), set(PATTERNS))
        self.assertEqual(policy['allowlist'].get('tests/check_music_failure_diagnostics.py'), ALLOWLIST['tests/check_music_failure_diagnostics.py'])

    def test_main_runs_the_scan_on_the_unsealed_directory_before_sealing(self):
        text = (ROOT / 'tools/build_binary.py').read_text(encoding='utf-8')
        main = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        source = ast.get_source_segment(text, main)
        scan = re.search(r"verify_distribution_hygiene\(built, run / 'validation-private'\)", source)
        manifest = re.search(r"manifest\s*=\s*package_manifest\(built\)", source)
        smoke = re.search(r"publish_smoke_summary\(built,", source)
        self.assertTrue(scan and manifest and smoke)
        self.assertLess(smoke.start(), scan.start(), 'the raw smoke receipts leave the directory before the scan')
        self.assertLess(scan.start(), manifest.start(), 'the scan runs before MANIFEST.json seals the directory')


if __name__ == '__main__':
    unittest.main(verbosity=2)
