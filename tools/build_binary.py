"""Build a self-contained native executable directory without models or caches.

Use the same script on the target OS/architecture. PyInstaller is not a cross
compiler. The full profile includes the installed numerical runtime packages.
"""

import argparse
import ast
import asyncio
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import tomllib
from urllib.parse import unquote
import uuid
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
# Isolated Python omits the script directory; use only this checked build-tool root.
sys.path.insert(0, str(ROOT / 'tools'))
CORE_PACKAGES = ['titanos', 'mcp', 'mcp_types', 'uvicorn', 'huggingface_hub']
MODEL_PACKAGES = ['torch', 'torchvision', 'transformers', 'diffusers', 'accelerate',
                  'numpy', 'scipy', 'PIL', 'imageio', 'imageio_ffmpeg',
                  'sentencepiece', 'tokenizers', 'safetensors', 'timm']
OPTIONAL_PACKAGES = {'onnxruntime': 'onnxruntime', 'openvino': 'openvino',
                     'openvino_genai': 'openvino-genai', 'openvino_tokenizers': 'openvino-tokenizers',
                     'TTS': 'coqui-tts', 'torchaudio': 'torchaudio', 'qwen_tts': 'qwen-tts',
                     'sherpa_onnx': 'sherpa-onnx==1.13.8'}
QWEN_PACKAGES = ['librosa', 'numba', 'llvmlite', 'soxr', 'soundfile', '_soundfile',
                 '_soundfile_data', 'sox', 'einops', 'audioread', 'lazy_loader']
RUNTIME_DEPENDENCY_EXCEPTIONS = {
    ('qwen-tts', 'gradio'): 'Upstream qwen_tts.cli demo UI only; that CLI is excluded. Titanos supplies its own API and browser UI.'}
# Distributions a redistributable build refuses unless the builder opts in by name (--allow-restricted), which the plan and
# DEPENDENCIES.json record: copyleft or non-commercial terms, or code or data that carries them. Any other distribution in the
# closure whose declared licence is GPL or AGPL (LGPL is not refused) or non-commercial is refused the same way.
RESTRICTED_DISTRIBUTIONS = {
    'coqui-tts': 'Coqui TTS: MPL-2.0 code whose XTTS-v2 weights are under the non-commercial Coqui Public Model License',
    'tts': 'Coqui TTS (the original distribution name): MPL-2.0 code, non-commercial XTTS-v2 weights',
    'ultralytics': 'AGPL-3.0',
    'phonemizer': 'GPL-3.0-or-later',
    'phonemizer-fork': 'GPL-3.0-or-later',
    'espeakng-loader': 'ships eSpeak NG, which is GPL-3.0-or-later; the wheel states no licence',
}
COPYLEFT_LICENSE = re.compile(r'(?<![A-Za-z])A?GPL(?:v?[0-9]|(?![A-Za-z]))|GNU (?:Affero )?General Public License')
NONCOMMERCIAL_LICENSE = re.compile(r'(?i)\bCC-BY(?:-SA)?-NC\b|non-?commercial|\bCPML\b')


def restricted_distributions(installed, allowed=()):
    """{canonical name: reason} for the distributions of installed ({name: importlib.metadata.Distribution}) that a
    redistributable build refuses and allowed does not name: RESTRICTED_DISTRIBUTIONS, and any distribution whose
    License-Expression, licence classifiers or short License field declare GPL, AGPL or non-commercial terms. A long License
    field is a bundled licence text (numpy carries the GCC runtime exception, for example), so it is not read."""
    allowed = {canonicalize_name(name) for name in allowed}
    refused = {}
    for name, distribution in installed.items():
        key = canonicalize_name(name)
        if key in allowed:
            continue
        if key in RESTRICTED_DISTRIBUTIONS:
            refused[key] = RESTRICTED_DISTRIBUTIONS[key]
            continue
        meta = distribution.metadata
        declared = [meta.get('License-Expression') or '',
                    *[value for value in meta.get_all('Classifier') or [] if value.startswith('License ::')]]
        short = meta.get('License') or ''
        if len(short) <= 100 and '\n' not in short:
            declared.append(short)
        for value in declared:
            if COPYLEFT_LICENSE.search(value) or NONCOMMERCIAL_LICENSE.search(value):
                refused[key] = 'declared licence: ' + value.strip()[:100]
                break
    return dict(sorted(refused.items()))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def normalize_dylib_diagnostics(source_dir, destination_dir):
    """Publish otool dependency lines unchanged, with basename-only headings."""
    source = Path(source_dir) / 'dylib-dependencies.json'
    target = Path(destination_dir) / 'dylib-dependencies.json'
    if not source.is_file():
        if target.exists():
            raise ValueError('Packaged dylib diagnostics have no source counterpart')
        return None
    original = source.read_bytes()
    records = json.loads(original)
    if not isinstance(records, dict):
        raise ValueError('Dylib diagnostics must map basenames to otool output')
    public = {}
    for name, output in records.items():
        if not isinstance(name, str) or not name or '/' in name or '\\' in name or not isinstance(output, str):
            raise ValueError('Invalid dylib diagnostic entry')
        heading, newline, dependencies = output.partition('\n')
        plain_heading = heading.removesuffix('\r')
        if not newline or not plain_heading.endswith(':') or Path(plain_heading[:-1]).name != name:
            raise ValueError('Dylib diagnostic filename heading differs from its key')
        public[name] = name + ':' + ('\r' if heading.endswith('\r') else '') + newline + dependencies
    normalized = (json.dumps(public, indent=2) + '\n').encode('utf-8')
    if target.read_bytes() not in (original, normalized):
        raise ValueError('Packaged dylib diagnostics differ from the verified source')
    target.write_bytes(normalized)
    return {'file': 'dylib-dependencies.json', 'source_sha256': hashlib.sha256(original).hexdigest(),
            'packaged_sha256': hashlib.sha256(normalized).hexdigest(),
            'normalization': 'Filename headings only; every dependency line retained unchanged.'}


def add_bootloader_metadata(destination, expected_version):
    """Retain the installed bootloader notice without adding a runtime dependency."""
    destination = Path(destination)
    distribution = metadata.distribution('pyinstaller')
    if distribution.version != expected_version:
        raise ValueError('Installed PyInstaller differs from the packaged bootloader version')
    candidates = [Path(distribution.locate_file(item)) for item in distribution.files or []
                  if Path(str(item)).name == 'COPYING.txt']
    if len(candidates) != 1 or not candidates[0].is_file():
        raise ValueError('Exact installed PyInstaller licensing notice is unavailable')
    notice = candidates[0].read_bytes()
    if b'Bootloader Exception' not in notice or b'Run-time Hooks' not in notice:
        raise ValueError('PyInstaller notice lacks the bootloader or runtime-hook terms')
    declared = distribution.metadata.get('License-Expression') or distribution.metadata.get('License')
    if not declared:
        raise ValueError('Installed PyInstaller license declaration is unavailable')
    relative = 'THIRD_PARTY_LICENSES/PyInstaller/COPYING.txt'
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != notice:
        raise ValueError('A different PyInstaller notice is already packaged')
    target.write_bytes(notice)
    notice_hash = hashlib.sha256(notice).hexdigest()
    native = {'name': 'PyInstaller bootloader', 'version': distribution.version,
              'role': 'Embedded bootloader, loader and runtime hooks; build tool, not a runtime Python dependency',
              'declared_license': declared, 'license_file': relative, 'license_sha256': notice_hash,
              'license_review': 'not_performed'}
    component = {'type': 'application', 'name': 'PyInstaller', 'version': distribution.version,
                 'purl': 'pkg:pypi/pyinstaller@' + distribution.version, 'scope': 'required',
                 'licenses': [{'license': {'name': declared}}],
                 'properties': [{'name': 'titanos:component-role', 'value': 'embedded-bootloader'},
                                {'name': 'titanos:license-file', 'value': relative},
                                {'name': 'titanos:license-sha256', 'value': notice_hash}]}
    names = {'pyinstaller', 'pyinstaller bootloader'}
    native_path = destination / 'NATIVE_COMPONENTS.json'
    entries = json.loads(native_path.read_text(encoding='utf-8'))
    entries = [entry for entry in entries if str(entry.get('name', '')).lower() not in names]
    native_path.write_text(json.dumps([*entries, native], indent=2) + '\n', encoding='utf-8')
    sbom_path = destination / 'sbom.cdx.json'
    sbom = json.loads(sbom_path.read_text(encoding='utf-8'))
    sbom['components'] = [entry for entry in sbom['components'] if str(entry.get('name', '')).lower() not in names]
    sbom['components'].append(component)
    sbom_path.write_text(json.dumps(sbom, indent=2) + '\n', encoding='utf-8')
    notices = destination / 'THIRD_PARTY_NOTICES.md'
    text = notices.read_text(encoding='utf-8')
    if '## Embedded PyInstaller bootloader' not in text:
        text += ('\n## Embedded PyInstaller bootloader\n\nThe exact installed PyInstaller notice, including its '
                 'bootloader exception and runtime-hook terms, is retained in ' + relative + '. '
                 'Its build/embedded component is recorded separately from runtime Python dependencies.\n')
        notices.write_text(text, encoding='utf-8')
    return native


def dependencies(initial):
    """Resolve installed runtime requirements without adding build-tool packages."""
    selected, pending = {}, list(initial)
    while pending:
        requirement = Requirement(pending.pop())
        name = canonicalize_name(requirement.name)
        extras = frozenset(requirement.extras)
        distribution = metadata.distribution(requirement.name)
        if requirement.specifier and distribution.version not in requirement.specifier:
            raise ValueError(f'Installed {requirement.name} {distribution.version} does not satisfy {requirement.specifier}')
        if name in selected and extras <= selected[name][1]:
            continue
        merged = extras | (selected[name][1] if name in selected else frozenset())
        selected[name] = (distribution, merged)
        for item in distribution.requires or []:
            child = Requirement(item)
            if (name, canonicalize_name(child.name)) in RUNTIME_DEPENDENCY_EXCEPTIONS:
                continue
            if child.marker is None or any(child.marker.evaluate({'extra': extra}) for extra in (merged | {''})):
                pending.append(str(child))
    return {name: value[0] for name, value in sorted(selected.items())}


def engine_excludes(closure, mapping=None):
    """Importable top-level modules whose installed distributions all lie outside the engine closure.

    PyInstaller follows optional imports (onnxruntime, torch, numpy, ...) whenever they happen to be
    installed in the build environment. Excluding them keeps an engine payload equal to its declared
    closure, exactly as if only those distributions were installed; the frozen application then reports
    the missing optional runtime at use time instead of shipping it.
    """
    keep = {canonicalize_name(name) for name in closure}
    mapping = metadata.packages_distributions() if mapping is None else mapping
    return sorted(module for module, distributions in mapping.items()
                  if module and not any(canonicalize_name(name) in keep for name in distributions))


def build_plan(profile, allow_restricted=()):
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    if metadata.version('titanos-engine') != project['version']:
        raise ValueError('Installed Titanos metadata differs from source; install this source in the build environment first')
    installed = {'titanos-engine': metadata.distribution('titanos-engine')}
    requirements = list(project['dependencies'])
    packages = list(CORE_PACKAGES)
    if profile == 'full':
        packages += MODEL_PACKAGES
        requirements += project['optional-dependencies']['models']
        for module, distribution in OPTIONAL_PACKAGES.items():
            if util.find_spec(module) is not None:
                packages.append(module)
                requirements.append(distribution)
        if 'qwen_tts' in packages:
            packages += QWEN_PACKAGES
            requirements += project['optional-dependencies']['qwen3-tts']
            from titanos.qwen_tts_backend import WRAPPER_REVISION
            provenance = metadata.distribution('qwen-tts').read_text('direct_url.json')
            if json.loads(provenance or '{}').get('vcs_info', {}).get('commit_id') != WRAPPER_REVISION:
                raise ValueError('The installed Qwen streaming wrapper does not match the pinned source revision')
    missing = [name for name in packages if util.find_spec(name) is None]
    if missing:
        raise ValueError('Build environment is missing: ' + ', '.join(missing))
    installed.update(dependencies(requirements))
    refused = restricted_distributions(installed, allow_restricted)
    if refused:
        raise ValueError('Refusing a redistributable build with copyleft or non-commercial distributions (uninstall them, or '
                         'opt in with --allow-restricted NAME): ' + '; '.join(f'{k}: {v}' for k, v in refused.items()))
    opted = {canonicalize_name(name) for name in allow_restricted}
    restricted_opt_in = [{'distribution': key, 'basis': RESTRICTED_DISTRIBUTIONS.get(key, 'declared licence'),
                          'opt_in': '--allow-restricted'} for key in sorted(opted) if key in {canonicalize_name(n) for n in installed}]
    excluded = engine_excludes(installed) if profile == 'engine' else []
    modules = {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
               for path in sorted((ROOT / 'src' / 'titanos').rglob('*')) if path.is_file() and path.suffix in {'.py', '.html', '.css', '.js', '.svg', '.json'}}
    return {'version': project['version'], 'profile': profile, 'source_root': str(ROOT),
            'os': platform.system().lower(), 'architecture': platform.machine(),
            'python': platform.python_version(), 'pyinstaller': metadata.version('pyinstaller'),
            'collect_packages': packages, 'distributions': sorted(installed), 'exclude_modules': excluded,
            'packages': {name: dist.version for name, dist in installed.items()},
            'dependency_exceptions': [{'parent': parent, 'dependency': child, 'reason': reason}
                                     for (parent, child), reason in RUNTIME_DEPENDENCY_EXCEPTIONS.items() if parent in installed],
            'restricted_opt_in': restricted_opt_in,
            'source_files': modules, 'weights_included': False,
            'build_recipe_files': {path.relative_to(ROOT).as_posix(): digest(path)
                                   for path in sorted((ROOT / 'tools').rglob('*'))
                                   if path.is_file() and path.suffix in {'.py', '.spec'}
                                   and 'validation' not in path.relative_to(ROOT / 'tools').parts},
            'scope': 'Installed runtime packages on this OS and architecture; build success does not qualify every model or accelerator'}


def tesseract_plan(directory):
    """Bind a model-free source-built native bundle without executing OCR."""
    root = Path(directory).resolve()
    record = json.loads((root / 'NATIVE_VERIFICATION.json').read_text(encoding='utf-8'))
    if os.name != 'nt' or platform.machine().lower() not in {'amd64', 'x86_64'} or record.get('PE_machine') != 'AMD64':
        raise ValueError('This Tesseract bundle is qualified for native Windows AMD64 only')
    if record.get('status') != 'PASSED' or record.get('direct_imports') != ['KERNEL32.dll']:
        raise ValueError('Tesseract requires verified native startup and dependency closure')
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest) | {'MANIFEST.json'} or any(n.endswith('.traineddata') for n in actual):
        raise ValueError('Tesseract bundle file set differs or includes model data')
    for name, row in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Tesseract native/source/license manifest mismatch: ' + name)
    if digest(root / 'tesseract.exe') != record['executable_sha256']:
        raise ValueError('Tesseract executable differs from native verification')
    provenance = json.loads((root / 'SOURCE_PROVENANCE.json').read_text())
    if {s['name'] for s in provenance['sources']} != {'tesseract', 'leptonica', 'libpng', 'zlib'}:
        raise ValueError('Tesseract bundle lacks complete source provenance')
    return {'directory': str(root), 'files': {n: digest(root / n) for n in sorted(actual)},
            'executable_sha256': record['executable_sha256'], 'source': provenance['sources']}


def copy_tesseract(destination, plan):
    if plan.get('native_tesseract'):
        expected = plan['native_tesseract']
        if tesseract_plan(expected['directory']) != expected:
            raise ValueError('Tesseract inputs changed during build')
        target = destination / '_internal/native/tesseract'
        shutil.copytree(expected['directory'], target)
        for name, checksum in expected['files'].items():
            if digest(target / name) != checksum:
                raise ValueError('Tesseract copy readback differs')


# The text members of an FFmpeg bundle that normalize_packaged_build_metadata rewrites in the packaged copy (the configure
# --prefix becomes <install-prefix>); every other member must ship byte-identical to the bundle, binaries aside (native_inventory).
FFMPEG_NORMALIZED_TEXT = ('buildconf.txt', 'license-output.txt', 'version.txt')
# Executable image signatures: ELF, PE (MZ) and the Mach-O / universal magics in both byte orders.
EXECUTABLE_MAGICS = (b'\x7fELF', b'MZ', b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe',
                     b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca')


def is_executable_image(path):
    """Whether the file starts with an ELF, PE or Mach-O magic; a text member or an archive does not."""
    with Path(path).open('rb') as stream:
        head = stream.read(4)
    return any(head.startswith(magic) for magic in EXECUTABLE_MAGICS)


def ffmpeg_bundle_binaries(root, native):
    """(binaries mapping, derived) for an FFmpeg bundle's BUILD.json: its own `binaries` when it has one, otherwise the ELF, PE
    or Mach-O members of its `members` inventory (member name -> sha256). A bundle with neither cannot be bound and is refused
    with a typed message; so is a `members` inventory that disagrees with the files on disk or omits a file the bundle carries."""
    root = Path(root)
    members = native.get('members')
    if members is not None:
        if not isinstance(members, dict) or not members:
            raise ValueError('FFmpeg BUILD.json members inventory is empty or malformed')
        for name, row in members.items():
            candidate = (root / name).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                raise ValueError('FFmpeg BUILD.json members inventory names a missing or external file: ' + str(name))
            if candidate.stat().st_size != row.get('bytes') or digest(candidate) != row.get('sha256'):
                raise ValueError('FFmpeg bundle member differs from its members inventory: ' + str(name))
        unlisted = sorted(path.relative_to(root).as_posix() for path in root.rglob('*')
                          if path.is_file() and path.name != 'BUILD.json' and path.relative_to(root).as_posix() not in members)
        if unlisted:
            raise ValueError('FFmpeg bundle carries files its members inventory does not list: ' + ', '.join(unlisted[:10]))
    if native.get('binaries') is not None:
        if not isinstance(native['binaries'], dict) or not native['binaries']:
            raise ValueError('FFmpeg BUILD.json binaries mapping is empty or malformed')
        return dict(native['binaries']), False
    if members is None:
        raise ValueError('FFmpeg BUILD.json has neither a binaries mapping nor a members inventory; the bundle cannot be bound')
    derived = {name: row['sha256'] for name, row in members.items() if is_executable_image(root / name)}
    if not derived:
        raise ValueError('FFmpeg BUILD.json has no binaries mapping and its members inventory holds no ELF, PE or Mach-O file to derive one from')
    return derived, True


def ffmpeg_plan(directory, retain_inventory=False):
    """Validate existing corresponding-source bindings without running FFmpeg.

    The bundle's BUILD.json binds the binaries either by its own `binaries` mapping or, for a bundle that carries a `members`
    inventory instead (the neutral Linux bundle of 2026-10-05), by the executable-image members (ffmpeg_bundle_binaries); the
    plan records which, and write_notices carries `binaries_derived_from_members` into NATIVE_COMPONENTS.json."""
    root = Path(directory).expanduser().resolve()
    native = json.loads((root / 'BUILD.json').read_text(encoding='utf-8'))
    if not native['license_from_binary'].startswith('LGPL-'):
        raise ValueError('The supplied FFmpeg profile must identify its LGPL source build')
    binaries, derived = ffmpeg_bundle_binaries(root, native)
    for name, expected in binaries.items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or digest(candidate) != expected:
            raise ValueError('FFmpeg binary provenance mismatch')
    sources = list((root / 'source').glob('ffmpeg-*.tar.*'))
    source = next((path for path in sources if digest(path) == native['source_sha256']), None)
    if source is None:
        raise ValueError('The FFmpeg corresponding-source archive is missing or differs')
    result = {'ffmpeg_dir': str(root), 'ffmpeg_binaries': binaries, 'ffmpeg_binaries_derived_from_members': derived}
    if retain_inventory:
        # The audio profile binds every recipe/notice/source byte as well as binaries.
        executable = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
        if executable not in binaries:
            raise ValueError('FFmpeg build lacks the target-platform executable')
        license_file = root / 'COPYING.LGPLv2.1'
        if not license_file.is_file() or not (root / 'LICENSE.md').is_file():
            raise ValueError('FFmpeg corresponding license texts are missing')
        with tarfile.open(source) as archive:
            matches = [item for item in archive.getmembers() if item.isfile() and Path(item.name).name == license_file.name]
            if len(matches) != 1 or archive.extractfile(matches[0]).read() != license_file.read_bytes():
                raise ValueError('FFmpeg license text differs from corresponding source')
        files = {}
        for path in sorted(root.rglob('*')):
            if path.is_file():
                if not path.resolve().is_relative_to(root):
                    raise ValueError('FFmpeg bundle contains an external file link')
                files[path.relative_to(root).as_posix()] = digest(path)
        result['native_ffmpeg'] = {'source_sha256': native['source_sha256'],
            'license': native['license_from_binary'], 'input_files': files,
            'source_build_binaries': binaries}
    return result


def native_audio_plan(directory):
    """Bind optional voice libraries to retained, checked native build inputs."""
    directory = Path(directory).resolve()
    layouts = [('libsndfile-minimal', '_soundfile_data', 'libsndfile_x64.dll', 'library_sha256'),
               ('soxr-built', 'soxr', 'soxr_ext.pyd', 'extension_sha256')]
    libraries = []
    for component, module, filename, hash_key in layouts:
        source = directory / component
        manifest = json.loads((source / 'MANIFEST.json').read_text(encoding='utf-8'))
        actual_files = {path.relative_to(source).as_posix() for path in source.rglob('*') if path.is_file()}
        if actual_files != {item['path'] for item in manifest} | {'MANIFEST.json'}:
            raise ValueError('Native audio tree contains missing or unrecorded files: ' + component)
        for item in manifest:
            path = (source / item['path']).resolve()
            if not path.is_relative_to(source) or digest(path) != item['sha256']:
                raise ValueError('Native audio source/notice manifest mismatch: ' + component)
        build = json.loads((source / 'BUILD.json').read_text(encoding='utf-8'))
        if not any(digest(path) == build['source_sha256'] for path in source.glob('*.tar.*')):
            raise ValueError('Matching corresponding native source archive is missing: ' + component)
        package = Path(util.find_spec(module).origin).parent
        library = package / filename
        if digest(library) != build[hash_key]:
            raise ValueError('Build import path does not select the verified native audio overlay: ' + str(library))
        libraries.append({'component': component, 'runtime_path': '_internal/' + module + '/' + filename,
                          'sha256': build[hash_key], 'source_sha256': build['source_sha256']})
    return {'directory': str(directory), 'libraries': libraries}


def normalize_packaged_build_metadata(destination, plan):
    """Normalize only newly generated metadata copies, never native inputs."""
    internal = (destination / '_internal').resolve()
    if plan.get('ffmpeg_dir'):
        for name in FFMPEG_NORMALIZED_TEXT:
            path = internal / 'native' / 'ffmpeg' / name
            if path.is_file():
                if not path.resolve().is_relative_to(internal):
                    raise ValueError('FFmpeg diagnostic path escaped the built application')
                text = path.read_text(encoding='utf-8')
                normalized = re.sub(r'--prefix=/[^\s]+', '--prefix=<install-prefix>', text)
                if text != normalized:
                    path.write_text(normalized, encoding='utf-8')
    for path in internal.glob('*dist-info/direct_url.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        if str(record.get('url', '')).startswith('file:'):
            if not path.resolve().is_relative_to(internal):
                raise ValueError('Local build metadata path escaped the built application')
            # Keep archive hashes and other provenance; the local build-machine
            # location is not an installable URL in the frozen application.
            record['url'] = 'file:///bundled-local-build-input'
            path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    for library in plan.get('native_audio', {}).get('libraries', []):
        location = (internal / 'native' / 'audio' / library['component']).resolve()
        if not location.is_relative_to(internal):
            raise ValueError('Native metadata path escaped the built application')
        record_path = location / 'BUILD.json'
        record = json.loads(record_path.read_text(encoding='utf-8'))
        normalized = []
        for flag in record.get('configure', []):
            key, separator, value = flag.partition('=')
            if separator and key in {'-DCMAKE_C_COMPILER', '-DCMAKE_CXX_COMPILER', '-DCMAKE_MAKE_PROGRAM'}:
                flag = key + '=' + Path(value).name
            normalized.append(flag)
        if 'configure' in record:
            record['configure'] = normalized
        record['metadata_normalization'] = 'Host-specific compiler paths normalized; native/source hashes and compiler versions unchanged. Build-only Python caches omitted.'
        record_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        for path in location.rglob('*.pyc'):
            if not path.resolve().is_relative_to(location):
                raise ValueError('Build cache path escaped the native provenance copy')
            path.unlink()
        manifest = [{'path': path.relative_to(location).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)}
                    for path in sorted(location.rglob('*')) if path.is_file() and path.name != 'MANIFEST.json']
        (location / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def music_runtime_plan(directory):
    from music_package import verify_runtime
    directory=Path(directory).expanduser().resolve()
    value=verify_runtime(directory)
    return {'directory':str(directory),'location':'runtimes/ace-step-1.5',
            'manifest_sha256':digest(directory/'runtime.json'),'code_revision':value['code_revision'],
            'files':len(value['files'])+1,'bytes':sum(row['bytes'] for row in value['files'].values())+(directory/'runtime.json').stat().st_size,
            'qualification':value.get('qualification',{'target_native':False})}


def supplement_runtime_licenses(destination, runtime_root, location, dependencies, catalog=None):
    """Isolated-runtime distributions whose inventory row lists no license file get a hash-bound notice copy in the application
    under THIRD_PARTY_LICENSES/<runtime>/<distribution>/: first any license/licence/copying/notice text the shipped dist-info
    actually carries (an older runtime inventory can have missed the British 'LICENCE' spelling), otherwise the retained
    upstream text pinned for that exact name and version in tools/licenses/provenance.json (or `catalog`, whose rows resolve
    beside it). The runtime payload itself is never modified, so its manifest stays bound. Fails closed with ValueError for a
    distribution that has neither. Returns the records written to DEPENDENCIES.json external_runtimes[].supplemented_licenses."""
    destination, runtime_root = Path(destination), Path(runtime_root)
    layout = json.loads((runtime_root / 'runtime.json').read_text(encoding='utf-8')).get('layout', {})
    site = runtime_root / layout.get('site_packages', 'python/Lib/site-packages')
    catalog_path = Path(catalog) if catalog else ROOT / 'tools' / 'licenses' / 'provenance.json'
    retained = json.loads(catalog_path.read_text(encoding='utf-8')) if catalog_path.is_file() else []
    tokens = ('license', 'licence', 'copying', 'notice')
    records, missing = [], []
    for dependency in dependencies:
        if dependency.get('license_files') or dependency.get('license_exception'):
            continue
        name, version = dependency['name'], dependency['version']
        folder = destination / 'THIRD_PARTY_LICENSES' / Path(location).name / canonicalize_name(name)
        copies = []
        for info in sorted(site.glob('*.dist-info')):
            stem = info.name[:-len('.dist-info')]
            if '-' not in stem or canonicalize_name(stem.rsplit('-', 1)[0]) != canonicalize_name(name) or stem.rsplit('-', 1)[1] != version:
                continue
            for path in sorted(info.rglob('*')):
                if path.is_file() and any(token in path.name.lower() for token in tokens):
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / (str(len(copies) + 1) + '-' + path.name)
                    shutil.copy2(path, target)
                    copies.append({'path': target.relative_to(destination).as_posix(), 'sha256': digest(target), 'source': 'shipped:' + path.relative_to(runtime_root).as_posix()})
        if not copies:
            for row in retained:
                if canonicalize_name(row['name']) == canonicalize_name(name) and row['version'] == version:
                    source = catalog_path.parent / row['file']
                    if digest(source) != row['sha256']:
                        raise ValueError('Retained upstream license hash mismatch for ' + name + ': ' + row['file'])
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / row['file']
                    shutil.copy2(source, target)
                    copies.append({'path': target.relative_to(destination).as_posix(), 'sha256': digest(target), 'source': 'retained:tools/licenses/' + row['file'], 'source_url': row.get('url')})
        if not copies:
            missing.append(name + ' ' + version)
            continue
        records.append({'name': name, 'version': version, 'declared_license': dependency.get('declared_license'), 'files': copies})
    if missing:
        raise ValueError('Isolated runtime distributions ship without a license text and have no retained upstream license in tools/licenses/provenance.json: ' + ', '.join(missing))
    return records


def cuda_vendor_component(scope, torch_version, notice_files, location):
    """One NATIVE_COMPONENTS.json row per packaged CUDA payload. The main runtime's torch wheel (under `_internal/torch/lib`)
    and the isolated ACE-Step runtime's torch wheel (under its own site-packages) each carry vendor libraries built against
    a different CUDA release, so each payload gets its own row naming the scope, the packaged library directory, the exact
    torch version, its `+cuNNN` build tag and the notice files retained for that wheel. The row makes one payload visible
    for human vendor-terms review; it records nothing about the terms themselves."""
    tag = str(torch_version).partition('+')[2]
    return {'name': 'NVIDIA CUDA runtime libraries (' + scope + ')', 'scope': scope, 'location': location,
            'torch_version': torch_version, 'cuda_build_tag': tag if tag.startswith('cu') else None,
            'status': 'Vendor redistributable terms are separate from Titanos, PyTorch and (for the isolated runtime) ACE-Step source licenses; '
                      'this row scopes one CUDA payload for human vendor-terms review',
            'notice_files': notice_files, 'review': 'not_performed'}


def include_music_runtime(destination, plan):
    from music_package import verify_runtime
    value=plan.get('native_music')
    if not value:return
    source=Path(value['directory']);destination=Path(destination);target=destination/value['location']
    if (destination/'MANIFEST.json').exists() or target.exists():
        raise ValueError('Music runtime must be included before sealing a fresh application')
    original=verify_runtime(source)
    if digest(source/'runtime.json')!=value['manifest_sha256']:
        raise ValueError('Music runtime changed after build planning')
    shutil.copytree(source,target)
    if verify_runtime(target)!=original or digest(target/'runtime.json')!=value['manifest_sha256']:
        raise ValueError('Copied music runtime differs from the planned payload')
    dependencies=json.loads((target/'DEPENDENCIES.json').read_text(encoding='utf-8'))['components']
    supplemented=supplement_runtime_licenses(destination,target,value['location'],dependencies)
    summary={key:item for key,item in value.items() if key!='directory'}
    native_path=destination/'NATIVE_COMPONENTS.json'
    native=json.loads(native_path.read_text(encoding='utf-8'))
    # The isolated runtime's torch wheel brings its own CUDA vendor libraries, built against a different CUDA release than the
    # main runtime's wheel. Scope that payload in its own row (before the runtime row) so both CUDA scopes stay visible.
    layout=json.loads((target/'runtime.json').read_text(encoding='utf-8')).get('layout',{})
    cuda_rows=[]
    for dependency in dependencies:
        if canonicalize_name(dependency['name'])=='torch' and '+cu' in str(dependency['version']):
            notices=[value['location']+'/'+path for path in dependency.get('license_files') or []] or 'THIRD_PARTY_LICENSES/'+Path(value['location']).name+'/torch'
            cuda_rows.append(cuda_vendor_component('isolated ACE-Step runtime',dependency['version'],notices,
                value['location']+'/'+layout.get('site_packages','python/Lib/site-packages')+'/torch/lib'))
    native.extend(cuda_rows)
    native.append({'name':'Isolated ACE-Step runtime',**summary,'inventories':{
        name:value['location']+'/'+name.upper()+'.json' for name in ('dependencies','sources','licenses')}})
    native_path.write_text(json.dumps(native,indent=2)+'\n',encoding='utf-8')
    inventory_path=destination/'DEPENDENCIES.json';inventory=json.loads(inventory_path.read_text(encoding='utf-8'))
    inventory.setdefault('external_runtimes',[]).append({'name':'ace-step-1.5','location':value['location'],
        'dependency_inventory':value['location']+'/DEPENDENCIES.json','manifest_sha256':value['manifest_sha256'],'supplemented_licenses':supplemented})
    inventory_path.write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    sbom_path=destination/'sbom.cdx.json';sbom=json.loads(sbom_path.read_text(encoding='utf-8'))
    sbom['components'].append({'type':'application','name':'ACE-Step isolated runtime','version':value['code_revision'],
        'bom-ref':'music:ace-step-1.5','properties':[{'name':'titanos:runtime-manifest','value':value['location']+'/runtime.json'}]})
    for dependency in dependencies:
        component={'type':'library','name':dependency['name'],'version':dependency['version'],
                   'bom-ref':'music:'+dependency['name']+'@'+dependency['version'],
                   'properties':[{'name':'titanos:runtime','value':value['location']}]}
        if dependency.get('declared_license'):component['licenses']=[{'license':{'name':dependency['declared_license']}}]
        sbom['components'].append(component)
    sbom_path.write_text(json.dumps(sbom,indent=2)+'\n',encoding='utf-8')
    with (destination/'THIRD_PARTY_NOTICES.md').open('a',encoding='utf-8') as stream:
        stream.write('\n## Optional isolated music runtime\n\nThe hash-bound `runtimes/ace-step-1.5` directory retains its own interpreter, dependency inventory, pinned source and component license texts. Model weights remain external. Construction and retained notices do not establish target execution or legal approval.'
                     +(' NATIVE_COMPONENTS.json lists the CUDA-built torch wheel of this runtime as a CUDA payload scoped to the isolated runtime, separately from any main-runtime CUDA row.' if cuda_rows else '')+'\n')
    if supplemented:
        with (destination/'THIRD_PARTY_NOTICES.md').open('a',encoding='utf-8') as stream:
            stream.write('\nRuntime distributions whose wheels ship no license text carry hash-bound copies under THIRD_PARTY_LICENSES/'+Path(value['location']).name+'/: '
                         +', '.join(sorted(item['name']+' '+item['version']+' ('+'; '.join(f['source'] for f in item['files'])+')' for item in supplemented))
                         +'. Sources are the shipped dist-info text or the retained upstream text pinned in tools/licenses/provenance.json; this is not a legal approval.\n')


def _inventory_path(root, name):
    """Read only a contained ordinary package member, never a notice-path escape."""
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or relative.as_posix() != name
            or '..' in relative.parts or '\\' in name or ':' in name):
        raise ValueError('Unsafe distribution inventory path: ' + str(name))
    path = Path(root).absolute() / name
    for item in (*reversed(path.parents), path):
        try:
            value = item.lstat()
        except OSError as error:
            raise ValueError('Unavailable distribution inventory member: ' + str(item) + ': ' + str(error)) from error
        if (stat.S_ISLNK(value.st_mode) or getattr(value, 'st_file_attributes', 0) & 0x400) \
                and not (item != path and _os_alias(item, value)):
            raise ValueError('Linked distribution inventory member: ' + str(item))
    return path


def _os_alias(item, value):
    """An ancestor symlink only root can change (owned by root inside a root-owned directory that group and others cannot
    write), as macOS /var and /tmp -> /private/... are; the same rule as titanos.link_policy.os_alias. Windows links and
    any other link stay refused."""
    if os.name == 'nt' or not stat.S_ISLNK(value.st_mode) or value.st_uid != 0:
        return False
    try:
        parent = os.lstat(os.path.dirname(os.fspath(item)) or os.sep)
    except OSError:
        return False
    return parent.st_uid == 0 and stat.S_ISDIR(parent.st_mode) and not parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


def _packaged_distribution_records(destination):
    """Inspect shipped metadata, without importing any packaged Python code."""
    destination = Path(destination).absolute()
    records = []
    for parent, directories, _ in os.walk(destination, followlinks=False):
        parent = Path(parent)
        for name in list(directories):
            # Corresponding source is documentation, not an installed runtime.
            if parent == destination and name == 'SOURCE':
                directories.remove(name)
                continue
            path = _inventory_path(destination, (parent / name).relative_to(destination).as_posix())
            if not name.lower().endswith('.dist-info'):
                continue
            directories.remove(name)
            member = path.relative_to(destination).as_posix()
            metadata_path = _inventory_path(destination, member + '/METADATA')
            dist = metadata.PathDistribution(path)
            headers = dist.metadata
            names, versions = headers.get_all('Name', []), headers.get_all('Version', [])
            if (len(names) != 1 or len(versions) != 1 or not names[0] or not versions[0]
                    or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', names[0])):
                raise ValueError('Malformed shipped distribution METADATA: ' + member)
            records.append({'name': canonicalize_name(names[0]), 'version': versions[0],
                            'dist_info': member, 'metadata_sha256': digest(metadata_path)})
    return sorted(records, key=lambda row: row['dist_info'])


def reconcile_packaged_distributions(destination, plan):
    """Add optional imports collected by PyInstaller to the final build inventory."""
    records = _packaged_distribution_records(destination)
    seen, added = set(), []
    versions = {canonicalize_name(name): version for name, version in plan['packages'].items()}
    for row in records:
        name, version = row['name'], row['version']
        if name in seen:
            raise ValueError('Duplicate shipped distribution: ' + name)
        seen.add(name)
        if name in versions and versions[name] != version:
            raise ValueError('Shipped distribution version differs from build plan: ' + name)
        if name not in versions:
            versions[name] = version
            added.append(name)
    plan['packages'] = dict(sorted(versions.items()))
    plan['distributions'] = sorted({canonicalize_name(n) for n in plan['distributions']} | seen)
    plan['collected_distribution_inventory'] = {
        'scope': 'Actual shipped dist-info reconciled after PyInstaller collection; not inferred from Requires-Dist alone',
        'added_distributions': sorted(added), 'distributions': records}
    return plan['collected_distribution_inventory']


ATTRIBUTION_PROPERTIES = frozenset({'titanos:component-role', 'titanos:linkage', 'titanos:source-bundle'})


def _component_property(item, name):
    values = [row.get('value') for row in item.get('properties', []) if isinstance(row, dict) and row.get('name') == name]
    return values[0] if values else None


def _pypi_purl(purl):
    """Canonical (name, version or None) from a pkg:pypi purl; None when the purl names another ecosystem."""
    if not isinstance(purl, str) or purl[:9].lower() != 'pkg:pypi/':
        return None
    body = purl[len('pkg:pypi/'):].split('#', 1)[0].split('?', 1)[0]
    name, _, version = body.partition('@')
    if not name or '/' in name:
        raise ValueError('Malformed PyPI purl: ' + purl)
    return canonicalize_name(unquote(name)), unquote(version) or None


def _component_kind(item):
    """Distinguish a Python distribution identity from a separately attributed component of the same name.

    A pkg:pypi purl is the distribution identity. A component carrying another ecosystem's purl or an
    explicit Titanos attribution property (native linkage, source bundle, component role) describes a
    distinct native/embedded component and is never counted as, or substituted for, the distribution.
    A bare name/version component without either is the legacy distribution shape.
    """
    purl = item.get('purl')
    if _pypi_purl(purl) is not None:
        return 'pypi'
    if purl is not None or any(isinstance(row, dict) and row.get('name') in ATTRIBUTION_PROPERTIES
                               for row in item.get('properties', [])):
        return 'attributed'
    return 'distribution'


def verify_broker_payload(destination, source_root=ROOT):
    """The cloud broker ships in the frozen build: every module of src/titanos/broker is present as source under
    _internal/titanos/broker (titanos is collected with include_py_files and the spec's runtime_module filter keeps every
    titanos.broker module). A packaging exclude that dropped any of them stops the build here."""
    source = Path(source_root) / 'src'
    expected = sorted(path.relative_to(source).as_posix() for path in (source / 'titanos' / 'broker').rglob('*.py')
                      if '__pycache__' not in path.parts)
    internal = Path(destination) / '_internal'
    missing = [name for name in expected if not (internal / name).is_file()]
    if not expected or missing:
        raise RuntimeError('The frozen build lacks broker modules: ' + (', '.join(missing[:10]) or 'src/titanos/broker is empty'))
    return expected


def validate_packaged_distributions(destination):
    """One-way payload -> inventory/notice validation; no executable is required.

    Native/embedded or inert inventory entries need not have dist-info. Isolated
    runtimes use their declared local inventory and scoped CycloneDX components.
    A native component may share a shipped distribution's name (Sherpa ships a
    PyPI wheel and a separately attributed native library); both records are
    preserved, and only Python distribution components establish identity.
    This checks attribution bytes, not legal approval or runtime compatibility.
    """
    destination = Path(destination).absolute()
    inventory = json.loads(_inventory_path(destination, 'DEPENDENCIES.json').read_text(encoding='utf-8'))
    sbom = json.loads(_inventory_path(destination, 'sbom.cdx.json').read_text(encoding='utf-8'))
    for row in inventory.get('vendored_web_assets', []):
        # vendored web assets and their retained license copies are re-hashed against the inventory that shipped them
        for item in row.get('files', []):
            shipped = destination / item['packaged_path']
            if not shipped.is_file() or digest(shipped) != item['sha256']:
                raise ValueError('Vendored web asset differs from DEPENDENCIES.json: ' + item['packaged_path'])
        for item in row.get('license_files', []):
            copied = destination / item['path']
            if not copied.is_file() or digest(copied) != item['sha256']:
                raise ValueError('Vendored web asset license copy differs from DEPENDENCIES.json: ' + item['path'])
    scopes = {'': {'root': destination, 'inventory': inventory, 'manifest': None}}
    for runtime in inventory.get('external_runtimes', []):
        location = runtime['location']
        root = _inventory_path(destination, location)
        if location in scopes:
            raise ValueError('Duplicate isolated runtime location: ' + location)
        path = _inventory_path(destination, runtime['dependency_inventory'])
        if not path.is_relative_to(root):
            raise ValueError('Isolated dependency inventory escaped its runtime')
        manifest = _inventory_path(destination, location + '/runtime.json')
        if digest(manifest) != runtime['manifest_sha256']:
            raise ValueError('Isolated runtime manifest differs: ' + location)
        scopes[location] = {'root': root, 'inventory': json.loads(path.read_text(encoding='utf-8')),
                            'manifest': json.loads(manifest.read_text(encoding='utf-8'))['files']}
    records = _packaged_distribution_records(destination)
    seen, attributed = set(), []
    for row in records:
        path = destination / row['dist_info']
        locations = [name for name in scopes if name and path.is_relative_to(scopes[name]['root'])]
        location = max(locations, key=len) if locations else ''
        if not location and not path.is_relative_to(destination / '_internal'):
            raise ValueError('Shipped distribution is outside a declared runtime: ' + row['dist_info'])
        identity = (location, row['name'])
        if identity in seen:
            raise ValueError('Duplicate shipped distribution: ' + row['name'])
        seen.add(identity)
        scope = scopes[location]
        matches = [item for item in scope['inventory']['components']
                   if canonicalize_name(item['name']) == row['name']]
        if len(matches) != 1 or matches[0].get('version') != row['version']:
            raise ValueError('DEPENDENCIES missing, duplicate or version mismatch for shipped distribution: ' + row['name'])
        candidates = [item for item in sbom['components']
                      if canonicalize_name(str(item.get('name', ''))) == row['name']
                      and (_component_property(item, 'titanos:runtime') or '') == location]
        kinds = {kind: [item for item in candidates if _component_kind(item) == kind]
                 for kind in ('pypi', 'distribution', 'attributed')}
        for item in kinds['attributed']:
            attributed.append({'name': row['name'], 'runtime': location, 'version': item.get('version'),
                               'purl': item.get('purl'), 'role': _component_property(item, 'titanos:component-role')})
        if kinds['pypi'] and kinds['distribution']:
            raise ValueError('SBOM has conflicting Python distribution components for shipped distribution: ' + row['name'])
        components = kinds['pypi'] or kinds['distribution']
        if len(components) != 1 or components[0].get('version') != row['version']:
            raise ValueError('SBOM missing, duplicate or version mismatch for shipped distribution: ' + row['name'])
        if kinds['pypi']:
            purl_name, purl_version = _pypi_purl(components[0]['purl'])
            if purl_name != row['name'] or purl_version not in (None, row['version']):
                raise ValueError('SBOM PyPI purl differs from the shipped distribution identity: ' + row['name'])
        notices = matches[0].get('license_files', [])
        if not notices:
            # an isolated runtime distribution without inventory notices may carry hash-bound license copies supplemented by the
            # application packager (supplement_runtime_licenses); they are re-hashed here against DEPENDENCIES.json
            runtime_entry = next((r for r in inventory.get('external_runtimes', []) if r.get('location') == location), None) if location else None
            supplements = [s for s in (runtime_entry or {}).get('supplemented_licenses', [])
                           if canonicalize_name(s['name']) == row['name'] and s.get('version') == row['version'] and s.get('files')]
            for supplement in supplements:
                for item in supplement['files']:
                    copied = destination / item['path']
                    if not copied.is_file() or digest(copied) != item['sha256']:
                        raise ValueError('Supplemented runtime license copy differs from DEPENDENCIES.json: ' + item['path'])
            if not supplements:
                exception = matches[0].get('license_exception', {})
                if not all(isinstance(exception.get(key), str) and exception[key].strip()
                           for key in ('reason', 'reviewed_by', 'reference')):
                    raise ValueError('Missing notices or explicit reviewed license exception: ' + row['name'])
        for notice in notices:
            if isinstance(notice, str) and scope['manifest'] is not None:
                name, expected = notice, scope['manifest'][notice]['sha256']
            else:
                name, expected = notice['path'], notice['sha256']
            notice_path = _inventory_path(scope['root'], name)
            if not notice_path.is_file() or digest(notice_path) != expected:
                raise ValueError('Packaged notice hash differs: ' + row['name'] + ': ' + name)
    return {'status': 'packaged_distributions_reconciled', 'distributions': records,
            'separately_attributed_components': attributed, 'native_ffmpeg': _verify_ffmpeg_inventory(destination),
            'scope': 'Every shipped dist-info outside corresponding SOURCE and every FFmpeg row of NATIVE_COMPONENTS.json (binaries and '
                     'members as shipped, retained LGPL texts, the notices section); one-way inventory/notice validation, no runtime execution or legal approval'}


def apply_attribution_supplements(destination, name, version, dist_info_dir, files, component, notes, catalog=None):
    """Copy retained attribution supplements for one packaged distribution into the bundle and describe them.

    `tools/licenses/supplements.json` rows are keyed by distribution name and version. Retained license and attribution
    files are hash-checked before copying; a declared dist-info member (for example an upstream SBOM) is copied from the
    packaged metadata when present, and its components are nested under the shipped SBOM component. Returns the extra
    DEPENDENCIES.json fields for the distribution (empty when no supplement applies)."""
    destination = Path(destination)
    catalog_path = Path(catalog or ROOT / 'tools' / 'licenses' / 'supplements.json')
    if not catalog_path.is_file():
        return {}  # no retained supplements declared for this source tree
    rows = [row for row in json.loads(catalog_path.read_text(encoding='utf-8')) if row['name'] == name and row['version'] == version]
    extra = {}
    for row in rows:
        for item in row.get('license_files', []):
            source = ROOT / 'tools' / 'licenses' / item['file']
            if digest(source) != item['sha256']:
                raise ValueError('Retained attribution supplement hash mismatch for ' + name + ': ' + item['file'])
            target = destination / item['target']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            files.append({'path': item['target'], 'sha256': digest(target), **({'source_url': item['url']} if item.get('url') else {})})
        for item in row.get('attribution_files', []):
            source = ROOT / 'tools' / 'licenses' / item['file']
            if digest(source) != item['sha256']:
                raise ValueError('Retained attribution supplement hash mismatch for ' + name + ': ' + item['file'])
            target = destination / item['target']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            extra[item['role']] = item['target']
        for item in row.get('dist_info_members', []):
            source = Path(dist_info_dir) / item['member'] if dist_info_dir else None
            if source is None or not source.is_file():
                continue
            target = destination / item['target']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            extra[item['role']] = item['target']
            if item.get('nest_components'):
                upstream = json.loads(target.read_text(encoding='utf-8'))
                children = []
                for child in upstream.get('components', []):
                    child = dict(child)
                    child['bom-ref'] = name + '-vendored:' + str(child.get('bom-ref') or child.get('name'))
                    child.setdefault('properties', []).append({'name': 'titanos:attribution-source', 'value': item['target']})
                    children.append(child)
                if children:
                    component['components'] = children
        extra['attribution_note'] = row['why']
        extra['license_review'] = row.get('license_review', 'not_performed')
        component.setdefault('properties', []).extend([
            {'name': 'titanos:license-review', 'value': row.get('license_review', 'not_performed')},
            *([{'name': 'titanos:source-attribution', 'value': extra['source_attribution']}] if 'source_attribution' in extra else [])])
        notes.append(row['notice_markdown'])
    return extra


# Studio's own web files; every other .js/.css under the packaged web directory must be a catalogued vendored asset.
STUDIO_WEB_ASSETS = frozenset({'app.js', 'i18n.js', 'style.css', 'retrieval.js'})


def apply_vendored_web_assets(destination, components, notes, catalog=None):
    """Vendored web assets (tools/licenses/vendored_assets.json): files shipped inside the application that are not Python
    distributions (the Swagger UI files behind /docs). Every catalogued file must be present in the built application with
    its catalogued sha256, its retained license/notice texts are hash-checked and copied under THIRD_PARTY_LICENSES/<name>/,
    an SBOM component with the ecosystem purl is appended and the notice section is queued. Returns the DEPENDENCIES.json
    records; raises ValueError when a shipped asset is missing or differs, and when any .js/.css under the packaged web
    directory is neither a Studio file nor catalogued (so a removed or renamed catalog cannot ship an unbound asset)."""
    destination = Path(destination)
    catalog_path = Path(catalog or ROOT / 'tools' / 'licenses' / 'vendored_assets.json')
    rows = json.loads(catalog_path.read_text(encoding='utf-8')) if catalog_path.is_file() else []
    catalogued = {Path(item['packaged_path']).name for row in rows for item in row['files']}
    web_dir = destination / '_internal' / 'titanos' / 'web'
    stray = sorted(p.name for p in web_dir.glob('*') if p.is_file() and p.suffix.lower() in {'.js', '.css'} and p.name not in STUDIO_WEB_ASSETS and p.name not in catalogued) if web_dir.is_dir() else []
    # every file under web/fonts (the vendored Chakra Petch faces and their OFL) is a catalogued vendored asset
    stray += sorted('fonts/' + p.name for p in (web_dir / 'fonts').glob('*') if p.is_file() and p.name not in catalogued) if (web_dir / 'fonts').is_dir() else []
    if stray:
        raise ValueError('Uncatalogued web asset shipped (add it to tools/licenses/vendored_assets.json with its license): ' + ', '.join(stray))
    records = []
    for row in rows:
        files = []
        for item in row['files']:
            shipped = destination / item['packaged_path']
            if not shipped.is_file():
                raise ValueError('Vendored web asset missing from the application: ' + item['packaged_path'])
            if digest(shipped) != item['sha256'] or shipped.stat().st_size != item['bytes']:
                raise ValueError('Vendored web asset differs from its catalog: ' + item['packaged_path'])
            files.append({'packaged_path': item['packaged_path'], 'route': item.get('route'), 'sha256': item['sha256'], 'bytes': item['bytes']})
        licenses = []
        for item in row['license_files']:
            source = ROOT / 'tools' / 'licenses' / item['file']
            if digest(source) != item['sha256']:
                raise ValueError('Retained vendored-asset license hash mismatch for ' + row['name'] + ': ' + item['file'])
            target = destination / item['target']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            licenses.append({'path': item['target'], 'sha256': digest(target), 'source_url': item['url']})
        components.append({'type': 'library', 'name': row['name'], 'version': row['version'], 'purl': row['purl'], 'scope': 'required',
                           'licenses': [{'expression': row['license_expression']}],
                           'externalReferences': [{'type': 'distribution', 'url': row['source_url']}],
                           'properties': [{'name': 'titanos:vendored-web-asset', 'value': f['packaged_path'] + ' sha256:' + f['sha256']} for f in files]
                                         + [{'name': 'titanos:source-integrity', 'value': row['source_integrity']},
                                            {'name': 'titanos:license-review', 'value': row.get('license_review', 'not_performed')}]})
        notes.append(row['notice_markdown'])
        records.append({'name': row['name'], 'version': row['version'], 'kind': row.get('kind', 'vendored web asset'), 'purl': row['purl'],
                        'declared_license': row['license_expression'], 'source_url': row['source_url'], 'source_integrity': row['source_integrity'],
                        'files': files, 'license_files': licenses, 'license_review': row.get('license_review', 'not_performed'), 'attribution_note': row['why']})
    return records


FFMPEG_LOCATION = '_internal/native/ffmpeg'
FFMPEG_LICENSE_TEXTS = ('COPYING.LGPLv2.1', 'LICENSE.md')
FFMPEG_BUILD_SCRIPT_SUFFIXES = {'.sh', '.py', '.ps1', '.bat', '.cmd'}
FFMPEG_NOTICE_HEADING = '## FFmpeg'


def ffmpeg_version(native, source_archive, version_text=None):
    """The bundle's FFmpeg version: BUILD.json `ffmpeg_version` and the corresponding-source archive name must agree, and the
    packaged version.txt (when the bundle carries one) must announce the same version."""
    match = re.fullmatch(r'ffmpeg-(\d[0-9A-Za-z.]*?)\.tar\.[0-9A-Za-z]+', Path(source_archive).name)
    declared = native.get('ffmpeg_version')
    if match is None and declared is None:
        raise ValueError('The FFmpeg version is neither declared in BUILD.json nor readable from the corresponding-source archive name')
    if match is not None and declared is not None and str(declared) != match.group(1):
        raise ValueError('FFmpeg BUILD.json ffmpeg_version differs from the corresponding-source archive name')
    version = str(declared) if declared is not None else match.group(1)
    if version_text is not None and not version_text.startswith('ffmpeg version ' + version + ' '):
        raise ValueError('The packaged FFmpeg version.txt announces another version than the bundle record')
    return version


def check_ffmpeg_lgpl_configuration(native):
    """The bundle configuration must exclude GPL and nonfree components (--disable-gpl and --disable-nonfree present, no
    --enable-gpl / --enable-nonfree, no gpl/nonfree truth and no external codec library); otherwise the LGPL claim of the
    notice would not describe the shipped binaries."""
    configure = [str(flag) for flag in native.get('configure', [])]
    problems = []
    for flag in ('--disable-gpl', '--disable-nonfree'):
        if flag not in configure:
            problems.append(flag + ' absent')
    for flag in configure:
        if flag.startswith(('--enable-gpl', '--enable-nonfree')):
            problems.append(flag + ' present')
    for key in ('gpl', 'nonfree'):
        if native.get(key) is True:
            problems.append(key + ' recorded true')
    if native.get('external_codec_libraries'):
        problems.append('external codec libraries: ' + ', '.join(str(x) for x in native['external_codec_libraries']))
    if problems:
        raise ValueError('FFmpeg bundle configuration does not exclude GPL and nonfree components: ' + '; '.join(problems))
    return configure


def include_ffmpeg_component(destination, plan, components, notes):
    """Bind the packaged FFmpeg bundle and write what the LGPL asks a distributor to say about it.

    - the bundle is re-planned (ffmpeg_plan) and must equal the planned binaries; native_inventory binds the packaged
      binaries (identical bytes, or verified Mach-O postprocessing);
    - every member of the bundle is hashed **as shipped** after normalize_packaged_build_metadata rewrote the configure
      --prefix in the text members: a changed member is accepted only when it is one of FFMPEG_NORMALIZED_TEXT, the normalized
      dylib diagnostics or a binary native_inventory accounted for, and keeps the bundle's hash under `normalized_from`
      (`source_build_sha256` for a postprocessed binary), so a verifier hashing the shipped members finds no mismatch;
    - the LGPL texts COPYING.LGPLv2.1 and LICENSE.md are copied to THIRD_PARTY_LICENSES/ffmpeg/ (COPYING.LGPLv2.1 is also
      checked against the corresponding-source archive) and THIRD_PARTY_NOTICES.md gains the FFmpeg section (version,
      LGPL-2.1-or-later, GPL and nonfree excluded by configuration, where the licence texts are, the complete corresponding
      source and build scripts shipped in the distribution with their hashes, and the written offer of LGPL-2.1 section 6);
    - the packaged BUILD.json and the NATIVE_COMPONENTS.json row carry the same record, `binaries_derived_from_members` when
      the bundle had no binaries mapping, and `license_files`; validate_packaged_distributions re-checks all of it.
    Returns the NATIVE_COMPONENTS.json row."""
    from native_inventory import finalize_native_inventory
    destination = Path(destination)
    source = Path(plan['ffmpeg_dir']).resolve()
    bound = ffmpeg_plan(source)
    if plan.get('ffmpeg_binaries') is not None and bound['ffmpeg_binaries'] != plan['ffmpeg_binaries']:
        raise ValueError('FFmpeg inputs changed during build')
    binaries, derived = bound['ffmpeg_binaries'], bound['ffmpeg_binaries_derived_from_members']
    target = (destination / FFMPEG_LOCATION).resolve()
    if not target.is_relative_to(destination.resolve()) or not target.is_dir():
        raise ValueError('The packaged FFmpeg directory is missing')
    native = finalize_native_inventory(destination, source, plan['pyinstaller'], binaries=binaries)
    diagnostics = normalize_dylib_diagnostics(source, target)
    if diagnostics is not None:
        native = {**native, 'diagnostic_metadata': diagnostics}
    bundle = json.loads((source / 'BUILD.json').read_text(encoding='utf-8'))
    configure = check_ffmpeg_lgpl_configuration(bundle)
    archive = next(path for path in sorted((source / 'source').glob('ffmpeg-*.tar.*')) if digest(path) == bundle['source_sha256'])
    version_text = (target / 'version.txt').read_text(encoding='utf-8') if (target / 'version.txt').is_file() else None
    version = ffmpeg_version(bundle, archive, version_text)
    # Members as shipped: every bundle file except BUILD.json (rewritten below) is in the packaged copy and nothing else is.
    inventory = bundle.get('members') or {}
    members, scripts, others = {}, {}, {}
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name == 'BUILD.json':
            continue
        name = path.relative_to(source).as_posix()
        shipped = (target / name).resolve()
        if not shipped.is_relative_to(target) or not shipped.is_file():
            raise ValueError('FFmpeg member missing from the packaged copy: ' + name)
        before = {'bytes': path.stat().st_size, 'sha256': inventory[name]['sha256'] if name in inventory else digest(path)}
        row = {'bytes': shipped.stat().st_size, 'sha256': digest(shipped)}
        if row['sha256'] != before['sha256']:
            if name in FFMPEG_NORMALIZED_TEXT:
                row['normalized_from'] = {**before, 'normalization': 'configure --prefix rewritten to <install-prefix>'}
            elif name == 'dylib-dependencies.json' and diagnostics is not None:
                row['normalized_from'] = {**before, 'normalization': diagnostics['normalization']}
            elif name in binaries and row['sha256'] == native['binaries'].get(name):
                row['source_build_sha256'] = before['sha256']
            else:
                raise ValueError('FFmpeg member changed during packaging: ' + name)
        members[name] = row
        if PurePosixPath(name).parent == PurePosixPath('source') and name != archive.relative_to(source).as_posix():
            (scripts if PurePosixPath(name).suffix.lower() in FFMPEG_BUILD_SCRIPT_SUFFIXES else others)[name] = row['sha256']
    extra = sorted(p.relative_to(target).as_posix() for p in target.rglob('*')
                   if p.is_file() and p.name != 'BUILD.json' and p.relative_to(target).as_posix() not in members)
    if extra:
        raise ValueError('The packaged FFmpeg directory carries files the bundle does not: ' + ', '.join(extra[:10]))
    if not scripts:
        raise ValueError('The FFmpeg bundle ships no build script beside its corresponding-source archive')
    # The licence texts: present as shipped, COPYING.LGPLv2.1 equal to the archive's, copied under THIRD_PARTY_LICENSES/ffmpeg/.
    for name in FFMPEG_LICENSE_TEXTS:
        if name not in members or 'normalized_from' in members[name]:
            raise ValueError('FFmpeg corresponding license text is missing from the bundle: ' + name)
    with tarfile.open(archive) as tar:
        matches = [item for item in tar.getmembers() if item.isfile() and Path(item.name).name == 'COPYING.LGPLv2.1']
        if len(matches) != 1 or tar.extractfile(matches[0]).read() != (target / 'COPYING.LGPLv2.1').read_bytes():
            raise ValueError('FFmpeg license text differs from corresponding source')
    licenses = []
    folder = destination / 'THIRD_PARTY_LICENSES' / 'ffmpeg'
    folder.mkdir(parents=True, exist_ok=True)
    for name in FFMPEG_LICENSE_TEXTS:
        copied = folder / name
        shutil.copy2(target / name, copied)
        if digest(copied) != members[name]['sha256']:
            raise ValueError('FFmpeg license copy readback differs: ' + name)
        licenses.append({'path': copied.relative_to(destination).as_posix(), 'sha256': members[name]['sha256'],
                         'source': FFMPEG_LOCATION + '/' + name})
    archive_name = archive.relative_to(source).as_posix()
    corresponding = {'archive': FFMPEG_LOCATION + '/' + archive_name, 'archive_sha256': bundle['source_sha256'],
                     'archive_bytes': members[archive_name]['bytes'], 'source_url': bundle.get('source_url'),
                     'source_changes': bundle.get('source_changes'),
                     'build_scripts': {FFMPEG_LOCATION + '/' + name: value for name, value in scripts.items()},
                     'other_files': {FFMPEG_LOCATION + '/' + name: value for name, value in others.items()}}
    # Keep corresponding-source and binary hashes, without the lab install path.
    record = {**native, 'version': version, 'binaries_derived_from_members': derived, 'members': members,
              'configure': ['--prefix=<install-prefix>' if flag.startswith('--prefix=') else flag for flag in configure],
              'license_files': licenses, 'corresponding_source': corresponding, 'notice_section': FFMPEG_NOTICE_HEADING,
              'metadata_normalization': 'Install prefix normalized in configure and in the packaged ' + ', '.join(FFMPEG_NORMALIZED_TEXT)
                                        + '; members record every file as shipped with the bundle hash under normalized_from; '
                                          'source archive and native binary hashes unchanged.'}
    for key in ('install_prefix', 'prefix'):
        if key in record:
            record[key] = '<install-prefix>'
    (target / 'BUILD.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    components.append({'type': 'application', 'name': 'FFmpeg', 'version': version,
                       'licenses': [{'expression': native['license_from_binary']}],
                       'externalReferences': [{'type': 'distribution', 'url': native['source_url']}]})
    texts = '; '.join(item['path'] + ' (sha256 ' + item['sha256'] + ')' for item in licenses)
    scripts_text = ', '.join(name + ' (sha256 ' + value + ')' for name, value in corresponding['build_scripts'].items())
    others_text = (' and ' + ', '.join(sorted(corresponding['other_files']))) if corresponding['other_files'] else ''
    notes.append(FFMPEG_NOTICE_HEADING + '\n\nThis application ships FFmpeg ' + version + ' under `' + FFMPEG_LOCATION + '/` ('
                 + str(len(binaries)) + ' executable and library files), built from source under the GNU Lesser General Public '
                 'License, version 2.1 or later (' + native['license_from_binary'] + '). Its configuration disables the GPL and '
                 'nonfree components (`--disable-gpl`, `--disable-nonfree`) and links no external codec library, so no GPL-licensed '
                 'or nonfree FFmpeg component is included; the configure line is recorded in `' + FFMPEG_LOCATION + '/BUILD.json` and '
                 'in the packaged `version.txt` and `license-output.txt` with the install prefix normalized. The licence texts are '
                 'retained in ' + texts + ' and beside the binaries. The complete corresponding source ships in this distribution: '
                 '`' + corresponding['archive'] + '` (sha256 ' + corresponding['archive_sha256'] + ', ' + str(corresponding['archive_bytes'])
                 + ' bytes' + ((', from ' + corresponding['source_url']) if corresponding['source_url'] else '') + ') with the build '
                 'script(s) ' + scripts_text + others_text + '; `NATIVE_COMPONENTS.json` records every member as shipped. Written offer: '
                 'in addition to the source shipped here, the distributor of this package offers to give any third party, for a '
                 'charge no more than the cost of performing the distribution, a complete machine-readable copy of the same '
                 'corresponding source for this FFmpeg build, valid for at least three years from the date of this distribution '
                 '(LGPL-2.1 section 6). These sentences describe the shipped bytes and the recorded configuration; they are not a '
                 'legal review.')
    return {'name': 'FFmpeg', 'location': FFMPEG_LOCATION, **record}


def _verify_ffmpeg_inventory(destination):
    """Every FFmpeg row of NATIVE_COMPONENTS.json is re-checked against the shipped bytes: the binaries and every member as shipped
    (sizes and hashes; nothing in the packaged directory outside the members but the rewritten BUILD.json, which must carry the
    same binaries and members), the retained COPYING.LGPLv2.1 and LICENSE.md under THIRD_PARTY_LICENSES/ffmpeg/, and the FFmpeg
    section of THIRD_PARTY_NOTICES.md naming the licence expression and the shipped corresponding-source archive. A row without
    members, license texts or the section is refused: that is the shape the 2026-10-06 builds of 5aa3311 shipped (no section, no
    retained texts, two stale member hashes in the Linux package). A packaged FFmpeg directory that no row binds (an empty or
    missing inventory) is refused as well; a distribution without the directory reports no FFmpeg."""
    destination = Path(destination).absolute()
    native_path = destination / 'NATIVE_COMPONENTS.json'
    packaged_dir = destination / FFMPEG_LOCATION
    unbound = 'Packaged FFmpeg directory ' + FFMPEG_LOCATION + ' is not bound by an FFmpeg row of NATIVE_COMPONENTS.json'
    entries = json.loads(_inventory_path(destination, 'NATIVE_COMPONENTS.json').read_text(encoding='utf-8')) if native_path.is_file() else None
    if not isinstance(entries, list):
        if packaged_dir.is_dir():
            raise ValueError(unbound)
        return []
    results = []
    for row in entries:
        if not isinstance(row, dict) or row.get('name') != 'FFmpeg':
            continue
        location = row.get('location')
        if not isinstance(location, str) or not location:
            raise ValueError('FFmpeg inventory row lacks its packaged location')
        root = _inventory_path(destination, location)
        if not root.is_dir():
            raise ValueError('Packaged FFmpeg directory is missing: ' + location)
        binaries, members, licenses = row.get('binaries'), row.get('members'), row.get('license_files')
        if not isinstance(binaries, dict) or not binaries:
            raise ValueError('FFmpeg inventory row lacks its binaries mapping')
        if not isinstance(members, dict) or not members:
            raise ValueError('FFmpeg inventory row lacks its members as shipped')
        if not isinstance(licenses, list) or not licenses:
            raise ValueError('FFmpeg inventory row lacks its retained license texts')
        for name, expected in binaries.items():
            path = _inventory_path(root, name)
            if not path.is_file() or digest(path) != expected:
                raise ValueError('Packaged FFmpeg binary differs from NATIVE_COMPONENTS.json: ' + name)
        for name, item in members.items():
            path = _inventory_path(root, name)
            if not path.is_file() or path.stat().st_size != item.get('bytes') or digest(path) != item.get('sha256'):
                raise ValueError('Packaged FFmpeg member differs from NATIVE_COMPONENTS.json: ' + name)
        shipped = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name != 'BUILD.json')
        if shipped != sorted(members):
            raise ValueError('Packaged FFmpeg directory differs from its members inventory: ' + ', '.join(sorted(set(shipped) ^ set(members))[:10]))
        copied = json.loads(_inventory_path(root, 'BUILD.json').read_text(encoding='utf-8'))
        if copied.get('binaries') != binaries or copied.get('members') != members:
            raise ValueError('Packaged FFmpeg BUILD.json disagrees with NATIVE_COMPONENTS.json')
        names = set()
        for item in licenses:
            path = _inventory_path(destination, item['path'])
            if not path.is_file() or digest(path) != item['sha256']:
                raise ValueError('Retained FFmpeg license text differs from NATIVE_COMPONENTS.json: ' + str(item.get('path')))
            names.add(PurePosixPath(item['path']).name)
        if not set(FFMPEG_LICENSE_TEXTS) <= names:
            raise ValueError('Retained FFmpeg license texts must include ' + ' and '.join(FFMPEG_LICENSE_TEXTS))
        heading = row.get('notice_section') or FFMPEG_NOTICE_HEADING
        text = _inventory_path(destination, 'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')
        section = [line for line in text.splitlines() if line.strip() == heading]
        archive = str((row.get('corresponding_source') or {}).get('archive') or '')
        if len(section) != 1 or str(row.get('license_from_binary') or 'LGPL-2.1') not in text or not archive or archive not in text:
            raise ValueError('THIRD_PARTY_NOTICES.md lacks the FFmpeg section naming the licence and the shipped corresponding source')
        results.append({'location': location, 'binaries': len(binaries), 'members': len(members), 'license_files': len(licenses),
                        'binaries_derived_from_members': bool(row.get('binaries_derived_from_members'))})
    if packaged_dir.is_dir() and not any(item['location'] == FFMPEG_LOCATION for item in results):
        raise ValueError(unbound)
    return results


def write_notices(destination, plan):
    normalize_packaged_build_metadata(destination, plan)
    observed = reconcile_packaged_distributions(destination, plan)
    shipped = {row['name']: row for row in observed['distributions']}
    directory = destination / 'THIRD_PARTY_LICENSES'
    directory.mkdir()
    inventory, components, supplement_notes = [], [], []
    retained_licenses = json.loads((ROOT / 'tools' / 'licenses' / 'provenance.json').read_text(encoding='utf-8'))
    for name in plan['distributions']:
        dist = (metadata.PathDistribution(Path(destination) / shipped[name]['dist_info'])
                if name in shipped else metadata.distribution(name))
        if dist.version != plan['packages'][name]:
            raise ValueError('Distribution changed before notice generation: ' + name)
        declared = dist.metadata.get('License-Expression') or dist.metadata.get('License') or ''
        files = []
        for item in dist.files or []:
            filename = Path(str(item)).name
            if not filename.upper().startswith(('LICENSE', 'LICENCE', 'COPYING', 'NOTICE', 'AUTHORS')):
                continue
            source = Path(dist.locate_file(item))
            if name in shipped:
                relative = source.relative_to(Path(destination))
                source = _inventory_path(destination, relative.as_posix())
            if not source.is_file():
                continue
            target = directory / name / (str(len(files) + 1) + '-' + filename)
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            files.append({'path': target.relative_to(destination).as_posix(), 'sha256': digest(target)})
        if not files:
            retained = next((row for row in retained_licenses if row['name'] == name and row['version'] == dist.version), None)
            if retained:
                source = ROOT / 'tools' / 'licenses' / retained['file']
                if digest(source) != retained['sha256']:
                    raise ValueError('Retained upstream license hash mismatch for ' + name)
                target = directory / name / retained['file']
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(source, target)
                files.append({'path': target.relative_to(destination).as_posix(), 'sha256': digest(target),
                              'source_url': retained['url']})
        component = {'type': 'library', 'name': name, 'version': dist.version,
                     'purl': f'pkg:pypi/{name}@{dist.version}', 'scope': 'required'}
        if dist.metadata.get('License-Expression'):
            component['licenses'] = [{'expression': dist.metadata['License-Expression']}]
        supplement = apply_attribution_supplements(destination, name, dist.version,
                                                   (Path(destination) / shipped[name]['dist_info']) if name in shipped else None,
                                                   files, component, supplement_notes)
        inventory.append({'name': name, 'version': dist.version, 'declared_license': declared,
                          'license_classifiers': [value for value in dist.metadata.get_all('Classifier', []) if value.startswith('License ::')],
                          'license_files': files, 'license_review': supplement.pop('license_review', 'not_performed'), **supplement})
        components.append(component)
    components.append({'type': 'library', 'name': 'CPython', 'version': platform.python_version(),
                       'licenses': [{'expression': 'PSF-2.0'}]})
    for source in (Path(sys.base_prefix) / 'LICENSE.txt', Path(sys.base_prefix) / 'LICENSE',
                   Path(sysconfig.get_path('stdlib')) / 'LICENSE.txt'):
        if source.is_file():
            shutil.copy2(source, directory / 'CPython-LICENSE.txt')
            break
    else:
        raise ValueError('CPython license text is missing from the interpreter prefix and standard library')
    native_components = []
    if plan.get('native_tesseract'):
        root = destination / '_internal/native/tesseract'
        native_components.append({'name': 'Tesseract', 'location': '_internal/native/tesseract',
                                  'executable_sha256': plan['native_tesseract']['executable_sha256'],
                                  'source': plan['native_tesseract']['source'], 'models_included': False})
        components.extend(json.loads((root / 'SBOM.cdx.json').read_text())['components'])
    if plan.get('ffmpeg_dir'):
        native_components.append(include_ffmpeg_component(destination, plan, components, supplement_notes))
    elif 'imageio_ffmpeg' in plan['collect_packages']:
        native_components.append({'name': 'FFmpeg from imageio_ffmpeg wheel', 'status': 'corresponding source and native dependencies require separate review'})
    if plan.get('native_audio'):
        for library in plan['native_audio']['libraries']:
            target = destination / library['runtime_path']
            if digest(target) != library['sha256']:
                raise ValueError('Packaged native audio library differs from its verified build: ' + str(target))
            component = library['component']
            location = '_internal/native/audio/' + component
            record = json.loads((destination / location / 'BUILD.json').read_text(encoding='utf-8'))
            native_components.append({'name': component, 'location': location, 'runtime_path': library['runtime_path'],
                                      'packaged_sha256': digest(target), 'source_build': record})
            components.append({'type': 'library', 'name': record.get('name', record.get('component')),
                               'version': record['version'], 'licenses': [{'expression': 'LGPL-2.1-or-later'}]})
        expected = next(x['sha256'] for x in plan['native_audio']['libraries'] if x['component'] == 'libsndfile-minimal')
        if any(digest(path) != expected for path in (destination / '_internal').rglob('*sndfile*.dll')):
            raise ValueError('An unverified libsndfile DLL was collected alongside the minimal library')
    if 'qwen_tts' in plan['collect_packages']:
        from titanos.qwen_tts_backend import WRAPPER_REVISION
        provenance = list((destination / '_internal').glob('qwen*dist-info/direct_url.json'))
        if len(provenance) != 1 or json.loads(provenance[0].read_text())['vcs_info']['commit_id'] != WRAPPER_REVISION:
            raise ValueError('Frozen Qwen wrapper commit metadata is missing or differs')
    if any('cu' in value for key, value in plan['packages'].items() if key == 'torch'):
        native_components.append(cuda_vendor_component('main runtime', plan['packages']['torch'], 'THIRD_PARTY_LICENSES/torch', '_internal/torch/lib'))
    vendored_web_assets = apply_vendored_web_assets(destination, components, supplement_notes)
    (destination / 'NATIVE_COMPONENTS.json').write_text(json.dumps(native_components, indent=2) + '\n', encoding='utf-8')
    (destination / 'DEPENDENCIES.json').write_text(json.dumps({'scope': 'Runtime dependency closure selected for this build; optional missing runtimes are excluded',
                      'python': platform.python_version(), 'components': inventory, 'vendored_web_assets': vendored_web_assets,
                      'dependency_exceptions': plan['dependency_exceptions'],
                      'restricted_opt_in': plan.get('restricted_opt_in', [])}, indent=2) + '\n', encoding='utf-8')
    (destination / 'sbom.cdx.json').write_text(json.dumps({'bomFormat': 'CycloneDX', 'specVersion': '1.6',
                      'serialNumber': 'urn:uuid:' + str(uuid.uuid4()), 'version': 1,
                      'metadata': {'timestamp': datetime.now(timezone.utc).isoformat(),
                          'component': {'type': 'application', 'name': 'Titanos', 'version': plan['version']}},
                      'components': components}, indent=2) + '\n', encoding='utf-8')
    (destination / 'THIRD_PARTY_NOTICES.md').write_text(
        '# Third-party components\n\nTitanos source retains its Apache-2.0 LICENSE and NOTICE. '
        'Bundled dependencies retain their own licenses. DEPENDENCIES.json records installed metadata '
        'and copied notice files; sbom.cdx.json records the selected runtime dependency closure. '
        'A missing license file or metadata value requires review before redistribution. '
        'This generated inventory is not a legal approval or a claim that every binary dependency '
        'uses an open-source license. GPU runtime redistributables can have vendor terms. '
        'Models and their licenses are separate and no model weights are included.\n'
        + ''.join('\n' + note for note in supplement_notes), encoding='utf-8')
    add_bootloader_metadata(destination, plan['pyinstaller'])
    if plan.get('native_sherpa'):
        from sherpa_package import include_native_sources
        include_native_sources(destination, plan['native_sherpa'])
    validate_packaged_distributions(destination)


def package_manifest(built):
    """MANIFEST.json's body: {relative posix path: {'bytes', 'sha256'}} for every file under the sealed build directory.

    The `SOURCE/<path>` keys are the binding for the exported source set (the lab build verifier and
    tests/check_public_export_hygiene.py read them); main() writes the same mapping into the build.
    """
    built = Path(built)
    return {path.relative_to(built).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
            for path in sorted(built.rglob('*')) if path.is_file()}


def source_package(destination):
    from fnmatch import fnmatchcase

    # This is an explicit export policy, not a parser for arbitrary Git ignores.
    private_directories = {'__pycache__', '.git', '.ssh', '.cache', '.pytest_cache', '.mypy_cache',
                           '.ruff_cache', '.titanos', '.venv', 'venv', 'env', 'node_modules',
                           'build', 'dist', 'release', 'weights', 'models', 'output', 'outputs',
                           'verification', 'hardware-window'}
    private_names = {'current.json', 'current.md', 'stop', 'root_release.json', 'release.json',
                     'build_release.json', 'driver_attempt.json', 'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519',
                     # lab-specific: the matrix kit's pinned population of this lab's pre-boundary RUN_RESULT hashes (tools/matrix_kit/
                     # legacy_results.json) describes one operator's retained runs, not the product; it stays internal to the repository
                     'legacy_results.json',
                     # lab_hosts.private.json: the private host map tools/lab_paths.py reads (inventory names to role labels); never
                     # exported (tools/public_export.json private_files)
                     'lab_hosts.private.json'}
    private_patterns = ('*.py[cod]', '*.safetensors', '*.gguf', '*.onnx', '*.pt', '*.pth', '*.ckpt', '*.bin',
                        '*.exe', '*.dll', '*.pyd', '*.so', '*.so.*', '*.dylib', '*.msi', '*.whl',
                        '*.zip', '*.7z', '*.tar', '*.tar.*', '*.tgz', '*.wav', '*.mp3', '*.flac', '*.mp4',
                        '*.log', '*.key', '*.pem', '*.pfx', '*.p12', 'credentials*.json', 'secrets*.json',
                        'token.json', 'tokens.json', '.env', '.env.*')

    def ordinary(path):
        info = path.lstat()
        if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Source export refuses links/reparse points: ' + path.relative_to(ROOT).as_posix())

    # Lab operational records and archived control trees the public export policy names (tools/public_export.json, excluded_paths:
    # docs/evidence, docs/CLUSTER_HANDOFF.md, docs/validation, tools/validation/gen01). Repository-relative, case-insensitive;
    # MANIFEST.in carries the same lines for the sdist and tests/check_public_export_hygiene.py asserts both copies agree.
    policy = json.loads((ROOT / 'tools' / 'public_export.json').read_text(encoding='utf-8'))
    excluded_paths = {str(item).replace('\\', '/').strip('/').casefold() for item in policy['excluded_paths']}
    if not excluded_paths or '' in excluded_paths:
        raise ValueError('tools/public_export.json must name the excluded repository paths')

    def public_source_ignore(directory, names):
        ignored = set()
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(ROOT).as_posix().casefold()
            lower = name.casefold()
            if (lower in private_directories or lower.endswith('.egg-info') or lower in private_names
                    or relative in excluded_paths):
                ignored.add(name)
                continue
            example = lower == '.env.example' or (lower.startswith('.env.') and lower.endswith('.example'))
            if not example and any(fnmatchcase(lower, pattern) for pattern in private_patterns):
                ignored.add(name)
                continue
            ordinary(path)
        return ignored

    source = destination / 'SOURCE'
    source.mkdir()
    for name in ['LICENSE', 'NOTICE', 'README.md', 'STATUS.md', 'CONTRIBUTING.md', 'SECURITY.md', 'CODE_OF_CONDUCT.md',
                 'pyproject.toml', 'MANIFEST.in', 'install.py']:
        ordinary(ROOT / name)
        shutil.copy2(ROOT / name, source / name)
    # The tree-kind marker tools/build_public_branch.py writes at the root of a public tree (tools/public_export.json tree_kinds.marker):
    # absent in the private repository, copied when present so a SOURCE/ copy made from a public tree selects the public kind in
    # tests/check_public_export_hygiene.py by itself (MANIFEST.in includes it for the sdist the same way).
    marker = ROOT / str((policy.get('tree_kinds') or {}).get('marker') or '.titanos-public-tree')
    if marker.is_file():
        ordinary(marker)
        shutil.copy2(marker, source / marker.name)
    for name in ['src', 'tools', 'tests', 'docs']:
        ordinary(ROOT / name)
        shutil.copytree(ROOT / name, source / name, ignore=public_source_ignore)


# Every module whose @mcp.tool declarations mcp_server.create_mcp registers (its own tools, then the register_mcp calls of
# runtime_api, assignment_api and broker/mcp_tools at the end of create_mcp). This tuple is the one source of truth for the
# frozen declaration gate: offline_smoke and smoke compare the frozen server's list_tools against the union of these files,
# and tests/check_offline_binary.py asserts that union against the real create_mcp registration, so a module added to
# create_mcp without a line here fails the check before it fails a build (the 2026-10-05 candidates were refused with
# "Offline stdio MCP declarations differ from source" because broker/mcp_tools.py was missing from the lists).
MCP_TOOL_SOURCES = ('src/titanos/mcp_server.py', 'src/titanos/runtime_api.py', 'src/titanos/assignment_api.py',
                    'src/titanos/broker/mcp_tools.py')


def mcp_tool_source_paths(root=ROOT):
    """The MCP_TOOL_SOURCES files under `root`; a missing file is refused, never silently dropped from the expected set."""
    paths = [Path(root) / name for name in MCP_TOOL_SOURCES]
    missing = [name for name, path in zip(MCP_TOOL_SOURCES, paths) if not path.is_file()]
    if missing:
        raise ValueError('MCP tool declaration sources are missing: ' + ', '.join(missing))
    return paths


def expected_mcp_tools(root=ROOT):
    """The tool names a frozen build must declare: declared_mcp_tools over every MCP_TOOL_SOURCES file."""
    return declared_mcp_tools(mcp_tool_source_paths(root))


def declared_mcp_tools(source):
    """Bind frozen discovery to the actual tool declarations, not a stale count."""
    names = set()
    sources = source if isinstance(source, (list, tuple)) else [source]
    for path in sources:
        for node in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8'))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if not isinstance(target, ast.Attribute) or target.attr != 'tool':
                    continue
                named = next((keyword.value for keyword in decorator.keywords if keyword.arg == 'name'), None) if isinstance(decorator, ast.Call) else None
                name = ast.literal_eval(named) if named is not None else node.name
                if not isinstance(name, str) or not name or name in names:
                    raise ValueError('Invalid or duplicate MCP tool declaration')
                names.add(name)
    if not names:
        raise ValueError('No MCP tool declarations found')
    return names


def offline_smoke(executable, directory):
    """Exercise local CLI and stdio declarations without serving HTTP or tools.

    Asyncio/stdio may use internal OS IPC (including a socketpair); this is not
    a claim that the operating system creates no sockets at all.
    """
    directory=Path(directory);results=[];primary=None;temporary=None
    registry={'status':'NOT_PERFORMED','reason':'offline_validation','model_calls':0,
              'request_count':0,'cli_add_count':0,'cleanup_errors':[],
              'temporary_directory_closed':False}
    try:
        temporary=tempfile.TemporaryDirectory(prefix='titanos-offline-check-');temp=temporary.name
        env={k:v for k,v in os.environ.items() if k not in {'PYTHONHOME','PYTHONPATH','PYTHONSTARTUP','PYTHONINSPECT'}
             and not k.startswith('_PYI') and not k.endswith(('_API_KEY','_TOKEN','_PASSWORD','_SECRET'))}
        env.update(TITANOS_HOME=temp,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                   CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
        config=str(Path(temp)/'models.json')
        for arguments in [['--version'],['settings','--json'],['init'],['list']]:
            result=subprocess.run([str(executable),'--config',config,*arguments],cwd=temp,env=env,
                                  capture_output=True,text=True,timeout=180)
            results.append({'arguments':arguments,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            if result.returncode:raise RuntimeError('Offline CLI check failed: '+' '.join(arguments)+'\n'+result.stderr[-4000:])
            if arguments==['settings','--json'] and not isinstance(json.loads(result.stdout).get('settings'),dict):
                raise RuntimeError('Offline settings output is not the expected metadata object')
        if json.loads(Path(config).read_text(encoding='utf-8')).get('models')!={}:raise RuntimeError('Offline registry is not empty')
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters
        async def check_stdio():
            expected=expected_mcp_tools()
            target=StdioServerParameters(command=str(executable),args=['--config',config,'mcp','--transport','stdio',
                '--url','http://127.0.0.1:1'],env=env,cwd=temp)  # Unused sentinel; no tools/resources are called.
            async with Client(target,read_timeout_seconds=30) as client:
                names={tool.name for tool in (await client.list_tools()).tools}
                if names!=expected:raise RuntimeError('Offline stdio MCP declarations differ from source')
                return {'transport':'stdio','protocol':client.protocol_version,'tools':len(names),
                        'tool_calls':0,'resource_reads':0,'serving_listener_started':False,'passed':True}
        results.append({'mcp':asyncio.run(check_stdio())})
    except BaseException as exc:primary=exc
    finally:
        if temporary is not None:
            try:temporary.cleanup();registry['temporary_directory_closed']=True
            except BaseException as exc:
                registry['cleanup_errors'].append({'type':type(exc).__name__,'message':str(exc)})
                if primary is None:primary=exc
        binary={'validation_mode':'offline','passed':False,'offline_passed':primary is None,'checks':results,
                'http_validation':'not_performed','registry_validation':'not_performed','model_validation':'not_performed',
                'ui_validation':'not_performed','scope':'Local CLI and stdio initialize/list_tools only; no HTTP serving or tool/resource calls',
                'IPC_note':'Stdio event loops may use internal OS socketpairs; no application serving route is started.'}
        if primary is not None:binary['error']={'type':type(primary).__name__,'message':str(primary)}
        (directory/'registry-smoke.json').write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
        (directory/'binary-smoke.json').write_text(json.dumps(binary,indent=2)+'\n',encoding='utf-8')
    if primary is not None:raise primary


def smoke(executable, directory, validation='full'):
    if validation not in {'full','offline'}:raise ValueError('Unknown validation mode')
    if validation=='offline':return offline_smoke(executable,directory)
    directory=Path(directory)
    registry={'status':'IN_PROGRESS','model_calls':0,'request_count':0,'cli_add_count':0,
              'limits':{'seconds':120,'requests':19 if os.name=='nt' else 14,'cli_add_commands':1},
              'request_count_scope':'Registry HTTP requests and explicit MCP tool calls; startup/discovery/handshake traffic excluded',
              'scope':'Empty-engine registry control/recovery only; no backend loading, inference or forwarding qualification',
              'checks':[],'snapshots':{},'cleanup_errors':[]}
    results=[];temporary=None;process=None;log=None;primary=None
    def save_registry():
        (directory/'registry-smoke.json').write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    def cleanup_error(stage,exc):
        nonlocal primary
        registry['cleanup_errors'].append({'stage':stage,'type':type(exc).__name__,'message':str(exc)})
        if primary is None:primary=exc
        else:primary.add_note('Additional smoke '+stage+' failure: '+repr(exc))
    try:
        save_registry();temporary=tempfile.TemporaryDirectory(prefix='titanos-frozen-check-');temp=temporary.name
        env = {k: v for k, v in os.environ.items() if k not in {'PYTHONHOME', 'PYTHONPATH', 'TITANOS_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'}}
        env['TITANOS_HOME'] = temp
        for arguments in [['--version'], ['doctor', '--backend', 'engine', '--check'], ['init'], ['list']]:
            result = subprocess.run([str(executable), *arguments], cwd=temp, env=env, capture_output=True, text=True, timeout=180)
            results.append({'arguments': arguments, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
            if result.returncode:
                raise RuntimeError('Frozen smoke failed: ' + ' '.join(arguments) + '\n' + result.stderr[-4000:])
        import httpx
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters
        import psutil
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        # The reservation closes before launch; it is not an atomic handoff. Readiness therefore requires the
        # retained child to be alive, to be the launched process, and to own this exact loopback listener.
        url = f'http://127.0.0.1:{port}'
        log_path=Path(temp)/'server.log';log=log_path.open('w',encoding='utf-8')
        process=subprocess.Popen([str(executable),'serve','--host','127.0.0.1','--port',str(port)],cwd=temp,env=env,stdout=log,stderr=log)
        registry['server_pid']=process.pid;save_registry()
        try:child=psutil.Process(process.pid);identity={'pid':process.pid,'create_time':child.create_time()}
        except psutil.Error as exc:raise RuntimeError('Retained frozen server child identity is unavailable: '+repr(exc)) from exc
        registry['server_identity']=identity;registry['listener_ownership']=[];save_registry()
        def log_tail():
            with log_path.open('rb') as reader:return reader.read(131072).decode('utf-8','replace')[-5000:]
        def startup_failure(message):return RuntimeError(message+log_tail())
        # Only these local addresses can receive this smoke's 127.0.0.1 connection; a listener on another
        # address that merely shares the port number (for example [::1]) is recorded but cannot answer it.
        reachable={'127.0.0.1','0.0.0.0','::','::ffff:127.0.0.1'}
        def owned_listener(stage):
            """Retained child liveness/identity plus exact 127.0.0.1:port LISTEN ownership; None while still unbound."""
            record={'stage':stage,'pid':process.pid,'port':port}
            def retain(**extra):registry['listener_ownership'].append({**record,**extra});save_registry()
            exit_code=process.poll()
            if exit_code is not None:
                retain(child_exit_code=exit_code);raise startup_failure('Frozen server child exited before '+stage+' (exit code '+str(exit_code)+'): ')
            try:
                # psutil caches create_time per handle, so a fresh handle re-reads it; is_running() applies psutil's own PID-reuse test.
                if not child.is_running() or psutil.Process(process.pid).create_time()!=identity['create_time']:
                    retain(identity_matched=False);raise startup_failure('Retained frozen server PID no longer identifies the launched process at '+stage+': ')
                for attempt in range(5):
                    try:table,rows='system',[(row.pid,row.laddr,row.status) for row in psutil.net_connections(kind='tcp')]
                    except psutil.AccessDenied:  # Non-root macOS cannot read the system table; the child's own table still proves its listener.
                        table,rows='child',[(process.pid,row.laddr,row.status) for row in child.net_connections(kind='tcp')]
                    listening=[{'pid':pid,'ip':addr.ip,'port':addr.port} for pid,addr,status in rows if status==psutil.CONN_LISTEN and addr and addr.port==port]
                    # Linux maps socket inodes to PIDs before reading the table, so a just-bound listener can briefly show pid None.
                    if attempt==4 or not any(row['pid'] is None for row in listening):break
                    time.sleep(.05)
            except psutil.Error as exc:
                retain(error=repr(exc));raise RuntimeError('Retained frozen server identity or listener table is unavailable at '+stage+': '+repr(exc)) from exc
            relevant=[row for row in listening if row['ip'] in reachable or row['pid']==process.pid]
            if not relevant:return None
            owned=all(row['pid']==process.pid and row['ip']=='127.0.0.1' for row in relevant)
            retain(table=table,table_reads=attempt+1,listeners=relevant,unrelated_listeners=[row for row in listening if row not in relevant],owned=owned)
            if not owned:raise RuntimeError('Listener on 127.0.0.1:'+str(port)+' is not owned by the retained frozen server child at '+stage+': '+repr(relevant))
            return relevant
        deadline=time.monotonic()+60
        while owned_listener('startup') is None:
            if time.monotonic()>deadline:raise startup_failure('Frozen server did not own its loopback listener before the deadline: ')
            time.sleep(.1)
        while True:
            # Ownership is proved immediately before and after every health response, so a dying or
            # replaced child cannot borrow another listener's HTTP 200.
            if owned_listener('before-health') is None:
                if time.monotonic()>deadline:raise startup_failure('Frozen server released its listener before answering /healthz: ')
                time.sleep(.1);continue
            try:healthy=httpx.get(url+'/healthz',trust_env=False,timeout=1).status_code==200
            except httpx.HTTPError:healthy=False
            if owned_listener('after-health') is None:raise RuntimeError('Frozen server listener vanished around the /healthz response')
            if healthy:break
            if time.monotonic()>deadline:raise startup_failure('Owned frozen server listener did not answer /healthz 200 before the deadline: ')
            time.sleep(.1)
        registry['readiness']={'owned_listener_before_health':True,'health_status':200,'owned_listener_after_health':True};save_registry()
        async def check_protocols():
            protocol_results=[]
            expected_tools=expected_mcp_tools()
            targets=[('streamable-http',url+'/mcp/'),('stdio',StdioServerParameters(command=str(executable),args=['mcp','--transport','stdio','--url',url],env=env,cwd=temp))]
            for transport,target in targets:
                async with Client(target,read_timeout_seconds=30) as client:
                    tools={tool.name for tool in (await client.list_tools()).tools}
                    if tools!=expected_tools:
                        raise RuntimeError('Frozen MCP tool inventory differs from its source declarations: '+repr(sorted(tools ^ expected_tools)))
                    result=await client.call_tool('list_models',{})
                    if result.is_error or result.structured_content!={'object':'list','data':[]}:
                        raise RuntimeError('Frozen MCP model discovery failed')
                    protocol_results.append({'transport':transport,'protocol':client.protocol_version,'tools':len(tools),'passed':True})
            return protocol_results
        results.append({'mcp':asyncio.run(check_protocols()),'server_started':True})
        registry['empty_mcp_discovery_first']=True
        if owned_listener('before-controls') is None:raise RuntimeError('Frozen server listener vanished before registry controls')
        registry_deadline=time.monotonic()+registry['limits']['seconds'];registry['deadline_monotonic']=registry_deadline
        path=Path(temp)/'models.json'
        def remaining():
            value=registry_deadline-time.monotonic()
            if value<=0:raise TimeoutError('Registry smoke fixed budget expired')
            if process.poll() is not None:raise RuntimeError('Same registry smoke server exited')
            return value
        def reserve(name,method,endpoint,payload=None):
            remaining();registry['request_count']+=1
            if registry['request_count']>registry['limits']['requests']:raise RuntimeError('Registry smoke request budget exceeded')
            entry={'name':name,'method':method,'endpoint':endpoint,'payload':payload};registry['checks'].append(entry);save_registry();return entry
        def request(name,method,endpoint,payload=None,status=200):
            entry=reserve(name,method,endpoint,payload)
            response=httpx.request(method,url+endpoint,json=payload,trust_env=False,timeout=min(10,remaining()))
            entry.update(status=response.status_code,body=response.text[:65536],body_truncated=len(response.text)>65536);save_registry()
            if response.status_code!=status or entry['body_truncated']:raise RuntimeError('Registry smoke unexpected response: '+name)
            remaining();return response.json()
        def snapshot(name):
            raw=path.read_bytes();registry['snapshots'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'utf8':raw.decode('utf-8')};save_registry();return raw,json.loads(raw)
        def catalog(name,expected):
            value=request(name,'GET','/v1/models')
            if {row['id']:row['task'] for row in value['data']}!=expected:raise RuntimeError('Registry live catalog differs: '+name)
        first=Path(temp)/'checkpoint-first';second=Path(temp)/'checkpoint-second';first.mkdir();second.mkdir()
        markers={first/'keep.txt':b'first placeholder; no weights',second/'keep.txt':b'second placeholder; no weights'}
        for marker,raw in markers.items():marker.write_bytes(raw)
        model='titanos-smoke-http';cli_model='titanos-smoke-cli'
        spec={'path':str(first),'task':'chat','backend':'transformers','device':'cpu','defaults':{'max_tokens':7}}
        request('register','POST','/api/models/register',{'name':model,'spec':spec})
        catalog('registered-catalog',{model:'chat'})
        _,disk=snapshot('registered')
        if Path(disk['models'][model]['path']).resolve()!=first.resolve() or disk['models'][model]['defaults']!={'max_tokens':7}:raise RuntimeError('Registered path/defaults differ')
        replacement={**spec,'path':str(second),'task':'vision','defaults':{'max_tokens':17,'temperature':.25}}
        request('replace','POST','/api/models/register',{'name':model,'spec':replacement,'replace':True})
        catalog('replaced-catalog',{model:'vision'});_,disk=snapshot('replaced')
        if Path(disk['models'][model]['path']).resolve()!=second.resolve() or disk['models'][model]['defaults']!=replacement['defaults']:raise RuntimeError('Replacement path/defaults differ')
        defaults=Path(temp)/'cli-defaults.json';defaults.write_text(json.dumps({'max_tokens':11}),encoding='utf-8')
        argv=[str(executable),'--config',str(path),'add',cli_model,str(first),'--task','chat','--backend','transformers','--device','cpu','--dtype','float32','--defaults',str(defaults)]
        registry['cli_add_count']+=1;save_registry()
        added=subprocess.run(argv,cwd=temp,env=env,capture_output=True,text=True,timeout=min(30,remaining()))
        registry['cli_add']={'argv':argv,'exit_code':added.returncode,'stdout':added.stdout,'stderr':added.stderr};save_registry()
        if added.returncode:raise RuntimeError('Compiled CLI registration failed')
        _,disk=snapshot('cli-added')
        if Path(disk['models'][cli_model]['path']).resolve()!=first.resolve() or disk['models'][cli_model]['defaults']!={'max_tokens':11}:raise RuntimeError('CLI path/defaults differ')
        catalog('before-explicit-reload',{model:'vision'})
        entry=reserve('mcp-explicit-reload','MCP','reload_registry',{})
        async def reload_over_mcp():
            async with Client(url+'/mcp/',read_timeout_seconds=min(30,remaining())) as client:
                return await client.call_tool('reload_registry',{})
        reloaded=asyncio.run(asyncio.wait_for(reload_over_mcp(),timeout=remaining()))
        entry.update(is_error=reloaded.is_error,result=reloaded.structured_content);save_registry()
        if reloaded.is_error or reloaded.structured_content.get('status')!='reloaded':raise RuntimeError('MCP registry reload failed')
        expected={model:'vision',cli_model:'chat'};catalog('reloaded-catalog',expected)
        from filelock import FileLock
        before,_=snapshot('before-lock-contention')
        with FileLock(str(path)+'.lock',timeout=0):
            refused=request('writer-contention','POST','/api/models/register',{'name':'titanos-smoke-blocked','spec':spec},409)
            if refused.get('error',{}).get('code')!='registry_busy' or path.read_bytes()!=before:raise RuntimeError('Registry contention changed disk or wrong error')
        catalog('after-contention',expected)
        if os.name=='nt':
            import ctypes
            from ctypes import wintypes as w
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.CreateFileW.argtypes=[w.LPCWSTR,w.DWORD,w.DWORD,w.LPVOID,w.DWORD,w.DWORD,w.HANDLE];kernel.CreateFileW.restype=w.HANDLE
            kernel.GetHandleInformation.argtypes=[w.HANDLE,ctypes.POINTER(w.DWORD)];kernel.GetHandleInformation.restype=w.BOOL
            kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
            handle=None;fault_primary=None
            registry['windows_persistence']={'desired_access':0x80000000,'share_mode':1,'security_attributes_null':True,'handle_closed':False}
            try:
                handle=kernel.CreateFileW(str(path),0x80000000,1,None,3,0x80,None)
                if handle==w.HANDLE(-1).value:handle=None;raise ctypes.WinError(ctypes.get_last_error())
                registry['windows_persistence']['handle']=int(handle)
                flags=w.DWORD()
                if not kernel.GetHandleInformation(handle,ctypes.byref(flags)):raise ctypes.WinError(ctypes.get_last_error())
                registry['windows_persistence']['handle_flags']=flags.value
                if flags.value&1:raise RuntimeError('Registry sharing handle must not be inheritable')
                before,_=snapshot('before-persistence-failure');offset=log_path.stat().st_size
                failed=request('persistence-failure','POST','/api/models/register',{'name':'titanos-smoke-failed','spec':spec},500)
                with log_path.open('rb') as reader:reader.seek(offset);trace_raw=reader.read(65537)
                trace=trace_raw[:65536].decode('utf-8','replace')
                registry['windows_persistence'].update(fresh_log_offset=offset,fresh_traceback=trace,fresh_trace_truncated=len(trace_raw)>65536,server_pid=process.pid);save_registry()
                # Frozen tracebacks need not print source text. Join frame line
                # numbers to the exact source bytes already bound by this build.
                required=[]
                for filename,function,statement in [('server.py','reload_registry','ifchange:write_config(path,candidate)'),('config.py','write_config','os.replace(tmp,path)')]:
                    source_file=ROOT/'src/titanos'/filename;lines=source_file.read_text(encoding='utf-8').splitlines()
                    matches=[number for number,line in enumerate(lines,1) if ''.join(line.split())==statement]
                    if len(matches)!=1:raise RuntimeError('Registry smoke source statement binding differs')
                    number=matches[0];pattern=r'File "[^"\r\n]*'+re.escape(filename)+r'", line '+str(number)+r', in '+re.escape(function)+r'(?:\r?\n|$)'
                    required.append({'file':filename,'function':function,'line':number,'statement':statement,'source_sha256':digest(source_file),'matched':bool(re.search(pattern,trace))})
                registry['windows_persistence']['trace_source_join']=required;save_registry()
                if len(trace_raw)>65536 or 'register_model' not in trace or not all(row['matched'] for row in required):raise RuntimeError('Fresh same-server persistence traceback missing or wrong frame')
                if path.read_bytes()!=before:raise RuntimeError('Failed native persistence changed registry bytes')
                unavailable=request('unavailable-after-persistence','GET','/v1/models',status=503)
                if unavailable.get('error',{}).get('code')!='registry_unavailable':raise RuntimeError('Failed retirement did not close registry admission')
            except BaseException as exc:fault_primary=exc;raise
            finally:
                if handle is not None:
                    try:
                        if not kernel.CloseHandle(handle):raise ctypes.WinError(ctypes.get_last_error())
                        registry['windows_persistence']['handle_closed']=True;handle=None
                    except BaseException as exc:
                        registry['windows_persistence']['close_error']=repr(exc)
                        if fault_primary is None:fault_primary=exc;raise
                        fault_primary.add_note('Registry sharing handle close failure: '+repr(exc))
                    finally:
                        try:save_registry()
                        except BaseException as exc:
                            if fault_primary is None:raise
                            fault_primary.add_note('Registry handle receipt failure: '+repr(exc))
            request('same-server-recovery','POST','/api/models/reload')
            catalog('recovered-catalog',expected)
            if request('recovered-empty-loaded','GET','/api/models/loaded')['data']!=[]:raise RuntimeError('Registry-only recovery loaded a model')
        else:registry['windows_persistence']={'status':'NOT_APPLICABLE_NON_WINDOWS'}
        missing=request('known-route-unknown-model','POST','/api/models/capabilities',{'model':'titanos-smoke-missing'},404)
        if missing.get('error',{}).get('code')!='model_not_found':raise RuntimeError('Unknown-model error contract differs')
        request('unregister-http','DELETE','/api/models/registrations/'+model)
        request('unregister-cli','DELETE','/api/models/registrations/'+cli_model)
        catalog('empty-catalog',{})
        if request('empty-loaded','GET','/api/models/loaded')['data']!=[]:raise RuntimeError('Registry smoke loaded a model')
        _,disk=snapshot('final-empty')
        if disk['models'] or any(marker.read_bytes()!=raw for marker,raw in markers.items()):raise RuntimeError('Unregister changed placeholder files or left registrations')
        remaining()
        if registry['request_count']!=registry['limits']['requests'] or registry['cli_add_count']!=1:raise RuntimeError('Registry smoke check count differs')
        registry.update(sequence_passed=True,placeholder_files_preserved=True,server_restarted=False);save_registry()
        results.append({'registry':'registry-smoke.json','request_count':registry['request_count'],'model_calls':0})
    except BaseException as exc:
        primary=exc;registry.update(status='FAILED',error={'type':type(exc).__name__,'message':str(exc),'notes':list(getattr(exc,'__notes__',()))})
    finally:
        if process is not None:
            try:process.terminate()
            except BaseException as exc:cleanup_error('server-terminate',exc)
            try:registry['server_exit_code']=process.wait(timeout=10)
            except BaseException as exc:cleanup_error('server-wait',exc)
        if log is not None:
            try:
                log.flush()
                with log_path.open('rb') as reader:raw_log=reader.read(131073)
                registry.update(server_log=raw_log[:131072].decode('utf-8','replace'),server_log_truncated=len(raw_log)>131072)
            except BaseException as exc:cleanup_error('server-log-readback',exc)
            try:log.close()
            except BaseException as exc:cleanup_error('server-log-close',exc)
        registry['status']='PASSED' if primary is None and registry.get('sequence_passed') else 'FAILED'
        try:save_registry()  # Persistent evidence exists before temporary files are removed.
        except BaseException as exc:cleanup_error('registry-report',exc)
        if temporary is not None:
            try:temporary.cleanup();registry['temporary_directory_closed']=True
            except BaseException as exc:cleanup_error('temporary-directory',exc)
        registry['status']='PASSED' if primary is None and registry.get('sequence_passed') else 'FAILED'
        registry['error_notes']=list(getattr(primary,'__notes__',())) if primary else []
        try:
            save_registry()
            (directory/'binary-smoke.json').write_text(json.dumps({'passed':primary is None,'checks':results,
                'readiness':'Retained child liveness/identity and exact loopback listener ownership proved before, around and after /healthz and before registry controls',
                'scope':'Native startup, empty MCP discovery and empty-engine registry controls; not model inference'},indent=2)+'\n',encoding='utf-8')
        except BaseException as exc:
            cleanup_error('final-report',exc);registry['status']='FAILED'
            try:save_registry()
            except BaseException as extra:primary.add_note('Registry failure receipt could not be saved: '+repr(extra))
    if primary is not None:raise primary


def publish_smoke_summary(bundle, private_directory):
    """Retain exact operational receipts privately; ship only a path-free summary."""
    bundle, private_directory = Path(bundle).resolve(), Path(private_directory).resolve()
    if private_directory == bundle or private_directory.is_relative_to(bundle):
        raise ValueError('Private smoke receipts must remain outside the distribution')
    private_directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in ('registry-smoke.json', 'binary-smoke.json'):
        source = bundle / name
        raw = source.read_bytes()
        destination = private_directory / name
        if destination.exists():
            if destination.read_bytes() != raw:
                raise ValueError('Private smoke receipt already exists with different bytes')
        else:
            with destination.open('xb') as stream:
                stream.write(raw)
        if digest(destination) != hashlib.sha256(raw).hexdigest():
            raise ValueError('Private smoke receipt copy differs')
        records[name] = json.loads(raw)
    registry, binary = records['registry-smoke.json'], records['binary-smoke.json']
    mode=binary.get('validation_mode','full')
    if mode=='offline':
        if (registry.get('status')!='NOT_PERFORMED' or registry.get('reason')!='offline_validation'
                or binary.get('passed') is not False or binary.get('offline_passed') is not True
                or registry.get('request_count')!=0 or registry.get('model_calls')!=0
                or registry.get('cleanup_errors') or not registry.get('temporary_directory_closed')
                or any(binary.get(key)!='not_performed' for key in ('http_validation','registry_validation','model_validation','ui_validation'))):
            raise ValueError('Cannot publish offline validation from incomplete or mislabeled evidence')
    elif mode!='full' or registry.get('status') != 'PASSED' or binary.get('passed') is not True or registry.get('model_calls') != 0:
        raise ValueError('Cannot publish a successful smoke summary from failed evidence')
    summary = {'passed': mode=='full', 'validation_mode':mode,'model_calls': 0,
               'scope': 'Native startup, empty MCP discovery and empty-engine registry controls; not model inference',
               'registry_requests': registry.get('request_count'),
               'CLI_registration_commands': registry.get('cli_add_count'),
               'cleanup_error_count': len(registry.get('cleanup_errors', [])),
               'temporary_directory_closed': registry.get('temporary_directory_closed'),
               'raw_receipts': 'Retained by the builder outside this distribution'}
    if mode=='offline':
        summary.update(offline_passed=True,scope=binary['scope'],IPC_note=binary['IPC_note'],
            http_validation='not_performed',registry_validation='not_performed',
            model_validation='not_performed',ui_validation='not_performed',full_smoke_passed=False)
    (bundle / 'BUILD_SMOKE.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for name in records:
        (bundle / name).unlink()
    return summary


def validate_smoke_summary(summary, validation='full'):
    """Consumers must deliberately select offline evidence; never infer a pass."""
    if validation not in {'full','offline'}:raise ValueError('Unknown validation mode')
    if summary.get('validation_mode','full')!=validation or summary.get('model_calls')!=0:
        raise ValueError('Build validation mode/scope differs from the requested verification')
    if validation=='full':
        if summary.get('passed') is not True:raise ValueError('Full build smoke did not pass')
    elif (summary.get('passed') is not False or summary.get('offline_passed') is not True
            or summary.get('full_smoke_passed') is not False
            or any(summary.get(k)!='not_performed' for k in ('http_validation','registry_validation','model_validation','ui_validation'))
            or summary.get('cleanup_error_count')!=0 or summary.get('temporary_directory_closed') is not True):
        raise ValueError('Offline build metadata is incomplete or overstates validation')


# ------------------------------------------------------------------------------------------------ distribution hygiene
# Before sealing, every file of the delivered directory is scanned for this lab's locators: UTF-8 text and the byte strings
# inside binaries alike (a configure prefix in a shared library, a pip direct_url.json, a filename in a .pyc, the members
# of a stored zip), because the retained SOURCE/-only scan of earlier candidates let a music runtime's pip provenance and a
# Linux FFmpeg bundle's build prefix through. tools/public_export.json stays the one statement of the pattern classes
# (tests/check_public_export_hygiene.py applies the same classes to the source copies and the tracked tree); the names of
# the building machine and account are read from the environment at scan time and never written into the repository.
# Byte strings are searched in their 8-bit spelling and in UTF-16LE (the Windows wide-string spelling a DLL carries; the
# 2026-10-06 verifier found 1,647 UTF-16LE copies of an upstream conda builder path in one collected DLL, outside the scan of
# that day). UTF-16BE/UTF-32 strings and compressed payloads (the PYZ archive inside the executable, compressed zip members
# are inflated; tarballs are not) are outside the scan's reach, and this is a hygiene guard for lab-private references only,
# not a secret scanner.
HYGIENE_CHUNK = 64 * 2**20
HYGIENE_OVERLAP = 4096      # bytes two neighbouring chunks share; longer than any locator match
HYGIENE_CONTEXT = 256       # left context a match needs for its look-behind; a match closer to a chunk start waits for the next chunk
HYGIENE_TEXT_LIMIT = 64 * 2**20
# The locator classes searched in every file: each carries the lower-case literals a match must contain, so a multi-gigabyte
# library is covered by a substring search and the pattern itself runs only around those anchors.
LOCATOR_ANCHORS = {'windows_user_profile': (b'users',), 'posix_user_profile': (b'/users/',), 'lab_drive_titanos': (b'titanos',),
                   'lab_drive_codex': (b'codex',), 'lab_codex_worktree': (b'codex',), 'linux_home': (b'/home/',),
                   'macos_user_home': (b'/users/',), 'lab_linux_host': (b'-server-',),
                   'windows_generated_machine_name': (b'desktop-', b'laptop-', b'win-')}
# Numeric or fingerprint shapes match random bytes; they run over decodable text only.
TEXT_ONLY_CLASSES = ('rfc1918_ipv4', 'drive_letter_pair', 'ssh_hostkey_fingerprint')
# Builder-local class: a file: URL of a drive or a home directory (pip provenance of a local wheel or source tree).
LOCAL_FILE_URL_CLASS = 'local_file_url'
LOCAL_FILE_URL = re.compile(r'(?i)file:///(?:[A-Za-z]:[\\/]|home/|users/)')
# Shapes that name this lab (a drive-rooted Titanos or Codex tree, an agent work tree, the lab's Linux hostname form): refused
# wherever they appear, third-party payload included.
LAB_CLASSES = frozenset({'lab_drive_titanos', 'lab_drive_codex', 'lab_codex_worktree', 'lab_linux_host'})
# Classes whose match ends in an account segment; the account is the private part. posix_user_profile is the MSYS, Cygwin or WSL
# spelling of a Windows user profile (a drive letter as a directory under /, /cygdrive/ or /mnt/, then Users/<account>).
ACCOUNT_CLASSES = {'windows_user_profile': re.compile(r'(?i)users[\\/]+(.*)$'), 'posix_user_profile': re.compile(r'(?i)/Users[/](.*)$'),
                   'linux_home': re.compile(r'/home[/](.*)$'),
                   'macos_user_home': re.compile(r'/Users[/](.*)$')}   # spelled with [/] so this source never matches the classes
MACHINE_NAME_CLASS = 'exporting_machine_name'
DENIED_NAME_CLASS = 'denied_name'
ARCHIVE_SUFFIXES = {'.zip', '.whl', '.egg', '.jar'}
UTF16_ENCODING = 'utf-16le'
UTF16_BEFORE, UTF16_AFTER = 512, 2048   # bytes of UTF-16LE window around an anchor: the 8-bit window's 256/1024 characters


def hygiene_policy(root=ROOT):
    """tools/public_export.json: the detection classes and the fixture allowlist (with the policy file's own escaped forms)."""
    policy = json.loads((Path(root) / 'tools' / 'public_export.json').read_text(encoding='utf-8'))
    missing = sorted((set(LOCATOR_ANCHORS) | set(TEXT_ONLY_CLASSES)) - set(policy['detection_patterns']))
    if missing:
        raise ValueError('tools/public_export.json lacks detection classes the distribution scan relies on: ' + ', '.join(missing))
    allowlist = {entry['path']: set(entry['literals']) for entry in policy.get('fixture_allowlist', [])}
    allowlist['tools/public_export.json'] = {form for literal in set().union(set(), *allowlist.values())
                                             for form in (literal, json.dumps(literal)[1:-1])}
    return {'patterns': {name: re.compile(pattern) for name, pattern in policy['detection_patterns'].items()}, 'allowlist': allowlist}


def hygiene_names(environ=None):
    """The names the scan refuses besides the pattern classes: the building machine's names and the private host map's names
    (whole word, in every file; the map is tools/lab_hosts.private.json when present), the building account (only as the
    account segment of a user-profile or home path: a bare word such as root or user would match everything) and the
    TITANOS_HYGIENE_DENY entries (comma or semicolon separated; both ways, for a lab account or host of another machine whose
    inputs this build carries). Read from the environment at scan time; never written into the repository or the distribution."""
    environ = os.environ if environ is None else environ
    machines = {value.strip() for value in (environ.get('COMPUTERNAME'), environ.get('HOSTNAME'), platform.node())
                if value and len(value.strip()) >= 3 and value.strip().casefold().split('.')[0] != 'localhost'}
    accounts = {value.strip() for value in (environ.get('USERNAME'), environ.get('USER'), environ.get('LOGNAME')) if value and value.strip()}
    try:
        accounts.add(Path.home().name)
    except (RuntimeError, KeyError):
        pass
    extra = {value.strip() for value in re.split(r'[,;]', environ.get('TITANOS_HYGIENE_DENY', '')) if value.strip()}
    try:
        import lab_paths
        private = set(lab_paths.private_names())
    except ImportError:
        private = set()
    return {'machines': sorted(machines | extra), 'accounts': sorted(accounts | extra | private), 'private': sorted(private)}


def hygiene_zone(name):
    """'owned' for what Titanos or this lab wrote (the builder's records, SOURCE/, the frozen titanos package, the lab-built
    native bundles, runtime inventories, every pip direct_url.json); 'third_party' for collected dependency payload, retained
    upstream sources and notices, and a relocated runtime interpreter. Every class refuses in the owned zone; in the
    third-party zone the lab shapes and the denied names refuse and the generic shapes (another project's /home/<account>
    example, a contributor's generated machine name in an AUTHORS file) are reported, not refused."""
    parts = PurePosixPath(name).parts
    if parts[-1] == 'direct_url.json' and len(parts) >= 2 and parts[-2].endswith('.dist-info'):
        return 'owned'
    if parts[0] == '_internal':
        return 'owned' if len(parts) >= 2 and parts[1] in {'titanos', 'native'} else 'third_party'
    if parts[0] in {'THIRD_PARTY_LICENSES', 'THIRD_PARTY_SOURCES'}:
        return 'third_party'
    if parts[0] == 'runtimes':
        return 'third_party' if len(parts) >= 3 and parts[2] in {'python', 'SOURCE', 'licenses'} else 'owned'
    return 'owned'


def _hygiene_windows(low, anchors, before=256, after=1024):
    """Merged [start, end) byte spans around every occurrence of an anchor literal in the lower-cased bytes."""
    spans = []
    for anchor in anchors:
        index = low.find(anchor)
        while index >= 0:
            spans.append((max(0, index - before), min(len(low), index + len(anchor) + after)))
            index = low.find(anchor, index + 1)
    return _merge_spans(spans)


def _merge_spans(spans):
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _hygiene_windows_utf16(low, anchors):
    """Byte spans around every occurrence of a UTF-16LE-encoded anchor, each span keeping the code-unit alignment of its
    anchor (an even span length starting at the anchor's parity); spans of the two parities are merged separately so a
    decoded window never straddles an alignment change."""
    by_parity = {0: [], 1: []}
    for anchor in anchors:
        index = low.find(anchor)
        while index >= 0:
            start = index - UTF16_BEFORE
            if start < 0:
                start = index % 2
            end = min(len(low), index + len(anchor) + UTF16_AFTER)
            if (end - start) % 2:
                end -= 1
            by_parity[index % 2].append((start, end))
            index = low.find(anchor, index + 1)
    return _merge_spans(by_parity[0]) + _merge_spans(by_parity[1])


def _utf16(anchor):
    return anchor.decode('latin-1').encode(UTF16_ENCODING)


class _HygieneScanner:
    def __init__(self, policy, names):
        self.patterns = policy['patterns']
        self.allowlist = policy['allowlist']
        self.lab_patterns = [self.patterns[label] for label in ('lab_linux_host', 'windows_generated_machine_name')]
        self.accounts = {value.casefold() for value in names['accounts']}
        words = sorted(set(names['machines']) | set(names['private']), key=len, reverse=True)
        self.word_pattern = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(word) for word in words) + r')(?![A-Za-z0-9])',
                                       re.IGNORECASE | re.ASCII) if words else None
        self.word_anchors = tuple(sorted({word.casefold().encode('latin-1', 'replace') for word in words}))
        self.hits = []

    def _denied_account(self, label, matched):
        account = ACCOUNT_CLASSES[label].search(matched)
        if account is None:
            return False
        value = account.group(1)
        return value.casefold() in self.accounts or any(pattern.search(value) for pattern in self.lab_patterns)

    def _record(self, name, zone, label, offset, line, action, reason=None, encoding=None):
        hit = {'file': name, 'class': label, 'zone': zone, 'offset': offset, 'line': line, 'action': action}
        if reason:
            hit['reason'] = reason
        if encoding:
            hit['encoding'] = encoding
        self.hits.append(hit)

    def _classify(self, zone, label, matched):
        """('refuse' | 'report', reason) for a locator match by zone; None when the fixture allowlist tolerates it."""
        if zone == 'owned':
            return 'refuse', None
        if label in LAB_CLASSES:
            return 'refuse', None
        if label in ACCOUNT_CLASSES and self._denied_account(label, matched):
            return 'refuse', 'denied_account'
        return 'report', 'third_party_residue'

    def scan_chunk(self, data, name, zone, base, accept_from, accept_until, line_base=None):
        """Locator classes, the local file: URL class and the denied words over one chunk's byte strings, in their 8-bit
        spelling (latin-1, 1:1) and in UTF-16LE (decoded around every wide-string anchor at the anchor's alignment; a hit
        carries encoding utf-16le and no line). Matches starting outside [accept_from, accept_until) belong to a
        neighbouring chunk and are left to it."""
        low = data.lower()
        allowed = self.allowlist.get(name[len('SOURCE/'):], set()) if name.startswith('SOURCE/') else set()
        classes = [(label, self.patterns[label], anchors) for label, anchors in LOCATOR_ANCHORS.items()]
        classes.append((LOCAL_FILE_URL_CLASS, LOCAL_FILE_URL, (b'file:///',)))
        if self.word_pattern is not None:
            classes.append((DENIED_NAME_CLASS, self.word_pattern, self.word_anchors))
        for label, pattern, anchors in classes:
            windows = [(start, end, data[start:end].decode('latin-1'), None) for start, end in _hygiene_windows(low, anchors)]
            windows += [(start, end, data[start:end].decode(UTF16_ENCODING, 'replace'), UTF16_ENCODING)
                        for start, end in _hygiene_windows_utf16(low, tuple(_utf16(anchor) for anchor in anchors))]
            for start, end, text, encoding in windows:
                for match in pattern.finditer(text):
                    # the byte where the match begins: 1:1 in latin-1; in UTF-16LE the encoded length of the text before it,
                    # so a surrogate pair (four bytes, one character) earlier in the window does not shift the offset
                    offset = start + (match.start() if encoding is None else len(text[:match.start()].encode(UTF16_ENCODING, 'replace')))
                    if not accept_from <= offset < accept_until or match.group(0) in allowed:
                        continue
                    line = line_base + data.count(b'\n', 0, offset) + 1 if line_base is not None and encoding is None else None
                    if label == DENIED_NAME_CLASS:
                        self._record(name, zone, label, base + offset, line, 'refuse', encoding=encoding)
                        continue
                    action, reason = self._classify(zone, label, match.group(0))
                    self._record(name, zone, label, base + offset, line, action, reason, encoding)

    def scan_text(self, text, name, zone):
        """The text-only classes over a decodable file (every zone refuses them in the owned zone; reported elsewhere)."""
        allowed = self.allowlist.get(name[len('SOURCE/'):], set()) if name.startswith('SOURCE/') else set()
        for label in TEXT_ONLY_CLASSES:
            for match in self.patterns[label].finditer(text):
                if match.group(0) in allowed:
                    continue
                line = text.count('\n', 0, match.start()) + 1
                action = 'refuse' if zone == 'owned' else 'report'
                self._record(name, zone, label, None, line, action, None if zone == 'owned' else 'third_party_residue')

    def scan_bytes(self, data, name, zone):
        """One in-memory payload (a small file or an archive member)."""
        text = None
        if len(data) <= HYGIENE_TEXT_LIMIT:
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                text = None
        self.scan_chunk(data, name, zone, 0, 0, len(data), line_base=0 if text is not None else None)
        if text is not None:
            self.scan_text(text, name, zone)

    def scan_file(self, path, name, zone):
        """A file of any size. Large files are read in chunks that overlap by HYGIENE_OVERLAP bytes; a match is accepted by
        the first chunk that holds it with HYGIENE_CONTEXT bytes of left context, so a locator spanning a boundary is seen
        once and its look-behind is evaluated against the real preceding bytes."""
        size = path.stat().st_size
        if size <= HYGIENE_TEXT_LIMIT:
            self.scan_bytes(path.read_bytes(), name, zone)
            return size
        with path.open('rb') as stream:
            carry, base = b'', 0
            data = stream.read(HYGIENE_CHUNK)
            while data:
                following = stream.read(HYGIENE_CHUNK)
                chunk = carry + data
                accept_from = HYGIENE_CONTEXT if carry else 0
                accept_until = len(chunk) if not following else len(chunk) - (HYGIENE_OVERLAP - HYGIENE_CONTEXT)
                self.scan_chunk(chunk, name, zone, base - len(carry), accept_from, accept_until)
                carry = chunk[-HYGIENE_OVERLAP:]
                base += len(data)
                data = following
        return size


def distribution_hygiene(destination, names=None, policy=None):
    """Scan every file under `destination` (archive members included) and return the report: refused and reported hits with
    file, class, zone, byte offset and line, never the matched text; counts; the classes applied. Raises nothing on hits."""
    destination = Path(destination).resolve()
    names = names if names is not None else hygiene_names()
    scanner = _HygieneScanner(policy or hygiene_policy(), names)
    files = members = total = 0
    zones = {'owned': 0, 'third_party': 0}
    for path in sorted(destination.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(destination).as_posix()
        zone = hygiene_zone(name)
        zones[zone] += 1
        files += 1
        total += scanner.scan_file(path, name, zone)
        if path.suffix.lower() in ARCHIVE_SUFFIXES:
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        if not member.is_dir():
                            scanner.scan_bytes(archive.read(member), name + '!' + member.filename, zone)
                            members += 1
            except zipfile.BadZipFile:
                pass
    refused = [hit for hit in scanner.hits if hit['action'] == 'refuse']
    reported = {}
    for hit in scanner.hits:
        if hit['action'] == 'report':
            key = (hit['file'], hit['class'], hit.get('encoding'))
            reported[key] = reported.get(key, 0) + 1
    return {'scope': 'Lab-private locators in every delivered file (text and byte strings in their 8-bit and UTF-16LE spellings; stored or '
                     'deflated zip members); not a secret scanner, not a legal review; UTF-16BE/UTF-32 strings and compressed payloads other '
                     'than zip are not covered; the text-only numeric and fingerprint classes run over decodable UTF-8 text only',
            'files_scanned': files, 'archive_members_scanned': members, 'bytes_scanned': total, 'zones': zones,
            'classes': {'everywhere': sorted(LOCATOR_ANCHORS) + [LOCAL_FILE_URL_CLASS, DENIED_NAME_CLASS],
                        'text_only': list(TEXT_ONLY_CLASSES), 'refused_in_third_party_zone': sorted(LAB_CLASSES) + [DENIED_NAME_CLASS, 'denied_account'],
                        'encodings': ['8-bit', UTF16_ENCODING]},
            'names': {key: len(value) for key, value in names.items()},
            'refused': refused,
            'reported': [{'file': file, 'class': label, 'count': count, **({'encoding': encoding} if encoding else {})}
                         for (file, label, encoding), count in sorted(reported.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or ''))]}


def hygiene_report(hits, limit=40):
    """The refusal text, one entry per file: the file, its zone, each class (with its encoding when not 8-bit) with its hit
    count and the first line or byte offset of that class; never the matched text."""
    per_file = {}
    for hit in hits:
        classes = per_file.setdefault((hit['file'], hit['zone']), {})
        label = hit['class'] + ('/' + hit['reason'] if hit.get('reason') else '') + ('[' + hit['encoding'] + ']' if hit.get('encoding') else '')
        where = ':' + str(hit['line']) if hit.get('line') is not None else ('@' + str(hit['offset']) if hit.get('offset') is not None else '')
        count, first = classes.get(label, (0, where))
        classes[label] = (count + 1, first)
    rows = []
    for (file, zone), classes in sorted(per_file.items())[:limit]:
        detail = ', '.join(f'{label} x{count} (first{first})' if first else f'{label} x{count}' for label, (count, first) in sorted(classes.items()))
        rows.append(f'{file} [{zone}] {detail}')
    return '; '.join(rows) + (f' ... {len(per_file) - limit} more file(s)' if len(per_file) > limit else '')


def verify_distribution_hygiene(destination, private_directory):
    """Run distribution_hygiene over the unsealed directory, retain the full report privately, refuse on any refused hit and
    otherwise write the path-free BUILD_HYGIENE.json summary into the distribution."""
    destination, private_directory = Path(destination).resolve(), Path(private_directory).resolve()
    if private_directory == destination or private_directory.is_relative_to(destination):
        raise ValueError('The private hygiene report must remain outside the distribution')
    summary_path = destination / 'BUILD_HYGIENE.json'
    if summary_path.exists() or (destination / 'MANIFEST.json').exists():
        raise ValueError('Distribution hygiene runs once, on the unsealed directory')
    report = distribution_hygiene(destination)
    private_directory.mkdir(parents=True, exist_ok=True)
    (private_directory / 'distribution-hygiene.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if report['refused']:
        files = len({hit['file'] for hit in report['refused']})
        raise ValueError(f'Distribution hygiene refused {len(report["refused"])} hit(s) in {files} file(s): ' + hygiene_report(report['refused']))
    counts, wide = {}, {}
    for row in report['reported']:
        counts[row['class']] = counts.get(row['class'], 0) + row['count']
        if row.get('encoding') == UTF16_ENCODING:
            wide[row['class']] = wide.get(row['class'], 0) + row['count']
    summary = {'passed': True, 'scope': report['scope'], 'files_scanned': report['files_scanned'],
               'archive_members_scanned': report['archive_members_scanned'], 'bytes_scanned': report['bytes_scanned'],
               'zones': report['zones'], 'classes': report['classes'], 'refused_hits': 0,
               'reported_third_party_residue': {'files': len({row['file'] for row in report['reported']}), 'by_class': dict(sorted(counts.items()))},
               # the UTF-16LE share of the residue above (wide strings of collected Windows binaries), counted separately
               'utf16le': {'refused_hits': 0, 'reported_third_party_residue': {
                   'files': len({row['file'] for row in report['reported'] if row.get('encoding') == UTF16_ENCODING}), 'by_class': dict(sorted(wide.items()))}},
               'full_report': 'Retained by the builder outside this distribution (file, class, zone, offset and line; no matched text)'}
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['engine', 'full'], default='full')
    parser.add_argument('--validation',choices=['full','offline'],default='full',
                        help='Full cold HTTP/registry checks (default), or explicit local CLI/stdio metadata checks with HTTP/model/UI unqualified')
    parser.add_argument('--output', type=Path, required=True, help='New standalone application directory')
    parser.add_argument('--work-dir', type=Path, required=True, help='Build scratch directory; caches can be reused')
    parser.add_argument('--plan', action='store_true', help='Print required/runtime package plan without building')
    parser.add_argument('--zip', action='store_true', help='Also create a portable ZIP beside the output directory')
    parser.add_argument('--ffmpeg-dir', type=Path, help='Source-built FFmpeg directory with BUILD.json, matching binaries and retained source')
    parser.add_argument('--sherpa-source-dir', type=Path, help='Hash-bound Sherpa1.13.8 native source/notices bundle; required when that optional runtime is included')
    parser.add_argument('--tesseract-dir', type=Path, help='Verified model-free native Tesseract bundle with complete source, licenses and manifest')
    parser.add_argument('--native-audio-dir', type=Path,
                        help='Verified libsndfile-minimal and soxr-built source/provenance directories for the Qwen speech profile')
    parser.add_argument('--music-runtime-dir', type=Path,
                        help='Verified model-free ACE runtime; included before final application manifest and ZIP sealing')
    parser.add_argument('--allow-restricted', action='append', default=[], metavar='NAME',
                        help='Include this copyleft or non-commercial distribution anyway (repeatable); recorded in the plan '
                             'and DEPENDENCIES.json. Such a build is for the operator, not for redistribution')
    args = parser.parse_args()
    output, work = args.output.expanduser().resolve(), args.work_dir.expanduser().resolve()
    if output == Path(output.anchor) or output == ROOT or ROOT.is_relative_to(output):
        parser.error('Output must be a new application directory, not a filesystem or source ancestor')
    if output.exists():
        parser.error('Output directory exists; choose a new output path')
    archive_path = output.parent / (output.name + '.zip')
    if args.zip and archive_path.exists():
        parser.error('ZIP output already exists; choose a new output path')
    plan = build_plan(args.profile, args.allow_restricted)
    plan['validation_mode']=args.validation
    if args.music_runtime_dir:plan['native_music']=music_runtime_plan(args.music_runtime_dir)
    if 'sherpa_onnx' in plan['collect_packages']:
        if args.sherpa_source_dir is None:
            parser.error('The optional Sherpa runtime requires --sherpa-source-dir with pinned mixed-license sources and notices')
        from sherpa_package import source_plan
        plan['native_sherpa'] = source_plan(args.sherpa_source_dir)
    elif args.sherpa_source_dir is not None:
        parser.error('--sherpa-source-dir requires the installed optional Sherpa runtime')
    if args.tesseract_dir:
        if 'PIL' not in plan['collect_packages']:
            parser.error('--tesseract-dir requires a runtime profile containing Pillow')
        plan['native_tesseract'] = tesseract_plan(args.tesseract_dir)
    if args.native_audio_dir:
        if 'qwen_tts' not in plan['collect_packages']:
            parser.error('--native-audio-dir requires the installed Qwen speech profile')
        plan['native_audio'] = native_audio_plan(args.native_audio_dir)
    elif 'qwen_tts' in plan['collect_packages']:
        parser.error('The Qwen binary profile requires --native-audio-dir and its verified source-built import overlay')
    if args.ffmpeg_dir:
        plan.update(ffmpeg_plan(args.ffmpeg_dir))
    elif args.profile == 'full' and args.zip:
        parser.error('A full redistributable ZIP requires --ffmpeg-dir with retained matching source; the wheel binary can be used for local candidate validation only')
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    work.mkdir(parents=True, exist_ok=True)
    run = work / ('run-' + uuid.uuid4().hex)
    run.mkdir()
    plan_file = run / 'build-plan.json'
    plan_file.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    env = {**os.environ, 'TITANOS_BINARY_PLAN': str(plan_file)}
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--distpath', str(run / 'dist'),
                    '--workpath', str(work / 'pyinstaller'), str(ROOT / 'tools' / 'titanos.spec')],
                   cwd=ROOT, env=env, check=True)
    built = (run / 'dist' / 'titanos').resolve()
    if not built.is_relative_to(run.resolve()) or not built.is_dir():
        raise RuntimeError('Expected build directory was not produced')
    executable = built / ('titanos.exe' if os.name == 'nt' else 'titanos')
    if not executable.is_file():
        raise RuntimeError('Expected executable was not produced')
    current_files = {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
                     for path in sorted((ROOT / 'src' / 'titanos').rglob('*'))
                     if path.is_file() and path.suffix in {'.py', '.html', '.css', '.js', '.svg', '.json'}}
    if current_files != plan['source_files']:
        raise RuntimeError(f'Source changed during compilation; candidate remains at {built}. Rebuild before publishing it.')
    source_package(built)
    copy_tesseract(built, plan)
    write_notices(built, plan)
    include_music_runtime(built, plan)
    validate_packaged_distributions(built)
    verify_broker_payload(built)
    for filename in ['LICENSE', 'NOTICE']:
        shutil.copy2(ROOT / filename, built / filename)
    public_plan = {key: value for key, value in plan.items() if key not in {'source_root', 'ffmpeg_dir', 'ffmpeg_binaries', 'native_audio', 'native_tesseract', 'native_sherpa', 'native_music'}}
    if plan.get('native_music'):
        public_plan['native_music']={key:value for key,value in plan['native_music'].items() if key!='directory'}
    if plan.get('native_sherpa'):
        public_plan['native_sherpa'] = {key: value for key, value in plan['native_sherpa'].items() if key != 'directory'}
    if plan.get('native_tesseract'):
        public_plan['native_tesseract'] = {key: value for key, value in plan['native_tesseract'].items() if key != 'directory'}
    if plan.get('native_audio'):
        public_plan['native_audio'] = {'libraries': plan['native_audio']['libraries']}
    (built / 'BUILD.json').write_text(json.dumps(public_plan, indent=2) + '\n', encoding='utf-8')
    smoke(executable, built, validation=args.validation)
    summary=publish_smoke_summary(built, run / 'validation-private')
    validate_smoke_summary(summary,args.validation)
    validate_packaged_distributions(built)
    hygiene = verify_distribution_hygiene(built, run / 'validation-private')
    manifest = package_manifest(built)
    (built / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    # The user named this resolved destination above; refuse replacement.
    if output.exists():
        raise ValueError('Output appeared during build; preserving both directories')
    shutil.copytree(built, output)
    if args.zip:
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    archive.write(path, Path(output.name) / path.relative_to(output))
    print(json.dumps({'passed': args.validation=='full','build_completed':True,'validation_mode':args.validation,
                      'offline_passed':summary.get('offline_passed',False),'directory': str(output), 'executable': str(output / executable.name),
                      'zip': str(archive_path) if args.zip else None,
                      'file_count': len(manifest), 'runtime_packages': len(plan['packages']),
                      'hygiene_files_scanned': hygiene['files_scanned'], 'hygiene_refused_hits': 0,
                      'bytes': sum(row['bytes'] for row in manifest.values()), 'weights_included': False}, indent=2))


if __name__ == '__main__':
    main()
