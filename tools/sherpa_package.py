"""Retain the pinned optional Sherpa runtime's native sources and mixed notices."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or path.as_posix() != name or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Invalid Sherpa source-pack path: ' + name)
    return path


def source_plan(directory):
    root = Path(directory).resolve(strict=True)
    manifest = root / 'SOURCE_BUNDLE.json'
    value = json.loads(manifest.read_text(encoding='utf-8'))
    if value.get('runtime_version') != '1.13.8' or not value.get('components') or not value.get('files'):
        raise ValueError('Expected the complete Sherpa1.13.8 native source/notices bundle')
    for path in root.rglob('*'):
        if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('Source/notices bundle must contain ordinary files and directories')
    for name, pin in value['files'].items():
        path = root / relative(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or path.stat().st_size != pin['bytes'] or digest(path) != pin['sha256']:
            raise ValueError('Sherpa source/notices binding differs: ' + name)
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} != set(value['files']) | {'SOURCE_BUNDLE.json'}:
        raise ValueError('Unexpected source/notices bundle member')
    platform = 'windows' if sys.platform == 'win32' else 'linux' if sys.platform.startswith('linux') else sys.platform
    if platform not in value['native_files']:
        raise ValueError('Sherpa source pack does not qualify this platform')
    for component in value['components']:
        if not component.get('license') and not component.get('license_name'):
            raise ValueError('Native component lacks its retained license designation')
        if not component.get('license_files') or any(name not in value['files'] for name in component['license_files']):
            raise ValueError('Native component license text is not in the source bundle')
    for native in value['native_files'][platform]:
        name = native['name']
        if relative(name).name != name or any(char in name for char in '*?[]'):
            raise ValueError('Native binary names must be literal basenames')
    return {'directory': str(root), 'manifest_sha256': digest(manifest), 'platform': platform,
            'runtime_version': value['runtime_version'], 'source_git_revision': value['source_git_revision']}


def include_native_sources(destination, plan):
    root = Path(plan['directory'])
    if source_plan(root) != plan:
        raise ValueError('Sherpa source pack changed during compilation')
    bundle = json.loads((root / 'SOURCE_BUNDLE.json').read_text(encoding='utf-8'))
    destination = Path(destination)
    target = destination / 'THIRD_PARTY_SOURCES' / 'sherpa-onnx-1.13.8'
    shutil.copytree(root, target)
    for name, pin in bundle['files'].items():
        if digest(target / name) != pin['sha256']:
            raise ValueError('Copied Sherpa source/notices file differs: ' + name)
    native = []
    for expected in bundle['native_files'][plan['platform']]:
        matches = [p for p in (destination / '_internal').rglob(expected['name'])
                   if p.is_file() and p.stat().st_size == expected['bytes'] and digest(p) == expected['sha256']]
        if not matches:
            raise ValueError('Pinned Sherpa native binary is missing or differs: ' + expected['name'])
        native.append({**expected, 'packaged_paths': [p.relative_to(destination).as_posix() for p in matches]})
    selected = [component for component in bundle['components'] if plan['platform'] in component.get('platforms', ['windows', 'linux'])]
    native_path = destination / 'NATIVE_COMPONENTS.json'
    inventory = json.loads(native_path.read_text(encoding='utf-8'))
    record = {'name': 'Optional Sherpa1.13.8 native runtime', 'version': '1.13.8', 'source_git_revision': bundle['source_git_revision'],
              'source_bundle': target.relative_to(destination).as_posix(), 'source_manifest_sha256': plan['manifest_sha256'],
              'native_files': native, 'components': selected, 'provenance_limits': bundle.get('provenance_limits', []),
              'license_scope': 'Titanos source remains Apache-2.0; bundled third-party native code retains its own terms, including static eSpeak GPL and Linux ALSA LGPL.',
              'license_review': 'not_performed', 'model_weights_included': False}
    inventory.append(record)
    native_path.write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
    sbom_path = destination / 'sbom.cdx.json'
    sbom = json.loads(sbom_path.read_text(encoding='utf-8'))
    for component in selected:
        entry = {'type': 'library', 'name': component['name'], 'version': component['version'],
                 'licenses': ([{'expression': component['license']}] if component.get('license') else
                              [{'license': {'name': component['license_name']}}]),
                 'properties': [{'name': 'titanos:component-role', 'value': 'optional-sherpa-native'},
                                {'name': 'titanos:source-bundle', 'value': record['source_bundle']},
                                {'name': 'titanos:linkage', 'value': component.get('linkage', 'retained runtime component')}],
                 'externalReferences': [{'type': 'vcs', 'url': component['source_url']}]}
        sbom['components'].append(entry)
    sbom_path.write_text(json.dumps(sbom, indent=2) + '\n', encoding='utf-8')
    dependencies_path = destination / 'DEPENDENCIES.json'
    dependencies = json.loads(dependencies_path.read_text(encoding='utf-8'))
    sherpa = next(component for component in selected if component['name'] == 'sherpa-onnx')
    for distribution in dependencies['components']:
        if distribution['name'] == 'sherpa-onnx-core' and not distribution.get('license_files'):
            distribution['license_files'] = [{'path': (target / name).relative_to(destination).as_posix(),
                                               'sha256': digest(target / name), 'source_url': sherpa['source_url']}
                                              for name in sherpa['license_files']]
            distribution['native_third_party_terms'] = 'See separate eSpeak/Piper/ALSA native components and retained notices; distribution Apache metadata is not a bundle-wide license.'
    dependencies_path.write_text(json.dumps(dependencies, indent=2) + '\n', encoding='utf-8')
    notices = destination / 'THIRD_PARTY_NOTICES.md'
    with notices.open('a', encoding='utf-8') as stream:
        stream.write('\n## Optional Sherpa speech runtime\n\nTitanos source retains Apache-2.0. This optional native runtime is not an Apache-only bundle: '
                     'it includes statically linked Piper/eSpeak and, on Linux, ALSA. Exact source archives, build references and license texts are retained in '
                     + record['source_bundle'] + '. See SOURCE_BUNDLE.json for each component\'s terms and provenance limits. '
                     'The SBOM lists these native components separately from Python distribution license declarations. '
                     'This section covers the sherpa-onnx and sherpa-onnx-core Python distributions (the -core wheel carries the native library; '
                     'both take their license texts from the retained source bundle, as recorded in DEPENDENCIES.json license_files). This is not a legal approval. '
                     'Model checkpoints and their licenses remain external.\n')
    return record
