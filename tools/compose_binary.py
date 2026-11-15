"""Compose a fresh engine build with an immutable, matching full runtime cohort.

Run under the same CPython/PyInstaller toolchain and an owning resource Job.
Only unchanged dependency files are hardlinked. Application/source/metadata are
independent copies from the fresh engine build; old Titanos code is never reused.
The normal cold CLI/HTTP/MCP checks run before a final manifest is published.
"""
import argparse
import hashlib
import importlib.util
import json
import marshal
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import types
import uuid
import zipfile

from PyInstaller.archive.readers import CArchiveReader
from build_binary import validate_packaged_distributions


METADATA = {'BUILD.json', 'MANIFEST.json', 'BUILD_SMOKE.json', 'DEPENDENCIES.json',
            'sbom.cdx.json', 'NATIVE_COMPONENTS.json', 'THIRD_PARTY_NOTICES.md', 'COMPOSITION.json'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def ordinary(path):
    path = path.absolute()
    require('..' not in path.parts, 'Package paths must be canonical without parent traversal')
    for item in (*reversed(path.parents), path):
        value = item.lstat()
        require(not stat.S_ISLNK(value.st_mode) and not getattr(value, 'st_file_attributes', 0) & 0x400,
                'Linked/reparse package input is not supported: ' + str(item))
    return path


def member(name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and path.as_posix() == name
            and '..' not in path.parts and '\\' not in name and ':' not in name,
            'Unsafe package member')
    return name


def inventory(root):
    manifest = read(root / 'MANIFEST.json')
    manifest['MANIFEST.json'] = pin(root / 'MANIFEST.json')
    require(len(manifest) == len({n.casefold() for n in manifest}), 'Case-colliding package members')
    actual = set()
    for path in root.rglob('*'):
        ordinary(path)
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(manifest), 'Package differs from its sealed manifest')
    for name in manifest:
        member(name)
    return manifest


def normal(code):
    return code.replace(co_filename='<bound-source>', co_consts=tuple(
        normal(item) if isinstance(item, types.CodeType) else item for item in code.co_consts))


def archive(path):
    container = CArchiveReader(str(path))
    names = [n for n, value in container.toc.items() if value[-1] == 'z']
    require(len(names) == 1, 'Expected one compiled PYZ')
    return container, container.open_embedded_archive(names[0])


def application(name):
    parts = PurePosixPath(name).parts
    return (name == 'titanos.exe' or parts[0] == 'SOURCE' or name.startswith('_internal/titanos/')
            or name.startswith('THIRD_PARTY_LICENSES/titanos-engine/')
            or len(parts) > 1 and parts[0] == '_internal' and parts[1].startswith('titanos_engine-'))


def crt_version(path):
    # pefile is an existing Windows PyInstaller dependency; parsing never loads
    # the DLL. The retained native cohort must not downgrade the engine's CRT.
    import pefile
    with pefile.PE(str(path), fast_load=True) as image:
        image.parse_data_directories(directories=[2])
        value = image.VS_FIXEDFILEINFO[0]
        return (value.FileVersionMS >> 16, value.FileVersionMS & 65535,
                value.FileVersionLS >> 16, value.FileVersionLS & 65535)


def bind_archives(core, runtime, core_build, runtime_build):
    require(core_build['profile'] == 'engine' and runtime_build['profile'] == 'full',
            'Composition requires a fresh engine and a separately sealed full build')
    for build in [core_build, runtime_build]:
        require(build['packages']['titanos-engine'] == build['version'], 'Application version differs from its own distribution metadata')
    require(core_build['os'] == runtime_build['os'] and core_build['architecture'] == runtime_build['architecture'],
            'Frozen platform or native extension architecture differs')
    require(core_build['python'] == runtime_build['python']
            and core_build['pyinstaller'] == runtime_build['pyinstaller'], 'Frozen toolchains differ')
    require(tuple(map(int, core_build['python'].split('.')[:2])) == sys.version_info[:2],
            'Use the matching Python bytecode interpreter')
    for name, value in core_build['packages'].items():
        if name != 'titanos-engine':
            require(runtime_build['packages'].get(name) == value, 'Dependency cohort differs: ' + name)
    ca, cp = archive(core / 'titanos.exe')
    ra, rp = archive(runtime / 'titanos.exe')
    for name in set(cp.toc) & set(rp.toc):
        if name == 'titanos' or name.startswith('titanos.'):
            continue
        left, right = cp.extract(name), rp.extract(name)
        require(normal(left) == normal(right) if isinstance(left, types.CodeType) and isinstance(right, types.CodeType)
                else left == right, 'Frozen dependency bytecode differs: ' + name)
    hooks = {}
    core_hooks = {n: normal(marshal.loads(ca.extract(n))) for n, v in ca.toc.items() if n.startswith('pyi_rth_') and v[-1] == 's'}
    for name, value in ra.toc.items():
        if name.startswith('pyi_rth_') and value[-1] == 's':
            code = normal(marshal.loads(ra.extract(name)))
            equivalent = [n for n, c in core_hooks.items() if c == code]
            require(equivalent, 'Full runtime hook is absent from the new engine: ' + name)
            hooks[name] = equivalent
    # The core's base library is kept only after comparing every stdlib code
    # object with the retained cohort. ZIP timestamps/filenames are not code.
    with zipfile.ZipFile(core / '_internal/base_library.zip') as left, zipfile.ZipFile(runtime / '_internal/base_library.zip') as right:
        require(set(left.namelist()) == set(right.namelist()), 'Base library membership differs')
        for name in left.namelist():
            a, b = left.read(name), right.read(name)
            require(a[:4] == b[:4] == importlib.util.MAGIC_NUMBER
                    and normal(marshal.loads(a[16:])) == normal(marshal.loads(b[16:])),
                    'Base library code differs: ' + name)
    return cp, rp, hooks


def assemble(core, runtime, output, *, core_manifest_sha256, runtime_manifest_sha256):
    core, runtime = ordinary(core), ordinary(runtime)
    output = output.absolute()
    ordinary(output.parent)
    require(not output.exists() and not output.is_relative_to(core) and not output.is_relative_to(runtime),
            'Output must be a fresh directory outside both retained inputs')
    require(output.drive.casefold() == runtime.drive.casefold(), 'Hardlinks require the runtime volume')
    require(digest(core / 'MANIFEST.json') == core_manifest_sha256, 'Fresh engine manifest identity differs')
    require(digest(runtime / 'MANIFEST.json') == runtime_manifest_sha256, 'Retained runtime manifest identity differs')
    cm, rm = inventory(core), inventory(runtime)
    # A sealed hash does not establish completeness of the component inventory.
    validate_packaged_distributions(core)
    validate_packaged_distributions(runtime)
    for root, manifest in [(core, cm), (runtime, rm)]:
        for name in ['BUILD.json', 'titanos.exe', '_internal/base_library.zip']:
            require(pin(root / name) == manifest[name], 'Executable/build identity differs: ' + name)
    cb, rb = read(core / 'BUILD.json'), read(runtime / 'BUILD.json')
    cp, rp, hooks = bind_archives(core, runtime, cb, rb)
    # Core identity includes all current application modules and source hashes.
    for name, expected in cb['source_files'].items():
        require(digest(core / 'SOURCE' / name) == expected, 'Fresh application source differs: ' + name)
        if name.endswith('.py'):
            module = name.removeprefix('src/').removesuffix('.py').replace('/', '.')
            module = module.removesuffix('.__init__')
            require(module in cp.toc and normal(cp.extract(module)) == normal(compile(
                (core / 'SOURCE' / name).read_bytes(), name, 'exec', dont_inherit=True)),
                'Fresh application PYZ differs: ' + module)
    chosen = {n.casefold(): ('core', n, value) for n, value in cm.items() if n not in METADATA}
    native_choices = {}
    for name, value in rm.items():
        if name in METADATA or application(name):
            continue
        key = name.casefold()
        if key in chosen:
            owner, present, prior = chosen[key]
            if prior != value and key in {'_internal/vcruntime140.dll', '_internal/vcruntime140_1.dll'}:
                cv, rv = crt_version(core / present), crt_version(runtime / name)
                require(cv[0] == rv[0] == 14 and rv >= cv, 'Retained native CRT would downgrade the engine')
                native_choices[name] = {'core': prior, 'runtime': value, 'core_version': cv,
                    'runtime_version': rv, 'selected': 'retained full native cohort'}
                chosen[key] = ('runtime', name, value)
                continue
            require(prior == value or present == '_internal/base_library.zip', 'Conflicting native/runtime member: ' + name)
        else:
            chosen[key] = ('runtime', name, value)
    output.mkdir()
    receipt = {'schema_version': 1, 'profile': 'composed-engine-full-runtime',
        'application_version': cb['version'], 'retained_runtime_application_version': rb['version'],
        'source_files': cb['source_files'], 'core_executable': pin(core / 'titanos.exe'),
        'runtime_executable': pin(runtime / 'titanos.exe'), 'core_manifest_sha256': core_manifest_sha256,
        'runtime_manifest_sha256': runtime_manifest_sha256, 'runtime_hooks_equivalent': hooks,
        'native_collision_choices': native_choices,
        'base_library': 'fresh engine; every stdlib code object compared with retained full cohort',
        'dependency_files': {}, 'dependency_bytecode': {}, 'namespace_directories': [],
        'old_application_reused': False, 'input_directories_modified': False,
        'qualification': None, 'qualification_note': 'Composition is dependency/source binding; model, hardware and quality qualification remain separate.'}
    for owner, name, expected in chosen.values():
        source = (core if owner == 'core' else runtime) / name
        require(pin(source) == expected, 'Input file differs from its sealed manifest: ' + name)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if owner == 'runtime':
            os.link(source, target)
            require(os.path.samefile(source, target), 'Hardlink identity differs')
            receipt['dependency_files'][name] = expected
        else:
            shutil.copyfile(source, target)
    for name in sorted(set(rp.toc) - set(cp.toc)):
        require(name != 'titanos' and not name.startswith('titanos.'), 'Old application module cannot be reused')
        path = output / '_internal' / name.replace('.', '/')
        kind = rp.toc[name][0]
        if kind == 3:  # PyInstaller's PEP420 namespace package marker.
            path.mkdir(parents=True, exist_ok=True)
            receipt['namespace_directories'].append(name)
            continue
        require(kind in {0, 1}, 'Unsupported PYZ member type')
        if path.with_suffix('.py').exists() or (path / '__init__.py').exists():
            continue
        target = path / '__init__.pyc' if kind == 1 else path.with_suffix('.pyc')
        require(not target.exists(), 'Refusing to replace an existing dependency bytecode file')
        target.parent.mkdir(parents=True, exist_ok=True)
        code = marshal.dumps(rp.extract(name))
        target.write_bytes(importlib.util.MAGIC_NUMBER + bytes(12) + code)
        receipt['dependency_bytecode'][name] = {'path': target.relative_to(output).as_posix(),
            'full_PYZ_marshaled_sha256': hashlib.sha256(code).hexdigest(), **pin(target)}
    # Metadata is always a new independent file; retained dependency inventories
    # stay attributable to their original cohort, never masquerade as a rebuild.
    for name in ['NATIVE_COMPONENTS.json', 'THIRD_PARTY_NOTICES.md']:
        require(pin(runtime / name) == rm[name], 'Runtime metadata differs: ' + name)
        shutil.copyfile(runtime / name, output / name)
    require(pin(runtime / 'DEPENDENCIES.json') == rm['DEPENDENCIES.json']
            and pin(core / 'DEPENDENCIES.json') == cm['DEPENDENCIES.json'], 'Dependency inventories differ')
    dependencies = read(runtime / 'DEPENDENCIES.json')
    application_components = [x for x in read(core / 'DEPENDENCIES.json')['components'] if x['name'] == 'titanos-engine']
    require(len(application_components) == 1 and application_components[0]['version'] == cb['version'],
            'Fresh application dependency metadata is missing or has the wrong version')
    require(sum(x['name'] == 'titanos-engine' for x in dependencies['components']) == 1,
            'Retained dependency inventory must identify the old application exactly once')
    dependencies['components'] = [application_components[0] if x['name'] == 'titanos-engine' else x for x in dependencies['components']]
    dependencies['scope'] = 'Retained full runtime cohort with fresh application metadata; see COMPOSITION.json'
    write(output / 'DEPENDENCIES.json', dependencies)
    for root, entries in [(core, cm), (runtime, rm)]:
        require(pin(root / 'sbom.cdx.json') == entries['sbom.cdx.json'], 'CycloneDX input identity differs')
    sbom, current_sbom = read(runtime / 'sbom.cdx.json'), read(core / 'sbom.cdx.json')
    app_component = current_sbom['metadata']['component']
    require(app_component['type'] == 'application' and app_component['name'] == 'Titanos'
            and app_component['version'] == cb['version'], 'Fresh CycloneDX application identity differs')
    current_app = [row for row in current_sbom['components'] if row['name'] == 'titanos-engine']
    require(len(current_app) == 1 and current_app[0]['version'] == cb['version'], 'Fresh CycloneDX distribution identity differs')
    require(sum(row['name'] == 'titanos-engine' for row in sbom['components']) == 1,
            'Retained CycloneDX inventory must identify the old application exactly once')
    sbom['components'] = [current_app[0] if row['name'] == 'titanos-engine' else row for row in sbom['components']]
    sbom['serialNumber'] = 'urn:uuid:' + str(uuid.uuid4())
    sbom['metadata'] = {**current_sbom['metadata'], 'component': app_component}
    sbom['metadata'].setdefault('properties', []).extend([
        {'name': 'titanos:core-manifest-sha256', 'value': core_manifest_sha256},
        {'name': 'titanos:retained-runtime-manifest-sha256', 'value': runtime_manifest_sha256}])
    write(output / 'sbom.cdx.json', sbom)
    build = {**cb, 'profile': receipt['profile'], 'collect_packages': rb['collect_packages'],
             'packages': {**rb['packages'], 'titanos-engine': cb['version']}, 'distributions': rb['distributions'],
             'composition': {'core_manifest_sha256': core_manifest_sha256, 'runtime_manifest_sha256': runtime_manifest_sha256,
                             'application_version': cb['version'], 'retained_runtime_application_version': rb['version'],
                             'dependency_inventories': 'retained matching full runtime; independent copies'}}
    for name in ['native_sherpa', 'native_audio', 'native_tesseract', 'native_music']:
        if name in rb:
            build[name] = rb[name]
    write(output / 'BUILD.json', build)
    validate_packaged_distributions(output)
    write(output / 'COMPOSITION.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['core', 'runtime', 'output', 'private-receipts']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--core-manifest-sha256', required=True)
    parser.add_argument('--runtime-manifest-sha256', required=True)
    parser.add_argument('--validation',choices=['full','offline'],default='full',
                        help='Full validation by default; offline performs only local CLI/stdio metadata checks')
    args = parser.parse_args()
    require(not args.private_receipts.exists(), 'Private receipt directory must be fresh')
    ordinary(args.private_receipts.parent)
    require(not any(args.private_receipts.absolute().is_relative_to(p.absolute()) for p in [args.core, args.runtime]),
            'Operational receipts cannot be inside either retained input')
    require(not args.private_receipts.absolute().is_relative_to(args.output.absolute()), 'Operational receipts cannot be inside the package')
    args.private_receipts.mkdir(parents=True)
    result = {'status': 'running', 'models_executed': 0,'validation_mode':args.validation}
    try:
        receipt = assemble(args.core, args.runtime, args.output,
            core_manifest_sha256=args.core_manifest_sha256, runtime_manifest_sha256=args.runtime_manifest_sha256)
        # The selected validation is mandatory; offline is never a full pass.
        require(digest(Path(__file__)) == read(args.core / 'BUILD.json')['build_recipe_files']['tools/compose_binary.py'],
                'Composition tool differs from the frozen current source')
        build_tool = Path(__file__).with_name('build_binary.py')
        require(digest(build_tool) == read(args.core / 'BUILD.json')['build_recipe_files']['tools/build_binary.py'],
                'Smoke tool differs from the frozen current source')
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import build_binary
        build_binary.smoke(args.output / 'titanos.exe', args.output,validation=args.validation)
        summary=build_binary.publish_smoke_summary(args.output, args.private_receipts / 'validation')
        build_binary.validate_smoke_summary(summary,args.validation)
        validate_packaged_distributions(args.output)
        build=read(args.output/'BUILD.json');build['validation_mode']=args.validation;write(args.output/'BUILD.json',build)
        for name, expected in receipt['dependency_files'].items():
            require(pin(args.output / name) == expected, 'Reused dependency changed during cold checks: ' + name)
        manifest = {p.relative_to(args.output).as_posix(): pin(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
        write(args.output / 'MANIFEST.json', manifest)
        result.update(status='passed_composed_directory' if args.validation=='full' else 'offline_composed_directory',
                      full_smoke_passed=summary['passed'],offline_passed=summary.get('offline_passed',False),
                      manifest=pin(args.output / 'MANIFEST.json'),
                      composition=pin(args.output / 'COMPOSITION.json'), executable=pin(args.output / 'titanos.exe'),
                      archive=None, archive_export='not_created', model_hardware_quality_qualification=None)
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write(args.private_receipts / 'RESULT.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
