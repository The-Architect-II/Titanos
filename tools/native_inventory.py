"""Bind packaged native bytes to their source build and verified postprocessing."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def _digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Native binary is missing or outside its directory: ' + name)
    return path


def _command(*arguments):
    return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT, timeout=30)


def _dependencies(path):
    return [line.strip().split(' (', 1)[0] for line in _command('otool', '-L', str(path)).splitlines()[1:]]


def _rpaths(path):
    lines = _command('otool', '-l', str(path)).splitlines()
    result = []
    for index, line in enumerate(lines):
        if line.strip() == 'cmd LC_RPATH':
            result.append(lines[index + 2].strip().removeprefix('path ').rsplit(' (offset ', 1)[0])
    return result


def finalize_native_inventory(bundle_dir: Path, ffmpeg_source_dir: Path, pyinstaller_version: str, binaries=None) -> dict:
    """Leave identical Windows bytes alone; verify and describe Mach-O changes.

    ``binaries`` always identifies the packaged bytes. ``source_build_binaries``
    retains the original compiler output when PyInstaller adjusts/signs Mach-O.
    The source archive and compiler recipe remain unchanged. A bundle whose
    BUILD.json carries no ``binaries`` mapping is bound through the mapping the
    caller derived from its ``members`` inventory (build_binary.ffmpeg_plan);
    without either the bundle is refused.
    """
    bundle, source = Path(bundle_dir).resolve(), Path(ffmpeg_source_dir).resolve()
    target = bundle / '_internal/native/ffmpeg'
    original = json.loads((source / 'BUILD.json').read_text(encoding='utf-8'))
    if binaries is not None:
        if original.get('binaries') is not None and original['binaries'] != binaries:
            raise ValueError('The supplied FFmpeg binaries mapping differs from the bundle BUILD.json')
        original = {**original, 'binaries': dict(binaries)}
    if not isinstance(original.get('binaries'), dict) or not original['binaries']:
        raise ValueError('FFmpeg BUILD.json has no binaries mapping and none was derived from its members inventory')
    actual = {}
    for name, expected in original['binaries'].items():
        if _digest(_inside(source, name)) != expected:
            raise ValueError('Native source-build binary no longer matches its receipt: ' + name)
        actual[name] = _digest(_inside(target, name))
    changed = [name for name in actual if actual[name] != original['binaries'][name]]
    if not changed:
        return original
    if platform.system() != 'Darwin':
        raise ValueError('Unexpected native binary changes outside verified Mach-O postprocessing')
    executable = _inside(target, 'ffmpeg')
    main_rpaths = _rpaths(executable)
    evidence, resolved_evidence, visited = {}, {}, set()
    pending = [(name, _inside(target, name)) for name in changed]
    while pending:
        name, after = pending.pop()
        if after in visited:
            continue
        visited.add(after)
        before = _inside(source, name)
        # Omit only the filename heading; retain addresses and every text byte.
        before_text = '\n'.join(_command('otool', '-s', '__TEXT', '__text', str(before)).splitlines()[1:])
        after_text = '\n'.join(_command('otool', '-s', '__TEXT', '__text', str(after)).splitlines()[1:])
        if not before_text or before_text != after_text:
            raise ValueError('Native executable code changed during packaging: ' + name)
        dependencies = _dependencies(after)
        own_rpaths = _rpaths(after)
        rpaths = own_rpaths + main_rpaths
        resolutions = {}
        identifiers = _command('otool', '-D', str(after)).splitlines()[1:]
        install_id = identifiers[0].strip() if identifiers else None
        for dependency in dependencies:
            # otool -L lists a dylib's own LC_ID_DYLIB before its dependencies.
            if dependency == install_id:
                continue
            if dependency.startswith(('/usr/lib/', '/System/Library/')):
                continue
            options = ([(p + '/' + dependency.removeprefix('@rpath/'), base)
                        for paths, base in ((own_rpaths, after.parent), (main_rpaths, executable.parent)) for p in paths]
                       if dependency.startswith('@rpath/') else [(dependency, after.parent)])
            resolved = [Path(p.replace('@loader_path', str(base)).replace('@executable_path', str(executable.parent))).resolve()
                        for p, base in options]
            selected = next((p for p in resolved if p.is_relative_to(bundle) and p.is_file()), None)
            if selected is None:
                raise ValueError('Packaged native dependency does not resolve inside the bundle: ' + dependency)
            resolutions[dependency] = selected.relative_to(bundle).as_posix()
            # PyInstaller may put the libraries actually loaded via @rpath at
            # _internal, in addition to the copies under native/ffmpeg.
            dependency_name = Path(dependency).name
            if dependency_name not in original['binaries']:
                raise ValueError('Native dependency has no corresponding source-build binary: ' + dependency_name)
            pending.append((dependency_name, selected))
        _command('codesign', '--verify', '--strict', str(after))
        item = {'source_binary': name, 'source_sha256': original['binaries'][name], 'packaged_sha256': _digest(after),
                          'text_section_unchanged': True, 'text_dump_sha256': hashlib.sha256(after_text.encode()).hexdigest(),
                          'dependencies_before': _dependencies(before), 'dependencies_after': dependencies,
                          'resolved_dependencies': resolutions, 'install_id': install_id, 'rpaths': rpaths, 'signature_verified': True}
        if after == _inside(target, name):
            evidence[name] = item
        else:
            resolved_evidence[after.relative_to(bundle).as_posix()] = item
    env = {k: v for k, v in os.environ.items() if k not in {'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH'}}
    subprocess.run([str(executable), '-version'], env=env, capture_output=True, check=True, timeout=30)
    record = {**original, 'source_build_binaries': original['binaries'], 'binaries': actual,
              'postprocessing': {'tool': 'PyInstaller', 'version': pyinstaller_version,
                                 'operation': 'Mach-O loader-path normalization and ad-hoc signing',
                                 'code_sections_unchanged': True, 'files': evidence,
                                 'resolved_dependency_files': resolved_evidence}}
    temporary = target / 'BUILD.json.tmp'
    try:
        temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        temporary.replace(target / 'BUILD.json')
    finally:
        temporary.unlink(missing_ok=True)
    return record
