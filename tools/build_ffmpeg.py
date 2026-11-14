"""Build a small LGPL FFmpeg from a supplied official source archive on Windows.

Uses an existing MSYS2 Bash and MinGW toolchain. Nothing is installed systemwide.
The output retains the source archive, configure recipe, notices and binary hashes.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import shlex
import subprocess
import tarfile


FLAGS = [
    '--target-os=mingw32', '--arch=x86_64', '--cc=gcc', '--cxx=g++',
    '--disable-autodetect', '--disable-gpl', '--disable-nonfree', '--disable-version3',
    '--disable-doc', '--disable-debug', '--disable-x86asm', '--disable-network',
    '--disable-pthreads', '--enable-w32threads', '--disable-static', '--enable-shared',
    '--disable-ffplay', '--disable-everything', '--enable-ffmpeg', '--enable-ffprobe',
    '--enable-protocol=file,pipe,fd', '--enable-demuxer=rawvideo,mov,wav,flac,ogg,mp3',
    '--enable-muxer=mp4,mov,rawvideo,image2pipe,null,pcm_f32le',
    '--enable-decoder=rawvideo,h264,mpeg4,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_f64le,mp3float,flac,vorbis,opus',
    '--enable-parser=h264,mpeg4video,mpegaudio,flac,opus,vorbis',
    '--enable-encoder=rawvideo,wrapped_avframe,mpeg4,h264_mf,pcm_f32le',
    '--enable-filter=scale,format,fps,null,aresample,aformat,anull', '--enable-mediafoundation', '--enable-d3d11va',
    '--extra-ldflags=-static-libgcc -static',
]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--bash', type=Path, required=True)
    parser.add_argument('--toolchain-dir', type=Path, required=True)
    parser.add_argument('--make', type=Path, required=True, help='Existing MSYS2 GNU make executable')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--package-existing', action='store_true', help='Package an existing build in the named work directory after checking its configuration')
    args = parser.parse_args()
    archive, bash, toolchain, work, output = [value.expanduser().resolve() for value in
        (args.source_archive, args.bash, args.toolchain_dir, args.work_dir, args.output)]
    make = args.make.expanduser().resolve()
    if os.name != 'nt':
        parser.error('This recipe targets native Windows; build the same FFmpeg features with the native compiler on other platforms')
    if not archive.is_file() or not bash.is_file() or not make.is_file() or not (toolchain / 'gcc.exe').is_file():
        parser.error('Source archive, existing Bash and MinGW compiler are required')
    if output.exists() or (work.exists() and not args.package_existing) or args.jobs < 1:
        parser.error('Choose new work/output directories and a positive job count')
    work.mkdir(parents=True, exist_ok=args.package_existing)
    source_dir = work / 'source'
    if not args.package_existing:
        source_dir.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(source_dir, filter='data')
    sources = list(source_dir.glob('ffmpeg-*'))
    if len(sources) != 1 or not (sources[0] / 'configure').is_file():
        raise ValueError('Expected one FFmpeg source root')
    source = sources[0]
    build = work / 'build'
    build.mkdir(exist_ok=args.package_existing)
    install = work / 'install'
    # Paths enter Bash as environment values, not interpolated shell code.
    recipe = work / 'build.sh'
    recipe.write_text('''#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/bin:$PATH"
make_bin="$(cygpath -u "$TITANOS_FFMPEG_MAKE")"
export PATH="$(dirname "$make_bin"):$(cygpath -u "$TITANOS_FFMPEG_TOOLCHAIN"):/usr/bin"
cd "$(cygpath -u "$TITANOS_FFMPEG_BUILD")"
source_dir="$(cygpath -u "$TITANOS_FFMPEG_SOURCE")"
install_dir="$(cygpath -u "$TITANOS_FFMPEG_INSTALL")"
"$source_dir/configure" --prefix="$install_dir" ''' + ' '.join(shlex.quote(flag) for flag in FLAGS) + '''
"$make_bin" -j"$TITANOS_FFMPEG_JOBS"
"$make_bin" install
''', encoding='utf-8', newline='\n')
    env = {**os.environ, 'TITANOS_FFMPEG_TOOLCHAIN': str(toolchain), 'TITANOS_FFMPEG_SOURCE': str(source),
           'TITANOS_FFMPEG_BUILD': str(build), 'TITANOS_FFMPEG_INSTALL': str(install),
           'TITANOS_FFMPEG_MAKE': str(make),
           'TITANOS_FFMPEG_JOBS': str(args.jobs)}
    env.pop('MSYS2_ARG_CONV_EXCL', None)
    env.pop('MSYS_NO_PATHCONV', None)
    if not args.package_existing:
        with (work / 'build.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(bash), '--noprofile', '--norc', recipe.as_posix()], env=env, stdout=log, stderr=log, check=True)
    executable = install / 'bin' / 'ffmpeg.exe'
    if not executable.is_file():
        raise RuntimeError('FFmpeg executable was not built')
    minimal_env = {**os.environ, 'PATH': str(Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32')}
    version = subprocess.check_output([str(executable), '-version'], env=minimal_env, stderr=subprocess.STDOUT, text=True, timeout=20)
    license_text = subprocess.check_output([str(executable), '-L'], env=minimal_env, stderr=subprocess.STDOUT, text=True, timeout=20)
    configuration = next(line.removeprefix('configuration: ') for line in version.splitlines() if line.startswith('configuration: '))
    if not set(FLAGS) <= set(shlex.split(configuration)):
        raise RuntimeError('Built configuration differs from this source recipe')
    if 'GNU Lesser General Public License' not in ' '.join(license_text.split()) or '--enable-gpl' in version or '--enable-nonfree' in version:
        raise RuntimeError('Built binary did not confirm the intended LGPL configuration')
    output.mkdir(parents=True)
    for binary in (install / 'bin').iterdir():
        if binary.is_file() and binary.suffix.lower() in {'.exe', '.dll'}:
            shutil.copy2(binary, output / binary.name)
    retained = output / 'source'
    retained.mkdir()
    shutil.copy2(archive, retained / archive.name)
    shutil.copy2(recipe, retained / 'build.sh')
    shutil.copy2(Path(__file__), retained / 'build_ffmpeg.py')
    for name in ['COPYING.LGPLv2.1', 'LICENSE.md']:
        shutil.copy2(source / name, output / name)
    notices = Path(__file__).parent / 'licenses'
    runtime_notices = [row for row in json.loads((notices / 'provenance.json').read_text(encoding='utf-8'))
                       if row['name'] in {'gcc-runtime', 'mingw-w64', 'winpthreads'}]
    (output / 'compiler-runtime-notices').mkdir()
    for notice in runtime_notices:
        license_file = notices / notice['file']
        if sha256(license_file) != notice['sha256']:
            raise ValueError('Compiler runtime notice checksum mismatch')
        shutil.copy2(license_file, output / 'compiler-runtime-notices' / notice['file'])
    (output / 'compiler-runtime-notices' / 'provenance.json').write_text(json.dumps(runtime_notices, indent=2) + '\n', encoding='utf-8')
    (retained / 'changes.diff').write_text('', encoding='utf-8')
    for suffix in ['.asc']:
        signature = Path(str(archive) + suffix)
        if signature.is_file():
            shutil.copy2(signature, retained / signature.name)
    for name in ['ffmpeg-devel.asc', 'signature-verification-corrected.log']:
        evidence = archive.parent / name
        if evidence.is_file():
            shutil.copy2(evidence, retained / name)
    (output / 'version.txt').write_text(version, encoding='utf-8')
    (output / 'license-output.txt').write_text(license_text, encoding='utf-8')
    buildconf = subprocess.check_output([str(executable), '-buildconf'], env=minimal_env, stderr=subprocess.STDOUT, text=True, timeout=20)
    (output / 'buildconf.txt').write_text(buildconf, encoding='utf-8')
    imports = {p.name: re.findall(r'DLL Name: (.+)', subprocess.check_output([str(toolchain / 'objdump.exe'), '-p', str(p)], text=True))
               for p in output.iterdir() if p.suffix.lower() in {'.exe', '.dll'}}
    provided = {name.lower() for name in imports}
    system = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32'
    unknown = sorted({name for names in imports.values() for name in names
                      if name.lower() not in provided and not name.lower().startswith(('api-ms-', 'ext-ms-')) and not (system / name).is_file()})
    if unknown:
        raise RuntimeError('Unexpected native runtime dependencies: ' + ', '.join(unknown))
    receipt = {'source_url': args.source_url, 'source_sha256': sha256(archive), 'source_changes': 'none',
               'configure': FLAGS, 'compiler': subprocess.check_output([str(toolchain / 'gcc.exe'), '--version'], text=True),
               'license_from_binary': 'LGPL-2.1-or-later', 'external_codec_libraries': [], 'video_codec': 'h264_mf',
               'runtime_dependencies': 'Windows system DLLs and bundled FFmpeg DLLs', 'pe_imports': imports,
               'binaries': {p.name: sha256(p) for p in output.iterdir() if p.suffix.lower() in {'.exe', '.dll'}}}
    (output / 'BUILD.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'directory': str(output), **receipt}, indent=2))


if __name__ == '__main__':
    main()
