#!/usr/bin/env bash
# Native Apple Silicon LGPL FFmpeg, using only Apple system codec frameworks.
# Usage: bash build_ffmpeg_macos.sh ARCHIVE NEW_WORK_DIR NEW_OUTPUT_DIR [JOBS]
set -euo pipefail
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Requires native Apple Silicon macOS' >&2; exit 1; }
[[ $# -ge 3 && $# -le 4 ]] || { echo 'Expected source archive, new work directory, new output directory, optional jobs' >&2; exit 1; }
ffmpeg_archive="$1"
ffmpeg_work="$2"
ffmpeg_output="$3"
ffmpeg_jobs="${4:-8}"
build_python="${TITANOS_BUILD_PYTHON:-python3}"
recipe_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
[[ -f "$ffmpeg_archive" && ! -e "$ffmpeg_work" && ! -e "$ffmpeg_output" && "$ffmpeg_jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'Source archive, fresh paths and positive job count required' >&2; exit 1; }
ffmpeg_archive="$(cd "$(dirname "$ffmpeg_archive")" && pwd)/$(basename "$ffmpeg_archive")"
ffmpeg_output="$("$build_python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$ffmpeg_output")"
[[ "$(shasum -a 256 "$ffmpeg_archive" | cut -d ' ' -f 1)" == cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635 ]] || { echo 'FFmpeg 9.0.1 source digest mismatch' >&2; exit 1; }
mkdir -p "$ffmpeg_work/source" "$ffmpeg_work/build"
ffmpeg_work="$(cd "$ffmpeg_work" && pwd)"
tar -xf "$ffmpeg_archive" -C "$ffmpeg_work/source"
ffmpeg_source="$ffmpeg_work/source/ffmpeg-9.0.1"
ffmpeg_compiler="$(xcrun --find clang)"
ffmpeg_sdk="$(xcrun --sdk macosx --show-sdk-path)"
export SDKROOT="$ffmpeg_sdk"
ffmpeg_flags=(
  --target-os=darwin --arch=aarch64 "--cc=$ffmpeg_compiler" "--sysroot=$ffmpeg_sdk"
  --disable-autodetect --disable-gpl --disable-nonfree --disable-version3
  --disable-doc --disable-debug --disable-asm --disable-network
  --enable-pthreads --disable-static --enable-shared --disable-ffplay
  --disable-everything --enable-ffmpeg --enable-ffprobe --enable-videotoolbox
  --enable-protocol=file,pipe,fd --enable-demuxer=rawvideo,mov,wav,flac,ogg,mp3
  --enable-muxer=mp4,mov,rawvideo,image2pipe,null,pcm_f32le
  --enable-decoder=rawvideo,h264,mpeg4,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_f64le,mp3float,flac,vorbis,opus
  --enable-parser=h264,mpeg4video,mpegaudio,flac,opus,vorbis
  --enable-encoder=rawvideo,wrapped_avframe,mpeg4,h264_videotoolbox,pcm_f32le
  --enable-filter=scale,format,fps,null,aresample,aformat,anull --enable-swresample --install-name-dir=@rpath
  --extra-ldflags=-Wl,-headerpad_max_install_names
)
printf '%s\n' "${ffmpeg_flags[@]}" > "$ffmpeg_work/configure-flags.txt"
cd "$ffmpeg_work/build"
"$ffmpeg_source/configure" "--prefix=$ffmpeg_work/install" "${ffmpeg_flags[@]}"
make -j"$ffmpeg_jobs"
make install
"$build_python" - "$ffmpeg_archive" "$ffmpeg_work" "$ffmpeg_output" "$recipe_path" "$ffmpeg_compiler" <<'PY'
import hashlib, io, json, os, pathlib, shutil, subprocess, sys, tempfile, wave
archive, work, output, recipe, compiler = map(pathlib.Path, sys.argv[1:])
output.mkdir(parents=True)
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()
for source in (work/'install/bin').iterdir():
    if source.name in {'ffmpeg','ffprobe'}:
        shutil.copy2(source, output/source.name)
for source in (work/'install/lib').glob('*.dylib'):
    shutil.copy2(source, output/source.name)
binaries=[p for p in output.iterdir() if p.is_file()]
dependencies={}
for binary in binaries:
    lines=subprocess.check_output(['otool','-L',str(binary)],text=True).splitlines()[1:]
    for line in lines:
        dependency=line.strip().split(' (',1)[0]
        name=pathlib.Path(dependency).name
        if (output/name).is_file():
            subprocess.run(['install_name_tool','-change',dependency,'@loader_path/'+name,str(binary)],check=True)
        elif not dependency.startswith(('/usr/lib/','/System/Library/')):
            raise RuntimeError('Unbundled non-system dependency: '+dependency)
    if binary.suffix=='.dylib':
        subprocess.run(['install_name_tool','-id','@loader_path/'+binary.name,str(binary)],check=True)
    subprocess.run(['codesign','--force','--sign','-',str(binary)],check=True)
    subprocess.run(['codesign','--verify','--strict',str(binary)],check=True)
    dependencies[binary.name]=subprocess.check_output(['otool','-L',str(binary)],text=True)
source=output/'source'
source.mkdir()
shutil.copy2(archive,source/archive.name)
shutil.copy2(recipe,source/recipe.name)
shutil.copy2(work/'configure-flags.txt',source/'configure-flags.txt')
(source/'changes.diff').write_text('')
for name in ['COPYING.LGPLv2.1','LICENSE.md']:
    shutil.copy2(work/'source/ffmpeg-9.0.1'/name,output/name)
clean_env={k:v for k,v in os.environ.items() if k not in {'DYLD_LIBRARY_PATH','DYLD_FALLBACK_LIBRARY_PATH'}}
clean_env['PATH']='/usr/bin:/bin'
with tempfile.TemporaryDirectory(prefix='relocation-',dir=work) as temporary:
    relocated=pathlib.Path(temporary)/'ffmpeg'
    shutil.copytree(output,relocated)
    executable=relocated/'ffmpeg'
    version=subprocess.check_output([str(executable),'-version'],env=clean_env,text=True,stderr=subprocess.STDOUT)
    license_text=subprocess.check_output([str(executable),'-L'],env=clean_env,text=True,stderr=subprocess.STDOUT)
    assert 'GNU Lesser General Public License' in ' '.join(license_text.split()) and '--enable-gpl' not in version
    clip=pathlib.Path(temporary)/'smoke.mp4'
    subprocess.run([str(executable),'-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','64x64',
                    '-framerate','4','-i','pipe:0','-frames:v','5','-c:v','h264_videotoolbox','-pix_fmt','yuv420p',str(clip)],
                   input=bytes([200,30,20])*64*64*5,env=clean_env,check=True)
    count=subprocess.check_output([str(relocated/'ffprobe'),'-v','error','-count_frames','-select_streams','v:0',
                    '-show_entries','stream=nb_read_frames','-of','json',str(clip)],env=clean_env,text=True)
    assert json.loads(count)['streams'][0]['nb_read_frames']=='5'
    wav=io.BytesIO()
    with wave.open(wav,'wb') as writer:
        writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(16000)
        writer.writeframes(bytes(8000))
    decoded=subprocess.run([str(executable),'-v','error','-i','pipe:0','-ac','1','-ar','16000',
                           '-f','f32le','-c:a','pcm_f32le','pipe:1'],input=wav.getvalue(),
                          env=clean_env,capture_output=True,check=True)
    assert len(decoded.stdout)==4000*4
(output/'version.txt').write_text(version)
(output/'license-output.txt').write_text(license_text)
(output/'dylib-dependencies.json').write_text(json.dumps(dependencies,indent=2))
receipt={'source_url':'https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.xz','source_sha256':digest(archive),'source_changes':'none',
         'configure':(work/'configure-flags.txt').read_text().splitlines(),
         'compiler':subprocess.check_output([str(compiler),'--version'],text=True),
         'license_from_binary':'LGPL-2.1-or-later','external_codec_libraries':[],
         'video_codec':'h264_videotoolbox','runtime_dependencies':'Bundled FFmpeg dylibs and macOS system frameworks',
         'relocated_encode_decode_smoke':{'passed':True,'frames':5,'width':64,'height':64},
         'audio_decode_smoke':{'passed':True,'input':'PCM16 WAV','output':'mono float32','samples':4000,'rate':16000},
         'binaries':{p.name:digest(p) for p in binaries}}
(output/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'passed':True,'directory':str(output),**receipt},indent=2))
PY
