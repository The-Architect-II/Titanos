# Native-platform PyInstaller build; configuration is emitted by build_binary.py.
import json
import os
from pathlib import Path
from importlib.util import find_spec

from PyInstaller.utils.hooks import collect_all, copy_metadata

settings = json.loads(Path(os.environ['TITANOS_BINARY_PLAN']).read_text(encoding='utf-8'))
root = Path(settings['source_root'])
datas, binaries, hiddenimports = [], [], []

def runtime_module(name):
    if not name.startswith('titanos.') and any(part in {'cli', 'commands'} for part in name.split('.')):
        return False
    return not any(part in {'tests', 'test', 'testing', 'benchmarks'} for part in name.split('.'))

for package in settings['collect_packages']:
    excludes = ['**/tests/**', '**/test/**', '**/__pycache__/**', '**/.cache/**',
                '**/*.safetensors', '**/*.gguf', '**/*.ckpt', '**/*.pt', '**/*.pth', '**/*.onnx']
    if settings.get('ffmpeg_dir') and package == 'imageio_ffmpeg':
        excludes += ['binaries/*', '**/binaries/*']
    package_data, package_binaries, package_imports = collect_all(
        package, include_py_files=True, filter_submodules=runtime_module,
        exclude_datas=excludes,
        on_error='warn once')
    datas.extend(package_data)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_imports)

# cuDNN and NVRTC load companion libraries by basename at runtime. ELF dependency
# scanning finds their frontends but misses these plugins in namespace subfolders.
if settings['os'] == 'linux' and settings['profile'] == 'full':
    namespace = find_spec('nvidia')
    for package_root in namespace.submodule_search_locations if namespace else []:
        for library in sorted(Path(package_root).glob('*/lib/*.so*')):
            if library.is_file():
                binaries.append((str(library), '.'))

# Frameworks inspect original source and installed version metadata at runtime.
# Collecting only bytecode breaks lazy architecture imports and TorchScript.
for distribution in settings['distributions']:
    datas.extend(copy_metadata(distribution))
datas.extend([(str(root / 'LICENSE'), '.'), (str(root / 'NOTICE'), '.')])
if settings.get('ffmpeg_dir'):
    datas.append((settings['ffmpeg_dir'], 'native/ffmpeg'))
if settings.get('native_audio'):
    for library in settings['native_audio']['libraries']:
        component = library['component']
        datas.append((str(Path(settings['native_audio']['directory']) / component), 'native/audio/' + component))
hiddenimports.extend(['titanos.device_probe', 'titanos.native_process', 'titanos.process_command',
                      'uvicorn.lifespan.on', 'uvicorn.protocols.http.h11_impl', 'uvicorn.loops.asyncio',
                      # inbound WebSocket upgrades (native passthrough Realtime/Live relays): uvicorn loads its
                      # WebSocket protocol by name, so the frozen build names it and the websockets package
                      'uvicorn.protocols.websockets.auto', 'uvicorn.protocols.websockets.websockets_impl',
                      'websockets'])

excluded_modules = ['tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'IPython', 'pytest', 'gradio']
excluded_modules += settings.get('exclude_modules', [])
if 'qwen_tts' not in settings['collect_packages']:
    excluded_modules += ['soundfile', '_soundfile', '_soundfile_data']

analysis = Analysis([str(root / 'tools' / 'frozen_entry.py')], pathex=[str(root / 'src')],
                    binaries=binaries, datas=datas, hiddenimports=sorted(set(hiddenimports)),
                    hookspath=[str(root / 'tools' / 'binary_hooks')] if settings.get('ffmpeg_dir') else [],
                    excludes=excluded_modules,
                    noarchive=False, optimize=0)
if settings['profile'] == 'engine':
    # Hooks may copy metadata for distributions whose modules are excluded above; the engine inventory
    # must equal its declared closure, so drop dist-info entries that belong to other distributions.
    from packaging.utils import canonicalize_name
    closure = {canonicalize_name(name) for name in settings['distributions']}

    def in_closure(entry):
        head = str(entry[0]).replace('\\', '/').split('/')[0]
        if not head.endswith('.dist-info'):
            return True
        return canonicalize_name(head[:-len('.dist-info')].rsplit('-', 1)[0]) in closure
    analysis.datas = [entry for entry in analysis.datas if in_closure(entry)]
archive = PYZ(analysis.pure)
executable = EXE(archive, analysis.scripts, [], exclude_binaries=True, name='titanos',
                 debug=False, strip=False, upx=False, console=True)
COLLECT(executable, analysis.binaries, analysis.datas, strip=False, upx=False, name='titanos')
