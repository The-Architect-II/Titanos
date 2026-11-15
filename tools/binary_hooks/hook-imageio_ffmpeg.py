"""Use the explicitly source-bound FFmpeg build, not the wheel's native binary."""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files('imageio_ffmpeg', excludes=['binaries/*'])
hiddenimports = collect_submodules('imageio_ffmpeg')
