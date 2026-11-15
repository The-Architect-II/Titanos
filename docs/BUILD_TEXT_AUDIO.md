# Standalone CPU text and audio profile

`tools/build_vendor_binary.py --runtime text-audio` packages the CPU text runtime together with ONNX Runtime VAD and the existing audio decoder. It includes full MCP, source, dependency notices, CycloneDX SBOM and native-source bindings. Models remain external. This profile does not include Vega/Qwen TTS or replace the retained full CUDA speech release.

Build in a new native environment on the target OS/architecture. For the current Windows x64 candidate, use Python 3.12 and the official `torch==2.8.0+cpu` wheel, then install the frozen Titanos source with `[sentiment,vad,build]`. Do not inherit another environment's site-packages. The source requirements retain Transformers 4.57.x and ONNX Runtime 1.27.0. This profile rejects CUDA/ROCm/XPU PyTorch metadata or CUDA dependency closure, and also checks the real Torch build before compilation.

```powershell
# Run only after the integrated source and external build inputs are frozen.
python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[sentiment,vad,build]"
python tools/build_vendor_binary.py --runtime text-audio --ffmpeg-dir D:/BuildInputs/ffmpeg-lgpl --plan --output D:/NewBuild/titanos-text-audio --work-dir D:/NewBuild/work
python tools/build_vendor_binary.py --runtime text-audio --ffmpeg-dir D:/BuildInputs/ffmpeg-lgpl --output D:/NewBuild/titanos-text-audio --work-dir D:/NewBuild/work --zip
```

`--ffmpeg-dir` is mandatory for both planning and building; there is no wheel-binary fallback for this profile. Supply the already-qualified source-built FFmpeg directory with its `BUILD.json`, all executable/DLL files, corresponding source archive, recipes and license texts. The planner checks recorded binary hashes, matching source SHA-256, and the LGPL text against that archive, then binds every input file. Compilation refuses changed inputs or altered collected source/notice files. The existing native inventory helper validates unchanged Windows binaries and permitted, verified Mach-O postprocessing separately.

The accepted Windows input is FFmpeg 9.0.1, LGPL-2.1-or-later, corresponding-source SHA-256 `cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635`, `ffmpeg.exe` SHA-256 `4adf14dc0f887d225407d0402cc2e81d9a88b38ca6173bfd1609da3d9878caec`. The retained qualified directory contains 28 files including nine native binaries, compiler runtime notices and original source/signature evidence. The private build-input receipt records its current absolute path; do not substitute an unrelated FFmpeg binary. Hash/notice binding preserves build provenance; model and package runtime qualification remain separate gates.

The collector includes `torch`, `transformers`, `tokenizers`, `safetensors`, `sentencepiece`, `numpy`, `onnxruntime`, `imageio_ffmpeg`, Titanos/core dependencies and full MCP. Metadata closure is retained. Model weight extensions and caches are excluded. The shared imageio-ffmpeg hook omits its bundled executable, and the frozen entry point selects `_internal/native/ffmpeg/ffmpeg.exe` by default. The existing explicit `IMAGEIO_FFMPEG_EXE` override remains configurable; qualification must clear ambient overrides and verify the selected decoder path.

The clean environment must omit torchvision, Diffusers, torchaudio, torchcodec, librosa, soundfile, soxr, Qwen TTS, Coqui TTS, Pillow and SciPy. These are neither added nor silently collected for generic ASR. Titanos supplies decoded mono float32 at the model's sample rate; the inspected Transformers 4.57.x Whisper feature extraction and generation paths use NumPy/Torch, while torchaudio, librosa and similar resamplers are optional paths. A checkpoint requiring extra libraries is outside this narrow profile until explicitly extended and qualified.

Before PyInstaller, a real import gate loads the chat/classifier/Whisper/ASR classes, ONNX Runtime and imageio-ffmpeg, checks CPU-only Torch and the CPU provider, and records the result. It does not instantiate a model, execute a decoder or probe a GPU. `--plan` remains metadata-only and does not run this gate. After build, the shared smoke checks exercise standalone CLI/server startup and both MCP transports with an empty registry. Genuine ASR/VAD/text inference and decoder-path proof against the resulting executable are separate required qualification.

Controlled plan/collector checks: `python tests/check_vendor_text_audio_profile.py`. Existing text, GenAI and graph profile selection is unchanged. This source addition alone is not a completed native build.
