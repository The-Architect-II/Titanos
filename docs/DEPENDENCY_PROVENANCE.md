# Dependency and distribution boundaries

Titanos source retains the repository's Apache-2.0 `LICENSE` and existing `NOTICE`. This does not replace the licenses of dependencies, native runtimes, checkpoints or voice references. The copyright and attribution notices should travel with the source without being silently rewritten.

The source dependency contract is `pyproject.toml`. It includes bounded version ranges, exact runtime/build pins and the Qwen streaming wrapper's exact Git revision. It is not a complete lock of every possible optional runtime. Each compiled profile needs its own actual dependency closure and native component inventory.

The existing builders produce `DEPENDENCIES.json`, `sbom.cdx.json`, `THIRD_PARTY_LICENSES`, `THIRD_PARTY_NOTICES.md`, `NATIVE_COMPONENTS.json`, `BUILD.json`, a source tree and a file manifest. Those are provenance and inventory artifacts. The code deliberately records `license_review: not_performed`; a successful build does not establish that every selected binary may be redistributed under the same terms.

| Component boundary | What must remain distinct |
| --- | --- |
| PyTorch | The upstream source license specifies redistribution conditions and notice retention; bundled third-party libraries have their own terms. See the [PyTorch 2.8.0 license](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE). |
| PyInstaller | The pinned documentation describes a packaging exception, continued dependency-license requirements, and separate terms for modified PyInstaller source. See the [6.22.2 license documentation](https://pyinstaller.org/en/v6.22.2/license.html). |
| FFmpeg and native audio | Configuration changes can change the applicable license. Match the exact binary to corresponding source, build recipe and notices; the existing minimal FFmpeg/audio collectors retain these bindings. See [FFmpeg's license and distribution guidance](https://ffmpeg.org/legal.html). |
| GPU/vendor runtimes | A wrapper's open-source license does not cover every bundled vendor DLL. CUDA redistribution is limited to designated components and conditions; DirectML and Qualcomm components are separately identified by the builders. See [NVIDIA's SDK agreement](https://docs.nvidia.com/cuda/eula/index.html) and the exact notices accompanying each selected native package. |
| Models and voices | Keep weights, personal voice references, embeddings and caches outside the source repository and application binaries. Their rights and licenses are separate from Titanos. |

The retained upstream notice files under `tools/licenses/` have corresponding provenance records. Preserve both the text and its version/hash/source association. Before releasing a compiled bundle, inspect its actual inventory for missing notices, dependency exceptions, vendor redistributables and corresponding native source. This document is not legal certification or clearance for a particular binary release.

The upstream links describe those projects' terms and policies. They do not substitute for the exact licenses and notices accompanying the versions actually selected for a bundle.

A source-only repository can contain the build recipes and license provenance without uploading compiled vendor payloads. Creating a private repository does not by itself authorize publishing a binary release or later making the repository public.
