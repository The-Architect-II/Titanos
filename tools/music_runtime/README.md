# ACE music runtime

`ace_adapter.py` is Titanos's one-request adapter for the pinned ACE-Step 1.5
XL-SFT recipe. This implementation and its model-free tests do not establish
successful native generation, performance or listening quality. Those require
separate validation with the packaged runtime and real selected model bundle.

## Packaging and launch

The adapter runs in an isolated Python 3.11+ runtime, separate from Titanos's
serving Python dependencies. Package it as the adapter named by
`runtimes/ace-step-1.5/runtime.json`, with relative executable/adapter paths and
a complete payload size/SHA-256 inventory. Declare `supported_variants` as
`["xl-sft"]`; this adapter does not support the Base or Turbo variants.
Retain the Python, native dependency and ACE source licenses and inventories.
Model weights and their notices remain in the separate model bundle.

Titanos supplies its isolated bootstrap, device masks, offline environment and
v1 request contract before model frameworks load:

```text
python -I -B <bootstrap> ace_adapter.py --request request.json --output output.wav --result result.json --progress progress.json
```

See [the parent contract](../../docs/MUSIC_RUNTIME_CONTRACT.md) for the exact
bootstrap, environment and receipt requirements. Output paths must be fresh.
The process handles one request and exits; the parent owns process termination
for cancellation. Progress contains actual stage names without guessed
percentages or a cooperative-cancellation claim.

## Runtime dependencies and model bundle

Install the official `acestep` package from commit
`ca1e85fe9430179831e6bc6be790c332190a3866`. The adapter checks 211 embedded hashes
of installed source/resources before importing ACE. Runtime packaging must
preserve these bytes, including `genres_vocab.txt`.

The selected dependency baseline is Torch 2.10, Transformers 4.57.6 and
Diffusers 0.37.1, with matching torchaudio and an appropriate Torch platform
build. The inference import closure also requires Accelerate, NumPy, einops,
vector-quantize-pytorch/einx, Loguru, pytorch-wavelets/PyWavelets, SoundFile and
their resolved dependencies. Build a platform-specific locked runtime and
retain its dependency inventory; this list is not a replacement for that lock
or proof of platform qualification. Do not merge this stack into an older
serving environment. Gradio, ModelScope, nano-vLLM and training dependencies
are not needed by the selected adapter path.

The model bundle uses the following directories:

```text
bundle/titanos-music.json
bundle/core/Qwen3-Embedding-0.6B/...
bundle/core/vae/...
bundle/dit/... including silence_latent.pt
bundle/planner/...
```

The manifest binds each component inventory and revision. Inventories bind
every payload file's size and SHA-256. Component pins are:

| Component | Revision |
| --- | --- |
| Core VAE and text encoder | `19671f406d603126926c1b7e2adc169acbcade22` |
| XL-SFT DiT | `d06de46b4622f781cf07f4a013a67d591ca52819` |
| 4B planner | `0a3ec94b557aea7d508da38b31cfe7341f6ff737` |

The adapter requires exact payload membership; download caches or other
unlisted files are rejected. Symlinks/reparse points and checkpoint executable
code are rejected. Explicit installed model classes load local safetensors.
The sole `.pt` exception is the inventoried silence tensor, loaded with
`weights_only=True` and checked for shape and finiteness. Stock service
initialization, checkpoint source copying, remote code and implicit downloads
are not part of the adapter path.

## Generation and evidence

The PT 4B planner executes first and is released before DiT, text encoder and
VAE loading. By default (`thinking` true, every `use_cot_*` flag false) it
writes audio codes from the request metadata, and audited ACE conditioning
converts them to precomputed 25 Hz hints. A request or a registry default may
change that route. `use_cot_metas` runs the planner's metadata pass, whose BPM,
key and time signature fill the values the request leaves out.
`use_cot_caption` (with `use_cot_metas`) has the DiT encode the planner's
caption instead of the prompt. `use_cot_language` (with `use_cot_metas`)
changes only what the metadata pass writes. `thinking` false (with
`use_cot_metas`) sends the DiT no planner codes, so no hints are built. The
parent and the child refuse every other combination. The receipt records the
flags, the planner's parsed plan (`planner_plan`) and what the DiT received
(`dit_conditioning`).

Components move to the selected device for execution and return to CPU between
stages. Chunked VAE decoding keeps its output buffer on CPU and does not fall
back to another execution device. Real forward hooks observe model parameter
device/dtype; merely entering a stage is not execution proof.

CPU requests require float32. NVIDIA CUDA accepts float32, float16 and BF16;
auto chooses BF16 only when the child observes support, otherwise float32.
Explicit CUDA requests cannot fall back to CPU. Other accelerator families are
unsupported by this initial adapter. These paths still require native
qualification on each delivered platform.

Instrumental mode supplies ACE's `[Instrumental]` lyric marker. Song mode
preserves supplied lyrics; lyric rewriting is disabled, and the caption is
rewritten only when a request sets `use_cot_caption`. Text-encoder
conditioning beyond the trained token window is rejected. Reference audio,
LoRA, editing, batching and alternative model variants are unsupported.

The v1 request contract carries the configured `max_steps`, `max_prompt_chars`
and `max_text_chars` values. The parent checks them before launch, including
direct Engine use, and the child checks them independently. Operators may raise
or lower each bound through the corresponding `TITANOS_MAX_*` setting; the
runner adds no hidden 1,000-step or fixed text-character cap. Its default of 50
steps must also fit the configured maximum. Larger character budgets do not
override the trained token window, and no input is silently truncated. The
trusted parent's private request JSON has no separate fixed byte ceiling.

Output is bounded stereo 48 kHz PCM16 WAV. The adapter checks finite/nonzero
samples after conversion, duration/sample/byte limits and complete file
readback. It removes only bounded model duration padding and records any peak
scale-down. Receipts bind the output bytes, source/bundle identity, visible GPU
UUIDs and actual execution placement. Successful serialization does not prove
musical quality or human acceptance.

## Portable checks

From the repository root, with Python 3.11+ and NumPy installed:

```text
python -B tests/check_ace_runner.py
```

These tests use fake framework/device observations, inert tiny manifests and
synthetic PCM. They exercise request bounds, visibility, placement and the
actual parent receipt validator, source/inventory tamper detection, lyric mode
mapping, independently configured step/text limits and WAV serialization.
They do not import ACE/Torch, load models,
query devices or use the network. The separate pinned-upstream source audit is
not required to run this portable CI check.
