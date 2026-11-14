# InstructPix2Pix image editing

Titanos recognizes a registered Diffusers image checkpoint whose `model_index.json` declares `StableDiffusionInstructPix2PixPipeline`. It loads that builtin class directly and requires its eight-channel instruction-trained UNet. This is a separate family from SD/SDXL image-to-image and inpainting; it follows a text instruction with a source image. Source support does not establish better editing quality.

The pinned Diffusers 0.35.2 pipeline accepts `image_guidance_scale`, ordinary text `guidance_scale`, negative prompt, steps, count and generator. Its call does not implement strength, masks, height or width. Titanos therefore rejects masks, strength and text-only generation for this family. It retains source dimensions; a supplied size must match exactly, and dimensions must fit the configured image policy and VAE scale. No implicit resize is performed. [Pinned pipeline source](https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py).

```python
from titanos.client import Client

with Client() as engine:
    result = engine.edit("instruction-artist", "source.png", "Make the cup blue",
                         image_guidance_scale=1.5, guidance_scale=7.5, steps=25, seed=42)
```

```console
titanos edit instruction-artist "Make the cup blue" --image source.png --image-guidance-scale 1.5 --steps 25 --seed 42
```

Multipart `/v1/images/edits`, Python `Client.edit`, CLI `edit` and MCP `edit_image` expose `image_guidance_scale`. It must be finite and at least one. Resolution is request → model default → `TITANOS_DEFAULT_IMAGE_GUIDANCE_SCALE` (1.5). Other standard controls preserve their existing request/model/process or native defaults. Image guidance is active when text `guidance_scale > 1`; the library's ordinary conditional-guidance behavior is retained. The new control is rejected for other image families rather than silently ignored.

Instruction-edit HTTP requests reject a supplied `strength` key even when blank, and reject supplied `width`/`height` fields. Resize the source image before editing if another geometry is needed. MCP reserves unsupported width/height aliases so values cannot be silently discarded; those aliases are not advertised as supported fields. Existing families retain their documented `size` control.

Model defaults accept only count, steps/num_inference_steps, seed, negative prompt, text/image guidance and response format. In this family, resize, strength, masks and pixel-preservation defaults are unsupported. The input geometry supplies the implicit output size. There is no promise that any source pixels remain unchanged.

The candidate is [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) at revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`. When using its separately prepared FP16 component files, set model `load_options` to `{"variant":"fp16"}`. This selects local files and is distinct from the requested execution dtype. Model components/weights remain external. The adapter does not download them, follow the publisher example's `safety_checker=None`, or rewrite model-index flags. Declared safety-checker and feature-extractor components are retained; their normal pipeline behavior remains in effect.

This slice permits builtin component identities, safetensors and one selected device. Loader options are limited to a simple local variant suffix and a boolean memory-loading option. It does not add custom pipelines, LoRA/IP-adapter controls, sharding, a generic image framework or dependencies beyond the existing Diffusers/Transformers/PyTorch/Pillow stack.

The existing shared request deadline/cancellation event reaches the denoising callback. Returned images must have the requested count and source-aligned RGB dimensions. Cancellation or invalid output triggers best-effort closure of every owned output, preserving the primary error and recording cleanup failures. Shared artifact publication/cleanup remains in the server.

Source tests use real PNGs and actual HTTP/client/CLI/MCP dispatch with a strict pipeline double that cannot accept ignored strength/size arguments. They cover family/control rejection, variant/component retention, guidance precedence and output cleanup. Native execution, hardware fit and asset quality require a new declared roster; previous failed SD inpainting cases remain failures and were not retried or tuned here.
