# Masked image editing

`/v1/images/edits` accepts one optional PNG mask with a compatible Diffusers SD/SDXL pipeline or prepared four-channel OpenVINO GenAI SD/SDXL export with a VAE encoder. A mask selects actual inpainting. Unsupported layouts fail; Titanos never silently drops a mask or falls back to an ordinary edit.

The `alpha-binary-v1` contract accepts a single-frame **8-bit RGBA PNG** mask. Alpha **0 (transparent) edits**; alpha **255 (opaque) keeps**. RGB values in the mask are ignored. Other alpha values, grayscale/palette masks, animation and a mask with no editable pixels are rejected. There is no automatic threshold, crop, feathering, rotation or resizing. The source must be opaque, and source and mask dimensions must match exactly. Omitted edit size uses those dimensions; a conflicting explicit size fails. Existing input/output size and byte limits still apply.

Use multipart `mask` for an uploaded file, or `mask_base64` for strict PNG base64 (an optional `data:image/png;base64,` prefix is accepted). They are mutually exclusive. Duplicate source/mask fields and the unsupported `mask_image` alias fail. Neither remote mask URLs nor mask bytes in model defaults are supported.

```python
from titanos.client import Client

with Client() as client:
    result = client.edit("my-image", "source.png", "A blue ceramic cup",
                         mask="mask.png", preserve_unmasked=True, seed=42)
```

```console
titanos edit my-image "A blue ceramic cup" --image source.png --mask mask.png --preserve-unmasked --seed 42
```

The MCP `edit_image` tool advertises `mask_base64` and `preserve_unmasked`; mask bytes use the same HTTP decoder. Studio image models have optional source and mask uploads. With no source selected, the existing image-generation job flow remains available; an edit is a cancellable request rather than a persisted background edit job.

Image count `n` is an optional positive integer on HTTP edits, Python `edit(..., n=2)`, CLI `edit --n 2`, and the flat MCP `edit_image` tool. MCP rejects booleans and nonintegers. The HTTP `ImageRequest` enforces `TITANOS_MAX_GENERATED_IMAGES`; omitted `n` inherits the registered model default, otherwise one image. An excessive request fails rather than being clipped or ignored. This applies to both masked and unmasked edits.

GenAI four-channel masked edits execute the requested count as serial native batch-one calls sharing an initialized inpainting owner. Image `i` uses C++ seed `(seed+i) modulo 2**32`; model details disclose all resolved seeds. No partial batch is published after cancellation or failure. Diffusers and ordinary GenAI text/unmasked batching are unchanged. This adaptation is source-tested, not yet proof of native model quality or successful crash recovery.

`preserve_unmasked` resolves request → model defaults → process `TITANOS_PRESERVE_UNMASKED` (default `true`). Explicit false survives every interface. The CLI exposes `--no-preserve-unmasked`. Supplying the control without a mask is an error. Model defaults must be booleans, and apply to image models only.

When enabled, Titanos copies original **decoded uint8 RGB pixels** into the generated image wherever mask alpha is 255. This is postprocessing, not model fidelity; VAE/latent preservation alone does not guarantee exact pixels. The output may have visible seams. It does not preserve source file bytes, metadata, transparency or color-profile interpretation. With false, Titanos returns the model image without this guarantee.

Only masked responses add a `masking` object: contract version, source/mask byte hashes, exact dimensions, editable/kept pixel counts, preservation value/origin, and `postprocessing` (`copy_unmasked_rgb` or `none`). These are extension fields; strict response consumers must allow them. Existing unmasked responses remain unchanged. Image generation and compositing are distinct: neither successful decoding nor exact outside copying proves the requested edit occurred inside the mask.

This source increment is tested with real PNG parsing and in-process HTTP/MCP, plus GenAI/Diffusers doubles. Native inpainting quality, exact packaged-DLL mask handling and native cancellation responsiveness still require qualification. No new model family, nine-channel export support, checker invocation or runtime quality claim is implied. Retained model safety-checker flags/files remain unchanged; the native adapter does not execute a separate checker merely because its files are present.
