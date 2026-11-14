# Prompted SAM masks

Register a complete local builtin Transformers SAM checkpoint (`model_type: sam`) as `image-segmentation`. This adapter accepts one image with **one box OR one point set**, including a 1/0 positive/negative label for every point. It does not run an automatic mask generator, infer class names, combine different prompts or choose a supposedly best mask.

```console
titanos add my-sam D:/Models/sam --task image-segmentation --device cpu
titanos segment my-sam source.png --box 40 30 180 200
titanos segment my-sam source.png --points points.json --point-labels labels.json --multimask
```

`points.json` contains e.g. `[[60.5,80],[100,120]]`; `labels.json` contains `[1,0]`. All coordinates refer to the decoded original RGB raster, with x to the right and y downward. Points require `0 <= x < width`, `0 <= y < height`. A box is `[x0,y0,x1,y1]`, with ordered corners and its upper edge allowed at width/height. Coordinates are never silently clipped, reordered or converted to a different public frame. Image decoding does not apply EXIF orientation.

POST `/v1/images/segment` accepts the common inline `ImageInput` fields, prompt fields `box` or `points` plus `point_labels`, optional `multimask` and finite `mask_threshold`, `worker`, and the existing `timeout` in milliseconds. `/v1/extensions/infer` uses `operation: "segmentation"`. Typed MCP `segment_image` accepts the same request as its `payload`.

```python
from titanos.client import Client

with Client() as engine:
    masks = engine.segment("my-sam", "source.png", box=[40,30,180,200])
    first = masks.masks[0]
    edited = engine.edit("artist", "source.png", "A blue ceramic cup",
                         mask_base64=first.png_base64, preserve_unmasked=True)
```

Each mask is a lossless RGBA PNG with **alpha 0 for selected/edit pixels and 255 for keep pixels**, directly matching [image editing](image-masks.md). An empty native mask is retained with `selected_pixels=0`; a subsequent edit will reject it as having no editable region. Segmentation itself does not copy or modify source pixels.

Single-mask mode is the default and returns SAM's dedicated single mask token (index 0). `multimask=true` returns the model's native alternatives in their original order, with output index and native mask-token index. It does not sort, average or select by score. Native predicted IoU is reported as a finite, **uncalibrated** score, without probability clamps or invented semantic labels. It does not prove the mask is correct, sufficiently precise for a requested edit, or covers every relevant object.

SAM's slow builtin processor resizes the longest side, pads to its configured square and scales prompts internally. The adapter verifies those scales, preserves the original prompt, then uses the processor's padding removal and bilinear postprocessing back to original dimensions. CPU float32 postprocessed logits are compared with `mask_threshold` to produce binary masks. The result reports this alignment, original geometry/RGB-byte hash, threshold, native output shape, model/processor classes, device and declared repository/revision. Serving validates each PNG's binary alpha, hash, dimensions and pixel count; the lightweight typed client checks structure/header/hash without requiring Pillow.

Controls resolve request → model defaults → process settings. Model defaults accept only `multimask` and `mask_threshold`. Process defaults are `TITANOS_SEGMENTATION_MULTIMASK=false`, `TITANOS_SEGMENTATION_MASK_THRESHOLD=0`, `TITANOS_SEGMENTATION_MAX_POINTS=64` and `TITANOS_SEGMENTATION_MAX_MASKS=3`. The latter caps requested output count, not the model's built-in number of alternatives: cap 1 permits a single-mask request from a three-alternative model, while its multimask request is rejected before preprocessing/forward. Alternatives are never silently truncated. Existing input-byte/pixel, tensor-element and response-body limits apply; header checks use a conservative configured mask-count and uncompressed RGBA/base64 budget before avoidable model work. `--deadline-ms` maps to the existing shared deadline; cancellation checks bracket decode, preprocessing, model execution, output host transfers, alignment and encoding. Neural-forward preemption is not promised.

Checkpoint loading uses local safetensors, direct `SamConfig`, `SamModel`, `SamImageProcessor` and `SamProcessor` classes. Custom code, external/nested backbone resolution, configuration/weight replacement, sharding and non-Transformers backends are rejected. Only builtin eager/SDPA attention and a boolean memory-loading option are accepted. Raw and parsed top-level/nested attention selectors, including private storage aliases, are checked before model loading; `trust_remote_code=false` alone does not prohibit Hub-kernel selectors. SAM's prompt encoder, vision encoder and processor must use coherent image/patch dimensions.

The CPU text/vision profile adds explicit SAM model and slow-processor import checks using its existing Torch, Transformers, NumPy and Pillow dependencies. It does not add TorchVision or the automatic-mask-generation pipeline. Cached native-package source was inspected at Transformers 4.57.6. [Official SAM guide](https://huggingface.co/docs/transformers/en/model_doc/sam), [4.57.6 processor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/sam/processing_sam.py), [4.57.6 mask alignment](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/sam/image_processing_sam.py).

Source tests use real PNGs, HTTP/extension/CLI/client/MCP requests, and SAM-shaped processor/model doubles. They establish coordinate, score, mask and lifecycle contracts; native checkpoint execution, device support and segmentation/edit quality require separate qualification.
