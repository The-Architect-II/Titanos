# Relative depth

Register a complete local **relative Depth Anything** Transformers checkpoint
with task `depth-estimation`. The checkpoint supplies its own slow image
processor and embedded DINOv2/DPT architecture; Titanos uses built-in code and
safetensors. External/pretrained-backbone resolution and timm are rejected.
Metric variants and other depth families are rejected because their output
semantics require a separate adapter. Models remain outside the application.

```sh
titanos add scene-depth /path/to/model --task depth-estimation --backend transformers --device cpu
titanos depth scene-depth /path/to/image.png --deadline-ms 30000
```

HTTP `POST /v1/images/depth` accepts `model`, one of the ordinary inline image
fields, optional `worker`, and an inference `timeout` in milliseconds. The same
operation is `depth` at `/v1/extensions/infer`; MCP provides `estimate_depth`
and the discovered extension schema. The Python client exposes
`client.depth(model, image_path, timeout=30000)` and returns a typed `DepthResult`.
The CLI's HTTP `--timeout` remains distinct from its inference `--deadline-ms`.

The result contains one image-aligned **relative inverse-depth** map. Larger
values indicate nearer surfaces; `units` is null and `calibrated` is false.
Values are not metres, probabilities, a point cloud or measurements from a depth
sensor. They are not normalized across images, so numeric magnitudes cannot be
compared across scenes as physical distances.

`data` is base64 of contiguous little-endian float32 in row-major `[height,width]`
order. `value_sha256` binds those exact bytes; `input_rgb_sha256` binds the decoded
input RGB pixels. `model_output_shape` records the model's original map shape.
The checkpoint processor resamples that map to the original decoded image size;
`alignment` declares this operation. No display normalization, clipping,
8-bit quantization, EXIF rotation or fabricated confidence is added by this
adapter. Consumers can separately create a labeled preview from the numeric map.

```python
import base64
import numpy as np

result = client.depth("scene-depth", "image.png")
values = np.frombuffer(base64.b64decode(result.data), dtype="<f4")
values = values.reshape(result.height, result.width)
```

Existing `TITANOS_MAX_INPUT_IMAGE_BYTES`, `TITANOS_MAX_INPUT_IMAGE_PIXELS`,
`TITANOS_MAX_TENSOR_ELEMENTS` and `TITANOS_MAX_BODY_BYTES` bound decoding and
numeric output. Header dimensions and the encoded value budget are checked
before RGB conversion and model dispatch; response metadata is also checked before returning.
Threads, device, dtype and queue/deadline limits use the existing configuration.
Sampling controls have no meaning for this deterministic depth head and are
rejected. Cancellation is checked before/after preprocessing, forward,
postprocessing and encoding; this does not preempt an active device kernel.

The native text/vision import gate includes Depth Anything, DINOv2 and the slow
DPT processor. Source tests use doubles; source support, native packaging,
actual device execution and model-quality qualification remain separate.
Existing binaries do not acquire this feature merely because source changed.

The proposed first small model is the publisher's
[Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).
The [author repository](https://github.com/DepthAnything/Depth-Anything-V2)
distinguishes the small checkpoint's Apache-2.0 terms from the larger variants.
The [author demo](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/app.py)
identifies the relative output as disparity. Titanos follows the installed
[Transformers4.57.6 depth implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/depth_anything/modeling_depth_anything.py)
and its processor contract. These references describe the model family; they
are not a Titanos runtime or depth-accuracy result.
