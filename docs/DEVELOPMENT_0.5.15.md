# 0.5.15 development changes

> **Historical record (Titanos 0.5.15).** This page is kept for provenance and is not updated. It describes that version's source and recorded results only; current behaviour is documented in [masked image editing](image-masks.md) and [OpenVINO GenAI image adapter](GENAI_IMAGES.md). Results recorded here do not qualify the current source or any later package.

Image edits now accept `n` through HTTP, MCP and the CLI, with the same configured
limits and model-default inheritance as generation. The 0.5.14 edit interfaces
did not expose that request field; its recorded one-image test used a model
default and is not evidence of request-count parity.

The OpenVINO GenAI image adapter now constructs `InpaintingPipeline` directly
before deriving text and ordinary edit wrappers from its shared components.
The previous conversion from an image-to-image owner copied uninitialized mask
helpers in the vendor-reported source and the first native masked request ended
in an access violation. The old failure remains separate from successor testing.

Generic four-channel inpainting uses native batch size 1. Titanos fulfills `n`
with sequential calls, using `(seed + image_index) modulo 2**32`; model details
record `native_batch_size` and `rng_seeds`. Every output is shape/dtype checked
and copied before the next call. Cancellation or failure closes all accumulated
outputs and returns no completed partial batch. Cleanup errors remain observable
without replacing an earlier generation failure.

Ordinary text-to-image and unmasked image-to-image keep their existing batching.
Models, vendor DLLs and numerical dependencies are unchanged. This is a source
workaround and an interface repair; native recovery, image usefulness and runtime
cancellation responsiveness require their own evidence.
