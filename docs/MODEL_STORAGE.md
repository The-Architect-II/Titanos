# Current model tensor storage

`/api/models/details` reports `tensor_storage` for the shared PyTorch loader,
including supported Transformer models and Diffusers pipeline components.
Each component groups its exposed parameters/buffers by current device and
dtype, with tensor/element counts and an offload-hook indicator. No tensor is
moved to CPU, copied or materialized to collect this metadata.

These are storage observations. A CPU or meta tensor may be offloaded to a
different device during computation. Packed or otherwise unexposed weights
are not a complete memory census, and CPU tokenization/scheduling is outside
these module counts. Use execution profiling when making execution claims.

Metadata failures report `complete: false` and an error type, preserving any
other observed components. They do not turn a successful inference into a
failure. Backend-specific details for ONNX/OpenVINO/native runtimes remain
separate. This field is additional metadata, not a placement control.
