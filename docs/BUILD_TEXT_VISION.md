# CPU text and vision bundle

The `text-vision` profile packages CPython, CPU PyTorch, Transformers, Pillow,
HTTP, the CLI and both MCP transports. TrOCR printed-text recognition and YOLOS
object detection use their slow image processors. Model folders remain outside
the application. Other vision architectures need their own dependency and
model qualification; this profile does not advertise every Transformers model.
The [relative-depth adapter](DEPTH.md) also imports Depth Anything, DINOv2 and
the slow DPT processor. Its native model/device qualification remains separate.
The [prompted SAM adapter](SEGMENTATION.md) adds explicit `SamConfig`, `SamModel`,
`SamImageProcessor` (slow) and `SamProcessor` imports, using the same dependencies.
The automatic mask-generation pipeline and TorchVision remain excluded.

Build in a new environment on the target OS and architecture. Install a CPU
PyTorch wheel from the official PyTorch CPU distribution, then this exact source
with its `vision` and `build` extras:

```sh
python -m pip install ".[vision,build]"
python tools/build_vendor_binary.py --runtime text-vision --plan --output /path/to/new/Titanos --work-dir /path/to/build-work
python tools/build_vendor_binary.py --runtime text-vision --zip --output /path/to/new/Titanos --work-dir /path/to/build-work
```

The plan validates dependencies without importing numerical runtimes. The real
build then imports the text, TrOCR and YOLOS classes before compilation, without
loading any weights. It rejects GPU PyTorch and incidental audio, diffusion,
TorchVision or SciPy packages. Use a separate environment: the existing
`text-audio` profile still requires Pillow to be absent.

The resulting directory contains the runtime libraries, source, dependency
inventory, license texts and hashes. Native startup and both MCP transports are
checked during packaging. Users unpack the entire directory and run `titanos`;
they do not install Python or pip. Register complete external `image-to-text`
or `object-detection` checkpoints through the normal model interfaces.

This build profile supplies no FFmpeg or audio pipeline. Printed-line OCR does
not supply document layout or bounding boxes. YOLOS returns its trained labels
and image-coordinate boxes; model quality, duplicates and missed detections
must be assessed on actual inputs after packaging.
