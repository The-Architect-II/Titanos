# OpenVINO GenAI standalone profile

`tools/build_vendor_binary.py --runtime genai` packages the existing Titanos OpenVINO GenAI text-generation adapter with OpenVINO, its native tokenizers, NumPy, HTTP and MCP. It uses the pinned `genai` dependency extra and the common native collector, source manifest and notice machinery.

Build natively for the target OS and architecture in a clean environment:

```text
python -m pip install ".[genai,build]"
python tools/build_vendor_binary.py --runtime genai --output /new/titanos-genai --work-dir /new/build-cache --zip
```

The complete application directory includes Python and its runtime libraries. Model weights remain external; configured hardware drivers are supplied by the operating system. Titanos uses the text-generation adapter in this profile, without separate PyTorch/Diffusers/media Python packages. Upstream GenAI libraries contain additional pipeline classes that are not new Titanos modality qualifications. The existing `openvino` profile remains the prepared graph/classifier/embedding profile.

Register a prepared GenAI checkpoint with `task: chat`, `backend: openvino-genai`, its external `path`, and an explicitly selected OpenVINO device. The directory needs the model IR and native tokenizer/detokenizer files; a generic Hugging Face checkpoint cannot be used directly. Runtime model loading checks these files. Device indices can change, so identify the physical device from current runtime discovery before assigning a GPU target.

Model defaults and request controls use the ordinary Titanos policy. `/v1/chat/completions`, `/v1/responses` and the MCP `responses` tool call the same GenAI adapter. GenAI handles inference while tokenization and orchestration can use CPU. The package profile itself does not establish a particular model/device's performance or execution placement.

`--without-mcp` retains its explicit sidecar meaning and is unnecessary for a full standalone application. Package startup is followed by actual executable model inference, output-limit and transport checks before a target is considered qualified.

## Lean image generation and editing profile

`genai-vision` adds Pillow to the same pinned OpenVINO GenAI, native tokenizers,
NumPy, HTTP and full MCP runtime. It supplies the prepared SD/SDXL text-to-image
and image-to-image adapter dependencies. Torch, Transformers, Diffusers and
Optimum belong in a separate model-export environment and are excluded from this
serving profile. The existing `genai` text profile and its sidecar option are
unchanged; `genai-vision` requires full MCP.

Use a new clean environment on the target OS and architecture:

```text
python -m pip install ".[genai-vision,build]"
python tools/build_vendor_binary.py --runtime genai-vision --plan --output /new/titanos-genai-vision --work-dir /new/build-cache
python tools/build_vendor_binary.py --runtime genai-vision --output /new/titanos-genai-vision --work-dir /new/build-cache --zip
```

The plan checks installed dependency metadata and rejects incidental Torch,
Transformers, Diffusers, Optimum and audio/video runtimes without importing them.
Before compilation, the build resolves native text, text-to-image and
image-to-image classes plus Pillow; startup and both MCP transports are checked
on the executable. Neither the plan nor these model-free checks qualify image
generation on a device. Prepared model folders remain external; VLM, inpainting,
other image families and model conversion are outside this profile.
