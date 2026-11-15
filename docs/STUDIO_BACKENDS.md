# Studio backend controls

The registration form offers every backend declared by the registry while
preserving its trust boundary. HTTP-safe combinations are selectable:
Transformers; Diffusers; Qwen3-TTS; Sherpa ONNX TTS; ONNX; OpenVINO; OpenVINO GenAI; Tesseract;
and ONNX-Diffusers. Default selects Diffusers for image/video, ONNX for
tensor/VAD, and Transformers otherwise. Explicit choices restrict the task
list to combinations accepted by `ModelSpec`.

Manual llama.cpp folder registration remains visible but CLI-only because it
requires trusted native GGUF/runtime options. The task-first setup can safely
select a pinned Hugging Face GGUF and server-owned CPU runtime profile, then
register only validated relative model/projector names. XTTS remains visible
but CLI-only because its legacy checkpoint needs
an explicit trusted-pickle decision. Studio does not send either as an
incomplete HTTP registration.

Compiled backends expose an optional relative graph name. Qwen3-TTS accepts a
voice-name to absolute-WAV JSON object. Tesseract selects image-to-text on CPU
and expects a folder containing LSTM `.traineddata` files. ONNX-Diffusers
accepts CPU directly or binds DirectML to a current non-software adapter index
and lowercase DXGI LUID from the cached ONNX inventory. **Load known devices**
does not refresh hardware or qualify execution; use `titanos devices --backend
... --refresh` first when necessary. The full workflow and task matrix are in
[Studio usage](STUDIO_USAGE.md).

Sherpa ONNX TTS uses a complete acquired VITS/Piper archive, CPU placement and
the separately packaged pinned `sherpa-onnx==1.13.8` /
`sherpa-onnx-core==1.13.8` runtime. Its GitHub release source, asset digest,
model license and runtime/native licenses remain separate provenance records.

ACE-Step 1.5 music uses the `ace-step` backend with a complete provider-created
core plus pinned XL bundle and a separately packaged, hash-bound isolated
runtime receipt. It supports only the controls declared by that exact bundle
and publishes channel-preserving WAV asset jobs. Source availability is not
song-quality or target-device qualification; see [Music](MUSIC.md).

Device suggestions remain backend/task-specific. For OpenVINO GenAI image
exports, a singleton unpinned `GPU` inventory address is presented as the
validator-supported `GPU.0` alias; CPU and GPU vendor UUIDs remain available,
while NPU rows are omitted because that image adapter accepts only CPU or one
explicit GPU. OpenVINO chat and generic compiled models retain their broader
runtime selectors.

For a Diffusers image/video checkpoint that requires a weight suffix, enter it in **Weight variant (optional)**. For example, the prepared FP16-only InstructPix2Pix folder requires `fp16`, serialized as `spec.load_options.variant`. This selects weight filenames; it does not select compute precision. Leave it empty for the existing loader defaults. Studio never guesses from filenames and hides/disables the control for other backends/tasks; retained input cannot leak into those registrations. The UI, HTTP registration endpoint and instruction adapter accept nonempty ASCII letters, numbers, underscores and hyphens, such as `fp16`, `non_ema` or `fp16-test`; paths, dots and whitespace are rejected. HTTP permits only this loader option for Diffusers image/video models. Custom pipeline, component and other loader overrides still require trusted local configuration, and existing builtin model restrictions still apply.

Studio reads `/api/info`, `/v1/models` and the existing read-only
`/api/models/capabilities` operation to display configured identity and select
controls. It does not infer model families from model names or filenames, and
does not call model-details/load endpoints for discovery. The Runtime panel is
available only after an explicit load or a catalog-reported loaded state. If
required capability information is missing or cannot be refreshed, the
affected action is blocked or described as unavailable rather than sent to a
different route.

For **Tesseract**, choose an image and optionally set the language, page mode and geometry output. Empty controls inherit model/server defaults; explicitly choosing text without geometry sends `include_geometry=false`. The prompt and token-sampling controls are hidden and excluded from the request. Returned text is shown directly, with native OCR metadata/geometry available in an expandable detail view. Native confidence is not presented as calibrated probability.

API/MCP registration accepts Tesseract's validated OCR model defaults: `language`, `psm`, `dpi`, `preserve_interword_spaces`, `char_whitelist`, `char_blacklist`, `include_geometry` and `max_output_bytes`. For example, `defaults: {"psm": 7, "include_geometry": false}` persists through registry reload. These backend-specific defaults do not enable OCR options on other backends or token-generation defaults on Tesseract. The Studio registration form itself leaves these defaults unchanged; its OCR request controls can override them per call.

For **InstructPix2Pix**, choose a source image and describe the change in the Instruction field. Available settings are text guidance, image guidance, steps, image count and seed. The mask and pixel-preservation controls are hidden/disabled; size and strength are not offered. Stale values from a previous SD/SDXL request cannot enter the instruction-edit request. A missing source is rejected even if the form handler is called directly.

Ordinary SD/SDXL models retain their existing generation or source-image/mask editing flow and size/steps/seed controls. Chat, VAD selection, model selection, history, image-job ownership and Stop behavior keep their prior paths. Requests hold a snapshot of their model capability profile and input files through response handling and cleanup; late refreshes cannot replace a newer profile or release another request's Stop control.

The source checks execute the actual JavaScript with
DOM/fetch/deferred-response doubles. They cover the complete safe registration
matrix, exact serialized fields, default/false handling, capability failures,
misleading model names, explicit lifecycle, task routing, rendering and
Refresh/Stop ownership. They do not establish live-browser rendering, native
backend execution or model quality; those remain separate validation steps.
