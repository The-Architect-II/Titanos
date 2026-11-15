# OpenVINO GenAI image adapter

For a prepared SD/SDXL repository, `titanos pull picture OpenVINO/stable-diffusion-v1-5-fp16-ov --task image --backend openvino-genai` recognizes the component export layout and downloads the complete pinned snapshot. It does not expect an LLM root graph. The loader still checks model-family and optional editing components; pull metadata alone does not qualify inference. Use `titanos add` for an existing complete export. The default image device is automatic; select `CPU` to require CPU execution.

Register `backend: openvino-genai`, `task: image`, `dtype: auto`, and `parallelism: single`. The default `device: auto` uses the same policy as API recommendations: honor an explicit matching installed-runtime device pin, otherwise prefer an available OpenVINO GPU and then CPU. An unavailable pin or a selected GPU's compile failure is an error, with no CPU retry. Explicit `CPU`, `GPU.N` and `OV-` identities retain their meaning. Existing GenAI chat registration still uses the LLM adapter. The `genai-vision` native profile adds Pillow to OpenVINO/GenAI/NumPy and full MCP; model weights remain external.

NPU image execution remains unsupported in this adapter. It requires a separate static reshape/compile path, illustrated in the [official OpenVINO GenAI sample](https://github.com/openvinotoolkit/openvino.genai/blob/2026.3.0.0/samples/python/image_generation/heterogeneous_stable_diffusion.py). Auto selection excludes NPU for images; an explicit NPU identity is rejected before pipeline construction. Implementing and qualifying that path is open work. Raw OpenVINO tensor/embedding graphs and GenAI chat have different device capabilities; their NPU support does not establish image support.

The admitted export has `model_index.json` with `_class_name` exactly `StableDiffusionPipeline` or `StableDiffusionXLPipeline`, a four-channel `unet/config.json`, `scheduler/scheduler_config.json`, and these components:

- `text_encoder`, `unet`, `vae_decoder`: each has `config.json`, `openvino_model.xml`, `openvino_model.bin`.
- `tokenizer`: `openvino_tokenizer.xml` and `.bin`.
- SDXL also requires `text_encoder_2` and `tokenizer_2` in the corresponding layouts.
- Optional `vae_encoder` must have the complete model/config files; its presence enables image-to-image. Without it, generation works and editing returns an explicit unsupported-capability error.

Normal Hugging Face snapshot file links into sibling cache blobs are supported: the identity binds each logical export name and target content hash. Directory links/traversal are rejected. All files are streamed into the export identity and checked again after loading. Keep compile caches outside the immutable export. Inpainting layouts, nine-channel UNets, other pipeline classes and model-family overrides are rejected. No source model is converted or downloaded during a request.

Native pipeline scope is the text encoder(s), tokenizer(s), scheduler, UNet and VAE decoder; image editing also uses the VAE encoder. A retained `safety_checker` or `feature_extractor` declaration/file is not evidence of execution. This adapter does not invoke a separate safety checker or classify output safety. Model details retain the original `requires_safety_checker` value and component declarations alongside that explicit limitation; flags and files are not rewritten or disabled. Native component/kernel execution still requires qualification.

Use the existing `/v1/images/generations`, `/v1/images/edits`, image jobs, CLI image/edit, Python image/edit and MCP generate_image/edit_image interfaces. `size`, `n`, `steps` (or the model-default alias `num_inference_steps`), `seed`, negative prompt, guidance and edit strength retain request → model → existing process/native default precedence. When both request and model omit the seed, a fresh stdlib random32 seed is passed explicitly; model details retain the most recent resolved seed and its source. Seeds use the native C++ generator and do not promise Diffusers pixel identity. Text generation uses full noise (`strength=1`); edit strength is explicit/model-default/process-default. Unsupported knobs are rejected on both routes: JSON `/v1/images/generations` refuses unknown fields with 422, and multipart `/v1/images/edits` refuses undeclared fields (`quality`, `background`, `output_format`, `output_compression`, `input_fidelity`, `stream`, `partial_images`, `user`, ...) with 400 `unsupported_capability` naming them instead of dropping them. Standard image size/count/step policy bounds apply. No full-prompt token-utilization or semantic-quality claim is made.

Input RGB is copied into a uint8 NHWC tensor. Each returned tensor must be uint8 NHWC with the requested dimensions and that native call's image count; invalid tensors fail before publication. Each output image owns copied pixels. For an export with a VAE encoder, `InpaintingPipeline(path, device, **options)` owns the components; `Image2ImagePipeline(owner)` and `Text2ImagePipeline(owner)` reuse them. The native model files are loaded through one owner, not three independent copies. Without a VAE encoder, text generation keeps its direct Text2Image construction. Optional [binary-alpha PNG masks](image-masks.md) use the initialized inpainting owner; no image-to-image → inpainting conversion or fallback is attempted. The same four-channel SD/SDXL exports are supported; specialized nine-channel layouts remain unsupported.

Masked requests use native batch one because the generic four-channel latent blender requires it. Request count `n` is fulfilled serially with the same owner and C++ seeds `(resolved_seed + image_index) modulo 2**32`. `last_generation` discloses `native_batch_size=1` and the `rng_seeds` list. Every result is validated and copied before the next call; cancellation or a later failure closes accumulated images and leaves `completed=false`. Ordinary text generation and unmasked edits keep their original native batching/seed behavior. Per-image native execution and request-level all-or-nothing publication are distinct.

The constructor order avoids a source-identified hazard: current upstream copies existing pipeline helpers when changing pipeline type, while mask helpers are initialized only for an inpainting owner. A native 2026.3.1.0 masked request crashed in the GenAI DLL; the exact crash offset has not been mapped to these source lines. The null-helper explanation is a hypothesis, and this source workaround still requires exact-DLL qualification. The cached 2026.3.1.0 Python stubs expose both owner-to-wrapper overloads. [Upstream helper initialization and batch constraint](https://raw.githubusercontent.com/openvinotoolkit/openvino.genai/master/src/cpp/src/image_generation/diffusion_pipeline.hpp), [upstream pipeline copy constructor](https://raw.githubusercontent.com/openvinotoolkit/openvino.genai/master/src/cpp/src/image_generation/stable_diffusion_pipeline.hpp).

Direct image routes use the existing cancellable request dispatcher, with the configured `TITANOS_CLIENT_TIMEOUT` as the server deadline and caller disconnect/MCP cancellation propagated. Image jobs use the same deadline/run helper plus their existing cancellation event. Decode, model execution and artifact writing share the absolute deadline; output files created by a cancelled/failed write are removed. Native callbacks return a boolean stop request between denoising steps; a stopped/partial tensor is rejected, never published as success. Native compile/encoder/VAE/kernel preemption is not promised.

For direct-process GenAI image jobs, `GET /api/jobs/{id}` exposes a
`native_generation` snapshot while generation is running and retains it in the
terminal job record. The snapshot binds the job ID and a fresh generation
request ID to each native call's entry, progress callbacks, first callback that
observed cancellation, stop request and native return or exception. Masked
batches retain separate native-call records. The callback captures its job
binding before entering the runtime, so a vendor callback thread does not lose
the association. Snapshots contain no prompts, images or latent tensors.

Read the evidence together with the job's execution boundary and result status.
A queued cancellation has no native progress. A callback stop request followed
by native return is different from cancellation first observed after native
return, when an already computed result is discarded. Further callbacks after a
stop request are recorded rather than described as a successful interruption.
Native-call entry records the invocation boundary; it does not prove a GPU
kernel ran. Callback/return observations do not promise immediate kernel
preemption. The [OpenVINO image sample](https://openvinotoolkit.github.io/openvino.genai/docs/samples/python/image_generation/)
documents the callback's boolean stop request.

Model details also retain the latest generation snapshot. Direct synchronous
requests have a local generation ID but no job ID; this is not a new HTTP
request-ID protocol. Multiprocessing replica workers do not inherit the job
observer: their model details can retain local generation evidence, but live
job binding through replica IPC is not qualified here. A snapshot with
`observability_complete=false`, or a job with `native_generation_error`, has
incomplete evidence. Diagnostic publication
failure does not change generation, cancellation or artifact semantics. A true
flag alone is insufficient: final proof also needs native return/exception and
terminal output-disposition fields from the same generation request.

Only the named compile properties are accepted: `INFERENCE_PRECISION_HINT`, `INFERENCE_NUM_THREADS`, `PERFORMANCE_HINT`, `NUM_STREAMS`, `CACHE_DIR`. CPU defaults use the current settings snapshot for threads and f32 inference hint; exported graph precision remains an export concern (`dtype=auto`). Titanos resolves `auto` to one concrete runtime address before construction, rather than passing OpenVINO `AUTO`. MULTI/HETERO routing and hidden device overrides are rejected. Details retain configured and resolved devices, the selection reason and observed runtime identity where available; these fields do not prove kernel execution.

The native callback contract is based on the cached 2026.3.1.0 surface plus official samples. Tests use vendor/image doubles. Qualification is specific to the package, model and operation. A later native Windows 0.5.25 SD1.5 batch on Intel Arc B580 passed one fixed HTTP-MCP n=3 regression, with three PNGs matching retained references (E-007 in the [capability matrix](CAPABILITY_MATRIX.md)). That result does not establish every callback/partial-return path, export family, edit mode, device or broad visual-quality gate. No generic pipeline framework or universal model-family support is implied.
