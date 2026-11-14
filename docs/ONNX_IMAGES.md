# Standalone ONNX image generation

`onnx-diffusers` runs the builtin Diffusers SD1 ONNX pipeline through Titanos's
existing image-generation HTTP, CLI, Python and MCP interfaces. The checkpoint
directory is external to the application. It requires the exported text encoder,
UNet, VAE decoder, safety checker, CLIP tokenizer/image processor and PNDM
scheduler. It does not implement image editing, SDXL, custom pipeline code,
automatic model conversion or automatic device selection. A fatal DirectML device-removed error is contained as described in [fatal DirectML context containment](FAULT_CONTAINMENT.md).

The first candidate is the publisher's
[CompVis SD 1.4 ONNX export](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/da5601014c467b382fcf42019ba920a903f2103e).
Keep that revision's complete files together, including `unet/weights.pb`.
The old `StableDiffusionOnnxPipeline` index name maps to the explicitly selected
builtin `OnnxStableDiffusionPipeline`; the model never selects imported code.
Keep the model's CreativeML OpenRAIL-M terms separate from Titanos's Apache-2.0
source license. Models are not included in the application archive.

For DirectML, obtain the adapter index and LUID from `titanos devices --backend onnx --refresh`.
Register the exact values in a model entry:

```json
{
  "path": "D:/Models/sd14-onnx",
  "task": "image",
  "backend": "onnx-diffusers",
  "device": "dml",
  "load_options": {
    "device_id": 2,
    "adapter_luid": "00000000:12345678",
    "profile_dir": "<lab-records>/ort-profiles"
  },
  "allow_cpu_fallback": true,
  "defaults": {"size": "512x512", "steps": 20, "guidance_scale": 7.5}
}
```

Those adapter values are examples, not the lab's current device binding. Both
values must match live DXGI enumeration. Software adapters are rejected. A
changed LUID requires a fresh deliberate registration. `allow_cpu_fallback`
defaults to false; explicitly enabling it permits unsupported ORT operators on
CPU. Python retry to another provider remains disabled. The native safety
checker is always invoked for this family.

Agents can discover the same values with MCP `list_devices(backend="onnx")`
and call `register_model` with `payload={"name": "my-image", "spec": ...}`.
HTTP `/api/models/register` accepts the same registration. Include only
`device_id` and `adapter_luid` in `load_options` for these public interfaces;
`profile_dir` is a trusted local configuration setting. Registration saves
selectors without loading a model. A later load verifies the live binding.
Studio can generate from registered ONNX image models and hides unsupported
source-image/mask editing. Its Add model form does not yet select this backend;
use the API, MCP or CLI to register it.

Requests accept `prompt`, `negative_prompt`, `size`, `n`, `steps`, `seed`,
`guidance_scale` and `response_format`. Normal `TITANOS_MAX_*` policies apply;
request controls override model defaults. `TITANOS_THREADS` sets the existing
ORT session thread count. A nonempty negative prompt requires guidance above
one, because this pipeline otherwise ignores it. Precision comes from the
export, not a runtime dtype override.

GPU execution must be established separately. Details report the selected
adapter and each component's active providers, CPU fallback policy and CPU
support work. NumPy noise/latents, tokenization, image preprocessing and the
Torch PNDM scheduler run on CPU. When `profile_dir` is set, each component's
first attempted inference after loading produces a separate ORT profile and
operator/provider summary, with a file hash. Later requests do not reuse that
profile as fresh proof. Retain the profiles alongside owned-process GPU-engine
counters matching the selected adapter's LUID. A provider name, red image or
annotation alone is not proof of AMD execution.

`load_options.disable_metacommands` is an optional **boolean** for `device=dml`.
It is accepted by local configuration and safe HTTP/MCP registration; strings,
integers, null and use with CPU are rejected. Omission sends no provider flag
and retains existing behavior. Explicit `true`/`false` is mapped internally to
the exact ONNX Runtime provider strings `"true"`/`"false"`; details expose the
requested provider options. Profile output paths remain restricted to trusted
local configuration.

This is a software diagnostic, **not a proven fix**. The matching
[ONNX Runtime1.24.4 source](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/providers/dml/dml_provider_factory.cc)
parses this flag and disables DirectML metacommands when true. This changes the
operator implementation choice without changing model weights, prompts, drivers
or Windows timeout policy. A failed GPU/runtime must be independently admitted
before any new test; adding this control does not qualify hardware or authorize
device recovery.

Cancellation is checked before/after each synchronous component call and at
every denoising callback. An in-flight ORT call is not preempted. Cancelled or
invalid outputs are closed without publishing a completed image. Model unload
releases the pipeline and sessions. Profile files remain in the explicitly
configured operator directory for evidence retention.

## Native application package

Use a fresh native Windows x64 environment. Install the CPU PyTorch 2.8 wheel,
then this source's `dml-vision`, `compiled` and `build` dependencies. The profile
pins Diffusers 0.35.2 and ONNX Runtime DirectML 1.24.4. It excludes CUDA/ROCm Torch
and unrelated audio/vision backends. Do not install another ONNX Runtime wheel
over the DirectML distribution.

```powershell
python tools/build_vendor_binary.py --runtime dml-vision --output D:/Build/Titanos-dml-vision --work-dir D:/Build/work --zip
```

The native archive contains the interpreter, runtime libraries, full MCP,
source, dependency inventory and notices. Existing DirectML 1.15.4 native-byte
provenance and Microsoft redistributable terms are retained and checked;
DirectML's binary license is distinct from ONNX Runtime's MIT wrapper license.
Hardware drivers remain installed by the operating system. Native startup and
MCP smokes do not qualify a model or GPU.

The implementation uses the exact
[Diffusers 0.35.2 ONNX pipeline API](https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py)
and existing Titanos sessions configured according to
[ONNX Runtime DirectML requirements](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).
Source tests use controlled doubles and generated solid-color PNGs; they are
not inference or image-quality evidence.
