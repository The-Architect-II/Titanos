# DirectML on Windows

DirectML runs supported prepared ONNX graphs on Windows GPUs. Titanos supports
typed tensor requests and sentence embeddings through this backend. A PyTorch or
Diffusers checkpoint needs an appropriate prepared graph; selecting DirectML does
not convert arbitrary model architectures. For the ONNX image adapter on DirectML, a fatal device-removed error is contained as described in [fatal DirectML context containment](FAULT_CONTAINMENT.md).

For a source installation, create a separate environment and install the matching
extra:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install '.[dml]'
# Native ARM64 Python uses .[dml-arm64] instead.
.venv\Scripts\titanos.exe devices --backend onnx --refresh
```

The release installer offers `--profile dml` for Windows x64 and
`--profile dml-arm64` for Windows ARM64. The x64 profile pins
`onnxruntime-directml==1.24.4`; the ARM64 profile pins
`onnxruntime-windowsml==1.25.2.202605110140`. These install into a new environment;
do not install multiple ONNX runtime distributions into one environment because
they own the same Python module. Existing Ryzen AI SDK environments may already
provide both VitisAI and DirectML; install `.[compiled]` there without replacing
the vendor runtime.

Copy the desired adapter's `provider_options` into `gpu.json`, for example:

```json
{"device_id":"1"}
```

Then check and register a prepared model:

```powershell
titanos doctor --backend onnx --device dml --provider-options gpu.json --check
titanos add embed C:\Models\prepared-minilm --task embeddings --backend onnx --device dml --provider-options gpu.json
titanos serve
```

`doctor` executes a small numerical fixture and reports observed node providers.
That check is separate from a trained model's quality and operator compatibility.
Diagnostics use the loaded ONNX module's version, so vendor distribution names
such as DirectML, WindowsML and VitisAI are recognized correctly.

DirectML sessions use sequential execution and disable ORT memory patterns as
required by the provider. CPU EP fallback and Python retry fallback remain off by
default. Software DXGI renderers are rejected for the GPU path. Some ORT versions
still list CPU as an available session provider; actual profile events establish
which provider executed nodes. Dynamic shape calculations can require CPU nodes,
causing a strict session to reject a graph. Preparing fixed shapes can remove that
requirement without changing trained weights. Operator compatibility and graphics
drivers still determine whether a particular graph can run.

Native tests exercised Radeon 840M and Ryzen 7800X3D Radeon graphics, Intel Iris Xe,
NVIDIA RTX 3500 Ada, and ARM64 Adreno with trained classifiers or embeddings.
These are representative checks, not universal GPU/model support or full-dataset
accuracy. See [validation](VALIDATION_0.5.2.md).

The tested ARM64 path used the DirectML DLL shipped inside the WindowsML wheel;
its ordinary CLI execution required no Windows App SDK bootstrap or system
installation. Other WindowsML features can have different prerequisites.

References: [DirectML provider requirements](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html),
[x64 distribution](https://pypi.org/project/onnxruntime-directml/1.24.4/),
[tested ARM64 distribution](https://pypi.org/project/onnxruntime-windowsml/1.25.2.202605110140/).
