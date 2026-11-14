# Standalone CPU text bundle

The `text` build profile freezes Titanos, PyTorch, Transformers, tokenizers and MCP into one application folder. Model weights stay in external directories. Users of a completed native bundle do not need to install Python or these libraries. Keep the whole extracted folder together; the executable uses its bundled native libraries.

Select a CPU PyTorch wheel in a clean build environment to produce a CPU bundle. The profile name alone does not select the PyTorch wheel. The private Windows x64 candidate uses Python 3.12, `torch==2.8.0+cpu`, `transformers==4.57.6`, `mcp==2.2.0` and `pyinstaller==6.22.2`. Build natively for the target OS and architecture; this candidate does not qualify other platforms or GPUs.

After installing the CPU wheel from PyTorch's official CPU index, install the source's `[sentiment,build]` extras with constraints retaining that wheel and Transformers version. Then run:

```text
python tools/build_vendor_binary.py --runtime text --output ./candidate-cpu-text --work-dir ./build-work --zip
```

Choose a new output directory. The builder includes MCP by default, records source/dependency identity and third-party notices, verifies startup, and creates a model-free archive. Actual model inference and interface qualification are separate from successful compilation; retain those receipts with a delivered candidate.

For a dedicated classifier, follow [Text sentiment](SENTIMENT.md), point its registry entry to a compatible local checkpoint and set `device: cpu`. Start the extracted application with:

```text
titanos.exe --config models.json serve
titanos.exe sentiment polarity "The repair worked well." --min-score 0.7
```

The same server exposes the browser, HTTP API and MCP interfaces. Existing compatible text-generation and embedding adapters share the included text libraries, but each architecture still needs a valid checkpoint and its own qualification. This bundle does not include the separate Diffusers, speech/audio or image/video dependency groups. Model licenses, attribution and conversion records travel with the external model assets.
