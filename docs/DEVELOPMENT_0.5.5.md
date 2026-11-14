# 0.5.5 development changes

> **Historical record (Titanos 0.5.5).** This page is kept for provenance and is not updated. It describes that version's source and recorded results only; current behaviour is documented in [text sentiment](SENTIMENT.md), [text retrieval](RETRIEVAL.md), [audio transcription and speech activity](AUDIO.md) and [standalone CPU text and audio profile](BUILD_TEXT_AUDIO.md). Results recorded here do not qualify the current source or any later package.

This development line combines structured language inference, sentiment composition, retrieval and audio operations through the existing standalone engine. Source support and a version number do not qualify every checkpoint or hardware target. Consult the supplied package's build, dependency and inference records for its exact scope.

## Language and retrieval

The sentiment API can return categorical labels without inventing numeric confidence. A declared native llama.cpp backend can constrain generation with a JSON schema. The classifier/LLM composition retains both stage results and selects a final decision only when its agreement policy passes. Both stages share the caller's deadline and cancellation. See [sentiment](SENTIMENT.md).

The new trained cross-encoder reranker jointly scores each query/document pair, preserves original document indices, and returns raw relevance logits with optional top-N selection. Embeddings and reranking reject overlength input by default. Explicit truncation records original and processed token counts and preserves the original input identity. See [retrieval](RETRIEVAL.md).

## Audio

Transcription keeps its existing JSON and plain-text responses. `verbose_json` adds decoded duration and available backend word or segment alignments. Unavailable language or quality measurements remain explicit. Silero-compatible ONNX VAD supplies speech intervals, configurable thresholds/timing and per-request state reset. See [audio](AUDIO.md).

The [CPU text/audio build profile](BUILD_TEXT_AUDIO.md) bundles CPython, CPU Torch, Transformers, ONNX Runtime, a source-bound FFmpeg decoder and full MCP. Models remain external. GGUF execution uses the separately pinned [managed native runtime](NATIVE.md). Users of native bundles do not need Python or pip installed.

## Existing callers

- Set `truncate:true` in a request, model default or the documented process setting to permit input cropping. Omitting it now rejects overlength input instead of silently dropping text.
- Strict clients for the old 0.5.4 sentiment schema must upgrade before using expanded categorical or composition results. Legacy weight-vector decoding remains an explicit option.
- Transcription defaults to `{"text":"..."}`. Timestamp/metadata extensions require `verbose_json`; the optional strict hosted-provider SDK schema is not claimed when it requires measurements the backend does not provide.
- Native context, slots, threads, batch/ubatch and model-file controls remain configurable. A model's physical context limit still applies.

HTTP, the Python client, CLI and MCP expose the new operations through the same model registry. Existing release bundles and their qualification records retain their original scope.
