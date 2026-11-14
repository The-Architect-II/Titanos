# Whisper transcription context

Transcription accepts an optional nonblank `prompt` with vocabulary or context. It conditions builtin Whisper generation through `WhisperProcessor.get_prompt_ids(..., return_tensors="pt")` and native `prompt_ids`. Titanos never inserts the context into the returned transcript or replaces recognized words.

```console
titanos transcribe my-whisper recording.wav --prompt "Project names: Titanos and Vega." --max-tokens 128
```

```python
from titanos.client import Client

with Client() as engine:
    result = engine.transcribe("my-whisper", "recording.wav",
                               prompt="Project names: Titanos and Vega.", max_tokens=128)
```

The same `prompt` is available on multipart `/v1/audio/transcriptions`, the `audio_transcription` extension, and the flat MCP `transcribe` tool. It may also be a transcription model default; a supplied request overrides that default. When both omit it, generation arguments and transcript formatting retain their existing behavior. The API rejects repeated/empty prompt fields, special-token text rejected by the Whisper tokenizer, unsupported ASR families, and context exceeding `TITANOS_MAX_PROMPT_CHARS` or the loaded decoder's token budget.

Whisper's previous-text window is derived from `max_target_positions // 2 - 1` (223 text tokens for a 448-position decoder). Titanos rejects overlong context before generation instead of allowing library truncation. `max_tokens` remains `max_new_tokens`; it is never silently reduced. The context IDs, special prefix and requested new tokens must fit the decoder. Long audio and static caches conservatively reserve the full previous-token window when previous-token conditioning is enabled. These checks can reject a requested budget even if a particular segment would finish early.

Prefix budgeting calls the builtin Whisper prefix helper with a request-local copy of the ASR pipeline's effective generation config and a CPU-only language-width stand-in. The pipeline's derived beam/language defaults and timestamp fallback are preserved; explicit per-call generation options still take precedence. Previous-token history uses the native precedence: a supplied non-None generation option, otherwise the effective config value, otherwise false. This performs no language detection or neural forward. Synthetic IDs are used only for counting and never sent to the model. A missing previous-text sentinel is filled in the request copy from the tokenizer's first prompt ID. Both shared pipeline and model configs remain unchanged. Context IDs are moved to the model device for generation.

This control uses native `first-segment` prompt conditioning. Conflicting low-level decoder-prefix overrides, replacement generation configs and `all-segments` prompt conditioning are rejected rather than ignored. Native long-form previous-segment behavior otherwise remains unchanged. No claim is made that supplied context persists verbatim in every later segment.

Verbose results add `prompt_conditioning` only when conditioning actually ran: context hash/character and token counts, source, native context limit, conservative decoder-prefix/previous-token budgets and available new tokens. These are capacity measurements, not token usage or recognition confidence. Plain JSON/text still returns the pipeline's original text. No confidence or language measurement is invented.

A configured no-speech VAD result still skips ASR before model loading or tokenization. Supplying context does not turn silence into text, force Whisper to load after a skip, or bypass the shared cancellation/deadline checks. Context on another ASR architecture is rejected if that ASR stage executes.

The implementation uses the existing Whisper processor, tokenizer and generation code; no new dependency is required. The relevant prompt/prefix/capacity helper ASTs match between the inspected cached Transformers 4.57.3 and 4.57.6 packages. This establishes source compatibility, not native execution or improved recognition. Earlier failed recognition takes and their gates remain valid evidence; a prompted take needs separate qualification.
