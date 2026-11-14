# Attributed text and audio translation

Existing text translation and audio translation outputs are unchanged when `include_provenance` is omitted or false, including ordinary JSON, plain audio-translation text and existing verbose audio metadata. Set `include_provenance:true` to receive a typed extension result with `schema_version:"1"`, exact stage text/hashes and actual available provenance. This is an opt-in Titanos extension, not an additional claim about the OpenAI standard.

Text uses `object:"text.translation"` and one NMT stage. Audio uses `object:"audio.translation"` with an ASR stage and, when explicitly configured, an NMT stage. The Python client returns `TextTranslationResult` or `AudioTranslationResult` for this option; existing calls still return their original dictionary/string forms. Plain `response_format:text` cannot carry this JSON extension and is rejected when combined with the option.

```json
{
  "model": "fr-en",
  "text": "Bonjour.",
  "source_language": "fr",
  "target_language": "en",
  "include_provenance": true
}
```

Send that request to `POST /v1/translations`. `POST /v1/audio/translations` accepts the same boolean as a multipart field, alongside `model` (ASR), `translation_model` (optional NMT), `language` (ASR hint), and separate NMT `source_language`/`target_language`. The existing `translation` and `audio_translation` extension operations accept it too. MCP provides `translate` and `translate_audio` typed-request tools; `extension_infer` remains compatible. `/v1/extensions` publishes the optional output schemas and their scope.

```python
result = client.translate("fr-en", "Bonjour.", "en",
                          source_language="fr", include_provenance=True)
print(result.text, result.stages[0].provenance.checkpoint_digest)

result = client.translate_audio("whisper", "french.wav",
    language="fr", translation_model="fr-en",
    source_language="fr", target_language="en", include_provenance=True)
```

```text
titanos translate fr-en "Bonjour." --source-language fr --target-language en --include-provenance
titanos translate-audio whisper french.wav --language fr --translation-model fr-en --source-language fr --target-language en --include-provenance
```

## What the result establishes

For layouts where the existing local fingerprint can be produced, safe single-device Transformers NMT loading hashes checkpoint/config/tokenizer bytes before and after loading, rejects changed or incompatible weights, and inspects actual parameter/buffer devices. The resulting checkpoint digest follows the existing Titanos fingerprint convention. Repository and revision remain explicitly registered identifiers; they are not inferred from the bytes or proof that a remote author published those bytes. Native qualification must bind them to admitted author artifacts separately.

Ordinary response-shape compatibility is distinct from loading validation. Supported verifiable layouts gain the stricter unchanged-bytes, compatible-weights and actual-device checks above. When that fingerprint is unavailable—for example, a valid Hugging Face cache snapshot whose safe weights are symlinks outside the snapshot—ordinary translation retains the prior loader path. Opt-in attribution then returns `unsupported_capability` before generation instead of inventing an identity. The shared sentiment fingerprint is not relaxed to accept that layout.

NMT stages record actual model/tokenizer classes, model type/device, exact input/output hashes, the explicit tokenizer prefix, processed-input hash, untruncated input token count/limit and output sequence length including special tokens. The inference wall timer covers tokenization, generation and decoding; it excludes model loading, queue wait, HTTP and later audio playback. It is not per-operation device profiling, translation confidence, calibration or semantic correctness.

The effective NMT source/target identifiers, registered pair, and loaded tokenizer choices are distinct fields. Text results retain whether the source ID was explicit, inherited from model defaults, or the request's built-in default. Multilingual tokenizers use supported language-code mappings or NLLB-shaped special language tokens, never arbitrary vocabulary words; the selected target token is recorded. Fixed models need an explicit matching registered pair, with loaded source/target tokenizer metadata checked when available. A fixed Marian fr-to-en model uses `fr`/`en` and no forced BOS token. T5-style fixed prompts remain explicit `translation_prefix` configuration.

Audio results keep `requested_asr_language` and `observed_asr_language` separate from `nmt_source_language` and `nmt_target_language`. The NMT input string and hash must exactly match ASR output; no cropping, transcript rewriting or fabricated translation timestamps are introduced. ASR stages retain genuine existing ASR metadata and registered model IDs. They do **not** claim a loaded ASR checkpoint fingerprint that the current backend does not expose. NMT attribution does not add acoustic emotion, diarization or speaker identity.

## Skips, errors and compatibility

An opt-in equal-language text request returns exact input with `translated:false` and a `passthrough` stage; it validates the registered model but does not load it or run NMT. The language IDs' equality is a caller/configuration fact, not language detection or validation of the checkpoint's direction. This opt-in shortcut does not change the legacy backend path.

A configured no-speech gate skips ASR and NMT. Stages retain their requested model IDs and input/output identities, with no invented loaded NMT provenance or token/timing measurements. ASR metadata attributes the skip to the actual VAD measurement. Direct Whisper translation has one ASR stage and no invented NMT model or language IDs.

Malformed or missing completed NMT provenance produces the existing structured `invalid_backend_response` error. Backend failures, cancellation and deadline expiration stay errors; no successful or partial translated result is fabricated. Text HTTP/MCP/extension requests use the existing cancellable dispatcher. Composed ASR/NMT shares one absolute deadline and caller signal, checked between stages and around NMT generation. Cancellation remains cooperative during model work/loading; this source change does not establish native interruption timing.

One existing default needs explicit handling: audio translation defaults its NMT target to `eng_Latn`, while a fixed Marian fr-en tokenizer uses `en`. The source slice does not introduce a broad alias/default rewrite. For that model, send `source_language=fr,target_language=en` and configure the same registered pair. Unsupported or mismatched directions fail rather than being relabeled. The existing `translation_executed` field now reflects NMT's false result for equal-language passthrough instead of implying generation occurred.

`tests/check_translation_provenance.py` uses controlled loaders, tensors and API backends to exercise byte/device binding, fixed/multilingual language IDs, exact ASR-to-NMT text, skip/error/cancel behavior, interfaces and legacy shapes. It does not execute a trained translation model. French-to-English semantics and packaged runtime behavior require their separately admitted native proof.
