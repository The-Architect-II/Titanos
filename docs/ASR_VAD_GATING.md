# Configured VAD before transcription

An ASR model can generate text on nonspeech input. The retained 0.5.5 robustness attempt demonstrated this on both silence and seeded noise. Its raw outputs and failed gates are unchanged. This optional pipeline checks a registered VAD model **before** ASR; it does not filter or rewrite an already produced transcript.

Set `vad_model` to an explicit registered `task="vad"` model such as a locally configured Silero ONNX checkpoint. When the detector returns no speech regions, the service can skip ASR and return empty text. When any region is detected, ASR receives the full original audio bytes. No cropping, concatenation or timestamp rewriting occurs. VAD can miss quiet speech; its existing detector knobs remain configurable and any request can disable the gate.

```python
client.transcribe("whisper", "recording.wav", vad_model="silero", response_format="verbose_json")
client.transcribe("whisper", "recording.wav", vad_enabled=False)
client.transcribe("whisper", "recording.wav", vad_model="silero",
                  vad_options={"threshold": 0.4, "min_silence_ms": 200})
```

```console
titanos transcribe whisper recording.wav --vad-model silero --response-format verbose_json
titanos transcribe whisper recording.wav --no-vad
```

`--vad` explicitly enables the configured model; `--no-vad` explicitly disables it. The existing CLI `--options` JSON file can supply `vad_options`. The same fields are available through transcription/translation multipart forms, the JSON `audio_transcription` / `audio_translation` extensions, Python client methods and MCP. Multipart `vad_options` is a JSON object encoded as a string; the client and MCP adapter encode it automatically. Studio offers model/server default, explicit disable, or a registered VAD model and requests verbose results so skip attribution is visible.

## Defaults and precedence

| Field | Resolution |
| --- | --- |
| `vad_model` | Non-null request → ASR model `defaults.vad_model` → `TITANOS_ASR_VAD_MODEL` |
| `vad_enabled` | Non-null request → ASR model `defaults.vad_enabled` → `TITANOS_ASR_VAD_MODE` |
| `vad_options` | Explicit request keys override ASR model `defaults.vad_options`; omitted detector keys use the selected VAD model's normal model/process defaults |

The process model default is empty (unset). Process mode defaults to `auto`, which enables gating only when a model is configured. `on` requires a configured model; `off` disables gating unless overridden by a higher-priority `vad_enabled`. An explicitly enabled gate without a model is an error. JSON `null` and omitted fields inherit; boolean `false` is preserved as an explicit disable. The model and enable flag resolve independently, so an explicit inherited `off` policy requires `vad_enabled=true` to override.

For example, set `TITANOS_ASR_VAD_MODEL=silero`, or register an ASR model with:

```json
{"defaults":{"vad_model":"silero","vad_options":{"threshold":0.4,"min_silence_ms":200}}}
```

No detector is chosen from the catalog automatically and no model is downloaded. An unknown model, wrong task, recursive ASR-as-VAD selection, invalid option or detector failure returns a genuine error. There is no fallback to ungated ASR when a configured check fails. The gate fields are removed before calling Transformers, so they cannot leak into library generation arguments.

`vad_options` supports `sample_rate`, `threshold`, `min_speech_ms`, `min_silence_ms`, `speech_pad_ms`, and an optional `worker`. The ASR request's `worker` applies only to ASR; detector routing uses its own option or normal registered-model routing. Frames are not returned by the composed gate, keeping its provenance bounded. This service-level composition uses the existing model/worker lifecycle; the underlying backend transcription primitive remains a single-model operation.

## Results, timing and cancellation

JSON and plain-text responses retain their established shapes, including `{"text":""}` or empty text for a skipped ASR stage. Opt into `verbose_json` for `vad_gate` containing the selected model, selection provenance, decision, `asr_executed`, complete VAD measurement/provenance and `audio_policy="full_original_audio"`.

A skip has `backend=null`, `language=null`, `segments=null`, `words=null`, unavailable ASR quality fields and `duration_source="vad_decoded_audio"`. It does not claim that Whisper measured confidence, language, timestamps or duration. A positive gate preserves the ASR backend's actual text/timing fields, adds the detector provenance and labels duration as `asr_decoded_audio`. A VAD speech score remains uncalibrated and is not an ASR confidence score.

One absolute deadline and caller-cancellation signal cover VAD, ASR and an optional subsequent NMT translation. Queue time counts. Cancellation or expiry after the detector prevents ASR from starting. Audio translation also uses the gate: a no-speech result skips ASR and NMT, and verbose output attributes that decision. Translated-word timestamps remain unavailable; no source transcript alignment is reassigned to translated text.

Controlled source checks cover skip/no-ASR invocation, complete-byte pass-through, independent worker selection, defaults/disable, cancellation/deadline boundaries, error propagation, provenance, translation skipping, library-keyword filtering and interface dispatch. The new pipeline still requires separately authorized real-model and packaged-runtime qualification. It does not retroactively turn the earlier nonspeech hallucinations into passes.
