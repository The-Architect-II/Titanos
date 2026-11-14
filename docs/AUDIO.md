# Audio transcription and speech activity

Titanos runs these models locally, with the same registry, worker isolation, lifecycle, authentication and cancellation machinery as other tasks. VAD measures speech activity; it does not identify a person, infer emotion or establish transcript correctness. Related pages: [Whisper transcription context](WHISPER_CONTEXT.md), [configured VAD before transcription](ASR_VAD_GATING.md), [anonymous diarization and composed audio analysis](AUDIO_ANALYSIS.md) and [Qwen3-TTS speech](QWEN_TTS.md).

## Transcription metadata

`POST /v1/audio/transcriptions` still defaults to `response_format=json`, returning only `{"text":"..."}`. `response_format=text` still returns plain text. Opt into `verbose_json` to receive decoded duration, sample rate/count, backend language when available, timestamps and quality availability. `json`, `text` and `verbose_json` are the supported formats; SRT/VTT are not implemented.

Use multipart `timestamp_granularities[]=segment` or `timestamp_granularities[]=word` with `verbose_json`. Select one granularity per request. Whisper supports segment or word alignments; CTC supports word alignments. The other sequence-to-sequence pipelines currently provide neither. An explicit unsupported request fails before forward; omitted granularity chooses segment for Whisper, word for CTC and unavailable otherwise. This is backend alignment, not independently verified timing. A missing start/end remains `null`; it is never filled from neighboring text or recording duration.

Verbose fields include:

| Field | Meaning |
| --- | --- |
| `duration`, `sample_rate`, `decoded_samples` | Duration and coordinates of decoded, resampled mono audio |
| `language`, `language_source` | Backend language label if supplied consistently; otherwise `null` / `unavailable` |
| `requested_language` | Caller hint, kept separate from backend detection |
| `segments` / `words` | Backend text and start/end seconds at the selected granularity; unselected/unavailable list is `null` |
| `timestamps` | Selected granularity, provenance and supported choices |
| `quality` | `confidence`, `avg_logprob`, `no_speech_prob` are `null`, with explicit unavailable status/reason |

No confidence, log probability, speaker label or language is estimated from the transcript. Quality fields remain unavailable because the current Transformers pipeline does not expose those measurements. Null quality is not a favorable or unfavorable quality score.

The ordinary OpenAI SDK supports the existing JSON/text responses and can retain verbose extensions. Its optional strict verbose-response validator can require fields or finite endpoints that this pipeline does not supply; that stricter contract is not claimed. Use the advertised Titanos schema and handle unavailable measurements explicitly.

```python
client.transcribe("whisper", "recording.wav")
client.transcribe("whisper", "recording.wav", response_format="verbose_json",
                  timestamp_granularities=["word"], timeout=30000)
```

```console
titanos transcribe whisper recording.wav --response-format verbose_json --timestamp-granularities segment --deadline-ms 30000
```

The MCP `transcribe` tool exposes the same options. `audio_transcription` is also available through `/v1/extensions/infer` with inline `audio_base64`. The multipart/API deadline is in milliseconds; the Python client's HTTP timeout remains seconds. Cancellation is checked during decoding, before and after pipeline inference, and between generated tokens on supported sequence-to-sequence pipelines. A CTC forward is cooperative at its boundary and cannot be interrupted inside a single framework operation.

Audio translation keeps its existing text/JSON formats and adds `verbose_json` for duration and explicit unavailable alignment/quality. Translation segment/word timestamps are not supported: neither direct translation nor an ASR-plus-NMT composition assigns source transcript timestamps to translated words. `/v1/audio/translations` always labels its output language English, as before; optional source-language metadata is separate.

## Standalone VAD

Install `titanos-engine[vad]` for ONNX Runtime CPU, NumPy and the existing FFmpeg decoder; Torch and Transformers are not required by this adapter. Keep the model external to the engine binary. Register a local folder with `task="vad"`, `backend="onnx"`, `device="cpu"` and `graph="silero_vad.onnx"` (the graph filename default). Only the Silero-compatible input/state/sample-rate graph ABI is supported. Loader overrides, remote checkpoint code, pickle, model sharding and other device providers are rejected.

The admitted reference is [Silero VAD v6.2.1](https://github.com/snakers4/silero-vad/tree/7e30209a3e901f9842f81b225f3e93d8199902b1), MIT, with `src/silero_vad/data/silero_vad.onnx` at Git blob `80c5592ef1f4c9ede3e357bbd02eb863358a6a9d`, 2,327,524 bytes. Its acquired SHA-256 is `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`. Titanos implements the published ONNX wrapper's frame/context/state ABI without importing its Torch wrapper. The implementation does not claim numerical equivalence to every upstream postprocessing option.

`POST /v1/audio/vad` accepts a model name and inline `audio_base64` (or an audio data URL), plus these options. Precedence is explicit non-null request, model defaults, then process settings.

| Request/model default | Process setting | Default |
| --- | --- | --- |
| `sample_rate` | `TITANOS_VAD_SAMPLE_RATE` | 16000; 8000 is also supported |
| `threshold` | `TITANOS_VAD_THRESHOLD` | 0.5, strictly between 0 and 1 |
| `min_speech_ms` | `TITANOS_VAD_MIN_SPEECH_MS` | 250 |
| `min_silence_ms` | `TITANOS_VAD_MIN_SILENCE_MS` | 100 |
| `speech_pad_ms` | `TITANOS_VAD_SPEECH_PAD_MS` | 30 |
| `return_frames` (request only) | — | false |

Audio is decoded to mono at the selected rate. One frame is 512 samples at 16 kHz or 256 at 8 kHz (32 ms), prefixed by 64 or 32 samples of prior context. Recurrent float32 state has shape `[2,1,128]`. Every request starts with zero state and zero context. State is never shared between recordings, including after cancellation, error or a rate change. The last frame is zero-padded for inference; returned coordinates stop at the actual decoded end and optional frames disclose `padded_samples`.

Speech begins at a frame start when its score reaches `threshold`. Silence begins when a score is at or below `max(0, threshold - 0.15)`; speech ends at that silence start once `min_silence_ms` has elapsed. A score reaching the speech threshold cancels pending silence. Intervals shorter than `min_speech_ms` are dropped before padding; surviving intervals are padded, clipped to the decoded recording, and overlapping padded intervals are merged. Durations are converted to sample counts with ceiling rounding. Boundaries are quantized to 32 ms frames, not arbitrary subframe estimates. Unfinished speech at EOF ends at the actual recording end.

The response contains ordered, nonoverlapping `segments` with both seconds and exact decoded sample indices, model/checkpoint/device provenance, source byte SHA-256, frame/context widths, effective parameters and `state_reset="per_request"`. Optional frames contain the model's sigmoid speech score labeled `speech_probability_uncalibrated`; no recording-level confidence is invented.

```python
result = client.vad("silero", "recording.wav", threshold=0.6, return_frames=True)
for segment in result.segments:
    print(segment.start, segment.end)
```

```console
titanos vad silero recording.wav --threshold 0.6 --min-silence-ms 150 --speech-pad-ms 30 --return-frames --deadline-ms 30000
```

The typed MCP `vad` tool and extension operation `vad` use the same JSON request. `/api/info`, OpenAPI and model capabilities expose the operation. `TITANOS_MAX_BODY_BYTES`, `TITANOS_MAX_AUDIO_SECONDS`, `TITANOS_MAX_AUDIO_SAMPLES`, `TITANOS_VAD_MAX_FRAMES` (default 20000) and the request deadline bound work and output. All decoded audio is checked before the first VAD forward; oversize input is rejected, not silently cut. FFmpeg accepts piped input only and cannot fetch nested network/file resources. Cancellation during decode kills and reaps its owned decoder; VAD checks cancellation before and after every frame.

Controlled source tests use deterministic doubles and validate format preservation, alignment availability, frame/state/reset behavior, interval boundaries, limits and cancellation. Actual pretrained VAD/ASR accuracy, target-device operation and native packaging require separate qualified runtime evidence. No new trained audio model result is claimed by source tests.
