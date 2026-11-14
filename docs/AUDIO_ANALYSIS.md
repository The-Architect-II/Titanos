# Anonymous diarization and composed audio analysis

Titanos can return anonymous speaker intervals by themselves or combine them with transcription and optional transcript-text sentiment. The two operations use registered local models; they do not identify a person, infer personality or vocal emotion, assign calibrated speaker confidence, or invent a separate transcript for overlapping speakers.

This implementation has focused inert/in-process tests. Those tests establish API and lifecycle semantics, not native diarization accuracy, speaker-count quality, model licensing clearance or end-to-end hardware qualification.

## Models and controls

Register the reviewed external diarization bundle with `task="diarization"`, `backend="sherpa-onnx"`, `device="cpu"`, `dtype="auto"` and ordinary single-device execution. The backend validates the local bundle against Titanos's trusted diarization catalog. Checkpoint downloads, arbitrary checkpoint Python and automatic fallback are not part of these inference requests. Model weights stay external.

Four controls use **request > model defaults > `TITANOS_DIARIZATION_*` settings**. Omission or null inherits; explicit zero is preserved where valid.

| Control | Meaning |
|---|---|
| `num_speakers` | Integer zero selects automatic clustering; positive values must fit the configured speaker limit. |
| `cluster_threshold` | Finite nonnegative native clustering threshold; **not a probability or confidence score**. |
| `min_duration_on` | Nonnegative minimum active-speaker interval duration, in seconds. |
| `min_duration_off` | Nonnegative interval-gap control, in seconds. |

The reviewed Sherpa 1.13.8 implementation bypasses clustering for audio of **ten seconds or less**. For those clips, the API rejects a nonzero speaker-count control or a clustering threshold other than `0.5`, rather than silently ignoring it. When effective controls need this check, a bounded decode validates duration before any model loads. Duration controls and speaker/chunk limits remain subject to configured resource bounds.

## HTTP

`POST /v1/audio/diarizations` accepts JSON:

```json
{
  "model": "speakers",
  "audio_base64": "<base64 of the complete audio file>",
  "timeout": 30000,
  "num_speakers": 0,
  "cluster_threshold": 0.5,
  "min_duration_on": 0.3,
  "min_duration_off": 0.5
}
```

An optional `worker` selects an existing matching assignment. `timeout` is the server inference deadline in milliseconds, including queue wait. Audio is supplied inline as base64 bytes or an `audio/*;base64` data URL; server file paths and remote audio URLs are not accepted.

The typed result includes `model`, `task="diarization"`, `sample_rate=16000`, `duration`, `decoded_samples`, original encoded-byte `input_sha256`, `speakers`, `segments`, `overlaps`, effective `parameters` and backend `provenance`.

- Speaker IDs are `speaker_0`, `speaker_1`, etc., assigned anew within each request. Equal IDs in separate requests do not identify the same person.
- Each segment has `speaker`, `start`, `end`, `start_sample` and `end_sample`. Coordinates describe half-open intervals of the decoded 16 kHz mono audio. Simultaneous segments remain separate.
- Each overlap has the same coordinates and a `speakers` array. It is the exact partition where at least two distinct labels are active, not a guessed confidence or an overwritten single-speaker timeline.
- The result contract checks sample/second consistency, bounds, anonymous label membership and exact overlap intersections. Extra identity/confidence fields are rejected.

`POST /v1/audio/analysis` composes registered stages:

```json
{
  "model": "asr",
  "diarization_model": "speakers",
  "audio_base64": "<base64 of the complete audio file>",
  "timeout": 30000,
  "timestamp_granularities": ["segment"],
  "vad_enabled": false,
  "diarization_options": {"num_speakers": 0},
  "text_analysis_model": "text-sentiment"
}
```

`model` is the ASR model. Optional `worker`, `language`, `prompt`, `max_tokens` and a single `timestamp_granularities` value (`segment` or `word`) retain the transcription contract. The composition always requests verbose ASR metadata; omitted timestamp granularity uses the backend's supported default. A vocabulary/context prompt conditions ASR and never replaces its output. Existing `vad_enabled`, `vad_model` and `vad_options` retain request/model/process precedence.

Use `diarization_worker` and `diarization_options` for the speaker stage. Optional `text_analysis_model`, `text_analysis_worker` and `text_analysis_options` select an existing dedicated sentiment classifier or supported local chat model. Text options include the existing score/margin, classifier input-limit or LLM generation/prompt/decoding controls; incompatible classifier/LLM controls are refused. `asr_uncertain` defaults to true for transcript analysis and remains an explicit uncertainty declaration, not measured ASR confidence.

All selected models, task roles, worker assignments and static effective controls are validated before the first model loads. Then the workflow runs optional configured VAD, transcription, diarization and optional transcript sentiment. Stages receive the **same full original audio bytes**, never VAD-trimmed fragments. Existing decoders may run separately: diarization resamples to 16 kHz mono, while ASR/VAD use their selected rates. One byte/duration/sample policy and one absolute deadline cover the workflow; decoded durations must agree within one sample at the coarser rate. There is no claim that only one media decode occurs.

The result contains `task="audio_analysis"`, the unchanged `text`, full `transcription` metadata, `diarization`, `speaker_alignment`, `speaker_attribution_status`, `speaker_segments` and optional typed `text_analysis`:

- `speaker_segments` preserves each backend ASR segment or word, including its original ID, text and timestamps. Each row adds `granularity`, intersecting `speakers`, `overlap` and `attribution`.
- `overlap=true` means that a genuine simultaneous-speaker interval intersects the ASR unit. Two sequential speakers within one broad ASR segment do not by themselves set it true.
- Missing backend timestamps remain null. Those units have no invented speaker labels and `attribution="timestamps_unavailable"`. If no alignment is returned at all, the array is empty and `speaker_alignment="unavailable"`.
- Finite, ordered ASR timestamps extending outside the decoded audio remain exactly as returned by ASR, but that unit receives no speaker labels, `overlap=false` and `attribution="timestamps_outside_audio"`. The API does not clamp, stretch or relabel those timestamps, and valid units remain attributed. Negative, nonfinite, nonnumeric or reversed timestamps still fail.
- `speaker_alignment` preserves the original segment/word granularity. Separately, `speaker_attribution_status` is `complete` when all returned units have usable in-range times, `partial` when only some do, and `unavailable` when none do (including no alignment). This is timestamp usability, not speaker/transcription confidence. Original mixed-audio text and optional transcript-text sentiment remain available when attribution is partial or unavailable.
- This is interval attribution of an existing mixed-audio transcript. It does not separate voices or manufacture per-speaker words during simultaneous speech.
- `text_analysis_scope="transcript_text_only"` explicitly distinguishes text sentiment from tone, personality or audio affect. Empty transcripts skip the requested text stage with `text_analysis_status="not_run_empty_transcript"`; they do not trigger a substitute generation.

Cancellation and disconnects propagate one shared signal; every stage checks the original deadline before and after work. Late results cannot be returned as success or start a later stage. Native Sherpa segmentation/clustering has cooperative-cancellation blind spots: cancellation may wait until an embedding callback or native return. No kernel preemption, instantaneous stop or hard wall-clock termination is promised.

## SDK, CLI and MCP

```python
from titanos.client import Client

with Client() as client:
    speakers = client.diarize("speakers", "meeting.wav", timeout=30000)
    analysis = client.analyze_audio(
        "asr", "meeting.wav", diarization_model="speakers",
        timestamp_granularities=["segment"], text_analysis_model="text-sentiment",
        timeout=30000,
    )
```

```text
titanos diarize speakers meeting.wav --deadline-ms 30000 --num-speakers 0
titanos analyze-audio asr meeting.wav --diarization-model speakers --timestamp-granularities segment --text-analysis-model text-sentiment --deadline-ms 30000
```

Both commands accept `--options` for the exact API JSON controls. Audio analysis also accepts `--diarization-options` and `--text-analysis-options` JSON files; explicit diarization flags override the corresponding JSON values. `--no-vad` preserves an explicit gate disable. CLI output is the same typed JSON, not a reformatted or fabricated transcript.

MCP exposes typed `diarize(payload)` and `analyze_audio(payload)` tools. They call these same HTTP contracts and return the same validated result types. `/v1/extensions`, `titanos://extensions` and typed `/v1/extensions/infer` discovery include operations `diarization` and `audio_analysis` and their schemas. These tools may load their explicitly selected local models; they are not passive inventory operations.

Existing deployed packages do not acquire this feature merely because source tests passed. Packaging, native qualification and deployment must bind the eventual source and runtime/model inputs explicitly.
