# Qwen3-TTS speech

Titanos can run Qwen3-TTS Base directly through its normal speech API. No HiveMind or PAIR service is used. The current implementation was validated on native Windows with the pinned 1.7B 12Hz Base checkpoint, a local PCM16 reference, BF16 CUDA and SDPA. Other checkpoint/hardware combinations need their own inference qualification.

For a source installation, first install a matching PyTorch and torchaudio runtime, then install `.[qwen3-tts]`. The extra pins the streaming wrapper to `dffdeeq/Qwen3-TTS-streaming` commit `13ef823ffe76aec4ac7e35d3fec28ecfc8d7427a`. The adapter checks its installed commit metadata. A portable binary must include that metadata and the complete numerical runtime; successful startup alone is not speech-inference proof.

Download a complete local Qwen3-TTS Base checkpoint, including the `speech_tokenizer` subdirectory. The tested model is `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, revision `fd4b254389122332181a7c3db7f27e918eec64e3`. Model weights and reference voices remain external to the application. The model card and streaming wrapper use Apache-2.0; a voice reference has its own ownership and permission requirements.

Create a voice mapping such as:

```json
{"my-voice":"/path/to/voices/reference.wav"}
```

Register that mapping using `titanos add` with `--task speech --backend qwen3-tts --voices voices.json`, an explicit local model directory and device. Alternatively, use the local registry or model registration API. The `voices` field maps an exact voice ID to a PCM16 WAV path. A voice is cloned from its x-vector by default: the preparation uses the first five seconds of channel zero at the original sample rate, and Qwen internally resamples the speaker-encoder input. The registry or registration API can also set `voice_texts`, mapping a voice ID to the exact transcript of its reference. Such a voice is cloned in context (ICL), as gim_api TTS Super does it: a reference of at most ten seconds is used whole and mixed to mono, a longer one is cut to the first ten seconds of channel zero. A descriptive voice label is never treated as a reference transcript.

The normal request shape is:

```json
{
  "model":"my-qwen-model",
  "input":"The local speech engine is ready.",
  "voice":"my-voice",
  "language":"en",
  "speed":1.0,
  "seed":101,
  "response_format":"wav"
}
```

Send it to `POST /v1/audio/speech`. Output is a native-rate PCM WAV, or with `"response_format":"pcm"` the bare little-endian PCM16, its format named in `X-Sample-Rate`, `X-Bits-Per-Sample` and `X-Channels`. The adapter does not adjust tempo or pitch, and an unknown voice ID fails rather than substituting another voice. It rejects empty/non-finite or frame-budget-exhausted output and supports cancellation through the engine's job path. One device serves a model; separate replicas can serve independent requests.

With `"stream_format":"audio"` the body is sent as the model makes it. The `wav` form opens with gim_api TTS Super's streaming header (RIFF and data chunk sizes 0x7FFFFFFF, the length being unknown) followed by PCM16 chunks; `pcm` sends the chunks bare. An error before any audio is an ordinary JSON error. A failure after audio was sent aborts the transfer, so a truncated take never looks whole. A client disconnect cancels the synthesis. The model stays busy while a client reads more slowly than the audio is made: 64 chunks are buffered, then synthesis waits until the client reads on or disconnects. `stream_format` `sse` is refused for local models.

These optional request fields override the registry for one request, within gim_api TTS Super's ranges: `temperature` and `subtalker_temperature` (0.05 to 2.0), `top_k` and `subtalker_top_k` (1 to 4096), `top_p` and `subtalker_top_p` (0.05 to 1.0), and `frame_budget`, the codec-frame cap at 12.5 frames a second (16 up to the model's `max_frames`; without it the cap is estimated from the text length). A value out of range is refused before inference. Other speech backends refuse these fields by name. Speech jobs (`POST /api/jobs`) take the same fields, but they always write a WAV file, so `stream_format` and `pcm` are refused there.

`POST /api/models/details` records the model and wrapper revision, parameter placement, voice mode, reference sample hash, speaker embedding hash, generation seed, duration and peak level. Registry defaults can set `reference_seconds`, `minimum_frames`, `frames_per_character`, `max_frames`, `emit_every_frames` and `decode_window_frames`. The tested defaults are 5 (10 for an ICL voice), 240, 3.0, 7500, 8 and 80 respectively. `first_emit_frames` makes the first chunk smaller than `emit_every_frames` for an earlier first sound, and `overlap_samples` with `seam_join` (`continuity`, gim_api's join, or `qwen`, the upstream crossfade) smooths chunk seams; unset, they keep steady chunks with no overlap. The sampling defaults (`do_sample`, `temperature`, `top_k`, `top_p` and their `subtalker_` forms) and `seed` can be set there too. Keep text in short complete sentences for dependable narration and compare generated speech with independent transcription before selecting it.

The Windows Qwen binary profile collects the source-built native audio overlay selected on the build import path. Pass `--native-audio-dir` containing verified `libsndfile-minimal` and `soxr-built` provenance trees to `tools/build_binary.py`, together with the existing `--ffmpeg-dir`. It validates their manifests, binds the actual imported libraries, copies corresponding source/build recipes and notices, then verifies the packaged libraries. A different native audio stack needs an equivalent documented build and qualification.

The optional Gradio dependency belongs to the upstream Qwen demo CLI. The frozen Titanos build excludes that CLI and records the `qwen-tts → gradio` dependency exception in `BUILD.json` and `DEPENDENCIES.json`. Titanos supplies its own browser, HTTP and MCP interfaces. The 12Hz inference path does not require an external SoX program; the upstream package also imports support for a separate tokenizer path that can log an unused SoX warning.

No bundled private voice references, embeddings or model weights are required or included. Source-built libsndfile and soxr retain their LGPL and component notices. The supplied libraries remain separately replaceable, with their exact native source and rebuild instructions. GPU runtime redistribution terms remain separate from the Titanos source license.
