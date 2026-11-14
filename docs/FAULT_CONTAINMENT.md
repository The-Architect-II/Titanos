# Fatal DirectML context containment

A native DirectML failure previously left the engine's runtime marked loaded.
The observed0x887A0006 failure was followed by an unresponsive unload. This
change prevents reuse of that failed context and keeps ownership visible;
it does not repair the GPU, driver or model.

The ONNX image adapter recognizes the native DML removed/hung/reset HRESULTs
0x887A0005,0x887A0006 and0x887A0007 in an ONNX Runtime DML error. Ordinary input
errors and allocation failures retain their previous behavior. The first fatal
error is retained with its component, process ID and configured adapter. A
subsequent inference attempt returns `device_unavailable` without another
native call. Provider metadata is cached during healthy loading so later status
inspection does not query the failed native sessions.

With the default local engine, the failed runtime reference is retained and
reported as unavailable and resident. The loaded-model inventory therefore
does not claim those resources were freed. Explicit load, model switching and
unload cannot silently reuse or destroy that context; they return503 instead.
The owning process must be restarted after independent device re-admission.
There is no automatic process creation, driver reset, device re-enable or
timeout-policy change. The adapter keeps the existing first-error profile
collection; it does not claim that every native error or hang is preemptible.

For applications requiring process containment, use the existing **explicit**
worker configuration. An already registered DML image model can be assigned:

```json
{
  "workers": {
    "image-worker": {
      "models": ["my-image"],
      "device": "dml",
      "threads": 4
    }
  }
}
```

The model's DXGI index/LUID remains in its model configuration. No CUDA GPU
assignment is added. Unassigned models keep the ordinary local residency policy.

After a fatal worker reply, its existing parent stops that owned worker using
bounded process cleanup and marks the worker unavailable. The request is never
replayed. New requests may use other healthy, already configured workers, but
cannot restart or reuse this failed worker object. Operator removal/recreation
of the explicit worker is required after re-admission. An explicit unload can
retry process cleanup; it does not clear the fatal marker.

If termination does not stop the process, the worker retains its process
reference, PID and last acknowledged model. Status reports running or unknown,
the stop error, and unconfirmed residency; it never reports stopped by clearing
the handle. Read-only details remain available. Closing the engine attempts all
worker cleanups, preserving the first error and additional failures. This
reports the tracked worker process; it does not invent proof about untracked
descendants or global device health.

PID observation can fail independently of liveness, for example when a
multiprocessing observer object has been closed. In that case `pid` is null,
`pid_verified` is false and `last_known_pid` retains the historical ID for
reporting only. PID-based tree termination is skipped unless the current owned
process is confirmed alive and its current PID getter succeeds. Cached IDs are
never termination targets; bounded methods on the retained process object may
still be attempted.

The controls are covered by model/vendor doubles and blocked-process/pipe
tests. They are source hardening, not native qualification. The original failed
request and its no-image verdict remain unchanged. The separate metacommand
diagnostic is not part of this patch.

[Microsoft's DXGI error definitions](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/dxgi-error)
describe these device-failure codes; their presence does not isolate the model,
runtime or driver as the initiating cause.

## Limit boundaries

Every configurable request limit is a typed refusal at a fixed place, never a silent clip. `tests/check_limit_boundaries.py`
lowers each limit to a small declared value, sends the just-under value (accepted and delivered whole to a recording
inert runtime) and the just-over value (refused before the model runs: at the schema, in the Guard, or inside the
adapter's own decode/validation step, `tensor_payload` or `media.read_audio`, whose `ValueError` `Engine.infer` converts
to 400), in-process and without a model, device, server listener or child process (the FFmpeg decode is an in-process
stand-in; the real decode is covered by `tests/check_media.py` and `tests/check_audio_decode.py`). Rows name the
setting (`TITANOS_` plus the upper-cased name), the refusal, where it happens when that is not the schema, and the test
method; `validator-level` and `unit-level` mark rows proven by a direct call rather than an HTTP exchange.

| Limit | Setting | Just-over refusal | Test |
| --- | --- | --- | --- |
| Request body with Content-Length | `max_body_bytes` | 413 `request_too_large` before any body byte is received, `Connection: close` | `test_body_bytes_declared_length` |
| Request body without Content-Length | `max_body_bytes` | 413 `request_too_large` once the accumulated chunks cross the limit | `test_body_bytes_chunked_accumulation` |
| JSON nesting depth | fixed 256 (`titanos.json_depth.JSON_MAX_DEPTH`) | 400 `invalid_request_error` from the Guard before any route parses the body, on `request.json()` and pydantic-bound routes alike | `test_json_nesting_beyond_limit_is_refused_by_the_guard`, `test_json_depth_scanner` |
| Inline base64 media | `max_body_bytes` (`max_input_image_bytes` for images) | 413 `request_too_large` | `test_inline_audio_bytes_unit` |
| Generation token ceiling | `max_tokens` | 422 `max_tokens exceeds TITANOS_MAX_TOKENS=N` (also `max_completion_tokens`, `max_output_tokens`) | `test_max_tokens_ceiling`, `test_embedding_max_tokens_is_bounded_by_max_tokens_only` |
| Prompt length | `max_prompt_chars` | 422 `prompt exceeds TITANOS_MAX_PROMPT_CHARS=N` | `test_prompt_chars` |
| Text length (message, text part, instructions, embedding text) | `max_text_chars` | 422 typed message | `test_text_chars_message_textpart_instructions`, `test_embedding_batch_and_text` |
| Speech input length | `max_speech_chars` | 422 `input exceeds configured text limit` | `test_speech_chars` |
| Chat message count | `max_messages` | 422 `messages exceeds TITANOS_MAX_MESSAGES=N` | `test_messages_count` |
| Tools and assistant tool calls | `max_tools` | 422 | `test_tools_and_tool_calls` |
| Stop sequences | `max_stop_sequences`, `max_stop_chars` | 422 | `test_stop_sequences_and_chars` |
| Images per chat, image data URL length | `max_images`, `max_input_image_bytes` | 422 | `test_images_per_chat_and_data_url_length` |
| Responses metadata | fixed: 16 pairs, key 64, value 512 | 422 (17 pairs; a 65-character key; a 513-character value) | `test_responses_metadata_hard_bound` |
| Embedding batch | `embedding_max_batch` | 422 `Embedding batch exceeds configured count or text length` | `test_embedding_batch_and_text` |
| Decoded audio duration and samples | `max_audio_seconds`, `max_audio_samples` | `ValueError` from `media.read_audio` inside the runtime step (the ASR adapter decodes through it), 400 `invalid_request_error` over HTTP; channels are downmixed, not refused | `test_audio_duration_samples_channels_unit`, `test_audio_duration_over_http` |
| Generated image size, steps, count | `max_image_dimension`, `max_image_pixels`, `max_steps`, `max_generated_images` | 422 on JSON routes; the multipart edit route answers 400 | `test_image_dimension_pixels_steps_count` |
| Source image bytes and pixels (edit) | `max_input_image_bytes`, `max_input_image_pixels` | 413 `request_too_large`; 400 `Image exceeds TITANOS_MAX_INPUT_IMAGE_PIXELS` | `test_input_image_bytes_and_pixels_on_edit` |
| Tensor elements, rank, input count | `max_tensor_elements`; fixed rank 8 and 64 inputs | 400 `invalid_request_error` from `tensor_payload` inside the runtime step (the compiled runtime's own validator); rank 400 the same way; input count 422 at the schema | `test_tensor_elements_rank_and_input_count` |
| Video frames and fps | `max_video_frames`, `max_video_fps` | 422 (validator-level: direct `VideoRequest`) | `test_video_frames_and_fps` |
| Music duration, lyrics, reference audio | schema 10..600 s, then `max_audio_seconds`; `max_text_chars`; `max_body_bytes` | 422 over HTTP for the schema range; validator-level for the settings bounds (the prompt clause is shadowed by the shared `max_prompt_chars` bound) | `test_music_duration_hard_schema_over_http`, `test_music_duration_settings_bound` |
| Music output budget | `max_music_samples`, `max_music_output_bytes`, frames above `max_audio_seconds` x rate | `ValueError` before encoding (unit-level: direct `media.write_music_wav`) | `test_music_output_budgets_unit` |
| Asset job queue | `queue_capacity` | 429 `queue_full` | `test_asset_queue_capacity` |
| Replica admission queue | `replica_queue_capacity` | 429 `rate_limit_error` | `test_replica_queue_capacity` |
| Job list size, Idempotency-Key length | fixed 1..100, 1..128 | 422; 400 | `test_list_limit_and_idempotency_key_length` |
| Defaults above ceilings | `default_*` vs `max_*` | `ValueError` at settings load | `test_defaults_may_not_exceed_ceilings` |

Kept visible as expected failures in the same file rather than papered over: the Starlette default multipart part size
(1 MiB per non-file part) refuses with a 400 outside the Titanos error envelope; `timeout` on extension requests has
no upper bound, so a caller can extend a deadline past `client_timeout`.

Bounded elsewhere and not repeated in this file: the Guard's body-read wait (503 `server_busy`) and body-read deadline
(408 `request_timeout`) in `tests/check_security_public_routes.py`; `VideoRequest.input_reference` length and
`pipeline_options` size in `tests/check_i2v.py`; retrieval, rerank and pair document/label counts and model-page fetch
ceilings in `tests/check_retrieval.py`, `tests/check_pair_classification.py` and `tests/check_model_inspection.py`;
native queue, depth/segmentation, diarization/VAD/rerank/sentiment and OCR budgets in their own `check_*.py`. Not
covered in-process anywhere: the API Plus transcribe workflow's VAD timing controls and post-decode budget checks (they
need VAD and diarization model stand-ins); the API Plus multipart file byte bound, which the Guard's `max_body_bytes`
413 shadows over HTTP; and process-level settings (uvicorn limits, MCP body cap, artifact TTL, acquisition bytes,
worker and native startup timeouts).
