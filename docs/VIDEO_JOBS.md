# Local asynchronous video jobs

POST `/api/video/jobs` with the same payload as `/v1/videos/generations`. It returns
HTTP 202 and a job ID. GET `/api/video/jobs/{id}` polls the status; DELETE requests
cancellation. Completed records contain the MP4 result URL. All routes use the
same bearer-key policy as inference. The synchronous route remains available.

## Reference image geometry

`size` is the requested output `WIDTHxHEIGHT`. Titanos decodes an inline
`input_reference` to RGB and passes that image and the requested dimensions to
the selected video pipeline; it does not apply a universal crop or letterbox.
The pipeline determines how the reference fits those dimensions.

For Wan image-to-video with Diffusers 0.35.2, preprocessing resizes the reference
directly to the output dimensions. Different aspect ratios therefore stretch
the conditioning image: a square reference sent with `size: "448x256"` is
widened. This happens before inference and is distinct from model-generated
motion or scene drift. To keep proportions, choose an output size with the
reference's aspect ratio and dimensions supported by the model, or deliberately
crop the reference to the desired ratio before submitting it. Cropping can
remove content; it is not automatic preservation of the full scene.

There is no common `resize_mode` request option. Do not assume another video
pipeline uses Wan's policy, or that matching proportions guarantees prompt,
identity or background fidelity in the generated frames.

By default, the queue allows eight queued/running jobs and one executing job;
`TITANOS_QUEUE_CAPACITY` and `TITANOS_JOB_WORKERS` configure those limits. Ordinary
model workers still govern GPU residency and interference with other requests.
Records are atomic JSON under the output directory's `jobs` folder. One server
owns that folder; use separate TITANOS_HOME values for separate installations.
Terminal records older than `TITANOS_ARTIFACT_TTL_SECONDS` (24 hours by default)
are reclaimed when the store is next opened. Generated artifact URLs follow the
same configurable retention window. Same-key submission replay requires the
original job record to remain retained; a reclaimed key can create new work.

States are queued, running, cancelling, cancelled, completed, failed or interrupted.
After restart, unfinished records become interrupted and are not replayed. Cancellation
is cooperative: pipelines with step callbacks stop between diffusion steps; other
pipelines can only finish their current operation before the result is discarded.
Loading weights or an individual kernel cannot be interrupted immediately. A
cancelling job is not reported cancelled until its execution ends. No partial video
is reported as a successful completed result.

A queued cancellation prevents generation when that callback gets its executor
turn; it may remain `cancelling` until then. If completion wins the publication
lock first, cancellation returns the completed record. Check `cleanup_error` as
well as status: a cancelled job can report a failure to remove its owned artifact.

## Client lifecycle contract

An asynchronous client (Cauldron is the reference consumer) can recover from a lost
answer, a dropped connection or its own restart without starting duplicate work:

- Send an `Idempotency-Key` (1 to 128 characters) on every submit, one key per
  attempt (for example `client:{job}:{step}:{attempt}`). The same key with the same
  body replays the existing record; a different body answers 409
  `idempotency_conflict`. Keys are matched per owner key. A deleted record no longer
  claims its key.
- If the submit answer is lost, `GET /api/jobs?idempotency_key=<key>` returns the job
  that key created (or an empty list when none was created); re-posting the same body
  with the same key does the same. `status=` and `task=` filter the list.
- `DELETE /api/jobs/{id}` answers the record plus `cancel_accepted`: `true` when the job
  was still queued, running or cancelling, `false` when it was already terminal (the
  record is returned unchanged). An accepted cancel reads `cancelling` until execution
  stops and always ends in a terminal status: `cancelled`, or `completed` or `failed`
  when the work finished first. `cancel_requested_at` stays on the record. Poll until a
  terminal status before reusing the GPU for other work.
- After a server restart, unfinished jobs read `interrupted`, with `interrupted_at` and
  `retryable: true`. The work is not replayed; replaying its key returns the
  interrupted record, so retry with a new key.

tests/check_job_contract.py checks these rules at the store and HTTP level.

This is Titanos's local job API, not a claim of OpenAI video API parity or a
distributed durable queue. tests/check_jobs.py checks state transitions and, with
`--video`, runs a real tiny diffusion model and decodes the resulting MP4.
