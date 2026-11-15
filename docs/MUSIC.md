# Local music and complete songs

Titanos treats music as its own asynchronous asset task. It is not speech
synthesis with a different label.

## Implemented contract

- Task: `music`
- Initial backend: `ace-step`
- Provider: `ace-step`, with pinned reviewed recipes `xl-sft`, `xl-turbo`, and
  `xl-base`
- HTTP: `POST /v1/music/generations` or the shared `POST /api/jobs`
- Agent interfaces: the same typed asset-job schema in Client, MCP, and CLI
- Output: authenticated `.wav` artifact URL with actual channels, sample rate,
  duration, native format, effective controls, and evidence

Simple mode asks for a description, instrumental or song mode, and lyrics when
the selected bundle declares lyric support. Advanced exposes only controls in
the model capability record: duration, vocal language, BPM, key/scale, seed,
guidance, steps, and reference audio. UI locale never changes the song language
or user lyrics.

## Trust and runtime boundary

The provider acquires the reviewed VAE, Qwen3 text encoder, selected XL DiT and
4B planner at immutable Hugging Face revisions. Unrelated default Turbo/base
and 1.7B planner weights are excluded. The DiT's pinned `silence_latent.pt` is
conditioning data for restricted `weights_only=True` loading, not executable
checkpoint code. The provider excludes checkpoint Python and creates
component inventories with every acquired file's size and SHA-256. Activation
keeps the bundle external and atomic.

Generation requires `TITANOS_HOME/music-runtime.json`, written by packaging for
an owned isolated environment. The serving Python process never imports the
music dependency stack. The receipt pins:

- ACE-Step implementation commit
  `ca1e85fe9430179831e6bc6be790c332190a3866`;
- the owned interpreter and audited adapter;
- every runtime file hash.

The bundle retains the upstream model's vocal language codes independently of
the four Studio UI languages. Reference-audio controls remain unavailable until
the trusted runner supports that path; a model-family feature is not sufficient
to claim an implemented Titanos control.

The adapter receives one bounded JSON request, complete local bundle path,
offline environment, output/result paths, and a request-bound progress file.
It cannot select a program or URL from the network request. Cancellation kills
the owned process tree; partial output is not published.

`xl-sft` is the research quality candidate and `xl-turbo` the speed candidate.
That ordering is not listening acceptance. The source contract does not claim
cloud parity, native Windows/Linux qualification, or song quality until the
packaged runtime and complete outputs are tested and heard.

The initial trusted runner implements **XL SFT only**. XL Turbo and XL Base remain
pinned discovery/comparison candidates; their runtime recommendation reports
`unsupported_adapter` and they are rejected before model loading. A packaged
runtime receipt explicitly lists its supported variants.

## Audio behavior

Music output preserves native mono or stereo channel count. Titanos validates
finite samples or a bounded PCM16 WAV, sample rate, frame count, duration and
artifact size. It never silently downmixes music or invents stereo. Existing
speech decoding/writing remains mono and unchanged.

`TITANOS_MAX_MUSIC_SAMPLES` (60,000,000 samples across channels) and
`TITANOS_MAX_MUSIC_OUTPUT_BYTES` (128 MiB) govern generated music independently
of the HTTP upload limits. These defaults accommodate 600 seconds of 48 kHz
stereo PCM16. `TITANOS_MAX_AUDIO_SECONDS` still bounds duration. All three can be
raised or lowered; the model's own declared duration range also applies.

The durable job record distinguishes queued/running/cancelling/terminal state,
native stage evidence, publication/discard disposition, cancellation and
restart interruption. Long songs remain downloadable artifacts rather than
being clipped to fit inline MCP media.

## Memory behavior

Each generation runs in the isolated child with the weights placed according
to the model's optional `placement` (`auto`, `offload` or `resident`; see
`CONFIGURATION.md`). `offload` keeps today's behaviour: every component lives
on the host and visits the device for its stage. `resident` builds every
component (planner, text encoder, DiT, VAE) empty on the meta device and
streams its safetensors tensor by tensor: one tensor's bytes are read into a
host buffer, copied to the device at the stored dtype, cast there, and the
buffer is dropped before the next tensor. No shard is ever mapped or opened
whole: on Windows a whole-file `safe_open` charges the shard to process commit
at open, and a copy-on-write `from_file` mapping is counted against a Job
memory cap but fails only later at page touch (both measured with small owned
probes, `JOB_CAP_SEMANTICS_2159Z.json`). Run r3 exhibited the failure class
this avoids (0xC0000005 during the first DiT shard load, through the earlier
`from_pretrained`/`safe_open` loader); which class r3 actually hit is not
proven and stays under investigation
(`ROOT_R3_NATIVE_CRASH_TIMELINE_CORRECTION.json`). The host transient is one
tensor (at most the 1.1 GB planner embedding). Every checkpoint key is accounted for: unexpected or
missing keys (other than the planner's tied `lm_head.weight`), or a tensor
that does not end on the device at the resolved dtype, refuse the run with
`resident_loader_refused:`. The receipt's `placement_check.loader` records the
files, tensors, bytes and largest tensor the loader actually streamed. Nothing
returns to the host and there is no CPU-offload fallback anywhere, so a device
out-of-memory fails the run rather than quietly moving a component to the host.
On WDDM, device allocations also charge the process's commit 1:1; the child
notes this in its resident `basis` and the launching kit does the host math.

Before loading any weight the child sizes the run from the bundle's safetensors
headers: every parameter of the four loaded components at the resolved dtype
(the F32-stored DiT is counted by element count, not file bytes) plus a fixed
4 GiB of headroom. An explicit `resident` placement refuses before loading when
the device cannot hold that whole bundle bill, when `accelerate` is missing, or
when the dtype is unsupported; the refusal names the measured numbers and is
surfaced verbatim in the job error. `auto` falls back to `offload` under the
same conditions. The result's `evidence.runtime.placement` records the
requested and resolved placement, measured free device bytes, resident, peak,
headroom and required bytes, and the deciding basis; the parent rejects results
whose placement evidence is missing, whose `resident` resolution has less free
device memory than the recorded bill, whose `auto` request resolved to
`offload` without an evidenced obstacle (accelerate absent or free below the
bill), or which contradict the receipt's `cpu_offload` flag.

Placement never lowers a host-memory admission floor. A lower host floor is
only honest when its containment is enforced by the launching harness; without
that, the full artifact-based floor applies unchanged, and receipts written
before this evidence block are read as they were, never relabelled.

## Licenses and alternatives

ACE-Step code and the reviewed core/XL cards declare MIT; retain code and
component notices separately, including Apache-2.0 Qwen components where used.
This is metadata/provenance, not legal approval.

MiniMax Music 3 remains a comparison adapter candidate under its custom model
license and safeguards. HeartMuLa 3B remains the independent permissive
replacement candidate. YuE2, DiffRhythm, LeVo, MuLaCover, Stable Audio and
unresolved SongBloom variants are not silently represented as interchangeable
or unrestricted ACE checkpoints. See the September 21 music information
diamond for exact terms and open native-quality work.
