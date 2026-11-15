# Performance: what was observed, the defaults that follow, the knobs to turn

Titanos is an inference engine that will be loaded by many callers at once. This page records what one measurement pass
(2026-10-06) observed per execution path, the two defaults it changed, the defaults it measured and kept, and the knobs an
operator turns when a server is utilized. Every figure here is **observed on one shared host**, not proven: the GPU carried
other owners' processes (52 to 81 GiB of its 97,887 MiB and 0 to 100% utilization, changing during the pass), the CPU was
30 to 100% busy with other work, and nothing was isolated. A higher number is "faster in that run", not optimal. Figures
are medians over the stated sample count with warm-up requests excluded; a figure without a sample count is a single
observation. Repeat a comparison on an idle host before quoting any of these as a product figure.

## What was measured

| Lane | Path | Build and runtime | Models (local inputs only) |
|---|---|---|---|
| L1 | llama.cpp through `titanos serve` (`--backend llama-cpp`), one GPU | Sealed Windows x64 CUDA build of 83b6f51 (`titanos.exe`, torch 2.8.0+cu128); `llama-server` b10809 `win-cuda-12.4-x64` | Qwen2.5-0.5B-Instruct Q4_K_M (the only GGUF among the inputs) |
| L2 | Transformers chat (`--backend transformers`), one GPU and CPU | Same build | Llama-3.2-3B-Instruct (bf16 checkpoint), SmolLM2-135M-Instruct |
| L3 | Speech (`--backend qwen3-tts`) and embeddings (Transformers and llama.cpp) | Same build and runtime | Qwen3-TTS-12Hz-1.7B-Base; paraphrase-MiniLM-L3-v2; the 0.5B GGUF at `pooling: mean` as the embedding GGUF stand-in |
| L4 | CPU, HTTP and startup; Windows against WSL Ubuntu | Same Windows build; the Linux x86_64 CPU build of 83b6f51 (torch 2.12.1+cpu) | SmolLM2-135M-Instruct (CPU) |

Host: Intel Core Ultra 9 285K (24 cores), 127 GiB RAM, one RTX PRO 6000 Blackwell (97,887 MiB) shared with other owners,
Windows 11; WSL Ubuntu 24.04 with 24 vCPU and 62 GiB. The builds carried the defaults of 83b6f51 (`native_slots` 2,
`native_context_size` 4096, `limit_concurrency` 32); the changed defaults below were measured as explicit registrations
and flags against those defaults.

## Defaults changed by this pass

| Setting | 83b6f51 | Now | Evidence (observed on the shared host) |
|---|---|---|---|
| `TITANOS_LIMIT_CONCURRENCY` | 32 | 64 | With 32, uvicorn answered a plain 503 (before Titanos saw the request) for 21% of `/healthz` requests at 40 open connections and 12.5% at 64, so a liveness probe of a busy server failed; with 64 none was refused at the same p50/p95 latency (19 / 66 ms at 40 connections, 24 / 123 ms at 64; 200 requests per level). The broker role already warned below 64. Every open connection counts, idle keep-alive ones included. |
| `TITANOS_NATIVE_SLOTS` | 2 | 4 | At the same total context (8192) and 4 concurrent streams, 4 slots gave a first token in 81 ms (per-stream decode 215 tok/s, aggregate 720) where 2 slots queued at 346 ms (462 per stream, 747 aggregate) and 1 slot at 670 ms; at 8 streams 760 ms against 1,029 ms (2 slots) and 1,500 ms (1 slot). 40 requests per level, 3 warm-ups excluded. GPU memory after load 1,051 MiB against 1,003 MiB for the 83b6f51 default (0.5B model, Q4_K_M). |
| `TITANOS_NATIVE_CONTEXT_SIZE` | 4096 | 8192 | Raised with the slots so the per-slot context stays 2,048 tokens: llama.cpp divides one context allocation among its slots, and a larger Ollama `options.num_ctx` resizes a runtime's slots ([NATIVE.md](NATIVE.md)). The KV cache is the memory cost of this pair of changes: about 48 MiB on the 0.5B model here, proportionally more on larger models (f16 KV for 8,192 tokens of this model was about 96 MiB). |

Both native values are defaults for a new registration without explicit `native` values; a saved registration keeps the
values it was written with, and `--native-options` or the environment still set either. Applies to the shipped binaries
only from the next build; the 2026-10-06 builds carry the old defaults.

## Defaults measured and kept

| Setting | Default | What was observed |
|---|---|---|
| `TITANOS_THREADS` (`serve --threads`) | min(8, cpu) | More PyTorch threads helped CPU prefill and embedding batches (24 threads: Llama-3.2-3B prefill of a 1,150-token prompt 12.4 to 8.7 s; MiniLM 64-text batches 1.3 to 1.6x more inputs/s) and did nothing for decode (3.1 to 3.4 tok/s), while on the 135M chat fixture 24 threads were 36% slower on Windows (7.05 s against 5.18 s per 128-token answer, n=20) and collapsed inside the 24-vCPU WSL VM (3 requests in 13 minutes at a load average of 20). Workload-dependent on a shared host: kept at 8, raise it per host for CPU batch work after a repeat on that host. |
| `TITANOS_KEEP_ALIVE_TIMEOUT` | 5 s | Idle loopback connections were dropped after 5 s as configured; the reconnect cost 5.4 ms on Windows and 1.5 ms in WSL (n=5). Raise it for remote or TLS clients only. |
| `TITANOS_NATIVE_THREADS` | 4 | With every layer offloaded, 8 threads against 4 changed nothing beyond noise (short first token 231 against 224 ms under a 98% busy co-tenant). |
| `TITANOS_NATIVE_GPU_LAYERS` | `auto` | `auto` and `all` both resolved to 25 of 25 layers with an identical 1,003 MiB footprint. |
| `TITANOS_NATIVE_BATCH_SIZE` / `UBATCH_SIZE` | unset (runtime default 2048 / 512) | 4096 / 1024 read 50,897 tok/s of prompt processing on a 1,201-token prompt against 43,410 (512 / 512) and 34,851 (default, under a heavier co-tenant load) with no single-stream change; for the GGUF embedding registration 4096 / 1024 with 32 slots read 31% more inputs/s at 64 x 50 tokens and 2.8x the single-text rate at 8 clients, nothing at 64 x 188 tokens. A registration knob to set for long prompts and embedding batches; not a default change without a quiet-GPU repeat. |
| `TITANOS_NATIVE_QUEUE_CAPACITY` | 32 | Not exercised by the pass: the 25 refusals at 40 streams on 2 slots were uvicorn's connection limit of 32 (`Exceeded concurrency limit` in the serve log, answered in 62 to 218 ms, no `queue_full`), which the new `TITANOS_LIMIT_CONCURRENCY` of 64 lifts; more slots, not a deeper queue, is the lever for throughput (above). |
| Transformers `dtype` | `auto` (float16 on CUDA) | bfloat16 (the checkpoint's native dtype) read about 10% faster than float16 on Llama-3.2-3B (36.1 against 32.4 tok/s short, 37.3 against 34.9 at 1,144 tokens, n=10 / 6) inside the co-tenant noise band, same 6,651 MiB. Register `--dtype bfloat16` for bf16 checkpoints on Ampere or newer after a quality comparison; not a default change. |
| Transformers attention | `sdpa` | eager read the same as sdpa at 100 to 1,144 tokens (32.3 against 32.4 tok/s). |
| Speech dtype | bfloat16 | float16 read the same real-time factor (1.94 to 2.01 against 1.97 to 2.11 s of compute per second of audio). |

## The paths, and the knob for each

### llama.cpp (`--backend llama-cpp`): slots are the concurrency lever

Default registration of 83b6f51 (2 slots, 4096 context), single stream, quiet co-tenant (0.5B Q4_K_M, GPU, n=10): first
token 25.6 ms (p95 42.9) for an 84-token prompt and 47.5 ms for 1,201 tokens; decode 653 tok/s per stream; prompt
processing 8,704 tok/s short and 41,510 tok/s long. The same registration at 8 streams queued six requests in the Titanos
pool (first token 887 ms against 112 ms with 8 slots) and at 40 streams 25 of 80 requests were refused with a plain 503 by uvicorn's connection limit of 32, before Titanos saw them.

| Slots (context) | 4 streams: first token / per-stream / aggregate tok/s | 8 streams: first token / aggregate | GPU memory after load |
|---|---|---|---|
| 1 (8192) | 670 ms / 675 / 547 | 1,500 ms / 557 | 1,057 MiB |
| 2 (8192) | 346 ms / 462 / 747 | 1,029 ms / 693 | 1,053 MiB |
| 4 (8192) | 81 ms / 215 / 720 | 760 ms / 604 | 1,051 MiB |
| 8 (16384) | 62 ms / 101 (stalled by a co-tenant burst) / 385 | 112 ms / 1,063 | 1,147 MiB |

Per-stream decode falls as streams share the GPU; the aggregate is what a loaded server delivers. Size `slots` to the
concurrency you expect and `context_size` to `slots x` the per-request context you need; the KV cache grows with the total.
Flash attention and KV cache types are not registration knobs in this version (`LLAMA_*` environment is stripped from the
child); b10809 defaults to `--flash-attn auto`, and a direct run showed `off` costing 32 MiB and a q8_0 KV cache saving 44
MiB on the 0.5B model with speed inside the noise. Load through Titanos took 3.7 to 5.3 s against 2.0 to 2.5 s for the same
`llama-server` started directly: Titanos hashes the checkpoint and every library beside the runtime (about 1.1 GB for the
CUDA build) on each start. Titanos's own forwarding cost on a quiet GPU is bounded by that 25.6 ms first token; under a
98% busy co-tenant the back-to-back pair read +35 to +65 ms single-stream with decode within 3%.

### Transformers chat (`--backend transformers`): one request at a time per process

One resident model and one engine lock per process: at 4 streams the aggregate equals the single-stream rate (32.3 against
32.4 tok/s on Llama-3.2-3B float16, GPU, quiet window) and the fourth caller waits 11.9 s for its first token. SmolLM2-135M
on the same GPU ran at 29.8 tok/s against the 3B model's 32.4: about 30 ms per generated token is spent in the Python
generation loop, not in the weights, which is also why a busy co-tenant cost this path 10x (3.2 to 32.4 tok/s for the same
registration across the day) against 5x on llama.cpp. The knobs today:

- Replica workers (`titanos worker` with the same model on the same GPU, [MULTIGPU.md](MULTIGPU.md)): two replicas gave
  51.5 to 56.0 aggregate tok/s at 2 to 4 streams (1.6 to 1.7x one process) for 13,302 MiB (2 x 6,651); per-stream decode
  fell to 25.6 to 29.5 tok/s. Size GPU memory as replicas x (weights + about 1 GiB) plus a reserve.
- Route chat models that exist as GGUF to the llama.cpp path; its 0.5B model gave 653 tok/s per stream and slots.
- CPU (`--device cpu`, float32): Llama-3.2-3B loaded in 8.3 to 9.1 s warm (14.4 GB working set), first token 1.3 to 1.6 s
  for 100 tokens and 8.7 to 12.4 s for 1,150 tokens, decode 3.1 to 3.4 tok/s; SmolLM2 24.9 tok/s at 8 threads.
- First-token and load: 70 ms (100 tokens) and 136 ms (1,143 tokens) on the GPU; load 8.3 s warm and 18.9 s on a cold
  page cache for 7.2 GB. `/api/models/unload` leaves about 1 GiB of CUDA context resident until the process exits; it is
  not a leak.

### Speech (`--backend qwen3-tts`): about 2 s of compute per second of audio

Qwen3-TTS-12Hz-1.7B-Base, bfloat16, GPU, voice registered with a 5 s reference, 5 runs per text: real-time factor 2.05
(9 words, 9.2 s for 4.5 s of audio), 2.11 (20 words, 14.8 s for 7.1 s) and 1.97 (32 words, 25.0 s for 12.7 s); GPU
utilization about 20% during synthesis, so the autoregressive talker is launch-bound. The first synthesis after load took
19.4 s (it builds the voice prompt); send one warm-up request after loading. Two concurrent callers serialize (the second
finished at 28.3 s against 13.9 s); replica workers (about 5 GiB each) are the concurrency lever. CPU float32 completed at a
real-time factor of 6.3 to 6.4 (27 s for 4.2 s of audio; 8.2 GiB working set). GPU memory +4.7 GiB after load, peak +4.9 GiB.

### Embeddings: batch texts per request

paraphrase-MiniLM-L3-v2 on the GPU (float16, n=50 single client, n=400 at 8 clients): one text cost 7 to 11 ms (87 to 140
requests/s), 64 texts of about 50 tokens cost 31.6 ms (1,684 inputs/s) and 64 texts of about 188 tokens 36.4 ms (1,491
inputs/s); 8 clients reached 2,495 and 2,142 inputs/s. float32 read the same as float16 (1,689 against 1,684 inputs/s).
On the CPU, `serve --threads 24` read 1.3x (758 against 581 inputs/s at 64 x 50 tokens) and 1.6x (309 against 196 at
64 x 188) the default 8 threads. The llama.cpp embedding path (the 0.5B chat GGUF at `pooling: mean`, 896 dimensions,
not an embedding model) read 106 inputs/s for 64 x 50 tokens and 58 inputs/s for 64 x 188 tokens at the default slots
on the GPU, 27 single-text requests/s (36 ms, of which about 20 ms is Titanos's forwarding against the direct 14 ms),
and 10 to 24x less on the CPU; more slots plus `batch_size` 4096 / `ubatch_size` 1024 in the registration gave the gains
named above. On this host the Transformers path with request batching was the fast embedding path.

### HTTP, startup and the stdio MCP bridge

- Startup to the first `/healthz` 200: 1.89 s median on Windows (n=5), 1.43 s in WSL (n=5); idle working set 129 MB
  (Windows) and 122 MB RSS (WSL); graceful stop 0.2 to 0.8 s with the listen socket gone.
- `/healthz` p50 0.7 ms at one client and 8.0 ms at 16 (1,344 to 1,748 requests/s) on Windows; `/v1/models` 0.8 / 10.2 ms.
  uvicorn runs one worker (fixed), so 16 clients give 1.3 to 1.5x the one-client throughput.
- Model load of the 135M fixture: 5.8 s median on Windows, 4.7 s in WSL, dominated by the first torch import in the frozen
  process. Streaming a 128-token answer sent 102 SSE frames (22.8 KB against 938 B non-streamed) with a 41 ms median gap.
- Open finding: streamed and concurrent chat requests grew the server by about 2 threads and 20 to 25 MB of commit per
  streamed request over 60 requests without a plateau (31 to 175 threads, 2.4 to 4.0 GB on Windows; the same pattern in
  WSL), while 60 non-streamed requests stayed flat. The per-token `asyncio.to_thread` hand-off in the streaming path is
  the suspected cause, not a proven one; it is under investigation and not changed here.
- stdio MCP (`titanos mcp --url ...`): spawn to `initialize` 1.25 s (n=5), `tools/list` 21.8 ms p50 for 72 tools (218 KB,
  n=100). Agents should keep one session.

## GPU memory rule before a load

Titanos checks free accelerator memory before ACE music loads and sharded loads only; a llama.cpp or single-device PyTorch
load is not admitted against free memory, and `llama-server` runs with `--fit off`, so a load that does not fit fails at
load time with the runtime's error. Before loading on a shared card, require free memory of at least the expected footprint
plus a reserve (the pass used 6 GiB): observed footprints after load were 1,003 to 1,147 MiB for the 0.5B GGUF (by slots and
context), 6,651 MiB for a 3B half-precision Transformers chat model (+0.9 GiB during inference), 4,735 MiB for the 1.7B
speech model (+4.9 GiB peak), 527 MiB for MiniLM-L3 (+1.0 GiB peak) and 1,317 MiB for the GGUF embedding registration.

## Not measured

Isolated-GPU figures (the card was shared throughout); any GGUF above 0.5B (none among the inputs); flash attention and
KV cache types through the shipped binary (no knob); multi-GPU split modes, sharding and tensor parallel (one GPU only);
Linux CUDA (the Linux build is CPU torch); quality effects of bfloat16, q8_0 KV or eager attention (no output comparison);
speech with `torch.compile` or CUDA graphs (hardcoded off); a real embedding GGUF, rerankers, ONNX and OpenVINO graphs;
remote or TLS clients; the mechanism behind the streaming growth. The pass's report and every receipt (raw timings,
request and response sizes, 1 Hz GPU samples, exact command lines and registrations) are retained with the 2026-10-06
performance records outside the repository.
