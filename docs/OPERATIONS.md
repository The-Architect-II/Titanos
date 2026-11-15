# Operations

Running Titanos as a service, upgrading it, reading and rotating its logs, the readiness and metrics routes an
orchestrator or load balancer can poll, HiveMind gateway registration, and what to back up under `TITANOS_HOME`.
Everything here is `titanos` itself or the host's own service manager; Titanos installs no agent and opens no outbound
connection on its own (the one exception, gateway registration, is explicit operator configuration).

## Run as a service

`titanos service install` first checks that the registry exists (`titanos init`) and that the executable can run titanos
(`<executable> --version`), then writes a unit for the host's service manager that runs `titanos --config <registry> serve`
with `TITANOS_HOME` set explicitly, registers it, starts it and waits for `GET /healthz` of the server it configured
(`--wait SECONDS`, default 30; `0` skips the probe). It exits 0 only when the server answered; otherwise it exits 3 with
the reason and the last log lines, so a unit that is registered but not serving is never reported as installed.
`titanos service stop` stops the server, `titanos service start` starts it again (with the same wait), `titanos service
status` reports whether the unit is registered and the server is running, and `titanos service uninstall` stops the
server and unregisters the unit. Every subcommand takes `--name N` (default `titanos`), `--system` (a machine-wide unit;
needs an elevated or root shell), `--dry-run` (print the files and commands, write and run nothing) and `--json`.

| Host | What `install` writes and runs | `--system` |
|---|---|---|
| Windows | A Scheduled Task `<name>` registered from the task definition `TITANOS_HOME\service\<name>.xml` (`schtasks /Create /XML`): a logon trigger for your account, run with your interactive token at least privilege, no run-time limit, one instance. It starts the launcher `TITANOS_HOME\service\<name>.cmd`; the launcher sets `TITANOS_HOME`, loads `TITANOS_HOME\service\<name>.env` and runs `titanos serve --pid-file TITANOS_HOME\service\<name>.pid`, appending the server's output to `TITANOS_HOME\logs\<name>.log`. `schtasks /Run` starts it at once (`start` runs it too). `stop` and `uninstall` run `schtasks /End`, which ends the launcher only, and then stop the server by the process id it recorded in the `.pid` file (`taskkill /PID <id> /T /F`, a forced stop); without that file nothing is killed. Never use `schtasks /End` alone to stop the server: the launcher ends and the server keeps serving. `status` reports the task as registered and reads the `.pid` file: a live process is running, a stale file is not (the task's own state is the launcher's). An ordinary (non-elevated) shell is enough: the plain `schtasks /Create /SC ONLOGON` form is a machine-wide trigger that Windows refuses to a standard user with `Access is denied`, which is why Titanos registers the per-user XML instead. If the `titanos.exe` beside the interpreter is a launcher stub without the package (`--version` fails), `install` refuses and names `--executable`. | `/SC ONSTART /RU SYSTEM` from an elevated prompt. The SYSTEM account must be able to read `TITANOS_HOME`, and a vault sealed by your account (DPAPI) is not readable by SYSTEM: keep `--system` for an installation whose credentials are `env:` references or re-enter them as SYSTEM. |
| Linux | A systemd user unit `$XDG_CONFIG_HOME/systemd/user/<name>.service` (`~/.config` by default) with `Environment=TITANOS_HOME=...`, `EnvironmentFile=-<home>/service/<name>.env`, `Restart=on-failure`; `systemctl --user daemon-reload` and `systemctl --user enable --now <name>`; `stop`/`start` are `systemctl --user stop|start <name>`. A user unit starts at login; `loginctl enable-linger $USER` keeps it running at boot and after logout. | `/etc/systemd/system/<name>.service`, `WantedBy=multi-user.target`, plain `systemctl`. |
| macOS | A launchd agent `~/Library/LaunchAgents/com.titanos.<name>.plist` (`RunAtLoad`, `KeepAlive` unless the exit was clean) whose program is `/bin/sh -c 'set -a; . <env file>; set +a; exec titanos ... serve'`, output to `<home>/logs/<name>.log`; `launchctl bootstrap gui/<uid>` and `launchctl kickstart -k`. `stop` is `launchctl bootout`; `start` loads the plist again (`bootstrap`, tolerated when it is already loaded) and `kickstart -k`. | `/Library/LaunchDaemons/com.titanos.<name>.plist` in the `system` domain. |

Why a Scheduled Task on Windows: a Windows Service proper is a program that speaks the Service Control Manager protocol.
A console program such as `titanos serve` cannot be registered with `sc.exe` from user space without a wrapper that
speaks it, and Titanos ships none; the task gives automatic start, restart on logon and `schtasks` control without one.
`titanos service status` says which form is installed.

**The environment file.** `TITANOS_HOME/service/<name>.env` holds every `TITANOS_*` variable of the shell that ran
`install` (except `TITANOS_HOME`, which the unit sets itself) plus `--host`, `--port` and `--role` when given. On Linux
and macOS it is written owner-only (`0600`) and its values are single-quoted; on Windows it is protected by the profile's
ACL, so keep `TITANOS_HOME` inside the profile (or set an ACL on the `service` directory yourself). `TITANOS_API_KEY`
therefore lives in that file only: never in the unit, the task definition, the launcher or a command line. Edit the file
and restart the service to change a setting; `uninstall` keeps it.

```sh
# a registry first (install refuses without one), then a keyed loopback server as a user service
titanos init
TITANOS_API_KEY=... TITANOS_PORT=11435 titanos service install    # exit 0 once GET /healthz answered, 3 with the log tail if not
titanos service status --json
curl -fsS http://127.0.0.1:11435/v1/health/ready

# see what would be written first
titanos service install --dry-run --json

# stop and start the server (the unit stays registered)
titanos service stop
titanos service start

# stop and unregister (TITANOS_HOME, the registry, models and the environment file stay)
titanos service uninstall
```

`status` exit codes: `0` running, `3` registered but not running, `4` not registered, `1` the service manager's command
failed (its last output line is in the message); `install` and `start` exit `3` too when the server did not answer
`GET /healthz` within `--wait` seconds (the message carries the reason, `timeout` or `exited`, and the last log lines). A
refusal names the field: an unknown platform, a `--name` outside `[A-Za-z][A-Za-z0-9._-]{0,63}`, no registry at the
configured path (`titanos init`), an executable whose `--version` fails, no `titanos` executable beside the interpreter
or on `PATH` (pass `--executable`).

Serve preconditions still apply inside the service: a registry (`install` checks it), a bind beyond loopback needs
`TITANOS_API_KEY` (or a minted admin key), the broker role needs a bootstrap key and native TLS or a loopback bind, the
managed role an explicit non-reserved port ([TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md)). A failed precondition exits before
binding; `install` and `start` report it (exit 3, reason `exited`), the service manager records the exit and, on Linux
and macOS, restarts the unit with the configured delay. Read the log before retrying.

## Upgrade

1. Note the running version: `titanos --version` or `GET /healthz`.
2. Stop the service: `titanos service stop` (on Windows this ends the task and the server process recorded in its pid file; `schtasks /End` alone ends only the launcher and leaves the old server running and answering) or, without a service, stop `titanos serve`. `titanos service status` must say `registered, not running`.
3. Upgrade the package in the same interpreter: `python -m pip install --upgrade titanos-engine` (a source checkout: `git pull` then `python -m pip install .`). The native llama.cpp runtime is separate: `titanos native-install` installs a new pinned build and `titanos upgrade` installs a versioned runtime beside the existing one; both keep the model registry.
4. Run `titanos list` and `titanos settings` offline: the registry and the effective settings must read without error before the server starts again.
5. Start the service again: `titanos service start` waits for `GET /healthz` (exit 3 with the log tail when the new version does not come up), then wait for `GET /v1/health/ready` to answer 200.
6. Downgrade the same way with a pinned version if the new one does not start; `TITANOS_HOME` is not rewritten by an upgrade.

The release process and what a published release contains are in [DISTRIBUTION.md](DISTRIBUTION.md).

## Logs and rotation

`titanos serve` writes its log to standard error (uvicorn's access and application log at `TITANOS_LOG_LEVEL`); the
broker's redaction filter keeps credentials, bearer tokens and key secrets out of every line. With `TITANOS_LOG_FILE` (an
absolute path) it also appends the uvicorn, access and application lines to that file and rotates it itself: at
`TITANOS_LOG_MAX_BYTES` (64 MiB) the file becomes `<file>.1`, `TITANOS_LOG_BACKUPS` (4) rotated files are kept, and
`TITANOS_LOG_MAX_AGE_DAYS`, when nonzero, deletes older rotated files. On Windows a rotation waits while another program
holds the file open without delete sharing, and lines keep going to the current file. Otherwise Titanos rotates
nothing itself. Where the lines go and how to bound them:

| Host | Where | Follow | Bound |
|---|---|---|---|
| Windows (task) | `TITANOS_HOME\logs\<name>.log` (appended by the launcher) | `Get-Content -Wait <file>` | `titanos service stop`, move the file aside, `titanos service start` (the server holds the file open while it runs, and `schtasks /End` alone does not stop it); or point the launcher at a dated file. There is no built-in size cap. |
| Linux (systemd) | The journal | `journalctl --user -u titanos -f` (`journalctl -u titanos -f` for `--system`) | `journald.conf` `SystemMaxUse=` / `RuntimeMaxUse=`, or `journalctl --vacuum-size=`, `--vacuum-time=` |
| macOS (launchd) | `<home>/logs/<name>.log` (`StandardOutPath` and `StandardErrorPath`) | `tail -f <file>` | A `newsyslog.conf` entry for the file, or `titanos service stop`, rotate, `titanos service start` |
| Any (foreground) | The terminal | - | Redirect to a file and rotate it with the host's tool |

Request-level facts live elsewhere: every answer carries `X-Request-ID`, broker decisions are listed by
`GET /api/broker/decisions`, usage by `GET /api/broker/usage`, and the append-only audit log (`titanos audit`,
`GET /api/audit`) records key, credential and policy changes with its own retention
(`TITANOS_BROKER_AUDIT_RETENTION_DAYS`; [BROKER_SECURITY.md](BROKER_SECURITY.md)).

## Readiness and liveness

Three routes, three questions. All are `GET`; the first two are public (no key, no body).

| Route | Question | Answer |
|---|---|---|
| `/healthz` | Is the process answering? | 200 `{"status": "ok", "version", "role"}` while the server runs and uvicorn has a connection slot: above `TITANOS_LIMIT_CONCURRENCY` (64) open connections uvicorn answers a plain 503 before Titanos sees the request, on this route too, so a liveness probe of a busy server must tolerate 503 or the limit must be raised ([PERFORMANCE.md](PERFORMANCE.md)). Liveness only. |
| `/v1/health/ready` | Will an inference request be served now? | 200 `{"status": "ready", "version", "role", "models", "checks"}`; otherwise 503 `{"status": "not_ready", "reason", "message", "checks", ...}` with `Retry-After: 1` and `Cache-Control: no-store`. |
| `/readyz` | Is a model resident in memory? | 200 with the loaded model ids; 503 while nothing is loaded, which is the normal idle state (models load on first use or `POST /api/models/load`). Needs a key on a keyed server. |

`/v1/health/ready` says 503 exactly when a request would not be served, so HiveMind, a load balancer or a container
orchestrator can take the server out of rotation on it:

| `reason` | When | Clears when |
|---|---|---|
| `registry_reloading` | A registry replacement (`POST /api/models/reload` or `POST /api/models/register`) is in progress | The reload finishes |
| `registry_unavailable` | A reload failed and the engine routes answer 503 `registry_unavailable` | A reload succeeds or the server restarts |
| `upstream_unhealthy` | Broker role: an enabled (required) upstream's circuit breaker is open; `checks.upstreams.unhealthy` names them | The breaker half-opens and a probe succeeds, or `POST /api/broker/upstreams/{id}/breaker/reset` |
| `broker_config_unavailable` | Broker role: `broker.json` did not load (the message carries the error) | The file is repaired and the server reloads it |

In standalone and managed roles upstreams are not consulted (a managed server's cloud use is parent-delegated); a
broker with no enabled upstream requires none and is ready. `models` is the number of registered models, not loaded
ones: a server with an empty registry is ready to answer `/v1/models` and to receive registrations. uvicorn binds its
socket only after startup completes, so there is no observable "starting" state; a connection refusal is one.

For a HiveMind engine-manager manifest, `titanos hivemind-manifest` already names `/v1/health/ready` as both the
`ready` and `health` probe ([TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md#hivemind)). A Docker `HEALTHCHECK` or a Kubernetes
probe should use `/v1/health/ready` for readiness and `/healthz` for liveness.

A busy server answers 503 to every request above `TITANOS_LIMIT_CONCURRENCY`, health probes included. Set
`TITANOS_HEALTH_PORT` to give probes a port of their own: `serve` then also listens there (same host, same TLS) for
`/healthz`, `/readyz` and `/v1/health/ready` only, with no concurrency limit (probes are access-logged like other requests), through the same
application and auth. Other paths answer 404 there. The health listener stops with the server, and a health port that
cannot be bound stops `serve` like the main port does.

## HiveMind registration

A managed Titanos can announce itself to a HiveMind gateway once, after it is ready: `serve --role managed
--register-with-gateway URL --job-type JT` does it at startup, and `titanos hivemind register-service URL --job-type JT`
does it for a server that is already running, from any host that can reach both. The request is
`POST <gateway>/register-service {"job_type": JT, "port": P}` with `Authorization: Bearer <key>` when a key is
available: `TITANOS_API_KEY` by default, or the variable `--key-env VAR` names when the gateway's key is not this
server's (the option takes a variable name; no `titanos` option takes a secret value). Without a key the request carries
no credential, which is right for a gateway on the same loopback. `--url` names the Titanos server whose
`/v1/health/ready` is probed first (default `TITANOS_URL`, then `http://127.0.0.1:11435`) and `--port` the port the
gateway will call (default: the port of `--url`). The command prints `{gateway, job_type, port, bearer_sent, status}`,
never the key, and exits 1 when the server never became ready or the gateway did not answer 2xx.

## Metrics

Titanos has no Prometheus exposition endpoint. The routes that exist, all JSON:

| Route | Scope | Content |
|---|---|---|
| `GET /api/broker/metrics` | `broker:read` | Counters of the broker facade: requests by surface, target class and outcome, fallbacks by trigger, circuit-breaker transitions, time-to-first-byte buckets, egress bytes by upstream, authentication failures and budget stops. Present in every role; mostly zero on a standalone server without a broker configuration. |
| `GET /api/broker/usage` | `broker:read` | Per-key, per-upstream, per-model and per-day usage rows with token counts and priced cost (`TITANOS_BROKER_USAGE_RETENTION_DAYS`). |
| `GET /api/broker/decisions` | `broker:read` | The recent routing decisions (`TITANOS_BROKER_DECISION_LOG_SIZE`). |
| `GET /api/info` | any key (`settings` block: `settings:read`) | Version, role, registered models and tasks, interface paths, the job queue capacity. |
| `GET /api/models/loaded`, `GET /readyz` | `models:read` | What is resident in memory now. |
| `GET /api/workers`, `GET /api/devices` | `models:read` | Worker processes and the accelerator inventory. |

A collector polls these with a key holding `broker:read` and `models:read` and converts the JSON itself. A keyless
loopback server answers them without a key.

## Backups of TITANOS_HOME

`TITANOS_HOME` (default `~/.titanos`) is the whole state of one installation:

| Path | What | Back up |
|---|---|---|
| `models.json` | The model registry (names, tasks, backends, devices, per-model defaults, worker assignments) | Yes; small; `titanos list` reads it offline |
| `models/` | Pulled and acquired weights | Yes, or re-pull them (`titanos pull NAME REPO`) and re-register; the registry alone does not restore weights |
| `broker.json` (or the file named by `TITANOS_BROKER_CONFIG`) | Upstreams, aliases, virtual models, policy, prices, advisors (no secrets) | Yes |
| `broker/keys.json` | Minted keys: SHA-256 digests, scopes, limits, budgets, expiry (never the secrets) | Yes; clients keep their own `tk_...` tokens |
| `broker/vault/` | The credential index (`index.json`: ids, sources, bindings) and the sealed-file store when that backend is in use | See the caveat below |
| `broker/audit/`, `broker/usage/`, `broker/decisions/` | Audit log, usage rows, and the decision log when `TITANOS_BROKER_DECISION_LOG_FILE` is on | Yes for the audit log (it is hash-chained: `titanos audit verify`); usage and decisions as you need them |
| `outputs/`, `acquisitions/`, `runtime-setups/` | Generated assets and job state, acquisition and runtime-setup records | Optional; artifacts expire by `TITANOS_ARTIFACT_TTL_SECONDS` |
| `service/`, `logs/` | The service environment file (holds `TITANOS_API_KEY`) and service logs | The environment file, as a secret |

Take the copy while the server is stopped, or accept that a reload in flight may leave `models.json` and the broker
files from slightly different instants (each file is written under its own lock; a copy made while the server runs
can still pair a newer registry with an older `broker.json`).

**The vault caveat.** The vault stores a credential's secret in the platform backend `titanos broker credential set`
chose (`TITANOS_BROKER_VAULT_BACKEND`, default `auto`): Windows DPAPI, the macOS keychain, the Linux Secret Service, or
the passphrase-sealed file (`TITANOS_BROKER_VAULT_PASSPHRASE_FILE`). On Windows a DPAPI-sealed value is bound to the
user account and the machine that sealed it: a copy of `broker/vault/` restored on another machine, under another account
or after a profile reset does not decrypt, and the keychain and Secret Service stores are not inside `TITANOS_HOME` at
all. A restorable backup of credentials therefore means one of: `env:` or `file:` references (the secret lives in the
service environment file or an owner-only file you back up yourself), the sealed-file backend with its passphrase file
backed up separately, or re-entering each credential after a restore (`titanos broker credential set <id> --upstream
<id>`, then `titanos broker credential validate <id>`). `titanos broker credential list` shows each credential's
`source` (`env`, `file` or `vault`); [BROKER_SECURITY.md](BROKER_SECURITY.md#credential-storage) has the backend
details.

Restore: `titanos service stop`, copy `TITANOS_HOME` back, run `titanos list` and `titanos broker status` offline,
`titanos service start`, then `GET /v1/health/ready` and `titanos broker credential validate` for each stored credential.

## Performance and tuning

[PERFORMANCE.md](PERFORMANCE.md) holds what the 2026-10-06 pass observed per path (llama.cpp slots and context, the
Transformers chat path, speech, embeddings, HTTP and startup), the two defaults that changed because of it
(`TITANOS_LIMIT_CONCURRENCY` 64; `TITANOS_NATIVE_SLOTS` 4 over `TITANOS_NATIVE_CONTEXT_SIZE` 8192), the defaults measured
and kept, and the knobs to turn for a server that is utilized: slots and context for the native path, replica workers
for the Transformers and speech paths, `serve --threads` for CPU batch work, request batching for embeddings, and the
GPU memory rule to apply before a load. Every figure there is observed on one shared host, not a product figure.

## GPU memory admission

Before a llama-cpp runtime starts (or restarts larger for `options.num_ctx`), Titanos estimates what it needs on each
selected GPU and refuses the start when that does not fit, before it creates a directory, a key or the runtime process:

- Estimate: the offloaded share of the GGUF files (all shards and the projector), the KV cache for the runtime's context
  (from the GGUF header's layer and head counts and the registration's cache types), and a compute margin of 512 MiB or
  10 % of the rest, whichever is larger. These are Titanos's own estimates, not a measurement.
- Limit: the GPU's free memory now (`nvidia-smi`, per UUID) minus `TITANOS_ADMISSION_RESERVE_MIB`, the memory other owners
  of the GPU keep. A resize is credited with the memory the stopped runtime releases. `TITANOS_ADMISSION_FREE_MIB`, when
  nonzero, caps the free memory read (it can lower it, never raise it).
- Refusal: 503 `insufficient_device_memory` with `Retry-After: 30`; the running runtime of a refused resize keeps serving.
- Mode: `TITANOS_NATIVE_ADMISSION=auto` (default) checks when the GGUF header and `nvidia-smi` can be read and otherwise
  admits, recording why in the worker's `admission` (`GET /api/workers`); `on` refuses when either cannot be read; `off`
  never checks. CPU placements are not checked; host memory is not measured.
- Exclusion: `TITANOS_EXCLUDE_GPU_UUIDS` lists GPUs native placement never uses, whatever `CUDA_VISIBLE_DEVICES` allows.

On a host where other services share a GPU, set the mode to `on`, a reserve for them and the excluded GPUs, for example
`TITANOS_NATIVE_ADMISSION=on`, `TITANOS_ADMISSION_RESERVE_MIB=12288` and `TITANOS_EXCLUDE_GPU_UUIDS=GPU-<uuid>`.
