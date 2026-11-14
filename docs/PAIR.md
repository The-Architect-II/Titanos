# NVIDIA PAIR integration

What has actually run against upstream
[NVIDIA/Personal-AI-Router at 13b68115fa2c9c1d94f1ead1358f8d5a527cfecf](https://github.com/NVIDIA/Personal-AI-Router/tree/13b68115fa2c9c1d94f1ead1358f8d5a527cfecf)
is exactly one recorded check: the Titanos 0.4.0 text bridge on Linux x86-64 CPU, in which the unmodified upstream
engine-manager and manual-node proxy forwarded real text requests through `tests/check_pair.py`
([VALIDATION.md, 0.4.0](VALIDATION.md#040-linux-x86-64-cpu)). No upstream source changes were needed for that run.
Nothing else on this page has been run against PAIR: not the current source, not Windows or macOS, and not the manifest
that `titanos pair-manifest` writes today, whose runtime arguments carry `--role managed` (the identity block is
byte-identical to the 0.4.0 one). Until the check at the end of this page is re-run with the current manifest, treat
the engine-manager lifecycle below as described, not tested.

## Engine-manager lifecycle

Install Titanos, register a model, then generate a manifest:

```sh
titanos pair-manifest --output titanos.json
```

Copy that file into PAIR's per-user `engines` directory and restart its engine
manager. The generated manifest targets the OS, installed executable, and registry
on the machine where it was generated; regenerate it after moving the installation.

| OS | Engine manifest directory |
|---|---|
| Linux | `$XDG_CONFIG_HOME/Nvidia Corporation/Personal AI Router/engines`, or `~/.config/Nvidia Corporation/Personal AI Router/engines` |
| Windows | `%LOCALAPPDATA%\Nvidia Corporation\Personal AI Router\engines` |
| macOS | `~/Library/Application Support/Nvidia Corporation/Personal AI Router/engines` |

PAIR's engine-manager JSON-RPC interface can then use:

```json
{"jsonrpc":"2.0","id":1,"method":"engine:start","params":{"engine":"titanos"}}
{"jsonrpc":"2.0","id":2,"method":"engine:action","params":{"engine":"titanos","action":"load_model","params":{"model":"chat"}}}
{"jsonrpc":"2.0","id":3,"method":"engine:models","params":{}}
{"jsonrpc":"2.0","id":4,"method":"engine:action","params":{"engine":"titanos","action":"unload_model","params":{"model":"chat"}}}
{"jsonrpc":"2.0","id":5,"method":"engine:stop","params":{"engine":"titanos"}}
```

The manifest uses PAIR's foreground-process lifecycle. Its probes use process
health, since an idle Titanos instance intentionally has no resident model.
Load, unload, and inventory use ordinary HTTP actions.

## Existing OpenAI proxy bridge

The upstream LM Studio proxy accepts manually registered OpenAI-compatible nodes.
Send the following to that proxy's JSON-RPC control channel:

```json
{"jsonrpc":"2.0","id":1,"method":"node/add-manual","params":{"id":"titanos-local","host":"127.0.0.1","port":<TITANOS_PORT>,"addresses":["127.0.0.1"],"models":["chat","embed"]}}
{"jsonrpc":"2.0","id":2,"method":"node/select","params":{"id":"titanos-local"}}
```

Replace `<TITANOS_PORT>` with the port that Titanos instance is configured to
serve on, and supply the actual Titanos model IDs. An empty model inventory is not
eligible for model-bearing inference. Update the manual node after changing
registrations. Applications then send their usual `/v1/chat/completions` requests
to the proxy's listening port.

This is a compatibility bridge through PAIR's existing LM Studio routing path.
It does not add a native Titanos desktop tile or new multimodal scheduling rules.
Direct Titanos clients can use all implemented modalities; upstream routing
policies still determine which operations PAIR recognizes.

The tested bridge uses a dedicated loopback Titanos instance without
`TITANOS_API_KEY`. The current engine-manager action schema does not express
custom authentication headers. Do not expose that instance directly on the LAN.
Titanos requires an API key for non-loopback binds; a LAN integration must carry
authentication correctly or use the enclosing trusted proxy transport.

A keyed bridge uses a scoped Titanos key (`titanos keys create --name pair
--scope inference --scope models:read`) rather than the bootstrap key, so the
bridge can run inference but cannot manage keys or credentials.

A Titanos started through `titanos pair-manifest` runs in managed role
([TOPOLOGY_ROLES.md](TOPOLOGY_ROLES.md)): without parent delegation it holds no
usable provider credentials, and a keyless managed instance makes no outbound
connection at all, so PAIR-driven calls to it are answered by local models only.

`titanos pair-manifest` now adds `--role managed` to the runtime arguments; the
identity block is unchanged. That changed manifest has never been run against
PAIR (the only recorded run is the 0.4.0 Linux CPU text bridge named at the top
of this page): re-run the check below before relying on it. The manifest also
carries an `integration` block (`enabled_by_default`, facade and backend ports,
port file, identity route); whether PAIR's loader reads it has not been checked
either way.

## Reproduce the upstream check

Build the two upstream Go services from the pinned revision, install Titanos, and run:

```sh
python tests/check_pair.py \
  --engine-manager /path/to/nvpair-engine-manager \
  --proxy /path/to/lmstudio-proxy \
  --model-dir /path/to/SmolLM2-135M-Instruct \
  --report pair-check.json
```

The check isolates PAIR configuration in a temporary XDG directory on Linux,
starts Titanos through PAIR, loads a real checkpoint, routes a real chat request
through the proxy, unloads the model, and stops the owned process.

Titanos 0.4 adds native tool/Responses support and video jobs for direct clients.
The existing PAIR proxy bridge has not thereby gained new scheduling policies or
protocol guarantees; qualify each desired PAIR route on its upstream version.
