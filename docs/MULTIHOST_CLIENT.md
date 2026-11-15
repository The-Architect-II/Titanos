# Using several standalone servers

An agent can use Titanos's optional Python `Pool` with independent Titanos
servers. Each server retains its own model registry, device assignment, limits,
jobs and artifacts. HiveMind is not required. Machines can use matching
or different operating systems and accelerators if the selected models and
interfaces are compatible.

```python
from contextlib import ExitStack
from titanos.client import Client
from titanos.pool import Endpoint, Pool

with ExitStack() as connections:
    desktop = connections.enter_context(Client("http://desktop:11435", api_key=""))
    laptop = connections.enter_context(Client("http://laptop:11435", api_key=""))
    pool = Pool({
        "desktop": Endpoint(desktop, {"detector": "yolos-desktop"}),
        "laptop": Endpoint(laptop, {"detector": "yolos-laptop"}),
    })
    with pool.lease("detector") as target:
        result = target.client.detect(target.model, "cats.png")
        print(target.host, target.model, result)

    # The same model/task may be pinned to a configured host:
    with pool.lease("detector", host="desktop") as target:
        result = target.client.detect(target.model, "cats.png")
```

Port 11435 is Titanos's default. On a host that also runs HiveMind, its
Ollama-compatible facade may hold 11435, so configure Titanos on another port
there and use that port in the Client URL. A Titanos running in broker role
([BROKER.md](BROKER.md)) is just another server: give it its own `Endpoint`
with a scoped key. Its cloud routing happens inside that server; the Pool stays
entirely client-side and never discovers, probes or balances servers itself.

Configure listening addresses, authentication and reachability separately on
each server. Clients accept API keys explicitly or through their existing
environment setting. Keep real keys outside source and logs. Each Client keeps
its own origin and credential; redirects and foreign-origin asset downloads
remain rejected. Constructing a Pool performs no discovery or network access.

`lease` chooses the configured, enabled matching host with the fewest active
leases in this Pool instance, breaking ties by oldest use. The context releases
the count even when a request raises. It does not retry, send health probes,
switch the selected host after a failure, or perform inference itself. A named
host that lacks the model or is disabled raises before a request; it never
falls back. Host names and server-model aliases appear in the returned Lease.
Only configure interchangeable models under the same logical name when that
substitution suits the workload; a name is not proof of identical weights.

Keep streaming iteration and any job waiting inside the `with` block when
they should count as active. A background job can outlive the block; its GPU
work is then not represented by this client's active count. Retain the Lease
and use its original Client for job status, cancellation, details and artifact
downloads. Never route those follow-ups through a new automatic selection.
Returned Lease values remain useful after the block, while supplied Clients
remain open. The caller owns Client lifetime; the pool never closes a supplied
Client, remote service, process, or model.

`pool.set_enabled("laptop", False)` excludes future leases without interrupting
active work. Re-enabling is explicit. `pool.status()` reports configured aliases,
enabled flags and local active lease counts; these are not remote health,
global load or GPU measurements. This helper has no queue, memory admission,
server proxy, service launcher or distributed tensor execution. Existing server
limits still apply. Cross-machine native qualification is recorded separately.
