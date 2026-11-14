"""Explicit leases over independent Titanos servers, without request replay."""
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from types import MappingProxyType
from typing import Mapping

from .client import Client


@dataclass(frozen=True)
class Endpoint:
    """A caller-owned Client and logical-name to server-model bindings."""
    client: Client
    models: Mapping[str, str]

    def __post_init__(self):
        models = dict(self.models)
        if not models or any(not isinstance(k, str) or not k.strip() or
                             not isinstance(v, str) or not v.strip()
                             for k, v in models.items()):
            raise ValueError('Each endpoint needs nonempty logical and server model names')
        if not isinstance(self.client, Client):
            raise TypeError('Endpoint requires a Titanos Client')
        object.__setattr__(self, 'models', MappingProxyType(models))


@dataclass(frozen=True)
class Lease:
    """Keep this origin/model for streams, artifact reads and job follow-ups."""
    host: str
    model: str
    client: Client


class Pool:
    """Least-active lease selection with oldest-use tie breaking.

    This scheduler counts leases issued by this Pool instance only. It neither
    discovers GPU availability nor owns the remote servers or supplied Clients.
    No network call, health probe, model operation or retry is implicit.
    """
    def __init__(self, endpoints: Mapping[str, Endpoint]):
        endpoints = dict(endpoints)
        if not endpoints or any(not isinstance(k, str) or not k.strip() or
                                not isinstance(v, Endpoint) for k, v in endpoints.items()):
            raise ValueError('Pool requires named Endpoint entries')
        self._endpoints = MappingProxyType(endpoints)
        self._active = dict.fromkeys(endpoints, 0)
        self._last_used = dict.fromkeys(endpoints, 0)
        self._sequence = 0
        self._disabled = set()
        self._lock = threading.Lock()

    def set_enabled(self, host: str, enabled: bool):
        """Exclude/include future leases; active requests are never interrupted."""
        if type(enabled) is not bool:
            raise ValueError('enabled must be a boolean')
        with self._lock:
            if host not in self._endpoints:
                raise ValueError(f'Unknown pool host: {host}')
            if enabled:
                self._disabled.discard(host)
            else:
                self._disabled.add(host)

    def status(self):
        """Local scheduling metadata; active counts are not remote GPU usage."""
        with self._lock:
            return [{'host': host, 'models': dict(endpoint.models),
                     'enabled': host not in self._disabled,
                     'active_leases': self._active[host]}
                    for host, endpoint in self._endpoints.items()]

    @contextmanager
    def lease(self, model: str, *, host: str | None = None):
        """Select exactly once; caller exceptions propagate without replay.

        Keep streaming iteration and desired job waiting inside the with block
        to count them as active. Saved Lease values retain their original Client
        and model for later artifact/job access after scheduling is released.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model must be a nonempty logical model name')
        if host is not None and (not isinstance(host, str) or not host.strip()):
            raise ValueError('host must be a nonempty configured host name')
        with self._lock:
            if host is not None and host not in self._endpoints:
                raise ValueError(f'Unknown pool host: {host}')
            candidates = [key for key, endpoint in self._endpoints.items()
                          if key not in self._disabled and model in endpoint.models
                          and (host is None or key == host)]
            if not candidates:
                raise ValueError(f'No enabled endpoint is configured for model {model!r}' +
                                 (f' on host {host!r}' if host is not None else ''))
            selected = min(candidates, key=lambda key: (self._active[key], self._last_used[key]))
            self._sequence += 1
            self._last_used[selected] = self._sequence
            self._active[selected] += 1
            endpoint = self._endpoints[selected]
            lease = Lease(selected, endpoint.models[model], endpoint.client)
        try:
            yield lease
        finally:
            with self._lock:
                self._active[selected] -= 1
