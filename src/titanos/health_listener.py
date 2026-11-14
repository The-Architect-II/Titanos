"""A health listener beside `titanos serve`, outside uvicorn's concurrency cap.

uvicorn answers 503 to every request once TITANOS_LIMIT_CONCURRENCY connections and tasks are open, health probes
included, so a busy server looks dead to its supervisor or to a gateway's health poller. With TITANOS_HEALTH_PORT set,
`titanos serve` runs a second uvicorn server in the same process and event loop, on the same host, with no concurrency
limit and no lifespan of its own. It passes only HEALTH_PATHS to the same application (its middleware, auth and handlers)
and answers 404 to any other path. Probes are access-logged like any other request: uvicorn decides that per process,
from the handlers of the uvicorn.access logger, not per server. The main server's lifespan and logging configuration own
startup and shutdown; when the main server stops, the health server stops too, and only the main server handles
signals. A health port that cannot be bound stops the process, as a main port that cannot be bound does.
"""
import asyncio
import contextlib
import json
import sys
from types import SimpleNamespace

import uvicorn

HEALTH_PATHS = frozenset({'/healthz', '/readyz', '/v1/health/ready'})
NOT_HEALTH = json.dumps({'error': {'message': 'Only /healthz, /readyz and /v1/health/ready are served on the health port',
                                   'type': 'not_found', 'param': None, 'code': 'not_found'}}).encode()
SHARED_OPTIONS = ('host', 'ssl_certfile', 'ssl_keyfile', 'log_level', 'timeout_keep_alive')


def health_app(app):
    """An ASGI application that forwards HEALTH_PATHS to app and refuses everything else."""
    async def health(scope, receive, send):
        if scope['type'] == 'http' and scope.get('path') in HEALTH_PATHS:
            await app(scope, receive, send)
        elif scope['type'] == 'http':
            await send({'type': 'http.response.start', 'status': 404,
                        'headers': [(b'content-type', b'application/json'), (b'content-length', str(len(NOT_HEALTH)).encode())]})
            await send({'type': 'http.response.body', 'body': NOT_HEALTH})
        elif scope['type'] == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
    return health


class HealthServer(uvicorn.Server):
    """The health server: the main server owns SIGINT/SIGTERM, and this one stops when it does."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


class ServerWithHealth(uvicorn.Server):
    """The main server, serving the health server in the same event loop for as long as it runs."""

    def __init__(self, config, health):
        super().__init__(config)
        self.health = health

    async def serve(self, sockets=None):
        task = asyncio.create_task(self.health.serve())
        try:
            await super().serve(sockets)
        finally:
            self.health.should_exit = True
            await asyncio.gather(task, return_exceptions=True)


def server(app, health_port, **options):
    """The (not yet started) server pair for uvicorn.run-style options: the main server with options, the health server
    on health_port with the same host, TLS, log level and keep-alive, no concurrency limit and no lifespan. Its Config
    leaves logging alone (log_config=None): the main server's Config configures it once."""
    shared = {key: options[key] for key in SHARED_OPTIONS if key in options}
    health = HealthServer(uvicorn.Config(health_app(app), port=health_port, lifespan='off', limit_concurrency=None,
                                         log_config=None, **shared))
    return ServerWithHealth(uvicorn.Config(app, **options), health)


def runner(health_port):
    """A drop-in for uvicorn.run(app, **options) that also serves health_app(app) on health_port."""
    def run(app, **options):
        pair = server(app, health_port, **options)
        try:
            pair.run()
        except KeyboardInterrupt:
            pass
        if not pair.started:
            sys.exit(3)  # uvicorn's STARTUP_FAILURE, as uvicorn.run exits
    return run


def with_health(module, health_port, **extra):
    """What `titanos serve` calls .run on: the uvicorn module itself without a health port or extra options, else a
    stand-in whose run serves the health listener beside the main server (runner) and adds extra to the main server's
    options (log_config from TITANOS_LOG_FILE, titanos.log_rotation)."""
    if not health_port and not extra:
        return module
    run = runner(health_port) if health_port else module.run
    return SimpleNamespace(run=lambda app, **options: run(app, **options, **extra))


__all__ = ['HEALTH_PATHS', 'HealthServer', 'ServerWithHealth', 'health_app', 'runner', 'server', 'with_health']
