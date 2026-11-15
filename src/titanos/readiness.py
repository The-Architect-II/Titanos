"""Truthful readiness for the public GET /v1/health/ready probe (broker/routes.py health_ready).

Orchestrators (HiveMind, load balancers) poll the route without a key, so it must say 503 whenever a request on the
inference routes would not be served, and 200 only when it would:
- the registry is loaded and not being replaced (reload_registry in server.py sets gate['reloading'] during a swap and
  gate['unavailable'] / EngineSlot.unavailable after a failed one, which makes every engine route answer 503
  registry_unavailable until a reload succeeds or the server restarts);
- in broker role, the broker configuration loaded (snapshot.load_error is None) and every required upstream is
  healthy. The required upstreams are the enabled upstreams of broker.json; an upstream is unhealthy while its circuit
  breaker is open (monitor.status(id)['breaker'] == 'open'). A broker with no enabled upstream requires none and is
  ready (standalone and managed roles never consult upstreams here: a managed server's cloud use is parent-delegated).
The 503 body names one typed reason (reason, message) next to the full checks block, so a poller can tell a reload
from an upstream outage without a key. uvicorn binds its socket only after the lifespan startup, so "starting" is not
observable over HTTP and is not a state here.
"""
from __future__ import annotations

from fastapi.responses import JSONResponse

from . import __version__

UNHEALTHY_BREAKER_STATES = frozenset({'open'})


def report(broker, *, gate, config, engine, role) -> tuple[int, dict]:
    """(status code, body) of the readiness probe. gate: the server's {'active', 'reloading', 'unavailable'} dict
    (None for an app that has none); engine: the EngineSlot; config: the registry Config (mutated in place on reload)."""
    gate = gate or {}
    reasons: list[tuple[str, str]] = []
    reloading = bool(gate.get('reloading'))
    unavailable = bool(gate.get('unavailable')) or bool(getattr(engine, 'unavailable', False))
    checks: dict = {'registry': 'reloading' if reloading else 'loaded',
                    'engine': 'unavailable' if unavailable else 'available'}
    if reloading:
        reasons.append(('registry_reloading', 'The model registry is being replaced; retry shortly'))
    elif unavailable:
        reasons.append(('registry_unavailable', 'The model registry is unavailable after a failed reload; '
                                                'POST /api/models/reload or restart the server'))
    if role == 'broker':
        checks['upstreams'] = _upstream_checks(broker, reasons)
    models = getattr(config, 'models', None)
    body = {'status': 'ready' if not reasons else 'not_ready', 'version': __version__, 'role': role,
            'models': len(models) if models is not None else 0, 'checks': checks}
    if reasons:
        body['reason'], body['message'] = reasons[0]
    return (200 if not reasons else 503), body


def _upstream_checks(broker, reasons: list) -> dict:
    try:
        snapshot = broker.snapshot()
    except Exception as exc:  # a broker.json that cannot be read: nothing can be routed, the reason says why
        reasons.append(('broker_config_unavailable', 'The broker configuration could not be loaded: ' + type(exc).__name__))
        return {'required': [], 'unhealthy': [], 'error': type(exc).__name__}
    load_error = getattr(snapshot, 'load_error', None)
    required = sorted(snapshot.config.enabled_upstreams())
    states = {}
    for upstream_id in required:
        try:
            states[upstream_id] = broker.upstream_health(upstream_id).get('breaker', 'closed')
        except Exception as exc:  # no breaker board yet: nothing has failed
            states[upstream_id] = 'closed (' + type(exc).__name__ + ')'
    unhealthy = sorted(uid for uid, state in states.items() if state in UNHEALTHY_BREAKER_STATES)
    out = {'required': required, 'unhealthy': unhealthy, 'breakers': states}
    if load_error:
        out['error'] = load_error
        reasons.append(('broker_config_unavailable', 'The broker configuration did not load: ' + str(load_error)))
    elif unhealthy:
        reasons.append(('upstream_unhealthy', 'Required upstreams are unhealthy (circuit breaker open): '
                        + ', '.join(unhealthy)))
    return out


def response(broker, *, gate, config, engine, role) -> JSONResponse:
    """The JSONResponse of GET /v1/health/ready (200 ready, 503 not_ready with reason and message)."""
    status, body = report(broker, gate=gate, config=config, engine=engine, role=role)
    headers = {'Cache-Control': 'no-store'}
    if status != 200:
        headers['Retry-After'] = '1'
    return JSONResponse(body, status_code=status, headers=headers)


__all__ = ['UNHEALTHY_BREAKER_STATES', 'report', 'response']
