"""Scope vocabulary, the route-to-scope table and principal checks.

ROUTE_SCOPES covers every OpenAPI operation of the integrated app (existing and broker routes) plus the mounts outside the
schema: /mcp, the documentation views and the Studio assets. Keys use the FastAPI route-path spelling ('{name:path}' where
the route declares it). Any route not listed requires admin (fail closed). Public routes have no scope.

Umbrella rules (allows): admin covers every scope; inference covers inference:*; cloud covers cloud:<upstream_id> (but not
cloud:store, which grants provider-side state and must be held explicitly, and not native, the R32 native passthrough);
broker:admin covers broker:read; models:write covers models:read. A value 'a|b' is satisfied by either scope.
'authenticated' is satisfied by any principal.

Native passthrough (broker/native.py): /native/<upstream>/<path> and the bare provider paths of the profile allowlists need
'native' here; the handler then adds cloud:<upstream>, cloud:store for stateful rows and broker:admin for account-wide
rows. A bare row marked bare_over_listed is decided before the surfaces and listed routes of the same shape; every other
bare row only after them. Where a model_over_listed or owner-bound bare row shares a listed route's shape (OpenAI
/v1/videos and Titanos' local video jobs), the route needs '<listed scope>|native' here and BrokerDispatch enforces the
listed scope itself when the request stays on the Titanos route.
"""
from __future__ import annotations

import fnmatch
import re

from .naming import UPSTREAM_ID_PATTERN, valid_upstream_id
from .surfaces import literal_weight, match as match_surface, template_regex

INFERENCE_SCOPES = frozenset({'inference:chat', 'inference:embeddings', 'inference:speech', 'inference:transcription',
                              'inference:images', 'inference:moderation', 'inference:plus', 'inference:tensor'})
SCOPES = frozenset({'admin', 'inference', 'mcp', 'jobs', 'models:read', 'models:write', 'settings:read', 'cloud',
                    'cloud:store', 'route:explicit', 'delegate', 'broker:read', 'broker:admin', 'credentials:write',
                    'keys:admin', 'audit:read', 'native'}) | INFERENCE_SCOPES
CLOUD_UPSTREAM_SCOPE = re.compile(r'cloud:(' + UPSTREAM_ID_PATTERN + r')')
AUTHENTICATED = 'authenticated'  # pseudo-scope: any authenticated principal (never granted)
ADMIN = 'admin'

BOOTSTRAP_SCOPES = frozenset({'admin'})
# Keyless loopback (no bootstrap key and no key record): 'cloud', 'native' and 'cloud:store' are added only when
# policy.anonymous_cloud is true in standalone role (anonymous_scopes); provider-side state still needs each upstream's
# data_policy.allow_store, so the operator opts in per upstream (connect allow_store). Never credentials:write, keys:admin, audit:read, route:explicit or delegate
# (the R29 keyless Studio principal adds credentials:write for its audited provider-setup writes only: broker/auth.py).
ANONYMOUS_LOOPBACK_SCOPES = frozenset({'inference', 'jobs', 'mcp', 'models:read', 'models:write', 'settings:read',
                                       'broker:read', 'broker:admin'})
KEYLESS_NEVER_SCOPES = frozenset({'credentials:write', 'keys:admin', 'audit:read', 'route:explicit', 'delegate'})
# `titanos keys create` without --scope mints this key (docs/BROKER.md "Keys, scopes, budgets and audit"): inference plus
# models:read, the scope GET /v1/models needs, so a stock SDK's models.list() works before its first inference call.
DEFAULT_KEY_SCOPES = ('inference', 'models:read')
_IMPLIED = {'broker:admin': 'broker:read', 'models:write': 'models:read'}

# Public routes of every app: server.PUBLIC_ROUTES plus the broker's /v1/health/ready. GET /api/version is added in
# managed role only (role.public_routes).
DOCS_PUBLIC = frozenset({'/docs', '/openapi.json', '/docs/assets/swagger-ui.css', '/docs/assets/swagger-ui-bundle.js'})
STUDIO_PUBLIC = frozenset({'/', '/ui/app.js', '/ui/i18n.js', '/ui/style.css', '/ui/retrieval.js',
                           '/ui/titanos-wordmark-v2.svg', '/ui/titanos-honeycomb-t-v2.svg'})
BASE_PUBLIC_ROUTES = frozenset({'/healthz', '/v1/health/ready'}) | DOCS_PUBLIC | STUDIO_PUBLIC
MANAGED_PUBLIC_ROUTES = frozenset({'/api/version'})

_R = 'models:read'
_W = 'models:write'
_BR = 'broker:read'
_BA = 'broker:admin'

ROUTE_SCOPES: dict[tuple[str, str], str] = {
    # model setup, acquisition and runtime control
    ('GET', '/api/acquisitions'): _R,
    ('POST', '/api/acquisitions'): _W,
    ('POST', '/api/acquisitions/plan'): _R,
    ('DELETE', '/api/acquisitions/{identifier}'): _W,
    ('GET', '/api/acquisitions/{identifier}'): _R,
    ('POST', '/api/acquisitions/{identifier}/activate'): _W,
    ('POST', '/api/acquisitions/{identifier}/retry'): _W,
    ('DELETE', '/api/acquisitions/{identifier}/weights'): _W,
    ('POST', '/api/assignments'): _W,
    ('GET', '/api/assignments/devices'): _R,
    ('GET', '/api/assignments/{model:path}'): _R,
    ('GET', '/api/devices'): _R,
    ('GET', '/api/info'): AUTHENTICATED,  # the settings block inside needs settings:read
    ('GET', '/api/model-providers'): _R,
    ('POST', '/api/model-providers/import'): _W,
    ('POST', '/api/model-providers/inspect'): _W,
    ('GET', '/api/model-providers/{provider}/search'): _R,
    ('POST', '/api/models/capabilities'): _R,
    ('POST', '/api/models/details'): _R,
    ('POST', '/api/models/load'): _W,
    ('GET', '/api/models/loaded'): _R,
    ('GET', '/api/models/memory'): _R,
    ('POST', '/api/models/register'): _W,
    ('DELETE', '/api/models/registrations/{name:path}'): _W,
    ('POST', '/api/models/reload'): _W,
    ('POST', '/api/models/unload'): _W,
    ('GET', '/api/ps'): _R,
    ('GET', '/api/runtime-profiles'): _R,
    ('GET', '/api/runtime-setups'): _R,
    ('POST', '/api/runtime-setups'): _W,
    ('POST', '/api/runtime-setups/plan'): _R,
    ('DELETE', '/api/runtime-setups/{identifier}'): _W,
    ('GET', '/api/runtime-setups/{identifier}'): _R,
    ('POST', '/api/runtime-setups/{identifier}/retry'): _W,
    ('GET', '/api/tags'): _R,
    ('GET', '/api/workers'): _R,
    ('GET', '/readyz'): _R,
    ('GET', '/v1/models'): _R,
    ('GET', '/v1/models/{name:path}'): _R,
    ('GET', '/v1/extensions'): _R,
    # Ollama
    ('POST', '/api/chat'): 'inference:chat',
    ('POST', '/api/generate'): 'inference:chat',
    ('POST', '/api/embed'): 'inference:embeddings',
    ('POST', '/api/embeddings'): 'inference:embeddings',
    # Ollama client surface (protocols.py): /api/show reads the registry; the library routes are typed 400 refusals that
    # keep the scope the real operation would need (GET /api/version is listed above with the role rows)
    ('POST', '/api/show'): _R,
    ('POST', '/api/pull'): _W,
    ('POST', '/api/create'): _W,
    ('POST', '/api/copy'): _W,
    ('DELETE', '/api/delete'): _W,
    # tensors
    ('POST', '/api/infer'): 'inference:tensor',
    # jobs and artifacts (owner-scoped in the handlers)
    ('GET', '/api/jobs'): 'jobs',
    ('POST', '/api/jobs'): 'jobs',
    ('DELETE', '/api/jobs/{identifier}'): 'jobs',
    ('GET', '/api/jobs/{identifier}'): 'jobs',
    ('GET', '/api/video/jobs'): 'jobs',
    ('POST', '/api/video/jobs'): 'jobs',
    ('DELETE', '/api/video/jobs/{identifier}'): 'jobs',
    ('GET', '/api/video/jobs/{identifier}'): 'jobs',
    ('POST', '/v1/music/generations'): 'jobs',
    ('GET', '/v1/videos'): 'jobs',
    ('POST', '/v1/videos'): 'jobs',
    ('POST', '/v1/videos/generations'): 'jobs',
    ('DELETE', '/v1/videos/{identifier}'): 'jobs',
    ('GET', '/v1/videos/{identifier}'): 'jobs',
    ('GET', '/v1/videos/{identifier}/content'): 'jobs',
    ('GET', '/v1/files/{name}'): AUTHENTICATED,  # artifacts bound to a key are visible to that key only
    # OpenAI-compatible inference
    ('POST', '/v1/chat/completions'): 'inference:chat',
    ('POST', '/v1/completions'): 'inference:chat',
    ('POST', '/v1/responses'): 'inference:chat',
    ('GET', '/v1/responses/{response_id}'): 'inference:chat',
    ('DELETE', '/v1/responses/{response_id}'): 'inference:chat',
    ('POST', '/v1/responses/{response_id}/cancel'): 'inference:chat',
    ('GET', '/v1/responses/{response_id}/input_items'): 'inference:chat',
    ('POST', '/v1/embeddings'): 'inference:embeddings',
    ('POST', '/v1/audio/speech'): 'inference:speech',
    ('POST', '/v1/audio/transcriptions'): 'inference:transcription',
    ('POST', '/v1/audio/translations'): 'inference:transcription',
    ('POST', '/v1/images/generations'): 'inference:images',
    ('POST', '/v1/images/edits'): 'inference:images',
    ('POST', '/v1/moderations'): 'inference:moderation',
    # API Plus
    ('POST', '/v1/audio/analysis'): 'inference:plus',
    ('POST', '/v1/audio/diarizations'): 'inference:plus',
    ('POST', '/v1/audio/vad'): 'inference:plus',
    ('POST', '/v1/classifications/text-pairs'): 'inference:plus',
    ('POST', '/v1/extensions/infer'): 'inference:plus',
    ('POST', '/v1/images/depth'): 'inference:plus',
    ('POST', '/v1/images/detect'): 'inference:plus',
    ('POST', '/v1/images/ocr'): 'inference:plus',
    ('POST', '/v1/images/segment'): 'inference:plus',
    ('POST', '/v1/rerank'): 'inference:plus',
    ('POST', '/v1/retrieval/answer'): 'inference:plus',
    ('POST', '/v1/sentiment'): 'inference:plus',
    ('POST', '/v1/sentiment/compose'): 'inference:plus',
    ('POST', '/v1/translations'): 'inference:plus',
    # Anthropic
    ('POST', '/v1/messages'): 'inference:chat',
    ('POST', '/v1/messages/count_tokens'): 'inference:chat',
    # ElevenLabs
    ('POST', '/v1/text-to-speech/{voice_id}'): 'inference:speech',
    ('POST', '/v1/text-to-speech/{voice_id}/stream'): 'inference:speech',
    ('POST', '/v1/speech-to-text'): 'inference:transcription',
    ('GET', '/v1/voices'): _R,
    ('GET', '/v1/voices/{voice_id}'): _R,
    # Gemini (the POST action decides chat or embeddings: see required_scope)
    ('GET', '/v1beta/models'): _R,
    ('GET', '/v1beta/models/{model:path}'): _R,
    ('POST', '/v1beta/models/{model_action:path}'): 'inference:chat',
    # broker status and control
    ('GET', '/api/broker'): _BR,
    ('POST', '/api/broker/reload'): _BA,
    ('GET', '/api/broker/profiles'): _BR,
    ('GET', '/api/broker/upstreams'): _BR,
    ('GET', '/api/broker/upstreams/{id}'): _BR,
    ('PUT', '/api/broker/upstreams/{id}'): _BA,
    ('DELETE', '/api/broker/upstreams/{id}'): _BA,
    ('POST', '/api/broker/upstreams/{id}/test'): _BA,
    ('GET', '/api/broker/upstreams/{id}/health'): _BR,
    ('POST', '/api/broker/upstreams/{id}/breaker/reset'): _BA,
    ('GET', '/api/broker/credentials'): 'credentials:write|broker:admin',
    ('PUT', '/api/broker/credentials/{id}'): 'credentials:write',
    ('POST', '/api/broker/credentials/{id}/rotate'): 'credentials:write',
    ('POST', '/api/broker/credentials/{id}/validate'): 'credentials:write',
    ('DELETE', '/api/broker/credentials/{id}'): 'credentials:write',
    ('GET', '/api/broker/models'): _BR,
    ('GET', '/api/broker/models/{name:path}'): _BR,
    ('PUT', '/api/broker/models/{name:path}'): _BA,
    ('DELETE', '/api/broker/models/{name:path}'): _BA,
    ('GET', '/api/broker/virtual-models'): _BR,
    ('GET', '/api/broker/virtual-models/{name:path}'): _BR,
    ('PUT', '/api/broker/virtual-models/{name:path}'): _BA,
    ('DELETE', '/api/broker/virtual-models/{name:path}'): _BA,
    ('GET', '/api/broker/policy'): _BR,
    ('PUT', '/api/broker/policy'): _BA,
    ('GET', '/api/broker/advisors'): _BR,
    ('PUT', '/api/broker/advisors/{id}'): _BA,
    ('POST', '/api/broker/advisors/{id}/test'): _BA,
    ('GET', '/api/broker/prices'): _BR,
    ('PUT', '/api/broker/prices'): _BA,
    ('GET', '/api/broker/catalog'): _BR,
    ('POST', '/api/broker/explain'): _BR,
    ('GET', '/api/broker/env-candidates'): _BA,  # names and booleans only (R30); keyless needs the R29 Studio header
    ('POST', '/api/broker/connect'): _BA,  # also credentials:write (checked in the handler, R31)
    ('GET', '/api/broker/decisions'): _BR,
    ('GET', '/api/broker/usage'): _BR,
    ('GET', '/api/broker/metrics'): _BR,
    # inbound keys
    ('GET', '/api/keys'): 'keys:admin',
    ('POST', '/api/keys'): 'keys:admin',
    ('GET', '/api/keys/self'): AUTHENTICATED,
    ('GET', '/api/keys/{id}'): 'keys:admin',
    ('PATCH', '/api/keys/{id}'): 'keys:admin',
    ('POST', '/api/keys/{id}/revoke'): 'keys:admin',
    ('POST', '/api/keys/{id}/rotate'): 'keys:admin',
    # audit
    ('GET', '/api/audit'): 'audit:read',
    ('GET', '/api/audit/verify'): 'audit:read',
    # registered only in managed role, where it is public; elsewhere it is absent (404)
    ('GET', '/api/version'): AUTHENTICATED,
    # mounts and views outside the OpenAPI schema; the exact public assets are answered before this table
    ('*', '/mcp'): 'mcp',
    ('GET', '/docs/assets/{name}'): AUTHENTICATED,
    ('GET', '/ui/{name}'): AUTHENTICATED,
}

_COMPILED = tuple(sorted(((method, template, template_regex(template), scope) for (method, template), scope in
                          ROUTE_SCOPES.items()), key=lambda row: -literal_weight(row[1])))


def normalize_template(template: str) -> str:
    """The OpenAPI spelling of a route template ('{name:path}' -> '{name}')."""
    return re.sub(r'\{([A-Za-z_][A-Za-z0-9_]*):path\}', r'{\1}', template)


def scope_for_template(method: str, template: str) -> str | None:
    """The ROUTE_SCOPES entry for an app route template in either the route or the OpenAPI spelling, or None if unlisted."""
    wanted = (method.upper(), normalize_template(template))
    for (row_method, row_template), scope in ROUTE_SCOPES.items():
        if (row_method, normalize_template(row_template)) == wanted:
            return scope
    return None


def required_scope(method: str, path: str, *, public_routes: frozenset[str] | None = None) -> str | None:
    """The scope a request needs before its body is read: None for public routes (every method), 'mcp' for the /mcp
    mount, the surface scope for broker surfaces (Gemini actions split on the last ':'), the ROUTE_SCOPES entry for listed
    routes, and 'admin' for anything unlisted (fail closed).

    A public path needs no scope for any method, so an unauthenticated HEAD, OPTIONS or POST to it keeps today's answer:
    the Guard's public framing (405 method_not_allowed for a method outside server.PUBLIC_METHODS, 400 unexpected_body)
    runs from framing alone and must run before any 401/403 decided from this scope."""
    method = method.upper()
    public = BASE_PUBLIC_ROUTES if public_routes is None else public_routes
    normalized = path.rstrip('/') or '/'
    if normalized in public:
        return None
    if normalized == '/mcp' or path.startswith('/mcp/'):
        return 'mcp'
    from .native import NATIVE_PREFIX, bare_may_win, bare_over_listed, bare_rows
    if path.startswith(NATIVE_PREFIX):
        return 'native'
    found = match_surface(method, path)
    if found is not None and found[0].kind != 'catalog':
        return found[0].scope
    if bare_over_listed(method, path):
        return 'native'
    if found is not None:
        return found[0].scope
    listed = listed_scope(method, path)
    if listed is not None:
        if listed != AUTHENTICATED and bare_may_win(method, path):
            # the provider's row wins for some requests (a cloud model in the body, an id bound to the caller):
            # either scope passes here; BrokerDispatch enforces the listed scope when the Titanos route answers
            return listed + '|native'
        return listed
    if bare_rows(method, path) or (method == 'GET' and bare_rows('WS', path)):
        return 'native'  # a GET on a WebSocket row is the upgrade request (answered by BrokerDispatch)
    return ADMIN


def listed_scope(method: str, path: str) -> str | None:
    """The ROUTE_SCOPES entry whose template matches (method, path), or None (no Titanos route of that shape)."""
    method = method.upper()
    normalized = path.rstrip('/') or '/'
    for row_method, _template, pattern, scope in _COMPILED:
        if row_method in (method, '*') and pattern.match(normalized):
            return scope
    return None


def valid_scope(scope: str) -> bool:
    """A grantable scope: the vocabulary or cloud:<upstream_id>."""
    if scope in SCOPES:
        return True
    found = CLOUD_UPSTREAM_SCOPE.fullmatch(scope) if isinstance(scope, str) else None
    return bool(found) and valid_upstream_id(found.group(1))


def allows(principal, scope: str | None) -> bool:
    """Whether principal holds scope under the umbrella rules (see the module docstring)."""
    if scope is None or scope == AUTHENTICATED:
        return principal is not None
    if principal is None:
        return False
    held = principal.scopes
    if ADMIN in held:
        return True
    for wanted in scope.split('|'):
        if wanted in held:
            return True
        if wanted.startswith('inference:') and 'inference' in held:
            return True
        if wanted.startswith('cloud:') and wanted != 'cloud:store' and 'cloud' in held:
            return True
        if any(implied == wanted and source in held for source, implied in _IMPLIED.items()):
            return True
    return False


def anonymous_scopes(*, anonymous_cloud: bool, role: str) -> frozenset[str]:
    """Scopes of the keyless loopback principal: cloud, native and cloud:store only when policy.anonymous_cloud is true
    in standalone role (the keyless loopback caller is the machine's owner; storing at a provider still needs the
    upstream's data_policy.allow_store)."""
    return ANONYMOUS_LOOPBACK_SCOPES | ({'cloud', 'native', 'cloud:store'} if anonymous_cloud and role == 'standalone'
                                       else set())


def _matches(patterns, value: str) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def can_use_target(principal, target) -> bool:
    """Visibility of a target to a principal (an invisible target is answered 404 model_not_found, never 403): the key's
    target-class, model and upstream constraints, and cloud or cloud:<upstream> for cloud and router targets."""
    if principal is None:
        return False
    constraints = principal.constraints
    if target.klass not in constraints.target_classes:
        return False
    if not _matches(constraints.models, target.ref):
        return False
    if target.klass == 'local':
        return True
    if not target.upstream_id or not _matches(constraints.upstreams, target.upstream_id):
        return False
    return allows(principal, 'cloud:' + target.upstream_id)


__all__ = ['ADMIN', 'ANONYMOUS_LOOPBACK_SCOPES', 'AUTHENTICATED', 'BASE_PUBLIC_ROUTES', 'BOOTSTRAP_SCOPES', 'DEFAULT_KEY_SCOPES',
           'DOCS_PUBLIC', 'INFERENCE_SCOPES', 'KEYLESS_NEVER_SCOPES', 'MANAGED_PUBLIC_ROUTES', 'ROUTE_SCOPES', 'SCOPES', 'STUDIO_PUBLIC',
           'allows', 'anonymous_scopes', 'can_use_target', 'listed_scope', 'normalize_template', 'required_scope',
           'scope_for_template', 'valid_scope']
