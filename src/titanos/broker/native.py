"""R32 guarded native passthrough: provider endpoints that no typed surface covers.

Path shape (chosen in the 2026-10-02 completeness round):
- /native/<upstream_id>/<path> for every allowlisted endpoint of the upstream's profile; <path> is relative to the
  profile's native root (profiles.NativeSpec.root: the upstream base_url for the OpenAI family, so an SDK sets
  base_url=http://host:11435/native/openai and keeps its own relative paths; the origin for Anthropic, Gemini and
  ElevenLabs, whose SDKs send /v1/..., /v1beta/..., /upload/v1beta/... and /v2/...). WebSocket endpoints (method 'WS')
  use the same shape.
- the provider's own public path (a "bare" path: NativeSpec.bare_prefix + row path, for example POST /v1/files or
  POST /v1/messages/batches), so an unmodified SDK pointed at Titanos reaches it. A bare path is served only when it is not
  a Titanos inference surface, not a Titanos route of the same method and shape (unless the row sets bare_over_listed, for
  example the Gemini model actions Titanos has no surface for), and exactly one enabled upstream the caller may see offers
  it; the X-Titanos-Upstream header names the upstream otherwise (and lets a bare path win over a listed Titanos route of
  the same shape, never over an inference surface). A row marked vendor_over_listed also wins over a listed route when
  the request carries that profile's own SDK key header and no Authorization header (ElevenLabs GET /v1/models). An
  owner-bound row also wins over a listed route when the id in the path is a native object bound to the caller
  (OpenAI GET /v1/files/{id} for a file uploaded through Titanos; any other id keeps the local artifact route). A
  model_over_listed row wins over a listed route when the body's model names a cloud model of a visible upstream that
  offers it (choose_by_model): OpenAI POST /v1/videos with sora-2, sora-2@<upstream> or an alias of it reaches OpenAI
  (and the created video id is bound to the caller, so retrieve, content and delete on /v1/videos/{id} follow it), while a
  local model keeps Titanos' local video job. Such paths need the listed route's scope or 'native' at the Guard; a
  request that stays on the Titanos route still needs the listed scope (BrokerDispatch answers 403 scope_denied).

Rules (enforced by the functions below and broker.dispatch, all before any egress):
- scopes: 'native' plus cloud:<upstream> (or cloud); 'cloud' never implies 'native' (R1-style); rows with store=True also
  need cloud:store and the upstream's data_policy.allow_store; account-wide rows (owner 'list' or admin=True) need
  broker:admin for key principals; the key's model globs apply to the ref native@<upstream>;
- the same role, delegation, hop, location, cloud-policy and readiness gates as the main path (managed role: refused
  unless delegated, and a delegated bare path must name its upstream);
- method and path allowlisted per profile row (profiles.NativeSpec.routes); anything else is 400
  broker_capability_unavailable before egress; dot and empty segments (also inside a decoded {x:path} parameter, and
  again after a model rewrite of the path) and the encoded
  separators %2F, %5C, %3F, %23 and %2E%2E anywhere in the raw path are refused (400 invalid_request), and so is a raw
  path whose prefix does not match its decoded form (a percent-encoded upstream id or /v1), so the row matched is always
  the path sent;
- model names: a JSON body's top-level model, each requests[].params.model (Anthropic Message Batches) and a {model}
  path parameter (Gemini :predict) that name a Titanos ref are rewritten to the provider's own id before egress: a
  qualified ref of this upstream (model@<upstream>) or an alias that resolves to it loses the suffix; a ref of another
  upstream, or a local model name on a bare path, is 400 invalid_request (normalize_models);
- credentials: injected by the egress client only, for exactly that upstream's origin; the inbound Authorization, vendor
  key headers, cookies and every X-Titanos-* header are never forwarded; inbound headers pass only from
  INBOUND_HEADER_ALLOW, content-type, accept and the row's request_headers; the account-selecting headers
  OpenAI-Organization and OpenAI-Project (ACCOUNT_HEADERS) pass only for broker:admin principals and the keyless
  loopback owner (another key could otherwise pick an organization or project the operator's credential reaches);
  credential-named query parameters are refused (401 query_credential_refused);
- answers: status and body as the provider sent them (2xx streamed through untouched, errors scrubbed), response headers
  allowlisted (adapters' list plus the row's response_headers); a provider 401 (or a 403 whose body says the key is
  invalid) is 502 upstream_auth_failed, never the client's own 401 (an ElevenLabs quota_exceeded 401 is answered 402
  with the scrubbed provider body); a rewritten resumable-upload URL (Gemini X-Goog-Upload-URL) carries an upload
  capability (UploadSessions: an unguessable token bound to the caller, the upstream and that upload session), so the
  SDK's follow-up chunks, which carry no key, are accepted for that one session only; JSON answers of a profile with
  rewrite_body_fields
  get those upstream URLs pointed at this server's /native/<upstream>/ path (Anthropic's batch results_url, which the
  SDK follows with the caller's key); an Anthropic Message Batch create body is checked like a Messages body for
  provider-side state (broker/anthropic_state.py);
- size caps: the request body is bounded by TITANOS_MAX_BODY_BYTES (the Guard); every answer by
  TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES;
- ownership: ids created through a 'create' row are bound to the calling key in handles.json (kind 'native'); 'bound' rows
  answer 404 not_found to any other key (admin principals and the keyless loopback operator may use any id);
- audit: every state-changing call (any method but GET, and every WebSocket session) appends a native.call audit record
  BEFORE egress (an unavailable audit log refuses the call); every request records a Decision (surface 'native');
- budgets: a budgeted key is reserved against the price row of native@<upstream> (402 budget_unpriced_target without
  one); a character-billed profile (adapters with native_usage, ElevenLabs) reserves characters, so a per_1k_chars row
  prices it; provider usage is not parsed (except where the adapter reads it from response headers: ElevenLabs
  character-cost), so otherwise the reservation stands as the spend (decision note cost:reserved_estimate).
"""
from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field, replace
from urllib.parse import parse_qsl, urlsplit

from .errors import BrokerError
from .profiles import INBOUND_HEADER_ALLOW, NATIVE_HEADER_NEVER, PROFILES, NativeRoute, Profile
from .surfaces import Surface, literal_weight, template_regex

NATIVE_PREFIX = '/native/'
UPSTREAM_HEADER = 'x-titanos-upstream'
NATIVE_REF_MODEL = 'native'
QUERY_CREDENTIALS = frozenset({'key', 'api_key', 'apikey', 'access_token'})
# headers that select which account, organization or project the shared credential acts for: privileged callers only
ACCOUNT_HEADERS = frozenset({'openai-organization', 'openai-project'})
# percent-encoded separators that would change the path once a provider front end decodes them
_ENCODED_SEPARATORS = re.compile(r'%(2f|5c|3f|23)|%2e%2e|\.%2e|%2e\.', re.IGNORECASE)
UPLOAD_PARAM = 'titanos_upload'
BASE_REQUEST_HEADERS = frozenset({'content-type', 'accept'}) | INBOUND_HEADER_ALLOW
PROTOCOL_BY_ADAPTER = {'openai_compat': 'openai', 'anthropic': 'anthropic', 'gemini': 'gemini',
                       'elevenlabs': 'elevenlabs', 'ollama_native': 'ollama'}
# The auth header each adapter's SDK sends; accepted as a Titanos credential on that profile's bare native paths (and,
# for every adapter, under /native/).
VENDOR_HEADER_BY_AUTH = {'anthropic': b'x-api-key', 'xi': b'xi-api-key', 'goog': b'x-goog-api-key'}
_UPSTREAM_ID = re.compile(r'[a-z][a-z0-9_-]{0,31}')
# Request headers that name the SDK family of a bare path both the OpenAI family and a vendor use (POST /v1/files is
# OpenAI's and Anthropic's): the Anthropic SDK always sends anthropic-version, the vendor SDKs their own key header.
FAMILY_HINTS = ((b'anthropic-version', 'anthropic'), (b'x-api-key', 'anthropic'), (b'x-goog-api-key', 'gemini'),
                (b'xi-api-key', 'elevenlabs'))


@dataclass(frozen=True)
class NativeMatch:
    """One request matched to an allowlisted row. rel: the path relative to the native root as the client spelled it
    (raw, percent-encoding kept) for egress; params: the decoded path parameters; bare: reached through the provider's
    public path; explicit: the upstream was named (path prefix or X-Titanos-Upstream); by_model: the visible candidate
    upstreams of a model_over_listed row whose bare path is also a Titanos route (OpenAI POST /v1/videos): undecided until
    the body's model is read (choose_by_model), empty for every decided match."""
    upstream_id: str
    profile_id: str
    route: NativeRoute
    rel: str
    params: dict = field(default_factory=dict)
    bare: bool = False
    explicit: bool = True
    by_model: tuple[str, ...] = ()


def _compile(profile: Profile):
    rows = []
    for route in profile.native.routes:
        rows.append((route, template_regex(route.path)))
    rows.sort(key=lambda item: -literal_weight(item[0].path))
    return tuple(rows)


_COMPILED = {pid: _compile(row) for pid, row in PROFILES.items() if row.native is not None}


def _head(path: str) -> str:
    """The first two path segments ('/v1/files' of '/v1/files/abc'): a cheap pre-filter for bare matching."""
    return '/' + '/'.join(path.split('/')[1:3])


# heads of every bare row path (bare prefix + row path), so a request that cannot be a bare native path costs one set
# lookup on the dispatch and scope fast paths
_BARE_HEADS = frozenset(_head((PROFILES[pid].native.bare_prefix or '') + route.path)
                        for pid, rows in _COMPILED.items() if PROFILES[pid].native.bare_prefix is not None
                        for route, _ in rows if route.bare)


def protocol_of(profile_id: str | None) -> str:
    row = PROFILES.get(profile_id or '')
    return PROTOCOL_BY_ADAPTER.get(row.adapter, 'openai') if row is not None else 'openai'


def _method(method: str) -> str:
    return 'WS' if method.upper() in ('WS', 'WEBSOCKET') else method.upper()


def match_route(profile_id: str, method: str, rel: str) -> tuple[NativeRoute, dict] | None:
    """The allowlisted row of profile_id for (method, rel) and its path parameters, or None."""
    method = _method(method)
    for route, pattern in _COMPILED.get(profile_id, ()):
        if route.method != method:
            continue
        found = pattern.match(rel)
        if found:
            return route, found.groupdict()
    return None


def routes_for(profile_id: str, rel: str) -> list[str]:
    """The methods allowlisted for rel (an empty list: the path is not allowlisted at all)."""
    return sorted({route.method for route, pattern in _COMPILED.get(profile_id, ()) if pattern.match(rel)})


def split_prefix(path: str) -> tuple[str, str] | None:
    """('<upstream_id>', '/<rest>') for /native/<upstream_id>/<rest>, or None."""
    if not path.startswith(NATIVE_PREFIX):
        return None
    upstream_id, slash, rest = path[len(NATIVE_PREFIX):].partition('/')
    if not upstream_id:
        return None
    return upstream_id, '/' + rest if slash else ''


def bare_rows(method: str, path: str) -> list[tuple[str, NativeRoute, str, dict]]:
    """Static bare matches: (profile_id, row, rel, params) of every profile whose bare prefix and row match. Collisions
    with Titanos routes are decided by the caller (scopes.required_scope and BrokerDispatch)."""
    method = _method(method)
    out = []
    if _head(path) not in _BARE_HEADS:
        return out
    for profile_id, rows in _COMPILED.items():
        spec = PROFILES[profile_id].native
        prefix = spec.bare_prefix
        if prefix is None or not (path.startswith(prefix + '/') if prefix else path.startswith('/')):
            continue
        rel = path[len(prefix):]
        for route, pattern in rows:
            if not route.bare or route.method != method:
                continue
            found = pattern.match(rel)
            if found:
                out.append((profile_id, route, rel, found.groupdict()))
                break
    return out


def bare_over_listed(method: str, path: str) -> bool:
    """Whether a bare row that wins over Titanos catalog and listed routes matches (method, path)."""
    return any(route.bare_over_listed for _, route, _, _ in bare_rows(method, path))


def vendor_families() -> dict[bytes, tuple[str, ...]]:
    """Path families (literal prefixes of the bare rows) where a profile's own SDK key header is accepted as a Titanos
    credential, per header name, plus /native for every vendor header."""
    out: dict[bytes, set[str]] = {}
    for profile_id in _COMPILED:
        row = PROFILES[profile_id]
        header = VENDOR_HEADER_BY_AUTH.get(row.auth)
        if header is None or row.native.bare_prefix is None:
            continue
        for route in row.native.routes:
            if not route.bare:
                continue
            literal = route.path.split('{', 1)[0]
            if '{' in route.path:
                literal = literal.rsplit('/', 1)[0]
            literal = literal.split(':', 1)[0].rstrip('/')
            if literal:
                out.setdefault(header, set()).add(row.native.bare_prefix + literal)
    for header in VENDOR_HEADER_BY_AUTH.values():
        out.setdefault(header, set()).add('/native')
    return {header: tuple(sorted(prefixes)) for header, prefixes in out.items()}


def pseudo_surface(method: str, protocol: str) -> Surface:
    """The Surface record a native request carries through the router, decisions, usage and metrics."""
    return Surface(name='native', method=_method(method), path='/native/{upstream_id}/{path:path}', protocol=protocol,
                   capability='native', body='none', model_from='none', model_absent='refuse', stream_from='none',
                   replay_safe=False, scope='native', handle_bound=False, kind='intercept')


# -- request checks -----------------------------------------------------------------------------------------------------
def check_rel(rel: str, params: dict) -> None:
    """Refuse dot segments and empty segments ('//'), also inside a decoded {x:path} parameter, the encoded separators
    %2F, %5C, %3F, %23 and %2E%2E anywhere in the raw path, and parameters that would leave their segment once decoded
    (controls, backslashes). Run again after any rewrite of the path (normalize_models), so everything the egress client
    would refuse as not a plain path is a 400 here, before any breaker or egress."""
    for segment in rel.split('/'):
        if segment in ('.', '..'):
            raise BrokerError('Dot segments are not allowed in native paths', 'invalid_request')
    if '//' in rel:
        raise BrokerError('Empty path segments are not allowed in native paths', 'invalid_request')
    if _ENCODED_SEPARATORS.search(rel):
        raise BrokerError('Encoded path separators (%2F, %5C, %3F, %23, %2E%2E) are not allowed in native paths',
                          'invalid_request')
    for name, value in params.items():
        if not isinstance(value, str) or not value or value in ('.', '..') or any(ord(c) < 32 for c in value) \
                or '\\' in value or any(part in ('', '.', '..') for part in value.split('/')):
            raise BrokerError(f'Path parameter {name} is not a plain identifier', 'invalid_request')


def query_items(query_string: bytes | str) -> list[tuple[str, str]]:
    """The inbound query as (name, value) pairs (repeats kept); a credential-named parameter is 401
    query_credential_refused (credentials never travel in URLs)."""
    if isinstance(query_string, bytes):
        query_string = query_string.decode('latin-1')
    items = parse_qsl(query_string or '', keep_blank_values=True)
    if any(name.lower() in QUERY_CREDENTIALS for name, _ in items):
        raise BrokerError('Credentials are not accepted in the URL query; send the key in a request header',
                          'query_credential_refused')
    return items


def privileged(principal, role: str) -> bool:
    """broker:admin principals and the keyless loopback owner in standalone role."""
    from .scopes import allows
    return principal is not None and (allows(principal, 'broker:admin') if principal.kind != 'anonymous_loopback'
                                      else role == 'standalone')


def forwarded_headers(profile: Profile, headers, *, account_headers: bool = False) -> dict[str, str]:
    """Inbound headers forwarded on the native lane: content-type, accept, INBOUND_HEADER_ALLOW and the row's
    request_headers; never NATIVE_HEADER_NEVER, credentials or x-titanos-*; ACCOUNT_HEADERS only when account_headers
    (privileged())."""
    allowed = BASE_REQUEST_HEADERS | (profile.native.request_headers if profile.native else frozenset())
    if not account_headers:
        allowed = allowed - ACCOUNT_HEADERS
    out: dict[str, str] = {}
    for name, value in headers:
        if isinstance(name, bytes):
            name = name.decode('latin-1')
        if isinstance(value, bytes):
            value = value.decode('latin-1')
        lowered = name.lower()
        if lowered in allowed and lowered not in NATIVE_HEADER_NEVER and not lowered.startswith('x-titanos-'):
            out[lowered] = value
    return out


def upstream_header(headers) -> str | None:
    for name, value in headers:
        if (name if isinstance(name, bytes) else name.encode('latin-1')).lower() == UPSTREAM_HEADER.encode():
            text = value.decode('latin-1') if isinstance(value, bytes) else str(value)
            return text.strip()
    return None


# -- authorization ------------------------------------------------------------------------------------------------------
def _not_found() -> BrokerError:
    return BrokerError('Not found', 'not_found', origin='policy')


def visible(principal, upstream_id: str, klass: str) -> bool:
    """The catalog visibility rule for an upstream (target class, upstream globs, cloud or cloud:<id>)."""
    from .scopes import allows
    if principal is None:
        return False
    if allows(principal, 'admin'):
        return True
    constraints = principal.constraints
    return (klass in constraints.target_classes
            and any(fnmatch.fnmatchcase(upstream_id, p) for p in constraints.upstreams)
            and allows(principal, 'cloud:' + upstream_id))


def native_target(snapshot, upstream_id: str):
    return snapshot.catalog.qualified_target(NATIVE_REF_MODEL, upstream_id)


def authorize(broker, snapshot, principal, hints, match: NativeMatch, surface: Surface):
    """(upstream, target, effective policy) after every gate in the module docstring, or BrokerError."""
    from .policy import effective_policy
    from .scopes import allows
    upstream = snapshot.config.upstreams.get(match.upstream_id)
    if upstream is None:
        raise _not_found()
    profile = PROFILES[upstream.profile]
    target = native_target(snapshot, match.upstream_id)
    if target is None or profile.native is None:
        raise _not_found()
    if not visible(principal, match.upstream_id, target.klass) or not any(
            fnmatch.fnmatchcase(target.ref, p) for p in principal.constraints.models):
        raise _not_found()
    if not allows(principal, 'native'):
        raise BrokerError('The native passthrough needs the native scope', 'scope_denied', origin='policy')
    route = match.route
    if route.store:
        if not allows(principal, 'cloud:store'):
            raise BrokerError('This endpoint keeps state at the provider and needs the cloud:store scope',
                              'scope_denied', origin='policy')
        if not upstream.data_policy.allow_store:
            raise BrokerError(f'Upstream {match.upstream_id} does not allow provider-side storage '
                              '(data_policy.allow_store)', 'data_policy_denied', origin='policy')
    if (route.admin or route.owner == 'list') and principal.kind in ('key', 'parent') \
            and not allows(principal, 'broker:admin'):
        raise BrokerError('This endpoint reaches the whole provider account and needs broker:admin', 'scope_denied',
                          origin='policy')
    if hints.hop:
        raise BrokerError('No target satisfies the requested location (requests from another broker hop stay local)',
                          'location_unsatisfiable', origin='policy')
    switches = snapshot.switches
    eff = effective_policy(broker.settings, snapshot.config, switches, principal, hints, surface)
    if eff.role == 'managed' and not eff.delegated:
        if getattr(switches, 'delegation', False):
            raise BrokerError('Cloud use in managed role needs parent delegation', 'delegation_required',
                              origin='policy')
        raise BrokerError('This role does not allow cloud targets', 'role_forbids', origin='policy')
    if eff.role == 'managed' and not match.explicit:
        raise BrokerError('A delegated native call must name its upstream (/native/<upstream> or X-Titanos-Upstream)',
                          'delegation_required', origin='policy')
    if eff.role != 'managed' and (not getattr(switches, 'cloud_egress', True) or (
            target.klass == 'router' and not getattr(switches, 'router_upstreams', True))):
        raise BrokerError('This role does not allow cloud targets', 'role_forbids', origin='policy')
    if eff.location == 'local_only':
        raise BrokerError('No target satisfies the requested location', 'location_unsatisfiable', origin='policy')
    if eff.cloud_policy == 'off':
        raise BrokerError('Cloud targets are disabled by policy', 'cloud_disabled', origin='policy')
    readiness = snapshot.catalog.readiness(match.upstream_id)
    if readiness is not None:
        if readiness[0] == 'disabled':
            raise _not_found()
        if readiness[1] == 'credential_unavailable':
            raise BrokerError(f'The credential of upstream {match.upstream_id} is not available in the server process',
                              'credential_unavailable')
        raise BrokerError(f'Upstream {match.upstream_id} has no validated credential', 'credential_unavailable')
    return upstream, target, eff


def family_hint(headers) -> str | None:
    """The adapter a bare request's SDK belongs to, from FAMILY_HINTS (None: no vendor hint, the OpenAI family)."""
    for name, _ in headers or ():
        name = name.lower() if isinstance(name, bytes) else str(name).lower().encode('latin-1')
        for hint, adapter in FAMILY_HINTS:
            if name == hint:
                return adapter
    return None


def owned_object(handles, principal, upstream_id: str, route: NativeRoute, params: dict) -> bool:
    """Whether the object named by an owner-bound row's path parameter was created through Titanos by this caller."""
    if handles is None or principal is None or route.owner != 'bound':
        return False
    oid = object_id(owner_param(route, params))
    if oid is None:
        return False
    try:
        binding = handles.owner('native', f'{upstream_id}:{oid}')
    except (OSError, ValueError):
        return False
    return binding is not None and binding.upstream_id == upstream_id and binding.key_id == principal.key_id


def resolve_bare(snapshot, principal, method: str, path: str, header_upstream: str | None, *,
                 listed: bool, headers=(), handles=None) -> NativeMatch | None:
    """The bare-path match for this request, or None when it is not a native bare path for this caller (the request
    then goes to the Titanos route as before). listed: (method, path) is also a Titanos route or catalog of the same
    shape; only bare_over_listed rows (or an explicit X-Titanos-Upstream) win there. With X-Titanos-Upstream the
    named upstream must offer the row (else 404 not_found); without it exactly one visible enabled upstream must (else
    None for zero, 400 invalid_request for several)."""
    rows = bare_rows(method, path)
    if not rows:
        return None
    config = snapshot.config
    if header_upstream is not None:
        if not _UPSTREAM_ID.fullmatch(header_upstream):
            raise BrokerError('X-Titanos-Upstream names an upstream id', 'invalid_request')
        upstream = config.upstreams.get(header_upstream)
        if upstream is None:
            raise _not_found()
        for profile_id, route, rel, params in rows:
            if profile_id == upstream.profile:
                return NativeMatch(header_upstream, profile_id, route, rel, params, bare=True, explicit=True)
        raise _not_found()
    candidates = []
    deferred = []  # model_over_listed rows on a listed path: decided by the body's model (choose_by_model)
    hint = family_hint(headers) if listed and any(route.vendor_over_listed for _, route, _, _ in rows) else None
    vendor_only = hint is not None and not any(
        (name.lower() if isinstance(name, bytes) else str(name).lower().encode('latin-1')) == b'authorization'
        for name, _ in headers or ())
    for upstream_id, upstream in sorted(config.upstreams.items()):
        if not upstream.enabled:
            continue
        for profile_id, route, rel, params in rows:
            if profile_id != upstream.profile:
                continue
            klass = 'router' if PROFILES[profile_id].kind == 'router' else 'cloud'
            if listed and not route.bare_over_listed and not (
                    route.vendor_over_listed and vendor_only and PROFILES[profile_id].adapter == hint) and not \
                    owned_object(handles, principal, upstream_id, route, params):
                if route.model_over_listed and visible(principal, upstream_id, klass):
                    deferred.append(NativeMatch(upstream_id, profile_id, route, rel, params, bare=True,
                                                explicit=False))
                continue
            if visible(principal, upstream_id, klass):
                candidates.append(NativeMatch(upstream_id, profile_id, route, rel, params, bare=True, explicit=False))
    if not candidates and deferred:
        return replace(deferred[0], by_model=tuple(match.upstream_id for match in deferred))
    if len(candidates) > 1:
        # the same public path in two SDK families: the vendor family when its headers are present, else OpenAI's
        hint = family_hint(headers)
        adapters = {c.upstream_id: PROFILES[c.profile_id].adapter for c in candidates}
        if hint is not None:
            candidates = [c for c in candidates if adapters[c.upstream_id] == hint]
        elif any(a == 'openai_compat' for a in adapters.values()):
            candidates = [c for c in candidates if adapters[c.upstream_id] == 'openai_compat']
    if not candidates:
        return None
    if len(candidates) > 1:
        raise BrokerError('Several upstreams offer this path (' + ', '.join(c.upstream_id for c in candidates)
                          + '); name one with the X-Titanos-Upstream header or use /native/<upstream>/...',
                          'invalid_request')
    return candidates[0]


def bare_may_win(method: str, path: str) -> bool:
    """Whether a bare row that can win over a listed Titanos route of the same shape for some requests matches: a
    model_over_listed row (decided by the body's model) or an owner-bound row (decided by the id's owner binding). Such a
    path needs the listed route's scope or 'native' at the Guard; BrokerDispatch then enforces the listed scope when the
    request stays on the Titanos route."""
    return any(route.model_over_listed or route.owner == 'bound' for _, route, _, _ in bare_rows(method, path))


# -- request bodies -----------------------------------------------------------------------------------------------------
_DISPOSITION_NAME = re.compile(rb'(?im)^content-disposition:[^\r\n]*?;\s*name="([^"]*)"')


def _boundary(content_type: str) -> bytes | None:
    if (content_type or '').split(';', 1)[0].strip().lower() != 'multipart/form-data':
        return None
    for part in content_type.split(';')[1:]:
        name, _, value = part.strip().partition('=')
        if name.strip().lower() == 'boundary' and value.strip():
            return value.strip().strip('"').encode('latin-1')
    return None


def _multipart_field_span(body: bytes, content_type: str, name: str) -> tuple[int, int] | None:
    """(start, end) of the value of the first non-file multipart/form-data field called name, or None."""
    boundary = _boundary(content_type)
    if not boundary or not body:
        return None
    delimiter = b'--' + boundary
    position = body.find(delimiter)
    while position != -1:
        start = position + len(delimiter)
        if body[start:start + 2] == b'--':
            return None
        if body[start:start + 2] == b'\r\n':
            start += 2
        following = body.find(b'\r\n' + delimiter, start)
        if following == -1:
            return None
        head_end = body.find(b'\r\n\r\n', start, following)
        if head_end != -1:
            head = body[start:head_end]
            found = _DISPOSITION_NAME.search(head)
            if found is not None and found.group(1) == name.encode('latin-1') and b'filename=' not in head.lower():
                return head_end + 4, following
        position = following + 2
    return None


def body_model(body: bytes, content_type: str):
    """The top-level model of a JSON, multipart/form-data or urlencoded request body (None when absent or unreadable)."""
    if not body:
        return None
    if is_json(content_type):
        document = parse_json(body)
        return document.get('model') if isinstance(document, dict) else None
    span = _multipart_field_span(body, content_type, 'model')
    if span is not None:
        try:
            return body[span[0]:span[1]].decode('utf-8')
        except UnicodeDecodeError:
            return None
    if (content_type or '').split(';', 1)[0].strip().lower() == 'application/x-www-form-urlencoded':
        try:
            items = parse_qsl(body.decode('utf-8'), keep_blank_values=True)
        except UnicodeDecodeError:
            return None
        return next((value for key, value in items if key == 'model'), None)
    return None


def choose_by_model(snapshot, principal, match: NativeMatch, model, *, local_names=()) -> NativeMatch | None:
    """Decide a by_model match (OpenAI POST /v1/videos, also Titanos' local video job route): the native match on the
    candidate upstream the body's model names, else None (the Titanos route answers, as before). The model names a cloud
    model of a candidate when it is (in order, local names first) a published alias, qualified ref or compat name that
    resolves to that upstream, a bare provider id that resolves there as on the typed surfaces (resolve_vendor), a ref
    '<model>@<candidate>', or, with exactly one candidate, a bare id of the profile's video model family
    (model_capabilities 'video': sora-*). A local model, an empty or non-string model and any other name stay local."""
    from .profiles import model_capabilities
    if not isinstance(model, str) or not model or model in local_names or not match.by_model:
        return None
    candidates = match.by_model
    catalog = snapshot.catalog
    found = None
    try:
        found = catalog.resolve_name(model, allow_virtual=False)
        if found is None:
            found = catalog.resolve_vendor(model, protocol_of(match.profile_id), principal)
    except BrokerError:
        found = None
    target = getattr(found, 'target', None) if found is not None else None
    chosen, named = None, False
    if target is not None:
        if getattr(target, 'klass', None) != 'local' and target.upstream_id in candidates:
            chosen, named = target.upstream_id, found.kind != 'vendor'
    elif '@' in model:
        name, _, suffix = model.rpartition('@')
        if name and suffix in candidates:
            chosen, named = suffix, True
    elif len(candidates) == 1 and 'video' in (model_capabilities(PROFILES[match.profile_id], model) or ()):
        chosen = candidates[0]
    if chosen is None:
        return None
    upstream = snapshot.config.upstreams.get(chosen)
    if upstream is None:
        return None
    return replace(match, upstream_id=chosen, profile_id=upstream.profile, by_model=(), explicit=named)


# -- ownership ----------------------------------------------------------------------------------------------------------
def object_id(value) -> str | None:
    """The provider object id an owner binding is keyed on: the last path segment of a name such as files/abc or
    cachedContents/xyz, without a ':action' suffix (files/abc:download -> abc)."""
    if not isinstance(value, str) or not value:
        return None
    last = value.rstrip('/').rsplit('/', 1)[-1]
    last = last.split(':', 1)[0]
    return last[:200] or None


def created_id(document) -> str | None:
    """The id of an object a 'create' row made: top-level id or name, else the id or name of a nested file, batch,
    voice or operation object (Gemini's {file:{name}}, ElevenLabs' {voice_id}, {dubbing_id})."""
    if not isinstance(document, dict):
        return None
    for key in ('id', 'name', 'voice_id', 'dubbing_id', 'pronunciation_dictionary_id', 'request_id'):
        found = object_id(document.get(key))
        if found:
            return found
    for nested in ('file', 'batch', 'response', 'operation', 'metadata'):
        inner = document.get(nested)
        if isinstance(inner, dict):
            for key in ('id', 'name'):
                found = object_id(inner.get(key))
                if found:
                    return found
    return None


def owner_param(route: NativeRoute, params: dict) -> str | None:
    if route.owner != 'bound':
        return None
    name = route.param or next(iter(params), None)
    return params.get(name) if name else None


def _owner_free(principal, role: str) -> bool:
    """Admin principals, and the keyless loopback operator in standalone role, may use any provider object id."""
    from .handles import is_admin
    return is_admin(principal) or (principal.kind == 'anonymous_loopback' and role == 'standalone')


def check_owner(handles, principal, role: str, upstream_id: str, route: NativeRoute, params: dict) -> None:
    value = owner_param(route, params)
    if value is None:
        return
    if _owner_free(principal, role):
        return
    oid = object_id(value)
    binding = handles.owner('native', f'{upstream_id}:{oid}') if oid else None
    if binding is None or binding.upstream_id != upstream_id or binding.key_id != principal.key_id:
        raise _not_found()


def bind_created(handles, principal, upstream_id: str, credential_id: str | None, document) -> str | None:
    oid = created_id(document)
    if oid is None:
        return None
    handles.bind('native', f'{upstream_id}:{oid}', upstream_id, credential_id, principal.key_id,
                 model=f'{NATIVE_REF_MODEL}@{upstream_id}')
    return oid


# -- answers ------------------------------------------------------------------------------------------------------------
def rewrite_url(value: str, *, upstream_origin_url: str, root_url: str, local_root: str) -> str:
    """An absolute URL on the upstream's root rewritten to this server's /native/<upstream>/ path (query kept); any other
    value unchanged."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    origin = urlsplit(upstream_origin_url)
    if (parts.scheme, parts.netloc.lower()) != (origin.scheme, origin.netloc.lower()):
        return value
    root_path = urlsplit(root_url).path.rstrip('/')
    path = parts.path
    if root_path and not (path == root_path or path.startswith(root_path + '/')):
        return value
    rel = path[len(root_path):] if root_path else path
    return local_root.rstrip('/') + rel + ('?' + parts.query if parts.query else '')


def rewrite_body_urls(data: bytes, fields: frozenset[str], **kw) -> bytes:
    """A JSON answer with the absolute upstream URLs of `fields` (top level, and in each item of a top-level 'data'
    list) rewritten by rewrite_url (kw: upstream_origin_url, root_url, local_root). The bytes are kept as the provider
    sent them except for the replaced string values; anything that is not such a JSON object passes unchanged."""
    document = parse_json(data)
    if not fields or not isinstance(document, dict):
        return data
    holders = [document] + [item for item in document.get('data') or [] if isinstance(item, dict)] \
        if isinstance(document.get('data'), list) else [document]
    out = data
    for holder in holders:
        for name in fields:
            value = holder.get(name)
            if not isinstance(value, str) or not value:
                continue
            new = rewrite_url(value, **kw)
            if new == value:
                continue
            old_text, new_text = json.dumps(value).encode('utf-8'), json.dumps(new).encode('utf-8')
            if old_text in out:
                out = out.replace(old_text, new_text)
            else:  # the provider escaped the value differently: re-serialize once with every field rewritten
                holder[name] = new
                out = b''
    if out == b'':
        for holder in holders:
            for name in fields:
                value = holder.get(name)
                if isinstance(value, str) and value:
                    holder[name] = rewrite_url(value, **kw)
        out = json.dumps(document, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return out


def is_json(content_type: str) -> bool:
    kind = (content_type or '').split(';', 1)[0].strip().lower()
    return kind == 'application/json' or kind.endswith('+json')


def parse_json(data: bytes):
    try:
        return json.loads(data.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        return None


def strip_frame(text: str, fields: frozenset[str]) -> str:
    """A client WebSocket text frame without the profile's credential fields (top level of a JSON object); anything that
    is not a JSON object passes unchanged."""
    if not fields or '{' not in text:
        return text
    try:
        document = json.loads(text)
    except ValueError:
        return text
    if not isinstance(document, dict) or not (fields & document.keys()):
        return text
    return json.dumps({k: v for k, v in document.items() if k not in fields}, ensure_ascii=False,
                      separators=(',', ':'))


def _model_value(value, upstream_id: str, catalog, local_names, bare: bool):
    """The provider's own id for one model name, or BrokerError (see normalize_models)."""
    if not isinstance(value, str) or not value:
        return value
    found = None
    if catalog is not None:
        try:
            found = catalog.resolve_name(value, allow_virtual=False)
        except BrokerError:
            found = None
        except Exception:
            found = None
    target = getattr(found, 'target', None) if found is not None else None
    remote = target is not None and getattr(target, 'klass', None) != 'local'
    if remote and target.upstream_id == upstream_id:
        return target.upstream_model  # a qualified ref of this upstream, or an alias that resolves to it
    if '@' in value:
        name, _, suffix = value.rpartition('@')
        if suffix == upstream_id and name:
            return name
        raise BrokerError(f'Model {value!r} names another upstream than {upstream_id}; a native call goes to exactly '
                          'one provider', 'invalid_request')
    if remote:
        return value  # a provider id that is also an alias elsewhere: sent as the client spelled it
    if bare and local_names is not None and value in local_names:
        raise BrokerError(f'Model {value!r} is a local model; {upstream_id} is a provider endpoint (name the provider '
                          'model, or use the local route)', 'invalid_request')
    return value


def normalize_models(match: NativeMatch, body: bytes, content_type: str, *, catalog=None, local_names=None
                     ) -> tuple[NativeMatch, bytes, list[str]]:
    """(match, body, notes) with Titanos model refs replaced by the provider's own ids before egress: a JSON body's
    top-level model and requests[].params.model, a multipart/form-data model field (only that field's bytes change), and
    a {model} path parameter. A JSON body is re-serialized only when a value changed; anything else passes untouched.
    The caller re-runs check_rel on the returned match (a rewritten path segment such as '..@<upstream>' -> '..'). BrokerError invalid_request for a ref of another upstream, or a
    local model name on a bare path."""
    notes: list[str] = []
    upstream_id = match.upstream_id
    value = match.params.get('model')
    if isinstance(value, str) and value:
        new = _model_value(value, upstream_id, catalog, local_names, match.bare)
        if new != value:
            rel = match.rel
            for spelled in (value, value.replace('@', '%40'), value.replace('@', '%40').lower()):
                if spelled in rel:
                    rel = rel.replace(spelled, new, 1)
                    break
            match = NativeMatch(match.upstream_id, match.profile_id, match.route, rel, {**match.params, 'model': new},
                                bare=match.bare, explicit=match.explicit)
            notes.append('native:model_rewritten')
    if body and is_json(content_type):
        document = parse_json(body)
        if isinstance(document, dict):
            changed = False
            if isinstance(document.get('model'), str):
                new = _model_value(document['model'], upstream_id, catalog, local_names, match.bare)
                if new != document['model']:
                    document['model'], changed = new, True
            for item in document.get('requests') or [] if isinstance(document.get('requests'), list) else []:
                params = item.get('params') if isinstance(item, dict) else None
                if isinstance(params, dict) and isinstance(params.get('model'), str):
                    new = _model_value(params['model'], upstream_id, catalog, local_names, match.bare)
                    if new != params['model']:
                        params['model'], changed = new, True
            if changed:
                body = json.dumps(document, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                notes.append('native:model_rewritten')
    elif body:
        span = _multipart_field_span(body, content_type, 'model')  # OpenAI SDK multipart creates (videos with a file)
        if span is not None:
            try:
                value = body[span[0]:span[1]].decode('utf-8')
            except UnicodeDecodeError:
                value = None
            new = _model_value(value, upstream_id, catalog, local_names, match.bare) if value else value
            if new != value:
                body = body[:span[0]] + new.encode('utf-8') + body[span[1]:]
                notes.append('native:model_rewritten')
    return match, body, notes


class UploadSessions:
    """Upload capabilities for rewritten resumable-upload URLs (Gemini X-Goog-Upload-URL). google-genai sends the upload
    and finalize chunks to the session URL without any key (Google's session URL is pre-authorized), so the URL Titanos
    hands out carries ?titanos_upload=<token>: 256 random bits, bound to the principal that started the upload, the
    upstream and the session's own path and upload_id. claim() returns that principal for a request on exactly that
    path and session (and only POST or PUT), until the session is finalized, cancelled or MAX_AGE passes. In-memory: a
    restart ends every open upload session."""
    MAX_AGE = 24 * 3600.0
    MAX_SESSIONS = 4096

    def __init__(self, clock=None):
        import threading
        import time
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple] = {}

    def issue(self, principal, upstream_id: str, local_path: str, upload_id: str | None) -> str:
        import secrets
        token = secrets.token_urlsafe(32)
        now = self._clock()
        with self._lock:
            for key in [k for k, v in self._sessions.items() if v[4] <= now]:
                self._sessions.pop(key, None)
            while len(self._sessions) >= self.MAX_SESSIONS:
                self._sessions.pop(next(iter(self._sessions)))
            self._sessions[token] = (principal, upstream_id, local_path, upload_id, now + self.MAX_AGE)
        return token

    def claim(self, token: str, method: str, path: str, upload_id: str | None):
        """The bound principal for this request, or None."""
        import hmac
        if not token or method.upper() not in ('POST', 'PUT'):
            return None
        with self._lock:
            found = None
            for key, value in self._sessions.items():
                if hmac.compare_digest(key.encode('ascii', 'replace'), token.encode('ascii', 'replace')):
                    found = (key, value)
                    break
            if found is None:
                return None
            key, (principal, upstream_id, local_path, bound_upload, expires) = found
            if expires <= self._clock():
                self._sessions.pop(key, None)
                return None
            if path != local_path or (bound_upload is not None and upload_id != bound_upload):
                return None
            return principal

    def finish(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def __len__(self) -> int:
        return len(self._sessions)


UPLOADS = UploadSessions()


def upload_capability(url: str, principal, upstream_id: str) -> str:
    """A rewritten upload URL (this server's /native/<upstream>/... path) with an upload capability appended."""
    from urllib.parse import parse_qsl as _parse, urlencode
    parts = urlsplit(url)
    query = _parse(parts.query, keep_blank_values=True)
    upload_id = next((v for k, v in query if k == 'upload_id'), None)
    token = UPLOADS.issue(principal, upstream_id, parts.path, upload_id)
    return url + ('&' if parts.query else '?') + urlencode({UPLOAD_PARAM: token})


def split_upload_token(query_string) -> tuple[str | None, str | None, bytes]:
    """(token, upload_id, query without the capability) of an inbound query string."""
    from urllib.parse import urlencode
    if isinstance(query_string, bytes):
        query_string = query_string.decode('latin-1')
    items = parse_qsl(query_string or '', keep_blank_values=True)
    token = next((v for k, v in items if k == UPLOAD_PARAM), None)
    upload_id = next((v for k, v in items if k == 'upload_id'), None)
    kept = [(k, v) for k, v in items if k != UPLOAD_PARAM]
    return token, upload_id, urlencode(kept).encode('latin-1')


def websocket_support() -> str | None:
    """The WebSocket library uvicorn would use to accept upgrades ('websockets' or 'wsproto'), or None."""
    import importlib.util
    for name in ('websockets', 'wsproto'):
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            continue
    return None


__all__ = ['ACCOUNT_HEADERS', 'BASE_REQUEST_HEADERS', 'NATIVE_PREFIX', 'NativeMatch', 'PROTOCOL_BY_ADAPTER',
           'UPLOAD_PARAM', 'UPLOADS', 'UPSTREAM_HEADER', 'UploadSessions', 'normalize_models', 'owned_object',
           'privileged',
           'authorize', 'bare_may_win', 'bare_over_listed', 'bare_rows', 'bind_created', 'body_model', 'check_owner',
           'check_rel', 'choose_by_model', 'created_id', 'family_hint',
           'forwarded_headers', 'is_json', 'match_route', 'native_target', 'object_id', 'owner_param', 'parse_json',
           'protocol_of', 'pseudo_surface', 'query_items', 'resolve_bare', 'rewrite_body_urls', 'rewrite_url',
           'routes_for', 'split_prefix',
           'strip_frame', 'upstream_header', 'vendor_families', 'visible', 'websocket_support']
