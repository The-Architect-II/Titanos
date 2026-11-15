"""The Anthropic-shaped model catalog: GET /v1/models and GET /v1/models/{model_id} for the Anthropic SDKs.

The Anthropic SDK's models.list() and models.retrieve() call the same paths as OpenAI clients but expect Anthropic's
shapes. A request that carries the anthropic-version header (every Anthropic SDK sends it; OpenAI clients never do) gets
Titanos' own catalog (exactly the entries GET /v1/models lists for that caller: local models, published aliases,
qualified refs and exposed virtual models) re-shaped:
- list: {data: [{type: 'model', id, display_name, created_at, ...}], has_more, first_id, last_id}, paged with limit
  (default 20, 1..1000), after_id and before_id as the provider pages;
- one model: the same object; an unknown name is a 404 in Anthropic's error envelope
  {type: 'error', error: {type: 'not_found_error', message}}; any other error keeps its status in that envelope;
- entries served by an Anthropic upstream carry the provider's own display_name, created_at, max_input_tokens,
  max_tokens and capabilities from the live catalog (adapters/anthropic.list_models_meta); other entries carry
  display_name = id, created_at from the OpenAI 'created' epoch and max_input_tokens when the catalog knows a context
  length;
- the list keeps only what a Messages client can call: local entries and remote entries whose capabilities include
  chat (no embedding, speech, transcription or image-only models), and an Anthropic upstream's qualified ref is listed
  by its bare provider id when that id resolves to it (profile extras bare_model_ids: exactly one visible Anthropic
  upstream lists it), so models.list() ids work in models.retrieve() and messages.create();
- models.retrieve() of such a bare id answers that model (the id as the client sent it).
x-api-key is accepted as the Titanos credential on these paths (broker/auth.py), so the stock SDK works in keyed mode.
The OpenAI shape is unchanged for every other client.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import parse_qsl

PATHS = ('/v1/models',)
DEFAULT_LIMIT = 20
MAX_LIMIT = 1000


def wants(scope) -> bool:
    """Whether a GET on /v1/models or /v1/models/{id} comes from an Anthropic SDK (anthropic-version header)."""
    if scope.get('type') != 'http' or str(scope.get('method', 'GET')).upper() != 'GET':
        return False
    path = scope.get('path', '') or ''
    if not (path == '/v1/models' or path.startswith('/v1/models/')):
        return False
    return any((name.lower() if isinstance(name, bytes) else str(name).lower().encode('latin-1')) == b'anthropic-version'
               for name, _ in scope.get('headers') or ())


def _iso(epoch) -> str:
    try:
        value = int(epoch)
    except (TypeError, ValueError):
        value = 0
    return datetime.fromtimestamp(max(0, value), tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _provider_meta(broker, name: str) -> dict:
    if broker is None:
        return {}
    try:
        catalog = broker.snapshot().catalog
        target = catalog.lookup(name)
        meta_of = getattr(catalog, 'model_meta', None)
        if target is None or meta_of is None or getattr(target, 'adapter', None) != 'anthropic':
            return {}
        return dict(meta_of(target.upstream_id, target.upstream_model) or {})
    except Exception:  # catalog trouble never fails a catalog read
        return {}


def _chat_capable(entry: dict) -> bool:
    """Local entries, and remote entries whose capabilities include chat."""
    info = entry.get('titanos') if isinstance(entry.get('titanos'), dict) else None
    if not entry.get('remote') or info is None:
        return True
    if info.get('class') == 'virtual':
        return info.get('capability') in (None, 'chat')
    caps = info.get('model_capabilities') or info.get('capabilities')
    return not isinstance(caps, list) or 'chat' in caps


def bare_id(broker, principal, ref: str) -> str | None:
    """The bare provider id of an Anthropic upstream's qualified ref when it resolves back to that ref, else None."""
    if broker is None or '@' not in ref:
        return None
    try:
        catalog = broker.snapshot().catalog
        model = ref.rsplit('@', 1)[0]
        found = catalog.resolve_vendor(model, 'anthropic', principal)
    except Exception:  # catalog trouble never fails a catalog read
        return None
    target = getattr(found, 'target', None)
    return model if target is not None and target.ref == ref else None


def model_object(entry: dict, broker=None, principal=None) -> dict | None:
    """One Anthropic model object from one OpenAI-shaped catalog entry (None when it has no id)."""
    name = entry.get('id')
    if not isinstance(name, str) or not name:
        return None
    shown = bare_id(broker, principal, name) if principal is not None else None
    out = {'type': 'model', 'id': shown or name, 'display_name': shown or name, 'created_at': _iso(entry.get('created'))}
    meta = _provider_meta(broker, name)
    provider = meta.get('anthropic') if isinstance(meta.get('anthropic'), dict) else {}
    for key in ('display_name', 'created_at', 'max_input_tokens', 'max_tokens', 'capabilities'):
        if key in provider:
            out[key] = provider[key]
    context = (entry.get('titanos') or {}).get('context_length') if isinstance(entry.get('titanos'), dict) else None
    if 'max_input_tokens' not in out and isinstance(context, int):
        out['max_input_tokens'] = context
    return out


def page(objects: list[dict], query: str) -> dict:
    """Anthropic list paging over objects (limit, after_id, before_id)."""
    params = dict(parse_qsl(query or '', keep_blank_values=True))
    try:
        limit = int(params.get('limit') or DEFAULT_LIMIT)
    except ValueError:
        limit = DEFAULT_LIMIT
    limit = max(1, min(MAX_LIMIT, limit))
    ids = [o['id'] for o in objects]
    items = objects
    if params.get('after_id') in ids:
        items = objects[ids.index(params['after_id']) + 1:]
        chosen = items[:limit]
        has_more = len(items) > limit
    elif params.get('before_id') in ids:
        items = objects[:ids.index(params['before_id'])]
        chosen = items[-limit:]
        has_more = len(items) > limit
    else:
        chosen = items[:limit]
        has_more = len(items) > limit
    return {'data': chosen, 'has_more': has_more, 'first_id': chosen[0]['id'] if chosen else None,
            'last_id': chosen[-1]['id'] if chosen else None}


_ERROR_TYPES = {400: 'invalid_request_error', 401: 'authentication_error', 403: 'permission_error',
                404: 'not_found_error', 413: 'request_too_large', 429: 'rate_limit_error', 503: 'overloaded_error'}


def error_body(status: int, document) -> dict:
    """Anthropic's error envelope for a Titanos error answer (OpenAI {error: {message}} or FastAPI {detail})."""
    message = None
    if isinstance(document, dict):
        error = document.get('error')
        if isinstance(error, dict):
            message = error.get('message')
        elif isinstance(document.get('detail'), str):
            message = document['detail']
    return {'type': 'error', 'error': {'type': _ERROR_TYPES.get(status, 'api_error' if status >= 500 else
                                                                'invalid_request_error'),
                                       'message': str(message or f'HTTP {status}')}}


def reshape(status: int, data: bytes, *, single: bool, query: str, broker=None, principal=None,
            requested: str | None = None) -> tuple[int, bytes]:
    try:
        document = json.loads(data.decode('utf-8')) if data else None
    except (ValueError, UnicodeDecodeError):
        document = None
    if not 200 <= status < 300:
        return status, _dumps(error_body(status, document))
    if single:
        found = model_object(document, broker, principal) if isinstance(document, dict) else None
        if found is None:
            return 404, _dumps(error_body(404, {'detail': 'model not found'}))
        if requested:
            found['id'] = requested
        return status, _dumps(found)
    rows = document.get('data') if isinstance(document, dict) else None
    objects = [o for o in (model_object(e, broker, principal) for e in rows or []
                           if isinstance(e, dict) and _chat_capable(e)) if o is not None]
    return status, _dumps(page(objects, query))


def _dumps(document) -> bytes:
    return json.dumps(document, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


async def serve(app, scope, receive, send, broker=None) -> None:
    """Run the inner catalog route and answer its result in Anthropic's shapes (module docstring)."""
    start: dict = {}
    chunks: list[bytes] = []

    async def capture(message):
        if message['type'] == 'http.response.start':
            start.update(message)
        elif message['type'] == 'http.response.body':
            chunks.append(message.get('body', b''))

    principal = scope.get('titanos.principal') or (broker.default_principal() if broker is not None else None)
    single = (scope.get('path') or '').startswith('/v1/models/')
    requested = None
    if single and broker is not None:
        name = scope['path'][len('/v1/models/'):]
        if name and '@' not in name and '/' not in name:
            try:
                found = broker.snapshot().catalog.resolve_vendor(name, 'anthropic', principal)
            except Exception:
                found = None
            target = getattr(found, 'target', None)
            if target is not None:  # the SDK's bare id: answer the qualified entry under the id the client sent
                from urllib.parse import quote
                requested = name
                path = '/v1/models/' + target.ref
                scope = dict(scope, path=path, raw_path=quote(path, safe='/@:').encode('latin-1'))
    await app(scope, receive, capture)
    status = int(start.get('status', 500))
    query = (scope.get('query_string') or b'').decode('latin-1')
    status, body = reshape(status, b''.join(chunks), single=single, query=query, broker=broker, principal=principal,
                           requested=requested)
    headers = [(k, v) for k, v in start.get('headers') or [] if k.lower() not in (b'content-length', b'content-type')]
    headers += [(b'content-type', b'application/json'), (b'content-length', str(len(body)).encode('latin-1'))]
    await send({'type': 'http.response.start', 'status': status, 'headers': headers})
    await send({'type': 'http.response.body', 'body': body, 'more_body': False})


__all__ = ['DEFAULT_LIMIT', 'MAX_LIMIT', 'bare_id', 'error_body', 'model_object', 'page', 'reshape', 'serve', 'wants']
