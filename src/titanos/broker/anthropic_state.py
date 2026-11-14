"""Provider-side state named inside Anthropic Messages bodies (R1 and decision 3 on a shared organisation credential).

A Messages request (POST /v1/messages, /v1/messages/count_tokens, and every request of a Message Batch) can reference
objects that live in the provider account behind the upstream's credential, not in the request:
- container: a code-execution container id to reuse (a string, or an object with an id), and custom skills
  (container.skills[] of type 'custom' name an uploaded skill of the account);
- file sources: document and image blocks whose source is {type: file, file_id}, and container_upload blocks
  ({type: container_upload, file_id}).
Files are workspace-scoped at the provider: any caller of the shared credential could read any file it can name. So,
before egress:
- a body that references any of these needs cloud:store AND the upstream's data_policy.allow_store (the router's
  plan.allow_store for that target); otherwise 403 scope_denied (no cloud:store) or 403 data_policy_denied;
- a file id that was uploaded through Titanos is bound to the uploading key (handles.json kind 'native', see
  broker/native.py); another key naming it gets 404 not_found, exactly as if the id did not exist. Admin principals and
  the keyless loopback operator in standalone role may name any id. Ids Titanos never saw (outputs of code execution,
  files uploaded outside Titanos) are not bound to anyone and pass with cloud:store.
Server tools that create state as a side effect (code execution writes output files) are not refused here: their outputs
are reachable only through the Files endpoints, which the native passthrough store-gates and owner-binds.
"""
from __future__ import annotations

from .errors import BrokerError

FILE_BLOCKS = frozenset({'document', 'image', 'container_upload'})
_MAX_DEPTH = 64


def _walk(value, depth: int = 0):
    if depth > _MAX_DEPTH:
        return
    if isinstance(value, dict):
        yield value
        for item in value.values():
            if isinstance(item, (dict, list)):
                yield from _walk(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item, depth + 1)


def file_ids(body) -> list[str]:
    """Every file_id a Messages body names in a document, image or container_upload block."""
    out = []
    for node in _walk({k: body.get(k) for k in ('messages', 'system')} if isinstance(body, dict) else None):
        kind = node.get('type')
        if kind == 'container_upload' and isinstance(node.get('file_id'), str):
            out.append(node['file_id'])
        elif kind in FILE_BLOCKS:
            source = node.get('source')
            if isinstance(source, dict) and source.get('type') == 'file' and isinstance(source.get('file_id'), str):
                out.append(source['file_id'])
    return out


def state_references(body) -> list[str]:
    """Labels of the provider-side state a Messages body names ('container', 'skills', 'file'); empty for none."""
    if not isinstance(body, dict):
        return []
    found = []
    container = body.get('container')
    if isinstance(container, str) and container:
        found.append('container')
    elif isinstance(container, dict):
        if container.get('id'):
            found.append('container')
        skills = container.get('skills')
        if isinstance(skills, list) and any(isinstance(s, dict) and s.get('type') == 'custom' for s in skills):
            found.append('skills')
    if file_ids(body):
        found.append('file')
    return found


def batch_params(body) -> list[dict]:
    """The params objects of a Message Batch create body."""
    requests = body.get('requests') if isinstance(body, dict) else None
    return [r['params'] for r in requests or [] if isinstance(r, dict) and isinstance(r.get('params'), dict)]


def _owner_free(principal, role: str) -> bool:
    from .handles import is_admin
    return is_admin(principal) or (getattr(principal, 'kind', None) == 'anonymous_loopback' and role == 'standalone')


def check(body, *, principal, allow_store: bool, store_scope: bool, handles, upstream_id: str, role: str) -> None:
    """Refuse (BrokerError) a body naming provider-side state the caller may not use (module docstring). allow_store: the
    caller holds cloud:store and the upstream allows storage; store_scope: the caller holds cloud:store (to tell the two
    refusals apart)."""
    labels = state_references(body)
    if not labels:
        return
    what = ', '.join(labels)
    if not allow_store:
        if not store_scope:
            raise BrokerError(f'This request names provider-side state ({what}) and needs the cloud:store scope',
                              'scope_denied', origin='policy')
        raise BrokerError(f'Upstream {upstream_id} does not allow provider-side storage (data_policy.allow_store); '
                          f'this request names provider-side state ({what})', 'data_policy_denied', origin='policy')
    if handles is None or _owner_free(principal, role):
        return
    from .native import object_id
    key_id = getattr(principal, 'key_id', None)
    for file_id in file_ids(body):
        oid = object_id(file_id)
        binding = handles.owner('native', f'{upstream_id}:{oid}') if oid else None
        if binding is not None and (binding.upstream_id != upstream_id or binding.key_id != key_id):
            raise BrokerError('Not found', 'not_found', origin='policy')


def check_batch(body, **kw) -> None:
    """check() for every request of a Message Batch create body."""
    for params in batch_params(body):
        check(params, **kw)


__all__ = ['FILE_BLOCKS', 'batch_params', 'check', 'check_batch', 'file_ids', 'state_references']
