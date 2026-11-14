"""HTTP control routes of the broker: /api/broker/*, /api/keys/* and /api/audit (register_admin_routes).

Scopes come from broker.scopes.ROUTE_SCOPES and are enforced by the Authenticator in the Guard before the body is read;
the handlers add the rules a route scope cannot express:
- Credential writes (PUT, rotate, validate, DELETE) and key mint/rotate need Authenticator.transport_secure (TLS or a
  loopback peer; uvicorn's proxy headers are pinned, so X-Forwarded-* cannot satisfy it): 403 transport_insecure. They
  are refused in keyless mode (403 keyless_mode_refused; the Authenticator refuses them first) and credential writes,
  validation and upstream tests are refused in managed role (409 role_forbids).
- A PUT /api/broker/policy from an anonymous_loopback principal that changes anonymous_cloud or allow_http_lan
  (config.KEYLESS_FORBIDDEN_POLICY_FIELDS) is 403 keyless_mode_refused; cloud_policy may not exceed
  TITANOS_BROKER_CLOUD_CEILING (400 invalid_request).
- broker.json writes go through config.update_broker_config (the shared lock, atomic replace): the new document is
  validated, checked for secrets (400 broker_config_contains_secret) and name conflicts with the local registry (409
  broker_name_conflict), refused in managed role when it would enable egress without delegation (409 role_forbids),
  audited inside the broker.json lock BEFORE the atomic replace (an audit that cannot be appended refuses the change
  with 503 audit_unavailable and leaves broker.json untouched), and the snapshot is reloaded.
- Vault writes are audited the same way: the audit record is appended before the vault changes (a rotation is audited
  before the staged value is swapped in), so an unavailable audit log refuses the write with nothing changed; a write
  the vault then refuses gets a follow-up record {ok: false, code}. Error messages name a file: reference by its file
  name only, never its path. An upstream base_url change resets the validation of the credentials bound to
  that upstream (credentials:write must validate them again).
- Secrets are accepted only in credential bodies, are never echoed, logged or audited, and every response that could
  carry a key secret (mint, rotate) or credential metadata is Cache-Control: no-store.
- Decisions and usage: broker:read sees its own key only; broker:admin sees all.
- R29 keyless Studio (auth.Authenticator.studio_request: standalone role, loopback bind, nothing enforced, the
  X-Titanos-Studio: 1 header and a JSON body): the principal 'keyless-studio' may write credentials (env: and file:
  references included), connect providers, and change policy.anonymous_cloud; allow_http_lan stays CLI or keyed
  broker:admin only, and keys are never minted keylessly. Every such write is audited with actor keyless-studio.
- R30 GET /api/broker/env-candidates: which well-known provider credential variable NAMES are set in the server process
  (names and booleans only). broker:admin; a keyless caller needs the Studio header.
- R31 POST /api/broker/connect (connect.py): one-step provider setup; broker:admin plus credentials:write (keyed callers
  over TLS or a loopback peer), or the keyless Studio principal.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Literal

from fastapi import Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import (AdvisorSpec, BrokerConfig, BrokerPolicy, PriceRow, PublishedModel, Upstream, VirtualModel,
                     check_name_conflicts, check_no_secrets, forbidden_policy_changes, parse_origin,
                     update_broker_config)
from .errors import BrokerError
from .keys import KeyPatch, KeySpec
from .profiles import PROFILES
from .scopes import allows
from .types import CLOUD_POLICIES

STUDIO_ID = 'keyless-studio'


def is_studio(principal) -> bool:
    """The R29 keyless Studio principal (auth.Authenticator.studio_principal)."""
    return getattr(principal, 'kind', None) == 'anonymous_loopback' and getattr(principal, 'id', None) == STUDIO_ID

NO_STORE = {'Cache-Control': 'no-store'}
ID_PATTERN = r'^[a-z][a-z0-9_-]{0,63}$'
_REFUSED_FILE = re.compile(r'Credential file (.+?) is refused: ')
_SEPARATORS = re.compile(r'[\\/]')


def studio_host_source(env: str | None, file: str | None, profile_id: str | None, base_url: str | None) -> None:
    """R29/R30 for the keyless Studio principal: a host source may name only one of the profile's well-known credential
    variable NAMES (profiles.Profile.env_names, the list env discovery offers), only for an upstream that keeps the
    profile's default origin (a base_url override would send the host value to an origin the caller picked), and never
    a file: path. BrokerError scope_denied otherwise. The full 'admin' scope (CLI, bootstrap key) is not limited."""
    if env is None and file is None:
        return
    if file is not None:
        raise BrokerError('file: credential sources are not accepted from the keyless Studio; send the secret itself, '
                          'use one of the provider\'s well-known environment variables, or run titanos broker '
                          'credential set on the host', 'scope_denied', origin='policy')
    profile = PROFILES.get(profile_id or '')
    names = profile.env_names if profile is not None else ()
    if env not in names:
        raise BrokerError('The keyless Studio may reference only the provider\'s well-known credential variables'
                          + (f' ({", ".join(names)})' if names else '') + '; run titanos broker credential set on the '
                          'host for any other variable', 'scope_denied', origin='policy')
    default = (profile.default_base_url or '').rstrip('/')
    if base_url is not None and base_url.rstrip('/') != default:
        raise BrokerError('An environment credential from the keyless Studio is sent only to the provider\'s default '
                          'address; this upstream has another base_url (send the secret itself, or use the titanos CLI '
                          'on the host)', 'scope_denied', origin='policy')


def without_path(exc: BrokerError, path: str | None) -> BrokerError:
    """exc with a file: reference's path reduced to its file name (an HTTP reader never learns host paths)."""
    message = str(exc)
    found = _REFUSED_FILE.match(message)
    if found:
        message = f'Credential file {_SEPARATORS.split(found.group(1))[-1]} is refused: ' + message[found.end():]
    if path:
        message = message.replace(path, _SEPARATORS.split(path)[-1])
    if message == str(exc):
        return exc
    return BrokerError(message, exc.code, status=exc.status, retry_after=exc.retry_after, origin=exc.origin,
                       headers=exc.headers)


class CredentialBody(BaseModel):
    """PUT /api/broker/credentials/{id}: exactly one of secret, env (a variable name) or file (an owner-only path)."""
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, populate_by_name=True)
    upstream: str = Field(min_length=1, max_length=64)
    secret: str | None = Field(None, min_length=1, max_length=8192)
    env: str | None = Field(None, min_length=1, max_length=128)
    file: str | None = Field(None, min_length=1, max_length=4096)
    purpose: Literal['inference', 'advisor'] = 'inference'
    validate_: bool = Field(True, alias='validate')


class RotateBody(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    secret: str | None = Field(None, min_length=1, max_length=8192)
    env: str | None = Field(None, min_length=1, max_length=128)
    file: str | None = Field(None, min_length=1, max_length=4096)


class ExplainBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    surface: str = Field('openai.chat', min_length=1, max_length=64)
    body: dict = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    key_id: str | None = Field(None, max_length=32)


def register_admin_routes(app, broker) -> None:
    """Attach the broker control routes (every route of http_api's ADMIN AND CONTROL, credential, catalog and policy,
    observability, key and audit groups)."""

    # -- helpers ---------------------------------------------------------------------------------------------------------
    def principal_of(request: Request):
        return request.scope.get('titanos.principal') or broker.default_principal()

    def actor(request: Request) -> str:
        return str(getattr(principal_of(request), 'id', 'anonymous'))

    def require_transport(request: Request) -> None:
        auth = broker.authenticator
        if is_studio(principal_of(request)) and not auth.enforced:
            return  # R29: a loopback Studio request in standalone role (the Authenticator checked every condition)
        if not auth.enforced:
            raise BrokerError('Keys and credentials cannot be written in keyless mode; use the titanos CLI '
                              '(titanos keys create, titanos broker credential set)', 'keyless_mode_refused',
                              origin='policy')
        if not auth.transport_secure(request.scope):
            raise BrokerError('Key and credential writes need TLS or a loopback client', 'transport_insecure',
                              origin='policy')

    def require_host_source(request: Request, body) -> None:
        """env: and file: sources name values that live on the server host (process variables, owner-only files). Binding
        one to an upstream whose base_url a broker:admin key controls would send that host value there, so naming a
        host source needs the admin scope; credentials:write alone may store a value it supplies itself."""
        host_source = getattr(body, 'env', None) is not None or getattr(body, 'file', None) is not None
        if host_source and not allows(principal_of(request), 'admin') and not is_studio(principal_of(request)):
            raise BrokerError('env: and file: credential sources name values on the server host; they need the admin '
                              'scope (send the secret itself, or run titanos broker credential set on the host)',
                              'scope_denied', origin='policy')

    def studio_source(request: Request, body, upstream_id: str, purpose: str = 'inference') -> None:
        """The narrower R29/R30 rule for the keyless Studio principal (which does not hold 'admin'): an env: source may
        name only one of the upstream profile's well-known credential variables, and only for an upstream that keeps the
        profile's default origin; file: sources are refused (see studio_host_source)."""
        principal = principal_of(request)
        if not is_studio(principal) or allows(principal, 'admin'):
            return
        upstream = config().upstreams.get(upstream_id) if purpose == 'inference' else None
        studio_host_source(getattr(body, 'env', None), getattr(body, 'file', None),
                           upstream.profile if upstream is not None else None,
                           upstream.base_url if upstream is not None else None)

    def refuse_managed(what: str) -> None:
        if broker.settings.role == 'managed':
            raise BrokerError(f'{what} is not available in managed role', 'role_forbids', status=409, origin='policy')

    def audit(kind: str, request: Request, subject: str, detail: dict) -> None:
        broker.audit.event(kind, actor(request), subject, detail)

    def audit_refusal(kind: str, request: Request, subject: str, exc: BrokerError) -> None:
        """Best-effort follow-up record when the vault refuses a write that was already audited."""
        try:
            audit(kind, request, subject, {'ok': False, 'code': exc.code})
        except Exception:
            pass

    def source_of(body) -> str:
        return 'vault' if body.secret is not None else ('env' if body.env is not None else 'file')

    def vault_put(vault, id: str, body, kw: dict, request: Request, kind: str):
        """The put of one credential source, after its audit record was appended (see the module docstring)."""
        try:
            if body.secret is not None:
                return vault.put_secret(id, body.secret, **kw)
            if body.env is not None:
                return vault.put_env_ref(id, body.env, **kw)
            return vault.put_file_ref(id, body.file, **kw)
        except BrokerError as exc:
            audit_refusal(kind, request, id, exc)
            raise without_path(exc, body.file) from None

    def config() -> BrokerConfig:
        return broker.snapshot().config

    def write(request: Request, kind: str, subject: str, change, detail: dict | None = None) -> BrokerConfig:
        """Apply change to the latest broker.json under the lock, with the checks in the module docstring."""
        old = config()

        def checked(cfg: BrokerConfig) -> None:
            change(cfg)
            try:
                candidate = BrokerConfig.model_validate(cfg.model_dump(mode='json'))
            except BrokerError:
                raise
            except ValueError as exc:
                raise BrokerError(f'The broker configuration is invalid: {str(exc)[:400]}', 'invalid_request') \
                    from None
            check_name_conflicts(candidate, broker.local_models().keys())
            if (broker.settings.role == 'managed' and not broker.switches.delegation
                    and broker._enables_egress(old, candidate)):
                raise BrokerError('This change would enable egress in managed role without delegation', 'role_forbids',
                                  status=409, origin='policy')
            check_no_secrets(candidate.model_dump(mode='json'))  # the write's own check, before anything is audited
            # Audited under the broker.json lock BEFORE the atomic replace: when the record cannot be appended
            # (AuditChainError, audit lock timeout) this raises, update_broker_config writes nothing, and the client
            # gets 503 audit_unavailable with broker.json and the live snapshot unchanged.
            audit(kind, request, subject, detail or {})
        try:
            updated = update_broker_config(broker.config_path, checked)
        except BrokerError:
            raise
        except ValueError as exc:
            raise BrokerError(f'The broker configuration is invalid: {str(exc)[:400]}', 'invalid_request') from None
        broker.reload(actor=actor(request))
        return updated

    def not_found(what: str):
        return BrokerError(f'{what} not found', 'not_found')

    def upstream_row(upstream_id: str, upstream: Upstream) -> dict:
        snap = broker.snapshot()
        readiness = snap.catalog.readiness(upstream_id)
        credential = None
        if upstream.credential:
            try:
                credential = broker.vault.record(upstream.credential).validation
            except BrokerError:
                credential = 'missing'
        row = upstream.model_dump(mode='json')
        row.update({'id': upstream_id, 'ready': readiness is None,
                    'readiness': None if readiness is None else {'reason': readiness[0], 'detail': readiness[1]},
                    'credential_validation': credential, 'catalog_status': snap.catalog.status_of(upstream_id)})
        return row

    def origin_for(upstream_id: str, purpose: str) -> str:
        cfg = config()
        if purpose == 'advisor':
            advisor = cfg.advisors.get(upstream_id)
            if advisor is None or not advisor.url:
                raise BrokerError(f'Advisor {upstream_id} has no url to bind a credential to', 'invalid_request')
            return parse_origin(advisor.url, allow_http_lan=cfg.policy.allow_http_lan)
        upstream = cfg.upstreams.get(upstream_id)
        if upstream is None:
            raise BrokerError(f'Upstream {upstream_id} does not exist', 'not_found')
        try:
            origin = upstream.origin(allow_http_lan=cfg.policy.allow_http_lan)
        except ValueError as exc:
            raise BrokerError(f'Upstream {upstream_id} has an unusable base_url: {exc}', 'invalid_request') from None
        if origin is None:
            raise BrokerError(f'Upstream {upstream_id} has no base_url', 'upstream_unconfigured')
        return origin

    async def probe(upstream_id: str, purpose: str, credential=None):
        snap = broker.snapshot()
        adapter = snap.adapters.get(upstream_id)
        if adapter is None:
            raise BrokerError(f'Upstream {upstream_id} has no adapter', 'upstream_unconfigured')
        deadline = time.monotonic() + float(broker.settings.broker_connect_timeout) + float(
            broker.settings.broker_first_byte_timeout)
        grant = snap.issuer.issue(purpose, upstream_id=upstream_id, decision_id=uuid.uuid4().hex,
                                  request_id=uuid.uuid4().hex, deadline=deadline, credential=credential)
        return await adapter.probe(grant, broker.egress)

    async def validate_credential(cred_id: str, upstream_id: str, *, staged: bool = False):
        """Probe the upstream with the credential (purpose validate). Returns (ok, code)."""
        cfg = config()
        upstream = cfg.upstreams.get(upstream_id)
        if upstream is None:
            return False, 'upstream_unconfigured'
        origin = origin_for(upstream_id, 'inference')
        handle = broker.vault.validation_handle(cred_id, origin=origin, template=upstream.effective_auth(),
                                                staged=staged)
        result = await probe(upstream_id, 'validate', credential=handle)
        return bool(result.ok), (None if result.ok else (result.code or f'status_{result.status}'))

    def credential_meta(record, request: Request | None = None) -> dict:
        data = record.to_dict()
        data.pop('path', None)  # a local file path is operator detail, not credential metadata for HTTP readers
        if request is not None and getattr(principal_of(request), 'kind', None) == 'anonymous_loopback'                 and data.get('source') in ('env', 'file'):
            # a keyless caller never learns anything derived from a host value (not even its 32-bit fingerprint)
            data['fingerprint'] = None
        return data

    def visible_key(principal, key_id: str | None) -> str | None:
        """broker:admin sees every key; broker:read sees its own key only."""
        if allows(principal, 'broker:admin'):
            return key_id
        own = getattr(principal, 'key_id', None)
        if own is None or (key_id is not None and key_id != own):
            raise BrokerError('broker:read principals see their own key only', 'scope_denied', origin='policy')
        return own

    # -- status ----------------------------------------------------------------------------------------------------------
    @app.get('/api/broker', tags=['broker'])
    def broker_status():
        snap = broker.snapshot()
        cfg = snap.config
        enabled = cfg.enabled_upstreams()
        healthy = 0
        for upstream_id, upstream in enabled.items():
            if snap.catalog.readiness(upstream_id) is None and broker.breakers.get(
                    upstream_id, upstream.credential).allow(consume=False):
                healthy += 1
        try:
            vault = {'backend': broker.vault.backend_name, 'sealed': bool(broker.vault.sealed)}
        except Exception:
            vault = {'backend': 'unavailable', 'sealed': False}
        try:
            report = broker.audit.verify()
            audit_state = {'ok': bool(report.ok), 'records': int(report.records)}
        except Exception:
            audit_state = {'ok': False, 'records': None}
        hop = None
        if broker.configured:
            try:
                hop = broker.instance_hop()
            except Exception:
                hop = None
        return {'role': broker.settings.role, 'orchestrated': broker.settings.role == 'managed',
                'cloud_policy': {'file': cfg.policy.cloud_policy, 'ceiling': broker.settings.broker_cloud_ceiling,
                                 'effective': broker.effective_cloud_policy()},
                'egress': broker.settings.broker_egress, 'vault': vault,
                'upstreams': {'total': len(cfg.upstreams), 'enabled': len(enabled), 'healthy': healthy},
                'virtual_models': sorted(cfg.virtual_models), 'advisors': sorted(cfg.advisors),
                'audit': audit_state, 'instance_hop': hop, 'load_error': snap.load_error}

    @app.post('/api/broker/reload', tags=['broker'])
    def broker_reload(request: Request):
        return broker.reload(actor=actor(request))

    @app.get('/api/broker/profiles', tags=['broker'])
    def broker_profiles():
        rows = []
        for profile in sorted(PROFILES.values(), key=lambda p: p.id):
            rows.append({'id': profile.id, 'kind': profile.kind, 'adapter': profile.adapter, 'auth': profile.auth,
                         'auth_optional': profile.auth_optional, 'default_base_url': profile.default_base_url,
                         'surfaces': sorted(profile.surfaces), 'data_modalities': sorted(profile.data_modalities),
                         'image_formats': sorted(profile.image_formats), 'features': sorted(profile.features),
                         'operator_surfaces': profile.operator_surfaces, 'static_headers': list(profile.static_headers)})
        return {'object': 'list', 'data': rows}

    # -- upstreams -------------------------------------------------------------------------------------------------------
    @app.get('/api/broker/upstreams', tags=['broker'])
    def upstream_list():
        cfg = config()
        return {'object': 'list', 'data': [upstream_row(uid, up) for uid, up in sorted(cfg.upstreams.items())]}

    @app.get('/api/broker/upstreams/{id}', tags=['broker'])
    def upstream_get(id: str):
        upstream = config().upstreams.get(id)
        if upstream is None:
            raise not_found('Upstream')
        return upstream_row(id, upstream)

    @app.put('/api/broker/upstreams/{id}', tags=['broker'])
    def upstream_put(id: str, upstream: Upstream, request: Request):
        if upstream.credential:
            try:
                record = broker.vault.record(upstream.credential)
            except BrokerError:
                raise BrokerError(f'Credential {upstream.credential} does not exist; write it first (PUT '
                                  f'/api/broker/credentials/{upstream.credential})', 'invalid_request') from None
            if record.upstream_id != id:
                raise BrokerError(f'Credential {upstream.credential} is bound to another upstream', 'invalid_request')
        previous = config().upstreams.get(id)

        def change(cfg):
            cfg.upstreams[id] = upstream
        updated = write(request, 'upstream.write', id, change, {'profile': upstream.profile, 'enabled': upstream.enabled})
        new = updated.upstreams[id]
        try:
            new_origin = new.origin(allow_http_lan=updated.policy.allow_http_lan)
        except ValueError:
            new_origin = None
        old_origin = None
        if previous is not None:
            try:
                old_origin = previous.origin(allow_http_lan=updated.policy.allow_http_lan)
            except ValueError:
                old_origin = None
        reset = []
        if previous is None or old_origin != new_origin:
            try:
                reset = broker.vault.invalidate_for_upstream(id, origin=new_origin)
            except BrokerError:
                reset = []
        row = upstream_row(id, new)
        row['credentials_reset'] = reset
        return JSONResponse(row, status_code=200 if previous is not None else 201)

    @app.delete('/api/broker/upstreams/{id}', tags=['broker'])
    def upstream_delete(id: str, request: Request):
        if id not in config().upstreams:
            raise not_found('Upstream')

        def change(cfg):
            cfg.upstreams.pop(id, None)
        write(request, 'upstream.write', id, change, {'deleted': True})
        return {'id': id, 'deleted': True}

    @app.post('/api/broker/upstreams/{id}/test', tags=['broker'])
    async def upstream_test(id: str, request: Request):
        refuse_managed('Upstream tests')
        if id not in config().upstreams:
            raise not_found('Upstream')
        result = await probe(id, 'test')
        return {'ok': bool(result.ok), 'status': result.status, 'latency_ms': result.latency_ms, 'code': result.code}

    @app.get('/api/broker/upstreams/{id}/health', tags=['broker'])
    def upstream_health(id: str):
        if id not in config().upstreams:
            raise not_found('Upstream')
        return broker.upstream_health(id)

    @app.post('/api/broker/upstreams/{id}/breaker/reset', tags=['broker'])
    def upstream_breaker_reset(id: str, request: Request):
        if id not in config().upstreams:
            raise not_found('Upstream')
        count = broker.breakers.reset(id)
        return {'id': id, 'reset': count}

    # -- credentials -----------------------------------------------------------------------------------------------------
    @app.get('/api/broker/credentials', tags=['broker'])
    def credential_list(request: Request):
        data = [credential_meta(r, request) for r in broker.vault.records()]
        return JSONResponse({'object': 'list', 'data': data}, headers=NO_STORE)

    @app.put('/api/broker/credentials/{id}', tags=['broker'])
    async def credential_put(id: str, body: CredentialBody, request: Request):
        require_transport(request)
        refuse_managed('Credential writes')
        sources = [name for name in ('secret', 'env', 'file') if getattr(body, name) is not None]
        if len(sources) != 1:
            raise BrokerError('Give exactly one of secret, env or file', 'invalid_request')
        require_host_source(request, body)
        studio_source(request, body, body.upstream, body.purpose)
        origin = origin_for(body.upstream, body.purpose)
        vault = broker.vault
        try:
            vault.record(id)
            existed = True
        except BrokerError:
            existed = False
        kw = {'upstream_id': body.upstream, 'origin': origin, 'purpose': body.purpose}
        # audited first: an unavailable audit log refuses the write before the vault changes
        audit('credential.put', request, id, {'upstream': body.upstream, 'source': source_of(body),
                                               'purpose': body.purpose})
        record = vault_put(vault, id, body, kw, request, 'credential.put')
        if body.purpose == 'inference':
            if body.validate_:
                ok, code = await validate_credential(id, body.upstream)
                # the outcome is audited before it is recorded; if that append fails the credential stays
                # 'unvalidated' (stored and audited above, but not usable until a later audited validation)
                audit('credential.validate', request, id, {'ok': ok, 'code': code})
                record = vault.mark_validated(id, ok, code, None, origin=origin if ok else None)
                if not ok:
                    return JSONResponse({'error': {'message': 'The upstream refused the credential',
                                                   'type': 'credential_validation_failed', 'param': None,
                                                   'code': 'credential_validation_failed'},
                                         'credential': credential_meta(record, request), 'status_class': code},
                                        status_code=422, headers={**NO_STORE, 'X-Titanos-Error-Origin': 'upstream'})
            else:
                record = vault.mark_validated(id, True, 'skipped', None)
        else:
            record = vault.mark_validated(id, True, 'skipped', None)
        return JSONResponse(credential_meta(record, request), status_code=200 if existed else 201, headers=NO_STORE)

    @app.post('/api/broker/credentials/{id}/rotate', tags=['broker'])
    async def credential_rotate(id: str, body: RotateBody, request: Request):
        require_transport(request)
        refuse_managed('Credential writes')
        sources = [name for name in ('secret', 'env', 'file') if getattr(body, name) is not None]
        if len(sources) != 1:
            raise BrokerError('Give exactly one of secret, env or file', 'invalid_request')
        require_host_source(request, body)
        vault = broker.vault
        current = vault.record(id)
        upstream_id = current.upstream_id
        studio_source(request, body, upstream_id, current.purpose)
        if current.source == 'vault' and body.secret is not None and current.purpose == 'inference':
            vault.stage_rotation(id, body.secret)  # the staging slot only: the live value keeps serving
            ok, code = await validate_credential(id, upstream_id, staged=True)
            if not ok:
                vault.discard_rotation(id)
                audit('credential.rotate', request, id, {'ok': False, 'code': code})
                raise BrokerError('The upstream refused the new credential; the current one keeps serving',
                                  'credential_validation_failed', origin='upstream')
            try:
                audit('credential.rotate', request, id, {'ok': True})  # before the swap: no unaudited rotation
            except BaseException:
                vault.discard_rotation(id)
                raise
            record = vault.commit_rotation(id)
            return JSONResponse(credential_meta(record, request), headers=NO_STORE)
        kw = {'upstream_id': upstream_id, 'origin': current.origin, 'purpose': current.purpose}
        audit('credential.rotate', request, id, {'source': source_of(body)})  # first, as for a put
        record = vault_put(vault, id, body, kw, request, 'credential.rotate')
        ok, code = (True, 'skipped') if current.purpose != 'inference' else await validate_credential(id, upstream_id)
        if current.purpose == 'inference':
            audit('credential.validate', request, id, {'ok': ok, 'code': code})  # before the outcome is recorded
        record = vault.mark_validated(id, ok, code, None)
        if not ok:
            raise BrokerError('The upstream refused the new credential', 'credential_validation_failed',
                              origin='upstream')
        return JSONResponse(credential_meta(record, request), headers=NO_STORE)

    @app.post('/api/broker/credentials/{id}/validate', tags=['broker'])
    async def credential_validate(id: str, request: Request):
        require_transport(request)
        refuse_managed('Credential validation')
        vault = broker.vault
        record = vault.record(id)
        if record.purpose != 'inference':
            raise BrokerError('Only inference credentials are validated against an upstream', 'invalid_request')
        origin = origin_for(record.upstream_id, 'inference')
        if record.origin != origin and not allows(principal_of(request), 'admin'):
            # re-binding to a changed base_url presents the stored value to the new origin; whoever may also edit
            # upstreams (broker:admin) could otherwise move a write-only credential to a host of their choosing
            raise BrokerError('The upstream origin changed since this credential was bound; re-binding it to the new '
                              'origin presents the credential there and needs the admin scope', 'scope_denied',
                              origin='policy')
        ok, code = await validate_credential(id, record.upstream_id)
        audit('credential.validate', request, id, {'ok': ok, 'code': code})  # before the outcome is recorded
        record = vault.mark_validated(id, ok, code, None, origin=origin if ok else None)
        if not ok:
            raise BrokerError('The upstream refused the credential', 'credential_validation_failed', origin='upstream')
        return JSONResponse(credential_meta(record, request), headers=NO_STORE)

    @app.delete('/api/broker/credentials/{id}', tags=['broker'])
    def credential_delete(id: str, request: Request):
        require_transport(request)
        refuse_managed('Credential writes')
        broker.vault.record(id)  # 404 not_found before anything is audited
        audit('credential.delete', request, id, {})  # before the delete, as for every other write
        try:
            broker.vault.delete(id)
        except BrokerError as exc:
            audit_refusal('credential.delete', request, id, exc)
            raise
        return JSONResponse({'id': id, 'deleted': True}, headers=NO_STORE)

    # -- published aliases and virtual models ----------------------------------------------------------------------------
    @app.get('/api/broker/models', tags=['broker'])
    def alias_list():
        return {'object': 'list', 'data': [{'name': name, **alias.model_dump(mode='json')}
                                           for name, alias in sorted(config().models.items())]}

    @app.get('/api/broker/models/{name:path}', tags=['broker'])
    def alias_get(name: str):
        alias = config().models.get(name)
        if alias is None:
            raise not_found('Published model')
        return {'name': name, **alias.model_dump(mode='json')}

    @app.put('/api/broker/models/{name:path}', tags=['broker'])
    def alias_put(name: str, alias: PublishedModel, request: Request):
        existed = name in config().models

        def change(cfg):
            cfg.models[name] = alias
        write(request, 'alias.write', name, change, {'target': alias.target})
        return JSONResponse({'name': name, **alias.model_dump(mode='json')}, status_code=200 if existed else 201)

    @app.delete('/api/broker/models/{name:path}', tags=['broker'])
    def alias_delete(name: str, request: Request):
        if name not in config().models:
            raise not_found('Published model')

        def change(cfg):
            cfg.models.pop(name, None)
        write(request, 'alias.write', name, change, {'deleted': True})
        return {'name': name, 'deleted': True}

    @app.get('/api/broker/virtual-models', tags=['broker'])
    def virtual_list():
        return {'object': 'list', 'data': [{'name': name, **vm.model_dump(mode='json')}
                                           for name, vm in sorted(config().virtual_models.items())]}

    @app.get('/api/broker/virtual-models/{name:path}', tags=['broker'])
    def virtual_get(name: str):
        vm = config().virtual_models.get(name)
        if vm is None:
            raise not_found('Virtual model')
        return {'name': name, **vm.model_dump(mode='json')}

    @app.put('/api/broker/virtual-models/{name:path}', tags=['broker'])
    def virtual_put(name: str, virtual: VirtualModel, request: Request):
        existed = name in config().virtual_models

        def change(cfg):
            cfg.virtual_models[name] = virtual
        write(request, 'virtual.write', name, change, {'capability': virtual.capability, 'strategy': virtual.strategy})
        return JSONResponse({'name': name, **virtual.model_dump(mode='json')}, status_code=200 if existed else 201)

    @app.delete('/api/broker/virtual-models/{name:path}', tags=['broker'])
    def virtual_delete(name: str, request: Request):
        if name not in config().virtual_models:
            raise not_found('Virtual model')

        def change(cfg):
            cfg.virtual_models.pop(name, None)
        write(request, 'virtual.write', name, change, {'deleted': True})
        return {'name': name, 'deleted': True}

    # -- policy, advisors and prices -------------------------------------------------------------------------------------
    @app.get('/api/broker/policy', tags=['broker'])
    def policy_get():
        cfg = config()
        return {**cfg.policy.model_dump(mode='json'), 'ceiling': broker.settings.broker_cloud_ceiling,
                'effective_cloud_policy': broker.effective_cloud_policy(), 'role': broker.settings.role}

    @app.put('/api/broker/policy', tags=['broker'])
    def policy_put(policy: BrokerPolicy, request: Request):
        principal = principal_of(request)
        current = config().policy
        changed = forbidden_policy_changes(current, policy)
        if is_studio(principal):
            changed = changed - {'anonymous_cloud'}  # R29: Studio may turn keyless cloud use on or off, never allow_http_lan
        if changed and (principal.kind == 'anonymous_loopback' or not broker.authenticator.enforced):
            raise BrokerError('anonymous_cloud and allow_http_lan cannot be changed in keyless mode; edit broker.json on '
                              'the host or use a key holding broker:admin', 'keyless_mode_refused', origin='policy')
        ceiling = broker.settings.broker_cloud_ceiling
        if CLOUD_POLICIES.index(policy.cloud_policy) > CLOUD_POLICIES.index(ceiling):
            raise BrokerError(f'cloud_policy {policy.cloud_policy} exceeds TITANOS_BROKER_CLOUD_CEILING={ceiling}',
                              'invalid_request', origin='policy')

        def change(cfg):
            cfg.policy = policy
        write(request, 'policy.write', 'policy', change, {'cloud_policy': policy.cloud_policy,
                                                         'default_location': policy.default_location,
                                                         'changed_guarded_fields': sorted(changed)})
        return policy_get()

    @app.get('/api/broker/advisors', tags=['broker'])
    def advisor_list():
        return {'object': 'list', 'data': [{'id': aid, **spec.model_dump(mode='json')}
                                           for aid, spec in sorted(config().advisors.items())]}

    @app.put('/api/broker/advisors/{id}', tags=['broker'])
    def advisor_put(id: str, advisor: AdvisorSpec, request: Request):
        existed = id in config().advisors

        def change(cfg):
            cfg.advisors[id] = advisor
        write(request, 'advisor.write', id, change, {'profile': advisor.profile})
        return JSONResponse({'id': id, **advisor.model_dump(mode='json')}, status_code=200 if existed else 201)

    @app.post('/api/broker/advisors/{id}/test', tags=['broker'])
    async def advisor_test(id: str, request: Request):
        refuse_managed('Advisor tests')
        spec = config().advisors.get(id)
        if spec is None:
            raise not_found('Advisor')
        if spec.profile == 'pending-contract':
            raise BrokerError('The pending-contract advisor slot has no wire contract', 'advisor_profile_unconfigured')
        advisor = broker.snapshot().advisors.get(id)
        if advisor is None or advisor.callable() is not None:
            raise BrokerError(f'Advisor {id} cannot be called ({getattr(advisor, "callable", lambda: None)()})',
                              'advisor_profile_unconfigured')
        from .advisor import AdvisorRequest, AdvisorSummary
        probe_request = AdvisorRequest(request_id=uuid.uuid4().hex, surface='openai.chat', capability='chat',
                                       requested_model='auto', summary=AdvisorSummary(), candidates=[])
        started = time.perf_counter()
        try:
            answer = await advisor.advise(probe_request, deadline=time.monotonic() + advisor.timeout_seconds())
            ok, reason = answer is not None, None if answer is not None else 'no_answer'
        except BrokerError as exc:
            ok, reason = False, exc.code
        except Exception as exc:  # transport trouble is the test's answer, not a server error
            ok, reason = False, type(exc).__name__
        return {'id': id, 'ok': ok, 'ms': round((time.perf_counter() - started) * 1000.0, 3), 'reason': reason}

    @app.get('/api/broker/prices', tags=['broker'])
    def price_list():
        return {'object': 'list', 'data': [row.model_dump(mode='json') for row in config().prices]}

    @app.put('/api/broker/prices', tags=['broker'])
    def price_put(rows: list[PriceRow], request: Request):
        def change(cfg):
            cfg.prices = list(rows)
        write(request, 'prices.write', 'prices', change, {'rows': len(rows)})
        return price_list()

    @app.get('/api/broker/catalog', tags=['broker'])
    async def catalog_get(request: Request, refresh: bool = False):
        principal = principal_of(request)
        if refresh:
            await broker.refresh_catalogs(principal, force=True)
        snap = broker.snapshot()
        status = {uid: value for uid, value in snap.catalog.status().items()
                  if broker.visible_upstream(principal, uid)}
        return {'object': 'list', 'data': snap.catalog.entries(principal), 'upstream_status': status}

    # -- observability ---------------------------------------------------------------------------------------------------
    @app.post('/api/broker/explain', tags=['broker'])
    async def explain(body: ExplainBody, request: Request, advisor: bool = False):
        principal = principal_of(request)
        subject = principal
        if body.key_id is not None:
            if not allows(principal, 'admin'):
                raise BrokerError('Explaining for another key needs admin', 'scope_denied', origin='policy')
            subject = broker.keys.to_principal(broker.keys.get(body.key_id))
        return await broker.explain(body.surface, body.body, body.headers, subject, advisor=advisor)

    @app.get('/api/broker/decisions', tags=['broker'])
    def decisions(request: Request, limit: int = Query(100, ge=1, le=10000), key_id: str | None = None,
                  model: str | None = None, since: str | None = None):
        principal = principal_of(request)
        try:
            data = broker.decisions.query(limit=limit, key_id=key_id, model=model, since=since, principal=principal)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f'Invalid query: {exc}', 'invalid_request') from None
        return {'object': 'list', 'data': data}

    @app.get('/api/broker/usage', tags=['broker'])
    def usage(request: Request, group_by: str = 'key,upstream,model,surface,day', since: str | None = None,
              until: str | None = None, key_id: str | None = None):
        principal = principal_of(request)
        key_id = visible_key(principal, key_id)
        groups = [g.strip() for g in group_by.split(',') if g.strip()]
        try:
            rows = broker.usage.rollup(groups, since=since, until=until, key_id=key_id)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f'Invalid usage query: {exc}', 'invalid_request') from None
        return {'object': 'list', 'rows': rows}

    @app.get('/api/broker/metrics', tags=['broker'])
    def metrics():
        return broker.metrics_dict()

    # -- R30 env discovery and R31 one-step connect ---------------------------------------------------------------------
    def keyless_needs_studio(principal) -> None:
        if principal.kind == 'anonymous_loopback' and not is_studio(principal):
            raise BrokerError('Keyless provider setup is accepted only from Studio on loopback (X-Titanos-Studio: 1); use '
                              'the titanos CLI (titanos broker quickstart, titanos broker connect)', 'keyless_mode_refused',
                              origin='policy')

    @app.get('/api/broker/env-candidates', tags=['broker'])
    def env_candidate_list(request: Request):
        from .connect import env_candidates
        principal = principal_of(request)
        keyless_needs_studio(principal)
        return JSONResponse({'object': 'list', 'data': env_candidates(), 'role': broker.settings.role},
                            headers=NO_STORE)

    @app.post('/api/broker/connect', tags=['broker'])
    async def provider_connect(body: dict, request: Request):
        from .connect import ConnectRequest, connect
        principal = principal_of(request)
        keyless_needs_studio(principal)
        if not is_studio(principal) and not allows(principal, 'credentials:write'):
            raise BrokerError('Provider connect stores a credential and needs credentials:write', 'scope_denied',
                              origin='policy')
        require_transport(request)
        refuse_managed('Provider connect')
        try:
            spec = ConnectRequest.model_validate(body)
        except ValueError as exc:
            raise BrokerError(f'Invalid connect request: {_errors(exc)}', 'invalid_request') from None
        studio = is_studio(principal)
        result = await connect(
            broker, spec, write=lambda kind, subject, change, detail: write(request, kind, subject, change, detail),
            audit=lambda kind, subject, detail: audit(kind, request, subject, detail),
            host_sources=studio or allows(principal, 'admin'),
            may_set_anonymous_cloud=studio or (broker.authenticator.enforced and allows(principal, 'broker:admin')),
            studio=studio and not allows(principal, 'admin'),
            keyless_owner=principal.kind == 'anonymous_loopback' and not broker.authenticator.enforced
            and bool(getattr(broker.authenticator, 'bind_loopback', False)))
        return JSONResponse(result, status_code=200, headers=NO_STORE)

    # -- inbound keys ----------------------------------------------------------------------------------------------------
    def key_answer(record, token: str | None = None) -> dict:
        public = record.public()
        out = {'id': record.id, 'name': public.get('name'), 'scopes': public.get('scopes'),
               'constraints': public.get('constraints'), 'limits': public.get('limits'), 'budget': public.get('budget'),
               'expires_at': public.get('expires_at'), 'status': public.get('status')}
        if token is not None:
            out['secret'] = token
        return out

    @app.get('/api/keys', tags=['keys'])
    def key_list():
        return {'object': 'list', 'data': [record.public() for record in broker.keys.list()]}

    @app.post('/api/keys', tags=['keys'], status_code=201)
    def key_mint(spec: KeySpec, request: Request):
        require_transport(request)
        record, token = broker.keys.mint(spec, created_by=actor(request))
        return JSONResponse(key_answer(record, token), status_code=201, headers=NO_STORE)

    @app.get('/api/keys/self', tags=['keys'])
    def key_self(request: Request):
        principal = principal_of(request)
        remaining = None
        if principal.budget is not None:
            try:
                remaining = broker.limiter.remaining(principal)
            except Exception:
                remaining = None
        constraints = principal.constraints
        return {'id': principal.id, 'kind': principal.kind, 'scopes': sorted(principal.scopes),
                'constraints': {'models': list(constraints.models), 'upstreams': list(constraints.upstreams),
                                'target_classes': sorted(constraints.target_classes),
                                'locations': sorted(constraints.locations)},
                'limits': None if principal.limits is None else {'rpm': principal.limits.rpm, 'tpm': principal.limits.tpm,
                                                                 'concurrent': principal.limits.concurrent},
                'budget_remaining': remaining,
                'expires_at': principal.expires_at.isoformat().replace('+00:00', 'Z') if principal.expires_at else None}

    @app.get('/api/keys/{id}', tags=['keys'])
    def key_get(id: str):
        return broker.keys.get(id).public()

    @app.patch('/api/keys/{id}', tags=['keys'])
    def key_patch(id: str, patch: KeyPatch, request: Request):
        return broker.keys.update(id, patch, actor=actor(request)).public()

    @app.post('/api/keys/{id}/revoke', tags=['keys'])
    def key_revoke(id: str, request: Request):
        return broker.keys.revoke(id, actor=actor(request)).public()

    @app.post('/api/keys/{id}/rotate', tags=['keys'])
    def key_rotate(id: str, request: Request):
        require_transport(request)
        record, token = broker.keys.rotate(id, actor=actor(request))
        return JSONResponse({'id': record.id, 'secret': token}, headers=NO_STORE)

    # -- audit -----------------------------------------------------------------------------------------------------------
    @app.get('/api/audit', tags=['audit'])
    def audit_query(since: str | None = None, kind: str | None = None, limit: int = Query(500, ge=1, le=10000)):
        try:
            data = broker.audit.query(since=since, kind=kind, limit=limit)
        except (TypeError, ValueError) as exc:
            raise BrokerError(f'Invalid audit query: {exc}', 'invalid_request') from None
        return {'object': 'list', 'data': data}

    @app.get('/api/audit/verify', tags=['audit'])
    def audit_verify():
        return broker.audit.verify().to_dict()


def _errors(exc) -> str:
    """Validation errors without input values (the body may carry a secret)."""
    errors = getattr(exc, 'errors', None)
    if callable(errors):
        try:
            return '; '.join('.'.join(str(x) for x in e.get('loc', ())) + ': ' + str(e.get('msg', ''))
                             for e in errors(include_input=False))[:400]
        except Exception:
            pass
    return 'the body does not match the schema'


__all__ = ['CredentialBody', 'ExplainBody', 'RotateBody', 'STUDIO_ID', 'is_studio', 'register_admin_routes']
