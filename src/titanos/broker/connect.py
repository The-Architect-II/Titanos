"""R30 environment key discovery and R31 one-step provider connect, shared by the HTTP admin routes
(GET /api/broker/env-candidates, POST /api/broker/connect) and the CLI (titanos broker quickstart, titanos broker connect).

env_candidates(): for every profile row that declares well-known credential variable NAMES (profiles.Profile.env_names),
which of them are set (non-empty) in this process. Names and booleans only: never a value, a length or a prefix.

connect(): in order, each step audited before it changes anything (broker.json writes go through the caller's write
callback, which validates, audits and replaces atomically; vault writes are audited first, as in admin.py):
1. refuse in managed role (409 role_forbids) and for slot profiles; the upstream id defaults to the provider id, the
   credential id to <upstream>-key;
2. create the upstream from its profile (or update an existing upstream of the same profile): base_url, Azure binding and
   models when given, catalog live when the profile has a models endpoint; it stays disabled until the credential
   validates;
3. store the credential (secret value, env: reference or file: reference; host sources need the caller's right to name
   them) bound to the upstream's exact origin, and validate it with the profile's free probe (purpose validate); a refused
   credential answers 422 credential_validation_failed and leaves the upstream disabled;
4. bind the credential, enable the upstream (unless enable is false) and raise policy.cloud_policy from off to explicit
   (never lowered, never above TITANOS_BROKER_CLOUD_CEILING); set policy.anonymous_cloud when asked (only for a caller
   allowed to change it: the offline CLI, a keyed broker:admin caller, or the R29 keyless Studio principal). In standalone
   role on a loopback bind with no key enforced (keyless_owner), an unset anonymous_cloud defaults to TRUE: the keyless
   loopback caller is the machine's owner, so a provider connected without a key works for the owner's keyless SDKs and
   Studio at once (owner direction 2026-10-02: put in a key and it just works); anonymous_cloud: false keeps it off.
   allow_store (optional) sets the upstream's data_policy.allow_store, which provider-side state (files, batches,
   stored Responses, conversations) needs besides the caller's cloud:store;
5. refresh the live catalog (purpose catalog) and report the discovered models;
6. optionally (aliases: true) publish an alias for every discovered model whose id is a valid local-style name that
   collides with nothing (local models, other aliases, virtual models), at most MAX_ALIASES, so a stock SDK can send the
   provider's bare model id.
The answer never contains a secret: the upstream summary, the credential metadata (fingerprint only; none for a keyless
caller's env: or file: reference), the discovered models, the published aliases, the effective cloud policy and next
steps. The R29 keyless Studio principal may name only the profile's well-known credential variables, only for the
profile's default origin, and never a file: path (admin.studio_host_source).
"""
from __future__ import annotations

import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import BrokerError
from .naming import LOCAL_NAME_RE, valid_upstream_id
from .profiles import PROFILES, env_names

MAX_ALIASES = 256
ID_RE = re.compile(r'[a-z][a-z0-9_-]{0,63}')
ALIAS_CAPABILITIES = ('chat', 'tools', 'vision', 'responses', 'embeddings', 'speech', 'transcription', 'images',
                      'moderation')
CLOUD_ORDER = ('off', 'explicit', 'fallback')


def env_candidates(environ: Mapping[str, str] | None = None) -> list[dict]:
    """[{provider, kind, variables: [{name, set}], any_set}] for every profile that declares credential variable names.
    Only names and booleans: a value is read solely to test that it is non-empty and is never returned or kept."""
    environ = os.environ if environ is None else environ
    rows = []
    for provider, names in sorted(env_names().items()):
        variables = [{'name': name, 'set': bool(environ.get(name))} for name in names]
        rows.append({'provider': provider, 'kind': PROFILES[provider].kind, 'variables': variables,
                     'any_set': any(v['set'] for v in variables)})
    return rows


def first_set(provider: str, environ: Mapping[str, str] | None = None) -> str | None:
    """The first well-known variable name of provider that is set, or None."""
    environ = os.environ if environ is None else environ
    for name in PROFILES[provider].env_names:
        if environ.get(name):
            return name
    return None


class ConnectCredential(BaseModel):
    """Exactly one of value (stored in the vault), env (a variable NAME on the server host) or file (an owner-only
    absolute path on the server host)."""
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    value: str | None = Field(None, min_length=1, max_length=8192)
    env: str | None = Field(None, min_length=1, max_length=128)
    file: str | None = Field(None, min_length=1, max_length=4096)

    @model_validator(mode='after')
    def exactly_one(self):
        if sum(x is not None for x in (self.value, self.env, self.file)) != 1:
            raise ValueError('give exactly one of value, env or file')
        return self

    def kind(self) -> str:
        return 'vault' if self.value is not None else ('env' if self.env is not None else 'file')


class ConnectRequest(BaseModel):
    """POST /api/broker/connect (R31)."""
    provider: str = Field(min_length=1, max_length=64)
    credential: ConnectCredential | None = None  # None only for profiles that need no credential (auth none)
    upstream: str | None = Field(None, max_length=32)
    credential_id: str | None = Field(None, max_length=64)
    models: list[str] | None = Field(None, max_length=4096)
    base_url: str | None = Field(None, max_length=2048)
    azure: dict | None = None
    enable: bool = True
    validate_: bool = Field(True, alias='validate')
    aliases: bool = False
    anonymous_cloud: bool | None = None
    allow_store: bool | None = None

    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, populate_by_name=True)


Write = Callable[[str, str, Callable, dict], Any]
Audit = Callable[[str, str, dict], None]


async def connect(broker, request: ConnectRequest, *, write: Write, audit: Audit, host_sources: bool,
                  may_set_anonymous_cloud: bool, local_names: Callable[[], Any] | None = None, studio: bool = False,
                  keyless_owner: bool = False) -> dict:
    """Run the R31 steps (module docstring). write(kind, subject, change, detail) applies change to broker.json with the
    admin write checks and returns the updated BrokerConfig, leaving broker.snapshot() current; audit(kind, subject,
    detail) appends one audit record. host_sources: the caller may name env:/file: sources. may_set_anonymous_cloud:
    the caller may change policy.anonymous_cloud. studio: the R29 keyless Studio principal (narrower host sources).
    keyless_owner: standalone role, loopback bind, nothing enforced (an unset anonymous_cloud defaults to true; no
    fingerprint of a host source is returned). Raises BrokerError for every refusal."""
    from .config import Upstream
    settings = broker.settings
    if settings.role == 'managed':
        raise BrokerError('Provider connect writes credentials, which managed role refuses', 'role_forbids', status=409,
                          origin='policy')
    provider = request.provider
    profile = PROFILES.get(provider)
    if profile is None or profile.kind == 'slot' or profile.adapter == 'none':
        raise BrokerError(f'Unknown or unusable provider profile {provider!r}', 'invalid_request')
    upstream_id = request.upstream or provider
    if not valid_upstream_id(upstream_id):
        raise BrokerError(f'Invalid upstream id {upstream_id!r}', 'invalid_request')
    credential = request.credential
    needs_credential = not (profile.auth == 'none' or (profile.auth_optional and credential is None))
    if credential is None and needs_credential:
        raise BrokerError(f'{provider} needs a credential (value, env or file)', 'invalid_request')
    cred_id = request.credential_id or f'{upstream_id}-key'
    if credential is not None and not ID_RE.fullmatch(cred_id):
        raise BrokerError(f'Invalid credential id {cred_id!r}', 'invalid_request')
    if credential is not None and credential.kind() != 'vault' and not host_sources:
        raise BrokerError('env: and file: credential sources name values on the server host; they need the admin scope '
                          '(or the keyless Studio on loopback)', 'scope_denied', origin='policy')
    if studio and credential is not None:
        from .admin import studio_host_source
        current_row = broker.snapshot().config.upstreams.get(upstream_id)
        studio_host_source(credential.env, credential.file, provider,
                           request.base_url if request.base_url is not None else
                           (current_row.base_url if current_row is not None else None))
    if request.anonymous_cloud is not None and not may_set_anonymous_cloud:
        raise BrokerError('anonymous_cloud can be set only from the CLI on the host, a keyed broker:admin caller or '
                          'the keyless Studio', 'keyless_mode_refused', origin='policy')

    current = broker.snapshot().config
    existing = current.upstreams.get(upstream_id)
    if existing is not None and existing.profile != provider:
        raise BrokerError(f'Upstream {upstream_id} already uses profile {existing.profile}; choose another --upstream',
                          'broker_name_conflict')

    # 2. the upstream, disabled until its credential validates (an existing upstream keeps its enabled state)
    spec = existing.model_dump(mode='json', exclude_unset=True) if existing is not None else {'profile': provider}
    spec['profile'] = provider
    if request.base_url is not None:
        spec['base_url'] = request.base_url
    if request.azure is not None:
        spec['azure'] = request.azure
    if request.models:
        spec['models'] = list(dict.fromkeys(request.models))
    if request.allow_store is not None:
        spec['data_policy'] = {**(spec.get('data_policy') or {}), 'allow_store': bool(request.allow_store)}
    if existing is None and profile.catalog_path:
        spec['catalog'] = 'live'
    if existing is None:
        spec['enabled'] = False
    if credential is None:
        spec.pop('credential', None)
    try:
        upstream = Upstream.model_validate(spec)
    except ValueError as exc:
        raise BrokerError(f'The upstream is invalid: {str(exc)[:300]}', 'invalid_request') from None

    def put_upstream(cfg):
        cfg.upstreams[upstream_id] = upstream
    cfg = write('upstream.write', upstream_id, put_upstream, {'profile': provider, 'enabled': upstream.enabled,
                                                             'via': 'connect'})
    try:
        origin = cfg.upstreams[upstream_id].origin(allow_http_lan=cfg.policy.allow_http_lan)
    except ValueError as exc:
        raise BrokerError(f'Upstream {upstream_id} has an unusable base_url: {exc}', 'invalid_request') from None
    if origin is None:
        raise BrokerError(f'Upstream {upstream_id} has no base_url; pass base_url', 'invalid_request')

    # 3. the credential, validated against the profile's free probe. With validation on, the candidate is stored and
    #    probed under a scratch id first and only a validated one replaces cred_id: a refused key, or an env:/file:
    #    reference this server cannot resolve (503 credential_unavailable), never overwrites a working record.
    record = None
    validation = 'not_needed'
    if credential is not None:
        vault = broker.vault
        kw = {'upstream_id': upstream_id, 'origin': origin, 'purpose': 'inference'}
        audit('credential.put', cred_id, {'upstream': upstream_id, 'source': credential.kind(), 'purpose': 'inference',
                                          'via': 'connect'})

        def put(target_id):
            try:
                if credential.value is not None:
                    return vault.put_secret(target_id, credential.value, **kw)
                if credential.env is not None:
                    return vault.put_env_ref(target_id, credential.env, **kw)
                return vault.put_file_ref(target_id, credential.file, **kw)
            except BrokerError as exc:
                try:
                    audit('credential.put', cred_id, {'ok': False, 'code': exc.code})
                except Exception:
                    pass
                from .admin import without_path
                raise without_path(exc, credential.file) from None

        if request.validate_:
            check_id = f'{cred_id[:48]}-chk-{uuid.uuid4().hex[:8]}'
            try:
                put(check_id)
                ok, code = await validate(broker, check_id, upstream_id, origin,
                                          cfg.upstreams[upstream_id].effective_auth())
            finally:
                try:
                    vault.delete(check_id)
                except Exception:  # absent if the put itself was refused
                    pass
            audit('credential.validate', cred_id, {'ok': ok, 'code': code, 'via': 'connect'})
            if not ok:
                raise BrokerError(f'The upstream refused the credential ({code}); {upstream_id} stays disabled',
                                  'credential_validation_failed', origin='upstream')
            put(cred_id)
            record = vault.mark_validated(cred_id, True, code, None, origin=origin)
            validation = 'ok' if code is None else code
        else:
            put(cred_id)
            record = vault.mark_validated(cred_id, True, 'skipped', None)
            validation = 'skipped'

    # 4. bind, enable and raise the cloud policy (never lowered, never above the ceiling)
    ceiling = settings.broker_cloud_ceiling
    raised = {}
    anonymous_cloud = request.anonymous_cloud
    if anonymous_cloud is None and keyless_owner and may_set_anonymous_cloud and settings.role == 'standalone':
        anonymous_cloud = True  # the keyless loopback caller is the owner (module docstring)

    def finish(cfg):
        row = cfg.upstreams[upstream_id]
        if credential is not None:
            row.credential = cred_id
        if request.enable:
            row.enabled = True
        policy = cfg.policy
        if policy.cloud_policy == 'off' and CLOUD_ORDER.index(ceiling) >= CLOUD_ORDER.index('explicit'):
            policy.cloud_policy = 'explicit'
            raised['cloud_policy'] = 'explicit'
        if anonymous_cloud is not None and policy.anonymous_cloud != anonymous_cloud:
            policy.anonymous_cloud = bool(anonymous_cloud)
            raised['anonymous_cloud'] = policy.anonymous_cloud
    cfg = write('broker.connect', upstream_id, finish, {'provider': provider, 'credential': cred_id if credential else None,
                                                        'enabled': bool(request.enable), 'policy': dict(raised)})

    # 5. discovery
    snapshot = broker.snapshot()
    catalog_status = None
    if cfg.upstreams[upstream_id].enabled and cfg.upstreams[upstream_id].catalog == 'live':
        catalog_status = await snapshot.catalog.refresh_upstream(upstream_id)
    discovered = snapshot.catalog.upstream_models(upstream_id) if upstream_id in snapshot.config.upstreams else []

    # 6. aliases for the provider's bare model ids (opt-in)
    published: list[str] = []
    skipped: list[str] = []
    if request.aliases and discovered:
        caps = sorted(set(snapshot.catalog.capabilities(upstream_id)) & set(ALIAS_CAPABILITIES)) or ['chat']
        local = set(local_names() if local_names is not None else broker.local_models().keys())
        taken = set(cfg.models) | set(cfg.virtual_models) | local
        wanted = []
        for model in discovered:
            target = f'{model}@{upstream_id}'
            if not LOCAL_NAME_RE.fullmatch(model) or len(model) > 128:
                skipped.append(model)
            elif model in taken:
                if cfg.models.get(model) is None or cfg.models[model].target != target:
                    skipped.append(model)
            else:
                wanted.append(model)
        if len(wanted) > MAX_ALIASES:
            skipped.extend(wanted[MAX_ALIASES:])
            wanted = wanted[:MAX_ALIASES]
        if wanted:
            from .config import PublishedModel

            def publish(cfg):
                for model in wanted:
                    cfg.models[model] = PublishedModel(target=f'{model}@{upstream_id}', capabilities=caps)
            cfg = write('alias.write', upstream_id, publish, {'aliases': len(wanted), 'via': 'connect'})
            published = wanted

    row = cfg.upstreams[upstream_id]
    effective = cfg.policy.cloud_policy if CLOUD_ORDER.index(cfg.policy.cloud_policy) <= CLOUD_ORDER.index(ceiling) \
        else ceiling
    steps = []
    if effective == 'off':
        steps.append(f'TITANOS_BROKER_CLOUD_CEILING=off keeps every cloud call refused; raise it to use {upstream_id}')
    if not cfg.policy.anonymous_cloud and settings.role == 'standalone':
        steps.append('Keyless callers (and keyless Studio) reach this upstream only with policy.anonymous_cloud true; '
                     'otherwise use a key holding cloud:' + upstream_id)
    if discovered and not published:
        steps.append(f'Call a model as <model>@{upstream_id}, for example {discovered[0]}@{upstream_id}, or publish '
                     'aliases (aliases: true)')
    return {'upstream': {'id': upstream_id, 'profile': provider, 'enabled': row.enabled, 'catalog': row.catalog,
                         'credential': row.credential, 'base_url': row.effective_base_url()},
            'credential': None if record is None else _meta(record, hide_fingerprint=keyless_owner),
            'validation': validation, 'allow_store': bool(row.data_policy.allow_store),
            'cloud_policy': {'file': cfg.policy.cloud_policy, 'ceiling': ceiling, 'effective': effective},
            'anonymous_cloud': cfg.policy.anonymous_cloud, 'policy_changes': raised,
            'catalog_status': catalog_status, 'models': discovered[:1000], 'models_total': len(discovered),
            'aliases': published, 'aliases_skipped': skipped[:200], 'next_steps': steps}


def _meta(record, *, hide_fingerprint: bool = False) -> dict:
    data = record.to_dict()
    data.pop('path', None)
    if hide_fingerprint and data.get('source') in ('env', 'file'):
        data['fingerprint'] = None
    return data


async def validate(broker, cred_id: str, upstream_id: str, origin: str, template: str) -> tuple[bool, str | None]:
    """Probe the upstream with the credential (purpose validate): (ok, code). A profile without a probe path validates
    as ok with code 'skipped'."""
    snap = broker.snapshot()
    adapter = snap.adapters.get(upstream_id)
    if adapter is None:
        return False, 'upstream_unconfigured'
    handle = broker.vault.validation_handle(cred_id, origin=origin, template=template)
    deadline = time.monotonic() + float(broker.settings.broker_connect_timeout) + float(
        broker.settings.broker_first_byte_timeout)
    grant = snap.issuer.issue('validate', upstream_id=upstream_id, decision_id=uuid.uuid4().hex,
                              request_id=uuid.uuid4().hex, deadline=deadline, credential=handle)
    result = await adapter.probe(grant, broker.egress)
    if result.ok:
        return True, (result.code if result.code == 'skipped' else None)
    return False, (result.code or f'status_{result.status}')


__all__ = ['ALIAS_CAPABILITIES', 'ConnectCredential', 'ConnectRequest', 'MAX_ALIASES', 'connect', 'env_candidates',
           'first_set', 'validate']
