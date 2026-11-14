"""BrokerConfig v1: the schema of TITANOS_HOME/broker.json (or TITANOS_BROKER_CONFIG), extra=forbid everywhere.

- broker.json names only credential ids; a secret-shaped string anywhere refuses the whole document
  (BrokerConfigSecretError: a ValueError at load, HTTP 400 broker_config_contains_secret).
- Upstream ids match ^[a-z][a-z0-9_-]{0,31}$ ('store' is reserved: cloud:store is a scope).
- base_url: https; http only to a loopback IP literal, or to an RFC1918 IP literal when policy.allow_http_lan. Userinfo,
  query and fragment are refused; unspecified, link-local, multicast and IPv4-mapped literals are refused outright.
- Published alias and virtual names follow the local-name grammar and may not collide with each other or with local
  registry names (check_name_conflicts, 409 broker_name_conflict).
- Writes are atomic replaces under a FileLock (the config.py pattern) with a bounded os.replace retry, because Windows
  answers PermissionError while another process holds the target open.
Nothing here performs network I/O; reading a missing file yields the default (disabled) configuration.
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import tempfile
import time
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import BrokerConfigSecretError, BrokerError
from .naming import LOCAL_NAME_RE, REF_MAX, parse_ref, valid_upstream_id
from .profiles import DATA_MODALITIES, OPERATOR_SURFACE_PROFILES, PROFILES
from .redact import looks_secret

KEYLESS_FORBIDDEN_POLICY_FIELDS = frozenset({'anonymous_cloud', 'allow_http_lan'})
CAPABILITIES = ('chat', 'tools', 'vision', 'responses', 'embeddings', 'speech', 'transcription', 'images', 'moderation')
VIRTUAL_CAPABILITIES = ('chat', 'embeddings', 'speech', 'transcription', 'images', 'moderation')
FALLBACK_TRIGGERS = ('queue_full', 'rate_limit_error', 'model_not_loaded', 'worker_unavailable', 'model_load_failed',
                     'device_unavailable', 'upstream_rate_limited', 'upstream_unavailable', 'upstream_timeout', 'network',
                     'breaker_open')
MULTI_ATTEMPT_CAPABILITIES = frozenset({'chat', 'completions', 'responses', 'embeddings'})
# RFC 1918 private ranges, built numerically (10/8, 172.16/12, 192.168/16)
RFC1918 = (ipaddress.IPv4Network((0x0A000000, 8)), ipaddress.IPv4Network((0xAC100000, 12)),
           ipaddress.IPv4Network((0xC0A80000, 16)))
AZURE_ENDPOINT_HOSTS: tuple[str, ...] = tuple(PROFILES['azure_openai'].extras['endpoint_hosts'])
# cloud metadata endpoints that are not link-local (169.254.0.0/16 is refused as link-local)
METADATA_ADDRESSES = frozenset({ipaddress.ip_address('fd00:ec2::254')})
REPLACE_ATTEMPTS = 8
REPLACE_DELAY = 0.05

# hide_input_in_errors: a refused document's values (possibly secret-shaped) never appear in validation messages
_Strict = ConfigDict(extra='forbid', allow_inf_nan=False, hide_input_in_errors=True)


def _ip_literal(host: str):
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


_NUMERIC_LABEL = re.compile(r'(?i)0x[0-9a-f]*|[0-9]+')


def parse_origin(base_url: str, *, allow_http_lan: bool) -> str:
    """'scheme://host:port' of base_url: host IDNA-encoded (punycode) and lower-case, trailing dot stripped, default port
    filled in. ValueError for anything but https (or http to a loopback IP literal, or an RFC1918 IP literal with
    allow_http_lan), userinfo, query, fragment, a malformed host or a blocked IP literal."""
    if not isinstance(base_url, str) or not base_url or base_url != base_url.strip() or len(base_url) > 2048:
        raise ValueError('base_url must be a non-empty URL without surrounding whitespace')
    if any(c in base_url for c in '\\\r\n\t '):
        raise ValueError('base_url contains characters that are not allowed')
    parts = urlsplit(base_url)
    if '%' in parts.netloc:
        raise ValueError('base_url host must not be percent-encoded')
    scheme = parts.scheme.lower()
    if scheme not in ('https', 'http'):
        raise ValueError('base_url must use https (or http to a loopback or allowed LAN IP literal)')
    if '@' in parts.netloc:
        raise ValueError('base_url must not carry userinfo')
    if parts.query or parts.fragment or base_url.endswith(('?', '#')):
        raise ValueError('base_url must not carry a query or fragment')
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError('base_url has an invalid port') from exc
    raw = parts.hostname
    if not raw:
        raise ValueError('base_url needs a host')
    raw = raw.rstrip('.')
    ip = _ip_literal(raw)
    if ip is not None:
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            raise ValueError('IPv4-mapped IPv6 literals are refused')
        if (ip.is_unspecified or ip.is_link_local or ip.is_multicast or (ip.is_reserved and not ip.is_loopback)
                or ip in METADATA_ADDRESSES):
            raise ValueError('base_url names a blocked address')
        host = ip.compressed
        display = '[' + host + ']' if ip.version == 6 else host
    else:
        try:
            host = raw.encode('idna').decode('ascii').lower()
        except UnicodeError as exc:
            raise ValueError('base_url host is not a valid domain name') from exc
        labels = host.split('.')
        if not all(labels) or any(len(label) > 63 or not label.replace('-', '').isalnum() or label.startswith('-')
                                  or label.endswith('-') for label in labels) or len(host) > 253:
            raise ValueError('base_url host is not a valid domain name')
        if host.replace('.', '').isdigit() or _NUMERIC_LABEL.fullmatch(labels[-1]):
            # a numeric last label makes the host an IPv4 address to URL parsers and resolvers (0xa9fea9fe,
            # 0251.0376.0251.0376, 169.254.43518): refused rather than treated as a domain name
            raise ValueError('base_url host is a numeric IPv4 spelling; write the dotted-quad IP literal')
        display = host
    if scheme == 'http':
        loopback = ip is not None and ip.is_loopback
        lan = ip is not None and isinstance(ip, ipaddress.IPv4Address) and any(ip in net for net in RFC1918)
        if not (loopback or (lan and allow_http_lan)):
            raise ValueError('http is allowed only to a loopback IP literal, or to an RFC1918 IP literal with '
                             'policy.allow_http_lan')
    if port is None:
        port = 443 if scheme == 'https' else 80
    if not 0 < port < 65536:
        raise ValueError('base_url has an invalid port')
    return f'{scheme}://{display}:{port}'


def _secret_strings(value, path='$'):
    """Paths of secret-shaped strings anywhere in a JSON-like document (keys included)."""
    if isinstance(value, str):
        if looks_secret(value):
            yield path
    elif isinstance(value, dict):
        for k, v in value.items():
            secret_key = isinstance(k, str) and looks_secret(k)
            if secret_key:
                yield path + '.<key>'
            yield from _secret_strings(v, f"{path}.{'<key>' if secret_key else k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _secret_strings(v, f'{path}[{i}]')


def check_no_secrets(document) -> None:
    """BrokerConfigSecretError when any string in document carries a secret shape (the location only, never the value)."""
    found = next(_secret_strings(document), None)
    if found is not None:
        raise BrokerConfigSecretError(f'Broker configuration contains a secret-shaped value at {found}; '
                                      'store it with `titanos broker credential set` and reference the credential id')


class BrokerPolicy(BaseModel):
    model_config = _Strict
    cloud_policy: Literal['off', 'explicit', 'fallback'] = 'off'
    default_location: Literal['local_only', 'prefer_local', 'auto', 'cloud_only'] = 'auto'
    anonymous_cloud: bool = False
    allow_http_lan: bool = False
    max_fallbacks: int = Field(2, ge=0, le=5)
    auto_model: str | None = None


class UpstreamTimeouts(BaseModel):
    model_config = _Strict
    connect: float = Field(10.0, gt=0, le=600)
    first_byte: float = Field(120.0, gt=0, le=3600)
    idle: float = Field(120.0, gt=0, le=3600)


class AzureBinding(BaseModel):
    model_config = _Strict
    resource: str = Field(pattern=r'^[a-z0-9][a-z0-9-]{1,62}$')
    endpoint_host: str = Field(default_factory=lambda: AZURE_ENDPOINT_HOSTS[0])
    api_version: str = Field(pattern=r'^[0-9]{4}-[0-9]{2}-[0-9]{2}(-preview)?$')
    deployments: dict[str, str] = Field(default_factory=dict)

    @field_validator('endpoint_host')
    @classmethod
    def known_endpoint_host(cls, value):
        if value not in AZURE_ENDPOINT_HOSTS:
            raise ValueError('azure.endpoint_host must be one of ' + ', '.join(AZURE_ENDPOINT_HOSTS))
        return value

    @field_validator('deployments')
    @classmethod
    def valid_deployments(cls, value):
        for model, deployment in value.items():
            if not model or len(model) > REF_MAX or not LOCAL_NAME_RE.fullmatch(deployment) or len(deployment) > 64:
                raise ValueError('azure.deployments maps model ids to deployment names')
        return value

    @property
    def base_url(self) -> str:
        return f'https://{self.resource}.{self.endpoint_host}/openai'


class DataPolicy(BaseModel):
    model_config = _Strict
    modalities: list[Literal['text', 'image', 'audio']] | None = None  # None: everything the profile supports
    allow_store: bool = False


class Upstream(BaseModel):
    model_config = _Strict
    profile: str
    base_url: str | None = None  # None: the profile default (or the azure binding)
    credential: str | None = Field(None, pattern=r'^[a-z][a-z0-9_-]{0,63}$')
    # auth scheme for profiles that offer a choice (custom_openai: bearer, azure_api_key or none); None = profile default
    auth: Literal['bearer', 'azure_api_key', 'none'] | None = None
    enabled: bool = False
    models: list[str] = Field(default_factory=list, max_length=4096)
    allow_any_model: bool | None = None  # None: the profile default (router kind: true)
    catalog: Literal['live', 'static'] = 'static'
    surfaces: list[str] | None = None
    anthropic_version: str | None = Field(None, pattern=r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
    anthropic_betas: list[str] = Field(default_factory=list, max_length=16)
    azure: AzureBinding | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeouts: UpstreamTimeouts | None = None
    max_concurrency: int = Field(16, ge=1, le=1024)
    data_policy: DataPolicy = Field(default_factory=DataPolicy)
    label: str | None = Field(None, max_length=128)

    @field_validator('models')
    @classmethod
    def valid_models(cls, value):
        if any(not isinstance(m, str) or not m or len(m) > REF_MAX or m != m.strip() for m in value):
            raise ValueError(f'upstream models are non-empty ids of at most {REF_MAX} characters')
        return value

    @field_validator('anthropic_betas')
    @classmethod
    def valid_betas(cls, value):
        import re
        if any(not re.fullmatch(r'[a-z0-9][a-z0-9.-]{0,63}', beta) for beta in value):
            raise ValueError('anthropic_betas are beta flag names')
        return value

    @model_validator(mode='after')
    def consistent_with_profile(self):
        row = PROFILES.get(self.profile)
        if row is None:
            raise ValueError(f'Unknown profile {self.profile!r}')
        if self.surfaces is not None:
            if self.profile not in OPERATOR_SURFACE_PROFILES:
                raise ValueError(f'surfaces may be set only for {", ".join(sorted(OPERATOR_SURFACE_PROFILES))}')
            if not set(self.surfaces) <= set(row.surfaces):
                raise ValueError('surfaces must be a subset of the profile surfaces: '
                                 + ', '.join(sorted(set(self.surfaces) - set(row.surfaces))))
        if self.auth is not None:
            options = tuple(row.extras.get('auth_options', ()))
            if self.auth not in options:
                raise ValueError(f'auth may be set only for profiles that offer a choice; {self.profile} offers '
                                 + (', '.join(options) or 'none'))
            if self.auth == 'none' and self.credential is not None:
                raise ValueError('auth none takes no credential')
        elif row.auth == 'none' and row.kind != 'slot' and self.credential is not None:
            # a router without inbound auth (switchyard) may forward what it receives: a credential needs auth bearer
            raise ValueError(f'profile {self.profile} sends no credential by default; set auth bearer to bind one')
        if (self.anthropic_version is not None or self.anthropic_betas) and self.profile != 'anthropic':
            raise ValueError('anthropic_version and anthropic_betas apply only to the anthropic profile')
        if self.profile == 'azure_openai':
            if self.azure is None:
                raise ValueError('the azure_openai profile requires an azure binding')
            if self.base_url is not None:
                raise ValueError('azure_openai derives its base_url from the azure binding')
        elif self.azure is not None:
            raise ValueError('azure applies only to the azure_openai profile')
        if row.kind != 'slot' and row.default_base_url is None and self.azure is None and self.base_url is None:
            raise ValueError(f'profile {self.profile} has no default base_url; set base_url')
        for name in self.headers:
            if name not in row.static_headers:
                raise ValueError(f'header {name!r} is not allowed for profile {self.profile}; allowed: '
                                 + (', '.join(row.static_headers) or 'none'))
        if self.data_policy.modalities is not None and not set(self.data_policy.modalities) <= row.data_modalities:
            raise ValueError('data_policy.modalities must be a subset of the profile data modalities: '
                             + ', '.join(sorted(row.data_modalities)))
        return self

    def effective_auth(self) -> str:
        """The auth template the upstream's credential must use: Upstream.auth, else the profile's."""
        return self.auth or PROFILES[self.profile].auth

    def effective_base_url(self) -> str | None:
        if self.azure is not None:
            return self.azure.base_url
        return self.base_url or PROFILES[self.profile].default_base_url

    def origin(self, *, allow_http_lan: bool) -> str | None:
        url = self.effective_base_url()
        return parse_origin(url, allow_http_lan=allow_http_lan) if url else None

    def effective_allow_any_model(self) -> bool:
        return PROFILES[self.profile].allow_any_model_default if self.allow_any_model is None else self.allow_any_model

    def effective_surfaces(self) -> frozenset[str]:
        row = PROFILES[self.profile]
        return frozenset(row.surfaces) if self.surfaces is None else frozenset(self.surfaces)

    def effective_modalities(self) -> frozenset[str]:
        row = PROFILES[self.profile]
        return row.data_modalities if self.data_policy.modalities is None else frozenset(self.data_policy.modalities)


class PublishedModel(BaseModel):
    model_config = _Strict
    target: str = Field(max_length=REF_MAX)
    capabilities: list[Literal['chat', 'tools', 'vision', 'responses', 'embeddings', 'speech', 'transcription', 'images',
                               'moderation', 'rerank']] = Field(default_factory=lambda: ['chat'])
    inject: dict[str, Any] = Field(default_factory=dict)

    @field_validator('target')
    @classmethod
    def qualified_target(cls, value):
        if parse_ref(value).kind != 'qualified':
            raise ValueError('a published alias targets a qualified ref <model>@<upstream>')
        return value


class VirtualTarget(BaseModel):
    model_config = _Strict
    ref: str = Field(min_length=1, max_length=REF_MAX)
    weight: int | None = Field(None, ge=1, le=1000)


class VirtualModel(BaseModel):
    model_config = _Strict
    capability: Literal['chat', 'embeddings', 'speech', 'transcription', 'images', 'moderation', 'rerank']
    strategy: Literal['ordered', 'weighted', 'cheapest', 'lowest_latency', 'advisor'] = 'ordered'
    targets: list[VirtualTarget] = Field(min_length=1, max_length=16)
    fallback_on: list[Literal['queue_full', 'rate_limit_error', 'model_not_loaded', 'worker_unavailable',
                              'model_load_failed', 'device_unavailable', 'upstream_rate_limited', 'upstream_unavailable',
                              'upstream_timeout', 'network', 'breaker_open']] = Field(
        default_factory=lambda: list(FALLBACK_TRIGGERS))
    max_attempts: int | None = Field(None, ge=1, le=6)
    advisor: str | None = None
    expose: bool = True

    @model_validator(mode='after')
    def advisor_strategy(self):
        if (self.strategy == 'advisor') != (self.advisor is not None):
            raise ValueError('strategy advisor and an advisor id go together')
        return self

    def effective_max_attempts(self, max_fallbacks: int | None = None) -> int:
        """max_attempts, else 3 for chat and embeddings virtual models and 1 for every other capability; capped at
        1 + max_fallbacks when given. The router passes min(policy.max_fallbacks, settings.broker_max_fallbacks)."""
        attempts = self.max_attempts
        if attempts is None:
            attempts = 3 if self.capability in MULTI_ATTEMPT_CAPABILITIES else 1
        if max_fallbacks is not None:
            attempts = min(attempts, 1 + max(0, max_fallbacks))
        return attempts


class AdvisorSpec(BaseModel):
    model_config = _Strict
    profile: Literal['generic-v1', 'hivemind-gim-router', 'pending-contract']
    url: str | None = None
    credential: str | None = Field(None, pattern=r'^[a-z][a-z0-9_-]{0,63}$')
    timeout_ms: int = Field(300, ge=1, le=5000)
    share_excerpt: bool = False
    label_map: dict[Literal['simple', 'complex', 'reasoning'], str] = Field(default_factory=dict)
    label: str | None = Field(None, max_length=128)


class VoiceMaps(BaseModel):
    model_config = _Strict
    openai_to_elevenlabs: dict[str, str] = Field(default_factory=dict)
    elevenlabs_to_openai: dict[str, str] = Field(default_factory=dict)


class PriceRow(BaseModel):
    model_config = _Strict
    match: str = Field(min_length=1, max_length=REF_MAX)
    currency: Literal['USD'] = 'USD'
    input_per_mtok: Decimal | None = None
    output_per_mtok: Decimal | None = None
    cached_input_per_mtok: Decimal | None = None
    cache_write_per_mtok: Decimal | None = None  # Anthropic cache writes (input price when absent)
    per_1k_chars: Decimal | None = None
    per_audio_minute: Decimal | None = None
    per_image: Decimal | None = None
    # Anthropic: 1-hour-TTL cache writes (cache_write_per_mtok, then input, when absent), server-side web search requests
    # (unpriced when absent: the reservation stands) and the fast-mode multiplier on token prices (unpriced when absent)
    cache_write_1h_per_mtok: Decimal | None = None
    per_web_search_request: Decimal | None = None
    fast_multiplier: Decimal | None = None
    # OpenAI modality tokens: audio input and output tokens (gpt-4o audio and realtime models, gpt-4o-transcribe,
    # gpt-4o-mini-tts) and image input tokens (gpt-image edits); the text token price applies when absent
    audio_input_per_mtok: Decimal | None = None
    audio_output_per_mtok: Decimal | None = None
    image_input_per_mtok: Decimal | None = None
    source: str = Field('operator', max_length=64)
    effective_from: date | None = None

    @field_validator('input_per_mtok', 'output_per_mtok', 'cached_input_per_mtok', 'cache_write_per_mtok', 'per_1k_chars',
                     'per_audio_minute', 'per_image', 'cache_write_1h_per_mtok', 'per_web_search_request',
                     'fast_multiplier', 'audio_input_per_mtok', 'audio_output_per_mtok', 'image_input_per_mtok')
    @classmethod
    def nonnegative(cls, value):
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError('prices are finite nonnegative decimals')
        return value


class Compat(BaseModel):
    model_config = _Strict
    hivemind_suffix_aliases: dict[str, str] = Field(default_factory=dict)

    @field_validator('hivemind_suffix_aliases')
    @classmethod
    def valid_aliases(cls, value):
        for suffix, upstream in value.items():
            if not suffix or len(suffix) > 64 or not valid_upstream_id(upstream):
                raise ValueError('hivemind_suffix_aliases maps a non-empty suffix to an upstream id')
        return value


class BrokerConfig(BaseModel):
    model_config = _Strict
    version: Literal[1] = 1
    policy: BrokerPolicy = Field(default_factory=BrokerPolicy)
    upstreams: dict[str, Upstream] = Field(default_factory=dict)
    models: dict[str, PublishedModel] = Field(default_factory=dict)
    virtual_models: dict[str, VirtualModel] = Field(default_factory=dict)
    advisors: dict[str, AdvisorSpec] = Field(default_factory=dict)
    voice_maps: VoiceMaps = Field(default_factory=VoiceMaps)
    prices: list[PriceRow] = Field(default_factory=list, max_length=4096)
    compat: Compat = Field(default_factory=Compat)

    @model_validator(mode='before')
    @classmethod
    def refuse_secrets(cls, data):
        check_no_secrets(data)  # constructor path: wrapped in a ValidationError that hides the input
        return data

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Refuse secret-shaped documents with BrokerConfigSecretError before schema validation."""
        check_no_secrets(obj.model_dump(mode='json') if isinstance(obj, BaseModel) else obj)
        return super().model_validate(obj, *args, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data, *args, **kwargs):
        return cls.model_validate(json.loads(json_data), *args, **kwargs)

    @model_validator(mode='after')
    def consistent(self):
        for uid, upstream in self.upstreams.items():
            if not valid_upstream_id(uid):
                raise ValueError(f'invalid upstream id {uid!r}: ^[a-z][a-z0-9_-]{{0,31}}$ (and not "store")')
            if upstream.base_url is not None:
                if PROFILES[upstream.profile].kind == 'slot':
                    raise BrokerError(f'Upstream {uid!r} uses the {upstream.profile} slot, which has no wire contract; '
                                      'it takes no base_url', 'advisor_profile_unconfigured')
                parse_origin(upstream.base_url, allow_http_lan=self.policy.allow_http_lan)
        for kind, names in (('alias', self.models), ('virtual model', self.virtual_models)):
            for name in names:
                if not LOCAL_NAME_RE.fullmatch(name):
                    raise ValueError(f'invalid {kind} name {name!r}')
        both = set(self.models) & set(self.virtual_models)
        if both:
            raise ValueError('names are both a published alias and a virtual model: ' + ', '.join(sorted(both)))
        for name, alias in self.models.items():
            upstream = parse_ref(alias.target).upstream
            if upstream not in self.upstreams:
                raise ValueError(f'alias {name!r} targets unknown upstream {upstream!r}')
        limit = 1 + self.policy.max_fallbacks
        for name, virtual in self.virtual_models.items():
            if virtual.max_attempts is not None and virtual.max_attempts > limit:
                raise ValueError(f'virtual model {name!r}: max_attempts exceeds 1 + policy.max_fallbacks ({limit})')
            if virtual.advisor is not None and virtual.advisor not in self.advisors:
                raise ValueError(f'virtual model {name!r} names unknown advisor {virtual.advisor!r}')
            for target in virtual.targets:
                ref = parse_ref(target.ref)
                if ref.kind == 'qualified' and ref.upstream not in self.upstreams:
                    raise ValueError(f'virtual model {name!r} targets unknown upstream {ref.upstream!r}')
                if ref.kind == 'plain' and target.ref in self.virtual_models:
                    raise ValueError(f'virtual model {name!r} may not target another virtual model')
            if virtual.strategy == 'weighted' and any(t.weight is None for t in virtual.targets):
                raise ValueError(f'virtual model {name!r}: weighted strategy needs a weight on every target')
        if self.policy.auto_model is not None and self.policy.auto_model not in self.virtual_models:
            raise ValueError('policy.auto_model must name a virtual model')
        for aid, advisor in self.advisors.items():
            if not valid_upstream_id(aid):
                raise ValueError(f'invalid advisor id {aid!r}')
            if advisor.profile == 'pending-contract' and advisor.url is not None:
                raise BrokerError('The pending-contract advisor slot has no wire contract; it takes no url',
                                  'advisor_profile_unconfigured')
            if advisor.url is not None:
                parse_origin(advisor.url, allow_http_lan=self.policy.allow_http_lan)
        return self

    def enabled_upstreams(self) -> dict[str, Upstream]:
        return {uid: u for uid, u in self.upstreams.items() if u.enabled}

    def is_disabled(self) -> bool:
        """The disabled lane: no enabled upstream, alias or virtual model."""
        return not self.enabled_upstreams() and not self.models and not self.virtual_models

    def published_names(self) -> set[str]:
        return set(self.models) | set(self.virtual_models)


def check_name_conflicts(cfg: BrokerConfig, local_names) -> None:
    """409 broker_name_conflict when a published alias or virtual name equals a local registry name, or another broker
    name. Called at load and by every PUT, and on registry reload candidates (a refusal leaves the registry unchanged)."""
    local = set(local_names)
    clashes = sorted((set(cfg.models) | set(cfg.virtual_models)) & local)
    both = sorted(set(cfg.models) & set(cfg.virtual_models))
    if clashes or both:
        raise BrokerError('Broker names collide with local registry or other broker names: ' + ', '.join(clashes + both),
                          'broker_name_conflict')


def forbidden_policy_changes(old: BrokerPolicy, new: BrokerPolicy) -> set[str]:
    """KEYLESS_FORBIDDEN_POLICY_FIELDS that differ between two policies (a keyless principal may change none)."""
    return {name for name in KEYLESS_FORBIDDEN_POLICY_FIELDS if getattr(old, name) != getattr(new, name)}


def broker_path(settings) -> Path:
    """settings.broker_config (TITANOS_BROKER_CONFIG) when set, else TITANOS_HOME/broker.json. No I/O."""
    if settings.broker_config:
        return Path(settings.broker_config).expanduser()
    return _home() / 'broker.json'


def broker_dir() -> Path:
    """TITANOS_HOME/broker: keys, vault, usage, decisions, audit, handles and ownership live here."""
    return _home() / 'broker'


def _home() -> Path:
    from ..config import home  # the same resolved TITANOS_HOME spelling as models.json state (no I/O at import)
    return home()


def parse_broker_config(text: str) -> BrokerConfig:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'broker.json is not valid JSON: {exc.msg} at line {exc.lineno}') from None
    check_no_secrets(document)
    return BrokerConfig.model_validate(document)


def read_broker_config(path) -> BrokerConfig:
    """The validated configuration at path; the default (disabled) configuration when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return BrokerConfig()
    return parse_broker_config(path.read_text(encoding='utf-8'))


def broker_lock(path, timeout: float = 30):
    """The shared CLI/server FileLock beside broker.json."""
    from filelock import FileLock
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path) + '.lock', timeout=timeout)


def replace_with_retry(source, destination, *, attempts: int = REPLACE_ATTEMPTS, delay: float = REPLACE_DELAY) -> None:
    """os.replace with a bounded retry on PermissionError (Windows refuses while a reader holds the target open)."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (2 ** attempt))


def _write_unlocked(path: Path, cfg: BrokerConfig) -> None:
    document = cfg.model_dump(mode='json', exclude_none=False)
    check_no_secrets(document)
    BrokerConfig.model_validate(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='broker-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(document, handle, indent=2, sort_keys=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_broker_config(path, cfg: BrokerConfig) -> None:
    """Validate and atomically replace broker.json under the shared lock."""
    path = Path(path).expanduser()
    with broker_lock(path):
        _write_unlocked(path, cfg)


def update_broker_config(path, change: Callable[[BrokerConfig], None]) -> BrokerConfig:
    """Apply a short mutation to the latest broker.json under the lock; an exception leaves the file unchanged. The
    callback mutates a fresh copy; the result is revalidated before the atomic replace."""
    path = Path(path).expanduser()
    with broker_lock(path):
        current = read_broker_config(path)
        working = current.model_copy(deep=True)
        change(working)
        validated = BrokerConfig.model_validate(working.model_dump(mode='json'))
        _write_unlocked(path, validated)
        return validated


__all__ = ['AdvisorSpec', 'AzureBinding', 'BrokerConfig', 'BrokerPolicy', 'CAPABILITIES', 'Compat', 'DataPolicy',
           'FALLBACK_TRIGGERS', 'KEYLESS_FORBIDDEN_POLICY_FIELDS', 'PriceRow', 'PublishedModel', 'RFC1918', 'Upstream',
           'UpstreamTimeouts', 'VIRTUAL_CAPABILITIES', 'VirtualModel', 'VirtualTarget', 'VoiceMaps', 'broker_dir',
           'broker_lock', 'broker_path', 'check_name_conflicts', 'check_no_secrets', 'forbidden_policy_changes',
           'parse_broker_config', 'parse_origin', 'read_broker_config', 'replace_with_retry', 'update_broker_config',
           'write_broker_config']
