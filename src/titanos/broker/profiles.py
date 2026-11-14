"""Data-only upstream profile table. A new OpenAI-compatible provider is one row.

This is the only module outside tests and fixtures where provider default hostnames appear (a source scan enforces it).
Profiles are never read from ambient environment variables (ANTHROPIC_BASE_URL, OPENAI_BASE_URL ...): an upstream's
base_url comes only from broker.json or the row's default_base_url.

Profile.surfaces maps the NATIVE surface names an upstream speaks (names from broker.surfaces) to path templates relative
to the upstream base_url; translation to other inbound protocols goes through broker.translate. ParamMap keys are fnmatch
globs over the upstream model id; refuse entries of the form 'field>N' refuse only values greater than N.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from .surfaces import SURFACES

Adapter = Literal['openai_compat', 'anthropic', 'gemini', 'elevenlabs', 'ollama_native', 'none']
ProfileKind = Literal['cloud', 'router', 'slot']
Auth = Literal['bearer', 'anthropic', 'xi', 'azure_api_key', 'goog', 'none']

ADAPTERS: tuple[str, ...] = ('openai_compat', 'anthropic', 'gemini', 'elevenlabs', 'ollama_native', 'none')
PROFILE_KINDS: tuple[str, ...] = ('cloud', 'router', 'slot')
# auth kind -> (header name, value template); the value is filled only by vault.CredentialHandle.apply in egress
AUTH_TEMPLATES: Mapping[str, tuple[str, str] | None] = MappingProxyType({
    'bearer': ('Authorization', 'Bearer {secret}'),
    'anthropic': ('x-api-key', '{secret}'),
    'xi': ('xi-api-key', '{secret}'),
    'azure_api_key': ('api-key', '{secret}'),
    'goog': ('x-goog-api-key', '{secret}'),
    'none': None,
})
DATA_MODALITIES = frozenset({'text', 'image', 'audio'})

# Always stripped before egress (Titanos request knobs, never provider fields): placement (worker, location, timeout)
# and the local-engine knobs Studio and local clients send (require_loaded, include_provenance, the VAD gate fields),
# which an OpenAI-strict provider would answer with 400 unknown parameter.
TITANOS_ONLY_FIELDS = frozenset({'worker', 'location', 'timeout', 'require_loaded', 'include_provenance',
                                 'vad_enabled', 'vad_model', 'vad_options'})
# The only inbound request headers forwarded upstream on passthrough (never authorization, x-api-key, xi-api-key,
# x-goog-api-key, cookie or x-titanos-*).
INBOUND_HEADER_ALLOW = frozenset({'anthropic-beta', 'openai-beta', 'accept'})

_EMPTY: Mapping = MappingProxyType({})


def _frozen_map(value: Mapping | None) -> Mapping:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class ParamMap:
    rename: Mapping[str, Mapping[str, str]] = field(default_factory=lambda: _EMPTY)  # 3.11: mappingproxy is unhashable, so not a plain default
    drop: Mapping[str, frozenset[str]] = field(default_factory=lambda: _EMPTY)  # 3.11: mappingproxy is unhashable, so not a plain default
    allow: frozenset[str] | None = None  # None: the provider's own native schema passes through on P paths
    refuse: Mapping[str, frozenset[str]] = field(default_factory=lambda: _EMPTY)  # 3.11: mappingproxy is unhashable, so not a plain default
    inject: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: _EMPTY)  # 3.11: mappingproxy is unhashable, so not a plain default
    tool_id_rule: str | None = None  # 'alnum9': tool-call ids remapped to 9 alphanumerics (Mistral)


TOOL_ID_RULES = frozenset({'alnum9'})

# -- R32 native passthrough -------------------------------------------------------------------------------------------
# A provider endpoint that no typed surface covers is reachable through the guarded native passthrough (broker/native.py):
# GET|POST|... /native/<upstream_id>/<path relative to the profile's native root>, or the provider's own public path (a
# "bare" path) when the row allows it and exactly one enabled upstream the caller may see offers it (X-Titanos-Upstream
# chooses otherwise). Only the (method, path) rows below are forwarded; everything else is refused before egress.
NATIVE_METHODS = frozenset({'GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'WS'})
NATIVE_OWNERS = frozenset({'none', 'create', 'bound', 'list'})
# Never forwarded on the native lane either (credentials, cookies, framing and Titanos hints are rebuilt or dropped).
NATIVE_HEADER_NEVER = frozenset({'authorization', 'x-api-key', 'xi-api-key', 'x-goog-api-key', 'api-key', 'cookie',
                                 'proxy-authorization', 'host', 'content-length', 'transfer-encoding', 'connection',
                                 'upgrade', 'te', 'trailer', 'keep-alive', 'accept-encoding', 'x-titanos-hop',
                                 'x-hivemind-peer-hop', 'sec-websocket-key', 'sec-websocket-version',
                                 'sec-websocket-extensions', 'sec-websocket-accept', 'sec-websocket-protocol', 'origin'})


@dataclass(frozen=True)
class NativeRoute:
    """One allowlisted provider endpoint.

    method: an HTTP method or 'WS' (a WebSocket endpoint, relayed frame by frame). path: a template relative to the native
    root ('{x}' one segment, '{x:path}' greedy, a literal suffix such as '{model}:predict' allowed).
    store: the call creates or reads provider-side state, so the caller needs cloud:store and the upstream
    data_policy.allow_store (R1). owner: 'create' binds the id of the created object (JSON id or name) to the calling key;
    'bound' answers only for the key that created the object named by path parameter `param` (admin and the keyless loopback
    operator may use any id); 'list' lists the whole provider account behind the shared credential, so a key needs
    broker:admin; 'none' carries no object id. admin: an account-level operation (fine-tuning, ephemeral tokens): a key
    needs broker:admin. bare: the provider's own public path (bare_prefix + path) also reaches this row; bare_over_listed lets
    that bare path win over a Titanos catalog route or a listed route of the same shape (never over an inference surface).
    vendor_over_listed: the bare path wins over a listed route of the same shape only when the request carries the
    profile's own SDK key header and no Authorization header (the ElevenLabs SDK's GET /v1/models with xi-api-key gets the
    provider's model list; an OpenAI client keeps Titanos' catalog). model_over_listed: the bare path wins over a listed
    route of the same shape when the request body's model names a cloud model of that upstream (OpenAI POST /v1/videos
    with sora-2, sora-2@<upstream> or an alias of it, decided by native.choose_by_model; a local model, or any other name,
    keeps Titanos' local video job). family: a documentation label."""
    method: str
    path: str
    store: bool = False
    owner: str = 'none'
    param: str | None = None
    admin: bool = False
    bare: bool = True
    bare_over_listed: bool = False
    family: str = ''
    vendor_over_listed: bool = False
    model_over_listed: bool = False


@dataclass(frozen=True)
class NativeSpec:
    """The native passthrough of one profile row. root 'base': paths are relative to the upstream base_url (OpenAI
    family: https://api.openai.com/v1 + /files); 'origin': relative to its scheme://host:port (Anthropic, Gemini with its
    /upload/v1beta and /v1beta trees, ElevenLabs with /v1 and /v2). bare_prefix: the inbound public path prefix that maps to
    the root ('/v1' for the OpenAI family, '' for origin roots), None for no bare paths. request_headers: inbound headers
    forwarded besides content-type, accept and INBOUND_HEADER_ALLOW (never NATIVE_HEADER_NEVER). response_headers: upstream
    response headers returned besides the adapters' allowlist. rewrite_headers: response headers whose absolute URL on the
    upstream origin is rewritten to this server's /native/<upstream_id>/ path (Gemini resumable upload URLs).
    strip_frame_fields: top-level JSON fields removed from client WebSocket text frames (credentials some SDKs send in
    the first frame). ws_subprotocols: client-offered WebSocket subprotocols forwarded upstream (anything else, such as
    a subprotocol carrying a key, is dropped). rewrite_body_fields: JSON answer fields (top level, and in each item of a
    top-level 'data' list) whose absolute URL on the upstream origin is rewritten to this server's /native/<upstream_id>/
    path, so an SDK that follows it never sends the caller's Titanos key to the provider (Anthropic's batch results_url)."""
    root: Literal['base', 'origin'] = 'base'
    bare_prefix: str | None = None
    routes: tuple[NativeRoute, ...] = ()
    request_headers: frozenset[str] = frozenset()
    response_headers: frozenset[str] = frozenset()
    rewrite_headers: frozenset[str] = frozenset()
    strip_frame_fields: frozenset[str] = frozenset()
    ws_subprotocols: frozenset[str] = frozenset()
    rewrite_body_fields: frozenset[str] = frozenset()


def _n(method: str, path: str, **kw) -> tuple[NativeRoute, ...]:
    """Rows for one path: method may be 'GET|POST|DELETE'."""
    return tuple(NativeRoute(m, path, **kw) for m in method.split('|'))


def _rows(*groups) -> tuple[NativeRoute, ...]:
    return tuple(row for group in groups for row in group)


@dataclass(frozen=True)
class Profile:
    id: str
    adapter: Adapter
    kind: ProfileKind
    default_base_url: str | None
    auth: Auth
    auth_optional: bool
    surfaces: Mapping[str, str]
    params: ParamMap = ParamMap()
    catalog_path: str | None = None
    probe_path: str | None = None
    allow_any_model_default: bool = False
    static_headers: tuple[str, ...] = ()
    ratelimit_headers: Mapping[str, str] = field(default_factory=lambda: _EMPTY)  # 3.11: mappingproxy is unhashable, so not a plain default
    request_id_headers: tuple[str, ...] = ()
    usage_headers: tuple[str, ...] = ()
    include_usage: bool = False
    hop_header: bool = False
    data_modalities: frozenset[str] = frozenset()
    image_formats: frozenset[str] = frozenset()
    # upstream.surfaces may narrow the row (custom_openai, nim_custom, litellm, hivemind_gateway, pair_facade)
    operator_surfaces: bool = False
    # wire features the row declares (json_schema, tools, vision, reasoning, thinking, prompt_caching)
    features: frozenset[str] = frozenset()
    # row-specific non-secret data (default anthropic_version, asymmetric embedding globs, reasoning-model globs ...)
    extras: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)  # 3.11: mappingproxy is unhashable, so not a plain default
    # R32: the guarded native passthrough (None: no native endpoints beyond the typed surfaces)
    native: NativeSpec | None = None
    # R30: well-known credential environment variable NAMES (never values) offered by env discovery and quickstart
    env_names: tuple[str, ...] = ()


_OPENAI_RATELIMIT = _frozen_map({'requests_remaining': 'x-ratelimit-remaining-requests',
                                 'tokens_remaining': 'x-ratelimit-remaining-tokens',
                                 'requests_reset': 'x-ratelimit-reset-requests',
                                 'tokens_reset': 'x-ratelimit-reset-tokens'})
_ANTHROPIC_RATELIMIT = _frozen_map({'requests_remaining': 'anthropic-ratelimit-requests-remaining',
                                    'tokens_remaining': 'anthropic-ratelimit-tokens-remaining',
                                    'input_tokens_remaining': 'anthropic-ratelimit-input-tokens-remaining',
                                    'output_tokens_remaining': 'anthropic-ratelimit-output-tokens-remaining'})

_OPENAI_PATHS = {
    'openai.chat': '/chat/completions',
    'openai.completions': '/completions',
    'openai.responses': '/responses',
    'openai.responses.retrieve': '/responses/{response_id}',
    'openai.responses.delete': '/responses/{response_id}',
    'openai.responses.cancel': '/responses/{response_id}/cancel',
    'openai.responses.input_items': '/responses/{response_id}/input_items',
    'openai.embeddings': '/embeddings',
    'openai.audio.speech': '/audio/speech',
    'openai.audio.transcriptions': '/audio/transcriptions',
    'openai.audio.translations': '/audio/translations',
    'openai.images.generations': '/images/generations',
    'openai.images.edits': '/images/edits',
    'openai.moderations': '/moderations',
}


# OpenAI-compatible providers also speak the de-facto rerank shape (api.openai.com has no rerank endpoint).
_COMPAT_PATHS = {**_OPENAI_PATHS, 'openai.rerank': '/rerank'}


def _paths(*names: str, base: Mapping[str, str] = _COMPAT_PATHS) -> Mapping[str, str]:
    return _frozen_map({name: base[name] for name in names})


def _openai_compat(id, base_url, surfaces, *, kind='cloud', auth='bearer', auth_optional=False, params=ParamMap(),
                   catalog=True, allow_any=None, static_headers=(), include_usage=True, hop=False,
                   data=frozenset({'text', 'image'}), image_formats=frozenset(), operator=False, features=frozenset(),
                   extras=None, native=None, env=()):
    return Profile(id=id, adapter='openai_compat', kind=kind, default_base_url=base_url, auth=auth,
                   auth_optional=auth_optional, surfaces=surfaces, params=params,
                   catalog_path='/models' if catalog else None, probe_path='/models' if catalog else None,
                   allow_any_model_default=(kind == 'router') if allow_any is None else allow_any,
                   static_headers=tuple(static_headers), ratelimit_headers=_OPENAI_RATELIMIT,
                   request_id_headers=('x-request-id',), include_usage=include_usage, hop_header=hop,
                   data_modalities=frozenset(data), image_formats=frozenset(image_formats), operator_surfaces=operator,
                   features=frozenset(features), extras=_frozen_map(extras), native=native, env_names=tuple(env))


_SAMPLING_DROP = frozenset({'temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 'logprobs', 'top_logprobs'})
# Local-engine sampling knobs (llama.cpp and Ollama spellings) that Studio and local clients send: providers that accept
# them get them mapped (repeat_penalty -> repetition_penalty on Together, Fireworks and OpenRouter, which also take top_k
# and min_p); strict OpenAI-shaped providers would answer 400 unknown parameter, so they are dropped with a note there
# (X-Titanos-Dropped names them).
_LOCAL_SAMPLING = frozenset({'repeat_penalty', 'top_k', 'min_p'})
_REPEAT_PENALTY_RENAME = _frozen_map({'*': {'repeat_penalty': 'repetition_penalty'}})

# Native passthrough allowlists (R32). Seeded from the 2026-10-02 provider audits; provider owners extend these rows.
# OpenAI: everything outside the typed surfaces that a stock SDK reaches (files, uploads, batches, vector stores,
# conversations, containers, stored chat completions, Responses compact and input_tokens, videos, fine-tuning, model
# deletion) and the Realtime WebSocket. Retired families (images/variations, assistants, threads) and organization/admin
# APIs are deliberately absent, so they are refused before egress.
_OPENAI_NATIVE = NativeSpec(
    root='base', bare_prefix='/v1',
    request_headers=frozenset({'idempotency-key', 'openai-beta', 'openai-organization', 'openai-project'}),
    response_headers=frozenset({'content-disposition', 'openai-version', 'openai-organization'}),
    ws_subprotocols=frozenset({'realtime'}),
    routes=_rows(
        _n('POST', '/files', store=True, owner='create', family='files'),
        _n('GET', '/files', owner='list', family='files'),
        _n('GET|DELETE', '/files/{file_id}', store=True, owner='bound', family='files'),
        _n('GET', '/files/{file_id}/content', store=True, owner='bound', family='files'),
        _n('POST', '/uploads', store=True, owner='create', family='uploads'),
        _n('POST', '/uploads/{upload_id}/parts', store=True, owner='bound', family='uploads'),
        _n('POST', '/uploads/{upload_id}/complete', store=True, owner='bound', family='uploads'),
        _n('POST', '/uploads/{upload_id}/cancel', store=True, owner='bound', family='uploads'),
        _n('POST', '/batches', store=True, owner='create', family='batches'),
        _n('GET', '/batches', owner='list', family='batches'),
        _n('GET', '/batches/{batch_id}', store=True, owner='bound', family='batches'),
        _n('POST', '/batches/{batch_id}/cancel', store=True, owner='bound', family='batches'),
        _n('POST', '/vector_stores', store=True, owner='create', family='vector_stores'),
        _n('GET', '/vector_stores', owner='list', family='vector_stores'),
        _n('GET|POST|DELETE', '/vector_stores/{vector_store_id}', store=True, owner='bound', family='vector_stores'),
        _n('POST', '/vector_stores/{vector_store_id}/search', store=True, owner='bound', family='vector_stores'),
        _n('GET|POST', '/vector_stores/{vector_store_id}/files', store=True, owner='bound', family='vector_stores'),
        _n('GET|POST|DELETE', '/vector_stores/{vector_store_id}/files/{file_id}', store=True, owner='bound',
           param='vector_store_id', family='vector_stores'),
        _n('GET', '/vector_stores/{vector_store_id}/files/{file_id}/content', store=True, owner='bound',
           param='vector_store_id', family='vector_stores'),
        _n('POST', '/vector_stores/{vector_store_id}/file_batches', store=True, owner='bound', family='vector_stores'),
        _n('GET', '/vector_stores/{vector_store_id}/file_batches/{batch_id}', store=True, owner='bound',
           param='vector_store_id', family='vector_stores'),
        _n('POST', '/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel', store=True, owner='bound',
           param='vector_store_id', family='vector_stores'),
        _n('GET', '/vector_stores/{vector_store_id}/file_batches/{batch_id}/files', store=True, owner='bound',
           param='vector_store_id', family='vector_stores'),
        _n('POST', '/conversations', store=True, owner='create', family='conversations'),
        _n('GET|POST|DELETE', '/conversations/{conversation_id}', store=True, owner='bound', family='conversations'),
        _n('GET|POST', '/conversations/{conversation_id}/items', store=True, owner='bound', family='conversations'),
        _n('GET|DELETE', '/conversations/{conversation_id}/items/{item_id}', store=True, owner='bound',
           param='conversation_id', family='conversations'),
        _n('POST', '/containers', store=True, owner='create', family='containers'),
        _n('GET', '/containers', owner='list', family='containers'),
        _n('GET|DELETE', '/containers/{container_id}', store=True, owner='bound', family='containers'),
        _n('GET|POST', '/containers/{container_id}/files', store=True, owner='bound', family='containers'),
        _n('GET|DELETE', '/containers/{container_id}/files/{file_id}', store=True, owner='bound',
           param='container_id', family='containers'),
        _n('GET', '/containers/{container_id}/files/{file_id}/content', store=True, owner='bound',
           param='container_id', family='containers'),
        # stored chat completions are created through the typed chat surface (store=true), so they carry no owner binding:
        # the whole account's list, read, update and delete need broker:admin
        _n('GET', '/chat/completions', store=True, owner='list', family='chat_completions_stored'),
        _n('GET|POST|DELETE', '/chat/completions/{completion_id}', store=True, owner='list',
           family='chat_completions_stored'),
        _n('GET', '/chat/completions/{completion_id}/messages', store=True, owner='list',
           family='chat_completions_stored'),
        _n('POST', '/responses/compact', family='responses'),
        _n('POST', '/responses/input_tokens', family='responses'),
        # Titanos' local video jobs serve the same bare paths: POST is decided by the body's model (model_over_listed);
        # retrieve, content and delete reach OpenAI for a video id created through Titanos by the caller (owner-bound)
        _n('POST', '/videos', store=True, owner='create', model_over_listed=True, family='videos'),
        _n('GET', '/videos', owner='list', family='videos'),
        _n('GET|DELETE', '/videos/{video_id}', store=True, owner='bound', family='videos'),
        _n('GET', '/videos/{video_id}/content', store=True, owner='bound', family='videos'),
        _n('POST', '/videos/{video_id}/remix', store=True, owner='bound', family='videos'),
        _n('GET|POST', '/fine_tuning/{rest:path}', store=True, admin=True, family='fine_tuning'),
        _n('DELETE', '/models/{model}', admin=True, family='models'),
        _n('WS', '/realtime', family='realtime'),
        # ephemeral client secrets let a browser talk to the provider directly, past metering and budgets: admin only
        _n('POST', '/realtime/client_secrets', admin=True, family='realtime'),
        # WebRTC and SIP calls (the SDP offer/answer is relayed, the media then flows between the browser or SIP trunk and
        # the provider, past metering) and the legacy session-token mints: admin only, like client_secrets
        _n('POST', '/realtime/calls', admin=True, family='realtime'),
        _n('POST', '/realtime/calls/{call_id}/accept', admin=True, family='realtime'),
        _n('POST', '/realtime/calls/{call_id}/hangup', admin=True, family='realtime'),
        _n('POST', '/realtime/calls/{call_id}/refer', admin=True, family='realtime'),
        _n('POST', '/realtime/calls/{call_id}/reject', admin=True, family='realtime'),
        _n('POST', '/realtime/sessions', admin=True, family='realtime'),
        _n('POST', '/realtime/transcription_sessions', admin=True, family='realtime'),
        # custom voices (eligible accounts): a consent recording, then the voice; both are provider-side state
        _n('POST', '/audio/voices', store=True, owner='create', family='custom_voices'),
        _n('POST', '/audio/voice_consents', store=True, owner='create', family='custom_voices'),
        _n('GET', '/audio/voice_consents', owner='list', family='custom_voices'),
        _n('GET|POST|DELETE', '/audio/voice_consents/{consent_id}', store=True, owner='bound', family='custom_voices'),
        _n('DELETE', '/fine_tuning/checkpoints/{checkpoint}/permissions/{permission_id}', store=True, admin=True,
           family='fine_tuning'),
        # evals and ChatKit (sessions mint client secrets; threads are account-wide): account-level, admin only
        _n('GET|POST', '/evals', store=True, admin=True, family='evals'),
        _n('GET|POST|DELETE', '/evals/{rest:path}', store=True, admin=True, family='evals'),
        _n('POST', '/chatkit/sessions', admin=True, family='chatkit'),
        _n('POST', '/chatkit/sessions/{session_id}/cancel', admin=True, family='chatkit'),
        _n('GET', '/chatkit/threads', store=True, admin=True, family='chatkit'),
        _n('GET|DELETE', '/chatkit/threads/{thread_id}', store=True, admin=True, family='chatkit'),
        _n('GET', '/chatkit/threads/{thread_id}/items', store=True, admin=True, family='chatkit'),
    ))

# Anthropic: Message Batches, Files, Skills and the Anthropic-shaped model list (origin root, so the SDK's /v1/... paths are
# kept); x-api-key and anthropic-version are set by the broker, anthropic-beta is forwarded. A batch's results_url is
# rewritten to this server (the SDK follows it with the caller's key). Managed Agents (agents, sessions with their SSE
# event streams, environments, vaults, deployments, memory stores) run and bill at the provider outside Titanos metering,
# and the Admin API (organizations/...) needs an admin credential: both are broker:admin only, the Admin API under
# /native/<upstream>/ only.
_ANTHROPIC_MANAGED = ('agents', 'sessions', 'environments', 'vaults', 'deployments', 'deployment_runs', 'memory_stores')
_ANTHROPIC_NATIVE = NativeSpec(
    root='origin', bare_prefix='',
    response_headers=frozenset({'content-disposition'}),
    rewrite_body_fields=frozenset({'results_url'}),
    routes=_rows(
        _n('POST', '/v1/messages/batches', store=True, owner='create', family='batches'),
        _n('GET', '/v1/messages/batches', owner='list', family='batches'),
        _n('GET|DELETE', '/v1/messages/batches/{batch_id}', store=True, owner='bound', family='batches'),
        _n('GET', '/v1/messages/batches/{batch_id}/results', store=True, owner='bound', family='batches'),
        _n('POST', '/v1/messages/batches/{batch_id}/cancel', store=True, owner='bound', family='batches'),
        _n('POST', '/v1/files', store=True, owner='create', family='files'),
        _n('GET', '/v1/files', owner='list', family='files'),
        # GET /v1/files/{name} is also Titanos' artifact route: the Anthropic SDK (x-api-key or anthropic-version, no
        # Authorization) reaches the provider's file metadata
        _n('GET', '/v1/files/{file_id}', store=True, owner='bound', family='files', vendor_over_listed=True),
        _n('DELETE', '/v1/files/{file_id}', store=True, owner='bound', family='files'),
        _n('GET', '/v1/files/{file_id}/content', store=True, owner='bound', family='files'),
        _n('POST', '/v1/skills', store=True, owner='create', family='skills'),
        _n('GET', '/v1/skills', owner='list', family='skills'),
        _n('GET|DELETE', '/v1/skills/{skill_id}', store=True, owner='bound', family='skills'),
        _n('GET|POST', '/v1/skills/{skill_id}/versions', store=True, owner='bound', family='skills'),
        _n('GET|DELETE', '/v1/skills/{skill_id}/versions/{version}', store=True, owner='bound', param='skill_id',
           family='skills'),
        # the bare /v1/models paths are Titanos' own catalog; /native/<upstream>/v1/models reaches Anthropic's
        _n('GET', '/v1/models', bare=False, family='models'),
        _n('GET', '/v1/models/{model_id}', bare=False, family='models'),
        *(_n(method, path, store=True, admin=True, family='managed_agents')
          for fam in _ANTHROPIC_MANAGED
          for method, path in (('GET|POST', f'/v1/{fam}'), ('GET|POST|DELETE', f'/v1/{fam}/{{object_id}}'),
                               ('GET|POST|DELETE', f'/v1/{fam}/{{object_id}}/{{rest:path}}'))),
        _n('GET|POST|DELETE', '/v1/organizations/{rest:path}', admin=True, bare=False, family='admin_api'),
    ))

# Gemini: Files (resumable upload under /upload/v1beta, upload URLs rewritten to this server), cached contents, batches,
# long-running operations, the model actions without a typed surface (Imagen predict, Veo predictLongRunning, batch
# generation) and the Live API WebSocket. x-goog-api-key is set by the broker; ?key= is refused.
_GEMINI_UPLOAD_HEADERS = frozenset({'x-goog-upload-protocol', 'x-goog-upload-command', 'x-goog-upload-offset',
                                    'x-goog-upload-header-content-length', 'x-goog-upload-header-content-type',
                                    'x-goog-upload-file-name'})
_GEMINI_NATIVE = NativeSpec(
    root='origin', bare_prefix='',
    request_headers=_GEMINI_UPLOAD_HEADERS,
    response_headers=frozenset({'x-goog-upload-url', 'x-goog-upload-status', 'x-goog-upload-size-received',
                                'x-goog-upload-chunk-granularity', 'x-goog-upload-control-url', 'content-disposition'}),
    rewrite_headers=frozenset({'x-goog-upload-url', 'x-goog-upload-control-url'}),
    routes=_rows(
        _n('POST', '/upload/v1beta/files', store=True, owner='create', family='files'),
        _n('POST', '/v1beta/files', store=True, owner='create', family='files'),
        _n('GET', '/v1beta/files', owner='list', family='files'),
        _n('GET|DELETE', '/v1beta/files/{file_id}', store=True, owner='bound', family='files'),
        _n('POST', '/v1beta/files:register', store=True, owner='create', family='files'),
        _n('POST', '/v1beta/cachedContents', store=True, owner='create', family='cached_contents'),
        _n('GET', '/v1beta/cachedContents', owner='list', family='cached_contents'),
        _n('GET|PATCH|DELETE', '/v1beta/cachedContents/{cache_id}', store=True, owner='bound',
           family='cached_contents'),
        _n('GET', '/v1beta/batches', owner='list', family='batches'),
        _n('GET|POST|PATCH|DELETE', '/v1beta/batches/{batch_id}', store=True, owner='bound', family='batches'),
        _n('POST', '/v1beta/models/{model}:predict', bare_over_listed=True, family='imagen'),
        _n('POST', '/v1beta/models/{model}:predictLongRunning', store=True, owner='create', bare_over_listed=True,
           family='veo'),
        _n('GET', '/v1beta/models/{model}/operations/{operation}', store=True, owner='bound', param='operation',
           bare_over_listed=True, family='operations'),
        _n('GET', '/v1beta/operations/{operation}', store=True, owner='bound', family='operations'),
        _n('POST', '/v1beta/models/{model}:batchGenerateContent', store=True, owner='create', bare_over_listed=True,
           family='batches'),
        _n('POST', '/v1beta/models/{model}:asyncBatchEmbedContent', store=True, owner='create',
           bare_over_listed=True, family='batches'),
        _n('WS', '/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent', family='live'),
        # -- Gemini owner additions (2026-10-02 completeness round; paths from the public API reference) --
        # file media download (batch results files, generated files): /download/v1beta/files/{id}:download?alt=media
        _n('GET', '/download/v1beta/files/{file_id}', store=True, owner='bound', family='files'),
        # the model catalog as Google publishes it (the bare /v1beta/models paths are Titanos' own catalog)
        _n('GET', '/v1beta/models', bare=False, family='models'),
        _n('GET', '/v1beta/models/{model}', bare=False, family='models'),
        # File Search stores (RAG): stores, their documents, long-running import / upload operations; the
        # :importFile and :uploadToFileSearchStore actions are POSTs on the store name
        _n('POST', '/v1beta/fileSearchStores', store=True, owner='create', family='file_search'),
        _n('GET', '/v1beta/fileSearchStores', owner='list', family='file_search'),
        _n('GET|POST|DELETE', '/v1beta/fileSearchStores/{store_id}', store=True, owner='bound', family='file_search'),
        _n('POST', '/upload/v1beta/fileSearchStores/{store_id}', store=True, owner='bound', family='file_search'),
        _n('GET', '/v1beta/fileSearchStores/{store_id}/operations/{operation}', store=True, owner='bound',
           param='store_id', family='file_search'),
        _n('GET', '/v1beta/fileSearchStores/{store_id}/upload/operations/{operation}', store=True, owner='bound',
           param='store_id', family='file_search'),
        _n('GET', '/v1beta/fileSearchStores/{store_id}/documents', store=True, owner='bound', family='file_search'),
        _n('GET|DELETE', '/v1beta/fileSearchStores/{store_id}/documents/{document_id}', store=True, owner='bound',
           param='store_id', family='file_search'),
        # Interactions API (model and agent interactions; stored at the provider by default, SSE when stream is true)
        _n('POST', '/v1beta/interactions', store=True, owner='create', family='interactions'),
        _n('GET|DELETE', '/v1beta/interactions/{interaction_id}', store=True, owner='bound', family='interactions'),
        _n('POST', '/v1beta/interactions/{interaction_id}/cancel', store=True, owner='bound', family='interactions'),
        # Agents, environments and custom voices: provider-side objects bound to the creating key
        _n('POST', '/v1beta/agents', store=True, owner='create', family='agents'),
        _n('GET', '/v1beta/agents', owner='list', family='agents'),
        _n('GET|DELETE', '/v1beta/agents/{agent_id}', store=True, owner='bound', family='agents'),
        _n('POST', '/v1beta/environments', store=True, owner='create', family='environments'),
        _n('GET', '/v1beta/environments', owner='list', family='environments'),
        _n('GET|DELETE', '/v1beta/environments/{environment_id}', store=True, owner='bound', family='environments'),
        _n('GET', '/v1beta/environments/{environment_id}/files/{file_path:path}', store=True, owner='bound',
           param='environment_id', family='environments'),
        _n('PUT', '/upload/v1beta/environments/{environment_id}/files/{file_path:path}', store=True, owner='bound',
           param='environment_id', family='environments'),
        _n('POST', '/v1beta/voices', store=True, owner='create', family='voices'),
        _n('GET', '/v1beta/voices', owner='list', family='voices'),
        _n('GET|DELETE', '/v1beta/voices/{voice_id}', store=True, owner='bound', family='voices'),
        # project-wide configuration: webhooks (event delivery and signing secrets), triggers (scheduled agent runs that
        # spend outside Titanos metering), server-managed credentials and ephemeral auth tokens (a browser talks to
        # the provider directly, past metering and budgets): broker:admin only
        _n('GET|POST', '/v1beta/webhooks', store=True, admin=True, family='webhooks'),
        _n('GET|POST|PATCH|DELETE', '/v1beta/webhooks/{webhook_id}', store=True, admin=True, family='webhooks'),
        _n('GET|POST', '/v1beta/triggers', store=True, admin=True, family='triggers'),
        _n('GET|PATCH', '/v1beta/triggers/{trigger_id}', store=True, admin=True, family='triggers'),
        _n('GET|POST', '/v1beta/triggers/{trigger_id}/executions', store=True, admin=True, family='triggers'),
        _n('GET|POST', '/v1beta/credentials', store=True, admin=True, family='credentials'),
        _n('GET|PATCH|DELETE', '/v1beta/credentials/{credential_id}', store=True, admin=True, family='credentials'),
        _n('POST', '/v1alpha/auth_tokens', admin=True, family='live'),
        # Live API v1alpha (affective dialog, proactive audio) and Lyria RealTime music
        _n('WS', '/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent', family='live'),
        _n('WS', '/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateMusic', family='live_music'),
    ))

# ElevenLabs: generation endpoints without a typed surface (timestamps, dialogue, sound effects, voice changer,
# isolation, forced alignment, music), voice search and settings, account reads (admin), history and dubbing (store and
# owner bound), and the stream-input WebSockets. xi-api-key is set by the broker; one sent inside a frame is removed.
_ELEVENLABS_NATIVE = NativeSpec(
    root='origin', bare_prefix='',
    response_headers=frozenset({'history-item-id', 'x-character-count', 'current-concurrent-requests',
                                'maximum-concurrent-requests', 'content-disposition', 'song-id', 'song_id',
                                'x-trace-id'}),
    strip_frame_fields=frozenset({'xi_api_key', 'xi-api-key', 'authorization'}),
    routes=_rows(
        _n('POST', '/v1/text-to-speech/{voice_id}/with-timestamps', family='tts'),
        _n('POST', '/v1/text-to-speech/{voice_id}/stream/with-timestamps', family='tts'),
        _n('POST', '/v1/text-to-dialogue', family='dialogue'),
        _n('POST', '/v1/text-to-dialogue/stream', family='dialogue'),
        _n('POST', '/v1/text-to-dialogue/with-timestamps', family='dialogue'),
        _n('POST', '/v1/text-to-dialogue/stream/with-timestamps', family='dialogue'),
        _n('POST', '/v1/sound-generation', family='sound_effects'),
        _n('POST', '/v1/speech-to-speech/{voice_id}', family='voice_changer'),
        _n('POST', '/v1/speech-to-speech/{voice_id}/stream', family='voice_changer'),
        _n('POST', '/v1/audio-isolation', family='audio_isolation'),
        _n('POST', '/v1/audio-isolation/stream', family='audio_isolation'),
        _n('GET', '/v1/audio-isolation/history', owner='list', family='audio_isolation'),
        _n('GET|DELETE', '/v1/audio-isolation/history/{history_item_id}', store=True, owner='list',
           family='audio_isolation'),
        _n('POST', '/v1/forced-alignment', family='forced_alignment'),
        _n('POST', '/v1/music', family='music'),
        _n('POST', '/v1/music/stream', family='music'),
        _n('POST', '/v1/music/plan', family='music'),
        _n('POST', '/v1/music/detailed', family='music'),
        _n('POST', '/v1/music/detailed/stream', family='music'),
        _n('POST', '/v1/music/stem-separation', family='music'),
        _n('POST', '/v1/music/video-to-music', family='music'),
        _n('POST', '/v1/music/upload', store=True, owner='create', family='music'),
        _n('GET|POST', '/v1/music/finetunes', store=True, admin=True, family='music'),
        _n('GET|PATCH|DELETE', '/v1/music/finetunes/{finetune_id}', store=True, admin=True, family='music'),
        _n('GET', '/v2/voices', family='voices'),
        _n('GET', '/v1/voices/{voice_id}/settings', family='voices'),
        _n('GET', '/v1/voices/settings/default', family='voices'),
        # the bare path is the typed voice lookup's shape: /native/<upstream>/... or X-Titanos-Upstream reaches it
        _n('GET', '/v1/voices/accents', family='voices'),
        _n('POST', '/v1/similar-voices', family='voices'),
        _n('GET', '/v1/voices/{voice_id}/samples/{sample_id}/audio', family='voices'),
        _n('DELETE', '/v1/voices/{voice_id}/samples/{sample_id}', store=True, owner='bound', param='voice_id',
           family='voices'),
        # professional voice cloning: the created voice is bound to the calling key, every follow-up call to its owner
        _n('POST', '/v1/voices/pvc', store=True, owner='create', family='voices_pvc'),
        _n('POST', '/v1/voices/pvc/{voice_id}', store=True, owner='bound', family='voices_pvc'),
        _n('GET|POST|DELETE', '/v1/voices/pvc/{voice_id}/{rest:path}', store=True, owner='bound', param='voice_id',
           family='voices_pvc'),
        _n('POST', '/v1/voices/{voice_id}/settings/edit', store=True, admin=True, family='voices'),
        _n('POST', '/v1/voices/add', store=True, owner='create', family='voices'),
        _n('POST', '/v1/voices/{voice_id}/edit', store=True, owner='bound', family='voices'),
        _n('DELETE', '/v1/voices/{voice_id}', store=True, owner='bound', family='voices'),
        _n('POST', '/v1/text-to-voice/design', family='voice_design'),
        _n('POST', '/v1/text-to-voice/create-previews', family='voice_design'),
        _n('POST', '/v1/text-to-voice', store=True, owner='create', family='voice_design'),
        _n('POST', '/v1/text-to-voice/{voice_id}/remix', family='voice_design'),
        _n('GET', '/v1/text-to-voice/{generated_voice_id}/stream', family='voice_design'),
        _n('POST', '/v1/voices/add/{public_user_id}/{voice_id}', store=True, owner='create', family='voices'),
        _n('GET', '/v1/shared-voices', family='voices'),
        # the bare GET /v1/models is Titanos' catalog; the ElevenLabs SDK (xi-api-key, no Authorization) gets the
        # provider's own model list there, everyone else through /native/<upstream>/v1/models
        _n('GET', '/v1/models', vendor_over_listed=True, family='models'),
        _n('GET', '/v1/user', admin=True, family='account'),
        _n('GET', '/v1/user/subscription', admin=True, family='account'),
        _n('GET', '/v1/usage/character-stats', admin=True, family='account'),
        # single-use tokens let a client reach the provider directly, past metering and budgets: admin only
        _n('POST', '/v1/single-use-token/{token_type}', admin=True, family='tokens'),
        _n('GET', '/v1/history', owner='list', family='history'),
        _n('POST', '/v1/history/download', store=True, owner='list', family='history'),
        _n('GET|DELETE', '/v1/history/{history_item_id}', store=True, owner='list', family='history'),
        _n('GET', '/v1/history/{history_item_id}/audio', store=True, owner='list', family='history'),
        _n('GET|DELETE', '/v1/speech-to-text/transcripts/{transcription_id}', store=True, owner='list',
           family='transcripts'),
        _n('POST', '/v1/dubbing', store=True, owner='create', family='dubbing'),
        _n('GET', '/v1/dubbing', owner='list', family='dubbing'),
        _n('GET|DELETE', '/v1/dubbing/{dubbing_id}', store=True, owner='bound', family='dubbing'),
        _n('GET', '/v1/dubbing/{dubbing_id}/audio/{language_code}', store=True, owner='bound', param='dubbing_id',
           family='dubbing'),
        _n('GET', '/v1/dubbing/{dubbing_id}/transcript/{language_code}', store=True, owner='bound',
           param='dubbing_id', family='dubbing'),
        _n('GET', '/v1/dubbing/{dubbing_id}/transcripts/{language_code}/format/{format_type}', store=True,
           owner='bound', param='dubbing_id', family='dubbing'),
        _n('GET', '/v1/dubbing/resource/{dubbing_id}', store=True, owner='bound', family='dubbing'),
        _n('GET|POST|PATCH|DELETE', '/v1/dubbing/resource/{dubbing_id}/{rest:path}', store=True, owner='bound',
           param='dubbing_id', family='dubbing'),
        _n('GET', '/v1/pronunciation-dictionaries', owner='list', family='pronunciation'),
        _n('POST', '/v1/pronunciation-dictionaries/add-from-file', store=True, owner='create', family='pronunciation'),
        _n('POST', '/v1/pronunciation-dictionaries/add-from-rules', store=True, owner='create',
           family='pronunciation'),
        _n('GET', '/v1/pronunciation-dictionaries/{pronunciation_dictionary_id}', store=True, owner='bound',
           family='pronunciation'),
        _n('POST', '/v1/pronunciation-dictionaries/{pronunciation_dictionary_id}/add-rules', store=True,
           owner='bound', family='pronunciation'),
        _n('POST', '/v1/pronunciation-dictionaries/{pronunciation_dictionary_id}/remove-rules', store=True,
           owner='bound', family='pronunciation'),
        _n('POST', '/v1/pronunciation-dictionaries/{pronunciation_dictionary_id}/set-rules', store=True,
           owner='bound', family='pronunciation'),
        _n('PATCH', '/v1/pronunciation-dictionaries/{pronunciation_dictionary_id}', store=True, owner='bound',
           family='pronunciation'),
        _n('GET', '/v1/pronunciation-dictionaries/{dictionary_id}/{version_id}/download', store=True, owner='bound',
           param='dictionary_id', family='pronunciation'),
        _n('WS', '/v1/text-to-speech/{voice_id}/stream-input', family='tts_websocket'),
        _n('WS', '/v1/text-to-speech/{voice_id}/multi-stream-input', family='tts_websocket'),
        _n('WS', '/v1/speech-to-text/realtime', family='stt_websocket'),
    ))

# OpenAI-compatible providers with documented endpoints outside the typed surfaces (paths relative to the base_url unless
# the root is 'origin'). Families whose public path is also an OpenAI path (files, batches, conversations) are bare=False:
# an OpenAI SDK's bare POST /v1/files must keep reaching the single OpenAI upstream when Mistral or Together is also
# configured, so these are reached through /native/<upstream>/... only.
def _files(create: str = '/files', *, bare: bool = False) -> tuple[NativeRoute, ...]:
    """The OpenAI-shaped Files family (upload, list, read, delete, content) relative to the native root."""
    return _rows(
        _n('POST', create, store=True, owner='create', bare=bare, family='files'),
        _n('GET', '/files', owner='list', bare=bare, family='files'),
        _n('GET|DELETE', '/files/{file_id}', store=True, owner='bound', bare=bare, family='files'),
        _n('GET', '/files/{file_id}/content', store=True, owner='bound', bare=bare, family='files'))


def _batches(prefix: str = '/batches', *, bare: bool = False) -> tuple[NativeRoute, ...]:
    """The OpenAI-shaped Batch family (create, list, read, cancel)."""
    return _rows(
        _n('POST', prefix, store=True, owner='create', bare=bare, family='batches'),
        _n('GET', prefix, owner='list', bare=bare, family='batches'),
        _n('GET', prefix + '/{batch_id}', store=True, owner='bound', bare=bare, family='batches'),
        _n('POST', prefix + '/{batch_id}/cancel', store=True, owner='bound', bare=bare, family='batches'))


_MISTRAL_NATIVE = NativeSpec(root='base', bare_prefix='/v1', routes=_rows(
    _n('POST', '/fim/completions', family='fim'),
    _n('POST', '/ocr', family='ocr'),
    _n('POST', '/chat/moderations', family='moderation'),
    _n('POST', '/moderations', family='moderation'),
    _n('POST', '/classifications', family='classifiers'),
    _n('POST', '/chat/classifications', family='classifiers'),
    _n('POST', '/agents/completions', family='agents'),
    # Mistral speech answers JSON {audio_data: base64} (or SSE) and names the voice voice_id: not the OpenAI speech
    # shape, so it is native-only (the typed openai.audio.speech surface is not on the row)
    _n('POST', '/audio/speech', bare=False, family='audio'),
    _n('GET', '/audio/voices', family='audio'),
    _files(), _n('GET', '/files/{file_id}/url', store=True, owner='bound', bare=False, family='files'),
    _batches('/batch/jobs', bare=True),
    _n('POST', '/conversations', store=True, owner='create', bare=False, family='conversations'),
    _n('GET', '/conversations', owner='list', bare=False, family='conversations'),
    _n('GET|POST|DELETE', '/conversations/{conversation_id}', store=True, owner='bound', bare=False,
       family='conversations'),
    _n('GET', '/conversations/{conversation_id}/history', store=True, owner='bound', bare=False, family='conversations'),
    _n('GET', '/conversations/{conversation_id}/messages', store=True, owner='bound', bare=False,
       family='conversations'),
    _n('POST', '/conversations/{conversation_id}/restart', store=True, owner='bound', bare=False,
       family='conversations'),
    _n('POST', '/agents', store=True, owner='create', bare=False, family='agents'),
    _n('GET', '/agents', owner='list', bare=False, family='agents'),
    _n('GET|PATCH|DELETE', '/agents/{agent_id}', store=True, owner='bound', bare=False, family='agents'),
    _n('GET|POST', '/fine_tuning/{rest:path}', store=True, admin=True, bare=False, family='fine_tuning'),
    _n('GET|POST', '/libraries', store=True, admin=True, bare=False, family='libraries'),
    _n('GET|POST|PUT|DELETE', '/libraries/{rest:path}', store=True, admin=True, bare=False, family='libraries'),
))
_TOGETHER_NATIVE = NativeSpec(root='base', bare_prefix='/v1', routes=_rows(
    _n('POST', '/videos', store=True, owner='create', bare=False, family='videos'),
    _n('GET', '/videos/{video_id}', store=True, owner='bound', bare=False, family='videos'),
    _files('/files/upload'), _batches(),
    _n('GET|POST', '/fine-tunes', store=True, admin=True, bare=False, family='fine_tuning'),
    _n('GET|POST|DELETE', '/fine-tunes/{rest:path}', store=True, admin=True, bare=False, family='fine_tuning'),
    _n('GET|POST', '/endpoints', admin=True, bare=False, family='endpoints'),
    _n('GET|PATCH|DELETE', '/endpoints/{endpoint_id}', admin=True, bare=False, family='endpoints'),
    _n('GET', '/hardware', bare=False, family='endpoints'),
))
_PERPLEXITY_NATIVE = NativeSpec(root='base', bare_prefix=None, routes=_rows(
    _n('POST', '/search', family='search'),
    _n('POST', '/async/chat/completions', store=True, owner='create', family='async'),
    _n('GET', '/async/chat/completions', owner='list', family='async'),
    _n('GET', '/async/chat/completions/{request_id}', store=True, owner='bound', family='async'),
))
_OPENROUTER_NATIVE = NativeSpec(root='base', bare_prefix=None, routes=_rows(
    _n('GET', '/generation', family='stats'),
    _n('GET', '/credits', admin=True, family='account'),
    _n('GET', '/key', admin=True, family='account'),
    _n('GET', '/providers', family='catalog'),
    _n('GET', '/models/{author}/{slug}/endpoints', family='catalog'),
))
# Fireworks: the inference tree sits under /inference/v1 and the control plane under /v1/accounts, so the native root is
# the origin. Image generation uses the workflow paths (text_to_image, Kontext and get_result) with async polling.
_FIREWORKS_NATIVE = NativeSpec(root='origin', bare_prefix=None, routes=_rows(
    _n('POST', '/inference/v1/workflows/{model:path}', family='images'),
    _n('GET|POST|PATCH|DELETE', '/v1/accounts/{rest:path}', store=True, admin=True, family='account'),
))
_DEEPSEEK_NATIVE = NativeSpec(root='origin', bare_prefix=None, routes=_rows(
    _n('POST', '/beta/completions', family='fim'),
    _n('POST', '/beta/chat/completions', family='prefix_completion'),
    _n('GET', '/user/balance', admin=True, family='account'),
))
_XAI_NATIVE = NativeSpec(root='base', bare_prefix=None, routes=_rows(
    # xAI image editing takes JSON (image as url or data URI, up to five references), not the OpenAI multipart form
    _n('POST', '/images/edits', family='images'),
    _n('POST', '/videos/generations', store=True, owner='create', family='videos'),
    _n('GET', '/videos/{request_id}', store=True, owner='bound', family='videos'),
    _n('POST', '/responses/compact', family='responses'),
    _n('POST', '/tokenize-text', family='tokenize'),
    _n('GET', '/language-models', family='catalog'),
    _n('GET', '/language-models/{model_id}', family='catalog'),
    _n('GET', '/image-generation-models', family='catalog'),
    _n('GET', '/image-generation-models/{model_id}', family='catalog'),
    _n('GET', '/api-key', admin=True, family='account'),
))
_GROQ_NATIVE = NativeSpec(root='base', bare_prefix=None, routes=_rows(_files(), _batches()))
# ai.api.nvidia.com: the NVIDIA API catalog's second origin (NeMo Retriever reranking, the genai image models and the
# visual NIMs). A credential is bound to one origin, so this is its own profile row with its own upstream.
_NVIDIA_AI_NATIVE = NativeSpec(root='base', bare_prefix=None, routes=_rows(
    _n('POST', '/retrieval/{org}/{model}/reranking', family='rerank'),
    _n('POST', '/retrieval/{org}/{model}/embeddings', family='embeddings'),
    _n('POST', '/genai/{model:path}', family='genai'),
    _n('POST', '/cv/{model:path}', family='cv'),
    _n('POST', '/vlm/{model:path}', family='vlm'),
))
# Azure OpenAI v1 (https://<resource>.openai.azure.com/openai/v1): the OpenAI families it serves beyond the typed surfaces.
_AZURE_V1_NATIVE = NativeSpec(
    root='base', bare_prefix=None,
    request_headers=frozenset({'idempotency-key', 'openai-beta'}),
    response_headers=frozenset({'content-disposition', 'apim-request-id'}),
    routes=tuple(route for route in _OPENAI_NATIVE.routes
                 if route.family in ('files', 'batches', 'vector_stores', 'videos', 'fine_tuning', 'responses')))
# Azure OpenAI classic data plane (https://<resource>.<endpoint host>/openai, api-version added by egress unless the
# client names one, for example a preview version): files, batches, vector stores, fine-tuning and Responses (preview)
# share OpenAI's paths; the Realtime WebSocket takes ?deployment=<name>. /native/<upstream>/... only (no bare paths, so an
# OpenAI SDK's bare /v1/files never lands on an Azure resource by accident).
_AZURE_NATIVE = NativeSpec(
    root='base', bare_prefix=None,
    request_headers=frozenset({'idempotency-key', 'openai-beta'}),
    response_headers=frozenset({'content-disposition', 'apim-request-id'}),
    ws_subprotocols=frozenset({'realtime'}),
    routes=tuple(route for route in _OPENAI_NATIVE.routes
                 if route.family in ('files', 'batches', 'vector_stores', 'fine_tuning', 'responses')
                 or (route.family == 'realtime' and route.method == 'WS')))
_OLLAMA_NATIVE = NativeSpec(root='origin', bare_prefix=None, routes=_rows(
    _n('POST', '/api/show', family='catalog'),
    _n('POST', '/api/web_search', family='tools'),
    _n('POST', '/api/web_fetch', family='tools'),
))

# NVIDIA NeMo Switchyard (github.com/NVIDIA-NeMo/Switchyard, Apache-2.0): a routing proxy (switchyard-server, default port
# 4000) that serves OpenAI Chat Completions, OpenAI Responses and Anthropic Messages on one origin, lists its route ids on
# GET /v1/models and has no inbound authentication. Its routing keys session affinity (classifier triggers, escalation,
# sub-agent routing) on the harness session headers below; they carry no credential and are forwarded on its row only.
SWITCHYARD_SESSION_HEADERS = frozenset({
    'x-switchyard-session-id', 'x-switchyard-agent-id', 'x-switchyard-parent-agent-id', 'x-switchyard-is-subagent',
    'x-switchyard-agent-kind', 'x-switchyard-agent-role', 'x-switchyard-task-id', 'x-switchyard-task-kind',
    'x-switchyard-turn-id', 'x-switchyard-request-id', 'x-switchyard-session-final', 'x-switchyard-origin',
    'x-nemo-relay-session-id', 'x-nemo-relay-subagent-id', 'x-dynamo-session-id', 'x-dynamo-parent-session-id',
    'x-dynamo-session-final', 'x-codex-turn-metadata', 'x-codex-parent-thread-id', 'x-openai-subagent',
    'x-claude-code-session-id', 'x-claude-code-agent-id', 'x-claude-code-parent-agent-id', 'x-session-id', 'session-id',
    'thread-id', 'x-task-id', 'x-client-request-id', 'user-agent'})
# The endpoints Switchyard serves beyond the typed surfaces (paths relative to its origin; /native/<upstream>/... only,
# because a bare /v1/responses/compact or /v1/decision would be ambiguous next to an OpenAI upstream). Its operator
# surfaces (stats, stats reset, routing session stats, Prometheus metrics) need broker:admin; /v1/decision runs the
# route's judge calls inside Switchyard and answers which target it would pick.
_SWITCHYARD_NATIVE = NativeSpec(
    root='origin', bare_prefix=None, request_headers=SWITCHYARD_SESSION_HEADERS,
    response_headers=frozenset({'x-model-router-selected-model'}),
    routes=_rows(
        _n('POST', '/v1/responses/compact', family='responses'),
        _n('POST', '/v1/responses/input_tokens', family='responses'),
        _n('POST', '/v1/decision', family='decision'),
        _n('GET', '/v1/models', family='catalog'),
        _n('GET', '/v1/stats', admin=True, family='operator'),
        _n('POST', '/v1/stats/reset', admin=True, family='operator'),
        _n('GET', '/v1/routing/session-stats', admin=True, family='operator'),
        _n('GET', '/metrics', admin=True, family='operator'),
        _n('GET', '/health', family='operator'),
    ))

# OpenAI model families -> what they serve ((glob, capabilities), first match wins; names from the public model list).
# Informational: the catalog reports it per model (titanos.model_capabilities); routing still uses the upstream's
# capabilities, so a new model family is never refused because this table does not know it yet.
_OPENAI_MODEL_CAPABILITIES = (
    ('gpt-image-*', ('images', 'image_edits')), ('chatgpt-image-*', ('images', 'image_edits')),
    ('dall-e-*', ('images',)), ('sora-*', ('video',)),
    ('*-tts*', ('speech',)), ('tts-*', ('speech',)),
    ('*-transcribe*', ('transcription',)), ('whisper-*', ('transcription', 'translation')),
    ('text-embedding-*', ('embeddings',)), ('*moderation*', ('moderation',)),
    ('*realtime*', ('realtime',)), ('gpt-audio*', ('chat', 'audio')), ('*-audio-preview*', ('chat', 'audio')),
    ('*-search-*', ('chat',)), ('computer-use-*', ('responses',)), ('codex-*', ('responses',)),
    ('*-deep-research*', ('responses',)), ('gpt-3.5-turbo-instruct*', ('completions',)),
    ('davinci-*', ('completions',)), ('babbage-*', ('completions',)),
    ('gpt-*', ('chat', 'responses', 'tools', 'vision')), ('chatgpt-*', ('chat', 'responses', 'vision')),
    ('o[0-9]*', ('chat', 'responses', 'tools', 'reasoning')),
)


def model_capabilities(row: 'Profile', upstream_model: str | None) -> tuple[str, ...] | None:
    """The row's per-model capability hint (extras model_capabilities, first matching glob), or None."""
    from fnmatch import fnmatchcase
    for glob, caps in row.extras.get('model_capabilities') or ():
        if fnmatchcase(upstream_model or '', glob):
            return tuple(caps)
    return None


_ROWS: tuple[Profile, ...] = (
    _openai_compat(
        'openai', 'https://api.openai.com/v1', _frozen_map(_OPENAI_PATHS),
        static_headers=('OpenAI-Organization', 'OpenAI-Project'), data={'text', 'image', 'audio'},
        image_formats={'png', 'jpeg', 'webp'},
        params=ParamMap(rename=_frozen_map({'o*': {'max_tokens': 'max_completion_tokens'},
                                            'gpt-5*': {'max_tokens': 'max_completion_tokens'}}),
                        drop=_frozen_map({'o*': frozenset({'temperature', 'top_p'}),
                                          'gpt-5*': frozenset({'temperature', 'top_p'}),
                                          '*': _LOCAL_SAMPLING})),
        features={'json_schema', 'tools', 'vision', 'reasoning'},
        # image_b64_only_globs '*': every OpenAI image model answers base64 and refuses response_format. gpt-image-* always
        # did; the live lane observed on 2026-10-02 that dall-e-2 now answers 400 unknown_parameter ("Unknown parameter:
        # 'response_format'.") as well. Azure OpenAI keeps its own row: dall-e deployments may still accept the field, while
        # gpt-image-* deployments refuse it there too (image_b64_only_globs 'gpt-image-*' on the azure_openai row).
        # sampling_kept_globs / sampling_kept_effort lift the reasoning-model temperature/top_p drop: gpt-5-chat* models
        # take sampling parameters, and GPT-5.1 and later take them with reasoning effort 'none' (Azure OpenAI deployments
        # of the same models follow this row through PARAMS_FROM). default_models: the provider's documented default for a
        # model-optional surface (moderations without model use omni-moderation-latest). model_capabilities: what each
        # OpenAI model family serves, reported per catalog entry (titanos.model_capabilities) for clients such as Studio.
        # bare_model_ids: a stock OpenAI SDK sends gpt-4o-mini, never a qualified ref; when no local model, alias or
        # virtual model has that name, it resolves on the single visible OpenAI upstream that lists it.
        # ratelimit_per_model: a 429 cools down that model only (OpenAI limits are per model), never the whole key.
        extras={'reasoning_globs': ('o*', 'gpt-5*'), 'image_b64_only_globs': ('*',),
                'bare_model_ids': True, 'ratelimit_per_model': True,
                'sampling_kept_globs': ('gpt-5-chat*',), 'sampling_kept_effort': 'none',
                'default_models': _frozen_map({'moderation': 'omni-moderation-latest'}),
                'model_capabilities': _OPENAI_MODEL_CAPABILITIES},
        native=_OPENAI_NATIVE, env=('OPENAI_API_KEY',)),
    Profile(
        id='azure_openai', adapter='openai_compat', kind='cloud', default_base_url=None, auth='azure_api_key',
        auth_optional=False,
        surfaces=_frozen_map({name: '/deployments/{deployment}' + _OPENAI_PATHS[name]
                              for name in ('openai.chat', 'openai.completions', 'openai.embeddings', 'openai.audio.speech',
                                           'openai.audio.transcriptions', 'openai.audio.translations',
                                           'openai.images.generations', 'openai.images.edits')}),
        catalog_path=None, probe_path=None, ratelimit_headers=_OPENAI_RATELIMIT,
        request_id_headers=('x-request-id', 'apim-request-id'), include_usage=True,
        data_modalities=frozenset({'text', 'image', 'audio'}), image_formats=frozenset({'png'}),
        features=frozenset({'json_schema', 'tools', 'vision'}),
        extras=_frozen_map({'url_template': 'https://{resource}.{endpoint_host}/openai',
                            'endpoint_hosts': ('openai.azure.com', 'cognitiveservices.azure.com'),
                            'api_version_query': 'api-version',
                            # gpt-image-* deployments answer base64 and refuse response_format (as on api.openai.com);
                            # dall-e deployments keep the field. Matched on the upstream model the deployment serves.
                            'image_b64_only_globs': ('gpt-image-*',),
                            # deployments are keyed by the OpenAI model id they serve, so the OpenAI hints apply
                            'model_capabilities': _OPENAI_MODEL_CAPABILITIES}),
        native=_AZURE_NATIVE, env_names=('AZURE_OPENAI_API_KEY',)),
    # Azure OpenAI v1 API (https://<resource>.openai.azure.com/openai/v1, Microsoft's recommended surface): the body's
    # model is the deployment name, no api-version, native Responses with stored responses. The operator sets base_url
    # and lists the deployment names in upstream.models (the v1 /models listing names base models, not deployments).
    _openai_compat(
        'azure_openai_v1', None,
        _paths('openai.chat', 'openai.completions', 'openai.responses', 'openai.responses.retrieve',
               'openai.responses.delete', 'openai.responses.cancel', 'openai.responses.input_items', 'openai.embeddings',
               'openai.audio.speech', 'openai.audio.transcriptions', 'openai.audio.translations',
               'openai.images.generations', 'openai.images.edits', base=_OPENAI_PATHS),
        auth='azure_api_key', catalog=False, operator=True, data={'text', 'image', 'audio'}, image_formats={'png'},
        features={'json_schema', 'tools', 'vision', 'reasoning'},
        extras={'image_b64_only_globs': ('gpt-image-*',), 'reasoning_effort_globs': ('o*', 'gpt-5*'),
                'stores_responses': True},
        native=_AZURE_V1_NATIVE, env=('AZURE_OPENAI_API_KEY',)),
    # Provider extras read by the OpenAI-compatible adapter and dispatch (all optional):
    # - reasoning_globs / non_reasoning_globs / reasoning_drop: on a reasoning model (matches a reasoning glob and no
    #   non-reasoning glob) the reasoning_drop fields are dropped with a note (xAI reasoning models answer an error for
    #   presence_penalty, frequency_penalty and stop);
    # - reasoning_effort_globs: the models that accept reasoning_effort; a reasoning_effort Titanos adds (advisor
    #   reasoning mode, a translated Anthropic/Gemini/Responses reasoning request) is sent only to them (a client's own
    #   reasoning_effort on a passthrough call is always forwarded);
    # - image_url_native: the provider answers its own image URLs, so a url (or default) request is passed through
    #   instead of being re-hosted; image_response_format_values maps OpenAI response_format values to the provider's;
    #   image_png_request: body fields that make the provider answer PNG (asset jobs need PNG);
    # - rerank_style 'nim': the provider speaks the NIM ranking shape (query.text, passages[].text -> rankings[index,
    #   logit]); rerank_model_path: the model id is part of the path ({org}/{name}, dots as underscores);
    # - reasoning_content_roundtrip: assistant reasoning_content is sent back on translated calls (DeepSeek thinking mode
    #   answers 400 in a tool loop without it);
    # - stores_responses: the provider stores Responses by default, so store is forced false unless the caller may store.
    _openai_compat(
        'xai', 'https://api.x.ai/v1',
        _paths('openai.chat', 'openai.responses', 'openai.responses.retrieve', 'openai.responses.delete',
               'openai.images.generations'),
        image_formats={'jpeg'}, features={'tools', 'vision', 'reasoning', 'json_schema'},
        params=ParamMap(drop=_frozen_map({'*': _LOCAL_SAMPLING})),
        extras={'image_url_native': True, 'stores_responses': True,
                'reasoning_globs': ('grok-3-mini*', 'grok-4*', 'grok-code*'), 'non_reasoning_globs': ('*non-reasoning*',),
                'reasoning_drop': ('presence_penalty', 'frequency_penalty', 'stop'),
                'reasoning_effort_globs': ('grok-3-mini*', 'grok-4.5*', 'grok-4.6*', 'grok-4.7*', 'grok-4.2*multi-agent*')},
        native=_XAI_NATIVE, env=('XAI_API_KEY',)),
    _openai_compat(
        'groq', 'https://api.groq.com/openai/v1',
        _paths('openai.chat', 'openai.responses', 'openai.audio.speech', 'openai.audio.transcriptions',
               'openai.audio.translations'),
        data={'text', 'image', 'audio'},
        params=ParamMap(refuse=_frozen_map({'*': frozenset({'logprobs', 'top_logprobs', 'logit_bias', 'n>1'})}),
                        drop=_frozen_map({'*': _LOCAL_SAMPLING})),
        features={'tools', 'json_schema', 'reasoning'},
        extras={'reasoning_effort_globs': ('openai/gpt-oss*', 'qwen/qwen3*')}, native=_GROQ_NATIVE,
        env=('GROQ_API_KEY',)),
    _openai_compat(
        'together', 'https://api.together.xyz/v1',
        _paths('openai.chat', 'openai.completions', 'openai.embeddings', 'openai.images.generations', 'openai.rerank',
               'openai.audio.speech', 'openai.audio.transcriptions', 'openai.audio.translations'),
        data={'text', 'image', 'audio'}, image_formats={'png', 'jpeg'},
        params=ParamMap(rename=_REPEAT_PENALTY_RENAME), features={'tools', 'json_schema', 'reasoning'},
        extras={'catalog_bare_array': True, 'image_url_native': True,
                'image_response_format_values': {'b64_json': 'base64'}, 'image_png_request': {'output_format': 'png'},
                'reasoning_effort_globs': ('openai/gpt-oss*',)},
        native=_TOGETHER_NATIVE, env=('TOGETHER_API_KEY',)),
    _openai_compat(
        'mistral', 'https://api.mistral.ai/v1', _paths('openai.chat', 'openai.embeddings', 'openai.audio.transcriptions'),
        include_usage=False, data={'text', 'image', 'audio'},
        params=ParamMap(rename=_frozen_map({'*': {'seed': 'random_seed'}}),
                        drop=_frozen_map({'*': frozenset({'logit_bias', 'logprobs', 'top_logprobs'}) | _LOCAL_SAMPLING}),
                        tool_id_rule='alnum9'),
        features={'tools', 'json_schema', 'vision'}, native=_MISTRAL_NATIVE, env=('MISTRAL_API_KEY',)),
    _openai_compat(
        'deepseek', 'https://api.deepseek.com',
        _frozen_map({'openai.chat': '/chat/completions'}), data={'text'},
        params=ParamMap(drop=_frozen_map({'deepseek-reasoner': _SAMPLING_DROP, '*': _LOCAL_SAMPLING})),
        features={'tools', 'reasoning'},
        extras={'reasoning_content_roundtrip': True}, native=_DEEPSEEK_NATIVE, env=('DEEPSEEK_API_KEY',)),
    _openai_compat(
        'fireworks', 'https://api.fireworks.ai/inference/v1',
        _paths('openai.chat', 'openai.completions', 'openai.embeddings', 'openai.responses', 'openai.responses.retrieve',
               'openai.responses.delete', 'openai.rerank'),
        params=ParamMap(rename=_REPEAT_PENALTY_RENAME), features={'tools', 'json_schema', 'vision', 'reasoning'},
        extras={'stores_responses': True, 'reasoning_effort_globs': ('*gpt-oss*', '*qwen3*')},
        native=_FIREWORKS_NATIVE, env=('FIREWORKS_API_KEY',)),
    # Fireworks serves speech-to-text on its own origin (OpenAI-shaped multipart); one credential binds to one origin, so
    # it is its own row (the operator lists the models, e.g. whisper-v3).
    _openai_compat(
        'fireworks_audio', 'https://audio-prod.api.fireworks.ai/v1',
        _paths('openai.audio.transcriptions', 'openai.audio.translations'), catalog=False, data={'audio'},
        env=('FIREWORKS_API_KEY',)),
    _openai_compat(
        'perplexity', 'https://api.perplexity.ai', _frozen_map({'openai.chat': '/chat/completions'}), catalog=False,
        include_usage=False, data={'text', 'image'},
        params=ParamMap(drop=_frozen_map({'*': frozenset({'repeat_penalty', 'min_p'})})),
        features={'json_schema', 'reasoning'},
        extras={'reasoning_effort_globs': ('sonar-deep-research*', 'sonar-reasoning*')},
        native=_PERPLEXITY_NATIVE, env=('PERPLEXITY_API_KEY',)),
    _openai_compat(
        'nvidia_api_catalog', 'https://integrate.api.nvidia.com/v1', _paths('openai.chat', 'openai.embeddings'),
        features={'tools', 'vision', 'json_schema', 'reasoning'},
        extras={'asymmetric_embedding_globs': ('nvidia/nv-embedqa-*', 'nvidia/llama-3.2-nv-embedqa-*',
                                               'nvidia/llama-3.2-nemoretriever-*', 'nvidia/llama-nemotron-embed-*',
                                               'nvidia/nv-embedcode-*',
                                               'snowflake/arctic-embed-*'),
                'input_type_values': ('query', 'passage'), 'reasoning_effort_globs': ('openai/gpt-oss*',),
                # NVIDIA cloud functions name a still-running invocation (202 Accepted) with these headers
                'response_headers': ('nvcf-reqid', 'nvcf-status')},
        env=('NGC_API_KEY', 'NVIDIA_API_KEY')),
    # The NVIDIA API catalog's second origin (ai.api.nvidia.com): NeMo Retriever reranking as the typed rerank surface
    # (NIM ranking shape, model id in the path) plus the genai, visual and retrieval NIMs through the native passthrough.
    _openai_compat(
        'nvidia_ai_endpoints', 'https://ai.api.nvidia.com/v1',
        _frozen_map({'openai.rerank': '/retrieval/{org}/{name}/reranking'}), catalog=False, data={'text', 'image'},
        extras={'rerank_style': 'nim', 'rerank_model_path': True}, native=_NVIDIA_AI_NATIVE,
        env=('NGC_API_KEY', 'NVIDIA_API_KEY')),
    _openai_compat(
        'nim_custom', None,
        _frozen_map({**_paths('openai.chat', 'openai.completions', 'openai.embeddings'), 'openai.rerank': '/ranking'}),
        auth_optional=True, hop=True, operator=True, features={'tools'}, extras={'rerank_style': 'nim'}),
    _openai_compat('cerebras', 'https://api.cerebras.ai/v1', _paths('openai.chat', 'openai.completions'), data={'text'},
                   params=ParamMap(drop=_frozen_map({'*': _LOCAL_SAMPLING})),
                   features={'tools', 'json_schema', 'reasoning'},
                   extras={'reasoning_effort_globs': ('gpt-oss*', 'qwen-3*', 'zai-glm*')}, env=('CEREBRAS_API_KEY',)),
    _openai_compat('sambanova', 'https://api.sambanova.ai/v1', _paths('openai.chat'), data={'text', 'image'},
                   features={'tools', 'reasoning'}, extras={'reasoning_effort_globs': ('gpt-oss*',)},
                   env=('SAMBANOVA_API_KEY',)),
    _openai_compat(
        'openrouter', 'https://openrouter.ai/api/v1',
        _paths('openai.chat', 'openai.completions', 'openai.embeddings', 'openai.responses', 'openai.rerank'),
        kind='router', static_headers=('HTTP-Referer', 'X-Title'), params=ParamMap(rename=_REPEAT_PENALTY_RENAME),
        features={'tools', 'vision', 'json_schema', 'reasoning'},
        extras={'served_by': 'router:openrouter'}, native=_OPENROUTER_NATIVE, env=('OPENROUTER_API_KEY',)),
    _openai_compat(
        'hf_router', 'https://router.huggingface.co/v1', _paths('openai.chat', 'openai.responses'), kind='router',
        include_usage=False, features={'tools', 'vision', 'json_schema'}, env=('HF_TOKEN',)),
    _openai_compat(
        'litellm', None,
        _paths('openai.chat', 'openai.completions', 'openai.embeddings', 'openai.audio.speech',
               'openai.audio.transcriptions', 'openai.audio.translations', 'openai.images.generations',
               'openai.images.edits', 'openai.moderations', 'openai.responses', 'openai.responses.retrieve',
               'openai.responses.delete', 'openai.rerank'),
        kind='router', auth_optional=True, hop=True, operator=True, data={'text', 'image', 'audio'},
        image_formats={'png'}, extras={'stores_responses': True}),
    _openai_compat(
        'hivemind_gateway', None,
        _paths('openai.chat', 'openai.completions', 'openai.embeddings', 'openai.audio.speech',
               'openai.audio.transcriptions', 'openai.images.generations'),
        kind='router', auth_optional=True, hop=True, operator=True, include_usage=False, data={'text', 'image', 'audio'},
        image_formats={'png'}, extras={'forward_request_id': True}),
    _openai_compat(
        'pair_facade', None, _paths('openai.chat', 'openai.completions', 'openai.embeddings'), kind='router',
        auth_optional=True, hop=True, operator=True, include_usage=False),
    _openai_compat(
        'custom_openai', None, _frozen_map(_COMPAT_PATHS), auth_optional=True, hop=True, operator=True,
        data={'text', 'image', 'audio'}, image_formats={'png'},
        extras={'auth_options': ('bearer', 'azure_api_key', 'none')}),
    Profile(
        id='anthropic', adapter='anthropic', kind='cloud', default_base_url='https://api.anthropic.com', auth='anthropic',
        auth_optional=False,
        surfaces=_frozen_map({'anthropic.messages': '/v1/messages', 'anthropic.count_tokens': '/v1/messages/count_tokens'}),
        catalog_path='/v1/models', probe_path='/v1/models', ratelimit_headers=_ANTHROPIC_RATELIMIT,
        request_id_headers=('request-id',), data_modalities=frozenset({'text', 'image'}),
        features=frozenset({'tools', 'vision', 'thinking', 'prompt_caching'}),
        # bare_model_ids: the Anthropic SDK's bare model ids (claude-opus-5-5) resolve on the single visible Anthropic
        # upstream that lists them, on the Anthropic Messages surfaces only (catalog.resolve_vendor)
        extras=_frozen_map({'default_anthropic_version': '2023-06-01',
                            'structured_outputs_beta': 'structured-outputs-2025-11-13', 'bare_model_ids': True,
                            'ratelimit_per_model': True,
                            'model_capabilities': (('claude-*', ('chat', 'tools', 'vision')),)}),
        native=_ANTHROPIC_NATIVE, env_names=('ANTHROPIC_API_KEY',)),
    Profile(
        id='gemini', adapter='gemini', kind='cloud', default_base_url='https://generativelanguage.googleapis.com/v1beta',
        auth='goog', auth_optional=False,
        surfaces=_frozen_map({'gemini.generateContent': '/models/{model}:generateContent',
                              'gemini.streamGenerateContent': '/models/{model}:streamGenerateContent?alt=sse',
                              'gemini.embedContent': '/models/{model}:embedContent',
                              'gemini.batchEmbedContents': '/models/{model}:batchEmbedContents',
                              'gemini.countTokens': '/models/{model}:countTokens'}),
        catalog_path='/models', probe_path='/models', data_modalities=frozenset({'text', 'image', 'audio'}),
        features=frozenset({'tools', 'vision', 'json_schema', 'thinking'}),
        # bare_model_ids: google-genai puts the model in the URL path (models/gemini-2.5-flash:generateContent) and can
        # send nothing else, so a bare id resolves on the single visible Gemini upstream that lists it
        extras=_frozen_map({'catalog_page_token': 'pageToken', 'bare_model_ids': True, 'ratelimit_per_model': True}),
        native=_GEMINI_NATIVE,
        env_names=('GEMINI_API_KEY', 'GOOGLE_API_KEY')),
    Profile(
        id='elevenlabs', adapter='elevenlabs', kind='cloud', default_base_url='https://api.elevenlabs.io', auth='xi',
        auth_optional=False,
        surfaces=_frozen_map({'elevenlabs.tts': '/v1/text-to-speech/{voice_id}',
                              'elevenlabs.tts_stream': '/v1/text-to-speech/{voice_id}/stream',
                              'elevenlabs.stt': '/v1/speech-to-text', 'elevenlabs.voices': '/v1/voices',
                              'elevenlabs.voice': '/v1/voices/{voice_id}'}),
        catalog_path='/v1/models', probe_path='/v1/models', request_id_headers=('request-id', 'x-request-id'),
        usage_headers=('character-cost',), data_modalities=frozenset({'text', 'audio'}),
        # bare_model_ids: the SDK's bare model ids resolve on the single visible ElevenLabs upstream (catalog.
        # resolve_vendor); default_models: the provider's own default when a text-to-speech request names no model and
        # no single local speech model would answer; catalog_extra_models: usable models GET /v1/models does not list;
        # response_headers: the provider's informational headers returned on the typed surfaces too
        extras=_frozen_map({'default_output_format': 'mp3_44100_128', 'speed_range': (0.7, 1.2),
                            'bare_model_ids': True, 'default_models': _frozen_map({'speech': 'eleven_multilingual_v2'}),
                            'catalog_extra_models': ('scribe_v1', 'scribe_v2'),
                            'response_headers': frozenset({'history-item-id', 'x-character-count',
                                                           'current-concurrent-requests',
                                                           'maximum-concurrent-requests', 'x-trace-id'})}),
        native=_ELEVENLABS_NATIVE, env_names=('ELEVENLABS_API_KEY',)),
    Profile(
        id='ollama_cloud', adapter='ollama_native', kind='cloud', default_base_url='https://ollama.com', auth='bearer',
        auth_optional=False,
        surfaces=_frozen_map({'ollama.chat': '/api/chat', 'ollama.generate': '/api/generate', 'ollama.embed': '/api/embed'}),
        catalog_path='/api/tags', probe_path='/api/tags', data_modalities=frozenset({'text', 'image'}),
        features=frozenset({'tools', 'vision'}), native=_OLLAMA_NATIVE, env_names=('OLLAMA_API_KEY',)),
    # NVIDIA NeMo Switchyard as a router upstream. base_url is its ORIGIN (no /v1): the OpenAI and Anthropic surfaces
    # share it. auth none by default (Switchyard has no inbound auth); 'bearer' only behind an authenticating proxy, and a
    # credential is accepted only with auth bearer, so Titanos never hands a key to a proxy that may forward it on.
    # anthropic.* pass through the Anthropic adapter (surface_adapters) instead of being re-encoded as chat; the session
    # headers reach Switchyard (inbound_headers); its selected backend comes back in x-model-router-selected-model
    # (response_headers, recorded as a decision note); /v1/models carries per-route tool_calling, vision and
    # context_window (catalog_capabilities); store=false is forced on Responses unless the upstream allows store; one
    # circuit breaker per route (breaker_per_model), because each route has its own backends.
    Profile(
        id='switchyard', adapter='openai_compat', kind='router', default_base_url='http://127.0.0.1:4000', auth='none',
        auth_optional=True,
        surfaces=_frozen_map({'openai.chat': '/v1/chat/completions', 'openai.responses': '/v1/responses',
                              'anthropic.messages': '/v1/messages',
                              'anthropic.count_tokens': '/v1/messages/count_tokens'}),
        catalog_path='/v1/models', probe_path='/v1/models', allow_any_model_default=True,
        ratelimit_headers=_OPENAI_RATELIMIT, request_id_headers=('x-request-id',), include_usage=True, hop_header=True,
        data_modalities=frozenset({'text', 'image'}), operator_surfaces=True,
        features=frozenset({'tools', 'vision', 'json_schema'}),
        extras=_frozen_map({'auth_options': ('none', 'bearer'),
                            'surface_adapters': _frozen_map({'anthropic.messages': 'anthropic',
                                                             'anthropic.count_tokens': 'anthropic'}),
                            'inbound_headers': SWITCHYARD_SESSION_HEADERS,
                            'response_headers': frozenset({'x-model-router-selected-model'}),
                            'forward_request_id': True, 'default_anthropic_version': '2023-06-01',
                            'catalog_capabilities': True, 'force_store_false': True, 'breaker_per_model': True}),
        native=_SWITCHYARD_NATIVE),
    # Named slot for an external router or advisor whose wire contract has not been supplied: contract_unknown and
    # unroutable (503 upstream_unconfigured); no protocol is assumed.
    Profile(id='pending_contract', adapter='none', kind='slot', default_base_url=None, auth='none', auth_optional=True,
            surfaces=_EMPTY, extras=_frozen_map({'status': 'contract_unknown'})),
)

PROFILES: Mapping[str, Profile] = MappingProxyType({row.id: row for row in _ROWS})
assert len(PROFILES) == len(_ROWS), 'duplicate profile id'

# Profiles whose upstream.surfaces may narrow the row (operator-declared endpoints).
OPERATOR_SURFACE_PROFILES = frozenset(row.id for row in _ROWS if row.operator_surfaces)
# Documented as not registered: their routes are absent rather than 501 stubs, and the broker matrix marks them na.
DEFERRED = ('aws_bedrock', 'vertex_ai', 'azure_entra', 'cohere_v2', 'deepgram', 'cartesia', 'stability',
            'openai_realtime', 'openai_files_batches', 'elevenlabs_stream_input')


def profile(id: str) -> Profile:
    """The profile row; KeyError names the unknown id."""
    try:
        return PROFILES[id]
    except KeyError:
        raise KeyError(f'Unknown broker profile: {id!r}') from None


def image_b64_only(row: Profile, upstream_model: str | None) -> bool:
    """Whether the row's image model always answers base64 and refuses response_format (extras image_b64_only_globs,
    fnmatch globs over the upstream model id): such a model is never sent response_format."""
    from fnmatch import fnmatchcase
    return any(fnmatchcase(upstream_model or '', glob) for glob in row.extras.get('image_b64_only_globs') or ())


def hostnames() -> frozenset[str]:
    """Every default provider hostname in the table (for source scans that keep them out of other modules)."""
    from urllib.parse import urlsplit
    import ipaddress
    names = {urlsplit(row.default_base_url).hostname for row in _ROWS if row.default_base_url}
    names.update(host for row in _ROWS for host in row.extras.get('endpoint_hosts', ()))

    def literal(host: str) -> bool:  # a local router's loopback default (switchyard) is an address, not a provider name
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return False
        return True
    return frozenset(name for name in names if name and not literal(name))


def _check_rows() -> None:
    for row in _ROWS:
        assert row.adapter in ADAPTERS and row.kind in PROFILE_KINDS and row.auth in AUTH_TEMPLATES, row.id
        assert set(row.surfaces) <= set(SURFACES), (row.id, set(row.surfaces) - set(SURFACES))
        assert row.data_modalities <= DATA_MODALITIES, row.id
        assert row.params.tool_id_rule in TOOL_ID_RULES | {None}, row.id
        assert all(name.isupper() and not name.startswith('TITANOS_') for name in row.env_names), row.id
        split = row.extras.get('surface_adapters') or {}
        assert all(name in row.surfaces and adapter in ADAPTERS for name, adapter in split.items()), row.id
        assert all(h == h.lower() for h in row.extras.get('inbound_headers', ())), row.id
        assert not set(row.extras.get('inbound_headers', ())) & (NATIVE_HEADER_NEVER - {'origin'}), row.id
        native = row.native
        if native is None:
            continue
        assert native.root in ('base', 'origin'), row.id
        assert native.bare_prefix is None or native.bare_prefix == '' or native.bare_prefix.startswith('/'), row.id
        lowered = {h for h in native.request_headers | native.response_headers | native.rewrite_headers}
        assert all(h == h.lower() for h in lowered) and not (native.request_headers & NATIVE_HEADER_NEVER), row.id
        assert native.rewrite_headers <= native.response_headers, row.id
        seen = set()
        for route in native.routes:
            assert route.method in NATIVE_METHODS and route.owner in NATIVE_OWNERS, (row.id, route)
            assert route.path.startswith('/') and '//' not in route.path, (row.id, route)
            assert (route.method, route.path) not in seen, (row.id, route)
            seen.add((route.method, route.path))
            if route.owner == 'bound':
                names = re.findall(r'\{([A-Za-z_][A-Za-z0-9_]*)(?::path)?\}', route.path)
                assert names and (route.param is None or route.param in names), (row.id, route)


_check_rows()


def env_names() -> dict[str, tuple[str, ...]]:
    """R30: {profile id: well-known credential environment variable NAMES} for the rows that declare any."""
    return {row.id: row.env_names for row in _ROWS if row.env_names}


__all__ = ['ADAPTERS', 'AUTH_TEMPLATES', 'DATA_MODALITIES', 'DEFERRED', 'INBOUND_HEADER_ALLOW', 'NATIVE_HEADER_NEVER',
           'NATIVE_METHODS', 'NATIVE_OWNERS', 'NativeRoute', 'NativeSpec', 'OPERATOR_SURFACE_PROFILES', 'PROFILES',
           'PROFILE_KINDS', 'ParamMap', 'Profile', 'TITANOS_ONLY_FIELDS', 'TOOL_ID_RULES', 'env_names', 'hostnames',
           'image_b64_only', 'model_capabilities', 'profile']
