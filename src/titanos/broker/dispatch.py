"""BrokerDispatch: the ASGI middleware inside Guard that routes inference requests to local or cloud targets.

Fast path: a request that matches no inference surface, a catalog surface, or any request while the broker is inactive
(no broker.json, or one with no enabled upstream, alias or virtual model, AND a bootstrap or anonymous principal) goes
straight to the inner app: the body is not read and the scope is not touched, so responses stay byte-identical.

Intercepted surfaces (broker.surfaces, kind intercept):
1. the Guard-buffered body is read once; the model and stream flag are peeked (inbound.peek: a JSON parse, or a multipart
   part parse); model_absent=local_passthrough (ElevenLabs text-to-speech without model_id) and, for bootstrap and
   keyless principals, unparseable bodies go to the unchanged local route (a scoped key gets 400 invalid_request); a
   blank model is 400 model_required;
2. a RouteContext is built and BrokerRouter.route() resolves the chain (every refusal before any egress);
3. admission (Limiter.admit) runs before the first hit; budgets are re-checked per attempt (Admission.retarget);
4. attempts run in chain order with pre-commit fallback only:
   - LOCAL: the original route runs through the inner app with scope['titanos.broker.target'] set and a fresh replay of
     the body. Only for aliases and virtual models is the body's model rewritten to the local name (a JSON round trip with
     Content-Length fixed; multipart rebuilt with the same boundary). LocalAttempt holds http.response.start only while the
     status is 4xx/5xx, reads error.code from the server.error envelope and falls over on FALLBACK_LOCAL codes; anything
     else is forwarded (commit). Gemini and ElevenLabs speech-to-text requests to local models go through the inner
     OpenAI route with an encoded body (inbound.LOCAL_BRIDGE). Until it commits, a local attempt is bounded by the
     request deadline (min of X-Titanos-Deadline-Ms and client_timeout): expiry cancels it with 504 deadline_exceeded;
   - CLOUD and ROUTER: GrantIssuer issues the EgressGrant; same-protocol surfaces are passed through natively (the
     adapter holds the first complete frame), other protocols are translated (the first IR event is held). The commit
     point is after the first non-error frame or event, within broker_first_byte_timeout and the deadline. Error-first
     streams, upstream 429/5xx, timeouts and connect failures fall back (FALLBACK_CLOUD) when the surface is replay_safe
     or no request bytes were sent; upstream 400/404/410/413/422 are passed through in the inbound envelope and never
     fall back. A provider 404 or 410 on a model-bearing surface is 404 model_not_found (origin upstream; the message
     says the provider reports the model unavailable, or retired for 410), a neutral breaker outcome that never marks
     the credential. Fallback never happens for images.edits, store=true, previous_response_id or handle-bound
     requests. An expired request deadline (before the attempt or while waiting for the first byte) is 504 deadline_exceeded with a
     neutral breaker outcome, never a counted upstream timeout;
5. after commit the body flows through a bounded queue (32); translated OpenAI/Anthropic/Responses streams get ': keepalive'
   comments every keepalive_seconds while idle (never before commit); a client disconnect cancels the reader, closes the
   upstream response and settles usage (outcome cancelled, 499); a mid-stream upstream failure ends the stream with the
   protocol's terminal frame (binary audio: the connection is aborted);
6. provenance headers are stamped on http.response.start (streams included) and on errors; the Decision and the usage
   record are written; budgets settle with the observed usage.
Handle-bound Responses routes (retrieve, delete, cancel, input_items) go to the upstream that owns the handle, for its
owner (or an admin) only; anyone else gets 404 response_not_found with zero egress. The owner still passes the main
path's gates (cloud scope, upstream and class constraints, the 1-hop rule, location, role and cloud policy).
X-Request-ID: the client's value when valid, else the parent trace (managed_trace_headers), else a uuid4 hex.

Follow-ups (WP7):
- R15: a committed stream is bounded by the overall request deadline (min of X-Titanos-Deadline-Ms and client_timeout):
  on expiry it ends with the inbound protocol's terminal frame (code deadline_exceeded; binary audio and plain JSON
  bodies are aborted without a terminating chunk), the connection closes and the upstream request is cancelled. The
  breaker records nothing (a caller's deadline is never an upstream failure).
- R16: a request on an active surface that names no model (ElevenLabs text-to-speech without model_id) still records a
  Decision and applies the caller's target-class, model (the registry's single speech model the local route will use)
  and location constraints, and the local rate limits, before the unchanged local route runs. When no single local
  speech model exists and exactly one enabled ElevenLabs upstream is visible to a caller who may use the cloud, the
  request (text-to-speech and its stream) goes to that upstream with the provider's default model instead
  (_vendor_default; decision note model_absent:provider_default).
- R19: the advisor's reasoning_mode becomes reasoning_effort (OpenAI-compatible 'reasoning' profiles), an Anthropic
  thinking budget or a Gemini thinkingBudget ('thinking' profiles) when the request sets none.
- R22: anthropic.count_tokens to a local target runs the local /v1/messages/count_tokens route (broker/routes.py), which
  counts llama-cpp models with the native apply-template + tokenize path and refuses other local models.
- Surfaces whose routes exist only because of the broker (BROKER_ONLY_SURFACES: moderations, count_tokens, the Gemini
  actions, ElevenLabs streaming text-to-speech and speech-to-text, the handle-bound Responses routes) are dispatched even
  while the broker is inactive, so local targets reach them through the same bridges and refusals are typed; every
  pre-existing route keeps the inactive fast path (no body read, byte-identical answers).

Completeness round (2026-10-02):
- R32 native passthrough (broker/native.py): /native/<upstream>/<path> and the bare provider paths of the profile
  allowlists, for HTTP and WebSocket scopes. While the broker is not configured both go to the inner app unchanged (no
  route existed there). A native request records a Decision (surface 'native', translation 'native'), passes the native
  gates, is admitted by the limiter against the price row of native@<upstream>, audited before egress when it changes
  state, and is answered with the provider's status, body and allowlisted headers (errors scrubbed, a provider 401 is 502
  upstream_auth_failed). A WebSocket is authenticated here (the Guard handles HTTP only), keyless cross-site upgrades are
  refused, and frames are relayed both ways until either side closes or the request deadline expires. An HTTP request to
  a WebSocket-only row answers 400 broker_capability_unavailable naming what is missing (uvicorn accepts WebSocket
  upgrades only with the websockets or wsproto package installed).
- Unknown-argument policy: passthrough paths forward every client argument untouched (profile parameter maps note what
  they rename or drop); a translated path maps what it can and records every dropped field as a dropped:<field> decision
  note; the names are also sent in the X-Titanos-Dropped response header (when known before the first byte), so nothing
  is dropped silently.
- Anthropic: GET /v1/models and /v1/models/{id} with the anthropic-version header answer Titanos' catalog in Anthropic's
  shapes (broker/anthropic_catalog.py, in every mode); an Anthropic passthrough body that names provider-side state
  (container, custom skills, file_id sources) is checked before egress by broker/anthropic_state.py (403 without
  cloud:store or the upstream's allow_store, 404 for another key's uploaded file); the advisor's reasoning mode is shaped
  for the upstream model (translate/anthropic_models.py).
- OpenAI: a moderations request without a model goes to the provider's documented default on the single visible
  OpenAI upstream that lists it (_upstream_default, catalog.vendor_default; decision note model_absent:provider_default),
  else 400 model_required as before. A url image request that cannot be re-hosted (output_format jpeg or webp, or
  stream=true) is refused before egress (_check_rehost_request). GET /v1/responses/{id}?stream=true is streamed with
  Responses terminal frames. OpenAI media event streams (images, transcription, speech sse) end mid-stream with the
  OpenAI data:{error} frame.

Reachability round (2026-10-02, round 2):
- On-demand live catalogs: the live catalog cache is in memory, so after a restart (or a CLI `broker connect`, whose
  fetch ran in another process) no live model is known until a fetch. Before routing, a request whose model (or
  X-Titanos-Target, or policy.auto_model) could resolve through a stale live catalog fetches exactly those catalogs
  (catalog.live_pending, Broker.ensure_catalogs: single-flight per upstream, bounded by the request deadline, a failing
  upstream retried at most once a minute, upstreams the caller cannot see never contacted); so does a request that
  names no model on a provider-default surface (catalog.default_pending), a vendor-shaped model lookup or listing
  (GET /v1/models/{id}, /v1beta/models[/{id}], /api/tags), a POST to an API Plus or job route that may use a cloud
  model, and a model-decided native row. A qualified ref model@upstream never answers 404 only because no client listed
  models first.
- OpenAI videos through base_url /v1: POST /v1/videos goes to the OpenAI upstream when the body's model names a cloud
  model of it (native.choose_by_model: sora-2, sora-2@<upstream>, an alias; local models first), else to Titanos' local
  video job route as before; retrieve, content and delete follow a video id created through Titanos by the caller
  (owner binding). A request that stays on the local route still needs that route's scope (jobs): 403 scope_denied.
- Native lane breaker accounting: a request refused locally (path checks, egress refusals, deadline, concurrency) is a
  neutral 'refused' outcome on the shared upstream breaker, never a counted failure, so a key cannot take an upstream
  offline for everyone without a byte reaching the provider; check_rel runs again after normalize_models rewrote the
  path.

Reachability round 3 (2026-10-02):
- Native lane breaker accounting follows the typed lane: only upstream_bad_response counts (UPSTREAM_FAULT_CODES). A
  refused redirect and a declared size over the media cap depend on which provider object the caller asks for (its own
  large file, video content, dubbing audio, container file), so they are neutral too: no key can open the shared breaker
  for every other key by asking for its own objects.
- On-demand catalogs never hold local work: an API Plus or job POST (_stage) fetches only the stale live catalogs its
  body's model fields (model, *_model) could resolve through, and only when the request could be placed in the cloud at
  all (no X-Titanos-Hop; effective location from the X-Titanos-Location header, else the body's location, default
  local_only for API Plus; jobs: speech and image tasks, which the server resolves with location auto). A local model,
  a local_only body and a hop request contact no provider. Every on-demand wait is capped at
  facade.CATALOG_WAIT_SECONDS (and the request deadline); the catalog routes skip hop requests, wait for the request
  deadline (capped the same way), and Ollama's GET /api/tags never waits (it starts the fetch and lists what is cached,
  as its clients poll with short timeouts). The retry window of a failing catalog runs from the end of the attempt.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from typing import Any
from urllib.parse import parse_qsl, quote

import httpx
from starlette.requests import Request

from . import anthropic_catalog
from . import context as broker_context
from . import inbound
from .adapters.base import RawRequest, filter_inbound_headers, profile_inbound_headers
from .envelopes import protocol_for, render_error, stream_terminal
from .errors import BrokerError
from .policy import effective_policy
from .profiles import PROFILES, image_b64_only
from .adapters.openai_compat import accepts_reasoning_effort
from .role import parse_parent_headers
from .scopes import allows
from .sse import KEEPALIVE, SSESplitter
from .surfaces import TOKEN_OUTPUT_SURFACES
from .surfaces import match as match_surface
from . import native as native_lane
from .translate.anthropic_messages import EFFORT_BUDGET, MIN_THINKING_BUDGET, _unsigned_tool_turn
from .translate.upstream_errors import UpstreamFailure, model_failure, network_error, passthrough_4xx
from .types import Attempt, Candidate, RouteContext, Served, Target, Usage

log = logging.getLogger('titanos.broker')

FALLBACK_LOCAL = frozenset({'queue_full', 'rate_limit_error', 'model_not_loaded', 'worker_unavailable',
                            'model_load_failed', 'device_unavailable'})
FALLBACK_CLOUD = frozenset({'upstream_rate_limited', 'upstream_unavailable', 'upstream_timeout', 'network',
                            'breaker_open'})
# BrokerErrors raised while sending that count as a failure on the native lane's shared (upstream, credential) breaker,
# exactly as on the typed lane: only an answer the upstream got wrong whatever was asked (a malformed body or encoding).
# Every other BrokerError there is a neutral outcome: a local refusal (egress_refused, deadline, concurrency), and also a
# redirect or a declared size over TITANOS_BROKER_MAX_UPSTREAM_MEDIA_BYTES, which depend on the object the caller chose
# (its own large file, video or dubbing; a storage redirect), so one key could otherwise open the breaker for every key.
UPSTREAM_FAULT_CODES = frozenset({'upstream_bad_response'})
KEEPALIVE_SECONDS = 15.0
QUEUE_SIZE = 32
BUFFER_LIMIT = 65536
KEEPALIVE_PROTOCOLS = frozenset({'openai', 'anthropic', 'responses'})
# Surfaces whose routes exist only because of the broker: dispatched even while the broker is inactive (no existing
# route, so no byte-identity to keep; local targets reach them through the bridges and the local count route).
BROKER_ONLY_SURFACES = frozenset({'openai.moderations', 'anthropic.count_tokens', 'gemini.generateContent',
                                  'gemini.streamGenerateContent', 'gemini.embedContent', 'gemini.batchEmbedContents',
                                  'gemini.countTokens', 'elevenlabs.tts_stream', 'elevenlabs.stt',
                                  'openai.responses.retrieve', 'openai.responses.delete', 'openai.responses.cancel',
                                  'openai.responses.input_items'})
REASONING_MODES = ('low', 'medium', 'high')
GEMINI_GENERATE = frozenset({'gemini.generateContent', 'gemini.streamGenerateContent'})
IMAGE_SURFACES = frozenset({'openai.images.generations', 'openai.images.edits'})
# surfaces whose provider makes the model optional: a request without one goes to the single visible upstream's
# documented default (profile extras default_models), see BrokerDispatch._upstream_default
UPSTREAM_DEFAULT_SURFACES = frozenset({'openai.moderations'})
DROPPED_HEADER_LIMIT = 1024
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
_TRIGGER_BY_KIND = {'rate_limited': 'upstream_rate_limited', 'server_error': 'upstream_unavailable',
                    'timeout': 'upstream_timeout', 'network': 'network'}


class StreamAborted(Exception):
    """Raised after commit to abort a binary stream without a terminating chunk (a truncated transfer, never a short file
    that looks complete)."""


def _error_code(data: bytes) -> str | None:
    try:
        document = json.loads(data.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    error = document.get('error')
    if isinstance(error, dict):
        code = error.get('code') or error.get('type')
        return code if isinstance(code, str) else None
    detail = document.get('detail')
    if isinstance(detail, dict) and isinstance(detail.get('status'), str):
        return detail['status']
    code = document.get('code')
    return code if isinstance(code, str) else None


def _latin1(value: str) -> str:
    try:
        value.encode('latin-1')
        if all(32 <= ord(c) < 127 for c in value):
            return value
    except UnicodeEncodeError:
        pass
    return quote(value, safe='@/:._-+=~')


class LocalAttempt:
    """Capturing send for one local attempt. A non-error http.response.start is forwarded at once (commit) with the
    provenance headers added. A 4xx/5xx start and its body (up to buffer_limit bytes) are held until the body ends; then
    error.code (the server.error envelope) decides: a code in fallback_codes means fallback (nothing is forwarded),
    anything else is flushed as it was. A body larger than buffer_limit is flushed (commit)."""

    def __init__(self, send, *, fallback_codes: frozenset = frozenset(), buffer_limit: int = BUFFER_LIMIT,
                 headers=()):
        self.send = send
        self.fallback_codes = frozenset(fallback_codes)
        self.buffer_limit = int(buffer_limit)
        self.extra_headers = list(headers)
        self.start: dict | None = None
        self.buffer = bytearray()
        self.committed = False
        self.fallback: str | None = None
        self.status: int | None = None
        self.error_code: str | None = None
        self.started = time.perf_counter()
        self.ms_to_headers: float | None = None
        self.ms_to_first_byte: float | None = None
        self.done = False
        self.closed = False  # set when the attempt is abandoned before commit (deadline): later messages are dropped

    async def _commit(self, start: dict) -> None:
        names = {k.lower() for k, _ in self.extra_headers}
        headers = [(k, v) for k, v in start.get('headers', []) if k.lower() not in names] + self.extra_headers
        self.committed = True
        await self.send(dict(start, headers=headers))

    async def flush(self) -> None:
        """Forward a held error response (the last attempt of a chain)."""
        if self.committed or self.start is None:
            return
        await self._commit(self.start)
        await self.send({'type': 'http.response.body', 'body': bytes(self.buffer), 'more_body': not self.done})
        self.fallback = None

    async def __call__(self, message: dict) -> None:
        if self.closed and not self.committed:
            return
        kind = message.get('type')
        if kind == 'http.response.start':
            self.status = int(message.get('status', 200))
            self.ms_to_headers = (time.perf_counter() - self.started) * 1000.0
            if self.status < 400 or not self.fallback_codes:
                await self._commit(message)
            else:
                self.start = message
            return
        if kind == 'http.response.body':
            if self.ms_to_first_byte is None:
                self.ms_to_first_byte = (time.perf_counter() - self.started) * 1000.0
            if self.committed:
                await self.send(message)
                return
            if self.fallback is not None:
                return
            if self.start is None:  # a body before any start: forward as is
                await self.send(message)
                return
            self.buffer.extend(message.get('body', b''))
            more = bool(message.get('more_body', False))
            if len(self.buffer) > self.buffer_limit:
                self.done = not more
                await self.flush()
                return
            if not more:
                self.done = True
                self.error_code = _error_code(bytes(self.buffer))
                if self.error_code in self.fallback_codes:
                    self.fallback = self.error_code
                    return
                await self.flush()
            return
        await self.send(message)


def request_hints(headers, settings, principal):
    """role.parse_parent_headers, plus the MANAGED HEADERS trace rule: when the request carries no valid X-Request-ID
    and a managed_trace_headers value is present (already restricted to at most 128 safe characters), that trace value
    is the request id echoed as X-Request-ID."""
    hints = parse_parent_headers(headers, settings, principal)
    if hints.trace_id and inbound.header(headers, 'x-request-id') != hints.request_id:
        hints = replace(hints, request_id=hints.trace_id)
    return hints


# An attempt needs at least this much of the caller's deadline: less cannot cover a connection and a first byte, so
# it is refused before any egress (no upstream hit, no spend, no breaker probe) and is 504 deadline_exceeded.
MIN_ATTEMPT_SECONDS = 0.01
# An attempt that fails within this much of the caller's deadline counts as that deadline (504 deadline_exceeded, a
# neutral breaker outcome): egress caps httpx's own timeouts at the same deadline, so the two race by microseconds.
DEADLINE_SLACK_SECONDS = 0.05


def _deadline_error() -> BrokerError:
    return BrokerError('The request deadline expired', 'deadline_exceeded')


def _replay(body: bytes, receive):
    sent = False

    async def replay():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': body, 'more_body': False}
        return await receive()
    return replay


async def _read_body(receive) -> bytes | None:
    chunks = bytearray()
    while True:
        message = await receive()
        if message.get('type') == 'http.disconnect':
            return None
        chunks.extend(message.get('body', b''))
        if not message.get('more_body', False):
            return bytes(chunks)


# the asset job tasks the server may place in the cloud (server.cloud_job_stage, resolved with location 'auto');
# video and music jobs run on local models only
JOB_CLOUD_TASKS = frozenset({'speech', 'image'})


def _stage_pending(broker, surface, body: bytes, hints, principal) -> list[str]:
    """The live-catalog upstreams whose stale catalog an API Plus or job POST could resolve through: none when the
    request cannot be placed in the cloud (effective location local_only: the X-Titanos-Location header, else the body's
    location, default local_only on API Plus routes; a job task that runs locally only), else the live_pending upstreams
    of every top-level model field of the body (model, and every *_model such as llm_model or classifier_model). A local
    model names nothing. Never raises for a malformed body (the route answers it)."""
    try:
        document = json.loads(body) if body else None
    except (ValueError, UnicodeDecodeError):
        return []
    if not isinstance(document, dict):
        return []
    if surface.name == 'jobs.create':
        if document.get('task') not in JOB_CLOUD_TASKS:
            return []
        location = hints.location or 'auto'
    else:
        stated = document.get('location')
        location = hints.location or (stated if isinstance(stated, str) and stated else 'local_only')
    if location == 'local_only':
        return []
    principal = principal or broker.default_principal()
    catalog = broker.snapshot().catalog
    out: list[str] = []
    for key, value in document.items():
        if isinstance(key, str) and (key == 'model' or key.endswith('_model')) and isinstance(value, str) and value:
            for upstream_id in catalog.live_pending(value, surface.protocol, principal):
                if upstream_id not in out:
                    out.append(upstream_id)
    return out


@dataclass
class _Failure:
    trigger: str | None = None
    request_bytes_sent: bool = True
    error: BrokerError | None = None
    local: LocalAttempt | None = None
    passthrough: UpstreamFailure | None = None
    final: bool = False


@dataclass
class _Run:
    scope: dict
    receive: Any
    send: Any
    surface: Any
    params: dict
    query: dict
    principal: Any
    hints: Any
    peek: inbound.Peek
    body: bytes
    protocol: str
    snapshot: Any
    started: float
    ctx: RouteContext | None = None
    plan: Any = None
    decision: Any = None
    admission: Any = None
    fallback_count: int = 0
    committed: bool = False
    served: Any = None
    status: int | None = None
    outcome: str = 'success'
    usage: Usage | None = None
    upstream_request_id: str | None = None
    ttfb_ms: float | None = None
    translation: str | None = None
    injected: bool = False
    notes: list = field(default_factory=list)
    router_ids: list = field(default_factory=list)  # X-Request-IDs this run holds in router.ROUTER_INFLIGHT
    observed_response_id: str | None = None
    sequence_number: int = 0


class _ResponsesObserver:
    """Reads the outgoing Responses SSE bytes for the response id (handle binding) and the last sequence_number (the
    response.failed terminal frame)."""

    JSON_LIMIT = 1 << 20

    def __init__(self, run: _Run, *, sse: bool = True):
        self.run = run
        self.sse = sse
        self.splitter = SSESplitter()
        self.body = bytearray()

    def finish(self) -> None:
        """A non-streamed answer: the response id from the JSON body."""
        if self.sse or self.run.observed_response_id is not None or not self.body:
            return
        try:
            document = json.loads(bytes(self.body).decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return
        if isinstance(document, dict) and isinstance(document.get('id'), str):
            self.run.observed_response_id = document['id']

    def __call__(self, chunk: bytes) -> None:
        if not self.sse:
            if len(self.body) + len(chunk) <= self.JSON_LIMIT:
                self.body.extend(chunk)
            return
        try:
            frames = self.splitter.feed(chunk)
        except BrokerError:
            return
        for _, frame in frames:
            if frame is None or not frame.data or frame.data == '[DONE]':
                continue
            try:
                payload = json.loads(frame.data)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            if isinstance(payload.get('sequence_number'), int):
                self.run.sequence_number = payload['sequence_number']
            response = payload.get('response')
            if self.run.observed_response_id is None and isinstance(response, dict) and \
                    isinstance(response.get('id'), str):
                self.run.observed_response_id = response['id']


def inject_reasoning(surface_name: str, document: dict, mode: str, features, notes: list, *,
                     model: str | None = None) -> dict:
    """R19: the advisor's reasoning_mode in a passthrough body when the request sets none and the target supports it:
    reasoning_effort (openai.chat, 'reasoning' profiles), Anthropic thinking (anthropic.messages, 'thinking' profiles;
    shaped for the upstream model by translate/anthropic_models.inject_thinking: budget_tokens on budget-mode models only
    when max_tokens leaves room for the minimum budget, no sampling override is set and no tool loop in progress lacks
    its signed thinking; adaptive thinking plus output_config.effort on 4.6 and later; without a model the budget form)
    or a Gemini thinking config (generateContent, 'thinking' profiles; translate.gemini.with_thinking: thinkingLevel on
    gemini-3 and later, thinkingBudget otherwise, nothing for models without thinking; without a model the budget)."""
    out = dict(document)
    budget = EFFORT_BUDGET.get(mode, 4096)
    if surface_name == 'openai.chat':
        if 'reasoning' in features and 'reasoning_effort' not in out:
            out['reasoning_effort'] = mode
            notes.append('injected:reasoning_effort')
    elif surface_name == 'anthropic.messages' and model and 'thinking' in features:
        from .translate.anthropic_models import inject_thinking
        out = inject_thinking(out, mode, model, notes)
    elif surface_name == 'anthropic.messages':
        max_tokens = out.get('max_tokens')
        messages = out.get('messages') if isinstance(out.get('messages'), list) else []
        if ('thinking' in features and 'thinking' not in out and isinstance(max_tokens, int)
                and not isinstance(max_tokens, bool) and max_tokens > MIN_THINKING_BUDGET
                and not any(key in out for key in ('temperature', 'top_k', 'top_p'))
                and not _unsigned_tool_turn(messages)):
            out['thinking'] = {'type': 'enabled', 'budget_tokens': max(MIN_THINKING_BUDGET, min(budget, max_tokens - 1))}
            notes.append('injected:thinking')
    elif surface_name in GEMINI_GENERATE and 'thinking' in features:
        # either field spelling; thinkingLevel on gemini-3 and later, none for models without thinking (TTS, image)
        from .translate.gemini import with_thinking
        out = with_thinking(out, mode, model, notes)
    return out


def inbound_estimate(run: '_Run'):
    """The router's estimate for this request (a local admission counts RPM, TPM and concurrency only)."""
    from .router import estimate_for
    ctx = run.ctx or RouteContext(surface=run.surface, requested_model=run.peek.model or '', principal=run.principal,
                                  hints=run.hints, body=run.peek.summary)
    return estimate_for(ctx)


class BrokerDispatch:
    """ASGI middleware. keepalive_seconds and queue_size are attributes so tests can scale time."""

    def __init__(self, app, broker):
        self.app = app
        self.broker = broker
        self.keepalive_seconds = KEEPALIVE_SECONDS
        self.queue_size = QUEUE_SIZE
        self.intercepted = 0  # bodies read and parsed (instrumentation: the inactive fast path never increments it)

    # -- entry -----------------------------------------------------------------------------------------------------------
    async def __call__(self, scope, receive, send):
        kind = scope.get('type')
        if kind == 'websocket':
            path = scope.get('path', '') or ''
            if native_lane.split_prefix(path) is None and not native_lane.bare_rows('WS', path):
                return await self.app(scope, receive, send)
            try:
                configured = self.broker.configured
            except Exception:
                log.exception('broker: activity check failed; passing the request through')
                configured = False
            if not configured:
                return await self.app(scope, receive, send)
            return await self._native_ws(scope, receive, send)
        if kind != 'http':
            return await self.app(scope, receive, send)
        method = scope.get('method', 'GET')
        found = match_surface(method, scope.get('path', ''))
        if found is None or found[0].kind == 'catalog':
            native = self._native_match(scope, method, catalog=found is not None)
            if native is not None:
                return await self._native_http(scope, receive, send, native)
            if found is None:
                denied = self._listed_scope_denied(scope, method)
                if denied is not None:
                    return await self._send_response(scope, receive, send,
                                                     render_error(protocol_for(scope.get('path', '')), denied))
            if found is not None:
                await self._warm_catalogs(scope, found[0], found[1])
            if found is not None and anthropic_catalog.wants(scope):
                # the Anthropic SDK's models.list / retrieve: Titanos' catalog in Anthropic's shapes
                return await anthropic_catalog.serve(self.app, scope, receive, send, self.broker)
            return await self.app(scope, receive, send)
        principal = scope.get('titanos.principal')
        try:
            active = self.broker.active(principal)
        except Exception:
            log.exception('broker: activity check failed; passing the request through')
            active = False
        surface, params = found
        if not active and surface.name not in BROKER_ONLY_SURFACES:
            return await self.app(scope, receive, send)
        if surface.kind == 'stage':
            return await self._stage(scope, receive, send, surface, principal)
        return await self._intercept(scope, receive, send, surface, params, principal)

    async def _stage(self, scope, receive, send, surface, principal):
        """API Plus and jobs: the principal and hints go into ContextVars for StageRouter and CloudInvoker."""
        protocol = protocol_for(scope.get('path', ''))
        try:
            hints = request_hints(scope.get('headers', []), self.broker.settings, principal)
        except BrokerError as exc:
            return await self._send_response(scope, receive, send, render_error(protocol, exc))
        if str(scope.get('method', '')).upper() == 'POST' and not hints.hop and hints.location != 'local_only' \
                and self._live_stale(principal):
            # the stage resolves its model on a worker thread, which cannot fetch: the stale live catalogs its body's
            # models could resolve through are fetched first, only when the request may be placed in the cloud
            body = await _read_body(receive)
            if body is None:
                return None  # the client went away
            receive = _replay(body, receive)
            try:
                pending = _stage_pending(self.broker, surface, body, hints, principal)
                if pending:
                    await self.broker.ensure_catalogs(principal, pending, deadline=hints.deadline)
            except Exception:
                log.warning('broker: on-demand live catalogs failed', exc_info=True)
        token = broker_context.bind(principal or self.broker.default_principal(), hints)
        try:
            return await self.app(scope, receive, send)
        finally:
            broker_context.reset(token)

    # -- on-demand live catalogs and listed scopes -------------------------------------------------------------------------
    def _live_stale(self, principal) -> bool:
        """Whether any enabled live-catalog upstream principal may see has a stale catalog (a cheap check before a stage
        body is parsed). Never raises."""
        broker = self.broker
        try:
            if not broker.switches.live_catalogs:
                return False
            snap = broker.snapshot()
            principal = principal or broker.default_principal()
            return any(upstream.catalog == 'live' and upstream.enabled and snap.catalog.stale(upstream_id)
                       and broker.visible_upstream(principal, upstream_id)
                       for upstream_id, upstream in snap.config.upstreams.items())
        except Exception:
            log.warning('broker: live catalog check failed', exc_info=True)
            return False

    async def _ensure_live(self, run: '_Run', *, default: bool = False) -> None:
        """Fetch the stale live catalogs that could make this request's model resolve (module docstring), bounded by the
        request deadline. default: the request names no model (catalog.default_pending). Never raises."""
        snapshot = run.snapshot
        try:
            if run.hints.hop:
                return  # a request from another broker hop stays local
            catalog = snapshot.catalog
            if default:
                pending = catalog.default_pending(run.surface.protocol, run.surface.capability, run.principal)
            else:
                explicit = run.hints.target_header
                name = explicit or (run.peek.model if isinstance(run.peek.model, str) else '')
                auto = snapshot.config.policy.auto_model
                if name == 'auto' and explicit is None and auto and catalog.resolve_name(name) is None:
                    name = auto
                pending = catalog.live_pending(name, run.surface.protocol, run.principal)
            if pending:
                await self.broker.ensure_catalogs(run.principal, pending, deadline=run.hints.deadline)
        except Exception:
            log.warning('broker: on-demand live catalogs failed', exc_info=True)

    async def _warm_catalogs(self, scope, surface, params) -> None:
        """Catalog routes: a vendor-shaped single-model lookup fetches the stale live catalogs its name could resolve
        through; a listing (Titanos', Gemini's or Ollama's shape) the stale ones the caller may see, as GET /v1/models
        always did. Never raises."""
        if surface.name not in ('openai.models', 'openai.model', 'gemini.models', 'gemini.model', 'ollama.tags'):
            return
        broker = self.broker
        principal = scope.get('titanos.principal')
        try:
            if not broker.active(principal):
                return
            hints = request_hints(scope.get('headers', []), broker.settings, principal)
            if hints.hop:
                return  # a request from another broker hop stays local: it never makes this server contact a provider
            # Ollama clients poll /api/tags with short timeouts: the fetch starts and the listing answers from the cache
            deadline = time.monotonic() if surface.name == 'ollama.tags' else hints.deadline
            principal = principal or broker.default_principal()
            if surface.name in ('openai.model', 'gemini.model'):
                name = str((params or {}).get('name') or (params or {}).get('model') or '')
                if name.startswith('models/'):
                    name = name[len('models/'):]
                protocol = 'anthropic' if anthropic_catalog.wants(scope) else surface.protocol
                pending = broker.snapshot().catalog.live_pending(name, protocol, principal)
                if pending:
                    await broker.ensure_catalogs(principal, pending, deadline=deadline)
            else:
                await broker.ensure_catalogs(principal, deadline=deadline)
        except BrokerError:
            return  # malformed hint headers: the catalog route answers from the cache
        except Exception:
            log.warning('broker: on-demand live catalogs failed', exc_info=True)

    @staticmethod
    def _listed_scope_denied(scope, method: str) -> BrokerError | None:
        """A path where a provider row may win over a Titanos route (native.bare_may_win: OpenAI /v1/videos) passes the
        Guard with '<listed scope>|native'; when the Titanos route answers, the caller must hold the listed scope."""
        path = scope.get('path', '') or ''
        if not native_lane.bare_may_win(method, path):
            return None
        principal = scope.get('titanos.principal')
        needed = scopes_listed(method, path)
        if principal is None or needed is None or allows(principal, needed):
            return None
        return BrokerError(f'This credential lacks the scope this route requires ({needed})', 'scope_denied',
                           origin='policy')

    # -- the intercepted path ---------------------------------------------------------------------------------------------
    async def _intercept(self, scope, receive, send, surface, params, principal):
        broker = self.broker
        protocol = protocol_for(scope.get('path', ''))
        body = await _read_body(receive)
        if body is None:
            return
        self.intercepted += 1
        query = dict(parse_qsl((scope.get('query_string') or b'').decode('latin-1'), keep_blank_values=True))
        if principal is None:
            principal = broker.default_principal()
        try:
            hints = request_hints(scope.get('headers', []), broker.settings, principal)
        except BrokerError as exc:
            return await self._send_response(scope, receive, send, render_error(protocol, exc))
        peek_ = await inbound.peek(surface, params, body, scope.get('headers', []), query)
        run = _Run(scope=scope, receive=receive, send=send, surface=surface, params=params, query=query,
                   principal=principal, hints=hints, peek=peek_, body=body, protocol=protocol,
                   snapshot=broker.snapshot(), started=time.perf_counter())
        token = broker_context.bind(principal, hints)
        try:
            if surface.handle_bound:
                return await self._handle_route(run)
            if peek_.parse_error and getattr(principal, 'kind', None) in ('key', 'parent'):
                # fail closed: a body the broker cannot read never reaches a local route unchecked for a scoped key
                # (visibility, constraints and limits are decided on the parsed model)
                return await self._send_response(scope, receive, send, render_error(protocol, BrokerError(
                    'The request body could not be parsed', 'invalid_request')))
            if peek_.parse_error:
                return await self.app(scope, _replay(body, receive), send)
            if peek_.model is None and (surface.protocol != 'openai' or surface.name in UPSTREAM_DEFAULT_SURFACES):
                await self._ensure_live(run, default=True)  # the provider default must be listed by a live catalog
            if peek_.model is None and surface.protocol != 'openai' and self._vendor_default(run):
                return await self._route_and_run(run)
            if peek_.model is None and surface.name in UPSTREAM_DEFAULT_SURFACES and self._upstream_default(run):
                return await self._route_and_run(run)
            if peek_.model is None and surface.model_absent == 'local_passthrough':
                return await self._model_absent(run)
            return await self._route_and_run(run)
        finally:
            broker_context.reset(token)

    def _upstream_default(self, run: '_Run') -> bool:
        """An OpenAI-protocol request on a surface whose provider makes the model optional (OpenAI moderations default
        to omni-moderation-latest; UPSTREAM_DEFAULT_SURFACES) that names no model goes to the provider's documented
        default on the ONE enabled, ready upstream with a default for the capability (profile extras default_models,
        catalog.vendor_default: it must list the model and be visible to the caller). The peeked model becomes that
        qualified ref and the router applies every gate; the decision notes model_absent:provider_default. False changes
        nothing (400 model_required as before)."""
        snapshot = run.snapshot
        try:
            target = snapshot.catalog.vendor_default('openai', run.surface.capability, run.principal)
        except BrokerError:
            return False
        if target is None:
            return False
        run.peek.model = target.ref
        run.notes.append('model_absent:provider_default')
        return True

    def _implied_local(self) -> str | None:
        """The registry's single speech model (the one the local text-to-speech route uses without model_id), or None."""
        try:
            models = self.broker.local_models()
        except Exception:
            return None
        names = [name for name, model in models.items() if getattr(model, 'task', None) == 'speech']
        return names[0] if len(names) == 1 else None

    def _vendor_default(self, run: _Run) -> bool:
        """A vendor-protocol request that names no model (ElevenLabs text-to-speech without model_id, which the provider
        answers with its own default model) goes to that provider when no single local speech model would answer it (the
        R16 local route keeps precedence), cloud use is possible for this caller (cloud policy not off, location not
        local_only) and exactly one enabled, ready upstream of the protocol with a default for the capability is visible
        (catalog.vendor_default). The peeked model becomes that qualified ref; the decision notes model_absent:
        provider_default. False changes nothing."""
        surface = run.surface
        if surface.capability != 'speech' or self._implied_local() is not None:
            return False
        snapshot = run.snapshot
        try:
            eff = effective_policy(self.broker.settings, snapshot.config, snapshot.switches, run.principal, run.hints,
                                   surface)
            if eff.cloud_policy == 'off' or eff.location == 'local_only':
                return False
            target = snapshot.catalog.vendor_default(surface.protocol, surface.capability, run.principal)
        except BrokerError:
            return False
        if target is None:
            return False
        run.peek.model = target.ref
        run.notes.append('model_absent:provider_default')
        return True

    async def _model_absent(self, run: _Run):
        """R16: an active-surface request that names no model (ElevenLabs text-to-speech without model_id). A Decision is
        recorded and the caller's constraints are applied before the unchanged local route runs: the 'local' target class,
        the model globs against the registry's single speech model (the model the local route will use), a location the
        caller may use that is not cloud_only, and the local rate limits (RPM, TPM, concurrency). Refusals are 404
        model_not_found (invisible) or 400 location_unsatisfiable, before the local route sees the body."""
        snapshot = run.snapshot
        principal = run.principal
        ctx = RouteContext(surface=run.surface, requested_model='', principal=principal, hints=run.hints,
                           body=run.peek.summary)
        run.ctx = ctx
        decision = snapshot.router.new_decision(ctx, inbound_protocol=run.protocol)
        run.decision = decision
        try:
            implied = self._implied_local()
            target = snapshot.catalog.local_target(implied) if implied else None
            if target is None:
                target = Target(ref=implied or 'local', klass='local', local_name=implied, adapter='local')
            decision.candidates = [Candidate(ref=target.ref, klass='local', eligible=True)]
            decision.notes.append('model_absent:local_route')
            try:
                constraints = principal.constraints
                if 'local' not in constraints.target_classes or (
                        implied is not None and not any(fnmatchcase(implied, p) for p in constraints.models)):
                    decision.candidates = [Candidate(ref=target.ref, klass='local', eligible=False, excluded='scope')]
                    raise BrokerError('Model not found', 'model_not_found', origin='policy')
                eff = effective_policy(self.broker.settings, snapshot.config, snapshot.switches, principal, run.hints,
                                       run.surface)
                decision.effective = {'cloud_policy': eff.cloud_policy, 'location': eff.location}
                if eff.location == 'cloud_only':
                    raise BrokerError('No target satisfies the requested location', 'location_unsatisfiable',
                                      origin='policy')
                decision.chain = [target.ref]
                run.admission = self.broker.limiter.admit(principal, target, inbound_estimate(run))
            except BrokerError as exc:
                run.outcome, run.status = 'refused', exc.status
                await self._refuse(run, exc)
                return
            await self._attempt_local(run, target, 0, False)
        finally:
            await self._finish(run)

    async def _route_and_run(self, run: _Run):
        broker = self.broker
        snapshot = run.snapshot
        router = snapshot.router
        await self._ensure_live(run)  # a stale live catalog never turns a well-formed ref into 404 model_not_found
        ctx = RouteContext(surface=run.surface, requested_model=run.peek.model if isinstance(run.peek.model, str) else '',
                           principal=run.principal, hints=run.hints, body=run.peek.summary)
        run.ctx = ctx
        decision = router.new_decision(ctx, inbound_protocol=run.protocol)
        run.decision = decision
        try:
            try:
                run.plan = await router.route(ctx, decision,
                                              advisor_input=lambda: inbound.advisor_input(run.surface, run.peek.json))
                run.admission = router.admit(run.plan, ctx)
            except BrokerError as exc:
                run.outcome, run.status = 'refused', exc.status
                await self._refuse(run, exc)
                return
            for note in run.plan.notes:
                if note not in run.notes:
                    run.notes.append(note)
            try:
                await self._execute(run)
            except BaseException:
                if not run.committed:
                    run.outcome = 'cancelled' if run.outcome == 'success' else run.outcome
                raise
        finally:
            await self._finish(run)

    # -- execution -------------------------------------------------------------------------------------------------------
    def _no_fallback(self, run: _Run) -> bool:
        body = run.peek.summary
        return (run.surface.name == 'openai.images.edits' or bool(body.store) or bool(body.previous_response_id)
                or run.surface.handle_bound)

    def _may_fallback(self, run: _Run, failure: _Failure) -> bool:
        if failure.final or failure.trigger is None or self._no_fallback(run):
            return False
        if failure.trigger not in run.plan.fallback_on:
            return False
        if failure.local is None and failure.request_bytes_sent and not run.surface.replay_safe:
            return False
        return True

    async def _execute(self, run: _Run) -> None:
        chain = run.plan.chain
        failure: _Failure | None = None
        for index, target in enumerate(chain):
            if index > 0 and (failure is None or not self._may_fallback(run, failure)):
                break
            left = self._left(run)
            if left is not None and left <= 0:
                run.decision.attempts.append(Attempt(ref=target.ref, outcome='refused', status=504,
                                                     error_code='deadline_exceeded', request_bytes_sent=False))
                failure = _Failure(error=_deadline_error(), final=True, request_bytes_sent=False)
                break
            if index > 0:
                run.fallback_count += 1
                self.broker.metrics.fallback(failure.trigger)
            has_next = (index + 1 < len(chain)) and not self._no_fallback(run)
            if target.klass == 'local':
                failure = await self._attempt_local(run, target, index, has_next)
            else:
                failure = await self._attempt_cloud(run, target, index, has_next)
            if failure is None:
                return
        await self._render_failure(run, failure)

    @staticmethod
    def _left(run: _Run) -> float | None:
        """Seconds left before the request deadline (min of X-Titanos-Deadline-Ms and client_timeout), or None."""
        deadline = run.hints.deadline
        return None if deadline is None else deadline - time.monotonic()

    async def _bounded(self, run: _Run, attempt, budget: float, by_deadline: bool):
        """asyncio.wait_for(attempt, budget) with the caller's deadline authoritative. On Python 3.11 wait_for can return
        the attempt's result, or raise its exception, after its own timeout fired (a cancelled httpx call that finishes or
        fails while being cancelled). Once the deadline that set the budget has passed, the attempt counts as
        asyncio.TimeoutError (504 deadline_exceeded, a neutral breaker outcome) and a produced response is closed unread."""
        def expired() -> bool:
            left = self._left(run)
            return by_deadline and left is not None and left <= DEADLINE_SLACK_SECONDS
        try:
            produced = await asyncio.wait_for(attempt, timeout=budget)
        except asyncio.TimeoutError:
            raise
        except Exception:
            if expired():
                raise asyncio.TimeoutError from None
            raise
        if expired():
            try:
                await produced.aclose()
            except Exception:
                pass
            raise asyncio.TimeoutError
        return produced

    async def _render_failure(self, run: _Run, failure: _Failure | None) -> None:
        if failure is None:
            await self._refuse(run, BrokerError('No target could serve the request', 'upstream_unavailable'))
            return
        if failure.local is not None:
            run.outcome, run.status = 'local_error', failure.local.status
            await failure.local.flush()
            run.committed = True
            return
        if failure.passthrough is not None:
            run.outcome, run.status = 'client_error', failure.passthrough.status
            response = passthrough_4xx(failure.passthrough.upstream, 'responses' if run.surface.name ==
                                       'openai.responses' else run.protocol)
            for name, value in self._provenance(run, error=True).items():
                response.headers[name] = value
            await self._send_response(run.scope, run.receive, run.send, response)
            run.committed = True
            return
        error = failure.error or BrokerError('No target could serve the request', 'upstream_unavailable')
        run.outcome, run.status = 'error', error.status
        await self._refuse(run, error)

    # -- provenance ------------------------------------------------------------------------------------------------------
    def _served_by(self, target) -> str:
        if target.klass == 'local':
            kind = self.broker.local_kind(target.local_name or target.ref)
            return f'local:{kind}'
        return f'{target.klass}:{target.upstream_id}'

    def _provenance(self, run: _Run, *, target=None, error: bool = False) -> dict[str, str]:
        headers = {'X-Request-ID': run.hints.request_id,
                   'X-Titanos-Fallback-Count': str(run.fallback_count)}
        if run.decision is not None:
            headers['X-Titanos-Decision-Id'] = run.decision.decision_id
        if run.peek.model:
            headers['X-Titanos-Requested-Model'] = _latin1(run.peek.model)
        if target is not None:
            headers['X-Titanos-Served-By'] = self._served_by(target)
            headers['X-Titanos-Served-Model'] = _latin1(target.ref)
        if run.translation:
            headers['X-Titanos-Translation'] = run.translation
        if run.upstream_request_id:
            headers['X-Titanos-Upstream-Request-ID'] = _latin1(str(run.upstream_request_id)[:128])
        if run.injected:
            headers['X-Titanos-Injected'] = 'max_tokens'
        dropped = self._dropped(run)
        if dropped:
            headers['X-Titanos-Dropped'] = dropped
        if (target is not None and target.klass != 'local' and getattr(self.broker.settings,
                                                                       'broker_debug_upstream_urls', False)
                and allows(run.principal, 'admin')):
            upstream = run.snapshot.config.upstreams.get(target.upstream_id)
            url = upstream.effective_base_url() if upstream is not None else None
            if url:
                headers['X-Titanos-Upstream-URL'] = _latin1(url)
        return headers

    @staticmethod
    def _dropped(run: _Run) -> str:
        """The request fields a translated (or parameter-mapped passthrough) path dropped, from the dropped:<field>
        notes known so far: comma-separated names, latin-1 safe, at most DROPPED_HEADER_LIMIT characters (the decision
        log keeps every note)."""
        names: list[str] = []
        notes = list(run.notes) + (list(run.decision.notes) if run.decision is not None else [])
        for note in notes:
            if not isinstance(note, str) or not note.startswith('dropped:'):
                continue
            name = note[len('dropped:'):].split(' ', 1)[0].strip()
            if name and name not in names:
                names.append(name)
        text = ', '.join(_latin1(name) for name in names)
        if len(text) > DROPPED_HEADER_LIMIT:
            text = text[:DROPPED_HEADER_LIMIT].rsplit(',', 1)[0] + ', ...'
        return text

    @staticmethod
    def _pairs(headers: dict[str, str]) -> list[tuple[bytes, bytes]]:
        return [(k.lower().encode('latin-1'), v.encode('latin-1')) for k, v in headers.items()]

    async def _send_response(self, scope, receive, send, response) -> None:
        await response(scope, receive, send)

    async def _refuse(self, run: _Run, exc: BrokerError) -> None:
        response = render_error(run.protocol, exc)
        for name, value in self._provenance(run, error=True).items():
            response.headers[name] = value
        if run.decision is not None and exc.code not in run.decision.notes:
            run.decision.notes.append(f'refused:{exc.code}')
        run.status = exc.status
        if run.outcome == 'success':
            run.outcome = 'refused'
        await self._send_response(run.scope, run.receive, run.send, response)

    # -- local attempts --------------------------------------------------------------------------------------------------
    def _local_body(self, run: _Run, target) -> tuple[bytes, list]:
        """The body (and headers) for a local attempt: the model rewritten only when the requested name is an alias or
        a virtual model."""
        headers = list(run.scope.get('headers', []))
        body = run.body
        local = target.local_name or target.ref
        if run.peek.model == local or run.peek.model is None:
            return body, headers
        field_name = run.surface.model_from.split(':', 1)[1] if ':' in run.surface.model_from else 'model'
        if run.surface.body == 'json' and isinstance(run.peek.json, dict):
            body = inbound.rewrite_json_model(run.peek.json, field_name, local)
        elif run.surface.body == 'multipart':
            boundary = inbound.multipart_boundary(inbound.header(headers, 'content-type') or '')
            if boundary is not None:
                body = inbound.rewrite_multipart_field(body, boundary, field_name, local)
        return body, inbound.set_content_length(headers, len(body))

    async def _attempt_local(self, run: _Run, target, index: int, has_next: bool) -> _Failure | None:
        name = run.surface.name
        run.translation = 'local'
        started = time.perf_counter()
        if name == 'openai.moderations':
            run.decision.attempts.append(Attempt(ref=target.ref, outcome='refused', status=503,
                                                 error_code='classifier_unavailable'))
            return _Failure(error=BrokerError('Moderation needs a cloud classifier target', 'classifier_unavailable'),
                            final=True, request_bytes_sent=False)
        # anthropic.count_tokens and gemini.countTokens run the local route (broker/routes.py): llama-cpp models are
        # counted with the native apply-template + tokenize path, other local models get a typed 400 (R22)
        codes =(run.plan.fallback_on & FALLBACK_LOCAL) if has_next else frozenset()
        if name in inbound.LOCAL_BRIDGE:
            return await self._bridge_local(run, target, index, codes)
        body, headers = self._local_body(run, target)
        scope = dict(run.scope, headers=headers)
        scope['titanos.broker.target'] = target
        if name == 'elevenlabs.tts_stream':  # the local route has no /stream variant: same answer, streamed as it comes
            voice = run.params.get('voice_id', '')
            path = f'/v1/text-to-speech/{voice}'
            scope['path'] = path
            scope['raw_path'] = path.encode('utf-8')
        run.translation = 'local'
        attempt = LocalAttempt(run.send, fallback_codes=codes,
                               headers=self._pairs(self._provenance(run, target=target)))
        if not await self._run_local(run, scope, body, attempt):
            run.decision.attempts.append(Attempt(ref=target.ref, outcome='timeout', status=504,
                                                 error_code='deadline_exceeded', request_bytes_sent=False))
            return _Failure(error=_deadline_error(), final=True, request_bytes_sent=False)
        record = Attempt(ref=target.ref, outcome='fallback' if attempt.fallback else
                         ('success' if (attempt.status or 500) < 400 else 'local_error'), status=attempt.status,
                         error_code=attempt.error_code, ms_to_headers=attempt.ms_to_headers,
                         ms_to_first_byte=attempt.ms_to_first_byte, request_bytes_sent=False)
        run.decision.attempts.append(record)
        if attempt.fallback:
            return _Failure(trigger=attempt.fallback, request_bytes_sent=False, local=attempt)
        if not attempt.committed and attempt.start is not None:
            await attempt.flush()
        run.committed = True
        run.decision.committed_at_attempt = index
        run.served = target
        run.status = attempt.status
        run.ttfb_ms = attempt.ms_to_first_byte or (time.perf_counter() - started) * 1000.0
        run.outcome = 'success' if (attempt.status or 500) < 400 else 'local_error'
        return None

    async def _run_local(self, run: _Run, scope: dict, body: bytes, attempt: LocalAttempt) -> bool:
        """Run the inner route. Until the attempt commits (forwards its response start) it is bounded by the request
        deadline: on expiry the route is cancelled, the attempt closed and False returned (504 deadline_exceeded).
        After commit the route runs to its end (its own deadline and the client's disconnect apply)."""
        task = asyncio.ensure_future(self.app(scope, _replay(body, run.receive), attempt))
        try:
            while True:
                left = self._left(run)
                if attempt.committed or left is None:
                    await task
                    return True
                if left <= 0:
                    break
                done, _ = await asyncio.wait({task}, timeout=min(left, 1.0))
                if task in done:
                    task.result()
                    return True
        except BaseException:
            if not task.done():
                task.cancel()
            raise
        if attempt.committed:  # committed at the very edge of the deadline: the answer goes on
            await task
            return True
        attempt.closed = True
        task.cancel()
        try:
            await task
        except BaseException:
            pass
        return False

    async def _bridge_local(self, run: _Run, target, index: int, codes: frozenset) -> _Failure | None:
        """L* cells: the inner OpenAI route serves a Gemini or ElevenLabs speech-to-text request for a local model."""
        notes: list = []
        local = target.local_name or target.ref
        try:
            path, body, content_type = inbound.bridge_request(run.surface, run.peek, local_name=local, notes=notes)
        except BrokerError as exc:
            return _Failure(error=exc, final=True, request_bytes_sent=False)
        run.translation = f'{run.protocol}->openai'
        for note in notes:
            if note not in run.notes:
                run.notes.append(note)
        headers = [(k, v) for k, v in run.scope.get('headers', []) if k.lower() not in (b'content-type',
                                                                                         b'content-length')]
        headers += [(b'content-type', content_type.encode('latin-1')),
                    (b'content-length', str(len(body)).encode('latin-1'))]
        scope = dict(run.scope, path=path, raw_path=path.encode('latin-1'), method='POST', headers=headers,
                     query_string=b'')
        scope['titanos.broker.target'] = target
        queue: asyncio.Queue = asyncio.Queue()

        async def collect(message):
            await queue.put(message)

        async def runner():
            try:
                await self.app(scope, _replay(body, run.receive), collect)
            finally:
                await queue.put(None)
        task = asyncio.create_task(runner())
        started = time.perf_counter()

        async def get_before_deadline():
            left = self._left(run)
            if left is None:
                return await queue.get()
            if left <= 0:
                raise asyncio.TimeoutError
            return await asyncio.wait_for(queue.get(), timeout=left)

        def expired() -> _Failure:
            run.decision.attempts.append(Attempt(ref=target.ref, outcome='timeout', status=504,
                                                 error_code='deadline_exceeded', request_bytes_sent=False))
            return _Failure(error=_deadline_error(), final=True, request_bytes_sent=False)
        try:
            try:
                start = await get_before_deadline()
            except asyncio.TimeoutError:
                return expired()
            if start is None:
                await task
                return _Failure(error=BrokerError('The local route sent no response', 'upstream_bad_response',
                                                  origin='local'), final=True)
            status = int(start.get('status', 200))
            streaming = run.peek.stream and status < 400
            if not streaming:
                chunks = bytearray()
                while True:
                    try:
                        message = await get_before_deadline()
                    except asyncio.TimeoutError:
                        return expired()
                    if message is None:
                        break
                    if message.get('type') == 'http.response.body':
                        chunks.extend(message.get('body', b''))
                await task
                code = _error_code(bytes(chunks)) if status >= 400 else None
                run.decision.attempts.append(Attempt(ref=target.ref, outcome='fallback' if code in codes else
                                                     ('success' if status < 400 else 'local_error'), status=status,
                                                     error_code=code, request_bytes_sent=False))
                if code in codes:
                    held = LocalAttempt(run.send, fallback_codes=codes,
                                        headers=self._pairs(self._provenance(run, target=target)))
                    held.start, held.buffer, held.status, held.done = start, chunks, status, True
                    return _Failure(trigger=code, request_bytes_sent=False, local=held)
                out_status, out_type, out_body = inbound.bridge_response(run.surface, status, bytes(chunks),
                                                                         requested=run.peek.model or local,
                                                                         peek_=run.peek)
                headers = self._provenance(run, target=target)
                await run.send({'type': 'http.response.start', 'status': out_status,
                                'headers': [(b'content-type', out_type.encode('latin-1'))] + self._pairs(headers)})
                await run.send({'type': 'http.response.body', 'body': out_body, 'more_body': False})
                run.committed, run.served, run.status = True, target, out_status
                run.decision.committed_at_attempt = index
                run.outcome = 'success' if out_status < 400 else 'local_error'
                run.ttfb_ms = (time.perf_counter() - started) * 1000.0
                return None

            async def body_chunks():
                while True:
                    message = await queue.get()
                    if message is None:
                        return
                    if message.get('type') == 'http.response.body' and message.get('body'):
                        yield message['body']
            stream = inbound.bridge_stream(run.surface, body_chunks(), requested=run.peek.model or local)
            produced = inbound.Produced(200, 'text/event-stream', stream, True, lambda: None,
                                        lambda: inbound.aclose(stream), sse_protocol='gemini')
            run.decision.attempts.append(Attempt(ref=target.ref, outcome='success', status=200,
                                                 request_bytes_sent=False))
            run.decision.committed_at_attempt = index
            await self._commit(run, target, produced, keepalive=False)
            if run.outcome == 'success':
                await task  # the inner route finished its stream; after a disconnect it is cancelled below
            return None
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass

    # -- cloud attempts --------------------------------------------------------------------------------------------------
    def _trigger(self, err) -> str | None:
        return _TRIGGER_BY_KIND.get(err.kind)

    async def _attempt_cloud(self, run: _Run, target, index: int, has_next: bool) -> _Failure | None:
        broker = self.broker
        snapshot = run.snapshot
        upstream = snapshot.config.upstreams[target.upstream_id]
        adapter = snapshot.adapters.get(target.upstream_id)
        from .health import breaker_for
        breaker = breaker_for(broker.breakers, target, upstream)  # per routed model on a breaker_per_model row
        started = time.perf_counter()

        def attempt(outcome, status=None, code=None, sent=False, ms_headers=None, ms_first=None):
            run.decision.attempts.append(Attempt(ref=target.ref, outcome=outcome, status=status, error_code=code,
                                                 ms_to_headers=ms_headers, ms_to_first_byte=ms_first,
                                                 request_bytes_sent=sent))

        if adapter is None:
            attempt('refused', 503, 'upstream_unconfigured')
            return _Failure(error=BrokerError('The upstream has no adapter', 'upstream_unconfigured'), final=True,
                            request_bytes_sent=False)
        if target.klass == 'router' and run.hints.request_id:
            # a request that a router sent back here (same X-Request-ID, which a router may forward while dropping
            # X-Titanos-Hop) and that would go to a router again is a loop: refused before egress (router docstring)
            from .router import ROUTER_INFLIGHT
            if run.hints.request_id in ROUTER_INFLIGHT and run.hints.request_id not in run.router_ids:
                attempt('refused', 508, 'loop_detected')
                return _Failure(error=BrokerError('This request (its X-Request-ID) is already being served through a '
                                                  'router upstream of this Titanos instance, and the router sent it '
                                                  'back to a router again', 'loop_detected', origin='policy'),
                                final=True, request_bytes_sent=False)
            ROUTER_INFLIGHT.add(run.hints.request_id)
            run.router_ids.append(run.hints.request_id)
        if adapter.native_protocol == 'anthropic' == run.surface.protocol and adapter.has(run.surface.name):
            # provider-side state named in an Anthropic passthrough body (container, custom skills, file_id sources)
            from . import anthropic_state
            try:
                anthropic_state.check(run.peek.json, principal=run.principal,
                                      allow_store=bool(run.plan.allow_store.get(target.ref)),
                                      store_scope=allows(run.principal, 'cloud:store'), handles=broker.handles,
                                      upstream_id=target.upstream_id, role=broker.settings.role)
            except BrokerError as exc:
                attempt('refused', exc.status, exc.code)
                return _Failure(error=exc, final=True, request_bytes_sent=False)
        left = self._left(run)
        if left is not None and left < MIN_ATTEMPT_SECONDS:  # before the breaker: never takes a probe slot
            attempt('refused', 504, 'deadline_exceeded')
            return _Failure(error=_deadline_error(), final=True, request_bytes_sent=False)
        permit = breaker.allow()
        if not permit:
            attempt('breaker_open', 503, 'breaker_open')
            return _Failure(trigger='breaker_open', request_bytes_sent=False,
                            error=BrokerError('The upstream is temporarily unavailable', 'breaker_open',
                                              retry_after=permit.retry_after))
        admission = run.admission
        if admission is not None and admission.target.ref != target.ref:
            try:
                admission.retarget(target)
            except BrokerError as exc:
                breaker.record('refused')
                attempt('refused', exc.status, exc.code)
                return _Failure(error=exc, final=True, request_bytes_sent=False)
        injected = admission.injected_max_tokens if admission is not None else None
        if run.surface.name not in TOKEN_OUTPUT_SURFACES:
            injected = None  # images, audio, embeddings, moderations and token counters never get an output cap
        notes: list = []
        try:
            grant = snapshot.issuer.issue('inference', plan_target=target, upstream_id=target.upstream_id,
                                          decision_id=run.decision.decision_id, request_id=run.hints.request_id,
                                          delegated=run.hints.delegated, reservation=admission,
                                          deadline=run.hints.deadline)
        except BrokerError as exc:
            breaker.record('refused')
            attempt('refused', exc.status, exc.code)
            return _Failure(error=exc, final=True, request_bytes_sent=False)
        # a split-protocol router speaks Anthropic Messages natively next to the OpenAI surfaces
        protocol_for = getattr(adapter, 'protocol_for', None)
        native_protocol = protocol_for(run.surface.name) if callable(protocol_for) else adapter.native_protocol
        passthrough = adapter.has(run.surface.name) and native_protocol == run.surface.protocol
        profile = PROFILES[upstream.profile]
        run.translation = 'passthrough' if passthrough else f'{run.protocol}->{native_protocol}'
        first_byte = float(upstream.timeouts.first_byte if upstream.timeouts else
                           getattr(broker.settings, 'broker_first_byte_timeout', 120.0))
        connect = float(upstream.timeouts.connect if upstream.timeouts else
                        getattr(broker.settings, 'broker_connect_timeout', 10.0))
        budget = first_byte + connect
        by_deadline = False
        left = self._left(run)
        if left is not None and left < budget:
            budget, by_deadline = max(0.0, left), True
        try:
            produced = await self._bounded(
                run, self._produce(run, target, adapter, profile, grant, passthrough, notes, injected), budget, by_deadline)
        except UpstreamFailure as failure:
            if not run.surface.handle_bound:
                failure = model_failure(failure, target.ref if run.peek.model in (target.ref, target.upstream_model)
                                        else None, upstream_model=target.upstream_model)
            err = failure.upstream
            breaker.record(err.kind, retry_after=err.retry_after)
            attempt(err.kind, failure.status, failure.code, err.request_bytes_sent)
            if failure.code == 'model_not_found' and err.status in (404, 410):
                note = f'upstream_model_unavailable:{err.status}'
                if note not in run.notes:
                    run.notes.append(note)
            if failure.passthrough:
                return _Failure(passthrough=failure, final=True)
            return _Failure(trigger=self._trigger(err), request_bytes_sent=err.request_bytes_sent, error=failure)
        except asyncio.TimeoutError:
            if by_deadline:
                # the caller's (or parent's) deadline ran out, not the upstream's first-byte budget: a neutral breaker
                # outcome (the half-open slot is released, nothing is counted) and 504 deadline_exceeded
                breaker.record('cancelled')
                attempt('timeout', 504, 'deadline_exceeded', True)
                return _Failure(error=_deadline_error(), final=True, request_bytes_sent=True)
            breaker.record('timeout')
            attempt('timeout', 504, 'upstream_timeout', True)
            return _Failure(trigger='upstream_timeout', request_bytes_sent=True,
                            error=BrokerError('The upstream did not answer in time', 'upstream_timeout',
                                              origin='upstream'))
        except httpx.HTTPError as exc:
            err = network_error(exc)
            breaker.record(err.kind)
            attempt(err.kind, err.status, 'upstream_unavailable', err.request_bytes_sent)
            return _Failure(trigger=self._trigger(err), request_bytes_sent=err.request_bytes_sent,
                            error=UpstreamFailure(err))
        except BrokerError as exc:
            sent = exc.code not in ('concurrency_exceeded', 'egress_refused', 'role_forbids', 'deadline_exceeded',
                                    'credential_unavailable', 'vault_sealed', 'upstream_unconfigured',
                                    'broker_capability_unavailable', 'invalid_request', 'model_required',
                                    'model_not_found')
            if exc.code == 'concurrency_exceeded':
                breaker.record('refused')
                attempt('refused', exc.status, exc.code, False)
                return _Failure(trigger='upstream_rate_limited', request_bytes_sent=False, error=exc)
            breaker.record('client_error' if exc.status < 500 else ('server_error' if exc.code ==
                                                                    'upstream_bad_response' else 'refused'))
            attempt('refused' if not sent else 'error', exc.status, exc.code, sent)
            if exc.code == 'upstream_bad_response':
                return _Failure(trigger='upstream_unavailable', request_bytes_sent=True, error=exc)
            return _Failure(error=exc, final=True, request_bytes_sent=sent)
        except BaseException:  # cancelled (or an unexpected error): the half-open probe slot must not stay taken
            breaker.record('cancelled')
            raise
        ms = (time.perf_counter() - started) * 1000.0
        breaker.record('success', headers=produced.headers)
        broker.latency.observe(target.ref, ms)
        selected = (produced.headers or {}).get('x-model-router-selected-model')
        if selected:  # the backend a router upstream chose: kept in the decision record
            notes.append('router_selected:' + _latin1(str(selected))[:128])
        attempt('success', produced.status, None, True, ms, ms)
        run.ttfb_ms = ms
        run.upstream_request_id = produced.upstream_request_id
        run.injected = injected is not None or 'injected:max_tokens' in notes  # budgets, and Anthropic translations
        for note in notes:
            if note not in run.notes:
                run.notes.append(note)
        if injected is not None and 'injected:max_tokens' not in run.notes:
            run.notes.append('injected:max_tokens')
        run.decision.committed_at_attempt = index
        await self._commit(run, target, produced, keepalive=not passthrough and produced.stream
                           and produced.sse_protocol in KEEPALIVE_PROTOCOLS)
        if run.surface.name == 'openai.responses' and run.observed_response_id and \
                run.plan.allow_store.get(target.ref) and (run.peek.json or {}).get('store', True) is not False:
            try:
                broker.handles.bind('responses', run.observed_response_id, target.upstream_id, upstream.credential,
                                    run.principal.key_id, model=target.ref)
            except (OSError, ValueError):
                log.warning('broker: could not record the response handle')
        return None

    def _passthrough_document(self, run: _Run, target, injected: int | None, notes: list) -> dict | None:
        document = run.peek.json
        if not isinstance(document, dict):
            return None
        out = dict(document)
        if (injected is not None and run.surface.name in TOKEN_OUTPUT_SURFACES
                and inbound.output_cap(run.surface.name, out) is None):
            out = inbound.inject_output_cap(run.surface.name, out, injected)
        mode = run.plan.reasoning_mode
        if mode in REASONING_MODES:
            profile = PROFILES[run.snapshot.config.upstreams[target.upstream_id].profile]
            features = profile.features
            if profile.adapter == 'openai_compat' and not accepts_reasoning_effort(profile, target.upstream_model):
                features = features - {'reasoning'}  # the row's reasoning_effort_globs exclude this model
            out = inject_reasoning(run.surface.name, out, mode, features, notes, model=target.upstream_model)
        return out

    async def _produce(self, run: _Run, target, adapter, profile, grant, passthrough: bool, notes: list,
                       injected: int | None) -> inbound.Produced:
        egress = self.broker.egress
        inject = run.plan.inject.get(target.ref)
        if not passthrough:
            return await inbound.translated_call(adapter, grant, egress, run.surface, run.peek,
                                                 requested=run.peek.model or target.ref, query=run.query,
                                                 params=run.params, notes=notes, inject=inject,
                                                 voice_maps=run.snapshot.config.voice_maps,
                                                 injected_max_tokens=injected,
                                                 reasoning_mode=run.plan.reasoning_mode
                                                 if run.plan.reasoning_mode in REASONING_MODES else None)
        document = self._passthrough_document(run, target, injected, notes)
        rehost = False
        if run.surface.name in IMAGE_SURFACES:
            b64_only = image_b64_only(profile, target.upstream_model)
            # a provider that answers its own image URLs (xAI, Together: extras image_url_native) serves a url or
            # default request itself; nothing is re-hosted (and a JPEG-only provider is not refused)
            rehost = self._wants_url(run, document, b64_only) and not (
                profile.extras.get('image_url_native') and not b64_only)
            if rehost:
                self._check_rehost(profile)
                self._check_rehost_request(run, document)
            if isinstance(document, dict):
                if b64_only:  # never send response_format to a model that refuses it (it answers base64 anyway)
                    if 'response_format' in document:
                        document = {k: v for k, v in document.items() if k != 'response_format'}
                        if 'dropped:response_format' not in notes:
                            notes.append('dropped:response_format')
                elif rehost:
                    document = dict(document, response_format='b64_json')
        allow = profile_inbound_headers(profile)  # a router's session headers, never credentials
        raw = RawRequest(method=run.scope.get('method', 'POST'), path_params=dict(run.params), query=dict(run.query),
                         headers=filter_inbound_headers(run.scope.get('headers', []), extra=allow),
                         json=document, raw=None if document is not None else run.body, form=run.peek.form,
                         header_allow=allow)
        result = await adapter.passthrough(grant, egress, run.surface, raw, notes=notes, inject=inject,
                                           allow_store=bool(run.plan.allow_store.get(target.ref)))
        content_type = result.content_type or 'application/octet-stream'
        stream = any(kind in content_type for kind in ('event-stream', 'ndjson')) or content_type.startswith('audio/')
        sse_protocol = None
        if 'event-stream' in content_type or 'ndjson' in content_type:
            sse_protocol = 'responses' if run.surface.name == 'openai.responses' else run.surface.protocol
        if rehost:
            data = b''.join([chunk async for chunk in result.body])
            await result.aclose()
            data = self._rehost(run, data)

            async def once():
                yield data
            usage = await result.usage

            async def nothing():
                return None
            return inbound.Produced(result.status, 'application/json', once(), False, lambda: usage, nothing,
                                    result.upstream_request_id, dict(result.headers))
        return inbound.Produced(result.status, content_type, result.body, stream, lambda: result.usage, result.aclose,
                                result.upstream_request_id, dict(result.headers), sse_protocol)

    # -- PNG url re-hosting ----------------------------------------------------------------------------------------------
    @staticmethod
    def _wants_url(run: _Run, document: dict | None, b64_only: bool) -> bool:
        """Whether the client gets re-hosted URLs: it asked for response_format url (images.generations; images.edits on a
        b64-only model, whose base64 answer is then re-hosted), or it named no format on images.generations of a model
        whose own default answer is a URL (OpenAI's documented default before every OpenAI image model became b64-only).
        A b64-only model asked for b64_json or nothing answers its base64 unchanged."""
        if isinstance(document, dict):
            fmt = document.get('response_format')
        else:
            fmt = next((v for k, v in run.peek.form or [] if k == 'response_format'), None)
        generations = run.surface.name == 'openai.images.generations'
        if fmt == 'url':
            return generations or b64_only
        return fmt is None and generations and not b64_only

    @staticmethod
    def _check_rehost_request(run: _Run, document: dict | None) -> None:
        """A url (re-hosted) image request that cannot be re-hosted is refused before egress, so the image is never paid
        for and then lost: re-hosting stores PNG only (output_format jpeg or webp), and a streamed image answer
        (stream=true: image_generation.* events) is not a JSON document to re-host."""
        if isinstance(document, dict):
            fmt, stream = document.get('output_format'), document.get('stream')
        else:
            form = run.peek.form or []
            fmt = next((v for k, v in form if k == 'output_format'), None)
            stream = next((v for k, v in form if k == 'stream'), None)
            stream = isinstance(stream, str) and stream.strip().lower() in ('true', '1')
        if fmt not in (None, 'png'):
            raise BrokerError(f'response_format url is re-hosted as PNG only; output_format {fmt!r} needs '
                              'response_format b64_json (or no response_format)', 'broker_capability_unavailable')
        if stream:
            raise BrokerError('response_format url cannot be combined with stream=true (the streamed events carry '
                              'base64 images); omit response_format or ask for b64_json', 'broker_capability_unavailable')

    def _check_rehost(self, profile) -> None:
        if self.broker.artifact_writer is None:
            raise BrokerError('Image url re-hosting is not available on this server', 'broker_capability_unavailable')
        if profile.image_formats and 'png' not in profile.image_formats:
            raise BrokerError(f'{profile.id} returns {", ".join(sorted(profile.image_formats))} images; url re-hosting '
                              'needs PNG (ask for b64_json)', 'broker_capability_unavailable')

    def _rehost(self, run: _Run, data: bytes) -> bytes:
        try:
            document = json.loads(data.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise BrokerError('The upstream image answer is not JSON', 'upstream_bad_response', origin='upstream') \
                from None
        items = document.get('data') if isinstance(document, dict) else None
        if not isinstance(items, list):
            raise BrokerError('The upstream image answer has no data', 'upstream_bad_response', origin='upstream')
        images = []  # every item is decoded and checked before any file is written (no orphans on a later failure)
        for item in items:
            raw = item.get('b64_json') if isinstance(item, dict) else None
            if not isinstance(raw, str):
                raise BrokerError('The upstream answered without image bytes', 'broker_capability_unavailable')
            try:
                image = base64.b64decode(raw)
            except (binascii.Error, ValueError):
                raise BrokerError('The upstream image is not valid base64', 'upstream_bad_response',
                                  origin='upstream') from None
            if not image.startswith(PNG_MAGIC):
                raise BrokerError('The upstream image is not a PNG; url re-hosting needs PNG',
                                  'broker_capability_unavailable')
            images.append((item, image))
        out = []
        written = []
        # an absolute url, built from this request's base URL exactly as the local image route and the job artifacts do
        # (str(request.base_url) + 'v1/files/...'), so SDK clients can fetch it directly
        base_url = str(Request(run.scope).base_url)
        try:
            for item, image in images:
                name = uuid.uuid4().hex + '.png'
                path = self.broker.artifact_writer(name)
                written.append(path)
                path.write_bytes(image)
                self.broker.ownership.bind('artifact', name, run.principal)  # fail closed: no url without an owner
                entry = {k: v for k, v in item.items() if k != 'b64_json'}
                entry['url'] = base_url + 'v1/files/' + name
                out.append(entry)
        except (OSError, ValueError) as exc:
            for path in written:
                try:
                    path.unlink()
                except OSError:
                    pass
            log.warning('broker: could not store or bind a re-hosted image (%s)', type(exc).__name__)
            raise BrokerError('The re-hosted image could not be stored', 'broker_capability_unavailable',
                              status=503) from None
        document['data'] = out
        return inbound.json_bytes(document)

    # -- commit and streaming --------------------------------------------------------------------------------------------
    async def _commit(self, run: _Run, target, produced: inbound.Produced, *, keepalive: bool) -> None:
        headers = {'content-type': produced.content_type}
        for name, value in (produced.headers or {}).items():
            lowered = name.lower()
            if lowered in ('content-type', 'content-length', 'transfer-encoding'):
                continue
            headers[lowered] = value
        if produced.stream:
            headers['cache-control'] = 'no-cache'
            headers['x-accel-buffering'] = 'no'
        # provenance replaces same-named upstream headers (the upstream's own x-request-id travels as
        # X-Titanos-Upstream-Request-ID), so the client never sees two X-Request-ID values
        headers.update({k.lower(): v for k, v in self._provenance(run, target=target).items()})
        await run.send({'type': 'http.response.start', 'status': produced.status, 'headers': self._pairs(headers)})
        run.committed = True
        run.served = target
        run.status = produced.status
        observer = None
        if run.surface.name == 'openai.responses':
            observer = _ResponsesObserver(run, sse='event-stream' in produced.content_type)
        protocol = produced.sse_protocol or ('elevenlabs' if produced.content_type.startswith('audio/') else None)
        outcome = await self._pump(run, produced, keepalive=keepalive, protocol=protocol, observer=observer)
        if observer is not None:
            observer.finish()
        run.outcome = outcome
        if outcome == 'cancelled':
            run.status = 499
        usage = produced.usage()
        if inspect.isawaitable(usage):
            try:
                usage = await asyncio.wait_for(asyncio.shield(usage), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                usage = None
        run.usage = usage if isinstance(usage, Usage) else None
        if outcome == 'deadline_exceeded':
            if 'stream:deadline_exceeded' not in run.notes:
                run.notes.append('stream:deadline_exceeded')
            if protocol in (None, 'elevenlabs', 'binary'):
                raise StreamAborted('the request deadline expired after commit')
        if outcome == 'stream_interrupted' and protocol in ('elevenlabs', 'binary'):
            raise StreamAborted('upstream stream interrupted after commit')

    async def _wait_disconnect(self, receive) -> None:
        while True:
            message = await receive()
            if message.get('type') == 'http.disconnect':
                return

    async def _pump(self, run: _Run, produced: inbound.Produced, *, keepalive: bool, protocol: str | None,
                    observer=None) -> str:
        queue: asyncio.Queue = asyncio.Queue(self.queue_size)
        end = object()
        uid = run.served.upstream_id if run.served is not None else None

        async def reader():
            try:
                async for chunk in produced.body:
                    if chunk:
                        await queue.put(chunk)
                await queue.put(end)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # a mid-stream upstream failure: the terminal frame follows
                await queue.put(exc)

        reader_task = asyncio.create_task(reader())
        watcher = asyncio.create_task(self._wait_disconnect(run.receive))
        outcome = 'success'
        try:
            while True:
                left = self._left(run)
                if left is not None and left <= 0:
                    # R15: the overall request deadline bounds a committed stream too
                    outcome = 'deadline_exceeded'
                    await self._deadline_terminal(run, protocol)
                    break
                timeout = self.keepalive_seconds if keepalive else None
                if left is not None:
                    timeout = left if timeout is None else min(timeout, left)
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({getter, watcher}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if getter not in done:
                    getter.cancel()
                    try:
                        await getter
                    except BaseException:
                        pass
                    if watcher in done:
                        outcome = 'cancelled'
                        break
                    left = self._left(run)
                    if keepalive and (left is None or left > 0):
                        await run.send({'type': 'http.response.body', 'body': KEEPALIVE, 'more_body': True})
                    continue
                item = getter.result()
                if item is end:
                    await run.send({'type': 'http.response.body', 'body': b'', 'more_body': False})
                    break
                if isinstance(item, BaseException):
                    outcome = 'stream_interrupted'
                    await self._terminal(run, item, protocol)
                    break
                if observer is not None:
                    observer(item)
                if uid is not None:
                    self.broker.metrics.egress_bytes(uid, len(item))
                await run.send({'type': 'http.response.body', 'body': item, 'more_body': True})
        finally:
            for task in (reader_task, watcher):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except BaseException:
                        pass
            try:
                await produced.aclose()
            except Exception:
                log.debug('broker: closing the upstream answer failed', exc_info=True)
        return outcome

    async def _deadline_terminal(self, run: _Run, protocol: str | None) -> None:
        """R15: end a committed stream whose request deadline expired with the protocol's terminal frame (code
        deadline_exceeded). No breaker outcome is recorded. Binary audio and plain JSON bodies get no frame: _commit
        raises StreamAborted so the transfer ends without a terminating chunk."""
        if protocol in (None, 'elevenlabs', 'binary'):
            return
        frame = stream_terminal(protocol, 'deadline_exceeded', 'The request deadline expired',
                                sequence_number=run.sequence_number + 1,
                                response={'id': run.observed_response_id} if run.observed_response_id else None)
        await run.send({'type': 'http.response.body', 'body': frame, 'more_body': False})

    async def _terminal(self, run: _Run, exc: BaseException, protocol: str | None) -> None:
        code = 'upstream_unavailable'
        if isinstance(exc, UpstreamFailure) and exc.upstream.kind == 'timeout':
            code = 'upstream_timeout'
        elif isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
            code = 'upstream_timeout'
        if run.served is not None and run.served.klass != 'local':
            upstream = run.snapshot.config.upstreams.get(run.served.upstream_id)
            if upstream is not None:
                from .health import breaker_for
                breaker_for(self.broker.breakers, run.served, upstream).record(
                    'timeout' if code == 'upstream_timeout' else 'network')
        message = 'The upstream stream was interrupted'
        if protocol in (None, 'elevenlabs', 'binary'):
            return  # binary audio: abort without a terminating chunk (StreamAborted after cleanup)
        frame = stream_terminal(protocol, code, message, sequence_number=run.sequence_number + 1,
                                response={'id': run.observed_response_id} if run.observed_response_id else None)
        await run.send({'type': 'http.response.body', 'body': frame, 'more_body': False})

    # -- handle-bound Responses routes -----------------------------------------------------------------------------------
    def _handle_target(self, run: _Run, binding):
        """The owning upstream's target for a handle-bound route, after the same principal and policy gates as the main
        path (the handle names its target explicitly, so it counts as named and delegable):
        visibility (route scope, target class, upstream constraint, cloud or cloud:<upstream>) -> 404 response_not_found;
        the 1-hop rule and a local_only location -> 400 location_unsatisfiable; the role and the effective cloud policy
        (off -> 403 cloud_disabled); readiness -> 404 response_not_found. The key's model globs apply to the model the
        handle recorded (R17); a binding read from the version-1 handles.json carries no model and skips that check."""
        snapshot = run.snapshot
        principal = run.principal
        upstream_id = binding.upstream_id
        upstream = snapshot.config.upstreams.get(upstream_id)
        target = snapshot.catalog.qualified_target('responses', upstream_id) if upstream is not None else None
        not_found = BrokerError('Response not found', 'response_not_found', origin='policy')
        if target is None:
            raise not_found
        constraints = getattr(principal, 'constraints', None)
        if (constraints is None or not allows(principal, run.surface.scope)
                or target.klass not in constraints.target_classes
                or not any(fnmatchcase(upstream_id, pattern) for pattern in constraints.upstreams)
                or not allows(principal, 'cloud:' + upstream_id)
                or (getattr(binding, 'model', None) is not None
                    and not any(fnmatchcase(binding.model, pattern) for pattern in constraints.models))):
            raise not_found
        if run.hints.hop:
            raise BrokerError('No target satisfies the requested location (requests from another broker hop stay '
                              'local)', 'location_unsatisfiable', origin='policy')
        switches = snapshot.switches
        eff = effective_policy(self.broker.settings, snapshot.config, switches, principal, run.hints, run.surface)
        if eff.role == 'managed' and not eff.delegated:
            if getattr(switches, 'delegation', False):
                raise BrokerError('Cloud use in managed role needs parent delegation', 'delegation_required',
                                  origin='policy')
            raise BrokerError('This role does not allow cloud targets', 'role_forbids', origin='policy')
        if eff.role != 'managed' and (not getattr(switches, 'cloud_egress', True) or (
                target.klass == 'router' and not getattr(switches, 'router_upstreams', True))):
            raise BrokerError('This role does not allow cloud targets', 'role_forbids', origin='policy')
        if eff.location == 'local_only':
            raise BrokerError('No target satisfies the requested location', 'location_unsatisfiable', origin='policy')
        if eff.cloud_policy == 'off':
            raise BrokerError('Cloud targets are disabled by policy', 'cloud_disabled', origin='policy')
        if snapshot.catalog.readiness(upstream_id) is not None:
            raise not_found
        if run.decision is not None:
            run.decision.effective = {'cloud_policy': eff.cloud_policy, 'location': eff.location}
        return target

    async def _handle_route(self, run: _Run):
        broker = self.broker
        snapshot = run.snapshot
        response_id = run.params.get('response_id', '')
        ctx = RouteContext(surface=run.surface, requested_model=response_id, principal=run.principal, hints=run.hints,
                           body=run.peek.summary)
        run.ctx = ctx
        decision = snapshot.router.new_decision(ctx, inbound_protocol=run.protocol)
        run.decision = decision
        try:
            try:
                binding = broker.handles.owned(run.principal, 'responses', response_id)
                target = self._handle_target(run, binding)
                adapter = snapshot.adapters.get(binding.upstream_id)
                if adapter is None:
                    raise BrokerError('Response not found', 'response_not_found', origin='policy')
                grant = snapshot.issuer.issue('inference', upstream_id=binding.upstream_id,
                                              decision_id=decision.decision_id, request_id=run.hints.request_id,
                                              delegated=run.hints.delegated, deadline=run.hints.deadline)
                run.translation = 'passthrough'
                raw = RawRequest(method=run.scope.get('method', 'GET'), path_params={'response_id': response_id},
                                 query=dict(run.query), headers=filter_inbound_headers(run.scope.get('headers', [])))
                try:
                    result = await adapter.passthrough(grant, broker.egress, run.surface, raw)
                except UpstreamFailure as failure:
                    if failure.passthrough:
                        run.outcome, run.status = 'client_error', failure.status
                        response = passthrough_4xx(failure.upstream, 'openai')
                        for name, value in self._provenance(run, error=True).items():
                            response.headers[name] = value
                        await self._send_response(run.scope, run.receive, run.send, response)
                        return
                    raise
            except BrokerError as exc:
                run.outcome = 'refused'
                await self._refuse(run, exc)
                return
            sse = 'event-stream' in (result.content_type or '')
            produced = inbound.Produced(result.status, result.content_type or 'application/json', result.body, sse,
                                        lambda: result.usage, result.aclose, result.upstream_request_id,
                                        dict(result.headers), 'responses' if sse else None)
            run.upstream_request_id = result.upstream_request_id
            decision.attempts.append(Attempt(ref=target.ref if target else binding.upstream_id, outcome='success',
                                             status=result.status, request_bytes_sent=True))
            decision.committed_at_attempt = 0
            await self._commit(run, target, produced, keepalive=False)
            if run.surface.name == 'openai.responses.delete' and 200 <= result.status < 300:
                broker.handles.forget('responses', response_id)
        finally:
            await self._finish(run)

    # -- R32 native passthrough ----------------------------------------------------------------------------------------
    def _native_match(self, scope, method: str, *, catalog: bool = False):
        """The NativeMatch of an HTTP or WebSocket request, ('error', BrokerError) for a native request refused before
        anything else is known (unknown upstream, path not allowlisted, ambiguous bare path), or None (not a native
        request: the inner app answers as before). Nothing is read here; the broker must be configured."""
        path = scope.get('path', '') or ''
        prefixed = native_lane.split_prefix(path)
        if prefixed is None and not native_lane.bare_rows(method, path):
            upgrade = (inbound.header(scope.get('headers', []), 'upgrade') or '').lower()
            if method.upper() == 'WS' or upgrade != 'websocket' or not native_lane.bare_rows('WS', path):
                return None
            method = 'WS'  # a WebSocket upgrade that reached the HTTP lane: answered with the typed WebSocket refusal
        try:
            if not self.broker.configured:
                return None
        except Exception:
            log.exception('broker: activity check failed; passing the request through')
            return None
        snapshot = self.broker.snapshot()
        principal = scope.get('titanos.principal') or self.broker.default_principal()
        if prefixed is not None:
            upstream_id, rel = prefixed
            upstream = snapshot.config.upstreams.get(upstream_id)
            if upstream is None or not rel:
                return ('error', BrokerError('Not found', 'not_found', origin='policy'), None)
            profile_id = upstream.profile
            found = native_lane.match_route(profile_id, method, rel)
            if found is None and method.upper() != 'WS':
                found = native_lane.match_route(profile_id, 'WS', rel)  # answered with a typed WebSocket-only refusal
            if found is None:
                methods = native_lane.routes_for(profile_id, rel)
                detail = (f' (allowlisted methods: {", ".join(methods)})' if methods else '')
                return ('error', BrokerError(f'{method.upper()} {rel} is not in the {profile_id} native passthrough '
                                             f'allowlist{detail}', 'broker_capability_unavailable'), profile_id)
            route, params = found
            raw = scope.get('raw_path')
            raw_rel = rel
            if isinstance(raw, (bytes, bytearray)):
                raw_text = bytes(raw).decode('latin-1').split('?', 1)[0]
                parts = native_lane.split_prefix(raw_text)
                if parts is None or parts[0] != upstream_id:
                    # a percent-encoded prefix: the decoded rel would be sent, which may differ from the row matched
                    return ('error', BrokerError('The /native/<upstream>/ prefix must not be percent-encoded',
                                                 'invalid_request'), profile_id)
                raw_rel = parts[1]
            return native_lane.NativeMatch(upstream_id, profile_id, route, raw_rel, params, bare=False, explicit=True)
        listed = catalog or (scopes_listed(method, path) is not None)
        header = native_lane.upstream_header(scope.get('headers', []))
        try:
            found = native_lane.resolve_bare(snapshot, principal, method, path, header,
                                             listed=listed and header is None, headers=scope.get('headers', []),
                                             handles=self.broker.handles)
        except BrokerError as exc:
            return ('error', exc, None)
        if found is not None and found.bare:
            prefix = PROFILES[found.profile_id].native.bare_prefix or ''
            raw = scope.get('raw_path')
            if isinstance(raw, (bytes, bytearray)):
                raw_text = bytes(raw).decode('latin-1').split('?', 1)[0]
                if not raw_text.startswith(prefix + '/') and not (prefix == '' and raw_text.startswith('/')):
                    return ('error', BrokerError('The provider path prefix must not be percent-encoded',
                                                 'invalid_request'), found.profile_id)
                found = replace(found, rel=raw_text[len(prefix):])
        return found

    def _native_run(self, scope, receive, send, principal, hints, match, protocol: str, method: str,
                    body: bytes = b'') -> _Run:
        from .limits import estimate_input_tokens
        from .types import BodySummary
        surface = native_lane.pseudo_surface(method, protocol)
        summary = BodySummary(bytes=len(body), approx_input_tokens=estimate_input_tokens(len(body)))
        peek_ = inbound.Peek(model=None, stream=False, summary=summary)
        run = _Run(scope=scope, receive=receive, send=send, surface=surface, params=dict(match.params) if match else {},
                   query={}, principal=principal, hints=hints, peek=peek_, body=body, protocol=protocol,
                   snapshot=self.broker.snapshot(), started=time.perf_counter())
        requested = f'{match.route.method} {match.route.path}' if match is not None else ''
        ctx = RouteContext(surface=surface, requested_model=requested, principal=principal, hints=hints, body=summary)
        run.ctx = ctx
        run.decision = run.snapshot.router.new_decision(ctx, inbound_protocol=protocol)
        run.translation = 'native'
        if match is not None:
            run.notes.append(f'native:{match.route.family or "endpoint"}')
            if match.bare:
                run.notes.append('native:bare_path')
        return run

    def _native_audit(self, run: _Run, match, event: str) -> None:
        detail = {'method': match.route.method, 'route': match.route.path, 'family': match.route.family,
                  'request_id': run.hints.request_id, 'event': event}
        self.broker.audit.event('native.call', str(getattr(run.principal, 'id', 'anonymous')), match.upstream_id, detail)

    async def _native_http(self, scope, receive, send, match):
        broker = self.broker
        method = str(scope.get('method', 'GET')).upper()
        principal = scope.get('titanos.principal') or broker.default_principal()
        error = None
        if isinstance(match, tuple):
            _, error, profile_id = match
            match = None
            protocol = native_lane.protocol_of(profile_id) if profile_id else protocol_for(scope.get('path', ''))
        else:
            protocol = native_lane.protocol_of(match.profile_id)
        body = await _read_body(receive)
        if body is None:
            return
        if match is not None and match.by_model:
            decided = await self._native_by_model(scope, principal, match, body)
            if decided is None:  # not a cloud model of a candidate: the Titanos route of the same shape, as before
                denied = self._listed_scope_denied(scope, method)
                if denied is not None:
                    return await self._send_response(scope, receive, send,
                                                     render_error(protocol_for(scope.get('path', '')), denied))
                return await self.app(scope, _replay(body, receive), send)
            match = decided
        self.intercepted += 1
        try:
            hints = request_hints(scope.get('headers', []), broker.settings, principal)
        except BrokerError as exc:
            return await self._send_response(scope, receive, send, render_error(protocol, exc))
        run = self._native_run(scope, receive, send, principal, hints, match, protocol, method, body)
        token = broker_context.bind(principal, hints)
        try:
            if error is not None:
                await self._refuse(run, error)
                return
            await self._native_execute(run, match, method, body)
        finally:
            broker_context.reset(token)
            await self._finish(run)

    async def _native_by_model(self, scope, principal, match, body: bytes):
        """The decided NativeMatch of a model_over_listed row (native.choose_by_model) or None (the Titanos route). The
        live catalogs the body's model could resolve through are fetched first when stale."""
        broker = self.broker
        content_type = inbound.header(scope.get('headers', []), 'content-type') or ''
        model = native_lane.body_model(body, content_type)
        if not isinstance(model, str) or not model:
            return None
        try:
            hints = request_hints(scope.get('headers', []), broker.settings, principal)
            deadline, hop = hints.deadline, hints.hop
        except BrokerError:
            deadline, hop = None, False
        try:
            # a hop request is refused cloud placement (native.authorize): it never makes this server fetch a catalog
            pending = [] if hop else broker.snapshot().catalog.live_pending(
                model, native_lane.protocol_of(match.profile_id), principal)
            if pending:
                await broker.ensure_catalogs(principal, pending, deadline=deadline)
        except Exception:
            log.warning('broker: on-demand live catalogs failed', exc_info=True)
        return native_lane.choose_by_model(broker.snapshot(), principal, match, model,
                                           local_names=_local_names(broker))

    async def _native_execute(self, run: _Run, match, method: str, body: bytes) -> None:
        broker = self.broker
        snapshot = run.snapshot
        principal = run.principal
        route = match.route
        decision = run.decision
        try:
            if route.method == 'WS':
                upgrade = (inbound.header(run.scope.get('headers', []), 'upgrade') or '').lower()
                if upgrade == 'websocket':
                    library = native_lane.websocket_support()
                    raise BrokerError('This server cannot accept WebSocket upgrades: uvicorn needs the websockets or wsproto '
                                      'package (pip install websockets), then restart titanos serve'
                                      if library is None else 'The WebSocket upgrade did not reach this server as a '
                                      'WebSocket; check proxies between the client and Titanos',
                                      'broker_capability_unavailable')
                raise BrokerError(f'{route.path} is a WebSocket endpoint; connect with a WebSocket client',
                                  'broker_capability_unavailable')
            native_lane.check_rel(match.rel, match.params)
            upload_token, _, query_string = native_lane.split_upload_token(run.scope.get('query_string') or b'')
            query = native_lane.query_items(query_string)
            upstream, target, eff = native_lane.authorize(broker, snapshot, principal, run.hints, match, run.surface)
            decision.effective = {'cloud_policy': eff.cloud_policy, 'location': eff.location}
            decision.candidates = [Candidate(ref=target.ref, klass=target.klass, eligible=True)]
            decision.chain = [target.ref]
            native_lane.check_owner(broker.handles, principal, broker.settings.role, match.upstream_id, route,
                                    match.params)
            content_type_in = inbound.header(run.scope.get('headers', []), 'content-type') or ''
            match, body, model_notes = native_lane.normalize_models(
                match, body, content_type_in, catalog=snapshot.catalog,
                local_names=_local_names(broker) if match.bare else None)
            native_lane.check_rel(match.rel, match.params)  # a rewritten {model} segment ('..@<upstream>' -> '..')
            run.notes.extend(note for note in model_notes if note not in run.notes)
            if PROFILES[upstream.profile].adapter == 'anthropic' and (route.method, route.path) == (
                    'POST', '/v1/messages/batches'):
                # the batch's requests may name file ids: another key's uploads stay invisible (the row is store-gated)
                from . import anthropic_state
                anthropic_state.check_batch(native_lane.parse_json(body), principal=principal, allow_store=True,
                                            store_scope=True, handles=broker.handles, upstream_id=match.upstream_id,
                                            role=broker.settings.role)
            from .limits import Estimate
            adapter_now = snapshot.adapters.get(match.upstream_id)
            if callable(getattr(adapter_now, 'native_usage', None)):
                # a character-billed provider (ElevenLabs character-cost): reserve characters (the body size, a high
                # bound of its text), so a per_1k_chars price row on native@<upstream> prices the call
                estimate = Estimate(input_tokens=0, max_output_tokens=0, characters=max(1, len(body)),
                                    token_output=False)
            else:
                estimate = Estimate(input_tokens=run.peek.summary.approx_input_tokens, max_output_tokens=0,
                                    token_output=False)
            run.admission = broker.limiter.admit(principal, target, estimate)
        except BrokerError as exc:
            run.outcome, run.status = 'refused', exc.status
            await self._refuse(run, exc)
            return
        adapter = snapshot.adapters.get(match.upstream_id)
        profile = PROFILES[upstream.profile]
        from .health import native_breaker
        breaker = native_breaker(broker.breakers, match.upstream_id, upstream)
        permit = breaker.allow()
        if not permit:
            decision.attempts.append(Attempt(ref=target.ref, outcome='breaker_open', status=503,
                                             error_code='breaker_open'))
            run.outcome = 'refused'
            await self._refuse(run, BrokerError('The upstream is temporarily unavailable', 'breaker_open',
                                                retry_after=permit.retry_after))
            return
        try:
            if method != 'GET':
                self._native_audit(run, match, 'http')
            grant = snapshot.issuer.issue('inference', plan_target=target, upstream_id=match.upstream_id,
                                          decision_id=decision.decision_id, request_id=run.hints.request_id,
                                          delegated=run.hints.delegated, reservation=run.admission,
                                          deadline=run.hints.deadline)
        except BrokerError as exc:
            breaker.record('refused')
            decision.attempts.append(Attempt(ref=target.ref, outcome='refused', status=exc.status, error_code=exc.code))
            run.outcome = 'refused'
            await self._refuse(run, exc)
            return
        except Exception as exc:  # the audit log could not be appended: nothing leaves the host
            breaker.record('refused')
            log.warning('broker: native call refused, audit unavailable (%s)', type(exc).__name__)
            run.outcome, run.status = 'refused', 503
            response = render_error(run.protocol, BrokerError('The audit log is unavailable; nothing was sent',
                                                              'upstream_unconfigured'))
            await self._send_response(run.scope, run.receive, run.send, response)
            return
        spec = profile.native
        options = egress_options(spec)
        headers = native_lane.forwarded_headers(profile, run.scope.get('headers', []),
                                                account_headers=native_lane.privileged(principal, broker.settings.role))
        started = time.perf_counter()
        left = self._left(run)
        timeout = None if left is None else max(0.0, left)
        try:
            response = await self._bounded(run, broker.egress.send(
                grant, upstream, profile, method, match.rel, content=body if body else None, params=query,
                headers=headers, stream=True, native=options), timeout, timeout is not None)
        except asyncio.TimeoutError:
            breaker.record('cancelled')
            decision.attempts.append(Attempt(ref=target.ref, outcome='timeout', status=504,
                                             error_code='deadline_exceeded', request_bytes_sent=True))
            run.outcome = 'error'
            await self._refuse(run, _deadline_error())
            return
        except httpx.HTTPError as exc:
            err = network_error(exc)
            breaker.record(err.kind)
            code = 'upstream_timeout' if err.kind == 'timeout' else 'upstream_unavailable'
            decision.attempts.append(Attempt(ref=target.ref, outcome=err.kind, status=err.status, error_code=code,
                                             request_bytes_sent=err.request_bytes_sent))
            run.outcome = 'error'
            await self._refuse(run, BrokerError(err.message or 'The upstream could not be reached', code,
                                                origin='network'))
            return
        except BrokerError as exc:
            # only an upstream's own bad answer counts against the shared breaker; a refusal decided here (egress_refused,
            # deadline, concurrency, an unconfigured upstream) sent nothing and must never open it for every other key
            breaker.record('server_error' if exc.code in UPSTREAM_FAULT_CODES else 'refused')
            decision.attempts.append(Attempt(ref=target.ref, outcome='refused', status=exc.status, error_code=exc.code))
            run.outcome = 'error'
            await self._refuse(run, exc)
            return
        except BaseException:
            breaker.record('cancelled')
            raise
        ms = (time.perf_counter() - started) * 1000.0
        status = response.status_code
        run.upstream_request_id = next((response.headers.get(name) for name in profile.request_id_headers
                                        if response.headers.get(name)), None)
        if not 200 <= status < 300:
            await self._native_error(run, match, target, adapter, breaker, response, ms)
            return
        breaker.record('success', headers=response.headers)
        broker.latency.observe(target.ref, ms)
        decision.attempts.append(Attempt(ref=target.ref, outcome='success', status=status, ms_to_headers=ms,
                                         ms_to_first_byte=ms, request_bytes_sent=True))
        decision.committed_at_attempt = 0
        run.ttfb_ms = ms
        content_type = response.headers.get('content-type', 'application/octet-stream')
        kept = {k: v for k, v in response.headers.items() if k.lower() not in ('content-type',)}
        if spec.rewrite_headers or spec.rewrite_body_fields:
            local_root = str(Request(run.scope).base_url) + 'native/' + match.upstream_id
            base_url = upstream.effective_base_url() or ''
            root_url = base_url if spec.root == 'base' else base_url
            if spec.root == 'origin':
                from urllib.parse import urlsplit
                parts = urlsplit(base_url)
                root_url = f'{parts.scheme}://{parts.netloc}'
            for name in spec.rewrite_headers:
                if name in kept:
                    rewritten = native_lane.rewrite_url(kept[name], upstream_origin_url=base_url, root_url=root_url,
                                                        local_root=local_root)
                    if rewritten != kept[name]:
                        # the SDK sends the next chunks to this URL without a key: an upload capability for this session
                        rewritten = native_lane.upload_capability(rewritten, principal, match.upstream_id)
                    kept[name] = rewritten
        if upload_token and 'final' in (inbound.header(run.scope.get('headers', []), 'x-goog-upload-command') or ''):
            native_lane.UPLOADS.finish(upload_token)  # the session is complete: its capability ends
        if (route.owner == 'create' or spec.rewrite_body_fields) and native_lane.is_json(content_type):
            try:
                data = await response.aread()
            except BrokerError as exc:
                await response.aclose()
                run.outcome = 'error'
                await self._refuse(run, exc)
                return
            finally:
                await response.aclose()
            document = native_lane.parse_json(data)
            if route.owner == 'create':
                try:
                    oid = native_lane.bind_created(broker.handles, principal, match.upstream_id, upstream.credential,
                                                   document)
                    if oid is None and route.store:
                        run.notes.append('native:unbound_object')
                except (OSError, ValueError):
                    log.warning('broker: could not record the native object owner')
                    run.notes.append('native:unbound_object')
            if spec.rewrite_body_fields:
                rewritten = native_lane.rewrite_body_urls(data, spec.rewrite_body_fields, upstream_origin_url=base_url,
                                                          root_url=root_url, local_root=local_root)
                if rewritten != data:
                    data = rewritten
                    run.notes.append('native:rewrote_body_urls')

            async def once():
                yield data

            produced = inbound.Produced(status, content_type, once(), False,
                                        (lambda: None) if route.owner == 'create' else
                                        _native_usage(adapter, route, response.headers), inbound._noop,
                                        run.upstream_request_id, kept, 'binary')
        else:
            streamed = any(token in content_type for token in ('event-stream', 'ndjson')) or \
                content_type.startswith(('audio/', 'video/', 'application/octet-stream'))
            produced = inbound.Produced(status, content_type, _response_chunks(response), streamed,
                                        _native_usage(adapter, route, response.headers), response.aclose,
                                        run.upstream_request_id, kept, 'binary')
        await self._commit(run, target, produced, keepalive=False)

    async def _native_error(self, run: _Run, match, target, adapter, breaker, response, ms: float) -> None:
        """A provider error answer on the native lane: 401 (and an auth 403) become 502 upstream_auth_failed; every other
        status passes through with the provider's scrubbed body and allowlisted headers."""
        from .redact import scrub
        status = response.status_code
        try:
            data = await response.aread()
        except BrokerError:
            data = b''
        finally:
            await response.aclose()
        err = adapter.map_error(status, response.headers, data) if adapter is not None else None
        kind = err.kind if err is not None else ('client_error' if status < 500 else 'server_error')
        if status == 401 and kind == 'server_error':
            status = 402  # a provider billing state sent as 401 (ElevenLabs quota_exceeded) is never the client's own 401
        breaker.record(kind, retry_after=getattr(err, 'retry_after', None))
        run.decision.attempts.append(Attempt(ref=target.ref, outcome=kind, status=status,
                                             error_code=getattr(err, 'provider_code', None), ms_to_headers=ms,
                                             request_bytes_sent=True))
        run.outcome = 'client_error' if status < 500 else 'error'
        if kind == 'auth':
            await self._refuse(run, BrokerError('The upstream refused the configured credential', 'upstream_auth_failed',
                                                origin='upstream'))
            return
        content_type = response.headers.get('content-type', 'application/json')
        try:
            text = data.decode('utf-8')
            body = scrub(text).encode('utf-8')
        except UnicodeDecodeError:
            body = b''
            content_type = 'application/json'
        if not body:
            exc = BrokerError(f'The upstream answered {status}', 'upstream_bad_response' if status >= 500 else
                              'invalid_request', status=status, origin='upstream')
            await self._refuse(run, exc)
            return
        headers = {k: v for k, v in response.headers.items() if k.lower() not in ('content-type',)}
        headers['content-type'] = content_type
        headers.update(self._provenance(run, error=True))
        headers['X-Titanos-Error-Origin'] = 'upstream'
        headers['X-Titanos-Served-By'] = self._served_by(target)
        run.status = status
        await run.send({'type': 'http.response.start', 'status': status,
                        'headers': self._pairs({k.lower(): v for k, v in headers.items()})})
        await run.send({'type': 'http.response.body', 'body': body, 'more_body': False})
        run.committed = True

    async def _native_ws(self, scope, receive, send):
        """A WebSocket on the native lane (module docstring): authenticated here (the Guard handles HTTP only), then
        matched, gated and relayed until either side closes or the request deadline expires."""
        from .types import Principal
        broker = self.broker
        message = await receive()
        if message.get('type') != 'websocket.connect':
            return
        auth = broker.authenticator
        headers = dict(scope.get('headers', []))
        outcome = auth.authenticate(scope, headers)
        if not isinstance(outcome, Principal):
            return await self._ws_deny(scope, send, outcome.status, outcome.code, outcome.message,
                                       protocol_for(scope.get('path', '')))
        principal = outcome
        match = self._native_match(scope, 'WS')
        error = None
        if match is None:
            match = ('error', BrokerError('Not found', 'not_found', origin='policy'), None)
        if isinstance(match, tuple):
            _, error, profile_id = match
            match = None
            protocol = native_lane.protocol_of(profile_id) if profile_id else protocol_for(scope.get('path', ''))
        else:
            protocol = native_lane.protocol_of(match.profile_id)
        if not auth.enforced:
            from ..server import cross_site_refusal
            refusal = cross_site_refusal(dict(scope, method='GET'), headers, False)
            if refusal is not None:
                return await self._ws_deny(scope, send, 403, 'cross_site_refused', refusal, protocol)
        try:
            hints = request_hints(scope.get('headers', []), broker.settings, principal)
        except BrokerError as exc:
            return await self._ws_deny(scope, send, exc.status, exc.code, str(exc), protocol)
        run = self._native_run(scope, receive, send, principal, hints, match, protocol, 'WS')
        token = broker_context.bind(principal, hints)
        try:
            if error is not None:
                raise error
            await self._native_ws_relay(run, match)
        except BrokerError as exc:
            run.outcome, run.status = ('refused' if run.outcome == 'success' else run.outcome), exc.status
            if not run.committed:
                if exc.code not in run.decision.notes:
                    run.decision.notes.append(f'refused:{exc.code}')
                await self._ws_deny(scope, send, exc.status, exc.code, str(exc), protocol, run=run)
        finally:
            broker_context.reset(token)
            await self._finish(run)

    async def _ws_deny(self, scope, send, status: int, code: str, message: str, protocol: str, run=None) -> None:
        """Refuse a WebSocket before accept: an HTTP error answer where the server supports the denial-response extension
        (the inbound protocol's envelope with the broker error headers), else a close frame (1008 for client and policy
        refusals, 1013 for retryable upstream trouble, 1011 otherwise) whose reason is the error code."""
        if 'websocket.http.response' in (scope.get('extensions') or {}):
            response = render_error(protocol, BrokerError(message, code if code in _ERROR_CODES else 'invalid_request',
                                                          status=status))
            pairs = [(k.lower().encode('latin-1'), v.encode('latin-1')) for k, v in response.headers.items()
                     if k.lower() != 'content-length']
            if run is not None:
                pairs += self._pairs(self._provenance(run, error=True))
            await send({'type': 'websocket.http.response.start', 'status': status, 'headers': pairs})
            await send({'type': 'websocket.http.response.body', 'body': response.body, 'more_body': False})
            return
        close = 1008 if status < 500 else (1013 if status in (503, 429) else 1011)
        await send({'type': 'websocket.close', 'code': close, 'reason': code[:120]})

    async def _native_ws_relay(self, run: _Run, match) -> None:
        from .egress import UpstreamHandshakeRefused, WebSocketProtocolError
        broker = self.broker
        snapshot = run.snapshot
        principal = run.principal
        decision = run.decision
        native_lane.check_rel(match.rel, match.params)
        query = native_lane.query_items(run.scope.get('query_string') or b'')
        upstream, target, eff = native_lane.authorize(broker, snapshot, principal, run.hints, match, run.surface)
        decision.effective = {'cloud_policy': eff.cloud_policy, 'location': eff.location}
        decision.candidates = [Candidate(ref=target.ref, klass=target.klass, eligible=True)]
        decision.chain = [target.ref]
        native_lane.check_owner(broker.handles, principal, broker.settings.role, match.upstream_id, match.route,
                                match.params)
        from .limits import Estimate
        run.admission = broker.limiter.admit(principal, target, Estimate(max_output_tokens=0, token_output=False))
        profile = PROFILES[upstream.profile]
        spec = profile.native
        adapter = snapshot.adapters.get(match.upstream_id)
        from .health import native_breaker
        breaker = native_breaker(broker.breakers, match.upstream_id, upstream)
        permit = breaker.allow()
        if not permit:
            decision.attempts.append(Attempt(ref=target.ref, outcome='breaker_open', status=503,
                                             error_code='breaker_open'))
            raise BrokerError('The upstream is temporarily unavailable', 'breaker_open', retry_after=permit.retry_after)
        try:
            self._native_audit(run, match, 'websocket')
        except BrokerError:
            breaker.record('refused')
            raise
        except Exception:
            breaker.record('refused')
            raise BrokerError('The audit log is unavailable; nothing was sent', 'upstream_unconfigured') from None
        try:
            grant = snapshot.issuer.issue('inference', plan_target=target, upstream_id=match.upstream_id,
                                          decision_id=decision.decision_id, request_id=run.hints.request_id,
                                          delegated=run.hints.delegated, reservation=run.admission,
                                          deadline=run.hints.deadline)
        except BrokerError:
            breaker.record('refused')  # nothing was sent: releases a half-open probe slot, counts nothing
            raise
        offered = [p.strip() for p in (inbound.header(run.scope.get('headers', []), 'sec-websocket-protocol') or '')
                   .split(',') if p.strip()]
        forwarded = [p for p in offered if p in spec.ws_subprotocols]
        dropped = [p for p in offered if p not in spec.ws_subprotocols]
        if dropped:
            run.notes.append('native:dropped_subprotocols')
        headers = native_lane.forwarded_headers(profile, run.scope.get('headers', []),
                                                account_headers=native_lane.privileged(principal, broker.settings.role))
        started = time.perf_counter()
        try:
            upstream_ws = await broker.egress.open_websocket(grant, upstream, profile, match.rel, params=query,
                                                             headers=headers, subprotocols=forwarded,
                                                             native=egress_options(spec))
        except UpstreamHandshakeRefused as refused:
            err = adapter.map_error(refused.status, refused.headers, refused.body) if adapter is not None else None
            kind = err.kind if err is not None else 'client_error'
            breaker.record(kind, retry_after=getattr(err, 'retry_after', None))
            decision.attempts.append(Attempt(ref=target.ref, outcome=kind, status=refused.status,
                                             error_code=getattr(err, 'provider_code', None), request_bytes_sent=True))
            run.outcome = 'client_error' if refused.status < 500 else 'error'
            if kind == 'auth':
                raise BrokerError('The upstream refused the configured credential', 'upstream_auth_failed',
                                  origin='upstream') from None
            from .redact import scrub
            text = scrub(refused.body.decode('utf-8', 'replace'))[:300] or f'The upstream answered {refused.status}'
            code = {'rate_limited': 'upstream_rate_limited', 'server_error': 'upstream_unavailable',
                    'timeout': 'upstream_timeout'}.get(kind, 'invalid_request')
            raise BrokerError(text, code, status=refused.status if kind == 'client_error' else None,
                              origin='upstream') from None
        except (httpx.HTTPError, *httpcore_errors(), OSError, WebSocketProtocolError) as exc:  # flat: 3.11+ refuses nested tuples
            breaker.record('network')
            decision.attempts.append(Attempt(ref=target.ref, outcome='network', status=503,
                                             error_code='upstream_unavailable', request_bytes_sent=True))
            run.outcome = 'error'
            raise BrokerError(f'The upstream WebSocket could not be opened ({type(exc).__name__})',
                              'upstream_unavailable', origin='network') from None
        except BrokerError as exc:  # refused here (egress_refused, deadline, concurrency): neutral, never a failure
            breaker.record('server_error' if exc.code in UPSTREAM_FAULT_CODES else 'refused')
            raise
        except BaseException:
            breaker.record('cancelled')
            raise
        ms = (time.perf_counter() - started) * 1000.0
        breaker.record('success')
        decision.attempts.append(Attempt(ref=target.ref, outcome='success', status=101, ms_to_headers=ms,
                                         ms_to_first_byte=ms, request_bytes_sent=True))
        decision.committed_at_attempt = 0
        run.ttfb_ms = ms
        provenance = self._provenance(run, target=target)
        accept = {'type': 'websocket.accept', 'subprotocol': upstream_ws.subprotocol,
                  'headers': self._pairs({k.lower(): v for k, v in provenance.items()})}
        await run.send(accept)
        run.committed, run.served, run.status = True, target, 101
        try:
            run.outcome = await self._ws_pump(run, upstream_ws, spec)
        finally:
            await upstream_ws.close(1000)
            uid = match.upstream_id
            self.broker.metrics.egress_bytes(uid, upstream_ws.bytes_in)
            decision.notes.append(f'native:ws_frames_bytes={upstream_ws.bytes_out}/{upstream_ws.bytes_in}')

    async def _ws_pump(self, run: _Run, upstream_ws, spec) -> str:
        from .egress import WebSocketProtocolError
        fields = spec.strip_frame_fields
        closing = {'code': 1000, 'reason': ''}

        async def client_to_upstream():
            while True:
                message = await run.receive()
                kind = message.get('type')
                if kind == 'websocket.disconnect':
                    closing['code'] = int(message.get('code') or 1000)
                    return 'cancelled'
                if kind != 'websocket.receive':
                    continue
                text = message.get('text')
                if text is not None:
                    stripped = native_lane.strip_frame(text, fields)
                    if stripped is not text and 'native:stripped_frame_credential' not in run.notes:
                        run.notes.append('native:stripped_frame_credential')
                    await upstream_ws.send_text(stripped)
                elif message.get('bytes') is not None:
                    await upstream_ws.send_bytes(message['bytes'])

        async def upstream_to_client():
            while True:
                try:
                    frame = await upstream_ws.recv()
                except WebSocketProtocolError:
                    closing['code'], closing['reason'] = 1002, 'upstream_protocol_error'
                    await run.send({'type': 'websocket.close', 'code': 1011, 'reason': 'stream_interrupted'})
                    return 'stream_interrupted'
                if frame is None:
                    code = upstream_ws.close_code or 1000
                    if code in (1005, 1006, 1015):
                        code = 1011 if code == 1006 else 1000
                    await run.send({'type': 'websocket.close', 'code': code,
                                    'reason': (upstream_ws.close_reason or '')[:120]})
                    return 'success' if code == 1000 else 'stream_interrupted'
                kind, data = frame
                if kind == 'text':
                    await run.send({'type': 'websocket.send', 'text': data})
                else:
                    await run.send({'type': 'websocket.send', 'bytes': data})

        tasks = [asyncio.create_task(client_to_upstream()), asyncio.create_task(upstream_to_client())]
        try:
            left = self._left(run)
            done, _ = await asyncio.wait(tasks, timeout=None if left is None else max(0.0, left),
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                run.notes.append('stream:deadline_exceeded')
                try:
                    await run.send({'type': 'websocket.close', 'code': 1008, 'reason': 'deadline_exceeded'})
                except Exception:
                    pass
                return 'deadline_exceeded'
            task = done.pop()
            try:
                return task.result()
            except (OSError, ConnectionError):
                return 'stream_interrupted'
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except BaseException:
                        pass

    # -- bookkeeping -----------------------------------------------------------------------------------------------------
    async def _finish(self, run: _Run) -> None:
        if run.router_ids:
            from .router import ROUTER_INFLIGHT
            for request_id in run.router_ids:
                ROUTER_INFLIGHT.remove(request_id)
            run.router_ids.clear()
        broker = self.broker
        decision = run.decision
        if decision is None:
            return
        total_ms = (time.perf_counter() - run.started) * 1000.0
        target = run.served
        cost = None
        if target is not None:
            decision.served = Served(ref=target.ref, klass=target.klass, upstream=target.upstream_id,
                                     upstream_model=target.upstream_model)
        decision.translation = run.translation
        decision.fallback_count = run.fallback_count
        decision.latency_ms = round(total_ms, 3)
        for note in run.notes:
            if note not in decision.notes:
                decision.notes.append(note)
        usage = run.usage
        prices = run.snapshot.prices
        basis = None
        if target is not None and target.klass != 'local' and prices is not None and run.committed:
            # the uploaded audio's measured duration prices a per-minute row when the provider reports no seconds
            upload = run.peek.summary.audio_measured if run.surface.capability == 'transcription' else None
            priced = prices.price(target.ref, usage, upload_audio_seconds=upload)
            cost, basis = priced.cost, priced.basis
        decision.usage = usage
        from .pricing import RESERVATION, money
        admission = run.admission
        if admission is not None:
            try:
                if run.committed and target is not None and target.klass != 'local':
                    admission.settle(usage, cost)
                    if cost is None and admission.charged is not None and admission.price is not None:
                        # a budgeted request whose answer carried nothing priceable: the reservation is the spend
                        # (fail closed), recorded with basis 'reservation' in the decision and the usage ledger so it
                        # survives a restart and is never an unexplained charge
                        cost, basis = admission.charged, RESERVATION
                        if 'cost:reserved_estimate' not in decision.notes:
                            decision.notes.append('cost:reserved_estimate')
                else:
                    admission.release()
            except Exception:
                log.warning('broker: settling the admission failed', exc_info=True)
        decision.cost_usd = money(cost) if cost is not None else None
        decision.price_basis = basis
        try:
            broker.decisions.append(decision)
        except Exception:
            log.warning('broker: recording the decision failed', exc_info=True)
        if target is not None:  # refusals are in the decision log, not the usage ledger
            broker.record_usage(run, decision, target, usage, cost, total_ms)
        klass = target.klass if target is not None else 'none'
        broker.metrics.request(run.surface.name, klass, run.outcome, run.ttfb_ms)
        log.info('broker request_id=%s decision=%s key=%s surface=%s served_by=%s status=%s ms=%d fallback=%d',
                 run.hints.request_id, decision.decision_id, decision.key_id or '-', run.surface.name,
                 self._served_by(target) if target is not None else '-', run.status, int(total_ms),
                 run.fallback_count)


def _local_names(broker) -> frozenset:
    """The local registry's model names (a bare native path naming one is refused), empty when unreadable."""
    try:
        return frozenset(broker.local_models().keys())
    except Exception:
        return frozenset()


def egress_options(spec):
    """The EgressClient NativeOptions of a profile's NativeSpec."""
    from .egress import NativeOptions
    return NativeOptions(relative_to=spec.root, request_headers=frozenset(spec.request_headers),
                         response_headers=frozenset(spec.response_headers))


def scopes_listed(method: str, path: str):
    from .scopes import listed_scope
    return listed_scope(method, path)


def httpcore_errors():
    import httpcore
    return httpcore.NetworkError, httpcore.TimeoutException, httpcore.ProtocolError


def _native_usage(adapter, route, headers):
    """The usage callable of a native answer: the adapter's native_usage(route, headers) when it reads usage from the
    provider's response headers (ElevenLabs character-cost), else nothing (the reservation stands)."""
    reader = getattr(adapter, 'native_usage', None)

    def usage():
        if reader is None:
            return None
        try:
            return reader(route, headers)
        except Exception:  # usage parsing never fails a committed answer
            return None
    return usage


def _response_chunks(response):
    """The raw body of an open egress response (bounded by the egress cap); httpx failures after commit end the stream
    (StreamAborted for the native lane, so the client sees a truncated transfer)."""
    async def chunks():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        except httpx.HTTPError as exc:
            raise UpstreamFailure(network_error(exc)) from None
        finally:
            await response.aclose()
    return chunks()


from .errors import ERROR_CODES as _ERROR_CODES  # noqa: E402

__all__ = ['BROKER_ONLY_SURFACES', 'BUFFER_LIMIT', 'BrokerDispatch', 'DROPPED_HEADER_LIMIT', 'FALLBACK_CLOUD',
           'FALLBACK_LOCAL', 'KEEPALIVE_SECONDS', 'LocalAttempt', 'QUEUE_SIZE', 'StreamAborted', 'egress_options',
           'inject_reasoning']
