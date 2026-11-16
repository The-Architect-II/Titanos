"""OpenAI-style HTTP endpoints plus explicit model lifecycle and video APIs."""

import asyncio
import base64
import hmac
import hashlib
import io
import json
import logging
import queue
import re
import struct
import wave
import sys
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from contextvars import copy_context
from pathlib import Path
from typing import Annotated, ClassVar, Literal
from urllib.parse import urlsplit

import anyio
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from . import __version__, agent_capability, json_depth, openai_params
from .config import Config, ModelSpec, WorkerSpec, home, read_config, registry_lock, write_config
from .engine import Engine, EngineError
from .providers import AcquisitionPlan, AcquisitionRequest, ProviderRegistry, import_model_link, request_from_plan
from .model_page import fetch_model_page
from .settings import load_settings, current_settings, _active_settings as _request_settings


def error(message, status=400, code="invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": code, "param": None, "code": code}}, status_code=status)


async def _settled(operation):
    """Cancellation must not abandon an owned operation that is still executing."""
    task = asyncio.ensure_future(operation)
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
        except BaseException:
            break
    if cancelled is not None:
        try:
            task.result()
        except BaseException as exc:
            detail = repr(exc) + ''.join('\n' + note for note in getattr(exc, '__notes__', ()))
            cancelled.add_note('Close operation also failed: ' + detail)
        raise cancelled
    return task.result()


# Unauthenticated public routes: readable by GET only (this FastAPI does not serve HEAD) and never body-bearing.
# This is the exact set the bearer check exempts; it doubles as the pre-read rejection set. The router's
# redirect_slashes variants ('/healthz/') are normalised onto it so they get the same pre-read answer.
# API documentation views: Swagger UI is served from the packaged swagger-ui-dist assets (tools/licenses/vendored_assets.json),
# never from a CDN; ReDoc is deliberately not served. The documentation shell, its two assets and the OpenAPI schema are
# readable without the bearer in both key modes (GET only, no body); every model/control route stays behind the bearer.
DOCS_ASSETS = {"swagger-ui.css": "text/css", "swagger-ui-bundle.js": "text/javascript"}
DOCS_ROUTES = frozenset({"/docs", "/openapi.json"} | {"/docs/assets/" + name for name in DOCS_ASSETS})
# Studio and the documentation shell are never framed by another origin (their public GETs bypass the cross-site check).
FRAME_DENY = {"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'"}
# /v1/health/ready is the broker readiness probe (orchestrators poll it without a key). GET /api/version is public only in
# managed role, where it is registered: the Guard takes its per-app public set from broker.role.public_routes(settings).
PUBLIC_ROUTES = frozenset({"/healthz", "/v1/health/ready", "/", "/ui/app.js", "/ui/i18n.js", "/ui/style.css",
                           "/ui/retrieval.js", "/ui/titanos-wordmark-v2.svg", "/ui/titanos-honeycomb-t-v2.svg"} | DOCS_ROUTES)
PUBLIC_METHODS = ("GET",)
# Request methods that cannot change state; a user-initiated navigation (Sec-Fetch-Site: none) is allowed for these.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# Body-read bounds are Settings fields (TITANOS_BODY_READ_SECONDS, TITANOS_BODY_READ_MIN_RATE,
# TITANOS_BODY_READ_CONCURRENCY, TITANOS_BODY_READ_WAIT_SECONDS): body_read_seconds is the minimum whole-body
# budget per request (expiry answers 408 and closes), body_read_min_rate (bytes/s) widens it to length / rate for a
# larger declared Content-Length, body_read_concurrency bounds simultaneous body-bearing reads and
# body_read_wait_seconds bounds the wait for a read slot before 503. The Guard reads the Settings instance the
# app was created with (the process environment at startup), so changing them takes a server restart.
# Keyless loopback mode: the only origins that count as this server's own (scheme http, exact port).
LOOPBACK_ORIGIN_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


def _bounded_error(message, status, code, headers=None):
    """Small JSON error whose response never touches the ASGI receive callable."""
    response = error(message, status, code)
    for name, value in (headers or {}).items():
        response.headers[name] = value
    return response


def declares_body(headers):
    """True when the request framing announces a body (Content-Length > 0 or chunked transfer)."""
    length = headers.get(b"content-length", b"").strip()
    if length and length != b"0":
        return True
    return b"chunked" in headers.get(b"transfer-encoding", b"").lower()


def declared_length(headers):
    """The Content-Length as an int, or None when absent or malformed (chunked bodies have none)."""
    length = headers.get(b"content-length", b"").strip()
    return int(length) if length.isdigit() else None


def body_read_budget(headers, settings=None):
    """Seconds allowed for the whole body: at least settings.body_read_seconds, widened for a declared length at
    settings.body_read_min_rate bytes/s. Without an explicit Settings the current (context or process) settings apply."""
    settings = settings or current_settings()
    length = declared_length(headers)
    if length is None:
        return settings.body_read_seconds
    return max(settings.body_read_seconds, length / settings.body_read_min_rate)


def public_path(path):
    """Path used for the public-route check: the router's trailing-slash redirect variants map onto the route."""
    return path.rstrip("/") or "/"


def referer_origin(referer):
    """scheme://authority of a Referer header value, or None when it carries no authority."""
    parts = urlsplit(referer.strip())
    if not parts.scheme or not parts.netloc:
        return None
    return parts.scheme + "://" + parts.netloc


def _authority(authority):
    """(host, port) of a lower-cased host[:port] authority (bracketed IPv6 aware); port 80 when omitted; None when malformed."""
    if authority.startswith("["):
        close = authority.find("]")
        if close < 0:
            return None
        host, rest = authority[:close + 1], authority[close + 1:]
    else:
        host, sep, port = authority.partition(":")
        rest = sep + port
    if not host or "/" in host:
        return None
    if rest == "":
        return host, 80
    if rest.startswith(":") and rest[1:].isdigit():
        return host, int(rest[1:])
    return None


def own_port(scope, headers):
    """The port a same-origin browser request must name: the Host header's port (default 80), else the bound port.

    In keyless mode KeylessHostGuard (TrustedHostMiddleware's loopback allowlist) has already limited the Host name, and a
    browser cannot forge Host, so the Host port equals the origin port the browser used to reach us
    (this also keeps Studio working through a local port-forward)."""
    host = headers.get(b"host", b"").decode("latin-1").strip().lower()
    parsed = _authority(host) if host else None
    if parsed is not None:
        return parsed[1]
    server = scope.get("server")
    if server and server[1]:
        return int(server[1])
    return 80


def origin_is_own(origin, port):
    """Case-insensitive match against http://{127.0.0.1,localhost,[::1]}:<port>; 'null' and other schemes are foreign."""
    value = origin.strip().lower()
    if not value.startswith("http://"):
        return False
    parsed = _authority(value[len("http://"):])
    return parsed is not None and parsed[0] in LOOPBACK_ORIGIN_HOSTS and parsed[1] == port


def cross_site_refusal(scope, headers, public):
    """Keyless loopback mode only: a browser-attributed request must be same-origin.

    Returns the refusal reason (HTTP 403) or None. Requests carrying neither Origin, Sec-Fetch-Site
    nor Referer (curl, SDKs, the MCP client, other local agents) are not browser requests and pass
    unchanged. Public routes stay readable by GET from any site. For every other route only
    Sec-Fetch-Site: same-origin (or an Origin in this server's own loopback set) passes; 'cross-site',
    'same-site' and Origin 'null' are refused before any inventory or engine call. Sec-Fetch-Site: none
    without an Origin is a user-initiated navigation (address bar, bookmark, pasted link) that no page can
    produce, so it passes for safe methods (GET/HEAD/OPTIONS) and is refused for state-changing ones. A
    request with more than one Origin header is refused. Referer is a fallback signal for browsers that
    predate Fetch Metadata: with no Origin and no Sec-Fetch-Site, a Referer whose origin is not this server's
    loopback origin is refused. Residual: a pre-Fetch-Metadata browser sending neither Origin nor Referer
    (Referrer-Policy: no-referrer) is indistinguishable from a local non-browser client and passes.
    When an API key is configured the bearer check stays the gate and this check is not applied."""
    method = scope.get("method", "GET").upper()
    raw = scope.get("headers", [])
    if sum(1 for name, _ in raw if name == b"origin") > 1:
        return "Cross-site request refused: more than one Origin header (keyless loopback mode)"
    origin = headers.get(b"origin")
    site = headers.get(b"sec-fetch-site")
    referer = headers.get(b"referer")
    if referer is not None and not referer.strip():
        referer = None  # a blank Referer carries no signal
    if origin is None and site is None and referer is None:
        return None
    if public and method in PUBLIC_METHODS:
        return None
    port = own_port(scope, headers)
    if origin is not None and not origin_is_own(origin.decode("latin-1"), port):
        return "Cross-site request refused: Origin is not this server's loopback origin (keyless loopback mode)"
    if site is not None:
        value = site.strip().lower()
        if value == b"none" and origin is None and method in SAFE_METHODS:
            return None
        if value != b"same-origin":
            return "Cross-site request refused: Sec-Fetch-Site must be same-origin for this route (keyless loopback mode)"
        return None
    if origin is None and referer is not None:
        source = referer_origin(referer.decode("latin-1"))
        if source is None or not origin_is_own(source, port):
            return "Cross-site request refused: Referer is not this server's loopback origin (keyless loopback mode)"
    return None


class Guard:
    """Authenticate before consuming uploads; bound chunked and fixed-length bodies.

    Order per request: public-route method/body rejection (no receive), authentication (keyed mode), keyless cross-site
    refusal, then a bounded body read (time budget + concurrency slots).

    With auth (a broker.auth.Authenticator, as create_app installs it), every non-public route is answered by
    auth.authenticate(scope, headers) before the body is read: it writes the Principal into this same scope dict
    (scope['titanos.principal'], so BrokerDispatch and the mounted /mcp see it), enforces the route scope (403
    scope_denied) and, in keyless mode, ignores presented credentials exactly as before (stock SDKs send placeholder
    keys). A refusal is an AuthDenial rendered in the route family's envelope. The keyless cross-site refusal is keyed
    on `not auth.enforced`. public_routes is this app's public set (PUBLIC_ROUTES, plus /api/version in managed role).
    Without auth the original single bearer compare against api_key applies."""
    def __init__(self, app, api_key=None, settings=None, gate=None, auth=None, public_routes=PUBLIC_ROUTES):
        self.settings = settings or load_settings()
        self.app, self.api_key, self.max_body = app, api_key, self.settings.max_body_bytes
        self.gate = gate
        self.auth = auth
        self.public_routes = frozenset(public_routes)
        self._slots = None
        self._slots_loop = None

    def enforced(self):
        """Whether credentials are checked (a bootstrap key or any key record); keyless loopback mode otherwise."""
        return self.auth.enforced if self.auth is not None else bool(self.api_key)

    def body_slots(self):
        """Per-loop semaphore over simultaneous body reads, sized from settings.body_read_concurrency when first used
        on each event loop (a later change to the setting takes effect only with a new app/loop, i.e. a restart)."""
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots_loop is not loop:
            self._slots = asyncio.Semaphore(self.settings.body_read_concurrency)
            self._slots_loop = loop
        return self._slots

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", [])); scope.setdefault("titanos.arrived", time.monotonic())  # request_deadline
        method = scope.get("method", "GET").upper()
        public = public_path(scope["path"]) in self.public_routes
        if public:
            # Public routes are answered from framing alone: nothing below calls receive() for them.
            if method not in PUBLIC_METHODS:
                return await _bounded_error("Method not allowed on this route", 405, "method_not_allowed",
                                            {"Allow": ", ".join(PUBLIC_METHODS)})(scope, receive, send)
            if declares_body(headers):
                return await _bounded_error("This route does not accept a request body", 400, "unexpected_body",
                                            {"Connection": "close"})(scope, receive, send)
        elif self.auth is not None:
            from .broker.envelopes import protocol_for
            scope['titanos.protocol'] = protocol_for(scope['path'])  # the route family's error envelope
            outcome = self.auth.authenticate(scope, headers)
            if not hasattr(outcome, 'scopes'):  # an AuthDenial: rendered in the route family's envelope
                return await outcome(scope, receive, send)
        elif self.api_key:
            if not hmac.compare_digest(headers.get(b"authorization", b""), ("Bearer " + self.api_key).encode()):
                return await error("Invalid API key", 401, "authentication_error")(scope, receive, send)
        if not self.enforced() and not scope.get("titanos.internal"):
            # Keyless loopback mode has no bearer; browser ambient authority is refused cross-site.
            # In-process MCP dispatch marks its scope and is not a browser request.
            refusal = cross_site_refusal(scope, headers, public)
            if refusal is not None:
                return await _bounded_error(refusal, 403, "cross_site_refused")(scope, receive, send)
        body = bytearray()
        if not public:
            length = declared_length(headers)
            if length is not None and length > self.max_body:
                # The framing alone proves the overflow: answer before receiving any of it.
                return await _bounded_error(f"Request exceeds configured {self.max_body} byte limit", 413,
                                            "request_too_large", {"Connection": "close"})(scope, receive, send)
            slot = None
            if declares_body(headers):
                slot = self.body_slots()
                try:
                    await asyncio.wait_for(slot.acquire(), self.settings.body_read_wait_seconds)
                except TimeoutError:
                    return await _bounded_error("Too many request bodies are being received; retry shortly", 503,
                                                "server_busy", {"Retry-After": "1", "Connection": "close"})(scope, receive, send)
            overflow = False
            budget = body_read_budget(headers, self.settings)
            try:
                async with asyncio.timeout(budget):
                    while True:
                        chunk = await receive()
                        if chunk["type"] == "http.disconnect":
                            return
                        body.extend(chunk.get("body", b""))
                        if len(body) > self.max_body:
                            overflow = True
                            break
                        if not chunk.get("more_body", False):
                            break
            except TimeoutError:
                return await _bounded_error(f"Request body was not received within {budget:g} s", 408,
                                            "request_timeout", {"Connection": "close"})(scope, receive, send)
            finally:
                if slot is not None:
                    slot.release()
            if overflow or json_depth.json_too_deep(body):  # nesting: a typed 400 before any route parses the body
                return await (error(f"Request exceeds configured {self.max_body} byte limit", 413, "request_too_large") if overflow else error(f"JSON nesting exceeds {json_depth.JSON_MAX_DEPTH} levels", 400, "invalid_request_error"))(scope, receive, send)
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        token = _request_settings.set(self.settings)
        # Count the actual internal HTTP operation, not its outer MCP transport.
        # This also covers stdio MCP, which has no outer HTTP request to count.
        mcp_transport = scope['path'].rstrip('/')=='/mcp' or scope['path'].startswith('/mcp/')
        # the registry gate never answers a public route (health, Studio shell and assets, documentation views) with 503
        tracked = self.gate is not None and not mcp_transport and scope['path'] not in {'/health','/v1/health'} and public_path(scope['path']) not in self.public_routes
        if tracked and self.gate['reloading']:
            _request_settings.reset(token)
            return await error('Model registry is reloading; retry shortly',503,'registry_reloading')(scope, replay, send)
        if tracked and self.gate.get('unavailable') and not (scope.get('method')=='POST' and scope['path']=='/api/models/reload'):
            _request_settings.reset(token)
            return await error('Model registry is unavailable after a failed reload; explicitly reload it or restart the server',503,'registry_unavailable')(scope, replay, send)
        if tracked:self.gate['active'] += 1
        primary = None
        # the authenticated principal for thread-side code on this request (artifact owner binding, API Plus stages)
        from .broker.context import PRINCIPAL as _principal_var
        principal_token = _principal_var.set(scope.get('titanos.principal'))
        try:
            return await self.app(scope, replay, send)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup_error = None
            try:
                # These callbacks retain opted-in ownership through the final ASGI
                # send, including iterators that never start and exceptional exits.
                for cleanup in reversed(scope.pop('titanos.cleanup', [])):
                    try:
                        with anyio.CancelScope(shield=True):
                            await _settled(cleanup())
                    except BaseException as exc:
                        if primary is not None:
                            primary.add_note('Request cleanup also failed: ' + repr(exc))
                        elif cleanup_error is None:
                            cleanup_error = exc
                        else:
                            cleanup_error.add_note('Additional request cleanup failure: ' + repr(exc))
                if primary is None and cleanup_error is not None:
                    raise cleanup_error
            finally:
                if tracked:self.gate['active'] -= 1
                _principal_var.reset(principal_token)
                _request_settings.reset(token)


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=128)
    worker: str | None = Field(default=None, max_length=128)
    # standard client fields (identity, prompt cache, storage, reasoning hints; embeddings dimensions; HiveMind's Ollama options
    # object on chat and completions) are split off by titanos.request_compat for the schemas that set _compat_kind: accepted ones
    # are named in X-Titanos-Ignored, unservable ones answer a typed 400 first, options.num_ctx is kept for context_option
    _compat_kind: ClassVar[str | None] = None
    _ignored: tuple = PrivateAttr(default=())
    _num_ctx: int | None = PrivateAttr(default=None)

    @model_validator(mode='wrap')
    @classmethod
    def _compat_fields(cls, data, handler):
        from .request_compat import split, unknown_fields
        data, ignored, num_ctx = split(cls._compat_kind, data);data, ignored = unknown_fields(cls._compat_kind, cls, data, ignored)
        instance = handler(data)
        if ignored:instance._ignored = tuple(ignored)
        instance._num_ctx = num_ctx
        return instance

    @model_validator(mode="after")
    def configured_limits(self):
        settings = current_settings()
        bounds = {'max_tokens':'max_tokens','max_completion_tokens':'max_tokens','max_output_tokens':'max_tokens',
                  'messages':'max_messages','tools':'max_tools','prompt':'max_prompt_chars',
                  'negative_prompt':'max_prompt_chars','instructions':'max_text_chars',
                  'steps':'max_steps','frames':'max_video_frames','fps':'max_video_fps','n':'max_choices' if isinstance(self,Generation) else 'max_generated_images'}
        for name, limit in bounds.items():
            value=getattr(self,name,None)
            size=len(value) if isinstance(value,(str,list,dict)) else value
            if size is not None and size>getattr(settings,limit):
                raise ValueError(f'{name} exceeds TITANOS_{limit.upper()}={getattr(settings,limit)}')
        value=getattr(self,'input',None)
        if isinstance(value,str):
            limit=settings.max_speech_chars if isinstance(self,Speech) else settings.max_text_chars
            if len(value)>limit:raise ValueError('input exceeds configured text limit')
        for msg in getattr(self,'messages',[]):
            if isinstance(msg.content,str) and len(msg.content)>settings.max_text_chars:
                raise ValueError('Message exceeds configured text limit')
        return self


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    detail: Literal["auto"] = "auto"


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image_url"]
    image_url: ImageURL


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[Annotated[TextPart | ImagePart, Field(discriminator="type")]] | None = None
    tool_calls: list[dict] | None = Field(default=None)
    tool_call_id: str | None = Field(default=None, max_length=256)
    name: str | None = Field(default=None, max_length=128)

    @model_validator(mode='after')
    def tool_message(self):
        settings=current_settings()
        if len(self.tool_calls or [])>settings.max_tools:
            raise ValueError('Message tool calls exceed configured limit')
        for part in self.content if isinstance(self.content,list) else []:
            if isinstance(part,TextPart) and len(part.text)>settings.max_text_chars:
                raise ValueError('Text part exceeds configured text limit')
            if isinstance(part,ImagePart) and len(part.image_url.url)>4*((settings.max_input_image_bytes+2)//3)+128:
                raise ValueError('Image data URL exceeds configured image limit')
        if self.role == 'tool' and (not self.tool_call_id or self.content is None):
            raise ValueError('Tool results require tool_call_id and content')
        if self.content is None and not (self.role == 'assistant' and self.tool_calls):
            raise ValueError('Message content is required except for assistant tool calls')
        for call in self.tool_calls or []:
            if self.role != 'assistant' or call.get('type') != 'function' or not call.get('id'):
                raise ValueError('Assistant tool calls require id, type=function and function')
            function=call.get('function',{})
            if not function.get('name') or not isinstance(function.get('arguments'),str):
                raise ValueError('Tool-call function arguments must be a JSON string')
        return self


class Generation(Payload):
    max_tokens: int = Field(default_factory=lambda:current_settings().default_max_tokens, ge=1)
    temperature: float = Field(default_factory=lambda:current_settings().default_temperature, ge=0, le=2)
    top_p: float = Field(default_factory=lambda:current_settings().default_top_p, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    top_k: int | None = Field(default=None,ge=0)
    min_p: float | None = Field(default=None,ge=0,le=1)
    repeat_penalty: float | None = Field(default=None,gt=0)
    # the OpenAI sampling fields the llama.cpp server (b10809) honours: forwarded to a llama-cpp model untouched; on the
    # in-process backends openai_params.local_generation maps logit_bias (transformers sequence_bias), drops inert values
    # and answers 400 unsupported_capability naming any other live value (n is bounded by TITANOS_MAX_CHOICES)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    logit_bias: dict[str, float] | None = Field(default=None, max_length=300)
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1)
    stream_options: dict[str, bool] | None = None

    @model_validator(mode="after")
    def valid_stops(self):
        stops = [self.stop] if isinstance(self.stop, str) else (self.stop or [])
        if len(stops) > current_settings().max_stop_sequences or any(not s or len(s) > current_settings().max_stop_chars for s in stops):
            raise ValueError(f"stop must contain at most {current_settings().max_stop_sequences} nonempty strings, each at most {current_settings().max_stop_chars} characters")
        if self.stream_options and set(self.stream_options) != {"include_usage"}:
            raise ValueError("stream_options supports include_usage only")
        return self


class Chat(Generation):
    require_loaded: bool = Field(default=False, strict=True)
    messages: list[Message] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    tools: list[dict] | None = Field(default=None)
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    _compat_kind: ClassVar[str] = 'chat'

    @model_validator(mode="after")
    def bounded_images(self):
        count = sum(isinstance(p, ImagePart) for m in self.messages if isinstance(m.content, list) for p in m.content)
        if count > current_settings().max_images:
            raise ValueError("A request accepts no more than the configured image count")
        if self.max_completion_tokens is not None and "max_tokens" in self.model_fields_set:
            raise ValueError("Use max_tokens or max_completion_tokens, not both")
        return self


class Completion(Generation):
    prompt: str = Field(min_length=1)
    # legacy completions take an alternatives count (0..5) rather than chat's boolean + top_logprobs
    logprobs: int | None = Field(default=None, ge=0, le=5, strict=True)
    _compat_kind: ClassVar[str] = 'completion'


class Embeddings(Payload):
    max_tokens: int | None = Field(default=None,gt=0,strict=True)
    truncate: bool | None = Field(default=None, strict=True, description='Explicitly allow tokenizer truncation; absent uses model/process policy, default false')
    input: str | list[str] = Field(min_length=1)
    encoding_format: Literal["float", "base64"] = "float"
    _compat_kind: ClassVar[str] = 'embeddings'  # user: accepted and named in X-Titanos-Ignored; dimensions: typed 400

    @model_validator(mode="after")
    def bounded_batch(self):
        texts = self.input if isinstance(self.input, list) else [self.input]
        if len(texts) > current_settings().embedding_max_batch or any(not text or len(text) > current_settings().max_text_chars for text in texts):
            raise ValueError("Embedding batch exceeds configured count or text length")
        return self


class TensorRequest(Payload):
    inputs: dict[str, dict] = Field(min_length=1, max_length=64)  # each {dtype, shape, data} or {dtype, shape, b64}
    output_encoding: Literal['list', 'base64'] = 'list'  # base64: each output as {dtype, shape, b64} (compiled.py)
    output_views: dict[str, dict] | None = None  # {output: {slice: [[start, stop], ...]}} per axis, applied before encoding (compiled.py)


# Speech fields only qwen3-tts reads: gim_api TTS Super's request sampling and the codec-frame cap (docs/QWEN_TTS.md)
QWEN_SPEECH = ('temperature', 'top_k', 'top_p', 'subtalker_temperature', 'subtalker_top_k', 'subtalker_top_p',
               'frame_budget')


class Speech(Payload):
    seed: int | None = Field(default=None,ge=0,le=2**32-1)
    input: str = Field(min_length=1)
    voice: str = "default"
    response_format: Literal["wav", "pcm"] = "wav"  # pcm: bare PCM16 at the model's rate, named in X-Sample-Rate
    stream_format: Literal["audio", "sse"] | None = None  # audio: the body streams as it is made (speech_stream.py)
    language: str | None = Field(default=None, max_length=12)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)  # honoured by sherpa-onnx; other backends refuse a value other than 1.0 by name
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    subtalker_temperature: float | None = None
    subtalker_top_k: int | None = None
    subtalker_top_p: float | None = None
    frame_budget: int | None = Field(default=None, ge=16)  # codec frames, 12.5 a second; at most the model's max_frames

    @field_validator('temperature', 'top_k', 'top_p', 'subtalker_temperature', 'subtalker_top_k', 'subtalker_top_p',
                     mode='before')
    @classmethod
    def qwen_sampling(cls, value, info):
        from .qwen_tts_backend import sampling_value
        return None if value is None else sampling_value(info.field_name, value)


class ImageRequest(Payload):
    prompt: str = Field(min_length=1)
    size: str = Field(default_factory=lambda:current_settings().default_image_size)
    n: int = Field(default=1, ge=1)
    steps: int = Field(default_factory=lambda:current_settings().default_steps, ge=1)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    negative_prompt: str | None = Field(default=None)
    guidance_scale: float | None = Field(default=None, ge=0)
    response_format: Literal["b64_json", "url"] = "b64_json"

    @model_validator(mode="after")
    def dimensions(self):
        if not re.fullmatch(r"\d+x\d+", self.size):
            raise ValueError("size must be WIDTHxHEIGHT")
        width, height = map(int, self.size.split("x"))
        if any(d < 32 or d > current_settings().max_image_dimension or d % 8 for d in (width, height)) or width * height > current_settings().max_image_pixels:
            raise ValueError("Dimensions must be multiples of 8, at least 32, and within configured dimension/pixel limits")
        return self


class VideoRequest(ImageRequest):
    prompt: str = Field(default="")
    input_reference: str | None = None
    pipeline_options: dict = Field(default_factory=dict,max_length=32)

    @model_validator(mode="after")
    def reference_or_prompt(self):
        if not self.prompt.strip() and self.input_reference is None:
            raise ValueError('Video requires a prompt or an input_reference')
        if self.input_reference is not None:
            if not self.input_reference.startswith('data:image/') or ';base64,' not in self.input_reference:
                raise ValueError('input_reference must be an inline image data URL')
            if len(self.input_reference)>4*((current_settings().max_input_image_bytes+2)//3)+128:
                raise ValueError('input_reference exceeds configured image byte limit')
        return self

    n: Literal[1] = 1
    frames: int = Field(default_factory=lambda:current_settings().default_video_frames, ge=1)
    fps: int = Field(default_factory=lambda:current_settings().default_video_fps, ge=1)
    response_format: Literal["url"] = "url"


class Responses(Payload):
    input: str | list[dict] = Field(min_length=1)
    instructions: str | None = None
    max_output_tokens: int = Field(default_factory=lambda:current_settings().default_max_tokens,ge=1)
    temperature: float = Field(default_factory=lambda:current_settings().default_temperature,ge=0,le=2)
    top_p: float = Field(default_factory=lambda:current_settings().default_top_p,gt=0,le=1)
    tools: list[dict] | None = Field(default=None)
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    text: dict | None = None
    stream: bool = False
    store: Literal[False] = False
    metadata: dict[str, str] | None = None
    _compat_kind: ClassVar[str] = 'responses'

    @model_validator(mode='after')
    def metadata_limits(self):
        if self.metadata is not None and (len(self.metadata) > 16 or any(len(k) > 64 or len(v) > 512 for k, v in self.metadata.items())):
            raise ValueError('metadata accepts at most 16 pairs with keys up to 64 and values up to 512 characters')
        return self


class ImageJob(ImageRequest):
    task: Literal['image']


class SpeechJob(Speech):
    task: Literal['speech']

    @model_validator(mode='after')
    def job_artifact(self):
        if self.stream_format is not None or self.response_format != 'wav':
            raise ValueError('A speech job writes a WAV file; stream_format and response_format pcm are POST /v1/audio/speech options')
        return self


class VideoJob(VideoRequest):
    task: Literal['video']


class MusicRequest(Payload):
    prompt:str=Field(min_length=1)
    mode:Literal['instrumental','song']='instrumental'
    lyrics:str|None=None
    duration:float=Field(default=30,ge=10,le=600)
    vocal_language:str|None=Field(default=None,max_length=32,pattern=r'^[A-Za-z][A-Za-z0-9_-]*$')
    bpm:int|None=Field(default=None,ge=30,le=300)
    key_scale:str|None=Field(default=None,max_length=32)
    seed:int|None=Field(default=None,ge=0,le=2**32-1)
    guidance_scale:float|None=Field(default=None,ge=0,le=20)
    steps:int|None=Field(default=None,ge=1)
    reference_audio_base64:str|None=None
    thinking:bool|None=None;use_cot_metas:bool|None=None;use_cot_caption:bool|None=None;use_cot_language:bool|None=None
    @model_validator(mode='after')
    def music_limits(self):
        settings=current_settings()
        if len(self.prompt)>settings.max_prompt_chars or self.lyrics is not None and len(self.lyrics)>settings.max_text_chars:
            raise ValueError('Music prompt or lyrics exceed configured text limits')
        if self.mode=='instrumental' and self.lyrics:
            raise ValueError('Instrumental music cannot include lyrics')
        if self.duration>settings.max_audio_seconds:
            raise ValueError('Music duration exceeds TITANOS_MAX_AUDIO_SECONDS')
        if self.reference_audio_base64 is not None and len(self.reference_audio_base64)>4*((settings.max_body_bytes+2)//3)+128:
            raise ValueError('Music reference audio exceeds the configured upload limit')
        return self


class MusicJob(MusicRequest):
    task:Literal['music']


AssetJob = Annotated[ImageJob | SpeechJob | VideoJob | MusicJob, Field(discriminator='task')]


class Registration(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$')
    spec: ModelSpec
    replace: bool = False


class AssignmentRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    model:str=Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$')
    mode:Literal['single','model_sharding','tensor_parallel']='single'
    groups:list[list[str]]=Field(default_factory=list,max_length=16)
    tensor_parallel_size:int=Field(default=2,ge=2,le=16)
    threads:int|None=Field(default=None,ge=1)
    timeout:float|None=Field(default=None,gt=0)
    policy:Literal['auto','cpu']|None=None

    @model_validator(mode='after')
    def stable_groups(self):
        pattern=re.compile(r'^(?:GPU|HIP|XPU|OV)-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
        if any(not group or len(group)>16 or len(set(group))!=len(group) or any(item!='cpu' and not pattern.fullmatch(item) for item in group) for group in self.groups):
            raise ValueError('Assignment groups require distinct complete GPU-/HIP-/XPU-/OV- UUIDs; mutable ordinals are not accepted')
        if self.policy is not None and self.groups:
            raise ValueError('Choose automatic/CPU policy or explicit device groups, not both')
        if self.policy is not None and self.mode!='single':
            raise ValueError('Automatic/CPU policy uses single mode; choose an explicit group for parallelism')
        if self.mode=='single' and any(len(group)!=1 for group in self.groups):
            raise ValueError('Single mode uses one device per replica group')
        cpu_tensor=self.mode=='tensor_parallel' and self.groups==[['cpu']]
        if any('cpu' in group and group!=['cpu'] for group in self.groups) or self.mode=='model_sharding' and any('cpu' in group for group in self.groups):
            raise ValueError('CPU supports single/replica or a CPU tensor group; it cannot mix with accelerator shards')
        if self.mode!='single' and not cpu_tensor and (len(self.groups)!=1 or len(self.groups[0])<2):
            raise ValueError('Sharding and tensor parallelism require one ordered group with at least two devices')
        if self.mode=='tensor_parallel' and self.groups and not cpu_tensor and self.tensor_parallel_size!=len(self.groups[0]):
            raise ValueError('Tensor parallel rank count must match the ordered device group')
        return self


class ProviderInspectRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    provider: str = Field(min_length=1,max_length=64)
    identifier: str = Field(min_length=1,max_length=512)
    revision: str | None = Field(default=None,min_length=1,max_length=256)


class ModelLinkRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str = Field(min_length=1,max_length=4096)


class AcquisitionActivation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    accepted_license_sha256: str | None = Field(default=None,pattern=r'^[a-f0-9]{64}$')
    load: bool = True
    runtime_profile: str | None = Field(default=None,pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]*$',max_length=80)


class EngineSlot:
    """One stable reference shared by HTTP and extension routes during reload."""
    def __init__(self, engine):self.target=engine;self.unavailable=False
    def __getattr__(self, name):
        if self.unavailable:raise EngineError('Model registry is unavailable after a failed reload',503,'registry_unavailable')
        return getattr(self.target,name)


def create_app(config: Config, api_key: str | None = None, output_dir: Path | None = None,
               settings=None, registry_path=None, runtime_catalog=None) -> FastAPI:
    settings = settings or load_settings()
    from .native import NativePool
    native = NativePool(config, settings=settings)
    local = Config(models={n:s for n,s in config.models.items() if n not in native.models},
                   workers={n:w for n,w in config.workers.items() if w.models[0] not in native.models})
    if local.workers:
        from .replicas import ReplicaEngine
        engine = ReplicaEngine(local, settings=settings)
    else:
        engine = Engine(local)
    engine=EngineSlot(engine)
    gate={'active':0,'reloading':False,'unavailable':False}
    outputs = output_dir or home() / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    from .jobs import VideoJobs
    jobs=VideoJobs(outputs/'jobs', settings=settings)
    from .acquisition import AcquisitionStore
    providers=ProviderRegistry()
    acquisitions=AcquisitionStore(outputs.parent/'acquisitions',providers,model_root=outputs.parent/'models',
                                  max_bytes=settings.acquisition_max_bytes,workers=settings.acquisition_workers,
                                  retention_seconds=settings.acquisition_retention_seconds)
    from .native_install import RuntimeSetup
    from .runtime_catalog import compatible_profiles,resolve_catalog
    runtime_catalog=None if runtime_catalog is False else resolve_catalog(runtime_catalog)
    runtime_state_root=outputs.parent if output_dir is None else outputs
    runtime_setup=RuntimeSetup(runtime_catalog,root=runtime_state_root/'runtime-setups') if runtime_catalog else None
    runtime_profiles=compatible_profiles(runtime_catalog) if runtime_catalog else []
    runtime_profile=next((row["id"] for row in runtime_profiles if row.get('recommended')),None)
    from .devices import visibility_constraints
    runtime_profiles_at=time.monotonic();runtime_visibility=visibility_constraints()
    runtime_profiles_error=None
    pending_cleanup=[]

    def cleanup_detail(exc):
        return repr(exc)+''.join('\n'+note for note in getattr(exc,'__notes__',()))

    settled = _settled

    async def dispose(objects, primary=None):
        """Attempt every owned close; retain failed objects for explicit recovery."""
        failures=[];seen=set()
        for kind,obj in objects:
            if id(obj) in seen:continue
            seen.add(id(obj))
            try:await settled(obj.close() if kind in {'native','broker'} else asyncio.to_thread(obj.close))
            except BaseException as exc:
                pending_cleanup.append((kind,obj));failures.append(exc)
        if failures:
            if primary is None:primary=failures.pop(0);raise_primary=True
            else:raise_primary=False
            for exc in failures:primary.add_note('Additional registry cleanup failure: '+cleanup_detail(exc))
            if raise_primary:raise primary

    @asynccontextmanager
    async def lifespan(app):
        # logging redaction re-applied after uvicorn's dictConfig, managed-role vault seal, audit role.start
        await broker.startup()
        if app.state.mcp is not None:
            async with app.state.mcp.session_manager.run():
                yield
        else:
            yield
        remaining=list(pending_cleanup);pending_cleanup.clear()
        await dispose([('jobs',jobs),('acquisitions',acquisitions),
                       *([('runtime_setup',runtime_setup)] if runtime_setup else []),
                       ('native',native),('engine',engine.target),('broker',broker),*remaining])

    app = FastAPI(title="Titanos", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, swagger_ui_oauth2_redirect_url=None,
                  description="Standalone multimodal inference. Model downloads are explicit CLI operations.")
    # The broker facade is lazy: no vault resolution, no network access and no file or directory is created here.
    from .broker.facade import install_broker
    from .broker.redact import install_logging_redaction
    install_logging_redaction()
    broker = install_broker(app, settings=settings, config=config, engine_slot=engine,
                            native_getter=lambda: app.state.native, bootstrap_key=api_key,
                            registry_path=registry_path, outputs=outputs)
    broker.bind_artifact_writer(lambda name: outputs / name)
    app.state.engine = engine
    app.state.native = native
    app.state.providers = providers
    app.state.acquisitions = acquisitions
    app.state.runtime_setup=runtime_setup
    app.state.runtime_profile=runtime_profile
    def runtime_profile_state(refresh=False):
        nonlocal runtime_profiles,runtime_profile,runtime_profiles_at,runtime_visibility,runtime_profiles_error
        if runtime_setup is None:return {'object':'list','data':[],'recommended':None,'metadata_status':'disabled'}
        current_scope=visibility_constraints()
        if refresh or current_scope!=runtime_visibility or time.monotonic()-runtime_profiles_at>settings.device_inventory_ttl:
            try:
                fresh=compatible_profiles(runtime_catalog)
            except (OSError,ValueError) as exc:
                runtime_profiles_error=str(exc);runtime_profile=None
            else:
                runtime_profiles=fresh;runtime_profile=next((row['id'] for row in fresh if row.get('recommended')),None)
                runtime_profiles_error=None
            runtime_profiles_at=time.monotonic();runtime_visibility=current_scope;app.state.runtime_profile=runtime_profile
        if runtime_profiles_error is not None:
            return {'object':'list','data':[{**row,'recommended':False} for row in runtime_profiles],
                    'recommended':None,'metadata_status':'unverified','reason':runtime_profiles_error}
        return {'object':'list','data':runtime_profiles,'recommended':runtime_profile,'metadata_status':'current',
                'visibility':runtime_visibility}
    if runtime_setup is not None:
        from .runtime_api import register_http
        register_http(app,runtime_setup)
        @app.get('/api/runtime-profiles')
        def runtime_profile_list(refresh:bool=False):
            return runtime_profile_state(refresh)
    app.state.registry_pending_cleanup = pending_cleanup
    app.state.mcp = None
    # Last added is outermost: KeylessHostGuard -> Guard (authentication, framing, body bounds) -> BrokerDispatch -> routes.
    from .broker.auth import KeylessHostGuard
    from .broker.dispatch import BrokerDispatch
    from .broker.role import public_routes as role_public_routes
    authenticator = broker.authenticator
    app.add_middleware(BrokerDispatch, broker=broker)
    app.add_middleware(Guard, api_key=api_key, settings=settings, gate=gate, auth=authenticator,
                       public_routes=role_public_routes(settings))
    # TrustedHostMiddleware's loopback Host allowlist (byte-identical refusal) while nothing is enforced; once a
    # bootstrap key or any key record exists every Host reaches the credential check.
    app.add_middleware(KeylessHostGuard, auth=authenticator)

    def principal_of(request):
        return request.scope.get('titanos.principal')

    @app.exception_handler(EngineError)
    async def engine_error(request, exc):
        from .broker.errors import BrokerError
        if isinstance(exc, BrokerError):
            # broker refusals in the route family's native envelope with X-Titanos-Error-Origin (and Retry-After,
            # x-should-retry where the errors table says so)
            from .broker.envelopes import protocol_for, render_error
            return render_error(protocol_for(request.url.path), exc)
        answer = error(str(exc), exc.status, exc.code); answer.headers.update(getattr(exc, "headers", None) or {}); return answer

    def audit_unavailable(request, exc):
        # every audited write appends its record before it changes anything (broker/admin.py, broker/keys.py), so a
        # failed append means the change was refused
        logging.error('broker audit log unavailable: %s', type(exc).__name__)
        return error('The audit log is unavailable; nothing was changed. Run titanos audit verify', 503,
                     'audit_unavailable')

    from .broker.audit import AuditChainError, LOCK_NAME as AUDIT_LOCK_NAME
    from filelock import Timeout as LockTimeout

    @app.exception_handler(AuditChainError)
    async def audit_chain_error(request, exc):
        return audit_unavailable(request, exc)

    @app.exception_handler(LockTimeout)
    async def lock_timeout(request, exc):
        # only the audit lock means the audit is unavailable; broker.json, keys and vault locks share the exception
        # type, and their contention is a retryable busy answer
        if Path(str(getattr(exc, 'lock_file', '') or '')).name == AUDIT_LOCK_NAME:
            return audit_unavailable(request, exc)
        logging.warning('broker state lock busy: %s', Path(str(getattr(exc, 'lock_file', '') or '')).name)
        return _bounded_error('A state file is locked by another writer; nothing was changed. Retry shortly', 503,
                              'server_busy', {'Retry-After': '1'})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        messages = [".".join(map(str, e["loc"])) + ": " + e["msg"] for e in exc.errors()[:8]]
        return error("; ".join(messages), 422)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logging.exception("Unhandled Titanos request", exc_info=exc)
        return error("Internal error; see server logs", 500, "server_error")

    @app.get("/healthz")
    def health():
        return {"status": "ok", "version": __version__, "role": settings.role}

    @app.get('/', include_in_schema=False)
    def studio():
        return FileResponse(Path(__file__).with_name('web') / 'index.html', headers={'Cache-Control': 'no-cache', **FRAME_DENY})

    @app.get('/ui/{name}', include_in_schema=False)
    def studio_asset(name: str):
        if name not in {'app.js', 'i18n.js', 'style.css', 'retrieval.js', 'titanos-wordmark-v2.svg', 'titanos-honeycomb-t-v2.svg'}:
            raise EngineError('File not found',404,'not_found')
        return FileResponse(Path(__file__).with_name('web') / name)

    @app.get('/docs', include_in_schema=False)
    def api_documentation():
        # Swagger UI from the packaged assets only: no CDN script/style, no remote favicon, validator badge disabled, so the
        # rendered page requests nothing outside this loopback origin. /redoc is not served (docs_url/redoc_url are None above).
        from fastapi.openapi.docs import get_swagger_ui_html
        page = get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + ' - API documentation',
                                   swagger_js_url='/docs/assets/swagger-ui-bundle.js', swagger_css_url='/docs/assets/swagger-ui.css',
                                   swagger_favicon_url='/ui/titanos-honeycomb-t-v2.svg', oauth2_redirect_url=None,
                                   swagger_ui_parameters={'validatorUrl': None})
        page.headers.update(FRAME_DENY)
        return page

    @app.get('/docs/assets/{name}', include_in_schema=False)
    def api_documentation_asset(name: str):
        if name not in DOCS_ASSETS:
            raise EngineError('File not found', 404, 'not_found')
        return FileResponse(Path(__file__).with_name('web') / name, media_type=DOCS_ASSETS[name], headers={'Cache-Control': 'no-cache'})

    @app.get('/api/info')
    def info(request: Request):
        profiles=runtime_profile_state()
        principal=principal_of(request)
        from .broker.scopes import allows
        # 'standalone': True keeps its documented meaning (no orchestrator dependency); role/orchestrated/broker are
        # additive. The settings block needs settings:read (keyless loopback, bootstrap, admin, or keys granted it).
        visible_settings=principal is None or allows(principal,'settings:read')
        return {'name':'Titanos','version':__version__,'standalone':True,
                'role':settings.role,'orchestrated':settings.role=='managed','broker':broker.info(principal),
                'inference_contracts':{'require_loaded':{'revision':1,'routes':['/v1/chat/completions']}},
                'interfaces':{'http':'/v1','openapi':'/openapi.json','studio':'/','responses':'/v1/responses','rerank':'/v1/rerank', 'vad':'/v1/audio/vad','music':'/v1/music/generations','assignments':'/api/assignments','retrieval_answer':'/v1/retrieval/answer','pair_classification':'/v1/classifications/text-pairs'},
                'model_setup':{'schema_version':1,'providers':'/api/model-providers','plan':'/api/acquisitions/plan',
                               'acquisitions':'/api/acquisitions','state_is_distinct_from_asset_jobs':True,
                               'runtime':({'configured':True,'recommended_profile':profiles['recommended'],
                                           'profiles':profiles['data'],'metadata_status':profiles['metadata_status'],'profiles_route':'/api/runtime-profiles',
                                           'plan':'/api/runtime-setups/plan','setups':'/api/runtime-setups'}
                                          if runtime_setup is not None else {'configured':False})},
                'tasks':sorted({s.task for s in config.models.values()}),
                'models':{n:{'task':s.task,'backend':s.backend,'device':s.device} for n,s in config.models.items()},
                'jobs':{'submit':'/api/jobs','tasks':['image','speech','video','music'],'capacity':settings.queue_capacity,
                        'idempotency_header':'Idempotency-Key','retention_seconds':settings.artifact_ttl_seconds,
                        'lookup':{'idempotency_key':'GET /api/jobs?idempotency_key=','status':'GET /api/jobs?status='},
                        'cancel':{'route':'DELETE /api/jobs/{id}','acknowledgement':'cancel_accepted',
                                  'terminal':['cancelled','completed','failed','interrupted']}},
                **({'settings':settings.describe()['settings']} if visible_settings else {}),
                'notes':[
                         'Configured capability is not hardware or model qualification.',
                         'Weights are loaded locally; inference never downloads a model.']}

    @app.get("/readyz")
    def ready():
        loaded = engine.catalog(loaded_only=True)["data"] + native.catalog(True)
        return JSONResponse({"ready": bool(loaded), "models": [m["id"] for m in loaded]}, status_code=200 if loaded else 503)

    @app.get("/v1/models")
    async def models(request: Request):
        principal=principal_of(request)
        if broker.active(principal):
            # R21: stale live catalogs of upstreams this principal may see are refreshed (never others)
            await broker.refresh_catalogs(principal)
        local=await asyncio.to_thread(lambda: engine.catalog()['data']+native.catalog())
        # local rows follow the key's model globs and target classes, as inference does (keyless/bootstrap unchanged)
        local=[m for m in local if broker.visible_local(principal,m.get('id'))]
        return {'object':'list','data':local+broker.catalog_for(principal)}

    @app.get("/v1/models/{name:path}")
    def model(name: str, request: Request):
        if (name in native.models or name in config.models) and not broker.visible_local(principal_of(request),name):
            raise EngineError(f"Unknown model {name!r}",404,"model_not_found")  # exactly the unknown-name answer
        if name in native.models:
            return next(m for m in native.catalog() if m['id']==name)
        if name not in config.models:
            # published aliases, qualified refs and exposed virtual models the caller may use (404 otherwise)
            entry=broker.model_entry(name,principal_of(request))
            if entry is not None:
                return entry
        engine.validate(name)
        return next(m for m in engine.catalog()["data"] if m["id"] == name)

    @app.get("/api/models/loaded")
    def loaded(request: Request):
        principal=principal_of(request)
        return {'object':'list','data':[m for m in engine.catalog(loaded_only=True)['data']+native.catalog(True)
                                        if broker.visible_local(principal,m.get('id'))]
                +broker.catalog_for(principal,loaded_only=True)}

    @app.get("/api/models/memory")
    def memory(request: Request):
        from .admission import memory_view
        principal=principal_of(request)
        return memory_view({name:spec for name,spec in config.models.items() if broker.visible_local(principal,name)},current_settings())

    @app.post("/api/models/load")
    async def load(payload: Payload):
        if payload.model in native.models:
            return await native.load(payload.model,payload.worker)
        return await asyncio.to_thread(engine.load,payload.model,payload.worker)

    @app.post("/api/models/unload")
    async def unload(payload: Payload):
        if payload.model in native.models:
            return await native.unload(payload.model,payload.worker)
        return await asyncio.to_thread(engine.unload,payload.model,payload.worker)

    async def reload_registry(change=None):
        nonlocal native
        if registry_path is None:
            raise EngineError('This embedded server has no writable registry path',409,'registry_read_only')
        if gate['reloading'] or gate['active']>1 or jobs.events:
            raise EngineError('Finish or cancel active inference/jobs before replacing the registry',409,'registry_busy')
        gate['reloading']=True
        lock=None;prepared=[];retirement_started=False;primary=None
        try:
            from filelock import Timeout
            path=Path(registry_path).expanduser().resolve()
            lock=registry_lock(path,timeout=0)
            try:lock.acquire()
            except Timeout:
                lock=None
                raise EngineError('Another writer owns the model registry; retry shortly',409,'registry_busy')
            if pending_cleanup:
                remaining=list(pending_cleanup);pending_cleanup.clear()
                await dispose(remaining)
            candidate=read_config(path) if path.exists() or change is None else Config()
            if change:change(candidate)
            candidate=Config.model_validate(candidate.model_dump())
            # a local name that would shadow a published alias or virtual model is refused (409 broker_name_conflict)
            # before the new pool is built and before anything is retired, so the registry stays available
            broker.on_registry_candidate(candidate)
            next_native=NativePool(candidate, settings=settings)
            prepared.append(('native',next_native))
            local=Config(models={n:s for n,s in candidate.models.items() if n not in next_native.models},
                         workers={n:w for n,w in candidate.workers.items() if w.models[0] not in next_native.models})
            if local.workers:
                from .replicas import ReplicaEngine
                next_engine=ReplicaEngine(local, settings=settings)
            else:next_engine=Engine(local)
            prepared.append(('engine',next_engine))
            retirement_started=True
            gate['unavailable']=engine.unavailable=True
            await settled(native.close())
            await settled(asyncio.to_thread(engine.target.close))
            # Persist only after preparation and retirement succeeded. The same
            # lock still owns the latest disk state; no stale rollback is written.
            if change:write_config(path,candidate)
            config.models=candidate.models
            config.workers=candidate.workers
            native=next_native
            app.state.native=native
            engine.target=next_engine
            broker.set_local_config(config)
            prepared.clear()
            gate['unavailable']=engine.unavailable=False
            return {'status':'reloaded','models':sorted(config.models),'loaded':[]}
        except BaseException as exc:
            primary=exc
            await dispose(prepared,primary)
            if retirement_started or pending_cleanup:
                gate['unavailable']=engine.unavailable=True
            raise
        finally:
            try:
                if lock is not None:lock.release()
            except BaseException as exc:
                gate['unavailable']=engine.unavailable=True
                if primary is None:raise
                primary.add_note('Additional registry lock release failure: '+cleanup_detail(exc))
            finally:gate['reloading']=False

    @app.post('/api/models/register')
    async def register_model(payload: Registration):
        if payload.spec.trust_remote_code or payload.spec.allow_pickle or payload.spec.native:
            raise EngineError('Executable loader options and legacy weights require trusted local CLI configuration')
        if payload.spec.load_options:
            if payload.spec.backend=='onnx-diffusers':
                # ModelSpec validates required DXGI selectors and the strict boolean.
                if set(payload.spec.load_options)-{'device_id','adapter_luid','disable_metacommands'}:
                    raise EngineError('HTTP ONNX image registration accepts only device_id, adapter_luid and boolean disable_metacommands; profile output directories require trusted local CLI configuration')
            else:
                variant=payload.spec.load_options.get('variant')
                if (payload.spec.backend!='diffusers' or payload.spec.task not in {'image','video'}
                        or set(payload.spec.load_options)!={'variant'}):
                    raise EngineError('HTTP registration only accepts the variant loader option for Diffusers image/video models; other loader options require trusted local CLI configuration')
                from .loader_policy import validate_weight_variant
                try:validate_weight_variant(variant)
                except ValueError as exc:raise EngineError(str(exc)) from exc
        allowed_defaults={'max_tokens','max_completion_tokens','max_new_tokens','temperature','top_p','top_k','min_p',
                          'repeat_penalty','stop','seed','size','n','steps','num_inference_steps','frames','num_frames','fps',
                          'negative_prompt','guidance_scale','image_guidance_scale','strength','preserve_unmasked','voice','language','threshold','nms','nms_iou','max_sequence_length','multimask','mask_threshold',
                          'max_output_tokens','max_input_tokens','min_score','min_margin','label_mapping',
                          'sentiment_prompt_version','sentiment_decoding','truncate','source_language','target_language',
                          'translation_prefix', 'sample_rate', 'min_speech_ms', 'min_silence_ms', 'speech_pad_ms',
                          'vad_enabled', 'vad_model', 'vad_options','mode','duration','vocal_language','bpm','key_scale','guidance_scale'}|({'thinking','use_cot_metas','use_cot_caption','use_cot_language'} if payload.spec.task=='music' else set())
        if payload.spec.task=='transcription':allowed_defaults.add('prompt')
        if payload.spec.backend=='tesseract':
            from .tesseract import CONTROL_NAMES
            allowed_defaults=CONTROL_NAMES
        if set(payload.spec.defaults)-allowed_defaults:
            raise EngineError('Library-specific defaults require trusted local CLI configuration; HTTP accepts documented inference controls')
        source=Path(payload.spec.path).expanduser().resolve()
        if not source.is_dir():raise EngineError('Checkpoint directory does not exist')
        spec=payload.spec.model_copy(update={'path':str(source)})
        if spec.device.startswith('cuda'):
            import torch
            if not torch.version.hip:
                from .devices import cuda_uuid
                from .hardware import select_device
                spec.device=await asyncio.to_thread(cuda_uuid,torch,select_device(spec.device))
        def change(cfg):
            if payload.name in cfg.models and not payload.replace:
                raise ValueError('Model exists; set replace=true to replace it')
            cfg.models[payload.name]=spec
        try:return await reload_registry(change)
        except ValueError as exc:raise EngineError(str(exc)) from exc

    @app.post('/api/models/reload')
    async def reload_models():
        return await reload_registry()

    @app.delete('/api/models/registrations/{name:path}')
    async def unregister_model(name: str):
        def change(cfg):
            if name not in cfg.models:raise EngineError('Unknown model',404,'model_not_found')
            if any(name in w.models for w in cfg.workers.values()):
                raise EngineError('Remove worker assignments in local configuration before unregistering this model',409)
            del cfg.models[name]
        return await reload_registry(change)

    @app.get("/api/workers")
    async def workers():
        return {"data": (engine.worker_status() if hasattr(engine, "worker_status") else [])+await asyncio.gather(*(w.observed_status() for w in native.workers))}

    @app.get('/api/devices')
    def devices(backend: Literal['nvidia','pytorch','openvino','onnx','system','all']='all',refresh:bool=False):
        from .devices import inventory
        if refresh:runtime_profile_state(True)
        return inventory(refresh,backend)

    def assignment_devices(model=None,refresh=False):
        from .devices import assignment_catalog
        return assignment_catalog(config.models[model] if model else None,refresh)

    @app.get('/api/assignments/devices')
    def assignment_device_list(model:str|None=None,refresh:bool=False):
        if model is not None and model not in config.models:raise EngineError('Unknown model',404,'model_not_found')
        if refresh:runtime_profile_state(True)
        return assignment_devices(model,refresh)

    def observed_assignment(model):
        observations=[]
        for worker in native.workers:
            if worker.model==model:
                state=worker.status()
                if state.get('loaded') and state.get('resolved_device'):
                    observations.append({'worker':worker.name,'device':state['resolved_device'],'kind':'native_loaded_placement',
                                         'verified_devices':state.get('verified_devices'),'kernel_execution_verified':False})
        owner=engine.target
        local=getattr(owner,'local',owner)
        if getattr(local,'current',None)==model and getattr(local,'states',{}).get(model)=='loaded':
            device=getattr(getattr(local,'runtime',None),'device',None)
            if device is not None:observations.append({'device':str(device),'kind':'loaded_runtime_device','kernel_execution_verified':False})
        for worker in getattr(owner,'workers',[]):
            meta=getattr(worker,'meta',{})
            if meta.get('current_model')==model and meta.get('verified_devices'):
                observations.append({'worker':worker.name,'device':','.join(row['uuid'] for row in meta['verified_devices']),
                    'verified_devices':meta['verified_devices'],'kind':'loaded_worker_uuid_mapping','kernel_execution_verified':False})
        if not observations:return None
        return {'device':'; '.join(dict.fromkeys(row['device'] for row in observations)),
                'observed':True,'observations':observations,'qualification':None}

    def assignment_view(model):
        workers=[(name,spec) for name,spec in config.workers.items() if model in spec.models]
        catalog=assignment_devices(model);present={row['id']:row for row in catalog['data']}
        groups=[list(spec.gpus) or ([spec.device] if spec.device!='auto' else []) for _,spec in workers]
        spec=config.models[model]
        selected=[identity for group in groups for identity in group]
        if not workers and spec.device.startswith(('GPU-','HIP-','XPU-','OV-')):selected=spec.device.split(',')
        missing=[identity for identity in selected if identity not in present or present[identity].get('present') is False]
        unavailable=[identity for identity in selected if identity in present and not present[identity].get('assignable')]
        cpu_policy=spec.device=='cpu' or spec.native is not None and spec.native.gpu_layers==0
        policy='pinned' if workers or spec.device not in {'auto','cpu'} else 'explicit_cpu' if cpu_policy else 'automatic'
        if policy!='automatic':
            selection={'status':'configured','device':('cpu' if policy=='explicit_cpu' else spec.device) if not workers else ','.join(identity for group in groups for identity in group),
                       'reason':'explicit_cpu' if policy=='explicit_cpu' else 'explicit_assignment','observed':False}
        else:
            from .devices import preferred_runtime_device,installed_runtime_preference
            family='pytorch' if spec.backend in {'transformers','diffusers','xtts','qwen3-tts'} else 'openvino' if spec.backend.startswith('openvino') else 'onnx' if spec.backend.startswith('onnx') else None
            preferred=None;selection_error=None
            if family:
                records=[{**row,**view} for row in catalog['data'] for view in row.get('runtimes',[]) if view['backend']==family]
                if family=='onnx':records.extend(catalog.get('providers',[]))
                try:preferred=preferred_runtime_device(records,backend=spec.backend if family!='pytorch' else family,task=spec.task,installed=installed_runtime_preference(),options=spec.load_options)
                except ValueError as exc:selection_error=str(exc)
            elif spec.backend in {'llama-cpp','ace-step'}:
                candidates=[row for row in catalog['data'] if row.get('assignable') and row['id']!='cpu']
                preferred=next(iter(candidates),None)
            else:
                preferred=next((row for row in catalog['data'] if row['id']=='cpu'),None)
            selection={'status':'recommended' if preferred else 'unknown','device':preferred['id'] if preferred else None,
                       'reason':preferred.get('selection_reason','cpu_only' if preferred['id']=='cpu' else 'compatible_visible_accelerator') if preferred else 'no_compatible_visible_accelerator',
                       'observed':False}
            if selection_error:selection['message']=selection_error
        from .capabilities import describe
        return {'model':model,'mode':workers[0][1].mode if workers else 'single',
                'groups':groups,
                'replicas':len(workers) if workers else 1,
                'tensor_parallel_size':workers[0][1].tensor_parallel_size if workers else config.models[model].tensor_parallel_size,
                'workers':[name for name,_ in workers],
                'missing_devices':missing,'unavailable_devices':unavailable,'ready':not missing and not unavailable,
                'automatic':policy=='automatic','configured_device':spec.device,'policy':policy,
                'selection':selection,'observed':observed_assignment(model),'modes':describe(spec).get('modes',{}),
                'visibility':catalog['visibility']}

    @app.get('/api/assignments/{model:path}')
    def get_assignment(model:str):
        if model not in config.models:raise EngineError('Unknown model',404,'model_not_found')
        return assignment_view(model)

    @app.post('/api/assignments')
    async def save_assignment(payload:AssignmentRequest):
        if payload.model not in config.models:raise EngineError('Unknown model',404,'model_not_found')
        spec=config.models[payload.model]
        from .capabilities import describe
        support=describe(spec).get('modes',{}).get(payload.mode,{})
        if payload.mode!='single' and not support.get('implemented'):
            raise EngineError(f"{payload.mode} is not implemented for {spec.backend}/{spec.task}",422,'unsupported_assignment')
        if support.get('execution_backend')=='torch.distributed':
            # A known unavailable collective cannot become available through a
            # hardware refresh. UUID families are explicit before any probe.
            for group in payload.groups:
                family='cuda' if all(v.startswith('GPU-') for v in group) else 'rocm' if all(v.startswith('HIP-') for v in group) else 'cpu' if group==['cpu'] else None
                collective=support.get('collectives',{}).get(family,{})
                if collective.get('available') is False:
                    raise EngineError('Requested collective is unavailable: '+collective.get('reason','unsupported_runtime'),422,'unsupported_assignment')
        from .devices import require_homogeneous
        catalog=assignment_devices(payload.model,True);present={row['id']:row for row in catalog['data']}
        missing=[identity for group in payload.groups for identity in group if identity not in present or present[identity].get('present') is False]
        if missing:raise EngineError('Selected stable devices are unavailable: '+', '.join(missing),409,'device_unavailable')
        unavailable=[identity for group in payload.groups for identity in group if not present[identity].get('assignable')]
        if unavailable:raise EngineError('Selected devices are masked, unsupported or lack a compatible runtime: '+', '.join(unavailable),409,'device_unavailable')
        for group in payload.groups:
            if len(group)>1:
                try:signature=require_homogeneous([present[identity] for identity in group])
                except ValueError as exc:raise EngineError(str(exc),422,'unsupported_assignment') from exc
                if not all(identity.startswith(('GPU-','HIP-')) for identity in group):
                    raise EngineError('Multi-device groups currently require NVIDIA/ROCm UUIDs; other runtime UUIDs support single-device assignments',422,'unsupported_assignment')
                collective=support.get('collectives',{}).get(signature['runtime_vendor'],{})
                if support.get('execution_backend')=='torch.distributed' and not collective:
                    raise EngineError('No distributed collective is declared for the selected runtime family',422,'unsupported_assignment')
                if collective.get('available') is False:
                    raise EngineError('Requested collective is unavailable: '+collective.get('reason','unsupported_runtime'),422,'unsupported_assignment')
            elif group==['cpu'] and payload.mode=='tensor_parallel':
                if support.get('execution_backend')!='torch.distributed':
                    raise EngineError('CPU tensor groups require the PyTorch distributed adapter',422,'unsupported_assignment')
                if support.get('collectives',{}).get('cpu',{}).get('available') is False:
                    raise EngineError('The CPU collective is unavailable in this runtime',422,'unsupported_assignment')
        safe=re.sub(r'[^A-Za-z0-9_-]+','-',payload.model).strip('-')[:72] or 'model'
        namespace='studio-'+hashlib.sha256(payload.model.encode('utf-8')).hexdigest()+'-'
        legacy=re.compile(r'studio-'+re.escape(safe)+r'-[1-9][0-9]*')
        def change(cfg):
            conflicts=[name for name,worker in cfg.workers.items() if payload.model in worker.models and (worker.models!=[payload.model] or not (name.startswith(namespace) or legacy.fullmatch(name)))]
            if conflicts:raise ValueError('Model belongs to an expert/shared worker assignment; edit that configuration explicitly: '+', '.join(conflicts))
            remaining={name:worker for name,worker in cfg.workers.items() if payload.model not in worker.models}
            model=cfg.models[payload.model]
            if payload.policy is not None:
                model.device='auto' if payload.policy=='auto' else 'cpu'
                if model.native is not None:
                    model.native.gpu_layers='auto' if payload.policy=='auto' else 0
                    model.native.split_mode='none';model.native.tensor_split=[]
            elif payload.groups and model.native is not None:
                if all(group==['cpu'] for group in payload.groups):
                    model.device='cpu';model.native.gpu_layers=0;model.native.split_mode='none';model.native.tensor_split=[]
                else:
                    model.device='auto'
                    if model.native.gpu_layers==0:model.native.gpu_layers='auto'
            for index,group in enumerate(payload.groups):
                name=namespace+str(index+1)
                if name in remaining:raise ValueError('Assignment worker ID is occupied by another model: '+name)
                remaining[name]=WorkerSpec(
                    models=[payload.model],device='cuda' if group[0].startswith(('GPU-','HIP-')) else group[0],
                    gpus=group if group[0].startswith(('GPU-','HIP-')) else [],mode=payload.mode,
                    tensor_parallel_size=payload.tensor_parallel_size,
                    **({'threads':payload.threads} if payload.threads is not None else {}),
                    **({'timeout':payload.timeout} if payload.timeout is not None else {}))
            cfg.workers=remaining
        try:await reload_registry(change)
        except ValueError as exc:raise EngineError(str(exc),409,'assignment_conflict') from exc
        return assignment_view(payload.model)

    @app.post("/api/models/details")
    async def details(payload: Payload):
        if payload.model in native.models:
            return await native.details(payload.model,payload.worker)
        return await asyncio.to_thread(engine.details,payload.model,payload.worker)

    @app.post("/api/models/capabilities")
    def capabilities(payload: Payload):
        if payload.model not in config.models:
            raise EngineError('Unknown model',404,'model_not_found')
        from .capabilities import describe
        from .model_inspection import RUNTIME_OBSERVABLE, reconcile_setup_with_runtime
        result = describe(config.models[payload.model])
        backend = config.models[payload.model].backend
        # Reconcile only backends whose loaded runtime can confirm the advisory; other resident runtimes are never queried
        # (their details walk tensors under the engine lock). Engines without the passive read keep the passive view.
        if payload.model not in native.models and backend in RUNTIME_OBSERVABLE and isinstance(result, dict) and isinstance(result.get('setup'), dict):
            observe = getattr(engine, 'loaded_details', None)
            details = observe(payload.model) if callable(observe) else None
            if details:
                result['setup'] = reconcile_setup_with_runtime(result['setup'], backend, details)
        return result

    async def provider_call(function,*args,**kwargs):
        try:return await asyncio.to_thread(function,*args,**kwargs)
        except ValueError as exc:raise EngineError(str(exc)) from exc
        except Exception as exc:
            logging.exception('Model provider operation failed')
            raise EngineError('Model provider operation failed; check connectivity, access, or retry later',503,'provider_error') from exc

    @app.get('/api/model-providers')
    def model_providers():
        return {'object':'list','data':providers.descriptors(),
                'link_import':'/api/model-providers/import','schema_version':1}

    @app.get('/api/model-providers/{provider}/search')
    async def provider_search(provider:str,task:str=Query(min_length=1,max_length=64),
                              query:str=Query(default='',max_length=512),limit:int=Query(default=20,ge=1,le=100)):
        try:selected=providers.get(provider)
        except ValueError as exc:raise EngineError(str(exc),404,'provider_not_found') from exc
        return {'object':'list','provider':provider,'task':task,
                'data':await provider_call(selected.search,query,task,limit)}

    @app.post('/api/model-providers/inspect')
    async def provider_inspect(payload:ProviderInspectRequest):
        try:selected=providers.get(payload.provider)
        except ValueError as exc:raise EngineError(str(exc),404,'provider_not_found') from exc
        return await provider_call(selected.inspect,payload.identifier,payload.revision)

    @app.post('/api/model-providers/import')
    async def provider_import(payload:ModelLinkRequest):
        try:imported=import_model_link(payload.url)
        except ValueError as exc:raise EngineError(str(exc)) from exc
        result={'imported':imported.model_dump(mode='json'),'metadata':None}
        if imported.provider is not None:
            try:selected=providers.get(imported.provider)
            except ValueError as exc:raise EngineError(str(exc),404,'provider_not_found') from exc
            result['metadata']=await provider_call(selected.inspect,imported.identifier,imported.revision)
        else:
            result['metadata']=await provider_call(
                fetch_model_page,payload.url,timeout=settings.model_page_timeout,
                max_bytes=settings.model_page_max_bytes,max_redirects=settings.model_page_max_redirects,
                max_links=settings.model_page_max_links,max_text_chars=settings.model_page_max_text_chars)
        return result

    @app.post('/api/acquisitions/plan')
    async def acquisition_plan(payload:AcquisitionRequest):
        try:selected=providers.get(payload.provider)
        except ValueError as exc:raise EngineError(str(exc),404,'provider_not_found') from exc
        return (await provider_call(selected.plan,payload)).model_dump(mode='json')

    @app.post('/api/acquisitions',status_code=202)
    async def submit_acquisition(plan:AcquisitionPlan,request:Request):
        key=request.headers.get('Idempotency-Key')
        if key is not None and not 1<=len(key)<=128:raise EngineError('Idempotency-Key must contain 1..128 characters')
        digest=hashlib.sha256(key.encode()).hexdigest() if key else None
        try:selected=providers.get(plan.provider)
        except ValueError as exc:raise EngineError(str(exc),404,'provider_not_found') from exc
        trusted=await provider_call(selected.plan,request_from_plan(plan))
        if trusted.fingerprint()!=plan.fingerprint():
            raise EngineError('Acquisition plan no longer matches provider metadata; create a fresh immutable plan',409,'acquisition_plan_changed')
        try:return await asyncio.to_thread(acquisitions.submit,trusted,digest)
        except ValueError as exc:raise EngineError(str(exc)) from exc

    @app.get('/api/acquisitions')
    async def list_acquisitions(limit:int=Query(default=100,ge=1,le=1000)):
        return await asyncio.to_thread(acquisitions.list,limit)

    @app.get('/api/acquisitions/{identifier}')
    async def get_acquisition(identifier:str):
        return await asyncio.to_thread(acquisitions.get,identifier)

    @app.delete('/api/acquisitions/{identifier}')
    async def cancel_acquisition(identifier:str):
        return await asyncio.to_thread(acquisitions.cancel,identifier)

    def persisted_plan_invalid(exc):
        # A stored plan that no longer satisfies the acquisition file-name rules is a typed refusal, not a 500.
        first=(exc.errors() or [{}])[0]
        detail=str(first.get('msg','invalid plan'))
        return EngineError('Persisted acquisition plan is no longer valid; plan the acquisition again: '+detail[:200],
                           422,'acquisition_plan_invalid')

    @app.post('/api/acquisitions/{identifier}/retry',status_code=202)
    async def retry_acquisition(identifier:str):
        from pydantic import ValidationError
        try:return await asyncio.to_thread(acquisitions.retry,identifier)
        except ValidationError as exc:raise persisted_plan_invalid(exc) from exc

    @app.post('/api/acquisitions/{identifier}/activate')
    async def activate_acquisition(identifier:str,payload:AcquisitionActivation):
        from pydantic import ValidationError
        record=await asyncio.to_thread(acquisitions.activation_ready,identifier,payload.accepted_license_sha256)
        try:plan=AcquisitionPlan.model_validate(record['plan'])
        except ValidationError as exc:raise persisted_plan_invalid(exc) from exc
        allowed={'task','backend','device','dtype','graph','load_options','defaults','native'}
        if set(plan.model_spec)-allowed:
            raise EngineError('Acquisition plan contains loader settings unavailable to network activation')
        values={**plan.model_spec,'path':record['prepared_path'],
                'source_repo':plan.candidate_id if plan.provider=='huggingface' else plan.source_url,
                'source_revision':plan.resolved_revision}
        managed_runtime=None
        if plan.backend=='llama-cpp':
            if values.get('native',{}).get('executable') is not None:
                raise EngineError('Acquisition activation cannot accept a client executable path',422,'unsupported_runtime')
            from .native_install import executable_path
            receipt_path=home()/'native-runtime.json'
            try:
                before=receipt_path.read_bytes();receipt=json.loads(before)
                executable=await asyncio.to_thread(executable_path)
                if receipt_path.read_bytes()!=before or Path(receipt['executable']).resolve()!=executable.resolve():
                    raise ValueError('Managed runtime changed during activation; retry after setup completes')
                if payload.runtime_profile is not None and receipt.get('profile')!=payload.runtime_profile:
                    raise EngineError('Prepare the selected runtime profile before activation; the active managed receipt belongs to another profile',409,'runtime_profile_mismatch')
                values['native']={**values['native'],'executable':str(executable)}
                managed_runtime={'runtime_profile':receipt.get('profile'),'runtime_fingerprint':receipt.get('fingerprint'),
                                 'runtime_binary_sha256':receipt.get('binary_sha256')}
            except FileNotFoundError as exc:
                if payload.runtime_profile is not None or payload.load:
                    raise EngineError('Prepare a verified managed native runtime before activation',409,'runtime_unavailable') from exc
                managed_runtime={'runtime_profile':None,'runtime_status':'deferred_unverified'}
            except (OSError,ValueError,KeyError) as exc:
                raise EngineError('A verified managed native runtime is required before activation: '+str(exc),409,'runtime_unavailable') from exc
        elif payload.runtime_profile is not None:
            raise EngineError('runtime_profile applies only to managed llama.cpp acquisitions',422,'unsupported_runtime')
        try:spec=ModelSpec.model_validate(values)
        except ValueError as exc:raise EngineError(str(exc)) from exc
        previous=config.models.get(plan.registry_name)
        changed=previous is None or previous.model_dump(mode='json')!=spec.model_dump(mode='json')
        def change(cfg):
            existing=cfg.models.get(plan.registry_name)
            same=existing is not None and existing.model_dump(mode='json')==spec.model_dump(mode='json')
            if existing is not None and not plan.replace and not same:
                raise ValueError('Model exists; set replace=true in the acquisition plan')
            cfg.models[plan.registry_name]=spec
        try:await reload_registry(change)
        except ValueError as exc:raise EngineError(str(exc)) from exc
        async def finish_activation():
            loaded=None
            if payload.load:
                if plan.registry_name in app.state.native.models:
                    loaded=await app.state.native.load(plan.registry_name,None)
                else:
                    loaded=await asyncio.to_thread(engine.load,plan.registry_name,None)
            activation={'model':plan.registry_name,'status':'loaded' if loaded else 'registered',
                        'provider':plan.provider,'resolved_revision':plan.resolved_revision}
            if managed_runtime:activation.update(managed_runtime)
            return await asyncio.to_thread(acquisitions.mark_active,identifier,activation,
                                           payload.accepted_license_sha256)
        finish=asyncio.create_task(finish_activation())
        try:return await asyncio.shield(finish)
        except BaseException as primary:
            committed=False
            if isinstance(primary,asyncio.CancelledError):
                try:
                    await finish
                    committed=True
                except BaseException as finish_error:
                    primary.add_note('Activation settlement failed: '+str(finish_error))
            if changed and not committed:
                def rollback(cfg):
                    if previous is None:cfg.models.pop(plan.registry_name,None)
                    else:cfg.models[plan.registry_name]=previous
                try:await reload_registry(rollback)
                except BaseException as rollback_error:
                    primary.add_note('Activation rollback also failed: '+str(rollback_error))
            raise

    @app.delete('/api/acquisitions/{identifier}/weights')
    async def delete_acquired_weights(identifier:str):
        record=await asyncio.to_thread(acquisitions.get,identifier)
        value=record.get('prepared_path')
        if value and any(Path(spec.path).resolve()==Path(value).resolve() for spec in config.models.values()):
            raise EngineError('Unload and remove the model registration before deleting its stored weights',409,'model_registered')
        return await asyncio.to_thread(acquisitions.delete_weights,identifier)

    @app.post("/api/infer")
    def tensors(payload: TensorRequest, request: Request):
        return within_budget(request, lambda cancel: engine.infer(payload.model, "tensor", payload.model_dump(), cancel=cancel))

    def model_defaults(payload):
        from .request_defaults import model_defaults as apply_defaults
        return apply_defaults(payload, config.models.get(payload.model))

    async def generate(payload: Generation, task, request):
        sized, payload = ({'num_ctx': payload._num_ctx} if payload._num_ctx else {}), model_defaults(payload)  # native slot size
        data = payload.model_dump(exclude_none=True)
        require_loaded = data.pop('require_loaded', False)
        if data.get("max_completion_tokens") is not None:
            data["max_tokens"] = data.pop("max_completion_tokens")
        if payload.model in native.models and openai_params.native_generation(native, payload.model):
            if require_loaded:
                return await native.proxy(request, '/v1/chat/completions', data, require_loaded=True, **sized)
            return openai_params.legacy_logprobs(await native.proxy(request,'/v1/chat/completions' if task=='chat' else '/v1/completions',data,**sized),task,data)
        engine.validate(payload.model, task);data = openai_params.local_generation(data, config.models.get(payload.model))
        if any(k in data for k in ('tools','tool_choice','parallel_tool_calls','response_format')) or any(m.get('role') in {'tool','developer'} or m.get('tool_calls') for m in data.get('messages',[])):
            raise EngineError(agent_capability.refusal(payload.model,config.models.get(payload.model),agent_capability.chat_capabilities(data),request.url.path),400,'unsupported_capability')
        ident, created = ("chatcmpl-" if task == "chat" else "cmpl-") + uuid.uuid4().hex, int(time.time())

        def response(result, chunk=False, delta=None):
            if task == "chat":
                choice = {"index": 0, "delta" if chunk else "message": delta if chunk else {"role": "assistant", "content": result["text"]},
                          "finish_reason": result.get("finish_reason")}
            else:
                choice = {"index": 0, "text": delta.get("content", "") if chunk else result["text"], "logprobs": None,
                          "finish_reason": result.get("finish_reason")}
            out = {"id": ident, "object": "chat.completion.chunk" if chunk and task == "chat" else "chat.completion" if task == "chat" else "text_completion",
                   "created": created, "model": payload.model, "choices": [choice]}
            if not chunk:
                out["usage"] = result["usage"]
            return out

        if require_loaded:
            loop = asyncio.get_running_loop()
            admission, outcome = loop.create_future(), loop.create_future()
            chunks = asyncio.Queue(maxsize=32)
            cancel, release = threading.Event(), threading.Event()
            worker = None

            def resolve(future, value):
                if not future.done():
                    future.set_result(value)

            def publish(future, value):
                loop.call_soon_threadsafe(resolve, future, value)

            def put(item):
                if cancel.is_set():
                    return
                pending = asyncio.run_coroutine_threadsafe(chunks.put(item), loop)
                while True:
                    try:
                        pending.result(timeout=0.1)
                        return
                    except FutureTimeout:
                        if cancel.is_set():
                            pending.cancel()
                            return

            def finished(result, exc):
                publish(outcome, (result, exc))
                # The owner exits its existing lock/busy context on this thread.
                release.wait()

            def run():
                try:
                    engine.infer(payload.model, task, data,
                                 put if payload.stream else None,
                                 cancel, require_loaded=True,
                                 on_admitted=lambda: publish(admission, None), on_finished=finished)
                except BaseException as exc:
                    publish(admission, exc)
                    publish(outcome, (None, exc))

            async def finish():
                cancel.set()
                release.set()
                if worker is not None:
                    await _settled(worker)

            async def execute():
                # A cancelled to_thread Task can finish before its thread. Keep
                # the executor future shielded until the real call has settled.
                actual = loop.run_in_executor(None, copy_context().run, run)
                try:
                    await asyncio.shield(actual)
                except asyncio.CancelledError as exc:
                    cancel.set()
                    refusal = EngineError('Request cancelled', 499, 'cancelled')
                    resolve(admission, refusal)
                    resolve(outcome, (None, refusal))
                    try:
                        await _settled(actual)
                    except BaseException as cleanup:
                        exc.add_note('Execution also failed while cancelling: ' + repr(cleanup))
                    raise

            def execution_done(task):
                # Cancellation before execute starts dispatches nothing, but the
                # handler still needs a terminal admission/outcome notification.
                exc = EngineError('Request cancelled', 499, 'cancelled') if task.cancelled() else task.exception()
                if exc is not None:
                    cancel.set()
                    resolve(admission, exc)
                    resolve(outcome, (None, exc))

            request.scope.setdefault('titanos.cleanup', []).append(finish)
            worker = asyncio.create_task(execute())
            worker.add_done_callback(execution_done)

            async def observe_disconnect():
                try:
                    while True:
                        message = await request.receive()
                        if message['type'] == 'http.disconnect':
                            raise EngineError('Request cancelled', 499, 'cancelled')
                except Exception as exc:
                    cancel.set()
                    resolve(admission, exc)
                    resolve(outcome, (None, exc))

            observer = asyncio.create_task(observe_disconnect())
            try:
                refusal = await admission
                if refusal is not None:
                    raise refusal
                if not payload.stream:
                    result, exc = await outcome
                    if exc is not None:
                        raise exc
            finally:
                observer.cancel()
                await _settled(asyncio.gather(observer, return_exceptions=True))
            if cancel.is_set():
                raise EngineError('Request cancelled', 499, 'cancelled')
            if not payload.stream:
                return response(result)

            async def loaded_events():
                try:
                    yield 'data: ' + json.dumps(response({}, True, {'role': 'assistant', 'content': ''})) + '\n\n'
                    # Completion/cancellation must wake an empty consumer even
                    # when cancellation stops the producer from adding chunks.
                    while not outcome.done() or not chunks.empty():
                        if not chunks.empty():
                            text = chunks.get_nowait()
                        else:
                            available = asyncio.create_task(chunks.get())
                            try:
                                await asyncio.wait({available, outcome}, return_when=asyncio.FIRST_COMPLETED)
                                if not available.done():
                                    continue
                                text = available.result()
                            finally:
                                if not available.done():available.cancel()
                                with anyio.CancelScope(shield=True):
                                    await _settled(asyncio.gather(available, return_exceptions=True))
                        yield 'data: ' + json.dumps(response({}, True, {'content': text})) + '\n\n'
                    result, exc = outcome.result()
                    if exc is not None:
                        yield 'data: ' + json.dumps({'error': {'message': str(exc), 'type': 'inference_error'}}) + '\n\n'
                    else:
                        value = response(result, True, {})
                        yield 'data: ' + json.dumps(value) + '\n\n'
                        if payload.stream_options and payload.stream_options.get('include_usage'):
                            value['choices'], value['usage'] = [], result['usage']
                            yield 'data: ' + json.dumps(value) + '\n\n'
                    yield 'data: [DONE]\n\n'
                finally:
                    cancel.set()

            return StreamingResponse(loaded_events(), media_type='text/event-stream',
                                     headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

        if not payload.stream:
            return response(await asyncio.to_thread(engine.infer, payload.model, task, data))

        async def events():
            q, cancel = queue.Queue(maxsize=32), threading.Event()

            def put(item):
                while not cancel.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        return
                    except queue.Full:
                        continue

            def run():
                try:
                    result = engine.infer(payload.model, task, data, lambda text: put(("text", text)), cancel)
                    put(("done", result))
                except Exception as exc:
                    put(("error", str(exc)))

            worker = asyncio.create_task(asyncio.to_thread(run))
            try:
                if task == "chat":
                    yield "data: " + json.dumps(response({}, True, {"role": "assistant", "content": ""})) + "\n\n"
                while True:
                    try:
                        kind, item = await asyncio.to_thread(q.get, True, 0.25)
                    except queue.Empty:
                        continue
                    if kind == "text":
                        value = response({}, True, {"content": item})
                    elif kind == "error":
                        yield "data: " + json.dumps({"error": {"message": item, "type": "inference_error"}}) + "\n\n"
                        break
                    else:
                        value = response(item, True, {})
                    yield "data: " + json.dumps(value) + "\n\n"
                    if kind == "done":
                        if payload.stream_options and payload.stream_options.get("include_usage"):
                            value["choices"], value["usage"] = [], item["usage"]
                            yield "data: " + json.dumps(value) + "\n\n"
                        break
                yield "data: [DONE]\n\n"
            finally:
                cancel.set()
                # The generation thread observes cancellation; its lock protects
                # weights until generation exits, even after a client disconnects.
                await asyncio.shield(worker)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/chat/completions")
    async def chat(payload: Chat, request: Request, response: Response = None):
        from .request_compat import attach, sized_context;ignored = (*sized_context(config.models.get(payload.model), payload), *payload._ignored)
        return attach(await generate(payload, "chat", request), response, ignored)

    @app.post("/v1/completions")
    async def completion(payload: Completion, request: Request, response: Response = None):
        from .request_compat import attach, sized_context;ignored = (*sized_context(config.models.get(payload.model), payload), *payload._ignored);return attach(await generate(payload, "completion", request), response, ignored)

    @app.post('/v1/responses')
    async def responses(payload: Responses, request: Request, response: Response = None):
        from .request_compat import attach
        return attach(await responses_answer(payload, request), response, payload._ignored)

    async def responses_answer(payload: Responses, request: Request):
        if payload.model in native.models:
            payload = model_defaults(payload)
            from .responses import native_stream;return native_stream(await native.proxy(request,'/v1/responses',payload.model_dump(exclude_none=True)),payload.model)
        from .responses import chat_request, make_response, adapt_response
        spec = config.models.get(payload.model)
        chat_payload = model_defaults(chat_request(payload, spec))
        if spec is not None and spec.task != 'vision' and any(
                isinstance(part, ImagePart) for message in chat_payload.messages
                if isinstance(message.content, list) for part in message.content):
            raise EngineError('Select a vision model for image input', 400, 'unsupported_capability')
        response = make_response(payload, chat_payload)
        return adapt_response(await generate(chat_payload, 'chat', request), response)

    @app.post("/v1/embeddings")
    async def embeddings(payload: Embeddings, request: Request, response: Response = None):
        from .request_compat import attach
        if payload.model in native.models:  # a GGUF embedding model: llama-server /v1/embeddings (openai_params.native_embeddings names the Titanos-only fields it refuses)
            return attach(await native.proxy(request, '/v1/embeddings', openai_params.native_embeddings(payload)), response, payload._ignored)
        data = payload.model_dump();data["input"] = [data["input"]] if isinstance(data["input"], str) else data["input"]
        from .embedding_output import format_embeddings
        return attach(format_embeddings(payload, await asyncio.to_thread(engine.infer, payload.model, "embeddings", data)), response, payload._ignored)

    def speech_defaults(payload):
        """model_defaults for speech. The Qwen fields (QWEN_SPEECH) and stream_format stay the client's: a qwen3-tts
        model applies its own sampling defaults, and another backend refuses a Qwen field by name."""
        spec = config.models.get(payload.model)
        named = [name for name in QWEN_SPEECH if getattr(payload, name) is not None]
        if named and spec is not None and spec.backend != 'qwen3-tts':
            raise EngineError(f"{', '.join(named)} apply to qwen3-tts speech models only", 400, 'unsupported_capability')
        from .request_defaults import model_defaults as apply_defaults
        return apply_defaults(payload, spec, exclude=frozenset(QWEN_SPEECH) | {'stream_format'})

    def speech_data(payload):
        """The engine's speech request: always a whole WAV (the route renders pcm and streams), with unset Qwen fields
        left out so other backends get the request they always did."""
        data = payload.model_dump(exclude={'task', 'stream_format'})
        data['response_format'] = 'wav'
        return {key: value for key, value in data.items() if value is not None or key not in QWEN_SPEECH}

    def complete_speech(payload: Speech):
        """The whole synthesis as one body: the speech route without stream_format, and ElevenLabs' route."""
        from .speech_stream import complete
        payload = speech_defaults(payload)
        return complete(engine.infer(payload.model, 'speech', speech_data(payload)), payload.response_format)

    @app.post("/v1/audio/speech")
    async def speech(payload: Speech, request: Request):
        """OpenAI speech: the whole body, or with stream_format audio the audio as it is made (speech_stream.py). sse is
        a cloud speech format the broker relays; a local model refuses it by name."""
        if payload.stream_format == 'sse':
            raise EngineError('stream_format sse is relayed for cloud speech models only; local speech streams with '
                              'stream_format audio', 400, 'unsupported_capability')
        if payload.stream_format is None:
            return await asyncio.to_thread(complete_speech, payload)
        from .speech_stream import stream
        payload = speech_defaults(payload)
        return await stream(engine, payload.model, speech_data(payload), payload.response_format, request)

    def artifact(suffix):
        # ponytail: local files with a 24-hour TTL; add durable object storage only
        # for a deployment that needs shared artifacts or retention guarantees.
        for old in outputs.iterdir():
            if re.fullmatch(r"[a-f0-9]{32}\.(png|mp4|wav)", old.name):
                try:
                    if old.stat().st_mtime < time.time() - settings.artifact_ttl_seconds:
                        old.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
        path = outputs / (uuid.uuid4().hex + suffix)
        bind_artifact(path.name)
        return path

    def bind_artifact(name, owner=None):
        """Bind an artifact to the key that produced it (the request's principal, or a job record's owner_key_id), so
        GET /v1/files/{name} answers that key (and admin, bootstrap and keyless principals) only. Keyless and bootstrap
        producers bind nothing (no file is written for them)."""
        from .broker.context import current_principal
        subject = owner if owner is not None else current_principal()
        key_id = subject if isinstance(subject, str) else getattr(subject, 'key_id', None)
        if key_id:
            broker.ownership.bind('artifact', name, key_id)

    def image_context(payload):
        from .api_plus import ExtensionPayload
        return ExtensionPayload(model=payload.model, worker=payload.worker)

    def image_boundary(cancel, deadline):
        if cancel is not None and cancel.is_set():
            raise EngineError('Image request cancelled',499,'cancelled')
        if deadline is not None and time.monotonic() >= deadline:
            raise EngineError('Image request deadline exceeded',504,'deadline_exceeded')

    def discard_artifacts(paths):
        errors=[]
        for path in paths:
            if path.parent != outputs or not re.fullmatch(r'[a-f0-9]{32}\.(png|mp4|wav)',path.name):
                errors.append('Refused cleanup of an unowned artifact path')
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f'{path.name}: {exc}')
        if errors:
            raise OSError('; '.join(errors))

    def image_result(payload, request, task="image", extra=None, cancel=None, deadline=None, owned_artifacts=None, masking=None):
        deadline = deadline if deadline is not None else time.monotonic() + settings.client_timeout
        image_boundary(cancel, deadline)
        from .instruction_edit import declared
        if task!='image-edit' and declared(config.models.get(payload.model)):
            raise EngineError('InstructPix2Pix requires image editing with a source image',400,'unsupported_capability')
        payload=model_defaults(payload)
        data = payload.model_dump()
        data["width"], data["height"] = map(int, data["size"].split("x"))
        data.update(extra or {})
        images = app.state.inference_run(image_context(payload), task, data, cancel, deadline)
        results, created = [], []
        try:
            try:
                if masking is not None and (not isinstance(images,list) or len(images) != data['n']):
                    raise EngineError('Inpainting returned an invalid image count',502,'invalid_backend_response')
                for im in images:
                    image_boundary(cancel, deadline)
                    if masking is not None:
                        from .image_mask import finish
                        finish(im, extra['image'], extra['mask_image'], masking['preserve_unmasked'])
                        image_boundary(cancel, deadline)
                    if payload.response_format == "url":
                        dest = artifact(".png"); created.append(dest)
                        if owned_artifacts is not None:owned_artifacts.append(dest)
                        im.save(dest)
                        results.append({"url": link_base(request) + "v1/files/" + dest.name})
                    else:
                        buf = io.BytesIO()
                        im.save(buf, format="PNG")
                        results.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
            finally:
                primary = sys.exc_info()[1]
                failures = []
                outputs = images if masking is None or isinstance(images, (list, tuple)) else (images,)
                for im in outputs:
                    try:
                        getattr(im, 'close', lambda: None)()
                    except BaseException as exc:
                        failures.append(exc)
                if failures:
                    note = 'Image output cleanup failed: ' + '; '.join(str(exc) for exc in failures)
                    if primary is not None:
                        primary.add_note(note)
                        logging.getLogger(__name__).error(note)
                    else:
                        for remaining in failures[1:]: failures[0].add_note(str(remaining))
                        raise failures[0]
            image_boundary(cancel, deadline)
            return {"created": int(time.time()), "data": results, **({'masking':masking} if masking is not None else {})}
        except BaseException:
            if owned_artifacts is None:discard_artifacts(created)
            raise

    @app.post("/v1/images/generations")
    async def images(payload: ImageRequest, request: Request):
        return await app.state.dispatch_cancellable(request, image_context(payload),
            lambda current, cancel, deadline: image_result(payload, request, cancel=cancel, deadline=deadline))

    IMAGE_EDIT_FORM_FIELDS = frozenset({'model', 'prompt', 'image', 'size', 'n', 'worker', 'steps', 'seed', 'negative_prompt', 'guidance_scale',
                                        'image_guidance_scale', 'response_format', 'strength', 'mask', 'mask_base64', 'preserve_unmasked'})

    @app.post("/v1/images/edits")
    async def edit(request: Request, model: Annotated[str, Form()], prompt: Annotated[str, Form()],
             image: Annotated[UploadFile, File()], size: Annotated[str | None, Form()] = None,
             n: Annotated[int | None, Form(ge=1)] = None,
             worker: Annotated[str | None, Form(max_length=128)] = None,
             steps: Annotated[int | None, Form(ge=1)] = None,
             seed: Annotated[int | None, Form(ge=0, le=2**32-1)] = None,
             negative_prompt: Annotated[str | None, Form()] = None,
             guidance_scale: Annotated[float | None, Form(ge=0)] = None,
             image_guidance_scale: Annotated[float | None,Form(ge=1)] = None,
             response_format: Annotated[Literal['b64_json', 'url'] | None, Form()] = None,
             strength: Annotated[float | None, Form(gt=0, le=1)] = None,
             mask: Annotated[UploadFile | None, File()] = None,
             mask_base64: Annotated[str | None, Form()] = None,
             preserve_unmasked: Annotated[str | None, Form()] = None):
        from .backends import image_input
        from pydantic import ValidationError
        form = await request.form()
        if 'mask_image' in form:
            raise EngineError('Use mask or mask_base64; mask_image is not a public field',400,'unsupported_capability')
        unknown = sorted(set(form.keys()) - IMAGE_EDIT_FORM_FIELDS)
        if unknown:
            # Undeclared multipart knobs (quality, background, output_format, input_fidelity, stream, ...) are refused, not dropped.
            raise EngineError('Unsupported multipart fields: ' + ', '.join(unknown) + '; supported: ' + ', '.join(sorted(IMAGE_EDIT_FORM_FIELDS)),400,'unsupported_capability')
        if any(len(form.getlist(key)) > 1 for key in ('image','mask','mask_base64','preserve_unmasked','n','image_guidance_scale')):
            raise EngineError('Image, mask, preservation and count fields must not be repeated')
        if any(key in form and value is None for key,value in (('mask',mask),('mask_base64',mask_base64),('preserve_unmasked',preserve_unmasked),('n',n),('image_guidance_scale',image_guidance_scale))):
            raise EngineError('Mask, preservation and count fields must not be empty')
        if mask is not None and mask_base64 is not None:
            raise EngineError('Use one mask upload or mask_base64, not both')
        masked = mask is not None or mask_base64 is not None
        if preserve_unmasked is not None and (not masked or preserve_unmasked not in {'true','false'}):
            raise EngineError('preserve_unmasked requires a mask and the literal true or false')
        try:
            values={'model':model,'prompt':prompt,'worker':worker}
            if n is not None:values['n']=n
            if size is not None:values['size']=size
            if steps is not None:values['steps']=steps
            for key,value in {'seed':seed,'negative_prompt':negative_prompt,'guidance_scale':guidance_scale,'response_format':response_format}.items():
                if value is not None:values[key]=value
            payload = ImageRequest(**values)
        except (ValueError, ValidationError) as exc:
            raise EngineError(str(exc)) from exc
        spec=config.models.get(model)
        from .instruction_edit import declared,validate_spec,guidance
        instruction=declared(spec)
        if instruction:
            validate_spec(spec)
            if {'width','height'} & form.keys():
                raise EngineError('InstructPix2Pix uses the source image size; omit width/height and resize the source before editing',400,'unsupported_capability')
            if masked or 'strength' in form:
                raise EngineError('InstructPix2Pix does not support masks or strength',400,'unsupported_capability')
        if image_guidance_scale is not None:
            if not instruction:
                raise EngineError('image_guidance_scale requires InstructPix2Pix',400,'unsupported_capability')
            guidance(image_guidance_scale)
        if masked and spec and spec.backend not in {'diffusers','openvino-genai'}:
            raise EngineError('This backend does not support inpainting',400,'unsupported_capability')
        preserve = (preserve_unmasked == 'true') if preserve_unmasked is not None else (spec.defaults.get('preserve_unmasked',settings.preserve_unmasked) if spec else settings.preserve_unmasked)
        if masked and type(preserve) is not bool:
            raise EngineError('preserve_unmasked must resolve to a boolean')
        if not instruction:
            if strength is None:strength=(spec.defaults.get("strength",settings.default_edit_strength) if spec else settings.default_edit_strength)
            if not isinstance(strength,(float,int)) or not 0 < strength <= 1:raise EngineError("Image edit strength must be greater than 0 and at most 1")
        def perform(current, cancel, deadline):
            image_boundary(cancel, deadline)
            im = canonical_mask = None
            owned = []
            try:
                raw = image.file.read(settings.max_input_image_bytes + 1)
                if len(raw) > settings.max_input_image_bytes:
                    raise EngineError('Source exceeds the configured image byte limit',413,'request_too_large')
                metadata = None
                current_payload = payload
                try:
                    if masked:
                        from .image_mask import decode, inline_mask, validate
                        mask_raw = mask.file.read(settings.max_input_image_bytes + 1) if mask is not None else inline_mask(mask_base64,settings.max_input_image_bytes)
                        im, canonical_mask, metadata = decode(raw,mask_raw,settings)
                        if size is not None and size != f'{im.width}x{im.height}':
                            raise EngineError('Masked edit size must match source dimensions exactly')
                        # Explicit geometry prevents model/process image-size defaults resizing the mask.
                        current_payload = ImageRequest(**{**values,'size':f'{im.width}x{im.height}'})
                        validate(canonical_mask,im,(im.width,im.height))
                        metadata.update(preserve_unmasked=preserve,
                            preservation_source='request' if preserve_unmasked is not None else 'model' if spec and 'preserve_unmasked' in spec.defaults else 'process',
                            postprocessing='copy_unmasked_rgb' if preserve else 'none')
                    else:
                        im = image_input("data:image/png;base64," + base64.b64encode(raw).decode())
                        if instruction:
                            if size is not None and size!=f'{im.width}x{im.height}':
                                raise EngineError('Instruction-edit size must match the source; resizing is not supported')
                            current_payload=ImageRequest(**{**values,'size':f'{im.width}x{im.height}'})
                except ValueError as exc:
                    image_boundary(cancel,deadline)
                    raise EngineError(str(exc)) from exc
                image_boundary(cancel, deadline)
                extra = {'image':im} if instruction else {'image':im,'strength':strength}
                if image_guidance_scale is not None:extra['image_guidance_scale']=image_guidance_scale
                if canonical_mask is not None:extra['mask_image']=canonical_mask
                try:
                    result = image_result(current_payload,request,'image-edit',extra,cancel,deadline,owned,metadata)
                finally:
                    try:
                        if canonical_mask is not None:canonical_mask.close();canonical_mask=None
                    finally:
                        if im is not None:im.close();im=None
                image_boundary(cancel,deadline)
                return result
            except BaseException:
                discard_artifacts(owned)
                raise
            finally:
                try:
                    if canonical_mask is not None:canonical_mask.close()
                finally:
                    if im is not None:im.close()
        return await app.state.dispatch_cancellable(request, image_context(payload), perform)

    def make_video(payload, base_url, cancel=None, owned_artifacts=None):
        from .media import write_video
        payload=model_defaults(payload)
        data = payload.model_dump()
        data["width"], data["height"] = map(int, data["size"].split("x"))
        if settings.video_codec=="h264_mf" and min(data["width"],data["height"])<64:
            raise EngineError("Windows Media Foundation H264 requires video dimensions of at least 64 pixels")
        frames = engine.infer(payload.model, "video", data, cancel=cancel)
        if cancel and cancel.is_set():
            raise EngineError('Video cancelled',499,'cancelled')
        dest = artifact(".mp4")
        if owned_artifacts is not None:owned_artifacts.append(dest)
        diagnostic = None
        try:
            if settings.video_diagnostics_dir:
                from .video_diagnostics import VideoDiagnostics
                diagnostic = VideoDiagnostics(settings, dest.stem, payload.model, payload.fps, len(frames))
                diagnostic.capture(frames, cancel)
                if cancel and cancel.is_set():
                    raise EngineError('Video cancelled',499,'cancelled')
            write_video(frames, dest, fps=payload.fps)
            if cancel and cancel.is_set():
                raise EngineError('Video cancelled',499,'cancelled')
            if diagnostic is not None:diagnostic.finish('encoded', video=dest)
        except BaseException as exc:
            if diagnostic is not None:
                try:diagnostic.finish('cancelled' if cancel and cancel.is_set() else 'failed', error=exc)
                except Exception as detail:
                    exc.add_note('Video diagnostic finalization failed: '+str(detail))
            dest.unlink(missing_ok=True)
            raise
        return {"id": dest.stem, "object": "video", "model": payload.model, "status": "completed",
                "url": base_url + "v1/files/" + dest.name, "fps": payload.fps, "frames": len(frames)}

    def local_admission(model,request,validate):
        """A key's constraints and local rate limits on routes the broker does not intercept (asset jobs, video and
        music): while the broker is active for the principal, a registered local model the key may not use is 404
        model_not_found (engine.validate's own answer for an unknown name, given before validate can reveal anything),
        then validate() runs and the Limiter admits the request (RPM; the concurrency slot is held until the caller
        releases the returned Admission). None while the broker is inactive (the unchanged path) or for names that are
        not local models (validate() answers them)."""
        principal=principal_of(request)
        target=broker.snapshot().catalog.local_target(model) if broker.active(principal) else None
        if target is None:
            validate()
            return None
        from .broker.limits import Estimate
        from .broker.scopes import can_use_target
        if not can_use_target(principal or broker.default_principal(),target):
            # exactly engine.validate's answer for an unknown name, so a key learns nothing about models it may not use
            raise EngineError(f"Unknown model {model!r}",404,"model_not_found")
        validate()
        return broker.limiter.admit(principal or broker.default_principal(),target,Estimate(max_output_tokens=0))

    @app.post("/v1/videos/generations")
    def video(payload: VideoRequest, request: Request):
        admission=local_admission(payload.model,request,lambda:None)
        try:
            return make_video(payload,link_base(request))
        finally:
            if admission is not None:admission.release()

    @app.post('/api/video/jobs',status_code=202)
    def video_job(payload: VideoRequest,request: Request):
        payload=model_defaults(payload)
        admission=local_admission(payload.model,request,lambda:engine.validate(payload.model,'video'))
        owned=[]
        return submit_job(payload,request,'video',lambda cancel:make_video(payload,link_base(request),cancel,owned),owned,
                          admission=admission)

    def submit_job(payload,request,task,generate,owned_artifacts=None,decision_id=None,on_refused=None,admission=None):
        key=request.headers.get('idempotency-key')
        if key is not None and (not key.strip() or len(key)>128):
            raise EngineError('Idempotency-Key must contain 1 to 128 characters')
        fingerprint=hashlib.sha256(json.dumps({'task':task,**payload.model_dump()},sort_keys=True).encode()).hexdigest()
        owner=getattr(principal_of(request),'key_id',None)

        def owned_generate(cancel):
            # job artifacts are bound to the job owner at completion (from the record's owner, not the thread context)
            result=generate(cancel)
            for path in owned_artifacts or ():
                bind_artifact(path.name, owner)
            return result
        # an Admission (a key's local rate limits, or a cloud job's budget reservation) is released exactly when the job
        # can no longer run: refused, replayed from its Idempotency-Key, or settled (finished, failed, cancelled)
        release=admission.release if admission is not None else None
        try:
            return jobs.submit(payload.model,owned_generate if owner else generate,task,
                               hashlib.sha256(key.encode()).hexdigest() if key else None,fingerprint,
                               cleanup=(lambda:discard_artifacts(owned_artifacts)) if owned_artifacts is not None else None,
                               owner_key_id=owner,decision_id=decision_id,on_settled=release)
        except BaseException:
            if on_refused is not None:on_refused()
            if release is not None:release()
            raise

    JobStatus=Literal['queued','running','cancelling','cancelled','completed','failed','interrupted']

    def job_list(request,task,limit,status,idempotency_key):
        # idempotency_key finds the caller's own job by the Idempotency-Key it submitted with (stored as its sha256)
        request_key=hashlib.sha256(idempotency_key.encode()).hexdigest() if idempotency_key is not None else None
        return jobs.list(task,limit,principal=principal_of(request),status=status,request_key=request_key)

    @app.get('/api/video/jobs')
    def video_job_list(request: Request, limit: int = Query(default=100,ge=1,le=100), status: JobStatus | None = None,
                       idempotency_key: str | None = Query(default=None,min_length=1,max_length=128)):
        return job_list(request,'video',limit,status,idempotency_key)

    @app.get('/api/jobs')
    def asset_job_list(request: Request, limit: int = Query(default=100,ge=1,le=100), status: JobStatus | None = None,
                       idempotency_key: str | None = Query(default=None,min_length=1,max_length=128),
                       task: Literal['image','speech','video','music'] | None = None):
        return job_list(request,task,limit,status,idempotency_key)

    def cloud_job_stage(payload,request):
        """The StageTarget of a cloud speech or image job target, or None for local models (and while the broker is
        inactive). Raises the router's refusal before anything is queued."""
        principal=principal_of(request)
        if payload.model in config.models or not broker.active(principal):
            return None
        task={'speech':'speech','image':'image'}.get(payload.task)
        if task is None:  # video and music run on local models only (engine.validate answers other names)
            return None
        stage=broker.stage_router().resolve_stage(payload.model,task,location='auto',principal=principal)
        return stage if stage.kind=='cloud' else None

    def submit_cloud_job(payload,request,stage):
        """A cloud speech or image job: Idempotency-Key is required (matched per owner), the budget is reserved at submit
        and the job runs CloudInvoker.infer_sync on its worker thread (never inside Engine.lock). Artifacts are written
        as /v1/files .wav or .png files bound to the job owner."""
        from .broker.errors import BrokerError
        from .broker.limits import Estimate
        if not (request.headers.get('idempotency-key') or '').strip():
            raise BrokerError('A cloud job needs an Idempotency-Key header','idempotency_key_required')
        principal=principal_of(request) or broker.default_principal()
        data=payload.model_dump(exclude={'task'})
        task=payload.task
        estimate=(Estimate(max_output_tokens=0,characters=len(data.get('input') or '')) if task=='speech' else
                  Estimate(max_output_tokens=0,images=int(data.get('n') or 1)))
        admission=broker.limiter.admit(principal,stage.target,estimate)
        decision_id=uuid.uuid4().hex
        owned=[]
        base_url=link_base(request)

        def generate_cloud(cancel):
            try:
                result=broker.invoker().infer_sync(stage.target,task,data,cancel=cancel,principal=principal,
                                                   admission=admission,decision_id=decision_id)
            finally:
                admission.release()  # a no-op once infer_sync settled it
            if cancel.is_set():raise EngineError('Job cancelled',499,'cancelled')
            if task=='speech':
                wav,_=result
                dest=artifact('.wav');owned.append(dest);dest.write_bytes(wav)
                return {'url':base_url+'v1/files/'+dest.name,'format':'wav','model':payload.model}
            entries=[]
            for image in result:
                dest=artifact('.png');owned.append(dest);dest.write_bytes(image)
                entries.append({'url':base_url+'v1/files/'+dest.name})
            return {'created':int(time.time()),'data':entries,'model':payload.model}
        return submit_job(payload,request,task,generate_cloud,owned,decision_id=decision_id,admission=admission)

    @app.post('/api/jobs',status_code=202)
    def asset_job(payload: AssetJob,request: Request):
        payload=speech_defaults(payload) if payload.task=='speech' else model_defaults(payload)
        stage=cloud_job_stage(payload,request)  # the cloud branch runs before engine.validate
        if stage is not None:
            named=[name for name in QWEN_SPEECH if getattr(payload,name)is not None] if payload.task=='speech' else []
            if named:  # a cloud stage builds its own upstream body and would drop them
                raise EngineError(f"{', '.join(named)} apply to local qwen3-tts speech models only",400,'unsupported_capability')
            return submit_cloud_job(payload,request,stage)
        admission=local_admission(payload.model,request,lambda:engine.validate(payload.model,payload.task))
        data=speech_data(payload) if payload.task=='speech' else payload.model_dump(exclude={'task'})
        owned=[]
        def generate_asset(cancel):
            if payload.task=='video':
                return make_video(VideoRequest(**data),link_base(request),cancel,owned)
            if payload.task=='image':
                return image_result(ImageRequest(**{**data,'response_format':'url'}),request,cancel=cancel,owned_artifacts=owned)
            if payload.task=='music':
                result=engine.infer(payload.model,'music',data,cancel=cancel)
                if cancel.is_set():raise EngineError('Music generation cancelled',499,'cancelled')
                wav=result.pop('wav',None)
                if wav is None:
                    from .media import write_music_wav
                    wav=write_music_wav(result.pop('samples'),result['sample_rate'])
                with wave.open(io.BytesIO(wav),'rb') as stream:
                    result.update(channels=stream.getnchannels(),sample_rate=stream.getframerate(),
                                  duration=stream.getnframes()/stream.getframerate())
                dest=artifact('.wav');owned.append(dest);dest.write_bytes(wav)
                return {'url':link_base(request)+'v1/files/'+dest.name,'format':'wav','model':payload.model,
                        'task':'music',**result}
            wav=engine.infer(payload.model,'speech',data,cancel=cancel)
            if cancel.is_set():raise EngineError('Speech cancelled',499,'cancelled')
            dest=artifact('.wav')
            owned.append(dest)
            dest.write_bytes(wav)
            return {'url':link_base(request)+'v1/files/'+dest.name,'format':'wav','model':payload.model}
        return submit_job(payload,request,payload.task,generate_asset,owned,admission=admission)

    @app.post('/v1/music/generations',status_code=202)
    def music_job(payload:MusicRequest,request:Request):
        return asset_job(MusicJob(task='music',**payload.model_dump()),request)

    @app.get('/api/video/jobs/{identifier}')
    @app.get('/api/jobs/{identifier}')
    def video_job_status(identifier: str, request: Request):
        return jobs.get(identifier, principal=principal_of(request))

    @app.delete('/api/video/jobs/{identifier}')
    @app.delete('/api/jobs/{identifier}')
    def video_job_cancel(identifier: str, request: Request):
        return jobs.cancel(identifier, principal=principal_of(request))

    def video_view(record):
        state={'running':'in_progress','cancelling':'in_progress','interrupted':'failed','cancelled':'failed'}.get(record['status'],record['status'])
        return {'id':record['id'],'object':'video','model':record['model'],'status':state,
                'created_at':int(record['created_at']),'progress':100 if state=='completed' else 0,
                'error':{'message':record['error']} if record.get('error') else None,
                'titanos_status':record['status'],**({'url':record['result']['url']} if record.get('result',{}).get('url') else {})}

    @app.post('/v1/videos',status_code=202)
    async def standard_video(request: Request):
        if 'application/json' in request.headers.get('content-type',''):
            values=await request.json()
        else:
            form=await request.form()
            values=dict(form)
        if not isinstance(values,dict):raise EngineError('Video request must be an object')
        if 'image' in values:
            if 'input_reference' in values:raise EngineError('Use input_reference or image, not both')
            values['input_reference']=values.pop('image')
        reference=values.get('input_reference')
        if reference is not None and hasattr(reference,'read'):
            raw=await reference.read()
            if len(raw)>settings.max_input_image_bytes:raise EngineError('Reference image is too large',413)
            values['input_reference']='data:image/png;base64,'+base64.b64encode(raw).decode()
        if isinstance(values.get('pipeline_options'),str):
            try:values['pipeline_options']=json.loads(values['pipeline_options'])
            except ValueError as exc:raise EngineError('pipeline_options must be a JSON object') from exc
        seconds=values.pop('seconds',None)
        if seconds is not None:
            try:
                duration=float(seconds)
                spec=config.models.get(values.get('model'))
                fps=int(values.get('fps',spec.defaults.get('fps',settings.default_video_fps) if spec else settings.default_video_fps))
                frames=round(duration*fps)
                if duration<=0 or frames<1:raise ValueError()
            except (ValueError,TypeError,OverflowError):raise EngineError('seconds must be a positive finite duration')
            if 'frames' in values:raise EngineError('Use seconds or frames, not both')
            values['frames']=frames
        try:payload=model_defaults(VideoRequest(**values))
        except ValueError as exc:raise EngineError(str(exc)) from exc
        return video_view(video_job(payload,request))

    @app.get('/v1/videos')
    def standard_video_list(request: Request, limit: int=Query(default=100,ge=1,le=100)):
        return {'object':'list','data':[video_view(r) for r in jobs.list('video',limit,principal=principal_of(request))['data']],'has_more':False}

    @app.get('/v1/videos/{identifier}/content')
    def standard_video_content(identifier: str, request: Request):
        record=jobs.get(identifier, principal=principal_of(request))
        if record.get('task','video')!='video':raise EngineError('Video not found',404,'not_found')
        if record['status']!='completed':raise EngineError('Video is not completed',409,'video_not_ready')
        return artifact_file(record['result']['url'].rsplit('/',1)[-1])

    @app.get('/v1/videos/{identifier}')
    def standard_video_get(identifier: str, request: Request):
        record=jobs.get(identifier, principal=principal_of(request))
        if record.get('task','video')!='video':raise EngineError('Video not found',404,'not_found')
        return video_view(record)

    @app.delete('/v1/videos/{identifier}')
    def standard_video_delete(identifier: str, request: Request):
        record=jobs.get(identifier, principal=principal_of(request))
        if record.get('task','video')!='video':raise EngineError('Video not found',404,'not_found')
        return jobs.delete(identifier, principal=principal_of(request))

    @app.get("/v1/files/{name}")
    def files(name: str, request: Request):
        principal=principal_of(request)
        if (re.fullmatch(r"[a-f0-9]{32}\.(png|mp4|wav)", name) and principal is not None
                and not broker.ownership.allows(principal,'artifact',name)):
            # artifacts bound to a key are visible to that key (and admin, bootstrap, keyless) only; never 403
            raise EngineError("File not found", 404, "not_found")
        return artifact_file(name)

    def artifact_file(name):
        if not re.fullmatch(r"[a-f0-9]{32}\.(png|mp4|wav)", name):
            raise EngineError("File not found", 404, "not_found")
        path = outputs / name
        if not path.is_file() or path.is_symlink() or path.stat().st_mtime < time.time() - settings.artifact_ttl_seconds:
            raise EngineError("File not found or expired", 404, "not_found")
        return FileResponse(path)

    def mcp_dispatch(inner):
        """The ASGI app mounted MCP tools dispatch into (spec MOUNTED MCP; threat T7). Each in-process call (a
        titanos.internal scope, which a client cannot forge) runs as the authenticated principal of the outer /mcp
        request: a principal already on the scope is kept, otherwise the one the Guard bound for the outer request
        (broker.context.PRINCIPAL; mcp 2.2 runs a tool call inside that request's context) is attached. The
        Authorization header is removed, so the bootstrap key mcp_server.py sends is never what authorizes a call that
        arrived over /mcp, and every inner route still needs its own scope. The Guard binds a principal for every
        request that reaches /mcp, so only in-process callers with no HTTP request at all (an embedder or a test
        calling app.state.mcp.call_tool directly) arrive without one; they keep today's semantics (the key create_mcp
        was given). WP8's ctx-based propagation in mcp_server.py replaces this interim."""
        from .broker.context import PRINCIPAL as principal_var
        from .broker.types import Principal

        async def dispatch(scope, receive, send):
            if scope.get('type') != 'http' or not scope.get('titanos.internal'):
                return await inner(scope, receive, send)
            principal = scope.get('titanos.principal')
            if not isinstance(principal, Principal):
                principal = principal_var.get()
            if not isinstance(principal, Principal):
                return await inner(scope, receive, send)
            headers = [(k, v) for k, v in scope.get('headers', []) if k.lower() != b'authorization']
            await inner({**scope, 'headers': headers, 'titanos.principal': principal}, receive, send)
        return dispatch

    from .api_plus import register_api_plus
    register_api_plus(app,engine,config,chat_defaults=model_defaults,native_gateway=lambda: app.state.native,
                      broker=broker)
    from .protocols import register_protocols
    register_protocols(app,config,engine,generate,embeddings,complete_speech,broker=broker)
    from .broker.routes import register_broker_routes
    register_broker_routes(app,broker,engine=engine,native_getter=lambda: app.state.native,config=config,gate=gate)
    from .broker.admin import register_admin_routes
    register_admin_routes(app,broker)
    try:
        from .mcp_server import create_mcp
    except ImportError:
        logging.info('MCP dependency is not installed; install titanos-engine[mcp] to enable /mcp')
    else:
        import inspect
        # the broker is passed once mcp_server.create_mcp accepts it (principal propagation, broker tools)
        extra={'broker':broker} if 'broker' in inspect.signature(create_mcp).parameters else {}
        app.state.mcp=create_mcp(app=mcp_dispatch(app),api_key=api_key,**extra)
        app.mount('/mcp',app.state.mcp.streamable_http_app(streamable_http_path='/',max_request_body_size=settings.max_body_bytes))
    return app

BUDGET_HEADER = "x-request-budget-ms"
BUDGET_MAX_MS = 600000
FORWARDED_HOST = re.compile(r"(?:[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?|\[[0-9A-Fa-f:.]{2,45}\])(?::[0-9]{1,5})?")


def request_deadline(request):
    """The monotonic time a request with X-Request-Budget-Ms must be answered by (its arrival at the Guard plus the
    budget), or None without the header. The value is ASCII digits alone, 1 to 600000 ms, named once (as the cluster
    facade checks it); anything else is 400 budget_invalid."""
    values = request.headers.getlist(BUDGET_HEADER)
    if not values:
        return None
    value = values[0].strip()
    if len(values) > 1 or not re.fullmatch(r"[0-9]{1,6}", value) or not 1 <= int(value) <= BUDGET_MAX_MS:
        raise EngineError(f"X-Request-Budget-Ms must be one whole number of milliseconds from 1 to {BUDGET_MAX_MS}",
                          400, "budget_invalid")
    return request.scope.get("titanos.arrived", time.monotonic()) + int(value) / 1000


def within_budget(request, run):
    """run(cancel) within the request's budget: without X-Request-Budget-Ms, run(None) as before; with it, a cancel
    event set when the budget passes, so a request still waiting for the model or a worker is refused unrun, and a
    cancellation after the budget is answered 504 budget_passed."""
    deadline = request_deadline(request)
    if deadline is None:
        return run(None)
    cancel = threading.Event()
    timer = threading.Timer(max(0.0, deadline - time.monotonic()), cancel.set)
    timer.daemon = True
    timer.start()
    try:
        if time.monotonic() >= deadline:
            cancel.set()
        return run(cancel)
    except EngineError as exc:
        if exc.code == "cancelled" and time.monotonic() >= deadline:
            raise EngineError("The request's budget passed before it was answered", 504, "budget_passed") from exc
        raise
    finally:
        timer.cancel()


def link_base(request):
    """Where the links to this server's files start (ending in /). A loopback peer (the cluster facade beside this
    engine) that sends X-Forwarded-Host gets links on that host, with X-Forwarded-Proto's http or https (else this
    request's scheme), and a relative /... path when the host is malformed, so a link never names this engine's own
    loopback address to a client of the facade. Without the header, or from any other peer, the address the client
    asked is used, as before."""
    import ipaddress
    path = urlsplit(str(request.base_url)).path
    peer = (request.scope.get("client") or ("",))[0]
    try:
        loopback = ipaddress.ip_address(peer).is_loopback
    except ValueError:
        loopback = False
    hosts = request.headers.getlist("x-forwarded-host")
    if not loopback or not hosts:
        return str(request.base_url)
    host = hosts[0].split(",")[0].strip()
    if not FORWARDED_HOST.fullmatch(host):
        return path
    proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return f"{proto if proto in ('http', 'https') else request.url.scheme}://{host}{path}"
