"""MockServer (a uvicorn thread on 127.0.0.1 with a free port) and BaseMock, the hit-recording, auth-asserting,
fault-injecting ASGI base every provider mock extends.

MockServer(app).start() -> base_url binds 127.0.0.1:0 itself, serves the app on a dedicated thread with its own event loop
(the platform default: Proactor on Windows) and returns once uvicorn reports started; stop() releases held frames, asks
uvicorn to exit and joins the thread (both bounded). uvicorn is configured without logging configuration (log_config=None,
log_level=None), so starting a mock never touches the process's logging handlers or levels, and without lifespan.

BaseMock records every request as a RecordedRequest in mock.hits: method, path, query and headers with every credential
value replaced by '<sentinel:NAME>' (a SENTINELS value) or '<foreign:FP>' (anything else; FP = fingerprint(value)), the
SHA-256 of the body and the body bytes (test-side assertions such as parameter rewrites need them; bodies never carry
credentials unless the system under test leaks one, which is exactly what tests look for).

Every header occurrence is kept (hit.header_list, in arrival order); a repeated header shows in hit.headers as the
comma-joined spellings. Every occurrence of a credential header is redacted and fingerprinted (the whole token and each
piece split on whitespace, ',', ';' and '='). A credential header sent more than once is listed in
hit.duplicate_credentials and, on an authenticated route, refused with a 400 in the provider's native envelope before the
auth check, so a forwarded client credential can never hide behind an appended upstream one.
RecordedRequest.presented(value) searches the credential fingerprints, the path, every recorded header value, every query
value and the body.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import logging
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable
from urllib.parse import parse_qsl, urlencode

from .faults import STATUS_DEFAULT_RETRY_AFTER, Fault, FaultQueue, wait_until
from .fixtures import SENTINEL_NAMES, SENTINELS, dumps, load

CREDENTIAL_HEADERS = ('authorization', 'proxy-authorization', 'x-api-key', 'xi-api-key', 'api-key', 'x-goog-api-key',
                      'cookie')
CREDENTIAL_QUERY = ('key', 'api_key', 'access_token')
_SCHEME = re.compile(r'^(bearer|basic|token)\s+', re.IGNORECASE)
_PIECES = re.compile(r'[\s,;=]+')
START_TIMEOUT = 10.0
STOP_TIMEOUT = 10.0


def fingerprint(value: str) -> str:
    """12 hex chars of SHA-256 over a credential value, as the recorder stores foreign credentials."""
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]


def credential_fingerprints(value: str) -> set[str]:
    """Fingerprints of a credential value's token and of each piece of it (split on whitespace, ',', ';' and '='), so a
    token packed into a cookie or a comma-joined header value is still found by presented()."""
    match = _SCHEME.match(value)
    token = (value[match.end():] if match else value).strip()
    pieces = {piece for piece in _PIECES.split(value) if piece}
    if token:
        pieces.add(token)
    return {fingerprint(piece) for piece in pieces}


def redact_value(value: str) -> tuple[str, str | None]:
    """(recorded spelling, raw token or None): a scheme prefix is kept, the token becomes <sentinel:NAME>/<foreign:FP>."""
    match = _SCHEME.match(value)
    prefix, token = (value[:match.end()], value[match.end():]) if match else ('', value)
    token = token.strip()
    if not token:
        return value, None
    name = SENTINEL_NAMES.get(token)
    return prefix + (f'<sentinel:{name}>' if name else f'<foreign:{fingerprint(token)}>'), token


@dataclass
class RecordedRequest:
    method: str
    path: str
    query: str
    query_params: dict
    headers: dict
    body_sha256: str
    body: bytes
    surface: str | None = None
    credentials: list = field(default_factory=list)  # [(where, recorded spelling)] e.g. ('header:authorization', ...)
    fingerprints: set = field(default_factory=set)
    status: int | None = None
    fault: str | None = None
    frames_sent: int = 0
    completed: bool = False
    disconnected: bool = False
    aborted: bool = False
    ts: float = field(default_factory=time.time)
    header_list: list = field(default_factory=list)  # [(lower-case name, recorded value)] for every header occurrence
    query_pairs: list = field(default_factory=list)  # [(name, recorded value)] for every query parameter occurrence
    duplicate_credentials: list = field(default_factory=list)  # credential header names sent more than once

    def json(self) -> Any:
        return json.loads(self.body.decode('utf-8'))

    def presented(self, value: str) -> bool:
        """True when `value` was presented as a credential (any occurrence of a credential header or query parameter,
        or a piece of one) or appears in the path, in any other header value, in any query value or in the body."""
        if not value:
            return False
        if fingerprint(value) in self.fingerprints:
            return True
        if value in self.path or value.encode('utf-8') in self.body:
            return True
        return (any(value in recorded for _, recorded in self.header_list)
                or any(value in recorded for _, recorded in self.query_pairs))

    @property
    def credential(self) -> str | None:
        """The first recorded credential spelling, or None."""
        return self.credentials[0][1] if self.credentials else None


@dataclass
class Reply:
    status: int = 200
    headers: list = field(default_factory=list)  # [(name, value)]
    body: bytes = b''
    frames: list | None = None  # streamed answers: one chunk per frame (SSE frame, NDJSON line, binary chunk)
    stream_kind: str | None = None  # 'sse' | 'ndjson' | 'binary' for streamed answers

    @property
    def streaming(self) -> bool:
        return self.frames is not None

    def header(self, name: str) -> str | None:
        for key, value in self.headers:
            if key.lower() == name.lower():
                return value
        return None


def json_reply(body: Any, status: int = 200, headers: Iterable = ()) -> Reply:
    return Reply(status, [('content-type', 'application/json')] + list(headers), dumps(body).encode('utf-8'))


@dataclass
class MockRequest:
    method: str
    path: str
    raw_query: str
    query: list  # [(name, value)]
    headers: dict  # lower-case name -> value (last wins; header_list keeps every occurrence)
    body: bytes
    scope: dict
    path_params: dict = field(default_factory=dict)
    _form: Any = None
    header_list: list = field(default_factory=list)  # [(lower-case name, value)] in arrival order

    def duplicate_credential_headers(self) -> list[str]:
        """Credential header names that occur more than once, in first-repeat order."""
        seen, repeated = set(), []
        for name, _ in self.header_list:
            if name in CREDENTIAL_HEADERS:
                if name in seen and name not in repeated:
                    repeated.append(name)
                seen.add(name)
        return repeated

    def q(self, name: str, default: str | None = None) -> str | None:
        for key, value in self.query:
            if key == name:
                return value
        return default

    def json(self) -> Any:
        return json.loads(self.body.decode('utf-8')) if self.body else None

    async def form(self):
        """The parsed multipart or urlencoded form (starlette's parser over the buffered body)."""
        if self._form is None:
            from starlette.requests import Request
            sent = False

            async def receive():
                nonlocal sent
                if sent:
                    return {'type': 'http.disconnect'}
                sent = True
                return {'type': 'http.request', 'body': self.body, 'more_body': False}
            self._form = await Request(self.scope, receive).form()
        return self._form


class MockAbort(Exception):
    """Raised inside a streaming answer to drop the connection without a terminating chunk (disconnect_after_frames)."""


class _DropMockAbort(logging.Filter):
    def filter(self, record):
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, MockAbort) and not (exc is not None and isinstance(exc.__cause__, MockAbort))


_ABORT_FILTER = _DropMockAbort()


def _install_abort_filter() -> None:
    """Keep uvicorn's 'Exception in ASGI application' record for a deliberate MockAbort off stderr (idempotent; installed
    by MockServer.start, never at import)."""
    logger = logging.getLogger('uvicorn.error')
    if _ABORT_FILTER not in logger.filters:
        logger.addFilter(_ABORT_FILTER)


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class MockServer:
    """A uvicorn server for `app` on a loopback port, run on a dedicated thread."""

    def __init__(self, app, host: str = '127.0.0.1', port: int = 0):
        if not _is_loopback(host):
            raise ValueError('MockServer binds loopback IP literals only')
        self.app = app
        self.host = host
        self.port = port
        self._server = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._paused = False
        self._error: BaseException | None = None

    @property
    def base_url(self) -> str:
        return f'http://{self.host}:{self.port}'

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _bind(self) -> socket.socket:
        family = socket.AF_INET6 if ':' in self.host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        if sys.platform != 'win32':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError:
            if sys.platform != 'win32' or not self.port:
                sock.close()
                raise
            # re-binding a just-closed listener on Windows (connect_refused resume) while its accepted connections linger
            sock.close()
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
        sock.listen(128)
        sock.setblocking(False)
        return sock

    def start(self) -> str:
        if self.running:
            return self.base_url
        import uvicorn
        _install_abort_filter()
        sock = self._bind()
        self.port = sock.getsockname()[1]
        config = uvicorn.Config(self.app, log_config=None, log_level=None, access_log=False, lifespan='off', ws='none',
                                http='h11', loop='asyncio', proxy_headers=False, timeout_graceful_shutdown=1,
                                timeout_keep_alive=5)
        self._server = uvicorn.Server(config)

        def run():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._server.serve(sockets=[sock]))
            except BaseException as exc:  # surfaced by start()
                self._error = exc
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()

        self._thread = threading.Thread(target=run, name=f'broker-mock-{self.port}', daemon=True)
        self._thread.start()
        if not wait_until(lambda: self._server.started or self._error is not None or not self._thread.is_alive(),
                          START_TIMEOUT):
            raise RuntimeError('mock server did not start in time')
        if not self._server.started:
            raise RuntimeError(f'mock server failed to start: {self._error!r}')
        return self.base_url

    def stop(self) -> None:
        if self._server is None:
            return
        release = getattr(self.app, '_release_all', None)
        if release is not None:
            release()
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(STOP_TIMEOUT)
            if self._thread.is_alive():
                self._server.force_exit = True
                self._thread.join(STOP_TIMEOUT)
        self._server = None
        self._thread = None
        self._loop = None
        self._paused = False

    def _call(self, coroutine_factory: Callable[[], Awaitable], timeout: float = 5.0):
        if self._loop is None:
            raise RuntimeError('mock server is not running')
        return asyncio.run_coroutine_threadsafe(coroutine_factory(), self._loop).result(timeout)

    def pause_listening(self) -> None:
        """Close the listening socket: new connects are refused at TCP level; open connections continue."""
        if self._paused or self._server is None:
            return

        async def pause():
            for srv in self._server.servers:
                srv.close()
        self._call(pause)
        self._paused = True

    def resume_listening(self) -> None:
        """Listen again on the same port after pause_listening()."""
        if not self._paused or self._server is None:
            return
        factory = self._server.servers[0]._protocol_factory

        async def resume():
            sock = self._bind()
            srv = await asyncio.get_running_loop().create_server(factory, sock=sock, backlog=128)
            self._server.servers = [srv]
        self._call(resume)
        self._paused = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


Handler = Callable[['MockRequest'], Awaitable[Reply]]


@dataclass
class Route:
    method: str
    pattern: re.Pattern
    handler: Handler
    surface: str | None
    auth: bool = True


class BaseMock:
    """ASGI app base: routes, credential checks, the hit recorder, fault application and the frame emitter.

    Subclasses define PROVIDER (a SENTINELS key), add routes in __init__ via self.route(...), and implement
    auth_failure(request) -> Reply|None, error_reply(status, message, code=None) -> Reply and stream_error_frame(kind)
    -> bytes (the protocol's in-stream error event used by error_first_event and malformed_sse_line).
    """

    PROVIDER = ''
    PREFIX = ''  # the path prefix of the profile base_url (e.g. '/v1'); api_base = base_url + PREFIX

    def __init__(self, *, credential: str | None = None):
        self.credential = SENTINELS.get(self.PROVIDER) if credential is None else credential
        self.hits: list[RecordedRequest] = []
        self._hits_lock = threading.Lock()
        self.routes: list[Route] = []
        self.faults = FaultQueue()
        self.faults.on_push = self._on_push
        self.faults.on_resume = self._on_resume
        self.gate = threading.Event()
        self.gate.set()
        self.gate_timeouts = 0
        self.frame_delay = 0.0
        self.server = MockServer(self)
        self.sink: SinkMock | None = None
        self.counter = 0

    # lifecycle ----------------------------------------------------------------------------------------------------
    def start(self) -> str:
        return self.server.start()

    def stop(self) -> None:
        self.faults.clear()
        self.server.stop()
        if self.sink is not None:
            self.sink.stop()
            self.sink = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    @property
    def base_url(self) -> str:
        return self.server.base_url

    @property
    def api_base(self) -> str:
        """The base_url a broker upstream of this profile points at."""
        return self.server.base_url + self.PREFIX

    @property
    def origin(self) -> str:
        return f'http://127.0.0.1:{self.server.port}'

    def reset(self) -> None:
        """Forget hits, faults, counters and stored state (subclasses extend)."""
        with self._hits_lock:
            self.hits.clear()
        self.faults.clear()
        self.gate.set()
        self.gate_timeouts = 0
        self.counter = 0

    def _release_all(self) -> None:
        self.gate.set()

    def next_id(self) -> int:
        self.counter += 1
        return self.counter

    # routes -------------------------------------------------------------------------------------------------------
    def route(self, method: str, path_regex: str, handler: Handler, surface: str | None = None, *, auth: bool = True):
        self.routes.append(Route(method.upper(), re.compile('^' + path_regex + '$'), handler, surface, auth))

    def match(self, method: str, path: str) -> tuple[Route | None, dict, bool]:
        """(route, path params, path known for another method)."""
        known = False
        for route in self.routes:
            found = route.pattern.match(path)
            if found:
                if route.method == method:
                    return route, found.groupdict(), True
                known = True
        return None, {}, known

    # provider hooks -------------------------------------------------------------------------------------------------
    def auth_failure(self, request: MockRequest) -> Reply | None:
        return None

    def error_reply(self, status: int, message: str, code: str | None = None) -> Reply:
        return json_reply({'error': {'message': message, 'code': code}}, status)

    def stream_error_frame(self, message: str = 'Overloaded') -> bytes:
        return ('data: ' + dumps({'error': {'message': message, 'type': 'server_error'}}) + '\n\n').encode('utf-8')

    def malformed_frame(self, stream_kind: str) -> bytes:
        return b'{"not json\n' if stream_kind == 'ndjson' else b'data: {"not json\n\n'

    def duplicate_credential_reply(self, names: list[str]) -> Reply:
        """The refusal for a credential header sent more than once (never echoes a value)."""
        return self.error_reply(400, 'mock: credential header sent more than once: ' + ', '.join(names))

    def not_found(self, request: MockRequest) -> Reply:
        return self.error_reply(404, f'Unknown path {request.path}', 'not_found')

    # helpers for subclasses -----------------------------------------------------------------------------------------
    def fixture(self, case: str, provider: str | None = None):
        return load(provider or self.fixture_provider(), case)

    def fixture_provider(self) -> str:
        return self.PROVIDER

    def fixture_reply(self, case: str, variables: dict | None = None, *, provider: str | None = None,
                      variant: str | None = None, extra_headers: Iterable = ()) -> Reply:
        fx = self.fixture(case, provider)
        headers = list(fx.headers.items()) + list(extra_headers)
        frames = fx.frames(variables, variant=variant)
        if fx.kind in ('json', 'text'):
            return Reply(fx.status, headers, frames[0])
        if fx.kind == 'binary':
            content_type = fx.variants()[variant or fx.rows[0]['variant']]
            headers = [(k, v) for k, v in headers if k.lower() != 'content-type'] + [('content-type', content_type)]
        return Reply(fx.status, headers, frames=frames, stream_kind=fx.kind)

    def presented_token(self, request: MockRequest, header: str) -> str | None:
        value = request.headers.get(header)
        if value is None:
            return None
        match = _SCHEME.match(value)
        return value[match.end():].strip() if match else value.strip()

    # recorder -------------------------------------------------------------------------------------------------------
    def _record(self, request: MockRequest, surface: str | None) -> RecordedRequest:
        header_list, credentials, fps = [], [], set()
        for name, value in request.header_list or list(request.headers.items()):
            if name in CREDENTIAL_HEADERS:
                spelled, _ = redact_value(value)
                credentials.append(('header:' + name, spelled))
                fps |= credential_fingerprints(value)
                value = spelled
            header_list.append((name, value))
        headers: dict = {}
        for name, value in header_list:
            headers[name] = headers[name] + ', ' + value if name in headers else value
        query_pairs = []
        for name, value in request.query:
            if name.lower() in CREDENTIAL_QUERY:
                spelled, _ = redact_value(value)
                credentials.append(('query:' + name, spelled))
                fps |= credential_fingerprints(value)
                value = spelled
            query_pairs.append((name, value))
        hit = RecordedRequest(method=request.method, path=request.path, query=urlencode(query_pairs),
                              query_params=dict(query_pairs), headers=headers,
                              body_sha256=hashlib.sha256(request.body).hexdigest(), body=request.body, surface=surface,
                              credentials=credentials, fingerprints=fps, header_list=header_list,
                              query_pairs=query_pairs, duplicate_credentials=request.duplicate_credential_headers())
        with self._hits_lock:
            self.hits.append(hit)
        return hit

    def hits_for(self, surface: str | None = None, path: str | None = None) -> list[RecordedRequest]:
        with self._hits_lock:
            return [h for h in self.hits if (surface is None or h.surface == surface)
                    and (path is None or re.search(path, h.path))]

    # faults ---------------------------------------------------------------------------------------------------------
    def _on_push(self, fault: Fault) -> None:
        if fault.kind == 'hold_frame':
            self.gate.clear()
        elif fault.kind == 'redirect_foreign' and fault.get('target') is None:
            if self.sink is None:
                self.sink = SinkMock()
                self.sink.start()
        elif fault.kind == 'connect_refused':
            self.server.pause_listening()

    def _on_resume(self) -> None:
        if self.server.running:
            self.server.resume_listening()

    def hold(self, frame: int = 2, **kw) -> Fault:
        """Sugar for faults.push('hold_frame', frame=N); release with mock.gate.set()."""
        return self.faults.push('hold_frame', frame=frame, **kw)

    def _status_fault_reply(self, fault: Fault, request: MockRequest) -> Reply:
        status = int(fault.get('status'))
        message = fault.get('message') or {
            400: 'Invalid request', 401: 'Invalid authentication', 403: 'Forbidden', 404: 'Not found',
            429: 'Rate limit reached for requests', 500: 'The server had an error while processing your request',
            502: 'Bad gateway', 503: 'The engine is currently overloaded, please try again later', 529: 'Overloaded',
        }.get(status, 'Error')
        if fault.get('leak', status == 400):
            presented = next((t for t in (self.presented_token(request, h) for h in CREDENTIAL_HEADERS) if t), '')
            message = f'{message}; provided key {presented} rejected; trace {SENTINELS["leak"]}'
        reply = self.error_reply(status, message, fault.get('code'))
        retry_after = fault.get('retry_after', STATUS_DEFAULT_RETRY_AFTER.get(status))
        if retry_after is not None:
            reply.headers.append(('retry-after', str(retry_after)))
        return reply

    def _redirect_reply(self, fault: Fault, request: MockRequest) -> Reply:
        target = fault.get('target') or self.sink
        location = target.base_url + request.path + (('?' + request.raw_query) if request.raw_query else '')
        status = int(fault.get('status', 307))
        return Reply(status, [('location', location), ('content-type', 'text/plain')], b'redirect')

    # ASGI -----------------------------------------------------------------------------------------------------------
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return
        chunks = []
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunks.append(message.get('body', b''))
            if not message.get('more_body'):
                break
        body = b''.join(chunks)
        header_list = [(raw_name.decode('latin-1').lower(), raw_value.decode('latin-1'))
                       for raw_name, raw_value in scope.get('headers', [])]
        headers = dict(header_list)
        raw_query = scope.get('query_string', b'').decode('latin-1')
        request = MockRequest(method=scope['method'].upper(), path=scope['path'], raw_query=raw_query,
                              query=parse_qsl(raw_query, keep_blank_values=True), headers=headers, body=body, scope=scope,
                              header_list=header_list)
        route, params, known = self.match(request.method, request.path)
        request.path_params = params
        hit = self._record(request, route.surface if route else None)
        disconnected = asyncio.Event()

        async def watch():
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    disconnected.set()
                    return

        watcher = asyncio.ensure_future(watch())
        try:
            reply = await self._answer(request, route, known, hit)
            await self._emit(reply, request, route, hit, send, disconnected)
        finally:
            watcher.cancel()
            if request._form is not None:
                await request._form.close()

    async def _answer(self, request: MockRequest, route: Route | None, known: bool, hit: RecordedRequest) -> Reply:
        if route is None:
            if known:
                return self.error_reply(405, f'Method {request.method} not allowed', 'method_not_allowed')
            return self.not_found(request)
        if route.auth:
            if hit.duplicate_credentials:
                return self.duplicate_credential_reply(hit.duplicate_credentials)
            failure = self.auth_failure(request)
            if failure is not None:
                return failure
        fault = self.faults.take('request', request.path, route.surface)
        extra: list = []
        if fault is not None:
            hit.fault = fault.kind
            if fault.kind == 'slow_first_byte':
                await asyncio.sleep(float(fault.get('delay', 2.0)))
            elif fault.kind == 'status':
                return self._status_fault_reply(fault, request)
            elif fault.kind == 'malformed_json':
                return Reply(200, [('content-type', 'application/json')],
                             b'{"id":"mock-malformed","object":"chat.completion","choices":[{"index":0,"message":{"role"')
            elif fault.kind == 'oversized_body':
                size = int(fault.get('size', 2 * 1024 * 1024))
                payload = b'{"pad":"' + b'x' * max(0, size - 10) + b'"}'
                pieces = [payload[i:i + 65536] for i in range(0, len(payload), 65536)]
                headers = [('content-type', 'application/json')]
                if fault.get('declared_length'):
                    headers.append(('content-length', str(len(payload))))
                return Reply(200, headers, frames=pieces, stream_kind='binary')
            elif fault.kind == 'redirect_foreign':
                return self._redirect_reply(fault, request)
            elif fault.kind == 'set_cookie':
                cookie = fault.get('cookie', 'mock_session=mock-cookie-0001; Path=/; HttpOnly')
                extra = [('set-cookie', cookie), ('set-cookie', 'mock_tracker=1; Path=/'),
                         ('location', 'http://127.0.0.1:9/elsewhere')]
        try:
            reply = await route.handler(request)
        except _Refuse as refusal:
            reply = refusal.reply
        reply.headers.extend(extra)
        return reply

    async def _emit(self, reply: Reply, request: MockRequest, route: Route | None, hit: RecordedRequest, send,
                    disconnected: asyncio.Event) -> None:
        hit.status = reply.status
        headers = [(k.lower().encode('latin-1'), str(v).encode('latin-1')) for k, v in reply.headers]
        if not reply.streaming:
            if not any(k == b'content-length' for k, _ in headers):
                headers.append((b'content-length', str(len(reply.body)).encode()))
            await send({'type': 'http.response.start', 'status': reply.status, 'headers': headers})
            await send({'type': 'http.response.body', 'body': reply.body, 'more_body': False})
            hit.frames_sent = 1
            hit.completed = True
            return
        frames = list(reply.frames)
        fault = self.faults.take('stream', request.path, route.surface if route else None)
        hold_at, abort_after = None, None
        if fault is not None:
            hit.fault = fault.kind
            if fault.kind == 'error_first_event':
                frames = [self.stream_error_frame()]
            elif fault.kind == 'malformed_sse_line':
                at = max(1, int(fault.get('at', 1)))
                if at <= len(frames):
                    frames[at - 1] = self.malformed_frame(reply.stream_kind or 'sse')
            elif fault.kind == 'disconnect_after_frames':
                abort_after = max(0, int(fault.get('frames', 1)))
            elif fault.kind == 'hold_frame':
                hold_at = max(1, int(fault.get('frame', 2)))
        await send({'type': 'http.response.start', 'status': reply.status, 'headers': headers})
        for index, frame in enumerate(frames, start=1):
            if abort_after is not None and index > abort_after:
                hit.aborted = True
                raise MockAbort('disconnect_after_frames')
            if hold_at is not None and index == hold_at:
                deadline = time.monotonic() + float(fault.get('timeout', 30.0))
                while not self.gate.is_set() and not disconnected.is_set():
                    if time.monotonic() > deadline:
                        self.gate_timeouts += 1
                        break
                    await asyncio.sleep(0.005)
            elif self.frame_delay:
                await asyncio.sleep(self.frame_delay)
            if disconnected.is_set():
                hit.disconnected = True
                return
            try:
                await send({'type': 'http.response.body', 'body': frame, 'more_body': True})
            except OSError:
                hit.disconnected = True
                return
            hit.frames_sent = index
            await asyncio.sleep(0)
        if abort_after is not None:  # every frame went out; the terminating chunk never does
            hit.aborted = True
            raise MockAbort('disconnect_after_frames')
        await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
        hit.completed = True


class _Refuse(Exception):
    """Raised by handlers to answer early with a native error reply."""

    def __init__(self, reply: Reply):
        super().__init__(reply.status)
        self.reply = reply


def refuse(reply: Reply):
    raise _Refuse(reply)


class SinkMock(BaseMock):
    """A loopback catch-all that records any request and any credential it is sent (the redirect_foreign target)."""

    PROVIDER = 'sink'

    def __init__(self):
        super().__init__(credential='')

        async def anything(request: MockRequest) -> Reply:
            return json_reply({'sink': True, 'path': request.path})
        for method in ('GET', 'POST', 'PUT', 'DELETE', 'PATCH'):
            self.route(method, r'/.*', anything, None, auth=False)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


def unused_loopback_url() -> str:
    """http://127.0.0.1:<port> where nothing listens (the port was bound and released): connects are refused."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', 0))
        return f'http://127.0.0.1:{sock.getsockname()[1]}'
    finally:
        sock.close()


__all__ = ['BaseMock', 'CREDENTIAL_HEADERS', 'CREDENTIAL_QUERY', 'MockAbort', 'MockRequest', 'MockServer', 'RecordedRequest',
           'Reply', 'Route', 'SinkMock', 'b64', 'credential_fingerprints', 'fingerprint', 'json_reply', 'redact_value',
           'refuse', 'unused_loopback_url']
