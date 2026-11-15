"""Shared machinery for the provider adapters: sending through the EgressClient with typed failures, the held first frame
(pre-commit error detection), UpstreamResult assembly with the usage tee, JSON bodies, multipart forms, probes and
catalog paging. No adapter reads broker.json, the vault or the environment; each receives its profile row and upstream
record at construction and its authority per call from the EgressGrant."""
from __future__ import annotations

import asyncio
import fnmatch
import json
import time
from collections.abc import AsyncIterator, Mapping

import httpx

from ..errors import BrokerError
from ..profiles import TITANOS_ONLY_FIELDS, Profile
from ..sse import ManagedStream, first_frame, is_error_frame, parse_ndjson, parse_sse, tee_usage
from ..surfaces import SURFACES, Surface
from ..translate.common import aclose, bad_upstream, refuse
from ..translate.ir import ErrorEvent, IRRequest
from ..translate.upstream_errors import UpstreamFailure, classify, classify_stream_error, network_error
from ..types import EgressGrant, Usage
from .base import ProbeResult, ProviderAdapter, RawRequest, UpstreamResult

DEFAULT_MAX_TOKENS = 4096
CATALOG_MAX_PAGES = 20


def matches(model: str, glob: str) -> bool:
    return fnmatch.fnmatchcase(model or '', glob)


def strip_titanos(body: dict) -> dict:
    return {k: v for k, v in body.items() if k not in TITANOS_ONLY_FIELDS}


def json_body(raw: RawRequest) -> dict:
    """The inbound JSON object (parsed from raw bytes when needed); 400 for anything else."""
    body = raw.json
    if body is None and raw.raw is not None:
        try:
            body = json.loads(raw.raw.decode('utf-8')) if raw.raw.strip() else {}
        except (ValueError, UnicodeDecodeError):
            raise refuse('The request body is not valid JSON', 'invalid_request') from None
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise refuse('The request body must be a JSON object', 'invalid_request')
    return dict(body)


def split_form(form) -> tuple[dict, list]:
    """(data fields, files) for httpx from a list of (name, value) pairs; file values are (filename, bytes, content_type)
    tuples. Repeated names keep every value (lists)."""
    data: dict = {}
    files: list = []
    for name, value in form or []:
        if name in TITANOS_ONLY_FIELDS:
            continue
        if isinstance(value, tuple):
            files.append((name, value))
        else:
            if name in data:
                data[name] = (data[name] if isinstance(data[name], list) else [data[name]]) + [str(value)]
            else:
                data[name] = str(value)
    return data, files


def resolved(value) -> asyncio.Future:
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


def request_id(profile: Profile, headers) -> str | None:
    for name in profile.request_id_headers:
        value = headers.get(name) if headers is not None else None
        if value:
            return value
    return None


def raw_bytes(response: httpx.Response) -> AsyncIterator[bytes]:
    """response.aiter_raw() with httpx failures turned into typed upstream failures (mid-stream timeouts, resets). The
    response (and with it the egress concurrency slot) is closed when the stream ends, fails or is closed, also when it
    is closed before the first chunk."""
    async def chunks():
        try:
            if response.is_stream_consumed:  # a transport that hands back an already-read body (egress caps it too)
                if response.content:
                    yield response.content
                return
            async for chunk in response.aiter_raw():
                yield chunk
        except httpx.HTTPError as exc:
            raise UpstreamFailure(network_error(exc)) from None
        finally:
            await response.aclose()
    return ManagedStream(chunks(), response.aclose)


class AdapterBase(ProviderAdapter):
    """Common construction and plumbing. profile: the profile row; upstream: the broker.json Upstream record (or any object
    with the same attributes); default_max_tokens: settings.broker_default_max_output_tokens; voice_maps: broker.json
    voice_maps (openai_to_elevenlabs / elevenlabs_to_openai)."""
    adapter_name = ''
    native_protocol = ''

    def __init__(self, profile: Profile, upstream=None, *, default_max_tokens: int = DEFAULT_MAX_TOKENS, voice_maps=None):
        if profile.adapter != self.adapter_name:
            raise ValueError(f'{type(self).__name__} serves {self.adapter_name} profiles, not {profile.id}')
        self.profile = profile
        self.upstream = upstream
        self.default_max_tokens = int(default_max_tokens)
        self.voice_maps = voice_maps

    # surfaces ----------------------------------------------------------------------------------------------------------
    def native_surfaces(self) -> frozenset[str]:
        effective = getattr(self.upstream, 'effective_surfaces', None)
        return frozenset(effective()) if callable(effective) else frozenset(self.profile.surfaces)

    def has(self, name: str) -> bool:
        return name in self.native_surfaces() and name in self.profile.surfaces

    def model_for(self, grant: EgressGrant, fallback: str | None = None) -> str:
        target = getattr(grant, 'target', None)
        model = getattr(target, 'upstream_model', None) or fallback
        if not model:
            raise BrokerError('No upstream model was resolved for this call', 'model_required')
        return model

    # errors ------------------------------------------------------------------------------------------------------------
    def map_error(self, status: int, headers, body: bytes):
        return classify(status, headers or {}, body or b'', adapter=self.adapter_name)

    # sending -----------------------------------------------------------------------------------------------------------
    async def send(self, grant, egress, method, path, *, stream=False, **kw) -> httpx.Response:
        try:
            response = await egress.send(grant, self.upstream, self.profile, method, path, stream=stream, **kw)
        except httpx.HTTPError as exc:
            raise UpstreamFailure(network_error(exc)) from None
        if not 200 <= response.status_code < 300:
            try:
                body = await response.aread()
            except httpx.HTTPError:
                body = b''
            finally:
                await response.aclose()
            raise UpstreamFailure(self.map_error(response.status_code, response.headers, body))
        return response

    def send_sync(self, grant, egress, method, path, **kw) -> httpx.Response:
        try:
            response = egress.send_sync(grant, self.upstream, self.profile, method, path, stream=False, **kw)
        except httpx.HTTPError as exc:
            raise UpstreamFailure(network_error(exc)) from None
        if not 200 <= response.status_code < 300:
            raise UpstreamFailure(self.map_error(response.status_code, response.headers, response.content))
        return response

    @staticmethod
    def parse_json(response: httpx.Response):
        try:
            return json.loads(response.content.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise bad_upstream('The upstream answered with a body that is not JSON') from None

    # results -----------------------------------------------------------------------------------------------------------
    async def held(self, response: httpx.Response, protocol: str):
        """(first frame, replay iterator): an error-first stream raises UpstreamFailure before anything is committed."""
        frame, replay = await first_frame(raw_bytes(response), protocol)
        if is_error_frame(frame, protocol):
            await aclose(replay)
            await response.aclose()  # the slot is free before the failure propagates (retry or fallback may need it)
            payload = frame if isinstance(frame, dict) else frame.data
            if isinstance(frame, dict) and isinstance(frame.get('error'), str):
                payload = {'error': {'message': frame['error'], 'type': 'api_error'}}
            raise UpstreamFailure(classify_stream_error(payload, adapter=self.adapter_name))
        return frame, replay

    async def stream_result(self, response: httpx.Response, protocol: str, *, strip_usage: bool = False,
                            usage: Usage | None = None) -> UpstreamResult:
        """An UpstreamResult whose body replays the held first frame and then streams through the usage tee. result.usage
        resolves when the body ends, fails or is closed, also when aclose() runs before the body was iterated (a client
        that went away between commit and the first byte): then with the request-side estimate or None."""
        _, replay = await self.held(response, protocol)
        body, observed = tee_usage(replay, protocol, strip_injected_usage=strip_usage)
        future = observed if usage is None else _fallback(observed, usage)  # usage: a request-side estimate

        async def close():
            try:
                await aclose(body)
            finally:
                await response.aclose()
                if not future.done():  # the tee settled synchronously; a _fallback future follows it here, not later
                    seen = (observed.result() if observed.done() and not observed.cancelled()
                            and observed.exception() is None else None)
                    future.set_result(seen or usage)
        return UpstreamResult(status=response.status_code, headers=dict(response.headers),
                              content_type=response.headers.get('content-type', ''), body=body,
                              upstream_request_id=request_id(self.profile, response.headers), usage=future,
                              aclose=close)

    async def body_result(self, response: httpx.Response, surface: Surface, content: bytes | None = None,
                          usage: Usage | None = None) -> UpstreamResult:
        """An UpstreamResult for a complete (non-streamed) body; usage from usage_from_body unless given."""
        data = response.content if content is None else content
        if usage is None and 'json' in response.headers.get('content-type', ''):
            try:
                parsed = json.loads(data.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                usage = self.usage_from_body(surface, parsed, response.headers)

        async def body():
            yield data

        async def close():
            await response.aclose()
        return UpstreamResult(status=response.status_code, headers=dict(response.headers),
                              content_type=response.headers.get('content-type', ''), body=body(),
                              upstream_request_id=request_id(self.profile, response.headers), usage=resolved(usage),
                              aclose=close)

    async def sse_frames(self, response: httpx.Response, protocol: str = 'openai'):
        """Parsed SSE frames after the error-first check (translated paths)."""
        _, replay = await self.held(response, protocol)
        return parse_sse(replay)

    async def ndjson_lines(self, response: httpx.Response):
        _, replay = await self.held(response, 'ollama')
        return parse_ndjson(replay)

    # probes and catalogs -------------------------------------------------------------------------------------------------
    async def probe(self, grant: EgressGrant, egress) -> ProbeResult:
        path = self.profile.probe_path
        if not path:
            return ProbeResult(ok=True, status=None, latency_ms=None, code='skipped')
        start = time.perf_counter()
        try:
            response = await egress.send(grant, self.upstream, self.profile, 'GET', path)
        except httpx.HTTPError as exc:
            return ProbeResult(ok=False, status=None, latency_ms=None, code=network_error(exc).kind)
        except BrokerError as exc:
            return ProbeResult(ok=False, status=None, latency_ms=None, code=exc.code)
        latency = (time.perf_counter() - start) * 1000.0
        if 200 <= response.status_code < 300:
            return ProbeResult(ok=True, status=response.status_code, latency_ms=latency)
        err = self.map_error(response.status_code, response.headers, response.content)
        from ..translate.upstream_errors import broker_code
        return ProbeResult(ok=False, status=response.status_code, latency_ms=latency, code=broker_code(err)[0])

    def static_models(self) -> list[str]:
        return list(getattr(self.upstream, 'models', None) or [])

    # IR helpers -----------------------------------------------------------------------------------------------------------
    async def peek_events(self, events) -> AsyncIterator:
        """The IR event iterator, raising UpstreamFailure when the very first event is an error."""
        iterator = events.__aiter__()
        try:
            first = await iterator.__anext__()
        except StopAsyncIteration:
            return _empty()
        if isinstance(first, ErrorEvent):
            await aclose(iterator)
            raise UpstreamFailure(classify_stream_error({'error': {'type': first.code, 'message': first.message}},
                                                        adapter=self.adapter_name))
        return ManagedStream(_chain(first, iterator), lambda: aclose(iterator))


async def _empty():
    return
    yield  # pragma: no cover


async def _chain(first, iterator):
    try:
        yield first
        async for item in iterator:
            yield item
    finally:
        await aclose(iterator)


def _fallback(future: asyncio.Future, usage: Usage) -> asyncio.Future:
    out = asyncio.get_running_loop().create_future()

    def done(source):
        if out.done():
            return
        if source.cancelled():
            out.set_result(usage)
        elif source.exception() is not None:
            out.set_result(usage)
        else:
            out.set_result(source.result() or usage)
    future.add_done_callback(done)
    return out


def surface(name: str) -> Surface:
    return SURFACES[name]


def ir_text(ir: IRRequest) -> str:
    from ..translate.ir import Text
    return ''.join(p.text for m in ir.messages for p in m.parts if isinstance(p, Text))


def header_map(headers) -> Mapping:
    return headers if headers is not None else {}


__all__ = ['AdapterBase', 'CATALOG_MAX_PAGES', 'DEFAULT_MAX_TOKENS', 'header_map', 'ir_text', 'json_body', 'matches',
           'raw_bytes', 'request_id', 'resolved', 'split_form', 'strip_titanos', 'surface']
