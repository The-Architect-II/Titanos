"""Fault injection for the loopback mocks: mock.faults.push(kind, **opts).

A pushed fault applies to the next matching request (FIFO; `times` requests, default 1). `path` (a regular expression
searched in the request path) and `surface` (the mock's surface name, e.g. 'openai.chat') narrow the match. Faults are
taken after the route and auth checks, so a wrong credential still gets the provider's native 401.

Request-phase kinds (taken before the handler runs):
  status            {status: 429|500|502|503|529|401|403|400|..., retry_after: seconds (429 default 7), message,
                     leak: bool (400 default True: the error message carries SENTINELS['leak'] and the presented
                     credential value, the way real providers echo a key fragment)} -> the provider's native error envelope
  slow_first_byte   {delay: seconds (default 2.0)} -> nothing is sent before the delay, then the normal answer
  malformed_json    {} -> 200 application/json whose body is truncated JSON
  oversized_body    {size: bytes (default 2 MiB), declared_length: bool (default False: chunked, no content-length)}
                     -> 200 application/json of `size` bytes, streamed in 64 KiB chunks
  redirect_foreign  {target: another mock (default: this mock's own lazily started sink), status: 307}
                     -> a redirect to the same path and query on the other loopback mock, which records any auth it sees
  set_cookie        {cookie: text} -> the normal answer plus two Set-Cookie headers and a Location header
  connect_refused   {seconds: float|None} -> applied at push time: the listener is closed, so the next connects are
                     refused at TCP level (httpx.ConnectError), until faults.resume()/clear() or `seconds` elapse.
                     It is never queued: a refused connection never reaches the mock, so it cannot be counted.
Stream-phase kinds (taken only by streaming answers: SSE, NDJSON or chunked binary; non-streaming answers leave them queued):
  error_first_event        {} -> HTTP 200 whose first frame is the protocol's error event, then the stream ends
  disconnect_after_frames  {frames: n (default 1)} -> n frames, then the connection is aborted without a terminating chunk
  malformed_sse_line       {at: 1-based frame (default 1)} -> that frame is replaced by an unparsable data line
  hold_frame               {frame: 1-based N (default 2), timeout: seconds (default 30)} -> frames before N are sent, then
                           the stream waits on mock.gate (a threading.Event, cleared at push) before sending frame N
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

REQUEST_KINDS = frozenset({'status', 'slow_first_byte', 'malformed_json', 'oversized_body', 'redirect_foreign', 'set_cookie'})
STREAM_KINDS = frozenset({'error_first_event', 'disconnect_after_frames', 'malformed_sse_line', 'hold_frame'})
IMMEDIATE_KINDS = frozenset({'connect_refused'})
KINDS = REQUEST_KINDS | STREAM_KINDS | IMMEDIATE_KINDS
STATUS_DEFAULT_RETRY_AFTER = {429: 7, 503: 3, 529: 3}


@dataclass
class Fault:
    kind: str
    opts: dict = field(default_factory=dict)
    path: str | None = None
    surface: str | None = None
    times: int = 1
    used: int = 0

    def matches(self, path: str, surface: str | None) -> bool:
        if self.path is not None and not re.search(self.path, path):
            return False
        if self.surface is not None and self.surface != surface:
            return False
        return True

    def get(self, name: str, default: Any = None) -> Any:
        return self.opts.get(name, default)


class FaultQueue:
    """Thread-safe FIFO of pushed faults. The owning mock installs `on_push` (prepares sinks, pauses listeners, clears
    the gate) and `on_resume` (reopens a paused listener)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._faults: list[Fault] = []
        self.applied: list[Fault] = []
        self.on_push: Callable[[Fault], None] | None = None
        self.on_resume: Callable[[], None] | None = None
        self._resume_timer: threading.Timer | None = None

    def push(self, kind: str, *, path: str | None = None, surface: str | None = None, times: int = 1, **opts) -> Fault:
        if kind not in KINDS:
            raise ValueError(f'Unknown fault kind {kind!r}; one of {sorted(KINDS)}')
        if times < 1:
            raise ValueError('times must be at least 1')
        if kind == 'status' and 'status' not in opts:
            raise ValueError("status faults need status=<code>")
        fault = Fault(kind, dict(opts), path, surface, times)
        if self.on_push is not None:
            self.on_push(fault)
        if kind in IMMEDIATE_KINDS:
            seconds = opts.get('seconds')
            if seconds is not None:
                self._resume_timer = threading.Timer(float(seconds), self.resume)
                self._resume_timer.daemon = True
                self._resume_timer.start()
            with self._lock:
                self.applied.append(fault)
            return fault
        with self._lock:
            self._faults.append(fault)
        return fault

    def take(self, phase: str, path: str, surface: str | None) -> Fault | None:
        """The first queued fault of `phase` ('request' or 'stream') matching the request; its use is counted."""
        kinds = REQUEST_KINDS if phase == 'request' else STREAM_KINDS
        with self._lock:
            for fault in self._faults:
                if fault.kind in kinds and fault.matches(path, surface):
                    fault.used += 1
                    if fault.used >= fault.times:
                        self._faults.remove(fault)
                    self.applied.append(fault)
                    return fault
        return None

    def pending(self) -> list[Fault]:
        with self._lock:
            return list(self._faults)

    def resume(self) -> None:
        """Reopen a listener closed by connect_refused (no-op otherwise)."""
        timer, self._resume_timer = self._resume_timer, None
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        if self.on_resume is not None:
            self.on_resume()

    def clear(self) -> None:
        with self._lock:
            self._faults.clear()
        self.resume()


def wait_until(predicate: Callable[[], bool], timeout: float, interval: float = 0.01) -> bool:
    """Bounded poll used by tests and the server start/stop paths."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


__all__ = ['Fault', 'FaultQueue', 'IMMEDIATE_KINDS', 'KINDS', 'REQUEST_KINDS', 'STATUS_DEFAULT_RETRY_AFTER', 'STREAM_KINDS',
           'wait_until']
