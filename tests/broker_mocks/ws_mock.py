"""WebSocketMock: a loopback RFC 6455 server for the native WebSocket relay checks (OpenAI Realtime, ElevenLabs
stream-input, Gemini Live), written on asyncio streams so no WebSocket library is needed.

It runs on its own thread and event loop on 127.0.0.1 and answers any path:
- handshake: the request head is recorded (credential header values redacted to <sentinel:NAME> / <foreign:FP> exactly as
  BaseMock records them); the configured credential header must carry the expected value, else 401 with a JSON error
  body (the provider's envelope is irrelevant here: the broker maps 401 to 502 upstream_auth_failed); a header listed in
  refuse_headers answers 400; the first offered subprotocol in `subprotocols` is selected;
- after the 101: a ping, then a text frame {"type": "session.created", "path": ..., "query": ...}; every client frame is
  checked to be masked; a text frame is recorded and echoed as {"type": "echo", "data": <parsed JSON or text>}, a binary
  frame is echoed as binary; the text frame "close-me" makes the server close with 1000 "bye"; "big-message" answers a
  fragmented text message (three frames); a pong is recorded (hit.pongs).
hits: one WsHit per connection (path, query, headers, credentials, frames received, pongs, close code).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
from dataclasses import dataclass, field
from urllib.parse import parse_qsl

from .faults import wait_until
from .fixtures import SENTINELS
from .server import CREDENTIAL_HEADERS, credential_fingerprints, redact_value

GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'


@dataclass
class WsHit:
    path: str
    query: list
    headers: dict
    credentials: list = field(default_factory=list)
    fingerprints: set = field(default_factory=set)
    status: int | None = None
    subprotocol: str | None = None
    frames: list = field(default_factory=list)  # [('text', str) | ('bytes', bytes)]
    pongs: int = 0
    unmasked: int = 0
    close_code: int | None = None

    def presented(self, value: str) -> bool:
        from .server import fingerprint
        return fingerprint(value) in self.fingerprints or any(value in v for v in self.headers.values())


class WebSocketMock:
    def __init__(self, *, credential_header: str = 'authorization', credential: str | None = None,
                 subprotocols=('realtime',), refuse_headers=()):
        self.credential_header = credential_header.lower()
        self.credential = credential if credential is not None else 'Bearer ' + SENTINELS['openai']
        self.subprotocols = tuple(subprotocols)
        self.refuse_headers = tuple(h.lower() for h in refuse_headers)
        self.hits: list[WsHit] = []
        self.port = 0
        self._loop = None
        self._server = None
        self._thread = None
        self._ready = threading.Event()

    @property
    def base_url(self) -> str:
        return f'http://127.0.0.1:{self.port}'

    # lifecycle --------------------------------------------------------------------------------------------------------
    def start(self) -> str:
        def run():
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)

            async def main():
                self._server = await asyncio.start_server(self._handle, '127.0.0.1', 0)
                self.port = self._server.sockets[0].getsockname()[1]
                self._ready.set()
                async with self._server:
                    await self._server.serve_forever()
            try:
                loop.run_until_complete(main())
            except (asyncio.CancelledError, RuntimeError):
                pass
            finally:
                loop.close()

        self._thread = threading.Thread(target=run, name='ws-mock', daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError('WebSocketMock did not start')
        return self.base_url

    def stop(self) -> None:
        if self._loop is not None and self._server is not None:
            async def close():
                self._server.close()
            try:
                asyncio.run_coroutine_threadsafe(close(), self._loop).result(5)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(10)
        self._server = self._loop = self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def wait_hits(self, count: int, timeout: float = 5.0) -> bool:
        return wait_until(lambda: len(self.hits) >= count, timeout)

    # protocol ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _frame(opcode: int, payload: bytes, fin: bool = True) -> bytes:
        head = bytearray([(0x80 if fin else 0) | opcode])
        if len(payload) < 126:
            head.append(len(payload))
        elif len(payload) < 65536:
            head.append(126)
            head += len(payload).to_bytes(2, 'big')
        else:
            head.append(127)
            head += len(payload).to_bytes(8, 'big')
        return bytes(head) + payload

    async def _read_frame(self, reader, hit: WsHit):
        first, second = await reader.readexactly(2)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        if not masked:
            hit.unmasked += 1
        length = second & 0x7F
        if length == 126:
            length = int.from_bytes(await reader.readexactly(2), 'big')
        elif length == 127:
            length = int.from_bytes(await reader.readexactly(8), 'big')
        mask = await reader.readexactly(4) if masked else b'\x00\x00\x00\x00'
        data = await reader.readexactly(length) if length else b''
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        return bool(first & 0x80), opcode, data

    async def _handle(self, reader, writer):
        try:
            head = await reader.readuntil(b'\r\n\r\n')
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.close()
            return
        lines = head.decode('latin-1').split('\r\n')
        target = lines[0].split(' ')[1] if len(lines[0].split(' ')) > 1 else '/'
        path, _, raw_query = target.partition('?')
        headers, credentials, fps = {}, [], set()
        for line in lines[1:]:
            name, sep, value = line.partition(':')
            if not sep:
                continue
            name, value = name.strip().lower(), value.strip()
            if name in CREDENTIAL_HEADERS:
                spelled, _ = redact_value(value)
                credentials.append(('header:' + name, spelled))
                fps |= credential_fingerprints(value)
                headers[name] = spelled
            else:
                headers[name] = value
        hit = WsHit(path=path, query=parse_qsl(raw_query, keep_blank_values=True), headers=headers,
                    credentials=credentials, fingerprints=fps)
        self.hits.append(hit)
        raw_credential = next((line.partition(':')[2].strip() for line in lines[1:]
                               if line.partition(':')[0].strip().lower() == self.credential_header), None)

        def refuse(status: int, message: str):
            body = json.dumps({'error': {'message': message, 'type': 'invalid_request_error'}}).encode('utf-8')
            hit.status = status
            writer.write((f'HTTP/1.1 {status} Refused\r\nContent-Type: application/json\r\nContent-Length: {len(body)}'
                          '\r\nConnection: close\r\n\r\n').encode('latin-1') + body)
        if raw_credential != self.credential:
            refuse(401, 'Incorrect API key provided')
            await writer.drain()
            writer.close()
            return
        if any(h in headers for h in self.refuse_headers):
            refuse(400, 'refused header')
            await writer.drain()
            writer.close()
            return
        key = headers.get('sec-websocket-key', '')
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode('ascii')).digest()).decode('ascii')
        offered = [p.strip() for p in headers.get('sec-websocket-protocol', '').split(',') if p.strip()]
        chosen = next((p for p in offered if p in self.subprotocols), None)
        hit.subprotocol = chosen
        hit.status = 101
        response = ('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n'
                    f'Sec-WebSocket-Accept: {accept}\r\n' + (f'Sec-WebSocket-Protocol: {chosen}\r\n' if chosen else '')
                    + '\r\n')
        writer.write(response.encode('latin-1'))
        writer.write(self._frame(0x9, b'mock-ping'))
        writer.write(self._frame(0x1, json.dumps({'type': 'session.created', 'path': path,
                                                  'query': hit.query}).encode('utf-8')))
        await writer.drain()
        parts, kind = [], None
        try:
            while True:
                fin, opcode, data = await self._read_frame(reader, hit)
                if opcode == 0xA:
                    hit.pongs += 1
                    continue
                if opcode == 0x9:
                    writer.write(self._frame(0xA, data))
                    await writer.drain()
                    continue
                if opcode == 0x8:
                    hit.close_code = int.from_bytes(data[:2], 'big') if len(data) >= 2 else 1005
                    writer.write(self._frame(0x8, data[:2]))
                    await writer.drain()
                    break
                if opcode in (0x1, 0x2):
                    kind, parts = opcode, [data]
                elif opcode == 0x0:
                    parts.append(data)
                if not fin:
                    continue
                message = b''.join(parts)
                if kind == 0x1:
                    text = message.decode('utf-8')
                    hit.frames.append(('text', text))
                    if text == 'close-me':
                        writer.write(self._frame(0x8, (1000).to_bytes(2, 'big') + b'bye'))
                        await writer.drain()
                        try:
                            await asyncio.wait_for(self._read_frame(reader, hit), 2)
                        except Exception:
                            pass
                        hit.close_code = 1000
                        break
                    if text == 'big-message':
                        whole = json.dumps({'type': 'big', 'data': 'x' * 300}).encode('utf-8')
                        writer.write(self._frame(0x1, whole[:100], fin=False))
                        writer.write(self._frame(0x0, whole[100:200], fin=False))
                        writer.write(self._frame(0x0, whole[200:], fin=True))
                        await writer.drain()
                        continue
                    try:
                        echoed = json.loads(text)
                    except ValueError:
                        echoed = text
                    writer.write(self._frame(0x1, json.dumps({'type': 'echo', 'data': echoed}).encode('utf-8')))
                else:
                    hit.frames.append(('bytes', message))
                    writer.write(self._frame(0x2, message))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass


__all__ = ['WebSocketMock', 'WsHit']
