"""Shared harness for the WP6 broker checks (routing, breakers, advisor, dispatch, stages).

- installs the loopback netguard (tests/broker_mocks) on import, before anything can connect;
- Lab: a temporary TITANOS_HOME with a broker.json, an in-memory vault holding validated synthetic sentinel credentials
  bound to the mocks' exact origins, and a lazily built titanos.broker.facade.Broker (loopback-only egress);
- FakeLocal: an inner ASGI app standing in for the unchanged local routes (it records every call and answers chat,
  completions, embeddings, transcription and speech; per-model typed errors in the server.error envelope);
- call(): a minimal ASGI driver that records every sent message with its time (streams, keepalives and disconnects can
  be asserted; httpx's ASGITransport buffers whole responses);
- key(): synthetic key principals.
No real secret is used: every credential is a WP4 sentinel.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / 'src', ROOT / 'tests'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from broker_mocks import SENTINELS, netguard  # noqa: E402

netguard.install()

from titanos.broker import inbound  # noqa: E402
from titanos.broker.config import BrokerConfig, write_broker_config  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.facade import Broker  # noqa: E402
from titanos.broker.types import KeyBudget, KeyConstraints, KeyLimits, Principal  # noqa: E402
from titanos.broker.vault.store import Vault, platform_default  # noqa: E402
from titanos.settings import Settings  # noqa: E402

SENTINEL_FOR_PROFILE = {'openai': 'openai', 'anthropic': 'anthropic', 'gemini': 'gemini', 'elevenlabs': 'elevenlabs',
                        'ollama_cloud': 'ollama_cloud', 'xai': 'xai', 'hivemind_gateway': 'hivemind_gateway',
                        'custom_openai': 'openai', 'openrouter': 'openrouter', 'mistral': 'mistral'}


class MemoryBackend:
    """An in-memory VaultBackend (synthetic sentinels only)."""

    def __init__(self, name):
        self.name, self.values = name, {}

    def available(self):
        return True

    def store(self, id, value):
        self.values[id] = bytes(value)

    def load(self, id):
        if id not in self.values:
            raise BrokerError('missing', 'credential_unavailable')
        return self.values[id]

    def delete(self, id):
        self.values.pop(id, None)


def local_config(models: dict) -> SimpleNamespace:
    """A registry stand-in: {name: task} or {name: (task, backend)}."""
    specs = {}
    for name, value in models.items():
        task, backend = (value, 'transformers') if isinstance(value, str) else value
        specs[name] = SimpleNamespace(task=task, backend=backend, defaults={}, voices={}, device='cpu')
    return SimpleNamespace(models=specs)


def settings(**kw) -> Settings:
    values = {'broker_egress': 'loopback-only'}
    values.update(kw)
    return Settings(**values)


_KEYS = [0]


def key(*scopes, models=('*',), upstreams=('*',), classes=('local', 'cloud', 'router'),
        locations=('local_only', 'prefer_local', 'auto', 'cloud_only'), limits=None, budget=None,
        default_location=None, kind='key', id=None) -> Principal:
    _KEYS[0] += 1
    return Principal(id=id or f'k{_KEYS[0]:09d}', kind=kind, scopes=frozenset(scopes),
                     constraints=KeyConstraints(models=tuple(models), upstreams=tuple(upstreams),
                                                target_classes=frozenset(classes), locations=frozenset(locations)),
                     limits=KeyLimits(**limits) if limits else None,
                     budget=KeyBudget(**{k: (Decimal(v) if k.startswith('usd') or k == 'max_request_usd' else v)
                                         for k, v in budget.items()}) if budget else None,
                     default_location=default_location)


BOOTSTRAP = Principal(id='bootstrap', kind='bootstrap', scopes=frozenset({'admin'}))


class Lab:
    """broker.json + vault + Broker over a temporary TITANOS_HOME. upstreams: {id: Upstream dict}; a credential named in an
    upstream is stored as the sentinel of its profile (or creds[id]) and validated for the upstream's origin unless
    listed in unvalidated."""

    def __init__(self, document: dict, *, local: dict | None = None, settings_kw: dict | None = None,
                 creds: dict | None = None, unvalidated=(), engine_slot=None, native_getter=None, write=True):
        self.home = Path(tempfile.mkdtemp(prefix='titanos-wp6-'))
        self.settings = settings(**(settings_kw or {}))
        self.document = document
        self.local = local_config(local or {})
        self.config = BrokerConfig.model_validate(document)
        if write:
            write_broker_config(self.home / 'broker.json', self.config)
        self.backend = MemoryBackend(platform_default())
        self.vault = Vault(self.settings, self.home, backends={platform_default(): self.backend})
        creds = dict(creds or {})
        for uid, upstream in self.config.upstreams.items():
            if not upstream.credential:
                continue
            sentinel = creds.get(uid) or SENTINEL_FOR_PROFILE.get(upstream.profile, 'openai')
            origin = upstream.origin(allow_http_lan=self.config.policy.allow_http_lan)
            self.vault.put_secret(upstream.credential, SENTINELS[sentinel], upstream_id=uid, origin=origin)
            if uid not in unvalidated:
                self.vault.mark_validated(upstream.credential, True, None, None)
        self.broker = Broker(settings=self.settings, config=self.local, home=self.home, vault=self.vault,
                             engine_slot=engine_slot, native_getter=native_getter)

    def advisor_credential(self, cred_id: str, url_origin: str, sentinel: str = 'advisor') -> None:
        self.vault.put_secret(cred_id, SENTINELS[sentinel], upstream_id=cred_id, origin=url_origin, purpose='advisor')
        self.vault.mark_validated(cred_id, True, None, None)

    def rewrite(self, document: dict) -> None:
        self.document = document
        self.config = BrokerConfig.model_validate(document)
        write_broker_config(self.home / 'broker.json', self.config)

    def close(self) -> None:
        try:
            asyncio.run(self.broker.close())
        except RuntimeError:
            pass
        shutil.rmtree(self.home, ignore_errors=True)


def upstream(mock, profile: str, *, credential: str | None = 'cred', enabled: bool = True, **kw) -> dict:
    out = {'profile': profile, 'base_url': mock.api_base, 'enabled': enabled}
    if credential:
        out['credential'] = credential
    out.update(kw)
    return out


def total_hits(*mocks) -> int:
    return sum(len(m.hits) for m in mocks)


# ---------------------------------------------------------------------------------------------------------------------
# the fake inner app

class FakeLocal:
    """Stands in for the local FastAPI routes behind BrokerDispatch. errors: {model: (status, code)}. Every call is
    recorded as a dict {path, method, model, body, headers, target, scope, receive}."""

    def __init__(self):
        self.calls: list[dict] = []
        self.errors: dict[str, tuple[int, str]] = {}

    async def __call__(self, scope, receive, send):
        chunks = bytearray()
        while True:
            message = await receive()
            if message.get('type') == 'http.disconnect':
                return
            chunks.extend(message.get('body', b''))
            if not message.get('more_body', False):
                break
        body = bytes(chunks)
        headers = {k.decode('latin-1').lower(): v.decode('latin-1') for k, v in scope.get('headers', [])}
        document, model = None, None
        content_type = headers.get('content-type', '')
        if 'multipart/form-data' in content_type:
            form = await inbound.parse_multipart(body, content_type)
            model = next((v for k, v in form if k in ('model', 'model_id') and isinstance(v, str)), None)
        else:
            try:
                document = json.loads(body) if body else {}  # raw bytes, as Starlette's request.json()
            except ValueError:
                document = None
            if isinstance(document, dict):
                model = document.get('model') or document.get('model_id')
        self.calls.append({'path': scope['path'], 'method': scope['method'], 'model': model, 'body': body,
                           'headers': headers, 'target': scope.get('titanos.broker.target'), 'scope': scope,
                           'receive': receive})
        if model in self.errors:
            status, code = self.errors[model]
            payload = json.dumps({'error': {'message': f'local {code}', 'type': code, 'param': None,
                                            'code': code}}).encode()
            await send({'type': 'http.response.start', 'status': status,
                        'headers': [(b'content-type', b'application/json')]})
            await send({'type': 'http.response.body', 'body': payload, 'more_body': False})
            return
        path = scope['path']
        if path.endswith('/chat/completions') and isinstance(document, dict) and document.get('stream'):
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-type', b'text/event-stream')]})
            for text in ('local ', 'hello'):
                chunk = {'id': 'chatcmpl-local', 'object': 'chat.completion.chunk', 'created': 1, 'model': model,
                         'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}
                await send({'type': 'http.response.body', 'body': b'data: ' + json.dumps(chunk).encode() + b'\n\n',
                            'more_body': True})
            final = {'id': 'chatcmpl-local', 'object': 'chat.completion.chunk', 'created': 1, 'model': model,
                     'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
            await send({'type': 'http.response.body', 'body': b'data: ' + json.dumps(final).encode() + b'\n\n'
                        + b'data: [DONE]\n\n', 'more_body': False})
            return
        if path.endswith('/embeddings'):
            answer = {'object': 'list', 'model': model, 'data': [{'object': 'embedding', 'index': 0,
                                                                  'embedding': [0.25, 0.5]}],
                      'usage': {'prompt_tokens': 2, 'total_tokens': 2}}
        elif path.endswith('/audio/transcriptions'):
            answer = {'text': 'local transcript'}
        elif path.startswith('/v1/text-to-speech') or path.endswith('/audio/speech'):
            await send({'type': 'http.response.start', 'status': 200, 'headers': [(b'content-type', b'audio/wav')]})
            await send({'type': 'http.response.body', 'body': b'RIFF-local-audio', 'more_body': False})
            return
        else:
            answer = {'id': 'chatcmpl-local', 'object': 'chat.completion', 'created': 1, 'model': model,
                      'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'local hello'},
                                   'finish_reason': 'stop'}],
                      'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
        data = json.dumps(answer).encode()
        await send({'type': 'http.response.start', 'status': 200,
                    'headers': [(b'content-type', b'application/json'),
                                (b'content-length', str(len(data)).encode())]})
        await send({'type': 'http.response.body', 'body': data, 'more_body': False})


# ---------------------------------------------------------------------------------------------------------------------
# the ASGI driver

@dataclass
class Reply:
    status: int | None
    headers: dict
    body: bytes
    messages: list = field(default_factory=list)  # (seconds since the call started, message)
    started: float | None = None  # seconds to http.response.start

    def json(self):
        return json.loads(self.body.decode('utf-8'))

    def header(self, name: str):
        return self.headers.get(name.lower())

    def chunks(self) -> list[bytes]:
        return [m['body'] for _, m in self.messages if m['type'] == 'http.response.body' and m.get('body')]


async def acall(app, method: str, path: str, *, json_body=None, body: bytes | None = None, headers=None,
                query: str = '', principal=None, content_type: str | None = None, disconnect_after: int | None = None,
                on_body=None) -> Reply:
    if json_body is not None:
        body = json.dumps(json_body).encode('utf-8')
        content_type = content_type or 'application/json'
    body = body or b''
    raw_headers = [(b'host', b'127.0.0.1:11435')]
    if content_type:
        raw_headers.append((b'content-type', content_type.encode('latin-1')))
    if body:
        raw_headers.append((b'content-length', str(len(body)).encode('latin-1')))
    for name, value in (headers or {}).items():
        raw_headers.append((name.lower().encode('latin-1'), str(value).encode('latin-1')))
    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': method, 'path': path,
             'raw_path': path.encode('utf-8'), 'query_string': query.encode('latin-1'), 'headers': raw_headers,
             'client': ('127.0.0.1', 50000), 'server': ('127.0.0.1', 11435), 'scheme': 'http', 'root_path': ''}
    if principal is not None:
        scope['titanos.principal'] = principal
    gone = asyncio.Event()
    delivered = False
    t0 = time.monotonic()
    reply = Reply(status=None, headers={}, body=b'')
    bodies = [0]

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': body, 'more_body': False}
        await gone.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        reply.messages.append((time.monotonic() - t0, message))
        if message['type'] == 'http.response.start':
            reply.status = message['status']
            reply.started = time.monotonic() - t0
            reply.headers = {k.decode('latin-1').lower(): v.decode('latin-1') for k, v in message.get('headers', [])}
        elif message['type'] == 'http.response.body':
            reply.body += message.get('body', b'')
            if message.get('body'):
                bodies[0] += 1
                if on_body is not None:
                    on_body(bodies[0], message)
                if disconnect_after is not None and bodies[0] >= disconnect_after:
                    gone.set()
    try:
        await app(scope, receive, send)
    finally:
        gone.set()
    return reply


async def _call_and_close(app, method, path, kw):
    try:
        return await acall(app, method, path, **kw)
    finally:
        broker = getattr(app, 'broker', None)
        if broker is not None:  # pooled connections never outlive their event loop
            await broker.close()


def call(app, method: str, path: str, **kw) -> Reply:
    return asyncio.run(_call_and_close(app, method, path, kw))


def run(coro_factory, broker=None):
    """asyncio.run(coro_factory()) and close the broker's pooled connections inside the same loop."""
    async def main():
        try:
            return await coro_factory()
        finally:
            if broker is not None:
                await broker.close()
    return asyncio.run(main())


def multipart(fields: list, boundary: str = 'wp6boundary') -> tuple[bytes, str]:
    chunks = []
    for name, value in fields:
        if isinstance(value, tuple):
            filename, data, ctype = value
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                          f'Content-Type: {ctype}\r\n\r\n'.encode() + data + b'\r\n')
        else:
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    chunks.append(f'--{boundary}--\r\n'.encode())
    return b''.join(chunks), f'multipart/form-data; boundary={boundary}'


__all__ = ['BOOTSTRAP', 'FakeLocal', 'Lab', 'MemoryBackend', 'Reply', 'SENTINELS', 'acall', 'call', 'key', 'run',
           'local_config', 'multipart', 'netguard', 'settings', 'total_hits', 'upstream']
