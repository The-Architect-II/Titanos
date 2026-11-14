"""Shared harness for the WP5 broker checks (adapters, translation golden, streaming, upstream errors).

- installs the loopback netguard (tests/broker_mocks) on import, before anything can connect;
- builds EgressGrants, credential handles (synthetic sentinels only) and a fresh EgressClient per call (loopback-only
  egress, no vault unless given);
- AzureHostTransport: a TEST-ONLY httpx transport that maps the exact Azure resource origin to the loopback AzureMock (no
  production base-URL override exists; ratified R12). Every other origin is refused inside the transport;
- MockTargets: the WP4 mocks started once per test class, one adapter per target column (OAI, AZ, ANT, GEM, OLC, XI);
- INBOUND: one decode/encode pair per inbound protocol, used to drive translated (T) cells end to end;
- main(cells): a unittest runner that records which tests passed and checks that every cited matrix cell names a test
  that exists and passed. When TITANOS_BROKER_RESULTS names a file, one JSON line {check, passed, failed, cells} is
  appended for the broker matrix builder (WP9).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / 'src', ROOT / 'tests'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from broker_mocks import (AnthropicMock, AzureMock, ElevenLabsMock, GeminiMock, OllamaCloudMock, OpenAIMock,  # noqa: E402
                          SENTINELS, netguard)

netguard.install()

import httpx  # noqa: E402

from titanos.broker.adapters.anthropic import AnthropicAdapter  # noqa: E402
from titanos.broker.adapters.elevenlabs import ElevenLabsAdapter  # noqa: E402
from titanos.broker.adapters.gemini import GeminiAdapter  # noqa: E402
from titanos.broker.adapters.ollama_native import OllamaNativeAdapter  # noqa: E402
from titanos.broker.adapters.openai_compat import OpenAICompatAdapter  # noqa: E402
from titanos.broker.config import Upstream  # noqa: E402
from titanos.broker.egress import EgressClient  # noqa: E402
from titanos.broker.profiles import PROFILES  # noqa: E402
from titanos.broker.role import switches_for  # noqa: E402
from titanos.broker.sse import NDJSONSplitter, SSESplitter  # noqa: E402
from titanos.broker.translate import anthropic_messages, gemini, ollama, openai_chat, responses  # noqa: E402
from titanos.broker.types import EgressGrant, Target  # noqa: E402
from titanos.broker.vault import CredentialHandle, Secret  # noqa: E402
from titanos.settings import Settings  # noqa: E402

LOOPBACK = Settings(broker_egress='loopback-only')
AZURE_RESOURCE = 'contoso'
AZURE_API_VERSION = '2024-10-21'
ADAPTERS = {'openai_compat': OpenAICompatAdapter, 'anthropic': AnthropicAdapter, 'gemini': GeminiAdapter,
            'elevenlabs': ElevenLabsAdapter, 'ollama_native': OllamaNativeAdapter}


def run(coro):
    return asyncio.run(coro)


def handle(origin: str, profile_id: str, sentinel: str, *, credential_id: str | None = None,
           template: str | None = None) -> CredentialHandle:
    template = template or PROFILES[profile_id].auth
    return CredentialHandle(origin, template, Secret(SENTINELS[sentinel]), credential_id=credential_id)


def grant(upstream, profile_id: str, model: str | None, credential: CredentialHandle | None, *,
          purpose: str = 'inference', upstream_id: str = 'up', deadline: float = 30.0) -> EgressGrant:
    origin = upstream.origin(allow_http_lan=False)
    target = Target(ref=f'{model}@{upstream_id}', klass='cloud', upstream_id=upstream_id, upstream_model=model,
                    adapter=PROFILES[profile_id].adapter, profile_id=profile_id) if model else None
    return EgressGrant(decision_id='dec-wp5', request_id='req-wp5', purpose=purpose, delegated=False, target=target,
                       upstream_id=upstream_id, origin=origin, credential=credential,
                       deadline=time.monotonic() + deadline, hop_header=None)


class AzureHostTransport(httpx.AsyncBaseTransport, httpx.BaseTransport):
    """Test-only: requests for exactly https://<resource>.<endpoint_host>:443 go to the loopback AzureMock (same path and
    query, same headers); anything else is refused here, so a wrong origin can never reach the mock."""

    def __init__(self, mock_origin: str, host: str):
        self.target = httpx.URL(mock_origin)
        self.host = host
        self.seen: list[httpx.URL] = []
        self._async = httpx.AsyncHTTPTransport()
        self._sync = httpx.HTTPTransport()

    def _rewrite(self, request: httpx.Request) -> httpx.Request:
        url = request.url
        if url.scheme != 'https' or url.host != self.host or (url.port or 443) != 443:
            raise httpx.ConnectError(f'test transport refuses {url.scheme}://{url.host}', request=request)
        self.seen.append(url)
        moved = url.copy_with(scheme=self.target.scheme, host=self.target.host, port=self.target.port)
        headers = [(k, v) for k, v in request.headers.raw if k.lower() != b'host']
        headers.insert(0, (b'Host', moved.netloc))
        return httpx.Request(request.method, moved, headers=headers, stream=request.stream,
                             extensions=request.extensions)

    async def handle_async_request(self, request):
        return await self._async.handle_async_request(self._rewrite(request))

    def handle_request(self, request):
        return self._sync.handle_request(self._rewrite(request))

    async def aclose(self):
        await self._async.aclose()

    def close(self):
        self._sync.close()


class Egress:
    """A fresh EgressClient for one asyncio.run (pooled connections never outlive their event loop)."""

    def __init__(self, *, transport=None, vault=None, settings=LOOPBACK):
        self.client = EgressClient(settings, vault, switches_for(settings), transport=transport,
                                   sync_transport=transport)

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *exc):
        await self.client.aclose()


class MockTarget:
    """One target column: a started mock, its Upstream record, the adapter and a credential handle."""

    def __init__(self, column, mock, profile_id, model, sentinel, *, upstream=None, transport=None, **adapter_kw):
        self.column, self.mock, self.profile_id, self.model, self.sentinel = column, mock, profile_id, model, sentinel
        self.profile = PROFILES[profile_id]
        self.upstream = upstream or Upstream(profile=profile_id, base_url=mock.api_base, enabled=True,
                                             credential='cred')
        self.transport = transport
        self.adapter = ADAPTERS[self.profile.adapter](self.profile, self.upstream, **adapter_kw)

    @property
    def origin(self) -> str:
        return self.upstream.origin(allow_http_lan=False)

    def grant(self, model: str | None = None, *, purpose: str = 'inference', credential_id: str | None = None):
        return grant(self.upstream, self.profile_id, model or self.model,
                     handle(self.origin, self.profile_id, self.sentinel, credential_id=credential_id), purpose=purpose)

    def egress(self, **kw) -> Egress:
        return Egress(transport=self.transport, **kw)


def start_targets(columns=('OAI', 'AZ', 'ANT', 'GEM', 'OLC', 'XI'), **kw) -> dict[str, MockTarget]:
    targets = {}
    if 'OAI' in columns:
        mock = OpenAIMock()
        mock.start()
        targets['OAI'] = MockTarget('OAI', mock, 'openai', 'gpt-4o-mini', 'openai')
    if 'AZ' in columns:
        mock = AzureMock()
        mock.start()
        upstream = Upstream(profile='azure_openai', enabled=True, credential='cred',
                            azure={'resource': AZURE_RESOURCE, 'api_version': AZURE_API_VERSION,
                                   'deployments': {'gpt-4o-mini': 'chat-dep', 'gpt-35-turbo-instruct': 'instruct-dep',
                                                   'text-embedding-3-small': 'embed-dep', 'tts-1': 'tts-dep',
                                                   'whisper-1': 'whisper-dep', 'dall-e-3': 'image-dep'}})
        host = httpx.URL(upstream.effective_base_url()).host
        targets['AZ'] = MockTarget('AZ', mock, 'azure_openai', 'gpt-4o-mini', 'azure_openai', upstream=upstream,
                                   transport=AzureHostTransport(mock.origin, host))
    if 'ANT' in columns:
        mock = AnthropicMock()
        mock.start()
        upstream = Upstream(profile='anthropic', base_url=mock.api_base, enabled=True, credential='cred',
                            anthropic_betas=kw.get('anthropic_betas', []))
        targets['ANT'] = MockTarget('ANT', mock, 'anthropic', 'claude-sonnet-4-5-20250929', 'anthropic',
                                    upstream=upstream, default_max_tokens=1024)
    if 'GEM' in columns:
        mock = GeminiMock(sse_crlf=kw.get('gemini_crlf', False))
        mock.start()
        targets['GEM'] = MockTarget('GEM', mock, 'gemini', 'gemini-2.5-flash', 'gemini')
    if 'OLC' in columns:
        mock = OllamaCloudMock()
        mock.start()
        targets['OLC'] = MockTarget('OLC', mock, 'ollama_cloud', 'gpt-oss:120b', 'ollama_cloud')
    if 'XI' in columns:
        mock = ElevenLabsMock()
        mock.start()
        targets['XI'] = MockTarget('XI', mock, 'elevenlabs', 'eleven_multilingual_v2', 'elevenlabs',
                                   voice_maps={'openai_to_elevenlabs': {'alloy': '21m00Tcm4TlvDq8ikWAM'},
                                               'elevenlabs_to_openai': {'21m00Tcm4TlvDq8ikWAM': 'alloy'}})
    return targets


def stop_targets(targets: dict) -> None:
    for target in targets.values():
        target.mock.stop()


# ---------------------------------------------------------------------------------------------------------------------
# inbound protocol drivers (decode the inbound request, encode the IR answer back)

def _response_obj(model):
    return responses.response_object(model=model)


INBOUND = {
    'openai.chat': {
        'decode': lambda body: openai_chat.decode_request(body),
        'response': lambda resp, model: openai_chat.encode_response(resp, model_name=model),
        'stream': lambda events, model: openai_chat.encode_stream(events, model_name=model, include_usage=True),
        'kind': 'sse'},
    'openai.completions': {
        'decode': lambda body: openai_chat.decode_completion_request(body),
        'response': lambda resp, model: openai_chat.encode_response(resp, model_name=model, completion=True),
        'stream': lambda events, model: openai_chat.encode_stream(events, model_name=model, include_usage=True,
                                                                  completion=True),
        'kind': 'sse'},
    'openai.responses': {
        'decode': lambda body: responses.decode_request(body),
        'response': lambda resp, model: responses.encode_response(resp, _response_obj(model)),
        'stream': lambda events, model: responses.encode_stream(events, _response_obj(model)),
        'kind': 'sse'},
    'anthropic.messages': {
        'decode': lambda body: anthropic_messages.decode_request(body),
        'response': lambda resp, model: anthropic_messages.encode_response(resp, model_name=model),
        'stream': lambda events, model: anthropic_messages.encode_stream(events, model_name=model),
        'kind': 'sse'},
    'gemini.generateContent': {
        'decode': lambda body: gemini.decode_request(body, model=body.get('_model', 'models/alias'),
                                                     stream=bool(body.get('_stream'))),
        'response': lambda resp, model: gemini.encode_response(resp, model_name=model),
        'stream': lambda events, model: gemini.encode_stream(events, model_name=model),
        'kind': 'sse'},
    'ollama.chat': {
        'decode': lambda body: ollama.decode_request(body, generate=False),
        'response': lambda resp, model: ollama.encode_response(resp, model_name=model, generate=False),
        'stream': lambda events, model: ollama.encode_ndjson_stream(events, model_name=model, generate=False),
        'kind': 'ndjson'},
    'ollama.generate': {
        'decode': lambda body: ollama.decode_request(body, generate=True),
        'response': lambda resp, model: ollama.encode_response(resp, model_name=model, generate=True),
        'stream': lambda events, model: ollama.encode_ndjson_stream(events, model_name=model, generate=True),
        'kind': 'ndjson'},
}

HELLO = 'Say hello.'
REQUESTS = {
    'openai.chat': lambda stream: {'model': 'alias', 'messages': [{'role': 'system', 'content': 'Be brief.'},
                                                                  {'role': 'user', 'content': HELLO}],
                                   'stream': stream, 'max_tokens': 64},
    'openai.completions': lambda stream: {'model': 'alias', 'prompt': HELLO, 'stream': stream, 'max_tokens': 64},
    'openai.responses': lambda stream: {'model': 'alias', 'input': HELLO, 'instructions': 'Be brief.', 'stream': stream,
                                        'max_output_tokens': 64},
    'anthropic.messages': lambda stream: {'model': 'alias', 'max_tokens': 64, 'system': 'Be brief.',
                                          'messages': [{'role': 'user', 'content': HELLO}], 'stream': stream},
    'gemini.generateContent': lambda stream: {'contents': [{'role': 'user', 'parts': [{'text': HELLO}]}],
                                              'systemInstruction': {'parts': [{'text': 'Be brief.'}]},
                                              'generationConfig': {'maxOutputTokens': 64}, '_stream': stream},
    'ollama.chat': lambda stream: {'model': 'alias', 'messages': [{'role': 'user', 'content': HELLO}], 'stream': stream},
    'ollama.generate': lambda stream: {'model': 'alias', 'prompt': HELLO, 'stream': stream},
}


def inbound_request(inbound: str, stream: bool) -> dict:
    body = REQUESTS[inbound](stream)
    return {k: v for k, v in body.items() if not k.startswith('_')} if inbound != 'gemini.generateContent' else body


async def translate(target: MockTarget, inbound: str, *, stream: bool, body: dict | None = None):
    """Drive one translated cell: decode the inbound body, invoke the target adapter, encode back. Returns the encoded
    inbound answer (a dict, or the joined stream bytes)."""
    driver = INBOUND[inbound]
    ir = driver['decode'](body if body is not None else REQUESTS[inbound](stream))
    async with target.egress() as egress:
        outcome = await target.adapter.invoke(target.grant(), egress, ir)
        if stream:
            return b''.join([chunk async for chunk in driver['stream'](outcome, 'alias')])
        return driver['response'](outcome, 'alias')


def sse_events(data: bytes) -> list[tuple[str | None, str]]:
    splitter = SSESplitter()
    frames = [frame for _, frame in splitter.feed(data) if frame is not None]
    return [(f.event, f.data) for f in frames]


def ndjson_lines(data: bytes) -> list[dict]:
    splitter = NDJSONSplitter()
    return [v for _, v in splitter.feed(data) + splitter.finish() if v is not None]


def stream_text(inbound: str, data: bytes) -> str:
    """The text a client would assemble from an encoded inbound stream (and an assertion on its terminal frame)."""
    if INBOUND[inbound]['kind'] == 'ndjson':
        lines = ndjson_lines(data)
        assert lines and lines[-1].get('done') is True, lines[-1:]
        return ''.join((l.get('message') or {}).get('content') or l.get('response') or '' for l in lines)
    events = sse_events(data)
    if inbound in ('openai.chat', 'openai.completions'):
        assert events[-1][1] == '[DONE]', events[-1:]
        out = ''
        for _, payload in events[:-1]:
            chunk = json.loads(payload)
            for choice in chunk.get('choices') or []:
                out += (choice.get('delta') or {}).get('content') or choice.get('text') or ''
        return out
    if inbound == 'openai.responses':
        assert events[-1][0] in ('response.completed', 'response.incomplete'), events[-1:]
        return ''.join(json.loads(p)['delta'] for e, p in events if e == 'response.output_text.delta')
    if inbound == 'anthropic.messages':
        assert events[-1][0] == 'message_stop', events[-1:]
        return ''.join((json.loads(p).get('delta') or {}).get('text', '') for e, p in events if e == 'content_block_delta')
    if inbound == 'gemini.generateContent':
        last = json.loads(events[-1][1])
        assert last['candidates'][0].get('finishReason'), last
        return ''.join(part.get('text', '') for _, p in events for c in json.loads(p).get('candidates') or []
                       for part in (c.get('content') or {}).get('parts') or [])
    raise AssertionError(inbound)


def response_text(inbound: str, doc: dict) -> str:
    if inbound == 'openai.chat':
        return doc['choices'][0]['message']['content'] or ''
    if inbound == 'openai.completions':
        return doc['choices'][0]['text']
    if inbound == 'openai.responses':
        return ''.join(c['text'] for item in doc['output'] if item['type'] == 'message' for c in item['content'])
    if inbound == 'anthropic.messages':
        return ''.join(b.get('text', '') for b in doc['content'] if b['type'] == 'text')
    if inbound == 'gemini.generateContent':
        return ''.join(p.get('text', '') for p in doc['candidates'][0]['content']['parts'])
    if inbound == 'ollama.chat':
        return doc['message']['content']
    if inbound == 'ollama.generate':
        return doc['response']
    raise AssertionError(inbound)


# ---------------------------------------------------------------------------------------------------------------------
# runner with matrix-cell results

class _Recording(unittest.TextTestResult):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.passed: list[str] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(_test_id(test))


def _test_id(test) -> str:
    return f'{type(test).__name__}.{test._testMethodName}'


def main(cells: dict[str, str] | None = None) -> None:
    """Run the calling module's tests; every cell must cite a test that exists and passed."""
    module = sys.modules['__main__']
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    names = set()

    def walk(item):
        if isinstance(item, unittest.TestSuite):
            for child in item:
                walk(child)
        else:
            names.add(_test_id(item))
    walk(suite)
    runner = unittest.TextTestRunner(verbosity=1, resultclass=_Recording)
    result = runner.run(suite)
    cells = dict(cells or {})
    unknown = sorted(f'{cell} -> {test}' for cell, test in cells.items() if test not in names)
    unproven = sorted(f'{cell} -> {test}' for cell, test in cells.items() if test in names and test not in result.passed)
    if unknown:
        print('FAIL: matrix cells cite tests that do not exist:\n  ' + '\n  '.join(unknown))
    if unproven:
        print('FAIL: matrix cells cite tests that did not pass:\n  ' + '\n  '.join(unproven))
    if cells:
        print(f'matrix cells proven by this check: {len(cells) - len(unproven) - len(unknown)}/{len(cells)}')
    report = os.environ.get('TITANOS_BROKER_RESULTS')
    if report:
        failed = sorted(_test_id(t) for t, _ in result.failures + result.errors)
        with open(report, 'a', encoding='utf-8') as handle_:
            handle_.write(json.dumps({'check': Path(module.__file__).name, 'passed': sorted(result.passed),
                                      'failed': failed, 'cells': cells}, sort_keys=True) + '\n')
    sys.exit(0 if result.wasSuccessful() and not unknown and not unproven else 1)


__all__ = ['ADAPTERS', 'AZURE_API_VERSION', 'AZURE_RESOURCE', 'AzureHostTransport', 'Egress', 'HELLO', 'INBOUND', 'LOOPBACK',
           'MockTarget', 'REQUESTS', 'grant', 'handle', 'inbound_request', 'main', 'ndjson_lines', 'response_text', 'run',
           'sse_events', 'start_targets', 'stop_targets', 'stream_text', 'translate']
