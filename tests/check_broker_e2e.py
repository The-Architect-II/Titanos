"""Broker end to end over a REAL `python -B -m titanos --config <tmp>/models.json serve --port <free>` subprocess.

Every server runs with a fresh TITANOS_HOME, TITANOS_BROKER_EGRESS=loopback-only, TITANOS_BROKER_VAULT_BACKEND=refs-only and
PYTHONPATH carrying tests/broker_mocks/netguard_site with TITANOS_TEST_NETGUARD=1, so the subprocess itself refuses every
non-loopback socket and writes each refused attempt to a report file (the counter this check reads). broker.json points at the
WP4 loopback mocks over http; credentials reach the server only as env: references whose values are the WP4 sentinels, set
over HTTP (keyed) or with the offline CLI before serve (keyless and managed). Clients are raw httpx with each family's native
auth header, and the official MCP 2.2 streamable-HTTP client.

- keyed standalone (bootstrap key, minted tk_ keys): OpenAI chat (non-stream and stream, with tools; [DONE]), responses
  (non-stream and the stream event family), embeddings, speech, transcription, images, moderations and models; Anthropic
  /v1/messages with x-api-key (passthrough and translated, streaming tool calls both ways); Gemini generateContent and
  streamGenerateContent (x-goog-api-key; passthrough and translated) and the ?key= refusal; ElevenLabs TTS, /stream and STT
  (xi-api-key); Ollama /api/chat (passthrough non-stream and NDJSON stream, and translated); error envelopes per family;
  provenance headers on every answer; virtual-model fallback before the first byte only (an upstream 503 and an error-first
  stream advance, a mid-stream disconnect ends with the terminal frame and no second hit); the breaker opens after three
  5xx, answers without a hit while open and closes after a reset; budgets (402 before any hit, x-should-retry false),
  invisible models (404) and missing scopes (403); our own hop gives 508 loop_detected and a foreign hop stays local;
  decisions, usage and metrics with per-key visibility; the audit chain verifies; MCP chat, broker_status and broker_explain
  (zero egress) with a scoped key;
- keyless standalone (no bootstrap key, policy.anonymous_cloud): the offline CLI validates the env: credential against the
  mock; a placeholder `Bearer local` and no header both reach the cloud alias; credential and key writes and an
  anonymous_cloud change are refused (403 keyless_mode_refused); ?key= is refused even keyless;
- broker role: cloud keys need a budget and an expiry within 365 days (400 budget_required); the pre-dispatch budget stops a
  minted key with 402 and no further upstream hit;
- managed role without delegation: cloud targets give 403 role_forbids with zero hits, the parent trace is echoed as
  X-Request-ID, the vault is sealed, /api/version and /v1/health/ready are public, credential writes and upstream tests are
  409 role_forbids, a malformed parent deadline is 400;
- managed role with delegation (managed_delegation=parent): a delegate key with X-Titanos-Delegate: cloud and an explicit
  target reaches the mock; no header or an alias is 403 delegation_required; the parent deadline gives 504; a foreign hop
  stays local and our own hop is 508.
Each server shuts down cleanly (uvicorn's shutdown sequence after CTRL_BREAK on Windows, SIGINT elsewhere) and its netguard
report is empty; the test process's own netguard records nothing either. No real secret is used and nothing leaves loopback.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests'
for entry in (ROOT / 'src', TESTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from broker_mocks import (AnthropicMock, ElevenLabsMock, GeminiMock, OllamaCloudMock, OpenAIMock, SENTINELS,  # noqa: E402
                          netguard, wait_until)

netguard.install()

import httpx  # noqa: E402

NETGUARD_SITE = TESTS / 'broker_mocks' / 'netguard_site'
START_TIMEOUT = 90.0
BIND_REFUSED = 'error while attempting to bind'  # asyncio's OSError text, which uvicorn logs before it exits
STOP_TIMEOUT = 30.0
WINDOWS = sys.platform == 'win32'
# credential id -> (env variable the vault record names, WP4 sentinel the variable holds)
CRED_ENV = {'oa': ('E2E_CRED_OPENAI', 'openai'), 'bk': ('E2E_CRED_BACKUP', 'openai'), 'fk': ('E2E_CRED_FLAKY', 'openai'),
            'an': ('E2E_CRED_ANTHROPIC', 'anthropic'), 'gm': ('E2E_CRED_GEMINI', 'gemini'),
            'el': ('E2E_CRED_ELEVENLABS', 'elevenlabs'), 'ol': ('E2E_CRED_OLLAMA', 'ollama_cloud')}
SERVER_ENV_DROP = ('TITANOS_API_KEY', 'TITANOS_ROLE', 'TITANOS_PORT', 'TITANOS_HOST', 'TITANOS_BROKER_CONFIG',
                   'TITANOS_MANAGED_DELEGATION', 'TITANOS_BEHIND_TLS_PROXY', 'TITANOS_TLS_CERTFILE', 'TITANOS_TLS_KEYFILE',
                   'TITANOS_BROKER_VAULT_BACKEND', 'TITANOS_BROKER_VAULT_PASSPHRASE_FILE', 'TITANOS_BROKER_DECISION_LOG_FILE',
                   'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')

GPT = 'gpt-4o-mini'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
OPENAI_TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather for a city',
                                                  'parameters': {'type': 'object',
                                                                 'properties': {'city': {'type': 'string'}}}}}]
ANTHROPIC_TOOLS = [{'name': 'get_weather', 'description': 'Weather for a city',
                    'input_schema': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}]
GEMINI_BODY = {'contents': [{'role': 'user', 'parts': [{'text': 'Say hello.'}]}]}
WAV = (b'RIFF' + (36).to_bytes(4, 'little') + b'WAVEfmt ' + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little')
       + (1).to_bytes(2, 'little') + (16000).to_bytes(4, 'little') + (32000).to_bytes(4, 'little')
       + (2).to_bytes(2, 'little') + (16).to_bytes(2, 'little') + b'data' + (0).to_bytes(4, 'little'))
VOICE = 'JBFqnCBsd6RMkjVDRZzb'
ANTHROPIC_VERSION = '2023-06-01'
PROVENANCE = ('x-request-id', 'x-titanos-decision-id', 'x-titanos-fallback-count', 'x-titanos-requested-model',
              'x-titanos-served-by', 'x-titanos-served-model', 'x-titanos-translation')
PRICES = [{'match': '*', 'input_per_mtok': '1.00', 'output_per_mtok': '2.00', 'per_1k_chars': '0.01',
           'per_audio_minute': '0.01', 'per_image': '0.01'}]
MOCKS: dict = {}


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def synthetic_key(label: str) -> str:
    """A synthetic bootstrap key (never a real credential)."""
    return f'e2e-{label}-bootstrap-' + os.urandom(8).hex()


class Serve:
    """A real `titanos serve` subprocess over a fresh TITANOS_HOME (see the module docstring). sitecustomize: optional
    source of an extra sitecustomize module chained after the netguard one (the redaction check raises log levels)."""

    def __init__(self, name: str, document: dict | None, *, bootstrap: str | None = None, role: str | None = None,
                 env: dict | None = None, sitecustomize: str | None = None, log_level: str | None = None):
        self.name = name
        self.root = Path(tempfile.mkdtemp(prefix=f'titanos-e2e-{name}-'))
        self.home = self.root / 'home'
        self.home.mkdir()
        self.registry = self.root / 'models.json'
        self.registry.write_text('{"models": {}}\n', encoding='utf-8')
        self.report = self.root / 'netguard.jsonl'
        self.log_path = self.root / 'serve.log'
        self.bootstrap = bootstrap
        self.role = role
        self.log_level = log_level
        self.port = free_port()
        self.url = f'http://127.0.0.1:{self.port}'
        self.proc: subprocess.Popen | None = None
        self._log = None
        self._clients: list = []
        self.stopped_cleanly: bool | None = None
        self.exit_code: int | None = None
        if document is not None:
            from titanos.broker.config import BrokerConfig, write_broker_config
            write_broker_config(self.home / 'broker.json', BrokerConfig.model_validate(document))
        path_entries = [str(NETGUARD_SITE)]
        if sitecustomize is not None:
            extra = self.root / 'site'
            extra.mkdir()
            (extra / 'sitecustomize.py').write_text(sitecustomize, encoding='utf-8')
            path_entries.append(str(extra))
        path_entries.append(str(ROOT / 'src'))
        base = {k: v for k, v in os.environ.items() if k not in SERVER_ENV_DROP}
        base.update({'TITANOS_HOME': str(self.home), 'TITANOS_BROKER_EGRESS': 'loopback-only',
                     'TITANOS_BROKER_VAULT_BACKEND': 'refs-only', 'TITANOS_TEST_NETGUARD': '1',
                     'TITANOS_TEST_NETGUARD_REPORT': str(self.report), 'PYTHONPATH': os.pathsep.join(path_entries),
                     'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
                     'COLUMNS': '400', 'NO_COLOR': '1', 'TERM': 'dumb', 'NO_PROXY': '127.0.0.1,localhost'})
        for var, sentinel in CRED_ENV.values():
            base[var] = SENTINELS[sentinel]
        if bootstrap:
            base['TITANOS_API_KEY'] = bootstrap
        if role:
            base['TITANOS_ROLE'] = role
        base.update(env or {})
        self.env = base

    # processes ------------------------------------------------------------------------------------------------------
    def cli(self, *argv: str, stdin: str | None = None, timeout: float = 90.0, keyed: bool = False,
            url: bool = False) -> subprocess.CompletedProcess:
        """`python -B -m titanos --config <registry> <argv>` with the server's environment. Offline (the default) drops the
        bootstrap key and the role, as an operator configuring the home before serve; url=True appends --url and keeps
        the bootstrap key in TITANOS_API_KEY when keyed."""
        env = dict(self.env)
        env.pop('TITANOS_ROLE', None)
        if not keyed:
            env.pop('TITANOS_API_KEY', None)
        args = list(argv) + (['--url', self.url] if url else [])
        return subprocess.run([sys.executable, '-B', '-m', 'titanos', '--config', str(self.registry), *args], env=env,
                              cwd=str(self.root), input=stdin, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout)

    def set_credential_offline(self, cred_id: str, upstream: str) -> subprocess.CompletedProcess:
        done = self.cli('broker', 'credential', 'set', cred_id, '--upstream', upstream, '--env', CRED_ENV[cred_id][0])
        if done.returncode != 0:
            raise RuntimeError(f'offline credential set {cred_id} failed ({done.returncode}): {done.stderr[-2000:]}')
        return done

    def start(self, attempts: int = 3) -> 'Serve':
        """serve on self.port; a port another socket took after free_port closed its probe (uvicorn's bind error)
        moves to a fresh port, up to attempts times."""
        argv = [sys.executable, '-B', '-m', 'titanos', '--config', str(self.registry), 'serve', '--host', '127.0.0.1',
                '--port', str(self.port)]
        if self.log_level:
            argv += ['--log-level', self.log_level]
        kwargs = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {'start_new_session': True}
        self._log = open(self.log_path, 'wb')
        self.proc = subprocess.Popen(argv, env=self.env, cwd=str(self.root), stdout=self._log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, **kwargs)
        deadline = time.monotonic() + START_TIMEOUT
        with httpx.Client(trust_env=False, timeout=2.0) as probe:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    if attempts > 1 and BIND_REFUSED in self.log():
                        self._log.close()
                        self.port = free_port()
                        self.url = f'http://127.0.0.1:{self.port}'
                        return self.start(attempts - 1)
                    raise RuntimeError(f'serve exited with {self.proc.returncode}:\n{self.log()[-4000:]}')
                try:
                    if probe.get(self.url + '/healthz').status_code == 200:
                        return self
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
        self.kill()
        raise RuntimeError(f'serve did not answer /healthz within {START_TIMEOUT:.0f} s:\n{self.log()[-4000:]}')

    def stop(self) -> bool:
        """Ask for a graceful shutdown (CTRL_BREAK_EVENT to the process group on Windows, SIGINT elsewhere) and wait.
        True when the process exited in time and its log shows uvicorn's complete shutdown sequence."""
        for _, client in self._clients:
            client.close()
        self._clients.clear()
        if self.proc is None:
            return bool(self.stopped_cleanly)
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT if WINDOWS else signal.SIGINT)
            except OSError:
                pass
            try:
                self.proc.wait(STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self.kill()
                self.stopped_cleanly = False
        self.exit_code = self.proc.returncode
        if self._log is not None:
            self._log.close()
            self._log = None
        if self.stopped_cleanly is None:
            text = self.log()
            self.stopped_cleanly = 'Application shutdown complete' in text and 'Finished server process' in text
        self.proc = None
        return self.stopped_cleanly

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.kill()
            if self._log is not None:
                self._log.close()
                self._log = None
            shutil.rmtree(self.root, ignore_errors=True)

    # observations ---------------------------------------------------------------------------------------------------
    def log(self) -> str:
        if self._log is not None:
            self._log.flush()
        try:
            return self.log_path.read_bytes().decode('utf-8', 'replace')
        except OSError:
            return ''

    def netguard_attempts(self) -> list[dict]:
        if not self.report.exists():
            return []
        return [json.loads(line) for line in self.report.read_text(encoding='utf-8').splitlines() if line.strip()]

    def client(self, token: str | None = None, **headers) -> httpx.Client:
        """An httpx client for this server, one per distinct (token, headers) and reused, closed when the server stops.
        Each idle keep-alive connection holds one of uvicorn's limit_concurrency slots (64 by default since the
        2026-10-06 performance pass, 32 before) until keep_alive_timeout passes, so a fresh client per request made the
        server answer a plain 503 once a fast run had opened more than 32 connections within five seconds."""
        if token is not None:
            headers.setdefault('Authorization', 'Bearer ' + token)
        for key, client in self._clients:
            if key == headers and not client.is_closed:
                return client
        client = httpx.Client(base_url=self.url, trust_env=False, timeout=60.0, headers=headers)
        self._clients.append((dict(headers), client))
        return client

    def home_files(self) -> dict[str, bytes]:
        out = {}
        for path in sorted(self.home.rglob('*')):
            if path.is_file():
                try:
                    out[path.relative_to(self.home).as_posix()] = path.read_bytes()
                except OSError:
                    pass
        return out


# ---------------------------------------------------------------------------------------------------------------------
# mocks and wire helpers

def start_mocks() -> dict:
    mocks = {'openai': OpenAIMock(), 'backup': OpenAIMock(), 'flaky': OpenAIMock(), 'anthropic': AnthropicMock(),
             'gemini': GeminiMock(), 'elevenlabs': ElevenLabsMock(), 'ollama': OllamaCloudMock()}
    started = []
    try:
        for mock_ in mocks.values():
            mock_.start()
            started.append(mock_)
    except BaseException:
        for mock_ in started:
            mock_.stop()
        raise
    return mocks


def setUpModule():
    MOCKS.update(start_mocks())


def tearDownModule():
    for mock_ in MOCKS.values():
        mock_.gate.set()
        mock_.stop()
    MOCKS.clear()


def upstream(mock_key: str, profile: str, credential: str, models: list, **kw) -> dict:
    out = {'profile': profile, 'base_url': MOCKS[mock_key].api_base, 'enabled': True, 'credential': credential,
           'models': models}
    out.update(kw)
    return out


def hits(*names: str, surface: str | None = None) -> int:
    """Recorded requests on the named mocks (all when none are named), optionally on one surface."""
    chosen = [MOCKS[n] for n in names] if names else list(MOCKS.values())
    return sum(len(m.hits_for(surface)) for m in chosen)


def sse_events(text: str) -> list[tuple[str | None, object]]:
    """[(event name or None, parsed data or the raw string)] of an SSE body (comments skipped)."""
    out = []
    for block in text.replace('\r\n', '\n').split('\n\n'):
        event, data = None, []
        for line in block.split('\n'):
            if line.startswith('event:'):
                event = line[6:].strip()
            elif line.startswith('data:'):
                data.append(line[5:].lstrip())
        if not data:
            continue
        raw = '\n'.join(data)
        try:
            out.append((event, json.loads(raw)))
        except ValueError:
            out.append((event, raw))
    return out


def ndjson(text: str) -> list:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def mcp_calls(url: str, token: str, calls: list) -> list:
    """Run [(tool, args)] over ONE session of the official MCP 2.2 streamable-HTTP client with the given bearer key."""
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    async def run():
        out = []
        async with httpx2.AsyncClient(headers={'Authorization': 'Bearer ' + token}, trust_env=False) as http:
            async with Client(streamable_http_client(url + '/mcp/', http_client=http), read_timeout_seconds=60) as client:
                for name, args in calls:
                    out.append(await client.call_tool(name, args))
        return out
    return asyncio.run(run())


def in_days(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---------------------------------------------------------------------------------------------------------------------
# the served case base

class ServedCase(unittest.TestCase):
    """One real server per class. Subclasses set name, bootstrap, role, env, offline (cred id -> upstream, set with the
    CLI before serve) and http_credentials (cred id -> upstream, PUT over HTTP after start), and override document()
    and prepare()."""

    name = 'served'
    bootstrap: str | None = None
    role: str | None = None
    env: dict = {}
    offline: dict = {}
    http_credentials: dict = {}
    serve: Serve

    @classmethod
    def document(cls) -> dict | None:
        return None

    @classmethod
    def prepare(cls) -> None:
        """Runs after start (mint keys, ...)."""

    @classmethod
    def setUpClass(cls):
        if cls is ServedCase:
            raise unittest.SkipTest('the served base class has no server of its own')
        netguard.reset_attempts()
        for mock_ in MOCKS.values():
            mock_.reset()
        cls.serve = Serve(cls.name, cls.document(), bootstrap=cls.bootstrap, role=cls.role, env=dict(cls.env))
        try:
            cls.cli_outputs = [cls.serve.set_credential_offline(cid, up) for cid, up in cls.offline.items()]
            cls.serve.start()
            cls.admin = cls.serve.client(cls.bootstrap) if cls.bootstrap else cls.serve.client()
            cls.credential_replies = {}
            for cid, up in cls.http_credentials.items():
                reply = cls.admin.put(f'/api/broker/credentials/{cid}',
                                      json={'upstream': up, 'env': CRED_ENV[cid][0], 'validate': True})
                if reply.status_code not in (200, 201):
                    raise RuntimeError(f'credential {cid}: {reply.status_code} {reply.text[:300]}')
                cls.credential_replies[cid] = reply
            cls.prepare()
        except BaseException:
            cls.serve.close()
            raise

    @classmethod
    def tearDownClass(cls):
        for mock_ in MOCKS.values():
            mock_.gate.set()
            mock_.faults.clear()
        cls.serve.close()

    @classmethod
    def mint(cls, name: str, *scopes: str, **fields) -> tuple[str, str]:
        reply = cls.admin.post('/api/keys', json={'name': name, 'scopes': list(scopes), **fields})
        if reply.status_code != 201:
            raise RuntimeError(f'mint {name}: {reply.status_code} {reply.text[:300]}')
        body = reply.json()
        return body['id'], body['secret']

    def provenance(self, reply, served_by: str, translation: str, requested: str | None = None) -> None:
        for name in PROVENANCE:
            self.assertIn(name, reply.headers, (name, reply.status_code, reply.text[:200]))
        self.assertEqual(reply.headers['x-titanos-served-by'], served_by)
        self.assertEqual(reply.headers['x-titanos-translation'], translation)
        if requested is not None:
            self.assertEqual(reply.headers['x-titanos-requested-model'], requested)

    def refused(self, reply, status: int, code: str | None = None) -> None:
        self.assertEqual(reply.status_code, status, reply.text[:300])
        if code is not None:
            self.assertEqual(reply.json()['error']['code'], code, reply.text[:300])

    def test_zz_clean_shutdown_and_zero_non_loopback_attempts(self):
        """Runs last in each class (alphabetical order): graceful stop, empty subprocess netguard report, and nothing
        recorded by the test process's own guard."""
        self.assertTrue(self.serve.stop(), 'serve did not shut down cleanly:\n' + self.serve.log()[-3000:])
        self.assertEqual(self.serve.netguard_attempts(), [], 'the server attempted a non-loopback connection')
        self.assertEqual(netguard.attempts(), [], 'the test process attempted a non-loopback connection')


# ---------------------------------------------------------------------------------------------------------------------

class GuardSelfTest(unittest.TestCase):
    def test_the_subprocess_guard_refuses_and_reports(self):
        """The environment every Serve uses really installs the netguard in a child interpreter: a TEST-NET connect and a
        lookup are refused before any packet or DNS query, and both land in the report file."""
        serve = Serve('guard', None)
        try:
            code = ('import socket, sys\n'
                    'mod = sys.modules.get("titanos_test_netguard")\n'
                    'assert mod is not None and mod.installed(), "guard not installed"\n'
                    'for call in (lambda: socket.socket().connect(("192.0.2.1", 9)),\n'
                    '             lambda: socket.getaddrinfo("example.invalid", 443)):\n'
                    '    try:\n'
                    '        call()\n'
                    '    except OSError as exc:\n'
                    '        print(type(exc).__name__)\n'
                    '    else:\n'
                    '        raise SystemExit("not refused")\n')
            done = subprocess.run([sys.executable, '-B', '-c', code], env=serve.env, capture_output=True, text=True,
                                  timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            self.assertEqual(done.stdout.split(), ['NetguardBlocked', 'NetguardLookupBlocked'])
            self.assertEqual([(a['host'], a['via']) for a in serve.netguard_attempts()],
                             [('192.0.2.1', 'socket.connect'), ('example.invalid', 'getaddrinfo')])
        finally:
            serve.close()


class KeyedStandalone(ServedCase):
    name = 'keyed'
    bootstrap = synthetic_key('keyed')
    http_credentials = {'oa': 'openai', 'bk': 'backup', 'fk': 'flaky', 'an': 'anthropic', 'gm': 'gemini',
                        'el': 'elevenlabs', 'ol': 'ollama'}
    # an opened breaker stays open far longer than any test step, so a slow host cannot see it half-open early
    env = {'TITANOS_BROKER_BREAKER_BASE_SECONDS': '60', 'TITANOS_BROKER_BREAKER_MAX_SECONDS': '120'}

    @classmethod
    def document(cls):
        return {
            'policy': {'cloud_policy': 'fallback'},
            'upstreams': {
                'openai': upstream('openai', 'openai', 'oa', [GPT, 'text-embedding-3-small', 'gpt-4o-mini-tts', 'whisper-1',
                                                             'dall-e-3', 'omni-moderation-latest'],
                                   data_policy={'allow_store': True}),
                'backup': upstream('backup', 'openai', 'bk', [GPT]),
                'flaky': upstream('flaky', 'openai', 'fk', [GPT]),
                'anthropic': upstream('anthropic', 'anthropic', 'an', ['claude-sonnet-4-5']),
                'gemini': upstream('gemini', 'gemini', 'gm', ['gemini-2.5-flash']),
                'elevenlabs': upstream('elevenlabs', 'elevenlabs', 'el', ['eleven_multilingual_v2', 'scribe_v1']),
                'ollama': upstream('ollama', 'ollama_cloud', 'ol', ['gpt-oss:120b']),
            },
            'models': {'mini': {'target': GPT + '@openai'}},
            'virtual_models': {'vm': {'capability': 'chat', 'targets': [{'ref': GPT + '@openai'}, {'ref': GPT + '@backup'}]}},
            'prices': PRICES,
        }

    @classmethod
    def prepare(cls):
        cls.full_id, cls.full = cls.mint('full', 'inference', 'cloud', 'cloud:store', 'broker:read', 'models:read', 'mcp')
        cls.other_id, cls.other = cls.mint('other', 'inference', 'cloud', 'broker:read')
        cls.local_only_id, cls.local_only = cls.mint('nocloud', 'inference', 'broker:read')
        cls.tight_id, cls.tight = cls.mint('tight', 'inference', 'cloud', budget={'usd_per_day': '0.0000001'})
        cls.c = cls.serve.client(cls.full)

    def chat(self, model, token=None, **extra):
        return self.serve.client(token or self.full).post('/v1/chat/completions',
                                                          json={'model': model, 'messages': CHAT, **extra})

    # credentials --------------------------------------------------------------------------------------------------
    def test_credentials_are_validated_env_references(self):
        for cid, reply in self.credential_replies.items():
            body = reply.json()
            self.assertEqual((body['source'], body['env_var'], body['validation']), ('env', CRED_ENV[cid][0], 'ok'), cid)
            self.assertEqual(reply.headers.get('cache-control'), 'no-store')
            self.assertNotIn(SENTINELS[CRED_ENV[cid][1]], reply.text)
        listing = self.admin.get('/api/broker/credentials')
        self.assertEqual(listing.status_code, 200, listing.text[:300])
        rows = listing.json()['data'] if isinstance(listing.json(), dict) else listing.json()
        self.assertEqual({row['id'] for row in rows}, set(self.http_credentials))
        for cid, sentinel in ((cid, SENTINELS[s]) for cid, (_, s) in CRED_ENV.items()):
            self.assertNotIn(sentinel, listing.text)
        # each upstream was probed once (validation) with its own credential, never another one
        for name, mock_ in MOCKS.items():
            for hit in mock_.hits:
                for _, spelled in hit.credentials:
                    self.assertIn('<sentinel:', spelled, (name, hit.path))
                    self.assertNotIn('<foreign:', spelled, (name, hit.path))

    # OpenAI family --------------------------------------------------------------------------------------------------
    def test_openai_chat_with_tools_stream_and_non_stream(self):
        reply = self.chat(GPT + '@openai', tools=OPENAI_TOOLS)
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        self.provenance(reply, 'cloud:openai', 'passthrough', GPT + '@openai')
        self.assertEqual(reply.headers['x-titanos-fallback-count'], '0')
        self.assertTrue(reply.headers.get('x-titanos-upstream-request-id'))
        message = reply.json()['choices'][0]['message']
        self.assertEqual(message['tool_calls'][0]['function']['name'], 'get_weather')
        stream = self.chat(GPT + '@openai', tools=OPENAI_TOOLS, stream=True)
        self.assertEqual(stream.status_code, 200, stream.text[:300])
        self.assertTrue(stream.headers['content-type'].startswith('text/event-stream'))
        self.provenance(stream, 'cloud:openai', 'passthrough')
        events = sse_events(stream.text)
        self.assertEqual(events[-1][1], '[DONE]')
        deltas = [e[1]['choices'][0]['delta'] for e in events[:-1] if isinstance(e[1], dict) and e[1].get('choices')]
        self.assertTrue(any(d.get('tool_calls') for d in deltas), 'no streamed tool call delta')
        alias = self.chat('mini')
        self.assertEqual(alias.status_code, 200, alias.text[:300])
        self.provenance(alias, 'cloud:openai', 'passthrough', 'mini')
        self.assertEqual(alias.headers['x-titanos-served-model'], GPT + '@openai')

    def test_openai_responses_embeddings_audio_images_moderations_models(self):
        responses = self.c.post('/v1/responses', json={'model': GPT + '@openai', 'input': 'hi'})
        self.assertEqual(responses.status_code, 200, responses.text[:300])
        self.provenance(responses, 'cloud:openai', 'passthrough')
        self.assertEqual(responses.json()['object'], 'response')
        stream = self.c.post('/v1/responses', json={'model': GPT + '@openai', 'input': 'hi', 'stream': True})
        self.assertEqual(stream.status_code, 200, stream.text[:300])
        events = [e for e in sse_events(stream.text) if isinstance(e[1], dict)]
        kinds = [data.get('type') for _, data in events]
        self.assertEqual(kinds[0], 'response.created')
        self.assertEqual(kinds[-1], 'response.completed')
        numbers = [data.get('sequence_number') for _, data in events]
        self.assertEqual(numbers, sorted(numbers))
        embeddings = self.c.post('/v1/embeddings', json={'model': 'text-embedding-3-small@openai', 'input': ['a', 'b']})
        self.assertEqual(embeddings.status_code, 200, embeddings.text[:300])
        self.assertEqual(len(embeddings.json()['data']), 2)
        self.provenance(embeddings, 'cloud:openai', 'passthrough')
        speech = self.c.post('/v1/audio/speech', json={'model': 'gpt-4o-mini-tts@openai', 'input': 'hi', 'voice': 'alloy',
                                                       'response_format': 'wav'})
        self.assertEqual(speech.status_code, 200, speech.text[:300])
        self.assertTrue(speech.headers['content-type'].startswith('audio/'))
        self.assertGreater(len(speech.content), 0)
        self.provenance(speech, 'cloud:openai', 'passthrough')
        transcription = self.c.post('/v1/audio/transcriptions', data={'model': 'whisper-1@openai'},
                                    files={'file': ('hello.wav', WAV, 'audio/wav')})
        self.assertEqual(transcription.status_code, 200, transcription.text[:300])
        self.assertIsInstance(transcription.json()['text'], str)
        self.provenance(transcription, 'cloud:openai', 'passthrough')
        hit = MOCKS['openai'].hits_for('openai.audio.transcriptions')[-1]
        self.assertIn(WAV, hit.body)
        images = self.c.post('/v1/images/generations', json={'model': 'dall-e-3@openai', 'prompt': 'a cat',
                                                             'response_format': 'b64_json'})
        self.assertEqual(images.status_code, 200, images.text[:300])
        self.assertTrue(images.json()['data'][0]['b64_json'])
        moderation = self.c.post('/v1/moderations', json={'model': 'omni-moderation-latest@openai', 'input': 'hi'})
        self.assertEqual(moderation.status_code, 200, moderation.text[:300])
        self.assertIn('flagged', moderation.json()['results'][0])
        models = self.c.get('/v1/models')
        self.assertEqual(models.status_code, 200, models.text[:300])
        ids = {row['id'] for row in models.json()['data']}
        self.assertTrue({'mini', 'vm'} <= ids, sorted(ids)[:20])

    # Anthropic --------------------------------------------------------------------------------------------------------
    def test_anthropic_messages_passthrough_and_translated_with_streaming_tools(self):
        client = self.serve.client(None, **{'x-api-key': self.full, 'anthropic-version': ANTHROPIC_VERSION})
        passthrough = client.post('/v1/messages', json={'model': 'claude-sonnet-4-5@anthropic', 'max_tokens': 64,
                                                        'messages': CHAT})
        self.assertEqual(passthrough.status_code, 200, passthrough.text[:300])
        self.provenance(passthrough, 'cloud:anthropic', 'passthrough')
        self.assertEqual(passthrough.json()['type'], 'message')
        hit = MOCKS['anthropic'].hits_for('anthropic.messages')[-1]
        self.assertEqual(hit.credential, '<sentinel:anthropic>')
        self.assertFalse(hit.presented(self.full))
        translated = client.post('/v1/messages', json={'model': GPT + '@openai', 'max_tokens': 64, 'messages': CHAT})
        self.assertEqual(translated.status_code, 200, translated.text[:300])
        self.provenance(translated, 'cloud:openai', 'anthropic->openai')
        self.assertEqual(translated.json()['type'], 'message')
        self.assertEqual(MOCKS['openai'].hits_for('openai.chat')[-1].credential, 'Bearer <sentinel:openai>')
        for model, translation in (('claude-sonnet-4-5@anthropic', 'passthrough'), (GPT + '@openai', 'anthropic->openai')):
            with self.subTest(model=model):
                stream = client.post('/v1/messages', json={'model': model, 'max_tokens': 64, 'messages': CHAT,
                                                           'tools': ANTHROPIC_TOOLS, 'stream': True})
                self.assertEqual(stream.status_code, 200, stream.text[:300])
                self.assertEqual(stream.headers['x-titanos-translation'], translation)
                events = sse_events(stream.text)
                names = [name for name, _ in events]
                self.assertEqual((names[0], names[-1]), ('message_start', 'message_stop'))
                blocks = [data['content_block'] for name, data in events if name == 'content_block_start']
                self.assertIn('tool_use', {block['type'] for block in blocks})

    # Gemini -----------------------------------------------------------------------------------------------------------
    def test_gemini_generate_and_stream_with_the_native_header(self):
        client = self.serve.client(None, **{'x-goog-api-key': self.full})
        generated = client.post('/v1beta/models/gemini-2.5-flash@gemini:generateContent', json=GEMINI_BODY)
        self.assertEqual(generated.status_code, 200, generated.text[:300])
        self.provenance(generated, 'cloud:gemini', 'passthrough')
        self.assertTrue(generated.json()['candidates'][0]['content']['parts'][0]['text'])
        self.assertEqual(MOCKS['gemini'].hits[-1].credential, '<sentinel:gemini>')
        stream = client.post('/v1beta/models/gemini-2.5-flash@gemini:streamGenerateContent?alt=sse', json=GEMINI_BODY)
        self.assertEqual(stream.status_code, 200, stream.text[:300])
        self.assertTrue(stream.headers['content-type'].startswith('text/event-stream'))
        chunks = [data for _, data in sse_events(stream.text)]
        self.assertTrue(chunks and all('candidates' in chunk for chunk in chunks if isinstance(chunk, dict)))
        self.assertEqual(MOCKS['gemini'].hits[-1].query_params.get('alt'), 'sse')
        translated = client.post(f'/v1beta/models/{GPT}@openai:generateContent', json=GEMINI_BODY)
        self.assertEqual(translated.status_code, 200, translated.text[:300])
        self.provenance(translated, 'cloud:openai', 'gemini->openai')
        before = hits()
        query = self.serve.client(None).post(f'/v1beta/models/gemini-2.5-flash@gemini:generateContent?key={self.full}',
                                             json=GEMINI_BODY)
        self.assertEqual(query.status_code, 401, query.text[:300])
        self.assertEqual(query.json()['error']['details'][0]['reason'], 'QUERY_CREDENTIAL_REFUSED')
        self.assertNotIn(self.full, query.text)
        self.assertEqual(hits(), before)

    # ElevenLabs -------------------------------------------------------------------------------------------------------
    def test_elevenlabs_tts_stream_and_stt_with_xi_api_key(self):
        client = self.serve.client(None, **{'xi-api-key': self.full})
        body = {'text': 'Hello there.', 'model_id': 'eleven_multilingual_v2@elevenlabs'}
        tts = client.post(f'/v1/text-to-speech/{VOICE}?output_format=mp3_44100_128', json=body)
        self.assertEqual(tts.status_code, 200, tts.text[:300])
        self.assertEqual(tts.headers['content-type'], 'audio/mpeg')
        self.assertGreater(len(tts.content), 0)
        self.provenance(tts, 'cloud:elevenlabs', 'passthrough')
        hit = MOCKS['elevenlabs'].hits_for('elevenlabs.tts')[-1]
        self.assertEqual((hit.path, hit.credential), (f'/v1/text-to-speech/{VOICE}', '<sentinel:elevenlabs>'))
        self.assertEqual(hit.json()['model_id'], 'eleven_multilingual_v2')
        stream = client.post(f'/v1/text-to-speech/{VOICE}/stream?output_format=mp3_44100_128', json=body)
        self.assertEqual(stream.status_code, 200, stream.text[:300])
        self.assertGreater(len(stream.content), 0)
        self.assertEqual(MOCKS['elevenlabs'].hits[-1].path, f'/v1/text-to-speech/{VOICE}/stream')
        stt = client.post('/v1/speech-to-text', data={'model_id': 'scribe_v1@elevenlabs'},
                          files={'file': ('hello.wav', WAV, 'audio/wav')})
        self.assertEqual(stt.status_code, 200, stt.text[:300])
        self.assertIsInstance(stt.json()['text'], str)
        self.provenance(stt, 'cloud:elevenlabs', 'passthrough')

    # Ollama -----------------------------------------------------------------------------------------------------------
    def test_ollama_chat_passthrough_stream_and_translated(self):
        single = self.c.post('/api/chat', json={'model': 'gpt-oss:120b@ollama', 'messages': CHAT, 'stream': False})
        self.assertEqual(single.status_code, 200, single.text[:300])
        self.provenance(single, 'cloud:ollama', 'passthrough')
        self.assertTrue(single.json()['done'])
        self.assertEqual(MOCKS['ollama'].hits[-1].credential, 'Bearer <sentinel:ollama_cloud>')
        stream = self.c.post('/api/chat', json={'model': 'gpt-oss:120b@ollama', 'messages': CHAT})
        self.assertEqual(stream.status_code, 200, stream.text[:300])
        self.assertIn('ndjson', stream.headers['content-type'])
        lines = ndjson(stream.text)
        self.assertTrue(lines[-1]['done'])
        translated = self.c.post('/api/chat', json={'model': GPT + '@openai', 'messages': CHAT, 'stream': False})
        self.assertEqual(translated.status_code, 200, translated.text[:300])
        self.provenance(translated, 'cloud:openai', 'ollama->openai')
        self.assertTrue(translated.json()['message']['content'])

    # envelopes --------------------------------------------------------------------------------------------------------
    def test_error_envelopes_per_family_with_zero_hits(self):
        before = hits()
        openai = self.chat('nope@openai')
        self.refused(openai, 404, 'model_not_found')
        anthropic = self.serve.client(None, **{'x-api-key': self.full, 'anthropic-version': ANTHROPIC_VERSION}).post(
            '/v1/messages', json={'model': 'nope@anthropic', 'max_tokens': 8, 'messages': CHAT})
        self.assertEqual(anthropic.status_code, 404, anthropic.text[:300])
        self.assertEqual((anthropic.json()['type'], anthropic.json()['error']['type']), ('error', 'not_found_error'))
        gemini = self.serve.client(None, **{'x-goog-api-key': self.full}).post(
            '/v1beta/models/nope@gemini:generateContent', json=GEMINI_BODY)
        self.assertEqual((gemini.status_code, gemini.json()['error']['status']), (404, 'NOT_FOUND'))
        ollama = self.c.post('/api/chat', json={'model': 'nope@ollama', 'messages': CHAT, 'stream': False})
        self.assertEqual(ollama.status_code, 404, ollama.text[:300])
        self.assertIsInstance(ollama.json()['error'], str)
        eleven = self.serve.client(None, **{'xi-api-key': self.full}).post(
            f'/v1/text-to-speech/{VOICE}', json={'text': 'hi', 'model_id': 'nope@elevenlabs'})
        self.assertEqual(eleven.status_code, 404, eleven.text[:300])
        self.assertIn('status', eleven.json()['detail'])
        for reply in (openai, anthropic, gemini, ollama, eleven):
            self.assertIn(reply.headers.get('x-titanos-error-origin'), ('titanos', 'policy'))
        self.assertEqual(hits(), before)

    # routing ------------------------------------------------------------------------------------------------------------
    def test_virtual_model_falls_back_before_the_first_byte_only(self):
        self.addCleanup(self.admin.post, '/api/broker/upstreams/openai/breaker/reset')
        openai_before = hits('openai', surface='openai.chat')
        backup_before = hits('backup', surface='openai.chat')
        MOCKS['openai'].faults.push('status', status=503, surface='openai.chat')
        reply = self.chat('vm')
        self.assertEqual(reply.status_code, 200, reply.text[:300])
        self.provenance(reply, 'cloud:backup', 'passthrough', 'vm')
        self.assertEqual(reply.headers['x-titanos-fallback-count'], '1')
        self.assertEqual(reply.headers['x-titanos-served-model'], GPT + '@backup')
        fallback_decision = reply.headers['x-titanos-decision-id']
        self.assertEqual((hits('openai', surface='openai.chat'), hits('backup', surface='openai.chat')),
                         (openai_before + 1, backup_before + 1))
        # an error event before any byte reached the client still falls back
        MOCKS['openai'].faults.push('error_first_event', surface='openai.chat')
        early = self.chat('vm', stream=True)
        self.assertEqual(early.status_code, 200, early.text[:300])
        self.assertEqual((early.headers['x-titanos-served-by'], early.headers['x-titanos-fallback-count']),
                         ('cloud:backup', '1'))
        self.assertNotIn('"error"', early.text)
        self.assertEqual(sse_events(early.text)[-1][1], '[DONE]')
        # after the first byte: no fallback, the terminal frame, and no second upstream hit
        backup_before = hits('backup', surface='openai.chat')
        MOCKS['openai'].faults.push('disconnect_after_frames', frames=2, surface='openai.chat')
        late = self.chat('vm', stream=True)
        self.assertEqual(late.status_code, 200, late.text[:300])
        self.assertEqual((late.headers['x-titanos-served-by'], late.headers['x-titanos-fallback-count']),
                         ('cloud:openai', '0'))
        events = sse_events(late.text)
        self.assertEqual(events[-1][1], '[DONE]')
        self.assertEqual(events[-2][1]['error']['type'], 'stream_interrupted')
        self.assertEqual(hits('backup', surface='openai.chat'), backup_before)
        decisions = self.admin.get('/api/broker/decisions', params={'limit': 200}).json()['data']
        by_id = {d['decision_id']: d for d in decisions}
        self.assertEqual(by_id[fallback_decision]['fallback_count'], 1)
        self.assertEqual(len(by_id[fallback_decision]['attempts']), 2)
        self.assertEqual(by_id[late.headers['x-titanos-decision-id']]['committed_at_attempt'], 0)

    def test_breaker_opens_answers_without_a_hit_and_resets(self):
        flaky = MOCKS['flaky']
        MOCKS['flaky'].faults.push('status', status=500, surface='openai.chat', times=3)
        for _ in range(3):
            failed = self.chat(GPT + '@flaky')
            self.assertGreaterEqual(failed.status_code, 500, failed.text[:300])
            self.assertEqual(failed.headers.get('x-titanos-error-origin'), 'upstream')
        before = len(flaky.hits)
        opened = self.chat(GPT + '@flaky')
        self.assertEqual(opened.status_code, 503, opened.text[:300])
        self.assertEqual(opened.headers.get('x-titanos-error-origin'), 'titanos')
        self.assertIn('retry-after', opened.headers)
        self.assertEqual(len(flaky.hits), before, 'an open breaker still reached the upstream')
        health = self.admin.get('/api/broker/upstreams/flaky/health').json()
        self.assertEqual((health['breaker'], health['opens']), ('open', 1))
        self.assertEqual(health['last_outcomes'][-3:], ['server_error'] * 3)
        reset = self.admin.post('/api/broker/upstreams/flaky/breaker/reset')
        self.assertEqual(reset.status_code, 200, reset.text[:300])
        healed = self.chat(GPT + '@flaky')
        self.assertEqual(healed.status_code, 200, healed.text[:300])
        self.assertEqual(len(flaky.hits), before + 1)
        self.assertEqual(self.admin.get('/api/broker/upstreams/flaky/health').json()['breaker'], 'closed')

    def test_budget_visibility_and_scope_refusals_happen_before_any_hit(self):
        before = hits()
        budget = self.chat(GPT + '@openai', token=self.tight)
        self.refused(budget, 402, 'budget_exceeded')
        self.assertEqual(budget.headers.get('x-should-retry'), 'false')
        self.assertTrue(budget.headers.get('x-titanos-decision-id'))
        invisible = self.chat(GPT + '@openai', token=self.local_only)
        self.refused(invisible, 404, 'model_not_found')
        alias = self.chat('mini', token=self.local_only)
        self.refused(alias, 404, 'model_not_found')
        scope = self.serve.client(self.other).get('/v1/models')
        self.refused(scope, 403, 'scope_denied')
        wrong = self.serve.client('tk_' + 'a' * 10 + '_' + 'b' * 43).post('/v1/chat/completions',
                                                                           json={'model': 'mini', 'messages': CHAT})
        self.assertEqual(wrong.status_code, 401, wrong.text[:300])
        self.assertEqual(hits(), before)

    def test_hop_loop_is_508_and_a_foreign_hop_stays_local(self):
        hop = self.admin.get('/api/broker').json()['instance_hop']
        self.assertTrue(hop)
        before = hits()
        loop = self.serve.client(self.full).post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT},
                                                 headers={'X-Titanos-Hop': hop})
        self.refused(loop, 508, 'loop_detected')
        self.assertEqual(loop.headers.get('x-should-retry'), 'false')
        foreign = self.serve.client(self.full).post('/v1/chat/completions',
                                                    json={'model': GPT + '@openai', 'messages': CHAT},
                                                    headers={'X-Titanos-Hop': 'another-instance'})
        self.refused(foreign, 400, 'location_unsatisfiable')
        self.assertEqual(hits(), before)

    # observability ------------------------------------------------------------------------------------------------------
    def test_decisions_usage_metrics_and_audit(self):
        own = self.chat(GPT + '@openai', token=self.other)
        self.assertEqual(own.status_code, 200, own.text[:300])
        mine = self.serve.client(self.other).get('/api/broker/decisions', params={'limit': 100})
        self.assertEqual(mine.status_code, 200, mine.text[:300])
        rows = mine.json()['data']
        self.assertTrue(rows)
        self.assertEqual({row['key_id'] for row in rows}, {self.other_id})
        self.assertIn(own.headers['x-titanos-decision-id'], {row['decision_id'] for row in rows})
        decision = next(row for row in rows if row['decision_id'] == own.headers['x-titanos-decision-id'])
        self.assertEqual((decision['served']['upstream'], decision['served']['klass'], decision['surface']),
                         ('openai', 'cloud', 'openai.chat'))
        peek = self.serve.client(self.other).get('/api/broker/decisions', params={'key_id': self.full_id})
        self.assertEqual((peek.status_code, peek.json()['data']), (200, []))  # filtered to the caller's own key
        self.refused(self.serve.client(self.other).get('/api/broker/usage', params={'key_id': self.full_id}), 403,
                     'scope_denied')
        everything = self.admin.get('/api/broker/decisions', params={'limit': 1000}).json()['data']
        self.assertTrue({self.other_id, self.full_id} <= {row['key_id'] for row in everything})
        usage = self.serve.client(self.other).get('/api/broker/usage', params={'group_by': 'key,upstream'}).json()['rows']
        self.assertEqual({row['key'] for row in usage}, {self.other_id})
        self.assertGreaterEqual(sum(row['requests'] for row in usage), 1)
        self.assertTrue(all(row['cost_usd'] is not None for row in usage))
        all_usage = self.admin.get('/api/broker/usage', params={'group_by': 'upstream'}).json()['rows']
        self.assertTrue({'openai'} <= {row['upstream'] for row in all_usage})
        metrics = self.admin.get('/api/broker/metrics')
        self.assertEqual(metrics.status_code, 200, metrics.text[:300])
        self.assertTrue(metrics.json()['requests_by_surface_class_outcome'])
        verify = self.admin.get('/api/audit/verify')
        self.assertEqual(verify.status_code, 200, verify.text[:300])
        self.assertTrue(verify.json()['ok'], verify.text[:300])
        self.assertGreater(verify.json()['records'], len(self.http_credentials))
        audit = self.admin.get('/api/audit', params={'limit': 500})
        self.assertEqual(audit.status_code, 200, audit.text[:300])
        kinds = {row.get('kind') for row in (audit.json()['data'] if isinstance(audit.json(), dict) else audit.json())}
        self.assertTrue(any(kind and kind.startswith('credential') for kind in kinds), kinds)
        self.assertTrue(any(kind and kind.startswith('key') for kind in kinds), kinds)
        self.refused(self.serve.client(self.other).get('/api/audit/verify'), 403, 'scope_denied')

    # MCP -----------------------------------------------------------------------------------------------------------------
    def test_mcp_chat_status_and_explain_with_a_scoped_key(self):
        chat_before = hits('openai', surface='openai.chat')
        out = mcp_calls(self.serve.url, self.full, [
            ('chat', {'payload': {'model': 'mini', 'messages': CHAT}}),
            ('broker_status', {}),
        ])
        chat, status = out
        self.assertFalse(chat.is_error, str(chat.content)[:300])
        self.assertEqual(chat.meta['served_by'], 'cloud:openai')
        self.assertEqual(chat.meta['served_model'], GPT + '@openai')
        self.assertTrue(chat.meta['decision_id'])
        self.assertEqual(chat.structured_content['object'], 'chat.completion')
        self.assertEqual(hits('openai', surface='openai.chat'), chat_before + 1)
        last = MOCKS['openai'].hits_for('openai.chat')[-1]
        self.assertFalse(last.presented(self.full))
        self.assertFalse(status.is_error, str(status.content)[:300])
        self.assertEqual(status.structured_content['role'], 'standalone')
        self.assertEqual(status.structured_content['egress'], 'loopback-only')
        before = hits()
        (explain,) = mcp_calls(self.serve.url, self.full, [
            ('broker_explain', {'surface': 'openai.chat', 'body': {'model': 'vm', 'messages': CHAT}})])
        self.assertFalse(explain.is_error, str(explain.content)[:300])
        decision = explain.structured_content
        self.assertTrue(decision['dry_run'])
        self.assertIsNone(decision['served'])
        self.assertEqual(decision['chain'][:2], [GPT + '@openai', GPT + '@backup'])
        self.assertEqual(hits(), before, 'explain reached an upstream')
        _, no_cloud = self.mint('mcp-local', 'mcp', 'inference')
        (refused,) = mcp_calls(self.serve.url, no_cloud, [('chat', {'payload': {'model': 'mini', 'messages': CHAT}})])
        self.assertTrue(refused.is_error)
        self.assertIn('HTTP 404', str(refused.content))
        with self.assertRaises(Exception):  # a key without the mcp scope cannot open a session at all
            mcp_calls(self.serve.url, self.local_only, [('broker_status', {})])
        self.assertEqual(hits(), before)


class KeylessStandalone(ServedCase):
    name = 'keyless'
    offline = {'oa': 'openai'}

    @classmethod
    def document(cls):
        return {'policy': {'cloud_policy': 'explicit', 'anonymous_cloud': True},
                'upstreams': {'openai': upstream('openai', 'openai', 'oa', [GPT])},
                'models': {'mini': {'target': GPT + '@openai'}}}

    def test_offline_cli_validated_the_env_reference_against_the_mock(self):
        (done,) = self.cli_outputs
        record = json.loads(done.stdout)
        self.assertEqual((record['source'], record['env_var'], record['validation']), ('env', 'E2E_CRED_OPENAI', 'ok'))
        self.assertNotIn(SENTINELS['openai'], done.stdout + done.stderr)
        self.assertNotIn('E2E_CRED_OPENAI', json.dumps(self.serve.home_files().get('broker.json', b'').decode()))

    def test_placeholder_and_absent_keys_reach_the_cloud_alias(self):
        for client in (self.serve.client('local'), self.serve.client()):
            reply = client.post('/v1/chat/completions', json={'model': 'mini', 'messages': CHAT})
            self.assertEqual(reply.status_code, 200, reply.text[:300])
            self.provenance(reply, 'cloud:openai', 'passthrough', 'mini')
            hit = MOCKS['openai'].hits_for('openai.chat')[-1]
            self.assertEqual(hit.credential, 'Bearer <sentinel:openai>')
            self.assertFalse(hit.presented('local'))
        stream = self.serve.client('local').post('/v1/chat/completions',
                                                 json={'model': GPT + '@openai', 'messages': CHAT, 'stream': True})
        self.assertEqual(stream.status_code, 200, stream.text[:300])
        self.assertEqual(sse_events(stream.text)[-1][1], '[DONE]')

    def test_keyless_writes_are_refused(self):
        before = hits()
        c = self.serve.client()
        self.refused(c.put('/api/broker/credentials/x', json={'upstream': 'openai', 'env': 'E2E_CRED_OPENAI'}), 403,
                     'keyless_mode_refused')
        self.refused(c.post('/api/keys', json={'name': 'x', 'scopes': ['inference']}), 403, 'keyless_mode_refused')
        shown = c.get('/api/broker/policy').json()
        policy = {name: shown[name] for name in ('cloud_policy', 'default_location', 'anonymous_cloud', 'allow_http_lan',
                                                 'max_fallbacks', 'auto_model')}
        policy['anonymous_cloud'] = False
        self.refused(c.put('/api/broker/policy', json=policy), 403, 'keyless_mode_refused')
        query = c.post('/v1beta/models/mini:generateContent?key=e2e-query-credential-0001', json=GEMINI_BODY)
        self.assertEqual(query.status_code, 401, query.text[:300])
        self.assertNotIn('e2e-query-credential-0001', query.text)
        info = c.get('/api/info').json()
        self.assertEqual((info['role'], info['standalone'], info['broker']['configured']), ('standalone', True, True))
        self.assertEqual(hits(), before)


class BrokerRole(ServedCase):
    name = 'broker'
    bootstrap = synthetic_key('broker')
    role = 'broker'
    http_credentials = {'oa': 'openai'}

    @classmethod
    def document(cls):
        return {'policy': {'cloud_policy': 'explicit'},
                'upstreams': {'openai': upstream('openai', 'openai', 'oa', [GPT])},
                'prices': [{'match': GPT + '@openai', 'input_per_mtok': '1.00', 'output_per_mtok': '2.00'}]}

    def test_cloud_keys_need_a_budget_and_an_expiry(self):
        for fields, needle in (({}, 'budget'), ({'budget': {'usd_per_day': '1.00'}}, 'expires_at'),
                               ({'budget': {'usd_per_day': '1.00'}, 'expires_at': in_days(400)}, '365')):
            reply = self.admin.post('/api/keys', json={'name': 'x', 'scopes': ['inference', 'cloud'], **fields})
            self.refused(reply, 400, 'budget_required')
            self.assertIn(needle, reply.json()['error']['message'])
        info = self.admin.get('/api/info').json()
        self.assertEqual((info['role'], info['orchestrated']), ('broker', False))

    def test_the_pre_dispatch_budget_stops_a_minted_key(self):
        _, token = self.mint('metered', 'inference', 'cloud', budget={'usd_per_day': '0.0001'}, expires_at=in_days(30))
        client = self.serve.client(token)
        served, stopped = 0, None
        for _ in range(12):
            before = hits('openai', surface='openai.chat')
            reply = client.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT, 'max_tokens': 10})
            if reply.status_code == 200:
                served += 1
                self.assertEqual(hits('openai', surface='openai.chat'), before + 1)
                continue
            stopped = reply
            self.assertEqual(hits('openai', surface='openai.chat'), before, 'a refused request reached the upstream')
            break
        self.assertGreaterEqual(served, 1)
        self.assertIsNotNone(stopped, 'the budget never stopped the key')
        self.refused(stopped, 402, 'budget_exceeded')
        self.assertEqual(stopped.headers.get('x-should-retry'), 'false')
        before = hits()
        again = client.post('/v1/chat/completions', json={'model': GPT + '@openai', 'messages': CHAT, 'max_tokens': 10})
        self.refused(again, 402, 'budget_exceeded')
        self.assertEqual(hits(), before)
        me = client.get('/api/keys/self').json()
        self.assertIsNotNone(me['expires_at'])
        self.assertIn('budget_remaining', me)


class ManagedRole(ServedCase):
    name = 'managed'
    bootstrap = synthetic_key('managed')
    role = 'managed'
    offline = {'oa': 'openai'}

    @classmethod
    def document(cls):
        return {'policy': {'cloud_policy': 'explicit'},
                'upstreams': {'openai': upstream('openai', 'openai', 'oa', [GPT])},
                'models': {'mini': {'target': GPT + '@openai'}}}

    def test_cloud_targets_are_refused_with_zero_hits_and_the_parent_trace_echoed(self):
        before = hits()
        for model in (GPT + '@openai', 'mini'):
            reply = self.admin.post('/v1/chat/completions', json={'model': model, 'messages': CHAT},
                                    headers={'x-hivemind-trace': 'trace-e2e-0001'})
            self.refused(reply, 403, 'role_forbids')
            self.assertEqual(reply.headers['x-request-id'], 'trace-e2e-0001')
        malformed = self.admin.post('/v1/chat/completions', json={'model': 'mini', 'messages': CHAT},
                                    headers={'X-Titanos-Deadline-Ms': 'soon'})
        self.refused(malformed, 400, 'invalid_request')
        self.assertEqual(hits(), before)

    def test_sealed_vault_public_routes_and_refused_writes(self):
        status = self.admin.get('/api/broker').json()
        self.assertEqual((status['role'], status['orchestrated'], status['vault']['sealed'],
                          status['cloud_policy']['effective']), ('managed', True, True, 'off'))
        public = self.serve.client()
        version = public.get('/api/version')
        self.assertEqual((version.status_code, version.json()['role']), (200, 'managed'))
        ready = public.get('/v1/health/ready')
        self.assertEqual((ready.status_code, ready.json()['status'], ready.json()['role']), (200, 'ready', 'managed'))
        self.assertEqual(public.get('/v1/models').status_code, 401)
        before = hits()
        self.refused(self.admin.put('/api/broker/credentials/oa2', json={'upstream': 'openai', 'env': 'E2E_CRED_OPENAI'}),
                     409, 'role_forbids')
        self.refused(self.admin.post('/api/broker/upstreams/openai/test'), 409, 'role_forbids')
        minted = self.admin.post('/api/keys', json={'name': 'member', 'scopes': ['inference']})
        self.assertEqual(minted.status_code, 201, minted.text[:300])
        self.assertEqual(hits(), before)


class ManagedDelegation(ServedCase):
    name = 'delegated'
    bootstrap = synthetic_key('delegated')
    role = 'managed'
    env = {'TITANOS_MANAGED_DELEGATION': 'parent'}
    offline = {'oa': 'openai'}

    @classmethod
    def document(cls):
        return ManagedRole.document()

    @classmethod
    def prepare(cls):
        cls.delegate_id, cls.delegate = cls.mint('parent', 'inference', 'cloud', 'delegate')
        _, cls.plain = cls.mint('plain', 'inference', 'cloud')

    def call(self, token, model=GPT + '@openai', **headers):
        return self.serve.client(token).post('/v1/chat/completions', json={'model': model, 'messages': CHAT},
                                             headers=headers)

    def test_delegated_explicit_target_reaches_the_mock(self):
        before = hits('openai', surface='openai.chat')
        ok = self.call(self.delegate, **{'X-Titanos-Delegate': 'cloud', 'X-Request-ID': 'parent-req-0001'})
        self.assertEqual(ok.status_code, 200, ok.text[:300])
        self.provenance(ok, 'cloud:openai', 'passthrough')
        self.assertEqual(ok.headers['x-request-id'], 'parent-req-0001')
        self.assertEqual(hits('openai', surface='openai.chat'), before + 1)
        hit = MOCKS['openai'].hits_for('openai.chat')[-1]
        self.assertEqual(hit.credential, 'Bearer <sentinel:openai>')
        for name in ('x-titanos-delegate', 'x-titanos-hop'):
            self.assertNotIn(name, hit.headers)
        self.assertFalse(hit.presented(self.delegate))

    def test_missing_conditions_are_refused_without_a_hit(self):
        before = hits()
        self.refused(self.call(self.delegate), 403, 'delegation_required')
        self.refused(self.call(self.delegate, 'mini', **{'X-Titanos-Delegate': 'cloud'}), 403, 'delegation_required')
        no_scope = self.call(self.plain, **{'X-Titanos-Delegate': 'cloud'})
        self.assertEqual(no_scope.status_code, 403, no_scope.text[:300])
        self.refused(self.call(self.delegate, **{'X-Titanos-Delegate': 'cloud', 'X-Titanos-Hop': 'another-instance'}), 400,
                     'location_unsatisfiable')
        hop = self.admin.get('/api/broker').json()['instance_hop']
        self.refused(self.call(self.delegate, **{'X-Titanos-Delegate': 'cloud', 'X-Titanos-Hop': hop}), 508,
                     'loop_detected')
        self.assertEqual(hits(), before)

    def test_parent_deadline_gives_504(self):
        MOCKS['openai'].faults.push('slow_first_byte', delay=1.5, surface='openai.chat')
        reply = self.call(self.delegate, **{'X-Titanos-Delegate': 'cloud', 'X-Titanos-Deadline-Ms': '300'})
        self.refused(reply, 504, 'deadline_exceeded')
        hit = MOCKS['openai'].hits_for('openai.chat')[-1]
        wait_until(lambda: hit.completed or hit.disconnected or hit.aborted, 5.0)


def load_tests(loader, standard_tests, pattern):
    """Every concrete case in a fixed order; the served base class is not a test case of its own."""
    suite = unittest.TestSuite()
    for case in (GuardSelfTest, KeyedStandalone, KeylessStandalone, BrokerRole, ManagedRole, ManagedDelegation):
        suite.addTests(loader.loadTestsFromTestCase(case))
    return suite


if __name__ == '__main__':
    unittest.main(verbosity=2)
