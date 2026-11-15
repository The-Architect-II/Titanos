"""Shared harness for the WP7 broker checks (the integrated app: Guard, KeylessHostGuard, BrokerDispatch and routes).

- importing it installs the loopback netguard (through tests/broker_wp6_support) before anything can connect;
- FullApp: create_app over a temporary TITANOS_HOME (set for the whole test, because the broker reads it lazily), an
  optional broker.json, synthetic local models answered by a recording fake engine (no model, no device), and an
  in-memory vault holding validated WP4 sentinels bound to the mocks' exact origins;
- mint(): real tk_ keys through the app's KeyStore (key principals come from the Guard, as in production);
- client(): a TestClient whose peer is a loopback address by default (transport_secure), or a LAN address.
No real secret is used and nothing leaves loopback.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import broker_wp6_support as S  # noqa: F401  (installs the netguard on import)
from broker_mocks import SENTINELS, netguard  # noqa: F401
from fastapi.testclient import TestClient

from titanos.broker.config import BrokerConfig, write_broker_config
from titanos.broker.keys import KeySpec
from titanos.broker.vault.store import Vault, platform_default
from titanos.config import Config, ModelSpec
from titanos.settings import Settings

WAV = b'RIFF' + (36).to_bytes(4, 'little') + b'WAVEfmt ' + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') \
    + (1).to_bytes(2, 'little') + (16000).to_bytes(4, 'little') + (32000).to_bytes(4, 'little') \
    + (2).to_bytes(2, 'little') + (16).to_bytes(2, 'little') + b'data' + (0).to_bytes(4, 'little')


class FakeEngine:
    """Records every engine.infer call and answers by task."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, name, task, data, emit=None, cancel=None, **kw):
        self.calls.append((name, task, data))
        if task == 'embeddings':
            return {'vectors': [[0.25, 0.5] for _ in data['input']], 'usage': {'prompt_tokens': 2, 'total_tokens': 2}}
        if task == 'speech':
            return WAV
        if task == 'transcription':
            return 'local transcript'
        if task == 'translation':
            return {'text': 'local translation', 'translated': True}
        if task == 'vad':
            import hashlib
            raw = data['audio']
            return {'segments': [{'start': 0.0, 'end': 0.5}], 'duration': 1.0, 'sample_rate': 16000,
                    'decoded_samples': 16000, 'input_sha256': hashlib.sha256(raw).hexdigest(), 'frames': None,
                    'threshold': 0.5, 'provenance': {}}
        if emit:
            emit('local ')
            emit('hello')
        return {'text': 'local hello', 'finish_reason': 'stop',
                'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}


class FullApp:
    """create_app over a temporary TITANOS_HOME. models: {name: task}; document: a broker.json dict or None."""

    def __init__(self, document: dict | None = None, *, models: dict | None = None, api_key: str | None = None,
                 settings_kw: dict | None = None, creds: dict | None = None, unvalidated=(), write: bool = True):
        self.root = Path(tempfile.mkdtemp(prefix='titanos-wp7-'))
        self.home = self.root / 'home'
        self.home.mkdir()
        self._env = mock.patch.dict(os.environ, {'TITANOS_HOME': str(self.home)})
        self._env.start()
        values = {'broker_egress': 'loopback-only'}
        values.update(settings_kw or {})
        self.settings = Settings(**values)
        self.backend = S.MemoryBackend(platform_default())
        self.vault = Vault(self.settings, self.home, backends={platform_default(): self.backend})
        if document is not None:
            cfg = BrokerConfig.model_validate(document)
            if write:
                write_broker_config(self.home / 'broker.json', cfg)
            creds = dict(creds or {})
            for uid, upstream in cfg.upstreams.items():
                if not upstream.credential:
                    continue
                sentinel = creds.get(uid) or S.SENTINEL_FOR_PROFILE.get(upstream.profile, 'openai')
                origin = upstream.origin(allow_http_lan=cfg.policy.allow_http_lan)
                self.vault.put_secret(upstream.credential, SENTINELS[sentinel], upstream_id=uid, origin=origin)
                if uid not in unvalidated:
                    self.vault.mark_validated(upstream.credential, True, None, None)
        specs = {}
        for name, value in (models or {}).items():  # {name: task} or {name: (task, backend)}
            task, backend = (value, None) if isinstance(value, str) else value
            spec = {'path': str(self.root), 'task': task, **({'backend': backend} if backend else {})}
            if task == 'vad' and backend is None:
                spec.update(backend='onnx', device='cpu')
            specs[name] = ModelSpec(**spec)
        self.config = Config(models=specs)
        from titanos.server import create_app
        self.app = create_app(self.config, api_key=api_key, output_dir=self.root / 'outputs', settings=self.settings)
        self.broker = self.app.state.broker
        self.broker._components['vault'] = self.vault
        self.engine = FakeEngine()
        self.app.state.engine.target.infer = self.engine
        self._clients: list[TestClient] = []

    def mint(self, *scopes, name: str = 'k', **fields) -> tuple[str, str]:
        """(key id, token) of a fresh key holding scopes (KeySpec fields as keyword arguments)."""
        record, token = self.broker.keys.mint(KeySpec(name=name, scopes=list(scopes), **fields), created_by='test')
        return record.id, token

    @property
    def http(self) -> TestClient:
        """The app's main client (loopback peer); its context runs the lifespan once (broker.startup, dispose)."""
        if not self._clients:
            client = TestClient(self.app, base_url='http://127.0.0.1:8000', client=('127.0.0.1', 50000))
            client.__enter__()
            self._clients.append(client)
        return self._clients[0]

    def client(self, *, peer: str = '127.0.0.1', scheme: str = 'http') -> TestClient:
        """An extra client with another peer address or scheme (no lifespan of its own)."""
        self.http  # the lifespan runs on the main client
        return TestClient(self.app, base_url=f'{scheme}://127.0.0.1:8000', client=(peer, 50000))

    def close(self) -> None:
        for client in self._clients:
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass
        self._env.stop()
        shutil.rmtree(self.root, ignore_errors=True)


def bearer(token: str) -> dict:
    return {'Authorization': 'Bearer ' + token}


__all__ = ['FakeEngine', 'FullApp', 'S', 'SENTINELS', 'WAV', 'bearer', 'netguard']
