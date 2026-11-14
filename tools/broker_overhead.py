"""Measure the latency BrokerDispatch adds to local chat, in process, against a fake engine (manual tier; never a CI gate).

Four apps are built by titanos.server.create_app, each over its own temporary TITANOS_HOME, with one local chat model whose
engine.infer is replaced by a fixed in-memory answer (no model, no device, no network):
  no-dispatch          keyless, no broker.json, BrokerDispatch removed from the middleware stack (the reference)
  inactive             keyless, no broker.json: BrokerDispatch's fast path (the request passes through untouched)
  keyed-no-dispatch    bootstrap key, a broker.json with one enabled cloud upstream, a minted inference key, and
                       BrokerDispatch removed (the reference for the active path: same Guard and key verification)
  active               the same app with BrokerDispatch: the router resolves the local target, records a Decision and
                       adds the provenance headers
The cloud upstream points at a loopback discard address and is never contacted (the requests name the local model);
egress is loopback-only, and when tests/broker_mocks/netguard.py is present (a repository checkout) it is installed so any
non-loopback connect would fail.

Requests are sent through Starlette's TestClient (in-process ASGI), non-streaming and streaming chat, in alternating
rounds so drift affects every variant alike. The TestClient's own httpx request log is silenced (harness cost); the
broker's per-request INFO line stays on unless --no-request-log, so the active figure includes the logging a served
request pays. The report gives per-variant latency percentiles and the median and p90 differences
inactive - no-dispatch and active - keyed-no-dispatch. The numbers describe this machine and this run only; the tool
asserts nothing about them. It fails (exit 1) only when a variant did not take the path it is named for.

    python tools/broker_overhead.py [--requests N] [--rounds R] [--warmup W] [--no-stream] [--no-request-log]
                                    [--json PATH]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

MODEL = 'chat-local'
CHAT = [{'role': 'user', 'content': 'Say hello.'}]
BOOTSTRAP = 'overhead-bootstrap-' + os.urandom(8).hex()  # synthetic, this process only
VARIANTS = ('no-dispatch', 'inactive', 'keyed-no-dispatch', 'active')
REFERENCE = {'inactive': 'no-dispatch', 'active': 'keyed-no-dispatch'}


def install_netguard() -> bool:
    """Install the repository's test netguard when it is present (a checkout); an installed package has none."""
    path = ROOT / 'tests' / 'broker_mocks' / 'netguard.py'
    if not path.is_file():
        return False
    import importlib.util
    spec = importlib.util.spec_from_file_location('titanos_overhead_netguard', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['titanos_overhead_netguard'] = module
    spec.loader.exec_module(module)
    module.install()
    return True


def fake_infer(name, task, data, emit=None, cancel=None, **kw):
    """The fixed answer of the fake engine (streaming emits two pieces)."""
    if emit:
        emit('local ')
        emit('hello')
    return {'text': 'local hello', 'finish_reason': 'stop',
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}


class Variant:
    def __init__(self, name: str, root: Path):
        self.name = name
        self.home = root / name
        self.home.mkdir(parents=True)
        self.token: str | None = None
        self.client = None
        self.samples: dict[str, list[float]] = {'chat': [], 'stream': []}
        self.intercepted: bool | None = None

    @contextlib.contextmanager
    def env(self):
        previous = os.environ.get('TITANOS_HOME')
        os.environ['TITANOS_HOME'] = str(self.home)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop('TITANOS_HOME', None)
            else:
                os.environ['TITANOS_HOME'] = previous

    def build(self, stack: contextlib.ExitStack) -> None:
        from fastapi.testclient import TestClient

        from titanos.config import Config, ModelSpec
        from titanos.server import create_app
        from titanos.settings import Settings
        keyed = self.name in ('keyed-no-dispatch', 'active')
        with self.env():
            if keyed:
                from titanos.broker.config import BrokerConfig, write_broker_config
                document = {'policy': {'cloud_policy': 'explicit'},
                            'upstreams': {'cloud': {'profile': 'openai', 'base_url': 'http://127.0.0.1:9/v1',
                                                    'enabled': True, 'models': ['remote-model']}}}
                write_broker_config(self.home / 'broker.json', BrokerConfig.model_validate(document))
            settings = Settings(broker_egress='loopback-only', broker_vault_backend='refs-only')
            weights = self.home / 'weights'
            weights.mkdir()
            config = Config(models={MODEL: ModelSpec(path=str(weights), task='chat')})
            app = create_app(config, api_key=BOOTSTRAP if keyed else None, output_dir=self.home / 'outputs',
                             settings=settings)
            app.state.engine.target.infer = fake_infer
            if self.name.endswith('no-dispatch'):
                from titanos.broker.dispatch import BrokerDispatch
                kept = [m for m in app.user_middleware if m.cls is not BrokerDispatch]
                if len(kept) != len(app.user_middleware) - 1:
                    raise RuntimeError('BrokerDispatch was not found exactly once in the middleware stack')
                app.user_middleware = kept
                app.middleware_stack = None
            self.client = stack.enter_context(TestClient(app, base_url='http://127.0.0.1:8000',
                                                         client=('127.0.0.1', 50000)))
            if keyed:
                from titanos.broker.keys import KeySpec
                _, self.token = app.state.broker.keys.mint(KeySpec(name='overhead', scopes=['inference']),
                                                           created_by='broker_overhead')

    def request(self, stream: bool) -> tuple[float, object]:
        headers = {'Authorization': 'Bearer ' + self.token} if self.token else {}
        body = {'model': MODEL, 'messages': CHAT, 'stream': stream}
        with self.env():
            start = time.perf_counter()
            reply = self.client.post('/v1/chat/completions', json=body, headers=headers)
            _ = reply.content
            elapsed = time.perf_counter() - start
        if reply.status_code != 200:
            raise RuntimeError(f'{self.name}: HTTP {reply.status_code} {reply.text[:200]}')
        return elapsed, reply


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


def summary(values: list[float]) -> dict:
    ms = [v * 1000 for v in values]
    return {'requests': len(ms), 'median_ms': round(statistics.median(ms), 4), 'p90_ms': round(percentile(ms, 0.9), 4),
            'p99_ms': round(percentile(ms, 0.99), 4), 'mean_ms': round(statistics.fmean(ms), 4),
            'min_ms': round(min(ms), 4)}


def check_path(variant: Variant, reply) -> None:
    """The variant took the path it is named for (the only condition the tool fails on)."""
    decided = 'x-titanos-decision-id' in reply.headers
    variant.intercepted = decided
    if variant.name == 'active':
        if not decided or reply.headers.get('x-titanos-served-by') != 'local:engine':
            raise RuntimeError('the active variant was not routed by BrokerDispatch to the local engine')
    elif decided:
        raise RuntimeError(f'the {variant.name} variant was intercepted by the broker')


def run(requests: int, rounds: int, warmup: int, stream: bool, request_log: bool = True) -> dict:
    guarded = install_netguard()
    import logging
    logging.getLogger('httpx').setLevel(logging.WARNING)  # the TestClient's own per-request line (harness cost)
    if not request_log:
        logging.getLogger('titanos.broker').setLevel(logging.WARNING)  # the broker's per-request INFO line
    root = Path(tempfile.mkdtemp(prefix='titanos-overhead-'))
    kinds = ('chat', 'stream') if stream else ('chat',)
    try:
        with contextlib.ExitStack() as stack:
            variants = [Variant(name, root) for name in VARIANTS]
            for variant in variants:
                variant.build(stack)
            for variant in variants:
                for kind in kinds:
                    for _ in range(warmup):
                        variant.request(kind == 'stream')
                _, reply = variant.request(False)
                check_path(variant, reply)
            per_round = max(1, requests // rounds)
            for _ in range(rounds):
                for variant in variants:
                    for kind in kinds:
                        for _ in range(per_round):
                            elapsed, _ = variant.request(kind == 'stream')
                            variant.samples[kind].append(elapsed)
            report = {'variants': {}, 'overhead': {}}
            for variant in variants:
                report['variants'][variant.name] = {kind: summary(variant.samples[kind]) for kind in kinds}
                report['variants'][variant.name]['intercepted'] = variant.intercepted
            for name, reference in REFERENCE.items():
                report['overhead'][f'{name} - {reference}'] = {
                    kind: {'median_ms': round(report['variants'][name][kind]['median_ms']
                                              - report['variants'][reference][kind]['median_ms'], 4),
                           'p90_ms': round(report['variants'][name][kind]['p90_ms']
                                           - report['variants'][reference][kind]['p90_ms'], 4)}
                    for kind in kinds}
    finally:
        shutil.rmtree(root, ignore_errors=True)
    report['run'] = {'requests_per_variant_and_kind': per_round * rounds, 'rounds': rounds, 'warmup': warmup,
                     'python': platform.python_version(), 'platform': platform.platform(), 'cpus': os.cpu_count(),
                     'netguard': guarded, 'broker_request_log': request_log}
    report['scope'] = ('In-process ASGI requests (TestClient) against a fake engine on this machine; observations of '
                       'this run, not thresholds or a performance promise')
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--requests', type=int, default=200, help='Measured requests per variant and kind (default 200)')
    parser.add_argument('--rounds', type=int, default=5, help='Alternating rounds the requests are split into (default 5)')
    parser.add_argument('--warmup', type=int, default=20, help='Unmeasured requests per variant and kind (default 20)')
    parser.add_argument('--no-stream', action='store_true', help='Measure non-streaming chat only')
    parser.add_argument('--json', type=Path, help='Also write the report to this file')
    parser.add_argument('--no-request-log', action='store_true',
                        help="Raise the titanos.broker logger to WARNING, separating the active path's per-request "
                             'INFO log line from its routing cost')
    args = parser.parse_args(argv)
    if not 1 <= args.requests <= 100000 or not 1 <= args.rounds <= args.requests or not 0 <= args.warmup <= 10000:
        parser.error('--requests 1..100000, --rounds 1..requests, --warmup 0..10000')
    os.environ.setdefault('TITANOS_BROKER_EGRESS', 'loopback-only')
    try:
        report = run(args.requests, args.rounds, args.warmup, not args.no_stream, not args.no_request_log)
    except RuntimeError as exc:
        print(f'broker_overhead: {exc}', file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
