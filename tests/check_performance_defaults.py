"""The serving defaults the 2026-10-06 performance pass changed, bound to the documents that state them.

Settings: limit_concurrency 64 (uvicorn refused /healthz with a plain 503 for 21% of requests at 40 open connections under
32, and 64 refused none at the same latency), native_slots 4 over native_context_size 8192 (2,048 tokens per slot as before;
four llama.cpp slots answered 4 streams with an 81 ms first token where two slots at the same total context queued at 346 ms,
for 48 MiB more GPU memory on a 0.5B model). The defaults reach a new NativeOptions, the broker role's precondition check and
uvicorn.run; a saved registration keeps the values it was written with; docs/CONFIGURATION.md, docs/OPTION_PARITY.md,
docs/BROKER.md, docs/OPERATIONS.md, docs/PERFORMANCE.md, docs/TOPOLOGY_ROLES.md and the deviation map carry the same numbers. The serve defaults the
pass measured and kept (threads min(8, cpu), keep_alive_timeout 5, native_threads 4, native_gpu_layers auto, batch sizes
unset, native queue 32 x 30 s) are asserted so that a change is deliberate. No model, process, listener or network."""
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from titanos.broker.role import check_serve_preconditions  # noqa: E402
from titanos.broker.types import KeyStoreSummary  # noqa: E402
from titanos.cli import main  # noqa: E402
from titanos.config import Config, ModelSpec, NativeOptions, read_config, write_config  # noqa: E402
from titanos.settings import Settings  # noqa: E402

CHANGED = {'limit_concurrency': 64, 'native_context_size': 8192, 'native_slots': 4}
KEPT = {'keep_alive_timeout': 5, 'native_threads': 4, 'native_gpu_layers': 'auto', 'native_batch_size': 0,
        'native_ubatch_size': 0, 'native_queue_capacity': 32, 'native_queue_timeout': 30.0, 'worker_threads': 4}
NATIVE_ENV = ('TITANOS_NATIVE_SLOTS', 'TITANOS_NATIVE_CONTEXT_SIZE', 'TITANOS_NATIVE_THREADS', 'TITANOS_NATIVE_BATCH_SIZE',
              'TITANOS_NATIVE_UBATCH_SIZE', 'TITANOS_NATIVE_GPU_LAYERS')


def doc(name):
    return (ROOT / 'docs' / name).read_text(encoding='utf-8')


def check_settings():
    """The changed and the kept defaults, their environment override and their /api/info description."""
    settings = Settings.from_env({})
    for name, value in {**CHANGED, **KEPT}.items():
        assert getattr(settings, name) == value, (name, getattr(settings, name))
    assert settings.threads == min(8, os.cpu_count() or 1)
    assert settings.native_context_size // settings.native_slots == 2048  # the per-slot context num_ctx is admitted against
    described = settings.describe()['settings']
    for name, value in CHANGED.items():
        assert described[name]['default'] == value and described[name]['env'] == 'TITANOS_' + name.upper(), name
    previous = Settings.from_env({'TITANOS_LIMIT_CONCURRENCY': '32', 'TITANOS_NATIVE_SLOTS': '2',
                                  'TITANOS_NATIVE_CONTEXT_SIZE': '4096'})
    assert (previous.limit_concurrency, previous.native_slots, previous.native_context_size) == (32, 2, 4096)
    assert Settings.from_env({}, limit_concurrency=48).limit_concurrency == 48
    for name, bad in (('TITANOS_LIMIT_CONCURRENCY', '0'), ('TITANOS_NATIVE_SLOTS', '0'), ('TITANOS_NATIVE_CONTEXT_SIZE', '256')):
        try:
            Settings.from_env({name: bad})
        except ValueError:
            continue
        raise AssertionError(f'{name}={bad} was accepted')


def check_native_options():
    """A new NativeOptions takes the process defaults; the environment and explicit values still win; a saved registration
    written under the previous defaults keeps its values when read under the new ones."""
    clean = {key: value for key, value in os.environ.items() if key not in NATIVE_ENV}
    with patch.dict(os.environ, clean, clear=True):
        fresh = NativeOptions(model_file='model.gguf')
        assert (fresh.context_size, fresh.slots, fresh.threads, fresh.gpu_layers) == (8192, 4, 4, 'auto'), fresh
        assert fresh.batch_size is None and fresh.ubatch_size is None and fresh.split_mode == 'none'
        explicit = NativeOptions(model_file='model.gguf', slots=2, context_size=4096)
        assert (explicit.context_size, explicit.slots) == (4096, 2)
    with patch.dict(os.environ, {**clean, 'TITANOS_NATIVE_SLOTS': '2', 'TITANOS_NATIVE_CONTEXT_SIZE': '4096'}, clear=True):
        legacy = NativeOptions(model_file='model.gguf')
        assert (legacy.context_size, legacy.slots) == (4096, 2), legacy
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / 'models.json'
            write_config(registry, Config(models={'gguf': ModelSpec(path='.', task='chat', backend='llama-cpp',
                                                                     native=NativeOptions(model_file='model.gguf'))}))
            saved = json.loads(registry.read_text(encoding='utf-8'))['models']['gguf']['native']
            assert (saved['context_size'], saved['slots']) == (4096, 2), saved
    with patch.dict(os.environ, clean, clear=True):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / 'models.json'
            registry.write_text(json.dumps({'models': {'gguf': {'path': '.', 'task': 'chat', 'backend': 'llama-cpp',
                                                                'native': {'model_file': 'model.gguf', 'context_size': 4096,
                                                                           'slots': 2}}}}), encoding='utf-8')
            kept = read_config(registry).models['gguf'].native
            assert (kept.context_size, kept.slots) == (4096, 2), kept
            # A registration saved without the two fields takes the new defaults.
            registry.write_text(json.dumps({'models': {'gguf': {'path': '.', 'task': 'chat', 'backend': 'llama-cpp',
                                                                'native': {'model_file': 'model.gguf'}}}}), encoding='utf-8')
            implicit = read_config(registry).models['gguf'].native
            assert (implicit.context_size, implicit.slots) == (8192, 4), implicit


def check_broker_role():
    """The broker role's concurrency warning no longer fires at the default; it still names the switch below 64."""
    common = {'keystore': KeyStoreSummary(), 'explicit_port': True, 'vault_backend': 'dpapi', 'audit_ok': True}
    assert check_serve_preconditions(Settings(role='broker'), bootstrap_key='fixture-bootstrap', **common) == []
    warnings = check_serve_preconditions(Settings(role='broker', limit_concurrency=32), bootstrap_key='fixture-bootstrap', **common)
    assert warnings == ['TITANOS_LIMIT_CONCURRENCY below 64 limits concurrent broker streams'], warnings


def check_serve_arguments():
    """titanos serve hands the default limit and keep-alive to uvicorn.run (uvicorn is replaced; nothing binds)."""
    with tempfile.TemporaryDirectory() as tmp:
        registry = Path(tmp) / 'models.json'
        write_config(registry, Config())
        clean = {key: value for key, value in os.environ.items() if not key.startswith('TITANOS_')}
        app_factory, server_run = Mock(return_value='fixture-app'), Mock()
        with patch.dict(os.environ, {**clean, 'TITANOS_HOME': tmp}, clear=True), \
             patch.dict('sys.modules', {'titanos.server': SimpleNamespace(create_app=app_factory)}), \
             patch('uvicorn.run', server_run):
            main(['--config', str(registry), 'serve', '--port', '12001'])
            kwargs = server_run.call_args.kwargs
            assert kwargs['limit_concurrency'] == 64 and kwargs['timeout_keep_alive'] == 5, kwargs
            server_run.reset_mock()
            main(['--config', str(registry), 'serve', '--port', '12001', '--limit-concurrency', '32'])
            assert server_run.call_args.kwargs['limit_concurrency'] == 32


def check_documents():
    """The documents state the code's defaults, and docs/PERFORMANCE.md labels its figures as observed."""
    configuration = doc('CONFIGURATION.md')
    for name, value in (('limit_concurrency', '64'), ('native_context_size', '8192'), ('native_slots', '4'),
                        ('native_threads', '4'), ('keep_alive_timeout', '5 seconds'), ('native_queue_capacity', '32')):
        assert re.search(rf'^\| `{name}` \| {re.escape(value)} \|', configuration, re.M), (name, value)
    parity = doc('OPTION_PARITY.md')
    assert '`TITANOS_NATIVE_CONTEXT_SIZE=8192` → `--ctx-size`' in parity and '`TITANOS_NATIVE_SLOTS=4` → `--parallel`' in parity
    assert '`TITANOS_LIMIT_CONCURRENCY` (64)' in doc('BROKER.md')
    operations = doc('OPERATIONS.md')
    assert '`TITANOS_LIMIT_CONCURRENCY` (64)' in operations and '](PERFORMANCE.md)' in operations
    native = doc('NATIVE.md')
    assert '`TITANOS_NATIVE_CONTEXT_SIZE` (8192) and `TITANOS_NATIVE_SLOTS` (4)' in native and '](PERFORMANCE.md)' in native
    performance = doc('PERFORMANCE.md')
    for row in ('| `TITANOS_LIMIT_CONCURRENCY` | 32 | 64 |', '| `TITANOS_NATIVE_SLOTS` | 2 | 4 |',
                '| `TITANOS_NATIVE_CONTEXT_SIZE` | 4096 | 8192 |'):
        assert row in performance, row
    assert 'observed' in performance and 'shared' in performance and 'warm-up' in performance
    assert '](PERFORMANCE.md)' in doc('README.md') and '(docs/PERFORMANCE.md)' in (ROOT / 'README.md').read_text(encoding='utf-8')
    topology = doc('TOPOLOGY_ROLES.md')
    assert "Titanos's default registration is 8192" in topology and 'over 4 slots, 2,048 per slot' in topology
    assert 'the standard registration is 4096' not in topology
    deviations = (ROOT / 'docs' / 'protocol_deviations.json').read_text(encoding='utf-8')
    assert 'TITANOS_NATIVE_CONTEXT_SIZE=8192' in deviations and 'TITANOS_NATIVE_SLOTS=4' in deviations
    assert 'TITANOS_NATIVE_CONTEXT_SIZE=4096' not in deviations and 'TITANOS_NATIVE_SLOTS=2' not in deviations


def main_check():
    check_settings()
    check_native_options()
    check_broker_role()
    check_serve_arguments()
    check_documents()
    print('PASS: 2026-10-06 serving defaults (limit_concurrency 64; native_slots 4 over native_context_size 8192, 2,048 per slot) '
          'in Settings, NativeOptions, the broker precondition and uvicorn.run; kept defaults asserted; saved registrations keep '
          'their values; CONFIGURATION, OPTION_PARITY, BROKER, OPERATIONS, NATIVE, PERFORMANCE, TOPOLOGY_ROLES and the deviation map agree; '
          'no model, process or listener')


if __name__ == '__main__':
    main_check()
