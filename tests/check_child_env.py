"""Child environments: no provider credential reaches a child process of the engine.

Native runtimes and helper children (titanos.native_process.child_environment, used for llama-server, its supervisor and
the device probe) start from an allowlist (titanos.child_env.allowlisted_env): operating-system basics, accelerator and
thread controls, Python plumbing and TITANOS_* settings, plus TITANOS_CHILD_ENV_ALLOW entries; credential names are
dropped even when allowed (known provider variables, *_API_KEY, *_BASE_URL, *_TOKEN, *_SECRET, *_PASSWORD and the broker
vault's env: names). Spawned replica workers and FFmpeg keep the rest of the environment but lose the same credential
names (scrubbed_env). Checked on synthetic environments, a real Python child, the llama-server argv/env of
NativeWorker.start_process (Popen captured), the replica worker scrub and the FFmpeg environment. No model, GPU or
network; one short-lived Python child."""
import contextlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from titanos import media, native, native_process, replicas  # noqa: E402
from titanos.child_env import allowlisted_env, scrubbed_env  # noqa: E402
from titanos.config import ModelSpec, NativeOptions  # noqa: E402
from titanos.settings import Settings, _active_settings  # noqa: E402

# the provider names the HiveMind-engine program lists, and other credential shapes
CREDENTIALS = ('HF_TOKEN', 'HF_AUTH_TOKEN', 'NGC_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'ELEVENLABS_API_KEY',
               'XAI_API_KEY', 'TITANOS_API_KEY', 'GITHUB_TOKEN', 'AWS_SECRET_ACCESS_KEY', 'AWS_ACCESS_KEY_ID',
               'MY_PROVIDER_TOKEN', 'DB_PASSWORD', 'OPENAI_BASE_URL', 'Hf_Token')
BASICS = {'PATH': os.environ.get('PATH', '/usr/bin'), 'SystemRoot': os.environ.get('SystemRoot', 'C:\\Windows'),
          'TEMP': 'temp', 'CUDA_VISIBLE_DEVICES': 'GPU-11111111-1111-1111-1111-111111111111', 'GGML_CUDA_NO_PINNED': '1',
          'TITANOS_HOME': 'home', 'OMP_NUM_THREADS': '4', 'PYTHONPATH': 'src', 'LC_ALL': 'C.UTF-8'}
OTHERS = {'RANDOM_SETTING': 'kept-by-scrub-only', 'HTTP_PROXY': 'http://user:pass@proxy.invalid:8080',
          'IMAGEIO_FFMPEG_EXE': 'ffmpeg', 'EXTRA_ONE': '1'}


def environment():
    return {**BASICS, **OTHERS, **{name: 'secret-value' for name in CREDENTIALS}}


def check_allowlist():
    kept = allowlisted_env(environment())
    assert set(kept) == set(BASICS) | {'IMAGEIO_FFMPEG_EXE'}, sorted(kept)
    extended = allowlisted_env({**environment(), 'TITANOS_CHILD_ENV_ALLOW': 'random_setting, EXTRA_*, MY_PROVIDER_TOKEN'})
    assert {'RANDOM_SETTING', 'EXTRA_ONE', 'TITANOS_CHILD_ENV_ALLOW'} <= set(extended), sorted(extended)
    assert not set(CREDENTIALS) & set(extended), 'a credential name stays dropped even when allowed'
    assert set(allowlisted_env(environment(), extra_allow=['HTTP_PROXY'])) == set(BASICS) | {'IMAGEIO_FFMPEG_EXE', 'HTTP_PROXY'}
    scrubbed = scrubbed_env(environment())
    assert set(scrubbed) == set(BASICS) | set(OTHERS), sorted(scrubbed)
    return len(CREDENTIALS)


def check_vault_names(home):
    vault = Path(home) / 'broker' / 'vault'
    vault.mkdir(parents=True)
    (vault / 'index.json').write_text(json.dumps({'version': 1, 'credentials': {
        'c1': {'id': 'c1', 'source': 'env', 'env_var': 'MY_VAULT_VAR'}}}), encoding='utf-8')
    env = {**environment(), 'MY_VAULT_VAR': 'v', 'TITANOS_CHILD_ENV_ALLOW': 'MY_VAULT_VAR'}
    with patch.dict(os.environ, {'TITANOS_HOME': str(home)}):
        out = native_process.child_environment(env)
    assert 'MY_VAULT_VAR' not in out and not set(CREDENTIALS) & set(out), sorted(out)


def check_real_child():
    env = native_process.child_environment({**os.environ, **{name: 'secret-value' for name in CREDENTIALS}, **OTHERS})
    run = subprocess.run([sys.executable, '-I', '-S', '-c', 'import json,os;print(json.dumps(sorted(os.environ)))'],
                         env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    names = {name.upper() for name in json.loads(run.stdout)}
    assert not {name.upper() for name in CREDENTIALS} & names and 'RANDOM_SETTING' not in names, sorted(names)


def check_native_launch(folder):
    """The environment NativeWorker.start_process hands to the llama-server supervisor."""
    model = folder / 'model.gguf'
    model.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 0) + b'\0' * 16)
    executable = folder / 'llama-server.exe'
    executable.write_bytes(b'inert executable bytes')
    spec = ModelSpec(path=str(folder), task='chat', backend='llama-cpp', device='cpu', native=NativeOptions(model_file='model.gguf'))
    placement = {'resolved_device': 'cpu', 'resolved_gpu_layers': 0, 'native_devices': ['none'], 'verified_devices': []}
    captured = []

    class Running:
        stdin = NS(close=lambda: None)

        def poll(self):
            return None

        def wait(self, *args, **kwargs):
            return 0

    def start(argv, **kwargs):
        captured.append(kwargs['env'])
        Path(argv[argv.index('titanos.native_process') + 1]).write_text('4242')
        return Running()
    worker = native.NativeWorker('w', 'm', spec)
    with patch.dict(os.environ, {**{name: 'secret-value' for name in CREDENTIALS}, **OTHERS}), \
         patch.object(native, 'executable_path', return_value=executable), \
         patch.object(native, 'resolve_native_placement', return_value=placement), \
         patch.object(native.subprocess, 'Popen', side_effect=start), \
         patch.object(native.httpx, 'Client', return_value=contextlib.nullcontext(NS(get=lambda url: NS(status_code=200)))):
        worker.start_process()
    worker.temp.cleanup()
    names = {name.upper() for name in captured[0]}
    assert not {name.upper() for name in CREDENTIALS} & names and 'RANDOM_SETTING' not in names, sorted(names)
    assert 'PATH' in names and 'TITANOS_HOME' in names, sorted(names)


def check_replica_and_ffmpeg():
    with patch.dict(os.environ, {**{name: 'secret-value' for name in CREDENTIALS}, **OTHERS}):
        env = media._ffmpeg_env()
        assert not set(CREDENTIALS) & set(env) and env['IMAGEIO_FFMPEG_EXE'] == 'ffmpeg' and env['RANDOM_SETTING'], sorted(env)
        replicas._scrub_environment()
        left = {name.upper() for name in os.environ}
        assert not {name.upper() for name in CREDENTIALS} & left, sorted(left)
        assert os.environ['IMAGEIO_FFMPEG_EXE'] == 'ffmpeg' and os.environ['RANDOM_SETTING'] == 'kept-by-scrub-only'


def main():
    token = _active_settings.set(Settings())
    try:
        names = check_allowlist()
        with tempfile.TemporaryDirectory(prefix='titanos-childenv-') as name:
            check_vault_names(Path(name) / 'home')
            check_native_launch(Path(name))
        check_real_child()
        check_replica_and_ffmpeg()
    finally:
        _active_settings.reset(token)
    print(json.dumps({'check': 'child_env', 'result': 'pass', 'credential_names': names,
                      'scope': 'synthetic environments, one Python child, captured llama-server launch; no model or GPU'}))


if __name__ == '__main__':
    main()
