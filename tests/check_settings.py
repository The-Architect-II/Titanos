"""Check settings boundaries and CLI precedence without starting inference."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from titanos.cli import main
from titanos.config import Config, write_config
from titanos.settings import Settings


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Invalid settings accepted")


def check():
    baseline = Settings.from_env({})
    assert baseline.port == 11435 and baseline.max_tokens == 8192
    assert baseline.max_tensor_elements == 4_000_000
    assert Settings.from_env({'TITANOS_MAX_TENSOR_ELEMENTS':'8000000'}).max_tensor_elements == 8_000_000
    # Front-door body-read budgets are ordinary settings: defaults, environment parsing, explicit override, positivity.
    assert (baseline.body_read_seconds, baseline.body_read_min_rate, baseline.body_read_concurrency,
            baseline.body_read_wait_seconds) == (30.0, 262144, 8, 2.0)
    tuned = Settings.from_env({'TITANOS_BODY_READ_SECONDS': '0.25', 'TITANOS_BODY_READ_MIN_RATE': '1024',
                              'TITANOS_BODY_READ_CONCURRENCY': '2', 'TITANOS_BODY_READ_WAIT_SECONDS': '0.5'})
    assert (tuned.body_read_seconds, tuned.body_read_min_rate, tuned.body_read_concurrency, tuned.body_read_wait_seconds) == (0.25, 1024, 2, 0.5)
    assert Settings.from_env({'TITANOS_BODY_READ_CONCURRENCY': '2'}, body_read_concurrency=3).body_read_concurrency == 3
    for key, bad in (('TITANOS_BODY_READ_SECONDS', '0'), ('TITANOS_BODY_READ_SECONDS', '-1'), ('TITANOS_BODY_READ_SECONDS', 'nan'),
                     ('TITANOS_BODY_READ_SECONDS', 'inf'), ('TITANOS_BODY_READ_SECONDS', 'slow'),
                     ('TITANOS_BODY_READ_MIN_RATE', '0'), ('TITANOS_BODY_READ_MIN_RATE', '-1'), ('TITANOS_BODY_READ_MIN_RATE', '1.5'),
                     ('TITANOS_BODY_READ_CONCURRENCY', '0'), ('TITANOS_BODY_READ_CONCURRENCY', '-8'), ('TITANOS_BODY_READ_CONCURRENCY', '1.5'),
                     ('TITANOS_BODY_READ_CONCURRENCY', 'many'), ('TITANOS_BODY_READ_WAIT_SECONDS', '0'), ('TITANOS_BODY_READ_WAIT_SECONDS', 'soon')):
        rejected(lambda: Settings.from_env({key: bad}))
    rejected(lambda: Settings(body_read_concurrency=True))
    described = baseline.describe()['settings']
    assert described['body_read_seconds'] == {'value': 30.0, 'default': 30.0, 'env': 'TITANOS_BODY_READ_SECONDS', 'type': 'float'}
    assert described['body_read_min_rate'] == {'value': 262144, 'default': 262144, 'env': 'TITANOS_BODY_READ_MIN_RATE', 'type': 'int'}
    assert described['body_read_concurrency'] == {'value': 8, 'default': 8, 'env': 'TITANOS_BODY_READ_CONCURRENCY', 'type': 'int'}
    assert described['body_read_wait_seconds'] == {'value': 2.0, 'default': 2.0, 'env': 'TITANOS_BODY_READ_WAIT_SECONDS', 'type': 'float'}
    env = {"TITANOS_PORT": "12000", "TITANOS_MAX_TOKENS": "65536", "TITANOS_DEFAULT_MAX_TOKENS": "2048",
           "TITANOS_CLIENT_TIMEOUT": "1200.5", "TITANOS_API_KEY": "fixture-not-a-real-key"}
    value = Settings.from_env(env, port=12001, host=None)
    assert value.port == 12001 and value.max_tokens == 65536 and value.client_timeout == 1200.5
    description = json.dumps(value.describe())
    assert "fixture-not-a-real-key" not in description and "TITANOS_API_KEY" not in description
    assert value.describe()["settings"]["port"] == {"value": 12001, "default": 11435, "env": "TITANOS_PORT", "type": "int"}
    for key, bad in (("TITANOS_PORT", "65536"), ("TITANOS_THREADS", "0"), ("TITANOS_CLIENT_TIMEOUT", "nan"),
                     ("TITANOS_MAX_AUDIO_SECONDS", "inf"), ("TITANOS_QUEUE_CAPACITY", "-1"),
                     ("TITANOS_MAX_STEPS", "1.5"), ("TITANOS_DEFAULT_TOP_P", "1.5"),
                     ("TITANOS_MAX_TENSOR_ELEMENTS", "0"), ("TITANOS_MAX_TENSOR_ELEMENTS", "nan"),
                     ("TITANOS_DEFAULT_IMAGE_SIZE", "511x512"), ("TITANOS_LOG_LEVEL", "unknown"),
                     ("TITANOS_MAX_TOKENS", "1"), ("TITANOS_HOST", "")):
        rejected(lambda: Settings.from_env({key: bad}))
    rejected(lambda: Settings(threads=True))
    rejected(lambda: Settings.from_env({}, misspelled_setting=1))
    assert Settings.from_env({"TITANOS_DEFAULT_TEMPERATURE": "0", "TITANOS_KEEP_ALIVE_TIMEOUT": "0"}).default_temperature == 0
    assert Settings.from_env({"TITANOS_MAX_STEPS": "1000", "TITANOS_MAX_VIDEO_FRAMES": "1025"}).max_video_frames == 1025
    with tempfile.TemporaryDirectory() as tmp:
        registry = Path(tmp) / "models.json"
        write_config(registry, Config())
        app_factory, server_run = Mock(return_value="fixture-app"), Mock()
        with patch.dict(os.environ, {"TITANOS_HOST": "127.0.0.1", "TITANOS_PORT": "12000", "TITANOS_THREADS": "2",
                                     "TITANOS_LIMIT_CONCURRENCY": "16", "TITANOS_KEEP_ALIVE_TIMEOUT": "12",
                                     "TITANOS_LOG_LEVEL": "warning"}), \
             patch.dict("sys.modules", {"titanos.server": SimpleNamespace(create_app=app_factory)}), \
             patch("uvicorn.run", server_run):
            main(["--config", str(registry), "serve", "--port", "12001", "--threads", "3"])
            args = server_run.call_args.kwargs
            assert args["port"] == 12001 and args["host"] == "127.0.0.1"
            assert args["limit_concurrency"] == 16 and args["timeout_keep_alive"] == 12 and args["log_level"] == "warning"
            assert os.environ["TITANOS_THREADS"] == "3"
        with patch.dict(os.environ, {"TITANOS_MAX_TOKENS": "16384"}), contextlib.redirect_stdout(io.StringIO()) as output:
            main(["settings", "--json"])
            assert json.loads(output.getvalue())["settings"]["max_tokens"]["value"] == 16384
        with patch.dict(os.environ, {"TITANOS_PORT": "wrong"}), contextlib.redirect_stderr(io.StringIO()):
            try:
                main(["settings", "--json"])
            except SystemExit as exc:
                assert exc.code == 1
            else:
                raise AssertionError("Invalid environment did not produce a clean CLI failure")
    print("PASS: environment/default/explicit precedence, raised ceilings, invalid/inconsistent settings, body-read budget env parsing, secret-free discovery, CLI serve flags (no inference)")


if __name__ == "__main__":
    check()
