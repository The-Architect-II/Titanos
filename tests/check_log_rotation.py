"""TITANOS_LOG_FILE: the server log file with size and age caps (titanos.log_rotation).

Covers: RotatingFile appending, rotating at log_max_bytes into .1 ... .N, keeping every line while the backups hold them
and at most (backups + 1) x max bytes plus one line otherwise, and deleting rotated files older than log_max_age_days at
startup and at rotation; the uvicorn logging configuration log_config builds (uvicorn, access and root lines in the file,
warnings still on standard error) with the broker's redaction filter applied to the file handler; the settings bounds; and
`titanos serve` passing log_config to uvicorn only when TITANOS_LOG_FILE is set (uvicorn replaced, nothing binds). Files
under a temporary directory only; no model, process, GPU or network."""
import io
import logging
import logging.config
import os
import re
import sys
import tempfile
import time
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from titanos.broker.redact import REGISTRY, install_logging_redaction, uninstall_logging_redaction  # noqa: E402
from titanos.cli import main  # noqa: E402
from titanos.config import Config, write_config  # noqa: E402
from titanos.log_rotation import RotatingFile, log_config  # noqa: E402
from titanos.settings import Settings  # noqa: E402


def files(base):
    """The log and its rotated files, newest first (titanos.log, .1, .2, ...)."""
    def index(path):
        suffix = path.name[len(base.name):].lstrip('.')
        return int(suffix) if suffix else 0
    return sorted(base.parent.glob(base.name + '*'), key=index)


def write(handler, count, name):
    logger = logging.Logger(name)
    logger.addHandler(handler)
    for number in range(count):
        logger.warning('line %05d %s', number, 'x' * 40)
    handler.close()


def check_rotation(folder):
    base = folder / 'keep' / 'titanos.log'
    write(RotatingFile(str(base), maxBytes=4096, backupCount=40), 1000, 'keep')
    numbers = [int(match) for path in reversed(files(base)) for match in re.findall(r'line (\d{5})', path.read_text('utf-8'))]
    assert numbers == list(range(1000)), 'every line kept, in order, across the rotated files'
    assert all(path.stat().st_size <= 4096 for path in files(base)), [path.stat().st_size for path in files(base)]
    capped = folder / 'cap' / 'titanos.log'
    write(RotatingFile(str(capped), maxBytes=4096, backupCount=2), 1000, 'cap')
    assert [path.name for path in files(capped)] == ['titanos.log', 'titanos.log.1', 'titanos.log.2'], files(capped)
    assert sum(path.stat().st_size for path in files(capped)) <= 3 * 4096
    last = int(re.findall(r'line (\d{5})', capped.read_text('utf-8'))[-1])
    assert last == 999, 'the newest line is in the current file'
    handler = RotatingFile(str(capped), maxBytes=4096, backupCount=2)  # append mode: existing lines stay
    handler.close()
    assert int(re.findall(r'line (\d{5})', capped.read_text('utf-8'))[-1]) == 999


def check_age(folder):
    base = folder / 'age' / 'titanos.log'
    base.parent.mkdir(parents=True)
    old = time.time() - 3 * 86400
    for index in (1, 2):
        rotated = base.with_name(f'titanos.log.{index}')
        rotated.write_text('old\n', encoding='utf-8')
        os.utime(rotated, (old, old))
    base.with_name('titanos.log.3').write_text('recent\n', encoding='utf-8')
    handler = RotatingFile(str(base), maxBytes=4096, backupCount=4, max_age_days=1)
    assert [path.name for path in files(base)] == ['titanos.log', 'titanos.log.3'], files(base)
    handler.close()
    handler = RotatingFile(str(base), maxBytes=4096, backupCount=4, max_age_days=0)  # 0 keeps everything
    handler.close()


def check_config(folder):
    path = folder / 'serve' / 'titanos.log'
    settings = Settings(log_file=str(path), log_max_bytes=8192, log_backups=3)
    secret = 'sk-' + 'rotation-check-' + os.urandom(8).hex()
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        logging.config.dictConfig(log_config(settings))
        REGISTRY.add(secret)
        install_logging_redaction()
        try:
            logging.getLogger('uvicorn.error').info('started %s', 'server')
            logging.getLogger('uvicorn.access').info('%s - "%s %s HTTP/%s" %d', '127.0.0.1:1', 'GET', '/healthz', '1.1', 200)
            logging.getLogger('titanos.check').warning('provider said %s', secret)
            logging.getLogger('titanos.check').info('below the stderr level')
        finally:
            uninstall_logging_redaction()
            REGISTRY.discard(secret)
            logging.shutdown()
            logging.config.dictConfig({'version': 1, 'disable_existing_loggers': False})
    text = path.read_text(encoding='utf-8')
    assert 'uvicorn.error: started server' in text and 'GET /healthz' in text and 'titanos.check: provider said' in text, text
    assert secret not in text and secret not in stderr.getvalue(), 'the redaction filter covers the file handler'
    assert 'titanos.check: provider said' in stderr.getvalue() and 'below the stderr level' not in stderr.getvalue()
    handler = log_config(settings)['handlers']['titanos_file']
    assert handler['maxBytes'] == 8192 and handler['backupCount'] == 3 and handler['filename'] == os.path.abspath(path)


def check_settings_and_serve(folder):
    defaults = Settings()
    assert (defaults.log_file, defaults.log_max_bytes, defaults.log_backups, defaults.log_max_age_days) == ('', 64 * 1024 * 1024, 4, 0.0)
    for bad in ({'log_file': 'relative.log'}, {'log_max_bytes': 100}, {'log_backups': -1}, {'log_max_age_days': -1.0}):
        try:
            Settings(**bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} accepted')
    registry = folder / 'models.json'
    write_config(registry, Config())
    clean = {key: value for key, value in os.environ.items() if not key.startswith('TITANOS_')}
    app_factory, server_run = Mock(return_value='fixture-app'), Mock()
    with patch.dict(os.environ, {**clean, 'TITANOS_HOME': str(folder)}, clear=True), \
         patch.dict('sys.modules', {'titanos.server': SimpleNamespace(create_app=app_factory)}), \
         patch('uvicorn.run', server_run):
        main(['--config', str(registry), 'serve', '--port', '12001'])
        assert 'log_config' not in server_run.call_args.kwargs, server_run.call_args.kwargs
        os.environ['TITANOS_LOG_FILE'] = str(folder / 'logs' / 'serve.log')
        main(['--config', str(registry), 'serve', '--port', '12001'])
        config = server_run.call_args.kwargs['log_config']
        assert config['handlers']['titanos_file']['filename'] == os.path.abspath(folder / 'logs' / 'serve.log'), config
        assert server_run.call_args.kwargs['limit_concurrency'] == 64


def main_check():
    with tempfile.TemporaryDirectory(prefix='titanos-logs-') as name:
        folder = Path(name)
        check_rotation(folder)
        check_age(folder)
        check_config(folder)
        check_settings_and_serve(folder)
    print('{"check": "log_rotation", "result": "pass", "scope": "rotating files under a temporary directory and a replaced '
          'uvicorn.run; no model, process or network"}')


if __name__ == '__main__':
    main_check()
