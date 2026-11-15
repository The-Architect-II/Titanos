"""Size- and age-bounded server log file for `titanos serve`.

Without TITANOS_LOG_FILE the server logs to standard error as before and nothing here runs. With it (an absolute path),
uvicorn's logging configuration also writes the uvicorn, access and application (root) loggers to that file, appending,
in plain text. When the file would pass TITANOS_LOG_MAX_BYTES it is renamed to <file>.1 (older ones shift to .2 ... up
to TITANOS_LOG_BACKUPS, the oldest dropped), so the directory holds at most (backups + 1) x max bytes plus one line.
TITANOS_LOG_MAX_AGE_DAYS, when nonzero, also deletes rotated files older than that at startup and at each rotation.
The broker's redaction filter is attached to these handlers at startup like any other (titanos.broker.redact).
On Windows a rename can fail while another program holds the file open without delete sharing; the line is kept,
logging reports the error once per attempt, and writing continues in the same file until a later rotation succeeds.
"""
import copy
import logging
import logging.handlers
import os
import time
from pathlib import Path

FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


class RotatingFile(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler (append mode, UTF-8) that also prunes rotated files older than max_age_days."""

    def __init__(self, filename, maxBytes, backupCount, max_age_days=0.0):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        super().__init__(filename, mode='a', maxBytes=maxBytes, backupCount=backupCount, encoding='utf-8')
        self.max_age_days = max_age_days
        self.prune()

    def prune(self):
        if not self.max_age_days:
            return
        cutoff = time.time() - self.max_age_days * 86400
        base = Path(self.baseFilename)
        for index in range(1, self.backupCount + 1):
            rotated = base.with_name(f'{base.name}.{index}')
            try:
                if rotated.stat().st_mtime < cutoff:
                    rotated.unlink()
            except OSError:
                pass

    def doRollover(self):
        super().doRollover()
        self.prune()


def log_config(settings, base=None):
    """uvicorn's logging configuration (a dictConfig mapping) with the TITANOS_LOG_FILE handler added to the uvicorn,
    uvicorn.access and root loggers; standard error keeps uvicorn's own handlers, and the root logger keeps warnings
    on standard error as Python's last-resort handler printed them."""
    if base is None:
        from uvicorn.config import LOGGING_CONFIG as base
    config = copy.deepcopy(base)
    config.setdefault('formatters', {})['titanos_file'] = {'format': FORMAT}
    config.setdefault('handlers', {})['titanos_file'] = {
        '()': RotatingFile, 'filename': os.path.abspath(settings.log_file), 'maxBytes': settings.log_max_bytes,
        'backupCount': settings.log_backups, 'max_age_days': settings.log_max_age_days, 'formatter': 'titanos_file'}
    config['handlers']['titanos_stderr'] = {'class': 'logging.StreamHandler', 'stream': 'ext://sys.stderr',
                                            'level': 'WARNING', 'formatter': 'titanos_file'}
    loggers = config.setdefault('loggers', {})
    for name in ('uvicorn', 'uvicorn.access'):
        entry = loggers.setdefault(name, {})
        entry['handlers'] = [*entry.get('handlers', []), 'titanos_file']
    config['root'] = {'level': settings.log_level.upper() if settings.log_level != 'trace' else 'DEBUG',
                      'handlers': ['titanos_stderr', 'titanos_file']}
    return config


def uvicorn_options(settings):
    """Extra uvicorn.run keyword arguments for settings: {} without TITANOS_LOG_FILE, else {'log_config': ...}."""
    return {'log_config': log_config(settings)} if settings.log_file else {}


__all__ = ['FORMAT', 'RotatingFile', 'log_config', 'uvicorn_options']
