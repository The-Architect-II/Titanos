"""A bounded native command tree; no shell, global environment or model runtime."""
import os
import re
import signal
import subprocess
import sys
import time
from .engine import EngineError

STDERR_TAIL_BYTES = 4096
# A credential label followed by = or :, optionally quoted (JSON/dict style), with an optional auth scheme word, then either a
# quoted value (spaces included) or one whitespace-free token. The label survives, the whole value goes.
_CREDENTIAL = re.compile(r'(?i)(api[_-]?key|token|secret|password|authorization)["\']?\s*[=:]\s*'
                         r'(?:"[^"\n]*"|\'[^\'\n]*\'|(?:(?:bearer|basic|token)\s+)?\S+)')
_CLI_CREDENTIAL = re.compile(r'(?i)(--?(?:api[-_]?key|token|secret|password))\s+\S+')  # --api-key VALUE / -token VALUE
_SCHEME_CREDENTIAL = re.compile(r'(?i)\b(bearer|basic)\s+\S+')  # a bare 'Bearer <token>' without a label
_JWT = re.compile(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*')  # three-segment token; each segment may be short
_WINDOWS_PATH = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s"\'<>|]+')
_POSIX_PATH = re.compile(r'(?<![A-Za-z0-9])/(?:home|Users|tmp|var|mnt|opt|root)/[^\s"\'<>|]+')
_LONG_RUN = re.compile(r'[A-Za-z0-9+/=_-]{32,}')
_SHA256 = re.compile(r'(?i)[a-f0-9]{64}')
_IDENTIFIER = re.compile(r'[A-Za-z_]+')  # a long run with no digit or base64 punctuation is identifier-shaped, not a secret


def _basename(match):
    return '<path>/' + re.split(r'[\\/]', match.group(0).rstrip('\\/'))[-1]


def redact(text):
    """Bounded, secret-safe rendering of child text: printable characters plus newline/tab only; credential-shaped assignments
    (labelled, quoted, CLI or 'Bearer' scheme forms), JWT-shaped tokens and long base64/hex-like runs (anything but a 64-hex
    sha256 or an identifier-shaped run) replaced; absolute Windows/POSIX paths reduced to '<path>/<final component>'; capped at
    STDERR_TAIL_BYTES chars. Everything else the child printed survives verbatim."""
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    text = _CREDENTIAL.sub(lambda m: m.group(1) + '=<redacted>', text)
    text = _CLI_CREDENTIAL.sub(lambda m: m.group(1) + ' <redacted>', text)
    text = _SCHEME_CREDENTIAL.sub(lambda m: m.group(1) + ' <redacted>', text)
    text = _JWT.sub('<redacted>', text)
    text = _WINDOWS_PATH.sub(_basename, text)
    text = _POSIX_PATH.sub(_basename, text)
    text = _LONG_RUN.sub(lambda m: m.group(0) if _SHA256.fullmatch(m.group(0)) or _IDENTIFIER.fullmatch(m.group(0)) else '<redacted>', text)
    return text[-STDERR_TAIL_BYTES:]


def stderr_tail(path):
    """The last STDERR_TAIL_BYTES bytes of a child's stderr log, decoded with replacement and redacted; '' when unreadable."""
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - STDERR_TAIL_BYTES))
            return redact(stream.read().decode('utf-8', errors='replace'))
    except OSError:
        return ''


def directory_bytes(directory, skip=frozenset({'input.png'})):
    """Bytes of the regular files directly in directory, outside skip. A child may rename or remove a file between the
    listing and the size read (the music adapter writes progress.json.tmp and replaces it onto progress.json many times a
    second); such a file is not counted now and is counted under its new name on the next poll, instead of failing the
    request with FileNotFoundError ([WinError 2]) or, for a file Windows is still deleting, PermissionError. Sizes come
    from os.stat, not the listing: on Windows a directory entry's cached size lags a file the child is still writing."""
    total = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name in skip:
                continue
            try:
                if entry.is_file():
                    total += os.stat(entry.path).st_size
            except (FileNotFoundError, PermissionError):
                continue
    return total


def run_owned(argv, directory, env, check, max_output_bytes, label='Native', observe=None):
    def budget():
        check()
        size = directory_bytes(directory)
        if size > max_output_bytes:
            raise EngineError(f'{label} output exceeded its byte budget', 413, 'request_too_large')
    process = job = None
    with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
        try:
            check()
            kwargs = dict(cwd=directory, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            if os.name == 'nt':
                from ._windows_job import OwnedJob
                job = OwnedJob(argv, **kwargs)
                process = job.process
            else:
                process = subprocess.Popen(argv, start_new_session=True, **kwargs)
            while process.poll() is None:
                budget()
                if observe:observe()
                time.sleep(.01)
            budget()
            if observe:observe()
            if process.returncode:
                # Keep diagnostics local to the request. The message never carries arbitrary paths/log contents; the exit code
                # and a bounded, redacted stderr tail ride on the exception (read here, while the request directory still
                # exists) so the caller can retain its own persistent failure record before the directory is deleted.
                stderr.flush()
                failure = EngineError(f'{label} process exited unsuccessfully', 502, 'native_process_failed')
                failure.returncode = process.returncode
                failure.stderr_tail = stderr_tail(directory / 'stderr.log')
                raise failure
        finally:
            if job is not None:
                job.__exit__(*sys.exc_info())
            elif process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
