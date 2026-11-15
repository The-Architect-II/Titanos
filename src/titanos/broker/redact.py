"""Central redaction for the broker: secret shapes, an exact-value registry, log scrubbing and the child-env deny list.

It extends owned_command.redact (the bounded child-output renderer): scrub() applies owned_command's credential rules
(labelled assignments, CLI flags, JWTs) and a 'Bearer'/'Basic' scheme followed by a token of at least 8 characters,
plus provider key shapes with minimum lengths, header values, query credentials and every registered exact value.
scrub_child() is owned_command.redact over scrub().

Log scrubbing is installed in two places by install_logging_redaction():
1. a global LogRecord factory that scrubs msg, args, exc_text and stack_info when any record is created, which covers
   child loggers (httpx, httpcore) whose records propagate past logger-level filters;
2. a RedactingFilter on every handler of the root, uvicorn, uvicorn.error and uvicorn.access loggers, re-applied at
   lifespan startup because uvicorn's dictConfig replaces handlers after create_app. It scrubs exc_text at emit.
Both are idempotent. Importing this module changes nothing.
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Mapping

from ..owned_command import _CLI_CREDENTIAL, _CREDENTIAL, _JWT
from ..owned_command import redact as _owned_redact

REDACTED = '<redacted>'
MIN_SECRET_LENGTH = 8
_B = r'(?<![A-Za-z0-9_-])'  # a token starts at a word boundary that also excludes '_' and '-'

SECRET_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(_B + r'sk-ant-[A-Za-z0-9_-]{20,}'),
    re.compile(_B + r'sk-proj-[A-Za-z0-9_-]{20,}'),
    re.compile(_B + r'sk-[A-Za-z0-9_-]{20,}'),
    re.compile(_B + r'xai-[A-Za-z0-9]{20,}'),
    re.compile(_B + r'gsk_[A-Za-z0-9]{20,}'),
    re.compile(_B + r'nvapi-[A-Za-z0-9_-]{20,}'),
    re.compile(_B + r'hf_[A-Za-z0-9]{20,}'),
    re.compile(_B + r'AIza[0-9A-Za-z_-]{35}'),
    re.compile(_B + r'tk_[a-z2-7]{10}_[A-Za-z0-9_-]{43}'),
    # header values: Authorization (any scheme), x-api-key, xi-api-key, api-key, x-goog-api-key
    re.compile(r'(?i)\b(proxy-authorization|authorization|x-goog-api-key|xi-api-key|x-api-key|api-key)'
               r'(["\']?\s*[:=]\s*["\']?)(?:(?:bearer|basic|token)\s+)?[^\s"\',;&]+'),
    re.compile(r'(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}'),
    # query credentials
    re.compile(r'(?i)([?&](?:key|api_key|access_token)=)[^&\s#"\']+'),
    # JWT-shaped tokens
    re.compile(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*'),
)
QUERY_CREDENTIAL = SECRET_PATTERNS[11]
# Ratified R2: a bare 'Bearer'/'Basic' scheme is redacted only when a token of at least 8 characters from the token68
# alphabet follows (no whitespace), so ordinary text such as 'basic auth' survives. owned_command._SCHEME_CREDENTIAL
# (any following word) still applies in scrub_child's owned_command.redact pass over child-process output.
SCHEME_CREDENTIAL = re.compile(r'(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}')


def _replacement(match: re.Match) -> str:
    groups = match.groups()
    if len(groups) >= 2 and groups[0] and groups[1] is not None:  # header label + separator
        return groups[0] + groups[1] + REDACTED
    if len(groups) == 1 and groups[0]:  # 'Bearer' or '?key=' prefix
        return groups[0] + (' ' if groups[0].lower() == 'bearer' else '') + REDACTED
    return REDACTED


class SecretRegistry:
    """Exact values of every resolved secret, the bootstrap key and issued tk_ tokens, scrubbed wherever they appear.
    Values shorter than MIN_SECRET_LENGTH are refused (a short value would scrub ordinary text)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._values: set[str] = set()
        self._pattern: re.Pattern | None = None

    def add(self, value: str) -> None:
        if not isinstance(value, str) or len(value) < MIN_SECRET_LENGTH:
            raise ValueError(f'Secret values shorter than {MIN_SECRET_LENGTH} characters are refused')
        with self._lock:
            if value not in self._values:
                self._values.add(value)
                self._pattern = None

    def discard(self, value: str) -> None:
        with self._lock:
            if value in self._values:
                self._values.discard(value)
                self._pattern = None

    def __len__(self) -> int:
        return len(self._values)

    def _compiled(self) -> re.Pattern | None:
        with self._lock:
            if self._pattern is None and self._values:
                self._pattern = re.compile('|'.join(re.escape(v) for v in sorted(self._values, key=len, reverse=True)))
            return self._pattern

    def contains_in(self, text: str) -> bool:
        pattern = self._compiled()
        return bool(pattern and pattern.search(text))

    def scrub(self, text: str) -> str:
        pattern = self._compiled()
        return pattern.sub(REDACTED, text) if pattern else text


REGISTRY = SecretRegistry()


def scrub(text) -> str:
    """Registered exact values, provider key shapes, header values, query credentials, JWTs and owned_command's
    credential forms replaced by <redacted> (labels kept). Non-str input is rendered with str() first."""
    if not isinstance(text, str):
        text = str(text)
    text = REGISTRY.scrub(text)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(_replacement, text)
    text = _CREDENTIAL.sub(lambda m: m.group(1) + '=' + REDACTED, text)
    text = _CLI_CREDENTIAL.sub(lambda m: m.group(1) + ' ' + REDACTED, text)
    text = SCHEME_CREDENTIAL.sub(lambda m: m.group(1) + ' ' + REDACTED, text)
    return _JWT.sub(REDACTED, text)


def scrub_child(text) -> str:
    """owned_command.redact over scrub(): bounded child output with paths reduced and long runs removed."""
    return _owned_redact(scrub(text))


def scrub_obj(obj, _depth: int = 0):
    """scrub() applied to every string (keys included) inside nested dicts, lists, tuples and sets; other leaves are kept."""
    if _depth > 32:
        return '<nested>'
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, Mapping):
        return {scrub_obj(k, _depth + 1): scrub_obj(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v, _depth + 1) for v in obj]
    if isinstance(obj, tuple):
        return tuple(scrub_obj(v, _depth + 1) for v in obj)
    if isinstance(obj, (set, frozenset)):
        return type(obj)(scrub_obj(v, _depth + 1) for v in obj)
    return obj


def looks_secret(text) -> bool:
    """True when text carries a secret shape (SECRET_PATTERNS) or a registered exact value."""
    if not isinstance(text, str):
        return False
    return any(p.search(text) for p in SECRET_PATTERNS) or REGISTRY.contains_in(text)


def _scrub_arg(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return scrub(value)
    rendered = str(value)
    cleaned = scrub(rendered)
    return value if cleaned == rendered else cleaned


def _scrub_record(record: logging.LogRecord) -> None:
    """Format first, then scrub: scrubbing the format string alone would turn a labelled placeholder
    ('Authorization: Bearer %s', 'api_key=%s') into '<redacted>' while its argument stays, and the resulting
    formatting error would print the raw arguments to stderr. A record whose arguments cannot be formatted keeps only
    its scrubbed format string and withholds the arguments, so logging never reports it with raw arguments.

    The scrubbed message keeps its positional arguments (formatters such as uvicorn's AccessFormatter read
    record.args) only when the separately scrubbed parts format to exactly the scrubbed formatted message;
    otherwise msg becomes that scrubbed message and args None."""
    if record.args:
        try:
            full = scrub(record.getMessage())
        except Exception:  # a broken record: keep the text, never the arguments (an unshaped secret has no pattern)
            record.msg, record.args = scrub(record.msg) + ' [log arguments withheld: the message did not format]', None
        else:
            msg = scrub(record.msg) if isinstance(record.msg, str) else _scrub_arg(record.msg)
            if isinstance(record.args, Mapping):
                args = {k: _scrub_arg(v) for k, v in record.args.items()}
            else:
                args = tuple(_scrub_arg(a) for a in record.args)
            try:
                parts_ok = isinstance(msg, str) and msg % args == full
            except Exception:
                parts_ok = False
            record.msg, record.args = (msg, args) if parts_ok else (full, None)
    else:
        record.msg = scrub(record.msg) if isinstance(record.msg, str) else _scrub_arg(record.msg)
    if record.exc_info and not record.exc_text:
        try:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        except Exception:  # pragma: no cover - a broken traceback must not break logging
            record.exc_text = None
    if record.exc_text:
        record.exc_text = scrub(record.exc_text)
    if record.stack_info:
        record.stack_info = scrub(record.stack_info)


class RedactingFilter(logging.Filter):
    """Handler filter: scrubs msg, args, exc_text and stack_info at emit. Never drops a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        _scrub_record(record)
        return True


REDACTED_LOGGERS = (None, 'uvicorn', 'uvicorn.error', 'uvicorn.access')
_INSTALL_LOCK = threading.Lock()


def install_logging_redaction() -> None:
    """Install the scrubbing LogRecord factory (once) and a RedactingFilter on every handler of the root and uvicorn
    loggers (again on each call, so lifespan startup restores filters after uvicorn's dictConfig)."""
    with _INSTALL_LOCK:
        current = logging.getLogRecordFactory()
        if not getattr(current, '_titanos_redacting', False):
            def factory(*args, **kwargs):
                record = current(*args, **kwargs)
                _scrub_record(record)
                return record
            factory._titanos_redacting = True
            factory._titanos_previous = current
            logging.setLogRecordFactory(factory)
        for name in REDACTED_LOGGERS:
            for handler in logging.getLogger(name).handlers:
                if not any(isinstance(f, RedactingFilter) for f in handler.filters):
                    handler.addFilter(RedactingFilter())


def uninstall_logging_redaction() -> None:
    """Remove the factory and the handler filters (tests only)."""
    with _INSTALL_LOCK:
        current = logging.getLogRecordFactory()
        while getattr(current, '_titanos_redacting', False):
            current = current._titanos_previous
        logging.setLogRecordFactory(current)
        for name in REDACTED_LOGGERS:
            for handler in logging.getLogger(name).handlers:
                for f in [f for f in handler.filters if isinstance(f, RedactingFilter)]:
                    handler.removeFilter(f)


# The child-env deny list and filter live in titanos.child_env (standard library only, so titanos.native_process can
# use them under `python -I -S`); they are re-exported here.
from ..child_env import PROVIDER_ENV_DENY, PROVIDER_ENV_DENY_SUFFIXES, child_env  # noqa: E402


__all__ = ['MIN_SECRET_LENGTH', 'PROVIDER_ENV_DENY', 'PROVIDER_ENV_DENY_SUFFIXES', 'QUERY_CREDENTIAL', 'REDACTED',
           'REDACTED_LOGGERS', 'REGISTRY', 'RedactingFilter', 'SCHEME_CREDENTIAL', 'SECRET_PATTERNS', 'SecretRegistry',
           'child_env', 'install_logging_redaction', 'looks_secret', 'scrub', 'scrub_child', 'scrub_obj',
           'uninstall_logging_redaction']
