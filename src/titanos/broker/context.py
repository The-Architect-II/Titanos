"""Per-request ContextVars, so thread-side code (API Plus stages, the artifact writer) sees the principal, hints and
request id of the request it serves.

Request-path threads get a copy of the context: asyncio.to_thread copies it, and Starlette's threadpool already carries
Titanos' _request_settings ContextVar the same way. Job worker threads do NOT inherit the context, so job ownership comes
from the job record (owner_key_id), never from these variables.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from .types import Hints, Principal

PRINCIPAL: ContextVar[Principal | None] = ContextVar('titanos_broker_principal', default=None)
HINTS: ContextVar[Hints | None] = ContextVar('titanos_broker_hints', default=None)
REQUEST_ID: ContextVar[str | None] = ContextVar('titanos_broker_request_id', default=None)


@dataclass(frozen=True)
class BindToken:
    principal: Token
    hints: Token
    request_id: Token


def current_principal() -> Principal | None:
    return PRINCIPAL.get()


def current_hints() -> Hints | None:
    return HINTS.get()


def current_request_id() -> str | None:
    return REQUEST_ID.get()


def bind(principal: Principal | None, hints: Hints | None) -> BindToken:
    """Set all three variables for the current context; pass the result to reset() in a finally block."""
    return BindToken(PRINCIPAL.set(principal), HINTS.set(hints),
                     REQUEST_ID.set(hints.request_id if hints is not None and hints.request_id else None))


def reset(token: BindToken) -> None:
    REQUEST_ID.reset(token.request_id)
    HINTS.reset(token.hints)
    PRINCIPAL.reset(token.principal)


__all__ = ['BindToken', 'HINTS', 'PRINCIPAL', 'REQUEST_ID', 'bind', 'current_hints', 'current_principal',
           'current_request_id', 'reset']
