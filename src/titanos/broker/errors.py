"""The broker's single error-code table and error type.

Every broker package raises BrokerError with a code from ERROR_CODES, never ad-hoc JSON; broker.envelopes renders the
code in the inbound protocol's native envelope. Rows marked no_retry also carry `x-should-retry: false`, because
Stainless-generated SDKs (OpenAI, Anthropic) otherwise retry 408, 409, 429 and every status of 500 or above.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..engine import EngineError

Origin = Literal['titanos', 'policy', 'upstream', 'network', 'local']
ORIGINS = frozenset({'titanos', 'policy', 'upstream', 'network', 'local'})


@dataclass(frozen=True)
class ErrorSpec:
    status: int
    anthropic_type: str
    gemini_status: str
    no_retry: bool


def anthropic_type(status: int) -> str:
    """Anthropic error type for an HTTP status (400 invalid_request_error ... others api_error)."""
    return {400: 'invalid_request_error', 401: 'authentication_error', 402: 'permission_error', 403: 'permission_error',
            404: 'not_found_error', 413: 'request_too_large', 429: 'rate_limit_error', 503: 'overloaded_error',
            529: 'overloaded_error'}.get(status, 'api_error')


def gemini_status(status: int) -> str:
    """Google canonical status for an HTTP status. 409 and 499 use the canonical ABORTED and CANCELLED names."""
    return {400: 'INVALID_ARGUMENT', 413: 'INVALID_ARGUMENT', 422: 'INVALID_ARGUMENT', 401: 'UNAUTHENTICATED',
            402: 'PERMISSION_DENIED', 403: 'PERMISSION_DENIED', 404: 'NOT_FOUND', 409: 'ABORTED', 429: 'RESOURCE_EXHAUSTED',
            499: 'CANCELLED', 503: 'UNAVAILABLE', 504: 'DEADLINE_EXCEEDED'}.get(status, 'INTERNAL')


# (status, no_retry) per code; the per-protocol type and status follow from the HTTP status.
_TABLE: dict[str, tuple[int, bool]] = {
    # 400
    'model_required': (400, False),
    'invalid_request': (400, False),
    'ambiguous_credentials': (400, False),
    'location_unsatisfiable': (400, False),
    'broker_capability_unavailable': (400, False),
    'provenance_unavailable_for_cloud': (400, False),
    'broker_config_contains_secret': (400, False),
    'idempotency_key_required': (400, False),
    'budget_required': (400, False),
    # 401
    'authentication_error': (401, False),
    'key_expired': (401, False),
    'key_revoked': (401, False),
    'query_credential_refused': (401, False),
    # 402
    'budget_exceeded': (402, True),
    'budget_unpriced_target': (402, True),
    # 403
    'scope_denied': (403, True),
    'cloud_disabled': (403, True),
    'role_forbids': (403, True),
    'delegation_required': (403, True),
    'transport_insecure': (403, True),
    'keyless_mode_refused': (403, True),
    'data_policy_denied': (403, True),
    # 404
    'model_not_found': (404, False),
    'response_not_found': (404, False),
    'not_found': (404, False),
    # 409
    'broker_name_conflict': (409, True),
    'advisor_profile_unconfigured': (409, True),
    'last_key_required': (409, True),
    'registry_busy': (409, False),
    # 413
    'request_too_large': (413, False),
    # 422
    'credential_validation_failed': (422, False),
    # 429
    'rate_limit_exceeded': (429, False),
    'concurrency_exceeded': (429, False),
    'upstream_rate_limited': (429, False),
    'auth_throttled': (429, False),
    # 499
    'cancelled': (499, False),
    # 502
    'upstream_auth_failed': (502, True),
    'upstream_bad_response': (502, False),
    'egress_refused': (502, True),
    'upstream_redirect_refused': (502, True),
    'upstream_response_too_large': (502, True),
    # 503
    'upstream_unavailable': (503, False),
    'breaker_open': (503, False),
    'credential_unavailable': (503, True),
    'vault_unavailable': (503, True),
    'vault_sealed': (503, True),
    'classifier_unavailable': (503, True),
    'upstream_unconfigured': (503, True),
    # 504
    'upstream_timeout': (504, False),
    'deadline_exceeded': (504, False),
    # 508
    'loop_detected': (508, True),
    # in-stream terminal type (status used only when it must be rendered before commit)
    'stream_interrupted': (502, False),
}

ERROR_CODES: dict[str, ErrorSpec] = {code: ErrorSpec(status, anthropic_type(status), gemini_status(status), no_retry)
                                     for code, (status, no_retry) in _TABLE.items()}

# 5xx codes whose meaning is "try again later or elsewhere"; every other 5xx code is no_retry (tests assert this).
RETRYABLE_5XX = frozenset({'upstream_bad_response', 'upstream_unavailable', 'breaker_open', 'upstream_timeout',
                           'deadline_exceeded', 'stream_interrupted'})


def spec(code: str) -> ErrorSpec:
    """The table row for code; KeyError for an unknown code (a broker package must not invent codes)."""
    return ERROR_CODES[code]


class BrokerError(EngineError):
    """A broker refusal or failure with a table code. status defaults to the table status; origin feeds
    X-Titanos-Error-Origin; headers are extra response headers (never secrets)."""

    def __init__(self, message: str, code: str, *, status: int | None = None, retry_after: float | None = None,
                 origin: Origin = 'titanos', headers: dict[str, str] | None = None):
        if code not in ERROR_CODES:
            raise ValueError(f'Unknown broker error code: {code!r}')
        if origin not in ORIGINS:
            raise ValueError(f'Unknown broker error origin: {origin!r}')
        super().__init__(message, ERROR_CODES[code].status if status is None else status, code)
        self.retry_after = retry_after
        self.origin = origin
        self.headers = dict(headers or {})

    @property
    def no_retry(self) -> bool:
        return ERROR_CODES[self.code].no_retry

    def response_headers(self) -> dict[str, str]:
        """X-Titanos-Error-Origin, Retry-After (whole seconds, rounded up) and x-should-retry: false where the table says so."""
        return error_headers(self.code, origin=self.origin, retry_after=self.retry_after, extra=self.headers)


class BrokerConfigSecretError(BrokerError, ValueError):
    """A broker document carries a secret-shaped string: ValueError at load, HTTP 400 broker_config_contains_secret."""

    def __init__(self, message: str = 'Broker configuration contains a secret-shaped value; reference a credential id instead'):
        super().__init__(message, 'broker_config_contains_secret')


class EgressRefused(BrokerError):
    """An outbound request was refused before connect (origin mismatch, blocked address, role switch or egress mode)."""

    def __init__(self, message: str = 'Outbound request refused', *, origin: Origin = 'policy'):
        super().__init__(message, 'egress_refused', origin=origin)


def error_headers(code: str, *, origin: str = 'titanos', retry_after: float | None = None,
                  extra: dict[str, str] | None = None) -> dict[str, str]:
    import math
    headers = {'X-Titanos-Error-Origin': origin}
    if retry_after is not None and math.isfinite(retry_after) and retry_after >= 0:
        headers['Retry-After'] = str(max(0, math.ceil(retry_after)))
    row = ERROR_CODES.get(code)
    if row is not None and row.no_retry:
        headers['x-should-retry'] = 'false'
    for name, value in (extra or {}).items():
        headers[name] = value
    return headers


__all__ = ['BrokerConfigSecretError', 'BrokerError', 'ERROR_CODES', 'EgressRefused', 'ErrorSpec', 'ORIGINS', 'Origin',
           'RETRYABLE_5XX', 'anthropic_type', 'error_headers', 'gemini_status', 'spec']
