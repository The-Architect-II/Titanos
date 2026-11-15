"""Upstream error mapping: an upstream status and error body (or an error-first stream event) become one UpstreamError,
then a broker error code rendered in the INBOUND protocol's envelope.

Table (status -> broker code, outcome kind):
- 400, 404, 410, 413, 422 -> passed through in the inbound envelope after scrub(), origin upstream; never fallback
  triggers and never counted by breakers (kind client_error). 404 and 410 answer 404: model_not_found when the 404 is
  about the model (names_model below), otherwise not_found. On a call that names a model (dispatch and the stage invoker
  apply model_failure; the handle-bound Responses routes do not) a 410 Gone, and a 404 that names_model, become 404
  model_not_found whose message says the provider reports the model retired (410) or unavailable (404), unless the
  provider code names another resource (names_other_resource: the code contains voice, file, response, thread,
  assistant, batch, vector_store or upload and not 'model'), which stays 404 not_found on a 410 too, whatever the
  message says. A 404 about anything else (ElevenLabs voice_not_found, a missing file, a missing response id, a bare
  404 without a body) stays 404 not_found with the provider's own scrubbed message. A Gemini 400 whose details carry reason API_KEY_INVALID is an
  auth failure instead (below).
- names_model (a 404 about the model): the provider code says so (OpenAI-compatible error.code model_not_found, Azure
  DeploymentNotFound - an Azure deployment is the model's address - or any code containing 'model'); a code naming
  another resource (<thing>_not_found / <Thing>NotFound such as voice_not_found, or any code containing voice, file,
  response, thread, assistant, batch, vector_store or upload) never is, whatever the message says; otherwise, for generic codes
  (Anthropic not_found_error, Gemini NOT_FOUND, ElevenLabs not_found, none) the message must refer to the model: it
  contains the requested or upstream model id, or the word model (Anthropic 'model: <id>', Gemini 'models/<id> is not
  found', OpenAI-compatible 'model ... not found'), or it is a provider's
  "Function '<id>': Not found for account" (a catalog that serves every model as a function).
- 401, 403 -> 502 upstream_auth_failed [no-retry], kind auth, never the client's own 401; not counted by breakers. The
  exception is a plain 403 in the provider's own envelope that is not a key rejection (the OpenAI-compatible, Anthropic,
  Gemini and ElevenLabs envelopes; for OpenAI an {error: {...}} envelope whose code is not invalid_api_key: organization
  verification, model access, unsupported region): the credential is fine and the provider refused this request for
  this account, so it passes through like a 400 (403 invalid_request in the inbound envelope; OpenAI-shaped clients keep
  the provider's error.code, error.type and error.param), never a fallback trigger.
- OpenAI-compatible quota and billing stops (error.code or error.type insufficient_quota, billing_hard_limit_reached,
  billing_not_active; OpenAI answers them as 429 or 400) are a provider billing state like a 402: 503
  upstream_unavailable, kind server_error, the provider's message kept (never retried as a rate limit with Retry-After).
- 408 and timeouts -> 504 upstream_timeout (kind timeout).
- 429 -> 429 upstream_rate_limited with the upstream Retry-After (Retry-After seconds or HTTP date, retry-after-ms, or a
  Gemini RetryInfo retryDelay), kind rate_limited, no breaker count. So is an OpenAI-compatible 498 (Groq: flex-tier
  capacity exceeded).
- 402 (provider billing) and 5xx (500, 502, 503, 504, 529 ...) -> 503 upstream_unavailable, kind server_error, counted. So
  is a 200 stream whose first event is an overloaded or api_error class error, and an ElevenLabs 400/401/403 whose code
  is a quota or credit code (ELEVENLABS_QUOTA_CODES: quota_exceeded, insufficient_credits ...).
- 3xx -> 502 upstream_redirect_refused; any other 4xx -> 502 upstream_bad_response.

Credential marking (ratified R10): a credential is marked failed:upstream_auth on a 401 (ElevenLabs: only when the
401's detail.code or legacy detail.status is a key-rejection code, ELEVENLABS_KEY_REJECTED, or the body carries no code,
because ElevenLabs also answers 401 quota_exceeded and missing_permissions; other providers: never for a restricted
key's per-endpoint refusal, scope_refusal, OpenAI's 401 'insufficient permissions ... Missing scopes'); on a 403 ONLY
when the body carries the provider's auth error code (Anthropic error.type authentication_error; Gemini error.details
reason API_KEY_INVALID; ElevenLabs detail.code or detail.status invalid_api_key; OpenAI-compatible error.code
invalid_api_key); and on a Gemini 400 with reason API_KEY_INVALID. A plain 403 (moderation, region, model permission)
still answers 502 upstream_auth_failed (OpenAI-compatible envelopes: passed through as 403, above) but never takes a
shared credential offline. egress.EgressClient calls credential_failure(); on an inference call (any caller chooses its
endpoint and body) it marks at once only an answer that carries the provider's key-rejection code (key_rejected), and
has the upstream's own probe endpoint confirm any other 401 first (Broker._auth_suspect), so no caller can take a shared
credential offline by calling an endpoint the operator's key has no permission for.

Messages are extracted from the provider's envelope, scrubbed (redact.scrub: registered secret values, key shapes) and
bounded; raw upstream bodies are never forwarded. Provider error codes are kept only when they are a plain identifier
that scrub() leaves unchanged (a code field that carries a key-shaped or registered secret value is dropped), because
passthrough_4xx copies the code into error.code for OpenAI-shaped clients.

Stream error events: Anthropic/Gemini/OpenAI chat {error: {...}}, Responses response.failed {response: {error}} and the
Responses 'error' event, whose code and message are top-level fields ({type: 'error', code, message, param}).
"""
from __future__ import annotations

import email.utils
import json
import math
import re
import time

from ..envelopes import body as envelope_body
from ..errors import BrokerError, error_headers
from ..redact import scrub
from ..types import UpstreamError

MESSAGE_MAX = 1000
# 403 passes through only for the OpenAI-compatible plain-403 case (classify); every other 403 is kind auth
PASSTHROUGH_STATUSES = frozenset({400, 403, 404, 410, 413, 422})
# OpenAI-compatible billing stops (error.code or error.type): a provider billing state, never a retryable rate limit
OPENAI_QUOTA_CODES = frozenset({'insufficient_quota', 'billing_hard_limit_reached', 'billing_not_active'})
_SAFE_PARAM = re.compile(r'^[A-Za-z0-9_.\[\]-]{1,128}$')
GONE_STATUSES = frozenset({404, 410})  # a provider's 'no such model' (404) and 'model retired' (410 Gone)
# provider codes that say the 404 is about the model (an Azure deployment is the model's address)
MODEL_NOT_FOUND_CODES = frozenset({'model_not_found', 'DeploymentNotFound'})
# generic not-found codes: whether they are about the model is read from the message
GENERIC_NOT_FOUND_CODES = frozenset({'not_found', 'not_found_error', 'NOT_FOUND', 'NotFound', '404',
                                     'invalid_request_error'})
_OTHER_RESOURCE_CODE = re.compile(r'(?:_not_?found|NotFound)$', re.IGNORECASE)
# a provider code naming one of these resources (and not 'model') is about that resource, never about the model
_OTHER_RESOURCE_WORD = re.compile(r'voice|file|response|thread|assistant|batch|vector_?store|upload', re.IGNORECASE)
_MODEL_WORD = re.compile(r'model', re.IGNORECASE)
_FUNCTION_NOT_FOUND = re.compile(r'\bfunction\b.*\bnot found for account\b', re.IGNORECASE | re.DOTALL)
_SAFE_CODE = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')
ADAPTERS = ('openai_compat', 'anthropic', 'gemini', 'elevenlabs', 'ollama_native')
# ElevenLabs 401 codes (detail.code, or the legacy detail.status) that reject the key itself; every other 401 code leaves
# the shared credential usable. Quota and credit codes are a provider billing state (503 upstream_unavailable, as a 402);
# missing_permissions (a restricted-scope key) and detected_unusual_activity stay 502 upstream_auth_failed, unmarked.
ELEVENLABS_KEY_REJECTED = frozenset({'invalid_api_key', 'missing_api_key', 'needs_authorization',
                                     'invalid_authorization_header', 'unauthorized', 'sign_in_required'})
ELEVENLABS_QUOTA_CODES = frozenset({'quota_exceeded', 'insufficient_credits', 'payment_required',
                                    'max_character_limit_exceeded'})
# OpenAI's 401 for a restricted (or project-limited) key on an endpoint outside its permissions (scope_refusal)
_SCOPE_REFUSAL = re.compile(r'insufficient permissions|missing scopes?\b', re.IGNORECASE)


class UpstreamFailure(BrokerError):
    """A classified upstream failure (non-2xx status or an error-first stream). .upstream is the UpstreamError; the broker
    code, status and Retry-After follow the table. Dispatch renders it with passthrough_4xx (client errors) or
    envelopes.render_error, and uses .upstream for breakers, fallback and the decision."""

    def __init__(self, err: UpstreamError):
        code, status = broker_code(err)
        super().__init__(err.message or 'Upstream error', code, status=status, retry_after=err.retry_after,
                         origin='upstream')
        self.upstream = err

    @property
    def passthrough(self) -> bool:
        return is_passthrough(self.upstream)


def _json(body) -> object:
    if isinstance(body, (dict, list)):
        return body
    if isinstance(body, (bytes, bytearray)):
        try:
            return json.loads(bytes(body).decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return None
    if isinstance(body, str):
        try:
            return json.loads(body)
        except ValueError:
            return None
    return None


def _text(value) -> str:
    message = scrub(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    if value is not None else '')
    message = ' '.join(message.split())
    return message[:MESSAGE_MAX] + ('...' if len(message) > MESSAGE_MAX else '')


def safe_code(code) -> str | None:
    """The provider error code when it is a plain identifier that scrub() leaves unchanged, else None."""
    if code is None or isinstance(code, bool) or not isinstance(code, (str, int)):
        return None
    text = str(code)
    if not _SAFE_CODE.match(text) or scrub(text) != text:
        return None
    return text


def _gemini_reasons(error: dict) -> list[str]:
    details = error.get('details') if isinstance(error, dict) else None
    return [d.get('reason') for d in details or [] if isinstance(d, dict) and isinstance(d.get('reason'), str)]


def provider_fields(body, adapter: str) -> tuple[str | None, str]:
    """(provider error code, raw message) from a provider error envelope (message not yet scrubbed)."""
    data = _json(body)
    raw_text = body.decode('utf-8', 'replace') if isinstance(body, (bytes, bytearray)) else str(body or '')
    if adapter == 'anthropic':
        error = data.get('error') if isinstance(data, dict) else None
        if isinstance(error, dict):
            return error.get('type'), error.get('message') or ''
    elif adapter == 'gemini':
        error = data.get('error') if isinstance(data, dict) else None
        if isinstance(error, dict):
            reasons = _gemini_reasons(error)
            return (reasons[0] if reasons else error.get('status')), error.get('message') or ''
    elif adapter == 'elevenlabs':
        detail = data.get('detail') if isinstance(data, dict) else None
        if isinstance(detail, dict):
            return elevenlabs_code(detail), detail.get('message') or ''
        if isinstance(detail, list) and detail:  # FastAPI-shaped validation errors
            first = detail[0] if isinstance(detail[0], dict) else {}
            loc = '.'.join(str(p) for p in first.get('loc') or [])
            return first.get('type'), (f'{loc}: ' if loc else '') + str(first.get('msg') or '')
    elif adapter == 'ollama_native':
        if isinstance(data, dict) and isinstance(data.get('error'), str):
            return None, data['error']
    if isinstance(data, dict):  # OpenAI-compatible family (OpenAI, Azure, OpenRouter, Mistral variants)
        error = data.get('error')
        if isinstance(error, dict):
            code = error.get('code')
            code = str(code) if isinstance(code, (str, int)) and not isinstance(code, bool) else error.get('type')
            return code, error.get('message') or ''
        if isinstance(error, str):
            return None, error
        if isinstance(data.get('message'), str):  # Mistral {object: error, message, code} and {message, request_id}
            code = data.get('code')
            return (str(code) if isinstance(code, (str, int)) and code is not None else data.get('type')), data['message']
        if isinstance(data.get('detail'), str):
            return None, data['detail']
    return None, raw_text


def elevenlabs_code(detail: dict) -> str | None:
    """The ElevenLabs error code of a {detail: {...}} envelope: the current 'code' field, else the legacy 'status'."""
    for name in ('code', 'status'):
        value = detail.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _elevenlabs_401_code(body) -> str | None:
    data = _json(body)
    detail = data.get('detail') if isinstance(data, dict) else None
    return elevenlabs_code(detail) if isinstance(detail, dict) else None


def scope_refusal(body, *, adapter: str | None) -> bool:
    """Whether an error body is a restricted key's refusal of one endpoint it has no permission for (OpenAI answers a
    restricted or project-limited key with HTTP 401 'You have insufficient permissions for this operation. Missing
    scopes: ...' on every endpoint outside its permissions): the key is genuine, so the answer never marks it."""
    _code, message = provider_fields(body, adapter or 'openai_compat')
    return bool(_SCOPE_REFUSAL.search(message or ''))


def key_rejected(status: int, body, *, adapter: str | None) -> bool:
    """Whether this answer carries the provider's own key-rejection code: OpenAI-compatible error.code (or top-level
    code) invalid_api_key, Anthropic error.type authentication_error, Gemini reason API_KEY_INVALID (400, 401 or 403),
    an ElevenLabs key-rejection code, Ollama's 401 'unauthorized'. Such an answer rejects the credential whatever endpoint
    or body the request named. credential_failure() also counts a 401 that carries no such code; an inference call
    (whose endpoint and body any caller chooses) marks the credential on such a 401 only after the upstream's own probe
    endpoint rejects it too (egress.EgressClient, Broker._auth_suspect)."""
    if status not in (400, 401, 403):
        return False
    data = _json(body)
    if not isinstance(data, dict):
        return False
    if adapter == 'gemini' or (adapter is None and isinstance(data.get('error'), dict)
                               and 'details' in data['error']):
        error = data.get('error')
        return isinstance(error, dict) and 'API_KEY_INVALID' in _gemini_reasons(error)
    if status == 400:
        return False
    if adapter == 'anthropic':
        error = data.get('error')
        return isinstance(error, dict) and error.get('type') == 'authentication_error'
    if adapter == 'elevenlabs':
        detail = data.get('detail')
        return isinstance(detail, dict) and elevenlabs_code(detail) in ELEVENLABS_KEY_REJECTED
    error = data.get('error')
    if adapter == 'ollama_native' and isinstance(error, str):
        return status == 401 and error.strip().lower() == 'unauthorized'
    if isinstance(error, dict):
        return error.get('code') == 'invalid_api_key'
    return data.get('code') == 'invalid_api_key'


def credential_failure(status: int, body, *, adapter: str | None) -> bool:
    """R10: whether this upstream answer proves the credential itself is rejected (mark failed:upstream_auth). A 401
    without the provider's key-rejection code (key_rejected) counts too; egress confirms it first on inference calls."""
    if status == 401:
        if adapter == 'elevenlabs':
            # ElevenLabs also answers 401 for an exhausted quota and for a restricted key that lacks one endpoint's
            # permission; only the key-rejection codes (or a 401 without a readable code) take the credential offline
            code = _elevenlabs_401_code(body)
            return code is None or code in ELEVENLABS_KEY_REJECTED
        # a restricted key refused one endpoint (OpenAI 'insufficient permissions ... Missing scopes'): the key is fine
        return key_rejected(status, body, adapter=adapter) or not scope_refusal(body, adapter=adapter)
    if status not in (400, 403):
        return False
    data = _json(body)
    if not isinstance(data, dict):
        return False
    if adapter == 'gemini' or (adapter is None and isinstance(data.get('error'), dict)
                               and 'details' in data['error']):
        error = data.get('error')
        return isinstance(error, dict) and 'API_KEY_INVALID' in _gemini_reasons(error)
    if status != 403:
        return False
    if adapter == 'anthropic':
        error = data.get('error')
        return isinstance(error, dict) and error.get('type') == 'authentication_error'
    if adapter == 'elevenlabs':
        detail = data.get('detail')
        return isinstance(detail, dict) and elevenlabs_code(detail) in ELEVENLABS_KEY_REJECTED
    error = data.get('error')
    if isinstance(error, dict):
        return error.get('code') == 'invalid_api_key'
    return data.get('code') == 'invalid_api_key'


def retry_after(headers, body=None, *, now: float | None = None) -> float | None:
    """Seconds from Retry-After (delta-seconds or HTTP date), retry-after-ms, or a Gemini RetryInfo retryDelay."""
    getter = getattr(headers, 'get', None)
    if getter is not None:
        value = getter('retry-after-ms') or getter('Retry-After-Ms')
        if value:
            try:
                ms = float(value)
                if math.isfinite(ms) and ms >= 0:
                    return ms / 1000.0
            except ValueError:
                pass
        value = getter('retry-after') or getter('Retry-After')
        if value:
            value = str(value).strip()
            try:
                seconds = float(value)
                if math.isfinite(seconds) and seconds >= 0:
                    return seconds
            except ValueError:
                try:
                    when = email.utils.parsedate_to_datetime(value)
                except (TypeError, ValueError):
                    when = None
                if when is not None:
                    return max(0.0, when.timestamp() - (time.time() if now is None else now))
    data = _json(body)
    error = data.get('error') if isinstance(data, dict) else None
    for detail in (error.get('details') if isinstance(error, dict) else None) or []:
        if isinstance(detail, dict) and str(detail.get('@type', '')).endswith('RetryInfo'):
            match = re.fullmatch(r'\s*([0-9]+(?:\.[0-9]+)?)s\s*', str(detail.get('retryDelay', '')))
            if match:
                return float(match.group(1))
    return None


def classify(status: int, headers, body: bytes, *, adapter: str) -> UpstreamError:
    """UpstreamError for an upstream non-2xx answer (message scrubbed, Retry-After parsed)."""
    code, raw = provider_fields(body, adapter)
    message = _text(raw) or f'Upstream answered {status}'
    after = retry_after(headers, body)
    param, type_ = openai_error_details(body) if adapter == 'openai_compat' else (None, None)
    if adapter == 'elevenlabs' and status in (400, 401, 403) and code in ELEVENLABS_QUOTA_CODES:
        kind = 'server_error'  # a provider billing state (like a 402), never a credential failure
    elif adapter == 'openai_compat' and status in (400, 429) and (safe_code(code) in OPENAI_QUOTA_CODES
                                                                  or type_ in OPENAI_QUOTA_CODES):
        kind, after = 'server_error', None  # a billing stop (like a 402): no Retry-After, never a rate limit
    elif status == 403 and _vendor_envelope(body, adapter) and not credential_failure(status, body, adapter=adapter):
        # a plain 403 in the provider's own envelope (OpenAI organization verification or model access, Anthropic
        # permission_error, Gemini PERMISSION_DENIED, an ElevenLabs permission detail): the provider refused this request
        # for this account, so it passes through like a 400 instead of reading as a Titanos credential failure
        kind = 'client_error'
    elif status in (401, 403) or (status == 400 and credential_failure(status, body, adapter=adapter)):
        kind = 'auth'
    elif status in PASSTHROUGH_STATUSES:
        kind = 'client_error'
    elif status == 408:
        kind = 'timeout'
    elif status == 429 or (status == 498 and adapter == 'openai_compat'):
        kind = 'rate_limited'  # Groq 498: flex-tier capacity exceeded, retry later (a capacity answer, not a bad request)
    elif status == 402 or status >= 500:
        kind = 'server_error'
    else:
        kind = 'client_error'
    return UpstreamError(kind=kind, status=status, provider_code=safe_code(code), message=message, retry_after=after,
                         provider_param=param, provider_type=type_)


def _openai_envelope(body) -> bool:
    """Whether body is an OpenAI-compatible error envelope ({error: {...}})."""
    data = _json(body)
    return isinstance(data, dict) and isinstance(data.get('error'), dict)


def _vendor_envelope(body, adapter: str) -> bool:
    """Whether body is the adapter's own error envelope: {error: {...}} for the OpenAI family, Anthropic ({type: error,
    error: {...}}) and Gemini (google.rpc.Status under error), {detail: {...}} for ElevenLabs."""
    data = _json(body)
    if not isinstance(data, dict):
        return False
    if adapter == 'elevenlabs':
        return isinstance(data.get('detail'), dict)
    return adapter in ('openai_compat', 'anthropic', 'gemini') and isinstance(data.get('error'), dict)


def openai_error_details(body) -> tuple[str | None, str | None]:
    """(error.param, error.type) of an OpenAI-compatible error envelope, each only when it is a plain identifier that
    scrub() leaves unchanged (a parameter path such as messages[0].content keeps its brackets)."""
    data = _json(body)
    error = data.get('error') if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return None, None
    param = error.get('param')
    param = param if isinstance(param, str) and _SAFE_PARAM.match(param) and scrub(param) == param else None
    return param, safe_code(error.get('type')) if isinstance(error.get('type'), str) else None


_STREAM_STATUS = {
    # Anthropic error.type
    'overloaded_error': 529, 'api_error': 500, 'rate_limit_error': 429, 'invalid_request_error': 400,
    'authentication_error': 401, 'permission_error': 403, 'not_found_error': 404, 'request_too_large': 413,
    'timeout_error': 504,
    # Google canonical status
    'UNAVAILABLE': 503, 'INTERNAL': 500, 'RESOURCE_EXHAUSTED': 429, 'INVALID_ARGUMENT': 400, 'DEADLINE_EXCEEDED': 504,
    'UNAUTHENTICATED': 401, 'PERMISSION_DENIED': 403, 'NOT_FOUND': 404,
    # OpenAI error.type
    'server_error': 500, 'rate_limit_exceeded': 429,
    # broker codes carried by IR ErrorEvents
    'upstream_unavailable': 503, 'upstream_rate_limited': 429, 'upstream_timeout': 504, 'upstream_auth_failed': 401,
}


def classify_stream_error(payload, *, adapter: str) -> UpstreamError:
    """UpstreamError for an in-stream error event (an error-first 200 stream, or a mid-stream error): the status is the
    equivalent HTTP status of the provider's error class (overloaded and api_error map to 5xx: upstream_unavailable)."""
    data = _json(payload) if not isinstance(payload, dict) else payload
    status = 500
    code, message = None, ''
    if isinstance(data, dict):
        error = data.get('error')
        if isinstance(error, dict):
            code = error.get('type') or error.get('status') or error.get('code')
            message = error.get('message') or ''
            if isinstance(error.get('code'), int) and not isinstance(error.get('code'), bool) and 400 <= error['code'] < 600:
                status = error['code']
            elif isinstance(code, str) and code in _STREAM_STATUS:
                status = _STREAM_STATUS[code]
        elif isinstance(error, str):
            message = error
        elif data.get('type') == 'error' and ('code' in data or 'message' in data):  # Responses 'error' event
            code = data.get('code')
            message = data.get('message') or ''
            if isinstance(code, str) and code in _STREAM_STATUS:
                status = _STREAM_STATUS[code]
        response = data.get('response')
        if isinstance(response, dict) and isinstance(response.get('error'), dict):  # Responses response.failed
            code = response['error'].get('code')
            message = response['error'].get('message') or ''
            if isinstance(code, str) and code in _STREAM_STATUS:
                status = _STREAM_STATUS[code]
    probe = classify(status, {}, b'', adapter=adapter)
    return UpstreamError(kind=probe.kind, status=status, provider_code=safe_code(code),
                         message=_text(message) or 'Upstream stream error', retry_after=None)


def network_error(exc: BaseException) -> UpstreamError:
    """UpstreamError for an httpx exception: connect failures (no request bytes sent) are network, timeouts timeout."""
    import httpx
    name = type(exc).__name__
    not_sent = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
    kind = 'timeout' if isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout) \
        else 'network'
    return UpstreamError(kind=kind, status=504 if kind == 'timeout' else 503, provider_code=name,
                         message=f'Upstream {"timed out" if kind == "timeout" else "connection failed"} ({name})',
                         request_bytes_sent=not not_sent)


def is_passthrough(err: UpstreamError) -> bool:
    return err.kind == 'client_error' and err.status in PASSTHROUGH_STATUSES


def names_other_resource(code: str | None) -> bool:
    """Whether a provider error code names a resource other than the model (it contains voice, file, response, thread,
    assistant, batch, vector_store or upload, and not 'model'): such an answer is never model_not_found, whatever its
    message says and whatever its status (404 or 410)."""
    code = code or ''
    return bool(code) and 'model' not in code.lower() and code not in MODEL_NOT_FOUND_CODES \
        and bool(_OTHER_RESOURCE_WORD.search(code))


def names_model(err: UpstreamError, *models: str | None) -> bool:
    """Whether a provider not-found answer is about the model (module docstring): by its provider code, else (generic
    codes only) by a message that contains one of `models` (requested ref, upstream model id), the word model, or the
    function-not-found sentence of a catalog that serves models as functions. A code naming another resource (voice_not_found ...) never is."""
    code = err.provider_code or ''
    if code in MODEL_NOT_FOUND_CODES or 'model' in code.lower():
        return True
    if names_other_resource(code):
        return False
    if code and code not in GENERIC_NOT_FOUND_CODES and _OTHER_RESOURCE_CODE.search(code):
        return False
    text = err.message or ''
    if _MODEL_WORD.search(text) or _FUNCTION_NOT_FOUND.search(text):
        return True
    for model in models:
        for name in {model, model.rsplit('@', 1)[0]} if model else ():
            if name and re.search(r'(?<![\w./:-])' + re.escape(name) + r'(?![\w./:-])', text):
                return True
    return False


def broker_code(err: UpstreamError) -> tuple[str, int]:
    """(broker error code, HTTP status) for an UpstreamError."""
    if err.kind == 'auth':
        return 'upstream_auth_failed', 502
    if is_passthrough(err):
        if err.status in GONE_STATUSES:
            about_model = err.provider_code in MODEL_NOT_FOUND_CODES or (err.status == 404 and names_model(err))
            return ('model_not_found' if about_model else 'not_found'), 404
        if err.status == 413:
            return 'request_too_large', 413
        return 'invalid_request', err.status
    if err.kind == 'timeout' or err.status == 408:
        return 'upstream_timeout', 504
    if err.kind == 'rate_limited' or (err.status == 429 and err.kind != 'server_error'):  # not a 429 billing stop
        return 'upstream_rate_limited', 429
    if err.kind in ('server_error', 'network'):
        return 'upstream_unavailable', 503
    if 300 <= err.status < 400:
        return 'upstream_redirect_refused', 502
    return 'upstream_bad_response', 502


def model_unavailable(err: UpstreamError, model: str | None = None, *, upstream_model: str | None = None) -> UpstreamError:
    """A provider 410 Gone, or a 404 that is about the model (names_model; model and upstream_model are the ids it may
    name), for a call that names a model, as 404 model_not_found (kind client_error: a neutral breaker outcome, never a
    fallback trigger, never a credential mark). The message names the model when given and says the provider reports it
    unavailable (404) or retired (410 Gone), followed by the provider's own scrubbed message. Any other error (a 404 about
    a voice, a file, a response id, or a bare 404; a 404 or 410 whose provider code names another resource,
    names_other_resource) is returned unchanged."""
    if err.kind != 'client_error' or err.status not in GONE_STATUSES:
        return err
    if names_other_resource(err.provider_code):  # a voice, file, response ... code: not about the model, even on 410
        return err
    if err.status == 404 and not names_model(err, model, upstream_model):
        return err
    what = f'model {model!r}' if model else 'the requested model'
    if err.status == 410:
        text = f'The provider reports {what} is retired (HTTP 410 Gone)'
    else:
        text = f'The provider reports {what} is not available (HTTP 404)'
    detail = err.message if err.message and err.message != f'Upstream answered {err.status}' else ''
    return UpstreamError(kind='client_error', status=err.status, provider_code='model_not_found',
                         message=_text(f'{text}: {detail}' if detail else text), retry_after=None,
                         request_bytes_sent=err.request_bytes_sent)


def model_failure(failure: 'UpstreamFailure', model: str | None = None, *,
                  upstream_model: str | None = None) -> 'UpstreamFailure':
    """The UpstreamFailure for model_unavailable(failure.upstream, model, upstream_model=...); the same object when
    nothing changes."""
    mapped = model_unavailable(failure.upstream, model, upstream_model=upstream_model)
    return failure if mapped is failure.upstream else UpstreamFailure(mapped)


def to_broker_error(err: UpstreamError, inbound_protocol: str) -> BrokerError:
    """The BrokerError for err (origin upstream; Retry-After kept). inbound_protocol selects the envelope at render time."""
    if inbound_protocol not in ('openai', 'anthropic', 'gemini', 'elevenlabs', 'ollama', 'responses'):
        raise ValueError(f'Unknown protocol: {inbound_protocol!r}')
    return UpstreamFailure(err)


def passthrough_4xx(err: UpstreamError, inbound_protocol: str):
    """A deterministic 400/403/404/413/422 response (a 410 answers 404) in the inbound envelope (scrubbed message; for
    OpenAI-shaped inbound clients the provider's own error code and error.param, and on the generic invalid_request
    passthrough its error.type, when they are plain identifiers, are kept, so an SDK's e.code / e.type / e.param read the
    provider's values)."""
    from starlette.responses import JSONResponse
    if not is_passthrough(err):
        raise ValueError('passthrough_4xx takes a 400, 403, 404, 410, 413 or 422 client error')
    code, status = broker_code(err)
    protocol = 'openai' if inbound_protocol == 'responses' else inbound_protocol
    document = envelope_body(protocol, status, code, err.message)
    if protocol == 'openai' and safe_code(err.provider_code):
        document['error']['code'] = err.provider_code
    if protocol == 'openai' and err.provider_type and code == 'invalid_request':
        # the provider's type on the generic passthrough (400, 403, 422); not_found, model_not_found and
        # request_too_large keep the Titanos code as type, which says more than invalid_request_error
        document['error']['type'] = err.provider_type
    if protocol == 'openai' and err.provider_param:
        document['error']['param'] = err.provider_param
    return JSONResponse(document, status_code=status, headers=error_headers(code, origin='upstream'))


__all__ = ['ADAPTERS', 'ELEVENLABS_KEY_REJECTED', 'ELEVENLABS_QUOTA_CODES', 'GENERIC_NOT_FOUND_CODES', 'GONE_STATUSES', 'MESSAGE_MAX', 'MODEL_NOT_FOUND_CODES',
           'OPENAI_QUOTA_CODES', 'PASSTHROUGH_STATUSES', 'UpstreamFailure', 'broker_code', 'classify',
           'classify_stream_error', 'openai_error_details',
           'credential_failure', 'elevenlabs_code', 'is_passthrough', 'key_rejected', 'model_failure',
           'model_unavailable', 'names_model', 'names_other_resource', 'network_error', 'passthrough_4xx',
           'provider_fields', 'retry_after', 'safe_code', 'scope_refusal', 'to_broker_error']
