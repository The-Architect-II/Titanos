"""Anthropic Messages loopback mock.

Routes (api_base = base_url): POST /v1/messages (stream and non-stream; tools; thinking), POST /v1/messages/count_tokens,
GET /v1/models, GET /v1/models/{id}; the Files API (POST /v1/files multipart with optional expires_in_seconds, GET
/v1/files in the GA {data, next_page} shape or, with anthropic-beta files-api-2025-04-14, the {data, has_more, first_id,
last_id} shape, GET /v1/files/{id}, GET /v1/files/{id}/content for downloadable files only, DELETE /v1/files/{id}) and
Message Batches (POST /v1/messages/batches, GET /v1/messages/batches, GET|DELETE /v1/messages/batches/{id}, POST
.../cancel, GET .../results as JSONL). Auth: x-api-key: <SENTINELS['anthropic']> plus an anthropic-version header;
otherwise the native {"type": "error", "error": {"type": "authentication_error" | "invalid_request_error", ...}} envelope.
Every answer echoes the inbound anthropic-beta header as x-mock-anthropic-beta (the passthrough fidelity assertion) and
records it in the hit. A request without max_tokens is refused with 400 invalid_request_error 'max_tokens: Field
required'.

Model rules (strict_models, on by default): a Messages, count_tokens or batch request is refused with 400
invalid_request_error (fixture error_400_model_rule) when its fields are not accepted by the named model, as the public
model and migration documentation states: thinking {type: enabled, budget_tokens} on Opus 4.7 and later, Sonnet 5, 5.5
and Fable; thinking {type: disabled} where thinking is always on (Opus 5.5, Sonnet 5.5, Fable); temperature, top_p or
top_k on Opus 4.7 and later, Sonnet 5 and Fable (Sonnet 5.5: non-default values); temperature together with top_p on
the 4.5 family and Opus 4.1; tool_choice any / tool on Opus 5.5, Sonnet 5.5 and Fable 5.1, and with thinking enabled on
any model; output_config.effort on models without effort (and xhigh / max where the model lacks them); a budget not
below max_tokens or under 1024; and a budget-mode tool loop whose last assistant turn does not start with thinking.
Model ids no rule names (test aliases such as 'a') are accepted as before.

The SSE sequence replays tests/fixtures/providers/anthropic/messages_stream*.jsonl: message_start, content_block_start
(text, tool_use, thinking), ping, content_block_delta (text_delta, input_json_delta, thinking_delta, signature_delta),
content_block_stop, message_delta, message_stop; messages_stream_error_first is a 200 stream whose first event is error
(overloaded_error), also produced for any stream by the error_first_event fault. A non-stream tool turn of a model whose
thinking display defaults to omitted answers messages_thinking_omitted (an empty signed thinking block).

Batches end at their next read (retrieve, list or results); results_url is this mock's own origin, so a broker must
rewrite it before a client follows it. add_generated_file() creates a downloadable file (a code-execution output).
"""
from __future__ import annotations

import json
import re

from .fixtures import dumps, load
from .server import BaseMock, MockRequest, Reply, json_reply, refuse

_ERROR_TYPES = {400: 'invalid_request_error', 401: 'authentication_error', 402: 'permission_error', 403: 'permission_error',
                404: 'not_found_error', 413: 'request_too_large', 429: 'rate_limit_error', 500: 'api_error',
                502: 'api_error', 503: 'overloaded_error', 529: 'overloaded_error'}
FILES_BETA = 'files-api-2025-04-14'
_CUSTOM_ID = re.compile(r'[A-Za-z0-9_-]{1,64}')

# Model families by id prefix (transcribed from the public model and migration documentation; see the docstring).
_ALWAYS_THINKING = ('claude-fable-5', 'claude-mythos', 'claude-opus-5-5', 'claude-sonnet-5-5')
_NO_FORCED_TOOL = ('claude-fable-5-1', 'claude-mythos-5-1', 'claude-opus-5-5', 'claude-sonnet-5-5')
_ADAPTIVE_ONLY = ('claude-fable-5', 'claude-mythos', 'claude-opus-5', 'claude-sonnet-5', 'claude-opus-4-7',
                  'claude-opus-4-8')
_DEFAULT_SAMPLING_ONLY = ('claude-sonnet-5-5',)
_EXCLUSIVE_SAMPLING = ('claude-sonnet-4-5', 'claude-haiku-4-5', 'claude-opus-4-1', 'claude-opus-4-5')
_NO_EFFORT = ('claude-sonnet-4-5', 'claude-haiku-4-5', 'claude-opus-4-1', 'claude-opus-4-0', 'claude-opus-4-2025',
              'claude-sonnet-4-0', 'claude-sonnet-4-2025', 'claude-3')
_EFFORTS_BY_FAMILY = (('claude-opus-4-5', ('low', 'medium', 'high')),
                      ('claude-opus-4-6', ('low', 'medium', 'high', 'max')),
                      ('claude-sonnet-4-6', ('low', 'medium', 'high', 'max')))
_ALL_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')


def _family(model: str, prefixes) -> bool:
    """Whether the model id starts with one of the family prefixes (a dated id matches its alias family)."""
    return any(model.startswith(p) for p in prefixes)


def _known(model: str) -> bool:
    return model.startswith('claude-')


def model_rule_error(body: dict) -> str | None:
    """The provider's refusal message for a Messages body the named model does not accept (None: accepted)."""
    model = str(body.get('model') or '')
    if not _known(model):
        return None
    thinking = body.get('thinking') if isinstance(body.get('thinking'), dict) else {}
    kind = thinking.get('type')
    if kind == 'enabled':
        if _family(model, _ADAPTIVE_ONLY):
            return 'thinking.type.enabled is not supported for this model. Use thinking.type.adaptive and effort.'
        budget = thinking.get('budget_tokens')
        if not isinstance(budget, int) or budget < 1024:
            return 'thinking.budget_tokens: Input should be greater than or equal to 1024'
        if isinstance(body.get('max_tokens'), int) and budget >= body['max_tokens']:
            return '`max_tokens` must be greater than `thinking.budget_tokens`.'
    if kind == 'disabled' and _family(model, _ALWAYS_THINKING):
        return 'thinking.type.disabled is not supported for this model.'
    for name in ('temperature', 'top_p', 'top_k'):
        if name not in body:
            continue
        if _family(model, _DEFAULT_SAMPLING_ONLY):
            if body[name] != {'temperature': 1, 'top_p': 1}.get(name):
                return f'{name}: only the default value is supported for this model.'
        elif _family(model, _ADAPTIVE_ONLY):
            return f'{name}: not supported for this model.'
    if _family(model, _EXCLUSIVE_SAMPLING) and 'temperature' in body and 'top_p' in body:
        return '`temperature` and `top_p` cannot both be specified for this model. Please use only one.'
    choice = body.get('tool_choice') if isinstance(body.get('tool_choice'), dict) else {}
    if choice.get('type') in ('any', 'tool'):
        if _family(model, _NO_FORCED_TOOL):
            return 'tool_choice: type "tool" and "any" are not supported for this model.'
        if kind in ('enabled', 'adaptive'):
            return 'Thinking may not be enabled when tool_choice forces tool use.'
    config = body.get('output_config') if isinstance(body.get('output_config'), dict) else {}
    effort = config.get('effort')
    if effort is not None:
        if _family(model, _NO_EFFORT):
            return 'output_config.effort: not supported for this model.'
        allowed = next((levels for prefix, levels in _EFFORTS_BY_FAMILY if _family(model, (prefix,))), _ALL_EFFORTS)
        if effort not in allowed:
            return f'output_config.effort: {effort!r} is not supported for this model.'
    if kind == 'enabled':
        for message in reversed(body.get('messages') or []):
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                continue
            blocks = message.get('content') if isinstance(message.get('content'), list) else []
            if any(isinstance(b, dict) and b.get('type') == 'tool_use' for b in blocks):
                first = blocks[0] if blocks and isinstance(blocks[0], dict) else {}
                if first.get('type') not in ('thinking', 'redacted_thinking'):
                    return ('messages: Expected `thinking` or `redacted_thinking`, but found `tool_use`. When thinking '
                            'is enabled, a final assistant message must start with a thinking block.')
            break
    return None


def count_input_tokens(body: dict) -> int:
    """The mock's deterministic token estimate: one token per 4 characters of the compact JSON of system, messages and tools."""
    text = dumps({'system': body.get('system'), 'messages': body.get('messages'), 'tools': body.get('tools')})
    return max(1, len(text) // 4)


class AnthropicMock(BaseMock):
    PROVIDER = 'anthropic'
    PREFIX = ''

    def __init__(self, *, credential: str | None = None, models: list[str] | None = None, strict_models: bool = True):
        super().__init__(credential=credential)
        self.models = models
        self.strict_models = strict_models
        self.overrides: dict[str, list[str]] = {}
        self.files: dict[str, dict] = {}
        self.batches: dict[str, dict] = {}
        self._file_counter = 0
        self._batch_counter = 0
        self.route('POST', r'/v1/messages', self.messages, 'anthropic.messages')
        self.route('POST', r'/v1/messages/count_tokens', self.count_tokens, 'anthropic.count_tokens')
        self.route('GET', r'/v1/models', self.models_list, 'anthropic.models')
        self.route('GET', r'/v1/models/(?P<model>[^/]+)', self.models_get, 'anthropic.models')
        self.route('POST', r'/v1/files', self.file_upload, 'anthropic.files')
        self.route('GET', r'/v1/files', self.file_list, 'anthropic.files')
        self.route('GET', r'/v1/files/(?P<file_id>[^/]+)/content', self.file_content, 'anthropic.files')
        self.route('GET', r'/v1/files/(?P<file_id>[^/]+)', self.file_get, 'anthropic.files')
        self.route('DELETE', r'/v1/files/(?P<file_id>[^/]+)', self.file_delete, 'anthropic.files')
        self.route('POST', r'/v1/messages/batches', self.batch_create, 'anthropic.batches')
        self.route('GET', r'/v1/messages/batches', self.batch_list, 'anthropic.batches')
        self.route('GET', r'/v1/messages/batches/(?P<batch_id>[^/]+)/results', self.batch_results, 'anthropic.batches')
        self.route('POST', r'/v1/messages/batches/(?P<batch_id>[^/]+)/cancel', self.batch_cancel, 'anthropic.batches')
        self.route('GET', r'/v1/messages/batches/(?P<batch_id>[^/]+)', self.batch_get, 'anthropic.batches')
        self.route('DELETE', r'/v1/messages/batches/(?P<batch_id>[^/]+)', self.batch_delete, 'anthropic.batches')

    def reset(self) -> None:
        super().reset()
        self.overrides.clear()
        self.files.clear()
        self.batches.clear()
        self._file_counter = 0
        self._batch_counter = 0

    def respond_with(self, surface: str, case: str) -> None:
        self.overrides.setdefault(surface, []).append(case)

    def _case(self, surface: str, default: str) -> str:
        queue = self.overrides.get(surface)
        return queue.pop(0) if queue else default

    def _beta_headers(self, request: MockRequest) -> list:
        beta = request.headers.get('anthropic-beta')
        return [('x-mock-anthropic-beta', beta)] if beta is not None else []

    def error_reply(self, status: int, message: str, code: str | None = None) -> Reply:
        return json_reply({'type': 'error', 'error': {'type': code or _ERROR_TYPES.get(status, 'api_error'),
                                                      'message': message},
                           'request_id': 'req_mock_error'}, status, [('request-id', 'req_mock_error')])

    def auth_failure(self, request: MockRequest) -> Reply | None:
        token = request.headers.get('x-api-key')
        if token is None:
            return self.error_reply(401, 'x-api-key header is required', 'authentication_error')
        if token.strip() != self.credential:
            return self.error_reply(401, 'invalid x-api-key', 'authentication_error')
        if not request.headers.get('anthropic-version'):
            return self.error_reply(400, 'anthropic-version: header is required', 'invalid_request_error')
        return None

    def stream_error_frame(self, message: str = 'Overloaded') -> bytes:
        payload = {'type': 'error', 'error': {'type': 'overloaded_error', 'message': message}}
        return ('event: error\ndata: ' + dumps(payload) + '\n\n').encode('utf-8')

    def malformed_frame(self, stream_kind: str) -> bytes:
        return b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":\n\n'

    def _body(self, request: MockRequest) -> dict:
        try:
            body = json.loads(request.body.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            refuse(self.error_reply(400, 'Request body is not valid JSON'))
        if not isinstance(body, dict):
            refuse(self.error_reply(400, 'Request body must be a JSON object'))
        return body

    def _reply(self, case: str, variables: dict, request: MockRequest) -> Reply:
        return self.fixture_reply(case, variables, extra_headers=self._beta_headers(request))

    def _rule_reply(self, body: dict, request: MockRequest) -> Reply | None:
        message = model_rule_error(body) if self.strict_models else None
        return self._reply('error_400_model_rule', {'message': message}, request) if message else None

    async def messages(self, request: MockRequest) -> Reply:
        body = self._body(request)
        for field in ('model', 'messages', 'max_tokens'):
            if body.get(field) in (None, ''):
                return self.error_reply(400, f'{field}: Field required')
        refused = self._rule_reply(body, request)
        if refused is not None:
            return refused
        stream = bool(body.get('stream'))
        tools = bool(body.get('tools')) and (body.get('tool_choice') or {}).get('type') != 'none'
        thinking = (body.get('thinking') or {}).get('type') in ('enabled', 'adaptive')
        model = str(body.get('model'))
        if tools and not stream and _family(model, _ADAPTIVE_ONLY) and \
                (body.get('thinking') or {}).get('display') != 'summarized':
            default = 'messages_thinking_omitted'
        else:
            suffix = '_thinking' if thinking else '_tools' if tools else ''
            default = ('messages_stream' if stream else 'messages') + suffix
        return self._reply(self._case('anthropic.messages', default), {'model': body['model']}, request)

    async def count_tokens(self, request: MockRequest) -> Reply:
        body = self._body(request)
        for field in ('model', 'messages'):
            if body.get(field) in (None, ''):
                return self.error_reply(400, f'{field}: Field required')
        refused = self._rule_reply(dict(body, max_tokens=body.get('max_tokens', 1 << 20)), request)
        if refused is not None:
            return refused
        return self._reply('count_tokens', {'input_tokens': count_input_tokens(body)}, request)

    def _catalog(self) -> dict:
        out = load('anthropic', 'models').body({})
        if self.models is not None:
            template = out['data'][0]
            out['data'] = [dict(template, id=m, display_name=m) for m in self.models]
            out['first_id'] = out['data'][0]['id'] if out['data'] else None
            out['last_id'] = out['data'][-1]['id'] if out['data'] else None
        return out

    async def models_list(self, request: MockRequest) -> Reply:
        if self.models is None:
            return self._reply('models', {}, request)
        return json_reply(self._catalog(), 200, self._beta_headers(request))

    async def models_get(self, request: MockRequest) -> Reply:
        name = request.path_params['model']
        for item in self._catalog()['data']:
            if item['id'] == name:
                headers = [(k, v) for k, v in self.fixture('models_get').headers.items() if k != 'content-type']
                return json_reply(item, 200, headers + self._beta_headers(request))
        return self.error_reply(404, f'model: {name}')

    # -- Files API -----------------------------------------------------------------------------------------------------
    def _file_vars(self, record: dict) -> dict:
        return {key: record[key] for key in ('file_id', 'filename', 'mime_type', 'size_bytes', 'downloadable',
                                             'expires_at')}

    def _file_object(self, record: dict) -> dict:
        return load('anthropic', 'files_get').body(self._file_vars(record))

    def add_generated_file(self, filename: str = 'output.csv', data: bytes = b'x,y\n1,2\n',
                           mime_type: str = 'text/csv') -> str:
        """A downloadable file, as code execution or a skill creates it; returns its id."""
        self._file_counter += 1
        file_id = f'file_mock{self._file_counter:04d}'
        self.files[file_id] = {'file_id': file_id, 'filename': filename, 'mime_type': mime_type,
                               'size_bytes': len(data), 'downloadable': True, 'expires_at': None, 'data': data}
        return file_id

    async def file_upload(self, request: MockRequest) -> Reply:
        form = await request.form()
        upload = form.get('file')
        if upload is None or isinstance(upload, str):
            return self.error_reply(400, 'file: Field required')
        data = await upload.read()
        expires_at = None
        if form.get('expires_in_seconds') is not None:
            try:
                seconds = int(form.get('expires_in_seconds'))
            except (TypeError, ValueError):
                seconds = -1
            if not 3600 <= seconds <= 7_776_000:
                return self.error_reply(400, 'expires_in_seconds: must be between 3600 and 7776000')
            expires_at = '2026-10-02T01:00:00Z'
        self._file_counter += 1
        file_id = f'file_mock{self._file_counter:04d}'
        record = {'file_id': file_id, 'filename': upload.filename or 'upload.bin',
                  'mime_type': upload.content_type or 'application/octet-stream', 'size_bytes': len(data),
                  'downloadable': False, 'expires_at': expires_at, 'data': data}
        self.files[file_id] = record
        return self._reply('files_upload', self._file_vars(record), request)

    async def file_list(self, request: MockRequest) -> Reply:
        records = [self._file_object(r) for r in sorted(self.files.values(), key=lambda r: r['file_id'], reverse=True)]
        try:
            limit = max(1, min(1000, int(request.q('limit') or 20)))
        except ValueError:
            return self.error_reply(400, 'limit: Input should be a valid integer')
        if FILES_BETA in (request.headers.get('anthropic-beta') or ''):
            if request.q('page') is not None:
                return self.error_reply(400, 'page: not supported with files-api-2025-04-14')
            ids = [r['id'] for r in records]
            if request.q('after_id') in ids:
                records = records[ids.index(request.q('after_id')) + 1:]
            page = records[:limit]
            return self._reply('files_list_beta', {'data': page, 'has_more': len(records) > limit,
                                                   'first_id': page[0]['id'] if page else None,
                                                   'last_id': page[-1]['id'] if page else None}, request)
        if request.q('after_id') is not None or request.q('before_id') is not None:
            return self.error_reply(400, 'after_id and before_id are not supported; use page')
        wanted = [value for name, value in request.query if name == 'ids[]']
        if wanted:
            return self._reply('files_list', {'data': [r for r in records if r['id'] in wanted], 'next_page': None},
                               request)
        start = 0
        cursor = request.q('page')
        if cursor:
            if not cursor.startswith('page_') or not cursor[5:].isdigit():
                return self.error_reply(400, 'page: invalid cursor')
            start = int(cursor[5:])
        page = records[start:start + limit]
        following = f'page_{start + limit}' if start + limit < len(records) else None
        return self._reply('files_list', {'data': page, 'next_page': following}, request)

    def _file(self, request: MockRequest) -> dict:
        file_id = request.path_params['file_id']
        record = self.files.get(file_id)
        if record is None:
            refuse(self._reply('error_404_file', {'file_id': file_id}, request))
        return record

    async def file_get(self, request: MockRequest) -> Reply:
        return self._reply('files_get', self._file_vars(self._file(request)), request)

    async def file_content(self, request: MockRequest) -> Reply:
        record = self._file(request)
        if not record['downloadable']:
            return self._reply('error_400_not_downloadable', {'file_id': record['file_id']}, request)
        data = record['data']
        frames = [data[i:i + 7] for i in range(0, len(data), 7)] or [b'']
        return Reply(200, [('content-type', record['mime_type']), ('request-id', 'req_mock0001'),
                           ('content-disposition', f'attachment; filename="{record["filename"]}"')],
                     frames=frames, stream_kind='binary')

    async def file_delete(self, request: MockRequest) -> Reply:
        record = self._file(request)
        del self.files[record['file_id']]
        return self._reply('files_delete', {'file_id': record['file_id']}, request)

    # -- Message Batches -----------------------------------------------------------------------------------------------
    def _batch_vars(self, batch: dict) -> dict:
        return {key: batch[key] for key in ('batch_id', 'processing_status', 'request_counts', 'ended_at',
                                            'cancel_initiated_at', 'results_url')}

    def _settle(self, batch: dict) -> None:
        """A batch ends at its next read: every request succeeds or errors (a canceled batch: canceled)."""
        if batch['processing_status'] == 'ended':
            return
        counts = {'processing': 0, 'succeeded': 0, 'errored': 0, 'canceled': 0, 'expired': 0}
        for item in batch['requests']:
            item['outcome'] = 'canceled' if batch['processing_status'] == 'canceling' else (
                'errored' if item['error'] else 'succeeded')
            counts[item['outcome']] += 1
        batch.update(processing_status='ended', request_counts=counts, ended_at='2026-10-02T00:05:00Z',
                     results_url=f'{self.base_url}/v1/messages/batches/{batch["batch_id"]}/results')

    def _batch(self, request: MockRequest) -> dict:
        batch_id = request.path_params['batch_id']
        batch = self.batches.get(batch_id)
        if batch is None:
            refuse(self.error_reply(404, f'Message batch {batch_id} not found', 'not_found_error'))
        return batch

    async def batch_create(self, request: MockRequest) -> Reply:
        body = self._body(request)
        requests = body.get('requests')
        if not isinstance(requests, list) or not requests:
            return self.error_reply(400, 'requests: List should have at least 1 item')
        seen, items = set(), []
        for index, item in enumerate(requests):
            custom_id = item.get('custom_id') if isinstance(item, dict) else None
            params = item.get('params') if isinstance(item, dict) else None
            if not isinstance(custom_id, str) or not _CUSTOM_ID.fullmatch(custom_id):
                return self.error_reply(400, f'requests.{index}.custom_id: String should match pattern')
            if custom_id in seen:
                return self.error_reply(400, f'requests.{index}.custom_id: duplicate custom_id {custom_id}')
            if not isinstance(params, dict):
                return self.error_reply(400, f'requests.{index}.params: Field required')
            seen.add(custom_id)
            error = None
            for field in ('model', 'messages', 'max_tokens'):
                if params.get(field) in (None, ''):
                    error = f'{field}: Field required'
            if error is None and 'fallbacks' in params:
                error = 'fallbacks: not supported in the Message Batches API'
            if error is None and self.strict_models:
                error = model_rule_error(params)
            items.append({'custom_id': custom_id, 'model': params.get('model'), 'error': error, 'outcome': None})
        self._batch_counter += 1
        batch_id = f'msgbatch_mock{self._batch_counter:04d}'
        batch = {'batch_id': batch_id, 'processing_status': 'in_progress',
                 'request_counts': {'processing': len(items), 'succeeded': 0, 'errored': 0, 'canceled': 0,
                                    'expired': 0},
                 'ended_at': None, 'cancel_initiated_at': None, 'results_url': None, 'requests': items}
        self.batches[batch_id] = batch
        return self._reply('batch_create', self._batch_vars(batch), request)

    async def batch_get(self, request: MockRequest) -> Reply:
        batch = self._batch(request)
        self._settle(batch)
        return self._reply('batch_get', self._batch_vars(batch), request)

    async def batch_list(self, request: MockRequest) -> Reply:
        batches = sorted(self.batches.values(), key=lambda b: b['batch_id'], reverse=True)
        for batch in batches:
            self._settle(batch)
        objects = [load('anthropic', 'batch_get').body(self._batch_vars(b)) for b in batches]
        try:
            limit = max(1, min(1000, int(request.q('limit') or 20)))
        except ValueError:
            return self.error_reply(400, 'limit: Input should be a valid integer')
        ids = [o['id'] for o in objects]
        if request.q('after_id') in ids:
            objects = objects[ids.index(request.q('after_id')) + 1:]
        page = objects[:limit]
        return self._reply('batch_list', {'data': page, 'has_more': len(objects) > limit,
                                          'first_id': page[0]['id'] if page else None,
                                          'last_id': page[-1]['id'] if page else None}, request)

    async def batch_cancel(self, request: MockRequest) -> Reply:
        batch = self._batch(request)
        if batch['processing_status'] == 'in_progress':
            batch.update(processing_status='canceling', cancel_initiated_at='2026-10-02T00:01:00Z')
        return self._reply('batch_create', self._batch_vars(batch), request)

    async def batch_delete(self, request: MockRequest) -> Reply:
        batch = self._batch(request)
        if batch['processing_status'] != 'ended':
            return self.error_reply(400, f'Message batch {batch["batch_id"]} is still processing; cancel it first')
        del self.batches[batch['batch_id']]
        return self._reply('batch_delete', {'batch_id': batch['batch_id']}, request)

    async def batch_results(self, request: MockRequest) -> Reply:
        batch = self._batch(request)
        self._settle(batch)
        if batch['processing_status'] != 'ended':
            return self._reply('error_400_batch_not_ended', {'batch_id': batch['batch_id']}, request)
        fx = load('anthropic', 'batch_results')
        frames = []
        for item in batch['requests']:
            if item['outcome'] == 'succeeded':
                frames.extend(fx.frames({'custom_id': item['custom_id'], 'model': item['model']}))
            elif item['outcome'] == 'errored':
                line = {'custom_id': item['custom_id'], 'result': {'type': 'errored', 'error': {
                    'type': 'error', 'error': {'type': 'invalid_request_error', 'message': item['error']}}}}
                frames.append((dumps(line) + '\n').encode('utf-8'))
            else:
                frames.append((dumps({'custom_id': item['custom_id'], 'result': {'type': item['outcome']}}) + '\n')
                              .encode('utf-8'))
        return Reply(200, list(fx.headers.items()), frames=frames or [b''], stream_kind='binary')


__all__ = ['AnthropicMock', 'FILES_BETA', 'count_input_tokens', 'model_rule_error']
