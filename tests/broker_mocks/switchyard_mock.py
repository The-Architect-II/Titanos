"""SwitchyardMock: a loopback stand-in for NVIDIA NeMo Switchyard 0.3.0 (switchyard-server; github.com/NVIDIA-NeMo/Switchyard,
Apache-2.0) as a router upstream. Wire shapes were transcribed from a real switchyard-server 0.3.0 run against a loopback
backend (ids and values here are synthetic):

- one origin, no /v1 in the base_url: POST /v1/chat/completions, /v1/responses and /v1/messages address any route by its
  id (the model field); the answer's model is the backend model the route selected, also sent as the
  x-model-router-selected-model response header (the noop route answers 'OK' and names itself);
- GET /v1/models: {object: list, data: [{id, object: model, type: model, created: 0, owned_by: switchyard, display_name,
  capabilities: {streaming, tool_calling, vision, context_window, supported_inbound_formats}}], models: [], first_id,
  last_id, has_more: false, default_model, model_pool};
- auxiliary: POST /v1/messages/count_tokens ({input_tokens}; 400 invalid_request_error 'route has no Anthropic target for
  token counting' without one), POST /v1/responses/input_tokens and /v1/responses/compact (400 code
  responses_auxiliary_unsupported without an OpenAI Responses target), POST /v1/decision ({selected: {target, model,
  llm_client: {format, base_url}, extra_body}, fallbacks});
- operator: GET /v1/stats, POST /v1/stats/reset, GET /metrics (Prometheus text), GET /health ({status: ok}),
  GET /v1/routing/session-stats (404 unless a routing log is configured);
- errors are rendered in the inbound format: OpenAI {error: {message, type, code}} (404 model_not_found, 400
  unsupported_capability for a route declaring vision = false or tool_calling = false), Anthropic {type: error, error:
  {type, message}} (404 not_found_error);
- streams: chat chunks, then a usage chunk (choices []) whether or not include_usage was asked, then [DONE]; the Anthropic
  event family with event: lines; the Responses event family with sequence_number;
- no inbound authentication by default (credential=None); require_auth=True expects Authorization: Bearer <credential> (an
  authenticating proxy in front of it).
"""
from __future__ import annotations

from .fixtures import dumps
from .server import BaseMock, MockRequest, Reply, json_reply

FORMATS = ['openai-chat-completions', 'openai-responses', 'anthropic-messages']

# route id -> (selected backend model, tool_calling, vision, context_window, has an Anthropic target, has a Responses target)
DEFAULT_ROUTES = {
    'switchyard': (None, None, None, None, False, False),  # type = "noop"
    'switchyard/general': ('mock-weak', True, False, 32768, False, False),
    'switchyard/claude': ('mock-strong', None, None, None, True, False),
    'switchyard/codex': ('mock-codex', True, True, 400000, False, True),
}


def _sse(event: str | None, data) -> bytes:
    text = data if isinstance(data, str) else dumps(data)
    return ((f'event: {event}\n' if event else '') + f'data: {text}\n\n').encode('utf-8')


class SwitchyardMock(BaseMock):
    PROVIDER = 'switchyard'
    PREFIX = ''

    def __init__(self, *, credential: str | None = None, require_auth: bool = False, routes: dict | None = None,
                 routing_log: bool = False):
        super().__init__(credential=credential)
        self.require_auth = require_auth
        self.routes_table = dict(DEFAULT_ROUTES if routes is None else routes)
        self.routing_log = routing_log
        self.text = 'Hi'
        self.reset_count = 0
        r = self.route
        r('GET', r'/v1/models', self.models, 'switchyard.models')
        r('POST', r'/v1/chat/completions', self.chat, 'openai.chat')
        r('POST', r'/v1/responses', self.responses, 'openai.responses')
        r('POST', r'/v1/messages', self.messages, 'anthropic.messages')
        r('POST', r'/v1/messages/count_tokens', self.count_tokens, 'anthropic.count_tokens')
        r('POST', r'/v1/responses/input_tokens', self.input_tokens, 'switchyard.input_tokens')
        r('POST', r'/v1/responses/compact', self.compact, 'switchyard.compact')
        r('POST', r'/v1/decision', self.decision, 'switchyard.decision')
        r('GET', r'/v1/stats', self.stats, 'switchyard.stats')
        r('POST', r'/v1/stats/reset', self.stats_reset, 'switchyard.stats_reset')
        r('GET', r'/v1/routing/session-stats', self.session_stats, 'switchyard.session_stats')
        r('GET', r'/metrics', self.metrics, 'switchyard.metrics')
        r('GET', r'/health', self.health, 'switchyard.health', auth=False)

    # provider hooks -----------------------------------------------------------------------------------------------------
    def auth_failure(self, request: MockRequest) -> Reply | None:
        if not self.require_auth:
            return None
        token = self.presented_token(request, 'authorization')
        if token != self.credential:
            return self.error_reply(401, 'missing or invalid authorization', 'invalid_api_key')
        return None

    def error_reply(self, status: int, message: str, code: str | None = None, *, anthropic: bool = False,
                    type_: str | None = None) -> Reply:
        if anthropic:
            kind = type_ or {404: 'not_found_error', 401: 'authentication_error', 429: 'rate_limit_error',
                             503: 'overloaded_error'}.get(status, 'invalid_request_error' if status < 500 else 'api_error')
            return json_reply({'type': 'error', 'error': {'type': kind, 'message': message}}, status)
        kind = type_ or code or ('invalid_request_error' if status < 500 else 'upstream_error')
        return json_reply({'error': {'message': message, 'type': kind, 'code': code or kind}}, status)

    def stream_error_frame(self, message: str = 'upstream stream failed') -> bytes:
        return _sse(None, {'error': {'message': message, 'type': 'upstream_error', 'code': 'upstream_error'}})

    # helpers ----------------------------------------------------------------------------------------------------------
    def served(self, route_id: str) -> str:
        return self.routes_table[route_id][0] or route_id

    def lookup(self, request: MockRequest, *, anthropic: bool = False):
        body = request.json() or {}
        model = body.get('model')
        if model not in self.routes_table:
            message = f'No route registered for model {model}'
            return body, None, self.error_reply(404, message, 'model_not_found', anthropic=anthropic,
                                                type_='not_found_error' if anthropic else 'model_not_found')
        _, tools, vision, _, _, _ = self.routes_table[model]
        if vision is False and _has_image(body):
            message = f'route {model} declares vision = false; remove the unsupported input or select another route'
            return body, None, self.error_reply(400, message, 'unsupported_capability', anthropic=anthropic,
                                                type_='invalid_request_error')
        if tools is False and body.get('tools'):
            message = f'route {model} declares tool_calling = false; remove the tools or select another route'
            return body, None, self.error_reply(400, message, 'unsupported_capability', anthropic=anthropic,
                                                type_='invalid_request_error')
        return body, model, None

    def usage(self, route_id: str) -> tuple[int, int]:
        return (0, 0) if self.routes_table[route_id][0] is None else (5, 1)

    def text_for(self, route_id: str) -> str:
        return 'OK' if self.routes_table[route_id][0] is None else self.text

    def selected(self, route_id: str) -> list:
        return [('x-model-router-selected-model', self.served(route_id))]

    # inference ----------------------------------------------------------------------------------------------------------
    async def models(self, request: MockRequest) -> Reply:
        ids = sorted(self.routes_table)
        data = []
        for route_id in ids:
            _, tools, vision, context, _, _ = self.routes_table[route_id]
            data.append({'id': route_id, 'object': 'model', 'type': 'model', 'created': 0, 'owned_by': 'switchyard',
                         'display_name': route_id,
                         'capabilities': {'streaming': True, 'tool_calling': tools, 'vision': vision,
                                          'context_window': context, 'supported_inbound_formats': FORMATS}})
        return json_reply({'object': 'list', 'data': data, 'models': [], 'first_id': ids[0] if ids else None,
                           'last_id': ids[-1] if ids else None, 'has_more': False,
                           'default_model': ids[0] if ids else None, 'model_pool': ids})

    async def chat(self, request: MockRequest) -> Reply:
        body, route_id, error = self.lookup(request)
        if error is not None:
            return error
        model, text = self.served(route_id), self.text_for(route_id)
        prompt, completion = self.usage(route_id)
        ident = 'chatcmpl_mock-' + str(self.next_id())
        usage = {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion}
        if body.get('stream'):
            frames = [_sse(None, {'id': ident, 'object': 'chat.completion.chunk', 'created': 0, 'model': model,
                                  'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text},
                                               'finish_reason': None}]}),
                      _sse(None, {'id': ident, 'object': 'chat.completion.chunk', 'created': 0, 'model': model,
                                  'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}),
                      _sse(None, {'id': ident, 'object': 'chat.completion.chunk', 'created': 0, 'model': model,
                                  'choices': [], 'usage': usage}),
                      _sse(None, '[DONE]')]
            return Reply(200, [('content-type', 'text/event-stream'), ('cache-control', 'no-cache')]
                         + self.selected(route_id), frames=frames, stream_kind='sse')
        return json_reply({'id': ident, 'object': 'chat.completion', 'created': 0, 'model': model,
                           'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                                        'finish_reason': 'stop'}], 'usage': usage}, headers=self.selected(route_id))

    def _response_object(self, ident: str, model: str, text: str, status: str, prompt: int, completion: int) -> dict:
        output = [] if status == 'in_progress' else [
            {'type': 'message', 'id': f'msg_{ident}_0', 'role': 'assistant', 'status': 'completed',
             'content': [{'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []}]}]
        return {'id': 'resp_' + ident, 'object': 'response', 'created_at': 0, 'completed_at': None, 'error': None,
                'incomplete_details': None, 'instructions': None, 'metadata': None, 'model': model, 'output': output,
                'parallel_tool_calls': True, 'frequency_penalty': None, 'presence_penalty': None, 'status': status,
                'temperature': None, 'tool_choice': 'auto', 'tools': [], 'top_p': None,
                'usage': {'input_tokens': prompt, 'output_tokens': completion, 'total_tokens': prompt + completion,
                          'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
                          'output_tokens_details': {'reasoning_tokens': 0}}}

    async def responses(self, request: MockRequest) -> Reply:
        body, route_id, error = self.lookup(request)
        if error is not None:
            return error
        model, text = self.served(route_id), self.text_for(route_id)
        prompt, completion = self.usage(route_id)
        ident = 'mock-' + str(self.next_id())
        if not body.get('stream'):
            return json_reply(self._response_object(ident, model, text, 'completed', prompt, completion),
                              headers=self.selected(route_id))
        item = f'msg_{ident}_0'
        part = {'type': 'output_text', 'text': '', 'annotations': [], 'logprobs': []}
        events = [
            ('response.created', {'response': self._response_object(ident, model, text, 'in_progress', 0, 0)}),
            ('response.output_item.added', {'output_index': 0, 'item': {'type': 'message', 'id': item,
                                                                        'role': 'assistant', 'status': 'in_progress',
                                                                        'content': []}}),
            ('response.content_part.added', {'item_id': item, 'output_index': 0, 'content_index': 0, 'part': part}),
            ('response.output_text.delta', {'item_id': item, 'output_index': 0, 'content_index': 0, 'delta': text,
                                            'logprobs': []}),
            ('response.output_text.done', {'item_id': item, 'output_index': 0, 'content_index': 0, 'text': text,
                                           'logprobs': []}),
            ('response.content_part.done', {'item_id': item, 'output_index': 0, 'content_index': 0,
                                            'part': dict(part, text=text)}),
            ('response.output_item.done', {'output_index': 0, 'item': {
                'type': 'message', 'id': item, 'role': 'assistant', 'status': 'completed',
                'content': [dict(part, text=text)]}}),
            ('response.completed', {'response': self._response_object(ident, model, text, 'completed', prompt,
                                                                      completion)}),
        ]
        frames = [_sse(name, {'type': name, **data, 'sequence_number': n}) for n, (name, data) in enumerate(events)]
        return Reply(200, [('content-type', 'text/event-stream'), ('cache-control', 'no-cache')]
                     + self.selected(route_id), frames=frames, stream_kind='sse')

    async def messages(self, request: MockRequest) -> Reply:
        body, route_id, error = self.lookup(request, anthropic=True)
        if error is not None:
            return error
        model, text = self.served(route_id), self.text_for(route_id)
        prompt, completion = self.usage(route_id)
        ident = 'msg_mock-' + str(self.next_id())
        if not body.get('stream'):
            return json_reply({'id': ident, 'type': 'message', 'role': 'assistant', 'model': model,
                               'content': [{'type': 'text', 'text': text}], 'stop_reason': 'end_turn',
                               'stop_sequence': None, 'usage': {'input_tokens': prompt, 'output_tokens': completion}},
                              headers=self.selected(route_id))
        events = [
            ('message_start', {'message': {'id': ident, 'type': 'message', 'role': 'assistant', 'model': model,
                                           'content': [], 'stop_reason': None, 'stop_sequence': None,
                                           'usage': {'input_tokens': prompt, 'output_tokens': 0}}}),
            ('content_block_start', {'index': 0, 'content_block': {'type': 'text', 'text': ''}}),
            ('content_block_delta', {'index': 0, 'delta': {'type': 'text_delta', 'text': text}}),
            ('content_block_stop', {'index': 0}),
            ('message_delta', {'delta': {'stop_reason': 'end_turn', 'stop_sequence': None, 'stop_details': None},
                               'usage': {'output_tokens': completion}}),
            ('message_stop', {}),
        ]
        frames = [_sse(name, {'type': name, **data}) for name, data in events]
        return Reply(200, [('content-type', 'text/event-stream'), ('cache-control', 'no-cache')]
                     + self.selected(route_id), frames=frames, stream_kind='sse')

    # auxiliary ----------------------------------------------------------------------------------------------------------
    async def count_tokens(self, request: MockRequest) -> Reply:
        body, route_id, error = self.lookup(request, anthropic=True)
        if error is not None:
            return error
        if not self.routes_table[route_id][4]:
            return self.error_reply(400, 'route has no Anthropic target for token counting', anthropic=True)
        return json_reply({'input_tokens': 7})

    def _responses_aux(self, request: MockRequest):
        body, route_id, error = self.lookup(request)
        if error is None and not self.routes_table[route_id][5]:
            error = self.error_reply(400, 'route has no OpenAI Responses target', 'responses_auxiliary_unsupported',
                                     type_='invalid_request_error')
        return body, route_id, error

    async def input_tokens(self, request: MockRequest) -> Reply:
        _, _, error = self._responses_aux(request)
        return error or json_reply({'object': 'response.input_tokens', 'input_tokens': 11})

    async def compact(self, request: MockRequest) -> Reply:
        _, route_id, error = self._responses_aux(request)
        if error is not None:
            return error
        return json_reply({'id': 'resp_cmp_mock-' + str(self.next_id()), 'object': 'response.compaction',
                           'created_at': 0, 'output': [{'type': 'compaction', 'id': 'cmp_mock_0',
                                                        'encrypted_content': 'mock-opaque-compaction'}],
                           'usage': {'input_tokens': 11, 'output_tokens': 3, 'total_tokens': 14,
                                     'input_tokens_details': {'cached_tokens': 0},
                                     'output_tokens_details': {'reasoning_tokens': 0}}},
                          headers=self.selected(route_id))

    async def decision(self, request: MockRequest) -> Reply:
        body = request.json() or {}
        inner = body.get('request') if isinstance(body.get('request'), dict) else {}
        route_id = inner.get('model')
        if route_id not in self.routes_table:
            return self.error_reply(404, f'No route registered for model {route_id}', 'model_not_found')
        served = self.served(route_id)
        return json_reply({'selected': {'target': served, 'model': served,
                                        'llm_client': {'format': 'openai_chat', 'base_url': 'http://127.0.0.1:9/v1'},
                                        'extra_body': {}}, 'fallbacks': []})

    # operator -----------------------------------------------------------------------------------------------------------
    async def stats(self, request: MockRequest) -> Reply:
        calls = sum(1 for hit in self.hits if hit.surface in ('openai.chat', 'openai.responses', 'anthropic.messages'))
        return json_reply({'total_requests': calls, 'total_errors': 0,
                           'total_tokens': {'prompt': 0, 'completion': 0, 'cached': 0, 'cache_creation': 0,
                                            'reasoning': 0, 'total': 0}, 'models': {}})

    async def stats_reset(self, request: MockRequest) -> Reply:
        self.reset_count += 1
        return json_reply({'status': 'reset'})

    async def session_stats(self, request: MockRequest) -> Reply:
        if not self.routing_log:
            return self.error_reply(404, f'Unknown path {request.path}', 'not_found')
        return json_reply({'sessions': []})

    async def metrics(self, request: MockRequest) -> Reply:
        text = ('# TYPE switchyard_build_info gauge\nswitchyard_build_info{version="0.3.0",otel_scope_name="switchyard"} 1\n')
        return Reply(200, [('content-type', 'text/plain; version=0.0.4; charset=utf-8')], text.encode('utf-8'))

    async def health(self, request: MockRequest) -> Reply:
        return json_reply({'status': 'ok'})


def _has_image(body: dict) -> bool:
    for message in body.get('messages') or ():
        content = message.get('content') if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else ():
            if isinstance(part, dict) and part.get('type') in ('image_url', 'image', 'input_image'):
                return True
    return False


__all__ = ['DEFAULT_ROUTES', 'SwitchyardMock']
