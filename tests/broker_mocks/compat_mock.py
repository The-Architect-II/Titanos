"""Loopback mocks for the OpenAI-compatible family rows beyond the reference OpenAI row: one OpenAIMock subclass whose
variant selects the provider's path prefix, its own fixtures (tests/fixtures/providers/<dir>/) and the wire behaviour its
public reference documents. Every OpenAIMock surface stays available under the variant's prefix (chat, completions,
responses, embeddings, audio, images, moderations, models), with these provider specifics:

- groq (/openai/v1): Responses refuses store=true and previous_response_id (Groq: "only false or null"); audio/speech
  takes response_format flac|mp3|mulaw|ogg|wav (default mp3) and answers binary audio; Files upload (native lane).
- together (/v1): images/generations takes response_format base64|url only (b64_json is a 400) and answers JPEG unless
  output_format=png; rerank answers {object: rerank, results: [{index, relevance_score, document}], usage}; audio/speech
  answers binary audio; audio/transcriptions answers {text}.
- deepseek (no prefix): chat answers reasoning_content; a request with tools whose history carries an assistant turn
  without reasoning_content is a 400 (thinking mode requires it back).
- perplexity (no prefix): chat answers citations and search_results; POST /search (native lane).
- fireworks (/inference/v1): rerank answers {object: list, data: [{index, relevance_score, document}], usage}.
- nvidia (/v1, integrate): chat may answer 202 Accepted with NVCF-REQID (mock.respond_with('openai.chat',
  'nvcf_accepted')).
- nvidia_ai (/v1, the second origin): POST /retrieval/<org>/<name>/reranking in the NIM ranking shape (query.text,
  passages[].text -> rankings [{index, logit}], ordered by logit).
- mistral (/v1): audio/transcriptions answers the Voxtral shape (text, language, segments, usage.prompt_audio_seconds).
- xai (/v1): reasoning models (grok-4*, grok-3-mini*) answer 400 for presencePenalty, frequencyPenalty and stop.
- openrouter (/api/v1): embeddings answer usage.cost.
Auth: Authorization: Bearer <SENTINELS[variant]> (nvidia_ai uses the nvidia sentinel)."""
from __future__ import annotations

import fnmatch

from .fixtures import FIXTURES_DIR, SENTINELS
from .openai_mock import OpenAIMock
from .server import MockRequest, Reply, json_reply

PREFIXES = {'groq': '/openai/v1', 'together': '/v1', 'deepseek': '', 'perplexity': '', 'fireworks': '/inference/v1',
            'nvidia': '/v1', 'nvidia_ai': '/v1', 'mistral': '/v1', 'xai': '/v1', 'openrouter': '/api/v1',
            'cerebras': '/v1'}
FIXTURE_DIRS = {'nvidia_ai': 'nvidia'}
GROQ_SPEECH_FORMATS = ('flac', 'mp3', 'mulaw', 'ogg', 'wav')
XAI_REASONING = ('grok-4*', 'grok-3-mini*')
XAI_UNSUPPORTED = {'presence_penalty': 'presencePenalty', 'frequency_penalty': 'frequencyPenalty', 'stop': 'stop'}


class CompatMock(OpenAIMock):
    VARIANTS = tuple(PREFIXES)

    def __init__(self, variant: str, **kw):
        if variant not in PREFIXES:
            raise ValueError(f'variant must be one of {tuple(PREFIXES)}')
        if variant == 'nvidia_ai' and 'credential' not in kw:
            kw['credential'] = SENTINELS['nvidia']
        self.dir = FIXTURE_DIRS.get(variant, variant)
        super().__init__(variant=variant, prefix=PREFIXES[variant], **kw)

    # fixtures: the variant's own directory first, then the OpenAI reference fixtures
    def fixture_provider_for(self, case: str) -> str:
        if (FIXTURES_DIR / self.dir / (case + '.jsonl')).is_file():
            return self.dir
        return 'openai'

    def _register(self) -> None:
        p = self.PREFIX
        own = {
            'groq': [('POST', p + r'/responses', self.groq_responses, 'openai.responses'),
                     ('POST', p + r'/audio/speech', self.groq_speech, 'openai.audio.speech'),
                     ('POST', p + r'/files', self.files_create, 'native')],
            'together': [('POST', p + r'/images/generations', self.together_images, 'openai.images.generations'),
                         ('POST', p + r'/rerank', self.rerank, 'openai.rerank'),
                         ('POST', p + r'/audio/speech', self.plain_speech, 'openai.audio.speech'),
                         ('POST', p + r'/audio/transcriptions', self.together_transcriptions,
                          'openai.audio.transcriptions')],
            'deepseek': [('POST', p + r'/chat/completions', self.deepseek_chat, 'openai.chat')],
            'perplexity': [('POST', p + r'/chat/completions', self.perplexity_chat, 'openai.chat'),
                           ('POST', p + r'/search', self.perplexity_search, 'native')],
            'fireworks': [('POST', p + r'/rerank', self.rerank, 'openai.rerank')],
            'nvidia_ai': [('POST', p + r'/retrieval/(?P<org>[^/]+)/(?P<name>[^/]+)/reranking', self.nim_rerank,
                           'openai.rerank')],
            'mistral': [('POST', p + r'/audio/transcriptions', self.mistral_transcriptions,
                         'openai.audio.transcriptions')],
            'xai': [('POST', p + r'/chat/completions', self.xai_chat, 'openai.chat')],
        }.get(self.variant, [])
        for method, regex, handler, surface in own:  # registered first: the first matching route answers
            self.route(method, regex, handler, surface)
        super()._register()

    # groq -------------------------------------------------------------------------------------------------------------
    async def groq_responses(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        self._require(body, 'input')
        if body.get('store') not in (None, False):
            return self.error_reply(400, "'store': only false or null values are currently supported",
                                    param='store')
        if body.get('previous_response_id') is not None:
            return self.error_reply(400, "'previous_response_id' is not supported", param='previous_response_id')
        if body.get('stream'):
            return await self.responses_create(request)
        return self.reply(self._case('openai.responses', 'responses_create'),
                          {'model': model, 'response_id': f'resp_mock{self.next_id():04d}'})

    async def groq_speech(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        self.request_model(request, body)
        self._require(body, 'input')
        self._require(body, 'voice')
        fmt = body.get('response_format', 'mp3')
        if fmt not in GROQ_SPEECH_FORMATS:
            return self.error_reply(400, f"response_format must be one of {', '.join(GROQ_SPEECH_FORMATS)}",
                                    param='response_format')
        return self.reply('speech', {}, variant=fmt if fmt in ('flac', 'mp3', 'wav') else 'mp3')

    async def files_create(self, request: MockRequest) -> Reply:
        form = await self.form_fields(request)
        upload = form.get('file')
        if upload is None:
            return self.error_reply(400, "Missing required parameter: 'file'.", param='file')
        return self.reply('files_create', {'file_id': f'file_mock{self.next_id():04d}',
                                           'filename': getattr(upload, 'filename', None) or 'upload.jsonl',
                                           'purpose': str(form.get('purpose') or 'batch')})

    # together ---------------------------------------------------------------------------------------------------------
    async def together_images(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        self._require(body, 'prompt')
        fmt = body.get('response_format')
        if fmt not in (None, 'base64', 'url'):
            return self.error_reply(400, "Input validation error: 'response_format' must be one of base64, url",
                                    param='response_format')
        if fmt == 'url' or fmt is None:
            case = 'images_url'
        else:
            case = 'images_b64_png' if body.get('output_format') == 'png' else 'images_b64_jpeg'
        return self.reply(case, {'model': model})

    async def plain_speech(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        self.request_model(request, body)
        self._require(body, 'input')
        return self.reply('speech', {}, variant='mp3')

    async def together_transcriptions(self, request: MockRequest) -> Reply:
        form = await self.form_fields(request)
        if form.get('file') is None:
            return self.error_reply(400, "Missing required parameter: 'file'.", param='file')
        return self.reply('transcriptions_json', {'model': str(form.get('model') or '')})

    # rerank (Together and Fireworks shapes) -------------------------------------------------------------------------------
    async def rerank(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        self._require(body, 'query')
        self._require(body, 'documents')
        return self.reply(self._case('openai.rerank', 'rerank'), {'model': model})

    # NIM ranking (the NVIDIA API catalog's second origin) ------------------------------------------------------------------
    async def nim_rerank(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        query = body.get('query')
        passages = body.get('passages')
        if not isinstance(query, dict) or not isinstance(query.get('text'), str):
            return self.error_reply(422, "body.query.text: field required")
        if not isinstance(passages, list) or not passages or \
                not all(isinstance(p, dict) and isinstance(p.get('text'), str) for p in passages):
            return self.error_reply(422, "body.passages: a non-empty list of {text} objects is required")
        expected = f"{request.path_params['org']}/{request.path_params['name']}"
        if str(body.get('model') or '').replace('.', '_') != expected:
            return self.error_reply(422, 'body.model does not name the model of the URL')
        # deterministic logits: shorter passages that share words with the query rank higher
        words = set(query['text'].lower().split())
        scored = [(round(len(words & set(p['text'].lower().split())) * 2.5 - len(p['text']) / 100.0, 4), i)
                  for i, p in enumerate(passages)]
        rankings = [{'index': i, 'logit': s} for s, i in sorted(scored, key=lambda item: (-item[0], item[1]))]
        return self.reply('rerank_nim', {'rankings': rankings})

    # deepseek ---------------------------------------------------------------------------------------------------------
    async def deepseek_chat(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        self._require(body, 'messages')
        if body.get('tools'):
            for message in body.get('messages') or []:
                if isinstance(message, dict) and message.get('role') == 'assistant' and \
                        not message.get('reasoning_content'):
                    return self.reply('error_missing_reasoning_content', {})
        if body.get('stream'):
            return await self.chat(request)
        return self.reply(self._case('openai.chat', 'chat_reasoning'), {'model': model})

    # perplexity -------------------------------------------------------------------------------------------------------
    async def perplexity_chat(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        self._require(body, 'messages')
        if body.get('stream'):
            return await self.chat(request)
        return self.reply(self._case('openai.chat', 'chat_citations'), {'model': model})

    async def perplexity_search(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        self._require(body, 'query')
        return self.reply('search', {})

    # mistral ----------------------------------------------------------------------------------------------------------
    async def mistral_transcriptions(self, request: MockRequest) -> Reply:
        form = await self.form_fields(request)
        if form.get('file') is None and not form.get('file_url') and not form.get('file_id'):
            return self.error_reply(422, 'one of file, file_url or file_id is required')
        return self.reply('transcriptions_voxtral', {'model': str(form.get('model') or '')})

    # xai --------------------------------------------------------------------------------------------------------------
    async def xai_chat(self, request: MockRequest) -> Reply:
        body = self.json_body(request)
        model = self.request_model(request, body)
        if any(fnmatch.fnmatchcase(model, glob) for glob in XAI_REASONING) and 'non-reasoning' not in model:
            for field, spelled in XAI_UNSUPPORTED.items():
                if field in body:
                    return json_reply({'code': 'Client specified an invalid argument',
                                       'error': f'Model {model} does not support parameter {spelled}.'}, 400)
        return await self.chat(request)


__all__ = ['CompatMock', 'PREFIXES']
