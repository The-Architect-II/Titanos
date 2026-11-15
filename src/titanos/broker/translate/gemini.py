"""Gemini (generateContent) codec and the embedContent / batchEmbedContents mapping.

- contents/parts and systemInstruction; roles user and model (tool results travel as user functionResponse parts).
- tools.functionDeclarations and toolConfig.functionCallingConfig: auto / required / none / one function map to
  AUTO / ANY / NONE / ANY with allowedFunctionNames.
- generationConfig: maxOutputTokens, temperature, topP, topK, stopSequences, seed, presencePenalty, frequencyPenalty,
  thinkingConfig.thinkingBudget; json_schema maps to responseMimeType 'application/json' plus responseSchema (the schema is
  reduced to the OpenAPI subset the Schema object accepts); json_object to responseMimeType alone.
- inlineData images; an https image URL is refused (400): Titanos never fetches URLs and fileData needs uploaded file URIs.
- functionCall parts carry no id, so ids are synthesized as call_<index>_<8hex> (deterministic); functionResponse parts are
  matched back by name. A thoughtSignature on a functionCall part travels as a signature-only Thinking just before the
  ToolUse and is put back on the functionCall part on encode.
- Gemini 3 rejects a follow-up turn whose functionCall lost its thoughtSignature. Clients of other protocols usually
  never echo signatures, so for gemini-3* models the first functionCall of a model turn that carries none gets Google's
  documented validator-skip signature (DUMMY_THOUGHT_SIGNATURE; note 'injected:thoughtSignature').
- Schemas: local $ref / $defs / definitions references are inlined before the reduction to the Schema object (recursive
  references are cut to the target's type and noted); const becomes a one-value enum; an enum of non-strings is dropped
  (noted), because the Schema object takes string enums only.
- usageMetadata: promptTokenCount (+ toolUsePromptTokenCount), candidatesTokenCount + thoughtsTokenCount (thinking is
  billed as output; the IR's output_tokens includes reasoning), cachedContentTokenCount, thoughtsTokenCount.
- Embeddings: Gemini embedding answers carry no token counts, so usage is estimated from the input text (4 characters
  per token, estimated=True) so budgets still count them.
- Field spellings: the REST API accepts the proto3 JSON lowerCamelCase names and the original snake_case names
  (generation_config, inline_data, function_call ...); camelize() normalizes a Gemini-inbound body before decoding, and
  output_cap / with_output_cap / with_thinking read and write whichever spelling the body already uses (never both).
- Model families (model_family, thinking_mode): thinking is mapped per model: gemini-3 and later take thinkingLevel
  (LOW / MEDIUM / HIGH; Pro models have no MEDIUM, so medium maps to HIGH there and the change is noted), gemini-2.5 and
  unknown names take thinkingBudget, and the families without thinking (TTS, image output, embedding, Imagen / Veo,
  Live, AQA, gemini-1.x / 2.0, Gemma) never get a thinking config (noted dropped:reasoning_effort when one was asked).
- Lossless schemas: a json_schema whose reduction to the Schema object would drop keys travels as responseJsonSchema
  (and tool parameters as parametersJsonSchema) on gemini-2.5 and later; other models keep the reduced form plus notes.
- Input media: inlineData audio parts decode to IR Audio and IR Audio encodes to inlineData; video, PDF and fileData
  parts have no IR form and are refused on translation (the passthrough forwards them untouched).
- Output media: inlineData images decode to IR Image and audio to IR Audio (non-stream); a chat translation to a TTS
  model is refused before egress (use /v1/audio/speech); executableCode / codeExecutionResult become fenced Text parts
  (noted converted:executableCode); groundingMetadata is noted dropped:groundingMetadata. In a translated stream an image
  or audio part cannot be represented and is noted dropped:image_output / dropped:audio_output.
Streams: each alt=sse data frame is a whole GenerateContentResponse (no event names, no [DONE]). The decoder emits a
functionCall as ToolCallStart + one ToolArgsDelta + BlockStop; the encoder emits one chunk per text delta and whole
functionCall parts, buffering one tool call's argument fragments until its BlockStop (documented exception to the
one-frame rule); invalid buffered arguments end the stream with the Gemini error chunk.
"""
from __future__ import annotations

import fnmatch
import json
import re
import uuid

from ..envelopes import stream_terminal
from ..sse import SSEFrame, gemini_usage, write_json_sse
from ..types import Usage
from .common import (StreamTail, b64, bad_upstream, closes_source, error_code, merge_roles, refuse, request_args,
                     stream_error_event, synth_tool_id, unb64)
from .ir import (Audio, BlockStop, ErrorEvent, Image, IRMessage, IRReasoning, IRRequest, IRResponse, IRTool,
                 IRToolChoice, Start, Stop, Text, TextDelta, Thinking, ThinkingDelta, ToolArgsDelta, ToolCallStart,
                 ToolResult, ToolUse, UsageEvent)

FILTER_REASONS = frozenset({'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'IMAGE_SAFETY',
                            'LANGUAGE'})
STOP_TO_GEMINI = {'end_turn': 'STOP', 'stop_sequence': 'STOP', 'tool_use': 'STOP', 'max_tokens': 'MAX_TOKENS',
                  'content_filter': 'SAFETY'}
SCHEMA_KEYS = frozenset({'type', 'format', 'title', 'description', 'nullable', 'enum', 'maxItems', 'minItems', 'properties',
                         'required', 'minProperties', 'maxProperties', 'minLength', 'maxLength', 'pattern', 'example',
                         'anyOf', 'propertyOrdering', 'default', 'items', 'minimum', 'maximum'})
DUMMY_THOUGHT_SIGNATURE = 'skip_thought_signature_validator'
GEMINI3_GLOBS = ('gemini-3*',)
REF_DEPTH = 8
_GEN_KNOWN = frozenset({'maxOutputTokens', 'temperature', 'topP', 'topK', 'stopSequences', 'seed', 'presencePenalty',
                        'frequencyPenalty', 'responseMimeType', 'responseSchema', 'responseJsonSchema', 'candidateCount',
                        'thinkingConfig'})

# Model families, as fnmatch globs over the lower-cased upstream model id (no 'models/' prefix). Order matters in
# model_family: the first family whose globs match wins.
MODEL_FAMILIES = (
    ('tts', ('*-tts', '*-tts-*', '*tts-preview*')),
    ('live', ('*-live', '*-live-*', '*native-audio*')),
    ('image', ('*-image', '*-image-*', '*image-generation*')),
    ('embedding', ('*embedding*',)),
    ('imagen', ('imagen-*',)),
    ('veo', ('veo-*',)),
    ('aqa', ('aqa', 'aqa-*')),
    ('gemma', ('gemma-*',)),
)
_MAJOR = re.compile(r'^gemini-(\d+)(?:\.(\d+))?')
# effort (OpenAI reasoning_effort, the advisor's mode, Anthropic-derived) -> thinkingLevel
EFFORT_TO_LEVEL = {'none': 'LOW', 'minimal': 'LOW', 'low': 'LOW', 'medium': 'MEDIUM', 'high': 'HIGH', 'xhigh': 'HIGH',
                   'max': 'HIGH'}
LEVEL_TO_EFFORT = {'MINIMAL': 'minimal', 'LOW': 'low', 'MEDIUM': 'medium', 'HIGH': 'high'}
EFFORT_BUDGET = {'none': 0, 'minimal': 512, 'low': 1024, 'medium': 4096, 'high': 16384, 'xhigh': 24576, 'max': 24576}
PRO_MIN_BUDGET = 128  # gemini-2.5-pro cannot turn thinking off; its smallest budget


def _note(notes, text):
    if notes is not None and text not in notes:
        notes.append(text)


# ---------------------------------------------------------------------------------------------------------------------
# model families and thinking

def model_family(model: str | None) -> str:
    """'tts' | 'live' | 'image' | 'embedding' | 'imagen' | 'veo' | 'aqa' | 'gemma' | 'gemini' (anything else)."""
    name = normalize_model(model or '').lower()
    for family, globs in MODEL_FAMILIES:
        if any(fnmatch.fnmatchcase(name, glob) for glob in globs):
            return family
    return 'gemini'


def thinking_mode(model: str | None) -> str | None:
    """'level' (gemini-3 and later: thinkingLevel), 'budget' (gemini-2.5, '*-latest' aliases and unknown names:
    thinkingBudget, which every thinking model accepts) or None (no thinking: TTS, image output, embedding, Imagen, Veo,
    Live, AQA, Gemma, gemini-1.x and 2.0)."""
    if model_family(model) != 'gemini':
        return None
    name = normalize_model(model or '').lower()
    found = _MAJOR.match(name)
    if found:
        major, minor = int(found.group(1)), int(found.group(2) or 0)
        if major >= 3:
            return 'level'
        if major < 2 or (major == 2 and minor < 5):
            return None
    return 'budget'


def json_schema_native(model: str | None) -> bool:
    """Whether the model takes responseJsonSchema / parametersJsonSchema (gemini-2.5 and later)."""
    if model_family(model) not in ('gemini', 'image'):
        return False
    found = _MAJOR.match(normalize_model(model or '').lower())
    if not found:
        return False
    major, minor = int(found.group(1)), int(found.group(2) or 0)
    return major >= 3 or (major == 2 and minor >= 5)


def _is_pro(model: str | None) -> bool:
    return '-pro' in normalize_model(model or '').lower()


def thinking_config(model: str | None, *, effort: str | None = None, budget: int | None = None,
                    notes: list | None = None) -> dict | None:
    """The thinkingConfig for a requested effort or budget on this model, or None (noted) when the model has no
    thinking. Level models prefer the effort (a bare budget is still accepted by Gemini 3 and is sent as is)."""
    mode = thinking_mode(model)
    if mode is None:
        _note(notes, 'dropped:reasoning_effort')
        return None
    effort = effort.lower() if isinstance(effort, str) else None
    if mode == 'level' and effort is not None:
        level = EFFORT_TO_LEVEL.get(effort)
        if level is None:
            _note(notes, 'dropped:reasoning_effort')
            return None
        if level == 'MEDIUM' and _is_pro(model):
            level = 'HIGH'  # Gemini 3 Pro offers LOW and HIGH only
        if LEVEL_TO_EFFORT.get(level) != effort:
            _note(notes, 'mapped:reasoning_effort')
        return {'thinkingLevel': level, 'includeThoughts': True}
    if budget is None or isinstance(budget, bool) or not isinstance(budget, int):
        if effort is None or effort not in EFFORT_BUDGET:
            _note(notes, 'dropped:reasoning_effort')
            return None
        budget = EFFORT_BUDGET[effort]
        _note(notes, 'mapped:reasoning_effort')
    if budget <= 0 and _is_pro(model):
        budget = PRO_MIN_BUDGET
        _note(notes, 'mapped:reasoning_effort')
    config = {'thinkingBudget': int(budget)}
    if budget > 0:
        config['includeThoughts'] = True
    return config


# ---------------------------------------------------------------------------------------------------------------------
# field spellings (lowerCamelCase and snake_case are both accepted by the REST API)

# values that are user data or JSON Schema: their keys are never renamed
_VERBATIM = frozenset({'args', 'response', 'responseJsonSchema', 'parametersJsonSchema', 'labels', 'data', 'metadata',
                       'example', 'default'})


def _camel(name: str) -> str:
    head, *rest = name.split('_')
    return head + ''.join(part[:1].upper() + part[1:] for part in rest)


def camelize(value, _depth: int = 0):
    """A Gemini request with every snake_case field name in lowerCamelCase (a name spelled both ways keeps the
    lowerCamelCase value). User data (function args and responses, labels, inline data) and JSON Schema values
    (responseJsonSchema, parametersJsonSchema) are left as sent; the property names of a Schema object are kept."""
    if _depth > 48:
        return value
    if isinstance(value, list):
        return [camelize(item, _depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict = {}
    for key, item in value.items():
        name = key
        if isinstance(key, str) and '_' in key.strip('_') and not key.startswith('_'):
            name = _camel(key)
            if name in value:  # both spellings: the lowerCamelCase one wins
                continue
        if name in _VERBATIM:
            out[name] = item
        elif name == 'properties' and isinstance(item, dict):
            out[name] = {k: camelize(v, _depth + 1) for k, v in item.items()}
        else:
            out[name] = camelize(item, _depth + 1)
    return out


def _gen_key(body: dict) -> str:
    """The generation config key a body uses ('generationConfig' unless it spells only generation_config)."""
    return 'generation_config' if 'generation_config' in body and 'generationConfig' not in body \
        else 'generationConfig'


def _get(mapping: dict, camel: str):
    if not isinstance(mapping, dict):
        return None
    if camel in mapping:
        return mapping[camel]
    snake = re.sub(r'([A-Z])', lambda m: '_' + m.group(1).lower(), camel)
    return mapping.get(snake)


def output_cap(body) -> int | None:
    """maxOutputTokens of a generateContent body in either spelling, or None."""
    if not isinstance(body, dict):
        return None
    value = _get(_get(body, 'generationConfig') or {}, 'maxOutputTokens')
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def with_output_cap(body: dict, value: int) -> dict:
    """The body with maxOutputTokens set inside the config object it already has (only when no cap is set)."""
    out = dict(body)
    if output_cap(out) is not None:
        return out
    key = _gen_key(out)
    config = dict(out.get(key) or {}) if isinstance(out.get(key), dict) else {}
    config['max_output_tokens' if key == 'generation_config' else 'maxOutputTokens'] = value
    out[key] = config
    return out


def with_thinking(body: dict, mode: str, model: str | None, notes: list) -> dict:
    """R19 on the Gemini passthrough: the advisor's mode as a thinkingConfig when the request sets none and the model
    thinks (thinkingLevel on gemini-3 and later, thinkingBudget otherwise); unchanged for models without thinking."""
    out = dict(body)
    key = _gen_key(out)
    config = out.get(key)
    config = dict(config) if isinstance(config, dict) else {}
    if _get(config, 'thinkingConfig') is not None:
        return out
    scratch: list = []
    if model is None:
        thinking = {'thinkingBudget': EFFORT_BUDGET.get(mode, 4096)}
    else:
        thinking = thinking_config(model, effort=mode, notes=scratch)
        if thinking is None:
            return out
        thinking.pop('includeThoughts', None)  # the client did not ask to see thoughts
    config['thinking_config' if key == 'generation_config' else 'thinkingConfig'] = thinking
    out[key] = config
    notes.append('injected:thinkingLevel' if 'thinkingLevel' in thinking else 'injected:thinkingBudget')
    return out


def normalize_model(name: str) -> str:
    """'models/<id>' -> '<id>'."""
    return name[len('models/'):] if isinstance(name, str) and name.startswith('models/') else name


def stop_from(reason: str | None, *, tool_called: bool) -> str:
    if reason == 'MAX_TOKENS':
        return 'max_tokens'
    if reason in FILTER_REASONS:
        return 'content_filter'
    return 'tool_use' if tool_called else 'end_turn'


# ---------------------------------------------------------------------------------------------------------------------
# schema reduction (JSON Schema -> the Gemini Schema object)

def _ref_target(root: dict, ref: str):
    if not isinstance(ref, str) or not ref.startswith('#/'):
        return None
    node = root
    for token in ref[2:].split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        if not isinstance(node, dict) or token not in node:
            return None
        node = node[token]
    return node if isinstance(node, dict) else None


def inline_refs(schema, *, notes=None, _root=None, _depth: int = 0, _seen: tuple = ()):
    """schema with local '#/...' $ref references replaced by their targets ($defs / definitions removed). A recursive
    reference or one deeper than REF_DEPTH keeps only the target's type (the recursion is cut there); a reference that
    is not local or not found becomes an untyped schema. Both are noted (dropped:schema.$ref)."""
    root = schema if _root is None else _root
    if isinstance(schema, list):
        return [inline_refs(v, notes=notes, _root=root, _depth=_depth, _seen=_seen) for v in schema]
    if not isinstance(schema, dict):
        return schema
    if '$ref' in schema:
        ref = schema['$ref']
        target = _ref_target(root, ref)
        if target is None or ref in _seen or len(_seen) >= REF_DEPTH:
            _note(notes, 'dropped:schema.$ref')
            rest = {k: v for k, v in schema.items() if k != '$ref'}
            if target is not None and 'type' not in rest:  # a recursion cut keeps the target's type, not its body
                rest['type'] = target.get('type') or 'object'
            return inline_refs(rest, notes=notes, _root=root, _depth=_depth, _seen=_seen)
        merged = dict(target)
        merged.update({k: v for k, v in schema.items() if k != '$ref'})
        return inline_refs(merged, notes=notes, _root=root, _depth=_depth, _seen=_seen + (ref,))
    out = {}
    for key, value in schema.items():
        if key in ('$defs', 'definitions') and _depth == 0:
            continue
        if key in ('properties', 'patternProperties') and isinstance(value, dict):
            out[key] = {k: inline_refs(v, notes=notes, _root=root, _depth=_depth + 1, _seen=_seen)
                        for k, v in value.items()}
        elif key in ('items', 'anyOf', 'oneOf', 'allOf', 'additionalProperties', 'prefixItems', 'not'):
            out[key] = inline_refs(value, notes=notes, _root=root, _depth=_depth + 1, _seen=_seen)
        else:
            out[key] = value
    return out


def to_gemini_schema(schema, *, notes=None, _depth: int = 0):
    if _depth == 0 and isinstance(schema, dict):
        schema = inline_refs(schema, notes=notes)
    if _depth > 32 or not isinstance(schema, dict):
        return schema
    if 'const' in schema and 'enum' not in schema:
        schema = dict(schema, enum=[schema['const']])
        schema.pop('const')
        if 'type' not in schema and isinstance(schema['enum'][0], str):
            schema['type'] = 'string'
    if isinstance(schema.get('enum'), list) and not all(isinstance(v, str) for v in schema['enum']):
        schema = {k: v for k, v in schema.items() if k != 'enum'}
        _note(notes, 'dropped:schema.enum (non-string values)')
    out = {}
    kind = schema.get('type')
    if isinstance(kind, list):
        kinds = [k for k in kind if k != 'null']
        if 'null' in kind:
            out['nullable'] = True
        kind = kinds[0] if kinds else 'string'
    if isinstance(kind, str):
        out['type'] = kind.upper()
    for key, value in schema.items():
        if key == 'type':
            continue
        if key not in SCHEMA_KEYS:
            _note(notes, f'dropped:schema.{key}')
            continue
        if key == 'properties' and isinstance(value, dict):
            out[key] = {k: to_gemini_schema(v, notes=notes, _depth=_depth + 1) for k, v in value.items()}
        elif key == 'items':
            out[key] = to_gemini_schema(value, notes=notes, _depth=_depth + 1)
        elif key == 'anyOf' and isinstance(value, list):
            out[key] = [to_gemini_schema(v, notes=notes, _depth=_depth + 1) for v in value]
        else:
            out[key] = value
    return out


def from_gemini_schema(schema, _depth: int = 0):
    if _depth > 32 or not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key == 'type' and isinstance(value, str):
            out['type'] = value.lower()
        elif key == 'properties' and isinstance(value, dict):
            out[key] = {k: from_gemini_schema(v, _depth + 1) for k, v in value.items()}
        elif key == 'items':
            out[key] = from_gemini_schema(value, _depth + 1)
        elif key == 'anyOf' and isinstance(value, list):
            out[key] = [from_gemini_schema(v, _depth + 1) for v in value]
        elif key != 'nullable':
            out[key] = value
    if schema.get('nullable') and isinstance(out.get('type'), str):
        out['type'] = [out['type'], 'null']
    return out


# ---------------------------------------------------------------------------------------------------------------------
# decode (Gemini inbound -> IR)

def _code_text(part: dict) -> str:
    """A fenced text rendering of an executableCode or codeExecutionResult part (code execution tool)."""
    if 'executableCode' in part:
        code = part.get('executableCode') or {}
        language = str(code.get('language') or 'python').lower()
        language = 'python' if language in ('python', 'language_unspecified') else language
        return f'\n```{language}\n{code.get("code") or ""}\n```\n'
    result = part.get('codeExecutionResult') or {}
    output = result.get('output') or ''
    outcome = result.get('outcome') or 'OUTCOME_OK'
    label = 'output' if outcome == 'OUTCOME_OK' else f'output ({outcome})'
    return f'\n```{label}\n{output}\n```\n'


def _parts_in(parts, *, role: str, ids: dict, counter: list, notes) -> tuple[list, list]:
    """(content parts, tool results) of one Gemini content."""
    out, results = [], []
    pending_signature = None
    for part in parts or []:
        if not isinstance(part, dict):
            raise refuse('parts must be objects', 'invalid_request')
        if 'text' in part:
            if part.get('thought'):
                out.append(Thinking(part['text'] or '', part.get('thoughtSignature')))
            elif part['text']:
                out.append(Text(part['text']))
        elif 'inlineData' in part:
            inline = part['inlineData'] or {}
            mime = inline.get('mimeType') or 'application/octet-stream'
            if mime.startswith('image/'):
                out.append(Image(media_type=mime, data=unb64(inline.get('data') or '')))
            elif mime.startswith('audio/'):
                out.append(Audio(media_type=mime.split(';', 1)[0], data=unb64(inline.get('data') or '')))
            else:
                raise refuse(f'inlineData {mime} cannot be translated for this target')
        elif 'executableCode' in part or 'codeExecutionResult' in part:
            out.append(Text(_code_text(part)))
            _note(notes, 'converted:executableCode')
        elif 'functionCall' in part:
            call = part['functionCall'] or {}
            name = str(call.get('name') or '')
            args = call.get('args') or {}
            call_id = call.get('id') or synth_tool_id(counter[0], name, args)
            counter[0] += 1
            ids.setdefault(name, []).append(call_id)
            if part.get('thoughtSignature'):
                out.append(Thinking('', part['thoughtSignature']))
            out.append(ToolUse(id=call_id, name=name, input_json=json.dumps(args, ensure_ascii=False,
                                                                            separators=(',', ':'))))
        elif 'functionResponse' in part:
            response = part['functionResponse'] or {}
            name = str(response.get('name') or '')
            queue = ids.get(name) or []
            call_id = response.get('id') or (queue.pop(0) if queue else synth_tool_id(counter[0], name, {}))
            payload = response.get('response')
            is_error = isinstance(payload, dict) and set(payload) == {'error'}
            if isinstance(payload, dict) and set(payload) <= {'output', 'error'} and payload:
                value = payload.get('error') if is_error else payload.get('output')
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            else:
                text = json.dumps(payload, ensure_ascii=False)
            results.append(ToolResult(tool_use_id=call_id, parts=[Text(text)], is_error=is_error))
        elif 'fileData' in part:
            raise refuse('fileData parts cannot be translated for this target')
        elif part.get('thoughtSignature'):
            pending_signature = part['thoughtSignature']
        else:
            raise refuse('This part type cannot be translated for this target')
    if pending_signature:
        _note(notes, 'dropped:thoughtSignature')
    return out, results


def decode_request(body: dict, *, model: str = '', stream: bool = False, notes: list | None = None) -> IRRequest:
    """IRRequest from a generateContent / streamGenerateContent body (model from the path)."""
    if not isinstance(body, dict):
        raise refuse('The request body needs contents', 'invalid_request')
    body = camelize(body)  # the REST API takes snake_case and lowerCamelCase field names alike
    if not isinstance(body.get('contents'), list):
        raise refuse('The request body needs contents', 'invalid_request')
    system = []
    instruction = body.get('systemInstruction')
    for part in (instruction or {}).get('parts') or []:
        if 'text' not in part:
            raise refuse('systemInstruction carries text only')
        system.append(Text(part['text']))
    ids: dict = {}
    counter = [0]
    messages = []
    for content in body['contents']:
        role = content.get('role') or 'user'
        if role not in ('user', 'model', 'function'):
            raise refuse('content roles are user and model', 'invalid_request')
        parts, results = _parts_in(content.get('parts'), role=role, ids=ids, counter=counter, notes=notes)
        if results:
            messages.append(IRMessage(role='tool', parts=results))
        if parts or not results:
            messages.append(IRMessage(role='assistant' if role == 'model' else 'user', parts=parts))
    tools = []
    for tool in body.get('tools') or []:
        declarations = tool.get('functionDeclarations') or tool.get('function_declarations')
        if not declarations or set(tool) - {'functionDeclarations', 'function_declarations'}:
            raise refuse('Only functionDeclarations tools can be translated')
        for decl in declarations:
            parameters = decl.get('parametersJsonSchema') or from_gemini_schema(decl.get('parameters') or {}) \
                or {'type': 'object', 'properties': {}}
            tools.append(IRTool(name=decl['name'], description=decl.get('description'), parameters=parameters))
    choice = None
    config = ((body.get('toolConfig') or {}).get('functionCallingConfig')) or {}
    mode = config.get('mode')
    if mode:
        allowed = config.get('allowedFunctionNames') or []
        if mode == 'AUTO':
            choice = IRToolChoice('auto')
        elif mode == 'NONE':
            choice = IRToolChoice('none')
        elif mode in ('ANY', 'VALIDATED'):
            choice = IRToolChoice('tool', allowed[0]) if len(allowed) == 1 else IRToolChoice('required')
            if len(allowed) > 1:
                _note(notes, 'widened:allowedFunctionNames')
        else:
            raise refuse(f'functionCallingConfig mode {mode!r} is not supported')
    gen = body.get('generationConfig') or {}
    if isinstance(gen.get('candidateCount'), int) and gen['candidateCount'] > 1:
        raise refuse('candidateCount > 1 is not supported for this target')
    for name in gen:
        if name not in _GEN_KNOWN:
            _note(notes, f'dropped:generationConfig.{name}')
    sampling = {}
    for src, dst in (('temperature', 'temperature'), ('topP', 'top_p'), ('topK', 'top_k'), ('seed', 'seed'),
                     ('presencePenalty', 'presence_penalty'), ('frequencyPenalty', 'frequency_penalty')):
        if gen.get(src) is not None:
            sampling[dst] = gen[src]
    response_format = None
    schema = gen.get('responseJsonSchema') or gen.get('responseSchema')
    if gen.get('responseMimeType') == 'application/json':
        response_format = ({'type': 'json_schema', 'json_schema': {
            'name': 'output', 'schema': schema if 'responseJsonSchema' in gen else from_gemini_schema(schema)}}
            if schema else {'type': 'json_object'})
    thinking = gen.get('thinkingConfig') or {}
    reasoning = None
    level = thinking.get('thinkingLevel')
    if isinstance(thinking.get('thinkingBudget'), int) and thinking['thinkingBudget'] > 0:
        reasoning = IRReasoning(enabled=True, budget_tokens=thinking['thinkingBudget'],
                                effort=LEVEL_TO_EFFORT.get(str(level).upper()) if level else None)
    elif isinstance(level, str) and level.upper() in LEVEL_TO_EFFORT:
        reasoning = IRReasoning(enabled=True, effort=LEVEL_TO_EFFORT[level.upper()])
    for name in ('safetySettings', 'cachedContent', 'labels'):
        if body.get(name):
            _note(notes, f'dropped:{name}')
    return IRRequest(kind='chat', model=normalize_model(model), system=system, messages=messages, tools=tools,
                     tool_choice=choice, sampling=sampling,
                     max_output_tokens=gen.get('maxOutputTokens') if isinstance(gen.get('maxOutputTokens'), int) else None,
                     stop=gen.get('stopSequences'), response_format=response_format, reasoning=reasoning, stream=stream,
                     extras_by_protocol={'notes': list(notes or [])})


# ---------------------------------------------------------------------------------------------------------------------
# encode (IR -> Gemini upstream request)

def _tool_response(part: ToolResult, name: str) -> dict:
    text = ''.join(p.text for p in part.parts if isinstance(p, Text))
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict) and not part.is_error:
        payload = value
    else:
        payload = {'error' if part.is_error else 'output': value if value is not None and not part.is_error else text}
    return {'functionResponse': {'name': name, 'response': payload}}


def needs_signature(model: str | None) -> bool:
    """Whether the model validates thought signatures on functionCall parts (Gemini 3)."""
    import fnmatch
    name = normalize_model(model or '')
    return any(fnmatch.fnmatchcase(name, glob) for glob in GEMINI3_GLOBS)


_AUDIO_MIME = {'audio/mpeg': 'audio/mp3', 'audio/x-wav': 'audio/wav', 'audio/wave': 'audio/wav', 'audio/x-flac': 'audio/flac',
               'audio/x-aiff': 'audio/aiff', 'audio/pcm16': 'audio/l16', 'audio/pcm': 'audio/l16'}


def _audio_mime(media_type: str) -> str:
    """The Gemini spelling of an audio media type (audio/wav, audio/mp3, audio/aiff, audio/aac, audio/ogg, audio/flac)."""
    kind = (media_type or 'audio/wav').split(';', 1)[0].strip().lower()
    return _AUDIO_MIME.get(kind, kind)


def encode_contents(ir: IRRequest, *, notes=None, model: str | None = None) -> list:
    names: dict[str, str] = {}
    contents = []
    dummy = needs_signature(model if model is not None else ir.model)
    for message in merge_roles([IRMessage('user' if m.role == 'tool' else m.role, list(m.parts)) for m in ir.messages]):
        parts = []
        signature = None
        first_call = True
        for part in message.parts:
            if isinstance(part, Text):
                if part.text:
                    parts.append({'text': part.text})
            elif isinstance(part, Image):
                if part.data is None:
                    raise refuse('https image URLs cannot be sent to Gemini; send the image inline (base64)')
                parts.append({'inlineData': {'mimeType': part.media_type, 'data': b64(part.data)}})
            elif isinstance(part, Audio):
                parts.append({'inlineData': {'mimeType': _audio_mime(part.media_type), 'data': b64(part.data)}})
            elif isinstance(part, ToolUse):
                names[part.id] = part.name
                call = {'functionCall': {'name': part.name, 'args': request_args(part.input_json)}}
                if signature:
                    call['thoughtSignature'] = signature
                    signature = None
                elif dummy and first_call and message.role == 'assistant':
                    call['thoughtSignature'] = DUMMY_THOUGHT_SIGNATURE
                    _note(notes, 'injected:thoughtSignature')
                first_call = False
                parts.append(call)
            elif isinstance(part, ToolResult):
                if part.tool_use_id not in names:
                    raise refuse('A tool result names a tool call that is not in the conversation', 'invalid_request')
                parts.append(_tool_response(part, names[part.tool_use_id]))
            elif isinstance(part, Thinking):
                if not part.text and part.signature:
                    signature = part.signature
                else:
                    _note(notes, 'dropped:thinking')
            else:
                raise refuse(f'{type(part).__name__} content cannot be sent to Gemini')
        if parts:
            contents.append({'role': 'model' if message.role == 'assistant' else 'user', 'parts': parts})
    return contents


def _schema_field(schema, reduced_name: str, json_name: str, lossless: bool, notes) -> dict:
    """{reduced_name: Schema object} or, when the reduction would drop keys and the model takes JSON Schema,
    {json_name: the schema as sent} (nothing dropped, no notes)."""
    scratch: list = []
    reduced = to_gemini_schema(schema, notes=scratch)
    if scratch and lossless and isinstance(schema, dict):
        return {json_name: schema}
    for note in scratch:
        _note(notes, note)
    return {reduced_name: reduced}


def encode_request(ir: IRRequest, profile=None, *, notes: list | None = None, model: str | None = None) -> dict:
    """A generateContent body. model: the upstream model (it travels in the path; default ir.model), used for the
    Gemini 3 thought-signature rule."""
    upstream_model = model if model is not None else ir.model
    family = model_family(upstream_model)
    if family == 'tts':
        raise refuse('Gemini TTS models answer audio only; use POST /v1/audio/speech (or the native generateContent '
                     'with responseModalities AUDIO)')
    if family in ('embedding', 'imagen', 'veo', 'live', 'aqa'):
        raise refuse(f'{normalize_model(upstream_model)} does not serve generateContent chat')
    lossless = json_schema_native(upstream_model)
    body: dict = {'contents': encode_contents(ir, notes=notes, model=model)}
    if ir.system:
        body['systemInstruction'] = {'parts': [{'text': p.text} for p in ir.system if isinstance(p, Text)]}
    if ir.tools:
        declarations = []
        for tool in ir.tools:
            decl = {'name': tool.name}
            if tool.description:
                decl['description'] = tool.description
            if (tool.parameters or {}).get('properties'):
                decl.update(_schema_field(tool.parameters, 'parameters', 'parametersJsonSchema', lossless, notes))
            declarations.append(decl)
        body['tools'] = [{'functionDeclarations': declarations}]
    if ir.tool_choice is not None:
        mode = {'auto': 'AUTO', 'required': 'ANY', 'none': 'NONE', 'tool': 'ANY'}[ir.tool_choice.mode]
        config = {'mode': mode}
        if ir.tool_choice.mode == 'tool':
            config['allowedFunctionNames'] = [ir.tool_choice.name]
        body['toolConfig'] = {'functionCallingConfig': config}
    gen: dict = {}
    if ir.max_output_tokens is not None:
        gen['maxOutputTokens'] = ir.max_output_tokens
    for src, dst in (('temperature', 'temperature'), ('top_p', 'topP'), ('top_k', 'topK'), ('seed', 'seed'),
                     ('presence_penalty', 'presencePenalty'), ('frequency_penalty', 'frequencyPenalty')):
        if ir.sampling.get(src) is not None:
            gen[dst] = ir.sampling[src]
    for name in ir.sampling:
        if name not in ('temperature', 'top_p', 'top_k', 'seed', 'presence_penalty', 'frequency_penalty'):
            _note(notes, f'dropped:{name}')
    if ir.stop:
        gen['stopSequences'] = list(ir.stop)
    if ir.response_format is not None:
        kind = ir.response_format.get('type')
        if kind in ('json_schema', 'json_object'):
            gen['responseMimeType'] = 'application/json'
            if kind == 'json_schema':
                gen.update(_schema_field((ir.response_format.get('json_schema') or {}).get('schema') or {},
                                         'responseSchema', 'responseJsonSchema', lossless, notes))
    if ir.reasoning is not None and (ir.reasoning.budget_tokens or ir.reasoning.effort):
        thinking = thinking_config(upstream_model, effort=ir.reasoning.effort, budget=ir.reasoning.budget_tokens,
                                   notes=notes)
        if thinking is not None:
            gen['thinkingConfig'] = thinking
    elif ir.reasoning is not None:
        _note(notes, 'dropped:reasoning_effort')
    if gen:
        body['generationConfig'] = gen
    return body


# ---------------------------------------------------------------------------------------------------------------------
# responses

def _usage_meta(usage: Usage | None) -> dict:
    """usageMetadata from an IR Usage (output_tokens includes reasoning): candidatesTokenCount = output - reasoning,
    thoughtsTokenCount = reasoning, totalTokenCount = input + output (sse.gemini_usage is the inverse)."""
    usage = usage or Usage()
    prompt, out = usage.input_tokens or 0, usage.output_tokens or 0
    reasoning = min(usage.reasoning_tokens or 0, out)
    meta = {'promptTokenCount': prompt, 'candidatesTokenCount': out - reasoning, 'totalTokenCount': prompt + out}
    if usage.cached_input_tokens:
        meta['cachedContentTokenCount'] = usage.cached_input_tokens
    if reasoning:
        meta['thoughtsTokenCount'] = reasoning
    return meta


def decode_response(body: dict, *, notes: list | None = None) -> IRResponse:
    """IRResponse from a generateContent answer. Images become IR Image parts and audio IR Audio parts (the encoders
    carry or note them); executableCode / codeExecutionResult become fenced Text; groundingMetadata and
    urlContextMetadata are kept in extras and noted (no inbound protocol has a field for them)."""
    if not isinstance(body, dict):
        raise bad_upstream('The upstream generateContent response is not an object')
    candidates = body.get('candidates') or []
    if not candidates:
        feedback = body.get('promptFeedback') or {}
        if feedback.get('blockReason'):
            return IRResponse(id=str(body.get('responseId') or ''), model=str(body.get('modelVersion') or ''),
                              stop='content_filter', usage=gemini_usage(body.get('usageMetadata')) or Usage())
        raise bad_upstream('The upstream generateContent response has no candidates')
    candidate = candidates[0]
    ids: dict = {}
    counter = [0]
    parts, _ = _parts_in((candidate.get('content') or {}).get('parts'), role='model', ids=ids, counter=counter,
                         notes=notes)
    extras = {}
    for name in ('groundingMetadata', 'urlContextMetadata', 'citationMetadata'):
        if candidate.get(name):
            extras[name] = candidate[name]
            _note(notes, f'dropped:{name}')
    called = any(isinstance(p, ToolUse) for p in parts)
    return IRResponse(id=str(body.get('responseId') or ''), model=str(body.get('modelVersion') or ''), content=parts,
                      stop=stop_from(candidate.get('finishReason'), tool_called=called),
                      usage=gemini_usage(body.get('usageMetadata')) or Usage(), extras=extras)


def _response_parts(resp_parts) -> list:
    parts, signature = [], None
    for part in resp_parts:
        if isinstance(part, Text):
            parts.append({'text': part.text})
        elif isinstance(part, Thinking):
            if part.text:
                parts.append({'text': part.text, 'thought': True})
            elif part.signature:
                signature = part.signature
        elif isinstance(part, ToolUse):
            try:
                args = json.loads(part.input_json or '{}')
            except ValueError:
                raise bad_upstream('Tool-call arguments are not valid JSON') from None
            call = {'functionCall': {'name': part.name, 'args': args}}
            if signature:
                call['thoughtSignature'], signature = signature, None
            parts.append(call)
        elif isinstance(part, (Image, Audio)) and part.data is not None:
            parts.append({'inlineData': {'mimeType': part.media_type, 'data': b64(part.data)}})
    return parts


def encode_response(resp: IRResponse, *, model_name: str) -> dict:
    return {'candidates': [{'content': {'parts': _response_parts(resp.content), 'role': 'model'},
                            'finishReason': STOP_TO_GEMINI.get(resp.stop, 'STOP'), 'index': 0}],
            'usageMetadata': _usage_meta(resp.usage), 'modelVersion': model_name,
            'responseId': resp.id or uuid.uuid4().hex}


# ---------------------------------------------------------------------------------------------------------------------
# streams

@closes_source
async def decode_stream(frames, *, notes: list | None = None):
    """IR events from alt=sse GenerateContentResponse frames. executableCode / codeExecutionResult parts stream as
    fenced text; image and audio parts and groundingMetadata have no IR stream event and are noted (dropped:...)."""
    started = False
    counter = [0]
    text_index = None
    thought_index = None
    next_index = 0
    called = False
    usage = None
    reason = None
    async for frame in frames:
        data = frame.json() if isinstance(frame, SSEFrame) else frame
        if not isinstance(data, dict):
            raise bad_upstream('The upstream stream sent a frame that is not an object')
        if 'error' in data:
            yield stream_error_event(data, adapter='gemini')
            return
        if not started:
            started = True
            yield Start(id=str(data.get('responseId') or uuid.uuid4().hex), model=str(data.get('modelVersion') or ''))
        for candidate in (data.get('candidates') or [])[:1]:
            for part in (candidate.get('content') or {}).get('parts') or []:
                if 'text' in part and part.get('thought'):
                    if thought_index is None:
                        thought_index, next_index = next_index, next_index + 1
                    if part.get('text'):
                        yield ThinkingDelta(thought_index, part['text'])
                elif 'text' in part or 'executableCode' in part or 'codeExecutionResult' in part:
                    if thought_index is not None:
                        yield BlockStop(thought_index)
                        thought_index = None
                    if text_index is None:
                        text_index, next_index = next_index, next_index + 1
                    text = part.get('text') if 'text' in part else _code_text(part)
                    if 'text' not in part:
                        _note(notes, 'converted:executableCode')
                    if text:
                        yield TextDelta(text_index, text)
                elif 'inlineData' in part or 'inline_data' in part:
                    inline = part.get('inlineData') or part.get('inline_data') or {}
                    mime = str(inline.get('mimeType') or inline.get('mime_type') or '')
                    _note(notes, 'dropped:audio_output' if mime.startswith('audio/') else 'dropped:image_output')
                elif 'functionCall' in part:
                    for open_index in (thought_index, text_index):
                        if open_index is not None:
                            yield BlockStop(open_index)
                    thought_index = text_index = None
                    call = part['functionCall'] or {}
                    name, args = str(call.get('name') or ''), call.get('args') or {}
                    index, next_index = next_index, next_index + 1
                    if part.get('thoughtSignature'):
                        yield ThinkingDelta(index, '', part['thoughtSignature'])
                        index, next_index = next_index, next_index + 1
                    call_id = call.get('id') or synth_tool_id(counter[0], name, args)
                    counter[0] += 1
                    called = True
                    yield ToolCallStart(index, call_id, name)
                    yield ToolArgsDelta(index, json.dumps(args, ensure_ascii=False, separators=(',', ':')))
                    yield BlockStop(index)
            if candidate.get('finishReason'):
                reason = candidate['finishReason']
            for name in ('groundingMetadata', 'urlContextMetadata', 'citationMetadata'):
                if candidate.get(name):
                    _note(notes, f'dropped:{name}')
        if isinstance(data.get('usageMetadata'), dict):
            usage = gemini_usage(data['usageMetadata'])
        feedback = data.get('promptFeedback') or {}
        if feedback.get('blockReason'):
            reason = 'SAFETY'
    for open_index in (thought_index, text_index):
        if open_index is not None:
            yield BlockStop(open_index)
    if started:
        yield Stop(stop_from(reason, tool_called=called))
        if usage is not None:
            yield UsageEvent(usage)


@closes_source
async def encode_stream(events, *, model_name: str):
    """Gemini alt=sse frames: one chunk per text delta, whole functionCall parts at BlockStop, then a final chunk with
    finishReason and usageMetadata."""
    tail = StreamTail()
    rid = uuid.uuid4().hex
    tools: dict[int, list] = {}  # index -> [name, fragments]
    signature = None

    def chunk(parts, **extra):
        candidate = {'content': {'parts': parts, 'role': 'model'}, 'index': 0}
        candidate.update(extra.pop('candidate', {}))
        return write_json_sse({'candidates': [candidate], **extra, 'modelVersion': model_name, 'responseId': rid})

    async for event in events:
        if isinstance(event, Start):
            rid = event.id or rid
        elif isinstance(event, TextDelta):
            if event.text:
                yield chunk([{'text': event.text}])
        elif isinstance(event, ThinkingDelta):
            if event.text:
                yield chunk([{'text': event.text, 'thought': True}])
            if event.signature:
                signature = event.signature
        elif isinstance(event, ToolCallStart):
            tools[event.index] = [event.name, []]
        elif isinstance(event, ToolArgsDelta):
            if event.index in tools:
                tools[event.index][1].append(event.partial_json)
        elif isinstance(event, BlockStop):
            if event.index in tools:
                name, fragments = tools.pop(event.index)
                text = ''.join(fragments)
                try:
                    args = json.loads(text) if text.strip() else {}
                except ValueError:
                    yield stream_terminal('gemini', 'upstream_bad_response', 'Tool-call arguments are not valid JSON')
                    return
                part = {'functionCall': {'name': name, 'args': args}}
                if signature:
                    part['thoughtSignature'], signature = signature, None
                yield chunk([part])
        elif tail.take(event) and isinstance(event, ErrorEvent):
            yield stream_terminal('gemini', error_code(event.code), event.message)
            return
    yield chunk([{'text': ''}], candidate={'finishReason': STOP_TO_GEMINI.get(tail.stop or 'end_turn', 'STOP')},
                usageMetadata=_usage_meta(tail.usage))


# ---------------------------------------------------------------------------------------------------------------------
# embeddings

def decode_embed_request(body: dict, *, batch: bool, model: str = '', notes: list | None = None) -> IRRequest:
    """IRRequest (kind embed) from an embedContent or batchEmbedContents body."""
    if not isinstance(body, dict):
        raise refuse('The request body must be a JSON object', 'invalid_request')
    body = camelize(body)
    requests = body.get('requests') if batch else [body]
    if not isinstance(requests, list) or not requests:
        raise refuse('requests must be a non-empty list', 'invalid_request')
    inputs, embed = [], {}
    for request in requests:
        parts = ((request or {}).get('content') or {}).get('parts') or []
        if not parts or any('text' not in p for p in parts):
            raise refuse('Only text content can be embedded by this target')
        inputs.append(''.join(p['text'] for p in parts))
        if request.get('outputDimensionality') is not None:
            embed['dimensions'] = request['outputDimensionality']
        if request.get('taskType'):
            embed['task_type'] = request['taskType']
    if embed.get('task_type'):
        _note(notes, 'dropped:taskType')
    return IRRequest(kind='embed', model=normalize_model(model), inputs=inputs,
                     extras_by_protocol={'embed': embed, 'notes': list(notes or [])})


def embed_request(ir: IRRequest, *, batch: bool | None = None) -> dict:
    """An embedContent body (one input) or a batchEmbedContents body (several, or batch=True)."""
    inputs = list(ir.inputs or [])
    embed = ir.extras_by_protocol.get('embed') or {}
    batch = len(inputs) > 1 if batch is None else batch

    def one(text):
        request = {'content': {'parts': [{'text': text}]}}
        if embed.get('dimensions') is not None:
            request['outputDimensionality'] = embed['dimensions']
        if embed.get('task_type'):
            request['taskType'] = embed['task_type']
        return request
    if not batch:
        return one(inputs[0])
    model = 'models/' + normalize_model(ir.model)
    return {'requests': [dict(one(text), model=model) for text in inputs]}


def estimate_embed_usage(texts) -> Usage:
    """Estimated input tokens of an embedding call (Gemini embedding answers carry no counts): 4 characters per token,
    at least one token per non-empty input."""
    tokens = 0
    for text in texts or []:
        if isinstance(text, str) and text:
            tokens += max(1, -(-len(text) // 4))
    return Usage(input_tokens=tokens, estimated=True)


def embed_texts(body) -> list[str]:
    """The input texts of an embedContent or batchEmbedContents body (for the usage estimate)."""
    if not isinstance(body, dict):
        return []
    requests = body.get('requests') if isinstance(body.get('requests'), list) else [body]
    texts = []
    for request in requests:
        parts = ((request or {}).get('content') or {}).get('parts') if isinstance(request, dict) else None
        for part in parts or []:
            if isinstance(part, dict) and isinstance(part.get('text'), str):
                texts.append(part['text'])
    return texts


def embed_response(body: dict, *, batch: bool, inputs=None) -> dict:
    """{'vectors': [[float]], 'usage': Usage} from an embedContent or batchEmbedContents response; usage is estimated
    from inputs (the request texts)."""
    if not isinstance(body, dict):
        raise bad_upstream('The upstream embedding response is not an object')
    rows = body.get('embeddings') if batch else [body.get('embedding')]
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get('values'), list) for r in rows):
        raise bad_upstream('The upstream embedding response has no values')
    return {'vectors': [r['values'] for r in rows], 'usage': estimate_embed_usage(inputs)}


def encode_embed_response(vectors, *, batch: bool) -> dict:
    if batch:
        return {'embeddings': [{'values': list(v)} for v in vectors]}
    return {'embedding': {'values': list(vectors[0])}}


__all__ = ['DUMMY_THOUGHT_SIGNATURE', 'EFFORT_BUDGET', 'EFFORT_TO_LEVEL', 'FILTER_REASONS', 'GEMINI3_GLOBS',
           'LEVEL_TO_EFFORT', 'MODEL_FAMILIES', 'STOP_TO_GEMINI', 'camelize', 'decode_embed_request', 'decode_request',
           'decode_response', 'decode_stream', 'embed_request', 'embed_response', 'embed_texts', 'encode_contents',
           'encode_embed_response', 'encode_request', 'encode_response', 'encode_stream', 'estimate_embed_usage',
           'from_gemini_schema', 'inline_refs', 'json_schema_native', 'model_family', 'needs_signature',
           'normalize_model', 'output_cap', 'stop_from', 'thinking_config', 'thinking_mode', 'to_gemini_schema',
           'with_output_cap', 'with_thinking']
