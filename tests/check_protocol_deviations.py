"""The provider-compatibility parity/deviation map is code-bound.

docs/protocol_deviations.json rows are asserted against the live FastAPI app: request-model fields (extra='forbid' schemas),
declared multipart Form()/File() parameters, absent upstream routes, and live refusal probes over a synthetic inference seam.
Rows with check broker_evidence (the broker-* sections and the accepted_auth rows) cite broker test ids: each must exist in the
test source and have passed in the evidence run embedded in docs/broker_matrix.json. The legend names every behaviour and check
class the rows use. docs/PROTOCOL_DEVIATIONS.md must equal the rendering of the JSON. The app runs under a temporary
TITANOS_HOME. No model weights, no network, no process."""
import ast
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))

from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import render_protocol_deviations as renderer  # noqa: E402
from titanos import protocols  # noqa: E402
from titanos.config import Config, ModelSpec  # noqa: E402
from titanos.server import create_app  # noqa: E402

MAP = ROOT / 'docs' / 'protocol_deviations.json'
BROKER_MATRIX = ROOT / 'docs' / 'broker_matrix.json'
DOC = ROOT / 'docs' / 'PROTOCOL_DEVIATIONS.md'
ACCEPTED = {'accepted_mapped', 'accepted_ignored', 'partially_mapped'}
WAV = b'RIFF' + (36).to_bytes(4, 'little') + b'WAVEfmt ' + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little') + (1).to_bytes(2, 'little') + (16000).to_bytes(4, 'little') + (32000).to_bytes(4, 'little') + (2).to_bytes(2, 'little') + (16).to_bytes(2, 'little') + b'data' + (0).to_bytes(4, 'little')
PNG = bytes.fromhex('89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da63f8ffff3f0300050001ff2b29fb0000000049454e44ae426082')


def route_table(app):
    models, forms, paths = {}, {}, set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        paths.add(route.path)
        names = set()
        for param in route.dependant.body_params:
            kind = type(param.field_info).__name__
            if kind in ('Form', 'File'):
                names.add(getattr(param, 'alias', None) or param.name)
            else:
                annotation = getattr(param.field_info, 'annotation', None)
                if hasattr(annotation, 'model_fields'):
                    models[route.path] = annotation
        if names:
            forms[route.path] = names
    return models, forms, paths


def model_names(model):
    names = set(model.model_fields)
    for name, field in model.model_fields.items():
        if field.alias:
            names.add(field.alias)
    return names


def check_rows(rows, models, forms, paths):
    failures, counted = [], Counter()
    for row in rows:
        check, route, field, behaviour = row['check'], row['route'], row['field'], row['behaviour']
        if check == 'documented':
            counted['documented'] += 1
            continue
        if check == 'broker_evidence':
            counted[check] += 1  # asserted by check_broker_evidence
            continue
        if check == 'route_absent':
            if route in paths:
                failures.append(f"{row['adapter']} {route}: recorded as not implemented but the route exists")
            counted[check] += 1
            continue
        if check == 'ollama_options_field':
            name = field[len('options.'):]
            present = name in protocols.Options.model_fields
        elif check == 'json_model_field':
            model = models.get(route)
            if model is None:
                failures.append(f"{row['adapter']} {route}: no JSON request model found for a json_model_field row ({field})")
                continue
            present = field in model_names(model)
            if behaviour == 'refused_422_unknown_field' and model.model_config.get('extra') != 'forbid':
                failures.append(f"{row['adapter']} {route} {field}: refused_422 recorded but {model.__name__} does not forbid extra fields")
        elif check == 'compat_field':
            # standard client fields the local routes accept and name in X-Titanos-Ignored, or refuse with a typed 400
            from titanos import request_compat
            kind = {'/v1/chat/completions': 'chat', '/v1/completions': 'completion', '/v1/responses': 'responses', '/v1/messages': 'anthropic',
                    '/v1/embeddings': 'embeddings'}.get(route)
            if kind is None:
                failures.append(f"{row['adapter']} {route} {field}: compat_field rows cover the chat, completions, Responses, Messages and embeddings routes")
                continue
            accepted = field in request_compat.KINDS[kind]
            refused = field in request_compat.REFUSED[kind]
            if behaviour == 'accepted_ignored' and not accepted or behaviour == 'refused_400' and not refused or \
                    behaviour == 'partially_mapped' and not refused or behaviour not in ('accepted_ignored', 'refused_400', 'partially_mapped'):
                failures.append(f"{row['adapter']} {route} {field}: {behaviour} disagrees with titanos.request_compat")
            counted[check] += 1
            continue
        elif check == 'multipart_form_field':
            declared = forms.get(route)
            if declared is None:
                failures.append(f"{row['adapter']} {route}: no multipart parameters found ({field})")
                continue
            present = field in declared
            if behaviour == 'refused_400':
                counted[check] += 1  # unknown fields (absent) or declared fields refused by code; the live probes cover the unknown case
                continue
        else:
            failures.append(f"unknown check class {check}")
            continue
        expected = behaviour in ACCEPTED or (behaviour == 'refused_400' and check != 'multipart_form_field' and present)
        if behaviour in ACCEPTED and not present:
            failures.append(f"{row['adapter']} {route} {field}: recorded {behaviour} but the field is not declared")
        elif behaviour == 'refused_422_unknown_field' and present:
            failures.append(f"{row['adapter']} {route} {field}: recorded refused_422 but the field is declared")
        counted[check] += 1
    return failures, counted


CITATION = re.compile(r'(?:(?P<file>(?<![\w/.-])(?:[\w/.-]+/)?[\w-]+\.(?:py|md)):|(?<=[\s,(;]):)(?P<nums>\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)'
                      r'(?P<paren>\s*\((?:[^()]|\([^()]*\))*\))?')
SOURCE_DIRS = ('', 'src/titanos/', 'src/titanos/broker/', 'src/titanos/broker/adapters/', 'tools/', 'tests/', 'docs/')
CITED_COLUMNS = ('code_ref', 'note', 'documented_in')
WINDOW = 1       # lines either side of a single-line citation that may hold the identifier
MAX_SPAN = 80    # a `N-M` citation names a block; anything wider is a region, not a citation
STOP_WORDS = {'header', 'headers', 'route', 'cloud', 'target', 'local', 'the', 'and', 'with', 'for', 'only', 'mode', 'request', 'response',
              'every', 'non', 'public', 'failure', 'after', 'first', 'bytes', 'commit', 'passthrough', 'router', 'query', 'default', 'targets',
              'all', 'any', 'from', 'top', 'level', 'other', 'fields', 'field', 'body', 'form', 'json', 'without', 'into', 'then', 'that'}
BEHAVIOUR_IDENTIFIERS = {'refused_422_unknown_field': ("extra='forbid'", 'extra="forbid"')}  # the schema line a 422 row binds to


def field_identifiers(field):
    """Identifiers a field name carries: its whitespace-separated words (punctuation trimmed) and the pieces split on
    separators, three or more characters and not a stop word; a TITANOS_NAME setting also yields the attribute name."""
    out = []
    for word in str(field or '').split():
        word = word.strip('();,:.')
        if len(word) >= 3 and word.lower() not in STOP_WORDS:
            out.append(word)
    for piece in re.split(r"""[\s,;:/()\[\]{}"'`=<>|+]+|\.(?!\w)|(?<=\w)\.(?=\w)|->""", str(field or '')):
        piece = piece.strip('.:*')
        if len(piece) >= 3 and piece.lower() not in STOP_WORDS and not piece.isdigit():
            out.append(piece)
            if re.fullmatch(r'TITANOS_[A-Z0-9_]+', piece):
                out.append(piece[len('TITANOS_'):].lower())
    return list(dict.fromkeys(out))


def parenthetical_phrases(paren):
    """The phrase a citation carries in parentheses, whole and split on its separators (pieces need a letter)."""
    text = paren.strip()[1:-1].strip()
    out = [text] if text else []
    for piece in re.split(r'[,;]| -> | and |\bwith\b|\bvia\b|\bthen\b', text):
        piece = piece.strip().strip('"\'`')
        if len(piece) >= 2 and re.search(r'[A-Za-z_]', piece):
            out.append(piece)
    return out


def names(identifier, window):
    """Whether identifier occurs as a whole token on one of the window's lines (a trailing `_`, `.` or `*` allows a prefix)."""
    ident = identifier.strip()
    if not ident:
        return False
    pattern = r'(?<![\w])' + re.escape(ident.rstrip('*')) + ('' if ident.endswith(('*', '_', '.')) else r'(?![\w])')
    return any(re.search(pattern, line) for line in window)


def check_code_refs(rows):
    """Every `file:N` / `file:N-M` citation in code_ref, note and documented_in (Python sources and the docs) names a file
    that exists and non-blank lines inside it, and the cited line (a window of WINDOW lines around it, or the whole span of
    a block of fewer than MAX_SPAN lines) contains an identifier the row names: the parenthetical that follows the
    citation, a word of the field name, the route, the schema line a refused_422 row binds to, or the row's explicit
    anchor for that line (row['anchors'][f'{path}:{start}']). Like tests/check_route_contract_matrix.py assert_citation,
    a citation that drifts onto another line fails here instead of waiting for a reviewer. An anchor whose citation is
    gone is a failure too."""
    failures, cited, lines = [], 0, {}
    for row in rows:
        where = f"{row['adapter']} {row['route']} {row['field']}"
        anchors = dict(row.get('anchors') or {})
        used = set()
        own = [row['field']] + field_identifiers(row['field']) + list(BEHAVIOUR_IDENTIFIERS.get(row['behaviour'], ()))
        if str(row['route']).startswith('/') and len(row['route']) >= 6:
            own.append(row['route'])
        for column in CITED_COLUMNS:
            current = None
            for match in CITATION.finditer(row.get(column) or ''):
                if match.group('file'):
                    current = match.group('file')
                if current is None:
                    continue
                rel = next((base + current for base in SOURCE_DIRS if (ROOT / (base + current)).is_file()), None)
                if rel is None:
                    failures.append(f"{where}: {column} cites {current}, not a file")
                    continue
                if rel not in lines:
                    lines[rel] = (ROOT / rel).read_text(encoding='utf-8', errors='replace').splitlines()
                source = lines[rel]
                candidates = (parenthetical_phrases(match.group('paren')) if match.group('paren') else []) + own
                for span in match.group('nums').split(','):
                    start, _, end = span.partition('-')
                    start, end = int(start), int(end) if end else int(start)
                    cited += 1
                    if not all(0 < n <= len(source) for n in (start, end)):
                        failures.append(f"{where}: {rel}:{span} is past the end ({len(source)} lines)")
                        continue
                    if any(not source[n - 1].strip() for n in (start, end)):
                        failures.append(f"{where}: {rel}:{span} is a blank line")
                        continue
                    if end < start or end - start >= MAX_SPAN:
                        failures.append(f"{where}: {rel}:{span} is not a block of fewer than {MAX_SPAN} lines")
                        continue
                    key = f'{rel}:{start}'
                    used.add(key)
                    explicit = anchors.get(key)
                    explicit = [] if explicit is None else [explicit] if isinstance(explicit, str) else list(explicit)
                    window = source[max(0, start - 1 - WINDOW):end + WINDOW] if start == end else source[start - 1:end]
                    if not any(names(identifier, window) for identifier in explicit + candidates):
                        failures.append(f"{where}: {column} {rel}:{span} names none of the row's identifiers: {source[start - 1].strip()[:100]!r}")
        stale = sorted(set(anchors) - used)
        if stale:
            failures.append(f"{where}: anchors without a citation: {stale}")
    return failures, cited


def test_ids(check):
    """Class.method ids a unittest run of a test module reports, read from its source (no import): each class's test methods
    plus those inherited from base classes defined in the same module."""
    path = ROOT / 'tests' / check
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding='utf-8'))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def methods(name, seen=()):
        node = classes.get(name)
        if node is None or name in seen:
            return set()
        own = {item.name for item in node.body if isinstance(item, ast.FunctionDef) and item.name.startswith('test_')}
        for base in node.bases:
            if isinstance(base, ast.Name):
                own |= methods(base.id, seen + (name,))
        return own
    return {f'{name}.{method}' for name in classes for method in methods(name)}


def check_broker_evidence(data):
    """broker_evidence rows: every cited test exists and passed in the evidence run of docs/broker_matrix.json; the legend
    covers every behaviour and check class in use; accepted_auth rows exist for x-api-key, xi-api-key and x-goog-api-key."""
    failures = []
    rows = data['rows']
    for name, legend in (('behaviour', data['behaviours']), ('check', data['checks'])):
        unknown = sorted({r[name] for r in rows} - set(legend))
        if unknown:
            failures.append(f'{name} values without a legend entry: {unknown}')
    run = json.loads(BROKER_MATRIX.read_text(encoding='utf-8'))['run']['checks'] if BROKER_MATRIX.is_file() else None
    sources = {}
    cited = 0
    for row in rows:
        evidence = row.get('evidence') or []
        if (row['check'] == 'broker_evidence') != bool(evidence):
            failures.append(f"{row['adapter']} {row['route']} {row['field']}: check broker_evidence and an evidence list go together")
        for item in evidence:
            cited += 1
            check, _, test = item.partition('::')
            if check not in sources:
                sources[check] = test_ids(check)
            if sources[check] is None or test not in sources[check]:
                failures.append(f"{row['adapter']} {row['field']}: cited test {item} does not exist")
            elif run is None or test not in run.get(check, {}).get('passed', []):
                failures.append(f"{row['adapter']} {row['field']}: cited test {item} did not pass in the docs/broker_matrix.json evidence run")
    auth = {(r['route'], r['field']) for r in rows if r['behaviour'] == 'accepted_auth'}
    for expected in (('/v1/messages', 'header x-api-key'), ('/v1/text-to-speech/{voice_id}', 'header: xi-api-key'),
                     ('/v1beta/models/{model_action}', 'header x-goog-api-key')):
        if expected not in auth:
            failures.append(f'no accepted_auth row for {expected}')
    for adapter in ('broker-openai', 'broker-anthropic', 'broker-gemini', 'broker-elevenlabs', 'broker-ollama'):
        if adapter not in renderer.ORDER or not any(r['adapter'] == adapter for r in rows):
            failures.append(f'section {adapter} is missing from the rows or the render ORDER')
    return failures, cited


def live_probes():
    seen = []

    def infer(name, task, data, emit=None, cancel=None):
        seen.append((name, task, data))
        if task == 'embeddings':
            return {'vectors': [[.1, .2] for _ in data['input']], 'usage': {'prompt_tokens': 2, 'total_tokens': 2}}
        if task == 'speech':
            return b'RIFF' + b'\0' * 40
        if emit:
            emit('Hello')
        return {'text': 'Hello', 'finish_reason': 'stop', 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}

    checks = 0
    with tempfile.TemporaryDirectory() as temp:
        cfg = Config(models={'chat': ModelSpec(path=temp, task='chat'), 'voice': ModelSpec(path=temp, task='speech'), 'asr': ModelSpec(path=temp, task='transcription'),
                             'painter': ModelSpec(path=temp, task='image', backend='diffusers')})
        app = create_app(cfg, output_dir=Path(temp) / 'outputs')
        app.state.engine.target.infer = infer
        with TestClient(app, base_url='http://localhost') as c:
            chat = {'model': 'chat', 'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': False}
            # Ollama: keep_alive and options.num_ctx are accepted and named in X-Titanos-Dropped (tests/check_ollama_compat.py); other unknown
            # top-level and options fields are schema refusals (422); an Ollama tool result (role tool + tool_name) that answers no pending
            # tool call is a typed 400 naming the message (the bound loop is tests/check_ollama_compat.py), never a schema dump
            r = c.post('/api/chat', json={**chat, 'keep_alive': '5m'})
            assert r.status_code == 200 and r.headers['x-titanos-dropped'] == 'keep_alive', (r.status_code, r.headers, r.text[:200]); checks += 1
            assert c.post('/api/chat', json={**chat, 'think': True}).status_code == 422; checks += 1
            r = c.post('/api/chat', json={**chat, 'options': {'num_ctx': 4096}})
            assert r.status_code == 200 and r.headers['x-titanos-dropped'] == 'options.num_ctx', (r.status_code, r.headers, r.text[:200]); checks += 1
            assert c.post('/api/chat', json={**chat, 'options': {'num_gpu': 1}}).status_code == 422; checks += 1
            r = c.post('/api/chat', json={**chat, 'messages': [{'role': 'user', 'content': 'Hi'}, {'role': 'tool', 'content': '3', 'tool_name': 'add'}]})
            assert r.status_code == 400 and r.json()['error']['message'].startswith("messages[1]: the tool result for tool_name 'add'") and 'Extra inputs' not in r.text, r.text; checks += 1
            assert c.post('/api/generate', json={'model': 'chat', 'prompt': 'Hi', 'raw': True, 'stream': False}).status_code == 422; checks += 1
            # Anthropic: malformed upstream-shaped blocks are 400 client errors, not KeyError 500s
            base = {'model': 'chat', 'max_tokens': 8}
            for content, label in ((['x'], 'non-dict block'), ([{'type': 'image'}], 'image without source'), ([{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png'}}], 'image source without data'),
                                   ([{'type': 'text'}], 'text without text'), ([{'type': 'tool_use', 'id': 'a'}], 'tool_use without name/input'), ([{'type': 'tool_result'}], 'tool_result without tool_use_id')):
                r = c.post('/v1/messages', json={**base, 'messages': [{'role': 'user', 'content': content}]})
                assert r.status_code == 400 and r.json()['error']['type'] == 'invalid_request_error', (label, r.status_code, r.text[:200]); checks += 1
            r = c.post('/v1/messages', json={**base, 'messages': [{'role': 'user', 'content': 'Hi'}], 'tools': [{'name': 'search'}]})
            assert r.status_code == 400, r.text; checks += 1
            r = c.post('/v1/messages', json={**base, 'messages': [{'role': 'user', 'content': 'Hi'}], 'metadata': {'user_id': 'u'}})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'metadata', (r.status_code, r.text[:200]); checks += 1
            assert c.post('/v1/messages', json={**base, 'messages': [{'role': 'user', 'content': 'Hi'}], 'betas': ['x']}).status_code == 422; checks += 1
            assert c.post('/v1/messages', json={**base, 'messages': [{'role': 'user', 'content': 'Hi'}], 'container': 'c'}).status_code == 400; checks += 1
            # OpenAI chat and Responses: the standard identity/cache/storage fields are named, unservable values are typed 400s
            r = c.post('/v1/chat/completions', json={**chat, 'user': 'u', 'prompt_cache_key': 'k', 'store': False})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'store, user, prompt_cache_key', (r.status_code, r.headers, r.text[:200]); checks += 1
            r = c.post('/v1/chat/completions', json={**chat, 'store': True})
            assert r.status_code == 400 and r.json()['error']['code'] == 'unsupported_capability', r.text[:200]; checks += 1
            # HiveMind's Ollama options object on the OpenAI routes: num_ctx named (no context option on transformers), other keys named, never a 422
            r = c.post('/v1/chat/completions', json={**chat, 'max_tokens': 32, 'temperature': 0, 'options': {'num_ctx': 4096}})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'options.num_ctx', (r.status_code, r.headers, r.text[:200]); checks += 1
            r = c.post('/v1/completions', json={'model': 'chat', 'prompt': 'Hi', 'options': {'num_ctx': 4096, 'num_gpu': 1}})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'options.num_ctx, options.num_gpu', (r.status_code, r.headers, r.text[:200]); checks += 1
            # the llama.cpp sampling fields are schema fields now: inert values are served, live ones a local backend lacks are typed 400s naming the field
            assert c.post('/v1/chat/completions', json={**chat, 'logit_bias': {}, 'n': 1, 'logprobs': False, 'presence_penalty': 0}).status_code == 200; checks += 1
            r = c.post('/v1/chat/completions', json={**chat, 'logprobs': True, 'top_logprobs': 2})
            assert r.status_code == 400 and r.json()['error']['code'] == 'unsupported_capability' and r.json()['error']['message'].startswith('logprobs'), r.text[:200]; checks += 1
            r = c.post('/v1/embeddings', json={'model': 'chat', 'input': 'Hi', 'dimensions': 8})
            assert r.status_code == 400 and r.json()['error']['message'].startswith('dimensions'), r.text[:200]; checks += 1
            r = c.post('/v1/responses', json={'model': 'chat', 'input': 'Hi', 'previous_response_id': 'resp_1'})
            assert r.status_code == 400 and r.json()['error']['code'] == 'unsupported_capability', r.text[:200]; checks += 1
            r = c.post('/v1/responses', json={'model': 'chat', 'input': 'Hi', 'include': ['reasoning.encrypted_content'], 'reasoning': {'effort': 'low'}})
            assert r.status_code == 200 and r.headers['x-titanos-ignored'] == 'include, reasoning', (r.status_code, r.text[:200]); checks += 1
            r = c.post('/v1/messages', json={**base, 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hi'}]}]})
            assert r.status_code == 200 and r.json()['content'][0]['text'] == 'Hello', r.text; checks += 1
            # ElevenLabs: malformed body and unsupported fields are 400s
            r = c.post('/v1/text-to-speech/default', content=b'not json', headers={'Content-Type': 'application/json'})
            assert r.status_code == 400, (r.status_code, r.text[:200]); checks += 1
            assert c.post('/v1/text-to-speech/default', json={'text': 'Hi', 'model_id': 'voice', 'voice_settings': {'stability': 0.5}}).status_code == 400; checks += 1
            assert c.post('/v1/text-to-speech/default?output_format=mp3_44100_128', json={'text': 'Hi', 'model_id': 'voice'}).status_code == 400; checks += 1
            assert c.post('/v1/text-to-speech/default', json={'text': 'Hi', 'model_id': 'voice'}).content.startswith(b'RIFF'); checks += 1
            # Multipart routes: unknown upstream fields are refused explicitly instead of being dropped
            for route, data, extra in (('/v1/audio/transcriptions', {'model': 'asr'}, {'temperature': '0'}), ('/v1/audio/transcriptions', {'model': 'asr'}, {'include[]': 'logprobs', 'stream': 'true'}),
                                       ('/v1/audio/translations', {'model': 'asr'}, {'prompt': 'glossary', 'temperature': '0.2'})):
                r = c.post(route, files={'file': ('a.wav', io.BytesIO(WAV), 'audio/wav')}, data={**data, **extra})
                assert r.status_code == 400 and r.json()['error']['code'] == 'unsupported_capability' and 'Unsupported multipart fields' in r.json()['error']['message'], (route, extra, r.status_code, r.text[:200]); checks += 1
                for name in extra:
                    assert name in r.json()['error']['message'], (name, r.text[:200])
            r = c.post('/v1/images/edits', files={'image': ('a.png', io.BytesIO(PNG), 'image/png')}, data={'model': 'painter', 'prompt': 'brighter', 'quality': 'high', 'output_format': 'png'})
            assert r.status_code == 400 and r.json()['error']['code'] == 'unsupported_capability' and 'quality' in r.json()['error']['message'] and 'output_format' in r.json()['error']['message'], r.text[:200]; checks += 1
            # the Studio's own transcription field set is never refused as unknown
            r = c.post('/v1/audio/transcriptions', files={'file': ('a.wav', io.BytesIO(WAV), 'audio/wav')}, data={'model': 'asr', 'language': 'en', 'vad_enabled': 'false', 'response_format': 'verbose_json'})
            assert 'Unsupported multipart fields' not in r.text, r.text[:200]; checks += 1
    return checks


def main():
    data = json.loads(MAP.read_text(encoding='utf-8'))
    rows = data['rows']
    with tempfile.TemporaryDirectory() as home:
        saved = os.environ.get('TITANOS_HOME')
        os.environ['TITANOS_HOME'] = home  # the live app never reads the caller's broker.json, keys or vault
        try:
            with tempfile.TemporaryDirectory() as temp:
                app = create_app(Config(models={}), output_dir=Path(temp) / 'outputs')
                models, forms, paths = route_table(app)
            failures, counted = check_rows(rows, models, forms, paths)
            evidence_failures, cited = check_broker_evidence(data)
            failures += evidence_failures
            citation_failures, citations = check_code_refs(rows)
            failures += citation_failures
            assert not failures, '\n'.join(failures[:40])
            rendered = renderer.render(data)
            current = DOC.read_text(encoding='utf-8').replace('\r\n', '\n') if DOC.exists() else ''
            assert current == rendered, 'docs/PROTOCOL_DEVIATIONS.md is out of date; run python tools/render_protocol_deviations.py'
            probes = live_probes()
        finally:
            if saved is None:
                os.environ.pop('TITANOS_HOME', None)
            else:
                os.environ['TITANOS_HOME'] = saved
    print(json.dumps({'passed': True, 'rows': len(rows), 'asserted_by_check': dict(counted), 'broker_evidence_ids': cited, 'line_citations': citations, 'live_probes': probes,
                      'scope': 'code-bound parity/deviation map; synthetic inference; vendor SDK compatibility not qualified'}))


if __name__ == '__main__':
    main()
