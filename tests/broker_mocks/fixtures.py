"""Provider wire fixtures: tests/fixtures/providers/<provider>/<case>.jsonl, the sentinel credentials, and the renderer the
mocks replay.

FORMAT
The first line of every fixture is a header object:
  source_url        the provider's PUBLIC API reference page the frames were transcribed from (https://), or for the
                    advisor fixtures the contract name (urn:) or the source file and lines (source:) they follow
  retrieved         the transcription date (YYYY-MM-DD)
  transcriber_note  how the rows were transcribed and what is synthetic
  kind              json | text | sse | ndjson | binary
  status            the HTTP status the mock answers with (default 200)
  headers           response headers (lower-case names, never credentials)
  request           optional example request {method, path, query, headers, json | form}; check_broker_fixtures.py sends it
                    to a fresh mock and compares the answer with the rendered rows byte for byte
  vars              the values the example request renders with
Every further line is one row:
  json    {"body": <any>}                                  exactly one row
  text    {"text": "<str>"}                                exactly one row
  sse     {"event": "<name>"|null, "data": <obj>|"[DONE]", "when": "<var>"}   one frame per row; 'when' keeps the row only
                                                                              when that var is truthy
          {"comment": "<text>"}                            an SSE comment frame (': <text>' plus a blank line)
  ndjson  {"line": <obj>}                                  one NDJSON line per row
  binary  {"variant": "<name>", "content_type": "<type>", "chunk_b64": "<base64>"}   chunks grouped by variant
Strings may carry ${name} placeholders: a string that is exactly "${name}" becomes the var's JSON value (any type); a
placeholder inside a longer string is replaced by str(value).

Bytes: JSON is rendered compactly (separators ',' and ':', non-ASCII kept); an SSE frame is 'event: <name>\\n' (when named)
plus 'data: <payload>\\n\\n'; an NDJSON line is the compact JSON plus '\\n'.

Hosts: rows use only 127.0.0.1 and example.invalid; credentials only the SENTINELS below (tests/check_broker_fixture_lint.py).
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

FIXTURES_DIR = Path(__file__).resolve().parents[1] / 'fixtures' / 'providers'
KINDS = ('json', 'text', 'sse', 'ndjson', 'binary')
HEADER_KEYS = ('source_url', 'retrieved', 'transcriber_note')

# Synthetic credentials (at least 8 characters, no provider key shape). A mock accepts exactly its sentinel; the hit
# recorder stores '<sentinel:NAME>' in place of a sentinel value and '<foreign:FP>' for anything else.
SENTINELS: dict[str, str] = {
    'openai': 'test-openai-not-a-key',
    'azure_openai': 'test-azure-not-a-key',
    'xai': 'test-xai-not-a-key',
    'mistral': 'test-mistral-not-a-key',
    'anthropic': 'test-anthropic-not-a-key',
    'gemini': 'test-gemini-not-a-key',
    'elevenlabs': 'test-elevenlabs-not-a-key',
    'ollama_cloud': 'test-ollama-not-a-key',
    'openrouter': 'test-openrouter-not-a-key',
    # the OpenAI-compatible family rows served by broker_mocks.compat_mock.CompatMock
    'groq': 'test-groq-not-a-key',
    'together': 'test-together-not-a-key',
    'deepseek': 'test-deepseek-not-a-key',
    'perplexity': 'test-perplexity-not-a-key',
    'fireworks': 'test-fireworks-not-a-key',
    'nvidia': 'test-nvidia-not-a-key',
    'cerebras': 'test-cerebras-not-a-key',
    'hivemind_gateway': 'test-hivemind-not-a-key',
    'pair_facade': 'test-pair-not-a-key',
    'advisor': 'test-advisor-not-a-key',
    # carried in error bodies by the status fault (leak=True) and never accepted as a credential
    'leak': 'test-leak-sentinel-not-a-key',
    # a well-formed credential that no mock accepts (each mock accepts only its own provider's sentinel)
    'wrong': 'test-wrong-not-a-key',
}
SENTINEL_NAMES: dict[str, str] = {value: name for name, value in SENTINELS.items()}
assert all(len(value) >= 8 for value in SENTINELS.values())

_PLACEHOLDER = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')
_WHOLE = re.compile(r'^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$')


def dumps(value: Any) -> str:
    """The compact JSON spelling every mock emits."""
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def substitute(value: Any, variables: dict[str, Any]) -> Any:
    """Replace ${name} placeholders recursively (see the module docstring). An unknown name raises KeyError."""
    if isinstance(value, str):
        whole = _WHOLE.match(value)
        if whole:
            return variables[whole.group(1)]
        return _PLACEHOLDER.sub(lambda m: str(variables[m.group(1)]), value)
    if isinstance(value, list):
        return [substitute(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, variables) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Fixture:
    provider: str
    case: str
    header: dict
    rows: tuple

    @property
    def kind(self) -> str:
        return self.header['kind']

    @property
    def status(self) -> int:
        return int(self.header.get('status', 200))

    @property
    def headers(self) -> dict:
        return dict(self.header.get('headers') or {})

    @property
    def path(self) -> Path:
        return FIXTURES_DIR / self.provider / (self.case + '.jsonl')

    def body(self, variables: dict | None = None) -> Any:
        """The json body (kind json) after substitution."""
        assert self.kind == 'json', (self.provider, self.case)
        return substitute(self.rows[0]['body'], dict(variables or {}))

    def frames(self, variables: dict | None = None, *, variant: str | None = None) -> list[bytes]:
        """The rendered frames: one per SSE row / NDJSON line / binary chunk, or one frame for json and text."""
        variables = dict(variables or {})
        kind = self.kind
        if kind == 'json':
            return [dumps(self.body(variables)).encode('utf-8')]
        if kind == 'text':
            return [substitute(self.rows[0]['text'], variables).encode('utf-8')]
        if kind == 'sse':
            out = []
            for row in self.rows:
                if row.get('when') and not variables.get(row['when']):
                    continue
                if 'comment' in row:
                    out.append((': ' + row['comment'] + '\n\n').encode('utf-8'))
                    continue
                out.append(sse_frame(row.get('event'), substitute(row['data'], variables)))
            return out
        if kind == 'ndjson':
            return [(dumps(substitute(row['line'], variables)) + '\n').encode('utf-8') for row in self.rows]
        if kind == 'binary':
            chosen = variant or self.rows[0]['variant']
            chunks = [base64.b64decode(row['chunk_b64']) for row in self.rows if row['variant'] == chosen]
            if not chunks:
                raise KeyError(f'{self.provider}/{self.case}: no binary variant {chosen!r}')
            return chunks
        raise ValueError(f'{self.provider}/{self.case}: unknown kind {kind!r}')

    def render(self, variables: dict | None = None, *, variant: str | None = None) -> bytes:
        return b''.join(self.frames(variables, variant=variant))

    def variants(self) -> dict[str, str]:
        """{variant: content type} of a binary fixture, in row order."""
        out: dict[str, str] = {}
        for row in self.rows:
            out.setdefault(row['variant'], row['content_type'])
        return out

    def event_names(self, variables: dict | None = None) -> list[str]:
        """SSE event names in order ('' for an unnamed frame); rows dropped by 'when' are skipped."""
        variables = dict(variables or {})
        return [row.get('event') or '' for row in self.rows
                if self.kind == 'sse' and 'comment' not in row and not (row.get('when') and not variables.get(row['when']))]

    def data_types(self) -> list[str]:
        """The 'type' field of every SSE data object (Anthropic and Responses events), in order."""
        return [row['data'].get('type', '') for row in self.rows if isinstance(row.get('data'), dict)]

    def delta_types(self) -> list[str]:
        """Anthropic content_block_delta delta types, in order."""
        return [row['data']['delta']['type'] for row in self.rows
                if isinstance(row.get('data'), dict) and row['data'].get('type') == 'content_block_delta']


def sse_frame(event: str | None, data: Any) -> bytes:
    payload = data if isinstance(data, str) else dumps(data)
    head = f'event: {event}\n' if event else ''
    return (head + f'data: {payload}\n\n').encode('utf-8')


def parse_sse(raw: bytes) -> list[tuple[str, str]]:
    """[(event name or '', data text)] of an SSE byte stream (LF or CRLF framing); comments and blank frames skipped."""
    text = raw.decode('utf-8').replace('\r\n', '\n')
    out = []
    for block in text.split('\n\n'):
        if not block.strip():
            continue
        event, data = '', []
        for line in block.split('\n'):
            if line.startswith(':'):
                continue
            name, _, value = line.partition(':')
            value = value[1:] if value.startswith(' ') else value
            if name == 'event':
                event = value
            elif name == 'data':
                data.append(value)
        if data or event:
            out.append((event, '\n'.join(data)))
    return out


def _parse_file(path: Path) -> tuple[dict, tuple]:
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines:
        raise ValueError(f'{path.name}: empty fixture')
    header = json.loads(lines[0])
    rows = tuple(json.loads(line) for line in lines[1:])
    missing = [key for key in HEADER_KEYS if key not in header]
    if missing:
        raise ValueError(f'{path.name}: header row lacks {missing}')
    if header.get('kind') not in KINDS:
        raise ValueError(f'{path.name}: kind must be one of {KINDS}')
    if header['kind'] in ('json', 'text') and len(rows) != 1:
        raise ValueError(f'{path.name}: kind {header["kind"]} needs exactly one row')
    if not rows:
        raise ValueError(f'{path.name}: no rows')
    return header, rows


@lru_cache(maxsize=None)
def load(provider: str, case: str) -> Fixture:
    """The fixture tests/fixtures/providers/<provider>/<case>.jsonl (cached; fixtures are read-only)."""
    path = FIXTURES_DIR / provider / (case + '.jsonl')
    header, rows = _parse_file(path)
    return Fixture(provider, case, header, rows)


def all_fixtures() -> Iterator[Fixture]:
    for path in sorted(FIXTURES_DIR.glob('*/*.jsonl')):
        yield load(path.parent.name, path.stem)


def providers() -> list[str]:
    return sorted(p.name for p in FIXTURES_DIR.iterdir() if p.is_dir()) if FIXTURES_DIR.is_dir() else []


__all__ = ['FIXTURES_DIR', 'Fixture', 'HEADER_KEYS', 'KINDS', 'SENTINELS', 'SENTINEL_NAMES', 'all_fixtures', 'dumps', 'load',
           'parse_sse', 'providers', 'sse_frame', 'substitute']
