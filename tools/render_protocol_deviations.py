"""Render docs/PROTOCOL_DEVIATIONS.md from docs/protocol_deviations.json.

The JSON is the maintained parity/deviation map for the provider-compatible adapters (INT-06) and the Ollama/llama.cpp option
parity map (CFG-03). tests/check_protocol_deviations.py asserts the mechanically checkable rows against the live app and fails
when the Markdown drifts from this rendering. Run: python tools/render_protocol_deviations.py [--check]"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs' / 'protocol_deviations.json'
TARGET = ROOT / 'docs' / 'PROTOCOL_DEVIATIONS.md'
TITLES = {'ollama': 'Ollama shapes (`/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/embed`, `/api/embeddings`)',
          'anthropic': 'Anthropic Messages (`/v1/messages`)', 'elevenlabs': 'ElevenLabs text-to-speech (`/v1/text-to-speech/{voice_id}`)',
          'openai': 'OpenAI-compatible `/v1` routes', 'option-parity': 'Option parity (Ollama options, llama.cpp flags, Titanos settings)',
          'broker-openai': 'Broker: OpenAI-compatible routes to cloud and router targets',
          'broker-anthropic': 'Broker: Anthropic Messages to and from cloud targets',
          'broker-gemini': 'Broker: Gemini routes (`/v1beta`) and Gemini targets',
          'broker-elevenlabs': 'Broker: ElevenLabs routes and ElevenLabs targets',
          'broker-ollama': 'Broker: Ollama routes and Ollama Cloud targets'}
ORDER = ['openai', 'ollama', 'anthropic', 'elevenlabs', 'broker-openai', 'broker-anthropic', 'broker-gemini', 'broker-elevenlabs', 'broker-ollama',
         'option-parity']


def cell(text):
    return str(text or '').replace('|', '\\|').replace('\n', ' ').strip()


def render(data):
    rows = data['rows']
    lines = ['# Protocol parity and deviation map', '',
             f"Generated from `docs/protocol_deviations.json` (schema {data['schema_version']}, basis {data['generated']}) by `tools/render_protocol_deviations.py`; do not edit by hand. "
             'Each row states what Titanos does with one upstream request field, response field or route, with the code reference that binds it. '
             f"Basis: {data['basis']}", '',
             '## Behaviours', ''] + [f"- `{k}`: {v}" for k, v in data['behaviours'].items()] + ['', '## Checks', ''] + [f"- `{k}`: {v}" for k, v in data['checks'].items()] + ['']
    counts = Counter(r['adapter'] for r in rows)
    lines += ['## Summary', '', '| Adapter | Rows | Mechanically asserted | Verified doc errors (still open) |', '|---|---|---|---|']
    for adapter in ORDER:
        sub = [r for r in rows if r['adapter'] == adapter]
        lines.append(f"| {adapter} | {len(sub)} | {sum(r['check'] != 'documented' for r in sub)} | {sum(bool(r.get('verified_doc_error')) for r in sub)} |")
    lines.append('')
    for adapter in ORDER:
        sub = [r for r in rows if r['adapter'] == adapter]
        if not sub:
            continue
        lines += [f"## {TITLES.get(adapter, adapter)}", '', f"{len(sub)} rows.", '', '| Route | Field | Behaviour | Check | Code | Note |', '|---|---|---|---|---|---|']
        for r in sub:
            check = r['check'] + (' (' + ', '.join(f'`{e}`' for e in r['evidence']) + ')' if r.get('evidence') else '')
            lines.append(f"| `{cell(r['route'])}` | `{cell(r['field'])}` | {r['behaviour']} | {check} | {cell(r.get('code_ref'))} | {cell(r.get('note'))} |")
        lines.append('')
    lines += ['## Scope', '', 'This map records request/response field semantics and route presence for the provider-compatible wire formats. It is code-bound documentation, not vendor-SDK compatibility qualification '
              '(the SDK lanes `tests/check_broker_sdk.py` and `tests/check_broker_sdk_vendors.py` drive stock clients against loopback mocks). '
              'Inbound credentials: when keys are enforced, `Authorization: Bearer` is accepted on every gated route, and the vendor headers are accepted on their own route family only '
              '(`accepted_auth` rows): `x-api-key` on `/v1/messages*`, `xi-api-key` on `/v1/text-to-speech*`, `/v1/speech-to-text` and `/v1/voices*`, and `x-goog-api-key` on `/v1beta/*`; '
              'a `?key=` query credential is refused 401 in every mode, and in keyless loopback mode presented credentials are ignored. `anthropic-version` is not validated locally. '
              'The broker-* sections record what the cloud broker does when a request is routed to a cloud or router target; local targets keep the local adapter rows. '
              'Model and hardware execution evidence lives in the capability matrix; broker support evidence lives in `docs/BROKER_MATRIX.md`.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='exit 1 when docs/PROTOCOL_DEVIATIONS.md differs from the rendering')
    args = parser.parse_args()
    text = render(json.loads(SOURCE.read_text(encoding='utf-8')))
    if args.check:
        current = TARGET.read_text(encoding='utf-8') if TARGET.exists() else ''
        if current.replace('\r\n', '\n') != text:
            raise SystemExit('docs/PROTOCOL_DEVIATIONS.md is out of date; run python tools/render_protocol_deviations.py')
        print('PROTOCOL_DEVIATIONS.md matches protocol_deviations.json')
        return
    TARGET.write_text(text, encoding='utf-8')
    print(f'wrote {TARGET} ({len(text.splitlines())} lines)')


if __name__ == '__main__':
    main()
