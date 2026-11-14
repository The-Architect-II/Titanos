"""OPS (d): the documented default key mint lets stock SDKs list models (broker/scopes.DEFAULT_KEY_SCOPES).

- `titanos keys create --name NAME` without --scope mints inference + models:read (offline keys.json and the printed
  record); the --scope help names the default;
- over the integrated app: a key with exactly those scopes answers GET /v1/models 200 with the local models (what a stock
  OpenAI SDK's models.list() calls first), GET /v1/models/{name} 200 and a chat completion is not scope-denied, while
  keys:admin routes stay 403; a key holding only `inference` (the old documented example) is 403 scope_denied on
  GET /v1/models, the failure the default removes;
- docs: BROKER.md documents the default (the sentence names both scopes next to `--scope`), and every `titanos keys create`
  example in docs/*.md and README.md that grants `inference` also grants `models:read`, so a copied example yields a key
  that works with an SDK.
Loopback only; the netguard asserts zero non-loopback connection attempts.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import netguard  # noqa: E402
from titanos.broker.scopes import DEFAULT_KEY_SCOPES, ROUTE_SCOPES  # noqa: E402
from titanos.cli import main  # noqa: E402

BOOT = 'ops-default-key-bootstrap-0001'
CHAT = {'model': 'chat-local', 'messages': [{'role': 'user', 'content': 'hi'}]}


def run(argv, *, env=None):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with mock.patch.dict(os.environ, env or {}):
        os.environ.pop('TITANOS_API_KEY', None)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                main(argv)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


class Guarded(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')


class OfflineMint(Guarded):
    def test_create_without_scope_mints_the_sdk_default(self):
        self.assertEqual(DEFAULT_KEY_SCOPES, ('inference', 'models:read'))
        self.assertEqual(ROUTE_SCOPES[('GET', '/v1/models')], 'models:read')
        home = tempfile.mkdtemp(prefix='titanos-ops-default-key-')
        self.addCleanup(shutil.rmtree, home, True)
        code, out, err = run(['keys', 'create', '--name', 'sdk'], env={'TITANOS_HOME': home})
        self.assertEqual(code, 0, err)
        record = json.loads(out)
        self.assertEqual(record['scopes'], list(DEFAULT_KEY_SCOPES))
        self.assertTrue(record['secret'].startswith('tk_'))
        stored = json.loads((Path(home) / 'broker' / 'keys.json').read_text(encoding='utf-8'))
        rows = stored['keys'] if isinstance(stored, dict) and isinstance(stored.get('keys'), (list, dict)) else stored
        text = json.dumps(rows)
        self.assertIn('"models:read"', text)
        self.assertNotIn(record['secret'], text)
        code, out, err = run(['keys', 'create', '--name', 'narrow', '--scope', 'inference'], env={'TITANOS_HOME': home})
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['scopes'], ['inference'])  # an explicit --scope list is taken as given
        code, out, _ = run(['keys', 'create', '--help'])
        self.assertEqual(code, 0)
        self.assertIn('models:read', out)
        self.assertNotIn('required', out.split('--scope', 1)[1].split('--models', 1)[0])


class SdkKey(Guarded):
    def setUp(self):
        super().setUp()
        self.full = W.FullApp(None, models={'chat-local': 'chat', 'emb': 'embeddings'}, api_key=BOOT)
        self.addCleanup(self.full.close)
        self.c = self.full.http

    def test_default_key_lists_models_and_infers(self):
        _, token = self.full.mint(*DEFAULT_KEY_SCOPES, name='sdk')
        listing = self.c.get('/v1/models', headers=W.bearer(token))
        self.assertEqual(listing.status_code, 200, listing.text[:300])
        self.assertEqual({row['id'] for row in listing.json()['data']}, {'chat-local', 'emb'})
        self.assertEqual(listing.json()['object'], 'list')
        self.assertEqual(self.c.get('/v1/models/chat-local', headers=W.bearer(token)).status_code, 200)
        chat = self.c.post('/v1/chat/completions', json=CHAT, headers=W.bearer(token))
        self.assertNotEqual(chat.status_code, 403, chat.text[:300])
        self.assertNotIn(chat.status_code, (401,), chat.text[:300])
        denied = self.c.get('/api/keys', headers=W.bearer(token))
        self.assertEqual((denied.status_code, denied.json()['error']['code']), (403, 'scope_denied'))

    def test_inference_only_key_cannot_list_models(self):
        _, token = self.full.mint('inference', name='old-example')
        answer = self.c.get('/v1/models', headers=W.bearer(token))
        self.assertEqual((answer.status_code, answer.json()['error']['code']), (403, 'scope_denied'), answer.text[:300])


class Documentation(unittest.TestCase):
    EXAMPLE = re.compile(r'titanos keys create[^\n`]*')

    def test_default_is_documented(self):
        text = (ROOT / 'docs' / 'BROKER.md').read_text(encoding='utf-8')
        sentence = next((s for s in re.split(r'(?<=[.!?])\s', text)
                         if 'without `--scope`' in s and 'models:read' in s and 'inference' in s), None)
        self.assertIsNotNone(sentence, 'BROKER.md explains that keys create without --scope mints inference + models:read')
        for scope in DEFAULT_KEY_SCOPES:
            self.assertIn('`' + scope + '`', sentence)

    def test_inference_examples_grant_models_read(self):
        offenders = []
        for path in sorted([*ROOT.glob('docs/*.md'), ROOT / 'README.md']):
            for line_number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
                for example in self.EXAMPLE.findall(line):
                    if '--scope inference' in example and 'models:read' not in example and 'admin' not in example:
                        offenders.append(f'{path.relative_to(ROOT).as_posix()}:{line_number}: {example.strip()}')
        self.assertEqual(offenders, [], 'a documented inference key example lacks models:read')


if __name__ == '__main__':
    unittest.main(verbosity=2)
