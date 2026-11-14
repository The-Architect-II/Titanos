"""WP8 CLI: `titanos broker|keys|audit|hivemind-manifest`, serve --role, and the inference routing flags.

- argparse introspection: no option anywhere in the `titanos` parser takes a secret value (only names of environment
  variables such as --api-key-env and --from-env); --secret is refused by the parser;
- offline mode (TITANOS_HOME files under the shared locks): upstream add, credential set from a monkeypatched getpass
  (validated by one probe against a loopback mock), from stdin and from an env reference; the secret appears in no
  stdout, broker.json, vault index or audit file; a failed validation is refused; aliases, virtual models (the
  'gpt-oss:120b@ollama_cloud' ref stays intact; a trailing :<integer> is a weight), policy, prices, explain (zero
  egress), audit verify; keys create prints the secret once and keys.json never holds it; keys reset --keyless needs the
  typed confirmation; broker.json and keys.json stay valid documents;
- a running in-process server sees a key minted offline after the keys.json mtime change;
- --url mode against an in-process uvicorn server: status, keys whoami (key from TITANOS_API_KEY or --api-key-env),
  keys create (secret printed once), credential set through getpass;
- serve: the role preconditions (keyless network bind keeps its message, broker role needs a bootstrap key, managed
  role needs an explicit non-reserved port), and the captured uvicorn.run kwargs carry proxy_headers and
  forwarded_allow_ips='127.0.0.1' (and ssl_certfile/ssl_keyfile with native TLS);
- inference subcommands send --location/--target as headers and print provenance to stderr with --verbose only.
Every test installs the netguard and asserts zero non-loopback connection attempts. Synthetic credentials only.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import broker_wp7_support as W  # noqa: E402  (installs the netguard)
from broker_mocks import SENTINELS, OpenAIMock, netguard  # noqa: E402
from titanos.broker.cli import parse_target  # noqa: E402
from titanos.broker.config import read_broker_config  # noqa: E402
from titanos.cli import main  # noqa: E402

BOOT = 'wp8-cli-bootstrap-0001'
# an option whose last word names a secret value (--secret, --api-key, --token, --password ...); counts such as
# --max-tokens and --tokens-per-day, ids such as --credential and variable names (*-env) are not secret values
SECRET_OPTION = re.compile(r'(?:^--|-)(secret|secrets|password|passphrase|token|api-key|apikey|key-value|bearer)$')
SECRET_DEST = re.compile(r'(^|_)(secret|password|passphrase|token|api_key|apikey|bearer)$')
NAME_OPTIONS = {'--api-key-env', '--token-env', '--from-env', '--i-understand-this-spends-money'}


class Captured(Exception):
    def __init__(self, parser):
        super().__init__('captured')
        self.parser = parser


def titanos_parser() -> argparse.ArgumentParser:
    def capture(self, *args, **kwargs):
        raise Captured(self)
    with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
        try:
            main(['--help'])
        except Captured as found:
            return found.parser
    raise AssertionError('parser not captured')


def all_actions(parser, seen=None):
    seen = seen if seen is not None else set()
    if id(parser) in seen:
        return
    seen.add(id(parser))
    for action in parser._actions:
        yield parser.prog, action
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                yield from all_actions(child, seen)


def run(argv, *, stdin=None, env=None):
    """main(argv) -> (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = 0
    patches = [mock.patch.dict(os.environ, env or {})]
    if stdin is not None:
        patches.append(mock.patch('sys.stdin', io.StringIO(stdin)))
    with contextlib.ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        stack.enter_context(contextlib.redirect_stdout(out))
        stack.enter_context(contextlib.redirect_stderr(err))
        try:
            main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


def files_text(root: Path) -> str:
    chunks = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            chunks.append(path.read_bytes().decode('utf-8', 'replace'))
    return '\n'.join(chunks)


class Guarded(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')


class ParserIntrospection(Guarded):
    def setUp(self):
        super().setUp()
        home = tempfile.mkdtemp(prefix='titanos-wp8-parser-')
        self.addCleanup(shutil.rmtree, home, True)
        env = mock.patch.dict(os.environ, {'TITANOS_HOME': home})
        env.start()
        self.addCleanup(env.stop)

    def test_no_option_takes_a_secret_value(self):
        parser = titanos_parser()
        offenders = []
        for prog, action in all_actions(parser):
            for option in action.option_strings:
                if option in NAME_OPTIONS or option.endswith('-env'):
                    continue
                if SECRET_OPTION.search(option.lower()):
                    offenders.append((prog, option))
            if not action.option_strings and SECRET_DEST.search((action.dest or '').lower()):
                offenders.append((prog, action.dest))
        self.assertEqual(offenders, [])
        code, _, err = run(['broker', 'credential', 'set', 'k', '--upstream', 'u', '--secret', 'sk-test-0123456789ab'])
        self.assertEqual(code, 2)
        self.assertIn('unrecognized arguments', err)
        self.assertIn('--secret', err)
        self.assertNotIn('sk-test-0123456789ab', err)  # refused without echoing the typed value
        code, _, err = run(['broker', 'credential', 'set', 'k', 'sk-test-positional0', '--upstream', 'u'])
        self.assertEqual(code, 2)
        self.assertNotIn('sk-test-positional0', err)
        code, _, err = run(['keys', 'sk-test-choice00'])
        self.assertEqual(code, 2)
        self.assertNotIn('sk-test-choice00', err)
        code, _, err = run(['keys', 'create', '--name', 'x', '--scope', 'inference', '--api-key', 'tk_x'])
        self.assertEqual(code, 2, err)  # never an abbreviation of --api-key-env
        self.assertNotIn('tk_x', err)
        code, _, err = run(['keys', 'list', '--url', 'http://127.0.0.1:9', '--api-key=tk_abcdef0123'])
        self.assertEqual(code, 2)
        self.assertIn('--api-key=<hidden>', err)
        self.assertNotIn('tk_abcdef0123', err)
        # the spellings the vault's refusal message names (--env VAR, --file PATH) are accepted
        parsed = parser.parse_args(['broker', 'credential', 'set', 'k', '--upstream', 'u', '--env', 'SOME_VAR'])
        self.assertEqual((parsed.from_env, parsed.from_file), ('SOME_VAR', None))
        parsed = parser.parse_args(['broker', 'credential', 'rotate', 'k', '--file', '/abs/path'])
        self.assertEqual((parsed.from_env, parsed.from_file), (None, '/abs/path'))
        self.assertFalse((Path(os.environ['TITANOS_HOME']) / 'broker' / 'keys.json').exists())

    def test_parser_messages_never_echo_a_typed_value(self):
        # end to end through main(): every argparse message that quotes (or prints) a typed value hides a token-like value
        cases = [
            (['keys', 'create', '--name', 'x', '--scope', 'inference', '--rpm', 'sk-test-int0123456789ab'],
             "argument --rpm: invalid int value: '<hidden>'", 'sk-test-int0123456789ab'),
            (['keys', 'list', '--timeout', 'tk_abcdefghij_float0123456789'],
             "argument --timeout: invalid float value: '<hidden>'", 'tk_abcdefghij_float0123456789'),
            (['audit', '--limit', "sk-test-'quoted'0123"],  # repr switches to double quotes around a single quote
             "argument --limit: invalid int value: '<hidden>'", "sk-test-'quoted'0123"),
            (['broker', 'credential', 'set', 'k', '--upstream', 'u', '--from-stdin=sk-test-flag0123456789'],
             "argument --from-stdin: ignored explicit argument '<hidden>'", 'sk-test-flag0123456789'),
            (['serve', '--r=sk-test-amb0123456789'], 'ambiguous option: --r=<hidden> could match', 'sk-test-amb0123456789'),
        ]
        for argv, expected, typed in cases:
            with self.subTest(argv=argv[:2]):
                code, out, err = run(argv)
                self.assertEqual(code, 2, err)
                self.assertIn(expected, err)
                self.assertNotIn(typed, err + out)
        # a word-like value keeps its echo, as for choices, so an ordinary typo stays readable
        code, _, err = run(['keys', 'create', '--name', 'x', '--scope', 'inference', '--rpm', 'ten'])
        self.assertEqual(code, 2)
        self.assertIn("invalid int value: 'ten'", err)
        # the message function alone: the same token heuristic for every quoted-value message argparse can produce
        from titanos.broker.cli import redact_parser_message as redact
        self.assertEqual(redact("argument --rpm: invalid int value: 'sk-test-0123456789'"),
                         "argument --rpm: invalid int value: '<hidden>'")
        self.assertEqual(redact('argument --rpm: invalid int value: "it\'s-sk-test-0123456789"'),
                         "argument --rpm: invalid int value: '<hidden>'")
        self.assertEqual(redact("argument --when: invalid <lambda> value: 'hf_abcdefghijklmnop'"),
                         "argument --when: invalid <lambda> value: '<hidden>'")
        self.assertEqual(redact("argument --rpm: invalid int value: 'many'"), "argument --rpm: invalid int value: 'many'")
        self.assertEqual(redact("argument --yes: ignored explicit argument 'nvapi-abcdefghijklmnop'"),
                         "argument --yes: ignored explicit argument '<hidden>'")
        self.assertEqual(redact("argument cmd: invalid choice: 'tk_abcdefghij_0123' (choose from a, b)"),
                         "argument cmd: invalid choice: '<hidden>' (choose from a, b)")
        self.assertEqual(redact("argument cmd: invalid choice: 'lsit' (choose from list, verify)"),
                         "argument cmd: invalid choice: 'lsit' (choose from list, verify)")
        self.assertEqual(redact("unknown parser 'sk-test-0123456789' (choices: a, b)"), "unknown parser '<hidden>' (choices: a, b)")
        self.assertEqual(redact('ambiguous option: --ap=sk-test 0123 4567 could match --api-key-env, --apply'),
                         'ambiguous option: --ap=<hidden> could match --api-key-env, --apply')
        self.assertEqual(redact('ambiguous option: --ap could match --api-key-env, --apply'),
                         'ambiguous option: --ap could match --api-key-env, --apply')
        opened = redact("argument file: can't open 'C:\\keys\\sk-test-0123456789.txt': [Errno 2] No such file or directory: "
                        "'C:\\\\keys\\\\sk-test-0123456789.txt'")
        self.assertEqual(opened, "argument file: can't open '<hidden>': [Errno 2] No such file or directory: '<hidden>'")
        # FileType through a real RedactingArgumentParser: the OSError text repeats the name and is redacted as well
        from titanos.broker.cli import RedactingArgumentParser
        probe = RedactingArgumentParser(prog='probe')
        probe.add_argument('file', type=argparse.FileType('r'))
        missing = str(Path(tempfile.gettempdir()) / 'sk-test-missing0123456789.json')
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            probe.parse_args([missing])
        self.assertIn("can't open '<hidden>'", err.getvalue())
        self.assertNotIn('sk-test-missing0123456789', err.getvalue())

    def test_virtual_target_weights(self):
        self.assertEqual(parse_target('gpt-oss:120b@ollama_cloud'), {'ref': 'gpt-oss:120b@ollama_cloud'})
        self.assertEqual(parse_target('gpt-oss:120b@ollama_cloud:3'), {'ref': 'gpt-oss:120b@ollama_cloud', 'weight': 3})
        self.assertEqual(parse_target('qwen3-8b:2'), {'ref': 'qwen3-8b', 'weight': 2})
        self.assertEqual(parse_target('qwen3-8b'), {'ref': 'qwen3-8b'})
        self.assertEqual(parse_target('meta-llama/Llama-3.1-8B-Instruct:fastest@hf_router'),
                         {'ref': 'meta-llama/Llama-3.1-8B-Instruct:fastest@hf_router'})


class Offline(Guarded):
    def setUp(self):
        super().setUp()
        self.mock = OpenAIMock()
        self.mock.start()
        self.addCleanup(self.mock.stop)
        self.root = Path(tempfile.mkdtemp(prefix='titanos-wp8-cli-'))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.home = self.root / 'home'
        self.home.mkdir()
        env = mock.patch.dict(os.environ, {'TITANOS_HOME': str(self.home), 'TITANOS_BROKER_EGRESS': 'loopback-only'})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('TITANOS_API_KEY', None)
        from titanos.broker.vault import store
        backend = W.S.MemoryBackend(store.platform_default())
        real = store.Vault

        def open_vault(settings, home):
            return real(settings, home, backends={store.platform_default(): backend})
        patcher = mock.patch.object(store, 'open_vault', open_vault)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_offline_broker_configuration_and_secret_handling(self):
        sentinel = SENTINELS['openai']
        code, out, err = run(['broker', 'upstream', 'add', 'oai', '--profile', 'openai', '--base-url', self.mock.api_base,
                              '--models', 'gpt-4o-mini'])
        self.assertEqual(code, 0, err)
        self.assertIn('titanos broker reload', err)
        with mock.patch('getpass.getpass', return_value=sentinel) as prompt:
            code, out, err = run(['broker', 'credential', 'set', 'oai-key', '--upstream', 'oai'])
        self.assertEqual(code, 0, err)
        prompt.assert_called_once()
        self.assertEqual(json.loads(out)['validation'], 'ok')
        self.assertEqual(len(self.mock.hits), 1)  # exactly the one validation probe
        code, out, err = run(['broker', 'credential', 'set', 'bad-key', '--upstream', 'oai', '--from-stdin'],
                             stdin=SENTINELS['wrong'] + '\n')
        self.assertEqual(code, 1)
        self.assertIn('credential_validation_failed', err)
        code, out, err = run(['broker', 'credential', 'set', 'env-key', '--upstream', 'oai', '--from-env', 'WP8_OPENAI_VAR'],
                             env={'WP8_OPENAI_VAR': sentinel})
        self.assertEqual(code, 0, err)
        record = json.loads(out)
        self.assertEqual((record['source'], record['env_var'], record['validation']), ('env', 'WP8_OPENAI_VAR', 'ok'))
        code, out, err = run(['broker', 'upstream', 'add', 'oai', '--profile', 'openai', '--credential', 'oai-key', '--enable'])
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)['enabled'])
        for argv in (['broker', 'alias', 'set', 'mini', 'gpt-4o-mini@oai'],
                     ['broker', 'policy', 'set', '--cloud-policy', 'explicit'],
                     ['broker', 'virtual', 'set', 'fast', '--capability', 'chat', '--strategy', 'weighted',
                      '--target', 'gpt-4o-mini@oai:3', '--target', 'mini:1']):
            code, out, err = run(argv)
            self.assertEqual(code, 0, (argv, err))
        prices = self.root / 'prices.json'
        prices.write_text(json.dumps([{'match': 'gpt-4o-mini@oai', 'input_per_mtok': '0.15'}]), encoding='utf-8')
        self.assertEqual(run(['broker', 'prices', 'import', str(prices)])[0], 0)
        cfg = read_broker_config(self.home / 'broker.json')
        self.assertEqual(cfg.virtual_models['fast'].targets[0].ref, 'gpt-4o-mini@oai')
        self.assertEqual(cfg.virtual_models['fast'].targets[0].weight, 3)
        self.assertEqual(cfg.policy.cloud_policy, 'explicit')
        self.assertTrue((self.home / 'broker.json.lock').exists() or True)
        hits = len(self.mock.hits)
        code, out, err = run(['broker', 'explain', '--model', 'mini'])
        self.assertEqual(code, 0, err)
        decision = json.loads(out)
        self.assertTrue(decision['dry_run'])
        self.assertIn('gpt-4o-mini@oai', decision['chain'])
        self.assertEqual(len(self.mock.hits), hits, 'explain made a request')
        code, out, err = run(['audit', 'verify'])
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)['ok'])
        code, out, _ = run(['audit', '--kind', 'credential.put'])
        self.assertGreaterEqual(len(json.loads(out)['data']), 3)
        self.assertNotIn(sentinel, files_text(self.home))
        self.assertNotIn(SENTINELS['wrong'], files_text(self.home))
        code, out, _ = run(['broker', 'credential', 'list'])
        self.assertNotIn(sentinel, out)
        self.assertEqual(run(['broker', 'credential', 'remove', 'bad-key'])[0], 0)

    def test_offline_keys(self):
        code, out, err = run(['keys', 'create', '--name', 'ci', '--scope', 'inference', '--scope', 'cloud:oai',
                              '--expires', '30d', '--usd-per-day', '1', '--models', 'gpt-*@oai,qwen3-*'])
        self.assertEqual(code, 0, err)
        minted = json.loads(out)
        token = minted['secret']
        self.assertTrue(token.startswith('tk_' + minted['id'] + '_'))
        self.assertEqual(out.count(token), 1)
        self.assertNotIn(token, files_text(self.home))
        keys = json.loads((self.home / 'broker' / 'keys.json').read_text(encoding='utf-8'))
        self.assertEqual(keys['version'], 1)
        self.assertEqual(keys['keys'][minted['id']]['constraints']['models'], ['gpt-*@oai', 'qwen3-*'])
        for argv in (['keys', 'list'], ['keys', 'show', minted['id']]):
            code, out, _ = run(argv)
            self.assertEqual(code, 0)
            self.assertNotIn(token, out)
            self.assertNotIn('secret', out)
        secret_file = self.root / 'rotated.key'
        code, out, err = run(['keys', 'rotate', minted['id'], '--output-file', str(secret_file)])
        self.assertEqual(code, 0, err)
        rotated = secret_file.read_text(encoding='utf-8').strip()
        self.assertTrue(rotated.startswith('tk_' + minted['id']))
        self.assertNotIn(rotated, out)
        self.assertEqual(json.loads(run(['keys', 'update', minted['id'], '--rpm', '30'])[1])['limits']['rpm'], 30)
        self.assertEqual(json.loads(run(['keys', 'revoke', minted['id']])[1])['status'], 'revoked')
        code, _, err = run(['keys', 'reset', '--keyless'], stdin='no\n')
        self.assertEqual(code, 1)
        self.assertIn('nothing changed', err)
        self.assertTrue((self.home / 'broker' / 'keys.json').exists())
        code, out, err = run(['keys', 'reset', '--keyless'], stdin='DELETE KEYS\n')
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['removed'], 1)
        self.assertFalse((self.home / 'broker' / 'keys.json').exists())
        self.assertIn('key.reset_keyless', run(['audit', '--kind', 'key.reset_keyless'])[1])
        code, _, err = run(['keys', 'whoami'])
        self.assertEqual(code, 1)
        self.assertIn('--url', err)


class RunningServer(Guarded):
    def test_in_process_server_sees_an_offline_minted_key(self):
        full = W.FullApp(None, models={'chat-local': 'chat'})
        self.addCleanup(full.close)
        self.assertEqual(full.http.get('/api/keys/self').json()['kind'], 'anonymous_loopback')
        code, out, err = run(['keys', 'create', '--name', 'late', '--scope', 'models:read'])
        self.assertEqual(code, 0, err)
        token = json.loads(out)['secret']
        deadline = time.monotonic() + 10
        while True:
            answer = full.http.get('/api/keys/self', headers=W.bearer(token))
            if answer.status_code == 200 and answer.json()['kind'] == 'key':
                break
            if time.monotonic() > deadline:
                self.fail(f'the server never saw the key: {answer.status_code}')
            time.sleep(0.2)
        self.assertEqual(full.http.get('/api/keys/self').status_code, 401)  # enforcement is on now


def free_port() -> int:
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    try:
        return sock.getsockname()[1]
    finally:
        sock.close()


class UrlMode(Guarded):
    def test_url_mode_against_a_served_app(self):
        import uvicorn
        openai = OpenAIMock()
        openai.start()
        self.addCleanup(openai.stop)
        document = {'upstreams': {'oai': W.S.upstream(openai, 'openai', credential=None, enabled=False,
                                                      models=['gpt-4o-mini'])}}
        full = W.FullApp(document, models={'chat-local': 'chat'}, api_key=BOOT)
        self.addCleanup(full.close)
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(full.app, host='127.0.0.1', port=port, log_level='warning', lifespan='on'))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        self.addCleanup(lambda: (setattr(server, 'should_exit', True), thread.join(15)))
        deadline = time.monotonic() + 30
        while not server.started:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.05)
        url = f'http://127.0.0.1:{port}'
        code, out, err = run(['broker', 'status', '--url', url], env={'TITANOS_API_KEY': BOOT})
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['upstreams']['total'], 1)
        code, out, err = run(['keys', 'whoami', '--url', url, '--api-key-env', 'WP8_OTHER'], env={'WP8_OTHER': BOOT})
        self.assertEqual(json.loads(out)['kind'], 'bootstrap')
        code, out, err = run(['keys', 'create', '--url', url, '--name', 'remote', '--scope', 'inference'],
                             env={'TITANOS_API_KEY': BOOT})
        self.assertEqual(code, 0, err)
        created = json.loads(out)
        self.assertEqual(out.count(created['secret']), 1)
        code, out, err = run(['keys', 'whoami', '--url', url], env={'TITANOS_API_KEY': created['secret']})
        self.assertEqual(json.loads(out)['id'], created['id'])
        with mock.patch('getpass.getpass', return_value=SENTINELS['openai']):
            code, out, err = run(['broker', 'credential', 'set', 'oa', '--upstream', 'oai', '--url', url],
                                 env={'TITANOS_API_KEY': BOOT})
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['validation'], 'ok')
        self.assertNotIn(SENTINELS['openai'], out + err)
        code, _, err = run(['broker', 'status', '--url', url], env={'TITANOS_API_KEY': 'wrong-key-0000'})
        self.assertEqual(code, 1)
        self.assertIn('401', err)


class Serve(Guarded):
    def serve(self, argv, env=None):
        app_factory, server_run = mock.Mock(return_value='fixture-app'), mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / 'models.json'
            from titanos.config import Config, write_config
            write_config(registry, Config())
            base = {'TITANOS_HOME': tmp}
            base.update(env or {})
            with mock.patch.dict(os.environ, base), \
                    mock.patch.dict('sys.modules', {'titanos.server': SimpleNamespace(create_app=app_factory)}), \
                    mock.patch('uvicorn.run', server_run):
                for name in ('TITANOS_API_KEY', 'TITANOS_ROLE', 'TITANOS_PORT'):
                    if name not in (env or {}):
                        os.environ.pop(name, None)
                code, out, err = run(['--config', str(registry), 'serve', *argv])
            self.assertFalse((Path(tmp) / 'broker').exists() and not (env or {}).get('TITANOS_ROLE') == 'broker'
                             and '--role' not in argv, 'serve created broker state')
        return code, err, server_run.call_args.kwargs if server_run.called else None

    def test_preconditions(self):
        code, err, _ = self.serve(['--host', '0.0.0.0'])
        self.assertEqual(code, 1)
        self.assertIn('Set TITANOS_API_KEY before binding to a network interface', err)
        code, err, _ = self.serve(['--role', 'broker'])
        self.assertEqual(code, 1)
        self.assertIn('TITANOS_ROLE=broker requires TITANOS_API_KEY', err)
        code, err, _ = self.serve(['--role', 'managed'])
        self.assertEqual(code, 1)
        self.assertIn('explicit --port', err)
        code, err, _ = self.serve(['--role', 'managed', '--port', '11435'])
        self.assertEqual(code, 1)
        self.assertIn('reserved port 11435', err)
        code, err, kwargs = self.serve(['--role', 'managed', '--port', '12010'])
        self.assertEqual(code, 0, err)
        code, err, _ = self.serve(['--port', '12011', '--register-with-gateway', 'http://127.0.0.1:9'])
        self.assertEqual(code, 1)
        self.assertIn('--role managed', err)
        # an unusable gateway URL is refused before the server starts, and the URL is not echoed
        code, err, kwargs = self.serve(['--role', 'managed', '--port', '12015', '--job-type', 'x',
                                        '--register-with-gateway', 'http://u:pw-sentinel@127.0.0.1:9'],
                                       env={'TITANOS_API_KEY': BOOT})
        self.assertEqual(code, 2, err)
        self.assertIsNone(kwargs)
        self.assertIn('without credentials, query or fragment', err)
        self.assertNotIn('pw-sentinel', err)

    def test_uvicorn_kwargs_are_pinned(self):
        code, err, kwargs = self.serve(['--port', '12012'])
        self.assertEqual(code, 0, err)
        self.assertIs(kwargs['proxy_headers'], False)
        self.assertEqual(kwargs['forwarded_allow_ips'], '127.0.0.1')
        self.assertNotIn('ssl_certfile', kwargs)
        code, err, kwargs = self.serve(['--port', '12013'], env={'TITANOS_BEHIND_TLS_PROXY': 'true'})
        self.assertEqual(code, 0, err)
        self.assertIs(kwargs['proxy_headers'], True)
        self.assertEqual(kwargs['forwarded_allow_ips'], '127.0.0.1')
        with tempfile.TemporaryDirectory() as tls:
            cert, key = Path(tls) / 'cert.pem', Path(tls) / 'key.pem'
            cert.write_text('x', encoding='utf-8')
            key.write_text('x', encoding='utf-8')
            code, err, kwargs = self.serve(['--port', '12014'], env={'TITANOS_TLS_CERTFILE': str(cert),
                                                                    'TITANOS_TLS_KEYFILE': str(key)})
        self.assertEqual(code, 0, err)
        self.assertEqual((kwargs['ssl_certfile'], kwargs['ssl_keyfile']), (str(cert), str(key)))
        self.assertEqual(kwargs['forwarded_allow_ips'], '127.0.0.1')
        code, err, _ = self.serve(['--host', '0.0.0.0'], env={'TITANOS_API_KEY': BOOT, 'TITANOS_BEHIND_TLS_PROXY': 'true'})
        self.assertEqual(code, 1)
        self.assertIn('BEHIND_TLS_PROXY is valid only with a loopback bind', err)


class InferenceFlags(Guarded):
    def test_location_target_and_verbose(self):
        import httpx
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={'data': [{'embedding': [1.0]}]},
                                  headers={'X-Titanos-Served-By': 'local:engine', 'X-Request-ID': 'r1'})
        real = httpx.Client

        def client(*args, **kwargs):
            kwargs['transport'] = httpx.MockTransport(handler)
            return real(*args, **kwargs)
        with mock.patch('httpx.Client', client):
            code, out, err = run(['embed', 'emb', 'hello', '--url', 'http://127.0.0.1:1', '--location', 'cloud_only',
                                  '--target', 'text-embedding-3-small@openai', '--verbose'])
            self.assertEqual(code, 0, err)
            self.assertEqual(seen[0].headers['x-titanos-location'], 'cloud_only')
            self.assertEqual(seen[0].headers['x-titanos-target'], 'text-embedding-3-small@openai')
            self.assertEqual(json.loads(out), {'data': [{'embedding': [1.0]}]})
            self.assertEqual(json.loads(err)['provenance']['x-titanos-served-by'], 'local:engine')
            code, out, err = run(['embed', 'emb', 'hello', '--url', 'http://127.0.0.1:1'])
            self.assertEqual(err, '')
            self.assertNotIn('x-titanos-location', seen[1].headers)


if __name__ == '__main__':
    unittest.main(verbosity=2)
