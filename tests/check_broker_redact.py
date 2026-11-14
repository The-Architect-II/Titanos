"""Broker redaction (WP1): every secret pattern, the exact-value registry (8-character minimum), nested scrub_obj, the
global LogRecord factory (records from the 'httpx' child logger reaching a root handler), handler filters restored after
logging.config.dictConfig, exc_text/args scrubbing, a uvicorn.access-shaped '?key=' record, and child_env. Synthetic
values only; no server, model or network."""
import io
import logging
import logging.config
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos.broker import redact  # noqa: E402
from titanos.broker.redact import REDACTED, REGISTRY, SecretRegistry, child_env, scrub, scrub_obj  # noqa: E402

HEX = os.urandom(16).hex()
SAMPLES = {
    'openai': 'sk-' + HEX[:24],
    'openai_project': 'sk-proj-' + HEX,
    'anthropic': 'sk-ant-api03-' + HEX,
    'xai': 'xai-' + HEX[:24],
    'groq': 'gsk_' + HEX[:24],
    'nvidia': 'nvapi-' + HEX,
    'huggingface': 'hf_' + HEX[:24],
    'google': 'AIza' + (HEX + HEX)[:35],
    'titanos_key': 'tk_abcdefgh23_' + (HEX + HEX)[:43],
    'jwt': 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmaXh0dXJlIn0.' + HEX[:20],
}
HEADER_FORMS = ['Authorization: Bearer {v}', 'authorization=Basic {v}', 'x-api-key: {v}', 'xi-api-key: {v}', 'api-key: {v}',
                'x-goog-api-key: {v}', '{{"Authorization": "Bearer {v}"}}', 'Proxy-Authorization: {v}', 'bearer {v}']
QUERY_FORMS = ['/v1beta/models/m:generateContent?key={v}', 'https://h.example.invalid/x?a=1&api_key={v}&b=2',
               '/cb?access_token={v}#frag']


class Patterns(unittest.TestCase):
    def test_every_pattern_is_scrubbed(self):
        for name, value in SAMPLES.items():
            with self.subTest(name=name):
                text = f'upstream said: {value} (end)'
                self.assertTrue(redact.looks_secret(value))
                out = scrub(text)
                self.assertNotIn(value, out)
                self.assertIn(REDACTED, out)
                self.assertTrue(out.startswith('upstream said: '))

    def test_header_and_query_forms_keep_labels(self):
        opaque = 'opaque' + HEX[:12]
        for form in HEADER_FORMS:
            with self.subTest(form=form):
                out = scrub(form.format(v=opaque))
                self.assertNotIn(opaque, out)
                self.assertIn(REDACTED, out)
        for form in QUERY_FORMS:
            with self.subTest(form=form):
                out = scrub(form.format(v=opaque))
                self.assertNotIn(opaque, out)
                self.assertRegex(out, r'(key|api_key|access_token)=<redacted>')
        self.assertTrue(redact.QUERY_CREDENTIAL.search('/x?key=' + opaque))

    def test_minimum_lengths_and_boundaries(self):
        for text in ('sk-short', 'task-' + HEX[:24], 'hf_short', 'xai-short', 'gsk_tiny', 'AIzaShort', 'max_tokens: 5',
                     'model gpt-4o-mini', 'tk_abc'):
            with self.subTest(text=text):
                self.assertEqual(scrub(text), text)
        self.assertEqual(scrub('mask-' + HEX[:24]), 'mask-' + HEX[:24])

    def test_scheme_rule_needs_a_token(self):
        # ratified R2: 'Bearer'/'Basic' is redacted only before a token of at least 8 token68 characters
        for text in ('Using basic auth for the proxy', 'bearer token missing', 'the Bearer scheme', 'BASIC usage: see docs',
                     'Bearer abc.def', 'basic, then bearer'):
            with self.subTest(text=text):
                self.assertEqual(scrub(text), text)
        # the boundary as ratified: any word of 8+ token characters after the scheme counts as a token
        self.assertEqual(scrub('Basic authentication failed'), 'Basic <redacted> failed')
        token = 'Zm9v' + HEX[:12] + '=='
        for text in (f'Bearer {token}', f'basic {token}', f'BEARER {HEX[:8]}', f'sent bearer {HEX}.x/y+z~ ok'):
            with self.subTest(text=text):
                out = scrub(text)
                self.assertIn(REDACTED, out)
                self.assertNotIn(HEX[:8], out)
        self.assertEqual(scrub(f'Bearer {token} then basic auth'), 'Bearer <redacted> then basic auth')

    def test_owned_command_rules_are_included(self):
        opaque = 'opaque' + HEX[:12]
        for text in (f'token={opaque}', f'password: "{opaque} more"', f'--api-key {opaque}', f'secret={opaque}'):
            with self.subTest(text=text):
                self.assertNotIn(opaque, scrub(text))
        child = redact.scrub_child('failed reading ' + SAMPLES['openai'] + ' at /opt/fixture-dir/file.txt')
        self.assertNotIn(SAMPLES['openai'], child)
        self.assertNotIn('/opt/fixture-dir', child)


class Registry(unittest.TestCase):
    def test_exact_values_without_pattern(self):
        registry = SecretRegistry()
        value = 'plain-words-' + HEX[:10]
        self.assertFalse(redact.looks_secret('x ' + value))
        registry.add(value)
        self.assertEqual(registry.scrub(f'a {value} b {value}'), f'a {REDACTED} b {REDACTED}')
        self.assertTrue(registry.contains_in('..' + value + '..'))
        longer = value + '-longer'
        registry.add(longer)
        self.assertEqual(registry.scrub(longer), REDACTED)  # longest first: no partial suffix survives
        registry.discard(value)
        registry.discard(longer)
        self.assertEqual(registry.scrub(value), value)
        self.assertEqual(len(registry), 0)

    def test_short_values_refused(self):
        registry = SecretRegistry()
        for value in ('', 'a', '1234567', None, 12345678):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    registry.add(value)
        registry.add('12345678')
        self.assertEqual(len(registry), 1)

    def test_global_registry_feeds_scrub(self):
        value = 'global-' + HEX[:12]
        REGISTRY.add(value)
        try:
            self.assertTrue(redact.looks_secret(value))
            self.assertEqual(scrub('k=' + value), 'k=' + REDACTED)
        finally:
            REGISTRY.discard(value)
        self.assertEqual(scrub('k=' + value), 'k=' + value)


class ScrubObj(unittest.TestCase):
    def test_nested_structures(self):
        secret = SAMPLES['openai']
        document = {'a': [1, 2.5, None, True, {'b': ('x', secret, {secret: 'key-as-secret'})}],
                    'c': {'d': {'e': {'f': 'Authorization: Bearer ' + 'z' * 20}}}, 'g': {secret}, 'h': b'bytes stay'}
        out = scrub_obj(document)
        self.assertNotIn(secret, repr(out))
        self.assertNotIn('z' * 20, repr(out))
        self.assertEqual(out['a'][:4], [1, 2.5, None, True])
        self.assertIsInstance(out['a'][4]['b'], tuple)
        self.assertIsInstance(out['g'], set)
        self.assertEqual(out['h'], b'bytes stay')
        self.assertEqual(document['a'][4]['b'][1], secret)  # the input is not mutated
        deep = current = {}
        for _ in range(50):
            current['n'] = {}
            current = current['n']
        scrub_obj(deep)  # bounded depth, no recursion error


class Logging(unittest.TestCase):
    def setUp(self):
        redact.uninstall_logging_redaction()
        self.addCleanup(redact.uninstall_logging_redaction)
        self.root = logging.getLogger()
        self.saved = (list(self.root.handlers), self.root.level)
        self.addCleanup(self.restore)
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(logging.Formatter('%(name)s %(message)s'))
        self.root.handlers = [self.handler]
        self.root.setLevel(logging.INFO)

    def restore(self):
        self.root.handlers, level = self.saved
        self.root.setLevel(level)

    def test_child_logger_records_are_scrubbed_at_root_handler(self):
        redact.install_logging_redaction()
        secret = SAMPLES['nvidia']
        child = logging.getLogger('httpx')
        child.setLevel(logging.INFO)
        child.info('HTTP Request: %s %s', 'POST', 'https://h.example.invalid/v1/chat?key=' + secret)
        logging.getLogger('httpcore.connection').info('send %r', {'authorization': 'Bearer ' + secret})
        logging.getLogger('titanos.broker').warning('literal %s message ' + secret, 'x')
        output = self.stream.getvalue()
        self.assertNotIn(secret, output)
        self.assertEqual(output.count(REDACTED), 3, output)
        self.assertIn('httpx HTTP Request: POST https://h.example.invalid/v1/chat?key=<redacted>', output)

    def test_labelled_placeholders_are_formatted_before_scrubbing(self):
        # An unshaped, unregistered value after a credential label is caught only by formatting first; scrubbing the
        # format string alone breaks formatting, drops the record and prints the raw arguments to stderr.
        redact.install_logging_redaction()
        value = 'UNSHAPEDsynthetic' + HEX[:16]
        shaped = SAMPLES['openai']
        log = logging.getLogger('httpx')
        log.setLevel(logging.INFO)
        cases = [('Authorization: Bearer %s', (value,)), ('x-api-key: %s', (value,)), ('x-api-key: %r', (value,)),
                 ('api_key=%s', (shaped,)), ('url ?key=%s&x=1', (shaped,)), ('token: %s', (value,)),
                 ('mapped authorization=%(v)s', ({'v': value},)), ('broken %s %s', (value,))]
        stderr = io.StringIO()
        with patch.object(sys, 'stderr', stderr):
            for fmt, args in cases:
                log.info(fmt, *args)
        output = self.stream.getvalue()
        self.assertEqual(stderr.getvalue(), '')  # no '--- Logging error ---' with raw arguments
        self.assertNotIn(value, output)
        self.assertNotIn(shaped, output)
        self.assertEqual(len(output.splitlines()), len(cases), output)  # no record was dropped
        self.assertIn('httpx Authorization=<redacted>', output)
        self.assertIn('httpx url ?key=<redacted>&x=1', output)
        self.assertIn('httpx broken %s %s [log arguments withheld', output)
        # ordinary placeholders are untouched and keep their arguments
        record = log.makeRecord('httpx', logging.INFO, __file__, 1, 'loaded %s in %d ms', ('model-a', 12), None)
        self.assertEqual((record.msg, record.args), ('loaded %s in %d ms', ('model-a', 12)))

    def test_factory_is_idempotent_and_handler_filters_reapply_after_dictconfig(self):
        redact.install_logging_redaction()
        factory = logging.getLogRecordFactory()
        redact.install_logging_redaction()
        self.assertIs(logging.getLogRecordFactory(), factory)
        self.assertEqual(sum(isinstance(f, redact.RedactingFilter) for f in self.handler.filters), 1)
        logging.config.dictConfig({'version': 1, 'disable_existing_loggers': False,
                                   'handlers': {'mem': {'class': 'logging.StreamHandler', 'stream': 'ext://sys.stderr'}},
                                   'loggers': {'uvicorn.access': {'handlers': ['mem'], 'level': 'INFO',
                                                                  'propagate': False}}})
        access = logging.getLogger('uvicorn.access')
        self.addCleanup(lambda: setattr(access, 'handlers', []))
        self.assertFalse(any(isinstance(f, redact.RedactingFilter) for h in access.handlers for f in h.filters))
        redact.install_logging_redaction()
        self.assertTrue(all(any(isinstance(f, redact.RedactingFilter) for f in h.filters) for h in access.handlers))
        self.assertIs(logging.getLogRecordFactory(), factory)

    def test_handler_filter_alone_scrubs_exc_text_and_args(self):
        # A record created before the factory existed (or by a foreign factory) is still scrubbed at emit.
        self.handler.addFilter(redact.RedactingFilter())
        secret = SAMPLES['anthropic']
        try:
            raise RuntimeError('upstream rejected key ' + secret)
        except RuntimeError:
            logging.getLogger('titanos.test').exception('call failed for %s', 'x-api-key: ' + secret)
        logging.getLogger('titanos.test').info('mapping %(who)s', {'who': secret})
        output = self.stream.getvalue()
        self.assertNotIn(secret, output)
        self.assertIn('Traceback', output)
        self.assertIn('RuntimeError: upstream rejected key <redacted>', output)

    def test_exc_text_scrubbed_by_factory(self):
        redact.install_logging_redaction()
        secret = SAMPLES['groq']
        record = None
        try:
            raise ValueError('bad ' + secret)
        except ValueError:
            record = logging.getLogger('titanos.test').makeRecord('titanos.test', logging.ERROR, __file__, 1, 'boom %s',
                                                                  (secret,), sys.exc_info())
        self.assertNotIn(secret, record.exc_text)
        self.assertNotIn(secret, record.getMessage())

    def test_uvicorn_access_record_with_query_key(self):
        redact.install_logging_redaction()
        sentinel = 'SENTINEL' + HEX[:16]
        access = logging.getLogger('uvicorn.access')
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)

        class AccessFormatter(logging.Formatter):  # the shape of uvicorn's AccessFormatter: positional args
            def formatMessage(self, record):
                client, method, path, version, status = record.args
                return f'{client} - "{method} {path} HTTP/{version}" {status}'
        handler.setFormatter(AccessFormatter())
        access.handlers = [handler]
        access.propagate = False
        self.addCleanup(lambda: (setattr(access, 'handlers', []), setattr(access, 'propagate', True)))
        redact.install_logging_redaction()
        access.setLevel(logging.INFO)
        access.info('%s - "%s %s HTTP/%s" %d', '127.0.0.1:5000', 'POST',
                    '/v1beta/models/gemini:generateContent?key=' + sentinel, '1.1', 401)
        output = stream.getvalue()
        self.assertNotIn(sentinel, output)
        self.assertIn('generateContent?key=<redacted> HTTP/1.1" 401', output)

    def test_uninstalled_state_changes_nothing(self):
        self.assertIs(logging.getLogRecordFactory(), logging.LogRecord)
        logging.getLogger('titanos.test').info('plain ' + SAMPLES['xai'])
        self.assertIn(SAMPLES['xai'], self.stream.getvalue())


class ChildEnv(unittest.TestCase):
    def test_provider_names_and_base_urls_dropped(self):
        environ = {name: 'value' for name in redact.PROVIDER_ENV_DENY}
        environ.update({'PATH': '/usr/bin', 'CUDA_VISIBLE_DEVICES': '0', 'MY_SERVICE_BASE_URL': 'https://x.invalid',
                        'COHERE_API_KEY': 'v', 'openai_api_key': 'lower', 'HOME': '/tmp', 'EXTRA_SECRET_NAME': 'v'})
        out = child_env(environ, extra_deny=('EXTRA_SECRET_NAME',))
        self.assertEqual(out, {'PATH': '/usr/bin', 'CUDA_VISIBLE_DEVICES': '0', 'HOME': '/tmp'})
        for name in ('OPENAI_API_KEY', 'ANTHROPIC_BASE_URL', 'OPENAI_BASE_URL', 'NGC_API_KEY', 'HF_TOKEN', 'GEMINI_API_KEY',
                     'TITANOS_API_KEY'):
            self.assertIn(name, redact.PROVIDER_ENV_DENY)
        self.assertEqual(environ['PATH'], '/usr/bin')  # input untouched

    def test_native_helper_filters_under_isolated_no_site_interpreter(self):
        """titanos.native_process runs as `python -I -S -B` with only src on sys.path (native.py module_command), so its
        child environment must come from the standard library alone: an env: record in the vault index and the provider
        names are dropped, and no titanos.broker module or pydantic is imported."""
        import json
        import subprocess
        import tempfile
        self.assertIs(redact.child_env, __import__('titanos.child_env', fromlist=['child_env']).child_env)
        with tempfile.TemporaryDirectory() as home:
            vault = Path(home) / 'broker' / 'vault'
            vault.mkdir(parents=True)
            (vault / 'index.json').write_text(json.dumps({'version': 1, 'staging': {}, 'credentials': {
                'c1': {'id': 'c1', 'source': 'env', 'env_var': 'MY_CLOUD_SECRET'}}}), encoding='utf-8')
            probe = ('import json,sys;sys.path.insert(0,sys.argv[1]);import titanos.native_process as n;'
                     'e=n.native_environment();print(json.dumps({"kept":sorted(k for k in e if k.startswith("PROBE_")),'
                     '"dropped":[k for k in ("MY_CLOUD_SECRET","OPENAI_API_KEY","SOME_BASE_URL") if k in e],'
                     '"modules":sorted(m for m in sys.modules if m.startswith("titanos.broker") or m=="pydantic")}))')
            env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
            env.update(TITANOS_HOME=home, MY_CLOUD_SECRET='v', OPENAI_API_KEY='v', SOME_BASE_URL='https://x.invalid',
                       PROBE_KEEP='1', TITANOS_CHILD_ENV_ALLOW='PROBE_*')  # a name off the allowlist needs this
            run = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', probe, str(ROOT / 'src')], env=env,
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        self.assertEqual(json.loads(run.stdout), {'kept': ['PROBE_KEEP'], 'dropped': [], 'modules': []})


class ResponseHeaders(unittest.TestCase):
    def test_allowed_upstream_headers_never_reflect_a_registered_secret(self):
        """An upstream that echoes the presented credential in an allowlisted header (request-id, rate-limit families)
        never hands it to the caller; headers outside the allowlist never pass at all."""
        from titanos.broker.adapters.base import filter_response_headers
        value = 'test-openai-not-a-key-' + os.urandom(6).hex()
        REGISTRY.add(value)
        try:
            kept = filter_response_headers({'X-Request-ID': 'req-' + value, 'x-ratelimit-remaining-requests': '9',
                                            'anthropic-ratelimit-tokens-reset': value, 'Set-Cookie': 'id=' + value,
                                            'Location': 'https://elsewhere.example/' + value})
        finally:
            REGISTRY.discard(value)
        self.assertEqual(kept, {'x-request-id': 'req-' + REDACTED, 'x-ratelimit-remaining-requests': '9',
                                'anthropic-ratelimit-tokens-reset': REDACTED})
        self.assertNotIn(value, repr(kept))


if __name__ == '__main__':
    unittest.main(verbosity=1)
