"""Broker settings (WP1): defaults, TITANOS_* parsing, choice and path validation, the TLS pair rule, secret-free describe(),
the existing check_settings assertions, and the preserved newline style of settings.py. No server, model or network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos.settings import Settings  # noqa: E402

ABSOLUTE = str(Path(tempfile.gettempdir()).resolve() / 'titanos-broker-fixture.pem')

DEFAULTS = {
    'role': 'standalone', 'broker_config': '', 'broker_cloud_ceiling': 'fallback', 'broker_egress': 'internet',
    'broker_vault_backend': 'auto', 'broker_vault_passphrase_file': '', 'broker_connect_timeout': 10.0,
    'broker_first_byte_timeout': 120.0, 'broker_stream_idle_timeout': 120.0, 'broker_max_fallbacks': 2,
    'broker_breaker_failures': 3, 'broker_breaker_base_seconds': 2.0, 'broker_breaker_max_seconds': 300.0,
    'broker_health_ttl': 300.0, 'broker_probe_idle_cutoff': 900.0, 'broker_catalog_ttl': 3600.0,
    'broker_advisor_timeout_ms': 300, 'broker_decision_log_size': 1000, 'broker_decision_log_file': False,
    'broker_usage_retention_days': 90, 'broker_audit_retention_days': 365, 'broker_secret_cache_seconds': 300.0,
    'broker_debug_upstream_urls': False, 'broker_default_max_output_tokens': 4096,
    'broker_max_upstream_json_bytes': 16777216, 'broker_max_upstream_media_bytes': 67108864,
    'broker_auth_failures_per_minute': 20, 'tls_certfile': '', 'tls_keyfile': '', 'behind_tls_proxy': False,
    'managed_delegation': 'off', 'managed_deadline_headers': 'x-titanos-deadline-ms',
    'managed_trace_headers': 'x-request-id,traceparent,x-hivemind-trace',
}
# one valid non-default environment value per field, and its parsed value
ENV = {
    'role': ('broker', 'broker'), 'broker_config': (ABSOLUTE, ABSOLUTE), 'broker_cloud_ceiling': ('explicit', 'explicit'),
    'broker_egress': ('loopback-only', 'loopback-only'), 'broker_vault_backend': ('refs-only', 'refs-only'),
    'broker_vault_passphrase_file': (ABSOLUTE, ABSOLUTE), 'broker_connect_timeout': ('2.5', 2.5),
    'broker_first_byte_timeout': ('30', 30.0), 'broker_stream_idle_timeout': ('45.5', 45.5),
    'broker_max_fallbacks': ('5', 5), 'broker_breaker_failures': ('7', 7), 'broker_breaker_base_seconds': ('1.5', 1.5),
    'broker_breaker_max_seconds': ('60', 60.0), 'broker_health_ttl': ('10', 10.0), 'broker_probe_idle_cutoff': ('20', 20.0),
    'broker_catalog_ttl': ('30', 30.0), 'broker_advisor_timeout_ms': ('5000', 5000), 'broker_decision_log_size': ('5', 5),
    'broker_decision_log_file': ('true', True), 'broker_usage_retention_days': ('7', 7),
    'broker_audit_retention_days': ('30', 30), 'broker_secret_cache_seconds': ('1', 1.0),
    'broker_debug_upstream_urls': ('1', True), 'broker_default_max_output_tokens': ('512', 512),
    'broker_max_upstream_json_bytes': ('1024', 1024), 'broker_max_upstream_media_bytes': ('2048', 2048),
    'broker_auth_failures_per_minute': ('3', 3), 'behind_tls_proxy': ('true', True),
    'managed_delegation': ('parent', 'parent'), 'managed_deadline_headers': ('x-titanos-deadline-ms,x-hivemind-deadline',
                                                                            'x-titanos-deadline-ms,x-hivemind-deadline'),
    'managed_trace_headers': ('x-request-id', 'x-request-id'),
}
# the 19 LF-only lines of settings.py before the broker fields were added (the file is otherwise CRLF)
LF_ONLY = [
    b"    native_gpu_layers: int | str = 'auto'", b'    device_probe_timeout: float = 30.0',
    b'    device_inventory_ttl: float = 30.0', b'        for f in fields(self):', b'            value = getattr(self, f.name)',
    b"            if f.name == 'native_gpu_layers':",
    b"                if not (type(value) is int and value >= 0 or type(value) is str and value in {'auto', 'all'}):",
    b"                    raise ValueError('TITANOS_NATIVE_GPU_LAYERS must be auto, all or a nonnegative integer')",
    b'                continue', b'            elif key in env:', b'                try:',
    b"                    if f.name == 'native_gpu_layers':", b'                        literal = env[key].strip().lower()',
    b"                        values[f.name] = literal if literal in {'auto', 'all'} else int(literal)",
    b'                    elif f.type is bool:',
    b"                    kind = 'integer, auto or all' if f.name == 'native_gpu_layers' else f.type.__name__",
    b'                    raise ValueError(f"{key} must be a valid {kind}") from exc',
    b'                                      "env": "TITANOS_" + f.name.upper(),',
    b'                                      "type": \'int_or_auto_all\' if f.name == \'native_gpu_layers\' else f.type.__name__}',
    b'',
]


class BrokerSettings(unittest.TestCase):
    def test_defaults_construct(self):
        for value in (Settings(), Settings.from_env({})):
            for name, default in DEFAULTS.items():
                self.assertEqual(getattr(value, name), default, name)
                self.assertIs(type(getattr(value, name)), type(default), name)

    def test_every_environment_name_parses(self):
        self.assertEqual(set(ENV), set(DEFAULTS) - {'tls_certfile', 'tls_keyfile'})
        for name, (text, parsed) in ENV.items():
            with self.subTest(name=name):
                value = Settings.from_env({'TITANOS_' + name.upper(): text})
                self.assertEqual(getattr(value, name), parsed)
        pair = Settings.from_env({'TITANOS_TLS_CERTFILE': ABSOLUTE, 'TITANOS_TLS_KEYFILE': ABSOLUTE})
        self.assertEqual((pair.tls_certfile, pair.tls_keyfile), (ABSOLUTE, ABSOLUTE))
        sealed = Settings.from_env({'TITANOS_BROKER_VAULT_BACKEND': 'sealed-file',
                                    'TITANOS_BROKER_VAULT_PASSPHRASE_FILE': ABSOLUTE})
        self.assertEqual(sealed.broker_vault_backend, 'sealed-file')
        self.assertEqual(Settings.from_env({'TITANOS_ROLE': 'broker'}, role='managed').role, 'managed')

    def test_bad_values_raise(self):
        bad = [('TITANOS_ROLE', 'orchestrator'), ('TITANOS_ROLE', ''), ('TITANOS_BROKER_CLOUD_CEILING', 'always'),
               ('TITANOS_BROKER_EGRESS', 'wan'), ('TITANOS_BROKER_VAULT_BACKEND', 'plaintext'),
               ('TITANOS_MANAGED_DELEGATION', 'yes'), ('TITANOS_BROKER_MAX_FALLBACKS', '6'),
               ('TITANOS_BROKER_MAX_FALLBACKS', '-1'), ('TITANOS_BROKER_MAX_FALLBACKS', '1.5'),
               ('TITANOS_BROKER_ADVISOR_TIMEOUT_MS', '5001'), ('TITANOS_BROKER_ADVISOR_TIMEOUT_MS', '0'),
               ('TITANOS_BROKER_CONNECT_TIMEOUT', '0'), ('TITANOS_BROKER_CONNECT_TIMEOUT', 'nan'),
               ('TITANOS_BROKER_FIRST_BYTE_TIMEOUT', 'inf'), ('TITANOS_BROKER_DECISION_LOG_FILE', 'maybe'),
               ('TITANOS_BROKER_DECISION_LOG_SIZE', '0'), ('TITANOS_BROKER_CONFIG', 'relative/broker.json'),
               ('TITANOS_BROKER_VAULT_PASSPHRASE_FILE', 'pass.txt'),
               ('TITANOS_BROKER_VAULT_BACKEND', 'sealed-file'), ('TITANOS_BEHIND_TLS_PROXY', 'yes'),
               ('TITANOS_MANAGED_DEADLINE_HEADERS', 'x-titanos-deadline-ms,'),
               ('TITANOS_MANAGED_TRACE_HEADERS', 'x request id'), ('TITANOS_MANAGED_DELEGATION', ''),
               ('TITANOS_BROKER_BREAKER_BASE_SECONDS', '301')]
        for key, value in bad:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    Settings.from_env({key: value})
        self.assertEqual(Settings.from_env({'TITANOS_BROKER_MAX_FALLBACKS': '0'}).broker_max_fallbacks, 0)
        with self.assertRaises(ValueError):
            Settings(broker_config=' ' + ABSOLUTE)
        with self.assertRaises(ValueError):
            Settings(broker_decision_log_file=1)
        with self.assertRaises(ValueError):
            Settings(broker_max_fallbacks=True)

    def test_tls_pair_and_proxy_rules(self):
        with self.assertRaises(ValueError):
            Settings(tls_certfile=ABSOLUTE)
        with self.assertRaises(ValueError):
            Settings(tls_keyfile=ABSOLUTE)
        with self.assertRaises(ValueError):
            Settings(tls_certfile='cert.pem', tls_keyfile='key.pem')
        Settings(tls_certfile=ABSOLUTE, tls_keyfile=ABSOLUTE)
        with self.assertRaises(ValueError):
            Settings(host='0.0.0.0', behind_tls_proxy=True)
        for host in ('127.0.0.1', '::1', 'localhost'):
            self.assertTrue(Settings(host=host, behind_tls_proxy=True).behind_tls_proxy)

    def test_describe_is_secret_free(self):
        environ = {'TITANOS_API_KEY': 'fixture-bootstrap-value-not-real', 'TITANOS_ROLE': 'broker',
                   'TITANOS_TLS_CERTFILE': ABSOLUTE, 'TITANOS_TLS_KEYFILE': ABSOLUTE}
        described = Settings.from_env(environ).describe()
        text = json.dumps(described)
        self.assertNotIn('fixture-bootstrap-value-not-real', text)
        self.assertNotIn('TITANOS_API_KEY', text)
        rows = described['settings']
        self.assertEqual(rows['role'], {'value': 'broker', 'default': 'standalone', 'env': 'TITANOS_ROLE', 'type': 'str'})
        self.assertEqual(rows['broker_max_fallbacks']['type'], 'int')
        self.assertEqual(rows['behind_tls_proxy']['type'], 'bool')
        for name in DEFAULTS:
            self.assertIn(name, rows)
            self.assertNotIn('secret', name.replace('secret_cache', ''))  # no field holds a secret value

    def test_existing_settings_check_still_passes(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('TITANOS_')}
        env.update(PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'tests' / 'check_settings.py')], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PASS', result.stdout)

    def test_settings_newline_style_preserved(self):
        """The repository stores settings.py with LF, so a fresh checkout is uniform: all LF, or all CRLF where
        core.autocrlf=true (Git for Windows' default, as on the windows-latest CI runner). Only a working copy that is
        actually mixed is held to the exact set of LF-only lines the broker edit must not have changed."""
        data = (ROOT / 'src' / 'titanos' / 'settings.py').read_bytes()
        self.assertNotIn(b'\r', data.replace(b'\r\n', b''), 'a bare CR is never valid')
        crlf = data.count(b'\r\n')
        if crlf == 0 or b'\n' not in data.replace(b'\r\n', b''):
            return  # uniformly LF or uniformly CRLF: checkout-dependent bytes, nothing to compare
        lf_only = [line for line in data.split(b'\n') if not line.endswith(b'\r')]
        self.assertEqual(lf_only, LF_ONLY)


if __name__ == '__main__':
    unittest.main(verbosity=1)
