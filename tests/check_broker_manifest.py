"""WP8 integration manifests.

- `titanos hivemind-manifest` output is validated by a Python port of HiveMind's engine-manager manifest rules
  (manifest.rs: required engine/display_name/manifest_version, <goos>/<goarch> platform keys, process mode needs
  runtime.bin, every action sets exactly one of http/cmd, and the {token} placeholders of the templated strings), with
  the Titanos rule narrower than HiveMind's list: only {bind}, {port}, {bin} and {cli} (never {host});
- its keys stay inside a schema fixture shaped like a bundled manifest (keys and placeholders only, no paths);
- serve runs with --role managed on an explicit non-reserved port; readiness and health are /v1/health/ready; identity is
  /api/info name Titanos; --register-with-gateway adds the gateway args;
- pair-manifest keeps its port refusals (11434, 11435, 11437, 1234, 1235, 8000, 8001 and equal ports), keeps the identity
  block byte-identical and gains only '--role','managed' in its runtime args; the CLI defaults are 11440/11441.
No server; the hardware probe is patched. The netguard is installed and every test asserts zero non-loopback
connection attempts.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

import broker_wp7_support  # noqa: E402,F401  (installs the netguard)
from broker_mocks import netguard  # noqa: E402
from titanos.broker.cli import hivemind_manifest  # noqa: E402
from titanos.cli import main, pair_manifest  # noqa: E402

TITANOS_PLACEHOLDERS = {'bind', 'port', 'bin', 'cli'}
HOST = {'os': 'linux', 'arch': 'x86_64'}
# Shaped like a bundled engine-manager manifest (ollama.json): the keys each level may use, nothing else.
BUNDLED_SHAPE = {
    'engine': None, 'display_name': None, 'manifest_version': None, 'integration': None,
    'detect': None, 'install': None, 'uninstall': None,
    'runtime': {'mode': None, 'bin': None, 'args': None, 'env': None, 'port': None, 'start': None, 'cli': None,
                'ready': {'http': None, 'tcp': None, 'status': None, 'timeout_s': None, 'interval_s': None},
                'stop': {'signal': None, 'grace_s': None, 'cmd': None},
                'health': {'http': None, 'tcp': None, 'status': None, 'timeout_s': None, 'interval_s': None},
                'bind': None, 'accelerators': None, 'version': None},
    'platforms': {'*': {'detect': None, 'install': None, 'uninstall': None,
                        'runtime': {'mode': None, 'bin': None, 'args': None, 'env': None, 'port': None, 'cli': None,
                                    'ready': None, 'health': None, 'bind': None, 'stop': None}}},
    'actions': {'*': {'description': None, 'http': {'method': None, 'path': None, 'body_schema': None}, 'cmd': None}},
}


def placeholders_in(text: str) -> list[str]:
    """Port of manifest.rs placeholders_in: {token} where token is non-empty [A-Za-z0-9_]."""
    out, i = [], 0
    while i < len(text):
        if text[i] == '{':
            end = text.find('}', i + 1)
            if end >= 0:
                token = text[i + 1:end]
                if token and re.fullmatch(r'[A-Za-z0-9_]+', token):
                    out.append(token)
                i = end + 1
                continue
        i += 1
    return out


def merged_runtime(manifest, key):
    runtime = dict(manifest.get('runtime') or {})
    runtime.update((manifest['platforms'][key] or {}).get('runtime') or {})
    return runtime


def validate(manifest: dict, allowed=TITANOS_PLACEHOLDERS) -> None:
    """Port of Manifest::validate / validate_resolved / templated_strings (the parts a Titanos manifest can reach)."""
    engine = manifest.get('engine', '')
    if not str(engine).strip() or not re.fullmatch(r'[A-Za-z0-9._-]+', engine) or engine in {'.', '..'}:
        raise ValueError('engine name')
    if not str(manifest.get('display_name', '')).strip():
        raise ValueError('display_name')
    if manifest.get('manifest_version') != 1:
        raise ValueError('manifest_version')
    if not manifest.get('platforms'):
        raise ValueError('platforms')
    for key in manifest['platforms']:
        parts = key.split('/')
        if len(parts) != 2 or not all(parts):
            raise ValueError('platform key ' + key)
        runtime = merged_runtime(manifest, key)
        mode = runtime.get('mode', 'process')
        if mode == 'process' and not str(runtime.get('bin', '')).strip():
            raise ValueError('process mode requires runtime.bin')
        if mode not in ('process', 'command'):
            raise ValueError('runtime.mode')
        detect = (manifest['platforms'][key] or {}).get('detect') or manifest.get('detect') or []
        strings = [*detect, runtime.get('bin') or '', *runtime.get('args', []), *(runtime.get('env') or {}).values(),
                   runtime.get('cli') or '', *[t for argv in runtime.get('start', []) for t in argv],
                   *((runtime.get('stop') or {}).get('cmd') or [])]
        for probe in ('ready', 'health'):
            for field in ('http', 'tcp'):
                strings.append((runtime.get(probe) or {}).get(field) or '')
        for text in strings:
            for token in placeholders_in(text):
                if token not in allowed:
                    raise ValueError(f'unknown placeholder {{{token}}} in {text!r}')
    for name, action in (manifest.get('actions') or {}).items():
        if bool(action.get('http')) == bool(action.get('cmd')):
            raise ValueError(f'action {name} must set exactly one of http or cmd')


def within_shape(value, shape, path='$'):
    if shape is None or not isinstance(value, dict):
        return
    for key, item in value.items():
        if key in shape:
            within_shape(item, shape[key], path + '.' + key)
        elif '*' in shape:
            within_shape(item, shape['*'], path + '.' + key)
        else:
            raise AssertionError(f'{path}.{key} is not a bundled-manifest key')


class Guarded(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')


class HiveMindManifest(Guarded):
    def build(self, **kw):
        with mock.patch('titanos.hardware.discover', return_value=HOST):
            return hivemind_manifest(Path('models.json'), 'titanos', 12345, 'inference_microservices', **kw)

    def test_placeholder_rules_and_shape(self):
        manifest = self.build()
        validate(manifest)
        within_shape(manifest, BUNDLED_SHAPE)
        runtime = manifest['runtime']
        self.assertEqual(runtime['args'], ['--config', 'models.json', 'serve', '--role', 'managed', '--host', '{bind}',
                                           '--port', '{port}'])
        self.assertEqual(runtime['port'], 12345)
        self.assertTrue(runtime['ready']['http'].endswith('/v1/health/ready'))
        self.assertTrue(runtime['health']['http'].endswith('/v1/health/ready'))
        self.assertEqual(manifest['integration']['identity'], {'path': '/api/info', 'name': 'Titanos'})
        self.assertEqual(manifest['integration']['job_type'], 'inference_microservices')
        self.assertEqual(list(manifest['platforms']), ['linux/amd64'])
        text = json.dumps(manifest)
        self.assertNotIn('{host}', text)

    def test_the_port_of_the_rules_rejects_host_and_unknown_tokens(self):
        manifest = self.build()
        bad = json.loads(json.dumps(manifest))
        bad['runtime']['args'][6] = '{host}'
        with self.assertRaises(ValueError):
            validate(bad)
        bad = json.loads(json.dumps(manifest))
        bad['actions']['chat']['cmd'] = ['x']
        with self.assertRaises(ValueError):
            validate(bad)
        self.assertEqual(placeholders_in('a {port} {bad-token} {x_1} {'), ['port', 'x_1'])

    def test_gateway_args_and_refusals(self):
        manifest = self.build(register_with_gateway='http://127.0.0.1:6112')
        validate(manifest)
        self.assertEqual(manifest['runtime']['args'][-4:], ['--register-with-gateway', 'http://127.0.0.1:6112',
                                                            '--job-type', 'inference_microservices'])
        with mock.patch('titanos.hardware.discover', return_value=HOST):
            for port in (11434, 11435, 1234, 1235, 8000, 8001, 0, 70000):
                with self.assertRaises(ValueError):
                    hivemind_manifest(Path('m.json'), 'titanos', port, 'jt')
            with self.assertRaises(ValueError):
                hivemind_manifest(Path('m.json'), 'titanos', 12345, 'bad job type')
            with self.assertRaises(ValueError):
                hivemind_manifest(Path('m.json'), 'titanos', 12345, 'jt', register_with_gateway='ftp://x')

    def test_cli_writes_the_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'titanos.json'
            with mock.patch('titanos.hardware.discover', return_value=HOST), contextlib.redirect_stdout(io.StringIO()):
                main(['--config', str(Path(folder) / 'models.json'), 'hivemind-manifest', '--port', '12346',
                      '--job-type', 'jt', '--output', str(out)])
            manifest = json.loads(out.read_text(encoding='utf-8'))
        validate(manifest)
        self.assertEqual(manifest['runtime']['port'], 12346)


class PairManifest(Guarded):
    def test_identity_unchanged_and_role_in_args(self):
        with mock.patch('titanos.hardware.discover', return_value={'os': 'windows', 'arch': 'x86_64'}):
            value = pair_manifest(Path('models.json'), 'titanos.exe')
        self.assertEqual(value['integration']['identity'], {'path': '/api/info', 'name': 'Titanos', 'standalone': True})
        runtime = value['platforms']['windows/amd64']['runtime']
        self.assertEqual(runtime['args'], ['--config', 'models.json', 'serve', '--role', 'managed', '--host', '{host}',
                                           '--port', '{port}'])

    def test_port_refusals_kept(self):
        with mock.patch('titanos.hardware.discover', return_value=HOST):
            for backend, facade in ((11435, 11440), (11437, 11440), (11441, 11437), (11441, 11434), (1234, 11440), (8001, 11440),
                                    (11440, 11440), (0, 1)):
                with self.assertRaises(ValueError):
                    pair_manifest(Path('m.json'), 'titanos', backend, facade)

    def test_cli_defaults_are_the_lan_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'titanos.json'
            with mock.patch('titanos.hardware.discover', return_value=HOST), contextlib.redirect_stdout(io.StringIO()):
                main(['--config', str(Path(folder) / 'models.json'), 'pair-manifest', '--output', str(out)])
            manifest = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual((manifest['integration']['facade_port'], manifest['integration']['backend_base_port']), (11440, 11441))
        self.assertEqual(manifest['platforms']['linux/amd64']['runtime']['port'], 11441)
        self.assertEqual(manifest['manifest_version'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
