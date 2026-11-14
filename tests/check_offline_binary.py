"""Check real offline packaging functions with inert subprocess and MCP effects.

Only test fixtures and receipts are written. No executable, HTTP server, model,
GPU probe, tool invocation or resource read is started by these checks.
"""
from pathlib import Path
import argparse
import ast
import asyncio
import copy
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/build_binary.py'
EVIDENCE = None
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
FUNCTIONS = {'declared_mcp_tools', 'mcp_tool_source_paths', 'expected_mcp_tools', 'offline_smoke', 'smoke',
             'publish_smoke_summary', 'validate_smoke_summary'}
LEGACY_SOURCES = ['mcp_server.py', 'runtime_api.py', 'assignment_api.py']   # the pre-2026-10-05 list: 60 of 72 tools

# The real registration, listed in-process before the MCP client is mocked below: mcp_server.create_mcp builds the server
# object and registers every tool module (no listener, no tool call, no network). The frozen-build gate is compared against
# this set, so a builder declaration list that drifts from create_mcp cannot be hidden by the inert client, which lists
# exactly these names.
sys.path.insert(0, str(ROOT / 'src'))
from titanos.mcp_server import create_mcp  # noqa: E402

REAL_TOOLS = frozenset(tool.name for tool in asyncio.run(create_mcp('http://127.0.0.1:1').list_tools()))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def declaration_nodes():
    """The builder's MCP_TOOL_SOURCES assignment plus the FUNCTIONS, in source order."""
    return [node for node in TREE.body
            if (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS)
            or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'MCP_TOOL_SOURCES'
                                                     for target in node.targets))]


class Checker(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='offline-binary-check-', dir=EVIDENCE)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.bundle = self.base / 'bundle'
        self.bundle.mkdir()
        self.private = self.base / 'private-receipts'
        self.calls = []
        self.clients = []
        self.fault = None
        self.forbidden = Mock(side_effect=AssertionError('Forbidden external effect'))
        self.environment = {
            'PATH': os.environ.get('PATH', ''),
            'TITANOS_HOME': 'Z:/unrelated-home',
            'TITANOS_URL': 'https://unused.invalid',
            'PYTHONPATH': 'untrusted', '_PYI_TEST': 'untrusted',
            'TITANOS_API_KEY': 'fixture-only', 'HF_TOKEN': 'fixture-only',
        }
        self.namespace = {
            'Path': Path, 'ast': ast, 'asyncio': asyncio, 'json': json,
            'hashlib': hashlib, 'ROOT': ROOT,
            'os': types.SimpleNamespace(environ=self.environment),
            'subprocess': types.SimpleNamespace(run=self.run_cli, Popen=self.forbidden),
            'socket': types.SimpleNamespace(socket=self.forbidden),
            'tempfile': types.SimpleNamespace(TemporaryDirectory=self.make_temporary),
            'digest': lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
        nodes = declaration_nodes()
        self.assertEqual({node.name for node in nodes if isinstance(node, ast.FunctionDef)}, FUNCTIONS)
        self.assertEqual(len(nodes), len(FUNCTIONS) + 1, 'MCP_TOOL_SOURCES must be one module-level assignment')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), self.namespace)
        self.expected = self.namespace['expected_mcp_tools']()
        owner = self

        class Client:
            protocol_version = 'inert-protocol'

            def __init__(self, target, **kwargs):
                self.target = target
                self.initialized = False
                self.closed = False
                self.list_calls = 0
                owner.clients.append(self)
                owner.assertIsInstance(target, dict)
                owner.assertEqual(target['command'], 'inert-titanos.exe')
                owner.assertEqual(target['args'][2:], [
                    'mcp', '--transport', 'stdio', '--url', 'http://127.0.0.1:1'])
                owner.assertEqual(target['args'][:2], ['--config', str(Path(target['cwd']) / 'models.json')])

            async def __aenter__(self):
                if owner.fault == 'initialize':
                    raise RuntimeError('inert initialization failure')
                self.initialized = True
                return self

            async def __aexit__(self, *unused):
                self.closed = True

            async def list_tools(self):
                owner.assertTrue(self.initialized)
                self.list_calls += 1
                names = sorted(REAL_TOOLS)   # what the frozen server would list: the real registration, not the builder's own list
                if owner.fault == 'missing_tool':
                    names.pop()
                elif owner.fault == 'extra_tool':
                    names.append('unexpected_tool')
                elif owner.fault == 'replaced_tool':
                    names[0] = 'unexpected_tool'
                return types.SimpleNamespace(tools=[types.SimpleNamespace(name=name) for name in names])

            async def call_tool(self, *args, **kwargs):
                return owner.forbidden(*args, **kwargs)

            async def read_resource(self, *args, **kwargs):
                return owner.forbidden(*args, **kwargs)

        modules = {name: types.ModuleType(name)
                   for name in ['mcp', 'mcp.client', 'mcp.client.stdio', 'httpx']}
        modules['mcp'].__path__ = []
        modules['mcp.client'].__path__ = []
        modules['mcp'].Client = Client
        modules['mcp.client.stdio'].StdioServerParameters = lambda **kwargs: kwargs
        for name in ['Client', 'AsyncClient', 'get', 'post', 'request']:
            setattr(modules['httpx'], name, self.forbidden)
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def make_temporary(self, **kwargs):
        temporary = tempfile.TemporaryDirectory(dir=self.base, **kwargs)
        self.addCleanup(temporary.cleanup)
        if self.fault != 'cleanup':
            return temporary

        def fail_cleanup():
            raise OSError('inert cleanup failure')

        return types.SimpleNamespace(name=temporary.name, cleanup=fail_cleanup)

    def run_cli(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        self.assertEqual(argv[0], 'inert-titanos.exe')
        self.assertEqual(argv[1:3], ['--config', str(Path(kwargs['cwd']) / 'models.json')])
        arguments = argv[3:]
        self.assertIn(arguments, [['--version'], ['settings', '--json'], ['init'], ['list']])
        env = kwargs['env']
        self.assertEqual(env['TITANOS_HOME'], kwargs['cwd'])
        for key in ['PYTHONPATH', '_PYI_TEST', 'TITANOS_API_KEY', 'HF_TOKEN']:
            self.assertNotIn(key, env)
        for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
            self.assertEqual(env[key], '1')
        for key in ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']:
            self.assertEqual(env[key], '')
        if self.fault == 'cli' and arguments == ['init']:
            return types.SimpleNamespace(returncode=2, stdout='', stderr='inert CLI failure')
        if arguments == ['init']:
            models = {'unexpected': {}} if self.fault == 'registry' else {}
            Path(argv[2]).write_text(json.dumps({'models': models}), encoding='utf-8')
        stdout = json.dumps({'settings': {}}) if arguments == ['settings', '--json'] else '{}'
        if self.fault == 'settings' and arguments == ['settings', '--json']:
            stdout = '{}'
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr='')

    def offline(self):
        self.namespace['offline_smoke'](Path('inert-titanos.exe'), self.bundle)
        return read(self.bundle / 'binary-smoke.json'), read(self.bundle / 'registry-smoke.json')

    def publish(self):
        return self.namespace['publish_smoke_summary'](self.bundle, self.private)

    def test_metadata_only_exact_inventory_and_no_external_effects(self):
        binary, registry = self.offline()
        self.assertFalse(binary['passed'])
        self.assertTrue(binary['offline_passed'])
        self.assertEqual(binary['validation_mode'], 'offline')
        self.assertEqual(registry['status'], 'NOT_PERFORMED')
        self.assertEqual([argv[3:] for argv, _ in self.calls], [
            ['--version'], ['settings', '--json'], ['init'], ['list']])
        self.assertEqual(len(self.clients), 1)
        self.assertTrue(self.clients[0].closed)
        self.assertEqual(self.clients[0].list_calls, 1)
        self.assertEqual(binary['checks'][-1]['mcp']['tools'], len(REAL_TOOLS))
        self.assertEqual(registry['request_count'], 0)
        self.forbidden.assert_not_called()

    def test_declaration_sources_are_the_real_registration(self):
        # The gate's expected set is the union of MCP_TOOL_SOURCES; it must equal what create_mcp registers. With the
        # pre-fix three-file list (src/titanos/broker/mcp_tools.py missing) the expected set lacks the 12 broker tools and
        # this assertion fails, exactly as the 2026-10-05 frozen builds were refused.
        sources = self.namespace['MCP_TOOL_SOURCES']
        self.assertIsInstance(sources, tuple)
        self.assertIn('src/titanos/broker/mcp_tools.py', sources)
        self.assertEqual(self.expected, REAL_TOOLS,
                         'MCP_TOOL_SOURCES differs from mcp_server.create_mcp: ' + repr(sorted(self.expected ^ REAL_TOOLS)))
        from titanos.broker.mcp_tools import TOOL_NAMES as broker_tools
        self.assertTrue(set(broker_tools) <= self.expected)
        legacy = self.namespace['declared_mcp_tools']([ROOT / 'src/titanos' / name for name in LEGACY_SOURCES])
        self.assertEqual(legacy, REAL_TOOLS - set(broker_tools), 'the legacy list declares exactly the non-broker tools')
        self.assertNotEqual(legacy, REAL_TOOLS, 'the legacy three-file list must not satisfy the gate')
        # a missing declaration source is refused, never silently dropped from the expected set
        partial = self.base / 'partial-tree'
        for name in sources[:-1]:
            (partial / name).parent.mkdir(parents=True, exist_ok=True)
            (partial / name).write_bytes((ROOT / name).read_bytes())
        with self.assertRaisesRegex(ValueError, sources[-1].replace('.', r'\.')):
            self.namespace['mcp_tool_source_paths'](partial)
        self.assertEqual(self.namespace['expected_mcp_tools'](ROOT), REAL_TOOLS)

    def test_dispatch_requires_explicit_offline_and_rejects_unknown_mode(self):
        self.assertEqual(self.namespace['smoke'].__defaults__, ('full',))
        offline = Mock(return_value='offline-result')
        self.namespace['offline_smoke'] = offline
        self.assertEqual(self.namespace['smoke']('exe', 'output', 'offline'), 'offline-result')
        offline.assert_called_once_with('exe', 'output')
        with self.assertRaisesRegex(ValueError, 'Unknown validation'):
            self.namespace['smoke']('exe', 'output', 'unknown')
        self.assertEqual(offline.call_count, 1)
        self.assertEqual(self.calls, [])

    def test_faults_keep_failure_receipts_and_cannot_publish(self):
        for fault in ['cli', 'settings', 'registry', 'initialize', 'missing_tool',
                      'extra_tool', 'replaced_tool', 'cleanup']:
            with self.subTest(fault=fault):
                self.fault = fault
                self.bundle = self.base / fault
                self.bundle.mkdir()
                self.private = self.base / (fault + '-private')
                with self.assertRaises((RuntimeError, OSError)):
                    self.offline()
                binary = read(self.bundle / 'binary-smoke.json')
                self.assertFalse(binary['offline_passed'])
                self.assertFalse(binary['passed'])
                self.assertIn('error', binary)
                with self.assertRaises(ValueError):
                    self.publish()
                self.assertFalse((self.bundle / 'BUILD_SMOKE.json').exists())
                self.assertTrue((self.bundle / 'binary-smoke.json').exists())
        self.forbidden.assert_not_called()

    def test_offline_publication_preserves_raw_receipts_privately(self):
        self.offline()
        originals = {name: (self.bundle / name).read_bytes()
                     for name in ['binary-smoke.json', 'registry-smoke.json']}
        summary = self.publish()
        self.assertEqual(summary, read(self.bundle / 'BUILD_SMOKE.json'))
        self.assertNotIn(str(self.base), json.dumps(summary))
        self.assertFalse(summary['passed'])
        self.assertFalse(summary['full_smoke_passed'])
        self.assertTrue(summary['offline_passed'])
        for name, raw in originals.items():
            self.assertEqual((self.private / name).read_bytes(), raw)
            self.assertFalse((self.bundle / name).exists())
        self.namespace['validate_smoke_summary'](summary, 'offline')
        with self.assertRaises(ValueError):
            self.namespace['validate_smoke_summary'](summary)

    def test_legacy_full_receipts_keep_two_argument_publication(self):
        registry = {'status': 'PASSED', 'model_calls': 0, 'request_count': 19,
                    'cli_add_count': 1, 'cleanup_errors': [], 'temporary_directory_closed': True}
        for name, value in [('registry-smoke.json', registry), ('binary-smoke.json', {'passed': True})]:
            (self.bundle / name).write_text(json.dumps(value), encoding='utf-8')
        summary = self.publish()
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['validation_mode'], 'full')
        self.assertEqual(summary['registry_requests'], 19)
        self.namespace['validate_smoke_summary'](summary)
        legacy_summary = {key: value for key, value in summary.items() if key != 'validation_mode'}
        self.namespace['validate_smoke_summary'](legacy_summary)
        with self.assertRaises(ValueError):
            self.namespace['validate_smoke_summary'](summary, 'offline')

    def test_summary_rejects_missing_failure_and_overstated_fields(self):
        self.offline()
        summary = self.publish()
        changes = {'passed': True, 'offline_passed': False, 'full_smoke_passed': True,
                   'model_calls': 1, 'cleanup_error_count': 1, 'temporary_directory_closed': False,
                   'http_validation': 'passed', 'registry_validation': 'passed',
                   'model_validation': 'passed', 'ui_validation': 'passed', 'validation_mode': 'full'}
        for key, value in changes.items():
            for missing in [False, True]:
                with self.subTest(key=key, missing=missing):
                    invalid = copy.deepcopy(summary)
                    if missing:
                        invalid.pop(key)
                    else:
                        invalid[key] = value
                    with self.assertRaises(ValueError):
                        self.namespace['validate_smoke_summary'](invalid, 'offline')
        with self.assertRaises(ValueError):
            self.namespace['validate_smoke_summary'](summary, 'unknown')

    def test_publisher_rejects_overstated_offline_receipts(self):
        binary, registry = self.offline()
        changes = [('binary', 'passed', True), ('binary', 'offline_passed', False),
                   ('binary', 'validation_mode', 'full'), ('binary', 'http_validation', 'passed'),
                   ('registry', 'status', 'PASSED'), ('registry', 'request_count', 1),
                   ('registry', 'model_calls', 1), ('registry', 'cleanup_errors', ['failed']),
                   ('registry', 'temporary_directory_closed', False)]
        for index, (record, key, value) in enumerate(changes):
            with self.subTest(record=record, key=key):
                self.bundle = self.base / f'bad-receipt-{index}'
                self.bundle.mkdir()
                self.private = self.base / f'bad-private-{index}'
                values = {'binary': copy.deepcopy(binary), 'registry': copy.deepcopy(registry)}
                values[record][key] = value
                for kind, receipt in values.items():
                    (self.bundle / f'{kind}-smoke.json').write_text(json.dumps(receipt), encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.publish()
                self.assertFalse((self.bundle / 'BUILD_SMOKE.json').exists())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.evidence:
        EVIDENCE = args.evidence.resolve()
        EVIDENCE.mkdir(exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checker))
    record = {'passed': result.wasSuccessful(), 'cases': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'real_registration_tools': len(REAL_TOOLS),
              'scope': 'Source functions with inert subprocess/MCP effects; the expected MCP declaration set is checked against the in-process create_mcp registration; no native executable, server, model or GPU execution'}
    if EVIDENCE:
        (EVIDENCE / 'RESULT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
