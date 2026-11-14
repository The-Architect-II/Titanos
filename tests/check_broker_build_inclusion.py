"""The cloud broker ships: every module of src/titanos/broker is in the wheel and in the frozen build.

- wheel: setuptools' PEP 517 backend builds a wheel from a temporary copy of the packaging inputs (pyproject.toml,
  MANIFEST.in, README.md, LICENSE, NOTICE and src/); its member list holds every broker module. Nothing is written under
  the checkout and nothing is downloaded (no build isolation, no index);
- frozen build: titanos is a core collected package of tools/build_binary.py; the engine profile's excludes never name
  titanos; the PyInstaller spec's runtime_module filter (tools/titanos.spec, read by AST) keeps every titanos.broker module;
  PyInstaller's collect_all('titanos', include_py_files=True, filter_submodules=runtime_module) - what the spec runs -
  returns every broker module as a hidden import and as a source file placed under titanos/broker; and
  build_binary.verify_broker_payload, which the real build runs on its output, refuses a built directory missing one
  broker module and accepts a complete one.
No PyInstaller build, no binary, no model, no network. Without PyInstaller (installed by the package tier's build extra)
the collection case is skipped and says so; the wheel and filter cases still run.
"""
import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
BROKER = SRC / 'titanos' / 'broker'


def broker_files():
    return sorted(path.relative_to(SRC).as_posix() for path in BROKER.rglob('*.py') if '__pycache__' not in path.parts)


def broker_modules():
    out = []
    for name in broker_files():
        dotted = name[:-3].replace('/', '.')
        out.append(dotted[:-len('.__init__')] if dotted.endswith('.__init__') else dotted)
    return sorted(out)


def spec_function(name):
    tree = ast.parse((ROOT / 'tools' / 'titanos.spec').read_text(encoding='utf-8'))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'tools/titanos.spec', 'exec'), scope)
    return scope[name]


def build_binary():
    spec = importlib.util.spec_from_file_location('titanos_build_binary', ROOT / 'tools' / 'build_binary.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Inventory(unittest.TestCase):
    def test_the_broker_package_is_not_empty(self):
        files = broker_files()
        self.assertIn('titanos/broker/__init__.py', files)
        for package in ('adapters', 'translate', 'vault'):
            self.assertIn(f'titanos/broker/{package}/__init__.py', files)
        self.assertGreater(len(files), 50)


class Wheel(unittest.TestCase):
    def test_every_broker_module_is_in_the_wheel(self):
        try:
            import setuptools  # noqa: F401
            import wheel  # noqa: F401
        except ImportError as exc:
            self.skipTest(f'untried: the wheel case needs setuptools and wheel ({exc.name} is missing)')
        work = Path(tempfile.mkdtemp(prefix='titanos-broker-wheel-'))
        self.addCleanup(shutil.rmtree, work, True)
        tree = work / 'src-tree'
        tree.mkdir()
        for name in ('pyproject.toml', 'MANIFEST.in', 'README.md', 'LICENSE', 'NOTICE'):
            shutil.copy2(ROOT / name, tree / name)
        shutil.copytree(SRC, tree / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
        out = work / 'dist'
        out.mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1')
        env.pop('PYTHONPATH', None)
        done = subprocess.run([sys.executable, '-B', '-c', 'import sys, setuptools.build_meta as backend; '
                               'print(backend.build_wheel(sys.argv[1]))', str(out)],
                              cwd=tree, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(done.returncode, 0, (done.stdout + done.stderr)[-2000:])
        wheels = list(out.glob('*.whl'))
        self.assertEqual(len(wheels), 1, wheels)
        with zipfile.ZipFile(wheels[0]) as archive:
            members = set(archive.namelist())
        missing = [name for name in broker_files() if name not in members]
        self.assertEqual(missing, [], 'broker modules missing from the wheel')
        self.assertFalse(any('/tests/' in name for name in members if name.startswith('titanos/broker/')))


class FrozenBuild(unittest.TestCase):
    def test_titanos_is_collected_and_never_excluded(self):
        source = (ROOT / 'tools' / 'build_binary.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        core = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'CORE_PACKAGES' for t in node.targets))
        self.assertIn('titanos', ast.literal_eval(core))
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'engine_excludes')
        scope = {}
        exec('from packaging.utils import canonicalize_name\nimport importlib.metadata as metadata', scope)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'tools/build_binary.py', 'exec'), scope)
        excluded = scope['engine_excludes']({'titanos-engine', 'fastapi'}, {'titanos': ['titanos-engine'],
                                                                           'numpy': ['numpy']})
        self.assertNotIn('titanos', excluded)

    def test_the_spec_filter_keeps_every_broker_module(self):
        runtime_module = spec_function('runtime_module')
        refused = [name for name in broker_modules() if not runtime_module(name)]
        self.assertEqual(refused, [])
        self.assertIn('titanos.broker.cli', broker_modules())  # 'cli' is refused only outside titanos.*
        self.assertFalse(runtime_module('some_dependency.cli'))

    def test_pyinstaller_collects_every_broker_module(self):
        try:
            from PyInstaller.utils.hooks import collect_all
        except ImportError:
            self.skipTest('untried: PyInstaller is not installed here (the package tier installs it with .[build])')
        runtime_module = spec_function('runtime_module')
        previous = os.environ.get('PYTHONPATH')
        os.environ['PYTHONPATH'] = str(SRC) + (os.pathsep + previous if previous else '')  # the isolated import child
        try:
            datas, _, imports = collect_all('titanos', include_py_files=True, filter_submodules=runtime_module,
                                            exclude_datas=['**/tests/**', '**/test/**', '**/__pycache__/**'],
                                            on_error='warn once')
        finally:
            if previous is None:
                os.environ.pop('PYTHONPATH', None)
            else:
                os.environ['PYTHONPATH'] = previous
        self.assertEqual(sorted(set(broker_modules()) - set(imports)), [], 'broker modules missing from hiddenimports')
        placed = {Path(dest.replace('\\', '/'), Path(src).name).as_posix() for src, dest in datas if src.endswith('.py')}
        self.assertEqual(sorted(set(broker_files()) - placed), [], 'broker sources missing from the collected datas')

    def test_verify_broker_payload_refuses_a_build_without_a_broker_module(self):
        module = build_binary()
        built = Path(tempfile.mkdtemp(prefix='titanos-broker-frozen-'))
        self.addCleanup(shutil.rmtree, built, True)
        for name in broker_files():
            target = built / '_internal' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'')
        self.assertEqual(module.verify_broker_payload(built), broker_files())
        (built / '_internal' / 'titanos' / 'broker' / 'egress.py').unlink()
        with self.assertRaisesRegex(RuntimeError, 'titanos/broker/egress.py'):
            module.verify_broker_payload(built)
        with self.assertRaises(RuntimeError):
            module.verify_broker_payload(built / 'missing')
        source = (ROOT / 'tools' / 'build_binary.py').read_text(encoding='utf-8')
        self.assertIn('    verify_broker_payload(built)\n', source.replace('\r\n', '\n'))  # the real build runs it


if __name__ == '__main__':
    unittest.main(verbosity=2)
