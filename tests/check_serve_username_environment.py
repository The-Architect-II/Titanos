"""`serve` provides a user name for runtime libraries that call getpass.getuser() when the process environment has none
of LOGNAME/USER/LNAME/USERNAME (the frozen Windows speech path otherwise fails with "No module named 'pwd'"). Unit checks
on the helper plus a static check of the serve call site; no torch, socket, server or model."""
import argparse
import getpass
import hashlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos import cli  # noqa: E402

EVIDENCE = None
BASE = {'SystemRoot': os.environ.get('SystemRoot', ''), 'PATH': os.environ.get('PATH', ''), 'TEMP': os.environ.get('TEMP', '')}


class Checker(unittest.TestCase):
    def test_provides_account_name_when_no_username_variable_is_set(self):
        with mock.patch.dict(os.environ, BASE, clear=True):
            self.assertFalse(any(os.environ.get(n) for n in cli.USERNAME_VARIABLES))
            name = cli.ensure_username_environment()
            self.assertTrue(name)
            self.assertEqual(os.environ['USERNAME'], name)
            self.assertNotIn('/', name)
            self.assertEqual(getpass.getuser(), name)  # the stdlib lookup that failed now succeeds without pwd

    def test_never_changes_an_existing_variable(self):
        for variable in cli.USERNAME_VARIABLES:
            with mock.patch.dict(os.environ, {**BASE, variable: 'alice'}, clear=True):
                self.assertIsNone(cli.ensure_username_environment())
                self.assertEqual(os.environ[variable], 'alice')
                self.assertEqual(set(os.environ) & set(cli.USERNAME_VARIABLES), {variable})

    def test_empty_variables_count_as_absent(self):
        with mock.patch.dict(os.environ, {**BASE, 'USERNAME': '', 'USER': ''}, clear=True):
            name = cli.ensure_username_environment()
            self.assertTrue(name)
            self.assertEqual(os.environ['USERNAME'], name)

    def test_falls_back_to_a_fixed_name_when_the_platform_has_none(self):
        with mock.patch.dict(os.environ, BASE, clear=True), mock.patch.object(cli, 'account_name', return_value=None):
            self.assertEqual(cli.ensure_username_environment(), 'titanos')
            self.assertEqual(getpass.getuser(), 'titanos')

    def test_account_name_survives_a_failing_getlogin(self):
        with mock.patch.object(os, 'getlogin', side_effect=OSError('no terminal')):
            name = cli.account_name()
        self.assertTrue(name is None or isinstance(name, str) and name)

    def test_serve_calls_the_helper_before_importing_the_server(self):
        source = (ROOT / 'src/titanos/cli.py').read_text(encoding='utf-8')
        serve = source[source.index('elif args.command == "serve":'):source.index('uvicorn.run(')]
        self.assertIn('ensure_username_environment()', serve)
        self.assertLess(serve.index('ensure_username_environment()'), serve.index('from .server import create_app'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    if args.evidence:
        EVIDENCE = args.evidence.resolve()
        EVIDENCE.mkdir(exist_ok=False)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checker))
    print(log.getvalue(), end='')
    record = {'passed': result.wasSuccessful(), 'cases': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'source_sha256': hashlib.sha256((ROOT / 'src/titanos/cli.py').read_bytes()).hexdigest(), 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Helper unit checks and a static serve call-site check; no torch, socket, listener, server or model'}
    if EVIDENCE:
        (EVIDENCE / 'run.log').write_text(log.getvalue(), encoding='utf-8')
        (EVIDENCE / 'RESULT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
