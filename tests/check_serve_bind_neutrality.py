"""`serve` neither warns speculatively about the chosen port nor changes the requested address; the docs keep neutral
bind-failure troubleshooting. Static source checks only; no socket or server."""
import argparse
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = None


class Checker(unittest.TestCase):
    def test_serve_has_no_success_path_port_warning(self):
        source = (ROOT / 'src/titanos/cli.py').read_text(encoding='utf-8')
        self.assertNotIn('ephemeral_port_hint', source)
        self.assertNotIn('ephemeral_port_range', source)
        serve = source[source.index('elif args.command == "serve":'):source.index('uvicorn.run(')]
        for banned in ('dynamic port range', 'WSAEACCES', '10013', 'warning:'):
            self.assertNotIn(banned, serve)

    def test_serve_passes_the_requested_port_unchanged(self):
        source = (ROOT / 'src/titanos/cli.py').read_text(encoding='utf-8')
        run_call = source[source.index('uvicorn.run('):source.index('log_level=settings.log_level)', source.index('uvicorn.run('))]
        self.assertIn('host=settings.host, port=settings.port', run_call)

    def test_docs_keep_neutral_bind_failure_troubleshooting(self):
        docs = (ROOT / 'docs/CONFIGURATION.md').read_text(encoding='utf-8')
        self.assertIn('## Bind failures', docs)
        self.assertIn('never substitutes another port', docs)
        self.assertIn('show excludedportrange protocol=tcp', docs)
        self.assertNotIn('nothing blocks Titanos', docs)
        self.assertNotIn('`serve` warns', docs)


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
              'scope': 'Static source/doc checks; no socket, listener, server or model'}
    if EVIDENCE:
        (EVIDENCE / 'run.log').write_text(log.getvalue(), encoding='utf-8')
        (EVIDENCE / 'RESULT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
