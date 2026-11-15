"""The broker support matrix (docs/broker_matrix.json, docs/BROKER_MATRIX.md) is a fixed point bound to evidence.

Checks:
- rebuilding with tools/build_broker_matrix.py from the run record and live evidence embedded in the committed JSON reproduces
  both files byte for byte (CRLF in a checkout normalized to LF; the rebuild carries no CR);
- every pass cell cites test ids that exist in the test source (class and method read by AST; for the SDK lanes and the
  JavaScript Studio checks, which report their own case ids, a string literal of the source) and passed in the embedded run
  record; every live-pass cell cites a live evidence id recorded for that cell; na and untried cells carry a reason, and
  untried cells cite nothing; every pass is labelled pass (mock) (or pass (mock, adapter column)) and nothing claims a live
  call without a live evidence id;
- no support claim without a cell: every intercept surface x provider profile x stream variant has exactly one cell whose mode
  is the one the real adapter classes give (P native, T supports(), na otherwise), every profile and surface is listed, every
  local mode is a cell, and every catalog and stage surface has at least one cell;
- the embedded run record covers every tests/check_broker_*.py except the matrix consumers, plus the two JavaScript Studio
  checks (a new check means a fresh run);
- the generator's semantics on synthetic results files: a cited test that did not run or was skipped makes its cell untried
  with the reason, a cited test that failed stops the build, live evidence turns a cell into live-pass and is refused for an
  na cell, for an unknown cell and for a value shaped like a secret or a path; several live files merge (an id repeated
  across them is refused); a live observation (outcome observation, note required) is listed on its cell and never changes
  the cell's state;
- the recording shim (--exec-check) records passed, failed and skipped tests of a synthetic check module;
- the generated files carry no lab locator (tools/public_export.json detection patterns).
In-process except the shim case (one short subprocess on a synthetic module); nothing is written outside a temporary directory.
The fresh full run is `python -B tools/build_broker_matrix.py --run --check` (every broker check, a few minutes)."""
import ast
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from broker_mocks import netguard  # noqa: E402

netguard.install()

import build_broker_matrix as builder  # noqa: E402

COMMITTED_JSON = ROOT / 'docs' / builder.JSON_OUT
COMMITTED_MD = ROOT / 'docs' / builder.MD_OUT


def test_ids(check, cache={}):
    """Ids a check can report, read from its source. A unittest module: the Class.method ids a run reports, read by AST
    (test methods of each class plus those it inherits from base classes defined in the same module; the id names the
    concrete class). A lane that reports its own case ids (the SDK lanes, which define no test methods, and the JavaScript
    Studio checks): the string literals of the source, so a cited case id must still be written in the check itself."""
    if check not in cache:
        path = ROOT / 'tests' / check
        if not path.is_file():
            cache[check] = None
        elif path.suffix == '.js':
            cache[check] = set(re.findall(r"'([A-Za-z][A-Za-z0-9_.]{2,159})'", path.read_text(encoding='utf-8')))
        else:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

            def methods(name, seen=()):
                node = classes.get(name)
                if node is None or name in seen:
                    return set()
                own = {item.name for item in node.body if isinstance(item, ast.FunctionDef) and item.name.startswith('test_')}
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        own |= methods(base.id, seen + (name,))
                return own
            ids = {f'{name}.{method}' for name in classes for method in methods(name)}
            if not ids:  # a lane that reports its own case ids (no unittest classes)
                ids = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
            cache[check] = ids
    return cache[check]


def write_results(path, run):
    with open(path, 'w', encoding='utf-8') as handle:
        for name, entry in sorted(run.items()):
            handle.write(json.dumps({'check': name, **entry}, sort_keys=True) + '\n')


class BrokerMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.committed = json.loads(COMMITTED_JSON.read_text(encoding='utf-8'))
        cls.data, cls.markdown = builder.build()
        cls.rendered = builder.render_json(cls.data)
        cls.cells = cls.data['cells']
        cls.temporary = tempfile.TemporaryDirectory(prefix='titanos-broker-matrix-check-')
        cls.tmp = Path(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connection was attempted')

    # fixed point
    def test_rebuild_reproduces_the_committed_files_byte_for_byte(self):
        self.assertNotIn('\r', self.rendered)
        self.assertNotIn('\r', self.markdown)
        self.assertEqual(COMMITTED_JSON.read_bytes().replace(b'\r\n', b'\n').decode('utf-8'), self.rendered,
                         'docs/broker_matrix.json is not the generator output; run python -B tools/build_broker_matrix.py --run')
        self.assertEqual(COMMITTED_MD.read_bytes().replace(b'\r\n', b'\n').decode('utf-8'), self.markdown,
                         'docs/BROKER_MATRIX.md is not the generator output; run python -B tools/build_broker_matrix.py --run')

    # evidence
    def test_every_cell_is_bound_to_evidence_or_a_reason(self):
        run = self.data['run']['checks']
        live = {record['evidence_id']: record for record in self.data['live_evidence']}
        self.assertEqual(set(self.data['vocabulary']['states']), set(builder.STATES))
        for cell_id, cell in self.cells.items():
            self.assertIn(cell['state'], builder.STATES, cell_id)
            if cell['state'] in ('pass', 'live-pass'):
                self.assertTrue(cell['evidence'] or cell['live'], f'{cell_id} passes without evidence')
            for evidence_id in cell['evidence']:
                check, _, test = evidence_id.partition('::')
                ids = test_ids(check)
                self.assertIsNotNone(ids, f'{cell_id}: {check} does not exist')
                self.assertIn(test, ids, f'{cell_id}: {evidence_id} is not a test in the source')
                self.assertIn(test, run[check]['passed'], f'{cell_id}: {evidence_id} did not pass in the evidence run')
            if cell['state'] == 'pass':
                self.assertEqual(cell['live'], [], cell_id)
                self.assertIn(cell['label'], ('pass (mock)', 'pass (mock, adapter column)'), cell_id)
                self.assertIsNone(cell['reason'], cell_id)
            elif cell['state'] == 'live-pass':
                self.assertTrue(cell['live'], f'{cell_id} claims live without a live evidence id')
                for evidence_id in cell['live']:
                    self.assertEqual(live[evidence_id]['cell'], cell_id)
            else:
                self.assertTrue(cell['reason'], f'{cell_id} ({cell["state"]}) has no reason')
                self.assertEqual(cell['live'], [], cell_id)
                if cell['state'] == 'untried':
                    self.assertEqual(cell['evidence'], [], f'{cell_id}: an untried cell cites no evidence')
        claimed_live = {i for cell in self.cells.values() for i in cell['live']}
        self.assertEqual(claimed_live, set(live), 'every live evidence id names exactly the cells that claim it')
        observed = {record['evidence_id']: record for record in self.data.get('live_observations') or []}
        self.assertFalse(set(observed) & set(live), 'an id is either live evidence or a live observation')
        for cell_id, cell in self.cells.items():
            for observation_id in cell.get('live_notes') or []:
                self.assertEqual(observed[observation_id]['cell'], cell_id)
                self.assertNotIn(observation_id, cell['live'], f'{cell_id}: an observation is never evidence')
        self.assertEqual({i for cell in self.cells.values() for i in cell.get('live_notes') or []}, set(observed),
                         'every live observation is listed on exactly the cell it names')
        if not live:
            self.assertNotIn('live-pass', {cell['state'] for cell in self.cells.values()})
            self.assertIn('reads **pass (mock)**', self.markdown)
        for check, entry in run.items():
            self.assertEqual((entry['returncode'], entry['failed']), (0, []), f'{check} failed in the evidence run')
            for test in entry['passed']:
                self.assertFalse(test in entry['skipped'], (check, test))

    def test_run_record_covers_every_broker_check(self):
        expected = {path.name for path in builder.discover_checks()}
        self.assertEqual(set(self.data['run']['checks']), expected,
                         'the evidence run does not match tests/check_broker_*.py; run python -B tools/build_broker_matrix.py --run')
        self.assertEqual(self.data['run']['excluded_checks'], ['check_broker_docs_claims.py', 'check_broker_matrix.py'])

    # no support claim without a cell
    def test_every_surface_profile_and_mode_has_its_cell(self):
        from titanos.broker.profiles import PROFILES
        from titanos.broker.surfaces import SURFACES
        classes = builder._adapter_classes()
        self.assertEqual({p['id'] for p in self.data['profiles']}, set(PROFILES))
        self.assertEqual({s['name'] for s in self.data['surfaces']}, set(SURFACES))
        expected = set()
        for surface in SURFACES.values():
            if surface.kind != 'intercept':
                prefix = surface.name + '|'
                self.assertTrue(any(cell_id.startswith(prefix) for cell_id in self.cells), f'{surface.name} ({surface.kind}) has no cell')
                continue
            for profile in PROFILES.values():
                adapter = None if profile.adapter == 'none' else classes[profile.adapter](profile, None)
                supported = adapter is not None and (surface.name in profile.surfaces or adapter.supports(profile, surface))
                for variant in builder._variants(surface):
                    cell_id = f'{surface.name}|{profile.id}|{variant}'
                    expected.add(cell_id)
                    cell = self.cells.get(cell_id)
                    self.assertIsNotNone(cell, f'no cell for {cell_id}')
                    self.assertEqual(cell['mode'] != 'na', supported, f'{cell_id}: mode {cell["mode"]} but supports() is {supported}')
                    if surface.name in profile.surfaces:
                        self.assertEqual(cell['mode'], 'P', cell_id)
            for variant in builder._variants(surface):
                self.assertIn(f'{surface.name}|local|{variant}', self.cells)
        provider = {cell_id for cell_id, cell in self.cells.items() if cell['section'] == 'provider'}
        self.assertEqual(provider, expected)
        for cell_id, cell in self.cells.items():
            if cell['section'] == 'provider' and cell['state'] == 'pass':
                self.assertIn(cell['proof'], ('profile', 'column'), cell_id)
                self.assertEqual(cell['label'] == 'pass (mock, adapter column)', cell['proof'] == 'column', cell_id)

    # generator semantics on synthetic inputs
    def run_copy(self):
        return copy.deepcopy(self.data['run']['checks'])

    def cited(self, state='pass', section='provider'):
        cell_id = next(i for i, c in sorted(self.cells.items()) if c['section'] == section and c['state'] == state)
        return cell_id, self.cells[cell_id]

    def test_results_files_drive_the_states(self):
        results = self.tmp / 'same.jsonl'
        write_results(results, self.run_copy())
        same, _ = builder.build(results=results)
        self.assertEqual({k: v['state'] for k, v in same['cells'].items()}, {k: v['state'] for k, v in self.cells.items()})
        cell_id, cell = self.cited()
        check, _, test = cell['evidence'][0].partition('::')
        missing = self.run_copy()
        missing[check]['passed'].remove(test)
        write_results(self.tmp / 'missing.jsonl', missing)
        data, _ = builder.build(results=self.tmp / 'missing.jsonl')
        self.assertEqual(data['cells'][cell_id]['state'], 'untried')
        self.assertIn('did not run', data['cells'][cell_id]['reason'])
        skipped = self.run_copy()
        skipped[check]['passed'].remove(test)
        skipped[check]['skipped'][test] = 'not on this host'
        write_results(self.tmp / 'skipped.jsonl', skipped)
        data, _ = builder.build(results=self.tmp / 'skipped.jsonl')
        self.assertEqual(data['cells'][cell_id]['state'], 'untried')
        self.assertIn('skipped on this host: not on this host', data['cells'][cell_id]['reason'])
        failed = self.run_copy()
        failed[check]['passed'].remove(test)
        failed[check]['failed'].append(test)
        failed[check]['returncode'] = 1
        write_results(self.tmp / 'failed.jsonl', failed)
        with self.assertRaises(SystemExit) as raised:
            builder.build(results=self.tmp / 'failed.jsonl')
        self.assertIn(test, str(raised.exception))

    def test_live_evidence_is_validated_and_applied(self):
        cell_id = next(i for i, c in sorted(self.cells.items()) if c['section'] == 'provider' and c['state'] == 'pass' and c['profile'] == 'openai')
        surface, profile, stream = cell_id.split('|')
        live = self.tmp / 'live.jsonl'
        live.write_text(json.dumps({'evidence_id': 'live-test-0001', 'outcome': 'pass', 'profile': profile, 'surface': surface,
                                    'stream': stream, 'observed_on': '2026-10-02'}) + '\n', encoding='utf-8')
        data, markdown = builder.build(live=live)
        self.assertEqual((data['cells'][cell_id]['state'], data['cells'][cell_id]['live']), ('live-pass', ['live-test-0001']))
        self.assertTrue(data['cells'][cell_id]['evidence'], 'the mock evidence stays cited beside the live id')
        self.assertIn('live-test-0001', markdown)
        na_id = next(i for i, c in sorted(self.cells.items()) if c['state'] == 'na')
        for record in ({'evidence_id': 'live-test-0002', 'outcome': 'pass', 'cell': na_id},
                       {'evidence_id': 'live-test-0003', 'outcome': 'pass', 'cell': 'no-such-cell'},
                       {'evidence_id': 'live-test-0004', 'outcome': 'fail', 'cell': cell_id},
                       {'evidence_id': 'live-test-0005', 'outcome': 'pass', 'cell': cell_id, 'note': 'sk-test-' + 'ab12' * 8},
                       {'evidence_id': 'live-test-0006', 'outcome': 'pass', 'cell': cell_id, 'note': 'C:\\private\\receipt.json'},
                       {'evidence_id': 'bad id', 'outcome': 'pass', 'cell': cell_id}):
            with self.subTest(record=record['evidence_id']):
                live.write_text(json.dumps(record) + '\n', encoding='utf-8')
                with self.assertRaises(SystemExit):
                    builder.build(live=live)

    def test_live_files_merge_and_observations_are_never_evidence(self):
        passing = sorted(i for i, c in self.cells.items() if c['section'] == 'provider' and c['state'] == 'pass' and c['profile'] == 'openai')
        first, second = passing[0], passing[1]
        noted = next(i for i, c in sorted(self.cells.items()) if c['section'] == 'provider' and c['state'] == 'pass'
                     and c['profile'] not in ('openai',) and not c.get('live_notes'))
        one, two = self.tmp / 'live-one.jsonl', self.tmp / 'live-two.jsonl'
        one.write_text(json.dumps({'evidence_id': 'live-merge-0001', 'outcome': 'pass', 'cell': first}) + '\n', encoding='utf-8')
        two.write_text(json.dumps({'evidence_id': 'live-merge-0002', 'outcome': 'pass', 'cell': second}) + '\n'
                       + json.dumps({'evidence_id': 'live-merge-0003', 'outcome': 'observation', 'cell': noted,
                                     'observed_on': '2026-10-02', 'note': 'the provider answered 403 for this account'}) + '\n'
                       + json.dumps({'evidence_id': 'live-merge-0004', 'outcome': 'observation', 'cell': first,
                                     'note': 'an earlier model was retired (410)'}) + '\n', encoding='utf-8')
        data, markdown = builder.build(live=[one, two])
        cells = data['cells']
        self.assertEqual((cells[first]['state'], cells[first]['live']), ('live-pass', ['live-merge-0001']))
        self.assertEqual((cells[second]['state'], cells[second]['live']), ('live-pass', ['live-merge-0002']))
        self.assertEqual(cells[first]['live_notes'], ['live-merge-0004'])
        self.assertEqual((cells[noted]['state'], cells[noted]['label'], cells[noted]['live'], cells[noted]['live_notes']),
                         (self.cells[noted]['state'], self.cells[noted]['label'], [], ['live-merge-0003']))
        self.assertEqual([r['evidence_id'] for r in data['live_evidence']], ['live-merge-0001', 'live-merge-0002'])
        self.assertEqual([r['evidence_id'] for r in data['live_observations']], ['live-merge-0003', 'live-merge-0004'])
        self.assertEqual((data['counts']['live_evidence'], data['counts']['live_observations']), (2, 2))
        self.assertIn('live-merge-0003', markdown)
        self.assertIn('Live observations (not evidence)', markdown)
        # a rebuild from the result reuses the embedded evidence and observations (fixed point)
        again, again_md = builder.build(base=data)
        self.assertEqual(builder.render_json(again), builder.render_json(data))
        self.assertEqual(again_md, markdown)
        na_id = next(i for i, c in sorted(self.cells.items()) if c['state'] == 'na')
        bad = self.tmp / 'live-bad.jsonl'
        for name, records in (
                ('an id repeated across files', [{'evidence_id': 'live-merge-0001', 'outcome': 'observation', 'cell': noted, 'note': 'x'}]),
                ('an observation without a note', [{'evidence_id': 'live-merge-0101', 'outcome': 'observation', 'cell': noted}]),
                ('an observation on an unknown cell', [{'evidence_id': 'live-merge-0102', 'outcome': 'observation', 'cell': 'no-such-cell', 'note': 'x'}]),
                ('an observation on an na cell', [{'evidence_id': 'live-merge-0103', 'outcome': 'observation', 'cell': na_id, 'note': 'x'}]),
                ('an observation note shaped like a secret', [{'evidence_id': 'live-merge-0104', 'outcome': 'observation', 'cell': noted,
                                                             'note': 'sk-test-' + 'ab12' * 8}]),
                ('a failed call recorded as fail', [{'evidence_id': 'live-merge-0105', 'outcome': 'fail', 'cell': noted, 'note': 'x'}])):
            with self.subTest(name):
                bad.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
                with self.assertRaises(SystemExit):
                    builder.build(live=[one, bad])

    def test_recording_shim_records_every_outcome(self):
        module = self.tmp / 'check_broker_zz_synthetic.py'
        module.write_text('import unittest\n\n\nclass Synthetic(unittest.TestCase):\n'
                          '    def test_a_passes(self):\n        pass\n\n'
                          '    def test_b_fails(self):\n        self.fail("synthetic")\n\n'
                          '    @unittest.skip("synthetic skip")\n    def test_c_skips(self):\n        pass\n\n'
                          '    def test_d_subtest_fails(self):\n        with self.subTest(i=1):\n            self.assertEqual(1, 2)\n\n\n'
                          'if __name__ == "__main__":\n    unittest.main()\n', encoding='utf-8')
        results = self.tmp / 'shim.jsonl'
        env = dict(os.environ, TITANOS_BROKER_RESULTS=str(results), PYTHONDONTWRITEBYTECODE='1')
        done = subprocess.run([sys.executable, '-B', str(ROOT / 'tools' / 'build_broker_matrix.py'), '--exec-check', str(module)],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        record = builder.read_results(results)['check_broker_zz_synthetic.py']
        self.assertEqual(record['passed'], ['Synthetic.test_a_passes'])
        self.assertEqual(record['failed'], ['Synthetic.test_b_fails', 'Synthetic.test_d_subtest_fails'])
        self.assertEqual(record['skipped'], {'Synthetic.test_c_skips': 'synthetic skip'})
        self.assertEqual(record['returncode'], 1)

    def test_generated_files_carry_no_lab_reference(self):
        policy = json.loads((ROOT / 'tools' / 'public_export.json').read_text(encoding='utf-8'))
        patterns = {name: re.compile(pattern) for name, pattern in policy['detection_patterns'].items()}
        for label, text in (('json', self.rendered), ('md', self.markdown)):
            for name, pattern in patterns.items():
                self.assertIsNone(pattern.search(text), f'{label}: {name} pattern matches the generated broker matrix')


if __name__ == '__main__':
    unittest.main(verbosity=2)
