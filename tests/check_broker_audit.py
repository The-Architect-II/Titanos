"""Broker hash-chained audit (WP3): the chain verifies; a tampered, deleted or reordered line is reported at the right
record; the chain continues across UTC days; two writer processes interleave under the shared FileLock and the chain
still verifies with contiguous sequence numbers; retention prunes old day files and the audit.prune anchor verifies;
no secret reaches a record; query filters; writable(); unknown kinds are refused. Synthetic values only; no server,
model or network (the writer subprocesses only touch a temporary directory)."""
from datetime import datetime, timedelta, timezone
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos.broker.audit import AUDIT_KINDS, GENESIS, AuditLog, record_digest  # noqa: E402
from titanos.broker.redact import REGISTRY  # noqa: E402


def install_netguard():
    """tests/broker_mocks/netguard.py (WP4), loaded by file path so the mock package itself is not imported; without that
    file, a minimal guard that refuses every non-loopback socket connect."""
    path = ROOT / 'tests' / 'broker_mocks' / 'netguard.py'
    if path.is_file():
        spec = importlib.util.spec_from_file_location('titanos_test_netguard', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.install()
        return

    def loopback(address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            return ipaddress.ip_address(str(host).split('%', 1)[0]).is_loopback
        except ValueError:
            return False
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded(original):
        def call(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6) and not loopback(address):
                raise ConnectionRefusedError('netguard: non-loopback connect refused in a broker check')
            return original(self, address)
        return call
    socket.socket.connect = guarded(connect)
    socket.socket.connect_ex = guarded(connect_ex)


install_netguard()
SENTINEL = 'test-anthropic-not-a-key-' + os.urandom(6).hex()
WRITER = r'''
import sys
from titanos.broker.audit import AuditLog
log = AuditLog(sys.argv[1])
for i in range(int(sys.argv[3])):
    log.event('config.reload', sys.argv[2], 'broker.json', {'i': i})
'''


class Clock:
    def __init__(self, now=datetime(2026, 10, 1, 23, 58, tzinfo=timezone.utc)):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / 'broker' / 'audit'
        self.clock = Clock()

    def tearDown(self):
        self.tmp.cleanup()

    def log(self, retention=365):
        return AuditLog(self.dir, retention, clock=self.clock)

    def lines(self, name):
        return (self.dir / name).read_text(encoding='utf-8').splitlines()

    def rewrite(self, name, lines):
        (self.dir / name).write_text(''.join(line + '\n' for line in lines), encoding='utf-8', newline='\n')


class Chain(Base):
    def test_chain_verifies_and_records_are_canonical(self):
        log = self.log()
        self.assertEqual(log.verify().to_dict(), {'ok': True, 'records': 0, 'first_bad': None, 'reason': None})
        digests = [log.event('key.mint', 'bootstrap', f'key{i}', {'n': i}) for i in range(5)]
        report = log.verify()
        self.assertTrue(report.ok)
        self.assertEqual(report.records, 5)
        records = [json.loads(line) for line in self.lines('2026-10-01.jsonl')]
        self.assertEqual([r['seq'] for r in records], [1, 2, 3, 4, 5])
        self.assertEqual(records[0]['prev_sha256'], GENESIS)
        self.assertEqual([r['sha256'] for r in records], digests)
        for prev, rec in zip(records, records[1:]):
            self.assertEqual(rec['prev_sha256'], prev['sha256'])
        self.assertEqual(sorted(records[0]), sorted(['seq', 'ts', 'kind', 'actor', 'subject', 'detail', 'prev_sha256',
                                                     'sha256']))
        body = {k: v for k, v in records[2].items() if k != 'sha256'}
        import hashlib
        expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(records[2]['sha256'], expected)
        self.assertEqual(record_digest(records[2]), expected)

    def test_tampered_line_is_detected_at_the_right_record(self):
        log = self.log()
        for i in range(6):
            log.event('policy.write', 'admin', 'policy', {'i': i})
        lines = self.lines('2026-10-01.jsonl')
        record = json.loads(lines[3])
        record['detail']['i'] = 99
        self.rewrite('2026-10-01.jsonl', lines[:3] + [json.dumps(record, sort_keys=True, separators=(',', ':'))]
                     + lines[4:])
        report = log.verify()
        self.assertEqual((report.ok, report.records, report.first_bad, report.reason),
                         (False, 3, '2026-10-01.jsonl:4', 'digest mismatch'))
        # a re-hashed forgery still breaks the link of the next record
        record['sha256'] = record_digest(record)
        self.rewrite('2026-10-01.jsonl', lines[:3] + [json.dumps(record)] + lines[4:])
        report = log.verify()
        self.assertEqual((report.ok, report.first_bad, report.reason), (False, '2026-10-01.jsonl:5', 'broken link'))
        # a deleted line breaks the link of its successor
        self.rewrite('2026-10-01.jsonl', lines[:2] + lines[3:])
        self.assertEqual(log.verify().first_bad, '2026-10-01.jsonl:3')
        # a deleted head is not anchored
        self.rewrite('2026-10-01.jsonl', lines[1:])
        report = log.verify()
        self.assertEqual((report.first_bad, report.reason),
                         ('2026-10-01.jsonl:1', 'chain head has no genesis or prune anchor'))
        self.rewrite('2026-10-01.jsonl', lines[:2] + ['{not json'] + lines[3:])
        self.assertEqual(log.verify().reason, 'unparseable line')

    def test_chain_continues_across_days(self):
        log = self.log()
        log.event('role.start', 'titanos', 'standalone', {})
        log.event('config.reload', 'admin', 'broker.json', {})
        self.clock.advance(minutes=5)  # 2026-10-02 00:03 UTC
        log.event('config.reload', 'admin', 'broker.json', {})
        self.assertEqual(sorted(p.name for p in self.dir.glob('*.jsonl')), ['2026-10-01.jsonl', '2026-10-02.jsonl'])
        first_day = json.loads(self.lines('2026-10-01.jsonl')[-1])
        second_day = json.loads(self.lines('2026-10-02.jsonl')[0])
        self.assertEqual((second_day['seq'], second_day['prev_sha256']), (3, first_day['sha256']))
        self.assertTrue(log.verify().ok)
        self.assertEqual(log.verify(since='2026-10-02').records, 1)
        # a fresh instance (another process, a restart) continues the same chain
        other = AuditLog(self.dir, clock=self.clock)
        other.event('config.reload', 'cli', 'broker.json', {})
        report = log.verify()
        self.assertEqual((report.ok, report.records), (True, 4))
        # a clock that goes backwards never writes behind the tail
        self.clock.now = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
        log.event('config.reload', 'admin', 'broker.json', {})
        self.assertEqual(len(self.lines('2026-10-02.jsonl')), 3)
        self.assertTrue(log.verify().ok)

    def test_two_writer_processes_interleave_under_the_lock(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
        count = 40
        procs = [subprocess.Popen([sys.executable, '-B', '-c', WRITER, str(self.dir), f'writer{n}', str(count)], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE) for n in range(2)]
        for proc in procs:
            _, err = proc.communicate(timeout=120)
            self.assertEqual(proc.returncode, 0, err.decode('utf-8', 'replace'))
        log = AuditLog(self.dir)
        report = log.verify()
        self.assertTrue(report.ok, report)
        self.assertEqual(report.records, 2 * count)
        records = log.query(limit=1000)
        self.assertEqual([r['seq'] for r in records], list(range(1, 2 * count + 1)))
        self.assertEqual({r['actor'] for r in records}, {'writer0', 'writer1'})


class Retention(Base):
    def test_prune_anchor_verifies(self):
        log = self.log(retention=0)  # automatic retention off; prune() is called explicitly
        for day in range(3):
            for i in range(2):
                log.event('upstream.write', 'admin', f'u{day}', {'i': i})
            self.clock.advance(days=1)
        self.assertEqual(len(list(self.dir.glob('*.jsonl'))), 3)
        last_pruned = json.loads(self.lines('2026-10-02.jsonl')[-1])
        anchor = log.prune(retention_days=1)  # today is 2026-10-04: files before 2026-10-03 go
        self.assertIsNotNone(anchor)
        self.assertEqual(sorted(p.name for p in self.dir.glob('*.jsonl')), ['2026-10-03.jsonl', '2026-10-04.jsonl'])
        prune = log.query(kind='audit.prune')[-1]
        self.assertEqual(prune['sha256'], anchor)
        self.assertEqual((prune['detail']['last_pruned_seq'], prune['detail']['last_pruned_sha256']),
                         (last_pruned['seq'], last_pruned['sha256']))
        report = log.verify()
        self.assertEqual((report.ok, report.records), (True, 3))
        log.event('config.reload', 'admin', 'broker.json', {})
        self.assertTrue(log.verify().ok)
        self.assertIsNone(log.prune(retention_days=1), 'nothing left to prune')
        # a forged anchor no longer matches its own digest
        lines = self.lines('2026-10-04.jsonl')
        forged = json.loads(lines[0])
        forged['detail']['last_pruned_seq'] = 1
        self.rewrite('2026-10-04.jsonl', [json.dumps(forged)] + lines[1:])
        self.assertFalse(log.verify().ok)

    def test_the_tail_file_is_never_pruned_and_retention_runs_from_event(self):
        log = self.log(retention=30)
        log.event('role.start', 'titanos', 'standalone', {})
        self.clock.advance(days=400)
        log.event('role.start', 'titanos', 'standalone', {})  # appends, then prunes the year-old file once a day
        names = sorted(p.name for p in self.dir.glob('*.jsonl'))
        self.assertEqual(names, ['2027-11-05.jsonl'])
        self.assertEqual([r['kind'] for r in log.query()], ['role.start', 'audit.prune'])
        self.assertTrue(log.verify().ok)
        quiet = AuditLog(Path(self.tmp.name) / 'other', 1, clock=self.clock)
        quiet.event('role.start', 'titanos', 'standalone', {})
        self.clock.advance(days=10)
        self.assertIsNone(quiet.prune(), 'the only file holds the tail and stays')


class Content(Base):
    def test_no_secret_in_records(self):
        REGISTRY.add(SENTINEL)
        shaped = 'sk-ant-' + 'x' * 30
        log = self.log()
        log.event('credential.put', 'admin ' + SENTINEL, 'openai', {'fingerprint': 'abcd1234', 'note': SENTINEL,
                                                                    'nested': [shaped, {'k': f'Bearer {SENTINEL}'}]})
        text = '\n'.join(p.read_text(encoding='utf-8') for p in self.dir.glob('*.jsonl'))
        self.assertNotIn(SENTINEL, text)
        self.assertNotIn(shaped, text)
        self.assertIn('abcd1234', text)
        self.assertTrue(log.verify().ok)

    def test_query_kinds_and_writable(self):
        log = self.log()
        self.assertTrue(log.writable())
        for kind in ('key.mint', 'auth.failure', 'key.mint', 'budget.stop'):
            log.event(kind, 'a', 's', {})
        self.assertEqual(len(log.query(kind='key.mint')), 2)
        self.assertEqual([r['kind'] for r in log.query(limit=2)], ['key.mint', 'budget.stop'])
        self.assertEqual(log.query(since='2026-10-02'), [])
        with self.assertRaises(ValueError):
            log.event('prompt.text', 'a', 's', {})
        self.assertIn('audit.prune', AUDIT_KINDS)
        blocker = Path(self.tmp.name) / 'not-a-dir'
        blocker.write_text('x')
        self.assertFalse(AuditLog(blocker / 'audit').writable())
        leftovers = {p.name for p in self.dir.iterdir() if not p.name.endswith('.jsonl')}
        self.assertLessEqual(leftovers, {'.audit.lock'}, 'writable() leaves no probe file behind')


if __name__ == '__main__':
    unittest.main()
