"""Broker accounting (WP3): price matching and specificity (most literal pattern, then later effective_from; future rows
ignored); Decimal cost and estimate against hand-computed values; unpriced quantities; the usage ledger's rollups per
key x upstream x model x day, filters, reload from disk and retention prune; decision visibility (own key versus admin)
and decisions scrubbed of sentinel secrets in the ring and the JSONL; ownership binding, visibility, TTL, the per-kind
cap and persistence. In-process only; no server, model or network."""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos.broker.decisions import DecisionLog  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.limits import Estimate  # noqa: E402
from titanos.broker.ownership import OwnershipStore  # noqa: E402
from titanos.broker.pricing import PriceTable, money, specificity  # noqa: E402
from titanos.broker.redact import REGISTRY  # noqa: E402
from titanos.broker.types import Attempt, Candidate, Decision, Principal, Served, Usage  # noqa: E402
from titanos.broker.usage import UsageLedger, UsageRecord, utc_ts  # noqa: E402


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
TODAY = date(2026, 10, 1)
SENTINEL = 'test-openai-not-a-key-' + os.urandom(6).hex()
SHAPED = 'sk-' + 'q' * 32


class Pricing(unittest.TestCase):
    def table(self, rows):
        return PriceTable(rows, today=lambda: TODAY)

    def test_specificity_and_ties(self):
        table = self.table([
            {'match': '*@openai', 'input_per_mtok': '1'},
            {'match': 'gpt-4.1*@openai', 'input_per_mtok': '2'},
            {'match': 'gpt-4.1-mini@openai', 'input_per_mtok': '3'},
            {'match': 'gpt-4.1-mini@openai', 'input_per_mtok': '4', 'effective_from': '2026-09-01'},
            {'match': 'gpt-4.1-mini@openai', 'input_per_mtok': '9', 'effective_from': '2026-12-01'},  # not yet
            {'match': 'gpt-?.1-nano@openai', 'input_per_mtok': '5'},
        ])
        self.assertEqual(table.match('gpt-4.1-mini@openai').input_per_mtok, Decimal('4'))
        self.assertEqual(table.match('gpt-4.1-nano@openai').input_per_mtok, Decimal('5'))
        self.assertEqual(table.match('gpt-4.1@openai').input_per_mtok, Decimal('2'))
        self.assertEqual(table.match('o3@openai').input_per_mtok, Decimal('1'))
        self.assertIsNone(table.match('GPT-4.1@OPENAI'.lower().replace('openai', 'groq')))
        self.assertIsNone(table.match('O3@OPENAI'))  # case-sensitive
        self.assertEqual(table.cost_basis('o3@openai'), '*@openai')
        self.assertEqual(table.cost_basis('x@groq'), 'unpriced')
        self.assertGreater(specificity('abc*'), specificity('ab*'))
        self.assertGreater(specificity('a*b'), specificity('a**b'))
        # identical patterns without dates: the later row wins
        later = self.table([{'match': 'm@u', 'input_per_mtok': '1'}, {'match': 'm@u', 'input_per_mtok': '2'}])
        self.assertEqual(later.match('m@u').input_per_mtok, Decimal('2'))

    def test_decimal_cost_and_estimate(self):
        table = self.table([{'match': 'gpt-4.1-mini@openai', 'input_per_mtok': '0.40', 'output_per_mtok': '1.60',
                             'cached_input_per_mtok': '0.10'}])
        cost = table.cost('gpt-4.1-mini@openai', Usage(input_tokens=1234, cached_input_tokens=234, output_tokens=567,
                                                       reasoning_tokens=100))
        # 1000 x 0.40/1e6 + 234 x 0.10/1e6 + 567 x 1.60/1e6 = 0.0004 + 0.0000234 + 0.0009072
        self.assertEqual(cost, Decimal('0.0013306'))
        self.assertIsInstance(cost, Decimal)
        self.assertEqual(table.estimate('gpt-4.1-mini@openai', Estimate(1000, 500)), Decimal('0.0012'))
        self.assertEqual(money(cost), '0.0013306')
        self.assertEqual(money(Decimal('1.500')), '1.5')
        self.assertEqual(money(Decimal('0E-10')), '0')
        self.assertIsNone(money(None))
        # cached tokens fall back to the input price
        plain = self.table([{'match': 'm@u', 'input_per_mtok': '2', 'output_per_mtok': '4'}])
        self.assertEqual(plain.cost('m@u', Usage(input_tokens=1000, cached_input_tokens=500, output_tokens=0)),
                         Decimal('0.002'))

    def test_media_prices_and_unpriced_quantities(self):
        table = self.table([
            {'match': 'tts@openai', 'per_1k_chars': '15'},
            {'match': 'whisper@openai', 'per_audio_minute': '0.006'},
            {'match': 'image@openai', 'per_image': '0.04'},
        ])
        self.assertEqual(table.cost('tts@openai', Usage(characters=250)), Decimal('3.75'))
        self.assertEqual(table.cost('whisper@openai', Usage(audio_seconds=90.0)), Decimal('0.009'))
        self.assertEqual(table.cost('image@openai', Usage(images=3)), Decimal('0.12'))
        self.assertEqual(table.estimate('tts@openai', Estimate(0, 0, characters=1000)), Decimal('15'))
        self.assertIsNone(table.cost('tts@openai', Usage(input_tokens=10)), 'a quantity without a price is unpriced')
        self.assertIsNone(table.estimate('image@openai', Estimate(10, 0)))
        self.assertEqual(table.cost('image@openai', Usage(images=0)), Decimal(0))
        self.assertIsNone(table.cost('nothing@here', Usage(images=1)))
        with self.assertRaises(Exception):
            PriceTable([{'match': 'x', 'input_per_mtok': '-1'}])


def record(day, key, upstream, model, *, hour=12, cost='0.01', inp=10, out=5, outcome='success', target='cloud',
           basis='row', surface='openai.chat', **extra):
    ts = utc_ts(datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc))
    return UsageRecord(ts=ts, request_id=os.urandom(4).hex(), key_id=key, upstream=upstream, served_ref=model,
                       surface=surface, target_class=target, input_tokens=inp, output_tokens=out, cost_usd=cost,
                       cost_basis=basis, outcome=outcome, **extra)


class Ledger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / 'usage'
        self.now = [datetime(2026, 10, 2, 18, tzinfo=timezone.utc)]

    def tearDown(self):
        self.tmp.cleanup()

    def ledger(self, retention=90):
        return UsageLedger(self.dir, retention_days=retention, clock=lambda: self.now[0])

    def fill(self, ledger):
        d1, d2 = date(2026, 10, 1), date(2026, 10, 2)
        for key in ('keyaaaaaaa', 'keybbbbbbb'):
            for upstream, model in (('openai', 'gpt@openai'), ('groq', 'llama@groq')):
                for day in (d1, d2):
                    ledger.record(record(day, key, upstream, model))
                    ledger.record(record(day, key, upstream, model, cost='0.02', inp=1, out=1))
        ledger.record(record(d2, 'keyaaaaaaa', 'groq', 'llama@groq', cost=None, basis='unpriced', outcome='server_error',
                             inp=None, out=None))
        ledger.record(record(d2, None, None, 'qwen3-8b', cost=None, basis='unpriced', target='local'))

    def test_rollups(self):
        ledger = self.ledger()
        self.fill(ledger)
        self.assertFalse(self.dir.joinpath('.x').exists())
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ['2026-10-01.jsonl', '2026-10-02.jsonl'])
        rows = ledger.rollup(['key', 'upstream', 'model', 'day'])
        self.assertEqual(len(rows), 2 * 2 * 2 + 1)
        cell = next(r for r in rows if (r['key'], r['upstream'], r['day']) == ('keyaaaaaaa', 'openai', '2026-10-01'))
        self.assertEqual((cell['requests'], cell['input_tokens'], cell['output_tokens'], cell['cost_usd']),
                         (2, 11, 6, '0.03'))
        unpriced = next(r for r in rows if (r['key'], r['upstream'], r['day']) == ('keyaaaaaaa', 'groq', '2026-10-02'))
        self.assertEqual((unpriced['requests'], unpriced['unpriced_requests'], unpriced['errors']), (3, 1, 1))
        by_key = {r['key']: r for r in ledger.rollup(['key'])}
        self.assertEqual(by_key['keyaaaaaaa']['requests'], 9)
        self.assertEqual(by_key['keyaaaaaaa']['cost_usd'], '0.12')
        self.assertEqual(by_key[None]['unpriced_requests'], 0, 'local targets are not unpriced cloud requests')
        only = ledger.rollup(['upstream'], since='2026-10-02', until=date(2026, 10, 2), key_id='keybbbbbbb')
        self.assertEqual({r['upstream']: r['requests'] for r in only}, {'groq': 2, 'openai': 2})
        self.assertEqual(ledger.totals(key_id='keybbbbbbb', since=date(2026, 10, 1), until=date(2026, 10, 1)),
                         (Decimal('0.06'), 34))
        with self.assertRaises(ValueError):
            ledger.rollup(['prompt'])
        # a fresh instance (a restart) rebuilds the same rollups from disk
        self.assertEqual(self.ledger().rollup(['key', 'upstream', 'model', 'day']), rows)

    def test_retention_prune(self):
        ledger = self.ledger(retention=1)
        ledger.record(record(date(2026, 9, 20), 'keyaaaaaaa', 'openai', 'gpt@openai'))
        ledger.record(record(date(2026, 10, 2), 'keyaaaaaaa', 'openai', 'gpt@openai'))
        self.assertEqual(len(ledger.rollup(['day'])), 2)
        self.assertEqual(ledger.prune(), 1)
        self.assertEqual([r['day'] for r in ledger.rollup(['day'])], ['2026-10-02'])
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ['2026-10-02.jsonl'])

    def test_records_are_scrubbed(self):
        REGISTRY.add(SENTINEL)
        ledger = self.ledger()
        ledger.record(record(date(2026, 10, 2), 'keyaaaaaaa', 'openai', 'gpt@openai',
                             upstream_request_id='req ' + SENTINEL + ' ' + SHAPED))
        text = (self.dir / '2026-10-02.jsonl').read_text(encoding='utf-8')
        self.assertNotIn(SENTINEL, text)
        self.assertNotIn(SHAPED, text)
        self.assertIn('<redacted>', text)


def decision(key_id, model, *, ts=None, note=''):
    d = Decision(ts=ts or utc_ts(), request_id=os.urandom(4).hex(), decision_id=os.urandom(6).hex(), key_id=key_id,
                 role='standalone', surface='openai.chat', inbound_protocol='openai', requested_model=model,
                 candidates=[Candidate(ref=model, klass='cloud', eligible=True)],
                 attempts=[Attempt(ref=model, outcome='success', status=200)],
                 served=Served(ref=model, klass='cloud', upstream='openai'), usage=Usage(input_tokens=3),
                 notes=[note] if note else [])
    return d


def principal(kind, scopes, pid='keyaaaaaaa'):
    return Principal(id=pid, kind=kind, scopes=frozenset(scopes))


class Decisions(unittest.TestCase):
    def test_visibility_own_versus_admin(self):
        log = DecisionLog(size=100)
        for _ in range(3):
            log.append(decision('keyaaaaaaa', 'gpt@openai'))
            log.append(decision('keybbbbbbb', 'llama@groq'))
        log.append(decision(None, 'qwen3-8b'))
        reader = principal('key', {'broker:read'})
        own = log.query(principal=reader)
        self.assertEqual({d['key_id'] for d in own}, {'keyaaaaaaa'})
        self.assertEqual(len(own), 3)
        self.assertEqual(log.query(principal=reader, key_id='keybbbbbbb'), [])
        admin_key = principal('key', {'broker:admin'}, pid='keycccccc2')
        self.assertEqual(len(log.query(principal=admin_key)), 7)
        self.assertEqual(len(log.query(principal=admin_key, key_id='keybbbbbbb')), 3)
        boot = principal('bootstrap', {'admin'}, pid='bootstrap')
        self.assertEqual(len(log.query(principal=boot, limit=2)), 2)
        self.assertEqual(log.query(principal=boot, limit=1)[0]['requested_model'], 'qwen3-8b')  # newest first
        self.assertEqual(len(log.query(principal=boot, model='llama@groq')), 3)
        anonymous = principal('anonymous_loopback', {'broker:admin', 'inference'}, pid='anonymous')
        self.assertEqual(len(log.query(principal=anonymous)), 7)
        with self.assertRaises(BrokerError) as caught:
            log.query(principal=principal('key', {'inference'}))
        self.assertEqual((caught.exception.code, caught.exception.status), ('scope_denied', 403))
        with self.assertRaises(BrokerError):
            log.query(principal=None)
        late = log.query(principal=boot, since=datetime.now(timezone.utc) + timedelta(hours=1))
        self.assertEqual(late, [])

    def test_ring_is_bounded_and_jsonl_optional(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = DecisionLog(size=5, path=Path(tmp.name) / 'decisions')
        for i in range(8):
            log.append(decision('keyaaaaaaa', f'm{i}', ts='2026-10-01T10:00:00.000Z'))
        self.assertEqual(len(log), 5)
        boot = principal('bootstrap', {'admin'}, pid='bootstrap')
        self.assertEqual([d['requested_model'] for d in log.query(principal=boot)], ['m7', 'm6', 'm5', 'm4', 'm3'])
        lines = (Path(tmp.name) / 'decisions' / '2026-10-01.jsonl').read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(lines), 8)
        self.assertEqual(json.loads(lines[0])['served']['ref'], 'm0')
        self.assertIsNone(DecisionLog(size=3).path)

    def test_decisions_are_scrubbed(self):
        REGISTRY.add(SENTINEL)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = DecisionLog(size=10, path=Path(tmp.name))
        stored = log.append(decision('keyaaaaaaa', 'gpt@openai', note=f'upstream said {SENTINEL} and {SHAPED}'))
        dumped = json.dumps(stored)
        self.assertNotIn(SENTINEL, dumped)
        self.assertNotIn(SHAPED, dumped)
        self.assertIn('<redacted>', dumped)
        on_disk = '\n'.join(p.read_text(encoding='utf-8') for p in Path(tmp.name).glob('*.jsonl'))
        self.assertNotIn(SENTINEL, on_disk)
        self.assertNotIn(SHAPED, on_disk)


class Ownership(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'broker' / 'ownership.json'
        self.t = [1_000_000.0]

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, **kw):
        return OwnershipStore(self.path, ttl_seconds=3600, clock=lambda: self.t[0], **kw)

    def test_binding_and_visibility(self):
        store = self.store()
        a = principal('key', {'inference'}, pid='keyaaaaaaa')
        b = principal('key', {'inference'}, pid='keybbbbbbb')
        admin_key = principal('key', {'admin'}, pid='keycccccc2')
        boot = principal('bootstrap', {'admin'}, pid='bootstrap')
        anon = principal('anonymous_loopback', {'inference'}, pid='anonymous')
        self.assertTrue(store.bind('artifact', '0' * 32 + '.png', a))
        self.assertEqual(store.owner('artifact', '0' * 32 + '.png'), 'keyaaaaaaa')
        for who, expected in ((a, True), (b, False), (admin_key, True), (boot, True), (anon, True), (None, False)):
            with self.subTest(who=getattr(who, 'id', None)):
                self.assertIs(store.allows(who, 'artifact', '0' * 32 + '.png'), expected)
        # an unbound name: admin, bootstrap and anonymous loopback only
        for who, expected in ((a, False), (admin_key, True), (boot, True), (anon, True)):
            self.assertIs(store.allows(who, 'artifact', 'f' * 32 + '.wav'), expected)
        # job-produced artifacts bind by key id (job threads have no request context)
        self.assertTrue(store.bind('artifact', '1' * 32 + '.wav', 'keybbbbbbb'))
        self.assertTrue(store.allows(b, 'artifact', '1' * 32 + '.wav'))
        self.assertTrue(store.bind('job', 'job-1', b))
        self.assertFalse(store.allows(a, 'job', 'job-1'))
        self.assertTrue(store.bind('response', 'resp_1', a))
        self.assertFalse(store.bind('artifact', '2' * 32 + '.png', boot), 'non-key principals bind nothing')
        self.assertIsNone(store.owner('artifact', '2' * 32 + '.png'))
        with self.assertRaises(ValueError):
            store.bind('secret', 'x', a)
        # persisted: a new instance (a restart) sees the bindings
        self.assertEqual(self.store().owner('job', 'job-1'), 'keybbbbbbb')

    def test_a_live_binding_is_never_moved_to_another_key(self):
        store = self.store()
        a = principal('key', {'inference'}, pid='keyaaaaaaa')
        b = principal('key', {'inference'}, pid='keybbbbbbb')
        self.assertTrue(store.bind('artifact', 'x.png', a))
        self.assertFalse(store.bind('artifact', 'x.png', b), 'a second binder cannot take the name')
        self.assertEqual((store.allows(a, 'artifact', 'x.png'), store.allows(b, 'artifact', 'x.png')), (True, False))
        self.t[0] += 10
        self.assertTrue(store.bind('artifact', 'x.png', a), 'the owner may refresh its binding')
        self.t[0] += 3601
        self.assertTrue(store.bind('artifact', 'x.png', b), 'an expired binding may be bound again')
        self.assertEqual(store.owner('artifact', 'x.png'), 'keybbbbbbb')
        store.unbind('artifact', 'x.png')
        self.assertTrue(store.bind('artifact', 'x.png', a), 'an unbound name may be bound again')

    def test_ttl_and_cap(self):
        store = self.store(max_per_kind=3)
        a = principal('key', {'inference'}, pid='keyaaaaaaa')
        for i in range(5):
            self.t[0] += 1
            store.bind('artifact', f'a{i}', a)
        names = [f'a{i}' for i in range(5)]
        self.assertEqual([store.owner('artifact', n) for n in names], [None, None] + ['keyaaaaaaa'] * 3)
        store.bind('job', 'j', a)
        self.assertEqual(store.owner('artifact', 'a4'), 'keyaaaaaaa', 'the cap is per kind')
        self.t[0] += 3601
        self.assertIsNone(store.owner('artifact', 'a4'))
        self.assertFalse(store.allows(a, 'artifact', 'a4'))
        document = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertLessEqual(len([k for k in document if k.startswith('artifact:')]), 3)


class PriceBases(unittest.TestCase):
    """Settlement picks one basis and says which (live lane follow-up: a token-reporting transcription on a per-minute
    row was charged the reservation, USD 0.0141644, with no recorded reason)."""

    def test_bases(self):
        table = PriceTable([
            {'match': 'minute@openai', 'per_audio_minute': '0.006'},
            {'match': 'tokens@openai', 'input_per_mtok': '3', 'output_per_mtok': '5', 'per_audio_minute': '0.003'},
            {'match': 'image@openai', 'per_image': '0.04'},
            {'match': 'gpt-image@openai', 'input_per_mtok': '5', 'output_per_mtok': '40', 'per_image': '0.04'},
            {'match': 'half@openai', 'input_per_mtok': '1'},
        ])
        live = Usage(input_tokens=29, output_tokens=9)
        self.assertEqual(table.price('minute@openai', live).basis, 'unmeasured')  # the live case before the upload
        priced = table.price('minute@openai', live, upload_audio_seconds=2.95)
        self.assertEqual((priced.cost, priced.basis), (Decimal('2.95') * Decimal('0.006') / 60, 'audio_seconds:upload'))
        priced = table.price('minute@openai', Usage(audio_seconds=2.0, input_tokens=29), upload_audio_seconds=9.0)
        self.assertEqual((priced.cost, priced.basis), (Decimal('0.0002'), 'audio_seconds:provider'))
        priced = table.price('tokens@openai', live, upload_audio_seconds=2.95)
        self.assertEqual((priced.cost, priced.basis), ((Decimal(29) * 3 + Decimal(9) * 5) / 1_000_000, 'tokens'))
        priced = table.price('tokens@openai', None, upload_audio_seconds=60.0)
        self.assertEqual((priced.cost, priced.basis), (Decimal('0.003'), 'audio_seconds:upload'))
        # images: a per-image row ignores the token counts gpt-image reports; a row with token prices bills tokens only
        image_usage = Usage(images=2, input_tokens=100, output_tokens=1000)
        self.assertEqual(table.price('image@openai', image_usage).basis, 'images')
        self.assertEqual(table.cost('image@openai', image_usage), Decimal('0.08'))
        self.assertEqual(table.price('gpt-image@openai', image_usage).basis, 'tokens')
        self.assertEqual(table.cost('gpt-image@openai', image_usage), (Decimal(100) * 5 + Decimal(1000) * 40) / 1_000_000)
        # fail closed: a reported token quantity without a price, nothing reported, no row
        self.assertEqual(table.price('half@openai', Usage(input_tokens=10, output_tokens=10)).basis, 'unmeasured')
        self.assertIsNone(table.cost('half@openai', Usage(input_tokens=10, output_tokens=10)))
        self.assertEqual(table.price('minute@openai', Usage()).basis, 'unmeasured')
        self.assertEqual(table.price('minute@openai', None).basis, 'unmeasured')
        self.assertEqual(table.price('nothing@here', live).basis, 'unpriced')
        record_ = UsageRecord(ts=utc_ts(), request_id='r', price_basis='audio_seconds:upload')
        self.assertEqual(UsageRecord.from_dict(record_.to_dict()).price_basis, 'audio_seconds:upload')
        self.assertIsNone(UsageRecord.from_dict({'ts': utc_ts(), 'request_id': 'old'}).price_basis)


if __name__ == '__main__':
    unittest.main()
