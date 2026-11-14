"""Broker admission (WP3): RPM, TPM and concurrency give 429 with Retry-After; usd/day, usd/month, tokens/day and
max_request_usd give 402 with x-should-retry: false; a budgeted key on an unpriced target gives 402
budget_unpriced_target while local targets cost nothing; the injected max_tokens estimate; settlement reconciles the
TPM bucket and the budget; release refunds; a refusal consumes nothing; UTC day and month rollover with a fake clock;
spend seeded from the usage ledger survives a restart; soft budgets; budget.stop audit records. In-process only; no
server, model or network."""
import dataclasses
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib.util
import ipaddress
from pathlib import Path
import socket
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos.broker.audit import AuditLog  # noqa: E402
from titanos.broker.envelopes import render_error  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.limits import Estimate, Limiter, estimate_input_tokens  # noqa: E402
from titanos.broker.pricing import PriceTable  # noqa: E402
from titanos.broker.types import KeyBudget, KeyLimits, Principal, Target, Usage  # noqa: E402
from titanos.broker.usage import UsageLedger, UsageRecord, utc_ts  # noqa: E402
from titanos.settings import Settings  # noqa: E402


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
CLOUD = Target(ref='gpt-4.1-mini@openai', klass='cloud', upstream_id='openai', upstream_model='gpt-4.1-mini')
LOCAL = Target(ref='qwen3-8b', klass='local', local_name='qwen3-8b')


class Clock:
    def __init__(self):
        self.mono = 5000.0
        self.now = datetime(2026, 10, 1, 23, 59, 0, tzinfo=timezone.utc)

    def advance(self, seconds):
        self.mono += seconds
        self.now += timedelta(seconds=seconds)


def key(limits=None, budget=None, kid='abcdefgh23'):
    return Principal(id=kid, kind='key', scopes=frozenset({'inference', 'cloud'}), limits=limits, budget=budget)


class Base(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.settings = Settings()
        self.tmp = tempfile.TemporaryDirectory()
        self.audit = AuditLog(Path(self.tmp.name) / 'audit', clock=lambda: self.clock.now)

    def tearDown(self):
        self.tmp.cleanup()

    def limiter(self, **kw):
        kw.setdefault('clock', lambda: self.clock.mono)
        kw.setdefault('now', lambda: self.clock.now)
        kw.setdefault('audit', self.audit)
        return Limiter(self.settings, **kw)

    def refused(self, code, status, call):
        with self.assertRaises(BrokerError) as caught:
            call()
        self.assertEqual((caught.exception.code, caught.exception.status), (code, status))
        return caught.exception


class RateLimits(Base):
    def test_rpm_bucket(self):
        limiter = self.limiter()
        p = key(KeyLimits(rpm=2))
        for _ in range(2):
            limiter.admit(p, LOCAL, Estimate(10, 0)).settle(None, None)
        exc = self.refused('rate_limit_exceeded', 429, lambda: limiter.admit(p, LOCAL, Estimate(10, 0)))
        self.assertAlmostEqual(exc.retry_after, 30.0, places=3)
        headers = exc.response_headers()
        self.assertEqual(headers['Retry-After'], '30')
        self.assertNotIn('x-should-retry', headers)
        response = render_error('openai', exc)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers['retry-after'], '30')
        self.clock.advance(30)
        limiter.admit(p, LOCAL, Estimate(10, 0)).settle(None, None)

    def test_tpm_bucket_and_settlement_reconciles(self):
        limiter = self.limiter()
        p = key(KeyLimits(tpm=1000))
        first = limiter.admit(p, LOCAL, Estimate(600, 0))
        exc = self.refused('rate_limit_exceeded', 429, lambda: limiter.admit(p, LOCAL, Estimate(600, 0)))
        self.assertGreater(exc.retry_after, 0)
        first.settle(Usage(input_tokens=80, output_tokens=20), None)  # 500 tokens over-reserved come back
        limiter.admit(p, LOCAL, Estimate(600, 0)).release()
        # an estimate above the whole bucket needs a full bucket (not an impossible one)
        self.clock.advance(60)
        limiter.admit(p, LOCAL, Estimate(5000, 0)).settle(None, None)
        self.refused('rate_limit_exceeded', 429, lambda: limiter.admit(p, LOCAL, Estimate(5000, 0)))

    def test_tpm_default_output_cap_counts_when_uncapped(self):
        self.assertEqual(estimate_input_tokens(0), 0)
        self.assertEqual(estimate_input_tokens(10), 4)
        self.assertEqual(estimate_input_tokens(3), 1)
        limiter = self.limiter()
        p = key(KeyLimits(tpm=self.settings.broker_default_max_output_tokens + 50))
        adm = limiter.admit(p, LOCAL, Estimate(100, None))  # 100 + the default output cap
        self.assertIsNone(adm.injected_max_tokens, 'no budget, no injection')
        self.refused('rate_limit_exceeded', 429, lambda: limiter.admit(p, LOCAL, Estimate(100, 0)))

    def test_concurrency(self):
        limiter = self.limiter()
        p = key(KeyLimits(concurrent=1))
        first = limiter.admit(p, LOCAL, Estimate(1, 0))
        exc = self.refused('concurrency_exceeded', 429, lambda: limiter.admit(p, LOCAL, Estimate(1, 0)))
        self.assertEqual(exc.response_headers()['Retry-After'], '1')
        first.release()
        with limiter.admit(p, LOCAL, Estimate(1, 0)):
            self.refused('concurrency_exceeded', 429, lambda: limiter.admit(p, LOCAL, Estimate(1, 0)))
        limiter.admit(p, LOCAL, Estimate(1, 0)).settle(None, None)

    def test_unlimited_principals(self):
        limiter = self.limiter()
        boot = Principal(id='bootstrap', kind='bootstrap', scopes=frozenset({'admin'}))
        for _ in range(50):
            limiter.admit(boot, CLOUD, Estimate(10, 10), price=None).settle(None, None)


class Budgets(Base):
    def test_usd_per_day_and_x_should_retry(self):
        limiter = self.limiter()
        p = key(budget=KeyBudget(usd_per_day=Decimal('1.00')))
        limiter.admit(p, CLOUD, Estimate(10, 10), price=Decimal('0.60'))
        exc = self.refused('budget_exceeded', 402, lambda: limiter.admit(p, CLOUD, Estimate(10, 10), price=Decimal('0.60')))
        self.assertEqual(exc.response_headers()['x-should-retry'], 'false')
        self.assertEqual(exc.origin, 'policy')
        response = render_error('anthropic', exc)
        self.assertEqual((response.status_code, response.headers['x-should-retry']), (402, 'false'))
        limiter.admit(p, CLOUD, Estimate(10, 10), price=Decimal('0.40'))  # exactly the remainder

    def test_usd_per_month_across_days_and_day_rollover(self):
        limiter = self.limiter()
        p = key(budget=KeyBudget(usd_per_day=Decimal('1.00'), usd_per_month=Decimal('1.50')))
        limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.90')).settle(None, Decimal('0.90'))
        self.refused('budget_exceeded', 402, lambda: limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.20')))
        self.clock.advance(120)  # 2026-10-02 00:01 UTC: a new day window
        limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.50')).settle(None, Decimal('0.50'))
        exc = self.refused('budget_exceeded', 402, lambda: limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.20')))
        self.assertIn('monthly', str(exc))
        self.clock.now = datetime(2026, 11, 1, 0, 0, 1, tzinfo=timezone.utc)
        limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.90'))

    def test_tokens_per_day_and_max_request_usd(self):
        limiter = self.limiter()
        p = key(budget=KeyBudget(tokens_per_day=1000))
        limiter.admit(p, CLOUD, Estimate(500, 100), price=Decimal('0'))
        self.refused('budget_exceeded', 402, lambda: limiter.admit(p, CLOUD, Estimate(500, 100), price=Decimal('0')))
        q = key(budget=KeyBudget(max_request_usd=Decimal('0.10')), kid='bbbbbbbb23')
        self.refused('budget_exceeded', 402, lambda: limiter.admit(q, CLOUD, Estimate(1, 1), price=Decimal('0.11')))
        limiter.admit(q, CLOUD, Estimate(1, 1), price=Decimal('0.10'))

    def test_unpriced_target_with_a_budget_fails_closed(self):
        limiter = self.limiter()
        p = key(budget=KeyBudget(usd_per_day=Decimal('5')))
        exc = self.refused('budget_unpriced_target', 402, lambda: limiter.admit(p, CLOUD, Estimate(10, 10), price=None))
        self.assertEqual(exc.response_headers()['x-should-retry'], 'false')
        # with a price table but no matching row (automatic pricing)
        table = PriceTable([{'match': '*@groq', 'input_per_mtok': '1', 'output_per_mtok': '1'}])
        self.refused('budget_unpriced_target', 402,
                     lambda: self.limiter(prices=table).admit(p, CLOUD, Estimate(10, 10)))
        # local inference costs 0: a budgeted key is only rate-limited there
        adm = limiter.admit(p, LOCAL, Estimate(10, None), price=None)
        self.assertIsNone(adm.injected_max_tokens)
        self.assertEqual(adm.price, Decimal(0))
        # an unbudgeted key may use an unpriced target
        limiter.admit(key(kid='cccccccc23'), CLOUD, Estimate(10, 10), price=None)

    def test_injected_max_tokens(self):
        self.settings = dataclasses.replace(Settings(), broker_default_max_output_tokens=256)
        table = PriceTable([{'match': 'gpt-4.1-mini@openai', 'input_per_mtok': '0.40', 'output_per_mtok': '1.60'}])
        limiter = self.limiter(prices=table)
        p = key(budget=KeyBudget(usd_per_day=Decimal('1')))
        adm = limiter.admit(p, CLOUD, Estimate(1000, None))
        self.assertEqual(adm.injected_max_tokens, 256)
        self.assertEqual(adm.estimate.max_output_tokens, 256)
        self.assertEqual(adm.price, Decimal(1000) * Decimal('0.40') / 1_000_000 + Decimal(256) * Decimal('1.60') / 1_000_000)
        self.assertEqual(adm.reserved_tokens, 1256)
        capped = limiter.admit(p, CLOUD, Estimate(1000, 50))
        self.assertIsNone(capped.injected_max_tokens)
        none = limiter.admit(p, CLOUD, Estimate(1000, 0))  # a surface without output tokens
        self.assertIsNone(none.injected_max_tokens)

    def test_settle_and_release_reconcile_reservations(self):
        limiter = self.limiter()
        p = key(budget=KeyBudget(usd_per_day=Decimal('1.00')))
        adm = limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.60'))
        self.assertEqual(limiter.remaining(p)['usd_today'], '0.40')
        adm.settle(Usage(input_tokens=1, output_tokens=1), Decimal('0.10'))
        adm.settle(Usage(), Decimal('5'))  # idempotent
        self.assertEqual(limiter.remaining(p)['usd_today'], '0.90')
        limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.85')).release()
        self.assertEqual(limiter.remaining(p)['usd_today'], '0.90')
        unknown = limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.30'))
        unknown.settle(None, None)  # no reported cost: the reserved estimate is spent (conservative)
        self.assertEqual(limiter.remaining(p)['usd_today'], '0.60')

    def test_a_refusal_consumes_nothing(self):
        limiter = self.limiter()
        p = key(KeyLimits(rpm=1, concurrent=1), KeyBudget(usd_per_day=Decimal('0.10')))
        self.refused('budget_exceeded', 402, lambda: limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('1')))
        adm = limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.05'))  # the RPM token and the slot are intact
        adm.settle(None, Decimal('0.05'))

    def test_spend_survives_restart_through_the_usage_ledger(self):
        ledger = UsageLedger(Path(self.tmp.name) / 'usage', clock=lambda: self.clock.now)
        ledger.record(UsageRecord(ts=utc_ts(self.clock.now), request_id='r1', key_id='abcdefgh23', target_class='cloud',
                                  upstream='openai', served_ref=CLOUD.ref, input_tokens=10, output_tokens=5,
                                  cost_usd='0.95', cost_basis='gpt-4.1-mini@openai'))
        restarted = self.limiter(usage=ledger)
        p = key(budget=KeyBudget(usd_per_day=Decimal('1.00')))
        self.refused('budget_exceeded', 402, lambda: restarted.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.10')))
        restarted.admit(p, CLOUD, Estimate(1, 1), price=Decimal('0.05'))

    def test_a_budget_without_any_bound_is_no_budget(self):
        # the same predicate as KeyStore's broker-role rule: {hard: true} alone bounds nothing
        limiter = self.limiter()
        p = key(budget=KeyBudget())
        adm = limiter.admit(p, CLOUD, Estimate(10, None), price=None)
        self.assertIsNone(adm.injected_max_tokens)
        adm.settle(None, None)

    def test_fallback_retarget_rechecks_the_budget_without_new_rate_tokens(self):
        other = Target(ref='gpt-4.1@azure', klass='cloud', upstream_id='azure', upstream_model='gpt-4.1')
        self.settings = dataclasses.replace(Settings(), broker_default_max_output_tokens=128)
        limiter = self.limiter()
        p = key(KeyLimits(rpm=1, concurrent=1), KeyBudget(usd_per_day=Decimal('1.00')))
        adm = limiter.admit(p, CLOUD, Estimate(10, None), price=Decimal('0.10'))
        self.assertEqual(adm.injected_max_tokens, 128)
        self.assertEqual(limiter.remaining(p)['usd_today'], '0.90')
        # the next attempt's target: no second RPM token, no second concurrency slot, a swapped reservation
        self.assertIs(adm.retarget(other, price=Decimal('0.20')), adm)
        self.assertEqual((adm.target, adm.price, adm.injected_max_tokens), (other, Decimal('0.20'), 128))
        self.assertEqual(limiter.remaining(p)['usd_today'], '0.80')
        # a refused retarget leaves the admission and its reservation unchanged
        self.refused('budget_exceeded', 402, lambda: adm.retarget(CLOUD, price=Decimal('1.50')))
        self.refused('budget_unpriced_target', 402, lambda: adm.retarget(CLOUD, price=None))
        self.assertEqual((adm.target, adm.price), (other, Decimal('0.20')))
        self.assertEqual(limiter.remaining(p)['usd_today'], '0.80')
        # a local fallback costs nothing and injects nothing
        adm.retarget(LOCAL)
        self.assertEqual((adm.price, adm.injected_max_tokens, adm.estimate.max_output_tokens), (Decimal(0), None, None))
        self.assertEqual(limiter.remaining(p)['usd_today'], '1.00')
        adm.retarget(other, price=Decimal('0.30'))
        adm.settle(None, Decimal('0.25'))
        self.assertEqual(limiter.remaining(p)['usd_today'], '0.75')
        with self.assertRaises(RuntimeError):
            adm.retarget(CLOUD, price=Decimal('0.01'))

    def test_soft_budget_admits_and_flags(self):
        limiter = self.limiter()
        p = key(budget=KeyBudget(usd_per_day=Decimal('0.10'), hard=False))
        adm = limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('1'))
        self.assertTrue(adm.over_budget)

    def test_budget_stops_are_audited_once_per_key_code_and_day(self):
        limiter = self.limiter()
        p = key(budget=KeyBudget(usd_per_day=Decimal('0.01')))
        for _ in range(3):
            self.refused('budget_exceeded', 402, lambda: limiter.admit(p, CLOUD, Estimate(1, 1), price=Decimal('1')))
        stops = self.audit.query(kind='budget.stop')
        self.assertEqual(len(stops), 1)
        self.assertEqual((stops[0]['actor'], stops[0]['subject'], stops[0]['detail']['code']),
                         ('abcdefgh23', CLOUD.ref, 'budget_exceeded'))
        self.assertEqual(limiter.budget_stops, 3)


class AudioBudgets(Base):
    """Transcription and translation reserve the upload's duration, so a budgeted key fails closed even when the answer
    carries no usage (response_format text, srt or vtt)."""

    def test_upload_duration_estimate(self):
        import io
        import wave
        from titanos.broker.limits import AUDIO_FLOOR_BYTES_PER_SECOND, estimate_audio_seconds
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b'\x00\x00' * 16000 * 3)
        self.assertAlmostEqual(estimate_audio_seconds(buffer.getvalue()), 3.0)
        self.assertEqual(estimate_audio_seconds(b'\x01' * (AUDIO_FLOOR_BYTES_PER_SECOND * 42)), 42.0)
        self.assertEqual(estimate_audio_seconds(b''), 1.0)  # never zero
        self.assertEqual(estimate_audio_seconds(b'RIFF\x00\x00\x00\x00WAVE' + b'\x00' * 2988), 3.0)  # broken header

    def test_router_estimate_and_pricing(self):
        from titanos.broker.router import estimate_for
        from titanos.broker.surfaces import SURFACES
        from titanos.broker.types import BodySummary, Hints, RouteContext
        surface = SURFACES['openai.audio.transcriptions']
        ctx = RouteContext(surface=surface, requested_model='whisper-1@openai', principal=key(), hints=Hints(),
                           body=BodySummary(audio_seconds=90.0, bytes=1000))
        estimate = estimate_for(ctx)
        self.assertEqual((estimate.audio_seconds, estimate.max_output_tokens), (90.0, 0))
        no_file = estimate_for(RouteContext(surface=surface, requested_model='m', principal=key(), hints=Hints(),
                                            body=BodySummary(bytes=5000)))
        self.assertEqual(no_file.audio_seconds, 5.0)
        minute = PriceTable([{'match': 'whisper-1@openai', 'per_audio_minute': '0.006'}])
        self.assertEqual(minute.estimate('whisper-1@openai', estimate), Decimal('0.009'))
        tokens = PriceTable([{'match': '*@openai', 'input_per_mtok': '6', 'output_per_mtok': '10'}])
        # 90 s -> 2250 audio input tokens and 900 transcript tokens
        self.assertEqual(tokens.estimate('gpt-4o-transcribe@openai', estimate),
                         Decimal(2250) * 6 / 1_000_000 + Decimal(900) * 10 / 1_000_000)
        chars_only = PriceTable([{'match': '*@openai', 'per_1k_chars': '0.3'}])
        self.assertIsNone(chars_only.estimate('whisper-1@openai', estimate))  # no audio price: unpriced (402)

    def test_unknown_usage_settles_at_the_reservation(self):
        table = PriceTable([{'match': 'whisper-1@openai', 'per_audio_minute': '0.60'}])
        self.assertIsNone(table.cost('whisper-1@openai', Usage()))  # nothing measured: unknown, not zero
        self.assertEqual(table.cost('whisper-1@openai', Usage(audio_seconds=0)), Decimal(0))
        limiter = self.limiter(prices=table)
        p = key(budget=KeyBudget(usd_per_day=Decimal('1.00')))
        target = Target(ref='whisper-1@openai', klass='cloud', upstream_id='openai', upstream_model='whisper-1')
        adm = limiter.admit(p, target, Estimate(max_output_tokens=0, audio_seconds=60.0))
        self.assertEqual(adm.price, Decimal('0.60'))
        adm.settle(None, table.cost(target.ref, Usage()))
        self.assertEqual(adm.charged, Decimal('0.60'))
        self.assertEqual(Decimal(limiter.remaining(p)['usd_today']), Decimal('0.40'))
        self.refused('budget_exceeded', 402,
                     lambda: limiter.admit(p, target, Estimate(max_output_tokens=0, audio_seconds=60.0)))


class MeasuredAudio(Base):
    """Live lane follow-up: the upload's duration is measured from the container (WAV, also streamed WAV with the
    0xFFFFFFFF placeholder sizes OpenAI's speech answer carries; FLAC STREAMINFO; MP3 frame headers), so a per-minute
    row is settled on the real duration instead of the 1000 B/s floor (the live run's 141644-byte, 2.95 s WAV was
    reserved as 141.644 s)."""

    @staticmethod
    def wav(seconds: float, rate: int = 24000, *, streamed: bool = False, channels: int = 1) -> bytes:
        import io
        import struct
        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(b'\x00\x00' * channels * int(round(seconds * rate)))
        data = bytearray(buffer.getvalue())
        if streamed:
            data[4:8] = struct.pack('<I', 0xFFFFFFFF)
            data[40:44] = struct.pack('<I', 0xFFFFFFFF)
        return bytes(data)

    @staticmethod
    def mp3(frames: int, *, id3: bool = True, tag: bool = True) -> bytes:
        # MPEG-1 Layer III, 128 kbit/s, 44.1 kHz, no padding: 417-byte frames of 1152 samples
        frame = b'\xff\xfb\x90\x00' + b'\x00' * 413
        head = b'ID3\x04\x00\x00\x00\x00\x00\x0a' + b'\x00' * 10 if id3 else b''
        return head + frame * frames + (b'TAG' + b'\x00' * 125 if tag else b'')

    @staticmethod
    def flac(total_samples: int, rate: int = 16000) -> bytes:
        packed = (rate << 44) | (0 << 41) | (15 << 36) | total_samples
        info = b'\x10\x00\x10\x00' + b'\x00' * 6 + packed.to_bytes(8, 'big') + b'\x00' * 16
        return b'fLaC' + b'\x80\x00\x00\x22' + info + b'\x00' * 100

    def test_measured_durations(self):
        from titanos.broker.limits import estimate_audio_seconds, measure_audio_seconds
        live = self.wav(2.95, streamed=True)
        self.assertEqual(len(live), 141644)
        self.assertAlmostEqual(measure_audio_seconds(live), 2.95)
        self.assertAlmostEqual(estimate_audio_seconds(live), 2.95)  # was 141.644 s at the floor
        self.assertAlmostEqual(measure_audio_seconds(self.wav(1.5, 16000, channels=2)), 1.5)
        self.assertAlmostEqual(measure_audio_seconds(self.mp3(100)), 100 * 1152 / 44100)
        self.assertAlmostEqual(measure_audio_seconds(self.mp3(50, id3=False, tag=False)), 50 * 1152 / 44100)
        self.assertAlmostEqual(measure_audio_seconds(self.flac(48000)), 3.0)
        for unknown in (b'', b'\x00' * 5000, b'OggS' + b'\x00' * 400, self.flac(0),
                        self.mp3(100)[:-200] + b'\x00' * 900,  # a broken frame chain is never guessed
                        b'RIFF\x00\x00\x00\x00WAVE' + b'\x00' * 2988):
            with self.subTest(unknown[:8]):
                self.assertIsNone(measure_audio_seconds(unknown))
        self.assertEqual(estimate_audio_seconds(b'\x00' * 5000), 5.0)  # unmeasurable: the floor, unchanged

    def test_reservation_on_a_row_with_both_prices_takes_the_larger(self):
        both = PriceTable([{'match': 'm@u', 'input_per_mtok': '3', 'output_per_mtok': '5', 'per_audio_minute': '0.6'}])
        estimate = Estimate(max_output_tokens=0, audio_seconds=60.0, token_output=False)
        self.assertEqual(both.estimate('m@u', estimate), Decimal('0.6'))  # per minute 0.6 > tokens 0.0075
        cheap = PriceTable([{'match': 'm@u', 'input_per_mtok': '3', 'output_per_mtok': '5',
                             'per_audio_minute': '0.0001'}])
        self.assertEqual(cheap.estimate('m@u', estimate), (Decimal(1500) * 3 + Decimal(600) * 5) / 1_000_000)

    def test_non_token_estimates_never_get_an_output_cap(self):
        limiter = self.limiter(prices=PriceTable([{'match': '*', 'per_image': '0.02', 'input_per_mtok': '1',
                                                   'output_per_mtok': '1'}]))
        p = key(budget=KeyBudget(usd_per_day=Decimal('10')))
        image = limiter.admit(p, CLOUD, Estimate(max_output_tokens=None, images=1, token_output=False))
        self.assertIsNone(image.injected_max_tokens)
        self.assertEqual(image.reserved_tokens, 0)
        self.assertEqual(image.price, Decimal('0.02'))
        image.release()
        chat = limiter.admit(p, CLOUD, Estimate(input_tokens=10))
        self.assertEqual(chat.injected_max_tokens, limiter.default_output)
        chat.release()


if __name__ == '__main__':
    unittest.main()
