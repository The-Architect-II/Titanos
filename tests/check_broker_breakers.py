"""Broker circuit breakers and the lazy HealthMonitor (WP6), all on a fake clock.

- a breaker opens after broker_breaker_failures consecutive counted failures (server_error, network, timeout);
- open durations grow exponentially (base x 2^(opens-1), capped at the max) within +-20% jitter;
- half-open admits exactly one probe; success closes, a failure re-opens with the duration doubled;
- client_error, auth, cancelled, refused and 429 are never counted; 429 sets a Retry-After cooldown without opening;
- the router skips open breakers and cooling upstreams (503 breaker_open with the minimum Retry-After), with zero hits;
- per-model rate limits (extras ratelimit_per_model: OpenAI, Anthropic, Gemini): a 429 on one model cools down that model
  only; another model (or modality) on the same key keeps serving, and the cooling model itself is answered 429
  upstream_rate_limited with Retry-After (the provider's own meaning), never 503; failures still open the shared breaker;
- HealthMonitor probes only half-open or recently used upstreams (never idle ones) and nothing when health_probes is off;
- rate-limit headroom is read from response headers.
"""
from __future__ import annotations

import asyncio
import random
import unittest

import broker_wp6_support as S
from broker_mocks import AnthropicMock, OpenAIMock, netguard

from titanos.broker import inbound
from titanos.broker.adapters.base import ProbeResult
from titanos.broker.errors import BrokerError
from titanos.broker.health import BreakerBoard, BreakerOpen, CircuitBreaker, HealthMonitor, Permit, ratelimit_headroom
from titanos.broker.role import parse_parent_headers, switches_for
from titanos.broker.surfaces import SURFACES
from titanos.broker.types import RouteContext


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def breaker(clock, **kw) -> CircuitBreaker:
    return CircuitBreaker('up', 'cred', failures=kw.pop('failures', 3), base=kw.pop('base', 2.0),
                          maximum=kw.pop('maximum', 300.0), clock=clock, rng=random.Random(7), **kw)


class Breaker(unittest.TestCase):
    def test_opens_after_n_counted_failures(self):
        clock = Clock()
        b = breaker(clock)
        for kind in ('server_error', 'network'):
            b.record(kind)
            self.assertEqual(b.state, 'closed')
        b.record('success')  # a success resets the run of failures
        for kind in ('timeout', 'server_error'):
            b.record(kind)
        self.assertEqual(b.state, 'closed')
        b.record('network')
        self.assertEqual(b.state, 'open')
        verdict = b.allow()
        self.assertIsInstance(verdict, BreakerOpen)
        self.assertFalse(verdict)
        self.assertGreater(verdict.retry_after, 0)

    def test_exponential_durations_within_jitter_bounds(self):
        clock = Clock()
        b = breaker(clock, failures=1, base=2.0, maximum=60.0)
        expected = [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
        for opens, base in enumerate(expected, start=1):
            b.record('server_error')  # closed -> open, or half-open probe failure -> open again
            self.assertEqual((b.state, b.opens), ('open', opens))
            self.assertEqual(b.duration(opens), base)
            self.assertGreaterEqual(b.last_duration, base * 0.8 - 1e-9)
            self.assertLessEqual(b.last_duration, base * 1.2 + 1e-9)
            clock.advance(b.last_duration + 0.001)
            self.assertEqual(b.state, 'half_open')
            self.assertTrue(b.allow().probe)
        durations = set()
        for seed in range(40):  # jitter really varies, and always within +-20%
            c = CircuitBreaker('u', None, failures=1, base=10.0, maximum=10.0, clock=Clock(), rng=random.Random(seed))
            c.record('network')
            self.assertTrue(8.0 <= c.last_duration <= 12.0)
            durations.add(round(c.last_duration, 6))
        self.assertGreater(len(durations), 20)

    def test_single_half_open_probe(self):
        clock = Clock()
        b = breaker(clock, failures=1)
        b.record('server_error')
        clock.advance(b.last_duration + 0.01)
        first = b.allow()
        self.assertIsInstance(first, Permit)
        self.assertTrue(first.probe)
        second = b.allow()
        self.assertIsInstance(second, BreakerOpen)
        self.assertEqual(second.reason, 'probe_in_flight')
        self.assertTrue(b.allow(consume=False) is not None)
        b.record('success')
        self.assertEqual((b.state, b.opens), ('closed', 0))
        self.assertFalse(b.allow().probe)
        # a failing probe re-opens with the duration doubled
        b.record('server_error')
        first_duration = b.duration(b.opens)
        clock.advance(b.last_duration + 0.01)
        self.assertTrue(b.allow().probe)
        b.record('timeout')
        self.assertEqual((b.state, b.opens, b.duration(b.opens)), ('open', 2, first_duration * 2))

    def test_client_error_auth_and_429_are_never_counted(self):
        clock = Clock()
        b = breaker(clock, failures=2)
        for _ in range(10):
            for kind in ('client_error', 'auth', 'cancelled', 'refused'):
                b.record(kind)
        self.assertEqual((b.state, b.consecutive), ('closed', 0))
        b.record('rate_limited', retry_after=7.0)
        b.record('rate_limited', headers={'retry-after': '3'})
        self.assertEqual(b.state, 'closed')
        verdict = b.allow()
        self.assertIsInstance(verdict, BreakerOpen)
        self.assertEqual(verdict.reason, 'cooldown')
        self.assertAlmostEqual(verdict.retry_after, 7.0)
        clock.advance(7.01)
        self.assertIsInstance(b.allow(), Permit)
        self.assertEqual(b.opens, 0)

    def test_board_transitions_audit_and_headroom(self):
        events = []

        class Audit:
            def event(self, kind, actor, subject, detail):
                events.append((kind, subject))
        clock = Clock()
        board = BreakerBoard(S.settings(broker_breaker_failures=1), clock=clock, rng=random.Random(1), audit=Audit())
        b = board.get('openai', 'cred')
        self.assertIs(board.get('openai', 'cred'), b)
        self.assertIsNot(board.get('openai', 'other'), b)
        b.record('server_error', headers={'x-ratelimit-remaining-requests': '12', 'x-ratelimit-remaining-tokens': '900',
                                          'x-ratelimit-reset-requests': '1s'})
        self.assertEqual(events, [('breaker.open', 'openai')])
        self.assertEqual(board.transitions, {'closed->open': 1})
        snap = board.snapshot()[0]
        self.assertEqual((snap['breaker'], snap['ratelimit']['requests_remaining']), ('open', 12.0))
        self.assertEqual(ratelimit_headroom({'anthropic-ratelimit-requests-remaining': '5',
                                             'anthropic-ratelimit-tokens-remaining': '100'}),
                         {'requests_remaining': 5.0, 'tokens_remaining': 100.0})
        self.assertEqual(board.reset('openai'), 2)
        self.assertEqual(b.state, 'closed')


class RouterSkipsOpenBreakers(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        self.openai, self.anthropic = OpenAIMock(), AnthropicMock()
        self.openai.start()
        self.anthropic.start()
        doc = {'policy': {'cloud_policy': 'fallback'},
               'upstreams': {'openai': S.upstream(self.openai, 'openai', credential='oa',
                                                  models=['gpt-4o-mini', 'gpt-4o']),
                             'anthropic': S.upstream(self.anthropic, 'anthropic', credential='an',
                                                     models=['claude-sonnet-4-5'])},
               'virtual_models': {'both': {'capability': 'chat', 'targets': [{'ref': 'gpt-4o-mini@openai'},
                                                                              {'ref': 'claude-sonnet-4-5@anthropic'}]}}}
        self.lab = S.Lab(doc)
        self.clock = Clock()
        self.lab.broker._components['breakers'] = BreakerBoard(S.settings(broker_breaker_failures=1), clock=self.clock,
                                                               rng=random.Random(3))

    def tearDown(self):
        self.lab.close()
        self.openai.stop()
        self.anthropic.stop()
        self.assertEqual(netguard.attempts(), [])

    def test_a_429_cools_down_one_model_only(self):
        from titanos.broker.health import breaker_for
        broker = self.lab.broker
        principal = S.key('inference', 'cloud')
        row = SURFACES['openai.chat']
        router = broker.snapshot().router

        def context(model):
            return RouteContext(row, model, principal, parse_parent_headers({}, broker.settings, principal),
                                inbound.summarize(row, {}, None, 10))
        snapshot = broker.snapshot()
        target = snapshot.catalog.qualified_target('gpt-4o-mini', 'openai')
        upstream = snapshot.config.upstreams['openai']
        breaker_for(broker.breakers, target, upstream).record('rate_limited', retry_after=6.0)
        with self.assertRaises(BrokerError) as caught:
            router.resolve(context('gpt-4o-mini@openai'))
        self.assertEqual((caught.exception.code, caught.exception.status), ('upstream_rate_limited', 429))
        self.assertLessEqual(caught.exception.retry_after, 6.0)
        self.assertEqual([t.ref for t in router.resolve(context('gpt-4o@openai')).chain], ['gpt-4o@openai'])
        self.assertEqual(broker.breakers.get('openai', 'oa').state, 'closed')
        self.clock.advance(7)
        self.assertEqual(len(router.resolve(context('gpt-4o-mini@openai')).chain), 1)
        # counted failures still open the shared breaker for every model of that key
        breaker_for(broker.breakers, target, upstream).record('server_error')
        with self.assertRaises(BrokerError) as caught:
            router.resolve(context('gpt-4o@openai'))
        self.assertEqual(caught.exception.code, 'breaker_open')
        self.assertEqual(len(self.openai.hits), 0)

    def test_open_breakers_and_cooldowns_are_skipped_before_egress(self):
        broker = self.lab.broker
        principal = S.key('inference', 'cloud')
        row = SURFACES['openai.chat']
        context = RouteContext(row, 'both', principal, parse_parent_headers({}, broker.settings, principal),
                               inbound.summarize(row, {}, None, 10))
        router = broker.snapshot().router
        board = broker.breakers
        board.get('openai', 'oa').record('server_error')
        plan = router.resolve(context)
        self.assertEqual([t.ref for t in plan.chain], ['claude-sonnet-4-5@anthropic'])
        board.get('anthropic', 'an').record('rate_limited', retry_after=4.0)
        with self.assertRaises(BrokerError) as caught:
            router.resolve(context)
        error = caught.exception
        self.assertEqual((error.code, error.status), ('breaker_open', 503))
        self.assertLessEqual(error.retry_after, 4.0)
        self.assertEqual(error.response_headers()['Retry-After'], str(int(-(-error.retry_after // 1))))
        self.assertEqual(len(self.openai.hits) + len(self.anthropic.hits), 0)
        self.clock.advance(400)
        self.assertEqual(len(router.resolve(context).chain), 2)

    def test_a_client_deadline_never_counts_against_the_shared_breaker(self):
        """An expired X-Titanos-Deadline-Ms (before the attempt, or while waiting for the upstream's first byte) is 504
        deadline_exceeded with a neutral breaker outcome: tiny deadlines from one key cannot open the shared
        (upstream, credential) breaker for every other key."""
        from titanos.broker.dispatch import BrokerDispatch
        broker = self.lab.broker
        broker._components['breakers'] = BreakerBoard(S.settings(broker_breaker_failures=2), clock=self.clock,
                                                      rng=random.Random(3))
        app = BrokerDispatch(S.FakeLocal(), broker)
        hasty = S.key('inference', 'cloud', id='khasty00001')
        body = {'model': 'gpt-4o-mini@openai', 'messages': [{'role': 'user', 'content': 'hi'}]}
        for _ in range(3):  # expires before the attempt: refused with zero hits
            reply = S.call(app, 'POST', '/v1/chat/completions', json_body=body, principal=hasty,
                           headers={'X-Titanos-Deadline-Ms': '1'})
            self.assertEqual((reply.status, reply.json()['error']['code']), (504, 'deadline_exceeded'))
        self.assertEqual(len(self.openai.hits), 0)
        self.openai.faults.push('slow_first_byte', delay=0.6, times=3)
        for _ in range(3):  # expires while the upstream is still silent (or still connecting)
            reply = S.call(app, 'POST', '/v1/chat/completions', json_body=body, principal=hasty,
                           headers={'X-Titanos-Deadline-Ms': '150'})
            self.assertEqual((reply.status, reply.json()['error']['code']), (504, 'deadline_exceeded'))
        self.assertEqual(broker.breakers.get('openai', 'oa').state, 'closed')
        self.openai.faults.clear()
        other = S.call(app, 'POST', '/v1/chat/completions', json_body=body,
                       principal=S.key('inference', 'cloud', id='kpatient001'))
        self.assertEqual((other.status, other.header('x-titanos-served-by')), (200, 'cloud:openai'))


class Monitor(unittest.TestCase):
    def make(self, *, probes_on=True):
        clock = Clock()
        board = BreakerBoard(S.settings(broker_breaker_failures=1, broker_probe_idle_cutoff=900.0,
                                        broker_health_ttl=300.0), clock=clock, rng=random.Random(5))
        probed = []

        async def probe(upstream_id):
            probed.append(upstream_id)
            return ProbeResult(ok=True, status=200, latency_ms=1.0)
        switches = switches_for(S.settings()) if probes_on else switches_for(S.settings(role='managed'))
        monitor = HealthMonitor(board, switches=switches, settings=S.settings(broker_probe_idle_cutoff=900.0,
                                                                              broker_health_ttl=300.0),
                                upstreams=lambda: [('recent', 'c1'), ('idle', 'c2'), ('half', 'c3'), ('never', None)],
                                probe=probe, clock=clock)
        return clock, board, monitor, probed

    def test_lazy_probes_only_for_recent_or_half_open_upstreams(self):
        clock, board, monitor, probed = self.make()
        board.get('idle', 'c2').touch()
        clock.advance(2000)  # idle is now past the cutoff
        board.get('recent', 'c1').touch()
        half = board.get('half', 'c3')
        half.record('server_error')
        clock.advance(half.last_duration + 1)
        self.assertEqual(half.state, 'half_open')
        done = asyncio.run(monitor.tick())
        self.assertEqual(sorted(done), ['half', 'recent'])
        self.assertEqual(half.state, 'closed')  # the probe succeeded
        self.assertEqual(asyncio.run(monitor.tick()), [])  # within the TTL nothing is probed again
        clock.advance(301)
        self.assertEqual(sorted(asyncio.run(monitor.tick())), ['half', 'recent'])
        clock.advance(10_000)  # everything idle: an idle Titanos makes no outbound connections
        self.assertEqual(asyncio.run(monitor.tick()), [])
        self.assertNotIn('never', probed)
        self.assertEqual(monitor.status('recent')['breaker'], 'closed')

    def test_no_probes_when_health_probes_are_off(self):
        clock, board, monitor, probed = self.make(probes_on=False)
        board.get('recent', 'c1').touch()
        half = board.get('half', 'c3')
        half.record('server_error')
        clock.advance(1000)
        self.assertEqual(asyncio.run(monitor.tick()), [])
        self.assertEqual(probed, [])

        async def start_and_stop():
            monitor.start()
            self.assertIsNone(monitor._task)
            await monitor.stop()
        asyncio.run(start_and_stop())

    def test_failed_probe_reopens(self):
        clock, board, monitor, probed = self.make()

        async def failing(upstream_id):
            return ProbeResult(ok=False, status=503, latency_ms=1.0, code='upstream_unavailable')
        monitor._probe = failing
        half = board.get('half', 'c3')
        half.record('server_error')
        clock.advance(half.last_duration + 1)
        asyncio.run(monitor.tick())
        self.assertEqual((half.state, half.opens), ('open', 2))


if __name__ == '__main__':
    unittest.main(verbosity=1)
