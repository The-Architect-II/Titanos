"""Broker routing (WP6): BrokerRouter, policy, GrantIssuer and explain, with every upstream configured, enabled and
credentialed (loopback mocks, synthetic sentinels, netguard installed; zero non-loopback attempts).

- unknown, typo'd and blank names, disabled and unvalidated upstreams: 404/400 and ZERO hits on every mock;
- cloud_policy x ceiling matrix; location matrix; missing cloud scope is 404 even with policy off (no 403 oracle);
  cloud:<other> denied; constraint globs; X-Titanos-Target needs route:explicit;
- alias collisions (load, registry candidates); HiveMind compat suffixes (longest first);
- virtual-model strategies (ordered, weighted, cheapest, lowest_latency) and max_attempts defaults (non-chat = 1);
- explain makes zero egress; data_policy denial; store forced false for Responses and chat completions;
- previous_response_id of another key: 404 with zero hits; the hop rules (local only, own hop 508);
- GrantIssuer refuses every purpose in managed role without delegation, is the only EgressGrant constructor, and
  EgressClient re-checks the purpose.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import unittest
from pathlib import Path

import broker_wp6_support as S
from broker_mocks import (AnthropicMock, ElevenLabsMock, GeminiMock, OllamaCloudMock, OpenAIMock, RouterUpstreamMock,
                          netguard)

from titanos.broker import inbound
from titanos.broker.config import BrokerConfig, check_name_conflicts
from titanos.broker.dispatch import BrokerDispatch
from titanos.broker.errors import BrokerError
from titanos.broker.facade import Broker
from titanos.broker.role import parse_parent_headers, switches_for
from titanos.broker.router import GrantIssuer
from titanos.broker.surfaces import SURFACES
from titanos.broker.types import GRANT_PURPOSES, RouteContext

ROOT = Path(__file__).resolve().parents[1]
MOCKS: dict = {}
CLOUD = S.key('inference', 'cloud', id='kcloud00001')
ADMIN_KEY = S.key('admin', id='kadmin00001')
CHAT = [{'role': 'user', 'content': 'Say hello.'}]


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['anthropic'] = AnthropicMock()
    MOCKS['gemini'] = GeminiMock()
    MOCKS['ollama'] = OllamaCloudMock()
    MOCKS['xi'] = ElevenLabsMock()
    MOCKS['hive'] = RouterUpstreamMock('hivemind')
    MOCKS['pair'] = RouterUpstreamMock('pair')
    for mock in MOCKS.values():
        mock.start()


def tearDownModule():
    for mock in MOCKS.values():
        mock.stop()


def document(**changes) -> dict:
    m = MOCKS
    doc = {
        'policy': {'cloud_policy': 'fallback'},
        'upstreams': {
            'openai': S.upstream(m['openai'], 'openai', credential='openai-cred', models=['gpt-4o-mini', 'gpt-4.1-mini'],
                                 data_policy={'modalities': ['text'], 'allow_store': True}),
            'anthropic': S.upstream(m['anthropic'], 'anthropic', credential='ant-cred', models=['claude-sonnet-4-5']),
            'gemini': S.upstream(m['gemini'], 'gemini', credential='gem-cred', models=['gemini-2.5-flash']),
            'ollama_cloud': S.upstream(m['ollama'], 'ollama_cloud', credential='olc-cred', models=['gpt-oss:120b']),
            'xi': S.upstream(m['xi'], 'elevenlabs', credential='xi-cred', models=['eleven_multilingual_v2']),
            'hive': S.upstream(m['hive'], 'hivemind_gateway', credential='hive-cred'),
            'offline': S.upstream(m['openai'], 'openai', credential='offline-cred', enabled=False, models=['gpt-4o-mini']),
            'unval': S.upstream(m['openai'], 'openai', credential='unval-cred', models=['gpt-4o-mini']),
            'slot': {'profile': 'pending_contract', 'enabled': True},
        },
        'models': {'mini': {'target': 'gpt-4o-mini@openai'}, 'claude': {'target': 'claude-sonnet-4-5@anthropic'}},
        'virtual_models': {
            'chat-auto': {'capability': 'chat', 'targets': [{'ref': 'qwen-local'}, {'ref': 'gpt-4o-mini@openai'}]},
            'cloud-first': {'capability': 'chat', 'targets': [{'ref': 'gpt-4o-mini@openai'}, {'ref': 'qwen-local'},
                                                              {'ref': 'claude-sonnet-4-5@anthropic'}]},
            'speech-vm': {'capability': 'speech', 'targets': [{'ref': 'tts-local'},
                                                              {'ref': 'eleven_multilingual_v2@xi'}]},
            'weighted': {'capability': 'chat', 'strategy': 'weighted',
                         'targets': [{'ref': 'gpt-4o-mini@openai', 'weight': 3},
                                     {'ref': 'claude-sonnet-4-5@anthropic', 'weight': 1}]},
            'cheap': {'capability': 'chat', 'strategy': 'cheapest',
                      'targets': [{'ref': 'gpt-4.1-mini@openai'}, {'ref': 'claude-sonnet-4-5@anthropic'},
                                  {'ref': 'gemini-2.5-flash@gemini'}]},
            'fast': {'capability': 'chat', 'strategy': 'lowest_latency',
                     'targets': [{'ref': 'gpt-4o-mini@openai'}, {'ref': 'claude-sonnet-4-5@anthropic'}]},
        },
        'compat': {'hivemind_suffix_aliases': {'_ollama_cloud': 'ollama_cloud', '_cloud': 'openai'}},
        'prices': [{'match': 'gpt-4.1-mini@openai', 'input_per_mtok': '2.00', 'output_per_mtok': '8.00'},
                   {'match': 'gemini-2.5-flash@gemini', 'input_per_mtok': '0.10', 'output_per_mtok': '0.40'}],
    }
    for key, value in changes.items():
        doc[key] = value
    return doc


LOCAL = {'qwen-local': ('chat', 'llama-cpp'), 'llama-b': 'chat', 'tts-local': 'speech'}


def ctx(broker, model, principal, *, surface='openai.chat', headers=None, body=None):
    row = SURFACES[surface]
    hints = parse_parent_headers(headers or {}, broker.settings, principal)
    return RouteContext(row, model, principal, hints, inbound.summarize(row, body or {}, None, 64))


def resolve(broker, model, principal=CLOUD, **kw):
    return broker.snapshot().router.resolve(ctx(broker, model, principal, **kw))


def chain(plan) -> list[str]:
    return [t.ref for t in plan.chain]


class Base(unittest.TestCase):
    unvalidated = ('unval',)

    def setUp(self):
        netguard.reset_attempts()
        for mock in MOCKS.values():
            mock.reset()
        self.lab = S.Lab(document(), local=LOCAL, unvalidated=self.unvalidated)
        self.broker = self.lab.broker
        self.local = S.FakeLocal()
        self.app = BrokerDispatch(self.local, self.broker)

    def tearDown(self):
        self.lab.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def hits(self) -> int:
        return S.total_hits(*MOCKS.values())

    def code(self, expected: str, fn, *args, **kw):
        with self.assertRaises(BrokerError) as caught:
            fn(*args, **kw)
        self.assertEqual(caught.exception.code, expected, str(caught.exception))
        return caught.exception

    def post(self, model, principal=CLOUD, path='/v1/chat/completions', **body):
        payload = {'model': model, 'messages': CHAT}
        payload.update(body)
        return S.call(self.app, 'POST', path, json_body=payload, principal=principal)


class LeakGuard(Base):
    def test_unknown_typo_and_blank_names_make_zero_hits(self):
        for name in ('gpt-4o-mini-typo', 'mini2', 'gpt-4o-mini@nope', 'not-listed@openai', 'gpt4o@openai',
                     'claude-sonnet-4-5@openai', 'models/gpt-4o-mini'):
            with self.subTest(name=name):
                reply = self.post(name)
                self.assertEqual((reply.status, reply.json()['error']['code']), (404, 'model_not_found'))
        for blank in ('', '   '):
            reply = self.post(blank)
            self.assertEqual((reply.status, reply.json()['error']['code']), (400, 'model_required'))
        reply = self.post('x' * 300)
        self.assertEqual((reply.status, reply.json()['error']['code']), (400, 'invalid_request'))
        missing = S.call(self.app, 'POST', '/v1/chat/completions', json_body={'messages': CHAT}, principal=CLOUD)
        self.assertEqual(missing.status, 400)
        self.assertEqual(self.hits(), 0)
        self.assertEqual(self.local.calls, [])

    def test_disabled_unvalidated_and_slot_upstreams_make_zero_hits(self):
        for name in ('gpt-4o-mini@offline', 'gpt-4o-mini@unval'):
            with self.subTest(name=name):
                reply = self.post(name)
                self.assertEqual((reply.status, reply.json()['error']['code']), (404, 'model_not_found'))
        self.code('model_not_found', resolve, self.broker, 'anything@slot')
        decision = self.broker.decisions.query(principal=S.BOOTSTRAP, limit=2)
        excluded = {c['excluded'] for d in decision for c in d['candidates']}
        self.assertTrue({'disabled', 'unvalidated'} <= excluded, excluded)
        self.assertEqual(self.hits(), 0)


class PolicyMatrix(Base):
    def test_cloud_policy_times_ceiling(self):
        order = ('off', 'explicit', 'fallback')
        for ceiling in order:
            for configured in order:
                with self.subTest(ceiling=ceiling, cloud_policy=configured):
                    self.lab.rewrite(document(policy={'cloud_policy': configured}))
                    broker = Broker(settings=S.settings(broker_cloud_ceiling=ceiling), config=self.lab.local,
                                    home=self.lab.home, vault=self.lab.vault)
                    effective = order[min(order.index(ceiling), order.index(configured))]
                    if effective == 'off':
                        self.code('cloud_disabled', resolve, broker, 'gpt-4o-mini@openai')
                        self.code('cloud_disabled', resolve, broker, 'mini')
                        self.assertEqual(chain(resolve(broker, 'chat-auto')), ['qwen-local'])
                    else:
                        self.assertEqual(chain(resolve(broker, 'gpt-4o-mini@openai')), ['gpt-4o-mini@openai'])
                        self.assertEqual(chain(resolve(broker, 'mini')), ['gpt-4o-mini@openai'])
                    if effective == 'explicit':  # a virtual model keeps only its first target's class
                        self.assertEqual(chain(resolve(broker, 'chat-auto')), ['qwen-local'])
                        self.assertEqual(chain(resolve(broker, 'cloud-first')),
                                         ['gpt-4o-mini@openai', 'claude-sonnet-4-5@anthropic'])
                    if effective == 'fallback':
                        self.assertEqual(chain(resolve(broker, 'chat-auto')), ['qwen-local', 'gpt-4o-mini@openai'])
        self.assertEqual(self.hits(), 0)

    def test_location_matrix(self):
        b = self.broker
        loc = lambda value: {'X-Titanos-Location': value}  # noqa: E731
        self.assertEqual(chain(resolve(b, 'cloud-first', headers=loc('local_only'))), ['qwen-local'])
        self.assertEqual(chain(resolve(b, 'chat-auto', headers=loc('cloud_only'))), ['gpt-4o-mini@openai'])
        self.assertEqual(chain(resolve(b, 'cloud-first', headers=loc('prefer_local'))),
                         ['qwen-local', 'gpt-4o-mini@openai', 'claude-sonnet-4-5@anthropic'])
        self.assertEqual(chain(resolve(b, 'cloud-first', headers=loc('auto'))),
                         ['gpt-4o-mini@openai', 'qwen-local', 'claude-sonnet-4-5@anthropic'])
        self.code('location_unsatisfiable', resolve, b, 'gpt-4o-mini@openai', headers=loc('local_only'))
        self.code('location_unsatisfiable', resolve, b, 'qwen-local', headers=loc('cloud_only'))
        self.code('invalid_request', resolve, b, 'qwen-local', headers=loc('somewhere'))
        # the key's own default and its allowed locations
        local_key = S.key('inference', 'cloud', default_location='local_only')
        self.assertEqual(chain(resolve(b, 'cloud-first', local_key)), ['qwen-local'])
        narrow = S.key('inference', 'cloud', locations=('local_only',))
        self.code('location_unsatisfiable', resolve, b, 'chat-auto', narrow, headers=loc('auto'))
        # broker.json default_location; prefer_local outside fallback keeps local targets only
        self.lab.rewrite(document(policy={'cloud_policy': 'explicit', 'default_location': 'prefer_local'}))
        self.broker.reload()
        self.assertEqual(chain(resolve(b, 'cloud-first')), ['qwen-local'])
        self.assertEqual(chain(resolve(b, 'gpt-4o-mini@openai')), ['gpt-4o-mini@openai'])
        # managed placement local_only is authoritative
        managed = Broker(settings=S.settings(role='managed', managed_delegation='parent'), config=self.lab.local,
                         home=self.lab.home, vault=self.lab.vault)
        self.code('location_unsatisfiable', resolve, managed, 'gpt-4o-mini@openai', S.key('inference', 'cloud', 'delegate'),
                  headers={'X-Hivemind-Placement': 'local_only', 'X-Titanos-Delegate': 'cloud',
                           'X-Titanos-Location': 'cloud_only'})
        self.assertEqual(self.hits(), 0)


class Visibility(Base):
    def test_missing_cloud_scope_is_404_even_with_policy_off(self):
        plain = S.key('inference')
        for configured in ('fallback', 'off'):
            with self.subTest(cloud_policy=configured):
                self.lab.rewrite(document(policy={'cloud_policy': configured}))
                self.broker.reload()
                for name in ('gpt-4o-mini@openai', 'mini', 'claude'):
                    reply = self.post(name, plain)
                    self.assertEqual((reply.status, reply.json()['error']['code']), (404, 'model_not_found'))
        visible = self.post('gpt-4o-mini@openai', CLOUD)
        self.assertEqual((visible.status, visible.json()['error']['code']), (403, 'cloud_disabled'))
        self.assertEqual(self.hits(), 0)

    def test_cloud_other_upstream_is_denied(self):
        only_ant = S.key('inference', 'cloud:anthropic')
        self.code('model_not_found', resolve, self.broker, 'gpt-4o-mini@openai', only_ant)
        self.assertEqual(chain(resolve(self.broker, 'claude', only_ant)), ['claude-sonnet-4-5@anthropic'])
        store_only = S.key('inference', 'cloud:store')
        self.code('model_not_found', resolve, self.broker, 'gpt-4o-mini@openai', store_only)
        self.assertEqual(self.hits(), 0)

    def test_constraint_globs(self):
        globbed = S.key('inference', 'cloud', models=('gpt-*@openai', 'qwen-*'))
        self.assertEqual(chain(resolve(self.broker, 'gpt-4.1-mini@openai', globbed)), ['gpt-4.1-mini@openai'])
        self.code('model_not_found', resolve, self.broker, 'claude', globbed)
        self.assertEqual(chain(resolve(self.broker, 'chat-auto', globbed)), ['qwen-local', 'gpt-4o-mini@openai'])
        upstreams = S.key('inference', 'cloud', upstreams=('anthropic',))
        self.code('model_not_found', resolve, self.broker, 'mini', upstreams)
        local_only = S.key('inference', 'cloud', classes=('local',))
        self.assertEqual(chain(resolve(self.broker, 'chat-auto', local_only)), ['qwen-local'])
        self.code('model_not_found', resolve, self.broker, 'mini', local_only)
        router_class = S.key('inference', 'cloud', classes=('router',))
        self.assertEqual(chain(resolve(self.broker, 'anything@hive', router_class)), ['anything@hive'])
        self.assertEqual(self.hits(), 0)

    def test_target_header_needs_route_explicit(self):
        header = {'X-Titanos-Target': 'claude-sonnet-4-5@anthropic'}
        reply = S.call(self.app, 'POST', '/v1/chat/completions', json_body={'model': 'qwen-local', 'messages': CHAT},
                       principal=CLOUD, headers=header)
        self.assertEqual((reply.status, reply.json()['error']['code']), (403, 'scope_denied'))
        self.assertEqual(self.hits(), 0)
        explicit = S.key('inference', 'cloud', 'route:explicit')
        self.assertEqual(chain(resolve(self.broker, 'qwen-local', explicit, headers=header)),
                         ['claude-sonnet-4-5@anthropic'])
        # still subject to policy and visibility
        self.code('model_not_found', resolve, self.broker, 'qwen-local', S.key('inference', 'route:explicit'),
                  headers=header)

    def test_target_header_naming_a_virtual_model_never_widens_explicit_policy(self):
        """cloud_policy=explicit: a virtual model keeps only its first target's class, and X-Titanos-Target naming
        the same virtual model is 400 (one target only), so the header cannot add cross-class fallback."""
        self.lab.rewrite(document(policy={'cloud_policy': 'explicit'}))
        broker = Broker(settings=S.settings(), config=self.lab.local, home=self.lab.home, vault=self.lab.vault)
        explicit = S.key('inference', 'cloud', 'route:explicit')
        self.assertEqual(chain(resolve(broker, 'chat-auto', explicit)), ['qwen-local'])
        error = self.code('invalid_request', resolve, broker, 'qwen-local', explicit,
                          headers={'X-Titanos-Target': 'chat-auto'})
        self.assertEqual(error.status, 400)
        local = S.FakeLocal()
        local.errors['qwen-local'] = (429, 'rate_limit_error')
        reply = S.call(BrokerDispatch(local, broker), 'POST', '/v1/chat/completions', principal=explicit,
                       headers={'X-Titanos-Target': 'chat-auto'}, json_body={'model': 'qwen-local', 'messages': CHAT})
        self.assertEqual((reply.status, reply.json()['error']['code']), (400, 'invalid_request'))
        self.assertEqual((local.calls, self.hits()), ([], 0))


class Names(Base):
    def test_alias_collisions(self):
        cfg = BrokerConfig.model_validate(document())
        with self.assertRaises(BrokerError) as caught:
            check_name_conflicts(cfg, ['mini', 'qwen-local'])
        self.assertEqual((caught.exception.code, caught.exception.status), ('broker_name_conflict', 409))
        from types import SimpleNamespace
        candidate = SimpleNamespace(models={'chat-auto': SimpleNamespace(task='chat', backend='transformers')})
        self.code('broker_name_conflict', self.broker.on_registry_candidate, candidate)
        self.broker.on_registry_candidate(SimpleNamespace(models={'fresh': SimpleNamespace(task='chat', backend='x')}))
        with self.assertRaises(ValueError):
            BrokerConfig.model_validate(dict(document(), virtual_models={'mini': {'capability': 'chat',
                                                                                  'targets': [{'ref': 'qwen-local'}]}}))
        # a broker.json that collides with the local registry leaves the broker disabled (and reload refuses it)
        clash = Broker(settings=self.lab.settings, config=S.local_config({'mini': 'chat'}), home=self.lab.home,
                       vault=self.lab.vault)
        with self.assertLogs('titanos.broker', 'ERROR'):
            self.assertEqual(clash.snapshot().load_error, 'broker_name_conflict')
        self.code('broker_name_conflict', clash.reload)

    def test_compat_suffix_longest_first(self):
        self.assertEqual(chain(resolve(self.broker, 'gpt-oss:120b_ollama_cloud')), ['gpt-oss:120b@ollama_cloud'])
        self.assertEqual(chain(resolve(self.broker, 'gpt-4o-mini_cloud')), ['gpt-4o-mini@openai'])
        self.code('model_not_found', resolve, self.broker, 'unlisted_cloud')
        self.assertEqual(self.hits(), 0)


class Strategies(Base):
    def test_ordered_weighted_cheapest_lowest_latency(self):
        b = self.broker
        self.assertEqual(chain(resolve(b, 'cloud-first')),
                         ['gpt-4o-mini@openai', 'qwen-local', 'claude-sonnet-4-5@anthropic'])
        firsts = [resolve(b, 'weighted').chain[0].upstream_id for _ in range(8)]
        self.assertEqual(firsts, ['openai', 'openai', 'anthropic', 'openai'] * 2)  # smooth weighted round-robin 3:1
        self.assertEqual(chain(resolve(b, 'cheap')),  # priced ascending, unpriced last
                         ['gemini-2.5-flash@gemini', 'gpt-4.1-mini@openai', 'claude-sonnet-4-5@anthropic'])
        self.broker.latency.observe('claude-sonnet-4-5@anthropic', 80.0)
        self.broker.latency.observe('gpt-4o-mini@openai', 400.0)
        self.assertEqual(chain(resolve(b, 'fast')), ['claude-sonnet-4-5@anthropic', 'gpt-4o-mini@openai'])
        self.assertEqual(self.hits(), 0)

    def test_max_attempts_defaults(self):
        b = self.broker
        self.assertEqual(resolve(b, 'cloud-first').max_attempts, 3)  # chat: 3, capped at 1 + max_fallbacks (2)
        speech = resolve(b, 'speech-vm', surface='openai.audio.speech', body={'input': 'hi'})
        self.assertEqual((speech.max_attempts, chain(speech)), (1, ['tts-local']))  # non-chat capabilities: 1
        self.assertEqual(resolve(b, 'mini').max_attempts, 1)  # a single target
        self.lab.rewrite(document(policy={'cloud_policy': 'fallback', 'max_fallbacks': 0}))
        b.reload()
        self.assertEqual(resolve(b, 'cloud-first').max_attempts, 1)
        capped = Broker(settings=S.settings(broker_max_fallbacks=1), config=self.lab.local, home=self.lab.home,
                        vault=self.lab.vault)
        self.lab.rewrite(document())
        self.assertEqual(resolve(capped, 'cloud-first').max_attempts, 2)


class ExplainAndDataPolicy(Base):
    def test_explain_makes_zero_egress(self):
        decision = asyncio.run(self.broker.explain('openai.chat', {'model': 'cloud-first', 'messages': CHAT},
                                                   {'X-Titanos-Location': 'auto'}, CLOUD))
        self.assertTrue(decision['dry_run'])
        self.assertIsNone(decision['served'])
        self.assertEqual(decision['chain'], ['gpt-4o-mini@openai', 'qwen-local', 'claude-sonnet-4-5@anthropic'])
        refused = asyncio.run(self.broker.explain('openai.chat', {'model': 'nope', 'messages': CHAT}, {}, CLOUD))
        self.assertIn('refused:model_not_found', refused['notes'])
        self.assertEqual(self.hits(), 0)
        self.assertEqual(self.broker.snapshot().issuer.issued, 0)

    def test_data_policy_denial(self):
        image = [{'role': 'user', 'content': [{'type': 'text', 'text': 'what is this?'},
                                              {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}]
        reply = S.call(self.app, 'POST', '/v1/chat/completions', json_body={'model': 'mini', 'messages': image},
                       principal=CLOUD)
        self.assertEqual((reply.status, reply.json()['error']['code']), (403, 'data_policy_denied'))
        self.assertEqual(self.hits(), 0)
        self.assertEqual(chain(resolve(self.broker, 'claude', body={'messages': image})), ['claude-sonnet-4-5@anthropic'])

    def test_store_forced_false_for_responses_and_chat(self):
        openai = MOCKS['openai']
        reply = S.call(self.app, 'POST', '/v1/responses', json_body={'model': 'mini', 'input': 'hi', 'store': True},
                       principal=CLOUD)
        self.assertEqual(reply.status, 200, reply.body[:300])
        self.assertIs(openai.hits[-1].json()['store'], False)
        reply = S.call(self.app, 'POST', '/v1/responses', json_body={'model': 'mini', 'input': 'hi'}, principal=CLOUD)
        self.assertIs(openai.hits[-1].json()['store'], False)
        reply = self.post('mini', store=True)
        self.assertEqual(reply.status, 200)
        self.assertIs(openai.hits[-1].json()['store'], False)
        notes = self.broker.decisions.query(principal=S.BOOTSTRAP, limit=1)[0]['notes']
        self.assertIn('forced:store=false', notes)
        # cloud:store plus the upstream's allow_store keeps it, and binds the handle to the caller
        storer = S.key('inference', 'cloud', 'cloud:store')
        reply = S.call(self.app, 'POST', '/v1/responses', json_body={'model': 'mini', 'input': 'hi', 'store': True},
                       principal=storer)
        self.assertIs(openai.hits[-1].json()['store'], True)
        response_id = reply.json()['id']
        self.assertEqual(self.broker.handles.owner('responses', response_id).key_id, storer.key_id)


class Handles(Base):
    def test_previous_response_id_of_another_key_is_404_with_zero_hits(self):
        owner, other = S.key('inference', 'cloud', 'cloud:store'), S.key('inference', 'cloud', 'cloud:store')
        self.broker.handles.bind('responses', 'resp_owned', 'openai', 'openai-cred', owner.key_id)
        body = {'model': 'mini', 'input': 'next', 'previous_response_id': 'resp_owned'}
        reply = S.call(self.app, 'POST', '/v1/responses', json_body=body, principal=other)
        self.assertEqual((reply.status, reply.json()['error']['code']), (404, 'response_not_found'))
        unknown = S.call(self.app, 'POST', '/v1/responses', json_body=dict(body, previous_response_id='resp_unknown'),
                         principal=owner)
        self.assertEqual((unknown.status, unknown.json()['error']['code']), (404, 'response_not_found'))
        for path in ('/v1/responses/resp_owned', '/v1/responses/resp_owned/input_items'):
            reply = S.call(self.app, 'GET', path, principal=other)
            self.assertEqual((reply.status, reply.json()['error']['code']), (404, 'response_not_found'))
        self.assertEqual(self.hits(), 0)
        # the owner reaches the owning upstream (the mock knows no such response: its own 400 passes through)
        S.call(self.app, 'POST', '/v1/responses', json_body=body, principal=owner)
        self.assertEqual(len(MOCKS['openai'].hits), 1)
        self.assertEqual(MOCKS['openai'].hits[0].json()['previous_response_id'], 'resp_owned')


class HandleTable(unittest.TestCase):
    def test_ttl_cap_persistence_and_ownership(self):
        import json
        import shutil
        import tempfile
        from titanos.broker.handles import HandleStore
        home = Path(tempfile.mkdtemp(prefix='titanos-wp6-handles-'))
        self.addCleanup(shutil.rmtree, home, True)
        now = [1000.0]
        path = home / 'broker' / 'handles.json'
        store = HandleStore(path, ttl_seconds=100, max_entries=3, clock=lambda: now[0])
        self.assertFalse(path.exists())  # nothing is written before the first binding
        for index in range(4):
            now[0] += 1
            store.bind('responses', f'resp_{index}', 'openai', 'oa', 'key1')
        self.assertIsNone(store.owner('responses', 'resp_0'))  # the oldest went first at the cap
        self.assertEqual(len(store), 3)
        document = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(document['version'], 2)  # R17: versioned layout, the binding records the served model
        document = document['handles']
        self.assertEqual(set(document), {'responses:resp_1', 'responses:resp_2', 'responses:resp_3'})
        self.assertEqual(set(document['responses:resp_1']), {'upstream_id', 'credential_id', 'key_id', 'created_at',
                                                             'expires_at', 'model'})
        again = HandleStore(path, ttl_seconds=100, clock=lambda: now[0])
        self.assertEqual(again.owner('responses', 'resp_3').upstream_id, 'openai')
        self.assertEqual(again.owned(S.key('inference', id='key1'), 'responses', 'resp_3').key_id, 'key1')
        with self.assertRaises(BrokerError) as caught:
            again.owned(S.key('inference', id='key2'), 'responses', 'resp_3')
        self.assertEqual(caught.exception.code, 'response_not_found')
        self.assertEqual(again.owned(S.BOOTSTRAP, 'responses', 'resp_3').key_id, 'key1')
        now[0] += 200
        self.assertIsNone(again.owner('responses', 'resp_3'))  # expired
        again.forget('responses', 'resp_2')
        with self.assertRaises(ValueError):
            again.bind('files', 'x', 'openai', None, None)


class Hops(Base):
    def test_hop_restricts_to_local_and_own_hop_is_508(self):
        ours = self.broker.instance_hop()
        reply = S.call(self.app, 'POST', '/v1/chat/completions', json_body={'model': 'qwen-local', 'messages': CHAT},
                       principal=CLOUD, headers={'X-Titanos-Hop': ours})
        self.assertEqual((reply.status, reply.json()['error']['code']), (508, 'loop_detected'))
        self.assertEqual(reply.header('x-should-retry'), 'false')
        foreign = {'X-Titanos-Hop': 'abcdef0123456789'}
        self.code('location_unsatisfiable', resolve, self.broker, 'mini', headers=foreign)
        self.assertEqual(chain(resolve(self.broker, 'cloud-first', headers=foreign)), ['qwen-local'])
        self.assertEqual(chain(resolve(self.broker, 'chat-auto', headers={'X-HiveMind-Peer-Hop': 'peer'})),
                         ['qwen-local'])
        self.assertEqual(self.hits(), 0)
        # egress to self-hosted routers carries our hop value
        reply = self.post('anything@hive')
        self.assertEqual(reply.status, 200)
        self.assertEqual(MOCKS['hive'].hops, [ours])
        self.assertEqual(reply.header('x-titanos-served-by'), 'router:hive')


class PairFacade(Base):
    """Router-as-upstream through the PAIR facade profile (pair_facade, a keyless loopback facade): a qualified ref and a
    published alias are served by router:<id>, egress carries this instance's hop header and no credential, streams pass
    through, and a request arriving with the facade's own hop back is 508 with zero hits."""

    def setUp(self):
        super().setUp()
        self.lab.close()
        doc = document()
        doc['upstreams']['pair'] = S.upstream(MOCKS['pair'], 'pair_facade', credential=None, models=['pair-chat'])
        doc['models']['pair-alias'] = {'target': 'pair-chat@pair'}
        self.lab = S.Lab(doc, local=LOCAL, unvalidated=self.unvalidated)
        self.broker = self.lab.broker
        self.app = BrokerDispatch(self.local, self.broker)

    def test_pair_facade_serves_as_a_router_upstream(self):
        ours = self.broker.instance_hop()
        pair = MOCKS['pair']
        for model in ('pair-chat@pair', 'pair-alias'):
            with self.subTest(model=model):
                reply = self.post(model)
                self.assertEqual(reply.status, 200, reply.body[:300])
                self.assertEqual(reply.header('x-titanos-served-by'), 'router:pair')
                self.assertEqual(reply.header('x-titanos-translation'), 'passthrough')
                hit = pair.hits[-1]
                self.assertEqual((hit.path, hit.json()['model']), ('/v1/chat/completions', 'pair-chat'))
                self.assertIsNone(hit.credential)  # keyless facade: no credential is presented
        self.assertEqual(pair.hops, [ours, ours])
        stream = self.post('pair-chat@pair', stream=True)
        self.assertEqual(stream.status, 200)
        self.assertTrue(stream.body.rstrip().endswith(b'data: [DONE]'))
        self.assertNotIn('stream_options', pair.hits[-1].json())  # the row declares no include_usage
        embed = S.call(self.app, 'POST', '/v1/embeddings', json_body={'model': 'pair-chat@pair', 'input': 'x'},
                       principal=CLOUD)
        self.assertEqual((embed.status, pair.hits[-1].path), (200, '/v1/embeddings'))
        before = self.hits()
        looped = S.call(self.app, 'POST', '/v1/chat/completions', principal=CLOUD,
                        json_body={'model': 'pair-chat@pair', 'messages': CHAT}, headers={'X-Titanos-Hop': ours})
        self.assertEqual((looped.status, looped.json()['error']['code']), (508, 'loop_detected'))
        self.assertEqual(self.hits(), before)


class Managed(Base):
    def managed(self, **kw):
        return Broker(settings=S.settings(role='managed', **kw), config=self.lab.local, home=self.lab.home,
                      vault=self.lab.vault)

    def test_delegation_needs_all_conditions_and_an_explicit_target(self):
        parent = S.key('inference', 'cloud', 'delegate', kind='parent')
        delegate = {'X-Titanos-Delegate': 'cloud'}
        b = self.managed(managed_delegation='parent')
        self.assertEqual(chain(resolve(b, 'gpt-4o-mini@openai', parent, headers=delegate)), ['gpt-4o-mini@openai'])
        self.code('delegation_required', resolve, b, 'gpt-4o-mini@openai', parent)
        self.assertEqual(chain(resolve(b, 'chat-auto', parent, headers=delegate)), ['qwen-local'])  # never fallback
        self.code('delegation_required', resolve, b, 'gpt-4o-mini@openai', S.key('inference', 'cloud'),
                  headers=delegate)  # no delegate scope
        self.code('role_forbids', resolve, self.managed(), 'gpt-4o-mini@openai', parent, headers=delegate)
        keyless = self.managed(managed_delegation='parent')
        self.code('model_not_found', resolve, keyless, 'gpt-4o-mini@openai', keyless.default_principal(),
                  headers=delegate)
        self.assertEqual(self.hits(), 0)

    def test_delegated_target_must_be_a_qualified_ref_or_one_x_titanos_target(self):
        """Roles condition 5: aliases and compat suffixes do not count as explicit for delegated egress, and
        X-Titanos-Target names ONE target (a virtual model there is 400, never a chain with fallback)."""
        parent = S.key('inference', 'cloud', 'delegate', 'route:explicit', kind='parent')
        delegate = {'X-Titanos-Delegate': 'cloud'}
        b = self.managed(managed_delegation='parent')
        self.code('delegation_required', resolve, b, 'mini', parent, headers=delegate)
        self.code('delegation_required', resolve, b, 'gpt-4o-mini_cloud', parent, headers=delegate)
        plan = resolve(b, 'qwen-local', parent, headers=dict(delegate, **{'X-Titanos-Target': 'mini'}))
        self.assertEqual((chain(plan), plan.fallback_on), (['gpt-4o-mini@openai'], frozenset()))
        for name in ('chat-auto', 'cloud-first'):
            error = self.code('invalid_request', resolve, b, 'qwen-local', parent,
                              headers=dict(delegate, **{'X-Titanos-Target': name}))
            self.assertEqual(error.status, 400)
        self.assertEqual(self.hits(), 0)

    def test_delegation_never_raises_the_configured_cloud_policy(self):
        parent = S.key('inference', 'cloud', 'delegate', kind='parent')
        delegate = {'X-Titanos-Delegate': 'cloud'}
        self.lab.rewrite(document(policy={'cloud_policy': 'off'}))
        b = self.managed(managed_delegation='parent')
        self.assertEqual(b.effective_cloud_policy(), 'off')
        self.code('cloud_disabled', resolve, b, 'gpt-4o-mini@openai', parent, headers=delegate)
        app = BrokerDispatch(S.FakeLocal(), b)
        reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent, headers=delegate,
                       json_body={'model': 'gpt-4o-mini@openai', 'messages': CHAT})
        self.assertEqual((reply.status, reply.json()['error']['code']), (403, 'cloud_disabled'))
        self.lab.rewrite(document())
        self.assertEqual(self.managed(managed_delegation='parent').effective_cloud_policy(), 'explicit')
        ceiling = Broker(settings=S.settings(role='managed', managed_delegation='parent', broker_cloud_ceiling='off'),
                         config=self.lab.local, home=self.lab.home, vault=self.lab.vault)
        self.code('cloud_disabled', resolve, ceiling, 'gpt-4o-mini@openai', parent, headers=delegate)
        self.assertEqual(self.hits(), 0)

    def test_managed_requests_through_dispatch(self):
        """End to end through BrokerDispatch: no egress without delegation; a delegated qualified ref is served by the
        cloud with one hit; a local typed failure is returned, never answered by the cloud; the parent deadline gives
        504 deadline_exceeded on a local target; the parent trace is echoed as X-Request-ID."""
        parent = S.key('inference', 'cloud', 'delegate', 'route:explicit', kind='parent')
        delegate = {'X-Titanos-Delegate': 'cloud'}
        plain = self.managed()
        app = BrokerDispatch(S.FakeLocal(), plain)
        for model, code in (('gpt-4o-mini@openai', 'role_forbids'), ('mini', 'role_forbids'),
                            ('anything@hive', 'role_forbids')):
            reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent, headers=delegate,
                           json_body={'model': model, 'messages': CHAT})
            self.assertEqual((reply.status, reply.json()['error']['code']), (403, code), model)
        reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent, headers=delegate,
                       json_body={'model': 'chat-auto', 'messages': CHAT})
        self.assertEqual((reply.status, reply.header('x-titanos-served-by')), (200, 'local:native'))
        self.assertEqual(self.hits(), 0)

        delegated = self.managed(managed_delegation='parent')
        local = S.FakeLocal()
        app = BrokerDispatch(local, delegated)
        reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent, headers=delegate,
                       json_body={'model': 'gpt-4o-mini@openai', 'messages': CHAT})
        self.assertEqual((reply.status, reply.header('x-titanos-served-by')), (200, 'cloud:openai'))
        self.assertEqual(len(MOCKS['openai'].hits), 1)
        local.errors['qwen-local'] = (429, 'rate_limit_error')
        for headers in (delegate, dict(delegate, **{'X-Titanos-Target': 'qwen-local'})):
            reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent, headers=headers,
                           json_body={'model': 'chat-auto', 'messages': CHAT})
            self.assertEqual((reply.status, reply.json()['error']['code']), (429, 'rate_limit_error'))
            self.assertEqual(reply.header('x-titanos-fallback-count'), '0')
        reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent,
                       headers=dict(delegate, **{'X-Titanos-Target': 'chat-auto'}),
                       json_body={'model': 'qwen-local', 'messages': CHAT})
        self.assertEqual((reply.status, reply.json()['error']['code']), (400, 'invalid_request'))
        self.assertEqual(len(MOCKS['openai'].hits), 1)
        self.assertEqual(self.hits(), 1)

        class Slow(S.FakeLocal):
            async def __call__(self, scope, receive, send):
                await asyncio.sleep(0.4)
                return await super().__call__(scope, receive, send)
        slow = Slow()
        app = BrokerDispatch(slow, plain)
        reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent, headers={'X-Titanos-Deadline-Ms': '100'},
                       json_body={'model': 'qwen-local', 'messages': CHAT})
        self.assertEqual((reply.status, reply.json()['error']['code']), (504, 'deadline_exceeded'))
        self.assertEqual(slow.calls, [])  # cancelled before the route read its body; nothing was forwarded
        reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent, headers={'X-Titanos-Deadline-Ms': '5000'},
                       json_body={'model': 'qwen-local', 'messages': CHAT})
        self.assertEqual(reply.status, 200)

        trace = '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01'
        app = BrokerDispatch(S.FakeLocal(), plain)
        reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent, headers={'traceparent': trace},
                       json_body={'model': 'qwen-local', 'messages': CHAT})
        self.assertEqual(reply.header('x-request-id'), trace)
        stored = plain.decisions.query(principal=S.BOOTSTRAP, limit=1)[0]
        self.assertEqual((stored['trace_id'], stored['request_id']), (trace, trace))
        reply = S.call(app, 'POST', '/v1/chat/completions', principal=parent,
                       headers={'traceparent': trace, 'X-Request-ID': 'client-request-1'},
                       json_body={'model': 'qwen-local', 'messages': CHAT})
        self.assertEqual(reply.header('x-request-id'), 'client-request-1')
        self.assertEqual(self.hits(), 1)

    def test_reload_refuses_enabling_egress_in_managed_role_without_delegation(self):
        quiet = document()
        for upstream in quiet['upstreams'].values():
            upstream['enabled'] = False
        self.lab.rewrite(quiet)
        managed = self.managed()
        self.assertEqual(managed.snapshot().config.enabled_upstreams(), {})
        self.lab.rewrite(document())
        with self.assertRaises(BrokerError) as caught:
            managed.reload()
        self.assertEqual((caught.exception.code, caught.exception.status), ('role_forbids', 409))
        self.assertEqual(managed.snapshot().config.enabled_upstreams(), {})
        self.assertEqual(self.managed(managed_delegation='parent').reload()['status'], 'reloaded')
        result = self.broker.reload()
        self.assertEqual((result['status'], result['snapshot']['enabled']), ('reloaded', 8))
        self.assertEqual(self.broker.settings.role, 'standalone')


class Facade(Base):
    def test_catalog_entries(self):
        entries = {e['id']: e for e in self.broker.catalog_for(CLOUD)}
        self.assertEqual(entries['mini']['owned_by'], 'openai')
        self.assertTrue(entries['mini']['remote'])
        self.assertEqual(entries['mini']['titanos']['status'], 'available')
        self.assertIn('gpt-4o-mini@openai', entries)
        self.assertNotIn('gpt-4o-mini@offline', entries)
        self.assertNotIn('gpt-4o-mini@unval', entries)
        self.assertEqual(entries['chat-auto']['owned_by'], 'titanos-virtual')
        self.assertNotIn('targets', entries['chat-auto']['titanos'])
        reader = {e['id']: e for e in self.broker.catalog_for(S.key('inference', 'cloud', 'broker:read'))}
        self.assertEqual(reader['chat-auto']['titanos']['targets'], ['qwen-local', 'gpt-4o-mini@openai'])
        plain = {e['id'] for e in self.broker.catalog_for(S.key('inference'))}
        self.assertTrue(plain <= {'chat-auto', 'cloud-first', 'speech-vm'}, plain)
        tags = self.broker.catalog_for(CLOUD, family='ollama')
        self.assertTrue(all(t['remote'] and t['details']['format'] == 'remote' for t in tags))
        self.assertEqual(self.broker.catalog_for(CLOUD, loaded_only=True), [])
        self.assertEqual(self.broker.model_entry('mini', CLOUD)['id'], 'mini')
        self.assertIsNone(self.broker.model_entry('mini', S.key('inference')))
        self.assertEqual(self.hits(), 0)

    def test_live_catalog_ttl_status_static_fallback_and_switch(self):
        doc = document()
        doc['upstreams']['openai']['catalog'] = 'live'
        doc['upstreams']['openai']['models'] = []
        doc['models'] = {}
        doc['virtual_models'] = {}
        self.lab.rewrite(doc)
        self.broker.reload()
        catalog = self.broker.snapshot().catalog
        self.code('model_not_found', resolve, self.broker, 'gpt-4.1-mini@openai')
        self.assertEqual(catalog.status()['openai'], {'source': 'static', 'fetched_at': None, 'error': None})
        status = S.run(lambda: catalog.refresh_upstream('openai'), self.broker)
        self.assertEqual(status['source'], 'live')
        self.assertTrue(status['fetched_at'])
        self.assertEqual([h.path for h in MOCKS['openai'].hits], ['/v1/models'])
        self.assertEqual(chain(resolve(self.broker, 'gpt-4.1-mini@openai')), ['gpt-4.1-mini@openai'])
        self.assertFalse(catalog.stale('openai'))
        MOCKS['openai'].faults.push('status', status=500)
        status = S.run(lambda: catalog.refresh_upstream('openai'), self.broker)
        self.assertEqual((status['source'], status['error']), ('live', 'upstream_unavailable'))
        self.assertEqual(chain(resolve(self.broker, 'gpt-4.1-mini@openai')), ['gpt-4.1-mini@openai'])
        managed = Broker(settings=S.settings(role='managed'), config=self.lab.local, home=self.lab.home,
                         vault=self.lab.vault)
        before = len(MOCKS['openai'].hits)
        off = S.run(lambda: managed.snapshot().catalog.refresh_upstream('openai'), managed)
        self.assertEqual(off['error'], 'live_catalogs_off')
        self.assertEqual(len(MOCKS['openai'].hits), before)

    def test_startup_info_and_managed_seal(self):
        from titanos.broker.redact import RedactingFilter, uninstall_logging_redaction
        import logging
        handler = logging.StreamHandler()
        root = logging.getLogger()
        root.addHandler(handler)
        self.addCleanup(root.removeHandler, handler)
        self.addCleanup(uninstall_logging_redaction)
        asyncio.run(self.broker.startup())
        self.assertTrue(any(isinstance(f, RedactingFilter) for f in handler.filters))
        kinds = [r['kind'] for r in self.broker.audit.query()]
        self.assertIn('role.start', kinds)
        info = self.broker.info(CLOUD)
        self.assertEqual(set(info), {'configured', 'cloud_policy_effective', 'upstreams_enabled', 'advisors',
                                     'vault_backend', 'sealed'})
        self.assertNotIn('127.0.0.1', repr(info))
        self.assertTrue(info['configured'])
        self.assertEqual(info['cloud_policy_effective'], 'fallback')
        self.assertTrue(self.broker.cloud_reachable())
        managed = Broker(settings=S.settings(role='managed'), config=self.lab.local, home=self.lab.home,
                         vault=self.lab.vault)
        with self.assertLogs('titanos.broker', 'WARNING') as logged:
            asyncio.run(managed.startup())
        self.assertTrue(any('configured upstreams ignored' in line for line in logged.output))
        self.assertTrue(managed.vault.sealed)
        self.assertEqual((managed.info()['sealed'], managed.info()['cloud_policy_effective']), (True, 'off'))
        self.assertFalse(managed.cloud_reachable())
        delegating = Broker(settings=S.settings(role='managed', managed_delegation='parent'), config=self.lab.local,
                            home=self.lab.home, vault=self.lab.vault)
        with self.assertLogs('titanos.broker', 'DEBUG') as logged:
            logging.getLogger('titanos.broker').debug('startup marker')
            asyncio.run(delegating.startup())
        self.assertFalse(any('configured upstreams ignored' in line for line in logged.output))
        self.assertEqual(self.hits(), 0)


class Grants(Base):
    unvalidated = ()

    def issuer(self, **kw):
        settings = S.settings(**kw)
        return GrantIssuer(settings, switches_for(settings), self.lab.vault, config=lambda: self.lab.config)

    def test_managed_role_refuses_every_purpose_without_delegation(self):
        issuer = self.issuer(role='managed')
        for purpose in GRANT_PURPOSES:
            with self.subTest(purpose=purpose):
                for delegated in (False, True):
                    self.code('role_forbids', issuer.issue, purpose, upstream_id='openai', delegated=delegated)
        delegated = self.issuer(role='managed', managed_delegation='parent')
        self.code('role_forbids', delegated.issue, 'inference', upstream_id='openai', delegated=False)
        for purpose in ('probe', 'catalog', 'advisor', 'validate', 'test'):
            self.code('role_forbids', delegated.issue, purpose, upstream_id='openai', delegated=True)
        grant = delegated.issue('inference', upstream_id='openai', delegated=True)
        self.assertEqual((grant.purpose, grant.delegated), ('inference', True))
        self.assertEqual(self.issuer().issue('probe', upstream_id='openai').purpose, 'probe')
        self.code('upstream_unconfigured', self.issuer().issue, 'inference', upstream_id='slot')

    def test_grant_issuer_is_the_only_egress_grant_constructor_and_egress_rechecks(self):
        src = ROOT / 'src' / 'titanos'
        found = []
        for path in sorted(src.rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    for inner in ast.walk(node):
                        if (isinstance(inner, ast.Call) and getattr(inner.func, 'id', None) == 'EgressGrant'):
                            found.append((path.relative_to(src).as_posix(), node.name))
        self.assertTrue(found)
        self.assertEqual({where for where in found}, {('broker/router.py', 'GrantIssuer'),
                                                     ('broker/router.py', 'issue')})
        grant = self.issuer().issue('inference', upstream_id='openai')
        from titanos.broker.egress import EgressClient
        from titanos.broker.profiles import PROFILES
        managed = S.settings(role='managed')
        egress = EgressClient(managed, self.lab.vault, switches_for(managed))
        upstream = self.lab.config.upstreams['openai']
        self.code('role_forbids', egress.send_sync, grant, upstream, PROFILES['openai'], 'GET', '/models')
        self.assertEqual(self.hits(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=1)
