"""Router advisors (WP6): generic-v1, the HiveMind GIM router profile and the inert pending-contract slot, against the WP4
loopback AdvisorMock (netguard installed; zero non-loopback attempts).

- generic-v1 and GIM answers reorder or narrow the deterministic chain; the GIM's fallback_chain is used (mapped through
  label_map);
- a slow advisor (timeout) keeps the deterministic order (advisor_timeout);
- invalid, foreign or malformed answers are ignored (advisor_invalid);
- no prompt text reaches an advisor unless share_excerpt is set (then at most 512 scrubbed characters);
- pending-contract is never called, and configuring a url for it is refused with 409 advisor_profile_unconfigured;
- advisors are not consulted in managed role; validate_advice unit rules.
"""
from __future__ import annotations

import asyncio
import json
import unittest

import broker_wp6_support as S
from broker_mocks import AdvisorMock, AnthropicMock, OpenAIMock, netguard

from titanos.broker import inbound
from titanos.broker.advisor import (CONTRACT, AdvisorCandidate, AdvisorResponse, GimRouterAdvisor, build_advisor,
                                    gim_label, validate_advice)
from titanos.broker.config import AdvisorSpec, BrokerConfig
from titanos.broker.errors import BrokerError
from titanos.broker.facade import Broker
from titanos.broker.role import parse_parent_headers
from titanos.broker.surfaces import SURFACES
from titanos.broker.types import RouteContext

GPT, CLAUDE, LOCAL = 'gpt-4o-mini@openai', 'claude-sonnet-4-5@anthropic', 'qwen-local'
PROMPT = 'Compare two sorting algorithms for large datasets. secret-ish words: not a key'
MOCKS: dict = {}


def setUpModule():
    MOCKS['openai'] = OpenAIMock()
    MOCKS['anthropic'] = AnthropicMock()
    MOCKS['generic'] = AdvisorMock('generic-v1')
    MOCKS['gim'] = AdvisorMock('gim')
    for mock in MOCKS.values():
        mock.start()


def tearDownModule():
    for mock in MOCKS.values():
        mock.stop()


def document(*, generic=None, gim=None, extra_advisors=None) -> dict:
    # generous timeouts keep slow first connections (debug interpreters, loaded hosts) from reading as timeouts
    advisors = {'gen': dict({'profile': 'generic-v1', 'url': MOCKS['generic'].url, 'timeout_ms': 5000}, **(generic or {})),
                'gim': dict({'profile': 'hivemind-gim-router', 'url': MOCKS['gim'].url, 'share_excerpt': True,
                             'timeout_ms': 5000,
                             'label_map': {'simple': LOCAL, 'complex': GPT, 'reasoning': CLAUDE}}, **(gim or {})),
                'slot': {'profile': 'pending-contract', 'label': 'operator router'}}
    advisors.update(extra_advisors or {})
    targets = [{'ref': LOCAL}, {'ref': GPT}, {'ref': CLAUDE}]
    return {'policy': {'cloud_policy': 'fallback', 'auto_model': 'auto-vm'},
            'upstreams': {'openai': S.upstream(MOCKS['openai'], 'openai', credential='oa', models=['gpt-4o-mini']),
                          'anthropic': S.upstream(MOCKS['anthropic'], 'anthropic', credential='an',
                                                  models=['claude-sonnet-4-5'])},
            'advisors': advisors,
            'virtual_models': {
                'smart': {'capability': 'chat', 'strategy': 'advisor', 'advisor': 'gen', 'targets': targets},
                'gimvm': {'capability': 'chat', 'strategy': 'advisor', 'advisor': 'gim', 'targets': targets},
                'slotvm': {'capability': 'chat', 'strategy': 'advisor', 'advisor': 'slot', 'targets': targets},
                'auto-vm': {'capability': 'chat', 'strategy': 'advisor', 'advisor': 'gen', 'targets': targets}}}


class AdvisorCase(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        for mock in MOCKS.values():
            mock.reset()
        self.labs = []

    def tearDown(self):
        for lab in self.labs:
            lab.close()
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connect was attempted')

    def lab(self, doc=None, **kw) -> S.Lab:
        lab = S.Lab(doc or document(), local={LOCAL: ('chat', 'llama-cpp')}, **kw)
        self.labs.append(lab)
        return lab

    def route(self, broker, model, *, principal=None, text=PROMPT):
        principal = principal or S.key('inference', 'cloud')
        row = SURFACES['openai.chat']
        body = {'model': model, 'messages': [{'role': 'user', 'content': text}]}
        context = RouteContext(row, model, principal, parse_parent_headers({}, broker.settings, principal),
                               inbound.summarize(row, body, None, len(json.dumps(body))))
        decision = broker.snapshot().router.new_decision(context)

        async def go():
            try:
                return await broker.snapshot().router.route(context, decision,
                                                            advisor_input=lambda: inbound.advisor_input(row, body))
            finally:
                await broker.close()
        plan = asyncio.run(go())
        return [t.ref for t in plan.chain], plan, decision

    def advisor_hits(self) -> int:
        return len(MOCKS['generic'].hits) + len(MOCKS['gim'].hits)


class Mapping(AdvisorCase):
    def test_generic_v1_reorders_the_chain(self):
        lab = self.lab()
        MOCKS['generic'].choice = CLAUDE
        refs, plan, decision = self.route(lab.broker, 'smart')
        self.assertEqual(refs, [CLAUDE, LOCAL, GPT])  # choice, then the mock's two fallbacks
        self.assertTrue(plan.advisor.valid)
        self.assertEqual((decision.advisor.choice, decision.advisor.profile), (CLAUDE, 'generic-v1'))
        sent = MOCKS['generic'].hits[-1].json()
        self.assertEqual(sent['contract'], CONTRACT)
        self.assertEqual([c['ref'] for c in sent['candidates']], [LOCAL, GPT, CLAUDE])
        self.assertEqual({c['class'] for c in sent['candidates']}, {'local', 'cloud'})
        self.assertNotIn('127.0.0.1', json.dumps(sent))  # no upstream URLs
        self.assertEqual(len(MOCKS['openai'].hits) + len(MOCKS['anthropic'].hits), 0)
        # the advisor may narrow: a choice without fallbacks keeps only the choice
        MOCKS['generic'].answer = {'contract': CONTRACT, 'choice': GPT, 'fallbacks': []}
        self.assertEqual(self.route(lab.broker, 'smart')[0], [GPT])
        # model 'auto' uses policy.auto_model and its advisor
        MOCKS['generic'].answer = None
        MOCKS['generic'].choice = GPT
        self.assertEqual(self.route(lab.broker, 'auto')[0][0], GPT)

    def test_gim_mapping_uses_the_fallback_chain(self):
        lab = self.lab()
        refs, plan, decision = self.route(lab.broker, 'gimvm')
        # recommended LLM_BIG_NIM (complex -> GPT); fallback_chain LLM_BIG, LLM_MID, LLM_SMALL -> complex, simple
        self.assertEqual(refs, [GPT, LOCAL])
        self.assertEqual(decision.advisor.profile, 'hivemind-gim-router')
        self.assertEqual(decision.advisor.fallbacks, (LOCAL,))
        sent = MOCKS['gim'].hits[-1]
        self.assertTrue(sent.path.endswith('/gim/llm_router/func_1'))
        self.assertEqual(sent.json()['messages'][0]['content'], PROMPT)  # share_excerpt is required for the GIM
        MOCKS['gim'].reasoning_mode = 'on'
        refs, plan, _ = self.route(lab.broker, 'gimvm')
        self.assertEqual(refs[0], CLAUDE)
        self.assertEqual(plan.reasoning_mode, 'medium')
        MOCKS['gim'].reasoning_mode = None
        MOCKS['gim'].recommend = 'LLM_MID'
        MOCKS['gim'].fallback_chain = ['LLM_BIG_NIM']
        self.assertEqual(self.route(lab.broker, 'gimvm')[0], [LOCAL, GPT])
        self.assertEqual(gim_label('LLM_SMALL'), 'simple')
        self.assertEqual(gim_label('LLM_BIG', {'reasoning_mode': 'on'}), 'reasoning')

    def test_gim_without_share_excerpt_is_never_called(self):
        lab = self.lab(document(gim={'share_excerpt': False}))
        refs, plan, _ = self.route(lab.broker, 'gimvm')
        self.assertEqual(refs, [LOCAL, GPT, CLAUDE])
        self.assertEqual((plan.advisor.valid, plan.advisor.reason), (False, 'advisor_unconfigured'))
        self.assertEqual(self.advisor_hits(), 0)


class Robustness(AdvisorCase):
    def test_timeout_keeps_the_deterministic_order(self):
        lab = self.lab(document(generic={'timeout_ms': 100}))
        MOCKS['generic'].choice = CLAUDE
        MOCKS['generic'].faults.push('slow_first_byte', delay=1.5)
        refs, plan, decision = self.route(lab.broker, 'smart')
        self.assertEqual(refs, [LOCAL, GPT, CLAUDE])
        self.assertEqual((plan.advisor.valid, plan.advisor.reason), (False, 'advisor_timeout'))
        self.assertLess(plan.advisor.ms, 1000)
        self.assertIn('advisor:advisor_timeout', decision.notes)

    def test_invalid_or_foreign_choices_are_ignored(self):
        lab = self.lab()
        answers = [{'contract': CONTRACT, 'choice': 'not-a-candidate@openai', 'fallbacks': []},
                   {'contract': CONTRACT, 'choice': GPT, 'fallbacks': ['foreign@elsewhere']},
                   {'contract': 'other/v9', 'choice': GPT, 'fallbacks': []},
                   {'choice': GPT},
                   {'contract': CONTRACT, 'choice': GPT, 'confidence': 7},
                   ['not', 'an', 'object']]
        for answer in answers:
            with self.subTest(answer=answer):
                MOCKS['generic'].answer = answer
                refs, plan, _ = self.route(lab.broker, 'smart')
                self.assertEqual(refs, [LOCAL, GPT, CLAUDE])
                self.assertEqual((plan.advisor.valid, plan.advisor.reason), (False, 'advisor_invalid'))
        MOCKS['generic'].answer = None
        MOCKS['generic'].faults.push('status', status=500)
        refs, plan, _ = self.route(lab.broker, 'smart')
        self.assertEqual((refs, plan.advisor.reason), ([LOCAL, GPT, CLAUDE], 'advisor_failed'))
        # the advisor cannot widen scope: an invisible candidate is never offered, and choosing it is refused
        narrow = S.key('inference', 'cloud:openai')
        MOCKS['generic'].answer = {'contract': CONTRACT, 'choice': CLAUDE, 'fallbacks': []}
        refs, plan, _ = self.route(lab.broker, 'smart', principal=narrow)
        self.assertEqual(refs, [LOCAL, GPT])
        self.assertNotIn(CLAUDE, json.dumps(MOCKS['generic'].hits[-1].json()['candidates']))

    def test_no_prompt_text_unless_share_excerpt(self):
        lab = self.lab()
        self.route(lab.broker, 'smart')
        sent = MOCKS['generic'].hits[-1]
        self.assertIsNone(sent.json()['summary']['excerpt'])
        self.assertNotIn(b'sorting algorithms', sent.body)
        self.assertEqual(sent.json()['summary']['message_count'], 1)
        shared = self.lab(document(generic={'share_excerpt': True}))
        long_text = 'word ' * 400 + 'sk-' + 'A' * 40
        self.route(shared.broker, 'smart', text=long_text)
        excerpt = MOCKS['generic'].hits[-1].json()['summary']['excerpt']
        self.assertLessEqual(len(excerpt), 512)
        self.assertNotIn('sk-' + 'A' * 40, MOCKS['generic'].hits[-1].body.decode())

    def test_advisor_credential_is_bearer_from_an_advisor_record(self):
        lab = self.lab(document(generic={'credential': 'adv-cred'}))
        lab.advisor_credential('adv-cred', MOCKS['generic'].origin)
        MOCKS['generic'].choice = GPT
        refs, plan, _ = self.route(lab.broker, 'smart')
        self.assertEqual(refs[0], GPT)
        self.assertTrue(MOCKS['generic'].hits[-1].presented(S.SENTINELS['advisor']))


class Slot(AdvisorCase):
    def test_pending_contract_is_never_called_and_a_url_is_409(self):
        lab = self.lab()
        before = lab.broker.snapshot().issuer.issued
        refs, plan, _ = self.route(lab.broker, 'slotvm')
        self.assertEqual(refs, [LOCAL, GPT, CLAUDE])
        self.assertEqual((plan.advisor.valid, plan.advisor.reason), (False, 'contract_unknown'))
        self.assertEqual(lab.broker.snapshot().issuer.issued, before)
        self.assertEqual(self.advisor_hits(), 0)
        bad = document()
        bad['advisors']['slot'] = {'profile': 'pending-contract', 'url': 'http://127.0.0.1:9/advise'}
        with self.assertRaises(BrokerError) as caught:
            BrokerConfig.model_validate(bad)
        self.assertEqual((caught.exception.code, caught.exception.status), ('advisor_profile_unconfigured', 409))
        (lab.home / 'broker.json').write_text(json.dumps(bad), encoding='utf-8')
        with self.assertRaises(BrokerError) as caught:
            lab.broker.reload()
        self.assertEqual(caught.exception.status, 409)
        advisor = build_advisor(AdvisorSpec(profile='pending-contract'), None, None, None)
        self.assertIsNone(asyncio.run(advisor.advise(None)))

    def test_managed_role_never_consults_advisors(self):
        lab = self.lab()
        managed = Broker(settings=S.settings(role='managed'), config=lab.local, home=lab.home, vault=lab.vault)
        refs, plan, _ = self.route(managed, 'smart', principal=S.key('inference', 'cloud'))
        self.assertEqual(refs, [LOCAL])
        self.assertIsNone(plan.advisor)
        self.assertEqual(self.advisor_hits(), 0)


class Validation(unittest.TestCase):
    def test_validate_advice(self):
        candidates = [AdvisorCandidate(ref=LOCAL, klass='local'), AdvisorCandidate(ref=GPT, klass='cloud')]
        ok = AdvisorResponse(contract=CONTRACT, choice=GPT, fallbacks=[LOCAL, GPT])
        self.assertEqual(validate_advice(ok, candidates), (GPT, [LOCAL]))
        self.assertEqual(validate_advice({'contract': CONTRACT, 'choice': LOCAL}, [LOCAL, GPT]), (LOCAL, []))
        for bad in (None, {'contract': CONTRACT, 'choice': 'x'}, {'contract': 'v0', 'choice': GPT},
                    {'contract': CONTRACT, 'choice': GPT, 'fallbacks': ['y']}, {'contract': CONTRACT},
                    {'contract': CONTRACT, 'choice': GPT, 'explanation': 'x' * 300}):
            self.assertIsNone(validate_advice(bad, candidates), bad)
        gim = GimRouterAdvisor('g', AdvisorSpec(profile='hivemind-gim-router', share_excerpt=True,
                                                label_map={'simple': LOCAL}))
        self.assertIsNone(gim.map_result({'recommended_model': 'LLM_BIG', 'fallback_chain': []}, [LOCAL, GPT]))


if __name__ == '__main__':
    unittest.main(verbosity=1)
