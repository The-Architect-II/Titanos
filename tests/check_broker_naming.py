"""Broker model-reference grammar (WP1): the last-'@' split, REF_MAX, blank refs, upstream-id shape and the longest-first
compat suffix expansion (the HiveMind '_cloud' shadowing '_ollama_cloud' regression). Pure functions; no I/O."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos.broker import naming  # noqa: E402
from titanos.broker.config import Compat  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.naming import ModelRef, expand_compat_suffix, format_ref, parse_ref  # noqa: E402


class LastAtSplit(unittest.TestCase):
    def test_qualified_refs(self):
        cases = {
            'gpt-4o-mini@openai': ('gpt-4o-mini', 'openai'),
            'accounts/fireworks/models/llama-v3p1-70b-instruct@fireworks':
                ('accounts/fireworks/models/llama-v3p1-70b-instruct', 'fireworks'),
            'anthropic/claude-sonnet-4.5@openrouter': ('anthropic/claude-sonnet-4.5', 'openrouter'),
            'meta-llama/Llama-3.1-8B-Instruct:fastest@hf_router': ('meta-llama/Llama-3.1-8B-Instruct:fastest', 'hf_router'),
            'meta-llama/Llama-3.1-8B-Instruct:cheapest@hf_router': ('meta-llama/Llama-3.1-8B-Instruct:cheapest', 'hf_router'),
            'gpt-oss:120b@ollama_cloud': ('gpt-oss:120b', 'ollama_cloud'),
            'user@domain-model@openai': ('user@domain-model', 'openai'),
            'models/gemini-2.0-flash@gemini': ('models/gemini-2.0-flash', 'gemini'),
            'x@a': ('x', 'a'),
            'm@' + 'u' * 32: ('m', 'u' * 32),
        }
        for text, (model, upstream) in cases.items():
            with self.subTest(text=text):
                ref = parse_ref(text)
                self.assertEqual(ref, ModelRef('qualified', model, upstream))
                self.assertEqual(str(ref), text)
                self.assertEqual(format_ref(model, upstream), text)

    def test_plain_refs(self):
        for text in ('qwen3-8b', 'gpt-4o-mini', 'model@', '@openai', 'm@Upper', 'm@9digits', 'm@has space',
                     'm@' + 'u' * 33, 'm@store', 'org/model:tag', 'm@open.ai'):
            with self.subTest(text=text):
                ref = parse_ref(text)
                self.assertEqual((ref.kind, ref.model, ref.upstream), ('plain', text, None))
                self.assertEqual(str(ref), text)

    def test_blank_and_length(self):
        for blank in ('', '   ', '\t', None, 7):
            with self.subTest(blank=blank):
                with self.assertRaises(BrokerError) as caught:
                    parse_ref(blank)
                self.assertEqual((caught.exception.code, caught.exception.status), ('model_required', 400))
        self.assertEqual(naming.REF_MAX, 256)
        exact = 'm' * 249 + '@openai'
        self.assertEqual(len(exact), 256)
        self.assertEqual(parse_ref(exact).upstream, 'openai')
        with self.assertRaises(BrokerError) as caught:
            parse_ref('m' * 250 + '@openai')
        self.assertEqual((caught.exception.code, caught.exception.status), ('invalid_request', 400))
        with self.assertRaises(ValueError):
            format_ref('m' * 250, 'openai')
        for model, upstream in (('', 'openai'), ('m', 'Bad'), ('m', 'store'), ('m', '')):
            with self.assertRaises(ValueError):
                format_ref(model, upstream)

    def test_local_names_cannot_collide_with_qualified_refs(self):
        self.assertTrue(naming.is_local_name('qwen3-8b'))
        self.assertTrue(naming.is_local_name('org/model.v2_q4'))
        for text in ('gpt@openai', 'gpt-oss:120b', '-lead', '', 'x' * 129):
            self.assertFalse(naming.is_local_name(text), text)
        self.assertTrue(naming.valid_upstream_id('ollama_cloud'))
        self.assertTrue(naming.valid_upstream_id('a-b_c9'))
        for bad in ('Openai', '9x', 'store', '', 'x' * 33, 'a.b'):
            self.assertFalse(naming.valid_upstream_id(bad), bad)


class CompatSuffix(unittest.TestCase):
    ALIASES = {'_cloud': 'openai', '_ollama_cloud': 'ollama_cloud', '_oc': 'ollama_cloud'}

    def test_longest_suffix_first(self):
        for compat in (self.ALIASES, Compat(hivemind_suffix_aliases=self.ALIASES)):
            with self.subTest(compat=type(compat).__name__):
                self.assertEqual(expand_compat_suffix('m_ollama_cloud', compat), 'm@ollama_cloud')
                self.assertEqual(expand_compat_suffix('gpt-oss:120b_ollama_cloud', compat), 'gpt-oss:120b@ollama_cloud')
                self.assertEqual(expand_compat_suffix('gpt-4o_cloud', compat), 'gpt-4o@openai')
                self.assertEqual(expand_compat_suffix('m_oc', compat), 'm@ollama_cloud')
                self.assertIsNone(expand_compat_suffix('_cloud', compat))  # nothing before the suffix
                self.assertIsNone(expand_compat_suffix('qwen3-8b', compat))
        # insertion order must not matter
        reversed_order = dict(reversed(list(self.ALIASES.items())))
        self.assertEqual(expand_compat_suffix('m_ollama_cloud', reversed_order), 'm@ollama_cloud')

    def test_expansion_is_always_a_qualified_ref(self):
        expanded = expand_compat_suffix('llama3_cloud', self.ALIASES)
        self.assertEqual(parse_ref(expanded), ModelRef('qualified', 'llama3', 'openai'))
        self.assertIsNone(expand_compat_suffix('m_cloud', None))
        self.assertIsNone(expand_compat_suffix('m_cloud', {}))
        self.assertIsNone(expand_compat_suffix('m_cloud', Compat()))


if __name__ == '__main__':
    unittest.main(verbosity=1)
