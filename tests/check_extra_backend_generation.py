"""ExtraBackend._generation precedence: request > model defaults > process settings, greedy at zero.

PYTHONPATH=src python -B tests/check_extra_backend_generation.py
No checkpoint loads and no torch import: titanos.extra_backends imports without torch or
transformers, so ``transformers`` is stubbed in sys.modules only for the StoppingCriteria base
class the cancel criterion subclasses, and titanos.extra_backends.load_settings is patched per
case with a real titanos.settings.Settings instance.
Closes the ROOT_CONFIG_PRECEDENCE_AUDIT finding that TITANOS_DEFAULT_TEMPERATURE and
TITANOS_DEFAULT_TOP_P were never applied to translation / image-to-text generation.
"""
import copy
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import titanos.extra_backends as extra_backends  # noqa: E402
from titanos.settings import Settings  # noqa: E402


class StoppingCriteria:
    """Stand-in for transformers.StoppingCriteria."""


TRANSFORMERS = types.ModuleType('transformers')
TRANSFORMERS.StoppingCriteria = StoppingCriteria
TRANSFORMERS.StoppingCriteriaList = list

PROCESS = dict(default_max_tokens=256, default_temperature=.7, default_top_p=1.0)
CASES = []


def generation(request=None, defaults=None, settings=None, cancel=None):
    """Call the real ExtraBackend._generation on an unloaded instance; inputs must stay unmodified."""
    backend = extra_backends.ExtraBackend.__new__(extra_backends.ExtraBackend)
    backend.spec = SimpleNamespace(defaults=defaults if defaults is not None else {})
    request = request if request is not None else {}
    before = (copy.deepcopy(backend.spec.defaults), copy.deepcopy(request))
    settings = settings or Settings(**PROCESS)
    with patch.dict(sys.modules, {'transformers': TRANSFORMERS}), \
            patch.object(extra_backends, 'load_settings', return_value=settings):
        kwargs = backend._generation(request, cancel)
    assert (backend.spec.defaults, request) == before, 'model defaults and request must not be mutated'
    return kwargs


class GenerationPrecedence(unittest.TestCase):
    def check(self, name, kwargs, expected, absent=()):
        for key, value in expected.items():
            self.assertIn(key, kwargs, f'{name}: {key} missing from {kwargs}')
            self.assertEqual(kwargs[key], value, f'{name}: {key}')
        for key in absent:
            self.assertNotIn(key, kwargs, f'{name}: {key} must be absent from {kwargs}')
        CASES.append(name)

    def test_c01_process_defaults_fill_empty_request_and_model(self):
        kwargs = generation({}, {})
        self.check('process_defaults_fill_empty_layers', kwargs,
                   {'temperature': .7, 'top_p': 1.0, 'do_sample': True, 'max_new_tokens': 256})
        self.assertEqual(set(kwargs), {'temperature', 'top_p', 'do_sample', 'max_new_tokens'})

    def test_c02_process_zero_temperature_selects_greedy(self):
        # independent AST reproduction: process max_tokens=99, temperature=0, top_p=.25, empty request and model.
        settings = Settings(default_max_tokens=99, default_temperature=0, default_top_p=.25)
        kwargs = generation({}, {}, settings)
        self.check('process_zero_temperature_is_greedy', kwargs, {'do_sample': False, 'max_new_tokens': 99},
                   absent=('temperature', 'top_p'))
        self.assertEqual(kwargs, {'do_sample': False, 'max_new_tokens': 99})

    def test_c03_model_defaults_override_process(self):
        kwargs = generation({}, {'temperature': .4, 'top_p': .8, 'generation_kwargs': {'num_beams': 4}})
        self.check('model_defaults_override_process', kwargs,
                   {'temperature': .4, 'top_p': .8, 'do_sample': True, 'num_beams': 4, 'max_new_tokens': 256})

    def test_c04_request_overrides_model_and_process(self):
        kwargs = generation({'temperature': .1, 'top_p': .6}, {'temperature': .4, 'top_p': .8})
        self.check('request_overrides_model_and_process', kwargs,
                   {'temperature': .1, 'top_p': .6, 'do_sample': True})

    def test_c05_explicit_request_zero_is_greedy_over_model_and_process(self):
        kwargs = generation({'temperature': 0}, {'temperature': .4, 'top_p': .8})
        self.check('request_zero_is_a_value_selecting_greedy', kwargs, {'do_sample': False},
                   absent=('temperature', 'top_p'))
        self.assertEqual(kwargs, {'do_sample': False, 'max_new_tokens': 256})

    def test_c06_model_zero_is_greedy_over_process(self):
        kwargs = generation({}, {'temperature': 0})
        self.check('model_zero_is_greedy_over_process', kwargs, {'do_sample': False},
                   absent=('temperature', 'top_p'))
        kwargs = generation({}, {'generation_kwargs': {'temperature': 0, 'top_p': .9}})
        self.check('generation_kwargs_zero_is_greedy_and_drops_top_p', kwargs, {'do_sample': False},
                   absent=('temperature', 'top_p'))

    def test_c07_model_do_sample_false_survives_positive_temperature(self):
        kwargs = generation({}, {'do_sample': False})
        self.check('model_do_sample_false_wins_over_setdefault', kwargs,
                   {'do_sample': False, 'temperature': .7, 'top_p': 1.0})
        kwargs = generation({'temperature': .7}, {'do_sample': False, 'temperature': .3})
        self.check('model_do_sample_false_wins_with_request_temperature', kwargs,
                   {'do_sample': False, 'temperature': .7, 'top_p': 1.0})

    def test_c08_max_tokens_precedence_and_ceiling(self):
        both = {'max_tokens': 30, 'max_new_tokens': 20}
        self.assertEqual(generation({'max_tokens': 40}, both)['max_new_tokens'], 40)
        self.assertEqual(generation({}, both)['max_new_tokens'], 30)
        self.assertEqual(generation({}, {'max_new_tokens': 20})['max_new_tokens'], 20)
        self.assertEqual(generation({}, {})['max_new_tokens'], 256)
        self.assertEqual(generation({'max_tokens': None}, both)['max_new_tokens'], 30)
        settings = Settings(max_tokens=64, default_max_tokens=32)
        self.assertEqual(generation({}, {}, settings)['max_new_tokens'], 32)
        with self.assertRaises(ValueError):
            generation({'max_tokens': 65}, {}, settings)
        with self.assertRaises(ValueError):
            generation({}, {'max_tokens': 65}, settings)
        # Unchanged `or` chain: a zero max_tokens is falsy and falls through to the next layer.
        # The API Plus contracts reject 0 upstream (gt=0 / ge=1), so _generation never sees it.
        self.assertEqual(generation({'max_tokens': 0}, {}, settings)['max_new_tokens'], 32)
        with self.assertRaises(ValueError):
            generation({'max_tokens': True}, {}, settings)
        with self.assertRaises(ValueError):
            generation({'max_tokens': 2.5}, {}, settings)
        CASES.append('max_tokens_precedence_unchanged_and_ceiling_enforced')

    def test_c09_cancel_event_adds_stopping_criterion(self):
        self.assertNotIn('stopping_criteria', generation({}, {}, cancel=None))
        cancel = threading.Event()
        kwargs = generation({}, {}, cancel=cancel)
        criteria = kwargs['stopping_criteria']
        self.assertIsInstance(criteria, list)
        self.assertEqual(len(criteria), 1)
        self.assertIsInstance(criteria[0], StoppingCriteria)
        self.assertFalse(criteria[0](None, None))
        cancel.set()
        self.assertTrue(criteria[0](None, None))
        CASES.append('cancel_event_becomes_stopping_criterion')

    def test_c10_none_is_omission_at_request_and_model_layers(self):
        kwargs = generation({'temperature': None, 'top_p': None}, {'temperature': .4})
        self.check('request_None_falls_to_model_then_process', kwargs,
                   {'temperature': .4, 'top_p': 1.0, 'do_sample': True})
        kwargs = generation({}, {'temperature': None, 'top_p': None})
        self.check('model_None_falls_to_process', kwargs, {'temperature': .7, 'top_p': 1.0, 'do_sample': True})
        kwargs = generation({'top_p': .5}, {})
        self.check('partial_request_mixes_layers', kwargs, {'temperature': .7, 'top_p': .5, 'do_sample': True})

    def test_c11_other_allowed_controls_keep_their_layers(self):
        kwargs = generation({'top_k': 0, 'repetition_penalty': 1.3},
                            {'top_k': 50, 'num_beams': 2, 'generation_kwargs': {'length_penalty': .9}})
        self.check('other_controls_request_over_generation_kwargs_over_defaults', kwargs,
                   {'top_k': 0, 'repetition_penalty': 1.3, 'num_beams': 2, 'length_penalty': .9,
                    'temperature': .7, 'top_p': 1.0, 'do_sample': True})


if __name__ == '__main__':
    result = unittest.main(exit=False, verbosity=1).result
    assert 'torch' not in sys.modules, 'no model framework may be imported by this test'
    if not result.wasSuccessful():
        sys.exit(1)
    print(f'PASS check_extra_backend_generation: {len(CASES)} cases ' + ', '.join(CASES))
