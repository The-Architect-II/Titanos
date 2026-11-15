"""Catalog selection only: fake devices, no probes, runtime imports or installs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hardware_selection_only', ROOT / 'src/titanos/hardware.py')
hardware = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hardware)
PROFILES = json.loads((ROOT / 'tools/profiles.json').read_text(encoding='utf-8'))


def inventory(*capabilities, system='windows', arch='x86_64'):
    return {'os': system, 'arch': arch, 'display_vendors': [],
            'nvidia': [{'index': index, 'capability': value, 'uuid': f'GPU-fake-{index}'}
                       for index, value in enumerate(capabilities)]}


class Tests(unittest.TestCase):
    def setUp(self):
        self.guards = [patch.object(hardware, 'discover', side_effect=AssertionError('No hardware probe')),
                       patch.object(hardware.subprocess, 'run', side_effect=AssertionError('No subprocess')),
                       patch.object(hardware.subprocess, 'check_output', side_effect=AssertionError('No subprocess')),
                       patch.object(hardware.urllib.request, 'build_opener', side_effect=AssertionError('No network'))]
        for guard in self.guards:guard.start()

    def tearDown(self):
        for guard in reversed(self.guards):guard.stop()
        self.assertNotIn('torch', sys.modules)

    def test_available_catalog_profile_replaces_missing_hardcoded_preference(self):
        catalog = {name: value for name, value in PROFILES.items() if name != 'cu126'}
        for capability in (7.0, 7.5, 8.9):
            with self.subTest(capability=capability):
                self.assertEqual(hardware.choose_profile(inventory(capability), catalog)[0], 'cu128')

    def test_available_platform_profile_replaces_incompatible_preference(self):
        catalog = copy.deepcopy(PROFILES)
        catalog['cu126']['platforms'] = ['linux-x86_64']
        self.assertEqual(hardware.choose_profile(inventory(7.5), catalog)[0], 'cu128')
        self.assertEqual(hardware.choose_profile(inventory(7.5, system='linux'), catalog)[0], 'cu126')

    def test_pinned_28_legacy_and_current_ranges(self):
        # These are this catalog's torch2.8 bounds, not a claim about newer wheels.
        self.assertIn('torch==2.8.0', PROFILES['cu126']['packages'])
        self.assertIn('torch==2.8.0', PROFILES['cu128']['packages'])
        for capability in (5.0, 6.0, 6.1, 6.2, 7.0, 7.5, 8.0, 8.6, 8.9, 9.0):
            with self.subTest(capability=capability):
                self.assertEqual(hardware.choose_profile(inventory(capability), PROFILES)[0], 'cu126')
        for capability in (10.0, 12.0, 12.1):
            self.assertEqual(hardware.choose_profile(inventory(capability), PROFILES)[0], 'cu128')

    def test_linux_aarch64_cuda_profile_covers_spark_and_leaves_jetson_orin_out(self):
        self.assertIn('torch==2.10.0', PROFILES['cu130']['packages'])
        for capability in (9.0, 10.0, 11.0, 12.1):
            with self.subTest(capability=capability):
                self.assertEqual(hardware.choose_profile(inventory(capability, system='linux', arch='aarch64'), PROFILES)[::2],
                                 ('cu130', 'cuda:0'))
        with self.assertRaisesRegex(ValueError, 'CC 8.7 on linux-aarch64'):
            hardware.choose_profile(inventory(8.7, system='linux', arch='aarch64'), PROFILES)
        with self.assertRaisesRegex(ValueError, 'no published build for linux-aarch64'):
            hardware.choose_profile(inventory(12.1, system='linux', arch='aarch64'), PROFILES, requested='cu128')
        with self.assertRaisesRegex(ValueError, 'no published build for windows-aarch64'):
            hardware.choose_profile(inventory(12.1, arch='aarch64'), PROFILES, requested='cu130')

    def test_mixed_inventory_and_selected_index_do_not_substitute_devices(self):
        self.assertEqual(hardware.choose_profile(inventory(6.1, 7.5), PROFILES)[0], 'cu126')
        self.assertEqual(hardware.choose_profile(inventory(7.0, 12.0), PROFILES)[0], 'cu128')
        mixed = inventory(6.1, 12.0)
        mixed['nvidia'][1]['index'] = 7
        before = copy.deepcopy(mixed)
        with self.assertRaisesRegex(ValueError, '--gpu'):
            hardware.choose_profile(mixed, PROFILES)
        self.assertEqual(hardware.choose_profile(mixed, PROFILES, gpu=7)[::2], ('cu128', 'cuda:7'))
        self.assertEqual(hardware.choose_profile(mixed, PROFILES, gpu=0)[::2], ('cu126', 'cuda:0'))
        self.assertEqual(mixed, before)
        with self.assertRaisesRegex(ValueError, 'not detected'):
            hardware.choose_profile(mixed, PROFILES, gpu=9)

    def test_catalog_ranges_determine_selection_not_gpu_generation_names(self):
        catalog = copy.deepcopy(PROFILES)
        catalog['cu126']['capability_min'] = 7.5
        catalog['cu128']['capability_min'] = 7.5
        with self.assertRaisesRegex(ValueError, 'catalog'):
            hardware.choose_profile(inventory(7.0), catalog)
        catalog['cu124'] = {**copy.deepcopy(PROFILES['cu126']), 'capability_max': 7.5}
        self.assertEqual(hardware.choose_profile(inventory(7.0), catalog)[0], 'cu124')
        self.assertEqual(hardware.choose_profile(inventory(7.5), catalog)[0], 'cu126')
        reverse = dict(reversed(list(catalog.items())))
        self.assertEqual(hardware.choose_profile(inventory(7.0), reverse)[0], 'cu124')

    def test_unknown_and_outside_catalog_capabilities_fail_without_cpu_fallback(self):
        for capability in (None, '7.5', True, float('nan'), float('inf'), -1):
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(ValueError, 'compute capability'):
                    hardware.choose_profile(inventory(capability), PROFILES)
        for capability in (3.5, 13.0, 99.0):
            with self.assertRaisesRegex(ValueError, 'catalog'):
                hardware.choose_profile(inventory(capability), PROFILES)
        with self.assertRaisesRegex(ValueError, 'windows-aarch64'):
            hardware.choose_profile(inventory(7.5, arch='aarch64'), PROFILES)

    def test_explicit_profiles_remain_exact_and_do_not_fall_back(self):
        for capability, requested in ((6.1, 'cu128'), (12.0, 'cu126')):
            with self.assertRaisesRegex(ValueError, 'does not cover'):
                hardware.choose_profile(inventory(capability), PROFILES, requested=requested)
        self.assertEqual(hardware.choose_profile(inventory(7.0), PROFILES, requested='cu128')[0], 'cu128')
        self.assertEqual(hardware.choose_profile(inventory(None), PROFILES, requested='cpu')[2], 'cpu')
        with self.assertRaisesRegex(ValueError, '--gpu'):
            hardware.choose_profile(inventory(7.5), PROFILES, requested='cpu', gpu=0)
        with self.assertRaisesRegex(ValueError, 'Unknown runtime profile'):
            hardware.choose_profile(inventory(7.5), PROFILES, requested='cu777')

    def test_missing_or_invalid_catalog_bounds_do_not_claim_compatibility(self):
        for lower, upper in ((None, 12.1), ('7.0', 12.1), (True, 12.1),
                             (7.0, float('inf')), (7.0, float('nan')), (8.0, 7.0)):
            with self.subTest(lower=lower, upper=upper):
                profile = {**PROFILES['cu128'], 'capability_min': lower, 'capability_max': upper}
                catalog = {'cpu': PROFILES['cpu'], 'cu128': profile}
                with self.assertRaisesRegex(ValueError, 'catalog'):
                    hardware.choose_profile(inventory(7.5), catalog)
                with self.assertRaisesRegex(ValueError, 'does not cover'):
                    hardware.choose_profile(inventory(7.5), catalog, requested='cu128')

    def test_non_cuda_defaults_and_detection_failure_stay_explicit(self):
        self.assertEqual(hardware.choose_profile(inventory(), PROFILES)[0], 'cpu')
        host = inventory(); host['display_vendors'] = ['intel']
        self.assertEqual(hardware.choose_profile(host, PROFILES)[0], 'dml')
        self.assertEqual(hardware.choose_profile(inventory(system='darwin', arch='aarch64'), PROFILES)[0], 'mps')
        host = inventory(7.5); host['nvidia_probe_error'] = 'inert failure'
        with self.assertRaisesRegex(ValueError, 'NVIDIA detection failed'):
            hardware.choose_profile(host, PROFILES)
        self.assertEqual(hardware.choose_profile(host, PROFILES, requested='cpu')[0], 'cpu')
        # An NVIDIA display adapter without nvidia-smi rows never lands on DirectML or CPU silently.
        host = inventory(); host['display_vendors'] = ['nvidia']
        with self.assertRaisesRegex(ValueError, 'nvidia-smi'):
            hardware.choose_profile(host, PROFILES)
        # Outside Windows x86_64 a display vendor alone selects nothing; the message lists that platform's non-CUDA profiles.
        host = inventory(system='linux'); host['display_vendors'] = ['intel']
        with self.assertRaisesRegex(ValueError, 'linux-x86_64: .*rocm64.*xpu'):
            hardware.choose_profile(host, PROFILES)
        self.assertEqual(hardware.choose_profile(host, PROFILES, requested='cpu')[0], 'cpu')
        host = inventory(arch='aarch64'); host['display_vendors'] = ['amd']
        with self.assertRaisesRegex(ValueError, 'windows-aarch64: .*dml-arm64.*qnn'):
            hardware.choose_profile(host, PROFILES)


if __name__ == '__main__':unittest.main()
