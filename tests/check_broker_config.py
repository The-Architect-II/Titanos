"""BrokerConfig (WP1): schema round trip, secret-shape refusal anywhere in the document, origin rules (https, loopback and
LAN http, userinfo/query/fragment, blocked literals, IDNA), name conflicts, keyless-forbidden policy fields and atomic writes
under the shared lock with the bounded Windows replace retry. Temporary directories only; no server, model or network."""
import ipaddress
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pydantic import ValidationError  # noqa: E402
from titanos.broker import config as bc  # noqa: E402
from titanos.broker.errors import BrokerConfigSecretError, BrokerError  # noqa: E402
from titanos.broker.redact import REGISTRY  # noqa: E402
from titanos.config import home  # noqa: E402
from titanos.settings import Settings  # noqa: E402

# A private-range literal computed from the network (synthetic; never a lab address).
LAN_IP = str(ipaddress.ip_network((0x0A000000, 8))[0x0A0B0C])
SYNTHETIC_KEY = 'sk-test-' + 'a1b2c3d4' * 4


def full_document():
    return {
        'version': 1,
        'policy': {'cloud_policy': 'explicit', 'default_location': 'prefer_local', 'anonymous_cloud': False,
                   'allow_http_lan': True, 'max_fallbacks': 3, 'auto_model': 'smart'},
        'upstreams': {
            'openai': {'profile': 'openai', 'credential': 'openai-main', 'enabled': True, 'models': ['gpt-4.1-mini'],
                       'catalog': 'live', 'headers': {'OpenAI-Project': 'proj_fixture'},
                       'timeouts': {'connect': 5, 'first_byte': 60, 'idle': 30},
                       'data_policy': {'modalities': ['text', 'image'], 'allow_store': False}, 'label': 'Primary'},
            'anthropic': {'profile': 'anthropic', 'credential': 'ant', 'anthropic_version': '2023-06-01',
                          'anthropic_betas': ['structured-outputs-2025-11-13']},
            'azure': {'profile': 'azure_openai', 'credential': 'az',
                      'azure': {'resource': 'fixture-resource', 'api_version': '2024-10-21',
                                'deployments': {'gpt-4o': 'gpt4o-deploy'}}},
            'lan_vllm': {'profile': 'custom_openai', 'base_url': f'http://{LAN_IP}:8000/v1', 'surfaces': ['openai.chat'],
                         'allow_any_model': True},
            'gw': {'profile': 'hivemind_gateway', 'base_url': 'http://127.0.0.1:6089/v1'},
            'slot': {'profile': 'pending_contract'},
        },
        'models': {'gpt-4.1-mini': {'target': 'gpt-4.1-mini@openai', 'capabilities': ['chat', 'tools', 'vision']},
                   'embed-qa': {'target': 'nvidia/nv-embedqa-e5-v5@openai', 'capabilities': ['embeddings'],
                                'inject': {'input_type': 'passage'}}},
        'virtual_models': {'smart': {'capability': 'chat', 'strategy': 'ordered',
                                     'targets': [{'ref': 'qwen3-8b'}, {'ref': 'gpt-4.1-mini@openai'}],
                                     'fallback_on': ['queue_full', 'upstream_unavailable'], 'max_attempts': 2},
                           'tts': {'capability': 'speech', 'strategy': 'advisor', 'advisor': 'local',
                                   'targets': [{'ref': 'local-tts'}]}},
        'advisors': {'local': {'profile': 'generic-v1', 'url': 'http://127.0.0.1:7000/advise', 'timeout_ms': 200,
                               'label_map': {'simple': 'qwen3-8b'}},
                     'later': {'profile': 'pending-contract'}},
        'voice_maps': {'openai_to_elevenlabs': {'alloy': 'voice-fixture'}, 'elevenlabs_to_openai': {'voice-fixture': 'alloy'}},
        'prices': [{'match': 'gpt-4.1-mini@openai', 'input_per_mtok': '0.40', 'output_per_mtok': '1.60',
                    'effective_from': '2026-10-01'}],
        'compat': {'hivemind_suffix_aliases': {'_cloud': 'openai'}},
    }


class Schema(unittest.TestCase):
    def test_round_trip(self):
        cfg = bc.BrokerConfig.model_validate(full_document())
        dumped = cfg.model_dump(mode='json')
        self.assertEqual(bc.BrokerConfig.model_validate(dumped), cfg)
        self.assertEqual(dumped['prices'][0]['input_per_mtok'], '0.40')
        self.assertEqual(json.loads(json.dumps(dumped)), dumped)
        self.assertEqual(cfg.virtual_models['smart'].effective_max_attempts(), 2)
        self.assertEqual(cfg.virtual_models['tts'].effective_max_attempts(), 1)
        self.assertEqual(bc.VirtualModel(capability='embeddings', targets=[{'ref': 'e'}]).effective_max_attempts(), 3)
        # the null default is still bounded by 1 + max_fallbacks when the caller passes the cap
        chat = bc.VirtualModel(capability='chat', targets=[{'ref': 'a'}])
        self.assertEqual(chat.effective_max_attempts(0), 1)
        self.assertEqual(chat.effective_max_attempts(1), 2)
        self.assertEqual(chat.effective_max_attempts(5), 3)
        self.assertEqual(cfg.virtual_models['smart'].effective_max_attempts(0), 1)
        self.assertEqual(cfg.virtual_models['tts'].effective_max_attempts(5), 1)
        self.assertEqual(cfg.upstreams['azure'].origin(allow_http_lan=False), 'https://fixture-resource.openai.azure.com:443')
        self.assertEqual(cfg.upstreams['openai'].origin(allow_http_lan=False), 'https://api.openai.com:443')
        self.assertTrue(cfg.upstreams['gw'].effective_allow_any_model())
        self.assertFalse(cfg.upstreams['openai'].effective_allow_any_model())
        self.assertEqual(cfg.upstreams['openai'].effective_modalities(), frozenset({'text', 'image'}))
        self.assertEqual(cfg.upstreams['lan_vllm'].effective_surfaces(), frozenset({'openai.chat'}))
        self.assertFalse(cfg.is_disabled())
        self.assertTrue(bc.BrokerConfig().is_disabled())
        self.assertEqual(bc.BrokerConfig().policy.cloud_policy, 'off')

    def test_extra_fields_and_bad_values_refused(self):
        cases = [
            {'unknown': 1},
            {'policy': {'cloud_policy': 'always'}},
            {'policy': {'max_fallbacks': 6}},
            {'upstreams': {'Bad': {'profile': 'openai'}}},
            {'upstreams': {'store': {'profile': 'openai'}}},
            {'upstreams': {'x': {'profile': 'no_such_profile'}}},
            {'upstreams': {'x': {'profile': 'openai', 'surprise': True}}},
            {'upstreams': {'x': {'profile': 'openai', 'surfaces': ['openai.chat']}}},  # not an operator-surface profile
            {'upstreams': {'x': {'profile': 'custom_openai', 'base_url': 'https://gw.example.invalid/v1',
                                 'surfaces': ['anthropic.messages']}}},
            {'upstreams': {'x': {'profile': 'custom_openai'}}},  # operator profile without base_url
            {'upstreams': {'x': {'profile': 'openai', 'headers': {'X-Unlisted': 'v'}}}},
            {'upstreams': {'x': {'profile': 'openai', 'anthropic_version': '2023-06-01'}}},
            {'upstreams': {'x': {'profile': 'azure_openai'}}},
            {'upstreams': {'x': {'profile': 'openai', 'data_policy': {'modalities': ['video']}}}},
            {'upstreams': {'x': {'profile': 'deepseek', 'data_policy': {'modalities': ['audio']}}}},
            {'upstreams': {'x': {'profile': 'openai', 'credential': 'Not_Valid'}}},
            {'models': {'m': {'target': 'plain-name'}}},
            {'models': {'m': {'target': 'gpt@missing'}}},
            {'models': {'bad@name': {'target': 'gpt@openai'}}, 'upstreams': {'openai': {'profile': 'openai'}}},
            {'virtual_models': {'v': {'capability': 'chat', 'targets': []}}},
            {'virtual_models': {'v': {'capability': 'chat', 'targets': [{'ref': 'a'}], 'max_attempts': 4}}},
            {'virtual_models': {'v': {'capability': 'chat', 'strategy': 'advisor', 'targets': [{'ref': 'a'}]}}},
            {'virtual_models': {'v': {'capability': 'chat', 'strategy': 'weighted', 'targets': [{'ref': 'a'}]}}},
            {'virtual_models': {'v': {'capability': 'chat', 'targets': [{'ref': 'm@nowhere'}]}}},
            {'virtual_models': {'v': {'capability': 'chat', 'targets': [{'ref': 'w'}]},
                                'w': {'capability': 'chat', 'targets': [{'ref': 'a'}]}}},
            {'policy': {'auto_model': 'nothing'}},
            {'prices': [{'match': '*', 'input_per_mtok': '-1'}]},
            {'prices': [{'match': '*', 'input_per_mtok': 'NaN'}]},
            {'prices': [{'match': '*', 'currency': 'EUR'}]},
            {'compat': {'hivemind_suffix_aliases': {'_cloud': 'Not An Id'}}},
            {'advisors': {'a': {'profile': 'generic-v1', 'url': 'http://example.invalid/advise'}}},
        ]
        for document in cases:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    bc.BrokerConfig.model_validate(document)
        with self.assertRaises(BrokerError) as caught:
            bc.BrokerConfig.model_validate({'advisors': {'a': {'profile': 'pending-contract',
                                                               'url': 'http://127.0.0.1:9/x'}}})
        self.assertEqual((caught.exception.code, caught.exception.status), ('advisor_profile_unconfigured', 409))
        with self.assertRaises(BrokerError) as caught:  # the upstream slot takes no url either
            bc.BrokerConfig.model_validate({'upstreams': {'slot': {'profile': 'pending_contract',
                                                                    'base_url': 'https://slot.example.invalid/v1'}}})
        self.assertEqual((caught.exception.code, caught.exception.status), ('advisor_profile_unconfigured', 409))


class Secrets(unittest.TestCase):
    def assert_refused(self, document):
        with self.assertRaises(BrokerConfigSecretError) as caught:
            bc.BrokerConfig.model_validate(document)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual((caught.exception.code, caught.exception.status), ('broker_config_contains_secret', 400))
        with self.assertRaises(BrokerConfigSecretError) as parsed:
            bc.parse_broker_config(json.dumps(document))
        with self.assertRaises(ValidationError) as constructed:  # constructor path: wrapped, input hidden
            bc.BrokerConfig(**document)
        for exc in (caught.exception, parsed.exception, constructed.exception):
            self.assertNotIn(SYNTHETIC_KEY, str(exc))
            self.assertNotIn('a1b2c3d4', str(exc))

    def test_secret_shape_anywhere_is_refused(self):
        placements = [
            lambda d: d['upstreams']['openai'].__setitem__('label', 'key ' + SYNTHETIC_KEY),
            lambda d: d['upstreams']['openai']['headers'].__setitem__('OpenAI-Project', SYNTHETIC_KEY),
            lambda d: d['upstreams']['openai']['models'].append(SYNTHETIC_KEY),
            lambda d: d['models'].__setitem__(SYNTHETIC_KEY, {'target': 'x@openai'}),
            lambda d: d['prices'][0].__setitem__('source', 'Bearer abcdefghijklmnop'),
            lambda d: d['voice_maps']['openai_to_elevenlabs'].__setitem__('alloy', 'xi-api-key: abcdefgh12345678'),
            lambda d: d['upstreams']['lan_vllm'].__setitem__('base_url', 'https://h.example.invalid/v1?key=abcdef123456'),
            lambda d: d['models']['embed-qa']['inject'].__setitem__('token', 'nvapi-' + 'Q' * 30),
            lambda d: d['advisors']['local'].__setitem__('label', 'hf_' + 'Z' * 24),
            lambda d: d['upstreams']['gw'].__setitem__('label', 'tk_abcdefghij_' + 'A' * 43),
            lambda d: d['upstreams']['gw'].__setitem__('label', 'AIza' + 'b' * 35),
            lambda d: d['upstreams']['gw'].__setitem__('label', 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig'),
        ]
        for place in placements:
            document = full_document()
            place(document)
            with self.subTest(document=json.dumps(document)[:80]):
                self.assert_refused(document)

    def test_registered_exact_value_is_refused(self):
        value = 'registered-fixture-value-' + os.urandom(6).hex()
        REGISTRY.add(value)
        try:
            document = full_document()
            document['upstreams']['openai']['label'] = value
            self.assert_refused(document)
        finally:
            REGISTRY.discard(value)
        document = full_document()
        document['upstreams']['openai']['label'] = value
        bc.BrokerConfig.model_validate(document)

    def test_ordinary_ids_are_not_secret_shaped(self):
        document = full_document()
        document['upstreams']['openai']['models'] = ['gpt-4.1-mini', 'o4-mini', 'task-classifier-large-v2',
                                                     'accounts/fireworks/models/llama-v3p1-70b-instruct',
                                                     'meta-llama/Llama-3.1-8B-Instruct:fastest', 'gpt-oss:120b']
        bc.BrokerConfig.model_validate(document)


class Origins(unittest.TestCase):
    def test_accepted_forms(self):
        cases = {
            'https://api.openai.com/v1': 'https://api.openai.com:443',
            'HTTPS://API.Example.INVALID./v1/': 'https://api.example.invalid:443',
            'https://h.example.invalid:8443': 'https://h.example.invalid:8443',
            'https://Bücher.example.invalid/x': 'https://xn--bcher-kva.example.invalid:443',
            'http://127.0.0.1:11440/v1': 'http://127.0.0.1:11440',
            'http://[::1]/v1': 'http://[::1]:80',
            'https://192.0.2.10/v1': 'https://192.0.2.10:443',
        }
        for url, origin in cases.items():
            with self.subTest(url=url):
                self.assertEqual(bc.parse_origin(url, allow_http_lan=False), origin)
        self.assertEqual(bc.parse_origin(f'http://{LAN_IP}:8000/v1', allow_http_lan=True), f'http://{LAN_IP}:8000')

    def test_refused_forms(self):
        refused = ['http://api.example.invalid/v1', f'http://{LAN_IP}:8000/v1', 'http://192.0.2.10/v1',
                   'https://user:pw@h.example.invalid', 'https://user@h.example.invalid', 'https://h.example.invalid/v1?a=1',
                   'https://h.example.invalid/v1#frag', 'https://h.example.invalid/v1?', 'ftp://h.example.invalid',
                   'file:///etc/passwd', 'https://', 'https://0.0.0.0/v1', 'https://169.254.169.254/latest',
                   'https://[fe80::1]/', 'https://224.0.0.1/', 'https://[::ffff:127.0.0.1]/', 'https://[fd00:ec2::254]/',
                   'https://h.example.invalid:0/', 'https://h.example.invalid:99999/', 'https://h%2eexample.invalid/',
                   'https://bad_host.example.invalid/', ' https://h.example.invalid', 'https://h.example.invalid\\@x/',
                   'https://-lead.example.invalid/', '',
                   # legacy numeric IPv4 spellings of the metadata and loopback addresses are not domain names
                   'https://0xa9.0xfe.0xa9.0xfe/', 'https://0xa9fea9fe/', 'https://2852039166/',
                   'https://0251.0376.0251.0376/', 'https://169.254.43518/', 'https://0x7f.1/', 'https://h.example.0x1/']
        for url in refused:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    bc.parse_origin(url, allow_http_lan=False)
        for url in ('http://api.example.invalid/v1', 'https://user@h.example.invalid', 'https://169.254.169.254/'):
            with self.assertRaises(ValueError):
                bc.parse_origin(url, allow_http_lan=True)

    def test_lan_http_follows_policy(self):
        document = {'upstreams': {'lan': {'profile': 'custom_openai', 'base_url': f'http://{LAN_IP}:8000/v1'}}}
        with self.assertRaises(ValidationError):
            bc.BrokerConfig.model_validate(document)
        bc.BrokerConfig.model_validate({**document, 'policy': {'allow_http_lan': True}})


class Names(unittest.TestCase):
    def test_conflicts(self):
        cfg = bc.BrokerConfig.model_validate(full_document())
        bc.check_name_conflicts(cfg, {'qwen3-8b', 'local-tts'})
        for local in ({'gpt-4.1-mini'}, {'smart'}):
            with self.subTest(local=local):
                with self.assertRaises(BrokerError) as caught:
                    bc.check_name_conflicts(cfg, local)
                self.assertEqual((caught.exception.code, caught.exception.status), ('broker_name_conflict', 409))
                self.assertEqual(caught.exception.response_headers()['x-should-retry'], 'false')
        document = full_document()
        document['virtual_models']['gpt-4.1-mini'] = {'capability': 'chat', 'targets': [{'ref': 'qwen3-8b'}]}
        with self.assertRaises(ValidationError):
            bc.BrokerConfig.model_validate(document)

    def test_keyless_forbidden_policy_fields(self):
        self.assertEqual(bc.KEYLESS_FORBIDDEN_POLICY_FIELDS, frozenset({'anonymous_cloud', 'allow_http_lan'}))
        old = bc.BrokerPolicy()
        self.assertEqual(bc.forbidden_policy_changes(old, bc.BrokerPolicy(cloud_policy='fallback')), set())
        self.assertEqual(bc.forbidden_policy_changes(old, bc.BrokerPolicy(anonymous_cloud=True, allow_http_lan=True)),
                         {'anonymous_cloud', 'allow_http_lan'})


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='titanos-broker-config-')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'home' / 'broker.json'

    def leftovers(self):
        # the lock file may or may not survive release (platform-dependent); no temporary file may remain
        return sorted(p.name for p in self.path.parent.iterdir() if p.name != 'broker.json.lock')

    def test_paths(self):
        with patch.dict(os.environ, {'TITANOS_HOME': self.tmp.name}):
            # the same resolved spelling as titanos.config.home() (models.json state), so lock and audit paths agree
            self.assertEqual(bc.broker_path(Settings()), Path(self.tmp.name).resolve() / 'broker.json')
            self.assertEqual(bc.broker_dir(), Path(self.tmp.name).resolve() / 'broker')
            self.assertEqual(bc.broker_dir().parent, home())
            self.assertFalse((Path(self.tmp.name) / 'broker').exists())  # path helpers create nothing
        explicit = str(Path(self.tmp.name).resolve() / 'elsewhere.json')
        self.assertEqual(bc.broker_path(Settings(broker_config=explicit)), Path(explicit))

    def test_missing_file_is_default_and_write_round_trips(self):
        self.assertEqual(bc.read_broker_config(self.path), bc.BrokerConfig())
        self.assertFalse(self.path.parent.exists())
        cfg = bc.BrokerConfig.model_validate(full_document())
        bc.write_broker_config(self.path, cfg)
        self.assertEqual(bc.read_broker_config(self.path), cfg)
        self.assertTrue(self.path.read_bytes().endswith(b'}\n'))
        self.assertNotIn(b'\r\n', self.path.read_bytes())
        self.assertEqual(self.leftovers(), ['broker.json'])

    def test_read_refuses_secret_and_bad_json(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({'upstreams': {'o': {'profile': 'openai', 'label': SYNTHETIC_KEY}}}), encoding='utf-8')
        with self.assertRaises(ValueError) as caught:
            bc.read_broker_config(self.path)
        self.assertNotIn(SYNTHETIC_KEY, str(caught.exception))
        self.path.write_text('{not json', encoding='utf-8')
        with self.assertRaises(ValueError):
            bc.read_broker_config(self.path)

    def test_update_holds_the_lock_and_retries_replace(self):
        bc.write_broker_config(self.path, bc.BrokerConfig())
        real_replace = os.replace
        calls = []

        def flaky_replace(src, dst):
            calls.append(dst)
            if len(calls) == 1:
                raise PermissionError(13, 'simulated sharing violation')
            return real_replace(src, dst)

        def change(cfg):
            from filelock import FileLock, Timeout
            with self.assertRaises(Timeout):
                FileLock(str(self.path) + '.lock', timeout=0).acquire()
            cfg.policy.cloud_policy = 'explicit'
            cfg.upstreams['openai'] = bc.Upstream(profile='openai', credential='openai-main')

        with patch.object(bc.os, 'replace', side_effect=flaky_replace), patch.object(bc.time, 'sleep') as sleep:
            result = bc.update_broker_config(self.path, change)
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once()
        self.assertEqual(result.policy.cloud_policy, 'explicit')
        self.assertEqual(bc.read_broker_config(self.path), result)
        self.assertEqual(self.leftovers(), ['broker.json'])

    def test_persistent_permission_error_is_bounded_and_leaves_file(self):
        bc.write_broker_config(self.path, bc.BrokerConfig())
        before = self.path.read_bytes()
        with patch.object(bc.os, 'replace', side_effect=PermissionError(13, 'held')) as replace, \
             patch.object(bc.time, 'sleep'):
            with self.assertRaises(PermissionError):
                bc.write_broker_config(self.path, bc.BrokerConfig(policy={'cloud_policy': 'explicit'}))
        self.assertEqual(replace.call_count, bc.REPLACE_ATTEMPTS)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.leftovers(), ['broker.json'])

    def test_failed_change_leaves_file_unchanged(self):
        bc.write_broker_config(self.path, bc.BrokerConfig())
        before = self.path.read_bytes()

        def bad_change(cfg):
            cfg.models['x'] = bc.PublishedModel(target='gpt@nowhere')

        with self.assertRaises(ValueError):
            bc.update_broker_config(self.path, bad_change)

        def failing(cfg):
            cfg.policy.cloud_policy = 'fallback'
            raise RuntimeError('caller aborted')

        with self.assertRaises(RuntimeError):
            bc.update_broker_config(self.path, failing)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=1)
