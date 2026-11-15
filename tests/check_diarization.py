"""Diarization contracts with inert Sherpa doubles; no native runtime or models."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from titanos import diarization as d
from titanos.audio_contract import DiarizationResult
from titanos.capabilities import describe
from titanos.config import Config, ModelSpec, WorkerSpec
from titanos.engine import Engine, EngineError
from titanos.settings import Settings, _active_settings


class NativeConfig:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def validate(self): return True


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = {'id': 'fixture-reviewed-pair', 'sample_rate': 16000,
                        'max_speakers_per_frame': 2, 'components': {},
                        'abi': dict(d.SEGMENTATION_ABI),
                        'runtime_pin': {'sherpa-onnx': '1.13.8', 'sherpa-onnx-core': '1.13.8'},
                        'license': 'synthetic test bytes',
                        'provenance': {'fixture': 'synthetic bytes; no trained-model claim'}}
        for role in ['segmentation', 'embedding']:
            raw = ('inert-' + role).encode()
            (self.root / (role + '.onnx')).write_bytes(raw)
            self.profile['components'][role] = {'path': role + '.onnx', 'bytes': len(raw),
                                                'sha256': hashlib.sha256(raw).hexdigest()}
        self.manifest = {'schema_version': 1, 'profile_id': self.profile['id'],
                         'components': self.profile['components']}
        self.write_manifest()
        catalog = self.root / 'trusted-profiles.json'
        catalog.write_text(json.dumps({'schema_version': 1, 'profiles': [self.profile]}))
        self.patch(patch.object(d, 'CATALOG', catalog))
        self.settings = Settings(threads=2)
        token = _active_settings.set(self.settings)
        self.addCleanup(_active_settings.reset, token)
        self.spec = ModelSpec(path=str(self.root), task='diarization', backend='sherpa-onnx')
        self.rows, self.created, self.calls = [], [], []
        self.on_process = None
        owner = self
        class Runtime:
            sample_rate = 16000
            def __init__(self, config):
                self.config = config
                owner.created.append(self)
            def process(self, audio, callback):
                owner.calls.append((audio, self.config))
                if owner.on_process: owner.on_process()
                callback(1, 2)
                return NS(sort_by_start_time=lambda: list(owner.rows))
        self.fake = NS(__version__='1.13.8', onnxruntime_version='1.28.2',
                       OfflineSpeakerDiarization=Runtime,
                       **{name: NativeConfig for name in ['OfflineSpeakerDiarizationConfig',
                         'OfflineSpeakerSegmentationModelConfig', 'OfflineSpeakerSegmentationPyannoteModelConfig',
                         'SpeakerEmbeddingExtractorConfig', 'FastClusteringConfig']})
        self.patch(patch.dict(sys.modules, {'sherpa_onnx': self.fake}))
        self.patch(patch.object(d, 'version', return_value='1.13.8'))
        self.audio = [0.0] * (16000 * 12)
        self.patch(patch('titanos.media.read_audio', side_effect=lambda raw, rate, cancel=None: self.audio))

    def patch(self, context):
        result = context.start()
        self.addCleanup(context.stop)
        return result

    def write_manifest(self):
        (self.root / 'diarization.json').write_text(json.dumps(self.manifest))

    def request(self, **controls):
        return d.SherpaDiarizationModel(self.spec).infer('diarization', {'audio': b'inert audio', **controls})

    def test_pinned_runtime_version_is_exported_string(self):
        self.assertIsInstance(self.fake.onnxruntime_version, str)
        model = d.SherpaDiarizationModel(self.spec)
        self.assertEqual(model.runtime_versions['sherpa_bundled_onnxruntime'], '1.28.2')
        self.assertEqual(len(self.created), 1)
        self.assertEqual(self.calls, [])
        model.close()
        self.assertIsNone(model.runtime)

    def test_overlap_labels_reset_and_no_identity_or_confidence(self):
        self.rows = [NS(speaker=17, start=3., end=7.), NS(speaker=42, start=1., end=5.),
                     NS(speaker=42, start=8., end=9.)]
        model = d.SherpaDiarizationModel(self.spec)
        result = model.infer('diarization', {'audio': b'inert audio'})
        self.assertEqual(result['speakers'], [{'id': 'speaker_0'}, {'id': 'speaker_1'}])
        self.assertEqual(result['segments'][0]['speaker'], 'speaker_0')
        self.assertEqual(result['overlaps'], [{'speakers': ['speaker_0', 'speaker_1'],
            'start': 3., 'end': 5., 'start_sample': 48000, 'end_sample': 80000}])
        self.assertEqual(result['parameters']['segmentation_chunks'], 3)
        self.assertEqual(result['input_sha256'], hashlib.sha256(b'inert audio').hexdigest())
        self.assertNotIn('confidence', result['segments'][0])
        self.assertNotIn('identity', result['speakers'][0])
        self.rows = [NS(speaker=991, start=0., end=1.)]
        again = model.infer('diarization', {'audio': b'inert audio'})
        self.assertEqual(again['speakers'], [{'id': 'speaker_0'}])
        self.assertEqual(again['overlaps'], [])
        DiarizationResult.model_validate(result)

    def test_silence_is_empty_not_an_invented_speaker(self):
        result = self.request(num_speakers=2)
        self.assertEqual([result[k] for k in ['speakers', 'segments', 'overlaps']], [[], [], []])
        self.assertEqual(result['decoded_samples'], 192000)
        self.assertEqual(result['provenance']['runtime_versions']['sherpa_bundled_onnxruntime'], '1.28.2')

    def test_engine_dispatch_and_unload_use_diarization_runtime(self):
        engine = Engine(Config(models={'speaker': self.spec}))
        self.addCleanup(engine.close)
        self.rows = [NS(speaker=7, start=1., end=2.)]
        result = engine.infer('speaker', 'diarization', {'audio': b'inert audio'})
        self.assertIsInstance(engine.runtime, d.SherpaDiarizationModel)
        self.assertEqual(result['segments'][0]['speaker'], 'speaker_0')
        self.assertEqual(engine.catalog(loaded_only=True)['data'][0]['id'], 'speaker')
        engine.unload('speaker')
        self.assertIsNone(engine.runtime)
        self.assertEqual(engine.catalog(loaded_only=True)['data'], [])

    def test_precedence_nulls_zero_and_constructor_duration_controls(self):
        self.spec.defaults = {'num_speakers': 4, 'cluster_threshold': .7, 'min_duration_on': .8}
        values = d.prepare(self.spec, {'num_speakers': 0, 'cluster_threshold': None, 'min_duration_on': 0},
                           Settings(diarization_min_duration_off=.9))
        self.assertEqual(values, {'num_speakers': 0, 'cluster_threshold': .7,
                                 'min_duration_on': 0., 'min_duration_off': .9})
        model = d.SherpaDiarizationModel(self.spec)
        model.infer('diarization', {'audio': b'x', 'min_duration_on': 0, 'min_duration_off': 0})
        self.assertEqual(len(self.created), 2)
        self.assertEqual(self.created[-1].config.min_duration_on, 0)
        self.assertEqual(self.created[-1].config.min_duration_off, 0)
        self.assertEqual(self.created[-1].config.segmentation.provider, 'cpu')
        self.assertEqual(self.created[-1].config.embedding.provider, 'cpu')
        self.assertEqual(self.created[-1].config.embedding.num_threads, 2)
        self.assertEqual(self.created[-1].config.clustering.num_clusters, 4)

    def test_invalid_controls_fail_before_engine_load(self):
        engine = Engine(Config(models={'speaker': self.spec}))
        with patch.object(engine, '_load') as load:
            for controls in [{'num_speakers': True}, {'num_speakers': -1}, {'num_speakers': 33},
                             {'cluster_threshold': float('nan')}, {'cluster_threshold': -1},
                             {'cluster_threshold': 1e100}, {'min_duration_on': 601},
                             {'temperature': .4}]:
                with self.subTest(controls=controls), self.assertRaises(EngineError):
                    engine.infer('speaker', 'diarization', {'audio': b'x', **controls})
            load.assert_not_called()
        with self.assertRaises(ValueError):
            d.prepare(self.spec, {'num_speakers': 2}, duration=10.)

    def test_short_audio_rejects_ignored_clustering_without_process(self):
        self.audio = [0.0] * 16000
        model = d.SherpaDiarizationModel(self.spec)
        for controls in [{'num_speakers': 2}, {'cluster_threshold': .7}]:
            with self.assertRaisesRegex(ValueError, 'bypass clustering'):
                model.infer('diarization', {'audio': b'x', **controls})
        self.assertEqual(self.calls, [])
        result = model.infer('diarization', {'audio': b'x'})
        self.assertFalse(result['parameters']['clustering_applied'])

    def test_chunks_and_speaker_limits_are_operator_configurable(self):
        with patch.object(d, 'current_settings', return_value=Settings(diarization_max_chunks=2)):
            with self.assertRaisesRegex(ValueError, 'MAX_CHUNKS'): self.request()
        self.assertEqual(self.calls, [])
        self.rows = [NS(speaker=i, start=float(i), end=float(i)+.5) for i in range(3)]
        with patch.object(d, 'current_settings', return_value=Settings(diarization_max_speakers=2)):
            with self.assertRaises(EngineError) as error: self.request()
        self.assertEqual(error.exception.code, 'invalid_backend_response')
        self.assertEqual(d.prepare(self.spec, {'num_speakers': 40}, Settings(diarization_max_speakers=40))['num_speakers'], 40)

    def test_cancellation_before_callback_and_after_native_return(self):
        model = d.SherpaDiarizationModel(self.spec)
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(EngineError) as error:
            model.infer('diarization', {'audio': b'x'}, cancel=cancel)
        self.assertEqual(error.exception.status, 499)
        self.assertEqual(self.calls, [])
        cancel.clear(); self.on_process = cancel.set
        with self.assertRaises(EngineError):
            model.infer('diarization', {'audio': b'x'}, cancel=cancel)
        self.assertEqual(len(self.calls), 1)
        cancel.clear()
        def no_callback(audio, callback):
            cancel.set()
            return NS(sort_by_start_time=lambda: self.fail('Cancelled output must not be read'))
        model.runtime.process = no_callback
        with self.assertRaises(EngineError):
            model.infer('diarization', {'audio': b'x'}, cancel=cancel)

    def test_invalid_native_output_and_bounded_eof_quantization(self):
        bad = [NS(speaker=-1, start=0., end=1.), NS(speaker=True, start=0., end=1.),
               NS(speaker=0, start=float('nan'), end=1.), NS(speaker=0, start=2., end=1.),
               NS(speaker=0, start=0., end=13.)]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(EngineError):
                d.segments_from_native([row], 192000, 32)
        with self.assertRaises(EngineError):
            d.segments_from_native([NS(speaker=0,start=0.,end=2.), NS(speaker=0,start=1.,end=3.)],192000,32)
        speakers, spans, clipped = d.segments_from_native([NS(speaker=4,start=11.,end=12.02)],192000,32)
        self.assertEqual((spans[0]['end_sample'], clipped), (192000, 1))

    def test_bundle_hash_cannot_be_self_authorized(self):
        (self.root/'embedding.onnx').write_bytes(b'bad')
        self.manifest['components']['embedding']['bytes'] = 3
        self.manifest['components']['embedding']['sha256'] = hashlib.sha256(b'bad').hexdigest()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'trusted|reviewed'):
            d.SherpaDiarizationModel(self.spec)
        self.assertEqual(self.created, [])

    def test_bundle_reparse_and_version_refuse_before_native(self):
        original = Path.lstat
        def linked(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == self.root/'embedding.onnx':
                return NS(st_mode=value.st_mode, st_file_attributes=0x400)
            return value
        with patch.object(Path, 'lstat', linked), self.assertRaisesRegex(ValueError, 'reparse'):
            d.inspect_bundle(self.root)
        with patch.object(d, 'version', return_value='1.0.0'), self.assertRaisesRegex(ValueError, '1.13.8'):
            d.SherpaDiarizationModel(self.spec)
        self.assertEqual(self.created, [])

    def test_changed_checkpoint_is_rejected_on_native_consumption(self):
        original = self.fake.OfflineSpeakerDiarization
        def change_on_load(config):
            runtime = original(config)
            (self.root/'segmentation.onnx').write_bytes(b'changed-during-load')
            return runtime
        self.fake.OfflineSpeakerDiarization = change_on_load
        with self.assertRaisesRegex(ValueError, 'changed during load'):
            d.SherpaDiarizationModel(self.spec)

    def test_cpu_registration_replica_policy_capabilities_and_close(self):
        for change in [{'backend': 'onnx'}, {'device': 'cuda:0'}, {'dtype': 'float32'},
                       {'trust_remote_code': True}, {'allow_pickle': True}, {'load_options': {'custom': True}},
                       {'defaults': {'voice': 'other'}}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ModelSpec.model_validate({**self.spec.model_dump(), **change})
        with self.assertRaises(ValueError):
            Config(models={'speaker': self.spec}, workers={'w': WorkerSpec(models=['speaker'],device='cuda:0')})
        cap = describe(self.spec)
        self.assertEqual(cap['task'], 'diarization'); self.assertIsNone(cap['qualified'])
        model = d.SherpaDiarizationModel(self.spec); model.close()
        self.assertIsNone(model.runtime)

    def test_settings_validation_and_environment_controls(self):
        settings = Settings.from_env({'TITANOS_DIARIZATION_NUM_SPEAKERS': '2',
            'TITANOS_DIARIZATION_CLUSTER_THRESHOLD': '0', 'TITANOS_DIARIZATION_MIN_DURATION_ON': '0',
            'TITANOS_DIARIZATION_MAX_CHUNKS': '900'})
        self.assertEqual(settings.diarization_max_chunks, 900)
        self.assertEqual(d.prepare(self.spec, {}, settings)['num_speakers'], 2)
        for change in [{'diarization_num_speakers': -1}, {'diarization_max_chunks': 0},
                       {'diarization_min_duration_on': 601}, {'diarization_cluster_threshold': float('inf')}]:
            with self.subTest(change=change), self.assertRaises(ValueError): Settings(**change)


if __name__ == '__main__': unittest.main(verbosity=2)
