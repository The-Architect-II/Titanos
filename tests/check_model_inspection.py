"""Passive setup diagnostics over inert folders; no runtime or model execution."""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from titanos.capabilities import describe
from titanos.config import Config, ModelSpec, write_config
from titanos.model_inspection import inspect_setup
from titanos.settings import Settings


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.discovery = patch('titanos.model_inspection.find_spec', return_value=object())
        self.discovery.start()
        self.addCleanup(self.discovery.stop)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

    def model(self, **kwargs):
        return ModelSpec(path=str(self.root), task='chat', device='cpu', **kwargs)

    def codes(self, value):
        return {row['code'] for row in value['issues']}

    def complete(self):
        self.put('config.json', {'model_type': 'unfamiliar_future_model', 'architectures': ['FutureModel'], 'torch_dtype': 'future_dtype'})
        self.put('model.safetensors', b'not loaded or claimed as valid weights')
        self.put('tokenizer.json', b'not parsed during passive vocabulary presence check')

    def test_empty_directory_actionable(self):
        value = describe(self.model())['setup']
        self.assertEqual(value['status'], 'blocked')
        self.assertTrue({'missing_file', 'missing_weights', 'missing_tokenizer'} <= self.codes(value))
        self.assertTrue(all(row['next_action'] for row in value['issues']))
        self.assertFalse(value['model_loaded'])

    def test_unfamiliar_architecture_dtype_is_not_rejected(self):
        self.complete()
        value = describe(self.model())
        self.assertEqual(value['setup']['status'], 'ready_for_load_check')
        self.assertEqual(value['setup']['architecture']['compatibility'], 'unverified')
        self.assertEqual(value['setup']['architecture']['torch_dtype'], 'future_dtype')
        self.assertIsNone(value['native_tp_plan'])
        self.assertIsNone(value['modes']['tensor_parallel']['native_plan_present'])
        self.assertTrue(value['modes']['tensor_parallel']['implemented'])
        self.assertIsNone(value['setup']['qualified'])

    def test_shards_missing_and_escaping(self):
        self.complete()
        (self.root/'model.safetensors').unlink()
        self.put('model.safetensors.index.json', {'weight_map': {'a': 'part1.safetensors', 'b': 'part2.safetensors', 'bad': '../outside.safetensors'}})
        self.put('part1.safetensors', b'inert shard')
        value = inspect_setup(self.model())
        self.assertIn('invalid_shard_path', self.codes(value))
        self.assertTrue(any(row['files'] == ['part2.safetensors'] for row in value['issues']))

    def test_preprocessing_and_definite_adapter_mismatch(self):
        self.complete()
        self.put('config.json', {'model_type': 'sam2'})
        spec = ModelSpec(path=str(self.root), task='image-segmentation', device='cpu')
        value = inspect_setup(spec)
        self.assertIn('unsupported_architecture', self.codes(value))
        self.assertTrue(any(row['files'] == ['preprocessor_config.json'] for row in value['issues']))

    def test_oversize_invalid_and_nonobject_metadata(self):
        self.complete()
        value = inspect_setup(self.model(), Settings(model_inspection_max_metadata_bytes=4))
        self.assertIn('metadata_limit', self.codes(value))
        self.assertNotIn('invalid_metadata', self.codes(value))
        for raw in (b'broken json', b'[]', b'\xff'):
            self.put('config.json', raw)
            self.assertIn('invalid_metadata', self.codes(inspect_setup(self.model())))

    def test_entry_limit_is_unknown_not_missing(self):
        self.complete()
        value = inspect_setup(self.model(), Settings(model_inspection_max_files=1))
        self.assertFalse(value['inventory']['complete'])
        self.assertIn('inspection_limit', self.codes(value))
        self.assertEqual(value['status'], 'needs_review')

    def test_links_not_followed_or_blanket_rejected(self):
        self.complete()
        target = self.root/'actual_config.json'
        (self.root/'config.json').rename(target)
        try:
            (self.root/'config.json').symlink_to(target)
        except OSError as exc:
            if getattr(exc, 'winerror', None) == 1314:
                self.skipTest('Windows symlink creation requires a capability absent from this test process')
            raise
        value = inspect_setup(self.model())
        self.assertEqual(value['status'], 'needs_review')
        self.assertNotIn('config.json', value['metadata'])
        self.assertEqual(value['inventory']['links_skipped'], 1)

    def test_graph_and_native_missing_files(self):
        for backend in ('onnx', 'openvino'):
            value = inspect_setup(ModelSpec(path=str(self.root), task='tensor', backend=backend, device='cpu'))
            self.assertEqual(value['status'], 'blocked')
            self.assertIn('graph_contract_unverified', self.codes(value))
        value = inspect_setup(ModelSpec(path=str(self.root), task='vision', backend='llama-cpp', device='cpu', native={'model_file':'first.gguf','model_files':['first.gguf','second.gguf'],'mmproj':'missing-projector.gguf'}))
        self.assertTrue(any(row['files']==['missing-projector.gguf'] for row in value['issues']))
        self.assertTrue(any(row['files']==['second.gguf'] for row in value['issues']))

    def test_openvino_onnx_and_weightless_ir_are_not_rejected(self):
        for name in ('model.onnx','weightless.xml'):
            self.put(name,b'inert graph; no parse or execution')
            value = inspect_setup(ModelSpec(path=str(self.root),task='tensor',backend='openvino',device='cpu',graph=name))
            self.assertEqual(value['status'],'needs_review')
            self.assertNotIn('missing_file',self.codes(value))

    def test_malformed_declared_classes_return_diagnostics(self):
        for kind in ([],{}):
            self.put('model_index.json',{'_class_name':kind})
            for backend in ('onnx-diffusers','openvino-genai'):
                value = describe(ModelSpec(path=str(self.root),task='image',backend=backend,device='cpu'))
                self.assertIn('unsupported_pipeline',self.codes(value['setup']))
        from titanos.music_backend import ACE_CODE_REVISION
        self.put('titanos-music.json',{'schema_version':1,'family':'ace-step-1.5','variant':[],
            'code_revision':ACE_CODE_REVISION,'model_revision':'x','components':{},'capabilities':{},'license':'test'})
        value = describe(ModelSpec(path=str(self.root),task='music',backend='ace-step',device='cpu'))
        self.assertFalse(value['music']['implemented'])
        self.assertEqual(value['setup']['status'],'blocked')

    def test_speech_and_diarization_require_distinct_bundles(self):
        speech = inspect_setup(ModelSpec(path=str(self.root), task='speech', backend='sherpa-onnx', device='cpu'))
        self.assertIn('missing_speech_files', self.codes(speech))
        diar = inspect_setup(ModelSpec(path=str(self.root), task='diarization', backend='sherpa-onnx', device='cpu'))
        self.assertTrue(any(row['files']==['diarization.json'] for row in diar['issues']))

    def test_vad_and_qwen_use_the_actual_adapter_paths(self):
        self.put('silero_vad.onnx', b'inert graph presence only')
        value = inspect_setup(ModelSpec(path=str(self.root), task='vad', backend='onnx', device='cpu'))
        self.assertNotIn('missing_file',self.codes(value))
        self.put('config.json', {'tts_model_type':'base'})
        self.put('model.safetensors', b'inert weights')
        spec = ModelSpec(path=str(self.root), task='speech', backend='qwen3-tts', device='cpu')
        value = inspect_setup(spec)
        self.assertTrue(any(row['files']==['speech_tokenizer/model.safetensors'] for row in value['issues']))
        self.assertNotIn('missing_tokenizer',self.codes(value))

    def test_transformers_model_subfolder_does_not_move_tokenizer(self):
        self.put('variant/config.json', {'model_type':'future'})
        self.put('variant/model.safetensors', b'inert weights')
        self.put('tokenizer.json', b'inert tokenizer')
        value = inspect_setup(self.model(load_options={'subfolder':'variant'}))
        self.assertEqual(value['status'],'needs_review')
        self.assertNotIn('missing_tokenizer',self.codes(value))

    def test_explicit_loader_configuration_remains_unverified(self):
        self.put('tokenizer.json',b'inert tokenizer')
        value = inspect_setup(self.model(load_options={'config':'operator-selected-config','state_dict':{}}))
        self.assertEqual(value['status'],'needs_review')
        self.assertIn('loader_options_unverified',self.codes(value))
        self.assertNotIn('missing_weights',self.codes(value))
        self.assertNotIn('missing_file',self.codes(value))

    def test_pipeline_missing_declared_components(self):
        self.put('model_index.json', {'_class_name':'FutureBuiltinPipeline','unet':['diffusers','UNet2DConditionModel'], 'tokenizer':['transformers','CLIPTokenizer'], 'scheduler':['diffusers','PNDMScheduler']})
        value = inspect_setup(ModelSpec(path=str(self.root), task='image', backend='diffusers', device='cpu'))
        self.assertNotIn('unsupported_pipeline', self.codes(value))
        self.assertTrue(any(row['files']==['unet/config.json'] for row in value['issues']))
        self.assertIn('pipeline_unverified', self.codes(value))

    def test_runtime_discovery_is_not_runtime_execution(self):
        self.complete()
        with patch('titanos.model_inspection.find_spec', return_value=None):
            value = inspect_setup(self.model())
        self.assertEqual(value['runtime']['status'], 'missing')
        self.assertIn('runtime_missing', self.codes(value))

    def test_no_framework_or_checkpoint_import_or_network(self):
        import builtins
        import socket
        original = builtins.__import__
        forbidden = {'torch','transformers','diffusers','onnxruntime','openvino','sherpa_onnx'}
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in forbidden:
                raise AssertionError('Passive inspection imported ' + name)
            return original(name, *args, **kwargs)
        self.complete()
        with patch('builtins.__import__', side_effect=guarded), patch.object(socket, 'socket', side_effect=AssertionError('No network')):
            describe(self.model())

    def test_cli_and_http_share_setup_without_loading(self):
        from titanos.cli import main
        from titanos.server import create_app
        from titanos.engine import Engine
        from fastapi.testclient import TestClient
        cfg = Config(models={'incomplete':self.model()})
        registry = self.root/'registry.json';write_config(registry, cfg)
        with patch.object(Engine, '_load', side_effect=AssertionError('No model load')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(['--config',str(registry),'capabilities','incomplete'])
            cli = json.loads(output.getvalue())['setup']
            with TestClient(create_app(cfg, output_dir=self.root/'output'), base_url='http://127.0.0.1') as client:
                response = client.post('/api/models/capabilities',json={'model':'incomplete'})
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json()['setup']['issues'],cli['issues'])
                self.assertEqual(client.get('/api/models/loaded').json()['data'],[])

    def test_mcp_and_sdk_preserve_setup(self):
        import httpx
        from fastapi.testclient import TestClient
        from titanos.client import Client
        from titanos.engine import Engine
        from titanos.server import create_app
        cfg = Config(models={'incomplete':self.model()})
        app = create_app(cfg, output_dir=self.root/'output')
        with patch.object(Engine, '_load', side_effect=AssertionError('No model load')):
            with TestClient(app, base_url='http://127.0.0.1') as transport:
                async def call():
                    result = await app.state.mcp.call_tool('model_capabilities', {'model':'incomplete'})
                    self.assertFalse(result.is_error)
                    value = result.structured_content
                    return value.get('result', value)['setup']
                mcp = asyncio.run(call())
                def handle(request):
                    response = transport.post(request.url.path,json=json.loads(request.content))
                    return httpx.Response(response.status_code,json=response.json())
                with Client(transport=httpx.MockTransport(handle)) as client:
                    sdk = client.capabilities('incomplete')['setup']
                self.assertEqual(mcp,sdk)
                self.assertEqual(mcp['status'],'blocked')

    def test_music_uses_only_bounded_cached_json(self):
        from titanos.music_backend import ACE_CODE_REVISION, ACE_MODELS
        self.put('titanos-music.json', {'schema_version':1,'family':'ace-step-1.5','variant':'xl-sft',
            'code_revision':ACE_CODE_REVISION,'model_revision':ACE_MODELS['xl-sft'],
            'components':{'core':{'path':'core.json','sha256':'a'*64,'revision':'test','license':'test'},
                          'dit':{'path':'dit.json','sha256':'b'*64,'revision':'test','license':'test'}},
            'capabilities':{'instrumental':True},'license':'test'})
        for name in ('core.json','dit.json'):
            self.put(name,{'revision':'test','files':[{'path':'weights.bin','bytes':1,'sha256':'c'*64}]})
        self.put('weights.bin',b'x')
        spec = ModelSpec(path=str(self.root),task='music',backend='ace-step',device='cpu')
        value = describe(spec)
        self.assertTrue(value['music']['controls']['seed'])
        from titanos.settings import _active_settings
        token = _active_settings.set(Settings(model_inspection_max_metadata_bytes=8))
        try:
            value = describe(spec)
        finally:
            _active_settings.reset(token)
        self.assertFalse(value['music']['implemented'])
        self.assertIn('metadata_limit',self.codes(value['setup']))

    def test_settings_bounds_and_environment(self):
        value = Settings.from_env({'TITANOS_MODEL_INSPECTION_MAX_FILES':'23','TITANOS_MODEL_INSPECTION_MAX_METADATA_BYTES':'700'})
        self.assertEqual((value.model_inspection_max_files,value.model_inspection_max_metadata_bytes),(23,700))
        with self.assertRaises(ValueError):Settings(model_inspection_max_files=0)
        with self.assertRaises(ValueError):Settings(model_inspection_max_metadata_bytes=-1)
        with self.assertRaises(ValueError):Settings(model_inspection_max_depth=0)
        self.put('nested/weights/model.onnx',b'inert graph')
        value = inspect_setup(ModelSpec(path=str(self.root),task='tensor',backend='onnx',graph='nested/weights/model.onnx'),Settings(model_inspection_max_depth=1))
        self.assertIn('inspection_depth',self.codes(value))
        self.assertEqual(value['limits']['max_depth'],1)
        self.assertNotIn('missing_file',self.codes(value))


class LinkPolicy(unittest.TestCase):
    """titanos.link_policy: OS-owned POSIX ancestor aliases (macOS /var, /tmp -> /private/...; merged-/usr Linux /bin) are
    followed, a link the user can change is not, and Windows follows none."""
    def test_root_owned_system_symlinks_are_aliases(self):
        from titanos.link_policy import os_alias, refused_link
        system=[p for p in ('/var','/tmp','/etc','/bin','/lib','/sbin') if os.path.islink(p) and os.lstat(p).st_uid==0]
        if os.name=='nt' or not system:
            self.skipTest('no root-owned system symlink on this host')
        for alias in system:
            self.assertTrue(os_alias(alias,os.lstat(alias)),alias)
            self.assertFalse(refused_link(alias,ancestor=True),alias)
            self.assertTrue(refused_link(alias,ancestor=False),alias)

    def test_user_links_and_windows_links_stay_refused(self):
        from titanos.link_policy import os_alias, refused_link
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'target';target.mkdir();link=Path(folder)/'link'
            try:link.symlink_to(target,target_is_directory=True)
            except OSError:self.skipTest('symlinks are not permitted for this account')
            # running as root (containers, WSL), the test's own link in its 0700 directory is one only root can change
            by_root=os.name!='nt' and os.geteuid()==0
            self.assertEqual(os_alias(link,os.lstat(link)),by_root)
            self.assertEqual(refused_link(link,ancestor=True),not by_root)
            self.assertTrue(refused_link(link,ancestor=False))
            self.assertFalse(refused_link(target,ancestor=True))
            self.assertFalse(os_alias(target,os.lstat(target)))  # a real directory is never an alias
    # The Checks tests above run in tempfile directories, which on macOS sit under /var -> /private/var, so they cover
    # inspection through an OS alias ancestor on that platform.


if __name__ == '__main__':
    unittest.main(verbosity=2)
