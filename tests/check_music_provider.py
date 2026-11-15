"""Pinned ACE-Step component selection, completeness and acquisition integrity."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import threading
import unittest
from unittest.mock import patch

from titanos.providers import AcquisitionRequest,AceStepProvider,ProviderRegistry,import_model_link,runtime_recommendation
from titanos.music_backend import inspect_bundle


def metadata():
    roles={
        'core': ('ACE-Step/Ace-Step1.5', AceStepProvider.core[1], [
            'README.md','vae/config.json','vae/diffusion_pytorch_model.safetensors',
            'Qwen3-Embedding-0.6B/config.json','Qwen3-Embedding-0.6B/model.safetensors',
            'Qwen3-Embedding-0.6B/tokenizer.json','Qwen3-Embedding-0.6B/tokenizer_config.json',
            'Qwen3-Embedding-0.6B/chat_template.jinja',
            'acestep-v15-turbo/model.safetensors','acestep-5Hz-lm-1.7B/model.safetensors',
            'Qwen3-Embedding-0.6B/tokenizer.py','Qwen3-Embedding-0.6B/tokenizer.json.py',
            'Qwen3-Embedding-0.6B/../tokenizer.json','README.py']),
        'dit': ('ACE-Step/acestep-v15-xl-sft', AceStepProvider.variants['xl-sft'][1], [
            'README.md','config.json','model.safetensors','silence_latent.pt','model.pt','modeling.py']),
        'planner': (AceStepProvider.planner[0], AceStepProvider.planner[1], [
            'README.md','config.json','model.safetensors','tokenizer.json','tokenizer_config.json']),
    }
    return {role:NS(id=repo,sha=pin,siblings=[NS(rfilename=name,size=10) for name in files])
            for role,(repo,pin,files) in roles.items()}


class Checks(unittest.TestCase):
    def plan(self,rows=None):
        rows=metadata() if rows is None else rows
        with patch('titanos.providers.HfApi') as api:
            api.return_value.model_info.side_effect=lambda repo,**kw:next(row for row in rows.values() if row.id==repo)
            return AceStepProvider().plan(AcquisitionRequest(provider='ace-step',identifier='xl-sft',
                registry_name='music',task='music',backend='ace-step',device='cpu'))

    def test_plan_and_acquire_complete_pinned_bundle(self):
        provider=AceStepProvider();plan=self.plan()
        names={row.path for row in plan.files}
        self.assertIn('planner:'+provider.planner[1],plan.resolved_revision)
        self.assertIn('dit/silence_latent.pt',names)
        self.assertIn('core/Qwen3-Embedding-0.6B/chat_template.jinja',names)
        self.assertFalse(any('1.7B' in name or 'v15-turbo' in name or name.endswith('.py') for name in names))
        original=copy.deepcopy(plan.provider_data)
        with tempfile.TemporaryDirectory() as folder:
            def download(repo_id,revision,allow_patterns,local_dir,**kwargs):
                for name in allow_patterns:
                    path=Path(local_dir)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x'*10)
                return str(local_dir)
            with patch('titanos.providers.snapshot_download',side_effect=download):
                result=provider.acquire(plan,Path(folder),lambda *args:None,threading.Event())
            root=Path(result['path']);manifest=json.loads((root/'titanos-music.json').read_text())
            self.assertEqual(set(manifest['components']),{'core','dit','planner'})
            self.assertEqual(result['license'],'MIT AND Apache-2.0')
            self.assertIn('ja',manifest['capabilities']['vocal_languages'])
            self.assertIn('zh',manifest['capabilities']['vocal_languages'])
            self.assertFalse(manifest['capabilities']['reference_audio'])
            self.assertEqual(plan.provider_data,original)
            inspect_bundle(root)
            (root/'dit/silence_latent.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed|checksum'):inspect_bundle(root)

    def test_missing_required_conditioning_or_component_rejected(self):
        for role,name in [('dit','silence_latent.pt'),('core','vae/config.json'),
                          ('core','Qwen3-Embedding-0.6B/model.safetensors'),('planner','tokenizer.json')]:
            with self.subTest(role=role,name=name):
                rows=metadata();rows[role].siblings=[row for row in rows[role].siblings if row.rfilename!=name]
                with self.assertRaisesRegex(ValueError,'incomplete'):self.plan(rows)

    def test_safe_file_filter_does_not_admit_checkpoint_code(self):
        for name in ['tokenizer.py','tokenizer.json.py','README.py','silence_latent.pkl',
                     '../config.json','a/../config.json','/config.json','C:/config.json']:
            for role in ['core','dit','planner']:
                self.assertFalse(AceStepProvider._allowed(name,role),(role,name))
        self.assertTrue(AceStepProvider._allowed('silence_latent.pt','dit'))
        self.assertFalse(AceStepProvider._allowed('silence_latent.pt','planner'))

    def test_registry_lists_real_music_provider(self):
        self.assertIn('ace-step',{row['id'] for row in ProviderRegistry().descriptors()})
        imported=import_model_link('https://huggingface.co/ACE-Step/acestep-v15-xl-sft/tree/d06de46b4622f781cf07f4a013a67d591ca52819')
        self.assertEqual((imported.provider,imported.identifier),('ace-step','xl-sft'))

    def test_unimplemented_variant_is_not_a_runnable_runtime_claim(self):
        request=AcquisitionRequest(provider='ace-step',identifier='xl-turbo',registry_name='music',
                                   task='music',backend='ace-step',device='cpu')
        with patch('titanos.music_backend.runtime_receipt',side_effect=AssertionError('No runtime scan needed')):
            value=runtime_recommendation(request)
        self.assertEqual(value['status'],'unsupported_adapter')
        self.assertEqual(value['supported_variants'],['xl-sft'])


if __name__=='__main__':unittest.main(verbosity=2)
