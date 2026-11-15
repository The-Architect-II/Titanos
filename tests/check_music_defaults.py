"""Direct music callers retain request > model > process precedence before spawn."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from titanos.config import ModelSpec
from titanos.music_backend import AceStepModel,ACE_CODE_REVISION,ACE_MODELS
from titanos.settings import Settings
from titanos.capabilities import describe


class Captured(Exception):pass


class Checks(unittest.TestCase):
    def capture(self,defaults,data,settings):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=object.__new__(AceStepModel)
            model.spec=ModelSpec(path=str(root),task='music',backend='ace-step',device='cpu',defaults=defaults)
            model.executable=Path(sys.executable);model.adapter=root/'never-executed.py'
            model.bundle={'root':root,'manifest':{'variant':'xl-sft','code_revision':ACE_CODE_REVISION,
                'model_revision':ACE_MODELS['xl-sft'],'capabilities':{'instrumental':True,'song':True,'lyrics':True,
                    'vocal_languages':['en'],'duration_min':10,'duration_max':600}}}
            captured={}
            def stop_before_spawn(argv,*args,**kwargs):
                captured.update(json.loads(Path(argv[argv.index('--request')+1]).read_text(encoding='utf-8')))
                raise Captured
            with patch('titanos.music_backend.current_settings',return_value=settings),patch('titanos.music_backend.run_owned',side_effect=stop_before_spawn):
                with self.assertRaises(Captured):model.infer('music',{'prompt':'Original quiet music',**data})
            return captured

    def test_control_precedence_and_null_omission(self):
        settings=Settings(default_steps=11,max_steps=50)
        self.assertEqual(self.capture({}, {},settings)['steps'],11)
        self.assertEqual(self.capture({'steps':7},{},settings)['steps'],7)
        self.assertEqual(self.capture({'steps':7},{'steps':None},settings)['steps'],7)
        request=self.capture({'steps':7,'duration':40,'mode':'song','vocal_language':'en'},{'steps':5,'lyrics':'Original words'},settings)
        self.assertEqual((request['steps'],request['duration'],request['mode'],request['vocal_language']),(5,40,'song','en'))

    def test_lowered_process_default_is_sent_explicitly(self):
        request=self.capture({}, {},Settings(default_steps=3,max_steps=3))
        self.assertEqual(request['steps'],3)
        self.assertEqual(request['runtime_contract']['max_steps'],3)

    def test_planner_route_flags_reach_the_child_only_when_set(self):
        settings=Settings(default_steps=11,max_steps=50);flags=('thinking','use_cot_metas','use_cot_caption','use_cot_language')
        self.assertFalse(set(self.capture({},{},settings))&set(flags))
        self.assertFalse(set(self.capture({},{name:None for name in flags},settings))&set(flags))
        request=self.capture({'thinking':False,'use_cot_metas':True},{},settings)
        self.assertEqual({name:request[name] for name in flags if name in request},{'thinking':False,'use_cot_metas':True})
        request=self.capture({'thinking':False,'use_cot_metas':True},{'thinking':True,'use_cot_caption':True},settings)
        self.assertEqual({name:request[name] for name in flags if name in request},{'thinking':True,'use_cot_metas':True,'use_cot_caption':True})

    def test_planner_flags_without_a_planner_route_are_refused_before_spawn(self):
        settings=Settings(default_steps=11,max_steps=50)
        for defaults,data in (({},{'use_cot_caption':True}),({},{'use_cot_language':True}),({},{'thinking':False}),
                              ({'use_cot_metas':True},{'use_cot_metas':False,'use_cot_caption':True}),({},{'thinking':'false','use_cot_metas':True})):
            with self.subTest(defaults=defaults,data=data),self.assertRaises(ValueError):self.capture(defaults,data,settings)

    def test_registry_defaults_accept_boolean_planner_flags_only(self):
        with tempfile.TemporaryDirectory() as directory:
            spec=ModelSpec(path=directory,task='music',backend='ace-step',device='cpu',defaults={'thinking':False,'use_cot_metas':True})
            self.assertEqual(spec.defaults,{'thinking':False,'use_cot_metas':True})
            for defaults in ({'thinking':'no'},{'use_cot_caption':1},{'planner':'on'}):
                with self.subTest(defaults=defaults),self.assertRaises(ValueError):
                    ModelSpec(path=directory,task='music',backend='ace-step',device='cpu',defaults=defaults)

    def test_capabilities_expose_real_sampling_controls_only_for_supported_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            spec=ModelSpec(path=str(root),task='music',backend='ace-step',device='cpu')
            for variant in ('xl-sft','xl-turbo'):
                manifest={'variant':variant,'capabilities':{'instrumental':True}}
                (root/'titanos-music.json').write_text(json.dumps(manifest),encoding='utf-8')
                with patch('titanos.music_backend.inspect_bundle',return_value={'manifest':manifest}):
                    music=describe(spec)['music']
                    if variant=='xl-sft':
                        self.assertTrue(all(music['controls'][name] for name in ('seed','guidance_scale','steps','thinking','use_cot_metas',
                                                                                 'use_cot_caption','use_cot_language')))
                    else:self.assertFalse(music['implemented'])


if __name__=='__main__':unittest.main(verbosity=2)
