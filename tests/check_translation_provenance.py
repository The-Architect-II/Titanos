"""Local attribution/control tests with doubles; no trained model or decoder runs."""
import asyncio,base64,contextlib,copy,hashlib,io,json,sys,tempfile,threading,time,types,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from titanos.api_plus import TranslationRequest, AudioTranslationRequest
from titanos.config import ModelSpec
from titanos.engine import EngineError
from titanos.extra_backends import ExtraBackend
from titanos.sentiment import checkpoint_fingerprint
from titanos.translation_contract import AudioTranslationResult, TextTranslationResult, text_digest
import check_asr_gating as audio_fixture


class Tensor:
    def __init__(self, count):self.shape=(1,count);self.device='cpu'
    def is_floating_point(self):return False
    def to(self,*args,**kwargs):return self


class Model:
    def __init__(self):self.config=NS(model_type='marian');self.calls=[];self.device='cpu';self.after=lambda:None
    def to(self,device):return self
    def eval(self):return self
    def parameters(self):return iter([NS(device=self.device)])
    def buffers(self):return iter([])
    def generate(self,**kwargs):self.calls.append(kwargs);self.after();return Tensor(4)


class Tokenizer:
    source_lang='fr';target_lang='en'
    def __init__(self):self.count=3;self.after=lambda:None;self.calls=[];self.decoded=0
    def get_vocab(self):return {'ordinary':9,'fr_XX':10,'en_XX':11}
    def __call__(self,text,**kwargs):self.calls.append((text,kwargs));self.after();return {'input_ids':Tensor(self.count)}
    def batch_decode(self,output,**kwargs):self.decoded+=1;return ['Hello.']


class Backend(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'config.json').write_text('{"model_type":"marian"}')
        (self.root/'tokenizer_config.json').write_text('{"source_lang":"fr","target_lang":"en"}')
        (self.root/'model.safetensors').write_bytes(b'fake checkpoint bytes; no numerical load')
        self.model=Model();self.tokenizer=Tokenizer();self.info={};self.on_load=lambda:None;self.load_calls=[]
        self.spec=ModelSpec(path=str(self.root),task='translation',device='cpu',source_repo='author/fixed-model',source_revision='pinned',
            defaults={'source_language':'fr','target_language':'en','max_tokens':12,'max_input_tokens':32})
        module=types.ModuleType('transformers')
        for name in ['AutoImageProcessor','AutoModelForObjectDetection','AutoModelForVision2Seq','AutoProcessor']:setattr(module,name,NS())
        def load(path,**kwargs):
            self.load_calls.append((path,kwargs));self.on_load()
            return (self.model,self.info) if kwargs.get('output_loading_info') else self.model
        module.AutoModelForSeq2SeqLM=NS(from_pretrained=load)
        module.AutoTokenizer=NS(from_pretrained=lambda *args,**kwargs:self.tokenizer)
        module.StoppingCriteria=object;module.StoppingCriteriaList=list
        self.patch=patch.dict(sys.modules,{'transformers':module});self.patch.start()
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def load(self):return ExtraBackend(self.spec,'cpu',{'local_files_only':True,'use_safetensors':True})
    def request(self,**updates):return {'text':'Bonjour.','source_language':'fr','target_language':'en','include_provenance':True,**updates}
    def test_actual_file_device_and_tokenizer_binding(self):
        backend=self.load();expected=checkpoint_fingerprint(self.root)
        self.spec.source_repo='changed-registration'
        value=backend.infer('translation',self.request())
        self.assertEqual(value['provenance']['repository'],'author/fixed-model')
        self.assertEqual(value['provenance']['checkpoint_digest'],expected)
        self.assertEqual(value['provenance']['device'],'cpu')
        self.assertEqual(value['language_selection']['mode'],'fixed_pair')
        self.assertEqual(value['language_selection']['tokenizer_source_language'],'fr')
        self.assertEqual(value['measurements']['input_tokens'],3)
        self.assertEqual(value['measurements']['output_sequence_tokens'],4)
        self.assertEqual(value['measurements']['processed_input_sha256'],text_digest('Bonjour.'))
        self.assertIs(self.tokenizer.calls[-1][1]['truncation'],False)
        self.assertNotIn('include_provenance',self.model.calls[0])
        self.assertTrue(self.load_calls[0][1]['use_safetensors'])
    def test_change_during_load_and_incompatible_weights_fail(self):
        self.on_load=lambda:(self.root/'model.safetensors').write_bytes(b'changed while loading')
        with self.assertRaisesRegex(ValueError,'changed while'):self.load()
        self.on_load=lambda:None;self.info={'missing_keys':['encoder.weight']}
        with self.assertRaisesRegex(ValueError,'missing or incompatible'):self.load()
    def test_observed_device_must_match(self):
        self.model.device='cuda:1'
        with self.assertRaisesRegex(ValueError,'requested device'):self.load()
    def test_fingerprint_unavailable_preserves_ordinary_loader_and_rejects_attribution(self):
        with patch('titanos.extra_backends.checkpoint_fingerprint',side_effect=ValueError('Checkpoint files must remain inside their model directory')):
            backend=self.load()
        self.assertNotIn('output_loading_info',self.load_calls[0][1])
        result=backend.infer('translation',self.request(include_provenance=False))
        self.assertEqual(result['text'],'Hello.');self.assertIsNone(result['provenance'])
        before=len(self.model.calls)
        with self.assertRaises(EngineError) as error:backend.infer('translation',self.request())
        self.assertEqual(error.exception.code,'unsupported_capability');self.assertIn('remain inside',str(error.exception))
        self.assertEqual(len(self.model.calls),before)
    def test_fixed_direction_and_equal_language_no_forward(self):
        backend=self.load()
        value=backend.infer('translation',self.request(target_language='fr'))
        self.assertFalse(value['translated']);self.assertEqual(value['text'],'Bonjour.');self.assertFalse(self.model.calls)
        with self.assertRaisesRegex(ValueError,'matching source_language'):backend.infer('translation',self.request(target_language='de'))
        self.spec.defaults.update(source_language='fra_Latn',target_language='eng_Latn')
        with self.assertRaisesRegex(ValueError,'loaded tokenizer pair'):backend.infer('translation',self.request(source_language='fra_Latn',target_language='eng_Latn'))
        self.assertFalse(self.model.calls)
    def test_multilingual_ids_and_full_input_limit(self):
        self.tokenizer.src_lang='fr_XX';self.tokenizer.lang_code_to_id={'fr_XX':10,'en_XX':11}
        backend=self.load();data=self.request(source_language='fr_XX',target_language='en_XX')
        value=backend.infer('translation',data)
        self.assertEqual(self.model.calls[-1]['forced_bos_token_id'],11)
        self.assertEqual(value['language_selection']['mode'],'multilingual')
        with self.assertRaisesRegex(ValueError,'Unsupported'):backend.infer('translation',{**data,'target_language':'ordinary'})
        self.tokenizer.count=33;before=len(self.model.calls)
        with self.assertRaisesRegex(ValueError,'max_input_tokens'):backend.infer('translation',data)
        self.assertEqual(len(self.model.calls),before)
    def test_nllb_language_tokens_do_not_accept_arbitrary_special_tokens(self):
        self.tokenizer.src_lang='fra_Latn';self.tokenizer.additional_special_tokens=['fra_Latn','eng_Latn','<mask>']
        self.tokenizer.get_vocab=lambda:{'fra_Latn':10,'eng_Latn':11,'<mask>':12}
        backend=self.load()
        value=backend.infer('translation',self.request(source_language='fra_Latn',target_language='eng_Latn'))
        self.assertEqual(value['language_selection']['forced_bos_token_id'],11)
        with self.assertRaisesRegex(ValueError,'Unsupported'):backend.infer('translation',self.request(source_language='fra_Latn',target_language='<mask>'))
    def test_cancellation_before_and_after_forward(self):
        backend=self.load();cancel=threading.Event();self.tokenizer.after=cancel.set
        with self.assertRaises(EngineError) as error:backend.infer('translation',self.request(),cancel)
        self.assertEqual(error.exception.status,499);self.assertFalse(self.model.calls)
        cancel.clear();self.tokenizer.after=lambda:None;self.model.after=cancel.set
        with self.assertRaises(EngineError):backend.infer('translation',self.request(),cancel)
        self.assertEqual(len(self.model.calls),1);self.assertEqual(self.tokenizer.decoded,0)


def backend_output(text):
    output='The translated text.'
    return {'text':output,'translated':True,'input_sha256':text_digest(text),'output_sha256':text_digest(output),
        'provenance':{'repository':None,'revision':None,'checkpoint_digest':'a'*64,'backend':'transformers','device':'cpu',
                      'model_class':'FakeSeq2Seq','tokenizer_class':'FakeTokenizer','model_type':'marian'},
        'language_selection':{'mode':'fixed_pair','source_language':'fra_Latn','target_language':'en',
            'tokenizer_source_language':None,'tokenizer_target_language':None},
        'measurements':{'processed_input_sha256':text_digest(text),'input_tokens':3,'input_token_limit':32,'output_sequence_tokens':5,'inference_seconds':.02}}


class API(unittest.TestCase):
    def setUp(self):
        audio_fixture.Gates.setUp(self);self.base_infer=self.app.state.engine.infer.side_effect;self.patch.stop()
        self.cfg.models['nmt'].defaults.update(source_language='fra_Latn', target_language='en')
        self.bad=None;self.nmt_error=None;self.wait_for_cancel=False;self.entered=threading.Event();self.stopped=threading.Event();self.after_asr=lambda:None
        def infer(name,task,data,emit=None,cancel=None):
            if task!='translation':
                value=self.base_infer(name,task,data,emit,cancel)
                if task=='transcription':self.after_asr()
                return value
            self.app.state.engine.target.validate(name,task);self.calls.append((name,task,copy.deepcopy(data)))
            if self.nmt_error:raise self.nmt_error
            if self.wait_for_cancel:
                self.entered.set();cancel.wait(2);self.stopped.set()
            value=backend_output(data['text'])
            value['language_selection'].update(source_language=data['source_language'],target_language=data['target_language'])
            value['language_selection'].update(registered_source_language=self.cfg.models[name].defaults.get('source_language'),
                                               registered_target_language=self.cfg.models[name].defaults.get('target_language'))
            if self.bad:self.bad(value)
            return value
        self.patch=patch.object(self.app.state.engine,'infer',side_effect=infer);self.patch.start()
    def tearDown(self):audio_fixture.Gates.tearDown(self)
    def text(self,**options):return self.http.post('/v1/translations',json={'model':'nmt','text':'Bonjour.','source_language':'fra_Latn','target_language':'en',**options})
    def audio(self,**options):return self.http.post('/v1/audio/translations',data={'model':'asr','translation_model':'nmt','source_language':'fra_Latn','target_language':'en','language':'fr',**options},files={'file':('audio.wav',b'original-audio','audio/wav')})
    def test_legacy_text_and_opt_in_schema(self):
        expected={'model':'nmt','text':'The translated text.','translations':[{'text':'The translated text.'}],
                  'source_language':'fra_Latn','target_language':'en','translated':True}
        self.assertEqual(self.text().json(),expected)
        result=TextTranslationResult.model_validate(self.text(include_provenance=True).json())
        self.assertEqual(result.stages[0].input_text,'Bonjour.')
        self.assertEqual(result.stages[0].measurements.input_tokens,3)
        self.assertEqual(result.schema_version,'1')
        self.assertEqual(result.source_language_origin,'request')
    def test_source_id_omission_uses_registered_default_without_language_detection(self):
        self.cfg.models['nmt'].defaults['source_language']='fra_Latn'
        response=self.http.post('/v1/translations',json={'model':'nmt','text':'Bonjour.','target_language':'en','include_provenance':True})
        value=TextTranslationResult.model_validate(response.json())
        self.assertEqual(value.source_language_origin,'model_default');self.assertIsNone(value.requested_source_language)
        self.assertEqual(value.source_language,'fra_Latn')
        self.assertEqual(value.stages[0].language_selection.registered_source_language,'fra_Latn')
        self.assertFalse(value.stages[0].language_selection.language_detection_performed)
    def test_equal_language_does_not_load_or_infer(self):
        response=self.text(include_provenance=True,target_language='fra_Latn')
        result=TextTranslationResult.model_validate(response.json());self.assertFalse(self.calls)
        self.assertFalse(result.translated);self.assertEqual(result.text,'Bonjour.')
        self.assertIsNone(result.stages[0].provenance);self.assertEqual(result.stages[0].status,'passthrough')
    def test_missing_malformed_and_wrong_backend_metadata_fail_closed(self):
        mutations=[lambda v:v.pop('provenance'),lambda v:v['provenance'].update(device=''),
            lambda v:v.update(input_sha256='b'*64),lambda v:v['language_selection'].update(target_language='de'),
            lambda v:v['measurements'].update(input_tokens=True),lambda v:v['measurements'].update(inference_seconds=float('nan'))]
        for mutate in mutations:
            self.bad=mutate;response=self.text(include_provenance=True)
            self.assertEqual(response.status_code,502,response.text)
            self.assertEqual(response.json()['error']['code'],'invalid_backend_response')
    def test_audio_skip_has_no_invented_asr_or_nmt_execution(self):
        result=AudioTranslationResult.model_validate(self.audio(include_provenance='true').json())
        self.assertEqual([task for _,task,_ in self.calls],['vad'])
        self.assertEqual([stage.status for stage in result.stages],['skipped','skipped'])
        self.assertTrue(all(stage.provenance is None and stage.measurements is None for stage in result.stages))
        self.assertIsNone(result.stages[0].asr_metadata.backend);self.assertFalse(result.translation_executed)
    def test_audio_exact_text_and_language_roles_legacy_shapes(self):
        self.speech=True
        legacy=self.audio().json()
        self.assertEqual(legacy,{'text':'The translated text.','model':'asr','translation_model':'nmt','language':'en','task':'audio_translation'})
        result=AudioTranslationResult.model_validate(self.audio(include_provenance='true').json())
        self.assertEqual(result.stages[1].input_text,result.stages[0].output_text)
        self.assertEqual(result.stages[1].input_text,'actual ASR text')
        self.assertEqual(result.requested_asr_language,'fr');self.assertIsNone(result.observed_asr_language)
        self.assertEqual(result.nmt_source_language,'fra_Latn');self.assertTrue(result.translation_executed)
        value=self.audio(response_format='verbose_json').json();self.assertNotIn('schema_version',value)
        self.assertIn('source_language',value);self.assertIsNone(value['source_language'])
        self.assertEqual(self.audio(response_format='text').text,'The translated text.')
        self.assertEqual(self.audio(response_format='text',include_provenance='true').status_code,400)
    def test_direct_audio_translation_has_no_nmt_identity_and_cancellation_between_stages(self):
        self.speech=True
        payload=AudioTranslationRequest(model='asr',audio_base64=base64.b64encode(b'original').decode(),language='fr',include_provenance=True)
        value=self.app.state.extension_handlers['audio_translation'](payload,None,time.monotonic()+5)
        typed=AudioTranslationResult.model_validate(value)
        self.assertEqual(len(typed.stages),1);self.assertIsNone(typed.nmt_source_language)
        self.assertEqual(typed.stages[0].asr_metadata.task,'translate')
        self.calls.clear();cancel=threading.Event();self.after_asr=cancel.set
        payload=payload.model_copy(update={'translation_model':'nmt','source_language':'fra_Latn','target_language':'en','vad_enabled':False})
        with self.assertRaises(EngineError) as error:self.app.state.extension_handlers['audio_translation'](payload,cancel,time.monotonic()+5)
        self.assertEqual(error.exception.status,499);self.assertEqual([task for _,task,_ in self.calls],['transcription'])
        self.calls.clear();self.after_asr=lambda:None;cancel.clear()
        with self.assertRaises(EngineError) as error:self.app.state.extension_handlers['translation'](TranslationRequest(model='nmt',text='x',target_language='en',include_provenance=True),cancel,time.monotonic()-1)
        self.assertEqual(error.exception.status,504);self.assertFalse(self.calls)
    def test_errors_and_shared_cancellation(self):
        self.nmt_error=EngineError('preserved backend failure',503,'nmt_unavailable')
        response=self.text(include_provenance=True);self.assertEqual(response.status_code,503);self.assertIn('preserved backend failure',response.text)
        self.nmt_error=None;cancel=threading.Event();cancel.set();before=len(self.calls)
        payload=TranslationRequest(model='nmt',text='x',source_language='en',target_language='en',include_provenance=True)
        with self.assertRaises(EngineError) as error:self.app.state.extension_handlers['translation'](payload,cancel,time.monotonic()+5)
        self.assertEqual(error.exception.status,499);self.assertEqual(len(self.calls),before)
        self.wait_for_cancel=True
        async def invoke():
            task=asyncio.create_task(self.app.state.mcp.call_tool('translate',{'payload':{'model':'nmt','text':'x','target_language':'en','include_provenance':True}}))
            self.assertTrue(await asyncio.to_thread(self.entered.wait,1));task.cancel()
            with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,2)
            self.assertTrue(self.stopped.is_set())
        asyncio.run(invoke())
    def test_python_cli_mcp_and_discovery_controls(self):
        from titanos.client import Client
        from titanos import cli
        self.speech=True
        text=self.text(include_provenance=True).json();audio=self.audio(include_provenance='true').json()
        with patch.object(Client,'_json',return_value=text) as called:
            with Client() as client:value=client.translate('nmt','Bonjour.','en',source_language='fra_Latn',include_provenance=True)
            self.assertIsInstance(value,TextTranslationResult);self.assertTrue(called.call_args.kwargs['json']['include_provenance'])
        with patch.object(Client,'_audio_upload',return_value=audio) as called:
            with Client() as client:value=client.translate_audio('asr',self.root/'file.wav',include_provenance=True,source_language='fra_Latn',target_language='en')
            self.assertIsInstance(value,AudioTranslationResult);self.assertEqual(called.call_args.args[3]['target_language'],'en')
        with patch.object(cli,'remote_command') as called:
            cli.main(['translate-audio','asr',str(self.root/'input.wav'),'--translation-model','nmt','--source-language','fr','--target-language','en','--include-provenance'])
            args=called.call_args.args[0];self.assertEqual(args.source_language,'fr');self.assertEqual(args.target_language,'en');self.assertTrue(args.include_provenance)
        output=io.StringIO()
        with patch.object(Client,'translate',return_value=TextTranslationResult.model_validate(text)),contextlib.redirect_stdout(output):
            cli.main(['translate','nmt','Bonjour.','--source-language','fra_Latn','--target-language','en','--include-provenance'])
        self.assertEqual(json.loads(output.getvalue())['object'],'text.translation')
        async def invoke():
            for name,payload,expected in [('translate',{'model':'nmt','text':'Bonjour.','source_language':'fra_Latn','target_language':'en','include_provenance':True},'text.translation'),
                ('translate_audio',{'model':'asr','translation_model':'nmt','audio_base64':base64.b64encode(b'original').decode(),'source_language':'fra_Latn','target_language':'en','include_provenance':True},'audio.translation')]:
                result=await self.app.state.mcp.call_tool(name,{'payload':payload});self.assertFalse(result.is_error)
                content=result.structured_content;content=content.get('result',content)
                self.assertEqual(content['object'],expected)
        asyncio.run(invoke())
        schema=self.http.get('/v1/extensions').json()['translation_provenance']
        self.assertFalse(schema['default']);self.assertEqual(schema['request_option'],'include_provenance')
        self.assertIn('stages',schema['audio_result_schema']['properties'])


if __name__=='__main__':unittest.main(verbosity=2)
