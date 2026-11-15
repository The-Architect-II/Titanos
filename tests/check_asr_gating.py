"""Controlled configured VAD-before-ASR semantics; no numerical framework/model."""
import asyncio,base64,hashlib,io,json,tempfile,threading,time
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from titanos.api_plus import ASRRequest
from titanos.audio_gating import resolve_gate
from titanos.config import Config,ModelSpec
from titanos.engine import EngineError
from titanos.server import create_app
from titanos.settings import Settings

class Gates(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.cfg=Config(models={'asr':ModelSpec(path=str(self.root),task='transcription',defaults={'vad_model':'silero'}),
                               'silero':ModelSpec(path=str(self.root),task='vad',backend='onnx'),
                               'nmt':ModelSpec(path=str(self.root),task='translation',defaults={'source_language':'fra_Latn','target_language':'en'})})
        self.calls=[];self.speech=False;self.after=lambda:None;self.failure=None;self.bad_identity=False
        self.app=create_app(self.cfg,output_dir=self.root/'out')
        def infer(name,task,data,emit=None,cancel=None):
            self.app.state.engine.target.validate(name,task);self.calls.append((name,task,data))
            if task=='vad':
                if self.failure:raise self.failure
                self.after()
                return {'model':name,'task':'vad','sample_rate':16000,'duration':1.,'decoded_samples':16000,'frame_samples':512,'context_samples':64,
                    'state_reset':'per_request','score_kind':'speech_probability_uncalibrated','frames':None,
                    'segments':[{'start':.2,'end':.8,'start_sample':3200,'end_sample':12800}] if self.speech else [],
                    'parameters':{'threshold':data.get('threshold') or .5},'input_sha256':'b'*64 if self.bad_identity else hashlib.sha256(data['audio']).hexdigest(),
                    'provenance':{'backend':'onnx','device':'cpu','checkpoint_digest':'a'*64}}
            if task=='translation':return {'text':'translated','translated':True}
            if data['response_format']!='verbose_json':return 'actual ASR text'
            return {'task':'translate' if data.get('translation') else 'transcribe','text':'actual ASR text','duration':1.,'sample_rate':16000,'decoded_samples':16000,
                    'language':None,'language_source':'unavailable','requested_language':data.get('language'),
                    'segments':[{'id':0,'start':0.,'end':1.,'text':'actual ASR text'}],'words':None,
                    'timestamps':{'source':'backend_alignment'},'quality':{'status':'unavailable','confidence':None},'backend':'transformers'}
        self.patch=patch.object(self.app.state.engine,'infer',side_effect=infer);self.patch.start()
        self.http=TestClient(self.app,base_url='http://127.0.0.1');self.http.__enter__()
    def tearDown(self):
        self.http.__exit__(None,None,None);self.patch.stop();self.temp.cleanup()
    def upload(self,**options):
        return self.http.post('/v1/audio/transcriptions',data={'model':'asr',**options},files={'file':('input.wav',b'unchanged-original','audio/wav')})

    def test_skip_preserves_plain_formats_and_truthful_verbose(self):
        response=self.upload();self.assertEqual(response.json(),{'text':''})
        self.assertEqual([t for _,t,_ in self.calls],['vad'])
        self.calls.clear();self.assertEqual(self.upload(response_format='text').text,'')
        self.assertEqual([t for _,t,_ in self.calls],['vad'])
        value=self.upload(response_format='verbose_json',language='en',**{'timestamp_granularities[]':'word'}).json()
        self.assertEqual(value['text'],'');self.assertIsNone(value['backend']);self.assertIsNone(value['language'])
        self.assertIsNone(value['segments']);self.assertIsNone(value['words']);self.assertIsNone(value['quality']['confidence'])
        self.assertEqual(value['duration_source'],'vad_decoded_audio');self.assertEqual(value['requested_language'],'en')
        gate=value['vad_gate'];self.assertFalse(gate['asr_executed']);self.assertEqual(gate['decision'],'no_speech_detected')
        self.assertEqual(gate['measurement']['provenance']['checkpoint_digest'],'a'*64)
        self.assertEqual(gate['measurement']['input_sha256'],hashlib.sha256(b'unchanged-original').hexdigest())

    def test_speech_passes_identical_original_bytes_and_distinct_workers(self):
        self.speech=True
        value=self.upload(response_format='verbose_json',worker='asr-worker',vad_options=json.dumps({'worker':'vad-worker','threshold':.6})).json()
        self.assertEqual([t for _,t,_ in self.calls],['vad','transcription'])
        self.assertIs(self.calls[0][2]['audio'],self.calls[1][2]['audio'])
        self.assertEqual(self.calls[1][2]['audio'],b'unchanged-original')
        self.assertEqual(self.calls[0][2]['worker'],'vad-worker');self.assertEqual(self.calls[1][2]['worker'],'asr-worker')
        self.assertEqual(self.calls[0][2]['threshold'],.6)
        self.assertEqual(value['segments'][0]['start'],0.)
        self.assertTrue(value['vad_gate']['asr_executed']);self.assertEqual(value['duration_source'],'asr_decoded_audio')

    def test_precedence_and_explicit_disable(self):
        process=Settings(asr_vad_model='process-vad')
        self.assertEqual(resolve_gate({}, {}, process)['model'],'process-vad')
        self.assertEqual(resolve_gate({}, {'vad_model':'model-vad'},process)['model'],'model-vad')
        value=resolve_gate({'vad_model':'request-vad','vad_options':{'threshold':.7}}, {'vad_model':'model-vad','vad_options':{'threshold':.6,'min_silence_ms':100}},process)
        self.assertEqual(value['model'],'request-vad');self.assertEqual(value['options'],{'threshold':.7,'min_silence_ms':100.})
        self.assertFalse(resolve_gate({'vad_enabled':False},{'vad_enabled':True},process)['enabled'])
        self.assertFalse(resolve_gate({}, {'vad_enabled':False},process)['enabled'])
        self.assertTrue(resolve_gate({'vad_enabled':True},{'vad_model':'x'},Settings(asr_vad_mode='off'))['enabled'])
        with self.assertRaises(ValueError):resolve_gate({'vad_enabled':True},{},Settings())
        with self.assertRaises(ValueError):resolve_gate({}, {}, Settings(asr_vad_mode='on'))
        self.assertFalse(resolve_gate({}, {}, Settings())['enabled'])
        value=self.upload(vad_enabled='false',response_format='verbose_json').json()
        self.assertEqual([t for _,t,_ in self.calls],['transcription'])
        self.assertNotIn('vad_gate',value);self.assertNotIn('duration_source',value)
        self.assertEqual(Settings.from_env({'TITANOS_ASR_VAD_MODEL':'silero'}).asr_vad_model,'silero')

    def test_mismatch_unknown_errors_and_no_fallback(self):
        for model,status in [('missing',404),('asr',400),('nmt',400)]:
            self.calls.clear();response=self.upload(vad_model=model)
            self.assertEqual(response.status_code,status,response.text);self.assertFalse(self.calls)
        self.failure=EngineError('retained gate error',503,'backend_unavailable')
        response=self.upload();self.assertEqual(response.status_code,503);self.assertIn('retained gate error',response.text)
        self.assertEqual([t for _,t,_ in self.calls],['vad'])
        for defaults in [{'vad_enabled':1},{'vad_model':False},{'vad_options':{'return_frames':True}}]:
            with self.assertRaises(ValueError):ModelSpec(path='local',task='transcription',defaults=defaults)
        self.failure=None;self.bad_identity=True;self.calls.clear()
        response=self.upload();self.assertEqual(response.status_code,502,response.text)
        self.assertEqual([t for _,t,_ in self.calls],['vad'])

    def test_shared_cancellation_and_absolute_deadline(self):
        self.speech=True;cancel=threading.Event();self.after=cancel.set
        payload=ASRRequest(model='asr',audio_base64=base64.b64encode(b'original').decode())
        with self.assertRaises(EngineError) as failure:self.app.state.extension_handlers['audio_transcription'](payload,cancel,time.monotonic()+5)
        self.assertEqual(failure.exception.status,499);self.assertEqual([t for _,t,_ in self.calls],['vad'])
        self.calls.clear();self.after=lambda:time.sleep(.03)
        response=self.upload(timeout='1');self.assertEqual(response.status_code,504,response.text)
        self.assertNotIn('transcription',[t for _,t,_ in self.calls])

    def test_translation_asr_stage_uses_gate_and_skips_nmt(self):
        response=self.http.post('/v1/audio/translations',data={'model':'asr','translation_model':'nmt','source_language':'fra_Latn','response_format':'verbose_json'},
            files={'file':('input.wav',b'original','audio/wav')})
        self.assertEqual(response.status_code,200,response.text);value=response.json()
        self.assertEqual(value['text'],'');self.assertFalse(value['translation_executed']);self.assertFalse(value['vad_gate']['asr_executed'])
        self.assertEqual([t for _,t,_ in self.calls],['vad'])
        self.calls.clear();self.speech=True
        response=self.http.post('/v1/extensions/infer',json={'operation':'audio_translation','model':'asr','translation_model':'nmt','source_language':'fra_Latn',
            'audio_base64':base64.b64encode(b'original').decode(),'response_format':'verbose_json'})
        self.assertEqual(response.status_code,200,response.text);value=response.json()
        self.assertEqual(value['text'],'translated');self.assertTrue(value['translation_executed'])
        self.assertEqual([t for _,t,_ in self.calls],['vad','transcription','translation'])
        self.assertEqual(self.calls[-1][2]['target_language'],'en')
        self.calls.clear()
        explicit=self.http.post('/v1/audio/translations',data={'model':'asr','translation_model':'nmt','source_language':'fra_Latn','target_language':'eng_Latn'},
            files={'file':('input.wav',b'original','audio/wav')})
        self.assertEqual(explicit.status_code,200,explicit.text);self.assertEqual(self.calls[-1][2]['target_language'],'eng_Latn')
        self.calls.clear()
        direct=self.http.post('/v1/audio/translations',data={'model':'asr','response_format':'verbose_json'},files={'file':('input.wav',b'original','audio/wav')}).json()
        self.assertTrue(direct['translation_executed']);self.assertEqual([t for _,t,_ in self.calls],['vad','transcription'])

    def test_gate_defaults_never_reach_pipeline_library(self):
        from titanos.asr import transcribe
        observed=[]
        class Pipeline:
            type='ctc'
            feature_extractor=NS(sampling_rate=16000)
            def __call__(self,data,**options):observed.append(options);return {'text':'text'}
        spec=ModelSpec(path='local',task='transcription',defaults={'vad_model':'silero','vad_enabled':True,'vad_options':{'threshold':.6},'chunk_length_s':20})
        with patch('titanos.asr.read_audio',return_value=[0.]):self.assertEqual(transcribe(Pipeline(),spec,{'audio':b'bytes'}),'text')
        self.assertEqual(observed,[{'chunk_length_s':20}])

    def test_python_cli_and_mcp_controls(self):
        from titanos.client import Client
        from titanos import cli
        from titanos.mcp_server import create_mcp
        path=self.root/'audio.wav';path.write_bytes(b'original')
        with patch.object(Client,'request',return_value=NS(text='',json=lambda:{'text':''})) as request:
            with Client() as client:client.transcribe('asr',path,vad_enabled=False,vad_options={'threshold':.6})
            data=request.call_args.kwargs['data'];self.assertIs(data['vad_enabled'],False);self.assertEqual(json.loads(data['vad_options']),{'threshold':.6})
        with patch.object(cli,'remote_command') as command:cli.main(['transcribe','asr',str(path),'--no-vad'])
        self.assertIs(command.call_args.args[0].vad_enabled,False)
        async def invoke():
            mcp=create_mcp(app=self.app)
            result=await mcp.call_tool('transcribe',{'model':'asr','audio_base64':base64.b64encode(b'original').decode(),'vad_enabled':False})
            return result
        self.calls.clear();asyncio.run(invoke());self.assertEqual([t for _,t,_ in self.calls],['transcription'])

if __name__=='__main__':unittest.main(verbosity=2)
