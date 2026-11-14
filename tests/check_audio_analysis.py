"""Audio composition through real HTTP/MCP/CLI/SDK contracts with inert providers only."""
import asyncio,base64,hashlib,io,json,sys,tempfile,threading,time,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import httpx
from fastapi.testclient import TestClient
from titanos.api_plus import AudioAnalysisRequest,DiarizationRequest
from titanos.audio_contract import AudioAnalysisResult,DiarizationResult,speaker_overlaps
from titanos.client import Client
from titanos.config import Config,ModelSpec
from titanos.engine import EngineError
from titanos.server import create_app
from titanos.settings import Settings

RAW=b'one original bounded audio fixture; provider decoding is mocked'
INLINE=base64.b64encode(RAW).decode()

class AudioAnalysis(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.config=Config(models={
   'asr':ModelSpec(path=str(self.root),task='transcription'),
   'diar':ModelSpec(path=str(self.root),task='diarization',backend='sherpa-onnx',device='cpu'),
   'vad':ModelSpec(path=str(self.root),task='vad',backend='onnx',device='cpu'),
   'sent':ModelSpec(path=str(self.root),task='sentiment'),
   'native-sent':ModelSpec(path=str(self.root),task='chat',backend='llama-cpp',device='cpu',native={'model_file':'unused.gguf'})})
  self.settings=Settings(max_audio_seconds=30,max_audio_samples=480000)
  self.app=create_app(self.config,api_key='',output_dir=self.root/'out',settings=self.settings,runtime_catalog=False)
  self.calls=[];self.cancel_objects=[];self.after=lambda task:None;self.bad=None;self.empty=False;self.timestamps=True;self.duration=12.
  self.decode=patch('titanos.media.read_audio',return_value=[0.]*192000);self.decode_mock=self.decode.start();self.addCleanup(self.decode.stop)
  self.mock=patch.object(self.app.state.engine.target,'infer',side_effect=self.infer);self.mock.start();self.addCleanup(self.mock.stop)
  self.http=TestClient(self.app,base_url="http://localhost");self.http.__enter__();self.addCleanup(self.http.__exit__,None,None,None)
 def diarization(self,data):
  segments=[{'speaker':speaker,'start':start,'end':end,'start_sample':int(start*16000),'end_sample':int(end*16000)}
    for speaker,start,end in [('speaker_0',.5,3.),('speaker_1',2.,4.),('speaker_0',6.,7.),('speaker_1',8.,9.)]]
  result={'model':'diar','task':'diarization','sample_rate':16000,'duration':self.duration,'decoded_samples':round(self.duration*16000),
   'input_sha256':hashlib.sha256(data['audio']).hexdigest(),'speakers':[{'id':'speaker_0'},{'id':'speaker_1'}],
   'segments':segments,'overlaps':speaker_overlaps(segments),'parameters':{k:data[k] for k in ['num_speakers','cluster_threshold','min_duration_on','min_duration_off']},
   'provenance':{'backend':'sherpa-onnx','device':'cpu','checkpoint_digest':'a'*64,'cancellation':'inert'}}
  if self.bad=='hash':result['input_sha256']='b'*64
  if self.bad=='overlap':result['overlaps']=[]
  if self.bad=='identity':result['speakers'][0]['id']='Alice'
  if self.bad=='duration':result['duration']=13.;result['decoded_samples']=208000
  if self.bad=='parameters':result['parameters']['cluster_threshold']=99.
  return result
 def infer(self,name,task,data,emit=None,cancel=None):
  self.calls.append((name,task,deepcopy(data)));self.cancel_objects.append(cancel)
  self.assertFalse(cancel.is_set())
  if task=='transcription':
   segments=[{'id':0,'start':.5,'end':4.,'text':'First words.'},{'id':1,'start':6.,'end':9.,'text':'Second words.'},{'id':2,'start':None,'end':None,'text':'Unaligned words.'}]
   word=data.get('timestamp_granularities')==['word']
   result={'task':'transcribe','text':'' if self.empty else 'First words. Second words. Unaligned words.',
    'duration':self.duration,'sample_rate':16000,'decoded_samples':round(self.duration*16000),'language':'english','language_source':'backend','requested_language':data.get('language'),
    'segments':segments if self.timestamps and not word else None,
    'words':[{'id':0,'start':2.,'end':2.5,'word':'First'}] if self.timestamps and word else None,
    'timestamps':{'granularity':'word' if word else 'segment','source':'backend_alignment' if self.timestamps else 'unavailable','supported':['segment','word']},
    'quality':{'confidence':None,'status':'unavailable'},'backend':'transformers'}
   if self.bad in {'alignment_overrun','all_alignment_overrun'}:
    if result['segments'] is not None:
     result['segments'][1].update(start=24.95,end=25.22)
     if self.bad=='all_alignment_overrun':result['segments'][0].update(start=25.0,end=25.3)
    if result['words'] is not None:result['words'][0].update(start=24.95,end=25.22)
   if self.bad=='invalid_alignment':result['segments'][0].update(start=self.invalid_pair[0],end=self.invalid_pair[1])
   if self.bad=='asr_budget':result.update(duration=31.,decoded_samples=496000)
  elif task=='diarization':result=self.diarization(data)
  elif task=='vad':
   result={'model':'vad','task':'vad','sample_rate':16000,'duration':self.duration,'decoded_samples':round(self.duration*16000),'frame_samples':512,'context_samples':64,
    'segments':[{'start':0.,'end':12.,'start_sample':0,'end_sample':192000}], 'frames':None,
    'parameters':{},'input_sha256':hashlib.sha256(data['audio']).hexdigest(),'provenance':{'backend':'onnx'}}
  elif task=='sentiment':
   result={'scores':{'negative':.05,'neutral':.05,'positive':.9},'score_kind':'softmax_uncalibrated','input_tokens':8,
    'provenance':{'backend':'transformers','device':'cpu','repository':None,'revision':None,'checkpoint_digest':'c'*64,'label_mapping':{'POSITIVE':'positive'}}}
  else:raise AssertionError('Unexpected inference task '+task)
  self.after(task);return result
 def body(self,**extra):return {'model':'asr','diarization_model':'diar','audio_base64':INLINE,**extra}
 def post(self,**extra):return self.http.post('/v1/audio/analysis',json=self.body(**extra))
 def test_standalone_controls_precedence_and_overlap_result(self):
  self.config.models['diar'].defaults={'cluster_threshold':.7,'min_duration_off':.9}
  response=self.http.post('/v1/audio/diarizations',json={'model':'diar','audio_base64':INLINE,'cluster_threshold':0.,'min_duration_off':0.})
  self.assertEqual(response.status_code,200,response.text);result=response.json()
  self.assertEqual(result['parameters']['cluster_threshold'],0.);self.assertEqual(result['parameters']['min_duration_off'],0.)
  self.assertEqual(result['parameters']['min_duration_on'],self.settings.diarization_min_duration_on)
  self.assertEqual(result['overlaps'][0]['speakers'],['speaker_0','speaker_1'])
  self.assertEqual((result['overlaps'][0]['start'],result['overlaps'][0]['end']),(2.,3.))
  self.assertNotIn('confidence',result);self.assertEqual([x[1] for x in self.calls],['diarization'])
 def test_all_static_stage_options_rejected_before_first_inference(self):
  cases=[{'diarization_model':'missing'},{'diarization_model':'asr'},{'diarization_worker':'missing'},
    {'text_analysis_model':'missing'},{'text_analysis_model':'vad'},
    {'text_analysis_model':'sent','text_analysis_options':{'max_tokens':3}},
    {'text_analysis_model':'sent','text_analysis_worker':'missing'},
    {'vad_enabled':True,'vad_model':'vad','vad_options':{'worker':'missing'}},
    {'vad_enabled':True,'vad_model':'vad','vad_options':{'min_speech_ms':31000}},
    {'diarization_options':{'num_speakers':True}},{'diarization_options':{'min_duration_off':-1}},
    {'text_analysis_options':{'min_score':.5}},{'max_tokens':True}]
  for fields in cases:
   with self.subTest(fields=fields):
    self.calls.clear();response=self.post(**fields);self.assertIn(response.status_code,[400,404,422],response.text);self.assertIn("application/json",response.headers["content-type"]);self.assertEqual(self.calls,[])
  self.config.models['asr'].defaults={'max_tokens':True}
  self.assertEqual(self.post().status_code,400);self.assertEqual(self.calls,[])
 def test_short_audio_nondefault_cluster_control_rejected_before_load(self):
  self.decode_mock.return_value=[0.]*16000
  response=self.post(diarization_options={'num_speakers':2})
  self.assertEqual(response.status_code,400,response.text);self.assertIn('bypass clustering',response.text);self.assertEqual(self.calls,[])
 def test_composition_fidelity_real_overlap_and_transcript_sentiment(self):
  response=self.post(vad_enabled=True,vad_model='vad',text_analysis_model='sent')
  self.assertEqual(response.status_code,200,response.text);r=response.json()
  self.assertEqual([x[1] for x in self.calls],['vad','transcription','diarization','sentiment'])
  self.assertTrue(all(x[2]['audio']==RAW for x in self.calls[:3]))
  self.assertEqual(self.calls[-1][2]['text'],r['text']);self.assertEqual(r['text'],r['transcription']['text'])
  first,second,unknown=r['speaker_segments']
  self.assertEqual(first['speakers'],['speaker_0','speaker_1']);self.assertTrue(first['overlap'])
  self.assertEqual(second['speakers'],['speaker_0','speaker_1']);self.assertFalse(second['overlap'])
  self.assertEqual(unknown['speakers'],[]);self.assertEqual(unknown['attribution'],'timestamps_unavailable');self.assertIsNone(unknown['start'])
  self.assertEqual(r['text_analysis']['input']['modality'],'asr_transcript');self.assertIsNone(r['text_analysis']['audio_affect'])
  self.assertEqual(r['text_analysis_status'],'completed');self.assertEqual(r['text_analysis_scope'],'transcript_text_only')
  self.assertEqual(r['text_analysis']['input']['sha256'],hashlib.sha256(r['text'].encode()).hexdigest())
 def test_missing_and_word_alignment_never_invent_segment_timestamps(self):
  self.timestamps=False;r=self.post().json();self.assertEqual(r['speaker_segments'],[]);self.assertEqual(r['speaker_alignment'],'unavailable')
  self.timestamps=True;r=self.post(timestamp_granularities=['word']).json();self.assertEqual(r['speaker_alignment'],'word')
  self.assertEqual(r['speaker_segments'][0]['text'],'First');self.assertTrue(r['speaker_segments'][0]['overlap'])
  self.assertIsNone(r['transcription']['segments']);self.assertEqual(r['transcription']['words'][0]['word'],'First')
 def test_bad_backend_measurement_stops_later_text_analysis(self):
  for bad in ['hash','overlap','identity','duration','parameters','asr_budget']:
   with self.subTest(bad=bad):
    self.bad=bad;self.calls.clear();response=self.post(text_analysis_model='sent')
    self.assertEqual(response.status_code,502,response.text);self.assertNotIn('sentiment',[x[1] for x in self.calls])
    if bad=='asr_budget':self.assertEqual([x[1] for x in self.calls],['transcription'])
 def test_cancellation_and_late_stage_share_one_deadline(self):
  payload=AudioAnalysisRequest(**self.body(text_analysis_model='sent'));handler=self.app.state.extension_handlers['audio_analysis']
  cancel=threading.Event();self.after=lambda task:cancel.set()
  with self.assertRaises(EngineError) as e:handler(payload,cancel,time.monotonic()+5)
  self.assertEqual(e.exception.status,499);self.assertEqual([x[1] for x in self.calls],['transcription'])
  self.assertTrue(all(x.is_set() for x in self.cancel_objects))
  # The deadline must outlast the in-process transcription stage under scheduler load (a 15 ms margin raced and failed
  # intermittently), while the diarization stage's sleep must outlast the deadline: ordering is asserted, not latency.
  self.calls.clear();self.after=lambda task:time.sleep(.5) if task=='diarization' else None
  with self.assertRaises(EngineError) as e:handler(payload,threading.Event(),time.monotonic()+.25)
  self.assertEqual(e.exception.status,504);self.assertEqual([x[1] for x in self.calls],['transcription','diarization'])
 def test_empty_transcript_never_calls_text_model(self):
  self.empty=True;r=self.post(text_analysis_model='sent').json()
  self.assertEqual(r['text_analysis_status'],'not_run_empty_transcript');self.assertIsNone(r['text_analysis']);self.assertEqual([x[1] for x in self.calls],['transcription','diarization'])
 def test_native_transcript_stage_uses_original_deadline_and_no_model_start(self):
  from unittest.mock import AsyncMock
  from titanos.native import NativeWorker
  observed=[]
  async def complete(data,**options):
   observed.append((deepcopy(data),options));self.assertFalse(options['cancel'].is_set())
   self.assertLessEqual(options['deadline'],began+.5)
   return {'text':'{"label":"positive"}','finish_reason':'stop','provenance':{'device':'cpu'}}
  began=time.monotonic()
  with patch.object(self.app.state.native,'complete',new=AsyncMock(side_effect=complete)), patch.object(NativeWorker,'start',new=AsyncMock(side_effect=AssertionError('Native start forbidden'))):
   response=self.post(timeout=400,text_analysis_model='native-sent',text_analysis_options={'max_tokens':11,'asr_uncertain':False})
  self.assertEqual(response.status_code,200,response.text);result=response.json()
  self.assertEqual([x[1] for x in self.calls],['transcription','diarization']);self.assertEqual(len(observed),1)
  self.assertEqual(observed[0][0]['max_tokens'],11);self.assertEqual(result['text_analysis']['method'],'llm')
  self.assertEqual(json.loads(observed[0][0]['messages'][-1]['content'])['text'],result['text'])
  self.assertEqual(result['text_analysis']['input']['modality'],'asr_transcript')
 def test_disconnect_and_task_cancellation_set_shared_event_and_settle_worker(self):
  handler=self.app.state.extension_handlers['audio_analysis']
  for mode in ['disconnect','task']:
   with self.subTest(mode=mode):
    entered=threading.Event();finished=threading.Event();calls=[]
    def blocked(name,task,data,emit=None,cancel=None):
     calls.append(task);entered.set()
     try:
      self.assertTrue(cancel.wait(1))
      raise EngineError('Inert provider observed shared cancellation',499,'cancelled')
     finally:finished.set()
    class Request:
     async def is_disconnected(self):return mode=='disconnect' and entered.is_set()
    async def exercise():
     task=asyncio.create_task(self.app.state.dispatch_cancellable(Request(),AudioAnalysisRequest(**self.body(timeout=700)),handler))
     until=time.monotonic()+1
     while not entered.is_set():
      self.assertLess(time.monotonic(),until);await asyncio.sleep(.002)
     if mode=='task':task.cancel()
     try:await asyncio.wait_for(task,2)
     except EngineError as exc:self.assertEqual(exc.status,499)
     except asyncio.CancelledError:self.assertEqual(mode,'task')
     else:self.fail('Cancelled request returned a result')
    with patch.object(self.app.state.engine.target,'infer',side_effect=blocked):asyncio.run(exercise())
    self.assertTrue(finished.is_set());self.assertEqual(calls,['transcription'])
 def test_typed_result_rejects_fabricated_speaker_attribution(self):
  result=self.post().json();AudioAnalysisResult.model_validate(result)
  for mutate in [lambda v:v['speaker_segments'][0].update(text='Invented transcript'),
                 lambda v:v['speaker_segments'][0].update(overlap=False),
                 lambda v:v['speaker_segments'][1].update(speakers=['speaker_7']),
                 lambda v:v['diarization'].update(overlaps=[])]:
   changed=deepcopy(result);mutate(changed)
   with self.assertRaises(ValueError):AudioAnalysisResult.model_validate(changed)
 def test_outside_audio_timestamps_preserved_but_unattributed(self):
  # Exact retained native-input duration;25.22 is a representative synthetic
  # backend overrun because the failed502 did not retain the raw ASR times.
  self.duration=25.2099375;self.bad='alignment_overrun'
  response=self.post(text_analysis_model='sent')
  self.assertEqual(response.status_code,200,response.text);result=response.json()
  first,outside,unknown=result['speaker_segments']
  self.assertEqual(result['speaker_alignment'],'segment');self.assertEqual(result['speaker_attribution_status'],'partial')
  self.assertEqual(first['attribution'],'interval_intersection');self.assertTrue(first['overlap']);self.assertEqual(first['speakers'],['speaker_0','speaker_1'])
  self.assertEqual((outside['id'],outside['text'],outside['start'],outside['end']),(1,'Second words.',24.95,25.22))
  self.assertEqual(outside['attribution'],'timestamps_outside_audio');self.assertEqual(outside['speakers'],[]);self.assertFalse(outside['overlap'])
  self.assertEqual(result['transcription']['segments'][1]['end'],25.22)
  self.assertEqual(unknown['attribution'],'timestamps_unavailable');self.assertIsNone(unknown['start'])
  self.assertEqual(result['text'],result['transcription']['text']);self.assertEqual(result['text_analysis_status'],'completed')
  AudioAnalysisResult.model_validate(result)
  tampered=deepcopy(result);tampered['speaker_segments'][1]['speakers']=['speaker_0']
  with self.assertRaises(ValueError):AudioAnalysisResult.model_validate(tampered)
  tampered=deepcopy(result);tampered['speaker_segments'][1]['end']=self.duration
  with self.assertRaises(ValueError):AudioAnalysisResult.model_validate(tampered)
 def test_outside_word_or_all_units_preserve_granularity_and_unavailable_status(self):
  self.duration=25.2099375;self.bad='alignment_overrun'
  response=self.post(timestamp_granularities=['word']);self.assertEqual(response.status_code,200,response.text);result=response.json()
  self.assertEqual(result['speaker_alignment'],'word');self.assertEqual(result['speaker_attribution_status'],'unavailable')
  self.assertEqual(result['speaker_segments'][0]['end'],25.22);self.assertEqual(result['transcription']['words'][0]['end'],25.22)
  self.assertEqual(result['speaker_segments'][0]['attribution'],'timestamps_outside_audio')
  self.bad='all_alignment_overrun';response=self.post();self.assertEqual(response.status_code,200,response.text)
  self.assertEqual(response.json()['speaker_attribution_status'],'unavailable')
 def test_invalid_numeric_or_reversed_alignment_still_refused(self):
  self.bad='invalid_alignment'
  for pair in [(2.,1.),(-1.,2.),(0.,float('inf')),(float('nan'),2.),('not-a-time',2.)]:
   with self.subTest(pair=pair):
    self.invalid_pair=pair;self.calls.clear();response=self.post(text_analysis_model='sent')
    self.assertEqual(response.status_code,502,response.text);self.assertNotIn('sentiment',[x[1] for x in self.calls])
 def test_typed_extensions_sdk_cli_and_mcp_reuse_http_contract(self):
  discovery=self.http.get('/v1/extensions').json();ops={x['operation'] for x in discovery['data']}
  self.assertTrue({'diarization','audio_analysis'}<=ops);self.assertIn('DiarizationOperation',discovery['schema']['$defs'])
  response=self.http.post('/v1/extensions/infer',json={'operation':'audio_analysis',**self.body()});self.assertEqual(response.status_code,200,response.text)
  file=self.root/'original.wav';file.write_bytes(RAW)
  def bridge(request):
   response=self.http.request(request.method,request.url.path,content=request.content,headers={'content-type':'application/json'})
   return httpx.Response(response.status_code,json=response.json())
  with Client(base_url='http://inert',transport=httpx.MockTransport(bridge)) as client:
   self.assertIsInstance(client.diarize('diar',file),DiarizationResult)
   self.assertIsInstance(client.analyze_audio('asr',file,diarization_model='diar'),AudioAnalysisResult)
  from titanos import cli
  with patch.object(cli,'remote_command') as command:
   cli.main(['analyze-audio','asr',str(file),'--diarization-model','diar','--num-speakers','0','--min-duration-off','0','--no-vad'])
  args=command.call_args.args[0];self.assertEqual(args.num_speakers,0);self.assertEqual(args.min_duration_off,0.);self.assertFalse(args.vad_enabled)
  observed={}
  def analyze(model,path,**options):observed.update(options);return AudioAnalysisResult.model_validate(self.post().json())
  with patch.object(Client,'analyze_audio',side_effect=analyze),patch('sys.stdout',new=io.StringIO()):cli.remote_command(args)
  self.assertEqual(observed['diarization_options'],{'num_speakers':0,'min_duration_off':0.});self.assertFalse(observed['vad_enabled'])
  from titanos.mcp_server import create_mcp
  async def mcp_call():
   mcp=create_mcp(app=self.app)
   tools=await mcp.list_tools();self.assertTrue({'diarize','analyze_audio'}<={x.name for x in tools})
   a=await mcp.call_tool('diarize',{'payload':{'model':'diar','audio_base64':INLINE}})
   b=await mcp.call_tool('analyze_audio',{'payload':self.body()})
   self.assertFalse(a.is_error);self.assertFalse(b.is_error)
  asyncio.run(mcp_call())

if __name__=='__main__':unittest.main(verbosity=2)
