"""Controlled audio semantics, public contracts and cancellation; no trained model."""
import asyncio
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from titanos.asr import transcribe
from titanos.audio_contract import VADResult
from titanos.client import Client
from titanos.config import Config, ModelSpec
from titanos.engine import EngineError
from titanos.server import create_app
from titanos.settings import Settings, _active_settings
from titanos.vad import SileroVAD, parameters, intervals


class Session:
    def __init__(self, scores=(.9, .9, .0, .0, .8)):
        self.scores = scores
        self.calls = []
        self.after = lambda: None
    def get_providers(self): return ['CPUExecutionProvider']
    def get_inputs(self): return [NS(name=name,type=dtype) for name,dtype in [('input','tensor(float)'),('state','tensor(float)'),('sr','tensor(int64)')]]
    def get_outputs(self): return [NS(name='output'),NS(name='stateN')]
    def run(self, names, data):
        self.calls.append({k: v.copy() for k,v in data.items()})
        index = int(data['state'][0,0,0])
        self.after()
        return np.array([[self.scores[index % len(self.scores)]]], dtype=np.float32), data['state'] + 1


class Pipeline:
    def __init__(self, kind='seq2seq_whisper'):
        self.type = kind
        self.feature_extractor = NS(sampling_rate=16000)
        self.model = NS(config=NS(model_type='whisper'), generation_config=NS(is_multilingual=True))
        self.calls = []
        self.result = {'text':'hello world', 'chunks':[{'text':'hello', 'timestamp':(0., .5), 'language':'english'},
                                                     {'text':' world', 'timestamp':(.5, None), 'language':'english'}]}
        self.after = lambda: None
    def __call__(self, value, **kwargs):
        self.calls.append(kwargs)
        self.after()
        return self.result


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.token = _active_settings.set(Settings(max_audio_seconds=10, vad_min_speech_ms=0, vad_min_silence_ms=64, vad_speech_pad_ms=0))
        self.spec = ModelSpec(path=str(self.root), task='vad', backend='onnx')
        self.vad = SileroVAD.__new__(SileroVAD)
        self.vad.spec, self.vad.session, self.vad.digest = self.spec, Session(), 'a'*64
    def tearDown(self):
        _active_settings.reset(self.token)
        self.temp.cleanup()
    def call(self, wave, **data):
        with patch('titanos.vad.read_audio', return_value=wave):
            return self.vad.infer('vad', {'audio':b'fixture', **data})

    def test_frame_state_context_reset_and_tail(self):
        for rate,width,context in [(16000,512,64),(8000,256,32)]:
            wave=np.arange(width*4+17,dtype=np.float32) / (width*5)
            self.vad.session.calls.clear()
            result=self.call(wave,sample_rate=rate,return_frames=True)
            VADResult.model_validate(result)
            calls=self.vad.session.calls
            self.assertEqual(len(calls),5)
            self.assertEqual(calls[0]['input'].shape,(1,width+context))
            self.assertTrue((calls[0]['state']==0).all())
            self.assertTrue((calls[0]['input'][0,:context]==0).all())
            np.testing.assert_equal(calls[1]['input'][0,:context],wave[width-context:width])
            self.assertEqual(calls[1]['state'][0,0,0],1)
            self.assertEqual(calls[0]['sr'].dtype,np.int64)
            self.assertEqual(result['frames'][-1]['padded_samples'],width-17)
            self.assertEqual(result['frames'][-1]['end_sample'],len(wave))
            again=self.call(wave,sample_rate=rate,return_frames=True)
            self.assertEqual(result,again)
            self.assertTrue((self.vad.session.calls[-5]['state']==0).all())

    def test_interval_boundaries_padding_and_minimum(self):
        values=parameters({'sample_rate':16000,'min_speech_ms':0,'min_silence_ms':64,'speech_pad_ms':0},{})
        self.assertEqual(intervals([.9,.9,0,0,.8],2065,16000,512,values),[
            {'start':0.,'end':.064,'start_sample':0,'end_sample':1024},
            {'start':.128,'end':2065/16000,'start_sample':2048,'end_sample':2065}])
        self.assertEqual(intervals([0]*5,2065,16000,512,values),[])
        values['min_speech_ms']=100
        self.assertEqual(intervals([.9,.9,0,0],2048,16000,512,values),[])
        values.update(min_speech_ms=0,speech_pad_ms=40)
        spans=intervals([.9,.9,0,0,.8],2065,16000,512,values)
        self.assertEqual(len(spans),1)
        self.assertEqual((spans[0]['start_sample'],spans[0]['end_sample']),(0,2065))

    def test_limits_bad_outputs_cancel_and_reuse(self):
        for bad in [{'threshold':False},{'sample_rate':44100},{'min_silence_ms':0},{'speech_pad_ms':float('nan')},{'return_frames':'yes'},
                    {'min_speech_ms':10**1000},{'min_silence_ms':10**1000},{'speech_pad_ms':10**1000}]:
            with self.assertRaises(ValueError):self.call(np.zeros(512,dtype=np.float32),**bad)
        self.assertEqual(len(self.vad.session.calls),0)
        token=_active_settings.set(Settings(vad_max_frames=1))
        try:
            with self.assertRaises(ValueError):self.call(np.zeros(513,dtype=np.float32))
        finally:_active_settings.reset(token)
        self.assertEqual(len(self.vad.session.calls),0)
        event=threading.Event();self.vad.session.after=event.set
        with patch('titanos.vad.read_audio',return_value=np.zeros(1024,dtype=np.float32)):
            with self.assertRaises(EngineError) as error:self.vad.infer('vad',{'audio':b'fixture'},cancel=event)
        self.assertEqual(error.exception.code,'cancelled')
        self.assertEqual(len(self.vad.session.calls),1)
        self.vad.session.after=lambda:None
        self.call(np.zeros(512,dtype=np.float32))
        self.assertEqual(self.vad.session.calls[-1]['state'][0,0,0],0)
        self.vad.session.scores=[float('nan')]
        with self.assertRaises(EngineError):self.call(np.zeros(512,dtype=np.float32))
        for fields in [{'device':'dml'},{'backend':'transformers'},{'graph':'../other.onnx'},{'defaults':{'threshold':False}},{'load_options':{'x':1}},
                       *[{'defaults':{key:10**1000}} for key in ('min_speech_ms','min_silence_ms','speech_pad_ms')]]:
            with self.assertRaises(ValueError):ModelSpec(path=str(self.root),task='vad',**{'backend':'onnx',**fields})

    def test_onnx_admission_fingerprint_and_abi(self):
        path=self.root/'silero_vad.onnx';path.write_bytes(b'controlled graph bytes')
        session=Session()
        with patch('titanos.compiled.onnx_session',return_value=session) as loader:
            model=SileroVAD(self.spec)
            self.assertEqual(loader.call_args.args,(path.resolve(),'cpu'))
            self.assertEqual(model.details()['checkpoint_digest'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertFalse(session.calls)
        session.get_inputs=lambda:[NS(name='wrong',type='tensor(float)')]
        with patch('titanos.compiled.onnx_session',return_value=session):
            with self.assertRaisesRegex(ValueError,'ABI'):SileroVAD(self.spec)
        def mutation(*args):path.write_bytes(b'changed');return Session()
        with patch('titanos.compiled.onnx_session',side_effect=mutation):
            with self.assertRaisesRegex(ValueError,'changed'):SileroVAD(self.spec)

    def test_asr_metadata_and_existing_text(self):
        pipeline=Pipeline()
        spec=ModelSpec(path=str(self.root),task='transcription')
        with patch('titanos.asr.read_audio',return_value=np.zeros(16000,dtype=np.float32)):
            self.assertEqual(transcribe(pipeline,spec,{'audio':b'wav'}),'hello world')
            result=transcribe(pipeline,spec,{'audio':b'wav','response_format':'verbose_json','language':'en'})
            self.assertTrue(pipeline.calls[-1]['return_timestamps'])
            self.assertEqual(result['duration'],1.)
            self.assertEqual(result['language'],'english')
            self.assertEqual(result['requested_language'],'en')
            self.assertIsNone(result['segments'][-1]['end'])
            self.assertTrue(all(result['quality'][k] is None for k in ['confidence','avg_logprob','no_speech_prob']))
            words=transcribe(pipeline,spec,{'audio':b'wav','response_format':'verbose_json','timestamp_granularities':['word']})
            self.assertEqual(words['words'][0]['word'],'hello')
            self.assertIsNone(words['segments'])
            before=len(pipeline.calls)
            for options in [{'timestamp_granularities':['word']},{'response_format':'verbose_json','timestamp_granularities':['word','segment']},
                            {'response_format':'verbose_json','translation':True,'timestamp_granularities':['word']}]:
                with self.assertRaises(ValueError):transcribe(pipeline,spec,{'audio':b'wav',**options})
            self.assertEqual(len(pipeline.calls),before)
            pipeline.type='seq2seq'
            result=transcribe(pipeline,spec,{'audio':b'wav','response_format':'verbose_json'})
            self.assertIsNone(result['segments'])
            self.assertEqual(result['timestamps']['source'],'unavailable')
            with self.assertRaises(ValueError):transcribe(pipeline,spec,{'audio':b'wav','response_format':'verbose_json','timestamp_granularities':['segment']})
            pipeline.type='ctc';event=threading.Event();pipeline.after=event.set
            with self.assertRaises(EngineError):transcribe(pipeline,spec,{'audio':b'wav'},event)

    def test_decoder_cooperative_cleanup(self):
        from titanos.media import read_audio
        event=threading.Event()
        class Process:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def communicate(self,**kw):
                if kw:
                    event.set()
                    raise subprocess.TimeoutExpired('fixture',.1)
                return b'',b''
            def kill(self):self.killed=True
        process=Process()
        with patch.dict(sys.modules,{'imageio_ffmpeg':NS(get_ffmpeg_exe=lambda:'fixture-ffmpeg')}),patch('titanos.media.subprocess.Popen',return_value=process) as opened:
            with self.assertRaises(EngineError):read_audio(b'fixture',16000,event)
            self.assertTrue(process.killed)
            args=opened.call_args[0][0]
            self.assertIn('-protocol_whitelist',args)
            self.assertEqual(args[args.index('-protocol_whitelist')+1],'pipe')

    def test_public_api_sdk_cli_and_mcp(self):
        cfg=Config(models={'voice':ModelSpec(path=str(self.root),task='transcription'),'vad':self.spec})
        app=create_app(cfg,output_dir=self.root/'outputs')
        pipeline=Pipeline('ctc')
        observed=[]
        def infer(name,task,data,emit=None,cancel=None):
            app.state.engine.target.validate(name,task)
            observed.append(data)
            if task=='vad':
                with patch('titanos.vad.read_audio',return_value=np.zeros(512,dtype=np.float32)):
                    return self.vad.infer(task,data,cancel=cancel)
            with patch('titanos.asr.read_audio',return_value=np.zeros(16000,dtype=np.float32)):
                return transcribe(pipeline,cfg.models[name],data,cancel)
        with patch.object(app.state.engine,'infer',side_effect=infer),TestClient(app,base_url='http://127.0.0.1') as client:
            upload={'file':('fixture.wav',b'fixture','audio/wav')}
            plain=client.post('/v1/audio/transcriptions',data={'model':'voice'},files=upload)
            self.assertEqual(plain.status_code,200,plain.text)
            self.assertEqual(plain.json(),{'text':'hello world'})
            text=client.post('/v1/audio/transcriptions',data={'model':'voice','response_format':'text'},files=upload)
            self.assertEqual(text.text,'hello world')
            verbose=client.post('/v1/audio/transcriptions',data={'model':'voice','response_format':'verbose_json','timestamp_granularities[]':'word'},files=upload)
            self.assertEqual(verbose.status_code,200,verbose.text)
            self.assertEqual(verbose.json()['words'][0]['word'],'hello')
            self.assertEqual(client.post('/v1/audio/transcriptions',data={'model':'voice','response_format':'srt'},files=upload).status_code,422)
            payload={'model':'vad','audio_base64':base64.b64encode(b'fixture').decode(),'return_frames':True,'min_speech_ms':0}
            result=client.post('/v1/audio/vad',json=payload)
            self.assertEqual(result.status_code,200,result.text)
            self.assertEqual(result.json()['input_sha256'],hashlib.sha256(b'fixture').hexdigest())
            self.assertEqual(client.post('/v1/extensions/infer',json={'operation':'vad',**payload}).status_code,200)
            self.assertEqual(client.post('/v1/audio/vad',json={**payload,'sample_rate':44100}).status_code,422)
            invalid=client.post('/api/models/register',json={'name':'bad-vad','spec':{'path':str(self.root),'task':'vad','backend':'onnx','defaults':{'min_speech_ms':10**1000}}})
            self.assertEqual(invalid.status_code,422,invalid.text)
        path=self.root/'fixture.wav';path.write_bytes(b'fixture')
        with patch.object(Client,'_json',return_value=result.json()) as request:
            with Client() as client:typed=client.vad('vad',path,threshold=.7)
            self.assertIsInstance(typed,VADResult)
            self.assertEqual(request.call_args.kwargs['json']['threshold'],.7)
        from titanos import cli
        with patch.object(sys,'argv',['titanos','vad','vad',str(path),'--sample-rate','8000','--return-frames']),patch.object(cli,'remote_command') as command:
            cli.main()
            self.assertEqual(command.call_args.args[0].sample_rate,8000)
        from titanos.mcp_server import create_mcp
        mcp=create_mcp('http://127.0.0.1:11435')
        tools=asyncio.run(mcp.list_tools())
        self.assertIn('vad',[t.name for t in tools])
        transcribe_tool=next(t for t in tools if t.name=='transcribe')
        self.assertIn('response_format',transcribe_tool.input_schema['properties'])
        async def invoke():
            internal=create_mcp(app=app)
            with patch.object(app.state.engine,'infer',side_effect=infer):
                result=await internal.call_tool('vad',{'payload':payload})
                transcript=await internal.call_tool('transcribe',{'model':'voice','audio_base64':payload['audio_base64'],
                    'response_format':'verbose_json','timestamp_granularities':['word']})
            self.assertIsNotNone(result)
            self.assertIsNotNone(transcript)
        asyncio.run(invoke())


if __name__=='__main__':unittest.main(verbosity=2)
