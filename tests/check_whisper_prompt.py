"""Whisper prompt conditioning, native budgets and real public dispatch with doubles."""
import asyncio
import base64
from contextlib import redirect_stdout
from copy import deepcopy
from email.parser import BytesParser
from email.policy import default
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, PropertyMock

import httpx
import numpy as np
from fastapi.testclient import TestClient
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from titanos import asr, cli
from titanos.client import Client
from titanos.config import Config, ModelSpec
from titanos.engine import EngineError
from titanos.server import create_app


class IDs:
    dtype='long'
    def __init__(self,value):self.value=np.asarray(value,dtype=np.int64);self.ndim=self.value.ndim;self.shape=self.value.shape
    def numel(self):return self.value.size
    def tolist(self):return self.value.tolist()
    def __getitem__(self,item):return IDs(self.value[item])
    def item(self):return int(self.value)
    def to(self,device):return self


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.spec=ModelSpec(path=str(self.root),task='transcription')
        self.calls=[];self.prompts=[];self.prefix_calls=[];self.ids=IDs([90,1,2]);self.after_ids=lambda:None
        self.wave=np.zeros(16000,dtype=np.float32);owner=self
        class Tokenizer:
            def get_prompt_ids(self,text,return_tensors):
                owner.prompts.append((text,return_tensors))
                if '<|' in text:raise ValueError('disallowed special token')
                owner.after_ids();return owner.ids
        class Processor:
            def __init__(self,feature_extractor,tokenizer):self.tokenizer=tokenizer
            def get_prompt_ids(self,text,return_tensors):return self.tokenizer.get_prompt_ids(text,return_tensors)
        class GenerationMixin:
            @staticmethod
            def _set_language_and_task(language,task,is_multilingual,generation):
                if language is not None:generation.language=language
                if task is not None:generation.task=task
            @staticmethod
            def _retrieve_init_tokens(proxy,features,batch,generation,config,frames,kwargs):
                owner.prefix_calls.append((proxy,features,batch,generation))
                assert features is None and batch==1 and proxy.device=='cpu'
                if generation.language is None:proxy.detect_language()
                return IDs([[1,10,11] if generation.return_timestamps else [1,10,11,12]])
        class Pipeline:
            type='seq2seq_whisper'
            tokenizer=Tokenizer()
            feature_extractor=NS(sampling_rate=16000,chunk_length=30)
            model=NS(device='cpu',config=NS(model_type='whisper',max_target_positions=64,vocab_size=200),
                generation_config=NS(prev_sot_token_id=None,decoder_start_token_id=1,language=None,task=None,
                    is_multilingual=True,lang_to_id={'<|en|>':10,'<|fr|>':13},max_new_tokens=None,cache_implementation=None,num_beams=1))
            def __call__(self,audio,**kwargs):owner.calls.append((audio,kwargs));return {'text':'unchanged recognized text'}
        self.pipeline=Pipeline()
        self.pipeline.generation_config=deepcopy(self.pipeline.model.generation_config)
        self.pipeline.generation_config.num_beams=5
        self.modules=patch.dict('sys.modules',{'torch':NS(long='long',tensor=lambda value,**kwargs:IDs(value),
            cuda=NS(is_available=lambda:False),backends=NS(mps=NS(is_available=lambda:False))),
            'transformers':NS(WhisperProcessor=Processor,StoppingCriteria=type('StoppingCriteria',(),{}),StoppingCriteriaList=list),
            'transformers.models.whisper.generation_whisper':NS(WhisperGenerationMixin=GenerationMixin)})
        self.modules.start();self.addCleanup(self.modules.stop)
        self.audio=patch('titanos.asr.read_audio',side_effect=lambda *args:self.wave);self.read=self.audio.start();self.addCleanup(self.audio.stop)
    def run_asr(self,**values):
        return asr.transcribe(self.pipeline,self.spec,{'audio':b'audio',**values})
    def test_ids_preserved_generation_limit_and_omission_unchanged(self):
        baseline=self.run_asr();self.assertEqual(baseline,'unchanged recognized text');self.assertEqual(self.calls[-1][1],{})
        self.assertFalse(self.prompts)
        original=deepcopy(vars(self.pipeline.model.generation_config))
        result=self.run_asr(prompt='Vocabulary names.',max_tokens=57)
        self.assertEqual(result,baseline)
        generation=self.calls[-1][1]['generate_kwargs']
        self.assertIs(generation['prompt_ids'],self.ids);self.assertEqual(generation['max_new_tokens'],57)
        self.assertEqual(generation['prompt_condition_type'],'first-segment')
        self.assertEqual(vars(self.pipeline.model.generation_config),original)
        self.assertEqual(generation['generation_config'].prev_sot_token_id,90)
        self.assertEqual(self.prompts[-1],('Vocabulary names.','pt'))
        with self.assertRaisesRegex(EngineError,'new decoder tokens'):self.run_asr(prompt='Vocabulary names.',max_tokens=58)
        self.assertEqual(len(self.calls),2)
        verbose=self.run_asr(prompt='Names',max_tokens=8,response_format='verbose_json',language='en',timestamp_granularities=['word'])
        self.assertEqual(verbose['prompt_conditioning']['decoder_prefix_budget'],3)
        self.assertEqual(verbose['text'],baseline);self.assertIsNone(verbose['quality']['confidence'])
        self.assertEqual(self.calls[-1][1]['return_timestamps'],'word')
        self.assertEqual(self.calls[-1][1]['generate_kwargs']['language'],'en')
    def test_token_cutoff_special_tokens_unsupported_family_and_cancel(self):
        for prompt in ('<|notimestamps|>',' '):
            with self.assertRaises((ValueError,EngineError)):self.run_asr(prompt=prompt)
        self.ids=IDs([90,*range(32)])
        with self.assertRaisesRegex(EngineError,'context limit of 31'):self.run_asr(prompt='long')
        self.read.assert_not_called();self.assertFalse(self.calls)
        self.ids=IDs([90,*range(31)])
        self.run_asr(prompt='boundary',max_tokens=28)
        self.assertEqual(self.calls[-1][1]['generate_kwargs']['prompt_ids'].numel(),32)
        self.pipeline.type='ctc'
        with self.assertRaisesRegex(EngineError,'builtin Whisper'):self.run_asr(prompt='Names')
        self.pipeline.type='seq2seq_whisper';cancel=threading.Event();self.after_ids=cancel.set;before=len(self.calls)
        with self.assertRaises(EngineError):asr.transcribe(self.pipeline,self.spec,{'audio':b'audio','prompt':'Names'},cancel)
        self.assertEqual(len(self.calls),before)
    def test_long_form_static_reserves_and_competing_prefixes(self):
        self.pipeline.generation_config.condition_on_prev_tokens=True
        self.wave=np.zeros(16000*31,dtype=np.float32)
        self.run_asr(prompt='Names',max_tokens=28)
        with self.assertRaisesRegex(EngineError,'28 new'):self.run_asr(prompt='Names',max_tokens=29)
        self.wave=np.zeros(16000,dtype=np.float32)
        self.pipeline.generation_config.cache_implementation='static'
        with self.assertRaisesRegex(EngineError,'28 new'):self.run_asr(prompt='Names',max_tokens=29)
        self.pipeline.generation_config.cache_implementation=None
        for kwargs in ({'prompt_ids':'injected'},{'decoder_input_ids':[1]},{'prompt_condition_type':'all-segments'}):
            self.spec=ModelSpec(path=str(self.root),task='transcription',defaults={'generate_kwargs':kwargs})
            before=len(self.calls)
            with self.assertRaises(EngineError):self.run_asr(prompt='Names')
            self.assertEqual(len(self.calls),before)
    def test_effective_pipeline_defaults_and_request_overrides_are_preserved(self):
        config=self.pipeline.generation_config
        config.language='fr';config.return_timestamps=True;config.condition_on_prev_tokens=False
        original_pipeline=deepcopy(vars(config));original_model=deepcopy(vars(self.pipeline.model.generation_config))
        # No explicit timestamp option: the pipeline fallback makes the prefix
        # three tokens, leaving58 new tokens for this short controlled input.
        self.run_asr(prompt='Names',max_tokens=58)
        forwarded=self.calls[-1][1]['generate_kwargs']['generation_config']
        self.assertEqual(forwarded.num_beams,5);self.assertEqual(forwarded.language,'fr')
        self.assertTrue(forwarded.return_timestamps);self.assertEqual(self.prefix_calls[-1][3].return_timestamps,True)
        self.spec=ModelSpec(path=str(self.root),task='transcription',defaults={'generate_kwargs':{'num_beams':2}})
        self.run_asr(prompt='Names',max_tokens=8,language='en')
        values=self.calls[-1][1]['generate_kwargs']
        self.assertEqual(values['num_beams'],2);self.assertEqual(values['language'],'en')
        self.assertEqual(values['generation_config'].language,'en')
        self.assertEqual(vars(config),original_pipeline);self.assertEqual(vars(self.pipeline.model.generation_config),original_model)
    def test_native_previous_token_precedence_and_false_fallback(self):
        self.wave=np.zeros(16000*31,dtype=np.float32)
        config=self.pipeline.generation_config
        # Exact native precedence: explicit non-None argument, otherwise config,
        # otherwise False. Exercise False, True, None and missing config values.
        for configured in (False,True,None,'missing'):
            if configured=='missing':vars(config).pop('condition_on_prev_tokens',None)
            else:config.condition_on_prev_tokens=configured
            for explicit in (None,False,True):
                options={} if explicit is None else {'condition_on_prev_tokens':explicit}
                self.spec=ModelSpec(path=str(self.root),task='transcription',defaults={'generate_kwargs':options})
                native=explicit if explicit is not None else getattr(config,'condition_on_prev_tokens',False)
                result=self.run_asr(prompt='Names',max_tokens=28,response_format='verbose_json')
                # verbose segment timestamps reserve a three-token prefix.
                self.assertEqual(result['prompt_conditioning']['previous_token_reserve'],32 if native else 3)
        config.condition_on_prev_tokens=False;self.spec=ModelSpec(path=str(self.root),task='transcription')
        self.run_asr(prompt='Names',max_tokens=40)
        self.spec=ModelSpec(path=str(self.root),task='transcription',defaults={'generate_kwargs':{'condition_on_prev_tokens':True}})
        with self.assertRaisesRegex(EngineError,'28 new'):self.run_asr(prompt='Names',max_tokens=40)
    def test_http_extension_mcp_and_model_default(self):
        app=create_app(Config(models={'asr':self.spec}),output_dir=self.root/'out')
        app.state.engine.target.infer=lambda name,task,data,**kwargs:asr.transcribe(self.pipeline,self.spec,data,kwargs.get('cancel'))
        body={'model':'asr','prompt':'Names','max_tokens':8}
        with TestClient(app,base_url='http://127.0.0.1') as client:
            response=client.post('/v1/audio/transcriptions',data={**body,'response_format':'verbose_json'},files={'file':('a.wav',b'audio')})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()['prompt_conditioning']['source'],'request')
            self.assertEqual(response.json()['text'],'unchanged recognized text')
            response=client.post('/v1/extensions/infer',json={**body,'operation':'audio_transcription','audio_base64':base64.b64encode(b'audio').decode()})
            self.assertEqual(response.status_code,200,response.text)
            async def invoke():
                schema=next(t.input_schema for t in await app.state.mcp.list_tools() if t.name=='transcribe')
                self.assertIn('prompt',schema['properties'])
                reply=await app.state.mcp.call_tool('transcribe',{**body,'audio_base64':base64.b64encode(b'audio').decode()})
                self.assertFalse(reply.is_error);self.assertEqual(self.prompts[-1][0],'Names')
                with self.assertRaises(ToolError):await app.state.mcp.call_tool('transcribe',{**body,'prompt':5,'audio_base64':'YQ=='})
            with patch.object(Context,'request_context',new_callable=PropertyMock,return_value=NS(request=None)):asyncio.run(invoke())
            before=len(self.calls)
            for value in ('',' '):
                response=client.post('/v1/audio/transcriptions',data={'model':'asr','prompt':value},files={'file':('a.wav',b'audio')})
                self.assertEqual(response.status_code,400);self.assertEqual(len(self.calls),before)
        self.spec=ModelSpec(path=str(self.root),task='transcription',defaults={'prompt':'Default names'})
        self.assertEqual(self.run_asr(response_format='verbose_json')['prompt_conditioning']['source'],'model_default')
    def test_vad_silence_skips_asr_and_tokenization(self):
        from check_asr_gating import Gates
        gate=Gates();gate.setUp()
        try:
            response=gate.upload(prompt='Names',response_format='verbose_json')
            self.assertEqual(response.status_code,200,response.text);self.assertEqual(response.json()['text'],'')
            self.assertEqual([task for _,task,_ in gate.calls],['vad'])
            self.assertFalse(response.json()['vad_gate']['asr_executed'])
            self.assertNotIn('prompt_conditioning',response.json());self.assertFalse(self.prompts)
        finally:gate.tearDown()
    def test_client_and_cli_preserve_context_and_omission(self):
        path=self.root/'audio.wav';path.write_bytes(b'audio');seen=[]
        def handle(request):
            message=BytesParser(policy=default).parsebytes(('Content-Type: '+request.headers['content-type']+'\r\n\r\n').encode()+request.content)
            seen.append({part.get_param('name',header='content-disposition'):part.get_payload(decode=True) for part in message.iter_parts()})
            return httpx.Response(200,json={'text':'unchanged'})
        with Client(transport=httpx.MockTransport(handle)) as client:
            client.transcribe('asr',path,prompt='Names');client.transcribe('asr',path)
        self.assertEqual(seen[0]['prompt'],b'Names');self.assertNotIn('prompt',seen[1])
        for options,expected in [(['--prompt','Names'],{'prompt':'Names'}),([], {})]:
            with patch.object(Client,'transcribe',return_value={'text':'unchanged'}) as transcribe,redirect_stdout(io.StringIO()):
                cli.main(['transcribe','asr',str(path),*options])
            self.assertEqual(transcribe.call_args.kwargs,expected)


if __name__=='__main__':unittest.main(verbosity=2)
