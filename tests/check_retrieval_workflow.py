"""Controlled inline-corpus workflow contracts; no model/native process execution."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from titanos.config import Config, ModelSpec, NativeOptions
from titanos.engine import EngineError
from titanos.retrieval_workflow import INSTRUCTION, json_hash, text_hash, validate_answer
from titanos.retrieval_prompts import INSTRUCTIONS
from titanos.retrieval_workflow_contract import Document, RetrievalAnswerRequest, RetrievalAnswerResult
from titanos.server import create_app
from titanos.settings import Settings

DOCS=[{'id':'A','text':'Alpha is the first label.','source_uri':'source:A'},
      {'id':'B','text':'Beta is a distractor.'}, {'id':'C','text':'Gamma is the final label.'}]


class Fixture:
    def __init__(self, folder, settings=None):
        self.cfg=Config(models={'e':ModelSpec(path=folder,task='embeddings'), 'r':ModelSpec(path=folder,task='rerank'),
            'l':ModelSpec(path=folder,task='chat',backend='llama-cpp',device='cpu',native=NativeOptions(model_file='fixture.gguf',gpu_layers=0),defaults={'max_tokens':32,'temperature':0})})
        self.app=create_app(self.cfg,output_dir=Path(folder),settings=settings)
        self.calls=[];self.native_calls=[];self.answer=None;self.finish='stop';self.bad_vector=None;self.failure=None
        self.delay=None;self.entered=threading.Event();self.stopped=threading.Event()
        self.prompt_version='retrieval-answer-v2'
        self.app.state.engine.target.infer=self.infer
        self.app.state.native.complete=self.native
        self.payload={'embedding_model':'e','rerank_model':'r','llm_model':'l','query':'Which labels are first and final?',
            'documents':copy.deepcopy(DOCS),'retrieve_top_k':3,'context_top_n':2,'timeout':1000}
    def infer(self,model,task,data,emit=None,cancel=None):
        self.calls.append((model,task,copy.deepcopy(data)))
        if self.failure==task:raise EngineError('Controlled model failure',503,'fixture_failure')
        if self.delay==task:
            self.entered.set();assert cancel.wait(2);self.stopped.set()
        if task=='embeddings':
            texts=data['input'];vectors=[[1.,0.],[.8,.6],[1.,0.],[1.,0.]]
            if self.bad_vector=='zero':vectors[1]=[0.,0.]
            if self.bad_vector=='dimension':vectors[1]=[1.]
            if self.bad_vector=='nan':vectors[1]=[float('nan'),1.]
            return {'vectors':vectors,'usage':{'prompt_tokens':12,'total_tokens':12},
                'input_details':[{'index':i,'input_tokens':3,'processed_tokens':3,'truncated':False,'input_sha256':text_hash(text)} for i,text in enumerate(texts)],
                'truncation':{'enabled':False,'side':'right','strategy':'single_sequence','max_tokens':256},
                'provenance':{'repository':None,'revision':None,'checkpoint_digest':'a'*64,'backend':'transformers','device':'cpu','pooling':'attention_mask_mean','normalization':'l2'}}
        documents=data['documents'];scores={DOCS[0]['text']:4.,DOCS[1]['text']:-1.,DOCS[2]['text']:2.}
        return {'results':[{'index':i,'relevance_score':scores[text]} for i,text in enumerate(documents)],'score_kind':'cross_encoder_logit',
            'usage':{'prompt_tokens':5*len(documents),'total_tokens':5*len(documents)},
            'input_details':[{'index':i,'input_tokens':5,'processed_tokens':5,'truncated':False,'input_sha256':text_hash(text),'query_sha256':text_hash(data['query'])} for i,text in enumerate(documents)],
            'truncation':{'enabled':False,'side':'right','strategy':'only_second','max_tokens':512},
            'provenance':{'checkpoint_digest':'b'*64,'backend':'transformers','device':'cpu'}}
    async def native(self,data,*,deadline,cancel,context_max_tokens):
        self.native_calls.append(copy.deepcopy(data))
        if self.delay=='native':
            self.entered.set()
            while not cancel.is_set() and time.monotonic()<deadline:await asyncio.sleep(.005)
            self.stopped.set();raise EngineError('Controlled native cancellation',499,'cancelled')
        documents=json.loads(data['messages'][1]['content'])['documents']
        assert [doc['document_id'] for doc in documents]==['A','C']
        assert data['messages'][0]['content']==INSTRUCTIONS[self.prompt_version]
        assert data['response_format']['type']=='json_schema'
        answer=self.answer if self.answer is not None else {'status':'answered','claims':[
            {'text':'Alpha is first.','citations':[{'document_id':'A','quote':DOCS[0]['text']}]},
            {'text':'Gamma is final.','citations':[{'document_id':'C','quote':DOCS[2]['text']}]}]}
        text=json.dumps(answer) if not isinstance(answer,str) else answer
        return {'text':text,'finish_reason':self.finish,'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30},
            'provenance':{'backend':'llama-cpp','device':'cpu','checkpoint_digest':'c'*64},
            'native_runtime':{'id':'native:l','binary_sha256':'e'*64,'runtime_digest':'f'*64},
            'context_accounting':{'method':'native_template_and_tokenizer','prompt_sha256':'d'*64,'input_tokens':20,
                'requested_input_limit':context_max_tokens,'model_context_tokens':4096,'output_reservation':data['max_tokens'],'verified_against_usage':True}}


def workflow():
    with tempfile.TemporaryDirectory() as folder:
        f=Fixture(folder)
        with TestClient(f.app,base_url='http://127.0.0.1') as client:
            def request(**updates):return client.post('/v1/retrieval/answer',json={**f.payload,**updates})
            response=request();assert response.status_code==200,response.text
            result=RetrievalAnswerResult.model_validate(response.json());assert result.status=='answered'
            assert [row.document_id for row in result.candidates]==['B','C','A']
            assert result.rerank_candidate_ids==['B','C','A'] and [row.index for row in result.rerank_result.results]==[2,1,0]
            assert result.context.source_ids==['A','C'] and result.context.messages_sha256==json_hash(f.native_calls[-1]['messages'])
            assert result.claims[0].citations[0].spans==[(0,len(DOCS[0]['text']))]
            assert result.support_validation.semantic_entailment=='not_automatically_verified'
            hidden=request(return_documents=False).json();assert all('text' not in source for source in hidden['documents']) and 'messages' not in hidden['context']
            assert hidden['claims'][0]['citations'][0]['quote']==DOCS[0]['text']
            base_calls=len(f.calls)
            for body in [{'documents':[DOCS[0],DOCS[0]]},{'truncate':True},{'stream':True},{'embedding_model':'r'},{'llm_worker':'missing'}]:
                assert request(**body).status_code in {400,404,422}
                assert len(f.calls)==base_calls
            for bad in ['zero','dimension','nan']:
                f.bad_vector=bad;before=len(f.native_calls);data=request().json()
                assert data['status']=='failed' and data['claims']==[] and data['stages'][0]['error']['code']=='invalid_embedding_result'
                assert len(f.native_calls)==before and data['stages'][1]['status']=='not_run'
            f.bad_vector=None;f.failure='rerank';data=request().json()
            assert data['status']=='failed' and data['stages'][0]['status']=='completed' and data['stages'][2]['status']=='failed' and data['stages'][3]['status']=='not_run'
            f.failure=None
            invalid=[ 'not json', {'status':'answered','claims':[]}, {'status':'abstained','claims':[{'text':'bad','citations':[]}]},
                {'status':'answered','claims':[{'text':'unsupported','citations':[{'document_id':'ghost','quote':DOCS[0]['text']}]}]},
                {'status':'answered','claims':[{'text':'unsupported','citations':[{'document_id':'C','quote':DOCS[0]['text']}]}]},
                {'status':'answered','claims':[{'text':'unsupported','citations':[{'document_id':'A','quote':DOCS[0]['text'],'url':'https://invented.invalid'}]}]}]
            for value in invalid:
                f.answer=value;data=request().json()
                assert data['status']=='failed' and data['claims']==[] and data['stages'][3]['status']=='completed' and data['stages'][4]['status']=='failed'
                assert data['support_validation']['citation_integrity']=='invalid'
            f.answer={'status':'abstained','claims':[]};data=request().json()
            assert data['status']=='abstained' and data['claims']==[] and all(stage['status']=='completed' for stage in data['stages'])
            f.answer=None;f.finish='length';data=request().json();assert data['status']=='failed' and data['reasons']==['invalid_answer_output']
            f.finish='stop'
            # Literal quotation is deliberately insufficient to certify a false claim.
            f.answer={'status':'answered','claims':[{'text':'Alpha is the final label.','citations':[{'document_id':'A','quote':DOCS[0]['text']}]}]}
            data=request().json();assert data['status']=='answered' and data['support_validation']['semantic_entailment']=='not_automatically_verified'
            f.answer=None
            data=client.post('/v1/extensions/infer',json={**f.payload,'operation':'retrieval_answer'}).json();assert data['status']=='answered'
            advertised=client.get('/v1/extensions').json();entry=next(row for row in advertised['data'] if row['operation']=='retrieval_answer');assert entry['models']==['l']
            for phase in ['embeddings','rerank']:
                f.delay=phase;data=request(timeout=10).json()
                assert data['status']=='failed' and data['reasons']==['deadline_exceeded'] and data['claims']==[]
                f.delay=None
    print('PASS workflow ordering/ID mapping, partial stages, invalid-citation failure, real abstention and explicit non-entailment contracts',flush=True)


async def cancellation():
    for phase in ['embeddings','rerank','native']:
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder);f.delay=phase
            try:
                task=asyncio.create_task(f.app.state.mcp.call_tool('retrieval_answer',{'payload':f.payload}))
                assert await asyncio.to_thread(f.entered.wait,1)
                task.cancel()
                try:await asyncio.wait_for(task,2)
                except asyncio.CancelledError:pass
                else:raise AssertionError('Caller cancellation was swallowed')
                assert f.stopped.is_set()
                if phase!='native':assert not f.native_calls
            finally:
                await f.app.state.native.close();await asyncio.to_thread(f.app.state.engine.close)
    print('PASS MCP cancellation reaches every model stage and joins the dispatcher',flush=True)


def quote_bounds():
    settings=Settings(retrieval_max_quote_matches=2)
    doc=Document(id='A',text='x x x')
    answer=json.dumps({'status':'answered','claims':[{'text':'x','citations':[{'document_id':'A','quote':'x'}]}]})
    try:validate_answer(answer,'stop',{'A':doc},settings)
    except EngineError as exc:assert exc.code=='ambiguous_citation'
    else:raise AssertionError('Unbounded quote span expansion accepted')
    duplicate='{"status":"answered","status":"abstained","claims":[]}'
    try:validate_answer(duplicate,'stop',{'A':doc},settings)
    except EngineError as exc:assert exc.code=='invalid_answer_output'
    else:raise AssertionError('Duplicate producer keys accepted')


if __name__=='__main__':
    workflow();asyncio.run(cancellation());quote_bounds()
