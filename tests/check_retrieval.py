"""Controlled retrieval boundary/integration tests; no trained model or GPU."""
import asyncio
from contextlib import nullcontext, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from titanos import cli
from titanos.backends import LoadedModel
from titanos.client import Client
from titanos.compiled import CompiledModel
from titanos.config import Config, ModelSpec
from titanos.engine import EngineError
from titanos.rerank import Reranker
from titanos.retrieval_contract import RerankResult
from titanos.server import create_app
from titanos.settings import Settings, _active_settings
from titanos.text_inputs import input_limit, prepare_inputs


class Device:
    def __init__(self, value):
        self.value = str(value)
        self.type, _, index = self.value.partition(':')
        self.index = int(index) if index else None
    def __str__(self): return self.value


class Tensor:
    def __init__(self, value): self.array = np.asarray(value)
    @property
    def device(self): return 'cpu'
    @property
    def shape(self): return self.array.shape
    def __getitem__(self, key): return Tensor(self.array[key])
    def to(self, *args, **kwargs): return self
    def float(self): return Tensor(self.array.astype(np.float32))
    def cpu(self): return self
    def tolist(self): return self.array.tolist()
    def item(self): return self.array.item()
    def unsqueeze(self, axis): return Tensor(np.expand_dims(self.array, axis))
    def sum(self, axis=None): return Tensor(self.array.sum(axis))
    def clamp(self, min): return Tensor(np.maximum(self.array, min))
    def __mul__(self, other): return Tensor(self.array * other.array)
    def __truediv__(self, other): return Tensor(self.array / other.array)
    def __int__(self): return int(self.array)


class Batch(dict):
    def to(self, device): return self


class Tokenizer:
    model_max_length = 20
    truncation_side = 'right'
    def __init__(self):
        self.calls, self.vocab, self.on_call = [], {}, None
    def words(self, value):
        result=[]
        for word in value.split():
            if word not in self.vocab: self.vocab[word]=len(self.vocab)+1
            result.append(self.vocab[word])
        return result
    def num_special_tokens_to_add(self, pair=False): return 3 if pair else 2
    def __call__(self, texts, text_pair=None, truncation=False, padding=False,
                 return_attention_mask=True, add_special_tokens=True, max_length=None):
        self.calls.append({'truncation':truncation,'max_length':max_length,'pair':text_pair is not None})
        if self.on_call: self.on_call()
        ids, types = [], []
        for index,text in enumerate(texts):
            query=self.words(text)
            if text_pair is None:
                room=(max_length-2) if truncation else len(query)
                if truncation: query=query[:room] if self.truncation_side=='right' else query[-room:]
                value=[101]+query+[102] if add_special_tokens else query
                kind=[0]*len(value)
            else:
                document=self.words(text_pair[index])
                if truncation:
                    assert truncation=='only_second'
                    room=max_length-len(query)-3
                    if room<1: raise ValueError('Query cannot fit')
                    document=document[:room] if self.truncation_side=='right' else document[-room:]
                value=[101]+query+[102]+document+[102]
                kind=[0]*(len(query)+2)+[1]*(len(document)+1)
            ids.append(value);types.append(kind)
        return {'input_ids':ids,'attention_mask':[[1]*len(row) for row in ids],'token_type_ids':types}
    def pad(self, encoded, padding=True, max_length=None, return_tensors=None):
        width=max_length if padding=='max_length' else max(map(len,encoded['input_ids']))
        result={key:np.asarray([row+[0]*(width-len(row)) for row in values],dtype=np.int64) for key,values in encoded.items()}
        return Batch({key:Tensor(value) for key,value in result.items()}) if return_tensors=='pt' else result


class Model:
    def __init__(self, config):
        self.config, self.calls, self.on_call = config, [], None
        self.position=NS(num_embeddings=20,padding_idx=None)
        self.parameter=NS(device=Device('cpu'))
        self.bad_logits=False
    def to(self, device): self.parameter.device=Device(device);return self
    def eval(self): return self
    def parameters(self): return iter([self.parameter])
    def buffers(self): return iter([])
    def named_modules(self): return iter([('embeddings.position_embeddings',self.position)])
    def __call__(self, **inputs):
        self.calls.append(inputs)
        if self.on_call: self.on_call()
        ids=inputs['input_ids'].array
        if self.bad_logits: return NS(logits=Tensor([[float('nan')]]))
        # A scripted scalar output exercises ordering; it is not relevance proof.
        score=float(ids[0,list(ids[0]).index(102)+1])-10 if self.config.num_labels==1 else 0
        hidden=np.stack([np.ones_like(ids),ids],axis=-1).astype(np.float32)
        return NS(logits=Tensor([[score]]),last_hidden_state=Tensor(hidden))


class RetrievalChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='titanos-retrieval-control-')
        self.root=Path(self.temp.name)
        (self.root/'config.json').write_text('{"fixture":true}')
        (self.root/'tokenizer.json').write_text('{"fixture":true}')
        (self.root/'model.safetensors').write_bytes(b'controlled fixture, not weights')
        self.tokenizer=Tokenizer()
        self.config=NS(num_labels=1,max_position_embeddings=20)
        self.model=Model(self.config);self.info={};self.loaded_kwargs=None
        def load(*args,**kwargs):self.loaded_kwargs=kwargs;return self.model,self.info
        fake_transformers=NS(AutoConfig=NS(from_pretrained=lambda *a,**k:self.config),
                             AutoTokenizer=NS(from_pretrained=lambda *a,**k:self.tokenizer),
                             AutoModelForSequenceClassification=NS(from_pretrained=load))
        normalize=lambda tensor,p,dim:Tensor(tensor.array/np.maximum(np.linalg.norm(tensor.array,axis=dim,keepdims=True),1e-12))
        fake_torch=NS(device=Device,inference_mode=nullcontext,float32='float32',nn=NS(functional=NS(normalize=normalize)),
                      cuda=NS(is_available=lambda:False),backends=NS(mps=NS(is_available=lambda:False)))
        self.original_modules={name:sys.modules.get(name) for name in ('torch','transformers')}
        sys.modules.update(torch=fake_torch,transformers=fake_transformers)
        self.settings=Settings(embedding_max_tokens=6,rerank_max_input_tokens=12,rerank_max_documents=4)
        self.context=_active_settings.set(self.settings)
    def tearDown(self):
        _active_settings.reset(self.context)
        for name,module in self.original_modules.items():
            if module is None:sys.modules.pop(name,None)
            else:sys.modules[name]=module
        self.temp.cleanup()
    def reranker(self, **defaults):
        spec=ModelSpec(path=str(self.root),task='rerank',device='cpu',defaults=defaults)
        return Reranker(spec,'cpu',{'local_files_only':True,'use_safetensors':True,'torch_dtype':'float32'})
    def encoder(self, **defaults):
        value=LoadedModel.__new__(LoadedModel)
        value.spec=ModelSpec(path=str(self.root),task='embeddings',device='cpu',defaults=defaults)
        value.processor=self.tokenizer;value.model=self.model;value.device='cpu'
        self.config.num_labels=2
        return value
    def compiled(self, fixed=8, **defaults):
        value=CompiledModel.__new__(CompiledModel)
        value.spec=ModelSpec(path=str(self.root),task='embeddings',backend='onnx',device='cpu',defaults=defaults)
        value.processor=self.tokenizer
        value.inputs={key:{'shape':[1,fixed],'dtype':'int64'} for key in ['input_ids','attention_mask']}
        value.calls=[]
        def run(feed):value.calls.append(feed);return {'sentence_embedding':np.asarray([[1.,2.]],dtype=np.float32)}
        value.run=run
        return value

    def test_explicit_embedding_policy_on_transformers_and_compiled(self):
        for compiled in [False,True]:
            with self.subTest(compiled=compiled):
                self.model.calls.clear()
                model=self.compiled() if compiled else self.encoder()
                execute=lambda data,cancel=None:model.infer('embeddings',data,cancel=cancel) if compiled else model.embeddings(data,cancel)
                texts=['short','one two three four five six seven']
                with self.assertRaisesRegex(ValueError,'Input 1'):execute({'input':texts})
                self.assertEqual(len(model.calls if compiled else self.model.calls),0)
                result=execute({'input':texts,'max_tokens':4,'truncate':True})
                self.assertEqual(len(result['vectors']),2)
                self.assertEqual(result['usage']['prompt_tokens'],7)
                self.assertEqual(result['input_details'][1]['input_tokens'],9)
                self.assertEqual(result['input_details'][1]['processed_tokens'],4)
                self.assertTrue(result['input_details'][1]['truncated'])
                self.assertEqual(result['input_details'][1]['input_sha256'],hashlib.sha256(texts[1].encode()).hexdigest())
                self.assertEqual(result['truncation']['max_tokens'],4)
                if compiled:
                    self.assertEqual(model.calls[0]['input_ids'].shape,(1,8))
                    self.assertEqual(model.calls[1]['attention_mask'].sum(),4)
                for vector in result['vectors']:self.assertAlmostEqual(np.linalg.norm(vector),1.,places=6)
                for bad in [0,False,'4']:
                    with self.assertRaises(ValueError):execute({'input':['x'],'max_tokens':bad})

    def test_policy_precedence_unknown_context_and_cancellation(self):
        encoder=self.encoder(truncate=True,max_tokens=4)
        self.assertTrue(encoder.embeddings({'input':['a b c d']})['input_details'][0]['truncated'])
        with self.assertRaises(ValueError):encoder.embeddings({'input':['a b c d'],'truncate':False})
        cancel=threading.Event();cancel.set()
        with self.assertRaises(EngineError):encoder.embeddings({'input':['x']},cancel)
        cancel.clear();self.tokenizer.on_call=cancel.set;self.model.calls.clear()
        with self.assertRaises(EngineError):encoder.embeddings({'input':['x']},cancel)
        self.assertFalse(self.model.calls)
        self.tokenizer.on_call=None;cancel.clear();self.model.on_call=cancel.set
        with self.assertRaises(EngineError):encoder.embeddings({'input':['x']},cancel)
        self.tokenizer.model_max_length=10**30;self.config.max_position_embeddings=514
        self.model.position.num_embeddings=514;self.model.position.padding_idx=1
        self.assertEqual(input_limit(self.tokenizer,1000,self.model),512)
        self.assertFalse(Settings.from_env({'TITANOS_EMBEDDING_TRUNCATE':'false'}).embedding_truncate)
        self.assertTrue(Settings.from_env({'TITANOS_RERANK_TRUNCATE':'1'}).rerank_truncate)
        with self.assertRaises(ValueError):Settings.from_env({'TITANOS_RERANK_TRUNCATE':'maybe'})

    def test_cross_encoder_score_order_and_limits(self):
        ranker=self.reranker()
        docs=['low','high','low']
        result=ranker.infer('rerank',{'query':'query','documents':docs})
        self.assertEqual([r['index'] for r in result['results']],[1,0,2])
        self.assertEqual(result['results'][1]['relevance_score'],result['results'][2]['relevance_score'])
        self.assertLess(result['results'][0]['relevance_score'],0)  # raw logits, no probability clamp
        self.assertEqual(result['score_kind'],'cross_encoder_logit')
        self.assertEqual(len(self.model.calls),3)
        self.model.calls.clear()
        with self.assertRaises(ValueError):ranker.infer('rerank',{'query':'query','documents':['low','a b c d e f g h i']})
        self.assertFalse(self.model.calls)
        result=ranker.infer('rerank',{'query':'query','documents':['a b c d e f g h i'],'max_input_tokens':7,'truncate':True})
        detail=result['input_details'][0]
        self.assertEqual((detail['input_tokens'],detail['processed_tokens'],detail['truncated']),(13,7,True))
        self.assertEqual(detail['query_sha256'],hashlib.sha256(b'query').hexdigest())
        self.assertEqual(result['truncation']['strategy'],'only_second')
        self.model.calls.clear()
        with self.assertRaisesRegex(ValueError,'Query'):ranker.infer('rerank',{'query':'a b c d','documents':['x y'],'max_input_tokens':6,'truncate':True})
        self.assertFalse(self.model.calls)
        self.model.bad_logits=True
        with self.assertRaises(EngineError) as error:ranker.infer('rerank',{'query':'q','documents':['x']})
        self.assertEqual(error.exception.status,502)

    def test_safe_loading_and_pair_cancellation(self):
        ranker=self.reranker()
        self.assertFalse(self.loaded_kwargs['trust_remote_code']);self.assertTrue(self.loaded_kwargs['use_safetensors'])
        self.info={'missing_keys':['classifier.weight']}
        with self.assertRaises(ValueError):self.reranker()
        self.info={};self.config.num_labels=2
        with self.assertRaises(ValueError):self.reranker()
        for fields in [{'allow_pickle':True},{'trust_remote_code':True},{'parallelism':'model_sharding'}]:
            with self.assertRaises(ValueError):ModelSpec(path=str(self.root),task='rerank',**fields)
        self.config.num_labels=1;cancel=threading.Event();self.model.on_call=cancel.set
        with self.assertRaises(EngineError) as error:ranker.infer('rerank',{'query':'q','documents':['a','b']},cancel)
        self.assertEqual(error.exception.code,'cancelled');self.assertEqual(len(self.model.calls),1)

    def test_api_python_cli_mcp_and_embedding_envelope(self):
        ranker=self.reranker()
        wrapper=LoadedModel.__new__(LoadedModel);wrapper.reranker=ranker
        cfg=Config(models={'rank':ranker.spec,'embed':ModelSpec(path=str(self.root),task='embeddings',device='cpu')})
        app=create_app(cfg,output_dir=self.root/'assets',settings=self.settings)
        events=[]
        def infer(name,task,data,emit=None,cancel=None):
            app.state.engine.target.validate(name,task)
            events.append(data)
            if data.get('query')=='slow':
                assert cancel.wait(1)
            if task=='rerank':return wrapper.infer(task,data,cancel=cancel)
            return {'vectors':[[1.0,0.0]],'usage':{'prompt_tokens':3,'total_tokens':3},
                    'input_details':[{'index':0,'input_tokens':3,'processed_tokens':3,'truncated':False,'input_sha256':hashlib.sha256(b'x').hexdigest()}],
                    'truncation':{'enabled':False,'side':'right','strategy':'single_sequence','max_tokens':6}}
        app.state.engine.target.infer=infer
        with TestClient(app,base_url='http://127.0.0.1') as http:
            payload={'model':'rank','query':'query','documents':['low','high','low'],'top_n':2,'return_documents':True}
            response=http.post('/v1/rerank',json=payload);self.assertEqual(response.status_code,200,response.text)
            body=response.json();typed=RerankResult.model_validate(body)
            self.assertEqual([r.index for r in typed.results],[1,0]);self.assertEqual(typed.results[0].document,{'text':'high'})
            self.assertEqual(len(typed.input_details),3)
            no_docs=http.post('/v1/rerank',json={**payload,'return_documents':False}).json()
            self.assertNotIn('document',no_docs['results'][0])
            self.assertEqual(http.post('/v1/rerank',json={**payload,'top_n':0}).status_code,422)
            self.assertEqual(http.post('/v1/rerank',json={**payload,'documents':['a']*5}).status_code,422)
            self.assertEqual(http.post('/v1/rerank',json={**payload,'query':'slow','timeout':1}).status_code,504)
            envelope=http.post('/v1/embeddings',json={'model':'embed','input':'x','encoding_format':'base64'}).json()
            self.assertEqual(envelope['object'],'list');self.assertEqual(envelope['data'][0]['object'],'embedding')
            self.assertEqual(envelope['data'][0]['embedding'],'AACAPwAAAAA=')
            self.assertEqual(envelope['usage']['prompt_tokens'],3);self.assertIn('input_details',envelope)
            ollama=http.post('/api/embed',json={'model':'embed','input':'x','truncate':False}).json()
            self.assertIn('input_details',ollama);self.assertEqual(ollama['prompt_eval_count'],3)
            before=len(events)
            self.assertEqual(http.post('/api/embeddings',json={'model':'embed','prompt':['x','y']}).status_code,400)
            self.assertEqual(len(events),before)
            def transport(request):
                value=http.request(request.method,request.url.path,json=json.loads(request.content))
                return httpx.Response(value.status_code,json=value.json())
            with Client('http://127.0.0.1',transport=httpx.MockTransport(transport)) as client:
                value=client.rerank('rank','query',['low','high'],return_documents=True,top_n=1)
                self.assertIsInstance(value,RerankResult);self.assertEqual(value.results[0].index,1)
                with patch('titanos.client.Client',return_value=client),patch.object(sys,'argv',['titanos','rerank','rank','query','low','high','--top-n','1','--return-documents','--truncate','--max-input-tokens','8']):
                    stdout=io.StringIO()
                    with redirect_stdout(stdout):cli.main()
                    self.assertEqual(json.loads(stdout.getvalue())['results'][0]['index'],1)
            async def mcp_check():
                self.assertIn('rerank',{tool.name for tool in await app.state.mcp.list_tools()})
                for tool,extra in [('rerank',{}),('extension_infer',{'operation':'rerank'})]:
                    result=await app.state.mcp.call_tool(tool,{'payload':{**payload,**extra}})
                    self.assertFalse(result.is_error,result)
                    value=result.structured_content
                    if value is None:value=json.loads(next(c.text for c in result.content if getattr(c,'type','')=='text'))
                    value=value.get('result',value)
                    self.assertEqual(value['results'][0]['index'],1)
            asyncio.run(mcp_check())

    def test_dynamic_graph_process_policy_and_invalid_backend_metadata(self):
        graph=self.compiled(fixed=None)
        result=graph.infer('embeddings',{'input':['a','a b'],'max_tokens':6})
        self.assertEqual(graph.calls[0]['input_ids'].shape,(1,4))
        self.assertFalse(any(row['truncated'] for row in result['input_details']))
        active=_active_settings.set(Settings(embedding_max_tokens=4,embedding_truncate=True))
        try:
            self.assertTrue(self.encoder().embeddings({'input':['a b c d']})['input_details'][0]['truncated'])
            with self.assertRaises(ValueError):self.encoder(truncate=False).embeddings({'input':['a b c d']})
        finally:_active_settings.reset(active)
        self.config.num_labels=1
        ranker=self.reranker()
        payload={'model':'rank','query':'q','documents':['a','b'],'truncate':False}
        valid=ranker.infer('rerank',{'query':'q','documents':['a','b']})
        app=create_app(Config(models={'rank':ranker.spec}),output_dir=self.root/'bad-output',settings=self.settings)
        bad={'value':valid}
        app.state.engine.target.infer=lambda *a,**k:bad['value']
        with TestClient(app,base_url='http://127.0.0.1') as http:
            for change in ['indices','identity','policy','usage']:
                raw=json.loads(json.dumps(valid))
                if change=='indices':raw['results'][1]['index']=raw['results'][0]['index']
                elif change=='identity':raw['input_details'][0]['input_sha256']='0'*64
                elif change=='policy':raw['truncation']['enabled']=True
                else:raw['usage']['prompt_tokens']=0
                bad['value']=raw
                response=http.post('/v1/rerank',json=payload)
                self.assertEqual(response.status_code,502,(change,response.text))


if __name__=='__main__':unittest.main(verbosity=2)
