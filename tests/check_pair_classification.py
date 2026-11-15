"""Native-label pair contracts with loader/tensor doubles; no trained inference."""
import asyncio,copy,hashlib,io,json,sys,tempfile,threading,time,unittest
from contextlib import nullcontext,redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient

from titanos import cli
from titanos.api_plus import PairClassificationRequest
from titanos.backends import LoadedModel
from titanos.client import Client
from titanos.config import Config,ModelSpec
from titanos.engine import EngineError
from titanos.pair_classifier import PairClassifier,native_labels
from titanos.pair_contract import PairClassificationResult,pair_hash
from titanos.server import create_app
from titanos.settings import Settings,_active_settings
import check_retrieval as helpers


class Tokenizer(helpers.Tokenizer):
    def __call__(self,texts,text_pair=None,truncation=False,padding=False,return_attention_mask=True,add_special_tokens=True,max_length=None):
        if text_pair is None or truncation!='only_first':
            return super().__call__(texts,text_pair,truncation,padding,return_attention_mask,add_special_tokens,max_length)
        self.calls.append({'truncation':truncation,'max_length':max_length,'pair':True})
        if self.on_call:self.on_call()
        ids=[];types=[]
        for premise,hypothesis in zip(texts,text_pair):
            left,right=self.words(premise),self.words(hypothesis);room=max_length-len(right)-3
            if room<1:raise ValueError('Hypothesis cannot fit')
            left=left[:room] if self.truncation_side=='right' else left[-room:]
            values=[101]+left+[102]+right+[102];ids.append(values);types.append([0]*(len(left)+2)+[1]*(len(right)+1))
        return {'input_ids':ids,'attention_mask':[[1]*len(row) for row in ids],'token_type_ids':types}


class Model(helpers.Model):
    def __init__(self,config):super().__init__(config);self.values=[2.,2.,-1.];self.bad_shape=False
    def __call__(self,**inputs):
        self.calls.append(inputs)
        if self.on_call:self.on_call()
        return NS(logits=helpers.Tensor([[0.]] if self.bad_shape else [self.values]))


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='titanos-pair-double-');self.root=Path(self.temp.name)
        (self.root/'config.json').write_text('{"fixture":true}')
        (self.root/'tokenizer.json').write_text('{"fixture":true}')
        (self.root/'model.safetensors').write_bytes(b'Controlled bytes, not a trained checkpoint')
        self.config=NS(num_labels=3,problem_type=None,id2label={0:'contradiction',1:'entailment',2:'neutral'},max_position_embeddings=20,model_type='controlled')
        self.model=Model(self.config);self.tokenizer=Tokenizer();self.info={};self.after_load=lambda:None;self.kwargs={}
        def load(*args,**kwargs):self.kwargs=kwargs;self.after_load();return self.model,self.info
        transformers=NS(AutoConfig=NS(from_pretrained=lambda *args,**kwargs:self.config),AutoTokenizer=NS(from_pretrained=lambda *args,**kwargs:self.tokenizer),
                        AutoModelForSequenceClassification=NS(from_pretrained=load))
        torch=NS(device=helpers.Device,inference_mode=nullcontext,cuda=NS(is_available=lambda:False),backends=NS(mps=NS(is_available=lambda:False)))
        self.original_modules={name:sys.modules.get(name) for name in ('torch','transformers')}
        sys.modules.update(torch=torch,transformers=transformers)
        self.settings=Settings(pair_max_input_tokens=12,pair_max_pairs=4,pair_max_labels=8)
        self.context=_active_settings.set(self.settings)
    def tearDown(self):
        _active_settings.reset(self.context)
        for name,value in self.original_modules.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
        self.temp.cleanup()
    def backend(self,**defaults):
        self.spec=ModelSpec(path=str(self.root),task='pair-classification',device='cpu',defaults=defaults)
        return PairClassifier(self.spec,'cpu',{'local_files_only':True,'use_safetensors':True})
    def pairs(self):return [{'id':'caller-a','premise':'first second','hypothesis':'third'}, {'premise':'fourth','hypothesis':'fifth'}]
    def test_native_labels_logits_softmax_ids_and_tie(self):
        backend=self.backend();raw=backend.infer('pair-classification',{'pairs':self.pairs()})
        result=PairClassificationResult.model_validate({**raw,'model':'native'})
        self.assertEqual([label.label for label in result.labels],['contradiction','entailment','neutral'])
        self.assertEqual([row.id for row in result.results],['caller-a','pair-1'])
        self.assertEqual(result.results[0].label_index,0);self.assertEqual(result.results[0].label,'contradiction')
        self.assertEqual(result.results[0].logits,[2.,2.,-1.]);self.assertAlmostEqual(sum(result.results[0].scores),1)
        self.assertFalse(result.calibrated);self.assertFalse(result.semantic_label_mapping_applied)
        self.assertEqual(result.results[0].input.pair_sha256,pair_hash('first second','third'))
        ids=self.model.calls[0]['input_ids'].array[0].tolist();self.assertEqual(ids[:4],[101,1,2,102])
        self.assertTrue(self.kwargs['use_safetensors']);self.assertFalse(self.kwargs['trust_remote_code'])
    def test_generic_names_preserved_and_head_mapping_validation(self):
        self.config.num_labels=2;self.config.id2label={'1':'LABEL_1','0':'LABEL_0'};self.model.values=[-1.,1.]
        result=self.backend().infer('pair-classification',{'pairs':self.pairs()})
        self.assertEqual(result['results'][0]['label'],'LABEL_1')
        for count,kind,mapping in [(1,None,{0:'score'}),(3,'regression',{0:'a',1:'b',2:'c'}),(3,'multi_label_classification',{0:'a',1:'b',2:'c'}),
                (2,None,{0:'same',1:'same'}),(2,None,{0:'a',2:'b'}),(2,None,{False:'a',1:'b'})]:
            with self.assertRaises(ValueError):native_labels(NS(num_labels=count,problem_type=kind,id2label=mapping),8)
    def test_saturated_scores_keep_strict_probability_bounds(self):
        self.model.values=[1000.,0.,0.]
        backend=self.backend()
        raw=backend.infer('pair-classification',{'pairs':self.pairs()})
        self.assertEqual(raw['results'][0]['scores'],[1.,0.,0.])
        PairClassificationResult.model_validate({**raw,'model':'native'})
        app=create_app(Config(models={'native':self.spec}),output_dir=self.root/'score-output',settings=self.settings)
        scores=[None]
        def infer(model,task,data,emit=None,cancel=None):
            value=copy.deepcopy(raw)
            if scores[0] is not None:value['results'][0]['scores']=scores[0]
            return value
        app.state.engine.target.infer=infer
        with TestClient(app,base_url='http://127.0.0.1') as client:
            body={'model':'native','pairs':self.pairs()}
            self.assertEqual(client.post('/v1/classifications/text-pairs',json=body).status_code,200)
            for invalid in ([1.00000005,-0.00000005,0.],[1.,0.00000005,-0.00000005]):
                scores[0]=invalid
                modified=copy.deepcopy(raw);modified['results'][0]['scores']=invalid
                with self.assertRaises(ValueError):PairClassificationResult.model_validate({**modified,'model':'native'})
                self.assertEqual(client.post('/v1/classifications/text-pairs',json=body).status_code,502)
                self.assertEqual(client.post('/v1/extensions/infer',json={**body,'operation':'pair_classification'}).status_code,502)
    def test_load_binding_rejects_missing_head_changed_files_and_label_changes(self):
        self.info={'missing_keys':['classifier.weight']}
        with self.assertRaisesRegex(ValueError,'missing or incompatible'):self.backend()
        self.info={};self.after_load=lambda:(self.root/'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Checkpoint changed'):self.backend()
        self.after_load=lambda:self.config.id2label.update({0:'new name'})
        with self.assertRaisesRegex(ValueError,'mapping differs'):self.backend()
    def test_all_pairs_checked_before_forward_and_premise_only_loss_reported(self):
        backend=self.backend();pairs=[self.pairs()[0],{'premise':'a b c d e f','hypothesis':'g h'}]
        with self.assertRaisesRegex(ValueError,'exceeding'):backend.infer('pair-classification',{'pairs':pairs,'max_input_tokens':8})
        self.assertFalse(self.model.calls)
        result=PairClassificationResult.model_validate({**backend.infer('pair-classification',{'pairs':pairs,'max_input_tokens':8,'truncate':True}),'model':'native'})
        changed=result.results[1].input;self.assertEqual((changed.input_tokens,changed.processed_tokens),(11,8))
        self.assertTrue(changed.truncated);self.assertNotEqual(changed.original_token_ids_sha256,changed.processed_token_ids_sha256)
        self.assertEqual(result.truncation.strategy,'only_first');self.assertEqual(result.truncation.preserved_sequence,'hypothesis')
        sent=self.model.calls[-1]['input_ids'].array[0].tolist();self.assertEqual(sent[-3:-1],self.tokenizer.words('g h'))
        before=len(self.model.calls)
        with self.assertRaisesRegex(ValueError,'Hypothesis'):backend.infer('pair-classification',{'pairs':[{'premise':'a','hypothesis':'a b c d e'}],'max_input_tokens':8,'truncate':True})
        self.assertEqual(len(self.model.calls),before)
    def test_model_process_request_omission_and_left_truncation(self):
        backend=self.backend(max_input_tokens=8,truncate=True);pairs=[{'premise':'a b c d e f','hypothesis':'g h'}]
        result=backend.infer('pair-classification',{'pairs':pairs});self.assertTrue(result['truncation']['enabled']);self.assertEqual(result['truncation']['max_tokens'],8)
        with self.assertRaises(ValueError):backend.infer('pair-classification',{'pairs':pairs,'truncate':False})
        self.tokenizer.truncation_side='left';self.model.calls.clear()
        result=backend.infer('pair-classification',{'pairs':pairs,'max_input_tokens':9})
        self.assertEqual(result['truncation']['side'],'left');self.assertEqual(result['truncation']['max_tokens'],9)
        self.assertEqual(self.model.calls[0]['input_ids'].array[0][1:5].tolist(),self.tokenizer.words('c d e f'))
        with self.assertRaises(ValueError):backend.infer('pair-classification',{'pairs':pairs,'max_input_tokens':13})
        token=_active_settings.set(Settings(pair_max_input_tokens=8,pair_truncate=True))
        try:
            backend=self.backend()
            self.assertTrue(backend.infer('pair-classification',{'pairs':pairs})['truncation']['enabled'])
            backend.spec.defaults['truncate']=False
            with self.assertRaises(ValueError):backend.infer('pair-classification',{'pairs':pairs})
            self.assertTrue(backend.infer('pair-classification',{'pairs':pairs,'truncate':True})['truncation']['enabled'])
        finally:_active_settings.reset(token)
    def test_cancellation_and_bad_logits_never_return_partial_success(self):
        backend=self.backend();cancel=threading.Event();self.tokenizer.on_call=cancel.set
        with self.assertRaises(EngineError):backend.infer('pair-classification',{'pairs':self.pairs()},cancel)
        self.assertFalse(self.model.calls);self.tokenizer.on_call=None;cancel.clear();self.model.on_call=cancel.set
        with self.assertRaises(EngineError):backend.infer('pair-classification',{'pairs':self.pairs()},cancel)
        self.assertEqual(len(self.model.calls),1)
        self.model.on_call=None;self.model.values=[float('nan'),1.,2.]
        with self.assertRaises(EngineError):backend.infer('pair-classification',{'pairs':self.pairs()})
        self.model.bad_shape=True
        with self.assertRaises(EngineError):backend.infer('pair-classification',{'pairs':self.pairs()})
    def test_registration_limits_and_transport_worker_filter(self):
        for value in [{'label_mapping':{'0':'entailment'}},{'temperature':.5},{'max_input_tokens':False},{'truncate':1}]:
            with self.assertRaises(ValueError):ModelSpec(path='local',task='pair-classification',defaults=value)
        for values in [{'backend':'onnx'},{'allow_pickle':True},{'trust_remote_code':True},{'parallelism':'model_sharding'},{'load_options':{'id2label':{}}}]:
            with self.assertRaises(ValueError):ModelSpec(path='local',task='pair-classification',**values)
        with self.assertRaises(ValueError):Settings(pair_max_labels=1)
        with self.assertRaises(ValueError):Settings(native_queue_timeout=10**1000)
        self.assertEqual(Settings(pair_max_input_tokens=10**1000).pair_max_input_tokens,10**1000)
        backend=self.backend();loaded=LoadedModel.__new__(LoadedModel);loaded.pair_classifier=backend
        value=loaded.infer('pair-classification',{'pairs':self.pairs(),'worker':None});self.assertEqual(len(value['results']),2)
    def test_http_extension_ids_bounds_and_malformed_backend(self):
        backend=self.backend();app=create_app(Config(models={'native':self.spec}),output_dir=self.root/'output',settings=self.settings)
        recorded=[];bad=[False]
        def infer(model,task,data,emit=None,cancel=None):
            recorded.append(copy.deepcopy(data));value=backend.infer(task,{key:val for key,val in data.items() if key!='worker'},cancel)
            if bad[0]:value['results'][0]['input']['premise_sha256']='0'*64
            return value
        app.state.engine.target.infer=infer
        with TestClient(app,base_url='http://127.0.0.1') as client:
            body={'model':'native','pairs':self.pairs()};response=client.post('/v1/classifications/text-pairs',json=body)
            self.assertEqual(response.status_code,200,response.text);self.assertEqual(response.json()['results'][0]['id'],'caller-a')
            self.assertIsNone(recorded[-1]['max_input_tokens']);self.assertIsNone(recorded[-1]['truncate'])
            response=client.post('/v1/extensions/infer',json={**body,'operation':'pair_classification','truncate':False})
            self.assertEqual(response.status_code,200);self.assertIs(recorded[-1]['truncate'],False)
            for changes in [{'pairs':[]},{'pairs':[self.pairs()[0]]*2},{'max_input_tokens':True},{'max_input_tokens':13},{'pairs':[{'premise':'x','hypothesis':'y'},{'id':'pair-0','premise':'x','hypothesis':'y'}]}]:
                before=len(recorded);self.assertEqual(client.post('/v1/classifications/text-pairs',json={**body,**changes}).status_code,422);self.assertEqual(len(recorded),before)
            bad[0]=True;self.assertEqual(client.post('/v1/classifications/text-pairs',json=body).status_code,502)
            info=client.get('/v1/extensions').json();self.assertIn('pair_classification',info)
            self.assertEqual(client.get('/api/info').json()['settings']['pair_max_pairs']['value'],4)
    def test_python_cli_and_typed_mcp_omission(self):
        backend=self.backend();result=PairClassificationResult.model_validate({**backend.infer('pair-classification',{'pairs':self.pairs()}),'model':'native'})
        seen=[]
        def handle(request):seen.append(json.loads(request.content));return httpx.Response(200,json=result.model_dump(mode='json'))
        with Client(transport=httpx.MockTransport(handle)) as client:
            self.assertIsInstance(client.classify_pairs('native',self.pairs()),PairClassificationResult)
            self.assertNotIn('truncate',seen[-1]);self.assertNotIn('max_input_tokens',seen[-1])
            client.classify_pairs('native',self.pairs(),truncate=False,max_input_tokens=8)
            self.assertIs(seen[-1]['truncate'],False);self.assertEqual(seen[-1]['max_input_tokens'],8)
        file=self.root/'pairs.json';file.write_text(json.dumps(self.pairs()))
        with patch.object(Client,'classify_pairs',return_value=result) as call,redirect_stdout(io.StringIO()):
            cli.main(['classify-pairs','native','--pairs-file',str(file),'--no-truncate','--max-input-tokens','8','--deadline-ms','500'])
        self.assertIs(call.call_args.kwargs['truncate'],False);self.assertEqual(call.call_args.kwargs['timeout'],500)
        app=create_app(Config(models={'native':self.spec}),output_dir=self.root/'mcp',settings=self.settings)
        app.state.engine.target.infer=lambda model,task,data,emit=None,cancel=None:backend.infer(task,{key:value for key,value in data.items() if key!='worker'},cancel)
        async def invoke():
            reply=await app.state.mcp.call_tool('classify_pairs',{'payload':{'model':'native','pairs':self.pairs()}})
            self.assertFalse(reply.is_error);data=reply.structured_content;data=data.get('result',data)
            self.assertEqual(data['object'],'text_pair.classification')
        try:asyncio.run(invoke())
        finally:app.state.engine.close();asyncio.run(app.state.native.close())
    def test_mcp_cancellation_reaches_classifier_and_joins_before_return(self):
        backend=self.backend();entered=threading.Event();stopped=threading.Event()
        app=create_app(Config(models={'native':self.spec}),output_dir=self.root/'cancel',settings=self.settings)
        def infer(model,task,data,emit=None,cancel=None):
            def wait():entered.set();cancel.wait(2);stopped.set()
            self.model.on_call=wait
            return backend.infer(task,{key:value for key,value in data.items() if key!='worker'},cancel)
        app.state.engine.target.infer=infer
        async def invoke():
            task=asyncio.create_task(app.state.mcp.call_tool('classify_pairs',{'payload':{'model':'native','pairs':self.pairs()}}))
            self.assertTrue(await asyncio.to_thread(entered.wait,1));task.cancel()
            with self.assertRaises(asyncio.CancelledError):await asyncio.wait_for(task,2)
            self.assertTrue(stopped.is_set());self.assertEqual(len(self.model.calls),1)
        try:asyncio.run(invoke())
        finally:app.state.engine.close();asyncio.run(app.state.native.close())


if __name__=='__main__':unittest.main(verbosity=2)
