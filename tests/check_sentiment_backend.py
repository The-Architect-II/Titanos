"""Boundary tests with library/tensor doubles; no Torch import, model or network."""

from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from titanos.backends import LoadedModel
from titanos.config import Config, ModelSpec
from titanos.engine import EngineError
from titanos.sentiment import SentimentClassifier, checkpoint_fingerprint, resolve_label_mapping


class Device:
    def __init__(self, value):
        self.value = str(value)
        kind, _, index = self.value.partition(':')
        self.type, self.index = kind, int(index) if index else None
    def __str__(self):
        return self.value


class Tensor:
    def __init__(self, values, device='cpu'):
        self.values, self.device = values, Device(device)
    @property
    def shape(self):
        if not isinstance(self.values, list): return ()
        return (len(self.values), len(self.values[0])) if self.values and isinstance(self.values[0], list) else (len(self.values),)
    @property
    def ndim(self): return len(self.shape)
    def to(self, device=None, **unused): return Tensor(self.values, device or self.device)
    def float(self): return self
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.values
    def __len__(self): return len(self.values)
    def __getitem__(self, key): return Tensor(self.values[key], self.device)


class Batch(dict):
    def to(self, device): return Batch({k:v.to(device) for k,v in self.items()})


class Tokenizer:
    model_max_length = 16
    pad_token_id = 0
    eos_token_id = 2
    chat_template = False
    def __init__(self):
        self.count, self.calls, self.on_call = 5, [], None
    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if self.on_call: self.on_call()
        return Batch(input_ids=Tensor([list(range(self.count))]), attention_mask=Tensor([[1]*self.count]))
    def decode(self, values, **unused): return 'synthetic'


class Model:
    dtype = 'float32'
    def __init__(self, config):
        self.config, self.calls, self.on_call = config, 0, None
        self.logits = [[0.0, 1.0, 2.0]]
        self.parameter = NS(device=Device('cpu'))
        self.position = NS(num_embeddings=16, padding_idx=None)
    def to(self, device): self.parameter.device = Device(device); return self
    def eval(self): return self
    def parameters(self): return iter([self.parameter])
    def buffers(self): return iter([])
    def named_modules(self): return iter([('encoder.embeddings.position_embeddings', self.position)])
    def __call__(self, **inputs):
        self.calls += 1
        if self.on_call: self.on_call()
        return NS(logits=Tensor(self.logits)) if inputs.get('return_dict', getattr(self.config,'return_dict',True)) else (Tensor(self.logits),)
    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return Tensor([list(range(kwargs['input_ids'].shape[-1]+kwargs['max_new_tokens']))])


def fake_softmax(tensor, dim):
    assert dim == -1
    values = tensor.values[0]
    scale = max(values)
    exp = [math.exp(value-scale) for value in values]
    return Tensor([[value/sum(exp) for value in exp]])


def fake_finite(tensor):
    value = all(math.isfinite(v) for row in tensor.values for v in row)
    return NS(all=lambda:NS(item=lambda:value))


class SentimentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='titanos-sentiment-unit-')
        self.root = Path(self.temp.name)
        (self.root/'config.json').write_text('{"fixture":true}')
        (self.root/'model.safetensors').write_bytes(b'synthetic-not-a-model')
        (self.root/'tokenizer.json').write_text('{"fixture":true}')
        self.config = NS(num_labels=3, id2label={0:'negative',1:'neutral',2:'positive'}, problem_type='single_label_classification', max_position_embeddings=16)
        self.model, self.tokenizer = Model(self.config), Tokenizer()
        self.info, self.on_load, self.load_kwargs = {}, None, None
        def load(path, **kwargs):
            self.load_kwargs = kwargs
            if self.on_load: self.on_load()
            return self.model, self.info
        fake_transformers = NS(AutoConfig=NS(from_pretrained=lambda *a,**k:self.config),
                               AutoModelForSequenceClassification=NS(from_pretrained=load),
                               AutoTokenizer=NS(from_pretrained=lambda *a,**k:self.tokenizer),
                               TextStreamer=object, StoppingCriteria=object, StoppingCriteriaList=list)
        fake_torch = NS(device=Device, inference_mode=nullcontext, softmax=fake_softmax,
                        isfinite=fake_finite, is_floating_point=lambda value:False, float32='float32')
        self.patcher = patch.dict(sys.modules, {'transformers':fake_transformers, 'torch':fake_torch})
        self.patcher.start()
    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()
    def spec(self, **extra):
        return ModelSpec(path=str(self.root),task='sentiment',device='cpu',source_repo='fixture/repo',source_revision='fixture-revision',**extra)
    def classifier(self, **extra):
        return SentimentClassifier(self.spec(**extra),'cpu',{'local_files_only':True,'torch_dtype':'float32','use_safetensors':True})

    def test_labels_require_actual_meaning_and_complete_bijection(self):
        self.assertEqual(resolve_label_mapping(self.config),{'0':'negative','1':'neutral','2':'positive'})
        self.config.id2label={0:'LABEL_0',1:'LABEL_1',2:'LABEL_2'}
        with self.assertRaises(ValueError): resolve_label_mapping(self.config)
        self.assertEqual(resolve_label_mapping(self.config,{'LABEL_0':'positive','LABEL_1':'negative','LABEL_2':'neutral'})['0'],'positive')
        self.assertEqual(resolve_label_mapping(self.config,{0:'negative',1:'neutral',2:'positive'})['2'],'positive')
        for mapping in [{'0':'negative'}, {'0':'negative','1':'positive','2':'positive'}, {'0':'negative','1':'neutral','9':'positive'}, {'0':'negative','1':'neutral','2':'unknown'}]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError): resolve_label_mapping(self.config,mapping)
        for kind in ['regression','multi_label_classification']:
            self.config.problem_type=kind
            with self.assertRaises(ValueError): resolve_label_mapping(self.config,{0:'negative',1:'neutral',2:'positive'})
        self.config.problem_type=None
        for count in [True,1,4,3.0]:
            self.config.num_labels=count
            with self.assertRaises(ValueError): resolve_label_mapping(self.config)

    def test_binary_and_softmax_provenance(self):
        self.config.return_dict=False
        classifier=self.classifier()
        result=classifier.infer('sentiment',{'text':'keep exact whitespace  '})
        self.assertEqual(set(result['scores']),{'negative','neutral','positive'})
        self.assertAlmostEqual(sum(result['scores'].values()),1.0)
        self.assertEqual(result['score_kind'],'softmax_uncalibrated')
        self.assertEqual(result['input_tokens'],5)
        self.assertEqual(result['provenance']['checkpoint_digest'],checkpoint_fingerprint(self.root))
        self.assertEqual(result['provenance']['repository'],'fixture/repo')
        self.assertEqual(self.tokenizer.calls[0],('keep exact whitespace  ',{'return_tensors':'pt','truncation':False,'padding':False}))
        self.assertTrue(self.load_kwargs['local_files_only'] and self.load_kwargs['use_safetensors'])
        self.assertFalse(self.load_kwargs['trust_remote_code'])
        self.config.num_labels=2;self.config.id2label={0:'POSITIVE',1:'NEGATIVE'};self.model.logits=[[2.0,0.0]]
        result=self.classifier().infer('sentiment',{'text':'binary'})
        self.assertEqual(set(result['scores']),{'negative','positive'})
        self.assertNotIn('neutral',result['scores'])
        self.assertGreater(result['scores']['positive'],result['scores']['negative'])

    def test_limits_and_roberta_position_offset_before_forward(self):
        classifier=self.classifier(defaults={'max_input_tokens':4})
        with self.assertRaisesRegex(ValueError,'5 tokens'):classifier.infer('sentiment',{'text':'too long'})
        self.assertEqual(self.model.calls,0)
        classifier.infer('sentiment',{'text':'request overrides default','max_input_tokens':6})
        self.assertEqual(self.model.calls,1)
        self.tokenizer.count=17
        with self.assertRaises(ValueError):classifier.infer('sentiment',{'text':'physical cap','max_input_tokens':100})
        self.assertEqual(self.model.calls,1)
        for value in [True,0,-1,4.0,'4']:
            with self.subTest(value=value),self.assertRaises(ValueError):classifier.infer('sentiment',{'text':'invalid','max_input_tokens':value})
        self.config.max_position_embeddings=514;self.model.position.num_embeddings=514;self.model.position.padding_idx=1;self.tokenizer.model_max_length=10**30;self.tokenizer.count=513
        with self.assertRaisesRegex(ValueError,'max_input_tokens=512'):self.classifier().infer('sentiment',{'text':'position offset','max_input_tokens':1000})
        self.assertEqual(self.model.calls,1)

    def test_cancellation_at_tokenization_and_forward_boundaries(self):
        classifier=self.classifier();cancel=threading.Event();cancel.set()
        with self.assertRaises(EngineError) as error:classifier.infer('sentiment',{'text':'cancelled'},cancel)
        self.assertEqual((error.exception.status,error.exception.code),(499,'cancelled'))
        self.assertEqual(self.model.calls,0);self.assertEqual(len(self.tokenizer.calls),0)
        cancel.clear();self.tokenizer.on_call=cancel.set
        with self.assertRaises(EngineError):classifier.infer('sentiment',{'text':'cancel during tokenization'},cancel)
        self.assertEqual(self.model.calls,0)
        cancel.clear();self.tokenizer.on_call=None;self.model.on_call=cancel.set
        with self.assertRaises(EngineError):classifier.infer('sentiment',{'text':'cancel during forward'},cancel)
        self.assertEqual(self.model.calls,1)

    def test_invalid_inputs_and_logits(self):
        classifier=self.classifier()
        for data in [{},{'text':''},{'text':'   '},{'text':['batch']},{'text':'a','temperature':0.2}]:
            with self.subTest(data=data),self.assertRaises(ValueError):classifier.infer('sentiment',data)
        with self.assertRaises(ValueError):classifier.infer('chat',{'text':'wrong task'})
        self.assertEqual(self.model.calls,0)
        for logits in [[[float('nan'),0.0,0.0]],[[float('inf'),0.0,0.0]],[[0.0,1.0]],[[0.0,1.0,2.0],[1.0,2.0,3.0]]]:
            self.model.logits=logits
            with self.subTest(logits=logits),self.assertRaises(ValueError):classifier.infer('sentiment',{'text':'invalid output'})

    def test_fingerprint_mutation_and_incomplete_weights(self):
        before=checkpoint_fingerprint(self.root)
        (self.root/'README.md').write_text('non-runtime note')
        self.assertEqual(before,checkpoint_fingerprint(self.root))
        (self.root/'tokenizer.json').write_text('{"fixture":"changed"}')
        self.assertNotEqual(before,checkpoint_fingerprint(self.root))
        self.info={'missing_keys':['classifier.weight']}
        with self.assertRaisesRegex(ValueError,'classification weights'):self.classifier()
        self.info={};self.on_load=lambda:(self.root/'model.safetensors').write_bytes(b'changed during load')
        with self.assertRaisesRegex(ValueError,'changed while'):self.classifier()
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'../outside.safetensors'}}))
        with self.assertRaisesRegex(ValueError,'missing or external'):checkpoint_fingerprint(self.root)

    def test_registry_single_device_and_safe_loading(self):
        for values in [{'parallelism':'model_sharding'},{'parallelism':'tensor_parallel'},{'allow_pickle':True},{'trust_remote_code':True},{'load_options':{'ignore_mismatched_sizes':True}},{'load_options':{'state_dict':{}}},{'defaults':{'max_input_tokens':False}}]:
            with self.subTest(values=values),self.assertRaises(ValueError):self.spec(**values)
        with self.assertRaises(ValueError):
            Config.model_validate({'models':{'s':self.spec().model_dump()},'workers':{'w':{'models':['s'],'mode':'model_sharding','gpus':['0','1']}}})

    def test_loaded_model_worker_dispatch_and_chat_policy_isolation(self):
        loaded=LoadedModel(self.spec())
        result=loaded.infer('sentiment',{'text':'transport metadata','worker':'worker-a'})
        self.assertEqual(result['input_tokens'],5)
        self.assertEqual(loaded.details()['sentiment']['checkpoint_digest'],result['provenance']['checkpoint_digest'])
        chat=LoadedModel.__new__(LoadedModel)
        chat.spec=ModelSpec(path=str(self.root),task='chat',device='cpu',defaults={'min_score':0.8,'min_margin':0.2,'num_beams':2,'top_k':5})
        chat.device='cpu';chat.tensor_parallel=False;chat.model=self.model;chat.processor=self.tokenizer
        chat.chat({'prompt':'sampling preserved','max_tokens':2,'temperature':0.4,'top_p':0.8,'top_k':7,'repeat_penalty':1.1},None,threading.Event())
        kwargs=self.model.generation_kwargs
        self.assertNotIn('min_score',kwargs);self.assertNotIn('min_margin',kwargs)
        self.assertEqual((kwargs['num_beams'],kwargs['top_k'],kwargs['temperature'],kwargs['top_p'],kwargs['repetition_penalty']),(2,7,0.4,0.8,1.1))


if __name__=='__main__':
    unittest.main(verbosity=2)
