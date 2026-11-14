"""Actual GenAI adapter forwarding with synthetic vendor boundaries; no model imports."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import copy,json,os,sys,tempfile,threading

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from titanos.genai import GenAIModel
from titanos.settings import Settings,_active_settings

class Tokenizer:
    def __init__(self):self.templates=[]
    def encode(self,prompt):return NS(input_ids=NS(shape=(1,2)))
    def apply_chat_template(self,messages,**kwargs):self.templates.append((messages,kwargs));return 'formatted conversation'

class Pipeline:
    def __init__(self,chunks=(),text='synthetic',reason='STOP'):
        self.calls=[];self.chunks=chunks;self.text=text;self.reason=reason;self.statuses=[]
    def generate(self,prompts,**kwargs):
        self.calls.append((prompts,kwargs))
        for chunk in self.chunks:
            state=kwargs['streamer'](chunk);self.statuses.append(state)
            if state!='RUNNING':break
        return NS(texts=[self.text],finish_reasons=[self.reason],perf_metrics=NS(get_num_input_tokens=lambda:2,get_num_generated_tokens=lambda:3))

def native_config():return NS(**{key:None for key in ['max_new_tokens','do_sample','apply_chat_template','num_return_sequences','echo','temperature','top_p','top_k','min_p','rng_seed','repetition_penalty','ignore_eos','stop_strings']})
vendor=NS(GenerationConfig=native_config,StreamingStatus=NS(CANCEL='CANCEL',STOP='STOP',RUNNING='RUNNING'),GenerationFinishReason=NS(LENGTH='LENGTH'))
cases=[]
def model(defaults=None,pipeline=None):
    value=object.__new__(GenAIModel);value.spec=NS(defaults=copy.deepcopy(defaults or {}),source_repo='fixture/repo',source_revision='fixture-revision')
    value.device='CPU';value.context=1024;value.tokenizer=Tokenizer();value.model=pipeline or Pipeline();return value

def infer(defaults,data,pipeline=None,emit=None,cancel=None):
    value=model(defaults,pipeline);before=copy.deepcopy(value.spec.defaults);original=copy.deepcopy(data)
    with patch.dict(sys.modules,{'openvino_genai':vendor}):result=value.infer('chat',data,emit,cancel)
    assert value.spec.defaults==before and data==original
    return value,result,value.model.calls[-1][1]

defaults={'max_tokens':11,'max_completion_tokens':13,'max_output_tokens':17,'top_k':5,'min_p':.2,'seed':101,'repeat_penalty':1.1,'stop':'OLD'}
request={'prompt':'fixture','max_tokens':32,'temperature':.4,'top_p':.8,'top_k':19,'min_p':.7,'seed':303,'repeat_penalty':1.4,'stop':'END'}
token=_active_settings.set(Settings(threads=12,default_max_tokens=256,default_temperature=.7,default_top_p=1.0))
try:
    value,result,kwargs=infer(defaults,request)
    assert {k:kwargs[k] for k in ['max_new_tokens','temperature','top_p','top_k','min_p','rng_seed','repetition_penalty']}=={'max_new_tokens':32,'temperature':.4,'top_p':.8,'top_k':19,'min_p':.7,'rng_seed':303,'repetition_penalty':1.4}
    assert not {'max_tokens','max_completion_tokens','max_output_tokens','seed','repeat_penalty','stop'}&kwargs.keys()
    assert (kwargs['num_return_sequences'],kwargs['apply_chat_template'],kwargs['echo'])==(1,False,False)
    assert result['usage']=={'prompt_tokens':2,'completion_tokens':3,'total_tokens':5}
    cases.append('reviewer_repro_request_overrides_defaults_and_canonical_aliases_never_leak')
    _,_,kwargs=infer({},request);assert (kwargs['top_k'],kwargs['min_p'],kwargs['repetition_penalty'],kwargs['rng_seed'])==(19,.7,1.4,303)
    cases.append('request_only_sampling_controls_forwarded')
    _,_,kwargs=infer(defaults,{'prompt':'fixture'})
    assert (kwargs['max_new_tokens'],kwargs['top_k'],kwargs['min_p'],kwargs['rng_seed'],kwargs['repetition_penalty'])==(11,5,.2,101,1.1)
    cases.append('omitted_request_inherits_canonical_defaults')
    _,_,kwargs=infer({'top_k':5,'min_p':.2,'seed':101,'temperature':.6},{'prompt':'fixture','temperature':0,'top_k':0,'min_p':0,'seed':0})
    assert (kwargs['do_sample'],kwargs['top_k'],kwargs['min_p'],kwargs['rng_seed'])==(False,0,0,0) and not {'temperature','top_p'}&kwargs.keys()
    cases.append('explicit_zero_survives_without_sampling_fallback')
    _,_,kwargs=infer({'temperature':None,'top_p':None},{'prompt':'fixture','top_k':None})
    assert (kwargs['temperature'],kwargs['top_p'],kwargs['max_new_tokens'])==(.7,1.0,256)
    cases.append('None_is_omission_not_zero')
    _,_,kwargs=infer({'rng_seed':5,'repetition_penalty':1.1,'ignore_eos':True},{'prompt':'fixture','seed':0,'repeat_penalty':1.7})
    assert (kwargs['rng_seed'],kwargs['repetition_penalty'],kwargs['ignore_eos'])==(0,1.7,True)
    cases.append('canonical_request_overrides_native_default_aliases')
    _,_,kwargs=infer({'max_new_tokens':23},{'prompt':'fixture'})
    assert kwargs['max_new_tokens']==23
    cases.append('native_token_default_resolves_to_same_context_checked_limit')
    _,_,kwargs=infer({'max_tokens':23},{'prompt':'fixture','max_completion_tokens':31})
    assert kwargs['max_new_tokens']==31 and 'max_completion_tokens' not in kwargs
    cases.append('request_token_alias_precedes_model_canonical_default')
    pipeline=Pipeline(('before E','ND after'),'before END after');emitted=[]
    _,result,kwargs=infer(defaults,request,pipeline,emitted.append)
    assert emitted==['before '] and result['text']=='before ' and pipeline.statuses==['RUNNING','STOP'] and result['finish_reason']=='stop'
    cases.append('request_stop_override_cross_chunk_stream_and_text_trim_unchanged')
    pipeline=Pipeline(('before OLD after',),'before OLD after');emitted=[]
    _,result,kwargs=infer({'stop':'OLD'},{'prompt':'fixture'},pipeline,emitted.append)
    assert emitted==['before '] and result['text']=='before ' and not {'stop','stop_strings'}&kwargs.keys()
    cases.append('model_default_stop_stays_in_wrapper')
    pipeline=Pipeline(('before OLD after',),'before OLD after');emitted=[]
    _,result,kwargs=infer({'stop':'OLD','stop_strings':{'OLD'}},{'prompt':'fixture','stop':[]},pipeline,emitted.append)
    assert result['text']=='before OLD after' and emitted==['before OLD after'] and 'stop_strings' not in kwargs
    cases.append('explicit_empty_stop_clears_canonical_and_native_default')
    event=threading.Event();event.set();emitted=[];pipeline=Pipeline(('not emitted',),'synthetic')
    infer({}, {'prompt':'fixture'},pipeline,emitted.append,event)
    assert pipeline.statuses==['CANCEL'] and emitted==[]
    cases.append('cancellation_stream_contract_unchanged')
    value=model();value.context=10
    with patch.dict(sys.modules,{'openvino_genai':vendor}):
        try:value.infer('chat',{'prompt':'fixture','max_tokens':9})
        except ValueError as exc:assert 'context' in str(exc)
        else:raise AssertionError('Physical context overflow accepted')
    assert not value.model.calls;cases.append('physical_context_overflow_still_rejects_before_generate')
    for label,defs,data in [('unsupported_portable_request',{}, {'prompt':'fixture','frequency_penalty':.5}),('unknown_request',{}, {'prompt':'fixture','invented_sampler':0}),('unsupported_portable_default',{'size':'512x512'},{'prompt':'fixture'}),('unknown_native_default',{'invented_sampler':0},{'prompt':'fixture'}),('request_multiple_results',{}, {'prompt':'fixture','n':2}),('default_multiple_results',{'n':2},{'prompt':'fixture'})]:
        value=model(defs)
        with patch.dict(sys.modules,{'openvino_genai':vendor}):
            try:value.infer('chat',data)
            except ValueError:pass
            else:raise AssertionError('Unsupported control silently accepted: '+label)
        assert value.model.calls==[];cases.append(label)
    _,_,kwargs=infer({'min_score':.8,'min_margin':.2,'sentiment_prompt_version':'label-v2','sentiment_decoding':'validate','retrieval_prompt_version':'retrieval-answer-v2'}, {'prompt':'fixture'})
    assert not {'min_score','min_margin','sentiment_prompt_version','sentiment_decoding','retrieval_prompt_version'}&kwargs.keys()
    cases.append('upstream_wrapper_defaults_not_native_kwargs')
    with patch.object(vendor,'GenerationConfig',lambda:NS(**vars(native_config()),future_native_control=False)):
        _,_,kwargs=infer({'future_native_control':True},{'prompt':'fixture'})
    assert kwargs['future_native_control'] is True;cases.append('new_native_property_discovered_without_vendor_allowlist')
    def older_config():
        result=native_config();del result.min_p;return result
    with patch.object(vendor,'GenerationConfig',older_config),patch.dict(sys.modules,{'openvino_genai':vendor}):
        value=model()
        try:value.infer('chat',{'prompt':'fixture','min_p':0})
        except ValueError as exc:assert 'min_p' in str(exc)
        else:raise AssertionError('Unsupported portable field was silently ignored')
        assert value.model.calls==[]
    cases.append('portable_control_missing_from_loaded_runtime_rejects_even_zero')
    value,result,kwargs=infer({}, {'messages':[{'role':'user','content':[{'type':'text','text':'hello'}]}]})
    assert value.tokenizer.templates==[([{'role':'user','content':'hello'}],{'add_generation_prompt':True})] and kwargs['apply_chat_template'] is False
    cases.append('chat_template_contract_unchanged')
    # Constructor uses explicit Settings12 even when raw environment says4;
    # explicit native load option wins over both. All vendor/compiled APIs are doubles.
    with tempfile.TemporaryDirectory(prefix='genai-construct-fixture-') as tmp:
        for name in ['openvino_model.xml','openvino_model.bin','openvino_tokenizer.xml','openvino_detokenizer.xml']:(Path(tmp)/name).write_bytes(b'fixture')
        calls=[]
        def construct(path,device,**options):calls.append(options);return NS(get_tokenizer=lambda:Tokenizer())
        fake_vendor=NS(LLMPipeline=construct);fake_ov=NS(Core=lambda:NS(available_devices=['CPU','GPU']))
        with patch.dict(os.environ,{'TITANOS_THREADS':'4'}),patch.dict(sys.modules,{'openvino':fake_ov,'openvino_genai':fake_vendor,'titanos.compiled':NS(compiled_device=lambda backend,device:device)}):
            for options,device in [({},'CPU'),({'INFERENCE_NUM_THREADS':7},'CPU'),({},'GPU')]:
                spec=NS(backend='openvino-genai',device=device,path=tmp,load_options=options)
                GenAIModel(spec);assert spec.load_options==options
        assert calls[0]['INFERENCE_NUM_THREADS']==12 and calls[0]['INFERENCE_PRECISION_HINT']=='f32'
        assert calls[1]['INFERENCE_NUM_THREADS']==7 and 'INFERENCE_NUM_THREADS' not in calls[2]
        cases.extend(['CPU_constructor_Settings12_over_environment4','CPU_explicit_load_option7_preserved','GPU_load_options_unchanged'])
finally:_active_settings.reset(token)

assert not any(name in sys.modules for name in ['openvino','openvino_genai','torch','transformers'])
print(json.dumps({'status':'PASSED_GENAI_ADAPTER_SYNTHETIC_BOUNDARIES','cases':cases,'case_count':len(cases),'actual_native_imports':0,'model_loads':0,'GPU_calls':0}))
