"""One bounded inline-corpus workflow; storage and semantic entailment are not implied."""
import hashlib
import json
import math
import time

from .engine import EngineError
from .retrieval_contract import InputDetail, TruncationInfo
from .retrieval_workflow_contract import (Candidate, Citation, Claim, ContextAccounting, ContextTrace, ModelProvenance,
    RetrievalAnswerResult, Source, Stage, StageError, StageUsage)
from .settings import current_settings
from .retrieval_prompts import INSTRUCTIONS, resolve_version

INSTRUCTION = INSTRUCTIONS['retrieval-answer-v2']


def text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def json_hash(value):
    return text_hash(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False))


def check_budget(cancel, deadline):
    if cancel is not None and cancel.is_set():
        raise EngineError('Caller cancelled retrieval workflow',499,'cancelled')
    if time.monotonic() >= deadline:
        raise EngineError('Retrieval workflow exceeded its shared deadline',504,'deadline_exceeded')


def answer_schema(ids, settings):
    citation={'type':'object','properties':{'document_id':{'type':'string','enum':ids},
        'quote':{'type':'string','minLength':1,'maxLength':settings.retrieval_max_quote_chars}},
        'required':['document_id','quote'],'additionalProperties':False}
    claim={'type':'object','properties':{'text':{'type':'string','minLength':1},
        'citations':{'type':'array','items':citation,'minItems':1,'maxItems':settings.retrieval_max_citations_per_claim}},
        'required':['text','citations'],'additionalProperties':False}
    return {'type':'object','properties':{'status':{'type':'string','enum':['answered','abstained']},
        'claims':{'type':'array','items':claim,'maxItems':settings.retrieval_max_claims}},
        'required':['status','claims'],'additionalProperties':False}


def validate_answer(text, finish_reason, documents, settings):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate key')
            result[key]=value
        return result
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON')
    try:
        if finish_reason!='stop' or not isinstance(text,str):raise ValueError('Incomplete output')
        value=json.loads(text,object_pairs_hook=pairs,parse_constant=invalid_constant)
        if not isinstance(value,dict) or set(value)!={'status','claims'} or value['status'] not in ['answered','abstained']:
            raise ValueError('Invalid answer envelope')
        if not isinstance(value['claims'],list) or len(value['claims'])>settings.retrieval_max_claims:
            raise ValueError('Invalid claims')
        if (value['status']=='abstained') != (not value['claims']):raise ValueError('Contradictory answer status')
    except (ValueError,TypeError,RecursionError) as exc:
        raise EngineError('LLM returned malformed, truncated or contradictory answer output',502,'invalid_answer_output') from exc
    claims=[]
    for item in value['claims']:
        if (not isinstance(item,dict) or set(item)!={'text','citations'} or not isinstance(item['text'],str)
                or not item['text'].strip() or len(item['text'])>settings.max_text_chars or not isinstance(item['citations'],list)
                or not 1<=len(item['citations'])<=settings.retrieval_max_citations_per_claim):
            raise EngineError('LLM returned an invalid claim/citation shape',502,'invalid_answer_output')
        try:item['text'].encode('utf-8')
        except UnicodeEncodeError as exc:
            raise EngineError('LLM claim text is not valid Unicode scalar text',502,'invalid_answer_output') from exc
        citations=[]
        for citation in item['citations']:
            if not isinstance(citation,dict) or set(citation)!={'document_id','quote'}:
                raise EngineError('Citation fields do not match the source-reference contract',502,'invalid_citation')
            identity,quote=citation['document_id'],citation['quote']
            if (not isinstance(identity,str) or identity not in documents or not isinstance(quote,str)
                    or not quote.strip() or len(quote)>settings.retrieval_max_quote_chars):
                raise EngineError('Citation must name a supplied context ID and a nonempty bounded quote',502,'invalid_citation')
            document=documents[identity];spans=[];start=0
            while True:
                index=document.text.find(quote,start)
                if index<0:break
                spans.append((index,index+len(quote)));start=index+1
                if len(spans)>settings.retrieval_max_quote_matches:
                    raise EngineError('Citation quote occurs too often to identify a bounded set of source spans',502,'ambiguous_citation')
            if not spans:
                raise EngineError('Citation quote does not occur literally in its named context document',502,'invalid_citation')
            citations.append(Citation(document_id=identity,quote=quote,document_sha256=text_hash(document.text),quote_sha256=text_hash(quote),spans=spans))
        claims.append(Claim(text=item['text'],citations=citations))
    return value['status'],claims


def cosine_candidates(response, texts, documents):
    try:
        data=response['data'];details=[InputDetail.model_validate(row) for row in response['input_details']]
        policy=TruncationInfo.model_validate(response['truncation'])
        if len(data)!=len(texts) or len(details)!=len(texts) or policy.enabled or policy.strategy!='single_sequence':raise ValueError('Incomplete or truncated embedding batch')
        vectors=[];dimension=None
        for index,(item,detail) in enumerate(zip(data,details)):
            vector=item['embedding']
            if item['index']!=index or detail.index!=index or detail.input_sha256!=text_hash(texts[index]) or detail.truncated or detail.input_tokens!=detail.processed_tokens or detail.processed_tokens>policy.max_tokens:
                raise ValueError('Embedding input identity mismatch')
            if not isinstance(vector,list) or not vector or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in vector):
                raise ValueError('Embedding vector is not finite')
            if dimension is None:dimension=len(vector)
            if len(vector)!=dimension:raise ValueError('Embedding dimensions differ')
            norm=math.sqrt(math.fsum(v*v for v in vector))
            if not math.isfinite(norm) or norm<=0:raise ValueError('Embedding norm is invalid')
            vectors.append([v/norm for v in vector])
        count=sum(row.processed_tokens for row in details)
        if response['usage']['prompt_tokens']!=count or response['usage']['total_tokens']!=count:raise ValueError('Embedding usage differs from processed inputs')
        provenance=ModelProvenance.model_validate(response['provenance'])
        if provenance.checkpoint_digest is None or provenance.backend!='transformers' or not provenance.device or provenance.pooling!='attention_mask_mean' or provenance.normalization!='l2':
            raise ValueError('Loaded encoder provenance/pooling is unavailable')
    except (KeyError,ValueError,TypeError,OverflowError) as exc:
        raise EngineError('Embedding result lacks consistent finite vectors, input fidelity or loaded encoder provenance',502,'invalid_embedding_result') from exc
    candidates=[Candidate(document_id=doc.id,original_index=index,text_sha256=text_hash(doc.text),
        cosine_similarity=math.fsum(a*b for a,b in zip(vectors[0],vectors[index+1]))) for index,doc in enumerate(documents)]
    candidates.sort(key=lambda item:(-item.cosine_similarity,item.original_index))
    return candidates,details,policy,provenance,json_hash([row['embedding'] for row in data])


def execute(payload, *, config, embed, rerank, native_complete, prepare_chat, validate_worker, cancel, deadline):
    settings=current_settings()
    roles=[(payload.embedding_model,'embeddings','transformers',payload.embedding_worker),
        (payload.rerank_model,'rerank','transformers',payload.rerank_worker),
        (payload.llm_model,'chat','llama-cpp',payload.llm_worker)]
    for model,task,backend,worker in roles:
        spec=config.models.get(model)
        if spec is None:raise EngineError('Unknown workflow model',404,'model_not_found')
        if spec.task!=task or spec.backend!=backend or spec.parallelism!='single' or spec.allow_pickle or spec.trust_remote_code:
            raise EngineError('Workflow requires a safe Transformers encoder/scalar reranker and owned native chat model',400,'unsupported_capability')
        validate_worker(model,worker)
    try:
        prompt_version=resolve_version(payload.prompt_version,config.models[payload.llm_model].defaults,settings)
    except ValueError as exc:
        raise EngineError(str(exc)) from exc
    # Resolve/validate static generation controls before any model work.
    prepare_chat([{'role':'user','content':payload.query}])
    texts=[payload.query]+[doc.text for doc in payload.documents]
    stages=[Stage(name=name,model=model,requested_worker=worker) for name,model,worker in [
        ('embedding',payload.embedding_model,payload.embedding_worker),('selection',None,None),
        ('rerank',payload.rerank_model,payload.rerank_worker),('generation',payload.llm_model,payload.llm_worker),('validation',None,None)]]
    sources=[Source(document_id=doc.id,original_index=index,text_sha256=text_hash(doc.text),title=doc.title,source_uri=doc.source_uri,
        text=doc.text if payload.return_documents else None) for index,doc in enumerate(payload.documents)]
    result=RetrievalAnswerResult.model_construct(schema_version='1' if prompt_version=='retrieval-answer-v1' else '2',status='failed',query_sha256=text_hash(payload.query),
        corpus_sha256=json_hash([doc.model_dump() for doc in payload.documents]),documents=sources,stages=stages)
    stage=stages[0];began=time.monotonic()
    try:
        check_budget(cancel,deadline)
        stage.input_sha256=json_hash(texts)
        embedded=embed(texts)
        candidates,details,policy,provenance,digest=cosine_candidates(embedded,texts,payload.documents)
        check_budget(cancel,deadline)
        stage.usage=StageUsage.model_validate(embedded['usage']);stage.status='completed';stage.output_sha256=digest;stage.provenance=provenance
        stage.elapsed_seconds=time.monotonic()-began;result.embedding_input_details=details;result.embedding_truncation=policy
        stage=stages[1];began=time.monotonic();check_budget(cancel,deadline)
        result.candidates=candidates[:payload.retrieve_top_k or settings.retrieval_top_k]
        result.rerank_candidate_ids=[row.document_id for row in result.candidates]
        stage.input_sha256=digest;stage.output_sha256=json_hash([row.model_dump() for row in result.candidates]);stage.status='completed'
        stage.elapsed_seconds=time.monotonic()-began
        stage=stages[2];began=time.monotonic();check_budget(cancel,deadline)
        selected=[payload.documents[row.original_index] for row in result.candidates]
        stage.input_sha256=json_hash({'query':payload.query,'documents':[doc.text for doc in selected]})
        ranked=rerank([doc.text for doc in selected]);result.rerank_result=ranked
        check_budget(cancel,deadline)
        stage.provenance=ModelProvenance.model_validate(ranked.provenance.model_dump());stage.usage=StageUsage.model_validate(ranked.usage.model_dump())
        stage.output_sha256=json_hash(ranked.model_dump());stage.status='completed';stage.elapsed_seconds=time.monotonic()-began
        context=[selected[row.index] for row in ranked.results[:payload.context_top_n or settings.retrieval_top_n]]
        stage=stages[3];began=time.monotonic();check_budget(cancel,deadline)
        messages=[{'role':'system','content':INSTRUCTIONS[prompt_version]},{'role':'user','content':json.dumps({'query':payload.query,
            'documents':[{'document_id':doc.id,'text':doc.text} for doc in context]},ensure_ascii=False,separators=(',', ':'))}]
        data=prepare_chat(messages)
        data['response_format']={'type':'json_schema','json_schema':{'name':'retrieval_answer_v1','strict':True,'schema':answer_schema([doc.id for doc in context],settings)}}
        result.context=ContextTrace(prompt_version=prompt_version,messages_sha256=json_hash(messages),source_ids=[doc.id for doc in context],messages=messages if payload.return_documents else None)
        stage.input_sha256=result.context.messages_sha256
        raw=native_complete(data,deadline,cancel,payload.context_max_tokens or settings.retrieval_context_tokens)
        check_budget(cancel,deadline)
        accounting=raw['context_accounting']
        counts=[accounting[key] for key in ['input_tokens','requested_input_limit','model_context_tokens','output_reservation']]
        if (accounting.get('method')!='native_template_and_tokenizer' or accounting.get('verified_against_usage') is not True
                or any(type(count) is not int or count<1 for count in counts)
                or counts[0]>counts[1] or counts[0]+counts[3]>counts[2] or raw['usage']['prompt_tokens']!=counts[0]):
            raise EngineError('Native context accounting does not establish full input use',502,'invalid_context_accounting')
        result.context.accounting=ContextAccounting.model_validate(accounting)
        stage.provenance=ModelProvenance.model_validate(raw['provenance'])
        if stage.provenance.backend!='llama-cpp' or not stage.provenance.device or stage.provenance.checkpoint_digest is None:
            raise EngineError('Native model provenance is incomplete',502,'invalid_backend_response')
        runtime=raw['native_runtime']
        binding=Stage.model_validate({**stage.model_dump(),'worker':runtime['id'],'binary_sha256':runtime['binary_sha256'],'runtime_digest':runtime['runtime_digest']})
        if not binding.worker or binding.binary_sha256 is None or binding.runtime_digest is None:
            raise EngineError('Native runtime binding is incomplete',502,'invalid_backend_response')
        stage.worker=binding.worker;stage.binary_sha256=binding.binary_sha256;stage.runtime_digest=binding.runtime_digest
        if payload.llm_worker is not None and stage.worker!=payload.llm_worker:
            raise EngineError('Native result used a different worker assignment',502,'invalid_backend_response')
        stage.usage=StageUsage.model_validate(raw['usage']);stage.output_sha256=text_hash(raw['text']) if isinstance(raw.get('text'),str) else None
        stage.status='completed';stage.elapsed_seconds=time.monotonic()-began
        stage=stages[4];began=time.monotonic();check_budget(cancel,deadline);stage.input_sha256=stages[3].output_sha256
        status,claims=validate_answer(raw.get('text'),raw.get('finish_reason'),{doc.id:doc for doc in context},settings)
        check_budget(cancel,deadline)
        result.status=status;result.claims=claims;result.reasons=['model_abstained_from_selected_context'] if status=='abstained' else []
        result.support_validation.citation_integrity='verified' if status=='answered' else 'not_applicable'
        stage.status='completed';stage.output_sha256=json_hash([claim.model_dump() for claim in claims]);stage.elapsed_seconds=time.monotonic()-began
    except (EngineError,ValueError,TypeError,KeyError,OverflowError) as exc:
        error=exc if isinstance(exc,EngineError) else EngineError('Invalid workflow stage response',502,'invalid_backend_response')
        stage.status='failed';stage.error=StageError(code=error.code,message=str(error),http_status=error.status);stage.elapsed_seconds=time.monotonic()-began
        result.status='failed';result.claims=[];result.reasons=[error.code]
        if stage.name=='validation' and error.code in {'invalid_answer_output','invalid_citation','ambiguous_citation'}:
            result.support_validation.citation_integrity='invalid'
        for following in stages[stages.index(stage)+1:]:
            following.error=StageError(code='preceding_stage_failed',message='An earlier workflow stage failed',http_status=424)
    return RetrievalAnswerResult.model_validate(result.model_dump())
