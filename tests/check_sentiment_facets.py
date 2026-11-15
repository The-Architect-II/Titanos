"""Finite facet task contracts and protocol dispatch; no models or held-out inputs."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
from pydantic import ValidationError

from titanos.api_plus import SentimentRequest
from titanos.config import Config, ModelSpec
from titanos.sentiment_contract import (FACET_OUTPUTS, SentimentResult, SentimentComposition,
    facet_label, facet_prefix_allowed_tokens, parse_llm_output, build_result)
from titanos.server import create_app


def truth_table():
    payload=SentimentRequest(model='l',text='A controlled input.',min_score=.99,min_margin=.99)
    class Tokens(list):
        def __getitem__(self,key):
            value=super().__getitem__(key);return Tokens(value) if isinstance(key,slice) else value
        def tolist(self):return list(self)
    class Tokenizer:
        def encode(self,text,add_special_tokens=False):return list(text.encode())
        def decode(self,tokens,skip_special_tokens=True):return bytes(tokens).decode()
    allowed=facet_prefix_allowed_tokens(Tokenizer(),2,300)
    for text in FACET_OUTPUTS:
        flags=json.loads(text);p,n,s=flags.values()
        expected='mixed' if p and n else 'positive' if p else 'negative' if n else 'neutral' if s else 'unknown'
        assert facet_label(flags)==expected
        raw=parse_llm_output(text,'stop','facets-v1')
        result=build_result(payload,raw,'llm',{'backend':'transformers','prompt_version':'facets-v1'},'facet_choices')
        assert result.schema_version=='3' and result.decision_basis=='facets'
        assert result.model_facets.model_dump()==flags and result.model_label is None
        assert result.scores is None and not result.score_thresholds_applied
        assert result.label==('unknown' if expected in {'mixed','unknown'} else expected)
        assert result.abstention_reasons==(['mixed_polarity'] if expected=='mixed' else ['insufficient_evidence'] if expected=='unknown' else [])
        tokens=list(text.encode())
        for index,token in enumerate(tokens):assert token in allowed(0,Tokens([9,9]+tokens[:index]))
        assert allowed(0,Tokens([9,9]+tokens))==[300]
        # Caller uncertainty always remains explicit, even when polarity was reported.
        asr=build_result(payload.model_copy(update={'input_modality':'asr_transcript','asr_uncertain':True}),raw,'llm',{'backend':'transformers'},'facet_choices')
        assert asr.status=='abstained' and asr.label=='unknown' and 'asr_uncertain' in asr.abstention_reasons
        forged=result.model_dump();forged.update(status='ok',label='neutral',abstention_reasons=[])
        if expected!='neutral':
            try:SentimentResult.model_validate(forged)
            except ValidationError:pass
            else:raise AssertionError('Invented outcome bypassed reported facet flags')
    for text,finish in [('{}','stop'),(FACET_OUTPUTS[0],'length'),
        ('{"positive_evaluation":1,"negative_evaluation":false,"neutral_statement":false}','stop'),
        ('{"positive_evaluation":false,"negative_evaluation":false,"neutral_statement":false,"abstain":true}','stop'),
        ('{"positive_evaluation":false,"positive_evaluation":true,"negative_evaluation":false,"neutral_statement":false}','stop')]:
        raw=parse_llm_output(text,finish,'facets-v1');assert raw['invalid_model_output']
        result=build_result(payload,raw,'llm',{'backend':'transformers'},'facet_choices')
        assert result.schema_version=='3' and result.model_facets is None and result.abstention_reasons==['invalid_model_output']


def protocols():
    with tempfile.TemporaryDirectory() as folder:
        cfg=Config(models={'c':ModelSpec(path=folder,task='sentiment'),'l':ModelSpec(path=folder,task='chat'),
            'g':ModelSpec(path=folder,task='chat',backend='openvino-genai',device='CPU')})
        app=create_app(cfg,output_dir=Path(folder));calls=[]
        flags={'positive_evaluation':True,'negative_evaluation':True,'neutral_statement':False}
        def infer(model,task,data,emit=None,cancel=None):
            calls.append(copy.deepcopy(data))
            if task=='sentiment':return {'scores':{'negative':.1,'neutral':.1,'positive':.8},'score_kind':'softmax_uncalibrated','provenance':{'backend':'transformers'}}
            content=flags if data.get('_sentiment_facet_choices') else {'label':'positive'}
            return {'text':json.dumps(content),'finish_reason':'stop','provenance':{'device':'cpu'}}
        app.state.engine.target.infer=infer
        with TestClient(app,base_url='http://127.0.0.1') as client:
            basic={'model':'l','text':'A controlled protocol input.'}
            ordinary=client.post('/v1/sentiment',json=basic).json()
            assert ordinary['schema_version']=='2' and 'model_facets' not in ordinary and ordinary['label']=='positive'
            facet={**basic,'prompt_version':'facets-v1'}
            result=client.post('/v1/sentiment',json=facet).json();SentimentResult.model_validate(result)
            assert result['schema_version']=='3' and result['model_facets']==flags and result['constraint_mode']=='facet_choices'
            assert result['label']=='unknown' and result['abstention_reasons']==['mixed_polarity']
            assert calls[-1]['_sentiment_facet_choices'] is True and '_sentiment_label_choices' not in calls[-1]
            composition={'classifier_model':'c','llm_model':'l','text':basic['text'],'prompt_version':'facets-v1'}
            result=client.post('/v1/sentiment/compose',json=composition).json();SentimentComposition.model_validate(result)
            assert result['status']=='abstained' and result['llm']['result']['model_facets']==flags
            assert result['classifier']['result']['schema_version']=='2' and 'model_facets' not in result['classifier']['result']
            for body in [{**facet,'decoding':'label_choices'},{**basic,'decoding':'facet_choices'},{**facet,'model':'g','decoding':'facet_choices'}]:
                before=len(calls);assert client.post('/v1/sentiment',json=body).status_code==400 and len(calls)==before
            extension=client.post('/v1/extensions/infer',json={**composition,'operation':'sentiment_compose'}).json()
            assert extension['llm']['result']['model_facets']==flags
            async def mcp():
                value=await app.state.mcp.call_tool('compose_sentiment',{'payload':composition})
                assert not value.is_error
                data=value.structured_content
                if data is None:data=json.loads(next(block.text for block in value.content if getattr(block,'type','')=='text'))
                data=data.get('result',data)
                assert data['llm']['result']['schema_version']=='3' and data['llm']['result']['model_facets']==flags
            asyncio.run(mcp())


if __name__=='__main__':
    truth_table();protocols()
    print('PASS: eight finite facet records, strict flags, exact derivation/ASR/malformed abstention, schema2 wire preservation and schema3 HTTP/extension/MCP composition; no model inference')
