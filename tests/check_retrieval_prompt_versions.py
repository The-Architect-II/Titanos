"""Prompt selection/wire compatibility only; fakes do not prove model semantics."""
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Literal

from fastapi.testclient import TestClient
from pydantic import ValidationError
from check_retrieval_workflow import Fixture
from titanos.config import Config, ModelSpec, NativeOptions, read_config, write_config
from titanos.retrieval_prompts import INSTRUCTIONS, resolve_version
from titanos.retrieval_workflow_contract import ContextTrace, RetrievalAnswerRequest, RetrievalAnswerResult
from titanos.server import create_app
from titanos.settings import Settings

V1='retrieval-answer-v1';V2='retrieval-answer-v2'


class LegacyContext(ContextTrace):
    prompt_version: Literal['retrieval-answer-v1']='retrieval-answer-v1'


class LegacyResult(RetrievalAnswerResult):
    schema_version: Literal['1']='1'
    context: LegacyContext | None=None


def selection():
    # Frozen byte identity of the original instruction at b3b2848.
    assert hashlib.sha256(INSTRUCTIONS[V1].encode()).hexdigest()=='26e9379fb109412faf6800e8ddda4827d3f730c25b87aa532dcb5b76fdcba5cc'
    assert INSTRUCTIONS[V1]!=INSTRUCTIONS[V2]
    assert Settings().retrieval_prompt_version==V2
    assert Settings.from_env({'TITANOS_RETRIEVAL_PROMPT_VERSION':V1}).retrieval_prompt_version==V1
    assert resolve_version(V1,{'retrieval_prompt_version':V2},Settings())==V1
    assert resolve_version(None,{'retrieval_prompt_version':V1},Settings())==V1
    assert resolve_version(None,{'retrieval_prompt_version':None},Settings())==V2
    for value in [False,{},[],1,'unknown','']:
        try:resolve_version(None,{'retrieval_prompt_version':value},Settings())
        except ValueError:pass
        else:raise AssertionError('Malformed model prompt default accepted')
        try:ModelSpec(path='local',task='chat',backend='llama-cpp',native=NativeOptions(model_file='local.gguf'),defaults={'retrieval_prompt_version':value})
        except ValueError:pass
        else:raise AssertionError('Malformed registration default accepted')
    try:ModelSpec(path='local',task='embeddings',defaults={'retrieval_prompt_version':V1})
    except ValueError:pass
    else:raise AssertionError('Prompt default attached to the wrong model role')
    try:Settings(retrieval_prompt_version='unknown')
    except ValueError:pass
    else:raise AssertionError('Unknown process prompt version accepted')


def dispatch():
    for process in [V1,V2]:
        with tempfile.TemporaryDirectory() as folder:
            f=Fixture(folder,Settings(retrieval_prompt_version=process));f.prompt_version=process
            with TestClient(f.app,base_url='http://127.0.0.1') as client:
                for explicit,default,expected in [(None,None,process),(None,V1,V1),(V2,V1,V2),(V1,V2,V1)]:
                    f.cfg.models['l'].defaults['retrieval_prompt_version']=default;f.prompt_version=expected
                    response=client.post('/v1/retrieval/answer',json={**f.payload,'prompt_version':explicit})
                    assert response.status_code==200,response.text
                    data=response.json();assert data['status']=='answered'
                    typed=RetrievalAnswerResult.model_validate(data)
                    assert typed.context.prompt_version==expected
                    assert typed.schema_version==('1' if expected==V1 else '2')
                    assert typed.context.messages[0]['content']==INSTRUCTIONS[expected]
                    supplied=json.loads(typed.context.messages[1]['content'])['documents']
                    assert supplied==[{'document_id':doc['id'],'text':doc['text']} for doc in [f.payload['documents'][0],f.payload['documents'][2]]]
                    if expected==V1:assert LegacyResult.model_validate(data).model_dump()==typed.model_dump()
                    else:
                        try:LegacyResult.model_validate(data)
                        except ValidationError:pass
                        else:raise AssertionError('Old strict client accepted the new wire contract')
                    mismatch=typed.model_copy(update={'schema_version':'2' if expected==V1 else '1'}).model_dump()
                    try:RetrievalAnswerResult.model_validate(mismatch)
                    except ValidationError:pass
                    else:raise AssertionError('Mismatched schema/context version accepted')
                before=len(f.calls)
                for value in ['unknown',{},[],False]:
                    response=client.post('/v1/retrieval/answer',json={**f.payload,'prompt_version':value})
                    assert response.status_code==422 and len(f.calls)==before
                f.cfg.models['l'].defaults['retrieval_prompt_version']=[]
                response=client.post('/v1/retrieval/answer',json=f.payload)
                assert response.status_code==400 and len(f.calls)==before
                assert 'retrieval prompt version' in response.json()['error']['message']
                info=client.get('/v1/extensions').json()['retrieval_answer']
                assert info['prompt_versions']==[V1,V2] and info['default_prompt_version']==process
                # The extension shares the same selection, not another prompt path.
                f.prompt_version=V1
                data=client.post('/v1/extensions/infer',json={**f.payload,'operation':'retrieval_answer','prompt_version':V1}).json()
                assert data['schema_version']=='1' and data['context']['prompt_version']==V1


def registration():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);registry=root/'models.json'
        spec=ModelSpec(path=str(root),task='chat',backend='llama-cpp',device='cpu',
            native=NativeOptions(model_file='not-loaded.gguf'),defaults={'retrieval_prompt_version':V1})
        write_config(registry,Config(models={'answer':spec}))
        assert read_config(registry).models['answer'].defaults['retrieval_prompt_version']==V1
        app=create_app(read_config(registry),output_dir=root/'output',registry_path=registry)
        with TestClient(app,base_url='http://127.0.0.1') as client:
            response=client.post('/api/models/register',json={'name':'answer','spec':{
                'path':str(root),'task':'chat','backend':'llama-cpp','device':'cpu',
                'native':{'model_file':'not-loaded.gguf'},'defaults':{'retrieval_prompt_version':V1}}})
            # Native executable/loader registration stays a trusted local action.
            assert response.status_code==400,response.text
            assert 'trusted local CLI configuration' in response.json()['error']['message']
            assert read_config(registry).models['answer'].defaults['retrieval_prompt_version']==V1
    print('PASS versions, request/model/process precedence, exact legacy prompt, strict schema boundary, invalid-default zero-inference and persisted local defaults/unchanged HTTP loader restriction; no real models')


if __name__=='__main__':selection();dispatch();registration()
