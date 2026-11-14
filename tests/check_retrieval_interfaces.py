"""Typed client/CLI/MCP workflow dispatch; controlled server, no model execution."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient

from check_retrieval_workflow import Fixture
from titanos import cli
from titanos.client import Client
from titanos.retrieval_workflow_contract import RetrievalAnswerResult


def main():
    with tempfile.TemporaryDirectory() as folder:
        f=Fixture(folder)
        with TestClient(f.app,base_url='http://127.0.0.1') as server:
            body=server.post('/v1/retrieval/answer',json=f.payload).json();seen=[]
            def handle(request):seen.append((request.url.path,json.loads(request.content)));return httpx.Response(200,json=body)
            with Client('http://127.0.0.1',transport=httpx.MockTransport(handle)) as client:
                result=client.retrieval_answer('l',f.payload['query'],f.payload['documents'],embedding_model='e',rerank_model='r',max_tokens=32,prompt_version='retrieval-answer-v2')
                assert isinstance(result,RetrievalAnswerResult) and result.status=='answered'
                client.extension('retrieval_answer','l',embedding_model='e',rerank_model='r',query=f.payload['query'],documents=f.payload['documents'])
            assert seen[0][0]=='/v1/retrieval/answer' and seen[0][1]['llm_model']=='l' and seen[0][1]['max_tokens']==32
            assert seen[0][1]['prompt_version']=='retrieval-answer-v2'
            assert seen[1][0]=='/v1/extensions/infer' and seen[1][1]['llm_model']=='l' and 'model' not in seen[1][1]
            file=Path(folder)/'documents.json';file.write_text(json.dumps(f.payload['documents']),encoding='utf-8')
            class CLIClient:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def retrieval_answer(self,*args,**kwargs):seen.append((args,kwargs));return result
            with patch('titanos.client.Client',return_value=CLIClient()),contextlib.redirect_stdout(io.StringIO()):
                cli.main(['retrieval-answer','l',f.payload['query'],'--embedding-model','e','--rerank-model','r','--documents-file',str(file),'--context-top-n','2','--deadline-ms','900','--no-return-documents','--prompt-version','retrieval-answer-v1'])
            assert seen[-1][1]['context_top_n']==2 and seen[-1][1]['timeout']==900 and seen[-1][1]['return_documents'] is False
            assert seen[-1][1]['prompt_version']=='retrieval-answer-v1'
            async def mcp():
                value=await f.app.state.mcp.call_tool('retrieval_answer',{'payload':{**f.payload,'prompt_version':'retrieval-answer-v2'}})
                assert not value.is_error
                data=value.structured_content
                if data is None:data=json.loads(next(block.text for block in value.content if getattr(block,'type','')=='text'))
                data=data.get('result',data);assert RetrievalAnswerResult.model_validate(data).status=='answered'
                assert data['schema_version']=='2' and data['context']['prompt_version']=='retrieval-answer-v2'
            asyncio.run(mcp())
            assert server.get('/ui/retrieval.js').status_code==200
            assert '/ui/retrieval.js' in server.get('/').text
            assert '/v1/retrieval/answer' in server.get('/openapi.json').json()['paths']
        from titanos.config import Config
        from titanos.server import create_app
        protected=create_app(Config(),api_key='controlled-ui-key',output_dir=Path(folder)/'auth')
        with TestClient(protected,base_url='http://127.0.0.1') as client:
            assert client.get('/ui/retrieval.js').status_code==200
            assert client.get('/v1/extensions').status_code==401
            assert client.post('/v1/retrieval/answer',json=f.payload).status_code==401
    print('PASS typed Python, extension alias, CLI controls, MCP workflow and browser asset discovery; controlled inference only',flush=True)


if __name__=='__main__':main()
