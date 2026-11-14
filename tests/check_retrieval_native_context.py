"""Actual gateway context accounting with controlled transport; no native/model process."""
import asyncio
import json
import time
import httpx

from check_sentiment_native import pool_for
from titanos.engine import EngineError


async def main():
    for mode in ['pass','over_limit','mismatch','unavailable']:
        pool,worker=pool_for('.',max_tokens=8);calls=[]
        async def handle(request):
            path=request.url.path;data=json.loads(request.content) if request.content else None;calls.append((path,data))
            if path=='/apply-template':
                if mode=='unavailable':return httpx.Response(404,json={'error':'unsupported'})
                return httpx.Response(200,json={'prompt':'Five chars: 雨雨雨雨雨'})
            if path=='/tokenize':
                assert data['add_special'] is True and data['parse_special'] is True
                return httpx.Response(200,json={'tokens':list(range(20))})
            if path=='/props':return httpx.Response(200,json={'default_generation_settings':{'n_ctx':64}})
            assert path=='/v1/chat/completions'
            return httpx.Response(200,json={'choices':[{'message':{'content':'{"status":"abstained","claims":[]}'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':21 if mode=='mismatch' else 20,'completion_tokens':6,'total_tokens':26}})
        await worker.client.aclose();worker.client=httpx.AsyncClient(base_url='http://owned-fixture',transport=httpx.MockTransport(handle))
        try:
            data={'model':'l','messages':[{'role':'user','content':'Controlled context.'}]}
            try:result=await pool.complete(data,deadline=time.monotonic()+2,context_max_tokens=19 if mode=='over_limit' else 20)
            except EngineError as exc:
                assert exc.code=={'over_limit':'context_limit','mismatch':'context_accounting_mismatch','unavailable':'context_accounting_unavailable'}[mode]
            else:
                assert mode=='pass' and result['context_accounting']['input_tokens']==20 and result['context_accounting']['verified_against_usage'] is True
            generated=sum(path=='/v1/chat/completions' for path,data in calls)
            assert generated==(1 if mode in {'pass','mismatch'} else 0)
            assert worker.active==worker.pending==0
        finally:await pool.close()
    # Like llama-server, /apply-template and /v1/chat/completions render the prompt from the same body, so the
    # template request must carry tools/tool_choice/parallel_tool_calls or usage differs (502 before the fix).
    def rendered(body):
        return 20+5*bool(body.get('tools'))+(body.get('tool_choice')=='required')+(body.get('parallel_tool_calls') is False)
    tools=[{'type':'function','function':{'name':'get_weather','parameters':{'type':'object','properties':{'city':{'type':'string'}}}}}]
    for extra in [{},{'tools':tools},{'tools':tools,'tool_choice':'required','parallel_tool_calls':False}]:
        pool,worker=pool_for('.',max_tokens=8);calls=[]
        async def handle(request):
            path=request.url.path;data=json.loads(request.content) if request.content else None;calls.append((path,data))
            if path=='/apply-template':return httpx.Response(200,json={'prompt':'|'.join(['t']*rendered(data))})
            if path=='/tokenize':return httpx.Response(200,json={'tokens':list(range(len(data['content'].split('|'))))})
            if path=='/props':return httpx.Response(200,json={'default_generation_settings':{'n_ctx':64}})
            n=rendered(data)
            return httpx.Response(200,json={'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}],'usage':{'prompt_tokens':n,'completion_tokens':1,'total_tokens':n+1}})
        await worker.client.aclose();worker.client=httpx.AsyncClient(base_url='http://owned-fixture',transport=httpx.MockTransport(handle))
        try:
            result=await pool.complete({'model':'l','messages':[{'role':'user','content':'Weather?'}],**extra},deadline=time.monotonic()+2,context_max_tokens=64)
            assert result['context_accounting']['input_tokens']==rendered(extra) and result['context_accounting']['verified_against_usage'] is True
            sent=dict(calls);assert all(sent['/apply-template'].get(k)==sent['/v1/chat/completions'].get(k) for k in ('messages','tools','tool_choice','parallel_tool_calls'))
        finally:await pool.close()
    print('PASS native token/template accounting, pre-generation context rejection, explicit unsupported/mismatch failures, tools/tool_choice/parallel_tool_calls templated as generated; no character estimate or native/model launch',flush=True)


if __name__=='__main__':asyncio.run(main())
