"""Protocol parity and incremental framing with a synthetic local inference seam."""
import json
import asyncio
from pathlib import Path
import tempfile
from fastapi.testclient import TestClient
from fastapi.responses import Response, StreamingResponse
from titanos.config import Config,ModelSpec
from titanos.server import create_app


def main():
    seen=[]
    def infer(name,task,data,emit=None,cancel=None):
        seen.append((name,task,data))
        if task=='embeddings':return {'vectors':[[.1,.2] for _ in data['input']],'usage':{'prompt_tokens':2,'total_tokens':2}}
        if task=='speech':return b'RIFF'+b'\0'*40
        if emit:emit('Hello');emit(' world')
        return {'text':'Hello world','finish_reason':'stop','usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}}
    with tempfile.TemporaryDirectory() as temp:
        cfg=Config(models={name:ModelSpec(path=temp,task=task) for name,task in [('chat','chat'),('embed','embeddings'),('voice','speech')]})
        app=create_app(cfg,output_dir=Path(temp)/'outputs');app.state.engine.target.infer=infer
        with TestClient(app,base_url='http://localhost') as c:
            body={'model':'chat','messages':[{'role':'user','content':'Hi'}],'stream':False,'options':{'num_predict':12,'temperature':0,'top_k':7,'min_p':.1}}
            r=c.post('/api/chat',json=body);assert r.status_code==200,r.text
            assert r.json()['message']['content']=='Hello world' and r.json()['eval_count']==2
            assert seen[-1][2]['max_tokens']==12 and seen[-1][2]['top_k']==7
            rows=[json.loads(line) for line in c.post('/api/chat',json={**body,'stream':True}).text.splitlines()]
            assert ''.join(row['message']['content'] for row in rows)=='Hello world'
            assert rows[-1]['done'] and rows[-1]['eval_count']==2 and sum(x['done'] for x in rows)==1
            r=c.post('/api/generate',json={'model':'chat','prompt':'Hi','stream':False});assert r.json()['response']=='Hello world'
            r=c.post('/api/embed',json={'model':'embed','input':['one','two']});assert len(r.json()['embeddings'])==2
            assert c.post('/api/embeddings',json={'model':'embed','prompt':'one'}).json()['embedding']==[.1,.2]
            assert len(c.get('/api/tags').json()['models'])==3
            assert c.post('/api/chat',json={**body,'options':{'num_gpu':1}}).status_code==422
            r=c.post('/api/chat',json={**body,'options':{'num_ctx':123}});assert r.status_code==200 and r.headers['x-titanos-dropped']=='options.num_ctx',r.text  # tests/check_ollama_compat.py
            anthropic={'model':'chat','messages':[{'role':'user','content':'Hi'}],'max_tokens':12,'system':'Be concise.'}
            r=c.post('/v1/messages',json=anthropic);assert r.status_code==200,r.text
            assert r.json()['content'][0]['text']=='Hello world' and r.json()['usage']=={'input_tokens':3,'output_tokens':2}
            assert seen[-1][2]['messages'][0]['role']=='system'
            # upstream-shaped blocks missing a required key are client errors, never a server KeyError
            assert c.post('/v1/messages',json={**anthropic,'system':[{'type':'text'}]}).status_code==400
            assert c.post('/v1/messages',json={**anthropic,'system':[{'type':'text','text':'A'},{'type':'text','text':'B'}]}).status_code==200
            assert c.post('/v1/messages',json={**anthropic,'messages':[{'role':'user','content':[{'type':'image'}]}]}).status_code==400
            r=c.post('/v1/messages',json={**anthropic,'stream':True});assert r.status_code==200
            events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: ')]
            assert events[0]['type']=='message_start' and events[-1]['type']=='message_stop'
            assert ''.join(e['delta']['text'] for e in events if e['type']=='content_block_delta')=='Hello world'
            assert events[-2]['usage']['output_tokens']==2
            r=c.post('/v1/text-to-speech/default?output_format=wav',json={'model_id':'voice','text':'Hi'});assert r.content.startswith(b'RIFF')
            assert c.post('/v1/text-to-speech/default?output_format=mp3',json={'text':'Hi'}).status_code==400
            # NativePool returns Response objects, not the dict used by Engine.
            app.state.native.models['native']=cfg.models['chat']
            async def native_proxy(request,route,data):
                if data.get('messages',[{}])[-1].get('content')=='fail':
                    return Response(json.dumps({'error':{'message':'model unavailable','type':'model_load_failed'}}),status_code=503,media_type='application/json')
                if route=='/v1/completions':
                    body={'choices':[{'text':'Native text','finish_reason':'stop'}],'usage':{'prompt_tokens':3,'completion_tokens':2}}
                else:
                    body={'choices':[{'message':{'role':'assistant','content':'Native answer','tool_calls':[{'id':'call1','type':'function','function':{'name':'get_weather','arguments':'{"city":"Paris"}'}}]},'finish_reason':'tool_calls'}],
                          'usage':{'prompt_tokens':7,'completion_tokens':4}}
                return Response(json.dumps(body),media_type='application/json')
            app.state.native.proxy=native_proxy
            r=c.post('/api/chat',json={'model':'native','messages':[{'role':'user','content':'Hi'}],'stream':False})
            assert r.status_code==200,r.text
            assert r.json()['message']['tool_calls'][0]['function']['arguments']=={'city':'Paris'}
            assert r.json()['eval_count']==4
            r=c.post('/api/generate',json={'model':'native','prompt':'Hi','stream':False})
            assert r.status_code==200 and r.json()['response']=='Native text',r.text
            r=c.post('/v1/messages',json={'model':'native','messages':[{'role':'user','content':'Hi'}],'max_tokens':12})
            assert r.status_code==200,r.text
            assert r.json()['stop_reason']=='tool_use' and r.json()['content'][1]=={'type':'tool_use','id':'call1','name':'get_weather','input':{'city':'Paris'}}
            for stream in [False,True]:
                for route,payload in [('/api/chat',{'model':'native','messages':[{'role':'user','content':'fail'}],'stream':stream}),
                                      ('/v1/messages',{'model':'native','messages':[{'role':'user','content':'fail'}],'max_tokens':12,'stream':stream})]:
                    r=c.post(route,json=payload)
                    assert r.status_code==503 and r.json()['error']['type']=='model_load_failed',r.text
    async def iterator_cleanup():
        from titanos.protocols import events
        closed=[]
        async def frames():
            try:
                # Split a non-ASCII character over chunks, as real network reads may.
                raw='data: {"choices":[{"delta":{"content":"caf\u00e9"}}]}\n\n'.encode()
                offset=raw.index(b'\xc3')+1
                yield raw[:offset];yield raw[offset:]
                await asyncio.sleep(60)
            finally:closed.append(True)
        response=StreamingResponse(frames())
        stream=events(response)
        first=await anext(stream)
        assert first['choices'][0]['delta']['content']=='caf\u00e9'
        await stream.aclose()
        assert closed,'Closing the protocol iterator must close the underlying response stream'
    asyncio.run(iterator_cleanup())
    print(json.dumps({'passed':True,'checks':['Ollama normal/NDJSON/text/embeddings','Anthropic normal/SSE and final usage',
                     'ElevenLabs WAV shape','sampling option translation and unsupported options',
                     'actual native Response JSON/tool conversion and upstream error preservation',
                     'fragmented UTF-8 SSE and iterator close propagation'],
                     'scope':'Synthetic inference proves protocol mappings; native checkpoint tests are separate'}))

if __name__=='__main__':main()
