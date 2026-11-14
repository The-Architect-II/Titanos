"""Real llama.cpp engine contracts; tool argument meaning is model quality."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import httpx
from titanos.config import read_config,write_config,Config,WorkerSpec


def weather_arguments(raw):
    value=json.loads(raw)
    assert isinstance(value,dict) and set(value)=={'city'} and isinstance(value['city'],str),value
    return {'arguments':value,'engine_status':'passed','model_quality':{
        'criterion':'Tool call selects the requested city Paris',
        'status':'passed' if value['city'].lower()=='paris' else 'failed'}}


def response_terminal(lines, max_tokens):
    events=[json.loads(line[6:]) for line in lines if line.startswith('data: ') and line[6:]!='[DONE]']
    terminal=[e for e in events if e.get('type') in {'response.completed','response.incomplete'}]
    assert len(terminal)==1,events
    final=terminal[0]['response']
    assert terminal[0]['type']=='response.'+final['status'] and final['usage']['output_tokens']<=max_tokens,final
    if final['status']=='incomplete':assert final['incomplete_details']['reason']=='max_output_tokens',final
    return final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--model',default='agent')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    spec=read_config(args.registry).models[args.model]
    checks=[]
    report={'acceptance_scope':'engine_contract','checks':checks,'success':False,'model':spec.model_dump(),'model_quality':{}}
    with tempfile.TemporaryDirectory(prefix='native-check-') as tmp:
        root=Path(tmp)
        cfg=Config(models={args.model:spec},workers={'native-a':WorkerSpec(models=[args.model])})
        registry=root/'registry.json';write_config(registry,cfg)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env={**os.environ,'TITANOS_HOME':str(root),'HF_HUB_OFFLINE':'1'}
        env.pop('TITANOS_API_KEY',None)
        with (root/'server.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','titanos','--config',str(registry),'serve','--port',str(port)],env=env,cwd=tmp,stdout=log,stderr=log)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=120,trust_env=False) as c:
                for unused in range(200):
                    try:
                        if c.get('/healthz').status_code==200:break
                    except httpx.HTTPError:pass
                    time.sleep(.1)
                else:raise AssertionError((root/'server.log').read_text())
                def post(route,data):
                    r=c.post(route,json={'model':args.model,**data});assert r.is_success,r.text
                    return r.json()
                msg=[{'role':'user','content':'Say hello.'}]
                reply=post('/v1/chat/completions',{'messages':msg,'temperature':0,'max_tokens':20})
                assert reply['choices'][0]['message']['content']
                checks.append('GGUF chat through independent HTTP process')
                report['runtime']=post('/api/models/details',{})
                assert report['runtime']['runtime']['total_slots']>=2
                tool={'type':'function','function':{'name':'weather','description':'Get current weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city'],'additionalProperties':False}}}
                query={'messages':[{'role':'user','content':'What is the weather in Paris? Call weather.'}],'tools':[tool],
                       'tool_choice':{'type':'function','function':{'name':'weather'}},'temperature':0,'max_tokens':100}
                result=post('/v1/chat/completions',query)
                call=result['choices'][0]['message']['tool_calls'][0]
                assert call['function']['name']=='weather'
                report['model_quality']['tool_call']=weather_arguments(call['function']['arguments'])
                followup=post('/v1/chat/completions',{'messages':query['messages']+[result['choices'][0]['message'],
                        {'role':'tool','tool_call_id':call['id'],'content':'Sunny, 22 degrees Celsius.'}],
                        'tools':[tool],'tool_choice':'none','temperature':0,'max_tokens':40})
                assert followup['choices'][0]['message']['content']
                checks.append('real function call and tool-result round trip with preserved IDs/arguments')
                with c.stream('POST','/v1/chat/completions',json={'model':args.model,**query,'stream':True}) as stream:
                    assert stream.status_code==200
                    deltas=[]
                    for line in stream.iter_lines():
                        if line.startswith('data: ') and line!='data: [DONE]':
                            value=json.loads(line[6:]);assert 'error' not in value,value
                            deltas += value.get('choices',[{}])[0].get('delta',{}).get('tool_calls',[])
                    assert deltas
                    arguments=''.join(x.get('function',{}).get('arguments','') for x in deltas)
                    report['model_quality']['streamed_tool_call']=weather_arguments(arguments)
                checks.append('incremental SSE tool-call arguments reconstruct valid JSON')
                schema={'type':'object','properties':{'number':{'type':'integer','const':7}},'required':['number'],'additionalProperties':False}
                result=post('/v1/chat/completions',{'messages':[{'role':'user','content':'Return number 7.'}],'temperature':0,'max_tokens':50,
                    'response_format':{'type':'json_schema','json_schema':{'name':'number','strict':True,'schema':schema}}})
                assert json.loads(result['choices'][0]['message']['content'])=={'number':7}
                checks.append('native grammar-constrained JSON schema result')
                result=post('/v1/responses',{'input':'Say hello.','temperature':0,'max_output_tokens':20})
                assert result['object']=='response' and result['output'][0]['content'][0]['text']
                with c.stream('POST','/v1/responses',json={'model':args.model,'input':'Say hello.','max_output_tokens':20,'stream':True}) as stream:
                    events=list(stream.iter_lines());assert stream.status_code==200
                    response_terminal(events,20)
                checks.append('Responses create and streaming termination, including the requested token cap')
                bad=c.post('/v1/chat/completions',json={'model':args.model,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'http://127.0.0.1/private'}}]}]})
                assert bad.status_code==400
                checks.append('remote media fetch refused')
                prompt={'messages':[{'role':'user','content':'Write an extremely long numbered list of facts about the solar system, keep going.'}],'max_tokens':200,'temperature':0}
                maximum=0
                with ThreadPoolExecutor(max_workers=2) as pool:
                    requests=[pool.submit(post,'/v1/chat/completions',prompt) for _ in range(2)]
                    while not all(r.done() for r in requests):
                        status=c.get('/api/workers').json()['data'][0]
                        maximum=max(maximum,sum(s['is_processing'] for s in status.get('native_slots',[])))
                        time.sleep(.05)
                    for result in requests:assert result.result()['choices'][0]['message']['content']
                assert maximum==2,maximum
                checks.append('two native slots simultaneously processing actual requests')
                report['maximum_processing_slots']=maximum
                with c.stream('POST','/v1/chat/completions',json={'model':args.model,**prompt,'stream':True}) as stream:
                    for line in stream.iter_lines():
                        if 'content' in line:break
                for unused in range(200):
                    status=c.get('/api/workers').json()['data'][0]
                    if status['active_requests']==0 and not any(s['is_processing'] for s in status.get('native_slots',[])):break
                    time.sleep(.05)
                else:raise AssertionError('Cancellation did not release native slot')
                checks.append('socket disconnect cancels generation and releases native slot')
                # Kill the native process after a real stream starts; the stream must fail, never restart/replay.
                pid=c.get('/api/workers').json()['data'][0]['pid']
                failed=False
                try:
                    with c.stream('POST','/v1/chat/completions',json={'model':args.model,**prompt,'stream':True}) as stream:
                        killed=False
                        for line in stream.iter_lines():
                            if 'content' in line and not killed:
                                os.kill(pid,signal.SIGTERM);killed=True
                except httpx.HTTPError:failed=True
                assert failed,'Native death appeared to be a successful stream'
                for unused in range(100):
                    if c.get('/api/workers').json()['data'][0]['active_requests']==0:break
                    time.sleep(.05)
                assert post('/v1/chat/completions',{'messages':msg,'max_tokens':20})['choices']
                checks.append('native process death fails stream; next request restarts without replay')
                post('/api/models/unload',{})
                assert not c.get('/api/models/loaded').json()['data']
                checks.append('explicit unload and clean shutdown')
                post('/api/models/load',{})
                child=c.get('/api/workers').json()['data'][0]['pid']
                # On Windows a venv python.exe is a launcher whose child is the real interpreter; kill the actual Titanos
                # process (the native child's parent) so the check measures Titanos death, not launcher death.
                import psutil
                owner=psutil.Process(child).parent()
                if owner is not None and owner.pid!=proc.pid:owner.kill()
                proc.kill();proc.wait()
                # os.kill(pid, 0) is a Unix liveness probe; on Windows it would terminate the process. psutil is platform-neutral.
                for unused in range(200):
                    if not psutil.pid_exists(child):break
                    time.sleep(.1)
                else:raise AssertionError('Native child survived abrupt Titanos death')
                checks.append('abrupt Titanos death closes watchdog pipe and releases native child')
                report['success']=True
        finally:
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            if not report['success']:report['server_log']=(root/'server.log').read_text()[-6000:]
            args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'success':report['success'],'checks':checks},indent=2))


if __name__=='__main__':main()
