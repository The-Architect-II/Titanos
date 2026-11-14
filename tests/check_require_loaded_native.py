"""Actual native methods with fake process/client objects; no process or network work."""
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import httpx
from titanos.config import Config,ModelSpec,NativeOptions
from titanos.engine import EngineError
from titanos.native import NativePool,NativeWorker
from titanos.settings import Settings


class Process:
    def __init__(self):self.dead=False;self.unreadable=False
    def poll(self):
        if self.unreadable:raise OSError('fake liveness unavailable')
        return 1 if self.dead else None


class Reply:
    def __init__(self,close_error=None):
        self.is_success=True;self.status_code=200;self.headers={'content-type':'application/json'}
        self.body=json.dumps({'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}],'usage':{}}).encode()
        self.closes=0;self.close_error=close_error
    @property
    def text(self):return self.body.decode()
    def json(self):return json.loads(self.body)
    async def aread(self):return self.body
    async def aiter_bytes(self):
        yield b'data: one\n\n'
        yield b'data: [DONE]\n\n'
    async def aclose(self):
        self.closes+=1
        if self.close_error:raise self.close_error


class Client:
    def __init__(self,reply=None):
        self.is_closed=False;self.reply=reply or Reply();self.sent=[];self.failure=None
        self.block=None;self.entered=asyncio.Event();self.cancelled=asyncio.Event();self.cancel_count=0;self.suppress_cancel=False
    def build_request(self,method,route,json):return (method,route,dict(json))
    async def send(self,request,stream):
        self.sent.append(request);self.entered.set()
        if self.block is not None:
            while not self.block.is_set():
                try:await self.block.wait()
                except asyncio.CancelledError:
                    self.cancel_count+=1;self.cancelled.set()
                    if not self.suppress_cancel:raise
        if self.failure:raise self.failure
        return self.reply


class Request:
    def __init__(self):self.scope={};self.disconnected=False
    async def is_disconnected(self):return self.disconnected


def pool(warm=True,slots=1):
    config=Config(models={'m':ModelSpec(path='.',task='chat',backend='llama-cpp',native=NativeOptions(model_file='never-loaded.gguf',slots=slots))})
    value=NativePool(config,Settings(native_queue_timeout=.25));worker=value.workers[0]
    worker.starts=worker.stops=0
    worker.process=Process() if warm else None;worker.ready=warm;worker.client=Client() if warm else None
    async def start():
        worker.starts+=1
        if worker.process is None:worker.process=Process();worker.ready=True;worker.client=Client()
    async def stop():
        worker.stops+=1;worker.ready=False;worker.process=None;worker.client=None
    worker.start=start;worker.stop=stop
    return value,worker


async def spin(predicate):
    async with asyncio.timeout(2):
        while not predicate():await asyncio.sleep(0)


async def rejection(action,code):
    try:await action()
    except EngineError as error:assert error.code==code,(error.code,code);return error
    raise AssertionError('Expected '+code)


async def lease_policy():
    p,w=pool(False)
    await rejection(lambda:p.lease('m',require_loaded=True).__aenter__(),'model_not_loaded')
    assert (w.starts,w.pending,w.active,p.waiting)==(0,0,0,0)
    async with p.lease('m'):assert w.active==1 and w.starts==1
    assert w.active==0
    for state,code in [('broken','worker_unavailable'),('closed','server_unavailable'),('dead','worker_unavailable'),('unreadable','worker_unavailable'),('client_closed','worker_unavailable'),('missing_client','model_not_loaded'),('not_ready','model_not_loaded')]:
        p,w=pool()
        if state in {'broken','closed'}:setattr(w,state,True)
        elif state in {'dead','unreadable'}:setattr(w.process,state,True)
        elif state=='client_closed':w.client.is_closed=True
        elif state=='missing_client':w.client=None
        else:w.ready=False
        await rejection(lambda:p.lease('m',require_loaded=True).__aenter__(),code)
        assert (w.starts,w.stops,w.pending,w.active,p.waiting)==(0,0,0,0,0)
    p,w=pool();p.closed=True
    await rejection(lambda:p.lease('m',require_loaded=True).__aenter__(),'server_unavailable')
    p,w=pool();await rejection(lambda:p.lease('unknown',require_loaded=True).__aenter__(),'model_not_found')


async def busy_and_recheck():
    p,w=pool();_,cold=pool(False);cold.name='cold';p.workers.append(cold)
    held=p.lease('m',require_loaded=True);assert await held.__aenter__() is w
    waiting=p.lease('m',require_loaded=True);task=asyncio.create_task(waiting.__aenter__());await spin(lambda:p.waiting==1)
    assert cold.starts==0 and not task.done()
    await rejection(lambda:p.unload('m',w.name),'model_busy')
    await held.__aexit__(None,None,None);assert await task is w
    await waiting.__aexit__(None,None,None);assert (w.starts,w.active,w.pending,p.waiting)==(0,0,0,0)
    await rejection(lambda:p.lease('m','cold',require_loaded=True).__aenter__(),'model_not_loaded')
    for change,code in [('cold','model_not_loaded'),('unreadable','worker_unavailable'),('pool_closed','server_unavailable')]:
        p,w=pool();await w.lock.acquire()
        context=p.lease('m',require_loaded=True);task=asyncio.create_task(context.__aenter__());await spin(lambda:w.pending==1)
        if change=='cold':w.ready=False
        elif change=='unreadable':w.process.unreadable=True
        else:p.closed=True
        w.lock.release();await rejection(lambda:task,code)
        assert (w.starts,w.stops,w.active,w.pending,p.waiting)==(0,0,0,0,0)
    # Pool close can win after the worker lock is obtained but before condition.
    p,w=pool();await w.lock.acquire();context=p.lease('m',require_loaded=True)
    task=asyncio.create_task(context.__aenter__());await spin(lambda:w.pending==1)
    await p.condition.acquire();w.lock.release();await spin(lambda:w.lock.locked())
    p.closed=True;p.condition.release();await rejection(lambda:task,'server_unavailable')
    assert (w.active,w.pending,w.starts,w.stops)==(0,0,0,0)


async def proxy_lifetime():
    for stream,consume in [(False,False),(True,False),(True,True)]:
        p,w=pool();req=Request();data={'model':'m','messages':[],'stream':stream,'require_loaded':True}
        response=await p.proxy(req,'/v1/chat/completions',data,require_loaded=True)
        assert data['require_loaded'] is True and 'require_loaded' not in w.client.sent[0][2]
        if stream and consume:assert len([part async for part in response.body_iterator])==2
        assert w.active==1 and w.starts==0 and w.client.reply.closes==0
        await rejection(lambda:p.unload('m',w.name),'model_busy')
        callback=req.scope['titanos.cleanup'][0];await callback();await callback()
        assert (w.active,w.pending,w.client.reply.closes)==(0,0,1)
    p,w=pool(False);req=Request()
    await rejection(lambda:p.proxy(req,'/v1/chat/completions',{'model':'m'},require_loaded=True),'model_not_loaded')
    await req.scope['titanos.cleanup'][0]();assert w.starts==0 and w.active==0
    p,w=pool();req=Request();error=OSError('fake response close failure');w.client.reply.close_error=error
    await p.proxy(req,'/v1/chat/completions',{'model':'m'},require_loaded=True)
    try:await req.scope['titanos.cleanup'][0]()
    except OSError as observed:assert observed is error
    else:raise AssertionError('Close error lost')
    assert w.active==0 and w.client.reply.closes==1
    p,w=pool(False);req=Request()
    await p.proxy(req,'/v1/chat/completions',{'model':'m','require_loaded':False})
    assert w.starts==1 and w.active==0 and 'titanos.cleanup' not in req.scope
    assert 'require_loaded' not in w.client.sent[0][2]


async def disconnect_and_settle():
    p,w=pool();held=p.lease('m',require_loaded=True);await held.__aenter__();req=Request()
    task=asyncio.create_task(p.proxy(req,'/v1/chat/completions',{'model':'m'},require_loaded=True))
    await spin(lambda:p.waiting==1);req.disconnected=True;await held.__aexit__(None,None,None)
    await rejection(lambda:task,'cancelled');await req.scope['titanos.cleanup'][0]()
    assert w.client.sent==[] and (w.active,w.pending,p.waiting)==(0,0,0)
    p,w=pool();client=w.client
    class ClosingRequest(Request):
        async def is_disconnected(self):
            if w.active and not p.closed:await p.close()
            return False
    req=ClosingRequest()
    await rejection(lambda:p.proxy(req,'/v1/chat/completions',{'model':'m'},require_loaded=True),'server_unavailable')
    await req.scope['titanos.cleanup'][0]();assert client.sent==[] and w.active==0 and w.starts==0
    p,w=pool();req=Request();w.client.failure=httpx.ReadError('fake transport death')
    await rejection(lambda:p.proxy(req,'/v1/chat/completions',{'model':'m'},require_loaded=True),'worker_unavailable')
    await req.scope['titanos.cleanup'][0]();assert w.broken and w.starts==0 and len(w.client.sent)==1 and w.active==0
    p,w=pool();req=Request();client=w.client;client.block=asyncio.Event();client.suppress_cancel=True
    task=asyncio.create_task(p.proxy(req,'/v1/chat/completions',{'model':'m'},require_loaded=True))
    await client.entered.wait();task.cancel()
    try:await task
    except asyncio.CancelledError:pass
    done=asyncio.create_task(req.scope['titanos.cleanup'][0]());await spin(lambda:client.cancel_count>=2)
    assert not done.done() and w.active==1
    done.cancel();await asyncio.sleep(0);done.cancel();await asyncio.sleep(0)
    assert w.active==1;client.block.set()
    try:await done
    except asyncio.CancelledError:pass
    assert (w.active,w.pending,client.reply.closes)==(0,0,1)


async def complete_policy():
    data={'model':'m','messages':[{'role':'user','content':'fake'}]}
    for body_flag,keyword in [(True,False),(False,True)]:
        p,w=pool();result=await p.complete({**data,'require_loaded':body_flag},require_loaded=keyword)
        assert result['text']=='ok' and w.starts==0 and w.active==0 and 'require_loaded' not in w.client.sent[0][2]
    p,w=pool(False);await rejection(lambda:p.complete({**data,'require_loaded':True}),'model_not_loaded');assert w.starts==0
    p,w=pool(False);await p.complete(data);assert w.starts==1 and 'require_loaded' not in w.client.sent[0][2]
    for invalid in ['true',1,None,[],{}]:
        p,w=pool();await rejection(lambda:p.complete({**data,'require_loaded':invalid}),'invalid_request_error');assert w.client.sent==[] and w.starts==0


async def main():
    for case in [lease_policy,busy_and_recheck,proxy_lifetime,disconnect_and_settle,complete_policy]:
        async with asyncio.timeout(5):await case()
        print('PASS',case.__name__)
    print('PASS actual native loaded-only admission/lifetime/cleanup/propagation; fake process/client only, no native process/network/model execution')


if __name__=='__main__':asyncio.run(main())
