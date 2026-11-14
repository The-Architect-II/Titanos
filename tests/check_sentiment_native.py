"""Owned native completion boundary with fake transports/workers; zero model/process launches."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from titanos.config import Config, ModelSpec, NativeOptions, write_config
from titanos.engine import EngineError
from titanos.native import NativePool, NativeWorker
from titanos.sentiment_contract import LLM_INSTRUCTION, LABEL_INSTRUCTION, sentiment_json_schema
from titanos.server import create_app


def spec(folder, **defaults):
    return ModelSpec(path=folder,task='chat',backend='llama-cpp',device='cpu',source_repo='fixture/repo',source_revision='fixture-revision',
        native=NativeOptions(model_file='first.gguf',slots=1,threads=4,batch_size=256,ubatch_size=128),defaults=defaults)


class FakeWorker:
    def __init__(self, model_spec, label='positive', phase=None):
        self.spec=model_spec;self.name='native:l';self.model='l';self.active=self.pending=self.completed=0
        self.broken=False;self.error=None;self.running=self.ready=True;self.lock=asyncio.Lock();self.started=int(time.time())
        self.checkpoint_digest='a'*64;self.calls=[];self.start_calls=0;self.phase=phase
        self.entered=asyncio.Event();self.stopped=asyncio.Event();self.read_closed=asyncio.Event()
        outer=self
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                outer.entered.set()
                await asyncio.Event().wait()
                yield b''
            async def aclose(self):outer.read_closed.set()
        async def handle(request):
            self.calls.append(json.loads(request.content))
            if phase=='read':return httpx.Response(200,stream=Body())
            return httpx.Response(200,json={'choices':[{'index':0,'message':{'role':'assistant','content':json.dumps({'label':label})},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':20,'completion_tokens':6,'total_tokens':26}})
        self.client=httpx.AsyncClient(base_url='http://owned-fixture',transport=httpx.MockTransport(handle))
    async def start(self):
        self.start_calls+=1
        if self.phase=='start':
            self.entered.set()
            try:await asyncio.Event().wait()
            finally:self.stopped.set()
    async def stop(self):
        self.running=self.ready=False;self.stopped.set();await self.client.aclose()
    async def close(self):await self.stop()
    def status(self):return {'id':self.name,'device':'cpu','binary_sha256':'b'*64,'checkpoint_digest':self.checkpoint_digest}


def pool_for(folder, label='positive', phase=None, **defaults):
    cfg=Config(models={'l':spec(folder,**defaults)})
    pool=NativePool(cfg);worker=FakeWorker(cfg.models['l'],label,phase);pool.workers=[worker]
    return pool,worker


async def boundary():
    with tempfile.TemporaryDirectory() as folder:
        pool,worker=pool_for(folder,max_completion_tokens=11,temperature=0)
        base={'model':'l','messages':[{'role':'user','content':'Controlled completion.'}]}
        try:
            reply=await pool.complete({**base,'response_format':{'type':'json_schema','json_schema':{'schema':sentiment_json_schema('label-v2')}}})
            assert json.loads(reply['text'])=={'label':'positive'} and reply['provenance']['checkpoint_digest']=='a'*64
            assert worker.calls[-1]['max_tokens']==11 and worker.calls[-1]['temperature']==0
            assert worker.calls[-1]['response_format']['json_schema']['schema']['properties']['label']['enum']==['negative','neutral','positive','mixed','unknown']
            await pool.complete({**base,'max_tokens':2});assert worker.calls[-1]['max_tokens']==2
            for bad in [{**base,'stream':True},{**base,'worker':'missing'},{**base,'max_tokens':0}]:
                before=worker.start_calls
                try:await pool.complete(bad)
                except EngineError:pass
                else:raise AssertionError('Invalid native request accepted')
                assert worker.start_calls==before
            pool.models['l'].defaults={'max_tokens':10**30}
            before=worker.start_calls
            try:await pool.complete(base)
            except EngineError as exc:assert exc.status==400
            else:raise AssertionError('Invalid native defaults accepted')
            assert worker.start_calls==before
        finally:await pool.close()
        for phase in ['queue','start','read']:
            for reason in ['deadline','caller']:
                pool,worker=pool_for(folder,phase=phase)
                if phase=='queue':worker.active=1
                event=threading.Event()
                task=asyncio.create_task(pool.complete(base,deadline=time.monotonic()+(.04 if reason=='deadline' else 2),cancel=event))
                if reason=='caller':
                    if phase=='queue':await asyncio.sleep(.01)
                    else:await asyncio.wait_for(worker.entered.wait(),1)
                    event.set()
                try:await asyncio.wait_for(task,1)
                except EngineError as exc:assert exc.code==('deadline_exceeded' if reason=='deadline' else 'cancelled')
                else:raise AssertionError('Native cancellation did not fail the request')
                assert pool.waiting==0 and worker.pending==0 and worker.active==(1 if phase=='queue' else 0)
                assert worker.completed==0
                if phase=='start':assert worker.stopped.is_set()
                if phase=='read':assert worker.read_closed.is_set()
                worker.active=0;await pool.close()
        # Actual NativeWorker.start cancellation joins its controlled startup thread.
        worker=NativeWorker('native:l','l',spec(folder));entered=threading.Event();exited=threading.Event();stops=[]
        def start(cancel):
            entered.set();assert cancel.wait(1);exited.set()
        original_stop=worker.stop
        async def stop():stops.append(True);await original_stop()
        worker.start_process=start;worker.stop=stop
        task=asyncio.create_task(worker.start());assert await asyncio.to_thread(entered.wait,1)
        task.cancel()
        try:await asyncio.wait_for(task,1)
        except asyncio.CancelledError:pass
        else:raise AssertionError('Startup caller cancellation was swallowed')
        assert exited.is_set() and len(stops)==2
    print('PASS native boundary: defaults, schemas, queue/start/read deadlines and cancellation, startup thread joined',flush=True)


def bridge_and_reload():
    with tempfile.TemporaryDirectory() as folder:
        cfg=Config(models={'l':spec(folder,max_tokens=11),'c':ModelSpec(path=folder,task='sentiment')})
        registry=Path(folder)/'models.json';write_config(registry,cfg);created=[]
        def factory(config, settings=None):
            pool=NativePool(config,settings=settings);worker=FakeWorker(config.models['l'],'positive' if not created else 'negative');pool.workers=[worker];created.append(pool);return pool
        with patch('titanos.native.NativePool',side_effect=factory):
            app=create_app(cfg,output_dir=Path(folder)/'out',registry_path=registry)
            def classifier(*args,**kwargs):return {'scores':{'negative':.1,'neutral':.1,'positive':.8},'score_kind':'softmax_uncalibrated','provenance':{'backend':'transformers'}}
            app.state.engine.target.infer=classifier
            with TestClient(app,base_url='http://127.0.0.1') as client:
                base={'model':'l','text':'A synthetic bridge input.','timeout':1000}
                reply=client.post('/v1/sentiment',json=base);assert reply.status_code==200,reply.text
                result=reply.json();assert result['label']=='positive' and result['constraint_mode']=='native_json_schema'
                assert result['provenance']['backend']=='llama-cpp' and result['provenance']['checkpoint_digest']=='a'*64
                assert created[0].workers[0].calls[-1]['messages'][0]['content']==LABEL_INSTRUCTION
                combined=client.post('/v1/sentiment/compose',json={'classifier_model':'c','llm_model':'l','text':base['text'],'timeout':1000}).json()
                assert combined['status']=='ok' and combined['classifier']['result']['label']==combined['llm']['result']['label']=='positive'
                bad=client.post('/v1/sentiment',json={**base,'decoding':'label_choices'});assert bad.status_code==400
                before=len(created[0].workers[0].calls)
                bad=client.post('/v1/sentiment/compose',json={'classifier_model':'c','llm_model':'l','llm_worker':'missing','text':base['text']});assert bad.status_code==404
                assert len(created[0].workers[0].calls)==before
                assert client.post('/api/models/reload').status_code==200
                result=client.post('/v1/sentiment',json=base).json()
                assert result['label']=='negative' and len(created)==2 and created[0].closed
                assert len(created[1].workers[0].calls)==1, 'Reload reused a stale native pool'
    print('PASS live-loop sentiment bridge and dynamic native gateway after registry reload',flush=True)


async def bridge_cancellation():
    with tempfile.TemporaryDirectory() as folder:
        cfg=Config(models={'l':spec(folder),'c':ModelSpec(path=folder,task='sentiment')})
        app=create_app(cfg,output_dir=Path(folder))
        worker=FakeWorker(cfg.models['l'],phase='read');app.state.native.workers=[worker]
        def classifier(*args,**kwargs):return {'scores':{'negative':.1,'neutral':.1,'positive':.8},'score_kind':'softmax_uncalibrated','provenance':{'backend':'transformers'}}
        app.state.engine.target.infer=classifier
        try:
            task=asyncio.create_task(app.state.mcp.call_tool('compose_sentiment',{'payload':{
                'classifier_model':'c','llm_model':'l','text':'A native cancellation fixture.','timeout':2000}}))
            await asyncio.wait_for(worker.entered.wait(),1)
            task.cancel()
            try:await asyncio.wait_for(task,1)
            except asyncio.CancelledError:pass
            else:raise AssertionError('Native MCP caller cancellation was swallowed')
            assert worker.read_closed.is_set() and worker.active==worker.pending==0 and worker.completed==0
        finally:
            await app.state.native.close();await asyncio.to_thread(app.state.engine.close)
    print('PASS actual MCP task cancellation traverses sync-stage/live-loop native bridge and closes its read/lease',flush=True)


async def owner_close():
    with tempfile.TemporaryDirectory() as folder:
        pool=NativePool(Config(models={'l':spec(folder)}));worker=pool.workers[0]
        entered=threading.Event();exited=threading.Event();late_start=[]
        def hashing(cancel):
            entered.set()
            if not cancel.wait(1):late_start.append(True)
            exited.set()
        worker.start_process=hashing
        load=asyncio.create_task(pool.load('l'))
        assert await asyncio.to_thread(entered.wait,1)
        await asyncio.wait_for(pool.close(),1)
        assert exited.is_set() and not late_start and worker._startup_task is None
        assert worker.closed and not worker.running
        try:await asyncio.wait_for(load,1)
        except EngineError as exc:assert exc.code=='server_unavailable'
        else:raise AssertionError('Closed pool completed a model load')
        assert worker.pending==worker.active==pool.waiting==0
        # A selected lease waiting on the worker lock must not start after close.
        pool=NativePool(Config(models={'l':spec(folder)}));worker=pool.workers[0];called=[]
        worker.start_process=lambda cancel:called.append(True)
        await worker.lock.acquire()
        load=asyncio.create_task(pool.load('l'));await asyncio.sleep(.01)
        assert worker.pending==1
        await asyncio.wait_for(pool.close(),1);worker.lock.release()
        try:await asyncio.wait_for(load,1)
        except EngineError as exc:assert exc.code=='server_unavailable'
        else:raise AssertionError('Queued startup survived pool close')
        assert not called and worker.pending==worker.active==0
        # Ordinary unload is reversible; only permanent close marks the worker closed.
        from types import SimpleNamespace
        pool=NativePool(Config(models={'l':spec(folder)}));worker=pool.workers[0];starts=[]
        class Process:
            def __init__(self):self.alive=True;self.stdin=SimpleNamespace(close=lambda:setattr(self,'alive',False))
            def poll(self):return None if self.alive else 0
            def wait(self,timeout=None):assert not self.alive;return 0
            def kill(self):self.alive=False
        def prepared(cancel):
            assert not cancel.is_set();starts.append(True)
            worker.process=Process();worker.url='http://owned-fixture';worker.token='test-only-not-a-service-key'
        worker.start_process=prepared
        await pool.load('l');assert worker.running and worker.ready
        await pool.unload('l');assert not worker.running and not worker.closed
        await pool.load('l');assert worker.running and worker.ready and len(starts)==2
        await pool.close();assert worker.closed and not worker.running
    print('PASS pool.close joins hashing before returning, blocks queued startup, and ordinary unload remains reloadable; zero processes launched',flush=True)


async def files_and_options():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        for name,content in [('first.gguf',b'GGUFfirst'),('second.gguf',b'GGUFsecond'),('runtime.exe',b'fixture launcher'),('runtime.dll',b'fixture implementation')]:
            (root/name).write_bytes(content)
        model=spec(folder);model.native.model_files=['first.gguf','second.gguf'];calls=[]
        worker=NativeWorker('native:l','l',model)
        class NoLaunch(Exception):pass
        def launch(args,**kwargs):calls.append(args);raise NoLaunch()
        try:
            with patch('titanos.native.executable_path',return_value=root/'runtime.exe'),patch('titanos.native.subprocess.Popen',side_effect=launch):
                try:worker.start_process()
                except NoLaunch:pass
                else:raise AssertionError('Expected intercepted launch boundary')
            assert set(worker.checkpoint_files)=={'first.gguf','second.gguf'}
            assert worker.checkpoint_files['second.gguf']['sha256']==hashlib.sha256(b'GGUFsecond').hexdigest()
            assert set(worker.runtime_files)=={'runtime.exe','runtime.dll'}
            args=calls[0];assert args[args.index('--batch-size')+1]=='256' and args[args.index('--ubatch-size')+1]=='128'
            for values in [{'model_file':'first.gguf','model_files':['second.gguf']},{'model_file':'first.gguf','batch_size':2,'ubatch_size':3}]:
                try:NativeOptions(**values)
                except ValidationError:pass
                else:raise AssertionError('Invalid shard/batch configuration accepted')
        finally:await worker.stop()
    print('PASS both GGUF shards and launcher/DLL fingerprinted; batch flags built; zero process launches',flush=True)


if __name__=='__main__':
    asyncio.run(boundary());bridge_and_reload();asyncio.run(bridge_cancellation());asyncio.run(owner_close());asyncio.run(files_and_options())
