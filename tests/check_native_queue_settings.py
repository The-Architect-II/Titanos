"""Exercise real queue admission/wait accounting with model-free worker doubles."""
import asyncio
import os
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from titanos.config import Config, ModelSpec, NativeOptions, write_config
from titanos.engine import EngineError
from titanos.native import NativePool
from titanos.server import create_app
from titanos.settings import Settings


class Worker:
    def __init__(self,spec):
        self.spec=spec;self.model='l';self.name='native:l';self.active=self.pending=self.completed=self.starts=0
        self.running=True;self.broken=False;self.lock=asyncio.Lock()
    async def start(self):self.starts+=1
    async def close(self):self.running=False


def pool_for(capacity,timeout,slots=1):
    config=Config(models={'l':ModelSpec(path='.',task='chat',backend='llama-cpp',
        native=NativeOptions(model_file='never-loaded.gguf',slots=slots))})
    pool=NativePool(config,settings=Settings(native_queue_capacity=capacity,native_queue_timeout=timeout))
    worker=Worker(config.models['l']);pool.workers=[worker]
    return pool,worker


async def waiting(pool,count):
    async with asyncio.timeout(1):
        while pool.waiting!=count:await asyncio.sleep(0)


async def use(pool,entered=None,release=None):
    async with pool.lease('l'):
        if entered is not None:entered.set()
        if release is not None:await release.wait()


async def capacity_and_wakeup():
    for capacity in [1,3]:
        pool,worker=pool_for(capacity,2,slots=2)
        release=asyncio.Event();parked=[]
        try:
            # Capacity1 must not prevent both declared active slots being used.
            async with pool.lease('l'):
                async with pool.lease('l'):
                    assert worker.active==2 and pool.waiting==0
                    entered=[asyncio.Event() for _ in range(capacity)]
                    parked=[asyncio.create_task(use(pool,event,release)) for event in entered]
                    await waiting(pool,capacity)
                    assert worker.active==2 and worker.starts==2
                    try:await use(pool)
                    except EngineError as exc:assert (exc.status,exc.code)==(503,'queue_full')
                    else:raise AssertionError('A full waiting queue admitted another request')
                    assert pool.waiting==capacity and not any(event.is_set() for event in entered)
                # Releasing a slot must wake an admitted waiter without polling.
                async with asyncio.timeout(1):
                    while not any(event.is_set() for event in entered):await asyncio.sleep(0)
                assert worker.active==2 and pool.waiting==capacity-1
                release.set()
            await asyncio.gather(*parked)
            assert pool.waiting==worker.active==worker.pending==0
            assert worker.starts==2+capacity
        finally:
            release.set()
            for task in parked:
                if not task.done():task.cancel()
            await asyncio.gather(*parked,return_exceptions=True);await pool.close()


async def timeout_and_cancellation():
    pool,worker=pool_for(1,.02)
    try:
        async with pool.lease('l'):
            try:await use(pool)
            except EngineError as exc:assert (exc.status,exc.code)==(503,'queue_timeout')
            else:raise AssertionError('Configured short wait did not time out')
            assert pool.waiting==0 and worker.active==1 and worker.pending==0 and worker.starts==1
        await use(pool)
        assert pool.waiting==worker.active==worker.pending==0 and worker.starts==2
    finally:await pool.close()

    # A long queue allowance must not renew a shorter full-request deadline.
    pool,worker=pool_for(2,2)
    data={'model':'l','messages':[{'role':'user','content':'controlled queue request'}],'max_tokens':4}
    try:
        async with pool.lease('l'):
            pending=asyncio.create_task(use(pool));await waiting(pool,1);pending.cancel()
            try:await pending
            except asyncio.CancelledError:pass
            else:raise AssertionError('Waiting caller cancellation was swallowed')
            assert pool.waiting==0 and worker.active==1 and worker.starts==1
            try:await pool.complete(data,deadline=time.monotonic()+.02)
            except EngineError as exc:assert (exc.status,exc.code)==(504,'deadline_exceeded')
            else:raise AssertionError('Queue timeout replaced the request deadline')
            assert pool.waiting==0 and worker.active==1 and worker.starts==1
            cancel=threading.Event();pending=asyncio.create_task(pool.complete(data,cancel=cancel))
            await waiting(pool,1);cancel.set()
            try:await asyncio.wait_for(pending,1)
            except EngineError as exc:assert (exc.status,exc.code)==(499,'cancelled')
            else:raise AssertionError('Shared caller cancellation failed while queued')
            assert pool.waiting==0 and worker.active==1 and worker.starts==1
    finally:await pool.close()


def settings_and_reload():
    defaults=Settings.from_env({})
    assert (defaults.native_queue_capacity,defaults.native_queue_timeout)==(32,30)
    configured=Settings.from_env({'TITANOS_NATIVE_QUEUE_CAPACITY':'7','TITANOS_NATIVE_QUEUE_TIMEOUT':'.125'})
    assert (configured.native_queue_capacity,configured.native_queue_timeout)==(7,.125)
    for name,values in [('native_queue_capacity',[0,-1,True,1.5,'2',float('nan'),float('inf')]),
                        ('native_queue_timeout',[0,-1,True,'2',float('nan'),float('inf')])]:
        for value in values:
            try:Settings(**{name:value})
            except ValueError:pass
            else:raise AssertionError((name,value,'Invalid queue setting accepted'))
    for key,value in [('TITANOS_NATIVE_QUEUE_CAPACITY','1.5'),('TITANOS_NATIVE_QUEUE_TIMEOUT','nan'),('TITANOS_NATIVE_QUEUE_TIMEOUT','0')]:
        try:Settings.from_env({key:value})
        except ValueError:pass
        else:raise AssertionError('Invalid environment setting accepted')
    with patch.dict(os.environ,{'TITANOS_NATIVE_QUEUE_CAPACITY':'7','TITANOS_NATIVE_QUEUE_TIMEOUT':'2'}):
        standalone=NativePool(Config())
        os.environ['TITANOS_NATIVE_QUEUE_CAPACITY']='19';os.environ['TITANOS_NATIVE_QUEUE_TIMEOUT']='4'
        assert (standalone.queue_capacity,standalone.queue_timeout)==(7,2)
        asyncio.run(standalone.close())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);registry=root/'models.json';write_config(registry,Config())
            settings=Settings(native_queue_capacity=2,native_queue_timeout=.25,queue_capacity=9)
            app=create_app(Config(),settings=settings,output_dir=root/'outputs',registry_path=registry)
            with TestClient(app,base_url='http://127.0.0.1') as client:
                old=app.state.native
                assert (old.queue_capacity,old.queue_timeout)==(2,.25)
                info=client.get('/api/info').json()
                assert info['jobs']['capacity']==9
                assert info['settings']['native_queue_capacity']=={'value':2,'default':32,'env':'TITANOS_NATIVE_QUEUE_CAPACITY','type':'int'}
                assert info['settings']['native_queue_timeout']=={'value':.25,'default':30.0,'env':'TITANOS_NATIVE_QUEUE_TIMEOUT','type':'float'}
                os.environ['TITANOS_NATIVE_QUEUE_CAPACITY']='23'
                assert client.post('/api/models/reload').status_code==200
                new=app.state.native;assert new is not old and old.closed
                assert (new.queue_capacity,new.queue_timeout)==(2,.25)
                assert client.get('/api/info').json()['settings']['native_queue_capacity']['value']==2


if __name__=='__main__':
    settings_and_reload();asyncio.run(capacity_and_wakeup());asyncio.run(timeout_and_cancellation())
    print('PASS configurable native waiting admission/full rejection, separate active slots, timeout/cancel cleanup and wakeup, shared deadline, immutable app settings/reload and discovery; zero model/process launches')
