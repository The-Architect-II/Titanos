"""Real threaded admission/wakeup with inert workers; never loads a model."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import os,tempfile,threading,time
from fastapi.testclient import TestClient
from titanos.config import Config,ModelSpec,WorkerSpec,write_config
from titanos.engine import EngineError
from titanos.replicas import ReplicaEngine
from titanos.settings import Settings
from titanos.server import create_app


def config():
    return Config(models={n:ModelSpec(path='.',task='image',backend='diffusers',device='cpu') for n in ['a','b']},
                  workers={n:WorkerSpec(models=[n],device='cpu') for n in ['a','b']})


def pending(engine,count):
    until=time.monotonic()+2
    while True:
        with engine.condition:
            if engine.pending==count:return
        assert time.monotonic()<until,'Waiting-count observation timed out'
        time.sleep(.001)


def request(engine,name,release=None,cancel=None):
    try:
        with engine.acquire(name,cancel=cancel):
            if release is not None:assert release.wait(2),'Controlled active request did not release'
        return 'ok'
    except EngineError as exc:return exc.status,exc.code


def capacity_and_wakeup():
    for capacity in [1,3]:
        engine=ReplicaEngine(config(),Settings(replica_queue_capacity=capacity,replica_queue_timeout=2))
        release=threading.Event()
        try:
            with ThreadPoolExecutor(max_workers=capacity) as pool:
                try:
                    with engine.acquire('a'):
                        assert engine.pending==0
                        waiters=[pool.submit(request,engine,'a',release) for _ in range(capacity)]
                        pending(engine,capacity)
                        assert request(engine,'a')==(429,'rate_limit_error')
                        # Full waiting capacity must not reject an idle other worker.
                        assert request(engine,'b')=='ok' and engine.pending==capacity
                    pending(engine,capacity-1)
                finally:release.set()
                assert all(f.result(timeout=2)=='ok' for f in waiters)
            assert engine.pending==0 and not any(w.busy for w in engine.workers)
            assert all(w.process is None for w in engine.workers)
        finally:release.set();engine.close()


def timeout_cancel_close():
    engine=ReplicaEngine(config(),Settings(replica_queue_capacity=1,replica_queue_timeout=.03))
    try:
        with engine.acquire('a'):
            assert request(engine,'a')==(503,'worker_busy')
            assert engine.pending==0
        assert request(engine,'a')=='ok'
    finally:engine.close()
    for operation in ['cancel','close']:
        engine=ReplicaEngine(config(),Settings(replica_queue_capacity=1,replica_queue_timeout=2));cancel=threading.Event()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                with engine.acquire('a'):
                    future=pool.submit(request,engine,'a',None,cancel);pending(engine,1)
                    if operation=='cancel':cancel.set()
                    else:engine.close()
                    expected=(499,'cancelled') if operation=='cancel' else (503,'invalid_request_error')
                    assert future.result(timeout=1)==expected
                    assert engine.pending==0
        finally:engine.close()


def settings_and_reload():
    defaults=Settings.from_env({});assert (defaults.replica_queue_capacity,defaults.replica_queue_timeout)==(32,30)
    selected=Settings.from_env({'TITANOS_REPLICA_QUEUE_CAPACITY':'1','TITANOS_REPLICA_QUEUE_TIMEOUT':'.125'})
    assert (selected.replica_queue_capacity,selected.replica_queue_timeout)==(1,.125)
    for name,values in [('replica_queue_capacity',[0,-1,True,1.5,'2',float('nan')]),('replica_queue_timeout',[0,-1,True,'2',float('inf')])]:
        for value in values:
            try:Settings(**{name:value})
            except ValueError:pass
            else:raise AssertionError((name,value,'Invalid setting accepted'))
    with patch.dict(os.environ,{'TITANOS_REPLICA_QUEUE_CAPACITY':'7','TITANOS_REPLICA_QUEUE_TIMEOUT':'2'}):
        direct=ReplicaEngine(config());os.environ['TITANOS_REPLICA_QUEUE_CAPACITY']='19'
        assert (direct.queue_capacity,direct.queue_timeout)==(7,2);direct.close()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);registry=root/'models.json';cfg=config();write_config(registry,cfg)
            app=create_app(cfg,settings=selected,output_dir=root/'output',registry_path=registry)
            with TestClient(app,base_url='http://127.0.0.1') as client:
                old=app.state.engine.target
                assert (old.queue_capacity,old.queue_timeout)==(1,.125)
                info=client.get('/api/info').json()['settings']
                assert info['replica_queue_capacity']=={'value':1,'default':32,'env':'TITANOS_REPLICA_QUEUE_CAPACITY','type':'int'}
                assert client.post('/api/models/reload').status_code==200
                new=app.state.engine.target;assert new is not old and old.closed
                assert (new.queue_capacity,new.queue_timeout)==(1,.125)
                assert all(w.process is None for w in old.workers+new.workers)


if __name__=='__main__':
    settings_and_reload();capacity_and_wakeup();timeout_cancel_close()
    print('PASS replica settings/env/reload, waiting-only capacity, idle-worker admission, wakeup, timeout, cancellation and close; zero model/process launches')
