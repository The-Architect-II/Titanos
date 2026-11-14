"""Actual direct/replica strict paths with inert runtimes, pipes and processes only."""
from pathlib import Path
import copy,os,sys,threading,unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from titanos.config import Config,ModelSpec,WorkerSpec
from titanos.engine import Engine,EngineError
from titanos.replicas import ReplicaEngine,Worker,_worker_main
from titanos.settings import Settings
import titanos.engine as engine_module
import titanos.replicas as replica_module
assert Path(engine_module.__file__).resolve().is_relative_to(ROOT/'src')
assert Path(replica_module.__file__).resolve().is_relative_to(ROOT/'src')

def config(workers=0):
    return Config(models={name:ModelSpec(path='.',task='chat',device='cpu') for name in ('m','other')},
                  workers={name:WorkerSpec(models=['m'],device='cpu',timeout=1) for name in ('one','two')[:workers]})

class Runtime:
    device='cpu'
    runtime_fault=None
    def __init__(self,error=None):self.calls=[];self.closed=False;self.error=error
    def infer(self,task,data,emit=None,cancel=None):
        self.calls.append(copy.deepcopy(data))
        if self.error:raise self.error
        if emit:emit('part')
        return {'text':'done'}
    def details(self):return {'device':'cpu','backend':'fake'}
    def close(self):self.closed=True

def warm(engine,name='m',runtime=None):
    runtime=runtime or Runtime();engine.current=name;engine.runtime=runtime;engine.states[name]='loaded';return runtime

class Process:
    pid=123
    def __init__(self):self.alive=True;self.unknown=False
    def is_alive(self):
        if self.unknown:raise OSError('inert unreadable process')
        return self.alive

class Pipe:
    def __init__(self,worker,mode='normal'):
        self.worker=worker;self.mode=mode;self.sent=[];self.replies=[];self.closed=False
    def send(self,message):
        self.sent.append(message)
        command,name,task,data=message[:4]
        meta={'pid':123,'current_model':None if command=='unload' else name,'completed_requests':1,'runtime':{}}
        if len(message)==4:
            self.replies=[('done',({'text':'done'},meta))];return
        expected={'request_id':message[4]['request_id'],'model':name,'pid':123}
        ack=('admitted',expected);done=('done',({'text':'done'},meta))
        if self.mode in ('normal','recv-replaced','recv-dead'):self.replies=[ack,('text','part'),done]
        elif self.mode=='missing':self.replies=[done]
        elif self.mode=='duplicate':self.replies=[ack,ack,done]
        elif self.mode=='wrong-call':self.replies=[('admitted',{**expected,'request_id':'0'*32}),done]
        elif self.mode=='wrong-pid':self.replies=[('admitted',{**expected,'pid':999}),done]
        elif self.mode=='malformed-ack':self.replies=[('admitted',None),done]
        elif self.mode=='malformed-result':self.replies=[ack,('done',None)]
        elif self.mode=='text-first':self.replies=[('text','early'),ack,done]
        elif self.mode=='ready-first':self.replies=[('ready',meta),ack,done]
        elif self.mode=='malformed-envelope':self.replies=[None]
        elif self.mode=='cold-child':self.replies=[('error',('Not loaded',409,'model_not_loaded',meta))]
        elif self.mode=='provider-error':self.replies=[ack,('error',('Provider failed',500,'inference_error',meta))]
        elif self.mode=='fatal-exit':self.replies=[ack,('error',('Device failed',503,'device_unavailable',{**meta,'runtime_fault':{'fatal':True}}))]
        elif self.mode=='death':self.worker.process.alive=False;self.replies=[]
        elif self.mode=='unknown':self.worker.process.unknown=True;self.replies=[]
        elif self.mode=='replaced':self.worker.process=Process();self.replies=[ack,done]
        else:raise AssertionError(self.mode)
    def poll(self,timeout):return bool(self.replies)
    def recv(self):
        reply=self.replies.pop(0)
        if self.mode=='recv-replaced':
            self.worker.process=Process();self.worker.connection=Pipe(self.worker)
        elif self.mode=='recv-dead':self.worker.process.alive=False
        elif self.mode=='fatal-exit' and reply[0]=='error':self.worker.process.alive=False
        return reply
    def close(self):self.closed=True

def ready_worker(worker,mode='normal'):
    worker.process=Process();worker.connection=Pipe(worker,mode);worker.cancel=threading.Event()
    worker.meta={'pid':123,'current_model':'m','completed_requests':0};worker.start=Mock(side_effect=AssertionError('Strict call started worker'))
    def stop(force=False):
        if worker.process:worker.process.alive=False
        worker.process=worker.connection=None;worker.meta['current_model']=None
    worker.stop=Mock(side_effect=stop)
    return worker

class Tests(unittest.TestCase):
    def test_default_lazy_path_and_control_copy(self):
        engine=Engine(config());runtime=Runtime()
        engine._load=Mock(side_effect=lambda name:warm(engine,name,runtime))
        original={'require_loaded':False,'max_tokens':3}
        self.assertEqual(engine.infer('m','chat',original),{'text':'done'})
        engine._load.assert_called_once_with('m');self.assertEqual(runtime.calls,[{'max_tokens':3}]);self.assertIn('require_loaded',original)

    def test_direct_cold_wrong_missing_and_fault_have_no_load_or_infer(self):
        for kind in ('cold','wrong','missing','fault'):
            with self.subTest(kind=kind):
                engine=Engine(config());runtime=Runtime()
                if kind!='cold':warm(engine,'other' if kind=='wrong' else 'm',runtime)
                if kind=='missing':engine.runtime=None
                if kind=='fault':runtime.runtime_fault={'code':'failed'}
                engine._load=Mock(side_effect=AssertionError('Strict loaded'))
                callbacks=Mock()
                with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True,on_admitted=callbacks,on_finished=callbacks)
                self.assertEqual(raised.exception.code,'device_unavailable' if kind=='fault' else 'model_not_loaded')
                self.assertEqual(raised.exception.status,503 if kind=='fault' else 409)
                self.assertEqual(runtime.calls,[]);engine._load.assert_not_called();callbacks.assert_not_called()

    def test_direct_unload_wins_and_cancelled_admission(self):
        engine=Engine(config());runtime=warm(engine);engine.unload('m')
        with patch.object(engine,'_load',side_effect=AssertionError('Reload')):
            with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True)
            self.assertEqual(raised.exception.code,'model_not_loaded')
        self.assertTrue(runtime.closed)
        warm(engine);cancel=threading.Event();cancel.set();admitted=Mock()
        with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},cancel=cancel,require_loaded=True,on_admitted=admitted)
        self.assertEqual(raised.exception.code,'cancelled');admitted.assert_not_called()

    def test_direct_ownership_survives_result_and_same_thread_releases(self):
        engine=Engine(config());runtime=warm(engine);outcome=threading.Event();release=threading.Event();unload_entered=threading.Event();owners=[]
        def admitted():owners.append(threading.get_ident());self.assertTrue(engine.lock.locked())
        def finished(result,error):
            self.assertEqual(result,{'text':'done'});self.assertIsNone(error);self.assertTrue(engine.lock.locked())
            owners.append(threading.get_ident());outcome.set();self.assertTrue(release.wait(3))
        def unload():unload_entered.set();return engine.unload('m')
        with ThreadPoolExecutor(max_workers=2) as pool:
            execution=pool.submit(engine.infer,'m','chat',{},require_loaded=True,on_admitted=admitted,on_finished=finished)
            try:
                self.assertTrue(outcome.wait(3));self.assertFalse(engine.lock.acquire(False));pending=pool.submit(unload)
                self.assertTrue(unload_entered.wait(3));self.assertFalse(pending.done());self.assertFalse(runtime.closed)
            finally:release.set()
            self.assertEqual(execution.result(3),{'text':'done'});pending.result(3)
        self.assertEqual(len(set(owners)),1);self.assertTrue(runtime.closed)

    def test_direct_error_callback_inside_lock_preserves_primary(self):
        primary=EngineError('provider failed',503,'device_unavailable');engine=Engine(config());warm(engine,runtime=Runtime(primary));seen=[]
        def finished(result,error):
            seen.append((result,error));self.assertTrue(engine.lock.locked());raise ValueError('secondary')
        with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True,on_finished=finished)
        self.assertIs(raised.exception,primary);self.assertEqual(seen,[(None,primary)]);self.assertTrue(primary.__notes__)
        self.assertFalse(engine.lock.locked())

    def test_direct_admission_callback_failure_does_not_execute_provider(self):
        engine=Engine(config());runtime=warm(engine);seen=[];error=EngineError('lost client',499,'cancelled')
        def admitted():raise error
        with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True,on_admitted=admitted,on_finished=lambda result,exc:seen.append(exc))
        self.assertIs(raised.exception,error);self.assertEqual(runtime.calls,[]);self.assertEqual(seen,[error])

    def test_replica_default_signature_and_local_strict_delegation(self):
        engine=ReplicaEngine(config(1));worker=engine.workers[0]
        calls=[];worker.call=lambda *args:calls.append(args) or 'old-double'
        self.assertEqual(engine.infer('m','chat',{'require_loaded':False}),'old-double');self.assertEqual(calls[0][3],{})
        runtime=warm(engine.local,'other');admitted=Mock();finished=Mock()
        engine.local._load=Mock(side_effect=AssertionError('Local reload'))
        self.assertEqual(engine.infer('other','chat',{'require_loaded':True},require_loaded=True,on_admitted=admitted,on_finished=finished),{'text':'done'})
        admitted.assert_called_once();finished.assert_called_once();self.assertEqual(runtime.calls,[{}])

    def test_strict_worker_rejections_never_start_or_send(self):
        for kind in ('cold','wrong','dead','unknown','fault','no-pipe','foreign'):
            with self.subTest(kind=kind):
                engine=ReplicaEngine(config(2));one=ready_worker(engine.workers[0]);two=ready_worker(engine.workers[1]);pipe=one.connection
                if kind=='cold':one.process=None;one.meta['current_model']=None
                if kind=='wrong':one.meta['current_model']='other'
                if kind=='dead':one.process.alive=False
                if kind=='unknown':one.process.unknown=True
                if kind=='fault':one.runtime_fault={'failed':True}
                if kind=='no-pipe':one.connection=None
                with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{'worker':'foreign' if kind=='foreign' else 'one'},require_loaded=True)
                self.assertIn(raised.exception.code,{'model_not_loaded','worker_not_found','worker_unavailable','worker_still_running','device_unavailable'})
                one.start.assert_not_called();two.start.assert_not_called();self.assertEqual(pipe.sent,[]);self.assertFalse(one.busy)

    def test_warm_busy_never_selects_cold_idle_and_keeps_queue(self):
        engine=ReplicaEngine(config(2),Settings(replica_queue_capacity=1,replica_queue_timeout=3))
        one=ready_worker(engine.workers[0]);cold=engine.workers[1];cold.start=Mock(side_effect=AssertionError('Cold fallback'));one.busy=True
        waiting=threading.Event();original_wait=engine.condition.wait
        def wait(timeout):waiting.set();return original_wait(timeout)
        engine.condition.wait=wait
        with ThreadPoolExecutor(max_workers=1) as pool:
            execution=pool.submit(engine.infer,'m','chat',{},require_loaded=True)
            try:
                self.assertTrue(waiting.wait(3))
                with engine.condition:
                    self.assertEqual(engine.pending,1);self.assertFalse(cold.busy);one.busy=False;engine.condition.notify_all()
                self.assertEqual(execution.result(3),{'text':'done'})
            finally:
                with engine.condition:one.busy=False;engine.condition.notify_all()
        cold.start.assert_not_called();self.assertEqual(engine.pending,0)

    def test_replica_finish_retains_busy_and_nonfatal_error_is_usable(self):
        engine=ReplicaEngine(config(1));worker=ready_worker(engine.workers[0]);worker.error='Earlier cancellation';worker.failed_model='m'
        outcome=threading.Event();release=threading.Event()
        def finished(result,error):
            self.assertTrue(worker.busy);self.assertIsNone(error);outcome.set();self.assertTrue(release.wait(3))
        with ThreadPoolExecutor(max_workers=1) as pool:
            execution=pool.submit(engine.infer,'m','chat',{},require_loaded=True,on_finished=finished)
            try:
                self.assertTrue(outcome.wait(3))
                with engine.condition:self.assertTrue(worker.busy)
                self.assertFalse(execution.done())
            finally:release.set()
            execution.result(3)
        self.assertFalse(worker.busy);worker.start.assert_not_called()

    def test_queued_strict_cancel_residency_loss_and_capacity_timeout(self):
        for change,code in [('cancel','cancelled'),('unload','model_not_loaded'),('death','worker_unavailable')]:
            with self.subTest(change=change):
                engine=ReplicaEngine(config(1),Settings(replica_queue_capacity=1,replica_queue_timeout=3));worker=ready_worker(engine.workers[0]);pipe=worker.connection
                worker.busy=True;cancel=threading.Event();waiting=threading.Event();old_wait=engine.condition.wait
                def wait(timeout):waiting.set();return old_wait(timeout)
                engine.condition.wait=wait
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending=pool.submit(engine.infer,'m','chat',{},cancel=cancel,require_loaded=True)
                    self.assertTrue(waiting.wait(3))
                    with engine.condition:
                        if change=='cancel':cancel.set()
                        elif change=='unload':worker.meta['current_model']=None
                        else:worker.process.alive=False
                        worker.busy=False;engine.condition.notify_all()
                    with self.assertRaises(EngineError) as raised:pending.result(3)
                self.assertEqual(raised.exception.code,code);self.assertEqual(pipe.sent,[]);self.assertEqual(engine.pending,0);worker.start.assert_not_called()
        engine=ReplicaEngine(config(1),Settings(replica_queue_capacity=1,replica_queue_timeout=1));worker=ready_worker(engine.workers[0]);worker.busy=True;engine.pending=1
        with self.assertRaises(EngineError) as raised:
            with engine.acquire('m',require_loaded=True):pass
        self.assertEqual(raised.exception.code,'rate_limit_error')
        engine.pending=0;clock=[0.]
        def wait(timeout):clock[0]=2.
        engine.condition.wait=wait
        with patch.object(replica_module,'time',NS(monotonic=lambda:clock[0])):
            with self.assertRaises(EngineError) as raised:
                with engine.acquire('m',require_loaded=True):pass
        self.assertEqual(raised.exception.code,'worker_busy');self.assertEqual(engine.pending,0)

    def test_replica_ordinary_unload_waits_for_response_release(self):
        engine=ReplicaEngine(config(1));worker=ready_worker(engine.workers[0]);worker.start=Mock(return_value=None)
        result_ready=threading.Event();release=threading.Event();unload_waiting=threading.Event();old_wait=engine.condition.wait
        def wait(timeout):unload_waiting.set();return old_wait(timeout)
        engine.condition.wait=wait
        def finished(result,error):result_ready.set();self.assertTrue(release.wait(3))
        with ThreadPoolExecutor(max_workers=2) as pool:
            inference=pool.submit(engine.infer,'m','chat',{},require_loaded=True,on_finished=finished)
            try:
                self.assertTrue(result_ready.wait(3));unload=pool.submit(engine.unload,'m','one')
                self.assertTrue(unload_waiting.wait(3));worker.start.assert_not_called();self.assertFalse(unload.done())
            finally:release.set()
            inference.result(3);unload.result(3)
        self.assertIsNone(worker.meta['current_model']);self.assertEqual(worker.start.call_count,1)
        with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True)
        self.assertEqual(raised.exception.code,'model_not_loaded');self.assertEqual(worker.start.call_count,1)

    def test_fatal_error_reply_after_child_exit_keeps_fault(self):
        engine=ReplicaEngine(config(1));worker=ready_worker(engine.workers[0],'fatal-exit');finished=Mock()
        with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True,on_finished=finished)
        self.assertEqual(raised.exception.code,'device_unavailable');self.assertEqual(worker.runtime_fault,{'fatal':True})
        worker.start.assert_not_called();worker.stop.assert_called_once_with(force=True);finished.assert_called_once()

    def test_ack_protocol_faults_retire_without_restart_or_replay(self):
        for mode in ('missing','duplicate','wrong-call','wrong-pid','malformed-ack','text-first','ready-first','malformed-envelope','malformed-result','death','unknown','replaced'):
            with self.subTest(mode=mode):
                engine=ReplicaEngine(config(1));worker=ready_worker(engine.workers[0],mode);pipe=worker.connection;admitted=Mock();finished=Mock()
                with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True,on_admitted=admitted,on_finished=finished)
                self.assertEqual(raised.exception.code,'worker_unavailable');worker.start.assert_not_called();self.assertEqual(len(pipe.sent),1);self.assertFalse(worker.busy)
                self.assertEqual(finished.call_count,admitted.call_count)
                if mode=='replaced':worker.stop.assert_not_called()
                else:worker.stop.assert_called_once_with(force=True)

    def test_child_refusal_is_not_admission_and_provider_error_settles(self):
        for mode,code in [('cold-child','model_not_loaded'),('provider-error','inference_error')]:
            engine=ReplicaEngine(config(1));worker=ready_worker(engine.workers[0],mode);admitted=Mock();finished=Mock()
            with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True,on_admitted=admitted,on_finished=finished)
            self.assertEqual(raised.exception.code,code);self.assertEqual(admitted.call_count,0 if mode=='cold-child' else 1)
            self.assertEqual(finished.call_count,admitted.call_count);worker.start.assert_not_called()

    def test_recv_cannot_admit_changed_or_dead_retained_worker(self):
        for mode in ('recv-replaced','recv-dead'):
            with self.subTest(mode=mode):
                engine=ReplicaEngine(config(1));worker=ready_worker(engine.workers[0],mode);pipe=worker.connection;admitted=Mock();finished=Mock()
                with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True,on_admitted=admitted,on_finished=finished)
                self.assertEqual(raised.exception.code,'worker_unavailable');self.assertEqual(len(pipe.sent),1)
                admitted.assert_not_called();finished.assert_not_called();worker.start.assert_not_called()
                if mode=='recv-replaced':worker.stop.assert_not_called()
                else:worker.stop.assert_called_once_with(force=True)

    def test_replica_admission_callback_error_drains_child_before_finish(self):
        engine=ReplicaEngine(config(1));worker=ready_worker(engine.workers[0]);pipe=worker.connection;error=EngineError('cancelled',499,'cancelled');seen=[]
        def admitted():raise error
        def finished(result,failed):self.assertEqual(pipe.replies,[]);self.assertTrue(worker.busy);seen.append(failed)
        with self.assertRaises(EngineError) as raised:engine.infer('m','chat',{},require_loaded=True,on_admitted=admitted,on_finished=finished)
        self.assertIs(raised.exception,error);self.assertEqual(seen,[error]);self.assertTrue(worker.cancel.is_set())

    def test_actual_child_ack_under_lock_and_passive_metadata(self):
        for loaded in (True,False):
            with self.subTest(loaded=loaded):
                cfg=config();engine=Engine(cfg);runtime=warm(engine) if loaded else Runtime();engine._load=Mock(side_effect=AssertionError('Child lazy load'))
                request_id='a'*32;sent=[]
                requests=iter([('infer','m','chat',{'worker':'one','require_loaded':False,'max_tokens':3},{'require_loaded':True,'request_id':request_id}),('close',None,None,None)])
                def send(value):
                    if value[0]=='admitted':
                        self.assertTrue(engine.lock.locked());self.assertEqual(runtime.calls,[])
                    sent.append(value)
                pipe=NS(recv=lambda:next(requests),send=send,close=lambda:None)
                assignment={'models':['m'],'gpus':[],'device':'cpu','mode':'single','threads':1,'profile':False}
                fake_os=NS(name='nt',environ={},getpid=lambda:123)
                with patch.object(replica_module,'Engine',return_value=engine),patch.object(replica_module,'os',fake_os):
                    _worker_main(pipe,threading.Event(),cfg.model_dump(),assignment)
                engine._load.assert_not_called()
                if loaded:
                    self.assertEqual([r[0] for r in sent],['ready','admitted','text','done'])
                    self.assertEqual(sent[1][1],{'request_id':request_id,'model':'m','pid':123});self.assertEqual(runtime.calls,[{'max_tokens':3}])
                else:
                    self.assertEqual([r[0] for r in sent],['ready','error']);self.assertEqual(sent[-1][1][2],'model_not_loaded')

if __name__=='__main__':unittest.main()
