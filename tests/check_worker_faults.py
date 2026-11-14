"""Fatal/blocked explicit-worker control tests using only process and pipe doubles."""
import multiprocessing as mp,threading,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from titanos.config import Config,ModelSpec,WorkerSpec
from titanos.engine import EngineError
from titanos.replicas import ReplicaEngine,Worker,process_running,process_identity,worker_main

FAULT={'code':'device_unavailable','hresult':'0x887A0006','component':'unet','pid':901,'message':'controlled native fault'}
class Process:
    def __init__(self,stoppable=False,unknown=False):self.pid=901;self.alive=True;self.stoppable=stoppable;self.unknown=unknown;self.actions=[]
    def is_alive(self):
        if self.unknown:raise OSError('process status unavailable')
        return self.alive
    def join(self,timeout):self.actions.append(('join',timeout))
    def terminate(self):
        self.actions.append(('terminate',))
        if self.stoppable:self.alive=False
    def kill(self):
        self.actions.append(('kill',))
        if self.stoppable:self.alive=False
class Pipe:
    def __init__(self,reply=None,error=None):self.reply=reply;self.error=error;self.sent=[];self.closed=False
    def send(self,value):
        self.sent.append(value)
        if self.error:raise self.error
    def poll(self,timeout):return self.reply is not None
    def recv(self):return self.reply
    def close(self):self.closed=True
def config(two=False):
    model=ModelSpec(path='controlled',task='image',backend='onnx-diffusers',device='dml',load_options={'device_id':0,'adapter_luid':'00000000:000190ef'})
    workers={'one':WorkerSpec(models=['image'],device='dml')}
    if two:workers['two']=WorkerSpec(models=['image'],device='dml')
    return Config(models={'image':model},workers=workers)
class Tests(unittest.TestCase):
    def setUp(self):
        # Even a mistakenly entered taskkill path is a double, never a real command.
        self.command=patch('subprocess.run',return_value=NS(returncode=0));self.command.start()
    def tearDown(self):self.command.stop()
    def worker(self,stoppable=False,unknown=False,reply=None,error=None):
        cfg=config();worker=Worker('one',cfg.workers['one'],cfg)
        worker.process=Process(stoppable,unknown);worker.connection=Pipe(reply,error);worker.cancel=threading.Event()
        worker.meta={'pid':901,'current_model':'image','completed_requests':0}
        return worker
    def test_unstoppable_process_identity_and_residency_are_retained(self):
        worker=self.worker();process,connection=worker.process,worker.connection
        with self.assertRaises(EngineError) as error:worker.stop()
        self.assertEqual(error.exception.code,'worker_still_running')
        self.assertIs(worker.process,process);self.assertIs(worker.connection,connection)
        self.assertEqual(worker.meta['pid'],901);self.assertEqual(worker.meta['current_model'],'image')
        self.assertTrue(worker.stop_failed);self.assertTrue(worker.meta['residency_unconfirmed'])
        sent=len(connection.sent)
        with self.assertRaises(EngineError):worker.call('infer','image','image',{})
        self.assertEqual(len(connection.sent),sent)
        info=worker.call('details','image');self.assertTrue(info['resident']);self.assertEqual(info['status'],'unavailable')
        process.stoppable=True;worker.call('unload','image')
        self.assertIsNone(worker.process);self.assertIsNone(worker.meta['current_model']);self.assertFalse(worker.unavailable)
    def test_unknown_process_state_is_not_reported_stopped(self):
        worker=self.worker(unknown=True);self.assertIsNone(process_running(worker.process))
        with self.assertRaises(EngineError):worker.stop(force=True)
        self.assertIsNotNone(worker.process);self.assertTrue(worker.unavailable)
        self.assertIsNone(worker.call('details','image')['resident'])
        engine=ReplicaEngine(config());engine.workers=[worker]
        status=engine.worker_status()[0];self.assertIsNone(status['running']);self.assertEqual(status['pid'],901)
        row=engine.catalog(loaded_only=True)['data'][0]
        self.assertEqual(row['status'],'unavailable');self.assertEqual(row['unconfirmed_workers'],['one'])
    def test_closed_never_started_process_reports_cache_but_cannot_target_pid(self):
        engine=ReplicaEngine(config());worker=engine.workers[0]
        process=mp.Process();process.close()  # No process is ever launched.
        worker.process=process;worker.stop_failed=True;worker.meta.update(pid=123456,current_model='image')
        with patch('subprocess.run') as command:
            status=engine.worker_status()[0]
            self.assertIsNone(status['running']);self.assertIsNone(status['pid'])
            self.assertFalse(status['pid_verified']);self.assertEqual(status['last_known_pid'],123456)
            details=engine.details('image','one')['process']
            self.assertIsNone(details['pid']);self.assertEqual(details['last_known_pid'],123456)
            with self.assertRaises(EngineError) as error:worker.stop(force=True)
            self.assertEqual(error.exception.code,'worker_still_running')
            self.assertEqual(command.call_count,0)
        self.assertIs(worker.process,process);self.assertEqual(worker.meta['current_model'],'image')
        self.assertIn('unverified',worker.meta['stop_error'])
        worker.process=None;engine.close()
    def test_failed_pid_getter_never_uses_cached_id_for_taskkill(self):
        worker=self.worker()
        class HiddenPID:
            def __init__(self):self.actions=[]
            @property
            def pid(self):raise ValueError('PID observer unavailable')
            def is_alive(self):return True
            def terminate(self):self.actions.append('terminate')
            def kill(self):self.actions.append('kill')
            def join(self,timeout):self.actions.append(('join',timeout))
        process=HiddenPID();worker.process=process;worker.meta['pid']=987654
        with patch('subprocess.run') as command:
            with self.assertRaises(EngineError):worker.stop(force=True)
            self.assertEqual(command.call_count,0)
        self.assertIn('terminate',process.actions);self.assertIn('kill',process.actions)
        self.assertEqual(process_identity(process,worker.meta),{'pid':None,'pid_verified':False,'last_known_pid':987654})
        self.assertIs(worker.process,process)
    def test_parent_unload_unknown_observer_uses_safe_stop_without_restarting_dead(self):
        engine=ReplicaEngine(config());worker=engine.workers[0]
        process=mp.Process();process.close();worker.process=process;worker.stop_failed=True;worker.meta.update(pid=123456,current_model='image')
        with patch('subprocess.run') as command:
            with self.assertRaises(EngineError) as error:engine.unload('image','one')
            self.assertEqual(error.exception.code,'worker_still_running');self.assertEqual(command.call_count,0)
        self.assertIs(worker.process,process)
        dead=Process();dead.alive=False;worker.process=dead;worker.stop_failed=False
        with patch.object(worker,'call',side_effect=AssertionError('Known dead worker must not restart')) as call:
            self.assertEqual(engine.unload('image','one')['status'],'unloaded');self.assertEqual(call.call_count,0)
        worker.process=None;engine.close()
    def test_fatal_reply_stops_owned_worker_and_stays_unavailable(self):
        meta={'pid':901,'current_model':'image','runtime_fault':FAULT}
        worker=self.worker(stoppable=True,reply=('error',('device failed',503,'device_unavailable',meta)))
        pipe=worker.connection;process=worker.process
        with self.assertRaises(EngineError) as error:worker.call('infer','image','image',{})
        self.assertEqual(error.exception.code,'device_unavailable');self.assertFalse(process.alive)
        self.assertIsNone(worker.process);self.assertTrue(worker.unavailable);self.assertIsNone(worker.meta['current_model'])
        self.assertFalse(worker.meta['runtime']['available']);self.assertFalse(worker.meta['runtime']['resident'])
        self.assertEqual(len(pipe.sent),1);self.assertEqual(pipe.sent[0][0],'infer')
        with self.assertRaises(EngineError):worker.call('infer','image','image',{})
        self.assertEqual(len(pipe.sent),1)
        self.assertFalse(worker.call('details','image')['resident'])
    def test_fatal_reply_preserves_primary_error_when_stop_fails(self):
        worker=self.worker(reply=('error',('native 887A0006',503,'device_unavailable',{'pid':901,'current_model':'image','runtime_fault':FAULT})))
        process=worker.process
        with self.assertRaises(EngineError) as error:worker.call('infer','image','image',{})
        self.assertEqual(str(error.exception),'native 887A0006');self.assertTrue(error.exception.__notes__)
        self.assertIs(worker.process,process);self.assertEqual(worker.meta['current_model'],'image')
    def test_unload_transport_failure_does_not_erase_last_resident_model(self):
        worker=self.worker(error=BrokenPipeError('controlled pipe failure'));process=worker.process;pipe=worker.connection
        with self.assertRaises(EngineError) as error:worker.call('unload','image')
        self.assertEqual(error.exception.code,'worker_unavailable');self.assertTrue(error.exception.__notes__)
        self.assertIs(worker.process,process);self.assertEqual(worker.meta['current_model'],'image')
        self.assertEqual([row[0] for row in pipe.sent],['unload'])
    def test_ordinary_model_error_keeps_existing_worker_available(self):
        worker=self.worker(reply=('error',('bad input',400,'invalid_request_error',{'pid':901,'current_model':'image'})))
        process=worker.process
        with self.assertRaises(EngineError) as error:worker.call('infer','image','image',{})
        self.assertEqual(error.exception.status,400);self.assertIs(worker.process,process);self.assertFalse(worker.unavailable)
        worker.connection.reply=('done',({'ok':True},{'pid':901,'current_model':'image'}))
        self.assertEqual(worker.call('infer','image','image',{}),{'ok':True})
        self.assertEqual(process.actions,[])
    def test_child_reports_unavailable_snapshot_and_skips_poisoned_unload(self):
        replies=[];commands=iter([('infer','image','image',{}),('close',None,None,None)])
        pipe=NS(send=lambda value:replies.append(value),recv=lambda:next(commands),close=lambda:None)
        class FailedEngine:
            current='image'
            def __init__(self,cfg):pass
            def infer(self,*a):raise EngineError('native fault',503,'device_unavailable')
            def _runtime_fault(self):return FAULT
            def details(self,name):return {'status':'unavailable','available':False,'resident':True,'runtime_fault':FAULT}
            def unload(self):raise AssertionError('Poisoned native unload must not be invoked')
        cfg=config();assignment=cfg.workers['one'].model_dump();assignment['resolved_gpus']=[]
        with patch('titanos.replicas.Engine',FailedEngine),patch.dict('os.environ',{}):
            worker_main(pipe,threading.Event(),cfg.model_dump(),assignment)
        error=next(value for kind,value in replies if kind=='error')
        self.assertEqual(error[2],'device_unavailable');self.assertFalse(error[3]['runtime']['available'])
        self.assertTrue(error[3]['runtime']['resident'])
    def test_scheduler_excludes_poisoned_worker_without_replaying_a_request(self):
        engine=ReplicaEngine(config(two=True));bad,healthy=engine.workers
        bad.runtime_fault=FAULT
        with engine.acquire('image') as selected:self.assertIs(selected,healthy)
        with self.assertRaises(EngineError):
            with engine.acquire('image','one'):self.fail('Poisoned worker admitted')
        self.assertNotEqual(engine.catalog()['data'][0]['status'],'unavailable')
        healthy.runtime_fault=FAULT
        self.assertEqual(engine.catalog()['data'][0]['status'],'unavailable')
        self.assertEqual(engine.catalog(loaded_only=True)['data'],[])
    def test_live_blocked_worker_is_visible_and_does_not_hide_other_cleanup(self):
        engine=ReplicaEngine(config(two=True));bad,other=engine.workers
        blocked=self.worker();bad.__dict__.update(blocked.__dict__)
        bad.meta['running']=False  # A stale child snapshot must not override a live handle.
        other.process=Process(stoppable=True);other.cancel=threading.Event();other.connection=Pipe()
        with self.assertRaises(EngineError):engine.close()
        status=engine.worker_status()[0];self.assertTrue(status['running']);self.assertEqual(status['pid'],901)
        self.assertTrue(status['unavailable']);self.assertIsNone(other.process)
        row=engine.catalog(loaded_only=True)['data'][0]
        self.assertEqual(row['status'],'unavailable');self.assertEqual(row['resident_workers'],['one']);self.assertEqual(row['loaded_workers'],[])

if __name__=='__main__':unittest.main()
