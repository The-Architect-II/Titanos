"""One inert regression for completed results, profiler failure and worker retirement."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import copy,json,os,sys,threading,unittest,weakref
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from titanos.config import Config,ModelSpec,WorkerSpec
from titanos.engine import EngineError
from titanos.replicas import _worker_main,Worker


class ProfilerPolicy(unittest.TestCase):
    def test_completed_result_and_primary_error_survive_bounded_retirement(self):
        payload={'audio':b'exact-already-completed-fixture'}
        for mode in ['exit_memory','events_memory','infer_and_exit','exit_runtime','infer_only','system_exit_and_exit']:
            with self.subTest(mode=mode):
                calls=[];profiles=[];secret='PRIVATE_EXCEPTION_REQUEST_CONFIG_VALUE';stop=SystemExit(7)
                class Profile:
                    def __enter__(self):calls.append('enter');return self
                    def __exit__(self,*args):
                        calls.append('exit')
                        if mode in {'exit_memory','infer_and_exit','system_exit_and_exit'}:raise MemoryError(secret)
                        if mode=='exit_runtime':raise RuntimeError(secret)
                    def events(self):
                        calls.append('events')
                        if mode=='events_memory':raise MemoryError(secret)
                        return []
                class Engine:
                    def __init__(self,cfg):self.current=None;self.runtime=None
                    def infer(self,*args):
                        calls.append('infer')
                        if mode=='system_exit_and_exit':raise stop
                        if mode in {'infer_and_exit','infer_only'}:raise EngineError(secret,422,'genuine_inference_error')
                        self.current='m';return payload
                    def details(self,name):calls.append('details');return {'model':'m'}
                    def _runtime_fault(self):return None
                    def unload(self):calls.append('unload')
                class Connection:
                    def __init__(self):self.sent=[];self.receives=0
                    def recv(self):
                        self.receives+=1
                        return ('infer','m','speech',{'input':secret,'worker':secret}) if self.receives==1 else ('close',None,None,None)
                    def send(self,value):
                        if value[0]=='done' and value[1][1].get('profiling_failure'):
                            assert profiles[0]() is None,'Failed collector still held before result marshalling'
                        self.sent.append(copy.deepcopy(value))
                    def close(self):calls.append('connection_close')
                connection=Connection();cfg=NS(models={'m':NS(backend='qwen3-tts',parallelism='single')})
                assignment={'gpus':[],'threads':1,'profile':True,'device':'cpu','mode':'single'}
                def create_profile(**kwargs):
                    value=Profile();profiles.append(weakref.ref(value));return value
                torch=NS(profiler=NS(profile=create_profile,ProfilerActivity=NS(CPU='cpu',CUDA='cuda')),cuda=NS(is_available=lambda:False),autograd=NS(DeviceType=NS(CUDA='cuda')))
                with patch.dict(sys.modules,{'torch':torch}),patch.dict(os.environ,dict(os.environ)),patch('titanos.replicas.Config',NS(model_validate=lambda value:cfg)),patch('titanos.replicas.Engine',Engine),self.assertLogs('titanos.replicas',level='ERROR') as logged:
                    if mode=='system_exit_and_exit':
                        with self.assertRaises(SystemExit) as raised:
                            _worker_main(connection,threading.Event(),{'private_config':secret},assignment)
                        self.assertIs(raised.exception,stop)
                    else:_worker_main(connection,threading.Event(),{'private_config':secret},assignment)
                if mode=='system_exit_and_exit':
                    self.assertFalse(any(kind in {'done','error'} for kind,_ in connection.sent))
                    self.assertEqual(calls.count('infer'),1);self.assertIn('connection_close',calls);self.assertNotIn('unload',calls)
                    self.assertNotIn(secret,'\n'.join(logged.output))
                    continue
                responses=[value for kind,value in connection.sent if kind in {'done','error'}];self.assertEqual(len(responses),1)
                kind,value=next((kind,value) for kind,value in connection.sent if kind in {'done','error'})
                meta=value[1] if kind=='done' else value[3]
                if mode in {'exit_memory','events_memory'}:
                    self.assertEqual(kind,'done');self.assertEqual(value[0],payload);self.assertEqual(meta['completed_requests'],1)
                    self.assertIsNone(meta['observed_gpu_indices']);self.assertEqual(meta['profiling_evidence'],{'requested':True,'available':False,'status':'failed'})
                    self.assertEqual(meta['profiling_failure']['phase'],'profile_exit' if mode=='exit_memory' else 'profile_events')
                else:
                    self.assertEqual(kind,'error');self.assertEqual(value[0],secret)
                    self.assertEqual(value[1:3],(422,'genuine_inference_error') if mode.startswith('infer') else (500,'worker_error'))
                self.assertEqual(calls.count('infer'),1)
                if mode!='infer_only':
                    self.assertEqual(connection.receives,1);self.assertNotIn('unload',calls);self.assertIsNone(meta['observed_gpu_indices'])
                else:self.assertEqual(connection.receives,2);self.assertIn('unload',calls)
                self.assertNotIn(secret,'\n'.join(logged.output));self.assertNotIn(str(Path(__file__).parent),'\n'.join(logged.output))
                diagnostic=json.loads(logged.records[0].getMessage().split('Worker failure: ',1)[1])
                self.assertTrue(diagnostic['trace'])
                self.assertTrue(all(set(frame)=={'file','function','line'} and '/' not in frame['file'] and '\\' not in frame['file'] for frame in diagnostic['trace']))

        # Exercise actual parent Worker.call; the existing bounded stop method is
        # represented by a spy, so this regression never creates/stops a process.
        for fail_stop in [False,True]:
            worker=Worker('w',WorkerSpec(models=['m']),Config(models={'m':ModelSpec(path='unused',task='chat')}))
            worker.process=NS(pid=123,is_alive=lambda:True);worker.cancel=NS(clear=lambda:None);worker.start=lambda:None
            meta={'current_model':'m','completed_requests':1,'observed_gpu_indices':None,'profiling_failure':{'type':'MemoryError','phase':'profile_exit'},'profiling_evidence':{'requested':True,'available':False,'status':'failed'}}
            sent=[];stops=[]
            worker.connection=NS(send=lambda value:sent.append(value),poll=lambda timeout:True,recv=lambda:('done',(payload,copy.deepcopy(meta))))
            def stop(force=False):
                stops.append(force)
                if fail_stop:
                    worker.stop_failed=True;worker.meta['stop_error']='retained stop failure'
                    raise EngineError('private cleanup value',503,'worker_still_running')
                worker.process=None;worker.connection=None;worker.meta['current_model']=None
            worker.stop=stop
            with patch('titanos.replicas._log_worker_exception'):
                self.assertIs(worker.call('infer','m','chat',{}),payload)
            self.assertEqual(stops,[True]);self.assertTrue(worker.unavailable);self.assertIsNone(worker.runtime_fault)
            self.assertEqual(worker.meta['profiling_retirement']['status'],'unconfirmed' if fail_stop else 'retired')
            del worker.start
            with self.assertRaises(EngineError) as error:worker.call('infer','m','chat',{})
            self.assertEqual(error.exception.code,'profiling_unavailable');self.assertEqual(len(sent),1)


if __name__=='__main__':unittest.main()
