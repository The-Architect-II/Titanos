"""Registry commit/recovery through actual HTTP/config code; no model providers."""
from pathlib import Path
import argparse, asyncio, hashlib, importlib.abc, json, logging, os, sys, tempfile, threading, types, unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'src'))
PROVIDER_ATTEMPTS=[]
class NoProviders(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers','diffusers','openvino','openvino_genai','onnxruntime','qwen_tts','mlx','faster_whisper','sentence_transformers'}:
            PROVIDER_ATTEMPTS.append(fullname);raise AssertionError('Provider import forbidden: '+fullname)
sys.meta_path.insert(0,NoProviders())
import httpx
from filelock import Timeout
from fastapi.responses import JSONResponse
from titanos import server, native, config as config_module
from titanos.config import Config, ModelSpec, read_config, registry_lock, update_config, write_config
from titanos.engine import Engine as ActualEngine, EngineError
from titanos.settings import Settings
ActualNative=native.NativePool
EVIDENCE=None

class RegistryTransaction(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=None
        if EVIDENCE:
            self.root=EVIDENCE/self._testMethodName;self.root.mkdir()
        else:
            self.tmp=tempfile.TemporaryDirectory(prefix='titanos-registry-');self.root=Path(self.tmp.name).resolve()
        self.directory=self.root/'checkpoint';self.directory.mkdir();self.path=self.root/'models.json'
        self.cfg=Config(models={'old':ModelSpec(path=str(self.directory),task='chat',device='cpu',defaults={'max_tokens':7})})
        write_config(self.path,self.cfg);self.before=self.path.read_bytes()
        self.fault=None;self.fail_cleanup=False;self.pools=[];self.engines=[];self.events=[];self.counts={'pool':0,'engine':0}
        self.primary=RuntimeError('inert primary lifecycle fault');self.extra=OSError('inert replacement cleanup fault')
        self.pause_close=False;self.close_started=threading.Event();self.close_release=threading.Event();self.tasks=[];self.threads=[]
        owner=self
        class Pool(ActualNative):
            def __init__(self,cfg,settings=None):
                owner.counts['pool']+=1;self.number=owner.counts['pool']
                if owner.fault=='pool_construct' and self.number==2:raise owner.primary
                super().__init__(cfg,settings=settings);self.close_calls=0;owner.pools.append(self)
            async def close(self):
                self.close_calls+=1;owner.events.append(('pool_close',self.number))
                await super().close()
                if owner.fault=='pool_close' and self.number==1:raise owner.primary
                if owner.fail_cleanup and self.number==2:raise owner.extra
        class Engine(ActualEngine):
            def __init__(self,cfg):
                owner.counts['engine']+=1;self.number=owner.counts['engine']
                if owner.fault=='engine_construct' and self.number==2:raise owner.primary
                super().__init__(cfg);self.close_calls=0;owner.engines.append(self)
            def close(self):
                self.close_calls+=1;owner.events.append(('engine_close',self.number))
                if self.number==1 and owner.pause_close:
                    owner.close_started.set()
                    if not owner.close_release.wait(5):raise TimeoutError('inert close release not received')
                if owner.fault=='engine_close' and self.number==1:raise owner.primary
                return super().close()
        class TinyRuntime:
            def __init__(self,spec):self.spec=spec;self.closed=False
            def close(self):self.closed=True
            def infer(self,task,data,emit=None,cancel=None):
                assert not self.closed and task=='chat'
                text=json.dumps({'path':self.spec.path,'defaults':self.spec.defaults,'received_max_tokens':data['max_tokens']})
                return {'text':text,'finish_reason':'stop','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
        backend=types.ModuleType('titanos.backends');backend.LoadedModel=TinyRuntime
        missing=object();previous_backend=sys.modules.get('titanos.backends',missing)
        sys.modules['titanos.backends']=backend
        def restore_backend():
            if previous_backend is missing:sys.modules.pop('titanos.backends',None)
            else:sys.modules['titanos.backends']=previous_backend
        self.addCleanup(restore_backend)
        self.patches=[patch.object(native,'NativePool',Pool),patch.object(server,'Engine',Engine)]
        for item in self.patches:item.start();self.addCleanup(item.stop)
        self.app=server.create_app(self.cfg,output_dir=self.root/'outputs',settings=Settings(),registry_path=self.path)
        self.gate=next(m.kwargs['gate'] for m in self.app.user_middleware if m.cls is server.Guard)
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app,raise_app_exceptions=False),base_url='http://localhost')
        self.payload={'name':'added','spec':{'path':str(self.directory),'task':'chat','device':'cpu'}}

    async def asyncTearDown(self):
        self.close_release.set()
        if self.tasks:await asyncio.gather(*self.tasks,return_exceptions=True)
        for thread in self.threads:await asyncio.to_thread(thread.join,5)
        await self.client.aclose()
        self.capture('final')
        for pool in self.pools:
            pool.workers=[];await ActualNative.close(pool)
        for engine in self.engines:ActualEngine.close(engine)
        self.assertEqual(PROVIDER_ATTEMPTS,[])
        if self.tmp:self.tmp.cleanup()

    def capture(self,label):
        if not EVIDENCE:return
        (self.root/(label+'-disk.json')).write_bytes(self.path.read_bytes())
        (self.root/'before.json').write_bytes(self.before)
        value={'gate':dict(self.gate),'disk_models':sorted(read_config(self.path).models),
            'slot_unavailable':self.app.state.engine.unavailable,'target_models':sorted(self.app.state.engine.target.config.models),
            'pool_close_calls':[p.close_calls for p in self.pools],'engine_close_calls':[e.close_calls for e in self.engines],
            'pending_cleanup':[(kind,getattr(obj,'number',None)) for kind,obj in self.app.state.registry_pending_cleanup]}
        (self.root/(label+'-state.json')).write_text(json.dumps(value,indent=2)+'\n')

    async def internal(self,method,path,**kwargs):
        async def app(scope,receive,send):await self.app({**scope,'titanos.internal':True},receive,send)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,raise_app_exceptions=False),base_url='http://localhost') as client:
            return await client.request(method,path,**kwargs)

    async def assert_available_models(self,names):
        response=await self.client.get('/v1/models');self.assertEqual(response.status_code,200)
        self.assertEqual(sorted(row['id'] for row in response.json()['data']),sorted(names))
        self.assertEqual(sorted(read_config(self.path).models),sorted(names))
        self.assertFalse(self.gate['reloading']);self.assertFalse(self.gate['unavailable']);self.assertEqual(self.gate['active'],0)
        self.assertFalse(self.app.state.native.closed)
        name='added' if 'added' in names else 'old';saved=read_config(self.path).models[name]
        loaded=await self.client.post('/api/models/load',json={'model':name});self.assertEqual(loaded.status_code,200)
        response=await self.client.post('/v1/chat/completions',json={'model':name,'messages':[{'role':'user','content':'inert fixture'}]})
        self.assertEqual(response.status_code,200)
        produced=json.loads(response.json()['choices'][0]['message']['content'])
        self.assertEqual(produced['path'],saved.path);self.assertEqual(produced['defaults'],saved.defaults)
        if 'max_tokens' in saved.defaults:self.assertEqual(produced['received_max_tokens'],saved.defaults['max_tokens'])
        self.assertEqual(self.app.state.engine.current,name)

    async def assert_unavailable(self):
        self.assertEqual((await self.client.get('/v1/models')).status_code,503)
        self.assertEqual((await self.internal('GET','/v1/models')).status_code,503)
        with self.assertRaises(EngineError):self.app.state.engine.catalog()
        self.assertTrue(self.gate['unavailable']);self.assertFalse(self.gate['reloading'])

    async def test_idle_registration_and_replacement_commit_matching_state(self):
        self.assertEqual((await self.client.post('/api/models/register',json=self.payload)).status_code,200)
        await self.assert_available_models(['old','added'])
        replacement={**self.payload,'replace':True,'spec':{**self.payload['spec'],'defaults':{'max_tokens':17}}}
        self.assertEqual((await self.client.post('/api/models/register',json=replacement)).status_code,200)
        self.assertEqual(read_config(self.path).models['added'].defaults,{'max_tokens':17})
        self.assertEqual(self.app.state.engine.target.config.models['added'].defaults,{'max_tokens':17})
        self.assertTrue(all(pool.closed for pool in self.pools[:-1]));await self.assert_available_models(['old','added'])

    async def test_pool_construction_failure_preserves_disk_and_live_engine(self):
        self.fault='pool_construct'
        self.assertEqual((await self.client.post('/api/models/register',json=self.payload)).status_code,500)
        self.assertEqual(self.path.read_bytes(),self.before);self.assertEqual(self.engines[0].close_calls,0)
        await self.assert_available_models(['old']);self.capture('failure')

    async def test_engine_construction_failure_disposes_unused_pool(self):
        self.fault='engine_construct'
        self.assertEqual((await self.client.post('/api/models/register',json=self.payload)).status_code,500)
        self.assertEqual(self.path.read_bytes(),self.before);self.assertTrue(self.pools[1].closed);self.assertEqual(self.engines[0].close_calls,0)
        await self.assert_available_models(['old']);self.capture('failure')

    async def test_pool_retirement_failure_blocks_then_explicit_reload_recovers(self):
        self.fault='pool_close'
        self.assertEqual((await self.client.post('/api/models/register',json=self.payload)).status_code,500)
        self.assertEqual(self.path.read_bytes(),self.before);await self.assert_unavailable();self.capture('failure')
        self.assertTrue(self.pools[1].closed);self.assertEqual(self.engines[1].close_calls,1)
        self.fault=None;self.assertEqual((await self.client.post('/api/models/reload')).status_code,200)
        await self.assert_available_models(['old'])

    async def test_engine_retirement_failure_blocks_then_explicit_reload_recovers(self):
        self.fault='engine_close'
        self.assertEqual((await self.client.post('/api/models/register',json=self.payload)).status_code,500)
        self.assertEqual(self.path.read_bytes(),self.before);self.assertTrue(self.pools[0].closed)
        await self.assert_unavailable();self.capture('failure')
        self.fault=None;self.assertEqual((await self.client.post('/api/models/reload')).status_code,200)
        await self.assert_available_models(['old'])

    async def test_persistence_failure_after_retirement_then_explicit_reload_recovers(self):
        with patch.object(config_module.os,'replace',side_effect=OSError('inert atomic persistence failure')):
            response=await self.client.post('/api/models/register',json=self.payload)
        self.assertEqual(response.status_code,500);self.assertEqual(self.path.read_bytes(),self.before)
        self.assertTrue(self.pools[0].closed);self.assertEqual(self.engines[0].close_calls,1)
        self.assertTrue(self.pools[1].closed);self.assertEqual(self.engines[1].close_calls,1)
        await self.assert_unavailable();self.capture('failure')
        self.assertEqual((await self.client.post('/api/models/reload')).status_code,200)
        await self.assert_available_models(['old']);self.assertIsNot(self.app.state.engine.target,self.engines[0])

    async def test_unused_cleanup_failure_preserves_primary_and_pending_reference(self):
        self.fault='engine_construct';self.fail_cleanup=True
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://localhost') as client:
            with self.assertRaises(RuntimeError) as caught:await client.post('/api/models/register',json=self.payload)
        self.assertIs(caught.exception,self.primary);self.assertIn('replacement cleanup',str(caught.exception.__notes__))
        self.assertEqual(self.path.read_bytes(),self.before);self.assertEqual(len(self.app.state.registry_pending_cleanup),1)
        await self.assert_unavailable();self.capture('failure')
        self.fault=None;self.fail_cleanup=False
        self.assertEqual((await self.client.post('/api/models/reload')).status_code,200)
        self.assertEqual(self.app.state.registry_pending_cleanup,[]);await self.assert_available_models(['old'])

    async def test_busy_external_request_refuses_internal_mutation(self):
        started=asyncio.Event();release=asyncio.Event()
        @self.app.get('/held-request')
        async def held():started.set();await release.wait();return {'done':True}
        task=asyncio.create_task(self.client.get('/held-request'));self.tasks.append(task)
        try:
            await asyncio.wait_for(started.wait(),2)
            response=await self.internal('POST','/api/models/register',json=self.payload)
            self.assertEqual(response.status_code,409);self.assertEqual(response.json()['error']['code'],'registry_busy')
            self.assertEqual(self.path.read_bytes(),self.before)
        finally:release.set();await task
        self.assertEqual((await self.internal('POST','/api/models/register',json=self.payload)).status_code,200)
        await self.assert_available_models(['old','added'])

    @unittest.skipUnless(os.name=='nt','Real inert Windows worker/Job lifecycle')
    async def test_inflight_owned_worker_and_cleanup_failure_preserve_registration(self):
        import ctypes,subprocess,time
        from ctypes import wintypes
        from titanos import _windows_job as windows
        folder=self.root/'owned-worker';folder.mkdir()
        release=folder/'RELEASE';done=folder/'done.json';active=folder/'active.json';tree=folder/'tree.json'
        code=r'''
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1])
grandchild=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(300)'],
    stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW,close_fds=True)
(root/'tree.json').write_text(json.dumps({'worker':os.getpid(),'grandchild':grandchild.pid}))
for line in sys.stdin.buffer:
    request=json.loads(line)
    (root/'active.json').write_text(json.dumps({'pid':os.getpid(),'request':request}))
    while not (root/'RELEASE').exists():time.sleep(.02)
    temporary=root/'done.tmp'
    temporary.write_text(json.dumps(request))
    temporary.replace(root/'done.json')
'''
        python=str(Path(getattr(sys,'_base_executable',sys.executable)).resolve())
        log=(folder/'worker.log').open('wb')
        job=windows.OwnedJob([python,'-I','-S','-B','-c',code,str(folder)],
                            stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,cwd=folder)
        owner=self
        class InertWorker:
            device='cpu'
            def __init__(self,spec):self.spec=spec;self.closed=False;self.close_error=None;self.close_calls=0
            def infer(self,task,data,emit=None,cancel=None):
                request={'path':self.spec.path,'defaults':self.spec.defaults,'received_max_tokens':data['max_tokens']}
                job.process.stdin.write(json.dumps(request).encode()+b'\n');job.process.stdin.flush()
                deadline=time.monotonic()+10
                while not done.exists():
                    if job.process.poll() is not None:raise RuntimeError('Inert worker exited during request')
                    if time.monotonic()>=deadline:raise TimeoutError('Inert request release timed out')
                    time.sleep(.02)
                return {'text':done.read_text(),'finish_reason':'stop','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
            def close(self):
                self.close_calls+=1
                if job.handle is not None:
                    try:job.__exit__(None,None,None)
                    except BaseException as error:self.close_error=error;raise
                self.closed=True
        previous=self.app.state.engine.target
        worker=InertWorker(self.cfg.models['old'])
        previous.runtime=worker;previous.current='old';previous.states['old']='loaded'
        prior_spec=worker.spec.model_dump(mode='json')
        marker=self.directory/'PRESERVE.txt';marker.write_bytes(b'inert external checkpoint marker')
        replacement_dir=self.root/'replacement';replacement_dir.mkdir()
        replacement={'name':'old','replace':True,'spec':{'path':str(replacement_dir),'task':'chat','device':'cpu','defaults':{'max_tokens':19}}}
        observations={'scope':'Actual source HTTP/registry/Engine with inert IPC worker and real Windows Job; no trained model or frozen server',
                      'model_calls':0,'inert_requests':1,'fault':'Test-injected CloseHandle false/ERROR_ACCESS_DENIED; not an OS/security refusal',
                      'prior_spec':prior_spec,'replacement_requested':replacement,'supervisor_job_handle':job.handle,'processes':{}}
        query=ctypes.WinDLL('kernel32',use_last_error=True)
        for name,args,result in (
            ('OpenProcess',[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD],wintypes.HANDLE),
            ('WaitForSingleObject',[wintypes.HANDLE,wintypes.DWORD],wintypes.DWORD),
            ('GetProcessTimes',[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)],wintypes.BOOL),
            ('CloseHandle',[wintypes.HANDLE],wintypes.BOOL)):
            function=getattr(query,name);function.argtypes,function.restype=args,result
        handles={};request_task=None
        try:
            deadline=time.monotonic()+10
            while not tree.exists():
                self.assertIsNone(job.process.poll());self.assertLess(time.monotonic(),deadline)
                await asyncio.sleep(.02)
            for role,pid in json.loads(tree.read_text()).items():
                handle=query.OpenProcess(0x100000|0x1000,False,pid)
                if not handle:raise ctypes.WinError(ctypes.get_last_error())
                handles[role]=handle
                times=[wintypes.FILETIME() for _ in range(4)]
                if not query.GetProcessTimes(handle,*[ctypes.byref(value) for value in times]):raise ctypes.WinError(ctypes.get_last_error())
                observations['processes'][role]={'pid':pid,'created_filetime':(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime}
            self.assertEqual(observations['processes']['worker']['pid'],job.process.pid)
            request_task=asyncio.create_task(self.client.post('/v1/chat/completions',json={
                'model':'old','messages':[{'role':'user','content':'inert metadata request'}]}))
            self.tasks.append(request_task)
            while not active.exists():
                self.assertLess(time.monotonic(),deadline);self.assertFalse(request_task.done())
                await asyncio.sleep(.02)
            self.assertEqual(json.loads(active.read_text())['pid'],job.process.pid)
            self.assertEqual(self.gate['active'],1);self.capture('inflight')
            rejected=await self.client.post('/api/models/register',json=replacement)
            self.assertEqual(rejected.status_code,409);self.assertEqual(rejected.json()['error']['code'],'registry_busy')
            self.assertFalse(request_task.done());self.assertEqual(worker.close_calls,0)
            self.assertEqual(self.path.read_bytes(),self.before);self.assertIs(self.app.state.engine.target,previous)
            self.assertEqual(worker.spec.model_dump(mode='json'),prior_spec)
            self.assertTrue(all(query.WaitForSingleObject(handle,0)==258 for handle in handles.values()))
            observations['busy_rejection']={'response':rejected.json(),'registry_unchanged':True,'runtime_preserved':True,'child_tree_alive':True}
            release.write_text('finish inert request',encoding='utf-8')
            response=await asyncio.wait_for(request_task,10);self.assertEqual(response.status_code,200)
            returned=json.loads(response.json()['choices'][0]['message']['content'])
            self.assertEqual(returned,{'path':prior_spec['path'],'defaults':prior_spec['defaults'],'received_max_tokens':7})
            self.assertEqual(self.gate['active'],0)
            actual_close=windows.kernel.CloseHandle;failed_handle=job.handle
            def injected_close(handle):
                if handle==failed_handle:ctypes.set_last_error(5);return False
                return actual_close(handle)
            with patch.object(windows.kernel,'CloseHandle',side_effect=injected_close):
                failed=await self.client.post('/api/models/register',json=replacement)
            self.assertEqual(failed.status_code,500);self.assertIs(worker.close_error.owned_job,job)
            self.assertEqual(job.cleanup_errors[-1]['step'],'close_job');self.assertIsNotNone(job.handle)
            self.assertEqual(self.path.read_bytes(),self.before);self.assertIs(self.app.state.engine.target,previous)
            self.assertIs(previous.runtime,worker);self.assertEqual(previous.current,'old')
            self.assertEqual(previous.config.models['old'].model_dump(mode='json'),prior_spec)
            self.assertEqual(self.cfg.models['old'].model_dump(mode='json'),prior_spec)
            self.assertEqual(worker.spec.model_dump(mode='json'),prior_spec)
            await self.assert_unavailable();self.capture('cleanup-failed')
            self.assertTrue(all(query.WaitForSingleObject(handle,5000)==0 for handle in handles.values()))
            observations['cleanup_failure']={'http_status':failed.status_code,'registry_unchanged':True,'old_runtime_reference_preserved':True,
                'registry_unavailable':True,'job_handle_retained':True,'child_tree_gone':True,'error':repr(worker.close_error),'cleanup_errors':job.cleanup_errors}
            recovered=await self.client.post('/api/models/reload');self.assertEqual(recovered.status_code,200)
            self.assertIsNone(job.handle);self.assertTrue(worker.closed);self.assertIsNot(self.app.state.engine.target,previous)
            self.assertEqual(self.path.read_bytes(),self.before);await self.assert_available_models(['old'])
            self.assertEqual(self.app.state.engine.target.config.models['old'].model_dump(mode='json'),prior_spec)
            self.assertEqual(marker.read_bytes(),b'inert external checkpoint marker')
            observations['recovery']={'response':recovered.json(),'registry_unchanged':True,'prior_registration_recovered':True,'external_marker_preserved':True}
        finally:
            release.write_text('owned fixture cleanup',encoding='utf-8')
            if request_task is not None and not request_task.done():await asyncio.wait_for(request_task,12)
            if job.handle is not None:job.__exit__(None,None,None)
            observations['final_child_tree_gone']=all(query.WaitForSingleObject(handle,5000)==0 for handle in handles.values())
            observations['final_job_handle_closed']=job.handle is None
            observations['final_observer_handles_closed']=all(bool(query.CloseHandle(handle)) for handle in handles.values())
            job.process.stdin.close();job.process._handle.Close();log.close()
            observations['final_popen_handle_closed']=True
            if EVIDENCE:(self.root/'owned-worker-result.json').write_text(json.dumps(observations,indent=2)+'\n')
            self.assertTrue(observations['final_child_tree_gone'] and observations['final_job_handle_closed'] and observations['final_observer_handles_closed'])

    async def test_outer_mcp_transport_does_not_double_count_internal_mutation(self):
        async def dispatch(scope,receive,send):
            response=await self.internal('POST','/api/models/register',json=self.payload)
            await JSONResponse(response.json(),status_code=response.status_code)(scope,receive,send)
        outer=server.Guard(dispatch,settings=Settings(),gate=self.gate)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=outer),base_url='http://localhost') as client:
            self.assertEqual((await client.post('/mcp',json={})).status_code,200)
        await self.assert_available_models(['old','added'])

    async def test_concurrent_writer_lock_refusal_then_reads_latest(self):
        with registry_lock(self.path):
            response=await self.client.post('/api/models/register',json=self.payload)
            self.assertEqual(response.status_code,409);self.assertEqual(self.path.read_bytes(),self.before)
        update_config(self.path,lambda cfg:cfg.models.__setitem__('writer',ModelSpec(path=str(self.directory),task='chat',device='cpu')))
        self.assertEqual((await self.client.post('/api/models/register',json=self.payload)).status_code,200)
        await self.assert_available_models(['old','writer','added'])

    async def test_concurrent_writer_survives_failed_retirement_without_stale_rollback(self):
        self.pause_close=True;self.fault='engine_close'
        task=asyncio.create_task(self.client.post('/api/models/register',json=self.payload));self.tasks.append(task)
        self.assertTrue(await asyncio.to_thread(self.close_started.wait,2))
        started=threading.Event();done=threading.Event();errors=[]
        def writer():
            started.set()
            try:update_config(self.path,lambda cfg:cfg.models.__setitem__('writer',ModelSpec(path=str(self.directory),task='chat',device='cpu')))
            except BaseException as exc:errors.append(repr(exc))
            finally:done.set()
        thread=threading.Thread(target=writer);self.threads.append(thread);thread.start()
        self.assertTrue(await asyncio.to_thread(started.wait,2));await asyncio.sleep(.05);self.assertFalse(done.is_set())
        self.close_release.set();self.assertEqual((await task).status_code,500)
        self.assertTrue(await asyncio.to_thread(done.wait,2));self.assertEqual(errors,[])
        self.assertEqual(sorted(read_config(self.path).models),['old','writer']);await self.assert_unavailable();self.capture('failure')
        self.fault=None;self.pause_close=False
        self.assertEqual((await self.client.post('/api/models/reload')).status_code,200)
        await self.assert_available_models(['old','writer'])

    async def test_cancellation_drains_close_thread_before_unlock_or_admission(self):
        self.pause_close=True
        task=asyncio.create_task(self.client.post('/api/models/register',json=self.payload));self.tasks.append(task)
        self.assertTrue(await asyncio.to_thread(self.close_started.wait,2));task.cancel('registry cancellation fixture');await asyncio.sleep(.05)
        self.assertFalse(task.done());self.assertTrue(self.gate['reloading']);self.assertEqual(self.path.read_bytes(),self.before)
        with self.assertRaises(Timeout):
            with registry_lock(self.path,timeout=0):pass
        self.close_release.set()
        with self.assertRaises(asyncio.CancelledError) as caught:await task
        self.assertEqual(caught.exception.args,('registry cancellation fixture',))
        self.assertEqual(self.path.read_bytes(),self.before);await self.assert_unavailable();self.capture('cancelled')
        self.pause_close=False;self.assertEqual((await self.client.post('/api/models/reload')).status_code,200)
        await self.assert_available_models(['old'])

    async def test_native_close_failure_waits_for_sibling_and_preserves_both_errors(self):
        started=asyncio.Event();release=asyncio.Event();finished=asyncio.Event();first=RuntimeError('first native close');second=OSError('second native close')
        class Failed:
            async def close(self):raise first
        class Delayed:
            async def close(self):started.set();await release.wait();finished.set();raise second
        self.pools[0].workers=[Failed(),Delayed()]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://localhost') as client:
            task=asyncio.create_task(client.post('/api/models/register',json=self.payload));self.tasks.append(task)
            try:
                await asyncio.wait_for(started.wait(),2);await asyncio.sleep(.02)
                self.assertFalse(task.done());self.assertTrue(self.gate['reloading']);self.assertEqual(self.path.read_bytes(),self.before)
                with self.assertRaises(Timeout):
                    with registry_lock(self.path,timeout=0):pass
            finally:release.set()
            with self.assertRaises(RuntimeError) as caught:await task
        self.assertIs(caught.exception,first);self.assertTrue(finished.is_set());self.assertIn('second native close',str(first.__notes__))
        self.assertEqual(self.path.read_bytes(),self.before);await self.assert_unavailable();self.capture('failure')

    async def test_cancelled_native_close_preserves_primary_and_sibling_notes(self):
        started=asyncio.Event();release=asyncio.Event();first=RuntimeError('first native failure');second=OSError('second native failure')
        class Failed:
            async def close(self):raise first
        class Delayed:
            async def close(self):started.set();await release.wait();raise second
        self.pools[0].workers=[Failed(),Delayed()]
        task=asyncio.create_task(self.client.post('/api/models/register',json=self.payload));self.tasks.append(task)
        try:
            await asyncio.wait_for(started.wait(),2);task.cancel('native cancellation fixture');await asyncio.sleep(.02)
            self.assertFalse(task.done());self.assertTrue(self.gate['reloading'])
        finally:release.set()
        with self.assertRaises(asyncio.CancelledError) as caught:await task
        self.assertEqual(caught.exception.args,('native cancellation fixture',))
        notes='\n'.join(caught.exception.__notes__)
        self.assertIn('first native failure',notes);self.assertIn('second native failure',notes)
        self.assertEqual(self.path.read_bytes(),self.before);await self.assert_unavailable();self.capture('cancelled')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path)
    parser.add_argument('--test',action='append',choices=[name for name in dir(RegistryTransaction) if name.startswith('test_')])
    args=parser.parse_args()
    if args.evidence:EVIDENCE=args.evidence.resolve();EVIDENCE.mkdir(exist_ok=False)
    logging.disable(logging.CRITICAL)
    suite=(unittest.TestSuite(RegistryTransaction(name) for name in args.test) if args.test else unittest.defaultTestLoader.loadTestsFromTestCase(RegistryTransaction))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'source_hashes':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['src/titanos/server.py','src/titanos/config.py','src/titanos/native.py','tests/check_registry_transaction.py']},
        'provider_import_attempts':PROVIDER_ATTEMPTS,'scope':'Actual in-process HTTP/config/Engine/NativePool with a tiny inert LoadedModel and controlled lifecycle/file/close-thread faults; no trained models, provider imports or remote endpoints'}
    if EVIDENCE:(EVIDENCE/'RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2));raise SystemExit(0 if result.wasSuccessful() else 1)
