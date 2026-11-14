"""Exercise the actual smoke function with inert transports/process/Win32 effects.

AST extraction avoids importing build-only packaging dependencies in core checks.
No compiled executable, application listener, model or native handle is started.
The process/listener identity table (psutil) is an inert double: readiness must
prove the retained child owns the exact loopback listener around every health
response, so a foreign, exited, replaced or unbound child is never accepted.
"""
from pathlib import Path
import argparse, ast, asyncio, ctypes, hashlib, json, os, re, sys, tempfile, types, unittest
from unittest.mock import patch
NS=types.SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=None
SOURCE=ROOT/'tools/build_binary.py'
TREE=ast.parse(SOURCE.read_text(encoding='utf-8'))
SMOKE=next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name=='smoke')
# The declaration gate's source of truth is the builder's MCP_TOOL_SOURCES tuple and the two helpers over it (the same objects
# smoke() calls), not a file list repeated here; tests/check_offline_binary.py asserts that set against the real registration.
DECLARATIONS=[node for node in TREE.body if (isinstance(node,ast.FunctionDef) and node.name in {'declared_mcp_tools','mcp_tool_source_paths','expected_mcp_tools'})
              or (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MCP_TOOL_SOURCES' for t in node.targets))]
assert len(DECLARATIONS)==4,'tools/build_binary.py must define MCP_TOOL_SOURCES, mcp_tool_source_paths, expected_mcp_tools and declared_mcp_tools'
declarations_namespace={'ast':ast,'Path':Path,'ROOT':ROOT}
exec(compile(ast.Module(body=DECLARATIONS,type_ignores=[]),str(SOURCE),'exec'),declarations_namespace)
declared_mcp_tools=declarations_namespace['declared_mcp_tools']
expected_mcp_tools=declarations_namespace['expected_mcp_tools']
TOOL_NAMES=expected_mcp_tools()
assert 'src/titanos/broker/mcp_tools.py' in declarations_namespace['MCP_TOOL_SOURCES'] and {'broker_status','whoami','chat','list_models'}<=TOOL_NAMES

class Function:
    def __init__(self,call):self.call=call
    def __call__(self,*args):return self.call(*args)

class Checker(unittest.TestCase):
    def run_smoke(self,windows=False,fault=None):
        own=tempfile.TemporaryDirectory(prefix='registry-smoke-check-',dir=EVIDENCE)
        base=Path(own.name);output=base/'output';output.mkdir()
        state=types.SimpleNamespace(now=0.0,live={},blocked=False,writer=False,unavailable=False,process=None,
                                    registry=None,log=None,close_calls=[],create_calls=[],cli_add=0,tool_calls=[],
                                    listener_pid=9999 if fault=='foreign_listener' else 4321,listener_ip='0.0.0.0' if fault=='foreign_address' else '127.0.0.1',
                                    bound=fault!='never_binds',create_time=1234.5,health_requests=0,sequence=[],system_reads=0)
        class Response:
            def __init__(self,status,body):self.status_code=status;self.body=body;self.text=json.dumps(body)
            def json(self):return self.body
        def save_disk(value):state.registry.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        def disk():return json.loads(state.registry.read_bytes())
        def expected_trace():
            text='Traceback (most recent call last):\n  File "server.py", line 1, in register_model\n'
            for filename,function,statement in [('server.py','reload_registry','ifchange:write_config(path,candidate)'),('config.py','write_config','os.replace(tmp,path)')]:
                lines=(ROOT/'src/titanos'/filename).read_text().splitlines()
                number=next(n for n,line in enumerate(lines,1) if ''.join(line.split())==statement)
                if fault in {'wrong_trace','close_and_trace'} and filename=='config.py':number+=1
                text+=f'  File "{filename}", line {number}, in {function}\n'
            return text+'PermissionError: controlled replacement failure\n'
        def request(method,url,json=None,**kwargs):
            endpoint=url.split('://',1)[-1].partition('/')[2];endpoint='/'+endpoint
            if endpoint=='/healthz':
                state.health_requests+=1;state.sequence.append('health')
                if fault=='replaced_after_health':state.listener_pid=7777
                if fault=='exits_after_health':state.process.stopped=True
                if fault=='pid_reused_after_health':state.create_time=999.0
                return Response(200,{'status':'ok'})  # A 200 alone is never readiness; ownership is proved around it.
            if endpoint=='/api/models/register':
                if state.writer:return Response(409,{'error':{'code':'registry_busy'}})
                if state.blocked:
                    state.unavailable=True;state.log.write(expected_trace());state.log.flush()
                    return Response(500,{'error':{'code':'server_error'}})
                value=disk();value['models'][json['name']]=json['spec'];save_disk(value);state.live=dict(value['models'])
                if fault=='timeout':state.now=121
                return Response(200,{'status':'reloaded'})
            if endpoint=='/v1/models':
                if state.unavailable:return Response(503,{'error':{'code':'registry_unavailable'}})
                return Response(200,{'object':'list','data':[{'id':n,'task':s['task']} for n,s in state.live.items()]})
            if endpoint=='/api/models/reload':
                if state.blocked:raise AssertionError('Recovery before exact handle close')
                state.unavailable=False;state.live=dict(disk()['models']);return Response(200,{'status':'reloaded'})
            if endpoint=='/api/models/loaded':return Response(200,{'object':'list','data':[]})
            if endpoint=='/api/models/capabilities':return Response(404,{'error':{'code':'model_not_found'}})
            if method=='DELETE' and endpoint.startswith('/api/models/registrations/'):
                value=disk();del value['models'][endpoint.rsplit('/',1)[1]];save_disk(value);state.live=dict(value['models']);return Response(200,{'status':'reloaded'})
            raise AssertionError('Unexpected endpoint (no loading/inference allowed): '+endpoint)
        def run(argv,**kwargs):
            if 'init' in argv:
                state.registry=Path(kwargs['env']['TITANOS_HOME'])/'models.json';save_disk({'version':1,'models':{},'workers':{}})
            if 'add' in argv:
                state.cli_add+=1;index=argv.index('add');name,path=argv[index+1:index+3];value=disk()
                value['models'][name]={'path':path,'task':'chat','backend':'transformers','device':'cpu','dtype':'float32','defaults':json_load(Path(argv[argv.index('--defaults')+1]))}
                save_disk(value)
            return types.SimpleNamespace(returncode=0,stdout='fixture',stderr='')
        class Process:
            pid=4321
            def __init__(self,argv,**kwargs):
                self.stopped=fault=='exited_child';state.process=self;state.log=kwargs['stdout'];self.argv=argv
                if fault=='exited_child':state.log.write('inert startup failure: bind refused by fixture\n');state.log.flush()
            def poll(self):return (3 if fault=='exited_child' else 0) if self.stopped else None
            def terminate(self):self.stopped=True
            def wait(self,timeout):return 0
        class Socket:
            def __enter__(self):return self
            def __exit__(self,*unused):pass
            def bind(self,address):assert address==('127.0.0.1',0)
            def getsockname(self):return ('127.0.0.1',43210)
        class Lock:
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):state.writer=True;return self
            def __exit__(self,*unused):state.writer=False
        class Client:
            protocol_version='inert-protocol'
            def __init__(self,*args,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*unused):pass
            async def list_tools(self):
                names=sorted(TOOL_NAMES)
                if fault=='discovery':names.pop()
                if fault=='exits_during_discovery':state.process.stopped=True
                return types.SimpleNamespace(tools=[types.SimpleNamespace(name=n) for n in names])
            async def call_tool(self,name,args):
                state.tool_calls.append(name)
                if name=='list_models':return types.SimpleNamespace(is_error=False,structured_content={'object':'list','data':[]})
                assert name=='reload_registry';state.live=dict(disk()['models'])
                return types.SimpleNamespace(is_error=False,structured_content={'status':'reloaded','models':sorted(state.live),'loaded':[]})
        def listener_rows():
            rows=[NS(pid=1,laddr=NS(ip='127.0.0.1',port=40001),status='LISTEN'),  # unrelated listener on another port
                  NS(pid=4321,laddr=NS(ip='127.0.0.1',port=43210),status='ESTABLISHED')]  # an accepted connection on the smoke port is not a listener
            if fault=='unrelated_ipv6_listener':rows.append(NS(pid=8888,laddr=NS(ip='::1',port=43210),status='LISTEN'))  # cannot receive a 127.0.0.1 connection
            if fault=='shared_port_foreign_listener':rows.append(NS(pid=8888,laddr=NS(ip='0.0.0.0',port=43210),status='LISTEN'))  # wildcard beside the child
            if state.bound:
                pid=state.listener_pid
                # Linux inode->PID mapping can lag a fresh bind: the row exists but its owner is unknown for a moment (or forever for another user).
                if fault=='persistent_unmapped_pid' or (fault=='transient_unmapped_pid' and state.system_reads<=2):pid=None
                rows.append(NS(pid=pid,laddr=NS(ip=state.listener_ip,port=43210),status='LISTEN'))
            return rows
        class PsutilError(Exception):pass
        class AccessDenied(PsutilError):pass
        class NoSuchProcess(PsutilError):pass
        class ChildProcess:
            def __init__(self,pid):
                if pid!=4321:raise NoSuchProcess(pid)
                self.pid=pid;self._create_time=state.create_time  # psutil reads create_time once per handle and caches it
            def create_time(self):return self._create_time
            def is_running(self):return not state.process.stopped and state.create_time==self._create_time  # psutil's own PID-reuse test
            def net_connections(self,kind='inet'):
                assert kind=='tcp';state.sequence.append('child-table')
                return [NS(laddr=row.laddr,status=row.status) for row in listener_rows() if row.pid==4321]
        def net_connections(kind='inet'):
            assert kind=='tcp'
            if fault=='system_table_denied':raise AccessDenied('inert non-root table denial')
            state.system_reads+=1;state.sequence.append('system-table');return listener_rows()
        psutil=types.ModuleType('psutil');psutil.Error=PsutilError;psutil.AccessDenied=AccessDenied;psutil.NoSuchProcess=NoSuchProcess
        psutil.CONN_LISTEN='LISTEN';psutil.Process=ChildProcess;psutil.net_connections=net_connections
        def create_file(*args):
            state.create_calls.append(args);self.assertEqual(args[1:],(0x80000000,1,None,3,0x80,None));state.blocked=True;return 81
        def information(handle,pointer):self.assertEqual(handle,81);pointer._obj.value=0;return 1
        def close_handle(handle):
            state.close_calls.append(handle)
            if fault=='close_and_trace':return 0
            state.blocked=False;return 1
        kernel=types.SimpleNamespace(CreateFileW=Function(create_file),GetHandleInformation=Function(information),CloseHandle=Function(close_handle))
        modules={}
        for name in ['httpx','mcp','mcp.client','mcp.client.stdio','filelock']:modules[name]=types.ModuleType(name)
        modules['httpx'].request=request;modules['httpx'].get=lambda url,**kw:request('GET',url,**kw);modules['httpx'].HTTPError=RuntimeError
        modules['mcp'].Client=Client;modules['mcp'].__path__=[];modules['mcp.client'].__path__=[]
        modules['mcp.client.stdio'].StdioServerParameters=lambda **kwargs:kwargs;modules['filelock'].FileLock=Lock;modules['psutil']=psutil
        missing=object();previous={name:sys.modules.get(name,missing) for name in modules};sys.modules.update(modules)
        namespace={'Path':Path,'os':types.SimpleNamespace(name='nt' if windows else 'posix',environ={},SEEK_END=os.SEEK_END),
            'tempfile':types.SimpleNamespace(TemporaryDirectory=lambda **kw:tempfile.TemporaryDirectory(dir=base,**kw)),
            'subprocess':types.SimpleNamespace(run=run,Popen=Process),'socket':types.SimpleNamespace(socket=Socket),
            'asyncio':asyncio,'time':types.SimpleNamespace(monotonic=lambda:state.now,sleep=lambda delay:setattr(state,'now',state.now+delay)),
            'hashlib':hashlib,'json':json,'re':re,'ROOT':ROOT,'declared_mcp_tools':declared_mcp_tools,'expected_mcp_tools':expected_mcp_tools,
            'digest':lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()}
        exec(compile(ast.Module(body=[SMOKE],type_ignores=[]),str(SOURCE),'exec'),namespace)
        caught=None
        try:
            with patch.object(ctypes,'WinDLL',lambda *a,**k:kernel,create=True),patch.object(ctypes,'get_last_error',lambda:6,create=True),patch.object(ctypes,'WinError',lambda n:OSError('inert close error '+str(n)),create=True):
                try:namespace['smoke'](Path('inert-titanos.exe'),output)
                except BaseException as exc:caught=exc
            report=json_load(output/'registry-smoke.json')
            self.assertTrue(state.process.stopped);self.assertEqual(report['model_calls'],0)
            if EVIDENCE:
                target=EVIDENCE/(self._testMethodName+'.json');target.write_text(json.dumps(report,indent=2)+'\n')
            return report,state,caught
        finally:
            for name,old_value in previous.items():
                if old_value is missing:sys.modules.pop(name,None)
                else:sys.modules[name]=old_value
            own.cleanup()

    def test_complete_non_windows_sequence(self):
        report,state,error=self.run_smoke();self.assertIsNone(error);self.assertEqual(report['status'],'PASSED');self.assertEqual(report['request_count'],14)
        self.assertEqual(state.cli_add,1);self.assertEqual(state.tool_calls[:2],['list_models','list_models']);self.assertTrue(report['placeholder_files_preserved'])
    def test_complete_windows_sequence_and_exact_handle(self):
        report,state,error=self.run_smoke(windows=True);self.assertIsNone(error);self.assertEqual(report['status'],'PASSED');self.assertEqual(report['request_count'],19)
        self.assertEqual(state.close_calls,[81]);self.assertTrue(report['windows_persistence']['handle_closed']);self.assertTrue(all(r['matched'] for r in report['windows_persistence']['trace_source_join']))
    def test_wrong_source_frame_is_not_a_persistence_pass(self):
        report,state,error=self.run_smoke(windows=True,fault='wrong_trace');self.assertIsNotNone(error);self.assertEqual(report['status'],'FAILED');self.assertEqual(state.close_calls,[81])
        self.assertIn('traceback',str(error));self.assertLess(report['request_count'],19)
    def test_handle_failure_does_not_replace_primary_trace_error(self):
        report,state,error=self.run_smoke(windows=True,fault='close_and_trace');self.assertIsInstance(error,RuntimeError);self.assertIn('traceback',str(error))
        self.assertEqual(state.close_calls,[81]);self.assertFalse(report['windows_persistence']['handle_closed']);self.assertTrue(error.__notes__);self.assertEqual(report['status'],'FAILED')
    def test_expired_budget_retains_partial_without_retry(self):
        report,state,error=self.run_smoke(fault='timeout');self.assertIsInstance(error,TimeoutError);self.assertEqual(report['status'],'FAILED');self.assertEqual(report['request_count'],1)
    def test_empty_full_tool_discovery_precedes_registry_changes(self):
        report,state,error=self.run_smoke(fault='discovery');self.assertIsNotNone(error);self.assertEqual(report['status'],'FAILED');self.assertEqual(report['request_count'],0);self.assertEqual(state.cli_add,0)

    def test_ownership_is_proved_before_health_and_at_every_readiness_stage(self):
        report,state,error=self.run_smoke();self.assertIsNone(error)
        self.assertEqual([row['stage'] for row in report['listener_ownership']],['startup','before-health','after-health','before-controls'])
        for row in report['listener_ownership']:
            self.assertTrue(row['owned']);self.assertEqual(row['table'],'system');self.assertEqual(row['listeners'],[{'pid':4321,'ip':'127.0.0.1','port':43210}])
        self.assertEqual(state.health_requests,1);self.assertEqual(state.sequence[:4],['system-table','system-table','health','system-table'])
        self.assertEqual(report['server_identity'],{'pid':4321,'create_time':1234.5});self.assertEqual(report['readiness']['health_status'],200)
    def test_foreign_listener_is_refused_before_any_health_request(self):
        report,state,error=self.run_smoke(fault='foreign_listener');self.assertIsInstance(error,RuntimeError);self.assertIn('not owned',str(error))
        self.assertEqual(state.health_requests,0);self.assertEqual(state.tool_calls,[]);self.assertEqual(report['status'],'FAILED');self.assertEqual(report['request_count'],0)
        self.assertEqual(report['listener_ownership'][-1]['listeners'],[{'pid':9999,'ip':'127.0.0.1','port':43210}]);self.assertFalse(report['listener_ownership'][-1]['owned'])
    def test_listener_on_another_address_is_not_the_requested_loopback_owner(self):
        report,state,error=self.run_smoke(fault='foreign_address');self.assertIsInstance(error,RuntimeError);self.assertIn('not owned',str(error))
        self.assertEqual(state.health_requests,0);self.assertEqual(report['listener_ownership'][-1]['listeners'][0]['ip'],'0.0.0.0')
    def test_exited_child_is_refused_although_health_would_answer_200(self):
        report,state,error=self.run_smoke(fault='exited_child');self.assertIsInstance(error,RuntimeError);self.assertIn('exited before startup',str(error));self.assertIn('bind refused by fixture',str(error))
        self.assertEqual(state.health_requests,0);self.assertEqual(report['listener_ownership'],[{'stage':'startup','pid':4321,'port':43210,'child_exit_code':3}]);self.assertEqual(report['status'],'FAILED')
    def test_child_exit_during_discovery_is_refused_before_controls(self):
        report,state,error=self.run_smoke(fault='exits_during_discovery');self.assertIsInstance(error,RuntimeError);self.assertIn('exited before before-controls',str(error))
        self.assertEqual(state.tool_calls,['list_models','list_models']);self.assertEqual(report['request_count'],0);self.assertEqual(state.cli_add,0)
        self.assertEqual([row['stage'] for row in report['listener_ownership']],['startup','before-health','after-health','before-controls']);self.assertEqual(report['listener_ownership'][-1]['child_exit_code'],0)
    def test_foreign_wildcard_listener_beside_the_child_is_refused(self):
        report,state,error=self.run_smoke(fault='shared_port_foreign_listener');self.assertIsInstance(error,RuntimeError);self.assertIn('not owned',str(error))
        self.assertEqual(state.health_requests,0);last=report['listener_ownership'][-1];self.assertFalse(last['owned'])
        self.assertEqual(sorted(row['pid'] for row in last['listeners']),[4321,8888]);self.assertEqual(last['unrelated_listeners'],[])
    def test_unrelated_ipv6_listener_on_the_same_port_number_is_recorded_not_fatal(self):
        report,state,error=self.run_smoke(fault='unrelated_ipv6_listener');self.assertIsNone(error);self.assertEqual(report['status'],'PASSED')
        first=report['listener_ownership'][0];self.assertEqual(first['listeners'],[{'pid':4321,'ip':'127.0.0.1','port':43210}]);self.assertEqual(first['unrelated_listeners'],[{'pid':8888,'ip':'::1','port':43210}])
    def test_transiently_unmapped_owner_is_reread_then_owned(self):
        report,state,error=self.run_smoke(fault='transient_unmapped_pid');self.assertIsNone(error);self.assertEqual(report['status'],'PASSED')
        self.assertEqual([row['table_reads'] for row in report['listener_ownership']],[3,1,1,1]);self.assertTrue(all(row['owned'] for row in report['listener_ownership']))
    def test_persistently_unmapped_owner_is_refused_after_bounded_rereads(self):
        report,state,error=self.run_smoke(fault='persistent_unmapped_pid');self.assertIsInstance(error,RuntimeError);self.assertIn('not owned',str(error))
        self.assertEqual(state.health_requests,0);last=report['listener_ownership'][-1];self.assertEqual(last['table_reads'],5);self.assertIsNone(last['listeners'][0]['pid']);self.assertFalse(last['owned'])
    def test_listener_replaced_after_health_is_refused_before_discovery(self):
        report,state,error=self.run_smoke(fault='replaced_after_health');self.assertIsInstance(error,RuntimeError);self.assertIn('after-health',str(error))
        self.assertEqual(state.health_requests,1);self.assertEqual(state.tool_calls,[]);self.assertEqual(report['request_count'],0)
        self.assertEqual([row['stage'] for row in report['listener_ownership']],['startup','before-health','after-health']);self.assertEqual(report['listener_ownership'][-1]['listeners'][0]['pid'],7777)
    def test_child_exit_after_health_is_refused_before_discovery(self):
        report,state,error=self.run_smoke(fault='exits_after_health');self.assertIsInstance(error,RuntimeError);self.assertIn('exited before after-health',str(error))
        self.assertEqual(state.health_requests,1);self.assertEqual(state.tool_calls,[]);self.assertEqual(report['listener_ownership'][-1]['child_exit_code'],0)
    def test_reused_pid_after_health_is_not_the_launched_child(self):
        report,state,error=self.run_smoke(fault='pid_reused_after_health');self.assertIsInstance(error,RuntimeError);self.assertIn('no longer identifies',str(error))
        self.assertEqual(state.tool_calls,[]);self.assertFalse(report['listener_ownership'][-1]['identity_matched'])
    def test_unbound_child_reaches_deadline_without_health_requests(self):
        report,state,error=self.run_smoke(fault='never_binds');self.assertIsInstance(error,RuntimeError);self.assertIn('did not own its loopback listener',str(error))
        self.assertEqual(state.health_requests,0);self.assertEqual(report['listener_ownership'],[]);self.assertGreater(state.now,60)
    def test_denied_system_table_falls_back_to_child_table_with_recorded_scope(self):
        report,state,error=self.run_smoke(fault='system_table_denied');self.assertIsNone(error);self.assertEqual(report['status'],'PASSED')
        self.assertEqual({row['table'] for row in report['listener_ownership']},{'child'});self.assertIn('child-table',state.sequence);self.assertNotIn('system-table',state.sequence)

def json_load(path):return json.loads(path.read_bytes())
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path);args=parser.parse_args()
    if args.evidence:EVIDENCE=args.evidence.resolve();EVIDENCE.mkdir(exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checker))
    record={'passed':result.wasSuccessful(),'cases':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Actual smoke function with inert subprocess, process/listener table, HTTP/MCP and Win32 effects; no native smoke, build, listener, provider or model execution'}
    if EVIDENCE:(EVIDENCE/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2));raise SystemExit(0 if result.wasSuccessful() else 1)
