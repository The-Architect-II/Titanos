"""Failure-path regressions for the actual private Windows Job implementation."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if os.name != 'nt':
    print(json.dumps({'skipped':'Native Windows Job implementation requires Windows'}))
    raise SystemExit(0)
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('actual_windows_job_fixture',ROOT/'src/titanos/_windows_job.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
results=[]

class Process:
    _handle=77
    def __init__(self, kill_error=None, wait_error=None):
        self.kill_error=kill_error;self.wait_error=wait_error;self.calls=[]
    def kill(self):
        self.calls.append('kill')
        if self.kill_error:raise self.kill_error
    def wait(self, timeout):
        self.calls.append('wait')
        if self.wait_error:raise self.wait_error
        return 0

def require(ok):
    if not ok:raise OSError('controlled CloseHandle failure')

def kernel(closes):
    return SimpleNamespace(CreateJobObjectW=lambda *a:55,SetInformationJobObject=lambda *a:True,
                           CloseHandle=lambda handle:closes.append(handle) or False)

# Constructor cleanup must try every owned cleanup step and preserve the original error.
for combined in (False,True):
    primary=ValueError('controlled assignment failure');closes=[]
    process=Process(RuntimeError('controlled kill failure') if combined else None,
                    TimeoutError('controlled wait failure') if combined else None)
    def failed_assign(self):raise primary
    with patch.object(module,'kernel',kernel(closes)),patch.object(module,'require',require),patch.object(module.subprocess,'Popen',return_value=process),patch.object(module.OwnedJob,'_assign',failed_assign):
        try:module.OwnedJob(['not-executed'])
        except BaseException as caught:
            assert caught is primary
        else:raise AssertionError('Assignment failure hidden')
    owned=primary.owned_job
    assert owned.handle==55 and owned.process is process
    assert process.calls==['kill','wait'] and closes==[55]
    expected=['kill_child','wait_child','close_job'] if combined else ['close_job']
    assert [row['step'] for row in owned.cleanup_errors]==expected
    assert len(primary.__notes__)==len(expected)
    results.append({'case':'constructor_combined' if combined else 'constructor_close_failure','passed':True})

# __exit__ must retain uncertain handles and preserve any active primary exception.
for primary_present,wait_failure in ((False,False),(True,False),(False,True),(True,True)):
    closes=[];primary=RuntimeError('controlled OCR deadline') if primary_present else None
    owned=module.OwnedJob.__new__(module.OwnedJob);owned.handle=55;owned.cleanup_errors=[]
    wait_error=TimeoutError('controlled wait failure') if wait_failure else None
    owned.process=Process(wait_error=wait_error);owned.active_count=lambda:0
    with patch.object(module,'kernel',kernel(closes)),patch.object(module,'require',require):
        try:
            returned=owned.__exit__(type(primary) if primary else None,primary,None)
        except BaseException as caught:
            assert not primary_present
            if wait_failure:assert caught is wait_error
            else:assert isinstance(caught,OSError)
            error=caught
        else:
            assert primary_present and returned is False
            error=primary
    assert error.owned_job is owned and owned.handle==55 and closes==[55]
    assert len(owned.cleanup_errors)==(2 if wait_failure else 1)
    assert len(getattr(error,'__notes__',[]))==(len(owned.cleanup_errors) if primary_present else len(owned.cleanup_errors)-1)
    results.append({'case':'exit','primary_present':primary_present,'wait_failure':wait_failure,'passed':True})

# A caller may explicitly recover the retained handle; old failures remain recorded.
with patch.object(module,'kernel',SimpleNamespace(CloseHandle=lambda handle:True)):
    previous=list(owned.cleanup_errors);owned._close_handle()
    assert owned.handle is None and owned.cleanup_errors==previous
results.append({'case':'explicit_handle_recovery','passed':True})

# The normal close still empties ownership, with no diagnostics invented.
owned=module.OwnedJob.__new__(module.OwnedJob);owned.handle=55;owned.cleanup_errors=[]
owned.process=Process();owned.active_count=lambda:0;closes=[]
with patch.object(module,'kernel',SimpleNamespace(CloseHandle=lambda handle:closes.append(handle) or True)):
    assert owned.__exit__(None,None,None) is False
assert owned.handle is None and owned.cleanup_errors==[] and closes==[55]
results.append({'case':'normal_close','passed':True})
print(json.dumps({'status':'WINDOWS_JOB_CLEANUP_REGRESSIONS_PASSED','cases':results,'native_processes':0,'model_imports':0,'host_calls':0}))
