"""The actual Windows OCR caller must preserve its primary error through cleanup."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

if os.name != 'nt':
    print(json.dumps({'skipped': 'The native Windows OCR caller branch requires Windows'}))
    raise SystemExit(0)
ROOT = Path(__file__).resolve().parents[1]
package = ModuleType('titanos_job_caller_fixture');package.__path__=[]
engine = ModuleType(package.__name__+'.engine')
engine.EngineError = type('EngineError', (RuntimeError,), {})
jobs = ModuleType(package.__name__+'._windows_job')
sys.modules[package.__name__] = package
sys.modules[engine.__name__] = engine
sys.modules[jobs.__name__] = jobs
results = []

for primary_expected in (True, False):
    primary = RuntimeError('controlled OCR output-budget failure')
    cleanup = OSError('controlled owned-job close failure')
    exits = []
    checks = [0]

    class Job:
        def __init__(self, argv, **kwargs):
            assert argv == ['controlled-native-ocr']
            self.process = SimpleNamespace(poll=lambda: None if primary_expected else 0, returncode=0)

        def __exit__(self, kind, error, traceback):
            exits.append((kind, error, traceback))
            if error is None:
                raise cleanup
            error.add_note('Owned job cleanup: '+repr(cleanup))
            return False

    jobs.OwnedJob = Job
    name = package.__name__+'.owned_command'
    spec = importlib.util.spec_from_file_location(name, ROOT/'src/titanos/owned_command.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

    def check():
        checks[0] += 1
        if primary_expected and checks[0] == 2:
            raise primary

    with tempfile.TemporaryDirectory(prefix='titanos-job-caller-') as temporary:
        try:
            module.run_owned(['controlled-native-ocr'], Path(temporary), {}, check, 1024)
        except BaseException as error:
            assert error is (primary if primary_expected else cleanup)
        else:
            raise AssertionError('Expected controlled failure was hidden')
    assert len(exits) == 1
    if primary_expected:
        assert exits[0][0] is RuntimeError and exits[0][1] is primary and exits[0][2] is not None
        assert any('close failure' in note for note in primary.__notes__)
    else:
        assert exits[0] == (None, None, None)
    results.append({'primary_error_present':primary_expected,'expected_error_preserved':True,'one_cleanup_call':True})

print(json.dumps({'status':'WINDOWS_OCR_CALLER_REGRESSIONS_PASSED','cases':results,'native_processes':0,'model_imports':0,'host_calls':0}))
