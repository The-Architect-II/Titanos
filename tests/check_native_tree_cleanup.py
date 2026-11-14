"""Windows-only native-supervisor lifecycle checks with inert Python processes.

No model, GPU, product API or network. Each fixture tree has an outer private
Job so even the regression failure cannot leave its sleeping processes alive.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1])
grandchild=subprocess.Popen([sys.executable,'-I','-S','-B','-c','import time;time.sleep(300)'],
    stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW,close_fds=True)
temporary=root/'ready.tmp'
temporary.write_text(json.dumps({'child':os.getpid(),'grandchild':grandchild.pid}))
temporary.replace(root/'ready.json')
while not (root/'EXIT').exists():time.sleep(.05)
raise SystemExit(17)
'''


def supervisor_command(python, directory, child, executable=None):
    arguments = [str(directory / 'native.pid'), python, '-I', '-S', '-B', str(child), str(directory)]
    if executable is not None:
        return [str(executable), '--titanos-internal', 'titanos.native_process', *arguments]
    bootstrap = (
        'import runpy,sys\n'
        'sys.path.insert(0,sys.argv.pop(1))\n'
        'sys.argv[0]="titanos.native_process"\n'
        'runpy.run_module("titanos.native_process",run_name="__main__")\n'
    )
    return [python, '-I', '-S', '-B', '-c', bootstrap, str(ROOT / 'src'), *arguments]


def run_case(case, workspace, executable=None):
    from titanos._windows_job import OwnedJob
    # Use the actual interpreter, not a Windows venv launcher that delegates to
    # another process; the retained Popen handle must be the exact supervisor.
    python = str(Path(getattr(sys, '_base_executable', sys.executable)).resolve())
    directory = workspace / case
    directory.mkdir()
    child = directory / 'inert_child.py'
    child.write_text(CHILD, encoding='utf-8')
    argv = supervisor_command(python, directory, child, executable)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    for name, arguments, result in (
        ('OpenProcess', [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        ('WaitForSingleObject', [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        ('GetProcessTimes', [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)], wintypes.BOOL),
        ('CloseHandle', [wintypes.HANDLE], wintypes.BOOL),
    ):
        function = getattr(kernel, name)
        function.argtypes, function.restype = arguments, result

    def gone(handle):
        result = kernel.WaitForSingleObject(handle, 0)
        if result not in (0, 258):
            raise ctypes.WinError(ctypes.get_last_error())
        return result == 0

    row = {'case': case, 'status': 'starting', 'processes': {}, 'model_calls': 0,
           'supervisor_command': argv}
    handles = {}
    job = None
    try:
        with (directory / 'supervisor.log').open('wb') as log:
            with OwnedJob(argv, cwd=directory, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT) as job:
                row['supervisor_pid'] = job.process.pid
                deadline = time.monotonic() + 15
                while not (directory / 'ready.json').exists():
                    if job.process.poll() is not None:
                        raise AssertionError('Inert supervisor exited before fixture readiness')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Inert child/grandchild readiness timed out')
                    time.sleep(.02)
                pids = json.loads((directory / 'ready.json').read_text())
                if int((directory / 'native.pid').read_text()) != pids['child']:
                    raise AssertionError('Supervisor PID record differs from the actual fixture child')
                for role, pid in pids.items():
                    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
                    if not handle:
                        raise ctypes.WinError(ctypes.get_last_error())
                    handles[role] = handle
                    created, exited, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
                    if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                                   ctypes.byref(kernel_time), ctypes.byref(user_time)):
                        raise ctypes.WinError(ctypes.get_last_error())
                    row['processes'][role] = {'pid': pid, 'created_filetime': (created.dwHighDateTime << 32) | created.dwLowDateTime}
                    if gone(handle):
                        raise AssertionError('Fixture child must be alive before the tested lifecycle event')
                if case == 'forced_supervisor_stop':
                    job.process.kill()  # Exact retained supervisor handle; never a PID signal.
                elif case == 'stdin_eof':
                    job.process.stdin.write(b'ignored until EOF\n')
                    job.process.stdin.flush()
                    time.sleep(.1)
                    if job.process.poll() is not None or any(gone(handle) for handle in handles.values()):
                        raise AssertionError('Control data stopped the native tree before stdin EOF')
                    job.process.stdin.close()
                elif case == 'native_exit':
                    (directory / 'EXIT').write_text('exit with status 17', encoding='utf-8')
                else:
                    raise ValueError('Unknown fixture case')
                row['supervisor_exit_code'] = job.process.wait(timeout=15)
                deadline = time.monotonic() + 5
                while not all(gone(handle) for handle in handles.values()) and time.monotonic() < deadline:
                    time.sleep(.02)
                row['gone_before_outer_cleanup'] = {role: gone(handle) for role, handle in handles.items()}
                if not all(row['gone_before_outer_cleanup'].values()):
                    raise AssertionError('Native child tree survived its supervisor lifecycle event')
                if case == 'native_exit' and row['supervisor_exit_code'] != 17:
                    raise AssertionError('Supervisor changed the normal native exit code')
                row['status'] = 'passed'
    except BaseException as error:
        row.update(status='failed', error=repr(error))
    finally:
        row['outer_job_handle_closed'] = job is not None and job.handle is None
        # Windows termination is asynchronous; retain the exact child handles
        # until they signal, including after the outer Job's emergency cleanup.
        deadline = time.monotonic() + 5
        while not all(gone(handle) for handle in handles.values()) and time.monotonic() < deadline:
            time.sleep(.02)
        row['gone_after_outer_cleanup'] = {role: gone(handle) for role, handle in handles.items()}
        row['process_handles_closed'] = True
        for handle in handles.values():
            if not kernel.CloseHandle(handle):
                row['process_handles_closed'] = False
        if job is not None:
            if job.process.stdin is not None:
                job.process.stdin.close()
            job.process._handle.Close()
        if not row['outer_job_handle_closed'] or not all(row['gone_after_outer_cleanup'].values()) or not row['process_handles_closed']:
            row['status'] = 'failed'
        if row['status'] != 'passed':
            row['supervisor_log_tail'] = (directory / 'supervisor.log').read_text(encoding='utf-8', errors='replace')[-4000:]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--case', choices=['forced_supervisor_stop', 'stdin_eof', 'native_exit'])
    parser.add_argument('--executable', type=Path,
                        help='Test a built titanos.exe internal native helper instead of the source supervisor')
    args = parser.parse_args()
    if os.name != 'nt':
        print('SKIP: native supervisor Job cleanup requires Windows')
        return 0
    sys.path.insert(0, str(ROOT / 'src'))
    source = ROOT / 'src/titanos/native_process.py'
    result = {'scope': 'Inert Windows child/grandchild lifecycle only; no model or GPU work', 'cases': []}
    executable = args.executable.resolve() if args.executable is not None else None
    if executable is None:
        result.update(mode='source', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    else:
        if not executable.is_file() or executable.name.casefold() != 'titanos.exe':
            parser.error('--executable must name an existing built titanos.exe')
        result.update(mode='frozen', executable=str(executable), executable_bytes=executable.stat().st_size,
                      executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
    with tempfile.TemporaryDirectory(prefix='titanos-native-tree-', dir=args.report.parent if args.report else None) as folder:
        for case in ([args.case] if args.case else ['forced_supervisor_stop', 'stdin_eof', 'native_exit']):
            row = run_case(case, Path(folder), executable)
            result['cases'].append(row)
            if row['status'] != 'passed':
                break
    result['status'] = 'passed' if all(row['status'] == 'passed' for row in result['cases']) else 'failed'
    if executable is not None:
        result['executable_unchanged'] = hashlib.sha256(executable.read_bytes()).hexdigest() == result['executable_sha256']
        if not result['executable_unchanged']:
            result['status'] = 'failed'
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
