"""Stop the owned native process when Titanos closes or dies (stdin pipe EOF)."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def child_environment(environ):
    """The environment a native runtime or helper child starts from: titanos.child_env.allowlisted_env keeps only the
    allowlisted names (and TITANOS_CHILD_ENV_ALLOW entries), then drops the known provider variables, every *_API_KEY,
    *_BASE_URL, *_TOKEN, *_SECRET and *_PASSWORD name, and every variable an env: credential record in the broker vault
    index names (TITANOS_HOME/broker/vault/index.json; an unreadable or absent index adds nothing). Standard library
    only: this module runs under `python -I -S`."""
    from .child_env import allowlisted_env
    return allowlisted_env(environ, extra_deny=vault_env_vars())


def vault_env_vars():
    """Variables named by env: credential records of the broker vault (empty when there is no vault index)."""
    from .child_env import vault_env_vars as named
    return named()


def native_environment():
    """Restore external-runtime library lookup inside this isolated helper only."""
    env = child_environment({k: v for k, v in os.environ.items()
                             if not k.startswith('_PYI') and k not in {'PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP'}})
    if getattr(sys, 'frozen', False):
        if os.name == 'nt':
            import ctypes
            reset = ctypes.WinDLL('kernel32', use_last_error=True).SetDllDirectoryW
            reset.argtypes = [ctypes.c_wchar_p]
            reset.restype = ctypes.c_int
            if not reset(None):
                raise ctypes.WinError(ctypes.get_last_error())
        elif 'LD_LIBRARY_PATH_ORIG' in env:
            env['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH_ORIG']
        else:
            env.pop('LD_LIBRARY_PATH', None)
    return env


def query_devices(executable, timeout, *, query='--list-devices'):
    """Whitelisted model-free queries; their children remain owned on timeout."""
    import math
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Device query timeout must be finite and positive')
    if query not in {'--list-devices','--version'}:
        raise ValueError('Unsupported native metadata query')
    args = [executable, query]
    env = native_environment()
    if os.name == 'nt':
        from ._windows_job import OwnedJob
        with OwnedJob(args, env=env, stdin=subprocess.DEVNULL) as job:
            return job.process.wait(timeout=timeout)
    return subprocess.run(args, env=env, stdin=subprocess.DEVNULL, timeout=timeout).returncode


def _windows_stdin_eof(finished):
    """Wait for the parent's pipe without blocking a buffered reader at exit."""
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    peek=kernel.PeekNamedPipe
    peek.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD,
                   ctypes.c_void_p,ctypes.POINTER(wintypes.DWORD),ctypes.c_void_p]
    peek.restype=wintypes.BOOL
    descriptor=sys.stdin.fileno();handle=msvcrt.get_osfhandle(descriptor)
    while not finished.wait(.02):
        available=wintypes.DWORD()
        if not peek(handle,None,0,None,ctypes.byref(available),None):
            error=ctypes.get_last_error()
            if error==109:return  # ERROR_BROKEN_PIPE: all parent writers closed.
            raise ctypes.WinError(error)
        if available.value and not os.read(descriptor,min(available.value,65536)):
            return


def _supervise(process,pid_path):
    pid_path.write_text(str(process.pid))
    lock=threading.Lock()
    def stop(*unused):
        with lock:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
    if os.name!='nt':
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
    finished=threading.Event();watch_errors=[]
    def watch():
        if os.name=='nt':
            try:_windows_stdin_eof(finished)
            except OSError as error:watch_errors.append(error)
            if not finished.is_set():stop()
        else:
            sys.stdin.buffer.read()
            stop()
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    try:return process.wait()
    finally:
        stop()
        if os.name=='nt':
            finished.set();watcher.join(timeout=5)
            if watcher.is_alive():raise RuntimeError('Native stdin watcher did not stop')
            if watch_errors:raise watch_errors[0]


def main():
    if sys.argv[1:2] in (['--query-devices'], ['--query-version']):
        if len(sys.argv) != 4:
            raise ValueError('Device query requires timeout and executable')
        return query_devices(sys.argv[3], float(sys.argv[2]),
                             query='--version' if sys.argv[1]=='--query-version' else '--list-devices')
    pid_path=Path(sys.argv[1])
    env = native_environment()
    if os.name=='nt':
        from ._windows_job import OwnedJob
        # The supervisor alone holds this non-inherited Job handle. Even forced
        # supervisor termination closes it and kills the owned native child tree.
        with OwnedJob(sys.argv[2:],stdin=subprocess.DEVNULL,env=env) as job:
            return _supervise(job.process,pid_path)
    process=subprocess.Popen(sys.argv[2:],stdin=subprocess.DEVNULL,env=env)
    return _supervise(process,pid_path)


if __name__=='__main__':sys.exit(main())
