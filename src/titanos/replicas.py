"""Bounded local process workers. Each replica owns its model and device visibility."""

import multiprocessing as mp
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace

from .config import Config
from .engine import Engine, EngineError, _finish_inference
from .settings import Settings, _active_settings, current_settings


def _log_worker_exception(exc, phase):
    """Log code locations, never exception values, request data or local paths."""
    try:
        frames = []
        cursor = exc.__traceback__
        while cursor is not None:
            code = cursor.tb_frame.f_code
            frames.append({'file': code.co_filename.replace('\\', '/').rsplit('/', 1)[-1],
                           'function': code.co_name, 'line': cursor.tb_lineno})
            if len(frames) > 32:
                frames.pop(0)
            cursor = cursor.tb_next
        logging.getLogger(__name__).error('Worker failure: %s', json.dumps({
            'type': type(exc).__name__, 'phase': phase, 'trace': frames}))
    except Exception:
        pass  # A logging failure must not replace the original worker error.


def _profiling_failed(meta, exc, phase):
    _log_worker_exception(exc, phase)
    meta['profiling_failure'] = {'type': type(exc).__name__, 'phase': phase}
    meta['profiling_evidence'] = {'requested': True, 'available': False, 'status': 'failed'}
    meta['observed_gpu_indices'] = None


def process_running(process):
    if process is None:return False
    try:return process.is_alive()
    except Exception:return None  # Unknown must not be reported as stopped.


def process_pid(process):
    if process is None:return None
    try:value=process.pid
    except Exception:return None
    return value if type(value) is int and value > 0 else None


def process_identity(process, metadata):
    """Cached IDs are historical reporting data, never termination targets."""
    observed=process_pid(process)
    cached=next((metadata.get(key) for key in ('pid','last_known_pid','last_pid')
                 if type(metadata.get(key)) is int and metadata[key]>0),None)
    return {'pid':observed,'pid_verified':observed is not None and process_running(process) is True,
            'last_known_pid':observed if observed is not None else cached}


def resolve_gpus(gpus):
    if not gpus:
        return []
    from .devices import admit_gpu_uuids, hip_records, require_homogeneous, scan
    if all(g.startswith('GPU-') for g in gpus):
        records = scan()
        selected = admit_gpu_uuids(gpus, physical=records)
        if len(selected) > 1:
            require_homogeneous([next(row for row in records if row['uuid'].lower() == g.lower()) for g in selected])
        return selected
    if all(g.startswith('HIP-') for g in gpus):
        records = hip_records()
        selected = admit_gpu_uuids(gpus, family='hip', visible=records)
        rows = [next(row for row in records if row['uuid'].lower() == g.lower()) for g in selected]
        if len(rows) > 1:
            require_homogeneous(rows)
        inherited = os.environ.get('HIP_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'))
        tokens = inherited.split(',') if inherited is not None else None
        indices = [int(row['runtime_address'].split(':')[-1]) for row in rows]
        if tokens is not None:
            if any(i >= len(tokens) for i in indices) or any(not token.strip().isdecimal() for token in tokens):
                raise ValueError('ROCm visibility cannot be narrowed without an unambiguous inherited ordinal map')
            return [tokens[i].strip() for i in indices]
        return list(map(str, indices))
    raise ValueError('Worker groups require one family of stable GPU/HIP UUIDs; numeric assignments must be repinned')


def worker_main(connection, cancel, configuration, assignment, settings=None):
    # Spawn does not inherit the application's ContextVar. Validate the plain
    # settings snapshot before parsing models or importing a backend, while the
    # saved worker assignment remains the more specific thread selection.
    effective = Settings(**settings) if settings is not None else current_settings()
    token = _active_settings.set(replace(effective, threads=assignment['threads']))
    try:
        return _worker_main(connection, cancel, configuration, assignment)
    finally:
        _active_settings.reset(token)


def _worker_main(connection, cancel, configuration, assignment):
    _scrub_environment()  # under spawn: credentials dropped, then CUDA visibility set, before any inference import
    if assignment['gpus']:
        if assignment['gpus'][0].startswith('HIP-'):
            os.environ['HIP_VISIBLE_DEVICES']=','.join(assignment['resolved_gpus'])
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(assignment['resolved_gpus'])
    elif assignment['device'] == 'cpu':
        os.environ.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
    os.environ['OMP_NUM_THREADS'] = str(assignment['threads'])
    os.environ['MKL_NUM_THREADS'] = str(assignment['threads'])
    os.environ['TITANOS_THREADS'] = str(assignment['threads'])
    os.environ['TITANOS_PROFILE_INFERENCE'] = '1' if assignment['profile'] else '0'
    cfg = Config.model_validate(configuration)
    from .config import WorkerSpec, worker_assignment
    selected = WorkerSpec.model_validate({k: v for k, v in assignment.items() if k != 'resolved_gpus'})
    cfg.models = {name: worker_assignment(spec, selected) for name, spec in cfg.models.items()}
    engine = Engine(cfg)
    if os.name != 'nt':
        import signal
        def stop_signal(*unused):
            if engine.runtime is not None and hasattr(engine.runtime, 'close'):
                engine.runtime.close()
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop_signal)
    from .devices import visibility_constraints
    meta = {'pid': os.getpid(), 'assignment': assignment, 'completed_requests': 0,
            'effective_visibility': visibility_constraints(), 'kernel_execution_verified': False}
    phase = 'device_verification'
    try:
        if assignment['gpus']:
            import torch
            if not torch.version.hip:
                from .devices import verify_visible
                meta['verified_devices'] = verify_visible(assignment['resolved_gpus'], torch)
            else:
                from .devices import normalized_uuid
                actual=['HIP-'+str(normalized_uuid(getattr(torch.cuda.get_device_properties(i),'uuid',None))) for i in range(torch.cuda.device_count())]
                if actual!=assignment['gpus']:
                    raise ValueError('ROCm runtime UUID/order mismatch; assignment was not substituted')
                meta['verified_devices']=[{'local_device':f'cuda:{i}','uuid':u} for i,u in enumerate(actual)]
        connection.send(('ready', meta))
        while True:
            message = connection.recv()
            if not isinstance(message, (tuple, list)) or len(message) not in (4, 5):
                raise ValueError('Invalid worker request envelope')
            command, name, task, data = message[:4]
            admission = message[4] if len(message) == 5 else None
            if len(message) == 5:
                if (command != 'infer' or not isinstance(admission, dict)
                        or set(admission) != {'require_loaded', 'request_id'}
                        or admission['require_loaded'] is not True
                        or not isinstance(admission['request_id'], str)
                        or len(admission['request_id']) != 32
                        or any(c not in '0123456789abcdef' for c in admission['request_id'])):
                    raise ValueError('Invalid worker admission envelope')
            if command == 'close':
                return
            try:
                phase = 'dispatch'
                if command == 'infer':
                    phase = 'profile_setup'
                    meta['profiling_evidence'] = {'requested': assignment['profile'], 'available': False, 'status': 'not_collected'}
                    profile = None
                    if assignment['profile'] and cfg.models[name].backend in {'transformers', 'diffusers', 'xtts', 'qwen3-tts'}:
                        import torch
                        activities = [torch.profiler.ProfilerActivity.CPU]
                        if torch.cuda.is_available():
                            activities.append(torch.profiler.ProfilerActivity.CUDA)
                        if cfg.models[name].parallelism != 'tensor_parallel':
                            profile = torch.profiler.profile(activities=activities)
                    started = time.monotonic()
                    phase = 'profile_enter' if profile else 'infer'
                    inference_error = None
                    completed = False
                    try:
                        with profile if profile else nullcontext():
                            phase = 'infer'
                            try:
                                payload = {k: v for k, v in data.items() if k not in {'worker', 'require_loaded'}}
                                emit = lambda text: connection.send(('text', text)) if not cancel.is_set() else None
                                from .jobs import _native_generation_observer
                                def child_progress(value):
                                    if isinstance(value,dict):
                                        connection.send(('progress',{key:item for key,item in value.items() if key!='job_id'}))
                                progress_binding=_native_generation_observer.set((None,child_progress))
                                try:
                                    if admission is not None:
                                        def admitted():
                                            connection.send(('admitted', {'request_id': admission['request_id'],
                                                                        'model': name, 'pid': os.getpid()}))
                                        result = engine.infer(name, task, payload, emit, cancel,
                                                              require_loaded=True, on_admitted=admitted)
                                    else:
                                        result = engine.infer(name, task, payload, emit, cancel)
                                finally:_native_generation_observer.reset(progress_binding)
                            except BaseException as exc:
                                inference_error = exc
                                raise
                            completed = True
                            phase = 'profile_exit' if profile else 'result_metadata'
                        if inference_error is not None:
                            raise inference_error  # A profiler cannot suppress a backend error.
                    except BaseException as exc:
                        if inference_error is not None:
                            if exc is not inference_error:
                                _profiling_failed(meta, exc, 'profile_exit')
                                inference_error.add_note('Additional profiler exit failure: '+type(exc).__name__)
                            phase = 'infer'
                            raise inference_error
                        if profile is not None:
                            _profiling_failed(meta, exc, phase)
                        if not (completed and phase == 'profile_exit' and isinstance(exc, MemoryError)):
                            raise
                    finally:
                        inference_error = None
                    phase = 'result_metadata'
                    meta['completed_requests'] += 1
                    meta['last_request_seconds'] = time.monotonic() - started
                    meta['observed_gpu_indices'] = None
                    if profile and not meta.get('profiling_failure'):
                        phase = 'profile_events'
                        try:
                            meta['observed_gpu_indices'] = sorted({e.device_index for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA})
                            meta['profiling_evidence'].update(available=True, status='collected')
                        except Exception as exc:
                            _profiling_failed(meta, exc, phase)
                            if not isinstance(exc, MemoryError):raise
                elif command == 'load':
                    phase = 'load'
                    result = engine.load(name)
                elif command == 'unload':
                    phase = 'unload'
                    result = engine.unload(name)
                else:
                    phase = 'details'
                    result = engine.details(name)
                if meta.get('profiling_failure'):
                    profile = None  # Release the failed collector before result marshalling.
                phase = 'runtime_metadata'
                meta['current_model'] = engine.current
                details = engine._loaded_details if admission is not None else engine.details
                meta['runtime'] = None if meta.get('profiling_failure') else details(engine.current) if engine.current else None
                connection.send(('done', (result, meta)))
                if meta.get('profiling_failure'):return
            except Exception as exc:
                if phase in {'profile_setup', 'profile_enter', 'profile_exit', 'profile_events'} and assignment['profile']:
                    if not meta.get('profiling_failure'):_profiling_failed(meta, exc, phase)
                else:
                    _log_worker_exception(exc, phase)
                if meta.get('profiling_failure'):
                    profile = None
                phase = 'error_metadata'
                meta['current_model'] = engine.current
                meta['runtime_fault'] = engine._runtime_fault()
                if meta['runtime_fault'] is not None:
                    details = engine._loaded_details if admission is not None else engine.details
                    meta['runtime'] = details(engine.current)
                connection.send(('error', (str(exc), getattr(exc, 'status', 500), getattr(exc, 'code', 'worker_error'), meta)))
                if meta.get('profiling_failure'):return
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        _log_worker_exception(exc, phase)
        connection.send(('error', (str(exc), 503, 'device_verification_failed', meta)))
    finally:
        try:
            # Do not run a poisoned device destructor in a graceful close path.
            # The existing parent worker owns bounded process termination.
            if not meta.get('profiling_failure') and engine._runtime_fault() is None:engine.unload()
        finally:connection.close()


class Worker:
    def __init__(self, name, assignment, config, settings=None):
        self.name, self.assignment = name, assignment
        self.settings = Settings(**asdict(settings or current_settings()))
        self.config = Config(models={n: config.models[n] for n in assignment.models})
        from .config import worker_assignment
        for spec in self.config.models.values():
            worker_assignment(spec, assignment)
        self.process = None
        self.connection = None
        self.cancel = None
        self.busy = False
        self.last_used = 0
        self.meta = {'current_model': None, 'completed_requests': 0}
        self.error = None
        self.failed_model = None
        self.runtime_fault = None
        self.profiling_failure = None
        self.stop_failed = False

    @property
    def unavailable(self):
        return self.runtime_fault is not None or getattr(self, 'profiling_failure', None) is not None or self.stop_failed or process_running(self.process) is None

    def unavailable_details(self, name):
        return {'backend':self.config.models[name].backend, 'device':'cuda:0' if self.assignment.gpus else self.assignment.device,
                'status':'unavailable', 'resident':process_running(self.process),
                'runtime_fault':self.runtime_fault, 'stop_error':self.meta.get('stop_error'),
                'profiling_failure':getattr(self, 'profiling_failure', None), 'profiling_evidence':self.meta.get('profiling_evidence'),
                'profiling_retirement':self.meta.get('profiling_retirement'),
                'process_identity':process_identity(self.process,self.meta)}

    def _retire_profiling(self):
        self.profiling_failure = dict(self.meta['profiling_failure'])
        try:
            self.stop(force=True)
        except Exception as exc:
            _log_worker_exception(exc, 'profile_retirement')
            self.meta['profiling_retirement'] = {'status':'unconfirmed', 'error_type':type(exc).__name__}
        else:
            self.meta['profiling_retirement'] = {'status':'retired'}

    def stop(self, force=False):
        failures=[]
        def attempt(label, action):
            try:action()
            except Exception as exc:failures.append(label+': '+type(exc).__name__+': '+str(exc))
        if self.process:
            if self.cancel:attempt('cancel', self.cancel.set)
            if self.connection and process_running(self.process) is not False and self.runtime_fault is None and not self.stop_failed and not force:
                attempt('close request', lambda:self.connection.send(('close', None, None, None)))
                attempt('graceful join', lambda:self.process.join(timeout=5))
            if process_running(self.process) is not False:
                current_pid=process_pid(self.process)
                if current_pid is None or process_running(self.process) is not True:
                    failures.append(('PID-based stop skipped: ' if os.name == 'nt' else '')+'current owned process identity is unverified')
                elif os.name == 'nt':
                    import subprocess
                    def kill_tree():
                        result=subprocess.run(['taskkill','/PID',str(current_pid),'/T','/F'],capture_output=True,timeout=10)
                        if result.returncode:raise OSError('taskkill did not confirm termination')
                    attempt('owned tree stop', kill_tree)
                if process_running(self.process) is not False:attempt('terminate', self.process.terminate)
                attempt('terminate join', lambda:self.process.join(timeout=8))
            if process_running(self.process) is not False:
                attempt('kill', self.process.kill)
                attempt('kill join', lambda:self.process.join(timeout=5))
            if process_running(self.process) is not False:
                self.stop_failed=True
                identity=process_identity(self.process,self.meta)
                self.meta.update(last_known_pid=identity['last_known_pid'], stop_error='; '.join(failures) or 'Process is still alive after bounded stop', residency_unconfirmed=True)
                raise EngineError(f"Worker {self.name} remains alive or unverified (observed PID {identity['pid']}, last known {identity['last_known_pid']}); its process and last model identity are retained",503,'worker_still_running')
            self.meta['last_pid']=process_identity(self.process,self.meta)['last_known_pid']
        if self.connection:attempt('connection close', self.connection.close)
        self.process = self.connection = None
        self.stop_failed = False
        self.meta['current_model'] = None
        if self.runtime_fault is not None:
            self.meta['runtime'] = {'status':'unavailable','available':False,'resident':False,'runtime_fault':self.runtime_fault}
        self.meta.pop('loading_model', None)
        self.meta.pop('residency_unconfirmed', None)
        self.meta.pop('stop_error', None)
        if failures:self.meta['stop_diagnostics']=failures

    def start(self):
        if self.unavailable:
            if getattr(self, 'profiling_failure', None) is not None:
                raise EngineError(f'Worker {self.name} is unavailable after a profiling failure; recreate the explicit worker before further inference',503,'profiling_unavailable')
            raise EngineError(f'Worker {self.name} is unavailable after a fatal runtime or failed stop; recreate the explicit worker only after re-admission',503,'device_unavailable' if self.runtime_fault is not None else 'worker_still_running')
        if self.process and self.process.is_alive():
            return
        self.stop()
        assignment = self.assignment.model_dump()
        assignment['resolved_gpus'] = resolve_gpus(assignment['gpus'])
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe()
        self.connection, self.cancel = parent, ctx.Event()
        self.process = ctx.Process(target=worker_main, args=(child, self.cancel, self.config.model_dump(), assignment, asdict(self.settings)), name='titanos-' + self.name, daemon=False)
        self.process.start()
        child.close()
        self.error = None

    def _require_loaded(self, name):
        if self.unavailable:
            if self.profiling_failure is not None:
                raise EngineError('Worker is unavailable after a profiling failure', 503, 'profiling_unavailable')
            raise EngineError('Worker is unavailable or its residency is unconfirmed', 503,
                              'device_unavailable' if self.runtime_fault is not None else 'worker_still_running')
        running = process_running(self.process)
        if running is not True:
            if self.process is not None or self.meta.get('current_model') is not None:
                raise EngineError('The loaded worker is no longer running', 503, 'worker_unavailable')
            raise EngineError('The requested model is not already loaded', 409, 'model_not_loaded')
        if self.meta.get('current_model') != name:
            raise EngineError('The requested model is not already loaded', 409, 'model_not_loaded')
        if self.connection is None or self.cancel is None or self.meta.get('residency_unconfirmed') or process_pid(self.process) is None:
            raise EngineError('Loaded worker communication or identity is unavailable', 503, 'worker_unavailable')

    def call(self, command, name, task=None, data=None, emit=None, cancel=None, *,
             require_loaded=False, on_admitted=None):
        import subprocess
        if type(require_loaded) is not bool or (require_loaded and command != 'infer'):
            raise EngineError('Invalid loaded-only worker operation')
        if isinstance(data, dict):
            data = {key: value for key, value in data.items() if key != 'require_loaded'}
        if require_loaded:
            if cancel and cancel.is_set():
                raise EngineError('Request cancelled', 499, 'cancelled')
            self._require_loaded(name)
        if self.unavailable:
            if command == 'details':return self.unavailable_details(name)
            if command == 'unload':
                self.stop()
                return {'model':name,'status':'unloaded','available':False}
        try:
            if not require_loaded:
                self.start()
        except (ValueError, OSError, ImportError, subprocess.SubprocessError) as exc:
            self.error, self.failed_model = str(exc), name
            error=EngineError(f'Cannot start worker {self.name}: {exc}',503,'worker_unavailable')
            try:self.stop(force=True)
            except Exception as cleanup:error.add_note('Worker cleanup: '+str(cleanup))
            raise error from exc
        process, connection, worker_cancel = self.process, self.connection, self.cancel
        worker_cancel.clear()
        if require_loaded and cancel and cancel.is_set():
            raise EngineError('Request cancelled', 499, 'cancelled')
        if command in {'load', 'infer', 'details'} and self.meta.get('current_model') != name:
            # Keep the last acknowledged resident model until the child replies.
            self.meta['loading_model'] = name
        deadline = time.monotonic() + self.assignment.timeout
        request_id = uuid.uuid4().hex if require_loaded else None
        expected_admission = {'request_id': request_id, 'model': name, 'pid': process_pid(process)}
        admitted = False
        admission_error = None
        try:
            if require_loaded:
                self._require_loaded(name)
                if self.process is not process or self.connection is not connection or self.cancel is not worker_cancel:
                    raise OSError('Loaded worker ownership changed before dispatch')
                connection.send((command, name, task, data, {'require_loaded': True, 'request_id': request_id}))
            else:
                connection.send((command, name, task, data))
            while True:
                if cancel and cancel.is_set():
                    worker_cancel.set()
                if require_loaded and (self.process is not process or self.connection is not connection or self.cancel is not worker_cancel):
                    raise OSError('Loaded worker ownership changed during the request')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Worker request exceeded its configured timeout')
                if not connection.poll(min(0.1, remaining)):
                    running = process_running(process) if require_loaded else process.is_alive()
                    if not running:
                        raise EOFError('Worker exited during the request')
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError('Worker response became ready after its configured timeout')
                reply = connection.recv()
                if require_loaded and (not isinstance(reply, (tuple, list)) or len(reply) != 2):
                    raise OSError('Invalid loaded worker response envelope')
                kind, value = reply
                if time.monotonic() >= deadline:
                    raise TimeoutError('Worker response was received after its configured timeout')
                if require_loaded and (self.process is not process or self.connection is not connection
                                       or self.cancel is not worker_cancel):
                    raise OSError('Loaded worker ownership changed while receiving its response')
                # A child may exit immediately after its fatal error reply. Keep
                # that fault, but never accept admission/success from a dead child.
                if require_loaded and kind != 'error' and (process_running(process) is not True
                                                          or process_pid(process) != expected_admission['pid']):
                    raise OSError('Loaded worker liveness changed while receiving its response')
                if kind == 'admitted':
                    if (not require_loaded or admitted or not isinstance(value, dict)
                            or set(value) != {'request_id', 'model', 'pid'}
                            or type(value['pid']) is not int or value != expected_admission):
                        raise OSError('Invalid or duplicate worker admission acknowledgement')
                    admitted = True
                    if on_admitted is not None:
                        try:on_admitted()
                        except BaseException as exc:
                            admission_error = exc
                            worker_cancel.set()
                elif require_loaded and (kind not in {'text','progress','error','done'} or (not admitted and kind != 'error')):
                    raise OSError('Worker response preceded admission or had an invalid kind')
                elif kind == 'ready':
                    self.meta.update(value)
                elif kind == 'text':
                    if emit and admission_error is None and not (cancel and cancel.is_set()):
                        emit(value)
                elif kind=='progress':
                    if not isinstance(value,dict) or len(value)>16:
                        raise OSError('Invalid worker progress event')
                    try:
                        from .jobs import _native_generation_observer
                        observer=_native_generation_observer.get()
                        if observer:
                            job_id,callback=observer;callback({**value,'job_id':job_id})
                    except (TypeError,ValueError) as exc:
                        raise OSError('Invalid worker progress event') from exc
                elif kind == 'error':
                    if require_loaded and (not isinstance(value, (tuple, list)) or len(value) != 4
                                           or not isinstance(value[3], dict)):
                        raise OSError('Invalid loaded worker error response')
                    message, status, code, self.meta = value
                    self.error = message
                    self.failed_model = name
                    error=EngineError(message,status,code)
                    if self.meta.get('profiling_failure'):
                        if self.meta.get('runtime_fault'):self.runtime_fault=self.meta['runtime_fault']
                        self._retire_profiling()
                    elif code == 'device_unavailable' and self.meta.get('runtime_fault'):
                        self.runtime_fault=self.meta['runtime_fault']
                        try:self.stop(force=True)
                        except Exception as cleanup:error.add_note('Poisoned worker stop: '+str(cleanup))
                    if admission_error is not None:
                        admission_error.add_note('Worker also failed: ' + error.code)
                        raise admission_error
                    raise error
                else:
                    if require_loaded and (not isinstance(value, (tuple, list)) or len(value) != 2
                                           or not isinstance(value[1], dict)):
                        raise OSError('Invalid loaded worker result response')
                    result, self.meta = value
                    self.error = None
                    self.failed_model = None
                    if self.meta.get('profiling_failure'):self._retire_profiling()
                    if admission_error is not None:raise admission_error
                    return result
        except (EOFError, BrokenPipeError, OSError, TimeoutError) as exc:
            self.error = str(exc)
            self.failed_model = name
            error=EngineError(f'Worker {self.name}: {exc}; request was not replayed',503,'worker_unavailable')
            if not require_loaded or (self.process is process and self.connection is connection):
                try:self.stop(force=True)
                except Exception as cleanup:error.add_note('Worker cleanup: '+str(cleanup))
            else:
                error.add_note('Bound worker ownership changed; no replacement was stopped or retried')
            if admission_error is not None:
                admission_error.add_note('Worker also failed: ' + error.code)
                raise admission_error from exc
            raise error from exc


class ReplicaEngine:
    def __init__(self, config, settings=None):
        settings = settings or current_settings()
        self.queue_capacity = settings.replica_queue_capacity
        self.queue_timeout = settings.replica_queue_timeout
        self.config = config
        self.workers = [Worker(name, spec, config, settings=settings) for name, spec in config.workers.items()]
        assigned = {name for worker in config.workers.values() for name in worker.models}
        # Unassigned models retain the ordinary single-process lifecycle when an
        # operator adds workers for other models.
        self.local = Engine(Config(models={n: s for n, s in config.models.items() if n not in assigned}))
        self.condition = threading.Condition()
        self.started = int(time.time())
        self.pending = 0
        self.sequence = 0
        self.closed = False

    def validate(self, name, task=None):
        Engine.validate(self, name, task)

    @contextmanager
    def acquire(self, name, worker=None, cancel=None, allow_unavailable=False, *, require_loaded=False):
        self.validate(name)
        candidates = [w for w in self.workers if name in w.assignment.models and (worker is None or w.name == worker)]
        if not candidates:
            raise EngineError('No matching worker assignment', 404, 'worker_not_found')
        with self.condition:
            queued = False
            deadline = time.monotonic() + self.queue_timeout
            try:
                while True:
                    if self.closed:
                        raise EngineError('Engine is shutting down', 503)
                    if cancel and cancel.is_set():
                        raise EngineError('Request cancelled', 499, 'cancelled')
                    if queued and time.monotonic() >= deadline:
                        raise EngineError('Timed out waiting for an available replica', 503, 'worker_busy')
                    if require_loaded:
                        eligible, refusals = [], []
                        for candidate in candidates:
                            try:candidate._require_loaded(name)
                            except EngineError as exc:refusals.append(exc)
                            else:eligible.append(candidate)
                        if not eligible:
                            raise next((exc for exc in refusals if exc.code != 'model_not_loaded'), refusals[0])
                    else:
                        eligible=[w for w in candidates if allow_unavailable or not getattr(w,'unavailable',False)]
                    if not eligible:
                        raise EngineError('All matching explicit workers are unavailable; no request was replayed',503,'device_unavailable')
                    idle = [w for w in eligible if not w.busy]
                    if idle:
                        chosen = min(idle, key=lambda w: (w.meta.get('current_model') != name, w.last_used))
                        chosen.busy = True
                        self.sequence += 1
                        chosen.last_used = self.sequence
                        break
                    if not queued:
                        if self.pending >= self.queue_capacity:
                            raise EngineError('Replica request queue is full', 429, 'rate_limit_error')
                        self.pending += 1
                        queued = True
                    self.condition.wait(min(0.1, max(0, deadline - time.monotonic())))
            finally:
                if queued:
                    self.pending -= 1
        try:
            yield chosen
        finally:
            with self.condition:
                chosen.busy = False
                self.condition.notify_all()

    def infer(self, name, task, data, emit=None, cancel=None, *, require_loaded=False,
              on_admitted=None, on_finished=None):
        self.validate(name, task)
        if type(require_loaded) is not bool:
            raise EngineError('require_loaded must be a boolean')
        data = {key: value for key, value in data.items() if key != 'require_loaded'}
        if name in self.local.config.models:
            if require_loaded:
                return self.local.infer(name, task, data, emit, cancel, require_loaded=True,
                                        on_admitted=on_admitted, on_finished=on_finished)
            return self.local.infer(name, task, data, emit, cancel)
        options = {'require_loaded': True} if require_loaded else {}
        with self.acquire(name, data.get('worker'), cancel, **options) as worker:
            if not require_loaded:
                return worker.call('infer', name, task, data, emit, cancel)
            admitted = False
            result = error = None
            def acknowledge():
                nonlocal admitted
                admitted = True
                if on_admitted is not None:on_admitted()
            try:
                result = worker.call('infer', name, task, data, emit, cancel,
                                     require_loaded=True, on_admitted=acknowledge)
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                if admitted:_finish_inference(on_finished, result, error)

    def load(self, name, worker=None):
        self.validate(name)
        if name in self.local.config.models:
            return self.local.load(name, worker)
        targets = [w.name for w in self.workers if name in w.assignment.models and (worker is None or w.name == worker)]
        if not targets:
            raise EngineError('No matching worker', 404)
        # Explicit load warms every replica; inference can load any one lazily.
        for target in targets:
            with self.acquire(name, target) as w:
                w.call('load', name)
        return {'model': name, 'status': 'loaded', 'workers': targets}

    def unload(self, name=None, worker=None):
        if name is None:
            return self.close()
        self.validate(name)
        if name in self.local.config.models:
            return self.local.unload(name, worker)
        targets = [w for w in self.workers if name in w.assignment.models and (worker is None or w.name == worker)]
        if not targets:
            raise EngineError('No matching worker', 404)
        for target in targets:
            with self.acquire(name, target.name, allow_unavailable=True) as w:
                if process_running(w.process) is not False:
                    w.call('unload', name)
        return {'model': name, 'status': 'unloaded'}

    def loaded_details(self, name):
        """Passive runtime details for the resident unassigned model only. Worker-assigned models live in other processes and are
        never queried here, so the capabilities route keeps its passive view for them; nothing is loaded or awaited."""
        if name in self.local.config.models:
            return self.local.loaded_details(name)
        return None

    def details(self, name, worker=None):
        self.validate(name)
        if name in self.local.config.models:
            return self.local.details(name, worker)
        with self.acquire(name, worker, allow_unavailable=True) as w:
            result=w.call('details', name)
            return {'worker': w.name, **result, 'process': {**w.meta,**process_identity(w.process,w.meta),
                                                         'running':process_running(w.process)}}

    def worker_status(self):
        with self.condition:
            return [{'id': w.name, 'busy': w.busy,
                     'assignment': w.assignment.model_dump(), 'last_error': w.error, 'last_error_model': w.failed_model, **w.meta,
                     'running': process_running(w.process), **process_identity(w.process,w.meta),
                     'unavailable':getattr(w,'unavailable',False)} for w in self.workers]

    def catalog(self, loaded_only=False):
        status = self.worker_status()
        rows = []
        for name, spec in self.config.models.items():
            if name in self.local.config.models:
                continue
            resident = [w['id'] for w in status if w['running'] is not False and w.get('current_model') == name]
            replicas = [w['id'] for w in status if w['id'] in resident and not w['unavailable']]
            unavailable = [w['id'] for w in status if w['unavailable'] and name in w['assignment']['models']]
            all_unavailable = unavailable and all(w['unavailable'] for w in status if name in w['assignment']['models'])
            state = 'loaded' if replicas else 'unavailable' if all_unavailable or resident else 'loading' if any(w.get('loading_model') == name for w in status) else 'error' if any(w.get('last_error_model') == name for w in status) else 'unloaded'
            if not loaded_only or resident:
                rows.append({'id': name, 'object': 'model', 'created': self.started, 'owned_by': 'local',
                             'task': spec.task, 'status': state, 'loaded_workers': replicas,
                             **({'unavailable_workers':unavailable,'resident_workers':resident,
                                 'unconfirmed_workers':[w['id'] for w in status if w['id'] in resident and (w['running'] is None or w.get('residency_unconfirmed'))]} if unavailable else {})})
        by_name = {row['id']: row for row in rows + self.local.catalog(loaded_only)['data']}
        return {'object': 'list', 'data': [by_name[name] for name in self.config.models if name in by_name]}

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        failures=[]
        for w in self.workers:
            if w.cancel:
                try:w.cancel.set()
                except Exception as exc:failures.append(exc)
        for w in self.workers:
            try:w.stop()
            except Exception as exc:failures.append(exc)
        try:self.local.close()
        except Exception as exc:failures.append(exc)
        if failures:
            for error in failures[1:]:failures[0].add_note(str(error))
            raise failures[0]


def _scrub_environment():
    """A spawned replica worker inherits the server's whole environment. Before any backend import or child process, drop
    what titanos.child_env.scrubbed_env drops (provider variables, credential-shaped names, vault env: names); the worker
    keeps everything else, because backends read variables no allowlist can know (IMAGEIO_FFMPEG_EXE, for example)."""
    from .child_env import scrubbed_env
    from .native_process import vault_env_vars
    kept = scrubbed_env(dict(os.environ), extra_deny=vault_env_vars())
    for name in [name for name in os.environ if name not in kept]:
        del os.environ[name]
