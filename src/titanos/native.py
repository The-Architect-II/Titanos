"""Owned llama.cpp processes; preserve native API semantics and concurrent slots."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time

import anyio
import httpx
from fastapi.responses import Response, StreamingResponse

from .engine import EngineError
from .native_install import executable_path
from .settings import current_settings


def native_devices(executable, env):
    """Enumerate the selected native cohort without the parent's DLL search path."""
    from .process_command import module_command
    timeout = current_settings().device_probe_timeout
    output = subprocess.check_output(module_command('titanos.native_process', '--query-devices', timeout, executable),
        env=env, stdin=subprocess.DEVNULL, text=True, stderr=subprocess.STDOUT, timeout=timeout + 5)
    output = re.sub(r'\x1b\[[0-9;]*m', '', output)
    _, header, table = output.partition('Available devices:')
    if not header:
        raise ValueError('Selected native runtime did not return its bounded device inventory')
    return {name: description.strip() for name, description in re.findall(
        r'^  ([A-Za-z][A-Za-z0-9_.-]*):\s*(.*?)\s+\([0-9]+ MiB, [0-9]+ MiB free\)\s*$', table, re.M)}


def resolve_native_placement(spec, executable, env):
    """Resolve configured policy within inherited scope, before reading weights."""
    from .devices import (admit_gpu_uuids, probe_visible_cuda, require_homogeneous, scan,
                          visibility_constraints, visible_nvidia_records)
    requested, layers = spec.device, spec.native.gpu_layers
    result = {'configured_device': requested, 'configured_gpu_layers': layers,
              'inherited_visibility': visibility_constraints(env), 'verified_devices': [],
              'execution_backend': 'llama.cpp', 'kernel_execution_verified': False}
    def cpu(reason):
        if requested != 'cpu' and layers not in {0, 'auto'}:
            raise ValueError('Requested GPU offload has no admitted native accelerator; CPU fallback is disabled')
        env.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
        return {**result, 'resolved_device': 'cpu', 'resolved_gpu_layers': 0,
                'native_devices': ['none'], 'selection_reason': reason}
    if requested == 'cpu' or requested == 'auto' and layers == 0:
        return cpu('explicit_cpu_or_zero_layers')
    if requested == 'auto' and (env.get('CUDA_VISIBLE_DEVICES', None) in {'', '-1'}
            or env.get('NVIDIA_VISIBLE_DEVICES', 'all').strip().lower() in {'', 'none', 'void', '-1'}):
        return cpu('inherited_visibility_excludes_cuda')
    available = native_devices(executable, env)
    cuda_names = [name for name in available if re.fullmatch(r'CUDA[0-9]+', name)]
    if requested == 'auto' and not cuda_names and shutil.which('nvidia-smi'):
        # A CPU-only executable is not evidence that this is a CPU-only host.
        # Resolve the unchanged inherited scope; never grant masked-out devices.
        visible = visible_nvidia_records(environ=env)
        if visible:
            raise ValueError('Automatic placement found an allowed NVIDIA GPU, but the selected llama.cpp runtime exposes no CUDA device. Prepare a compatible CUDA runtime and select its executable, or explicitly choose device=cpu/gpu_layers=0.')
    if requested.startswith('GPU-') or requested.startswith('native:CUDA') or requested == 'auto' and cuda_names:
        if not cuda_names:
            raise ValueError('Selected llama.cpp runtime exposes no CUDA device; install a compatible GPU runtime')
        observed = probe_visible_cuda(library_dir=executable.parent, environ=env)
        installed = scan()
        physical = visible_nvidia_records(installed, environ=env, visible=observed, library_dir=executable.parent)
        if requested == 'auto':
            if not physical:
                return cpu('no_visible_native_accelerator')
            selectors = [first_eligible(physical)]
        elif requested.startswith('native:CUDA'):
            by_name = {'CUDA' + str(i): row['uuid'] for i, row in enumerate(observed)}
            names = requested.removeprefix('native:').split(',')
            if any(name not in by_name or name not in available for name in names):
                raise ValueError('Requested native CUDA address is not in the current masked runtime')
            selectors = [by_name[name] for name in names]
        else:
            selectors = requested.split(',')
        selected = refuse_excluded(admit_gpu_uuids(selectors, physical=installed, visible=observed, environ=env))
        rows = [next(row for row in physical if row['uuid'].lower() == identity.lower()) for identity in selected]
        if len(rows) > 1:
            result['group_signature'] = require_homogeneous(rows)
        env['CUDA_VISIBLE_DEVICES'] = ','.join(selected)
        verified = probe_visible_cuda(library_dir=executable.parent, environ=env)
        if [r.get('uuid', '').lower() for r in verified] != [identity.lower() for identity in selected]:
            raise ValueError('Native CUDA UUID/count/order differs after narrowing; assignment was not substituted')
        return {**result, 'resolved_device': ','.join(selected), 'resolved_gpu_layers': layers,
                'native_devices': ['CUDA' + str(i) for i in range(len(selected))],
                'verified_devices': verified, 'selection_reason': 'first_visible_native_gpu' if requested == 'auto' else 'explicit_gpu_selection'}
    if requested == 'auto':
        if result['inherited_visibility'] or not available:
            return cpu('no_accelerator_with_verifiable_visibility')
        names = [next(iter(available))]
    else:
        names = requested.removeprefix('native:').split(',')
        if any(name not in available for name in names):
            raise ValueError('Native device is absent from this binary --list-devices output')
        if result['inherited_visibility']:
            raise ValueError('Native non-CUDA device names cannot prove compliance with inherited accelerator masks')
        if len(names) > 1:
            raise ValueError('Native non-CUDA grouped execution needs verifiable homogeneous identities; use one device per replica')
    return {**result, 'resolved_device': 'native:' + ','.join(names), 'resolved_gpu_layers': layers,
            'native_devices': names, 'selection_reason': 'native_runtime_address',
            'verified_devices': [{'native_device': name, 'name': available[name], 'identity_kind': 'runtime_address'} for name in names]}


def native_offload(log):
    counts = re.findall(r'offloaded\s+(\d+)/(\d+)\s+layers to GPU', log)
    buffers = re.findall(r'^.*?\b([A-Za-z][A-Za-z0-9_.-]*)\s+model buffer size\s*=\s*([0-9.]+)\s+MiB', log, re.M)
    return {'gpu_layers': int(counts[-1][0]) if counts else None,
            'total_layers': int(counts[-1][1]) if counts else None,
            'model_buffer_mib': {name: float(size) for name, size in buffers},
            'evidence_source': 'native_load_log', 'kernel_execution_verified': False}


def validate_media(value):
    """Native runtimes must not fetch client-supplied URLs or local files."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'image_url', 'video_url', 'audio_url'}:
                url = item.get('url') if isinstance(item, dict) else item
                if not isinstance(url, str) or not url.startswith('data:'):
                    raise EngineError('Media must be inline data URLs; remote URLs and local file access are disabled')
            if key in {'file_id', 'file_url', 'file_data'}:
                raise EngineError('Native file inputs are not supported')
            validate_media(item)
    elif isinstance(value, list):
        for item in value:
            validate_media(item)


class NativeWorker:
    def __init__(self, name, model, spec, timeout=600):
        self.name, self.model, self.spec = name, model, spec
        self.process = self.client = self.temp = None
        self.active = self.pending = self.completed = 0
        self.error = None
        self.broken = False
        self.ready = False
        self.closed = False
        self.lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._startup_task = self._startup_cancel = None
        self.timeout=timeout
        self.started = int(time.time())
        self.placement = None; self.slot_context = None  # tokens per slot this runtime starts with; None: registered
        self.offload = None

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start_process(self, cancelled=None):
        def check_cancel():
            if cancelled is not None and cancelled.is_set():
                raise EngineError('Native startup cancelled', 499, 'cancelled')
        check_cancel()
        options = self.spec.native
        executable = executable_path(options.executable)
        from .native_process import child_environment
        # provider credentials (known variables, *_API_KEY, *_BASE_URL, vault env: references) never reach the child
        env = child_environment({k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_', 'MTMD_')) and k not in {'HF_TOKEN','HUGGING_FACE_HUB_TOKEN','TITANOS_API_KEY'}})
        self.placement = resolve_native_placement(self.spec, executable, env)
        self.verified_devices = self.placement['verified_devices']
        self.offload = None
        check_cancel()
        root = Path(self.spec.path).resolve()
        def checkpoint(name):
            path = root / name
            # Hub snapshots are symlinks to the cache; their file names stay inside the snapshot.
            if not path.is_file():
                raise ValueError(f'GGUF checkpoint not found: {name}')
            with path.open('rb') as stream:
                if stream.read(4) != b'GGUF':
                    raise ValueError(f'Invalid GGUF checkpoint: {name}')
            return str(path)
        model = checkpoint(options.model_file)
        projector = checkpoint(options.mmproj) if options.mmproj else None
        files = options.model_files or [options.model_file]
        if options.mmproj and options.mmproj not in files:
            files = [*files, options.mmproj]
        checkpoint_files = {}
        for name in files:
            path = Path(checkpoint(name))
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    check_cancel()
                    digest.update(block)
            checkpoint_files[name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        self.checkpoint_files = checkpoint_files
        self.checkpoint_digest = hashlib.sha256(json.dumps(checkpoint_files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        runtime_files = {}
        libraries = [path for path in executable.parent.iterdir() if path.is_file() and (path == executable or any(suffix.lower() in {'.dll','.so','.dylib'} for suffix in path.suffixes))]
        for path in libraries:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    check_cancel()
                    digest.update(block)
            runtime_files[path.name] = {'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
        self.runtime_files = runtime_files
        self.runtime_digest = hashlib.sha256(json.dumps(runtime_files,sort_keys=True,separators=(',', ':')).encode()).hexdigest(); admit_runtime(self)  # memory admission before any directory, key or process
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        self.temp = tempfile.TemporaryDirectory(prefix='titanos-native-')
        token = secrets.token_urlsafe(32)
        key = Path(self.temp.name, 'key')
        key.write_text(token)
        key.chmod(0o600)
        args = [str(executable), '--model', model, '--alias', self.model,
                '--host', '127.0.0.1', '--port', str(port), '--api-key-file', str(key),
                '--ctx-size', str(slot_context(self) * options.slots), '--parallel', str(options.slots),
                '--threads', str(options.threads), '--n-gpu-layers', str(self.placement['resolved_gpu_layers']),
                '--split-mode', options.split_mode, '--jinja', '--cont-batching', '--metrics',
                '--no-context-shift', '--fit', 'off', *attention_arguments(options), *reasoning_arguments(options),
                *embedding_arguments(self.spec)]
        if self.placement['resolved_device'] != 'cpu':
            # b10809 maps library INFO (including offloaded layers) to TRACE.
            # Keep the positive-offload check observable; health alone is not proof.
            args += ['--log-verbosity', '4']
        if options.batch_size is not None:
            args += ['--batch-size', str(options.batch_size)]
        if options.ubatch_size is not None:
            args += ['--ubatch-size', str(options.ubatch_size)]
        args += ['--device', ','.join(self.placement['native_devices'])]
        if options.tensor_split:
            args += ['--tensor-split', ','.join(map(str, options.tensor_split))]
        if projector:
            args += ['--mmproj', projector]
        self.log = Path(self.temp.name, 'runtime.log')
        from .process_command import module_command
        pid_path=Path(self.temp.name,'runtime.pid')
        check_cancel()
        with self.log.open('wb') as output:
            self.process = subprocess.Popen(module_command('titanos.native_process', pid_path, *args), env=env, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT)
        self.url, self.token = f'http://127.0.0.1:{port}', token
        self.binary_sha256 = next(identity['sha256'] for name,identity in runtime_files.items() if executable.parent / name == executable)
        deadline = time.monotonic() + options.startup_timeout
        with httpx.Client(trust_env=False, timeout=2, headers={'Authorization':'Bearer '+token}) as client:
            while time.monotonic() < deadline:
                check_cancel()
                if not self.running:
                    raise ValueError('Native runtime exited during model loading: '+self.log.read_text(errors='replace')[-2500:])
                try:
                    if client.get(self.url+'/health').status_code == 200:
                        self.offload = native_offload(self.log.read_text(errors='replace'))
                        if self.placement['resolved_device'] != 'cpu' and not (self.offload['gpu_layers'] or 0) > 0:
                            raise ValueError('Native runtime did not report GPU layer offload; explicit accelerator execution was not accepted')
                        self.native_pid=int(pid_path.read_text())
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
        raise ValueError('Native model loading exceeded startup_timeout')

    async def start(self):
        if self.closed:
            raise EngineError('Native worker is closed',503,'server_unavailable')
        if self.running and self.ready and not self.broken:
            return
        await self.stop()
        if self.closed:
            raise EngineError('Native worker is closed',503,'server_unavailable')
        startup = None
        try:
            cancelled = threading.Event()
            startup=asyncio.create_task(asyncio.to_thread(self.start_process, cancelled))
            self._startup_cancel, self._startup_task = cancelled, startup
            try:
                await asyncio.shield(startup)
            except asyncio.CancelledError:
                cancelled.set()
                await self.stop()
                raise
            if self.closed or cancelled.is_set():
                raise EngineError('Native worker closed during startup',503,'server_unavailable')
            self.client = httpx.AsyncClient(base_url=self.url, headers={'Authorization':'Bearer '+self.token}, trust_env=False,
                                           timeout=httpx.Timeout(self.timeout, connect=5), limits=httpx.Limits(max_connections=self.spec.native.slots+4))
            self.error = None
            self.broken = False
            self.ready = True
        except (asyncio.CancelledError, AdmissionRefused):  # a refusal started nothing
            raise
        except Exception as exc:
            self.error = str(exc)
            await self.stop()
            if self.closed:
                raise EngineError('Native worker closed during startup',503,'server_unavailable') from exc
            raise EngineError('Native model load failed: '+str(exc), 503, 'model_load_failed') from exc
        finally:
            if self._startup_task is startup and startup is not None and startup.done():
                self._startup_task = self._startup_cancel = None

    async def stop(self):
        # This also joins a hashing/startup thread which has not created a process
        # yet. Serial cleanup prevents close and a cancelled start racing handles.
        with anyio.CancelScope(shield=True):
            async with self._stop_lock:
                self.ready=False
                startup = self._startup_task
                if self._startup_cancel is not None:
                    self._startup_cancel.set()
                if startup is not None:
                    await asyncio.shield(asyncio.gather(startup,return_exceptions=True))
                    if self._startup_task is startup:
                        self._startup_task = self._startup_cancel = None
                if self.process is not None:
                    if self.running:
                        self.process.stdin.close()
                        try:
                            await asyncio.to_thread(self.process.wait, 15)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            await asyncio.to_thread(self.process.wait)
                    self.process = None
                if self.client:
                    await self.client.aclose()
                    self.client = None
                if self.temp:
                    self.temp.cleanup()
                    self.temp = None

    async def close(self):
        self.closed = True
        self.ready = False
        await self.stop()

    def status(self):
        return {'id':self.name,'model':self.model,'pid':getattr(self,'native_pid',None) if self.running else None,
                'running':self.running,'active_requests':self.active,'pending_requests':self.pending,
                'loaded':self.running and self.ready and not self.broken and not self.closed,
                'slots':self.spec.native.slots,'slot_context':slot_context(self),'completed_requests':self.completed,'last_error':self.error,
                'backend':'llama-cpp','device':self.placement['resolved_device'] if self.running and self.placement else self.spec.device,
                'configured_device':self.spec.device,'resolved_device':self.placement['resolved_device'] if self.running and self.placement else None,
                'placement':self.placement, 'offload':self.offload if self.running else None,
                'split_mode':self.spec.native.split_mode, 'execution_backend':'llama.cpp',
                'binary_sha256':getattr(self,'binary_sha256',None),
                'checkpoint_digest':getattr(self,'checkpoint_digest',None),
                'model_files':getattr(self,'checkpoint_files',{}),
                'runtime_files':getattr(self,'runtime_files',{}),
                'runtime_digest':getattr(self,'runtime_digest',None),
                'verified_devices':getattr(self,'verified_devices',None),'admission':getattr(self,'admission',None)}

    async def observed_status(self):
        result=self.status()
        if self.running and self.client:
            try:
                response=await self.client.get('/slots',timeout=2)
                response.raise_for_status()
                result['native_slots']=[{k:v for k,v in row.items() if k in {'id','is_processing','n_ctx','id_task'}} for row in response.json()]
            except (httpx.HTTPError,ValueError):
                result['native_slots']=None
        return result


class NativePool:
    def __init__(self, config, settings=None):
        settings = settings if settings is not None else current_settings()
        self.queue_capacity = settings.native_queue_capacity
        self.queue_timeout = settings.native_queue_timeout
        self.models = {n:s for n,s in config.models.items() if s.backend == 'llama-cpp'}
        self.workers = []
        for name, spec in self.models.items():
            assigned = [w for w,a in config.workers.items() if name in a.models] or ['native:'+name]
            for worker in assigned:
                assignment=config.workers.get(worker)
                effective=spec.model_copy(deep=True)
                if assignment:
                    from .config import native_assignment
                    effective=native_assignment(spec,assignment)
                self.workers.append(NativeWorker(worker,name,effective,assignment.timeout if assignment else settings.worker_timeout))
        self.condition = asyncio.Condition()
        self.waiting = 0
        self.closed = False

    def candidates(self, model, worker=None):
        found = [w for w in self.workers if w.model == model and (worker is None or w.name == worker)]
        if not found:
            raise EngineError('No matching native model/worker', 404, 'model_not_found')
        return found

    def _require_loaded(self, worker, model):
        if self.closed or worker.closed:
            raise EngineError('Native worker is closed',503,'server_unavailable')
        if worker.broken:
            raise EngineError('Native worker is unavailable; request was not replayed',503,'worker_unavailable')
        if worker.model != model or worker.process is None or not worker.ready or worker.client is None:
            raise EngineError('Requested model is not loaded',409,'model_not_loaded')
        try:
            running = worker.running
        except Exception as exc:
            raise EngineError('Native worker liveness is unavailable',503,'worker_unavailable') from exc
        if not running or worker.client.is_closed:
            raise EngineError('Native worker is unavailable; request was not replayed',503,'worker_unavailable')

    @asynccontextmanager
    async def lease(self, model, worker=None, *, require_loaded=False, num_ctx=None):
        if type(require_loaded) is not bool:
            raise EngineError('require_loaded must be a boolean')
        selected = None
        async with self.condition:
            if self.closed:
                raise EngineError('Native pool is closed',503,'server_unavailable')
            if self.waiting >= self.queue_capacity:
                raise EngineError('Native queue unavailable or full', 503, 'queue_full')
            self.waiting += 1
            try:
                async with asyncio.timeout(self.queue_timeout):
                    while selected is None:
                        if self.closed:
                            raise EngineError('Server is shutting down',503,'server_unavailable')
                        candidates = self.candidates(model, worker)
                        if require_loaded:
                            usable, failures = [], []
                            for current in candidates:
                                try:self._require_loaded(current, model)
                                except EngineError as exc:failures.append(exc)
                                else:usable.append(current)
                            if not usable:
                                raise next((exc for exc in failures if exc.code != 'model_not_loaded'), failures[0])
                            candidates = usable
                        choices = [w for w in candidates if w.active+w.pending < w.spec.native.slots and context_fits(w, model, num_ctx, require_loaded)]
                        if choices:
                            selected = min(choices, key=lambda w:(w.active+w.pending,w.completed))
                            selected.pending += 1
                        else:
                            await self.condition.wait()
            except TimeoutError as exc:
                raise EngineError('Native queue timeout', 503, 'queue_timeout') from exc
            finally:
                self.waiting -= 1
        acquired = False
        try:
            async with selected.lock:
                if self.closed:
                    raise EngineError('Native pool closed before startup',503,'server_unavailable')
                if not require_loaded and (not selected.running or selected.broken) and selected.active:
                    raise EngineError('Native worker died with requests in flight; request was not replayed',503,'worker_unavailable')
                if not require_loaded:
                    await start_sized(self, selected, model, num_ctx)
                if self.closed:
                    if not require_loaded:await selected.stop()
                    raise EngineError('Native pool closed during startup',503,'server_unavailable')
                async with self.condition:
                    if require_loaded:
                        # Unload owns selected.lock; pool close owns condition.
                        # No await separates this final check from reservation.
                        self._require_loaded(selected, model)
                    selected.pending -= 1
                    selected.active += 1
                    acquired = True
            yield selected
        finally:
            async with self.condition:
                if acquired:
                    selected.active -= 1
                else:
                    selected.pending -= 1
                self.condition.notify_all()

    async def load(self, model, worker=None):
        for selected in self.candidates(model, worker):
            async with self.lease(model, selected.name):
                pass
        return {'model':model,'status':'loaded'}

    async def unload(self, model, worker=None):
        for selected in self.candidates(model, worker):
            async with selected.lock:
                if selected.active or selected.pending:
                    raise EngineError('Native worker has active requests; retry unload when idle',409,'model_busy')
                await selected.stop(); selected.slot_context = None
        return {'model':model,'status':'unloaded'}

    async def close(self):
        async with self.condition:
            self.closed = True
            self.condition.notify_all()
        results=await asyncio.gather(*(w.close() for w in self.workers),return_exceptions=True)
        failures=[value for value in results if isinstance(value,BaseException)]
        if failures:
            for extra in failures[1:]:failures[0].add_note('Additional native worker close failure: '+repr(extra)+''.join('\n'+note for note in getattr(extra,'__notes__',())))
            raise failures[0]

    def catalog(self, loaded_only=False):
        result=[]
        for name,spec in self.models.items():
            workers=self.candidates(name)
            state='loaded' if any(w.running and w.ready and not w.broken for w in workers) else 'loading' if any(w.pending for w in workers) else 'error' if any(w.error for w in workers) else 'unloaded'
            if not loaded_only or state=='loaded':
                result.append({'id':name,'object':'model','created':workers[0].started,'owned_by':'local','task':spec.task,'status':state,
                               'capabilities':(['embeddings'] if spec.task=='embeddings' else ['chat','tools','json_schema','responses','continuous_batching'])+(['vision'] if spec.task=='vision' else [])})
        return result

    async def details(self, model, worker=None):
        async with self.lease(model,worker) as selected:
            result=selected.status()
            response=await selected.client.get('/props')
            response.raise_for_status()
            props=response.json()
            result['runtime']={k:props[k] for k in ('total_slots','default_generation_settings','chat_template_caps','modalities') if k in props}
            response=await selected.client.get('/slots')
            if response.is_success:
                result['slots_state']=[{k:v for k,v in row.items() if k in {'id','is_processing','n_ctx','n_decoded','n_remaining','id_task'}} for row in response.json()]
            return result

    async def complete(self, data, *, deadline=None, cancel=None, context_max_tokens=None, require_loaded=False):
        """Nonstreaming chat over an owned worker; queue/start/read share one budget.

        This calls the worker client directly, never Titanos' public HTTP route.
        The complete native response is retained alongside normalized text/provenance.
        """
        from .server import Chat
        from .request_defaults import model_defaults
        from .settings import current_settings
        deadline = deadline if deadline is not None else time.monotonic() + current_settings().client_timeout
        if type(require_loaded) is not bool:
            raise EngineError('require_loaded must be a boolean')
        try:
            payload = Chat.model_validate(data)
            spec = self.models.get(payload.model)
            if spec is None:
                raise EngineError('Unknown native model', 404, 'model_not_found')
            payload = model_defaults(payload, spec)
        except ValueError as exc:
            raise EngineError('Invalid native completion request: ' + str(exc)) from exc
        if payload.stream:
            raise EngineError('Native complete is nonstreaming; use the streaming gateway')
        if context_max_tokens is not None and (type(context_max_tokens) is not int or context_max_tokens<1):
            raise EngineError('context_max_tokens must be a positive integer')
        values = payload.model_dump(exclude_none=True)
        requested_loaded = values.pop('require_loaded', False)
        if type(requested_loaded) is not bool:
            raise EngineError('require_loaded must be a boolean')
        require_loaded = require_loaded or requested_loaded
        if values.get('max_completion_tokens') is not None:
            values['max_tokens'] = values.pop('max_completion_tokens')
        worker = values.pop('worker', None)
        validate_media(values)
        self.candidates(payload.model, worker)  # Reject an invalid assignment before queue/start.

        async def request():
            context = self.lease(payload.model, worker, require_loaded=True) if require_loaded else self.lease(payload.model, worker)
            async with context as selected:
                response = None
                try:
                    accounting = None
                    if context_max_tokens is not None:
                        # This leased model/template/tokenizer on the generation body (tools too),
                        # never a character estimate or a second public Titanos request.
                        async def preparation(route, body=None):
                            prepared = await (selected.client.get(route) if body is None else selected.client.post(route,json=body))
                            try:
                                if not prepared.is_success:
                                    raise EngineError('Native context accounting is unavailable',502,'context_accounting_unavailable')
                                return prepared.json()
                            except ValueError as exc:
                                raise EngineError('Invalid native context accounting response',502,'context_accounting_unavailable') from exc
                            finally:
                                await prepared.aclose()
                        try:
                            prompt = (await preparation('/apply-template',values))['prompt']
                            if not isinstance(prompt,str):raise ValueError('Template is not text')
                            tokens = (await preparation('/tokenize',{'content':prompt,'add_special':True,'parse_special':True,'with_pieces':False}))['tokens']
                            capacity = (await preparation('/props'))['default_generation_settings']['n_ctx']
                            if not isinstance(tokens,list) or not tokens or any(type(token) is not int or token<0 for token in tokens) or type(capacity) is not int or capacity<1:
                                raise ValueError('Invalid context tokens/capacity')
                        except (ValueError,TypeError,KeyError) as exc:
                            raise EngineError('Native context accounting is unavailable',502,'context_accounting_unavailable') from exc
                        if len(tokens)>context_max_tokens or len(tokens)+values['max_tokens']>capacity:
                            raise EngineError('Full retrieved context and output reservation exceed the native context limit; no input was clipped',400,'context_limit')
                        accounting={'method':'native_template_and_tokenizer','prompt_sha256':hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                            'input_tokens':len(tokens),'requested_input_limit':context_max_tokens,'model_context_tokens':capacity,
                            'output_reservation':values['max_tokens'],'verified_against_usage':False}
                    response = await selected.client.send(selected.client.build_request('POST','/v1/chat/completions',json=values),stream=True)
                    await response.aread()
                    if not response.is_success:
                        raise EngineError('Native completion rejected the request: ' + response.text[:1000], response.status_code, 'native_request_failed')
                    try:
                        body = response.json()
                        choices = body['choices']
                        if not isinstance(choices, list) or len(choices) != 1:
                            raise ValueError('Expected one completion choice')
                        choice = choices[0]
                        text = choice['message'].get('content')
                        finish = choice.get('finish_reason')
                    except (ValueError, TypeError, KeyError, AttributeError) as exc:
                        raise EngineError('Invalid native completion response', 502, 'invalid_backend_response') from exc
                    if accounting is not None:
                        actual = body.get('usage',{}).get('prompt_tokens')
                        if type(actual) is not int or actual!=accounting['input_tokens']:
                            raise EngineError('Native prompt accounting differs from preparation; exact context use is not established',502,'context_accounting_mismatch')
                        accounting['verified_against_usage']=True
                    selected.completed += 1
                    result = {'text':text, 'finish_reason':finish, 'usage':body.get('usage', {}), 'native_response':body,
                        'provenance':{'repository':selected.spec.source_repo, 'revision':selected.spec.source_revision,
                            'backend':'llama-cpp', 'device':(getattr(selected, 'placement', None) or {}).get('resolved_device', selected.spec.device),
                            'checkpoint_digest':getattr(selected,'checkpoint_digest',None)},
                        'native_runtime':selected.status()}
                    if accounting is not None:
                        result['context_accounting']=accounting
                    return result
                except httpx.HTTPError as exc:
                    selected.broken = True
                    selected.error = str(exc)
                    raise EngineError('Native completion failed; request was not replayed',503,'worker_unavailable') from exc
                finally:
                    if response is not None:
                        await response.aclose()

        task = asyncio.create_task(request())
        try:
            while True:
                if self.closed:
                    raise EngineError('Native pool is closed',503,'server_unavailable')
                if cancel is not None and cancel.is_set():
                    raise EngineError('Caller cancelled native completion',499,'cancelled')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EngineError('Native completion exceeded its shared deadline',504,'deadline_exceeded')
                if task.done():
                    return await task
                await asyncio.wait({task},timeout=min(.05,remaining))
        except BaseException:
            task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.shield(asyncio.gather(task,return_exceptions=True))
            raise

    async def count_tokens(self, model, messages, *, tools=None, deadline=None):
        """Input tokens of a chat request on the leased worker's own template and tokenizer: the same /apply-template
        then /tokenize sequence as complete()'s exact context accounting, never a character estimate. messages are
        OpenAI chat messages; tools (OpenAI function tools) are passed to the template when given."""
        from .settings import current_settings
        if model not in self.models:
            raise EngineError('Unknown native model', 404, 'model_not_found')
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise EngineError('messages must be a nonempty list of message objects')
        deadline = deadline if deadline is not None else time.monotonic() + current_settings().client_timeout
        body = {'messages': messages}
        if tools:
            body['tools'] = tools

        async def count():
            async with self.lease(model) as selected:
                async def call(route, payload):
                    response = await selected.client.post(route, json=payload)
                    try:
                        if not response.is_success:
                            raise EngineError('Native token counting is unavailable', 502, 'invalid_backend_response')
                        return response.json()
                    except ValueError as exc:
                        raise EngineError('Invalid native token counting response', 502, 'invalid_backend_response') from exc
                    finally:
                        await response.aclose()
                try:
                    prompt = (await call('/apply-template', body))['prompt']
                    if not isinstance(prompt, str):
                        raise ValueError('Template is not text')
                    tokens = (await call('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True,
                                                       'with_pieces': False}))['tokens']
                    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
                        raise ValueError('Invalid tokens')
                except (ValueError, TypeError, KeyError) as exc:
                    raise EngineError('Native token counting is unavailable', 502, 'invalid_backend_response') from exc
                except httpx.HTTPError as exc:
                    raise EngineError('Native token counting failed', 503, 'worker_unavailable') from exc
                return len(tokens)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EngineError('Native token counting exceeded its deadline', 504, 'deadline_exceeded')
        try:
            return await asyncio.wait_for(count(), timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise EngineError('Native token counting exceeded its deadline', 504, 'deadline_exceeded') from exc

    async def proxy(self, request, route, data, *, require_loaded=False, num_ctx=None):
        data=dict(data)
        data.pop('require_loaded', None)
        if type(require_loaded) is not bool:
            raise EngineError('require_loaded must be a boolean')
        if require_loaded:
            return await self._loaded_proxy(request, route, data, num_ctx)
        worker=data.pop('worker',None)
        validate_media(data)
        context=self.lease(data['model'],worker,num_ctx=num_ctx)
        selected=await context.__aenter__()
        response=None
        try:
            response=await selected.client.send(selected.client.build_request('POST',route,json=data),stream=True)
            if data.get('stream') and response.is_success:
                async def stream():
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                        selected.completed += 1
                    except httpx.HTTPError as exc:
                        selected.broken=True
                        selected.error=str(exc)
                        raise
                    finally:
                        await response.aclose()
                        await context.__aexit__(None,None,None)
                return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
            body=asyncio.create_task(response.aread())
            try:
                while not body.done():
                    if await request.is_disconnected():
                        body.cancel()
                        raise EngineError('Request cancelled',499,'cancelled')
                    await asyncio.wait({body},timeout=0.1)
                content=await body
                if response.is_success:
                    selected.completed += 1
                return Response(content,status_code=response.status_code,media_type=response.headers.get('content-type','application/json'))
            finally:
                if not body.done():
                    body.cancel()
                await asyncio.gather(body,return_exceptions=True)
        except httpx.HTTPError as exc:
            selected.error=str(exc)
            selected.broken=True
            raise EngineError('Native request failed; request was not replayed',503,'worker_unavailable') from exc
        finally:
            if response is None or not(data.get('stream') and response.is_success):
                if response is not None:
                    await response.aclose()
                await context.__aexit__(None,None,None)

    async def _loaded_proxy(self, request, route, data, num_ctx=None):
        """The request Guard, not response construction, ends this strict lease."""
        worker=data.pop('worker',None)
        validate_media(data)
        context=self.lease(data['model'],worker,require_loaded=True,num_ctx=num_ctx)
        operation=response=selected=cleanup_task=None
        entered=False
        primary=None
        admitted,proceed=asyncio.Event(),asyncio.Event()

        async def finish():
            failures=[]
            if operation is not None:
                if not operation.done():operation.cancel()
                # This is the actual async operation, never a cancelled thread wrapper.
                await asyncio.gather(operation,return_exceptions=True)
            try:
                if response is not None:
                    try:await response.aclose()
                    except BaseException as exc:failures.append(exc)
            finally:
                if entered:
                    try:await context.__aexit__(None,None,None)
                    except BaseException as exc:failures.append(exc)
            if failures:
                error=primary if primary is not None else failures.pop(0)
                for extra in failures:error.add_note('Additional native response/lease cleanup failure: '+repr(extra))
                raise error

        async def cleanup():
            nonlocal cleanup_task
            if cleanup_task is None:cleanup_task=asyncio.create_task(finish())
            cancelled=None
            while not cleanup_task.done():
                try:await asyncio.shield(cleanup_task)
                except asyncio.CancelledError as exc:cancelled=cancelled or exc
                except BaseException:break
            if cancelled is not None:
                try:cleanup_task.result()
                except BaseException as exc:cancelled.add_note('Native cleanup also failed: '+repr(exc))
                raise cancelled
            return cleanup_task.result()

        # Registered before acquisition so failed/never-entered responses retain custody.
        request.scope.setdefault('titanos.cleanup',[]).append(cleanup)

        async def dispatch():
            nonlocal entered,response,selected
            selected=await context.__aenter__();entered=True
            client=selected.client
            admitted.set()
            await proceed.wait()
            self._require_loaded(selected,data['model'])
            try:
                response=await client.send(client.build_request('POST',route,json=data),stream=True)
                if data.get('stream') and response.is_success:
                    async def stream():
                        nonlocal primary
                        try:
                            async for chunk in response.aiter_bytes():yield chunk
                            selected.completed+=1
                        except httpx.HTTPError as exc:
                            selected.broken=True;selected.error=str(exc);primary=exc;raise
                        except BaseException as exc:
                            primary=exc;raise
                        # Guard cleanup closes the response and lease after ASGI unwind.
                    return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})
                content=await response.aread()
                if response.is_success:selected.completed+=1
                return Response(content,status_code=response.status_code,media_type=response.headers.get('content-type','application/json'))
            except httpx.HTTPError as exc:
                selected.error=str(exc);selected.broken=True
                raise EngineError('Native request failed; request was not replayed',503,'worker_unavailable') from exc

        operation=asyncio.create_task(dispatch())
        try:
            while not admitted.is_set() and not operation.done():
                if await request.is_disconnected():
                    operation.cancel()
                    raise EngineError('Request cancelled',499,'cancelled')
                await asyncio.wait({operation},timeout=.05)
            # One receive owner: check again after queued admission, before send.
            if not operation.done():
                if await request.is_disconnected():
                    raise EngineError('Request cancelled',499,'cancelled')
                proceed.set()
            while not operation.done():
                if await request.is_disconnected():
                    operation.cancel()
                    raise EngineError('Request cancelled',499,'cancelled')
                await asyncio.wait({operation},timeout=.05)
            return await asyncio.shield(operation)
        except BaseException as exc:
            primary=exc
            if not operation.done():operation.cancel()
            raise


def attention_arguments(options):
    """llama-server attention and KV-cache arguments from NativeOptions: --flash-attn only when it departs from the
    runtime's auto default, --cache-type-k/--cache-type-v only when the registration names them, so a registration that
    sets none of them keeps its argv byte for byte. Owned by NativeWorker.start_process; a separate function so the argv
    contract is checkable without a process."""
    args = []
    if options.flash_attn != 'auto':
        args += ['--flash-attn', options.flash_attn]
    if options.cache_type_k:
        args += ['--cache-type-k', options.cache_type_k]
    if options.cache_type_v:
        args += ['--cache-type-v', options.cache_type_v]
    return args


def reasoning_arguments(options):
    """llama-server --reasoning and --reasoning-budget from NativeOptions.reasoning and reasoning_budget, each only when
    the registration names it, so a registration without them keeps its argv byte for byte (the runtime's own default
    applies). --reasoning off alone does not stop a model whose template ignores it (Nemotron 3.5 Bolt E4B still thinks
    under b11402); a budget of 0 ends the thinking at once. Owned by NativeWorker.start_process."""
    args = ['--reasoning', options.reasoning] if options.reasoning else []
    if options.reasoning_budget is not None:
        args += ['--reasoning-budget', str(options.reasoning_budget)]
    return args


def embedding_arguments(spec):
    """llama-server arguments for a GGUF embedding model (ModelSpec task embeddings): --embeddings, plus --pooling when the
    registration names one (native.pooling); the GGUF's own pooling type applies otherwise. Chat and vision models add nothing.
    Owned by NativeWorker.start_process; a separate function so the argv contract is checkable without a process."""
    if spec.task != 'embeddings':
        return []
    pooling = getattr(spec.native, 'pooling', None)
    return ['--embeddings', *(['--pooling', pooling] if pooling else [])]


def slot_context(worker):
    """Tokens per slot of worker's runtime: the size it was last started or resized to, else the registered one."""
    from .native_context import registered_slot_context
    return worker.slot_context or registered_slot_context(worker.spec.native)


def context_fits(worker, model, num_ctx, require_loaded=False):
    """NativePool.lease's selection test for options.num_ctx: true without a larger need, for a runtime already that large,
    for a stopped worker (it starts at the needed size) and for an idle running one (it restarts larger). A busy runtime
    that is too small is skipped until its requests finish. Raises the 400 context_limit above the model's largest
    per-slot context, and under require_loaded (which never starts or restarts) for a loaded runtime that is too small."""
    from .native_context import needed_slot_context
    need = needed_slot_context(worker.spec, num_ctx, model)
    if need is None or slot_context(worker) >= need or not worker.running:
        return True
    if require_loaded:
        raise EngineError(f'The loaded runtime of {model} has {slot_context(worker)} tokens per slot, below options.num_ctx '
                          f'{num_ctx}; send the request without require_loaded so the runtime restarts larger', 400, 'context_limit')
    return worker.active + worker.pending == 0


async def start_sized(pool, worker, model, num_ctx):
    """NativePool.lease's start under worker.lock. When num_ctx needs more tokens per slot than worker's runtime has, wait
    (within the pool's queue timeout) for its in-flight requests to finish, stop it and start it at the larger size; a
    failed start keeps the previous size for the next one. Otherwise start() as before (a no-op for a ready runtime)."""
    from .native_context import needed_slot_context
    need = needed_slot_context(worker.spec, num_ctx, model)
    if need is None or slot_context(worker) >= need:
        return await worker.start()
    if worker.running:
        try:
            async with asyncio.timeout(pool.queue_timeout):
                async with pool.condition:
                    await pool.condition.wait_for(lambda: worker.active == 0 or pool.closed)
        except TimeoutError as exc:
            raise EngineError('Native context resize waited too long for in-flight requests', 503, 'queue_timeout') from exc
        if pool.closed:
            raise EngineError('Native pool closed during a context resize', 503, 'server_unavailable')
        # admitted against free memory plus what the stopped runtime releases, before anything is stopped
        held = ((getattr(worker, 'admission', None) or {}).get('estimate') or {}).get('total', 0)
        await asyncio.to_thread(admit_runtime, worker, need, held)
        import logging
        logging.info('native context resize: %s from %s to %s tokens per slot (options.num_ctx %s)', worker.name,
                     slot_context(worker), need, num_ctx)
        await worker.stop()
    previous, worker.slot_context = worker.slot_context, need
    try:
        await worker.start()
    except BaseException:
        worker.slot_context = previous
        raise


from .admission import AdmissionRefused  # noqa: E402  (NativeWorker.start lets a refusal through unchanged)


def runtime_files(worker):
    """The GGUF files worker's runtime loads: the model file first (its header carries the facts), then the other shards
    and the projector."""
    options = worker.spec.native
    root = Path(worker.spec.path).resolve()
    names = [options.model_file, *[name for name in options.model_files if name != options.model_file]]
    if options.mmproj and options.mmproj not in names:
        names.append(options.mmproj)
    return [root / name for name in names]


def admit_runtime(worker, slot_context_value=None, released_bytes=0):
    """titanos.admission.admit for worker's resolved placement at its (or the given) per-slot context; the decision is kept
    on worker.admission and reported by status(). Raises AdmissionRefused (503 insufficient_device_memory)."""
    from .admission import admit
    decision = admit(worker.spec, worker.placement, files=runtime_files(worker), settings=current_settings(),
                     slot_context=slot_context_value or worker.slot_context, released_bytes=released_bytes)
    worker.admission = decision
    return decision


def excluded_gpus():
    """Lower-case GPU UUIDs TITANOS_EXCLUDE_GPU_UUIDS removes from native placement."""
    return {item.strip().lower() for item in current_settings().exclude_gpu_uuids.split(',') if item.strip()}


def first_eligible(rows):
    """Automatic placement: the first visible NVIDIA GPU that is not excluded; ValueError when every one is."""
    excluded = excluded_gpus()
    for row in rows:
        if row['uuid'].lower() not in excluded:
            return row['uuid']
    raise ValueError('Every visible NVIDIA GPU is excluded by TITANOS_EXCLUDE_GPU_UUIDS; choose device=cpu or allow a GPU')


def refuse_excluded(selected):
    """Explicit or automatic placement after admission: ValueError when it names an excluded GPU (fail closed)."""
    hits = [identity for identity in selected if identity.lower() in excluded_gpus()]
    if hits:
        raise ValueError('Native placement names a GPU excluded by TITANOS_EXCLUDE_GPU_UUIDS: ' + ', '.join(hits))
    return selected
