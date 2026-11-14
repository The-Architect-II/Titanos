"""Native Transformers tensor parallelism: one cooperating process per rank."""

import multiprocessing as mp
import os
import secrets
import tempfile
import time
from contextlib import nullcontext
from datetime import timedelta
from multiprocessing.connection import wait
from pathlib import Path

from .config import ModelSpec


def distributed_backend(torch, device, size):
    """Preflight availability before allocating a rendezvous or spawning ranks."""
    kind = device.split(':')[0]
    backend = {'cpu': 'gloo', 'cuda': 'nccl', 'xpu': 'xccl'}.get(kind)
    dist = torch.distributed
    if not backend or not dist.is_available() or not dist.is_backend_available(backend):
        hint = ' Standard Windows PyTorch does not provide NCCL; native llama.cpp tensor splitting is a separate backend.' if kind == 'cuda' else ''
        raise ValueError(f'{backend or kind}_unavailable: distributed tensor parallelism requires an installed {backend or kind} backend.' + hint)
    if not 2 <= size <= 16:
        raise ValueError('Distributed tensor parallelism requires 2–16 ranks')
    if kind == 'cuda':
        if torch.cuda.device_count() != size:
            raise ValueError('Tensor parallel rank count must equal the visible assigned GPU count')
        from .backends import require_cuda_group
        require_cuda_group(torch)
    if kind == 'xpu':
        if torch.xpu.device_count() != size:
            raise ValueError('Tensor parallel rank count must equal the visible assigned XPU count')
        from .devices import normalized_uuid, require_homogeneous
        rows = []
        for i in range(size):
            props = torch.xpu.get_device_properties(i)
            identity = normalized_uuid(getattr(props, 'uuid', None))
            if not identity:
                raise ValueError('XPU tensor parallelism requires observed stable UUIDs')
            rows.append({'uuid': 'XPU-' + identity, 'runtime_vendor': 'xpu',
                         'architecture': getattr(props, 'architecture', None)})
        require_homogeneous(rows)
    return backend


def validate_plan(config, size):
    configs = (config, getattr(config, 'text_config', None), getattr(config, 'language_config', None))
    cfg = next((c for c in configs if getattr(c, 'base_model_tp_plan', None)), None)
    if cfg is None:
        raise ValueError('This architecture has no native Transformers tensor parallel plan; choose single or model_sharding explicitly')
    plan = cfg.base_model_tp_plan
    # Plain column-sharded attention must retain whole heads on every rank.
    # Other native strategies have their own loader/runtime constraints.
    for field, projection in [('num_attention_heads', 'q_proj'), ('num_key_value_heads', 'k_proj')]:
        count = getattr(cfg, field, None)
        if count and any(key.endswith(projection) and strategy == 'colwise' for key, strategy in plan.items()) and count % size:
            raise ValueError(f'{field}={count} is incompatible with {size} tensor-parallel ranks for this native plan')
    return plan


def rank_main(connection, cancel, values, rank, size, backend, rendezvous, profile):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(size))
    import torch
    import torch.distributed as dist
    from .backends import LoadedModel
    spec = ModelSpec.model_validate(values)
    kind = spec.device.split(':')[0]
    if kind != 'cpu':
        getattr(torch, kind).set_device(rank)
        spec.device = f'{kind}:{rank}'
    try:
        dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=size,
                                timeout=timedelta(seconds=120))
        model = LoadedModel(spec)
        if not getattr(model.model, '_tp_plan', None):
            raise ValueError('The loaded architecture has no native tensor parallel plan')
        meta = {'rank': rank, 'pid': os.getpid(), **model.details(), 'completed_requests': 0}
        connection.send(('ready', meta))
        while True:
            task, data = connection.recv()
            if task == 'close':
                return
            profiler = None
            if profile:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if kind == 'cuda':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                profiler = torch.profiler.profile(activities=activities)
            started = time.monotonic()
            # Every rank uses a streamer and the same stopping criteria. Only
            # rank zero emits text; stopping decisions are broadcast per token.
            def emit(text):
                if rank == 0 and not cancel.is_set():
                    connection.send(('text', text))
            with profiler if profiler else nullcontext():
                result = model.infer(task, data, emit, cancel)
            meta['completed_requests'] += 1
            meta['last_request_seconds'] = time.monotonic() - started
            meta['observed_gpu_indices'] = sorted({e.device_index for e in profiler.events()
                if e.device_type == torch.autograd.DeviceType.CUDA}) if profiler else None
            connection.send(('done', (result if rank == 0 else None, meta)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(('error', f'Rank {rank}: {type(exc).__name__}: {exc}'))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


class TensorParallelModel:
    def __init__(self, spec):
        import torch
        import torch.distributed as dist
        from transformers import AutoConfig
        from .hardware import select_device
        self.device = select_device(spec.device)
        kind = self.device.split(':')[0]
        self.size = spec.tensor_parallel_size
        backend = distributed_backend(torch, self.device, self.size)
        self.distributed_backend = backend
        cfg = AutoConfig.from_pretrained(spec.path, local_files_only=True, trust_remote_code=spec.trust_remote_code)
        validate_plan(cfg, self.size)
        self.processes, self.connections, self.ranks = [], [], {}
        self.directory = tempfile.TemporaryDirectory(prefix='titanos-tp-')
        ctx = mp.get_context('spawn')
        self.cancel = ctx.Event()
        values = spec.model_dump()
        values['device'] = self.device
        try:
            for rank in range(self.size):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=rank_main, args=(child, self.cancel, values, rank, self.size,
                    backend, (Path(self.directory.name) / 'rendezvous').as_uri(),
                    os.environ.get('TITANOS_PROFILE_INFERENCE') == '1'), name=f'titanos-tp-{rank}')
                proc.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(proc)
            self.collect('ready')
        except BaseException:
            self.close()
            raise

    def collect(self, expected, emit=None, cancel=None):
        pending = set(self.connections)
        deadline = time.monotonic() + 600
        result = None
        while pending:
            if cancel and cancel.is_set():
                self.cancel.set()
            if time.monotonic() > deadline:
                raise TimeoutError('Tensor parallel rank group timed out')
            for connection in wait(pending, timeout=0.1):
                rank = self.connections.index(connection)
                try:
                    kind, value = connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f'Tensor parallel rank {rank} exited; request was not replayed') from exc
                if kind == 'error':
                    raise RuntimeError(value)
                if kind == 'text':
                    if emit and not (cancel and cancel.is_set()):
                        emit(value)
                elif kind == expected:
                    if kind == 'ready':
                        self.ranks[rank] = value
                    else:
                        output, self.ranks[rank] = value
                        if rank == 0:
                            result = output
                    pending.remove(connection)
                else:
                    raise RuntimeError(f'Unexpected tensor parallel response: {kind}')
            if any(not p.is_alive() and c in pending and not c.poll() for p, c in zip(self.processes, self.connections)):
                raise RuntimeError('Tensor parallel rank exited; request was not replayed')
        return result

    def infer(self, task, data, emit=None, cancel=None):
        if not self.processes:
            raise RuntimeError('Tensor parallel group failed; unload and reload the model before another request')
        data = dict(data)
        # Sampling must make identical token choices on all ranks.
        if data.get('seed') is None:
            data['seed'] = secrets.randbits(31)
        self.cancel.clear()
        try:
            for connection in self.connections:
                connection.send((task, data))
            return self.collect('done', emit, cancel)
        except BaseException:
            self.close()
            raise

    def details(self):
        return {'backend': 'transformers', 'device': self.device, 'parallelism': 'tensor_parallel',
                'execution_backend': 'torch.distributed', 'distributed_backend': self.distributed_backend,
                'world_size': self.size, 'running': bool(self.processes) and all(p.is_alive() for p in self.processes),
                'ranks': [self.ranks[k] for k in sorted(self.ranks)],
                'placement': 'native tensor parallel plans; profiling reports actual GPU execution separately'}

    def close(self):
        self.cancel.set()
        for connection in self.connections:
            try:
                connection.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        deadline = time.monotonic() + 2
        for proc in self.processes:
            proc.join(timeout=max(0, deadline - time.monotonic()))
        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
        deadline = time.monotonic() + 2
        for proc in self.processes:
            proc.join(timeout=max(0, deadline - time.monotonic()))
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)
        for connection in self.connections:
            connection.close()
        self.processes, self.connections = [], []
        self.directory.cleanup()
