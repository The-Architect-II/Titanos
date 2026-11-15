"""GPU memory admission for llama-cpp starts (titanos.admission).

Covers: the estimate (offloaded GGUF bytes + KV cache for context_size + compute margin) from a synthetic header; the
decision against injected free memory per GPU UUID minus TITANOS_ADMISSION_RESERVE_MIB, TITANOS_ADMISSION_FREE_MIB as a
cap that can only lower the reading, released bytes for a resize, tensor_split shares and the off/auto/on modes when the
header or nvidia-smi cannot be read; NativeWorker.start_process refusing before any directory, key or process exists and
NativeWorker.start passing the refusal through unchanged; a resize refused before the running runtime is stopped; the HTTP
answer (503 insufficient_device_memory with Retry-After). Free memory is always injected: no nvidia-smi query, model,
llama-server, GPU or network."""
import asyncio
import json
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from fastapi.testclient import TestClient  # noqa: E402

from titanos import admission, native  # noqa: E402
from titanos.admission import MIB, AdmissionRefused, admit, estimate  # noqa: E402
from titanos.config import Config, ModelSpec, NativeOptions  # noqa: E402
from titanos.native import NativePool  # noqa: E402
from titanos.server import create_app  # noqa: E402
from titanos.settings import Settings, _active_settings  # noqa: E402

A = 'GPU-11111111-1111-1111-1111-111111111111'
B = 'GPU-22222222-2222-2222-2222-222222222222'
WEIGHTS = 64 * MIB


def gguf(path, weights=WEIGHTS):
    def text(value):
        return struct.pack('<Q', len(value)) + value.encode()
    pairs = [('general.architecture', 8, text('qwen2')), ('qwen2.context_length', 4, struct.pack('<I', 32768)),
             ('qwen2.block_count', 4, struct.pack('<I', 24)), ('qwen2.embedding_length', 4, struct.pack('<I', 896)),
             ('qwen2.attention.head_count', 4, struct.pack('<I', 14)), ('qwen2.attention.head_count_kv', 4, struct.pack('<I', 2))]
    header = b'GGUF' + struct.pack('<IQQ', 3, 0, len(pairs)) + b''.join(text(k) + struct.pack('<I', t) + v for k, t, v in pairs)
    with open(path, 'wb') as stream:
        stream.write(header)
        stream.truncate(weights)  # sparse padding stands in for tensor data
    return Path(path)


def spec_for(folder, device=A, **options):
    gguf(folder / 'model.gguf')
    return ModelSpec(path=str(folder), task='chat', backend='llama-cpp', device=device,
                     native=NativeOptions(model_file='model.gguf', context_size=8192, slots=4, gpu_layers='all', **options))


def placed(*devices):
    return {'resolved_device': ','.join(devices), 'resolved_gpu_layers': 'all', 'native_devices': ['CUDA0'], 'verified_devices': []}


def refusal(call):
    try:
        call()
    except AdmissionRefused as exc:
        assert exc.status == 503 and exc.code == 'insufficient_device_memory' and exc.headers == {'Retry-After': '30'}
        return exc
    raise AssertionError('admitted')


def check_estimate(folder):
    spec = spec_for(folder)
    files = [folder / 'model.gguf']
    need = estimate(spec, files=files)
    kv = 24 * 2 * (64 + 64) * 2 * 8192  # 12 KiB per position x 8192 positions = 96 MiB
    assert need['weights'] == WEIGHTS and need['kv_cache'] == kv and need['compute'] == 512 * MIB, need
    assert need['total'] == WEIGHTS + kv + 512 * MIB and need['context_size'] == 8192 and need['offload_fraction'] == 1.0
    assert estimate(spec, files=files, slot_context=8192)['kv_cache'] == 4 * kv  # 8192 per slot x 4 slots
    half = spec.model_copy(update={'native': spec.native.model_copy(update={'gpu_layers': 12})})
    assert estimate(half, files=files)['weights'] == WEIGHTS // 2 and estimate(half, files=files)['kv_cache'] == kv // 2
    quantized = spec.model_copy(update={'native': spec.native.model_copy(update={'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0',
                                                                                 'flash_attn': 'on'})})
    assert estimate(quantized, files=files)['kv_cache'] == kv * 34 // 64
    return need


def check_decisions(folder, need):
    spec = spec_for(folder)
    files = [folder / 'model.gguf']
    total_mib = -(-need['total'] // MIB)
    plain = Settings(native_admission='on', admission_reserve_mib=12288)
    assert admit(spec, placed('cpu'), files=files, settings=plain)['checked'] is False
    assert admit(spec, placed(A), files=files, settings=Settings(native_admission='off'), free={A: 1})['checked'] is False
    fits = admit(spec, placed(A), files=files, settings=plain, free={A: 12288 + total_mib})
    assert fits['checked'] and fits['devices'][0]['required_mib'] == total_mib and fits['reserve_mib'] == 12288, fits
    exc = refusal(lambda: admit(spec, placed(A), files=files, settings=plain, free={A: 12288 + total_mib - 1}))
    assert f'needs about {total_mib} MiB on {A}' in str(exc) and '12288 MiB reserve' in str(exc) and 'nothing was started' in str(exc)
    # the injected cap only lowers the reading
    capped = Settings(native_admission='on', admission_free_mib=total_mib - 1)
    refusal(lambda: admit(spec, placed(A), files=files, settings=capped, free={A: 90000}))
    assert admit(spec, placed(A), files=files, settings=Settings(native_admission='on', admission_free_mib=90000), free={A: total_mib})['checked']
    # a resize is credited with the memory the stopped runtime releases
    released = admit(spec, placed(A), files=files, settings=plain, free={A: 12288}, released_bytes=need['total'])
    assert released['devices'][0]['available_mib'] >= total_mib, released
    # shares across two GPUs follow tensor_split
    split = spec.model_copy(update={'native': spec.native.model_copy(update={'tensor_split': [3.0, 1.0], 'split_mode': 'layer'})})
    two = admit(split, placed(A, B), files=files, settings=Settings(native_admission='on'), free={A: total_mib, B: total_mib})
    assert [row['required_mib'] for row in two['devices']] == [-(-(need['total'] * 3 // 4) // MIB), -(-(need['total'] // 4) // MIB)]
    # unreadable header or memory: auto admits and says why, on refuses
    bad = spec.model_copy(update={'native': spec.native.model_copy(update={'model_file': 'missing.gguf'})})
    auto = admit(bad, placed(A), files=[folder / 'missing.gguf'], settings=Settings(), free={A: 1})
    assert auto['checked'] is False and auto['mode'] == 'auto' and auto['reason'], auto
    refusal(lambda: admit(bad, placed(A), files=[folder / 'missing.gguf'], settings=plain, free={A: 1}))
    with patch.object(admission.shutil, 'which', return_value=None):
        assert admit(spec, placed(A), files=files, settings=Settings())['checked'] is False
        exc = refusal(lambda: admit(spec, placed(A), files=files, settings=plain))
        assert 'could not measure' in str(exc)
    assert admit(spec, placed(A), files=files, settings=Settings(), free={B: 90000})['checked'] is False  # GPU not listed
    for bad_value in ({'native_admission': 'sometimes'}, {'exclude_gpu_uuids': 'GPU-1'}, {'exclude_gpu_uuids': '0'}):
        try:
            Settings(**bad_value)
        except ValueError:
            continue
        raise AssertionError(f'{bad_value} accepted')
    assert Settings(exclude_gpu_uuids=f' {B} , {A}').exclude_gpu_uuids


def check_start(folder, need):
    """start_process refuses after placement and before the directory, key and runtime process; start() lets it through."""
    total_mib = -(-need['total'] // MIB)
    spec = spec_for(folder)
    executable = folder / 'llama-server.exe'
    executable.write_bytes(b'inert executable bytes')
    worker = native.NativeWorker('w', 'm', spec)
    token = _active_settings.set(Settings(native_admission='on', admission_reserve_mib=1024))
    try:
        with patch.object(native, 'executable_path', return_value=executable), \
             patch.object(native, 'resolve_native_placement', return_value=placed(A)), \
             patch.object(admission, 'free_memory_mib', return_value={A.lower(): total_mib}), \
             patch.object(native.subprocess, 'Popen', side_effect=AssertionError('a runtime process was started')):
            refusal(worker.start_process)
            assert worker.temp is None and worker.process is None, 'nothing was created before the refusal'
            try:
                asyncio.run(worker.start())
            except AdmissionRefused as exc:
                assert exc.code == 'insufficient_device_memory' and worker.temp is None and worker.error is None
            else:
                raise AssertionError('start admitted')
    finally:
        _active_settings.reset(token)


def check_resize(folder, need):
    """A resize is admitted before the running runtime is stopped: a refusal leaves it serving."""
    gguf(folder / 'model.gguf')
    cfg = Config(models={'m': ModelSpec(path=str(folder), task='chat', backend='llama-cpp', device=A,
                                        native=NativeOptions(model_file='model.gguf', context_size=8192, slots=4, gpu_layers='all'))})
    pool = NativePool(cfg, Settings(native_queue_timeout=1.0))
    worker = pool.workers[0]
    events = []
    worker.process, worker.ready, worker.client = NS(poll=lambda: None), True, NS(is_closed=False)
    worker.placement = placed(A)
    worker.admission = {'estimate': need}

    async def start():
        events.append('start')

    async def stop():
        events.append('stop')
    worker.start, worker.stop = start, stop
    token = _active_settings.set(Settings(native_admission='on', admission_reserve_mib=1024))
    try:
        # 32768 per slot needs 1 GiB more KV than the free memory holds, even with the running runtime's memory credited
        with patch.object(admission, 'free_memory_mib', return_value={A.lower(): 1024 + 64}):
            async def scenario():
                async with pool.lease('m', num_ctx=32768):
                    pass
            try:
                asyncio.run(scenario())
            except AdmissionRefused:
                pass
            else:
                raise AssertionError('resize admitted')
        assert events == [] and worker.running and worker.slot_context is None, events
    finally:
        _active_settings.reset(token)


def check_http(folder, need):
    gguf(folder / 'model.gguf')
    executable = folder / 'llama-server.exe'
    executable.write_bytes(b'inert executable bytes')
    cfg = Config(models={'native': ModelSpec(path=str(folder), task='chat', backend='llama-cpp', device=A,
                                             native=NativeOptions(model_file='model.gguf', context_size=8192, slots=4, gpu_layers='all'))})
    settings = Settings(native_admission='on', admission_reserve_mib=12288)
    app = create_app(cfg, output_dir=folder / 'outputs', settings=settings)
    with patch.object(native, 'executable_path', return_value=executable), \
         patch.object(native, 'resolve_native_placement', return_value=placed(A)), \
         patch.object(admission, 'free_memory_mib', return_value={A.lower(): 12288 + 100}), \
         patch.object(native.subprocess, 'Popen', side_effect=AssertionError('a runtime process was started')):
        with TestClient(app, base_url='http://127.0.0.1') as client:
            r = client.post('/v1/chat/completions', json={'model': 'native', 'messages': [{'role': 'user', 'content': 'Hi'}]})
            assert r.status_code == 503 and r.headers.get('retry-after') == '30', (r.status_code, dict(r.headers), r.text)
            assert r.json()['error']['code'] == 'insufficient_device_memory', r.text
            workers = client.get('/api/workers').json()
            assert json.dumps(workers).count('"running": true') == 0, workers


def check_memory(folder, need):
    """GET /api/models/memory: the llama-cpp estimate and the weights-size heuristic against fake per-GPU readings, the
    reserve, 'unknown' without an estimate, no device rows for a cpu model or without nvidia-smi; nothing is loaded."""
    gguf(folder / 'model.gguf')
    dense = folder / 'dense'
    dense.mkdir(exist_ok=True)
    with open(dense / 'model.safetensors', 'wb') as stream:
        stream.truncate(100 * MIB)
    (dense / 'config.json').write_text('{}')
    cfg = Config(models={
        'native': ModelSpec(path=str(folder), task='chat', backend='llama-cpp',
                            native=NativeOptions(model_file='model.gguf', context_size=8192, slots=4, gpu_layers='all')),
        'dense': ModelSpec(path=str(dense), task='chat'),
        'host': ModelSpec(path=str(dense), task='chat', device='cpu'),
        'gone': ModelSpec(path=str(folder / 'absent'), task='chat')})
    app = create_app(cfg, output_dir=folder / 'outputs', settings=Settings(admission_reserve_mib=1024))
    heuristic = -(-100 * MIB * 12 // 10)
    readings = {A.lower(): 1024 + 200, B.lower(): 1024 + 110}
    with patch.object(admission, 'free_memory_mib', return_value=readings), \
         patch.object(native.subprocess, 'Popen', side_effect=AssertionError('a runtime process was started')):
        with TestClient(app, base_url='http://127.0.0.1') as client:
            body = client.get('/api/models/memory').json()
            rows = {row['model']: row for row in body['data']}
            assert body['free_memory']['available'] and body['reserve_mib'] == 1024, body
            assert rows['native']['method'] == 'gguf_estimate' and rows['native']['estimate_bytes'] == need['total'], rows
            assert [d['fits'] for d in rows['native']['devices']] == [False, False]  # 64 MiB + KV + 512 MiB margin > 200 MiB
            assert rows['dense']['method'] == 'weights_size_heuristic' and rows['dense']['estimate_bytes'] == heuristic, rows
            assert rows['dense']['detail'] == {'weights_on_disk': 100 * MIB, 'factor': 1.2}
            assert [(d['uuid'], d['available_bytes'], d['fits']) for d in rows['dense']['devices']] == [
                (A.lower(), 200 * MIB, True), (B.lower(), 110 * MIB, False)], rows['dense']
            assert rows['host']['devices'] == [] and rows['host']['estimate_bytes'] == heuristic
            assert rows['gone']['estimate_bytes'] is None and {d['fits'] for d in rows['gone']['devices']} == {'unknown'}
            assert not client.get('/api/models/loaded').json()['data']
            with patch.object(admission, 'free_memory_mib', side_effect=ValueError('Memory admission requires nvidia-smi')):
                body = client.get('/api/models/memory').json()
    assert body['free_memory']['available'] is False and 'nvidia-smi' in body['free_memory']['reason'], body
    assert all(row['devices'] == [] for row in body['data']) and body['data'][0]['estimate_bytes'] == need['total']


def check_memory_placement(folder, need):
    """memory_view rows follow the registered placement: pinned UUIDs only with their tensor_split share (one not listed
    is 'unknown'), none for native:DEVICE, excluded GPUs skipped for llama-cpp auto; the TITANOS_ADMISSION_FREE_MIB cap;
    an unreadable GGUF header is 'unknown' with a reason; a folder with .bin and .safetensors copies counts one format."""
    split = spec_for(folder, device=f'{A},{B}', split_mode='layer', tensor_split=[3, 1])
    auto = spec_for(folder, device='auto')
    vulkan = spec_for(folder, device='native:Vulkan0')
    pinned = spec_for(folder, device='GPU-33333333-3333-3333-3333-333333333333')
    dual = folder / 'dual'
    dual.mkdir()
    for name, size in (('model.safetensors', 10 * MIB), ('pytorch_model.bin', 10 * MIB)):
        with open(dual / name, 'wb') as stream:
            stream.truncate(size)
    bad = folder / 'bad'
    bad.mkdir()
    (bad / 'model.gguf').write_bytes(b'NOPE' + bytes(64))
    broken = ModelSpec(path=str(bad), task='chat', backend='llama-cpp',
                       native=NativeOptions(model_file='model.gguf', context_size=8192, slots=4, gpu_layers='all'))
    models = {'split': split, 'auto': auto, 'vulkan': vulkan, 'pinned': pinned, 'broken': broken,
              'dual': ModelSpec(path=str(dual), task='chat')}
    view = admission.memory_view(models, Settings(admission_free_mib=500, exclude_gpu_uuids=B), free={A: 4000, B: 4000})
    rows = {row['model']: row for row in view['data']}
    total = need['total']
    assert [(d['uuid'], d['estimate_bytes'], d['free_bytes']) for d in rows['split']['devices']] == [
        (A, -(-total * 3 // 4), 500 * MIB), (B, -(-total // 4), 500 * MIB)], rows['split']
    assert [d['fits'] for d in rows['split']['devices']] == [False, True], rows['split']
    assert [d['uuid'] for d in rows['auto']['devices']] == [A.lower()] and rows['vulkan']['devices'] == []
    assert [(d['free_bytes'], d['fits']) for d in rows['pinned']['devices']] == [(None, 'unknown')], rows['pinned']
    assert rows['broken']['estimate_bytes'] is None and 'GGUF' in rows['broken']['detail']['reason'], rows['broken']
    assert {d['fits'] for d in rows['broken']['devices']} == {'unknown'}
    assert rows['dual']['detail']['weights_on_disk'] == 10 * MIB, rows['dual']


def main():
    with tempfile.TemporaryDirectory(prefix='titanos-admission-') as name:
        folder = Path(name)
        need = check_estimate(folder)
        check_decisions(folder, need)
        check_start(folder, need)
        check_resize(folder, need)
        check_http(folder, need)
        check_memory(folder, need)
        check_memory_placement(folder, need)
    print('{"check": "admission", "result": "pass", "scope": "injected free memory, synthetic GGUF headers, fake runtimes; '
          'no nvidia-smi, model, llama-server, GPU or network"}')


if __name__ == '__main__':
    main()
