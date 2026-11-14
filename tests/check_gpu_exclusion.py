"""TITANOS_EXCLUDE_GPU_UUIDS in native (llama-cpp) placement: the setting accepts complete NVIDIA GPU UUIDs only; automatic
placement takes the first visible GPU that is not excluded and fails when every one is; an explicit selection of an
excluded GPU fails closed after the usual visibility admission; CUDA_VISIBLE_DEVICES is narrowed to the chosen UUID.
Synthetic device inventories (patched nvidia-smi scan, CUDA probe and llama-server device list): no process, GPU or
network."""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from titanos import native  # noqa: E402
from titanos.config import ModelSpec  # noqa: E402
from titanos.settings import Settings, _active_settings  # noqa: E402

A = 'GPU-11111111-1111-1111-1111-111111111111'
B = 'GPU-22222222-2222-2222-2222-222222222222'


def check_setting():
    assert Settings().exclude_gpu_uuids == '' and Settings(exclude_gpu_uuids=f' {B} , {A}').exclude_gpu_uuids
    assert Settings.from_env({'TITANOS_EXCLUDE_GPU_UUIDS': B}).exclude_gpu_uuids == B
    for bad in ('GPU-1', '0', 'cuda:1', A + ',1'):
        try:
            Settings(exclude_gpu_uuids=bad)
        except ValueError as exc:
            assert 'TITANOS_EXCLUDE_GPU_UUIDS' in str(exc)
            continue
        raise AssertionError(f'{bad!r} accepted')


def check_exclusion():
    rows = [{'uuid': B}, {'uuid': A}]
    token = _active_settings.set(Settings(exclude_gpu_uuids=B))
    try:
        assert native.first_eligible(rows) == A
        assert native.refuse_excluded([A]) == [A]
        try:
            native.refuse_excluded([B])
        except ValueError as exc:
            assert 'excluded by TITANOS_EXCLUDE_GPU_UUIDS' in str(exc)
        else:
            raise AssertionError('an excluded GPU was accepted')
        try:
            native.first_eligible([{'uuid': B}])
        except ValueError as exc:
            assert 'Every visible NVIDIA GPU is excluded' in str(exc)
        else:
            raise AssertionError('all-excluded automatic placement chose a GPU')
        # resolve_native_placement: automatic placement skips the excluded first GPU; an explicit excluded UUID fails closed
        physical = [{'uuid': B, 'index': 0, 'compute_capability': '12.0'}, {'uuid': A, 'index': 1, 'compute_capability': '12.0'}]

        def observed(library_dir=None, environ=None):
            mask = environ.get('CUDA_VISIBLE_DEVICES')
            identities = mask.split(',') if mask else [B, A]
            return [{'uuid': identity, 'local_device': f'CUDA{i}'} for i, identity in enumerate(identities)]
        with patch('titanos.devices.scan', return_value=physical), \
             patch('titanos.devices.probe_visible_cuda', side_effect=observed), \
             patch.object(native, 'native_devices', return_value={'CUDA0': 'Synthetic GPU', 'CUDA1': 'Synthetic GPU'}):
            auto = ModelSpec(path='inert', task='chat', backend='llama-cpp', native={'model_file': 'model.gguf', 'gpu_layers': 'all'})
            env = {}
            result = native.resolve_native_placement(auto, Path('llama-server.exe'), env)
            assert result['resolved_device'] == A and env['CUDA_VISIBLE_DEVICES'] == A, (result, env)
            explicit = auto.model_copy(update={'device': B})
            try:
                native.resolve_native_placement(explicit, Path('llama-server.exe'), {})
            except ValueError as exc:
                assert 'excluded by TITANOS_EXCLUDE_GPU_UUIDS' in str(exc)
            else:
                raise AssertionError('an explicit excluded GPU was placed')
    finally:
        _active_settings.reset(token)



def main():
    check_setting()
    check_exclusion()
    print('{"check": "gpu_exclusion", "result": "pass", "scope": "synthetic device inventories; no process, GPU or network"}')


if __name__ == '__main__':
    main()
