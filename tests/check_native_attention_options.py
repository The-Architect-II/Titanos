"""Native attention, KV-cache and reasoning options: argv contract and validation, no process."""
from titanos.config import NativeOptions
from titanos.native import attention_arguments, reasoning_arguments


def main():
    assert attention_arguments(NativeOptions(model_file='m.gguf')) == [], 'unset options must add no argument'
    assert attention_arguments(NativeOptions(model_file='m.gguf', flash_attn='off')) == ['--flash-attn', 'off']
    assert attention_arguments(NativeOptions(model_file='m.gguf', flash_attn='on', cache_type_k='q8_0', cache_type_v='q8_0')) == [
        '--flash-attn', 'on', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0']
    assert attention_arguments(NativeOptions(model_file='m.gguf', cache_type_k='f16', cache_type_v='f16')) == [
        '--cache-type-k', 'f16', '--cache-type-v', 'f16']
    for bad in ({'cache_type_v': 'q8_0'}, {'cache_type_v': 'q4_0', 'flash_attn': 'off'}, {'flash_attn': 'yes'},
                {'cache_type_k': 'q3_k'}, {'flash_attn': True}):
        try:
            NativeOptions(model_file='m.gguf', **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted {bad}')
    assert reasoning_arguments(NativeOptions(model_file='m.gguf')) == [], 'unset reasoning must add no argument'
    for value in ('on', 'off', 'auto'):
        assert reasoning_arguments(NativeOptions(model_file='m.gguf', reasoning=value)) == ['--reasoning', value]
    for bad in ('none', 'Off', True, 0):
        try:
            NativeOptions(model_file='m.gguf', reasoning=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted reasoning={bad!r}')
    assert reasoning_arguments(NativeOptions(model_file='m.gguf', reasoning='off', reasoning_budget=0)) == [
        '--reasoning', 'off', '--reasoning-budget', '0']
    assert reasoning_arguments(NativeOptions(model_file='m.gguf', reasoning_budget=-1)) == ['--reasoning-budget', '-1']
    for bad in (-2, True, '0', 1.5):
        try:
            NativeOptions(model_file='m.gguf', reasoning_budget=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted reasoning_budget={bad!r}')
    print('native attention options: pass')


if __name__ == '__main__':
    main()
