"""Launch Titanos's own helper modules from source or a frozen executable."""
import sys


def module_command(module, *arguments):
    if module not in {'titanos.native_process', 'titanos.device_probe'}:
        raise ValueError('Unknown Titanos helper module')
    prefix = [sys.executable, '--titanos-internal'] if getattr(sys, 'frozen', False) else [sys.executable, '-m']
    return [*prefix, module, *map(str, arguments)]
