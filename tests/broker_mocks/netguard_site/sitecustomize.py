"""Installs the broker test netguard (tests/broker_mocks/netguard.py) at interpreter start in subprocesses.

Active only when TITANOS_TEST_NETGUARD=1 and this directory is on PYTHONPATH (site imports the first sitecustomize it finds;
`python -S` or `-I` skips site and therefore the guard). netguard.py is loaded by file path under the module name
'titanos_test_netguard', so it needs neither tests/ on sys.path nor the package. Blocked attempts are appended to the file
named by TITANOS_TEST_NETGUARD_REPORT. A sitecustomize later on sys.path (a venv's own) is still executed afterwards.
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

if os.environ.get('TITANOS_TEST_NETGUARD', '').strip() == '1' and 'titanos_test_netguard' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('titanos_test_netguard',
                                                   os.path.join(os.path.dirname(_HERE), 'netguard.py'))
    _module = importlib.util.module_from_spec(_spec)
    sys.modules['titanos_test_netguard'] = _module
    _spec.loader.exec_module(_module)
    _module.install()


def _chain():
    """Run the next sitecustomize on sys.path, if any, so this one does not shadow it."""
    for entry in sys.path:
        try:
            if not entry or os.path.normcase(os.path.abspath(entry)) == os.path.normcase(_HERE):
                continue
            candidate = os.path.join(entry, 'sitecustomize.py')
            if os.path.isfile(candidate):
                spec = importlib.util.spec_from_file_location('_titanos_chained_sitecustomize', candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return
        except Exception:
            return


_chain()
