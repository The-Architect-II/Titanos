"""Unchanged native bytes retain exact metadata; unexpected edits fail closed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

spec = importlib.util.spec_from_file_location('native_inventory', Path(__file__).resolve().parents[1] / 'tools/native_inventory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    source, bundle = root / 'source', root / 'bundle'
    target = bundle / '_internal/native/ffmpeg'
    source.mkdir()
    target.mkdir(parents=True)
    (source / 'ffmpeg').write_bytes(b'synthetic unchanged native bytes')
    (target / 'ffmpeg').write_bytes((source / 'ffmpeg').read_bytes())
    record = {'binaries': {'ffmpeg': module._digest(source / 'ffmpeg')}, 'license_from_binary': 'fixture'}
    content = json.dumps(record, indent=4).encode()
    (source / 'BUILD.json').write_bytes(content)
    (target / 'BUILD.json').write_bytes(content)
    assert module.finalize_native_inventory(bundle, source, 'fixture') == record
    assert (target / 'BUILD.json').read_bytes() == content
    (target / 'ffmpeg').write_bytes(b'changed fixture')
    try:
        module.finalize_native_inventory(bundle, source, 'fixture')
    except (ValueError, subprocess.SubprocessError):
        pass
    else:
        raise AssertionError('Unverified native changes accepted')
    assert (target / 'BUILD.json').read_bytes() == content
print('PASS: unchanged bytes preserve native metadata; unverified edits cannot rewrite provenance (synthetic integrity fixture)')
