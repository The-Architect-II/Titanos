"""Verify lean vision dependency boundaries without numerical imports/builds."""
from contextlib import ExitStack, redirect_stdout
import ast
import io
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace as NS
from unittest.mock import patch

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_vendor_binary as builder


def main():
    source=ast.parse((ROOT/'tools/build_vendor_binary.py').read_text(encoding='utf-8'))
    gate=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='text_vision_imports')
    imports={alias.name for node in ast.walk(gate) if isinstance(node,ast.ImportFrom) and node.module=='transformers' for alias in node.names}
    assert {'SamConfig','SamModel','SamImageProcessor','SamProcessor'} <= imports
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    requested = []
    def dependencies(values):
        requested[:] = values
        result = {Requirement(value).name: NS(version='fixture') for value in values}
        result['torch'] = NS(version='2.8.0+cpu')
        return result
    def available(name):
        return None if name in builder.TEXT_VISION_EXCLUDED else object()
    def rejected(action, message):
        try:
            action()
        except ValueError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError('Missing rejection: ' + message)
    with ExitStack() as stack:
        stack.enter_context(patch.object(builder.metadata, 'version', side_effect=lambda name: project['version'] if name == 'titanos-engine' else '6.22.2'))
        stack.enter_context(patch.object(builder.metadata, 'distribution', return_value=NS(version=project['version'])))
        stack.enter_context(patch.object(builder.util, 'find_spec', side_effect=available))
        stack.enter_context(patch.object(builder.common, 'dependencies', side_effect=dependencies))
        plan = builder.plan('text-vision')
        assert plan['mcp_included'] and not plan['weights_included']
        assert {'PIL', 'torch', 'transformers', 'mcp', 'mcp_types'} <= set(plan['collect_packages'])
        assert not set(builder.TEXT_VISION_EXCLUDED).intersection(plan['collect_packages'])
        assert 'pillow' in {Requirement(value).name for value in requested}
        assert 'PIL' not in plan['exclude_modules'] and 'PIL' in builder.TEXT_AUDIO_EXCLUDED
        assert not plan.get('ffmpeg_dir') and not plan.get('native_ffmpeg')
        assert plan['runtime_import_validation'] == 'required before compilation; not run by --plan'
        rejected(lambda: builder.plan('text-vision', without_mcp=True), 'requires full MCP')
        rejected(lambda: builder.plan('text-vision', ffmpeg_dir=Path('unused')), '--ffmpeg-dir')
        with patch.object(builder.util, 'find_spec', side_effect=lambda name: None if name == 'PIL' else available(name)):
            rejected(lambda: builder.plan('text-vision'), 'missing: PIL')
        for incidental in ('torchvision', 'scipy', 'imageio_ffmpeg'):
            with patch.object(builder.util, 'find_spec', side_effect=lambda name: object() if name == incidental else available(name)):
                rejected(lambda: builder.plan('text-vision'), 'clean CPU')
        def accelerator(values):
            result = dependencies(values)
            result['torch'].version = '2.8.0+cu128'
            return result
        with patch.object(builder.common, 'dependencies', side_effect=accelerator):
            rejected(lambda: builder.plan('text-vision'), 'CPU PyTorch')
        with patch.object(sys, 'argv', ['builder', '--runtime', 'text-vision', '--plan', '--output', 'unused', '--work-dir', 'unused']), \
             patch.object(builder, 'text_vision_imports', side_effect=AssertionError('Plan imported a numerical runtime')), \
             patch.object(builder.subprocess, 'run', side_effect=AssertionError('Plan compiled')):
            output = io.StringIO()
            with redirect_stdout(output):
                builder.main()
            assert json.loads(output.getvalue())['profile'] == 'vendor-text-vision'
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    print(json.dumps({'passed': True, 'scope': 'Profile dependency, MCP, CPU, exclusion and plan-only boundaries with doubles; native import gate/build/model quality remain separate'}))


if __name__ == '__main__':
    main()
