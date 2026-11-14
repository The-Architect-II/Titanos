"""Verify the lean GenAI image profile with resolver/import/collector doubles."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import tomllib
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_vendor_binary as builder


def main():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    requested = []

    def dependencies(values):
        requested[:] = values
        return {Requirement(value).name: NS(version='fixture') for value in values}

    def available(name):
        return None if name in builder.GENAI_VISION_EXCLUDED else object()

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
        plan = builder.plan('genai-vision')
        assert plan['profile'] == 'vendor-genai-vision' and plan['runtime'] == 'genai-vision'
        assert plan['mcp_included'] and not plan['weights_included'] and plan['dependency_exceptions'] == []
        assert plan['collect_packages'] == ['titanos', 'uvicorn', 'huggingface_hub', 'mcp', 'mcp_types',
            'numpy', 'openvino', 'openvino_genai', 'openvino_tokenizers', 'PIL']
        assert requested == project['dependencies'] + project['optional-dependencies']['genai-vision']
        assert project['optional-dependencies']['genai-vision'] == project['optional-dependencies']['genai'] + ['pillow>=11,<13']
        assert 'pillow' in plan['distributions']
        assert {'torch', 'transformers', 'diffusers', 'optimum'} <= set(plan['exclude_modules'])
        assert not set(plan['exclude_modules']).intersection(plan['collect_packages'])
        assert not plan.get('ffmpeg_dir') and not plan.get('native_ffmpeg')
        assert plan['runtime_import_validation'] == 'required before compilation; not run by --plan'
        rejected(lambda: builder.plan('genai-vision', without_mcp=True), 'requires full MCP')
        rejected(lambda: builder.plan('genai-vision', ffmpeg_dir=Path('unused')), '--ffmpeg-dir')
        for missing in ['PIL', 'openvino_tokenizers']:
            with patch.object(builder.util, 'find_spec', side_effect=lambda name: None if name == missing else available(name)):
                rejected(lambda: builder.plan('genai-vision'), 'missing: ' + missing)
        for incidental in ['torch', 'transformers', 'diffusers', 'optimum', 'imageio_ffmpeg']:
            with patch.object(builder.util, 'find_spec', side_effect=lambda name: object() if name == incidental else available(name)):
                rejected(lambda: builder.plan('genai-vision'), 'incidental packages present: ' + incidental)

        # Existing text-only selection and explicit MCP sidecar stay unchanged.
        text = builder.plan('genai')
        assert text['collect_packages'] == plan['collect_packages'][:-1]
        assert requested == project['dependencies'] + project['optional-dependencies']['genai']
        assert 'exclude_modules' not in text and 'runtime_import_validation' not in text
        sidecar = builder.plan('genai', without_mcp=True)
        assert not sidecar['mcp_included'] and not {'mcp', 'mcp_types', 'PIL'}.intersection(sidecar['collect_packages'])
        assert sidecar['dependency_exceptions'][0]['dependency'] == 'mcp'

        with tempfile.TemporaryDirectory(prefix='titanos-genai-vision-cli-') as directory:
            output_path, work_path = Path(directory) / 'output', Path(directory) / 'work'
            argv = ['builder', '--runtime', 'genai-vision', '--output', str(output_path), '--work-dir', str(work_path)]
            with patch.object(sys, 'argv', argv + ['--plan']), \
                 patch.object(builder, 'genai_vision_imports', side_effect=AssertionError('Plan imported native runtime')), \
                 patch.object(builder.subprocess, 'run', side_effect=AssertionError('Plan launched a build')):
                output = io.StringIO()
                with redirect_stdout(output):
                    builder.main()
                assert json.loads(output.getvalue())['profile'] == 'vendor-genai-vision'
            # Stop at the real CLI's pre-compilation gate; no compiler is run.
            with patch.object(sys, 'argv', argv), \
                 patch.object(builder, 'genai_vision_imports', side_effect=ValueError('native import gate fixture')) as gate, \
                 patch.object(builder.subprocess, 'run', side_effect=AssertionError('Test launched a build')):
                rejected(builder.main, 'native import gate fixture')
                gate.assert_called_once_with()
            assert not output_path.exists() and not work_path.exists()

    # Resolve only fake modules/classes; imports must never construct a pipeline.
    runtime_modules = {name: ModuleType(name) for name in ['numpy', 'openvino', 'openvino_genai', 'openvino_tokenizers', 'PIL']}
    for module in runtime_modules.values():
        module.__version__ = 'fixture'
    for name in ['LLMPipeline', 'Text2ImagePipeline', 'Image2ImagePipeline']:
        setattr(runtime_modules['openvino_genai'], name, object())
    runtime_modules['PIL'].Image = object()
    with patch.dict(sys.modules, runtime_modules):
        imported = builder.genai_vision_imports()
        assert imported['passed'] and imported['pillow'] == 'fixture'
        del runtime_modules['openvino_genai'].Image2ImagePipeline
        try:
            builder.genai_vision_imports()
        except ImportError:
            pass
        else:
            raise AssertionError('Missing native image-to-image class was accepted')

    collected, metadata, analysis_options = [], [], {}
    hooks = ModuleType('PyInstaller.utils.hooks')

    def collect(name, **options):
        collected.append(name)
        return [(name + '/data', name)], [(name + '/native.dll', name)], [name + '.runtime']

    def analysis(*args, **kwargs):
        analysis_options.update(kwargs)
        return NS(pure=[], scripts=[], binaries=kwargs['binaries'], datas=kwargs['datas'])

    hooks.collect_all = collect
    hooks.copy_metadata = lambda name: metadata.append(name) or [(name + '.dist-info', name + '.dist-info')]
    with tempfile.TemporaryDirectory(prefix='titanos-genai-vision-collector-') as directory:
        plan_path = Path(directory) / 'plan.json'
        plan_path.write_text(json.dumps(plan), encoding='utf-8')
        with patch.dict(os.environ, {'TITANOS_BINARY_PLAN': str(plan_path)}), \
             patch.dict(sys.modules, {'PyInstaller': ModuleType('PyInstaller'),
                                    'PyInstaller.utils': ModuleType('PyInstaller.utils'),
                                    'PyInstaller.utils.hooks': hooks}):
            runpy.run_path(str(ROOT / 'tools/titanos.spec'), init_globals={
                'Analysis': analysis, 'PYZ': lambda *a, **k: None,
                'EXE': lambda *a, **k: None, 'COLLECT': lambda *a, **k: None})
    assert collected == plan['collect_packages'] and metadata == plan['distributions']
    assert set(plan['exclude_modules']) <= set(analysis_options['excludes'])
    assert {'mcp.runtime', 'mcp_types.runtime', 'PIL.runtime', 'openvino_genai.runtime',
            'openvino_tokenizers.runtime'} <= set(analysis_options['hiddenimports'])
    assert not {'torch', 'transformers', 'diffusers', 'optimum', 'numpy', 'openvino',
                'openvino_genai', 'openvino_tokenizers', 'PIL'}.intersection(sys.modules)
    print(json.dumps({'passed': True,
        'scope': 'GenAI vision selection, full MCP, exclusions, clean environment, import-gate and collector contracts with doubles; no native imports, builds or inference'}))


if __name__ == '__main__':
    main()
