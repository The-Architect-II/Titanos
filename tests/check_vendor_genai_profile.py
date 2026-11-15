"""Vendor-profile and collector contracts without importing numerical runtimes.

Discovery/resolution and PyInstaller are stubbed. This verifies the builder's
selection and collector inputs, not installed DLL completeness or inference.
"""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import tomllib
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_vendor_binary as builder


def main():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    requested = []

    def resolution(requirements):
        requested[:] = requirements
        return {**{Requirement(value).name: SimpleNamespace(version='fixture') for value in requirements},
                'transitive-fixture': SimpleNamespace(version='fixture')}

    core = ['titanos', 'uvicorn', 'huggingface_hub', 'mcp', 'mcp_types']
    compiled = ['numpy', 'transformers', 'tokenizers', 'safetensors', 'sentencepiece']
    with ExitStack() as stack:
        stack.enter_context(patch.object(builder.metadata, 'version',
            side_effect=lambda name: project['version'] if name == 'titanos-engine' else '6.22.2'))
        stack.enter_context(patch.object(builder.metadata, 'distribution', return_value=SimpleNamespace(version=project['version'])))
        stack.enter_context(patch.object(builder.util, 'find_spec', return_value=object()))
        stack.enter_context(patch.object(builder.common, 'dependencies', side_effect=resolution))
        genai = builder.plan('genai')
        assert genai['collect_packages'] == core + ['numpy', 'openvino', 'openvino_genai', 'openvino_tokenizers']
        assert requested == project['dependencies'] + project['optional-dependencies']['genai']
        assert genai['profile'] == 'vendor-genai' and genai['mcp_included'] and genai['weights_included'] is False
        assert not {'torch', 'transformers', 'diffusers', 'tokenizers', 'sentencepiece', 'imageio', 'qwen_tts'}.intersection(genai['collect_packages'])
        assert genai['dependency_exceptions'] == []
        with patch.object(builder.util, 'find_spec', side_effect=lambda name: None if name == 'openvino_tokenizers' else object()):
            try:
                builder.plan('genai')
            except ValueError as exc:
                assert 'openvino_tokenizers' in str(exc)
            else:
                raise AssertionError('Missing GenAI tokenizer package was accepted')
        sidecar = builder.plan('genai', without_mcp=True)
        assert sidecar['mcp_included'] is False and not {'mcp', 'mcp_types'}.intersection(sidecar['collect_packages'])
        assert not any(Requirement(value).name == 'mcp' for value in requested)
        assert sidecar['dependency_exceptions'][0]['dependency'] == 'mcp'

        # Existing graph/frontend profiles keep their established module and
        # direct-requirement selections, including native ARM64 WindowsML.
        for architecture in ['AMD64', 'ARM64']:
            with patch.object(builder.platform, 'machine', return_value=architecture):
                legacy = {
                    'qnn': (['onnxruntime', 'onnxruntime_qnn'], ['onnxruntime==1.27.0', 'onnxruntime-qnn==2.6.0']),
                    'onnx': (['onnxruntime'], ['onnxruntime==1.27.0']),
                    'coreml': (['onnxruntime'], ['onnxruntime==1.27.0']),
                    'openvino': (['openvino'], ['openvino==2026.3.1']),
                    'dml': (['onnxruntime'], ['onnxruntime-windowsml' if architecture == 'ARM64' else 'onnxruntime-directml']),
                    'mcp': ([], []),
                }
                for runtime, (modules, requirements) in legacy.items():
                    for omitted in [False, True] if runtime != 'mcp' else [False]:
                        value = builder.plan(runtime, without_mcp=omitted)
                        base = core[:3] if omitted else core
                        dependencies = [r for r in project['dependencies'] if not omitted or Requirement(r).name != 'mcp']
                        assert value['collect_packages'] == base + ([] if runtime == 'mcp' else compiled) + modules
                        assert requested == dependencies + ([] if runtime == 'mcp' else project['optional-dependencies']['compiled']) + requirements
                        assert value['weights_included'] is False and value['mcp_included'] is not omitted
        with patch.object(sys, 'argv', ['builder', '--runtime', 'genai', '--plan', '--output', 'unused-output', '--work-dir', 'unused-work']), \
             patch.object(builder.subprocess, 'run', side_effect=AssertionError('Plan must not launch a build')):
            output = io.StringIO()
            with redirect_stdout(output):
                builder.main()
            assert json.loads(output.getvalue())['profile'] == 'vendor-genai'

    # Execute the unchanged shared spec with collector stubs so assertions bind
    # to actual collect_all/copy_metadata/Analysis arguments, not textual matches.
    collected, metadata, analysis_options = [], [], {}
    hooks = ModuleType('PyInstaller.utils.hooks')

    def collect(name, **options):
        collected.append((name, options))
        return [(name + '/data', name)], [(name + '/native.dll', name)], [name + '.runtime']

    hooks.collect_all = collect
    hooks.copy_metadata = lambda name: metadata.append(name) or [(name + '.dist-info', name + '.dist-info')]

    def analysis(*args, **kwargs):
        analysis_options.update(kwargs)
        analysis_options['entry'] = args[0]
        return SimpleNamespace(pure=[], scripts=[], binaries=kwargs['binaries'], datas=kwargs['datas'])

    with tempfile.TemporaryDirectory(prefix='titanos-genai-profile-') as directory:
        plan_path = Path(directory) / 'plan.json'
        plan_path.write_text(json.dumps(genai), encoding='utf-8')
        with patch.dict(os.environ, {'TITANOS_BINARY_PLAN': str(plan_path)}), \
             patch.dict(sys.modules, {'PyInstaller': ModuleType('PyInstaller'),
                                    'PyInstaller.utils': ModuleType('PyInstaller.utils'),
                                    'PyInstaller.utils.hooks': hooks}):
            runpy.run_path(str(ROOT / 'tools/titanos.spec'), init_globals={
                'Analysis': analysis, 'PYZ': lambda *a, **k: None,
                'EXE': lambda *a, **k: None, 'COLLECT': lambda *a, **k: None})
    assert [name for name, _ in collected] == genai['collect_packages']
    assert metadata == genai['distributions'] and 'transitive-fixture' in metadata
    for name, options in collected:
        assert options['include_py_files'] is True
        assert {'**/*.safetensors', '**/*.gguf', '**/*.ckpt', '**/*.pt', '**/*.pth', '**/*.onnx', '**/tests/**'} <= set(options['exclude_datas'])
        assert options['filter_submodules'](name + '.runtime')
        assert not options['filter_submodules'](name + '.tests.example')
    assert analysis_options['entry'] == [str(ROOT / 'tools/frozen_entry.py')]
    assert {'mcp.runtime', 'mcp_types.runtime', 'openvino_genai.runtime', 'openvino_tokenizers.runtime'} <= set(analysis_options['hiddenimports'])
    assert not {'torch', 'transformers', 'openvino', 'openvino_genai', 'openvino_tokenizers', 'numpy'}.intersection(sys.modules)
    print(json.dumps({'passed': True, 'checks': ['GenAI modules and pinned project extra', 'MCP default and explicit sidecar',
        'Missing tokenizer package fails plan', 'Legacy graph/frontend profiles on AMD64 and ARM64',
        'Plan-only CLI does not start a build', 'Shared collector data/DLL/source/metadata inputs and weight exclusions'],
        'scope': 'Static selection/collector contracts with discovery, resolver and PyInstaller stubs; no native imports, build or inference'}))


if __name__ == '__main__':
    main()
