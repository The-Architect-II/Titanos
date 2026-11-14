"""Text bundle selection and collection with metadata/library doubles only."""

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
    def resolve(requirements):
        requested[:] = requirements
        result = {Requirement(value).name:NS(version='fixture') for value in requirements}
        result.update(torch=NS(version='2.8.0+cpu'), **{'transitive-fixture':NS(version='fixture')})
        return result
    with ExitStack() as stack:
        stack.enter_context(patch.object(builder.metadata,'version',side_effect=lambda name:project['version'] if name=='titanos-engine' else '6.22.2'))
        stack.enter_context(patch.object(builder.metadata,'distribution',return_value=NS(version=project['version'])))
        stack.enter_context(patch.object(builder.util,'find_spec',return_value=object()))
        stack.enter_context(patch.object(builder.common,'dependencies',side_effect=resolve))
        result = builder.plan('text')
        assert result['collect_packages'] == ['titanos','uvicorn','huggingface_hub','mcp','mcp_types','numpy','transformers','tokenizers','safetensors','sentencepiece','torch']
        assert requested == project['dependencies'] + project['optional-dependencies']['compiled'] + project['optional-dependencies']['sentiment']
        assert result['profile']=='vendor-text' and result['runtime']=='text' and result['mcp_included'] is True
        assert result['weights_included'] is False and result['packages']['torch']=='2.8.0+cpu'
        assert 'sentiment' in result['scope'] and 'chat' in result['scope']
        assert not {'diffusers','torchvision','torchaudio','imageio','qwen_tts','openvino','openvino_genai','onnxruntime'}.intersection(result['collect_packages'])
        for missing in ['torch','transformers','sentencepiece']:
            with patch.object(builder.util,'find_spec',side_effect=lambda name:None if name==missing else object()):
                try: builder.plan('text')
                except ValueError as exc: assert missing in str(exc)
                else: raise AssertionError('Missing text runtime dependency accepted: '+missing)
        sidecar = builder.plan('text',without_mcp=True)
        assert not sidecar['mcp_included'] and not {'mcp','mcp_types'}.intersection(sidecar['collect_packages'])
        assert not any(Requirement(value).name=='mcp' for value in requested)
        assert sidecar['dependency_exceptions'][0]['dependency']=='mcp'
        with patch.object(sys,'argv',['builder','--runtime','text','--plan','--output','unused-output','--work-dir','unused-work']), \
             patch.object(builder.subprocess,'run',side_effect=AssertionError('Plan cannot launch a build')):
            stdout=io.StringIO()
            with redirect_stdout(stdout): builder.main()
            assert json.loads(stdout.getvalue())['profile']=='vendor-text'

    collected, copied_metadata, options = [], [], {}
    hooks=ModuleType('PyInstaller.utils.hooks')
    def collect(name,**kwargs):
        collected.append((name,kwargs))
        return [(name+'/data',name)],[(name+'/library.dll',name)],[name+'.runtime']
    hooks.collect_all=collect
    hooks.copy_metadata=lambda name:copied_metadata.append(name) or [(name+'.dist-info',name+'.dist-info')]
    def analysis(*args,**kwargs):
        options.update(kwargs)
        return NS(pure=[],scripts=[],binaries=kwargs['binaries'],datas=kwargs['datas'])
    with tempfile.TemporaryDirectory(prefix='titanos-text-profile-') as directory:
        plan=Path(directory)/'plan.json';plan.write_text(json.dumps(result))
        with patch.dict(os.environ,{'TITANOS_BINARY_PLAN':str(plan)}),patch.dict(sys.modules,{
            'PyInstaller':ModuleType('PyInstaller'),'PyInstaller.utils':ModuleType('PyInstaller.utils'),'PyInstaller.utils.hooks':hooks}):
            runpy.run_path(str(ROOT/'tools/titanos.spec'),init_globals={'Analysis':analysis,'PYZ':lambda *a,**k:None,'EXE':lambda *a,**k:None,'COLLECT':lambda *a,**k:None})
    assert [name for name,_ in collected]==result['collect_packages']
    assert copied_metadata==result['distributions'] and 'transitive-fixture' in copied_metadata
    assert {'torch.runtime','transformers.runtime','mcp.runtime','mcp_types.runtime'}<=set(options['hiddenimports'])
    assert {'**/*.safetensors','**/*.gguf','**/*.onnx','**/tests/**'}<=set(next(kwargs for name,kwargs in collected if name=='torch')['exclude_datas'])
    assert not {'torch','transformers','numpy','diffusers','openvino'}.intersection(sys.modules)
    print(json.dumps({'passed':True,'checks':['text roots and project sentiment requirements','MCP default/explicit sidecar','missing dependency failures','plan CLI does not compile','shared collector includes text modules and transitive metadata, excludes weights'],'scope':'Static plan/collector checks with doubles; no numerical imports, installation, model execution or build'}))


if __name__=='__main__':
    main()
