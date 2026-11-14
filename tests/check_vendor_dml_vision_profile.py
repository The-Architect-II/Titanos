"""Build profile policy with metadata doubles; no native imports or models."""
import json
from pathlib import Path
import sys
import tomllib
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import build_vendor_binary as builder

def main():
    project=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
    requested=[]; torch_version='2.8.0+cpu'
    def dependencies(values):
        requested[:]=values
        result={Requirement(value).name:NS(version='fixture') for value in values}
        result['torch']=NS(version=torch_version)
        return result
    def reject(action,needle):
        try:action()
        except ValueError as exc:assert needle in str(exc),str(exc)
        else:raise AssertionError('Missing policy rejection: '+needle)
    with ExitStack() as stack:
        stack.enter_context(patch.object(builder.metadata,'version',side_effect=lambda name:project['version'] if name=='titanos-engine' else '6.22.2'))
        stack.enter_context(patch.object(builder.metadata,'distribution',return_value=NS(version=project['version'])))
        stack.enter_context(patch.object(builder.util,'find_spec',return_value=object()))
        stack.enter_context(patch.object(builder.common,'dependencies',side_effect=dependencies))
        stack.enter_context(patch.object(builder.platform,'system',return_value='Windows'))
        stack.enter_context(patch.object(builder.platform,'machine',return_value='AMD64'))
        plan=builder.plan('dml-vision')
        assert plan['mcp_included'] and not plan['weights_included']
        assert {'onnxruntime','torch','diffusers','PIL','scipy','mcp','mcp_types'}<=set(plan['collect_packages'])
        assert not set(plan['collect_packages'])&set(plan['exclude_modules'])
        assert 'onnxruntime-directml' in plan['distributions'] and 'onnxruntime' not in plan['distributions']
        assert requested==project['dependencies']+project['optional-dependencies']['compiled']+project['optional-dependencies']['dml-vision']
        assert plan['runtime_import_validation'].startswith('required')
        reject(lambda:builder.plan('dml-vision',without_mcp=True),'requires full MCP')
        with patch.object(builder.platform,'machine',return_value='ARM64'):
            reject(lambda:builder.plan('dml-vision'),'native Windows AMD64')
        with patch.object(builder.platform,'system',return_value='Linux'):
            reject(lambda:builder.plan('dml-vision'),'native Windows AMD64')
        for torch_version in ['2.8.0+cu128','2.8.0+rocm6.4']:
            reject(lambda:builder.plan('dml-vision'),'CPU PyTorch wheel')
        torch_version='2.8.0+cpu'
        for absent in ['diffusers','torch','PIL','onnxruntime']:
            with patch.object(builder.util,'find_spec',side_effect=lambda name:None if name==absent else object()):
                reject(lambda:builder.plan('dml-vision'),'missing: '+absent)
    assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules and 'diffusers' not in sys.modules
    print(json.dumps({'passed':True,'scope':'DML vision profile dependencies, full MCP, architecture, CPU scheduler build policy and missing-package errors; all metadata doubles'}))

if __name__=='__main__':main()
