"""Managed runtime selection and acquisition retain accelerator intent; no downloads/models."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from titanos.providers import AcquisitionRequest, ModelProvider, runtime_recommendation
from titanos.runtime_catalog import compatible_profiles, packaged_catalog


GPU='GPU-12345678-1234-1234-1234-123456789abc'


def profiles(capability=None, driver=None, visible=True):
    devices=[] if capability is None else [{'index':0,'uuid':GPU,'name':'Fixture GPU','capability':capability,'driver':'fixture'}]
    hardware={'os':'windows','arch':'x86_64','nvidia':devices,'display_vendors':[]}
    with patch('titanos.runtime_catalog.discover',return_value=hardware), \
         patch('titanos.runtime_catalog.cuda_driver_version',return_value=driver), \
         patch('titanos.devices.visible_nvidia_records',return_value=devices if visible else [],create=True):
        return compatible_profiles(packaged_catalog())


def check_profiles():
    rows=profiles(12.0,13030)
    assert rows[0]['id']=='win-cuda-13.3-x64' and rows[0]['recommended']
    assert sum(row['recommended'] for row in rows)==1
    assert all(row['one_stop_supported'] for row in rows)
    assert any(row['id']=='win-cpu-x64' for row in rows)
    # Blackwell must not be automatically directed to the packaged CUDA12.4 build.
    assert profiles(12.0,12040)[0]['id']=='win-cpu-x64'
    assert profiles(8.9,12040)[0]['id']=='win-cuda-12.4-x64'
    assert profiles(6.1,13030)[0]['id']=='win-cuda-12.4-x64'
    assert profiles(float('nan'),13030)[0]['id']=='win-cpu-x64'
    assert profiles(12.0,None)[0]['id']=='win-cpu-x64'
    assert profiles(12.0,13030,visible=False)[0]['id']=='win-cpu-x64'
    assert profiles()[0]['id']=='win-cpu-x64'
    hardware={'os':'windows','arch':'x86_64','nvidia':[{'uuid':GPU,'capability':12.0}]}
    with patch('titanos.runtime_catalog.discover',return_value=hardware), \
         patch('titanos.devices.visible_nvidia_records',side_effect=ValueError('Numeric CUDA visibility is unverified'),create=True):
        rows=compatible_profiles(packaged_catalog())
    assert 'unverified' in rows[0]['selection_reason']


def check_acquisition():
    base=dict(provider='huggingface',identifier='fixture/model',registry_name='chat',task='chat',backend='llama-cpp',model_file='model.gguf')
    provider=ModelProvider()
    for device in ('auto','cpu',GPU,'native:Vulkan0'):
        request=AcquisitionRequest(**base,device=device)
        with patch('titanos.native_install.executable_path',return_value=Path('fixture/llama-server.exe')):
            runtime=runtime_recommendation(request)
        assert runtime['requested_device']==device
        plan=provider._base_plan(request,{'source_url':'https://fixtures.invalid/model','requested_revision':'v1','resolved_revision':'a'*40,'runtime':runtime},[{'path':'model.gguf','size':4}])
        assert plan.device==device and plan.model_spec['device']==device
        assert plan.model_spec['native']=={'model_file':'model.gguf'}
    for device in ('GPU-0','native:Vulkan0,CUDA1','cuda:0','native:--bad'):
        try:AcquisitionRequest(**base,device=device)
        except ValueError:pass
        else:raise AssertionError('Acquisition accepted an ambiguous/native multi-device selector: '+device)


def check_version_query():
    from titanos import native_process
    with patch.object(native_process.sys,'argv',['internal','--query-version','2.5','runtime']), \
         patch.object(native_process,'query_devices',return_value=0) as query:
        assert native_process.main()==0
        query.assert_called_once_with('runtime',2.5,query='--version')
    with patch.object(native_process.os,'name','posix'), \
         patch.object(native_process,'native_environment',return_value={'CLEAN':'1'}), \
         patch.object(native_process.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
        assert native_process.query_devices('runtime',2.5,query='--version')==0
        assert run.call_args.args==(['runtime','--version'],)
        assert run.call_args.kwargs['env']=={'CLEAN':'1'}
        assert run.call_args.kwargs['timeout']==2.5
        try:native_process.query_devices('runtime',2.5,query='--model')
        except ValueError:pass
        else:raise AssertionError('Metadata helper accepted a non-metadata query')


if __name__=='__main__':
    check_profiles();check_acquisition();check_version_query()
    print('PASS: driver/visibility-aware managed CUDA preference, explicit CPU availability and preserved acquisition device intent; no native/model execution.')
