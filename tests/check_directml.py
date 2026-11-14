"""DirectML session contract; hardware/model qualification is a separate real test."""
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
from titanos.compiled import onnx_session
from titanos.compiled import runtime_check
from titanos.devices import runtime_records, pin_runtime

def main():
    fake=MagicMock()
    fake.ExecutionMode.ORT_SEQUENTIAL='sequential'
    fake.get_available_providers.return_value=['DmlExecutionProvider','CPUExecutionProvider']
    fake.InferenceSession.return_value.get_providers.return_value=['DmlExecutionProvider']
    options=MagicMock();fake.SessionOptions.return_value=options
    adapter={'device_id':2,'name':'Example GPU','luid':'00000000:12345678','software':False}
    with patch.dict(sys.modules,{'onnxruntime':fake}),patch('titanos.compiled.platform.system',return_value='Windows'),patch('titanos.dxgi.adapters',return_value=[adapter]):
        onnx_session('model.onnx','dml',options={'device_id':2})
        assert options.enable_mem_pattern is False
        assert options.execution_mode=='sequential'
        options.add_session_config_entry.assert_called_once_with('session.disable_cpu_ep_fallback','1')
        assert fake.InferenceSession.call_args.kwargs['providers']==[('DmlExecutionProvider',{'device_id':'2'})]
        fake.InferenceSession.return_value.disable_fallback.assert_called_once()
        for value in [True,-1,1.5,'auto','-1']:
            try:onnx_session('unused.onnx','dml',options={'device_id':value})
            except ValueError:pass
            else:raise AssertionError('Invalid adapter selector accepted: '+repr(value))
        for index in [0,99]:
            try:onnx_session('unused.onnx','dml',options={'device_id':index})
            except ValueError:pass
            else:raise AssertionError('Absent adapter accepted')
        with patch('titanos.dxgi.adapters',return_value=[dict(adapter,software=True)]):
            try:onnx_session('unused.onnx','dml',options={'device_id':2})
            except ValueError:pass
            else:raise AssertionError('Software renderer accepted as GPU')
        records=runtime_records('onnx')
        device=next(r for r in records if r.get('provider')=='DmlExecutionProvider')
        assert device['uuid'] is None and device['identity_kind']=='dxgi_luid'
        assert device['provider_options']=={'device_id':'2'}
    # Vendor distributions need not install metadata named "onnxruntime".
    with patch('importlib.metadata.version',side_effect=RuntimeError('generic distribution absent')):
        result=runtime_check('onnx','cpu')
        assert result['version'] and not result.get('check',{}).get('error'),result
    with patch('titanos.devices.runtime_records',return_value=[{'runtime_address':'GPU','uuid':'OV-example'}]):
        assert pin_runtime('GPU.0','openvino')=='OV-example'
        try:pin_runtime('GPU.999','openvino')
        except ValueError:pass
        else:raise AssertionError('Missing OpenVINO device pinned')
    import json
    from pathlib import Path
    from titanos.hardware import choose_profile
    profiles=json.loads((Path(__file__).resolve().parents[1]/'tools/profiles.json').read_text())
    host={'os':'windows','arch':'x86_64','nvidia':[],'display_vendors':['amd']}
    assert choose_profile(host,profiles)[0]=='dml'
    assert choose_profile(dict(host,arch='aarch64'),profiles,'dml-arm64')[2]=='dml'
    try:choose_profile(dict(host,nvidia=[{'index':0,'capability':8.9}]),profiles,'dml',gpu=0)
    except ValueError:pass
    else:raise AssertionError('CUDA index silently interpreted as DirectML adapter')
    print('DirectML selector/session/fallback contract passed; no hardware inference claimed by this test')

if __name__=='__main__':main()
