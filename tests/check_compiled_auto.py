"""Compiled auto selection through real adapters with inert vendor doubles."""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from titanos.compiled import CompiledModel, compiled_placement
from titanos.config import ModelSpec
from titanos.devices import preferred_runtime_device, runtime_records
from titanos.genai_images import GenAIImageModel


GPU = '11111111-1111-1111-1111-111111111111'
NPU = '22222222-2222-2222-2222-222222222222'


class Port:
    def __init__(self, name, dtype='float32', shape=(1,2)):
        self.any_name=name
        self.partial_shape=[NS(is_static=True,get_length=lambda n=n:n) for n in shape]
        self.element_type=NS(to_dtype=lambda:np.dtype(dtype))


class Core:
    def __init__(self, devices, embedding=False, failure=None, observed=None):
        self.available_devices=devices;self.calls=[];self.embedding=embedding
        self.failure=failure;self.observed=observed
    def get_property(self, device, name):
        if name=='SUPPORTED_PROPERTIES':return ['FULL_DEVICE_NAME','DEVICE_UUID','DEVICE_TYPE']
        if name=='FULL_DEVICE_NAME':return 'Fixture '+device
        if name=='DEVICE_TYPE':return 'INTEGRATED' if device.startswith('GPU') else device
        if name=='DEVICE_UUID':return GPU if device.startswith('GPU') else NPU if device.startswith('NPU') else '0'*32
        raise AssertionError(name)
    def read_model(self, path):return object()
    def compile_model(self, model, device, options):
        self.calls.append((device,options))
        if self.failure:raise self.failure
        outer=self
        class Compiled:
            inputs=[Port(n,'int64') for n in ['input_ids','attention_mask']] if outer.embedding else [Port('x')]
            outputs=[Port('last_hidden_state' if outer.embedding else 'y')]
            def get_property(self, name):
                assert name=='EXECUTION_DEVICES'
                return outer.observed if outer.observed is not None else [device]
            def __call__(self, feed):
                value=np.tile(np.array([1,2,3],dtype=np.float32),(1,2,1)) if outer.embedding else np.array([[feed['x'].sum()]],dtype=np.float32)
                return {self.outputs[0]:value}
        return Compiled()


class Tokenizer:
    model_max_length=16
    def __call__(self,texts,**kwargs):return {'input_ids':[[1,2] for _ in texts],'attention_mask':[[1,1] for _ in texts]}
    def pad(self,encoded,**kwargs):return {key:np.asarray(value,dtype=np.int64) for key,value in encoded.items()}


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for n in ['openvino_model.xml','model.onnx']:(self.root/n).write_bytes(b'inert graph bytes')
        context=patch('titanos.devices.installed_runtime_preference',return_value=None)
        context.start();self.addCleanup(context.stop)
    def spec(self, backend='openvino', task='tensor', **kwargs):
        return ModelSpec(path=str(self.root),backend=backend,task=task,**kwargs)
    def vendor(self, core):return patch.dict(sys.modules,{'openvino':NS(Core=lambda:core)})

    def test_gpu_advice_matches_raw_tensor_execution_and_no_cuda_mask_reinterpretation(self):
        core=Core(['CPU','NPU','GPU.0'])
        with self.vendor(core),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            rows=runtime_records('openvino',core=core)
            advice=preferred_runtime_device(rows,backend='openvino',task='tensor')
            model=CompiledModel(self.spec())
            result=model.infer('tensor',{'inputs':{'x':{'dtype':'float32','shape':[1,2],'data':[2,3]}}})
        self.assertEqual(advice['device'],model.device)
        self.assertEqual(core.calls[0][0],'GPU.0')
        self.assertEqual(result['outputs']['y']['data'],[5.0])
        self.assertEqual(model.details()['selection']['uuid'],'OV-'+GPU)
        self.assertEqual(result['execution']['execution_devices'],['GPU.0'])
        self.assertFalse(result['execution']['selection']['kernel_execution_verified'])

    def test_npu_auto_raw_embeddings_with_actual_adapter_pooling(self):
        core=Core(['CPU','NPU'],embedding=True)
        tokenizer=NS(AutoTokenizer=NS(from_pretrained=lambda *a,**k:Tokenizer()))
        with self.vendor(core),patch.dict(sys.modules,{'transformers':tokenizer}):
            model=CompiledModel(self.spec(task='embeddings'))
            value=model.infer('embeddings',{'input':['one','two']})
        self.assertEqual(model.device,'NPU')
        self.assertEqual(value['usage']['prompt_tokens'],4)
        np.testing.assert_allclose(np.linalg.norm(value['vectors'],axis=1),[1,1])
        self.assertEqual(model.details()['selection']['uuid'],'OV-'+NPU)

    def test_genai_chat_npu_selection_reaches_pipeline_without_cpu_retry(self):
        from titanos.genai import GenAIModel
        for name in ('openvino_model.bin','openvino_tokenizer.xml','openvino_detokenizer.xml'):
            (self.root/name).write_bytes(b'inert export bytes')
        constructor=Mock(return_value=NS(get_tokenizer=lambda:object()))
        with self.vendor(Core(['CPU','NPU'])),patch.dict(sys.modules,{'openvino_genai':NS(LLMPipeline=constructor)}):
            model=GenAIModel(self.spec(backend='openvino-genai',task='chat'))
            self.assertEqual(model.device,'NPU')
            self.assertEqual(constructor.call_args.args[1],'NPU')
            constructor.reset_mock();constructor.side_effect=RuntimeError('inert unsupported NPU graph')
            with self.assertRaisesRegex(RuntimeError,'unsupported NPU graph'):
                GenAIModel(self.spec(backend='openvino-genai',task='chat'))
            self.assertEqual(constructor.call_count,1)
            self.assertEqual(constructor.call_args.args[1],'NPU')

    def test_explicit_cpu_and_installed_pin_precede_auto_but_missing_pins_refuse(self):
        core=Core(['CPU','GPU'])
        with self.vendor(core):
            model=CompiledModel(self.spec(device='CPU'))
            self.assertEqual(model.device,'CPU')
            with patch('titanos.devices.installed_runtime_preference',return_value={'backend':'openvino','device':'CPU'}):
                self.assertEqual(CompiledModel(self.spec()).device,'CPU')
            with patch('titanos.devices.installed_runtime_preference',return_value={'backend':'openvino','device':'GPU.0'}):
                self.assertEqual(CompiledModel(self.spec()).device,'GPU')
            with patch('titanos.devices.installed_runtime_preference',return_value={'backend':'openvino','device':'GPU.9'}):
                with self.assertRaisesRegex(ValueError,'pin'):CompiledModel(self.spec())
            self.assertEqual(CompiledModel(self.spec(device='OV-'+GPU)).device,'GPU')
            with self.assertRaises(ValueError):CompiledModel(self.spec(device='GPU.9'))

    def test_selected_accelerator_failure_or_cpu_execution_is_not_retried(self):
        for failure, observed in [(RuntimeError('inert unsupported graph'),None),(None,['CPU'])]:
            core=Core(['CPU','GPU.0'],failure=failure,observed=observed)
            with self.vendor(core),self.assertRaises((ValueError,RuntimeError)):
                CompiledModel(self.spec())
            self.assertEqual([device for device,_ in core.calls],['GPU.0'])

    def test_npu_image_is_excluded_explicitly_while_gpu_auto_reaches_pipeline(self):
        export={'family':'sd','image_edit':False}
        constructor=Mock(return_value=NS())
        genai=NS(Text2ImagePipeline=constructor)
        for devices, expected in [(['CPU','NPU','GPU.0'],'GPU.0'),(['CPU','NPU'],'CPU')]:
            core=Core(devices)
            with self.vendor(core),patch.dict(sys.modules,{'openvino_genai':genai}), \
                 patch('titanos.genai_images.inspect_export',return_value=export):
                model=GenAIImageModel(self.spec(backend='openvino-genai',task='image'))
                self.assertEqual(model.device,expected)
                self.assertEqual(constructor.call_args.args[1],expected)
                self.assertEqual(model.selection['configured_device'],'auto')
        with self.vendor(Core(['CPU','NPU'])),patch.dict(sys.modules,{'openvino_genai':genai}), \
             patch('titanos.genai_images.inspect_export',return_value=export):
            before=constructor.call_count
            with self.assertRaisesRegex(ValueError,'NPU image execution'):
                GenAIImageModel(self.spec(backend='openvino-genai',task='image',device='OV-'+NPU))
            self.assertEqual(constructor.call_count,before)
        with self.assertRaisesRegex(ValueError,'NPU static-shape'):
            self.spec(backend='openvino-genai',task='image',device='NPU')

    def test_directml_auto_binds_nonzero_hardware_adapter_and_never_picks_software(self):
        calls=[]
        class SessionOptions:
            def add_session_config_entry(self,*args):pass
        def session(path,sess_options,providers):
            calls.append(providers)
            return NS(disable_fallback=lambda:None,get_providers=lambda:[providers[0][0]],
                      get_inputs=lambda:[NS(name='x',shape=[1,2],type='tensor(float)')],
                      get_outputs=lambda:[NS(name='y')],run=lambda _,feed:[np.array([[5]],dtype=np.float32)])
        ort=NS(SessionOptions=SessionOptions,ExecutionMode=NS(ORT_SEQUENTIAL='sequential'),
               get_available_providers=lambda:['CPUExecutionProvider','DmlExecutionProvider'],InferenceSession=session)
        hardware=[{'device_id':0,'name':'software','software':True,'luid':'00000000:00000000'},
                  {'device_id':2,'name':'Fixture GPU','software':False,'luid':'11111111:22222222'}]
        with patch.dict(sys.modules,{'onnxruntime':ort}),patch('titanos.dxgi.adapters',return_value=hardware), \
             patch('titanos.compiled.platform.system',return_value='Windows'):
            model=CompiledModel(self.spec(backend='onnx'))
            self.assertEqual(model.device,'dml')
            self.assertEqual(calls[-1], [('DmlExecutionProvider',{'device_id':'2'})])
            with self.assertRaisesRegex(ValueError,'adapter'):
                CompiledModel(self.spec(backend='onnx',load_options={'device_id':'0'}))
            with patch('titanos.devices.installed_runtime_preference',return_value={'backend':'onnx','device':'cpu'}):
                self.assertEqual(CompiledModel(self.spec(backend='onnx')).device,'cpu')

    def test_changed_directml_luid_is_refused_before_session_creation(self):
        from titanos.compiled import onnx_session
        options=NS(add_session_config_entry=lambda *a:None)
        construct=Mock()
        ort=NS(SessionOptions=lambda:options,ExecutionMode=NS(ORT_SEQUENTIAL=0),
               get_available_providers=lambda:['DmlExecutionProvider'],InferenceSession=construct)
        with patch.dict(sys.modules,{'onnxruntime':ort}),patch('titanos.compiled.platform.system',return_value='Windows'), \
             patch('titanos.dxgi.adapters',return_value=[{'device_id':2,'software':False,'luid':'new'}]):
            with self.assertRaisesRegex(ValueError,'identity changed'):
                onnx_session('inert.onnx','dml',options={'device_id':'2'},expected_adapter={'luid':'old'})
        construct.assert_not_called()

    def test_other_onnx_provider_selection_remains_provider_not_hardware_proof(self):
        ort=NS(get_available_providers=lambda:['CPUExecutionProvider','QNNExecutionProvider'])
        with patch.dict(sys.modules,{'onnxruntime':ort}):value=compiled_placement('onnx','auto')
        self.assertEqual(value['device'],'qnn')
        self.assertIsNone(value['selection']['uuid'])
        self.assertFalse(value['selection']['kernel_execution_verified'])


if __name__=='__main__':unittest.main(verbosity=2)
