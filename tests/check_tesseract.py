"""Native OCR contract/process tests with synthetic outputs; never run real OCR."""
import base64
from contextlib import redirect_stdout
import ctypes
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from PIL import Image
from titanos.config import Config, ModelSpec, WorkerSpec
from titanos.engine import Engine, EngineError
from titanos.owned_command import run_owned
from titanos.server import create_app
from titanos.settings import Settings
from titanos.tesseract import TesseractModel, controls, parse_tsv, executable_path


def raises(call, kind=ValueError):
    try: call()
    except kind: return
    raise AssertionError('Expected rejection')


def contract():
    base = dict(path='model', task='image-to-text', backend='tesseract')
    assert ModelSpec(**base).device == 'cpu'
    for changes in [dict(device='cuda'), dict(dtype='float32'), dict(task='chat'), dict(parallelism='tensor_parallel'),
                    dict(trust_remote_code=True), dict(defaults={'max_tokens': 4}), dict(load_options={'x': 1})]:
        raises(lambda: ModelSpec(**{**base, **changes}))
    raises(lambda: Config(models={'ocr': ModelSpec(**base)}, workers={'gpu': WorkerSpec(models=['ocr'], device='cuda')}))
    for data in [{'temperature': 0}, {'max_tokens': 4}, {'prompt': ''}, {'language': '../eng'}, {'psm': 0},
                 {'max_output_bytes': 10**9}, {'char_whitelist': 'a\nb'}]:
        raises(lambda: controls(data))
    assert controls({'psm': 7}, {'psm': 3, 'language': 'eng'})['psm'] == 7
    assert Settings.from_env({'TITANOS_OCR_PSM': '7', 'TITANOS_OCR_MAX_OUTPUT_BYTES': '1024'}).ocr_max_output_bytes == 1024
    raises(lambda: Settings(ocr_psm=12))
    from titanos.cli import main
    fake = MagicMock();fake.__enter__.return_value=fake;fake.ocr.return_value={'text':'synthetic'}
    with patch('titanos.client.Client', return_value=fake), redirect_stdout(io.StringIO()):
        main(['ocr','ocr','input.png','--psm','7','--language','eng','--include-geometry','--deadline-ms','1000'])
    assert fake.ocr.call_args.kwargs == {'psm':7,'language':'eng','include_geometry':True,'timeout':1000}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp);exe = root/'tesseract.exe';exe.write_bytes(b'metadata-only fake')
        with patch('titanos.tesseract.current_settings', return_value=Settings(tesseract_executable=str(exe))):
            raises(lambda: executable_path(root))
        with patch('titanos.tesseract.current_settings', return_value=Settings(tesseract_executable='relative.exe')):
            raises(lambda: executable_path(root))
        with patch('titanos.tesseract.current_settings', return_value=Settings()), patch.object(sys, 'frozen', False, create=True):
            raises(lambda: executable_path(root))
        bundle=root/'bundle/native/tesseract';bundle.mkdir(parents=True);(bundle/('tesseract.exe' if os.name=='nt' else 'tesseract')).write_bytes(b'trusted metadata')
        with patch('titanos.tesseract.current_settings', return_value=Settings()), patch.object(sys, 'frozen', True, create=True), patch.object(sys, '_MEIPASS', str(root/'bundle'), create=True):
            assert executable_path(root/'models').parent == bundle.resolve()


HEADER='level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
TSV=HEADER+'1\t1\t0\t0\t0\t0\t0\t0\t50\t20\t-1\t\n5\t1\t1\t1\t1\t1\t2\t3\t20\t10\t97.4\tMixed\n'


def adapter_and_interfaces():
    calls=[];directories=[]
    def fake_native(argv, directory, env, check, limit):
        check();calls.append(argv);directories.append(directory)
        assert '-l' in argv and argv[argv.index('--oem')+1]=='1'
        assert '--psm' in argv and 'python' not in Path(argv[0]).name.lower()
        assert Image.open(directory/'input.png').size == (50,20)
        (directory/'output.txt').write_bytes('Mixed Case: 42.\r\n'.encode())
        if 'tessedit_create_tsv=1' in argv:(directory/'output.tsv').write_text(TSV,encoding='utf-8')
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);model=root/'model';model.mkdir();(model/'eng.traineddata').write_bytes(b'synthetic data never loaded')
        exe=root/'tesseract.exe';exe.write_bytes(b'metadata-only fake binary')
        settings=Settings(tesseract_executable=str(exe),ocr_psm=7)
        spec=ModelSpec(path=str(model),task='image-to-text',backend='tesseract')
        image=Image.new('RGB',(50,20),'white');stream=io.BytesIO();image.save(stream,format='PNG')
        payload={'model':'ocr','image_base64':base64.b64encode(stream.getvalue()).decode(),'psm':7,'include_geometry':True}
        with patch('titanos.tesseract.current_settings',return_value=settings),patch('titanos.owned_command.run_owned',side_effect=fake_native):
            adapter=TesseractModel(spec);details=adapter.details();assert details['resident_model'] is False
            direct=adapter.infer('image-to-text',{'image':image,'include_geometry':True})
            assert direct['text']=='Mixed Case: 42.' and direct['raw_text']=='Mixed Case: 42.\r\n'
            assert direct['geometry']['rows'][1]['conf']==97.4
            # worker_main passes this callback to Engine.infer for every task.
            # A callback's presence is not a client request for token streaming.
            emitted=[];worker_engine=Engine(Config(models={'ocr':spec}))
            try:
                final=worker_engine.infer('ocr','image-to-text',{'image':image,'include_geometry':True},
                                          emit=lambda part:emitted.append(('chunk',part)))
                assert isinstance(final,dict) and final==direct and emitted==[]
                raises(lambda:worker_engine.infer('ocr','image-to-text',{'image':image,'stream':True},emit=emitted.append),EngineError)
                assert emitted==[]
            finally:worker_engine.close()
            app=create_app(Config(models={'ocr':spec}),output_dir=root/'outputs',settings=settings)
            with TestClient(app,base_url='http://127.0.0.1') as client:
                for route in ['/v1/images/ocr','/v1/extensions/infer']:
                    data=payload if route.endswith('/ocr') else {**payload,'operation':'ocr'}
                    result=client.post(route,json=data);assert result.status_code==200,result.text
                    assert result.json()['text']==direct['text'] and result.json()['geometry']==direct['geometry']
                for extra in [{'temperature':0},{'max_tokens':12},{'prompt':''}]:
                    with patch('titanos.api_plus._image',side_effect=AssertionError('Unsupported request decoded image')):
                        response=client.post('/v1/images/ocr',json={**payload,**extra});assert response.status_code==400,response.text
                schema=client.get('/openapi.json').json();assert 'include_geometry' in schema['components']['schemas']['OCRRequest']['properties']
                from pydantic import TypeAdapter
                from titanos.api_plus import ExtensionRequest
                extension_schema=TypeAdapter(ExtensionRequest).json_schema()
                assert 'include_geometry' in extension_schema['$defs']['OCROperation']['properties']
                assert app.state.mcp is not None
            assert 'torch' not in sys.modules
        assert all(not directory.exists() for directory in directories),'Request temporary directory leaked'
        with patch('titanos.tesseract.current_settings',return_value=settings):
            (model/'eng.traineddata').write_bytes(b'')
            raises(lambda:TesseractModel(spec))
    assert parse_tsv(TSV,50,20)[0]['conf']==-1
    raises(lambda:parse_tsv(TSV.replace('97.4','nan'),50,20))
    raises(lambda:parse_tsv(TSV,10,20))
    with patch('titanos.tesseract.current_settings',return_value=Settings(ocr_max_tsv_rows=1)):
        raises(lambda:parse_tsv(TSV,50,20))
    return len(calls)


def native_process_controls():
    # A real harmless Python child exercises process mechanics, not OCR or a model.
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);env=dict(os.environ)
        run_owned([sys.executable,'-c','print("synthetic process check")'],root,env,lambda:None,4096)
        assert b'synthetic process check' in (root/'stdout.log').read_bytes()
        raises(lambda:run_owned([sys.executable,'-c','import time;print("x"*10000,flush=True);time.sleep(30)'],root,env,lambda:None,1000),EngineError)
        start=time.monotonic()
        def deadline():
            if time.monotonic()-start>.15:raise EngineError('synthetic deadline',504,'deadline_exceeded')
        raises(lambda:run_owned([sys.executable,'-c','import time;time.sleep(30)'],root,env,deadline,4096),EngineError)
        assert time.monotonic()-start<3
        if os.name=='nt':
            kernel=ctypes.WinDLL('kernel32');kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            handle=None
            script="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);Path('child.pid').write_text(str(p.pid));time.sleep(30)"
            start=time.monotonic()
            def cancelled():
                nonlocal handle
                if (root/'child.pid').exists():
                    handle=kernel.OpenProcess(0x100000,False,int((root/'child.pid').read_text()));assert handle
                    raise EngineError('synthetic cancellation',499,'cancelled')
                assert time.monotonic()-start<5
            try:
                raises(lambda:run_owned([sys.executable,'-c',script],root,env,cancelled,4096),EngineError)
                assert handle and kernel.WaitForSingleObject(handle,1000)==0,'Owned descendant survived cancellation'
            finally:
                if handle:kernel.CloseHandle(handle)


if __name__=='__main__':
    contract();count=adapter_and_interfaces();native_process_controls()
    assert 'torch' not in sys.modules
    print(json.dumps({'passed':True,'synthetic_adapter_calls':count,'real_OCR_calls':0,'primary_models_loaded':0,'owned_process_controls':'output budget, deadline, cancellation and descendant cleanup'}))
