"""Different model names must never overwrite each other's saved workers."""
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient
from titanos.config import Config,ModelSpec,WorkerSpec,read_config,write_config
from titanos.server import create_app

A='GPU-11111111-1111-1111-1111-111111111111'
INVENTORY={'devices':[{'id':A,'uuid':A,'present':True,'name':'controlled fixture',
    'visibility':{'status':'visible','constraints':{}},'runtimes':[{'backend':'pytorch','id':A,'uuid':A,
        'runtime_address':'cuda:0','available':True,'identity_kind':'vendor_uuid','identity_strength':'vendor_runtime_uuid'}]}]}

def check():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);model=root/'model';model.mkdir();registry=root/'models.json'
        spec=ModelSpec(path=str(model),task='chat',backend='transformers',device='cpu')
        cfg=Config(models={'a/b':spec,'a-b':spec},workers={
            'studio-a-b-1':WorkerSpec(models=['a/b'],device='cpu')})
        write_config(registry,cfg)
        app=create_app(cfg,registry_path=registry,output_dir=root/'outputs',runtime_catalog=False)
        with patch('titanos.devices.inventory',return_value=INVENTORY),TestClient(app,base_url='http://127.0.0.1') as client:
            def save(name,threads):
                response=client.post('/api/assignments',json={'model':name,'mode':'single','groups':[[A]],'threads':threads})
                assert response.status_code==200,response.text
                return response.json()
            first=save('a-b',2)
            stored=read_config(registry)
            assert stored.workers['studio-a-b-1'].models==['a/b']
            assert stored.workers[first['workers'][0]].models==['a-b']
            second=save('a/b',3)
            assert first['workers']!=second['workers']
            stored=read_config(registry)
            preserved=stored.workers[first['workers'][0]].model_dump()
            assert 'studio-a-b-1' not in stored.workers
            assert stored.workers[second['workers'][0]].threads==3
            again=save('a/b',4)
            assert again['workers']==second['workers']
            assert read_config(registry).workers[first['workers'][0]].model_dump()==preserved
        # A manually occupied generated key must refuse before touching disk.
        occupied='studio-'+hashlib.sha256(b'a-b').hexdigest()+'-1'
        cfg=Config(models={'a/b':spec,'a-b':spec},workers={occupied:WorkerSpec(models=['a/b'])})
        write_config(registry,cfg);before=registry.read_bytes()
        app=create_app(cfg,registry_path=registry,output_dir=root/'outputs2',runtime_catalog=False)
        with patch('titanos.devices.inventory',return_value=INVENTORY),TestClient(app,base_url='http://127.0.0.1') as client:
            response=client.post('/api/assignments',json={'model':'a-b','mode':'single','groups':[[A]]})
            assert response.status_code==409,response.text
            assert registry.read_bytes()==before
    print('PASS: colliding sanitized model names coexist; legacy own assignment migrates; updates preserve other workers; occupied ID refuses without disk mutation. No models/devices executed.')


if __name__=='__main__':check()
