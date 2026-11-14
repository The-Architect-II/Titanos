"""Typed Studio assignment seam over existing WorkerSpec; no worker/model process starts."""
from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from titanos.config import Config,ModelSpec,read_config,write_config
from titanos.server import create_app

A='GPU-11111111-1111-1111-1111-111111111111'
B='GPU-22222222-2222-2222-2222-222222222222'
ROWS=[{'id':identity,'uuid':identity,'name':'RTX 4090 '+label,'runtime_address':'cuda:'+str(index),'present':True,'memory_mib':24564,
       'architecture':'8.9','visibility':{'status':'visible','constraints':{}},'discovery_status':'current',
       'runtimes':[{'backend':'pytorch','id':identity,'uuid':identity,'runtime_address':'cuda:'+str(index),
                    'available':True,'identity_kind':'vendor_uuid','identity_strength':'vendor_runtime_uuid'}]}
      for index,(identity,label) in enumerate([(A,'A'),(B,'B')])]

if __name__=='__main__':
  with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);registry=root/'models.json';model=root/'model';model.mkdir()
    cfg=Config(models={'chat':ModelSpec(path=str(model),task='chat',backend='transformers')});write_config(registry,cfg)
    app=create_app(cfg,registry_path=registry,output_dir=root/'outputs',runtime_catalog=False)
    with patch('titanos.devices.inventory',return_value={'backend':'nvidia','devices':ROWS}),TestClient(app,base_url='http://127.0.0.1') as client:
      choices=client.get('/api/assignments/devices').json()['data']
      assert choices[0]['id']==A and choices[0]['label']=='RTX 4090 A' and choices[0]['stable'] is True
      payload={'model':'chat','mode':'model_sharding','groups':[[A,B]],'tensor_parallel_size':2}
      saved=client.post('/api/assignments',json=payload);assert saved.status_code==200,saved.text
      assignment=saved.json();assert assignment['mode']=='model_sharding' and assignment['groups']==[[A,B]]
      stored=read_config(registry);worker=stored.workers[assignment['workers'][0]];assert worker.gpus==[A,B] and worker.mode=='model_sharding'
      assert client.get('/api/assignments/chat').json()['groups']==[[A,B]]
      missing=client.post('/api/assignments',json={**payload,'groups':[[A,'GPU-99999999-9999-9999-9999-999999999999']]})
      assert missing.status_code==409 and 'unavailable' in missing.text.lower()
      ordinal=client.post('/api/assignments',json={**payload,'groups':[['0',B]]});assert ordinal.status_code==422
      replicas=client.post('/api/assignments',json={'model':'chat','mode':'single','groups':[[A],[B]],'tensor_parallel_size':2})
      assert replicas.status_code==200 and replicas.json()['replicas']==2
      stored=read_config(registry);assert sorted(stored.workers)==sorted(replicas.json()['workers']) and len(stored.workers)==2
  print('PASS: typed stable-ID assignment saves sharding and distinct replicas, rejects ordinal and missing devices; no worker/model execution.')
