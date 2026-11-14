"""Exercise application settings, assets and live registry changes with fake weights."""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from titanos.config import Config, ModelSpec, write_config
from titanos.server import create_app
from titanos.settings import Settings
from titanos.jobs import VideoJobs


def main():
    seen=[]
    class Model:
        def __init__(self,spec):self.spec=spec;self.device='cpu'
        def infer(self,task,data,emit=None,cancel=None):
            seen.append((self.spec.path,task,data.copy()))
            if task=='image':return [Image.new('RGB',(data['width'],data['height']),'green')]
            if task=='speech':return b'RIFF'+b'\0'*40
            return {'text':'ok','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},'finish_reason':'stop'}
    with tempfile.TemporaryDirectory() as tmp,patch('titanos.backends.LoadedModel',Model):
        root=Path(tmp).resolve();a=root/'a';b=root/'b';a.mkdir();b.mkdir()
        registry=root/'models.json'
        cfg=Config(models={'chat':ModelSpec(path=str(a),task='chat',defaults={'max_tokens':512,'temperature':0.1}),
                           'image':ModelSpec(path=str(a),task='image',backend='diffusers')})
        write_config(registry,cfg)
        settings=Settings(max_tokens=65536,default_max_tokens=1024,default_top_p=0.8,max_image_dimension=8192,max_image_pixels=8192**2)
        app=create_app(cfg,output_dir=root/'output',settings=settings,registry_path=registry)
        with TestClient(app,base_url='http://localhost') as c:
            assert c.get('/').status_code==200 and b'Titanos' in c.get('/').content
            assert c.get('/ui/app.js').status_code==200
            assert c.get('/ui/i18n.js').status_code==200
            for asset in ('titanos-wordmark-v2.svg','titanos-honeycomb-t-v2.svg'):
                reply=c.get('/ui/'+asset);assert reply.status_code==200 and reply.headers['content-type'].startswith('image/svg+xml')
            assert c.get('/ui/not-a-studio-asset.svg').status_code==404
            assert c.get('/api/info').json()['settings']['max_tokens']['value']==65536
            assert c.get('/api/models/loaded').json()['data']==[]
            loaded=c.post('/api/models/load',json={'model':'chat'});assert loaded.status_code==200 and loaded.json()['status']=='loaded'
            details=c.post('/api/models/details',json={'model':'chat'});assert details.status_code==200 and details.json()['device']=='cpu'
            unloaded=c.post('/api/models/unload',json={'model':'chat'});assert unloaded.status_code==200
            assert c.get('/api/models/loaded').json()['data']==[]
            payload={'model':'chat','messages':[{'role':'user','content':'hi'}]}
            assert c.post('/v1/chat/completions',json=payload).status_code==200
            assert seen[-1][2]['max_tokens']==512 and seen[-1][2]['temperature']==0.1 and seen[-1][2]['top_p']==0.8
            assert c.post('/v1/chat/completions',json={**payload,'max_tokens':32768,'temperature':1.2}).status_code==200
            assert seen[-1][2]['max_tokens']==32768 and seen[-1][2]['temperature']==1.2
            assert c.post('/v1/chat/completions',json={**payload,'max_tokens':65537}).status_code==422
            assert c.post('/v1/chat/completions',json={**payload,'max_completion_tokens':300}).status_code==200
            assert seen[-1][2]['max_tokens']==300
            before=len(seen)
            body={'task':'image','model':'image','prompt':'green','size':'32x32','steps':1}
            job=c.post('/api/jobs',json=body,headers={'Idempotency-Key':'one'}).json()
            for _ in range(100):
                result=c.get('/api/jobs/'+job['id']).json()
                if result['status']=='completed':break
                time.sleep(0.02)
            assert result['status']=='completed',result
            assert c.post('/api/jobs',json=body,headers={'Idempotency-Key':'one'}).json()['id']==job['id']
            assert len(seen)==before+1
            assert c.post('/api/jobs',json={**body,'prompt':'different'},headers={'Idempotency-Key':'one'}).status_code==409
            asset=c.get(result['result']['data'][0]['url']);assert Image.open(io.BytesIO(asset.content)).size==(32,32)
            assert c.get('/api/jobs').json()['data'][0]['task']=='image'
            assert c.get('/api/video/jobs').json()['data']==[]
            change={'name':'chat','replace':True,'spec':{'path':str(b),'task':'chat'}}
            r=c.post('/api/models/register',json=change);assert r.status_code==200,r.text
            assert c.post('/v1/chat/completions',json=payload).status_code==200
            assert Path(seen[-1][0])==b
            assert c.delete('/api/models/registrations/chat').status_code==200
            assert c.post('/v1/chat/completions',json=payload).status_code==404
            assert b.is_dir()
        reopened=VideoJobs(root/'output'/'jobs',settings=settings)
        assert reopened.get(job['id'])['status']=='completed'
        reopened.close()
        (root/'output'/'jobs'/('a'*32+'.json')).write_text('{')
        recovered=VideoJobs(root/'output'/'jobs',settings=settings)
        assert recovered.get(job['id'])['status']=='completed'
        assert list((root/'output'/'jobs').glob('*.invalid-*'))
        recovered.close()
        secured=create_app(Config(),api_key='fixture-secret',output_dir=root/'secured-output')
        with TestClient(secured,base_url='http://localhost') as client:
            assert client.get('/').status_code==200
            for asset in ('app.js','i18n.js','style.css','retrieval.js','titanos-wordmark-v2.svg','titanos-honeycomb-t-v2.svg'):
                response=client.get('/ui/'+asset)
                assert response.status_code==200,(asset,response.text)
                if asset.endswith('.svg'):assert response.headers['content-type'].startswith('image/svg+xml')
            assert client.get('/ui/not-a-studio-asset.svg').status_code==401
            # keyed mode: the documentation shell, its packaged assets and the schema stay readable without the bearer; ReDoc is not served;
            # an unlisted asset name is not public and meets the bearer check first
            assert client.get('/docs').status_code==200 and 'cdn.jsdelivr.net' not in client.get('/docs').text
            assert client.get('/').headers.get('x-frame-options')=='DENY' and client.get('/docs').headers.get('x-frame-options')=='DENY'   # Studio and docs are never framed by another origin
            assert client.get('/openapi.json').status_code==200
            assert client.get('/docs/assets/swagger-ui.css').headers['content-type'].startswith('text/css')
            assert client.get('/docs/assets/swagger-ui-bundle.js').status_code==200
            assert client.get('/redoc').status_code==401   # not served; keyed mode meets the bearer first (404 keyless: tests/check_docs_offline.py)
            assert client.get('/docs/assets/not-a-docs-asset.js').status_code==401
            assert client.get('/api/info').status_code==401
            assert client.get('/api/info',headers={'Authorization':'Bearer fixture-secret'}).status_code==200
    print(json.dumps({'passed':True,'checks':['app routes','explicit load/details/unload without discovery load','request > model > environment defaults','raised token ceiling',
                     'durable image jobs and idempotency','decoded generated image','live model replacement and removal',
                     'restart persistence and corrupt-record preservation','authenticated deployment public Studio assets and protected APIs'],
                     'scope':'Synthetic models prove control paths; no inference quality or hardware claim'}))

if __name__=='__main__':main()
