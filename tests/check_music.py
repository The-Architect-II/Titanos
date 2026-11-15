"""Typed music job, stereo publication and backend boundaries with a fake adapter."""
import io
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import wave

import numpy as np
from fastapi.testclient import TestClient

from titanos.config import Config,ModelSpec
from titanos.media import write_music_wav
from titanos.server import create_app


class FakeMusic:
    def __init__(self,spec):self.spec=spec;self.device='cpu'
    def infer(self,task,data,emit=None,cancel=None):
        assert task=='music' and data['mode']=='song' and data['lyrics']=='original words'
        from titanos.jobs import _native_generation_observer
        job_id,observe=_native_generation_observer.get();observe({'job_id':job_id,'backend':'ace-step','stage':'sampling','completed':1,'total':2})
        samples=np.stack([np.linspace(-.2,.2,2400,dtype=np.float32),np.linspace(.2,-.2,2400,dtype=np.float32)])
        return {'samples':samples,'sample_rate':24000,'native_format':'float32-planar',
                'effective':{'duration':.1,'seed':7},'evidence':{'family':'ace-step-1.5'}}
    def close(self):pass


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);model=root/'bundle';model.mkdir()
        spec=ModelSpec(path=str(model),task='music',backend='ace-step',device='cpu',
                       source_repo='ACE-Step/acestep-v15-xl-sft',source_revision='d06de46b4622f781cf07f4a013a67d591ca52819')
        try:ModelSpec(path=str(model),task='music',backend='transformers')
        except ValueError:pass
        else:raise AssertionError('Music accepted an unrelated backend')
        raw=write_music_wav(np.zeros((2,32),dtype=np.float32),48000)
        with wave.open(io.BytesIO(raw),'rb') as stream:assert stream.getnchannels()==2 and stream.getframerate()==48000
        app=create_app(Config(models={'music':spec}),output_dir=root/'outputs',runtime_catalog=False)
        with patch('titanos.music_backend.AceStepModel',FakeMusic),TestClient(app,base_url='http://127.0.0.1') as client:
            payload={'task':'music','model':'music','prompt':'optimistic indie pop','mode':'song','lyrics':'original words','duration':30,'seed':7}
            submitted=client.post('/api/jobs',json=payload,headers={'Idempotency-Key':'music-one'})
            assert submitted.status_code==202,submitted.text
            job=submitted.json()
            for _ in range(200):
                job=client.get('/api/jobs/'+job['id']).json()
                if job['status'] in {'completed','failed','cancelled'}:break
                time.sleep(.01)
            assert job['status']=='completed',job
            result=job['result'];assert result['channels']==2 and result['sample_rate']==24000 and result['format']=='wav'
            assert job['native_generation']['stage']=='sampling' and job['native_generation']['job_result_disposition']=='published'
            audio=client.get(result['url']);assert audio.status_code==200
            with wave.open(io.BytesIO(audio.content),'rb') as stream:assert stream.getnchannels()==2
            repeat=client.post('/api/jobs',json=payload,headers={'Idempotency-Key':'music-one'}).json();assert repeat['id']==job['id']
            bad=client.post('/api/jobs',json={**payload,'stems':True});assert bad.status_code==422
    print('PASS: first-class typed music job publishes finite stereo WAV with metadata/idempotency; unsupported controls reject; fake adapter only.')
