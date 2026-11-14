"""Persistent job transitions plus an actual tiny video job and decoded artifact."""
import argparse
import json
from pathlib import Path
import tempfile
import threading
import time
from fastapi.testclient import TestClient
from titanos.jobs import VideoJobs
from titanos.config import Config,ModelSpec
from titanos.server import create_app


def until(call, terminal):
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        value=call()
        if value['status'] in terminal:return value
        time.sleep(.02)
    raise AssertionError('Job did not reach terminal state')


def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path);p.add_argument('--report',type=Path)
    args=p.parse_args();checks=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        store=VideoJobs(root/'queue')
        ready=threading.Event()
        def blocking(cancel):
            ready.set();cancel.wait(5)
            return {'ignored':'cancelled result'}
        job=store.submit('video',blocking);assert ready.wait(2)
        store.cancel(job['id']);assert until(lambda:store.get(job['id']),{'cancelled'})['status']=='cancelled'
        good=store.submit('video',lambda cancel:{'artifact':'done'})
        assert until(lambda:store.get(good['id']),{'completed'})['result']=={'artifact':'done'}
        interrupted={'id':'a'*32,'status':'running','model':'video','created_at':time.time()}
        store.write(interrupted);store.close()
        reopened=VideoJobs(root/'queue')
        assert reopened.get(good['id'])['status']=='completed'
        assert reopened.get(interrupted['id'])['status']=='interrupted'
        reopened.close();checks.append('cancellation, persisted completion, restart interruption without replay')
        if args.video:
            cfg=Config(models={'video':ModelSpec(path=str(args.video.resolve()),task='video',backend='diffusers',device='cpu',dtype='float32',defaults={'max_sequence_length':16})})
            with TestClient(create_app(cfg,output_dir=root/'outputs'),base_url='http://localhost') as c:
                r=c.post('/api/video/jobs',json={'model':'video','prompt':'A bird','size':'64x64','steps':1,'frames':5,'fps':4})
                assert r.status_code==202,r.text
                identifier=r.json()['id']
                done=until(lambda:c.get('/api/video/jobs/'+identifier).json(),{'completed','failed'})
                assert done['status']=='completed',done
                artifact=c.get(done['result']['url']);assert artifact.status_code==200
                path=root/'video.mp4';path.write_bytes(artifact.content)
                import imageio.v3 as iio
                frames=list(iio.imiter(path));assert len(frames)==5
                checks.append('real asynchronous diffusion video job; downloaded MP4 decodes to five frames')
    result={'success':True,'checks':checks}
    if args.report:args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
