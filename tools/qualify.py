"""Run selected acceptance suites on this machine; preserve failures and provenance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',type=Path,help='Seven-modality checkpoint directory')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--chat',type=Path,help='Chat checkpoint for replica/multi-GPU checks')
    parser.add_argument('--tp-model',type=Path)
    parser.add_argument('--gpus',nargs='+',default=[])
    parser.add_argument('--native-registry',type=Path)
    parser.add_argument('--native-model',default='agent')
    parser.add_argument('--timeout',type=int,default=3600)
    args=parser.parse_args()
    if args.timeout<1:parser.error('timeout must be positive')
    from titanos import __version__
    from titanos.hardware import discover
    from titanos.devices import inventory
    root=Path(__file__).resolve().parents[1]
    out=args.output.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=False)
    packages={}
    for name in ('titanos-engine','torch','transformers','diffusers','onnxruntime','openvino','openvino-genai'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    source_digest=hashlib.sha256()
    for path in sorted((root/'src/titanos').glob('*.py')):
        source_digest.update(path.name.encode());source_digest.update(path.read_bytes())
    report={'run_id':str(uuid.uuid4()),'started_at':time.time(),'version':__version__,
            'hardware':discover(),'packages':packages,'source_sha256':source_digest.hexdigest(),
            'checks':[],'inventory':{},'scope':'Selected models, runtime and tests only; absent hardware is not a pass'}
    for backend in ('system','nvidia','pytorch','openvino','onnx'):
        try:report['inventory'][backend]=inventory(True,backend)
        except Exception as exc:report['inventory'][backend]={'available':False,'reason':str(exc)}
    commands=[('contracts',['tests/check.py','--report',str(out/'contracts.json')]),('hub',['tests/check_hub.py'])]
    if 'torch' in packages:commands.append(('devices',['tests/check_devices.py']))
    commands.append(('job-lifecycle',['tests/check_jobs.py','--report',str(out/'jobs.json')]))
    if args.models:
        commands.append(('modalities',['tests/check.py','--models',str(args.models.resolve()),'--device',args.device,'--report',str(out/'modalities.json')]))
    if args.chat:
        command=['tests/check_parallel.py','--model',str(args.chat.resolve()),'--report',str(out/'parallel.json'),'--require-all']
        if args.gpus:command+=['--gpus',*args.gpus]
        else:command+=['--mode','replicas']
        if args.tp_model:command+=['--tp-model',str(args.tp_model.resolve())]
        commands.append(('parallel',command))
    if args.native_registry:
        commands.append(('native',['tests/check_native.py','--registry',str(args.native_registry.resolve()),'--model',args.native_model,'--report',str(out/'native.json')]))
    if not any((args.models,args.chat,args.native_registry)):
        report['checks'].append({'name':'model-inference','status':'not_requested','reason':'Supply --models, --chat or --native-registry for model execution'})
    for name,arguments in commands:
        started=time.monotonic()
        record={'name':name,'argv':[sys.executable,*arguments]}
        with (out/(name+'.log')).open('w') as log:
            process=None
            try:
                process=subprocess.Popen(record['argv'],cwd=root,stdout=log,stderr=log,start_new_session=os.name!='nt',
                                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
                code=process.wait(timeout=args.timeout)
                record.update(status='passed' if code==0 else 'failed',exit_code=code)
            except subprocess.TimeoutExpired:
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=log,stderr=log)
                else:os.killpg(process.pid,signal.SIGKILL)
                process.wait()
                record.update(status='failed',reason='timeout')
        record['seconds']=round(time.monotonic()-started,3)
        report['checks'].append(record)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['success']=all(x['status']=='passed' for x in report['checks'])
    report['finished_at']=time.time()
    report['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='report.json'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['success'] else 1


if __name__=='__main__':sys.exit(main())
