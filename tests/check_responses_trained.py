"""Bounded parity check with a caller-supplied local trained checkpoint.

python tests/check_responses_trained.py --model-dir /path/to/checkpoint --output /new/evidence
Requires the models extra and never downloads weights. CPU is the default;
opt into CUDA with --device cuda:0 and set CUDA_VISIBLE_DEVICES to the chosen GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
args=parser.parse_args()
root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
model=args.model_dir.resolve();weights=model/'model.safetensors'
assert weights.is_file(), 'Use the known safetensors checkpoint; no download or legacy pickle fallback'
with weights.open('rb') as stream:weight_hash=hashlib.file_digest(stream,'sha256').hexdigest()
if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  HF_HOME=str(root/'hf-cache'),TORCH_HOME=str(root/'torch-cache'),
                  TITANOS_HOME=str(root/'home'),TITANOS_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
if args.device.startswith('cuda'):
    assert torch.cuda.is_available(), 'Explicit CUDA request requires an available device'
import httpx,uvicorn,titanos
from titanos.config import Config,ModelSpec
from titanos.server import create_app
from titanos.settings import Settings

def save(name,value):(root/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
def output_text(response):
    return ''.join(part['text'] for item in response['output'] for part in item['content'] if part['type']=='output_text')

cfg=Config(models={'trained-check':ModelSpec(path=str(model),task='chat',device=args.device,dtype='float32',
                                      defaults={'max_output_tokens':24,'temperature':0})})
app=create_app(cfg,output_dir=root/'assets',settings=Settings(threads=1,log_level='warning'))
sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen();port=sock.getsockname()[1]
server=uvicorn.Server(uvicorn.Config(app,log_level='warning'))
thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
started=time.monotonic()
try:
    deadline=time.monotonic()+10
    while not server.started:
        assert thread.is_alive() and time.monotonic()<deadline
        time.sleep(.01)
    prompt='Reply with one short sentence describing a friendly robot.'
    with httpx.Client(base_url=f'http://127.0.0.1:{port}',trust_env=False,timeout=60) as client:
        baseline=client.post('/v1/chat/completions',json={'model':'trained-check','messages':[{'role':'user','content':prompt}]})
        baseline.raise_for_status();baseline=baseline.json();save('chat-baseline.json',baseline)
        response=client.post('/v1/responses',json={'model':'trained-check','input':prompt})
        response.raise_for_status();response=response.json();save('responses.json',response)
        streamed=client.post('/v1/responses',json={'model':'trained-check','input':prompt,'stream':True})
        streamed.raise_for_status();(root/'responses.sse').write_text(streamed.text,encoding='utf-8')
        events=[json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith('data: ')]
        final=events[-1]['response']
        expected=baseline['choices'][0]['message']['content']
        assert expected and output_text(response)==expected and output_text(final)==expected
        assert ''.join(e['delta'] for e in events if e['type']=='response.output_text.delta')==expected
        expected_status='incomplete' if baseline['choices'][0]['finish_reason']=='length' else 'completed'
        assert response['status']==final['status']==expected_status
        assert response['max_output_tokens']==final['max_output_tokens']==24
        for value in [response,final]:
            assert value['usage']['input_tokens']==baseline['usage']['prompt_tokens']
            assert value['usage']['output_tokens']==baseline['usage']['completion_tokens']<=24
        details=client.post('/api/models/details',json={'model':'trained-check'});details.raise_for_status()
        details=details.json();assert details['device']==args.device
        save('model-details.json',details)
        parameter_devices=sorted({str(p.device) for p in app.state.engine.target.runtime.model.parameters()})
        assert parameter_devices==[args.device], 'Every model parameter must be on the requested device'
        if args.device=='cpu':
            assert not torch.cuda.is_initialized(), 'CPU check must not initialize CUDA'
        else:
            assert torch.cuda.is_initialized() and torch.cuda.memory_allocated(0)>0
        save('result.json',{'status':'PASSED','version':titanos.__version__,'device':args.device,'torch':torch.__version__,
             'threads':torch.get_num_threads(),'weights_sha256':weight_hash,'text':expected,
             'chat_responses_and_sse_text_identical':True,'token_usage_identical':True,'cuda_initialized':torch.cuda.is_initialized(),
             'parameter_devices':parameter_devices,'visible_gpus':details['visible_gpus'],
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(0) if args.device.startswith('cuda') else 0,
             'request_count':3,'elapsed_seconds':time.monotonic()-started,
             'source_runtime_only':True,'standalone_binary_qualification':False})
finally:
    server.should_exit=True;thread.join(10);sock.close()
    save('cleanup.json',{'server_stopped':not thread.is_alive(),'cuda_initialized':torch.cuda.is_initialized()})
    assert not thread.is_alive()
print(json.dumps(json.loads((root/'result.json').read_text())))
