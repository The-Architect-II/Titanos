"""Actual isolated adapter process protocol with fake stdlib worker; no music model."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

from titanos.config import ModelSpec
from titanos.engine import EngineError
from titanos.music_backend import ACE_CODE_REVISION,AceStepModel


ADAPTER=r'''
import argparse,json,wave,struct,os
p=argparse.ArgumentParser()
for name in ["request","output","result","progress"]:p.add_argument("--"+name,required=True)
a=p.parse_args();request=json.load(open(a.request))
contract=request['runtime_contract']
assert contract['device']=='cpu' and os.environ['CUDA_VISIBLE_DEVICES']==''
assert 'HF_TOKEN' not in os.environ and 'TITANOS_API_KEY' not in os.environ
json.dump({"stage":"sampling","completed":1,"total":1},open(a.progress,"w"))
with wave.open(a.output,"wb") as w:
 w.setnchannels(2);w.setsampwidth(2);w.setframerate(24000)
 w.writeframes(b"".join(struct.pack("<hh",i,-i) for i in range(1,2401)))
runtime={'contract_version':1,'backend':'cpu','resolved_device':'cpu','device_uuid':None,
 'visible_devices':[],'identity_source':'parameter.device','torch_version':'fixture-no-torch',
 'visibility':contract['visibility'],'torch_num_threads':contract['threads'],
 'torch_num_interop_threads':contract['interop_threads'],
 'placement':{'requested':contract['placement'],'resolved':'host','device_free_bytes':None,'required_bytes':0,
              'total_resident_bytes':0,'peak_set_bytes':0,'headroom_bytes':0,'accelerate_version':None,
              'basis':'fixture worker; nothing was loaded or measured'},
 'parameter_placements':[{'component':name,'stage':'execute','device':'cpu','dtype':'float32'}
                         for name in ('dit','text_encoder','vae')]}
json.dump({"native_format":"pcm16-stereo","effective":{"seed":request.get("seed"),'dtype':'float32','cpu_offload':False,
 'threads':contract['threads'],'interop_threads':contract['interop_threads']},
 "evidence":{"adapter":"fixture-not-model-placement-proof","runtime":runtime}},open(a.result,"w"))
'''


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
  with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);bundle=root/'bundle';bundle.mkdir()
    components={}
    for role in ('core','dit'):
      directory=bundle/role;directory.mkdir();artifact=directory/'model.safetensors';artifact.write_bytes(role.encode())
      inventory=directory/'component.inventory.json'
      inventory.write_text(json.dumps({'repository':'fixture/'+role,'revision':'r-'+role,
        'files':[{'path':f'{role}/model.safetensors','bytes':artifact.stat().st_size,'sha256':digest(artifact)}]}))
      components[role]={'path':f'{role}/component.inventory.json','sha256':digest(inventory),'revision':'r-'+role,'license':'mit'}
    manifest={'schema_version':1,'family':'ace-step-1.5','variant':'xl-sft','code_revision':ACE_CODE_REVISION,
      'model_revision':'d06de46b4622f781cf07f4a013a67d591ca52819','components':components,
      'capabilities':{'instrumental':True,'song':True,'lyrics':True,'vocal_languages':[],
        'duration_min':10,'duration_max':600,'bpm':True,'key_scale':True,'reference_audio':True},'license':'mit'}
    (bundle/'titanos-music.json').write_text(json.dumps(manifest))
    adapter=root/'adapter.py';adapter.write_text(ADAPTER)
    receipt={'code_revision':ACE_CODE_REVISION,'destination':str(root),'files':{}}
    spec=ModelSpec(path=str(bundle),task='music',backend='ace-step',device='cpu',
      source_repo='ACE-Step/acestep-v15-xl-sft',source_revision=manifest['model_revision'])
    with patch('titanos.music_backend.runtime_receipt',return_value=(receipt,Path(sys.executable),adapter)):
      model=AceStepModel(spec)
      result=model.infer('music',{'prompt':'fixture','mode':'song','lyrics':'words','duration':10,'seed':9})
      assert result['channels']==2 and result['sample_rate']==24000 and result['effective']['seed']==9
      cancel=threading.Event();cancel.set()
      try:model.infer('music',{'prompt':'fixture','mode':'instrumental','duration':10},cancel=cancel)
      except EngineError as exc:assert exc.code=='cancelled'
      else:raise AssertionError('Pre-start music cancellation ignored')
  print('PASS: isolated owned music adapter process publishes stereo metadata and honors cancellation; fake worker only.')
