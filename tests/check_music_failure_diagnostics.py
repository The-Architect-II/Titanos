"""A failed owned music adapter leaves a bounded, secret-safe diagnostic behind: the exit code and redacted stderr tail are captured
before the request directory is deleted, retained under TITANOS_HOME/outputs/music-failures, and the API-visible error names only the
exit code and the diagnostic's relative file name. Synthetic stdlib child processes only; no ACE, torch, GPU or model."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from titanos.config import ModelSpec
from titanos.engine import EngineError
from titanos.music_backend import ACE_CODE_REVISION,FAILURE_DIAGNOSTIC_BYTES,AceStepModel
from titanos.owned_command import STDERR_TAIL_BYTES,redact

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'/'matrix_kit'))
import run_cells  # noqa: E402

REFUSAL='placement_refused_before_load: resident placement: device free 1024 < required 5 = 1 resident bytes'
SECRET='SECRET123456789012345678901234567890'
SHA='f6940ca9e5c6530cb65371563f64a5034a75a9a3a8b8b4d6e64ea2d1188bee3a'

PREAMBLE=r'''
import argparse,json,sys
p=argparse.ArgumentParser()
for name in ["request","output","result","progress"]:p.add_argument("--"+name,required=True)
a=p.parse_args();request=json.load(open(a.request))
json.dump({"stage":"loading","completed":0,"total":1},open(a.progress,"w"))
'''
ADAPTER_FAIL=PREAMBLE+r'''
sys.stderr.write("api_key='''+SECRET+r'''\n")
sys.stderr.write("Traceback (most recent call last):\n  File \"adapter.py\", line 1, in <module>\nValueError: synthetic adapter failure\n")
sys.exit(3)
'''
ADAPTER_RAISE=PREAMBLE+r'''
raise RuntimeError("synthetic loader exception with absolute paths in the traceback")
'''
ADAPTER_BIG=PREAMBLE+r'''
for i in range(3000):sys.stderr.write(f"line {i:05d} of a very long stderr stream\n")
sys.stderr.write("last line marker\n")
sys.exit(2)
'''
ADAPTER_REFUSE=PREAMBLE+r'''
json.dump({"effective":{},"native_format":"pcm16-wav","evidence":{"status":"failed","runtime":{"placement_refused":{"device_free_bytes":1024}},
 "error":{"type":"ValueError","message":"'''+REFUSAL+r'''"}}},open(a.result,"w"))
sys.stderr.write("'''+REFUSAL+r'''\n")
sys.exit(1)
'''


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def failure_dir():return Path(os.environ['TITANOS_HOME'])/'outputs'/'music-failures'


def fail(model,**request):
    try:model.infer('music',{'prompt':'fixture prompt text that must never be retained','mode':'instrumental','duration':10,'seed':7,'steps':4,**request})
    except EngineError as exc:return exc
    raise AssertionError('the failing adapter did not raise')


if __name__=='__main__':
  # the redaction itself: printable only, credential assignments and long opaque runs replaced, a sha256 kept, bounded
  text='Authorization: Bearer abc\nTOKEN = xyz\x00\x1b[31m\nsha '+SHA+'\nkey '+'A1'*20+' ok\nname '+'z'*40+'\n'+'z9'*5000
  out=redact(text)
  assert 'Bearer' not in out and 'xyz' not in out and 'Authorization=<redacted>' in out and 'TOKEN=<redacted>' in out,out[:200]
  assert SHA in out and 'A1'*20 not in out and 'z9'*16 not in out and '\x00' not in out and '\x1b' not in out and len(out)<=STDERR_TAIL_BYTES,out[:200]
  assert 'name '+'z'*40 in out and out.endswith('<redacted>'),out[-120:]  # an identifier-shaped run is kept; the opaque flood collapses
  # the credential VALUE goes in every common shape, not just the scheme word: short bearer token, JSON/dict-quoted keys,
  # CLI arguments, a quoted value with a space, a bare Bearer JWT (every segment, including short ones)
  out=redact('Authorization: Bearer abc123def456ghi')
  assert out=='Authorization=<redacted>' and 'abc123def456ghi' not in out,out
  out=redact('{"api_key": "shortkey12345", "token": "tok_1234567890"}')
  assert 'shortkey12345' not in out and 'tok_1234567890' not in out and 'api_key=<redacted>' in out and 'token=<redacted>' in out,out
  out=redact('--api-key hunter2secret --token abc --password pw1')
  assert 'hunter2secret' not in out and ' abc' not in out and 'pw1' not in out and out.startswith('--api-key <redacted>'),out
  out=redact('password="p@ss word" next')
  assert 'p@ss' not in out and 'ss word' not in out and out=='password=<redacted> next',out
  out=redact('Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc\nheader eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig1')
  assert 'eyJzdWIiOiIxIn0' not in out and 'eyJhbGci' not in out and 'sig1' not in out,out
  # diagnostic value kept: an uppercase sha256, identifier-shaped long runs (letters/underscores) and labels that are not credentials
  out=redact('SHA '+SHA.upper()+' torch.nn.modules.activation.MultiheadAttention_forward_impl_internal max_tokens: 512')
  assert SHA.upper() in out and 'MultiheadAttention_forward_impl_internal' in out and 'max_tokens: 512' in out,out
  # absolute paths reduce to their final component (Windows and POSIX); relative names survive. The Windows path is a synthetic
  # user-profile fixture (account `example`, allowlisted in tools/public_export.json); no real account name is written here.
  out=redact(r'  File "C:\Users\example\AppData\Local\Temp\tmpm4trjas8\child.py", line 5'+'\n/home/example/venv/lib/site-packages/torch/x.py:3 adapter.py ok')
  assert out=='  File "<path>/child.py", line 5\n<path>/x.py:3 adapter.py ok',out
  with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);bundle=root/'bundle';bundle.mkdir();home=root/'home';home.mkdir()
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
    adapter=root/'adapter.py'
    receipt={'code_revision':ACE_CODE_REVISION,'destination':str(root),'files':{}}
    spec=ModelSpec(path=str(bundle),task='music',backend='ace-step',device='cpu',
      source_repo='ACE-Step/acestep-v15-xl-sft',source_revision=manifest['model_revision'])
    with patch.dict(os.environ,{'TITANOS_HOME':str(home)}),patch('titanos.music_backend.runtime_receipt',return_value=(receipt,Path(sys.executable),adapter)):
      model=AceStepModel(spec)
      # (a) a nonzero adapter exit: generic bounded message with the exit code and the relative diagnostic name; the file keeps the
      # traceback without the credential, never the prompt, and stays within its byte bound
      adapter.write_text(ADAPTER_FAIL)
      exc=fail(model)
      assert exc.code=='native_process_failed' and exc.status==502,(exc.code,exc.status)
      message=str(exc)
      assert message.startswith('Music adapter process exited unsuccessfully (exit code 3; diagnostic outputs/music-failures/') and message.endswith('.json)'),message
      assert 'synthetic adapter failure' not in message and SECRET[:10] not in message and str(home) not in message,message
      files=sorted(failure_dir().glob('*.json'))
      assert len(files)==1 and files[0].name in message,(files,message)
      assert files[0].stat().st_size<=FAILURE_DIAGNOSTIC_BYTES
      raw=files[0].read_text(encoding='utf-8');body=json.loads(raw)
      assert body['returncode']==3 and 'synthetic adapter failure' in body['stderr_tail'] and 'ValueError' in body['stderr_tail'],body
      assert SECRET[:10] not in raw and 'api_key=<redacted>' in body['stderr_tail'],body['stderr_tail']
      assert 'fixture prompt' not in raw and str(root) not in raw and str(home) not in raw,raw
      assert body['request']=={'duration':10.0,'steps':4,'seed':7,'mode':'instrumental'},body['request']
      assert body['contract']['device']=='cpu' and body['contract']['requested_device']=='cpu' and body['runner']=={'executable':Path(sys.executable).name,'adapter':'adapter.py'},body
      assert body['progress']=={'stage':'loading','completed':0,'total':1} and body['result_error'] is None,body
      assert getattr(exc,'returncode',None)==3 and exc.diagnostic==files[0].name
      # (a2) a real interpreter traceback: the exception type/message and the adapter's basename survive, the absolute temp paths
      # of the adapter and the interpreter do not (reduced to <path>/<name>)
      adapter.write_text(ADAPTER_RAISE)
      exc=fail(model)
      assert 'exit code 1' in str(exc),str(exc)
      tail=exc.stderr_tail
      assert 'RuntimeError: synthetic loader exception' in tail and 'Traceback (most recent call last)' in tail and '"<path>/adapter.py"' in tail,tail
      assert str(root) not in tail and str(root.parent) not in tail and str(Path(sys.executable).parent) not in tail,tail
      files=sorted(failure_dir().glob('*.json'));assert len(files)==2
      raised=[json.loads(p.read_text(encoding='utf-8')) for p in files if p.name in str(exc)][0]
      assert raised['returncode']==1 and raised['stderr_tail']==tail and raised['result_error'] is None,raised
      assert str(root) not in json.dumps(raised) and 'fixture prompt' not in json.dumps(raised)
      # (b) a flood of stderr: the retained tail is bounded and keeps the end of the stream
      adapter.write_text(ADAPTER_BIG)
      exc=fail(model)
      assert 'exit code 2' in str(exc),str(exc)
      assert len(exc.stderr_tail)<=STDERR_TAIL_BYTES and 'last line marker' in exc.stderr_tail and 'line 00000' not in exc.stderr_tail
      files=sorted(failure_dir().glob('*.json'));assert len(files)==3
      big=[json.loads(p.read_text(encoding='utf-8')) for p in files if p.name in str(exc)][0]
      assert big['returncode']==2 and len(big['stderr_tail'])<=STDERR_TAIL_BYTES and 'last line marker' in big['stderr_tail']
      assert all(p.stat().st_size<=FAILURE_DIAGNOSTIC_BYTES for p in files)
      # (c) a pre-load placement refusal still reaches the caller verbatim (503 placement_refused) and is retained too
      adapter.write_text(ADAPTER_REFUSE)
      exc=fail(model)
      assert (str(exc),exc.code,exc.status)==(REFUSAL,'placement_refused',503),(str(exc),exc.code,exc.status)
      assert exc.__cause__.code=='native_process_failed' and exc.__cause__.returncode==1
      files=sorted(failure_dir().glob('*.json'));assert len(files)==4
      refusals=[json.loads(p.read_text(encoding='utf-8')) for p in files]
      refusal=[b for b in refusals if b['result_error'] is not None]
      assert len(refusal)==1 and refusal[0]['returncode']==1 and refusal[0]['result_error']=={'type':'ValueError','message':REFUSAL} and REFUSAL in refusal[0]['stderr_tail'],refusal
      assert model.placement is None and model.device is None and model.last_generation is None  # a failed or refused run is never placement evidence
      # a diagnostic that cannot be written never masks the failure: the message says so and keeps the exit code
      adapter.write_text(ADAPTER_FAIL)
      with patch('titanos.music_backend.home',return_value=root/'missing'/'x'/'file.txt'/'not-a-dir'):
        (root/'missing'/'x').mkdir(parents=True);(root/'missing'/'x'/'file.txt').write_text('occupied')
        exc=fail(model)
      assert str(exc)=='Music adapter process exited unsuccessfully (exit code 3; diagnostic_unavailable)' and exc.diagnostic is None,str(exc)
      assert len(list(failure_dir().glob('*.json')))==4
      # ... including when home() itself raises (pathlib's RuntimeError for an unresolvable home directory, or anything else):
      # the child's 502 stays a 502 and a refusal stays a verbatim 503, never a 500 from the retention path
      with patch('titanos.music_backend.home',side_effect=RuntimeError('Could not determine home directory.')):
        exc=fail(model)
        assert (str(exc),exc.code,exc.status)==('Music adapter process exited unsuccessfully (exit code 3; diagnostic_unavailable)','native_process_failed',502),str(exc)
        adapter.write_text(ADAPTER_REFUSE)
        exc=fail(model)
        assert (str(exc),exc.code,exc.status)==(REFUSAL,'placement_refused',503),(str(exc),exc.code,exc.status)
      assert len(list(failure_dir().glob('*.json')))==4
    # (d) matrix kit: a failed attempt's exercise carries the cell root's retained diagnostics (newest first, bounded, at most four);
    # a pass is untouched
    cell=root/'cell';folder_=cell/'outputs'/'music-failures';folder_.mkdir(parents=True)
    for stamp,code in (('20260925T010000Z-1-aaaaaaaa',3),('20260925T020000Z-1-bbbbbbbb',1)):
      utc=f'2026-09-25T{stamp[9:11]}:00:00+00:00'
      (folder_/f'{stamp}.json').write_text(json.dumps({'schema':1,'utc':utc,'returncode':code,'stderr_tail':'ValueError: synthetic',
        'file':'forged.json','problem':'forged problem'}),encoding='utf-8')
    (folder_/'20260925T000000Z-1-cccccccc.json').write_text('{'+'"x":"'+'y'*FAILURE_DIAGNOSTIC_BYTES+'"}',encoding='utf-8')
    (folder_/'notes.txt').write_text('ignored',encoding='utf-8')
    rows=run_cells.failure_diagnostics(cell)
    assert [r['file'] for r in rows]==['20260925T020000Z-1-bbbbbbbb.json','20260925T010000Z-1-aaaaaaaa.json','20260925T000000Z-1-cccccccc.json'],rows
    assert rows[0]['returncode']==1 and rows[1]['returncode']==3 and 'problem' in rows[2] and 'x' not in rows[2],rows
    # a record's own 'file'/'problem' keys never overwrite the row markers; they are kept aside; without `since` no attribution tag
    assert 'problem' not in rows[0] and rows[0]['record_reserved_keys']=={'file':'forged.json','problem':'forged problem'} and 'attributed' not in rows[0],rows[0]
    exercise=run_cells.attach_failure_diagnostics({'state':'fail','error':'RuntimeError: music job failed','steps':[]},cell)
    assert len(exercise['diagnostics'])==3 and json.dumps(exercise),exercise
    # with the attempt's utc_start, rows from earlier attempts/runs in the reused cell root are tagged as prior evidence
    exercise=run_cells.attach_failure_diagnostics({'state':'fail','error':'x','steps':[]},cell,since='2026-09-25T01:30:00+00:00')
    tags={r['file']:r['attributed'] for r in exercise['diagnostics']}
    assert tags['20260925T020000Z-1-bbbbbbbb.json'] is True and tags['20260925T010000Z-1-aaaaaaaa.json'] is False,tags
    assert isinstance(tags['20260925T000000Z-1-cccccccc.json'],bool),tags  # no record utc: falls back to the file's mtime
    rows=run_cells.failure_diagnostics(cell,since='2126-01-01T00:00:00+00:00')
    assert [r['attributed'] for r in rows]==[False,False,False],rows  # every row predates a start in the future
    rows=run_cells.failure_diagnostics(cell,since='not a timestamp')
    assert all(r['attributed'] is False for r in rows),rows  # an unparsable start attributes nothing rather than everything
    passed=run_cells.attach_failure_diagnostics({'state':'pass','steps':[]},cell)
    assert 'diagnostics' not in passed,passed
    assert run_cells.attach_failure_diagnostics({'state':'fail','error':'x','steps':[]},root/'empty-cell')['diagnostics']==[]
    for i in range(6):(folder_/f'20260925T03000{i}Z-1-dddddddd.json').write_text(json.dumps({'returncode':i}),encoding='utf-8')
    assert len(run_cells.failure_diagnostics(cell))==run_cells.FAILURE_DIAGNOSTIC_LIMIT==4
  print('PASS: a failed owned music adapter leaves a bounded, redacted exit-code/stderr diagnostic under TITANOS_HOME before its temp directory is deleted; '
        'the API error carries only the exit code and relative file name; refusals stay verbatim; the kit receipt lists the diagnostics. Synthetic children only.')
