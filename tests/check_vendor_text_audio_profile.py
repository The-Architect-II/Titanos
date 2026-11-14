"""CPU text/audio profile and corresponding-source collection; no native build."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tarfile
import tempfile
import tomllib
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import build_vendor_binary as builder


def main():
    project=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
    requested=[]
    def dependencies(values):
        requested[:]=values
        result={Requirement(value).name:NS(version='fixture') for value in values}
        result.update(torch=NS(version='2.8.0+cpu'),**{'transitive-fixture':NS(version='fixture')})
        return result
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);ffmpeg=root/'ffmpeg';ffmpeg.mkdir();(ffmpeg/'source').mkdir()
        executable='ffmpeg.exe' if os.name=='nt' else 'ffmpeg'
        (ffmpeg/executable).write_bytes(b'controlled native binary')
        license_bytes=b'controlled license bytes\n'
        (ffmpeg/'COPYING.LGPLv2.1').write_bytes(license_bytes)
        (ffmpeg/'LICENSE.md').write_text('controlled source license overview')
        archive=ffmpeg/'source/ffmpeg-fixture.tar.gz'
        with tarfile.open(archive,'w:gz') as output:
            entry=tarfile.TarInfo('ffmpeg-fixture/COPYING.LGPLv2.1');entry.size=len(license_bytes)
            output.addfile(entry,io.BytesIO(license_bytes))
        record={'license_from_binary':'LGPL-2.1-or-later', 'source_sha256':builder.common.digest(archive),
                'binaries':{executable:builder.common.digest(ffmpeg/executable)}}
        (ffmpeg/'BUILD.json').write_text(json.dumps(record))
        with ExitStack() as stack:
            stack.enter_context(patch.object(builder.metadata,'version',side_effect=lambda name:project['version'] if name=='titanos-engine' else '6.22.2'))
            stack.enter_context(patch.object(builder.metadata,'distribution',return_value=NS(version=project['version'])))
            stack.enter_context(patch.object(builder.util,'find_spec',side_effect=lambda name:None if name in builder.TEXT_AUDIO_EXCLUDED else object()))
            stack.enter_context(patch.object(builder.common,'dependencies',side_effect=dependencies))
            result=builder.plan('text-audio',ffmpeg_dir=ffmpeg)
            assert result['collect_packages']==['titanos','uvicorn','huggingface_hub','mcp','mcp_types','numpy','transformers','tokenizers','safetensors','sentencepiece','torch','onnxruntime','imageio_ffmpeg']
            assert requested==project['dependencies']+project['optional-dependencies']['compiled']+project['optional-dependencies']['sentiment']+project['optional-dependencies']['vad']
            assert result['profile']=='vendor-text-audio' and result['mcp_included'] and not result['weights_included']
            assert result['native_ffmpeg']['source_sha256']==record['source_sha256']
            assert result['native_ffmpeg']['input_files']['COPYING.LGPLv2.1']==hashlib.sha256(license_bytes).hexdigest()
            assert 'transitive-fixture' in result['distributions']
            for options in [{},{'without_mcp':True,'ffmpeg_dir':ffmpeg}]:
                try:builder.plan('text-audio',**options)
                except ValueError as exc:assert 'requires full MCP' in str(exc)
                else:raise AssertionError('Missing explicit FFmpeg/full MCP accepted')
            with patch.object(builder.util,'find_spec',return_value=object()):
                try:builder.plan('text-audio',ffmpeg_dir=ffmpeg)
                except ValueError as exc:assert 'clean CPU' in str(exc)
                else:raise AssertionError('Incidental media environment accepted')
            def cuda(values):
                value=dependencies(values);value['torch'].version='2.8.0+cu128';return value
            with patch.object(builder.common,'dependencies',side_effect=cuda):
                try:builder.plan('text-audio',ffmpeg_dir=ffmpeg)
                except ValueError as exc:assert 'CPU PyTorch' in str(exc)
                else:raise AssertionError('CUDA wheel accepted')
            with patch.object(sys,'argv',['builder','--runtime','text-audio','--ffmpeg-dir',str(ffmpeg),'--plan','--output','unused','--work-dir','unused-work']),patch.object(builder,'text_audio_imports',side_effect=AssertionError('Plan imported numerical runtimes')),patch.object(builder.subprocess,'run',side_effect=AssertionError('Plan compiled')):
                output=io.StringIO()
                with redirect_stdout(output):builder.main()
                assert json.loads(output.getvalue())['profile']=='vendor-text-audio'
        (ffmpeg/'COPYING.LGPLv2.1').write_bytes(b'not the corresponding license')
        try:builder.common.ffmpeg_plan(ffmpeg,retain_inventory=True)
        except ValueError as exc:assert 'license text differs' in str(exc)
        else:raise AssertionError('Mismatched license accepted')
        (ffmpeg/'COPYING.LGPLv2.1').write_bytes(license_bytes)
        (ffmpeg/executable).write_bytes(b'tampered')
        try:builder.common.ffmpeg_plan(ffmpeg,retain_inventory=True)
        except ValueError as exc:assert 'binary provenance' in str(exc)
        else:raise AssertionError('Mismatched native bytes accepted')

        collected, copied, options=[],[],{}
        hooks=ModuleType('PyInstaller.utils.hooks')
        def collect(name,**kwargs):
            collected.append((name,kwargs));return [(name+'/data',name)],[(name+'/lib.dll',name)],[name+'.runtime']
        hooks.collect_all=collect
        hooks.copy_metadata=lambda name:copied.append(name) or [(name+'.dist-info',name+'.dist-info')]
        def analysis(*args,**kwargs):
            options.update(kwargs);return NS(pure=[],scripts=[],binaries=kwargs['binaries'],datas=kwargs['datas'])
        plan=root/'plan.json';plan.write_text(json.dumps(result))
        with patch.dict(os.environ,{'TITANOS_BINARY_PLAN':str(plan)}),patch.dict(sys.modules,{'PyInstaller':ModuleType('PyInstaller'),'PyInstaller.utils':ModuleType('PyInstaller.utils'),'PyInstaller.utils.hooks':hooks}):
            runpy.run_path(str(ROOT/'tools/titanos.spec'),init_globals={'Analysis':analysis,'PYZ':lambda *a,**k:None,'EXE':lambda *a,**k:None,'COLLECT':lambda *a,**k:None})
        assert [name for name,_ in collected]==result['collect_packages']
        assert copied==result['distributions']
        assert (str(ffmpeg.resolve()),'native/ffmpeg') in options['datas']
        assert str(ROOT/'tools/binary_hooks') in options['hookspath']
        ffmpeg_options=next(value for name,value in collected if name=='imageio_ffmpeg')
        assert {'binaries/*','**/binaries/*'}<=set(ffmpeg_options['exclude_datas'])
        assert {'**/*.onnx','**/*.safetensors','**/*.gguf'}<=set(ffmpeg_options['exclude_datas'])
        assert set(builder.TEXT_AUDIO_EXCLUDED)<=set(options['excludes'])
        assert {'torch.runtime','onnxruntime.runtime','imageio_ffmpeg.runtime','mcp.runtime'}<=set(options['hiddenimports'])
        runtime=root/'native/ffmpeg';runtime.mkdir(parents=True)
        (runtime/executable).write_bytes(b'entry-path fixture')
        (runtime/'BUILD.json').write_text('{}')
        import frozen_entry
        fake_cli=ModuleType('titanos.cli');fake_cli.main=lambda:os.environ['IMAGEIO_FFMPEG_EXE']
        with patch.dict(sys.modules,{'titanos.cli':fake_cli}),patch.dict(os.environ),patch.object(sys,'frozen',True,create=True),patch.object(sys,'_MEIPASS',str(root),create=True),patch.object(sys,'argv',['titanos']),patch.object(frozen_entry.multiprocessing,'freeze_support'):
            os.environ.pop('IMAGEIO_FFMPEG_EXE',None)
            assert frozen_entry.main()==str(runtime/executable)
    imports={name:ModuleType(name) for name in ['torch','transformers','transformers.pipelines','transformers.models','transformers.models.whisper','transformers.models.whisper.modeling_whisper','transformers.models.whisper.generation_whisper','onnxruntime','imageio_ffmpeg']}
    imports['torch'].__version__='2.8.0+cpu';imports['torch'].version=NS(cuda=None,hip=None)
    for name in ['AutoTokenizer','AutoProcessor','AutoModel','AutoModelForCausalLM','AutoModelForSequenceClassification','AutoModelForSpeechSeq2Seq','AutoModelForCTC','WhisperFeatureExtractor','WhisperTokenizer','WhisperProcessor','pipeline']:
        setattr(imports['transformers'],name,object())
    imports['transformers.pipelines'].AutomaticSpeechRecognitionPipeline=object()
    imports['transformers.models.whisper.modeling_whisper'].WhisperForConditionalGeneration=object()
    imports['transformers.models.whisper.generation_whisper'].WhisperGenerationMixin=object()
    imports['onnxruntime'].__version__='fixture';imports['onnxruntime'].get_available_providers=lambda:['CPUExecutionProvider']
    imports['imageio_ffmpeg'].__version__='fixture'
    with patch.dict(sys.modules,imports):
        assert builder.text_audio_imports()['passed']
        imports['torch'].version.cuda='12.8'
        try:builder.text_audio_imports()
        except ValueError as exc:assert 'CPU-only' in str(exc)
        else:raise AssertionError('CUDA build passed import gate')
    assert not {'torch','transformers','onnxruntime','imageio_ffmpeg','numpy'} & set(sys.modules)
    print(json.dumps({'passed':True,'checks':['full MCP/explicit FFmpeg required','CPU text+ONNX+decoder dependency closure','incidental media and CUDA rejected','native/source/license hash checks','plan does not compile or import numerical frameworks','shared collector excludes wheel FFmpeg and weights, retains native source/notices/metadata','frozen entry selects qualified FFmpeg','future import gate checks ASR/chat modules and CPU build with doubles'],'scope':'Controlled packaging plan/collector checks only; no numerical imports, native build or model run'}))

if __name__=='__main__':main()
