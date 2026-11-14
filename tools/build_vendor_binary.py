"""Freeze one installed vendor runtime with external models and retained notices.

Run natively on the target architecture. --without-mcp explicitly builds an
inference sidecar for a separate MCP frontend; it never substitutes crypto.
"""
import argparse
from datetime import datetime, timezone
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib
import uuid
import zipfile
import re
import struct

import build_binary as common
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]
TEXT_AUDIO_EXCLUDED = ['torchvision', 'diffusers', 'torchaudio', 'torchcodec', 'librosa',
                       'soundfile', 'soxr', 'qwen_tts', 'TTS', 'PIL', 'scipy']
TEXT_VISION_EXCLUDED = [name for name in TEXT_AUDIO_EXCLUDED if name != 'PIL'] + ['onnxruntime', 'imageio', 'imageio_ffmpeg']
GENAI_VISION_EXCLUDED = TEXT_VISION_EXCLUDED + ['torch', 'transformers', 'optimum']

def pe_sections(path):
    data=Path(path).read_bytes();offset=struct.unpack_from('<I',data,60)[0]
    machine,count=struct.unpack_from('<HH',data,offset+4);size=struct.unpack_from('<H',data,offset+20)[0]
    sections={}
    for i in range(count):
        entry=offset+24+size+40*i;name=data[entry:entry+8].rstrip(b'\0').decode()
        length,raw=struct.unpack_from('<II',data,entry+16)
        sections[name]=common.hashlib.sha256(data[raw:raw+length]).hexdigest()
    return {'machine':hex(machine),'sections':sections}

def add_native_notices(built,settings,crypto_dir=None):
    """Keep native vendor terms separate from their Python wrapper licenses."""
    notices=[]
    for name in settings['distributions']:
        dist=metadata.distribution(name)
        for item in dist.files or []:
            leaf=Path(str(item)).name
            if any(x in leaf.lower() for x in ['license','notice','eula','copyright']):
                source=Path(dist.locate_file(item))
                if not source.is_file():continue
                target=built/'THIRD_PARTY_LICENSES'/name/('retained-'+str(len(notices)+1)+'-'+leaf)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
                notices.append({'distribution':name,'path':target.relative_to(built).as_posix(),'sha256':common.digest(target)})
    components=[]
    if 'onnxruntime-qnn' in settings['distributions']:
        components.append({'name':'Qualcomm libraries in onnxruntime-qnn','wrapper_version':settings['packages']['onnxruntime-qnn'],
          'license':'QTI AI Stack proprietary license; distinct from wrapper MIT license',
          'terms':'Section 1 permits object-code distribution incorporated into an application, not standalone SDK distribution; preserve all notices and applicable restrictions.',
          'license_path':next(n['path'] for n in notices if n['distribution']=='onnxruntime-qnn' and n['path'].endswith('Qualcomm_LICENSE.pdf'))})
    if settings['runtime'] in {'dml', 'dml-vision'}:
        origin=json.loads((ROOT/'tools/licenses/DirectML-1.15.4-provenance.json').read_text(encoding='utf-8'))
        libraries=list((built/'_internal').rglob('DirectML.dll'))
        if not libraries or any(pe_sections(p) not in origin['native_pe_sections'].values() for p in libraries):raise ValueError('DirectML native sections differ from retained official 1.15.4 package provenance')
        folder=built/'THIRD_PARTY_LICENSES/DirectML';folder.mkdir(parents=True)
        shutil.copy2(ROOT/'tools/licenses/DirectML-1.15.4-LICENSE.txt',folder/'LICENSE.txt')
        shutil.copy2(ROOT/'tools/licenses/DirectML-1.15.4-provenance.json',folder/'provenance.json')
        components.append({'name':'DirectML','version':'1.15.4','license':'Microsoft binary redistributable terms, distinct from ONNX Runtime MIT license',
                           'license_path':'THIRD_PARTY_LICENSES/DirectML/LICENSE.txt','source_package':origin,
                           'packaged_libraries':{p.relative_to(built).as_posix():common.digest(p) for p in libraries}})
    if crypto_dir:
        crypto_dir=crypto_dir.resolve();record=json.loads((crypto_dir/'BUILD.json').read_text(encoding='utf-8'))
        qa=json.loads((crypto_dir/'QA.json').read_text(encoding='utf-8'))
        if record.get('status')!='built' or qa.get('status')!='passed':raise ValueError('Native crypto build and QA must pass before packaging')
        target=built/'NATIVE_SOURCE/crypto';target.mkdir(parents=True)
        for source in record['sources']:
            path=crypto_dir/source['file']
            if common.digest(path)!=source['sha256']:raise ValueError('Native crypto source archive hash differs')
            if path.name.endswith('.tar.gz'):shutil.copy2(path,target/path.name)
        recipe=crypto_dir.parent/'build_native_crypto.py'
        if common.digest(recipe)!=record['recipe_sha256']:raise ValueError('Native crypto build recipe changed')
        shutil.copy2(recipe,target/'build_native_crypto.py')
        license_file=crypto_dir/('openssl-'+record['openssl_version'])/'LICENSE.txt'
        shutil.copy2(license_file,target/'OpenSSL-LICENSE.txt')
        for name,value in [('BUILD.json',record),('QA.json',qa)]:
            text=json.dumps(value,indent=2)
            for variant in [str(crypto_dir),str(crypto_dir).replace('\\','/')]:text=text.replace(variant.replace('\\','\\\\'),'<NATIVE_CRYPTO_BUILD>')
            text=re.sub(r'[A-Za-z]:(?:\\\\|/)+Users(?:\\\\|/)+[^\\/\"]+','<USER_PROFILE>',text)
            (target/name).write_text(text,encoding='utf-8')
        libraries=list((built/'_internal/cryptography/hazmat/bindings').glob('_rust*.pyd'))
        if len(libraries)!=1 or common.digest(libraries[0])!=qa['rust_module_sha256']:raise ValueError('Packaged crypto native module differs from qualified module')
        components.append({'name':'OpenSSL','version':record['openssl_version'],'license':'Apache-2.0','linkage':'statically linked in qualified native cryptography module','provenance':'NATIVE_SOURCE/crypto'})
    (built/'VENDOR_RUNTIME_NOTICES.json').write_text(json.dumps({'native_components':components,'additional_notice_files':notices,'scope':'Native binary licensing is distinct from Titanos source licensing; driver installation and model weights are outside this bundle'},indent=2),encoding='utf-8')

def plan(runtime,without_mcp=False,ffmpeg_dir=None):
    project=tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))['project']
    if metadata.version('titanos-engine')!=project['version']:
        raise ValueError('Install this exact Titanos source in the isolated build environment first')
    requirements=list(project['dependencies'])
    packages=['titanos','uvicorn','huggingface_hub']
    exceptions=[]
    if runtime == 'text-audio' and (without_mcp or ffmpeg_dir is None):
        raise ValueError('text-audio requires full MCP and an explicit --ffmpeg-dir with qualified native source')
    if runtime in {'text-vision', 'genai-vision', 'dml-vision'} and without_mcp:
        raise ValueError(runtime + ' requires full MCP')
    if ffmpeg_dir is not None and runtime != 'text-audio':
        raise ValueError('--ffmpeg-dir applies to the text-audio vendor profile')
    if without_mcp:
        requirements=[r for r in requirements if Requirement(r).name.lower()!='mcp']
        exceptions.append({'parent':'titanos-engine','dependency':'mcp','reason':'Explicit inference-sidecar profile; MCP must run in a separate frontend. No crypto or authentication substitute.'})
    else:packages+=['mcp','mcp_types']
    if runtime in {'genai', 'genai-vision'}:
        packages+=['numpy','openvino','openvino_genai','openvino_tokenizers']
        requirements+=project['optional-dependencies'][runtime]
        if runtime == 'genai-vision':
            packages += ['PIL']
    elif runtime!='mcp':
        packages+=['numpy','transformers','tokenizers','safetensors','sentencepiece']
        requirements+=project['optional-dependencies']['compiled']
    if runtime in {'text', 'text-audio'}:
        packages+=['torch'];requirements+=project['optional-dependencies']['sentiment']
    elif runtime == 'text-vision':
        packages += ['torch', 'PIL']
        requirements += project['optional-dependencies']['vision']
    elif runtime=='qnn':
        packages+=['onnxruntime','onnxruntime_qnn'];requirements+=['onnxruntime==1.27.0','onnxruntime-qnn==2.6.0']
    elif runtime in {'onnx','coreml'}:
        packages+=['onnxruntime'];requirements+=['onnxruntime==1.27.0']
    elif runtime=='dml-vision':
        if platform.system() != 'Windows' or platform.machine().lower() not in {'amd64', 'x86_64'}:
            raise ValueError('dml-vision currently requires native Windows AMD64')
        packages += ['onnxruntime', 'torch', 'diffusers', 'PIL', 'scipy']
        requirements += project['optional-dependencies']['dml-vision']
    elif runtime=='dml':
        packages+=['onnxruntime']
        distribution='onnxruntime-windowsml' if platform.machine().lower() in {'arm64','aarch64'} else 'onnxruntime-directml'
        requirements.append(distribution)
    elif runtime=='openvino':
        packages+=['openvino'];requirements+=['openvino==2026.3.1']
    if runtime == 'text-audio':
        packages += ['onnxruntime', 'imageio_ffmpeg']
        requirements += project['optional-dependencies']['vad']
        incidental = [p for p in TEXT_AUDIO_EXCLUDED if util.find_spec(p) is not None]
        if incidental:
            raise ValueError('Use a new clean CPU environment for text-audio; incidental packages present: ' + ', '.join(incidental))
    if runtime == 'text-vision':
        incidental = [p for p in TEXT_VISION_EXCLUDED if util.find_spec(p) is not None]
        if incidental:
            raise ValueError('Use a new clean CPU environment for text-vision; incidental packages present: ' + ', '.join(incidental))
    if runtime == 'genai-vision':
        incidental = [p for p in GENAI_VISION_EXCLUDED if util.find_spec(p) is not None]
        if incidental:
            raise ValueError('Use a new clean environment for genai-vision; incidental packages present: ' + ', '.join(incidental))
    missing=[p for p in packages if util.find_spec(p) is None]
    if missing:raise ValueError('Build environment is missing: '+', '.join(missing))
    installed=common.dependencies(requirements);installed['titanos-engine']=metadata.distribution('titanos-engine')
    if runtime in {'text-audio', 'text-vision', 'dml-vision'} and (any(x in installed['torch'].version.lower() for x in ['+cu', '+rocm', '+hip', '+xpu']) or any(name.startswith('nvidia-') for name in installed)):
        raise ValueError(runtime + ' requires a CPU PyTorch wheel, not CUDA/ROCm dependencies')
    source_files={p.relative_to(ROOT).as_posix():common.digest(p) for p in sorted((ROOT/'src/titanos').rglob('*')) if p.is_file() and p.suffix in {'.py','.html','.css','.js'}}
    result = {'version':project['version'],'profile':'vendor-'+runtime,'runtime':runtime,'mcp_included':not without_mcp,
            'source_root':str(ROOT),'os':platform.system().lower(),'architecture':platform.machine(),'python':platform.python_version(),
            'pyinstaller':metadata.version('pyinstaller'),'collect_packages':packages,'distributions':sorted(installed),
            'packages':{k:v.version for k,v in sorted(installed.items())},'dependency_exceptions':exceptions,
            'source_files':source_files,'weights_included':False,
            'scope':('CPU Transformers text/ASR and ONNX VAD with source-bound FFmpeg and external weights; model compatibility requires separate qualification'
                     if runtime=='text-audio' else 'CPU Transformers text and Pillow image decoding for slow-processor TrOCR/YOLOS adapters with external weights; other vision families and model quality require separate qualification'
                     if runtime=='text-vision' else 'OpenVINO GenAI text and SD/SDXL image-generation/editing runtime with Pillow and prepared external models; model/device quality requires separate qualification'
                     if runtime=='genai-vision' else 'OpenVINO GenAI text-generation runtime with prepared external models; no PyTorch/media qualification implied'
                     if runtime=='genai' else 'Transformers text runtime for sentiment, chat and embeddings with external models; no media pipeline qualification implied'
                     if runtime=='text' else 'One installed vendor graph runtime; no model weights or general PyTorch/media qualification implied')}
    if runtime == 'text-audio':
        result.update(common.ffmpeg_plan(ffmpeg_dir, retain_inventory=True))
        result['exclude_modules'] = TEXT_AUDIO_EXCLUDED
        result['runtime_import_validation'] = 'required before compilation; not run by --plan'
    if runtime == 'text-vision':
        result['exclude_modules'] = TEXT_VISION_EXCLUDED
        result['runtime_import_validation'] = 'required before compilation; not run by --plan'
    if runtime == 'dml-vision':
        result['exclude_modules'] = ['torchvision', 'torchaudio', 'openvino', 'openvino_genai', 'optimum', 'qwen_tts', 'TTS', 'librosa', 'imageio', 'imageio_ffmpeg']
        result['runtime_import_validation'] = 'required before compilation; not run by --plan'
        result['scope'] = 'SD1 ONNX on explicit DirectML with CPU Torch scheduling, full MCP, external weights and per-component profiling; native inference unqualified'
    if runtime == 'genai-vision':
        result['exclude_modules'] = GENAI_VISION_EXCLUDED
        result['runtime_import_validation'] = 'required before compilation; not run by --plan'
    return result


def dml_vision_imports():
    """Metadata/import only; no model or execution session is constructed."""
    import torch
    import onnxruntime as ort
    from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel, PNDMScheduler
    from transformers import CLIPTokenizer, CLIPImageProcessor
    if torch.version.cuda or torch.version.hip:
        raise ValueError("dml-vision requires the CPU PyTorch wheel for scheduling")
    if "DmlExecutionProvider" not in ort.get_available_providers():
        raise ValueError("The build runtime lacks DirectML")
    return {"imports": [c.__name__ for c in [OnnxStableDiffusionPipeline, OnnxRuntimeModel, PNDMScheduler, CLIPTokenizer, CLIPImageProcessor]],
            "torch": torch.__version__, "ort": ort.__version__, "providers": ort.get_available_providers(), "model_loaded": False}


def text_audio_imports():
    """Exercise real lazy ASR/chat imports in the clean env, without a checkpoint."""
    import torch
    if any(getattr(torch.version, name, None) is not None for name in ['cuda', 'hip', 'xpu']):
        raise ValueError('text-audio needs a CPU-only PyTorch build')
    from transformers import (AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM,
        AutoModelForSequenceClassification, AutoModelForSpeechSeq2Seq, AutoModelForCTC,
        WhisperFeatureExtractor, WhisperTokenizer, pipeline)
    from transformers.pipelines import AutomaticSpeechRecognitionPipeline
    from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
    from transformers import WhisperProcessor
    from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
    import onnxruntime
    import imageio_ffmpeg
    if 'CPUExecutionProvider' not in onnxruntime.get_available_providers():
        raise ValueError('ONNX Runtime CPU provider is unavailable')
    return {'passed': True, 'torch': torch.__version__, 'torch_cuda': None, 'torch_hip': None,
            'onnxruntime': onnxruntime.__version__, 'imageio_ffmpeg': imageio_ffmpeg.__version__,
            'scope': 'Actual chat/classifier/Whisper/ASR imports; no model instantiation, native decoder execution or GPU probe'}


def text_vision_imports():
    """Resolve the actual slow TrOCR/YOLOS classes without loading weights."""
    import torch
    if any(getattr(torch.version, name, None) is not None for name in ['cuda', 'hip', 'xpu']):
        raise ValueError('text-vision needs a CPU-only PyTorch build')
    from PIL import Image, __version__ as pillow_version
    from transformers import (AutoImageProcessor, AutoModelForObjectDetection,
        AutoModelForSeq2SeqLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer,
        AutoModelForCausalLM, AutoModelForSequenceClassification,
        DeiTImageProcessor, YolosImageProcessor, VisionEncoderDecoderModel,
        TrOCRForCausalLM, DeiTModel, YolosForObjectDetection, AutoModelForDepthEstimation,
        DepthAnythingForDepthEstimation, DPTImageProcessor, Dinov2Model,
        SamConfig, SamModel, SamImageProcessor, SamProcessor)
    return {'passed': True, 'torch': torch.__version__, 'pillow': pillow_version,
            'torch_cuda': None, 'torch_hip': None,
            'scope': 'Actual text, slow TrOCR/YOLOS, relative Depth Anything/DINOv2 and builtin SAM/slow processor imports; no model instantiation, image decode, numerical forward or GPU probe'}


def genai_vision_imports():
    """Resolve the native text/image classes without models or a device probe."""
    import numpy
    import openvino
    import openvino_tokenizers
    from openvino_genai import LLMPipeline, Text2ImagePipeline, Image2ImagePipeline
    from PIL import Image, __version__ as pillow_version
    return {'passed': True, 'numpy': numpy.__version__, 'openvino': openvino.__version__,
            'pillow': pillow_version,
            'scope': 'Actual GenAI text, text-to-image and image-to-image imports; no model instantiation, image decode, numerical forward or device probe'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',choices=['qnn','dml','dml-vision','openvino','genai','genai-vision','text','text-audio','text-vision','onnx','coreml','mcp'],required=True)
    parser.add_argument('--without-mcp',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--zip',action='store_true')
    parser.add_argument('--native-crypto-dir',type=Path,help='Task-local upstream OpenSSL/cryptography sources, built wheel and passed native QA')
    parser.add_argument('--ffmpeg-dir',type=Path,help='Required for text-audio: qualified source-built LGPL FFmpeg, original source and notices')
    parser.add_argument('--tesseract-dir',type=Path,help='Verified source-built model-free Windows AMD64 Tesseract bundle')
    args=parser.parse_args()
    if args.runtime=='mcp' and args.without_mcp:parser.error('MCP frontend cannot omit MCP')
    settings=plan(args.runtime,args.without_mcp,args.ffmpeg_dir)
    if args.tesseract_dir:
        if 'PIL' not in settings['collect_packages']:
            parser.error('--tesseract-dir requires a vision runtime containing Pillow')
        settings['native_tesseract'] = common.tesseract_plan(args.tesseract_dir)
    if args.plan:print(json.dumps(settings,indent=2));return
    output=args.output.resolve();work=args.work_dir.resolve()
    if output.exists() or output==Path(output.anchor) or ROOT.is_relative_to(output):
        parser.error('Output must be a new application directory outside the source tree')
    if args.runtime == 'text-audio':
        settings['runtime_import_validation'] = text_audio_imports()
    elif args.runtime == 'text-vision':
        settings['runtime_import_validation'] = text_vision_imports()
    elif args.runtime == 'dml-vision':
        settings['runtime_import_validation'] = dml_vision_imports()
    elif args.runtime == 'genai-vision':
        settings['runtime_import_validation'] = genai_vision_imports()
    work.mkdir(parents=True,exist_ok=True);run=work/('vendor-'+uuid.uuid4().hex);run.mkdir()
    private_plan=run/'plan.json';private_plan.write_text(json.dumps(settings,indent=2),encoding='utf-8')
    env=dict(os.environ);env['TITANOS_BINARY_PLAN']=str(private_plan)
    with (run/'pyinstaller.log').open('wb') as log:
        subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--distpath',str(run/'dist'),'--workpath',str(work/('cache-'+args.runtime)),str(ROOT/'tools/titanos.spec')],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    built=run/'dist/titanos'
    observed={p.relative_to(ROOT).as_posix():common.digest(p) for p in sorted((ROOT/'src/titanos').rglob('*')) if p.is_file() and p.suffix in {'.py','.html','.css','.js'}}
    if observed!=settings['source_files']:raise ValueError('Runtime source changed during build')
    if settings.get('native_ffmpeg'):
        if common.ffmpeg_plan(args.ffmpeg_dir, retain_inventory=True)['native_ffmpeg'] != settings['native_ffmpeg']:
            raise ValueError('FFmpeg input changed during compilation')
        packaged = built / '_internal/native/ffmpeg'
        for name, expected in settings['native_ffmpeg']['input_files'].items():
            if name in settings['native_ffmpeg']['source_build_binaries']:
                continue  # Shared native inventory checks Windows identity / verified Mach-O postprocessing.
            if common.digest(packaged / name) != expected:
                raise ValueError('FFmpeg collection differs from its pinned input: ' + name)
    common.copy_tesseract(built, settings)
    common.source_package(built);common.write_notices(built,settings);add_native_notices(built,settings,args.native_crypto_dir)
    normalized=[]
    for path in (built/'_internal').glob('*.dist-info/direct_url.json'):
        value=json.loads(path.read_text(encoding='utf-8'))
        if value.get('url','').startswith('file:'):
            value['url']='file:///bundled-local-build-input';path.write_text(json.dumps(value,indent=2),encoding='utf-8');normalized.append(path.relative_to(built).as_posix())
    executable=built/('titanos.exe' if os.name=='nt' else 'titanos')
    clean={k:v for k,v in os.environ.items() if k.upper() not in {'PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','TITANOS_API_KEY'}}
    if os.name=='nt':
        windows=os.environ.get('SYSTEMROOT') or os.environ.get('SystemRoot');clean['PATH']=str(Path(windows)/'System32')+os.pathsep+windows
    check=subprocess.run([str(executable),'--version'],env=clean,cwd=run,capture_output=True,text=True,timeout=120)
    if check.returncode or settings['version'] not in check.stdout:raise RuntimeError('Frozen startup failed: '+check.stderr[-1000:])
    public={k:v for k,v in settings.items() if k not in {'source_root', 'ffmpeg_dir', 'native_tesseract'}}
    if settings.get('native_tesseract'):
        public['native_tesseract'] = {key: value for key, value in settings['native_tesseract'].items() if key != 'directory'}
    public.update(built_at=datetime.now(timezone.utc).isoformat(),local_metadata_normalized=normalized,
                  validation={'startup':'passed','version_stdout':check.stdout.strip(),'inference':'Must be qualified separately on actual hardware'})
    (built/'BUILD.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    if args.runtime in {'text-audio', 'text-vision', 'genai-vision', 'dml-vision'}:
        common.smoke(executable, built)
    (built/'START_HERE.md').write_text('Titanos '+settings['version']+' — '+args.runtime+' runtime\n\nRun `titanos.exe --help` and configure external model paths. This folder includes its Python/runtime dependencies. Model weights are separate. '+('MCP is included. ' if settings['mcp_included'] else 'This explicit inference sidecar omits MCP; use a separate MCP frontend. ')+'See BUILD.json, SOURCE, DEPENDENCIES.json and THIRD_PARTY_LICENSES for exact scope, versions and notices. Hardware drivers remain an operating-system prerequisite.\n',encoding='utf-8')
    common.validate_packaged_distributions(built)
    manifest={p.relative_to(built).as_posix():common.digest(p) for p in sorted(built.rglob('*')) if p.is_file()}
    (built/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    output.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(built,output)
    result={'output':str(output),'executable_sha256':common.digest(output/executable.name),'files':len(manifest),'mcp_included':settings['mcp_included']}
    if args.zip:
        archive=output.with_name(output.name+'.zip')
        if archive.exists():raise ValueError('Archive already exists; choose a new output name')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file():z.write(p,output.name+'/'+p.relative_to(output).as_posix())
        result.update(archive=str(archive),archive_sha256=common.digest(archive))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
