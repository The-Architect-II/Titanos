"""Owned isolated ACE-Step 1.5 music adapter contract.

The serving interpreter never imports the incompatible music stack. Packaging
installs an audited pinned adapter/runtime and writes music-runtime.json.
"""
from __future__ import annotations

import hashlib
import base64
import binascii
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
import wave

from .config import home
from .engine import EngineError
from .owned_command import redact,run_owned
from .settings import current_settings

ACE_CODE_REVISION="ca1e85fe9430179831e6bc6be790c332190a3866"
ACE_SUPPORTED_VARIANTS=("xl-sft",)
ACE_MODELS={
    "xl-sft":"d06de46b4622f781cf07f4a013a67d591ca52819",
    "xl-turbo":"d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee",
    "xl-base":"220c1166efbdd9583eafcb12eb160594bbfcb241",
}

_VISIBILITY_KEYS=("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES",
                  "GPU_DEVICE_ORDINAL","ZE_AFFINITY_MASK","ONEAPI_DEVICE_SELECTOR","SYCL_DEVICE_FILTER")
_DRIVER_KEYS=("CUDA_DEVICE_ORDER","CUDA_PATH","CUDA_HOME","ROCM_PATH","HIP_PATH","ONEAPI_ROOT",
              "NVIDIA_VISIBLE_DEVICES","NVIDIA_DRIVER_CAPABILITIES","PYTORCH_CUDA_ALLOC_CONF",
              "PYTORCH_ALLOC_CONF")
_DTYPES={"float32","float16","bfloat16"}
_PLACEMENT_KEYS={"requested","resolved","device_free_bytes","required_bytes","total_resident_bytes",
                 "peak_set_bytes","headroom_bytes","accelerate_version","basis"}
_PLACEMENT_REFUSAL="placement_refused_before_load:"
# Windows loads executables and DLLs through paths bounded by MAX_PATH (259 characters plus the terminator) even where the file
# system and a long-path-aware Python can address longer names (observed: the relocated runtime imported torch with 275-character
# header paths, while a 161-character parent made cudnn DLL loading fail with WinError 206). The portable runtime's loader-relevant
# files (.exe/.dll/.pyd) are therefore checked against this bound before any child is launched, and the refusal names the
# directory budget the caller has to meet.
MAX_RUNTIME_PATH=259
_LOADER_SUFFIXES=(".exe",".dll",".pyd")
FAILURE_DIAGNOSTICS_DIR=Path("outputs")/"music-failures"  # under TITANOS_HOME
FAILURE_DIAGNOSTIC_BYTES=16384


def _cuda_tokens(value):
    from .devices import normalized_uuid
    if value.strip() in {"","-1"}:return []
    tokens=[]
    for token in value.split(","):
        token=token.strip()
        if re.fullmatch(r"[0-9]+",token):token=str(int(token))
        elif token.startswith("GPU-") and normalized_uuid(token):token="GPU-"+normalized_uuid(token)
        else:raise ValueError("Music CUDA visibility requires distinct ordinals or full physical GPU UUIDs")
        if token in tokens:raise ValueError("Music CUDA visibility contains a duplicate assignment")
        tokens.append(token)
    if any(t.startswith("GPU-") for t in tokens) and any(t.isdigit() for t in tokens):
        raise ValueError("Mixed UUID/ordinal CUDA visibility is ambiguous; use a UUID-only mask")
    return tokens


def _device_contract(spec,env,settings):
    """Constrain the child before imports; do not import the parent's Torch stack."""
    requested=spec.device
    if not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?|GPU-[A-Fa-f0-9-]+",requested):
        raise ValueError("This isolated music contract supports CPU/CUDA only; the explicit device was not substituted")
    device=requested;expected=None
    if requested!="cpu":
        tokens=_cuda_tokens(env["CUDA_VISIBLE_DEVICES"]) if "CUDA_VISIBLE_DEVICES" in env else None
        if requested.startswith("GPU-"):
            from .devices import normalized_uuid,pin
            identity=normalized_uuid(requested)
            if not identity:raise ValueError("Music requires a complete physical GPU UUID")
            expected="GPU-"+identity
            if tokens is not None and (not tokens or any(t.isdigit() for t in tokens) or expected not in tokens):
                raise ValueError("Explicit music GPU is outside or ambiguous within inherited visibility; use a matching UUID mask")
            expected=pin([expected])[0]
            env["CUDA_VISIBLE_DEVICES"]=expected;device="cuda:0"
        elif requested.startswith("cuda") or requested=="auto" and tokens is not None:
            if requested=="auto" and not tokens:device="cpu"
            else:
                index=int(requested.split(":",1)[1]) if ":" in requested else 0
                if tokens is not None and index>=len(tokens):
                    raise ValueError("Requested music CUDA ordinal is absent from inherited visibility")
                selected=tokens[index] if tokens is not None else str(index)
                if selected.startswith("GPU-"):
                    from .devices import pin
                    expected=pin([selected])[0];selected=expected
                # Numeric selectors are CUDA ordinals, never nvidia-smi indices.
                # The trusted child must report their actual runtime UUID before use.
                env["CUDA_VISIBLE_DEVICES"]=selected;device="cuda:0"
    if device=="cpu":
        for name in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","GPU_DEVICE_ORDINAL"):
            env[name]=""
        if spec.dtype not in {"auto","float32"}:
            raise ValueError("The music CPU runner supports float32 only; dtype was not substituted")
    return {"version":1,"requested_device":requested,"device":device,"expected_uuid":expected,
            "visibility":{name:env[name] for name in _VISIBILITY_KEYS if name in env},
            "dtype":spec.dtype,"threads":settings.threads,"interop_threads":1,
            "max_audio_seconds":settings.max_audio_seconds,"max_output_bytes":settings.max_music_output_bytes,
            "max_samples":settings.max_music_samples,"max_steps":settings.max_steps,
            "max_prompt_chars":settings.max_prompt_chars,"max_text_chars":settings.max_text_chars,
            "placement":spec.placement or "auto"}


def _validate_generation_limits(data,settings):
    """Direct Engine callers must obey the same configured limits as HTTP."""
    prompt=data.get("prompt");lyrics=data.get("lyrics")
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>settings.max_prompt_chars:
        raise ValueError("Music prompt exceeds TITANOS_MAX_PROMPT_CHARS or is empty/invalid")
    if lyrics is not None and (not isinstance(lyrics,str) or len(lyrics)>settings.max_text_chars):
        raise ValueError("Music lyrics exceed TITANOS_MAX_TEXT_CHARS or are invalid")
    steps=data.get("steps")
    if steps is None:steps=50  # Same selected-recipe default as the child.
    if type(steps) is not int or not 1<=steps<=settings.max_steps:
        raise ValueError("Music steps must be an integer within 1..TITANOS_MAX_STEPS")


def _child_environment(executable,directory):
    """Allow driver context, not credentials or the serving Python/frozen runtime."""
    shared_prefixes={Path(value).resolve() for value in ("/","/usr","/usr/local","/opt")}
    roots=[Path(value).resolve() for value in (sys.prefix,sys.base_prefix)
           if Path(value).resolve() not in shared_prefixes]
    if getattr(sys,"_MEIPASS",None):roots.append(Path(sys._MEIPASS).resolve())
    def paths(value):
        values=[]
        for item in value.split(os.pathsep):
            if not item or not Path(item).is_absolute():continue
            path=Path(item).resolve()
            if any(path.is_relative_to(root) for root in roots):continue
            values.append(item)
        return os.pathsep.join(values)
    # USERNAME/USER/LOGNAME are identity strings, not credentials: torch's inductor cache resolution calls getpass.getuser() while
    # torch._dynamo/_inductor import, and without any of them (this environment is deliberately minimal) that raises mid-import, the
    # lazy re-import registers torch/_dynamo/package.py's cache artifact twice and every diffusers import fails with "Artifact of
    # type=precompile already registered in mega-cache artifact factory" (run acemusic-resident-20260925-r2). The cache directories
    # below are pinned inside the request directory as well, so the child never derives a path from the user name or the host TEMP.
    env={name:os.environ[name] for name in (*_VISIBILITY_KEYS,*_DRIVER_KEYS,
         "SystemRoot","WINDIR","ComSpec","LANG","LC_ALL","LC_CTYPE","TZ","USERNAME","USER","LOGNAME") if name in os.environ}
    env["PATH"]=os.pathsep.join(filter(None,(str(executable.parent),paths(os.environ.get("PATH","")))))
    for name in ("LD_LIBRARY_PATH","LIBPATH","DYLD_LIBRARY_PATH","DYLD_FALLBACK_LIBRARY_PATH"):
        value=paths(os.environ.get(name+"_ORIG",os.environ.get(name,"")))
        if value:env[name]=value
    env.update(HOME=str(directory),USERPROFILE=str(directory),TMP=str(directory),TEMP=str(directory),
               XDG_CACHE_HOME=str(directory/"cache"),HF_HOME=str(directory/"hf-cache"),
               TORCHINDUCTOR_CACHE_DIR=str(directory/"inductor-cache"),TRITON_CACHE_DIR=str(directory/"triton-cache"),
               PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    return env


def _python_command(executable,adapter):
    # A PyInstaller Windows parent passes SetDllDirectory state independently of
    # environment variables. Reset it in the child before importing the runner,
    # without changing the serving process's DLL state while other requests run.
    bootstrap=("import os,runpy,sys\n"
               "if os.name=='nt':\n"
               " import ctypes\n"
               " k=ctypes.WinDLL('kernel32',use_last_error=True)\n"
               " k.SetDllDirectoryW.argtypes=[ctypes.c_wchar_p]\n"
               " if not k.SetDllDirectoryW(None):raise ctypes.WinError(ctypes.get_last_error())\n"
               "sys.argv=sys.argv[1:]\n"
               "runpy.run_path(sys.argv[0],run_name='__main__')\n")
    # -X utf8: the child's stdout/stderr are log files, and -I ignores PYTHON* variables, so without UTF-8 mode Python encodes with the
    # locale codec (cp1252 on Windows) and any non-Latin-1 character printed by the pinned source or its libraries (transformers'
    # auto_docstring prints a U+1F6A8 warning when the ACE model class is imported) raises UnicodeEncodeError before any weights load.
    return [str(executable),"-I","-B","-X","utf8","-c",bootstrap,str(adapter)]


def _verify_runtime(metadata,contract):
    from .devices import normalized_uuid
    def invalid():raise EngineError("Music runner did not verify the requested device/dtype/resource contract",502,"invalid_backend_response")
    evidence=metadata.get("evidence");effective=metadata.get("effective")
    if not isinstance(evidence,dict) or not isinstance(effective,dict):invalid()
    runtime=evidence.get("runtime")
    if not isinstance(runtime,dict) or runtime.get("contract_version")!=1 or "device_uuid" not in runtime:invalid()
    device=runtime.get("resolved_device")
    if not isinstance(device,str) or not re.fullmatch(r"cpu|cuda:[0-9]+",device):invalid()
    if contract["device"]!="auto" and device!=contract["device"]:invalid()
    if runtime.get("visibility")!=contract["visibility"] or not isinstance(runtime.get("torch_version"),str) or not runtime["torch_version"]:invalid()
    for key,observed in (("threads","torch_num_threads"),("interop_threads","torch_num_interop_threads")):
        if (type(runtime.get(observed)) is not int or runtime[observed]!=contract[key]
                or type(effective.get(key)) is not int or effective[key]!=contract[key]):invalid()
    visible=runtime.get("visible_devices")
    if not isinstance(visible,list) or len(visible)>128:invalid()
    if device=="cpu":
        if (runtime.get("backend")!="cpu" or runtime.get("device_uuid") is not None or visible
                or runtime.get("identity_source")!="parameter.device" or effective.get("dtype")!="float32"):invalid()
    else:
        if runtime.get("backend")!="cuda" or runtime.get("identity_source")!="torch.cuda.get_device_properties":invalid()
        identities=[]
        for index,row in enumerate(visible):
            if not isinstance(row,dict) or row.get("local_device")!=f"cuda:{index}":invalid()
            identity=row.get("uuid")
            if not isinstance(identity,str) or not identity.startswith("GPU-") or not normalized_uuid(identity):invalid()
            identities.append("GPU-"+normalized_uuid(identity))
        if len(set(identities))!=len(identities):invalid()
        index=int(device.split(":")[1]);identity=runtime.get("device_uuid")
        if index>=len(identities) or identity!=identities[index]:invalid()
        if contract["device"]=="cuda:0" and len(identities)!=1:invalid()
        if contract["expected_uuid"] is not None and identity!=contract["expected_uuid"]:invalid()
        if effective.get("dtype")=="bfloat16" and runtime.get("bfloat16_supported") is not True:invalid()
    dtype=effective.get("dtype")
    if not isinstance(dtype,str) or dtype not in _DTYPES or contract["dtype"]!="auto" and dtype!=contract["dtype"]:invalid()
    # Weight placement evidence: what the child decided and measured before loading, not the request echoed back.
    placement=runtime.get("placement")
    if not isinstance(placement,dict) or set(placement)!=_PLACEMENT_KEYS:invalid()
    resolved=placement["resolved"]
    if placement["requested"]!=contract["placement"] or resolved not in {"host","offload","resident"}:invalid()
    if (resolved=="host")!=(device=="cpu") or contract["placement"]=="resident" and resolved!="resident":invalid()  # an explicit resident contract accepts only a resident resolution on every device: a cpu 'host' receipt under 'resident' is a downgrade the child must have refused before loading
    if device!="cpu" and contract["placement"] in {"resident","offload"} and resolved!=contract["placement"]:invalid()
    # Resident means no CPU offload anywhere; the receipt's effective flag must agree with the resolution.
    if type(effective.get("cpu_offload")) is not bool or effective["cpu_offload"]!=(resolved=="offload"):invalid()
    free=placement["device_free_bytes"]
    if (free is None)!=(device=="cpu") or free is not None and (type(free) is not int or free<0):invalid()
    for key in ("required_bytes","total_resident_bytes","peak_set_bytes","headroom_bytes"):
        if type(placement[key]) is not int or placement[key]<0:invalid()
    if placement["required_bytes"]<placement["total_resident_bytes"] or placement["total_resident_bytes"]<placement["peak_set_bytes"]:invalid()
    # A resident resolution the block's own numbers contradict (less free device memory than the bundle bill) is not evidence.
    if resolved=="resident" and free<placement["required_bytes"]:invalid()
    version=placement["accelerate_version"]
    if version is not None and (not isinstance(version,str) or not version.strip() or len(version)>64):invalid()
    if resolved=="resident" and version is None:invalid()
    # auto may fall back to offload only for an evidenced obstacle: accelerate absent or free below the bill (an unsupported bfloat16
    # is already rejected above, since the receipt's dtype would then contradict bfloat16_supported).
    if device!="cpu" and contract["placement"]=="auto" and resolved=="offload" and version is not None and free>=placement["required_bytes"]:invalid()
    basis=placement["basis"]
    if not isinstance(basis,str) or not basis.strip() or len(basis)>400:invalid()
    placements=runtime.get("parameter_placements")
    if not isinstance(placements,list) or not placements or len(placements)>128:invalid()
    executed={}
    for row in placements:
        stage=row.get("stage") if isinstance(row,dict) else None
        if (stage not in {"execute","offload","retain"} or not isinstance(row.get("component"),str)
                or not isinstance(row.get("device"),str) or row["device"] not in {"cpu",device}
                or not isinstance(row.get("dtype"),str) or row["dtype"] not in _DTYPES):invalid()
        # Offload rows are host-side observations after a stage; retain rows exist only when resident.
        if stage=="offload" and (resolved=="resident" or row["device"]!="cpu"):invalid()
        if stage=="retain" and (resolved!="resident" or row["device"]!=device) or resolved=="resident" and row["device"]!=device:invalid()  # resident: no component ever sat on the host, so a cpu row of any stage contradicts the resolution
        if stage=="execute" and row["device"]==device:
            executed.setdefault(row["component"],set()).add(row["dtype"])
    if not {"dit","text_encoder","vae"}<=set(executed) or executed["dit"]!={dtype}:invalid()
    if contract["dtype"]!="auto" and any(executed[name]!={dtype} for name in ("dit","text_encoder","vae")):invalid()
    return runtime


def _bounded_json(path,limit=65536):
    """Parsed JSON of a child-written file when it exists, is at most `limit` bytes and parses; else None."""
    try:
        if not path.is_file() or path.stat().st_size>limit:return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):return None


def _retain_failure(exc,paths,runner,contract,request,job_id):
    """Persist a bounded, secret-safe record of a failed adapter run under TITANOS_HOME/outputs/music-failures while the request
    directory still exists. Carries the child's exit code and redacted stderr tail, the result/progress evidence the child wrote,
    runner basenames and the non-text request/contract summary. The structured fields never carry the prompt or lyrics or a path;
    the stderr tail is credential-redacted with absolute paths reduced to their final component, but otherwise keeps whatever the
    child printed to stderr (a traceback, or any text the child chose to echo). Returns the file name, or None when it could not be
    written; strictly best effort, so any exception here is swallowed and the original failure is never masked."""
    result_path,progress_path=paths
    result=_bounded_json(result_path);error=None
    if isinstance(result,dict) and isinstance(result.get("evidence"),dict) and isinstance(result["evidence"].get("error"),dict):
        row=result["evidence"]["error"]
        error={"type":redact(str(row.get("type")))[:100],"message":redact(str(row.get("message")))[:2000]}
    progress=_bounded_json(progress_path)
    if isinstance(progress,dict):
        progress={key:(value[:200] if isinstance(value,str) else value if isinstance(value,(int,float,bool)) or value is None else str(type(value).__name__))
                  for key,value in progress.items() if key in {"stage","completed","total","native_cancelled"}}
    else:progress=None
    stamp=datetime.now(timezone.utc)
    record={"schema":1,"utc":stamp.isoformat(),"job_id":job_id,"returncode":getattr(exc,"returncode",None),
            "stderr_tail":getattr(exc,"stderr_tail","") or "","result_error":error,"progress":progress,
            "runner":{name:Path(path).name for name,path in runner.items()},
            "contract":{key:contract.get(key) for key in ("device","requested_device","dtype","placement","threads")},
            "request":{key:request.get(key) for key in ("duration","steps","seed","mode")}}
    body=json.dumps(record,indent=1,sort_keys=True)
    while len(body.encode("utf-8"))>FAILURE_DIAGNOSTIC_BYTES:
        if not record["stderr_tail"]:return None  # cannot bound the file without its stderr text: retain nothing rather than an oversized record
        record["stderr_tail"]=record["stderr_tail"][:max(0,len(record["stderr_tail"])-(len(body.encode("utf-8"))-FAILURE_DIAGNOSTIC_BYTES)-64)]
        body=json.dumps(record,indent=1,sort_keys=True)
    name=f"{stamp.strftime('%Y%m%dT%H%M%S')}Z-{os.getpid()}-{secrets.token_hex(4)}.json"
    try:
        directory=home()/FAILURE_DIAGNOSTICS_DIR  # home() can raise RuntimeError (no resolvable home) as well as OSError/ValueError
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/name).open("x",encoding="utf-8") as stream:stream.write(body)
    except Exception:return None  # noqa: BLE001 - best effort by contract
    return name


def _raise_refusal(result_path,cause):
    """Surface the child's pre-load placement refusal verbatim; every other child failure stays generic."""
    try:
        if not result_path.is_file() or result_path.stat().st_size>65536:return
        message=json.loads(result_path.read_text(encoding="utf-8"))["evidence"]["error"]["message"]
    except (OSError,ValueError,KeyError,TypeError):return
    if isinstance(message,str) and message.startswith(_PLACEMENT_REFUSAL) and re.fullmatch(r"[ -~]{1,400}",message):
        raise EngineError(message,503,"placement_refused") from cause


def _digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        while block:=stream.read(1024*1024):value.update(block)
    return value.hexdigest()


def _ordinary(root,value,label):
    path=(root/value).resolve()
    if (Path(value).is_absolute() or ".." in Path(value).parts or not path.is_relative_to(root)
            or not path.is_file() or path.is_symlink()):
        raise ValueError(f"Invalid {label} path in music bundle")
    return path


def inspect_bundle(directory,verify=True,*,read_json=None):
    root=Path(directory).resolve()
    if not root.is_dir() or root.is_symlink():raise ValueError("Music bundle must be an ordinary local directory")
    manifest_path=root/"titanos-music.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():raise ValueError("Music bundle requires titanos-music.json")
    read_json = read_json or (lambda path: json.loads(path.read_text(encoding="utf-8")))
    value=read_json(manifest_path)
    allowed={"schema_version","family","variant","code_revision","model_revision","components","capabilities","license"}
    if set(value)!=allowed or value.get("schema_version")!=1 or value.get("family")!="ace-step-1.5":
        raise ValueError("Unsupported music bundle manifest")
    if value.get("code_revision")!=ACE_CODE_REVISION or ACE_MODELS.get(value.get("variant"))!=value.get("model_revision"):
        raise ValueError("ACE-Step code/model revision is not an approved pinned recipe")
    components=value.get("components")
    if not isinstance(components,dict) or not {"core","dit"}<=set(components) or len(components)>8:
        raise ValueError("ACE-Step bundle requires bounded core and dit components")
    files={}
    for name,row in components.items():
        if not isinstance(row,dict) or set(row)!={"path","sha256","revision","license"}:
            raise ValueError("Music component identity is incomplete")
        path=_ordinary(root,row["path"],"component")
        if not isinstance(row["sha256"],str) or len(row["sha256"])!=64 or verify and _digest(path)!=row["sha256"]:
            raise ValueError("Music component checksum mismatch: "+name)
        inventory=read_json(path)
        entries=inventory.get("files")
        if inventory.get("revision")!=row["revision"] or not isinstance(entries,list) or not entries or len(entries)>10_000:
            raise ValueError("Music component inventory is incomplete")
        for entry in entries:
            if not isinstance(entry,dict) or set(entry)!={"path","bytes","sha256"}:
                raise ValueError("Music component inventory entry is invalid")
            artifact=_ordinary(root,entry["path"],"component artifact")
            if artifact.stat().st_size!=entry["bytes"] or verify and _digest(artifact)!=entry["sha256"]:
                raise ValueError("Music component changed after acquisition: "+entry["path"])
        files[name]={"path":row["path"],"bytes":path.stat().st_size,"sha256":row["sha256"],
                     "revision":row["revision"],"license":row["license"]}
    capabilities=value.get("capabilities")
    if not isinstance(capabilities,dict) or set(capabilities)-{"instrumental","song","lyrics","vocal_languages","duration_min","duration_max","bpm","key_scale","reference_audio"}:
        raise ValueError("Music capability declaration is invalid")
    if not capabilities.get("instrumental") or not isinstance(capabilities.get("vocal_languages",[]),list):
        raise ValueError("Music bundle must declare actual supported modes/languages")
    return {"root":root,"manifest":value,"files":files}


def runtime_receipt():
    location=home()/"music-runtime.json"
    portable=not location.exists()
    if portable:
        location=Path(sys.executable).resolve().parent/"runtimes"/"ace-step-1.5"/"runtime.json"
    if not location.is_file():raise ValueError("Music runtime is not installed; provide the verified portable music runtime")
    if location.is_symlink():raise ValueError("Music runtime receipt must be an ordinary file")
    value=json.loads(location.read_text(encoding="utf-8"))
    required={"schema_version","family","code_revision","executable","adapter","files"}
    if not portable:required.add("destination")
    if not required<=set(value) or value["schema_version"]!=1 or value["family"]!="ace-step-1.5" or value["code_revision"]!=ACE_CODE_REVISION:
        raise ValueError("Installed music runtime receipt is incompatible")
    root=location.parent if portable else Path(value["destination"])
    if root.is_symlink():raise ValueError("Music runtime directory must be ordinary")
    root=root.resolve()
    if not root.is_dir():raise ValueError("Music runtime directory is missing")
    def runtime_file(name):
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Music runtime files must be safe relative POSIX paths")
        candidate=root/relative
        if not candidate.is_file():raise ValueError("Music runtime file is missing")
        for part in [candidate,*candidate.parents]:
            if part==root:break
            if part.is_symlink() or getattr(part.lstat(),"st_file_attributes",0)&0x400:
                raise ValueError("Music runtime file indirection is not allowed")
        path=candidate.resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError("Music runtime file is missing or escaped")
        return path
    executable=runtime_file(value["executable"]) if portable else Path(value["executable"]).resolve()
    adapter=runtime_file(value["adapter"]) if portable else Path(value["adapter"]).resolve()
    if not root.is_dir() or any(not path.is_relative_to(root) or not path.is_file() for path in (executable,adapter)):
        raise ValueError("Installed music runtime is missing owned files")
    if not isinstance(value["files"],dict) or not value["files"]:raise ValueError("Music runtime requires a file inventory")
    hashes={}
    for name,pin in value["files"].items():
        path=runtime_file(name)
        digest=pin.get("sha256") if isinstance(pin,dict) else pin
        if portable and (not isinstance(pin,dict) or type(pin.get("bytes")) is not int or path.stat().st_size!=pin["bytes"]):
            raise ValueError("Portable music runtime file size differs")
        if not isinstance(digest,str) or not re.fullmatch(r"[a-f0-9]{64}",digest) or _digest(path)!=digest:
            raise ValueError("Installed music runtime changed after preparation")
        hashes[name]=digest
    if portable:
        actual=set()
        for path in root.rglob("*"):
            if path.is_symlink() or getattr(path.lstat(),"st_file_attributes",0)&0x400:
                raise ValueError("Portable music runtime contains file indirection")
            if path.is_file():actual.add(path.relative_to(root).as_posix())
        if actual!=set(hashes)|{"runtime.json"}:raise ValueError("Portable music runtime inventory membership differs")
        loader_files=[name for name in hashes if name.lower().endswith(_LOADER_SUFFIXES)] or [value["executable"]]
        longest_name=max(loader_files,key=len)
        longest=len(str(root/longest_name))
        if longest>MAX_RUNTIME_PATH:
            budget=MAX_RUNTIME_PATH-(longest-len(str(root)))
            raise ValueError(f"Portable music runtime path too long for Windows executable/DLL loading: {longest} characters ({longest_name}); "
                             f"install the application under a directory whose runtime folder path is at most {budget} characters (limit {MAX_RUNTIME_PATH})")
    for path in (executable,adapter):
        key=path.relative_to(root).as_posix()
        if hashes.get(key)!=_digest(path):raise ValueError("Installed music runtime executable/adapter is not hash-bound")
    return {**value,"destination":str(root),"files":hashes,"receipt_path":str(location),
            "receipt_sha256":_digest(location),"portable":portable},executable,adapter


class AceStepModel:
    def __init__(self,spec):
        if spec.backend!="ace-step" or spec.task!="music":raise ValueError("AceStepModel requires backend=ace-step and task=music")
        self.spec=spec;self.bundle=inspect_bundle(spec.path,verify=False)
        if self.bundle['manifest']['variant'] not in ACE_SUPPORTED_VARIANTS:
            raise ValueError('The installed Titanos music adapter supports XL SFT; this variant remains a discovery candidate')
        self.receipt,self.executable,self.adapter=runtime_receipt()
        if self.bundle['manifest']['variant'] not in self.receipt.get('supported_variants',ACE_SUPPORTED_VARIANTS):
            raise ValueError('The installed music runtime does not support this variant')
        self.bundle=inspect_bundle(spec.path)
        self.device=None;self.device_uuid=None;self.placement=None;self.last_generation=None

    def infer(self,task,data,emit=None,cancel=None):
        if task!="music":raise ValueError("ACE-Step adapter implements music only")
        if cancel is not None and cancel.is_set():raise EngineError("Music generation cancelled",499,"cancelled")
        settings=current_settings()
        data={**self.spec.defaults,**{key:value for key,value in data.items() if value is not None}}
        data.setdefault('steps',settings.default_steps)
        capabilities=self.bundle["manifest"]["capabilities"];mode=data.get("mode","instrumental")
        if mode not in {"instrumental","song"} or not capabilities.get(mode):
            raise ValueError("Selected music bundle does not support this music mode")
        lyrics=data.get("lyrics")
        if mode=="song" and lyrics and not capabilities.get("lyrics"):raise ValueError("Selected bundle does not support supplied lyrics")
        vocal_language=data.get("vocal_language")
        if vocal_language and vocal_language not in capabilities.get("vocal_languages",[]):raise ValueError("Unsupported vocal language for selected music bundle")
        if data.get("reference_audio_base64") is not None and not capabilities.get("reference_audio"):
            raise ValueError("Selected music bundle does not support reference audio")
        _validate_generation_limits(data,settings)
        duration=float(data.get("duration",self.spec.defaults.get("duration",30)))
        minimum=float(capabilities.get("duration_min",10));maximum=min(float(capabilities.get("duration_max",600)),settings.max_audio_seconds)
        if not minimum<=duration<=maximum:raise ValueError(f"Music duration must be {minimum:g}..{maximum:g} seconds")
        request={key:value for key,value in data.items() if key in {"prompt","mode","lyrics","duration","vocal_language","bpm","key_scale","seed","guidance_scale","steps",*PLANNER_FLAGS} and value is not None};_validate_planner_route(request)
        request.update(bundle=str(self.bundle["root"]),variant=self.bundle["manifest"]["variant"],duration=duration,mode=mode,
                       code_revision=ACE_CODE_REVISION,model_revision=self.bundle["manifest"]["model_revision"],
                       offline=True)
        with tempfile.TemporaryDirectory(prefix="titanos-music-") as folder:
            root=Path(folder);request_path=root/"request.json"
            env=_child_environment(self.executable,root)
            contract=_device_contract(self.spec,env,settings)
            request["runtime_contract"]=contract
            env["TITANOS_MUSIC_BUNDLE"]=str(self.bundle["root"])
            for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","TITANOS_THREADS"):
                env[name]=str(contract["threads"])
            request_path.write_text(json.dumps(request),encoding="utf-8")
            output=root/"output.wav";result_path=root/"result.json";progress_path=root/"progress.json"
            reference=data.get("reference_audio_base64")
            if reference is not None:
                try:encoded=reference.split(",",1)[1] if reference.startswith("data:audio/") and ";base64," in reference else reference;raw=base64.b64decode(encoded,validate=True)
                except (ValueError,binascii.Error,IndexError) as exc:raise ValueError("Reference audio must be valid base64 audio") from exc
                if not raw or len(raw)>settings.max_body_bytes:raise ValueError("Reference audio is empty or oversized")
                reference_path=root/"reference.audio";reference_path.write_bytes(raw);request["reference_audio"]=str(reference_path)
                request_path.write_text(json.dumps(request),encoding="utf-8")
            observer=None
            try:
                from .jobs import _native_generation_observer
                observer=_native_generation_observer.get()
            except ImportError:pass
            last={}
            def check():
                if cancel is not None and cancel.is_set():raise EngineError("Music generation cancelled",499,"cancelled")
            def observe():
                nonlocal last
                if not 0<(_file_size(progress_path) or 0)<=65536:return  # gone mid-replace or oversized: next poll
                try:value=json.loads(progress_path.read_text(encoding="utf-8"))
                except (OSError,ValueError):return
                if value==last:return
                if not isinstance(value,dict) or set(value)-{"stage","completed","total","native_cancelled"}:return
                last=value
                if observer:
                    job_id,callback=observer;callback({"job_id":job_id,"backend":"ace-step","stage":value.get("stage"),
                        "completed":value.get("completed"),"total":value.get("total"),"native_cancelled":value.get("native_cancelled",False)})
            try:
                run_owned([*_python_command(self.executable,self.adapter),"--request",str(request_path),"--output",str(output),
                          "--result",str(result_path),"--progress",str(progress_path)],root,env,check,
                          settings.max_music_output_bytes,label="Music adapter",observe=observe)
            except EngineError as exc:
                returncode=getattr(exc,"returncode",None)
                if exc.code=="native_process_failed" and returncode is not None:
                    # The request directory is still alive here: retain the bounded diagnostic first, then surface a pre-load
                    # placement refusal verbatim as before; every other child exit stays generic, naming only the exit code and
                    # the diagnostic file's relative name inside TITANOS_HOME (never log text or an absolute path).
                    try:
                        name=_retain_failure(exc,(result_path,progress_path),{"executable":self.executable,"adapter":self.adapter},
                                             contract,request,observer[0] if observer else None)
                    except Exception:name=None  # noqa: BLE001 - retention never replaces the child's failure (or a refusal) with a 500
                    _raise_refusal(result_path,exc)
                    detail=f"diagnostic {FAILURE_DIAGNOSTICS_DIR.as_posix()}/{name}" if name else "diagnostic_unavailable"
                    failure=EngineError(f"Music adapter process exited unsuccessfully (exit code {returncode}; {detail})",502,"native_process_failed")
                    failure.returncode=returncode;failure.stderr_tail=getattr(exc,"stderr_tail","");failure.diagnostic=name
                    raise failure from exc
                if exc.code=="native_process_failed":_raise_refusal(result_path,exc)
                raise
            check()
            if not output.is_file() or output.stat().st_size>settings.max_music_output_bytes or not result_path.is_file():
                raise EngineError("Music adapter produced no bounded result",502,"invalid_backend_response")
            with wave.open(str(output),"rb") as stream:
                channels,rate,frames,width=stream.getnchannels(),stream.getframerate(),stream.getnframes(),stream.getsampwidth()
            if (channels not in {1,2} or width!=2 or rate<8000 or rate>192000 or frames<=0
                    or frames>settings.max_audio_seconds*rate or frames*channels>settings.max_music_samples):
                raise EngineError("Music adapter returned invalid audio metadata",502,"invalid_backend_response")
            metadata=json.loads(result_path.read_text(encoding="utf-8"))
            allowed={"effective","evidence","native_format"}
            if not isinstance(metadata,dict) or set(metadata)-allowed:raise EngineError("Music adapter returned invalid metadata",502,"invalid_backend_response")
            runtime=_verify_runtime(metadata,contract)
            self.device=runtime["resolved_device"];self.device_uuid=runtime["device_uuid"];self.placement=runtime["placement"]["resolved"]
            self.last_generation={"sample_rate":rate,"channels":channels,"frames":frames,"duration":frames/rate,
                                  "effective":metadata.get("effective",{}),"evidence":metadata.get("evidence",{})}
            return {"wav":output.read_bytes(),"sample_rate":rate,"channels":channels,"duration":frames/rate,
                    "native_format":metadata.get("native_format","pcm16-wav"),
                    "effective":metadata.get("effective",{}),"evidence":metadata.get("evidence",{})}

    def details(self):
        return {"backend":"ace-step","task":"music","device":self.device,"device_uuid":self.device_uuid,
                "placement":self.placement,"requested_placement":self.spec.placement or "auto",
                "requested_device":self.spec.device,"requested_dtype":self.spec.dtype,
                "device_evidence":"child runtime and parameter placement" if self.last_generation else "not yet observed",
                "variant":self.bundle["manifest"]["variant"],
                "source":{"repository":self.spec.source_repo,"revision":self.spec.source_revision},
                "components":self.bundle["files"],"runtime":{"code_revision":self.receipt["code_revision"],
                "executable_sha256":self.receipt["files"].get(Path(self.executable).relative_to(self.receipt["destination"]).as_posix())},
                "capabilities":self.bundle["manifest"]["capabilities"],"last_generation":self.last_generation,
                "qualification":None,"qualification_note":"Configured bundle/runtime identity is not listening-quality or target-device proof"}

    def close(self):pass


def _file_size(path):
    """path's size, or None while it does not exist or cannot be read (the adapter replaces progress.json while the parent
    polls it; Windows can report a file being replaced as missing or access-denied for a moment)."""
    try:
        return path.stat().st_size
    except OSError:
        return None


# ACE GenerationParams planner route flags a music request or a registry default may set; the child resolves absent flags to
# thinking on with every use_cot_* off (tools/music_runtime/ace_adapter.py PLANNER_DEFAULTS) and re-checks these rules.
PLANNER_FLAGS=("thinking","use_cot_metas","use_cot_caption","use_cot_language")


def _validate_planner_route(request):
    """Refuse planner flag combinations the pinned ACE pipeline would silently ignore, before the child starts."""
    flags={name:request[name] for name in PLANNER_FLAGS if name in request}
    if any(type(value) is not bool for value in flags.values()):raise ValueError("Music planner flags must be true or false")
    metas=flags.get("use_cot_metas",False)
    if not metas and (flags.get("use_cot_caption") or flags.get("use_cot_language")):
        raise ValueError("Music use_cot_caption and use_cot_language need use_cot_metas")
    if not flags.get("thinking",True) and not metas:raise ValueError("Music thinking=false needs use_cot_metas so the planner still runs")
