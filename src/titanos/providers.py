"""Task-first model providers with immutable, secret-free acquisition plans."""
from __future__ import annotations

import hashlib
from importlib import metadata as package_metadata
from importlib.util import find_spec
import json
import logging
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
from typing import Any, Literal
import unicodedata
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

import httpx
from huggingface_hub import HfApi, get_token, snapshot_download
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .link_policy import os_alias

ProviderId = Literal["huggingface", "github-sherpa", "github-sherpa-diarization", "ace-step"]
SAFE_NAME = r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$"
HF_TASKS = {
    "chat": "text-generation",
    "vision": "image-text-to-text",
    "embeddings": "feature-extraction",
    "transcription": "automatic-speech-recognition",
    "speech": "text-to-speech",
    "image": "text-to-image",
    "video": "text-to-video",
    "music": "text-to-audio",
    "translation": "translation",
    "image-to-text": "image-to-text",
    "object-detection": "object-detection",
    "sentiment": "text-classification",
    "rerank": "text-ranking",
    "pair-classification": "text-classification",
    "depth-estimation": "depth-estimation",
    "image-segmentation": "image-segmentation",
}
TRANSFORMERS_TASKS = {
    "chat", "vision", "embeddings", "transcription", "speech", "translation", "image-to-text",
    "object-detection", "sentiment", "rerank", "pair-classification", "depth-estimation",
    "image-segmentation",
}
RUNTIME_MODULES = {
    "transformers": ["torch", "transformers"],
    "diffusers": ["torch", "diffusers", "transformers"],
    "onnx": ["onnxruntime"],
    "onnx-diffusers": ["onnxruntime", "diffusers"],
    "openvino": ["openvino"],
    "openvino-genai": ["openvino_genai"],
    "sherpa-onnx": ["sherpa_onnx"],
}


def runtime_recommendation(request):
    if request.backend=="ace-step":
        from .music_backend import ACE_SUPPORTED_VARIANTS
        if request.identifier not in ACE_SUPPORTED_VARIANTS:
            return {"family":"ace-step","status":"unsupported_adapter","requested_device":request.device,
                    "supported_variants":list(ACE_SUPPORTED_VARIANTS),"qualification":None,
                    "note":"This pinned model is a discovery candidate; the current trusted Titanos adapter implements XL SFT only"}
        try:
            from .music_backend import runtime_receipt
            receipt,executable,_=runtime_receipt();status,missing="available",[]
            if request.identifier not in receipt.get('supported_variants',ACE_SUPPORTED_VARIANTS):
                status,missing="unsupported_adapter",["runtime supporting the selected model variant"]
        except (OSError,ValueError,KeyError):
            receipt,executable,status,missing=None,None,"missing_runtime",["managed ACE-Step runtime"]
        return {"family":"ace-step","status":status,"required_modules":[],"missing_modules":missing,
                "requested_device":request.device,"known_devices":[],"managed_executable":str(executable) if executable else None,
                "runtime_code_revision":receipt.get("code_revision") if receipt else None,
                "synthetic_check":"not_run","measured_evidence":None,"qualification":None,
                "note":"Isolated runtime receipt presence is not song quality or device execution proof"}
    if request.backend=="llama-cpp":
        profile=None
        try:
            from .native_install import executable_path
            executable=str(executable_path())
            status,missing="available",[]
            from .config import home
            try:
                receipt=json.loads((home()/'native-runtime.json').read_text(encoding='utf-8'))
                if receipt.get('executable')==executable and isinstance(receipt.get('profile'),str):
                    profile=receipt['profile']
            except (OSError,ValueError,TypeError):
                pass
        except (OSError,ValueError,KeyError):
            executable=None;status,missing="missing_runtime",["managed llama.cpp runtime"]
        return {"family":"llama-cpp","status":status,"required_modules":[],"missing_modules":missing,
                "requested_device":request.device,"known_devices":[],"managed_executable":executable,
                "managed_profile":profile,
                "synthetic_check":"not_run","measured_evidence":None,"qualification":None,
                "note":"Managed runtime presence is checked from its hash-bound receipt; accelerator support and model execution remain unqualified until load/inference"}
    modules = RUNTIME_MODULES.get(request.backend, [])
    missing = [name for name in modules if find_spec(name) is None]
    inventory_backend = "openvino" if request.backend.startswith("openvino") else "onnx" if request.backend.startswith("onnx") else "pytorch"
    devices = []
    try:
        from .devices import inventory
        snapshot = inventory(False, inventory_backend)
        devices = [{"identity": row.get("uuid") or row.get("runtime_address") or row.get("id"),
                    "name": row.get("name"), "present": row.get("present")}
                   for row in snapshot.get("devices", []) if row.get("present") is not False]
    except Exception:
        pass
    return {
        "family": request.backend, "status": "available" if not missing else "missing_runtime",
        "required_modules": modules, "missing_modules": missing, "requested_device": request.device,
        "known_devices": devices, "synthetic_check": "not_run", "measured_evidence": None,
        "qualification": None,
        "note": "Runtime/package and inventory presence are not model execution or quality proof; explicit unavailable devices never fall back silently",
    }


# Win32 (RtlIsDosDeviceName_U) compares the base name up to the first '.', after trimming trailing spaces and
# dots, so 'com1 .txt' and 'nul . txt' are device paths; the validator computes that base before matching.
WINDOWS_RESERVED_DEVICE = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])$", re.IGNORECASE)
WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"|?*' + chr(92))
# huggingface_hub interprets allow_patterns as fnmatch globs; a bracket class can select an unvalidated sibling
# (and a bracketed name never matches itself), so glob metacharacters are refused in acquisition names outright.
GLOB_METACHARACTERS = frozenset("[]{}")
# Unicode categories that render invisibly or reorder text (C0/C1 controls, format characters, line/paragraph separators).
INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
MAX_COMPONENT_LENGTH = 255  # characters and UTF-8 bytes; the smaller cross-platform NTFS/ext4/APFS bound


def _bounded(component):
    text = repr(component)
    return text if len(text) <= 64 else text[:61] + "..."


def validate_acquisition_path(value):
    """Reject a relative acquisition file name before any downloader may write it.

    The acquisition tree is shared across hosts, so Windows rules apply everywhere: drive-relative and
    alternate-data-stream names (any ':'), backslashes, absolute forms, dot/dot-dot/empty components,
    reserved device names, trailing dots or spaces, control or forbidden characters, leading '~'
    components and names that are not already NFC (NTFS is normalisation-agnostic and APFS is not, so a
    non-NFC name would not identify one file across the shared tree). Errors name only the offending
    component, never the whole selection."""
    if not isinstance(value, str) or not value:
        raise ValueError("Acquisition file name must be a non-empty relative path")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("Acquisition file name is not NFC-normalised: " + _bounded(value))
    if "\\" in value:
        raise ValueError("Acquisition file name contains a backslash: " + _bounded(value))
    if ":" in value:
        raise ValueError("Acquisition file name contains a drive or stream separator ':': " + _bounded(value))
    if value.startswith("/") or ntpath.isabs(value) or ntpath.splitdrive(value)[0]:
        raise ValueError("Acquisition file name is absolute or drive-relative: " + _bounded(value))
    for component in value.split("/"):
        if component == "":
            raise ValueError("Acquisition file name has an empty path component: " + _bounded(value))
        if component in {".", ".."}:
            raise ValueError("Acquisition file name has a '.' or '..' component: " + _bounded(component))
        if len(component) > MAX_COMPONENT_LENGTH or len(component.encode("utf-8")) > MAX_COMPONENT_LENGTH:
            raise ValueError("Acquisition file name component exceeds " + str(MAX_COMPONENT_LENGTH) + " characters or bytes: "
                             + _bounded(component))
        if component.startswith("~"):
            raise ValueError("Acquisition file name component starts with '~': " + _bounded(component))
        if component.endswith((".", " ")):
            raise ValueError("Acquisition file name component ends with '.' or ' ': " + _bounded(component))
        if WINDOWS_RESERVED_DEVICE.match(component.split(".", 1)[0].rstrip(" .")):
            raise ValueError("Acquisition file name component is a Windows reserved device name: " + _bounded(component))
        for char in component:
            if ord(char) < 0x20 or ord(char) == 0x7F or unicodedata.category(char) in INVISIBLE_CATEGORIES:
                raise ValueError("Acquisition file name component contains a control or invisible character: "
                                 + _bounded(component))
            if char in WINDOWS_FORBIDDEN_CHARACTERS:
                raise ValueError("Acquisition file name component contains a forbidden character "
                                 + repr(char) + ": " + _bounded(component))
            if char in GLOB_METACHARACTERS:
                raise ValueError("Acquisition file name component contains a glob metacharacter "
                                 + repr(char) + ": " + _bounded(component))
    return value


def literal_pattern(path):
    """Escape a validated file name for huggingface_hub allow_patterns (fnmatch globs) so it matches only itself.

    Defence in depth: the validator already refuses '[', ']', '*' and '?', so this is the identity for
    accepted names; it keeps a persisted or future name from ever acting as a glob."""
    return path.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")


def check_case_insensitive_collisions(paths):
    """Reject a selection whose names collide on a case-insensitive filesystem (Windows/macOS staging) or where one
    selected name is the directory prefix of another; either pair would be merged or fail when written."""
    seen = {}
    for path in paths:
        lowered = path.lower()
        if lowered in seen:
            raise ValueError("Acquisition selection has two file names that differ only in case: " + _bounded(path))
        seen[lowered] = path
    for lowered, path in seen.items():
        prefix = lowered + "/"
        if any(other.startswith(prefix) for other in seen):
            raise ValueError("Acquisition selection names a file that is also a directory of another file: " + _bounded(path))


def unplanned_snapshot_files(root, planned):
    """Regular files under a downloaded snapshot (excluding huggingface_hub's .cache) that the plan did not name.

    Symbolic links are not followed; a link to a directory is reported like an unplanned file."""
    extra = []
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [name for name in dirnames if name != ".cache"]
        for name in list(dirnames):
            if (here / name).is_symlink():
                dirnames.remove(name)
                extra.append((here / name).relative_to(root).as_posix())
        for name in filenames:
            relative = (here / name).relative_to(root).as_posix()
            if relative not in planned:
                extra.append(relative)
    return sorted(extra)


class ProviderFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=1024)
    size: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    url: str | None = None

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value):
        validate_acquisition_path(value)
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
            raise ValueError("Provider file paths must be safe relative POSIX paths")
        return value


class AcquisitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = Field(min_length=1, max_length=64)
    identifier: str = Field(min_length=1, max_length=512)
    revision: str | None = Field(default=None, min_length=1, max_length=256)
    registry_name: str = Field(pattern=SAFE_NAME)
    task: str = Field(min_length=1, max_length=64)
    backend: str = Field(min_length=1, max_length=64)
    device: str = Field(default="auto", max_length=128)
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    graph: str | None = Field(default=None, max_length=512)
    model_file: str | None = Field(default=None, max_length=512)
    mmproj: str | None = Field(default=None, max_length=512)
    variant: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    replace: bool = False

    @model_validator(mode="after")
    def compatible_shape(self):
        if self.provider == "github-sherpa":
            if self.task != "speech" or self.backend != "sherpa-onnx" or self.device not in {"auto", "cpu"}:
                raise ValueError("GitHub Sherpa acquisition supports CPU speech through backend=sherpa-onnx")
        elif self.provider == "github-sherpa-diarization":
            if (self.task != "diarization" or self.backend != "sherpa-onnx"
                    or self.device not in {"auto", "cpu"} or self.dtype != "auto"
                    or self.graph or self.model_file or self.mmproj or self.variant):
                raise ValueError("Pinned Sherpa diarization requires CPU/auto, dtype=auto and no component overrides")
        elif self.provider == "huggingface":
            if self.backend not in {"transformers","diffusers","onnx","onnx-diffusers",
                                    "openvino","openvino-genai","llama-cpp"}:
                raise ValueError("Hugging Face acquisition does not implement this Titanos backend")
            if self.backend == "transformers" and self.task not in TRANSFORMERS_TASKS:
                raise ValueError("Transformers does not implement this Titanos task")
            if self.backend == "diffusers" and self.task not in {"image", "video"}:
                raise ValueError("Diffusers acquisition supports image or video")
            tasks={"onnx":{"embeddings","tensor","vad"},"openvino":{"embeddings","tensor"},
                   "openvino-genai":{"chat","image"},"onnx-diffusers":{"image"}}
            if self.backend in tasks and self.task not in tasks[self.backend]:
                raise ValueError(f"{self.backend} does not implement this Titanos task")
            if self.backend=="llama-cpp":
                native_device=(self.device in {"auto","cpu"}
                    or re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',self.device)
                    or re.fullmatch(r'native:[A-Za-z][A-Za-z0-9_.-]*',self.device))
                if self.task not in {"chat","vision"} or not native_device or self.dtype!="auto":
                    raise ValueError("Managed llama.cpp acquisition requires chat/vision, dtype=auto, and auto, cpu, a full GPU UUID or native:DEVICE; configure multi-device groups after registration")
                if not self.model_file or (self.task=="vision" and not self.mmproj):
                    raise ValueError("Managed llama.cpp acquisition requires an explicit GGUF model file and vision projector")
        elif self.provider=="ace-step":
            if self.task!="music" or self.backend!="ace-step" or self.device not in {"auto","cpu"} and not self.device.startswith(("cuda","GPU-","HIP-")):
                raise ValueError("ACE-Step acquisition supports the music task on an explicit supported local device")
            if self.identifier not in {"xl-sft","xl-turbo","xl-base"}:
                raise ValueError("Unknown ACE-Step pinned bundle recipe")
        return self


class AcquisitionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    provider: str
    provider_title: str
    candidate_id: str
    registry_name: str
    replace: bool = False
    task: str
    backend: str
    device: str
    source_url: str
    requested_revision: str
    resolved_revision: str
    files: list[ProviderFile] = Field(min_length=1,max_length=10_000)
    expected_bytes: int | None = Field(default=None, ge=0)
    digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    archive_format: Literal["tar.bz2"] | None = None
    license: str | None = None
    gated: bool | str | None = None
    resume_supported: bool = False
    compatibility: dict[str, Any]
    runtime: dict[str, Any]
    model_spec: dict[str, Any]
    provider_data: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self):
        # Runtime presence/device inventory is a current observation, not remote
        # artifact identity. Submission replans and stores the trusted fresh view.
        return hashlib.sha256(self.model_dump_json(exclude={"provider_data","runtime"}).encode()).hexdigest()


class ImportedModelLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str | None
    identifier: str | None
    revision: str | None
    canonical_url: str
    compatibility: Literal["provider-recognized", "metadata-only"]


def _card_license(info):
    card = getattr(info, "card_data", None)
    if hasattr(card, "to_dict"):
        card = card.to_dict()
    if isinstance(card, dict):
        value = card.get("license")
        if isinstance(value, str) and value.strip():
            value=value.strip().lower()
            return None if value in {"other","unknown","custom","n/a","none","not specified","see license"} else value
    value = getattr(card, "license", None)
    if not isinstance(value,str) or not value.strip():return None
    value=value.strip().lower()
    return None if value in {"other","unknown","custom","n/a","none","not specified","see license"} else value


def _expected_backend(task, library=None):
    if library=="diffusers" or task in {"image","video","text-to-image","text-to-video","image-to-image"}:
        return "diffusers"
    return "transformers"


def _license_evidence(root):
    evidence=[]
    for path in sorted(root.rglob("*")):
        if len(evidence)>=64:break
        if path.is_file() and path.name.lower().startswith(("license","copying","notice","model_card","readme")):
            if path.stat().st_size<=1_000_000 and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()):
                raw=path.read_bytes()
                evidence.append({"path":path.relative_to(root).as_posix(),
                                 "sha256":hashlib.sha256(raw).hexdigest(),
                                 "text":raw.decode("utf-8","replace")[:20_000]})
    return evidence


def _hf_allowed(path, request):
    lower = path.lower()
    common = (
        lower.endswith((".json", ".txt", ".model", ".vocab", ".merges", ".jinja", ".tiktoken"))
        or "tokenizer" in lower or "preprocessor" in lower or "processor" in lower
        or Path(path).name.lower().startswith(("readme", "license", "notice"))
    )
    if request.backend == "llama-cpp":
        chosen = {request.model_file, request.mmproj} - {None}
        if path in chosen:
            return True
        if request.model_file:
            match = re.fullmatch(r"(.+)-\d{5}-of-(\d{5})\.gguf", request.model_file)
            if match and re.fullmatch(re.escape(match[1]) + r"-\d{5}-of-" + re.escape(match[2]) + r"\.gguf", path):
                return True
        return common
    if request.backend in {"onnx","onnx-diffusers"}:
        return common or lower.endswith((".onnx",".onnx_data",".data"))
    if request.backend in {"openvino","openvino-genai"}:
        return common or (("openvino" in PurePosixPath(lower).parts or PurePosixPath(lower).name.startswith("openvino_"))
                          and lower.endswith((".xml",".bin",".blob",".safetensors")))
    if lower.endswith(".safetensors"):
        if request.variant:
            return f".{request.variant.lower()}." in lower
        return not re.search(r"\.(?:fp16|bf16|int8|int4)\.",lower)
    return common


class ModelProvider:
    name = ""
    title = ""

    def descriptor(self):
        return {"id": self.name, "title": self.title, "operations": ["search", "inspect", "plan", "acquire"]}

    def search(self, query, task, limit=20):
        raise NotImplementedError

    def inspect(self, identifier, revision=None):
        raise NotImplementedError

    def plan(self, request: AcquisitionRequest) -> AcquisitionPlan:
        raise NotImplementedError

    def acquire(self, plan: AcquisitionPlan, staging: Path, progress, cancel):
        raise NotImplementedError

    def _base_plan(self, request, metadata, files, **extra):
        rows = [ProviderFile.model_validate(row) for row in files]
        expected = None if any(row.size is None for row in rows) else sum(row.size for row in rows)
        if expected is None:
            raise ValueError("Provider did not supply exact file sizes for a bounded acquisition plan")
        spec = {"task": request.task, "backend": request.backend, "device": request.device, "dtype": request.dtype}
        if request.graph:
            spec["graph"] = request.graph
        if request.variant:
            spec["load_options"] = {"variant": request.variant}
        if request.backend == "llama-cpp":
            spec["native"] = {key: value for key, value in {
                "model_file": request.model_file, "mmproj": request.mmproj}.items() if value}
        return AcquisitionPlan(
            provider=self.name, provider_title=self.title, candidate_id=request.identifier,
            registry_name=request.registry_name, replace=request.replace, task=request.task, backend=request.backend,
            device=request.device, source_url=metadata["source_url"],
            requested_revision=metadata["requested_revision"],
            resolved_revision=metadata["resolved_revision"], files=rows,
            expected_bytes=expected, license=metadata.get("license"), gated=metadata.get("gated"),
            compatibility={"status": "compatible_plan", "qualification": None,
                           "note": "File/task/backend checks are not model quality or hardware qualification"},
            runtime=metadata.get("runtime", {"status": "requires-packaged-runtime", "qualification": None}),
            model_spec=spec, provider_data=metadata.get("provider_data", {}), **extra)


class HuggingFaceProvider(ModelProvider):
    name = "huggingface"
    title = "Hugging Face"

    def __init__(self, token=None):
        self.token = token

    def _token(self):
        return get_token() if self.token is None else self.token

    def _api(self):
        return HfApi(token=self._token() or False)

    @staticmethod
    def _describe(info):
        recommended=("llama-cpp" if "gguf" in info.id.lower()
                     and getattr(info,"pipeline_tag",None) in {None,"text-generation","image-text-to-text"}
                     else _expected_backend(getattr(info, "pipeline_tag", None),getattr(info, "library_name", None)))
        return {
            "provider": "huggingface", "id": info.id, "name": info.id.rsplit("/", 1)[-1],
            "author": info.id.split("/", 1)[0] if "/" in info.id else None,
            "task": getattr(info, "pipeline_tag", None), "library": getattr(info, "library_name", None),
            "downloads": getattr(info, "downloads", None), "likes": getattr(info, "likes", None),
            "gated": getattr(info, "gated", None), "revision": getattr(info, "sha", None),
            "license": _card_license(info), "source_url": "https://huggingface.co/" + info.id,
            "recommended_backend":recommended,
            "compatibility": {"status": "candidate", "qualification": None,
                              "note": "Hub tags are discovery metadata, not Titanos runtime proof"},
        }

    def search(self, query, task, limit=20):
        if limit < 1 or limit > 100:
            raise ValueError("Search limit must be 1..100")
        pipeline = HF_TASKS.get(task, task)
        from .hub import hub_errors
        rows=[]
        with hub_errors():
            for info in self._api().list_models(search=query or None,pipeline_tag=pipeline,sort="downloads",
                                                direction=-1,limit=limit,full=True):
                row=self._describe(info);row["titanos_task"]=task
                if row["recommended_backend"]!="llama-cpp":row["recommended_backend"]=_expected_backend(task,row.get("library"))
                rows.append(row)
        return rows

    def inspect(self, identifier, revision=None):
        requested = revision or "main"
        from .hub import hub_errors
        with hub_errors():info = self._api().model_info(identifier, revision=requested, files_metadata=True)
        if not getattr(info, "sha", None):
            raise ValueError("Hugging Face did not resolve an immutable revision")
        files = [{"path": row.rfilename, "size": getattr(row, "size", None)}
                 for row in (getattr(info, "siblings", None) or [])]
        return {**self._describe(info), "requested_revision": requested,
                "resolved_revision": info.sha, "files": files,
                "source_url": "https://huggingface.co/" + identifier + "/tree/" + quote(info.sha, safe="")}

    def plan(self, request):
        metadata = self.inspect(request.identifier, request.revision)
        selected = sorted((row for row in metadata["files"] if _hf_allowed(row["path"], request)),
                          key=lambda row: row["path"].lower())
        for row in selected:
            validate_acquisition_path(row["path"])
        check_case_insensitive_collisions([row["path"] for row in selected])
        if request.backend == "llama-cpp" and not request.model_file:
            raise ValueError("llama-cpp acquisition requires an explicitly selected GGUF model_file")
        if not selected or (request.backend not in {"onnx", "openvino", "openvino-genai", "onnx-diffusers", "llama-cpp"}
                            and not any(row["path"].lower().endswith(".safetensors") for row in selected)):
            raise ValueError("No safe compatible checkpoint files were selected")
        if request.backend in {"onnx","onnx-diffusers"} and not any(row["path"].lower().endswith(".onnx") for row in selected):
            raise ValueError("ONNX acquisition requires at least one prepared .onnx graph")
        if request.backend in {"openvino","openvino-genai"} and not any(row["path"].lower().endswith(".xml") for row in selected):
            raise ValueError("OpenVINO acquisition requires at least one prepared .xml graph")
        metadata["runtime"] = runtime_recommendation(request)
        metadata["provider_data"] = {"repository": request.identifier}
        plan=self._base_plan(request, metadata, selected, resume_supported=True)
        plan.compatibility["selection"]="Pinned SafeTensors and required text/config assets were selected; pickle checkpoints and unrequested weight variants were excluded"
        return plan

    def acquire(self, plan, staging, progress, cancel):
        if cancel.is_set():
            raise RuntimeError("Acquisition cancelled")
        data = plan.provider_data
        from tqdm.auto import tqdm
        class ProviderProgress(tqdm):
            def __init__(self,*args,**kwargs):
                kwargs['disable']=True
                super().__init__(*args,**kwargs)
            def update(self,value=1):
                result=super().update(value)
                progress("downloading_files",int(self.n),int(self.total or len(plan.files)))
                if cancel.is_set():raise RuntimeError("Acquisition cancelled")
                return result
        for row in plan.files:
            validate_acquisition_path(row.path)
        check_case_insensitive_collisions([row.path for row in plan.files])
        progress("downloading_files", 0, len(plan.files))
        from .hub import hub_errors
        with hub_errors():
            path = snapshot_download(
                data["repository"], revision=plan.resolved_revision, token=self._token() or False,
                allow_patterns=[literal_pattern(row.path) for row in plan.files],local_dir=staging/"snapshot",
                tqdm_class=ProviderProgress,
            )
        if cancel.is_set():
            raise RuntimeError("Acquisition cancelled")
        progress("verifying", plan.expected_bytes or 0, plan.expected_bytes)
        root = Path(path)
        unplanned = unplanned_snapshot_files(root, {row.path for row in plan.files})
        if unplanned:
            raise ValueError("Downloaded snapshot contains files the plan did not name: " + ", ".join(unplanned[:8]))
        missing = [row.path for row in plan.files if not (root / row.path).is_file()]
        if missing:
            raise ValueError("Downloaded snapshot is incomplete: " + ", ".join(missing[:8]))
        integrity=[]
        for row in plan.files:
            file=root/row.path
            if file.is_symlink() or not file.resolve().is_relative_to(root.resolve()):
                raise ValueError("Downloaded snapshot contains an unsafe linked file: "+row.path)
            if row.size is not None and file.stat().st_size!=row.size:
                raise ValueError("Downloaded snapshot file size differs from the immutable plan: "+row.path)
            with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            integrity.append({'path':row.path,'bytes':file.stat().st_size,'sha256':digest})
        shutil.rmtree(root/".cache",ignore_errors=True)
        return {"path": str(root.resolve()), "license": plan.license,
                "license_evidence":_license_evidence(root), "integrity":integrity, "reused": False}


class AceStepProvider(ModelProvider):
    name="ace-step"
    title="ACE-Step 1.5 · official pinned bundles"
    code_revision="ca1e85fe9430179831e6bc6be790c332190a3866"
    core=("ACE-Step/Ace-Step1.5","19671f406d603126926c1b7e2adc169acbcade22")
    planner=("ACE-Step/acestep-5Hz-lm-4B","0a3ec94b557aea7d508da38b31cfe7341f6ff737")
    # Model language codes from the pinned implementation's constants.py.
    vocal_languages="ar az bg bn ca cs da de el en es fa fi fr he hi hr ht hu id is it ja ko la lt ms ne nl no pa pl pt ro ru sa sk sr sv sw ta te th tl tr uk ur vi yue zh unknown".split()
    variants={
        "xl-sft":("ACE-Step/acestep-v15-xl-sft","d06de46b4622f781cf07f4a013a67d591ca52819","Quality candidate · upstream XL SFT claim"),
        "xl-turbo":("ACE-Step/acestep-v15-xl-turbo","d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee","Fast candidate · upstream XL Turbo claim"),
        "xl-base":("ACE-Step/acestep-v15-xl-base","220c1166efbdd9583eafcb12eb160594bbfcb241","Base editing-capable candidate"),
    }

    @staticmethod
    def _allowed(path,role):
        candidate=PurePosixPath(path)
        if (candidate.is_absolute() or any(part in {"", ".", ".."} for part in path.split("/"))
                or "\\" in path or ":" in path):return False
        name=candidate.name.lower()
        notice=(len(candidate.parts)==1 and
                (name in {"readme","license","notice","copying"} or
                 name in {prefix+suffix for prefix in ("readme","license","notice","copying")
                          for suffix in (".md",".txt")}))
        if notice:return True
        if role=="core":
            if candidate.parts[0] not in {"Qwen3-Embedding-0.6B","vae"} or len(candidate.parts)!=2:return False
        elif role in {"dit","planner"}:
            if len(candidate.parts)!=1:return False
        else:return False
        # Only this pinned tensor data file is loaded through weights_only=True.
        if role=="dit" and path=="silence_latent.pt":return True
        return candidate.suffix.lower() in {".safetensors",".json",".txt",".model",".vocab",".merges",".tiktoken",".jinja"}

    @staticmethod
    def _require_components(role,files):
        names={row["path"].removeprefix(role+"/") for row in files}
        if role=="core":
            required={"vae/config.json","vae/diffusion_pytorch_model.safetensors",
                      "Qwen3-Embedding-0.6B/config.json","Qwen3-Embedding-0.6B/model.safetensors",
                      "Qwen3-Embedding-0.6B/tokenizer.json","Qwen3-Embedding-0.6B/tokenizer_config.json"}
        else:
            required={"config.json"}
            if role=="dit":required.add("silence_latent.pt")
            else:required.update({"tokenizer.json","tokenizer_config.json"})
            weights={name for name in names if name.endswith(".safetensors")}
            if not weights:raise ValueError("ACE-Step component has no safetensors weights: "+role)
            if any("-of-" in name for name in weights):required.add("model.safetensors.index.json")
        if required-names:
            raise ValueError("ACE-Step component is incomplete: "+role+": "+", ".join(sorted(required-names)))

    def search(self,query,task,limit=20):
        if task not in {"","music"}:return []
        value=query.casefold().strip();rows=[]
        for identifier,(repo,revision,note) in self.variants.items():
            if value and value not in (identifier+" "+repo+" "+note).casefold():continue
            rows.append({"provider":self.name,"id":identifier,"name":"ACE-Step 1.5 "+identifier.upper(),
                "author":"ACE-Step","task":"music","titanos_task":"music","backend":"ace-step",
                "recommended_backend":"ace-step","revision":revision,"license":"mit","gated":False,
                "source_url":"https://huggingface.co/"+repo,
                "compatibility":{"status":"candidate","qualification":None,
                    "note":note+"; selection is not local listening qualification"}})
        return rows[:limit]

    def _metadata(self,identifier,revision=None):
        if identifier not in self.variants:raise ValueError("Unknown ACE-Step pinned bundle recipe")
        variant_repo,variant_revision,note=self.variants[identifier]
        if revision is not None and revision!=variant_revision:
            raise ValueError("ACE-Step bundle revision must match the reviewed pinned recipe")
        api=HfApi(token=get_token() or False);values=[]
        from .hub import hub_errors
        with hub_errors():
            for role,(repo,pin) in (("core",self.core),("dit",(variant_repo,variant_revision)),("planner",self.planner)):
                info=api.model_info(repo,revision=pin,files_metadata=True)
                if getattr(info,"sha",None)!=pin:raise ValueError("ACE-Step repository did not resolve the reviewed immutable revision")
                files=[{"path":f"{role}/{row.rfilename}","size":getattr(row,"size",None)}
                       for row in (getattr(info,"siblings",None) or []) if self._allowed(row.rfilename,role)]
                self._require_components(role,files)
                values.extend(files)
        return {"source_url":"https://huggingface.co/"+variant_repo,"requested_revision":revision or variant_revision,
                "resolved_revision":f"core:{self.core[1]};dit:{variant_revision};planner:{self.planner[1]}","files":sorted(values,key=lambda row:row["path"].lower()),
                "license":"MIT AND Apache-2.0","gated":False,"runtime":None,
                "provider_data":{"variant":identifier,"repositories":{"core":{"id":self.core[0],"revision":self.core[1]},
                    "dit":{"id":variant_repo,"revision":variant_revision},
                    "planner":{"id":self.planner[0],"revision":self.planner[1]}}},"note":note}

    def inspect(self,identifier,revision=None):
        return {"provider":self.name,"id":identifier,"name":"ACE-Step 1.5 "+identifier.upper(),
                "task":"music","recommended_backend":"ace-step",**self._metadata(identifier,revision)}

    def plan(self,request):
        metadata=self._metadata(request.identifier,request.revision)
        metadata["runtime"]=runtime_recommendation(request)
        plan=self._base_plan(request,metadata,metadata["files"],resume_supported=True)
        plan.compatibility.update(selection="Selected VAE, text encoder, XL DiT and 4B planner; unrelated default models and checkpoint code are excluded",
                                  code_revision=self.code_revision)
        return plan

    def acquire(self,plan,staging,progress,cancel):
        root=staging/"bundle";root.mkdir(parents=True,exist_ok=True);integrity=[];evidence=[]
        repositories={role:dict(row) for role,row in plan.provider_data["repositories"].items()}
        total=len(plan.files);completed=0
        for role in ("core","dit","planner"):
            rows=[row for row in plan.files if row.path.startswith(role+"/")]
            target=root/role
            if cancel.is_set():raise RuntimeError("Acquisition cancelled")
            for row in rows:validate_acquisition_path(row.path)
            path=Path(snapshot_download(repositories[role]["id"],revision=repositories[role]["revision"],
                token=get_token() or False,allow_patterns=[literal_pattern(row.path.removeprefix(role+"/")) for row in rows],local_dir=target))
            for row in rows:
                file=path/row.path.removeprefix(role+"/")
                if file.is_symlink() or not file.resolve().is_relative_to(path.resolve()) or not file.is_file():
                    raise ValueError("ACE-Step snapshot contains an unsafe or missing file")
                if row.size is not None and file.stat().st_size!=row.size:raise ValueError("ACE-Step component size changed")
                with file.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
                integrity.append({"path":row.path,"bytes":file.stat().st_size,"sha256":digest});completed+=1
                progress("downloading_files",completed,total)
            shutil.rmtree(path/".cache",ignore_errors=True)
            inventory=target/"component.inventory.json"
            inventory.write_text(json.dumps({"repository":repositories[role]["id"],"revision":repositories[role]["revision"],
                "files":[row for row in integrity if row["path"].startswith(role+"/")]},indent=2),encoding="utf-8")
            raw=inventory.read_bytes()
            repositories[role]["inventory_path"]=f"{role}/component.inventory.json"
            repositories[role]["inventory_sha256"]=hashlib.sha256(raw).hexdigest()
            evidence.extend([{**item,"path":role+"/"+item["path"]} for item in _license_evidence(target)])
        capabilities={"instrumental":True,"song":True,"lyrics":True,"vocal_languages":list(self.vocal_languages),
                      "duration_min":10,"duration_max":600,"bpm":True,"key_scale":True,"reference_audio":False}
        manifest={"schema_version":1,"family":"ace-step-1.5","variant":plan.candidate_id,
            "code_revision":self.code_revision,"model_revision":repositories["dit"]["revision"],
            "components":{role:{"path":row["inventory_path"],"sha256":row["inventory_sha256"],
                "revision":row["revision"],"license":"MIT AND Apache-2.0" if role=="core" else "mit"} for role,row in repositories.items()},
            "capabilities":capabilities,"license":"MIT AND Apache-2.0"}
        (root/"titanos-music.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        return {"path":str(root.resolve()),"license":manifest["license"],"license_evidence":evidence,"integrity":integrity,"reused":False}


class GitHubSherpaProvider(ModelProvider):
    name = "github-sherpa"
    title = "GitHub · Sherpa ONNX TTS"
    api_url = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/tts-models"
    allowed_download_hosts = {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}

    def __init__(self, release_loader=None):
        self.release_loader = release_loader or self._load_release

    def _load_release(self):
        response = httpx.get(self.api_url, headers={"Accept": "application/vnd.github+json",
                                                    "User-Agent": "Titanos/0.5"}, timeout=30,
                             follow_redirects=False,trust_env=False)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _rows(release):
        rows = []
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if not name.endswith(".tar.bz2"):
                continue
            identifier = name[:-8]
            if not identifier.startswith("vits-"):
                continue
            digest = str(asset.get("digest") or "").removeprefix("sha256:") or None
            rows.append({
                "provider": "github-sherpa", "id": identifier, "name": identifier,
                "author": "k2-fsa/sherpa-onnx", "task": "speech", "backend": "sherpa-onnx",
                "recommended_backend": "sherpa-onnx", "revision": release.get("tag_name"),
                "resolved_revision": f"github-release:{release.get('id')}:asset:{asset.get('id')}:{digest or 'missing-digest'}",
                "size": asset.get("size"), "downloads": asset.get("download_count"),
                "license": None, "gated": False, "source_url": asset.get("browser_download_url"),
                "compatibility": {"status": "candidate", "qualification": None,
                                  "note": "Archive structure and model license are verified before activation"},
                "_asset": asset, "_release": release,
            })
        return rows

    def search(self, query, task, limit=20):
        if task not in {"speech", ""}:
            return []
        value = query.lower().strip()
        rows = [row for row in self._rows(self.release_loader()) if not value or value in row["id"].lower()]
        return [{key: item for key, item in row.items() if not key.startswith("_")} for row in rows[:limit]]

    def inspect(self, identifier, revision=None):
        release = self.release_loader()
        if revision is not None and revision != release.get("tag_name"):
            raise ValueError("Requested GitHub release tag does not exist; no fallback was used")
        row = next((item for item in self._rows(release) if item["id"] == identifier), None)
        if row is None:
            raise ValueError("Sherpa release asset not found")
        asset = row["_asset"]
        return {**{key: value for key, value in row.items() if not key.startswith("_")},
                "requested_revision": revision or release["tag_name"],
                "files": [{"path": asset["name"], "size": asset.get("size"),
                           "sha256": str(asset.get("digest") or "").removeprefix("sha256:") or None,
                           "url": asset.get("browser_download_url")}],
                "provider_data": {"release_id": release["id"], "asset_id": asset["id"],
                                  "download_url": asset["browser_download_url"]}}

    def plan(self, request):
        metadata = self.inspect(request.identifier, request.revision)
        digest = metadata["files"][0].get("sha256")
        if not digest or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Sherpa release asset requires an immutable SHA-256 digest")
        metadata["runtime"] = runtime_recommendation(request)
        metadata["runtime"]["runtime_pin"] = {"sherpa-onnx": "1.13.8", "sherpa-onnx-core": "1.13.8"}
        return self._base_plan(request, metadata, metadata["files"], digest=digest,
                               archive_format="tar.bz2", resume_supported=True)

    def acquire(self, plan, staging, progress, cancel):
        logger=logging.getLogger('httpx');previous=logger.level;logger.setLevel(logging.WARNING)
        try:
            try:return self._acquire(plan,staging,progress,cancel)
            except httpx.HTTPError:
                raise ValueError("GitHub release download failed; check connectivity or retry later") from None
            except ValueError as exc:
                if "exceeded the immutable planned byte size" in str(exc):
                    (staging/plan.files[0].path).unlink(missing_ok=True)
                raise
        finally:logger.setLevel(previous)

    def _acquire(self, plan, staging, progress, cancel):
        staging.mkdir(parents=True, exist_ok=True)
        archive = staging / plan.files[0].path
        offset = archive.stat().st_size if archive.exists() else 0
        complete=False
        if plan.expected_bytes is not None and offset==plan.expected_bytes:
            with archive.open('rb') as stream:complete=hashlib.file_digest(stream,'sha256').hexdigest()==plan.digest
        append=False
        if not complete:
            headers = {"User-Agent": "Titanos/0.5"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with httpx.stream("GET", plan.provider_data["download_url"], headers=headers, timeout=60,
                              follow_redirects=True,trust_env=False) as response:
                response.raise_for_status()
                final = urlsplit(str(response.url))
                if final.scheme != "https" or final.hostname not in self.allowed_download_hosts:
                    raise ValueError("GitHub asset redirected outside the approved HTTPS release hosts")
                append = offset > 0 and response.status_code == 206
                if not append:
                    offset = 0
                mode = "ab" if append else "wb"
                with archive.open(mode) as output:
                    for chunk in response.iter_bytes(1024 * 256):
                        if cancel.is_set():
                            raise RuntimeError("Acquisition cancelled")
                        if plan.expected_bytes is not None and offset+len(chunk)>plan.expected_bytes:
                            raise ValueError("GitHub asset exceeded the immutable planned byte size")
                        output.write(chunk)
                        offset += len(chunk)
                        progress("downloading", offset, plan.expected_bytes)
        else:progress("verifying",offset,plan.expected_bytes)
        if plan.expected_bytes is not None and archive.stat().st_size != plan.expected_bytes:
            raise ValueError("Downloaded Sherpa archive size differs from the immutable plan")
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != plan.digest:
            archive.unlink(missing_ok=True)
            raise ValueError("Downloaded Sherpa archive SHA-256 mismatch")
        extract_root = staging / "extracted"
        if extract_root.exists():shutil.rmtree(extract_root)
        extract_root.mkdir(exist_ok=True)
        progress("extracting", 0, None)
        with tarfile.open(archive, "r:bz2") as bundle:
            members = bundle.getmembers()
            if len(members) > 50_000:
                raise ValueError("Sherpa archive contains too many entries")
            total = 0
            for member in members:
                path = PurePosixPath(member.name)
                if ("\\" in member.name or ":" in member.name or path.is_absolute() or ".." in path.parts
                        or member.issym() or member.islnk() or member.isdev()):
                    raise ValueError("Sherpa archive contains an unsafe path or link")
                total += member.size
                if total > 4 * 1024**3:
                    raise ValueError("Sherpa archive expands beyond the configured safety budget")
            for index, member in enumerate(members):
                if cancel.is_set():
                    raise RuntimeError("Acquisition cancelled")
                target = extract_root.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(member)
                    if source is None:
                        raise ValueError("Sherpa archive member could not be read")
                    with source, target.open("wb") as output:
                        while True:
                            chunk = source.read(1024 * 256)
                            if not chunk:
                                break
                            if cancel.is_set():
                                raise RuntimeError("Acquisition cancelled")
                            output.write(chunk)
                progress("extracting", index + 1, len(members))
        children = [path for path in extract_root.iterdir()]
        model_root = children[0] if len(children) == 1 and children[0].is_dir() else extract_root
        licenses = _license_evidence(model_root)
        return {"path": str(model_root.resolve()), "license": None,
                "license_evidence": licenses, "archive_sha256": digest,
                "integrity":[{'path':plan.files[0].path,'bytes':archive.stat().st_size,'sha256':digest}],
                "reused": append}


class GitHubSherpaDiarizationProvider(ModelProvider):
    """One reviewed segmentation/embedding pair; weights stay external."""
    name = "github-sherpa-diarization"
    title = "Sherpa ONNX · anonymous diarization"
    allowed_hosts = {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com",
                     "raw.githubusercontent.com", "huggingface.co", "modelscope.cn"}

    def __init__(self, catalog=None):
        self.catalog = catalog if catalog is not None else json.loads(
            Path(__file__).with_name("data").joinpath("diarization-profiles.json").read_text(encoding="utf-8"))
        if self.catalog.get("schema_version") != 1:
            raise ValueError("Unsupported diarization catalog schema")

    def _profile(self, identifier, revision=None):
        profile = next((p for p in self.catalog["profiles"] if p["id"] == identifier), None)
        if profile is None or revision not in {None, profile["revision"]}:
            raise ValueError("Unknown pinned diarization profile or revision; no fallback was used")
        return profile

    def search(self, query, task, limit=20):
        if task not in {"", "diarization"}:
            return []
        query = query.casefold().strip()
        return [self.inspect(p["id"]) for p in self.catalog["profiles"]
                if not query or query in (p["id"] + " " + p["title"]).casefold()][:limit]

    def inspect(self, identifier, revision=None):
        p = self._profile(identifier, revision)
        digest = hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"provider": self.name, "id": p["id"], "name": p["title"], "author": "pyannote / 3D-Speaker / sherpa-onnx",
                "task": "diarization", "backend": "sherpa-onnx", "recommended_backend": "sherpa-onnx",
                "revision": p["revision"], "requested_revision": revision or p["revision"],
                "resolved_revision": "diarization-profile:" + p["revision"] + ":" + digest,
                "source_url": "https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html",
                "size": sum(f["size"] for f in p["downloads"]), "license": p["license"], "gated": False,
                "files": p["downloads"], "components": p["components"],
                "compatibility": {"status": "candidate", "qualification": None,
                    "note": "Pinned CPU model pair; anonymous within-recording speakers, at most two simultaneous voices. Not an accuracy claim."},
                "provider_data": {"profile_id": p["id"], "profile_revision": p["revision"]}}

    def plan(self, request):
        # Validate even for direct Python callers, not only HTTP's request model.
        request = AcquisitionRequest.model_validate(request.model_dump())
        if request.provider != self.name:
            raise ValueError("Diarization request names another provider")
        p = self._profile(request.identifier, request.revision)
        metadata = self.inspect(request.identifier, request.revision)
        modules = ["sherpa_onnx"]
        missing = [name for name in modules if find_spec(name) is None]
        versions = {}
        for name in p["runtime_pin"]:
            try: versions[name] = package_metadata.version(name)
            except package_metadata.PackageNotFoundError: versions[name] = None
        status = "missing_runtime" if missing or any(v is None for v in versions.values()) else (
            "available" if versions == p["runtime_pin"] else "unsupported_runtime")
        metadata["runtime"] = {"family": "sherpa-onnx", "status": status, "required_modules": modules,
            "missing_modules": missing, "runtime_pin": p["runtime_pin"], "observed_package_versions": versions,
            "requested_device": "cpu", "known_devices": [], "qualification": None,
            "note": "Package presence is not model/device execution or diarization quality proof"}
        request = request.model_copy(update={"device": "cpu"})
        return self._base_plan(request, metadata, metadata["files"], resume_supported=True)

    @classmethod
    def _url(cls, value):
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
        if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in {None, 443}
                or not (host in cls.allowed_hosts or host.endswith(".hf.co") or host.endswith(".huggingface.co"))):
            raise ValueError("Diarization asset requires an approved public HTTPS model host")
        return value

    @staticmethod
    def _ordinary(path):
        for parent in (path, *path.parents):
            if (parent.is_symlink() or (parent.exists() and getattr(parent.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))) \
                    and not (parent != path and os_alias(parent, parent.lstat())):  # macOS /var -> /private/var
                raise ValueError("Linked diarization staging paths are not allowed")

    def _download_file(self, row, target, progress, cancel, base, total):
        self._ordinary(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if cancel.is_set(): raise RuntimeError("Acquisition cancelled")
        offset = target.stat().st_size if target.exists() else 0
        if offset >= row.size:
            with target.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if offset == row.size and digest == row.sha256:
                progress("verifying", base + offset, total)
                return True
            target.unlink()
            raise ValueError("Cached diarization component integrity failed; retry to download it again")
        url = self._url(row.url)
        headers = {"User-Agent": "Titanos/0.5"}
        if offset: headers["Range"] = f"bytes={offset}-"
        with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as client:
            for redirect in range(6):
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        if redirect == 5: raise ValueError("Too many diarization asset redirects")
                        url = self._url(urljoin(url, response.headers.get("location", "")))
                        continue
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("content-range", ""))
                        if not match or int(match[1]) != offset or int(match[3]) != row.size:
                            raise ValueError("Diarization resume range does not match the pinned file")
                    elif response.status_code == 206:
                        raise ValueError("Unexpected partial diarization response")
                    else: offset = 0
                    with target.open("ab" if append else "wb") as output:
                        for chunk in response.iter_bytes(256 * 1024):
                            if cancel.is_set(): raise RuntimeError("Acquisition cancelled")
                            if offset + len(chunk) > row.size:
                                raise ValueError("Diarization asset exceeded the immutable planned byte size")
                            output.write(chunk); offset += len(chunk)
                            progress("downloading", base + offset, total)
                    break
        if offset != row.size:
            raise ValueError("Diarization asset size differs from the immutable plan")
        with target.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row.sha256:
            target.unlink()
            raise ValueError("Diarization component SHA-256 mismatch")
        return False

    def acquire(self, plan, staging, progress, cancel):
        # Plans can be deserialized by agents: never trust altered URLs or files.
        fresh = self.plan(request_from_plan(plan))
        if fresh.fingerprint() != plan.fingerprint():
            raise ValueError("Diarization plan differs from the trusted profile")
        p = self._profile(plan.candidate_id, plan.requested_revision)
        self._ordinary(staging); root = staging / "bundle"
        self._ordinary(root); root.mkdir(parents=True, exist_ok=True)
        progress("downloading", 0, plan.expected_bytes)
        base = 0; reused = True
        logger = logging.getLogger("httpx"); previous = logger.level; logger.setLevel(logging.WARNING)
        try:
            for row in fresh.files:
                reused = self._download_file(row, root / row.path, progress, cancel, base, fresh.expected_bytes) and reused
                base += row.size
        except httpx.HTTPError:
            raise ValueError("Pinned diarization download failed; check connectivity or retry later") from None
        finally: logger.setLevel(previous)
        if cancel.is_set(): raise RuntimeError("Acquisition cancelled")
        manifest = {"schema_version": 1, "profile_id": p["id"], "components": p["components"],
                    "sample_rate": p["sample_rate"], "max_speakers_per_frame": p["max_speakers_per_frame"],
                    "provenance": p["provenance"]}
        (root / "diarization.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        progress("prepared", fresh.expected_bytes, fresh.expected_bytes)
        return {"path": str(root.resolve()), "license": fresh.license, "license_evidence": _license_evidence(root),
                "integrity": [{"path": row.path, "bytes": row.size, "sha256": row.sha256} for row in fresh.files],
                "reused": reused}


class ProviderRegistry:
    def __init__(self, providers=None):
        values = providers or [HuggingFaceProvider(), GitHubSherpaProvider(), GitHubSherpaDiarizationProvider(), AceStepProvider()]
        self.providers = {provider.name: provider for provider in values}
        if len(self.providers) != len(values) or any(not name for name in self.providers):
            raise ValueError("Provider IDs must be unique and nonempty")

    def names(self):
        return sorted(self.providers)

    def descriptors(self):
        return [self.providers[name].descriptor() for name in self.names()]

    def get(self, name):
        try:
            return self.providers[name]
        except KeyError as exc:
            raise ValueError("Unknown model provider") from exc


def request_from_plan(plan):
    plan=AcquisitionPlan.model_validate(plan);spec=plan.model_spec
    options=spec.get('load_options') if isinstance(spec.get('load_options'),dict) else {}
    native=spec.get('native') if isinstance(spec.get('native'),dict) else {}
    return AcquisitionRequest(
        provider=plan.provider,identifier=plan.candidate_id,revision=plan.requested_revision,
        registry_name=plan.registry_name,replace=plan.replace,task=plan.task,backend=plan.backend,
        device=plan.device,dtype=spec.get('dtype','auto'),graph=spec.get('graph'),
        variant=options.get('variant'),model_file=native.get('model_file'),mmproj=native.get('mmproj'))


def import_model_link(value):
    parts = urlsplit(value.strip())
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("Model links must be public HTTPS URLs without embedded credentials")
    clean = urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, "", ""))
    host = parts.hostname.lower()
    segments = [unquote(item) for item in parts.path.split("/") if item]
    if host == "huggingface.co" and len(segments) >= 2:
        identifier = "/".join(segments[:2])
        revision = None
        if len(segments) >= 4 and segments[2] in {"tree", "blob", "resolve"}:
            revision = segments[3]
        ace={"ACE-Step/acestep-v15-xl-sft":"xl-sft","ACE-Step/acestep-v15-xl-turbo":"xl-turbo",
             "ACE-Step/acestep-v15-xl-base":"xl-base"}.get(identifier)
        if ace:
            return ImportedModelLink(provider="ace-step",identifier=ace,revision=revision,
                                     canonical_url=clean,compatibility="provider-recognized")
        return ImportedModelLink(provider="huggingface", identifier=identifier, revision=revision,
                                 canonical_url=clean, compatibility="provider-recognized")
    if (host == "github.com" and len(segments) == 6
            and segments[:4] == ["k2-fsa", "sherpa-onnx", "releases", "download"]):
        revision, filename = segments[4:]
        if filename.endswith(".tar.bz2"):
            return ImportedModelLink(provider="github-sherpa", identifier=filename[:-8],
                                     revision=revision, canonical_url=clean,
                                     compatibility="provider-recognized")
    return ImportedModelLink(provider=None, identifier=None, revision=None,
                             canonical_url=clean, compatibility="metadata-only")
