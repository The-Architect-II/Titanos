"""Construct a relocatable ACE runtime tree without running it or copying weights.

Use a self-contained interpreter prefix, a completed non-editable site-packages,
the trusted adapter file, and the pinned ACE source/notices. Publication creates
one previously absent directory atomically. No pip, shell or model command runs.
"""
from __future__ import annotations

import argparse
import ctypes
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import sys
import tempfile

FAMILY = "ace-step-1.5"
CODE_REVISION = "ca1e85fe9430179831e6bc6be790c332190a3866"
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".onnx", ".ckpt", ".pt"}
# pip writes <dist>.dist-info/direct_url.json (PEP 610) when a distribution was installed from a local wheel, archive or source
# tree; its url is then a file:/// URL of the installing machine (runtime r4 carried four with the lab's records drive). The
# record is provenance of one installation, not of the packaged runtime, so packaging leaves it out and verification refuses it.
PIP_PROVENANCE_NAME = "direct_url.json"
PIP_PROVENANCE_PATTERN = "*.dist-info/" + PIP_PROVENANCE_NAME
PIP_PROVENANCE_REASON = ("PEP 610 pip provenance of the packaging machine's installation (a file:/// URL of a local wheel, archive "
                         "or source tree); excluded from the runtime, hashes and versions stay in METADATA/RECORD")


def is_pip_provenance(name):
    """True for <anything>/<dist>.dist-info/direct_url.json (POSIX-relative runtime path)."""
    path = PurePosixPath(name)
    return path.name == PIP_PROVENANCE_NAME and path.parent.name.endswith(".dist-info")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(value):
    if not isinstance(value, str) or not value or value in {".", ".."} or "\\" in value or ":" in value:
        raise ValueError("Expected a normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in {".", ".."} for p in path.parts):
        raise ValueError("Path escapes or is not normalized: " + value)
    return Path(*path.parts)


def physical_path(value, *, directory=True, missing=False):
    path = Path(os.path.abspath(value))
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Indirection in input/output path: " + str(item))
    if not missing and (not path.is_dir() if directory else not path.is_file()):
        raise ValueError("Missing ordinary input: " + str(path))
    return path.resolve(strict=not missing)


def tree(root, *, skip=(), allow_internal_links=True, omit_caches=True, omit_provenance=None):
    """Inspect links before following them; materialized output contains no links.

    omit_provenance, when a list, receives the runtime-relative names of pip provenance records (is_pip_provenance) that are
    left out instead of yielded; None yields them like any other file (verification sees everything)."""
    root = physical_path(root)
    skipped = {relative(p).as_posix() for p in skip}

    def visit(folder, prefix, ancestors):
        resolved = folder.resolve(strict=True)
        if not resolved.is_relative_to(root) or resolved in ancestors:
            raise ValueError("Escaping/cyclic runtime directory: " + str(folder))
        for item in sorted(folder.iterdir(), key=lambda p: p.name):
            name = (prefix / item.name).as_posix()
            if name in skipped or (omit_caches and ("__pycache__" in PurePosixPath(name).parts or item.suffix == ".pyc")):
                continue
            info = item.lstat()
            link = stat.S_ISLNK(info.st_mode)
            if getattr(info, "st_file_attributes", 0) & 0x400 and not link:
                raise ValueError("Unsupported reparse point: " + str(item))
            if link and (not allow_internal_links or not item.resolve(strict=True).is_relative_to(root)):
                raise ValueError("Escaping/runtime output link: " + str(item))
            if item.is_dir():
                yield from visit(item, PurePosixPath(name), ancestors | {resolved})
            elif item.is_file():
                if item.suffix.lower() in WEIGHT_SUFFIXES:
                    raise ValueError("Model weights must remain external: " + name)
                if omit_provenance is not None and is_pip_provenance(name):
                    omit_provenance.append(name)
                    continue
                yield name, item
            else:
                raise ValueError("Unsupported runtime entry: " + str(item))

    yield from visit(root, PurePosixPath(), set())


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def copy_file(source, target, expected=None):
    before = {"bytes": source.stat().st_size, "sha256": digest(source)}
    if expected is not None and before != {k: expected[k] for k in ("bytes", "sha256")}:
        raise ValueError("Pinned source differs: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    os.chmod(target, (source.stat().st_mode & 0o777) | stat.S_IWUSR)
    if target.stat().st_size != before["bytes"] or digest(target) != before["sha256"] or digest(source) != before["sha256"]:
        raise ValueError("Input changed during copy: " + str(source))


def copy_tree(source, target, *, skip=(), pins=None, omit_provenance=None):
    entries = list(tree(source, skip=skip, omit_provenance=omit_provenance))
    names = [name for name, _ in entries]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Case-ambiguous runtime paths")
    if pins is not None and set(names) != set(pins):
        raise ValueError("Pinned source membership differs")
    for name, path in entries:
        copy_file(path, target / relative(name), pins[name] if pins is not None else None)


def validate_startup_paths(root, site):
    hooks = []
    for path in site.rglob("*.pth"):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "__editable__" in line or "__editable__" in path.name:
                raise ValueError("Editable installs are not portable: " + path.name)
            if line.startswith(("import ", "import\t")):
                hooks.append(path.relative_to(root).as_posix())
                continue  # Trusted dependency startup code is bound, never run by this tool.
            if Path(line).is_absolute() or PureWindowsPath(line).is_absolute():
                raise ValueError("Absolute .pth dependency: " + line)
            target = path.parent / relative(line.replace("\\", "/"))
            if not target.exists() or not target.resolve().is_relative_to(root):
                raise ValueError("Missing/escaping .pth dependency: " + line)
    for path in (root / "python").glob("*._pth"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line == "import site":
                continue
            if line.startswith(("import ", "import\t")):
                raise ValueError("Unsupported embedded-Python startup command")
            if Path(line).is_absolute() or PureWindowsPath(line).is_absolute() or ".." in PurePosixPath(line.replace("\\", "/")).parts:
                raise ValueError("Nonportable embedded-Python path: " + line)
            referenced = path.parent / line.replace("\\", "/")
            if not referenced.exists() or not referenced.resolve().is_relative_to(root / "python"):
                raise ValueError("Missing embedded-Python payload: " + line)
    return sorted(set(hooks))


def dependency_inventory(root, site):
    dependencies = []
    for metadata in sorted(site.glob("*.dist-info/METADATA")):
        value = Parser().parsestr(metadata.read_text(encoding="utf-8"))
        if not value.get("Name") or not value.get("Version"):
            raise ValueError("Dependency lacks name/version: " + str(metadata))
        notices = [p.relative_to(root).as_posix() for p in metadata.parent.rglob("*")
                   if p.is_file() and any(token in p.name.lower() for token in ("license", "licence", "copying", "notice"))]
        dependencies.append({"name": value["Name"], "version": value["Version"],
                             "declared_license": value.get("License-Expression") or value.get("License"),
                             "requires_dist": value.get_all("Requires-Dist", []),
                             "metadata": metadata.relative_to(root).as_posix(), "license_files": notices})
    if not dependencies:
        raise ValueError("Runtime requires an installed dependency inventory")
    return dependencies


def publish(stage, target):
    """Atomic no-replace directory publication, never a merge into an old runtime."""
    if os.name == "nt":
        os.rename(stage, target)  # Windows rejects any existing destination.
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise RuntimeError("Atomic no-replace rename unavailable; preserving target")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(stage), -100, os.fsencode(target), 1):
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number), str(target))
        return
    raise RuntimeError("Atomic directory publication is supported on Windows/Linux only")


def verify_runtime(directory):
    root = physical_path(directory)
    value = read_json(physical_path(root / "runtime.json", directory=False))
    if value.get("schema_version") != 1 or value.get("family") != FAMILY or value.get("code_revision") != CODE_REVISION:
        raise ValueError("Unknown music runtime identity")
    files = value.get("files")
    if not isinstance(files, dict) or not files or "runtime.json" in files:
        raise ValueError("Invalid runtime file inventory")
    actual = dict(tree(root, allow_internal_links=False, omit_caches=False))
    provenance = sorted(name for name in actual if is_pip_provenance(name))
    if provenance:
        raise ValueError("Runtime carries pip provenance records with local file URLs (" + PIP_PROVENANCE_PATTERN
                         + "); rebuild it with this tool, which excludes them: " + ", ".join(provenance))
    if set(actual) != set(files) | {"runtime.json"}:
        raise ValueError("Runtime membership differs")
    for name, pin in files.items():
        relative(name)
        if not isinstance(pin, dict) or set(pin) != {"bytes", "sha256"} or type(pin["bytes"]) is not int or pin["bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]):
            raise ValueError("Invalid runtime file pin: " + name)
        if actual[name].stat().st_size != pin["bytes"] or digest(actual[name]) != pin["sha256"]:
            raise ValueError("Runtime payload differs: " + name)
    for key in ("executable", "adapter"):
        name = value.get(key)
        relative(name)
        if name not in files:
            raise ValueError("Runtime entry is not hash-bound: " + key)
    return value


def package_runtime(*, interpreter, site_packages, adapter, source, source_pin, notices,
                    output, executable, site_relative):
    interpreter = physical_path(interpreter)
    site_packages = physical_path(site_packages)
    source = physical_path(source)
    notices = physical_path(notices)
    adapter = physical_path(adapter, directory=False)
    executable = relative(executable).as_posix()
    site_relative = relative(site_relative).as_posix()
    if (interpreter / "pyvenv.cfg").exists():
        raise ValueError("Use the self-contained interpreter prefix, not a venv")
    if not (interpreter / executable).is_file() or adapter.suffix != ".py":
        raise ValueError("Missing interpreter or trusted Python adapter")
    pin = read_json(physical_path(source_pin, directory=False))
    if pin.get("commit") != CODE_REVISION or pin.get("repository") != "ace-step/ACE-Step-1.5" or not pin.get("files"):
        raise ValueError("Expected the pinned official ACE source inventory")
    target = physical_path(output, missing=True)
    if os.path.lexists(target):
        raise FileExistsError("Preserve existing runtime: " + str(target))
    for entry in (interpreter, site_packages, source, notices):
        if target.is_relative_to(entry) or entry.is_relative_to(target):
            raise ValueError("Runtime output must be separate from every input")
    if adapter.is_relative_to(target):
        raise ValueError("Runtime output contains its own adapter input")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + target.name + "-", dir=target.parent))
    try:
        # Relocate the prefix, not pyvenv.cfg or absolute venv scripts/shebangs.
        interpreter_provenance, site_provenance = [], []
        copy_tree(interpreter, stage / "python", skip=(site_relative,), omit_provenance=interpreter_provenance)
        site = stage / "python" / site_relative
        copy_tree(site_packages, site, omit_provenance=site_provenance)
        # runtime-relative names of the pip provenance records left out, recorded in both inventories below
        excluded_provenance = (["python/" + name for name in interpreter_provenance]
                               + ["python/" + site_relative + "/" + name for name in site_provenance])
        # Embedded Windows distributions can disable site discovery via ._pth.
        # Add only a relative data path; never execute a setup command.
        embedded_path_files = []
        for path in (stage / "python").glob("*._pth"):
            text = path.read_text(encoding="utf-8")
            if site_relative not in {line.strip().replace("\\", "/") for line in text.splitlines()}:
                path.write_text(text.rstrip() + "\n" + site_relative + "\n", encoding="utf-8")
            embedded_path_files.append(path.relative_to(stage).as_posix())
        copy_file(adapter, stage / "adapter" / adapter.name)
        copy_tree(source, stage / "SOURCE" / FAMILY, pins=pin["files"])
        copy_tree(notices, stage / "licenses")
        hooks = validate_startup_paths(stage, site)
        dependencies = dependency_inventory(stage, site)
        license_files = [name for name, _ in tree(stage) if name.startswith("licenses/")
                         or PurePosixPath(name).name.lower().startswith(("license", "copying", "notice"))]
        if not license_files:
            raise ValueError("Music runtime must retain license texts/notices")
        exclusions = {"pip_provenance": {"pattern": PIP_PROVENANCE_PATTERN, "reason": PIP_PROVENANCE_REASON,
                                         "files": sorted(excluded_provenance)}}
        write_json(stage / "DEPENDENCIES.json", {"scope": "Installed direct-class runtime, not stock ACE web/training extras",
                                                 "exclusions": exclusions, "components": dependencies})
        write_json(stage / "SOURCES.json", {"repository": pin["repository"], "code_revision": CODE_REVISION,
                    "source_directory": "SOURCE/" + FAMILY, "source_files": pin["files"]})
        write_json(stage / "LICENSES.json", {"files": sorted(license_files), "scope": "Component terms retained; no blanket MIT/Apache-only native-bundle claim"})
        entries = dict(tree(stage, allow_internal_links=False))
        manifest = {"schema_version": 1, "family": FAMILY, "code_revision": CODE_REVISION,
                    "executable": "python/" + executable, "adapter": "adapter/" + adapter.name,
                    "runtime_contract_version": 1, "supported_devices": ["cpu", "cuda"],
                    "supported_variants": ["xl-sft"],
                    "layout": {"interpreter_prefix": "python", "site_packages": "python/" + site_relative,
                               "venv_configuration_included": False, "startup_import_files": hooks,
                               "embedded_relative_site_files": embedded_path_files},
                    "inventories": {"dependencies": "DEPENDENCIES.json", "sources": "SOURCES.json", "licenses": "LICENSES.json"},
                    "exclusions": exclusions,
                    "qualification": {"construction": "full_hash_manifest", "model_execution": False, "target_native": False,
                                      "limits": "No interpreter/model execution during packaging. Native dependencies and moved-prefix imports need target qualification."},
                    "files": {name: {"bytes": path.stat().st_size, "sha256": digest(path)} for name, path in sorted(entries.items())}}
        write_json(stage / "runtime.json", manifest)
        verify_runtime(stage)
        publish(stage, target)
        return target / "runtime.json"
    finally:
        if stage.exists():
            if stage.resolve() != stage or not stage.is_relative_to(target.parent):
                raise RuntimeError("Refusing cleanup outside fresh owned staging directory")
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    for name in ("interpreter", "site-packages", "adapter", "source", "source-pin", "notices"):
        parser.add_argument("--" + name, type=Path)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--output", type=Path)
    target.add_argument("--portable-root", type=Path, help="Unsealed Titanos staging root; emits runtimes/ace-step-1.5")
    parser.add_argument("--executable", help="Interpreter path relative to its self-contained prefix")
    parser.add_argument("--site-relative", help="Site-packages path relative to the interpreter prefix")
    args = parser.parse_args()
    if args.verify:
        value = verify_runtime(args.verify)
        print(json.dumps({"verified": True, "files": len(value["files"]), "native_execution": False}))
        return
    if args.portable_root:
        if (args.portable_root / "MANIFEST.json").exists():
            parser.error("Add the runtime during construction, before sealing the outer package manifest")
        args.output = args.portable_root / "runtimes" / FAMILY
    required = ("interpreter", "site_packages", "adapter", "source", "source_pin", "notices", "output", "executable", "site_relative")
    if any(getattr(args, name) is None for name in required):
        parser.error("Packaging needs all input paths, an output and relative interpreter/site layout")
    result = package_runtime(**{name: getattr(args, name) for name in required})
    print(json.dumps({"runtime_manifest": str(result), "native_execution": False}))


if __name__ == "__main__":
    main()
