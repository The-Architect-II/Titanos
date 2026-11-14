"""Hardware diagnostics and a stdlib-only, isolated runtime installer.

The release builder copies this file as install.py; no Titanos install is needed
to bootstrap. Runtime selection never downloads or mutates a serving environment.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


def discover():
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    info = {"os": platform.system().lower(), "arch": arch, "python": platform.python_version(),
            "libc": list(platform.libc_ver()), "nvidia": [], "display_vendors": []}
    command = shutil.which("nvidia-smi")
    if command:
        try:
            r = subprocess.run([command, "--query-gpu=index,name,compute_cap,driver_version,uuid", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True, timeout=10)
            for row in csv.reader(io.StringIO(r.stdout)):
                index, name, capability, driver, identity = [v.strip() for v in row]
                info["nvidia"].append({"index": int(index), "name": name, "capability": float(capability), "driver": driver, "uuid": identity})
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            info["nvidia_probe_error"] = str(exc)
    for path in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            vendor = {"0x1002": "amd", "0x8086": "intel", "0x10de": "nvidia"}.get(path.read_text().strip())
            if vendor and vendor not in info["display_vendors"]:
                info["display_vendors"].append(vendor)
        except OSError:
            pass
    if info["os"] == "windows":
        try:
            powershell = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
            r = subprocess.run([str(powershell), "-NoProfile", "-Command",
                                "@(Get-CimInstance Win32_VideoController | Select-Object PNPDeviceID) | ConvertTo-Json -Compress"],
                               capture_output=True, text=True, check=True, timeout=10)
            rows = json.loads(r.stdout or "[]")
            for row in ([rows] if isinstance(rows, dict) else rows):
                ident = row.get("PNPDeviceID", "") or ""
                for code, vendor in (("VEN_1002", "amd"), ("VEN_8086", "intel"), ("VEN_10DE", "nvidia")):
                    if code in ident and vendor not in info["display_vendors"]:
                        info["display_vendors"].append(vendor)
        except (OSError, subprocess.SubprocessError, ValueError):
            info["display_probe_error"] = "Could not enumerate Windows video controllers"
    return info


def select_device(requested="auto"):
    if requested == 'cpu':
        return 'cpu'
    import torch
    def checked(address):
        if address == 'cpu':return address
        if address == 'mps':
            api=getattr(getattr(torch,'backends',None),'mps',None)
            if api is None or not api.is_available():
                raise ValueError('Requested MPS device is unavailable; CPU was not substituted')
            return address
        match=re.fullmatch(r'(cuda|xpu)(?::(\d+))?',address)
        if not match:raise ValueError('Supported PyTorch device names: cpu, cuda[:index], xpu[:index], mps or a reported vendor UUID')
        family,index=match[1],int(match[2] or 0)
        api=getattr(torch,family,None)
        masks=('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL') if family=='cuda' and torch.version.hip else ('CUDA_VISIBLE_DEVICES',) if family=='cuda' else ('ZE_AFFINITY_MASK',)
        if any(key in os.environ and os.environ[key].strip() in {'','-1'} for key in masks):
            raise ValueError('Requested accelerator is masked by inherited visibility; CPU was not substituted')
        if api is None or not api.is_available() or index>=api.device_count():
            raise ValueError('Requested '+address+' is unavailable in this runtime visibility; CPU was not substituted')
        return family+':'+str(index)
    if requested.startswith(('HIP-','XPU-')):
        from .devices import resolve_identity
        return checked(resolve_identity(requested,'pytorch'))
    if requested.startswith('GPU-'):
        from .devices import runtime_device
        return checked(runtime_device(requested, torch))
    if requested != "auto":
        return checked(requested)
    from .devices import installed_runtime_preference,preferred_runtime_device,runtime_records
    record=installed_runtime_preference()
    if record is not None:
        if record.get("backend", "pytorch") == "pytorch":
            if record.get('gpu_uuid'):
                from .devices import runtime_device
                return checked(runtime_device(record['gpu_uuid'], torch))
            if str(record.get("device", "")).startswith("cuda") and not torch.version.hip:
                raise ValueError("Legacy CUDA installation has no pinned UUID; select an explicit GPU UUID or reinstall the runtime")
            if record.get('device') not in (None,'auto'):
                return checked(record['device'])
    choice=preferred_runtime_device(runtime_records('pytorch'),backend='pytorch',installed=record)
    if choice is None:raise ValueError('No compatible visible PyTorch device was observed')
    return checked(choice['device'])


def doctor(device="auto", check=False, backend="pytorch", provider_options=None):
    if backend=='engine':
        from . import __version__
        return {'engine_version':__version__,'hardware':discover(),'check':{'passed':True,'kind':'engine import','scope':'No numerical runtime or model execution implied'}}
    if backend != "pytorch":
        from .compiled import runtime_check
        return {**discover(), **runtime_check(backend, device, check, provider_options)}
    if provider_options:
        raise ValueError('Provider options apply to compiled runtimes, not PyTorch')
    from importlib.metadata import PackageNotFoundError, version
    info = discover()
    info["packages"] = {}
    for package in ("titanos-engine", "torch", "transformers", "diffusers"):
        try:
            info["packages"][package] = version(package)
        except PackageNotFoundError:
            info["packages"][package] = None
    try:
        import torch
        selected = select_device(device)
        info.update(device=selected, cuda_runtime=torch.version.cuda, rocm_runtime=torch.version.hip,
                    cuda_architectures=torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
                    mps_available=torch.backends.mps.is_available(),
                    xpu_available=hasattr(torch, "xpu") and torch.xpu.is_available())
        if selected.startswith('cuda') and not torch.version.hip:
            info['selected_gpu_uuid'] = str(torch.cuda.get_device_properties(selected).uuid)
        if check:
            torch.manual_seed(0)
            network = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()
            inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            with torch.inference_mode():
                expected = network(inputs)
                actual = network.to(selected)(inputs.to(selected)).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            info["check"] = {"passed": True, "kind": "two-layer neural network", "device": selected,
                             "scope": "runtime operation; model and modality acceptance is separate"}
    except Exception as exc:
        info["check"] = {"passed": False, "error": str(exc)}
    return info


def choose_profile(info, profiles, requested="auto", gpu=None):
    platform_key = info["os"] + "-" + info["arch"]
    device = None
    if gpu is not None:
        devices = [d for d in info["nvidia"] if d["index"] == gpu]
        if not devices:
            raise ValueError(f"NVIDIA device {gpu} was not detected")
    else:
        devices = info["nvidia"]
    if requested == "auto" and info.get("nvidia_probe_error"):
        raise ValueError("NVIDIA detection failed; inspect drivers or explicitly choose --profile cpu")
    capabilities = []
    if devices and (requested == "auto" or (isinstance(requested, str) and requested.startswith('cu'))):
        for device_info in devices:
            capability = device_info.get('capability')
            if type(capability) not in (int, float) or not math.isfinite(capability) or capability <= 0:
                raise ValueError("NVIDIA compute capability is unknown or invalid; inspect detection before selecting a CUDA profile")
            capabilities.append(capability)

    def covers_cuda(profile):
        lower, upper = profile.get('capability_min'), profile.get('capability_max')
        return (type(lower) in (int, float) and type(upper) in (int, float)
                and math.isfinite(lower) and math.isfinite(upper) and 0 <= lower < upper
                and all(lower <= capability < upper for capability in capabilities))

    if requested == "auto":
        if devices:
            # Versioned catalog bounds, not GPU-generation names, own compatibility.
            compatible = [name for name, profile in profiles.items()
                          if re.fullmatch(r'cu\d+', name) and platform_key in profile.get('platforms', [])
                          and covers_cuda(profile)]
            if not compatible:
                selected = ', '.join(f"GPU {d['index']} CC {c:g}" for d, c in zip(devices, capabilities))
                raise ValueError(f"No CUDA profile in this catalog covers {selected} on {platform_key}; "
                                 "select --gpu INDEX for separate environments or provide a compatible versioned catalog. "
                                 "CPU requires explicit --profile cpu; no fallback was selected")
            requested = min(compatible, key=lambda name: int(name[2:]))
        elif info["os"] == "darwin" and info["arch"] == "aarch64":
            requested = "mps"
        elif "nvidia" in info.get("display_vendors", []):
            # An NVIDIA adapter whose nvidia-smi is absent must not land on DirectML or CPU silently.
            raise ValueError("An NVIDIA display adapter is present but nvidia-smi reported no GPU; put the driver's nvidia-smi on PATH "
                             "and rerun, or select --profile explicitly; see docs/GETTING_STARTED.md")
        elif info['os'] == 'windows' and info.get('display_vendors') and info['arch'] == 'x86_64':
            requested = 'dml'
        elif info.get("display_vendors"):
            # Only profiles published for this platform; CUDA profiles need nvidia-smi rows and are handled above.
            choices = ", ".join(sorted(name for name, profile in profiles.items()
                                       if platform_key in profile.get("platforms", []) and not re.fullmatch(r"cu\d+", name)))
            raise ValueError(f"Select an explicit runtime profile for {platform_key}: {choices or 'none published'}; "
                             "see docs/GETTING_STARTED.md")
        else:
            requested = "cpu"
    if requested not in profiles:
        raise ValueError(f"Unknown runtime profile {requested!r}")
    profile = profiles[requested]
    if gpu is not None and not requested.startswith('cu'):
        raise ValueError('--gpu selects NVIDIA CUDA indices; use DirectML device_id in model provider options')
    if platform_key not in profile["platforms"]:
        raise ValueError(f"Profile {requested} has no published build for {platform_key}; see docs/HARDWARE.md")
    if requested.startswith("cu"):
        if not devices:
            raise ValueError("No NVIDIA GPU detected; repair the driver or select --profile cpu")
        if not covers_cuda(profile):
            raise ValueError("Selected CUDA profile does not cover the selected GPUs; choose --gpu INDEX or a different profile")
        device = "cuda:" + str(devices[0]["index"])
    return requested, profile, device or profile["device"]


def validate_http_url(location):
    parsed = urllib.parse.urlparse(location)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
        raise ValueError("Remote releases require HTTPS; HTTP is allowed only on loopback")
    return parsed


class ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def read_location(location):
    parsed = urllib.parse.urlparse(location)
    if parsed.scheme in {"http", "https"}:
        validate_http_url(location)
        handlers = [ReleaseRedirect()]
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
            handlers.append(urllib.request.ProxyHandler({}))
        with urllib.request.build_opener(*handlers).open(location, timeout=120) as response:
            return response.read(64 * 1024 * 1024 + 1)
    if parsed.scheme and not (len(parsed.scheme) == 1 and os.name == "nt"):
        raise ValueError("Use a local path or an HTTPS release catalog")
    return Path(location).read_bytes()


def install_main(argv=None):
    parser = argparse.ArgumentParser(description="Install an isolated Titanos runtime from a versioned release catalog")
    parser.add_argument("--catalog", required=True, help="Local release.json path or HTTPS URL (loopback HTTP allowed)")
    parser.add_argument("--catalog-sha256", help="Optional out-of-band catalog checksum")
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--plan", action="store_true", help="Print selection without installing anything")
    args = parser.parse_args(argv)
    try:
        if not (3, 11) <= sys.version_info[:2] < (3, 14):
            raise ValueError("This release requires Python 3.11–3.13")
        raw = read_location(args.catalog)
        if len(raw) > 1024 * 1024:
            raise ValueError("Release catalog exceeds 1 MiB")
        if args.catalog_sha256 and hashlib.sha256(raw).hexdigest() != args.catalog_sha256:
            raise ValueError("Release catalog checksum mismatch")
        catalog = json.loads(raw)
        if catalog["schema_version"] != 1:
            raise ValueError("Unsupported release catalog schema")
        if not re.fullmatch(r"\d+\.\d+\.\d+", catalog["version"]):
            raise ValueError("Invalid release version")
        detected = discover()
        name, profile, device = choose_profile(detected, catalog["profiles"], args.profile, args.gpu)
        target = (args.target or Path("~/.titanos/runtimes").expanduser() / (catalog["version"] + "-" + name)).resolve()
        plan = {"version": catalog["version"], "profile": name, "device": device, "target": str(target),
                "hardware": detected, "validation": profile["validation"], "backend": profile.get("backend", "pytorch")}
        if name.startswith('cu'):
            plan['gpu_uuid'] = next(d['uuid'] for d in detected['nvidia'] if d['index'] == int(device.split(':')[1]))
        print(json.dumps(plan, indent=2), flush=True)
        if args.plan:
            return
        if target.exists():
            raise ValueError("Target already exists. Choose a new --target; existing environments are never overwritten")
        wheel = catalog["wheel"]
        if not re.fullmatch(r"titanos_engine-[A-Za-z0-9_.+-]+\.whl", wheel["file"]):
            raise ValueError("Invalid release wheel filename")
        if urllib.parse.urlparse(args.catalog).scheme in {"http", "https"}:
            location = urllib.parse.urljoin(args.catalog, wheel["file"])
        else:
            location = str(Path(args.catalog).parent / wheel["file"])
        payload = read_location(location)
        if len(payload) > 64 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != wheel["sha256"]:
            raise ValueError("Wheel checksum mismatch or file exceeds 64 MiB")
        with tempfile.TemporaryDirectory(prefix="titanos-install-") as tmp:
            wheel_path = Path(tmp) / wheel["file"]
            wheel_path.write_bytes(payload)
            import venv
            venv.EnvBuilder(with_pip=True).create(target)
            python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            def pip(*params):
                subprocess.run([str(python), "-m", "pip", "--disable-pip-version-check", *params], check=True)
            pip("install", "--only-binary=:all:", "--index-url", profile["index_url"], *profile["packages"])
            pins = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
            constraints = Path(tmp) / "runtime.txt"
            constraints.write_text(pins, encoding="utf-8")
            extra=profile.get('extra','models')
            pip("install", "-c", str(constraints), str(wheel_path) + ('['+extra+']' if extra else ''))
            pip("check")
            # The receipt fixes auto device selection to the installed profile.
            receipt = dict(plan, catalog_sha256=hashlib.sha256(raw).hexdigest(), wheel_sha256=wheel["sha256"])
            if receipt.get('gpu_uuid'):
                device = receipt['gpu_uuid']
            (target / "titanos-runtime.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            check = subprocess.run([str(python), "-m", "titanos", "doctor", "--check", "--device", device, "--backend", profile.get("backend", "pytorch"),
                                    "--output", str(target / "hardware-check.json")], check=False)
            if check.returncode:
                raise ValueError(f"Runtime inference check failed; details in {target / 'hardware-check.json'}. No CPU fallback was substituted")
            (target / "requirements.lock.txt").write_text(subprocess.check_output([str(python), "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"], text=True), encoding="utf-8")
            executable = target / ("Scripts/titanos.exe" if os.name == "nt" else "bin/titanos")
            print(f"Installed and checked. Run: {executable} init\nThen register a model and run: {executable} serve")
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Titanos install: {exc}\n")


if __name__ == "__main__":
    install_main()
