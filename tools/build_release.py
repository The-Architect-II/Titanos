"""Build a portable wheel and a static, checksum-addressed release catalog.

Beside the wheel the release carries what a pinned-interpreter install reads (tools/build_python_lock.py): each
source wheel tools/python-lock.json names, built from its pinned commit, and one hash lock per platform. release.json's
python_lock section names them with the pinned CPython build for each platform and the indexes the locks come from.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_python_lock import build_source_wheels, load_pins, lock_path  # noqa: E402


def described(path):
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "release"
    out.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(out), str(root)], check=True)
    wheels = sorted(out.glob("titanos_engine-*.whl"), key=lambda p: p.stat().st_mtime)
    wheel = wheels[-1]
    version = wheel.name.split("-")[1]
    pins = load_pins()
    sources = build_source_wheels(out, pins)
    locks = {}
    for platform in pins["platforms"]:
        locks[platform] = out / lock_path(platform).name
        shutil.copyfile(lock_path(platform), locks[platform])
    catalog = {
        "schema_version": 1, "version": version,
        "wheel": {"file": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()},
        "profiles": json.loads((root / "tools/profiles.json").read_text(encoding="utf-8")),
        "python_lock": {
            "python": pins["python"], "indexes": pins["indexes"],
            "wheels": [described(wheel)] + [described(path) for path in sources],
            "platforms": {platform: {"interpreter": pins["platforms"][platform]["interpreter"], "lock": described(locks[platform])}
                          for platform in pins["platforms"]},
        },
    }
    (out / "release.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(root / "src/titanos/hardware.py", out / "install.py")
    shutil.copyfile(root / "tools/native-profiles.json", out / "native-profiles.json")
    files = [wheel, *sources, *locks.values(), out / "install.py", out / "release.json", out / "native-profiles.json"]
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"Release ready at {out}")


if __name__ == "__main__":
    main()
