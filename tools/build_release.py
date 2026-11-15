"""Build a portable wheel and a static, checksum-addressed release catalog."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "release"
    out.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(out), str(root)], check=True)
    wheels = sorted(out.glob("titanos_engine-*.whl"), key=lambda p: p.stat().st_mtime)
    wheel = wheels[-1]
    version = wheel.name.split("-")[1]
    catalog = {
        "schema_version": 1, "version": version,
        "wheel": {"file": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()},
        "profiles": json.loads((root / "tools/profiles.json").read_text(encoding="utf-8")),
    }
    (out / "release.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(root / "src/titanos/hardware.py", out / "install.py")
    shutil.copyfile(root / "tools/native-profiles.json", out / "native-profiles.json")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in (wheel, out / "install.py", out / "release.json", out / "native-profiles.json")]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"Release ready at {out}")


if __name__ == "__main__":
    main()
