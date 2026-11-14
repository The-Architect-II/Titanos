"""Install from an actual loopback HTTP release feed into a fresh environment."""

import argparse
import functools
import http.server
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, type=Path)
    p.add_argument("--models", type=Path)
    p.add_argument("--profile", default="cpu")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    release = root / "release"
    target = args.target.resolve()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    requests = []
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *values):
            requests.append(self.path)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(release)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    import sys
    bootstrap = [sys.executable, str(release / "install.py")]
    url = f"http://127.0.0.1:{server.server_port}/release.json"
    report = {"passed": []}
    try:
        subprocess.run(bootstrap + ["--catalog", url, "--profile", args.profile, "--target", str(target)], env=env, check=True)
        assert "/release.json" in requests and any(s.endswith(".whl") for s in requests)
        report["passed"].append("catalog and wheel fetched through a real loopback HTTP server")
        python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        location = subprocess.check_output([str(python), "-I", "-c", "import titanos; print(titanos.__file__)"], env=env, text=True).strip()
        assert Path(location).is_relative_to(target), location
        report["passed"].append("wheel imports exclusively from the fresh runtime, with isolated Python")
        versions = (target / "requirements.lock.txt").read_text()
        release_version = json.loads((release / "release.json").read_text())["version"]
        assert f"titanos-engine=={release_version}" in versions and "file://" not in versions
        report["passed"].append("version snapshot retains no expired temporary wheel path")
        with tempfile.TemporaryDirectory(prefix="titanos-tamper-") as tmp:
            directory = Path(tmp)
            catalog = json.loads((release / "release.json").read_text())
            catalog["wheel"]["sha256"] = "0" * 64
            (directory / "release.json").write_text(json.dumps(catalog))
            (directory / catalog["wheel"]["file"]).write_bytes((release / catalog["wheel"]["file"]).read_bytes())
            rejected = subprocess.run(bootstrap + ["--catalog", str(directory / "release.json"), "--profile", args.profile, "--target", str(directory / "untrusted")], env=env, capture_output=True, text=True)
            assert rejected.returncode != 0 and "checksum mismatch" in rejected.stderr
            assert not (directory / "untrusted").exists()
            report["passed"].append("tampered wheel checksum rejected before environment creation")
        if args.models:
            subprocess.run([str(python), str(root / "tests/check.py"), "--models", str(args.models.resolve()),
                            "--report", str(target / "model-acceptance.json")], env=env, check=True)
            report["passed"].append("all real model acceptance checks passed from the installed wheel")
        report["success"] = True
        report["hardware"] = json.loads((target / "hardware-check.json").read_text())
        report["wheel_sha256"] = json.loads((release / "release.json").read_text())["wheel"]["sha256"]
    finally:
        server.shutdown(); server.server_close()
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
