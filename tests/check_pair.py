"""Exercise unmodified PAIR engine-manager and proxy against real Titanos.

Requires locally built PAIR executables; it changes only a temporary config tree.
Linux is used for the isolated XDG configuration in this check.
"""

import argparse
import json
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

from titanos.cli import pair_manifest
from titanos.config import Config, ModelSpec, write_config


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RPC:
    def __init__(self, args, env, log):
        self.process = subprocess.Popen(args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        self.messages = queue.Queue()
        self.ident = 0
        def read():
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        threading.Thread(target=read, daemon=True).start()

    def call(self, method, params=None):
        self.ident += 1
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.ident, "method": method, "params": params or {}}) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            item = self.messages.get(timeout=max(0.1, deadline - time.monotonic()))
            if item.get("id") == self.ident:
                assert "error" not in item, item
                return item["result"]
        raise TimeoutError(method)

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine-manager", required=True)
    p.add_argument("--proxy", required=True)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if not sys.platform.startswith("linux"):
        p.error("This temporary XDG isolation check currently targets Linux")
    report = {"passed": []}
    with tempfile.TemporaryDirectory(prefix="titanos-pair-") as folder:
        root = Path(folder)
        cfg = root / "models.json"
        write_config(cfg, Config(models={"chat": ModelSpec(path=str(args.model_dir.resolve()), task="chat", device="cpu")}))
        executable = str(Path(sys.executable).parent / "titanos")
        manifests = root / "config/Nvidia Corporation/Personal AI Router/engines"
        manifests.mkdir(parents=True)
        (manifests / "titanos.json").write_text(json.dumps(pair_manifest(cfg, executable)))
        env = dict(os.environ, XDG_CONFIG_HOME=str(root / "config"), TITANOS_HOME=str(root / "data"),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4")
        env.pop("TITANOS_API_KEY", None)
        with (root / "upstream.log").open("w+") as log:
            manager = RPC([args.engine_manager, "--loaded-poll-interval", "0"], env, log)
            proxy = None
            try:
                result = manager.call("engine:describe", {"engine": "titanos"})
                assert result, result
                port = free_port()
                result = manager.call("engine:start", {"engine": "titanos", "port": port})
                print("manager start", result, flush=True)
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False) as c:
                    assert c.get("/healthz").status_code == 200
                    result = manager.call("engine:action", {"engine": "titanos", "action": "load_model", "params": {"model": "chat"}})
                    assert c.get("/readyz").status_code == 200
                    inventory = manager.call("engine:models")
                    assert "chat" in inventory["models"], inventory
                    report["passed"].append("upstream manifest validation, owned server launch, load action and model inventory")
                proxy_port = free_port()
                proxy = RPC([args.proxy, "--port", str(proxy_port), "--ignore-persisted-port", "--cluster-dir", str(root / "cluster")], env, log)
                proxy.call("node/add-manual", {"id": "titanos-local", "host": "127.0.0.1", "port": port, "addresses": ["127.0.0.1"], "models": ["chat"]})
                proxy.call("node/select", {"id": "titanos-local"})
                with httpx.Client(base_url=f"http://127.0.0.1:{proxy_port}", timeout=120, trust_env=False) as c:
                    r = c.post("/v1/chat/completions", json={"model": "chat", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 8, "temperature": 0})
                    assert r.status_code == 200, r.text
                    assert r.json()["usage"]["completion_tokens"] > 0
                    report["reply"] = r.json()["choices"][0]["message"]["content"]
                report["passed"].append("unmodified PAIR manual-node OpenAI proxy forwarded real text inference")
                manager.call("engine:action", {"engine": "titanos", "action": "unload_model", "params": {"model": "chat"}})
                manager.call("engine:stop", {"engine": "titanos"})
                report["passed"].append("upstream unload action and owned-process stop")
                report["success"] = True
            except Exception:
                log.flush(); log.seek(0)
                print(log.read()[-6000:], file=sys.stderr)
                raise
            finally:
                if proxy:
                    proxy.close()
                manager.close()
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
