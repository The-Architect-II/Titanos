"""Runnable acceptance check: base contracts, optionally real local checkpoints.

Run after installation: python tests/check.py [--models DIRECTORY] [--report FILE]
Pass/fail covers inference and interface contracts, not factual intelligence,
cosmetic prompt adherence, or human presentation acceptance.
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from titanos.config import Config, ModelSpec, read_config, write_config
from titanos.hardware import choose_profile
from titanos.server import create_app


def contracts(root, passed):
    path = root / "models.json"
    write_config(path, Config(models={"missing": ModelSpec(path="./absent", task="chat")}))
    cfg = read_config(path)
    assert Path(cfg.models["missing"].path).resolve() == (root / "absent").resolve()
    with TestClient(create_app(cfg, api_key="test-secret", output_dir=root / "outputs"), base_url="http://127.0.0.1") as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/models").status_code == 401
        c.headers["Authorization"] = "Bearer test-secret"
        assert c.get("/v1/models").json()["data"][0]["status"] == "unloaded"
        assert c.get("/readyz").status_code == 503
        assert c.post("/v1/chat/completions", json={"model": "unknown", "messages": [{"role": "user", "content": "x"}]}).status_code == 404
        assert c.post("/v1/embeddings", json={"model": "missing", "input": "x"}).status_code == 400
        bad = c.post("/v1/chat/completions", json={"model": "missing", "messages": [], "tools": []})
        assert bad.status_code == 422 and "error" in bad.json()
        assert c.post("/api/models/load", json={"model": "missing"}).status_code == 503
        assert c.get("/v1/models").json()["data"][0]["status"] == "error"
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/files/not-an-artifact").status_code == 404
        modes = c.post('/api/models/capabilities', json={'model': 'missing'}).json()['modes']
        assert modes['model_sharding']['qualified'] is None and modes['tensor_parallel']['qualified'] is None
        assert modes['replicas']['meaning'] != modes['tensor_parallel']['meaning']
    with TestClient(create_app(Config(), output_dir=root / "outputs"), base_url="http://evil.example") as c:
        assert c.get("/v1/models").status_code == 400
    passed.append("registry, auth, validation, load failure recovery, host and artifact boundaries")
    env = dict(os.environ)
    env.pop("TITANOS_API_KEY", None)
    denied = subprocess.run([sys.executable, "-m", "titanos", "--config", str(path), "serve", "--host", "0.0.0.0"],
                            env=env, capture_output=True, text=True, timeout=15)
    assert denied.returncode != 0 and "TITANOS_API_KEY" in denied.stderr
    passed.append("unauthenticated network bind rejected before listening")

    profiles = json.loads((Path(__file__).resolve().parents[1] / "tools/profiles.json").read_text())
    base = {"os": "linux", "arch": "x86_64", "nvidia": [], "display_vendors": []}
    assert choose_profile(base, profiles)[0] == "cpu"
    pascal = dict(base, nvidia=[{"index": 0, "capability": 6.1}])
    blackwell = dict(base, nvidia=[{"index": 0, "capability": 12.0}])
    assert choose_profile(pascal, profiles)[0] == "cu126"
    assert choose_profile(blackwell, profiles)[0] == "cu128"
    mixed = dict(base, nvidia=[{"index": 0, "capability": 6.1}, {"index": 1, "capability": 12.0}])
    try:
        choose_profile(mixed, profiles)
        raise AssertionError("Mixed architectures must not silently select an incompatible build")
    except ValueError:
        pass
    assert choose_profile(mixed, profiles, gpu=1)[2] == "cuda:1"
    assert choose_profile(dict(base, os='darwin'), profiles, requested='onnx-intel-mac')[0] == 'onnx-intel-mac'
    try:
        choose_profile(dict(base, os="darwin"), profiles)
        raise AssertionError("Intel macOS must not claim a current wheel exists")
    except ValueError:
        pass
    passed.append("runtime selection: CPU, Pascal, Blackwell, mixed GPUs, unsupported platform")


def live(root, models, passed, observations, device="cpu", executable=None, dtype="float32"):
    import socket
    from PIL import Image
    cfg = Config(models={t: ModelSpec(path=str(models / t), task=t,
                                      backend="diffusers" if t in {"image", "video"} else "transformers",
                                      device=device, dtype=dtype, allow_pickle=t == "image",
                                      load_options={"safety_checker": None} if t == "image" else {},
                                      defaults={"max_sequence_length": 16} if t == "video" else {})
                         for t in ["chat", "vision", "embeddings", "speech", "transcription", "image", "video"]})
    if device.startswith('cuda'):
        import torch
        if not torch.version.hip:
            value = str(torch.cuda.get_device_properties(device).uuid)
            for spec in cfg.models.values():
                spec.device = value if value.startswith('GPU-') else 'GPU-' + value
    path = root / "models.json"
    write_config(path, cfg)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, TITANOS_HOME=str(root), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("TITANOS_API_KEY", None)
    launch = [str(executable)] if executable else [sys.executable, "-m", "titanos"]
    if executable:
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        env.pop('VIRTUAL_ENV', None)
        if os.name == 'nt':
            system_root = Path(env.get('SystemRoot', 'C:/Windows'))
            env['PATH'] = os.pathsep.join(map(str, [system_root / 'System32', system_root, system_root / 'System32' / 'Wbem']))
        else:
            env['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin'
        observations['live_runtime'] = {'executable': str(executable), 'ambient_python_paths_removed': True}
    command = [*launch, "--config", str(path), "serve", "--port", str(port), "--threads", "4"]
    with (root / "server.log").open("w+") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180, trust_env=False) as c:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        if c.get("/healthz").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    assert process.poll() is None and time.monotonic() < deadline, "Server failed to start"
                    time.sleep(0.1)

                def post(url, **kw):
                    r = c.post(url, **kw)
                    assert r.status_code == 200, f"{url}: {r.status_code}: {r.text[:2000]}"
                    return r

                prompt = {"model": "chat", "messages": [{"role": "user", "content": "Say hello in one short sentence."}], "temperature": 0, "max_tokens": 24}
                normal = post("/v1/chat/completions", json=prompt).json()
                assert 0 < normal["usage"]["completion_tokens"] <= prompt['max_tokens']
                assert isinstance(normal['choices'][0]['message']['content'], str) and normal['choices'][0]['message']['content'].strip()
                chunks = []
                with c.stream("POST", "/v1/chat/completions", json=dict(prompt, stream=True, stream_options={"include_usage": True})) as r:
                    assert r.status_code == 200
                    lines = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
                assert lines[-1] == "[DONE]"
                for line in lines[:-1]:
                    event = json.loads(line)
                    if event["choices"]:
                        chunks.append(event["choices"][0]["delta"].get("content", ""))
                assert "".join(chunks) == normal["choices"][0]["message"]["content"]
                assert json.loads(lines[-2])["usage"] == normal["usage"]
                observations["chat"] = normal["choices"][0]["message"]["content"]
                completion = post("/v1/completions", json={"model": "chat", "prompt": "Hello", "max_tokens": 4, "temperature": 0}).json()
                assert completion["object"] == "text_completion"
                passed.append("live text generation, true SSE, usage, completion endpoint")
                long_request = dict(prompt, messages=[{"role": "user", "content": "Write a long story about a journey through a forest."}], stream=True, max_tokens=1024)
                with c.stream("POST", "/v1/chat/completions", json=long_request) as r:
                    for line in r.iter_lines():
                        if line.startswith("data: ") and "[DONE]" not in line:
                            event = json.loads(line[6:])
                            if event.get("choices") and event["choices"][0]["delta"].get("content"):
                                break
                # A disconnected stream must release the inference lock.
                with httpx.Client(base_url=str(c.base_url), timeout=15, trust_env=False) as control:
                    assert control.post("/api/models/unload", json={"model": "chat"}).status_code == 200
                passed.append("stream disconnect releases the model for subsequent lifecycle operations")
                print("PASS text and streaming", flush=True)

                r = post("/v1/embeddings", json={"model": "embeddings", "input": ["a cat", "a dog"]}).json()
                assert len(r["data"]) == 2 and len(r["data"][0]["embedding"]) == 384
                assert abs(sum(v*v for v in r["data"][0]["embedding"]) - 1) < 1e-4
                image = Image.new("RGB", (64, 64), "red")
                buf = io.BytesIO(); image.save(buf, format="PNG")
                inline = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
                r = post("/v1/chat/completions", json={"model": "vision", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": inline}}, {"type": "text", "text": "What color is this image?"}]}], "max_tokens": 16, "temperature": 0}).json()
                assert r["usage"]["completion_tokens"] > 0
                observations["vision"] = r["choices"][0]["message"]["content"]
                passed.append("real embeddings and image-conditioned vision generation")
                print("PASS embeddings and vision", flush=True)

                wav = post("/v1/audio/speech", json={"model": "speech", "input": "Hello world. The quick brown fox jumps over the lazy dog.", "voice": "default", "response_format": "wav"}).content
                assert wav[:4] == b"RIFF" and len(wav) > 1000
                result = post("/v1/audio/transcriptions", data={"model": "transcription"}, files={"file": ("test.wav", wav, "audio/wav")}).json()
                assert result["text"].strip()
                observations["transcription"] = result["text"]
                assert c.post("/v1/audio/transcriptions", data={"model": "transcription"}, files={"file": ("bad.wav", b"invalid", "audio/wav")}).status_code == 400
                passed.append("real speech synthesis, transcription round trip, malformed audio rejection")
                print("PASS speech and transcription", flush=True)

                result = post("/v1/images/generations", json={"model": "image", "prompt": "A red cube", "size": "64x64", "steps": 2, "seed": 1}).json()
                png = base64.b64decode(result["data"][0]["b64_json"])
                assert Image.open(io.BytesIO(png)).size == (64, 64)
                result = post("/v1/images/edits", data={"model": "image", "prompt": "a blue cube", "size": "64x64", "steps": 4, "strength": 0.5}, files={"image": ("image.png", png, "image/png")}).json()
                assert result["data"][0]["b64_json"]
                result = post("/v1/videos/generations", json={"model": "video", "prompt": "A red cube", "size": "64x64", "steps": 2, "frames": 5, "seed": 1}).json()
                mp4 = c.get(result["url"])
                assert mp4.status_code == 200 and b"ftyp" in mp4.content[:32]
                assert result["frames"] == 5
                import imageio
                clip = root / "clip.mp4"
                clip.write_bytes(mp4.content)
                with imageio.get_reader(clip) as reader:
                    assert reader.count_frames() == 5
                observations["video_frames"] = result["frames"]
                passed.append("real tiny diffusion image, image edit, and video-to-MP4 export")
                assert c.get("/api/models/loaded").json()["data"][0]["id"] == "video"
                assert len(c.get("/api/models/loaded").json()["data"]) == 1
                post("/api/models/unload", json={"model": "video"})
                assert not c.get("/api/models/loaded").json()["data"]
                post("/api/models/load", json={"model": "chat"})
                assert c.get("/readyz").status_code == 200
                passed.append("cross-modality eviction, unload, reload and readiness")
                print("PASS image, edit, video and lifecycle", flush=True)
        except Exception:
            log.flush(); log.seek(0)
            print(log.read()[-6000:], file=sys.stderr)
            raise
        finally:
            if os.name == 'nt':
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            else:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            log.close()
            # Windows can release inherited log handles just after the child exits.
            # Retry only this task-owned file; a persistent lock remains a test failure.
            for attempt in range(60):
                try:
                    (root / 'server.log').unlink()
                    break
                except PermissionError:
                    if attempt == 59:raise
                    time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype",default="float32",choices=["auto","float32","float16","bfloat16"])
    parser.add_argument("--executable", type=Path, help="Run model checks against a bundled executable with ambient Python paths removed")
    args = parser.parse_args()
    if args.executable:
        args.executable = args.executable.expanduser().resolve()
        if not args.executable.is_file():
            parser.error('Executable does not exist')
    result = {"acceptance_scope": "engine_contract", "model_quality_status": "not_scored",
              "passed": [], "observations": {}}
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="titanos-check-") as directory:
            root = Path(directory)
            contracts(root, result["passed"])
            if args.models:
                live(root, args.models.resolve(), result["passed"], result["observations"], args.device, args.executable, args.dtype)
        result["success"] = True
    except Exception as exc:
        result.update(success=False, error=str(exc))
        raise
    finally:
        result["seconds"] = round(time.monotonic() - started, 2)
        if args.report:
            args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
