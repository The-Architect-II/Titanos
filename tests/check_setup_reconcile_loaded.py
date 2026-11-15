"""The compatibility advisory is reconciled with the observed loaded runtime, never by suppression.

Pure checks of `reconcile_setup_with_runtime`; the engine's non-loading, non-blocking `loaded_details` with a fake resident
runtime; the HTTP capabilities route on an unloaded model (passive result unchanged), on a resident model (observed row,
verified_at_load), under a busy engine lock (passive view kept, request not blocked), and on a worker-bearing registry whose
engine is a ReplicaEngine (no 500; unassigned resident models still reconcile). No model weights, no worker process, no network."""
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titanos.model_inspection import RUNTIME_OBSERVABLE, reconcile_setup_with_runtime  # noqa: E402

WRAPPER = "13ef823ffe76aec4ac7e35d3fec28ecfc8d7427a"
DETAILS = {"backend": "qwen3-tts", "device": "cuda:0", "dtype": "torch.bfloat16", "wrapper_revision": WRAPPER, "voice_mode": "x_vector_only",
           "registered_voices": ["vega"], "reference_bindings": {"vega": {"file_sha256": "e" * 64, "seconds": 5}}}


class FakeResidentRuntime:
    """Stands in for a loaded QwenTTSModel: reports details, has no fault, and records that nothing else was called."""
    device = "cuda:0"

    def __init__(self):
        self.calls = 0

    def details(self):
        self.calls += 1
        return dict(DETAILS)


def passive(extra=()):
    return {"schema_version": 1, "status": "needs_review", "advisory": True, "model_loaded": False, "qualified": None,
            "issues": [{"code": "voice_wrapper_unverified", "severity": "warning", "message": "The streaming wrapper revision and PCM16 reference voices were not loaded.", "files": [], "next_action": "Use the documented pinned Qwen streaming wrapper and registered PCM16 reference audio."}, *extra]}


def make_resident(engine, name, runtime):
    with engine.state_lock:
        engine.current, engine.runtime = name, runtime
        engine.states[name] = "loaded"


def make_unloaded(engine, name):
    with engine.state_lock:
        engine.current, engine.runtime = None, None
        engine.states[name] = "unloaded"


def main():
    checks = 0
    # loaded runtime confirms wrapper + voices -> observed info row, verified_at_load, model_loaded true
    out = reconcile_setup_with_runtime(passive(), "qwen3-tts", DETAILS)
    assert out["status"] == "verified_at_load" and out["model_loaded"] is True, out["status"]; checks += 1
    row = out["issues"][0]
    assert row["code"] == "voice_wrapper_observed" and row["severity"] == "info" and row["next_action"] is None; checks += 1
    assert row["message"].startswith("Loaded: streaming wrapper revision " + WRAPPER[:12]) and "(vega)" in row["message"]; checks += 1
    assert row["observed"]["voices"] == ["vega"] and row["observed"]["wrapper_revision"] == WRAPPER; checks += 1
    assert out["observed_at_load"]["reconciled_codes"] == ["voice_wrapper_observed"]; checks += 1
    # the passive input is not mutated
    assert passive()["issues"][0]["code"] == "voice_wrapper_unverified"; checks += 1
    # errors are never suppressed: a missing-file error keeps the status blocked while the observed row is still recorded
    err = {"code": "missing_file", "severity": "error", "message": "Required component missing.", "files": ["speech_tokenizer/model.safetensors"], "next_action": "Restore the file."}
    out = reconcile_setup_with_runtime(passive([err]), "qwen3-tts", DETAILS)
    assert out["status"] == "blocked" and [r["code"] for r in out["issues"]] == ["voice_wrapper_observed", "missing_file"]; checks += 1
    # an unrelated warning keeps needs_review
    warn = {"code": "layout_unverified", "severity": "warning", "message": "x", "files": [], "next_action": "y"}
    assert reconcile_setup_with_runtime(passive([warn]), "qwen3-tts", DETAILS)["status"] == "needs_review"; checks += 1
    # not loaded / no bindings / faulted runtime / other backend -> unchanged object
    for details in (None, {}, {"reference_bindings": {}}, {"wrapper_revision": WRAPPER}, {**DETAILS, "status": "unavailable"}):
        assert reconcile_setup_with_runtime(passive(), "qwen3-tts", details) == passive(); checks += 1
    assert reconcile_setup_with_runtime(passive(), "transformers", DETAILS) == passive(); checks += 1
    # only qwen3-tts is observable: the generic xtts runtime details carry no voice bindings, so its advisory stays passive
    assert set(RUNTIME_OBSERVABLE) == {"qwen3-tts"}; checks += 1
    x = {"schema_version": 1, "status": "needs_review", "issues": [{"code": "voice_reference_unverified", "severity": "warning", "message": "XTTS reference audio and checkpoint compatibility were not loaded.", "files": [], "next_action": "n"}]}
    assert reconcile_setup_with_runtime(x, "xtts", {"backend": "xtts", "device": "cpu", "reference_bindings": {"vega": {}}}) == x; checks += 1
    # an advisory list without the targeted row is returned unchanged even for a loaded qwen3-tts runtime
    no_row = {**passive(), "issues": [warn]}
    assert reconcile_setup_with_runtime(no_row, "qwen3-tts", DETAILS) == no_row; checks += 1
    checks += engine_checks()
    checks += route_checks()
    checks += replica_route_checks()
    print(f"PASS: {checks} reconciliation checks (observed rows replace only the load-verifiable advisory; errors/warnings kept; unloaded, faulted and "
          f"non-observable runtimes unchanged; engine read never loads and never blocks; route reconciles a resident model, keeps the passive view "
          f"under a busy lock, and stays 200 on a worker-bearing ReplicaEngine)")


def engine_checks():
    from titanos.config import Config, ModelSpec
    from titanos.engine import Engine
    checks = 0
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); (root / "speech").mkdir()
        cfg = Config(models={"speech": ModelSpec(path=str(root / "speech"), task="speech", backend="qwen3-tts", device="cpu")})
        engine = Engine(cfg)
        # unloaded model: no details and, above all, no load attempt
        assert engine.loaded_details("speech") is None and engine.states["speech"] == "unloaded"; checks += 1
        assert engine.loaded_details("absent") is None; checks += 1
        # resident fake runtime: details are returned exactly once per read, without a load
        runtime = FakeResidentRuntime(); make_resident(engine, "speech", runtime)
        assert engine.loaded_details("speech") == DETAILS and runtime.calls == 1; checks += 1
        assert engine.loaded_details("absent") is None and runtime.calls == 1, "another name must not read the resident runtime"; checks += 1
        # a faulted resident runtime is not reported
        runtime.runtime_fault = "device lost"
        assert engine.loaded_details("speech") is None and runtime.calls == 1; checks += 1
        del runtime.runtime_fault
        # busy engine lock (an inference in progress): the read returns None promptly instead of waiting
        assert engine.lock.acquire(blocking=False)
        try:
            started = time.monotonic(); result = engine.loaded_details("speech"); waited = time.monotonic() - started
            assert result is None and waited < 1.0 and runtime.calls == 1, (result, waited); checks += 1
        finally:
            engine.lock.release()
        assert engine.loaded_details("speech") == DETAILS and runtime.calls == 2; checks += 1
        make_unloaded(engine, "speech")
        assert engine.loaded_details("speech") is None and runtime.calls == 2; checks += 1
    return checks


def route_checks():
    from fastapi.testclient import TestClient
    from titanos.config import Config, ModelSpec, read_config, write_config
    from titanos.server import create_app
    checks = 0
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); (root / "speech").mkdir(); (root / "chat").mkdir()
        cfg = Config(models={"speech": ModelSpec(path=str(root / "speech"), task="speech", backend="qwen3-tts", device="cpu"),
                             "chat": ModelSpec(path=str(root / "chat"), task="chat", backend="transformers", device="cpu")})
        write_config(root / "models.json", cfg)
        app = create_app(read_config(root / "models.json"), api_key="test-secret", output_dir=root / "outputs")
        engine = app.state.engine.target
        with TestClient(app, base_url="http://127.0.0.1") as c:
            c.headers["Authorization"] = "Bearer test-secret"
            caps = lambda name: c.post("/api/models/capabilities", json={"model": name})  # noqa: E731
            setup = caps("speech").json()["setup"]
            assert setup["model_loaded"] is False and "observed_at_load" not in setup; checks += 1
            assert any(row["code"] == "voice_wrapper_unverified" for row in setup["issues"]), "passive advisory must remain for an unloaded model"; checks += 1
            assert c.get("/v1/models").json()["data"][0]["status"] == "unloaded", "capabilities must not load the model"; checks += 1
            # resident model: the route reconciles to the observed row
            runtime = FakeResidentRuntime(); make_resident(engine, "speech", runtime)
            setup = caps("speech").json()["setup"]
            # the empty checkpoint directory still has real missing-file/runtime errors: they are kept and the status stays
            # blocked; only the load-verifiable advisory is replaced and the residency is reported
            errors = [row for row in setup["issues"] if row["severity"] == "error"]
            assert setup["status"] == ("blocked" if errors else "verified_at_load") and setup["model_loaded"] is True and setup["observed_at_load"]["backend"] == "qwen3-tts", setup["status"]; checks += 1
            assert errors and all(row["code"] in {"missing_file", "runtime_missing"} for row in errors), "passive errors must survive reconciliation"; checks += 1
            codes = [row["code"] for row in setup["issues"]]
            assert "voice_wrapper_observed" in codes and "voice_wrapper_unverified" not in codes; checks += 1
            observed = next(row for row in setup["issues"] if row["code"] == "voice_wrapper_observed")
            assert observed["observed"]["wrapper_revision"] == WRAPPER and observed["observed"]["voices"] == ["vega"] and observed["next_action"] is None; checks += 1
            # a non-observable resident backend is never queried: its details are not read and its setup stays passive
            make_resident(engine, "chat", runtime); before = runtime.calls
            chat_setup = caps("chat").json()["setup"]
            assert runtime.calls == before and chat_setup.get("model_loaded") is False and "observed_at_load" not in chat_setup; checks += 1
            make_resident(engine, "speech", runtime)
            # busy engine lock: the request is answered promptly with the passive view
            assert engine.lock.acquire(blocking=False)
            try:
                started = time.monotonic(); r = caps("speech"); waited = time.monotonic() - started
                assert r.status_code == 200 and waited < 5.0, (r.status_code, waited); checks += 1
                setup = r.json()["setup"]
                assert setup["model_loaded"] is False and "voice_wrapper_unverified" in [row["code"] for row in setup["issues"]]; checks += 1
            finally:
                engine.lock.release()
            assert caps("speech").json()["setup"]["model_loaded"] is True; checks += 1
            make_unloaded(engine, "speech")
            assert caps("speech").json()["setup"]["model_loaded"] is False; checks += 1
    return checks


def replica_route_checks():
    """A registry with a worker assignment gives the server a ReplicaEngine; the route must stay 200 for every model, and an
    unassigned resident model still reconciles through the replica's local engine. Workers are constructed lazily (no process)."""
    from fastapi.testclient import TestClient
    from titanos.config import Config, ModelSpec, WorkerSpec, read_config, write_config
    from titanos.replicas import ReplicaEngine
    from titanos.server import create_app
    checks = 0
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); (root / "speech").mkdir(); (root / "chat").mkdir()
        cfg = Config(models={"speech": ModelSpec(path=str(root / "speech"), task="speech", backend="qwen3-tts", device="cpu"),
                             "chat": ModelSpec(path=str(root / "chat"), task="chat", backend="transformers", device="cpu")},
                     workers={"studio-cpu-1": WorkerSpec(models=["chat"], device="cpu")})
        write_config(root / "models.json", cfg)
        app = create_app(read_config(root / "models.json"), api_key="test-secret", output_dir=root / "outputs")
        engine = app.state.engine.target
        assert isinstance(engine, ReplicaEngine), type(engine); checks += 1
        assert engine.loaded_details("chat") is None and engine.loaded_details("speech") is None; checks += 1
        c = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)  # no lifespan: nothing to start or dispose
        c.headers["Authorization"] = "Bearer test-secret"
        for name in ("chat", "speech"):
            r = c.post("/api/models/capabilities", json={"model": name})
            assert r.status_code == 200, (name, r.status_code, r.text[:200]); checks += 1
            assert r.json()["setup"]["model_loaded"] is False; checks += 1
        # the unassigned speech model becomes resident in the replica's local engine: reconciled like a single-process server
        runtime = FakeResidentRuntime(); make_resident(engine.local, "speech", runtime)
        setup = c.post("/api/models/capabilities", json={"model": "speech"}).json()["setup"]
        assert setup["model_loaded"] is True and "voice_wrapper_observed" in [row["code"] for row in setup["issues"]] and runtime.calls == 1; checks += 1
        # the worker-assigned model is never queried through the local engine
        assert c.post("/api/models/capabilities", json={"model": "chat"}).status_code == 200 and runtime.calls == 1; checks += 1
        make_unloaded(engine.local, "speech")
    return checks


if __name__ == "__main__":
    main()
