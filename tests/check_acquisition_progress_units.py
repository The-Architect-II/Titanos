"""Acquisition and runtime-setup progress records name their unit, so a per-file download stage is never presented as bytes.

Pure record checks against the store's progress helper and the runtime-setup record shape; no network, no model, no server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titanos.acquisition import PROGRESS_UNITS, progress_unit  # noqa: E402


def main():
    checks = 0
    assert progress_unit("downloading_files") == "files"; checks += 1
    assert progress_unit("extracting") == "files"; checks += 1
    for stage in ("queued", "downloading", "verifying", "prepared", "unknown_future_stage"):
        assert progress_unit(stage) == "bytes", stage; checks += 1
    assert set(PROGRESS_UNITS) == {"downloading_files", "extracting"}; checks += 1
    # the store's progress closure and terminal record carry the unit field (source-level contract)
    source = (Path(__file__).resolve().parents[1] / "src/titanos/acquisition.py").read_text(encoding="utf-8")
    assert '"unit": progress_unit(stage)' in source; checks += 1
    assert source.count('"unit": "bytes"') >= 3, "queued, retry and prepared records must carry unit"; checks += 1
    native = (Path(__file__).resolve().parents[1] / "src/titanos/native_install.py").read_text(encoding="utf-8")
    assert "'unit': 'files' if stage == 'extracting'" in native and "'stage': 'queued', 'completed': 0, 'total': None, 'unit': 'bytes'" in native; checks += 1
    checks += store_level_record_checks()
    print(f"PASS: {checks} progress-unit checks (files for downloading_files/extracting, bytes otherwise; records carry unit; store records observed)")


def store_level_record_checks():
    """Drive the real AcquisitionStore with the controlled fixture provider: a per-file stage is recorded as files, the
    terminal prepared record as bytes."""
    import tempfile
    import threading
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_acquisition import ControlledProvider, wait  # noqa: E402
    from titanos.acquisition import AcquisitionStore  # noqa: E402
    from titanos.providers import AcquisitionRequest, ProviderRegistry  # noqa: E402

    class FileCountingProvider(ControlledProvider):
        name = "controlled"

        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def acquire(self, plan, staging, progress, cancel):
            staging.mkdir(parents=True, exist_ok=True)
            progress("downloading_files", 0, 12)
            self.entered.set()
            self.release.wait(5)
            progress("downloading_files", 12, 12)
            model = staging / "model"
            model.mkdir(exist_ok=True)
            (model / "model.safetensors").write_bytes(b"fixture")
            return {"path": str(model), "license": "apache-2.0", "license_evidence": [], "reused": False}

    provider = FileCountingProvider()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        store = AcquisitionStore(root / "acquisitions", ProviderRegistry([provider]), model_root=root / "models", max_bytes=1024, workers=1)
        store.start()
        try:
            request = AcquisitionRequest(provider="controlled", identifier="fixture/model", revision="v1", registry_name="fixture", task="embeddings", backend="transformers", device="cpu")
            job = store.submit(provider.plan(request))
            assert job["progress"]["unit"] == "bytes" and job["progress"]["stage"] == "queued"
            assert provider.entered.wait(5), "provider never entered"
            mid = None
            for _ in range(200):
                mid = store.get(job["id"])["progress"]
                if mid["stage"] == "downloading_files":
                    break
                time.sleep(.01)
            assert mid["stage"] == "downloading_files" and mid["unit"] == "files" and mid["total"] == 12, mid
            provider.release.set()
            final = wait(store, job["id"])
            assert final["status"] == "prepared" and final["progress"]["unit"] == "bytes" and final["progress"]["stage"] == "prepared", final["progress"]
        finally:
            store.close()
    return 3


if __name__ == "__main__":
    main()
