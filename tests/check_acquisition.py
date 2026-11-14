"""Durable acquisition mechanics with controlled providers; no network/model runtime."""
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest

from titanos.acquisition import AcquisitionStore,evidence_digest
from titanos.engine import EngineError
from titanos.providers import AcquisitionRequest, ModelProvider, ProviderRegistry


class ControlledProvider(ModelProvider):
    name = "controlled"
    title = "Controlled fixture"

    def __init__(self):
        self.mode = "success"
        self.entered = threading.Event()
        self.calls = 0

    def search(self, query, task, limit=20):
        return []

    def inspect(self, identifier, revision=None):
        return {"source_url": "https://fixtures.invalid/model", "requested_revision": revision or "v1",
                "resolved_revision": "a" * 40, "license": "apache-2.0", "gated": False}

    def plan(self, request):
        return self._base_plan(request, self.inspect(request.identifier, request.revision),
                               [{"path": "model.safetensors", "size": 12}])

    def acquire(self, plan, staging, progress, cancel):
        self.calls += 1
        staging.mkdir(parents=True, exist_ok=True)
        partial = staging / "partial.bytes"
        completed = partial.stat().st_size if partial.exists() else 0
        for index in range(completed, 12):
            if cancel.is_set():
                raise RuntimeError("Acquisition cancelled")
            with partial.open("ab") as stream:
                stream.write(bytes([index]))
            self.entered.set()  # only once a partial byte exists: a cancel that wins the race must still leave one to resume
            progress("downloading", index + 1, 12)
            if self.mode == "slow":
                time.sleep(.02)
            if self.mode == "fail" and index == 3:
                raise ValueError("fixture integrity failure")
        model = staging / "model"
        model.mkdir(exist_ok=True)
        (model / "model.safetensors").write_bytes(b"fixture")
        if self.mode=="unknown-license":
            rows=[{"path":"LICENSE","sha256":"1"*64,"text":"one"},{"path":"NOTICE","sha256":"2"*64,"text":"two"}]
            return {"path":str(model),"license":None,"license_evidence":rows,"reused":completed>0}
        return {"path": str(model), "license": "apache-2.0", "license_evidence": [], "reused": completed > 0}


def wait(store, identifier, terminal=None):
    terminal = terminal or {"prepared", "failed", "cancelled", "interrupted"}
    for _ in range(300):
        value = store.get(identifier)
        if value["status"] in terminal:
            return value
        time.sleep(.01)
    raise AssertionError("acquisition did not reach terminal state")


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.provider = ControlledProvider()
        self.providers = ProviderRegistry([self.provider])
        self.request = AcquisitionRequest(provider="controlled", identifier="fixture/model", revision="v1",
                                          registry_name="fixture", task="embeddings", backend="transformers", device="cpu")
        self.plan = self.provider.plan(self.request)

    def store(self, **kwargs):
        store = AcquisitionStore(self.root / "acquisitions", self.providers,
                                 model_root=self.root / "models", max_bytes=1024, workers=1, **kwargs)
        store.start()
        self.addCleanup(store.close)
        return store

    def test_prepare_progress_atomic_promotion_and_restart_reuse(self):
        store = self.store()
        job = store.submit(self.plan)
        result = wait(store, job["id"])
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["progress"]["completed"], 12)
        model = Path(result["prepared_path"])
        self.assertTrue((model / "model.safetensors").is_file())
        manifest = json.loads((model / "titanos-acquisition.json").read_text())
        self.assertEqual(manifest["plan_fingerprint"], self.plan.fingerprint())
        self.assertEqual(manifest["resolved_revision"], "a" * 40)
        store.mark_active(job["id"], {"model": "fixture", "status": "registered"})
        self.assertEqual(store.get(job["id"])["status"], "active")
        again = store.submit(self.plan)
        self.assertEqual(again["id"],job["id"])
        prepared = wait(store, again["id"],{"prepared","active"})
        self.assertEqual(prepared["status"],"active")
        self.assertEqual(Path(prepared["prepared_path"]), model)
        self.assertEqual(self.provider.calls,1)
        store.close()
        restarted=self.store();same=restarted.submit(self.plan)
        self.assertEqual(same["id"],job["id"]);self.assertEqual(self.provider.calls,1)

    def test_distinct_registry_destinations_and_missing_storage_refusal(self):
        store=self.store()
        first=wait(store,store.submit(self.plan)["id"])
        other=self.provider.plan(self.request.model_copy(update={"registry_name":"other"}))
        second=wait(store,store.submit(other)["id"])
        self.assertNotEqual(first["prepared_path"],second["prepared_path"])
        shutil.rmtree(second["prepared_path"])
        with self.assertRaisesRegex(EngineError,"storage"):
            store.mark_active(second["id"],{"model":"other"})

    def test_cancel_retry_resume_and_no_false_activation(self):
        self.provider.mode = "slow"
        store = self.store()
        job = store.submit(self.plan)
        self.assertTrue(self.provider.entered.wait(1))
        store.cancel(job["id"])
        cancelled = wait(store, job["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertNotIn("prepared_path", cancelled)
        with self.assertRaisesRegex(EngineError, "prepared"):
            store.mark_active(job["id"], {})
        retried = store.retry(job["id"])
        self.assertEqual(retried["id"], job["id"])
        prepared = wait(store, job["id"])
        self.assertEqual(prepared["status"], "prepared")
        self.assertTrue(prepared["reused"])

    def test_failure_and_oversize_never_prepare(self):
        self.provider.mode = "fail"
        store = self.store()
        failed = wait(store, store.submit(self.plan)["id"])
        self.assertEqual(failed["status"], "failed")
        self.assertNotIn("prepared_path", failed)
        oversized = self.plan.model_copy(update={"expected_bytes": 2048})
        with self.assertRaisesRegex(EngineError, "storage"):
            store.submit(oversized)

    def test_unknown_license_requires_combined_evidence_digest(self):
        self.provider.mode="unknown-license";store=self.store()
        prepared=wait(store,store.submit(self.plan)["id"])
        rows=prepared["license_evidence"]
        self.assertEqual(prepared["license_evidence_digest"],evidence_digest(rows))
        with self.assertRaisesRegex(EngineError,"all acquired"):
            store.mark_active(prepared["id"],{},rows[0]["sha256"])
        active=store.mark_active(prepared["id"],{},prepared["license_evidence_digest"])
        self.assertEqual(active["license_acknowledgement"]["sha256"],prepared["license_evidence_digest"])

    def test_restart_marks_inflight_interrupted_without_replay(self):
        root = self.root / "acquisitions"
        root.mkdir()
        identifier = "1" * 32
        record = {"id": identifier, "object": "model.acquisition", "status": "running",
                  "created_at": time.time(), "updated_at": time.time(), "attempt": 1,
                  "plan": self.plan.model_dump(mode="json"), "progress": {"stage": "downloading", "completed": 2, "total": 12}}
        (root / f"{identifier}.json").write_text(json.dumps(record))
        store = self.store()
        value = store.get(identifier)
        self.assertEqual(value["status"], "interrupted")
        self.assertIn("not replayed", value["error"])
        self.assertEqual(self.provider.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
