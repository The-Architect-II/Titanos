"""Durable model acquisition state, separate from inference asset jobs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid

from .engine import EngineError
from .providers import AcquisitionPlan, ProviderRegistry


TERMINAL = {"prepared", "active", "failed", "cancelled", "interrupted", "weights_deleted", "deleted"}


# Progress units per stage: file-count stages report files, everything else reports bytes. The record carries the
# unit explicitly so clients never label a file count as bytes.
PROGRESS_UNITS = {"downloading_files": "files", "extracting": "files"}


def progress_unit(stage):
    return PROGRESS_UNITS.get(stage, "bytes")


def evidence_digest(rows):
    values=sorted((str(row.get("path","")),str(row.get("sha256",""))) for row in rows)
    if not values or any(not re.fullmatch(r"[a-f0-9]{64}",digest) for _,digest in values):return None
    return hashlib.sha256("".join(path+":"+digest+"\n" for path,digest in values).encode()).hexdigest()
RUNNING = {"queued", "running", "cancelling"}


class AcquisitionStore:
    def __init__(self, root, providers=None, *, model_root=None, max_bytes=8 * 1024**3,
                 workers=1, retention_seconds=7 * 86400):
        self.root = Path(root)
        self.model_root = Path(model_root or self.root.parent / "models")
        self.providers = providers or ProviderRegistry()
        self.max_bytes = int(max_bytes)
        self.workers = int(workers)
        self.retention_seconds = float(retention_seconds)
        if self.max_bytes < 1 or self.workers < 1 or self.retention_seconds <= 0:
            raise ValueError("Acquisition limits must be positive")
        self.lock = threading.RLock()
        self.events = {}
        self.executor = None
        self.owner = None

    def start(self):
        with self.lock:
            if self.owner is not None:
                return
            from filelock import FileLock, Timeout
            self.root.mkdir(parents=True, exist_ok=True)
            self.model_root.mkdir(parents=True, exist_ok=True)
            owner = FileLock(str(self.root / "owner.lock"), thread_local=False)
            try:
                owner.acquire(timeout=0)
            except Timeout as exc:
                raise EngineError("Another server owns this acquisition directory", 409, "acquisition_store_busy") from exc
            try:
                now = time.time()
                for path in self.root.glob("*.json"):
                    try:
                        record = json.loads(path.read_text(encoding="utf-8"))
                        if record["id"] != path.stem or not re.fullmatch(r"[a-f0-9]{32}", record["id"]):
                            raise ValueError("Invalid acquisition identity")
                        if record["status"] in RUNNING:
                            record.update(status="interrupted",
                                          error="Previous server stopped before completion; acquisition was not replayed")
                            self.write(record)
                        elif (record.get("status") in {"failed","cancelled","interrupted","deleted","weights_deleted"}
                              and record["updated_at"] < now - self.retention_seconds):
                            path.unlink()
                            shutil.rmtree(self.root/record["id"],ignore_errors=True)
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                        path.rename(path.with_suffix(".invalid-" + uuid.uuid4().hex))
                self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="titanos-acquire")
                self.owner = owner
            except Exception:
                owner.release()
                raise

    def write(self, record):
        record["updated_at"] = time.time()
        self.root.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.root, prefix="acquisition-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.root / (record["id"] + ".json"))
        finally:
            Path(name).unlink(missing_ok=True)

    @staticmethod
    def public(record):
        value = copy.deepcopy(record)
        value.pop("request_key", None)
        value.pop("fingerprint", None)
        return value

    def _record(self, identifier):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise EngineError("Acquisition not found", 404, "not_found")
        path = self.root / (identifier + ".json")
        if not path.is_file():
            raise EngineError("Acquisition not found", 404, "not_found")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] == "deleted":
            raise EngineError("Acquisition deleted", 404, "not_found")
        return record

    def get(self, identifier):
        with self.lock:
            self.start()
            return self.public(self._record(identifier))

    def list(self, limit=100):
        if limit < 1 or limit > 1000:
            raise ValueError("Acquisition list limit must be 1..1000")
        with self.lock:
            self.start()
            rows = [json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("*.json")]
            rows = [row for row in rows if row.get("status") != "deleted"]
            rows.sort(key=lambda row: row["created_at"], reverse=True)
            return {"object": "list", "data": [self.public(row) for row in rows[:limit]]}

    @staticmethod
    def _safe_destination(plan):
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", plan.candidate_id).strip("-")[:80] or "model"
        identity = plan.fingerprint()[:20]
        return name + "-" + identity

    def _promote(self, identifier, plan, result):
        source = Path(result["path"]).resolve()
        job_root = (self.root / identifier).resolve()
        if not source.is_relative_to(job_root):
            return {**result, "path": str(source), "reused": True}
        destination = (self.model_root / plan.provider / self._safe_destination(plan)).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1, "provider": plan.provider, "candidate_id": plan.candidate_id,
            "requested_revision": plan.requested_revision, "resolved_revision": plan.resolved_revision,
            "plan_fingerprint": plan.fingerprint(), "files": [row.model_dump(mode="json") for row in plan.files],
            "archive_sha256": result.get("archive_sha256"), "license": result.get("license"),
            "license_evidence": result.get("license_evidence", []),
            "license_evidence_digest":evidence_digest(result.get("license_evidence",[])),
            "integrity":result.get("integrity",[]),
        }
        if destination.exists():
            path = destination / "titanos-acquisition.json"
            if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("plan_fingerprint") != plan.fingerprint():
                raise ValueError("Prepared model destination exists with a different acquisition identity")
            shutil.rmtree(job_root,ignore_errors=True)
            return {**result, "path": str(destination), "reused": True}
        (source / "titanos-acquisition.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(source, destination)
        shutil.rmtree(job_root,ignore_errors=True)
        return {**result, "path": str(destination)}

    def _dispatch(self, record):
        identifier = record["id"]
        plan = AcquisitionPlan.model_validate(record["plan"])
        event = self.events[identifier] = threading.Event()

        def progress(stage, completed, total):
            with self.lock:
                if self.events.get(identifier) is not event:
                    return
                record["progress"] = {"stage": stage, "completed": int(completed),
                                      "total": int(total) if total is not None else None, "unit": progress_unit(stage)}
                self.write(record)

        def run():
            failure = None
            result = None
            try:
                with self.lock:
                    if self.events.get(identifier) is not event:
                        return
                    if event.is_set():
                        raise RuntimeError("Acquisition cancelled")
                    record.update(status="running", error=None, attempt=record.get("attempt", 0) + 1)
                    self.write(record)
                staging = self.root / identifier
                result = self.providers.get(plan.provider).acquire(plan, staging, progress, event)
                if event.is_set():
                    raise RuntimeError("Acquisition cancelled")
                result = self._promote(identifier, plan, result)
            except Exception as exc:
                failure = exc
            with self.lock:
                if event.is_set():
                    record.update(status="cancelled", error="Acquisition cancelled")
                elif failure is not None:
                    record.update(status="failed", error=str(failure))
                else:
                    record.update(status="prepared", prepared_path=result["path"],
                                  license=result.get("license"), license_evidence=result.get("license_evidence", []),
                                  license_evidence_digest=evidence_digest(result.get("license_evidence",[])),
                                  archive_sha256=result.get("archive_sha256"),
                                  integrity=result.get("integrity",[]),reused=bool(result.get("reused")))
                    record["progress"] = {"stage": "prepared",
                                          "completed": plan.expected_bytes or record.get("progress", {}).get("completed", 0),
                                          "total": plan.expected_bytes, "unit": "bytes"}
                self.write(record)
                self.events.pop(identifier, None)

        try:
            self.executor.submit(run)
        except Exception:
            self.events.pop(identifier, None)
            raise

    def submit(self, plan, request_key=None):
        plan = AcquisitionPlan.model_validate(plan)
        self.start()
        if plan.expected_bytes is None:
            raise EngineError("Provider did not supply a bounded storage estimate",422,"acquisition_size_unknown")
        if plan.expected_bytes > self.max_bytes:
            raise EngineError("Acquisition exceeds the configured storage download limit",413,"acquisition_too_large")
        required=plan.expected_bytes*(6 if plan.archive_format else 1)+128*1024**2
        free=shutil.disk_usage(self.model_root).free
        if free<required:
            raise EngineError(f"Insufficient storage: acquisition needs about {required} bytes but {free} are free",
                              507,"insufficient_storage")
        with self.lock:
            fingerprint = plan.fingerprint()
            records=[json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("*.json")]
            if request_key:
                for previous in records:
                    if previous.get("request_key") == request_key:
                        if previous.get("fingerprint") != fingerprint:
                            raise EngineError("Idempotency key already used for another acquisition", 409,
                                              "idempotency_conflict")
                        return self.public(previous)
            for previous in records:
                if (previous.get("fingerprint")==fingerprint
                        and previous.get("status") in {"queued","running","cancelling","prepared","active"}):
                    return self.public(previous)
            identifier = uuid.uuid4().hex
            record = {"id": identifier, "object": "model.acquisition", "status": "queued",
                      "created_at": time.time(), "updated_at": time.time(), "attempt": 0,
                      "plan": plan.model_dump(mode="json"),
                      "progress": {"stage": "queued", "completed": 0, "total": plan.expected_bytes, "unit": "bytes"},
                      "fingerprint": fingerprint}
            if request_key:
                record["request_key"] = request_key
            self.write(record)
            self._dispatch(record)
            return self.public(record)

    def cancel(self, identifier):
        with self.lock:
            self.start()
            record = self._record(identifier)
            event = self.events.get(identifier)
            if event is not None:
                event.set()
                record["status"] = "cancelling"
                self.write(record)
            return self.public(record)

    def retry(self, identifier):
        with self.lock:
            self.start()
            record = self._record(identifier)
            if record["status"] not in {"failed", "cancelled", "interrupted"}:
                raise EngineError("Only failed, cancelled or interrupted acquisitions can retry", 409,
                                  "acquisition_not_retryable")
            record.update(status="queued", error=None,
                          progress={"stage": "queued", "completed": 0,
                                    "total": AcquisitionPlan.model_validate(record["plan"]).expected_bytes, "unit": "bytes"})
            self.write(record)
            self._dispatch(record)
            return self.public(record)

    def prepared(self, identifier):
        with self.lock:
            record = self._record(identifier)
            if record["status"] != "prepared":
                raise EngineError("Acquisition is not prepared for activation", 409, "acquisition_not_prepared")
            return self.public(record)

    def mark_active(self, identifier, activation, accepted_license_sha256=None):
        with self.lock:
            record = self.activation_ready(identifier, accepted_license_sha256)
            record.update(status="active", activation=activation)
            if record.get("license") is None:
                record["license_acknowledgement"]={"sha256":accepted_license_sha256,"accepted_at":time.time()}
            self.write(record)
            return self.public(record)

    def activation_ready(self, identifier, accepted_license_sha256=None):
        with self.lock:
            record = self._record(identifier)
            if record["status"] != "prepared":
                raise EngineError("Acquisition must be prepared before activation", 409,
                                  "acquisition_not_prepared")
            prepared=Path(record.get("prepared_path","")).resolve()
            root=self.model_root.resolve()
            manifest=prepared/"titanos-acquisition.json"
            if (not prepared.is_relative_to(root) or not prepared.is_dir() or prepared.is_symlink()
                    or not manifest.is_file()):
                raise EngineError("Prepared model storage is missing or no longer owned by this acquisition",409,
                                  "prepared_storage_missing")
            try:identity=json.loads(manifest.read_text(encoding="utf-8")).get("plan_fingerprint")
            except (OSError,ValueError):identity=None
            if identity!=record.get("fingerprint"):
                raise EngineError("Prepared model storage identity does not match this acquisition",409,
                                  "prepared_storage_mismatch")
            evidence = record.get("license_evidence", [])
            if record.get("license") is None and not evidence:
                raise EngineError("Provider did not supply model license evidence; activation is blocked",
                                  409, "model_license_unknown")
            if record.get("license") is None and evidence:
                expected=record.get("license_evidence_digest") or evidence_digest(evidence)
                if accepted_license_sha256 != expected:
                    raise EngineError("Review all acquired model license evidence and acknowledge its combined SHA-256 before activation",
                                      409, "license_acknowledgement_required")
            return record

    def delete(self, identifier):
        with self.lock:
            record = self._record(identifier)
            if identifier in self.events:
                raise EngineError("Cancel the acquisition before deleting its record", 409, "acquisition_active")
            record["status"] = "deleted"
            self.write(record)
            shutil.rmtree(self.root/identifier,ignore_errors=True)
            return {"id": identifier, "object": "model.acquisition", "deleted": True}

    def delete_weights(self,identifier):
        with self.lock:
            record=self._record(identifier)
            if identifier in self.events:raise EngineError('Cancel the acquisition before deleting weights',409,'acquisition_active')
            value=record.get('prepared_path')
            if not value:raise EngineError('Acquisition has no managed prepared weights',409,'weights_unavailable')
            path=Path(value).resolve();root=self.model_root.resolve()
            if not path.is_relative_to(root):
                raise EngineError('Provider cache weights are externally managed; remove them with the provider cache tool',409,'external_cache')
            if path.exists():shutil.rmtree(path)
            record.update(status='weights_deleted',weights_deleted_at=time.time())
            record.pop('prepared_path',None);record.pop('activation',None)
            self.write(record)
            return self.public(record)

    def close(self):
        with self.lock:
            for event in self.events.values():
                event.set()
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        if self.owner:
            self.owner.release()
            self.owner = None
