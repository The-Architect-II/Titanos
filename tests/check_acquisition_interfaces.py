"""HTTP acquisition lifecycle with a controlled provider; no network/model execution."""
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from titanos.config import Config, read_config, write_config
from titanos.providers import AcquisitionRequest, ModelProvider
from titanos.server import create_app


class Provider(ModelProvider):
    name = "controlled"
    title = "Controlled"

    def search(self, query, task, limit=20):
        return [{"provider":self.name,"id":"controlled/model","name":"Controlled model","task":task,
                 "backend":"transformers","source_url":"https://fixtures.invalid/model",
                 "compatibility":{"status":"candidate"}}]

    def inspect(self, identifier, revision=None):
        return {"provider":self.name,"id":identifier,"source_url":"https://fixtures.invalid/model",
                "requested_revision":revision or "v1","resolved_revision":"b"*40,
                "files":[{"path":"model.safetensors","size":7}],"license":"apache-2.0","gated":False}

    def plan(self, request):
        return self._base_plan(request,self.inspect(request.identifier,request.revision),
                               [{"path":"model.safetensors","size":7}])

    def acquire(self, plan, staging, progress, cancel):
        model=staging/"model";model.mkdir(parents=True,exist_ok=True)
        progress("downloading",7,7)
        (model/"model.safetensors").write_bytes(b"fixture")
        return {"path":str(model),"license":"apache-2.0","license_evidence":[],"reused":False}


def wait(client,identifier):
    for _ in range(200):
        value=client.get("/api/acquisitions/"+identifier).json()
        if value["status"] in {"prepared","failed","cancelled"}:return value
        time.sleep(.01)
    raise AssertionError("acquisition did not finish")


def main():
    with tempfile.TemporaryDirectory(prefix="titanos-acquisition-http-") as folder:
        root=Path(folder);registry=root/"models.json";write_config(registry,Config())
        app=create_app(read_config(registry),registry_path=registry,output_dir=root/"outputs")
        provider=Provider();app.state.providers.providers={provider.name:provider}
        with TestClient(app,base_url="http://127.0.0.1") as client:
            providers=client.get("/api/model-providers").json()
            assert providers["data"][0]["id"]=="controlled"
            search=client.get("/api/model-providers/controlled/search",params={"task":"embeddings","query":"mini"}).json()
            assert search["data"][0]["compatibility"]["status"]=="candidate"
            page={"source_url":"https://models.example.org/card","final_url":"https://models.example.org/card",
                  "title":"Public model","description":"Unverified page","license_claim":"apache-2.0",
                  "model_links":[],"license_links":[],"content_sha256":"1"*64,"content_bytes":10,
                  "redirects":[],"compatibility":"metadata-only","qualification":None,"note":"Unverified"}
            with patch("titanos.server.fetch_model_page",return_value=page) as fetch:
                imported=client.post("/api/model-providers/import",json={"url":"https://models.example.org/card"}).json()
            assert imported["imported"]["compatibility"]=="metadata-only" and imported["metadata"]["title"]=="Public model"
            fetch.assert_called_once()
            request=AcquisitionRequest(provider="controlled",identifier="controlled/model",revision="v1",
                                       registry_name="embed",task="embeddings",backend="transformers",device="cpu")
            plan=client.post("/api/acquisitions/plan",json=request.model_dump(mode="json")).json()
            assert plan["resolved_revision"]=="b"*40 and plan["expected_bytes"]==7
            job=client.post("/api/acquisitions",json=plan,headers={"Idempotency-Key":"one"}).json()
            assert client.post("/api/acquisitions",json=plan,headers={"Idempotency-Key":"one"}).json()["id"]==job["id"]
            prepared=wait(client,job["id"]);assert prepared["status"]=="prepared"
            assert "embed" not in {row["id"] for row in client.get("/v1/models").json()["data"]}
            failed_load=client.post(f"/api/acquisitions/{job['id']}/activate",json={"load":True})
            assert failed_load.status_code>=400
            assert "embed" not in {row["id"] for row in client.get("/v1/models").json()["data"]}
            assert client.get(f"/api/acquisitions/{job['id']}").json()["status"]=="prepared"
            active=client.post(f"/api/acquisitions/{job['id']}/activate",json={"load":False}).json()
            assert active["status"]=="active" and active["activation"]["status"]=="registered"
            saved=read_config(registry).models["embed"]
            assert saved.source_repo=="https://fixtures.invalid/model" and saved.source_revision=="b"*40
            assert Path(saved.path).is_dir()
            assert client.get("/api/acquisitions").json()["data"][0]["status"]=="active"
            assert client.delete("/api/acquisitions/not-an-id").status_code==404
            info=client.get("/api/info").json()
            assert info["model_setup"]["state_is_distinct_from_asset_jobs"] is True
        print(json.dumps({"passed":True,"scope":"controlled provider HTTP plan/acquire/activate; no model inference"}))


if __name__=="__main__":main()
