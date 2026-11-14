"""Managed GGUF acquisition can register the existing native worker contract."""
from pathlib import Path
import os
import tempfile
import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from titanos.config import Config,read_config,write_config
from titanos.providers import AcquisitionRequest,ModelProvider
from titanos.server import create_app


class Provider(ModelProvider):
    name="controlled-native";title="Controlled native"
    def search(self,*args):return []
    def inspect(self,identifier,revision=None):
        return {"source_url":"https://fixtures.invalid/native","requested_revision":revision or "v1",
                "resolved_revision":"b"*40,"license":"apache-2.0","gated":False}
    def plan(self,request):
        return self._base_plan(request,self.inspect(request.identifier,request.revision),
                               [{"path":"model-q4_k_m.gguf","size":8}])
    def acquire(self,plan,staging,progress,cancel):
        root=staging/"model";root.mkdir(parents=True);(root/"model-q4_k_m.gguf").write_bytes(b"GGUFtest")
        progress("downloading",8,8);return {"path":str(root),"license":"apache-2.0","license_evidence":[]}


if __name__=="__main__":
    with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'TITANOS_HOME':folder}):
        root=Path(folder);registry=root/"models.json";write_config(registry,Config())
        app=create_app(Config(),registry_path=registry,output_dir=root/"outputs",runtime_catalog=False)
        app.state.providers.providers={"controlled-native":Provider()}
        with TestClient(app,base_url="http://127.0.0.1") as client:
            request=AcquisitionRequest(provider="controlled-native",identifier="fixture/gguf",
                registry_name="managed-native",task="chat",backend="llama-cpp",device="cpu",
                model_file="model-q4_k_m.gguf")
            plan=client.post("/api/acquisitions/plan",json=request.model_dump(mode="json")).json()
            assert plan["model_spec"]["native"]["model_file"]=="model-q4_k_m.gguf"
            job=client.post("/api/acquisitions",json=plan).json()
            for _ in range(200):
                job=client.get("/api/acquisitions/"+job["id"]).json()
                if job["status"]=="prepared":break
                time.sleep(.01)
            active=client.post("/api/acquisitions/"+job["id"]+"/activate",json={"load":False})
            assert active.status_code==200,active.text
            assert active.json()["activation"]["status"]=="registered"
            model=read_config(registry).models["managed-native"]
            assert model.backend=="llama-cpp" and model.device=="cpu"
            assert model.native.model_file=="model-q4_k_m.gguf" and model.native.executable is None
    print("PASS: exact GGUF acquisition registers CPU NativeWorker options without a client executable path or model call.")
