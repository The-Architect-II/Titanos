"""Client, CLI and MCP contract parity with controlled acquisition metadata."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from titanos.cli import main
from titanos.client import Client
from titanos.config import Config, write_config
from titanos.providers import AcquisitionRequest, ModelProvider
from titanos.server import create_app


class Provider(ModelProvider):
    name="controlled";title="Controlled"
    def search(self,query,task,limit=20):return [{"provider":self.name,"id":"controlled/model","name":"Controlled","titanos_task":task}]
    def inspect(self,identifier,revision=None):return {"source_url":"https://fixtures.invalid/model","requested_revision":revision or "v1","resolved_revision":"d"*40,"license":"apache-2.0","gated":False}
    def plan(self,request):return self._base_plan(request,self.inspect(request.identifier,request.revision),[{"path":"model.safetensors","size":7}])
    def acquire(self,plan,staging,progress,cancel):
        model=staging/"model";model.mkdir(parents=True,exist_ok=True);(model/"model.safetensors").write_bytes(b"fixture");progress("downloading",7,7)
        return {"path":str(model),"license":"apache-2.0","license_evidence":[],"integrity":[],"reused":False}


def client_contract():
    seen=[]
    def handle(request):
        body=json.loads(request.content) if request.content else None;seen.append((request.method,request.url.path,body))
        replies={
            "/api/model-providers":{"data":[{"id":"controlled"}]},
            "/api/model-providers/controlled/search":{"data":[{"id":"controlled/model"}]},
            "/api/model-providers/inspect":{"id":"controlled/model"},
            "/api/model-providers/import":{"imported":{"compatibility":"metadata-only"}},
            "/api/acquisitions/plan":{"provider":"controlled","schema_version":1},
            "/api/acquisitions":{"id":"a"*32,"status":"queued","data":[]},
            "/api/acquisitions/"+("a"*32):{"id":"a"*32,"status":"prepared"},
            "/api/acquisitions/"+("a"*32)+"/retry":{"id":"a"*32,"status":"queued"},
            "/api/acquisitions/"+("a"*32)+"/activate":{"id":"a"*32,"status":"active"},
        }
        return httpx.Response(200 if request.method!="POST" or request.url.path!="/api/acquisitions" else 202,json=replies[request.url.path])
    with Client(transport=httpx.MockTransport(handle)) as client:
        assert client.providers()["data"][0]["id"]=="controlled"
        assert client.discover_models("controlled","embeddings")["data"][0]["id"]=="controlled/model"
        client.inspect_model_source("controlled","controlled/model","v1");client.import_model_link("https://example.org/model")
        client.plan_acquisition(provider="controlled",identifier="controlled/model",registry_name="x",task="embeddings",backend="transformers")
        job=client.submit_acquisition({"schema_version":1,"provider":"controlled"},idempotency_key="key")
        assert client.acquisition(job["id"])["status"]=="prepared"
        client.cancel_acquisition(job["id"]);client.retry_acquisition(job["id"]);client.activate_acquisition(job["id"],load=False)
    assert any(path=="/api/acquisitions" for _,path,_ in seen)


def cli_contract():
    calls=[]
    class Stub:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def discover_models(self,*args):calls.append(("discover",args));return {"data":[]}
        def plan_acquisition(self,**kwargs):calls.append(("plan",kwargs));return {"schema_version":1,"provider":"controlled"}
        def submit_acquisition(self,plan,**kwargs):calls.append(("submit",plan,kwargs));return {"id":"a"*32,"status":"queued"}
        def wait_acquisition(self,identifier,**kwargs):return {"id":identifier,"status":"prepared"}
        def activate_acquisition(self,identifier,**kwargs):calls.append(("activate",identifier,kwargs));return {"id":identifier,"status":"active"}
        def acquisitions(self):return {"data":[]}
    output=io.StringIO()
    with patch("titanos.client.Client",return_value=Stub()),redirect_stdout(output):
        main(["discover","controlled","--task","embeddings"])
        main(["acquire","embed","controlled","controlled/model","--task","embeddings","--backend","transformers","--activate","--no-load"])
        main(["acquisitions","list"])
    assert any(row[0]=="activate" and row[2]["load"] is False for row in calls)


def mcp_contract():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);registry=root/"models.json";write_config(registry,Config())
        app=create_app(Config(),registry_path=registry,output_dir=root/"out")
        app.state.providers.providers={"controlled":Provider()}
        async def checks():
            listed=await app.state.mcp.call_tool("list_model_providers",{})
            assert not listed.is_error
            payload=AcquisitionRequest(provider="controlled",identifier="controlled/model",revision="v1",registry_name="embed",task="embeddings",backend="transformers",device="cpu")
            planned=await app.state.mcp.call_tool("plan_model_acquisition",{"payload":payload.model_dump(mode="json")})
            assert not planned.is_error
            plan=planned.structured_content;plan=plan.get("result",plan)
            submitted=await app.state.mcp.call_tool("submit_model_acquisition",{"payload":plan})
            assert not submitted.is_error
        with TestClient(app,base_url="http://127.0.0.1"):
            asyncio.run(checks())


if __name__=="__main__":
    client_contract();cli_contract();mcp_contract()
    print("PASS: Client, CLI and MCP acquisition parity.")
