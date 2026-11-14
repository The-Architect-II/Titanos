"""Activation cancellation/final-write consistency with controlled artifacts."""
import asyncio
from contextlib import suppress
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

import httpx

from titanos.config import Config,write_config
from titanos.providers import AcquisitionRequest,ModelProvider
from titanos.server import create_app


class Provider(ModelProvider):
    name="controlled";title="Controlled"
    def search(self,*args):return []
    def inspect(self,identifier,revision=None):
        return {"source_url":"https://fixtures.invalid/model","requested_revision":revision or "v1",
                "resolved_revision":"a"*40,"license":"apache-2.0","gated":False}
    def plan(self,request):return self._base_plan(request,self.inspect(request.identifier,request.revision),[{"path":"model.safetensors","size":7}])
    def acquire(self,plan,staging,progress,cancel):
        root=staging/"model";root.mkdir(parents=True);(root/"model.safetensors").write_bytes(b"fixture")
        progress("downloading",7,7);return {"path":str(root),"license":"apache-2.0","license_evidence":[]}


async def wait(client,identifier):
    for _ in range(200):
        record=(await client.get("/api/acquisitions/"+identifier)).json()
        if record["status"]=="prepared":return record
        await asyncio.sleep(.01)
    raise AssertionError("acquisition not prepared")


async def main():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);registry=root/"models.json";write_config(registry,Config())
        app=create_app(Config(),registry_path=registry,output_dir=root/"outputs")
        app.state.providers.providers={"controlled":Provider()}
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app,raise_app_exceptions=False),base_url="http://127.0.0.1") as client:
                request=AcquisitionRequest(provider="controlled",identifier="fixture/model",registry_name="cancelled",
                    task="embeddings",backend="transformers",device="cpu")
                plan=(await client.post("/api/acquisitions/plan",json=request.model_dump(mode="json"))).json()
                job=(await client.post("/api/acquisitions",json=plan)).json();await wait(client,job["id"])
                entered,release=threading.Event(),threading.Event()
                def load(*args):
                    entered.set();assert release.wait(5);return object()
                app.state.engine.load=load
                operation=asyncio.create_task(client.post(f"/api/acquisitions/{job['id']}/activate",json={"load":True}))
                assert await asyncio.to_thread(entered.wait,5)
                operation.cancel();await asyncio.sleep(.02);release.set()
                with suppress(asyncio.CancelledError):await operation
                record=(await client.get(f"/api/acquisitions/{job['id']}")).json()
                models={row["id"] for row in (await client.get("/v1/models")).json()["data"]}
                assert (record["status"]=="active")==("cancelled" in models),(record,models)

                request=request.model_copy(update={"registry_name":"write-failure"})
                plan=(await client.post("/api/acquisitions/plan",json=request.model_dump(mode="json"))).json()
                job=(await client.post("/api/acquisitions",json=plan)).json();await wait(client,job["id"])
                with patch.object(app.state.acquisitions,"mark_active",side_effect=OSError("fixture write failure")):
                    failed=await client.post(f"/api/acquisitions/{job['id']}/activate",json={"load":False})
                assert failed.status_code==500
                models={row["id"] for row in (await client.get("/v1/models")).json()["data"]}
                assert "write-failure" not in models
                assert (await client.get(f"/api/acquisitions/{job['id']}")).json()["status"]=="prepared"
    print("PASS: cancelled activation settles coherently; final-record failure rolls registry back.")


if __name__=="__main__":asyncio.run(main())
