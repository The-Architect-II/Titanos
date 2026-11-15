"""Runtime planning/API/Client/MCP/CLI integration; no runtime download or model call."""
import asyncio
from contextlib import redirect_stdout
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import httpx

from titanos.cli import main
from titanos.client import Client
from titanos.config import Config
from titanos.mcp_server import create_mcp
from titanos.server import create_app


ROOT=Path(__file__).resolve().parents[1]


async def app_contract(catalog,temp,profile):
    app=create_app(Config(),api_key="fixture-key",output_dir=Path(temp)/"outputs",runtime_catalog=catalog)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url="http://127.0.0.1") as client:
            denied=await client.post("/api/runtime-setups/plan",json={"profile":profile})
            assert denied.status_code==401
            headers={"Authorization":"Bearer fixture-key"}
            plan=await client.post("/api/runtime-setups/plan",json={"profile":profile},headers=headers)
            assert plan.status_code==200 and plan.json()["requires_user_python"] is False
            info=(await client.get("/api/info",headers=headers)).json()
            assert info["model_setup"]["runtime"]["recommended_profile"]==profile
            profiles=(await client.get("/api/runtime-profiles",headers=headers)).json()
            assert any(row["id"]==profile and row["one_stop_supported"] for row in profiles["data"])
        mcp=create_mcp(api_key="fixture-key",app=app)
        names={tool.name for tool in await mcp.list_tools()}
        assert {"list_runtime_profiles","list_runtime_setups","plan_runtime_setup","prepare_runtime","get_runtime_setup","cancel_runtime_setup","retry_runtime_setup"}<=names
        result=await mcp.call_tool("plan_runtime_setup",{"payload":{"profile":profile}})
        assert not result.is_error
    finally:
        app.state.runtime_setup.close()
        await app.state.native.close()
        app.state.engine.target.close()


async def packaged_default_contract(temp,profile):
    app=create_app(Config(),output_dir=Path(temp)/"default-outputs")
    try:
        assert app.state.runtime_setup is not None
        assert Path(app.state.runtime_setup.catalog).as_posix().endswith("titanos/data/native-profiles.json")
        assert app.state.runtime_profile==profile
    finally:
        app.state.runtime_setup.close();await app.state.native.close();app.state.engine.target.close()


def client_contract():
    def handle(request):
        if request.url.path.endswith("/plan"):return httpx.Response(200,json={"profile":"win-cpu-x64","status":"absent"})
        if request.url.path=="/api/runtime-profiles":return httpx.Response(200,json={"data":[{"id":"win-cpu-x64"}]})
        if request.url.path=="/api/runtime-setups" and request.method=="POST":return httpx.Response(202,json={"id":"1"*32,"status":"queued"})
        if request.url.path=="/api/runtime-setups":return httpx.Response(200,json={"data":[]})
        return httpx.Response(200,json={"id":"1"*32,"status":"prepared"})
    with Client(transport=httpx.MockTransport(handle)) as client:
        assert client.plan_runtime_setup("win-cpu-x64")["status"]=="absent"
        assert client.runtime_profiles()["data"][0]["id"]=="win-cpu-x64"
        job=client.submit_runtime_setup("win-cpu-x64")
        assert client.wait_runtime_setup(job["id"])["status"]=="prepared"
        client.runtime_setups();client.cancel_runtime_setup(job["id"]);client.retry_runtime_setup(job["id"])


def cli_contract():
    calls=[]
    class Stub:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def plan_runtime_setup(self,profile):calls.append(("plan",profile));return {"profile":profile}
        def runtime_profiles(self):calls.append(("profiles",));return {"data":[]}
        def runtime_setups(self):calls.append(("list",));return {"data":[]}
    with patch("titanos.client.Client",return_value=Stub()),redirect_stdout(io.StringIO()):
        main(["runtime-prepare","win-cpu-x64","--plan"])
        main(["runtime-profiles"])
        main(["runtime-setups","list"])
    assert calls==[("plan","win-cpu-x64"),("profiles",),("list",)]


def pinned_release_contract(packaged):
    """b12284 is an additional pinned catalog chosen by path; the packaged default stays b10809."""
    from titanos.native_install import plan
    from titanos.runtime_catalog import packaged_catalog
    default=json.loads(packaged.read_text(encoding="utf-8"))
    assert default["release"]=="b10809" and Path(packaged_catalog()).name=="native-profiles.json"
    pinned_path=packaged.with_name("native-profiles-b12284.json")
    pinned=json.loads(pinned_path.read_text(encoding="utf-8"))
    assert (pinned["version"],pinned["runtime"],pinned["release"],pinned["commit"])==(1,"llama.cpp","b12284","9e9619d030bbe6b584d826a0a6cf06e38251f52b")
    # every b10809 profile whose asset names b12284 still publishes, plus win-cuda-13.5-x64 in the shape of b10809's
    # win-cuda-13.3-x64; the renamed OpenVINO 2026.4.1 and the Windows ARM64 CUDA 13.5 builds are not mirrored
    assert set(default["profiles"])-set(pinned["profiles"])=={"ubuntu-openvino-2026.3.1-x64","win-openvino-2026.3.1-x64","win-cuda-13.3-x64","win-cuda-13.4-arm64"}
    assert set(pinned["profiles"])-set(default["profiles"])=={"win-cuda-13.5-x64"}
    for name,item in pinned["profiles"].items():
        old=default["profiles"].get(name) or default["profiles"]["win-cuda-13.3-x64"]
        assert (item["os"],item["arch"])==(old["os"],old["arch"])
        assert [a["file"] for a in item["assets"]]==[a["file"].replace("b10809","b12284").replace("cuda-13.3","cuda-13.5") for a in old["assets"]]
        for asset in item["assets"]:
            assert asset["url"]=="https://github.com/ggml-org/llama.cpp/releases/download/b12284/"+asset["file"]
            assert len(asset["sha256"])==64 and int(asset["sha256"],16)>=0 and isinstance(asset["size"],int) and asset["size"]>0
    # the five archives downloaded, hash-verified and run on 2026-11-12/13
    digests={a["file"]:a["sha256"] for item in pinned["profiles"].values() for a in item["assets"]}
    assert digests["llama-b12284-bin-win-cuda-12.4-x64.zip"]=="2bf31552cee22652f76a108461a1074548ba84e73294dad1061493875255673f"
    assert digests["cudart-llama-bin-win-cuda-12.4-x64.zip"]=="8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
    assert digests["llama-b12284-bin-win-cpu-x64.zip"]=="6b3d2c7034b0c0bac65aa0524ddf01ba9c2b622ce7e32675976df642627defa9"
    assert digests["llama-b12284-bin-win-cuda-13.5-x64.zip"]=="8da9b5e4533c940837f68ca00f2b2e631bd0a05baf044a85a8f069ad25f8fec5"
    assert digests["cudart-llama-bin-win-cuda-13.5-x64.zip"]=="c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba"
    with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{"TITANOS_HOME":temp}), \
         patch("titanos.native_install.discover",return_value={"os":"windows","arch":"x86_64","nvidia":[]}):
        result=plan(pinned_path,"win-cuda-12.4-x64")
        assert result["release"]=="b12284" and len(result["assets"])==2 and result["status"]=="absent"
        assert Path(result["destination"]).name.startswith("b12284-win-cuda-12.4-x64-")
        result=plan(pinned_path,"win-cuda-13.5-x64")
        assert result["release"]=="b12284" and len(result["assets"])==2 and result["status"]=="absent"
        assert Path(result["destination"]).name.startswith("b12284-win-cuda-13.5-x64-")
        assert Path(plan(packaged,"win-cuda-12.4-x64")["destination"]).name.startswith("b10809-win-cuda-12.4-x64-")


if __name__=="__main__":
    catalog=ROOT/"tools"/"native-profiles.json"
    packaged=ROOT/"src"/"titanos"/"data"/"native-profiles.json"
    assert hashlib.sha256(catalog.read_bytes()).digest()==hashlib.sha256(packaged.read_bytes()).digest()
    from titanos.runtime_catalog import resolve_catalog
    with patch.dict(os.environ,{"TITANOS_NATIVE_CATALOG":str(catalog)}):
        assert Path(resolve_catalog()).resolve()==catalog.resolve()
        assert resolve_catalog(str(packaged))==str(packaged)
    from titanos.hardware import discover
    hardware=discover();profile={("windows","x86_64"):"win-cpu-x64",("windows","aarch64"):"win-cpu-arm64",
        ("linux","x86_64"):"ubuntu-x64",("linux","aarch64"):"ubuntu-arm64",
        ("darwin","x86_64"):"macos-x64",("darwin","aarch64"):"macos-arm64"}[(hardware["os"],hardware["arch"])]
    # This contract covers an explicit CPU host; accelerator defaults have their own fixture checks.
    with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{"TITANOS_HOME":temp}), \
         patch('titanos.runtime_catalog.discover',return_value={'os':hardware['os'],'arch':hardware['arch'],'nvidia':[]}):
        output=io.StringIO()
        with redirect_stdout(output):
            main(["native-install","--profile",profile,"--plan"])
        assert json.loads(output.getvalue())["status"]=="absent"
        asyncio.run(app_contract(catalog,temp,profile))
        asyncio.run(packaged_default_contract(temp,profile))
    client_contract();cli_contract();pinned_release_contract(packaged)
    print("PASS: packaged compatible profiles, durable setup listing, native runtime plan, authenticated HTTP, Client, expanded MCP and CLI, the additional pinned b12284 catalog; no runtime/model execution.")
