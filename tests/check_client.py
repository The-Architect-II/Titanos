"""Client/CLI protocol checks using synthetic HTTP responses, never model proof.

Run: python tests/check_client.py
"""

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import httpx

from titanos.client import Client, TitanosError
from titanos.cli import main


def raises(kind, call):
    try:
        call()
    except kind as exc:
        return exc
    raise AssertionError(f"Expected {kind.__name__}")


def check():
    seen = []
    identifier = "a" * 32
    output_bytes = b"synthetic asset for transport testing"
    state = {"status": "completed", "redirect": False, "broken": False, "stream": "normal"}

    class BrokenBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("fixture connection lost")

    def respond(request):
        seen.append(request)
        assert request.headers.get("authorization") == "Bearer fixture-key"
        path = request.url.path
        body = json.loads(request.content) if request.headers.get("content-type") == "application/json" else {}
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fixture", "task": "chat"}]})
        if path == "/api/info":
            return httpx.Response(200, json={"engine": "titanos"})
        if path == "/v1/chat/completions":
            assert body["model"] == "fixture" and body["messages"]
            if body.get("stream"):
                if state["stream"] == "error":
                    events = [{"error": {"message": "fixture inference error", "code": "inference_error"}}]
                elif state["stream"] == "tool":
                    events = [
                        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":'}}]}}]},
                        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"test"}'}}]}, "finish_reason": "tool_calls"}]},
                    ]
                else:
                    events = [{"choices": [{"delta": {"content": text}}]} for text in ("Hello", " world")]
                    events.append({"choices": [], "usage": {"completion_tokens": 2}})
                content = ": keepalive\n\n" + "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                if state["stream"] != "truncated":
                    content += "data: [DONE]\n\n"
                return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "fixture answer"}}]})
        if path == "/v1/embeddings":
            assert body["input"]
            return httpx.Response(200, json={"data": [{"embedding": [0.5, -0.5]}]})
        if path in {"/v1/audio/transcriptions", "/v1/audio/translations"}:
            assert b'name="model"' in request.content and b'name="file"' in request.content
            assert b"fixture audio" in request.content
            if b"\r\ntext\r\n" in request.content:
                return httpx.Response(200, text="fixture transcription")
            return httpx.Response(200, json={"text": "fixture transcription"})
        if path == "/v1/translations":
            assert body["text"] == "hello" and body["target_language"] == "fra_Latn"
            return httpx.Response(200, json={"text": "fixture translation"})
        if path in {"/v1/images/ocr", "/v1/images/detect"}:
            assert body["image_data_url"].startswith("data:image/png;base64,")
            return httpx.Response(200, json={"text": "fixture OCR"} if path.endswith("ocr") else {"detections": []})
        if path == "/v1/extensions/infer":
            assert body["operation"] == "translation" and body["model"] == "fixture"
            return httpx.Response(200, json={"text": "fixture extension"})
        if path == "/v1/audio/speech":
            assert body["input"] == "hello"
            return httpx.Response(200, content=output_bytes, headers={"content-type": "audio/wav"})
        if path in {"/v1/images/generations", "/v1/images/edits"}:
            if path.endswith("edits"):
                assert b'name="image"' in request.content and b"fixture image" in request.content
                assert b'name="strength"' in request.content
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(output_bytes).decode()}] * body.get("n", 1)})
        if path == "/v1/videos/generations":
            assert body["prompt"] == "clip"
            return httpx.Response(200, json={"url": "/v1/files/" + identifier + ".mp4"})
        if path in {"/api/jobs", "/api/video/jobs"}:
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": identifier, "status": state["status"]}]})
            assert request.headers.get("idempotency-key") == "fixture-job"
            return httpx.Response(202, json={"id": identifier, "status": "queued"})
        if path == "/api/jobs/" + identifier:
            status = "cancelling" if request.method == "DELETE" else state["status"]
            return httpx.Response(200, json={"id": identifier, "status": status,
                                            "result": {"url": "/v1/files/" + identifier + ".mp4"}})
        if path.startswith("/v1/files/"):
            if state["redirect"]:
                return httpx.Response(302, headers={"location": "https://elsewhere.invalid/asset"})
            if state["broken"]:
                return httpx.Response(200, stream=BrokenBody())
            return httpx.Response(200, content=output_bytes)
        return httpx.Response(409, json={"error": {"message": "fixture conflict", "code": "fixture_conflict"}})

    transport = httpx.MockTransport(respond)
    with tempfile.TemporaryDirectory() as tmp, Client(api_key="fixture-key", transport=transport) as client:
        root = Path(tmp)
        sound, picture = root / "input.wav", root / "input.png"
        sound.write_bytes(b"fixture audio")
        picture.write_bytes(b"fixture image")
        assert client.models()["data"][0]["id"] == "fixture"
        assert client.info()["engine"] == "titanos"
        prompt = [{"role": "user", "content": "hello"}]
        assert client.chat("fixture", prompt)["choices"][0]["message"]["content"] == "fixture answer"
        events = list(client.chat("fixture", prompt, stream=True))
        assert events[-1]["usage"]["completion_tokens"] == 2
        assert "".join(e["choices"][0]["delta"]["content"] for e in events[:-1]) == "Hello world"
        for status in ("error", "truncated"):
            state["stream"] = status
            raises(TitanosError, lambda: list(client.chat("fixture", prompt, stream=True)))
        state["stream"] = "normal"
        assert client.embeddings("fixture", ["a", "b"])["data"][0]["embedding"] == [0.5, -0.5]
        assert client.transcribe("fixture", sound, worker="a")["text"] == "fixture transcription"
        assert client.transcribe("fixture", sound, response_format="text") == "fixture transcription"
        assert client.translate_audio("fixture", sound)["text"] == "fixture transcription"
        assert client.translate("fixture", "hello", "fra_Latn")["text"] == "fixture translation"
        assert client.ocr("fixture", picture)["text"] == "fixture OCR"
        assert client.detect("fixture", picture, threshold=0.5)["detections"] == []
        assert client.extension("translation", "fixture", text="hello", target_language="fra_Latn")["text"] == "fixture extension"
        assert client.speech("fixture", "hello") == output_bytes
        asset = client.image("fixture", "picture")["data"][0]
        assert client.save_asset(asset, root / "image.png").read_bytes() == output_bytes
        assert client.edit("fixture", picture, "edit", strength=0.5)["data"]
        video = client.video("fixture", "clip")
        assert client.save_asset(video, root / "clip.mp4").read_bytes() == output_bytes
        client.submit_video("fixture", "clip", idempotency_key="fixture-job")
        client.submit_job("speech", "fixture", input="hello", idempotency_key="fixture-job")
        assert client.jobs()["data"][0]["id"] == identifier
        assert client.wait_job(identifier, timeout=1)["status"] == "completed"
        state["status"] = "failed"
        assert client.wait_job(identifier, timeout=1)["status"] == "failed"
        state["status"] = "running"
        raises(TimeoutError, lambda: client.wait_job(identifier, timeout=0.01, poll_interval=0.01))
        assert not any(r.method == "DELETE" for r in seen), "A wait timeout must not cancel work"
        assert client.cancel_job(identifier)["status"] == "cancelling"
        state["status"] = "completed"
        for bad in (0, -1, float("inf"), float("nan")):
            raises(ValueError, lambda: client.wait_job(identifier, timeout=bad))
        error = raises(TitanosError, lambda: client.details("missing"))
        assert error.status_code == 409 and error.code == "fixture_conflict"
        count = len(seen)
        raises(ValueError, lambda: client.save_asset({"url": "https://elsewhere.invalid/asset"}, root / "keep.bin"))
        assert len(seen) == count, "A foreign asset must not receive a request or token"
        keep = root / "keep.bin"
        keep.write_bytes(b"keep me")
        for field in ("redirect", "broken"):
            state[field] = True
            raises(TitanosError, lambda: client.save_asset(video, keep))
            assert keep.read_bytes() == b"keep me"
            state[field] = False
        assert not list(root.glob(".titanos-*")), "Failed downloads left temporary output behind"

        def invoke(argv, stdin="", expect=0):
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch("sys.stdin", io.StringIO(stdin)):
                try:
                    main(argv)
                except SystemExit as exc:
                    assert exc.code == expect, errors.getvalue()
                else:
                    assert expect == 0, output.getvalue()
            return output.getvalue()

        def cli_client(*args, **kwargs):
            return Client(*args, **kwargs, api_key="fixture-key", transport=transport)

        with patch("titanos.client.Client", side_effect=cli_client):
            history = root / "history.json"
            invoke(["run", "fixture", "first", "--history", str(history)])
            assert len(json.loads(history.read_text())) == 2
            invoke(["run", "fixture", "second", "--history", str(history), "--json"])
            assert len(json.loads(history.read_text())) == 4
            chat_body = json.loads(seen[-1].content)
            assert chat_body["messages"][1]["content"] == "fixture answer", "History was not sent back"
            chunks = invoke(["run", "fixture", "-", "--stream", "--json"], "piped prompt")
            assert len(chunks.splitlines()) == 3 and json.loads(chunks.splitlines()[-1])["usage"]
            invoke(["run", "fixture", "see", "--image", str(picture)])
            assert json.loads(seen[-1].content)["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
            state["stream"] = "tool"
            invoke(["run", "fixture", "tool", "--stream", "--history", str(history)])
            call = json.loads(history.read_text())[-1]["tool_calls"][0]
            assert call["id"] == "call_1" and call["function"]["arguments"] == '{"q":"test"}'
            before = history.read_bytes()
            state["stream"] = "truncated"
            invoke(["run", "fixture", "fail", "--stream", "--history", str(history)], expect=1)
            assert history.read_bytes() == before, "Incomplete response overwrote conversation history"
            state["stream"] = "normal"
            invoke(["models"])
            invoke(["info"])
            invoke(["embed", "fixture", "a", "b"])
            invoke(["transcribe", "fixture", str(sound)])
            invoke(["translate-audio", "fixture", str(sound)])
            invoke(["translate", "fixture", "hello", "--target-language", "fra_Latn"])
            invoke(["ocr", "fixture", str(picture)])
            invoke(["detect", "fixture", str(picture), "--threshold", "0.5"])
            invoke(["speak", "fixture", "hello", "--output", str(root / "speech.wav")])
            invoke(["image", "fixture", "picture", "--n", "2", "--output", str(root / "two.png")])
            assert (root / "two-1.png").read_bytes() == output_bytes
            assert (root / "two-2.png").read_bytes() == output_bytes
            invoke(["edit", "fixture", "edit", "--image", str(picture), "--strength", "0.5"])
            invoke(["video", "fixture", "clip", "--idempotency-key", "fixture-job", "--output", str(root / "out.mp4")])
            assert (root / "out.mp4").read_bytes() == output_bytes
            invoke(["jobs", "list"])
            invoke(["jobs", "get", identifier])
            invoke(["jobs", "wait", identifier, "--output", str(root / "job.mp4")])
            request = root / "request.json"
            request.write_text(json.dumps({"task": "image", "model": "fixture", "prompt": "image"}))
            invoke(["jobs", "submit", str(request), "--idempotency-key", "fixture-job"])
            state["status"] = "failed"
            result = invoke(["jobs", "wait", identifier], expect=1)
            assert json.loads(result)["status"] == "failed", "Agent did not receive structured failed-job status"
            invoke(["jobs", "cancel", identifier])
    assert not any(r.url.host == "elsewhere.invalid" for r in seen)
    print("PASS: client/CLI multimodal requests, SSE/tools/history, jobs/deadlines/errors, same-origin authenticated atomic downloads (synthetic protocol fixtures; no model inference)")


if __name__ == "__main__":
    check()
