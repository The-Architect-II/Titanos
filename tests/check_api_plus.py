"""Extended inference contracts, with optional real local checkpoint execution.

python tests/check_api_plus.py --models DIR --detection-image FILE --device cpu --report FILE
Model directory layout: t5-small/, trocr-small-printed/, yolos-tiny/. No download occurs.
Engine contracts determine pass/fail. Model accuracy observations are reported
separately and never turn a valid inference response into an engine failure.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import tempfile
import time

from fastapi.testclient import TestClient

from titanos.config import Config, ModelSpec, write_config


def text_observation(body, seconds, criterion, matched):
    assert isinstance(body.get('text'), str) and body['text'].strip(), body
    return {'response': body, 'seconds': seconds, 'engine_status': 'passed',
            'model_quality': {'criterion': criterion, 'status': 'passed' if matched else 'failed'}}


def contracts():
    config = Config(models={"missing": ModelSpec(path="./absent", task="translation")})
    from titanos.server import create_app
    with tempfile.TemporaryDirectory(prefix="titanos-api-plus-") as tmp, TestClient(create_app(config, output_dir=Path(tmp)), base_url="http://127.0.0.1") as client:
        discovery = client.get("/v1/extensions").json()
        assert discovery["data"][0]["models"] == ["missing"]
        assert discovery["schema"]["discriminator"]["propertyName"] == "operation"
        assert client.post("/v1/translations", json={"model": "unknown", "text": "hello", "target_language": "fra_Latn"}).status_code == 404
        assert client.post("/v1/translations", json={"model": "missing", "text": "hello", "target_language": "fra_Latn"}).status_code == 503
        invalid = client.post("/v1/extensions/infer", json={"operation": "invented-task", "model": "missing"})
        assert invalid.status_code == 422
        duplicate = client.post("/v1/images/ocr", json={"model": "missing", "image_base64": "AA==", "image": "AA=="})
        assert duplicate.status_code == 422
        invalid_image = client.post("/v1/images/ocr", json={"model": "missing", "image_base64": "not-base64"})
        assert invalid_image.status_code == 400
        untyped_bytes = client.post("/v1/images/ocr", json={"model": "missing", "image_buffer": [[True, 1]]})
        assert untyped_bytes.status_code == 422
        not_english = client.post("/v1/extensions/infer", json={"operation": "audio_translation", "model": "missing", "audio_base64": "AA==", "target_language": "fra_Latn"})
        assert not_english.status_code == 422
        no_pair = client.post("/v1/extensions/infer", json={"operation": "audio_translation", "model": "missing", "audio_base64": "AA==", "translation_model": "missing"})
        assert no_pair.status_code == 422
    from titanos.settings import Settings
    settings = Settings(max_tokens=12000, default_max_tokens=9000, max_text_chars=5)
    with tempfile.TemporaryDirectory(prefix="titanos-api-limits-") as tmp, TestClient(create_app(Config(), settings=settings, output_dir=Path(tmp)), base_url="http://127.0.0.1") as client:
        payload = {"model": "unknown", "text": "hello", "target_language": "fra_Latn", "max_tokens": 10000}
        assert client.post("/v1/translations", json=payload).status_code == 404  # Raised app cap passes validation.
        assert client.post("/v1/translations", json={**payload, "max_tokens": 12001}).status_code == 422
        assert client.post("/v1/translations", json={**payload, "text": "longer"}).status_code == 422
    with tempfile.TemporaryDirectory(prefix="titanos-api-reload-") as tmp:
        path = Path(tmp)/"models.json"
        cfg = Config()
        write_config(path, cfg)
        checkpoint = Path(tmp)/"checkpoint"
        checkpoint.mkdir()
        with TestClient(create_app(cfg, registry_path=path, output_dir=Path(tmp)/"assets"), base_url="http://127.0.0.1") as client:
            response = client.post("/api/models/register", json={"name": "new-translation", "spec": {"path": str(checkpoint), "task": "translation"}})
            assert response.status_code == 200, response.text
            assert "new-translation" in client.get("/v1/extensions").json()["data"][0]["models"]
            response = client.post("/v1/translations", json={"model": "new-translation", "text": "hello", "target_language": "fra_Latn"})
            assert response.status_code == 503, response.text  # New model is recognized; weights are deliberately absent.
    return ["discovery, unknown/missing models, unsupported operations, image ambiguity/type checks, English translation and explicit NMT language binding", "app-level raised token caps and lowered text caps, registry reload reaches extension engine"]


def live(models, detection_image, device):
    from PIL import Image, ImageDraw, ImageFont
    defaults = {"source_language": "eng_Latn", "target_language": "fra_Latn",
                "translation_prefix": "translate English to French: ", "max_tokens": 40, "num_beams": 1}
    config = Config(models={
        "translate": ModelSpec(path=str(models/"t5-small"), task="translation", device=device, dtype="float32", defaults=defaults),
        "ocr": ModelSpec(path=str(models/"trocr-small-printed"), task="image-to-text", device=device, dtype="float32", defaults={"max_tokens": 40}),
        "detect": ModelSpec(path=str(models/"yolos-tiny"), task="object-detection", device=device, dtype="float32"),
    })
    if (models/"whisper-base").is_dir() and (models/"mms-tts-fra").is_dir():
        config.models["french-speech"] = ModelSpec(path=str(models/"mms-tts-fra"), task="speech", device=device, dtype="float32")
        config.models["audio-translate"] = ModelSpec(path=str(models/"whisper-base"), task="transcription", device=device, dtype="float32")
    image = Image.new("RGB", (640, 96), "white")
    draw = ImageDraw.Draw(image)
    font = None
    for candidate in ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 54)
            break
    draw.text((25, 14), "HELLO WORLD", fill="black", font=font or ImageFont.load_default(size=54))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    observations = {}
    from titanos.server import create_app
    with tempfile.TemporaryDirectory(prefix="titanos-api-plus-") as tmp, TestClient(create_app(config, output_dir=Path(tmp)), base_url="http://127.0.0.1") as client:
        start = time.monotonic()
        response = client.post("/v1/translations", json={"model": "translate", "text": "The house is wonderful.", "target_language": "fra_Latn"})
        assert response.status_code == 200, response.text
        body = response.json()
        observations["translation"] = text_observation(body, time.monotonic()-start,
            'French translation contains maison', 'maison' in body.get('text', '').lower())
        observations["translation"]["runtime"] = client.app.state.engine.details("translate")
        unsupported_pair = client.post("/v1/translations", json={"model": "translate", "text": "Hello", "target_language": "deu_Latn"})
        assert unsupported_pair.status_code == 400, unsupported_pair.text
        start = time.monotonic()
        response = client.post("/v1/extensions/infer", json={"operation": "ocr", "model": "ocr", "image_data_url": data})
        assert response.status_code == 200, response.text
        body = response.json()
        observations["ocr"] = text_observation(body, time.monotonic()-start,
            'OCR recovers HELLO and WORLD', all(word in body.get('text', '').upper() for word in ('HELLO', 'WORLD')))
        observations["ocr"]["fixture_text"] = "HELLO WORLD"
        observations["ocr"]["runtime"] = client.app.state.engine.details("ocr")
        if detection_image:
            start = time.monotonic()
            response = client.post("/v1/images/detect", json={"model": "detect", "image_base64": base64.b64encode(detection_image.read_bytes()).decode(), "threshold": 0.5})
            assert response.status_code == 200, response.text
            body = response.json()
            assert isinstance(body.get('objects'), list), body
            assert all(0 <= o["score"] <= 1 and o["box"]["xmax"] >= o["box"]["xmin"] for o in body["objects"])
            observations["detection"] = {"response": body, "seconds": time.monotonic()-start,
                'engine_status': 'passed', 'model_quality': {'criterion': 'A cat is detected in the fixture',
                'status': 'passed' if any(o['label'] == 'cat' for o in body['objects']) else 'failed'}}
            observations["detection"]["runtime"] = client.app.state.engine.details("detect")
        if "audio-translate" in config.models:
            phrase = "Le chat est sur la table."
            speech = client.post("/v1/audio/speech", json={"model": "french-speech", "input": phrase})
            assert speech.status_code == 200, speech.text
            start = time.monotonic()
            response = client.post("/v1/audio/translations", files={"file": ("french.wav", speech.content, "audio/wav")}, data={"model": "audio-translate", "language": "fr"})
            assert response.status_code == 200, response.text
            body = response.json()
            observations["audio_translation"] = {**text_observation(body, time.monotonic()-start,
                    'English audio translation recovers cat and table', all(word in body.get('text', '').lower() for word in ('cat', 'table'))),
                    "source_text": phrase, "speech_bytes": len(speech.content),
                    "runtime": client.app.state.engine.details("audio-translate"),
                    "fixture_note": "French speech generated locally by MMS; its CC-BY-NC-4.0 checkpoint is validation-only and not bundled"}
    return observations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path)
    parser.add_argument("--detection-image", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {"acceptance_scope": "engine_contract", "contract_checks": contracts(), "device_requested": args.device, "real_models": {}}
    if args.models:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        report["real_models"] = live(args.models.resolve(), args.detection_image, args.device)
    report["passed"] = True
    report['model_quality_status'] = ('not_run' if not report['real_models'] else
        'failed' if any(row['model_quality']['status'] == 'failed' for row in report['real_models'].values()) else 'passed')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
