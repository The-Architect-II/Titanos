"""Visible-browser Studio fixture with real HTTP routes and fake model mechanics.

This server is intentionally not a model, native-runtime, quality, or hardware
test. It owns only the requested loopback listener and temporary fixture root.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
from unittest.mock import patch

from PIL import Image
import uvicorn

from titanos.config import Config, ModelSpec, write_config
from titanos.engine import EngineError
from titanos.server import create_app


class FakeModel:
    """Deterministic adapter used only for browser mechanics."""

    def __init__(self, spec):
        self.spec = spec
        self.device = "cpu"
        self.closed = False

    def details(self):
        return {
            "backend": self.spec.backend,
            "device": self.device,
            "runtime": "studio-browser-fake",
            "scope": "UI mechanics only; no trained model or native runtime",
        }

    def close(self):
        self.closed = True

    @staticmethod
    def _cancel_boundary(cancel):
        if cancel is not None and cancel.is_set():
            raise EngineError("Fixture request cancelled", 499, "cancelled")

    def infer(self, task, data, emit=None, cancel=None):
        self._cancel_boundary(cancel)
        if task in {"chat", "vision"}:
            text = "Fixture response from the fake Studio browser server."
            if emit:
                emit("Fixture response ")
                self._cancel_boundary(cancel)
                emit("from the fake Studio browser server.")
            return {
                "text": text,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 4, "completion_tokens": 9, "total_tokens": 13},
            }
        if task == "image":
            if "slow" in data.get("prompt", "").lower():
                for _ in range(100):
                    time.sleep(0.05)
                    self._cancel_boundary(cancel)
            width, height = map(int, data.get("size", "64x64").split("x"))
            return [Image.new("RGB", (width, height), (38, 155, 91))]
        if task == "depth-estimation":
            image = data["image"]
            width, height = image.size
            values = b"".join(struct.pack("<f", float(x + y + 1)) for y in range(height) for x in range(width))
            return {
                "task": task,
                "semantics": "relative_inverse_depth",
                "units": None,
                "larger_values": "nearer",
                "calibrated": False,
                "width": width,
                "height": height,
                "encoding": "base64_float32_le",
                "layout": "row_major_yx",
                "data": base64.b64encode(values).decode(),
                "value_sha256": hashlib.sha256(values).hexdigest(),
                "input_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "model_output_shape": [height, width],
                "alignment": "processor_post_process_to_input_size",
                "normalization": "none",
                "model_class": "StudioFixtureDepth",
                "processor_class": "StudioFixtureProcessor",
                "device": self.device,
            }
        if task == "image-segmentation":
            image = data["image"]
            width, height = image.size
            multimask = bool(data.get("multimask", False))
            threshold = float(data.get("mask_threshold", 0))
            count = 2 if multimask else 1
            masks = []
            for index in range(count):
                mask = Image.new("RGBA", (width, height), (0, 0, 0, 255))
                for y in range(height):
                    for x in range(width // (index + 2)):
                        mask.putpixel((x, y), (0, 0, 0, 0))
                from io import BytesIO

                stream = BytesIO()
                mask.save(stream, format="PNG")
                raw = stream.getvalue()
                mask.close()
                masks.append(
                    {
                        "index": index,
                        "native_mask_token_index": index + int(multimask),
                        "predicted_iou": 0.6 - index * 0.1,
                        "png_base64": base64.b64encode(raw).decode(),
                        "png_sha256": hashlib.sha256(raw).hexdigest(),
                        "selected_pixels": (width // (index + 2)) * height,
                    }
                )
            prompt = {
                key: data[key]
                for key in ("box", "points", "point_labels")
                if data.get(key) is not None
            }
            prompt.update(multimask=multimask, mask_threshold=threshold)
            return {
                "model": "segment-fixture",
                "task": task,
                "mask_contract": "alpha-binary-v1",
                "score_kind": "predicted_iou_uncalibrated",
                "calibrated": False,
                "width": width,
                "height": height,
                "input_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "multimask": multimask,
                "mask_threshold": threshold,
                "prompt": prompt,
                "masks": masks,
                "model_output_shape": [1, 1, count, 4, 4],
                "alignment": "processor_resize_pad_then_post_process_to_original_pixels",
                "model_class": "StudioFixtureSAM",
                "processor_class": "StudioFixtureProcessor",
                "device": self.device,
            }
        raise EngineError(f"Fake browser fixture does not implement {task}", 400, "unsupported_capability")


def fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    models = root / "external-models"
    models.mkdir(exist_ok=True)
    specs = {}
    for name, task, backend in [
        ("chat-fixture", "chat", "transformers"),
        ("image-fixture", "image", "diffusers"),
        ("depth-fixture", "depth-estimation", "transformers"),
        ("segment-fixture", "image-segmentation", "transformers"),
    ]:
        directory = models / name
        directory.mkdir(exist_ok=True)
        (directory / "EXTERNAL_MODEL_MARKER.txt").write_text(
            "Fake Studio browser fixture; this directory must survive registry removal.\n",
            encoding="utf-8",
        )
        specs[name] = ModelSpec(path=str(directory), task=task, backend=backend, device="cpu")
    add_directory = models / "browser-added"
    add_directory.mkdir(exist_ok=True)
    marker = add_directory / "EXTERNAL_MODEL_MARKER.txt"
    marker.write_text("Registration removal must not delete this file.\n", encoding="utf-8")
    registry = root / "models.json"
    config = Config(models=specs)
    write_config(registry, config)
    return config, registry, marker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Owned fixture directory; defaults to a fresh system temporary directory",
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("The fake Studio browser fixture is loopback-only")
    if args.root is None:
        # Retain the printed fixture and cleanup receipt for browser inspection.
        args.root = Path(tempfile.mkdtemp(prefix="titanos-studio-fixture-"))
    config, registry, marker = fixture(args.root.resolve())
    app = create_app(config, output_dir=args.root / "outputs", registry_path=registry)
    stop_file = args.root / "STOP"
    stop_file.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "warning": "FAKE BROWSER FIXTURE ONLY",
                "scope": "Visible UI and real HTTP mechanics; no model/runtime/hardware claim",
                "url": f"http://{args.host}:{args.port}/",
                "registry": str(registry),
                "registration_test_path": str(marker.parent),
                "preservation_marker": str(marker),
                "graceful_stop_file": str(stop_file),
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info"))
        def watch_stop_file():
            while not server.should_exit:
                if stop_file.exists():
                    server.should_exit = True
                    return
                time.sleep(0.2)
        threading.Thread(target=watch_stop_file, name="studio-fixture-stop", daemon=True).start()
        with patch("titanos.backends.LoadedModel", FakeModel):
            server.run()
    finally:
        receipt = {
            "fixture_stopped": True,
            "external_marker_preserved": marker.is_file(),
            "marker": str(marker),
        }
        (args.root / "fixture-cleanup.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        if not marker.is_file():
            raise RuntimeError("Studio browser fixture external checkpoint marker was deleted")


if __name__ == "__main__":
    main()
