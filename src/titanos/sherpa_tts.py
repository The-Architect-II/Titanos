"""Safe local Sherpa ONNX VITS/Piper speech adapter."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import io
from pathlib import Path
import wave

from .engine import EngineError
from .settings import current_settings


def _one(root, names, pattern=None):
    matches = []
    for name in names:
        matches.extend(root.rglob(name))
    if pattern:
        matches.extend(root.rglob(pattern))
    matches = sorted({path.resolve() for path in matches if path.is_file()})
    if len(matches) != 1 or not matches[0].is_relative_to(root):
        raise ValueError(f"Sherpa model needs exactly one safe {names[0] if names else pattern}")
    return matches[0]


def inspect_model(directory):
    root = Path(directory).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Sherpa model path must be an ordinary local directory")
    model = _one(root, ["model.onnx"], "*.onnx")
    tokens = _one(root, ["tokens.txt"])
    data_dirs = [path.resolve() for path in root.rglob("espeak-ng-data") if path.is_dir()]
    if len(data_dirs) > 1 or any(not path.is_relative_to(root) or path.is_symlink() for path in data_dirs):
        raise ValueError("Sherpa archive contains ambiguous or linked espeak-ng-data")
    lexicons = [path.resolve() for path in root.rglob("lexicon.txt") if path.is_file()]
    if len(lexicons) > 1:
        raise ValueError("Sherpa archive contains ambiguous lexicons")
    files = []
    for path in (model, tokens, *data_dirs, *lexicons):
        if path.is_file():
            files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"root": root, "model": model, "tokens": tokens,
            "data_dir": data_dirs[0] if data_dirs else None,
            "lexicon": lexicons[0] if lexicons else None, "files": files}


class SherpaTTSModel:
    def __init__(self, spec):
        if spec.backend != "sherpa-onnx" or spec.task != "speech":
            raise ValueError("SherpaTTSModel requires backend=sherpa-onnx and task=speech")
        layout = inspect_model(spec.path)
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ValueError("Install the packaged sherpa-onnx runtime for this speech model") from exc
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(layout["model"]), tokens=str(layout["tokens"]),
            lexicon=str(layout["lexicon"] or ""),
            data_dir=str(layout["data_dir"] or ""),
        )
        model = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits, num_threads=current_settings().threads, debug=False, provider="cpu")
        config = sherpa_onnx.OfflineTtsConfig(model=model)
        if not config.validate():
            raise ValueError("Sherpa ONNX rejected the acquired model component set")
        self.spec, self.device, self.layout = spec, "cpu", layout
        self.runtime = sherpa_onnx.OfflineTts(config)
        self.runtime_versions={"sherpa-onnx":version("sherpa-onnx"),
                               "sherpa-onnx-core":version("sherpa-onnx-core")}
        if self.runtime_versions!={"sherpa-onnx":"1.13.8","sherpa-onnx-core":"1.13.8"}:
            raise ValueError("Sherpa runtime must match the packaged 1.13.8 Python/native contract")
        self.last_generation = None

    def infer(self, task, data, emit=None, cancel=None):
        if task != "speech":
            raise ValueError("Sherpa ONNX implements speech only")
        if cancel is not None and cancel.is_set():
            raise EngineError("Speech synthesis cancelled", 499, "cancelled")
        text = data.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Speech input must be nonblank")
        voice = data.get("voice", self.spec.defaults.get("voice", "default"))
        speaker = self.spec.defaults.get("speaker_id", 0)
        if voice not in {None, "", "default", "alloy"}:
            try:
                speaker = int(voice)
            except (TypeError, ValueError) as exc:
                raise ValueError("Sherpa voice must be default or a numeric speaker ID") from exc
        observed_progress=0.0
        def callback(samples,progress):
            nonlocal observed_progress
            observed_progress=float(progress)
            return 1 if cancel is not None and cancel.is_set() else 0
        audio = self.runtime.generate(text=text, sid=speaker, speed=float(data.get("speed") or 1.0), callback=callback)
        if cancel is not None and cancel.is_set():
            raise EngineError("Speech synthesis cancelled after native return", 499, "cancelled")
        import numpy as np
        samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
        rate = int(audio.sample_rate)
        settings = current_settings()
        if (not len(samples) or len(samples) > settings.max_audio_samples
                or rate < 8000 or rate > 192000 or not np.isfinite(samples).all()):
            raise EngineError("Sherpa ONNX returned invalid or oversized audio", 502, "invalid_backend_response")
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        output = io.BytesIO()
        with wave.open(output, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(pcm)
        self.last_generation = {"speaker_id": speaker, "sample_rate": rate,
                                "samples": len(samples), "seconds": len(samples) / rate,
                                "native_progress":observed_progress}
        return output.getvalue()

    def details(self):
        return {
            "backend": "sherpa-onnx", "device": self.device, "task": "speech",
            "model_files": self.layout["files"], "repository": self.spec.source_repo,
            "revision": self.spec.source_revision, "last_generation": self.last_generation,
            "runtime_versions":self.runtime_versions,
            "qualification": None,
            "qualification_note": "Acquired model/runtime compatibility is separate from voice quality qualification",
        }

    def close(self):
        self.runtime = None
