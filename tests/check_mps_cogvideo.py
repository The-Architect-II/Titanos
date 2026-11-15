"""CogVideoX MPS positional compatibility: CPU parity plus optional real video.

python tests/check_mps_cogvideo.py [--model /pinned/video/checkpoint --report FILE]
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.models.embeddings import CogVideoXPatchEmbed

from titanos.backends import LoadedModel, _prepare_mps_cogvideo
from titanos.config import ModelSpec


def check(model=None):
    original = CogVideoXPatchEmbed._get_positional_embeddings
    options = dict(patch_size=2, in_channels=4, embed_dim=32, text_embed_dim=32,
                   sample_width=16, sample_height=16, sample_frames=9, max_text_seq_length=16)
    patched, untouched = CogVideoXPatchEmbed(**options), CogVideoXPatchEmbed(**options)
    pipeline = SimpleNamespace(components={"transformer": torch.nn.ModuleList([patched]), "scheduler": object()})
    expected = original(patched, 8, 16, 5, device="cpu").float()
    assert _prepare_mps_cogvideo(pipeline) == [patched]
    assert patched.pos_embedding.dtype == torch.float32
    assert untouched.pos_embedding.dtype == torch.float64
    actual = patched._get_positional_embeddings(8, 16, 5, device="cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert patched._titanos_position_calls == 1
    assert CogVideoXPatchEmbed._get_positional_embeddings is original
    assert untouched._get_positional_embeddings.__func__ is original
    result = {"cpu_reference_parity": True, "instance_only": True,
              "scope": "CogVideoX cached/dynamic positional compatibility; not every MPS model"}
    if model:
        if not torch.backends.mps.is_available():
            raise RuntimeError("The requested native check requires MPS; CPU is not a substitute")
        runtime = LoadedModel(ModelSpec(path=str(model), task="video", backend="diffusers", device="mps", dtype="float32",
                                       defaults={"max_sequence_length": 16}))
        devices = {str(parameter.device) for component in runtime.model.components.values()
                   if isinstance(component, torch.nn.Module) for parameter in component.parameters()}
        assert devices and all(device.startswith("mps") for device in devices), devices
        frames = runtime.infer("video", {"prompt": "A red cube", "steps": 2, "width": 32, "height": 32, "frames": 5, "seed": 1})
        assert len(frames) == 5
        details = runtime.details()
        assert details["compatibility"] and details["compatibility"][0]["dynamic_calls"] > 0
        result.update(native_mps=True, model_parameter_devices=sorted(devices), frames=len(frames), runtime=details,
                      positional_computation="CPU float64 sinusoidal math cast to float32 and transferred to MPS; model stays on MPS")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check(args.model)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
