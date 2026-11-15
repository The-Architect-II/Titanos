"""Explicitly fetch pinned acceptance models. Weights are never bundled.

Lab tool for the pinned check fixtures; it is not reachable from the titanos CLI or server. It fetches through the same
explicit, validated plan as `titanos pull` (titanos.hub): the pinned revision's sibling list is resolved, the excluded
weight formats and directories are dropped by name, every kept name passes the shared Windows-safe acquisition rule,
literal-escaped allow_patterns are handed to the downloader, and an unplanned downloaded file is refused."""

import argparse
import fnmatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import HfApi  # noqa: E402

from titanos import hub  # noqa: E402

MODELS = {
    "chat": ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
    "vision": ("HuggingFaceTB/SmolVLM-256M-Instruct", "7e3e67edbbed1bf9888184d9df282b700a323964"),
    "embeddings": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "speech": ("facebook/mms-tts-eng", "c71de0fe7204c83f1c10820a7d696d0b450048ba"),
    "transcription": ("openai/whisper-tiny.en", "87c7102498dcde7456f24cfd30239ca606ed9063"),
    "image": ("hf-internal-testing/tiny-stable-diffusion-pipe", "3ee6c9f225f088ad5d35b624b6514b091e6a4849"),
    "video": ("hf-internal-testing/tiny-cogvideox-pipe", "2b019731f25da96b76c262904d2ccd2f29975053"),
}
# Excluded by name (matched on the lowercased name, no host-dependent folding); pickle formats stay for the image fixture.
EXCLUDED_PATTERNS = ("*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite", "onnx/*", "openvino/*")
PICKLE_PATTERNS = ("*.bin", "*.pt", "*.pth", "*.ckpt")


def select(files, task):
    """The explicit, validated plan for one fixture: every sibling except the excluded formats (hub.plan_selection)."""
    ignored = EXCLUDED_PATTERNS if task == "image" else EXCLUDED_PATTERNS + PICKLE_PATTERNS
    kept = [name for name in files if not any(fnmatch.fnmatchcase(name.lower(), pattern) for pattern in ignored)]
    return hub.plan_selection(files, kept)


def fetch(api, directory, task, repo, revision):
    with hub.hub_errors():
        info = api.model_info(repo, revision=revision)
        if not info.sha:
            raise ValueError("Hub did not resolve an immutable model revision")
        plan = select(hub.sibling_names(info), task)
        return hub.fetch_snapshot(repo, info.sha, None, Path(directory) / task, plan)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    api = HfApi()
    for task, (repo, revision) in MODELS.items():
        print(f"Downloading {task}: {repo}@{revision}", flush=True)
        fetch(api, args.directory, task, repo, revision)
