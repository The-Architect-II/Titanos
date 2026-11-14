"""Hugging Face acquisition rejects Windows-hostile file names at planning time.

Attack-shaped names are inert fixtures: no network, no model execution, and the downloader is an
in-process mock that records whether it was ever invoked. Finding source: an independent read-only
API security review (providers.py HF acquisition, drive-relative and alternate-data-stream names)."""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import threading
import unittest
from unittest.mock import patch

from huggingface_hub.utils import filter_repo_objects

from titanos.providers import (
    AceStepProvider,
    AcquisitionRequest,
    HuggingFaceProvider,
    ProviderFile,
    check_case_insensitive_collisions,
    literal_pattern,
    unplanned_snapshot_files,
    validate_acquisition_path,
)

REPO = "fixture-org/fixture-model"
SHA = "a" * 40
BENIGN = [("config.json", 100), ("model.safetensors", 200), ("tokenizer/vocab.json", 50)]

# Each malicious name is planned twice: once through the llama-cpp explicit-file route (which selects
# any exact sibling name) and, when _hf_allowed would select it, through the transformers common-file route.
MALICIOUS = [
    "D:config.json",                # drive-relative: would resolve to the current directory of drive D:
    "config.json:Zone.Identifier",  # NTFS alternate data stream
    "NUL",                          # reserved device name
    "nul.txt",                      # reserved device name with extension
    "COM1",                         # reserved serial device
    "weights/../x",                 # parent traversal
    "weights\\x",                   # backslash separator
    "/etc/passwd",                  # absolute POSIX path
    "trailing.",                    # trailing dot is stripped by Win32
    "a b ",                         # trailing space is stripped by Win32
    "con\x01fig.json",              # control character
    "com1 .txt",                    # Win32 trims trailing spaces from the base name before the device check
    "COM1 .txt",
    "nul . txt",
    "D[!a].gguf",                   # fnmatch bracket class: huggingface_hub would select the sibling 'D:.gguf'
    "x{a,b}.json",                  # brace form, refused for future-proofing
    "x\x85y.json",                  # C1 control (NEL)
    "x\u200b.json",                 # zero-width space (Cf)
    "x\u202ejson",                  # right-to-left override (Cf)
    "x\ufeff.json",                 # byte-order mark (Cf)
    "b" * 256,                      # component longer than the 255-character bound
]


def _info(siblings):
    return NS(id=REPO, pipeline_tag="feature-extraction", library_name="transformers", downloads=1, likes=0,
              gated=False, sha=SHA, card_data={"license": "apache-2.0"},
              siblings=[NS(rfilename=name, size=size) for name, size in siblings])


class _Downloader:
    """Inert stand-in for snapshot_download: records calls and materialises the requested files."""

    def __init__(self, sizes, extra=()):
        self.sizes = sizes
        self.extra = list(extra)  # (relative name, bytes) written in addition to the requested files
        self.calls = []

    def __call__(self, repo_id, revision, token, allow_patterns, local_dir, tqdm_class=None):
        self.calls.append({"repo_id": repo_id, "revision": revision, "token": token,
                           "allow_patterns": list(allow_patterns), "local_dir": Path(local_dir)})
        for name in allow_patterns:
            target = Path(local_dir) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x" * self.sizes[name])
        for name, size in self.extra:
            target = Path(local_dir) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"y" * size)
        return str(local_dir)


class Checks(unittest.TestCase):
    def _plan(self, siblings, downloader, backend="transformers", task="embeddings", model_file=None):
        with patch("titanos.providers.HfApi") as api, patch("titanos.providers.get_token", return_value=None), \
                patch("titanos.providers.snapshot_download", downloader):
            api.return_value.model_info.return_value = _info(siblings)
            request = AcquisitionRequest(provider="huggingface", identifier=REPO, revision="main", registry_name="fixture",
                                         task=task, backend=backend, device="cpu", model_file=model_file)
            return HuggingFaceProvider().plan(request)

    def test_malicious_names_fail_planning_before_any_download(self):
        for name in MALICIOUS:
            with self.subTest(name=name, route="llama-cpp explicit file"):
                downloader = _Downloader({})
                with self.assertRaises(ValueError) as caught:
                    self._plan(BENIGN + [(name, 7)], downloader, backend="llama-cpp", task="chat", model_file=name)
                message = str(caught.exception)
                self.assertIn("Acquisition file name", message)
                self.assertNotIn("model.safetensors", message, "error must name the offending component, not the selection")
                self.assertEqual(downloader.calls, [], "downloader must never be reached for " + repr(name))
        for name in ("D:config.json", "nul.txt", "con\x01fig.json", "tokenizer.json:Zone.Identifier"):
            with self.subTest(name=name, route="transformers common file"):
                downloader = _Downloader({})
                with self.assertRaises(ValueError):
                    self._plan(BENIGN + [(name, 7)], downloader)
                self.assertEqual(downloader.calls, [])

    def test_validator_rules_and_bounded_messages(self):
        for bad in ("~1/x", "a//b", "a/./b", "x?.json", "x<y>.bin", "e\u0301.json", "lpt9.bin", "com\u00b9", "x/", "C:/x",
                    "com1 .txt", "COM1 .txt", "nul . txt", "LPT9.tar.gz", "aux.x.y", "nul .", "NUL.",
                    "x\x85y", "x\u200b.json", "x\u202ejson", "x\ufeff.json", "x\u2028y", "x\u2029y", "x\x7f",
                    "D[!a].gguf", "model[!a].gguf", "w[1].gguf", "a]b", "x{a,b}.json", "x}y", "b" * 256, "d/" + "b" * 256 + "/x",
                    "\u00e9" * 200):  # 200 characters but 400 UTF-8 bytes
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_acquisition_path(bad)
        for good in ("config.json", "model.safetensors", "tokenizer/vocab.json", "model-00001-of-00002.safetensors",
                     "onnx/model.onnx_data", "\u00e9.json", "console.txt", "nul2.txt", "a~b", "README.md",
                     "com1x.txt", "nul-2.txt", "conf.json", "a" * 255, "d/" + "a" * 255 + "/x.json", "a b.json", "x(1).json"):
            with self.subTest(good=good):
                self.assertEqual(validate_acquisition_path(good), good)
        long_name = "z" * 300 + "."
        with self.assertRaises(ValueError) as caught:
            validate_acquisition_path(long_name)
        self.assertLess(len(str(caught.exception)), 160, "messages stay bounded for long components")
        for bad in ("D:config.json", "D[!a].gguf", "com1 .txt", "x\u200b.json"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ProviderFile(path=bad, size=1)
        self.assertEqual(literal_pattern("a[b]*?.gguf"), "a[[]b][*][?].gguf")
        self.assertEqual(literal_pattern("config.json"), "config.json")

    def test_glob_metacharacter_bypass_of_the_validator_is_closed(self):
        # llama-cpp explicit file: the bracket name passes _hf_allowed and would make huggingface_hub select 'D:.gguf'.
        downloader = _Downloader({})
        with self.assertRaisesRegex(ValueError, "glob metacharacter"):
            self._plan(BENIGN + [("D[!a].gguf", 7), ("D:.gguf", 7)], downloader, backend="llama-cpp", task="chat", model_file="D[!a].gguf")
        self.assertEqual(downloader.calls, [])
        # transformers route with no user file choice: the bracketed selected name versus the excluded fp16 sibling.
        downloader = _Downloader({})
        with self.assertRaisesRegex(ValueError, "glob metacharacter"):
            self._plan(BENIGN + [("[!_][!_]model.[f]p16.safetensors", 7), ("D:model.fp16.safetensors", 7)], downloader)
        self.assertEqual(downloader.calls, [])
        # huggingface_hub's own filter, fed the escaped pattern, selects only the literal name.
        self.assertEqual(list(filter_repo_objects(["w[!a].gguf", "w:.gguf", "W:.GGUF", "w/.gguf", "wx.gguf"],
                                                  allow_patterns=[literal_pattern("w[!a].gguf")])), ["w[!a].gguf"])
        self.assertEqual(list(filter_repo_objects(["w[!a].gguf", "w:.gguf"], allow_patterns=["w[!a].gguf"])), ["w:.gguf"],
                         "the unescaped pattern really does select the unvalidated sibling (huggingface_hub fnmatch)")
        # The allow_patterns a benign plan sends select exactly plan.files against a sibling list with hostile neighbours.
        downloader = _Downloader(dict(BENIGN))
        plan = self._plan(BENIGN, downloader)
        planned = [row.path for row in plan.files]
        patterns = [literal_pattern(name) for name in planned]
        self.assertEqual(patterns, planned)
        siblings = planned + ["D:.gguf", "D:config.json", "config[!a].json", "model.fp16.safetensors",
                              "tokenizer/vocab.json.bak", "xconfig.json", "config.json/x"]
        self.assertEqual(sorted(filter_repo_objects(siblings, allow_patterns=patterns)), sorted(planned))

    def test_case_insensitive_and_directory_collisions_fail_planning(self):
        downloader = _Downloader({})
        with self.assertRaisesRegex(ValueError, "differ only in case"):
            self._plan(BENIGN + [("Config.json", 100)], downloader)
        self.assertEqual(downloader.calls, [])
        downloader = _Downloader({})
        with self.assertRaisesRegex(ValueError, "also a directory"):
            self._plan(BENIGN + [("tokenizer", 5)], downloader)  # selected as a tokenizer asset, collides with tokenizer/vocab.json
        self.assertEqual(downloader.calls, [])
        check_case_insensitive_collisions(["a.json", "b/a.json", "ab.json", "b/c/a.json"])
        with self.assertRaises(ValueError):
            check_case_insensitive_collisions(["B/a.json", "b/A.json"])
        with self.assertRaises(ValueError):
            check_case_insensitive_collisions(["B", "b/a.json"])

    def test_files_the_plan_did_not_name_fail_verification(self):
        downloader = _Downloader(dict(BENIGN), extra=[("extra.bin", 3), ("tokenizer/merges.txt", 2), (".cache/huggingface/x.lock", 1)])
        plan = self._plan(BENIGN, downloader)
        with tempfile.TemporaryDirectory() as tmp, patch("titanos.providers.snapshot_download", downloader), \
                patch("titanos.providers.get_token", return_value=None):
            with self.assertRaisesRegex(ValueError, "did not name: extra.bin, tokenizer/merges.txt$"):
                HuggingFaceProvider().acquire(plan, Path(tmp), lambda *event: None, threading.Event())
            self.assertEqual(len(downloader.calls), 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".cache" / "sub").mkdir(parents=True)
            (root / ".cache" / "sub" / "lock").write_bytes(b"")
            (root / "a" / "b").mkdir(parents=True)
            (root / "a" / "b" / "c.json").write_bytes(b"{}")
            (root / "d.json").write_bytes(b"{}")
            self.assertEqual(unplanned_snapshot_files(root, {"a/b/c.json", "d.json"}), [])
            self.assertEqual(unplanned_snapshot_files(root, {"d.json"}), ["a/b/c.json"])

    def test_ace_step_acquire_rechecks_persisted_names_before_download(self):
        downloader = _Downloader(dict(BENIGN))
        plan = self._plan(BENIGN, downloader)
        repositories = {role: {"id": "fixture/" + role, "revision": SHA} for role in ("core", "dit", "planner")}
        hostile = plan.model_copy(update={"files": [ProviderFile.model_construct(path="core/D:vae.safetensors", size=1)],
                                          "provider_data": {"variant": "xl-sft", "repositories": repositories}})
        with tempfile.TemporaryDirectory() as tmp, patch("titanos.providers.snapshot_download", downloader), \
                patch("titanos.providers.get_token", return_value=None):
            with self.assertRaisesRegex(ValueError, "':'"):
                AceStepProvider().acquire(hostile, Path(tmp), lambda *event: None, threading.Event())
        self.assertEqual(downloader.calls, [])

    def test_benign_plan_reaches_the_downloader_with_expected_arguments(self):
        downloader = _Downloader(dict(BENIGN))
        plan = self._plan(BENIGN, downloader)
        self.assertEqual([row.path for row in plan.files], ["config.json", "model.safetensors", "tokenizer/vocab.json"])
        self.assertEqual(downloader.calls, [])
        events = []
        with tempfile.TemporaryDirectory() as tmp, patch("titanos.providers.snapshot_download", downloader), \
                patch("titanos.providers.get_token", return_value=None):
            staging = Path(tmp)
            result = HuggingFaceProvider().acquire(plan, staging, lambda *event: events.append(event), threading.Event())
            self.assertEqual(len(downloader.calls), 1)
            call = downloader.calls[0]
            self.assertEqual(call["repo_id"], REPO)
            self.assertEqual(call["revision"], SHA)
            self.assertIs(call["token"], False)
            self.assertEqual(call["allow_patterns"], ["config.json", "model.safetensors", "tokenizer/vocab.json"])
            self.assertEqual(call["local_dir"], staging / "snapshot")
            self.assertEqual([row["path"] for row in result["integrity"]], call["allow_patterns"])
            self.assertEqual(sum(row["bytes"] for row in result["integrity"]), plan.expected_bytes)
        self.assertEqual(events[0], ("downloading_files", 0, 3))

    def test_tampered_persisted_plan_is_rejected_before_download(self):
        downloader = _Downloader(dict(BENIGN))
        plan = self._plan(BENIGN, downloader)
        hostile = plan.model_copy(update={"files": [ProviderFile.model_construct(path="D:config.json", size=100)]})
        with tempfile.TemporaryDirectory() as tmp, patch("titanos.providers.snapshot_download", downloader), \
                patch("titanos.providers.get_token", return_value=None):
            with self.assertRaisesRegex(ValueError, "':'"):
                HuggingFaceProvider().acquire(hostile, Path(tmp), lambda *event: None, threading.Event())
        self.assertEqual(downloader.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
