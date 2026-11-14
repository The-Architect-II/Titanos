"""`titanos pull` (src/titanos/hub.py) never hands the downloader a Windows-hostile or glob-ambiguous sibling name.

Same Daybreak path-write family as the providers.py fix: every pull backend now computes an explicit list of
resolved sibling names, validates each with the shared acquisition-path rule before download, refuses case-only
variants, passes literal-escaped allow_patterns, and rejects an unplanned file after download. Attack-shaped names
are inert fixtures: no network, no model, and the downloader is an in-process fake that records its calls and
materialises exactly the files its patterns select, modelling huggingface_hub 0.36 faithfully: fnmatch.fnmatch (which
folds case and '/' vs '\\' through ntpath.normcase on Windows) and the os.path.join(*name.split('/')) write rule."""
import fnmatch
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from titanos import hub

ROOT = Path(__file__).resolve().parents[1]
REPO = "fixture-org/fixture-model"
SHA = "c" * 40


def _info(names):
    return NS(id=REPO, pipeline_tag="text-generation", library_name="transformers", downloads=1, likes=0,
              gated=False, sha=SHA, siblings=[NS(rfilename=name) for name in names])


def _args(backend="llama-cpp", task="chat", cache_dir=None, **extra):
    base = dict(anonymous=True, token_env=None, source=REPO, revision=None, backend=backend, task=task, graph=None,
                cache_dir=cache_dir, model_file=None, mmproj=None, native_options=None, allow_pickle=False)
    base.update(extra)
    return NS(**base)


class _Downloader:
    """Inert stand-in for snapshot_download: records calls and writes the fake-repository files its patterns select."""

    def __init__(self, names, extra=()):
        self.names = list(names)
        self.extra = list(extra)  # relative names written in addition to the selected files (an unplanned write)
        self.calls = []

    def __call__(self, repo_id, revision, token, local_dir, allow_patterns=None, ignore_patterns=None):
        self.calls.append({"repo_id": repo_id, "revision": revision, "token": token, "local_dir": Path(local_dir),
                           "allow_patterns": None if allow_patterns is None else list(allow_patterns),
                           "ignore_patterns": ignore_patterns})
        selected = [name for name in self.names
                    if allow_patterns is None or any(fnmatch.fnmatch(name, p) for p in allow_patterns)]
        for name in [*selected, *self.extra]:
            target = Path(local_dir) / os.path.join(*name.split("/"))  # huggingface_hub's local path rule
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(name.encode())  # the bytes identify which sibling landed at a path
        return str(local_dir)

    @property
    def selected(self):
        return [name for name in self.names if any(fnmatch.fnmatch(name, p) for p in self.calls[-1]["allow_patterns"])]


class Checker(unittest.TestCase):
    def pull(self, names, args, extra=()):
        downloader = _Downloader(names, extra)
        with tempfile.TemporaryDirectory() as tmp:
            args.cache_dir = args.cache_dir or tmp
            with patch.object(hub, "HfApi", return_value=NS(model_info=lambda *a, **k: _info(names))), \
                    patch.object(hub, "snapshot_download", downloader):
                try:
                    result = hub.pull(args)
                except ValueError as exc:
                    return downloader, None, str(exc), Path(args.cache_dir)
                # Files materialised under the returned directory, recorded before the temporary directory is removed.
                downloader.written = sorted(p.relative_to(result[0]).as_posix() for p in Path(result[0]).rglob("*") if p.is_file())
            return downloader, result, None, Path(args.cache_dir)

    # (a) hostile sibling names are refused before the downloader is ever called
    def test_hostile_selected_or_metadata_sibling_refused_before_download(self):
        cases = [  # (repository siblings, args, message fragment)
            (["model.gguf", "D:config.json"], _args(model_file="model.gguf"), "drive"),
            (["model.gguf", "NUL.json"], _args(model_file="model.gguf"), "reserved device"),
            (["con.gguf", "config.json"], _args(model_file="con.gguf"), "reserved device"),
            (["model.gguf", "chat.jinja:Zone.Identifier"], _args(model_file="model.gguf", mmproj="chat.jinja:Zone.Identifier"), "':'"),
            (["model.safetensors", "tokenizer.json:Zone.Identifier"], _args(backend="transformers"), "':'"),
            (["model.safetensors", "D:config.json"], _args(backend="transformers"), "drive"),
            (["model.safetensors", "NUL.json"], _args(backend="transformers"), "reserved device"),
            (["model.onnx", "con.gguf"], _args(backend="onnx", task="embeddings"), "reserved device"),
            (["openvino_model.xml", "openvino_model.bin", "x/../y.json"], _args(backend="openvino", task="embeddings"), "'..'"),
        ]
        for names, args, fragment in cases:
            downloader, result, error, _ = self.pull(names, args)
            self.assertIsNone(result, names)
            self.assertIn(fragment, error, names)
            self.assertEqual(downloader.calls, [], names)
            # The message names one bounded component, never the whole selection.
            self.assertNotIn("model.gguf", error, names)

    def test_hostile_unselected_sibling_is_never_selected_by_the_llama_cpp_metadata_patterns(self):
        # An ADS name does not match '*.json' case-sensitively, so it is neither planned nor fetched.
        downloader, result, error, _ = self.pull(["model.gguf", "config.json", "tokenizer.json:Zone.Identifier"],
                                                  _args(model_file="model.gguf"))
        self.assertIsNone(error)
        self.assertEqual(downloader.calls[0]["allow_patterns"], ["config.json", "model.gguf"])
        self.assertEqual(downloader.selected, ["model.gguf", "config.json"])

    # (b) bracket patterns cannot select an unvalidated sibling
    def test_bracket_pattern_sibling_is_refused_and_would_be_literal_escaped(self):
        downloader, result, error, _ = self.pull(["m[1].gguf", "m1.gguf", "config.json"], _args(model_file="m[1].gguf"))
        self.assertIsNone(result)
        self.assertIn("glob metacharacter", error)
        self.assertEqual(downloader.calls, [])
        # Defence in depth: even if such a name reached the downloader it would be escaped to match only itself.
        [pattern] = hub.literal_patterns(["m[1].gguf"])
        self.assertEqual(pattern, "m[[]1].gguf")
        self.assertTrue(fnmatch.fnmatchcase("m[1].gguf", pattern))
        self.assertFalse(fnmatch.fnmatchcase("m1.gguf", pattern))
        # A bracket sibling that is merely present is not selected by the literal patterns of a benign plan.
        downloader, result, error, _ = self.pull(["m1.gguf", "m[1].gguf", "config.json"], _args(model_file="m1.gguf"))
        self.assertIsNone(error)
        self.assertEqual(downloader.calls[0]["allow_patterns"], ["config.json", "m1.gguf"])
        self.assertEqual(downloader.selected, ["m1.gguf", "config.json"])

    # (c) case variants are refused before download
    def test_case_variant_siblings_are_refused_before_download(self):
        for names, args in [(["model.gguf", "MODEL.GGUF", "config.json"], _args(model_file="model.gguf")),
                            (["model.gguf", "config.json", "CONFIG.JSON"], _args(model_file="model.gguf")),
                            (["model.safetensors", "Model.safetensors"], _args(backend="transformers")),
                            (["model.onnx", "MODEL.onnx"], _args(backend="onnx", task="embeddings"))]:
            downloader, result, error, _ = self.pull(names, args)
            self.assertIsNone(result, names)
            self.assertIn("only in case", error, names)
            self.assertEqual(downloader.calls, [], names)

    def test_unselected_backslash_sibling_is_refused_before_download(self):
        # huggingface_hub's Windows fnmatch folds '\\' to '/': the literal pattern of the validated 'sub/other.gguf' would
        # also select 'sub\\other.gguf', which the downloader writes to the planned path (the faithful fake shows it).
        names = ["sub/other.gguf", "sub\\other.gguf", "config.json"]
        downloader, result, error, _ = self.pull(names, _args(model_file="sub/other.gguf"))
        self.assertIsNone(result)
        self.assertIn("path separator", error)
        self.assertIn("sub\\\\other.gguf", error)
        self.assertNotIn("sub/other.gguf", error)
        self.assertEqual(downloader.calls, [])
        if os.name == "nt":  # what the pre-check prevents: the fake, like huggingface_hub, over-selects on Windows
            with tempfile.TemporaryDirectory() as tmp:
                downloader(REPO, revision=SHA, token=False, local_dir=tmp, allow_patterns=hub.literal_patterns(["sub/other.gguf"]))
                self.assertEqual(downloader.selected, ["sub/other.gguf", "sub\\other.gguf"])
                self.assertEqual((Path(tmp) / "sub" / "other.gguf").read_bytes(), b"sub\\other.gguf")
        self.assertEqual(hub.downloader_fold("Sub\\Other.GGUF"), "sub/other.gguf")
        # A selected backslash name is refused by the validator itself, whichever backend selected it.
        downloader, result, error, _ = self.pull(["model.safetensors", "sub\\config.json"], _args(backend="transformers"))
        self.assertIn("backslash", error)
        self.assertEqual(downloader.calls, [])

    def test_repository_file_in_the_downloader_reserved_cache_directory_is_refused(self):
        # The post-download walk leaves huggingface_hub's own .cache metadata alone, so nothing may be planned into it.
        for names, args in [(["model.onnx", ".cache/huggingface/download/model.onnx.metadata", "config.json"],
                             _args(backend="onnx", task="embeddings")),
                            (["model.safetensors", ".cache/config.json"], _args(backend="transformers")),
                            (["model.gguf", ".cache/config.json"], _args(model_file="model.gguf"))]:
            downloader, result, error, _ = self.pull(names, args)
            self.assertIsNone(result, names)
            self.assertIn("reserved .cache", error, names)
            self.assertIn(".cache/", error, names)
            self.assertEqual(downloader.calls, [], names)
        # A .cache sibling that is not selected is neither planned nor fetched, and the downloader's own .cache is skipped.
        downloader, result, error, _ = self.pull(["model.gguf", "config.json", ".cache/README.md"], _args(model_file="model.gguf"),
                                                  extra=[".cache/huggingface/download/model.gguf.metadata"])
        self.assertIsNone(error)
        self.assertEqual(downloader.calls[0]["allow_patterns"], ["config.json", "model.gguf"])
        self.assertEqual(downloader.written, [".cache/huggingface/download/model.gguf.metadata", "config.json", "model.gguf"])

    # (d) an unplanned downloaded file is rejected and a directory this pull created is removed
    def test_unplanned_downloaded_file_is_rejected_and_created_target_removed(self):
        downloader, result, error, base = self.pull(["model.gguf", "config.json"], _args(model_file="model.gguf"),
                                                     extra=["extra/payload.bin"])
        self.assertIsNone(result)
        self.assertIn("did not name", error)
        self.assertIn("extra/payload.bin", error)
        self.assertEqual(len(downloader.calls), 1)
        self.assertFalse(downloader.calls[0]["local_dir"].exists(), "half snapshot left behind")

    def test_unplanned_file_rejection_keeps_a_pre_existing_target_but_removes_what_the_pull_introduced(self):
        names = ["model.gguf", "config.json"]
        downloader = _Downloader(names, extra=["extra/payload.bin", "tokenizer/added.json",
                                               ".cache/huggingface/download/model.gguf.metadata"])
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(model_file="model.gguf", cache_dir=tmp)
            target = hub.destination(args, _info(names))
            (target / "tokenizer").mkdir(parents=True)
            (target / "q8.gguf").write_bytes(b"previous quantization of the same revision")
            (target / "tokenizer" / "vocab.json").write_bytes(b"kept")
            (target / ".cache" / "huggingface" / "download").mkdir(parents=True)
            (target / ".cache" / "huggingface" / "download" / "q8.gguf.metadata").write_bytes(b"kept metadata")
            with patch.object(hub, "HfApi", return_value=NS(model_info=lambda *a, **k: _info(names))), \
                    patch.object(hub, "snapshot_download", downloader):
                with self.assertRaisesRegex(ValueError, "extra/payload.bin"):
                    hub.pull(args)
                self.assertTrue((target / "q8.gguf").is_file(), "pre-existing directory must not be removed")
                self.assertTrue((target / "tokenizer" / "vocab.json").is_file(), "pre-existing file in a shared directory kept")
                self.assertTrue((target / ".cache" / "huggingface" / "download" / "q8.gguf.metadata").is_file())
                # Everything the refused pull introduced is gone: the unplanned files, their new directories, the planned
                # files of the same pull and the downloader's metadata for them.
                self.assertFalse((target / "extra").exists(), "refused unplanned write left on disk")
                self.assertFalse((target / "tokenizer" / "added.json").exists())
                self.assertFalse((target / "model.gguf").exists())
                self.assertFalse((target / "config.json").exists())
                self.assertFalse((target / ".cache" / "huggingface" / "download" / "model.gguf.metadata").exists())
                self.assertEqual(sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()),
                                 [".cache/huggingface/download/q8.gguf.metadata", "q8.gguf", "tokenizer/vocab.json"])
                # A second pull of the same revision cannot launder the refused file into an accepted snapshot.
                downloader.extra = []
                self.assertEqual(hub.pull(args), (str(target), SHA))
                self.assertTrue((target / "model.gguf").is_file())
                self.assertEqual(sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()),
                                 [".cache/huggingface/download/q8.gguf.metadata", "config.json", "model.gguf", "q8.gguf",
                                  "tokenizer/vocab.json"])

    def test_link_introduced_into_a_pre_existing_target_is_unlinked_not_followed(self):
        names = ["model.gguf", "config.json"]
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(model_file="model.gguf", cache_dir=tmp)
            target = hub.destination(args, _info(names))
            target.mkdir(parents=True)
            (target / "q8.gguf").write_bytes(b"previous quantization")
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "kept.txt").write_bytes(b"outside")

            def linking_downloader(repo_id, revision, token, local_dir, allow_patterns):
                for name in allow_patterns:
                    (Path(local_dir) / name).write_bytes(b"x")
                link = Path(local_dir) / "linked"
                try:
                    os.symlink(outside, link, target_is_directory=True)
                except (OSError, NotImplementedError):
                    if os.name != "nt":
                        self.skipTest("symlink creation is not permitted for this account")
                    import _winapi
                    _winapi.CreateJunction(str(outside), str(link))
                return str(local_dir)

            with patch.object(hub, "HfApi", return_value=NS(model_info=lambda *a, **k: _info(names))), \
                    patch.object(hub, "snapshot_download", linking_downloader):
                with self.assertRaisesRegex(ValueError, "symbolic link or junction: 'linked'"):
                    hub.pull(args)
            self.assertTrue((target / "q8.gguf").is_file())
            self.assertFalse((target / "linked").exists() or (target / "linked").is_symlink(), "introduced link removed")
            self.assertFalse((target / "model.gguf").exists())
            self.assertTrue((outside / "kept.txt").is_file(), "removal must not follow the link")

    def test_link_in_snapshot_is_rejected(self):
        names = ["model.gguf", "config.json"]
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(model_file="model.gguf", cache_dir=tmp)
            target = hub.destination(args, _info(names))
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "kept.txt").write_bytes(b"outside")

            def linking_downloader(repo_id, revision, token, local_dir, allow_patterns):
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                for name in allow_patterns:
                    (Path(local_dir) / name).write_bytes(b"x")
                link = Path(local_dir) / "linked"
                try:
                    os.symlink(outside, link, target_is_directory=True)
                except (OSError, NotImplementedError):
                    if os.name != "nt":
                        self.skipTest("symlink creation is not permitted for this account")
                    import _winapi  # a junction needs no privilege on Windows
                    _winapi.CreateJunction(str(outside), str(link))
                return str(local_dir)

            with patch.object(hub, "HfApi", return_value=NS(model_info=lambda *a, **k: _info(names))), \
                    patch.object(hub, "snapshot_download", linking_downloader):
                with self.assertRaisesRegex(ValueError, "symbolic link or junction: 'linked'"):
                    hub.pull(args)
            self.assertFalse(target.exists(), "created target with a link inside must be removed")
            self.assertTrue((outside / "kept.txt").is_file(), "removal must not follow the link")

    # (e) normal GGUF split, projector and metadata pulls are preserved
    def test_split_gguf_projector_and_metadata_pull_is_preserved(self):
        names = ["q4-00001-of-00002.gguf", "q4-00002-of-00002.gguf", "q8.gguf", "mmproj.gguf", "config.json",
                 "tokenizer.json", "chat.jinja", "LICENSE", "README.md", "tokenizer/tokenizer_config.json", "notes.txt"]
        downloader, result, error, base = self.pull(names, _args(model_file="q4-00001-of-00002.gguf", mmproj="mmproj.gguf"))
        self.assertIsNone(error)
        expected = ["LICENSE", "README.md", "chat.jinja", "config.json", "mmproj.gguf", "q4-00001-of-00002.gguf",
                    "q4-00002-of-00002.gguf", "tokenizer.json", "tokenizer/tokenizer_config.json"]
        call = downloader.calls[0]
        self.assertEqual(call["allow_patterns"], expected)
        self.assertEqual(call["allow_patterns"], hub.literal_patterns(expected))
        self.assertIsNone(call["ignore_patterns"])
        self.assertEqual(call["repo_id"], REPO)
        self.assertEqual(call["revision"], SHA)
        self.assertIs(call["token"], False)
        self.assertEqual(sorted(downloader.selected), expected)
        target = base.resolve() / ("fixture-org--fixture-model@" + SHA[:12])
        self.assertEqual(call["local_dir"], target)
        self.assertEqual(result, (str(target), SHA))
        self.assertEqual(downloader.written, expected)
        # Split without the projector, and an incomplete split, behave as before.
        downloader, result, error, _ = self.pull(names, _args(model_file="q8.gguf"))
        self.assertIsNone(error)
        self.assertNotIn("mmproj.gguf", downloader.calls[0]["allow_patterns"])
        downloader, result, error, _ = self.pull([n for n in names if n != "q4-00002-of-00002.gguf"],
                                                  _args(model_file="q4-00001-of-00002.gguf"))
        self.assertIn("Incomplete split", error)
        self.assertEqual(downloader.calls, [])

    def test_metadata_siblings_match_case_insensitively_on_every_platform(self):
        # Resolved to the existing names (never a glob), so validation and the collision checks still apply to each.
        names = ["model.gguf", "CONFIG.JSON", "Readme.md", "License.txt", "template.JINJA", "notes.txt", "licenses/Apache.txt"]
        downloader, result, error, _ = self.pull(names, _args(model_file="model.gguf"))
        self.assertIsNone(error)
        self.assertEqual(downloader.calls[0]["allow_patterns"],
                         ["CONFIG.JSON", "License.txt", "Readme.md", "licenses/Apache.txt", "model.gguf", "template.JINJA"])
        self.assertEqual(sorted(downloader.written), sorted(downloader.calls[0]["allow_patterns"]))
        self.assertEqual(hub.matching_siblings(["a.JSON", "b.json", "c.jsonl", "x.json:Zone.Identifier"], hub.METADATA_PATTERNS),
                         {"a.JSON", "b.json"})

    # (h) the lab fixture fetcher (tools/fetch_checkpoints.py) uses the same validated plan and post-download check
    def test_fetch_checkpoints_tool_plans_and_fetches_through_hub(self):
        spec = importlib.util.spec_from_file_location("fetch_checkpoints", ROOT / "tools" / "fetch_checkpoints.py")
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        files = ["config.json", "model.safetensors", "pytorch_model.bin", "onnx/model.onnx", "openvino/openvino_model.xml",
                 "tf_model.h5", "tokenizer/vocab.json"]
        self.assertEqual(tool.select(files, "chat"), ["config.json", "model.safetensors", "tokenizer/vocab.json"])
        self.assertEqual(tool.select(files, "image"), ["config.json", "model.safetensors", "pytorch_model.bin", "tokenizer/vocab.json"])
        for hostile in ("D:config.json", "NUL.json", "m[1].safetensors", "x/../y.json", "Config.json"):
            with self.subTest(hostile=hostile), self.assertRaises(ValueError):
                tool.select(files + [hostile], "chat")
        downloader = _Downloader(files)
        api = NS(model_info=lambda repo, revision: _info(files))
        with tempfile.TemporaryDirectory() as tmp, patch.object(hub, "snapshot_download", downloader):
            path = tool.fetch(api, tmp, "chat", REPO, SHA)
            self.assertEqual(Path(path), Path(tmp) / "chat")
            call = downloader.calls[0]
            self.assertEqual(call["allow_patterns"], ["config.json", "model.safetensors", "tokenizer/vocab.json"])
            self.assertIsNone(call["ignore_patterns"])
            self.assertEqual(call["revision"], SHA)
            self.assertEqual(sorted(p.relative_to(path).as_posix() for p in Path(path).rglob("*") if p.is_file()),
                             call["allow_patterns"])
            downloader.extra = ["extra/payload.bin"]
            with self.assertRaisesRegex(ValueError, "did not name"):
                tool.fetch(api, tmp, "vision", REPO, SHA)
            self.assertFalse((Path(tmp) / "vision").exists())

    # (f) onnx-diffusers: the required set plus metadata are all named explicitly
    def test_onnx_diffusers_required_set_and_metadata_are_listed_explicitly(self):
        from titanos.onnx_images import COMPONENTS
        required = {name + "/model.onnx" for name in COMPONENTS} | {
            "model_index.json", "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
            "tokenizer/merges.txt", "tokenizer/vocab.json", "feature_extractor/preprocessor_config.json"}
        names = sorted(required | {"unet/weights.pb", "README.md", "LICENSE.md"})
        downloader, result, error, _ = self.pull(names, _args(backend="onnx-diffusers", task="image"))
        self.assertIsNone(error)
        call = downloader.calls[0]
        self.assertEqual(call["allow_patterns"], names)
        self.assertTrue(required <= set(call["allow_patterns"]))
        self.assertIn("README.md", call["allow_patterns"])
        self.assertIn("unet/weights.pb", call["allow_patterns"], "external graph data must stay with its graph")
        self.assertIsNone(call["ignore_patterns"])
        self.assertEqual(sorted(downloader.selected), names)
        # A hostile sibling in the export is refused before download.
        downloader, result, error, _ = self.pull(names + ["unet/weights.pb:Zone.Identifier"],
                                                  _args(backend="onnx-diffusers", task="image"))
        self.assertIn("':'", error)
        self.assertEqual(downloader.calls, [])
        downloader, result, error, _ = self.pull([n for n in names if n != "safety_checker/model.onnx"],
                                                  _args(backend="onnx-diffusers", task="image"))
        self.assertIn("incomplete", error)
        self.assertEqual(downloader.calls, [])

    # (g) default transformers path: every kept sibling validated and listed, excluded formats absent
    def test_transformers_path_lists_every_kept_sibling(self):
        names = ["config.json", "model.safetensors", "tokenizer/vocab.json", "pytorch_model.bin", "weights.gguf",
                 "tf_model.h5", "MODEL.PT", "generation_config.json"]
        downloader, result, error, _ = self.pull(names, _args(backend="transformers"))
        self.assertIsNone(error)
        call = downloader.calls[0]
        self.assertEqual(call["allow_patterns"], ["config.json", "generation_config.json", "model.safetensors", "tokenizer/vocab.json"])
        self.assertIsNone(call["ignore_patterns"])
        self.assertEqual(sorted(downloader.selected), call["allow_patterns"])
        downloader, result, error, _ = self.pull(names, _args(backend="transformers", allow_pickle=True))
        self.assertIsNone(error)
        self.assertIn("pytorch_model.bin", downloader.calls[0]["allow_patterns"])
        self.assertIn("MODEL.PT", downloader.calls[0]["allow_patterns"])
        self.assertNotIn("weights.gguf", downloader.calls[0]["allow_patterns"])
        downloader, result, error, _ = self.pull(["weights.gguf", "config.json"], _args(backend="transformers"))
        self.assertIn("No supported checkpoint", error)
        self.assertEqual(downloader.calls, [])

    def test_whole_repository_backends_name_every_sibling(self):
        for names, args in [(["model.onnx", "model.onnx_data", "config.json"], _args(backend="onnx", task="embeddings")),
                            (["openvino_model.xml", "openvino_model.bin", "config.json"], _args(backend="openvino", task="embeddings"))]:
            downloader, result, error, _ = self.pull(names, args)
            self.assertIsNone(error, names)
            self.assertEqual(downloader.calls[0]["allow_patterns"], sorted(names))
            self.assertEqual(sorted(downloader.selected), sorted(names))


if __name__ == "__main__":
    unittest.main(verbosity=2)
