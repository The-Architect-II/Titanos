"""Provider contracts use controlled metadata; no network or model execution."""
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from titanos.providers import (
    AcquisitionRequest,
    GitHubSherpaProvider,
    HuggingFaceProvider,
    ImportedModelLink,
    ModelProvider,
    ProviderRegistry,
    _expected_backend,
    import_model_link,
)


class FixtureProvider(ModelProvider):
    name = "fixture"
    title = "Fixture provider"

    def search(self, query, task, limit=20):
        return [{"provider": self.name, "id": "fixture/model", "name": "Fixture", "task": task,
                 "source_url": "https://models.invalid/fixture/model", "compatibility": {"status": "candidate"}}]

    def inspect(self, identifier, revision=None):
        return {**self.search("", "embeddings", 1)[0], "requested_revision": revision,
                "resolved_revision": "f" * 40, "files": [{"path": "model.safetensors", "size": 10}],
                "license": "apache-2.0", "gated": False}

    def plan(self, request):
        return self._base_plan(request, self.inspect(request.identifier, request.revision),
                               files=[{"path": "model.safetensors", "size": 10}])

    def acquire(self, plan, staging, progress, cancel):
        raise AssertionError("Provider contract test never acquires")


class Checks(unittest.TestCase):
    def test_registry_extension_and_schema(self):
        registry = ProviderRegistry([FixtureProvider()])
        self.assertEqual(registry.names(), ["fixture"])
        self.assertEqual(registry.get("fixture").search("", "embeddings")[0]["task"], "embeddings")
        with self.assertRaises(ValueError):
            registry.get("missing")
        request = AcquisitionRequest(provider="fixture", identifier="fixture/model", revision="main",
                                     registry_name="embed", task="embeddings", backend="transformers", device="cpu")
        plan = registry.get("fixture").plan(request)
        self.assertEqual(plan.resolved_revision, "f" * 40)
        self.assertEqual(plan.model_spec["task"], "embeddings")
        self.assertEqual(plan.expected_bytes, 10)
        self.assertNotIn("token", plan.model_dump_json())

    def test_huggingface_exact_resolution_metadata_and_safe_files(self):
        info = NS(id="sentence-transformers/paraphrase-MiniLM-L3-v2", pipeline_tag="feature-extraction",
                  library_name="sentence-transformers", downloads=10, likes=2, gated=False, sha="a" * 40,
                  card_data={"license": "apache-2.0"},
                  siblings=[NS(rfilename="config.json", size=100), NS(rfilename="model.safetensors", size=200),
                            NS(rfilename="pytorch_model.bin", size=300), NS(rfilename="README.md", size=20)])
        with patch("titanos.providers.HfApi") as api, patch("titanos.providers.get_token", return_value=None):
            api.return_value.model_info.return_value = info
            provider = HuggingFaceProvider()
            request = AcquisitionRequest(provider="huggingface", identifier=info.id, revision="refs/pr/7",
                                         registry_name="mini", task="embeddings", backend="transformers", device="cpu")
            plan = provider.plan(request)
            api.return_value.model_info.assert_called_once_with(info.id, revision="refs/pr/7", files_metadata=True)
            self.assertEqual(plan.requested_revision, "refs/pr/7")
            self.assertEqual(plan.resolved_revision, "a" * 40)
            self.assertEqual(plan.license, "apache-2.0")
            self.assertEqual([row.path for row in plan.files], ["config.json", "model.safetensors", "README.md"])
            self.assertNotIn("pytorch_model.bin", plan.model_dump_json())
            self.assertEqual(plan.compatibility["status"], "compatible_plan")
            info.card_data={"license":"other"}
            self.assertIsNone(provider._describe(info)["license"])
            info.sha = None
            with self.assertRaisesRegex(ValueError, "immutable"):
                provider.plan(request)
            info.sha="a"*40;info.siblings=[NS(rfilename="config.json",size=100),NS(rfilename="onnx/remote.py",size=5)]
            compiled=request.model_copy(update={"backend":"onnx"})
            with self.assertRaisesRegex(ValueError,r"\.onnx graph"):
                provider.plan(compiled)
            info.id="Qwen/Qwen2.5-0.5B-Instruct-GGUF";info.pipeline_tag="text-generation"
            info.siblings=[NS(rfilename="README.md",size=20),
                           NS(rfilename="qwen2.5-0.5b-instruct-q4_k_m.gguf",size=491400032),
                           NS(rfilename="qwen2.5-0.5b-instruct-q8_0.gguf",size=531000000)]
            native=AcquisitionRequest(provider="huggingface",identifier=info.id,revision="main",
                                      registry_name="qwen-native",task="chat",backend="llama-cpp",
                                      device="cpu",model_file="qwen2.5-0.5b-instruct-q4_k_m.gguf")
            native_plan=provider.plan(native)
            self.assertEqual([row.path for row in native_plan.files],
                             ["qwen2.5-0.5b-instruct-q4_k_m.gguf","README.md"])
            self.assertEqual(native_plan.model_spec["native"]["model_file"],native.model_file)
            available=native_plan.model_copy(update={"runtime":{**native_plan.runtime,"status":"available","missing_modules":[]}})
            self.assertEqual(native_plan.fingerprint(),available.fingerprint())

    def test_github_sherpa_release_is_independent_digest_bound_source(self):
        release = {
            "id": 123,
            "tag_name": "tts-models",
            "published_at": "2025-06-30T02:22:05Z",
            "html_url": "https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models",
            "assets": [
                {"id": 9, "name": "vits-piper-en_US-amy-low.tar.bz2", "size": 67095344,
                 "browser_download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2",
                 "digest": "sha256:" + "c" * 64, "download_count": 5},
                {"id": 10, "name": "checksums.txt", "size": 30, "browser_download_url": "https://example.invalid/checksums"},
                {"id": 11, "name": "unsafe.zip", "size": 10, "browser_download_url": "https://example.invalid/unsafe"},
                {"id": 12, "name": "kokoro-en-v0_19.tar.bz2", "size": 20,
                 "browser_download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2",
                 "digest": "sha256:"+"d"*64,"download_count":1},
            ],
        }
        provider = GitHubSherpaProvider(release_loader=lambda: release)
        rows = provider.search("amy", "speech", 5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "github-sherpa")
        self.assertEqual(rows[0]["backend"], "sherpa-onnx")
        self.assertEqual(provider.search("","speech",20),rows)
        request = AcquisitionRequest(provider="github-sherpa", identifier="vits-piper-en_US-amy-low",
                                     registry_name="amy", task="speech", backend="sherpa-onnx", device="cpu")
        plan = provider.plan(request)
        self.assertEqual(plan.digest, "c" * 64)
        self.assertEqual(plan.resolved_revision, "github-release:123:asset:9:" + "c" * 64)
        self.assertEqual(plan.archive_format, "tar.bz2")
        self.assertTrue(plan.resume_supported)
        release["assets"][0]["digest"] = None
        with self.assertRaisesRegex(ValueError, "digest"):
            provider.plan(request)

    def test_link_import_recognizes_real_providers_and_keeps_general_metadata_only(self):
        self.assertEqual(_expected_backend("text-to-image","diffusers"),"diffusers")
        self.assertEqual(_expected_backend("image","diffusers"),"diffusers")
        hf = import_model_link("https://huggingface.co/owner/model/tree/abc123")
        self.assertEqual(hf, ImportedModelLink(provider="huggingface", identifier="owner/model",
                                               revision="abc123", canonical_url="https://huggingface.co/owner/model/tree/abc123",
                                               compatibility="provider-recognized"))
        github = import_model_link("https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2")
        self.assertEqual(github.provider, "github-sherpa")
        self.assertEqual(github.identifier, "vits-piper-en_US-amy-low")
        self.assertEqual(github.revision, "tts-models")
        explicit = import_model_link("https://github.com/k2-fsa/sherpa-onnx/releases/download/explicit-other-release/vits-piper-en_US-amy-low.tar.bz2")
        self.assertEqual(explicit.revision, "explicit-other-release")
        provider = GitHubSherpaProvider(release_loader=lambda: {"tag_name": "tts-models", "assets": []})
        with self.assertRaisesRegex(ValueError, "no fallback"):
            provider.inspect(explicit.identifier, explicit.revision)
        for suffix in ("blob/main/vits-piper-en_US-amy-low.tar.bz2",
                       "releases/download/tts-models/nested/vits-piper-en_US-amy-low.tar.bz2",
                       "vits-piper-en_US-amy-low.tar.bz2"):
            link = import_model_link("https://github.com/k2-fsa/sherpa-onnx/" + suffix)
            self.assertIsNone(link.provider)
            self.assertEqual(link.compatibility, "metadata-only")
        generic = import_model_link("https://models.example.org/cards/useful-model")
        self.assertIsNone(generic.provider)
        self.assertEqual(generic.compatibility, "metadata-only")
        with self.assertRaises(ValueError):
            import_model_link("file:///tmp/model")
        with self.assertRaises(ValueError):
            import_model_link("https://user:secret@example.org/model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
