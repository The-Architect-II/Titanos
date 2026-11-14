"""Tiny filesystem construction tests only; no interpreter/model stack executes."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("music_package", ROOT / "tools/music_package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


@unittest.skipUnless(os.name=='nt' or sys.platform.startswith('linux'), 'Music runtime publication currently supports Windows/Linux')
class MusicPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="titanos-music-package-")
        self.root = Path(self.temp.name).resolve()
        self.interpreter = self.root / "interpreter"
        self.site = self.root / "owned-env/site-packages"
        self.source = self.root / "source"
        self.notices = self.root / "notices"
        self.adapter = self.root / "adapter.py"
        self.output = self.root / "portable/runtimes/ace-step-1.5"
        self.write(self.interpreter / "bin/python3.11", b"fixture bytes, never an executed interpreter")
        self.write(self.interpreter / "lib/python3.11/os.py", b"# fixture stdlib\n")
        # A donor prefix may contain a different pip/setuptools. It must not leak.
        self.write(self.interpreter / "lib/python3.11/site-packages/old.py", b"old baseline")
        self.write(self.site / "example/__init__.py", b"raise AssertionError('Do not import fixture')\n")
        self.write(self.site / "example-1.0.dist-info/METADATA", b"Name: example\nVersion: 1.0\nLicense-Expression: MIT\n")
        self.write(self.site / "example-1.0.dist-info/licenses/LICENSE", b"fixture license text\n")
        self.write(self.source / "acestep/__init__.py", b"# source fixture; not the real implementation\n")
        self.write(self.source / "LICENSE", b"fixture source notice\n")
        self.write(self.notices / "NOTICE.txt", b"fixture runtime notice\n")
        self.write(self.adapter, b"raise AssertionError('Packaging must never execute adapter')\n")
        self.pin = self.root / "source-pin.json"
        self.pin.write_text(json.dumps({"repository": "ace-step/ACE-Step-1.5", "commit": package.CODE_REVISION,
            "files": {p.relative_to(self.source).as_posix(): {"bytes": p.stat().st_size, "sha256": package.digest(p)}
                      for p in self.source.rglob("*") if p.is_file()}}))

    def tearDown(self):
        assert self.root.resolve() == self.root and self.root.name.startswith("titanos-music-package-")
        self.temp.cleanup()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def build(self, **overrides):
        args = dict(interpreter=self.interpreter, site_packages=self.site, adapter=self.adapter,
                    source=self.source, source_pin=self.pin, notices=self.notices, output=self.output,
                    executable="bin/python3.11", site_relative="lib/python3.11/site-packages")
        args.update(overrides)
        return package.package_runtime(**args)

    def test_complete_relative_bundle_moves_without_old_prefix(self):
        manifest_path = self.build()
        value = package.verify_runtime(self.output)
        self.assertNotIn(str(self.root), manifest_path.read_text())
        self.assertNotIn("destination", value)
        self.assertNotIn("python/lib/python3.11/site-packages/old.py", value["files"])
        self.assertEqual(value["qualification"]["target_native"], False)
        self.assertEqual(set(value["inventories"]), {"dependencies", "sources", "licenses"})
        moved = self.root / "moved-runtime"
        assert self.output.resolve().is_relative_to(self.root) and moved.parent.resolve() == self.root
        self.output.rename(moved)
        self.assertEqual(package.verify_runtime(moved), value)
        for name, pin in value["files"].items():
            self.assertEqual(hashlib.sha256((moved / name).read_bytes()).hexdigest(), pin["sha256"])

    def test_missing_tampered_and_extra_payload_refused(self):
        self.build()
        payload = self.output / "adapter/adapter.py"
        original = payload.read_bytes()
        payload.write_bytes(b"changed")
        with self.assertRaises(ValueError): package.verify_runtime(self.output)
        payload.unlink()
        with self.assertRaises(ValueError): package.verify_runtime(self.output)
        payload.write_bytes(original)
        self.write(self.output / "__pycache__/unexpected.pyc", b"unexpected startup payload")
        with self.assertRaises(ValueError): package.verify_runtime(self.output)

    def test_pinned_source_missing_or_changed_is_not_published(self):
        self.write(self.source / "acestep/__init__.py", b"changed source")
        with self.assertRaises(ValueError): self.build()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.output.parent.glob(".ace-step-1.5-*")))

    def test_external_weights_and_venv_are_rejected(self):
        self.write(self.site / "model.safetensors", b"not actually weights")
        with self.assertRaises(ValueError): self.build()
        (self.site / "model.safetensors").unlink()
        self.write(self.interpreter / "pyvenv.cfg", b"home = /old/absolute/prefix\n")
        with self.assertRaises(ValueError): self.build()
        self.assertFalse(self.output.exists())

    def test_absolute_pth_and_missing_dependency_refused(self):
        path = self.site / "bad.pth"
        self.write(path, b"C:/old/venv/site-packages\n")
        with self.assertRaises(ValueError): self.build()
        path.write_text("missing-directory\n")
        with self.assertRaises(ValueError): self.build()
        path.write_text("import __editable___finder; __editable___finder.install()\n")
        with self.assertRaises(ValueError): self.build()

    def test_embedded_python_gets_relative_site_path(self):
        self.write(self.interpreter / "python.exe", b"fixture")
        self.write(self.interpreter / "python311.zip", b"fixture stdlib archive")
        self.write(self.interpreter / "python311._pth", b"python311.zip\n.\n")
        self.build(executable="python.exe", site_relative="Lib/site-packages")
        value = package.verify_runtime(self.output)
        self.assertEqual(value["executable"], "python/python.exe")
        self.assertIn("Lib/site-packages", (self.output / "python/python311._pth").read_text())

    def test_existing_target_and_publication_race_preserve_other_files(self):
        self.output.mkdir(parents=True)
        marker = self.output / "preserve"
        marker.write_text("original")
        with self.assertRaises(FileExistsError): self.build()
        self.assertEqual(marker.read_text(), "original")
        marker.unlink(); self.output.rmdir()
        publish = package.publish
        def race(stage, target):
            target.mkdir()
            (target / "other-owner").write_text("retained")
            publish(stage, target)
        with patch.object(package, "publish", race):
            with self.assertRaises(OSError): self.build()
        self.assertEqual((self.output / "other-owner").read_text(), "retained")
        self.assertFalse(list(self.output.parent.glob(".ace-step-1.5-*")))

    def test_escaping_link_refused_before_read_or_publication(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside")
        link = self.site / "escape.py"
        try: link.symlink_to(outside)
        except OSError as error: self.skipTest("Local symlink creation unavailable: " + str(error))
        with self.assertRaises(ValueError): self.build()
        self.assertEqual(outside.read_text(), "outside")
        self.assertFalse(self.output.exists())

    def test_pip_provenance_records_are_excluded_recorded_and_refused_on_return(self):
        # pip's PEP 610 record of a local install carries a file:/// URL of the packaging machine (runtime r4 shipped four of them);
        # the fixture URL is a neutral placeholder, not a lab path, assembled from parts so this file carries no contiguous
        # literal the builder's distribution scan (local file URL class) would refuse inside SOURCE/
        provenance = {"url": "file:" + "///" + "D:/wheel-cache/example-1.0-py3-none-any.whl", "archive_info": {"hash": "sha256=00"}}
        self.write(self.site / "example-1.0.dist-info/direct_url.json", json.dumps(provenance).encode())
        self.write(self.interpreter / "lib/python3.11/site-packages/old-1.0.dist-info/direct_url.json", b"{}")  # donor prefix, skipped anyway
        self.write(self.site / "example/direct_url.json", b"not a dist-info record; kept")
        self.build()
        value = package.verify_runtime(self.output)
        site = "python/lib/python3.11/site-packages/"
        self.assertNotIn(site + "example-1.0.dist-info/direct_url.json", value["files"])
        self.assertFalse((self.output / site / "example-1.0.dist-info/direct_url.json").exists())
        self.assertIn(site + "example/direct_url.json", value["files"])
        self.assertIn(site + "example-1.0.dist-info/METADATA", value["files"])
        self.assertEqual(value["exclusions"]["pip_provenance"]["pattern"], "*.dist-info/direct_url.json")
        self.assertEqual(value["exclusions"]["pip_provenance"]["files"], [site + "example-1.0.dist-info/direct_url.json"])
        dependencies = json.loads((self.output / "DEPENDENCIES.json").read_text())
        self.assertEqual(dependencies["exclusions"], value["exclusions"])
        self.assertNotIn("wheel-cache", (self.output / "runtime.json").read_text() + (self.output / "DEPENDENCIES.json").read_text())
        for name in self.output.rglob("*"):
            self.assertFalse(name.is_file() and package.is_pip_provenance(name.relative_to(self.output).as_posix()), name)
        # a runtime that still carries such a record (r4) is refused even when its manifest pins the file
        planted = self.output / site / "example-1.0.dist-info/direct_url.json"
        planted.write_bytes(json.dumps(provenance).encode())
        manifest = self.output / "runtime.json"
        current = json.loads(manifest.read_text())
        current["files"][site + "example-1.0.dist-info/direct_url.json"] = {"bytes": planted.stat().st_size, "sha256": package.digest(planted)}
        manifest.write_text(json.dumps(current))
        with self.assertRaisesRegex(ValueError, "pip provenance"): package.verify_runtime(self.output)
        planted.unlink()
        with self.assertRaisesRegex(ValueError, "membership"): package.verify_runtime(self.output)

    def test_runtime_without_provenance_records_lists_an_empty_exclusion(self):
        self.build()
        value = package.verify_runtime(self.output)
        self.assertEqual(value["exclusions"]["pip_provenance"]["files"], [])
        self.assertIn("file:///", value["exclusions"]["pip_provenance"]["reason"])

    def test_manifest_path_escape_and_duplicate_keys_rejected(self):
        self.build()
        manifest = self.output / "runtime.json"
        original = manifest.read_text()
        value = json.loads(original)
        value["adapter"] = "../outside.py"
        manifest.write_text(json.dumps(value))
        with self.assertRaises(ValueError): package.verify_runtime(self.output)
        manifest.write_text(original.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,'))
        with self.assertRaises(ValueError): package.verify_runtime(self.output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
