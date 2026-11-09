from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/featurebench_native_mini"
if not SOURCE.exists():
    SOURCE = Path(__file__).resolve().parents[1] / "scripts"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load("resume_common", "common.py")
with patch.dict(sys.modules, {"common": common}):
    images = load("resume_images", "cached_images.py")


class OfflineResultTests(unittest.TestCase):
    def test_diagnostic_summary_gets_the_actual_saved_exit_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "native.traj.json").write_text(json.dumps({"info": {"exit_status": "Submitted"}}))
            diagnostic = {"model_api_calls": 0, "simulated_model_responses": 2}
            result = common.offline_agent_info(root, diagnostic)
            self.assertEqual(result["exit_status"], "Submitted")
            self.assertEqual(result["model_api_calls"], 0)
            self.assertNotIn("exit_status", diagnostic)

    def test_failed_or_missing_status_cannot_become_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for info in [{"exit_status": "TimeExceeded"}, {}]:
                (root / "native.traj.json").write_text(json.dumps({"info": info}))
                with self.subTest(info=info), self.assertRaises((ValueError, KeyError)):
                    common.offline_agent_info(root, {"model_api_calls": 0})


class CachedImageTests(unittest.TestCase):
    def fixture(self, root):
        blobs = root / "layout/blobs/sha256"
        blobs.mkdir(parents=True)
        def blob(data):
            key = "sha256:" + hashlib.sha256(data).hexdigest()
            (blobs / key.split(":")[1]).write_bytes(data)
            return key
        config = blob(b'{"original":"config"}')
        layer = blob(b"original compressed layer")
        manifest = blob(json.dumps({"config": {"digest": config}, "layers": [{"digest": layer}]}).encode())
        image = {"repository": "original/repo", "digest": manifest, "config_digest": config,
                 "blobs": [config, layer, manifest]}
        plan = {"cache": str(root / "layout"), "images": [image], "unique_blobs": {
            key: {"size": (blobs / key.split(":")[1]).stat().st_size} for key in image["blobs"]}}
        (root / "plan.json").write_text(json.dumps(plan))
        (root / "verified.json").write_text("{}")
        settings = {"plan": str(root / "plan.json"), "plan_sha256": images.digest(root / "plan.json"),
                    "verified_blobs": str(root / "verified.json")}
        return settings, image, blobs / layer.split(":")[1]

    def test_pinned_manifest_and_layer_bytes_are_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            settings, expected, layer = self.fixture(Path(folder))
            before = layer.read_bytes()
            _, actual = images.verified_image(settings, "original/repo")
            self.assertEqual(actual, expected)
            self.assertEqual(layer.read_bytes(), before)

    def test_same_length_corruption_is_detected_without_deleting_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            settings, _, layer = self.fixture(Path(folder))
            layer.write_bytes(b"x" * layer.stat().st_size)
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                images.verified_image(settings, "original/repo")
            self.assertTrue(layer.exists())

    def test_changed_plan_is_not_silently_resolved_to_latest(self):
        with tempfile.TemporaryDirectory() as folder:
            settings, _, _ = self.fixture(Path(folder))
            Path(settings["plan"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "plan changed"):
                images.verified_image(settings, "original/repo")


if __name__ == "__main__":
    unittest.main()
