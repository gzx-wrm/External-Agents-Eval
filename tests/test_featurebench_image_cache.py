from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/featurebench_image_cache/cache.py"
SPEC = importlib.util.spec_from_file_location("featurebench_image_cache", SCRIPT)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class BlobIntegrityTests(unittest.TestCase):
    def fixture(self, root, data=b"original compressed layer"):
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        path = root / "blobs/sha256" / digest.split(":")[1]
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        return path, {"blobs": [digest]}, {
            "cache": str(root), "unique_blobs": {digest: {"size": len(data)}}}, digest

    def test_expected_bytes_are_verified_and_never_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path, image, plan, digest = self.fixture(Path(directory))
            before = path.read_bytes()
            verified = {}
            cache.verify_image(image, plan, verified)
            self.assertIn(digest, verified)
            self.assertEqual(path.read_bytes(), before)

    def test_same_length_corruption_is_detected_and_removed_for_refetch(self):
        with tempfile.TemporaryDirectory() as directory:
            path, image, plan, _ = self.fixture(Path(directory))
            path.write_bytes(b"x" * path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "digest differs"):
                cache.verify_image(image, plan, {})
            self.assertFalse(path.exists())

    def test_truncated_object_is_not_accepted_as_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            path, image, plan, _ = self.fixture(Path(directory))
            path.write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "length differs"):
                cache.verify_image(image, plan, {})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
