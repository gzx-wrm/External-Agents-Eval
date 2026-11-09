from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/featurebench_native_mini"
if not SOURCE.exists():
    SOURCE = Path(__file__).resolve().parents[1] / "scripts"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load("iers_common", "common.py")
with patch.dict(sys.modules, {"common": common}):
    module = load("iers_runtime_data", "iers_data.py")


class IERSDataTests(unittest.TestCase):
    def fixture(self, folder, expiry="28 June 2030", iso="2030-06-28T00:00:00+00:00"):
        root = Path(folder)
        data = ("# File expires on " + expiry + "\n57754 1 1 2017 37\n").encode()
        (root / "Leap_Second.dat").write_bytes(data)
        config = root / "iers.json"
        settings = {"file": "Leap_Second.dat", "sha256": hashlib.sha256(data).hexdigest(),
                    "expires": iso, "source": "https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat"}
        config.write_text(json.dumps(settings))
        return config, settings, data

    def test_asset_hash_and_expiry_are_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            config, settings, data = self.fixture(folder)
            self.assertEqual(module.verified_data(config), (settings, data))
            (config.parent / "Leap_Second.dat").write_bytes(data + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                module.verified_data(config)

    def test_expired_file_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = self.fixture(folder, "28 June 2020", "2020-06-28T00:00:00+00:00")
            with self.assertRaisesRegex(ValueError, "expired"):
                module.verified_data(config)

    def test_only_the_pinned_public_data_file_is_uploaded_and_receipted(self):
        with tempfile.TemporaryDirectory() as folder:
            config, settings, data = self.fixture(folder)
            target = "/opt/miniconda3/envs/testbed/lib/python3.12/site-packages/astropy_iers_data/data/Leap_Second.dat"
            container = Mock(name="container")
            container.name = "astropy-reference"
            container.exec_run.side_effect = [
                (0, json.dumps({"path": target, "sha256": "old-data"}).encode()),
                (0, json.dumps(settings["sha256"]).encode())]
            container.put_archive.return_value = True
            receipt = config.parent / "receipt.json"
            result = module.install_leap_seconds(container, config, receipt)
            destination, archive = container.put_archive.call_args.args
            self.assertEqual(destination, str(Path(target).parent))
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                self.assertEqual(tar.getnames(), ["Leap_Second.dat"])
                self.assertEqual(tar.extractfile("Leap_Second.dat").read(), data)
            self.assertEqual(json.loads(receipt.read_text()), result)
            self.assertFalse(result["tests_changed"])
            self.assertFalse(result["package_version_changed"])

    def test_missing_configuration_leaves_other_tasks_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            container = Mock()
            self.assertIsNone(module.install_leap_seconds(container, Path(folder) / "missing.json",
                                                          Path(folder) / "receipt.json"))
            container.exec_run.assert_not_called()
            container.put_archive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
