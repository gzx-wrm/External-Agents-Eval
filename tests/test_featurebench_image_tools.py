from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/featurebench_image_tools/strip_caches.py"
spec = importlib.util.spec_from_file_location("featurebench_cache_repack", PATH)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def layer(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in entries:
            member = name if isinstance(name, tarfile.TarInfo) else tarfile.TarInfo(name)
            member.mode = 0o755
            member.uid = 1000
            member.mtime = 1234567890
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data) if data else None)
    return stream.getvalue()


class CacheRepackTests(unittest.TestCase):
    def test_cache_policy_keeps_installed_packages_and_source_backups(self):
        for path in ["root/.cache/pip/http-v2/file", "opt/miniconda3/pkgs/example.conda",
                     "opt/miniconda3/pkgs/example.tar.bz2", "opt/miniconda3/pkgs/cache/index.json"]:
            self.assertTrue(tool.cache_path(path), path)
        for path in ["opt/miniconda3/envs/pytorch_base/lib/torch.so", "opt/miniconda3/envs/testbed/bin/python",
                     "opt/miniconda3/pkgs/python-3.10/lib/python.py", "root/my_repo/source.py",
                     "root/.cache/huggingface/model.bin", "usr/include/header.h"]:
            self.assertFalse(tool.cache_path(path), path)

    def test_preserves_runtime_bytes_links_and_extended_attributes(self):
        regular = tarfile.TarInfo("opt/miniconda3/envs/testbed/runtime.py")
        regular.pax_headers = {"SCHILY.xattr.user.example": "value"}
        link = tarfile.TarInfo("opt/miniconda3/envs/testbed/alias.py")
        link.type = tarfile.LNKTYPE
        link.linkname = regular.name
        source = layer([(regular, b"runtime code"), (link, b""),
                        ("root/.cache/pip/archive.whl", b"cache payload")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layer.tar"
            report = tool.strip_layer(io.BytesIO(source), path)
            self.assertEqual(report["removed_bytes"], len(b"cache payload"))
            with tarfile.open(path) as archive:
                self.assertEqual(archive.extractfile(regular.name).read(), b"runtime code")
                self.assertEqual(archive.extractfile(link.name).read(), b"runtime code")
                self.assertEqual(archive.getmember(regular.name).uid, 1000)
                self.assertEqual(archive.getmember(regular.name).pax_headers["SCHILY.xattr.user.example"], "value")
                self.assertNotIn("root/.cache/pip/archive.whl", archive.getnames())

    def test_refuses_to_break_runtime_link_into_removed_cache(self):
        member = tarfile.TarInfo("usr/bin/runtime")
        member.type = tarfile.SYMTYPE
        member.linkname = "/root/.cache/pip/runtime"
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "runtime link"):
            tool.strip_layer(io.BytesIO(layer([(member, b"")])), Path(directory) / "bad.tar")

    def test_repack_preserves_layer_boundaries_and_runtime_config(self):
        first = layer([("root/.cache/pip/file", b"download"), ("usr/lib/runtime", b"installed")])
        second = layer([("root/my_repo/source.py", b"original source")])
        config = {"architecture": "amd64", "os": "linux",
                  "config": {"Env": ["PATH=/opt/bin"], "Cmd": ["bash"], "WorkingDir": "/testbed", "User": "0"},
                  "rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(x).hexdigest() for x in [first, second]]},
                  "history": [{"created_by": "base"}, {"created_by": "project"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tar"
            with tarfile.open(source, "w") as archive:
                for name, data in [("config.json", json.dumps(config).encode()), ("a/layer.tar", first), ("b/layer.tar", second),
                                   ("manifest.json", json.dumps([{"Config": "config.json", "Layers": ["a/layer.tar", "b/layer.tar"]}]).encode())]:
                    tool.add_bytes(archive, name, data)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            destination = io.BytesIO()
            audit = tool.repack(source, destination, root / "work", "prototype:test")
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            self.assertEqual(len(audit["layers"]), 2)
            self.assertTrue(audit["layers"][0]["changed"])
            self.assertFalse(audit["layers"][1]["changed"])
            with tarfile.open(fileobj=io.BytesIO(destination.getvalue())) as archive:
                manifest = json.load(archive.extractfile("manifest.json"))[0]
                new = json.load(archive.extractfile(manifest["Config"]))
                for key, value in config["config"].items():
                    self.assertEqual(new["config"][key], value)
                self.assertEqual(new["rootfs"]["diff_ids"][1], config["rootfs"]["diff_ids"][1])
                self.assertEqual(archive.extractfile(manifest["Layers"][1]).read(), second)
            compressed = io.BytesIO()
            compressed_audit = tool.repack(source, compressed, root / "work", "prototype:test",
                                           compressed_workdir=root / "compressed")
            self.assertEqual(compressed_audit["candidate_image_id"], audit["candidate_image_id"])
            with tarfile.open(fileobj=io.BytesIO(compressed.getvalue())) as archive:
                manifest = json.load(archive.extractfile("manifest.json"))[0]
                self.assertEqual(gzip.decompress(archive.extractfile(manifest["Layers"][1]).read()), second)


if __name__ == "__main__":
    unittest.main()
