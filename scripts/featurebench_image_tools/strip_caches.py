"""Repack a Docker-save archive without installation/download caches.

Installed environments, CUDA, project backups and image runtime configuration
are retained. This produces a NEW prototype image, never an in-place rewrite.
Original layer boundaries are retained so identically cleaned common layers can
still be shared between project images. No application code or package changes.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import posixpath
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path, PurePosixPath


def cache_path(name):
    path = PurePosixPath(posixpath.normpath(name.lstrip("/")))
    text = path.as_posix()
    for prefix in ("root/.cache/pip", "opt/miniconda3/pkgs/cache"):
        if text == prefix or text.startswith(prefix + "/"):
            return True
    return (path.parent.as_posix() == "opt/miniconda3/pkgs"
            and (path.name.endswith(".conda") or path.name.endswith(".tar.bz2")))


def metadata(member):
    # PAX spellings for these values are already represented by TarInfo fields.
    represented = {"path", "linkpath", "mtime", "uid", "gid", "uname", "gname", "size"}
    return {"name": member.name, "type": member.type.hex(), "size": member.size,
            "mode": member.mode, "uid": member.uid, "gid": member.gid,
            "uname": member.uname, "gname": member.gname, "mtime": float(member.mtime),
            "linkname": member.linkname, "devmajor": member.devmajor, "devminor": member.devminor,
            "pax": {k: v for k, v in member.pax_headers.items() if k not in represented}}


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data)
        return data


class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            if stream.write(data) != len(data):
                raise OSError("Short write while preserving the candidate archive")
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def add_record(state, member, content_hash):
    value = json.dumps(metadata(member), sort_keys=True, ensure_ascii=False).encode()
    state.update(len(value).to_bytes(8, "big"))
    state.update(value)
    state.update(bytes.fromhex(content_hash))


def stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def retained_digest(path):
    state = hashlib.sha256()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            content = hashlib.sha256()
            if member.isreg():
                with archive.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        content.update(chunk)
            add_record(state, member, content.hexdigest())
    return state.hexdigest()


def strip_layer(stream, output, *, reserve_bytes=0):
    state = hashlib.sha256()
    removed_bytes = removed_entries = kept_entries = 0
    with tarfile.open(fileobj=stream, mode="r|*") as source, tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as target:
        for member in source:
            if cache_path(member.name):
                removed_entries += 1
                removed_bytes += member.size
                continue
            if member.issparse():
                raise ValueError("Sparse file needs explicit handling: " + member.name)
            if member.islnk() or member.issym():
                name = member.linkname if member.islnk() or member.linkname.startswith("/") else posixpath.join(
                    posixpath.dirname(member.name), member.linkname)
                if cache_path(name):
                    raise ValueError("Retained runtime link points into a removed cache: " + member.name)
            if reserve_bytes and shutil.disk_usage(Path(output).parent).free < member.size + reserve_bytes:
                raise RuntimeError("Insufficient work space; refusing to fill the source filesystem")
            reader = HashReader(source.extractfile(member)) if member.isreg() else None
            target.addfile(copy.copy(member), reader)
            add_record(state, member, reader.hash.hexdigest() if reader else hashlib.sha256().hexdigest())
            kept_entries += 1
    after = retained_digest(output)
    if after != state.hexdigest():
        raise ValueError("Retained file content or metadata changed during cache filtering")
    return {"removed_bytes": removed_bytes, "removed_entries": removed_entries,
            "kept_entries": kept_entries, "retained_files_sha256": after,
            "retained_content_and_metadata_verified": True}


def add_bytes(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


def repack(source_path, destination, workdir, tag, *, reserve_bytes=0, progress=None, compressed_workdir=None):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if compressed_workdir:
        compressed_workdir = Path(compressed_workdir)
        compressed_workdir.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(source_path, "r:*") as source, tarfile.open(fileobj=destination, mode="w|") as target:
        manifests = json.load(source.extractfile("manifest.json"))
        if len(manifests) != 1:
            raise ValueError("Export exactly one source image")
        manifest = manifests[0]
        config_data = source.extractfile(manifest["Config"]).read()
        config = json.loads(config_data)
        original_config = copy.deepcopy(config)
        original_id = "sha256:" + hashlib.sha256(config_data).hexdigest()
        diffs = config["rootfs"]["diff_ids"]
        if len(diffs) != len(manifest["Layers"]):
            raise ValueError("Image configuration and layer count differ")
        paths, new_diffs = [], []
        for number, (name, old_diff) in enumerate(zip(manifest["Layers"], diffs, strict=True)):
            temporary = workdir / f"layer-{number:02d}.tar"
            compressed = None
            try:
                with source.extractfile(name) as stream:
                    record = strip_layer(stream, temporary, reserve_bytes=reserve_bytes)
                changed = record["removed_entries"] > 0
                if changed:
                    with temporary.open("rb") as stream:
                        digest = stream_sha256(stream)
                    new_diff = "sha256:" + digest
                    size = temporary.stat().st_size
                    stream = temporary.open("rb")
                else:
                    new_diff = old_diff
                    size = source.getmember(name).size
                    stream = source.extractfile(name)
                if compressed_workdir:
                    compressed = compressed_workdir / f"layer-{number:02d}.tar.gz"
                    with stream, compressed.open("wb") as out, gzip.GzipFile(filename="", fileobj=out, mode="wb", compresslevel=1, mtime=0) as gz:
                        shutil.copyfileobj(stream, gz, length=1024 * 1024)
                    stream = compressed.open("rb")
                    size = compressed.stat().st_size
                path = f"layers/{number:02d}/{new_diff.removeprefix('sha256:')}/layer.tar" + (".gz" if compressed else "")
                member = tarfile.TarInfo(path)
                member.size = size
                with stream:
                    target.addfile(member, stream)
                paths.append(path)
                new_diffs.append(new_diff)
                record.update(layer=number, changed=changed, original_diff_id=old_diff, new_diff_id=new_diff)
                if compressed:
                    record["transfer_compressed_bytes"] = size
                records.append(record)
                if progress:
                    progress(record)
            finally:
                temporary.unlink(missing_ok=True)
                if compressed:
                    compressed.unlink(missing_ok=True)
        config["rootfs"]["diff_ids"] = new_diffs
        labels = config["config"].setdefault("Labels", {}) or {}
        labels.update({"external_eval.prototype": "cache-only-v1", "external_eval.source_image": original_id})
        config["config"]["Labels"] = labels
        config.setdefault("history", []).append({"created_by": "offline cache-only repack v1", "empty_layer": True})
        new_config = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        new_id = hashlib.sha256(new_config).hexdigest()
        add_bytes(target, new_id + ".json", new_config)
        add_bytes(target, "manifest.json", json.dumps([{"Config": new_id + ".json", "RepoTags": [tag], "Layers": paths}]).encode())
    before, after = copy.deepcopy(original_config["config"]), copy.deepcopy(config["config"])
    before.pop("Labels", None)
    after.pop("Labels", None)
    if before != after:
        raise ValueError("Image runtime configuration changed")
    return {"source_image_id": original_id, "candidate_image_id": "sha256:" + new_id,
            "tag": tag, "layers": records, "removed_cache_bytes": sum(r["removed_bytes"] for r in records),
            "runtime_config_preserved": True, "layer_boundaries_preserved": True,
            "source_archive_modified": False, "active_experiment_adopted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--docker-host", required=True)
    parser.add_argument("--target-storage", type=Path, required=True)
    parser.add_argument("--archive-output", type=Path)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    pressure = []
    archive_file = None
    archive_partial = None
    if args.archive_output:
        if args.archive_output.exists():
            raise FileExistsError("Candidate archive already exists")
        archive_partial = args.archive_output.with_suffix(args.archive_output.suffix + ".partial")
        archive_file = archive_partial.open("xb")
    with (args.workdir / "docker-load.log").open("wb") as log:
        process = subprocess.Popen(["docker", "-H", args.docker_host, "image", "load"],
                                   stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)

        def guard():
            while not stopped.wait(1):
                if shutil.disk_usage(args.target_storage).free < 8 * 1024**3:
                    pressure.append("Prototype loader stopped to preserve 8GiB for the active experiment")
                    process.terminate()
                    return
                if args.archive_output and shutil.disk_usage(args.archive_output.parent).free < 3 * 1024**3:
                    pressure.append("Prototype loader stopped to preserve 3GiB on the archive filesystem")
                    process.terminate()
                    return

        watcher = threading.Thread(target=guard, daemon=True)
        watcher.start()
        try:
            def progress(record):
                print(json.dumps({"layer": record["layer"], "cache_removed_MiB": round(record["removed_bytes"] / 1024**2, 2),
                                  "retained_content_verified": True}), flush=True)

            destination = TeeWriter(process.stdin, archive_file) if archive_file else process.stdin
            report = repack(args.source, destination, args.workdir, args.tag,
                            reserve_bytes=2 * 1024**3, progress=progress,
                            compressed_workdir=args.target_storage / "image-optimization-20261030-r1/transfer")
            process.stdin.close()
            if process.wait() or pressure:
                raise RuntimeError("Docker load failed; see docker-load.log. " + "; ".join(pressure))
            if archive_file:
                archive_file.close()
                archive_partial.replace(args.archive_output)
                report["candidate_archive"] = str(args.archive_output)
                report["candidate_archive_bytes"] = args.archive_output.stat().st_size
            (args.workdir / "repack-audit.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({"completed": True, "candidate_image_id": report["candidate_image_id"],
                              "cache_removed_GiB": round(report["removed_cache_bytes"] / 1024**3, 3)}), flush=True)
        finally:
            stopped.set()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            if archive_file and not archive_file.closed:
                archive_file.close()
            if archive_partial:
                archive_partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
