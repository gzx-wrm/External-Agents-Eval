"""Persist the original Lite images in one deduplicated OCI layout using regctl."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _fetch_frozen_manifest(item):
    record, binary, environment = item
    repo = record["original_repository"]
    reference = "docker.io/" + repo + "@" + record["original_digest"]
    result = subprocess.run([binary, "manifest", "get", reference, "--format", "raw-body"],
                            env=environment, check=True, capture_output=True, timeout=120)
    raw = result.stdout
    if "sha256:" + hashlib.sha256(raw).hexdigest() != record["original_digest"]:
        raise ValueError("Registry returned a different frozen manifest")
    manifest = json.loads(raw)
    if manifest["config"]["digest"] != record["config_digest"]:
        raise ValueError("Original image configuration digest changed")
    return record, raw, manifest


def frozen_manifest(item):
    for attempt in range(4):
        try:
            return _fetch_frozen_manifest(item)
        except (OSError, subprocess.SubprocessError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def prepare(root, source, cache, mount):
    if (root / "plan.json").exists():
        raise FileExistsError("A plan already exists; inspect or resume it")
    if not mount.is_mount() or os.stat(mount).st_dev == os.stat(root).st_dev:
        raise ValueError("The dedicated cache filesystem must be mounted")
    task_file = source / "manifests/tasks-public.json"
    tasks = json.loads(task_file.read_text())
    counts = {}
    for task in tasks:
        counts[task["image_name"]] = counts.get(task["image_name"], 0) + 1
    if len(tasks) != 30 or len(counts) != 13:
        raise ValueError("Expected the original Lite30 and thirteen images")
    records = [json.loads((source / "preflight/images" / (name.rsplit("/", 1)[-1] + ".json")).read_text())
               for name in sorted(counts)]
    images, blobs = [], {}
    environment = os.environ.copy()
    environment["REGCTL_CONFIG"] = str(root / "regctl.json")
    environment["DOCKER_CONFIG"] = str(root / "empty-docker-config")
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifests = list(pool.map(frozen_manifest, [
            (record, str(root / "bin/regctl"), environment) for record in records]))
    for record, raw, manifest in manifests:
        digest = record["original_digest"]
        name = record["original_repository"]
        (root / "frozen-manifests").mkdir(exist_ok=True)
        (root / "frozen-manifests" / (digest.split(":")[1] + ".json")).write_bytes(raw)
        descriptors = [manifest["config"], *manifest["layers"], {"digest": digest, "size": len(raw)}]
        for descriptor in descriptors:
            previous = blobs.setdefault(descriptor["digest"], {"size": descriptor["size"]})
            if previous["size"] != descriptor["size"]:
                raise ValueError("One blob digest has inconsistent sizes")
        images.append({"repository": name, "digest": digest, "config_digest": record["config_digest"],
                       "task_count": counts[name], "source": "docker.io/" + name + "@" + digest,
                       "destination": "ocidir://" + str(cache) + ":" + name.rsplit("/", 1)[-1],
                       "compressed_bytes": sum(layer["size"] for layer in manifest["layers"]),
                       "blobs": [x["digest"] for x in descriptors]})
    total = sum(value["size"] for value in blobs.values())
    if shutil.disk_usage(mount).free < total + 5 * 1024**3:
        raise ValueError("Insufficient space for the deduplicated cache and 5GiB reserve")
    cache.mkdir(parents=True, exist_ok=True)
    images.sort(key=lambda image: image["compressed_bytes"])
    plan = {"created_at": now(), "task_count": len(tasks), "image_count": len(images),
            "mount": str(mount), "cache": str(cache), "source_run": str(source),
            "task_manifest_sha256": sha256(task_file), "total_unique_bytes": total,
            "unique_blobs": blobs, "images": images, "model_api_calls": 0,
            "regctl": str(root / "bin/regctl"), "script_sha256": sha256(Path(__file__))}
    save(root / "plan.json", plan)
    save(cache.parent / "lite-cache-plan.json", plan)
    print(json.dumps({"prepared": True, "images": len(images), "unique_GiB": round(total / 1024**3, 3)}), flush=True)


def verify_image(image, plan, verified):
    base = Path(plan["cache"]) / "blobs/sha256"
    for digest in image["blobs"]:
        path = base / digest.split(":")[1]
        size = plan["unique_blobs"][digest]["size"]
        info = path.stat()
        previous = verified.get(digest)
        if info.st_size != size:
            path.unlink()
            raise ValueError("Cached blob length differs: " + digest)
        if previous and previous["size"] == size and previous["mtime_ns"] == info.st_mtime_ns:
            continue
        if sha256(path) != digest:
            path.unlink()  # Only this corrupt, content-addressed cache object.
            raise ValueError("Cached blob digest differs; removed invalid cache object: " + digest)
        verified[digest] = {"size": size, "mtime_ns": info.st_mtime_ns, "verified_at": now()}


def run(root):
    plan = json.loads((root / "plan.json").read_text())
    cache, mount = Path(plan["cache"]), Path(plan["mount"])
    if not mount.is_mount() or sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("Cache mount or frozen downloader source differs")
    lock_file = (root / "download.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verified_file = root / "verified-blobs.json"
    verified = json.loads(verified_file.read_text()) if verified_file.exists() else {}
    state = {"status": "running", "started_at": now(), "pid": os.getpid(), "images_ready": 0,
             "image_count": len(plan["images"]), "total_unique_bytes": plan["total_unique_bytes"],
             "cache": str(cache), "model_api_calls": 0,
             "images": {x["repository"]: {"status": "queued"} for x in plan["images"]}}
    mutex, stopped = threading.Lock(), threading.Event()
    process = None
    pressure = []

    def publish():
        bytes_present, objects_present, directory_bytes = 0, 0, 0
        blob_dir = cache / "blobs/sha256"
        for digest, descriptor in plan["unique_blobs"].items():
            path = blob_dir / digest.split(":")[1]
            try:
                present = path.is_file() and path.stat().st_size == descriptor["size"]
            except FileNotFoundError:
                present = False
            if present:
                bytes_present += descriptor["size"]
                objects_present += 1
        for folder, _, names in os.walk(cache):
            for name in names:
                try:
                    directory_bytes += (Path(folder) / name).stat().st_size
                except FileNotFoundError:
                    pass  # Atomic rename while a blob completes.
        transfer_file = root / "range-progress.json"
        transfer = json.loads(transfer_file.read_text()) if transfer_file.exists() else None
        with mutex:
            state.update(updated_at=now(), complete_blob_bytes=bytes_present,
                         complete_blob_count=objects_present, cache_directory_bytes=directory_bytes,
                         free_bytes=shutil.disk_usage(mount).free,
                         images_ready=sum(x["status"] == "ready" for x in state["images"].values()))
            if transfer:
                active_digest = transfer.get("digest", "").split(":")[-1]
                active = int(transfer.get("active_completed_bytes", 0))
                if (cache / "blobs/sha256" / active_digest).is_file():
                    active = 0
                state.update(transfer_mode="segmented-http", range_transfer=transfer,
                             effective_downloaded_bytes=bytes_present + active + int(transfer.get("waiting_retained_bytes", 0)),
                             download_speed_bytes=(transfer.get("download_speed_bytes", 0)
                                                   if time.time() - transfer.get("updated_at", 0) < 20 else 0))
            save(root / "monitor-status.json", state)

    def reporter():
        while not stopped.wait(5):
            publish()
            if shutil.disk_usage(mount).free < 5 * 1024**3:
                pressure.append("Download stopped to retain 5GiB cache disk reserve")
                if process and process.poll() is None:
                    process.terminate()
                return

    environment = os.environ.copy()
    environment["REGCTL_CONFIG"] = str(root / "regctl.json")
    environment["DOCKER_CONFIG"] = str(root / "empty-docker-config")
    temporary = cache / ".transfer-tmp"
    temporary.mkdir(exist_ok=True)
    environment["TMPDIR"] = str(temporary)
    watcher = threading.Thread(target=reporter, daemon=True)
    watcher.start()
    try:
        for image in plan["images"]:
            name = image["repository"]
            for attempt in range(1, 6):
                with mutex:
                    state["images"][name] = {"status": "downloading", "attempt": attempt,
                                             "started_at": now(), "digest": image["digest"]}
                publish()
                log_file = root / "logs" / (name.rsplit("/", 1)[-1] + ".log")
                log_file.parent.mkdir(exist_ok=True)
                with log_file.open("ab") as log:
                    command = [plan["regctl"], "image", "copy", "--force-recursive", image["source"], image["destination"]]
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    exit_code = process.wait()
                if pressure:
                    raise RuntimeError(pressure[0])
                try:
                    if exit_code:
                        raise RuntimeError("regctl image copy exited with " + str(exit_code))
                    with mutex:
                        state["images"][name]["status"] = "verifying"
                    verify_image(image, plan, verified)
                    actual = subprocess.check_output([plan["regctl"], "image", "digest", image["destination"]],
                                                     env=environment, text=True).strip()
                    if actual != image["digest"]:
                        raise ValueError("Cached image manifest digest differs from original")
                    save(verified_file, verified)
                    with mutex:
                        state["images"][name].update(status="ready", verified_at=now(),
                                                     original_manifest_digest_preserved=True)
                    print(json.dumps({"image": name, "status": "ready", "verified_at": now()}), flush=True)
                    break
                except Exception as error:
                    with mutex:
                        state["images"][name].update(status="retry_wait" if attempt < 5 else "failed",
                                                     error_type=type(error).__name__, error=str(error))
                    print(json.dumps({"image": name, "attempt": attempt, "error": str(error)}), flush=True)
                    publish()
                    if attempt < 5:
                        time.sleep(min(30, 2**attempt))
            publish()
        with mutex:
            state.update(status="completed" if all(x["status"] == "ready" for x in state["images"].values())
                         else "completed_with_errors", finished_at=now())
        publish()
        save(root / "result.json", state)
    except Exception as error:
        with mutex:
            state.update(status="blocked" if pressure else "failed", error_type=type(error).__name__,
                         error=str(error), finished_at=now())
        raise
    finally:
        if process and process.poll() is None:
            process.terminate()
        stopped.set()
        watcher.join(timeout=10)
        publish()
        lock_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--mount", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root.resolve(), args.source.resolve(), args.cache.resolve(), args.mount.resolve())
    else:
        run(args.root.resolve())


if __name__ == "__main__":
    main()
