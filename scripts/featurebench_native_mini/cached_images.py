"""Load pinned original images from the verified persistent OCI layout."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from common import now, save


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def verified_image(settings, repository):
    plan_path = Path(settings["plan"])
    if digest(plan_path) != settings["plan_sha256"]:
        raise ValueError("Frozen OCI cache plan changed")
    plan = json.loads(plan_path.read_text())
    image = next(item for item in plan["images"] if item["repository"] == repository)
    verified = json.loads(Path(settings["verified_blobs"]).read_text())
    blobs = Path(plan["cache"]) / "blobs/sha256"
    for expected in image["blobs"]:
        path = blobs / expected.split(":")[1]
        info = path.stat()
        size = plan["unique_blobs"][expected]["size"]
        if info.st_size != size:
            raise ValueError("Cached object size differs: " + expected)
        prior = verified.get(expected, {})
        unchanged = prior.get("size") == size and prior.get("mtime_ns") == info.st_mtime_ns
        # Always independently hash the manifest and image configuration.
        if expected in {image["digest"], image["config_digest"]} or not unchanged:
            if digest(path) != expected:
                raise ValueError("Cached object SHA256 differs: " + expected)
    manifest = json.loads((blobs / image["digest"].split(":")[1]).read_text())
    if manifest["config"]["digest"] != image["config_digest"]:
        raise ValueError("OCI manifest/config binding differs")
    return plan, image


def ensure_cached_image(root, repository, client):
    settings = json.loads((root / "configs/image-cache.json").read_text())
    plan, image = verified_image(settings, repository)
    cache_root = Path(settings["plan"]).parent
    log_path = root / "logs" / (repository.rsplit("/", 1)[-1] + "-oci-load.log")
    try:
        loaded = client.images.get(image["config_digest"])
        reused = True
    except __import__("docker").errors.ImageNotFound:
        reused = False
        data_root = Path(client.info()["DockerRootDir"])
        if shutil.disk_usage(data_root).free < 5 * 1024**3:
            raise RuntimeError("Insufficient space in the dedicated Docker image store")
        environment = os.environ.copy()
        environment.update(REGCTL_CONFIG=str(cache_root / "regctl.json"),
                           DOCKER_CONFIG=str(cache_root / "empty-docker-config"))
        command = [str(cache_root / "bin/regctl"), "image", "export", image["destination"],
                   "--name", repository + ":latest"]
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("ab") as log:
            exporter = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, env=environment)
            try:
                loader = subprocess.Popen(["docker", "image", "load"], stdin=exporter.stdout,
                                          stdout=log, stderr=subprocess.STDOUT, env=environment)
                exporter.stdout.close()
                load_code = loader.wait()
                export_code = exporter.wait(timeout=60)
                if load_code or export_code:
                    raise RuntimeError("OCI export/Docker load failed; see " + str(log_path))
            finally:
                if exporter.poll() is None:
                    exporter.terminate()
                    exporter.wait(timeout=15)
        loaded = client.images.get(image["config_digest"])
    if loaded.id != image["config_digest"]:
        raise ValueError("Loaded image ID differs from frozen original configuration")
    loaded.tag(repository, tag="latest")
    save(root / "preflight/images" / (repository.rsplit("/", 1)[-1] + ".json"), {
        "original_repository": repository, "original_digest": image["digest"],
        "config_digest": image["config_digest"], "image_id": loaded.id,
        "download_route": "persistent-oci-cache", "network_download": False,
        "existing_docker_image_reused": reused, "cache": plan["cache"], "verified_at": now(),
    })
    return loaded.id
