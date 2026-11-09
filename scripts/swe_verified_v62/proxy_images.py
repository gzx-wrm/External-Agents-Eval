"""Prepare the recorded official image through a host-only proxy before scoring."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import threading

from stream_evaluate import load, now, save


class ProxyImages:
    def __init__(self, root, proxy):
        self.root = root
        self.proxy = proxy
        self.binary = root / "image-tools/crane"
        provenance = load(root / "image-tools/verified-download.json")
        if hashlib.sha256(self.binary.read_bytes()).hexdigest() != provenance["binary_sha256"]:
            raise ValueError("Image download tool differs from the verified release")
        self.lock = threading.Lock()

    @staticmethod
    def validate_layer_cache(directory):
        removed = []
        for path in directory.glob("sha256:*"):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != path.name.removeprefix("sha256:"):
                removed.append(path.name)
                path.unlink()
        return removed

    def available(self, tasks):
        import docker

        client = docker.from_env(timeout=20)
        try:
            ids = {image.id for image in client.images.list(all=True)}
        finally:
            client.close()
        return {task for task in tasks if load(self.root / "outputs" / task / "image.json")["id"] in ids}

    def prepare(self, task):
        import docker

        metadata = load(self.root / "outputs" / task / "image.json")
        client = docker.from_env(timeout=20)
        try:
            try:
                image = client.images.get(metadata["id"])
            except docker.errors.ImageNotFound:
                image = None
            if image is not None:
                repository, tag = metadata["name"].removeprefix("docker.io/").rsplit(":", 1)
                image.tag(repository, tag=tag)
                if client.images.get(metadata["name"]).id != metadata["id"]:
                    raise ValueError("Cached image tag differs from the recorded generation image")
                directory = self.root / "proxy-images" / task
                directory.mkdir(parents=True, exist_ok=True)
                state = {"task": task, "phase": "ready", "cache_hit": True,
                         "expected_image_id": metadata["id"], "image_id_verified": True,
                         "source_digest": metadata["repo_digests"][0], "finished_at": now(), "model_api_calls": 0}
                save(directory / "status.json", state)
                return state
        finally:
            client.close()
        return self._prepare_missing(task)

    def _prepare_missing(self, task):
        import docker

        # One download at a time; the independent official-test pool stays at 3.
        with self.lock:
            metadata = load(self.root / "outputs" / task / "image.json")
            expected = metadata["id"]
            reference = metadata["repo_digests"][0]
            directory = self.root / "proxy-images" / task
            directory.mkdir(parents=True, exist_ok=True)
            state = {"task": task, "phase": "checking_cache", "started_at": now(),
                     "source_digest": reference, "expected_image_id": expected,
                     "proxy": self.proxy, "model_api_calls": 0}
            save(directory / "status.json", state)
            client = docker.from_env(timeout=600)
            try:
                try:
                    image = client.images.get(expected)
                    state["cache_hit"] = True
                except docker.errors.ImageNotFound:
                    if shutil.disk_usage(self.root).free < 25 * 1024**3:
                        raise RuntimeError("Less than 25 GiB free before proxy image download")
                    state.update(phase="proxy_download", cache_hit=False, download_timeout_sec=1200)
                    save(directory / "status.json", state)
                    archive = directory / "image.tar"
                    cache = self.root / "proxy-images/layer-cache"
                    cache.mkdir(exist_ok=True)
                    command = [str(self.binary), "pull", "--platform", "linux/amd64",
                               "--format", "tarball", "--cache_path", str(cache), reference, str(archive)]
                    environment = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
                    environment.update(HTTP_PROXY=self.proxy, HTTPS_PROXY=self.proxy,
                                       ALL_PROXY=self.proxy, NO_PROXY="localhost,127.0.0.1")
                    # Crane can leave a partial cached layer after a transport
                    # error. Verify it before one recovery attempt; never reuse
                    # a truncated layer or rerun any official tests here.
                    for attempt in (1, 2):
                        state["download_attempt"] = attempt
                        save(directory / "status.json", state)
                        try:
                            with (directory / "download.log").open("ab") as log:
                                result = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL,
                                                        stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                            succeeded = result.returncode == 0
                        except subprocess.TimeoutExpired:
                            succeeded = False
                        if succeeded:
                            break
                        state["invalid_cached_layers_removed"] = self.validate_layer_cache(cache)
                    else:
                        raise RuntimeError("Proxy image download failed; inspect download.log")
                    state.update(phase="loading_image", downloaded_at=now(), archive_bytes=archive.stat().st_size)
                    save(directory / "status.json", state)
                    with (directory / "docker-load.log").open("ab") as log:
                        result = subprocess.run(["docker", "load", "--input", str(archive)],
                                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                    if result.returncode:
                        raise RuntimeError("Docker image import failed; inspect docker-load.log")
                    image = client.images.get(expected)
                    archive.unlink()
                if image.id != expected:
                    raise ValueError("Prepared image differs from the original generation image")
                repository, tag = metadata["name"].removeprefix("docker.io/").rsplit(":", 1)
                image.tag(repository, tag=tag)
                if client.images.get(metadata["name"]).id != expected:
                    raise ValueError("Official image tag does not reference the verified image")
                state.update(phase="ready", image_id_verified=True, finished_at=now())
                save(directory / "status.json", state)
                return state
            except Exception as error:
                state.update(phase="failed", error_type=type(error).__name__, error=str(error), finished_at=now())
                save(directory / "status.json", state)
                raise
            finally:
                client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--proxy", required=True)
    args = parser.parse_args()
    print(ProxyImages(args.root.resolve(), args.proxy).prepare(args.task))
