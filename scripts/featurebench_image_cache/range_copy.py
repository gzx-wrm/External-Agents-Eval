"""Pre-fetch large original blobs with resumable HTTP ranges, then run regctl."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def resolve_blob(repository, digest, size):
    query = urllib.parse.urlencode({"service": "registry.docker.io", "scope": "repository:" + repository + ":pull"})
    request = subprocess.run([
        "/usr/bin/curl", "--fail", "--silent", "--show-error", "--retry", "2",
        "--connect-timeout", "8", "--max-time", "30", "https://auth.docker.io/token?" + query,
    ], capture_output=True, check=True, timeout=100)
    token = json.loads(request.stdout)["token"]
    response = subprocess.run([
        "/usr/bin/curl", "--fail", "--silent", "--show-error", "--location", "--header", "@-",
        "--range", "0-0", "--max-filesize", "1048576", "--connect-timeout", "8", "--max-time", "35",
        "--output", "/dev/null", "--write-out", "%{url_effective}\n%{http_code}\n",
        "https://registry-1.docker.io/v2/" + repository + "/blobs/" + digest,
    ], input=("Authorization: Bearer " + token + "\n").encode(), capture_output=True,
        check=True, timeout=45)
    lines = response.stdout.decode().splitlines()
    if lines[-1] != "206" or urllib.parse.urlsplit(lines[0]).scheme != "https":
        raise RuntimeError("Official blob endpoint did not accept a checked Range request")
    return lines[0]


def aria_download(root, *, url, path, digest, size, connections=8, on_progress=None):
    tools = root / "tools/aria2/unpacked"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(tools / "usr/lib/x86_64-linux-gnu")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    secret = secrets.token_urlsafe(24)
    configuration = root / "acceleration" / ("aria-runtime-" + secrets.token_hex(4) + ".conf")
    configuration.parent.mkdir(parents=True, exist_ok=True)
    options = ["enable-rpc=true", "rpc-listen-all=false", "rpc-listen-port=" + str(port),
               "rpc-secret=" + secret, "continue=true", "always-resume=true", "auto-file-renaming=false",
               "file-allocation=none", "max-concurrent-downloads=1", "split=" + str(connections),
               "max-connection-per-server=" + str(connections), "min-split-size=16M",
               "check-integrity=true", "checksum=sha-256=" + digest.split(":")[1],
               "connect-timeout=15", "timeout=45", "max-tries=5", "retry-wait=2",
               "http-accept-gzip=false", "follow-torrent=false", "follow-metalink=false",
               "enable-color=false", "console-log-level=warn", "summary-interval=0",
               "download-result=hide", "auto-save-interval=5", "stop-with-process=" + str(os.getpid())]
    descriptor = os.open(configuration, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write("\n".join(options) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    logfile = root / "logs" / ("range-" + digest.split(":")[1] + ".log")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def rpc(method, parameters=None):
        body = {"jsonrpc": "2.0", "id": "cache", "method": "aria2." + method,
                "params": ["token:" + secret, *(parameters or [])]}
        request = urllib.request.Request("http://127.0.0.1:" + str(port) + "/jsonrpc",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=5) as response:
            data = json.load(response)
        if "error" in data:
            raise RuntimeError("Local downloader control failed")
        return data["result"]

    process = None
    try:
        with logfile.open("ab") as log:
            process = subprocess.Popen([str(tools / "usr/bin/aria2c"), "--conf-path=" + str(configuration),
                                        "--input-file=-"], env=environment, stdin=subprocess.PIPE,
                                       stdout=log, stderr=subprocess.STDOUT)
            process.stdin.write((url + "\n dir=" + str(path.parent) + "\n out=" + path.name + "\n").encode())
            process.stdin.close()
            startup_deadline = time.monotonic() + 30
            finished = None
            while process.poll() is None:
                try:
                    global_stats = rpc("getGlobalStat")
                    fields = ["gid", "status", "completedLength", "totalLength", "downloadSpeed", "connections", "errorCode"]
                    active = rpc("tellActive", [fields])
                    completed = rpc("tellStopped", [0, 10, fields])
                except OSError:
                    if time.monotonic() > startup_deadline:
                        raise RuntimeError("Downloader control did not become available")
                    time.sleep(0.5)
                    continue
                if on_progress:
                    on_progress({"digest": digest, "active": active, "stopped": completed,
                                 "download_speed": int(global_stats.get("downloadSpeed", 0))})
                if completed and int(global_stats["numActive"]) == 0 and int(global_stats["numWaiting"]) == 0:
                    finished = completed[-1]
                    rpc("shutdown")
                    process.wait(timeout=15)
                    break
                time.sleep(1)
            if not finished or finished.get("status") != "complete":
                raise RuntimeError("Range download did not finish; see " + str(logfile))
        if path.stat().st_size != size or sha256(path) != digest:
            raise RuntimeError("Range-downloaded blob differs from the original digest")
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        configuration.unlink(missing_ok=True)


def prefetch(root, image, plan):
    cache = Path(plan["cache"])
    checkpoint_file = root / "acceleration/checkpoints.json"
    checkpoints = json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else {}
    for digest in sorted(image["blobs"], key=lambda key: plan["unique_blobs"][key]["size"], reverse=True):
        size = plan["unique_blobs"][digest]["size"]
        target = cache / "blobs/sha256" / digest.split(":")[1]
        if size < 64 * 1024**2 or (target.exists() and target.stat().st_size == size):
            continue
        partial = cache / ".resume" / (digest.split(":")[1] + ".part")
        url = resolve_blob(image["repository"], digest, size)

        def progress(data):
            records = data["active"] or data["stopped"]
            retained = 0
            for other, checkpoint in checkpoints.items():
                if other == digest:
                    continue
                if not (cache / "blobs/sha256" / other.split(":")[1]).exists():
                    retained += checkpoint["size"]
            save(root / "range-progress.json", {
                "updated_at": time.time(), "image": image["repository"], "digest": digest,
                "active_completed_bytes": int(records[-1].get("completedLength", 0)) if records else 0,
                "waiting_retained_bytes": retained, "download_speed_bytes": data["download_speed"],
                "connections": int(records[-1].get("connections", 0)) if records else 0,
                "size": size, "status": "downloading", "original_prefix_reused": digest in checkpoints,
            })

        print(json.dumps({"range_download": digest, "size": size, "resume_bytes": partial.stat().st_size if partial.exists() else 0}), flush=True)
        aria_download(root, url=url, path=partial, digest=digest, size=size, on_progress=progress)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(target)
        checkpoint = checkpoints.get(digest)
        if checkpoint:
            old = Path(checkpoint["original_path"])
            if (old.exists() and old.parent == target.parent and old.stat().st_size == checkpoint["size"]
                    and old.stat().st_mtime_ns == checkpoint["mtime_ns"]):
                old.unlink()
        waiting = sum(item["size"] for key, item in checkpoints.items()
                      if not (cache / "blobs/sha256" / key.split(":")[1]).exists())
        save(root / "range-progress.json", {"updated_at": time.time(), "image": image["repository"],
             "digest": digest, "status": "blob_verified", "active_completed_bytes": 0,
             "waiting_retained_bytes": waiting, "download_speed_bytes": 0, "size": size})


def main():
    root = Path(os.environ.get("FEATUREBENCH_CACHE_JOB", Path(__file__).resolve().parent.parent))
    arguments = sys.argv[1:]
    binary = str(root / "bin/regctl")
    if arguments[:2] == ["image", "copy"]:
        plan = json.loads((root / "plan.json").read_text())
        image = next(item for item in plan["images"] if item["source"] == arguments[-2])
        prefetch(root, image, plan)
    os.execv(binary, [binary, *arguments])


if __name__ == "__main__":
    main()
