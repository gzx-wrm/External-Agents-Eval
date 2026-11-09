"""Keep experiment image layers cached while existing workers remove task tags."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def normalize(tag):
    return tag.removeprefix("docker.io/")


def retain_once(client, allowed, retained, *, allow_new):
    images = client.images.list(all=True)
    tags = {normalize(tag): image for image in images for tag in image.tags}
    pinned = restored = 0
    for original in allowed:
        image = tags.get(original)
        if image is not None and (allow_new or original in retained):
            image_id = image.id
            repository = "external-eval-cache/swe-" + hashlib.sha256(original.encode()).hexdigest()[:16]
            tag = image_id.removeprefix("sha256:")[:20]
            cache_tag = repository + ":" + tag
            if cache_tag not in tags:
                image.tag(repository, tag=tag)
                pinned += 1
            retained[original] = {"image_id": image_id, "cache_tag": cache_tag}
        elif image is None and original in retained:
            entry = retained[original]
            cached = tags.get(entry["cache_tag"])
            if cached is not None and cached.id == entry["image_id"]:
                repository, tag = original.rsplit(":", 1)
                cached.tag(repository, tag=tag)
                restored += 1
    return pinned, restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=25)
    args = parser.parse_args()
    import docker
    import fcntl

    root = args.root.resolve()
    lock = (root / "image-cache.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    task_ids = json.loads(args.manifest.read_text())["task_ids"]
    allowed = {"swebench/sweb.eval.x86_64." + task.lower().replace("__", "_1776_") + ":latest" for task in task_ids}
    path = root / "image-cache-manifest.json"
    retained = json.loads(path.read_text()) if path.exists() else {}
    retained = {name: value for name, value in retained.items() if name in allowed}
    totals = {"pins_created": 0, "original_tags_restored": 0}
    client = docker.from_env(timeout=20)
    try:
        while True:
            free = shutil.disk_usage(root).free / 1024**3
            error = None
            try:
                pins, restores = retain_once(client, allowed, retained, allow_new=free >= args.min_free_gib)
                totals["pins_created"] += pins
                totals["original_tags_restored"] += restores
                save(path, retained)
            except docker.errors.APIError as exception:
                error = type(exception).__name__
            launch = json.loads((root / "launch.json").read_text())
            alive = process_alive(launch["pid"])
            save(root / "image-cache-status.json", {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "retaining" if alive else "finished_retaining",
                "allowed_task_count": len(task_ids), "retained_image_count": len(retained),
                "free_disk_gib": round(free, 1), "min_free_gib": args.min_free_gib,
                "new_pins_paused_for_disk": free < args.min_free_gib,
                "cache_policy": "retain until explicit cleanup after evaluation and P0 checks",
                "physical_image_copies_created": False, "last_error_type": error, **totals,
            })
            if not alive:
                break
            time.sleep(2)
    finally:
        client.close()


if __name__ == "__main__":
    main()
