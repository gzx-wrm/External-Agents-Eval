"""One-shot continuation of image failures that occurred before any model ran."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / "continuation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = json.loads((root / "launch.json").read_text())

    def record(phase, **values):
        data = {"phase": phase, "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(),
                "original_pid": launch["pid"], "workers": 3, "model_resampling": False, **values}
        temporary = root / "continuation-status.tmp"
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(root / "continuation-status.json")

    record("waiting_for_current_batch")
    while True:
        process = Path("/proc") / str(launch["pid"])
        live = process.exists() and (process / "stat").read_text().split()[2] != "Z"
        if not live:
            if not (root / "result.json").exists():
                record("original_batch_incomplete_no_automatic_model_restart")
                return
            break
        time.sleep(20)

    tasks = json.loads((root / "manifests/public-tasks.json").read_text())
    ready, failed = [], []
    for task in tasks:
        output = root / "outputs" / task["task_id"]
        receipt = output / "result.json"
        if not receipt.exists() or (output / "native.traj.json").exists() or (output / "prediction.json").exists():
            continue
        result = json.loads(receipt.read_text())
        if (result.get("status") != "infrastructure_or_stage_error"
                or "images/create" not in result.get("error", "")):
            continue
        record("preparing_unsampled_image", task_id=task["task_id"])
        okay = False
        with (output / "image-recovery.log").open("ab") as log:
            for _ in range(3):
                try:
                    process = subprocess.run(["docker", "pull", task["image"]], stdout=log,
                                             stderr=subprocess.STDOUT, timeout=1800, check=False)
                    okay = process.returncode == 0
                except subprocess.TimeoutExpired:
                    okay = False
                if okay:
                    break
        if okay:
            ready.append(task["task_id"])
        else:
            failed.append(task["task_id"])

    if not ready:
        record("finished", remaining_image_failures=failed, recovered_task_ids=[])
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history = root / "infrastructure-history" / stamp
    history.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / "result.json", history / "first-pass-result.json")
    for name in ready:
        output = root / "outputs" / name
        # Recheck immediately before opening the original slot to its first sample.
        if (output / "native.traj.json").exists() or (output / "prediction.json").exists():
            raise RuntimeError("Refusing to repeat a task that already entered the model")
        destination = history / name
        destination.mkdir()
        (output / "result.json").rename(destination / "result.json")
    record("continuing_unsampled_cases", recovered_task_ids=ready, remaining_image_failures=failed)
    with (root / "logs/image-case-continuation.log").open("ab") as log:
        process = subprocess.run(launch["command"], cwd=root, env=os.environ.copy(), stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False)
    record("finished", recovered_task_ids=ready, remaining_image_failures=failed, exit_code=process.returncode)


if __name__ == "__main__":
    main()
