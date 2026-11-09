"""One-shot image wait → offline preflight → one V6.3 smoke; never resample."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import docker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run = root / "runs/smoke-packaging-20261028-r1"
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "launch.json").exists():
        raise ValueError("This smoke was already launched; inspect existing status instead")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Inject DEEPSEEK_API_KEY through the process environment")
    state = {"pid": os.getpid(), "workflow": "v6.3", "workers": 1, "root": str(root),
             "run": str(run), "model": "DeepSeek V4.1 Flash", "credentials_persisted": False,
             "started_at": datetime.now(timezone.utc).isoformat()}

    def record(phase, **values):
        state.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        target = run / "supervisor-status.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(target)
        print(json.dumps(state), flush=True)

    (run / "launch.json").write_text(json.dumps(state, indent=2))
    client = docker.from_env(timeout=30)
    try:
        task = json.loads((root / "manifests/task-private.json").read_text())
        record("waiting_for_official_image", task_id=task["instance_id"])
        while True:
            try:
                image = client.images.get(task["image_name"])
                break
            except docker.errors.ImageNotFound:
                download = root / "preflight/image-download.json"
                if download.exists() and json.loads(download.read_text()).get("phase") == "failed":
                    record("failed", failed_stage="image_download")
                    raise SystemExit(2) from None
                time.sleep(15)
        expected = json.loads((root / "preflight/registry-source.json").read_text())
        if image.id != expected["config_digest"]:
            raise ValueError("Downloaded image config differs from the official manifest")
        metadata = {"name": task["image_name"], "id": image.id,
                    "repo_digests": image.attrs.get("RepoDigests", []),
                    "size_bytes": image.attrs["Size"], "source_digest": expected["original_digest"]}
        (root / "preflight/image.json").write_text(json.dumps(metadata, indent=2))
        environment = dict(os.environ, PYTHONPATH=str(root / "project") + ":" + str(root / "vendor/FeatureBench"))
        python = "/root/v62-swe-verified100-20261026/venv312/bin/python"
        for mode, directory in [("preflight", root / "runs/preflight-packaging-20261028-r1"),
                                ("generate", run)]:
            record(mode)
            directory.mkdir(parents=True, exist_ok=True)
            command = [python, str(root / "project/scripts/featurebench_v63/smoke.py"), mode,
                       "--root", str(root), "--run", str(directory)]
            with (directory / "launcher.log").open("ab") as log:
                result = subprocess.run(command, env=environment, cwd=root / "project",
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        check=False)
            if result.returncode:
                record("failed", failed_stage=mode, exit_code=result.returncode)
                raise SystemExit(result.returncode)
        record("completed")
    finally:
        client.close()


if __name__ == "__main__":
    main()
