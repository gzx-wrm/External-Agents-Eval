"""Launch the prepared comparison; accept a host-side credential on stdin."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

import batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "launch.json").exists():
        raise SystemExit("Launch receipt exists; inspect its PID/results before recovery")
    selection, _, _ = batch.validate(root)
    ready = batch.read(root / "preflight/readiness.json")
    if not ready["passed"] or ready["selection_sha256"] != batch.sha256_json(selection):
        raise SystemExit("Preflight does not match the frozen selection")
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("Provide the API credential on stdin")
    pp = batch.read(root / "configs/codescout.json")
    request = Request(pp["base_url"].rstrip("/") + "/models",
                      headers={"Authorization": "Bearer " + key})
    with urlopen(request, timeout=30) as response:
        models = json.load(response)
    if not any(model["id"] == pp["model"] for model in models.get("data", [])):
        raise SystemExit("Configured model alias is unavailable")
    env = os.environ.copy()
    env.update(DEEPSEEK_API_KEY=key, DO_NOT_TRACK="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    command = [sys.executable, str(Path(batch.__file__).resolve()), "run", "--root", str(root),
               "--workers", "3"]
    (root / "logs").mkdir(exist_ok=True)
    with (root / "logs/runner.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=batch.PROJECT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    batch.save(root, "launch.json", {
        "pid": process.pid, "started_at": batch.now(), "command": command,
        "root": str(root), "experiment_id": "codescout_native_mini",
        "task_ids": [record["task_id"] for record in selection["tasks"]],
        "task_count": selection["task_count"], "workers": 3,
        "model": "deepseek-flash", "reasoning_effort": "high", "max_tokens": 65536,
        "agent_timeout_sec": 10800, "preprocessing_timeout_sec": pp["max_seconds"],
        "verifier_timeout_sec": 1800, "cpus": 2, "memory_mb": 8192,
        "scheduling": "each slot preprocesses, solves, then immediately runs official evaluation",
        "credentials_persisted": False, "selection_sha256": batch.sha256_json(selection),
    })
    print(json.dumps({"pid": process.pid, "root": str(root), "task_count": selection["task_count"],
                      "workers": 3, "started_at": batch.now()}), flush=True)


if __name__ == "__main__":
    main()
