"""Launch an explicitly approved review batch with a host-only stdin credential."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--preflight-pid", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "launch.json").exists():
        raise SystemExit("Launch receipt exists; inspect its PID before attempting recovery")
    runner_path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("ft_review_launch_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    native = runner.load_native(root)
    selection = runner.validate_selection(root, native)
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("A host-side API credential is required on standard input")
    request = urllib.request.Request(
        "https://api.deepseek.com/v1/models", headers={"Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, models = response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit("DeepSeek authentication failed: HTTP " + str(error.code)) from None
    if not any(item.get("id") == "deepseek-flash" for item in models.get("data", [])):
        raise SystemExit("The frozen deepseek-flash model alias is unavailable")
    if args.preflight_pid:
        runner.save(root / "startup-status.json", {
            "status": "waiting_for_preflight", "pid": os.getpid(),
            "preflight_pid": args.preflight_pid, "workers": 3,
            "task_count": len(selection["tasks"]), "model_api_calls": 0,
            "credentials_persisted": False,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        while not (root / "preflight/readiness.json").exists():
            command = Path(f"/proc/{args.preflight_pid}/cmdline")
            if not command.exists() or str(root).encode() not in command.read_bytes():
                runner.save(root / "startup-status.json", {
                    "status": "preflight_failed", "pid": os.getpid(),
                    "model_api_calls": 0, "reason": "Preflight ended without readiness receipt",
                })
                raise SystemExit("Preflight ended without readiness receipt; inspect preflight.log")
            time.sleep(5)
    ready = runner.read(root / "preflight/readiness.json")
    if not ready["passed"] or ready["selection_sha256"] != runner.sha256_json(selection):
        raise SystemExit("Preflight does not match the frozen selection")
    # Recheck immediately before dispatch; this may have waited for image pulls.
    runner.validate_selection(root, native)
    env = os.environ.copy()
    env.update(DEEPSEEK_API_KEY=key, DO_NOT_TRACK="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    command = [sys.executable, str(runner_path), "run", "--root", str(root), "--workers", "3"]
    (root / "logs").mkdir(exist_ok=True)
    with (root / "logs/runner.log").open("ab") as log:
        process = subprocess.Popen(
            command, cwd=runner.PROJECT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    receipt = {
        "experiment_id": "ft_m1_review", "status": "started", "pid": process.pid,
        "started_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "project": str(runner.PROJECT), "python": sys.executable, "command": command,
        "task_count": len(selection["tasks"]),
        "task_ids": [row["task_id"] for row in selection["tasks"]], "workers": 3,
        "model": "DeepSeek V4.1 Flash", "reasoning_effort": "high", "max_output_tokens": 65536,
        "start_at": "I/R", "ledger_reused": True, "p0_reused": True,
        "lane_deadline_sec": 2700, "role_timeout_sec": 1800, "mini_step_limit": 250,
        "scoring": "unchanged official DeepSWE verifier immediately after each review",
        "credentials_persisted": False, "authentication_preflight_http_status": status,
        "selection_sha256": runner.sha256_json(selection),
        "launcher_sha256": runner.sha256_bytes(Path(__file__).read_bytes()),
    }
    runner.save(root / "launch.json", receipt)
    runner.save(root / "startup-status.json", {
        "status": "review_started", "pid": os.getpid(), "review_pid": process.pid,
        "task_count": len(selection["tasks"]), "workers": 3,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
