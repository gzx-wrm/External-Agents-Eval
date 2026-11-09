"""Launch once, with the model credential received through stdin only."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from common import now, save


def main():
    root = Path(sys.argv[1]).resolve()
    if (root / "launch.json").exists():
        raise SystemExit("A launch receipt already exists; inspect the same run before resuming")
    if not json.loads((root / "preflight/core-readiness.json").read_text())["passed"]:
        raise SystemExit("The real native Mini protocol/container preflight has not passed")
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("A transient host-side API credential is required")
    request = urllib.request.Request("https://api.deepseek.com/v1/models",
                                     headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, models = response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit("Model authentication failed: HTTP " + str(error.code)) from None
    if not any(model.get("id") == "deepseek-flash" for model in models.get("data", [])):
        raise SystemExit("The approved DeepSeek model alias is unavailable")
    environment = os.environ.copy()
    environment.update(DEEPSEEK_API_KEY=key, DOCKER_HOST="unix:///run/featurebench-lite-docker.sock",
                       PYTHONPATH="/root/v63-featurebench-20261028/vendor/FeatureBench")
    command = ["/root/v62-swe-verified100-20261026/venv312/bin/python",
               str(root / "scripts/run.py"), "--root", str(root)]
    sources = [*sorted((root / "scripts").glob("*.py")),
               *sorted((root / "configs").glob("*.yaml")),
               *sorted((root / "configs").glob("*.json")),
               root / "manifests/tasks-private.json", root / "manifests/tasks-public.json"]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    with (root / "logs/runner.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {
        "experiment_id": "native_mini_featurebench_lite", "status": "started",
        "pid": process.pid, "started_at": now(), "root": str(root), "command": command,
        "task_count": len(json.loads((root / "manifests/tasks-private.json").read_text())),
        "workers": 3, "model": "DeepSeek V4.1 Flash", "model_alias": "deepseek-flash",
        "reasoning_effort": "high", "max_output_tokens": 65536, "agent_timeout_sec": 10800,
        "step_limit": 0, "cost_limit": 0, "tools": ["bash"], "ledger_included": False,
        "container_cpus": 2, "container_memory_mb": 6144, "network": "none",
        "scheduling": "group images for disk capacity; three slots each validate CPU, solve, then officially score",
        "cpu_policy": "no model calls unless positive reference and negative control pass; GPU dependency deferred",
        "level2_cpu_reference": "original pristine implementation test probe, not an official Lv2 gold score",
        "docker_host": environment["DOCKER_HOST"], "credentials_persisted": False,
        "authentication_preflight_http_status": status, "source_sha256": hashes,
    }
    cpu_config = root / "configs/cpu-evaluation.json"
    receipt.update(solver_network="none", evaluation_configuration=(
        json.loads(cpu_config.read_text()) if cpu_config.exists() else {"network": "none"}))
    save(root / "launch.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "source_sha256"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
