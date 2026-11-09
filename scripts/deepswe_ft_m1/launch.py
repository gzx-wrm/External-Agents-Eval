"""Start an approved ft_m1 batch; read credentials only from standard input."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    if (root / "launch.json").exists():
        raise SystemExit("ft_m1 already has a launch receipt; inspect it before resuming")
    selection = json.loads((root / "selection.json").read_text())
    base = Path(selection["m0_root"])
    for relative, expected in selection["source_sha256"].items():
        if hashlib.sha256((base / relative).read_bytes()).hexdigest() != expected:
            raise SystemExit("Frozen M0 reference changed: " + relative)
    for record in selection["tasks"]:
        if hashlib.sha256(Path(record["m0_result_path"]).read_bytes()).hexdigest() != record["m0_result_sha256"]:
            raise SystemExit("An original M0 result changed")
    if not json.loads((root / "preflight/readiness.json").read_text())["passed"]:
        raise SystemExit("ft_m1 preflight has not passed")
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("A host-side API credential is required")
    request = urllib.request.Request("https://api.deepseek.com/v1/models",
                                     headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            models = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit("DeepSeek authentication failed: HTTP " + str(error.code)) from None
    if not any(item.get("id") == "deepseek-flash" for item in models.get("data", [])):
        raise SystemExit("The approved deepseek-flash model alias is unavailable")
    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = key
    command = ["/root/mini-swe-verified100-20261027/venv312/bin/python",
               str(root / "scripts/run.py"), "run", "--root", str(root), "--workers", "3"]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [*sorted((root / "scripts").glob("*.py")),
                           *sorted((root / "configs").glob("*.yaml")), root / "selection.json"]}
    with (root / "logs/runner.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {
        "experiment_id": "ft_m1", "status": "started", "pid": process.pid,
        "started_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "command": command, "task_count": len(selection["tasks"]),
        "m0_reward_filter": selection.get("m0_reward_filter", 0),
        "task_ids": [t["task_id"] for t in selection["tasks"]],
        "workers": 3, "model": "DeepSeek V4.1 Flash", "model_alias": "deepseek-flash",
        "reasoning_effort": "high", "max_output_tokens": 65536,
        "agent_timeout_sec": 10800, "step_limit": 0, "cost_limit": 0,
        "container_cpus": 2, "container_memory_mb": 8192,
        "scheduling": "each slot solves and immediately runs the unchanged official verifier",
        "input_change": "append original FT pre-Solver ledger and approved brief guidance",
        "evidence_packet_included": False, "m0_conversation_or_patch_reused": False,
        "credentials_persisted": False, "authentication_preflight_http_status": status,
        "source_sha256": hashes,
    }
    (root / "launch.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "source_sha256"},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
