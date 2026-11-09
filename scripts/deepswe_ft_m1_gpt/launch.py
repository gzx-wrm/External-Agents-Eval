"""Launch after the human has confirmed pipeline and reasoning settings."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run import configuration, digest, load, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoints-file", type=Path, required=True)
    parser.add_argument("--reasoning-effort", choices=["high", "xhigh"], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "launch.json").exists():
        raise ValueError("A launch receipt already exists; inspect it before continuing")
    selection, tasks = load(root)
    readiness = json.loads((root / "preflight/readiness.json").read_text())
    if not readiness["passed"] or readiness["reasoning_effort"] != args.reasoning_effort:
        raise ValueError("Matching offline preflight is required")
    if args.endpoints_file.stat().st_mode & 0o077:
        raise ValueError("Credential file must be readable only by its owner")
    config = configuration(root, args.reasoning_effort, args.endpoints_file)
    original = __import__("yaml").safe_load((root / "configs/frozen-m1.yaml").read_text())
    if config["agent"] != original["agent"] or config["environment"] != original["environment"]:
        raise ValueError("Agent prompts, tools, isolation, or budgets changed")
    native.save(root / "configs/model.json", config)
    selection.update(reasoning_effort=args.reasoning_effort, settings_approval_pending=False,
                     pipeline="frozen FT ledger + native Mini; fresh Solver; no added review")
    native.save(root / "selection.json", selection)
    command = [sys.executable, str(root / "scripts/run.py"), "run", "--root", str(root),
               "--reasoning-effort", args.reasoning_effort, "--endpoints-file", str(args.endpoints_file)]
    (root / "logs").mkdir(exist_ok=True)
    with (root / "logs/runner.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(),
                                   start_new_session=True)
    native.save(root / "launch.json", {
        "experiment_id": "ft_m1_gpt61sol", "pid": process.pid, "status": "started",
        "started_at": native.now(), "root": str(root), "command": command,
        "task_count": len(tasks), "task_ids": [x["task_id"] for x in tasks],
        "workers": 3, "model": "gpt-6.1-sol", "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": 65536, "agent_timeout_sec": 10800, "step_limit": 0, "cost_limit": 0,
        "container_cpus": 2, "container_memory_mb": 8192, "network": "none",
        "evidence_packet_included": False, "review_included": False,
        "frozen_original_ft_ledger": True, "previous_patch_or_conversation_reused": False,
        "official_evaluation": "unchanged DeepSWE verifier immediately after each solve",
        "routing": "round-robin initial endpoint by task; sticky conversation; identical-request failover",
        "api": "streaming Responses with original Mini system prompt passed explicitly",
        "credentials": "host-only private file; no keys in configuration or model input",
        "selection_sha256": digest(root / "selection.json"),
    })
    print(json.dumps({"started": True, "pid": process.pid, "root": str(root), "workers": 3}), flush=True)


if __name__ == "__main__":
    main()
