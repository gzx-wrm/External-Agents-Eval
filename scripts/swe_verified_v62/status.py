"""Read one smoke batch's stage progress without exposing model text or credentials."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def inspect(root):
    launch = json.loads((root / "launch.json").read_text())
    try:
        os.kill(launch["pid"], 0)
        alive = True
    except OSError:
        alive = False
    tasks = []
    for task_id in launch["task_ids"]:
        output = root / "outputs" / task_id
        workflow = output / "agent/v6_review_repair"
        row = {"task": task_id, "status": "running" if alive else "not_running"}
        result = output / "result.json"
        if result.exists():
            payload = json.loads(result.read_text())
            row["status"] = payload["status"]
            if "error" in payload:
                row["error"] = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", payload["error"])
        progress = sorted(workflow.glob("**/progress.json"), key=lambda p: p.stat().st_mtime)
        if progress:
            latest = progress[-1]
            details = json.loads(latest.read_text())
            row.update(stage=str(latest.parent.relative_to(workflow)), calls=details["model_calls"],
                       call_state=details["state"], updated_at=details["updated_at"])
            if "error_type" in details:
                row["query_error_type"] = details["error_type"]
        packet = workflow / "v6.2/evidence.json"
        if packet.exists():
            row["evidence_spans"] = len(json.loads(packet.read_text())["evidence"])
        ledger = workflow / "v6/pre-solver-ledger.json"
        if ledger.exists():
            envelope = json.loads(ledger.read_text())
            row["ledger_quality"] = envelope["ledger_quality"]
            if envelope.get("ledger"):
                row["ledger_sizes"] = {field: len(envelope["ledger"][field]) for field in
                                       ("obligations", "preservation_obligations", "open_questions")}
        ft = workflow / "v6.2/ledger/result.json"
        if ft.exists():
            row["ft_status"] = json.loads(ft.read_text())["status"]
        elif (ft.parent / "request.json").exists():
            row["stage"] = "FT ledger: awaiting response"
        final = workflow / "manifest.json"
        if final.exists():
            manifest = json.loads(final.read_text())
            row["workflow_state"] = manifest["workflow_state"]
            row["route"] = manifest.get("route")
        tasks.append(row)
    return {"launcher_alive": alive, "tasks": tasks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.root)
    temporary = args.root / "status.json.tmp"
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.root / "status.json")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
