"""One-shot batch continuation for approved, pre-model UTF-8 index failures.

The parent experiment is never edited. Wait for its controller to exit, then
run only the matching unsampled cases using the fixed CodeScout implementation.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import batch


def parent_alive(root):
    receipt = batch.read(root / "launch.json")
    directory = Path("/proc") / str(receipt["pid"])
    try:
        command = (directory / "cmdline").read_bytes()
        state = (directory / "stat").read_text().split(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z" and str(root).encode() in command and b"batch.py" in command


def eligible_cases(root, selection):
    selected = []
    for record in selection["tasks"]:
        tid = record["task_id"]
        output = root / "outputs" / tid
        if not (output / "result.json").exists():
            continue
        result = batch.read(output / "result.json")
        log = root / "logs" / (tid + ".preprocessing.log")
        text = log.read_text() if log.exists() else ""
        if (result.get("reward") is None
                and result.get("failed_phase") == "codescout_preprocessing"
                and "UnicodeDecodeError" in text and "header.decode" in text
                and "unexpected end of data" in text):
            if (list((output / "preprocessing/codescout").rglob("request.json"))
                    or (output / "generation-started.json").exists()
                    or (output / "generation/native.traj.json").exists()):
                raise ValueError("Refusing to resample a case that already reached the model: " + tid)
            selected.append(tid)
    return selected


def composite(parent, recovery, smoke, status):
    records = {}
    roots = [smoke, parent, recovery] if smoke else [parent, recovery]
    for root in roots:
        summary = root / "result.json"
        if summary.exists():
            for result in batch.read(summary)["results"]:
                records[result["task_id"]] = {"result_root": str(root), **result}
    values = list(records.values())
    batch.save(status, "composite-current.json", {
        "task_count": len(values), "results": values,
        "passed": sum(result.get("reward") == 1 for result in values),
        "official_failed": sum(result.get("reward") == 0 for result in values),
        "unscored": sum(result.get("reward") is None for result in values),
        "official_scores_unchanged": True,
        "vulture_mapping_warning": bool(smoke), "updated_at": batch.now(),
    })


def continue_batch(args, key):
    parent, recovery, status = args.parent_root.resolve(), args.root.resolve(), args.status_root.resolve()
    selection = batch.read(parent / "selection.json")
    batch.save(status, "status.json", {"status": "waiting_for_parent", "pid": os.getpid(),
                                      "parent_root": str(parent), "recovery_root": str(recovery),
                                      "workers": 3, "started_at": batch.now(),
                                      "credentials_persisted": False})
    while parent_alive(parent):
        time.sleep(30)
    if not (parent / "result.json").exists():
        batch.save(status, "status.json", {"status": "parent_interrupted", "parent_root": str(parent),
                                          "reason": "Parent exited without a terminal batch summary"})
        raise SystemExit("Parent ended without a complete summary; inspect existing progress")
    ids = eligible_cases(parent, selection)
    if not ids:
        composite(parent, recovery, args.prior_smoke, status)
        batch.save(status, "status.json", {"status": "completed_no_eligible_cases", "task_count": 0})
        return
    sources = list(dict.fromkeys(Path(record["source_bundle"]) for record in selection["tasks"]))
    batch.prepare(recovery, sources, ids, config_file=parent / "configs/codescout.json")
    batch.save(status, "status.json", {"status": "recovering", "task_ids": ids, "workers": 3,
                                      "parent_root": str(parent), "recovery_root": str(recovery),
                                      "started_at": batch.now(), "credentials_persisted": False})
    env = os.environ.copy()
    env.update(DEEPSEEK_API_KEY=key, DO_NOT_TRACK="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    runner = Path(batch.__file__).resolve()
    for command in [
        [sys.executable, str(runner), "preprocess", "--root", str(recovery), "--task-id", ids[0]],
        [sys.executable, str(runner), "preflight", "--root", str(recovery)],
    ]:
        subprocess.run(command, env=env, check=True, cwd=batch.PROJECT,
                       timeout=batch.Config(**batch.read(parent / "configs/codescout.json")).max_seconds)
    batch.save(recovery, "launch.json", {"pid": os.getpid(), "started_at": batch.now(),
                                         "task_ids": ids, "task_count": len(ids), "workers": 3,
                                         "continuation_of": str(parent), "credentials_persisted": False})
    # Use the same controller in this process; its workers retain the bound of 3.
    os.environ.update(DEEPSEEK_API_KEY=key, DO_NOT_TRACK="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
    batch.run_batch(recovery, 3)
    composite(parent, recovery, args.prior_smoke, status)
    batch.save(status, "status.json", {"status": "completed", "task_ids": ids,
                                      "recovery_root": str(recovery), "finished_at": batch.now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--status-root", type=Path, required=True)
    parser.add_argument("--prior-smoke", type=Path)
    args = parser.parse_args()
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("Provide the host-only credential on stdin")
    try:
        continue_batch(args, key)
    except Exception as error:
        batch.save(args.status_root, "status.json", {
            "status": "failed", "error_type": type(error).__name__,
            "error": str(error).replace(key, "[REDACTED]"), "finished_at": batch.now(),
        })
        raise


if __name__ == "__main__":
    main()
