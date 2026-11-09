"""V6.6 stage C only, on frozen A outputs (ledger + evidence packet); no Pi or FT call.

prepare copies the frozen inputs; run makes one check-point request per task (with the
approved transport retry and one corrective retry). Outputs use the A layout
(a/<task>/...), so the probe stage can take this root as --a-root.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.workflows.solve_review_repair.artifacts import ArtifactStore  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.v66_checkpoints import (  # noqa: E402
    CheckpointClient,
)
from external_agents_eval.workflows.solve_review_repair.v66_config import V66Config  # noqa: E402

EXPERIMENT_ID = "v6.6-c-only"
FROZEN_SOURCES = (
    "scripts/deepswe_v66_c/run.py",
    "external_agents_eval/workflows/solve_review_repair/v66_checkpoints.py",
    "external_agents_eval/workflows/solve_review_repair/v66_prompts.py",
    "external_agents_eval/workflows/solve_review_repair/v66_transport.py",
    "external_agents_eval/workflows/solve_review_repair/v66_config.py",
)
INPUTS = {"ledger.json": "ledger.json", "evidence.json": "v6_review_repair/v6.2/evidence.json"}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(root, source, task_ids, note):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    rows = []
    for tid in task_ids:
        frozen = root / "inputs" / tid
        frozen.mkdir(parents=True)
        for name, relative in INPUTS.items():
            shutil.copyfile(source / "a" / tid / relative, frozen / name)
        rows.append({"task_id": tid, "source": str(source / "a" / tid),
                     "inputs_sha256": {name: file_sha256(frozen / name) for name in INPUTS}})
    config = V66Config()
    selection = {"experiment_id": EXPERIMENT_ID, "note": note, "tasks": rows, "task_count": len(rows),
                 "code_sha256": {name: file_sha256(PROJECT / name) for name in FROZEN_SOURCES},
                 "prompt_version": config.workflow_prompt_version,
                 "config": {"max_per_target": config.checkpoint_max_per_target,
                            "max_attempts": config.checkpoint_max_attempts,
                            "solver_view_points": config.solver_view_max_points,
                            "batch_max_targets": config.checkpoint_batch_max_targets}}
    write(root / "selection.json", selection)
    return selection


def run_task(root, row, model, key, config):
    tid = row["task_id"]
    frozen = root / "inputs" / tid
    if {name: file_sha256(frozen / name) for name in INPUTS} != row["inputs_sha256"]:
        raise ValueError(f"Frozen input changed: {tid}")
    output = root / "a" / tid
    if (output / "a-result.json").exists():
        return
    if output.exists():
        raise RuntimeError(f"{tid}: a C attempt exists without a result; not resampled")
    output.mkdir(parents=True)
    shutil.copyfile(frozen / "ledger.json", output / "ledger.json")
    ledger, packet = read(frozen / "ledger.json"), read(frozen / "evidence.json")
    task = packet["task"]
    packet.update(task={"description": task["description"], "title": task["title"]})  # FT order
    client = CheckpointClient(model=model["model"], base_url=model["base_url"], api_key=key,
                              max_per_target=config.checkpoint_max_per_target,
                              max_attempts=config.checkpoint_max_attempts,
                              timeout_sec=config.checkpoint_timeout_sec,
                              max_solver_points=config.solver_view_max_points,
                              batch_max_targets=config.checkpoint_batch_max_targets)
    result = asyncio.run(client.generate(packet=packet, ledger=ledger,
                                         artifacts=ArtifactStore(output / "v6_review_repair", secrets=(key,))))
    record = {"task_id": tid, "status": "checkpoints_ready" if result["status"] == "passed" else "checkpoints_invalid",
              "checkpoints": {k: result.get(k) for k in ("status", "attempts", "summary", "elapsed_seconds",
                                                         "degraded_batches", "failed_batches")}}
    write(output / "a-result.json", record)
    print(json.dumps({"task_id": tid, "status": record["status"],
                      "summary": (result.get("summary") or {})}), flush=True)


def run(root, selection, workers=1):
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    if {name: file_sha256(PROJECT / name) for name in FROZEN_SOURCES} != selection["code_sha256"]:
        raise ValueError("Frozen code changed")
    config = V66Config()
    model = json.loads((PROJECT / "configs/v6.1-deepseek-v4.1-flash.json").read_text())["default"]
    # Tasks are independent; each keeps its own sequential batches and artifacts.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(run_task, root, row, model, key, config): row["task_id"]
                   for row in selection["tasks"]}
        failures = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - other tasks continue; reported below
                failures.append(futures[future])
                print(json.dumps({"task_id": futures[future], "status": "runner_error",
                                  "error": f"{type(error).__name__}: {error}"}), flush=True)
    if failures:
        raise RuntimeError("C runner failed for: " + ", ".join(sorted(failures)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--from-root", type=Path)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--note")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if not (args.from_root and args.task_id):
            parser.error("prepare requires --from-root and --task-id")
        selection = prepare(root, args.from_root.resolve(), args.task_id, args.note)
    else:
        selection = read(root / "selection.json")
        run(root, selection, args.workers)
    print(json.dumps({"mode": args.mode, "experiment_id": EXPERIMENT_ID, "task_count": selection["task_count"]}))


if __name__ == "__main__":
    main()
