"""ft_m1: the frozen M0 runner with only pre-Solver FT ledgers added to input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import m0_run as native
import yaml

from ledger_input import GUIDANCE, HEADING, augment_task, load_frozen_ledger


def inputs(root):
    selection = json.loads((root / "selection.json").read_text())
    expected_count = selection.get("task_count", 8)
    expected_reward = selection.get("m0_reward_filter", 0)
    tasks = {task["task_id"]: task for task in native.load_tasks(root)}
    chosen = []
    for record in selection["tasks"]:
        task = tasks[record["task_id"]]
        if task["task_checksum"] != record["task_checksum"] or record["m0_reward"] != expected_reward:
            raise ValueError("M0/FT task identity or selected outcome differs")
        ledger = load_frozen_ledger(root, record)
        augmented = augment_task(task, ledger)
        # Pin the exact original M0 image, without changing its contents.
        augmented["image"] = record["m0_image"]["repo_digests"][0]
        native.save(root / "inputs" / task["task_id"] / "task.json", augmented)
        native.save(root / "outputs" / task["task_id"] / "input-audit.json", {
            "experiment_id": "ft_m1", "task_id": task["task_id"],
            "task_checksum": task["task_checksum"],
            "original_instruction_sha256": hashlib.sha256(task["instruction"].encode()).hexdigest(),
            "augmented_instruction_sha256": hashlib.sha256(augmented["instruction"].encode()).hexdigest(),
            "ledger_sha256": record["ledger_sha256"],
            "ledger_file_sha256": record["ledger_file_sha256"],
            "source_snapshot": record["source_snapshot"],
            "source_snapshot_sha256": record["source_snapshot_sha256"],
            "counts": {name: len(ledger[name]) for name in
                       ["obligations", "preservation_obligations", "open_questions"]},
            "evidence_packet_included": False, "review_guidance_included": False,
            "ledger_modified": False,
        })
        chosen.append(augmented)
    if (selection["experiment_id"] != "ft_m1" or len(chosen) != expected_count
            or len({task["task_id"] for task in chosen}) != expected_count):
        raise ValueError("Task selection differs from the frozen ft_m1 batch")
    return selection, chosen


def preflight(root, selection, tasks, config):
    parent = Path(selection["m0_root"])
    reference = json.loads((parent / "preflight/readiness.json").read_text())
    if not reference["passed"]:
        raise ValueError("The M0 official gold/control checks are not valid")
    output = root / "preflight" / tasks[0]["task_id"]
    client = native.docker.from_env(timeout=1800)
    try:
        image, _ = native.get_image(client, tasks[0], output)
        if image.id != selection["tasks"][0]["m0_image"]["id"]:
            raise ValueError("Preflight image differs from M0")
        _, protocol = native.generate(tasks[0], root, output / "offline", config, image, "offline")
    finally:
        client.close()
    trajectory = json.loads((output / "offline/native.traj.json").read_text())
    user = next(message["content"] for message in trajectory["messages"] if message["role"] == "user")
    if tasks[0]["instruction"] not in user or user.count(HEADING) != 1 or user.count(GUIDANCE) != 1:
        raise ValueError("Native Mini did not receive exactly one frozen ledger supplement")
    # Every other agent/scoring function is the byte-for-byte frozen M0 source.
    native.save(root / "preflight/readiness.json", {
        "passed": True, "experiment_id": "ft_m1", "model_api_calls": 0,
        "task_count": len(tasks), "offline_protocol": protocol,
        "actual_native_user_message_verified": True,
        "all_ledger_hashes_verified": True, "evidence_packet_included": False,
        "official_gold_control_reused_from": str(parent / "preflight/readiness.json"),
        "finished_at": native.now(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "preflight", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers != 3:
        raise ValueError("This experiment uses the approved three pipeline slots")
    root = args.root.resolve()
    selection, tasks = inputs(root)
    config = native.prepare_config(root)
    original = yaml.safe_load((Path(selection["m0_root"]) / "configs/native-deepseek-v4.1-flash.yaml").read_text())
    if config != original:
        raise ValueError("Mini prompts, tools, model parameters, or budgets differ from M0")
    if args.mode == "prepare":
        native.save(root / "preflight/plan.json", {
            "experiment_id": "ft_m1", "task_ids": [task["task_id"] for task in tasks],
            "m0_reward_filter": selection.get("m0_reward_filter", 0),
            "workers": args.workers, "configuration_identical_to_m0": True,
            "input_change": "append original FT pre-Solver ledger and approved brief guidance",
            "evidence_packet_included": False, "model_api_calls": 0,
        })
    elif args.mode == "preflight":
        preflight(root, selection, tasks, config)
    else:
        native.run_batch(root, tasks, config, args.workers)


if __name__ == "__main__":
    main()
