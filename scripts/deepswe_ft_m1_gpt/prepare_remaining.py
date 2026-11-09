"""Prepare the complementary FT25 with the already verified five-case runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path


def remaining_ids(suite, previous):
    if len(suite) != 30 or len(set(suite)) != 30:
        raise ValueError("Expected the fixed thirty unique tasks")
    if len(previous) != 5 or len(set(previous)) != 5 or not set(previous) <= set(suite):
        raise ValueError("Previous five must be distinct members of the fixed suite")
    return [task for task in suite if task not in set(previous)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    args = parser.parse_args()
    root, runtime = args.root.resolve(), args.runtime_source.resolve()
    if root.exists():
        raise FileExistsError("Use a new batch; existing runs remain intact")
    sys.path.insert(0, str(runtime / "scripts"))
    runner = importlib.import_module("run")
    if Path(runner.__file__).resolve() != runtime / "scripts/run.py":
        raise ValueError("Unexpected runtime module")
    runner.load(runtime)  # Verify the corrected runtime and its frozen source hashes.
    import yaml
    from ledger_input import augment_task, load_frozen_ledger

    previous_root = args.previous_run.resolve()
    previous_file = previous_root / "selection.json"
    previous = json.loads(previous_file.read_text())
    all_tasks = runner.native.load_tasks(args.source_run[0])
    suite = [task["task_id"] for task in all_tasks]
    excluded = [record["task_id"] for record in previous["tasks"]]
    chosen_ids = remaining_ids(suite, excluded)
    prior_results = {}
    for name in excluded:
        result_path = runtime / "outputs" / name / "result.json"
        if not result_path.exists():
            result_path = previous_root / "outputs" / name / "result.json"
        result = json.loads(result_path.read_text())
        if result.get("status") != "scored" or result.get("reward") not in [0, 1]:
            raise ValueError("A previous task has no effective official result: " + name)
        prior_results[name] = {"path": str(result_path), "sha256": runner.digest(result_path),
                               "reward": result["reward"]}
    records = {}
    frozen_config = yaml.safe_load((runtime / "configs/frozen-m1.yaml").read_text())
    for source in args.source_run:
        source = source.resolve()
        source_selection = json.loads((source / "selection.json").read_text())
        original = {task["task_id"]: task for task in runner.native.load_tasks(source)}
        source_config = yaml.safe_load((source / "configs/native-deepseek-v4.1-flash.yaml").read_text())
        if source_config != frozen_config:
            raise ValueError("Frozen M1 configuration differs")
        for record in source_selection["tasks"]:
            name = record["task_id"]
            if name in records:
                raise ValueError("Duplicate task in source M1 runs")
            ledger = load_frozen_ledger(source, record)
            task = augment_task(original[name], ledger)
            task["image"] = record["m0_image"]["repo_digests"][0]
            if task != json.loads((source / "inputs" / name / "task.json").read_text()):
                raise ValueError("Proposed input differs from frozen M1: " + name)
            records[name] = (source, record, task, original[name])
    if set(records) != set(suite):
        raise ValueError("M1 sources do not cover the same fixed thirty")
    root.mkdir()
    (root / "scripts").mkdir()
    for name in ["run.py", "launch.py", "routed_model.py", "m0_run.py", "ledger_input.py", "sanitize_repository.py"]:
        shutil.copyfile(runtime / "scripts" / name, root / "scripts" / name)
    shutil.copyfile(Path(__file__), root / "scripts/prepare_remaining.py")
    (root / "configs").mkdir()
    shutil.copyfile(runtime / "configs/frozen-m1.yaml", root / "configs/frozen-m1.yaml")
    for name in ["grader", "manifests"]:
        (root / name).symlink_to(args.source_run[0].resolve() / name, target_is_directory=True)
    selected = []
    for name in chosen_ids:
        source, record, task, original = records[name]
        folder = root / "inputs" / name
        folder.mkdir(parents=True)
        ledger_file = folder / "ledger.json"
        shutil.copyfile(source / record["ledger_file"], ledger_file)
        task_file = folder / "task.json"
        runner.native.save(task_file, task)
        selected.append({"task_id": name, "task_checksum": task["task_checksum"],
                         "task_file": str(task_file.relative_to(root)), "task_file_sha256": runner.digest(task_file),
                         "ledger_file": str(ledger_file.relative_to(root)),
                         "ledger_sha256": record["ledger_sha256"], "ledger_file_sha256": runner.digest(ledger_file),
                         "image": record["m0_image"], "m1_root": str(source),
                         "m1_result_sha256": runner.digest(source / "outputs" / name / "result.json"),
                         "original_instruction_sha256": hashlib.sha256(original["instruction"].encode()).hexdigest(),
                         "augmented_instruction_sha256": hashlib.sha256(task["instruction"].encode()).hexdigest()})
    selection = {
        "experiment_id": "ft_m1_gpt61sol", "batch": "remaining25", "tasks": selected, "task_count": 25,
        "previous_five_run": str(previous_root), "previous_selection_sha256": runner.digest(previous_file),
        "previous_effective_results": prior_results, "disjoint_from_previous_five": True,
        "combined_unique_tasks": len(set(chosen_ids) | set(excluded)),
        "runtime_source": str(runtime), "runtime_copied_byte_for_byte": True,
        "fresh_solver_from_original_repository": True, "previous_patch_or_conversation_reused": False,
        "evidence_packet_included": False, "review_guidance_included": False,
        "model": "gpt-6.1-sol", "reasoning_effort": "xhigh", "settings_approval_pending": False,
        "official_gold_control_reference": str(args.source_run[0].resolve() / "preflight/readiness.json"),
        "suite_notes": ["SQLite excluded; True Myth replacement retained with its existing training-overlap provenance."],
        "source_sha256": {str(path.relative_to(root)): runner.digest(path) for path in [
            *sorted((root / "scripts").glob("*.py")), root / "configs/frozen-m1.yaml"]},
    }
    runner.native.save(root / "selection.json", selection)
    print(json.dumps({"prepared": True, "root": str(root), "task_count": 25,
                      "excluded_previous_five": excluded, "runtime_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
