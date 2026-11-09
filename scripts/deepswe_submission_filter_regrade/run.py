"""Re-grade stored DeepSWE patches with the v6.6.4 submission filter; no model calls.

For each source root (outputs/<task>/model.patch is the patch that was graded), newly
added test files are stripped (external_agents_eval.benchmarks.deepswe.submission_filter).
Unchanged patches keep their recorded reward; changed patches are graded again with the
unchanged official verifier in the pinned M0 image. Results go to a new root; the source
roots are only read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.benchmarks.deepswe.submission_filter import (  # noqa: E402
    filter_record,
    strip_added_tests,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes  # noqa: E402

EXPERIMENT_ID = "v6.6.4-submission-filter-regrade"
FROZEN_SOURCES = ("scripts/deepswe_submission_filter_regrade/run.py",
                  "external_agents_eval/benchmarks/deepswe/submission_filter.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rows_for(label, source):
    rows = []
    for output in sorted((source / "outputs").iterdir()):
        patch_path, result_path = output / "model.patch", output / "result.json"
        if not (patch_path.exists() and result_path.exists()):
            continue
        recorded = read(result_path)
        patch = patch_path.read_text()
        submitted, removed = strip_added_tests(patch)
        rows.append({"label": label, "task_id": output.name, "source": str(output),
                     "recorded_status": recorded.get("status"), "recorded_reward": recorded.get("reward"),
                     "patch_sha256": sha256_bytes(patch.encode()), "removed": removed,
                     "changed": bool(removed)})
    return rows


def prepare(root, m0_root, sources, note):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir()
    copies = {"m0_run.py": "run.py", "native_compat.py": "native_compat.py",
              "sanitize_repository.py": "sanitize_repository.py"}
    for target, source in copies.items():
        shutil.copyfile(m0_root / "scripts" / source, root / "scripts" / target)
    (root / "manifests").mkdir()
    for name in ("public-tasks.json", "eval30_current.json", "task-file-sha256.json"):
        shutil.copyfile(m0_root / "manifests" / name, root / "manifests" / name)
    (root / "grader").symlink_to(m0_root / "grader", target_is_directory=True)
    rows = [row for label, source in sources for row in rows_for(label, source)]
    selection = {"experiment_id": EXPERIMENT_ID, "note": note, "m0_root": str(m0_root),
                 "sources": {label: str(source) for label, source in sources},
                 "code_sha256": {name: file_sha256(PROJECT / name) for name in FROZEN_SOURCES},
                 "rows": rows, "changed": sum(row["changed"] for row in rows)}
    save(root / "selection.json", selection)
    return selection


def run(root, selection, workers):
    v63 = load_module("regrade_v63_runner", PROJECT / "scripts/deepswe_v63_heldout/run.py")
    native = v63.load_native(root)
    if {name: file_sha256(PROJECT / name) for name in FROZEN_SOURCES} != selection["code_sha256"]:
        raise ValueError("Frozen code changed")
    tasks = {task["task_id"]: task for task in native.load_tasks(root)}
    m0 = Path(selection["m0_root"])
    lock = threading.Lock()

    def grade(row):
        output = root / "outputs" / row["label"] / row["task_id"]
        if (output / "result.json").exists():
            return read(output / "result.json")
        patch = (Path(row["source"]) / "model.patch").read_text()
        if sha256_bytes(patch.encode()) != row["patch_sha256"]:
            raise ValueError(f"Stored patch changed: {row['label']}/{row['task_id']}")
        submitted, removed = strip_added_tests(patch)
        output.mkdir(parents=True, exist_ok=True)
        (output / "submitted.patch").write_text(submitted)
        save(output / "submission-filter.json", filter_record(patch, submitted, removed))
        image = read(m0 / "outputs" / row["task_id"] / "image.json")
        result = native.grade(root, tasks[row["task_id"]], image["id"], submitted, output / "evaluation")
        result.update(label=row["label"], recorded_reward=row["recorded_reward"], removed=removed)
        save(output / "result.json", result)
        with lock:
            print(json.dumps({"label": row["label"], "task_id": row["task_id"],
                              "recorded": row["recorded_reward"], "filtered": result.get("reward")}), flush=True)
        return result

    changed = [row for row in selection["rows"] if row["changed"]]
    graded = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(grade, row): row for row in changed}
        for future in as_completed(futures):
            row = futures[future]
            try:
                graded[(row["label"], row["task_id"])] = future.result()
            except Exception as error:  # noqa: BLE001 - keep other tasks; report the failure
                graded[(row["label"], row["task_id"])] = {"status": "regrade_error", "reward": None,
                                                          "error": f"{type(error).__name__}: {error}"}
    summary = {}
    for label in selection["sources"]:
        rows = [row for row in selection["rows"] if row["label"] == label]
        final = {row["task_id"]: (graded[(label, row["task_id"])].get("reward") if row["changed"]
                                  else row["recorded_reward"]) for row in rows}
        summary[label] = {
            "tasks": len(rows),
            "recorded_passed": sum(row["recorded_reward"] == 1 for row in rows),
            "filtered_passed": sum(value == 1 for value in final.values()),
            "changed": [{"task_id": row["task_id"], "removed": row["removed"],
                         "recorded": row["recorded_reward"], "filtered": final[row["task_id"]],
                         "status": graded[(label, row["task_id"])].get("status")}
                        for row in rows if row["changed"]],
        }
    save(root / "result.json", {"experiment_id": EXPERIMENT_ID, "summary": summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--m0-root", type=Path)
    parser.add_argument("--source", action="append", default=[],
                        help="label=path of a root whose outputs/<task>/model.patch was graded")
    parser.add_argument("--note")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if not (args.m0_root and args.source):
            parser.error("prepare requires --m0-root and at least one --source")
        sources = [(item.split("=", 1)[0], Path(item.split("=", 1)[1]).resolve()) for item in args.source]
        selection = prepare(root, args.m0_root.resolve(), sources, args.note)
        print(json.dumps({"mode": "prepare", "rows": len(selection["rows"]), "changed": selection["changed"],
                          "changed_tasks": [f"{r['label']}/{r['task_id']}: {r['removed']}"
                                            for r in selection["rows"] if r["changed"]]}, indent=2))
    else:
        print(json.dumps(run(root, read(root / "selection.json"), args.workers), indent=2))


if __name__ == "__main__":
    main()
