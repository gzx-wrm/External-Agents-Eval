"""Serial official scoring alongside an already-running generation process.

Run outside the frozen project snapshot. A handoff marker is required so that
the former generation-then-grading supervisor cannot launch a second scorer.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


MODEL = "v62-mini-deepseek-v4.1-flash"
TERMINAL = {"resolved", "unresolved", "evaluation_error", "submission_error", "empty_patch"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def scan(root, ids):
    generated, errors, missing = [], [], []
    for task in ids:
        output = root / "outputs" / task
        path = output / "result.json"
        if not path.exists():
            missing.append(task)
            continue
        result = load(path)
        if result.get("status") == "generated" and (output / "prediction.json").exists():
            generated.append(task)
        else:
            errors.append({"instance_id": task, **result})
    generated.sort(key=lambda task: (root / "outputs" / task / "result.json").stat().st_mtime_ns)
    return generated, errors, missing


def submission(root, task):
    output = root / "outputs" / task
    prediction_path = output / "prediction.json"
    prediction = load(prediction_path)
    patch_path = output / "agent/v6_review_repair/outputs/v5-p1.patch"
    patch = prediction.get("model_patch")
    if prediction.get("instance_id") != task or prediction.get("model_name_or_path") != MODEL:
        raise ValueError("Prediction identity differs from the frozen run")
    if not isinstance(patch, str) or patch_path.read_bytes() != patch.encode():
        raise ValueError("Submitted patch differs from the final workflow patch")
    return prediction, {
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
    }


def official_report_path(verifier, run_id, task):
    return verifier / "logs/run_evaluation" / run_id / MODEL / task / "report.json"


def read_outcome(path, task):
    row = load(path)[task]
    if not isinstance(row.get("resolved"), bool):
        raise ValueError("Official report has no boolean resolved field")
    counts = {}
    for name, result in row.get("tests_status", {}).items():
        counts[name] = {key: len(result.get(key, [])) for key in ("success", "failure")}
    return {"status": "resolved" if row["resolved"] else "unresolved", "test_counts": counts}


def resources_ready(root):
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    return available >= 4 * 1024**3 and shutil.disk_usage(root).free >= 15 * 1024**3


def worker(args):
    root = args.root.resolve()
    verifier = root / "verifier"
    verifier.mkdir(exist_ok=True)
    lock = (verifier / "streaming.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = process_identity(os.getpid())
    save(root / "streaming-verifier-ready.json", {"pid": os.getpid(), "process_start": identity, "ready_at": now()})
    handoff_path = verifier / "handoff.json"
    while not handoff_path.exists():
        time.sleep(1)
    handoff = load(handoff_path)
    if handoff["verifier_pid"] != os.getpid() or handoff["verifier_process_start"] != identity:
        raise RuntimeError("Handoff marker does not belong to this process")
    if process_identity(handoff["previous_supervisor_pid"]) is not None:
        raise RuntimeError("Previous supervisor still exists; refuse concurrent scoring")

    import docker
    from swebench.harness.reporting import make_run_report
    from swebench.harness.run_evaluation import run_instances

    os.chdir(verifier)
    ids = load(root / "selected98.json")["task_ids"]
    references = {row["instance_id"]: row for row in load(root / "grader/reference_tasks.json")}
    if not set(ids).issubset(references):
        raise ValueError("Official reference dataset does not cover the selected tasks")
    completed = {}
    for task in ids:
        receipt = verifier / "queue" / task / "evaluation.json"
        if receipt.exists() and load(receipt).get("status") in TERMINAL:
            row = load(receipt)
            if "patch_sha256" in row and submission(root, task)[1]["patch_sha256"] != row["patch_sha256"]:
                raise ValueError("A previously scored patch changed: " + task)
            completed[task] = row

    def generation_alive():
        return process_identity(args.generation_pid) == args.generation_start

    def publish(active=None, resource_wait=False, finished=False):
        generated, errors, missing = scan(root, ids)
        counts = collections.Counter(row["status"] for row in completed.values())
        pending = [task for task in generated if task not in completed and task != active]
        phase = "generation_and_official_evaluation" if generation_alive() else "official_evaluation"
        status = {
            "phase": "finished" if finished else phase, "updated_at": now(),
            "supervisor_pid": os.getpid(), "generation_pid": args.generation_pid,
            "generation_alive": generation_alive(), "generation_workers": 3, "verifier_workers": 1,
            "run_id": args.run_id, "task_count": len(ids), "generated": len(generated),
            "generation_error_count": len(errors), "generation_errors": errors,
            "not_generated_count": len(missing), "active_evaluation": active,
            "queued_for_evaluation": len(pending), "evaluation_counts": dict(counts),
            "official_completed": counts["resolved"] + counts["unresolved"],
            "resolved_ids": sorted(task for task, row in completed.items() if row["status"] == "resolved"),
            "unresolved_ids": sorted(task for task, row in completed.items() if row["status"] == "unresolved"),
            "evaluation_error_ids": sorted(task for task, row in completed.items() if row["status"] in {"evaluation_error", "submission_error"}),
            "waiting_for_resources": resource_wait,
        }
        if finished:
            status["status"] = "evaluated_with_errors" if errors or missing or status["evaluation_error_ids"] else "evaluated"
        save(root / "monitor-status.json", status)
        save(root / "supervisor-status.json", status)
        return generated, errors, missing, pending

    while True:
        generated, errors, missing, pending = publish()
        if not pending:
            if not generation_alive():
                # Re-scan after observing process exit: the final prediction may
                # have been committed between the previous scan and that exit.
                if not publish()[3]:
                    break
                continue
            time.sleep(10)
            continue
        if not resources_ready(root):
            publish(resource_wait=True)
            time.sleep(10)
            continue
        task = pending[0]
        case_dir = verifier / "queue" / task
        receipt_path = case_dir / "evaluation.json"
        report_path = official_report_path(verifier, args.run_id, task)
        receipt = {"instance_id": task, "status": "evaluating", "started_at": now(),
                   "official_report_path": str(report_path)}
        publish(active=task)
        try:
            prediction, hashes = submission(root, task)
            if receipt_path.exists():
                previous = load(receipt_path)
                if previous.get("patch_sha256") not in (None, hashes["patch_sha256"]):
                    raise ValueError("Patch changed since the previous scoring attempt")
            receipt.update(hashes)
            save(receipt_path, receipt)
            save(case_dir / "submission.json", prediction)
            if prediction["model_patch"] == "":
                receipt["status"] = "empty_patch"
            else:
                print(json.dumps({"event": "evaluation_started", "instance_id": task, "at": now()}), flush=True)
                if not report_path.exists():
                    # This is the same official evaluation core used by the CLI.
                    # Its per-instance cleanup only touches this scoring image;
                    # do not run the CLI's global image-cache cleanup alongside generation.
                    run_instances(
                        predictions={task: prediction}, instances=[references[task]],
                        cache_level="env", clean=False, force_rebuild=False,
                        max_workers=1, run_id=args.run_id, timeout=1800, namespace="swebench",
                    )
                if not report_path.exists():
                    raise RuntimeError("Official evaluator ended without report.json; inspect run_instance.log")
                receipt.update(read_outcome(report_path, task))
            _, after = submission(root, task)
            if after != hashes:
                raise ValueError("Prediction changed while scoring")
        except Exception as error:
            receipt["status"] = "evaluation_error" if "patch_sha256" in receipt else "submission_error"
            receipt["error"] = str(error)
        receipt["finished_at"] = now()
        save(receipt_path, receipt)
        completed[task] = receipt
        publish()
        print(json.dumps({"event": "evaluation_finished", **receipt}), flush=True)
        if receipt["status"] in {"resolved", "unresolved"} and not (root / "streaming-first-case-validation.json").exists():
            save(root / "streaming-first-case-validation.json", {
                "instance_id": task, "validated_at": now(), "official_report_path": str(report_path),
                "patch_unchanged": True, "prediction_unchanged": True,
                "generation_process_unchanged": generation_alive(), "generation_pid": args.generation_pid,
                "verifier_workers": 1, "status": receipt["status"], "test_counts": receipt.get("test_counts", {}),
            })

    generated, errors, missing = scan(root, ids)
    predictions = {task: submission(root, task)[0] for task in generated}
    temporary = verifier / "predictions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(value) + "\n" for value in predictions.values()))
    temporary.replace(verifier / "predictions.jsonl")
    save(root / "generation-summary.json", {
        "task_count": len(ids), "generated_ids": generated, "infrastructure_or_stage_errors": errors,
        "missing_ids": missing, "empty_patch_ids": [task for task, pred in predictions.items() if pred["model_patch"] == ""],
        "generation_exit_code": None, "exit_code_note": "Adopted existing process; use per-task terminal results",
        "finished_at": now(),
    })
    if predictions:
        client = docker.from_env()
        try:
            report_path = make_run_report(predictions, [references[task] for task in ids], args.run_id, client, namespace="swebench")
        finally:
            client.close()
        save(root / "streaming-final-report.json", {"official_report_path": str(report_path.resolve()), "finished_at": now()})
    publish(finished=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--generation-pid", type=int, required=True)
    parser.add_argument("--generation-start", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    worker(args)


if __name__ == "__main__":
    main()
