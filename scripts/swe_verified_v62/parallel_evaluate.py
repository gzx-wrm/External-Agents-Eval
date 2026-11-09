"""Resume saved final patches with a bounded pool of official grading workers."""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import os
from pathlib import Path
import time

from stream_evaluate import (
    TERMINAL, load, now, official_report_path, read_outcome, resources_ready,
    save, scan, submission,
)


def evaluate_one(root, run_id, task, reference, run_instances, prepare_image=None):
    directory = root / "verifier/queue" / task
    receipt_path = directory / "evaluation.json"
    report_path = official_report_path(root / "verifier", run_id, task)
    receipt = {"instance_id": task, "status": "evaluating", "started_at": now(),
               "official_report_path": str(report_path)}
    try:
        prediction, hashes = submission(root, task)
        previous = load(receipt_path) if receipt_path.exists() else {}
        if previous.get("patch_sha256") not in (None, hashes["patch_sha256"]):
            raise ValueError("Patch changed since the previous scoring attempt")
        receipt.update(hashes)
        save(receipt_path, receipt)
        save(directory / "submission.json", prediction)
        if prediction["model_patch"] == "":
            receipt["status"] = "empty_patch"
        else:
            if not report_path.exists():
                if prepare_image is not None:
                    receipt["phase"] = "image_preparation"
                    save(receipt_path, receipt)
                    receipt["image_preparation"] = prepare_image(task)
                    receipt["phase"] = "official_test"
                    save(receipt_path, receipt)
                # The outer pool owns concurrency. Each official call scores one
                # instance and performs only its own per-instance image cleanup.
                run_instances(
                    predictions={task: prediction}, instances=[reference],
                    cache_level="env", clean=False, force_rebuild=False,
                    max_workers=1, run_id=run_id, timeout=1800, namespace="swebench",
                )
            if not report_path.exists():
                raise RuntimeError("Official evaluator ended without report.json; inspect run_instance.log")
            receipt.update(read_outcome(report_path, task))
        if submission(root, task)[1] != hashes:
            raise ValueError("Prediction changed while scoring")
    except Exception as error:
        receipt.update(status="evaluation_error" if "patch_sha256" in receipt else "submission_error",
                       error=str(error))
    receipt["finished_at"] = now()
    save(receipt_path, receipt)
    return receipt


def worker(args):
    import docker
    from swebench.harness.reporting import make_run_report
    from swebench.harness.run_evaluation import run_instances

    root = args.root.resolve()
    verifier = root / "verifier"
    verifier.mkdir(exist_ok=True)
    lock = (verifier / "streaming.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(verifier)  # Set once before worker threads start.
    ids = load(root / "selected98.json")["task_ids"]
    generated, generation_errors, missing = scan(root, ids)
    if missing:
        raise ValueError("Generation must be terminal before final parallel grading")
    references = {row["instance_id"]: row for row in load(root / "grader/reference_tasks.json")}
    completed = {}
    for task in generated:
        _, hashes = submission(root, task)
        path = verifier / "queue" / task / "evaluation.json"
        if path.exists():
            row = load(path)
            if row.get("patch_sha256") not in (None, hashes["patch_sha256"]):
                raise ValueError("A previously scored patch changed: " + task)
            if row.get("status") in TERMINAL:
                completed[task] = row
    running = {}
    prepare_image = None
    image_manager = None
    if args.image_proxy:
        from proxy_images import ProxyImages
        image_manager = ProxyImages(root, args.image_proxy)
        prepare_image = image_manager.prepare

    def publish(finished=False, resource_wait=False):
        active = sorted(running.values())
        counts = collections.Counter(row["status"] for row in completed.values())
        status = {
            "phase": "finished" if finished else "official_evaluation", "updated_at": now(),
            "supervisor_pid": os.getpid(), "generation_alive": False,
            "generation_workers": 3, "verifier_workers": args.workers,
            "run_id": args.run_id, "task_count": len(ids), "generated": len(generated),
            "generation_error_count": len(generation_errors), "generation_errors": generation_errors,
            "not_generated_count": 0, "active_evaluations": active,
            "active_evaluation": active[0] if len(active) == 1 else None,
            "queued_for_evaluation": len(generated) - len(completed) - len(running),
            "evaluation_counts": dict(counts),
            "official_completed": counts["resolved"] + counts["unresolved"],
            "resolved_ids": sorted(task for task, row in completed.items() if row["status"] == "resolved"),
            "unresolved_ids": sorted(task for task, row in completed.items() if row["status"] == "unresolved"),
            "evaluation_error_ids": sorted(task for task, row in completed.items() if row["status"] in {"evaluation_error", "submission_error"}),
            "waiting_for_resources": resource_wait,
            "image_proxy": args.image_proxy,
        }
        if finished:
            status["status"] = "evaluated_with_errors" if generation_errors or status["evaluation_error_ids"] else "evaluated"
        save(root / "monitor-status.json", status)
        save(root / "supervisor-status.json", status)

    publish()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            pending = [task for task in generated if task not in completed and task not in running.values()]
            if image_manager is not None and pending:
                available = image_manager.available(pending)
                pending.sort(key=lambda task: task not in available)
            resource_wait = False
            while pending and len(running) < args.workers:
                if not resources_ready(root):
                    resource_wait = True
                    break
                task = pending.pop(0)
                future = pool.submit(evaluate_one, root, args.run_id, task, references[task], run_instances, prepare_image)
                running[future] = task
            publish(resource_wait=resource_wait)
            if not running:
                if not pending:
                    break
                time.sleep(5)
                continue
            done, _ = wait(running, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                completed[task] = future.result()
                print({"event": "evaluation_finished", "instance_id": task,
                       "status": completed[task]["status"], "at": now()}, flush=True)

    predictions = {task: submission(root, task)[0] for task in generated}
    import json
    path = verifier / "predictions.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in predictions.values()))
    temporary.replace(path)
    save(root / "generation-summary.json", {
        "task_count": len(ids), "generated_ids": generated,
        "infrastructure_or_stage_errors": generation_errors, "missing_ids": missing,
        "empty_patch_ids": [task for task, row in predictions.items() if row["model_patch"] == ""],
        "generation_exit_code": None, "exit_code_note": "Generation ended before parallel scorer takeover",
        "finished_at": now(),
    })
    if predictions:
        client = docker.from_env()
        try:
            report = make_run_report(predictions, [references[task] for task in ids], args.run_id,
                                     client, namespace="swebench")
        finally:
            client.close()
        save(root / "streaming-final-report.json", {"official_report_path": str(report.resolve()), "finished_at": now()})
    publish(finished=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--image-proxy", help="Host-only HTTP/SOCKS proxy used to prefetch the recorded official images")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    worker(args)


if __name__ == "__main__":
    main()
