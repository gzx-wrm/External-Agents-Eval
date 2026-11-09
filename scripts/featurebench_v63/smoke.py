"""One CPU Level 1 smoke: official setup, unchanged V6.3, official P0/P1 scores."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import docker
from external_agents_eval.benchmarks.featurebench.adapter import (
    FeatureBenchEnvironment,
    check_worktree_imports,
    prepare_task,
    task_container,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def evaluate(root, run, stage):
    environment = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    environment["PYTHONPATH"] = str(root / "vendor/FeatureBench")
    command = [str(root / "venv312/bin/python"),
               str(root / "project/scripts/featurebench_v63/evaluate.py"),
               "--root", str(root), "--run", str(run), "--stage", stage]
    with (run / f"scoring-{stage}.log").open("ab") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
    return json.loads((run / "scoring" / stage / "summary.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "generate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "result.json").exists():
        raise ValueError("Run already has an outcome; use a distinct run for another attempt")
    task = json.loads((root / "manifests/task-private.json").read_text())
    public = json.loads((root / "manifests/task-public.json").read_text())
    if public["instance_id"] != task["instance_id"]:
        raise ValueError("Public and scoring task IDs differ")
    status = {"mode": args.mode, "task_id": task["instance_id"], "pid": os.getpid(),
              "workflow": "v6.3", "started_at": datetime.now(timezone.utc).isoformat()}

    def phase(name, **values):
        status.update(phase=name, **values, updated_at=datetime.now(timezone.utc).isoformat())
        save(run / "status.json", status)
        print(json.dumps(status), flush=True)

    client = docker.from_env(timeout=600)
    container = None
    try:
        image = json.loads((root / "preflight/image.json").read_text())
        if client.images.get(task["image_name"]).id != image["id"]:
            raise ValueError("Task image tag changed after selection")
        phase("preparing_task")
        container = task_container(client, image["id"], task, name="fb63-" + uuid.uuid4().hex[:12])
        container.start()
        environment = FeatureBenchEnvironment(container, instance_id=task["instance_id"])
        audit = asyncio.run(prepare_task(environment, task, run / "preparation"))
        phase("task_prepared", tree_sha=audit["tree_sha"])
        if args.mode == "preflight":
            asyncio.run(check_worktree_imports(environment, audit, run / "preparation"))
        else:
            preflight = json.loads((root / "preflight/readiness.json").read_text())
            if not preflight["passed"] or preflight["tree_sha"] != audit["tree_sha"]:
                raise ValueError("Generation baseline differs from the validated official task")
            from external_agents_eval.benchmarks.featurebench.agents import FeatureBenchV63Agent

            phase("workflow_running")
            agent = FeatureBenchV63Agent(
                logs_dir=run / "agent/logs", model_name="deepseek-flash",
                stage_model_config=root / "project/configs/v6.1-deepseek-v4.1-flash.json",
                extra_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
            )
            context = SimpleNamespace(metadata={})
            asyncio.run(agent.run(public["problem_statement"], environment, context))
            artifact = run / "agent/v6_review_repair"
            for label, name in [("P0", "v6-p0.patch"), ("P1", "v5-p1.patch")]:
                payload = (artifact / "outputs" / name).read_bytes()
                (run / f"{label}.patch").write_bytes(payload)
                save(run / f"prediction-{label}.json", {
                    "instance_id": task["instance_id"], "model_name_or_path": "v63-mini-deepseek-v4.1-flash",
                    "model_patch": payload.decode(), "patch_sha256": hashlib.sha256(payload).hexdigest(),
                })
            save(run / "workflow-metadata.json", context.metadata)
            phase("patches_ready")
    except Exception as error:  # noqa: BLE001 - record a redacted per-run outcome
        secret = os.environ.get("DEEPSEEK_API_KEY", "")
        message = str(error).replace(secret, "[REDACTED]") if secret else str(error)
        phase("failed", error=message, error_type=type(error).__name__)
        save(run / "result.json", status)
        raise SystemExit(2) from None
    finally:
        if container is not None:
            container.remove(force=True)
        client.close()

    stages = ["gold", "control"] if args.mode == "preflight" else ["P0", "P1"]
    results = {}
    try:
        for stage in stages:
            phase("official_evaluation", scoring_stage=stage)
            results[stage] = evaluate(root, run, stage)
    except (OSError, subprocess.CalledProcessError) as error:
        phase("scoring_failed", error_type=type(error).__name__, results=results)
        save(run / "result.json", status)
        raise SystemExit(2) from None
    if args.mode == "preflight":
        passed = (results["gold"]["resolved"] and results["gold"]["patch_applied"]
                  and results["control"]["patch_applied"] and not results["control"]["resolved"])
        save(root / "preflight/readiness.json", {
            "passed": passed, "tree_sha": audit["tree_sha"], "results": results,
            "model_api_calls": 0, "run": str(run),
        })
        if not passed:
            phase("preflight_failed", results=results)
            save(run / "result.json", status)
            raise SystemExit(2)
    phase("completed", results=results)
    save(run / "result.json", status)


if __name__ == "__main__":
    main()
