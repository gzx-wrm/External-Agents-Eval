"""Fixed FT ledger + fixed ft_m1 P0 -> Mini V6 I/R/S/Repair -> official verifier.

Prepare and preflight make no model calls. Run is explicit and never regenerates
A or Solver. All artifacts go to a new root; historical runs are read-only.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.agents.mini_ft_review import (
    FrozenMiniReviewAgent,
    normalization_command,
    write_frozen_fork,
)
from external_agents_eval.benchmarks.swebench.isolated_environment import IsolatedEnvironment
from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_bytes, sha256_json


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        text = text.replace(key, "[REDACTED]")
    temporary = path.with_name(path.name + f".{threading.get_ident()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def load_native(root):
    if importlib.metadata.version("mini-swe-agent") != "2.4.6":
        raise ValueError("This frozen experiment requires mini-SWE-agent 2.4.6")
    path = root / "scripts/m0_run.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ft_review_frozen_m0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_files(hashes):
    for name, expected in hashes.items():
        if sha256_bytes(Path(name).read_bytes()) != expected:
            raise ValueError("Frozen source changed: " + name)


def prepare(root, sources, task_ids):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty review experiment directory")
    chosen, common_hashes, base = {}, None, None
    scripts = ("m0_run.py", "native_compat.py", "sanitize_repository.py")
    for source in sources:
        selection = read(source / "selection.json")
        if selection["experiment_id"] != "ft_m1":
            raise ValueError("Expected a frozen ft_m1 source")
        native = load_native(source)
        tasks = {task["task_id"]: task for task in native.load_tasks(source)}
        hashes = {name: sha256_bytes((source / "scripts" / name).read_bytes()) for name in scripts}
        # All selected tasks must use the same frozen M0 runtime and grader.
        current_base = Path(selection["m0_root"]).resolve()
        if common_hashes is not None and (hashes != common_hashes or current_base != base):
            raise ValueError("ft_m1 sources have different native runtimes or grader roots")
        common_hashes, base = hashes, current_base
        for row in selection["tasks"]:
            tid = row["task_id"]
            if task_ids and tid not in task_ids:
                continue
            if tid in chosen:
                raise ValueError("Multiple P0 sources selected for " + tid)
            output = source / "outputs" / tid
            result, audit = read(output / "result.json"), read(output / "input-audit.json")
            prediction = read(output / "prediction.json")
            ledger_path = source / row["ledger_file"]
            ledger_bytes = ledger_path.read_bytes()
            ledger = json.loads(ledger_bytes)
            task = tasks[tid]
            patch = (output / "model.patch").read_bytes()
            image = read(output / "image.json")
            if (result["status"] != "scored" or result.get("reward") not in (0, 1)
                    or prediction.get("exit_status") != "Submitted"):
                raise ValueError("Select a completed, valid ft_m1 P0: " + tid)
            if (task["task_checksum"] != row["task_checksum"]
                    or sha256_bytes(ledger_bytes) != row["ledger_file_sha256"]
                    or sha256_json(ledger) != row["ledger_sha256"]
                    or audit["ledger_sha256"] != row["ledger_sha256"]
                    or audit.get("evidence_packet_included") is not False
                    or audit.get("ledger_modified") is not False
                    or image["id"] != row["m0_image"]["id"]
                    or prediction["model_patch"].encode() != patch
                    or result.get("patch_sha256") != sha256_bytes(patch)):
                raise ValueError("ft_m1 source identity or artifact mismatch: " + tid)
            files = [output / name for name in [
                "result.json", "input-audit.json", "prediction.json", "model.patch",
                "image.json", "isolation.json", "environment-baseline.patch",
            ]] + [ledger_path, source / "selection.json",
                  source / "configs/native-deepseek-v4.1-flash.yaml"]
            chosen[tid] = {
                "task_id": tid, "task": task, "source_root": str(source),
                "m1_output": str(output), "ledger_file": str(ledger_path),
                "ledger_sha256": row["ledger_sha256"], "p0_sha256": sha256_bytes(patch),
                "image": row["m0_image"], "m1_reward": result["reward"],
                "source_sha256": {str(p): sha256_bytes(p.read_bytes()) for p in files},
            }
    if not chosen or (task_ids and task_ids != set(chosen)):
        raise ValueError("Requested task is missing from the supplied ft_m1 runs")
    if base is None or not read(base / "preflight/readiness.json")["passed"]:
        raise ValueError("Original native scorer gold/control preflight is unavailable")
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(exist_ok=True)
    for name in scripts:
        shutil.copyfile(sources[0] / "scripts" / name, root / "scripts" / name)
    (root / "grader").symlink_to(base / "grader", target_is_directory=True)
    selection = {
        "experiment_id": "ft_m1_review", "start_at": "I/R", "m0_root": str(base),
        "tasks": list(chosen.values()), "native_source_sha256": common_hashes,
        "model_config_sha256": sha256_bytes(
            (PROJECT / "configs/v6.1-deepseek-v4.1-flash.json").read_bytes()),
        "review_adapter_sha256": {
            str(path): sha256_bytes(path.read_bytes()) for path in [
                Path(__file__).resolve(), PROJECT / "external_agents_eval/agents/mini_ft_review.py",
            ]
        },
        "official_preflight": str(base / "preflight/readiness.json"),
        "ledger_and_p0_reused": True, "new_evidence_packet": False,
    }
    save(root / "selection.json", selection)
    save(root / "preflight/plan.json", {
        "experiment_id": "ft_m1_review", "task_count": len(chosen), "model_api_calls": 0,
        "pipeline": "frozen FT ledger + frozen ft_m1 P0 -> I/R -> S -> Repair/accept/hold -> verifier",
        "review_backend": "existing Mini V6.3 file delivery + unchanged V6 controller",
        "model": "DeepSeek V4.1 Flash", "workers_default": 3,
    })
    return selection


def validate_selection(root, native):
    selection = read(root / "selection.json")
    if selection["experiment_id"] != "ft_m1_review":
        raise ValueError("Wrong review experiment")
    verify_files({str(root / "scripts" / name): digest
                  for name, digest in selection["native_source_sha256"].items()})
    verify_files(selection["review_adapter_sha256"])
    config_path = PROJECT / "configs/v6.1-deepseek-v4.1-flash.json"
    if sha256_bytes(config_path.read_bytes()) != selection["model_config_sha256"]:
        raise ValueError("Review model configuration changed")
    # This is the frozen native loader, including every official task-file hash.
    for source in {row["source_root"] for row in selection["tasks"]}:
        native.load_tasks(Path(source))
    for row in selection["tasks"]:
        verify_files(row["source_sha256"])
    return selection


def review_case(native, root, row, *, preflight=False):
    import yaml

    task = dict(row["task"])
    task["image"] = row["image"]["repo_digests"][0]
    source = Path(row["m1_output"])
    output = root / ("preflight" if preflight else "outputs") / task["task_id"]
    output.mkdir(parents=True, exist_ok=True)
    client, container, env = native.docker.from_env(timeout=1800), None, None
    added = False
    try:
        image, added = native.get_image(client, task, output)
        if image.id != row["image"]["id"]:
            raise ValueError("Review image differs from frozen ft_m1 image")
        config = yaml.safe_load((Path(row["source_root"]) /
                                 "configs/native-deepseek-v4.1-flash.yaml").read_text())
        options = json.loads(json.dumps(config["environment"]))
        options["image"] = image.id
        options["run_args"] += ["--cpus=" + str(task["cpus"]),
                                "--memory=" + str(task["memory_mb"]) + "m"]
        env = native.get_environment(options)
        container = client.containers.get(env.container_id)
        isolation = native.prepare_repository(root, env, container, task, output)
        baseline = (output / "environment-baseline.patch").read_bytes()
        if baseline != (source / "environment-baseline.patch").read_bytes():
            raise ValueError("Image environment baseline differs from frozen ft_m1")
        base = isolation["synthetic_commit"]
        native.upload(container, baseline, "/tmp/ft-review-baseline.patch")
        native.upload(container, source / "model.patch", "/tmp/ft-review-p0.patch")
        receipt = json.loads(native.checked(env, normalization_command(
            base, "/tmp/ft-review-baseline.patch", "/tmp/ft-review-p0.patch",
            "/tmp/ft-review-normalized.patch",
        )).splitlines()[-1])
        bridge = IsolatedEnvironment(container, instance_id=task["task_id"])
        converted = output / "normalized-p0.patch"
        asyncio.run(bridge.download_file("/tmp/ft-review-normalized.patch", converted))
        canonical_baseline = output / "normalized-environment.patch"
        asyncio.run(bridge.download_file("/tmp/ft-review-normalized.patch.baseline", canonical_baseline))
        ledger = read(row["ledger_file"])
        provenance = {"m1_output": str(source), "original_p0_sha256": row["p0_sha256"],
                      "normalized_p0_sha256": sha256_bytes(converted.read_bytes()),
                      "original_baseline_sha256": sha256_bytes(baseline),
                      "v6_baseline_sha256": sha256_bytes(canonical_baseline.read_bytes()),
                      "ledger_sha256": row["ledger_sha256"], **receipt}
        save(output / "p0-normalization.json", provenance)
        fork = write_frozen_fork(
            output, task_id=task["task_id"], base_commit=base,
            baseline_patch=canonical_baseline.read_bytes(),
            ledger=ledger, candidate_patch=converted.read_bytes(), provenance=provenance,
        )
        if preflight:
            return {"task_id": task["task_id"], "passed": True, "model_api_calls": 0,
                    "candidate_unchanged": True, "isolation": isolation,
                    "normalization": provenance}
        agent = FrozenMiniReviewAgent(
            logs_dir=output / "agent/logs", model_name="deepseek-flash",
            stage_model_config=PROJECT / "configs/v6.1-deepseek-v4.1-flash.json",
            extra_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]}, fork_plan=str(fork),
        )
        context = SimpleNamespace(metadata={})
        asyncio.run(agent.run(task["instruction"], bridge, context))
        workflow = output / "agent/v6_review_repair"
        manifest = read(workflow / "manifest.json")
        if manifest["workflow_state"] != "completed":
            raise RuntimeError("Review did not complete; no fallback result is scored")
        # V6's P1 is a delta from the environment baseline. Export /app against
        # the original synthetic HEAD so the unchanged native scorer sees the
        # same complete-patch format as M0/M1, including committed/new files.
        patch = native.checked(env, native.snapshot_command(base))
        if manifest["route"] in {"accept", "hold"} and sha256_bytes(patch.encode()) != row["p0_sha256"]:
            raise RuntimeError("A non-repair route changed the frozen native P0")
        (output / "model.patch").write_text(patch)
        save(output / "review-result.json", {
            "status": "reviewed", "route": manifest["route"],
            "repair_executed": manifest["repair_executed"],
            "p0_sha256": row["p0_sha256"], "p1_sha256": sha256_bytes(patch.encode()),
            "ledger_sha256": row["ledger_sha256"], "metadata": context.metadata,
        })
        container.remove(force=True)
        container, env.container_id = None, None
        # Official test files enter a separate fresh container only after every
        # model stage has ended. No score enters I/R/S/Repair.
        result = native.grade(root, task, image.id, patch, output / "evaluation")
        result.update(p0_reward=row["m1_reward"], route=manifest["route"],
                      repair_executed=manifest["repair_executed"],
                      p0_sha256=row["p0_sha256"], p1_sha256=sha256_bytes(patch.encode()))
        save(output / "result.json", result)
        return result
    finally:
        if container is not None:
            container.remove(force=True)
        if env is not None:
            env.container_id = None
        if added and not preflight:
            try:
                client.images.remove(image.id, noprune=False)
            except native.docker.errors.APIError as error:
                save(output / "image-cleanup.json", {"error": str(error)})
        client.close()


def run_batch(root, selection, native, workers):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    preflight = read(root / "preflight/readiness.json")
    if not preflight["passed"] or preflight["selection_sha256"] != sha256_json(selection):
        raise ValueError("Run preflight for this frozen selection first")
    lock = threading.Lock()
    states = {row["task_id"]: {"phase": "queued"} for row in selection["tasks"]}
    results = {}

    def one(row):
        tid = row["task_id"]
        with lock:
            states[tid] = {"phase": "I/R_S_repair_then_verifier", "updated_at": native.now()}
            save(root / "monitor-status.json", {"workers": workers, "tasks": states})
        output = root / "outputs" / tid
        try:
            verify_files(row["source_sha256"])
            if (output / "result.json").exists():
                result = read(output / "result.json")
            else:
                result = review_case(native, root, row)
        except Exception as error:  # noqa: BLE001 - archive per-task stage/infrastructure failures
            result = {"task_id": tid, "status": "infrastructure_or_stage_error", "reward": None,
                      "error_type": type(error).__name__, "error": str(error)}
            save(output / "result.json", result)
        with lock:
            results[tid] = result
            states[tid] = {"phase": result["status"], "reward": result.get("reward"),
                           "updated_at": native.now()}
            save(root / "monitor-status.json", {"workers": workers, "tasks": states})
            save(root / "result.json", {"task_count": len(states), "terminal_count": len(results),
                                       "results": list(results.values()), "updated_at": native.now()})
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in selection["tasks"]]):
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-run", action="append", type=Path, default=[])
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--workers", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if not args.source_run:
            parser.error("prepare requires --source-run (repeat for the two ft_m1 batches)")
        selection = prepare(root, [p.resolve() for p in args.source_run], set(args.task_id or []))
    else:
        native = load_native(root)
        selection = validate_selection(root, native)
        if args.mode == "preflight":
            checks = [review_case(native, root, row, preflight=True) for row in selection["tasks"]]
            save(root / "preflight/readiness.json", {
                "passed": all(row["passed"] for row in checks), "model_api_calls": 0,
                "selection_sha256": sha256_json(selection), "checks": checks,
                "scope": "all P0 tree conversions and fresh-container isolation; no model review",
                "official_gold_control_reused_from": selection["official_preflight"],
            })
        else:
            run_batch(root, selection, native, args.workers)
    print(json.dumps({"mode": args.mode, "task_count": len(selection["tasks"]),
                      "experiment_id": "ft_m1_review"}))


if __name__ == "__main__":
    main()
