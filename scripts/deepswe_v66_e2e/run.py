"""V6.6 end to end on frozen V6.6 A outputs (ledger + check points from --a-root).

Per task: frozen ledger + check points (never resampled) -> native Mini Solver with the
ft_m1 ledger supplement plus the check-point section -> official P0 score -> V6.3 Mini
review with check points in I/R/S -> official P1 score.

prepare and preflight make no model calls; run is explicit. Outputs go to a new root.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.agents.mini_ft_review import (  # noqa: E402
    normalization_command,
    write_frozen_fork,
)
from external_agents_eval.agents.mini_v66_review import V66MiniReviewAgent  # noqa: E402
from external_agents_eval.benchmarks.deepswe.submission_filter import (  # noqa: E402
    filter_record,
    strip_added_tests,
)
from external_agents_eval.benchmarks.swebench.isolated_environment import (  # noqa: E402
    IsolatedEnvironment,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import (  # noqa: E402
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.v66_review_prompts import (  # noqa: E402
    SOLVER_HEADING,
    augment_solver_task,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v63 = load_module("v66e2e_v63_runner", PROJECT / "scripts/deepswe_v63_heldout/run.py")
ledger_input, read, save, load_native = v63.ledger_input, v63.read, v63.save, v63.load_native

EXPERIMENT_ID = "v6.6-e2e"
STAGE_MODEL_CONFIG = PROJECT / "configs/v6.1-deepseek-v4.1-flash.json"
FROZEN_SOURCES = tuple(dict.fromkeys(v63.FROZEN_SOURCES + (
    "scripts/deepswe_v66_e2e/run.py",
    "external_agents_eval/agents/mini_v66_review.py",
    "external_agents_eval/workflows/solve_review_repair/v66_review_prompts.py",
    "external_agents_eval/workflows/solve_review_repair/v66_review_checks.py",
    "external_agents_eval/agents/mini_v62.py",
    "external_agents_eval/agents/mini_v63.py",
    "external_agents_eval/benchmarks/deepswe/submission_filter.py",
)))
A_FILES = {
    "ledger.json": "ledger.json",
    "checkpoints.json": "v6_review_repair/v6.6/checkpoints/checkpoints.json",
    "review-targets.json": "v6_review_repair/v6.6/checkpoints/review-targets.json",
    "solver-view.md": "v6_review_repair/v6.6/checkpoints/solver-view.md",
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {name: file_sha256(PROJECT / name) for name in FROZEN_SOURCES}


def prepare(root, m0_root, a_root, v63_root, task_ids, note):
    import yaml

    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    if not read(m0_root / "preflight/readiness.json")["passed"]:
        raise ValueError("The M0 official gold/control preflight did not pass")
    for tid in task_ids:
        record = read(a_root / "a" / tid / "a-result.json")
        if record["status"] != "checkpoints_ready":
            raise ValueError(f"No valid V6.6 check points for {tid}: {record['status']}")
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
    (root / "configs").mkdir()
    native = load_native(root)
    tasks = {task["task_id"]: task for task in native.load_tasks(root)}
    config = native.prepare_config(root)
    if config != yaml.safe_load((m0_root / "configs/native-deepseek-v4.1-flash.yaml").read_text()):
        raise ValueError("Mini prompts, tools, model parameters, or budgets differ from M0")
    rows = []
    for tid in task_ids:
        frozen = root / "inputs/a" / tid
        frozen.mkdir(parents=True)
        for name, relative in A_FILES.items():
            shutil.copyfile(a_root / "a" / tid / relative, frozen / name)
        image = read(m0_root / "outputs" / tid / "image.json")
        ledger = read(frozen / "ledger.json")
        rows.append({
            "task_id": tid, "task": tasks[tid],
            "image": {"id": image["id"], "repo_digests": image["repo_digests"]},
            "a_source": str(a_root / "a" / tid),
            "a_files_sha256": {name: file_sha256(frozen / name) for name in A_FILES},
            "ledger_sha256": sha256_json(ledger),
            "m0_reward": read(m0_root / "outputs" / tid / "result.json").get("reward"),
            "v63_reward": (read(v63_root / "outputs" / tid / "result.json").get("reward")
                           if (v63_root / "outputs" / tid / "result.json").exists() else None),
        })
    selection = {
        "experiment_id": EXPERIMENT_ID, "note": note, "task_count": len(rows), "tasks": rows,
        "m0_root": str(m0_root), "a_root": str(a_root), "v63_root": str(v63_root),
        "native_source_sha256": {name: file_sha256(root / "scripts" / name) for name in copies},
        "code_sha256": code_hashes(),
        "official_preflight": str(m0_root / "preflight/readiness.json"),
        "pipeline": [
            "frozen A ledger + V6.6 check points from a_root (no resampling)",
            "Solver: native Mini M0 config; ft_m1 ledger supplement + '## Check points' section",
            "P0 official score (v6.6.4: newly added test files stripped before grading)",
            "Review: V6.3 FrozenMiniReviewAgent with check points appended to I/R (initial) and S prompts;"
            " v6.6.4 caps one I/R/S Bash result at 30,000 characters",
            "P1 official score (v6.6.4: newly added test files stripped before grading)",
        ],
    }
    save(root / "selection.json", selection)
    return selection


def validate(root, native):
    import yaml

    selection = read(root / "selection.json")
    if selection["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Wrong experiment")
    for name, digest in selection["native_source_sha256"].items():
        if file_sha256(root / "scripts" / name) != digest:
            raise ValueError("Frozen native source changed: " + name)
    if code_hashes() != selection["code_sha256"]:
        changed = [k for k, v in code_hashes().items() if selection["code_sha256"].get(k) != v]
        raise ValueError("Frozen experiment code changed: " + ", ".join(changed))
    native.load_tasks(root)
    config = native.prepare_config(root)
    m0 = Path(selection["m0_root"])
    if config != yaml.safe_load((m0 / "configs/native-deepseek-v4.1-flash.yaml").read_text()):
        raise ValueError("Mini configuration differs from M0")
    for row in selection["tasks"]:
        for name, digest in row["a_files_sha256"].items():
            if file_sha256(root / "inputs/a" / row["task_id"] / name) != digest:
                raise ValueError(f"Frozen A output changed: {row['task_id']}/{name}")
    return selection, config


def solver_input(root, row):
    frozen = root / "inputs/a" / row["task_id"]
    ledger = read(frozen / "ledger.json")
    augmented = augment_solver_task(ledger_input.augment_task(row["task"], ledger),
                                    (frozen / "solver-view.md").read_text())
    augmented["image"] = row["image"]["repo_digests"][0]
    return augmented, ledger


def submission(patch, output):
    """v6.6.4: the graded patch omits newly added test files; the raw patch stays model.patch."""
    submitted, removed = strip_added_tests(patch)
    (output / "submitted.patch").write_text(submitted)
    save(output / "submission-filter.json", filter_record(patch, submitted, removed))
    return submitted, removed


def run_solver(native, root, row, config):
    """M0's solve-then-grade loop on the ledger + check-point augmented task."""
    tid = row["task_id"]
    output = root / "solver" / tid
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("A Solver attempt already exists without a result; not resampled")
    augmented, ledger = solver_input(root, row)
    save(root / "inputs" / tid / "task.json", augmented)
    save(output / "input-audit.json", {
        "experiment_id": EXPERIMENT_ID, "task_id": tid,
        "augmented_instruction_sha256": sha256_bytes(augmented["instruction"].encode()),
        "ledger_sha256": row["ledger_sha256"], "a_files_sha256": row["a_files_sha256"],
        "evidence_packet_included": False, "checkpoint_section_included": True,
    })
    client = native.docker.from_env(timeout=1800)
    try:
        v63.require_image(client, row)
        image, added = native.get_image(client, augmented, output)
        if added or image.id != row["image"]["id"]:
            raise RuntimeError("Solver image is not the pinned M0 image")
        model_patch, info = native.generate(augmented, root, output, config, image, "generate")
        submitted, removed = submission(model_patch, output)
        result = native.grade(root, augmented, image.id, submitted, output / "evaluation")
        result.update(exit_status=info.get("exit_status"),
                      patch_sha256=sha256_bytes(model_patch.encode()),
                      submitted_patch_sha256=sha256_bytes(submitted.encode()),
                      removed_test_files=removed, finished_at=native.now())
    except Exception as error:  # noqa: BLE001 - retain per-task stage/infrastructure outcomes
        result = {"status": "infrastructure_or_stage_error", "task_id": tid,
                  "error_type": type(error).__name__, "error": str(error), "reward": None,
                  "finished_at": native.now()}
    finally:
        client.close()
    save(output / "result.json", result)
    return result


def review_case(native, root, row, *, solver_dir, preflight=False):
    """ft_m1_review.review_case with V66MiniReviewAgent; everything else unchanged."""
    import yaml

    task = dict(row["task"])
    task["image"] = row["image"]["repo_digests"][0]
    source = Path(solver_dir)
    output = root / ("preflight" if preflight else "outputs") / task["task_id"]
    output.mkdir(parents=True, exist_ok=True)
    client, container, env = native.docker.from_env(timeout=1800), None, None
    try:
        image, added = native.get_image(client, task, output)
        if added or image.id != row["image"]["id"]:
            raise ValueError("Review image differs from the pinned M0 image")
        config = yaml.safe_load((root / "configs/native-deepseek-v4.1-flash.yaml").read_text())
        options = json.loads(json.dumps(config["environment"]))
        options["image"] = image.id
        options["run_args"] += ["--cpus=" + str(task["cpus"]), "--memory=" + str(task["memory_mb"]) + "m"]
        env = native.get_environment(options)
        container = client.containers.get(env.container_id)
        isolation = native.prepare_repository(root, env, container, task, output)
        baseline = (output / "environment-baseline.patch").read_bytes()
        if baseline != (source / "environment-baseline.patch").read_bytes():
            raise ValueError("Image environment baseline differs from the Solver baseline")
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
        frozen = root / "inputs/a" / task["task_id"]
        ledger = read(frozen / "ledger.json")
        p0 = (source / "model.patch").read_bytes()
        provenance = {"m1_output": str(source), "original_p0_sha256": sha256_bytes(p0),
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
                    "candidate_unchanged": True, "normalization": provenance}
        agent = V66MiniReviewAgent(
            logs_dir=output / "agent/logs", model_name="deepseek-flash",
            stage_model_config=STAGE_MODEL_CONFIG,
            extra_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]}, fork_plan=str(fork),
            checkpoints=str(frozen / "review-targets.json"),
        )
        context = SimpleNamespace(metadata={})
        asyncio.run(agent.run(task["instruction"], bridge, context))
        manifest = read(output / "agent/v6_review_repair/manifest.json")
        if manifest["workflow_state"] != "completed":
            raise RuntimeError("Review did not complete; no fallback result is scored")
        patch = native.checked(env, native.snapshot_command(base))
        if manifest["route"] in {"accept", "hold"} and sha256_bytes(patch.encode()) != sha256_bytes(p0):
            raise RuntimeError("A non-repair route changed the frozen P0")
        (output / "model.patch").write_text(patch)
        container.remove(force=True)
        container, env.container_id = None, None
        submitted, removed = submission(patch, output)
        result = native.grade(root, task, image.id, submitted, output / "evaluation")
        result.update(route=manifest["route"], repair_executed=manifest["repair_executed"],
                      p0_sha256=sha256_bytes(p0), p1_sha256=sha256_bytes(patch.encode()),
                      submitted_patch_sha256=sha256_bytes(submitted.encode()), removed_test_files=removed,
                      checkpoint_delivery=delivery_audit(output / "agent/v6_review_repair"))
        return result
    finally:
        if container is not None:
            container.remove(force=True)
        if env is not None:
            env.container_id = None
        client.close()


def delivery_audit(workflow):
    """What the reviewers actually received and returned (the controller prompt is not enough)."""
    transcripts = sorted(workflow.rglob("mini.json"))
    texts = [path.read_text(errors="replace") for path in transcripts]
    errors = sorted(workflow.rglob("checkpoint-results-errors.json"))
    return {"mini_transcripts": len(texts),
            "with_batch_checkpoints": sum("CHECK POINTS FOR THIS BATCH" in text for text in texts),
            "with_checkpoint_results": sum("checkpoint_results" in text for text in texts),
            "result_correction_rounds": sum(len(json.loads(path.read_text())) for path in errors)}


def run_case(native, root, row, config, update):
    tid = row["task_id"]
    stage = "Solver"
    try:
        s_path = root / "solver" / tid / "result.json"
        if s_path.exists():
            solver = read(s_path)
        else:
            update(tid, "solving_then_p0_score")
            solver = run_solver(native, root, row, config)
        if solver["status"] == "infrastructure_or_stage_error":
            return {**solver, "stage": "Solver"}
        stage = "review"
        update(tid, "I/R_S_repair_then_verifier", p0_reward=solver.get("reward"))
        v63.ensure_image(native, row)
        result = review_case(native, root, row, solver_dir=root / "solver" / tid)
        result.update(experiment_id=EXPERIMENT_ID, p0_reward=solver.get("reward"),
                      p0_status=solver.get("status"), solver_exit_status=solver.get("exit_status"),
                      m0_reward=row["m0_reward"], v63_reward=row["v63_reward"])
        return result
    except Exception as error:  # noqa: BLE001 - archive per-task stage/infrastructure failures
        return {"task_id": tid, "status": "infrastructure_or_stage_error", "stage": stage,
                "reward": None, "error_type": type(error).__name__, "error": str(error),
                "finished_at": native.now()}


def run_batch(root, selection, native, config, workers):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    readiness = read(root / "preflight/readiness.json")
    if not readiness["passed"] or readiness["selection_sha256"] != sha256_json(selection):
        raise ValueError("Run the zero-API preflight for this frozen selection first")
    lock = threading.Lock()
    states = {row["task_id"]: {"phase": "queued"} for row in selection["tasks"]}
    results = {}

    def update(tid, phase, **values):
        with lock:
            states[tid] = {**states[tid], "phase": phase, "updated_at": native.now(), **values}
            save(root / "monitor-status.json", {"workers": workers, "tasks": states})

    def one(row):
        tid = row["task_id"]
        path = root / "outputs" / tid / "result.json"
        result = read(path) if path.exists() else run_case(native, root, row, config, update)
        save(path, result)
        with lock:
            results[tid] = result
        update(tid, result["status"], reward=result.get("reward"), p0_reward=result.get("p0_reward"))
        with lock:
            save(root / "result.json", {
                "experiment_id": EXPERIMENT_ID, "task_count": len(states),
                "terminal_count": len(results),
                "p1_passed": sum(r.get("reward") == 1 for r in results.values()),
                "p0_passed": sum(r.get("p0_reward") == 1 for r in results.values()),
                "results": list(results.values()), "updated_at": native.now(),
            })
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in selection["tasks"]]):
            future.result()
    save(root / "launcher-result.json", {"completed": True, "terminal_count": len(results),
                                         "finished_at": native.now()})


def preflight(root, selection, native, config):
    from external_agents_eval.agents.mini_v66_review import insert_batch_section
    from external_agents_eval.workflows.solve_review_repair.v66_review_checks import batch_section
    from external_agents_eval.workflows.solve_review_repair.v66_review_prompts import summary_addendum

    if not read(selection["official_preflight"])["passed"]:
        raise ValueError("The M0 official gold/control preflight is not valid")
    client = native.docker.from_env(timeout=1800)
    try:
        images = [v63.require_image(client, row).id for row in selection["tasks"]]
    finally:
        client.close()
    addenda = {}
    for row in selection["tasks"]:
        points = read(root / "inputs/a" / row["task_id"] / "review-targets.json")
        targets = {target for point in points for target in point["targets"]}
        # Render through the adapter-level insertion used at run time.
        adapted = insert_batch_section("BODY\n\nV6.3 FILE DELIVERY (rules)\n", batch_section(points, targets))
        addenda[row["task_id"]] = {"batch_section_in_adapted_prompt": "CHECK POINTS FOR THIS BATCH" in adapted,
                                   "batch_chars": len(adapted), "S_chars": len(summary_addendum(points)),
                                   "points": len(points)}
    row = selection["tasks"][0]
    output = root / "preflight" / row["task_id"]
    augmented, _ = solver_input(root, row)
    client = native.docker.from_env(timeout=1800)
    try:
        image, added = native.get_image(client, augmented, output / "solver-offline")
        if added or image.id != row["image"]["id"]:
            raise ValueError("Preflight image differs from M0")
        _, protocol = native.generate(augmented, root, output / "solver-offline", config, image, "offline")
    finally:
        client.close()
    trajectory = read(output / "solver-offline/native.traj.json")
    user = next(m["content"] for m in trajectory["messages"] if m["role"] == "user")
    injected = (row["task"]["instruction"] in user and user.count(ledger_input.HEADING) == 1
                and user.count(SOLVER_HEADING) == 1)
    handoff = review_case(native, root, row, solver_dir=output / "solver-offline", preflight=True)
    passed = bool(injected and handoff["passed"]
                  and all(v["batch_section_in_adapted_prompt"] for v in addenda.values()))
    save(root / "preflight/readiness.json", {
        "passed": passed, "experiment_id": EXPERIMENT_ID, "model_api_calls": 0,
        "selection_sha256": sha256_json(selection), "images_verified": len(images),
        "solver_offline_protocol": protocol, "ledger_and_checkpoints_injected_once": injected,
        "review_handoff": handoff, "addenda": addenda,
        "official_gold_control_reused_from": selection["official_preflight"],
        "finished_at": native.now(),
    })
    if not passed:
        raise RuntimeError("Preflight failed; see preflight/readiness.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--m0-root", type=Path)
    parser.add_argument("--a-root", type=Path)
    parser.add_argument("--v63-root", type=Path)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--note")
    parser.add_argument("--workers", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if not (args.m0_root and args.a_root and args.v63_root and args.task_id):
            parser.error("prepare requires --m0-root, --a-root, --v63-root and --task-id")
        selection = prepare(root, args.m0_root.resolve(), args.a_root.resolve(), args.v63_root.resolve(),
                            args.task_id, args.note)
    else:
        native = load_native(root)
        selection, config = validate(root, native)
        if args.mode == "preflight":
            preflight(root, selection, native, config)
        else:
            run_batch(root, selection, native, config, args.workers)
    print(json.dumps({"mode": args.mode, "experiment_id": EXPERIMENT_ID,
                      "task_count": selection["task_count"]}))


if __name__ == "__main__":
    main()
