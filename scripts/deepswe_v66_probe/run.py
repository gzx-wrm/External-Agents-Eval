"""V6.6 probe stage P on frozen V6.6.1 A outputs (no Solver, review or scoring).

Per task: open check points -> native Mini prober in an offline baseline container ->
/tmp/v66-probe copied out -> every claimed observation re-run in a fresh baseline
container -> verified readings become inferred analog check points. Outputs use the A
layout (a/<task>/...), so the V6.6 end-to-end runner can take this root as --a-root.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_json  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.v66_checkpoints import (  # noqa: E402
    delivery_views,
)
from external_agents_eval.workflows.solve_review_repair.v66_probe import (  # noqa: E402
    PROBE_DIR,
    RESULT_PATH,
    convert,
    load_result,
    open_points,
    probe_task,
    validate_result,
    verify,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v63 = load_module("v66probe_v63_runner", PROJECT / "scripts/deepswe_v63_heldout/run.py")
read, save, load_native = v63.read, v63.save, v63.load_native

EXPERIMENT_ID = "v6.6-probe"
# Allocation by the number of open check points (approved 2026-11-04).
PROBE_STEPS_BASE, PROBE_STEPS_PER_POINT, PROBE_STEPS_CAP = 20, 15, 160
PROBE_WALL_BASE_SEC, PROBE_WALL_PER_POINT_SEC, PROBE_WALL_CAP_SEC = 600, 120, 2400
PROBE_STEP_LIMIT = PROBE_STEPS_CAP  # recorded in the config; run_probe sets the per-task value
PROBE_WALL_TIME_SEC = PROBE_WALL_CAP_SEC
VERIFY_TIMEOUT_SEC = 300
SOLVER_VIEW_POINTS = 12
FROZEN_SOURCES = tuple(dict.fromkeys(v63.FROZEN_SOURCES + (
    "scripts/deepswe_v66_probe/run.py",
    "external_agents_eval/workflows/solve_review_repair/v66_probe.py",
    "external_agents_eval/workflows/solve_review_repair/v66_checkpoints.py",
    "external_agents_eval/workflows/solve_review_repair/v66_review_prompts.py",
)))
A_FILES = {"ledger.json": "ledger.json",
           "checkpoints.json": "v6_review_repair/v6.6/checkpoints/checkpoints.json",
           "review-targets.json": "v6_review_repair/v6.6/checkpoints/review-targets.json",
           "solver-view.md": "v6_review_repair/v6.6/checkpoints/solver-view.md"}
PROBE_HEADER = """{{task}}

You can execute bash commands. Work step by step: find the analog, write the probe under
/tmp/v66-probe/, run it, read its output, then record the result. When /tmp/v66-probe/result.json
is complete, finish by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

"""


def probe_limits(open_count):
    return {"step_limit": min(PROBE_STEPS_CAP, PROBE_STEPS_BASE + PROBE_STEPS_PER_POINT * open_count),
            "wall_time_limit_seconds": min(PROBE_WALL_CAP_SEC,
                                           PROBE_WALL_BASE_SEC + PROBE_WALL_PER_POINT_SEC * open_count)}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {name: file_sha256(PROJECT / name) for name in FROZEN_SOURCES}


def probe_config(config):
    """M0's Mini model/environment/templates; only the task section and limits differ."""
    result = copy.deepcopy(config)
    template = config["agent"]["instance_template"]
    marker = "## Command Execution Rules"
    if marker not in template:
        raise ValueError("Unexpected upstream Mini instance template")
    result["agent"]["instance_template"] = PROBE_HEADER + template[template.index(marker):]
    result["agent"].update(step_limit=PROBE_STEP_LIMIT, wall_time_limit_seconds=PROBE_WALL_TIME_SEC,
                           cost_limit=0)
    return result


def prepare(root, m0_root, a_root, task_ids, note):
    import yaml

    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    for tid in task_ids:
        if read(a_root / "a" / tid / "a-result.json")["status"] != "checkpoints_ready":
            raise ValueError(f"No valid V6.6.1 check points for {tid}")
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
        raise ValueError("Mini configuration differs from M0")
    (root / "configs/probe-mini.yaml").write_text(
        yaml.safe_dump(probe_config(config), allow_unicode=True, sort_keys=False))
    rows = []
    for tid in task_ids:
        frozen = root / "inputs/a" / tid
        frozen.mkdir(parents=True)
        for name, relative in A_FILES.items():
            shutil.copyfile(a_root / "a" / tid / relative, frozen / name)
        image = read(m0_root / "outputs" / tid / "image.json")
        rows.append({"task_id": tid, "task": tasks[tid],
                     "image": {"id": image["id"], "repo_digests": image["repo_digests"]},
                     "a_source": str(a_root / "a" / tid),
                     "a_files_sha256": {name: file_sha256(frozen / name) for name in A_FILES},
                     "open_points": len(open_points(read(frozen / "review-targets.json")))})
    selection = {"experiment_id": EXPERIMENT_ID, "note": note, "task_count": len(rows), "tasks": rows,
                 "m0_root": str(m0_root), "a_root": str(a_root), "code_sha256": code_hashes(),
                 "native_source_sha256": {n: file_sha256(root / "scripts" / n) for n in copies},
                 "probe_config_sha256": file_sha256(root / "configs/probe-mini.yaml"),
                 "limits": {"steps": f"min({PROBE_STEPS_CAP}, {PROBE_STEPS_BASE} + {PROBE_STEPS_PER_POINT} x open)",
                            "wall_time_sec": f"min({PROBE_WALL_CAP_SEC}, {PROBE_WALL_BASE_SEC} + {PROBE_WALL_PER_POINT_SEC} x open)",
                            "verify_timeout_sec": VERIFY_TIMEOUT_SEC}}
    save(root / "selection.json", selection)
    return selection


def validate(root, native):
    import yaml

    selection = read(root / "selection.json")
    if selection["experiment_id"] != EXPERIMENT_ID or code_hashes() != selection["code_sha256"]:
        raise ValueError("Wrong experiment or frozen code changed")
    for name, digest in selection["native_source_sha256"].items():
        if file_sha256(root / "scripts" / name) != digest:
            raise ValueError("Frozen native source changed: " + name)
    if file_sha256(root / "configs/probe-mini.yaml") != selection["probe_config_sha256"]:
        raise ValueError("Probe configuration changed")
    native.load_tasks(root)
    for row in selection["tasks"]:
        for name, digest in row["a_files_sha256"].items():
            if file_sha256(root / "inputs/a" / row["task_id"] / name) != digest:
                raise ValueError(f"Frozen A output changed: {row['task_id']}/{name}")
    return selection, yaml.safe_load((root / "configs/probe-mini.yaml").read_text())


def baseline_environment(native, root, row, config, output):
    output.mkdir(parents=True, exist_ok=True)
    task = dict(row["task"])
    task["image"] = row["image"]["repo_digests"][0]
    options = copy.deepcopy(config["environment"])
    options["image"] = row["image"]["id"]
    options["run_args"] += ["--cpus=" + str(task["cpus"]), "--memory=" + str(task["memory_mb"]) + "m"]
    environment = native.get_environment(options)
    client = native.docker.from_env(timeout=1800)
    try:
        container = client.containers.get(environment.container_id)
    finally:
        client.close()
    native.prepare_repository(root, environment, container, task, output)
    return environment, container


def close(native, environment):
    if environment is not None and environment.container_id:
        client = native.docker.from_env(timeout=300)
        try:
            client.containers.get(environment.container_id).remove(force=True)
        except native.docker.errors.NotFound:
            pass
        finally:
            client.close()
        environment.container_id = None


def run_probe(native, root, row, config, output, *, model_patch=None):
    """Prober in container A, then finish(): verification in a fresh container B."""
    frozen = root / "inputs/a" / row["task_id"]
    points, ledger = read(frozen / "review-targets.json"), read(frozen / "ledger.json")
    output.mkdir(parents=True, exist_ok=True)
    record = {"task_id": row["task_id"], "open_points": len(open_points(points)), "started_at": native.now()}
    result_text, archive = None, None
    if open_points(points):
        task_text = probe_task(row["task"], ledger, points)
        (output / "probe-task.md").write_text(task_text)
        environment = None
        try:
            environment, container = baseline_environment(native, root, row, config, output / "prober")
            limits = probe_limits(len(open_points(points)))
            record["limits"] = limits
            agent = native.DefaultAgent(native.get_model(config=config["model"]), environment,
                                        **{**config["agent"], **limits}, output_path=output / "probe.traj.json")
            if model_patch is not None:
                with patch("litellm.completion", model_patch):
                    info = agent.run(task_text)
            else:
                info = agent.run(task_text)
            record["exit_status"] = info.get("exit_status")
            raw = environment.execute({"command": f"cat {RESULT_PATH}"}, timeout=60)
            (output / "result.json").write_text(raw["output"])
            result_text = raw["output"] if raw["returncode"] == 0 else None
            try:
                chunks, _ = container.get_archive(PROBE_DIR)
                archive = b"".join(chunks)
                (output / "probe-files.tar").write_bytes(archive)
            except native.docker.errors.NotFound:
                archive = None
        finally:
            close(native, environment)
    return finish(native, root, row, config, output, record, result_text, archive)


def finish(native, root, row, config, output, record, result_text, archive):
    """Validate the prober's claims, re-run them on a fresh baseline, convert, write."""
    frozen = root / "inputs/a" / row["task_id"]
    points = read(frozen / "review-targets.json")
    checked: dict = {}
    if open_points(points):
        resolved, problems = validate_result(load_result(result_text) if result_text else None, points)
        record.update(problems=problems, claimed_resolved=sorted(resolved))
        if resolved and archive is not None:
            environment = None
            try:
                environment, container = baseline_environment(native, root, row, config, output / "verifier")
                container.put_archive("/tmp", archive)

                def rerun(command):
                    result = environment.execute({"command": command}, timeout=VERIFY_TIMEOUT_SEC)
                    return result["output"]

                checked = verify(resolved, rerun)
            finally:
                close(native, environment)
        save(output / "verification.json", checked)
    converted = convert(points, checked)
    views = delivery_views({"checkpoints": converted}, SOLVER_VIEW_POINTS)
    target = root / "a" / row["task_id"]
    layout = target / "v6_review_repair/v6.6/checkpoints"
    layout.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(frozen / "ledger.json", target / "ledger.json")
    save(layout / "checkpoints.json", {"checkpoints": converted})
    save(layout / "review-targets.json", views["review_targets"])
    (layout / "solver-view.md").write_text(views["solver_text"] + "\n")
    record.update(verified=sorted(cp for cp, entry in checked.items() if entry["verified"]),
                  unverified=sorted(cp for cp, entry in checked.items() if not entry["verified"]),
                  status="checkpoints_ready", finished_at=native.now())
    save(target / "a-result.json", record)
    return record


def reverify(root, selection, native, config, source):
    """Re-validate and re-verify a finished run's prober outputs; no model call, no resampling."""
    results = []
    for row in selection["tasks"]:
        tid = row["task_id"]
        old = source / "probe" / tid
        output = root / "probe" / tid
        if output.exists():
            raise RuntimeError(f"{tid} already has probe outputs here; use a new root")
        output.mkdir(parents=True)
        for name in ("probe-task.md", "probe.traj.json", "result.json", "probe-files.tar"):
            if (old / name).exists():
                shutil.copyfile(old / name, output / name)
        before = read(source / "a" / tid / "a-result.json")
        if before["open_points"] and not (output / "probe-task.md").exists():
            raise RuntimeError(f"{tid}: the source run has no prober outputs")
        text = (output / "result.json").read_text() if (output / "result.json").exists() else None
        archive = (output / "probe-files.tar").read_bytes() if (output / "probe-files.tar").exists() else None
        record = {"task_id": tid, "open_points": before["open_points"], "exit_status": before.get("exit_status"),
                  "reverified_from": str(source / "probe" / tid), "started_at": native.now()}
        if before["open_points"]:
            v63.ensure_image(native, row)
        results.append(finish(native, root, row, config, output, record, text, archive))
        print(json.dumps({"task_id": tid, "verified": results[-1].get("verified")}), flush=True)
    save(root / "result.json", {"experiment_id": EXPERIMENT_ID, "reverified_from": str(source),
                                "results": results, "updated_at": native.now()})


def run_batch(root, selection, native, config, workers):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    readiness = read(root / "preflight/readiness.json")
    if not readiness["passed"] or readiness["selection_sha256"] != sha256_json(selection):
        raise ValueError("Run the zero-API preflight for this frozen selection first")
    lock, results = threading.Lock(), {}

    def one(row):
        tid = row["task_id"]
        path = root / "a" / tid / "a-result.json"
        try:
            if path.exists():
                result = read(path)
            elif (root / "probe" / tid).exists():
                raise RuntimeError("A probe attempt exists without a result; not resampled")
            else:
                v63.ensure_image(native, row)
                result = run_probe(native, root, row, config, root / "probe" / tid)
        except Exception as error:  # noqa: BLE001 - keep per-task failures explicit
            result = {"task_id": tid, "status": "stage_error", "error_type": type(error).__name__,
                      "error": str(error)}
            save(root / "errors" / f"{tid}.json", result)
        with lock:
            results[tid] = result
            save(root / "result.json", {"experiment_id": EXPERIMENT_ID, "results": list(results.values()),
                                        "updated_at": native.now()})
        print(json.dumps({"task_id": tid, "status": result["status"], "verified": result.get("verified")}),
              flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in selection["tasks"]]):
            future.result()


def preflight(root, selection, native, config):
    """Stub model: write a resolved entry with a real probe file, then submit (0 API calls)."""
    import litellm

    row = next(row for row in selection["tasks"] if row["open_points"])
    points = read(root / "inputs/a" / row["task_id"] / "review-targets.json")
    cp_id, point = open_points(points)[0]
    token = "PROBE-OK-" + uuid.uuid4().hex[:8]
    result = {"checkpoints": [
        {"cp": cp, "decision": "open", "reading": None, "analog": None, "probe": None, "reason": "stub"}
        for cp, _ in open_points(points) if cp != cp_id] + [
        {"cp": cp_id, "decision": "resolved", "reading": point["alternatives"][0],
         "analog": {"path": "README.md", "line_start": 1, "line_end": 1, "same_kind_reason": "preflight stub"},
         "probe": {"command": f"cat {PROBE_DIR}/ok.txt", "observation_quote": token}, "reason": "stub"}]}
    commands = [f"mkdir -p {PROBE_DIR} && printf '%s\\n' {token} > {PROBE_DIR}/ok.txt && cat > {RESULT_PATH} <<'EOF'\n"
                + json.dumps(result) + "\nEOF", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        command = commands[min(len(calls), len(commands)) - 1]
        return litellm.ModelResponse(model=kwargs["model"], choices=[{
            "index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Preflight probe.", "reasoning_content": "",
                "tool_calls": [{"id": f"pre-{len(calls)}", "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})}}],
            }}], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    scratch = root / "preflight-run"
    record = run_probe(native, scratch_root(root, scratch), row, config, scratch / "probe" / row["task_id"],
                       model_patch=completion)
    converted = read(scratch / "a" / row["task_id"] / "v6_review_repair/v6.6/checkpoints/checkpoints.json")
    became_analog = any(p.get("probe", {}).get("cp") == cp_id and p["resolution"] == "analog"
                        for p in converted["checkpoints"])
    passed = record["verified"] == [cp_id] and became_analog and len(calls) == 2
    save(root / "preflight/readiness.json", {
        "passed": passed, "experiment_id": EXPERIMENT_ID, "model_api_calls": 0, "simulated_responses": len(calls),
        "selection_sha256": sha256_json(selection), "task_id": row["task_id"], "record": record,
        "verified_in_fresh_container": record["verified"], "converted_to_analog": became_analog,
        "finished_at": native.now()})
    if not passed:
        raise RuntimeError("Preflight failed; see preflight/readiness.json")


def scratch_root(root, scratch):
    """Preflight writes into a scratch tree that shares the frozen inputs and native files."""
    scratch.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "scripts", "grader", "configs", "manifests"):
        link = scratch / name
        if not link.exists():
            link.symlink_to(root / name, target_is_directory=True)
    return scratch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "run", "reverify"])
    parser.add_argument("--from-root", type=Path, help="reverify: finished probe root to re-check")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--m0-root", type=Path)
    parser.add_argument("--a-root", type=Path)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--note")
    parser.add_argument("--workers", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if not (args.m0_root and args.a_root and args.task_id):
            parser.error("prepare requires --m0-root, --a-root and --task-id")
        selection = prepare(root, args.m0_root.resolve(), args.a_root.resolve(), args.task_id, args.note)
    else:
        native = load_native(root)
        selection, config = validate(root, native)
        if args.mode == "preflight":
            preflight(root, selection, native, config)
        elif args.mode == "reverify":
            if not args.from_root:
                parser.error("reverify requires --from-root")
            reverify(root, selection, native, config, args.from_root.resolve())
        else:
            run_batch(root, selection, native, config, args.workers)
    print(json.dumps({"mode": args.mode, "experiment_id": EXPERIMENT_ID, "task_count": selection["task_count"]}))


if __name__ == "__main__":
    main()
