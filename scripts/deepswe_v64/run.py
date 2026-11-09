"""V6.4 on the fixed DeepSWE eval30: frozen FT ledger -> Solver -> G(P0) + I/R -> S
-> route -> Repair -> G(P1) -> official verifier.

Each slot runs one task end to end: native Mini Solver with the frozen FT ledger,
then the V6.4 review in a fresh container, then the unchanged official verifier in
another fresh container. P0 is scored separately for analysis only, after every
model stage of that task has ended. prepare/preflight make no model calls.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

from external_agents_eval.agents.mini_ft_review import normalization_command, write_frozen_fork  # noqa: E402
from external_agents_eval.agents.mini_v64 import MiniV64ReviewAgent  # noqa: E402
from external_agents_eval.benchmarks.swebench.isolated_environment import IsolatedEnvironment  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.artifacts import (  # noqa: E402
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.build_gate import BuildGate  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.v64_config import (  # noqa: E402
    V64_BUILD_GATE_TIMEOUT_SEC,
)
from gate_executor import DockerGateExecutor  # noqa: E402
from ledger_input_v64 import GUIDANCE, HEADING, augment_task, load_frozen_ledger  # noqa: E402

EXPERIMENT_ID = "v6.4"
SMOKE_EXPERIMENT_ID = "v6.4-smoke"
SCRIPTS = ("m0_run.py", "native_compat.py", "sanitize_repository.py")
MANIFESTS = ("public-tasks.json", "eval30_current.json", "task-file-sha256.json")
STAGE_MODEL_CONFIG = PROJECT / "configs/v6.1-deepseek-v4.1-flash.json"
FROZEN_SOURCES = [
    "scripts/deepswe_v64/run.py",
    "scripts/deepswe_v64/gate_executor.py",
    "scripts/deepswe_v64/ledger_input_v64.py",
    "external_agents_eval/agents/mini_v64.py",
    "external_agents_eval/agents/mini_v63.py",
    "external_agents_eval/agents/mini_v62.py",
    "external_agents_eval/agents/mini_ft_review.py",
    "external_agents_eval/agents/mini_review_files.py",
    "external_agents_eval/agents/pier/pi_role_runner.py",
    "external_agents_eval/agents/pier/pi_v6_solve_review_repair.py",
    "external_agents_eval/workflows/solve_review_repair/v5_controller.py",
    "external_agents_eval/workflows/solve_review_repair/v5_prompts.py",
    "external_agents_eval/workflows/solve_review_repair/v5_config.py",
    "external_agents_eval/workflows/solve_review_repair/v5_contracts.py",
    "external_agents_eval/workflows/solve_review_repair/v6_controller.py",
    "external_agents_eval/workflows/solve_review_repair/v64_config.py",
    "external_agents_eval/workflows/solve_review_repair/build_gate.py",
    "configs/v6.1-deepseek-v4.1-flash.json",
]
SKILL_DIR = "external_agents_eval/workflows/solve_review_repair/skills/v6.4-review-note-files"


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


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha256(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sha256_bytes(b"".join(
        p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0" for p in files))


def code_hashes():
    hashes = {name: file_sha256(PROJECT / name) for name in FROZEN_SOURCES}
    hashes[SKILL_DIR + "/"] = tree_sha256(PROJECT / SKILL_DIR)
    return hashes


def load_native(root):
    if importlib.metadata.version("mini-swe-agent") != "2.4.6":
        raise ValueError("This experiment requires mini-SWE-agent 2.4.6")
    path = Path(root) / "scripts/m0_run.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"v64_frozen_m0_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# prepare: bind the fixed eval30, frozen FT ledgers and unchanged native runtime
# ---------------------------------------------------------------------------

def prepare(root, sources, smoke_task=None):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty v6.4 experiment directory")
    rows, native_hashes, m0_root = {}, None, None
    for source in sources:
        source = Path(source).resolve()
        selection = read(source / "selection.json")
        if selection["experiment_id"] != "ft_m1":
            raise ValueError("Ledger sources must be the frozen ft_m1 batches")
        native = load_native(source)
        tasks = {task["task_id"]: task for task in native.load_tasks(source)}
        hashes = {name: file_sha256(source / "scripts" / name) for name in SCRIPTS}
        base = Path(selection["m0_root"]).resolve()
        if native_hashes is not None and (hashes != native_hashes or base != m0_root):
            raise ValueError("ft_m1 sources use different native runtimes or M0 roots")
        native_hashes, m0_root = hashes, base
        for record in selection["tasks"]:
            tid = record["task_id"]
            if tid in rows:
                raise ValueError("Task appears in more than one ledger source: " + tid)
            task = tasks[tid]
            if task["task_checksum"] != record["task_checksum"]:
                raise ValueError("Task identity differs from ft_m1: " + tid)
            ledger_path = source / record["ledger_file"]
            load_frozen_ledger(ledger_path, file_sha256=record["ledger_file_sha256"],
                               ledger_sha256=record["ledger_sha256"])
            ft_output = source / "outputs" / tid
            rows[tid] = {
                "task_id": tid, "task_checksum": task["task_checksum"],
                "base_commit": task["base_commit"], "image": record["m0_image"],
                "ledger_source": str(ledger_path), "ledger_file_sha256": record["ledger_file_sha256"],
                "ledger_sha256": record["ledger_sha256"], "ledger_quality": record.get("ledger_quality"),
                # Used only by the zero-API gate preflight, never as model input.
                "ft_m1_output": str(ft_output),
                "ft_m1_p0_sha256": file_sha256(ft_output / "model.patch"),
            }
    order = read(m0_root / "manifests/eval30_current.json")["task_ids"]
    if len(order) != 30 or set(order) != set(rows):
        raise ValueError("The ft_m1 ledgers must cover exactly the fixed eval30")
    if smoke_task is not None:
        if smoke_task not in order:
            raise ValueError("Smoke task is outside the fixed eval30")
        # A separate end-to-end validation directory; never merged into eval30 results.
        order = [smoke_task]
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir()
    for name in SCRIPTS:
        shutil.copyfile(Path(sources[0]) / "scripts" / name, root / "scripts" / name)
    (root / "manifests").mkdir()
    for name in MANIFESTS:
        shutil.copyfile(m0_root / "manifests" / name, root / "manifests" / name)
    (root / "grader").symlink_to(m0_root / "grader", target_is_directory=True)
    (root / "configs").mkdir()
    native = load_native(root)
    tasks = {task["task_id"]: task for task in native.load_tasks(root)}
    config = native.prepare_config(root)
    import yaml
    original = yaml.safe_load((m0_root / "configs/native-deepseek-v4.1-flash.yaml").read_text())
    if config != original:
        raise ValueError("Mini prompts, tools, model parameters or budgets differ from M0")
    ordered = []
    for tid in order:
        row = dict(rows[tid])
        destination = root / "inputs/ledgers" / f"{tid}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row.pop("ledger_source"), destination)
        row["ledger_file"] = str(destination.relative_to(root))
        ledger = load_frozen_ledger(destination, file_sha256=row["ledger_file_sha256"],
                                   ledger_sha256=row["ledger_sha256"])
        task = dict(tasks[tid])
        task["image"] = row["image"]["repo_digests"][0]
        augmented = augment_task(task, ledger)
        save(root / "inputs" / tid / "original-task.json", task)
        save(root / "inputs" / tid / "task.json", augmented)
        row["original_instruction_sha256"] = hashlib.sha256(task["instruction"].encode()).hexdigest()
        row["solver_instruction_sha256"] = hashlib.sha256(augmented["instruction"].encode()).hexdigest()
        ordered.append(row)
    selection = {
        "experiment_id": SMOKE_EXPERIMENT_ID if smoke_task else EXPERIMENT_ID,
        "task_count": len(ordered), "tasks": ordered,
        "pipeline": "frozen FT ledger -> native Mini Solver -> G(P0) + I/R -> S -> route -> "
                    "Repair -> G(P1) -> unchanged official verifier",
        "m0_root": str(m0_root), "official_preflight": str(m0_root / "preflight/readiness.json"),
        "native_source_sha256": native_hashes, "code_sha256": code_hashes(),
        "solver_guidance_sha256": sha256_bytes((HEADING + "\n\n" + GUIDANCE).encode()),
        "a_rerun": False, "ledger_modified": False, "evidence_packet_included": False,
        "p0_scored_for_analysis_only": True, "workers": 3,
        "build_gate_timeout_sec": V64_BUILD_GATE_TIMEOUT_SEC,
    }
    save(root / "selection.json", selection)
    save(root / "preflight/plan.json", {
        "experiment_id": selection["experiment_id"], "task_ids": order, "model_api_calls": 0,
        "native_configuration_identical_to_m0": True,
        "solver_input_change": "frozen FT ledger plus V6.4 English guidance",
    })
    return selection


def validate(root, native):
    selection = read(root / "selection.json")
    if selection["experiment_id"] not in {EXPERIMENT_ID, SMOKE_EXPERIMENT_ID}:
        raise ValueError("Wrong experiment directory")
    if selection["experiment_id"] == EXPERIMENT_ID and len(selection["tasks"]) != 30:
        raise ValueError("The formal v6.4 run covers the full eval30")
    for name, digest in selection["native_source_sha256"].items():
        if file_sha256(root / "scripts" / name) != digest:
            raise ValueError("Frozen native runtime changed: " + name)
    if code_hashes() != selection["code_sha256"]:
        changed = [k for k, v in code_hashes().items() if selection["code_sha256"].get(k) != v]
        raise ValueError("V6.4 source changed after prepare: " + ", ".join(changed))
    native.load_tasks(root)
    for row in selection["tasks"]:
        load_frozen_ledger(root / row["ledger_file"], file_sha256=row["ledger_file_sha256"],
                           ledger_sha256=row["ledger_sha256"])
    return selection


# ---------------------------------------------------------------------------
# shared review-container steps
# ---------------------------------------------------------------------------

def open_review_container(native, root, task, image, config, output):
    Path(output).mkdir(parents=True, exist_ok=True)
    options = json.loads(json.dumps(config["environment"]))
    options["image"] = image.id
    options["run_args"] += ["--cpus=" + str(task["cpus"]), "--memory=" + str(task["memory_mb"]) + "m"]
    env = native.get_environment(options)
    client = native.docker.from_env(timeout=1800)
    try:
        container = client.containers.get(env.container_id)
    finally:
        client.close()
    isolation = native.prepare_repository(root, env, container, task, output)
    return env, container, isolation


def normalize_p0(native, env, container, task, isolation, solver_dir, output):
    baseline = (output / "environment-baseline.patch").read_bytes()
    if baseline != (Path(solver_dir) / "environment-baseline.patch").read_bytes():
        raise ValueError("Review image baseline differs from the Solver baseline")
    base = isolation["synthetic_commit"]
    native.upload(container, baseline, "/tmp/v64-baseline.patch")
    native.upload(container, Path(solver_dir) / "model.patch", "/tmp/v64-p0.patch")
    receipt = json.loads(native.checked(env, normalization_command(
        base, "/tmp/v64-baseline.patch", "/tmp/v64-p0.patch", "/tmp/v64-normalized.patch",
    )).splitlines()[-1])
    bridge = IsolatedEnvironment(container, instance_id=task["task_id"])
    converted = output / "normalized-p0.patch"
    asyncio.run(bridge.download_file("/tmp/v64-normalized.patch", converted))
    canonical = output / "normalized-environment.patch"
    asyncio.run(bridge.download_file("/tmp/v64-normalized.patch.baseline", canonical))
    return bridge, base, converted.read_bytes(), canonical.read_bytes(), receipt


def make_gate(native, task, image, log_dir):
    executor = DockerGateExecutor(
        lambda: native.docker.from_env(timeout=1800), image_id=image.id, cpus=task["cpus"],
        memory_mb=task["memory_mb"], task_id=task["task_id"], log_dir=log_dir,
    )
    return BuildGate(executor, run_timeout_sec=V64_BUILD_GATE_TIMEOUT_SEC)


def analysis_plan(p0: bytes, final: bytes) -> str:
    """P0 needs its own verifier run only when the delivered patch differs."""
    return "reuse_final_score" if sha256_bytes(p0) == sha256_bytes(final) else "score_p0_separately"


# ---------------------------------------------------------------------------
# one task end to end
# ---------------------------------------------------------------------------

def run_case(native, root, row, config, update):
    tid = row["task_id"]
    output = root / "outputs" / tid
    solver_dir, review_dir = output / "solver", output / "review"
    task = read(root / "inputs" / tid / "task.json")
    original = read(root / "inputs" / tid / "original-task.json")
    if hashlib.sha256(task["instruction"].encode()).hexdigest() != row["solver_instruction_sha256"]:
        raise ValueError("Solver input changed after prepare")
    ledger = load_frozen_ledger(root / row["ledger_file"], file_sha256=row["ledger_file_sha256"],
                                ledger_sha256=row["ledger_sha256"])
    client = native.docker.from_env(timeout=1800)
    added, env, container, image = False, None, None, None
    stage = "image_preparation"
    try:
        update(tid, stage)
        image, added = native.get_image(client, task, output)
        if image.id != row["image"]["id"]:
            raise ValueError("Task image differs from the frozen M0/ft_m1 image")
        stage = "solving"
        if (solver_dir / "prediction.json").exists():
            # Never resample an existing Solver output.
            prediction = read(solver_dir / "prediction.json")
            p0 = (solver_dir / "model.patch").read_text()
            info = read(solver_dir / "agent-info.json")
            if prediction["model_patch"] != p0:
                raise ValueError("Existing Solver output is inconsistent")
        else:
            update(tid, stage)
            p0, info = native.generate(task, root, solver_dir, config, image, "generate")
        update(tid, "reviewing", solver_exit_status=info.get("exit_status"))
        stage = "review_setup"
        env, container, isolation = open_review_container(native, root, task, image, config, review_dir)
        bridge, base, normalized, canonical, receipt = normalize_p0(
            native, env, container, task, isolation, solver_dir, review_dir)
        provenance = {"m1_output": str(solver_dir), "original_p0_sha256": sha256_bytes(p0.encode()),
                      "normalized_p0_sha256": sha256_bytes(normalized),
                      "v6_baseline_sha256": sha256_bytes(canonical),
                      "ledger_sha256": row["ledger_sha256"], **receipt}
        save(review_dir / "p0-normalization.json", provenance)
        fork = write_frozen_fork(review_dir, task_id=tid, base_commit=base, baseline_patch=canonical,
                                 ledger=ledger, candidate_patch=normalized, provenance=provenance)
        stage = "review"
        agent = MiniV64ReviewAgent(
            logs_dir=review_dir / "agent/logs", model_name="deepseek-flash",
            stage_model_config=STAGE_MODEL_CONFIG,
            extra_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]}, fork_plan=str(fork),
            build_gate=make_gate(native, task, image, review_dir / "build-gate-logs"),
        )
        context = SimpleNamespace(metadata={})
        asyncio.run(agent.run(original["instruction"], bridge, context))
        workflow = review_dir / "agent/v6_review_repair"
        manifest = read(workflow / "manifest.json")
        if manifest["workflow_state"] != "completed":
            raise RuntimeError("Review did not complete; no fallback result is scored")
        gate = read(workflow / "v5/build-gate.json")
        final = native.checked(env, native.snapshot_command(base))
        if ((manifest["route"] in {"accept", "hold"} or gate["delivered"] == "P0")
                and sha256_bytes(final.encode()) != sha256_bytes(p0.encode())):
            raise RuntimeError("A path that delivers P0 changed the Solver patch")
        (output / "model.patch").write_text(final)
        container.remove(force=True)
        container, env.container_id = None, None
        # Official tests enter fresh containers only after every model stage ended.
        stage = "official_evaluation"
        update(tid, stage, route=manifest["route"])
        result = native.grade(root, task, image.id, final, output / "evaluation")
        stage = "p0_analysis_evaluation"
        plan = analysis_plan(p0.encode(), final.encode())
        if plan == "reuse_final_score":
            p0_result = dict(result, reused_final_score=True)
        else:
            update(tid, stage)
            p0_result = native.grade(root, task, image.id, p0, output / "p0-evaluation")
        result.update(
            experiment_id=read(root / "selection.json")["experiment_id"], route=manifest["route"],
            repair_executed=manifest["repair_executed"], build_gate=gate,
            solver_exit_status=info.get("exit_status"),
            p0_sha256=sha256_bytes(p0.encode()), final_sha256=sha256_bytes(final.encode()),
            p0_analysis={"status": p0_result.get("status"), "reward": p0_result.get("reward"),
                         "official": p0_result.get("official"), "plan": plan},
            finished_at=native.now(),
        )
        return result
    except Exception as error:  # noqa: BLE001 - archive per-task stage/infrastructure failures
        return {"task_id": tid, "status": "infrastructure_or_stage_error", "stage": stage,
                "reward": None, "error_type": type(error).__name__, "error": str(error),
                "finished_at": native.now()}
    finally:
        if container is not None:
            container.remove(force=True)
        if env is not None:
            env.container_id = None
        if added and image is not None:
            try:
                client.images.remove(image.id, noprune=False)
            except native.docker.errors.APIError as error:
                save(output / "image-cleanup.json", {"error": str(error)})
        client.close()


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
        update(tid, result["status"], reward=result.get("reward"))
        with lock:
            save(root / "result.json", {
                "task_count": len(states), "terminal_count": len(results),
                "passed": sum(r.get("reward") == 1 for r in results.values()),
                "valid_failed": sum(r.get("reward") == 0 for r in results.values()),
                "results": list(results.values()), "updated_at": native.now(),
            })
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in selection["tasks"]]):
            future.result()
    save(root / "launcher-result.json", {"completed": True, "terminal_count": len(results),
                                         "finished_at": native.now()})


# ---------------------------------------------------------------------------
# zero-API preflights
# ---------------------------------------------------------------------------

def preflight(root, selection, native, config):
    official = read(selection["official_preflight"])
    if not official["passed"]:
        raise ValueError("The original official gold/control preflight is not valid")
    row = selection["tasks"][0]
    task = read(root / "inputs" / row["task_id"] / "task.json")
    original = read(root / "inputs" / row["task_id"] / "original-task.json")
    output = root / "preflight" / row["task_id"]
    client = native.docker.from_env(timeout=1800)
    try:
        image, _ = native.get_image(client, task, output)
        if image.id != row["image"]["id"]:
            raise ValueError("Preflight image differs from the frozen image")
        _, protocol = native.generate(task, root, output / "offline", config, image, "offline")
    finally:
        client.close()
    trajectory = read(output / "offline/native.traj.json")
    user = next(m["content"] for m in trajectory["messages"] if m["role"] == "user")
    if (original["instruction"] not in user or user.count(HEADING) != 1
            or user.count(GUIDANCE) != 1):
        raise ValueError("Native Mini did not receive exactly one V6.4 ledger supplement")
    save(root / "preflight/readiness.json", {
        "passed": True, "experiment_id": selection["experiment_id"], "model_api_calls": 0,
        "selection_sha256": sha256_json(selection), "offline_protocol": protocol,
        "actual_native_user_message_verified": True,
        "official_gold_control_reused_from": selection["official_preflight"],
        "finished_at": native.now(),
    })


def preflight_gate(root, selection, native, config, task_ids=None):
    """G(base) and G(ft_m1 P0) on real images; no model call, no official tests."""
    summary = {}
    for row in selection["tasks"]:
        tid = row["task_id"]
        if task_ids and tid not in task_ids:
            continue
        task = read(root / "inputs" / tid / "task.json")
        output = root / "preflight/gate" / tid
        client = native.docker.from_env(timeout=1800)
        env = container = None
        added = False
        try:
            image, added = native.get_image(client, task, output)
            env, container, isolation = open_review_container(native, root, task, image, config, output)
            _, _, normalized, _, _ = normalize_p0(
                native, env, container, task, isolation, row["ft_m1_output"], output)
            container.remove(force=True)
            container, env.container_id = None, None
            gate = make_gate(native, task, image, output / "logs")
            report = asyncio.run(gate.check("P0", normalized))
            save(output / "report.json", report)
            summary[tid] = {"verdict": report["verdict"], "elapsed_sec": report.get("elapsed_sec"),
                            "commands": [c["command"] for c in report["commands"]]}
        except Exception as error:  # noqa: BLE001 - record and continue the dry run
            summary[tid] = {"verdict": "preflight_error", "error": f"{type(error).__name__}: {error}"}
        finally:
            if container is not None:
                container.remove(force=True)
            if env is not None:
                env.container_id = None
            if added:
                try:
                    client.images.remove(task["image"], noprune=False)
                except native.docker.errors.APIError:
                    pass
            client.close()
        save(root / "preflight/gate-summary.json", {"model_api_calls": 0, "tasks": summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "preflight-gate", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--ledger-source", action="append", type=Path, default=[])
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--smoke-task", help="prepare a separate one-task end-to-end validation")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.workers != 3:
        raise ValueError("V6.4 uses the approved three pipeline slots")
    if args.mode == "prepare":
        if len(args.ledger_source) != 2:
            parser.error("prepare requires both ft_m1 batches as --ledger-source")
        selection = prepare(root, args.ledger_source, args.smoke_task)
    else:
        native = load_native(root)
        selection = validate(root, native)
        config = native.prepare_config(root)
        if args.mode == "preflight":
            preflight(root, selection, native, config)
        elif args.mode == "preflight-gate":
            preflight_gate(root, selection, native, config, set(args.task_id or []))
        else:
            if args.task_id:
                raise ValueError("The formal v6.4 run covers the full eval30")
            run_batch(root, selection, native, config, args.workers)
    print(json.dumps({"mode": args.mode, "task_count": len(selection["tasks"]),
                      "experiment_id": selection["experiment_id"]}))


if __name__ == "__main__":
    main()
