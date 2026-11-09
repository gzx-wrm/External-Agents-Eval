"""V6.5 on DeepSWE: frozen FT ledger -> Solver -> G(P0) -> review view -> I/R + X->D -> S
-> route -> Repair -> G(P1) -> official verifier.

Identical to the V6.4 runner except for the review agent (MiniV65ReviewAgent), the
removal of raw-P0 copies from the review container, a pilot mode over a task subset
that reuses Solver outputs of an earlier run with an identical Solver input, the
zero-API ``preflight-view`` and ``summarize``. prepare/preflight make no model calls.
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
V64_SCRIPTS = PROJECT / "scripts/deepswe_v64"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(V64_SCRIPTS))

from external_agents_eval.agents.mini_ft_review import normalization_command, write_frozen_fork  # noqa: E402
from external_agents_eval.agents.mini_v65 import MiniV65ReviewAgent, MiniV65WorkspaceManager  # noqa: E402
from external_agents_eval.benchmarks.swebench.isolated_environment import IsolatedEnvironment  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.artifacts import (  # noqa: E402
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.build_gate import (  # noqa: E402
    FAIL as GATE_FAIL,
)
from external_agents_eval.workflows.solve_review_repair.build_gate import (  # noqa: E402
    PASS as GATE_PASS,
)
from external_agents_eval.workflows.solve_review_repair.build_gate import BuildGate  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.v64_config import (  # noqa: E402
    V64_BUILD_GATE_TIMEOUT_SEC,
)
from external_agents_eval.workflows.solve_review_repair.v65_config import V65Config  # noqa: E402
from gate_executor import DockerGateExecutor  # noqa: E402
from ledger_input_v64 import GUIDANCE, HEADING, augment_task, load_frozen_ledger  # noqa: E402

EXPERIMENT_ID = "v6.5"
PILOT_EXPERIMENT_ID = "v6.5-pilot"
SCRIPTS = ("m0_run.py", "native_compat.py", "sanitize_repository.py")
MANIFESTS = ("public-tasks.json", "eval30_current.json", "task-file-sha256.json")
STAGE_MODEL_CONFIG = PROJECT / "configs/v6.1-deepseek-v4.1-flash.json"
SOLVER_FILES = ("prediction.json", "model.patch", "agent-info.json", "environment-baseline.patch",
                "isolation.json", "native.traj.json")
FROZEN_SOURCES = [
    "scripts/deepswe_v65/run.py",
    "scripts/deepswe_v64/gate_executor.py",
    "scripts/deepswe_v64/ledger_input_v64.py",
    "external_agents_eval/agents/mini_v65.py",
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
    "external_agents_eval/workflows/solve_review_repair/v5_state.py",
    "external_agents_eval/workflows/solve_review_repair/v6_controller.py",
    "external_agents_eval/workflows/solve_review_repair/v64_config.py",
    "external_agents_eval/workflows/solve_review_repair/build_gate.py",
    "external_agents_eval/workflows/solve_review_repair/review_view.py",
    "external_agents_eval/workflows/solve_review_repair/v65_config.py",
    "external_agents_eval/workflows/solve_review_repair/v65_prompts.py",
    "external_agents_eval/workflows/solve_review_repair/v65_contracts.py",
    "external_agents_eval/workflows/solve_review_repair/v65_state.py",
    "external_agents_eval/workflows/solve_review_repair/v65_controller.py",
    "configs/v6.1-deepseek-v4.1-flash.json",
]
SKILL_DIR = "external_agents_eval/workflows/solve_review_repair/skills/v6.5-review-note-files"
RAW_P0_TMP = ("/tmp/v65-baseline.patch", "/tmp/v65-p0.patch", "/tmp/v65-normalized.patch",
              "/tmp/v65-normalized.patch.baseline")


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
    spec = importlib.util.spec_from_file_location(f"v65_frozen_m0_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(root, sources, *, task_ids=None, solver_source=None):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty v6.5 experiment directory")
    if bool(task_ids) != bool(solver_source):
        raise ValueError("A pilot needs both --task-id and --solver-source")
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
            rows[tid] = {
                "task_id": tid, "task_checksum": task["task_checksum"],
                "base_commit": task["base_commit"], "image": record["m0_image"],
                "ledger_source": str(ledger_path), "ledger_file_sha256": record["ledger_file_sha256"],
                "ledger_sha256": record["ledger_sha256"], "ledger_quality": record.get("ledger_quality"),
            }
    order = read(m0_root / "manifests/eval30_current.json")["task_ids"]
    if len(order) != 30 or set(order) != set(rows):
        raise ValueError("The ft_m1 ledgers must cover exactly the fixed eval30")
    reuse_rows = {}
    if task_ids:
        unknown = sorted(set(task_ids) - set(order))
        if unknown:
            raise ValueError("Pilot tasks outside the eval30: " + ", ".join(unknown))
        order = [tid for tid in order if tid in set(task_ids)]
        solver_source = Path(solver_source).resolve()
        reuse_rows = {row["task_id"]: row for row in read(solver_source / "selection.json")["tasks"]}
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
    ordered, reused = [], {}
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
        if task_ids:
            # Reuse only an output produced from this exact Solver input.
            earlier = reuse_rows[tid]
            for key in ("solver_instruction_sha256", "ledger_sha256", "task_checksum"):
                if earlier[key] != row[key]:
                    raise ValueError(f"Solver input differs from the reused run ({key}): {tid}")
            source_dir = solver_source / "outputs" / tid / "solver"
            target_dir = root / "outputs" / tid / "solver"
            target_dir.mkdir(parents=True)
            for name in SOLVER_FILES:
                shutil.copyfile(source_dir / name, target_dir / name)
            reused[tid] = {"source": str(source_dir),
                           "files_sha256": {name: file_sha256(target_dir / name) for name in SOLVER_FILES}}
        ordered.append(row)
    if task_ids:
        save(root / "solver-reuse.json", {
            "policy": "Pilot reviews frozen Solver outputs of an earlier run whose Solver input "
                      "is byte-identical; no Solver is resampled.",
            "source_run": str(solver_source), "reused": reused,
        })
    selection = {
        "experiment_id": PILOT_EXPERIMENT_ID if task_ids else EXPERIMENT_ID,
        "task_count": len(ordered), "tasks": ordered,
        "pipeline": "frozen FT ledger -> native Mini Solver -> G(P0) -> review view -> "
                    "I/R + X->D -> S -> route -> Repair -> G(P1) -> unchanged official verifier",
        "m0_root": str(m0_root), "official_preflight": str(m0_root / "preflight/readiness.json"),
        "native_source_sha256": native_hashes, "code_sha256": code_hashes(),
        "solver_guidance_sha256": sha256_bytes((HEADING + "\n\n" + GUIDANCE).encode()),
        "review_config": json.loads(json.dumps(V65Config().__dict__, default=list)),
        "a_rerun": False, "ledger_modified": False, "evidence_packet_included": False,
        "p0_scored_for_analysis_only": True, "workers": 3,
        "build_gate_timeout_sec": V64_BUILD_GATE_TIMEOUT_SEC,
        "solver_reuse": str(root / "solver-reuse.json") if task_ids else None,
    }
    save(root / "selection.json", selection)
    return selection


def validate(root, native):
    selection = read(root / "selection.json")
    if selection["experiment_id"] not in {EXPERIMENT_ID, PILOT_EXPERIMENT_ID}:
        raise ValueError("Wrong experiment directory")
    if selection["experiment_id"] == EXPERIMENT_ID and len(selection["tasks"]) != 30:
        raise ValueError("The formal v6.5 run covers the full eval30")
    for name, digest in selection["native_source_sha256"].items():
        if file_sha256(root / "scripts" / name) != digest:
            raise ValueError("Frozen native runtime changed: " + name)
    if code_hashes() != selection["code_sha256"]:
        changed = [k for k, v in code_hashes().items() if selection["code_sha256"].get(k) != v]
        raise ValueError("V6.5 source changed after prepare: " + ", ".join(changed))
    native.load_tasks(root)
    for row in selection["tasks"]:
        load_frozen_ledger(root / row["ledger_file"], file_sha256=row["ledger_file_sha256"],
                           ledger_sha256=row["ledger_sha256"])
    if selection.get("solver_reuse"):
        for tid, entry in read(selection["solver_reuse"])["reused"].items():
            for name, digest in entry["files_sha256"].items():
                if file_sha256(root / "outputs" / tid / "solver" / name) != digest:
                    raise ValueError(f"Reused Solver output changed: {tid}/{name}")
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
    native.upload(container, baseline, RAW_P0_TMP[0])
    native.upload(container, Path(solver_dir) / "model.patch", RAW_P0_TMP[1])
    receipt = json.loads(native.checked(env, normalization_command(
        base, RAW_P0_TMP[0], RAW_P0_TMP[1], RAW_P0_TMP[2],
    )).splitlines()[-1])
    bridge = IsolatedEnvironment(container, instance_id=task["task_id"])
    converted = output / "normalized-p0.patch"
    asyncio.run(bridge.download_file(RAW_P0_TMP[2], converted))
    canonical = output / "normalized-environment.patch"
    asyncio.run(bridge.download_file(RAW_P0_TMP[3], canonical))
    # V6.5: reviewers must not find the raw P0 in the container.
    native.checked(env, "rm -f -- " + " ".join(RAW_P0_TMP))
    return bridge, base, converted.read_bytes(), canonical.read_bytes(), receipt


def make_gate(native, task, image, log_dir):
    executor = DockerGateExecutor(
        lambda: native.docker.from_env(timeout=1800), image_id=image.id, cpus=task["cpus"],
        memory_mb=task["memory_mb"], task_id=task["task_id"], log_dir=log_dir,
    )
    return BuildGate(executor, run_timeout_sec=V64_BUILD_GATE_TIMEOUT_SEC)


def analysis_plan(p0: bytes, final: bytes) -> str:
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
        agent = MiniV65ReviewAgent(
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
        raise ValueError("Run the zero-API preflight-view for this frozen selection first")
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
# zero-API preflight: build every review view in a real review container
# ---------------------------------------------------------------------------

async def _view_for_task(bridge, base, normalized, gate, output):
    workspace = MiniV65WorkspaceManager(environment=bridge, transfer_dir=output / ".transfer",
                                        repository="/app", redact=lambda text: text)
    environment_patch = await workspace.capture_patch(worktree="/app", base_commit=base)
    await workspace.prepare(base_commit=base)
    p0_gate = (await gate.check("P0", normalized))["verdict"]
    attempts, chosen = [], None
    try:
        for level in V65Config().review_view_levels:
            view, manifest = await workspace.build_review_view(
                base_commit=base, baseline_patch=environment_patch, candidate_patch=normalized,
                level=level)
            (output / f"view-{level}.patch").write_bytes(view)
            attempt = {"level": level, "view_sha256": sha256_bytes(view),
                       "stripped_item_count": manifest.get("stripped_item_count"),
                       "removed_test_files": manifest.get("removed_test_files"),
                       "reverted_test_files": manifest.get("reverted_test_files"),
                       "skipped": manifest.get("skipped")}
            if p0_gate == GATE_PASS and view != normalized:
                attempt["gate"] = (await gate.check(f"VIEW-{level}", view))["verdict"]
                if attempt["gate"] == GATE_FAIL:
                    attempts.append(attempt)
                    continue
            attempts.append(attempt)
            chosen = level
            break
    finally:
        await workspace.discard_review_view(candidate_patch=normalized)
    return {"p0_gate": p0_gate, "chosen_level": chosen, "attempts": attempts}


def preflight_view(root, selection, native, config):
    official = read(selection["official_preflight"])
    if not official["passed"]:
        raise ValueError("The original official gold/control preflight is not valid")
    summary = {}
    for row in selection["tasks"]:
        tid = row["task_id"]
        task = read(root / "inputs" / tid / "task.json")
        output = root / "preflight/view" / tid
        client = native.docker.from_env(timeout=1800)
        env = container = None
        added = False
        try:
            image, added = native.get_image(client, task, output)
            if image.id != row["image"]["id"]:
                raise ValueError("Preflight image differs from the frozen image")
            env, container, isolation = open_review_container(native, root, task, image, config, output)
            bridge, base, normalized, _, _ = normalize_p0(
                native, env, container, task, isolation, root / "outputs" / tid / "solver", output)
            gate = make_gate(native, task, image, output / "gate-logs")
            report = asyncio.run(_view_for_task(bridge, base, normalized, gate, output))
            save(output / "report.json", report)
            summary[tid] = report
        except Exception as error:  # noqa: BLE001 - record and continue the dry run
            summary[tid] = {"error": f"{type(error).__name__}: {error}"}
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
        save(root / "preflight/view-summary.json", {"model_api_calls": 0, "tasks": summary})
    passed = all("error" not in item for item in summary.values())
    save(root / "preflight/readiness.json", {
        "passed": passed, "experiment_id": selection["experiment_id"], "model_api_calls": 0,
        "selection_sha256": sha256_json(selection),
        "view_summary": str(root / "preflight/view-summary.json"),
        "official_gold_control_reused_from": selection["official_preflight"],
        "finished_at": native.now(),
    })
    return summary


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarize(root, selection, compare=None):
    reference = {}
    if compare:
        reference = {item["task"]: item for item in read(Path(compare) / "final-summary.json")["rows"]}
    rows = []
    for row in selection["tasks"]:
        tid = row["task_id"]
        out = root / "outputs" / tid
        result = read(out / "result.json") if (out / "result.json").exists() else {}
        workflow = out / "review/agent/v6_review_repair"

        def load(relative, default=None):
            path = workflow / relative
            return read(path) if path.exists() else default

        decision = load("v5/decision.json", {})
        view = load("v65/review-view.json", {})
        targets = load("v65/disambiguation-targets.json", {"targets": []})
        assumptions = load("v65/assumptions.json", {})
        d_report = (load("v5/exports/review-D.json", {}) or {}).get("content", {})
        official = result.get("official") or {}
        p0 = (result.get("p0_analysis") or {}).get("official") or {}
        rows.append({
            "task": tid, "reward": result.get("reward"), "status": result.get("status"),
            "final": f'{official.get("f2p_passed")}/{official.get("f2p_total")} '
                     f'{official.get("p2p_passed")}/{official.get("p2p_total")}',
            "p0_reward": (result.get("p0_analysis") or {}).get("reward"),
            "p0": f'{p0.get("f2p_passed")}/{p0.get("f2p_total")} {p0.get("p2p_passed")}/{p0.get("p2p_total")}',
            "route": decision.get("route"), "view_level": view.get("level"),
            "assumptions": len(assumptions.get("assumptions", [])),
            "assumption_status": assumptions.get("status"),
            "d_targets": [item["id"] for item in targets.get("targets", [])],
            "d_status": {item["content"]["target_id"]: item["content"]["status"]
                         for item in d_report.get("assessments", [])},
            "derived": [item["id"] for item in d_report.get("additional_targets", [])],
            "repair_requests": len((decision.get("plan") or {}).get("repair_requests", [])),
            "barrier_clean": (load("v65/barrier-audit.json", {}) or {}).get("clean"),
            "reference": {key: reference.get(tid, {}).get(key)
                          for key in ("reward", "final", "route")} if reference else None,
        })
    summary = {"experiment_id": selection["experiment_id"], "rows": rows,
               "pass": sum(item["reward"] == 1 for item in rows),
               "reference_pass": sum((item["reference"] or {}).get("reward") == 1 for item in rows)
               if reference else None}
    save(root / "final-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight-view", "run", "summarize"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--ledger-source", action="append", type=Path, default=[])
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--solver-source", type=Path)
    parser.add_argument("--compare", type=Path, help="earlier run whose final-summary.json to compare")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.workers != 3:
        raise ValueError("V6.5 uses the approved three pipeline slots")
    if args.mode == "prepare":
        if len(args.ledger_source) != 2:
            parser.error("prepare requires both ft_m1 batches as --ledger-source")
        selection = prepare(root, args.ledger_source, task_ids=args.task_id,
                            solver_source=args.solver_source)
    else:
        native = load_native(root)
        selection = validate(root, native)
        config = native.prepare_config(root)
        if args.mode == "preflight-view":
            preflight_view(root, selection, native, config)
        elif args.mode == "summarize":
            summarize(root, selection, args.compare)
        else:
            run_batch(root, selection, native, config, args.workers)
    print(json.dumps({"mode": args.mode, "task_count": len(selection["tasks"]),
                      "experiment_id": selection["experiment_id"]}))


if __name__ == "__main__":
    main()
