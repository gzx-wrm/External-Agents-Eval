"""V6.6 A-only experiment: Pi A with working grep/find and the evidence plan ->
frozen observed evidence -> unchanged FT ledger request -> check-point stage C.

No Solver, review or scoring runs. prepare and preflight make no model calls; run
is explicit. The A sandbox, repository isolation and filtered egress are the
V6.3 held-out runner's, unchanged. Outputs go to a new root.
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

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.agents.config import get_agent_spec  # noqa: E402
from external_agents_eval.agents.pier.pi_model_config import PiStageModelConfig  # noqa: E402
from external_agents_eval.agents.pier.pi_role_runner import (  # noqa: E402
    ContainerGitWorkspaceManager,
    PiRoleRunner,
)
from external_agents_eval.agents.pier.pi_solve_review_repair import (  # noqa: E402
    issue_title_from_instruction,
)
from external_agents_eval.agents.pier.pi_v5_solve_review_repair import (  # noqa: E402
    managed_delivery_from_instruction,
)
from external_agents_eval.workflows.solve_review_repair.artifacts import (  # noqa: E402
    ArtifactStore,
    sha256_bytes,
    sha256_json,
)
from external_agents_eval.workflows.solve_review_repair.models import TaskInput  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.v66_checkpoints import (  # noqa: E402
    CheckpointClient,
)
from external_agents_eval.workflows.solve_review_repair.v66_config import V66Config  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.v66_controller import (  # noqa: E402
    INFRASTRUCTURE_MARKER,
    V66Controller,
)
from external_agents_eval.workflows.solve_review_repair.v66_transport import (  # noqa: E402
    V66FTLedgerClient,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v63 = load_module("v66a_v63_runner", PROJECT / "scripts/deepswe_v63_heldout/run.py")
read, save, load_native = v63.read, v63.save, v63.load_native

EXPERIMENT_ID = "v6.6-a-only"
STAGE_MODEL_CONFIG = v63.STAGE_MODEL_CONFIG
FROZEN_SOURCES = tuple(dict.fromkeys(v63.FROZEN_SOURCES + (
    "scripts/deepswe_v66_a/run.py",
    "external_agents_eval/workflows/solve_review_repair/v66_config.py",
    "external_agents_eval/workflows/solve_review_repair/v66_prompts.py",
    "external_agents_eval/workflows/solve_review_repair/v66_checkpoints.py",
    "external_agents_eval/workflows/solve_review_repair/v66_controller.py",
    "external_agents_eval/workflows/solve_review_repair/v66_transport.py",
)))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {name: file_sha256(PROJECT / name) for name in FROZEN_SOURCES}


def runtime_manifest(archive):
    archive = Path(archive).resolve()
    manifest = json.loads((archive.parent / "runtime.json").read_text())
    if (manifest["archive"] != str(archive) or file_sha256(archive) != manifest["archive_sha256"]
            or manifest["pi_version"] != "0.84.2" or manifest["node_version"] != "v22.19.0"
            or manifest.get("rg_fd_included") is not True):
        raise ValueError("V6.6 needs the rg/fd Pi runtime that matches its manifest")
    return {**manifest, "manifest_sha256": file_sha256(archive.parent / "runtime.json")}


def prepare(root, m0_root, archive, task_ids, baselines, note):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    runtime = runtime_manifest(archive)
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir()
    for target, source in {"m0_run.py": "run.py", "native_compat.py": "native_compat.py",
                           "sanitize_repository.py": "sanitize_repository.py"}.items():
        shutil.copyfile(m0_root / "scripts" / source, root / "scripts" / target)
    (root / "manifests").mkdir()
    for name in ("public-tasks.json", "eval30_current.json", "task-file-sha256.json"):
        shutil.copyfile(m0_root / "manifests" / name, root / "manifests" / name)
    (root / "grader").symlink_to(m0_root / "grader", target_is_directory=True)
    native = load_native(root)
    tasks = {task["task_id"]: task for task in native.load_tasks(root)}
    rows = []
    for tid in task_ids:
        image = read(m0_root / "outputs" / tid / "image.json")
        baseline = next((base / "inputs/ledgers" / f"{tid}.json" for base in baselines
                         if (base / "inputs/ledgers" / f"{tid}.json").exists()), None)
        rows.append({
            "task_id": tid, "task": tasks[tid],
            "image": {"id": image["id"], "repo_digests": image["repo_digests"]},
            "v63_ledger": str(baseline) if baseline else None,
            "v63_ledger_sha256": file_sha256(baseline) if baseline else None,
        })
    selection = {
        "experiment_id": EXPERIMENT_ID, "note": note, "task_count": len(rows), "tasks": rows,
        "m0_root": str(m0_root), "code_sha256": code_hashes(), "pi_runtime": runtime,
        "config": {"checkpoint_max_per_target": V66Config().checkpoint_max_per_target,
                   "checkpoint_max_attempts": V66Config().checkpoint_max_attempts,
                   "solver_view_max_points": V66Config().solver_view_max_points,
                   "checkpoint_batch_max_targets": V66Config().checkpoint_batch_max_targets,
                   "evidence_plan_continuations": V66Config().evidence_plan_continuations,
                   "prompt_version": V66Config().workflow_prompt_version,
                   "a_role_timeout_sec": v63.A_ROLE_TIMEOUT_SEC},
        "stages": ["A: Pi exploration (historical prompt + evidence plan; read/grep/find/ls with rg/fd)",
                   "FT: unchanged single tool-free ledger request",
                   "C: tool-free check-point request with up to one corrective retry",
                   "FT and C transport: resend only when no valid response arrived (429/5xx, "
                   "connection/timeout, empty/null/non-JSON/no-choices body); waits 30/60/120 s"],
        "solver_review_scoring": False,
    }
    save(root / "selection.json", selection)
    return selection


def validate(root, native):
    selection = read(root / "selection.json")
    if selection["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Wrong experiment")
    if code_hashes() != selection["code_sha256"]:
        changed = [k for k, v in code_hashes().items() if selection["code_sha256"].get(k) != v]
        raise ValueError("Frozen experiment code changed: " + ", ".join(changed))
    runtime_manifest(selection["pi_runtime"]["archive"])
    native.load_tasks(root)
    return selection


async def build_a(sandbox, task, output, api_key):
    """V6.3 held-out build_ledger with V66Controller, then stage C."""
    tid, instruction = task["task_id"], task["instruction"]
    store_root = output / "v6_review_repair"
    model_config = PiStageModelConfig.from_value(STAGE_MODEL_CONFIG)
    store = ArtifactStore(store_root, secrets=(api_key, sandbox.token))

    def redact(text):
        value = "" if text is None else str(text)
        return sandbox.redact(value.replace(api_key, "[REDACTED]"))

    environment = v63.PiContainerEnvironment(sandbox.container, instance_id=tid,
                                             proxy_env=sandbox.proxy_env)
    env = environment.agent_process_env({key: api_key for key in model_config.required_env_keys()})
    env.update(v63.PI_ENV)
    runner = PiRoleRunner(
        environment=environment, logs_root=store_root, model=model_config.default.model,
        provider=model_config.default.provider, stage_model_config=model_config,
        agent_version=get_agent_spec("pi").upstream_ref, env=env, redact=redact,
    )
    workspace = ContainerGitWorkspaceManager(environment=environment,
                                             transfer_dir=store_root / ".transfer",
                                             repository="/app", redact=redact)
    head = await environment.exec("git rev-parse --verify 'HEAD^{commit}'", cwd="/app")
    if head.return_code:
        raise RuntimeError("Cannot resolve the sanitized A baseline commit")
    base_commit = head.stdout.strip()
    try:
        environment_patch = await workspace.capture_patch(worktree="/app", base_commit=base_commit)
        baseline_sha256 = sha256_bytes(environment_patch)
        store.write_bytes("inputs/environment.patch", environment_patch)
        paths = await workspace.prepare(base_commit=base_commit)
        if environment_patch:
            await workspace.apply_patch(worktree=paths.spec_review, patch=environment_patch)
        task_input = TaskInput(
            task_id=tid, issue_title=issue_title_from_instruction(instruction, tid),
            issue_body=instruction, repository_name=tid.split("-", 1)[0],
            base_commit=base_commit, managed_delivery=managed_delivery_from_instruction(instruction),
        )
        config = V66Config(role_timeout_sec=v63.A_ROLE_TIMEOUT_SEC,
                           model=model_config.for_role("solver").model)
        a_model = model_config.for_role("A")
        controller = V66Controller(
            # FT's own request code with the approved transport retry (no resampling).
            ledger_client=V66FTLedgerClient(model=a_model.model, base_url=a_model.effective_base_url,
                                            api_key=api_key),
            checkpoint_client=CheckpointClient(
                model=a_model.model, base_url=a_model.effective_base_url, api_key=api_key,
                max_per_target=config.checkpoint_max_per_target,
                max_attempts=config.checkpoint_max_attempts, timeout_sec=config.checkpoint_timeout_sec,
                max_solver_points=config.solver_view_max_points,
                batch_max_targets=config.checkpoint_batch_max_targets),
            role_runner=runner, workspace=workspace, artifacts=store, config=config,
        )
        ledger = await controller.prepare_ledger(
            task_input, base_worktree=paths.spec_review, baseline_patch=environment_patch,
            stage_id=v63.a_stage_id(tid, base_commit, instruction, baseline_sha256, model_config, config),
        )
        status = {"base_commit": base_commit, "ledger_quality": controller.ledger_quality,
                  "ledger_warnings": list(controller.ledger_warnings),
                  "ledger_error": controller.ledger_error}
        if ledger is None:
            return None, status, None
        checkpoints = await controller.prepare_checkpoints(ledger)
        return ledger, status, checkpoints
    finally:
        await workspace.cleanup()


def run_a(native, root, row, api_key, runtime):
    tid = row["task_id"]
    output = root / "a" / tid
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("An A attempt already exists without a terminal record; not resampled")
    output.mkdir(parents=True, exist_ok=True)
    started = native.now()
    client = native.docker.from_env(timeout=1800)
    sandbox = None
    try:
        image = v63.require_image(client, row)
        sandbox = v63.ASandbox(client, image, row["task"],
                               PiStageModelConfig.from_value(STAGE_MODEL_CONFIG).domains())
        isolation = v63.prepare_a_container(native, root, sandbox, row["task"], output, runtime)
        ledger, status, checkpoints = asyncio.run(build_a(sandbox, row["task"], output, api_key))
    finally:
        if sandbox is not None:
            sandbox.close(output)
        client.close()
    workflow = output / "v6_review_repair"
    packet = read(workflow / "v6.2/evidence.json") if (workflow / "v6.2/evidence.json").exists() else {}
    record = {
        "task_id": tid, "started_at": started, "finished_at": native.now(), **status,
        "rg_fd_on_path": isolation["rg_fd_on_path"],
        "tool_counts": packet.get("tool_counts"),
        "evidence_files": len({item["path"] for item in packet.get("evidence", [])}),
        "evidence_spans": len(packet.get("evidence", [])),
        "skipped_reads": len(packet.get("skipped_reads", [])),
        "evidence_plan_check": read(workflow / "v6.6/evidence-plan-check.json")
        if (workflow / "v6.6/evidence-plan-check.json").exists() else None,
    }
    if ledger is not None:
        path = output / "ledger.json"
        path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
        record.update(ledger_sha256=sha256_json(ledger), counts={
            name: len(ledger[name]) for name in ("obligations", "preservation_obligations", "open_questions")})
    if checkpoints is not None:
        record["checkpoints"] = {key: checkpoints.get(key) for key in
                                 ("status", "attempts", "summary", "elapsed_seconds", "error")}
    record["status"] = ("checkpoints_ready" if checkpoints and checkpoints["status"] == "passed"
                        else "ledger_only" if ledger is not None
                        # Pi never finished normally: infrastructure, rerun in a new root.
                        else "infrastructure_error" if INFRASTRUCTURE_MARKER in (status["ledger_error"] or "")
                        else "ledger_unavailable")
    save(output / "a-result.json", record)
    return record


def run_batch(root, selection, native, workers):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    readiness = read(root / "preflight/readiness.json")
    if not readiness["passed"] or readiness["selection_sha256"] != sha256_json(selection):
        raise ValueError("Run the zero-API preflight for this frozen selection first")
    lock, results = threading.Lock(), {}

    def one(row):
        path = root / "a" / row["task_id"] / "a-result.json"
        try:
            result = read(path) if path.exists() else run_a(native, root, row, api_key,
                                                             selection["pi_runtime"])
        except Exception as error:  # noqa: BLE001 - keep per-task stage failures explicit
            result = {"task_id": row["task_id"], "status": "stage_error",
                      "error_type": type(error).__name__, "error": str(error)}
            save(root / "errors" / f"{row['task_id']}.json", result)
        with lock:
            results[row["task_id"]] = result
            save(root / "result.json", {"experiment_id": EXPERIMENT_ID, "task_count": len(selection["tasks"]),
                                        "terminal_count": len(results), "results": list(results.values()),
                                        "updated_at": native.now()})
        print(json.dumps({"task_id": row["task_id"], "status": result["status"]}), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, row) for row in selection["tasks"]]):
            future.result()


def preflight(root, selection, native):
    client = native.docker.from_env(timeout=1800)
    try:
        images = [v63.require_image(client, row).id for row in selection["tasks"]]
    finally:
        client.close()
    row = selection["tasks"][0]
    output = root / "preflight" / row["task_id"]
    output.mkdir(parents=True, exist_ok=True)
    client = native.docker.from_env(timeout=1800)
    sandbox = None
    try:
        image = v63.require_image(client, row)
        allowlist = PiStageModelConfig.from_value(STAGE_MODEL_CONFIG).domains()
        sandbox = v63.ASandbox(client, image, row["task"], allowlist)
        isolation = v63.prepare_a_container(native, root, sandbox, row["task"], output,
                                            selection["pi_runtime"])
        search = v63.container_checked(
            sandbox.container, "PATH=" + v63.PI_ENV["PATH"] + " rg --files /app | head -3 && "
            "PATH=" + v63.PI_ENV["PATH"] + " fd --type f . /app | head -3").splitlines()
        pi_probe = asyncio.run(v63.probe_pi(sandbox, row["task"], output))
    finally:
        if sandbox is not None:
            sandbox.close(output)
        client.close()
    proxy_log = (output / "egress-proxy.log").read_text()
    reached = (proxy_log.count(f"ALLOW CONNECT {allowlist[0]}:443") >= 2
               and pi_probe["authentication_rejected"])
    tools = sorted(Path(path).name for path in isolation["rg_fd_on_path"])
    passed = bool(tools == ["fd", "rg"] and len(search) >= 2 and reached)
    save(root / "preflight/readiness.json", {
        "passed": passed, "experiment_id": EXPERIMENT_ID, "model_api_calls": 0,
        "selection_sha256": sha256_json(selection), "images_verified": len(images),
        "rg_fd_on_path": isolation["rg_fd_on_path"], "rg_fd_search_sample": search,
        "egress_probe": isolation["egress_probe"], "pi_probe": pi_probe,
        "pi_reached_model_api_through_proxy": reached, "finished_at": native.now(),
    })
    if not passed:
        raise RuntimeError("Preflight failed; see preflight/readiness.json")


def report(root, selection):
    """Human-readable comparison of the frozen V6.3 ledger and the V6.6 A outputs."""
    lines = ["# V6.6 A-only outputs", ""]
    for row in selection["tasks"]:
        tid = row["task_id"]
        output = root / "a" / tid
        record = read(output / "a-result.json") if (output / "a-result.json").exists() else {}
        lines += [f"## {tid}", "", f"- status: {record.get('status')}",
                  f"- tool counts: {record.get('tool_counts')}",
                  f"- evidence files/spans: {record.get('evidence_files')}/{record.get('evidence_spans')}"]
        if row.get("v63_ledger"):
            old = read(row["v63_ledger"])
            lines.append("- v6.3 ledger O/P/Q: " + "/".join(
                str(len(old[k])) for k in ("obligations", "preservation_obligations", "open_questions")))
        if record.get("counts"):
            lines.append("- v6.6 ledger O/P/Q: " + "/".join(
                str(record["counts"][k]) for k in ("obligations", "preservation_obligations", "open_questions")))
        path = output / "v6_review_repair/v6.6/checkpoints/checkpoints.json"
        if path.exists():
            ledger = read(output / "ledger.json")
            text = {item["id"]: (item["content"].get("statement") or item["content"].get("question"))
                    for field in ("obligations", "preservation_obligations", "open_questions")
                    for item in ledger[field]}
            lines += [f"- check points: {record['checkpoints']['summary']}",
                      f"- evidence plan check: {record.get('evidence_plan_check')}", "",
                      "  solver view:"]
            view = output / "v6_review_repair/v6.6/checkpoints/solver-view.md"
            lines += ["  " + line for line in view.read_text().splitlines()] + ["", "  all check points:"]
            for point in read(path)["checkpoints"]:
                extra = (f" | alternatives: {point['alternatives']} | favored: {point['favored']}"
                         if point["resolution"] == "open" else "")
                lines.append(f"  - {','.join(point['targets'])}/{point['facet']} [{point['resolution']}] given "
                             f"{point['given']}; when {point['when']}; then {point['then']}{extra}")
            lines += ["", "  ledger:"] + [f"  - {key}: {value}" for key, value in text.items()]
        lines.append("")
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "run", "report"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--m0-root", type=Path)
    parser.add_argument("--pi-runtime", type=Path)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--v63-root", action="append", type=Path, default=[],
                        help="roots whose inputs/ledgers hold the V6.3 baseline ledgers")
    parser.add_argument("--note")
    parser.add_argument("--workers", type=int, default=3, choices=[1, 2, 3])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if not (args.m0_root and args.pi_runtime and args.task_id):
            parser.error("prepare requires --m0-root, --pi-runtime and --task-id")
        selection = prepare(root, args.m0_root.resolve(), args.pi_runtime.resolve(), args.task_id,
                            [path.resolve() for path in args.v63_root], args.note)
    else:
        native = load_native(root)
        selection = validate(root, native)
        if args.mode == "preflight":
            preflight(root, selection, native)
        elif args.mode == "run":
            run_batch(root, selection, native, args.workers)
        else:
            report(root, selection)
    print(json.dumps({"mode": args.mode, "experiment_id": EXPERIMENT_ID,
                      "task_count": selection["task_count"]}))


if __name__ == "__main__":
    main()
