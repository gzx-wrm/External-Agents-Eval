"""V6.8 generation-only experiment: Mini agents explore the baseline repository and write
obligations, bound code objects and targeted check points (no Solver, review or scoring).

Per task: a plan episode (obligations) and one batch episode per three obligations, each a
native Mini agent with M0's model, environment and templates (only the task section, step and
wall limits differ) in a fresh offline baseline container prepared exactly like M0's. The
optional tool-free control (design split) gets the agent's obligations and batches plus the
frozen V6.6 evidence. prepare and preflight make no model calls; run is explicit; outputs go to
a new root; a finished episode is never resampled.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from external_agents_eval.workflows.solve_review_repair import v68_generation as v68  # noqa: E402
from external_agents_eval.workflows.solve_review_repair import v68_lint  # noqa: E402
from external_agents_eval.workflows.solve_review_repair.ft_protocol import render_user, strict_json  # noqa: E402

EXPERIMENT_ID = "v6.8-generation"
RUNNER_VERSION = "v6.8.1"
M0_ROOTS = ("/root/mini-swe-deepswe-batch3-30-20261105-w3-r1", "/root/mini-swe-deepswe-ft30-20261028",
            "/root/mini-swe-deepswe-heldout30-20261102")
V67_ROOTS = ("/root/v67-gen-20261106-dev", "/root/v67-gen-20261106-holdout")
NATIVE_COPIES = {"m0_run.py": "run.py", "native_compat.py": "native_compat.py",
                 "sanitize_repository.py": "sanitize_repository.py"}
MARKER = "## Command Execution Rules"
FROZEN_SOURCES = (
    "scripts/deepswe_v68_gen/run.py",
    "external_agents_eval/workflows/solve_review_repair/v68_generation.py",
    "external_agents_eval/workflows/solve_review_repair/v68_lint.py",
    "external_agents_eval/workflows/solve_review_repair/ft_protocol.py",
    "scripts/deepswe_v67_gen/run.py",
)
LINT_PATH = PROJECT / "external_agents_eval/workflows/solve_review_repair/v68_lint.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v67run = load_module("v68_v67_runner", PROJECT / "scripts/deepswe_v67_gen/run.py")


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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {name: sha(PROJECT / name) for name in FROZEN_SOURCES}


def load_native(root):
    if importlib.metadata.version("mini-swe-agent") != "2.4.6":
        raise ValueError("This experiment requires mini-SWE-agent 2.4.6 (M0's version)")
    path = root / "scripts/m0_run.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("v68_frozen_m0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode_config(config, header, steps, wall):
    """M0's Mini model/environment/templates; only the task section and limits differ."""
    result = copy.deepcopy(config)
    template = config["agent"]["instance_template"]
    if MARKER not in template:
        raise ValueError("Unexpected upstream Mini instance template")
    result["agent"]["instance_template"] = header + template[template.index(MARKER):]
    result["agent"].update(step_limit=steps, wall_time_limit_seconds=wall, cost_limit=0)
    return result


def task_sources():
    """task_id -> (split, batch, a_dir, evidence sha) from the frozen V6.7 selections."""
    rows = {}
    for root in V67_ROOTS:
        for row in read(Path(root) / "selection.json")["tasks"]:
            rows[row["task_id"]] = row
    return rows


def resolve_image(m0_root, task):
    path = Path(m0_root) / "outputs" / task["task_id"] / "image.json"
    if path.exists():
        image = read(path)
        return {"id": image["id"], "repo_digests": image["repo_digests"]}
    import docker

    client = docker.from_env(timeout=60)
    try:
        image = client.images.get(task["image"])
        return {"id": image.id, "repo_digests": image.attrs.get("RepoDigests", []), "resolved_from": task["image"]}
    finally:
        client.close()


def prepare(root, task_ids, control, note):
    import yaml

    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir()
    for target, source in NATIVE_COPIES.items():
        digests = {sha(Path(m) / "scripts" / source) for m in M0_ROOTS}
        if len(digests) != 1:
            raise ValueError("M0 roots disagree on " + source)
        shutil.copyfile(Path(M0_ROOTS[0]) / "scripts" / source, root / "scripts" / target)
    native = load_native(root)
    (root / "configs").mkdir()
    config = native.prepare_config(root)
    for m0 in M0_ROOTS:
        if config != yaml.safe_load((Path(m0) / "configs/native-deepseek-v4.1-flash.yaml").read_text()):
            raise ValueError("Mini configuration differs from M0: " + m0)
    (root / "configs/plan-mini.yaml").write_text(yaml.safe_dump(
        episode_config(config, v68.plan_header(), v68.PLAN_STEP_LIMIT, v68.PLAN_WALL_TIME_SEC),
        allow_unicode=True, sort_keys=False))
    (root / "configs/batch-mini.yaml").write_text(yaml.safe_dump(
        episode_config(config, v68.batch_header(), v68.BATCH_STEP_LIMIT, v68.BATCH_WALL_TIME_SEC),
        allow_unicode=True, sort_keys=False))
    sources = task_sources()
    manifests = {m0: {t["task_id"]: t for t in read(Path(m0) / "manifests/public-tasks.json")} for m0 in M0_ROOTS}
    rows = []
    for tid in task_ids:
        homes = [m0 for m0 in M0_ROOTS if tid in manifests[m0]]
        if len(homes) != 1:
            raise ValueError(f"{tid}: expected exactly one M0 root, found {homes}")
        task = manifests[homes[0]][tid]
        source = sources[tid]
        evidence = Path(source["a_dir"]) / "v6_review_repair/v6.2/evidence.json"
        if sha(evidence) != source["evidence_sha256"]:
            raise ValueError(f"{tid}: frozen evidence changed")
        rows.append({"task_id": tid, "split": source["split"], "batch": source["batch"], "m0_root": homes[0],
                     "task": task, "image": resolve_image(homes[0], task), "evidence": str(evidence),
                     "evidence_sha256": source["evidence_sha256"]})
    selection = {"experiment_id": EXPERIMENT_ID, "version": v68.VERSION, "runner_version": RUNNER_VERSION,
                 "note": note, "task_count": len(rows), "tasks": rows, "control": control,
                 "code_sha256": code_hashes(),
                 "native_source_sha256": {n: sha(root / "scripts" / n) for n in NATIVE_COPIES},
                 "config_sha256": {n: sha(root / "configs" / n) for n in ("plan-mini.yaml", "batch-mini.yaml")},
                 "limits": {"plan": [v68.PLAN_STEP_LIMIT, v68.PLAN_WALL_TIME_SEC],
                            "batch": [v68.BATCH_STEP_LIMIT, v68.BATCH_WALL_TIME_SEC],
                            "batch_max_targets": v68.BATCH_MAX_TARGETS,
                            "control_max_attempts": v68.CONTROL_MAX_ATTEMPTS}}
    save(root / "selection.json", selection)
    return selection


def validate(root):
    import yaml

    selection = read(root / "selection.json")
    if selection["experiment_id"] != EXPERIMENT_ID or code_hashes() != selection["code_sha256"]:
        raise ValueError("Wrong experiment or frozen code changed")
    for name, digest in selection["native_source_sha256"].items():
        if sha(root / "scripts" / name) != digest:
            raise ValueError("Frozen native source changed: " + name)
    configs = {}
    for name, digest in selection["config_sha256"].items():
        if sha(root / "configs" / name) != digest:
            raise ValueError("Episode configuration changed: " + name)
        configs[name.split("-")[0]] = yaml.safe_load((root / "configs" / name).read_text())
    return selection, configs


# --- containers -----------------------------------------------------------------------------------

def baseline_environment(native, root, row, config, output):
    output.mkdir(parents=True, exist_ok=True)
    task = dict(row["task"])
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


def untar_json(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and name.endswith((".json", ".jsonl", ".md", ".py")):
                (destination / name).write_bytes(tar.extractfile(member).read())


def run_episode(native, root, row, config, output, files, *, model_patch=None):
    """One Mini episode in a fresh baseline container; /tmp/v68 is copied out afterwards."""
    if (output / "episode.json").exists():
        return read(output / "episode.json")
    if output.exists():
        raise RuntimeError(f"{output} has a partial episode; not resampled")
    record = {"task_id": row["task_id"], "started_at": native.now()}
    environment = None
    try:
        environment, container = baseline_environment(native, root, row, config, output / "container")
        for name, data in files.items():
            native.upload(container, data, f"{v68.WORK_DIR}/{name}")
        agent = native.DefaultAgent(native.get_model(config=config["model"]), environment, **config["agent"],
                                    output_path=output / "traj.json")
        try:
            if model_patch is not None:
                with patch("litellm.completion", model_patch):
                    info = agent.run(row["task"]["instruction"])
            else:
                info = agent.run(row["task"]["instruction"])
            record["exit_status"] = info.get("exit_status")
        except Exception as error:  # noqa: BLE001 - keep what the agent wrote
            record.update(exit_status="agent_error", error_type=type(error).__name__, error=str(error)[:2000])
        record["steps"] = sum(1 for m in getattr(agent, "messages", [])
                              if isinstance(m, dict) and m.get("role") == "assistant")
        lint = environment.execute({"command": f"cd {v68.WORK_DIR} && python lint.py"}, timeout=120)
        record["lint_ok"] = lint["returncode"] == 0
        (output / "lint.txt").write_text(lint["output"])
        chunks, _ = container.get_archive(v68.WORK_DIR)
        untar_json(b"".join(chunks), output / "files")
    finally:
        close(native, environment)
    record["finished_at"] = native.now()
    save(output / "episode.json", record)
    return record


def load_episode_data(output, scope):
    data, errors = v68_lint.load_dir(str(output / "files"), scope)
    return data, errors


# --- per task -------------------------------------------------------------------------------------

def common_files(row):
    return {"task.md": row["task"]["instruction"].encode(), "lint.py": LINT_PATH.read_bytes()}


def run_task(native, root, row, configs, containers, control_slots, control_on, client, *, model_patch=None):
    tid = row["task_id"]
    out = root / "tasks" / tid
    if (out / "result.json").exists():
        return read(out / "result.json")
    files = {**common_files(row), "scope.json": json.dumps({"mode": "plan"}).encode()}
    with containers:
        plan = run_episode(native, root, row, configs["plan"], out / "plan", files, model_patch=model_patch)
    data, errors = load_episode_data(out / "plan", {"mode": "plan"})
    listed = [o for o in data["obligations"] if isinstance(o, dict) and o.get("id")]
    obligations = v68.capped(listed)
    if not obligations:
        result = {"task_id": tid, "status": "no_obligations", "plan": plan, "errors": errors}
        save(out / "result.json", result)
        return result
    batches = v68.target_batches(obligations)
    plan_bytes = json.dumps(obligations, ensure_ascii=False, indent=2).encode()

    def agent_batch(index, targets):
        scope = {"mode": "batch", "targets": targets}
        batch_files = {**common_files(row), "obligations.json": plan_bytes, "scope.json": json.dumps(scope).encode()}
        with containers:
            record = run_episode(native, root, row, configs["batch"], out / f"batch-{index:02d}", batch_files,
                                 model_patch=model_patch)
        batch_data, batch_errors = load_episode_data(out / f"batch-{index:02d}", scope)
        return {"batch": index, "targets": targets, "record": record, "parse_errors": batch_errors,
                "entities": batch_data["entities"], "items": batch_data["items"],
                "not_applicable": batch_data["not_applicable"]}

    def control_batch(index, targets):
        with control_slots:
            return run_control(root, row, obligations, index, targets, client)

    with ThreadPoolExecutor(max_workers=len(batches) * (2 if control_on else 1)) as pool:
        agent_futures = [pool.submit(agent_batch, i, t) for i, t in enumerate(batches, 1)]
        control_futures = [pool.submit(control_batch, i, t) for i, t in enumerate(batches, 1)] if control_on else []
        agent_results = [f.result() for f in agent_futures]
        control_results = [f.result() for f in control_futures]
    arms = {"agent": agent_results}
    if control_on:
        arms["control"] = control_results
    summary = {}
    for arm, results in arms.items():
        merged = v68.merge_batches(obligations, results)
        save(out / arm / "data.json", merged)
        (out / arm / "ledger.md").write_text(v68.ledger_markdown(merged))
        (out / arm / "checkpoints.md").write_text(v68.checkpoints_markdown(merged))
        summary[arm] = {**v68.summarize(merged),
                        "batches": [{"batch": r["batch"], "targets": r["targets"],
                                     "lint_ok": (r.get("record") or {}).get("lint_ok"),
                                     "exit_status": (r.get("record") or {}).get("exit_status"),
                                     "steps": (r.get("record") or {}).get("steps"),
                                     "attempts": r.get("attempts")} for r in results]}
    result = {"task_id": tid, "status": "done", "plan": plan, "obligations": len(obligations),
              "obligations_listed": len(listed),
              "batches": len(batches), "summary": summary, "finished_at": native.now()}
    save(out / "result.json", result)
    return result


def run_control(root, row, obligations, index, targets, client):
    """Tool-free control: same item rules, the agent's obligations/batch, frozen evidence."""
    out = root / "tasks" / row["task_id"] / f"control-{index:02d}"
    packet = read(row["evidence"])
    messages = v68.control_messages(render_user(packet, packet["evidence"]), obligations, targets)
    repo = v68_lint.EvidenceRepo(packet["evidence"])
    scope = {"mode": "batch", "targets": targets}
    attempts, data = [], {"entities": [], "items": [], "not_applicable": []}
    transport, usage = [], []
    for attempt in range(1, v68.CONTROL_MAX_ATTEMPTS + 1):
        result = v67run.call(client, messages, out / f"attempt-{attempt}", transport, usage)
        try:
            payload = strict_json(result["content"])
            data, errors = v68.control_data(payload)
        except ValueError as error:
            errors = [f"json: {error}"]
        if not errors:
            lint_errors, _, _ = v68_lint.lint(row["task"]["instruction"],
                                              {**data, "obligations": obligations}, repo, scope)
            errors = lint_errors
        if result["finish_reason"] != "stop":
            errors = [f"finish: {result['finish_reason']}", *errors]
        save(out / f"attempt-{attempt}/lint.json", {"errors": errors})
        attempts.append({"attempt": attempt, "finish_reason": result["finish_reason"], "errors": len(errors)})
        if not errors:
            break
        messages = [*messages, {"role": "assistant", "content": result["content"]},
                    {"role": "user", "content": v67run.correction(errors)}]
    save(out / "transport.json", transport)
    save(out / "usage.json", usage)
    return {"batch": index, "targets": targets, "attempts": attempts, "record": {"lint_ok": attempts[-1]["errors"] == 0},
            "entities": data["entities"], "items": data["items"], "not_applicable": data["not_applicable"]}


def run_all(root, selection, configs, max_containers, *, model_patch=None, client=None, only=None):
    native = load_native(root)
    containers = threading.BoundedSemaphore(max_containers)
    control_slots = threading.BoundedSemaphore(6)
    rows = [r for r in selection["tasks"] if not only or r["task_id"] in only]
    lock, results = threading.Lock(), {}

    def one(row):
        tid = row["task_id"]
        try:
            ensure_image(native, row)
            result = run_task(native, root, row, configs, containers, control_slots, selection["control"],
                              client, model_patch=model_patch)
        except Exception as error:  # noqa: BLE001 - keep per-task failures explicit
            result = {"task_id": tid, "status": "stage_error", "error_type": type(error).__name__,
                      "error": str(error)[:2000]}
            save(root / "errors" / f"{tid}.json", result)
        with lock:
            results[tid] = result
            save(root / "result.json", {"experiment_id": EXPERIMENT_ID, "results": list(results.values()),
                                        "updated_at": native.now()})
        print(json.dumps({"task_id": tid, "status": result["status"],
                          "items": {a: s.get("items") for a, s in (result.get("summary") or {}).items()}}), flush=True)

    with ThreadPoolExecutor(max_workers=len(rows)) as pool:
        for future in as_completed([pool.submit(one, row) for row in rows]):
            future.result()
    return results


def ensure_image(native, row):
    client = native.docker.from_env(timeout=60)
    try:
        image = client.images.get(row["image"]["id"])
        if image.id != row["image"]["id"]:
            raise ValueError("Local task image differs from the frozen image")
    finally:
        client.close()


# --- preflight (stub model, zero API calls) -------------------------------------------------------

def preflight(root, selection, configs):
    import litellm

    row = selection["tasks"][0]
    instruction = " ".join(row["task"]["instruction"].split())
    quote = instruction[:60]
    obligations = [{"id": "O1", "statement": "stub", "task_quote": quote}]
    na = [{"obligation": "O1", "ext": ext, "reason": "preflight stub entry"} for ext in v68.EXTENSIONS]
    plan_cmd = (f"cat > {v68.WORK_DIR}/obligations.json <<'JSON'\n{json.dumps(obligations)}\nJSON\n"
                f"python {v68.WORK_DIR}/lint.py")
    batch_cmd = (f"echo '[]' > {v68.WORK_DIR}/entities.json && : > {v68.WORK_DIR}/items.jsonl && "
                 f"cat > {v68.WORK_DIR}/not_applicable.json <<'JSON'\n{json.dumps(na)}\nJSON\n"
                 f"python {v68.WORK_DIR}/lint.py")
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        prompt = json.dumps(kwargs.get("messages", [])[:2])
        first = batch_cmd if "Your target obligations" in prompt else plan_cmd
        turn = sum(1 for m in kwargs.get("messages", []) if m.get("role") == "assistant")
        command = first if turn == 0 else v68.SUBMIT
        return litellm.ModelResponse(model=kwargs["model"], choices=[{
            "index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Preflight.", "reasoning_content": "",
                "tool_calls": [{"id": f"pre-{len(calls)}", "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})}}],
            }}], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    scratch = root / "preflight-run"
    scratch.mkdir(exist_ok=True)
    for name in ("scripts", "configs"):
        if not (scratch / name).exists():
            (scratch / name).symlink_to(root / name, target_is_directory=True)
    preflight_selection = {**selection, "tasks": [row], "control": False}
    results = run_all(scratch, preflight_selection, configs, 2, model_patch=completion)
    result = results[row["task_id"]]
    passed = (result.get("status") == "done" and result["plan"].get("lint_ok")
              and all(b["lint_ok"] for b in result["summary"]["agent"]["batches"]))
    isolation = read(scratch / "tasks" / row["task_id"] / "plan/container/isolation.json")
    from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_json

    save(root / "preflight/readiness.json", {
        "passed": bool(passed), "model_api_calls": 0, "simulated_responses": len(calls),
        "selection_sha256": sha256_json(selection), "task_id": row["task_id"], "result": result,
        "isolation": isolation, "finished_at": load_native(root).now()})
    if not passed:
        raise RuntimeError("Preflight failed; see preflight/readiness.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "run"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--note")
    parser.add_argument("--containers", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        selection = prepare(root, args.task_id, args.control, args.note)
    else:
        selection, configs = validate(root)
        if args.mode == "preflight":
            preflight(root, selection, configs)
        else:
            from external_agents_eval.workflows.solve_review_repair.artifacts import sha256_json

            readiness = read(root / "preflight/readiness.json")
            if not readiness["passed"] or readiness["selection_sha256"] != sha256_json(selection):
                raise ValueError("Run the zero-API preflight for this frozen selection first")
            if not os.environ.get("DEEPSEEK_API_KEY"):
                raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
            client = v67run.DeepSeekClient(os.environ["DEEPSEEK_API_KEY"]) if selection["control"] else None
            run_all(root, selection, configs, args.containers, client=client, only=args.task_id)
    print(json.dumps({"mode": args.mode, "experiment_id": EXPERIMENT_ID, "task_count": selection["task_count"]}))


if __name__ == "__main__":
    main()
