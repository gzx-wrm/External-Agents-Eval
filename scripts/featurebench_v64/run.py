"""featurebench_v6.4_woG on the 27 CPU-scored FeatureBench Lite tasks.

Entries (each task runs in fresh containers of its original image):
  prepare          bind the 27 tasks, their native-Mini (M0) results, per-task scoring
                   environments and the frozen code; zero API.
  preflight        per task: official preparation reproduces M0's task tree, every role
                   worktree imports its own code, A's Pi reads alias /testbed; zero API.
  ledger           A only (unchanged V6.3 Mini A), one sample per task, frozen.
  preflight-solver native Solver protocol with the frozen ledger supplement; zero API.
  run --from solver   frozen ledger -> native Mini Solver -> I/R -> S -> route -> Repair
                   -> official FeatureBench scoring (final; P0 separately for analysis).
  run --from A     the complete pipeline, A included.
  launch <entry>   read the model key from stdin and start one entry detached.
No build gate (woG). Reference/control checks are reused from M0, not rerun.
Expanded Docker images are removed after their image group; the OCI cache is never touched.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
for entry in (str(PROJECT), str(PROJECT / "scripts/deepswe_v64")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ledger_input_v64 import GUIDANCE, HEADING, augment_task, canonical_sha256  # noqa: E402

EXPERIMENT_ID = "featurebench_v6.4_woG"
NATIVE_PYTHON = "/root/v62-swe-verified100-20261026/venv312/bin/python"
SCORE_PYTHON = "/root/v63-featurebench-20261028/venv312/bin/python"
OFFICIAL_SOURCE = "/root/v63-featurebench-20261028/vendor/FeatureBench"
DOCKER_HOST = "unix:///run/featurebench-lite-docker.sock"
# Native-Mini (M0) batches; a later batch supersedes an earlier one per instance_id.
M0_RUNS = [
    "/root/native-mini-featurebench-lite-20261028-w3-r1",
    "/root/native-mini-featurebench-lite-precheck10-20261030-w3-r1",
    "/root/native-mini-featurebench-lite-resume7-20261031-w3-r1",
    "/root/native-mini-featurebench-lite-astropy-iers-20261031-w1-r1",
    "/root/native-mini-featurebench-lite-testsfetcher-hfmirror-20261101-w1-r1",
]
# Latest native runtime (identical to earlier ones except IERS hooks that act only when configured).
NATIVE_RUNTIME = "/root/native-mini-featurebench-lite-astropy-iers-20261031-w1-r1/scripts"
NATIVE_SCRIPTS = ("generate.py", "evaluate.py", "common.py", "native_core.py", "native_compat.py",
                  "sanitize_repository.py", "pristine_reference.py", "cached_images.py", "iers_data.py")
IMAGE_CACHE_CONFIG = "/root/native-mini-featurebench-lite-resume7-20261031-w3-r1/configs/image-cache.json"
EXCLUDED = {
    "huggingface__trl.02a34777.test_data_utils.827a9d15.lv1":
        "excluded: GPU (bf16) dependency and removed upstream Hub models",
    "linkedin__Liger-Kernel.c856fbab.test_poly_norm.7b0e3399.lv1": "deferred: real GPU dependency",
    "linkedin__Liger-Kernel.c856fbab.test_fused_neighborhood_attention.78217be4.lv2":
        "deferred: real GPU (Triton) dependency",
}
TASK_COUNT = 27
WORKERS = 3
# Small, fast image groups first so process problems surface early.
GROUP_PRIORITY = ("pydantic", "metaflow", "sympy", "sphinx", "seaborn", "xarray", "astropy",
                  "pandas", "pytorch_lightning", "mlflow", "transformers")
STAGE_MODEL_CONFIG = PROJECT / "configs/v6.1-deepseek-v4.1-flash.json"
FROZEN_SOURCES = [
    "scripts/featurebench_v64/run.py",
    "scripts/deepswe_v64/ledger_input_v64.py",
    "external_agents_eval/benchmarks/featurebench/v64.py",
    "external_agents_eval/benchmarks/featurebench/editable_assets.py",
    "external_agents_eval/benchmarks/featurebench/adapter.py",
    "external_agents_eval/benchmarks/featurebench/agents.py",
    "external_agents_eval/benchmarks/swebench/isolated_environment.py",
    "external_agents_eval/agents/mini_v64.py",
    "external_agents_eval/agents/mini_v63.py",
    "external_agents_eval/agents/mini_v62.py",
    "external_agents_eval/agents/mini_ft_review.py",
    "external_agents_eval/agents/mini_review_files.py",
    "external_agents_eval/agents/mini_runtime_assets.py",
    "external_agents_eval/agents/pi_file_tools.py",
    "external_agents_eval/agents/pi_file_tools.mjs",
    "external_agents_eval/agents/pier/pi_role_runner.py",
    "external_agents_eval/agents/pier/pi_v6_solve_review_repair.py",
    "external_agents_eval/workflows/solve_review_repair/v5_controller.py",
    "external_agents_eval/workflows/solve_review_repair/v5_prompts.py",
    "external_agents_eval/workflows/solve_review_repair/v5_config.py",
    "external_agents_eval/workflows/solve_review_repair/v5_contracts.py",
    "external_agents_eval/workflows/solve_review_repair/v6_controller.py",
    "external_agents_eval/workflows/solve_review_repair/v64_config.py",
    "external_agents_eval/workflows/solve_review_repair/ft_evidence.py",
    "configs/v6.1-deepseek-v4.1-flash.json",
]
SKILL_DIR = "external_agents_eval/workflows/solve_review_repair/skills/v6.4-review-note-files"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def now():
    return datetime.now(timezone.utc).isoformat()


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def tree_sha256(root) -> str:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sha256_bytes(b"".join(
        p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0" for p in files))


def code_hashes(project=PROJECT):
    hashes = {name: file_sha256(Path(project) / name) for name in FROZEN_SOURCES}
    hashes[SKILL_DIR + "/"] = tree_sha256(Path(project) / SKILL_DIR)
    return hashes


def private_digest(task) -> str:
    return sha256_bytes(json.dumps(task, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode())


def augment_statement(statement: str, ledger: dict) -> str:
    """The DeepSWE V6.4 supplement, appended to FeatureBench's public problem statement."""
    return augment_task({"instruction": statement}, ledger)["instruction"]


def image_group(name: str) -> str:
    return name.rsplit("/", 1)[-1].removeprefix("featurebench-specs_").split("-instance_")[0]


def group_order(images):
    def key(name):
        group = image_group(name)
        rank = next((i for i, prefix in enumerate(GROUP_PRIORITY) if group.startswith(prefix)),
                    len(GROUP_PRIORITY))
        return rank, name
    return sorted(set(images), key=key)


def library_name(task) -> str | None:
    if task["level"] != 1:
        return None  # Lv2 starts from an empty repository; there is no package to import yet.
    settings = task["repo_settings"]
    settings = json.loads(settings) if isinstance(settings, str) else settings
    return settings.get("library_name")


def latest_m0(runs=None):
    latest = {}
    for run in runs or M0_RUNS:
        for path in sorted(Path(run).glob("outputs/*/result.json")):
            latest[path.parent.name] = Path(run)
    return latest


def scoring_config_dir(root, tid):
    return Path(root) / "configs/tasks" / tid


def task_cpu_config(root, tid):
    path = scoring_config_dir(root, tid) / "cpu-evaluation.json"
    return path if path.exists() else None


def task_iers_config(root, tid):
    path = scoring_config_dir(root, tid) / "iers-leap-seconds.json"
    return path if path.exists() else None


def reference_passed(summary: dict) -> bool:
    if summary.get("cpu_reference_kind") == "pristine_original_tests":
        return bool(summary.get("completed") and not summary.get("error") and summary.get("probe_passed")
                    and summary.get("tests_executed", 0) > 0 and summary.get("test_exit_code") == 0
                    and summary.get("tests_failed") == 0 and summary.get("tests_errors") == 0)
    return bool(summary.get("completed") and not summary.get("error") and summary.get("resolved")
                and summary.get("f2p_total", 0) > 0)


def control_passed(summary: dict) -> bool:
    return bool(not summary.get("error") and summary.get("patch_applied") and not summary.get("resolved"))


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(root: Path):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new, empty experiment directory")
    latest = latest_m0()
    private, public = {}, {}
    for run in M0_RUNS:
        for name, store in (("tasks-private.json", private), ("tasks-public.json", public)):
            path = Path(run) / "manifests" / name
            if path.exists():
                for task in read(path):
                    store.setdefault(task["instance_id"], task)
    rows, tasks, tasks_public = [], [], []
    for tid, run in sorted(latest.items()):
        result = read(run / "outputs" / tid / "result.json")
        if tid in EXCLUDED:
            if result.get("status") == "scored":
                raise ValueError("An excluded task has an M0 score: " + tid)
            continue
        if result.get("status") != "scored":
            raise ValueError("A selected task has no valid M0 score: " + tid)
        task, pub = private[tid], public[tid]
        if pub["private_task_sha256"] != private_digest(task):
            raise ValueError("Frozen task content differs from its public digest: " + tid)
        output = run / "outputs" / tid
        reference, control = read(output / "reference/summary.json"), read(output / "control/summary.json")
        if not reference_passed(reference) or not control_passed(control):
            raise ValueError("M0 reference/control did not pass: " + tid)
        isolation = read(output / "generation/isolation.json")
        row = {
            "task_id": tid, "level": task["level"], "image_name": task["image_name"],
            "library": library_name(task), "m0_tree_sha": isolation["tree_sha"],
            "m0": {"run": str(run), "result_sha256": file_sha256(output / "result.json"),
                   "status": result["status"], "reward": result["reward"],
                   "official": {k: result["official"].get(k) for k in
                                ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total", "resolved")},
                   "patch_sha256": file_sha256(output / "generation/model.patch")},
            "reused_checks": {
                name: {"path": str(output / rel), "sha256": file_sha256(output / rel)}
                for name, rel in (("reference", "reference/summary.json"),
                                  ("control", "control/summary.json"),
                                  ("cpu_readiness", "cpu-readiness.json"))
            },
        }
        directory = scoring_config_dir(root, tid)
        cpu = run / "configs/cpu-evaluation.json"
        if cpu.exists():
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cpu, directory / "cpu-evaluation.json")
            row["scoring_environment"] = {"cpu_config_source": str(cpu), "sha256": file_sha256(cpu),
                                          "settings": read(cpu)}
        else:
            row["scoring_environment"] = {"cpu_config_source": None,
                                          "settings": "official defaults with network none (M0 batch 1)"}
        iers = run / "configs/iers-leap-seconds.json"
        if iers.exists():
            settings = read(iers)
            data = (iers.parent / settings["file"]).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data, directory / data.name)
            save(directory / "iers-leap-seconds.json", {**settings, "file": data.name,
                                                       "original_configuration": str(iers)})
            row["iers_data"] = {"source": str(iers), "data_sha256": file_sha256(data)}
        rows.append(row)
        tasks.append(task)
        tasks_public.append(pub)
    if len(rows) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} scored tasks, found {len(rows)}")
    order = {name: index for index, name in enumerate(group_order(r["image_name"] for r in rows))}
    rows.sort(key=lambda r: (order[r["image_name"]], r["task_id"]))
    root.mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir()
    for name in NATIVE_SCRIPTS:
        shutil.copyfile(Path(NATIVE_RUNTIME) / name, root / "scripts" / name)
    first = Path(M0_RUNS[0]) / "configs/native-deepseek-v4.1-flash.yaml"
    for run in M0_RUNS[1:]:
        if (Path(run) / "configs/native-deepseek-v4.1-flash.yaml").read_bytes() != first.read_bytes():
            raise ValueError("M0 batches used different native Solver configurations")
    (root / "configs").mkdir(exist_ok=True)
    shutil.copyfile(first, root / "configs/native-deepseek-v4.1-flash.yaml")
    shutil.copyfile(IMAGE_CACHE_CONFIG, root / "configs/image-cache.json")
    save(root / "manifests/tasks-private.json", tasks)
    save(root / "manifests/tasks-public.json", tasks_public)
    for task in tasks:
        save(root / "inputs" / task["instance_id"] / "task-private.json", task)
    selection = {
        "experiment_id": EXPERIMENT_ID, "task_count": len(rows), "tasks": rows,
        "excluded": EXCLUDED, "workers": WORKERS, "build_gate": False,
        "pipeline": "A (V6.3 Mini A, frozen) -> native Mini Solver + ledger -> I/R -> S -> route -> "
                    "Repair -> official FeatureBench scoring",
        "review": {"prompt_version": "v5-6", "controller_version": "v5-5", "workflow": "6.4",
                   "build_gate_enabled": False},
        "stage_model_config": str(STAGE_MODEL_CONFIG.relative_to(PROJECT)),
        "solver_guidance_sha256": sha256_bytes((HEADING + "\n\n" + GUIDANCE).encode()),
        "native_runtime": {name: file_sha256(root / "scripts" / name) for name in NATIVE_SCRIPTS},
        "native_config_sha256": file_sha256(root / "configs/native-deepseek-v4.1-flash.yaml"),
        "code_sha256": code_hashes(), "project": str(PROJECT),
        "m0_runs": M0_RUNS, "reference_control_reused_from_m0": True,
        "p0_scored_for_analysis_only": True,
        "image_policy": "expanded Docker copies removed after each image group; OCI cache never modified",
        "prepared_at": now(),
    }
    save(root / "selection.json", selection)
    return selection


def validate(root: Path):
    selection = read(root / "selection.json")
    if selection["experiment_id"] != EXPERIMENT_ID or len(selection["tasks"]) != TASK_COUNT:
        raise ValueError("Wrong experiment directory")
    if code_hashes() != selection["code_sha256"]:
        changed = [k for k, v in code_hashes().items() if selection["code_sha256"].get(k) != v]
        raise ValueError("Source changed after prepare: " + ", ".join(changed))
    for name, digest in selection["native_runtime"].items():
        if file_sha256(root / "scripts" / name) != digest:
            raise ValueError("Native runtime changed: " + name)
    if file_sha256(root / "configs/native-deepseek-v4.1-flash.yaml") != selection["native_config_sha256"]:
        raise ValueError("Native Solver configuration changed")
    for row in selection["tasks"]:
        task = read(root / "inputs" / row["task_id"] / "task-private.json")
        if task["instance_id"] != row["task_id"] or task["image_name"] != row["image_name"]:
            raise ValueError("Task input changed: " + row["task_id"])
        for check in row["reused_checks"].values():
            if file_sha256(check["path"]) != check["sha256"]:
                raise ValueError("Reused M0 check changed: " + check["path"])
    return selection


# ---------------------------------------------------------------------------
# runtime bindings (native FeatureBench runtime lives under <root>/scripts)
# ---------------------------------------------------------------------------

class Runtime:
    def __init__(self, root: Path):
        if os.environ.get("DOCKER_HOST") != DOCKER_HOST:
            raise ValueError("Use the dedicated FeatureBench Docker engine: DOCKER_HOST=" + DOCKER_HOST)
        for entry in (str(root / "scripts"), OFFICIAL_SOURCE):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        import docker
        import generate
        import native_core
        from cached_images import ensure_cached_image
        from featurebench.infer.models import TaskInstance
        from iers_data import install_leap_seconds
        from minisweagent.environments import get_environment

        self.root = root
        self.docker = docker
        self.generate = generate
        self.native = native_core
        self.ensure_cached_image = ensure_cached_image
        self.TaskInstance = TaskInstance
        self.install_leap_seconds = install_leap_seconds
        self.get_environment = get_environment
        # Validated once against M0's frozen Solver configuration (never rewritten concurrently).
        self.config = generate.configuration(root)

    def client(self):
        return self.docker.from_env(timeout=1800)

    def open_prepared(self, task, image_id, output):
        """Fresh task container prepared exactly like the native Solver's (official init + sanitize)."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        options = copy.deepcopy(self.config["environment"])
        options["image"] = image_id
        runtime = self.TaskInstance.from_dict(task).get_docker_runtime_config()
        options["env"].update(runtime["env_vars"])
        for assignment in runtime["env_exports"]:
            key, value = assignment.removeprefix("export ").split("=", 1)
            options["env"][key] = value
        if runtime["shm_size"]:
            options["run_args"].append("--shm-size=" + runtime["shm_size"])
        environment = self.get_environment(options)
        client = self.client()
        try:
            container = client.containers.get(environment.container_id)
            iers = task_iers_config(self.root, task["instance_id"])
            if iers is not None:
                self.install_leap_seconds(container, iers, output / "iers-data.json")
            audit = self.generate.prepare(self.root, environment, container, task, output)
        except BaseException:
            self.close(environment, client)
            raise
        return environment, container, audit, client

    def close(self, environment, client):
        try:
            if environment is not None and environment.container_id:
                try:
                    client.containers.get(environment.container_id).remove(force=True)
                except self.docker.errors.NotFound:
                    pass
                environment.container_id = None
        finally:
            client.close()

    def load_image(self, repository):
        client = self.client()
        try:
            return self.ensure_cached_image(self.root, repository, client)
        finally:
            client.close()

    def remove_expanded_image(self, image_id):
        client = self.client()
        try:
            client.images.remove(image_id, force=True, noprune=False)
            return None
        except self.docker.errors.APIError as error:
            return str(error)
        finally:
            client.close()


def task_input(root, tid):
    return read(Path(root) / "inputs" / tid / "task-private.json")


def set_aside(path: Path):
    """Keep an interrupted attempt's files; the next attempt starts in a fresh directory."""
    path = Path(path)
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path.rename(path.with_name(f"{path.name}.incomplete-{stamp}"))


def check_tree(audit, expected, label):
    if audit["tree_sha"] != expected:
        raise ValueError(f"{label} task tree {audit['tree_sha']} differs from M0's {expected}")


# ---------------------------------------------------------------------------
# preflight (zero API)
# ---------------------------------------------------------------------------

def preflight_task(rt, row, image_id):
    from external_agents_eval.benchmarks.featurebench.v64 import FeatureBenchV64Environment
    from external_agents_eval.benchmarks.featurebench.v64 import check_role_worktrees

    tid = row["task_id"]
    output = rt.root / "preflight/tasks" / tid
    environment = client = None
    try:
        environment, container, audit, client = rt.open_prepared(task_input(rt.root, tid), image_id,
                                                                 output / "preparation")
        check_tree(audit, row["m0_tree_sha"], "Preflight")
        bridge = FeatureBenchV64Environment(container, instance_id=tid)
        report = asyncio.run(check_role_worktrees(bridge, base_commit=audit["synthetic_commit"],
                                                  library=row["library"], output=output))
        return {"passed": report["passed"], "tree_sha": audit["tree_sha"], "level": row["level"],
                "library": row["library"], "pi_read": report.get("pi_read"),
                "roles": {k: v.get("inside_worktree", v.get("import_check", v.get("error", "")[:200]))
                          for k, v in report["roles"].items()}}
    except Exception as error:  # noqa: BLE001 - record per task and continue
        return {"passed": False, "error": f"{type(error).__name__}: {error}"[:3000]}
    finally:
        if client is not None:
            rt.close(environment, client)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def ledger_paths(root, tid):
    return Path(root) / "ledgers" / f"{tid}.json", Path(root) / "outputs" / tid / "ledger/ledger-result.json"


def load_ledger(root, tid):
    path, result_path = ledger_paths(root, tid)
    if not result_path.exists():
        return None, None
    result = read(result_path)
    if result["status"] != "ledger_ready":
        return None, result
    if file_sha256(path) != result["ledger_file_sha256"]:
        raise ValueError("Frozen ledger file changed: " + tid)
    ledger = read(path)
    if canonical_sha256(ledger) != result["ledger_sha256"]:
        raise ValueError("Frozen ledger content changed: " + tid)
    return ledger, result


def ledger_stage(rt, row, image_id, update):
    """Unchanged V6.3 Mini A; one sample; stops before Solver."""
    from external_agents_eval.benchmarks.featurebench.v64 import FeatureBenchV64Environment
    from external_agents_eval.benchmarks.featurebench.v64 import FeatureBenchLedgerAgent, LedgerReady

    tid = row["task_id"]
    ledger_file, result_path = ledger_paths(rt.root, tid)
    if result_path.exists():
        return read(result_path)  # Never resample A.
    attempts = sorted(result_path.parent.glob("attempt-*")) if result_path.parent.exists() else []
    output = result_path.parent / f"attempt-{len(attempts) + 1}"
    task = task_input(rt.root, tid)
    update(tid, "ledger_A")
    environment = client = None
    try:
        environment, container, audit, client = rt.open_prepared(task, image_id, output / "preparation")
        check_tree(audit, row["m0_tree_sha"], "A")
        bridge = FeatureBenchV64Environment(container, instance_id=tid)
        agent = FeatureBenchLedgerAgent(
            logs_dir=output / "agent/logs", model_name="deepseek-flash",
            stage_model_config=STAGE_MODEL_CONFIG,
            extra_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
        )
        started = now()
        try:
            asyncio.run(agent.run(task["problem_statement"], bridge, SimpleNamespace(metadata={})))
        except LedgerReady as ready:
            outcome = ready
        else:
            raise RuntimeError("The ledger-only entry did not stop after A")
        envelope = output / "agent/v6_review_repair/v6/pre-solver-ledger.json"
        result = {"task_id": tid, "attempt": output.name, "previous_attempts": len(attempts),
                  "started_at": started, "finished_at": now(),
                  "tree_sha": audit["tree_sha"], "base_commit": audit["synthetic_commit"],
                  "ledger_quality": outcome.quality, "ledger_warnings": outcome.warnings,
                  "envelope": str(envelope) if envelope.exists() else None}
        if outcome.ledger is None:
            result.update(status="ledger_unavailable", ledger_error=str(outcome.error)[:3000])
        else:
            ledger_file.parent.mkdir(parents=True, exist_ok=True)
            ledger_file.write_text(json.dumps(outcome.ledger, ensure_ascii=False, indent=2) + "\n")
            result.update(
                status="ledger_ready", ledger_file=str(ledger_file),
                ledger_file_sha256=file_sha256(ledger_file), ledger_sha256=canonical_sha256(outcome.ledger),
                counts={k: len(outcome.ledger.get(k, [])) for k in
                        ("obligations", "preservation_obligations", "open_questions")},
            )
        save(result_path, result)
        return result
    finally:
        if client is not None:
            rt.close(environment, client)


def solver_stage(rt, row, image_id, ledger, update):
    tid = row["task_id"]
    output = rt.root / "outputs" / tid / "solver"
    if (output / "generation.json").exists():
        generation = read(output / "generation.json")  # Never resample an existing Solver output.
        patch = (output / "model.patch").read_text()
        if sha256_bytes(patch.encode()) != generation["patch_sha256"]:
            raise ValueError("Existing Solver output is inconsistent")
        return patch, generation
    set_aside(output)
    task = task_input(rt.root, tid)
    statement = augment_statement(task["problem_statement"], ledger)
    if statement.count(HEADING) != 1 or task["problem_statement"] not in statement:
        raise ValueError("Solver input does not carry exactly one ledger supplement")
    save(output / "solver-input.json", {
        "original_sha256": sha256_bytes(task["problem_statement"].encode()),
        "solver_statement_sha256": sha256_bytes(statement.encode()),
        "ledger_sha256": canonical_sha256(ledger), "guidance_heading": HEADING,
    })
    update(tid, "solving")
    environment = client = None
    try:
        environment, container, audit, client = rt.open_prepared(task, image_id, output)
        check_tree(audit, row["m0_tree_sha"], "Solver")
        info = rt.native.make_agent(rt.config, environment, output).run(statement)
        patch = rt.native.checked(environment, rt.native.snapshot_command(audit["synthetic_commit"]))
        (output / "model.patch").write_text(patch)
        generation = {"instance_id": tid, "agent_info": info, "tree_sha": audit["tree_sha"],
                      "patch_sha256": sha256_bytes(patch.encode()), "patch_bytes": len(patch.encode()),
                      "ledger_included": True, "ledger_sha256": canonical_sha256(ledger)}
        save(output / "generation.json", generation)
        return patch, generation
    finally:
        if client is not None:
            rt.close(environment, client)


def count_events(path: Path):
    counts = {"reviewer_worktree_restored": 0, "final_worktree_discarded": 0}
    if not path.exists():
        return counts
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("event") or event.get("type") or event.get("name")
        if kind == "V5_REVIEW_INTEGRITY_WARNING":
            counts["reviewer_worktree_restored"] += 1
        elif kind == "V6_FINAL_WORKTREE_NORMALIZED" and event.get("discarded_changes"):
            counts["final_worktree_discarded"] += 1
    return counts


def review_stage(rt, row, image_id, ledger, p0, generation, update):
    from external_agents_eval.agents.mini_ft_review import normalization_command, write_frozen_fork
    from external_agents_eval.agents.mini_v62 import MiniWorkspaceManager
    from external_agents_eval.benchmarks.featurebench.v64 import FeatureBenchV64Environment
    from external_agents_eval.benchmarks.featurebench.v64 import FeatureBenchV64ReviewAgent

    tid = row["task_id"]
    output = rt.root / "outputs" / tid / "review"
    set_aside(output)
    task = task_input(rt.root, tid)
    update(tid, "reviewing", solver_exit_status=generation["agent_info"].get("exit_status"))
    environment = client = None
    try:
        environment, container, audit, client = rt.open_prepared(task, image_id, output / "preparation")
        check_tree(audit, generation["tree_sha"], "Review")
        base = audit["synthetic_commit"]
        bridge = FeatureBenchV64Environment(container, instance_id=tid)
        probe = MiniWorkspaceManager(environment=bridge, repository="/testbed",
                                     transfer_dir=output / ".baseline-probe")
        environment_patch = asyncio.run(probe.capture_patch(worktree="/testbed", base_commit=base))
        if environment_patch:
            raise RuntimeError("Sanitized FeatureBench baseline unexpectedly has an environment patch")
        rt.native.upload(container, b"", "/tmp/v64-baseline.patch")
        rt.native.upload(container, p0.encode(), "/tmp/v64-p0.patch")
        receipt = json.loads(rt.native.checked(environment, normalization_command(
            base, "/tmp/v64-baseline.patch", "/tmp/v64-p0.patch", "/tmp/v64-normalized.patch",
        )).splitlines()[-1])
        normalized_path = output / "normalized-p0.patch"
        asyncio.run(bridge.download_file("/tmp/v64-normalized.patch", normalized_path))
        normalized = normalized_path.read_bytes()
        provenance = {"m1_output": str(rt.root / "outputs" / tid / "solver"),
                      "original_p0_sha256": sha256_bytes(p0.encode()),
                      "normalized_p0_sha256": sha256_bytes(normalized),
                      "v6_baseline_sha256": sha256_bytes(b""), "ledger_sha256": canonical_sha256(ledger),
                      **receipt}
        save(output / "p0-normalization.json", provenance)
        fork = write_frozen_fork(output, task_id=tid, base_commit=base, baseline_patch=b"",
                                 ledger=ledger, candidate_patch=normalized, provenance=provenance)
        agent = FeatureBenchV64ReviewAgent(
            logs_dir=output / "agent/logs", model_name="deepseek-flash",
            stage_model_config=STAGE_MODEL_CONFIG,
            extra_env={"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]}, fork_plan=str(fork),
        )
        asyncio.run(agent.run(task["problem_statement"], bridge, SimpleNamespace(metadata={})))
        workflow = output / "agent/v6_review_repair"
        manifest = read(workflow / "manifest.json")
        if manifest["workflow_state"] != "completed":
            raise RuntimeError("Review did not complete; no fallback result is scored")
        final = rt.native.checked(environment, rt.native.snapshot_command(base))
        if manifest["route"] in {"accept", "hold"} and sha256_bytes(final.encode()) != sha256_bytes(p0.encode()):
            raise RuntimeError("A path that delivers P0 changed the Solver patch")
        (rt.root / "outputs" / tid / "model.patch").write_text(final)
        return final, manifest, count_events(workflow / "events.jsonl")
    finally:
        if client is not None:
            rt.close(environment, client)


def score(root, tid, patch_text, output):
    output = Path(output)
    set_aside(output)
    output.mkdir(parents=True, exist_ok=True)
    patch_path = output / "candidate.patch"
    patch_path.write_text(patch_text)
    command = [SCORE_PYTHON, str(root / "scripts/evaluate.py"), "--task",
               str(root / "inputs" / tid / "task-private.json"), "--output", str(output),
               "--stage", "model", "--model-patch", str(patch_path)]
    cpu = task_cpu_config(root, tid)
    if cpu is not None:
        command += ["--cpu-config", str(cpu)]
    environment = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    environment.update(PYTHONPATH=OFFICIAL_SOURCE, DOCKER_HOST=DOCKER_HOST)
    with (output.parent / (output.name + ".log")).open("ab") as log:
        subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = read(output / "summary.json")
    if not summary.get("completed") or summary.get("error"):
        return {"status": "scoring_infrastructure_error", "reward": None, "official": summary}
    if summary.get("patch_applied") and summary.get("f2p_total", 0) == 0:
        return {"status": "scoring_no_test_results", "reward": None, "official": summary}
    return {"status": "scored", "reward": int(bool(summary["resolved"])), "official": summary}


def run_case(rt, row, image_id, start, update):
    tid = row["task_id"]
    stage = "ledger"
    try:
        ledger, ledger_result = load_ledger(rt.root, tid)
        if ledger_result is None:
            if start != "A":
                raise ValueError("No frozen ledger; run the ledger entry first")
            ledger_result = ledger_stage(rt, row, image_id, update)
            ledger, ledger_result = load_ledger(rt.root, tid)
        if ledger is None:
            # V6.1 rule: an unusable A ledger ends the task before P0 (framework failure).
            return {"task_id": tid, "status": "terminated_before_p0", "reward": 0,
                    "ledger_status": ledger_result["status"], "finished_at": now()}
        stage = "solver"
        p0, generation = solver_stage(rt, row, image_id, ledger, update)
        stage = "review"
        final, manifest, integrity = review_stage(rt, row, image_id, ledger, p0, generation, update)
        stage = "official_evaluation"
        update(tid, stage, route=manifest["route"])
        result = score(rt.root, tid, final, rt.root / "outputs" / tid / "evaluation")
        stage = "p0_analysis_evaluation"
        if sha256_bytes(p0.encode()) == sha256_bytes(final.encode()):
            p0_result = dict(result, reused_final_score=True)
        else:
            update(tid, stage)
            p0_result = score(rt.root, tid, p0, rt.root / "outputs" / tid / "p0-evaluation")
        result.update(
            task_id=tid, experiment_id=EXPERIMENT_ID, level=row["level"], route=manifest["route"],
            repair_executed=manifest.get("repair_executed"), ledger_quality=ledger_result.get("ledger_quality"),
            solver_exit_status=generation["agent_info"].get("exit_status"),
            p0_sha256=sha256_bytes(p0.encode()), final_sha256=sha256_bytes(final.encode()),
            p0_analysis={"status": p0_result.get("status"), "reward": p0_result.get("reward"),
                         "official": p0_result.get("official"),
                         "reused_final_score": p0_result.get("reused_final_score", False)},
            m0_reward=row["m0"]["reward"], integrity=integrity, finished_at=now(),
        )
        return result
    except Exception as error:  # noqa: BLE001 - archive per-task stage/infrastructure failures
        message = f"{type(error).__name__}: {error}"
        secret = os.environ.get("DEEPSEEK_API_KEY")
        if secret:
            message = message.replace(secret, "[REDACTED]")
        return {"task_id": tid, "status": "infrastructure_or_stage_error", "stage": stage, "reward": None,
                "error": message[:4000], "finished_at": now()}


# ---------------------------------------------------------------------------
# batch driver: image groups in order, three slots inside a group
# ---------------------------------------------------------------------------

def run_groups(rt, rows, worker, *, status_path, summary_path, summarize):
    lock = threading.Lock()
    states = {row["task_id"]: {"phase": "queued"} for row in rows}
    results = {}

    def update(tid, phase, **values):
        with lock:
            states[tid] = {**states[tid], "phase": phase, "updated_at": now(), **values}
            save(status_path, {"experiment_id": EXPERIMENT_ID, "workers": WORKERS, "tasks": states,
                               "updated_at": now()})

    images = []
    for row in rows:
        if row["image_name"] not in images:
            images.append(row["image_name"])
    cleanup = []
    for repository in images:
        group = [row for row in rows if row["image_name"] == repository]
        for row in group:
            update(row["task_id"], "image_preparation", image=repository)
        try:
            image_id = rt.load_image(repository)
        except Exception as error:  # noqa: BLE001 - image failures are explicit per-task outcomes
            for row in group:
                results[row["task_id"]] = {"task_id": row["task_id"], "status": "image_infrastructure_error",
                                           "error": f"{type(error).__name__}: {error}"[:3000]}
                update(row["task_id"], "image_infrastructure_error")
            continue
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(worker, row, image_id, update): row for row in group}
            for future in as_completed(futures):
                row = futures[future]
                value = future.result()
                with lock:
                    results[row["task_id"]] = value
                    save(summary_path, summarize(results, len(rows)))
                update(row["task_id"], value.get("status", "done"), reward=value.get("reward"))
        cleanup.append({"image": repository, "error": rt.remove_expanded_image(image_id)})
        save(status_path.parent / (status_path.stem + "-images.json"), cleanup)
    return results


def ledger_summary(results, total):
    return {"experiment_id": EXPERIMENT_ID, "stage": "ledger", "task_count": total,
            "terminal_count": len(results),
            "ready": sum(r.get("status") == "ledger_ready" for r in results.values()),
            "unavailable": sum(r.get("status") == "ledger_unavailable" for r in results.values()),
            "errors": sum(r.get("status") not in {"ledger_ready", "ledger_unavailable"}
                          for r in results.values()),
            "results": results, "updated_at": now()}


def run_summary(results, total):
    values = list(results.values())
    return {"experiment_id": EXPERIMENT_ID, "task_count": total, "terminal_count": len(values),
            "passed": sum(r.get("reward") == 1 for r in values),
            "valid_failed": sum(r.get("reward") == 0 for r in values),
            "p0_passed": sum((r.get("p0_analysis") or {}).get("reward") == 1 for r in values),
            "m0_passed": sum(r.get("m0_reward") == 1 for r in values if "m0_reward" in r),
            "results": values, "updated_at": now()}


def require_preflight(root, selection, rows):
    readiness = read(root / "preflight/readiness.json")
    if readiness.get("selection_sha256") != canonical_sha256(selection):
        raise ValueError("Run the zero-API preflight for this frozen selection first")
    failed = [row["task_id"] for row in rows if not readiness["tasks"].get(row["task_id"], {}).get("passed")]
    if failed:
        raise ValueError("Preflight did not pass for: " + ", ".join(failed))


def select_rows(selection, task_ids):
    rows = selection["tasks"]
    if task_ids:
        unknown = set(task_ids) - {row["task_id"] for row in rows}
        if unknown:
            raise ValueError("Unknown task IDs: " + ", ".join(sorted(unknown)))
        rows = [row for row in rows if row["task_id"] in task_ids]
    return rows


def mode_preflight(root, selection, rows):
    rt = Runtime(root)
    path = root / "preflight/readiness.json"
    previous = read(path) if path.exists() else {}
    tasks = dict(previous.get("tasks", {})) if previous.get("selection_sha256") == canonical_sha256(selection) else {}

    def worker(row, image_id, update):
        update(row["task_id"], "preflight")
        return preflight_task(rt, row, image_id)

    def summarize(results, total):
        return {"results": results, "updated_at": now()}

    results = run_groups(rt, rows, worker, status_path=root / "preflight/monitor-status.json",
                         summary_path=root / "preflight/progress.json", summarize=summarize)
    tasks.update(results)
    save(path, {"experiment_id": EXPERIMENT_ID, "selection_sha256": canonical_sha256(selection),
                "model_api_calls": 0, "tasks": tasks,
                "passed": len(tasks) == TASK_COUNT and all(v.get("passed") for v in tasks.values()),
                "finished_at": now()})


def mode_ledger(root, selection, rows):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    require_preflight(root, selection, rows)
    rt = Runtime(root)

    def worker(row, image_id, update):
        try:
            return ledger_stage(rt, row, image_id, update)
        except Exception as error:  # noqa: BLE001 - no ledger is frozen; record and continue
            message = f"{type(error).__name__}: {error}"
            secret = os.environ.get("DEEPSEEK_API_KEY")
            if secret:
                message = message.replace(secret, "[REDACTED]")
            value = {"task_id": row["task_id"], "status": "infrastructure_or_stage_error",
                     "error": message[:4000], "finished_at": now()}
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            save(root / "outputs" / row["task_id"] / "ledger/errors" / f"{stamp}.json", value)
            return value

    run_groups(rt, rows, worker, status_path=root / "ledger-monitor-status.json",
               summary_path=root / "ledger-progress.json", summarize=ledger_summary)
    results = {}
    for row in selection["tasks"]:
        _, result_path = ledger_paths(root, row["task_id"])
        if result_path.exists():
            results[row["task_id"]] = read(result_path)
    save(root / "ledger-summary.json", ledger_summary(results, TASK_COUNT))


def mode_preflight_solver(root, selection):
    """Native Solver protocol with a real frozen ledger and simulated model responses."""
    rt = Runtime(root)
    row = next(r for r in selection["tasks"] if load_ledger(root, r["task_id"])[0] is not None)
    tid = row["task_id"]
    ledger, _ = load_ledger(root, tid)
    task = task_input(root, tid)
    statement = augment_statement(task["problem_statement"], ledger)
    output = root / "preflight/solver-protocol" / tid
    image_id = rt.load_image(row["image_name"])
    environment = client = None
    try:
        environment, container, audit, client = rt.open_prepared(task, image_id, output)
        check_tree(audit, row["m0_tree_sha"], "Solver preflight")
        protocol = rt.native.offline_probe(rt.config, environment, output, {"instruction": statement})
    finally:
        if client is not None:
            rt.close(environment, client)
    trajectory = read(output / "native.traj.json")
    user = next(m["content"] for m in trajectory["messages"] if m["role"] == "user")
    if isinstance(user, list):
        user = "".join(part.get("text", "") for part in user if isinstance(part, dict))
    passed = (task["problem_statement"] in user and user.count(HEADING) == 1 and user.count(GUIDANCE) == 1
              and trajectory["info"]["exit_status"] == "Submitted")
    save(root / "preflight/solver-protocol.json", {
        "passed": passed, "task_id": tid, "model_api_calls": 0, "offline_protocol": protocol,
        "actual_native_user_message_verified": passed, "finished_at": now(),
    })
    if not passed:
        raise ValueError("Native Mini did not receive exactly one ledger supplement")


def mode_run(root, selection, rows, start):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Supply DEEPSEEK_API_KEY through the host process environment")
    require_preflight(root, selection, rows)
    if start == "solver":
        missing = [row["task_id"] for row in selection["tasks"] if not ledger_paths(root, row["task_id"])[1].exists()]
        if missing:
            raise ValueError("Ledger stage incomplete for: " + ", ".join(missing))
        protocol = root / "preflight/solver-protocol.json"
        if not protocol.exists() or not read(protocol)["passed"]:
            raise ValueError("Run preflight-solver first")
    rt = Runtime(root)

    def worker(row, image_id, update):
        path = root / "outputs" / row["task_id"] / "result.json"
        if path.exists():
            return read(path)
        result = run_case(rt, row, image_id, start, update)
        save(path, result)
        return result

    run_groups(rt, rows, worker, status_path=root / "monitor-status.json", summary_path=root / "result.json",
               summarize=run_summary)
    save(root / "launcher-result.json", {"completed": True, "finished_at": now()})


def mode_launch(root, argv):
    """Read the key from stdin, verify it, and start one entry detached; nothing is persisted."""
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("A transient API credential is required on stdin")
    request = urllib.request.Request("https://api.deepseek.com/v1/models", headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, models = response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit("Model authentication failed: HTTP " + str(error.code)) from None
    if not any(model.get("id") == "deepseek-flash" for model in models.get("data", [])):
        raise SystemExit("The approved DeepSeek model alias is unavailable")
    environment = os.environ.copy()
    environment.update(DEEPSEEK_API_KEY=key, DOCKER_HOST=DOCKER_HOST,
                       PYTHONPATH=os.pathsep.join([str(PROJECT), OFFICIAL_SOURCE]),
                       PYTHONDONTWRITEBYTECODE="1")
    command = [NATIVE_PYTHON, str(Path(__file__).resolve()), *argv, "--root", str(root)]
    name = "-".join(argv).replace("--", "")
    (root / "logs").mkdir(exist_ok=True)
    with (root / "logs" / f"{name}.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"experiment_id": EXPERIMENT_ID, "entry": argv, "pid": process.pid, "started_at": now(),
               "command": command, "credentials_persisted": False,
               "authentication_preflight_http_status": status}
    save(root / f"launch-{name}.json", receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["prepare", "preflight", "ledger", "preflight-solver", "run", "launch"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--from", dest="start", choices=["A", "solver"], default="solver")
    parser.add_argument("--entry", nargs=argparse.REMAINDER, help="launch: the entry and its options")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        selection = prepare(root)
        print(json.dumps({"mode": "prepare", "task_count": selection["task_count"]}))
        return
    if args.mode == "launch":
        if not args.entry or args.entry[0] not in {"ledger", "run"}:
            parser.error("launch requires --entry ledger|run [options]")
        validate(root)
        mode_launch(root, args.entry)
        return
    selection = validate(root)
    rows = select_rows(selection, set(args.task_id or []))
    if args.mode == "preflight":
        mode_preflight(root, selection, rows)
    elif args.mode == "ledger":
        mode_ledger(root, selection, rows)
    elif args.mode == "preflight-solver":
        mode_preflight_solver(root, selection)
    else:
        mode_run(root, selection, rows, args.start)
    print(json.dumps({"mode": args.mode, "task_count": len(rows), "experiment_id": EXPERIMENT_ID}))


if __name__ == "__main__":
    main()
