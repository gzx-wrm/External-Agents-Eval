"""Fixed DeepSWE30: CodeScout -> unchanged native Mini -> official verifier."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SCRIPT.parent))

import run as preprocessing

from external_agents_eval.preprocessors.codescout.artifacts import ArtifactStore, sha256_json
from external_agents_eval.preprocessors.codescout.index import build_index
from external_agents_eval.preprocessors.codescout.pipeline import (
    Config,
    Preprocessor,
    validate_synthesis,
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(root, name, value):
    ArtifactStore(root, secrets=(os.environ.get("DEEPSEEK_API_KEY", ""),)).write_json(name, value)


def tree_hashes(folder):
    return {str(path.relative_to(folder)): sha(path) for path in sorted(Path(folder).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def implementation_hashes():
    paths = [*SCRIPT.parent.glob("*.py"),
             *(PROJECT / "external_agents_eval/preprocessors/codescout").glob("*.py")]
    return {str(path.relative_to(PROJECT)): sha(path) for path in sorted(paths)}


def load_native(root):
    if importlib.metadata.version("mini-swe-agent") != "2.4.6":
        raise ValueError("This experiment requires the frozen native Mini 2.4.6 runtime")
    return preprocessing.load_native(root)


def prepare(root, sources, task_ids=(), reuse_root=None, config_file=None):
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new empty experiment root")
    root.mkdir(parents=True, exist_ok=True)
    reference, candidates, common = None, {}, None
    names = ["m0_run.py", "native_compat.py", "sanitize_repository.py"]
    for source in sources:
        native = load_native(source)
        tasks = native.load_tasks(source)
        suite = read(source / "manifests/eval30_current.json")
        hashes = {name: sha(source / "scripts" / name) for name in names}
        hashes["native-config"] = sha(source / "configs/native-deepseek-v4.1-flash.yaml")
        if common is not None and common != hashes:
            raise ValueError("Native Mini code/configuration differs between source bundles")
        common = hashes
        if reference is None:
            reference = source
            original_tasks = {task["task_id"]: task for task in tasks}
            canonical = suite["task_ids"]
        elif canonical != suite["task_ids"] or original_tasks != {
                task["task_id"]: task for task in tasks}:
            raise ValueError("Source bundles differ in original tasks")
        selection = read(source / "selection.json")
        for record in selection["tasks"]:
            tid = record["task_id"]
            if tid in candidates:
                raise ValueError("Source selections overlap: " + tid)
            if record["task_checksum"] != original_tasks[tid]["task_checksum"]:
                raise ValueError("Source task identity differs: " + tid)
            candidates[tid] = {"task_id": tid, "task_checksum": record["task_checksum"],
                               "m0_image": record["m0_image"], "source_bundle": str(source)}
    if (len(canonical) != 30 or set(candidates) != set(canonical) or
            "sqlite-utils-safe-import-checkpoints" in canonical):
        raise ValueError("Both source selections must cover exactly the current frozen 30 tasks")
    chosen = set(task_ids) if task_ids else set(canonical)
    if not chosen <= set(canonical):
        raise ValueError("Requested task is outside the frozen current30")
    parent = Path(read(reference / "selection.json")["m0_root"])
    official = parent / "preflight/readiness.json"
    if not read(official)["passed"]:
        raise ValueError("Original official gold/control checks are not valid")
    (root / "scripts").mkdir()
    (root / "configs").mkdir()
    shutil.copytree(reference / "manifests", root / "manifests")
    (root / "grader").symlink_to((reference / "grader").resolve(), target_is_directory=True)
    for name in names:
        shutil.copyfile(reference / "scripts" / name, root / "scripts" / name)
    for name in ["native-deepseek-v4.1-flash.yaml", "upstream-mini.yaml"]:
        shutil.copyfile(reference / "configs" / name, root / "configs" / name)
    config_file = config_file or PROJECT / "configs/codescout-deepseek-v4.1-flash.json"
    config = Config(**read(config_file))
    save(root, "configs/codescout.json", asdict(config))
    records = []
    for tid in canonical:
        if tid not in chosen:
            continue
        record = candidates[tid]
        record["preprocessing_reused"] = False
        if reuse_root and (reuse_root / tid / "codescout/result.json").exists():
            source = reuse_root / tid
            result, manifest = read(source / "codescout/result.json"), read(source / "codescout/manifest.json")
            snapshot = read(source / "snapshot.json")
            issue = original_tasks[tid]["instruction"]
            if (result["status"] != "completed" or manifest["config"] != asdict(config)
                    or snapshot["image_id"] != record["m0_image"]["id"]
                    or snapshot["task_checksum"] != record["task_checksum"]
                    or manifest["issue_sha256"] != hashlib.sha256(issue.encode()).hexdigest()
                    or read(source / "original-task.json") != original_tasks[tid]):
                raise ValueError("Frozen preprocessing does not match the selected baseline: " + tid)
            validate_synthesis((source / "codescout/enhanced-issue.md").read_text())
            destination = root / "outputs" / tid / "preprocessing" / "codescout"
            shutil.copytree(source / "codescout", destination)
            shutil.copyfile(destination / "result.json", destination / "original-generation-result.json")
            record.update(preprocessing_reused=True, preprocessing_source=str(source),
                          imported_cache_sha256=tree_hashes(destination))
        records.append(record)
    selection = {
        "experiment_id": "codescout_native_mini", "task_count": len(records), "tasks": records,
        "m0_root": str(parent), "canonical_task_ids": canonical,
        "native_sha256": common, "code_sha256": implementation_hashes(),
        "config_sha256": {name: sha(root / "configs" / name) for name in
                          ["native-deepseek-v4.1-flash.yaml", "upstream-mini.yaml", "codescout.json"]},
        "official_preflight": str(official), "official_preflight_sha256": sha(official),
        "workers": 3, "input_change": "CodeScout enhanced problem statement only",
        "ft_ledger_or_review_included": False, "original_solver_patch_or_conversation_reused": False,
    }
    save(root, "selection.json", selection)
    return selection


def validate(root):
    selection = read(root / "selection.json")
    for name, expected in selection["code_sha256"].items():
        if sha(PROJECT / name) != expected:
            raise ValueError("Experiment implementation changed: " + name)
    for name, expected in selection["config_sha256"].items():
        if sha(root / "configs" / name) != expected:
            raise ValueError("Frozen configuration changed: " + name)
    for name in ["m0_run.py", "native_compat.py", "sanitize_repository.py"]:
        if sha(root / "scripts" / name) != selection["native_sha256"][name]:
            raise ValueError("Frozen native runner changed: " + name)
    if (sha(selection["official_preflight"]) != selection["official_preflight_sha256"] or
            not read(selection["official_preflight"])["passed"]):
        raise ValueError("Official scorer preflight reference changed")
    native = load_native(root)
    tasks = {task["task_id"]: task for task in native.load_tasks(root)}
    for record in selection["tasks"]:
        if tasks[record["task_id"]]["task_checksum"] != record["task_checksum"]:
            raise ValueError("Task checksum changed")
    return selection, native, tasks


class ReuseOnly:
    backend = "openai-compatible-http"

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Frozen preprocessing cache is incomplete; refusing a new sample")


def preprocess_case(root, tid):
    selection, native, tasks = validate(root)
    record = next(item for item in selection["tasks"] if item["task_id"] == tid)
    task, output = tasks[tid], root / "outputs" / tid
    folder = output / "preprocessing"
    folder.mkdir(parents=True, exist_ok=True)
    config = Config(**read(root / "configs/codescout.json"))
    source = preprocessing.snapshot(root, task, record, folder, native)
    index = build_index(source, max_file_bytes=config.max_file_bytes)
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if record["preprocessing_reused"]:
        client = ReuseOnly()
        for name, expected in record["imported_cache_sha256"].items():
            path = folder / "codescout" / name
            # These outputs are rewritten by an identical cached replay; raw
            # request/response bytes remain the immutable reuse boundary.
            if (name.endswith(("request.json", "response.json", "output.txt")) and
                    sha(path) != expected):
                raise ValueError("Imported raw preprocessing cache changed")
    else:
        if not key:
            raise ValueError("Inject DEEPSEEK_API_KEY into the host process environment")
        client = preprocessing.ChatClient(config, key)
    enhanced = Preprocessor(config, client, folder / "codescout", secrets=(key,)).run(
        task["instruction"], index)
    augmented = preprocessing.augment_task(task, enhanced)
    augmented["image"] = record["m0_image"]["repo_digests"][0]
    save(root, f"outputs/{tid}/augmented-task.json", augmented)
    save(root, f"outputs/{tid}/input-audit.json", {
        "task_checksum": task["task_checksum"], "image_id": record["m0_image"]["id"],
        "original_instruction_sha256": hashlib.sha256(task["instruction"].encode()).hexdigest(),
        "enhanced_instruction_sha256": hashlib.sha256(enhanced.encode()).hexdigest(),
        "repository_sha256": index.fingerprint, "preprocessing_reused": record["preprocessing_reused"],
        "ft_ledger_or_review_included": False, "solver_patch_or_conversation_reused": False,
    })
    return augmented


def preflight(root):
    selection, native, _tasks = validate(root)
    record = next((item for item in selection["tasks"] if item["preprocessing_reused"]), None)
    if record is None:
        record = next((item for item in selection["tasks"] if
                       (root / "outputs" / item["task_id"] / "augmented-task.json").exists()), None)
    if record is None:
        raise ValueError("Preprocess one selected task before running preflight")
    tid = record["task_id"]
    task = preprocess_case(root, tid)
    output = root / "preflight" / tid
    output.mkdir(parents=True, exist_ok=True)
    client = native.docker.from_env(timeout=1800)
    import yaml

    config = yaml.safe_load((root / "configs/native-deepseek-v4.1-flash.yaml").read_text())
    try:
        image, _ = native.get_image(client, task, output)
        if image.id != record["m0_image"]["id"]:
            raise ValueError("Preflight image differs from M0")
        _, protocol = native.generate(task, root, output / "offline", config, image, "offline")
    finally:
        client.close()
    messages = read(output / "offline/native.traj.json")["messages"]
    user = next(message["content"] for message in messages if message["role"] == "user")
    if task["instruction"] not in user:
        raise ValueError("Native Mini did not receive the exact enhanced problem statement")
    save(root, "preflight/readiness.json", {
        "passed": True, "selection_sha256": sha256_json(selection), "task_id": tid,
        "native_offline_protocol": protocol, "enhanced_input_verified": True,
        "new_model_api_calls": 0, "official_gold_control_reference": selection["official_preflight"],
        "finished_at": now(),
    })


def solver_usage(path):
    total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0}
    for message in read(path).get("messages", []):
        usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
        total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["completion_tokens"] += usage.get("completion_tokens", 0)
        total["cached_prompt_tokens"] += usage.get("prompt_cache_hit_tokens", (
            usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    return total


def run_one(root, record, task, native, config, update):
    tid, phase = task["task_id"], "image_preparation"
    output = root / "outputs" / tid
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / "result.json"
    if result_file.exists() and read(result_file).get("status") == "scored":
        result = read(result_file)
        update(tid, "scored", reward=result["reward"], reused_existing_outcome=True)
        return result
    client, added = None, False
    pinned = {**task, "image": record["m0_image"]["repo_digests"][0]}
    started = time.monotonic()
    try:
        client = native.docker.from_env(timeout=1800)
        update(tid, phase)
        image, added = native.get_image(client, pinned, output)
        if image.id != record["m0_image"]["id"]:
            raise ValueError("Task image differs from M0")
        phase = "codescout_preprocessing"
        update(tid, phase, reused=record["preprocessing_reused"])
        log = root / "logs" / (tid + ".preprocessing.log")
        log.parent.mkdir(exist_ok=True)
        with log.open("ab") as stream:
            subprocess.run([sys.executable, str(SCRIPT), "preprocess", "--root", str(root),
                            "--task-id", tid], check=True, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=Config(**read(root / "configs/codescout.json")).max_seconds)
        augmented = read(output / "augmented-task.json")
        phase = "solving"
        generation = output / "generation"
        info_file, patch_file = generation / "agent-info.json", generation / "model.patch"
        update(tid, phase)
        if info_file.exists() and patch_file.exists():
            info, patch = read(info_file), patch_file.read_text()
        else:
            sentinel = output / "generation-started.json"
            if sentinel.exists():
                raise RuntimeError("Interrupted Solver invocation; do not automatically resample")
            save(root, str(sentinel.relative_to(root)), {"started_at": now(), "image_id": image.id})
            patch, info = native.generate(augmented, root, generation, config, image, "generate")
        phase = "official_evaluation"
        update(tid, phase, exit_status=info.get("exit_status"))
        result = native.grade(root, pinned, image.id, patch, output / "evaluation")
        pp = read(output / "preprocessing/codescout/result.json")
        result.update(
            exit_status=info.get("exit_status"), native_agent_completed=info.get("exit_status") == "Submitted",
            patch_sha256=hashlib.sha256(patch.encode()).hexdigest(), preprocessing={
                key: pp[key] for key in ["targets", "analyzed", "retained", "model_calls_in_protocol",
                                       "prompt_tokens", "completion_tokens", "new_prompt_tokens",
                                       "new_completion_tokens", "new_request_attempts"]},
            preprocessing_reused=record["preprocessing_reused"],
            solver_usage=solver_usage(generation / "native.traj.json"),
            elapsed_seconds=round(time.monotonic() - started, 3), finished_at=now(),
        )
    except Exception as error:  # noqa: BLE001 - preserve explicit per-case stage failures
        result = {"status": "infrastructure_or_stage_error", "task_id": tid, "reward": None,
                  "failed_phase": phase, "error_type": type(error).__name__,
                  "error": ArtifactStore(root, secrets=(os.environ.get("DEEPSEEK_API_KEY", ""),)).redact(str(error)),
                  "finished_at": now()}
    finally:
        if added:
            try:
                client.images.remove(pinned["image"], noprune=False)
            except native.docker.errors.APIError:
                pass
        if client is not None:
            client.close()
    save(root, f"outputs/{tid}/result.json", result)
    update(tid, result["status"], reward=result.get("reward"), failed_phase=result.get("failed_phase"))
    return result


def run_batch(root, workers=3):
    selection, native, tasks = validate(root)
    ready = read(root / "preflight/readiness.json")
    if not ready["passed"] or ready["selection_sha256"] != sha256_json(selection):
        raise ValueError("Preflight does not match the frozen experiment selection")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Inject DEEPSEEK_API_KEY into the host process environment")
    import yaml

    config = yaml.safe_load((root / "configs/native-deepseek-v4.1-flash.yaml").read_text())
    states = {record["task_id"]: {"phase": "queued"} for record in selection["tasks"]}
    lock = threading.Lock()

    def update(tid, phase, **values):
        with lock:
            states[tid].update(phase=phase, updated_at=now(), **values)
            save(root, "monitor-status.json", {"workers": workers, "task_count": len(states),
                                                "tasks": states, "updated_at": now()})
        print(json.dumps({"task_id": tid, "phase": phase, **values}), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, root, record, tasks[record["task_id"]], native, config, update)
                   for record in selection["tasks"]]
        results = [future.result() for future in as_completed(futures)]
    save(root, "result.json", {"task_count": len(results), "results": results,
                               "passed": sum(result.get("reward") == 1 for result in results),
                               "valid_failed": sum(result.get("reward") == 0 for result in results),
                               "unscored": sum(result.get("reward") is None for result in results),
                               "finished_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preprocess", "preflight", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--reuse-preprocessing", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers != 3:
        parser.error("Keep the frozen comparison's three pipeline slots")
    root = args.root.resolve()
    if args.mode == "prepare":
        if not args.source:
            parser.error("Prepare requires both frozen --source bundles")
        prepare(root, [path.resolve() for path in args.source], args.task_id,
                args.reuse_preprocessing.resolve() if args.reuse_preprocessing else None, args.config)
    elif args.mode == "preprocess":
        if len(args.task_id) != 1:
            parser.error("Preprocess requires exactly one --task-id")
        preprocess_case(root, args.task_id[0])
    elif args.mode == "preflight":
        preflight(root)
    else:
        run_batch(root, args.workers)


if __name__ == "__main__":
    main()
