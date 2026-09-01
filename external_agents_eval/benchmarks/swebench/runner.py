from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

import docker
from external_agents_eval.agents import HermesAdapter, PiAdapter
from external_agents_eval.agents.base import AgentRunContext
from external_agents_eval.agents.config import DEFAULT_MODEL, get_agent_spec
from external_agents_eval.benchmarks.swebench.manifest import (
    TaskManifest,
    atomic_write_json,
    ensure_result_manifest,
)
from external_agents_eval.benchmarks.swebench.task_images import (
    TaskImageRecord,
    audit_task_images,
    ensure_task_image_audit,
)
from external_agents_eval.docker_io import (
    RuntimeArchiveCache,
    decode_output,
    disk_free_bytes,
    put_files,
    redact_secret,
    write_log,
)
from external_agents_eval.failures import classify_failure


@dataclass(frozen=True)
class EvaluationPaths:
    root: Path
    predictions: Path
    metadata: Path
    logs: Path
    trajectories: Path

    @classmethod
    def create(cls, results_dir: Path, agent_name: str) -> EvaluationPaths:
        root = results_dir.expanduser().resolve() / agent_name
        paths = cls(
            root=root,
            predictions=root / "predictions",
            metadata=root / "metadata",
            logs=root / "logs",
            trajectories=root / "trajectories",
        )
        for path in paths.__dict__.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        return paths


def default_agent_name(agent_id: str, model: str = DEFAULT_MODEL) -> str:
    model_slug = model.replace("/", "_").replace("-", "_")
    return f"{agent_id}_{model_slug}_random50_seed20260818"


def build_prompt(entry: dict[str, Any]) -> str:
    return (
        "Work on the checked-out repository in /testbed and solve the SWE-bench issue below.\n"
        "Inspect the repository, edit the implementation directly, and run relevant tests when useful.\n"
        "Do not modify tests or benchmark metadata. Do not inspect or print environment secrets.\n"
        "Leave the final solution as changes in the working tree; the harness will collect git diff.\n\n"
        f"Instance: {entry['instance_id']}\n"
        f"Base commit: {entry['base_commit']}\n\n"
        "Problem statement:\n"
        f"{entry['problem_statement']}"
    )


def load_entries(manifest: TaskManifest) -> list[dict[str, Any]]:
    dataset = load_dataset(manifest.dataset_name)["test"]
    by_id = {entry["instance_id"]: entry for entry in dataset}
    missing = [instance_id for instance_id in manifest.task_ids if instance_id not in by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} manifest tasks are missing from {manifest.dataset_name}: "
            + ", ".join(missing[:5])
        )
    return [by_id[instance_id] for instance_id in manifest.task_ids]


def _prediction_complete(path: Path, instance_id: str) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("instance_id") == instance_id
        and payload.get("generation_status") == "success"
        and isinstance(payload.get("model_patch"), str)
        and isinstance(payload.get("model_name_or_path"), str)
    )


def _task_logger(path: Path, instance_id: str) -> logging.Logger:
    logger = logging.getLogger(f"external_agents_eval.{instance_id}.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _exec_logged(container, command, log_path: Path, *, workdir: str = "/", secret: str):
    result = container.exec_run(command, workdir=workdir)
    output = redact_secret(decode_output(result.output), secret)
    write_log(log_path, [f"$ {command!r}", output], secret=secret)
    return result, output


def _prepare_repository(container, entry: dict[str, Any], log_path: Path, secret: str) -> None:
    head_result, head = _exec_logged(
        container,
        ["git", "rev-parse", "HEAD"],
        log_path,
        workdir="/testbed",
        secret=secret,
    )
    if head_result.exit_code != 0:
        raise RuntimeError("Cannot read task repository HEAD")
    if head.strip() != entry["base_commit"]:
        raise RuntimeError(
            f"Task image HEAD {head.strip()} does not match base_commit {entry['base_commit']}"
        )
    reset_result, _ = _exec_logged(
        container,
        ["git", "reset", "--hard", entry["base_commit"]],
        log_path,
        workdir="/testbed",
        secret=secret,
    )
    if reset_result.exit_code != 0:
        raise RuntimeError("Cannot reset disposable task repository to its base commit")
    clean_result, _ = _exec_logged(
        container,
        ["git", "clean", "-fd"],
        log_path,
        workdir="/testbed",
        secret=secret,
    )
    if clean_result.exit_code != 0:
        raise RuntimeError("Cannot clean disposable task repository")


def _collect_patch(container, base_commit: str, log_path: Path, secret: str) -> str:
    # Intent-to-add makes ordinary untracked source files visible to git diff.
    add_result, add_output = _exec_logged(
        container,
        ["git", "add", "-N", "--", "."],
        log_path,
        workdir="/testbed",
        secret=secret,
    )
    if add_result.exit_code != 0:
        write_log(log_path, [f"git add -N warning: {add_output}"], secret=secret)
    result = container.exec_run(
        ["git", "diff", "--binary", "--no-ext-diff", base_commit, "--"],
        workdir="/testbed",
    )
    patch = redact_secret(decode_output(result.output), secret)
    if result.exit_code != 0:
        raise RuntimeError(f"git diff failed with status {result.exit_code}")
    write_log(log_path, [f"Collected model patch ({len(patch)} bytes)"], secret=secret)
    return patch


def _write_failure(
    prediction_path: Path,
    metadata_path: Path,
    *,
    instance_id: str,
    model_name_or_path: str,
    error: str,
    metadata: dict[str, Any],
) -> None:
    kind = classify_failure(error)
    atomic_write_json(
        prediction_path,
        {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": "",
            "proposed_model_patches": [],
            "generation_status": "failed",
            "generation_error": error,
            "failure_kind": kind.value,
        },
    )
    metadata.update(
        {
            "generation_status": "failed",
            "generation_error": error,
            "failure_kind": kind.value,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    atomic_write_json(metadata_path, metadata)


def _create_task_container(client, test_spec, task_image: TaskImageRecord, run_id: str):
    """Create directly from the audited immutable image ID; never build or pull."""
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    if test_spec.instance_id != task_image.instance_id:
        raise RuntimeError(
            f"Task image audit mismatch: {test_spec.instance_id} != {task_image.instance_id}"
        )
    if test_spec.instance_image_key != task_image.image_name:
        raise RuntimeError(
            f"Task image audit mismatch: {test_spec.instance_image_key} != {task_image.image_name}"
        )
    # Resolving by ID guarantees that a mutable :latest tag cannot drift during a run.
    image = client.images.get(task_image.image_id)
    if image.id != task_image.image_id:
        raise RuntimeError(
            f"Audited task image {task_image.image_id} is no longer available locally"
        )
    config = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]
    user = "root" if not config.get("execute_test_as_nonroot", False) else "nonroot"
    return client.containers.create(
        image=task_image.image_id,
        name=test_spec.get_instance_container_name(run_id),
        user=user,
        detach=True,
        command="tail -f /dev/null",
        nano_cpus=config.get("nano_cpus"),
        platform=task_image.platform,
    )


def _process_entry(
    entry: dict[str, Any],
    *,
    adapter,
    paths: EvaluationPaths,
    task_image: TaskImageRecord,
    runtime_archive: Path,
    runtime_image_name: str,
    runtime_image_id: str,
    model: str,
    timeout_seconds: int,
    api_key: str,
    resume: bool,
) -> dict[str, Any]:
    instance_id = entry["instance_id"]
    prediction_path = paths.predictions / f"{instance_id}.json"
    metadata_path = paths.metadata / f"{instance_id}.json"
    log_path = paths.logs / f"{instance_id}_docker.log"
    trajectory_path = paths.trajectories / f"{instance_id}{adapter.transcript_suffix}"
    if prediction_path.exists() and _prediction_complete(prediction_path, instance_id):
        return {"instance_id": instance_id, "status": "skipped"}
    if prediction_path.exists() and not resume:
        return {
            "instance_id": instance_id,
            "status": "blocked",
            "error": "Incomplete result exists; rerun with --resume",
        }

    spec = adapter.spec
    model_name_or_path = paths.root.name
    metadata = {
        "instance_id": instance_id,
        "agent_id": spec.agent_id,
        "model": model,
        "upstream_url": spec.upstream_url,
        "upstream_ref": spec.upstream_ref,
        "upstream_commit": spec.upstream_commit,
        "runtime_image": runtime_image_name,
        "runtime_image_id": runtime_image_id,
        "task_image": task_image.as_json(),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    client = docker.from_env()
    container = None
    logger = _task_logger(log_path, instance_id)
    started = time.monotonic()
    try:
        from swebench.harness.test_spec import make_test_spec

        test_spec = make_test_spec(entry)
        run_id = dt.datetime.now(dt.timezone.utc).strftime("external_%Y%m%d_%H%M%S_%f")
        container = _create_task_container(client, test_spec, task_image, run_id)
        container.start()
        write_log(log_path, [f"Started task container {container.name}"], secret=api_key)

        put_files(container, "/", {"external_eval/eval.sh": test_spec.eval_script})
        setup_result, _setup_output = _exec_logged(
            container,
            ["/bin/bash", "/external_eval/eval.sh"],
            log_path,
            secret=api_key,
        )
        metadata["environment_setup_exit_code"] = setup_result.exit_code
        if setup_result.exit_code != 0:
            write_log(
                log_path,
                ["Environment setup/test probe exited non-zero; continuing as HGM does."],
                secret=api_key,
            )
        _prepare_repository(container, entry, log_path, api_key)

        RuntimeArchiveCache.install(container, runtime_archive, destination="/opt")
        write_log(log_path, [f"Installed {spec.agent_id} runtime"], secret=api_key)
        context = AgentRunContext(
            instance_id=instance_id,
            prompt=build_prompt(entry),
            model=model,
            timeout_seconds=timeout_seconds,
            api_key=api_key,
        )
        run_result = adapter.run(container, context)
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        temp_trajectory = trajectory_path.with_suffix(trajectory_path.suffix + ".tmp")
        temp_trajectory.write_text(redact_secret(run_result.transcript, api_key))
        temp_trajectory.replace(trajectory_path)
        write_log(
            log_path,
            [f"Agent process exited with status {run_result.exit_code}"],
            secret=api_key,
        )
        if run_result.exit_code != 0:
            raise RuntimeError(f"Agent process exited with status {run_result.exit_code}")

        patch = _collect_patch(container, entry["base_commit"], log_path, api_key)
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": patch,
            "proposed_model_patches": [],
            "generation_status": "success",
        }
        atomic_write_json(prediction_path, prediction)
        metadata.update(run_result.metadata)
        metadata.update(
            {
                "generation_status": "success",
                "empty_patch": not bool(patch.strip()),
                "patch_bytes": len(patch.encode()),
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "duration_seconds": round(time.monotonic() - started, 3),
                "trajectory_path": str(trajectory_path),
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        atomic_write_json(metadata_path, metadata)
        return {
            "instance_id": instance_id,
            "status": "success",
            "empty_patch": not bool(patch.strip()),
        }
    except Exception as error:  # noqa: BLE001 - per-task failure boundary
        error_text = redact_secret(str(error), api_key)
        write_log(log_path, [f"ERROR: {error_text}"], secret=api_key)
        metadata["duration_seconds"] = round(time.monotonic() - started, 3)
        _write_failure(
            prediction_path,
            metadata_path,
            instance_id=instance_id,
            model_name_or_path=model_name_or_path,
            error=error_text,
            metadata=metadata,
        )
        return {"instance_id": instance_id, "status": "failed", "error": error_text}
    finally:
        if container is not None:
            try:
                from swebench.harness.docker_build import cleanup_container

                cleanup_container(client, container, logger)
            except Exception as cleanup_error:  # noqa: BLE001 - best-effort cleanup
                write_log(
                    log_path,
                    [f"Container cleanup warning: {cleanup_error}"],
                    secret=api_key,
                )
        _close_logger(logger)
        client.close()


def generate(
    *,
    agent_id: str,
    manifest: TaskManifest,
    results_dir: Path,
    agent_name: str | None,
    model: str,
    num_workers: int,
    hours_per_task: int,
    max_turns: int,
    resume: bool,
) -> dict[str, int]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Use scripts/run_with_deepseek_key.sh so the key "
            "is entered without appearing in shell history."
        )
    if num_workers <= 0 or hours_per_task <= 0 or max_turns <= 0:
        raise ValueError("num_workers, hours_per_task, and max_turns must be positive")
    spec = get_agent_spec(agent_id)
    resolved_name = agent_name or default_agent_name(agent_id, model)
    paths = EvaluationPaths.create(results_dir, resolved_name)
    ensure_result_manifest(paths.root / "task_manifest.json", manifest)
    entries = load_entries(manifest)

    client = docker.from_env()
    try:
        task_images = audit_task_images(client, entries)
        ensure_task_image_audit(paths.root / "task_image_audit.json", task_images)
        task_images_by_id = {record.instance_id: record for record in task_images}
        archive_cache = RuntimeArchiveCache(Path(".runtime_cache").resolve())
        runtime_details: dict[str, tuple[Path, str, str]] = {}
        for architecture in sorted({record.docker_architecture for record in task_images}):
            runtime_image_name = spec.runtime_image(architecture)
            runtime_image = client.images.get(runtime_image_name)
            runtime_archive = archive_cache.ensure(
                client,
                runtime_image_name,
                spec.runtime_root,
                expected_architecture=architecture,
            )
            runtime_details[architecture] = (
                runtime_archive,
                runtime_image_name,
                runtime_image.id,
            )
    finally:
        client.close()

    before_free = disk_free_bytes(paths.root)
    adapter = PiAdapter() if agent_id == "pi" else HermesAdapter(max_turns=max_turns)
    counts = {"success": 0, "failed": 0, "skipped": 0, "blocked": 0, "empty_patch": 0}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _process_entry,
                entry,
                adapter=adapter,
                paths=paths,
                task_image=task_images_by_id[entry["instance_id"]],
                runtime_archive=runtime_details[
                    task_images_by_id[entry["instance_id"]].docker_architecture
                ][0],
                runtime_image_name=runtime_details[
                    task_images_by_id[entry["instance_id"]].docker_architecture
                ][1],
                runtime_image_id=runtime_details[
                    task_images_by_id[entry["instance_id"]].docker_architecture
                ][2],
                model=model,
                timeout_seconds=hours_per_task * 3600,
                api_key=api_key,
                resume=resume,
            ): entry["instance_id"]
            for entry in entries
        }
        for future in as_completed(futures):
            result = future.result()
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1
            if result.get("empty_patch"):
                counts["empty_patch"] += 1
            print(f"[{status}] {result['instance_id']}", flush=True)

    after_free = disk_free_bytes(paths.root)
    summary = {
        **counts,
        "task_count": len(entries),
        "disk_free_before_bytes": before_free,
        "disk_free_after_bytes": after_free,
        "disk_delta_bytes": after_free - before_free,
    }
    atomic_write_json(paths.root / "generation_summary.json", summary)
    return counts
