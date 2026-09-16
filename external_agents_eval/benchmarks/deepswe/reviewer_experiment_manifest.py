from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from external_agents_eval.benchmarks.deepswe.config import DEEPSWE_COMMIT, DEEPSWE_REVISION
from external_agents_eval.benchmarks.swebench.manifest import atomic_write_json
from external_agents_eval.workflows.solve_review_repair.models import ExistingPatch

DEVELOPMENT_TASK_IDS = (
    "abs-stepped-slices",
    "effect-sse-httpapi-streaming",
)

EVALUATION_TASK_IDS = (
    "clack-async-autocomplete-options",
    "fastapi-deprecation-response-headers",
    "fastapi-implicit-head-options",
    "kombu-single-active-consumer-priority",
    "tomlkit-toml-table-converters",
    "valibot-recursive-schema-composition",
    "ytt-jsonpath-query-api",
    "bandit-interprocedural-taint-checks",
    "cattrs-partial-structuring-recovery",
    "expr-try-catch-errors",
    "httpx-deterministic-cookie-store",
    "obsidian-linter-link-format-conversion",
    "returns-validated-error-accumulation",
    "tengo-callable-instance-isolation",
    "koota-pair-relation-tracking",
    "opa-template-string-reconstruction",
)


@dataclass(frozen=True)
class ReviewerExperimentManifest:
    path: Path
    source_job_id: str
    source_job_dir: Path
    source_manifest_path: Path
    source_manifest_sha256: str
    task_ids: tuple[str, ...]
    patches: dict[str, ExistingPatch]
    raw: dict[str, Any]
    metric_task_ids: tuple[str, ...] | None = None
    excluded_task_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrozenFewShots:
    root: Path
    stage_a: tuple[Path, ...]
    stage_b: tuple[Path, ...]
    repair: tuple[Path, ...]
    mode: str
    contract_version: str
    prompt_version: str | None
    case_ids: tuple[str, ...]
    file_sha256: tuple[tuple[str, str], ...]
    digest: str


def load_frozen_few_shots(path: str | Path) -> FrozenFewShots:
    root = Path(path).expanduser().resolve()
    provenance_path = root / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing few-shot provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise TypeError(f"Few-shot provenance must be a JSON object: {provenance_path}")
    mode = provenance.get("mode")
    provenance_version = provenance.get("schema_version")
    prompt_version = provenance.get("prompt_version")
    if prompt_version not in {None, "4.0"}:
        raise ValueError(f"Invalid few-shot prompt version: {provenance_path}")
    if provenance_version == "1.0":
        contract_version = "1.0"
        case_ids = DEVELOPMENT_TASK_IDS
        if (
            mode not in {"public_only", "reference_assisted_development_only"}
            or tuple(provenance.get("development_task_ids") or ()) != DEVELOPMENT_TASK_IDS
        ):
            raise ValueError(f"Invalid few-shot provenance contract: {provenance_path}")
    elif provenance_version in {"2.0", "3.0"}:
        contract_version = provenance.get("contract_version")
        case_ids = tuple(provenance.get("case_ids") or ())
        stage_relationships = provenance.get("stage_relationships") or {}
        if (
            mode != f"synthetic_public_contracts_v{provenance_version.split('.')[0]}"
            or contract_version != provenance_version
            or not case_ids
            or len(case_ids) > 8
            or len(case_ids) != len(set(case_ids))
            or any(not re.fullmatch(r"synthetic-[a-z0-9-]+", item) for item in case_ids)
            or provenance.get("synthetic_only") is not True
            or provenance.get("uses_benchmark_artifacts") is not False
            or set(stage_relationships) != set(case_ids)
            or any(
                relationship
                not in {
                    "continuous_synthetic_scenario",
                    "independent_counterfactual_ledger",
                }
                for relationship in stage_relationships.values()
            )
        ):
            raise ValueError(
                f"Invalid v{provenance_version.split('.')[0]} few-shot provenance "
                f"contract: {provenance_path}"
            )
    else:
        raise ValueError(f"Invalid few-shot provenance contract: {provenance_path}")
    if prompt_version == "4.0" and contract_version != "3.0":
        raise ValueError("Few-shot prompt v4.0 requires contract v3.0")
    if mode == "reference_assisted_development_only":
        if provenance.get("user_authorized") is not True:
            raise ValueError("Reference-assisted few-shots require explicit user authorization")
        source_tasks = provenance.get("source_sha256") or {}
        if set(source_tasks) != set(DEVELOPMENT_TASK_IDS):
            raise ValueError("Reference-assisted provenance must cover exactly the development set")
        for task_id, sources in source_tasks.items():
            if (
                not isinstance(sources, dict)
                or set(sources)
                != {
                    "instruction",
                    "p0",
                    "solution",
                    "test_patch",
                }
                or any(
                    not re.fullmatch(r"[0-9a-f]{64}", str(digest)) for digest in sources.values()
                )
            ):
                raise ValueError(f"Invalid reference source digests for {task_id}")
    stage_a: list[Path] = []
    stage_b: list[Path] = []
    repair: list[Path] = []
    forbidden_markers = (
        "hidden test",
        "verifier",
        "grader",
        "solution.patch",
        "reference solution",
        "f2p",
        "p2p",
    )
    for task_id in case_ids:
        task_dir = root / task_id
        files = {
            "stage_a_input": task_dir / "stage_a_input.json",
            "stage_a_output": task_dir / "stage_a_output.json",
            "stage_b_input_summary": task_dir / "stage_b_input_summary.json",
            "stage_b_output": task_dir / "stage_b_output.json",
        }
        for label, file_path in files.items():
            if not file_path.is_file():
                raise FileNotFoundError(f"Missing {label} for development case {task_id}")
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError(f"Few-shot artifact must be a JSON object: {file_path}")
            if payload.get("task_id") != task_id:
                raise ValueError(f"Few-shot task_id mismatch: {file_path}")
            lowered = json.dumps(payload, ensure_ascii=False).lower()
            marker = next((item for item in forbidden_markers if item in lowered), None)
            if marker:
                raise ValueError(
                    f"Few-shot artifact contains forbidden marker {marker!r}: {file_path}"
                )
        stage_a.extend((files["stage_a_input"], files["stage_a_output"]))
        stage_b.extend((files["stage_b_input_summary"], files["stage_b_output"]))
        if contract_version in {"2.0", "3.0"}:
            repair_files = (
                task_dir / "repair_input_summary.json",
                task_dir / "repair_audit_output.json",
            )
            for file_path in repair_files:
                if not file_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {contract_version} repair example for synthetic case "
                        f"{task_id}: {file_path}"
                    )
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or payload.get("task_id") != task_id:
                    raise ValueError(
                        f"Invalid {contract_version} repair few-shot artifact: {file_path}"
                    )
                lowered = json.dumps(payload, ensure_ascii=False).lower()
                marker = next((item for item in forbidden_markers if item in lowered), None)
                if marker:
                    raise ValueError(
                        f"Few-shot artifact contains forbidden marker {marker!r}: {file_path}"
                    )
            repair.extend(repair_files)
    file_sha256 = tuple(
        (str(file_path.relative_to(root)), _sha256_file(file_path))
        for file_path in sorted((*stage_a, *stage_b, *repair, provenance_path))
    )
    digest = hashlib.sha256(
        json.dumps(file_sha256, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FrozenFewShots(
        root=root,
        stage_a=tuple(stage_a),
        stage_b=tuple(stage_b),
        repair=tuple(repair),
        mode=mode,
        contract_version=contract_version,
        prompt_version=prompt_version,
        case_ids=case_ids,
        file_sha256=file_sha256,
        digest=digest,
    )


def freeze_few_shots(source: FrozenFewShots, destination: str | Path) -> FrozenFewShots:
    """Copy versioned audited artifacts to a content-addressed read-only snapshot."""
    target = Path(destination).expanduser().resolve()
    if target.exists():
        frozen = load_frozen_few_shots(target)
        if frozen.file_sha256 != source.file_sha256 or frozen.digest != source.digest:
            raise ValueError(f"Existing few-shot snapshot has unexpected content: {target}")
        return frozen

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for relative, expected in source.file_sha256:
            source_file = source.root / relative
            destination_file = staging / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination_file)
            if _sha256_file(source_file) != expected:
                raise ValueError(f"Few-shot source changed while being frozen: {source_file}")
            if _sha256_file(destination_file) != expected:
                raise ValueError(f"Few-shot snapshot digest mismatch: {destination_file}")
            destination_file.chmod(0o444)
        staging.rename(target)
        for directory, child_directories, _ in os.walk(target, topdown=False):
            for child in child_directories:
                (Path(directory) / child).chmod(0o555)
            Path(directory).chmod(0o555)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    frozen = load_frozen_few_shots(target)
    if frozen.file_sha256 != source.file_sha256 or frozen.digest != source.digest:
        raise ValueError(f"Frozen few-shot snapshot failed final verification: {target}")
    return frozen


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_base_commit(task_dir: Path) -> str:
    """Read the task image's declared repository commit."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    match = re.search(
        r"^ARG\s+BASE_SHA=([0-9a-f]{7,64})\s*$",
        dockerfile.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Cannot find BASE_SHA in {dockerfile}")
    return match.group(1)


def git_object_ids_equivalent(left: str, right: str) -> bool:
    """Compare full or safely abbreviated hexadecimal Git object IDs."""
    pattern = re.compile(r"[0-9a-fA-F]{7,64}")
    if pattern.fullmatch(left) is None or pattern.fullmatch(right) is None:
        return False
    normalized_left = left.lower()
    normalized_right = right.lower()
    return normalized_left.startswith(normalized_right) or normalized_right.startswith(
        normalized_left
    )


def build_reviewer_experiment_manifest(
    *,
    source_job_dir: str | Path,
    checkout: str | Path,
    output: str | Path,
    scope: str = "near-miss16",
) -> Path:
    """Freeze P0 metadata without opening verifier, hidden-test, or solution artifacts."""
    job_dir = Path(source_job_dir).expanduser().resolve()
    checkout_dir = Path(checkout).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    run_record_path = job_dir.parent / f"{job_dir.name}.run.json"
    run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    if run_record.get("benchmark_revision") != DEEPSWE_REVISION:
        raise ValueError("Source job is not DeepSWE v1.1")
    if run_record.get("agent") != "pi":
        raise ValueError("The first experiment requires a Pi P0 source job")
    source_manifest = Path(run_record["manifest"]).expanduser().resolve()
    source_manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_task_ids = tuple(source_manifest_payload.get("task_ids") or ())
    if (
        source_manifest_payload.get("benchmark_revision") != DEEPSWE_REVISION
        or source_manifest_payload.get("task_count") != len(source_task_ids)
        or not source_task_ids
        or len(source_task_ids) != len(set(source_task_ids))
    ):
        raise ValueError("Source DeepSWE manifest has an invalid task selection")
    if scope == "near-miss16":
        task_ids = EVALUATION_TASK_IDS
        metric_task_ids = EVALUATION_TASK_IDS
        excluded_task_ids: tuple[str, ...] = ()
        frozen_task_ids = (*DEVELOPMENT_TASK_IDS, *EVALUATION_TASK_IDS)
        selection_method = "fixed-near-miss-failures-minus-development-cases-v1"
    elif scope == "random60":
        task_ids = source_task_ids
        excluded_task_ids = tuple(
            task_id for task_id in source_task_ids if task_id in DEVELOPMENT_TASK_IDS
        )
        if set(excluded_task_ids) != set(DEVELOPMENT_TASK_IDS):
            raise ValueError("The random60 source manifest must contain both development tasks")
        metric_task_ids = tuple(
            task_id for task_id in source_task_ids if task_id not in DEVELOPMENT_TASK_IDS
        )
        frozen_task_ids = task_ids
        selection_method = "source-manifest-all-tasks-with-development-exclusions-v1"
    else:
        raise ValueError(f"Unsupported reviewer experiment scope: {scope}")
    trials: dict[str, tuple[Path, dict[str, Any]]] = {}
    job_ids: set[str] = set()
    for trial_dir in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        config_path = trial_dir / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        task_path = Path(config.get("task", {}).get("path", ""))
        task_id = task_path.name
        if task_id:
            if task_id in trials:
                raise ValueError(f"Source job has duplicate trials for {task_id}")
            trials[task_id] = (trial_dir, config)
            job_id = config.get("job_id")
            if isinstance(job_id, str) and job_id:
                job_ids.add(job_id)
    if len(job_ids) != 1:
        raise ValueError("Source trials do not share one absolute Pier job ID")
    missing = [task_id for task_id in frozen_task_ids if task_id not in trials]
    if missing:
        raise ValueError("Source job lacks required P0 tasks: " + ", ".join(missing))

    tasks: dict[str, Any] = {}
    for task_id in frozen_task_ids:
        trial_dir, _ = trials[task_id]
        patch_path = (trial_dir / "artifacts" / "model.patch").resolve()
        if not patch_path.is_file():
            raise FileNotFoundError(f"Source P0 is missing for {task_id}: {patch_path}")
        patch_sha = _sha256_file(patch_path)
        tasks[task_id] = {
            "source_trial_id": trial_dir.name,
            "base_commit": task_base_commit(checkout_dir / "tasks" / task_id),
            "patch_path": str(patch_path),
            "patch_sha256": patch_sha,
            "tracked_diff_sha256": patch_sha,
        }
    payload = {
        "schema_version": "1.0",
        "experiment": "specification-reviewer-repair-once",
        "benchmark": "deepswe",
        "benchmark_revision": DEEPSWE_REVISION,
        "benchmark_source_commit": DEEPSWE_COMMIT,
        "source_job": {
            "id": next(iter(job_ids)),
            "path": str(job_dir),
            "run_record_path": str(run_record_path.resolve()),
            "run_record_sha256": _sha256_file(run_record_path),
        },
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": _sha256_file(source_manifest),
        },
        "development_task_ids": list(DEVELOPMENT_TASK_IDS),
        "task_ids": list(task_ids),
        "task_count": len(task_ids),
        "metric_task_ids": list(metric_task_ids),
        "metric_task_count": len(metric_task_ids),
        "excluded_task_ids": list(excluded_task_ids),
        "selection_method": selection_method,
        "tasks": tasks,
    }
    atomic_write_json(output_path, payload)
    return output_path


def load_reviewer_experiment_manifest(path: str | Path) -> ReviewerExperimentManifest:
    resolved = Path(path).expanduser().resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version") != "1.0"
        or raw.get("experiment") != "specification-reviewer-repair-once"
    ):
        raise ValueError(f"Not a reviewer experiment manifest: {resolved}")
    if (
        raw.get("benchmark_revision") != DEEPSWE_REVISION
        or raw.get("benchmark_source_commit") != DEEPSWE_COMMIT
    ):
        raise ValueError("Reviewer experiment manifest does not target the pinned DeepSWE v1.1")
    if tuple(raw.get("development_task_ids") or ()) != DEVELOPMENT_TASK_IDS:
        raise ValueError("Development task IDs differ from the frozen experiment design")
    source_job = raw.get("source_job") or {}
    source_manifest = raw.get("source_manifest") or {}
    if not isinstance(source_job.get("id"), str) or not source_job["id"]:
        raise ValueError("Reviewer experiment manifest has no source Pier job ID")
    source_job_dir = Path(source_job.get("path", "")).expanduser().resolve()
    run_record_path = Path(source_job.get("run_record_path", "")).expanduser().resolve()
    if _sha256_file(run_record_path) != source_job.get("run_record_sha256"):
        raise ValueError("Source Pier run record digest mismatch")
    source_manifest_path = Path(source_manifest.get("path", "")).expanduser().resolve()
    if _sha256_file(source_manifest_path) != source_manifest.get("sha256"):
        raise ValueError("Source DeepSWE manifest digest mismatch")
    source_manifest_payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_task_ids = tuple(source_manifest_payload.get("task_ids") or ())
    selection_method = raw.get("selection_method")
    task_ids = tuple(raw.get("task_ids") or ())
    metric_task_ids = tuple(raw.get("metric_task_ids") or task_ids)
    excluded_task_ids = tuple(raw.get("excluded_task_ids") or ())
    if raw.get("task_count") != len(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("Reviewer task IDs are invalid")
    if selection_method == "fixed-near-miss-failures-minus-development-cases-v1":
        if task_ids != EVALUATION_TASK_IDS:
            raise ValueError("Evaluation task IDs differ from the frozen 16-task design")
        if metric_task_ids != task_ids or excluded_task_ids:
            raise ValueError("The near-miss design cannot exclude metric tasks")
        expected_records = set(task_ids) | set(DEVELOPMENT_TASK_IDS)
    elif selection_method == "source-manifest-all-tasks-with-development-exclusions-v1":
        if task_ids != source_task_ids:
            raise ValueError("Full reviewer task IDs differ from the source manifest")
        expected_excluded = tuple(
            task_id for task_id in task_ids if task_id in DEVELOPMENT_TASK_IDS
        )
        expected_metrics = tuple(
            task_id for task_id in task_ids if task_id not in DEVELOPMENT_TASK_IDS
        )
        if excluded_task_ids != expected_excluded or metric_task_ids != expected_metrics:
            raise ValueError("Full reviewer metric exclusions are invalid")
        if raw.get("metric_task_count") != len(metric_task_ids):
            raise ValueError("Full reviewer metric task count is invalid")
        expected_records = set(task_ids)
    else:
        raise ValueError("Unsupported reviewer experiment selection method")
    if set(metric_task_ids) & set(excluded_task_ids):
        raise ValueError("Metric and excluded reviewer tasks must be disjoint")
    if set(metric_task_ids) | set(excluded_task_ids) != set(task_ids):
        raise ValueError("Metric and excluded reviewer tasks must cover the run set")
    task_records = raw.get("tasks") or {}
    if set(task_records) != expected_records:
        raise ValueError("P0 task records do not match the frozen reviewer task set")
    patches: dict[str, ExistingPatch] = {}
    for task_id, record in task_records.items():
        patch_path = Path(record["patch_path"]).expanduser().resolve()
        expected_patch_path = (
            source_job_dir / record["source_trial_id"] / "artifacts" / "model.patch"
        ).resolve()
        if patch_path != expected_patch_path:
            raise ValueError(f"P0 path is outside the frozen source trial for {task_id}")
        if _sha256_file(patch_path) != record.get("patch_sha256"):
            raise ValueError(f"P0 digest mismatch for {task_id}")
        if not re.fullmatch(r"[0-9a-f]{7,64}", str(record.get("base_commit", ""))):
            raise ValueError(f"Invalid base commit for {task_id}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("tracked_diff_sha256", ""))):
            raise ValueError(f"Invalid tracked diff digest for {task_id}")
        patches[task_id] = ExistingPatch(
            task_id=task_id,
            base_commit=record["base_commit"],
            path=patch_path,
            sha256=record["patch_sha256"],
            tracked_diff_sha256=record["tracked_diff_sha256"],
            source_job_id=source_job["id"],
            source_trial_id=record["source_trial_id"],
        )
    return ReviewerExperimentManifest(
        path=resolved,
        source_job_id=source_job["id"],
        source_job_dir=source_job_dir,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest["sha256"],
        task_ids=task_ids,
        patches=patches,
        raw=raw,
        metric_task_ids=metric_task_ids,
        excluded_task_ids=excluded_task_ids,
    )
