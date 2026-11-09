from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from external_agents_eval.benchmarks.deepswe.config import (
    DEEPSWE_COMMIT,
    DEEPSWE_REPOSITORY,
    DEEPSWE_REVISION,
    DEEPSWE_TASK_COUNT,
)
from external_agents_eval.benchmarks.swebench.manifest import atomic_write_json


@dataclass(frozen=True)
class DeepSWEManifest:
    path: Path
    task_ids: tuple[str, ...]
    sample_seed: int
    sample_size: int
    population_count: int
    population_fingerprint: str
    raw: dict[str, Any]


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ensure_checkout(path: str | Path) -> Path:
    """Clone the immutable DeepSWE source, or validate an existing checkout."""
    checkout = Path(path).expanduser().resolve()
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--no-checkout", DEEPSWE_REPOSITORY, str(checkout))
        _git("checkout", "--detach", DEEPSWE_COMMIT, cwd=checkout)
    return validate_checkout(checkout)


def validate_checkout(path: str | Path) -> Path:
    checkout = Path(path).expanduser().resolve()
    if not (checkout / ".git").is_dir():
        raise ValueError(f"DeepSWE checkout is not a Git repository: {checkout}")
    head = _git("rev-parse", "HEAD", cwd=checkout)
    if head != DEEPSWE_COMMIT:
        raise ValueError(
            f"DeepSWE checkout is at {head}, expected pinned v{DEEPSWE_REVISION} "
            f"commit {DEEPSWE_COMMIT}"
        )
    dirty = _git("status", "--porcelain", cwd=checkout)
    if dirty:
        raise ValueError(
            f"DeepSWE checkout has local changes: {checkout}. Use a clean pinned checkout."
        )
    discover_tasks(checkout / "tasks", require_v11=True)
    return checkout


def discover_tasks(tasks_dir: str | Path, *, require_v11: bool = True) -> list[Path]:
    root = Path(tasks_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"DeepSWE task directory not found: {root}")
    tasks = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "task.toml").is_file() and (path / "instruction.md").is_file()
    )
    if not tasks:
        raise ValueError(f"No valid DeepSWE tasks found in {root}")
    if require_v11:
        invalid: list[str] = []
        for task in tasks:
            config = (task / "task.toml").read_text(encoding="utf-8")
            if 'environment_mode = "separate"' not in config or "-v1.1" not in config:
                invalid.append(task.name)
        if invalid:
            raise ValueError(
                "Checkout does not contain uniform DeepSWE v1.1 tasks: " + ", ".join(invalid[:5])
            )
    return tasks


def population_fingerprint(tasks: list[Path]) -> str:
    digest = hashlib.sha256()
    for task in sorted(tasks, key=lambda item: item.name):
        digest.update(task.name.encode())
        digest.update(b"\0")
        for filename in ("task.toml", "instruction.md"):
            digest.update(filename.encode())
            digest.update(b"\0")
            digest.update((task / filename).read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def select_task_ids(tasks: list[Path], *, sample_size: int, seed: int) -> tuple[str, ...]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    task_ids = [task.name for task in sorted(tasks, key=lambda item: item.name)]
    if sample_size > len(task_ids):
        raise ValueError(f"sample_size={sample_size} exceeds the {len(task_ids)} available tasks")
    random.Random(seed).shuffle(task_ids)
    return tuple(task_ids[:sample_size])


def build_manifest_payload(
    checkout: str | Path,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    resolved = validate_checkout(checkout)
    tasks = discover_tasks(resolved / "tasks")
    if len(tasks) != DEEPSWE_TASK_COUNT:
        raise ValueError(
            f"Pinned DeepSWE v{DEEPSWE_REVISION} should contain {DEEPSWE_TASK_COUNT} "
            f"tasks, found {len(tasks)}"
        )
    task_ids = select_task_ids(tasks, sample_size=sample_size, seed=seed)
    population_hash = population_fingerprint(tasks)
    selection_hash = hashlib.sha256("\n".join(task_ids).encode()).hexdigest()
    return {
        "schema_version": 1,
        "benchmark": "deepswe",
        "benchmark_revision": DEEPSWE_REVISION,
        "source": {
            "repository": DEEPSWE_REPOSITORY,
            "commit": DEEPSWE_COMMIT,
        },
        "sampling": {
            "algorithm": "sorted-task-ids/python-random-shuffle-v1",
            "seed": seed,
            "size": sample_size,
        },
        "population": {
            "task_count": len(tasks),
            "fingerprint_sha256": population_hash,
        },
        "task_count": len(task_ids),
        "task_ids": list(task_ids),
        "selection_fingerprint_sha256": selection_hash,
    }


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        try:
            current = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read existing DeepSWE manifest {output}: {error}") from error
        if current != payload:
            raise ValueError(f"DeepSWE manifest already exists with different contents: {output}")
        return output
    atomic_write_json(output, payload)
    return output


def load_manifest(path: str | Path) -> DeepSWEManifest:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read DeepSWE manifest {resolved}: {error}") from error
    if not isinstance(raw, dict) or raw.get("benchmark") != "deepswe":
        raise ValueError(f"Not a DeepSWE manifest: {resolved}")
    if raw.get("benchmark_revision") != DEEPSWE_REVISION:
        raise ValueError(f"Unsupported DeepSWE revision in {resolved}")
    source = raw.get("source") or {}
    if source.get("repository") != DEEPSWE_REPOSITORY or source.get("commit") != DEEPSWE_COMMIT:
        raise ValueError(f"DeepSWE source pin differs from the supported v1.1 source: {resolved}")
    task_ids = raw.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError(f"DeepSWE manifest has invalid task_ids: {resolved}")
    if raw.get("task_count") != len(task_ids):
        raise ValueError(f"DeepSWE manifest task_count does not match task_ids: {resolved}")
    sampling = raw.get("sampling") or {}
    population = raw.get("population") or {}
    if sampling.get("algorithm") != "sorted-task-ids/python-random-shuffle-v1":
        raise ValueError(f"Unsupported DeepSWE sampling algorithm: {resolved}")
    if sampling.get("size") != len(task_ids):
        raise ValueError(f"DeepSWE sampling size does not match task_ids: {resolved}")
    if not isinstance(sampling.get("seed"), int):
        raise ValueError(  # noqa: TRY004 - malformed file value, not a caller type error
            f"DeepSWE manifest has an invalid sampling seed: {resolved}"
        )
    excluded = sampling.get("excluded_task_ids", [])
    if (
        not isinstance(excluded, list)
        or any(not isinstance(task_id, str) or not task_id for task_id in excluded)
        or len(excluded) != len(set(excluded))
    ):
        raise ValueError(f"DeepSWE manifest has invalid excluded_task_ids: {resolved}")
    if set(excluded).intersection(task_ids):
        raise ValueError(f"DeepSWE manifest selects an excluded task: {resolved}")
    fingerprint = population.get("fingerprint_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(f"DeepSWE manifest has an invalid population fingerprint: {resolved}")
    selection_fingerprint = raw.get("selection_fingerprint_sha256")
    expected_selection_fingerprint = hashlib.sha256("\n".join(task_ids).encode()).hexdigest()
    if selection_fingerprint != expected_selection_fingerprint:
        raise ValueError(f"DeepSWE selection fingerprint differs from task_ids: {resolved}")
    population_count = population.get("task_count")
    if not isinstance(population_count, int) or population_count <= 0:
        raise ValueError(f"DeepSWE manifest has an invalid population count: {resolved}")
    return DeepSWEManifest(
        path=resolved,
        task_ids=tuple(task_ids),
        sample_seed=sampling["seed"],
        sample_size=sampling["size"],
        population_count=population_count,
        population_fingerprint=fingerprint,
        raw=raw,
    )


def validate_manifest_against_checkout(manifest: DeepSWEManifest, checkout: str | Path) -> Path:
    resolved = validate_checkout(checkout)
    tasks = discover_tasks(resolved / "tasks")
    task_ids = {task.name for task in tasks}
    missing = [task_id for task_id in manifest.task_ids if task_id not in task_ids]
    if missing:
        raise ValueError("Manifest tasks are absent from checkout: " + ", ".join(missing[:5]))
    if len(tasks) != manifest.population_count:
        raise ValueError("DeepSWE population count differs from the manifest")
    if population_fingerprint(tasks) != manifest.population_fingerprint:
        raise ValueError("DeepSWE population fingerprint differs from the manifest")
    sampling = manifest.raw.get("sampling") or {}
    excluded = set(sampling.get("excluded_task_ids", []))
    unknown_exclusions = sorted(excluded - task_ids)
    if unknown_exclusions:
        raise ValueError(
            "Excluded tasks are absent from checkout: " + ", ".join(unknown_exclusions[:5])
        )
    candidates = [task for task in tasks if task.name not in excluded]
    if "candidate_count" in sampling and sampling["candidate_count"] != len(candidates):
        raise ValueError("DeepSWE candidate count differs from the manifest")
    candidate_hash = hashlib.sha256("\n".join(task.name for task in candidates).encode()).hexdigest()
    if (
        "candidate_fingerprint_sha256" in sampling
        and sampling["candidate_fingerprint_sha256"] != candidate_hash
    ):
        raise ValueError("DeepSWE candidate fingerprint differs from the manifest")
    expected_task_ids = select_task_ids(
        candidates,
        sample_size=manifest.sample_size,
        seed=manifest.sample_seed,
    )
    if expected_task_ids != manifest.task_ids:
        raise ValueError("DeepSWE task selection differs from the seeded manifest")
    return resolved
