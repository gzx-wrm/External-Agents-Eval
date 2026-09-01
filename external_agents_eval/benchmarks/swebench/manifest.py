from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskManifest:
    path: Path
    dataset_name: str
    task_ids: tuple[str, ...]
    raw: dict[str, Any]


def load_task_manifest(path: str | Path) -> TaskManifest:
    resolved = Path(path).expanduser().resolve()
    try:
        raw = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read task manifest {resolved}: {error}") from error

    if not isinstance(raw, dict):
        raise ValueError(  # noqa: TRY004 - malformed file value, not a caller type error
            f"Task manifest must be a JSON object: {resolved}"
        )
    task_ids = raw.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError(f"Task manifest must contain a non-empty task_ids list: {resolved}")
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ValueError(f"Task manifest contains an invalid task ID: {resolved}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Task manifest contains duplicate task IDs: {resolved}")
    declared_count = raw.get("task_count")
    if declared_count is not None and declared_count != len(task_ids):
        raise ValueError(
            f"task_count={declared_count} does not match len(task_ids)={len(task_ids)}"
        )
    dataset_name = raw.get("dataset_name", "princeton-nlp/SWE-bench_Lite")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(f"Task manifest has an invalid dataset_name: {resolved}")
    return TaskManifest(
        path=resolved,
        dataset_name=dataset_name,
        task_ids=tuple(task_ids),
        raw=raw,
    )


def persisted_manifest(source: TaskManifest) -> dict[str, Any]:
    payload = dict(source.raw)
    payload["dataset_name"] = source.dataset_name
    payload["task_count"] = len(source.task_ids)
    payload["task_ids"] = list(source.task_ids)
    payload["source_task_manifest"] = str(source.path)
    return payload


def ensure_result_manifest(path: Path, manifest: TaskManifest) -> None:
    expected = persisted_manifest(manifest)
    if path.exists():
        try:
            current = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read existing result manifest {path}: {error}") from error
        if current != expected:
            raise ValueError(
                f"Task selection differs from existing result manifest {path}; "
                "use a fresh result directory or the original manifest"
            )
        return
    atomic_write_json(path, expected)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temp_path.replace(path)
