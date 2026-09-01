from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from external_agents_eval.benchmarks.deepswe.manifest import DeepSWEManifest


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _task_id(result: dict[str, Any], selected: set[str]) -> str | None:
    candidates: list[str] = []
    task_name = result.get("task_name")
    if isinstance(task_name, str):
        candidates.extend((task_name, task_name.rsplit("/", 1)[-1]))

    task_id = result.get("task_id")
    config = result.get("config")
    for value in (
        task_id.get("path") if isinstance(task_id, dict) else None,
        config.get("task", {}).get("path") if isinstance(config, dict) else None,
    ):
        if isinstance(value, str):
            candidates.append(Path(value).name)

    return next((candidate for candidate in candidates if candidate in selected), None)


def collect_summary(job_dir: str | Path, manifest: DeepSWEManifest) -> dict[str, Any]:
    root = Path(job_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Pier job directory not found: {root}")
    selected = set(manifest.task_ids)
    trials_by_task: dict[str, list[dict[str, Any]]] = {}
    extra_tasks: set[str] = set()
    result_paths = set(root.glob("*/result.json"))
    result_paths.update(root.glob("*/results.json"))  # Pier < 0.3 compatibility
    for result_path in sorted(result_paths):
        result = _load_json(result_path)
        if result is None:
            continue
        task_name = _task_id(result, selected)
        if task_name is None:
            raw_task_name = result.get("task_name")
            if isinstance(raw_task_name, str):
                extra_tasks.add(raw_task_name)
            continue
        result["_trial_dir"] = str(result_path.parent)
        trials_by_task.setdefault(task_name, []).append(result)

    counts: Counter[str] = Counter()
    details: dict[str, Any] = {}
    rewards: list[float] = []
    for task_id in manifest.task_ids:
        trials = trials_by_task.get(task_id, [])
        if not trials:
            counts["pending"] += 1
            details[task_id] = {"status": "pending"}
            continue
        latest = max(
            trials,
            key=lambda value: str(value.get("finished_at") or value.get("started_at") or ""),
        )
        verifier = latest.get("verifier_result") or {}
        reward_map = verifier.get("rewards") if isinstance(verifier, dict) else None
        reward = reward_map.get("reward") if isinstance(reward_map, dict) else None
        exception = latest.get("exception_info")
        if isinstance(reward, (int, float)):
            numeric_reward = float(reward)
            rewards.append(numeric_reward)
            status = "passed" if numeric_reward == 1.0 else "failed"
        elif exception:
            status = "error"
        else:
            status = "incomplete"
        counts[status] += 1
        details[task_id] = {
            "status": status,
            "reward": reward,
            "rewards": reward_map,
            "trial_dir": latest["_trial_dir"],
            "attempt_count": len(trials),
            "exception": exception,
        }
    evaluated_pass_rate = (sum(rewards) / len(rewards)) if rewards else None
    return {
        "benchmark": "deepswe",
        "benchmark_revision": "1.1",
        "job_dir": str(root),
        "task_count": len(manifest.task_ids),
        "evaluated_count": len(rewards),
        "pass_rate": (evaluated_pass_rate if len(rewards) == len(manifest.task_ids) else None),
        "evaluated_pass_rate": evaluated_pass_rate,
        "counts": dict(sorted(counts.items())),
        "extra_tasks": sorted(extra_tasks),
        "details": details,
    }
