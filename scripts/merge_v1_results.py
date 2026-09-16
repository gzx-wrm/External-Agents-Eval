"""Build the deterministic aggregate view for the complete v1 58-task run.

This is a post-hoc reporting utility.  It does not alter any source trial and
does not expose aggregate verifier evidence to an agent.  The aggregate view
uses the frozen experiment rule: primary58 first, then canary3 and recovery11
for their rerun task sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from external_agents_eval.benchmarks.deepswe.report import (
    collect_reviewer_experiment_summary,
)
from external_agents_eval.benchmarks.deepswe.reviewer_experiment_manifest import (
    load_reviewer_experiment_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "deepswe"
BASELINE_MANIFEST = ROOT / "manifests" / "deepswe_v1.1_pi_random60_p0_20260901.json"
SOURCE_JOBS = {
    "primary58": "deepswe-v1.1-pi-review-repair-primary58-v1",
    "canary3": "deepswe-v1.1-pi-review-repair-canary3-v13",
    "recovery11": "deepswe-v1.1-pi-review-repair-recovery11-v13",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _task_id(result: dict[str, Any]) -> str:
    task_name = result.get("task_name")
    if isinstance(task_name, str) and task_name:
        return task_name.rsplit("/", 1)[-1]
    task_id = result.get("task_id")
    if isinstance(task_id, dict) and isinstance(task_id.get("path"), str):
        return Path(task_id["path"]).name
    config = result.get("config")
    if isinstance(config, dict):
        task = config.get("task")
        if isinstance(task, dict) and isinstance(task.get("path"), str):
            return Path(task["path"]).name
    raise ValueError("Cannot infer task ID from trial result")


def _index_job(path: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for result_path in sorted(path.glob("*/result.json")):
        task = _task_id(_read(result_path))
        if task in indexed:
            raise ValueError(f"Duplicate trial for {task} in {path}")
        indexed[task] = result_path.parent
    return indexed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(output: Path) -> dict[str, Any]:
    frozen = load_reviewer_experiment_manifest(BASELINE_MANIFEST)
    metric_tasks = tuple(frozen.metric_task_ids or frozen.task_ids)
    excluded = set(frozen.excluded_task_ids)
    if len(metric_tasks) != 58 or len(excluded) != 2:
        raise ValueError("The aggregate is defined for the frozen random60/58-task manifest")

    indexes = {
        label: _index_job(RESULTS / job_name) for label, job_name in SOURCE_JOBS.items()
    }
    canary_tasks = set(indexes["canary3"])
    recovery_tasks = set(indexes["recovery11"])
    if canary_tasks & recovery_tasks:
        raise ValueError("Canary and recovery task sets must be disjoint")
    if (canary_tasks | recovery_tasks) - set(metric_tasks):
        raise ValueError("Rerun task set contains a non-metric task")

    selected: dict[str, tuple[str, Path]] = {}
    for task in metric_tasks:
        if task in canary_tasks:
            selected[task] = ("canary3", indexes["canary3"][task])
        elif task in recovery_tasks:
            selected[task] = ("recovery11", indexes["recovery11"][task])
        else:
            selected[task] = ("primary58", indexes["primary58"][task])

    counts = {label: sum(source == label for source, _ in selected.values()) for label in SOURCE_JOBS}
    if counts != {"primary58": 44, "canary3": 3, "recovery11": 11}:
        raise ValueError(f"Unexpected deterministic source coverage: {counts}")

    output.mkdir(parents=True, exist_ok=False)
    trials_dir = output / "trials"
    trials_dir.mkdir()
    index_payload: dict[str, Any] = {}
    for task in metric_tasks:
        source, trial_dir = selected[task]
        link = trials_dir / task
        link.symlink_to(trial_dir.resolve(), target_is_directory=True)
        index_payload[task] = {
            "source": source,
            "source_job": SOURCE_JOBS[source],
            "source_trial_dir": str(trial_dir.resolve()),
            "aggregate_trial_link": str(link.relative_to(output)),
        }

    # The existing collector expects trial directories directly below its job
    # root, so collect through the aggregate's dedicated trial-link view.
    summary = collect_reviewer_experiment_summary(trials_dir, frozen)
    if summary.get("evaluated_count") != 58:
        raise ValueError(f"Aggregate did not evaluate all 58 metric tasks: {summary.get('evaluated_count')}")
    summary["job_dir"] = str(output.resolve())
    summary["aggregate"] = True
    summary["source_jobs"] = SOURCE_JOBS
    summary["source_coverage"] = counts
    summary["selection_rule"] = "primary58, overridden by disjoint canary3 and recovery11 task sets"

    aggregate_manifest = {
        "schema_version": "1.0",
        "aggregate": True,
        "benchmark": "deepswe",
        "benchmark_revision": "1.1",
        "experiment": "specification-reviewer-repair-once",
        "task_count": 58,
        "metric_task_ids": list(metric_tasks),
        "excluded_task_ids": sorted(excluded),
        "source_jobs": SOURCE_JOBS,
        "source_coverage": counts,
        "selection_rule": "primary58, overridden by disjoint canary3 and recovery11 task sets",
        "task_index": "task_index.json",
        "summary": "review-summary.json",
    }
    _write_json(output / "manifest.json", aggregate_manifest)
    _write_json(output / "task_index.json", index_payload)
    _write_json(output / "review-summary.json", summary)
    _write_json(
        output / "aggregate_result.json",
        {
            "aggregate": True,
            "benchmark": "deepswe",
            "benchmark_revision": "1.1",
            "task_count": 58,
            "p0_passed_count": summary["p0_passed_count"],
            "p1_passed_count": summary["p1_passed_count"],
            "p0_pass_rate": summary["p0_pass_rate"],
            "p1_pass_rate": summary["p1_pass_rate"],
            "transitions": summary["transitions"],
            "source_coverage": counts,
            "review_summary": "review-summary.json",
        },
    )
    (output / "README.md").write_text(
        "# DeepSWE v1 complete 58-task aggregate\n\n"
        "This is a post-hoc aggregate view. Source trials are preserved in the three source jobs; "
        "`trials/` contains symlinks selected by the deterministic primary58 → canary3 → recovery11 rule.\n\n"
        "See `review-summary.json` for the paired P0/P1 summary and `task_index.json` for provenance.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "deepswe-v1.1-pi-review-repair-v1-complete58",
    )
    args = parser.parse_args()
    summary = build(args.output.expanduser().resolve())
    print(json.dumps({k: summary[k] for k in ("evaluated_count", "p0_passed_count", "p1_passed_count", "transitions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
