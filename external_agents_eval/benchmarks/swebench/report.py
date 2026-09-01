from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from external_agents_eval.benchmarks.swebench.config import DEFAULT_SWEBENCH_RUNNER


def remove_test_changes(model_patch: str) -> str:
    lines = model_patch.splitlines(keepends=True)
    filtered: list[str] = []
    skip = False
    for line in lines:
        if line.startswith("diff --git a/"):
            pieces = line.split()
            target = pieces[-1] if pieces else ""
            skip = target.startswith("b/") and any(
                marker in target
                for marker in ("/test/", "/tests/", "/testing/", "/test_", "/tox.ini")
            )
        if not skip:
            filtered.append(line)
    return "".join(filtered)


def load_predictions(
    prediction_dir: Path,
    task_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    selected = set(task_ids)
    for path in sorted(prediction_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("generation_status") != "success":
            continue
        instance_id = payload.get("instance_id")
        if (
            instance_id in selected
            and isinstance(instance_id, str)
            and isinstance(payload.get("model_patch"), str)
        ):
            predictions[instance_id] = payload
    return predictions


def write_predictions_jsonl(prediction_dir: Path, predictions: dict[str, dict[str, Any]]) -> Path:
    output = prediction_dir / "all_preds.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for payload in predictions.values():
            handle.write(
                json.dumps(
                    {
                        "instance_id": payload["instance_id"],
                        "model_name_or_path": payload["model_name_or_path"],
                        "model_patch": remove_test_changes(payload["model_patch"]),
                    }
                )
                + "\n"
            )
    temporary.replace(output)
    return output


def run_official_report(
    *,
    agent_dir: Path,
    dataset_name: str,
    task_ids: tuple[str, ...],
    num_workers: int,
    run_id: str,
    runner_path: Path = DEFAULT_SWEBENCH_RUNNER,
    retries: int = 2,
) -> Path:
    prediction_dir = agent_dir / "predictions"
    predictions = load_predictions(prediction_dir, task_ids)
    if not predictions:
        raise RuntimeError(f"No successful predictions found in {prediction_dir}")
    predictions_jsonl = write_predictions_jsonl(prediction_dir, predictions)
    runner = runner_path.expanduser().resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"SWE-bench evaluation runner not found: {runner}")
    output_dir = agent_dir / "official"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(runner),
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions_jsonl),
        "--max_workers",
        str(num_workers),
        "--run_id",
        run_id,
        "--instance_ids",
        *task_ids,
        "--force_rebuild",
        "false",
        "--cache_level",
        "instance",
        "--clean",
        "false",
    ]
    for attempt in range(retries + 1):
        try:
            subprocess.run(command, check=True, cwd=output_dir)
            return output_dir
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            print(f"Official report attempt {attempt + 1} failed; retrying...", flush=True)
    raise AssertionError("unreachable")
