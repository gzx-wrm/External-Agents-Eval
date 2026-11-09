"""Build the remaining 25 full-evidence inputs without selecting by reward."""

from __future__ import annotations

import argparse
import json
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator

from build_f_pilot import (
    HERE,
    PROJECT_ROOT,
    RESULTS_ROOT,
    TASKS_ROOT,
    ensure_baseline,
    git,
    json_bytes,
    raw_read_paths,
    render_user,
    sha256,
    spec_input,
)
from run_f_inference import validate_output


def resolve_commit(repository_url: str, revision: str) -> str:
    if len(revision) == 40:
        return revision
    repository = repository_url.removeprefix("https://github.com/").removesuffix(".git")
    request = Request(
        f"https://github.com/{repository}/commit/{revision}.patch",
        headers={"User-Agent": "external-agents-eval"},
    )
    with urlopen(request, timeout=30) as response:
        commit = response.readline().decode().split()[1]
    if len(commit) != 40 or not commit.startswith(revision):
        raise ValueError(f"Resolved commit does not match {revision}: {commit}")
    return commit


def ledger_target(path: Path) -> dict:
    ledger = json.loads(path.read_text())
    target = {"summary": ledger["summary"]}
    for field in ("obligations", "preservation_obligations", "open_questions"):
        target[field] = [entry["content"] for entry in ledger[field]]
    return target


def annotation_paths(trials: list[tuple[str, Path]]) -> dict[str, Path]:
    source_runs = {task_id: trial / "v6_review_repair" for task_id, trial in trials}
    annotations = {}
    for fork_path in sorted((RESULTS_ROOT / "manual_annotations").glob("*/fork.json")):
        fork = json.loads(fork_path.read_text())
        task_id = fork["task_id"]
        if task_id not in source_runs:
            continue
        if Path(fork["source_run"]).resolve() != source_runs[task_id].resolve():
            raise ValueError(f"Annotation refers to another source run: {fork_path}")
        if task_id in annotations:
            raise ValueError(f"Multiple annotations require an explicit choice: {task_id}")
        annotations[task_id] = Path(fork["reuse"]["A"])
    return annotations


def build_record(
    task_id: str, trial: Path, prompt: str, validator: Draft202012Validator,
    annotation_path: Path | None = None,
) -> tuple[dict, dict, dict]:
    metadata = tomllib.loads((TASKS_ROOT / task_id / "task.toml").read_text())["metadata"]
    commit = resolve_commit(metadata["repository_url"], metadata["base_commit_hash"])
    repository = ensure_baseline(task_id, metadata["repository_url"], commit)
    a_root = trial / "v6_review_repair" / "v5"
    prompt_path = a_root / "specification" / "attempt-1" / "prompt.md"
    trajectory_path = prompt_path.parent / "trajectory.json"
    ledger_path = a_root / "ledger.json"
    a_input = spec_input(prompt_path)
    requested_paths = raw_read_paths(trajectory_path, a_input["base_root"])
    tracked_paths = set(git(repository, "ls-tree", "-r", "--name-only", commit).decode().splitlines())
    evidence = []
    skipped_reads = []
    for path in requested_paths:
        if path not in tracked_paths:
            skipped_reads.append({"path": path, "reason": "Not present in the frozen baseline."})
            continue
        content = git(repository, "show", f"{commit}:{path}")
        text = content.decode("utf-8")
        evidence.append(
            {
                "path": path,
                "line_start": 1,
                "line_end": len(text.splitlines()),
                "sha256": sha256(content),
                "text": text,
            }
        )
    if not evidence:
        raise ValueError(f"No frozen baseline files available for {task_id}")
    teacher_target = ledger_target(ledger_path)
    selected_path = annotation_path or ledger_path
    target = ledger_target(selected_path) if annotation_path else teacher_target
    label_source = "human_annotation" if annotation_path else "teacher_raw"
    record = {
        "schema_version": "f_pilot_v1",
        "task_id": task_id,
        "split": "development_only",
        "label_source": label_source,
        "label_status": f"{label_source}_mechanically_valid_semantics_ungraded",
        "repository_url": metadata["repository_url"],
        "declared_base_commit": metadata["base_commit_hash"],
        "base_commit": commit,
        "task": a_input["task"],
        "managed_delivery": a_input.get("managed_delivery", []),
        "evidence_rule": "Complete frozen baseline files requested by A's read tool, deduplicated and sorted by relative path; absent baseline paths recorded separately; no target-guided snippets.",
        "evidence": evidence,
        "skipped_reads": skipped_reads,
        "target": target,
        "source_corrections": [],
        "provenance": {
            "source_trial": str(trial.relative_to(PROJECT_ROOT)),
            "spec_prompt": str(prompt_path.relative_to(PROJECT_ROOT)),
            "a_trajectory": str(trajectory_path.relative_to(PROJECT_ROOT)),
            "teacher_ledger": str(ledger_path.relative_to(PROJECT_ROOT)),
            "teacher_ledger_sha256": sha256(ledger_path.read_bytes()),
            "selected_label_ledger": str(selected_path.relative_to(PROJECT_ROOT)),
            "selected_label_ledger_sha256": sha256(selected_path.read_bytes()),
        },
    }
    if annotation_path:
        record["teacher_target"] = teacher_target
        fork_path = annotation_path.parent / "fork.json"
        record["provenance"]["annotation_fork"] = str(fork_path.relative_to(PROJECT_ROOT))
        record["provenance"]["annotation_fork_sha256"] = sha256(fork_path.read_bytes())
    _, checks = validate_output(
        json.dumps(target), {"task": record["task"], "evidence": evidence}, validator
    )
    record["target_validation"] = checks
    if not checks["sources_valid"]:
        record["label_status"] = f"{label_source}_has_mechanical_errors"
    sample = {
        "prompt": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": render_user(record)},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}
        ],
    }
    stats = {
        "task_id": task_id,
        "trial": trial.name,
        "base_commit": commit,
        "evidence_files": len(evidence),
        "evidence_lines": sum(item["line_end"] for item in evidence),
        "evidence_characters": sum(len(item["text"]) for item in evidence),
        "evidence_sha256": sha256(json_bytes(evidence)),
        "prompt_characters": sum(len(item["content"]) for item in sample["prompt"]),
        "completion_characters": len(sample["completion"][0]["content"]),
        "label_status": record["label_status"],
        "label_source": label_source,
        "selected_label_ledger": record["provenance"]["selected_label_ledger"],
        "selected_label_ledger_sha256": record["provenance"]["selected_label_ledger_sha256"],
        "target_schema_valid": checks["schema_valid"],
        "target_sources_valid": checks["sources_valid"],
        "target_validation_errors": len(checks["errors"]),
        "obligations": len(target["obligations"]),
        "preservation_obligations": len(target["preservation_obligations"]),
        "open_questions": len(target["open_questions"]),
        "skipped_reads": len(skipped_reads),
    }
    return record, sample, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data" / "f_remaining_25_v4")
    args = parser.parse_args()
    selection = json.loads((HERE / "selection.json").read_text())
    pilot_ids = {entry["task_id"] for entry in selection["tasks"]}
    batch = RESULTS_ROOT / selection["source_batch"]
    trials = []
    for trial in sorted(path for path in batch.iterdir() if path.is_dir()):
        config = json.loads((trial / "config.json").read_text())
        task_id = Path(config["task"]["path"]).name
        if task_id not in pilot_ids:
            trials.append((task_id, trial))
    trials.sort()
    if len(trials) != 25 or len({task_id for task_id, _ in trials}) != 25:
        raise ValueError("Expected exactly 25 distinct tasks outside the five-task pilot.")
    annotations = annotation_paths(trials)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir(exist_ok=True)
    prompt = (HERE / "f_prompt_v4.txt").read_text().strip()
    validator = Draft202012Validator(json.loads((HERE / "spec_output_schema.json").read_text()))
    built = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(build_record, task_id, trial, prompt, validator, annotations.get(task_id)): task_id
            for task_id, trial in trials
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                record, sample, stats = future.result()
                (records_dir / f"{task_id}.json").write_bytes(json_bytes(record))
                built[task_id] = (sample, stats)
                print(f"Built {task_id}: {stats['evidence_files']} files; {stats['label_source']}; {stats['target_validation_errors']} label issues", flush=True)
            except Exception as error:
                failures[task_id] = f"{type(error).__name__}: {error}"
                print(f"Failed {task_id}: {failures[task_id]}", flush=True)
    (args.output_dir / "build_failures.json").write_bytes(json_bytes(failures))
    if failures:
        raise SystemExit("Some inputs could not be built; see build_failures.json.")
    samples = [built[task_id][0] for task_id, _ in trials]
    stats = [built[task_id][1] for task_id, _ in trials]
    (args.output_dir / "training_candidates.jsonl").write_bytes(b"".join(json_bytes(sample) for sample in samples))
    (args.output_dir / "manifest.json").write_bytes(
        json_bytes(
            {
                "schema_version": "f_pilot_manifest_v1",
                "builder_sha256": sha256(Path(__file__).read_bytes()),
                "source_batch": selection["source_batch"],
                "split": "development_only",
                "status": "full_evidence_inputs_ready_labels_audited_semantics_ungraded",
                "prompt_version": "f_prompt_v4_v61_fixed_evidence",
                "prompt_sha256": sha256(prompt.encode()),
                "sample_count": len(samples),
                "excluded_pilot_task_ids": sorted(pilot_ids),
                "teacher_outputs_modified": False,
                "label_selection_policy": "Prefer existing human annotation; otherwise preserve the original teacher candidate. No automatic label correction or reward filtering.",
                "label_source_counts": {
                    "human_annotation": len(annotations),
                    "teacher_raw": len(samples) - len(annotations),
                },
                "tasks": stats,
            }
        )
    )


if __name__ == "__main__":
    main()
