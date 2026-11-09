"""Build five auditable fixed-evidence A-Agent SFT candidates from V6.1 runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "deepswe"
TASKS_ROOT = PROJECT_ROOT / ".benchmarks" / "deep-swe-clean-v4.1-taskgraph-clone" / "tasks"
OUTPUT_ROOT = HERE / "data" / "f_pilot_5"
BASELINE_CACHE = HERE / ".cache" / "baselines"

sys.path.insert(0, str(PROJECT_ROOT))
from external_agents_eval.workflows.solve_review_repair.v5_contracts import (  # noqa: E402
    validate_spec_output,
)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def ensure_baseline(task_id: str, url: str, commit: str) -> Path:
    repo = BASELINE_CACHE / task_id
    repo.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        git(repo, "init", "-q")
    if subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        git(repo, "fetch", "-q", "--depth=1", url, commit)
    actual = git(repo, "rev-parse", commit).decode().strip()
    if actual != commit:
        raise ValueError(f"Wrong baseline commit for {task_id}: {actual}")
    return repo


def spec_input(prompt_path: Path) -> dict[str, Any]:
    prompt = prompt_path.read_text()
    marker = "SPEC INPUT\n"
    if marker not in prompt:
        raise ValueError(f"No SPEC INPUT in {prompt_path}")
    body = prompt.split(marker, 1)[1].lstrip()
    value, _ = json.JSONDecoder().raw_decode(body)
    if not isinstance(value, dict) or not isinstance(value.get("task"), dict):
        raise ValueError(f"Invalid SPEC INPUT in {prompt_path}")
    return value


def raw_read_paths(trajectory_path: Path, base_root: str) -> list[str]:
    trajectory = json.loads(trajectory_path.read_text())
    prefix = base_root.rstrip("/") + "/"
    paths: set[str] = set()
    for step in trajectory["steps"]:
        for call in step.get("tool_calls", []):
            if call.get("function_name") != "read":
                continue
            path = call.get("arguments", {}).get("path", "")
            if not isinstance(path, str) or not path.startswith(prefix):
                continue
            relative = path[len(prefix) :]
            if not relative or any(part in {"", ".", ".."} for part in Path(relative).parts):
                raise ValueError(f"Invalid read path: {path}")
            paths.add(relative)
    if not paths:
        raise ValueError(f"No baseline read calls in {trajectory_path}")
    return sorted(paths)


def target_from_ledger(ledger_path: Path) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text())
    target = {"summary": ledger["summary"]}
    for field in ("obligations", "preservation_obligations", "open_questions"):
        target[field] = [entry["content"] for entry in ledger[field]]
    return validate_spec_output(target)


def repair_one_past_eof_sources(target: dict[str, Any], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record and correct the two observed teacher citations ending one line beyond EOF."""
    lengths = {item["path"]: item["line_end"] for item in evidence}
    corrections = []
    for field in ("obligations", "preservation_obligations", "open_questions"):
        for entry_index, entry in enumerate(target[field]):
            for source_index, source in enumerate(entry["sources"]):
                if source["kind"] != "repository" or source["path"] not in lengths:
                    continue
                last_line = lengths[source["path"]]
                if source["line_end"] == last_line + 1 and source["line_start"] <= last_line:
                    corrections.append(
                        {
                            "field": field,
                            "entry_index": entry_index,
                            "source_index": source_index,
                            "original": dict(source),
                            "corrected_line_end": last_line,
                            "reason": "Teacher citation ends one line past frozen baseline EOF; review required.",
                        }
                    )
                    source["line_end"] = last_line
    return corrections


def check_sources(record: dict[str, Any]) -> dict[str, int]:
    task = record["task"]
    evidence = {item["path"]: item for item in record["evidence"]}
    counts = {"task": 0, "repository": 0}
    for field in ("obligations", "preservation_obligations", "open_questions"):
        for entry in record["target"][field]:
            for source in entry["sources"]:
                kind = source["kind"]
                counts[kind] += 1
                if kind == "task":
                    if source["quote"] not in task[source["field"]]:
                        raise ValueError(f"Noncontiguous task quote in {record['task_id']}: {source}")
                else:
                    item = evidence.get(source["path"])
                    if item is None or not (1 <= source["line_start"] <= source["line_end"] <= item["line_end"]):
                        raise ValueError(f"Repository source missing from evidence in {record['task_id']}: {source}")
                    lines = item["text"].splitlines()
                    if not any(line.strip() for line in lines[source["line_start"] - 1 : source["line_end"]]):
                        raise ValueError(f"Empty cited range in {record['task_id']}: {source}")
    return counts


def render_user(record: dict[str, Any]) -> str:
    task_input = {"task": record["task"], "managed_delivery": record["managed_delivery"]}
    chunks = ["ORIGINAL TASK\n" + json.dumps(task_input, ensure_ascii=False, indent=2)]
    chunks.append("FROZEN PUBLIC REPOSITORY EVIDENCE")
    for item in record["evidence"]:
        lines = item["text"].splitlines()
        numbered = "\n".join(f"{index} {line}" for index, line in enumerate(lines, 1))
        chunks.append(f"FILE {item['path']} (baseline lines 1-{item['line_end']})\n{numbered}")
    return "\n\n".join(chunks)


def build_one(batch: Path, task_id: str, reason: str, system_prompt: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trials = list(batch.glob(task_id + "__*"))
    if len(trials) != 1:
        raise ValueError(f"Expected one trial for {task_id}; found {len(trials)}")
    trial = trials[0]
    task_dir = TASKS_ROOT / task_id
    metadata = tomllib.loads((task_dir / "task.toml").read_text())["metadata"]
    repo_url = metadata["repository_url"]
    commit = metadata["base_commit_hash"]
    a_root = trial / "v6_review_repair" / "v5"
    prompt_path = a_root / "specification" / "attempt-1" / "prompt.md"
    trajectory_path = a_root / "specification" / "attempt-1" / "trajectory.json"
    ledger_path = a_root / "ledger.json"
    a_input = spec_input(prompt_path)
    baseline = ensure_baseline(task_id, repo_url, commit)
    paths = raw_read_paths(trajectory_path, a_input["base_root"])
    evidence = []
    for path in paths:
        content = git(baseline, "show", f"{commit}:{path}")
        text = content.decode("utf-8")
        lines = text.splitlines()
        evidence.append(
            {
                "path": path,
                "line_start": 1,
                "line_end": len(lines),
                "sha256": sha256(content),
                "text": text,
            }
        )
    target = target_from_ledger(ledger_path)
    source_corrections = repair_one_past_eof_sources(target, evidence)
    record = {
        "schema_version": "f_pilot_v1",
        "task_id": task_id,
        "split": "development_only",
        "label_status": "teacher_candidate_requires_semantic_review",
        "selection_reason": reason,
        "repository_url": repo_url,
        "base_commit": commit,
        "task": a_input["task"],
        "managed_delivery": a_input.get("managed_delivery", []),
        "evidence_rule": "Full baseline text of each unique repository file opened by A's read tool, sorted by relative path; no target-guided snippet selection.",
        "evidence": evidence,
        "target": target,
        "source_corrections": source_corrections,
        "provenance": {
            "source_trial": str(trial.relative_to(PROJECT_ROOT)),
            "spec_prompt": str(prompt_path.relative_to(PROJECT_ROOT)),
            "a_trajectory": str(trajectory_path.relative_to(PROJECT_ROOT)),
            "teacher_ledger": str(ledger_path.relative_to(PROJECT_ROOT)),
        },
    }
    counts = check_sources(record)
    sample = {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": render_user(record)},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(record["target"], ensure_ascii=False, separators=(",", ":"))}
        ],
    }
    # No task ID, teacher output, evaluation result, or provenance in the student prompt.
    stats = {
        "task_id": task_id,
        "trial": trial.name,
        "base_commit": commit,
        "evidence_files": len(evidence),
        "evidence_characters": sum(len(item["text"]) for item in evidence),
        "prompt_characters": sum(len(item["content"]) for item in sample["prompt"]),
        "completion_characters": len(sample["completion"][0]["content"]),
        "obligations": len(record["target"]["obligations"]),
        "preservation_obligations": len(record["target"]["preservation_obligations"]),
        "open_questions": len(record["target"]["open_questions"]),
        "source_counts": counts,
        "source_corrections": len(source_corrections),
        "evidence_sha256": sha256(json_bytes(evidence)),
    }
    return record, sample, stats


def main() -> None:
    selection = json.loads((HERE / "selection.json").read_text())
    batch = RESULTS_ROOT / selection["source_batch"]
    reward_stats = json.loads((batch / "result.json").read_text())["stats"]["evals"][
        "pi-v6-review-repair__deepseek-flash__tasks"
    ]["reward_stats"]["reward"]
    passed = set(reward_stats["1"])
    system_prompt = (HERE / "f_prompt_v1.txt").read_text().strip()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records_dir = OUTPUT_ROOT / "records"
    records_dir.mkdir(exist_ok=True)
    samples = []
    stats = []
    for item in selection["tasks"]:
        task_id = item["task_id"]
        if not any(name.startswith(task_id + "__") for name in passed):
            raise ValueError(f"Selected task did not pass original batch: {task_id}")
        record, sample, task_stats = build_one(batch, task_id, item["reason"], system_prompt)
        (records_dir / f"{task_id}.json").write_bytes(json_bytes(record))
        samples.append(sample)
        stats.append(task_stats)
        print(f"Built {task_id}: {task_stats['evidence_files']} evidence files; {task_stats['prompt_characters']} prompt chars")
    (OUTPUT_ROOT / "training_candidates.jsonl").write_bytes(b"".join(json_bytes(item) for item in samples))
    manifest = {
        "schema_version": "f_pilot_manifest_v1",
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "source_batch": selection["source_batch"],
        "split": "development_only",
        "status": "structurally_valid_teacher_candidates_not_semantically_approved",
        "prompt_version": "f_prompt_v1",
        "prompt_sha256": sha256(system_prompt.encode()),
        "sample_count": len(samples),
        "tasks": stats,
    }
    (OUTPUT_ROOT / "manifest.json").write_bytes(json_bytes(manifest))


if __name__ == "__main__":
    main()
