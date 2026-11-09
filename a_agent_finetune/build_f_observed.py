"""Build all 30 F samples using only frozen repository text actually returned to A."""

from __future__ import annotations

import argparse
import copy
import json
import posixpath
import re
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

from build_f_compact import merge, render_user
from build_f_pilot import (
    BASELINE_CACHE, HERE, PROJECT_ROOT, RESULTS_ROOT, TASKS_ROOT,
    ensure_baseline, git, json_bytes, sha256, spec_input,
)
from build_f_remaining import annotation_paths, ledger_target, resolve_commit
from run_f_inference import validate_output

FOOTER = re.compile(
    r"\n\n\[(?:Showing lines [^\n]+|\d+ more lines in file\. Use offset=\d+ to continue\.)\]$"
)
FIELDS = ("obligations", "preservation_obligations", "open_questions")
LABEL_REVIEW_NOTES = {
    "numba-stencil-boundary-modes": [{
        "field": "preservation_obligations", "entry_index": 9,
        "status": "wording_requires_human_review",
        "note": "任务明确要求 llvmlite 0.46.0，但冻结基线的最低版本为 0.47.0；人工标签中的‘最低版本要求不得改变’存在参照对象歧义。已删除环境修改行引用，未改动标签正文。",
    }],
    "obsidian-linter-auto-table-of-contents": [{
        "field": "obligations", "entry_index": 26,
        "status": "advisory_task_interpretation",
        "note": "空 TOC 用一行留白是对任务留白要求的 advisory 解释；blockquote 实现不是该行为的公开契约，已移除其引用。",
    }],
}


def returned_span(content: str, arguments: dict, baseline: str) -> tuple[int, int]:
    """Honor the actual read return, including Pi's byte cap and EOF conventions."""
    lines = baseline.splitlines()
    start = int(arguments.get("offset", 1))
    footer = FOOTER.search(content)
    if footer:
        end = int(re.search(r"Use offset=(\d+)", footer.group()).group(1)) - 1
        body = content[:footer.start()]
    else:
        end = start + int(arguments.get("limit", len(lines))) - 1
        body = content
    end = min(end, len(lines))
    if start > end:
        raise ValueError("Read returned no baseline lines")
    expected = "\n".join(lines[start - 1:end])
    if body.rstrip("\n") != expected.rstrip("\n"):
        raise ValueError(f"Tool return differs from frozen baseline at lines {start}-{end}")
    return start, end


def build_record(task_id: str, trial: Path, annotation: Path | None) -> dict:
    metadata = tomllib.loads((TASKS_ROOT / task_id / "task.toml").read_text())["metadata"]
    revision = metadata["base_commit_hash"]
    cached = BASELINE_CACHE / task_id
    commit = git(cached, "rev-parse", revision).decode().strip() if (cached / ".git").exists() else resolve_commit(metadata["repository_url"], revision)
    repository = ensure_baseline(task_id, metadata["repository_url"], commit)
    tree = {}
    for line in git(repository, "ls-tree", "-r", commit).decode().splitlines():
        metadata_fields, path = line.split("\t", 1)
        tree[path] = metadata_fields.split()[0]
    tracked = set(tree)
    a_root = trial / "v6_review_repair/v5"
    prompt_path = a_root / "specification/attempt-1/prompt.md"
    trajectory_path = prompt_path.parent / "trajectory.json"
    a_input = spec_input(prompt_path)
    prefix = a_input["base_root"].rstrip("/") + "/"
    trajectory = json.loads(trajectory_path.read_text())
    files: dict[str, bytes] = {}
    spans = defaultdict(list)
    reads = []
    skipped = []
    tool_counts = Counter()
    inventories = []
    environment_exclusions = []
    for step in trajectory["steps"]:
        results = {result["source_call_id"]: result["content"] for result in step.get("observation", {}).get("results", [])}
        for call in step.get("tool_calls", []):
            name = call["function_name"]
            tool_counts[name] += 1
            arguments = call["arguments"]
            content = results[call["tool_call_id"]]
            if name in {"ls", "find"}:
                inventories.append({"tool": name, "arguments": arguments, "returned_content": content, "tool_call_id": call["tool_call_id"]})
            if name != "read":
                continue
            absolute = arguments.get("path", "")
            if not absolute.startswith(prefix):
                skipped.append({"path": absolute, "reason": "Outside the declared frozen base_root", "tool_call_id": call["tool_call_id"]})
                continue
            path = absolute[len(prefix):]
            if path not in tracked:
                skipped.append({"path": path, "reason": "Not a tracked file in the frozen baseline", "tool_call_id": call["tool_call_id"]})
                continue
            requested_path = path
            if tree[path] == "120000":
                destination = git(repository, "show", f"{commit}:{path}").decode().strip()
                path = posixpath.normpath(posixpath.join(posixpath.dirname(path), destination))
                if path not in tracked or tree[path] == "120000":
                    raise ValueError(f"Unresolved baseline symlink: {requested_path} -> {path}")
            if path not in files:
                files[path] = git(repository, "show", f"{commit}:{path}")
            if int(arguments.get("offset", 1)) > len(files[path].decode().splitlines()):
                skipped.append({"path": path, "reason": "Empty file or requested offset beyond EOF; no source lines returned", "tool_call_id": call["tool_call_id"]})
                continue
            baseline = files[path].decode()
            try:
                start, end = returned_span(content, arguments, baseline)
                spans[path].append((start, end))
            except ValueError:
                # This batch contains one documented environment compatibility edit.
                original = "_min_llvmlite_version = (0, 47, 0)"
                patched = "_min_llvmlite_version = (0, 46, 0)"
                if task_id != "numba-stencil-boundary-modes" or path != "numba/__init__.py" or original not in baseline:
                    raise
                start, end = returned_span(content, arguments, baseline.replace(original, patched))
                changed_line = baseline.splitlines().index(original) + 1
                for first, last in ((start, changed_line - 1), (changed_line + 1, end)):
                    if first <= last:
                        spans[path].append((first, last))
                environment_exclusions.append({"path": path, "line": changed_line, "baseline": original, "observed": patched, "reason": "Environment dependency compatibility edit; excluded from frozen public evidence", "tool_call_id": call["tool_call_id"]})
            reads.append({"path": path, "requested_path": requested_path, "line_start": start, "line_end": end, "arguments": arguments, "step_id": step["step_id"], "tool_call_id": call["tool_call_id"], "returned_content_sha256": sha256(content.encode())})
    evidence = []
    for path, intervals in sorted(spans.items()):
        raw_lines = files[path].decode().splitlines(keepends=True)
        for start, end in merge(intervals):
            text = "".join(raw_lines[start - 1:end])
            evidence.append({"path": path, "line_start": start, "line_end": end, "text": text, "sha256": sha256(text.encode()), "full_file_sha256": sha256(files[path])})
    original_ledger = a_root / "ledger.json"
    selected = annotation or original_ledger
    target = ledger_target(selected)
    return {
        "schema_version": "f_observed_v1", "task_id": task_id, "split": "development_only",
        "repository_url": metadata["repository_url"], "base_commit": commit,
        "task": a_input["task"], "managed_delivery": a_input.get("managed_delivery", []),
        "evidence_rule": "Only baseline text actually returned by A's read calls under frozen base_root; verify actual responses against baseline, honor truncation, merge overlapping/adjacent ranges and sort by path/line. No label-guided evidence selection or supplementary files.",
        "evidence": evidence, "read_provenance": reads, "skipped_reads": skipped,
        "tool_counts": dict(tool_counts), "repository_inventory_provenance": inventories,
        "environment_modified_lines_excluded": environment_exclusions,
        "label_source": "human_annotation" if annotation else "teacher_raw",
        "target_original": copy.deepcopy(target), "target": target, "source_corrections": [],
        "provenance": {
            "source_trial": str(trial.relative_to(PROJECT_ROOT)),
            "spec_prompt": str(prompt_path.relative_to(PROJECT_ROOT)),
            "a_trajectory": str(trajectory_path.relative_to(PROJECT_ROOT)),
            "a_trajectory_sha256": sha256(trajectory_path.read_bytes()),
            "teacher_ledger": str(original_ledger.relative_to(PROJECT_ROOT)),
            "teacher_ledger_sha256": sha256(original_ledger.read_bytes()),
            "selected_label_ledger": str(selected.relative_to(PROJECT_ROOT)),
            "selected_label_ledger_sha256": sha256(selected.read_bytes()),
        },
    }


def apply_corrections(record: dict, corrections: list[dict]) -> None:
    for correction in sorted(corrections, key=lambda item: (item["field"], item["entry_index"], -item["source_index"])):
        entry = record["target"][correction["field"]][correction["entry_index"]]
        source = entry["sources"][correction["source_index"]]
        if source != correction["before"]:
            raise ValueError(f"Correction no longer matches original source: {record['task_id']}")
        if correction["after"] is None:
            entry["sources"].pop(correction["source_index"])
        else:
            entry["sources"][correction["source_index"]] = correction["after"]
        record["source_corrections"].append(correction)


def reference_audit(record: dict) -> list[dict]:
    """Bind every original citation to its final location or documented removal."""
    corrections = {
        (item["field"], item["entry_index"], item["source_index"]): item
        for item in record["source_corrections"]
    }
    rows = []
    for field in FIELDS:
        for entry_index, entry in enumerate(record["target_original"][field]):
            for source_index, original in enumerate(entry["sources"]):
                correction = corrections.get((field, entry_index, source_index))
                source = correction["after"] if correction else original
                row = {
                    "field": field, "entry_index": entry_index, "source_index": source_index,
                    "entry_text": entry.get("statement", entry.get("question")),
                    "original_source": original, "final_source": source,
                    "action": "removed" if source is None else "corrected" if correction else "unchanged",
                    "correction_reason": correction["reason"] if correction else None,
                }
                if source is None:
                    row["check"] = "removed_with_reason"
                elif source["kind"] == "task":
                    row["check"] = "exact_task_quote" if source["quote"] in record["task"][source["field"]] else "invalid_task_quote"
                else:
                    evidence = next((item for item in record["evidence"] if
                        item["path"] == source["path"] and
                        item["line_start"] <= source["line_start"] <= source["line_end"] <= item["line_end"]
                    ), None)
                    row["check"] = "visible_baseline_range" if evidence else "missing_evidence"
                    if evidence:
                        first = source["line_start"] - evidence["line_start"]
                        last = source["line_end"] - evidence["line_start"] + 1
                        row["excerpt"] = "\n".join(
                            f"{number} {line}" for number, line in enumerate(
                                evidence["text"].splitlines()[first:last], source["line_start"]
                            )
                        )
                        row["evidence_span_sha256"] = evidence["sha256"]
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data/f_teacher_observed_30_v4")
    parser.add_argument("--corrections", type=Path, default=HERE / "observed_citation_corrections.json")
    args = parser.parse_args()
    selection = json.loads((HERE / "selection.json").read_text())
    batch = RESULTS_ROOT / selection["source_batch"]
    trials = []
    for trial in batch.iterdir():
        if trial.is_dir():
            task_id = Path(json.loads((trial / "config.json").read_text())["task"]["path"]).name
            trials.append((task_id, trial))
    trials.sort()
    if len(trials) != 30 or len({task for task, _ in trials}) != 30:
        raise ValueError("Expected exactly 30 unique tasks")
    annotations = annotation_paths(trials)
    corrections = json.loads(args.corrections.read_text()) if args.corrections.exists() else {}
    system = (HERE / "f_prompt_v4.txt").read_text().strip()
    validator = Draft202012Validator(json.loads((HERE / "spec_output_schema.json").read_text()))
    records_dir = args.output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "labels").mkdir(exist_ok=True)
    samples = []
    stats = []
    audits = []
    for task_id, trial in trials:
        record = build_record(task_id, trial, annotations.get(task_id))
        apply_corrections(record, corrections.get(task_id, []))
        _, checks = validate_output(json.dumps(record["target"]), record, validator)
        record["target_validation"] = checks
        record["label_status"] = "references_verified_label_semantics_not_fully_audited" if checks["sources_valid"] else "references_require_correction"
        record["label_review_notes"] = LABEL_REVIEW_NOTES.get(task_id, [])
        sample = {"prompt": [{"role": "system", "content": system}, {"role": "user", "content": render_user(record, record["evidence"])}], "completion": [{"role": "assistant", "content": json.dumps(record["target"], ensure_ascii=False, separators=(",", ":"))}]}
        (records_dir / f"{task_id}.json").write_bytes(json_bytes(record))
        (args.output_dir / "labels" / f"{task_id}.json").write_bytes(json_bytes(record["target"]))
        audits.append({"task_id": task_id, "record_sha256": sha256(json_bytes(record)), "label_review_notes": record["label_review_notes"], "references": reference_audit(record)})
        samples.append(sample)
        stats.append({"task_id": task_id, "base_commit": record["base_commit"], "label_source": record["label_source"], "evidence_spans": len(record["evidence"]), "evidence_files": len({item['path'] for item in record['evidence']}), "evidence_lines": sum(item["line_end"] - item["line_start"] + 1 for item in record["evidence"]), "evidence_sha256": sha256(json_bytes(record["evidence"])), "target_sources_valid": checks["sources_valid"], "validation_errors": len(checks["errors"]), "source_corrections": len(record["source_corrections"])})
        print(f"{task_id}: {stats[-1]['evidence_lines']} visible lines; {len(checks['errors'])} citation errors", flush=True)
    (args.output_dir / "training_candidates.jsonl").write_bytes(b"".join(json_bytes(row) for row in samples))
    (args.output_dir / "citation_audit.json").write_bytes(json_bytes({
        "schema_version": "f_observed_citation_audit_v1",
        "indexing": "entry_index and source_index are zero-based indices into target_original",
        "scope": "All original references checked for exact task quotes or visible frozen-baseline ranges; repository locators and their claim context reviewed and obvious irrelevant/misplaced anchors corrected. This is not independent verification of every obligation's correctness or completeness.",
        "original_source_counts": dict(Counter(row["original_source"]["kind"] for task in audits for row in task["references"])),
        "final_source_counts": dict(Counter(row["final_source"]["kind"] for task in audits for row in task["references"] if row["final_source"])),
        "actions": dict(Counter(row["action"] for task in audits for row in task["references"])),
        "tasks": audits,
    }))
    (args.output_dir / "manifest.json").write_bytes(json_bytes({"schema_version": "f_observed_manifest_v1", "source_batch": selection["source_batch"], "sample_count": 30, "split": "development_only", "evidence_condition": "teacher_observed_only", "prompt_version": "f_prompt_v4_v61_fixed_evidence", "prompt_sha256": sha256(system.encode()), "builder_sha256": sha256(Path(__file__).read_bytes()), "corrections_sha256": sha256(args.corrections.read_bytes()) if args.corrections.exists() else None, "label_source_counts": dict(Counter(item['label_source'] for item in stats)), "tasks": stats}))


if __name__ == "__main__":
    main()
