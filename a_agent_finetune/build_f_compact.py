"""Create source-auditable 32K/64K training variants of the F pilot candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from build_f_pilot import HERE, OUTPUT_ROOT, spec_input


LIMITS = {"32k": 31_000, "64k": 63_000}  # Reserve space for model chat-template tokens.
CHUNK_LINES = 100


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def add_span(spans: dict[str, list[tuple[int, int]]], path: str, start: int, end: int) -> bool:
    old = spans.get(path, [])
    new = merge([*old, (start, end)])
    if new == old:
        return False
    spans[path] = new
    return True


def source_anchors(record: dict, file_lines: dict[str, list[str]]) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {}
    for field in ("obligations", "preservation_obligations", "open_questions"):
        for entry in record["target"][field]:
            for source in entry["sources"]:
                if source["kind"] != "repository":
                    continue
                path = source["path"]
                start = max(1, source["line_start"] - 8)
                end = min(len(file_lines[path]), source["line_end"] + 8)
                add_span(spans, path, start, end)
    return spans


def read_candidates(record: dict, file_lines: dict[str, list[str]]) -> list[tuple[str, int, int]]:
    prompt_path = HERE.parent / record["provenance"]["spec_prompt"]
    base_root = spec_input(prompt_path)["base_root"].rstrip("/") + "/"
    trajectory_path = HERE.parent / record["provenance"]["a_trajectory"]
    trajectory = json.loads(trajectory_path.read_text())
    focused = []
    broad = []
    for step in trajectory["steps"]:
        for call in step.get("tool_calls", []):
            if call.get("function_name") != "read":
                continue
            args = call.get("arguments", {})
            path = args.get("path", "")
            if not isinstance(path, str) or not path.startswith(base_root):
                continue
            path = path[len(base_root) :]
            if path not in file_lines:
                continue
            line_count = len(file_lines[path])
            start = max(1, int(args.get("offset", 1)))
            end = min(line_count, start + int(args["limit"]) - 1) if "limit" in args else line_count
            if start > end:
                continue
            if "limit" in args and int(args["limit"]) <= 120:
                focused.append((path, max(1, start - 4), min(line_count, end + 4)))
            else:
                for chunk_start in range(start, end + 1, CHUNK_LINES):
                    broad.append((path, chunk_start, min(end, chunk_start + CHUNK_LINES - 1)))
    # Focused A reads get priority. The remaining chunks follow A's read order.
    return list(dict.fromkeys([*focused, *broad]))


def make_evidence(spans: dict[str, list[tuple[int, int]]], full: dict) -> list[dict]:
    files = {item["path"]: item for item in full["evidence"]}
    evidence = []
    for path in sorted(spans):
        lines = files[path]["text"].splitlines(keepends=True)
        for start, end in spans[path]:
            text = "".join(lines[start - 1 : end])
            evidence.append(
                {
                    "path": path,
                    "line_start": start,
                    "line_end": end,
                    "full_file_sha256": files[path]["sha256"],
                    "text": text,
                }
            )
    return evidence


def render_user(record: dict, evidence: list[dict]) -> str:
    task_input = {"task": record["task"], "managed_delivery": record["managed_delivery"]}
    chunks = ["ORIGINAL TASK\n" + json.dumps(task_input, ensure_ascii=False, indent=2)]
    chunks.append("FROZEN PUBLIC REPOSITORY EVIDENCE")
    for item in evidence:
        numbered = "\n".join(
            f"{line_no} {line}" for line_no, line in enumerate(item["text"].splitlines(), item["line_start"])
        )
        chunks.append(f"FILE {item['path']} (baseline lines {item['line_start']}-{item['line_end']})\n{numbered}")
    return "\n\n".join(chunks)


def sample_for(record: dict, spans: dict[str, list[tuple[int, int]]], system: str) -> tuple[dict, list[dict]]:
    evidence = make_evidence(spans, record)
    sample = {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": render_user(record, evidence)},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(record["target"], ensure_ascii=False, separators=(",", ":"))}
        ],
    }
    return sample, evidence


def tokens_for(sample: dict, tokenizer) -> int:
    return sum(
        len(tokenizer.encode(message["content"], add_special_tokens=False).ids)
        for field in ("prompt", "completion")
        for message in sample[field]
    )


def check_evidence_sources(record: dict, evidence: list[dict]) -> None:
    for field in ("obligations", "preservation_obligations", "open_questions"):
        for entry in record["target"][field]:
            for source in entry["sources"]:
                if source["kind"] != "repository":
                    continue
                if not any(
                    item["path"] == source["path"]
                    and item["line_start"] <= source["line_start"]
                    and source["line_end"] <= item["line_end"]
                    for item in evidence
                ):
                    raise ValueError(f"Citation missing from compact evidence: {record['task_id']} {source}")


def fill(
    record: dict,
    initial: dict[str, list[tuple[int, int]]],
    candidates: list[tuple[str, int, int]],
    system: str,
    tokenizer,
    budget: int,
) -> tuple[dict[str, list[tuple[int, int]]], dict, list[dict], int]:
    spans = copy.deepcopy(initial)
    sample, evidence = sample_for(record, spans, system)
    total = tokens_for(sample, tokenizer)
    if total > budget:
        raise ValueError(f"Mandatory source anchors exceed {budget} text tokens: {record['task_id']}")
    for path, start, end in candidates:
        proposal = copy.deepcopy(spans)
        if not add_span(proposal, path, start, end):
            continue
        trial_sample, trial_evidence = sample_for(record, proposal, system)
        trial_tokens = tokens_for(trial_sample, tokenizer)
        if trial_tokens <= budget:
            spans, sample, evidence, total = proposal, trial_sample, trial_evidence, trial_tokens
    check_evidence_sources(record, evidence)
    return spans, sample, evidence, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    try:
        from tokenizers import Tokenizer, __version__ as tokenizers_version
    except ImportError as exc:
        raise SystemExit("Install Python tokenizers to build compact samples") from exc
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    system = (HERE / "f_prompt_v1.txt").read_text().strip()
    selected = json.loads((HERE / "selection.json").read_text())["tasks"]
    records = [
        json.loads((OUTPUT_ROOT / "records" / f"{item['task_id']}.json").read_text())
        for item in selected
    ]
    data = {"32k": [], "64k": []}
    compact_records = {"32k": [], "64k": []}
    stats = []
    for record in records:
        file_lines = {item["path"]: item["text"].splitlines() for item in record["evidence"]}
        anchors = source_anchors(record, file_lines)
        candidates = read_candidates(record, file_lines)
        spans_32, sample_32, evidence_32, count_32 = fill(
            record, anchors, candidates, system, tokenizer, LIMITS["32k"]
        )
        _, sample_64, evidence_64, count_64 = fill(
            record, spans_32, candidates, system, tokenizer, LIMITS["64k"]
        )
        for size, sample, evidence in (
            ("32k", sample_32, evidence_32),
            ("64k", sample_64, evidence_64),
        ):
            data[size].append(sample)
            compact_records[size].append(
                {
                    "task_id": record["task_id"],
                    "split": "development_only",
                    "label_status": record["label_status"],
                    "base_commit": record["base_commit"],
                    "evidence_rule": "Teacher read windows plus source-guided anchors; training-pilot only, not a heldout evidence builder.",
                    "evidence": evidence,
                    "target": record["target"],
                    "full_record": f"../records/{record['task_id']}.json",
                }
            )
        stats.append(
            {
                "task_id": record["task_id"],
                "32k_text_tokens": count_32,
                "64k_text_tokens": count_64,
                "32k_snippets": len(evidence_32),
                "64k_snippets": len(evidence_64),
            }
        )
        print(f"{record['task_id']}: 32k={count_32}; 64k={count_64}", flush=True)
    for size in LIMITS:
        output = OUTPUT_ROOT / f"training_candidates_{size}.jsonl"
        output.write_text("".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in data[size]))
        folder = OUTPUT_ROOT / f"records_{size}"
        folder.mkdir(exist_ok=True)
        for item in compact_records[size]:
            (folder / f"{item['task_id']}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
    (OUTPUT_ROOT / "compact_manifest.json").write_text(
        json.dumps(
            {
                "status": "teacher_candidates_not_semantically_approved",
                "model": args.model,
                "model_revision": args.revision,
                "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                "tokenizers_package_version": tokenizers_version,
                "counting_rule": "Message content only; model chat-template overhead excluded.",
                "max_text_tokens": LIMITS,
                "method": "All target repository citations retained with +/-8 baseline lines, then A's bounded reads (<=120 lines) and broad reads in trajectory order, greedily up to budget. The source-guided anchor step makes these development/training candidates only.",
                "tasks": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
