"""Run fixed-evidence A samples against an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
FIELDS = ("obligations", "preservation_obligations", "open_questions")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"Non-JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def validate_output(
    text: str, sample: dict[str, Any], validator: Draft202012Validator
) -> tuple[Any, dict[str, Any]]:
    checks = {
        "json_valid": False,
        "schema_valid": False,
        "sources_valid": False,
        "semantic_support_checked": False,
        "errors": [],
    }
    try:
        output = strict_json(text)
    except ValueError as error:
        checks["errors"].append({"check": "json", "message": str(error)})
        return None, checks
    checks["json_valid"] = True
    for error in validator.iter_errors(output):
        checks["errors"].append(
            {
                "check": "schema",
                "path": "/".join(str(part) for part in error.absolute_path),
                "message": error.message,
            }
        )
    if checks["errors"]:
        return output, checks

    # Match the non-empty text and duplicate checks in validate_spec_output.
    def check_text(value: Any, path: str = "") -> None:
        if isinstance(value, str) and not value.strip():
            checks["errors"].append({"check": "schema", "path": path, "message": "Blank string"})
        elif isinstance(value, dict):
            for key, child in value.items():
                check_text(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_text(child, f"{path}/{index}")

    check_text(output)
    statements = [
        " ".join(item["statement"].casefold().split())
        for field in FIELDS[:2]
        for item in output[field]
    ]
    if len(statements) != len(set(statements)):
        checks["errors"].append({"check": "schema", "message": "Duplicate O/P statements"})
    for field in FIELDS:
        for index, item in enumerate(output[field]):
            sources = [json.dumps(source, sort_keys=True) for source in item["sources"]]
            if len(sources) != len(set(sources)):
                checks["errors"].append(
                    {
                        "check": "schema",
                        "path": f"{field}/{index}/sources",
                        "message": "Duplicate sources",
                    }
                )
            for source_index, source in enumerate(item["sources"]):
                if source["kind"] == "repository" and (
                    type(source["line_start"]) is not int
                    or type(source["line_end"]) is not int
                    or source["line_end"] < source["line_start"]
                ):
                    checks["errors"].append(
                        {
                            "check": "schema",
                            "path": f"{field}/{index}/sources/{source_index}",
                            "message": "Invalid integer line range",
                        }
                    )
    checks["schema_valid"] = not checks["errors"]
    if not checks["schema_valid"]:
        return output, checks

    for field in FIELDS:
        for index, item in enumerate(output[field]):
            for source_index, source in enumerate(item["sources"]):
                path = f"{field}/{index}/sources/{source_index}"
                if source["kind"] == "task":
                    valid = source["quote"] in sample["task"][source["field"]]
                    message = "Quote is not contiguous text in the original task field"
                else:
                    valid = False
                    for evidence in sample["evidence"]:
                        if (
                            evidence["path"] == source["path"]
                            and evidence["line_start"] <= source["line_start"]
                            and source["line_end"] <= evidence["line_end"]
                        ):
                            lines = evidence["text"].splitlines()
                            start = source["line_start"] - evidence["line_start"]
                            end = source["line_end"] - evidence["line_start"] + 1
                            valid = any(line.strip() for line in lines[start:end])
                            if valid:
                                break
                    message = "Repository range is absent from or empty in the supplied evidence"
                if not valid:
                    checks["errors"].append({"check": "source", "path": path, "message": message})
    checks["sources_valid"] = not checks["errors"]
    return output, checks


def load_samples(data_dir: Path, variant: str) -> tuple[list[dict[str, Any]], Path]:
    suffix = "" if variant == "full" else f"_{variant}"
    data_file = data_dir / f"training_candidates{suffix}.jsonl"
    rows = [
        json.loads(line)
        for line in data_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = []
    for entry, row in zip(manifest["tasks"], rows, strict=True):
        task_id = entry["task_id"]
        if Path(task_id).name != task_id or task_id in {".", ".."}:
            raise ValueError(f"Invalid task ID: {task_id}")
        full = json.loads((data_dir / "records" / f"{task_id}.json").read_text(encoding="utf-8"))
        record = (
            full
            if variant == "full"
            else json.loads(
                (data_dir / f"records_{variant}" / f"{task_id}.json").read_text(encoding="utf-8")
            )
        )
        if full["task_id"] != task_id or record["task_id"] != task_id:
            raise ValueError(f"Task ID does not match record: {task_id}")
        # Bind metadata/anchors to the exact input being sent, without using the completion.
        task_input = {"task": full["task"], "managed_delivery": full["managed_delivery"]}
        chunks = [
            "ORIGINAL TASK\n" + json.dumps(task_input, ensure_ascii=False, indent=2),
            "FROZEN PUBLIC REPOSITORY EVIDENCE",
        ]
        for evidence in record["evidence"]:
            numbered = "\n".join(
                f"{n} {line}"
                for n, line in enumerate(evidence["text"].splitlines(), evidence["line_start"])
            )
            chunks.append(
                f"FILE {evidence['path']} (baseline lines {evidence['line_start']}-{evidence['line_end']})\n{numbered}"
            )
        expected_user = "\n\n".join(chunks)
        if [message["role"] for message in row["prompt"]] != ["system", "user"] or row["prompt"][1][
            "content"
        ] != expected_user:
            raise ValueError(f"Prompt does not match task/evidence record: {task_id}")
        samples.append(
            {
                "task_id": task_id,
                "task": full["task"],
                "evidence": record["evidence"],
                "base_commit": record["base_commit"],
                "messages": row["prompt"],
            }
        )
    return samples, data_file


def run_one(
    sample: dict[str, Any], args: argparse.Namespace, validator: Draft202012Validator
) -> dict[str, Any]:
    task_dir = args.output_dir / sample["task_id"]
    task_dir.mkdir()
    request_body = {
        "model": args.model,
        "messages": sample["messages"],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    write_json(task_dir / "request.json", request_body)
    result = {
        "task_id": sample["task_id"],
        "base_commit": sample["base_commit"],
        "status": "request_error",
        "usage": {},
        "finish_reason": None,
    }
    start = time.perf_counter()
    try:
        request = Request(
            args.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ.get("F_INFERENCE_API_KEY", "EMPTY"),
            },
        )
        try:
            with urlopen(request, timeout=args.timeout) as response:
                raw = response.read().decode("utf-8")
                result["http_status"] = response.status
        except HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            (task_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
            result["http_status"] = error.code
            raise ValueError(f"HTTP {error.code}; see raw_response.txt") from error
        (task_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
        response = json.loads(raw)
        write_json(task_dir / "response.json", response)
        result["usage"] = response.get("usage") or {}
        result["returned_model"] = response.get("model")
        result["response_id"] = response.get("id")
        choice = response["choices"][0]
        result["finish_reason"] = choice.get("finish_reason")
        result["truncated"] = result["finish_reason"] == "length"
        content = choice["message"].get("content")
        if not isinstance(content, str):
            raise TypeError("Response has no text content")
        (task_dir / "output.txt").write_text(content, encoding="utf-8")
        output, checks = validate_output(content, sample, validator)
        write_json(task_dir / "validation.json", checks)
        result["validation"] = checks
        if output is not None:
            write_json(task_dir / "spec_output.json", output)
        if result["truncated"]:
            result["status"] = "truncated"
        elif result["finish_reason"] != "stop":
            result["status"] = "incomplete"
        elif checks["sources_valid"]:
            result["status"] = "passed"
        else:
            result["status"] = "invalid_output"
    except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    write_json(task_dir / "result.json", result)
    return result


def make_summary(results: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    return {
        "planned_tasks": planned,
        "completed_tasks": len(results),
        "passed": sum(row["status"] == "passed" for row in results),
        "request_errors": sum(row["status"] == "request_error" for row in results),
        "truncated": sum(row["status"] == "truncated" for row in results),
        "incomplete": sum(row["status"] == "incomplete" for row in results),
        "invalid_outputs": sum(row["status"] == "invalid_output" for row in results),
        **{
            key: sum(row.get("validation", {}).get(key, False) for row in results)
            for key in ("json_valid", "schema_valid", "sources_valid")
        },
        "usage": {
            key: sum(row["usage"].get(key) or 0 for row in results)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in results), 3),
        "semantic_support_checked": False,
        "tasks": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data" / "f_pilot_5")
    parser.add_argument("--variant", choices=("32k", "64k", "full"), default="32k")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--limit", type=int, help="Run only the first N tasks")
    parser.add_argument("--task-id", help="Run one named task from the selected dataset")
    parser.add_argument("--exclude-task-id", action="append", default=[], help="Exclude a completed task; may be repeated")
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.timeout <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("max-tokens, timeout and limit must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be new or empty; use a new name for another attempt")
    samples, data_file = load_samples(args.data_dir, args.variant)
    unknown_exclusions = set(args.exclude_task_id) - {sample["task_id"] for sample in samples}
    if unknown_exclusions:
        parser.error(f"Excluded tasks are not present in the selected dataset: {sorted(unknown_exclusions)}")
    samples = [sample for sample in samples if sample["task_id"] not in args.exclude_task_id]
    if args.task_id is not None:
        samples = [sample for sample in samples if sample["task_id"] == args.task_id]
        if not samples:
            parser.error(f"Task is not present in the selected dataset: {args.task_id}")
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        parser.error("No samples to run")
    schema_path = HERE / "spec_output_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "run.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "base_url": args.base_url,
            "model": args.model,
            "variant": args.variant,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "timeout": args.timeout,
            "enable_thinking": False,
            "task_ids": [sample["task_id"] for sample in samples],
            "data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
            "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "passed_means": "Complete response with valid JSON, SpecOutput and visible source anchors; no semantic evaluation",
        },
    )
    results = []
    for sample in samples:
        print(f"Running {sample['task_id']} ...", flush=True)
        result = run_one(sample, args, validator)
        results.append(result)
        write_json(args.output_dir / "summary.json", make_summary(results, len(samples)))
        print(
            f"  {result['status']} | {result['elapsed_seconds']}s | {result['usage']}", flush=True
        )
    passed = sum(result["status"] == "passed" for result in results)
    print(
        f"Complete: {passed}/{len(samples)} passed structural/source checks; {args.output_dir / 'summary.json'}"
    )
    return 0 if passed == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
