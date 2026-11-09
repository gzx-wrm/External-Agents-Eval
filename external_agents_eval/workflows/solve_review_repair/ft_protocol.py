"""Frozen FT evidence/rendering/validation functions; see ft/provenance.json."""
from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import Draft202012Validator

FIELDS = ("obligations", "preservation_obligations", "open_questions")


FOOTER = re.compile(
    r"\n\n\[(?:Showing lines [^\n]+|\d+ more lines in file\. Use offset=\d+ to continue\.)\]$"
)

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

def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result

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
