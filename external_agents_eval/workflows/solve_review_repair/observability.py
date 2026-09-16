from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import sha256_bytes

READ_LIKE_TOOLS = frozenset({"read", "grep", "find", "ls"})
EXPLICIT_WRITE_TOOLS = frozenset({"edit", "write"})


def observe_repair_sequence(
    trajectory_path: Path,
    *,
    audit_schema_path: str,
    audit_output_path: str,
    frozen_inputs: dict[str, str],
) -> dict[str, Any]:
    """Summarize observable Repair tool ordering without claiming a hard edit gate."""
    base = {
        "schema_version": "1.0",
        "trajectory_artifact": "repair/trajectory.json",
        "audit_schema_path": audit_schema_path,
        "frozen_inputs": sorted(frozen_inputs),
        "limitations": [
            "This is an observation of recorded tool calls, not proof of semantic reasoning.",
            "A bash call can read or mutate files in ways that cannot be classified reliably.",
            "The final Repair audit is a self-report and does not prove when reasoning occurred.",
        ],
    }
    try:
        raw = trajectory_path.read_bytes()
        trajectory = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        return {
            **base,
            "status": "unavailable",
            "error": str(error),
            "trajectory_sha256": None,
            "first_explicit_repository_write": None,
            "audit_schema_read_before_explicit_write": None,
            "frozen_input_reads_before_explicit_write": [],
            "shell_steps_before_explicit_write": [],
        }
    if not isinstance(trajectory, dict):
        return {
            **base,
            "status": "unavailable",
            "error": "trajectory root is not an object",
            "trajectory_sha256": sha256_bytes(raw),
            "first_explicit_repository_write": None,
            "audit_schema_read_before_explicit_write": None,
            "frozen_input_reads_before_explicit_write": [],
            "shell_steps_before_explicit_write": [],
        }

    events: list[dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_id = step.get("step_id")
        for index, call in enumerate(step.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            tool = str(call.get("function_name") or "")
            arguments = call.get("arguments") or {}
            serialized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
            events.append(
                {
                    "step_id": step_id,
                    "tool_call_index": index,
                    "tool": tool,
                    "arguments": serialized,
                }
            )

    first_write_index: int | None = None
    first_write: dict[str, Any] | None = None
    for index, event in enumerate(events):
        if event["tool"] not in EXPLICIT_WRITE_TOOLS:
            continue
        if audit_output_path and audit_output_path in event["arguments"]:
            continue
        first_write_index = index
        first_write = {
            "step_id": event["step_id"],
            "tool_call_index": event["tool_call_index"],
            "tool": event["tool"],
        }
        break

    boundary = first_write_index if first_write_index is not None else len(events)
    preceding = events[:boundary]
    audit_schema_read = any(
        event["tool"] in READ_LIKE_TOOLS and audit_schema_path in event["arguments"]
        for event in preceding
    )
    frozen_reads = sorted(
        name
        for name, path in frozen_inputs.items()
        if path
        and any(
            event["tool"] in READ_LIKE_TOOLS and path in event["arguments"] for event in preceding
        )
    )
    shell_steps = sorted(
        {
            event["step_id"]
            for event in preceding
            if event["tool"] == "bash" and event["step_id"] is not None
        },
        key=str,
    )
    if first_write is None:
        status = "no_explicit_repository_write_observed"
        schema_result: bool | None = None
    elif audit_schema_read:
        status = "schema_read_before_explicit_repository_write"
        schema_result = True
    else:
        status = "schema_read_not_observed_before_explicit_repository_write"
        schema_result = False
    return {
        **base,
        "status": status,
        "error": None,
        "trajectory_sha256": sha256_bytes(raw),
        "first_explicit_repository_write": first_write,
        "audit_schema_read_before_explicit_write": schema_result,
        "frozen_input_reads_before_explicit_write": frozen_reads,
        "shell_steps_before_explicit_write": shell_steps,
    }
