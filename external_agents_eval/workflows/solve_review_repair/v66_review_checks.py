"""V6.6 per-batch check points for I/R and validation of their checkpoint_results.

The section is inserted by the Mini review adapter into every round's prompt, for the
targets of that batch only. Each assessed note then carries a '## Checkpoint results'
section; the batch closes only when every listed check point has enough results.
"""
from __future__ import annotations

from typing import Any

from .v66_review_prompts import _format, numbered

RESULTS_HEADING = "## Checkpoint results"
OUTCOMES = {"met", "not_met"}
MIN_RESULTS = {"task": 1}  # open and inferred check points need 2 distinct partitions
DECISION_MIN_RESULTS = 2


def batch_points(points: list[dict[str, Any]], target_ids: set[str]) -> list[tuple[str, dict[str, Any]]]:
    return [(f"CP{index}", point) for index, point in numbered(points)
            if set(point["targets"]) & target_ids]


def batch_section(points: list[dict[str, Any]], target_ids: set[str]) -> str:
    mine = batch_points(points, target_ids)
    if not mine:
        return ""
    lines = "\n".join(f"- {_format(int(cp_id[2:]), point)}" for cp_id, point in mine)
    return f"""CHECK POINTS FOR THIS BATCH (V6.6)
Check points derived from the task, the ledger and repository evidence for the targets of
this batch:
{lines}

For each check point above:
- probe it from the task and the check point alone, not from the candidate code;
- [task] check points need at least 1 result; [analog], [contract], [convention] and [open]
  check points need at least 2 results whose inputs come from different partitions (for
  example normal / empty / absent / boundary / invalid / mixed inputs, different orderings
  in time, or different instances or contexts);
- for an [open] check point, record which candidate reading the candidate implements;
- verdicts: a failed [task] check point makes its target violated; a failed inferred check
  point makes its target violated, citing its basis; an [open] reading that quoted repository
  evidence contradicts makes the target violated, quoting that evidence.
In the note of each listed target, add right after '## Assessment' a section
'{RESULTS_HEADING}' with one fenced json object:
{{"checkpoint_results": [{{"cp": "CP7", "outcome": "met" or "not_met",
  "implemented_reading": "<open check points: the candidate reading implemented, or 'other'>",
  "results": [{{"input": "<what you ran>", "partition": "<which partition the input belongs to>",
               "observed": "<what you observed>"}}]}}]}}
A check point listing several targets of this batch may be recorded in any one of their notes.
The batch closes only when every check point above has its results.
"""


def checkpoint_result_errors(
    notes: dict[str, str], points: list[dict[str, Any]], assessed: set[str],
    parse: Any,
) -> list[str]:
    """notes: target id -> note text for this batch's assessments; parse: metadata()."""
    relevant = batch_points(points, assessed)
    if not relevant:
        return []
    entries: dict[str, list[dict[str, Any]]] = {}
    errors = []
    for target, text in notes.items():
        if not any(target in point["targets"] for _, point in relevant):
            continue
        if text.splitlines().count(RESULTS_HEADING) == 0:
            continue
        try:
            value = parse(text, RESULTS_HEADING)
        except Exception as error:  # noqa: BLE001 - surfaced to the reviewer verbatim
            errors.append(f"{target}: {error}")
            continue
        items = value.get("checkpoint_results")
        if not isinstance(items, list):
            errors.append(f"{target}: {RESULTS_HEADING} needs a checkpoint_results array")
            continue
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("cp"), str):
                entries.setdefault(item["cp"], []).append(item)
    for cp_id, point in relevant:
        items = entries.get(cp_id, [])
        if not items:
            errors.append(f"{cp_id}: no checkpoint_results entry in the notes of {sorted(set(point['targets']) & assessed)}")
            continue
        results = [result for item in items for result in (item.get("results") or [])
                   if isinstance(result, dict)]
        complete = [result for result in results
                    if all(str(result.get(key) or "").strip() for key in ("input", "partition", "observed"))]
        if len(complete) != len(results) or not complete:
            errors.append(f"{cp_id}: every result needs non-empty input, partition and observed")
            continue
        if any(item.get("outcome") not in OUTCOMES for item in items):
            errors.append(f"{cp_id}: outcome must be 'met' or 'not_met'")
        if point["resolution"] != "task":
            partitions = {" ".join(result["partition"].lower().split()) for result in complete}
            if len(partitions) < DECISION_MIN_RESULTS:
                errors.append(f"{cp_id}: needs at least {DECISION_MIN_RESULTS} results from different partitions")
        if point["resolution"] == "open":
            allowed = set(point["alternatives"]) | {"other"}
            if any((item.get("implemented_reading") or "").strip() not in allowed for item in items):
                errors.append(f"{cp_id}: implemented_reading must be one of the candidate readings or 'other'")
    return errors
