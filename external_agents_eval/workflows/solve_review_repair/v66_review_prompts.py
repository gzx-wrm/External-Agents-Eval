"""V6.6 downstream text: check points for the Solver input, I/R lanes and S.

Each addendum is appended to the unchanged V6.3 prompt; nothing else changes.
"""
from __future__ import annotations

from typing import Any

SOLVER_HEADING = "## Check points"
SOLVER_GUIDANCE = (
    "These check points were derived from the task, the ledger and the repository before "
    "implementation. [task] points are required by the task text. [analog], [contract] and "
    "[convention] points are references inferred from existing repository code or widely used "
    "conventions. [open] points list candidate readings of details the task leaves open: choose "
    "the reading that matches the closest existing behavior in this repository and apply it "
    "consistently."
)


def augment_solver_task(task: dict[str, Any], solver_view: str) -> dict[str, Any]:
    """Append the ranked check points after the frozen ledger supplement."""
    result = dict(task)
    result["instruction"] = (task["instruction"].rstrip("\n") + "\n\n" + SOLVER_HEADING + "\n\n"
                             + SOLVER_GUIDANCE + "\n\n" + solver_view.strip() + "\n")
    return result


def _format(index: int, point: dict[str, Any]) -> str:
    text = (f"CP{index} [{point['resolution']}] {','.join(point['targets'])}/{point['facet']}: "
            f"given {point['given']}; when {point['when']}; then {point['then']}")
    if point["resolution"] == "open":
        text += " Candidate readings: " + " | ".join(point["alternatives"])
        if point.get("favored"):
            text += f". Favored: {point['favored']}"
    elif point.get("rules_out"):
        text += f" Excludes: {point['rules_out']}"
    if point.get("basis_quote"):
        text += f' Basis: "{point["basis_quote"]}" ({point.get("basis_behavior") or ""})'
    elif point.get("convention"):
        text += f" Convention: {point['convention']} ({point.get('basis_behavior') or ''})"
    return text


def numbered(points: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    return list(enumerate(points, 1))


def lane_addendum(role: str, points: list[dict[str, Any]]) -> str:
    """I gets check points on O targets, R those on P targets."""
    prefix = "O" if role == "I" else "P"
    mine = [(index, point) for index, point in numbered(points)
            if any(target.startswith(prefix) for target in point["targets"])]
    if not mine:
        return ""
    lines = "\n".join(f"- {_format(index, point)}" for index, point in mine)
    return f"""CHECK POINTS (V6.6)
A coverage planner derived these observable check points for your targets from the task,
the ledger and frozen repository evidence. For each check point listed below:
- write a probe from the task and the check point alone, not from the candidate code, and
  record the probe and what you observed in the review note of every target it lists;
- for an [open] check point, probe each candidate reading and record which one the
  candidate implements;
- a [task] check point that fails makes its target violated;
- an [analog], [contract] or [convention] check point that fails makes its target violated;
  state in the note that the violated check point is inferred and quote its basis;
- if the candidate implements a reading of an [open] check point that quoted repository
  evidence (in the check point or found by you) contradicts, the target is violated; quote
  that evidence in the note.
Check points do not replace your targets, their importance or the note format. Mark a
target violated only for observed behavior, never for a missing probe.
{lines}
"""


def summary_addendum(points: list[dict[str, Any]]) -> str:
    if not points:
        return ""
    lines = "\n".join(f"- {_format(index, point)}" for index, point in numbered(points))
    return f"""CHECK POINTS AND DECISION ORDER (V6.6)
Reviewers probed the check points below. Decide findings that rest on a check point with
this order of authority: task text > existing repository analog > downstream contract >
widely used convention. A violated assessment that rests on an inferred check point
([analog], [contract], [convention]) or on an [open] check point is a confirmed required
violation, and is repaired, when the quoted basis shows the behavior and no task text
contradicts it; dismiss it when the basis does not show the behavior or the task text
allows the candidate's behavior. Cite the check point and its basis in the repair request.
{lines}
"""
