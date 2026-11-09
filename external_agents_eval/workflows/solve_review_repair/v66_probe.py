"""V6.6 probe stage P: decide open check points by running existing behavior.

Between C and the Solver, a tool-enabled prober runs the closest existing analog in
the frozen baseline (or reads a dependency's type definition) for each open check
point. A reading is adopted only when the host re-runs the probe command in a fresh
baseline container and the quoted observation appears in its output. Adopted points
become inferred analog check points; everything else stays open.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from .v66_review_prompts import numbered

PROBE_DIR = "/tmp/v66-probe"
RESULT_PATH = f"{PROBE_DIR}/result.json"


def _squash(text: str) -> str:
    return " ".join(text.split())


INDEX = re.compile(r"^\(?(\d+)\)\s*")


def match_reading(reading: str, alternatives: list[str]) -> str | None:
    """Exact candidate text, optionally prefixed by its own (n) index; nothing else."""
    reading = reading.strip()
    exact = next((alt for alt in alternatives if alt.strip() == reading), None)
    if exact is not None:
        return exact
    found = INDEX.match(reading)
    if not found or not 1 <= int(found.group(1)) <= len(alternatives):
        return None
    candidate, rest = alternatives[int(found.group(1)) - 1], reading[found.end():].strip()
    return candidate if not rest or rest == candidate.strip() else None


def open_points(points: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return [(f"CP{index}", point) for index, point in numbered(points) if point["resolution"] == "open"]


def probe_task(task: dict[str, Any], ledger: dict[str, Any], points: list[dict[str, Any]]) -> str:
    """The prober's task text; the Mini instance template adds the command rules."""
    statements = {item["id"]: item["content"].get("statement") or item["content"].get("question")
                  for field in ("obligations", "preservation_obligations", "open_questions")
                  for item in ledger[field]}
    lines = []
    for cp_id, point in open_points(points):
        targets = "; ".join(f"{target}: {statements.get(target, '')}" for target in point["targets"])
        readings = "\n".join(f"    ({n}) {text}" for n, text in enumerate(point["alternatives"], 1))
        lines.append(f"- {cp_id} [{point['facet']}] targets {targets}\n  given {point['given']}; "
                     f"when {point['when']}; then {point['then']}\n  Candidate readings:\n{readings}")
    return f"""You are the V6.6 evidence prober. /app is the frozen baseline repository: the
requested feature below is NOT implemented, and you must not implement it. Put every probe
file under {PROBE_DIR}/. Do not leave changes in /app.

REQUESTED FEATURE (context only)
{task['instruction']}

OPEN CHECK POINTS
{chr(10).join(lines)}

For each open check point:
1. Find the closest existing feature in /app with the same kind of behavior (an analog), or
   the definition of a type, schema or format the requested data is handed to, including
   installed dependencies (for example node_modules, site-packages or the Go module cache).
2. Write a small probe under {PROBE_DIR}/ that exercises that existing behavior in exactly the
   situation of the check point, and run it. Reading a dependency's definition with sed or cat
   also counts as a probe.
3. Choose a reading only when the observed analog behavior clearly corresponds to one
   candidate reading and the analog is the same kind of behavior as the requested one.
   Otherwise leave the check point open. The task text always wins over an analog.

Write {RESULT_PATH} as JSON:
{{"checkpoints": [{{"cp": "<check point id, e.g. CP12>", "decision": "resolved" or "open",
  "reading": "<one candidate reading copied exactly, or null>",
  "analog": {{"path": "<file>", "line_start": 1, "line_end": 9, "same_kind_reason": "<why it is the same kind of behavior>"}} or null,
  "probe": {{"command": "<one shell command, run from /app, that reproduces the observation using only files under {PROBE_DIR}>",
             "observation_quote": "<exact text that command prints>"}} or null,
  "reason": "<short explanation>"}}]}}
List every open check point exactly once. Write {RESULT_PATH} first with every check point
open, then update it after finishing each check point, so finished work survives if you run
out of steps. A resolved entry needs analog, probe and reading.
The command is re-run on a fresh copy of the baseline; the observation_quote must appear in
its output. Never use the network. When {RESULT_PATH} is complete, submit."""


def validate_result(payload: Any, points: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return candidate resolutions by CP id and every problem found (problems keep a point open)."""
    expected = dict(open_points(points))
    problems: list[str] = []
    entries = payload.get("checkpoints") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}, ["result has no checkpoints array"]
    resolved: dict[str, dict[str, Any]] = {}
    seen = set()
    for entry in entries:
        cp_id = entry.get("cp") if isinstance(entry, dict) else None
        if cp_id not in expected:
            problems.append(f"unknown check point {cp_id!r}")
            continue
        if cp_id in seen:
            problems.append(f"{cp_id} listed twice")
            continue
        seen.add(cp_id)
        if entry.get("decision") != "resolved":
            continue
        point = expected[cp_id]
        exact = match_reading(entry.get("reading") or "", point["alternatives"])
        probe, analog = entry.get("probe") or {}, entry.get("analog") or {}
        if exact is None:
            problems.append(f"{cp_id}: reading is not one of the candidate readings")
        elif not (probe.get("command") or "").strip() or not (probe.get("observation_quote") or "").strip():
            problems.append(f"{cp_id}: a resolved point needs a probe command and observation quote")
        elif not (analog.get("path") and (analog.get("same_kind_reason") or "").strip()):
            problems.append(f"{cp_id}: a resolved point needs an analog and a same-kind reason")
        else:
            resolved[cp_id] = {**entry, "reading": exact}
    for cp_id in expected:
        if cp_id not in seen:
            problems.append(f"{cp_id} missing from the result")
    return resolved, problems


def verify(resolved: dict[str, dict[str, Any]], run: Callable[[str], str]) -> dict[str, dict[str, Any]]:
    """Re-run each probe; keep a resolution only if its quote appears in the fresh output."""
    checked = {}
    for cp_id, entry in resolved.items():
        try:
            output = run(entry["probe"]["command"])
        except Exception as error:  # noqa: BLE001 - an unrunnable probe keeps the point open
            output, failure = "", f"{type(error).__name__}: {error}"
        else:
            failure = None
        verified = failure is None and _squash(entry["probe"]["observation_quote"]) in _squash(output)
        checked[cp_id] = {**entry, "verified": verified, "rerun_output_tail": output[-4000:],
                          "rerun_error": failure}
    return checked


def convert(points: list[dict[str, Any]], checked: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace verified open points by inferred analog points; keep the rest unchanged."""
    result = []
    for index, point in numbered(points):
        entry = checked.get(f"CP{index}")
        if point["resolution"] != "open" or not entry or not entry["verified"]:
            result.append(point)
            continue
        others = [alt for alt in point["alternatives"] if alt != entry["reading"]]
        analog = entry["analog"]
        result.append({
            **point,
            "then": f"{point['then']} Resolved by probe: {entry['reading']}",
            "resolution": "analog", "strength": "inferred",
            "rules_out": "; ".join(others),
            "sources": [{"kind": "repository", "path": analog["path"],
                         "line_start": analog.get("line_start") or 1,
                         "line_end": analog.get("line_end") or analog.get("line_start") or 1}],
            "basis_quote": entry["probe"]["observation_quote"],
            "basis_behavior": f"{analog['same_kind_reason']} (probe: {entry['probe']['command']})",
            "alternatives": [], "favored": None,
            "probe": {"cp": f"CP{index}", "command": entry["probe"]["command"],
                      "verified": True, "reason": entry.get("reason")},
        })
    return result


def load_result(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None
