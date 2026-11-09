"""V6.8 patch-conditioned review (no Solver, no repair; user, 2026-11-08).

The V6.8.1 targeted question pool is used against an existing failing candidate patch: in a fresh
offline baseline container with the candidate applied, a native Mini agent goes through the
questions of one obligation batch (the V6.8.1 generation batches, three obligations each), finds
out what the candidate does for each condition (code reading, small probes) and marks it ok,
defect (with expected behavior, severity and evidence) or unsure. v68_review_lint checks the file;
the episode submits only after LINT OK. The defects of all batches form the review report.
"""
from __future__ import annotations

import json
from typing import Any

VERSION = "v6.8-review-1"
WORK_DIR = "/tmp/v68"
STEP_LIMIT, WALL_TIME_SEC = 60, 1500  # same limits as the V6.8.1 batch episodes
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

HEADER = """You are reviewing a candidate implementation of the following software task. You do NOT fix it.

<task>
{{task}}
</task>

The candidate's patch is already applied to the repository at /app (baseline plus the patch); the
diff is in %(work)s/candidate.diff and the task text in %(work)s/task.md. Any change under /app is
discarded; you may create temporary probe files, and you write your results only to
%(work)s/review.jsonl.

GOAL
Hidden acceptance tests will be run on this candidate. Implementations of this task tend to fail them
on behaviors such as other code paths that must behave the same, lifecycle/state interactions, edge
values, the exact form of a new API and rules of external standards. %(work)s/questions.jsonl lists
check questions about such behaviors (your share; their ids are in %(work)s/scope.json). Each question
names a condition and two or more readings; none is marked correct, the order means nothing, and its
observations are facts about the baseline before the patch (line numbers may have shifted), not
answers. Find out what the candidate actually does for each condition and judge whether that is right.

FOR EACH QUESTION
1. Determine the candidate's actual behavior for the condition: read the changed code and the code it
   calls; when it is cheap, run a small probe (a short script or snippet that calls the public API with
   the question's inputs) and observe the result. Probe several conditions per script where possible.
2. Decide the status:
   - "defect": the candidate crashes or raises where it should not, does not handle the condition, or
     behaves in a way that the task text, the cited standard or the analogous existing code in the
     repository contradicts; or it implements a reading that is clearly less supported by the task text
     and the repository than another reading. Give the expected behavior, a severity and evidence.
   - "ok": the behavior is what the task text, the standard and the repository support, or a reading at
     least as well supported as the others.
   - "unsure": you could not determine the behavior, or which reading is better supported.
3. Report only what you observed or read; never guess the candidate's behavior.

FILE %(work)s/review.jsonl (one JSON object per line; a later line for the same id replaces an earlier one)
{"id": "CP12", "status": "ok|defect|unsure", "candidate_behavior": "what the candidate does for this condition",
 "expected": "the behavior it should have (defect only)", "severity": "high|medium|low (defect only)",
 "evidence": [{"kind": "probe", "detail": "command and the relevant output"},
              {"kind": "code", "path": "repo/relative/path", "line": 42, "detail": "what this line does"}]}
severity: high = crash, wrong result for a condition the task text names, or a clear violation of the
cited standard; medium = a less supported reading or an inconsistency with analogous code; low = minor.
Append lines with heredocs, e.g.
cat >> %(work)s/review.jsonl <<'REVIEW'
{"id": "CP3", "status": "ok", "candidate_behavior": "..."}
REVIEW

WORKFLOW AND BUDGET
You have at most %(steps)d commands and %(minutes)d minutes. Start with
`cat %(work)s/scope.json; python -c "import json;[print(json.dumps(json.loads(l),indent=1)) for l in open('%(work)s/questions.jsonl')]" | head -c 20000`
and `git -C /app diff --stat; head -c 20000 %(work)s/candidate.diff`. Group related questions, probe
where it decides the status, append review lines as you go (every id in scope needs a line), then run
`python %(work)s/lint.py` and fix every problem it reports.
When `python %(work)s/lint.py` prints LINT OK, finish by issuing exactly:
`%(submit)s`
Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

""" % {"work": WORK_DIR, "steps": STEP_LIMIT, "minutes": WALL_TIME_SEC // 60, "submit": SUBMIT}


def header() -> str:
    return HEADER


def _sq(text: Any) -> str:
    return " ".join(str(text or "").split())


def question_lines(data: dict[str, Any], ids: list[str]) -> bytes:
    """questions.jsonl for the given CP ids (CPn is the n-th V6.8.1 item)."""
    obligations = {o.get("id"): o for o in data.get("obligations") or []}
    entities = {e.get("id"): e for e in data.get("entities") or [] if isinstance(e, dict)}
    out = []
    for cid in ids:
        item = data["items"][int(cid[2:]) - 1]
        obligation = obligations.get(item.get("obligation")) or {}
        entity = entities.get(item.get("entity"))
        row = {"id": cid, "obligation": f"{item.get('obligation')}: {_sq(obligation.get('statement'))}",
               "given": item.get("given"), "when": item.get("when"), "then": item.get("then"),
               "readings": item.get("alternatives") or []}
        if entity:
            row["code_object"] = f"{entity.get('symbol')} ({entity.get('path')}:{entity.get('line')}, baseline)"
        if item.get("observations"):
            row["observations"] = [{"text": o.get("text"), "where": f"{o.get('path')}:{o.get('line')} (baseline)"}
                                   for o in item["observations"] if isinstance(o, dict)]
        if isinstance(item.get("spec"), dict) and item["spec"].get("name"):
            row["spec"] = item["spec"]
        if item.get("task_quote"):
            row["task_quote"] = item["task_quote"]
        out.append(json.dumps(row, ensure_ascii=False))
    return ("\n".join(out) + "\n").encode()


def batches(data: dict[str, Any]) -> list[list[str]]:
    """CP ids grouped by their V6.8.1 generation batch."""
    groups: dict[Any, list[str]] = {}
    for index, item in enumerate(data["items"], 1):
        groups.setdefault(item.get("batch"), []).append(f"CP{index}")
    return [groups[key] for key in sorted(groups, key=lambda k: (k is None, k))]


def defects_markdown(data: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Review report for grading: defects only, high severity first; ids D1.. (source ids hidden)."""
    defects = [r for r in rows if r.get("status") == "defect"]
    defects.sort(key=lambda r: SEVERITY_ORDER.get(r.get("severity"), 3))
    out = [f"# Review findings ({len(defects)} defects)", ""]
    for number, row in enumerate(defects, 1):
        item = data["items"][int(row["id"][2:]) - 1]
        out.append(f"## D{number} severity={row.get('severity')}")
        out.append(f"- condition: given {_sq(item.get('given'))}; when {_sq(item.get('when'))}; then {_sq(item.get('then'))}")
        out.append(f"- candidate behavior: {_sq(row.get('candidate_behavior'))}")
        out.append(f"- expected: {_sq(row.get('expected'))}")
        for evidence in row.get("evidence") or []:
            if isinstance(evidence, dict):
                where = f" {evidence.get('path')}:{evidence.get('line')}" if evidence.get("kind") == "code" else ""
                out.append(f"- evidence ({evidence.get('kind')}{where}): {_sq(evidence.get('detail'))[:600]}")
        out.append("")
    return "\n".join(out)
