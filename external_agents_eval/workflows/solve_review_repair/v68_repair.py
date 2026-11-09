"""V6.8 end-to-end repair test (user, 2026-11-08).

Issue list per failing candidate: the V6.8.1 questions on which the patch-conditioned review reported
a defect AND the direction C1 resolver (decided without seeing the patch) chose a reading other than
the one the candidate follows (candidate reading mapped from the review's behavior description). At
most MAX_ISSUES per candidate, highest resolver confidence first. Each issue gives the condition, the
candidate's current behavior with the reviewer's evidence, and the resolver's reading as the expected
behavior with its evidence. No reference solution or hidden test enters the list.

A native Mini agent (M0's model and templates) gets the issues in a fresh offline baseline container
with the candidate applied, checks each issue, fixes the ones it agrees with and records one decision
per issue (checked by v68_repair_lint).
"""
from __future__ import annotations

import json
from typing import Any

VERSION = "v6.8-repair-1"
WORK_DIR = "/tmp/v68"
STEP_LIMIT, WALL_TIME_SEC = 150, 3600
MAX_ISSUES = 20
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

HEADER = """You are improving an implementation of the following software task.

<task>
{{task}}
</task>

The implementation is applied to the repository at /app (the baseline plus the implementation's diff,
which is also in %(work)s/candidate.diff; the task text is in %(work)s/task.md). An automated review
compared it with the task and listed possible issues in %(work)s/issues.md: for each, a condition, what
the implementation currently does there (with the reviewer's evidence), and the behavior the review
expects, with its reasoning. The review can be wrong: its expected behavior may misread the task, or
the current behavior may already be right.

FOR EACH ISSUE
1. Confirm the current behavior: read the code; run a small probe when it is cheap.
2. Decide from the task text, the existing code and any standard involved whether the expected behavior
   is what the task requires and the current behavior is not. Change the code only when you are
   convinced; otherwise keep the current behavior.
3. Keep each change minimal and limited to its issue; do not change other behavior.

RULES
- Change source files under /app only for fixes. Put probe scripts under /tmp, never under /app, and do
  not add new test files to /app.
- After your changes, run the existing tests that cover the files you changed, and fix what you broke.
- Record one decision per issue in %(work)s/decisions.jsonl (one JSON object per line; a later line for
  the same id replaces an earlier one):
  {"id": "I3", "decision": "changed|kept", "note": "what you changed, or why you kept the current behavior"}
  Append lines with heredocs, e.g.
  cat >> %(work)s/decisions.jsonl <<'DECISIONS'
  {"id": "I1", "decision": "kept", "note": "..."}
  DECISIONS

WORKFLOW AND BUDGET
You have at most %(steps)d commands and %(minutes)d minutes. Start with `cat %(work)s/issues.md` and
`git -C /app diff --stat`. Group related issues. When every issue has a decision, run
`python %(work)s/lint.py` and fix every problem it reports. When it prints LINT OK, finish by issuing exactly:
`%(submit)s`
Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

""" % {"work": WORK_DIR, "steps": STEP_LIMIT, "minutes": WALL_TIME_SEC // 60, "submit": SUBMIT}


def header() -> str:
    return HEADER


def _sq(text: Any) -> str:
    return " ".join(str(text or "").split())


def select_issues(data: dict[str, Any], review_rows: dict[str, dict], candidate: dict[str, int],
                  answers: dict[str, dict]) -> list[dict[str, Any]]:
    """Review defect AND resolver reading != candidate reading; top MAX_ISSUES by resolver confidence."""
    chosen = []
    for cid, row in review_rows.items():
        reading, answer = candidate.get(cid), answers.get(cid)
        if row.get("status") != "defect" or not isinstance(reading, int) or reading < 0 or answer is None:
            continue
        if answer["reading"] == reading:
            continue
        chosen.append((-float(answer["confidence"]), int(cid[2:]), cid))
    chosen.sort()
    issues = []
    for number, (_, _, cid) in enumerate(chosen[:MAX_ISSUES], 1):
        item, row, answer = data["items"][int(cid[2:]) - 1], review_rows[cid], answers[cid]
        readings = item.get("alternatives") or []
        expected = readings[answer["reading"] - 1] if answer["reading"] > 0 else answer.get("instead")
        issues.append({"id": f"I{number}", "question": cid, "given": item.get("given"), "when": item.get("when"),
                       "then": item.get("then"), "task_quote": item.get("task_quote"),
                       "current": row.get("candidate_behavior"),
                       "evidence": [e for e in row.get("evidence") or [] if isinstance(e, dict)][:2],
                       "expected": expected, "expected_reading": answer["reading"],
                       "candidate_reading": candidate[cid], "basis": answer.get("basis"),
                       "rationale": _sq(answer.get("evidence")), "why": _sq(answer.get("why")),
                       "confidence": answer["confidence"]})
    return issues


def _evidence_line(evidence: dict[str, Any]) -> str:
    where = f" {evidence.get('path')}:{evidence.get('line')}" if evidence.get("kind") == "code" else ""
    return f"  - {evidence.get('kind')}{where}: {_sq(evidence.get('detail'))[:400]}"


def issues_markdown(issues: list[dict[str, Any]]) -> str:
    out = [f"# Possible issues ({len(issues)})", ""]
    for issue in issues:
        out += [f"## {issue['id']}",
                f"- Condition: given {_sq(issue['given'])}; when {_sq(issue['when'])}",
                f"- Question: {_sq(issue['then'])}"]
        if issue.get("task_quote"):
            out.append(f"- Task passage: \"{_sq(issue['task_quote'])}\"")
        out.append(f"- Current behavior: {_sq(issue['current'])}")
        if issue["evidence"]:
            out += ["- Reviewer's evidence:"] + [_evidence_line(e) for e in issue["evidence"]]
        out.append(f"- Expected behavior: {_sq(issue['expected'])}")
        reason = " ".join(x for x in (issue.get("rationale"), issue.get("why")) if x)
        if reason:
            out.append(f"- Reasoning ({issue.get('basis') or 'unspecified'}): {reason}")
        out.append("")
    return "\n".join(out)


def scope(issues: list[dict[str, Any]]) -> bytes:
    return json.dumps({"ids": [issue["id"] for issue in issues]}).encode()
