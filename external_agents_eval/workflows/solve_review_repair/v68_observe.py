"""V6.8 observation protocol (evaluation labels; user, 2026-11-08).

A native Mini agent records, for each V6.8.1 question of one batch, which reading an
implementation applied to /app follows (1..n, 0 = none of the readings), without judging it.
Used with the reference solution to build evaluation-only reading labels: the labels never enter
any method's input. The prompt does not say whose implementation it is.
"""
from __future__ import annotations

import json
from typing import Any

from external_agents_eval.workflows.solve_review_repair import v68_review as rv

VERSION = "v6.8-observe-1"
WORK_DIR = rv.WORK_DIR
STEP_LIMIT, WALL_TIME_SEC = rv.STEP_LIMIT, rv.WALL_TIME_SEC  # same limits as the review episodes
SUBMIT = rv.SUBMIT

HEADER = """You are documenting how an implementation of the following software task behaves. You do NOT change or judge it.

<task>
{{task}}
</task>

The implementation is already applied to the repository at /app (baseline plus the implementation's
diff); the diff is in %(work)s/candidate.diff and the task text in %(work)s/task.md. Any change under
/app is discarded; you may create temporary probe files, and you write your results only to
%(work)s/labels.jsonl.

%(work)s/questions.jsonl lists questions about specific conditions (your share; their ids are in
%(work)s/scope.json). Each question names a condition and numbered readings (possible behaviors);
observations are facts about the baseline before the implementation (line numbers may have shifted).
For each question, find out which reading the implementation actually follows.

FOR EACH QUESTION
1. Determine the implementation's actual behavior for the condition: read the changed code and the code it
   calls; when it is cheap, run a small probe (a short script or snippet that calls the public API with
   the question's inputs) and observe the result. Probe several conditions per script where possible.
2. Record:
   - "observed": you determined the behavior. "reading" is the number of the reading it follows, or 0 if
     it follows none of them (describe what it does instead in "behavior").
   - "not_applicable": the question's premise does not hold for this implementation (for example the API
     it asks about does not exist in that form); say why in "behavior".
   - "undetermined": you could not determine the behavior.
   Do not judge whether the behavior is right; only describe it. Never guess.

FILE %(work)s/labels.jsonl (one JSON object per line; a later line for the same id replaces an earlier one)
{"id": "CP12", "status": "observed|not_applicable|undetermined", "reading": 2,
 "behavior": "what the implementation does for this condition",
 "evidence": [{"kind": "probe", "detail": "command and the relevant output"},
              {"kind": "code", "path": "repo/relative/path", "line": 42, "detail": "what this line does"}]}
Append lines with heredocs, e.g.
cat >> %(work)s/labels.jsonl <<'LABELS'
{"id": "CP3", "status": "observed", "reading": 1, "behavior": "...", "evidence": [...]}
LABELS

WORKFLOW AND BUDGET
You have at most %(steps)d commands and %(minutes)d minutes. Start with
`cat %(work)s/scope.json; python -c "import json;[print(json.dumps(json.loads(l),indent=1)) for l in open('%(work)s/questions.jsonl')]" | head -c 20000`
and `git -C /app diff --stat; head -c 20000 %(work)s/candidate.diff`. First write the lines you can settle
from the code alone, then probe the rest; append lines as you go (every id in scope needs a line), then run
`python %(work)s/lint.py` and fix every problem it reports.
When `python %(work)s/lint.py` prints LINT OK, finish by issuing exactly:
`%(submit)s`
Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

""" % {"work": WORK_DIR, "steps": STEP_LIMIT, "minutes": WALL_TIME_SEC // 60, "submit": SUBMIT}


def header() -> str:
    return HEADER


def question_lines(data: dict[str, Any], ids: list[str]) -> bytes:
    """questions.jsonl with numbered readings ("1. ...")."""
    out = []
    for line in rv.question_lines(data, ids).decode().splitlines():
        row = json.loads(line)
        row["readings"] = [f"{n}. {text}" for n, text in enumerate(row.get("readings") or [], 1)]
        out.append(json.dumps(row, ensure_ascii=False))
    return ("\n".join(out) + "\n").encode()
