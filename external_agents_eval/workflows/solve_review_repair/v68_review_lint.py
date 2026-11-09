"""V6.8 patch-review checker (standalone: copied into the review container as lint.py).

Checks /tmp/v68/review.jsonl against /tmp/v68/scope.json ("ids"): every question id in scope has
a line (a later line for the same id replaces an earlier one), statuses are ok/defect/unsure,
candidate_behavior is stated, and every defect has an expected behavior, a severity and evidence
(probe output or a real code path:line in the patched repository).
Python 3.6 compatible; no project imports.
"""
import json
import os
import sys

STATUSES = ("ok", "defect", "unsure")
SEVERITIES = ("high", "medium", "low")
EVIDENCE_KINDS = ("probe", "code")
MAX_LISTED_ERRORS = 40


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def load(directory):
    """Returns (scope ids, rows, parse errors)."""
    with open(os.path.join(directory, "scope.json"), encoding="utf-8") as handle:
        ids = list(json.load(handle)["ids"])
    rows, errors = [], []
    path = os.path.join(directory, "review.jsonl")
    if not os.path.exists(path):
        return ids, rows, ["review.jsonl is missing"]
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                errors.append("review.jsonl line %d is not valid JSON: %s" % (number, error))
                continue
            if not isinstance(row, dict):
                errors.append("review.jsonl line %d is not a JSON object" % number)
                continue
            row["_line"] = number
            rows.append(row)
    return ids, rows, errors


def final_rows(ids, rows):
    """The last line per in-scope id, in scope order."""
    last = {}
    for row in rows:
        if row.get("id") in ids:
            last[row["id"]] = row
    return [last[i] for i in ids if i in last]


def _line_count(root, path):
    full = path if os.path.isabs(path) else os.path.join(root, path)
    if not os.path.isfile(full):
        return None
    with open(full, encoding="utf-8", errors="replace") as handle:
        return len(handle.read().split("\n"))


def lint(ids, rows, repo_root):
    errors = []
    for row in rows:
        if row.get("id") not in ids:
            errors.append("review.jsonl line %d: id %r is not a question id in scope.json"
                          % (row["_line"], row.get("id")))
    final = final_rows(ids, rows)
    for row in final:
        where = "review.jsonl line %d (%s)" % (row["_line"], row["id"])
        status = row.get("status")
        if status not in STATUSES:
            errors.append(where + ": status must be one of " + ", ".join(STATUSES))
            continue
        if not _text(row.get("candidate_behavior")):
            errors.append(where + ": candidate_behavior (what the candidate actually does here) is empty")
        if status != "defect":
            continue
        if not _text(row.get("expected")):
            errors.append(where + ": a defect needs expected (the behavior the candidate should have)")
        if row.get("severity") not in SEVERITIES:
            errors.append(where + ": a defect needs severity high, medium or low")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(where + ": a defect needs evidence (a probe with its output, or code path:line)")
            continue
        for number, item in enumerate(evidence, 1):
            if not isinstance(item, dict) or item.get("kind") not in EVIDENCE_KINDS or not _text(item.get("detail")):
                errors.append("%s: evidence %d needs kind probe|code and detail" % (where, number))
                continue
            if item["kind"] == "code":
                path, line = item.get("path"), item.get("line")
                count = _line_count(repo_root, path) if _text(path) else None
                if count is None:
                    errors.append("%s: evidence %d: path %r does not exist in /app" % (where, number, path))
                elif not isinstance(line, int) or isinstance(line, bool) or not 1 <= line <= count:
                    errors.append("%s: evidence %d: line must be an integer within %s (1-%d)"
                                  % (where, number, path, count))
    reviewed = set(row["id"] for row in final)
    missing = [i for i in ids if i not in reviewed]
    if missing:
        errors.append("%d question ids have no review line: %s%s" % (
            len(missing), ", ".join(missing[:30]), " ..." if len(missing) > 30 else ""))
    stats = {"in_scope": len(ids), "reviewed": len(reviewed),
             "defects": sum(1 for r in final if r.get("status") == "defect"),
             "unsure": sum(1 for r in final if r.get("status") == "unsure")}
    return errors, stats


def main(argv):
    directory = argv[1] if len(argv) > 1 else "/tmp/v68"
    repository = argv[2] if len(argv) > 2 else "/app"
    ids, rows, parse_errors = load(directory)
    errors, stats = lint(ids, rows, repository)
    problems = parse_errors + errors
    if problems:
        print("LINT FAILED (%d problems)" % len(problems))
        for problem in problems[:MAX_LISTED_ERRORS]:
            print("- " + problem)
        if len(problems) > MAX_LISTED_ERRORS:
            print("- ... %d more" % (len(problems) - MAX_LISTED_ERRORS))
    else:
        print("LINT OK")
    print("stats: " + json.dumps(stats, sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
