"""V6.8 repair decision checker (standalone: copied into the repair container as lint.py).

Checks /tmp/v68/decisions.jsonl against /tmp/v68/scope.json ("ids"): every issue id has a line (a
later line for the same id replaces an earlier one), decision is changed or kept, and the note says
what changed or why the current behavior was kept. Python 3.6 compatible; no project imports.
"""
import json
import os
import sys

DECISIONS = ("changed", "kept")
MAX_LISTED_ERRORS = 40


def load(directory):
    """Returns (scope ids, rows, parse errors)."""
    with open(os.path.join(directory, "scope.json"), encoding="utf-8") as handle:
        ids = list(json.load(handle)["ids"])
    path = os.path.join(directory, "decisions.jsonl")
    if not os.path.exists(path):
        return ids, [], ["decisions.jsonl is missing"]
    rows, errors = [], []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                errors.append("decisions.jsonl line %d is not valid JSON: %s" % (number, error))
                continue
            if not isinstance(row, dict):
                errors.append("decisions.jsonl line %d is not a JSON object" % number)
                continue
            row["_line"] = number
            rows.append(row)
    return ids, rows, errors


def final_rows(ids, rows):
    last = {}
    for row in rows:
        if row.get("id") in ids:
            last[row["id"]] = row
    return [last[i] for i in ids if i in last]


def lint(ids, rows):
    errors = []
    for row in rows:
        if row.get("id") not in ids:
            errors.append("decisions.jsonl line %d: id %r is not an issue id" % (row["_line"], row.get("id")))
    final = final_rows(ids, rows)
    for row in final:
        where = "decisions.jsonl line %d (%s)" % (row["_line"], row["id"])
        if row.get("decision") not in DECISIONS:
            errors.append(where + ": decision must be changed or kept")
        note = row.get("note")
        if not isinstance(note, str) or not note.strip():
            errors.append(where + ": note (what you changed, or why you kept the current behavior) is empty")
    decided = set(row["id"] for row in final)
    missing = [i for i in ids if i not in decided]
    if missing:
        errors.append("%d issue ids have no decision: %s" % (len(missing), ", ".join(missing)))
    stats = {"issues": len(ids), "decided": len(decided),
             "changed": sum(1 for r in final if r.get("decision") == "changed"),
             "kept": sum(1 for r in final if r.get("decision") == "kept")}
    return errors, stats


def main(argv):
    directory = argv[1] if len(argv) > 1 else "/tmp/v68"
    ids, rows, parse_errors = load(directory)
    errors, stats = lint(ids, rows)
    problems = parse_errors + errors
    if problems:
        print("LINT FAILED (%d problems)" % len(problems))
        for problem in problems[:MAX_LISTED_ERRORS]:
            print("- " + problem)
    else:
        print("LINT OK")
    print("stats: " + json.dumps(stats, sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
