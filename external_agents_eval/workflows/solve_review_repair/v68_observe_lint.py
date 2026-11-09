"""V6.8 observation checker (standalone: copied into the observation container as lint.py).

Checks /tmp/v68/labels.jsonl against /tmp/v68/scope.json ("ids") and /tmp/v68/questions.jsonl:
every question id in scope has a line (a later line for the same id replaces an earlier one);
status is observed / not_applicable / undetermined; an observed line names the reading the
implementation follows (1..n, or 0 when none of the readings matches) with a behavior description
and evidence (probe output or a real code path:line). Python 3.6 compatible; no project imports.
"""
import json
import os
import sys

STATUSES = ("observed", "not_applicable", "undetermined")
EVIDENCE_KINDS = ("probe", "code")
MAX_LISTED_ERRORS = 40


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _jsonl(path, errors, name):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                errors.append("%s line %d is not valid JSON: %s" % (name, number, error))
                continue
            if not isinstance(row, dict):
                errors.append("%s line %d is not a JSON object" % (name, number))
                continue
            row["_line"] = number
            rows.append(row)
    return rows


def load(directory):
    """Returns (scope ids, readings count per id, rows, parse errors)."""
    errors = []
    with open(os.path.join(directory, "scope.json"), encoding="utf-8") as handle:
        ids = list(json.load(handle)["ids"])
    counts = {}
    for question in _jsonl(os.path.join(directory, "questions.jsonl"), errors, "questions.jsonl"):
        counts[question.get("id")] = len(question.get("readings") or [])
    path = os.path.join(directory, "labels.jsonl")
    if not os.path.exists(path):
        return ids, counts, [], errors + ["labels.jsonl is missing"]
    return ids, counts, _jsonl(path, errors, "labels.jsonl"), errors


def final_rows(ids, rows):
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


def lint(ids, counts, rows, repo_root):
    errors = []
    for row in rows:
        if row.get("id") not in ids:
            errors.append("labels.jsonl line %d: id %r is not a question id in scope.json" % (row["_line"], row.get("id")))
    final = final_rows(ids, rows)
    for row in final:
        where = "labels.jsonl line %d (%s)" % (row["_line"], row["id"])
        status = row.get("status")
        if status not in STATUSES:
            errors.append(where + ": status must be one of " + ", ".join(STATUSES))
            continue
        if status == "undetermined":
            continue
        if not _text(row.get("behavior")):
            errors.append(where + ": behavior (what the implementation does, or why the premise does not hold) is empty")
        if status != "observed":
            continue
        reading, count = row.get("reading"), counts.get(row["id"], 0)
        if not isinstance(reading, int) or isinstance(reading, bool) or not 0 <= reading <= count:
            errors.append("%s: reading must be an integer 1-%d (the matching reading) or 0 (none matches)" % (where, count))
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(where + ": observed needs evidence (a probe with its output, or code path:line)")
            continue
        for number, item in enumerate(evidence, 1):
            if not isinstance(item, dict) or item.get("kind") not in EVIDENCE_KINDS or not _text(item.get("detail")):
                errors.append("%s: evidence %d needs kind probe|code and detail" % (where, number))
                continue
            if item["kind"] == "code":
                path, line = item.get("path"), item.get("line")
                total = _line_count(repo_root, path) if _text(path) else None
                if total is None:
                    errors.append("%s: evidence %d: path %r does not exist in /app" % (where, number, path))
                elif not isinstance(line, int) or isinstance(line, bool) or not 1 <= line <= total:
                    errors.append("%s: evidence %d: line must be an integer within %s (1-%d)" % (where, number, path, total))
    labelled = set(row["id"] for row in final)
    missing = [i for i in ids if i not in labelled]
    if missing:
        errors.append("%d question ids have no line: %s%s" % (
            len(missing), ", ".join(missing[:30]), " ..." if len(missing) > 30 else ""))
    stats = {"in_scope": len(ids), "labelled": len(labelled),
             "observed": sum(1 for r in final if r.get("status") == "observed"),
             "not_applicable": sum(1 for r in final if r.get("status") == "not_applicable"),
             "undetermined": sum(1 for r in final if r.get("status") == "undetermined")}
    return errors, stats


def main(argv):
    directory = argv[1] if len(argv) > 1 else "/tmp/v68"
    repository = argv[2] if len(argv) > 2 else "/app"
    ids, counts, rows, parse_errors = load(directory)
    errors, stats = lint(ids, counts, rows, repository)
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
