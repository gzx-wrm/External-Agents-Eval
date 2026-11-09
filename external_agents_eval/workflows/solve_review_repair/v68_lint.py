"""V6.8 analysis-output checker (standalone: also copied into the task container as lint.py).

Checks the analysis files an agent writes under /tmp/v68 (or the same structure from a
tool-free control response): obligations quote the task (at most MAX_OBLIGATIONS), entities
point at real lines, items are questions (v6.8.1: no requirement items), and every
obligation x extension and every entity x obligation pair is covered or explicitly marked
not applicable.
Python 3.6 compatible; no project imports.
"""
import json
import os
import re
import sys

EXTENSIONS = ("entity", "boundary", "analog_form", "external_spec")
ENTITY_KINDS = ("entry_point", "sibling", "lifecycle", "existing_test", "analog", "other")
KINDS = ("question",)  # v6.8.1: targeted items are questions only
MAX_OBLIGATIONS = 12  # v6.8.1: hard cap (plan)
SYMBOL_WINDOW = 3
SOFT_MAX_ITEMS = 220
MIN_REASON_CHARS = 10
MAX_LISTED_ERRORS = 40


def normalize(text):
    return " ".join((text or "").replace("`", "").split())


def quote_in_task(quote, task_text):
    """Exact (whitespace/backtick-insensitive) task quote; '...' joins fragments in order."""
    task = normalize(task_text)
    fragments = [normalize(part) for part in re.split(r"\.\.\.|…", quote or "")]
    fragments = [part for part in fragments if part]
    if not fragments or sum(len(part) for part in fragments) < 8:
        return False
    position = 0
    for part in fragments:
        found = task.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


class FileRepo(object):
    """Lines of real repository files (inside the container)."""

    def __init__(self, root):
        self.root = root
        self.cache = {}

    def lines(self, path):
        if path in self.cache:
            return self.cache[path]
        full = path if os.path.isabs(path) else os.path.join(self.root, path)
        result = None
        if os.path.isfile(full):
            try:
                with open(full, encoding="utf-8", errors="replace") as handle:
                    result = dict(enumerate(handle.read().split("\n"), 1))
            except OSError:
                result = None
        self.cache[path] = result
        return result


class EvidenceRepo(object):
    """Lines known from frozen evidence excerpts (tool-free control)."""

    def __init__(self, evidence):
        self.files = {}
        for item in evidence:
            known = self.files.setdefault(item["path"], {})
            for offset, text in enumerate((item.get("text") or "").split("\n")):
                known[item["line_start"] + offset] = text

    def lines(self, path):
        if path.startswith("./"):
            path = path[2:]
        return self.files.get(path) or self.files.get(path.lstrip("/"))


def load_scope(directory):
    path = os.path.join(directory, "scope.json")
    if not os.path.exists(path):
        return {"mode": "full"}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_dir(directory, scope=None):
    """Read the analysis files; returns (data, parse_errors)."""
    scope = scope or {"mode": "full"}
    data = {"obligations": [], "entities": [], "items": [], "not_applicable": []}
    errors = []
    for name in ("obligations", "entities", "not_applicable"):
        path = os.path.join(directory, name + ".json")
        if not os.path.exists(path):
            if name == "obligations" or (name == "entities" and scope["mode"] != "plan"):
                errors.append("%s.json is missing" % name)
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
        except ValueError as error:
            errors.append("%s.json is not valid JSON: %s" % (name, error))
            continue
        if not isinstance(value, list):
            errors.append("%s.json must be a JSON list" % name)
            continue
        data[name] = value
    path = os.path.join(directory, "items.jsonl")
    if not os.path.exists(path):
        if scope["mode"] != "plan":
            errors.append("items.jsonl is missing")
    else:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError as error:
                    errors.append("items.jsonl line %d is not valid JSON: %s" % (number, error))
                    continue
                if isinstance(item, dict):
                    item["_line"] = number
                    data["items"].append(item)
                else:
                    errors.append("items.jsonl line %d is not a JSON object" % number)
    return data, errors


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _line_ok(repo, path, line):
    if not _text(path) or not isinstance(line, int) or isinstance(line, bool):
        return "needs a string path and an integer line"
    lines = repo.lines(path)
    if lines is None:
        return "path %s does not exist" % path
    if line not in lines:
        return "line %d is not in %s" % (line, path)
    return None


def lint(task_text, data, repo, scope=None):
    """Returns (errors, warnings, stats). scope: None (full), {"mode": "plan"} (obligations only)
    or {"mode": "batch", "targets": [...]} (entities/items/not_applicable for these obligations;
    obligations.json is the frozen plan)."""
    scope = scope or {"mode": "full"}
    errors, warnings = [], []
    obligations = {}
    for index, item in enumerate(data.get("obligations") or [], 1):
        where = "obligation #%d" % index
        if not isinstance(item, dict):
            errors.append(where + " is not an object")
            continue
        oid = item.get("id")
        if not (isinstance(oid, str) and re.match(r"^O\d+$", oid)) or oid in obligations:
            errors.append(where + ": id must be a unique O<number>")
            continue
        obligations[oid] = item
        if not _text(item.get("statement")):
            errors.append(oid + ": statement is empty")
        if not quote_in_task(item.get("task_quote"), task_text):
            errors.append(oid + ": task_quote is not an exact quote of the task text")
    if not obligations:
        errors.append("no obligations")
    if scope["mode"] != "batch" and len(obligations) > MAX_OBLIGATIONS:
        errors.append("%d obligations; at most %d are allowed: merge closely related behaviors into one obligation"
                      % (len(obligations), MAX_OBLIGATIONS))
    if scope["mode"] == "plan":
        stats = {"obligations": len(obligations)}
        return errors, warnings, stats
    targets = set(scope.get("targets") or obligations)
    unknown = sorted(targets - set(obligations))
    if unknown:
        errors.append("scope targets not in obligations.json: " + ", ".join(unknown))

    entities = {}
    for index, item in enumerate(data.get("entities") or [], 1):
        where = "entity #%d" % index
        if not isinstance(item, dict):
            errors.append(where + " is not an object")
            continue
        eid = item.get("id")
        if not (isinstance(eid, str) and re.match(r"^E\d+$", eid)) or eid in entities:
            errors.append(where + ": id must be a unique E<number>")
            continue
        entities[eid] = item
        if item.get("kind") not in ENTITY_KINDS:
            errors.append("%s: kind must be one of %s" % (eid, ", ".join(ENTITY_KINDS)))
        problem = _line_ok(repo, item.get("path"), item.get("line"))
        if problem:
            errors.append("%s: %s" % (eid, problem))
        elif not _text(item.get("symbol")):
            errors.append(eid + ": symbol is empty")
        else:
            lines = repo.lines(item["path"])
            window = [lines.get(n, "") for n in range(item["line"] - SYMBOL_WINDOW, item["line"] + SYMBOL_WINDOW + 1)]
            if not any(item["symbol"] in text for text in window):
                errors.append("%s: symbol %r does not appear within %d lines of %s:%d"
                              % (eid, item["symbol"], SYMBOL_WINDOW, item["path"], item["line"]))
        linked = item.get("obligations")
        if not isinstance(linked, list) or not linked or any(o not in targets for o in linked):
            errors.append(eid + ": obligations must be a non-empty list of your target obligation ids ("
                          + ", ".join(sorted(targets)) + ")")
        if not _text(item.get("why")):
            errors.append(eid + ": why is empty")

    not_applicable = set()
    for index, item in enumerate(data.get("not_applicable") or [], 1):
        where = "not_applicable #%d" % index
        if not isinstance(item, dict):
            errors.append(where + " is not an object")
            continue
        if item.get("obligation") not in targets or item.get("ext") not in EXTENSIONS:
            errors.append(where + ": needs a target obligation and ext in " + ", ".join(EXTENSIONS))
            continue
        if not (_text(item.get("reason")) and len(item["reason"].strip()) >= MIN_REASON_CHARS):
            errors.append(where + ": reason is missing or too short")
            continue
        not_applicable.add((item["obligation"], item["ext"], item.get("entity")))

    covered, entity_pairs, seen = set(), set(), set()
    counts = dict((ext, 0) for ext in EXTENSIONS)
    for item in data.get("items") or []:
        where = "items.jsonl line %s" % item.get("_line", "?")
        ext, kind, oid = item.get("ext"), item.get("kind"), item.get("obligation")
        if ext not in EXTENSIONS:
            errors.append(where + ": ext must be one of " + ", ".join(EXTENSIONS))
            continue
        if oid not in targets:
            errors.append(where + ": obligation must be one of your target obligation ids ("
                          + ", ".join(sorted(targets)) + ")")
            continue
        if kind not in KINDS:
            errors.append(where + ": kind must be \"question\" (requirement items are not allowed; when the task "
                          "text decides the outcome, still list the readings and put the deciding sentence in task_quote)")
            continue
        for field in ("given", "when", "then"):
            if not _text(item.get(field)):
                errors.append("%s: %s is empty" % (where, field))
        signature = normalize(" | ".join(str(item.get(f)) for f in ("given", "when", "then"))).lower()
        if signature in seen:
            errors.append(where + ": duplicate of an earlier item")
        seen.add(signature)
        entity = item.get("entity")
        if ext == "entity" and entity not in entities:
            errors.append(where + ": ext entity needs a known entity id")
        elif entity is not None and entity not in entities:
            errors.append(where + ": entity must be null or a known entity id")
        alternatives = item.get("alternatives") or []
        if not isinstance(alternatives, list):
            errors.append(where + ": alternatives must be a list")
            alternatives = []
        quote = item.get("task_quote")
        if quote is not None and quote != "" and not quote_in_task(quote, task_text):
            errors.append(where + ": task_quote must be null or an exact quote of the task text")
        distinct = set(normalize(a).lower() for a in alternatives if _text(a))
        if len(distinct) < 2:
            errors.append(where + ": a question needs at least two distinct alternatives (readings)")
        for banned in ("favored", "decision", "answer", "rules_out"):
            if item.get(banned):
                errors.append("%s: a question must not carry %s" % (where, banned))
        observations = item.get("observations") or []
        if not isinstance(observations, list):
            errors.append(where + ": observations must be a list")
            observations = []
        for number, observation in enumerate(observations, 1):
            if not isinstance(observation, dict) or not _text(observation.get("text")):
                errors.append("%s: observation %d needs text" % (where, number))
                continue
            problem = _line_ok(repo, observation.get("path"), observation.get("line"))
            if problem:
                errors.append("%s: observation %d: %s" % (where, number, problem))
        if ext == "external_spec":
            spec = item.get("spec") or {}
            if not (isinstance(spec, dict) and _text(spec.get("name")) and _text(spec.get("rule"))):
                errors.append(where + ": external_spec needs spec.name (the standard) and spec.rule (the rule)")
        counts[ext] += 1
        covered.add((oid, ext))
        if ext == "entity" and entity in entities:
            entity_pairs.add((entity, oid))

    for oid in sorted(targets & set(obligations)):
        for ext in EXTENSIONS:
            if (oid, ext) not in covered and (oid, ext, None) not in not_applicable:
                errors.append("%s x %s: no item and no not_applicable entry with a reason" % (oid, ext))
    for eid, entity in entities.items():
        for oid in entity.get("obligations") or []:
            if oid in targets and (eid, oid) not in entity_pairs and (oid, "entity", eid) not in not_applicable:
                errors.append("%s x %s: no ext=entity item for this entity and obligation" % (eid, oid))
    total = len(data.get("items") or [])
    if total > SOFT_MAX_ITEMS:
        warnings.append("%d items (soft maximum %d): merge near-duplicates" % (total, SOFT_MAX_ITEMS))
    stats = {"obligations": len(obligations), "entities": len(entities), "items": total,
             "items_by_ext": counts, "not_applicable": len(not_applicable)}
    return errors, warnings, stats


def report(parse_errors, errors, warnings, stats):
    problems = list(parse_errors) + list(errors)
    lines = []
    if problems:
        lines.append("LINT FAILED (%d problems)" % len(problems))
        lines.extend("- " + p for p in problems[:MAX_LISTED_ERRORS])
        if len(problems) > MAX_LISTED_ERRORS:
            lines.append("- ... %d more" % (len(problems) - MAX_LISTED_ERRORS))
    else:
        lines.append("LINT OK")
    lines.extend("warning: " + w for w in warnings)
    lines.append("stats: " + json.dumps(stats, sort_keys=True))
    return "\n".join(lines)


def main(argv):
    directory = argv[1] if len(argv) > 1 else "/tmp/v68"
    repository = argv[2] if len(argv) > 2 else "/app"
    with open(os.path.join(directory, "task.md"), encoding="utf-8") as handle:
        task_text = handle.read()
    scope = load_scope(directory)
    data, parse_errors = load_dir(directory, scope)
    errors, warnings, stats = lint(task_text, data, FileRepo(repository), scope)
    print(report(parse_errors, errors, warnings, stats))
    return 1 if (parse_errors or errors) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
