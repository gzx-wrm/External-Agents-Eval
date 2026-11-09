"""V6.7 ledger and check-point generation (tool-free, on frozen V6.6 A evidence).

Changes against V6.6.4, as approved on 2026-11-06:
1. Inferred points are questions. Only the task text can decide a check point
   ("requirement"); everything else is a "question" with every reasonable reading and
   optional observations (what the closest analog does, what a contract or a named
   standard says). Observations never choose a reading, and no inferred point carries
   an exclusion.
2. The Solver view lists every requirement first, with its exact task quote, then the
   questions; there is no "Resolved by probe" and no "Excludes" for inferred points.
3. Ledger fidelity: an interface form the task does not state is not an obligation, and
   a restated task sentence keeps its conditions. Mechanical checks flag call/signature
   spans that neither the task nor the evidence contains, and requirement points whose
   paraphrase drops a qualifier of their quote.
4. Coverage: two facets are added (interaction with existing mechanisms; external
   specification semantics) and questions have their own, larger per-target limit.
"""
from __future__ import annotations

import copy
import difflib
import json
import re
from collections import defaultdict
from typing import Any

from jsonschema import Draft202012Validator

from .ft_inference import FT_SYSTEM_PROMPT
from .v66_checkpoints import (
    MERGED_ABSENCE,
    BatchReport,
    _evidence_lines,
    _source_text,
    _squash,
    ledger_targets,
)
from .v66_prompts import FACETS as V66_FACETS

VERSION = "v6.7.0"

FACETS = {
    **V66_FACETS,
    "D9": ("interaction with existing mechanisms",
           "how the target behaves together with the existing mechanisms it touches: caching, "
           "lifecycle, reset and re-configuration, cancellation, error and retry paths, and "
           "ordering"),
    "D10": ("external specification semantics",
            "rules of a named external standard, protocol, data format or programming-language "
            "semantics that the target's inputs or outputs follow; name the specification and "
            "the rule"),
}
KINDS = ("requirement", "question")
EVIDENCE_KINDS = ("analog", "contract", "convention")
MAX_REQUIREMENTS_PER_TARGET = 4
MAX_QUESTIONS_PER_TARGET = 8
SOLVER_VIEW_MAX_CHARS = 24000
DUPLICATE_RATIO = 0.9
# Words whose loss changes the meaning of a quoted condition.
QUALIFIERS = ("not", "n't", "never", "unless", "except", "without", "only", "even", "explicitly",
              "until", "at most", "at least", "exactly", "once")

LEDGER_FIDELITY = """V6.7 FIDELITY RULES
- An obligation may name an interface form (function or method signature, argument order,
  call syntax, return type, field, file or option name) only when the task text states it or
  the supplied repository evidence shows that exact existing interface. When the task names a
  capability without its call form, state the capability without a signature and add an open
  question listing the reasonable forms, including the form of the closest existing analog
  shown in the evidence.
- When an obligation restates task text, keep every condition and qualifier of that text
  (for example not, only, unless, except, even when, until, at most, explicitly); never
  narrow or broaden what the task says.
"""


def ledger_system_prompt() -> str:
    """The unchanged FT prompt with the V6.7 fidelity rules before its output contract."""
    marker = "Return exactly one JSON object"
    head, found, tail = FT_SYSTEM_PROMPT.partition(marker)
    if not found:
        raise ValueError("FT prompt layout changed")
    return head + LEDGER_FIDELITY + "\n" + found + tail


_TEXT = {"type": "string", "minLength": 1}
_NULLABLE_TEXT = {"type": ["string", "null"]}
_TARGET = {"type": "string", "pattern": "^[OP][0-9]{3}$"}
_SOURCE = {"oneOf": [
    {"type": "object", "additionalProperties": False, "required": ["kind", "field", "quote"],
     "properties": {"kind": {"const": "task"}, "field": {"enum": ["title", "description"]},
                    "quote": _TEXT}},
    {"type": "object", "additionalProperties": False,
     "required": ["kind", "path", "line_start", "line_end"],
     "properties": {"kind": {"const": "repository"}, "path": _TEXT,
                    "line_start": {"type": "integer", "minimum": 1},
                    "line_end": {"type": "integer", "minimum": 1}}},
]}
_OBSERVATION = {
    "type": "object", "additionalProperties": False,
    "required": ["kind", "observation", "basis_quote", "convention", "sources"],
    "properties": {"kind": {"enum": list(EVIDENCE_KINDS)}, "observation": _TEXT,
                   "basis_quote": _NULLABLE_TEXT, "convention": _NULLABLE_TEXT,
                   "sources": {"type": "array", "items": _SOURCE}},
}
CHECKPOINT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object", "additionalProperties": False, "required": ["checkpoints", "coverage"],
    "properties": {
        "checkpoints": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["targets", "facet", "kind", "given", "when", "then", "sources",
                         "rules_out", "alternatives", "observations"],
            "properties": {
                "targets": {"type": "array", "minItems": 1, "uniqueItems": True, "items": _TARGET},
                "facet": {"enum": list(FACETS)},
                "kind": {"enum": list(KINDS)},
                "given": _TEXT, "when": _TEXT, "then": _TEXT,
                "sources": {"type": "array", "items": _SOURCE},
                "rules_out": _NULLABLE_TEXT,
                "alternatives": {"type": "array", "items": _TEXT},
                "observations": {"type": "array", "items": _OBSERVATION},
            },
        }},
        "coverage": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["target", "facet", "status", "reason"],
            "properties": {"target": _TARGET, "facet": {"enum": list(FACETS)},
                           "status": {"enum": ["covered", "not_applicable"]},
                           "reason": _NULLABLE_TEXT},
        }},
    },
}
ITEM_SCHEMA = CHECKPOINT_SCHEMA["properties"]["checkpoints"]["items"]
COVERAGE_ITEM_SCHEMA = CHECKPOINT_SCHEMA["properties"]["coverage"]["items"]

CHECKPOINT_EXAMPLE = """EXAMPLE (synthetic; not from this repository)
Task: "Add parse_env(text) returning a dict from KEY=VALUE lines; lines starting with # are
comments. A line without = is not an error." Ledger: O001 "parse_env returns a dict of
KEY=VALUE lines, ignoring comment lines." Evidence: config.py lines 40-52 show parse_ini, whose
line 44 is `key, value = (part.strip() for part in line.split("=", 1))`.
- targets [O001], D4, requirement: given a line "FOO" without "=", when parsed, then no error
  is raised; sources [task "A line without = is not an error."]; rules_out "raising ValueError
  for a line without =".
- targets [O001], D2, question: given "A = 1 ", when parsed, then whether surrounding spaces of
  key and value are kept is not stated by the task; alternatives ["keys and values are
  stripped: {"A": "1"}", "text is kept as written: {"A ": " 1 "}"]; observations [{kind analog,
  observation "the existing parse_ini strips spaces around keys and values", basis_quote
  `part.strip() for part in line.split("=", 1)`, sources [config.py 44-44]}].
- O001/D6 not_applicable: "parse_env is a pure function with no state"."""


def _facet_lines() -> str:
    return "\n".join(f"{key} {name}: {question}." for key, (name, question) in FACETS.items())


def checkpoint_system_prompt() -> str:
    return f"""ROLE: Coverage Planner (V6.7 C; fixed-evidence inference)
Use only the original task, the frozen requirement ledger and the public repository evidence
supplied in the input. Never access hidden tests, verifier output, reference solutions or
historical evaluations. You have no tools; do not claim to have read omitted files or executed
commands.

Goal: expand the ledger into observable check points, so that implementers and reviewers see
every detail where reasonable implementations of a target would differ: the task's own
decisions as requirements, and every detail the task leaves undecided as a question.

For every O and P target in the current batch, answer each facet's question for that target's
own inputs, outputs, state and interactions:
{_facet_lines()}

Create a check point only when both gates hold:
- observable: a caller or test can observe the difference through the public surface named by
  the task, the target or the existing code it extends;
- divergent: at least two reasonable implementations of the target would behave differently
  there. A check point that only restates its target is not allowed.

Two kinds of check points:
- requirement: the task text itself decides the outcome. Quote in sources the exact task text
  that decides it. given/when/then must keep every condition and qualifier of that quote
  (not, only, unless, except, even when, until, at most, explicitly, ...). rules_out may name a
  reasonable implementation that contradicts the quote, or be null. alternatives and
  observations are empty.
- question: the task text does not decide the outcome. then states what is undecided;
  alternatives list every reasonable reading (at least two); rules_out is null. Attach in
  observations the facts that bear on the question: what the closest existing analog in the
  evidence does (kind analog), what a type, schema or format the data reaches requires (kind
  contract), or what a named external standard, protocol, format or language rule says (kind
  convention). analog and contract observations cite repository lines and copy into basis_quote
  the exact text that shows the behavior; a convention observation names the specification in
  convention. Observations report facts only; never choose or rank a reading, and never present
  an analog or a convention as a decision of the task.
Never add behavior the task does not request, and never let a check point reject input the task
allows. Where the task explicitly allows any choice, the facet is not_applicable.

When the same rule applies to several targets, write one check point listing all of them in
targets. coverage must list every target in the current batch with all ten facets exactly once;
a facet is covered exactly when some check point lists that target and facet.

Keep omitted, explicit null and empty as separate cases; never merge them into one reading such
as "unset or empty". At most {MAX_REQUIREMENTS_PER_TARGET} requirements and
{MAX_QUESTIONS_PER_TARGET} questions per target; keep those whose divergence would most change
observable results. Write given/when/then about observable behavior, not implementation
structure. Task quotes must be exact contiguous text; repository sources must lie inside the
supplied evidence lines. Return exactly one JSON object conforming to the schema below, without
markdown fences or commentary.

{CHECKPOINT_EXAMPLE}

OUTPUT JSON SCHEMA
{json.dumps(CHECKPOINT_SCHEMA, ensure_ascii=False, indent=2)}"""


def _rule_text(point: dict[str, Any]) -> str:
    return _squash(" | ".join((point["given"], point["when"], point["then"]))).lower()


def _same_rule(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (a["facet"] == b["facet"] and a["kind"] == b["kind"]
            and difflib.SequenceMatcher(None, _rule_text(a), _rule_text(b)).ratio() >= DUPLICATE_RATIO)


def limits(kind: str) -> int:
    return MAX_REQUIREMENTS_PER_TARGET if kind == "requirement" else MAX_QUESTIONS_PER_TARGET


def check_batch(payload: Any, *, ledger: dict[str, Any], packet: dict[str, Any],
                batch: list[str] | None = None) -> BatchReport:
    """Every contract violation of one output; batch=None checks all ledger targets."""
    report = BatchReport()
    for error in Draft202012Validator(CHECKPOINT_SCHEMA).iter_errors(payload):
        path = list(error.absolute_path)
        message = f"schema: {'/'.join(map(str, path)) or '(root)'}: {error.message}"
        if len(path) >= 2 and path[0] == "checkpoints" and isinstance(path[1], int):
            report.points[path[1]].append(message)
        else:
            report.batch.append(message)
    if not isinstance(payload, dict) or not isinstance(payload.get("checkpoints"), list):
        report.structure, report.batch = report.batch, []
        return report
    targets = ledger_targets(ledger)
    scope = list(targets) if batch is None else list(batch)
    task_text = {"title": packet["task"]["title"], "description": packet["task"]["description"]}
    lines = _evidence_lines(packet)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    covered: dict[tuple[str, str], int] = defaultdict(int)
    points = [(index, point) for index, point in enumerate(payload["checkpoints"])
              if index not in report.points]
    for index, point in points:
        errors = report.points[index]
        where = f"checkpoints/{index} ({','.join(point['targets'])}/{point['facet']})"
        unknown = [target for target in point["targets"] if target not in targets]
        if unknown:
            errors.append(f"{where}: unknown target {', '.join(unknown)}")
            continue
        outside = [target for target in point["targets"] if target not in scope]
        if outside:
            errors.append(f"{where}: targets outside this batch: {', '.join(outside)}")
            continue
        for target in point["targets"]:
            counts[(target, point["kind"])] += 1
            covered[(target, point["facet"])] += 1
        kinds = {source["kind"] for source in point["sources"]}
        if point["kind"] == "requirement":
            if "task" not in kinds:
                errors.append(f"{where}: a requirement needs the exact task quote that decides it")
            if point["alternatives"] or point["observations"]:
                errors.append(f"{where}: a requirement has no alternatives or observations; "
                              "if the task does not decide it, make it a question")
        else:
            if len(point["alternatives"]) < 2:
                errors.append(f"{where}: a question needs at least two alternatives")
            if point["rules_out"] is not None:
                errors.append(f"{where}: a question lists alternatives; rules_out must be null")
        for source in point["sources"]:
            text = _source_text(source, task_text, lines)
            if source["kind"] == "task":
                if source["quote"] not in text:
                    errors.append(f"{where}: task quote is not exact: {source['quote'][:80]}")
            elif text is None:
                errors.append(f"{where}: repository lines are outside the supplied evidence: "
                              f"{source['path']} {source['line_start']}-{source['line_end']}")
        for number, item in enumerate(point["observations"]):
            label = f"{where} observation {number}"
            if item["kind"] == "convention":
                if not (item["convention"] or "").strip():
                    errors.append(f"{label}: name the convention or specification")
                continue
            texts = []
            for source in item["sources"]:
                text = _source_text(source, task_text, lines) if source["kind"] == "repository" else None
                if source["kind"] != "repository":
                    errors.append(f"{label}: {item['kind']} observations cite repository lines")
                elif text is None:
                    errors.append(f"{label}: repository lines are outside the supplied evidence: "
                                  f"{source['path']} {source['line_start']}-{source['line_end']}")
                else:
                    texts.append(text)
            if not texts:
                errors.append(f"{label}: an {item['kind']} observation cites repository lines")
            quote = _squash(item["basis_quote"] or "")
            if not quote:
                errors.append(f"{label}: copy the decisive text into basis_quote")
            elif texts and not any(quote in _squash(text) for text in texts):
                errors.append(f"{label}: basis_quote is not exact text from the cited lines")
        for name in ("given", "when", "then"):
            if MERGED_ABSENCE.search(point[name]):
                errors.append(f"{where}: {name} merges omitted/null/empty; write separate cases")
    report.points = defaultdict(list, {index: errors for index, errors in report.points.items() if errors})
    for (target, kind), count in counts.items():
        if count > limits(kind):
            report.batch.append(f"{target}: {count} {kind} check points exceed the limit of {limits(kind)}")
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            (i, a), (j, b) = points[first], points[second]
            if _same_rule(a, b):
                report.batch.append(f"checkpoints/{i} and checkpoints/{j}: the same rule; merge them into one "
                                    f"check point listing targets {sorted(set(a['targets']) | set(b['targets']))}")
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), list) else []
    seen: dict[tuple[str, str], str] = {}
    for entry in coverage:
        if not Draft202012Validator(COVERAGE_ITEM_SCHEMA).is_valid(entry):
            continue
        key = (entry["target"], entry["facet"])
        if key in seen:
            report.batch.append(f"coverage: duplicate {key[0]}/{key[1]}")
        seen[key] = entry["status"]
        if entry["status"] == "not_applicable" and not (entry["reason"] or "").strip():
            report.batch.append(f"coverage: {key[0]}/{key[1]} needs a not_applicable reason")
    for target in scope:
        for facet in FACETS:
            status = seen.get((target, facet))
            if status is None:
                report.batch.append(f"coverage: missing {target}/{facet}")
            elif (status == "covered") != bool(covered.get((target, facet))):
                report.batch.append(f"coverage: {target}/{facet} is {status} but has "
                                    f"{covered.get((target, facet), 0)} check points")
    for key in seen:
        if key[0] not in targets:
            report.batch.append(f"coverage: unknown target {key[0]}")
        elif key[0] not in scope:
            report.batch.append(f"coverage: target outside this batch {key[0]}")
    return report


def merge_duplicates(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    for point in points:
        for earlier in kept:
            if _same_rule(earlier, point):
                added = [target for target in point["targets"] if target not in earlier["targets"]]
                earlier["targets"] = earlier["targets"] + added
                merged.append({"into_targets": list(earlier["targets"]), "facet": point["facet"],
                               "from_targets": list(point["targets"])})
                break
        else:
            kept.append(point)
    return kept, merged


def enforce_limit(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep each target's first requirements and questions up to their limits, in output order."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    kept, trimmed = [], []
    for point in points:
        kind = point["kind"]
        allowed = [t for t in point["targets"] if counts[(t, kind)] < limits(kind)]
        removed = [t for t in point["targets"] if t not in allowed]
        if removed:
            trimmed.append({"facet": point["facet"], "kind": kind, "targets_removed": removed,
                            "point_dropped": not allowed})
        if not allowed:
            continue
        point["targets"] = allowed
        for target in allowed:
            counts[(target, kind)] += 1
        kept.append(point)
    return kept, trimmed


def rebuild_coverage(points: list[dict[str, Any]], model_coverage: Any,
                     scope: list[str]) -> list[dict[str, Any]]:
    covered = {(target, point["facet"]) for point in points for target in point["targets"]}
    reasons: dict[tuple[str, str], str] = {}
    for entry in model_coverage if isinstance(model_coverage, list) else []:
        if (Draft202012Validator(COVERAGE_ITEM_SCHEMA).is_valid(entry)
                and entry["status"] == "not_applicable" and (entry["reason"] or "").strip()):
            reasons.setdefault((entry["target"], entry["facet"]), entry["reason"])
    rows = []
    for target in scope:
        for facet in FACETS:
            status = ("covered" if (target, facet) in covered
                      else "not_applicable" if (target, facet) in reasons else "uncovered")
            rows.append({"target": target, "facet": facet, "status": status,
                         "reason": reasons.get((target, facet)) if status == "not_applicable" else None})
    return rows


def finalize_batch(payload: Any, report: BatchReport, *, scope: list[str]):
    """After the last attempt: drop invalid points, merge duplicates, trim, rebuild coverage."""
    if report.structure:
        coverage = [{"target": t, "facet": f, "status": "uncovered", "reason": None}
                    for t in scope for f in FACETS]
        return [], coverage, {"status": "failed", "errors": report.structure}
    dropped = [{"index": index, "errors": errors} for index, errors in sorted(report.points.items())]
    points = [copy.deepcopy(point) for index, point in enumerate(payload["checkpoints"])
              if index not in report.points]
    points, merged = merge_duplicates(points)
    points, trimmed = enforce_limit(points)
    coverage = rebuild_coverage(points, payload.get("coverage"), scope)
    return points, coverage, {"status": "valid" if not report.messages() else "degraded",
                              "dropped": dropped, "merged": merged, "trimmed": trimmed,
                              "unresolved": report.batch}


# --- mechanical fidelity checks --------------------------------------------------------------

CODE_SPAN = re.compile(r"`([^`]+)`")


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def unquoted_interfaces(text: str, task: str, evidence_text: str) -> list[str]:
    """Call or signature spans (with parentheses) that neither the task nor the evidence contains."""
    haystack = _norm(task) + "\n" + _norm(evidence_text)
    return [span for span in CODE_SPAN.findall(text or "")
            if "(" in span and _norm(span) not in haystack]


def _has(word: str, text: str) -> bool:
    if word == "n't":
        return "n't" in text
    return re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text) is not None


NEGATIONS = ("not", "n't", "never")


def dropped_qualifiers(point: dict[str, Any]) -> list[str]:
    """Qualifiers of the cited task quote that the point's own wording lost (heuristic flag)."""
    quotes = " ".join(s["quote"] for s in point["sources"] if s["kind"] == "task").lower()
    body = " ".join(filter(None, (point["given"], point["when"], point["then"],
                                  point.get("rules_out")))).lower()
    missing = []
    if (any(_has(word, quotes) for word in NEGATIONS)
            and not any(_has(word, body) for word in (*NEGATIONS, "no", "without"))):
        missing.append("not")
    missing += [word for word in QUALIFIERS
                if word not in NEGATIONS and _has(word, quotes) and not _has(word, body)]
    return missing


def fidelity_flags(ledger: dict[str, Any], points: list[dict[str, Any]], packet: dict[str, Any]) -> dict[str, Any]:
    task = packet["task"]["title"] + "\n" + packet["task"]["description"]
    evidence = "\n".join(item["text"] for item in packet["evidence"])
    ledger_flags = []
    for field in ("obligations", "preservation_obligations"):
        for item in ledger[field]:
            spans = unquoted_interfaces(item["content"]["statement"], task, evidence)
            if spans:
                ledger_flags.append({"id": item["id"], "unquoted_interfaces": spans})
    point_flags = []
    for index, point in enumerate(points, 1):
        entry: dict[str, Any] = {}
        if point["kind"] == "requirement":
            missing = dropped_qualifiers(point)
            if missing:
                entry["dropped_qualifiers"] = missing
        spans = unquoted_interfaces(" ".join((point["given"], point["when"], point["then"])), task, evidence)
        if spans and point["kind"] == "requirement":
            entry["unquoted_interfaces"] = spans
        if entry:
            point_flags.append({"cp": f"CP{index}", **entry})
    return {"ledger": ledger_flags, "checkpoints": point_flags}


# --- delivery ----------------------------------------------------------------------------------

def _requirement_line(cp: str, point: dict[str, Any], flags: dict[str, Any]) -> str:
    quotes = " / ".join(_squash(s["quote"]) for s in point["sources"] if s["kind"] == "task")
    line = (f"- {cp} {','.join(point['targets'])}/{point['facet']}: given {point['given']}; "
            f"when {point['when']}; then {point['then']} (task: \"{quotes[:400]}\")")
    if flags.get("dropped_qualifiers") or flags.get("unquoted_interfaces"):
        line += " [check against the task quote; the paraphrase may be incomplete]"
    return line


def _question_line(cp: str, point: dict[str, Any]) -> str:
    readings = " | ".join(f"({n}) {text}" for n, text in enumerate(point["alternatives"], 1))
    line = (f"- {cp} {','.join(point['targets'])}/{point['facet']}: given {point['given']}; "
            f"when {point['when']}; undecided: {point['then']} Readings: {readings}")
    notes = []
    for item in point["observations"]:
        if item["kind"] == "convention":
            notes.append(f"{item['convention']}: {item['observation']}")
        else:
            where = ", ".join(f"{s['path']}:{s['line_start']}-{s['line_end']}" for s in item["sources"]
                              if s["kind"] == "repository")
            notes.append(f"{item['kind']} {where}: {item['observation']}")
    if notes:
        line += " Observations (facts, not decisions): " + "; ".join(notes)
    return line


def delivery_views(payload: dict[str, Any], flags: dict[str, Any],
                   max_chars: int = SOLVER_VIEW_MAX_CHARS) -> dict[str, Any]:
    points = payload["checkpoints"]
    by_cp = {entry["cp"]: entry for entry in flags["checkpoints"]}
    numbered = [(f"CP{index}", point) for index, point in enumerate(points, 1)]
    requirements = [_requirement_line(cp, p, by_cp.get(cp, {})) for cp, p in numbered if p["kind"] == "requirement"]
    questions = [(cp, p) for cp, p in numbered if p["kind"] == "question"]
    # New-behavior questions before preservation questions; output order otherwise.
    questions.sort(key=lambda item: 0 if any(t.startswith("O") for t in item[1]["targets"]) else 1)
    sections = ["### Requirements stated by the task (the quoted task text governs)", *requirements]
    notes = [f"- {entry['id']} names {', '.join('`' + s + '`' for s in entry['unquoted_interfaces'])}, which "
             "neither the task nor the repository evidence states; treat that interface form as undecided."
             for entry in flags["ledger"]]
    if notes:
        sections += ["### Ledger notes", *notes]
    sections.append("### Questions the task leaves undecided (decide them yourself from the task and the "
                    "code; none of these is a requirement)")
    shown, dropped = [], 0
    budget = max_chars - len("\n".join(sections))
    for cp, point in questions:
        line = _question_line(cp, point)
        if len(line) + 1 > budget:
            dropped += 1
            continue
        shown.append(line)
        budget -= len(line) + 1
    text = "\n".join(sections + shown)
    return {"solver_text": text, "review_targets": points, "requirements": len(requirements),
            "questions_shown": len(shown), "questions_dropped_from_solver_view": dropped,
            "chars": len(text)}


def summarize(payload: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    points = payload["checkpoints"]

    def tally(key: str) -> dict[str, int]:
        return {value: sum(1 for p in points if p[key] == value) for value in sorted({p[key] for p in points})}

    return {"targets": len(ledger["obligations"]) + len(ledger["preservation_obligations"]),
            "checkpoints": len(points), "by_kind": tally("kind"), "by_facet": tally("facet"),
            "observations": sum(len(p["observations"]) for p in points),
            "covered_cells": sum(1 for e in payload["coverage"] if e["status"] == "covered"),
            "uncovered_cells": sum(1 for e in payload["coverage"] if e["status"] == "uncovered")}
