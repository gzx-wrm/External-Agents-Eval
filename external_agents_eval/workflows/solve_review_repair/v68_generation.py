"""V6.8 targeted requirement analysis (generation only; no Solver, review or scoring).

Mini agents explore the baseline repository themselves (no frozen evidence, no separate
evidence agent, no V6.6 ledger). Per task:
  plan episode  -> /tmp/v68/obligations.json: what the task requires, each with an exact quote;
  batch episodes (three obligations each; user, 2026-11-07), each in a fresh container ->
    entities.json  concrete code objects bound to the target obligations (entry points,
                   sibling implementations, lifecycle hooks, existing tests, nearest analog);
    items.jsonl    check points from four targeted extensions (entity binding, boundary
                   instantiation, analog API-form parity, external-spec naming), each a
                   question (two or more readings, observations as facts, no favored reading);
                   v6.8.1: no requirement items (v6.8.0 requirement items misapplied task quotes);
    not_applicable.json  target obligation x extension pairs that do not apply, with a reason.
lint.py (v68_lint) checks the files mechanically; an episode submits only after LINT OK.
v6.8.1 (user, 2026-11-07): questions only; at most 12 obligations (lint in the plan episode; the
runner keeps the first 12 if a plan episode ends without LINT OK).
The tool-free control gets the same item rules, the agent's obligations and batches, and the
frozen V6.6 evidence instead of a repository.
"""
from __future__ import annotations

import json
from typing import Any

from external_agents_eval.workflows.solve_review_repair import v68_lint

VERSION = "v6.8.1"
WORK_DIR = "/tmp/v68"
BATCH_MAX_TARGETS = 3  # user, 2026-11-07: three obligations per check-point batch
PLAN_STEP_LIMIT, PLAN_WALL_TIME_SEC = 20, 600
BATCH_STEP_LIMIT, BATCH_WALL_TIME_SEC = 60, 1500
CONTROL_MAX_ATTEMPTS = 2
EXTENSIONS = v68_lint.EXTENSIONS
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

GOAL = """\
GOAL
Implementers of this task tend to miss behaviors that hidden acceptance tests check: other code
paths that must behave the same, lifecycle/state interactions, edge values, the exact form of a
new API, and rules of external standards. The analysis surfaces such conditions as concrete,
checkable items BEFORE anyone implements the task. You do not implement anything.
"""

PLAN_RULES = """\
OBLIGATIONS
List the distinct behaviors the task requires: AT MOST %d obligations (lint rejects more).
Each obligation has an exact, verbatim quote of the task text that states it (task_quote; join
fragments of one passage with "..."). Do not add behaviors the task does not ask for; do not
paraphrase the quote. Merge closely related behaviors (same sentence, same mechanism, same API)
into one obligation; keep independent behaviors separate.
""" % v68_lint.MAX_OBLIGATIONS

ITEM_RULES = """\
STEP A - ENTITIES (bind your target obligations to concrete code objects)
For each target obligation find the concrete existing code objects it touches, with a real path
and line:
- entry_point: public functions, methods, CLI commands, config options or exported APIs through
  which callers (and tests) will reach the new/changed behavior;
- sibling: other implementations that must behave the same or be extended alongside: other
  adapters/dialects/backends/drivers/variants of the same interface, registry members, parallel
  paths (sync/async, stream/non-stream, single/batch, CLI/API, render/print), overloads;
- lifecycle: hooks and state transitions that interact with state the obligation adds or changes:
  init/first run, reset, re-configure, re-observe/re-register, dispose/close, abort/cancel,
  error/retry, cache invalidation, ordering of events;
- existing_test: existing tests for the touched modules (they show how tests call these APIs);
- analog: the nearest existing feature/API with a similar shape (for API-form parity);
- other: anything else concrete the obligation must interact with.
Each entity names a symbol that literally appears within 3 lines of the cited line.

STEP B - ITEMS (four targeted extensions; be specific, breadth over depth)
ext = "entity": for every (entity, obligation) pair, ask how the obligation's behavior applies to,
  passes through or interacts with that object, e.g. "when the run is aborted through the QUnit
  adapter, are the collected results still emitted?", "does reset() clear the new state?",
  "does the CLI command also accept the new flag?", "must the existing receiver stay unmodified?".
ext = "boundary": find the quantities and shapes in the obligation (counts, indices, offsets,
  widths, sizes, durations, sequence numbers, ranges, collections, optional fields, strings,
  nested or recursive structures) and instantiate their edges: zero, empty, one, many, first,
  last, inclusive/exclusive end and off-by-one, nested/recursive, omitted vs null vs empty string,
  degenerate input that needs no work; also repeated calls (second call accumulates, replaces or
  is ignored? idempotent?) and ordering (before/after initialization, change after the first run).
ext = "analog_form": for every new API or output, compare with the nearest analog entity: call
  form (method vs function vs callable object), argument shape and order, return type/shape
  (names vs objects, list vs iterator), time base and units, naming, error type, string or repr
  format, serialization. The analog's form is one reading, never the answer.
ext = "external_spec": name each protocol, data format, standard or language feature the task
  involves (RFC, HTTP, MIME, CSS, SQL, JSON Schema, a language grammar or scoping rule, ...) and the
  specific rule that matters (spec.name, spec.rule), e.g. header continuation lines, scope of a
  binding inside a comprehension, which syntax positions allow a declaration.

ITEM RULES
- Every item is a question (kind "question"): list at least two concrete readings in
  alternatives; do not mark any reading as favored, preferred, correct or excluded. Repository
  facts go to observations (text + path + line); they inform, they do not decide.
- Do not restate what the task text plainly says. Ask where an implementer could plausibly go
  wrong. If a task sentence bears on the question, put it in task_quote (exact text) and still
  list the readings; a quote never decides how it applies to a specific object or edge.
- given/when/then name the concrete condition (inputs, values, call sequence, object) and the
  concrete observable outcome. No generic "consider edge cases" items.
- Every target obligation x extension needs at least one item or a not_applicable entry with a
  reason. Every (entity, obligation) pair of entities.json needs at least one ext="entity" item.
- Prefer short specific items over long ones; merge duplicates. Typical: 6-15 items per target
  obligation.
"""

ITEM_FORMATS = """\
FILES (JSON; all under {work})
entities.json      [{{"id": "E1", "kind": "entry_point|sibling|lifecycle|existing_test|analog|other",
                      "symbol": "identifier on/near the line", "path": "repo/relative/path", "line": 123,
                      "obligations": ["O1", "O3"], "why": "one sentence"}}, ...]
items.jsonl        one JSON object per line:
                   {{"ext": "entity|boundary|analog_form|external_spec", "obligation": "O1",
                     "entity": "E2" or null, "kind": "question",
                     "given": "...", "when": "...", "then": "...",
                     "task_quote": "exact task text" or null, "alternatives": ["...", "..."],
                     "observations": [{{"text": "fact", "path": "...", "line": 45}}],
                     "spec": {{"name": "RFC 7230", "rule": "..."}} or null}}
not_applicable.json [{{"obligation": "O2", "ext": "external_spec", "reason": "...", "entity": null}}]
Paths are relative to the repository root /app.
""".format(work=WORK_DIR)

FINISH = """\
When `python {work}/lint.py` prints LINT OK, finish by issuing exactly:
`{submit}`
Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>
""".format(work=WORK_DIR, submit=SUBMIT)

TASK_BLOCK = """<task>
{{task}}
</task>

The task text is also in %s/task.md. The repository is at /app (baseline, before the task).
Any change under /app is discarded; write only under %s/.
""" % (WORK_DIR, WORK_DIR)

INTRO = "You are a requirement analyst for the following software task. You do NOT implement it.\n\n"

PLAN_HEADER = (INTRO + TASK_BLOCK + "\n" + GOAL + "\nThis first step only lists the obligations.\n\n"
               + PLAN_RULES + """
FILE
{work}/obligations.json  [{{"id": "O1", "statement": "...", "task_quote": "exact task text"}}, ...]

WORKFLOW AND BUDGET
You have at most {steps} commands and {minutes} minutes. Look at the repository only as much as
needed to understand the task's terms. Write {work}/obligations.json, run `python {work}/lint.py`
and fix every problem it reports.
""".format(work=WORK_DIR, steps=PLAN_STEP_LIMIT, minutes=PLAN_WALL_TIME_SEC // 60) + FINISH + "\n")

HEREDOC = "cat >> " + WORK_DIR + "/items.jsonl <<'" + "ITEMS'\n{\"ext\": \"boundary\", ...}\n" + "ITEMS\n"

BATCH_HEADER = (INTRO + TASK_BLOCK + "\n" + GOAL + """
The obligations of the task are frozen in {work}/obligations.json. Your target obligations are
listed in {work}/scope.json ("targets"); analyze only these. You may read files and run small
scripts or existing analog code to observe current behavior; do not build the whole project or
run the full test suite.

""".format(work=WORK_DIR) + ITEM_RULES + "\n" + ITEM_FORMATS + "\nAppend items with heredocs, e.g.\n"
                + HEREDOC + """
WORKFLOW AND BUDGET
You have at most {steps} commands and {minutes} minutes. Start with `cat {work}/obligations.json
{work}/scope.json`. Spend roughly half on exploration (grep -rn, find, sed -n, nl -ba), write
entities.json early, append items per obligation as you go, then run `python {work}/lint.py` and
fix every problem it reports.
""".format(work=WORK_DIR, steps=BATCH_STEP_LIMIT, minutes=BATCH_WALL_TIME_SEC // 60) + FINISH + "\n")

CONTROL_SYSTEM = ("You are a requirement analyst for a software task. You do NOT implement it.\n"
                  "You cannot run tools. The user message gives the task text, frozen excerpts of the baseline\n"
                  "repository (path and line numbers), the frozen obligations and your target obligations. Cite only\n"
                  "paths and lines that appear in those excerpts.\n\n" + GOAL + "\n" + ITEM_RULES + """
OUTPUT
Return exactly one JSON object and nothing else:
{"entities": [...], "items": [...], "not_applicable": [...]}
with the element shapes below (items is a list of item objects instead of JSON lines).
""" + ITEM_FORMATS)


def plan_header() -> str:
    return PLAN_HEADER


def batch_header() -> str:
    return BATCH_HEADER


def capped(obligations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """v6.8.1 hard cap: the first MAX_OBLIGATIONS obligations (a plan without LINT OK may list more)."""
    return obligations[:v68_lint.MAX_OBLIGATIONS]


def target_batches(obligations: list[dict[str, Any]], size: int = BATCH_MAX_TARGETS) -> list[list[str]]:
    ids = [item["id"] for item in obligations]
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def control_messages(rendered_user: str, obligations: list[dict[str, Any]], targets: list[str]) -> list[dict[str, str]]:
    return [{"role": "system", "content": CONTROL_SYSTEM},
            {"role": "user", "content": rendered_user.rstrip("\n")
             + "\n\nFROZEN OBLIGATIONS\n" + json.dumps(obligations, ensure_ascii=False, indent=2)
             + "\n\nYOUR TARGET OBLIGATIONS: " + ", ".join(targets)
             + "\nAnalyze only these targets as instructed and return the JSON object."}]


def control_data(payload: Any) -> tuple[dict[str, Any], list[str]]:
    """Map a control JSON object onto the lint data structure (obligations are added by the caller)."""
    if not isinstance(payload, dict):
        return {"entities": [], "items": [], "not_applicable": []}, ["output is not a JSON object"]
    errors, data = [], {}
    for name in ("entities", "items", "not_applicable"):
        value = payload.get(name, [])
        if not isinstance(value, list):
            errors.append(f"{name} must be a list")
            value = []
        data[name] = value
    bad = [i for i, item in enumerate(data["items"], 1) if not isinstance(item, dict)]
    if bad:
        errors.append(f"items {bad[:5]} are not objects")
    data["items"] = [dict(item, _line=index) for index, item in enumerate(data["items"], 1) if isinstance(item, dict)]
    return data, errors


def merge_batches(obligations: list[dict[str, Any]], batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate batch outputs; entity ids are renumbered globally (E1..), item refs follow."""
    merged = {"obligations": obligations, "entities": [], "items": [], "not_applicable": []}
    for batch in batches:
        mapping = {}
        for entity in batch.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            new_id = f"E{len(merged['entities']) + 1}"
            mapping[entity.get("id")] = new_id
            merged["entities"].append({**entity, "id": new_id, "batch": batch.get("batch")})
        for item in batch.get("items") or []:
            if isinstance(item, dict):
                entity = item.get("entity")
                merged["items"].append({**{k: v for k, v in item.items() if k != "_line"},
                                        "entity": mapping.get(entity, entity) if entity else None,
                                        "batch": batch.get("batch")})
        for entry in batch.get("not_applicable") or []:
            if isinstance(entry, dict):
                merged["not_applicable"].append({**entry, "batch": batch.get("batch"),
                                                 "entity": mapping.get(entry.get("entity"), entry.get("entity"))})
    return merged


# --- rendering for blind grading ------------------------------------------------------------------

def _sq(text: Any) -> str:
    return " ".join(str(text or "").split())


def ledger_markdown(data: dict[str, Any]) -> str:
    out = ["# Requirement ledger", "", "## Obligations"]
    for item in data.get("obligations") or []:
        out.append(f"- {item.get('id')}: {_sq(item.get('statement'))}")
        out.append(f"  task quote: {_sq(item.get('task_quote'))[:300]}")
    out += ["", "## Code objects bound to obligations"]
    for item in data.get("entities") or []:
        out.append(f"- {item.get('id')} [{item.get('kind')}] {item.get('symbol')} at {item.get('path')}:{item.get('line')}"
                   f" (obligations {','.join(item.get('obligations') or [])}): {_sq(item.get('why'))}")
    return "\n".join(out) + "\n"


def checkpoints_markdown(data: dict[str, Any]) -> str:
    items = data.get("items") or []
    out = [f"# Check points ({len(items)})", ""]
    for index, item in enumerate(items, 1):
        entity = f" entity={item['entity']}" if item.get("entity") else ""
        out.append(f"## CP{index} kind={item.get('kind')} ext={item.get('ext')} target={item.get('obligation')}{entity}")
        out.append(f"- given: {_sq(item.get('given'))}\n- when: {_sq(item.get('when'))}\n- then: {_sq(item.get('then'))}")
        if item.get("alternatives"):
            out.append("- readings: " + " | ".join(_sq(a) for a in item["alternatives"]))
        for observation in item.get("observations") or []:
            if isinstance(observation, dict):
                out.append(f"- observation ({observation.get('path')}:{observation.get('line')}): "
                           f"{_sq(observation.get('text'))}")
        spec = item.get("spec") or {}
        if isinstance(spec, dict) and spec.get("name"):
            out.append(f"- spec: {_sq(spec.get('name'))}: {_sq(spec.get('rule'))}")
        if item.get("task_quote"):
            out.append(f"- task quote: {_sq(item['task_quote'])[:300]}")
        out.append("")
    return "\n".join(out)


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    items = data.get("items") or []
    return {"obligations": len(data.get("obligations") or []), "entities": len(data.get("entities") or []),
            "items": len(items), "by_ext": {ext: sum(i.get("ext") == ext for i in items) for ext in EXTENSIONS},
            "questions": sum(i.get("kind") == "question" for i in items),
            "requirements": sum(i.get("kind") == "requirement" for i in items)}
