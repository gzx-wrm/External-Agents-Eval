"""V6.6 prompts: the A evidence plan and the tool-free check-point stage C.

The facets come from general test-design practice (category partition, boundary
values, design by contract, API compatibility).  The only worked example is a
synthetic task; nothing here refers to a benchmark task.
"""
from __future__ import annotations

import json
from typing import Any

from .ft_protocol import render_user

FACETS = {
    "D1": ("input partitions",
           "normal, empty, absent, boundary, invalid and mixed values of each explicit input, "
           "and of each implicit input the behavior reads: files, configuration, defaults and "
           "environment, and the context that consumes its output (for example the namespace an "
           "expression is evaluated in, or the parser that reads it)"),
    "D2": ("output representation",
           "exact form and type, the value used for none or unset, normalization of "
           "whitespace, case and order, serialization and qualified names"),
    "D3": ("optionality and defaults",
           "optional versus required fields and their defaults; omitted, explicit null and empty "
           "are three separate cases"),
    "D4": ("acceptance boundary",
           "valid inputs that must not be rejected or warned about, and errors that must "
           "or must not be raised"),
    "D5": ("delivery",
           "where results are returned, stored or sent, and who observes them"),
    "D6": ("state and lifecycle",
           "initialization, first use, repeated use, reset and cleanup, ordering in time"),
    "D7": ("identity and scope",
           "identifiers and handles, reuse across instances or contexts"),
    "D8": ("consistency",
           "agreement with the closest existing analog and with the contracts of types "
           "the data is passed to"),
}
RESOLUTIONS = ("task", "analog", "contract", "convention", "open")

# Appended to the unchanged historical A prompt.  Only read returns become evidence.
EVIDENCE_PLAN = """EVIDENCE PLAN (V6.6)
Only the file lines you obtain with read are passed to later stages; search
results and your final answer are not. Before finishing, use grep, find and ls to
locate and then read:
1. Entry points: the code where each requested behavior will be exposed or
   invoked, and the code it must call into.
2. Analogs: for each requested behavior, the closest existing feature in this
   repository with the same kind of behavior (a similar option, type, validator,
   state tracker, serializer, error path or protocol handler). Read its
   implementation and at least one of its existing tests.
3. Downstream contracts: when requested data is handed to another type, schema or
   format defined in base_root, its definition and validation rules.
4. Conventions: in those files, the lines showing how absent, empty or unset
   values are represented; optional versus required fields and defaults;
   normalization of whitespace, case and order; where results are stored or
   delivered; identifiers in output; first use, repeated use and reset; and which
   inputs are accepted versus rejected.
Read only the lines you need with offset/limit. Do not modify any file."""

# One continuation turn in the same A session when the plan was not followed.
EVIDENCE_PLAN_MISSING = {
    "search": "used grep or find to locate the closest existing analogs",
    "test_read": "read any existing test file",
}


def evidence_plan_continuation(missing: list[str]) -> str:
    gaps = "; ".join(EVIDENCE_PLAN_MISSING[key] for key in missing)
    return f"""EVIDENCE PLAN CHECK (V6.6)
Your exploration has not yet {gaps}. Continue the same exploration: use grep and
find to locate the closest existing analog of each requested behavior, then read
its implementation and at least one of its existing tests with read, together with
the definitions of any type or format the requested data is handed to. Do not
modify any file. Then finish with the same final answer format as before.
"""


def evidence_plan_prompt(base_prompt: str) -> str:
    return base_prompt.rstrip("\n") + "\n\n" + EVIDENCE_PLAN + "\n"


def _facet_lines() -> str:
    return "\n".join(f"{key} {name}: {question}." for key, (name, question) in FACETS.items())


CHECKPOINT_EXAMPLE = """EXAMPLE (synthetic; not from this repository)
Task: "Add parse_env(text) returning a dict from KEY=VALUE lines; lines starting
with # are comments." Ledger: O001 "parse_env returns a dict of KEY=VALUE lines,
ignoring comment lines." Evidence: config.py lines 40-52 show parse_ini, whose line
44 is `key, value = (part.strip() for part in line.split("=", 1))`.
- targets [O001], D2, analog: given "A = 1 ", when parsed, then the result is
  {"A": "1"}; rules_out "keeping the surrounding spaces as part of key and value";
  sources [config.py 44-44]; basis_quote `part.strip() for part in line.split("=", 1)`;
  basis_behavior "the existing parser strips spaces around keys and values".
- targets [O001], D4, open: given a line without "=", when parsed, then the behavior
  is open; alternatives ["the line is skipped", "ValueError is raised"]; favored null.
- O001/D6 not_applicable: "parse_env is a pure function with no state".
- O001/D3 not_applicable: "the task allows any key; no optional field exists"."""


def checkpoint_system_prompt(max_per_target: int) -> str:
    return f"""ROLE: Coverage Planner (V6.6 C; fixed-evidence inference)
Use only the original task, the frozen requirement ledger and the public
repository evidence supplied in the input. Never access hidden tests, verifier
output, reference solutions or historical evaluations. You have no tools; do not
claim to have read omitted files or executed commands.

Goal: expand the ledger into the smallest set of observable check points a
correct implementation must satisfy, so that implementers and reviewers do not
silently pick one reading of a detail the task leaves unstated.

For every O and P target in the current batch, answer each facet's question for that target's own
inputs, outputs and state:
{_facet_lines()}

Create a check point only when both gates hold:
- observable: a caller or test can observe the difference through the public
  surface named by the task or the target;
- divergent: at least two reasonable implementations of the target would behave
  differently there.
A check point that only restates its target is not allowed. Every check point
that is not open must name in rules_out a reasonable implementation it excludes:
one that a careful engineer could build from the target text alone. "Not
implementing the target" does not count. If you cannot name one, there is no
divergence: mark the facet not_applicable. Where the task explicitly allows any
choice, the facet is not_applicable, not open.

When the same rule applies to several targets, write one check point listing all
of them in targets. coverage must list every target in the current batch with all eight facets exactly
once; a facet is covered exactly when some check point lists that target and facet.

Resolve each check point with the strongest basis available, in this order:
task (an exact quote from the task) > analog (repository lines showing the same
kind of behavior) > contract (repository lines defining a type or format the data
reaches) > convention (a widely used language, protocol or format convention that
you name explicitly). A task resolution is required; analog, contract and
convention resolutions are inferred. If nothing decides it, make it open: list
every reasonable reading in alternatives. Never present a guess as resolved, never
add behavior the task does not request, and never let an inferred point reject
input the task allows.

Basis: an analog or contract check point, and an open check point whose favored
reading is set, must copy into basis_quote the exact text from the cited lines that
shows the behavior, and state in basis_behavior what that text does. A convention
check point names the convention and states in basis_behavior what it requires.
Set favored only from such a basis; otherwise leave it null.

Keep omitted, explicit null and empty as separate cases; never merge them into one
reading such as "unset or empty". At most {max_per_target} check points per target; keep
those whose divergence would most change observable results. Write given/when/then
about observable behavior, not implementation structure. Task quotes must be exact
contiguous text; repository sources must lie inside the supplied evidence lines.
Return exactly one JSON object conforming to the schema below, without markdown
fences or commentary.

{CHECKPOINT_EXAMPLE}

OUTPUT JSON SCHEMA
{json.dumps(CHECKPOINT_SCHEMA, ensure_ascii=False, indent=2)}"""


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
CHECKPOINT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object", "additionalProperties": False, "required": ["checkpoints", "coverage"],
    "properties": {
        "checkpoints": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["targets", "facet", "given", "when", "then", "resolution", "strength",
                         "rules_out", "sources", "basis_quote", "basis_behavior", "convention",
                         "alternatives", "favored"],
            "properties": {
                "targets": {"type": "array", "minItems": 1, "uniqueItems": True, "items": _TARGET},
                "facet": {"enum": list(FACETS)},
                "given": _TEXT, "when": _TEXT, "then": _TEXT,
                "resolution": {"enum": list(RESOLUTIONS)},
                "strength": {"enum": ["required", "inferred", "open"]},
                "rules_out": _NULLABLE_TEXT,
                "sources": {"type": "array", "items": _SOURCE},
                "basis_quote": _NULLABLE_TEXT,
                "basis_behavior": _NULLABLE_TEXT,
                "convention": _NULLABLE_TEXT,
                "alternatives": {"type": "array", "items": _TEXT},
                "favored": _NULLABLE_TEXT,
            },
        }},
        "coverage": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["target", "facet", "status", "reason"],
            "properties": {
                "target": _TARGET,
                "facet": {"enum": list(FACETS)},
                "status": {"enum": ["covered", "not_applicable"]},
                "reason": _NULLABLE_TEXT,
            },
        }},
    },
}


def ledger_view(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """O and P targets with IDs, statements and sources; open questions are context."""
    view = []
    for field in ("obligations", "preservation_obligations", "open_questions"):
        for item in ledger[field]:
            content = item["content"]
            view.append({"id": item["id"], "kind": field,
                         "text": content.get("statement") or content.get("question"),
                         "importance": content.get("importance"),
                         "sources": content.get("sources", [])})
    return view


def batch_instruction(targets: list[str], index: int, total: int) -> str:
    return (f"CURRENT BATCH {index} of {total}\n"
            f"Targets in this batch: {', '.join(targets)}\n"
            "Write check points and coverage only for these targets. The other ledger targets are "
            "context; they are handled in other batches. A check point lists only targets of this batch.")


def checkpoint_user(packet: dict[str, Any], ledger: dict[str, Any],
                    batch: tuple[list[str], int, int] | None = None) -> str:
    """The FT user rendering (task + numbered evidence) with the frozen ledger inserted.

    The batch instruction comes last, so every batch shares the long cached prefix.
    """
    rendered = render_user(packet, packet["evidence"])
    task, _, evidence = rendered.partition("\n\nFROZEN PUBLIC REPOSITORY EVIDENCE")
    text = (task + "\n\nFROZEN REQUIREMENT LEDGER\n"
            + json.dumps(ledger_view(ledger), ensure_ascii=False, indent=2)
            + "\n\nFROZEN PUBLIC REPOSITORY EVIDENCE" + evidence)
    return text if batch is None else text.rstrip("\n") + "\n\n" + batch_instruction(*batch)
