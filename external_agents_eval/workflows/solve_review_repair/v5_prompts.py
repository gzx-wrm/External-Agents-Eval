"""Compact, versioned V5 prompt builders."""

from __future__ import annotations

import json
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


COMMON_REVIEW_BOUNDARY = """Use only the original task and public repository
artifacts available to this role. Never access hidden tests, verifier output,
reference solutions, historical evaluations, the Solver trajectory, or another
live reviewer session. Do not use the network or install dependencies.

Do not modify, delete, rename, or commit base-tracked or candidate-tracked files.
Use the assigned scratch root for probes, logs, dependency caches, and other
disposable files. Scratch is outside the review artifact namespace and may be
cleaned after the batch; if a dependency tree must be reused, use a real copy,
reflink/COW copy, or an isolated environment. Do not create hard links from a
protected repository or worktree into scratch because writes would mutate the
protected snapshot. Keep every declared note under the output root as a normal
file, and keep declared probe/log materials bounded and reproducible. Task text
and applicable public contracts define required behavior; candidate code,
candidate tests, and reviewer probes may reveal behavior but cannot create
requirements. Task quotes must be exact contiguous text and repository
requirement sources must refer to the frozen base. Record blocked checks and
uncertainty instead of inventing results. Exclude only entries explicitly listed
in managed_delivery."""


REVIEWER_FEW_SHOTS = """SYNTHETIC EXAMPLES — WRITING GUIDANCE ONLY

The following examples are abstract writing examples. Their target IDs, paths,
facts and wording are not evidence, requirements or task-specific hints. Do not
copy them into the submitted batch.

Example 1 — incomplete coverage is not satisfied:

Target contract:
A public operation must return the documented value for a valid input, preserve
the documented boundary behavior, and reject an invalid input with the documented
error.

Observed:
The valid-input case and the boundary case were checked and matched. The invalid
case was not executed.

Correct assessment:
{
  "target_id": "I/EXAMPLE-COVERAGE",
  "status": "unresolved",
  "expected": "The public operation satisfies the valid-input, boundary-input and invalid-input clauses of the contract.",
  "observed": "The valid-input and boundary-input calls matched the expected results; no invalid-input call was executed.",
  "reason": "The evidence covers only part of the target. The missing invalid-input check prevents satisfied, but does not establish a violation.",
  "code_refs": [],
  "review_note_path": "<OUTPUT_ROOT>/example-coverage.md"
}

Example 2 — raw shape and lifecycle evidence must be preserved:

{
  "new_targets": [],
  "assessments": [
    {
      "target_id": "I/EXAMPLE-RETURN-SHAPE",
      "status": "violated",
      "expected": "The documented public entry point returns a Result object directly.",
      "observed": "The raw public call returned Wrapper[Result]. The inner Result appeared correct only after an additional extraction step.",
      "reason": "The contract concerns the direct return shape of the public call. Extracting or adapting the inner value cannot prove that contract.",
      "code_refs": [],
      "review_note_path": "<OUTPUT_ROOT>/example-return-shape.md"
    },
    {
      "target_id": "R/EXAMPLE-LIFECYCLE",
      "status": "unresolved",
      "expected": "The transition leaves the old resource inactive and invokes its cleanup callback exactly once.",
      "observed": "The old resource was observed as inactive, but callback invocation and call count were not recorded.",
      "reason": "The final state alone does not establish the complete lifecycle contract. Missing side-effect evidence prevents satisfied.",
      "code_refs": [],
      "review_note_path": "<OUTPUT_ROOT>/example-lifecycle.md"
    }
  ],
  "revisit_target_ids": [],
  "done": false
}

Before submitting the batch, perform an internal consistency check for every
assessment: the status must follow from the complete expected-versus-observed
comparison; observed must describe the raw documented call; and every required
clause, side effect and transition must either be evidenced or explicitly marked
unresolved."""


_SCHEMA_DEFS: dict[str, Any] = {
    "Text": {"type": "string", "minLength": 1},
    "Strings": {
        "type": "array",
        "items": {"$ref": "#/$defs/Text"},
        "uniqueItems": True,
    },
    "TaskSource": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "field", "quote"],
        "properties": {
            "kind": {"const": "task"},
            "field": {"enum": ["title", "description"]},
            "quote": {"$ref": "#/$defs/Text"},
        },
    },
    "RepositorySource": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "path", "line_start", "line_end"],
        "properties": {
            "kind": {"const": "repository"},
            "path": {"$ref": "#/$defs/Text"},
            "line_start": {"type": "integer", "minimum": 1},
            "line_end": {"type": "integer", "minimum": 1},
        },
    },
    "Source": {
        "oneOf": [
            {"$ref": "#/$defs/TaskSource"},
            {"$ref": "#/$defs/RepositorySource"},
        ]
    },
    "Sources": {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": {"$ref": "#/$defs/Source"},
    },
    "TargetDraft": {
        "type": "object",
        "additionalProperties": False,
        "required": ["statement", "importance", "sources"],
        "properties": {
            "statement": {"$ref": "#/$defs/Text"},
            "importance": {"enum": ["required", "advisory"]},
            "sources": {"$ref": "#/$defs/Sources"},
        },
    },
    "QuestionDraft": {
        "type": "object",
        "additionalProperties": False,
        "required": ["question", "blocking", "sources"],
        "properties": {
            "question": {"$ref": "#/$defs/Text"},
            "blocking": {"type": "boolean"},
            "sources": {"$ref": "#/$defs/Sources"},
        },
    },
    "CodeReference": {
        "type": "object",
        "additionalProperties": False,
        "required": ["snapshot", "path", "line_start", "line_end"],
        "properties": {
            "snapshot": {"enum": ["base", "candidate"]},
            "path": {"$ref": "#/$defs/Text"},
            "line_start": {"type": "integer", "minimum": 1},
            "line_end": {"type": "integer", "minimum": 1},
        },
    },
    "Assessment": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "target_id",
            "status",
            "expected",
            "observed",
            "reason",
            "code_refs",
            "review_note_path",
        ],
        "properties": {
            "target_id": {"$ref": "#/$defs/Text"},
            "status": {
                "enum": ["satisfied", "violated", "unresolved", "disputed"]
            },
            "expected": {"$ref": "#/$defs/Text"},
            "observed": {"$ref": "#/$defs/Text"},
            "reason": {"$ref": "#/$defs/Text"},
            "code_refs": {
                "type": "array",
                "uniqueItems": True,
                "items": {"$ref": "#/$defs/CodeReference"},
            },
            "review_note_path": {"$ref": "#/$defs/Text"},
        },
    },
    "SpecOutput": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary",
            "obligations",
            "preservation_obligations",
            "open_questions",
        ],
        "properties": {
            "summary": {"$ref": "#/$defs/Text"},
            "obligations": {
                "type": "array",
                "items": {"$ref": "#/$defs/TargetDraft"},
            },
            "preservation_obligations": {
                "type": "array",
                "items": {"$ref": "#/$defs/TargetDraft"},
            },
            "open_questions": {
                "type": "array",
                "items": {"$ref": "#/$defs/QuestionDraft"},
            },
        },
    },
    "ReviewBatch": {
        "type": "object",
        "additionalProperties": False,
        "required": ["new_targets", "assessments", "revisit_target_ids", "done"],
        "properties": {
            "new_targets": {
                "type": "array",
                "items": {"$ref": "#/$defs/TargetDraft"},
            },
            "assessments": {
                "type": "array",
                "items": {"$ref": "#/$defs/Assessment"},
            },
            "revisit_target_ids": {"$ref": "#/$defs/Strings"},
            "done": {"type": "boolean"},
        },
    },
    "FindingDecision": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "finding_ref",
            "disposition",
            "reason",
            "supporting_refs",
            "sources",
        ],
        "properties": {
            "finding_ref": {"$ref": "#/$defs/Text"},
            "disposition": {"enum": ["repair", "dismiss", "unresolved"]},
            "reason": {"$ref": "#/$defs/Text"},
            "supporting_refs": {"$ref": "#/$defs/Strings"},
            "sources": {"$ref": "#/$defs/Sources"},
        },
    },
    "RepairRequest": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "finding_refs",
            "change",
            "validation_record_refs",
            "preserve_refs",
        ],
        "properties": {
            "finding_refs": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"$ref": "#/$defs/Text"},
            },
            "change": {"$ref": "#/$defs/Text"},
            "validation_record_refs": {"$ref": "#/$defs/Strings"},
            "preserve_refs": {"$ref": "#/$defs/Strings"},
        },
    },
    "SummaryOutput": {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "decisions", "repair_requests", "new_questions"],
        "properties": {
            "summary": {"$ref": "#/$defs/Text"},
            "decisions": {
                "type": "array",
                "items": {"$ref": "#/$defs/FindingDecision"},
            },
            "repair_requests": {
                "type": "array",
                "items": {"$ref": "#/$defs/RepairRequest"},
            },
            "new_questions": {
                "type": "array",
                "items": {"$ref": "#/$defs/QuestionDraft"},
            },
        },
    },
}


def _schema_references(value: Any) -> set[str]:
    references: set[str] = set()
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            references.add(reference.removeprefix("#/$defs/"))
        for child in value.values():
            references.update(_schema_references(child))
    elif isinstance(value, list):
        for child in value:
            references.update(_schema_references(child))
    return references


def _output_schema(name: str) -> dict[str, Any]:
    required = {name}
    pending = [name]
    while pending:
        definition = _SCHEMA_DEFS[pending.pop()]
        references = _schema_references(definition)
        for reference in references - required:
            if reference not in _SCHEMA_DEFS:
                raise ValueError(f"unknown V5 schema definition: {reference}")
            required.add(reference)
            pending.append(reference)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": f"#/$defs/{name}",
        "$defs": {
            key: value for key, value in _SCHEMA_DEFS.items() if key in required
        },
    }


REVIEW_BATCH_SCHEMA = _output_schema("ReviewBatch")
SPEC_OUTPUT_SCHEMA = _output_schema("SpecOutput")
SUMMARY_OUTPUT_SCHEMA = _output_schema("SummaryOutput")


def correction_prompt(
    *, original_output: str, output_schema: dict[str, Any], errors: list[dict[str, str]]
) -> str:
    correction_input = {
        "original_output": original_output,
        "output_schema": output_schema,
        "errors": errors,
    }
    return f"""ROLE: V5 JSON format corrector
Repair presentation and unambiguous JSON syntax or shape defects only. Preserve
every requirement, source, assessment, status, observation, reference, done
value, decision and repair recommendation. Do not invent missing evidence or
IDs, redirect a reference, change a review_note_path, create a review note, or
resolve a semantic disagreement. Return only one corrected JSON object.

CORRECTION INPUT
{_json(correction_input)}
"""


def spec_prompt(
    task: dict[str, Any],
    repository: dict[str, Any],
    *,
    managed_delivery: list[dict[str, Any]] | None = None,
) -> str:
    spec_input = {
        "task": task,
        "base_root": repository["base_root"],
        "patch_base": repository.get("patch_base", "base-commit"),
        "baseline_patch_sha256": repository.get("baseline_patch_sha256"),
        "managed_delivery": managed_delivery or [],
    }
    return f"""ROLE: Specification Reviewer (V5 A)
{COMMON_REVIEW_BOUNDARY}

Work before the candidate is revealed. Read the complete task and relevant frozen
base context. Extract independently assessable requested obligations and adjacent
source-grounded preservation obligations. Preserve explicit scope, error behavior,
state transitions and lifecycle requirements without choosing an implementation.
Preservation targets must describe applicable public contracts related to the
change; do not preserve task-authorized old behavior or implementation accidents.

Use required when violation makes the requested work or applicable compatibility
contract incorrect. Use advisory only for a source-grounded non-blocking
expectation. Put genuine ambiguity in open_questions and mark it blocking only
when it prevents an important judgment. Avoid duplicates across O/P and allow an
empty preservation list. Output no IDs; the Controller assigns them. Inspect only
the base_root below. The candidate diff path is deliberately absent.

SPEC INPUT
{_json(spec_input)}

OUTPUT JSON SCHEMA
{_json(SPEC_OUTPUT_SCHEMA)}
"""


def reviewer_prompt(
    *,
    task: dict[str, Any],
    repository: dict[str, Any],
    ledger: dict[str, Any],
    state: dict[str, Any],
    role: str,
    skill_path: str,
    output_root: str,
    initial: bool = True,
    managed_delivery: list[dict[str, Any]] | None = None,
    limits: dict[str, Any] | None = None,
) -> str:
    role_guidance = (
        """Read the complete task, diff, ledger and relevant public entry points.
Treat the ledger as a fallible checklist. Determine whether every O target holds
through observable calls and workflows, including relevant defaults, errors,
interactions and lifecycle sequences. Derive inputs and expected results from the
task or public contracts independently of candidate tests. Try to falsify apparent
completeness; distinguish missing evidence from success. Register an omitted
requested behavior, and mark an unsupported or inapplicable target disputed."""
        if role == "I"
        else """Work independently of I. Read the complete task and diff, cover every
P target, then inspect affected callers, shared interfaces, configuration,
initialization and public tests. P targets are anchors, not the limit of the impact
review: reuse an applicable P, register an omitted contract as R, and assess an O
when it records correct new behavior a repair must preserve. Compare equivalent
base/candidate inputs where practical. A failure also present at base is not by
itself a regression; new or task-authorized behavior need not match base. Candidate
tests and passing counts do not define or prove a contract. Do not invent findings."""
    )
    initial_input = {
        "role": "implementation" if role == "I" else "regression",
        "task": task,
        "repository": {
            key: repository[key]
            for key in (
                "base_root",
                "candidate_root",
                "diff_path",
                "scratch_root",
                "patch_base",
                "baseline_patch_sha256",
            )
            if key in repository
        },
        "ledger": ledger,
        "managed_delivery": managed_delivery or [],
        "limits": limits or {},
        "review_notes": state["review_notes"],
    }
    input_section = (
        "INITIAL REVIEW INPUT\n" + _json(initial_input)
        if initial
        else """Continue the same review against the same frozen snapshots. The
Controller state below is authoritative. Do not repeat accepted assessments.
If mode=retry, replace the rejected batch after resolving the listed errors; do
not weaken a target or erase an observed failure."""
    )
    return f"""ROLE: V5 {role} Reviewer
{COMMON_REVIEW_BOUNDARY}

{role_guidance}

Before marking a target satisfied, make its observable contract explicit: use the
normal public entry point and preserve the task or public contract's invocation,
return, error, and state-transition shape. Treat every explicitly required clause
as part of the target. Exercise a representative success case
and, when meaningful, at least one boundary case and one invalid or negative case;
for stateful or lifecycle behavior, include a relevant interaction or transition
sequence. Record the raw observable result, including its type or error when that
is part of the contract. Compare each observation directly with the expected
behavior. A passing probe establishes only the assertions it actually checks, and
must not be made to pass by silently changing the call or adapting the result.
If a case is not meaningful or cannot run, state that limitation. Any contrary or
unexplained observation prevents `satisfied`; use `violated` or `unresolved` as
appropriate.

{REVIEWER_FEW_SHOTS}

Use satisfied only when evidence supports the complete target in the stated
scope; violated for an established mismatch; unresolved when evidence cannot
settle the behavior; disputed when the target itself lacks support or applicability.
No tests collected, a blocked build, or an unexecuted command cannot demonstrate
success. Explain base/candidate differences and contrary observations.
The Controller may record and clean incidental changes made in the disposable
review worktree after this batch. Such cleanup does not change the assessment;
report what was actually observed and keep all intentional probes and logs under
the assigned scratch root.

At the start of this session read {skill_path}, follow its template link, and use
its synthetic examples only as writing guidance when relevant. Before each batch,
write and read back one Markdown note for every submitted assessment below
{output_root}. The note records observable review actions and limits; it is not an
internal reasoning transcript or proof that a command ran. New targets have no ID
until the next round. Request reopen before replacement. Set done only after the
resulting pending set is empty, this batch adds no targets or revision requests,
and the role's scope review is complete. I must cover every O target. R must cover
every P/R target and finish the diff-impact review; an O target is optional for R.
For every batch, len(new_targets) + len(assessments) +
len(revisit_target_ids) must be <= batch_size; the limit counts all three arrays
together. Return no overall verdict or repair plan.

{input_section}

CONTROLLER STATE
{_json(state)}

REVIEW NOTE SKILL: {skill_path}
OUTPUT ROOT: {output_root}
OUTPUT SCHEMA
{_json(REVIEW_BATCH_SCHEMA)}
"""


def summary_prompt(
    summary_input: dict[str, Any],
    *,
    freeze_id: str | None = None,
    source_bindings: list[dict[str, Any]] | None = None,
) -> str:
    return f"""ROLE: V5 Summary Reviewer (S)
{COMMON_REVIEW_BOUNDARY}

Assess the supplied reviews and plan confirmed repairs; the Controller derives
accept, repair or hold.

Adjudicate I/R disagreements independently from the task, applicable contracts,
and supplied evidence, without favoring either reviewer. Explain the decision and
retain unresolved when evidence is insufficient. The controller has supplied an
immutable artifact index; use its note, trace-manifest and materials-manifest
paths when checking a claim, and treat a missing or incomplete artifact as an
explicit limitation. Every path in review_notes is relative to artifact_root.

Read the task, ledger, both reports, source bindings and review-note catalog. Read
every non-satisfied assessment's note and any satisfied notes relevant to a
decision. Resolve effective versions under artifact_root and corroborate decisive
empirical claims or conflicts through the supplied trace/material manifests.
Notes are reviewer accounts, not proof of execution. Review reports may also
contain review_integrity diagnostics about incidental disposable-worktree edits;
treat those diagnostics as a limitation to weigh, not as a semantic finding by
themselves. Do not run new tests or create observations. Give exactly one decision for every
required_decision_ref. You may additionally challenge a contradicted satisfied
assessment only as unresolved. Use repair only for a confirmed required violation;
do not upgrade a question, disputed or unresolved assessment. Dismiss only when
evidence establishes invalidity, duplication, resolution or irrelevance; otherwise
use unresolved. For duplicates, cite the retained assessment and keep one repair
for a confirmed violation. Record important omissions or blocking conflicts in
new_questions. Incomplete reports cannot support acceptance.

The allowed_* catalogs in SUMMARY INPUT are authoritative. Copy each reference or
source exactly from its matching catalog: IDs are namespaced (A/Q001, I/O001,
R/P001), never bare Q001/O001/P001. Artifact paths locate evidence and may be named
in reason, but never belong in supporting_refs, validation_record_refs or
preserve_refs. Candidate code_refs are evidence locations, not sources unless the
exact source object appears in allowed_sources. supporting_refs may cite A questions
or effective assessments, never another S decision.

Group repair decisions by root cause. Their finding_refs must cover every repair
decision exactly once. validation_record_refs must come from effective assessments;
preserve_refs must come from satisfied assessments and omit any you challenge.
Static repair findings may have no validation reference if the missing runtime
coverage is stated; validation references are leads, not passing checks. Keep
unresolved scope explicit and do not claim P1 is verified.

SUMMARY INPUT
{_json(summary_input)}

CONTROLLER FREEZE ID
{freeze_id or "unavailable"}

BOUND SOURCES
{_json(source_bindings or [])}

OUTPUT JSON SCHEMA
{_json(SUMMARY_OUTPUT_SCHEMA)}
"""


def repair_prompt(
    task: dict[str, Any],
    p0_sha256: str,
    summary_output: dict[str, Any],
    *,
    review_context: dict[str, Any] | None = None,
) -> str:
    return f"""ROLE: V5 Repair agent
Apply one evidence-grounded repair to the supplied P0. Read the task, requests and
preservation context. Verify important counterexamples where possible, preserve
unrelated correct behavior, and run relevant checks in the repair worktree. Repair
has no additional structured output contract in V5's first experiment; its process
result, tool trajectory and candidate patch are captured by the controller.

TASK
{_json(task)}

P0 SHA256: {p0_sha256}

SUMMARY PLAN
{_json(summary_output)}

FROZEN REVIEW CONTEXT
{_json(review_context or {})}
"""


__all__ = [
    "REVIEW_BATCH_SCHEMA",
    "SPEC_OUTPUT_SCHEMA",
    "SUMMARY_OUTPUT_SCHEMA",
    "correction_prompt",
    "repair_prompt",
    "reviewer_prompt",
    "spec_prompt",
    "summary_prompt",
]
