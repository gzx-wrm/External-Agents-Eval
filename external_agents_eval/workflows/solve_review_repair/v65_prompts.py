"""V6.5 prompt builders (review prompt version v5-7).

I/R and Repair keep their V5-6 text byte for byte; V6.5 only inserts a REVIEW VIEW
notice into the I/R prompts. New here: the tool-less assumption extractor (X), the
disambiguation reviewer (D) and the summary prompt that adjudicates with D.
"""
from __future__ import annotations

import json
from typing import Any

from .v5_config import (
    REVIEW_MATERIAL_MAX_FILE_BYTES,
    REVIEW_MATERIAL_MAX_FILES,
    REVIEW_MATERIAL_MAX_TOTAL_BYTES,
)
from .v5_prompts import (
    COMMON_REVIEW_BOUNDARY,
    OUTPUT_CONTRACT_MARKER,
    REVIEW_BATCH_SCHEMA,
    SUMMARY_OUTPUT_SCHEMA,
)

V57_PROMPT_VERSION = "v5-7"
REVIEW_RECORDS_MARKER = "\nREVIEW RECORDS AND BATCHES\n"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _mib(value: int) -> str:
    return f"{value / (1024 * 1024):g}"


# ---------------------------------------------------------------------------
# A: review view notice
# ---------------------------------------------------------------------------

_WITHHELD = {
    "strip-all-tests": (
        "Comments and docstrings that P0 added are withheld, and so are P0's additions "
        "and changes to test files: test files appear as in the base repository."
    ),
    "strip-new-tests": (
        "Comments and docstrings that P0 added are withheld, and so are test files that "
        "P0 added. P0's changes to existing test files are shown."
    ),
    "strip-comments": (
        "Comments and docstrings that P0 added are withheld. P0's test-file changes are "
        "shown."
    ),
}


def view_notice(level: str | None) -> str:
    """Return the REVIEW VIEW section for a view level; empty for the raw P0."""
    if level is None:
        return ""
    return f"""REVIEW VIEW
The candidate is shown as a behavioral view of P0. {_WITHHELD[level]} Withheld
text is not program behavior; the public behavior you review is unchanged, and the
diff at diff_path is this view. Judge behavior from the task, public contracts and
your own probes, not from explanations in the code. A target whose subject is the
content of a withheld comment or test file cannot be judged from this view: mark it
unresolved and say so."""


def insert_view_notice(prompt: str, notice: str) -> str:
    if not notice:
        return prompt
    if prompt.count(REVIEW_RECORDS_MARKER) != 1:
        raise ValueError("Expected exactly one REVIEW RECORDS AND BATCHES section")
    head, tail = prompt.split(REVIEW_RECORDS_MARKER, 1)
    return head + "\n" + notice + "\n" + REVIEW_RECORDS_MARKER + tail


# ---------------------------------------------------------------------------
# C: assumption extraction (X)
# ---------------------------------------------------------------------------

ASSUMPTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["assumptions"],
    "properties": {
        "assumptions": {"type": "array", "items": {"$ref": "#/$defs/Assumption"}},
    },
    "$defs": {
        "CandidateRef": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "line_start", "line_end"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "line_start": {"type": "integer", "minimum": 1},
                "line_end": {"type": "integer", "minimum": 1},
            },
        },
        "Assumption": {
            "type": "object",
            "additionalProperties": False,
            "required": ["claim", "tension", "related_target_ids", "candidate_refs"],
            "properties": {
                "claim": {"type": "string", "minLength": 20, "maxLength": 600},
                "tension": {"enum": ["conflict", "interpretation", "unspecified"]},
                "related_target_ids": {
                    "type": "array", "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "candidate_refs": {
                    "type": "array", "items": {"$ref": "#/$defs/CandidateRef"},
                },
            },
        },
    },
}


def assumption_prompt(
    *,
    task: dict[str, Any],
    ledger: dict[str, Any],
    candidate_patch: str,
    withheld: list[dict[str, Any]],
    max_items: int,
) -> str:
    return f"""ROLE: V6.5 Assumption extractor (X)
You read a candidate patch P0 for the task below, including the comments, docstrings
and tests it adds. Later reviewers see P0 without that explanatory text; your output
is the only thing they receive from it. Do not judge whether P0 is correct.

List the interpretive choices P0 makes: places where P0 picks a behavior that the
task and ledger do not state exactly. Include especially:
- a behavior that departs from, narrows or extends a literal ledger obligation or
  task sentence (tension "conflict");
- a choice between readings of an ambiguous point, including any ledger open question
  that P0 settles (tension "interpretation");
- a behavior the task does not mention but P0's comments or tests treat as intended
  (tension "unspecified").
Comments that explain a choice and tests that encode one are the main signals.

Write each claim as a neutral, checkable statement of P0's observable behavior: the
public entry point, a concrete input or condition, and the result P0 produces. Do not
include reasons, justifications or quotations from comments (no "because", "since",
"so that"); do not say whether the behavior is right. Name the ledger IDs (O, P or Q)
each claim concerns in related_target_ids, and cite the candidate lines in
candidate_refs (post-image paths and line numbers of P0). Return at most {max_items}
claims, the most consequential first; return an empty list if P0 makes no such choice.

Filesystem and shell tools are disabled. Submit exactly one JSON object matching the
OUTPUT SCHEMA with: submit_json '<complete JSON>'

TASK
{_json(task)}

LEDGER
{_json(ledger)}

P0 (complete unified diff)
{candidate_patch}

TEXT WITHHELD FROM LATER REVIEWERS (comments and docstrings P0 added)
{_json(withheld)}

OUTPUT SCHEMA
{_json(ASSUMPTION_OUTPUT_SCHEMA)}
"""


# ---------------------------------------------------------------------------
# D: disambiguation reviewer
# ---------------------------------------------------------------------------

_D_GUIDANCE = """Your targets are ledger open questions (Q...) and implementation assumptions
(X...): neutral claims about what the candidate does, extracted by the Controller
from text you cannot see. Resolve each target with evidence:
1. Readings. State the concrete readings the target admits, normally two or three,
   each as an observable behavior through a public entry point. For an assumption,
   one reading is the claimed candidate behavior and another is what a literal
   reading of the task or ledger requires.
2. Discriminating probe. Construct an input whose observable result differs between
   the readings, run it against candidate_root through the project's own harness when
   practical, and record which reading the candidate implements. If no input separates
   the readings through the public interface, the target is immaterial.
3. Evidence, in priority order: (a) the task text, explicit or directly implied;
   (b) the base repository's public contracts and conventions in base_root, such as
   analogous existing features, documentation, existing tests and naming; (c) general
   ecosystem knowledge. A lower class never overrides a higher one; in particular
   general knowledge, such as a language specification, does not override what the
   task states. The candidate's own implementation is not evidence of the intended
   reading.
4. Status. satisfied: the candidate implements the reading the evidence supports.
   violated: class (a) or (b) evidence supports a reading the candidate does not
   implement. unresolved: the evidence does not settle the readings. disputed: the
   readings are not observably different, or the target does not concern the task.
5. Derived obligation. For each violated target, also register in new_targets one
   required target that states the supported behavior as an observable obligation and
   cites only task quotes or base repository lines. In the next round assess that
   D-numbered target, using the discriminating probe as its evidence. Register at most
   one derived obligation per violated target.
Do not edit tracked files, do not assess O/P targets and do not plan repairs."""


def disambiguation_prompt(
    *,
    task: dict[str, Any],
    repository: dict[str, Any],
    ledger: dict[str, Any],
    state: dict[str, Any],
    role: str,
    skill_path: str,
    output_root: str,
    initial: bool,
    managed_delivery: list[dict[str, Any]] | None,
    limits: dict[str, Any] | None,
    view: str = "",
    **_: Any,
) -> str:
    if role != "D":
        raise ValueError("disambiguation_prompt is only for the D lane")
    limits = dict(limits or {})
    if "batch_size" not in limits or "time_budget_sec" not in limits:
        raise ValueError("V5-7 D prompts require batch_size and time_budget_sec")
    initial_input = {
        "role": "disambiguation",
        "task": task,
        "repository": {
            key: repository[key]
            for key in ("base_root", "candidate_root", "diff_path", "scratch_root",
                        "patch_base", "baseline_patch_sha256")
            if key in repository
        },
        "ledger": ledger,
        "managed_delivery": managed_delivery or [],
        "limits": limits,
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
    lane_minutes = max(1, int(limits["time_budget_sec"]) // 60)
    view_section = (view + "\n\n") if view else ""
    return f"""ROLE: V5 D Reviewer (disambiguation)
{COMMON_REVIEW_BOUNDARY}

{_D_GUIDANCE}

Record the raw observable result of every probe, including its type or error when
that matters. A probe establishes only what it actually checks; do not adapt a call
or its result to make a reading appear. If a probe cannot run, say so and judge the
target unresolved unless static class (a) or (b) evidence settles it.

{view_section}REVIEW RECORDS AND BATCHES
At the start of this session read {skill_path} and follow its template, including its
disambiguation reference; use its synthetic examples only as writing guidance. Write
and read back one review note, at the path the Controller assigns in
review_notes.target_paths, for every assessment you submit. In the note's review
steps record the readings, the discriminating probe and its result, and the evidence
with its class (a), (b) or (c).
- In each batch, assessments, new targets and reopen requests (revisit_target_ids)
  together must not exceed {limits["batch_size"]}.
- New targets receive IDs from the Controller in the next round; do not assess them
  before then. Request a reopen before replacing an accepted assessment.
- Your scope closes only when the pending set is empty and the batch adds no targets
  or reopen requests. D must cover every Q/X target and every derived D target.
- Return no overall verdict or repair plan.

LIMITS
- Under the output root, only notes/, probes/ and logs/ are archived. probes/ and
  logs/ together may hold at most {REVIEW_MATERIAL_MAX_FILES} files, at most
  {_mib(REVIEW_MATERIAL_MAX_FILE_BYTES)} MiB per file and
  {_mib(REVIEW_MATERIAL_MAX_TOTAL_BYTES)} MiB in total. Put builds and caches elsewhere,
  for example work/ under the output root or /tmp.
- Code reference line ranges must exist in the named snapshot.
- The whole lane has about {lane_minutes} minutes; the Controller state shows the
  time remaining.

{input_section}

CONTROLLER STATE
{_json(state)}

REVIEW NOTE SKILL: {skill_path}
OUTPUT ROOT: {output_root}
{OUTPUT_CONTRACT_MARKER.strip()}
Return one ReviewBatch JSON object that matches OUTPUT SCHEMA. Give each assessment
the review_note_path of its saved note, and set done=true only when your scope closes
as described under REVIEW RECORDS AND BATCHES.

OUTPUT SCHEMA
{_json(REVIEW_BATCH_SCHEMA)}
"""


# ---------------------------------------------------------------------------
# S with the D report
# ---------------------------------------------------------------------------

def summary_prompt_v65(
    summary_input: dict[str, Any],
    *,
    freeze_id: str | None = None,
    source_bindings: list[dict[str, Any]] | None = None,
) -> str:
    return f"""ROLE: V5 Summary Reviewer (S)
{COMMON_REVIEW_BOUNDARY}

Assess the supplied reviews and plan confirmed repairs; the Controller derives
accept, repair or hold.

Adjudicate disagreements among I, R and D independently from the task, applicable
contracts, and supplied evidence, without favoring any reviewer. Explain the decision
and retain unresolved when evidence is insufficient. The controller has supplied an
immutable artifact index; use its note, trace-manifest and materials-manifest paths
when checking a claim, and treat a missing or incomplete artifact as an explicit
limitation. Every path in review_notes is relative to artifact_root.

Reviewers, and your worktree, show a behavioral view of P0 described in review_view:
comments, docstrings and test changes P0 added may be withheld.

Read the task, ledger, every report, source bindings and review-note catalog. Read
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

The D report resolves the ledger questions (Q...) and implementation assumptions
(X...) listed in disambiguation_targets. For an A/Q question, use D's assessment of
the same Q ID as evidence: dismiss it when D establishes a resolution (satisfied: the
candidate follows the reading the evidence supports; disputed: the readings are not
observably different) and its note and materials corroborate that; otherwise keep it
unresolved. A violated D/Q or D/X assessment shows that task or base-repository
evidence supports a reading the candidate does not implement; the repairable finding
is its derived obligation, a D-numbered target with its own assessment. Repair a
derived obligation only when its cited task or base-repository sources support it
and its probe shows the candidate departs from it; general ecosystem knowledge does
not override the task.

The allowed_* catalogs in SUMMARY INPUT are authoritative. Copy each reference or
source exactly from its matching catalog: IDs are namespaced (A/Q001, I/O001,
R/P001, D/Q001, D/X001, D/D001), never bare Q001/O001/P001. Artifact paths locate
evidence and may be named in reason, but never belong in supporting_refs,
validation_record_refs or preserve_refs. Candidate code_refs are evidence locations,
not sources unless the exact source object appears in allowed_sources. supporting_refs
may cite A questions or effective assessments, never another S decision.

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


__all__ = [
    "ASSUMPTION_OUTPUT_SCHEMA",
    "V57_PROMPT_VERSION",
    "assumption_prompt",
    "disambiguation_prompt",
    "insert_view_notice",
    "summary_prompt_v65",
    "view_notice",
]
