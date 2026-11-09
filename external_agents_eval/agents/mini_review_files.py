"""Translate Bash-authored review records into the unchanged V6 ReviewBatch.

Only file delivery lives here. Target allocation, accepted judgments, reopening,
source checks, durable commits and S/Repair decisions remain in the V6 controller.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from external_agents_eval.workflows.solve_review_repair.ft_protocol import strict_json
from external_agents_eval.workflows.solve_review_repair.v5_contracts import (
    V5ContractError,
    validate_review_batch,
    validate_review_batch_shape,
    validate_review_note,
)
from external_agents_eval.workflows.solve_review_repair.v5_prompts import REVIEW_BATCH_SCHEMA

PROTOCOL = "mini-v6.3-review-files-1"
SCOPE_PATH = "logs/review-scope.md"
ASSESSMENT_FIELDS = {"target_id", "status", "expected", "observed", "reason", "code_refs"}

# Run in the task container, without importing the host project or reading outside
# the current attempt. Temporary files (*.md.tmp) are never candidate records.
READ_RECORDS = r"""import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
try:
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Current review output root is unavailable')
    notes = root / 'notes'
    if notes.is_symlink():
        raise ValueError('notes must be a real directory')
    paths = list(notes.glob('*.md')) if notes.is_dir() else []
    scope = root / 'logs/review-scope.md'
    if scope.exists() or scope.is_symlink():
        paths.append(scope)
    files = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if (path.is_symlink() or path.parent.is_symlink() or not path.is_file()
                or root.resolve() not in path.resolve().parents):
            raise ValueError('Not a regular current-attempt record: ' + relative)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('Review record exceeds 16 MiB: ' + relative)
        files[relative] = path.read_text(encoding='utf-8')
    print(json.dumps({'files': files}))
except (OSError, ValueError) as error:
    print(json.dumps({'error': str(error)}))
"""


def controller_state(prompt: str) -> dict[str, Any]:
    """Consume the existing, explicitly delimited controller input, not prose."""
    marker = "\nCONTROLLER STATE\n"
    if prompt.count(marker) != 1:
        raise ValueError("Expected exactly one V6 controller state in Reviewer input")
    state, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())
    # Access required bindings now; an incompatible prompt must fail explicitly.
    for key in ("batch_size", "pending_target_ids", "accepted", "revisable_target_ids"):
        state[key]
    state["review_notes"]["output_root"]
    state["review_notes"]["target_paths"]
    return state


def metadata(text: str, heading: str) -> dict[str, Any]:
    marker = heading + "\n"
    if text.splitlines().count(heading) != 1:
        raise V5ContractError(f"Record requires one {heading} section")
    section = text.split(marker, 1)[1].split("\n## ", 1)[0].strip()
    if not section.startswith("```json\n") or not section.endswith("\n```"):
        raise V5ContractError(f"{heading} must contain exactly one fenced JSON object")
    try:
        value = strict_json(section[len("```json\n"):-len("\n```")])
    except ValueError as error:
        raise V5ContractError(f"Invalid {heading} JSON: {error}") from error
    if not isinstance(value, dict):
        raise V5ContractError(f"{heading} must contain a JSON object")
    return value


def collect_batch(
    files: dict[str, str], *, role: str, state: dict[str, Any], archive: Path,
) -> tuple[dict[str, Any] | None, str]:
    """Validate a snapshot; absence means continue, never fabricate an empty batch."""
    allowed = {item["path"]: item["target_id"] for item in state["review_notes"]["target_paths"]}
    archive.mkdir(parents=True, exist_ok=True)
    assessments = []
    for relative, text in sorted(files.items()):
        if relative != SCOPE_PATH and relative not in allowed:
            raise V5ContractError(f"Unexpected record {relative}; use current controller target_paths")
        path = archive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if relative == SCOPE_PATH:
            continue
        target_id = allowed[relative]
        validate_review_note(archive, relative, target_id)
        assessment = metadata(text, "## Assessment")
        if set(assessment) != ASSESSMENT_FIELDS or assessment.get("target_id") != target_id:
            raise V5ContractError(f"{relative}: assessment fields or target_id do not match")
        assessments.append({**assessment, "review_note_path": relative})

    if SCOPE_PATH not in files:
        return None, f"Found {len(assessments)} record(s); write {SCOPE_PATH} last to close this batch"

    validate_review_note(archive, SCOPE_PATH, f"{role}_SCOPE")
    scope = metadata(files[SCOPE_PATH], "## Scope")
    if set(scope) != {"scope_complete", "new_targets", "revisit_target_ids"}:
        raise V5ContractError("Scope fields must be scope_complete, new_targets, revisit_target_ids")
    if not isinstance(scope["scope_complete"], bool):
        raise V5ContractError("scope_complete must be a boolean")
    pending = set(state["pending_target_ids"]) | set(state["revisable_target_ids"])
    remaining = pending - {item["target_id"] for item in assessments}
    done = bool(scope["scope_complete"] and not remaining
                and not scope["new_targets"] and not scope["revisit_target_ids"])
    payload = {
        "new_targets": scope["new_targets"], "assessments": assessments,
        "revisit_target_ids": scope["revisit_target_ids"], "done": done,
    }
    validate_review_batch_shape(payload)
    accepted = {item["target_id"] for item in state["accepted"]}
    enrolled = pending | accepted
    validate_review_batch(
        payload, role=role, enrolled_target_ids=enrolled,
        effective_target_ids=accepted, revisable_target_ids=set(state["revisable_target_ids"]),
        optional_target_ids=set(allowed.values()) - enrolled,
        output_root=archive, batch_size=state["batch_size"],
    )
    if not done and not (assessments or payload["new_targets"] or payload["revisit_target_ids"]):
        return None, "No review judgment or new work delivered; continue the current review"
    return payload, "Review files validated; the V6 controller will commit and allocate new IDs"


FILE_INSTRUCTIONS = """V6.3 FILE DELIVERY (replaces final-answer/batch-JSON delivery only)
Use native Bash to write records under $REVIEW_OUTPUT_ROOT, the current attempt
root injected by the adapter. Do not guess a round directory. Keep the existing
review-note skill's narrative sections and evidence rules. For each eligible
target write notes/<TARGET_ID>.md, starting with '# Review note: <TARGET_ID>'.
Before '## Summary', add '## Assessment' with one fenced ```json object containing
exactly target_id, status, expected, observed, reason, code_refs. These have the
same meanings as in the original Assessment examples. Do not include
review_note_path: the adapter derives it from the record filename. Keep all four
original narrative headings: Summary, Review steps, Limits and conflicting
observations, Reproduction materials. Every section must contain actual content.
Write a *.md.tmp file and mv it to *.md when finished; temporary files are ignored.

Write logs/review-scope.md LAST to close this batch. It starts with
'# Review note: <ROLE>_SCOPE', then '## Scope' with a fenced JSON object:
{"scope_complete": false, "new_targets": [], "revisit_target_ids": []}
Replace <ROLE> with I or R. Include the same four narrative headings with the
scope actually inspected and its limitations. Set scope_complete=true only after
your role's scope review is complete (including R's diff-impact review). New
targets retain the original TargetDraft schema and receive IDs from the V6
controller NEXT round; do not invent IDs or assess them before registration.
Reopening and the combined batch_size limit remain unchanged. The adapter derives
done only after all pending targets are covered, scope is complete, and this
batch adds no targets or reopen requests. unresolved/disputed are valid judgments,
not successful behavior. S retains all accept/hold/repair decisions.

After each Bash call the adapter validates current records and reports errors.
Fix invalid files in this SAME conversation. A valid closed batch is handed to
the controller automatically; do not call submit_json, print a completion marker,
or emit a final JSON report. Merely listing files or calling tools is not progress.
"""


def adapt_prompt(prompt: str) -> str:
    controller_state(prompt)
    if "\nOUTPUT SCHEMA\n" not in prompt:
        raise ValueError("Missing original ReviewBatch schema in Reviewer input")
    # Preserve task, role guidance, examples and authoritative state. Replace only
    # the output contract; the controller's original prompt is archived separately.
    body = prompt.rsplit("\nOUTPUT SCHEMA\n", 1)[0]
    definitions = {
        key: value for key, value in REVIEW_BATCH_SCHEMA["$defs"].items()
        if key not in {"Assessment", "ReviewBatch"}
    }
    return (body + "\n\n" + FILE_INSTRUCTIONS + "\nUNCHANGED V6 FIELD DEFINITIONS\n"
            + json.dumps({"$defs": definitions}, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# V6.4: one coherent file-delivery contract. The V5-6 controller prompt keeps its
# body protocol-neutral and ends with an OUTPUT CONTRACT section; this adapter
# replaces only that section. Record validation (collect_batch) is unchanged.
# ---------------------------------------------------------------------------

FILE_CONTRACT_V64 = """OUTPUT CONTRACT (files)
Write every record under $REVIEW_OUTPUT_ROOT, the current attempt root injected by
the adapter; do not guess round directories.
- Assessment: write the note for each assessed target at its review_notes.target_paths
  entry (notes/<TARGET_ID>.md). It begins '# Review note: <TARGET_ID>', then
  '## Assessment' with one fenced json object containing exactly target_id, status,
  expected, observed, reason and code_refs, followed by the sections Summary, Review
  steps, Limits and conflicting observations, and Reproduction materials, each with
  real content. Do not write review_note_path; the adapter derives it from the path.
- Close each batch by writing logs/review-scope.md last. It begins
  '# Review note: <ROLE>_SCOPE' (ROLE is I or R), then '## Scope' with one fenced json
  object {"scope_complete": false, "new_targets": [], "revisit_target_ids": []},
  followed by the same four sections describing the scope actually inspected and its
  limits. Set scope_complete=true only when your scope closes as described under
  REVIEW RECORDS AND BATCHES. new_targets use the TargetDraft definition below.
- Write each record as *.md.tmp and mv it to *.md when finished; temporary files are
  ignored.
- After every command the adapter validates the records and reports errors; fix them
  in this same conversation. A valid closed batch is handed to the Controller
  automatically. Do not call submit_json, print a completion marker, or write a final
  JSON report. Listing files or calling tools without writing records is not progress.
- unresolved and disputed are valid judgments, not successful behavior. S makes all
  accept/hold/repair decisions.

SYNTHETIC EXAMPLES - WRITING GUIDANCE ONLY
The examples are abstract. Their target IDs, paths, facts and wording are not
evidence, requirements or task-specific hints. Do not copy them.

Example 1 - incomplete coverage is not satisfied. Contract: a public operation must
return the documented value for a valid input, preserve the documented boundary
behavior, and reject an invalid input with the documented error. Observed: the valid
and boundary cases matched; the invalid case was not executed. Assessment block:
```json
{"target_id": "EXAMPLE-COVERAGE", "status": "unresolved", "expected": "The public operation satisfies the valid-input, boundary-input and invalid-input clauses of the contract.", "observed": "The valid-input and boundary-input calls matched the expected results; no invalid-input call was executed.", "reason": "The evidence covers only part of the target. The missing invalid-input check prevents satisfied, but does not establish a violation.", "code_refs": []}
```

Example 2 - raw shape and lifecycle evidence must be preserved:
```json
{"target_id": "EXAMPLE-RETURN-SHAPE", "status": "violated", "expected": "The documented public entry point returns a Result object directly.", "observed": "The raw public call returned Wrapper[Result]. The inner Result appeared correct only after an additional extraction step.", "reason": "The contract concerns the direct return shape of the public call. Extracting or adapting the inner value cannot prove that contract.", "code_refs": []}
```
```json
{"target_id": "EXAMPLE-LIFECYCLE", "status": "unresolved", "expected": "The transition leaves the old resource inactive and invokes its cleanup callback exactly once.", "observed": "The old resource was observed as inactive, but callback invocation and call count were not recorded.", "reason": "The final state alone does not establish the complete lifecycle contract. Missing side-effect evidence prevents satisfied.", "code_refs": []}
```

Before closing the batch, check every assessment: the status must follow from the
complete expected-versus-observed comparison; observed must describe the raw
documented call; and every required clause, side effect and transition must either
be evidenced or explicitly marked unresolved.
"""


def adapt_prompt_v64(prompt: str) -> str:
    """Replace the V5-6 JSON OUTPUT CONTRACT with the single file contract."""
    from external_agents_eval.workflows.solve_review_repair.v5_prompts import (
        OUTPUT_CONTRACT_MARKER,
    )

    controller_state(prompt)
    if prompt.count(OUTPUT_CONTRACT_MARKER) != 1:
        raise ValueError("Expected exactly one V5-6 OUTPUT CONTRACT in Reviewer input")
    body = prompt.split(OUTPUT_CONTRACT_MARKER, 1)[0]
    definitions = {
        key: value for key, value in REVIEW_BATCH_SCHEMA["$defs"].items()
        if key not in {"Assessment", "ReviewBatch"}
    }
    return (body + "\n" + FILE_CONTRACT_V64 + "\nFIELD DEFINITIONS\n"
            + json.dumps({"$defs": definitions}, ensure_ascii=False, indent=2) + "\n")
