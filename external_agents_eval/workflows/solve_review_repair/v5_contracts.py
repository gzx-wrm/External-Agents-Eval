"""Deterministic V5 review-batch and Markdown-note validation.

This module deliberately keeps semantic rules outside the model.  JSON shape is
validated first, then the controller checks target membership, batch size and
the required immutable note for every submitted assessment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .validators import extract_single_json_object

STATUSES = frozenset({"satisfied", "violated", "unresolved", "disputed"})
MODES = frozenset({"continue", "retry", "revise"})
NOTE_HEADINGS = (
    "## Summary",
    "## Review steps",
    "## Limits and conflicting observations",
    "## Reproduction materials",
)
NOTE_TEMPLATE_PLACEHOLDERS = frozenset(
    {
        "TARGET_ID",
        "One or two sentences describing the reviewed scope and conclusion.",
        "Purpose of this step",
        (
            "Describe what was actually inspected or executed, what was observed, "
            "and why it matters. Cite relevant snapshot/path/line ranges."
        ),
        "For an executed command only: actual cwd and base/candidate snapshot.",
        "Exact command actually executed; omit this block for a static-only step.",
        "Decisive observed output and known exit code, or state it was not recorded.",
        "Add only the additional steps needed to support this assessment.",
        (
            "State unverified scope, blocking conditions, contrary results, and any "
            "change in probe inputs or expectations. Do not convert missing coverage "
            "into success."
        ),
        (
            "List existing probe/input/log paths, purpose, and replay conditions. "
            "State missing materials or uncertain versions; for static-only review "
            "say so."
        ),
    }
)


class V5ContractError(ValueError):
    """A deterministic V5 protocol violation."""


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V5ContractError(f"{name} must be a non-empty string")
    return value


def _validate_string_list(
    value: Any, name: str, *, min_items: int = 0
) -> list[str]:
    if not isinstance(value, list) or len(value) < min_items:
        raise V5ContractError(f"{name} must be an array")
    for index, item in enumerate(value):
        _required_string(item, f"{name}[{index}]")
    if len(value) != len(set(value)):
        raise V5ContractError(f"{name} contains duplicates")
    return value


def _validate_source(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise V5ContractError(f"{name} must be an object")
    kind = value.get("kind")
    if kind == "task":
        if set(value) != {"kind", "field", "quote"}:
            raise V5ContractError(f"{name} has invalid task-source fields")
        if value["field"] not in {"title", "description"}:
            raise V5ContractError(f"{name}.field is invalid")
        _required_string(value["quote"], f"{name}.quote")
    elif kind == "repository":
        if set(value) != {"kind", "path", "line_start", "line_end"}:
            raise V5ContractError(f"{name} has invalid repository-source fields")
        _required_string(value["path"], f"{name}.path")
        start = value["line_start"]
        end = value["line_end"]
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 1
            or end < start
        ):
            raise V5ContractError(f"{name} has an invalid line range")
    else:
        raise V5ContractError(f"{name}.kind is invalid")


def _validate_sources(value: Any, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise V5ContractError(f"{name} must be a non-empty array")
    fingerprints: set[str] = set()
    for index, source in enumerate(value):
        _validate_source(source, f"{name}[{index}]")
        fingerprint = json.dumps(source, ensure_ascii=False, sort_keys=True)
        if fingerprint in fingerprints:
            raise V5ContractError(f"{name} contains duplicate sources")
        fingerprints.add(fingerprint)


def _validate_target_draft(target: Any, name: str) -> None:
    if not isinstance(target, dict) or set(target) != {"statement", "importance", "sources"}:
        raise V5ContractError(f"{name} is invalid")
    _required_string(target["statement"], f"{name}.statement")
    if target["importance"] not in {"required", "advisory"}:
        raise V5ContractError(f"{name}.importance is invalid")
    _validate_sources(target["sources"], f"{name}.sources")


def _validate_question_draft(question: Any, name: str) -> None:
    if not isinstance(question, dict) or set(question) != {"question", "blocking", "sources"}:
        raise V5ContractError(f"{name} is invalid")
    _required_string(question["question"], f"{name}.question")
    if not isinstance(question["blocking"], bool):
        raise V5ContractError(f"{name}.blocking must be boolean")
    _validate_sources(question["sources"], f"{name}.sources")


def validate_managed_delivery(
    value: Any, *, task: dict[str, str]
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise V5ContractError("managed_delivery must be an array")
    fingerprints: set[str] = set()
    for index, source in enumerate(value):
        name = f"managed_delivery[{index}]"
        _validate_source(source, name)
        if source["kind"] != "task":
            raise V5ContractError(f"{name} must be a task source")
        field_text = task[source["field"]]
        if source["quote"] not in field_text:
            raise V5ContractError(
                f"{name}.quote is not contiguous text from task {source['field']}"
            )
        fingerprint = json.dumps(source, ensure_ascii=False, sort_keys=True)
        if fingerprint in fingerprints:
            raise V5ContractError("managed_delivery contains duplicate sources")
        fingerprints.add(fingerprint)
    return value


def validate_review_batch_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise V5ContractError("ReviewBatch must be an object")
    required = {"new_targets", "assessments", "revisit_target_ids", "done"}
    missing = sorted(required - payload.keys())
    extra = sorted(set(payload) - required)
    if missing:
        raise V5ContractError("missing ReviewBatch fields: " + ", ".join(missing))
    if extra:
        raise V5ContractError("unknown ReviewBatch fields: " + ", ".join(extra))
    if not isinstance(payload["new_targets"], list):
        raise V5ContractError("new_targets must be an array")
    if not isinstance(payload["assessments"], list):
        raise V5ContractError("assessments must be an array")
    _validate_string_list(payload["revisit_target_ids"], "revisit_target_ids")
    if not isinstance(payload["done"], bool):
        raise V5ContractError("done must be boolean")

    new_ids: list[str] = []
    for index, target in enumerate(payload["new_targets"]):
        _validate_target_draft(target, f"new_targets[{index}]")
        new_ids.append(target["statement"])
    if len({" ".join(item.casefold().split()) for item in new_ids}) != len(new_ids):
        raise V5ContractError("a batch cannot declare duplicate new targets")

    assessment_ids: list[str] = []
    for index, assessment in enumerate(payload["assessments"]):
        if not isinstance(assessment, dict):
            raise V5ContractError(f"assessments[{index}] must be an object")
        required_assessment = {
            "target_id", "status", "expected", "observed", "reason", "code_refs", "review_note_path"
        }
        if set(assessment) != required_assessment:
            raise V5ContractError(f"assessments[{index}] has invalid fields")
        target_id = _required_string(assessment["target_id"], f"assessments[{index}].target_id")
        assessment_ids.append(target_id)
        if assessment["status"] not in STATUSES:
            raise V5ContractError(f"assessments[{index}].status is invalid")
        for field in ("expected", "observed", "reason", "review_note_path"):
            _required_string(assessment[field], f"assessments[{index}].{field}")
        if not isinstance(assessment["code_refs"], list):
            raise V5ContractError(f"assessments[{index}].code_refs must be an array")
        seen_refs: set[str] = set()
        for ref_index, reference in enumerate(assessment["code_refs"]):
            name = f"assessments[{index}].code_refs[{ref_index}]"
            if not isinstance(reference, dict) or set(reference) != {
                "snapshot",
                "path",
                "line_start",
                "line_end",
            }:
                raise V5ContractError(f"{name} is invalid")
            if reference["snapshot"] not in {"base", "candidate"}:
                raise V5ContractError(f"{name}.snapshot is invalid")
            _required_string(reference["path"], f"{name}.path")
            start, end = reference["line_start"], reference["line_end"]
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 1
                or end < start
            ):
                raise V5ContractError(f"{name} has an invalid line range")
            fingerprint = json.dumps(reference, sort_keys=True)
            if fingerprint in seen_refs:
                raise V5ContractError(f"assessments[{index}].code_refs contains duplicates")
            seen_refs.add(fingerprint)

    if len(set(assessment_ids)) != len(assessment_ids):
        raise V5ContractError("a batch cannot assess the same target twice")
    revisit = payload["revisit_target_ids"]
    if set(assessment_ids) & set(revisit):
        raise V5ContractError("reopen and replacement assessment must use separate batches")
    return payload


def _safe_relative_path(root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise V5ContractError("review note path must be relative")
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise V5ContractError("review note path escapes output_root") from error
    current = root_resolved
    for part in Path(relative_path).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise V5ContractError("review note path traverses a symlink")
    return candidate


def validate_review_note(output_root: str | Path, relative_path: str, target_id: str) -> bytes:
    root = Path(output_root)
    unresolved = root / relative_path
    if unresolved.is_symlink():
        raise V5ContractError(
            f"review note is not a regular file: {relative_path}"
        )
    path = _safe_relative_path(root, relative_path)
    if not path.is_file() or path.is_symlink():
        raise V5ContractError(f"review note is not a regular file: {relative_path}")
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise V5ContractError(f"review note cannot be read: {relative_path}") from error
    if not text.strip():
        raise V5ContractError(f"review note is empty: {relative_path}")
    title = f"# Review note: {target_id}"
    if not text.startswith(title + "\n") and text.strip() != title:
        raise V5ContractError(f"review note title does not match {target_id}")
    lines = text.splitlines()
    heading_lines: list[int] = []
    for heading in NOTE_HEADINGS:
        matches = [index for index, line in enumerate(lines) if line.strip() == heading]
        if len(matches) != 1:
            count = "missing" if not matches else "duplicated"
            raise V5ContractError(
                f"review note {relative_path} has {count} heading: {heading}"
            )
        heading_lines.append(matches[0])
    if heading_lines != sorted(heading_lines):
        raise V5ContractError(
            f"review note {relative_path} headings are out of order"
        )
    for index, heading_line in enumerate(heading_lines):
        end = heading_lines[index + 1] if index + 1 < len(heading_lines) else len(lines)
        if not "\n".join(lines[heading_line + 1 : end]).strip():
            raise V5ContractError(
                f"review note {relative_path} has an empty section: {NOTE_HEADINGS[index]}"
            )
    cursor = 0
    while True:
        start = text.find("{{", cursor)
        if start < 0:
            break
        end = text.find("}}", start + 2)
        if end < 0:
            break
        placeholder = " ".join(text[start + 2 : end].split())
        if placeholder in NOTE_TEMPLATE_PLACEHOLDERS:
            raise V5ContractError(
                f"review note contains a template placeholder: {relative_path}"
            )
        cursor = end + 2
    return content


def validate_output_schema(payload: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """Validate model output shape before deterministic protocol semantics."""

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise V5ContractError(f"invalid V5 output schema: {error.message}") from error
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda item: [str(part) for part in item.absolute_path],
    )
    if errors:
        messages: list[str] = []
        for error in errors[:8]:
            path = "$" + "".join(
                f"[{part}]" if isinstance(part, int) else f".{part}"
                for part in error.absolute_path
            )
            messages.append(f"{path}: {error.message}")
        if len(errors) > len(messages):
            messages.append(f"... and {len(errors) - len(messages)} more errors")
        raise V5ContractError("JSON Schema validation failed: " + "; ".join(messages))
    assert isinstance(payload, dict)
    return payload


def validate_review_batch(
    payload: Any,
    *,
    role: str,
    enrolled_target_ids: set[str],
    effective_target_ids: set[str],
    revisable_target_ids: set[str],
    optional_target_ids: set[str] = frozenset(),
    output_root: str | Path,
    batch_size: int,
    done: bool | None = None,
) -> dict[str, Any]:
    payload = validate_review_batch_shape(payload)
    if role not in {"I", "R", "implementation", "regression"}:
        raise V5ContractError(f"unsupported reviewer role: {role}")
    total = len(payload["new_targets"]) + len(payload["assessments"]) + len(payload["revisit_target_ids"])
    if total > batch_size:
        raise V5ContractError(f"batch contains {total} entries, limit is {batch_size}")
    known = set(enrolled_target_ids) | set(revisable_target_ids) | set(optional_target_ids)
    for assessment in payload["assessments"]:
        target_id = assessment["target_id"]
        if target_id not in known:
            raise V5ContractError(f"assessment references unknown target {target_id}")
        if target_id in effective_target_ids and target_id not in revisable_target_ids:
            raise V5ContractError(f"assessment replaces non-reopened target {target_id}")
        expected_note_path = f"notes/{target_id}.md"
        if assessment["review_note_path"] != expected_note_path:
            raise V5ContractError(
                f"assessment {target_id} must use assigned note path {expected_note_path}"
            )
        validate_review_note(output_root, assessment["review_note_path"], target_id)
    for target_id in payload["revisit_target_ids"]:
        if target_id not in effective_target_ids:
            raise V5ContractError(f"cannot reopen target without an effective judgment: {target_id}")
        if target_id in revisable_target_ids:
            raise V5ContractError(f"target is already revisable: {target_id}")
    if done is not None and payload["done"] != done:
        raise V5ContractError("done value does not match controller expectation")
    return payload


def parse_json_object(text: str) -> dict[str, Any]:
    text, _normalization = extract_single_json_object(text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise V5ContractError(f"invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise V5ContractError("JSON output must be an object")
    return value


def parse_review_batch(text: str) -> dict[str, Any]:
    return validate_review_batch_shape(parse_json_object(text))


def validate_spec_output(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise V5ContractError("SpecOutput must be an object")
    required = {"summary", "obligations", "preservation_obligations", "open_questions"}
    if set(payload) != required:
        raise V5ContractError("SpecOutput fields do not match the V5 contract")
    _required_string(payload["summary"], "summary")
    for field in ("obligations", "preservation_obligations"):
        if not isinstance(payload[field], list):
            raise V5ContractError(f"{field} must be an array")
        for index, target in enumerate(payload[field]):
            _validate_target_draft(target, f"{field}[{index}]")
    if not isinstance(payload["open_questions"], list):
        raise V5ContractError("open_questions must be an array")
    for index, question in enumerate(payload["open_questions"]):
        _validate_question_draft(question, f"open_questions[{index}]")
    statements = [
        " ".join(item["statement"].casefold().split())
        for item in [*payload["obligations"], *payload["preservation_obligations"]]
    ]
    if len(statements) != len(set(statements)):
        raise V5ContractError("obligations and preservation obligations must be unique")
    return payload


def validate_summary_output(
    payload: Any,
    required_refs: set[str],
    *,
    decision_context: dict[str, dict[str, Any]] | None = None,
    allowed_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise V5ContractError("SummaryOutput must be an object")
    required = {"summary", "decisions", "repair_requests", "new_questions"}
    if set(payload) != required:
        raise V5ContractError("SummaryOutput fields do not match the V5 contract")
    _required_string(payload["summary"], "summary")
    decisions = payload["decisions"]
    if not isinstance(decisions, list):
        raise V5ContractError("decisions must be an array")
    decision_refs: list[str] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise V5ContractError(f"decisions[{index}] must be an object")
        fields = {"finding_ref", "disposition", "reason", "supporting_refs", "sources"}
        if set(decision) != fields:
            raise V5ContractError(f"decisions[{index}] has invalid fields")
        decision_refs.append(_required_string(decision["finding_ref"], f"decisions[{index}].finding_ref"))
        if decision["disposition"] not in {"repair", "dismiss", "unresolved"}:
            raise V5ContractError(f"decisions[{index}].disposition is invalid")
        _required_string(decision["reason"], f"decisions[{index}].reason")
        _validate_string_list(
            decision["supporting_refs"], f"decisions[{index}].supporting_refs"
        )
        _validate_sources(decision["sources"], f"decisions[{index}].sources")
    if len(decision_refs) != len(set(decision_refs)):
        raise V5ContractError("S must decide each finding at most once")
    decision_ref_set = set(decision_refs)
    missing_required = set(required_refs) - decision_ref_set
    extra_refs = decision_ref_set - set(required_refs)
    if missing_required:
        raise V5ContractError(
            "S decision coverage mismatch: missing="
            + ",".join(sorted(missing_required))
        )
    if extra_refs and decision_context is None:
        raise V5ContractError(
            "S decision coverage mismatch: unknown=" + ",".join(sorted(extra_refs))
        )
    if decision_context is not None:
        invalid_extra = [
            decision["finding_ref"]
            for decision in decisions
            if decision["finding_ref"] in extra_refs
            and not (
                decision_context.get(decision["finding_ref"], {}).get("kind")
                == "assessment"
                and decision_context.get(decision["finding_ref"], {}).get("status")
                == "satisfied"
                and decision["disposition"] == "unresolved"
            )
        ]
        if invalid_extra:
            raise V5ContractError(
                "S decision coverage mismatch: invalid extra="
                + ",".join(sorted(invalid_extra))
            )
    if decision_context is not None:
        known_refs = set(decision_context)
        for index, decision in enumerate(decisions):
            unknown_support = set(decision["supporting_refs"]) - known_refs
            if unknown_support:
                raise V5ContractError(
                    f"decisions[{index}] has unknown supporting refs: "
                    + ",".join(sorted(unknown_support))
                )
            context = decision_context.get(decision["finding_ref"], {})
            if decision["disposition"] == "repair" and not (
                context.get("kind") == "assessment"
                and context.get("status") == "violated"
                and context.get("importance") == "required"
            ):
                raise V5ContractError(
                    f"decisions[{index}] may repair only a required violated assessment"
                )
    if allowed_sources is not None:
        allowed = {
            json.dumps(source, ensure_ascii=False, sort_keys=True)
            for source in allowed_sources
        }
        for index, decision in enumerate(decisions):
            for source in decision["sources"]:
                if json.dumps(source, ensure_ascii=False, sort_keys=True) not in allowed:
                    raise V5ContractError(
                        f"decisions[{index}] cites a source outside the frozen catalog"
                    )
    requests = payload["repair_requests"]
    if not isinstance(requests, list):
        raise V5ContractError("repair_requests must be an array")
    repairs = {item["finding_ref"] for item in decisions if item["disposition"] == "repair"}
    challenged = {
        item["finding_ref"]
        for item in decisions
        if item["disposition"] == "unresolved"
    }
    requested: list[str] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict) or set(request) != {"finding_refs", "change", "validation_record_refs", "preserve_refs"}:
            raise V5ContractError(f"repair_requests[{index}] is invalid")
        refs = request["finding_refs"]
        _validate_string_list(
            refs, f"repair_requests[{index}].finding_refs", min_items=1
        )
        requested.extend(refs)
        _required_string(request["change"], f"repair_requests[{index}].change")
        for field in ("validation_record_refs", "preserve_refs"):
            _validate_string_list(
                request[field], f"repair_requests[{index}].{field}"
            )
            if decision_context is not None:
                unknown = set(request[field]) - set(decision_context)
                if unknown:
                    raise V5ContractError(
                        f"repair_requests[{index}].{field} has unknown refs: "
                        + ",".join(sorted(unknown))
                    )
        if decision_context is not None:
            invalid_validation = [
                ref
                for ref in request["validation_record_refs"]
                if decision_context[ref].get("kind") != "assessment"
            ]
            if invalid_validation:
                raise V5ContractError(
                    f"repair_requests[{index}].validation_record_refs must "
                    "reference effective assessments"
                )
            invalid_preserve = [
                ref
                for ref in request["preserve_refs"]
                if decision_context[ref].get("kind") != "assessment"
                or decision_context[ref].get("status") != "satisfied"
                or ref in challenged
            ]
            if invalid_preserve:
                raise V5ContractError(
                    f"repair_requests[{index}].preserve_refs must reference "
                    "unchallenged satisfied assessments"
                )
    if len(requested) != len(set(requested)):
        raise V5ContractError("a repair decision may appear in only one repair request")
    if set(requested) != repairs:
        raise V5ContractError("repair requests must cover exactly all repair decisions")
    if not isinstance(payload["new_questions"], list):
        raise V5ContractError("new_questions must be an array")
    for index, question in enumerate(payload["new_questions"]):
        _validate_question_draft(question, f"new_questions[{index}]")
        if allowed_sources is not None:
            for source in question["sources"]:
                if json.dumps(source, ensure_ascii=False, sort_keys=True) not in allowed:
                    raise V5ContractError(
                        f"new_questions[{index}] cites a source outside the frozen catalog"
                    )
    return payload


__all__ = [
    "MODES",
    "NOTE_HEADINGS",
    "STATUSES",
    "V5ContractError",
    "parse_json_object",
    "parse_review_batch",
    "validate_managed_delivery",
    "validate_output_schema",
    "validate_review_batch",
    "validate_review_batch_shape",
    "validate_review_note",
    "validate_spec_output",
    "validate_summary_output",
]
