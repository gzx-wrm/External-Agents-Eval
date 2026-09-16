from __future__ import annotations

from collections import Counter
from typing import Any


def obligation_index(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in ledger["obligations"]}


def validate_task_source_anchors(
    sources: list[dict[str, Any]], *, title: str, description: str
) -> list[str]:
    errors: list[str] = []
    fields = {"title": title, "description": description}
    for source in sources:
        if source.get("kind") != "task":
            continue
        field = source.get("field")
        quote = source.get("quote")
        if field not in fields or not isinstance(quote, str) or quote not in fields[field]:
            errors.append(f"{source.get('id', 'source')} task quote is not exact in {field}")
    return errors


def validate_ledger(payload: dict[str, Any], *, task_id: str) -> list[str]:
    errors: list[str] = []
    if payload["task_id"] != task_id:
        errors.append(f"task_id {payload['task_id']!r} does not match {task_id!r}")
    sources = payload["sources"]
    source_ids = [item["id"] for item in sources]
    _append_duplicate_error(errors, source_ids, "source IDs")
    known_sources = set(source_ids)
    source_kinds = {item["id"]: item["kind"] for item in sources}

    obligations = payload["obligations"]
    obligation_ids = [item["id"] for item in obligations]
    _append_duplicate_error(errors, obligation_ids, "obligation IDs")
    for obligation in obligations:
        _validate_definition(
            errors,
            obligation,
            known_sources=known_sources,
            source_kinds=source_kinds,
        )

    questions = payload["open_questions"]
    question_ids = [item["id"] for item in questions]
    _append_duplicate_error(errors, question_ids, "open-question IDs")
    for question in questions:
        _validate_source_refs(errors, question["id"], question["source_refs"], known_sources)
    return errors


def validate_assessment(
    payload: dict[str, Any],
    *,
    task_id: str,
    patch_sha256: str,
    ledger_sha256: str,
    ledger: dict[str, Any],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    protocol: list[str] = []
    if payload["task_id"] != task_id:
        protocol.append("assessment task_id does not match Controller input")
    if payload["patch_sha256"] != patch_sha256:
        protocol.append("assessment patch_sha256 does not match candidate implementation")
    if payload["ledger_sha256"] != ledger_sha256:
        protocol.append("assessment ledger_sha256 does not match frozen ledger")

    sources = [*ledger["sources"], *payload["additional_sources"]]
    source_ids = [item["id"] for item in sources]
    _append_duplicate_error(errors, source_ids, "combined source IDs")
    known_sources = set(source_ids)
    source_kinds = {item["id"]: item["kind"] for item in sources}

    original = obligation_index(ledger)
    late_items = payload["late_obligations"]
    late_ids = [item["id"] for item in late_items]
    _append_duplicate_error(errors, late_ids, "late obligation IDs")
    late = {item["id"]: item for item in late_items}
    for obligation in late_items:
        _validate_definition(
            errors,
            obligation,
            known_sources=known_sources,
            source_kinds=source_kinds,
        )

    definitions = {**original, **late}
    assessments = payload["obligation_assessments"]
    assessed_refs = [item["obligation_ref"] for item in assessments]
    _append_duplicate_error(errors, assessed_refs, "obligation assessments")
    missing = sorted(set(definitions) - set(assessed_refs))
    extra = sorted(set(assessed_refs) - set(definitions))
    if missing:
        errors.append("missing obligation assessments: " + ", ".join(missing))
    if extra:
        errors.append("unknown obligation assessments: " + ", ".join(extra))

    assessment_by_ref = {item["obligation_ref"]: item for item in assessments}
    for item in assessments:
        ref = item["obligation_ref"]
        source_status = item["source_status"]
        implementation_status = item["implementation_status"]
        source_note = item["source_note"]
        evidence = item["implementation_evidence"]
        validation = item["validation_evidence"]
        gap = item["gap"]
        if ref.startswith("L") and source_status != "supported":
            errors.append(f"{ref} late obligation must use source_status=supported")
        if source_status == "supported":
            if source_note is not None:
                errors.append(f"{ref} supported obligation must use source_note=null")
            if implementation_status == "not_applicable":
                errors.append(f"{ref} supported obligation cannot be not_applicable")
        else:
            if not _nonempty(source_note):
                errors.append(f"{ref} {source_status} obligation requires source_note")
            if implementation_status != "not_applicable":
                errors.append(f"{ref} {source_status} obligation must be not_applicable")
            if evidence or validation or gap is not None:
                errors.append(
                    f"{ref} not_applicable obligation requires empty evidence and gap=null"
                )
        if implementation_status == "covered":
            if not evidence:
                errors.append(f"{ref} is covered without implementation evidence")
            if gap is not None:
                errors.append(f"{ref} is covered but gap is not null")
            failed = [item["command"] for item in validation if item["result"] != "passed"]
            if failed:
                errors.append(
                    f"{ref} covered evidence contains non-passing validation: "
                    + ", ".join(failed)
                )
        elif implementation_status in {"partial", "missing", "uncertain"}:
            if not _nonempty(gap):
                errors.append(f"{ref} implementation status {implementation_status} needs gap")
        _validate_evidence_lines(errors, ref, evidence)

    regressions = payload["unexpected_regressions"]
    regression_ids = [item["id"] for item in regressions]
    _append_duplicate_error(errors, regression_ids, "unexpected regression IDs")
    regression_by_id = {item["id"]: item for item in regressions}
    for regression in regressions:
        ref = regression["id"]
        _validate_source_refs(errors, ref, regression["source_refs"], known_sources)
        _validate_evidence_lines(errors, ref, regression["implementation_evidence"])
        if not regression["implementation_evidence"] and not regression["validation_evidence"]:
            errors.append(f"{ref} regression requires implementation or validation evidence")

    effective = {
        ref: definition
        for ref, definition in definitions.items()
        if assessment_by_ref.get(ref, {}).get("source_status") == "supported"
    }
    requests = payload["repair_requests"]
    request_ids = [item["id"] for item in requests]
    _append_duplicate_error(errors, request_ids, "repair request IDs")
    known_findings = {*definitions, *regression_by_id}
    requested_findings: set[str] = set()
    for request in requests:
        refs = set(request["finding_refs"])
        requested_findings.update(refs)
        unknown = sorted(refs - known_findings)
        if unknown:
            errors.append(
                f"repair request {request['id']} references unknown findings: "
                + ", ".join(unknown)
            )
        statuses: list[str] = []
        for ref in refs:
            if ref.startswith("R"):
                regression = regression_by_id.get(ref)
                if regression is not None:
                    if not regression["blocking"]:
                        errors.append(
                            f"repair request {request['id']} uses nonblocking regression {ref}"
                        )
                    statuses.append("regression")
                continue
            definition = effective.get(ref)
            item = assessment_by_ref.get(ref)
            if definition is None or item is None:
                errors.append(
                    f"repair request {request['id']} uses non-effective obligation {ref}"
                )
                continue
            if definition["criticality"] != "must":
                errors.append(f"repair request {request['id']} uses should obligation {ref}")
            statuses.append(item["implementation_status"])
        action = request["action"]
        allowed = {"partial", "missing", "regression"} if action == "fix" else {"uncertain"}
        invalid = sorted({status for status in statuses if status not in allowed})
        if invalid:
            errors.append(
                f"repair request {request['id']} action {action} conflicts with findings: "
                + ", ".join(invalid)
            )

    must_gaps = {
        ref
        for ref, definition in effective.items()
        if definition["criticality"] == "must"
        and assessment_by_ref[ref]["implementation_status"] in {"partial", "missing"}
    }
    uncertain_must = {
        ref
        for ref, definition in effective.items()
        if definition["criticality"] == "must"
        and assessment_by_ref[ref]["implementation_status"] == "uncertain"
    }
    unclear_must = {
        ref
        for ref, definition in original.items()
        if definition["criticality"] == "must"
        and assessment_by_ref.get(ref, {}).get("source_status") == "unclear"
    }
    blocking_regressions = {
        item["id"] for item in regressions if item["blocking"]
    }
    actionable = must_gaps | uncertain_must | blocking_regressions
    uncovered = sorted(actionable - requested_findings)
    verdict = payload["verdict"]
    if verdict == "approve":
        not_covered = sorted(must_gaps | uncertain_must)
        if not_covered:
            errors.append(
                "approve conflicts with uncovered supported must obligations: "
                + ", ".join(not_covered)
            )
        if unclear_must:
            errors.append(
                "approve conflicts with unclear must obligations: "
                + ", ".join(sorted(unclear_must))
            )
        if blocking_regressions:
            errors.append("approve conflicts with a blocking unexpected regression")
        if requests:
            errors.append("approve verdict cannot include repair requests")
    elif verdict in {"request_changes", "replan"}:
        if not requests:
            errors.append(f"{verdict} verdict requires at least one repair request")
        if uncovered:
            errors.append(
                f"{verdict} omits actionable findings from repair requests: "
                + ", ".join(uncovered)
            )
    else:
        if requests:
            errors.append("abstain verdict cannot include repair requests")
        if not unclear_must and not uncertain_must:
            errors.append("abstain requires an unclear or implementation-uncertain must obligation")
    return errors, protocol


def build_effective_review_set(
    *,
    task_id: str,
    ledger_sha256: str,
    assessment_sha256: str,
    ledger: dict[str, Any],
    assessment: dict[str, Any],
) -> dict[str, Any]:
    original = obligation_index(ledger)
    late = {item["id"]: item for item in assessment["late_obligations"]}
    statuses = {
        item["obligation_ref"]: item["source_status"]
        for item in assessment["obligation_assessments"]
    }
    definitions = {**original, **late}
    return {
        "schema_version": "3.0",
        "task_id": task_id,
        "ledger_sha256": ledger_sha256,
        "assessment_sha256": assessment_sha256,
        "effective_obligations": sorted(
            ref for ref in definitions if statuses.get(ref) == "supported"
        ),
        "nonbinding_obligations": sorted(
            ref for ref in original if statuses.get(ref) == "unsupported"
        ),
        "unresolved_obligations": sorted(
            ref for ref in original if statuses.get(ref) == "unclear"
        ),
        "late_obligation_count": len(late),
        "repair_request_ids": [item["id"] for item in assessment["repair_requests"]],
        "verdict": assessment["verdict"],
    }


def build_request_contexts(
    *, ledger: dict[str, Any], assessment: dict[str, Any]
) -> list[dict[str, Any]]:
    definitions = {
        **obligation_index(ledger),
        **{item["id"]: item for item in assessment["late_obligations"]},
    }
    assessments = {
        item["obligation_ref"]: item for item in assessment["obligation_assessments"]
    }
    regressions = {item["id"]: item for item in assessment["unexpected_regressions"]}
    result: list[dict[str, Any]] = []
    for order, request in enumerate(assessment["repair_requests"], start=1):
        refs = request["finding_refs"]
        obligation_items = [definitions[ref] for ref in refs if ref in definitions]
        regression_items = [regressions[ref] for ref in refs if ref in regressions]
        assessment_items = [assessments[ref] for ref in refs if ref in assessments]
        source_ids: set[str] = set()
        for item in obligation_items:
            source_ids.update(item["source_refs"])
        for item in regression_items:
            source_ids.update(item["source_refs"])
        result.append(
            {
                "order": order,
                "request": request,
                "obligations": obligation_items,
                "regressions": regression_items,
                "assessments": assessment_items,
                "source_ids": sorted(source_ids),
            }
        )
    return result


def validate_repair_audit(
    payload: dict[str, Any],
    *,
    task_id: str,
    candidate_patch_sha256: str,
    assessment_sha256: str,
    effective_review_set_sha256: str,
    assessment: dict[str, Any],
    effective_review_set: dict[str, Any],
    actual_changed_paths: set[str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    protocol: list[str] = []
    bindings = {
        "task_id": task_id,
        "candidate_patch_sha256": candidate_patch_sha256,
        "assessment_sha256": assessment_sha256,
        "effective_review_set_sha256": effective_review_set_sha256,
    }
    for field, expected in bindings.items():
        if payload[field] != expected:
            protocol.append(f"repair audit {field} does not match frozen input")
    if set(payload["changed_paths"]) != actual_changed_paths:
        errors.append("repair audit changed_paths do not match candidate-to-repaired diff")

    requests = {item["id"]: item for item in assessment["repair_requests"]}
    known_findings = {
        *effective_review_set["effective_obligations"],
        *(item["id"] for item in assessment["unexpected_regressions"]),
    }
    changes = payload["changes"]
    _append_duplicate_error(errors, [item["change_id"] for item in changes], "change IDs")
    covered_requests: set[str] = set()
    covered_findings: dict[str, set[str]] = {}
    implemented_paths: set[str] = set()
    for change in changes:
        request_refs = set(change["request_refs"])
        finding_refs = set(change["finding_refs"])
        unknown_requests = sorted(request_refs - set(requests))
        if unknown_requests:
            errors.append(
                f"{change['change_id']} references unknown requests: "
                + ", ".join(unknown_requests)
            )
        unknown_findings = sorted(finding_refs - known_findings)
        if unknown_findings:
            errors.append(
                f"{change['change_id']} references unknown findings: "
                + ", ".join(unknown_findings)
            )
        covered_requests.update(request_refs)
        for request_ref in request_refs:
            covered_findings.setdefault(request_ref, set()).update(finding_refs)
            request = requests.get(request_ref)
            if request is not None:
                extra = finding_refs - set(request["finding_refs"])
                if extra and not change["out_of_request_scope"]:
                    errors.append(
                        f"{change['change_id']} has findings outside {request_ref} without scope flag"
                    )
        if not request_refs and not change["out_of_request_scope"]:
            errors.append(f"{change['change_id']} without request_refs must be out_of_request_scope")
        if change["out_of_request_scope"] != _nonempty(change["scope_rationale"]):
            errors.append(
                f"{change['change_id']} scope_rationale must be non-empty exactly for out-of-scope work"
            )
        decision = change["decision"]
        changed = set(change["changed_paths"])
        if decision == "implement":
            if not changed:
                errors.append(f"{change['change_id']} implemented change has no changed_paths")
            if not change["validation"]:
                errors.append(f"{change['change_id']} implemented change has no validation")
            implemented_paths.update(changed)
        else:
            if changed:
                errors.append(f"{change['change_id']} non-implement decision has changed_paths")
            if decision == "no_change":
                if not change["validation"]:
                    errors.append(f"{change['change_id']} no_change decision needs validation")
                invalid = [
                    ref
                    for ref in request_refs
                    if requests.get(ref, {}).get("action") != "investigate"
                ]
                if invalid:
                    errors.append(
                        f"{change['change_id']} no_change is valid only for investigate requests"
                    )
        if not changed.issubset(actual_changed_paths):
            errors.append(f"{change['change_id']} cites paths outside the actual diff")
    if implemented_paths != actual_changed_paths:
        errors.append("repair audit implemented changes do not cover the exact actual diff")

    unaddressed = [item["request_ref"] for item in payload["unaddressed_requests"]]
    _append_duplicate_error(errors, unaddressed, "unaddressed request refs")
    unknown_unaddressed = sorted(set(unaddressed) - set(requests))
    if unknown_unaddressed:
        errors.append("unknown unaddressed requests: " + ", ".join(unknown_unaddressed))
    overlap = sorted(covered_requests & set(unaddressed))
    if overlap:
        errors.append("requests cannot be addressed and unaddressed: " + ", ".join(overlap))
    omitted = sorted(set(requests) - covered_requests - set(unaddressed))
    if omitted:
        errors.append("repair audit omits requests: " + ", ".join(omitted))
    for request_ref in covered_requests:
        request = requests.get(request_ref)
        if request is None:
            continue
        missing = sorted(set(request["finding_refs"]) - covered_findings.get(request_ref, set()))
        if missing:
            errors.append(
                f"repair audit {request_ref} omits findings: " + ", ".join(missing)
            )
    return errors, protocol


def source_status_counts(assessment: dict[str, Any]) -> Counter[str]:
    return Counter(item["source_status"] for item in assessment["obligation_assessments"])


def _validate_definition(
    errors: list[str],
    item: dict[str, Any],
    *,
    known_sources: set[str],
    source_kinds: dict[str, str],
) -> None:
    ref = item["id"]
    _validate_source_refs(errors, ref, item["source_refs"], known_sources)
    kinds = {source_kinds.get(source) for source in item["source_refs"]}
    if item["basis"] == "explicit_task" and "task" not in kinds:
        errors.append(f"{ref} explicit_task basis requires a task source")
    if item["basis"] in {"repository_contract", "compatibility_inferred"} and (
        "repository" not in kinds
    ):
        errors.append(f"{ref} repository basis requires a repository source")


def _validate_evidence_lines(
    errors: list[str], ref: str, evidence: list[dict[str, Any]]
) -> None:
    for item in evidence:
        if item["line_end"] < item["line_start"]:
            errors.append(f"{ref} evidence line_end precedes line_start")


def _validate_source_refs(
    errors: list[str], owner: str, refs: list[str], known_sources: set[str]
) -> None:
    unknown = sorted(set(refs) - known_sources)
    if unknown:
        errors.append(f"{owner} references unknown sources: " + ", ".join(unknown))


def _append_duplicate_error(errors: list[str], values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        errors.append(f"{label} must be unique")


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
